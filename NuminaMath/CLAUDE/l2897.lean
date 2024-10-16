import Mathlib

namespace NUMINAMATH_CALUDE_sphere_surface_area_from_circumscribing_cube_l2897_289742

theorem sphere_surface_area_from_circumscribing_cube (cube_volume : ℝ) (sphere_surface_area : ℝ) : 
  cube_volume = 8 → sphere_surface_area = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_circumscribing_cube_l2897_289742


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2897_289744

theorem partial_fraction_decomposition :
  ∃! (A B C : ℚ),
    ∀ (x : ℝ), x ≠ 1 → x ≠ 4 → x ≠ -2 →
      (x^3 - x - 4) / ((x - 1) * (x - 4) * (x + 2)) =
      A / (x - 1) + B / (x - 4) + C / (x + 2) ∧
      A = 4/9 ∧ B = 28/9 ∧ C = -1/3 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2897_289744


namespace NUMINAMATH_CALUDE_library_visitors_average_l2897_289770

/-- Calculates the average number of visitors per day in a 30-day month starting with a Sunday -/
def averageVisitors (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalSundays := 5
  let totalOtherDays := 30 - totalSundays
  let totalVisitors := sundayVisitors * totalSundays + otherDayVisitors * totalOtherDays
  totalVisitors / 30

theorem library_visitors_average :
  averageVisitors 1000 700 = 750 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l2897_289770


namespace NUMINAMATH_CALUDE_inequality_of_squares_l2897_289797

theorem inequality_of_squares (x : ℝ) : (x - 1)^2 ≠ x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_squares_l2897_289797


namespace NUMINAMATH_CALUDE_mika_stickers_l2897_289791

/-- The number of stickers Mika has left after various additions and subtractions -/
def stickers_left (initial : ℕ) (bought : ℕ) (birthday : ℕ) (given : ℕ) (used : ℕ) : ℕ :=
  initial + bought + birthday - given - used

/-- Theorem stating that Mika is left with 2 stickers -/
theorem mika_stickers :
  stickers_left 20 26 20 6 58 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_l2897_289791


namespace NUMINAMATH_CALUDE_sequence_conditions_l2897_289794

theorem sequence_conditions (a : ℝ) : 
  let a₁ : ℝ := 1
  let a₂ : ℝ := 1
  let a₃ : ℝ := 1
  let a₄ : ℝ := a
  let a₅ : ℝ := a
  (a₁ = a₂ * a₃) ∧ 
  (a₂ = a₁ * a₃) ∧ 
  (a₃ = a₁ * a₂) ∧ 
  (a₄ = a₁ * a₅) ∧ 
  (a₅ = a₁ * a₄) := by
sorry

end NUMINAMATH_CALUDE_sequence_conditions_l2897_289794


namespace NUMINAMATH_CALUDE_population_difference_l2897_289798

/-- Given that the sum of populations of City A and City B exceeds the sum of populations
    of City B and City C by 5000, prove that the population of City A exceeds
    the population of City C by 5000. -/
theorem population_difference (A B C : ℕ) 
  (h : A + B = B + C + 5000) : A - C = 5000 := by
  sorry

end NUMINAMATH_CALUDE_population_difference_l2897_289798


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2897_289768

-- 1. Prove that (-10) - (-22) + (-8) - 13 = -9
theorem problem_1 : (-10) - (-22) + (-8) - 13 = -9 := by sorry

-- 2. Prove that (-7/9 + 5/6 - 3/4) * (-36) = 25
theorem problem_2 : (-7/9 + 5/6 - 3/4) * (-36) = 25 := by sorry

-- 3. Prove that the solution to 6x - 7 = 4x - 5 is x = 1
theorem problem_3 : ∃ x : ℝ, 6*x - 7 = 4*x - 5 ∧ x = 1 := by sorry

-- 4. Prove that the solution to (x-3)/2 - (2x)/3 = 1 is x = -15
theorem problem_4 : ∃ x : ℝ, (x-3)/2 - (2*x)/3 = 1 ∧ x = -15 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2897_289768


namespace NUMINAMATH_CALUDE_unique_solution_l2897_289709

theorem unique_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * Real.sqrt b - c = a) ∧ 
  (b * Real.sqrt c - a = b) ∧ 
  (c * Real.sqrt a - b = c) →
  a = 4 ∧ b = 4 ∧ c = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l2897_289709


namespace NUMINAMATH_CALUDE_ellipse_circle_tangent_relation_l2897_289720

-- Define the ellipse
def is_on_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle
def is_on_circle (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Define the tangent line
def is_on_line (x y k m : ℝ) : Prop := y = k * x + m

-- Define the condition that the line is tangent to the circle
def is_tangent_to_circle (k m r : ℝ) : Prop := m^2 = (1 + k^2) * r^2

-- Main theorem
theorem ellipse_circle_tangent_relation 
  (a b r k m x₁ y₁ x₂ y₂ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hr : 0 < r) (hrb : r < b) :
  is_on_ellipse x₁ y₁ a b ∧ 
  is_on_ellipse x₂ y₂ a b ∧
  is_on_line x₁ y₁ k m ∧
  is_on_line x₂ y₂ k m ∧
  is_tangent_to_circle k m r ∧
  x₁ * x₂ + y₁ * y₂ = 0 →
  r^2 * (a^2 + b^2) = a^2 * b^2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_circle_tangent_relation_l2897_289720


namespace NUMINAMATH_CALUDE_simplify_expression_l2897_289707

theorem simplify_expression (x : ℝ) (h : x^8 ≠ 1) :
  4 / (1 + x^4) + 2 / (1 + x^2) + 1 / (1 + x) + 1 / (1 - x) = 8 / (1 - x^8) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2897_289707


namespace NUMINAMATH_CALUDE_construct_remaining_vertices_l2897_289749

/-- Represents a point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → Point2D

/-- Represents a parallel projection of a regular hexagon onto a plane -/
structure ParallelProjection where
  original : RegularHexagon
  projected : Fin 6 → Point2D

/-- Given three consecutive projected vertices of a regular hexagon, 
    the remaining three vertices can be uniquely determined -/
theorem construct_remaining_vertices 
  (p : ParallelProjection) 
  (h : ∃ (i : Fin 6), 
       (p.projected i).x ≠ (p.projected (i + 1)).x ∨ 
       (p.projected i).y ≠ (p.projected (i + 1)).y) :
  ∃! (q : ParallelProjection), 
    (∃ (i : Fin 6), 
      q.projected i = p.projected i ∧ 
      q.projected (i + 1) = p.projected (i + 1) ∧ 
      q.projected (i + 2) = p.projected (i + 2)) ∧
    (∀ (j : Fin 6), q.projected j = p.projected j) :=
  sorry

end NUMINAMATH_CALUDE_construct_remaining_vertices_l2897_289749


namespace NUMINAMATH_CALUDE_lcm_10_14_20_l2897_289762

theorem lcm_10_14_20 : Nat.lcm 10 (Nat.lcm 14 20) = 140 := by sorry

end NUMINAMATH_CALUDE_lcm_10_14_20_l2897_289762


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l2897_289783

theorem simultaneous_equations_solution (n : ℝ) :
  n ≠ (1/2 : ℝ) ↔ ∃ (x y : ℝ), y = (3*n + 1)*x + 2 ∧ y = (5*n - 2)*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l2897_289783


namespace NUMINAMATH_CALUDE_max_reflections_l2897_289711

/-- The angle between two lines in degrees -/
def angle_between_lines : ℝ := 12

/-- The maximum angle of incidence before reflection becomes impossible -/
def max_angle : ℝ := 90

/-- The number of reflections -/
def n : ℕ := 7

/-- Theorem stating that 7 is the maximum number of reflections possible -/
theorem max_reflections :
  (n : ℝ) * angle_between_lines ≤ max_angle ∧
  ((n + 1) : ℝ) * angle_between_lines > max_angle :=
sorry

end NUMINAMATH_CALUDE_max_reflections_l2897_289711


namespace NUMINAMATH_CALUDE_food_allocation_l2897_289781

/-- Given a total budget allocated among three categories in a specific ratio,
    calculate the amount allocated to the second category. -/
def allocate_budget (total : ℚ) (ratio1 ratio2 ratio3 : ℕ) : ℚ :=
  (total * ratio2) / (ratio1 + ratio2 + ratio3)

/-- Theorem stating that given a total budget of 1800 allocated in the ratio 5:4:1,
    the amount allocated to the second category is 720. -/
theorem food_allocation :
  allocate_budget 1800 5 4 1 = 720 := by
  sorry

end NUMINAMATH_CALUDE_food_allocation_l2897_289781


namespace NUMINAMATH_CALUDE_quadruple_reappearance_l2897_289723

/-- The transformation function that generates the next quadruple -/
def transform (q : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let (a, b, c, d) := q
  (a * b, b * c, c * d, d * a)

/-- The sequence of quadruples generated by repeatedly applying the transformation -/
def quadruple_sequence (initial : ℝ × ℝ × ℝ × ℝ) : ℕ → ℝ × ℝ × ℝ × ℝ
  | 0 => initial
  | n + 1 => transform (quadruple_sequence initial n)

theorem quadruple_reappearance (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∃ (n : ℕ), n > 0 ∧ quadruple_sequence (a, b, c, d) n = (a, b, c, d)) →
  a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 :=
sorry

end NUMINAMATH_CALUDE_quadruple_reappearance_l2897_289723


namespace NUMINAMATH_CALUDE_max_pages_proof_l2897_289732

/-- The cost in cents to copy one page -/
def cost_per_page : ℕ := 3

/-- The budget in dollars -/
def budget : ℕ := 15

/-- The maximum number of pages that can be copied -/
def max_pages : ℕ := (budget * 100) / cost_per_page

theorem max_pages_proof : max_pages = 500 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_proof_l2897_289732


namespace NUMINAMATH_CALUDE_decimal_119_equals_base6_315_l2897_289756

/-- Converts a natural number to its base 6 representation as a list of digits -/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 6) ((m % 6) :: acc)
    go n []

/-- Converts a list of base 6 digits to its decimal (base 10) value -/
def fromBase6 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => 6 * acc + d) 0

theorem decimal_119_equals_base6_315 : toBase6 119 = [3, 1, 5] ∧ fromBase6 [3, 1, 5] = 119 := by
  sorry

#eval toBase6 119  -- Should output [3, 1, 5]
#eval fromBase6 [3, 1, 5]  -- Should output 119

end NUMINAMATH_CALUDE_decimal_119_equals_base6_315_l2897_289756


namespace NUMINAMATH_CALUDE_intersection_equality_implies_possible_a_l2897_289702

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a*x + 2 = 0}

-- Define the set of possible values for a
def possible_a : Set ℝ := {-1, 0, 2/3}

-- Theorem statement
theorem intersection_equality_implies_possible_a :
  ∀ a : ℝ, (M ∩ N a = N a) → a ∈ possible_a :=
by sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_possible_a_l2897_289702


namespace NUMINAMATH_CALUDE_correct_arrangement_count_l2897_289760

/-- The number of ways to arrange 2 boys and 3 girls in a row with specific conditions -/
def arrangementCount : ℕ :=
  let totalPeople : ℕ := 5
  let boys : ℕ := 2
  let girls : ℕ := 3
  let boyA : ℕ := 1
  48

/-- Theorem stating that the number of arrangements satisfying the given conditions is 48 -/
theorem correct_arrangement_count :
  arrangementCount = 48 :=
by sorry

end NUMINAMATH_CALUDE_correct_arrangement_count_l2897_289760


namespace NUMINAMATH_CALUDE_largest_s_value_l2897_289774

/-- The largest possible value of s for regular polygons P₁ (r-gon) and P₂ (s-gon) 
    satisfying the given conditions -/
theorem largest_s_value : ℕ := by
  /- Define the interior angle of a regular n-gon -/
  let interior_angle (n : ℕ) : ℚ := (n - 2 : ℚ) * 180 / n

  /- Define the relationship between r and s based on the interior angle ratio -/
  let r_s_relation (r s : ℕ) : Prop :=
    interior_angle r / interior_angle s = 29 / 28

  /- Define the conditions on r and s -/
  let valid_r_s (r s : ℕ) : Prop :=
    r ≥ s ∧ s ≥ 3 ∧ r_s_relation r s

  /- The theorem states that 114 is the largest value of s satisfying all conditions -/
  have h1 : ∃ (r : ℕ), valid_r_s r 114 := sorry
  have h2 : ∀ (s : ℕ), s > 114 → ¬∃ (r : ℕ), valid_r_s r s := sorry

  exact 114

end NUMINAMATH_CALUDE_largest_s_value_l2897_289774


namespace NUMINAMATH_CALUDE_sqrt_s6_plus_s3_l2897_289759

theorem sqrt_s6_plus_s3 (s : ℝ) : Real.sqrt (s^6 + s^3) = |s| * Real.sqrt (s * (s^3 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_s6_plus_s3_l2897_289759


namespace NUMINAMATH_CALUDE_choose_three_from_ten_l2897_289767

theorem choose_three_from_ten : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_ten_l2897_289767


namespace NUMINAMATH_CALUDE_remaining_segments_length_l2897_289739

/-- Represents the dimensions of the initial polygon --/
structure PolygonDimensions where
  vertical1 : ℝ
  horizontal1 : ℝ
  vertical2 : ℝ
  horizontal2 : ℝ
  vertical3 : ℝ
  horizontal3 : ℝ

/-- Calculates the total length of segments in the polygon --/
def totalLength (d : PolygonDimensions) : ℝ :=
  d.vertical1 + d.horizontal1 + d.vertical2 + d.horizontal2 + d.vertical3 + d.horizontal3

/-- Theorem: The length of remaining segments after removal is 21 units --/
theorem remaining_segments_length
  (d : PolygonDimensions)
  (h1 : d.vertical1 = 10)
  (h2 : d.horizontal1 = 5)
  (h3 : d.vertical2 = 4)
  (h4 : d.horizontal2 = 3)
  (h5 : d.vertical3 = 4)
  (h6 : d.horizontal3 = 2)
  (h7 : totalLength d = 28)
  (h8 : ∃ (removed : ℝ), removed = 7) :
  totalLength d - 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_remaining_segments_length_l2897_289739


namespace NUMINAMATH_CALUDE_cosine_translation_monotonicity_l2897_289721

/-- Given a function g(x) = 2cos(2x - π/3) that is monotonically increasing
    in the intervals [0, a/3] and [2a, 7π/6], prove that π/3 ≤ a ≤ π/2. -/
theorem cosine_translation_monotonicity (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (a / 3), Monotone (fun x => 2 * Real.cos (2 * x - π / 3))) ∧
  (∀ x ∈ Set.Icc (2 * a) (7 * π / 6), Monotone (fun x => 2 * Real.cos (2 * x - π / 3))) →
  π / 3 ≤ a ∧ a ≤ π / 2 :=
by sorry

end NUMINAMATH_CALUDE_cosine_translation_monotonicity_l2897_289721


namespace NUMINAMATH_CALUDE_valid_seating_arrangements_l2897_289735

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to seat 9 people in a row with the given constraint. -/
def seating_arrangements : ℕ :=
  factorial 9 - factorial 7 * factorial 3

/-- Theorem stating the number of valid seating arrangements. -/
theorem valid_seating_arrangements :
  seating_arrangements = 332640 := by sorry

end NUMINAMATH_CALUDE_valid_seating_arrangements_l2897_289735


namespace NUMINAMATH_CALUDE_jolene_babysitting_l2897_289785

theorem jolene_babysitting (babysitting_rate : ℕ) (car_wash_rate : ℕ) (num_cars : ℕ) (total_raised : ℕ) :
  babysitting_rate = 30 →
  car_wash_rate = 12 →
  num_cars = 5 →
  total_raised = 180 →
  ∃ (num_families : ℕ), num_families * babysitting_rate + num_cars * car_wash_rate = total_raised ∧ num_families = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_jolene_babysitting_l2897_289785


namespace NUMINAMATH_CALUDE_rosas_phone_calls_l2897_289763

/-- Rosa's phone calls over two weeks -/
theorem rosas_phone_calls (last_week : ℝ) (this_week : ℝ) 
  (h1 : last_week = 10.2)
  (h2 : this_week = 8.6) :
  last_week + this_week = 18.8 := by
  sorry

end NUMINAMATH_CALUDE_rosas_phone_calls_l2897_289763


namespace NUMINAMATH_CALUDE_problem_solution_l2897_289755

/-- The distance between two points A and B, given the conditions of the problem -/
def distance_AB : ℝ := 1656

/-- Jun Jun's speed -/
def v_jun : ℝ := 14

/-- Ping's speed -/
def v_ping : ℝ := 9

/-- Distance from C to the point where Jun Jun turns back -/
def d_turn : ℝ := 100

/-- Distance from C to the point where Jun Jun catches up with Ping -/
def d_catchup : ℝ := 360

theorem problem_solution :
  ∃ (d_AC d_BC : ℝ),
    d_AC + d_BC = distance_AB ∧
    d_AC / d_BC = v_jun / v_ping ∧
    d_AC - d_catchup = d_turn + d_catchup ∧
    (d_AC - d_catchup) / (d_BC + d_catchup) = v_ping / v_jun :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2897_289755


namespace NUMINAMATH_CALUDE_production_improvement_l2897_289747

/-- Represents the production efficiency of a team --/
structure ProductionTeam where
  initial_time : ℕ  -- Initial completion time in hours
  ab_swap_reduction : ℕ  -- Time reduction when swapping A and B
  cd_swap_reduction : ℕ  -- Time reduction when swapping C and D

/-- Calculates the time reduction when swapping both A with B and C with D --/
def time_reduction (team : ProductionTeam) : ℕ :=
  -- Definition to be proved
  108

theorem production_improvement (team : ProductionTeam) 
  (h1 : team.initial_time = 9)
  (h2 : team.ab_swap_reduction = 1)
  (h3 : team.cd_swap_reduction = 1) :
  time_reduction team = 108 := by
  sorry


end NUMINAMATH_CALUDE_production_improvement_l2897_289747


namespace NUMINAMATH_CALUDE_joker_selection_ways_l2897_289750

def total_cards : ℕ := 54
def jokers : ℕ := 2
def standard_cards : ℕ := 52

def ways_to_pick_joker_first (cards : ℕ) (jokers : ℕ) : ℕ :=
  jokers * (cards - 1)

def ways_to_pick_joker_second (cards : ℕ) (standard_cards : ℕ) (jokers : ℕ) : ℕ :=
  standard_cards * jokers

theorem joker_selection_ways :
  ways_to_pick_joker_first total_cards jokers +
  ways_to_pick_joker_second total_cards standard_cards jokers = 210 :=
by sorry

end NUMINAMATH_CALUDE_joker_selection_ways_l2897_289750


namespace NUMINAMATH_CALUDE_remaining_cooking_time_l2897_289793

def total_potatoes : ℕ := 15
def cooked_potatoes : ℕ := 8
def cooking_time_per_potato : ℕ := 9

theorem remaining_cooking_time :
  (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 63 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cooking_time_l2897_289793


namespace NUMINAMATH_CALUDE_noemi_initial_amount_l2897_289737

def initial_amount (roulette_loss blackjack_loss remaining : ℕ) : ℕ :=
  roulette_loss + blackjack_loss + remaining

theorem noemi_initial_amount :
  initial_amount 400 500 800 = 1700 := by
  sorry

end NUMINAMATH_CALUDE_noemi_initial_amount_l2897_289737


namespace NUMINAMATH_CALUDE_stock_price_change_l2897_289713

theorem stock_price_change (initial_price : ℝ) (initial_price_positive : initial_price > 0) :
  let price_after_decrease := initial_price * (1 - 0.08)
  let final_price := price_after_decrease * (1 + 0.10)
  let net_change_percentage := (final_price - initial_price) / initial_price * 100
  net_change_percentage = 1.2 := by
sorry

end NUMINAMATH_CALUDE_stock_price_change_l2897_289713


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l2897_289728

theorem complex_fraction_sum : (1 - 2*Complex.I) / (1 + Complex.I) + (1 + 2*Complex.I) / (1 - Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l2897_289728


namespace NUMINAMATH_CALUDE_power_nap_duration_l2897_289745

theorem power_nap_duration : (1 / 5 : ℚ) * 60 = 12 := by sorry

end NUMINAMATH_CALUDE_power_nap_duration_l2897_289745


namespace NUMINAMATH_CALUDE_m_less_than_neg_two_l2897_289701

/-- A quadratic function f(x) = x^2 + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

/-- The proposition that there exists a positive x_0 where f(x_0) < 0 -/
def exists_positive_root (m : ℝ) : Prop :=
  ∃ x_0 : ℝ, x_0 > 0 ∧ f m x_0 < 0

/-- Theorem: If there exists a positive x_0 where f(x_0) < 0, then m < -2 -/
theorem m_less_than_neg_two (m : ℝ) (h : exists_positive_root m) : m < -2 := by
  sorry

end NUMINAMATH_CALUDE_m_less_than_neg_two_l2897_289701


namespace NUMINAMATH_CALUDE_matthews_cracker_distribution_l2897_289789

theorem matthews_cracker_distribution (total_crackers : ℕ) (crackers_per_friend : ℕ) (num_friends : ℕ) :
  total_crackers = 36 →
  crackers_per_friend = 6 →
  total_crackers = crackers_per_friend * num_friends →
  num_friends = 6 := by
sorry

end NUMINAMATH_CALUDE_matthews_cracker_distribution_l2897_289789


namespace NUMINAMATH_CALUDE_a_must_be_negative_l2897_289741

theorem a_must_be_negative (a b : ℝ) (hb : b > 0) (h : a / b < -2/3) : a < 0 := by
  sorry

end NUMINAMATH_CALUDE_a_must_be_negative_l2897_289741


namespace NUMINAMATH_CALUDE_equal_distribution_of_cards_l2897_289780

theorem equal_distribution_of_cards (total_cards : ℕ) (num_friends : ℕ) 
  (h1 : total_cards = 455) (h2 : num_friends = 5) :
  total_cards / num_friends = 91 :=
by sorry

end NUMINAMATH_CALUDE_equal_distribution_of_cards_l2897_289780


namespace NUMINAMATH_CALUDE_count_squarish_numbers_l2897_289782

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_squarish (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  is_perfect_square n ∧
  (∀ d, d ∈ n.digits 10 → d ≠ 0) ∧
  is_perfect_square (n / 100) ∧
  is_perfect_square (n % 100) ∧
  is_two_digit (n / 100) ∧
  is_two_digit (n % 100)

theorem count_squarish_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_squarish n) ∧ s.card = 2 := by sorry

end NUMINAMATH_CALUDE_count_squarish_numbers_l2897_289782


namespace NUMINAMATH_CALUDE_min_distance_ant_spider_l2897_289700

/-- The minimum distance between a point on the unit circle and a corresponding point on the x-axis -/
theorem min_distance_ant_spider :
  let f : ℝ → ℝ := λ a => Real.sqrt ((a - (1 - 2*a))^2 + (Real.sqrt (1 - a^2))^2)
  ∃ a : ℝ, ∀ x : ℝ, f x ≥ f a ∧ f a = Real.sqrt 14 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_ant_spider_l2897_289700


namespace NUMINAMATH_CALUDE_function_equality_l2897_289703

theorem function_equality (a b : ℝ) : 
  (∀ x, (x^2 + 4*x + 3 = (a*x + b)^2 + 4*(a*x + b) + 3)) → 
  ((a + b = -8) ∨ (a + b = 4)) := by
sorry

end NUMINAMATH_CALUDE_function_equality_l2897_289703


namespace NUMINAMATH_CALUDE_least_four_digit_solution_l2897_289761

theorem least_four_digit_solution (x : ℕ) : x = 1002 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧
  (∀ y : ℕ, y ≥ 1000 ∧ y < 10000 →
    (5 * y ≡ 10 [ZMOD 10] ∧
     3 * y + 20 ≡ 29 [ZMOD 12] ∧
     -3 * y + 2 ≡ 2 * y [ZMOD 30]) →
    x ≤ y) ∧
  (5 * x ≡ 10 [ZMOD 10]) ∧
  (3 * x + 20 ≡ 29 [ZMOD 12]) ∧
  (-3 * x + 2 ≡ 2 * x [ZMOD 30]) := by
sorry

end NUMINAMATH_CALUDE_least_four_digit_solution_l2897_289761


namespace NUMINAMATH_CALUDE_circumradius_inradius_inequality_l2897_289753

/-- A triangle with its circumradius and inradius -/
structure Triangle where
  R : ℝ  -- Circumradius
  r : ℝ  -- Inradius

/-- Predicate to check if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop :=
  t.R = 2 * t.r

theorem circumradius_inradius_inequality (t : Triangle) :
  t.R ≥ 2 * t.r ∧ (t.R = 2 * t.r ↔ is_equilateral t) :=
sorry

end NUMINAMATH_CALUDE_circumradius_inradius_inequality_l2897_289753


namespace NUMINAMATH_CALUDE_wrong_number_correction_l2897_289736

theorem wrong_number_correction (n : ℕ) (initial_avg correct_avg : ℚ) 
  (first_error second_correct : ℚ) :
  n = 10 ∧ 
  initial_avg = 40.2 ∧ 
  correct_avg = 40.1 ∧ 
  first_error = 17 ∧ 
  second_correct = 31 →
  ∃ second_error : ℚ,
    n * initial_avg - first_error - second_error + second_correct = n * correct_avg ∧
    second_error = 15 :=
by sorry

end NUMINAMATH_CALUDE_wrong_number_correction_l2897_289736


namespace NUMINAMATH_CALUDE_store_profit_l2897_289784

theorem store_profit (price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) :
  price = 64 ∧ 
  profit_percent = 60 ∧ 
  loss_percent = 20 →
  let cost1 := price / (1 + profit_percent / 100)
  let cost2 := price / (1 - loss_percent / 100)
  price * 2 - (cost1 + cost2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_store_profit_l2897_289784


namespace NUMINAMATH_CALUDE_cos_150_degrees_l2897_289787

theorem cos_150_degrees :
  Real.cos (150 * π / 180) = -(1 / 2) := by sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l2897_289787


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l2897_289727

/-- A circle with a given center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line tangent to two circles -/
structure TangentLine where
  circle1 : Circle
  circle2 : Circle
  tangentPoint1 : ℝ × ℝ
  tangentPoint2 : ℝ × ℝ

/-- The y-intercept of a line tangent to two specific circles -/
def yIntercept (line : TangentLine) : ℝ :=
  sorry

/-- The main theorem stating the y-intercept of the tangent line -/
theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (2, 0), radius := 2 }
  let c2 : Circle := { center := (5, 0), radius := 1 }
  ∀ (line : TangentLine),
    line.circle1 = c1 →
    line.circle2 = c2 →
    line.tangentPoint1.1 > 2 →
    line.tangentPoint1.2 > 0 →
    line.tangentPoint2.1 > 5 →
    line.tangentPoint2.2 > 0 →
    yIntercept line = 2 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l2897_289727


namespace NUMINAMATH_CALUDE_min_value_theorem_l2897_289718

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_eq : 2*m + n = 2) :
  (2/m) + (1/n) ≥ 9/2 ∧ ∃ m n, m > 0 ∧ n > 0 ∧ 2*m + n = 2 ∧ (2/m) + (1/n) = 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2897_289718


namespace NUMINAMATH_CALUDE_teacher_grading_problem_l2897_289757

/-- Calculates the number of problems left to grade given the total number of worksheets,
    the number of graded worksheets, and the number of problems per worksheet. -/
def problems_left_to_grade (total_worksheets : ℕ) (graded_worksheets : ℕ) (problems_per_worksheet : ℕ) : ℕ :=
  (total_worksheets - graded_worksheets) * problems_per_worksheet

/-- Proves that given 9 total worksheets, 5 graded worksheets, and 4 problems per worksheet,
    there are 16 problems left to grade. -/
theorem teacher_grading_problem :
  problems_left_to_grade 9 5 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_teacher_grading_problem_l2897_289757


namespace NUMINAMATH_CALUDE_yuna_survey_l2897_289792

theorem yuna_survey (math_lovers : ℕ) (korean_lovers : ℕ) (both_lovers : ℕ)
  (h1 : math_lovers = 27)
  (h2 : korean_lovers = 28)
  (h3 : both_lovers = 22) :
  math_lovers + korean_lovers - both_lovers = 33 := by
  sorry

end NUMINAMATH_CALUDE_yuna_survey_l2897_289792


namespace NUMINAMATH_CALUDE_count_negative_numbers_l2897_289796

theorem count_negative_numbers : ∃ (negative_count : ℕ), 
  negative_count = 2 ∧ 
  negative_count = (if (-1 : ℚ)^2007 < 0 then 1 else 0) + 
                   (if (|(-1 : ℚ)|^3 : ℚ) < 0 then 1 else 0) + 
                   (if (-1 : ℚ)^18 > 0 then 1 else 0) + 
                   (if (18 : ℚ) < 0 then 1 else 0) := by
  sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l2897_289796


namespace NUMINAMATH_CALUDE_sin_cos_value_l2897_289740

theorem sin_cos_value (x : Real) (h : 2 * Real.sin x = 5 * Real.cos x) : 
  Real.sin x * Real.cos x = 10 / 29 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_value_l2897_289740


namespace NUMINAMATH_CALUDE_estimated_percentage_is_5_7_l2897_289790

/-- Represents the data from the household survey -/
structure SurveyData where
  total_households : ℕ
  ordinary_families : ℕ
  high_income_families : ℕ
  ordinary_sample_size : ℕ
  high_income_sample_size : ℕ
  ordinary_with_3plus_houses : ℕ
  high_income_with_3plus_houses : ℕ

/-- Calculates the estimated percentage of families with 3 or more houses -/
def estimatePercentage (data : SurveyData) : ℚ :=
  let ordinary_estimate := (data.ordinary_families : ℚ) * (data.ordinary_with_3plus_houses : ℚ) / (data.ordinary_sample_size : ℚ)
  let high_income_estimate := (data.high_income_families : ℚ) * (data.high_income_with_3plus_houses : ℚ) / (data.high_income_sample_size : ℚ)
  let total_estimate := ordinary_estimate + high_income_estimate
  (total_estimate / (data.total_households : ℚ)) * 100

/-- The survey data for the household study -/
def surveyData : SurveyData := {
  total_households := 100000,
  ordinary_families := 99000,
  high_income_families := 1000,
  ordinary_sample_size := 990,
  high_income_sample_size := 100,
  ordinary_with_3plus_houses := 50,
  high_income_with_3plus_houses := 70
}

/-- Theorem stating that the estimated percentage of families with 3 or more houses is 5.7% -/
theorem estimated_percentage_is_5_7 :
  estimatePercentage surveyData = 57/10 := by
  sorry


end NUMINAMATH_CALUDE_estimated_percentage_is_5_7_l2897_289790


namespace NUMINAMATH_CALUDE_number_problem_l2897_289778

theorem number_problem (x : ℚ) : 4 * x + 7 * x = 55 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2897_289778


namespace NUMINAMATH_CALUDE_sqrt_200_simplification_l2897_289771

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_simplification_l2897_289771


namespace NUMINAMATH_CALUDE_bob_has_22_pennies_l2897_289733

/-- The number of pennies Alex currently has -/
def alex_pennies : ℕ := sorry

/-- The number of pennies Bob currently has -/
def bob_pennies : ℕ := sorry

/-- Condition 1: If Alex gives Bob two pennies, Bob will have four times as many pennies as Alex has left -/
axiom condition1 : bob_pennies + 2 = 4 * (alex_pennies - 2)

/-- Condition 2: If Bob gives Alex two pennies, Bob will have twice as many pennies as Alex has -/
axiom condition2 : bob_pennies - 2 = 2 * (alex_pennies + 2)

/-- Theorem: Bob currently has 22 pennies -/
theorem bob_has_22_pennies : bob_pennies = 22 := by sorry

end NUMINAMATH_CALUDE_bob_has_22_pennies_l2897_289733


namespace NUMINAMATH_CALUDE_critical_point_of_cubic_l2897_289716

/-- The function f(x) = x^3 -/
def f (x : ℝ) : ℝ := x^3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2

theorem critical_point_of_cubic (x : ℝ) : 
  (f' x = 0 ↔ x = 0) :=
sorry

#check critical_point_of_cubic

end NUMINAMATH_CALUDE_critical_point_of_cubic_l2897_289716


namespace NUMINAMATH_CALUDE_number_of_pieces_l2897_289731

-- Define the rod length in meters
def rod_length_meters : ℝ := 42.5

-- Define the piece length in centimeters
def piece_length_cm : ℝ := 85

-- Define the conversion factor from meters to centimeters
def meters_to_cm : ℝ := 100

-- Theorem to prove
theorem number_of_pieces : 
  ⌊(rod_length_meters * meters_to_cm) / piece_length_cm⌋ = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pieces_l2897_289731


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2897_289795

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S = (√3/4)(a² + b² - c²), then the measure of angle C is π/3. -/
theorem triangle_angle_measure (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := Real.sqrt 3 / 4 * (a^2 + b^2 - c^2)
  S = 1/2 * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) →
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_measure_l2897_289795


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l2897_289775

theorem log_sum_equals_two :
  2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l2897_289775


namespace NUMINAMATH_CALUDE_triangle_inequality_l2897_289746

theorem triangle_inequality (a b c : ℝ) (x y z : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) 
  (h3 : 0 ≤ x ∧ x ≤ π) (h4 : 0 ≤ y ∧ y ≤ π) (h5 : 0 ≤ z ∧ z ≤ π)
  (h6 : x + y + z = π) : 
  b * c + c * a - a * b < 
  b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ∧
  b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ≤ 
  (1 / 2) * (a^2 + b^2 + c^2) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2897_289746


namespace NUMINAMATH_CALUDE_product_reciprocals_equals_one_l2897_289717

theorem product_reciprocals_equals_one :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_reciprocals_equals_one_l2897_289717


namespace NUMINAMATH_CALUDE_apples_per_case_l2897_289758

theorem apples_per_case (total_apples : ℕ) (num_cases : ℕ) (h1 : total_apples = 1080) (h2 : num_cases = 90) :
  total_apples / num_cases = 12 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_case_l2897_289758


namespace NUMINAMATH_CALUDE_pyramid_has_one_base_l2897_289726

/-- A pyramid is a polyhedron with a polygonal base and triangular faces meeting at a point (apex) --/
structure Pyramid where
  base : Set Point
  apex : Point
  faces : Set (Set Point)

/-- Any pyramid has only one base --/
theorem pyramid_has_one_base (p : Pyramid) : ∃! b : Set Point, b = p.base := by
  sorry

end NUMINAMATH_CALUDE_pyramid_has_one_base_l2897_289726


namespace NUMINAMATH_CALUDE_june_election_win_l2897_289729

theorem june_election_win (total_students : ℕ) (boy_percentage : ℚ) (june_male_vote_percentage : ℚ) 
  (h_total : total_students = 200)
  (h_boy : boy_percentage = 3/5)
  (h_june_male : june_male_vote_percentage = 27/40)
  : ∃ (min_female_vote_percentage : ℚ), 
    min_female_vote_percentage ≥ 1/4 ∧ 
    (boy_percentage * june_male_vote_percentage + (1 - boy_percentage) * min_female_vote_percentage) * total_students > total_students / 2 := by
  sorry

end NUMINAMATH_CALUDE_june_election_win_l2897_289729


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2897_289714

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2897_289714


namespace NUMINAMATH_CALUDE_gcd_binomial_integer_l2897_289738

theorem gcd_binomial_integer (m n : ℕ) (h1 : 1 ≤ m) (h2 : m ≤ n) :
  ∃ k : ℤ, (Nat.gcd m n : ℚ) / n * (n.choose m : ℚ) = k := by
  sorry

end NUMINAMATH_CALUDE_gcd_binomial_integer_l2897_289738


namespace NUMINAMATH_CALUDE_removed_triangles_area_l2897_289722

theorem removed_triangles_area (s : ℝ) (x : ℝ) : 
  s = 16 → 
  (s - 2*x)^2 + (s - 2*x)^2 = s^2 →
  2 * x^2 = 768 - 512 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_removed_triangles_area_l2897_289722


namespace NUMINAMATH_CALUDE_partitions_6_3_l2897_289764

def partitions (n : ℕ) (k : ℕ) : ℕ := sorry

theorem partitions_6_3 : partitions 6 3 = 7 := by sorry

end NUMINAMATH_CALUDE_partitions_6_3_l2897_289764


namespace NUMINAMATH_CALUDE_women_at_gathering_l2897_289773

/-- The number of women at a social gathering --/
def number_of_women (number_of_men : ℕ) (dances_per_man : ℕ) (dances_per_woman : ℕ) : ℕ :=
  (number_of_men * dances_per_man) / dances_per_woman

/-- Theorem: At a social gathering with the given conditions, 20 women attended --/
theorem women_at_gathering :
  let number_of_men : ℕ := 15
  let dances_per_man : ℕ := 4
  let dances_per_woman : ℕ := 3
  number_of_women number_of_men dances_per_man dances_per_woman = 20 := by
sorry

#eval number_of_women 15 4 3

end NUMINAMATH_CALUDE_women_at_gathering_l2897_289773


namespace NUMINAMATH_CALUDE_january_salary_l2897_289765

/-- Given the average salaries for two four-month periods and the salary for May,
    prove that the salary for January is 5700. -/
theorem january_salary
  (avg_jan_to_apr : (jan + feb + mar + apr) / 4 = 8000)
  (avg_feb_to_may : (feb + mar + apr + may) / 4 = 8200)
  (may_salary : may = 6500)
  : jan = 5700 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_l2897_289765


namespace NUMINAMATH_CALUDE_jimin_seokjin_money_sum_l2897_289786

/-- Calculates the total amount of money for a person given their coin distribution --/
def calculate_total (coins_100 : Nat) (coins_50 : Nat) (coins_10 : Nat) : Nat :=
  100 * coins_100 + 50 * coins_50 + 10 * coins_10

/-- Represents the coin distribution and total money for Jimin and Seokjin --/
theorem jimin_seokjin_money_sum :
  let jimin_total := calculate_total 5 1 0
  let seokjin_total := calculate_total 2 0 7
  jimin_total + seokjin_total = 820 := by
  sorry

#check jimin_seokjin_money_sum

end NUMINAMATH_CALUDE_jimin_seokjin_money_sum_l2897_289786


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l2897_289704

theorem polynomial_multiplication (x : ℝ) :
  (3 * x^2 - 4 * x + 5) * (-2 * x^2 + 3 * x - 7) =
  -6 * x^4 + 17 * x^3 - 43 * x^2 + 43 * x - 35 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l2897_289704


namespace NUMINAMATH_CALUDE_mikes_second_job_hours_l2897_289752

/-- Given Mike's total wages, wages from his first job, and hourly rate at his second job,
    calculate the number of hours he worked at his second job. -/
theorem mikes_second_job_hours
  (total_wages : ℕ)
  (first_job_wages : ℕ)
  (second_job_hourly_rate : ℕ)
  (h1 : total_wages = 160)
  (h2 : first_job_wages = 52)
  (h3 : second_job_hourly_rate = 9) :
  (total_wages - first_job_wages) / second_job_hourly_rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_mikes_second_job_hours_l2897_289752


namespace NUMINAMATH_CALUDE_function_characterization_l2897_289748

theorem function_characterization (f : ℝ → ℝ) (C : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x ≥ 0) →
  (∀ x : ℝ, x ≥ 0 → f (f x) = x^4) →
  (∀ x : ℝ, x ≥ 0 → f x ≤ C * x^2) →
  C ≥ 1 →
  (∀ x : ℝ, x ≥ 0 → f x = x^2) :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l2897_289748


namespace NUMINAMATH_CALUDE_meaningful_expression_l2897_289766

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 2)) / (x - 1)) ↔ x ≥ -2 ∧ x ≠ 1 := by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2897_289766


namespace NUMINAMATH_CALUDE_bananas_cantaloupe_cost_l2897_289777

/-- Represents the cost of groceries -/
structure GroceryCost where
  apples : ℝ
  bananas : ℝ
  cantaloupe : ℝ
  dates : ℝ

/-- The total cost of all items is $40 -/
def total_cost (g : GroceryCost) : Prop :=
  g.apples + g.bananas + g.cantaloupe + g.dates = 40

/-- A carton of dates costs three times as much as a sack of apples -/
def dates_cost (g : GroceryCost) : Prop :=
  g.dates = 3 * g.apples

/-- The price of a cantaloupe is equal to half the sum of the price of a sack of apples and a bunch of bananas -/
def cantaloupe_cost (g : GroceryCost) : Prop :=
  g.cantaloupe = (g.apples + g.bananas) / 2

/-- The main theorem: Given the conditions, the cost of a bunch of bananas and a cantaloupe is $8 -/
theorem bananas_cantaloupe_cost (g : GroceryCost) 
  (h1 : total_cost g) 
  (h2 : dates_cost g) 
  (h3 : cantaloupe_cost g) : 
  g.bananas + g.cantaloupe = 8 := by
  sorry

end NUMINAMATH_CALUDE_bananas_cantaloupe_cost_l2897_289777


namespace NUMINAMATH_CALUDE_savings_in_cents_l2897_289799

/-- The in-store price of the appliance in dollars -/
def in_store_price : ℚ := 99.99

/-- The price of one payment in the TV commercial in dollars -/
def tv_payment : ℚ := 29.98

/-- The number of payments in the TV commercial -/
def num_payments : ℕ := 3

/-- The shipping and handling charge in dollars -/
def shipping_charge : ℚ := 9.98

/-- The total cost from the TV advertiser in dollars -/
def tv_total_cost : ℚ := tv_payment * num_payments + shipping_charge

/-- The savings in dollars -/
def savings : ℚ := in_store_price - tv_total_cost

/-- Convert dollars to cents -/
def dollars_to_cents (dollars : ℚ) : ℕ := (dollars * 100).ceil.toNat

theorem savings_in_cents : dollars_to_cents savings = 7 := by sorry

end NUMINAMATH_CALUDE_savings_in_cents_l2897_289799


namespace NUMINAMATH_CALUDE_valid_a_values_l2897_289772

def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2 - a + 1}

theorem valid_a_values :
  ∀ a : ℝ, (A a ⊇ B a) → (a = -1 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_valid_a_values_l2897_289772


namespace NUMINAMATH_CALUDE_garden_trees_l2897_289769

/-- The number of trees in a garden with given specifications -/
def num_trees (yard_length : ℕ) (tree_spacing : ℕ) : ℕ :=
  yard_length / tree_spacing + 1

/-- Theorem stating the number of trees in the garden -/
theorem garden_trees : num_trees 700 28 = 26 := by
  sorry

end NUMINAMATH_CALUDE_garden_trees_l2897_289769


namespace NUMINAMATH_CALUDE_namjoon_position_l2897_289734

theorem namjoon_position (total_students : ℕ) (position_from_left : ℕ) :
  total_students = 15 →
  position_from_left = 7 →
  total_students - position_from_left + 1 = 9 :=
by sorry

end NUMINAMATH_CALUDE_namjoon_position_l2897_289734


namespace NUMINAMATH_CALUDE_not_in_range_iff_a_in_interval_l2897_289754

/-- The function g(x) = x^2 + ax + 3 -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

/-- Theorem: -3 is not in the range of g(x) if and only if a is in the open interval (-√24, √24) -/
theorem not_in_range_iff_a_in_interval (a : ℝ) :
  (∀ x : ℝ, g a x ≠ -3) ↔ a > -Real.sqrt 24 ∧ a < Real.sqrt 24 := by
  sorry

end NUMINAMATH_CALUDE_not_in_range_iff_a_in_interval_l2897_289754


namespace NUMINAMATH_CALUDE_jersey_revenue_proof_l2897_289730

/-- The amount of money made from selling jerseys -/
def jersey_revenue (price_per_jersey : ℕ) (jerseys_sold : ℕ) : ℕ :=
  price_per_jersey * jerseys_sold

/-- Proof that the jersey revenue is $25,740 -/
theorem jersey_revenue_proof :
  jersey_revenue 165 156 = 25740 := by
  sorry

end NUMINAMATH_CALUDE_jersey_revenue_proof_l2897_289730


namespace NUMINAMATH_CALUDE_henrys_money_l2897_289725

/-- Henry's money calculation -/
theorem henrys_money (initial_amount : ℕ) (birthday_gift : ℕ) (spent_amount : ℕ) : 
  initial_amount = 11 → birthday_gift = 18 → spent_amount = 10 → 
  initial_amount + birthday_gift - spent_amount = 19 := by
  sorry

#check henrys_money

end NUMINAMATH_CALUDE_henrys_money_l2897_289725


namespace NUMINAMATH_CALUDE_number_divisible_by_56_l2897_289710

/-- The number formed by concatenating digits a, 7, 8, 3, and b -/
def number (a b : ℕ) : ℕ := a * 10000 + 7000 + 800 + 30 + b

/-- Theorem stating that 47832 is divisible by 56 -/
theorem number_divisible_by_56 : 
  number 4 2 % 56 = 0 :=
sorry

end NUMINAMATH_CALUDE_number_divisible_by_56_l2897_289710


namespace NUMINAMATH_CALUDE_brendas_cakes_l2897_289788

/-- Theorem: If Brenda bakes x cakes per day for 9 days, sells half of the total cakes,
    and has 90 cakes left after selling, then x = 20. -/
theorem brendas_cakes (x : ℕ) : x * 9 / 2 = 90 → x = 20 := by sorry

end NUMINAMATH_CALUDE_brendas_cakes_l2897_289788


namespace NUMINAMATH_CALUDE_cyclic_fraction_inequality_l2897_289751

theorem cyclic_fraction_inequality (a b x y z : ℝ) (ha : a > 0) (hb : b > 0) :
  x / (a * y + b * z) + y / (a * z + b * x) + z / (a * x + b * y) ≥ 3 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_fraction_inequality_l2897_289751


namespace NUMINAMATH_CALUDE_polar_equation_circle_and_ray_l2897_289708

/-- The polar equation (ρ - 1)(θ - π) = 0 with ρ ≥ 0 represents the union of a circle and a ray -/
theorem polar_equation_circle_and_ray (ρ θ : ℝ) :
  ρ ≥ 0 → (ρ - 1) * (θ - Real.pi) = 0 → 
  (∃ (x y : ℝ), x^2 + y^2 = 1) ∨ 
  (∃ (t : ℝ), t ≥ 0 → ∃ (x y : ℝ), x = -t ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_polar_equation_circle_and_ray_l2897_289708


namespace NUMINAMATH_CALUDE_softball_team_ratio_l2897_289705

theorem softball_team_ratio :
  ∀ (men women : ℕ),
  women = men + 4 →
  men + women = 18 →
  (men : ℚ) / women = 7 / 11 := by
sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l2897_289705


namespace NUMINAMATH_CALUDE_sufficient_conditions_for_positive_product_l2897_289779

theorem sufficient_conditions_for_positive_product (a b : ℝ) :
  ((a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) ∨ (a > 1 ∧ b > 1)) → a * b > 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_conditions_for_positive_product_l2897_289779


namespace NUMINAMATH_CALUDE_equation_system_solution_l2897_289776

theorem equation_system_solution : 
  ∀ (x y z : ℝ), 
    z ≠ 0 →
    3 * x - 4 * y - 2 * z = 0 →
    x - 2 * y + 5 * z = 0 →
    (2 * x^2 - x * y) / (y^2 + 4 * z^2) = 744 / 305 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l2897_289776


namespace NUMINAMATH_CALUDE_smallest_sum_of_abs_l2897_289712

def matrix_squared (a b c d : ℤ) : Matrix (Fin 2) (Fin 2) ℤ :=
  !![a^2 + b*c, a*b + b*d;
     a*c + c*d, b*c + d^2]

theorem smallest_sum_of_abs (a b c d : ℤ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  matrix_squared a b c d = !![9, 0; 0, 9] →
  (∃ (w x y z : ℤ), w ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    matrix_squared w x y z = !![9, 0; 0, 9] ∧
    |w| + |x| + |y| + |z| < |a| + |b| + |c| + |d|) ∨
  |a| + |b| + |c| + |d| = 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_abs_l2897_289712


namespace NUMINAMATH_CALUDE_bridge_length_l2897_289706

/-- The length of a bridge given specific train parameters -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 100 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 275 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l2897_289706


namespace NUMINAMATH_CALUDE_waist_size_conversion_l2897_289743

/-- Converts inches to centimeters given the conversion rates and waist size --/
def inches_to_cm (inches_per_foot : ℚ) (cm_per_foot : ℚ) (waist_inches : ℚ) : ℚ :=
  (waist_inches / inches_per_foot) * cm_per_foot

/-- Theorem: Given the conversion rates and waist size, proves that 40 inches equals 100 cm --/
theorem waist_size_conversion :
  let inches_per_foot : ℚ := 10
  let cm_per_foot : ℚ := 25
  let waist_inches : ℚ := 40
  inches_to_cm inches_per_foot cm_per_foot waist_inches = 100 := by
  sorry

end NUMINAMATH_CALUDE_waist_size_conversion_l2897_289743


namespace NUMINAMATH_CALUDE_pure_imaginary_square_root_l2897_289724

theorem pure_imaginary_square_root (a : ℝ) :
  (∃ (b : ℝ), (a - Complex.I) ^ 2 = Complex.I * b) → (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_square_root_l2897_289724


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_least_addition_to_1024_for_25_divisibility_l2897_289719

theorem least_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
by sorry

theorem least_addition_to_1024_for_25_divisibility :
  ∃ (x : ℕ), x < 25 ∧ (1024 + x) % 25 = 0 ∧ ∀ (y : ℕ), y < x → (1024 + y) % 25 ≠ 0 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_least_addition_to_1024_for_25_divisibility_l2897_289719


namespace NUMINAMATH_CALUDE_correct_answers_for_given_score_l2897_289715

/-- Represents the scoring system for a test -/
structure TestScore where
  total_questions : ℕ
  correct_answers : ℕ
  incorrect_answers : ℕ
  score : ℤ

/-- Calculates the score based on correct and incorrect answers -/
def calculate_score (correct : ℕ) (incorrect : ℕ) : ℤ :=
  (correct : ℤ) - 2 * (incorrect : ℤ)

/-- Theorem stating the number of correct answers given the conditions -/
theorem correct_answers_for_given_score
  (test : TestScore)
  (h1 : test.total_questions = 100)
  (h2 : test.correct_answers + test.incorrect_answers = test.total_questions)
  (h3 : test.score = calculate_score test.correct_answers test.incorrect_answers)
  (h4 : test.score = 76) :
  test.correct_answers = 92 := by
  sorry

end NUMINAMATH_CALUDE_correct_answers_for_given_score_l2897_289715
