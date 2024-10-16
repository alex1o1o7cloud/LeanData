import Mathlib

namespace NUMINAMATH_CALUDE_fraction_division_simplification_l3129_312958

theorem fraction_division_simplification (m : ℝ) (h1 : m ≠ 2) (h2 : m ≠ -3) :
  (m - 3) / (2 * m - 4) / (m + 2 - 5 / (m - 2)) = 1 / (2 * m + 6) := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_simplification_l3129_312958


namespace NUMINAMATH_CALUDE_train_crossing_time_l3129_312975

/-- Time taken for a train to cross a man running in the same direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 220 →
  train_speed = 80 * 1000 / 3600 →
  man_speed = 8 * 1000 / 3600 →
  (train_length / (train_speed - man_speed)) = 11 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3129_312975


namespace NUMINAMATH_CALUDE_lloyd_excess_rate_multiple_l3129_312956

/-- Calculates the multiple of regular rate for excess hours --/
def excessRateMultiple (regularHours : Float) (regularRate : Float) (totalHours : Float) (totalEarnings : Float) : Float :=
  let regularEarnings := regularHours * regularRate
  let excessHours := totalHours - regularHours
  let excessEarnings := totalEarnings - regularEarnings
  let excessRate := excessEarnings / excessHours
  excessRate / regularRate

/-- Proves that given Lloyd's work conditions, the multiple of his regular rate for excess hours is 2.5 --/
theorem lloyd_excess_rate_multiple :
  let regularHours : Float := 7.5
  let regularRate : Float := 4.5
  let totalHours : Float := 10.5
  let totalEarnings : Float := 67.5
  excessRateMultiple regularHours regularRate totalHours totalEarnings = 2.5 := by
  sorry

#eval excessRateMultiple 7.5 4.5 10.5 67.5

end NUMINAMATH_CALUDE_lloyd_excess_rate_multiple_l3129_312956


namespace NUMINAMATH_CALUDE_triangle_side_length_l3129_312977

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 3 → c = 4 → Real.cos C = -(1/4 : ℝ) →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  b = 7/2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3129_312977


namespace NUMINAMATH_CALUDE_episodes_per_season_l3129_312969

theorem episodes_per_season
  (series1_seasons series2_seasons : ℕ)
  (episodes_lost_per_season : ℕ)
  (remaining_episodes : ℕ)
  (h1 : series1_seasons = 12)
  (h2 : series2_seasons = 14)
  (h3 : episodes_lost_per_season = 2)
  (h4 : remaining_episodes = 364) :
  (remaining_episodes + episodes_lost_per_season * (series1_seasons + series2_seasons)) / (series1_seasons + series2_seasons) = 16 := by
sorry

end NUMINAMATH_CALUDE_episodes_per_season_l3129_312969


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l3129_312931

/-- Given an equation mx^2 - my^2 = n where m and n are real numbers and mn < 0,
    the curve represented by this equation is a hyperbola with foci on the y-axis. -/
theorem equation_represents_hyperbola (m n : ℝ) (h : m * n < 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), m * x^2 - m * y^2 = n ↔ y^2 / a^2 - x^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l3129_312931


namespace NUMINAMATH_CALUDE_largest_unreachable_proof_l3129_312950

/-- The largest integer that cannot be expressed as a non-negative linear combination of 17 and 11 -/
def largest_unreachable : ℕ := 159

/-- The width of the paper in half-inches -/
def paper_width : ℕ := 17

/-- The length of the paper in inches -/
def paper_length : ℕ := 11

/-- A predicate that checks if a natural number can be expressed as a non-negative linear combination of paper_width and paper_length -/
def is_reachable (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a * paper_width + b * paper_length

theorem largest_unreachable_proof :
  (∀ n > largest_unreachable, is_reachable n) ∧
  ¬is_reachable largest_unreachable :=
sorry

end NUMINAMATH_CALUDE_largest_unreachable_proof_l3129_312950


namespace NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_squared_l3129_312937

theorem ceiling_negative_seven_fourths_squared : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_squared_l3129_312937


namespace NUMINAMATH_CALUDE_investment_total_l3129_312970

theorem investment_total (rate1 rate2 amount1 amount2 total_income : ℚ)
  (h1 : rate1 = 85 / 1000)
  (h2 : rate2 = 64 / 1000)
  (h3 : amount1 = 3000)
  (h4 : amount2 = 5000)
  (h5 : rate1 * amount1 + rate2 * amount2 = 575) :
  amount1 + amount2 = 8000 :=
by sorry

end NUMINAMATH_CALUDE_investment_total_l3129_312970


namespace NUMINAMATH_CALUDE_game_winnable_iff_game_not_winnable_equal_game_winnable_greater_l3129_312981

/-- Represents a winning strategy for the card game -/
structure WinningStrategy (n k : ℕ) :=
  (moves : ℕ)
  (strategy : Unit)  -- Placeholder for the actual strategy

/-- The existence of a winning strategy for the card game -/
def winnable (n k : ℕ) : Prop :=
  ∃ (s : WinningStrategy n k), true

/-- Main theorem: The game is winnable if and only if n > k, given n ≥ k ≥ 2 -/
theorem game_winnable_iff (n k : ℕ) (h1 : n ≥ k) (h2 : k ≥ 2) :
  winnable n k ↔ n > k :=
sorry

/-- The game is not winnable when n = k -/
theorem game_not_winnable_equal (n : ℕ) (h : n ≥ 2) :
  ¬ winnable n n :=
sorry

/-- The game is winnable when n > k -/
theorem game_winnable_greater (n k : ℕ) (h1 : n > k) (h2 : k ≥ 2) :
  winnable n k :=
sorry

end NUMINAMATH_CALUDE_game_winnable_iff_game_not_winnable_equal_game_winnable_greater_l3129_312981


namespace NUMINAMATH_CALUDE_distance_center_to_line_l3129_312964

/-- The distance from the center of the unit circle to a line ax + by + c = 0, 
    where a^2 + b^2 ≠ 4c^2 and c ≠ 0, is 1/2. -/
theorem distance_center_to_line (a b c : ℝ) 
  (h1 : a^2 + b^2 ≠ 4 * c^2) (h2 : c ≠ 0) : 
  let d := |c| / Real.sqrt (a^2 + b^2)
  d = 1/2 := by sorry

end NUMINAMATH_CALUDE_distance_center_to_line_l3129_312964


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3129_312914

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- a_1, a_2, a_5 form a geometric sequence -/
def geometric_subseq (a : ℕ → ℝ) : Prop :=
  (a 2) ^ 2 = a 1 * a 5

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a) (h_geom : geometric_subseq a) : 
  a 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3129_312914


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l3129_312955

theorem smallest_solution_of_equation :
  ∀ x : ℚ, 3 * (8 * x^2 + 10 * x + 12) = x * (8 * x - 36) →
  x ≥ (-3 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l3129_312955


namespace NUMINAMATH_CALUDE_integral_sqrt_rational_l3129_312905

open Real MeasureTheory

/-- The definite integral of 5√(x+24) / ((x+24)^2 * √x) from x = 1 to x = 8 is equal to 1/8 -/
theorem integral_sqrt_rational : 
  ∫ x in (1 : ℝ)..8, (5 * Real.sqrt (x + 24)) / ((x + 24)^2 * Real.sqrt x) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_rational_l3129_312905


namespace NUMINAMATH_CALUDE_origin_symmetry_coordinates_l3129_312904

def point_symmetry (x y : ℝ) : ℝ × ℝ := (-x, -y)

theorem origin_symmetry_coordinates :
  let P : ℝ × ℝ := (-2, 3)
  let P1 : ℝ × ℝ := point_symmetry P.1 P.2
  P1 = (2, -3) := by sorry

end NUMINAMATH_CALUDE_origin_symmetry_coordinates_l3129_312904


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l3129_312913

theorem sqrt_twelve_minus_sqrt_three_equals_sqrt_three : 
  Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l3129_312913


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l3129_312951

theorem max_value_cos_sin : 
  ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5 ∧ 
  ∃ y : ℝ, 3 * Real.cos y + 4 * Real.sin y = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l3129_312951


namespace NUMINAMATH_CALUDE_power_of_two_six_l3129_312957

theorem power_of_two_six : 2^3 * 2^3 = 2^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_six_l3129_312957


namespace NUMINAMATH_CALUDE_fibonacci_parity_l3129_312936

def E : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | n + 3 => E (n + 2) + E (n + 1)

def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem fibonacci_parity : 
  isEven (E 2021) ∧ ¬isEven (E 2022) ∧ ¬isEven (E 2023) := by sorry

end NUMINAMATH_CALUDE_fibonacci_parity_l3129_312936


namespace NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l3129_312945

/-- A parabola with vertex at the origin and axis perpendicular to the x-axis -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = -2 * p * x

/-- A hyperbola with the standard form (x²/a²) - (y²/b²) = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : ℝ → ℝ → Prop := fun x y => x^2 / a^2 - y^2 / b^2 = 1

/-- The theorem stating the equations of the parabola and hyperbola -/
theorem parabola_hyperbola_intersection
  (para : Parabola) (hyp : Hyperbola)
  (axis_passes_focus : para.eq 1 0) -- Parabola's axis passes through (1,0)
  (intersection : para.eq (-3/2) (Real.sqrt 6) ∧ hyp.eq (-3/2) (Real.sqrt 6)) :
  (para.p = 2 ∧ para.eq = fun x y => y^2 = -4*x) ∧
  (hyp.a^2 = 1/4 ∧ hyp.b^2 = 3/4 ∧ hyp.eq = fun x y => 4*x^2 - (4/3)*y^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l3129_312945


namespace NUMINAMATH_CALUDE_unique_solution_fifth_root_equation_l3129_312921

theorem unique_solution_fifth_root_equation (x : ℝ) :
  (((x^3 + 2*x)^(1/5) = (x^5 - 2*x)^(1/3)) ↔ (x = 0)) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_fifth_root_equation_l3129_312921


namespace NUMINAMATH_CALUDE_hotel_visit_permutations_l3129_312976

def number_of_permutations (n : ℕ) : ℕ := Nat.factorial n

def constrained_permutations (n : ℕ) : ℕ :=
  number_of_permutations n / 4

theorem hotel_visit_permutations :
  constrained_permutations 5 = 30 := by sorry

end NUMINAMATH_CALUDE_hotel_visit_permutations_l3129_312976


namespace NUMINAMATH_CALUDE_min_dot_product_of_tangents_l3129_312952

-- Define a circle with radius 1
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define a point outside the circle
def PointOutside (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 > 1

-- Define tangent points
def TangentPoints (p a b : ℝ × ℝ) : Prop :=
  a ∈ Circle ∧ b ∈ Circle ∧
  ((p.1 - a.1) * a.1 + (p.2 - a.2) * a.2 = 0) ∧
  ((p.1 - b.1) * b.1 + (p.2 - b.2) * b.2 = 0)

-- Define dot product of vectors
def DotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem min_dot_product_of_tangents :
  ∀ p : ℝ × ℝ, PointOutside p →
  ∀ a b : ℝ × ℝ, TangentPoints p a b →
  ∃ m : ℝ, m = -3 + 2 * Real.sqrt 2 ∧
  ∀ x y : ℝ × ℝ, TangentPoints p x y →
  DotProduct (x.1 - p.1, x.2 - p.2) (y.1 - p.1, y.2 - p.2) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_of_tangents_l3129_312952


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3129_312961

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3129_312961


namespace NUMINAMATH_CALUDE_inequality_proof_l3129_312930

theorem inequality_proof (x y : ℝ) 
  (h : x^2 + x*y + y^2 = (x + y)^2 - x*y ∧ 
       x^2 + x*y + y^2 = (x + y - Real.sqrt (x*y)) * (x + y + Real.sqrt (x*y))) : 
  x + y + Real.sqrt (x*y) ≤ 3*(x + y - Real.sqrt (x*y)) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3129_312930


namespace NUMINAMATH_CALUDE_peanuts_added_l3129_312968

theorem peanuts_added (initial_peanuts final_peanuts : ℕ) 
  (h1 : initial_peanuts = 10)
  (h2 : final_peanuts = 18) :
  final_peanuts - initial_peanuts = 8 := by
sorry

end NUMINAMATH_CALUDE_peanuts_added_l3129_312968


namespace NUMINAMATH_CALUDE_gold_distribution_theorem_l3129_312926

/-- The number of gold nuggets -/
def n : ℕ := 2020

/-- The sum of masses of all nuggets -/
def total_mass : ℕ := n * (n + 1) / 2

/-- The maximum difference in mass between the two chests -/
def max_diff : ℕ := n

/-- The guaranteed amount of gold in the heavier chest -/
def guaranteed_mass : ℕ := total_mass / 2 + max_diff / 2

theorem gold_distribution_theorem :
  ∃ (chest_mass : ℕ), chest_mass ≥ guaranteed_mass ∧ 
  chest_mass ≤ total_mass - (total_mass / 2 - max_diff / 2) :=
sorry

end NUMINAMATH_CALUDE_gold_distribution_theorem_l3129_312926


namespace NUMINAMATH_CALUDE_bush_spacing_l3129_312989

theorem bush_spacing (yard_side_length : ℕ) (num_sides : ℕ) (num_bushes : ℕ) :
  yard_side_length = 16 →
  num_sides = 3 →
  num_bushes = 12 →
  (yard_side_length * num_sides) / num_bushes = 4 := by
  sorry

end NUMINAMATH_CALUDE_bush_spacing_l3129_312989


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3129_312954

theorem triangle_perimeter (a b c : ℝ) (ha : a = 28) (hb : b = 16) (hc : c = 18) :
  a + b + c = 62 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3129_312954


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3129_312994

theorem election_winner_percentage :
  ∀ (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ),
    winner_votes = 806 →
    margin = 312 →
    total_votes = winner_votes + (winner_votes - margin) →
    (winner_votes : ℚ) / (total_votes : ℚ) = 62 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3129_312994


namespace NUMINAMATH_CALUDE_range_of_f_on_interval_l3129_312992

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem range_of_f_on_interval :
  ∃ (y : ℝ), y ∈ Set.Icc 0 (Real.exp (Real.pi / 2)) ↔
  ∃ (x : ℝ), x ∈ Set.Icc 0 Real.pi ∧ f x = y :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_on_interval_l3129_312992


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3129_312983

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 1) :
  z.im = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3129_312983


namespace NUMINAMATH_CALUDE_cyclic_ratio_inequality_l3129_312915

theorem cyclic_ratio_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  b^2 / a + c^2 / b + a^2 / c ≥ Real.sqrt (3 * (a^2 + b^2 + c^2)) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_ratio_inequality_l3129_312915


namespace NUMINAMATH_CALUDE_arrange_balls_count_l3129_312911

/-- The number of ways to arrange balls of different colors in a row -/
def arrangeColoredBalls (red : ℕ) (yellow : ℕ) (white : ℕ) : ℕ :=
  Nat.choose (red + yellow + white) white *
  Nat.choose (red + yellow) red *
  Nat.choose yellow yellow

/-- Theorem stating that arranging 2 red, 3 yellow, and 4 white balls results in 1260 arrangements -/
theorem arrange_balls_count : arrangeColoredBalls 2 3 4 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_arrange_balls_count_l3129_312911


namespace NUMINAMATH_CALUDE_max_odd_numbers_in_even_product_l3129_312963

theorem max_odd_numbers_in_even_product (numbers : Finset ℕ) :
  numbers.card = 7 →
  (numbers.prod (fun x ↦ x)) % 2 = 0 →
  (numbers.filter (fun x ↦ x % 2 = 1)).card ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_odd_numbers_in_even_product_l3129_312963


namespace NUMINAMATH_CALUDE_bananas_bought_l3129_312993

theorem bananas_bought (initial : ℕ) (eaten : ℕ) (remaining : ℕ) : 
  eaten = 1 → remaining = 11 → initial = eaten + remaining := by sorry

end NUMINAMATH_CALUDE_bananas_bought_l3129_312993


namespace NUMINAMATH_CALUDE_expense_ratios_l3129_312939

def initial_amount : ℚ := 120
def books_expense : ℚ := 25
def clothes_expense : ℚ := 40
def snacks_expense : ℚ := 10

def total_spent : ℚ := books_expense + clothes_expense + snacks_expense

theorem expense_ratios :
  (books_expense / total_spent = 1 / 3) ∧
  (clothes_expense / total_spent = 4 / 3) ∧
  (snacks_expense / total_spent = 2 / 15) := by
  sorry

end NUMINAMATH_CALUDE_expense_ratios_l3129_312939


namespace NUMINAMATH_CALUDE_min_break_even_quantity_l3129_312902

/-- The cost function for a product -/
def cost (x : ℕ) : ℝ := 3000 + 20 * x - 0.1 * x^2

/-- The revenue function for a product -/
def revenue (x : ℕ) : ℝ := 25 * x

/-- The break-even condition -/
def breaks_even (x : ℕ) : Prop := revenue x ≥ cost x

theorem min_break_even_quantity :
  ∃ (x : ℕ), x > 0 ∧ x < 240 ∧ breaks_even x ∧
  ∀ (y : ℕ), y > 0 ∧ y < 240 ∧ breaks_even y → y ≥ 150 :=
sorry

end NUMINAMATH_CALUDE_min_break_even_quantity_l3129_312902


namespace NUMINAMATH_CALUDE_club_M_members_eq_five_l3129_312912

/-- The number of people who joined club M in a company with the following conditions:
  - There are 60 people in total
  - There are 3 clubs: M, S, and Z
  - 18 people joined S
  - 11 people joined Z
  - Members of M did not join any other club
  - At most 26 people did not join any club
-/
def club_M_members : ℕ := by
  -- Define the total number of people
  let total_people : ℕ := 60
  -- Define the number of people in club S
  let club_S_members : ℕ := 18
  -- Define the number of people in club Z
  let club_Z_members : ℕ := 11
  -- Define the maximum number of people who didn't join any club
  let max_no_club : ℕ := 26
  
  -- The actual proof would go here
  sorry

theorem club_M_members_eq_five : club_M_members = 5 := by
  sorry

end NUMINAMATH_CALUDE_club_M_members_eq_five_l3129_312912


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3129_312999

def repeating_decimal_12 : ℚ := 12 / 99
def repeating_decimal_003 : ℚ := 3 / 999
def repeating_decimal_00005 : ℚ := 5 / 99999

theorem sum_of_repeating_decimals :
  repeating_decimal_12 + repeating_decimal_003 + repeating_decimal_00005 = 124215 / 999999 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3129_312999


namespace NUMINAMATH_CALUDE_quadratic_shift_theorem_l3129_312928

/-- The quadratic function y = x^2 - 4x - 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x - 3

/-- The shifted quadratic function y = x^2 - 4x - 3 + a -/
def f_shifted (x a : ℝ) : ℝ := f x + a

theorem quadratic_shift_theorem :
  /- The value of a that makes the parabola pass through (0,1) is 4 -/
  (∃ a : ℝ, f_shifted 0 a = 1 ∧ a = 4) ∧
  /- The values of a that make the parabola intersect the coordinate axes at exactly 2 points are 3 and 7 -/
  (∃ a₁ a₂ : ℝ, 
    ((f_shifted 0 a₁ = 0 ∨ (∃ x : ℝ, x ≠ 0 ∧ f_shifted x a₁ = 0)) ∧
     (∃! x : ℝ, f_shifted x a₁ = 0)) ∧
    ((f_shifted 0 a₂ = 0 ∨ (∃ x : ℝ, x ≠ 0 ∧ f_shifted x a₂ = 0)) ∧
     (∃! x : ℝ, f_shifted x a₂ = 0)) ∧
    a₁ = 3 ∧ a₂ = 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_shift_theorem_l3129_312928


namespace NUMINAMATH_CALUDE_siblings_total_weight_l3129_312986

def total_weight (weight1 weight2 backpack1 backpack2 : ℕ) : ℕ :=
  weight1 + weight2 + backpack1 + backpack2

theorem siblings_total_weight :
  ∀ (antonio_weight antonio_sister_weight : ℕ),
    antonio_weight = 50 →
    antonio_sister_weight = antonio_weight - 12 →
    total_weight antonio_weight antonio_sister_weight 5 3 = 96 := by
  sorry

end NUMINAMATH_CALUDE_siblings_total_weight_l3129_312986


namespace NUMINAMATH_CALUDE_sector_area_l3129_312974

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (area : ℝ) : 
  arc_length = 3 → central_angle = 1 → area = (arc_length * arc_length) / (2 * central_angle) → area = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3129_312974


namespace NUMINAMATH_CALUDE_complex_calculation_l3129_312944

theorem complex_calculation (a b : ℂ) (ha : a = 3 + 2*Complex.I) (hb : b = 2 - 2*Complex.I) :
  3*a - 4*b = 1 + 14*Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_calculation_l3129_312944


namespace NUMINAMATH_CALUDE_problem_solution_l3129_312920

theorem problem_solution : ∃ x : ℝ, (6000 - (x / 21)) = 5995 ∧ x = 105 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3129_312920


namespace NUMINAMATH_CALUDE_laptop_price_l3129_312972

/-- Given that 20% of a price is $240, prove that the full price is $1200 -/
theorem laptop_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (full_price : ℝ) 
  (h1 : upfront_payment = 240)
  (h2 : upfront_percentage = 20)
  (h3 : upfront_payment = upfront_percentage / 100 * full_price) : 
  full_price = 1200 := by
  sorry

#check laptop_price

end NUMINAMATH_CALUDE_laptop_price_l3129_312972


namespace NUMINAMATH_CALUDE_min_value_of_function_lower_bound_achievable_l3129_312959

theorem min_value_of_function (x : ℝ) (hx : x > 0) :
  (3 + x + x^2) / (1 + x) ≥ -1 + 2 * Real.sqrt 3 :=
sorry

theorem lower_bound_achievable :
  ∃ x > 0, (3 + x + x^2) / (1 + x) = -1 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_lower_bound_achievable_l3129_312959


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l3129_312935

-- Define the motion equation
def s (t : ℝ) : ℝ := 3 + t^2

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_2 :
  v 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l3129_312935


namespace NUMINAMATH_CALUDE_f_of_5_eq_110_l3129_312900

/-- The polynomial function f(x) = 3x^4 - 20x^3 + 38x^2 - 35x - 40 -/
def f (x : ℝ) : ℝ := 3 * x^4 - 20 * x^3 + 38 * x^2 - 35 * x - 40

/-- Theorem: f(5) = 110 -/
theorem f_of_5_eq_110 : f 5 = 110 := by
  sorry

end NUMINAMATH_CALUDE_f_of_5_eq_110_l3129_312900


namespace NUMINAMATH_CALUDE_power_of_power_l3129_312990

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3129_312990


namespace NUMINAMATH_CALUDE_square_condition_implies_value_l3129_312947

theorem square_condition_implies_value (k : ℕ) :
  (∃ m n : ℕ, (4 * k + 5 = m^2) ∧ (9 * k + 4 = n^2)) →
  (7 * k + 4 = 39) := by
sorry

end NUMINAMATH_CALUDE_square_condition_implies_value_l3129_312947


namespace NUMINAMATH_CALUDE_CaCl2_formation_theorem_l3129_312946

/-- Represents a chemical reaction equation --/
structure ReactionEquation :=
  (reactants : List (String × ℕ))
  (products : List (String × ℕ))

/-- Represents the available moles of reactants --/
structure AvailableReactants :=
  (HCl : ℝ)
  (CaCO3 : ℝ)

/-- Calculates the moles of CaCl2 formed in the reaction --/
def molesOfCaCl2Formed (equation : ReactionEquation) (available : AvailableReactants) : ℝ :=
  sorry

/-- The main theorem stating that 3 moles of CaCl2 are formed --/
theorem CaCl2_formation_theorem (equation : ReactionEquation) (available : AvailableReactants) :
  equation.reactants = [("CaCO3", 1), ("HCl", 2)] ∧
  equation.products = [("CaCl2", 1), ("CO2", 1), ("H2O", 1)] ∧
  available.HCl = 6 ∧
  available.CaCO3 = 3 →
  molesOfCaCl2Formed equation available = 3 :=
by sorry

end NUMINAMATH_CALUDE_CaCl2_formation_theorem_l3129_312946


namespace NUMINAMATH_CALUDE_two_color_no_power_of_two_sum_l3129_312916

theorem two_color_no_power_of_two_sum :
  ∃ (f : ℕ → Bool), ∀ (a b : ℕ), a ≠ b → f a = f b → ¬∃ (n : ℕ), a + b = 2^n :=
sorry

end NUMINAMATH_CALUDE_two_color_no_power_of_two_sum_l3129_312916


namespace NUMINAMATH_CALUDE_sixth_row_third_number_l3129_312910

/-- Represents the sequence of positive odd numbers -/
def oddSequence (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of elements in the nth row of the table -/
def rowSize (n : ℕ) : ℕ := 2^(n - 1)

/-- Represents the total number of elements up to and including the nth row -/
def totalElements (n : ℕ) : ℕ := 2^n - 1

theorem sixth_row_third_number : 
  let rowNumber := 6
  let positionInRow := 3
  oddSequence (totalElements (rowNumber - 1) + positionInRow) = 67 := by
  sorry

end NUMINAMATH_CALUDE_sixth_row_third_number_l3129_312910


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3129_312967

/-- The speed of a boat in still water, given downstream travel information and current speed. -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (travel_time_minutes : ℝ) :
  current_speed = 8 →
  downstream_distance = 36.67 →
  travel_time_minutes = 44 →
  ∃ (boat_speed : ℝ), boat_speed = 42 ∧ 
    (boat_speed + current_speed) * (travel_time_minutes / 60) = downstream_distance :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3129_312967


namespace NUMINAMATH_CALUDE_problem_solution_l3129_312917

theorem problem_solution (x : ℝ) (f : ℝ → ℝ) 
  (h1 : x > 0) 
  (h2 : x + 17 = 60 * f x) 
  (h3 : x = 3) : 
  f 3 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3129_312917


namespace NUMINAMATH_CALUDE_joans_kittens_l3129_312906

/-- Given that Joan initially had 8 kittens and received 2 more from her friends,
    prove that she now has 10 kittens in total. -/
theorem joans_kittens (initial : Nat) (received : Nat) (total : Nat) : 
  initial = 8 → received = 2 → total = initial + received → total = 10 := by
sorry

end NUMINAMATH_CALUDE_joans_kittens_l3129_312906


namespace NUMINAMATH_CALUDE_sequence_inequality_l3129_312982

theorem sequence_inequality (a : ℕ → ℕ) 
  (h0 : ∀ n, a n > 0)
  (h1 : a 1 > a 0)
  (h2 : ∀ k, 2 ≤ k → k ≤ 100 → a k = 3 * a (k-1) - 2 * a (k-2)) :
  a 100 > 2^99 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3129_312982


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3129_312953

theorem polynomial_factorization (x : ℝ) :
  6 * (x + 4) * (x + 7) * (x + 9) * (x + 11) - 5 * x^2 =
  (3 * x^2 + 93 * x) * (2 * x^2 + 178 * x + 5432) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3129_312953


namespace NUMINAMATH_CALUDE_geometric_progression_sum_180_l3129_312978

theorem geometric_progression_sum_180 :
  ∃ (a b c d : ℝ) (e f g h : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0 ∧
    a + b + c + d = 180 ∧
    b / a = c / b ∧ c / b = d / c ∧
    c = a + 36 ∧
    e + f + g + h = 180 ∧
    f / e = g / f ∧ g / f = h / g ∧
    g = e + 36 ∧
    ((a = 9/2 ∧ b = 27/2 ∧ c = 81/2 ∧ d = 243/2) ∨
     (a = 12 ∧ b = 24 ∧ c = 48 ∧ d = 96)) ∧
    ((e = 9/2 ∧ f = 27/2 ∧ g = 81/2 ∧ h = 243/2) ∨
     (e = 12 ∧ f = 24 ∧ g = 48 ∧ h = 96)) ∧
    (a ≠ e ∨ b ≠ f ∨ c ≠ g ∨ d ≠ h) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_180_l3129_312978


namespace NUMINAMATH_CALUDE_dog_grouping_theorem_l3129_312927

/-- The number of ways to divide 12 dogs into groups of 4, 6, and 2,
    with Fluffy in the 4-dog group and Nipper in the 6-dog group -/
def dog_grouping_ways : ℕ :=
  let total_dogs : ℕ := 12
  let group1_size : ℕ := 4
  let group2_size : ℕ := 6
  let group3_size : ℕ := 2
  let remaining_dogs : ℕ := total_dogs - 2  -- Fluffy and Nipper are already placed
  Nat.choose remaining_dogs (group1_size - 1) * Nat.choose (remaining_dogs - (group1_size - 1)) (group2_size - 1)

theorem dog_grouping_theorem : dog_grouping_ways = 2520 := by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_theorem_l3129_312927


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l3129_312949

theorem modulus_of_complex_number (z : ℂ) : z = 3 - 2*I → Complex.abs z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l3129_312949


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3129_312942

theorem complex_equation_solution (a : ℝ) : 
  Complex.abs (a - 2 + (4 + 3 * Complex.I) / (1 + 2 * Complex.I)) = Real.sqrt 3 * a → 
  a = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3129_312942


namespace NUMINAMATH_CALUDE_missing_village_population_l3129_312925

def village_populations : List ℕ := [803, 900, 1100, 945, 980, 1249]

theorem missing_village_population 
  (total_villages : ℕ) 
  (average_population : ℕ) 
  (known_populations : List ℕ) 
  (h1 : total_villages = 7)
  (h2 : average_population = 1000)
  (h3 : known_populations = village_populations)
  (h4 : known_populations.length = 6) :
  ∃ (missing_population : ℕ), 
    missing_population = total_villages * average_population - known_populations.sum ∧
    missing_population = 1023 :=
by sorry

end NUMINAMATH_CALUDE_missing_village_population_l3129_312925


namespace NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l3129_312997

/-- 
Given:
- Joel is currently 5 years old
- Joel's dad is currently 32 years old

Prove that Joel will be 27 years old when his dad is twice as old as him.
-/
theorem joel_age_when_dad_twice_as_old (joel_current_age : ℕ) (dad_current_age : ℕ) :
  joel_current_age = 5 →
  dad_current_age = 32 →
  ∃ (future_joel_age : ℕ), 
    future_joel_age + joel_current_age = dad_current_age ∧
    2 * future_joel_age = future_joel_age + dad_current_age ∧
    future_joel_age = 27 :=
by sorry

end NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l3129_312997


namespace NUMINAMATH_CALUDE_fortieth_term_is_81_l3129_312971

/-- An arithmetic sequence starting from 3 with common difference 2 -/
def arithmeticSequence (n : ℕ) : ℕ := 3 + 2 * (n - 1)

/-- The 40th term of the arithmetic sequence is 81 -/
theorem fortieth_term_is_81 : arithmeticSequence 40 = 81 := by
  sorry

end NUMINAMATH_CALUDE_fortieth_term_is_81_l3129_312971


namespace NUMINAMATH_CALUDE_grade_assignment_count_l3129_312938

/-- The number of students in the class -/
def num_students : ℕ := 12

/-- The number of different grades -/
def num_grades : ℕ := 4

/-- Theorem: The number of ways to assign grades to students -/
theorem grade_assignment_count : (num_grades : ℕ) ^ num_students = 16777216 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l3129_312938


namespace NUMINAMATH_CALUDE_halloween_candy_l3129_312991

theorem halloween_candy (debby_candy : ℕ) (sister_candy : ℕ) (remaining_candy : ℕ) :
  debby_candy = 32 →
  sister_candy = 42 →
  remaining_candy = 39 →
  debby_candy + sister_candy - remaining_candy = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_l3129_312991


namespace NUMINAMATH_CALUDE_fruit_basket_problem_l3129_312979

theorem fruit_basket_problem (total_fruit : ℕ) 
  (jacques_apples jacques_pears gillian_apples gillian_pears : ℕ) : 
  total_fruit = 25 →
  jacques_apples = 1 →
  jacques_pears = 3 →
  gillian_apples = 3 →
  gillian_pears = 2 →
  ∃ (initial_apples initial_pears : ℕ),
    initial_apples + initial_pears = total_fruit ∧
    initial_apples - jacques_apples - gillian_apples = 
      initial_pears - jacques_pears - gillian_pears →
  initial_pears = 13 := by
sorry

end NUMINAMATH_CALUDE_fruit_basket_problem_l3129_312979


namespace NUMINAMATH_CALUDE_train_length_calculation_l3129_312940

/-- The length of each train in kilometers -/
def train_length : ℝ := 0.06

/-- The speed of the faster train in km/hr -/
def fast_train_speed : ℝ := 48

/-- The speed of the slower train in km/hr -/
def slow_train_speed : ℝ := 36

/-- The time taken for the faster train to pass the slower train in seconds -/
def passing_time : ℝ := 36

theorem train_length_calculation :
  let relative_speed := fast_train_speed - slow_train_speed
  let relative_speed_km_per_sec := relative_speed / 3600
  2 * train_length = relative_speed_km_per_sec * passing_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3129_312940


namespace NUMINAMATH_CALUDE_inverse_of_10_mod_1729_l3129_312907

theorem inverse_of_10_mod_1729 : ∃ x : ℕ, x ≤ 1728 ∧ (10 * x) % 1729 = 1 :=
by
  use 1537
  sorry

end NUMINAMATH_CALUDE_inverse_of_10_mod_1729_l3129_312907


namespace NUMINAMATH_CALUDE_crown_cost_l3129_312966

/-- Given a total payment of $22,000 for a crown including a 10% tip,
    prove that the original cost of the crown was $20,000. -/
theorem crown_cost (total_payment : ℝ) (tip_percentage : ℝ) (h1 : total_payment = 22000)
    (h2 : tip_percentage = 0.1) : 
  ∃ (original_cost : ℝ), 
    original_cost * (1 + tip_percentage) = total_payment ∧ 
    original_cost = 20000 := by
  sorry

end NUMINAMATH_CALUDE_crown_cost_l3129_312966


namespace NUMINAMATH_CALUDE_equal_elevation_angles_l3129_312919

/-- Given two flagpoles of heights h and k, separated by 2a units on a horizontal plane,
    this theorem characterizes the set of points where the angles of elevation to the tops
    of the poles are equal. -/
theorem equal_elevation_angles
  (h k a : ℝ) (h_pos : h > 0) (k_pos : k > 0) (a_pos : a > 0) :
  let P : ℝ × ℝ → Prop := λ (x, y) ↦ 
    h / Real.sqrt ((x + a)^2 + y^2) = k / Real.sqrt ((x - a)^2 + y^2)
  (h = k → ∀ y, P (0, y)) ∧ 
  (h ≠ k → ∃ c r, ∀ x y, P (x, y) ↔ (x - c)^2 + y^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_equal_elevation_angles_l3129_312919


namespace NUMINAMATH_CALUDE_highest_red_probability_l3129_312998

/-- Represents the contents of a bag --/
structure Bag where
  red : ℕ
  white : ℕ

/-- The probability of drawing a red ball from a bag --/
def redProbability (bag : Bag) : ℚ :=
  bag.red / (bag.red + bag.white)

/-- The average probability of drawing a red ball from two bags --/
def averageProbability (bag1 bag2 : Bag) : ℚ :=
  (redProbability bag1 + redProbability bag2) / 2

/-- The theorem stating the highest probability of drawing a red ball --/
theorem highest_red_probability :
  ∃ (bag1 bag2 : Bag),
    bag1.red + bag2.red = 5 ∧
    bag1.white + bag2.white = 12 ∧
    bag1.red + bag1.white > 0 ∧
    bag2.red + bag2.white > 0 ∧
    averageProbability bag1 bag2 = 5/8 ∧
    ∀ (other1 other2 : Bag),
      other1.red + other2.red = 5 →
      other1.white + other2.white = 12 →
      other1.red + other1.white > 0 →
      other2.red + other2.white > 0 →
      averageProbability other1 other2 ≤ 5/8 :=
by sorry

end NUMINAMATH_CALUDE_highest_red_probability_l3129_312998


namespace NUMINAMATH_CALUDE_simplify_expression_l3129_312965

theorem simplify_expression : (5 + 4 + 6) / 3 - 2 / 3 = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3129_312965


namespace NUMINAMATH_CALUDE_marble_prism_weight_l3129_312933

/-- Calculates the weight of a rectangular prism with a square base -/
def weight_rectangular_prism (height : ℝ) (base_side : ℝ) (density : ℝ) : ℝ :=
  height * base_side * base_side * density

/-- Proves that the weight of the given marble rectangular prism is 86400 kg -/
theorem marble_prism_weight :
  weight_rectangular_prism 8 2 2700 = 86400 := by
  sorry

end NUMINAMATH_CALUDE_marble_prism_weight_l3129_312933


namespace NUMINAMATH_CALUDE_arithmetic_seq_sum_l3129_312985

-- Define an arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_seq_sum (a : ℕ → ℝ) :
  is_arithmetic_seq a →
  a 4 + a 6 + a 8 + a 10 + a 12 = 120 →
  a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_sum_l3129_312985


namespace NUMINAMATH_CALUDE_min_points_for_proximity_l3129_312924

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define the distance function between two points on the circle
def circleDistance (p q : Circle) : ℝ := sorry

-- Define the sequence of points
def circlePoints : ℕ → Circle := sorry

-- Theorem statement
theorem min_points_for_proximity :
  ∀ n : ℕ, n < 20 →
  ∃ i j : ℕ, i < j ∧ j < n ∧ circleDistance (circlePoints i) (circlePoints j) ≥ 1/5 :=
sorry

end NUMINAMATH_CALUDE_min_points_for_proximity_l3129_312924


namespace NUMINAMATH_CALUDE_max_value_theorem_l3129_312929

/-- A function that checks if three numbers can form a triangle with non-zero area -/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if a set of eleven consecutive integers contains a triangle-forming trio -/
def has_triangle_trio (start : ℕ) : Prop :=
  ∃ i j k, i < j ∧ j < k ∧ k < start + 11 ∧ can_form_triangle (start + i) (start + j) (start + k)

/-- The theorem stating that 499 is the maximum value satisfying the condition -/
theorem max_value_theorem : 
  (∀ start : ℕ, 5 ≤ start ∧ start ≤ 489 → has_triangle_trio start) ∧
  ¬(∀ start : ℕ, 5 ≤ start ∧ start ≤ 490 → has_triangle_trio start) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3129_312929


namespace NUMINAMATH_CALUDE_down_payment_calculation_l3129_312948

/-- Given a loan with the following conditions:
  * The loan has 0% interest
  * The loan is to be paid back in 5 years
  * Monthly payments are $600.00
  * The total loan amount (including down payment) is $46,000
  This theorem proves that the down payment is $10,000 -/
theorem down_payment_calculation (loan_amount : ℝ) (years : ℕ) (monthly_payment : ℝ) :
  loan_amount = 46000 ∧ 
  years = 5 ∧ 
  monthly_payment = 600 →
  loan_amount - (years * 12 : ℝ) * monthly_payment = 10000 :=
by sorry

end NUMINAMATH_CALUDE_down_payment_calculation_l3129_312948


namespace NUMINAMATH_CALUDE_duck_cow_leg_count_l3129_312903

theorem duck_cow_leg_count :
  ∀ (num_ducks : ℕ),
  let num_cows : ℕ := 12
  let total_heads : ℕ := num_ducks + num_cows
  let total_legs : ℕ := 2 * num_ducks + 4 * num_cows
  total_legs - 2 * total_heads = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_duck_cow_leg_count_l3129_312903


namespace NUMINAMATH_CALUDE_no_integer_solution_l3129_312984

theorem no_integer_solution :
  ¬ ∃ (x y : ℤ), 19 * x^2 - 76 * y^2 = 1976 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3129_312984


namespace NUMINAMATH_CALUDE_parallel_vectors_difference_magnitude_l3129_312995

/-- Given two vectors a and b in ℝ², where a is parallel to b, 
    prove that the magnitude of their difference is 2√5 -/
theorem parallel_vectors_difference_magnitude 
  (a b : ℝ × ℝ) 
  (h_a : a = (1, 2)) 
  (h_b : b.2 = 6) 
  (h_parallel : ∃ (k : ℝ), a = k • b) : 
  ‖a - b‖ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_difference_magnitude_l3129_312995


namespace NUMINAMATH_CALUDE_three_number_problem_l3129_312901

theorem three_number_problem (a b c : ℚ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 6 * (b + c))
  (second_eq : b = 9 * c) :
  a - c = 177 / 7 := by
sorry

end NUMINAMATH_CALUDE_three_number_problem_l3129_312901


namespace NUMINAMATH_CALUDE_total_cost_after_rebate_l3129_312934

def polo_shirt_price : ℕ := 26
def polo_shirt_quantity : ℕ := 3
def necklace_price : ℕ := 83
def necklace_quantity : ℕ := 2
def computer_game_price : ℕ := 90
def computer_game_quantity : ℕ := 1
def rebate : ℕ := 12

theorem total_cost_after_rebate :
  (polo_shirt_price * polo_shirt_quantity +
   necklace_price * necklace_quantity +
   computer_game_price * computer_game_quantity) - rebate = 322 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_after_rebate_l3129_312934


namespace NUMINAMATH_CALUDE_factorial_plus_one_divisible_implies_prime_l3129_312973

theorem factorial_plus_one_divisible_implies_prime (n : ℕ) :
  (n! + 1) % (n + 1) = 0 → Nat.Prime (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorial_plus_one_divisible_implies_prime_l3129_312973


namespace NUMINAMATH_CALUDE_total_vehicles_on_highway_l3129_312922

theorem total_vehicles_on_highway : 
  ∀ (num_trucks : ℕ) (num_cars : ℕ),
  num_trucks = 100 →
  num_cars = 2 * num_trucks →
  num_cars + num_trucks = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_total_vehicles_on_highway_l3129_312922


namespace NUMINAMATH_CALUDE_stating_consecutive_sum_equals_odd_divisors_l3129_312980

/-- 
Given a positive integer n, count_consecutive_sum n returns the number of ways
n can be represented as a sum of one or more consecutive positive integers.
-/
def count_consecutive_sum (n : ℕ+) : ℕ := sorry

/-- 
Given a positive integer n, count_odd_divisors n returns the number of odd
divisors of n.
-/
def count_odd_divisors (n : ℕ+) : ℕ := sorry

/-- 
Theorem stating that for any positive integer n, the number of ways n can be
represented as a sum of one or more consecutive positive integers is equal to
the number of odd divisors of n.
-/
theorem consecutive_sum_equals_odd_divisors (n : ℕ+) :
  count_consecutive_sum n = count_odd_divisors n := by sorry

end NUMINAMATH_CALUDE_stating_consecutive_sum_equals_odd_divisors_l3129_312980


namespace NUMINAMATH_CALUDE_point_on_or_outside_circle_l3129_312962

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}

-- Define the point P as a function of a
def P (a : ℝ) : ℝ × ℝ := (a, 2 - a)

-- Theorem statement
theorem point_on_or_outside_circle : 
  ∀ a : ℝ, (P a) ∈ C ∨ (P a) ∉ interior C :=
sorry

end NUMINAMATH_CALUDE_point_on_or_outside_circle_l3129_312962


namespace NUMINAMATH_CALUDE_evaluate_expression_l3129_312909

theorem evaluate_expression : (-2 : ℤ) ^ (4^2) + 2^(4^2) = 2^17 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3129_312909


namespace NUMINAMATH_CALUDE_expression_value_l3129_312923

theorem expression_value (a b : ℝ) (h : a + 3*b = 4) : 2*a + 6*b - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3129_312923


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l3129_312918

theorem right_triangle_side_length 
  (Q R S : ℝ × ℝ) 
  (right_angle_Q : (R.1 - Q.1) * (S.1 - Q.1) + (R.2 - Q.2) * (S.2 - Q.2) = 0) 
  (cos_R : (Q.1 - R.1) / Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) = 5/13) 
  (RS_length : Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 13) : 
  Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) = 12 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l3129_312918


namespace NUMINAMATH_CALUDE_constant_term_of_expansion_l3129_312943

theorem constant_term_of_expansion (x : ℝ) (x_pos : x > 0) : 
  ∃ (c : ℝ), c = 240 ∧ 
  ∀ (k : ℕ), k ≤ 6 → 
    (Nat.choose 6 k * (2^k) * x^(6 - 3/2 * k : ℝ) = c ↔ k = 4) :=
sorry

end NUMINAMATH_CALUDE_constant_term_of_expansion_l3129_312943


namespace NUMINAMATH_CALUDE_smallest_m_for_nth_root_in_T_l3129_312960

def T : Set ℂ := {z | ∃ x y : ℝ, z = x + y * Complex.I ∧ 1/2 ≤ x ∧ x ≤ Real.sqrt 2 / 2}

theorem smallest_m_for_nth_root_in_T : 
  (∀ n : ℕ+, n ≥ 12 → ∃ z ∈ T, z ^ (n : ℕ) = 1) ∧ 
  (∀ m : ℕ+, m < 12 → ∃ n : ℕ+, n ≥ m ∧ ∀ z ∈ T, z ^ (n : ℕ) ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_nth_root_in_T_l3129_312960


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l3129_312996

theorem sum_of_fifth_powers (a b u v : ℝ) 
  (h1 : a * u + b * v = 5)
  (h2 : a * u^2 + b * v^2 = 11)
  (h3 : a * u^3 + b * v^3 = 30)
  (h4 : a * u^4 + b * v^4 = 76) :
  a * u^5 + b * v^5 = 8264 / 319 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l3129_312996


namespace NUMINAMATH_CALUDE_strategies_conversion_l3129_312987

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8^i)) 0

/-- The number of strategies in base 8 -/
def strategies_base8 : List Nat := [2, 3, 4]

theorem strategies_conversion :
  base8_to_base10 strategies_base8 = 282 := by
  sorry

end NUMINAMATH_CALUDE_strategies_conversion_l3129_312987


namespace NUMINAMATH_CALUDE_driver_distance_theorem_l3129_312932

/-- Calculates the total distance traveled by a driver given their speed and driving durations. -/
def total_distance_traveled (speed : ℝ) (first_duration second_duration : ℝ) : ℝ :=
  speed * (first_duration + second_duration)

/-- Theorem stating that a driver traveling at 60 mph for 4 hours and 9 hours will cover 780 miles. -/
theorem driver_distance_theorem :
  let speed := 60
  let first_duration := 4
  let second_duration := 9
  total_distance_traveled speed first_duration second_duration = 780 := by
  sorry

#check driver_distance_theorem

end NUMINAMATH_CALUDE_driver_distance_theorem_l3129_312932


namespace NUMINAMATH_CALUDE_z_plus_one_is_pure_imaginary_l3129_312988

theorem z_plus_one_is_pure_imaginary : 
  let z : ℂ := (-2 * Complex.I) / (1 + Complex.I)
  ∃ (y : ℝ), z + 1 = y * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_z_plus_one_is_pure_imaginary_l3129_312988


namespace NUMINAMATH_CALUDE_savings_for_three_shirts_l3129_312941

/-- The cost of a single item -/
def itemCost : ℕ := 10

/-- The discount percentage for the second item -/
def secondItemDiscount : ℚ := 1/2

/-- The discount percentage for the third item -/
def thirdItemDiscount : ℚ := 3/5

/-- Calculate the savings for a given number of items -/
def calculateSavings (n : ℕ) : ℚ :=
  if n ≤ 1 then 0
  else if n = 2 then secondItemDiscount * itemCost
  else secondItemDiscount * itemCost + thirdItemDiscount * itemCost

theorem savings_for_three_shirts :
  calculateSavings 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_savings_for_three_shirts_l3129_312941


namespace NUMINAMATH_CALUDE_triangle_perimeter_triangle_perimeter_proof_l3129_312908

/-- Given a triangle with sides of lengths 15 cm, 6 cm, and 12 cm, its perimeter is 33 cm. -/
theorem triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b c : ℝ),
      a = 15 ∧ b = 6 ∧ c = 12 ∧
      perimeter = a + b + c ∧
      perimeter = 33

-- The proof is omitted
theorem triangle_perimeter_proof : triangle_perimeter 33 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_triangle_perimeter_proof_l3129_312908
