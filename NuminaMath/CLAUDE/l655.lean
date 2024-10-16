import Mathlib

namespace NUMINAMATH_CALUDE_school_population_theorem_l655_65533

theorem school_population_theorem :
  ∀ (boys girls : ℕ),
  boys + girls = 300 →
  girls = (boys * 100) / 300 →
  boys = 225 := by
sorry

end NUMINAMATH_CALUDE_school_population_theorem_l655_65533


namespace NUMINAMATH_CALUDE_angle_DAE_is_10_degrees_l655_65500

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def angle (A B C : ℝ × ℝ) : ℝ := sorry

theorem angle_DAE_is_10_degrees 
  (A B C D O E: ℝ × ℝ) 
  (triangle : Triangle A B C) 
  (h1 : angle A C B = 60)
  (h2 : angle C B A = 70)
  (h3 : D.1 = B.1 + (C.1 - B.1) * ((A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2)) / ((C.1 - B.1)^2 + (C.2 - B.2)^2))
  (h4 : D.2 = B.2 + (C.2 - B.2) * ((A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2)) / ((C.1 - B.1)^2 + (C.2 - B.2)^2))
  (h5 : ∃ r, Circle O r = {A, B, C})
  (h6 : E.1 = 2 * O.1 - A.1 ∧ E.2 = 2 * O.2 - A.2) :
  angle D A E = 10 := by
sorry


end NUMINAMATH_CALUDE_angle_DAE_is_10_degrees_l655_65500


namespace NUMINAMATH_CALUDE_year_spans_53_or_54_weeks_l655_65501

/-- A year is either common (365 days) or leap (366 days) -/
inductive Year
  | Common
  | Leap

/-- Definition of how many days are in a year -/
def daysInYear (y : Year) : ℕ :=
  match y with
  | Year.Common => 365
  | Year.Leap => 366

/-- Definition of when a year covers a week -/
def yearCoversWeek (daysInYear : ℕ) (weekStartDay : ℕ) : Prop :=
  daysInYear - weekStartDay ≥ 6

/-- Theorem stating that a year can span either 53 or 54 weeks -/
theorem year_spans_53_or_54_weeks (y : Year) :
  ∃ (n : ℕ), (n = 53 ∨ n = 54) ∧
    (∀ (w : ℕ), w ≤ n → yearCoversWeek (daysInYear y) ((w - 1) * 7)) ∧
    (∀ (w : ℕ), w > n → ¬yearCoversWeek (daysInYear y) ((w - 1) * 7)) :=
  sorry

end NUMINAMATH_CALUDE_year_spans_53_or_54_weeks_l655_65501


namespace NUMINAMATH_CALUDE_sum_reciprocals_minus_products_l655_65526

theorem sum_reciprocals_minus_products (a b c : ℚ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a + b + c = a * b * c) : 
  a / b + a / c + b / a + b / c + c / a + c / b - a * b - b * c - c * a = -3 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_minus_products_l655_65526


namespace NUMINAMATH_CALUDE_min_jumps_to_visit_all_l655_65555

/-- Represents a jump on the circle -/
inductive Jump
| Two  : Jump  -- Jump of 2 points
| Three : Jump -- Jump of 3 points

/-- The number of points on the circle -/
def numPoints : ℕ := 2016

/-- Function to calculate the total distance covered by a sequence of jumps -/
def totalDistance (jumps : List Jump) : ℕ :=
  jumps.foldl (fun acc jump => acc + match jump with
    | Jump.Two => 2
    | Jump.Three => 3) 0

/-- Predicate to check if a sequence of jumps visits all points -/
def visitsAllPoints (jumps : List Jump) : Prop :=
  totalDistance jumps % numPoints = 0 ∧ 
  jumps.length ≥ numPoints

/-- The main theorem stating the minimum number of jumps required -/
theorem min_jumps_to_visit_all : 
  ∃ (jumps : List Jump), visitsAllPoints jumps ∧ 
    jumps.length = 2017 ∧ 
    (∀ (other_jumps : List Jump), visitsAllPoints other_jumps → 
      other_jumps.length ≥ 2017) := by
  sorry

end NUMINAMATH_CALUDE_min_jumps_to_visit_all_l655_65555


namespace NUMINAMATH_CALUDE_vector_problem_l655_65522

/-- Given vectors in R² -/
def a : Fin 2 → ℝ := ![4, 2]
def b : Fin 2 → ℝ := ![-1, 2]
def c (m : ℝ) : Fin 2 → ℝ := ![2, m]

/-- Dot product of two vectors in R² -/
def dot (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

/-- Parallel vectors in R² -/
def parallel (u v : Fin 2 → ℝ) : Prop := ∃ k : ℝ, u = fun i => k * (v i)

theorem vector_problem (m : ℝ) :
  (dot a (c m) < m^2 → (m > 4 ∨ m < -2)) ∧
  (parallel (fun i => a i + c m i) b → m = -14) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l655_65522


namespace NUMINAMATH_CALUDE_unique_number_satisfying_condition_l655_65598

theorem unique_number_satisfying_condition : ∃! x : ℚ, ((x / 3) * 24) - 7 = 41 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_condition_l655_65598


namespace NUMINAMATH_CALUDE_min_value_theorem_l655_65563

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := (x-1)^2 + (y-3)^2 = 4

-- Define the distance function
def dist_squared (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (x₁ - x₂)^2 + (y₁ - y₂)^2

-- Define the condition |PC₁| = |PC₂|
def point_condition (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧
  dist_squared a b x₁ y₁ = dist_squared a b x₂ y₂

-- Define the expression to be minimized
def expr_to_minimize (a b : ℝ) : ℝ := a^2 + b^2 - 6*a - 4*b + 13

-- State the theorem
theorem min_value_theorem :
  ∃ (min : ℝ), min = 8/5 ∧
  ∀ (a b : ℝ), point_condition a b → expr_to_minimize a b ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l655_65563


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l655_65582

/-- Two lines are parallel if and only if their slopes are equal but they are not identical --/
def are_parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ / n₁ = m₂ / n₂ ∧ m₁ / c₁ ≠ m₂ / c₂

/-- The problem statement --/
theorem parallel_lines_a_value :
  ∀ a : ℝ,
  are_parallel 1 a 6 (a - 2) 3 (2 * a) →
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l655_65582


namespace NUMINAMATH_CALUDE_expand_expression_l655_65515

theorem expand_expression (x : ℝ) : (x + 4) * (5 * x - 10) = 5 * x^2 + 10 * x - 40 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l655_65515


namespace NUMINAMATH_CALUDE_inscribed_triangle_angle_60_l655_65560

/-- Represents a triangle inscribed in a circle -/
structure InscribedTriangle where
  /-- Measure of arc PQ -/
  arc_pq : ℝ
  /-- Measure of arc QR -/
  arc_qr : ℝ
  /-- Measure of arc RP -/
  arc_rp : ℝ
  /-- The sum of all arcs is 360° -/
  sum_arcs : arc_pq + arc_qr + arc_rp = 360

/-- Theorem: If a triangle is inscribed in a circle with the given arc measures,
    then one of its interior angles is 60° -/
theorem inscribed_triangle_angle_60 (t : InscribedTriangle)
  (h1 : ∃ x : ℝ, t.arc_pq = x + 80 ∧ t.arc_qr = 3*x - 30 ∧ t.arc_rp = 2*x + 10) :
  ∃ θ : ℝ, θ = 60 ∧ (θ = t.arc_qr / 2 ∨ θ = t.arc_rp / 2 ∨ θ = t.arc_pq / 2) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_angle_60_l655_65560


namespace NUMINAMATH_CALUDE_original_mixture_volume_l655_65564

/-- Proves that given a mixture with 20% alcohol, if adding 2 litres of water
    results in a new mixture with 17.647058823529413% alcohol,
    then the original mixture volume was 15 litres. -/
theorem original_mixture_volume
  (original_alcohol_percentage : Real)
  (added_water : Real)
  (new_alcohol_percentage : Real)
  (h1 : original_alcohol_percentage = 0.20)
  (h2 : added_water = 2)
  (h3 : new_alcohol_percentage = 0.17647058823529413)
  : ∃ (original_volume : Real),
    original_volume * original_alcohol_percentage /
    (original_volume + added_water) = new_alcohol_percentage ∧
    original_volume = 15 := by
  sorry

end NUMINAMATH_CALUDE_original_mixture_volume_l655_65564


namespace NUMINAMATH_CALUDE_ball_cost_l655_65513

theorem ball_cost (C : ℝ) : 
  (C / 2 + C / 6 + C / 12 + 5 = C) → C = 20 := by sorry

end NUMINAMATH_CALUDE_ball_cost_l655_65513


namespace NUMINAMATH_CALUDE_remainder_99_101_div_9_l655_65572

theorem remainder_99_101_div_9 : (99 * 101) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_99_101_div_9_l655_65572


namespace NUMINAMATH_CALUDE_field_trip_girls_fraction_l655_65534

theorem field_trip_girls_fraction (b : ℚ) (h1 : b > 0) : 
  let g := 2 * b
  let girls_on_trip := (5 / 6) * g
  let boys_on_trip := (1 / 2) * b
  let total_on_trip := girls_on_trip + boys_on_trip
  girls_on_trip / total_on_trip = 10 / 13 := by
sorry


end NUMINAMATH_CALUDE_field_trip_girls_fraction_l655_65534


namespace NUMINAMATH_CALUDE_cost_of_tomato_seeds_l655_65538

theorem cost_of_tomato_seeds :
  let pumpkin_cost : ℚ := 5/2
  let chili_cost : ℚ := 9/10
  let pumpkin_packets : ℕ := 3
  let tomato_packets : ℕ := 4
  let chili_packets : ℕ := 5
  let total_spent : ℚ := 18
  ∃ tomato_cost : ℚ, 
    tomato_cost = 3/2 ∧
    pumpkin_cost * pumpkin_packets + tomato_cost * tomato_packets + chili_cost * chili_packets = total_spent :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_tomato_seeds_l655_65538


namespace NUMINAMATH_CALUDE_probability_third_is_three_l655_65580

-- Define the set of permutations
def T : Finset (Fin 6 → Fin 6) :=
  (Finset.univ.filter (λ σ : Fin 6 → Fin 6 => Function.Injective σ ∧ σ 0 ≠ 1))

-- Define the probability of the event
def prob_third_is_three (T : Finset (Fin 6 → Fin 6)) : ℚ :=
  (T.filter (λ σ : Fin 6 → Fin 6 => σ 2 = 2)).card / T.card

-- Theorem statement
theorem probability_third_is_three :
  prob_third_is_three T = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_third_is_three_l655_65580


namespace NUMINAMATH_CALUDE_tria_currency_base_l655_65528

/-- Converts a number from base r to base 10 -/
def toBase10 (digits : List Nat) (r : Nat) : Nat :=
  digits.foldr (fun d acc => d + r * acc) 0

/-- The problem statement -/
theorem tria_currency_base : ∃! r : Nat, r > 1 ∧
  toBase10 [5, 3, 2] r + toBase10 [2, 6, 0] r + toBase10 [2, 0, 8] r = toBase10 [1, 0, 0, 0] r :=
by
  sorry

end NUMINAMATH_CALUDE_tria_currency_base_l655_65528


namespace NUMINAMATH_CALUDE_lucky_number_theorem_l655_65576

/-- A "lucky number" is a three-digit positive integer that can be expressed as m(m+3) for some positive integer m. -/
def is_lucky_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m > 0 ∧ n = m * (m + 3)

/-- The largest "lucky number". -/
def largest_lucky_number : ℕ := 990

/-- The sum of all N where M and N are both "lucky numbers" and M - N = 350. -/
def sum_of_satisfying_N : ℕ := 614

theorem lucky_number_theorem :
  (∀ n : ℕ, is_lucky_number n → n ≤ largest_lucky_number) ∧
  (∀ M N : ℕ, is_lucky_number M → is_lucky_number N → M - N = 350 →
    N = 460 ∨ N = 154) ∧
  (sum_of_satisfying_N = 614) := by sorry

end NUMINAMATH_CALUDE_lucky_number_theorem_l655_65576


namespace NUMINAMATH_CALUDE_divided_square_plot_area_l655_65537

/-- Represents a rectangular plot -/
structure RectangularPlot where
  length : ℝ
  width : ℝ

/-- Represents a square plot divided into 8 equal rectangular parts -/
structure DividedSquarePlot where
  part : RectangularPlot
  perimeter : ℝ

/-- The perimeter of a rectangular plot -/
def RectangularPlot.perimeter (r : RectangularPlot) : ℝ :=
  2 * (r.length + r.width)

/-- The area of a square plot -/
def square_area (side : ℝ) : ℝ :=
  side * side

theorem divided_square_plot_area (d : DividedSquarePlot) 
    (h1 : d.part.length = 2 * d.part.width)
    (h2 : d.perimeter = d.part.perimeter) :
    square_area (4 * d.part.width) = (4 * d.perimeter^2) / 9 := by
  sorry

end NUMINAMATH_CALUDE_divided_square_plot_area_l655_65537


namespace NUMINAMATH_CALUDE_max_difference_intersection_points_l655_65502

/-- The first function f(x) = 2 - x^2 + 2x^3 -/
def f (x : ℝ) : ℝ := 2 - x^2 + 2*x^3

/-- The second function g(x) = 3 + 2x^2 + 2x^3 -/
def g (x : ℝ) : ℝ := 3 + 2*x^2 + 2*x^3

/-- Theorem stating that the maximum difference between y-coordinates of intersection points is 4√3/9 -/
theorem max_difference_intersection_points :
  ∃ (x₁ x₂ : ℝ), f x₁ = g x₁ ∧ f x₂ = g x₂ ∧ 
  ∀ (y₁ y₂ : ℝ), (∃ (x : ℝ), f x = g x ∧ (y₁ = f x ∨ y₁ = g x)) →
                 (∃ (x : ℝ), f x = g x ∧ (y₂ = f x ∨ y₂ = g x)) →
                 |y₁ - y₂| ≤ 4 * Real.sqrt 3 / 9 :=
by sorry

end NUMINAMATH_CALUDE_max_difference_intersection_points_l655_65502


namespace NUMINAMATH_CALUDE_card_trick_strategy_exists_l655_65565

/-- Represents a card in the set of 29 cards -/
def Card := Fin 29

/-- Represents the strategy for selecting two cards to show -/
def Strategy := (Card × Card) → (Card × Card)

/-- Checks if two cards are adjacent in the circular arrangement -/
def adjacent (a b : Card) : Prop :=
  b.val = (a.val % 29 + 1) ∨ a.val = (b.val % 29 + 1)

/-- Determines if a strategy is valid for guessing hidden cards -/
def valid_strategy (s : Strategy) : Prop :=
  ∀ (hidden : Card × Card),
    let shown := s hidden
    ∃! (guessed : Card × Card),
      (guessed = hidden ∧ ¬adjacent guessed.1 guessed.2) ∨
      (guessed = hidden ∧ adjacent guessed.1 guessed.2)

/-- Theorem stating that there exists a valid strategy for the card trick -/
theorem card_trick_strategy_exists : ∃ (s : Strategy), valid_strategy s := by
  sorry

end NUMINAMATH_CALUDE_card_trick_strategy_exists_l655_65565


namespace NUMINAMATH_CALUDE_correct_calculation_l655_65568

theorem correct_calculation (x : ℕ) (h : x - 6 = 51) : x * 6 = 342 := by
  sorry

#check correct_calculation

end NUMINAMATH_CALUDE_correct_calculation_l655_65568


namespace NUMINAMATH_CALUDE_solve_for_k_l655_65567

theorem solve_for_k : ∀ k : ℤ, 
  (∀ x : ℤ, 2*x - 3 = 3*x - 2 + k ↔ x = 2) → 
  k = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l655_65567


namespace NUMINAMATH_CALUDE_road_trip_gas_cost_l655_65571

/-- Calculates the total cost of filling up a car's gas tank at multiple stations -/
theorem road_trip_gas_cost (tank_capacity : ℝ) (prices : List ℝ) : 
  tank_capacity = 12 ∧ 
  prices = [3, 3.5, 4, 4.5] →
  (prices.map (λ price => tank_capacity * price)).sum = 180 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_gas_cost_l655_65571


namespace NUMINAMATH_CALUDE_handle_break_even_point_l655_65523

/-- Represents the break-even point calculation for a company producing handles --/
theorem handle_break_even_point
  (fixed_cost : ℝ)
  (variable_cost : ℝ)
  (selling_price : ℝ)
  (break_even_quantity : ℝ)
  (h1 : fixed_cost = 7640)
  (h2 : variable_cost = 0.60)
  (h3 : selling_price = 4.60)
  (h4 : break_even_quantity = 1910) :
  fixed_cost + variable_cost * break_even_quantity = selling_price * break_even_quantity :=
by
  sorry

#check handle_break_even_point

end NUMINAMATH_CALUDE_handle_break_even_point_l655_65523


namespace NUMINAMATH_CALUDE_school_boys_count_l655_65589

theorem school_boys_count :
  ∀ (boys girls : ℕ),
  (boys : ℚ) / girls = 5 / 13 →
  girls = boys + 64 →
  boys = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l655_65589


namespace NUMINAMATH_CALUDE_x_value_l655_65553

theorem x_value : Real.sqrt (20 - 17 - 2 * 0 - 1 + 7) = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l655_65553


namespace NUMINAMATH_CALUDE_division_fraction_equality_l655_65561

theorem division_fraction_equality : (2 / 7) / (1 / 14) = 4 := by sorry

end NUMINAMATH_CALUDE_division_fraction_equality_l655_65561


namespace NUMINAMATH_CALUDE_carrots_not_used_l655_65532

theorem carrots_not_used (total : ℕ) (before_lunch_fraction : ℚ) (end_of_day_fraction : ℚ) : 
  total = 300 →
  before_lunch_fraction = 2 / 5 →
  end_of_day_fraction = 3 / 5 →
  (total - (before_lunch_fraction * total).num - (end_of_day_fraction * (total - (before_lunch_fraction * total).num)).num) = 72 := by
  sorry

end NUMINAMATH_CALUDE_carrots_not_used_l655_65532


namespace NUMINAMATH_CALUDE_tetrahedron_triangle_existence_l655_65524

/-- Represents a tetrahedron with edge lengths -/
structure Tetrahedron where
  edges : Fin 6 → ℝ
  edge_positive : ∀ i, edges i > 0

/-- Checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Main theorem: In any tetrahedron, there exists a vertex such that 
    the edges connected to it can form a triangle -/
theorem tetrahedron_triangle_existence (t : Tetrahedron) : 
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    can_form_triangle (t.edges i) (t.edges j) (t.edges k) :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_triangle_existence_l655_65524


namespace NUMINAMATH_CALUDE_child_ticket_cost_is_4_l655_65525

/-- The cost of a child's ticket at a ball game -/
def child_ticket_cost (num_adults num_children : ℕ) (adult_ticket_cost total_bill : ℚ) : ℚ :=
  (total_bill - num_adults * adult_ticket_cost) / num_children

theorem child_ticket_cost_is_4 :
  child_ticket_cost 10 11 8 124 = 4 := by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_is_4_l655_65525


namespace NUMINAMATH_CALUDE_min_value_expression_l655_65591

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) :
  x^2 + 8*x*y + 4*y^2 + 4*z^2 ≥ 384 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l655_65591


namespace NUMINAMATH_CALUDE_divisibility_rules_l655_65599

-- Define a function to get the last digit of a natural number
def lastDigit (n : ℕ) : ℕ := n % 10

-- Define a function to get the last two digits of a natural number
def lastTwoDigits (n : ℕ) : ℕ := n % 100

-- Define a function to check if a number is even
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define a function to sum the digits of a natural number
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

theorem divisibility_rules (n : ℕ) :
  (n % 2 = 0 ↔ isEven (lastDigit n)) ∧
  (n % 5 = 0 ↔ lastDigit n = 0 ∨ lastDigit n = 5) ∧
  (n % 3 = 0 ↔ sumOfDigits n % 3 = 0) ∧
  (n % 4 = 0 ↔ lastTwoDigits n % 4 = 0) ∧
  (n % 25 = 0 ↔ lastTwoDigits n % 25 = 0) :=
by sorry


end NUMINAMATH_CALUDE_divisibility_rules_l655_65599


namespace NUMINAMATH_CALUDE_line_inclination_angle_l655_65570

theorem line_inclination_angle (x y : ℝ) :
  let line_equation := (2 * x - 2 * y - 1 = 0)
  let slope := (2 : ℝ) / 2
  let angle_of_inclination := Real.arctan slope
  line_equation → angle_of_inclination = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l655_65570


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l655_65529

theorem contrapositive_equivalence (x : ℝ) :
  (x^2 < 1 → -1 ≤ x ∧ x < 1) ↔ (x < -1 ∨ x ≥ 1 → x^2 ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l655_65529


namespace NUMINAMATH_CALUDE_largest_lcm_with_18_l655_65588

theorem largest_lcm_with_18 :
  max (Nat.lcm 18 3) (max (Nat.lcm 18 6) (max (Nat.lcm 18 9) (max (Nat.lcm 18 12) (max (Nat.lcm 18 15) (Nat.lcm 18 18))))) = 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_18_l655_65588


namespace NUMINAMATH_CALUDE_fraction_equality_l655_65595

theorem fraction_equality (x y : ℝ) (h : (x + y) / (1 - x * y) = Real.sqrt 5) :
  |1 - x * y| / (Real.sqrt (1 + x^2) * Real.sqrt (1 + y^2)) = Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l655_65595


namespace NUMINAMATH_CALUDE_local_face_value_difference_l655_65510

def number : ℕ := 96348621

def digit_position (n : ℕ) (d : ℕ) : ℕ :=
  (n.digits 10).reverse.indexOf d

def local_value (n : ℕ) (d : ℕ) : ℕ :=
  d * (10 ^ (digit_position n d))

def face_value (d : ℕ) : ℕ := d

theorem local_face_value_difference :
  local_value number 8 - face_value 8 = 7992 := by
  sorry

end NUMINAMATH_CALUDE_local_face_value_difference_l655_65510


namespace NUMINAMATH_CALUDE_fraction_product_cube_main_problem_l655_65544

theorem fraction_product_cube (a b c d e f : ℚ) :
  (a / b) ^ 3 * (c / d) ^ 3 * (e / f) ^ 3 = ((a * c * e) / (b * d * f)) ^ 3 :=
sorry

theorem main_problem : 
  (8 / 9) ^ 3 * (1 / 3) ^ 3 * (2 / 5) ^ 3 = 4096 / 2460375 :=
sorry

end NUMINAMATH_CALUDE_fraction_product_cube_main_problem_l655_65544


namespace NUMINAMATH_CALUDE_ellipse_properties_l655_65506

/-- Definition of the ellipse C -/
def Ellipse (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

/-- Definition of a point being on the ellipse -/
def OnEllipse (p : ℝ × ℝ) : Prop :=
  Ellipse p.1 p.2

/-- The focus of the ellipse -/
def F : ℝ × ℝ := (1, 0)

/-- A line perpendicular to the x-axis passing through F -/
def PerpendicularLine (y : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 1}

/-- The dot product of two points -/
def DotProduct (p q : ℝ × ℝ) : ℝ :=
  p.1 * q.1 + p.2 * q.2

/-- The circle with diameter MN passes through a fixed point -/
def CirclePassesThroughFixedPoint (m n : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t = 1 ∨ t = 3 ∧ 
    (2 - t)^2 + (m.2 - 0) * (n.2 - 0) / ((m.1 - t) * (n.1 - t)) = 1

theorem ellipse_properties :
  ∀ (a b : ℝ × ℝ),
    OnEllipse a ∧ OnEllipse b ∧
    (∃ y, a ∈ PerpendicularLine y ∧ b ∈ PerpendicularLine y) →
    DotProduct a b = 1/2 →
    (∀ (m n : ℝ × ℝ), 
      OnEllipse m ∧ OnEllipse n ∧ m.1 = 2 ∧ n.1 = 2 →
      CirclePassesThroughFixedPoint m n) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l655_65506


namespace NUMINAMATH_CALUDE_polynomial_inequality_l655_65581

theorem polynomial_inequality (x : ℝ) : 
  x^6 + 4*x^5 + 2*x^4 - 6*x^3 - 2*x^2 + 4*x - 1 ≥ 0 ↔ 
  x ≤ -1 - Real.sqrt 2 ∨ x = (-1 - Real.sqrt 5) / 2 ∨ x ≥ -1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l655_65581


namespace NUMINAMATH_CALUDE_max_candies_eaten_l655_65562

/-- The maximum sum of products of pairs from a set of n elements -/
def maxProductSum (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of initial elements on the board -/
def initialCount : ℕ := 30

/-- The theorem stating the maximum number of candies Karlson could eat -/
theorem max_candies_eaten :
  maxProductSum initialCount = 435 := by
  sorry

end NUMINAMATH_CALUDE_max_candies_eaten_l655_65562


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l655_65559

theorem fractional_equation_solution (x a : ℝ) : 
  (2 * x + a) / (x + 1) = 1 → x < 0 → a > 1 ∧ a ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l655_65559


namespace NUMINAMATH_CALUDE_book_shelf_problem_l655_65590

theorem book_shelf_problem (total_books : ℕ) (books_moved : ℕ) 
  (h1 : total_books = 180)
  (h2 : books_moved = 15)
  (h3 : ∃ (upper lower : ℕ), 
    upper + lower = total_books ∧ 
    (lower + books_moved) = 2 * (upper - books_moved)) :
  ∃ (original_upper original_lower : ℕ),
    original_upper = 75 ∧ 
    original_lower = 105 ∧
    original_upper + original_lower = total_books := by
  sorry

end NUMINAMATH_CALUDE_book_shelf_problem_l655_65590


namespace NUMINAMATH_CALUDE_total_apples_is_45_l655_65556

/-- The number of apples given to each person -/
def apples_per_person : ℝ := 15.0

/-- The number of people who received apples -/
def number_of_people : ℝ := 3.0

/-- The total number of apples given -/
def total_apples : ℝ := apples_per_person * number_of_people

/-- Theorem stating that the total number of apples is 45.0 -/
theorem total_apples_is_45 : total_apples = 45.0 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_is_45_l655_65556


namespace NUMINAMATH_CALUDE_intersection_when_a_half_range_of_a_when_disjoint_l655_65585

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_when_a_half :
  A (1/2) ∩ B = {x | 0 < x ∧ x < 1} := by sorry

theorem range_of_a_when_disjoint (a : ℝ) :
  (A a).Nonempty → (A a ∩ B = ∅) →
  (-2 < a ∧ a ≤ -1/2) ∨ (a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_half_range_of_a_when_disjoint_l655_65585


namespace NUMINAMATH_CALUDE_blue_crayons_count_l655_65575

theorem blue_crayons_count (blue : ℕ) (red : ℕ) : 
  red = 4 * blue →  -- Condition 1: Red crayons are four times blue crayons
  blue > 0 →        -- Condition 2: There is at least one blue crayon
  blue + red = 15 → -- Condition 3: Total number of crayons is 15
  blue = 3 :=        -- Conclusion: Number of blue crayons is 3
by
  sorry

end NUMINAMATH_CALUDE_blue_crayons_count_l655_65575


namespace NUMINAMATH_CALUDE_min_value_theorem_l655_65508

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_line : m * 1 - n * (-1) - 1 = 0) : 
  (1 / m + 2 / n) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_min_value_theorem_l655_65508


namespace NUMINAMATH_CALUDE_student_weight_l655_65573

theorem student_weight (student sister brother : ℝ) 
  (h1 : student - 5 = 2 * sister)
  (h2 : student + sister + brother = 150)
  (h3 : brother = sister - 10) :
  student = 82.5 := by
sorry

end NUMINAMATH_CALUDE_student_weight_l655_65573


namespace NUMINAMATH_CALUDE_base10_216_equals_base9_260_l655_65517

/-- Converts a natural number from base 10 to base 9 --/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 9 to a natural number in base 10 --/
def fromBase9 (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if all digits in a list are less than 9 --/
def validBase9Digits (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, d < 9

theorem base10_216_equals_base9_260 :
  let base9Digits := [2, 6, 0]
  validBase9Digits base9Digits ∧ fromBase9 base9Digits = 216 :=
by
  sorry

end NUMINAMATH_CALUDE_base10_216_equals_base9_260_l655_65517


namespace NUMINAMATH_CALUDE_value_range_of_f_l655_65574

/-- The function f(x) = x^2 - 2x -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The domain of f is [0, +∞) -/
def domain : Set ℝ := { x | x ≥ 0 }

theorem value_range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | y ≥ -1 } := by sorry

end NUMINAMATH_CALUDE_value_range_of_f_l655_65574


namespace NUMINAMATH_CALUDE_robotics_camp_age_problem_l655_65579

theorem robotics_camp_age_problem (total_members : ℕ) (overall_avg_age : ℕ) 
  (num_girls num_boys num_adults : ℕ) (avg_age_girls avg_age_boys : ℕ) :
  total_members = 60 →
  overall_avg_age = 20 →
  num_girls = 30 →
  num_boys = 20 →
  num_adults = 10 →
  avg_age_girls = 18 →
  avg_age_boys = 22 →
  num_girls + num_boys + num_adults = total_members →
  (avg_age_girls * num_girls + avg_age_boys * num_boys + 
   22 * num_adults : ℕ) / total_members = overall_avg_age :=
by sorry

end NUMINAMATH_CALUDE_robotics_camp_age_problem_l655_65579


namespace NUMINAMATH_CALUDE_count_squares_in_G_l655_65535

-- Define the set G
def G : Set (ℤ × ℤ) := {p | 3 ≤ |p.1| ∧ |p.1| ≤ 7 ∧ 3 ≤ |p.2| ∧ |p.2| ≤ 7}

-- Define a square with vertices in G
def Square (a b c d : ℤ × ℤ) : Prop :=
  a ∈ G ∧ b ∈ G ∧ c ∈ G ∧ d ∈ G ∧
  (a.1 - b.1)^2 + (a.2 - b.2)^2 = (b.1 - c.1)^2 + (b.2 - c.2)^2 ∧
  (b.1 - c.1)^2 + (b.2 - c.2)^2 = (c.1 - d.1)^2 + (c.2 - d.2)^2 ∧
  (c.1 - d.1)^2 + (c.2 - d.2)^2 = (d.1 - a.1)^2 + (d.2 - a.2)^2 ∧
  (a.1 - b.1)^2 + (a.2 - b.2)^2 ≥ 36

-- Theorem statement
theorem count_squares_in_G :
  (∃ (s : Finset (ℤ × ℤ × ℤ × ℤ)), s.card = 4 ∧
    ∀ (q : ℤ × ℤ × ℤ × ℤ), q ∈ s ↔ Square q.1 q.2.1 q.2.2.1 q.2.2.2) :=
sorry

end NUMINAMATH_CALUDE_count_squares_in_G_l655_65535


namespace NUMINAMATH_CALUDE_acute_angle_vector_range_l655_65545

def a (x : ℝ) : ℝ × ℝ := (x, 2)
def b : ℝ × ℝ := (-3, 6)

theorem acute_angle_vector_range (x : ℝ) :
  (∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ Real.cos θ = (a x).1 * b.1 + (a x).2 * b.2 / (Real.sqrt ((a x).1^2 + (a x).2^2) * Real.sqrt (b.1^2 + b.2^2))) →
  x < 4 ∧ x ≠ -1 :=
by sorry

end NUMINAMATH_CALUDE_acute_angle_vector_range_l655_65545


namespace NUMINAMATH_CALUDE_min_value_problem_l655_65587

theorem min_value_problem (x y : ℝ) 
  (h1 : x - 1 ≥ 0)
  (h2 : x - y + 1 ≤ 0)
  (h3 : x + y - 4 ≤ 0) :
  ∃ (m : ℝ), m = 1/4 ∧ ∀ (a b : ℝ), 
    a - 1 ≥ 0 → a - b + 1 ≤ 0 → a + b - 4 ≤ 0 → 
    a / (b + 1) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l655_65587


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l655_65541

theorem gcd_of_three_numbers :
  Nat.gcd 105 (Nat.gcd 1001 2436) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l655_65541


namespace NUMINAMATH_CALUDE_distance_between_points_l655_65547

-- Define the equation of the curve
def on_curve (x y : ℝ) : Prop := y^2 + x^3 = 2*x*y + 4

-- Define the theorem
theorem distance_between_points (e a b : ℝ) 
  (h1 : on_curve e a) 
  (h2 : on_curve e b) 
  (h3 : a ≠ b) : 
  |a - b| = 2 * Real.sqrt (e^2 - e^3 + 4) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l655_65547


namespace NUMINAMATH_CALUDE_train_platform_equal_length_l655_65536

/-- Given a train and platform with specific properties, prove that their lengths are equal --/
theorem train_platform_equal_length 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_speed = 108 * 1000 / 60) -- 108 km/hr converted to m/min
  (h2 : train_length = 900)
  (h3 : crossing_time = 1) :
  train_length = train_speed * crossing_time - train_length := by
  sorry

#check train_platform_equal_length

end NUMINAMATH_CALUDE_train_platform_equal_length_l655_65536


namespace NUMINAMATH_CALUDE_integral_odd_function_integral_even_function_integral_positive_function_exists_counterexample_for_D_incorrect_proposition_l655_65507

-- Define the necessary concepts
def continuous (f : ℝ → ℝ) : Prop := sorry
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def integral (f : ℝ → ℝ) (a b : ℝ) : ℝ := sorry

-- Theorem A
theorem integral_odd_function (f : ℝ → ℝ) (α : ℝ) :
  continuous f → odd_function f → integral f (-α) α = 0 := by sorry

-- Theorem B
theorem integral_even_function (f : ℝ → ℝ) (a : ℝ) :
  continuous f → even_function f → integral f (-a) a = 2 * integral f 0 a := by sorry

-- Theorem C
theorem integral_positive_function (f : ℝ → ℝ) (a b : ℝ) :
  continuous f → (∀ x ∈ [a, b], f x > 0) → integral f a b > 0 := by sorry

-- Theorem D (false)
theorem exists_counterexample_for_D :
  ∃ f : ℝ → ℝ, ∃ a b : ℝ,
    continuous f ∧ 
    integral f a b > 0 ∧ 
    ¬(∀ x ∈ [a, b], f x > 0) := by sorry

-- Main theorem
theorem incorrect_proposition :
  ¬(∀ f : ℝ → ℝ, ∀ a b : ℝ,
    continuous f → integral f a b > 0 → (∀ x ∈ [a, b], f x > 0)) := by sorry

end NUMINAMATH_CALUDE_integral_odd_function_integral_even_function_integral_positive_function_exists_counterexample_for_D_incorrect_proposition_l655_65507


namespace NUMINAMATH_CALUDE_integer_ratio_problem_l655_65540

theorem integer_ratio_problem (A B C D : ℕ) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 →
  (A + B + C + D) / 4 = 16 →
  ∃ k : ℕ, A = k * B →
  B = C - 2 →
  D = 2 →
  A / B = 28 := by
sorry

end NUMINAMATH_CALUDE_integer_ratio_problem_l655_65540


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l655_65543

theorem isosceles_triangle_condition (a b c A B C : ℝ) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (a > 0) → (b > 0) → (c > 0) →
  (a * Real.cos C + c * Real.cos B = b) →
  (a = b ∧ A = B) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l655_65543


namespace NUMINAMATH_CALUDE_laptop_final_price_l655_65593

/-- Calculate the final price of a laptop given the original price, two discount rates, and a recycling fee rate. -/
theorem laptop_final_price
  (original_price : ℝ)
  (discount1 : ℝ)
  (discount2 : ℝ)
  (recycling_fee_rate : ℝ)
  (h1 : original_price = 1000)
  (h2 : discount1 = 0.1)
  (h3 : discount2 = 0.25)
  (h4 : recycling_fee_rate = 0.05) :
  let price_after_discount1 := original_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  let recycling_fee := price_after_discount2 * recycling_fee_rate
  let final_price := price_after_discount2 + recycling_fee
  final_price = 708.75 :=
by
  sorry

end NUMINAMATH_CALUDE_laptop_final_price_l655_65593


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l655_65504

theorem smallest_positive_integer_congruence :
  ∃! (x : ℕ), x > 0 ∧ (45 * x + 13) % 30 = 5 ∧ ∀ (y : ℕ), y > 0 ∧ (45 * y + 13) % 30 = 5 → x ≤ y :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l655_65504


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_l655_65514

def Digits : Finset ℕ := {4, 5, 6, 7, 8, 9, 10}

def is_valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧ 
  a ≥ 100 ∧ a < 1000 ∧ 
  b ≥ 100 ∧ b < 1000 ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.range 10 \ {0}))) = 7 ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits ∧ (d ∈ (Finset.range 10 \ {0}) ∨ d = 10)) ((Finset.range 10 \ {0}) ∪ {10}))) = 6

theorem smallest_sum_of_digits : 
  ∀ a b : ℕ, is_valid_pair a b → a + b ≥ 1245 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_l655_65514


namespace NUMINAMATH_CALUDE_temperature_conversion_l655_65583

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 50 → k = 122 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l655_65583


namespace NUMINAMATH_CALUDE_democrat_ratio_l655_65527

theorem democrat_ratio (total : ℕ) (female_dem : ℕ) :
  total = 840 →
  female_dem = 140 →
  (∃ (female male : ℕ),
    female + male = total ∧
    2 * female_dem = female ∧
    4 * female_dem = male) →
  3 * (2 * female_dem) = total :=
by
  sorry

end NUMINAMATH_CALUDE_democrat_ratio_l655_65527


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l655_65597

theorem quadratic_inequality_solution_set (a b : ℝ) : 
  (∀ x, x^2 - (a+1)*x + b ≤ 0 ↔ -4 ≤ x ∧ x ≤ 3) → a + b = -14 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l655_65597


namespace NUMINAMATH_CALUDE_total_pizza_slices_l655_65542

theorem total_pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_pizzas = 36) 
  (h2 : slices_per_pizza = 12) : 
  num_pizzas * slices_per_pizza = 432 :=
by sorry

end NUMINAMATH_CALUDE_total_pizza_slices_l655_65542


namespace NUMINAMATH_CALUDE_sum_of_four_integers_with_product_5_4_l655_65584

theorem sum_of_four_integers_with_product_5_4 (a b c d : ℕ+) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 5^4 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) + (d : ℕ) = 156 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_integers_with_product_5_4_l655_65584


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l655_65512

/-- The repeating decimal 0.363636... -/
def repeating_decimal : ℚ := 0.363636

/-- The fraction 4/11 -/
def fraction : ℚ := 4 / 11

/-- Theorem stating that the repeating decimal 0.363636... equals 4/11 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l655_65512


namespace NUMINAMATH_CALUDE_guess_two_digit_number_l655_65586

/-- A two-digit number is between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem guess_two_digit_number (x : ℕ) (h : TwoDigitNumber x) :
  (2 * x + 5) * 5 = 715 → x = 69 := by
  sorry

end NUMINAMATH_CALUDE_guess_two_digit_number_l655_65586


namespace NUMINAMATH_CALUDE_three_digit_number_divisible_by_seven_l655_65548

theorem three_digit_number_divisible_by_seven :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 3 ∧ (n / 100) % 10 = 5 ∧ n % 7 = 0 ∧ n = 553 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_divisible_by_seven_l655_65548


namespace NUMINAMATH_CALUDE_cherry_tree_leaves_l655_65551

theorem cherry_tree_leaves (original_plan : ℕ) (actual_multiplier : ℕ) (leaves_per_tree : ℕ) : 
  original_plan = 7 → 
  actual_multiplier = 2 → 
  leaves_per_tree = 100 → 
  (original_plan * actual_multiplier * leaves_per_tree) = 1400 := by
sorry

end NUMINAMATH_CALUDE_cherry_tree_leaves_l655_65551


namespace NUMINAMATH_CALUDE_male_salmon_count_l655_65566

theorem male_salmon_count (total : ℕ) (female : ℕ) (h1 : total = 971639) (h2 : female = 259378) :
  total - female = 712261 := by
  sorry

end NUMINAMATH_CALUDE_male_salmon_count_l655_65566


namespace NUMINAMATH_CALUDE_max_transitions_to_wiki_l655_65558

theorem max_transitions_to_wiki (channel_a channel_b channel_c : ℕ) :
  channel_a = 850 * 6 / 100 ∧
  channel_b = 1500 * 42 / 1000 ∧
  channel_c = 4536 / 72 →
  max channel_b channel_c = 63 :=
by
  sorry

end NUMINAMATH_CALUDE_max_transitions_to_wiki_l655_65558


namespace NUMINAMATH_CALUDE_notebook_cost_l655_65503

/-- The cost of a notebook and pencil, given their relationship -/
theorem notebook_cost (notebook_cost pencil_cost : ℝ)
  (total_cost : notebook_cost + pencil_cost = 3.20)
  (cost_difference : notebook_cost = pencil_cost + 2.50) :
  notebook_cost = 2.85 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l655_65503


namespace NUMINAMATH_CALUDE_complex_projective_transformation_properties_l655_65550

-- Define a complex projective transformation
noncomputable def ComplexProjectiveTransformation := ℂ → ℂ

-- State the theorem
theorem complex_projective_transformation_properties
  (f : ComplexProjectiveTransformation) :
  (∃ (a b c d : ℂ), ∀ z, f z = (a * z + b) / (c * z + d)) ∧
  (∃! (p q : ℂ), f p = p ∧ f q = q) :=
sorry

end NUMINAMATH_CALUDE_complex_projective_transformation_properties_l655_65550


namespace NUMINAMATH_CALUDE_complement_union_theorem_l655_65518

-- Define the universal set U
def U : Set ℕ := {x | 0 ≤ x ∧ x < 6}

-- Define set A
def A : Set ℕ := {1, 3, 5}

-- Define set B
def B : Set ℕ := {x ∈ U | x^2 + 4 = 5*x}

-- Theorem statement
theorem complement_union_theorem : 
  (U \ A) ∪ (U \ B) = {0, 2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l655_65518


namespace NUMINAMATH_CALUDE_sum_equals_three_sqrt_fourteen_over_seven_l655_65596

theorem sum_equals_three_sqrt_fourteen_over_seven
  (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 1) 
  (h2 : x + 2*y + 3*z = Real.sqrt 14) : 
  x + y + z = (3 * Real.sqrt 14) / 7 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_three_sqrt_fourteen_over_seven_l655_65596


namespace NUMINAMATH_CALUDE_perpendicular_line_correct_parallel_lines_correct_l655_65592

-- Define the given line l
def line_l (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Define point A
def point_A : ℝ × ℝ := (3, 2)

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := x + 2 * y - 7 = 0

-- Define the parallel lines
def parallel_line_1 (x y : ℝ) : Prop := 2 * x - y + 6 = 0
def parallel_line_2 (x y : ℝ) : Prop := 2 * x - y - 4 = 0

-- Theorem for the perpendicular line
theorem perpendicular_line_correct :
  (perp_line point_A.1 point_A.2) ∧
  (∀ x y : ℝ, line_l x y → (x - point_A.1) * 1 + (y - point_A.2) * 2 = 0) :=
sorry

-- Theorem for the parallel lines
theorem parallel_lines_correct :
  (∀ x y : ℝ, (parallel_line_1 x y ∨ parallel_line_2 x y) →
    (abs (6 - 1) / Real.sqrt (2^2 + 1) = Real.sqrt 5 ∨
     abs (-4 - 1) / Real.sqrt (2^2 + 1) = Real.sqrt 5)) ∧
  (∀ x y : ℝ, line_l x y → (2 * 1 + 1 * 1 = 2 * 1 + 1 * 1)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_correct_parallel_lines_correct_l655_65592


namespace NUMINAMATH_CALUDE_matrix_power_result_l655_65552

theorem matrix_power_result (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B.mulVec (![7, -3]) = ![-14, 6]) :
  (B^4).mulVec (![7, -3]) = ![112, -48] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_result_l655_65552


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_4652_l655_65519

theorem largest_prime_factor_of_4652 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4652 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 4652 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_4652_l655_65519


namespace NUMINAMATH_CALUDE_dave_walking_probability_l655_65557

/-- Represents the number of gates in the airport terminal -/
def total_gates : ℕ := 15

/-- Represents the number of gates Dave can be assigned to -/
def dave_gates : ℕ := 10

/-- Represents the distance between adjacent gates in feet -/
def gate_distance : ℕ := 100

/-- Represents the maximum walking distance in feet -/
def max_walk_distance : ℕ := 300

/-- Calculates the number of valid gate combinations for Dave's initial and new gates -/
def total_combinations : ℕ := dave_gates * (dave_gates - 1)

/-- Calculates the number of valid gate combinations where Dave walks 300 feet or less -/
def valid_combinations : ℕ := 58

/-- The probability of Dave walking 300 feet or fewer to his new gate -/
def probability : ℚ := valid_combinations / total_combinations

theorem dave_walking_probability :
  probability = 29 / 45 := by sorry

end NUMINAMATH_CALUDE_dave_walking_probability_l655_65557


namespace NUMINAMATH_CALUDE_investment_problem_l655_65520

def total_investment : ℝ := 1000
def silver_rate : ℝ := 0.04
def gold_rate : ℝ := 0.06
def years : ℕ := 3
def final_amount : ℝ := 1206.11

def silver_investment (x : ℝ) : ℝ := x * (1 + silver_rate) ^ years
def gold_investment (x : ℝ) : ℝ := (total_investment - x) * (1 + gold_rate) ^ years

theorem investment_problem (x : ℝ) :
  silver_investment x + gold_investment x = final_amount →
  x = 228.14 := by sorry

end NUMINAMATH_CALUDE_investment_problem_l655_65520


namespace NUMINAMATH_CALUDE_container_capacity_is_20_l655_65509

-- Define the capacity of the container
def container_capacity : ℝ := 20

-- Define the initial fill percentage
def initial_fill_percentage : ℝ := 0.30

-- Define the final fill percentage
def final_fill_percentage : ℝ := 0.75

-- Define the amount of water added
def water_added : ℝ := 9

-- Theorem stating the container capacity is 20 liters
theorem container_capacity_is_20 :
  (final_fill_percentage * container_capacity - initial_fill_percentage * container_capacity = water_added) ∧
  (container_capacity = 20) :=
sorry

end NUMINAMATH_CALUDE_container_capacity_is_20_l655_65509


namespace NUMINAMATH_CALUDE_average_equation_solution_l655_65577

theorem average_equation_solution (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 84 → a = 32 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l655_65577


namespace NUMINAMATH_CALUDE_karls_drive_distance_l655_65578

/-- Represents the problem of calculating Karl's total drive distance --/
theorem karls_drive_distance :
  -- Conditions
  let miles_per_gallon : ℝ := 35
  let tank_capacity : ℝ := 14
  let initial_drive : ℝ := 350
  let gas_bought : ℝ := 8
  let final_tank_fraction : ℝ := 1/2

  -- Definitions derived from conditions
  let initial_gas_used : ℝ := initial_drive / miles_per_gallon
  let remaining_gas_after_initial_drive : ℝ := tank_capacity - initial_gas_used
  let gas_after_refuel : ℝ := remaining_gas_after_initial_drive + gas_bought
  let final_gas : ℝ := tank_capacity * final_tank_fraction
  let gas_used_second_leg : ℝ := gas_after_refuel - final_gas
  let second_leg_distance : ℝ := gas_used_second_leg * miles_per_gallon
  let total_distance : ℝ := initial_drive + second_leg_distance

  -- Theorem statement
  total_distance = 525 := by
  sorry

end NUMINAMATH_CALUDE_karls_drive_distance_l655_65578


namespace NUMINAMATH_CALUDE_f_properties_l655_65511

def f (x : ℝ) := |2*x + 1| - |x - 2|

theorem f_properties :
  (∀ x : ℝ, f x > 2 ↔ (x > 1 ∨ x < -5)) ∧
  (∀ t : ℝ, (∀ x : ℝ, f x ≥ t^2 - (11/2)*t) ↔ (1/2 ≤ t ∧ t ≤ 5)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l655_65511


namespace NUMINAMATH_CALUDE_temperature_sum_l655_65594

theorem temperature_sum (t1 t2 t3 k1 k2 k3 : ℚ) : 
  t1 = 5 / 9 * (k1 - 32) →
  t2 = 5 / 9 * (k2 - 32) →
  t3 = 5 / 9 * (k3 - 32) →
  t1 = 105 →
  t2 = 80 →
  t3 = 45 →
  k1 + k2 + k3 = 510 := by
sorry

end NUMINAMATH_CALUDE_temperature_sum_l655_65594


namespace NUMINAMATH_CALUDE_bikes_total_price_l655_65516

/-- The total price of Marion's and Stephanie's bikes -/
def total_price (marion_price stephanie_price : ℕ) : ℕ :=
  marion_price + stephanie_price

/-- Theorem stating the total price of Marion's and Stephanie's bikes -/
theorem bikes_total_price :
  ∃ (marion_price stephanie_price : ℕ),
    marion_price = 356 ∧
    stephanie_price = 2 * marion_price ∧
    total_price marion_price stephanie_price = 1068 :=
by
  sorry


end NUMINAMATH_CALUDE_bikes_total_price_l655_65516


namespace NUMINAMATH_CALUDE_string_cutting_game_winner_first_player_wins_iff_sum_odd_l655_65531

/-- Represents the result of the string-cutting game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- The string-cutting game on an m × n grid -/
def stringCuttingGame (m n : ℕ) : GameResult :=
  if (m + n) % 2 = 0 then
    GameResult.SecondPlayerWins
  else
    GameResult.FirstPlayerWins

/-- Theorem stating the winning condition for the string-cutting game -/
theorem string_cutting_game_winner (m n : ℕ) :
  stringCuttingGame m n = 
    if (m + n) % 2 = 0 then
      GameResult.SecondPlayerWins
    else
      GameResult.FirstPlayerWins := by
  sorry

/-- Corollary: The first player wins if and only if m + n is odd -/
theorem first_player_wins_iff_sum_odd (m n : ℕ) :
  stringCuttingGame m n = GameResult.FirstPlayerWins ↔ (m + n) % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_string_cutting_game_winner_first_player_wins_iff_sum_odd_l655_65531


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l655_65530

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l655_65530


namespace NUMINAMATH_CALUDE_farm_horses_cows_l655_65539

theorem farm_horses_cows (h c : ℕ) : 
  h = 4 * c →                             -- Initial ratio of horses to cows is 4:1
  (h - 15) / (c + 15) = 7 / 3 →           -- New ratio after transaction is 7:3
  (h - 15) - (c + 15) = 60 :=              -- Difference after transaction is 60
by
  sorry

end NUMINAMATH_CALUDE_farm_horses_cows_l655_65539


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_negative_one_l655_65569

open Real

theorem sin_cos_sum_equals_negative_one :
  sin (200 * π / 180) * cos (110 * π / 180) + cos (160 * π / 180) * sin (70 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_negative_one_l655_65569


namespace NUMINAMATH_CALUDE_time_at_6_oclock_l655_65549

/-- Represents a clock with ticks at each hour -/
structure Clock where
  /-- The time between each tick (in seconds) -/
  tick_interval : ℝ
  /-- The total time for all ticks at 12 o'clock (in seconds) -/
  total_time_at_12 : ℝ

/-- Calculates the time between first and last ticks for a given hour -/
def time_between_ticks (c : Clock) (hour : ℕ) : ℝ :=
  c.tick_interval * (hour - 1)

/-- Theorem stating the time between first and last ticks at 6 o'clock -/
theorem time_at_6_oclock (c : Clock) 
  (h1 : c.total_time_at_12 = 66)
  (h2 : c.tick_interval = c.total_time_at_12 / 11) :
  time_between_ticks c 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_time_at_6_oclock_l655_65549


namespace NUMINAMATH_CALUDE_product_digit_range_l655_65554

theorem product_digit_range : 
  ∀ (a b : ℕ), 
    1 ≤ a ∧ a ≤ 9 → 
    100 ≤ b ∧ b ≤ 999 → 
    (100 ≤ a * b ∧ a * b ≤ 9999) := by
  sorry

end NUMINAMATH_CALUDE_product_digit_range_l655_65554


namespace NUMINAMATH_CALUDE_min_tiles_for_room_l655_65505

/-- Represents the dimensions of a room in centimeters -/
structure Room where
  length : ℕ
  breadth : ℕ

/-- Represents a square tile with a given side length in centimeters -/
structure Tile where
  side : ℕ

/-- Calculates the number of tiles needed to cover a room, including wastage -/
def tilesNeeded (room : Room) (tile : Tile) : ℕ :=
  let roomArea := room.length * room.breadth
  let tileArea := tile.side * tile.side
  let baseTiles := (roomArea + tileArea - 1) / tileArea  -- Ceiling division
  let wastage := (baseTiles * 11 + 9) / 10  -- 10% wastage, rounded up
  baseTiles + wastage

/-- Theorem stating the minimum number of tiles required -/
theorem min_tiles_for_room (room : Room) (tile : Tile) :
  room.length = 888 ∧ room.breadth = 462 ∧ tile.side = 22 →
  tilesNeeded room tile ≥ 933 :=
by sorry

end NUMINAMATH_CALUDE_min_tiles_for_room_l655_65505


namespace NUMINAMATH_CALUDE_necessary_sufficient_condition_l655_65546

theorem necessary_sufficient_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) ↔
  (a ≤ -2 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_necessary_sufficient_condition_l655_65546


namespace NUMINAMATH_CALUDE_grape_juice_percentage_l655_65521

-- Define the initial mixture volume
def initial_volume : ℝ := 30

-- Define the initial percentage of grape juice
def initial_grape_percentage : ℝ := 0.1

-- Define the volume of grape juice added
def added_grape_volume : ℝ := 10

-- Define the resulting percentage of grape juice
def resulting_grape_percentage : ℝ := 0.325

theorem grape_juice_percentage :
  let initial_grape_volume := initial_volume * initial_grape_percentage
  let total_grape_volume := initial_grape_volume + added_grape_volume
  let final_volume := initial_volume + added_grape_volume
  (total_grape_volume / final_volume) = resulting_grape_percentage := by
sorry

end NUMINAMATH_CALUDE_grape_juice_percentage_l655_65521
