import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_monotonicity_l2562_256264

-- Define a polynomial function
variable (P : ℝ → ℝ)

-- Define strict monotonicity
def StrictlyMonotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem polynomial_monotonicity
  (h1 : StrictlyMonotonic (λ x => P (P x)))
  (h2 : StrictlyMonotonic (λ x => P (P (P x))))
  : StrictlyMonotonic P := by
  sorry

end NUMINAMATH_CALUDE_polynomial_monotonicity_l2562_256264


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l2562_256266

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The main theorem stating that if A(a,-2) and B(4,b) are symmetric with respect to the origin, then a-b = -6 -/
theorem symmetric_points_difference (a b : ℝ) : 
  symmetric_wrt_origin a (-2) 4 b → a - b = -6 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l2562_256266


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_of_fraction_l2562_256220

/-- The number of zeros between the decimal point and the first non-zero digit
    in the decimal representation of 7/8000 -/
def zeros_before_first_nonzero : ℕ :=
  3

/-- The fraction we're considering -/
def fraction : ℚ :=
  7 / 8000

theorem zeros_before_first_nonzero_of_fraction :
  zeros_before_first_nonzero = 3 :=
sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_of_fraction_l2562_256220


namespace NUMINAMATH_CALUDE_triangle_medians_l2562_256272

/-- In a triangle with sides b and c, and median m_a to side a,
    the other two medians m_b and m_c can be expressed in terms of b, c, and m_a. -/
theorem triangle_medians (b c m_a : ℝ) (hb : b > 0) (hc : c > 0) (hma : m_a > 0) :
  let m_b := (1/2) * Real.sqrt (3*b^2 + 6*c^2 - 8*m_a^2)
  let m_c := (1/2) * Real.sqrt (6*b^2 + 3*c^2 - 8*m_a^2)
  m_b > 0 ∧ m_c > 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_medians_l2562_256272


namespace NUMINAMATH_CALUDE_men_left_bus_l2562_256259

/-- Represents the state of passengers on the bus --/
structure BusState where
  men : ℕ
  women : ℕ

/-- The initial state of the bus --/
def initialState : BusState :=
  { men := 48, women := 24 }

/-- The final state of the bus after some men leave and 8 women enter --/
def finalState : BusState :=
  { men := 32, women := 32 }

/-- The number of women who entered the bus in city Y --/
def womenEntered : ℕ := 8

theorem men_left_bus (initial : BusState) (final : BusState) :
  initial.men + initial.women = 72 →
  initial.women = initial.men / 2 →
  final.men = final.women →
  final.women = initial.women + womenEntered →
  initial.men - final.men = 16 := by
  sorry

#check men_left_bus initialState finalState

end NUMINAMATH_CALUDE_men_left_bus_l2562_256259


namespace NUMINAMATH_CALUDE_random_events_count_l2562_256260

/-- Represents an event --/
inductive Event
| ClassPresident
| StrongerTeamWins
| BirthdayProblem
| SetInclusion
| PainterDeath
| JulySnow
| EvenSum
| RedLights

/-- Determines if an event is random --/
def isRandomEvent : Event → Bool
| Event.ClassPresident => true
| Event.StrongerTeamWins => true
| Event.BirthdayProblem => true
| Event.SetInclusion => false
| Event.PainterDeath => false
| Event.JulySnow => true
| Event.EvenSum => false
| Event.RedLights => true

/-- List of all events --/
def allEvents : List Event := [
  Event.ClassPresident,
  Event.StrongerTeamWins,
  Event.BirthdayProblem,
  Event.SetInclusion,
  Event.PainterDeath,
  Event.JulySnow,
  Event.EvenSum,
  Event.RedLights
]

/-- Theorem: The number of random events in the list is 5 --/
theorem random_events_count :
  (allEvents.filter isRandomEvent).length = 5 := by sorry

end NUMINAMATH_CALUDE_random_events_count_l2562_256260


namespace NUMINAMATH_CALUDE_violet_balloons_lost_l2562_256269

theorem violet_balloons_lost (initial_violet : ℕ) (remaining_violet : ℕ) 
  (h1 : initial_violet = 7) 
  (h2 : remaining_violet = 4) : 
  initial_violet - remaining_violet = 3 := by
sorry

end NUMINAMATH_CALUDE_violet_balloons_lost_l2562_256269


namespace NUMINAMATH_CALUDE_z_range_l2562_256242

theorem z_range (x y : ℝ) (hx : x ≥ 0) (hxy : y ≥ x) (hsum : 4 * x + 3 * y ≤ 12) :
  let z := (x + 2 * y + 3) / (x + 1)
  2 ≤ z ∧ z ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_z_range_l2562_256242


namespace NUMINAMATH_CALUDE_kim_pizza_purchase_l2562_256214

/-- Given that Kim buys pizzas where each pizza has 12 slices, 
    the total cost is $72, and 5 slices cost $10, 
    prove that Kim bought 3 pizzas. -/
theorem kim_pizza_purchase : 
  ∀ (slices_per_pizza : ℕ) (total_cost : ℚ) (five_slice_cost : ℚ),
    slices_per_pizza = 12 →
    total_cost = 72 →
    five_slice_cost = 10 →
    (total_cost / (slices_per_pizza * (five_slice_cost / 5))) = 3 := by
  sorry

#check kim_pizza_purchase

end NUMINAMATH_CALUDE_kim_pizza_purchase_l2562_256214


namespace NUMINAMATH_CALUDE_units_digit_G_100_l2562_256226

/-- Modified Fermat number -/
def G (n : ℕ) : ℕ := 3^(2^n) + 1

/-- Units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_G_100 : units_digit (G 100) = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_100_l2562_256226


namespace NUMINAMATH_CALUDE_both_selected_probability_l2562_256232

theorem both_selected_probability 
  (ram_prob : ℚ) 
  (ravi_prob : ℚ) 
  (h1 : ram_prob = 6 / 7) 
  (h2 : ravi_prob = 1 / 5) : 
  ram_prob * ravi_prob = 6 / 35 := by
sorry

end NUMINAMATH_CALUDE_both_selected_probability_l2562_256232


namespace NUMINAMATH_CALUDE_wildlife_sanctuary_count_l2562_256276

theorem wildlife_sanctuary_count (total_heads total_legs : ℕ) 
  (h1 : total_heads = 300) 
  (h2 : total_legs = 780) : ∃ (birds insects : ℕ),
  birds + insects = total_heads ∧
  2 * birds + 6 * insects = total_legs ∧
  birds = 255 := by
sorry

end NUMINAMATH_CALUDE_wildlife_sanctuary_count_l2562_256276


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l2562_256289

theorem abs_inequality_equivalence (x : ℝ) : 
  (1 ≤ |x + 3| ∧ |x + 3| ≤ 4) ↔ ((-7 ≤ x ∧ x ≤ -4) ∨ (-2 ≤ x ∧ x ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l2562_256289


namespace NUMINAMATH_CALUDE_function_inequality_implies_k_value_l2562_256206

/-- The function f(x) = x^2 + 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- The theorem stating that if f(x) ≤ kx for all x in (1,5], then k = 36/5 -/
theorem function_inequality_implies_k_value (k : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x ≤ 5 → f x ≤ k * x) → k = 36/5 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_k_value_l2562_256206


namespace NUMINAMATH_CALUDE_circle_equation_and_extrema_l2562_256248

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line x + y + 5 = 0
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 + 5 = 0}

theorem circle_equation_and_extrema 
  (C : ℝ × ℝ) 
  (h1 : C ∈ Line) 
  (h2 : (0, 2) ∈ Circle C ((0 - C.1)^2 + (2 - C.2)^2).sqrt) 
  (h3 : (1, 1) ∈ Circle C ((1 - C.1)^2 + (1 - C.2)^2).sqrt) :
  (∃ (r : ℝ), Circle C r = {p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 + 2)^2 = 25}) ∧ 
  (∀ (P : ℝ × ℝ), P ∈ Circle C ((0 - C.1)^2 + (2 - C.2)^2).sqrt → 
    3 * P.1 - 4 * P.2 ≤ 24) ∧
  (∀ (P : ℝ × ℝ), P ∈ Circle C ((0 - C.1)^2 + (2 - C.2)^2).sqrt → 
    3 * P.1 - 4 * P.2 ≥ -26) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_and_extrema_l2562_256248


namespace NUMINAMATH_CALUDE_reward_function_satisfies_requirements_l2562_256263

theorem reward_function_satisfies_requirements :
  let f : ℝ → ℝ := λ x => 2 * Real.sqrt x - 6
  let domain : Set ℝ := { x | 25 ≤ x ∧ x ≤ 1600 }
  (∀ x ∈ domain, ∀ y ∈ domain, x < y → f x < f y) ∧
  (∀ x ∈ domain, f x ≤ 90) ∧
  (∀ x ∈ domain, f x ≤ x / 5) :=
by sorry

end NUMINAMATH_CALUDE_reward_function_satisfies_requirements_l2562_256263


namespace NUMINAMATH_CALUDE_exam_average_is_36_l2562_256230

/-- The overall average of marks obtained by all boys in an examination. -/
def overall_average (total_boys : ℕ) (passed_boys : ℕ) (avg_passed : ℕ) (avg_failed : ℕ) : ℚ :=
  let failed_boys := total_boys - passed_boys
  ((passed_boys * avg_passed + failed_boys * avg_failed) : ℚ) / total_boys

/-- Theorem stating that the overall average of marks is 36 given the conditions. -/
theorem exam_average_is_36 :
  overall_average 120 105 39 15 = 36 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_is_36_l2562_256230


namespace NUMINAMATH_CALUDE_power_function_quadrants_l2562_256288

/-- A function f(x) = (m^2 - 5m + 7)x^m is a power function with its graph
    distributed in the first and third quadrants if and only if m = 3 -/
theorem power_function_quadrants (m : ℝ) : 
  (∀ x ≠ 0, ∃ f : ℝ → ℝ, f x = (m^2 - 5*m + 7) * x^m) ∧ 
  (∀ x > 0, (m^2 - 5*m + 7) * x^m > 0) ∧
  (∀ x < 0, (m^2 - 5*m + 7) * x^m < 0) ∧
  (m^2 - 5*m + 7 = 1) ↔ 
  m = 3 := by sorry

end NUMINAMATH_CALUDE_power_function_quadrants_l2562_256288


namespace NUMINAMATH_CALUDE_unique_solution_system_l2562_256247

theorem unique_solution_system (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + x*y + y^2) * (y^2 + y*z + z^2) * (z^2 + z*x + x^2) = x*y*z ∧
  (x^4 + x^2*y^2 + y^4) * (y^4 + y^2*z^2 + z^4) * (z^4 + z^2*x^2 + x^4) = x^3*y^3*z^3 →
  x = 1/3 ∧ y = 1/3 ∧ z = 1/3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2562_256247


namespace NUMINAMATH_CALUDE_perpendicular_and_equal_intercepts_l2562_256221

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 4 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def line3 (x y : ℝ) : Prop := 3 * x + 4 * y - 15 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (3, 2)

-- Define the perpendicular line l1
def l1 (x y : ℝ) : Prop := 4 * x - 3 * y - 6 = 0

-- Define the two possible lines with equal intercepts
def l2_case1 (x y : ℝ) : Prop := 2 * x - 3 * y = 0
def l2_case2 (x y : ℝ) : Prop := x + y - 5 = 0

theorem perpendicular_and_equal_intercepts :
  (∀ x y : ℝ, line1 x y ∧ line2 x y → (x, y) = P) →
  (∀ x y : ℝ, l1 x y → (x, y) = P ∨ (3 * x + 4 * y ≠ 15)) ∧
  ((∀ x y : ℝ, l2_case1 x y → (x, y) = P) ∨ (∀ x y : ℝ, l2_case2 x y → (x, y) = P)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_and_equal_intercepts_l2562_256221


namespace NUMINAMATH_CALUDE_money_division_l2562_256290

/-- Represents the share of money for each person -/
structure Share :=
  (a : ℚ)
  (b : ℚ)
  (c : ℚ)

/-- The problem statement and proof -/
theorem money_division (s : Share) : 
  s.c = 64 ∧ 
  s.b = 0.65 * s.a ∧ 
  s.c = 0.40 * s.a → 
  s.a + s.b + s.c = 328 := by
sorry


end NUMINAMATH_CALUDE_money_division_l2562_256290


namespace NUMINAMATH_CALUDE_abc_product_magnitude_l2562_256265

theorem abc_product_magnitude (a b c : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → 
  a ≠ b → b ≠ c → c ≠ a →
  (a + 1 / b^2 = b + 1 / c^2) →
  (b + 1 / c^2 = c + 1 / a^2) →
  |a * b * c| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_magnitude_l2562_256265


namespace NUMINAMATH_CALUDE_rancher_corn_cost_l2562_256216

/-- Represents the rancher's situation --/
structure RancherSituation where
  sheep : Nat
  cattle : Nat
  grassAcres : Nat
  grassPerCowPerMonth : Nat
  grassPerSheepPerMonth : Nat
  monthsPerBagForCow : Nat
  monthsPerBagForSheep : Nat
  cornBagPrice : Nat

/-- Calculates the yearly cost of feed corn for the rancher --/
def yearlyCornCost (s : RancherSituation) : Nat :=
  let totalGrassPerMonth := s.cattle * s.grassPerCowPerMonth + s.sheep * s.grassPerSheepPerMonth
  let grazingMonths := s.grassAcres / totalGrassPerMonth
  let cornMonths := 12 - grazingMonths
  let cornForSheep := (cornMonths * s.sheep + s.monthsPerBagForSheep - 1) / s.monthsPerBagForSheep
  let cornForCattle := cornMonths * s.cattle / s.monthsPerBagForCow
  (cornForSheep + cornForCattle) * s.cornBagPrice

/-- Theorem stating that the rancher needs to spend $360 on feed corn each year --/
theorem rancher_corn_cost :
  let s : RancherSituation := {
    sheep := 8,
    cattle := 5,
    grassAcres := 144,
    grassPerCowPerMonth := 2,
    grassPerSheepPerMonth := 1,
    monthsPerBagForCow := 1,
    monthsPerBagForSheep := 2,
    cornBagPrice := 10
  }
  yearlyCornCost s = 360 := by sorry

end NUMINAMATH_CALUDE_rancher_corn_cost_l2562_256216


namespace NUMINAMATH_CALUDE_proposition_truth_l2562_256274

theorem proposition_truth (p q : Prop) (hp : p) (hq : ¬q) : (¬p) ∨ (¬q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l2562_256274


namespace NUMINAMATH_CALUDE_tangent_line_equation_y_coordinate_range_min_length_NQ_l2562_256255

noncomputable section

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l
def line_l (t x : ℝ) : Prop := x = t

-- Define the condition for t
def t_condition (t : ℝ) : Prop := 1 < t ∧ t < 2

-- Define point P
def point_P (t y : ℝ) : Prop := line_l t t ∧ y > 0

-- Define the distance between two points
def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem tangent_line_equation (t : ℝ) (h1 : t = 4/3) (h2 : t_condition t) :
  ∃ (x y : ℝ), point_P t y ∧ distance 0 0 t y = 5/3 →
  (y = 1 ∨ 24*x - 7*y - 25 = 0) :=
sorry

theorem y_coordinate_range (t : ℝ) (h : t = 4/3) :
  ∃ (y : ℝ), point_P t y ∧ 
  (∃ (x1 y1 x2 y2 : ℝ), circle_O x1 y1 ∧ circle_O x2 y2 ∧
   2*x2 = x1 + t ∧ 2*y2 = y1 + y) →
  -Real.sqrt (65/9) ≤ y ∧ y ≤ Real.sqrt (65/9) :=
sorry

theorem min_length_NQ (t : ℝ) (h : t_condition t) :
  ∃ (x_R y_R x_N y_N x_Q y_Q : ℝ),
  circle_O x_R y_R ∧
  distance x_R y_R t 0 = 1 ∧
  circle_O x_N y_N ∧
  (y_N - 0) / (x_N - t) = (y_R - 0) / (x_R - t) ∧
  x_Q = t/2 ∧ y_Q = 0 →
  ∀ (t' : ℝ), t_condition t' →
  distance x_N y_N x_Q y_Q ≥ Real.sqrt 14 / 8 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_y_coordinate_range_min_length_NQ_l2562_256255


namespace NUMINAMATH_CALUDE_jeff_donuts_per_day_l2562_256211

/-- The number of days Jeff makes donuts -/
def days : ℕ := 12

/-- The number of donuts Jeff eats per day -/
def jeff_eats_per_day : ℕ := 1

/-- The total number of donuts Chris eats -/
def chris_eats_total : ℕ := 8

/-- The number of donuts that fit in each box -/
def donuts_per_box : ℕ := 10

/-- The number of boxes Jeff can fill -/
def boxes_filled : ℕ := 10

/-- The number of donuts Jeff makes each day -/
def donuts_per_day : ℕ := 10

theorem jeff_donuts_per_day :
  ∃ (d : ℕ), 
    d * days - (jeff_eats_per_day * days) - chris_eats_total = boxes_filled * donuts_per_box ∧
    d = donuts_per_day :=
by sorry

end NUMINAMATH_CALUDE_jeff_donuts_per_day_l2562_256211


namespace NUMINAMATH_CALUDE_golden_section_proportion_l2562_256294

/-- Golden section point of a line segment -/
def is_golden_section_point (A B C : ℝ) : Prop :=
  (B - A) / (C - A) = (C - A) / (B - C)

theorem golden_section_proportion (A B C : ℝ) 
  (h1 : is_golden_section_point A B C) 
  (h2 : C - A > B - C) : 
  (B - A) / (C - A) = (C - A) / (B - C) := by
  sorry

end NUMINAMATH_CALUDE_golden_section_proportion_l2562_256294


namespace NUMINAMATH_CALUDE_solve_system_for_q_l2562_256215

theorem solve_system_for_q : 
  ∀ p q : ℚ, 3 * p + 4 * q = 8 → 4 * p + 3 * q = 13 → q = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_for_q_l2562_256215


namespace NUMINAMATH_CALUDE_set_equality_l2562_256296

theorem set_equality (A B X : Set α) 
  (h1 : A ∪ B ∪ X = A ∪ B) 
  (h2 : A ∩ X = A ∩ B) 
  (h3 : B ∩ X = A ∩ B) : 
  X = A ∩ B := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l2562_256296


namespace NUMINAMATH_CALUDE_smallest_prime_angle_in_right_triangle_l2562_256202

/-- Checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- Theorem: Smallest angle b in a right triangle with prime angles -/
theorem smallest_prime_angle_in_right_triangle :
  ∀ a b : ℕ,
  (a : ℝ) + (b : ℝ) = 90 →
  isPrime a →
  isPrime b →
  (a : ℝ) > (b : ℝ) + 2 →
  b ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_angle_in_right_triangle_l2562_256202


namespace NUMINAMATH_CALUDE_billy_free_time_l2562_256298

/-- Proves that Billy has 16 hours of free time each day of the weekend given the specified conditions. -/
theorem billy_free_time (video_game_percentage : ℝ) (reading_percentage : ℝ)
  (pages_per_hour : ℕ) (pages_per_book : ℕ) (books_read : ℕ) :
  video_game_percentage = 0.75 →
  reading_percentage = 0.25 →
  pages_per_hour = 60 →
  pages_per_book = 80 →
  books_read = 3 →
  (books_read * pages_per_book : ℝ) / pages_per_hour / reading_percentage = 16 :=
by sorry

end NUMINAMATH_CALUDE_billy_free_time_l2562_256298


namespace NUMINAMATH_CALUDE_percent_change_condition_l2562_256210

theorem percent_change_condition (a b r N : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < r ∧ 0 < N ∧ r < 50 →
  (N * (1 + a / 100) * (1 - b / 100) ≤ N * (1 + r / 100) ↔ a - b - a * b / 100 ≤ r) :=
by sorry

end NUMINAMATH_CALUDE_percent_change_condition_l2562_256210


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2562_256240

theorem complex_expression_equality : 
  let a : ℂ := 3 - 2*I
  let b : ℂ := 2 + 3*I
  3*a + 4*b = 17 + 6*I := by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2562_256240


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l2562_256277

theorem no_real_roots_quadratic : 
  {x : ℝ | x^2 - x + 1 = 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l2562_256277


namespace NUMINAMATH_CALUDE_triangle_similarity_l2562_256244

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents an excircle of a triangle -/
structure Excircle where
  center : Point
  radius : ℝ

/-- Defines if a triangle is acute and scalene -/
def isAcuteScalene (t : Triangle) : Prop := sorry

/-- Defines the C-excircle of a triangle -/
def cExcircle (t : Triangle) : Excircle := sorry

/-- Defines the B-excircle of a triangle -/
def bExcircle (t : Triangle) : Excircle := sorry

/-- Defines the intersection point of two lines -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

/-- Defines a point symmetric to another point with respect to a third point -/
def symmetricPoint (p center : Point) : Point := sorry

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop := sorry

/-- Main theorem -/
theorem triangle_similarity (t : Triangle) (h : isAcuteScalene t) :
  let c_ex := cExcircle t
  let b_ex := bExcircle t
  let M := sorry -- Point where C-excircle is tangent to AB
  let N := sorry -- Point where C-excircle is tangent to extension of BC
  let P := sorry -- Point where B-excircle is tangent to AC
  let Q := sorry -- Point where B-excircle is tangent to extension of BC
  let A1 := lineIntersection M N P Q
  let A2 := symmetricPoint t.A A1
  let B1 := sorry -- Defined analogously to A1
  let B2 := symmetricPoint t.B B1
  let C1 := sorry -- Defined analogously to A1
  let C2 := symmetricPoint t.C C1
  let t2 : Triangle := ⟨A2, B2, C2⟩
  areSimilar t t2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_l2562_256244


namespace NUMINAMATH_CALUDE_jamie_hourly_rate_l2562_256252

/-- Represents Jamie's flyer delivery job -/
structure FlyerJob where
  days_per_week : ℕ
  hours_per_day : ℕ
  total_weeks : ℕ
  total_earnings : ℕ

/-- Calculates the hourly rate given a flyer delivery job -/
def hourly_rate (job : FlyerJob) : ℚ :=
  job.total_earnings / (job.days_per_week * job.hours_per_day * job.total_weeks)

/-- Theorem stating that Jamie's hourly rate is $10 -/
theorem jamie_hourly_rate :
  let job : FlyerJob := {
    days_per_week := 2,
    hours_per_day := 3,
    total_weeks := 6,
    total_earnings := 360
  }
  hourly_rate job = 10 := by sorry

end NUMINAMATH_CALUDE_jamie_hourly_rate_l2562_256252


namespace NUMINAMATH_CALUDE_mark_leftover_money_l2562_256281

-- Define the given conditions
def old_hourly_wage : ℝ := 40
def raise_percentage : ℝ := 0.05
def hours_per_day : ℝ := 8
def days_per_week : ℝ := 5
def old_weekly_bills : ℝ := 600
def personal_trainer_cost : ℝ := 100

-- Define the calculation steps
def new_hourly_wage : ℝ := old_hourly_wage * (1 + raise_percentage)
def weekly_hours : ℝ := hours_per_day * days_per_week
def weekly_earnings : ℝ := new_hourly_wage * weekly_hours
def new_weekly_expenses : ℝ := old_weekly_bills + personal_trainer_cost

-- Theorem to prove
theorem mark_leftover_money :
  weekly_earnings - new_weekly_expenses = 980 := by
  sorry


end NUMINAMATH_CALUDE_mark_leftover_money_l2562_256281


namespace NUMINAMATH_CALUDE_power_sum_difference_l2562_256236

theorem power_sum_difference : 3^(1+2+3+4) - (3^1 + 3^2 + 3^3 + 3^4) = 58929 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l2562_256236


namespace NUMINAMATH_CALUDE_tenth_pirate_coins_l2562_256268

/-- Represents the number of pirates --/
def num_pirates : ℕ := 10

/-- Represents the initial number of silver coins --/
def initial_silver : ℕ := 1050

/-- Represents the number of silver coins each pirate takes --/
def silver_per_pirate : ℕ := 100

/-- Calculates the remaining gold coins after k pirates have taken their share --/
def remaining_gold (initial_gold : ℕ) (k : ℕ) : ℚ :=
  (num_pirates - k : ℚ) / num_pirates * initial_gold

/-- Calculates the number of gold coins the 10th pirate receives --/
def gold_for_last_pirate (initial_gold : ℕ) : ℚ :=
  remaining_gold initial_gold (num_pirates - 1)

/-- Calculates the number of silver coins the 10th pirate receives --/
def silver_for_last_pirate : ℕ :=
  initial_silver - (num_pirates - 1) * silver_per_pirate

/-- Theorem stating that the 10th pirate receives 494 coins in total --/
theorem tenth_pirate_coins (initial_gold : ℕ) :
  ∃ (gold_coins : ℕ), gold_for_last_pirate initial_gold = gold_coins ∧
  gold_coins + silver_for_last_pirate = 494 :=
sorry

end NUMINAMATH_CALUDE_tenth_pirate_coins_l2562_256268


namespace NUMINAMATH_CALUDE_prob_six_queen_ace_l2562_256223

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of cards of each rank (e.g., 6, Queen, Ace) in a standard deck -/
def CardsPerRank : ℕ := 4

/-- Probability of drawing a specific sequence of three cards from a standard deck -/
def prob_specific_sequence (deck_size : ℕ) (cards_per_rank : ℕ) : ℚ :=
  (cards_per_rank : ℚ) / deck_size *
  (cards_per_rank : ℚ) / (deck_size - 1) *
  (cards_per_rank : ℚ) / (deck_size - 2)

theorem prob_six_queen_ace :
  prob_specific_sequence StandardDeck CardsPerRank = 16 / 33150 := by
  sorry

end NUMINAMATH_CALUDE_prob_six_queen_ace_l2562_256223


namespace NUMINAMATH_CALUDE_min_area_triangle_AOB_l2562_256287

/-- The minimum area of triangle AOB given the specified conditions -/
theorem min_area_triangle_AOB (m n : ℝ) : 
  (∃ (x y : ℝ), mx + ny - 1 = 0 ∧ x^2 + y^2 = 4) →  -- line l intersects circle
  (∃ (x1 x2 y1 y2 : ℝ), 
    mx1 + ny1 - 1 = 0 ∧ x1^2 + y1^2 = 4 ∧   -- first intersection point
    mx2 + ny2 - 1 = 0 ∧ x2^2 + y2^2 = 4 ∧   -- second intersection point
    (x1 - x2)^2 + (y1 - y2)^2 = 4) →        -- chord length is 2
  (∀ (a b : ℝ), ma - 1 = 0 → nb - 1 = 0 →   -- A and B coordinates
    1/2 * |a| * |b| ≥ 3 ∧                   -- area of triangle AOB
    (∃ (a' b' : ℝ), ma' - 1 = 0 ∧ nb' - 1 = 0 ∧ 1/2 * |a'| * |b'| = 3)) -- minimum exists
  := by sorry

end NUMINAMATH_CALUDE_min_area_triangle_AOB_l2562_256287


namespace NUMINAMATH_CALUDE_sum_lent_is_500_l2562_256209

/-- The sum of money lent -/
def P : ℝ := 500

/-- The annual interest rate as a decimal -/
def R : ℝ := 0.04

/-- The time period in years -/
def T : ℝ := 8

/-- The simple interest formula -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

theorem sum_lent_is_500 : 
  simple_interest P R T = P - 340 → P = 500 := by
  sorry

end NUMINAMATH_CALUDE_sum_lent_is_500_l2562_256209


namespace NUMINAMATH_CALUDE_problem_statement_l2562_256256

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  (2/(a-1) + 1/(b-2) ≥ 2) ∧ (2*a + b ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2562_256256


namespace NUMINAMATH_CALUDE_adam_first_year_students_l2562_256261

/-- The number of students Adam teaches per year after the first year -/
def students_per_year : ℕ := 50

/-- The total number of years Adam teaches -/
def total_years : ℕ := 10

/-- The total number of students Adam teaches in 10 years -/
def total_students : ℕ := 490

/-- The number of students Adam taught in the first year -/
def first_year_students : ℕ := total_students - (students_per_year * (total_years - 1))

theorem adam_first_year_students :
  first_year_students = 40 := by sorry

end NUMINAMATH_CALUDE_adam_first_year_students_l2562_256261


namespace NUMINAMATH_CALUDE_abs_fraction_inequality_l2562_256249

theorem abs_fraction_inequality (x : ℝ) :
  |((3 - x) / 4)| < 1 ↔ 2 < x ∧ x < 7 := by sorry

end NUMINAMATH_CALUDE_abs_fraction_inequality_l2562_256249


namespace NUMINAMATH_CALUDE_lilliputian_matchboxes_theorem_l2562_256200

/-- The scale factor between Gulliver's homeland and Lilliput -/
def scale_factor : ℕ := 12

/-- The number of Lilliputian matchboxes that can fit into a matchbox from Gulliver's homeland -/
def lilliputian_matchboxes_count : ℕ := scale_factor ^ 3

/-- Theorem stating that the number of Lilliputian matchboxes that can fit into a matchbox from Gulliver's homeland is 1728 -/
theorem lilliputian_matchboxes_theorem : lilliputian_matchboxes_count = 1728 := by
  sorry

end NUMINAMATH_CALUDE_lilliputian_matchboxes_theorem_l2562_256200


namespace NUMINAMATH_CALUDE_frank_work_hours_l2562_256228

/-- The number of hours Frank worked per day -/
def hours_per_day : ℕ := 8

/-- The number of days Frank worked -/
def days_worked : ℕ := 4

/-- The total number of hours Frank worked -/
def total_hours : ℕ := hours_per_day * days_worked

theorem frank_work_hours : total_hours = 32 := by
  sorry

end NUMINAMATH_CALUDE_frank_work_hours_l2562_256228


namespace NUMINAMATH_CALUDE_valid_a_values_l2562_256275

def A : Set ℝ := {-1, 1/2, 1}

def B (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 = 1 ∧ a ≥ 0}

def full_eating (X Y : Set ℝ) : Prop := X ⊆ Y ∨ Y ⊆ X

def partial_eating (X Y : Set ℝ) : Prop :=
  (∃ x, x ∈ X ∩ Y) ∧ ¬(X ⊆ Y) ∧ ¬(Y ⊆ X)

def valid_relationship (a : ℝ) : Prop :=
  full_eating A (B a) ∨ partial_eating A (B a)

theorem valid_a_values : {a : ℝ | valid_relationship a} = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_valid_a_values_l2562_256275


namespace NUMINAMATH_CALUDE_abs_four_minus_xy_gt_two_abs_x_minus_y_l2562_256219

theorem abs_four_minus_xy_gt_two_abs_x_minus_y 
  (x y : ℝ) (hx : |x| < 2) (hy : |y| < 2) : 
  |4 - x * y| > 2 * |x - y| := by
  sorry

end NUMINAMATH_CALUDE_abs_four_minus_xy_gt_two_abs_x_minus_y_l2562_256219


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2562_256218

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 6 + 2 * a 4 * a 5 + a 5 ^ 2 = 25 →
  a 4 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2562_256218


namespace NUMINAMATH_CALUDE_milton_zoology_books_l2562_256286

theorem milton_zoology_books : 
  ∀ (z b : ℕ), 
    z + b = 80 → 
    b = 4 * z → 
    z = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_milton_zoology_books_l2562_256286


namespace NUMINAMATH_CALUDE_negation_of_forall_greater_than_five_l2562_256293

theorem negation_of_forall_greater_than_five (S : Set ℝ) :
  (¬ ∀ x ∈ S, x > 5) ↔ (∃ x ∈ S, x ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_greater_than_five_l2562_256293


namespace NUMINAMATH_CALUDE_probability_one_of_each_interpreter_l2562_256257

def team_size : ℕ := 5
def english_interpreters : ℕ := 3
def russian_interpreters : ℕ := 2

theorem probability_one_of_each_interpreter :
  let total_combinations := Nat.choose team_size 2
  let favorable_combinations := Nat.choose english_interpreters 1 * Nat.choose russian_interpreters 1
  (favorable_combinations : ℚ) / total_combinations = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_of_each_interpreter_l2562_256257


namespace NUMINAMATH_CALUDE_complex_equation_ratio_l2562_256267

theorem complex_equation_ratio (a b : ℝ) : 
  (Complex.mk a b) * (Complex.mk 1 1) = Complex.mk 7 (-3) → a / b = -2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_ratio_l2562_256267


namespace NUMINAMATH_CALUDE_complex_exponentiation_165_deg_60_l2562_256227

theorem complex_exponentiation_165_deg_60 : 
  (Complex.exp (Complex.I * Real.pi * 165 / 180)) ^ 60 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_exponentiation_165_deg_60_l2562_256227


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2562_256299

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q) →  -- geometric sequence
  a 2 = 1 →  -- a_2 = 1
  a 8 = a 6 + 6 * a 4 →  -- a_8 = a_6 + 6a_4
  a 3 = Real.sqrt 3 :=  -- a_3 = √3
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2562_256299


namespace NUMINAMATH_CALUDE_limit_cosine_ratio_l2562_256201

theorem limit_cosine_ratio : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
    |(1 - Real.cos (2*x)) / (Real.cos (7*x) - Real.cos (3*x)) + (1/10)| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_cosine_ratio_l2562_256201


namespace NUMINAMATH_CALUDE_cheerleader_ratio_is_half_l2562_256245

/-- Represents the number of cheerleaders for each uniform size -/
structure CheerleaderCounts where
  total : ℕ
  size2 : ℕ
  size6 : ℕ
  size12 : ℕ

/-- The ratio of cheerleaders needing size 12 to those needing size 6 -/
def size12to6Ratio (counts : CheerleaderCounts) : ℚ :=
  counts.size12 / counts.size6

/-- Theorem stating the ratio of cheerleaders needing size 12 to those needing size 6 -/
theorem cheerleader_ratio_is_half (counts : CheerleaderCounts)
  (h_total : counts.total = 19)
  (h_size2 : counts.size2 = 4)
  (h_size6 : counts.size6 = 10)
  (h_sum : counts.total = counts.size2 + counts.size6 + counts.size12) :
  size12to6Ratio counts = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cheerleader_ratio_is_half_l2562_256245


namespace NUMINAMATH_CALUDE_power_expression_equality_l2562_256254

theorem power_expression_equality (c d : ℝ) 
  (h1 : (80 : ℝ) ^ c = 4) 
  (h2 : (80 : ℝ) ^ d = 5) : 
  (16 : ℝ) ^ ((1 - c - d) / (2 * (1 - d))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_expression_equality_l2562_256254


namespace NUMINAMATH_CALUDE_first_digit_powers_of_3_and_7_l2562_256297

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def first_digit (n : ℕ) : ℕ :=
  if n < 10 then n else first_digit (n / 10)

theorem first_digit_powers_of_3_and_7 :
  ∃ (m n : ℕ), is_three_digit (3^m) ∧ is_three_digit (7^n) ∧ 
  first_digit (3^m) = first_digit (7^n) ∧
  first_digit (3^m) = 3 ∧
  ∀ (k : ℕ), k ≠ 3 → 
    ¬(∃ (p q : ℕ), is_three_digit (3^p) ∧ is_three_digit (7^q) ∧ 
    first_digit (3^p) = first_digit (7^q) ∧ first_digit (3^p) = k) :=
by sorry

end NUMINAMATH_CALUDE_first_digit_powers_of_3_and_7_l2562_256297


namespace NUMINAMATH_CALUDE_distance_is_15_miles_l2562_256285

/-- Represents the walking scenario with distance, speed, and time. -/
structure WalkScenario where
  distance : ℝ
  speed : ℝ
  time : ℝ

/-- The original walking scenario. -/
def original : WalkScenario := sorry

/-- The scenario with increased speed. -/
def increased_speed : WalkScenario := sorry

/-- The scenario with decreased speed. -/
def decreased_speed : WalkScenario := sorry

theorem distance_is_15_miles :
  (∀ s : WalkScenario, s.distance = s.speed * s.time) →
  (increased_speed.speed = original.speed + 0.5) →
  (increased_speed.time = 4/5 * original.time) →
  (decreased_speed.speed = original.speed - 0.5) →
  (decreased_speed.time = original.time + 2.5) →
  (original.distance = increased_speed.distance) →
  (original.distance = decreased_speed.distance) →
  original.distance = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_15_miles_l2562_256285


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2562_256258

theorem complex_fraction_simplification :
  (2 : ℂ) / (1 + Complex.I) = 1 - Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2562_256258


namespace NUMINAMATH_CALUDE_y_derivative_l2562_256273

noncomputable def y (x : ℝ) : ℝ := 
  -1 / (3 * (Real.sin x)^3) - 1 / (Real.sin x) + (1/2) * Real.log ((1 + Real.sin x) / (1 - Real.sin x))

theorem y_derivative (x : ℝ) (hx : Real.cos x ≠ 0) (hsx : Real.sin x ≠ 0) : 
  deriv y x = 1 / (Real.cos x * (Real.sin x)^4) :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l2562_256273


namespace NUMINAMATH_CALUDE_jeremy_tylenol_duration_l2562_256241

/-- Calculates the duration in days for which Jeremy takes Tylenol -/
def tylenol_duration (dose_mg : ℕ) (dose_interval_hours : ℕ) (total_pills : ℕ) (mg_per_pill : ℕ) : ℕ :=
  let total_mg := total_pills * mg_per_pill
  let total_doses := total_mg / dose_mg
  let total_hours := total_doses * dose_interval_hours
  total_hours / 24

/-- Theorem stating that Jeremy takes Tylenol for 14 days -/
theorem jeremy_tylenol_duration :
  tylenol_duration 1000 6 112 500 = 14 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_tylenol_duration_l2562_256241


namespace NUMINAMATH_CALUDE_correct_statements_count_l2562_256212

/-- Represents the correctness of a statement -/
inductive Correctness
| correct
| incorrect

/-- Evaluates the correctness of statement 1 -/
def statement1 : Correctness := Correctness.correct

/-- Evaluates the correctness of statement 2 -/
def statement2 : Correctness := Correctness.incorrect

/-- Evaluates the correctness of statement 3 -/
def statement3 : Correctness := Correctness.incorrect

/-- Evaluates the correctness of statement 4 -/
def statement4 : Correctness := Correctness.correct

/-- Counts the number of correct statements -/
def countCorrect (s1 s2 s3 s4 : Correctness) : Nat :=
  match s1, s2, s3, s4 with
  | Correctness.correct, Correctness.correct, Correctness.correct, Correctness.correct => 4
  | Correctness.correct, Correctness.correct, Correctness.correct, Correctness.incorrect => 3
  | Correctness.correct, Correctness.correct, Correctness.incorrect, Correctness.correct => 3
  | Correctness.correct, Correctness.correct, Correctness.incorrect, Correctness.incorrect => 2
  | Correctness.correct, Correctness.incorrect, Correctness.correct, Correctness.correct => 3
  | Correctness.correct, Correctness.incorrect, Correctness.correct, Correctness.incorrect => 2
  | Correctness.correct, Correctness.incorrect, Correctness.incorrect, Correctness.correct => 2
  | Correctness.correct, Correctness.incorrect, Correctness.incorrect, Correctness.incorrect => 1
  | Correctness.incorrect, Correctness.correct, Correctness.correct, Correctness.correct => 3
  | Correctness.incorrect, Correctness.correct, Correctness.correct, Correctness.incorrect => 2
  | Correctness.incorrect, Correctness.correct, Correctness.incorrect, Correctness.correct => 2
  | Correctness.incorrect, Correctness.correct, Correctness.incorrect, Correctness.incorrect => 1
  | Correctness.incorrect, Correctness.incorrect, Correctness.correct, Correctness.correct => 2
  | Correctness.incorrect, Correctness.incorrect, Correctness.correct, Correctness.incorrect => 1
  | Correctness.incorrect, Correctness.incorrect, Correctness.incorrect, Correctness.correct => 1
  | Correctness.incorrect, Correctness.incorrect, Correctness.incorrect, Correctness.incorrect => 0

theorem correct_statements_count :
  countCorrect statement1 statement2 statement3 statement4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_statements_count_l2562_256212


namespace NUMINAMATH_CALUDE_worker_pay_calculation_l2562_256213

/-- Calculates the total pay for a worker given their regular pay rate, 
    regular hours, overtime hours, and overtime pay rate multiplier. -/
def totalPay (regularRate : ℝ) (regularHours : ℝ) (overtimeHours : ℝ) (overtimeMultiplier : ℝ) : ℝ :=
  regularRate * regularHours + regularRate * overtimeMultiplier * overtimeHours

theorem worker_pay_calculation :
  let regularRate : ℝ := 3
  let regularHours : ℝ := 40
  let overtimeHours : ℝ := 8
  let overtimeMultiplier : ℝ := 2
  totalPay regularRate regularHours overtimeHours overtimeMultiplier = 168 := by
sorry

end NUMINAMATH_CALUDE_worker_pay_calculation_l2562_256213


namespace NUMINAMATH_CALUDE_jerseys_sold_equals_tshirts_sold_l2562_256291

theorem jerseys_sold_equals_tshirts_sold (jersey_profit : ℕ) (tshirt_profit : ℕ) 
  (tshirts_sold : ℕ) (jersey_cost_difference : ℕ) :
  jersey_profit = 115 →
  tshirt_profit = 25 →
  tshirts_sold = 113 →
  jersey_cost_difference = 90 →
  jersey_profit = tshirt_profit + jersey_cost_difference →
  ∃ (jerseys_sold : ℕ), jerseys_sold = tshirts_sold :=
by sorry


end NUMINAMATH_CALUDE_jerseys_sold_equals_tshirts_sold_l2562_256291


namespace NUMINAMATH_CALUDE_opposite_sides_inequality_l2562_256224

/-- Given two points P and A on opposite sides of a line, prove that P satisfies a specific inequality --/
theorem opposite_sides_inequality (x y : ℝ) :
  (3*x + 2*y - 8) * (3*1 + 2*2 - 8) < 0 →
  3*x + 2*y > 8 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_inequality_l2562_256224


namespace NUMINAMATH_CALUDE_parabola_intersection_probability_l2562_256203

/-- Represents the outcome of rolling a fair six-sided die -/
inductive SixSidedDie : Type
  | one | two | three | four | five | six

/-- Represents the outcome of rolling a fair four-sided die (2 to 5) -/
inductive FourSidedDie : Type
  | two | three | four | five

/-- Represents a parabola of the form y = x^2 + ax + b -/
structure Parabola1 where
  a : SixSidedDie
  b : SixSidedDie

/-- Represents a parabola of the form y = x^2 + px^2 + cx + d -/
structure Parabola2 where
  p : FourSidedDie
  c : SixSidedDie
  d : SixSidedDie

/-- Returns true if two parabolas intersect -/
def intersect (p1 : Parabola1) (p2 : Parabola2) : Bool :=
  sorry

/-- Probability that two randomly chosen parabolas intersect -/
def intersection_probability : ℚ :=
  sorry

theorem parabola_intersection_probability :
  intersection_probability = 209 / 216 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_probability_l2562_256203


namespace NUMINAMATH_CALUDE_fifth_number_21st_row_l2562_256246

/-- Represents the triangular array of numbers -/
def TriangularArray (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem fifth_number_21st_row :
  (∀ n : ℕ, TriangularArray n n = n^2) →
  (∀ n k : ℕ, k < n → TriangularArray n (k+1) = TriangularArray n k + 1) →
  (∀ n : ℕ, TriangularArray (n+1) 1 = TriangularArray n n + 1) →
  TriangularArray 21 5 = 405 :=
sorry

end NUMINAMATH_CALUDE_fifth_number_21st_row_l2562_256246


namespace NUMINAMATH_CALUDE_xy_equation_solution_l2562_256295

theorem xy_equation_solution (x y : ℕ+) (p q : ℕ) :
  x ≥ y →
  x * y - (x + y) = 2 * p + q →
  p = Nat.gcd x y →
  q = Nat.lcm x y →
  ((x = 9 ∧ y = 3) ∨ (x = 5 ∧ y = 5)) :=
by sorry

end NUMINAMATH_CALUDE_xy_equation_solution_l2562_256295


namespace NUMINAMATH_CALUDE_triangle_square_perimeter_difference_l2562_256217

theorem triangle_square_perimeter_difference (d : ℤ) : 
  (∃ (t s : ℝ), 3 * t - 4 * s = 1575 ∧ t - s = d ∧ s > 0) ↔ d > 525 :=
sorry

end NUMINAMATH_CALUDE_triangle_square_perimeter_difference_l2562_256217


namespace NUMINAMATH_CALUDE_pie_difference_l2562_256270

/-- The number of apple pies baked on Mondays and Fridays -/
def monday_friday_apple : ℕ := 16

/-- The number of apple pies baked on Wednesdays -/
def wednesday_apple : ℕ := 20

/-- The number of cherry pies baked on Tuesdays -/
def tuesday_cherry : ℕ := 14

/-- The number of cherry pies baked on Thursdays -/
def thursday_cherry : ℕ := 18

/-- The number of apple pies baked on Saturdays -/
def saturday_apple : ℕ := 10

/-- The number of cherry pies baked on Saturdays -/
def saturday_cherry : ℕ := 8

/-- The number of apple pies baked on Sundays -/
def sunday_apple : ℕ := 6

/-- The number of cherry pies baked on Sundays -/
def sunday_cherry : ℕ := 12

/-- The total number of apple pies baked in one week -/
def total_apple : ℕ := 2 * monday_friday_apple + wednesday_apple + saturday_apple + sunday_apple

/-- The total number of cherry pies baked in one week -/
def total_cherry : ℕ := tuesday_cherry + thursday_cherry + saturday_cherry + sunday_cherry

theorem pie_difference : total_apple - total_cherry = 16 := by
  sorry

end NUMINAMATH_CALUDE_pie_difference_l2562_256270


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l2562_256278

theorem fraction_equals_zero (x : ℝ) :
  x = 3 → (2 * x - 6) / (5 * x + 10) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l2562_256278


namespace NUMINAMATH_CALUDE_justine_coloring_ratio_l2562_256262

/-- Given a total number of sheets, number of binders, and sheets used by Justine,
    prove that the ratio of sheets Justine colored to total sheets in her binder is 1:2 -/
theorem justine_coloring_ratio 
  (total_sheets : ℕ) 
  (num_binders : ℕ) 
  (sheets_used : ℕ) 
  (h1 : total_sheets = 2450)
  (h2 : num_binders = 5)
  (h3 : sheets_used = 245)
  : (sheets_used : ℚ) / (total_sheets / num_binders) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_justine_coloring_ratio_l2562_256262


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2562_256251

theorem inequality_solution_set (x : ℝ) :
  (1 / (x + 2) + 4 / (x + 4) ≥ 1) ↔ (x > -2 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2562_256251


namespace NUMINAMATH_CALUDE_train_speed_problem_l2562_256271

theorem train_speed_problem (v : ℝ) : 
  v > 0 →  -- The speed of the first train is positive
  (∃ t : ℝ, t > 0 ∧  -- There exists a positive time t when the trains meet
    v * t = 25 * t + 60 ∧  -- One train travels 60 km more than the other
    v * t + 25 * t = 540) →  -- Total distance traveled equals the distance between stations
  v = 31.25 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2562_256271


namespace NUMINAMATH_CALUDE_school_students_count_l2562_256292

/-- Represents the donation and student information for a school --/
structure SchoolDonation where
  total_donation : ℕ
  average_donation_7_8 : ℕ
  grade_9_intended_donation : ℕ
  grade_9_rejection_rate : ℚ

/-- Calculates the total number of students in the school based on donation information --/
def total_students (sd : SchoolDonation) : ℕ :=
  sd.total_donation / sd.average_donation_7_8

/-- Theorem stating that the total number of students in the school is 224 --/
theorem school_students_count (sd : SchoolDonation) 
  (h1 : sd.total_donation = 13440)
  (h2 : sd.average_donation_7_8 = 60)
  (h3 : sd.grade_9_intended_donation = 100)
  (h4 : sd.grade_9_rejection_rate = 2/5) :
  total_students sd = 224 := by
  sorry

#eval total_students { 
  total_donation := 13440, 
  average_donation_7_8 := 60, 
  grade_9_intended_donation := 100, 
  grade_9_rejection_rate := 2/5 
}

end NUMINAMATH_CALUDE_school_students_count_l2562_256292


namespace NUMINAMATH_CALUDE_coeff_x_neg_one_proof_l2562_256229

/-- The coefficient of x^(-1) in the expansion of (√x - 2/x)^7 -/
def coeff_x_neg_one : ℤ := -280

/-- The binomial coefficient (7 choose 3) -/
def binom_7_3 : ℕ := Nat.choose 7 3

theorem coeff_x_neg_one_proof :
  coeff_x_neg_one = binom_7_3 * (-8) :=
sorry

end NUMINAMATH_CALUDE_coeff_x_neg_one_proof_l2562_256229


namespace NUMINAMATH_CALUDE_office_age_problem_l2562_256235

theorem office_age_problem (total_persons : ℕ) (avg_age_all : ℕ) 
  (group1_persons : ℕ) (avg_age_group1 : ℕ) (group2_persons : ℕ) 
  (age_15th_person : ℕ) : 
  total_persons = 16 → 
  avg_age_all = 15 → 
  group1_persons = 5 → 
  avg_age_group1 = 14 → 
  group2_persons = 9 → 
  age_15th_person = 26 → 
  (avg_age_all * total_persons - avg_age_group1 * group1_persons - age_15th_person) / group2_persons = 16 := by
sorry

end NUMINAMATH_CALUDE_office_age_problem_l2562_256235


namespace NUMINAMATH_CALUDE_rectangular_to_polar_l2562_256208

theorem rectangular_to_polar :
  let x : ℝ := 3
  let y : ℝ := 3 * Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ r = 6 ∧ θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_l2562_256208


namespace NUMINAMATH_CALUDE_green_pepper_weight_is_half_total_l2562_256282

/-- The weight of green peppers bought by Hannah's Vegetarian Restaurant -/
def green_pepper_weight : ℝ := 0.33333333335

/-- The total weight of peppers bought by Hannah's Vegetarian Restaurant -/
def total_pepper_weight : ℝ := 0.6666666667

/-- Theorem stating that the weight of green peppers is half the total weight -/
theorem green_pepper_weight_is_half_total :
  green_pepper_weight = total_pepper_weight / 2 := by sorry

end NUMINAMATH_CALUDE_green_pepper_weight_is_half_total_l2562_256282


namespace NUMINAMATH_CALUDE_eighteen_wheel_truck_toll_l2562_256207

/-- Calculates the number of axles for a truck given the total number of wheels,
    the number of wheels on the front axle, and the number of wheels on each other axle -/
def calculate_axles (total_wheels : ℕ) (front_axle_wheels : ℕ) (other_axle_wheels : ℕ) : ℕ :=
  1 + (total_wheels - front_axle_wheels) / other_axle_wheels

/-- Calculates the toll for a truck given the number of axles -/
def calculate_toll (axles : ℕ) : ℚ :=
  0.50 + 0.50 * (axles - 2)

theorem eighteen_wheel_truck_toll :
  let total_wheels : ℕ := 18
  let front_axle_wheels : ℕ := 2
  let other_axle_wheels : ℕ := 4
  let axles := calculate_axles total_wheels front_axle_wheels other_axle_wheels
  calculate_toll axles = 2 := by sorry

end NUMINAMATH_CALUDE_eighteen_wheel_truck_toll_l2562_256207


namespace NUMINAMATH_CALUDE_profit_percentage_l2562_256250

/-- Given that the cost price of 150 articles equals the selling price of 120 articles,
    prove that the percent profit is 25%. -/
theorem profit_percentage (cost selling : ℝ) (h : 150 * cost = 120 * selling) :
  (selling - cost) / cost * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l2562_256250


namespace NUMINAMATH_CALUDE_find_m_value_l2562_256204

/-- Given x and y values, prove that m = 3 when y is linearly related to x with equation y = 1.3x + 0.8 -/
theorem find_m_value (x : Fin 5 → ℝ) (y : Fin 5 → ℝ) (m : ℝ) : 
  x 0 = 1 ∧ x 1 = 3 ∧ x 2 = 4 ∧ x 3 = 5 ∧ x 4 = 7 ∧
  y 0 = 1 ∧ y 1 = m ∧ y 2 = 2*m+1 ∧ y 3 = 2*m+3 ∧ y 4 = 10 ∧
  (∀ i : Fin 5, y i = 1.3 * x i + 0.8) →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_m_value_l2562_256204


namespace NUMINAMATH_CALUDE_max_y_coordinate_polar_curve_l2562_256234

theorem max_y_coordinate_polar_curve (θ : Real) :
  let r := Real.sin (2 * θ)
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  |y| ≤ 4 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_max_y_coordinate_polar_curve_l2562_256234


namespace NUMINAMATH_CALUDE_shaded_area_semicircle_pattern_shaded_area_semicircle_pattern_correct_l2562_256233

/-- The area of the shaded region in a pattern of semicircles -/
theorem shaded_area_semicircle_pattern (pattern_length : Real) (semicircle_diameter : Real) 
  (h1 : pattern_length = 2 * 12) -- 2 feet converted to inches
  (h2 : semicircle_diameter = 3) -- diameter in inches
  : Real :=
  18 * Real.pi

theorem shaded_area_semicircle_pattern_correct 
  (pattern_length : Real) (semicircle_diameter : Real) 
  (h1 : pattern_length = 2 * 12) -- 2 feet converted to inches
  (h2 : semicircle_diameter = 3) -- diameter in inches
  : shaded_area_semicircle_pattern pattern_length semicircle_diameter h1 h2 = 18 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_semicircle_pattern_shaded_area_semicircle_pattern_correct_l2562_256233


namespace NUMINAMATH_CALUDE_pear_banana_weight_equality_l2562_256239

/-- Given that 10 pears weigh the same as 6 bananas, 
    prove that 50 pears weigh the same as 30 bananas. -/
theorem pear_banana_weight_equality :
  ∀ (pear_weight banana_weight : ℕ → ℝ),
  (∀ n : ℕ, pear_weight (10 * n) = banana_weight (6 * n)) →
  pear_weight 50 = banana_weight 30 :=
by
  sorry

end NUMINAMATH_CALUDE_pear_banana_weight_equality_l2562_256239


namespace NUMINAMATH_CALUDE_complex_number_properties_l2562_256231

theorem complex_number_properties : ∃ (z : ℂ), 
  z = 2 / (Complex.I - 1) ∧ 
  z^2 = 2 * Complex.I ∧ 
  z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l2562_256231


namespace NUMINAMATH_CALUDE_candy_distribution_l2562_256283

theorem candy_distribution (n : ℕ) (h1 : n > 0) : 
  (100 % n = 1) ↔ n = 11 := by sorry

end NUMINAMATH_CALUDE_candy_distribution_l2562_256283


namespace NUMINAMATH_CALUDE_root_cubic_value_l2562_256238

theorem root_cubic_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2014 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_root_cubic_value_l2562_256238


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2562_256284

theorem quadratic_inequality_solution_set :
  {x : ℝ | 3 * x^2 - 5 * x - 8 > 0} = Set.Iio (-4/3) ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2562_256284


namespace NUMINAMATH_CALUDE_closest_point_l2562_256205

def u (s : ℝ) : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 3 + 6*s
  | 1 => -2 + 4*s
  | 2 => 4 + 2*s

def b : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 1
  | 1 => 7
  | 2 => 6

def direction_vector : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 6
  | 1 => 4
  | 2 => 2

theorem closest_point (s : ℝ) :
  (∀ t : ℝ, ‖u s - b‖ ≤ ‖u t - b‖) ↔ s = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_closest_point_l2562_256205


namespace NUMINAMATH_CALUDE_tank_length_is_six_l2562_256243

def tank_volume : ℝ := 72
def tank_width : ℝ := 4
def tank_depth : ℝ := 3

theorem tank_length_is_six :
  let length := tank_volume / (tank_width * tank_depth)
  length = 6 := by sorry

end NUMINAMATH_CALUDE_tank_length_is_six_l2562_256243


namespace NUMINAMATH_CALUDE_arithmetic_equality_l2562_256237

theorem arithmetic_equality : 19 * 17 + 29 * 17 + 48 * 25 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l2562_256237


namespace NUMINAMATH_CALUDE_jose_peanuts_l2562_256222

theorem jose_peanuts (kenya_peanuts : ℕ) (difference : ℕ) (h1 : kenya_peanuts = 133) (h2 : difference = 48) :
  kenya_peanuts - difference = 85 := by
  sorry

end NUMINAMATH_CALUDE_jose_peanuts_l2562_256222


namespace NUMINAMATH_CALUDE_log_equation_solution_l2562_256279

theorem log_equation_solution (y : ℝ) (h : y > 0) :
  Real.log y / Real.log 3 + Real.log y / Real.log 9 = 5 → y = 3^(10/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2562_256279


namespace NUMINAMATH_CALUDE_multiple_of_x_l2562_256225

theorem multiple_of_x (x y z k : ℕ+) : 
  (k * x = 5 * y) ∧ (5 * y = 8 * z) ∧ (x + y + z = 33) → k = 40 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_x_l2562_256225


namespace NUMINAMATH_CALUDE_carl_typing_speed_l2562_256280

theorem carl_typing_speed :
  ∀ (hours_per_day : ℕ) (total_words : ℕ) (total_days : ℕ),
    hours_per_day = 4 →
    total_words = 84000 →
    total_days = 7 →
    (total_words / total_days) / (hours_per_day * 60) = 50 := by
  sorry

end NUMINAMATH_CALUDE_carl_typing_speed_l2562_256280


namespace NUMINAMATH_CALUDE_product_greater_than_sum_l2562_256253

theorem product_greater_than_sum {a b : ℝ} (ha : a > 2) (hb : b > 2) : a * b > a + b := by
  sorry

end NUMINAMATH_CALUDE_product_greater_than_sum_l2562_256253
