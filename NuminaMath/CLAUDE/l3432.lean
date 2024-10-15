import Mathlib

namespace NUMINAMATH_CALUDE_sum_34_47_in_base5_l3432_343243

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_34_47_in_base5 :
  toBase5 (34 + 47) = [3, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_34_47_in_base5_l3432_343243


namespace NUMINAMATH_CALUDE_min_shading_for_symmetry_l3432_343267

/-- Represents a triangular figure with some shaded triangles -/
structure TriangularFigure where
  total_triangles : Nat
  shaded_triangles : Nat
  h_shaded_le_total : shaded_triangles ≤ total_triangles

/-- Calculates the minimum number of additional triangles to shade for axial symmetry -/
def min_additional_shading (figure : TriangularFigure) : Nat :=
  sorry

/-- Theorem stating the minimum additional shading for the given problem -/
theorem min_shading_for_symmetry (figure : TriangularFigure) 
  (h_total : figure.total_triangles = 54)
  (h_some_shaded : figure.shaded_triangles > 0)
  (h_not_all_shaded : figure.shaded_triangles < 54) :
  min_additional_shading figure = 6 :=
sorry

end NUMINAMATH_CALUDE_min_shading_for_symmetry_l3432_343267


namespace NUMINAMATH_CALUDE_broker_commission_slump_l3432_343290

theorem broker_commission_slump (initial_rate final_rate : ℝ) 
  (initial_business final_business : ℝ) (h1 : initial_rate = 0.04) 
  (h2 : final_rate = 0.05) (h3 : initial_rate * initial_business = final_rate * final_business) :
  (initial_business - final_business) / initial_business = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_broker_commission_slump_l3432_343290


namespace NUMINAMATH_CALUDE_part_one_part_two_l3432_343245

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem part_one (t : Triangle) (h1 : t.a + t.b + t.c = 16) (h2 : t.a = 4) (h3 : t.b = 5) :
  Real.cos t.C = -1/5 := by sorry

-- Part 2
theorem part_two (t : Triangle) (h1 : t.a + t.b + t.c = 16) 
  (h2 : Real.sin t.A + Real.sin t.B = 3 * Real.sin t.C)
  (h3 : 1/2 * t.a * t.b * Real.sin t.C = 18 * Real.sin t.C) :
  t.a = 6 ∧ t.b = 6 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3432_343245


namespace NUMINAMATH_CALUDE_sum_of_square_and_triangular_l3432_343279

theorem sum_of_square_and_triangular (k : ℕ) :
  let Sₖ := (6 * 10^k - 1) * 10^(k+2) + 5 * 10^(k+1) + 1
  let n := 2 * 10^(k+1) - 1
  Sₖ = n^2 + n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_and_triangular_l3432_343279


namespace NUMINAMATH_CALUDE_forum_posts_theorem_l3432_343259

/-- Calculates the total number of questions and answers posted on a forum in a day. -/
def forum_posts (members : ℕ) (questions_per_hour : ℕ) (answer_ratio : ℕ) : ℕ :=
  let questions_per_day := questions_per_hour * 24
  let answers_per_day := questions_per_day * answer_ratio
  members * (questions_per_day + answers_per_day)

/-- Theorem: Given the forum conditions, the total posts in a day is 1,008,000. -/
theorem forum_posts_theorem :
  forum_posts 1000 7 5 = 1008000 := by
  sorry

end NUMINAMATH_CALUDE_forum_posts_theorem_l3432_343259


namespace NUMINAMATH_CALUDE_bus_journey_time_l3432_343231

/-- Represents the journey of Xiao Ming to school -/
structure Journey where
  subway_time : ℕ -- Time taken by subway in minutes
  bus_time : ℕ -- Time taken by bus in minutes
  transfer_time : ℕ -- Time taken for transfer in minutes
  total_time : ℕ -- Total time of the journey in minutes

/-- Theorem stating the correct time spent on the bus -/
theorem bus_journey_time (j : Journey) 
  (h1 : j.subway_time = 30)
  (h2 : j.bus_time = 50)
  (h3 : j.transfer_time = 6)
  (h4 : j.total_time = 40)
  (h5 : j.total_time = j.subway_time + j.bus_time + j.transfer_time) :
  ∃ (actual_bus_time : ℕ), actual_bus_time = 10 ∧ 
    j.total_time = (j.subway_time - (j.subway_time - actual_bus_time)) + actual_bus_time + j.transfer_time :=
by sorry

end NUMINAMATH_CALUDE_bus_journey_time_l3432_343231


namespace NUMINAMATH_CALUDE_pythagorean_diagonal_l3432_343262

/-- 
For a right triangle with width 2m (where m ≥ 3 and m is a positive integer) 
and the difference between the diagonal and the height being 2, 
the diagonal is equal to m² - 1.
-/
theorem pythagorean_diagonal (m : ℕ) (h : m ≥ 3) : 
  let width : ℕ := 2 * m
  let diagonal : ℕ := m^2 - 1
  let height : ℕ := diagonal - 2
  width^2 + height^2 = diagonal^2 := by sorry

end NUMINAMATH_CALUDE_pythagorean_diagonal_l3432_343262


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3432_343235

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x, 3 * x^2 + m * x + 9 = 0) ↔ m = 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3432_343235


namespace NUMINAMATH_CALUDE_lunch_spending_l3432_343230

theorem lunch_spending (total : ℝ) (difference : ℝ) (friend_spent : ℝ) : 
  total = 72 → difference = 11 → friend_spent = total / 2 + difference / 2 → friend_spent = 41.5 := by
  sorry

end NUMINAMATH_CALUDE_lunch_spending_l3432_343230


namespace NUMINAMATH_CALUDE_integral_f_equals_two_l3432_343204

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if -1 ≤ x ∧ x ≤ 1 then x^3 + Real.sin x
  else if 1 < x ∧ x ≤ 2 then 2
  else 0  -- We need to define f for all real numbers

-- State the theorem
theorem integral_f_equals_two : 
  ∫ x in (-1)..(2), f x = 2 := by sorry

end NUMINAMATH_CALUDE_integral_f_equals_two_l3432_343204


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3432_343272

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 * a 3 * a 11 = 8 →
  a 2 * a 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3432_343272


namespace NUMINAMATH_CALUDE_cubic_factorization_l3432_343211

theorem cubic_factorization (x : ℝ) : x^3 - 9*x = x*(x+3)*(x-3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3432_343211


namespace NUMINAMATH_CALUDE_probability_multiple_6_or_8_l3432_343293

def is_multiple_of_6_or_8 (n : ℕ) : Bool :=
  n % 6 = 0 || n % 8 = 0

def count_multiples (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_6_or_8 |>.length

theorem probability_multiple_6_or_8 :
  count_multiples 100 / 100 = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_6_or_8_l3432_343293


namespace NUMINAMATH_CALUDE_unit_digit_of_product_l3432_343247

theorem unit_digit_of_product : (5 + 1) * (5^3 + 1) * (5^6 + 1) * (5^12 + 1) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_product_l3432_343247


namespace NUMINAMATH_CALUDE_words_per_page_l3432_343268

theorem words_per_page (total_pages : ℕ) (max_words_per_page : ℕ) (total_words_mod : ℕ) :
  total_pages = 154 →
  max_words_per_page = 120 →
  total_words_mod = 250 →
  ∃ (words_per_page : ℕ),
    words_per_page ≤ max_words_per_page ∧
    (total_pages * words_per_page) % 227 = total_words_mod % 227 ∧
    words_per_page = 49 := by
  sorry

end NUMINAMATH_CALUDE_words_per_page_l3432_343268


namespace NUMINAMATH_CALUDE_hourly_wage_calculation_l3432_343292

/-- The hourly wage in dollars -/
def hourly_wage : ℝ := 12.5

/-- The number of hours worked per week -/
def hours_worked : ℝ := 40

/-- The pay per widget in dollars -/
def pay_per_widget : ℝ := 0.16

/-- The number of widgets produced per week -/
def widgets_produced : ℝ := 1250

/-- The total earnings for the week in dollars -/
def total_earnings : ℝ := 700

theorem hourly_wage_calculation :
  hourly_wage * hours_worked + pay_per_widget * widgets_produced = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_hourly_wage_calculation_l3432_343292


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_no_intersection_l3432_343215

-- Define a structure for a 3D space
structure Space3D where
  -- Add any necessary fields

-- Define a line in 3D space
structure Line where
  -- Add any necessary fields

-- Define a plane in 3D space
structure Plane where
  -- Add any necessary fields

-- Define parallelism between a line and a plane
def parallel (l : Line) (p : Plane) : Prop := sorry

-- Define intersection between two lines
def intersect (l1 l2 : Line) : Prop := sorry

-- Define a function to get a line in a plane
def line_in_plane (p : Plane) : Line := sorry

-- Theorem statement
theorem line_parallel_to_plane_no_intersection 
  (a : Line) (α : Plane) : 
  parallel a α → ∀ l : Line, (∃ p : Plane, l = line_in_plane p) → ¬(intersect a l) := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_no_intersection_l3432_343215


namespace NUMINAMATH_CALUDE_lucy_age_l3432_343254

/-- Given the ages of Inez, Zack, Jose, and Lucy, prove Lucy's age --/
theorem lucy_age (inez zack jose lucy : ℕ) 
  (h1 : lucy = jose + 2)
  (h2 : jose + 6 = zack)
  (h3 : zack = inez + 4)
  (h4 : inez = 18) : 
  lucy = 18 := by
  sorry

end NUMINAMATH_CALUDE_lucy_age_l3432_343254


namespace NUMINAMATH_CALUDE_fence_cost_l3432_343265

theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 59) :
  4 * Real.sqrt area * price_per_foot = 4012 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_l3432_343265


namespace NUMINAMATH_CALUDE_flat_terrain_distance_l3432_343295

theorem flat_terrain_distance (total_time : ℚ) (total_distance : ℕ) 
  (speed_uphill speed_flat speed_downhill : ℚ) 
  (h_total_time : total_time = 29 / 15)
  (h_total_distance : total_distance = 9)
  (h_speed_uphill : speed_uphill = 4)
  (h_speed_flat : speed_flat = 5)
  (h_speed_downhill : speed_downhill = 6) :
  ∃ (x y : ℕ), 
    x + y ≤ total_distance ∧
    x / speed_uphill + y / speed_flat + (total_distance - x - y) / speed_downhill = total_time ∧
    y = 3 := by
  sorry

end NUMINAMATH_CALUDE_flat_terrain_distance_l3432_343295


namespace NUMINAMATH_CALUDE_conic_is_parabola_l3432_343270

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  |x - 3| = Real.sqrt ((y + 4)^2 + x^2)

-- Define what it means for an equation to describe a parabola
def describes_parabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x y, f x y ↔ y = a * x^2 + b * x + c

-- Theorem statement
theorem conic_is_parabola : describes_parabola conic_equation := by
  sorry

end NUMINAMATH_CALUDE_conic_is_parabola_l3432_343270


namespace NUMINAMATH_CALUDE_range_of_a_l3432_343252

-- Define the propositions P and Q as functions of a
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def Q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (5 - 2*a)^x < (5 - 2*a)^y

-- Define the theorem
theorem range_of_a : 
  (∀ a : ℝ, (P a ∨ Q a) ∧ ¬(P a ∧ Q a)) → 
  {a : ℝ | a ≤ -2} = {a : ℝ | ∀ a' : ℝ, (P a' ∨ Q a') ∧ ¬(P a' ∧ Q a') → a ≤ a'} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3432_343252


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l3432_343249

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 2 - 5 * Complex.I) * (2 * Real.sqrt 3 + 2 * Complex.I)) = 4 * Real.sqrt 43 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l3432_343249


namespace NUMINAMATH_CALUDE_big_toenail_count_l3432_343283

/-- Represents the capacity and contents of a toenail jar -/
structure ToenailJar where
  capacity : ℕ  -- Total capacity in terms of regular toenails
  regularSize : ℕ  -- Size of a regular toenail (set to 1)
  bigSize : ℕ  -- Size of a big toenail
  regularCount : ℕ  -- Number of regular toenails in the jar
  remainingSpace : ℕ  -- Remaining space in terms of regular toenails
  bigCount : ℕ  -- Number of big toenails in the jar

/-- Theorem stating the number of big toenails in the jar -/
theorem big_toenail_count (jar : ToenailJar)
  (h1 : jar.capacity = 100)
  (h2 : jar.bigSize = 2 * jar.regularSize)
  (h3 : jar.regularCount = 40)
  (h4 : jar.remainingSpace = 20)
  : jar.bigCount = 10 := by
  sorry

end NUMINAMATH_CALUDE_big_toenail_count_l3432_343283


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3432_343219

theorem absolute_value_inequality (x y ε : ℝ) (h_pos : ε > 0) 
  (hx : |x - 2| < ε) (hy : |y - 2| < ε) : |x - y| < 2 * ε := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3432_343219


namespace NUMINAMATH_CALUDE_smallest_x_for_fifth_power_l3432_343273

theorem smallest_x_for_fifth_power (x : ℕ) (K : ℤ) : 
  (x = 135000 ∧ 
   180 * x = K^5 ∧ 
   ∀ y : ℕ, y < x → ¬∃ L : ℤ, 180 * y = L^5) :=
sorry

end NUMINAMATH_CALUDE_smallest_x_for_fifth_power_l3432_343273


namespace NUMINAMATH_CALUDE_frog_corner_probability_l3432_343209

/-- Represents a position on the 3x3 grid -/
inductive Position
| Center
| Edge
| Corner

/-- Represents the number of hops made -/
def MaxHops : Nat := 4

/-- Probability of reaching a corner from a given position in n hops -/
noncomputable def reachCornerProb (pos : Position) (n : Nat) : Real :=
  sorry

/-- The main theorem to prove -/
theorem frog_corner_probability :
  reachCornerProb Position.Center MaxHops = 25 / 32 := by
  sorry

end NUMINAMATH_CALUDE_frog_corner_probability_l3432_343209


namespace NUMINAMATH_CALUDE_square_expression_is_perfect_square_l3432_343261

theorem square_expression_is_perfect_square (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  (2 * l - n - k) * (2 * l - n + k) / 2 = (l - n)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_expression_is_perfect_square_l3432_343261


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3432_343225

-- Define the quadratic function
def f (a b x : ℝ) := x^2 - a*x + b

-- Define the solution set
def solution_set (a b : ℝ) := {x : ℝ | f a b x < 0}

-- State the theorem
theorem quadratic_inequality_solution (a b : ℝ) :
  solution_set a b = {x : ℝ | -1 < x ∧ x < 3} →
  a + b = -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3432_343225


namespace NUMINAMATH_CALUDE_percent_y_of_x_l3432_343256

theorem percent_y_of_x (x y : ℝ) (h : 0.2 * (x - y) = 0.14 * (x + y)) :
  y = (300 / 17) / 100 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_y_of_x_l3432_343256


namespace NUMINAMATH_CALUDE_minimum_sum_l3432_343284

theorem minimum_sum (x y z : ℝ) (h : (4 / x) + (2 / y) + (1 / z) = 1) :
  x + 8 * y + 4 * z ≥ 64 ∧
  (x + 8 * y + 4 * z = 64 ↔ x = 16 ∧ y = 4 ∧ z = 4) := by
  sorry

end NUMINAMATH_CALUDE_minimum_sum_l3432_343284


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_seventeen_is_dual_palindrome_seventeen_is_smallest_dual_palindrome_l3432_343208

/-- Checks if a natural number is a palindrome in the given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ k : ℕ, k > 15 → 
    (isPalindrome k 2 ∧ isPalindrome k 4) → 
    k ≥ 17 := by sorry

theorem seventeen_is_dual_palindrome : 
  isPalindrome 17 2 ∧ isPalindrome 17 4 := by sorry

theorem seventeen_is_smallest_dual_palindrome : 
  ∀ k : ℕ, k > 15 → 
    (isPalindrome k 2 ∧ isPalindrome k 4) → 
    k = 17 :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_seventeen_is_dual_palindrome_seventeen_is_smallest_dual_palindrome_l3432_343208


namespace NUMINAMATH_CALUDE_seven_people_arrangement_l3432_343281

def number_of_people : ℕ := 7
def number_of_special_people : ℕ := 3

def arrangements (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem seven_people_arrangement :
  let regular_people := number_of_people - number_of_special_people
  let gaps := regular_people + 1
  arrangements regular_people regular_people * arrangements gaps number_of_special_people = 1440 :=
sorry

end NUMINAMATH_CALUDE_seven_people_arrangement_l3432_343281


namespace NUMINAMATH_CALUDE_initial_amount_is_21_l3432_343213

/-- Represents the money transactions between three people A, B, and C. -/
structure MoneyTransaction where
  a_initial : ℚ
  b_initial : ℚ := 5
  c_initial : ℚ := 9

/-- Calculates the final amounts after all transactions. -/
def final_amounts (mt : MoneyTransaction) : ℚ × ℚ × ℚ :=
  let a1 := mt.a_initial - (mt.b_initial + mt.c_initial)
  let b1 := 2 * mt.b_initial
  let c1 := 2 * mt.c_initial
  
  let a2 := a1 + (a1 / 2)
  let b2 := b1 - ((a1 / 2) + (c1 / 2))
  let c2 := c1 + (c1 / 2)
  
  let a3 := a2 + 3 * a2 + 3 * b2
  let b3 := b2 + 3 * b2 + 3 * c2
  let c3 := c2 - (3 * a2 + 3 * b2)
  
  (a3, b3, c3)

/-- Theorem stating that if the final amounts are (24, 16, 8), then A started with 21 cents. -/
theorem initial_amount_is_21 (mt : MoneyTransaction) : 
  final_amounts mt = (24, 16, 8) → mt.a_initial = 21 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_is_21_l3432_343213


namespace NUMINAMATH_CALUDE_beth_dive_tanks_l3432_343220

/-- Calculates the number of supplemental tanks needed for a scuba dive. -/
def supplementalTanksNeeded (totalDiveTime primaryTankDuration supplementalTankDuration : ℕ) : ℕ :=
  (totalDiveTime - primaryTankDuration) / supplementalTankDuration

/-- Proves that for the given dive parameters, 6 supplemental tanks are needed. -/
theorem beth_dive_tanks : 
  supplementalTanksNeeded 8 2 1 = 6 := by
  sorry

#eval supplementalTanksNeeded 8 2 1

end NUMINAMATH_CALUDE_beth_dive_tanks_l3432_343220


namespace NUMINAMATH_CALUDE_beads_taken_out_l3432_343237

theorem beads_taken_out (green brown red left : ℕ) : 
  green = 1 → brown = 2 → red = 3 → left = 4 → 
  (green + brown + red) - left = 2 := by
  sorry

end NUMINAMATH_CALUDE_beads_taken_out_l3432_343237


namespace NUMINAMATH_CALUDE_eighth_term_is_one_l3432_343288

-- Define the sequence a_n
def a (n : ℕ+) : ℤ := (-1) ^ n.val

-- Theorem statement
theorem eighth_term_is_one : a 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_one_l3432_343288


namespace NUMINAMATH_CALUDE_all_squares_similar_l3432_343236

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define similarity for squares
def similar (s1 s2 : Square) : Prop :=
  ∃ k : ℝ, k > 0 ∧ s1.side = k * s2.side

-- Theorem: Any two squares are similar
theorem all_squares_similar (s1 s2 : Square) : similar s1 s2 := by
  sorry


end NUMINAMATH_CALUDE_all_squares_similar_l3432_343236


namespace NUMINAMATH_CALUDE_moles_of_CH3COOH_l3432_343274

-- Define the chemical reaction
structure Reaction where
  reactant1 : ℝ  -- moles of CH3COOH
  reactant2 : ℝ  -- moles of NaOH
  product1  : ℝ  -- moles of NaCH3COO
  product2  : ℝ  -- moles of H2O

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.reactant1 = r.reactant2 ∧ r.reactant1 = r.product1 ∧ r.reactant1 = r.product2

-- Theorem statement
theorem moles_of_CH3COOH (r : Reaction) 
  (h1 : r.reactant2 = 1)  -- 1 mole of NaOH is used
  (h2 : r.product1 = 1)   -- 1 mole of NaCH3COO is formed
  (h3 : balanced_equation r)  -- The reaction follows the balanced equation
  : r.reactant1 = 1 :=  -- The number of moles of CH3COOH combined is 1
by sorry

end NUMINAMATH_CALUDE_moles_of_CH3COOH_l3432_343274


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l3432_343229

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z ≥ 3) :
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 :=
sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z ≥ 3) :
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1) ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l3432_343229


namespace NUMINAMATH_CALUDE_combined_final_price_theorem_l3432_343258

def calculate_final_price (cost_price repairs discount_rate tax_rate : ℝ) : ℝ :=
  let total_cost := cost_price + repairs
  let discounted_price := total_cost * (1 - discount_rate)
  discounted_price * (1 + tax_rate)

def cycle_a_price := calculate_final_price 1800 200 0.10 0.05
def cycle_b_price := calculate_final_price 2400 300 0.12 0.06
def cycle_c_price := calculate_final_price 3200 400 0.15 0.07

theorem combined_final_price_theorem :
  cycle_a_price + cycle_b_price + cycle_c_price = 7682.76 := by
  sorry

end NUMINAMATH_CALUDE_combined_final_price_theorem_l3432_343258


namespace NUMINAMATH_CALUDE_girls_only_wind_count_l3432_343269

/-- Represents the number of students in different categories of the school bands -/
structure BandParticipation where
  wind_boys : ℕ
  wind_girls : ℕ
  string_boys : ℕ
  string_girls : ℕ
  total_students : ℕ
  boys_in_both : ℕ

/-- Calculates the number of girls participating only in the wind band -/
def girls_only_wind (bp : BandParticipation) : ℕ :=
  bp.wind_girls - (bp.total_students - (bp.wind_boys + bp.wind_girls + bp.string_boys + bp.string_girls - bp.boys_in_both) - bp.boys_in_both)

/-- The main theorem stating that given the specific band participation numbers, 
    the number of girls participating only in the wind band is 10 -/
theorem girls_only_wind_count : 
  let bp : BandParticipation := {
    wind_boys := 100,
    wind_girls := 80,
    string_boys := 80,
    string_girls := 100,
    total_students := 230,
    boys_in_both := 60
  }
  girls_only_wind bp = 10 := by sorry

end NUMINAMATH_CALUDE_girls_only_wind_count_l3432_343269


namespace NUMINAMATH_CALUDE_intersection_distance_l3432_343226

/-- The distance between intersection points of a line and a circle --/
theorem intersection_distance (t : ℝ) : 
  let x : ℝ → ℝ := λ t => -1 + (Real.sqrt 3 / 2) * t
  let y : ℝ → ℝ := λ t => (1 / 2) * t
  let l : ℝ → ℝ × ℝ := λ t => (x t, y t)
  let C : ℝ → ℝ := λ θ => 4 * Real.cos θ
  ∃ P Q : ℝ × ℝ, P ≠ Q ∧ 
    (P.1 - 2)^2 + P.2^2 = 4 ∧
    (Q.1 - 2)^2 + Q.2^2 = 4 ∧
    P.1 - Real.sqrt 3 * P.2 + 1 = 0 ∧
    Q.1 - Real.sqrt 3 * Q.2 + 1 = 0 ∧
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l3432_343226


namespace NUMINAMATH_CALUDE_butterfly_collection_l3432_343296

theorem butterfly_collection (total : ℕ) (black : ℕ) :
  total = 11 →
  black = 5 →
  ∃ (blue yellow : ℕ),
    blue = 2 * yellow ∧
    blue + yellow + black = total ∧
    blue = 4 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_collection_l3432_343296


namespace NUMINAMATH_CALUDE_restaurant_outdoor_section_area_l3432_343294

/-- The area of a rectangular section with width 7 feet and length 5 feet is 35 square feet. -/
theorem restaurant_outdoor_section_area :
  let width : ℝ := 7
  let length : ℝ := 5
  width * length = 35 := by sorry

end NUMINAMATH_CALUDE_restaurant_outdoor_section_area_l3432_343294


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3432_343285

/-- Given a sphere with surface area 256π cm², its volume is 2048/3 π cm³. -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 4 * π * r^2 = 256 * π → (4 / 3) * π * r^3 = (2048 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3432_343285


namespace NUMINAMATH_CALUDE_intersection_value_l3432_343263

theorem intersection_value (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a + 2, a^2 + 4}
  (A ∩ B = {3}) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_value_l3432_343263


namespace NUMINAMATH_CALUDE_amy_school_year_hours_l3432_343287

/-- Represents Amy's work schedule and earnings --/
structure WorkSchedule where
  summer_hours_per_week : ℕ
  summer_weeks : ℕ
  summer_earnings : ℕ
  school_year_weeks : ℕ
  school_year_earnings : ℕ

/-- Calculates the required hours per week for the school year --/
def required_school_year_hours (schedule : WorkSchedule) : ℚ :=
  (schedule.summer_hours_per_week : ℚ) * (schedule.summer_weeks : ℚ) * (schedule.school_year_earnings : ℚ) /
  ((schedule.summer_earnings : ℚ) * (schedule.school_year_weeks : ℚ))

/-- Theorem stating that Amy needs to work 12 hours per week during the school year --/
theorem amy_school_year_hours (schedule : WorkSchedule)
  (h1 : schedule.summer_hours_per_week = 36)
  (h2 : schedule.summer_weeks = 10)
  (h3 : schedule.summer_earnings = 3000)
  (h4 : schedule.school_year_weeks = 30)
  (h5 : schedule.school_year_earnings = 3000) :
  required_school_year_hours schedule = 12 := by
  sorry


end NUMINAMATH_CALUDE_amy_school_year_hours_l3432_343287


namespace NUMINAMATH_CALUDE_probability_graduate_degree_l3432_343232

/-- Represents the number of college graduates with a graduate degree -/
def G : ℕ := 3

/-- Represents the number of college graduates without a graduate degree -/
def C : ℕ := 16

/-- Represents the number of non-college graduates -/
def N : ℕ := 24

/-- The ratio of college graduates with a graduate degree to non-college graduates is 1:8 -/
axiom ratio_G_N : G * 8 = N * 1

/-- The ratio of college graduates without a graduate degree to non-college graduates is 2:3 -/
axiom ratio_C_N : C * 3 = N * 2

/-- The probability that a randomly picked college graduate has a graduate degree -/
def prob_graduate_degree : ℚ := G / (G + C)

/-- Theorem: The probability that a randomly picked college graduate has a graduate degree is 3/19 -/
theorem probability_graduate_degree : prob_graduate_degree = 3 / 19 := by sorry

end NUMINAMATH_CALUDE_probability_graduate_degree_l3432_343232


namespace NUMINAMATH_CALUDE_ordering_of_exponentials_l3432_343264

theorem ordering_of_exponentials (x a b : ℝ) :
  x > 0 → 1 < b^x → b^x < a^x → 1 < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ordering_of_exponentials_l3432_343264


namespace NUMINAMATH_CALUDE_terrys_total_spending_l3432_343201

/-- Terry's spending over three days --/
def terrys_spending (monday_amount : ℝ) : ℝ :=
  let tuesday_amount := 2 * monday_amount
  let wednesday_amount := 2 * (monday_amount + tuesday_amount)
  monday_amount + tuesday_amount + wednesday_amount

/-- Theorem: Terry's total spending is $54 --/
theorem terrys_total_spending : terrys_spending 6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_terrys_total_spending_l3432_343201


namespace NUMINAMATH_CALUDE_candy_bar_payment_l3432_343280

/-- Calculates the number of dimes used to pay for a candy bar -/
def dimes_used (quarter_value : ℕ) (nickel_value : ℕ) (dime_value : ℕ) 
  (num_quarters : ℕ) (num_nickels : ℕ) (change : ℕ) (candy_cost : ℕ) : ℕ :=
  let total_paid := candy_cost + change
  let paid_without_dimes := num_quarters * quarter_value + num_nickels * nickel_value
  (total_paid - paid_without_dimes) / dime_value

theorem candy_bar_payment :
  dimes_used 25 5 10 4 1 4 131 = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_payment_l3432_343280


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l3432_343238

theorem tax_free_items_cost 
  (total_paid : ℝ)
  (sales_tax : ℝ)
  (tax_rate : ℝ)
  (h1 : total_paid = 30)
  (h2 : sales_tax = 1.28)
  (h3 : tax_rate = 0.08)
  : ∃ (tax_free_cost : ℝ), tax_free_cost = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_tax_free_items_cost_l3432_343238


namespace NUMINAMATH_CALUDE_coefficient_x2y2_in_expansion_expansion_coefficient_is_18_l3432_343289

theorem coefficient_x2y2_in_expansion : ℕ → Prop :=
  fun n => (Finset.sum (Finset.range 4) fun i =>
    (Finset.sum (Finset.range 5) fun j =>
      if i + j = 4 then
        (Nat.choose 3 i) * (Nat.choose 4 j)
      else
        0)) = n

theorem expansion_coefficient_is_18 : coefficient_x2y2_in_expansion 18 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x2y2_in_expansion_expansion_coefficient_is_18_l3432_343289


namespace NUMINAMATH_CALUDE_inscribed_trapezoid_theorem_l3432_343224

/-- An isosceles trapezoid inscribed in a circle -/
structure InscribedTrapezoid where
  r : ℝ  -- radius of the circle
  a : ℝ  -- half of the shorter base
  c : ℝ  -- half of the longer base
  h : 0 < r ∧ 0 < a ∧ 0 < c ∧ a < c  -- conditions for a valid trapezoid

/-- Theorem: For an isosceles trapezoid inscribed in a circle, r^2 = ac -/
theorem inscribed_trapezoid_theorem (t : InscribedTrapezoid) : t.r^2 = t.a * t.c := by
  sorry

end NUMINAMATH_CALUDE_inscribed_trapezoid_theorem_l3432_343224


namespace NUMINAMATH_CALUDE_intersection_point_l3432_343221

/-- The point (3, 2) is the unique solution to the system of equations x + y = 5 and x - y = 1 -/
theorem intersection_point : ∃! p : ℝ × ℝ, p.1 + p.2 = 5 ∧ p.1 - p.2 = 1 ∧ p = (3, 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3432_343221


namespace NUMINAMATH_CALUDE_stacy_paper_pages_per_day_l3432_343227

/-- Given a paper with a certain number of pages due in a certain number of days,
    calculate the number of pages that need to be written per day to finish on time. -/
def pages_per_day (total_pages : ℕ) (days : ℕ) : ℚ :=
  total_pages / days

/-- Theorem stating that for a 100-page paper due in 5 days,
    the number of pages to be written per day is 20. -/
theorem stacy_paper_pages_per_day :
  pages_per_day 100 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_stacy_paper_pages_per_day_l3432_343227


namespace NUMINAMATH_CALUDE_solve_for_y_l3432_343276

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 3*x + 6 = y + 2) (h2 : x = -5) : y = 44 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3432_343276


namespace NUMINAMATH_CALUDE_parabola_equation_correct_l3432_343205

/-- Represents a parabola with equation ax^2 + bx + c --/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if a parabola has a vertical axis of symmetry --/
def has_vertical_axis_of_symmetry (p : Parabola) : Prop :=
  p.a ≠ 0

/-- Computes the vertex of a parabola --/
def vertex (p : Parabola) : ℚ × ℚ :=
  (- p.b / (2 * p.a), - (p.b^2 - 4*p.a*p.c) / (4 * p.a))

/-- Checks if a point lies on the parabola --/
def point_on_parabola (p : Parabola) (x y : ℚ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- The main theorem --/
theorem parabola_equation_correct :
  let p : Parabola := { a := 2/9, b := -4/3, c := 0 }
  has_vertical_axis_of_symmetry p ∧
  vertex p = (3, -2) ∧
  point_on_parabola p 6 0 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_correct_l3432_343205


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3432_343242

theorem absolute_value_inequality (m : ℝ) :
  (∀ x : ℝ, |x - 3| + |x + 4| ≥ |2*m - 1|) ↔ -3 ≤ m ∧ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3432_343242


namespace NUMINAMATH_CALUDE_five_toppings_from_eight_l3432_343255

theorem five_toppings_from_eight (n m : ℕ) (hn : n = 8) (hm : m = 5) :
  Nat.choose n m = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_toppings_from_eight_l3432_343255


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l3432_343214

theorem simplify_complex_fraction (a b : ℝ) 
  (h1 : a + b ≠ 0) 
  (h2 : a - 2*b ≠ 0) 
  (h3 : a^2 - b^2 ≠ 0) 
  (h4 : a^2 - 4*a*b + 4*b^2 ≠ 0) : 
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l3432_343214


namespace NUMINAMATH_CALUDE_ampersand_composition_l3432_343266

def ampersand_right (x : ℝ) : ℝ := 9 - x

def ampersand_left (x : ℝ) : ℝ := x - 9

theorem ampersand_composition : ampersand_left (ampersand_right 10) = -10 := by
  sorry

end NUMINAMATH_CALUDE_ampersand_composition_l3432_343266


namespace NUMINAMATH_CALUDE_miles_driven_with_thirty_dollars_l3432_343244

theorem miles_driven_with_thirty_dollars (miles_per_gallon : ℝ) (dollars_per_gallon : ℝ) (budget : ℝ) :
  miles_per_gallon = 40 →
  dollars_per_gallon = 4 →
  budget = 30 →
  (budget / dollars_per_gallon) * miles_per_gallon = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_miles_driven_with_thirty_dollars_l3432_343244


namespace NUMINAMATH_CALUDE_carnival_wait_time_l3432_343212

/-- Carnival Ride Wait Time Problem -/
theorem carnival_wait_time (total_time roller_coaster_wait giant_slide_wait : ℕ)
  (roller_coaster_rides tilt_a_whirl_rides giant_slide_rides : ℕ)
  (h1 : total_time = 4 * 60)
  (h2 : roller_coaster_wait = 30)
  (h3 : giant_slide_wait = 15)
  (h4 : roller_coaster_rides = 4)
  (h5 : tilt_a_whirl_rides = 1)
  (h6 : giant_slide_rides = 4) :
  ∃ tilt_a_whirl_wait : ℕ,
    total_time = roller_coaster_wait * roller_coaster_rides +
                 tilt_a_whirl_wait * tilt_a_whirl_rides +
                 giant_slide_wait * giant_slide_rides ∧
    tilt_a_whirl_wait = 60 :=
by sorry

end NUMINAMATH_CALUDE_carnival_wait_time_l3432_343212


namespace NUMINAMATH_CALUDE_additional_oil_amount_l3432_343271

-- Define the original price, reduced price, and additional amount
def original_price : ℝ := 42.75
def reduced_price : ℝ := 34.2
def additional_amount : ℝ := 684

-- Define the price reduction percentage
def price_reduction : ℝ := 0.2

-- Theorem statement
theorem additional_oil_amount :
  reduced_price = original_price * (1 - price_reduction) →
  additional_amount / reduced_price = 20 := by
sorry

end NUMINAMATH_CALUDE_additional_oil_amount_l3432_343271


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l3432_343207

/-- Represents a cube with given dimensions -/
structure Cube where
  size : ℕ
  deriving Repr

/-- Represents the modified cube structure after tunneling -/
structure ModifiedCube where
  original : Cube
  smallCubeSize : ℕ
  removedCenters : ℕ
  deriving Repr

/-- Calculates the surface area of the modified cube structure -/
def surfaceArea (mc : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating the surface area of the specific modified cube structure -/
theorem modified_cube_surface_area :
  let originalCube : Cube := { size := 12 }
  let modifiedCube : ModifiedCube := {
    original := originalCube,
    smallCubeSize := 2,
    removedCenters := 6
  }
  surfaceArea modifiedCube = 1824 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l3432_343207


namespace NUMINAMATH_CALUDE_complement_of_union_is_four_l3432_343228

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3}
def N : Set ℕ := {2, 5}

theorem complement_of_union_is_four :
  (M ∪ N)ᶜ = {4} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_union_is_four_l3432_343228


namespace NUMINAMATH_CALUDE_three_white_marbles_possible_l3432_343234

/-- Represents the possible operations on the urn --/
inductive Operation
  | op1 : Operation  -- Remove 4 black, add 2 black
  | op2 : Operation  -- Remove 3 black and 1 white, add 1 black
  | op3 : Operation  -- Remove 2 black and 2 white, add 2 white and 1 black
  | op4 : Operation  -- Remove 1 black and 3 white, add 3 white
  | op5 : Operation  -- Remove 4 white, add 2 black and 1 white

/-- Represents the state of the urn --/
structure UrnState :=
  (white : ℕ)
  (black : ℕ)

/-- Applies an operation to the urn state --/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.op1 => ⟨state.white, state.black - 2⟩
  | Operation.op2 => ⟨state.white - 1, state.black - 2⟩
  | Operation.op3 => ⟨state.white, state.black - 1⟩
  | Operation.op4 => ⟨state.white, state.black - 1⟩
  | Operation.op5 => ⟨state.white - 3, state.black + 2⟩

/-- Theorem: It's possible to reach a state with 3 white marbles --/
theorem three_white_marbles_possible :
  ∃ (ops : List Operation), 
    let finalState := ops.foldl applyOperation ⟨150, 150⟩
    finalState.white = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_white_marbles_possible_l3432_343234


namespace NUMINAMATH_CALUDE_eulers_formula_two_power_inequality_l3432_343217

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Statement 1: Euler's formula
theorem eulers_formula (x : ℝ) : Complex.exp (i * x) = Complex.cos x + i * Complex.sin x := by sorry

-- Statement 2: Inequality for 2^x
theorem two_power_inequality (x : ℝ) (h : x ≥ 0) : 
  (2 : ℝ) ^ x ≥ 1 + x * Real.log 2 + (x * Real.log 2)^2 / 2 := by sorry

end NUMINAMATH_CALUDE_eulers_formula_two_power_inequality_l3432_343217


namespace NUMINAMATH_CALUDE_tennis_net_max_cuts_l3432_343241

/-- Represents a grid of squares -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Calculates the total number of edges in a grid -/
def total_edges (g : Grid) : ℕ :=
  (g.rows + 1) * g.cols + (g.cols + 1) * g.rows

/-- Calculates the maximum number of edges that can be cut without disconnecting the grid -/
def max_cuttable_edges (g : Grid) : ℕ :=
  (g.rows - 1) * (g.cols - 1)

/-- Theorem stating that for a 100 × 10 grid, the maximum number of cuttable edges is 891 -/
theorem tennis_net_max_cuts :
  let g : Grid := ⟨10, 100⟩
  max_cuttable_edges g = 891 :=
by sorry

end NUMINAMATH_CALUDE_tennis_net_max_cuts_l3432_343241


namespace NUMINAMATH_CALUDE_smallest_root_of_g_l3432_343299

-- Define the function g(x)
def g (x : ℝ) : ℝ := 10 * x^4 - 14 * x^2 + 4

-- State the theorem
theorem smallest_root_of_g :
  ∃ (r : ℝ), g r = 0 ∧ r = -1 ∧ ∀ (x : ℝ), g x = 0 → x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_root_of_g_l3432_343299


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_remainder_l3432_343203

theorem polynomial_division_quotient_remainder 
  (x : ℝ) (h : x ≠ 1) : 
  ∃ (q r : ℝ), 
    x^5 + 5 = (x - 1) * q + r ∧ 
    q = x^4 + x^3 + x^2 + x + 1 ∧ 
    r = 6 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_remainder_l3432_343203


namespace NUMINAMATH_CALUDE_rain_gauge_calculation_l3432_343239

theorem rain_gauge_calculation : 
  let initial_water : ℝ := 2
  let rate_2pm_to_4pm : ℝ := 4
  let rate_4pm_to_7pm : ℝ := 3
  let rate_7pm_to_9pm : ℝ := 0.5
  let duration_2pm_to_4pm : ℝ := 2
  let duration_4pm_to_7pm : ℝ := 3
  let duration_7pm_to_9pm : ℝ := 2
  
  initial_water + 
  (rate_2pm_to_4pm * duration_2pm_to_4pm) + 
  (rate_4pm_to_7pm * duration_4pm_to_7pm) + 
  (rate_7pm_to_9pm * duration_7pm_to_9pm) = 20 := by
sorry

end NUMINAMATH_CALUDE_rain_gauge_calculation_l3432_343239


namespace NUMINAMATH_CALUDE_f_eval_neg_one_l3432_343248

-- Define the polynomials f and g
def f (p q r : ℝ) (x : ℝ) : ℝ := x^4 + 2*x^3 + q*x^2 + 200*x + r
def g (p : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + 2*x + 20

-- State the theorem
theorem f_eval_neg_one (p q r : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g p x = 0 ∧ g p y = 0 ∧ g p z = 0) →
  (∀ x : ℝ, g p x = 0 → f p q r x = 0) →
  f p q r (-1) = -6319 :=
by sorry

end NUMINAMATH_CALUDE_f_eval_neg_one_l3432_343248


namespace NUMINAMATH_CALUDE_gcd_18_24_l3432_343223

theorem gcd_18_24 : Nat.gcd 18 24 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_24_l3432_343223


namespace NUMINAMATH_CALUDE_dot_movement_l3432_343297

-- Define a square
structure Square where
  side : ℝ
  center : ℝ × ℝ

-- Define a point on the square
structure Point where
  x : ℝ
  y : ℝ

-- Define the operations
def fold_diagonal (s : Square) (p : Point) : Point :=
  sorry

def rotate_90_clockwise (s : Square) (p : Point) : Point :=
  sorry

def unfold (s : Square) (p : Point) : Point :=
  sorry

-- Define the initial and final positions
def top_right (s : Square) : Point :=
  sorry

def top_center (s : Square) : Point :=
  sorry

-- Theorem statement
theorem dot_movement (s : Square) :
  let initial_pos := top_right s
  let folded_pos := fold_diagonal s initial_pos
  let rotated_pos := rotate_90_clockwise s folded_pos
  let final_pos := unfold s rotated_pos
  final_pos = top_center s :=
sorry

end NUMINAMATH_CALUDE_dot_movement_l3432_343297


namespace NUMINAMATH_CALUDE_triangle_trigonometric_identities_l3432_343202

/-- 
Given a triangle with sides a, b, c, angles α, β, γ, semi-perimeter p, inradius r, and circumradius R,
this theorem states two trigonometric identities related to the triangle.
-/
theorem triangle_trigonometric_identities 
  (a b c : ℝ) 
  (α β γ : ℝ) 
  (p r R : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = Real.pi)
  (h_semi_perimeter : p = (a + b + c) / 2)
  (h_inradius : r > 0)
  (h_circumradius : R > 0) :
  (Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2 = (p^2 - r^2 - 4*r*R) / (2*R^2) ∧
  4*R^2 * Real.cos α * Real.cos β * Real.cos γ = p^2 - (2*R + r)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identities_l3432_343202


namespace NUMINAMATH_CALUDE_lecture_arrangements_l3432_343291

theorem lecture_arrangements (n : ℕ) (h : n = 3) : Nat.factorial n = 6 := by
  sorry

end NUMINAMATH_CALUDE_lecture_arrangements_l3432_343291


namespace NUMINAMATH_CALUDE_sin_240_degrees_l3432_343286

theorem sin_240_degrees : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l3432_343286


namespace NUMINAMATH_CALUDE_light_bulb_probabilities_l3432_343257

/-- Represents a light bulb factory -/
inductive Factory
| A
| B

/-- Properties of the light bulb inventory -/
structure LightBulbInventory where
  total : ℕ
  factoryA_fraction : ℝ
  factoryB_fraction : ℝ
  factoryA_firstclass_rate : ℝ
  factoryB_firstclass_rate : ℝ

/-- The specific light bulb inventory in the problem -/
def problem_inventory : LightBulbInventory :=
  { total := 50
  , factoryA_fraction := 0.6
  , factoryB_fraction := 0.4
  , factoryA_firstclass_rate := 0.9
  , factoryB_firstclass_rate := 0.8
  }

/-- The probability of randomly selecting a first-class product from Factory A -/
def prob_firstclass_A (inv : LightBulbInventory) : ℝ :=
  inv.factoryA_fraction * inv.factoryA_firstclass_rate

/-- The expected value of first-class products from Factory A when selecting two light bulbs -/
def expected_firstclass_A_two_selections (inv : LightBulbInventory) : ℝ :=
  2 * prob_firstclass_A inv

/-- Main theorem stating the probabilities for the given inventory -/
theorem light_bulb_probabilities :
  prob_firstclass_A problem_inventory = 0.54 ∧
  expected_firstclass_A_two_selections problem_inventory = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_light_bulb_probabilities_l3432_343257


namespace NUMINAMATH_CALUDE_book_purchase_problem_l3432_343210

theorem book_purchase_problem (total_volumes : ℕ) (paperback_price hardcover_price : ℚ) 
  (discount : ℚ) (total_cost : ℚ) :
  total_volumes = 12 ∧ 
  paperback_price = 16 ∧ 
  hardcover_price = 27 ∧ 
  discount = 6 ∧ 
  total_cost = 278 →
  ∃ (h : ℕ), 
    h = 8 ∧ 
    h ≤ total_volumes ∧ 
    (h > 5 → hardcover_price * h + paperback_price * (total_volumes - h) - discount = total_cost) ∧
    (h ≤ 5 → hardcover_price * h + paperback_price * (total_volumes - h) = total_cost) :=
by sorry

end NUMINAMATH_CALUDE_book_purchase_problem_l3432_343210


namespace NUMINAMATH_CALUDE_square_last_digits_l3432_343260

theorem square_last_digits :
  (∃ n : ℕ, n^2 ≡ 444 [ZMOD 1000]) ∧
  (∀ k : ℤ, (1000*k + 38)^2 ≡ 444 [ZMOD 1000]) ∧
  (¬ ∃ n : ℤ, n^2 ≡ 4444 [ZMOD 10000]) := by
  sorry

end NUMINAMATH_CALUDE_square_last_digits_l3432_343260


namespace NUMINAMATH_CALUDE_solve_equation_l3432_343278

theorem solve_equation (x : ℝ) : 3 * x = 2 * x + 6 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3432_343278


namespace NUMINAMATH_CALUDE_santa_gift_combinations_l3432_343253

theorem santa_gift_combinations (n : ℤ) : 
  ∃ k : ℤ, n^5 - n = 30 * k := by sorry

end NUMINAMATH_CALUDE_santa_gift_combinations_l3432_343253


namespace NUMINAMATH_CALUDE_deck_cost_l3432_343282

def rare_cards : ℕ := 19
def uncommon_cards : ℕ := 11
def common_cards : ℕ := 30

def rare_cost : ℚ := 1
def uncommon_cost : ℚ := 0.5
def common_cost : ℚ := 0.25

def total_cost : ℚ := rare_cards * rare_cost + uncommon_cards * uncommon_cost + common_cards * common_cost

theorem deck_cost : total_cost = 32 := by sorry

end NUMINAMATH_CALUDE_deck_cost_l3432_343282


namespace NUMINAMATH_CALUDE_a_greater_than_b_l3432_343298

open Real

theorem a_greater_than_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : exp a + 2*a = exp b + 3*b) : a > b := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l3432_343298


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l3432_343250

theorem arithmetic_mean_after_removal (s : Finset ℕ) (a : ℕ → ℝ) :
  Finset.card s = 75 →
  (Finset.sum s a) / 75 = 60 →
  72 ∈ s →
  48 ∈ s →
  let s' := s.erase 72 ∩ s.erase 48
  (Finset.sum s' a) / 73 = 60 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l3432_343250


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3432_343218

theorem unique_quadratic_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_unique : ∃! x, (5*a + 2*b)*x^2 + a*x + b = 0) : 
  ∃ x, (5*a + 2*b)*x^2 + a*x + b = 0 ∧ x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3432_343218


namespace NUMINAMATH_CALUDE_smallest_days_to_triple_l3432_343206

def borrowed_amount : ℝ := 20
def interest_rate : ℝ := 0.12

def amount_owed (days : ℕ) : ℝ :=
  borrowed_amount + borrowed_amount * interest_rate * days

def is_at_least_triple (days : ℕ) : Prop :=
  amount_owed days ≥ 3 * borrowed_amount

theorem smallest_days_to_triple : 
  (∀ d : ℕ, d < 17 → ¬(is_at_least_triple d)) ∧ 
  (is_at_least_triple 17) :=
sorry

end NUMINAMATH_CALUDE_smallest_days_to_triple_l3432_343206


namespace NUMINAMATH_CALUDE_red_balls_count_l3432_343233

/-- The number of red balls in a box with specific conditions -/
def num_red_balls (total : ℕ) (blue : ℕ) : ℕ :=
  let green := 3 * blue
  let red := (total - blue - green) / 3
  red

/-- Theorem stating that the number of red balls is 4 under given conditions -/
theorem red_balls_count :
  num_red_balls 36 6 = 4 :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l3432_343233


namespace NUMINAMATH_CALUDE_yeongju_shortest_wire_l3432_343277

-- Define the wire lengths in centimeters
def suzy_length : ℝ := 9.8
def yeongju_length : ℝ := 8.9
def youngho_length : ℝ := 9.3

-- Define the conversion factor from cm to mm
def cm_to_mm : ℝ := 10

-- Theorem to prove Yeongju has the shortest wire
theorem yeongju_shortest_wire :
  let suzy_mm := suzy_length * cm_to_mm
  let yeongju_mm := yeongju_length * cm_to_mm
  let youngho_mm := youngho_length * cm_to_mm
  yeongju_mm < suzy_mm ∧ yeongju_mm < youngho_mm :=
by sorry

end NUMINAMATH_CALUDE_yeongju_shortest_wire_l3432_343277


namespace NUMINAMATH_CALUDE_solve_for_y_l3432_343251

-- Define the variables
variable (n x y : ℝ)

-- Define the conditions
def condition1 : Prop := (n + 200 + 300 + x) / 4 = 250
def condition2 : Prop := (300 + 150 + n + x + y) / 5 = 200

-- Theorem statement
theorem solve_for_y (h1 : condition1 n x) (h2 : condition2 n x y) : y = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3432_343251


namespace NUMINAMATH_CALUDE_smaller_square_area_percentage_l3432_343246

/-- Represents a square inscribed in a circle with another smaller square -/
structure InscribedSquares where
  -- Radius of the circle
  r : ℝ
  -- Side length of the larger square
  s : ℝ
  -- Side length of the smaller square
  x : ℝ
  -- The larger square is inscribed in the circle
  h1 : r = s * Real.sqrt 2 / 2
  -- The smaller square has one side coinciding with the larger square
  h2 : x ≤ s
  -- Two vertices of the smaller square are on the circle
  h3 : (s/2 + x)^2 + x^2 = r^2

/-- The theorem stating that the area of the smaller square is 4% of the larger square -/
theorem smaller_square_area_percentage (sq : InscribedSquares) (h : sq.s = 4) :
  (sq.x^2) / (sq.s^2) = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_smaller_square_area_percentage_l3432_343246


namespace NUMINAMATH_CALUDE_eleven_motorcycles_in_lot_l3432_343222

/-- Represents the number of motorcycles in a parking lot --/
def motorcycles_in_lot (total_wheels car_count : ℕ) : ℕ :=
  (total_wheels - 5 * car_count) / 2

/-- Theorem: Given the conditions in the problem, there are 11 motorcycles in the parking lot --/
theorem eleven_motorcycles_in_lot :
  motorcycles_in_lot 117 19 = 11 := by
  sorry

#eval motorcycles_in_lot 117 19

end NUMINAMATH_CALUDE_eleven_motorcycles_in_lot_l3432_343222


namespace NUMINAMATH_CALUDE_vector_parallel_value_l3432_343275

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem vector_parallel_value : 
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ → ℝ × ℝ := λ m ↦ (m, -4)
  ∀ m : ℝ, parallel a (b m) → m = 6 := by
sorry

end NUMINAMATH_CALUDE_vector_parallel_value_l3432_343275


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l3432_343200

/-- Given a geometric sequence with first term b₁ = 2, 
    the minimum value of 3b₂ + 4b₃ is -9/8 -/
theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) :
  b₁ = 2 →
  (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) →
  (∀ c₂ c₃ : ℝ, (∃ s : ℝ, c₂ = 2 * s ∧ c₃ = 2 * s^2) → 
    3 * b₂ + 4 * b₃ ≤ 3 * c₂ + 4 * c₃) →
  3 * b₂ + 4 * b₃ = -9/8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l3432_343200


namespace NUMINAMATH_CALUDE_calculation_difference_l3432_343240

def correct_calculation : ℤ := 12 - (3 + 2) * 2

def incorrect_calculation : ℤ := 12 - 3 + 2 * 2

theorem calculation_difference :
  correct_calculation - incorrect_calculation = -11 := by
  sorry

end NUMINAMATH_CALUDE_calculation_difference_l3432_343240


namespace NUMINAMATH_CALUDE_hiking_team_gloves_l3432_343216

/-- The minimum number of gloves needed for a hiking team -/
theorem hiking_team_gloves (participants : ℕ) (gloves_per_pair : ℕ) : 
  participants = 43 → gloves_per_pair = 2 → participants * gloves_per_pair = 86 := by
  sorry

end NUMINAMATH_CALUDE_hiking_team_gloves_l3432_343216
