import Mathlib

namespace disprove_line_tangent_to_circle_l4012_401218

theorem disprove_line_tangent_to_circle :
  ∃ (a b : ℝ), a^2 + b^2 ≠ 0 ∧ a^2 + b^2 ≠ 1 :=
by
  sorry

end disprove_line_tangent_to_circle_l4012_401218


namespace common_ratio_is_three_l4012_401294

/-- Geometric sequence with sum of first n terms -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_geometric : ∀ n, a (n + 1) = (a 1) * (a 2 / a 1) ^ n

/-- The common ratio of a geometric sequence is 3 given specific conditions -/
theorem common_ratio_is_three (seq : GeometricSequence)
  (h1 : seq.a 5 = 2 * seq.S 4 + 3)
  (h2 : seq.a 6 = 2 * seq.S 5 + 3) :
  seq.a 2 / seq.a 1 = 3 := by
  sorry

end common_ratio_is_three_l4012_401294


namespace geometric_sequence_ratio_l4012_401265

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 3 = 7 →                     -- Condition 1
  (a 1 + a 2 + a 3 = 21) →      -- Condition 2 (S_3 = 21)
  q = -0.5 ∨ q = 1 :=           -- Conclusion
by
  sorry


end geometric_sequence_ratio_l4012_401265


namespace min_slope_tangent_l4012_401276

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - 1 / (a * x)

theorem min_slope_tangent (a : ℝ) (h : a > 0) :
  let k := (deriv (f a)) 1
  ∀ b > 0, k ≤ (deriv (f b)) 1 ↔ a = 1/2 := by
  sorry

end min_slope_tangent_l4012_401276


namespace right_triangle_hypotenuse_equals_area_l4012_401299

theorem right_triangle_hypotenuse_equals_area 
  (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : m ≠ n) : 
  let x : ℝ := (m^2 + n^2) / (m * n * (m^2 - n^2))
  let leg1 : ℝ := (m^2 - n^2) * x
  let leg2 : ℝ := 2 * m * n * x
  let hypotenuse : ℝ := (m^2 + n^2) * x
  let area : ℝ := (1/2) * leg1 * leg2
  hypotenuse = area :=
by sorry

end right_triangle_hypotenuse_equals_area_l4012_401299


namespace max_subsets_of_N_l4012_401234

/-- The set M -/
def M : Finset ℕ := {0, 2, 3, 7}

/-- The set N -/
def N : Finset ℕ := Finset.image (λ (p : ℕ × ℕ) => p.1 * p.2) (M.product M)

/-- Theorem: The maximum number of subsets of N is 128 -/
theorem max_subsets_of_N : Finset.card (Finset.powerset N) = 128 := by
  sorry

end max_subsets_of_N_l4012_401234


namespace brick_length_l4012_401263

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: Given a rectangular prism with width 4, height 3, and surface area 164, its length is 10 -/
theorem brick_length (w h SA : ℝ) (hw : w = 4) (hh : h = 3) (hSA : SA = 164) :
  ∃ l : ℝ, surface_area l w h = SA ∧ l = 10 :=
sorry

end brick_length_l4012_401263


namespace triangle_area_l4012_401241

/-- The area of a triangle with vertices (5, -2), (10, 5), and (5, 5) is 17.5 square units. -/
theorem triangle_area : 
  let v1 : ℝ × ℝ := (5, -2)
  let v2 : ℝ × ℝ := (10, 5)
  let v3 : ℝ × ℝ := (5, 5)
  let area := (1/2) * abs ((v2.1 - v1.1) * (v3.2 - v1.2) - (v3.1 - v1.1) * (v2.2 - v1.2))
  area = 17.5 := by sorry

end triangle_area_l4012_401241


namespace parabola_chord_theorem_l4012_401230

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y = x^2 -/
def parabola (p : Point) : Prop := p.y = p.x^2

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Checks if a point divides a line segment in a given ratio -/
def divides_in_ratio (p1 p2 p : Point) (m n : ℝ) : Prop :=
  n * (p.x - p1.x) = m * (p2.x - p.x) ∧ n * (p.y - p1.y) = m * (p2.y - p.y)

theorem parabola_chord_theorem (A B C : Point) :
  parabola A ∧ parabola B ∧  -- A and B lie on the parabola
  C.x = 0 ∧ C.y = 15 ∧  -- C is on y-axis with y-coordinate 15
  collinear A B C ∧  -- A, B, and C are collinear
  divides_in_ratio A B C 5 3 →  -- C divides AB in ratio 5:3
  ((A.x = -5 ∧ B.x = 3) ∨ (A.x = 5 ∧ B.x = -3)) := by sorry

end parabola_chord_theorem_l4012_401230


namespace little_john_sweets_expenditure_l4012_401247

theorem little_john_sweets_expenditure
  (initial_amount : ℚ)
  (final_amount : ℚ)
  (amount_per_friend : ℚ)
  (num_friends : ℕ)
  (h1 : initial_amount = 8.5)
  (h2 : final_amount = 4.85)
  (h3 : amount_per_friend = 1.2)
  (h4 : num_friends = 2) :
  initial_amount - final_amount - (↑num_friends * amount_per_friend) = 1.25 :=
by sorry

end little_john_sweets_expenditure_l4012_401247


namespace set_inclusion_implies_a_value_l4012_401239

theorem set_inclusion_implies_a_value (A B : Set ℤ) (a : ℤ) : 
  A = {0, 1} → 
  B = {-1, 0, a+3} → 
  A ⊆ B → 
  a = -2 := by sorry

end set_inclusion_implies_a_value_l4012_401239


namespace unique_integer_solution_l4012_401242

theorem unique_integer_solution : ∃! (x : ℕ), x > 0 ∧ (4 * x)^2 - 2 * x = 3178 :=
by
  -- The proof would go here
  sorry

end unique_integer_solution_l4012_401242


namespace coefficient_x5_proof_l4012_401293

/-- The coefficient of x^5 in the expansion of (1+x^3)(1-2x)^6 -/
def coefficient_x5 : ℤ := -132

/-- The expansion of (1+x^3)(1-2x)^6 -/
def expansion (x : ℝ) : ℝ := (1 + x^3) * (1 - 2*x)^6

theorem coefficient_x5_proof : 
  (deriv^[5] expansion 0) / 120 = coefficient_x5 := by sorry

end coefficient_x5_proof_l4012_401293


namespace imaginary_part_of_i_minus_one_l4012_401249

def imaginary_part (z : ℂ) : ℝ := z.im

theorem imaginary_part_of_i_minus_one : imaginary_part (Complex.I - 1) = 1 := by
  sorry

end imaginary_part_of_i_minus_one_l4012_401249


namespace rotten_eggs_count_l4012_401223

theorem rotten_eggs_count (total : ℕ) (prob : ℚ) (h_total : total = 36) (h_prob : prob = 47619047619047615 / 10000000000000000) :
  ∃ (rotten : ℕ), rotten = 3 ∧
    (rotten : ℚ) / total * ((rotten : ℚ) - 1) / (total - 1) = prob :=
by sorry

end rotten_eggs_count_l4012_401223


namespace children_retaking_test_l4012_401206

theorem children_retaking_test (total : Float) (passed : Float) 
  (h1 : total = 698.0) (h2 : passed = 105.0) : 
  total - passed = 593.0 := by
  sorry

end children_retaking_test_l4012_401206


namespace abc_inequality_l4012_401217

theorem abc_inequality (a b c : ℝ) (h1 : a + b + c = 1) (h2 : c = -1) :
  a * b + a * c + b * c ≤ -1 := by
  sorry

end abc_inequality_l4012_401217


namespace total_song_requests_l4012_401221

/-- Represents the total number of song requests --/
def T : ℕ := 30

/-- Theorem stating that the total number of song requests is 30 --/
theorem total_song_requests :
  T = 30 ∧
  T = (1/2 : ℚ) * T + (1/6 : ℚ) * T + 5 + 2 + 1 + 2 :=
by sorry

end total_song_requests_l4012_401221


namespace katyas_age_l4012_401215

def insert_zero (n : ℕ) : ℕ :=
  (n / 10) * 100 + (n % 10)

theorem katyas_age :
  ∃! n : ℕ, n ≥ 10 ∧ n < 100 ∧ 6 * n = insert_zero n ∧ n = 18 :=
by sorry

end katyas_age_l4012_401215


namespace ten_row_triangle_pieces_l4012_401246

/-- Calculates the number of unit rods in an n-row triangle -/
def unitRods (n : ℕ) : ℕ := n * (3 + 3 * n) / 2

/-- Calculates the number of connectors in an n-row triangle -/
def connectors (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the total number of pieces in an n-row triangle -/
def totalPieces (n : ℕ) : ℕ := unitRods n + connectors (n + 1)

theorem ten_row_triangle_pieces :
  totalPieces 10 = 231 ∧ unitRods 2 = 9 ∧ connectors 3 = 6 := by sorry

end ten_row_triangle_pieces_l4012_401246


namespace complex_exponential_sum_l4012_401259

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (1/3 : ℂ) + (2/5 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (1/3 : ℂ) - (2/5 : ℂ) * Complex.I :=
by sorry

end complex_exponential_sum_l4012_401259


namespace ascending_order_l4012_401224

theorem ascending_order (a b c : ℝ) (ha : a = 60.7) (hb : b = 0.76) (hc : c = Real.log 0.76) :
  c < b ∧ b < a := by sorry

end ascending_order_l4012_401224


namespace joe_weight_lifting_ratio_l4012_401278

/-- Joe's weight-lifting competition problem -/
theorem joe_weight_lifting_ratio :
  ∀ (total first second : ℕ),
  total = first + second →
  first = 600 →
  total = 1500 →
  first = 2 * (second - 300) →
  first = second :=
λ total first second h1 h2 h3 h4 =>
  sorry

end joe_weight_lifting_ratio_l4012_401278


namespace potato_bundle_price_l4012_401225

/-- Calculates the price of potato bundles given the harvest and sales information --/
theorem potato_bundle_price
  (potato_count : ℕ)
  (potato_bundle_size : ℕ)
  (carrot_count : ℕ)
  (carrot_bundle_size : ℕ)
  (carrot_bundle_price : ℚ)
  (total_revenue : ℚ)
  (h1 : potato_count = 250)
  (h2 : potato_bundle_size = 25)
  (h3 : carrot_count = 320)
  (h4 : carrot_bundle_size = 20)
  (h5 : carrot_bundle_price = 2)
  (h6 : total_revenue = 51) :
  (total_revenue - (carrot_count / carrot_bundle_size * carrot_bundle_price)) / (potato_count / potato_bundle_size) = 1.9 := by
sorry

end potato_bundle_price_l4012_401225


namespace pentagon_area_sum_l4012_401255

theorem pentagon_area_sum (a b : ℤ) (h1 : 0 < b) (h2 : b < a) : 
  let P := (a, b)
  let Q := (b, a)
  let R := (-b, a)
  let S := (-b, -a)
  let T := (b, -a)
  let pentagon_area := a * (3 * b + a)
  pentagon_area = 792 → a + b = 45 := by
sorry

end pentagon_area_sum_l4012_401255


namespace study_tour_students_l4012_401207

/-- Represents the number of students participating in the study tour. -/
def num_students : ℕ := 46

/-- Represents the number of dormitories. -/
def num_dormitories : ℕ := 6

theorem study_tour_students :
  (∃ (n : ℕ), n = num_dormitories ∧
    6 * n + 10 = num_students ∧
    8 * (n - 1) + 4 < num_students ∧
    num_students < 8 * (n - 1) + 8) :=
by sorry

end study_tour_students_l4012_401207


namespace magnitude_of_4_minus_15i_l4012_401253

theorem magnitude_of_4_minus_15i :
  let z : ℂ := 4 - 15 * I
  Complex.abs z = Real.sqrt 241 := by
sorry

end magnitude_of_4_minus_15i_l4012_401253


namespace existence_of_m_l4012_401251

theorem existence_of_m (a b : ℝ) (h : a < b) :
  ∃ m : ℝ, m * a > m * b :=
sorry

end existence_of_m_l4012_401251


namespace total_pennies_thrown_l4012_401222

/-- The number of pennies thrown by each person -/
structure PennyThrowers where
  rachelle : ℕ
  gretchen : ℕ
  rocky : ℕ
  max : ℕ
  taylor : ℕ

/-- The conditions of the penny-throwing problem -/
def penny_throwing_conditions (pt : PennyThrowers) : Prop :=
  pt.rachelle = 720 ∧
  pt.gretchen = pt.rachelle / 2 ∧
  pt.rocky = pt.gretchen / 3 ∧
  pt.max = pt.rocky * 4 ∧
  pt.taylor = pt.max / 5

/-- The theorem stating that the total number of pennies thrown is 1776 -/
theorem total_pennies_thrown (pt : PennyThrowers) 
  (h : penny_throwing_conditions pt) : 
  pt.rachelle + pt.gretchen + pt.rocky + pt.max + pt.taylor = 1776 := by
  sorry


end total_pennies_thrown_l4012_401222


namespace wonderful_quadratic_range_l4012_401244

/-- A function is wonderful on a domain if it's monotonic and there exists an interval [a,b] in the domain
    such that the range of f on [a,b] is exactly [a,b] --/
def IsWonderful (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  Monotone f ∧ ∃ a b, a < b ∧ Set.Icc a b ⊆ D ∧
    (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b) ∧
    (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y)

theorem wonderful_quadratic_range (m : ℝ) :
  IsWonderful (fun x => x^2 + m) (Set.Iic 0) →
  m ∈ Set.Ioo (-1) (-3/4) :=
sorry

end wonderful_quadratic_range_l4012_401244


namespace dog_park_problem_l4012_401203

theorem dog_park_problem (total_dogs : ℕ) (spotted_dogs : ℕ) (pointy_eared_dogs : ℕ) :
  (2 * spotted_dogs = total_dogs) →  -- Half of the dogs have spots
  (5 * pointy_eared_dogs = total_dogs) →  -- 1/5 of the dogs have pointy ears
  (spotted_dogs = 15) →  -- 15 dogs have spots
  pointy_eared_dogs = 6 :=
by sorry

end dog_park_problem_l4012_401203


namespace first_hour_speed_l4012_401238

theorem first_hour_speed 
  (total_time : ℝ) 
  (average_speed : ℝ) 
  (second_part_time : ℝ) 
  (second_part_speed : ℝ) 
  (h1 : total_time = 4) 
  (h2 : average_speed = 55) 
  (h3 : second_part_time = 3) 
  (h4 : second_part_speed = 60) : 
  ∃ (first_hour_speed : ℝ), first_hour_speed = 40 ∧ 
    average_speed * total_time = first_hour_speed * (total_time - second_part_time) + 
      second_part_speed * second_part_time :=
by sorry

end first_hour_speed_l4012_401238


namespace quadratic_function_m_not_two_l4012_401277

/-- Given a quadratic function y = a(x-m)^2 where a > 0, 
    if it passes through points (-1,p) and (3,q) where p < q, 
    then m ≠ 2 -/
theorem quadratic_function_m_not_two 
  (a m p q : ℝ) 
  (h1 : a > 0)
  (h2 : a * (-1 - m)^2 = p)
  (h3 : a * (3 - m)^2 = q)
  (h4 : p < q) : 
  m ≠ 2 := by
  sorry

end quadratic_function_m_not_two_l4012_401277


namespace inequality_solution_set_l4012_401296

theorem inequality_solution_set : 
  {x : ℝ | 5 - x^2 > 4*x} = Set.Ioo (-5 : ℝ) 1 := by sorry

end inequality_solution_set_l4012_401296


namespace rhombus_perimeter_l4012_401210

/-- Given a rhombus with diagonals of 10 inches and 24 inches, its perimeter is 52 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end rhombus_perimeter_l4012_401210


namespace remaining_battery_life_is_eight_hours_l4012_401250

/-- Represents the battery life of a phone -/
structure PhoneBattery where
  inactiveLife : ℝ  -- Battery life when not in use (in hours)
  activeLife : ℝ    -- Battery life when used constantly (in hours)

/-- Calculates the remaining battery life -/
def remainingBatteryLife (battery : PhoneBattery) 
  (usedTime : ℝ)     -- Time the phone has been used (in hours)
  (totalTime : ℝ)    -- Total time since last charge (in hours)
  : ℝ :=
  sorry

/-- Theorem: Given the conditions, the remaining battery life is 8 hours -/
theorem remaining_battery_life_is_eight_hours 
  (battery : PhoneBattery)
  (h1 : battery.inactiveLife = 18)
  (h2 : battery.activeLife = 2)
  (h3 : remainingBatteryLife battery 0.5 6 = 8) :
  ∃ (t : ℝ), t = 8 ∧ remainingBatteryLife battery 0.5 6 = t := by
  sorry

end remaining_battery_life_is_eight_hours_l4012_401250


namespace arithmetic_sequence_ratio_l4012_401261

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ratio
  (a : ℕ → ℝ)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_arithmetic : arithmetic_sequence a)
  (h_sub_sequence : ∃ k : ℝ, a 1 + k = (1/2) * a 3 ∧ (1/2) * a 3 + k = 2 * a 2) :
  (a 8 + a 9) / (a 7 + a 8) = Real.sqrt 2 + 1 := by
sorry

end arithmetic_sequence_ratio_l4012_401261


namespace min_value_expression_l4012_401270

theorem min_value_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (min : ℝ), min = 2 * Real.sqrt 3 ∧
  ∀ (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0),
    x^2 + y^2 + z^2 + 1/x^2 + y/x + z/y ≥ min ∧
    ∃ (a' b' c' : ℝ) (ha' : a' ≠ 0) (hb' : b' ≠ 0) (hc' : c' ≠ 0),
      a'^2 + b'^2 + c'^2 + 1/a'^2 + b'/a' + c'/b' = min :=
by sorry

end min_value_expression_l4012_401270


namespace equation_solution_l4012_401281

theorem equation_solution :
  ∃ x : ℚ, (x - 60) / 3 = (5 - 3 * x) / 4 ∧ x = 255 / 13 := by
  sorry

end equation_solution_l4012_401281


namespace find_5b_l4012_401235

theorem find_5b (a b : ℚ) (h1 : 3 * a + 4 * b = 0) (h2 : a = b - 3) : 5 * b = 45 / 7 := by
  sorry

end find_5b_l4012_401235


namespace polynomial_evaluation_part1_polynomial_evaluation_part2_l4012_401228

/-- Part 1: Polynomial evaluation given a condition -/
theorem polynomial_evaluation_part1 (x : ℝ) (h : x^2 - x = 3) :
  x^4 - 2*x^3 + 3*x^2 - 2*x + 2 = 17 := by
  sorry

/-- Part 2: Polynomial evaluation given a condition -/
theorem polynomial_evaluation_part2 (x y : ℝ) (h : x^2 + y^2 = 1) :
  2*x^4 + 3*x^2*y^2 + y^4 + y^2 = 2 := by
  sorry

end polynomial_evaluation_part1_polynomial_evaluation_part2_l4012_401228


namespace right_triangle_area_l4012_401254

theorem right_triangle_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_ratio : a / b = 3 / 4) (h_hypotenuse : c = 10) : 
  (1 / 2) * a * b = 24 := by
  sorry

end right_triangle_area_l4012_401254


namespace scale_length_calculation_l4012_401283

/-- Calculates the total length of a scale given the number of equal parts and the length of each part. -/
def totalScaleLength (numParts : ℕ) (partLength : ℝ) : ℝ :=
  numParts * partLength

/-- Theorem: The total length of a scale with 5 equal parts, each 25 inches long, is 125 inches. -/
theorem scale_length_calculation :
  totalScaleLength 5 25 = 125 := by
  sorry

end scale_length_calculation_l4012_401283


namespace min_value_abs_sum_l4012_401271

theorem min_value_abs_sum (x : ℝ) : 
  |x - 4| + |x + 7| + |x - 5| ≥ 4 ∧ ∃ y : ℝ, |y - 4| + |y + 7| + |y - 5| = 4 :=
sorry

end min_value_abs_sum_l4012_401271


namespace impossible_conditions_l4012_401264

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let (xa, ya) := A
  let (xb, yb) := B
  let (xc, yc) := C
  (xc - xa) * (yb - ya) = (xb - xa) * (yc - ya) ∧  -- collinearity check
  (xb - xa)^2 + (yb - ya)^2 = 144 ∧                -- AB = 12
  (xc - xb) * (xa - xb) + (yc - yb) * (ya - yb) = 0 -- ∠ABC = 90°

-- Define a point inside the triangle
def InsideTriangle (P : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xa, ya) := A
  let (xb, yb) := B
  let (xc, yc) := C
  (xp - xa) * (yb - ya) < (xb - xa) * (yp - ya) ∧
  (xp - xb) * (yc - yb) < (xc - xb) * (yp - yb) ∧
  (xp - xc) * (ya - yc) < (xa - xc) * (yp - yc)

-- Define the point D on AC
def PointOnAC (D : ℝ × ℝ) (A C : ℝ × ℝ) : Prop :=
  let (xd, yd) := D
  let (xa, ya) := A
  let (xc, yc) := C
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ xd = xa + t * (xc - xa) ∧ yd = ya + t * (yc - ya)

-- Define P being on BD
def POnBD (P : ℝ × ℝ) (B D : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xb, yb) := B
  let (xd, yd) := D
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ xp = xb + t * (xd - xb) ∧ yp = yb + t * (yd - yb)

-- Define BD > 6√2
def BDGreaterThan6Sqrt2 (B D : ℝ × ℝ) : Prop :=
  let (xb, yb) := B
  let (xd, yd) := D
  (xd - xb)^2 + (yd - yb)^2 > 72

-- Define P above the median of BC
def PAboveMedianBC (P : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xa, ya) := A
  let (xb, yb) := B
  let (xc, yc) := C
  let xm := (xb + xc) / 2
  let ym := (yb + yc) / 2
  (xp - xa) * (ym - ya) > (xm - xa) * (yp - ya)

theorem impossible_conditions (A B C : ℝ × ℝ) (h : Triangle A B C) :
  ¬∃ (P D : ℝ × ℝ), 
    InsideTriangle P A B C ∧ 
    PointOnAC D A C ∧ 
    POnBD P B D ∧ 
    BDGreaterThan6Sqrt2 B D ∧ 
    PAboveMedianBC P A B C :=
  sorry

end impossible_conditions_l4012_401264


namespace symmetry_point_l4012_401284

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of symmetry with respect to a horizontal line -/
def isSymmetricHorizontal (p q : Point2D) (y_line : ℝ) : Prop :=
  p.x = q.x ∧ y_line - p.y = q.y - y_line

theorem symmetry_point :
  let p : Point2D := ⟨3, -2⟩
  let q : Point2D := ⟨3, 4⟩
  let y_line : ℝ := 1
  isSymmetricHorizontal p q y_line :=
by
  sorry

end symmetry_point_l4012_401284


namespace cost_of_juices_l4012_401298

/-- The cost of juices and sandwiches problem -/
theorem cost_of_juices (sandwich_cost juice_cost : ℚ) : 
  (2 * sandwich_cost = 6) →
  (sandwich_cost + juice_cost = 5) →
  (5 * juice_cost = 10) :=
by
  sorry

end cost_of_juices_l4012_401298


namespace stuffed_animals_difference_l4012_401274

theorem stuffed_animals_difference (mckenna kenley tenly : ℕ) : 
  mckenna = 34 →
  kenley = 2 * mckenna →
  mckenna + kenley + tenly = 175 →
  tenly - kenley = 5 := by
  sorry

end stuffed_animals_difference_l4012_401274


namespace polynomial_factorization_l4012_401287

theorem polynomial_factorization (m n : ℝ) : 
  (∀ x, x^2 + m*x + n = (x + 1)*(x + 3)) → m - n = 1 := by
  sorry

end polynomial_factorization_l4012_401287


namespace all_divisors_end_in_one_l4012_401233

theorem all_divisors_end_in_one (n : ℕ+) :
  ∀ d : ℕ, d > 0 → d ∣ ((10^(5^n.val) - 1) / 9) → d % 10 = 1 := by
  sorry

end all_divisors_end_in_one_l4012_401233


namespace adjacent_zero_point_functions_range_l4012_401229

def adjacent_zero_point_functions (f g : ℝ → ℝ) : Prop :=
  ∀ (α β : ℝ), f α = 0 → g β = 0 → |α - β| ≤ 1

def f (x : ℝ) : ℝ := x - 1

def g (a x : ℝ) : ℝ := x^2 - a*x - a + 3

theorem adjacent_zero_point_functions_range (a : ℝ) :
  adjacent_zero_point_functions f (g a) → a ∈ Set.Icc 2 (7/3) := by
  sorry

end adjacent_zero_point_functions_range_l4012_401229


namespace harveys_steaks_l4012_401209

theorem harveys_steaks (initial_steaks : ℕ) 
  (h1 : initial_steaks - 17 = 12) 
  (h2 : 17 ≥ 4) : initial_steaks = 33 :=
by
  sorry

end harveys_steaks_l4012_401209


namespace john_supermarket_spending_l4012_401216

def supermarket_spending (total : ℚ) : Prop :=
  let fruits_veg := (1 : ℚ) / 5 * total
  let meat := (1 : ℚ) / 3 * total
  let bakery := (1 : ℚ) / 10 * total
  let dairy := (1 : ℚ) / 6 * total
  let candy_magazine := total - (fruits_veg + meat + bakery + dairy)
  let magazine := (15 : ℚ) / 4  -- $3.75 as a rational number
  candy_magazine = (29 : ℚ) / 2 ∧  -- $14.50 as a rational number
  candy_magazine - magazine = (43 : ℚ) / 4  -- $10.75 as a rational number

theorem john_supermarket_spending :
  ∃ (total : ℚ), supermarket_spending total ∧ total = (145 : ℚ) / 2 :=
sorry

end john_supermarket_spending_l4012_401216


namespace problem_solution_l4012_401204

-- Define the function f
def f (m : ℕ) (x : ℝ) : ℝ := |x - m| + |x|

-- State the theorem
theorem problem_solution (m : ℕ) (α β : ℝ) 
  (h1 : m > 0)
  (h2 : ∃ x : ℝ, f m x < 2)
  (h3 : α > 1)
  (h4 : β > 1)
  (h5 : f m α + f m β = 6) :
  (m = 1) ∧ ((4 / α) + (1 / β) ≥ 9 / 4) := by
  sorry

end problem_solution_l4012_401204


namespace sine_cosine_arithmetic_progression_l4012_401285

theorem sine_cosine_arithmetic_progression
  (x y z : ℝ)
  (h_sin_ap : 2 * Real.sin y = Real.sin x + Real.sin z)
  (h_sin_increasing : Real.sin x < Real.sin y ∧ Real.sin y < Real.sin z) :
  ¬ (2 * Real.cos y = Real.cos x + Real.cos z) :=
by sorry

end sine_cosine_arithmetic_progression_l4012_401285


namespace min_width_proof_l4012_401212

/-- The minimum width of a rectangular area satisfying the given conditions -/
def min_width : ℝ := 10

/-- The length of the rectangular area -/
def length (w : ℝ) : ℝ := w + 15

/-- The area of the rectangular region -/
def area (w : ℝ) : ℝ := w * length w

theorem min_width_proof :
  (∀ w : ℝ, w > 0 → area w ≥ 200 → w ≥ min_width) ∧
  (area min_width ≥ 200) :=
sorry

end min_width_proof_l4012_401212


namespace smallest_m_theorem_l4012_401275

/-- The smallest positive value of m for which the equation 12x^2 - mx - 360 = 0 has integral solutions -/
def smallest_m : ℕ := 12

/-- The equation 12x^2 - mx - 360 = 0 has integral solutions -/
def has_integral_solutions (m : ℤ) : Prop :=
  ∃ x : ℤ, 12 * x^2 - m * x - 360 = 0

/-- The theorem stating that the smallest positive m for which the equation has integral solutions is 12 -/
theorem smallest_m_theorem : 
  (∀ m : ℕ, m < smallest_m → ¬(has_integral_solutions m)) ∧ 
  (has_integral_solutions smallest_m) :=
sorry

end smallest_m_theorem_l4012_401275


namespace intersection_nonempty_implies_a_less_than_one_l4012_401213

theorem intersection_nonempty_implies_a_less_than_one (a : ℝ) : 
  let M := {x : ℝ | x ≤ 1}
  let P := {x : ℝ | x > a}
  (M ∩ P).Nonempty → a < 1 := by
  sorry

end intersection_nonempty_implies_a_less_than_one_l4012_401213


namespace characterize_function_l4012_401227

open Set Function Real

-- Define the interval (1,∞)
def OpenOneInfty : Set ℝ := {x : ℝ | x > 1}

-- Define the property for the function
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ OpenOneInfty → y ∈ OpenOneInfty →
    (x^2 ≤ y ∧ y ≤ x^3) → ((f x)^2 ≤ f y ∧ f y ≤ (f x)^3)

-- The main theorem
theorem characterize_function :
  ∀ f : ℝ → ℝ, (∀ x, x ∈ OpenOneInfty → f x ∈ OpenOneInfty) →
    SatisfiesProperty f →
    ∃ k : ℝ, k > 0 ∧ ∀ x ∈ OpenOneInfty, f x = exp (k * log x) := by
  sorry

end characterize_function_l4012_401227


namespace lindas_tv_cost_l4012_401269

def lindas_problem (original_savings : ℝ) (furniture_fraction : ℝ) : ℝ :=
  original_savings * (1 - furniture_fraction)

theorem lindas_tv_cost :
  lindas_problem 500 (4/5) = 100 := by sorry

end lindas_tv_cost_l4012_401269


namespace powerlifting_bodyweight_l4012_401282

theorem powerlifting_bodyweight (initial_total : ℝ) (total_gain_percent : ℝ) (weight_gain : ℝ) (final_ratio : ℝ) :
  initial_total = 2200 →
  total_gain_percent = 15 →
  weight_gain = 8 →
  final_ratio = 10 →
  ∃ initial_weight : ℝ,
    initial_weight > 0 ∧
    (initial_total * (1 + total_gain_percent / 100)) / (initial_weight + weight_gain) = final_ratio ∧
    initial_weight = 245 := by
  sorry

#check powerlifting_bodyweight

end powerlifting_bodyweight_l4012_401282


namespace average_marks_combined_classes_l4012_401243

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 30 →
  n2 = 50 →
  avg1 = 40 →
  avg2 = 80 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 65 :=
by sorry

end average_marks_combined_classes_l4012_401243


namespace fuel_tank_capacity_l4012_401202

/-- Represents a cylindrical fuel tank -/
structure FuelTank where
  capacity : ℝ
  initial_percentage : ℝ
  initial_volume : ℝ

/-- Theorem stating the capacity of the fuel tank -/
theorem fuel_tank_capacity (tank : FuelTank)
  (h1 : tank.initial_percentage = 0.25)
  (h2 : tank.initial_volume = 60)
  : tank.capacity = 240 := by
  sorry

#check fuel_tank_capacity

end fuel_tank_capacity_l4012_401202


namespace floor_equation_solution_l4012_401273

theorem floor_equation_solution (x : ℝ) : 
  (Int.floor (2 * x) + Int.floor (3 * x) = 8 * x - 7 / 2) ↔ (x = 13 / 16 ∨ x = 17 / 16) :=
sorry

end floor_equation_solution_l4012_401273


namespace length_of_AB_prime_l4012_401262

-- Define the points
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 9)

-- Define the condition for A' and B' to be on the line y = x
def on_diagonal (p : ℝ × ℝ) : Prop := p.1 = p.2

-- Define the condition for AA' and BB' to intersect at C
def intersect_at_C (A' B' : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ,
    C = (A.1 + t₁ * (A'.1 - A.1), A.2 + t₁ * (A'.2 - A.2)) ∧
    C = (B.1 + t₂ * (B'.1 - B.1), B.2 + t₂ * (B'.2 - B.2))

-- State the theorem
theorem length_of_AB_prime : 
  ∃ A' B' : ℝ × ℝ,
    on_diagonal A' ∧ 
    on_diagonal B' ∧ 
    intersect_at_C A' B' ∧ 
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2.5 * Real.sqrt 2 := by
  sorry

end length_of_AB_prime_l4012_401262


namespace fraction_of_length_equality_l4012_401245

theorem fraction_of_length_equality : (2 / 7 : ℚ) * 3 = (3 / 7 : ℚ) * 2 := by sorry

end fraction_of_length_equality_l4012_401245


namespace exists_inner_sum_greater_than_outer_sum_l4012_401252

/-- Represents a triangular pyramid (tetrahedron) --/
structure TriangularPyramid where
  base_edge_length : ℝ
  lateral_edge_length : ℝ

/-- Calculates the sum of edge lengths of a triangular pyramid --/
def sum_of_edges (pyramid : TriangularPyramid) : ℝ :=
  3 * pyramid.base_edge_length + 3 * pyramid.lateral_edge_length

/-- Represents two triangular pyramids with a common base, where one is inside the other --/
structure NestedPyramids where
  outer : TriangularPyramid
  inner : TriangularPyramid
  inner_inside_outer : inner.base_edge_length = outer.base_edge_length
  inner_lateral_edge_shorter : inner.lateral_edge_length < outer.lateral_edge_length

/-- Theorem: There exist nested pyramids where the sum of edges of the inner pyramid
    is greater than the sum of edges of the outer pyramid --/
theorem exists_inner_sum_greater_than_outer_sum :
  ∃ (np : NestedPyramids), sum_of_edges np.inner > sum_of_edges np.outer := by
  sorry


end exists_inner_sum_greater_than_outer_sum_l4012_401252


namespace decreasing_quadratic_implies_a_equals_negative_one_l4012_401205

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 1

-- State the theorem
theorem decreasing_quadratic_implies_a_equals_negative_one :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → y < 2 → f a x > f a y) → a = -1 := by
  sorry

end decreasing_quadratic_implies_a_equals_negative_one_l4012_401205


namespace range_of_a_l4012_401267

theorem range_of_a (p q : Prop) (a : ℝ) 
  (hp : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x)
  (hq : ∃ x : ℝ, x^2 - 4*x + a ≤ 0) :
  a ∈ Set.Icc (Real.exp 1) 4 := by
  sorry

end range_of_a_l4012_401267


namespace perpendicular_implies_parallel_skew_perpendicular_parallel_implies_perpendicular_l4012_401279

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- Define non-coincidence for lines and planes
variable (non_coincident_lines : Line → Line → Prop)
variable (non_coincident_planes : Plane → Plane → Prop)

variable (m n : Line)
variable (α β γ : Plane)

-- Axioms
axiom non_coincident_mn : non_coincident_lines m n
axiom non_coincident_αβ : non_coincident_planes α β
axiom non_coincident_βγ : non_coincident_planes β γ
axiom non_coincident_αγ : non_coincident_planes α γ

-- Theorem 1
theorem perpendicular_implies_parallel 
  (h1 : perpendicular m α) (h2 : perpendicular m β) : 
  plane_parallel α β :=
sorry

-- Theorem 2
theorem skew_perpendicular_parallel_implies_perpendicular 
  (h1 : skew m n)
  (h2 : perpendicular m α) (h3 : parallel m β)
  (h4 : perpendicular n β) (h5 : parallel n α) :
  plane_perpendicular α β :=
sorry

end perpendicular_implies_parallel_skew_perpendicular_parallel_implies_perpendicular_l4012_401279


namespace log3_negative_implies_x_negative_but_not_conversely_l4012_401258

-- Define the logarithm function with base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Theorem statement
theorem log3_negative_implies_x_negative_but_not_conversely :
  (∀ x : ℝ, log3 (x + 1) < 0 → x < 0) ∧
  (∃ x : ℝ, x < 0 ∧ log3 (x + 1) ≥ 0) :=
sorry

end log3_negative_implies_x_negative_but_not_conversely_l4012_401258


namespace katie_sold_four_pastries_l4012_401272

/-- The number of pastries sold at a bake sale -/
def pastries_sold (cupcakes cookies leftover : ℕ) : ℕ :=
  cupcakes + cookies - leftover

/-- Proof that Katie sold 4 pastries at the bake sale -/
theorem katie_sold_four_pastries :
  pastries_sold 7 5 8 = 4 := by
  sorry

end katie_sold_four_pastries_l4012_401272


namespace bird_reserve_theorem_l4012_401240

/-- Represents the composition of birds in the Goshawk-Eurasian Nature Reserve -/
structure BirdReserve where
  total : ℝ
  hawks : ℝ
  paddyfield_warblers : ℝ
  kingfishers : ℝ

/-- The conditions of the bird reserve -/
def reserve_conditions (b : BirdReserve) : Prop :=
  b.hawks = 0.3 * b.total ∧
  b.paddyfield_warblers = 0.4 * (b.total - b.hawks) ∧
  b.kingfishers = 0.25 * b.paddyfield_warblers

/-- The theorem to be proved -/
theorem bird_reserve_theorem (b : BirdReserve) 
  (h : reserve_conditions b) : 
  (b.total - b.hawks - b.paddyfield_warblers - b.kingfishers) / b.total = 0.35 := by
  sorry

end bird_reserve_theorem_l4012_401240


namespace sum_of_digits_of_square_l4012_401220

def square_of_1222222221 : ℕ := 1493822537037038241

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_square :
  sum_of_digits square_of_1222222221 = 80 := by
  sorry

end sum_of_digits_of_square_l4012_401220


namespace town_population_l4012_401232

theorem town_population (total_population : ℕ) 
  (h1 : total_population < 6000)
  (h2 : ∃ (boys girls : ℕ), girls = (11 * boys) / 10 ∧ boys + girls = total_population * 10 / 21)
  (h3 : ∃ (women men : ℕ), men = (23 * women) / 20 ∧ women + men = total_population * 20 / 43)
  (h4 : ∃ (children adults : ℕ), children = (6 * adults) / 5 ∧ children + adults = total_population) :
  total_population = 3311 := by
sorry

end town_population_l4012_401232


namespace banana_cantaloupe_cost_l4012_401214

def cost_problem (apple banana cantaloupe date : ℝ) : Prop :=
  apple + banana + cantaloupe + date = 40 ∧
  date = 3 * apple ∧
  banana = cantaloupe - 2

theorem banana_cantaloupe_cost 
  (apple banana cantaloupe date : ℝ)
  (h : cost_problem apple banana cantaloupe date) :
  banana + cantaloupe = 20 := by
sorry

end banana_cantaloupe_cost_l4012_401214


namespace real_part_of_sum_l4012_401211

theorem real_part_of_sum (z₁ z₂ : ℂ) (h₁ : z₁ = 4 + 19 * Complex.I) (h₂ : z₂ = 6 + 9 * Complex.I) :
  (z₁ + z₂).re = 10 := by
  sorry

end real_part_of_sum_l4012_401211


namespace instantaneous_velocity_one_l4012_401200

-- Define the motion equation
def s (t : ℝ) : ℝ := 3 * t^2 - 2

-- Define the instantaneous velocity (derivative of s with respect to t)
def v (t : ℝ) : ℝ := 6 * t

-- Theorem: The time at which the instantaneous velocity is 1 is 1/6
theorem instantaneous_velocity_one (t : ℝ) : v t = 1 ↔ t = 1/6 := by
  sorry

end instantaneous_velocity_one_l4012_401200


namespace problem_solution_l4012_401219

/-- The function f(x) defined in the problem -/
def f (c : ℝ) (x : ℝ) : ℝ := c * x^4 + (c^2 - 3) * x^2 + 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (c : ℝ) (x : ℝ) : ℝ := 4 * c * x^3 + 2 * (c^2 - 3) * x

theorem problem_solution :
  ∃! c : ℝ,
    (∀ x : ℝ, x < -1 → (f_derivative c x < 0)) ∧
    (∀ x : ℝ, -1 < x ∧ x < 0 → (f_derivative c x > 0)) :=
by
  sorry

end problem_solution_l4012_401219


namespace f_one_equals_four_l4012_401248

/-- The function f(x) = x^2 + ax - 3a - 9 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 3*a - 9

/-- The theorem stating that f(1) = 4 given the conditions -/
theorem f_one_equals_four (a : ℝ) (h : ∀ x : ℝ, f a x ≥ 0) : f a 1 = 4 := by
  sorry

end f_one_equals_four_l4012_401248


namespace marions_bike_cost_l4012_401286

theorem marions_bike_cost (marion_cost stephanie_cost total : ℕ) : 
  stephanie_cost = 2 * marion_cost →
  total = marion_cost + stephanie_cost →
  total = 1068 →
  marion_cost = 356 := by
sorry

end marions_bike_cost_l4012_401286


namespace root_product_sum_l4012_401208

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧ 
  (∀ x, Real.sqrt 2020 * x^3 - 4040 * x^2 + 4 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₂ * (x₁ + x₃) = 2 := by
sorry

end root_product_sum_l4012_401208


namespace quadratic_roots_product_l4012_401256

theorem quadratic_roots_product (b c : ℝ) :
  (∀ x, x^2 + b*x + c = 0 → ∃ y, y^2 + b*y + c = 0 ∧ x * y = 20) →
  c = 20 := by
sorry

end quadratic_roots_product_l4012_401256


namespace general_trigonometric_equation_l4012_401280

theorem general_trigonometric_equation (θ : Real) : 
  Real.sin θ ^ 2 + Real.cos (θ + Real.pi / 6) ^ 2 + Real.sin θ * Real.cos (θ + Real.pi / 6) = 3/4 := by
  sorry

end general_trigonometric_equation_l4012_401280


namespace four_xy_even_l4012_401260

theorem four_xy_even (x y : ℕ) (hx : Even x) (hy : Even y) (hxpos : 0 < x) (hypos : 0 < y) : 
  Even (4 * x * y) := by
  sorry

end four_xy_even_l4012_401260


namespace man_age_twice_son_age_l4012_401295

/-- Proves that the number of years until a man's age is twice his son's age is 2,
    given that the man is currently 24 years older than his son and the son is currently 22 years old. -/
theorem man_age_twice_son_age (son_age : ℕ) (man_age : ℕ) (years : ℕ) : 
  son_age = 22 →
  man_age = son_age + 24 →
  man_age + years = 2 * (son_age + years) →
  years = 2 := by
  sorry

end man_age_twice_son_age_l4012_401295


namespace tangent_circle_equation_l4012_401257

/-- A circle tangent to the y-axis with center on the line x - 3y = 0 and passing through (6, 1) -/
structure TangentCircle where
  -- Center of the circle
  center : ℝ × ℝ
  -- Radius of the circle
  radius : ℝ
  -- The circle is tangent to the y-axis
  tangent_to_y_axis : center.1 = radius
  -- The center is on the line x - 3y = 0
  center_on_line : center.1 = 3 * center.2
  -- The circle passes through (6, 1)
  passes_through_point : (center.1 - 6)^2 + (center.2 - 1)^2 = radius^2

/-- The equation of the circle is either (x-3)² + (y-1)² = 9 or (x-111)² + (y-37)² = 111² -/
theorem tangent_circle_equation (c : TangentCircle) :
  (∀ x y, (x - 3)^2 + (y - 1)^2 = 9) ∨
  (∀ x y, (x - 111)^2 + (y - 37)^2 = 111^2) :=
sorry

end tangent_circle_equation_l4012_401257


namespace min_value_theorem_l4012_401226

theorem min_value_theorem (a : ℝ) (h : a > 3) :
  a + 4 / (a - 3) ≥ 7 ∧ (a + 4 / (a - 3) = 7 ↔ a = 5) := by sorry

end min_value_theorem_l4012_401226


namespace right_triangle_least_side_l4012_401292

theorem right_triangle_least_side (a b c : ℝ) : 
  a = 8 → b = 15 → c^2 = a^2 + b^2 → min a (min b c) = 8 := by
  sorry

end right_triangle_least_side_l4012_401292


namespace zongzi_theorem_l4012_401291

/-- Represents the prices and quantities of zongzi --/
structure ZongziData where
  honey_price : ℝ
  meat_price : ℝ
  honey_quantity : ℕ
  meat_quantity : ℕ
  meat_sold_before : ℕ

/-- Represents the selling prices and profit --/
structure SaleData where
  honey_sell_price : ℝ
  meat_sell_price : ℝ
  meat_price_increase : ℝ
  meat_price_discount : ℝ
  total_profit : ℝ

/-- Main theorem stating the properties of zongzi prices and quantities --/
theorem zongzi_theorem (data : ZongziData) (sale : SaleData) : 
  data.meat_price = data.honey_price + 2.5 ∧ 
  300 / data.meat_price = 2 * (100 / data.honey_price) ∧
  data.honey_quantity = 100 ∧
  data.meat_quantity = 200 ∧
  sale.honey_sell_price = 6 ∧
  sale.meat_sell_price = 10 ∧
  sale.meat_price_increase = 1.1 ∧
  sale.meat_price_discount = 0.9 ∧
  sale.total_profit = 570 →
  data.honey_price = 5 ∧
  data.meat_price = 7.5 ∧
  data.meat_sold_before = 85 := by
  sorry

#check zongzi_theorem

end zongzi_theorem_l4012_401291


namespace cosine_value_implies_expression_value_l4012_401201

theorem cosine_value_implies_expression_value (x : ℝ) 
  (h1 : x ∈ Set.Ioo (-π/2) 0) 
  (h2 : Real.cos (π/2 + x) = 4/5) : 
  (Real.sin (2*x) - 2 * (Real.sin x)^2) / (1 + Real.tan x) = -168/25 := by
  sorry

end cosine_value_implies_expression_value_l4012_401201


namespace smallest_in_S_l4012_401231

def S : Set ℝ := {1, -2, -1.7, 0, Real.pi}

theorem smallest_in_S : ∀ x ∈ S, -2 ≤ x := by sorry

end smallest_in_S_l4012_401231


namespace min_m_plus_n_l4012_401237

/-- The set T of real numbers satisfying the given condition -/
def T : Set ℝ := Set.Iic 1

/-- The theorem stating the minimum value of m + n -/
theorem min_m_plus_n (m n : ℝ) (h_m : m > 1) (h_n : n > 1)
  (h_exists : ∃ x₀ : ℝ, ∀ x t : ℝ, t ∈ T → |x - 1| - |x - 2| ≥ t)
  (h_log : ∀ t ∈ T, Real.log m / Real.log 3 * Real.log n / Real.log 3 ≥ t) :
  m + n ≥ 6 ∧ ∃ m₀ n₀ : ℝ, m₀ > 1 ∧ n₀ > 1 ∧ m₀ + n₀ = 6 ∧
    (∀ t ∈ T, Real.log m₀ / Real.log 3 * Real.log n₀ / Real.log 3 ≥ t) :=
sorry

end min_m_plus_n_l4012_401237


namespace expression_evaluation_l4012_401288

theorem expression_evaluation : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end expression_evaluation_l4012_401288


namespace milk_water_ratio_change_l4012_401268

/-- Proves that adding 60 litres of water to a 60-litre mixture with initial milk to water ratio of 2:1 results in a new ratio of 1:2 -/
theorem milk_water_ratio_change (initial_volume : ℚ) (initial_milk_ratio : ℚ) (initial_water_ratio : ℚ) (added_water : ℚ) :
  initial_volume = 60 →
  initial_milk_ratio = 2 →
  initial_water_ratio = 1 →
  added_water = 60 →
  let initial_milk := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let new_water := initial_water + added_water
  let new_milk_ratio := initial_milk / new_water
  let new_water_ratio := new_water / new_water
  new_milk_ratio = 1 ∧ new_water_ratio = 2 :=
by sorry

end milk_water_ratio_change_l4012_401268


namespace circle_intersection_range_l4012_401297

noncomputable section

-- Define the circle M
def circle_M (a : ℝ) (x y : ℝ) : Prop :=
  (x - Real.sqrt a)^2 + (y - Real.sqrt a)^2 = 9

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (2, 0)

-- Define the condition for point P
def exists_P (a : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, circle_M a P.1 P.2 ∧
    (P.1 - point_A.1) * (point_B.1 - point_A.1) +
    (P.2 - point_A.2) * (point_B.2 - point_A.2) = 0

-- State the theorem
theorem circle_intersection_range :
  ∀ a : ℝ, exists_P a ↔ 1/2 ≤ a ∧ a ≤ 25/2 :=
sorry

end

end circle_intersection_range_l4012_401297


namespace amount_in_paise_l4012_401289

theorem amount_in_paise : 
  let a : ℝ := 190
  let percentage : ℝ := 0.5
  let amount_in_rupees : ℝ := percentage / 100 * a
  let paise_per_rupee : ℕ := 100
  ⌊amount_in_rupees * paise_per_rupee⌋ = 95 := by sorry

end amount_in_paise_l4012_401289


namespace unit_vectors_sum_squares_lower_bound_l4012_401236

theorem unit_vectors_sum_squares_lower_bound 
  (p q r : EuclideanSpace ℝ (Fin 3)) 
  (hp : ‖p‖ = 1) (hq : ‖q‖ = 1) (hr : ‖r‖ = 1) : 
  ‖p + q‖^2 + ‖p + r‖^2 + ‖q + r‖^2 ≥ 0 :=
by sorry

end unit_vectors_sum_squares_lower_bound_l4012_401236


namespace power_of_power_l4012_401290

theorem power_of_power (x : ℝ) : (x^2)^3 = x^6 := by
  sorry

end power_of_power_l4012_401290


namespace distinct_colorings_count_l4012_401266

/-- Represents the symmetries of a square -/
inductive SquareSymmetry
| Rotation0 | Rotation90 | Rotation180 | Rotation270
| ReflectionSide1 | ReflectionSide2
| ReflectionDiag1 | ReflectionDiag2

/-- Represents a coloring of the square's disks -/
structure SquareColoring :=
(blue1 : Fin 4)
(blue2 : Fin 4)
(red : Fin 4)
(green : Fin 4)

/-- The group of symmetries of a square -/
def squareSymmetryGroup : List SquareSymmetry :=
[SquareSymmetry.Rotation0, SquareSymmetry.Rotation90, SquareSymmetry.Rotation180, SquareSymmetry.Rotation270,
 SquareSymmetry.ReflectionSide1, SquareSymmetry.ReflectionSide2,
 SquareSymmetry.ReflectionDiag1, SquareSymmetry.ReflectionDiag2]

/-- Checks if a coloring is valid (2 blue, 1 red, 1 green) -/
def isValidColoring (c : SquareColoring) : Bool :=
  c.blue1 ≠ c.blue2 ∧ c.blue1 ≠ c.red ∧ c.blue1 ≠ c.green ∧
  c.blue2 ≠ c.red ∧ c.blue2 ≠ c.green ∧ c.red ≠ c.green

/-- Checks if a coloring is fixed by a given symmetry -/
def isFixedBy (c : SquareColoring) (s : SquareSymmetry) : Bool := sorry

/-- Counts the number of colorings fixed by each symmetry -/
def countFixedColorings (s : SquareSymmetry) : Nat := sorry

/-- The main theorem: there are 3 distinct colorings under symmetry -/
theorem distinct_colorings_count :
  (List.sum (List.map countFixedColorings squareSymmetryGroup)) / squareSymmetryGroup.length = 3 := sorry

end distinct_colorings_count_l4012_401266
