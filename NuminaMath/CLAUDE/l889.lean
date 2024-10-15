import Mathlib

namespace NUMINAMATH_CALUDE_flour_scoops_l889_88908

/-- Given a bag of flour, the amount needed for a recipe, and the size of a measuring cup,
    calculate the number of scoops to remove from the bag. -/
def scoop_count (bag_size : ℚ) (recipe_amount : ℚ) (measure_size : ℚ) : ℚ :=
  (bag_size - recipe_amount) / measure_size

theorem flour_scoops :
  let bag_size : ℚ := 8
  let recipe_amount : ℚ := 6
  let measure_size : ℚ := 1/4
  scoop_count bag_size recipe_amount measure_size = 8 := by sorry

end NUMINAMATH_CALUDE_flour_scoops_l889_88908


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l889_88972

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < -1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l889_88972


namespace NUMINAMATH_CALUDE_f_properties_f_inv_property_l889_88932

/-- A function f(x) that is directly proportional to x-3 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x - 3 * k

/-- The theorem stating the properties of f -/
theorem f_properties (k : ℝ) :
  f k 4 = 3 →
  (∀ x, f k x = 3 * x - 9) ∧
  (∃ x, f k x = -12 ∧ x = -1) := by
  sorry

/-- The inverse function of f -/
noncomputable def f_inv (k : ℝ) (y : ℝ) : ℝ := (y + 3 * k) / k

/-- Theorem stating that f_inv(-12) = -1 when f(4) = 3 -/
theorem f_inv_property (k : ℝ) :
  f k 4 = 3 →
  f_inv k (-12) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_properties_f_inv_property_l889_88932


namespace NUMINAMATH_CALUDE_bill_more_sticks_than_ted_l889_88992

/-- Represents the number of objects thrown by a person -/
structure ThrowCount where
  sticks : ℕ
  rocks : ℕ

/-- Calculates the total number of objects thrown -/
def ThrowCount.total (t : ThrowCount) : ℕ := t.sticks + t.rocks

theorem bill_more_sticks_than_ted (bill : ThrowCount) (ted : ThrowCount) : 
  bill.total = 21 → 
  ted.rocks = 2 * bill.rocks → 
  ted.sticks = 10 → 
  ted.rocks = 10 → 
  bill.sticks - ted.sticks = 6 := by
sorry

end NUMINAMATH_CALUDE_bill_more_sticks_than_ted_l889_88992


namespace NUMINAMATH_CALUDE_complex_equation_solution_l889_88986

theorem complex_equation_solution (z : ℂ) (h : (3 + z) * Complex.I = 1) : z = -3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l889_88986


namespace NUMINAMATH_CALUDE_add_1723_minutes_to_midnight_l889_88956

-- Define a custom datatype for date and time
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

-- Define the starting date and time
def startDateTime : DateTime :=
  { year := 2023, month := 5, day := 5, hour := 0, minute := 0 }

-- Define the number of minutes to add
def minutesToAdd : Nat := 1723

-- Theorem to prove
theorem add_1723_minutes_to_midnight :
  addMinutes startDateTime minutesToAdd =
    { year := 2023, month := 5, day := 6, hour := 4, minute := 43 } :=
  sorry

end NUMINAMATH_CALUDE_add_1723_minutes_to_midnight_l889_88956


namespace NUMINAMATH_CALUDE_symmetric_points_line_equation_l889_88942

/-- Given two points P and Q that are symmetric about a line l, 
    prove that the equation of line l is x - y + 1 = 0 -/
theorem symmetric_points_line_equation 
  (a b : ℝ) 
  (h : a ≠ b - 1) :
  let P : ℝ × ℝ := (a, b)
  let Q : ℝ × ℝ := (b - 1, a + 1)
  ∀ (l : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ x - y + 1 = 0) →
    (∀ (R : ℝ × ℝ), R ∈ l → (dist P R = dist Q R)) →
    true :=
by sorry


end NUMINAMATH_CALUDE_symmetric_points_line_equation_l889_88942


namespace NUMINAMATH_CALUDE_tangent_implies_positive_derivative_l889_88961

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that the tangent line at (2,3) passes through (-1,2)
def tangent_condition (f : ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), k * (-1 - 2) + 3 = 2 ∧ HasDerivAt f k 2

-- State the theorem
theorem tangent_implies_positive_derivative (f : ℝ → ℝ) 
  (h : tangent_condition f) : 
  ∃ (d : ℝ), HasDerivAt f d 2 ∧ d > 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_implies_positive_derivative_l889_88961


namespace NUMINAMATH_CALUDE_hammingDistance_bounds_hammingDistance_triangle_inequality_l889_88955

/-- A byte is a list of booleans representing binary digits. -/
def Byte := List Bool

/-- The Hamming distance between two bytes is the number of positions at which they differ. -/
def hammingDistance (u v : Byte) : Nat :=
  (u.zip v).filter (fun (a, b) => a ≠ b) |>.length

/-- Theorem stating that the Hamming distance between two bytes is bounded by 0 and the length of the bytes. -/
theorem hammingDistance_bounds (u v : Byte) (h : u.length = v.length) :
    0 ≤ hammingDistance u v ∧ hammingDistance u v ≤ u.length := by
  sorry

/-- Theorem stating the triangle inequality for Hamming distance. -/
theorem hammingDistance_triangle_inequality (u v w : Byte) 
    (hu : u.length = v.length) (hv : v.length = w.length) :
    hammingDistance u v ≤ hammingDistance w u + hammingDistance w v := by
  sorry

end NUMINAMATH_CALUDE_hammingDistance_bounds_hammingDistance_triangle_inequality_l889_88955


namespace NUMINAMATH_CALUDE_total_cookies_l889_88991

/-- Given 26 bags of cookies with 2 cookies in each bag, prove that the total number of cookies is 52. -/
theorem total_cookies (num_bags : ℕ) (cookies_per_bag : ℕ) 
  (h1 : num_bags = 26) 
  (h2 : cookies_per_bag = 2) : 
  num_bags * cookies_per_bag = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l889_88991


namespace NUMINAMATH_CALUDE_kathleen_store_visits_l889_88943

/-- The number of bottle caps Kathleen buys each time she goes to the store -/
def bottle_caps_per_visit : ℕ := 5

/-- The total number of bottle caps Kathleen bought last month -/
def total_bottle_caps : ℕ := 25

/-- The number of times Kathleen went to the store last month -/
def store_visits : ℕ := total_bottle_caps / bottle_caps_per_visit

theorem kathleen_store_visits : store_visits = 5 := by
  sorry

end NUMINAMATH_CALUDE_kathleen_store_visits_l889_88943


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l889_88994

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3 + a 10 = 3) →
  (a 3 * a 10 = -5) →
  a 5 + a 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l889_88994


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l889_88960

def prob_boy_or_girl : ℚ := 1 / 2

def family_size : ℕ := 4

theorem prob_at_least_one_boy_one_girl :
  1 - (prob_boy_or_girl ^ family_size + prob_boy_or_girl ^ family_size) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l889_88960


namespace NUMINAMATH_CALUDE_exists_same_color_rectangle_l889_88976

/-- A color type representing red, black, and blue -/
inductive Color
  | Red
  | Black
  | Blue

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A function that assigns a color to each point in the plane -/
def colorFunction : Point → Color := sorry

/-- A type representing a rectangle in the plane -/
structure Rectangle where
  topLeft : Point
  topRight : Point
  bottomLeft : Point
  bottomRight : Point

/-- A predicate that checks if all vertices of a rectangle have the same color -/
def sameColorVertices (rect : Rectangle) : Prop :=
  colorFunction rect.topLeft = colorFunction rect.topRight ∧
  colorFunction rect.topLeft = colorFunction rect.bottomLeft ∧
  colorFunction rect.topLeft = colorFunction rect.bottomRight

/-- Theorem stating that there exists a rectangle with vertices of the same color -/
theorem exists_same_color_rectangle : ∃ (rect : Rectangle), sameColorVertices rect := by
  sorry


end NUMINAMATH_CALUDE_exists_same_color_rectangle_l889_88976


namespace NUMINAMATH_CALUDE_choir_size_l889_88989

/-- Given an orchestra with female and male students, and a choir with three times
    the number of people in the orchestra, calculate the number of people in the choir. -/
theorem choir_size (female_students male_students : ℕ) 
  (h1 : female_students = 18) 
  (h2 : male_students = 25) : 
  3 * (female_students + male_students) = 129 := by
  sorry

end NUMINAMATH_CALUDE_choir_size_l889_88989


namespace NUMINAMATH_CALUDE_parabola_a_values_l889_88946

/-- The parabola equation y = ax^2 -/
def parabola (a : ℝ) (x y : ℝ) : Prop := y = a * x^2

/-- The point M with coordinates (2, 1) -/
def point_M : ℝ × ℝ := (2, 1)

/-- The distance from point M to the directrix is 2 -/
def distance_to_directrix : ℝ := 2

/-- The possible values of a -/
def possible_a_values : Set ℝ := {1/4, -1/12}

/-- Theorem stating the possible values of a for the given conditions -/
theorem parabola_a_values :
  ∀ a : ℝ,
  (∃ y : ℝ, parabola a (point_M.1) y) →
  (∃ d : ℝ, d = distance_to_directrix ∧ 
    ((a > 0 ∧ d = point_M.2 + 1/(4*a)) ∨
     (a < 0 ∧ d = -1/(4*a) - point_M.2))) →
  a ∈ possible_a_values :=
sorry

end NUMINAMATH_CALUDE_parabola_a_values_l889_88946


namespace NUMINAMATH_CALUDE_heron_height_calculation_l889_88919

theorem heron_height_calculation (a b c : ℝ) (h : ℝ) :
  a = 20 ∧ b = 99 ∧ c = 101 →
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  h = 2 * area / b →
  h = 20 := by
  sorry

end NUMINAMATH_CALUDE_heron_height_calculation_l889_88919


namespace NUMINAMATH_CALUDE_alcohol_percentage_first_vessel_l889_88957

theorem alcohol_percentage_first_vessel
  (vessel1_capacity : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_alcohol_percentage : ℝ)
  (total_liquid : ℝ)
  (final_vessel_capacity : ℝ)
  (final_alcohol_percentage : ℝ)
  (h1 : vessel1_capacity = 3)
  (h2 : vessel2_capacity = 5)
  (h3 : vessel2_alcohol_percentage = 40)
  (h4 : total_liquid = 8)
  (h5 : final_vessel_capacity = 10)
  (h6 : final_alcohol_percentage = 27.5) :
  ∃ (vessel1_alcohol_percentage : ℝ),
    vessel1_alcohol_percentage = 25 ∧
    (vessel1_alcohol_percentage / 100) * vessel1_capacity +
    (vessel2_alcohol_percentage / 100) * vessel2_capacity =
    (final_alcohol_percentage / 100) * final_vessel_capacity :=
by sorry

end NUMINAMATH_CALUDE_alcohol_percentage_first_vessel_l889_88957


namespace NUMINAMATH_CALUDE_savings_proof_l889_88999

def original_savings (furniture_fraction : ℚ) (tv_cost : ℕ) : ℕ :=
  4 * tv_cost

theorem savings_proof (furniture_fraction : ℚ) (tv_cost : ℕ) 
  (h1 : furniture_fraction = 3/4) 
  (h2 : tv_cost = 210) : 
  original_savings furniture_fraction tv_cost = 840 := by
  sorry

end NUMINAMATH_CALUDE_savings_proof_l889_88999


namespace NUMINAMATH_CALUDE_initial_amount_proof_l889_88969

/-- Proves that Rs 100 at 5% interest for 48 years produces the same interest as Rs 600 at 10% interest for 4 years -/
theorem initial_amount_proof (amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time1 : ℝ) (time2 : ℝ) : 
  amount = 100 ∧ rate1 = 0.05 ∧ rate2 = 0.10 ∧ time1 = 48 ∧ time2 = 4 →
  amount * rate1 * time1 = 600 * rate2 * time2 :=
by
  sorry

#check initial_amount_proof

end NUMINAMATH_CALUDE_initial_amount_proof_l889_88969


namespace NUMINAMATH_CALUDE_chicken_problem_l889_88900

/-- The problem of calculating the difference in number of chickens bought by John and Ray -/
theorem chicken_problem (chicken_cost : ℕ) (john_extra : ℕ) (ray_less : ℕ) (ray_chickens : ℕ) :
  chicken_cost = 3 →
  john_extra = 15 →
  ray_less = 18 →
  ray_chickens = 10 →
  (john_extra + ray_less + ray_chickens * chicken_cost) / chicken_cost - ray_chickens = 11 :=
by sorry

end NUMINAMATH_CALUDE_chicken_problem_l889_88900


namespace NUMINAMATH_CALUDE_conditions_sufficient_not_necessary_l889_88903

theorem conditions_sufficient_not_necessary (m : ℝ) (h : m > 0) :
  (∀ x y a : ℝ, |x - a| < m ∧ |y - a| < m → |x - y| < 2*m) ∧
  (∃ x y a : ℝ, |x - y| < 2*m ∧ (|x - a| ≥ m ∨ |y - a| ≥ m)) :=
by sorry

end NUMINAMATH_CALUDE_conditions_sufficient_not_necessary_l889_88903


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l889_88929

-- Define the vectors
def a : Fin 2 → ℝ := ![(-1 : ℝ), 3]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 2]

-- Define the dot product
def dot_product (u v : Fin 2 → ℝ) : ℝ :=
  (u 0) * (v 0) + (u 1) * (v 1)

-- Define perpendicularity
def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  dot_product u v = 0

-- The main theorem
theorem perpendicular_vectors_x_value :
  ∃ x : ℝ, perpendicular (a + b x) a ∧ x = 16 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l889_88929


namespace NUMINAMATH_CALUDE_calculate_expression_l889_88930

theorem calculate_expression : -5^2 - (-3)^3 * (2/9) - 9 * |-(2/3)| = -25 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l889_88930


namespace NUMINAMATH_CALUDE_part1_solution_set_part2_a_range_l889_88982

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x - |x - 3|

-- Part 1
theorem part1_solution_set :
  {x : ℝ | f 1 x + |2*x - 5| ≥ 6} = {x : ℝ | x ≥ 4 ∨ x ≤ 0} := by sorry

-- Part 2
theorem part2_a_range :
  ∀ a : ℝ, (∀ y : ℝ, -1 ≤ y ∧ y ≤ 2 → ∃ x : ℝ, g a x = y) →
  (a ≤ 1 ∨ a ≥ 5) := by sorry

end NUMINAMATH_CALUDE_part1_solution_set_part2_a_range_l889_88982


namespace NUMINAMATH_CALUDE_fifth_term_is_x_l889_88934

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Define our specific sequence
def our_sequence (x y : ℝ) : ℕ → ℝ
| 0 => x + 2*y
| 1 => x - 2*y
| 2 => x + y
| 3 => x - y
| n + 4 => our_sequence x y 3 + (n + 1) * (our_sequence x y 1 - our_sequence x y 0)

theorem fifth_term_is_x (x y : ℝ) :
  is_arithmetic_sequence (our_sequence x y) →
  our_sequence x y 4 = x :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_x_l889_88934


namespace NUMINAMATH_CALUDE_tissues_left_is_1060_l889_88978

/-- The number of tissues Tucker has left after all actions. -/
def tissues_left : ℕ :=
  let brand_a_per_box := 160
  let brand_b_per_box := 180
  let brand_c_per_box := 200
  let brand_a_boxes := 4
  let brand_b_boxes := 6
  let brand_c_boxes := 2
  let brand_a_used := 250
  let brand_b_used := 410
  let brand_c_used := 150
  let brand_b_given := 2
  let brand_c_received := 110

  let brand_a_left := brand_a_per_box * brand_a_boxes - brand_a_used
  let brand_b_left := brand_b_per_box * brand_b_boxes - brand_b_used - brand_b_per_box * brand_b_given
  let brand_c_left := brand_c_per_box * brand_c_boxes - brand_c_used + brand_c_received

  brand_a_left + brand_b_left + brand_c_left

theorem tissues_left_is_1060 : tissues_left = 1060 := by
  sorry

end NUMINAMATH_CALUDE_tissues_left_is_1060_l889_88978


namespace NUMINAMATH_CALUDE_solution_to_equation_l889_88996

theorem solution_to_equation : ∃ y : ℝ, (7 - y = 10) ∧ (y = -3) := by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l889_88996


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l889_88988

theorem smallest_number_with_given_remainders : ∃ (a : ℕ), (
  (a % 3 = 1) ∧
  (a % 6 = 3) ∧
  (a % 7 = 4) ∧
  (∀ b : ℕ, b < a → (b % 3 ≠ 1 ∨ b % 6 ≠ 3 ∨ b % 7 ≠ 4))
) ∧ a = 39 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l889_88988


namespace NUMINAMATH_CALUDE_no_infinite_sequence_exists_l889_88959

theorem no_infinite_sequence_exists : ¬ ∃ (a : ℕ → ℕ+), ∀ n : ℕ, 
  (a (n + 2) : ℝ) = (a (n + 1) : ℝ) + Real.sqrt ((a (n + 1) : ℝ) + (a n : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_exists_l889_88959


namespace NUMINAMATH_CALUDE_property_P_theorems_seq_012_has_property_P_l889_88921

/-- Definition of a sequence with property P -/
def has_property_P (a : ℕ → ℝ) (k : ℕ) : Prop :=
  k ≥ 3 ∧
  (∀ i j, 1 ≤ i → i ≤ j → j ≤ k → (∃ n ≤ k, a n = a j + a i ∨ a n = a j - a i)) ∧
  (∀ i, 1 ≤ i → i < k → a i < a (i + 1)) ∧
  0 ≤ a 1

theorem property_P_theorems (a : ℕ → ℝ) (k : ℕ) (h : has_property_P a k) :
  (∀ i ≤ k, a k - a i ∈ Set.range (fun n => a n)) ∧
  (k ≥ 5 → ∃ d : ℝ, ∀ i < k, a (i + 1) - a i = d) :=
by sorry

/-- The sequence 0, 1, 2 has property P -/
theorem seq_012_has_property_P :
  has_property_P (fun n => if n = 1 then 0 else if n = 2 then 1 else 2) 3 :=
by sorry

end NUMINAMATH_CALUDE_property_P_theorems_seq_012_has_property_P_l889_88921


namespace NUMINAMATH_CALUDE_only_common_term_is_one_l889_88917

def x : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℕ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

theorem only_common_term_is_one : ∀ n : ℕ, x n = y n ↔ n = 0 := by sorry

end NUMINAMATH_CALUDE_only_common_term_is_one_l889_88917


namespace NUMINAMATH_CALUDE_simplified_ratio_l889_88953

def initial_money : ℕ := 91
def spent_money : ℕ := 21

def money_left : ℕ := initial_money - spent_money

def ratio_numerator : ℕ := money_left
def ratio_denominator : ℕ := spent_money

theorem simplified_ratio :
  (ratio_numerator / (Nat.gcd ratio_numerator ratio_denominator)) = 10 ∧
  (ratio_denominator / (Nat.gcd ratio_numerator ratio_denominator)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_simplified_ratio_l889_88953


namespace NUMINAMATH_CALUDE_vasyas_numbers_l889_88927

theorem vasyas_numbers (x y : ℝ) : x + y = x * y ∧ x + y = x / y ∧ x * y = x / y → x = (1 : ℝ) / 2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l889_88927


namespace NUMINAMATH_CALUDE_spinsters_to_cats_ratio_l889_88933

theorem spinsters_to_cats_ratio 
  (spinsters : ℕ) 
  (cats : ℕ) 
  (x : ℚ)
  (ratio_condition : spinsters / cats = x / 9)
  (difference_condition : cats = spinsters + 63)
  (spinsters_count : spinsters = 18) :
  spinsters / cats = 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_spinsters_to_cats_ratio_l889_88933


namespace NUMINAMATH_CALUDE_square_sum_equality_l889_88920

theorem square_sum_equality (y : ℝ) : 
  (y - 2)^2 + 2*(y - 2)*(5 + y) + (5 + y)^2 = (2*y + 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l889_88920


namespace NUMINAMATH_CALUDE_corner_sum_6x12_board_l889_88905

/-- Represents a rectangular board filled with consecutive numbers -/
structure NumberBoard where
  rows : Nat
  cols : Nat
  total_numbers : Nat

/-- Returns the number at a given position on the board -/
def NumberBoard.number_at (board : NumberBoard) (row : Nat) (col : Nat) : Nat :=
  (row - 1) * board.cols + col

/-- Theorem stating that the sum of corner numbers on a 6x12 board is 146 -/
theorem corner_sum_6x12_board :
  let board : NumberBoard := ⟨6, 12, 72⟩
  (board.number_at 1 1) + (board.number_at 1 12) +
  (board.number_at 6 1) + (board.number_at 6 12) = 146 := by
  sorry

#check corner_sum_6x12_board

end NUMINAMATH_CALUDE_corner_sum_6x12_board_l889_88905


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_of_roots_l889_88902

theorem sum_of_fourth_powers_of_roots (p q r : ℝ) : 
  (p^3 - 2*p^2 + 3*p - 4 = 0) → 
  (q^3 - 2*q^2 + 3*q - 4 = 0) → 
  (r^3 - 2*r^2 + 3*r - 4 = 0) → 
  p^4 + q^4 + r^4 = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_of_roots_l889_88902


namespace NUMINAMATH_CALUDE_ratio_of_angles_l889_88980

-- Define the circle and triangle
def Circle : Type := Unit
def Point : Type := Unit
def Triangle : Type := Unit

-- Define the center of the circle
def O : Point := sorry

-- Define the vertices of the triangle
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry

-- Define point E
def E : Point := sorry

-- Define the inscribed triangle
def triangle_ABC : Triangle := sorry

-- Define the arcs
def arc_AB : ℝ := 100
def arc_BC : ℝ := 80

-- Define the perpendicular condition
def OE_perp_AC : Prop := sorry

-- Define the angles
def angle_OBE : ℝ := sorry
def angle_BAC : ℝ := sorry

-- State the theorem
theorem ratio_of_angles (circle : Circle) (triangle_ABC : Triangle) 
  (h1 : arc_AB = 100)
  (h2 : arc_BC = 80)
  (h3 : OE_perp_AC) :
  angle_OBE / angle_BAC = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_ratio_of_angles_l889_88980


namespace NUMINAMATH_CALUDE_gcf_of_60_90_150_l889_88944

theorem gcf_of_60_90_150 : Nat.gcd 60 (Nat.gcd 90 150) = 30 := by sorry

end NUMINAMATH_CALUDE_gcf_of_60_90_150_l889_88944


namespace NUMINAMATH_CALUDE_christinas_driving_time_l889_88909

theorem christinas_driving_time 
  (total_distance : ℝ) 
  (christina_speed : ℝ) 
  (friend_speed : ℝ) 
  (friend_time : ℝ) 
  (h1 : total_distance = 210)
  (h2 : christina_speed = 30)
  (h3 : friend_speed = 40)
  (h4 : friend_time = 3) :
  (total_distance - friend_speed * friend_time) / christina_speed * 60 = 180 :=
by sorry

end NUMINAMATH_CALUDE_christinas_driving_time_l889_88909


namespace NUMINAMATH_CALUDE_jerry_firecracker_fraction_l889_88958

/-- Given:
  * Jerry bought 48 firecrackers initially
  * 12 firecrackers were confiscated
  * 1/6 of the remaining firecrackers were defective
  * Jerry set off 15 good firecrackers
Prove that Jerry set off 1/2 of the good firecrackers -/
theorem jerry_firecracker_fraction :
  let initial_firecrackers : ℕ := 48
  let confiscated_firecrackers : ℕ := 12
  let defective_fraction : ℚ := 1/6
  let set_off_firecrackers : ℕ := 15
  let remaining_firecrackers := initial_firecrackers - confiscated_firecrackers
  let good_firecrackers := remaining_firecrackers - (defective_fraction * remaining_firecrackers).num
  (set_off_firecrackers : ℚ) / good_firecrackers = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_jerry_firecracker_fraction_l889_88958


namespace NUMINAMATH_CALUDE_angle_measure_problem_l889_88918

/-- Given two supplementary angles C and D, where C is 5 times D, prove that the measure of angle C is 150°. -/
theorem angle_measure_problem (C D : ℝ) : 
  C + D = 180 →  -- C and D are supplementary
  C = 5 * D →    -- C is 5 times D
  C = 150 :=     -- The measure of angle C is 150°
by
  sorry

end NUMINAMATH_CALUDE_angle_measure_problem_l889_88918


namespace NUMINAMATH_CALUDE_integer_solutions_equation_l889_88985

theorem integer_solutions_equation : 
  {(x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y} = 
  {(-1, -1), (-1, 0), (0, -1), (0, 0), (5, 2), (-6, 2)} := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_equation_l889_88985


namespace NUMINAMATH_CALUDE_equation_solution_l889_88945

theorem equation_solution :
  ∃ x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ∧ x = -5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l889_88945


namespace NUMINAMATH_CALUDE_sum_of_integers_l889_88947

theorem sum_of_integers (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e → 
  (8 - a) * (8 - b) * (8 - c) * (8 - d) * (8 - e) = 120 →
  (Prime a ∨ Prime b ∨ Prime c ∨ Prime d ∨ Prime e) →
  a + b + c + d + e = 34 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l889_88947


namespace NUMINAMATH_CALUDE_angle_between_3_and_7_l889_88967

/-- Represents a clock with equally spaced rays -/
structure Clock :=
  (num_rays : ℕ)
  (ray_spacing : ℝ)
  (h_positive_rays : 0 < num_rays)
  (h_spacing : ray_spacing = 360 / num_rays)

/-- Calculates the angle between two hour positions on a clock -/
def angle_between_hours (clock : Clock) (hour1 hour2 : ℕ) : ℝ :=
  let diff := (hour2 - hour1 + clock.num_rays) % clock.num_rays
  clock.ray_spacing * min diff (clock.num_rays - diff)

/-- Theorem: The smaller angle between 3 o'clock and 7 o'clock on a 12-hour clock is 120 degrees -/
theorem angle_between_3_and_7 :
  ∀ (c : Clock), c.num_rays = 12 → angle_between_hours c 3 7 = 120 :=
by sorry

end NUMINAMATH_CALUDE_angle_between_3_and_7_l889_88967


namespace NUMINAMATH_CALUDE_total_cartridge_cost_l889_88935

def black_and_white_cost : ℕ := 27
def color_cost : ℕ := 32
def num_black_and_white : ℕ := 1
def num_color : ℕ := 3

theorem total_cartridge_cost :
  num_black_and_white * black_and_white_cost + num_color * color_cost = 123 :=
by sorry

end NUMINAMATH_CALUDE_total_cartridge_cost_l889_88935


namespace NUMINAMATH_CALUDE_disk_arrangement_area_l889_88928

theorem disk_arrangement_area :
  ∀ (r : ℝ),
  r > 0 →
  r = 2 - Real.sqrt 3 →
  (12 : ℝ) * π * r^2 = π * (84 - 48 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_disk_arrangement_area_l889_88928


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l889_88906

/-- A linear function that does not pass through the third quadrant -/
structure LinearFunctionNotInThirdQuadrant where
  k : ℝ
  b : ℝ
  not_in_third_quadrant : ∀ x y : ℝ, y = k * x + b → ¬(x < 0 ∧ y < 0)

/-- Theorem: For a linear function y = kx + b that does not pass through the third quadrant,
    k is negative and b is non-negative -/
theorem linear_function_not_in_third_quadrant
  (f : LinearFunctionNotInThirdQuadrant) : f.k < 0 ∧ f.b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l889_88906


namespace NUMINAMATH_CALUDE_five_objects_three_categories_l889_88973

/-- The number of ways to distribute n distinguishable objects into k distinct categories -/
def distributionWays (n k : ℕ) : ℕ := k ^ n

/-- Theorem: There are 243 ways to distribute 5 distinguishable objects into 3 distinct categories -/
theorem five_objects_three_categories : distributionWays 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_five_objects_three_categories_l889_88973


namespace NUMINAMATH_CALUDE_probability_three_heads_five_tosses_l889_88998

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def probability_k_heads (n k : ℕ) : ℚ :=
  (n.choose k : ℚ) * (1/2)^k * (1/2)^(n-k)

/-- The probability of getting exactly 3 heads in 5 tosses of a fair coin is 5/16 -/
theorem probability_three_heads_five_tosses :
  probability_k_heads 5 3 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_heads_five_tosses_l889_88998


namespace NUMINAMATH_CALUDE_decryption_theorem_l889_88954

-- Define the encryption functions
def encrypt_a (a : ℤ) : ℤ := a + 1
def encrypt_b (a b : ℤ) : ℤ := 2 * b + a
def encrypt_c (c : ℤ) : ℤ := 3 * c - 4

-- Define the theorem
theorem decryption_theorem (a b c : ℤ) :
  encrypt_a a = 21 ∧ encrypt_b a b = 22 ∧ encrypt_c c = 23 →
  a = 20 ∧ b = 1 ∧ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_decryption_theorem_l889_88954


namespace NUMINAMATH_CALUDE_range_of_a_l889_88916

theorem range_of_a (P Q : Prop) (h_or : P ∨ Q) (h_not_and : ¬(P ∧ Q))
  (h_P : P ↔ ∀ x : ℝ, x^2 - 2*x > a)
  (h_Q : Q ↔ ∃ x : ℝ, x^2 + 2*a*x + 2 = 0) :
  (-2 < a ∧ a < -1) ∨ (a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l889_88916


namespace NUMINAMATH_CALUDE_sum_of_variables_l889_88949

theorem sum_of_variables (a b c : ℝ) 
  (eq1 : a + 2*b + 3*c = 13) 
  (eq2 : 4*a + 3*b + 2*c = 17) : 
  a + b + c = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_variables_l889_88949


namespace NUMINAMATH_CALUDE_total_weight_is_8040_l889_88937

/-- Represents the catering setup for an event -/
structure CateringSetup where
  numTables : Nat
  settingsPerTable : Nat
  backupPercentage : Rat
  forkWeight : Rat
  knifeWeight : Rat
  spoonWeight : Rat
  largePlateWeight : Rat
  smallPlateWeight : Rat
  wineGlassWeight : Rat
  waterGlassWeight : Rat
  tableDecorationWeight : Rat

/-- Calculates the total weight of all items for the catering setup -/
def totalWeight (setup : CateringSetup) : Rat :=
  let totalSettings := setup.numTables * setup.settingsPerTable * (1 + setup.backupPercentage)
  let silverwareWeight := totalSettings * (setup.forkWeight + setup.knifeWeight + setup.spoonWeight)
  let plateWeight := totalSettings * (setup.largePlateWeight + setup.smallPlateWeight)
  let glassWeight := totalSettings * (setup.wineGlassWeight + setup.waterGlassWeight)
  let decorationWeight := setup.numTables * setup.tableDecorationWeight
  silverwareWeight + plateWeight + glassWeight + decorationWeight

/-- Theorem stating that the total weight for the given setup is 8040 ounces -/
theorem total_weight_is_8040 (setup : CateringSetup) 
    (h1 : setup.numTables = 15)
    (h2 : setup.settingsPerTable = 8)
    (h3 : setup.backupPercentage = 1/4)
    (h4 : setup.forkWeight = 7/2)
    (h5 : setup.knifeWeight = 4)
    (h6 : setup.spoonWeight = 9/2)
    (h7 : setup.largePlateWeight = 14)
    (h8 : setup.smallPlateWeight = 10)
    (h9 : setup.wineGlassWeight = 7)
    (h10 : setup.waterGlassWeight = 9)
    (h11 : setup.tableDecorationWeight = 16) :
    totalWeight setup = 8040 := by
  sorry


end NUMINAMATH_CALUDE_total_weight_is_8040_l889_88937


namespace NUMINAMATH_CALUDE_reciprocal_sum_l889_88931

theorem reciprocal_sum : (1 / (1 / 4 + 1 / 6) : ℚ) = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_l889_88931


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l889_88952

theorem rectangular_solid_volume 
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 20)
  (h_front : front_area = 15)
  (h_bottom : bottom_area = 12) :
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    a * c = bottom_area ∧ 
    a * b * c = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l889_88952


namespace NUMINAMATH_CALUDE_dads_strawberry_weight_l889_88926

/-- Given the total weight of strawberries picked by Marco and his dad,
    and the weight of Marco's strawberries, calculate the weight of his dad's strawberries. -/
theorem dads_strawberry_weight 
  (total_weight : ℕ) 
  (marcos_weight : ℕ) 
  (h1 : total_weight = 20)
  (h2 : marcos_weight = 3) : 
  total_weight - marcos_weight = 17 := by
  sorry

#check dads_strawberry_weight

end NUMINAMATH_CALUDE_dads_strawberry_weight_l889_88926


namespace NUMINAMATH_CALUDE_tea_mixture_profit_l889_88963

/-- Proves that the given tea mixture achieves the desired profit -/
theorem tea_mixture_profit (x y : ℝ) : 
  x + y = 100 →
  0.32 * x + 0.40 * y = 34.40 →
  x = 70 ∧ y = 30 ∧ 
  (0.43 * 100 / (0.32 * x + 0.40 * y) - 1) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_profit_l889_88963


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l889_88990

theorem arcsin_equation_solution (x : ℝ) :
  (Real.arcsin x + Real.arcsin (3 * x) = π / 4) →
  (x = Real.sqrt (1 / (9 + 4 * Real.sqrt 2)) ∨
   x = -Real.sqrt (1 / (9 + 4 * Real.sqrt 2))) ∧
  (x ≥ -1 ∧ x ≤ 1) ∧ (3 * x ≥ -1 ∧ 3 * x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l889_88990


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l889_88981

/-- For a quadratic equation qx^2 - 8x + 2 = 0 with q ≠ 0, it has only one solution iff q = 8 -/
theorem unique_solution_quadratic (q : ℝ) (hq : q ≠ 0) :
  (∃! x : ℝ, q * x^2 - 8 * x + 2 = 0) ↔ q = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l889_88981


namespace NUMINAMATH_CALUDE_prob_not_red_is_six_sevenths_l889_88941

/-- The number of red jelly beans in the bag -/
def red_beans : ℕ := 4

/-- The number of green jelly beans in the bag -/
def green_beans : ℕ := 7

/-- The number of yellow jelly beans in the bag -/
def yellow_beans : ℕ := 5

/-- The number of blue jelly beans in the bag -/
def blue_beans : ℕ := 9

/-- The number of purple jelly beans in the bag -/
def purple_beans : ℕ := 3

/-- The total number of jelly beans in the bag -/
def total_beans : ℕ := red_beans + green_beans + yellow_beans + blue_beans + purple_beans

/-- The probability of selecting a jelly bean that is not red -/
def prob_not_red : ℚ := (green_beans + yellow_beans + blue_beans + purple_beans : ℚ) / total_beans

theorem prob_not_red_is_six_sevenths : prob_not_red = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_red_is_six_sevenths_l889_88941


namespace NUMINAMATH_CALUDE_quadratic_congruence_solution_unique_solution_modulo_l889_88983

theorem quadratic_congruence_solution :
  ∃ (x : ℕ), x^2 - x + 2 ≡ 0 [ZMOD 7] ∧ x ≡ 4 [ZMOD 7] := by sorry

theorem unique_solution_modulo :
  ∀ (n : ℕ), n ≥ 2 →
    (∃! (x : ℕ), x^2 - x + 2 ≡ 0 [ZMOD n]) ↔ n = 7 := by sorry

end NUMINAMATH_CALUDE_quadratic_congruence_solution_unique_solution_modulo_l889_88983


namespace NUMINAMATH_CALUDE_centroid_perpendicular_triangle_area_l889_88911

/-- Given a triangle ABC with sides a, b, c, and area S, prove that the area of the triangle 
    formed by the bases of perpendiculars dropped from the centroid to the sides of ABC 
    is equal to (4/9) * (a² + b² + c²) / (a² * b² * c²) * S³ -/
theorem centroid_perpendicular_triangle_area 
  (a b c : ℝ) 
  (S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : S > 0) : 
  ∃ (S_new : ℝ), S_new = (4/9) * (a^2 + b^2 + c^2) / (a^2 * b^2 * c^2) * S^3 := by
  sorry

end NUMINAMATH_CALUDE_centroid_perpendicular_triangle_area_l889_88911


namespace NUMINAMATH_CALUDE_special_triangle_sum_range_l889_88964

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Ensure angles are positive and sum to π
  angle_sum : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π
  -- Ensure sides are positive
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Define the specific conditions for our triangle
def SpecialTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 + t.a * t.b = 4 ∧ t.c = 2

-- State the theorem
theorem special_triangle_sum_range (t : Triangle) (h : SpecialTriangle t) :
  2 < 2 * t.a + t.b ∧ 2 * t.a + t.b < 4 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sum_range_l889_88964


namespace NUMINAMATH_CALUDE_men_in_first_group_l889_88940

/-- The number of hours worked per day -/
def hours_per_day : ℕ := 6

/-- The number of days taken by the first group to complete the work -/
def days_first_group : ℕ := 18

/-- The number of men in the second group -/
def men_second_group : ℕ := 15

/-- The number of days taken by the second group to complete the work -/
def days_second_group : ℕ := 12

/-- The theorem stating that the number of men in the first group is 10 -/
theorem men_in_first_group : 
  ∃ (men_first_group : ℕ), 
    men_first_group * days_first_group * hours_per_day = 
    men_second_group * days_second_group * hours_per_day ∧
    men_first_group = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_men_in_first_group_l889_88940


namespace NUMINAMATH_CALUDE_dilation_determinant_l889_88910

theorem dilation_determinant (D : Matrix (Fin 3) (Fin 3) ℝ) 
  (h1 : D = Matrix.diagonal (λ _ => (5 : ℝ))) 
  (h2 : ∀ (i j : Fin 3), i ≠ j → D i j = 0) : 
  Matrix.det D = 125 := by
  sorry

end NUMINAMATH_CALUDE_dilation_determinant_l889_88910


namespace NUMINAMATH_CALUDE_sarah_apples_l889_88923

theorem sarah_apples (boxes : ℕ) (apples_per_box : ℕ) (h1 : boxes = 7) (h2 : apples_per_box = 7) :
  boxes * apples_per_box = 49 := by
  sorry

end NUMINAMATH_CALUDE_sarah_apples_l889_88923


namespace NUMINAMATH_CALUDE_consecutive_squares_determinant_l889_88924

theorem consecutive_squares_determinant (n : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := λ i j ↦ 
    (n + (i.val * 3 + j.val : ℕ))^2
  Matrix.det M = -6^3 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_determinant_l889_88924


namespace NUMINAMATH_CALUDE_min_value_theorem_l889_88907

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 6) :
  1 / (a + 2) + 2 / (b + 1) ≥ 9 / 10 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2 * b₀ = 6 ∧ 1 / (a₀ + 2) + 2 / (b₀ + 1) = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l889_88907


namespace NUMINAMATH_CALUDE_total_teachers_l889_88951

theorem total_teachers (num_departments : ℕ) (teachers_per_department : ℕ) 
  (h1 : num_departments = 15) 
  (h2 : teachers_per_department = 35) : 
  num_departments * teachers_per_department = 525 := by
  sorry

end NUMINAMATH_CALUDE_total_teachers_l889_88951


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l889_88984

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_a2 : a 2 = 3) 
  (h_a5 : a 5 = 12) : 
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l889_88984


namespace NUMINAMATH_CALUDE_two_digit_sum_divisible_by_11_four_digit_divisible_by_11_l889_88913

-- Problem 1
theorem two_digit_sum_divisible_by_11 (a b : ℕ) (h1 : a < 10) (h2 : b < 10) :
  ∃ k : ℤ, (10 * a + b) + (10 * b + a) = 11 * k :=
sorry

-- Problem 2
theorem four_digit_divisible_by_11 (m n : ℕ) (h1 : m < 10) (h2 : n < 10) :
  ∃ k : ℤ, 1000 * m + 100 * n + 10 * n + m = 11 * k :=
sorry

end NUMINAMATH_CALUDE_two_digit_sum_divisible_by_11_four_digit_divisible_by_11_l889_88913


namespace NUMINAMATH_CALUDE_max_sets_production_l889_88997

/-- Represents the number of sets produced given the number of workers assigned to bolts and nuts -/
def sets_produced (bolt_workers : ℕ) (nut_workers : ℕ) : ℕ :=
  min (25 * bolt_workers) ((20 * nut_workers) / 2)

/-- Theorem stating that 40 bolt workers and 100 nut workers maximize set production -/
theorem max_sets_production :
  ∀ (b n : ℕ),
    b + n = 140 →
    sets_produced b n ≤ sets_produced 40 100 :=
by sorry

end NUMINAMATH_CALUDE_max_sets_production_l889_88997


namespace NUMINAMATH_CALUDE_fifteenth_term_of_inverse_proportional_sequence_l889_88995

/-- A sequence where each term after the first is inversely proportional to the preceding term -/
def InverseProportionalSequence (a : ℕ → ℚ) : Prop :=
  ∃ k : ℚ, k ≠ 0 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) * a n = k

theorem fifteenth_term_of_inverse_proportional_sequence
  (a : ℕ → ℚ)
  (h_seq : InverseProportionalSequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 4) :
  a 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_inverse_proportional_sequence_l889_88995


namespace NUMINAMATH_CALUDE_copies_equal_totient_l889_88966

/-- The pattern function that generates the next row -/
def nextRow (row : List (List Nat)) : List (List Nat) := sorry

/-- The number of copies of n in row n of the pattern -/
def copiesInRow (n : Nat) : Nat := sorry

/-- Euler's Totient function -/
def φ (n : Nat) : Nat := sorry

/-- Theorem stating that the number of copies of 2019 in row 2019 is equal to φ(2019) -/
theorem copies_equal_totient :
  copiesInRow 2019 = φ 2019 := by sorry

end NUMINAMATH_CALUDE_copies_equal_totient_l889_88966


namespace NUMINAMATH_CALUDE_a_10_equals_1023_l889_88962

def sequence_a : ℕ → ℕ
  | 0 => 1
  | n + 1 => sequence_a n + 2^(n + 1)

theorem a_10_equals_1023 : sequence_a 9 = 1023 := by
  sorry

end NUMINAMATH_CALUDE_a_10_equals_1023_l889_88962


namespace NUMINAMATH_CALUDE_max_square_field_size_l889_88914

/-- The maximum size of a square field that can be fully fenced given the specified conditions -/
theorem max_square_field_size (wire_cost : ℝ) (budget : ℝ) : 
  wire_cost = 30 → 
  budget = 120000 → 
  (budget / wire_cost : ℝ) < 4000 → 
  (budget / wire_cost / 4 : ℝ) ^ 2 = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_max_square_field_size_l889_88914


namespace NUMINAMATH_CALUDE_partnership_profit_l889_88968

/-- Given the investments of three partners and the profit share of one partner, 
    calculate the total profit of the partnership. -/
theorem partnership_profit 
  (investment_A investment_B investment_C : ℕ) 
  (profit_share_A : ℕ) 
  (h1 : investment_A = 2400)
  (h2 : investment_B = 7200)
  (h3 : investment_C = 9600)
  (h4 : profit_share_A = 1125) :
  (investment_A + investment_B + investment_C) * profit_share_A / investment_A = 9000 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l889_88968


namespace NUMINAMATH_CALUDE_notebook_cost_l889_88904

theorem notebook_cost (notebook_cost pen_cost : ℚ) 
  (total_cost : notebook_cost + pen_cost = 5/2)
  (price_difference : notebook_cost = pen_cost + 2) :
  notebook_cost = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l889_88904


namespace NUMINAMATH_CALUDE_existence_of_floor_representation_l889_88979

def is_valid_sequence (f : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, 1 ≤ i → 1 ≤ j → i + j ≤ 1997 →
    f i + f j ≤ f (i + j) ∧ f (i + j) ≤ f i + f j + 1

theorem existence_of_floor_representation (f : ℕ → ℕ) :
  is_valid_sequence f →
  ∃ x : ℝ, ∀ n : ℕ, 1 ≤ n → n ≤ 1997 → f n = ⌊n * x⌋ :=
sorry

end NUMINAMATH_CALUDE_existence_of_floor_representation_l889_88979


namespace NUMINAMATH_CALUDE_correct_remaining_time_l889_88987

/-- Represents a food item with its cooking times -/
structure FoodItem where
  name : String
  recommendedTime : Nat
  actualTime : Nat

/-- Calculates the remaining cooking time in seconds for a food item -/
def remainingTimeInSeconds (food : FoodItem) : Nat :=
  (food.recommendedTime - food.actualTime) * 60

/-- The main theorem to prove -/
theorem correct_remaining_time (frenchFries chickenNuggets mozzarellaSticks : FoodItem)
  (h1 : frenchFries.name = "French Fries" ∧ frenchFries.recommendedTime = 12 ∧ frenchFries.actualTime = 2)
  (h2 : chickenNuggets.name = "Chicken Nuggets" ∧ chickenNuggets.recommendedTime = 18 ∧ chickenNuggets.actualTime = 5)
  (h3 : mozzarellaSticks.name = "Mozzarella Sticks" ∧ mozzarellaSticks.recommendedTime = 8 ∧ mozzarellaSticks.actualTime = 3) :
  remainingTimeInSeconds frenchFries = 600 ∧
  remainingTimeInSeconds chickenNuggets = 780 ∧
  remainingTimeInSeconds mozzarellaSticks = 300 := by
  sorry


end NUMINAMATH_CALUDE_correct_remaining_time_l889_88987


namespace NUMINAMATH_CALUDE_triangle_theorem_l889_88977

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem --/
theorem triangle_theorem (t : Triangle) :
  (t.b / (t.a + t.c) = (t.a + t.b - t.c) / (t.a + t.b)) →
  (t.A = π / 3) ∧
  (t.A = π / 3 ∧ t.a = 15 ∧ t.b = 10 → Real.cos t.B = Real.sqrt 6 / 3) := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l889_88977


namespace NUMINAMATH_CALUDE_sum_of_common_elements_l889_88970

/-- Arithmetic progression with first term 4 and common difference 3 -/
def arithmetic_seq (n : ℕ) : ℕ := 4 + 3 * n

/-- Geometric progression with first term 10 and common ratio 2 -/
def geometric_seq (k : ℕ) : ℕ := 10 * 2^k

/-- Sequence of common elements in both progressions -/
def common_seq (m : ℕ) : ℕ := 10 * 4^m

theorem sum_of_common_elements : 
  (Finset.range 10).sum common_seq = 3495250 := by sorry

end NUMINAMATH_CALUDE_sum_of_common_elements_l889_88970


namespace NUMINAMATH_CALUDE_victor_percentage_l889_88922

def max_marks : ℕ := 500
def victor_marks : ℕ := 460

theorem victor_percentage : 
  (victor_marks : ℚ) / max_marks * 100 = 92 := by sorry

end NUMINAMATH_CALUDE_victor_percentage_l889_88922


namespace NUMINAMATH_CALUDE_base8_digit_product_12345_l889_88915

/-- Converts a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Computes the product of a list of natural numbers --/
def listProduct (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

/-- The product of the digits in the base 8 representation of 12345 (base 10) is 0 --/
theorem base8_digit_product_12345 :
  listProduct (toBase8 12345) = 0 := by
  sorry

end NUMINAMATH_CALUDE_base8_digit_product_12345_l889_88915


namespace NUMINAMATH_CALUDE_sum_divisible_by_nine_l889_88901

theorem sum_divisible_by_nine : 
  ∃ k : ℕ, 8230 + 8231 + 8232 + 8233 + 8234 + 8235 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_nine_l889_88901


namespace NUMINAMATH_CALUDE_congruence_problem_l889_88925

theorem congruence_problem (x : ℤ) :
  (4 * x + 9) % 20 = 3 → (3 * x + 15) % 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l889_88925


namespace NUMINAMATH_CALUDE_refrigerator_cash_price_l889_88936

/-- The cash price of a refrigerator given installment payment details --/
theorem refrigerator_cash_price 
  (deposit : ℕ) 
  (num_installments : ℕ) 
  (installment_amount : ℕ) 
  (cash_savings : ℕ) : 
  deposit = 3000 →
  num_installments = 30 →
  installment_amount = 300 →
  cash_savings = 4000 →
  deposit + num_installments * installment_amount - cash_savings = 8000 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_cash_price_l889_88936


namespace NUMINAMATH_CALUDE_gcd_65536_49152_l889_88971

theorem gcd_65536_49152 : Nat.gcd 65536 49152 = 16384 := by
  sorry

end NUMINAMATH_CALUDE_gcd_65536_49152_l889_88971


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l889_88939

theorem integer_roots_of_cubic (a : ℤ) : 
  (∃ x : ℤ, x^3 + 5*x^2 + a*x + 8 = 0) ↔ 
  a ∈ ({-71, -42, -24, -14, 4, 14, 22, 41} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l889_88939


namespace NUMINAMATH_CALUDE_p_satisfies_conditions_l889_88965

/-- The quadratic polynomial p(x) that satisfies given conditions -/
def p (x : ℚ) : ℚ := (12/5) * x^2 - (36/5) * x - 216/5

/-- Theorem stating that p(x) satisfies the required conditions -/
theorem p_satisfies_conditions : 
  p (-3) = 0 ∧ p 6 = 0 ∧ p 2 = -48 := by
  sorry

end NUMINAMATH_CALUDE_p_satisfies_conditions_l889_88965


namespace NUMINAMATH_CALUDE_correct_flight_distance_l889_88975

/-- The total distance Peter needs to fly from Germany to Russia and then back to Spain -/
def total_flight_distance (spain_russia_distance spain_germany_distance : ℕ) : ℕ :=
  (spain_russia_distance - spain_germany_distance) + 2 * spain_germany_distance

/-- Theorem stating the correct total flight distance given the problem conditions -/
theorem correct_flight_distance :
  total_flight_distance 7019 1615 = 8634 := by
  sorry

end NUMINAMATH_CALUDE_correct_flight_distance_l889_88975


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l889_88912

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l889_88912


namespace NUMINAMATH_CALUDE_inequality_solution_l889_88974

theorem inequality_solution :
  let ineq1 : ℝ → Prop := λ x => x > 1
  let ineq2 : ℝ → Prop := λ x => x > 4
  let ineq3 : ℝ → Prop := λ x => 2 - x > -1
  let ineq4 : ℝ → Prop := λ x => x < 2
  (∀ x : ℤ, (ineq1 x ∧ ineq3 x) ↔ x = 2) ∧
  (∀ x : ℤ, ¬(ineq1 x ∧ ineq2 x ∧ x = 2)) ∧
  (∀ x : ℤ, ¬(ineq1 x ∧ ineq4 x ∧ x = 2)) ∧
  (∀ x : ℤ, ¬(ineq2 x ∧ ineq3 x ∧ x = 2)) ∧
  (∀ x : ℤ, ¬(ineq2 x ∧ ineq4 x ∧ x = 2)) ∧
  (∀ x : ℤ, ¬(ineq3 x ∧ ineq4 x ∧ x = 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l889_88974


namespace NUMINAMATH_CALUDE_smallest_steps_l889_88938

theorem smallest_steps (n : ℕ) : 
  n > 20 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 2 ∧ 
  (∀ m : ℕ, m > 20 ∧ m % 6 = 4 ∧ m % 7 = 2 → n ≤ m) → 
  n = 58 := by
sorry

end NUMINAMATH_CALUDE_smallest_steps_l889_88938


namespace NUMINAMATH_CALUDE_trig_identity_l889_88948

theorem trig_identity : 
  Real.cos (54 * π / 180) * Real.cos (24 * π / 180) + 
  2 * Real.sin (12 * π / 180) * Real.cos (12 * π / 180) * Real.sin (126 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l889_88948


namespace NUMINAMATH_CALUDE_money_division_l889_88993

theorem money_division (p q r : ℕ) (total : ℝ) (h1 : p + q + r = 22) (h2 : 12 * total / 22 - 7 * total / 22 = 5000) :
  7 * total / 22 - 3 * total / 22 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l889_88993


namespace NUMINAMATH_CALUDE_complex_number_difference_l889_88950

theorem complex_number_difference : 
  let z : ℂ := (Complex.I * (-6 + Complex.I)) / Complex.abs (3 - 4 * Complex.I)
  (z.re - z.im) = 1 := by sorry

end NUMINAMATH_CALUDE_complex_number_difference_l889_88950
