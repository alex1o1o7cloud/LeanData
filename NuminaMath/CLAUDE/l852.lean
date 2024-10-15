import Mathlib

namespace NUMINAMATH_CALUDE_min_value_of_expression_l852_85207

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  let A := (a^3 + b^3) / (8*a*b + 9 - c^2) + 
           (b^3 + c^3) / (8*b*c + 9 - a^2) + 
           (c^3 + a^3) / (8*c*a + 9 - b^2)
  ∀ x, A ≥ x → x ≤ 3/8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l852_85207


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l852_85236

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 5 / (1 + 2 * I)
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l852_85236


namespace NUMINAMATH_CALUDE_triangle_right_angled_l852_85277

theorem triangle_right_angled (A B C : Real) (a b c : Real) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  a = -c * Real.cos (A + C) →
  a^2 + b^2 = c^2 :=
sorry

end NUMINAMATH_CALUDE_triangle_right_angled_l852_85277


namespace NUMINAMATH_CALUDE_rebecca_eggs_count_l852_85252

/-- Proves that Rebecca has 13 eggs given the problem conditions -/
theorem rebecca_eggs_count :
  ∀ (total_items : ℕ) (group_size : ℕ) (num_groups : ℕ) (num_marbles : ℕ),
    group_size = 2 →
    num_groups = 8 →
    num_marbles = 3 →
    total_items = group_size * num_groups →
    total_items = num_marbles + (total_items - num_marbles) →
    (total_items - num_marbles) = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_rebecca_eggs_count_l852_85252


namespace NUMINAMATH_CALUDE_sphere_circular_views_l852_85271

-- Define a type for geometric bodies
inductive GeometricBody
  | Cone
  | Sphere
  | Cylinder
  | HollowCylinder

-- Define a function to check if a view is circular
def isCircularView (body : GeometricBody) (view : String) : Prop :=
  match body, view with
  | GeometricBody.Sphere, _ => True
  | _, _ => False

-- Main theorem
theorem sphere_circular_views :
  ∀ (body : GeometricBody),
    (isCircularView body "main" ∧
     isCircularView body "left" ∧
     isCircularView body "top") →
    body = GeometricBody.Sphere :=
by sorry

end NUMINAMATH_CALUDE_sphere_circular_views_l852_85271


namespace NUMINAMATH_CALUDE_complex_magnitude_l852_85228

theorem complex_magnitude (z : ℂ) : Complex.abs (z - (1 + 2*I)) = 0 → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l852_85228


namespace NUMINAMATH_CALUDE_square_sum_and_product_l852_85243

theorem square_sum_and_product (a b : ℝ) 
  (h1 : (a + b)^2 = 7) 
  (h2 : (a - b)^2 = 3) : 
  a^2 + b^2 = 5 ∧ a * b = 1 := by
sorry

end NUMINAMATH_CALUDE_square_sum_and_product_l852_85243


namespace NUMINAMATH_CALUDE_points_per_question_l852_85281

theorem points_per_question (correct_answers : ℕ) (final_score : ℕ) : 
  correct_answers = 5 → final_score = 15 → (final_score / correct_answers : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_points_per_question_l852_85281


namespace NUMINAMATH_CALUDE_equation_solution_l852_85254

theorem equation_solution : ∃ y : ℝ, y = (18 : ℝ) / 4 ∧ (8 * y^2 + 50 * y + 3) / (4 * y + 21) = 2 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l852_85254


namespace NUMINAMATH_CALUDE_expression_equals_one_l852_85208

theorem expression_equals_one :
  (144^2 - 12^2) / (120^2 - 18^2) * ((120 - 18) * (120 + 18)) / ((144 - 12) * (144 + 12)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l852_85208


namespace NUMINAMATH_CALUDE_root_product_sum_l852_85279

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧
  (Real.sqrt 2015) * x₁^3 - 4030 * x₁^2 + 2 = 0 ∧
  (Real.sqrt 2015) * x₂^3 - 4030 * x₂^2 + 2 = 0 ∧
  (Real.sqrt 2015) * x₃^3 - 4030 * x₃^2 + 2 = 0 →
  x₂ * (x₁ + x₃) = 2 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l852_85279


namespace NUMINAMATH_CALUDE_first_half_speed_l852_85245

/-- Proves that given a 60-mile trip where the average speed on the second half is 16 mph faster
    than the first half, and the average speed for the entire trip is 30 mph,
    the average speed during the first half is 24 mph. -/
theorem first_half_speed (total_distance : ℝ) (speed_increase : ℝ) (total_avg_speed : ℝ)
  (h1 : total_distance = 60)
  (h2 : speed_increase = 16)
  (h3 : total_avg_speed = 30) :
  ∃ (first_half_speed : ℝ),
    first_half_speed > 0 ∧
    (total_distance / 2) / first_half_speed + (total_distance / 2) / (first_half_speed + speed_increase) = total_distance / total_avg_speed ∧
    first_half_speed = 24 :=
by sorry

end NUMINAMATH_CALUDE_first_half_speed_l852_85245


namespace NUMINAMATH_CALUDE_m_range_theorem_l852_85261

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∀ x y : ℝ, y = x + m - 2 → ¬(x < 0 ∧ y > 0)

def q (m : ℝ) : Prop := 0 < 1 - m ∧ m < 1

-- Define the range of m
def m_range (m : ℝ) : Prop := m ≤ 0 ∨ (1 ≤ m ∧ m ≤ 2)

-- State the theorem
theorem m_range_theorem :
  (∀ m : ℝ, ¬(p m ∧ q m)) →
  (∀ m : ℝ, p m ∨ q m) →
  ∀ m : ℝ, m_range m ↔ (p m ∨ q m) :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l852_85261


namespace NUMINAMATH_CALUDE_martian_traffic_light_signals_l852_85249

/-- Represents a Martian traffic light configuration -/
def MartianTrafficLight := Fin 6 → Bool

/-- The number of bulbs in the traffic light -/
def num_bulbs : Nat := 6

/-- Checks if two configurations are indistinguishable under the given conditions -/
def indistinguishable (c1 c2 : MartianTrafficLight) : Prop :=
  sorry

/-- Counts the number of distinguishable configurations -/
def count_distinguishable_configs : Nat :=
  sorry

/-- Theorem stating the number of distinguishable Martian traffic light signals -/
theorem martian_traffic_light_signals :
  count_distinguishable_configs = 44 :=
sorry

end NUMINAMATH_CALUDE_martian_traffic_light_signals_l852_85249


namespace NUMINAMATH_CALUDE_prime_divisor_of_fermat_number_l852_85230

theorem prime_divisor_of_fermat_number (p k : ℕ) : 
  Prime p → p ∣ (2^(2^k) + 1) → (2^(k+1) ∣ (p - 1)) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisor_of_fermat_number_l852_85230


namespace NUMINAMATH_CALUDE_complex_equation_sum_l852_85291

theorem complex_equation_sum (a b : ℝ) : 
  (a - Complex.I = 2 + b * Complex.I) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l852_85291


namespace NUMINAMATH_CALUDE_triangles_in_4x4_grid_l852_85289

/-- Represents a triangular grid with side length n --/
def TriangularGrid (n : ℕ) := Unit

/-- Counts the number of triangles in a triangular grid --/
def countTriangles (grid : TriangularGrid 4) : ℕ := sorry

/-- Theorem: The number of triangles in a 4x4 triangular grid is 20 --/
theorem triangles_in_4x4_grid :
  ∀ (grid : TriangularGrid 4), countTriangles grid = 20 := by sorry

end NUMINAMATH_CALUDE_triangles_in_4x4_grid_l852_85289


namespace NUMINAMATH_CALUDE_assign_roles_for_five_men_six_women_l852_85232

/-- The number of ways to assign roles in a play --/
def assignRoles (numMen numWomen : ℕ) : ℕ :=
  let maleRoles := 2
  let femaleRoles := 2
  let eitherGenderRoles := 2
  let remainingActors := numMen + numWomen - maleRoles - femaleRoles
  (numMen.descFactorial maleRoles) *
  (numWomen.descFactorial femaleRoles) *
  (remainingActors.descFactorial eitherGenderRoles)

/-- Theorem stating the number of ways to assign roles for 5 men and 6 women --/
theorem assign_roles_for_five_men_six_women :
  assignRoles 5 6 = 25200 := by
  sorry

end NUMINAMATH_CALUDE_assign_roles_for_five_men_six_women_l852_85232


namespace NUMINAMATH_CALUDE_blonde_girls_count_l852_85246

/-- Represents the choir composition -/
structure Choir :=
  (initial_total : ℕ)
  (added_blonde : ℕ)
  (black_haired : ℕ)

/-- Calculates the initial number of blonde-haired girls in the choir -/
def initial_blonde (c : Choir) : ℕ :=
  c.initial_total - c.black_haired

/-- Theorem stating the initial number of blonde-haired girls in the specific choir -/
theorem blonde_girls_count (c : Choir) 
  (h1 : c.initial_total = 80)
  (h2 : c.added_blonde = 10)
  (h3 : c.black_haired = 50) :
  initial_blonde c = 30 := by
  sorry

end NUMINAMATH_CALUDE_blonde_girls_count_l852_85246


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l852_85297

theorem sqrt_sum_inequality (x y α : ℝ) :
  Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α) →
  x + y ≥ 2 * α := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l852_85297


namespace NUMINAMATH_CALUDE_smallest_value_in_ratio_l852_85227

theorem smallest_value_in_ratio (a b c d x y z : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a < b ∧ b < c)
  (h_ratio : ∃ k : ℝ, x = k * a ∧ y = k * b ∧ z = k * c)
  (h_sum : x + y + z = d) :
  min x (min y z) = d * a / (a + b + c) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_value_in_ratio_l852_85227


namespace NUMINAMATH_CALUDE_turtles_jumped_off_l852_85250

/-- The fraction of turtles that jumped off the log --/
def fraction_jumped_off (initial : ℕ) (remaining : ℕ) : ℚ :=
  let additional := 3 * initial - 2
  let total := initial + additional
  (total - remaining) / total

/-- Theorem stating that the fraction of turtles that jumped off is 1/2 --/
theorem turtles_jumped_off :
  fraction_jumped_off 9 17 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_turtles_jumped_off_l852_85250


namespace NUMINAMATH_CALUDE_solution_value_a_l852_85273

theorem solution_value_a (a x y : ℝ) : 
  a * x - 3 * y = 0 ∧ x + y = 1 ∧ 2 * x + y = 0 → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_a_l852_85273


namespace NUMINAMATH_CALUDE_train_length_l852_85259

/-- The length of a train given its speed and time to cross a post -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 40 → time_s = 19.8 → 
  (speed_kmh * 1000 / 3600) * time_s = 220 := by sorry

end NUMINAMATH_CALUDE_train_length_l852_85259


namespace NUMINAMATH_CALUDE_broker_income_slump_l852_85235

/-- 
Proves that if a broker's income remains unchanged when the commission rate 
increases from 4% to 5%, then the percentage slump in business is 20%.
-/
theorem broker_income_slump (X : ℝ) (Y : ℝ) (h : X > 0) :
  (0.04 * X = 0.05 * Y) →  -- Income remains unchanged
  (Y / X = 0.8)            -- Percentage slump in business is 20%
  := by sorry

end NUMINAMATH_CALUDE_broker_income_slump_l852_85235


namespace NUMINAMATH_CALUDE_adam_book_purchase_l852_85219

/-- The number of books Adam bought on his shopping trip -/
def books_bought : ℕ := sorry

/-- The number of books Adam had before shopping -/
def initial_books : ℕ := 56

/-- The number of shelves in Adam's bookcase -/
def num_shelves : ℕ := 4

/-- The average number of books per shelf in Adam's bookcase -/
def avg_books_per_shelf : ℕ := 20

/-- The number of books left over after filling the bookcase -/
def leftover_books : ℕ := 2

/-- The theorem stating how many books Adam bought -/
theorem adam_book_purchase :
  books_bought = 
    num_shelves * avg_books_per_shelf + leftover_books - initial_books :=
by sorry

end NUMINAMATH_CALUDE_adam_book_purchase_l852_85219


namespace NUMINAMATH_CALUDE_sum_mod_seven_l852_85253

theorem sum_mod_seven : (5000 + 5001 + 5002 + 5003 + 5004) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_seven_l852_85253


namespace NUMINAMATH_CALUDE_tiling_cost_theorem_l852_85205

/-- Calculates the total cost of tiling a wall -/
def total_tiling_cost (wall_width wall_height tile_length tile_width tile_cost : ℕ) : ℕ :=
  let wall_area := wall_width * wall_height
  let tile_area := tile_length * tile_width
  let num_tiles := (wall_area + tile_area - 1) / tile_area  -- Ceiling division
  num_tiles * tile_cost

/-- Theorem: The total cost of tiling the given wall is 540,000 won -/
theorem tiling_cost_theorem : 
  total_tiling_cost 36 72 3 4 2500 = 540000 := by
  sorry

end NUMINAMATH_CALUDE_tiling_cost_theorem_l852_85205


namespace NUMINAMATH_CALUDE_nested_sqrt_value_l852_85288

theorem nested_sqrt_value : 
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_value_l852_85288


namespace NUMINAMATH_CALUDE_tan_two_implies_fraction_five_l852_85294

theorem tan_two_implies_fraction_five (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implies_fraction_five_l852_85294


namespace NUMINAMATH_CALUDE_horner_method_v3_l852_85241

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℚ) (x : ℚ) : ℚ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^4 - x^3 + 3x^2 + 7 -/
def f (x : ℚ) : ℚ := 2 * x^4 - x^3 + 3 * x^2 + 7

theorem horner_method_v3 :
  let coeffs := [2, -1, 3, 0, 7]
  let x := 3
  horner coeffs x = 54 ∧ f x = horner coeffs x := by sorry

#check horner_method_v3

end NUMINAMATH_CALUDE_horner_method_v3_l852_85241


namespace NUMINAMATH_CALUDE_xy_value_l852_85284

theorem xy_value (x y : ℝ) (h : |x - 2*y| + (5*x - 7*y - 3)^2 = 0) : x^y = 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l852_85284


namespace NUMINAMATH_CALUDE_min_value_of_function_l852_85296

theorem min_value_of_function (x : ℝ) (h : x ≥ 0) :
  (3 * x^2 + 9 * x + 20) / (7 * (2 + x)) ≥ 10 / 7 ∧
  ∃ y ≥ 0, (3 * y^2 + 9 * y + 20) / (7 * (2 + y)) = 10 / 7 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l852_85296


namespace NUMINAMATH_CALUDE_common_first_digit_is_two_l852_85222

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def first_digit (n : ℕ) : ℕ :=
  if n < 10 then n
  else first_digit (n / 10)

def three_digit_powers_of_2 : Set ℕ :=
  {n | ∃ m : ℕ, n = 2^m ∧ is_three_digit n}

def three_digit_powers_of_3 : Set ℕ :=
  {n | ∃ m : ℕ, n = 3^m ∧ is_three_digit n}

theorem common_first_digit_is_two :
  ∃! d : ℕ, (∃ n ∈ three_digit_powers_of_2, first_digit n = d) ∧
            (∃ m ∈ three_digit_powers_of_3, first_digit m = d) ∧
            d = 2 :=
sorry

end NUMINAMATH_CALUDE_common_first_digit_is_two_l852_85222


namespace NUMINAMATH_CALUDE_birthday_45_days_later_l852_85210

/-- Days of the week represented as integers mod 7 -/
inductive DayOfWeek : Type
| Sunday : DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek

/-- Function to add days to a given day of the week -/
def addDays (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match (start, days % 7) with
  | (DayOfWeek.Sunday, 0) => DayOfWeek.Sunday
  | (DayOfWeek.Sunday, 1) => DayOfWeek.Monday
  | (DayOfWeek.Sunday, 2) => DayOfWeek.Tuesday
  | (DayOfWeek.Sunday, 3) => DayOfWeek.Wednesday
  | (DayOfWeek.Sunday, 4) => DayOfWeek.Thursday
  | (DayOfWeek.Sunday, 5) => DayOfWeek.Friday
  | (DayOfWeek.Sunday, 6) => DayOfWeek.Saturday
  | (DayOfWeek.Monday, 0) => DayOfWeek.Monday
  | (DayOfWeek.Monday, 1) => DayOfWeek.Tuesday
  | (DayOfWeek.Monday, 2) => DayOfWeek.Wednesday
  | (DayOfWeek.Monday, 3) => DayOfWeek.Thursday
  | (DayOfWeek.Monday, 4) => DayOfWeek.Friday
  | (DayOfWeek.Monday, 5) => DayOfWeek.Saturday
  | (DayOfWeek.Monday, 6) => DayOfWeek.Sunday
  | (DayOfWeek.Tuesday, 0) => DayOfWeek.Tuesday
  | (DayOfWeek.Tuesday, 1) => DayOfWeek.Wednesday
  | (DayOfWeek.Tuesday, 2) => DayOfWeek.Thursday
  | (DayOfWeek.Tuesday, 3) => DayOfWeek.Friday
  | (DayOfWeek.Tuesday, 4) => DayOfWeek.Saturday
  | (DayOfWeek.Tuesday, 5) => DayOfWeek.Sunday
  | (DayOfWeek.Tuesday, 6) => DayOfWeek.Monday
  | (DayOfWeek.Wednesday, 0) => DayOfWeek.Wednesday
  | (DayOfWeek.Wednesday, 1) => DayOfWeek.Thursday
  | (DayOfWeek.Wednesday, 2) => DayOfWeek.Friday
  | (DayOfWeek.Wednesday, 3) => DayOfWeek.Saturday
  | (DayOfWeek.Wednesday, 4) => DayOfWeek.Sunday
  | (DayOfWeek.Wednesday, 5) => DayOfWeek.Monday
  | (DayOfWeek.Wednesday, 6) => DayOfWeek.Tuesday
  | (DayOfWeek.Thursday, 0) => DayOfWeek.Thursday
  | (DayOfWeek.Thursday, 1) => DayOfWeek.Friday
  | (DayOfWeek.Thursday, 2) => DayOfWeek.Saturday
  | (DayOfWeek.Thursday, 3) => DayOfWeek.Sunday
  | (DayOfWeek.Thursday, 4) => DayOfWeek.Monday
  | (DayOfWeek.Thursday, 5) => DayOfWeek.Tuesday
  | (DayOfWeek.Thursday, 6) => DayOfWeek.Wednesday
  | (DayOfWeek.Friday, 0) => DayOfWeek.Friday
  | (DayOfWeek.Friday, 1) => DayOfWeek.Saturday
  | (DayOfWeek.Friday, 2) => DayOfWeek.Sunday
  | (DayOfWeek.Friday, 3) => DayOfWeek.Monday
  | (DayOfWeek.Friday, 4) => DayOfWeek.Tuesday
  | (DayOfWeek.Friday, 5) => DayOfWeek.Wednesday
  | (DayOfWeek.Friday, 6) => DayOfWeek.Thursday
  | (DayOfWeek.Saturday, 0) => DayOfWeek.Saturday
  | (DayOfWeek.Saturday, 1) => DayOfWeek.Sunday
  | (DayOfWeek.Saturday, 2) => DayOfWeek.Monday
  | (DayOfWeek.Saturday, 3) => DayOfWeek.Tuesday
  | (DayOfWeek.Saturday, 4) => DayOfWeek.Wednesday
  | (DayOfWeek.Saturday, 5) => DayOfWeek.Thursday
  | (DayOfWeek.Saturday, 6) => DayOfWeek.Friday
  | _ => DayOfWeek.Sunday  -- This case should never happen

theorem birthday_45_days_later (birthday : DayOfWeek) :
  birthday = DayOfWeek.Tuesday → addDays birthday 45 = DayOfWeek.Friday :=
by sorry

end NUMINAMATH_CALUDE_birthday_45_days_later_l852_85210


namespace NUMINAMATH_CALUDE_sqrt_less_than_linear_l852_85260

theorem sqrt_less_than_linear (x : ℝ) (hx : x > 0) : 
  Real.sqrt (1 + x) < 1 + x / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_less_than_linear_l852_85260


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l852_85226

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | 0 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l852_85226


namespace NUMINAMATH_CALUDE_sum_inequality_l852_85234

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_abc : a + b + c = 1) :
  (1 / (b * c + a + 1 / a)) + (1 / (a * c + b + 1 / b)) + (1 / (a * b + c + 1 / c)) ≤ 27 / 31 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l852_85234


namespace NUMINAMATH_CALUDE_properties_of_negative_23_l852_85202

theorem properties_of_negative_23 :
  let x : ℝ := -23
  (∃ y : ℝ, x + y = 0 ∧ y = 23) ∧
  (∃ z : ℝ, x * z = 1 ∧ z = -1/23) ∧
  (abs x = 23) := by
  sorry

end NUMINAMATH_CALUDE_properties_of_negative_23_l852_85202


namespace NUMINAMATH_CALUDE_sum_of_first_six_primes_mod_seventh_prime_l852_85265

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17

theorem sum_of_first_six_primes_mod_seventh_prime : 
  (first_six_primes.sum) % seventh_prime = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_six_primes_mod_seventh_prime_l852_85265


namespace NUMINAMATH_CALUDE_total_lives_game_lives_calculation_l852_85217

theorem total_lives (initial_lives : ℕ) (extra_lives_level1 : ℕ) (extra_lives_level2 : ℕ) :
  initial_lives + extra_lives_level1 + extra_lives_level2 =
  initial_lives + extra_lives_level1 + extra_lives_level2 :=
by sorry

theorem game_lives_calculation :
  let initial_lives : ℕ := 2
  let extra_lives_level1 : ℕ := 6
  let extra_lives_level2 : ℕ := 11
  initial_lives + extra_lives_level1 + extra_lives_level2 = 19 :=
by sorry

end NUMINAMATH_CALUDE_total_lives_game_lives_calculation_l852_85217


namespace NUMINAMATH_CALUDE_weight_measurements_l852_85214

/-- The set of available weights in pounds -/
def weights : List ℕ := [1, 3, 9, 27]

/-- The maximum weight that can be weighed using the given weights -/
def max_weight : ℕ := 40

/-- The number of different weights that can be measured -/
def different_weights : ℕ := 40

/-- Theorem stating the maximum weight and number of different weights -/
theorem weight_measurements :
  (weights.sum = max_weight) ∧
  (∀ w : ℕ, w > 0 ∧ w ≤ max_weight → ∃ combination : List ℕ, combination.all (· ∈ weights) ∧ combination.sum = w) ∧
  (different_weights = max_weight) :=
sorry

end NUMINAMATH_CALUDE_weight_measurements_l852_85214


namespace NUMINAMATH_CALUDE_log_inequality_l852_85242

theorem log_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |Real.log a| > |Real.log b|) : a * b < 1 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l852_85242


namespace NUMINAMATH_CALUDE_colored_copies_count_l852_85221

/-- Represents the number of copies and their costs --/
structure CopyData where
  totalCopies : ℕ
  regularHoursCopies : ℕ
  coloredRegularCost : ℚ
  coloredAfterHoursCost : ℚ
  whiteCopyCost : ℚ
  totalBill : ℚ

/-- Theorem stating that given the conditions, the number of colored copies is 300 --/
theorem colored_copies_count (data : CopyData)
  (h1 : data.totalCopies = 400)
  (h2 : data.regularHoursCopies = 180)
  (h3 : data.coloredRegularCost = 10/100)
  (h4 : data.coloredAfterHoursCost = 8/100)
  (h5 : data.whiteCopyCost = 5/100)
  (h6 : data.totalBill = 45/2)
  : ∃ (coloredCopies : ℕ), coloredCopies = 300 ∧ 
    (coloredCopies : ℚ) * data.coloredRegularCost * (data.regularHoursCopies : ℚ) / data.totalCopies +
    (coloredCopies : ℚ) * data.coloredAfterHoursCost * (data.totalCopies - data.regularHoursCopies : ℚ) / data.totalCopies +
    (data.totalCopies - coloredCopies : ℚ) * data.whiteCopyCost = data.totalBill :=
sorry

end NUMINAMATH_CALUDE_colored_copies_count_l852_85221


namespace NUMINAMATH_CALUDE_all_admissible_triangles_finite_and_generable_l852_85285

-- Define an admissible angle
def AdmissibleAngle (n : ℕ) (m : ℕ) : ℚ := (m * 180) / n

-- Define an admissible triangle
structure AdmissibleTriangle (n : ℕ) where
  angle1 : ℕ
  angle2 : ℕ
  angle3 : ℕ
  sum_180 : AdmissibleAngle n angle1 + AdmissibleAngle n angle2 + AdmissibleAngle n angle3 = 180
  angle1_pos : angle1 > 0
  angle2_pos : angle2 > 0
  angle3_pos : angle3 > 0

-- Define a function to check if two triangles are similar
def areSimilar (n : ℕ) (t1 t2 : AdmissibleTriangle n) : Prop :=
  (t1.angle1 = t2.angle1 ∧ t1.angle2 = t2.angle2 ∧ t1.angle3 = t2.angle3) ∨
  (t1.angle1 = t2.angle2 ∧ t1.angle2 = t2.angle3 ∧ t1.angle3 = t2.angle1) ∨
  (t1.angle1 = t2.angle3 ∧ t1.angle2 = t2.angle1 ∧ t1.angle3 = t2.angle2)

-- Define the set of all possible admissible triangles
def AllAdmissibleTriangles (n : ℕ) : Set (AdmissibleTriangle n) :=
  {t : AdmissibleTriangle n | True}

-- Define the process of cutting triangles
def CutTriangle (n : ℕ) (t : AdmissibleTriangle n) : 
  Option (AdmissibleTriangle n × AdmissibleTriangle n) :=
  sorry -- Implementation details omitted

-- The main theorem
theorem all_admissible_triangles_finite_and_generable 
  (n : ℕ) (h_prime : Nat.Prime n) (h_gt_3 : n > 3) :
  ∃ (S : Set (AdmissibleTriangle n)),
    Finite S ∧ 
    (∀ t ∈ AllAdmissibleTriangles n, ∃ s ∈ S, areSimilar n t s) ∧
    (∀ t ∈ S, CutTriangle n t = none) :=
  sorry


end NUMINAMATH_CALUDE_all_admissible_triangles_finite_and_generable_l852_85285


namespace NUMINAMATH_CALUDE_stratified_sampling_ratio_l852_85223

theorem stratified_sampling_ratio (total : ℕ) (first_year : ℕ) (second_year : ℕ) (selected_first : ℕ) :
  total = first_year + second_year →
  first_year = 30 →
  second_year = 40 →
  selected_first = 6 →
  (selected_first * second_year) / first_year = 8 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_ratio_l852_85223


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l852_85203

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l852_85203


namespace NUMINAMATH_CALUDE_xy_sum_squared_l852_85239

theorem xy_sum_squared (x y : ℝ) (h1 : x * y = -3) (h2 : x + y = -4) :
  x^2 + 3*x*y + y^2 = 13 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_squared_l852_85239


namespace NUMINAMATH_CALUDE_lunch_breakfast_difference_l852_85268

def muffin_cost : ℚ := 2
def coffee_cost : ℚ := 4
def soup_cost : ℚ := 3
def salad_cost : ℚ := 5.25
def lemonade_cost : ℚ := 0.75

def breakfast_cost : ℚ := muffin_cost + coffee_cost
def lunch_cost : ℚ := soup_cost + salad_cost + lemonade_cost

theorem lunch_breakfast_difference :
  lunch_cost - breakfast_cost = 3 := by sorry

end NUMINAMATH_CALUDE_lunch_breakfast_difference_l852_85268


namespace NUMINAMATH_CALUDE_ten_ways_to_distribute_albums_l852_85231

/-- Represents the number of ways to distribute albums to friends -/
def distribute_albums (photo_albums : ℕ) (stamp_albums : ℕ) (friends : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 10 ways to distribute 4 albums to 4 friends -/
theorem ten_ways_to_distribute_albums :
  distribute_albums 2 3 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_ways_to_distribute_albums_l852_85231


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l852_85290

-- Define the color type
inductive Color
  | BLUE
  | GREEN
  | RED
  | YELLOW

-- Define the coloring function type
def ColoringFunction := ℤ → Color

-- Define the property that the coloring function must satisfy
def ValidColoring (f : ColoringFunction) : Prop :=
  ∀ a b c d : ℤ, f a = f b ∧ f b = f c ∧ f c = f d ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0) →
    3 * a - 2 * b ≠ 2 * c - 3 * d

-- State the theorem
theorem exists_valid_coloring : ∃ f : ColoringFunction, ValidColoring f := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l852_85290


namespace NUMINAMATH_CALUDE_two_digit_sum_doubled_l852_85204

theorem two_digit_sum_doubled (J L M K : ℕ) 
  (h_digits : J < 10 ∧ L < 10 ∧ M < 10 ∧ K < 10)
  (h_sum : (10 * J + M) + (10 * L + K) = 79) :
  2 * ((10 * J + M) + (10 * L + K)) = 158 := by
sorry

end NUMINAMATH_CALUDE_two_digit_sum_doubled_l852_85204


namespace NUMINAMATH_CALUDE_orange_count_l852_85251

theorem orange_count (initial_apples : ℕ) (initial_oranges : ℕ) : 
  initial_apples = 14 →
  (initial_apples : ℚ) / (initial_apples + initial_oranges - 14 : ℚ) = 70 / 100 →
  initial_oranges = 20 :=
by sorry

end NUMINAMATH_CALUDE_orange_count_l852_85251


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l852_85293

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_complement_equality : A ∩ (U \ B) = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l852_85293


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l852_85282

theorem at_least_one_geq_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 1 / y ≥ 2) ∨ (y + 1 / z ≥ 2) ∨ (z + 1 / x ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l852_85282


namespace NUMINAMATH_CALUDE_sphere_cube_surface_area_comparison_l852_85247

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

noncomputable def cube_volume (a : ℝ) : ℝ := a^3
noncomputable def cube_surface_area (a : ℝ) : ℝ := 6 * a^2

theorem sphere_cube_surface_area_comparison 
  (r a : ℝ) 
  (h_positive : r > 0 ∧ a > 0) 
  (h_equal_volume : sphere_volume r = cube_volume a) : 
  cube_surface_area a > sphere_surface_area r :=
by
  sorry

#check sphere_cube_surface_area_comparison

end NUMINAMATH_CALUDE_sphere_cube_surface_area_comparison_l852_85247


namespace NUMINAMATH_CALUDE_equation_solution_l852_85244

theorem equation_solution : 
  let S : Set ℝ := {x | 3 * x * (x - 2) = 2 * (x - 2)}
  S = {2/3, 2} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l852_85244


namespace NUMINAMATH_CALUDE_secant_slope_on_curve_l852_85225

def f (x : ℝ) : ℝ := x^2 + x

theorem secant_slope_on_curve (Δx : ℝ) (Δy : ℝ) 
  (h1 : f 2 = 6)  -- P(2, 6) is on the curve
  (h2 : f (2 + Δx) = 6 + Δy)  -- Q(2 + Δx, 6 + Δy) is on the curve
  (h3 : Δx ≠ 0)  -- Ensure Δx is not zero for division
  : Δy / Δx = Δx + 5 := by
  sorry

#check secant_slope_on_curve

end NUMINAMATH_CALUDE_secant_slope_on_curve_l852_85225


namespace NUMINAMATH_CALUDE_first_month_sale_is_2500_l852_85237

/-- Calculates the sale in the first month given the sales in other months and the average -/
def first_month_sale (second_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (average : ℕ) : ℕ :=
  4 * average - (second_month + third_month + fourth_month)

/-- Proves that the sale in the first month is 2500 given the conditions -/
theorem first_month_sale_is_2500 :
  first_month_sale 4000 3540 1520 2890 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_is_2500_l852_85237


namespace NUMINAMATH_CALUDE_line_passes_through_third_quadrant_l852_85267

theorem line_passes_through_third_quadrant 
  (A B C : ℝ) (h1 : A * B < 0) (h2 : B * C < 0) :
  ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ A * x + B * y + C = 0 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_third_quadrant_l852_85267


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l852_85295

/-- Given a circle with equation (x+1)^2 + (y-1)^2 = 4, prove that its center is (-1,1) and its radius is 2 -/
theorem circle_center_and_radius :
  ∃ (C : ℝ × ℝ) (r : ℝ),
    (∀ (x y : ℝ), (x + 1)^2 + (y - 1)^2 = 4 ↔ (x - C.1)^2 + (y - C.2)^2 = r^2) ∧
    C = (-1, 1) ∧
    r = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l852_85295


namespace NUMINAMATH_CALUDE_soda_filling_time_difference_l852_85299

/-- Proves that the additional time needed to fill 12 barrels with a leak is 24 minutes -/
theorem soda_filling_time_difference 
  (normal_time : ℕ) 
  (leak_time : ℕ) 
  (barrel_count : ℕ) 
  (h1 : normal_time = 3)
  (h2 : leak_time = 5)
  (h3 : barrel_count = 12) :
  leak_time * barrel_count - normal_time * barrel_count = 24 := by
  sorry

end NUMINAMATH_CALUDE_soda_filling_time_difference_l852_85299


namespace NUMINAMATH_CALUDE_min_value_of_f_l852_85269

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x

theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧
  f x = -20 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 3 → f y ≥ f x :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l852_85269


namespace NUMINAMATH_CALUDE_pills_remaining_l852_85283

def initial_pills : ℕ := 200
def daily_dose : ℕ := 12
def days : ℕ := 14

theorem pills_remaining : initial_pills - (daily_dose * days) = 32 := by
  sorry

end NUMINAMATH_CALUDE_pills_remaining_l852_85283


namespace NUMINAMATH_CALUDE_investment_scientific_notation_l852_85272

/-- Represents the scientific notation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

/-- The total industrial investment in yuan -/
def total_investment : ℝ := 314.86 * 10^9

theorem investment_scientific_notation :
  to_scientific_notation total_investment = ScientificNotation.mk 3.1486 10 sorry := by
  sorry

end NUMINAMATH_CALUDE_investment_scientific_notation_l852_85272


namespace NUMINAMATH_CALUDE_power_fraction_evaluation_l852_85201

theorem power_fraction_evaluation :
  ((2^1010)^2 - (2^1008)^2) / ((2^1009)^2 - (2^1007)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_evaluation_l852_85201


namespace NUMINAMATH_CALUDE_min_hypotenuse_right_triangle_l852_85286

theorem min_hypotenuse_right_triangle (k : ℝ) (h : k > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + c = k ∧
  a^2 + b^2 = c^2 ∧
  c = (Real.sqrt 2 - 1) * k ∧
  ∀ (a' b' c' : ℝ), a' > 0 → b' > 0 → c' > 0 →
    a' + b' + c' = k → a'^2 + b'^2 = c'^2 → c' ≥ (Real.sqrt 2 - 1) * k := by
  sorry

end NUMINAMATH_CALUDE_min_hypotenuse_right_triangle_l852_85286


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l852_85213

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of smaller boxes that can fit in a larger box -/
def maxBoxes (carton : BoxDimensions) (soapBox : BoxDimensions) : ℕ :=
  (carton.length / soapBox.length) * (carton.width / soapBox.height) * (carton.height / soapBox.width)

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  let carton : BoxDimensions := ⟨48, 25, 60⟩
  let soapBox : BoxDimensions := ⟨8, 6, 5⟩
  maxBoxes carton soapBox = 300 := by
  sorry

#eval maxBoxes ⟨48, 25, 60⟩ ⟨8, 6, 5⟩

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l852_85213


namespace NUMINAMATH_CALUDE_harkamal_payment_l852_85287

/-- The total amount Harkamal paid for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Harkamal paid 965 for his purchase -/
theorem harkamal_payment : total_amount 8 70 9 45 = 965 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_payment_l852_85287


namespace NUMINAMATH_CALUDE_three_digit_cube_ending_777_l852_85224

theorem three_digit_cube_ending_777 :
  ∃! x : ℕ, 100 ≤ x ∧ x < 1000 ∧ x^3 % 1000 = 777 :=
by
  use 753
  sorry

end NUMINAMATH_CALUDE_three_digit_cube_ending_777_l852_85224


namespace NUMINAMATH_CALUDE_women_fair_hair_percentage_is_twenty_percent_l852_85240

/-- Represents the percentage of fair-haired employees who are women -/
def fair_haired_women_ratio : ℝ := 0.4

/-- Represents the percentage of employees who have fair hair -/
def fair_haired_ratio : ℝ := 0.5

/-- Calculates the percentage of employees who are women with fair hair -/
def women_fair_hair_percentage : ℝ := fair_haired_women_ratio * fair_haired_ratio

theorem women_fair_hair_percentage_is_twenty_percent :
  women_fair_hair_percentage = 0.2 := by sorry

end NUMINAMATH_CALUDE_women_fair_hair_percentage_is_twenty_percent_l852_85240


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l852_85206

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^4 + 4*x^2 + 4) = (x^2 + 2) * q x := by
  sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l852_85206


namespace NUMINAMATH_CALUDE_chord_length_problem_l852_85298

/-- The length of the chord formed by the intersection of a line and a circle -/
def chord_length (line_point : ℝ × ℝ) (parallel_line : ℝ → ℝ → ℝ → Prop) 
  (circle_center : ℝ × ℝ) (circle_radius : ℝ) : ℝ :=
  sorry

/-- The problem statement -/
theorem chord_length_problem :
  let line_point := (1, 0)
  let parallel_line := λ x y c => x - Real.sqrt 2 * y + c = 0
  let circle_center := (6, Real.sqrt 2)
  let circle_radius := Real.sqrt 12
  chord_length line_point parallel_line circle_center circle_radius = 6 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_problem_l852_85298


namespace NUMINAMATH_CALUDE_gcf_of_75_and_100_l852_85255

theorem gcf_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_100_l852_85255


namespace NUMINAMATH_CALUDE_jacob_tank_fill_time_l852_85238

/-- Represents the number of days needed to fill a water tank -/
def days_to_fill_tank (tank_capacity_liters : ℕ) (rain_collection_ml : ℕ) (river_collection_ml : ℕ) : ℕ :=
  (tank_capacity_liters * 1000) / (rain_collection_ml + river_collection_ml)

/-- Theorem stating that it takes 20 days to fill Jacob's water tank -/
theorem jacob_tank_fill_time :
  days_to_fill_tank 50 800 1700 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jacob_tank_fill_time_l852_85238


namespace NUMINAMATH_CALUDE_sqrt_square_equality_implies_geq_l852_85218

theorem sqrt_square_equality_implies_geq (a : ℝ) : 
  Real.sqrt ((a - 2)^2) = a - 2 → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_equality_implies_geq_l852_85218


namespace NUMINAMATH_CALUDE_soda_cost_calculation_l852_85216

theorem soda_cost_calculation (regular_bottles : ℕ) (regular_price : ℚ) 
  (diet_bottles : ℕ) (diet_price : ℚ) (regular_discount : ℚ) (diet_tax : ℚ) :
  regular_bottles = 49 →
  regular_price = 120/100 →
  diet_bottles = 40 →
  diet_price = 110/100 →
  regular_discount = 10/100 →
  diet_tax = 8/100 →
  (regular_bottles : ℚ) * regular_price * (1 - regular_discount) + 
  (diet_bottles : ℚ) * diet_price * (1 + diet_tax) = 10044/100 := by
sorry

end NUMINAMATH_CALUDE_soda_cost_calculation_l852_85216


namespace NUMINAMATH_CALUDE_fourth_grade_students_l852_85263

def final_student_count (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

theorem fourth_grade_students :
  final_student_count 10 4 42 = 48 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l852_85263


namespace NUMINAMATH_CALUDE_diamond_seven_three_l852_85256

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := 4 * a - 2 * b

-- Theorem statement
theorem diamond_seven_three : diamond 7 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_diamond_seven_three_l852_85256


namespace NUMINAMATH_CALUDE_susan_remaining_distance_l852_85209

/-- The total number of spaces on the board game --/
def total_spaces : ℕ := 72

/-- Susan's movements over 5 turns --/
def susan_movements : List ℤ := [12, -3, 0, 4, -3]

/-- The theorem stating the remaining distance Susan needs to move --/
theorem susan_remaining_distance :
  total_spaces - (susan_movements.sum) = 62 := by sorry

end NUMINAMATH_CALUDE_susan_remaining_distance_l852_85209


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l852_85270

theorem right_triangle_hypotenuse (x : ℚ) :
  let a := 9
  let b := 3 * x + 6
  let c := x + 15
  (a + b + c = 45) →
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  (max a (max b c) = 75 / 4) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l852_85270


namespace NUMINAMATH_CALUDE_composite_function_equality_l852_85275

theorem composite_function_equality (x : ℝ) (hx : x > 0) :
  Real.sin (Real.log (Real.sqrt x)) = Real.sin ((1 / 2) * Real.log x) := by
  sorry

end NUMINAMATH_CALUDE_composite_function_equality_l852_85275


namespace NUMINAMATH_CALUDE_new_sales_tax_percentage_l852_85292

/-- Proves that the new sales tax percentage is 3 1/3% given the conditions --/
theorem new_sales_tax_percentage
  (market_price : ℝ)
  (original_tax_rate : ℝ)
  (savings : ℝ)
  (h1 : market_price = 10800)
  (h2 : original_tax_rate = 3.5 / 100)
  (h3 : savings = 18) :
  let original_tax := market_price * original_tax_rate
  let new_tax := original_tax - savings
  let new_tax_rate := new_tax / market_price
  new_tax_rate = 10 / 3 / 100 := by sorry

end NUMINAMATH_CALUDE_new_sales_tax_percentage_l852_85292


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_b_eq_neg_twelve_l852_85274

/-- The equation 4(3x-b) = 3(4x+16) has infinitely many solutions x if and only if b = -12 -/
theorem infinite_solutions_iff_b_eq_neg_twelve :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_b_eq_neg_twelve_l852_85274


namespace NUMINAMATH_CALUDE_cube_fourth_power_difference_l852_85264

theorem cube_fourth_power_difference (x y z w : ℕ+) 
  (h1 : x^3 = y^2) 
  (h2 : z^4 = w^3) 
  (h3 : z - x = 22) : 
  ∃ (p q : ℕ+), 
    x = p^2 ∧ 
    y = p^3 ∧ 
    z = q^3 ∧ 
    w = q^4 ∧ 
    q^3 - p^2 = 22 ∧ 
    w - y = (q^(4/3))^3 - p^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_fourth_power_difference_l852_85264


namespace NUMINAMATH_CALUDE_ac_price_is_1500_l852_85233

-- Define the price ratios
def car_ratio : ℚ := 5
def ac_ratio : ℚ := 3
def scooter_ratio : ℚ := 2

-- Define the price difference between scooter and air conditioner
def price_difference : ℚ := 500

-- Define the tax rate for the car
def car_tax_rate : ℚ := 0.1

-- Define the discount rate for the air conditioner
def ac_discount_rate : ℚ := 0.15

-- Define the original price of the air conditioner
def original_ac_price : ℚ := 1500

-- Theorem statement
theorem ac_price_is_1500 :
  ∃ (x : ℚ),
    scooter_ratio * x = ac_ratio * x + price_difference ∧
    original_ac_price = ac_ratio * x :=
by sorry

end NUMINAMATH_CALUDE_ac_price_is_1500_l852_85233


namespace NUMINAMATH_CALUDE_kennel_dogs_count_l852_85257

theorem kennel_dogs_count (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 2 / 3 →
  cats = dogs - 6 →
  dogs = 18 := by
sorry

end NUMINAMATH_CALUDE_kennel_dogs_count_l852_85257


namespace NUMINAMATH_CALUDE_baker_remaining_pastries_l852_85248

theorem baker_remaining_pastries (pastries_made pastries_sold : ℕ) 
  (h1 : pastries_made = 148)
  (h2 : pastries_sold = 103) :
  pastries_made - pastries_sold = 45 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_pastries_l852_85248


namespace NUMINAMATH_CALUDE_can_display_rows_l852_85220

/-- Represents a display of cans arranged in rows -/
structure CanDisplay where
  topRowCans : ℕ
  rowIncrement : ℕ
  totalCans : ℕ

/-- Calculates the number of rows in a can display -/
def numberOfRows (display : CanDisplay) : ℕ :=
  sorry

/-- Theorem: A display with 3 cans in the top row, 2 more cans in each subsequent row, 
    and 225 total cans has 15 rows -/
theorem can_display_rows (display : CanDisplay) 
  (h1 : display.topRowCans = 3)
  (h2 : display.rowIncrement = 2)
  (h3 : display.totalCans = 225) : 
  numberOfRows display = 15 := by
  sorry

end NUMINAMATH_CALUDE_can_display_rows_l852_85220


namespace NUMINAMATH_CALUDE_counterexample_exists_l852_85280

theorem counterexample_exists : ∃ (a b : ℝ), a > b ∧ a^2 ≤ b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l852_85280


namespace NUMINAMATH_CALUDE_roots_product_theorem_l852_85229

theorem roots_product_theorem (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) → 
  q = 16/3 := by
sorry

end NUMINAMATH_CALUDE_roots_product_theorem_l852_85229


namespace NUMINAMATH_CALUDE_f_properties_l852_85212

noncomputable def f (x : ℝ) : ℝ := ((Real.sin x - Real.cos x) * Real.sin (2 * x)) / Real.sin x

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_properties :
  (∀ x : ℝ, f x ≠ 0 ↔ x ∉ {y | ∃ k : ℤ, y = k * Real.pi}) ∧
  (∃ T : ℝ, T > 0 ∧ is_periodic f T ∧ ∀ S, (S > 0 ∧ is_periodic f S) → T ≤ S) ∧
  (∀ x : ℝ, f x ≥ 0 ↔ ∃ k : ℤ, x ∈ Set.Icc (Real.pi / 4 + k * Real.pi) (Real.pi / 2 + k * Real.pi)) ∧
  (∃ m : ℝ, m > 0 ∧ is_even (fun x ↦ f (x + m)) ∧
    ∀ n : ℝ, (n > 0 ∧ is_even (fun x ↦ f (x + n))) → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l852_85212


namespace NUMINAMATH_CALUDE_average_speed_calculation_l852_85266

def initial_reading : ℕ := 2332
def final_reading : ℕ := 2552
def driving_time_day1 : ℕ := 5
def driving_time_day2 : ℕ := 3

theorem average_speed_calculation :
  let total_distance : ℕ := final_reading - initial_reading
  let total_time : ℕ := driving_time_day1 + driving_time_day2
  (total_distance : ℚ) / (total_time : ℚ) = 27.5 := by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l852_85266


namespace NUMINAMATH_CALUDE_root_sum_eighth_power_l852_85200

theorem root_sum_eighth_power (r s : ℝ) : 
  r^2 - r * Real.sqrt 5 + 1 = 0 ∧ 
  s^2 - s * Real.sqrt 5 + 1 = 0 → 
  r^8 + s^8 = 47 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_eighth_power_l852_85200


namespace NUMINAMATH_CALUDE_original_movie_length_l852_85278

/-- The original length of a movie, given the length of a cut scene and the final length -/
theorem original_movie_length (cut_scene_length final_length : ℕ) :
  cut_scene_length = 8 ∧ final_length = 52 →
  cut_scene_length + final_length = 60 := by
  sorry

#check original_movie_length

end NUMINAMATH_CALUDE_original_movie_length_l852_85278


namespace NUMINAMATH_CALUDE_a_gt_3_sufficient_not_necessary_for_abs_a_gt_3_l852_85258

theorem a_gt_3_sufficient_not_necessary_for_abs_a_gt_3 :
  (∃ a : ℝ, a > 3 → |a| > 3) ∧ 
  (∃ a : ℝ, |a| > 3 ∧ ¬(a > 3)) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_3_sufficient_not_necessary_for_abs_a_gt_3_l852_85258


namespace NUMINAMATH_CALUDE_cone_surface_area_l852_85262

/-- The surface area of a cone with lateral surface as a sector of a circle 
    with radius 2 and central angle π/2 is 5π/4 -/
theorem cone_surface_area : 
  ∀ (cone : Real → Real → Real),
  (∀ r θ, cone r θ = 2 * π * r^2 * (θ / (2 * π)) + π * r^2) →
  cone 2 (π / 2) = 5 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l852_85262


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l852_85211

theorem least_subtraction_for_divisibility :
  ∃! x : ℕ, x ≤ 11 ∧ (427398 - x) % 12 = 0 ∧ ∀ y : ℕ, y < x → (427398 - y) % 12 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l852_85211


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l852_85215

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ x y : ℝ, x > y → x + 1 > y) ∧
  (∃ x y : ℝ, x + 1 > y ∧ ¬(x > y)) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l852_85215


namespace NUMINAMATH_CALUDE_sequence_properties_l852_85276

/-- The sum of the first n terms of sequence a_n -/
def S (n : ℕ) : ℚ := (1/2) * n^2 + (1/2) * n

/-- The general term of sequence a_n -/
def a (n : ℕ) : ℚ := n

/-- The n-th term of sequence b_n -/
def b (n : ℕ) : ℚ := a n * 2^(n-1)

/-- The sum of the first n terms of sequence b_n -/
def T (n : ℕ) : ℚ := (n-1) * 2^n + 1

theorem sequence_properties (n : ℕ) :
  (∀ k, S k = (1/2) * k^2 + (1/2) * k) →
  (a n = n) ∧
  (T n = (n-1) * 2^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l852_85276
