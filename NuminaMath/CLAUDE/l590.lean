import Mathlib

namespace NUMINAMATH_CALUDE_oil_price_rollback_l590_59068

def current_price : ℝ := 1.4
def liters_today : ℝ := 10
def liters_friday : ℝ := 25
def total_liters : ℝ := liters_today + liters_friday
def total_spend : ℝ := 39

theorem oil_price_rollback :
  let friday_price := (total_spend - current_price * liters_today) / liters_friday
  current_price - friday_price = 0.4 := by sorry

end NUMINAMATH_CALUDE_oil_price_rollback_l590_59068


namespace NUMINAMATH_CALUDE_receipts_change_after_price_reduction_and_sales_increase_l590_59055

/-- Calculates the percentage change in total receipts when price is reduced and sales increase -/
theorem receipts_change_after_price_reduction_and_sales_increase
  (original_price : ℝ)
  (original_sales : ℝ)
  (price_reduction_percent : ℝ)
  (sales_increase_percent : ℝ)
  (h1 : price_reduction_percent = 30)
  (h2 : sales_increase_percent = 50)
  : (((1 - price_reduction_percent / 100) * (1 + sales_increase_percent / 100) - 1) * 100 = 5) := by
  sorry

end NUMINAMATH_CALUDE_receipts_change_after_price_reduction_and_sales_increase_l590_59055


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l590_59009

theorem min_distance_between_curves (a b c d : ℝ) 
  (h1 : (a + 3 * Real.log a) / b = 1)
  (h2 : (d - 3) / (2 * c) = 1) :
  (∀ x y z w : ℝ, (x + 3 * Real.log x) / y = 1 → (w - 3) / (2 * z) = 1 → 
    (a - c)^2 + (b - d)^2 ≤ (x - z)^2 + (y - w)^2) ∧
  (a - c)^2 + (b - d)^2 = 9/5 * Real.log (9/Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l590_59009


namespace NUMINAMATH_CALUDE_cash_percentage_is_twenty_percent_l590_59069

def total_amount : ℝ := 137500
def raw_materials : ℝ := 80000
def machinery : ℝ := 30000

def cash : ℝ := total_amount - (raw_materials + machinery)

theorem cash_percentage_is_twenty_percent :
  (cash / total_amount) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cash_percentage_is_twenty_percent_l590_59069


namespace NUMINAMATH_CALUDE_cuboid_volumes_sum_l590_59010

theorem cuboid_volumes_sum (length width height1 height2 : ℝ) 
  (h1 : length = 44)
  (h2 : width = 35)
  (h3 : height1 = 7)
  (h4 : height2 = 3) :
  length * width * height1 + length * width * height2 = 15400 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volumes_sum_l590_59010


namespace NUMINAMATH_CALUDE_pens_left_in_jar_l590_59095

/-- Represents the number of pens of each color in the jar -/
structure PenCount where
  blue : ℕ
  black : ℕ
  red : ℕ
  green : ℕ
  purple : ℕ

def initial_pens : PenCount :=
  { blue := 15, black := 27, red := 12, green := 10, purple := 8 }

def removed_pens : PenCount :=
  { blue := 8, black := 9, red := 3, green := 5, purple := 6 }

def remaining_pens (initial : PenCount) (removed : PenCount) : PenCount :=
  { blue := initial.blue - removed.blue,
    black := initial.black - removed.black,
    red := initial.red - removed.red,
    green := initial.green - removed.green,
    purple := initial.purple - removed.purple }

def total_pens (pens : PenCount) : ℕ :=
  pens.blue + pens.black + pens.red + pens.green + pens.purple

theorem pens_left_in_jar :
  total_pens (remaining_pens initial_pens removed_pens) = 41 := by
  sorry

end NUMINAMATH_CALUDE_pens_left_in_jar_l590_59095


namespace NUMINAMATH_CALUDE_problem_1_l590_59053

theorem problem_1 (α : Real) (h : 2 * Real.sin α - Real.cos α = 0) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) + 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -10/3 := by
sorry

end NUMINAMATH_CALUDE_problem_1_l590_59053


namespace NUMINAMATH_CALUDE_theater_seat_increment_l590_59032

/-- Represents a theater with a specific seating arrangement -/
structure Theater where
  num_rows : ℕ
  first_row_seats : ℕ
  last_row_seats : ℕ
  total_seats : ℕ

/-- 
  Given a theater with 23 rows, where the first row has 14 seats, 
  the last row has 56 seats, and the total number of seats is 770, 
  prove that the number of additional seats in each row compared 
  to the previous row is 2.
-/
theorem theater_seat_increment (t : Theater) 
  (h1 : t.num_rows = 23)
  (h2 : t.first_row_seats = 14)
  (h3 : t.last_row_seats = 56)
  (h4 : t.total_seats = 770) : 
  (t.last_row_seats - t.first_row_seats) / (t.num_rows - 1) = 2 := by
  sorry


end NUMINAMATH_CALUDE_theater_seat_increment_l590_59032


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l590_59001

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 1 - x) ↔ x ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l590_59001


namespace NUMINAMATH_CALUDE_distance_Cara_approx_l590_59049

/-- The distance between two skaters on a frozen lake --/
def distance_CD : ℝ := 100

/-- Cara's skating speed in meters per second --/
def speed_Cara : ℝ := 9

/-- Danny's skating speed in meters per second --/
def speed_Danny : ℝ := 6

/-- The angle between Cara's path and the line CD in degrees --/
def angle_Cara : ℝ := 75

/-- The time it takes for Cara and Danny to meet --/
noncomputable def meeting_time : ℝ := 
  let a : ℝ := 45
  let b : ℝ := -1800 * Real.cos (angle_Cara * Real.pi / 180)
  let c : ℝ := 10000
  (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)

/-- The distance Cara skates before meeting Danny --/
noncomputable def distance_Cara : ℝ := speed_Cara * meeting_time

/-- Theorem stating that the distance Cara skates is approximately 27.36144 meters --/
theorem distance_Cara_approx : 
  ∃ ε > 0, abs (distance_Cara - 27.36144) < ε :=
by sorry

end NUMINAMATH_CALUDE_distance_Cara_approx_l590_59049


namespace NUMINAMATH_CALUDE_alicia_science_books_l590_59028

/-- Represents the number of science books Alicia bought -/
def science_books : ℕ := sorry

/-- Represents the cost of a math book -/
def math_book_cost : ℕ := 3

/-- Represents the cost of a science book -/
def science_book_cost : ℕ := 3

/-- Represents the cost of an art book -/
def art_book_cost : ℕ := 2

/-- Represents the number of math books Alicia bought -/
def math_books : ℕ := 2

/-- Represents the number of art books Alicia bought -/
def art_books : ℕ := 3

/-- Represents the total cost of all books -/
def total_cost : ℕ := 30

/-- Theorem stating that Alicia bought 6 science books -/
theorem alicia_science_books : 
  math_books * math_book_cost + art_books * art_book_cost + science_books * science_book_cost = total_cost → 
  science_books = 6 := by
  sorry

end NUMINAMATH_CALUDE_alicia_science_books_l590_59028


namespace NUMINAMATH_CALUDE_dividend_proof_l590_59004

theorem dividend_proof (divisor quotient remainder dividend : ℕ) : 
  divisor = 17 →
  remainder = 6 →
  dividend = divisor * quotient + remainder →
  dividend = 159 :=
by
  sorry

end NUMINAMATH_CALUDE_dividend_proof_l590_59004


namespace NUMINAMATH_CALUDE_hostel_expenditure_increase_l590_59065

/-- Calculates the increase in total expenditure for a hostel after accommodating more students. -/
theorem hostel_expenditure_increase
  (initial_students : ℕ)
  (additional_students : ℕ)
  (average_decrease : ℚ)
  (new_total_expenditure : ℚ)
  (h1 : initial_students = 100)
  (h2 : additional_students = 20)
  (h3 : average_decrease = 5)
  (h4 : new_total_expenditure = 5400) :
  let total_students := initial_students + additional_students
  let new_average := new_total_expenditure / total_students
  let original_average := new_average + average_decrease
  let original_total_expenditure := original_average * initial_students
  new_total_expenditure - original_total_expenditure = 400 :=
by sorry

end NUMINAMATH_CALUDE_hostel_expenditure_increase_l590_59065


namespace NUMINAMATH_CALUDE_plane_cost_calculation_l590_59082

/-- The cost of taking a plane to the Island of Mysteries --/
def plane_cost : ℕ := 600

/-- The cost of taking a boat to the Island of Mysteries --/
def boat_cost : ℕ := 254

/-- The amount saved by taking the boat instead of the plane --/
def savings : ℕ := 346

/-- Theorem stating that the plane cost is equal to the boat cost plus the savings --/
theorem plane_cost_calculation : plane_cost = boat_cost + savings := by
  sorry

end NUMINAMATH_CALUDE_plane_cost_calculation_l590_59082


namespace NUMINAMATH_CALUDE_fish_tank_problem_l590_59076

/-- Given 3 fish tanks with a total of 100 fish, where two tanks have twice as many fish as the first tank, prove that the first tank contains 20 fish. -/
theorem fish_tank_problem (first_tank : ℕ) : 
  first_tank + 2 * first_tank + 2 * first_tank = 100 → first_tank = 20 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_problem_l590_59076


namespace NUMINAMATH_CALUDE_quadratic_inequality_max_value_l590_59014

theorem quadratic_inequality_max_value (a b c : ℝ) (ha : a > 0) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 2 * a * x + b) →
  (∃ M : ℝ, M = 2 * Real.sqrt 2 - 2 ∧ 
    ∀ k : ℝ, k * (a^2 + c^2) ≤ b^2 → k ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_max_value_l590_59014


namespace NUMINAMATH_CALUDE_probability_five_green_marbles_l590_59046

def total_marbles : ℕ := 20
def green_marbles : ℕ := 12
def purple_marbles : ℕ := 8
def total_draws : ℕ := 10
def green_draws : ℕ := 5

theorem probability_five_green_marbles :
  (Nat.choose total_draws green_draws : ℚ) * ((green_marbles : ℚ) / total_marbles) ^ green_draws * ((purple_marbles : ℚ) / total_marbles) ^ (total_draws - green_draws) =
  (Nat.choose 10 5 : ℚ) * (12 / 20) ^ 5 * (8 / 20) ^ 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_green_marbles_l590_59046


namespace NUMINAMATH_CALUDE_min_sum_squares_l590_59097

theorem min_sum_squares (x y z : ℝ) (h : 2*x + 3*y + 4*z = 11) :
  x^2 + y^2 + z^2 ≥ 121/29 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l590_59097


namespace NUMINAMATH_CALUDE_triangle_centroid_distance_sum_l590_59017

/-- Given a triangle ABC with centroid G, if GA² + GB² + GC² = 72, then AB² + AC² + BC² = 216 -/
theorem triangle_centroid_distance_sum (A B C G : ℝ × ℝ) : 
  (G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) →
  ((G.1 - A.1)^2 + (G.2 - A.2)^2 + 
   (G.1 - B.1)^2 + (G.2 - B.2)^2 + 
   (G.1 - C.1)^2 + (G.2 - C.2)^2 = 72) →
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 + 
   (A.1 - C.1)^2 + (A.2 - C.2)^2 + 
   (B.1 - C.1)^2 + (B.2 - C.2)^2 = 216) := by
sorry

end NUMINAMATH_CALUDE_triangle_centroid_distance_sum_l590_59017


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l590_59035

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit on the floor -/
def maxTiles (floor : Dimensions) (tile : Dimensions) : ℕ :=
  let horizontal := (floor.length / tile.length) * (floor.width / tile.width)
  let vertical := (floor.length / tile.width) * (floor.width / tile.length)
  max horizontal vertical

/-- Theorem stating the maximum number of tiles that can be accommodated on the floor -/
theorem max_tiles_on_floor :
  let floor : Dimensions := ⟨390, 150⟩
  let tile : Dimensions := ⟨65, 25⟩
  maxTiles floor tile = 36 := by
  sorry

#eval maxTiles ⟨390, 150⟩ ⟨65, 25⟩

end NUMINAMATH_CALUDE_max_tiles_on_floor_l590_59035


namespace NUMINAMATH_CALUDE_parabola_sum_is_vertical_l590_59087

/-- Original parabola function -/
def original_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Reflected and left-translated parabola -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 - b * (x + 3) + c

/-- Reflected and right-translated parabola -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * (x - 4) + c

/-- Sum of f and g -/
def f_plus_g (a b c : ℝ) (x : ℝ) : ℝ := f a b c x + g a b c x

theorem parabola_sum_is_vertical (a b c : ℝ) :
  ∃ A C : ℝ, ∀ x : ℝ, f_plus_g a b c x = A * x^2 + C :=
sorry

end NUMINAMATH_CALUDE_parabola_sum_is_vertical_l590_59087


namespace NUMINAMATH_CALUDE_polynomial_integral_theorem_l590_59074

/-- A polynomial of degree at most 2 -/
def Polynomial2 := ℝ → ℝ

/-- The definite integral of a polynomial from a to b -/
noncomputable def integral (f : Polynomial2) (a b : ℝ) : ℝ := sorry

/-- The condition that the integrals sum to zero -/
def integralCondition (f : Polynomial2) (p q r : ℝ) : Prop :=
  integral f (-1) p - integral f p q + integral f q r - integral f r 1 = 0

theorem polynomial_integral_theorem :
  ∃! (p q r : ℝ), 
    -1 < p ∧ p < q ∧ q < r ∧ r < 1 ∧
    (∀ f : Polynomial2, integralCondition f p q r) ∧
    p = 1 / Real.sqrt 2 ∧ q = 0 ∧ r = -1 / Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_polynomial_integral_theorem_l590_59074


namespace NUMINAMATH_CALUDE_ten_liter_barrel_emptying_ways_l590_59078

def emptyBarrel (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | n + 2 => emptyBarrel (n + 1) + emptyBarrel n

theorem ten_liter_barrel_emptying_ways :
  emptyBarrel 10 = 89 := by sorry

end NUMINAMATH_CALUDE_ten_liter_barrel_emptying_ways_l590_59078


namespace NUMINAMATH_CALUDE_no_triangle_with_cube_sum_equal_product_l590_59060

theorem no_triangle_with_cube_sum_equal_product : ¬∃ (x y z : ℝ), 
  (0 < x ∧ 0 < y ∧ 0 < z) ∧  -- positive real numbers
  (x + y > z ∧ y + z > x ∧ z + x > y) ∧  -- triangle inequality
  (x^3 + y^3 + z^3 = (x+y)*(y+z)*(z+x)) := by
  sorry


end NUMINAMATH_CALUDE_no_triangle_with_cube_sum_equal_product_l590_59060


namespace NUMINAMATH_CALUDE_sequence_inequality_l590_59086

theorem sequence_inequality (a : ℕ → ℕ) 
  (h1 : a 2 > a 1) 
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 2) = 3 * a (n + 1) - 2 * a n) : 
  a 2021 > 2^2019 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l590_59086


namespace NUMINAMATH_CALUDE_min_value_of_expression_l590_59033

theorem min_value_of_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 3 * m + n = 1) :
  (1 / m + 3 / n) ≥ 12 ∧ ∃ m n, m > 0 ∧ n > 0 ∧ 3 * m + n = 1 ∧ 1 / m + 3 / n = 12 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l590_59033


namespace NUMINAMATH_CALUDE_no_solutions_for_star_equation_l590_59023

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y

-- Theorem statement
theorem no_solutions_for_star_equation :
  ¬ ∃ y : ℝ, star 2 y = 20 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_star_equation_l590_59023


namespace NUMINAMATH_CALUDE_files_remaining_l590_59096

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 13)
  (h2 : video_files = 30)
  (h3 : deleted_files = 10) :
  music_files + video_files - deleted_files = 33 := by
  sorry

end NUMINAMATH_CALUDE_files_remaining_l590_59096


namespace NUMINAMATH_CALUDE_tallest_building_height_l590_59093

theorem tallest_building_height :
  ∀ (h1 h2 h3 h4 : ℝ),
    h2 = h1 / 2 →
    h3 = h2 / 2 →
    h4 = h3 / 5 →
    h1 + h2 + h3 + h4 = 180 →
    h1 = 100 := by
  sorry

end NUMINAMATH_CALUDE_tallest_building_height_l590_59093


namespace NUMINAMATH_CALUDE_benny_eggs_count_l590_59042

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Benny bought -/
def dozens_bought : ℕ := 7

/-- The total number of eggs Benny bought -/
def total_eggs : ℕ := dozens_bought * eggs_per_dozen

theorem benny_eggs_count : total_eggs = 84 := by
  sorry

end NUMINAMATH_CALUDE_benny_eggs_count_l590_59042


namespace NUMINAMATH_CALUDE_gcd_102_238_l590_59084

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l590_59084


namespace NUMINAMATH_CALUDE_adjacent_pairs_after_10_minutes_l590_59080

/-- Represents the number of adjacent pairs of the same letter after n minutes -/
def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n+1 => 2^n - 1 - a n

/-- The transformation rule applied for n minutes -/
def transform (n : ℕ) : String :=
  match n with
  | 0 => "A"
  | n+1 => String.replace (transform n) "A" "AB" |>.replace "B" "BA"

theorem adjacent_pairs_after_10_minutes :
  (transform 10).length = 1024 ∧ a 10 = 341 := by
  sorry

#eval a 10  -- Should output 341

end NUMINAMATH_CALUDE_adjacent_pairs_after_10_minutes_l590_59080


namespace NUMINAMATH_CALUDE_green_bean_to_onion_ratio_l590_59079

def potato_count : ℕ := 2
def carrot_to_potato_ratio : ℕ := 6
def onion_to_carrot_ratio : ℕ := 2
def green_bean_count : ℕ := 8

def carrot_count : ℕ := potato_count * carrot_to_potato_ratio
def onion_count : ℕ := carrot_count * onion_to_carrot_ratio

theorem green_bean_to_onion_ratio :
  (green_bean_count : ℚ) / onion_count = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_green_bean_to_onion_ratio_l590_59079


namespace NUMINAMATH_CALUDE_max_salary_proof_l590_59071

/-- The number of players in a team -/
def team_size : ℕ := 25

/-- The minimum salary for a player -/
def min_salary : ℕ := 15000

/-- The total salary cap for a team -/
def salary_cap : ℕ := 850000

/-- The maximum possible salary for a single player -/
def max_player_salary : ℕ := 490000

theorem max_salary_proof :
  (team_size - 1) * min_salary + max_player_salary = salary_cap ∧
  ∀ (x : ℕ), x > max_player_salary →
    (team_size - 1) * min_salary + x > salary_cap :=
by sorry

end NUMINAMATH_CALUDE_max_salary_proof_l590_59071


namespace NUMINAMATH_CALUDE_committee_selection_l590_59081

theorem committee_selection (n m : ℕ) (hn : n = 20) (hm : m = 3) :
  Nat.choose n m = 1140 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l590_59081


namespace NUMINAMATH_CALUDE_lcm_nine_six_l590_59052

theorem lcm_nine_six : Nat.lcm 9 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_lcm_nine_six_l590_59052


namespace NUMINAMATH_CALUDE_complex_power_four_l590_59092

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Theorem: (1+i)^4 = -4 -/
theorem complex_power_four : (1 + i)^4 = -4 := by sorry

end NUMINAMATH_CALUDE_complex_power_four_l590_59092


namespace NUMINAMATH_CALUDE_equation_solution_a_l590_59019

theorem equation_solution_a : 
  ∀ (a b c : ℤ), 
  (∀ x : ℝ, (x - a) * (x - 5) + 4 = (x + b) * (x + c)) → 
  (a = 0 ∨ a = 1) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_a_l590_59019


namespace NUMINAMATH_CALUDE_infinite_solutions_diophantine_equation_l590_59058

theorem infinite_solutions_diophantine_equation :
  ∃ f g h : ℕ → ℕ,
    (∀ t : ℕ, (f t)^2 + (g t)^3 = (h t)^5) ∧
    (∀ t₁ t₂ : ℕ, t₁ ≠ t₂ → (f t₁, g t₁, h t₁) ≠ (f t₂, g t₂, h t₂)) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_diophantine_equation_l590_59058


namespace NUMINAMATH_CALUDE_tomatoes_left_theorem_l590_59006

/-- Calculates the number of tomatoes left after processing -/
def tomatoes_left (plants : ℕ) (tomatoes_per_plant : ℕ) : ℕ :=
  let total := plants * tomatoes_per_plant
  let dried := total / 2
  let remaining := total - dried
  let marinara := remaining / 3
  remaining - marinara

/-- Theorem: Given 18 plants with 7 tomatoes each, after processing, 42 tomatoes are left -/
theorem tomatoes_left_theorem : tomatoes_left 18 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_theorem_l590_59006


namespace NUMINAMATH_CALUDE_proposition_p_and_not_q_l590_59043

theorem proposition_p_and_not_q : 
  (∃ x : ℝ, x^2 - x + 1 ≥ 0) ∧ 
  ¬(∀ a b : ℝ, a^2 < b^2 → a < b) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_and_not_q_l590_59043


namespace NUMINAMATH_CALUDE_smallest_stairs_l590_59059

theorem smallest_stairs (n : ℕ) : 
  n > 10 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 → 
  n ≥ 52 ∧ 
  ∃ (m : ℕ), m > 10 ∧ m % 6 = 4 ∧ m % 7 = 3 ∧ m = 52 := by
  sorry

end NUMINAMATH_CALUDE_smallest_stairs_l590_59059


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l590_59039

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 17)
  (h2 : c + a = 18)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 36 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l590_59039


namespace NUMINAMATH_CALUDE_hot_sauce_duration_l590_59073

-- Define constants
def serving_size : ℚ := 1/2
def servings_per_day : ℕ := 3
def quart_in_ounces : ℕ := 32
def container_size_difference : ℕ := 2

-- Define the container size
def container_size : ℕ := quart_in_ounces - container_size_difference

-- Define daily usage
def daily_usage : ℚ := serving_size * servings_per_day

-- Theorem to prove
theorem hot_sauce_duration :
  (container_size : ℚ) / daily_usage = 20 := by sorry

end NUMINAMATH_CALUDE_hot_sauce_duration_l590_59073


namespace NUMINAMATH_CALUDE_largest_multiple_of_daytona_sharks_l590_59056

def daytona_sharks : ℕ := 12
def cape_may_sharks : ℕ := 32

theorem largest_multiple_of_daytona_sharks : 
  ∃ (m : ℕ), m * daytona_sharks < cape_may_sharks ∧ 
  ∀ (n : ℕ), n * daytona_sharks < cape_may_sharks → n ≤ m ∧ 
  m = 2 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_daytona_sharks_l590_59056


namespace NUMINAMATH_CALUDE_octal_calculation_l590_59045

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Addition operation for octal numbers --/
def octal_add (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Subtraction operation for octal numbers --/
def octal_sub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Conversion from decimal to octal --/
def to_octal (n : ℕ) : OctalNumber :=
  sorry

/-- Theorem: ($451_8 + 162_8) - 123_8 = 510_8$ in base 8 --/
theorem octal_calculation : 
  octal_sub (octal_add (to_octal 451) (to_octal 162)) (to_octal 123) = to_octal 510 :=
by sorry

end NUMINAMATH_CALUDE_octal_calculation_l590_59045


namespace NUMINAMATH_CALUDE_sin_2x_derivative_at_pi_6_l590_59020

theorem sin_2x_derivative_at_pi_6 (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin (2 * x)) :
  deriv f (π / 6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_derivative_at_pi_6_l590_59020


namespace NUMINAMATH_CALUDE_yulgi_pocket_money_l590_59018

/-- Proves that Yulgi's pocket money is 3600 won given the problem conditions -/
theorem yulgi_pocket_money :
  ∀ (y g : ℕ),
  y + g = 6000 →
  (y + g) - (y - g) = 4800 →
  y > g →
  y = 3600 := by
sorry

end NUMINAMATH_CALUDE_yulgi_pocket_money_l590_59018


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l590_59044

theorem max_product_sum_2000 : 
  (∃ (a b : ℤ), a + b = 2000 ∧ a * b = 1000000) ∧
  (∀ (x y : ℤ), x + y = 2000 → x * y ≤ 1000000) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l590_59044


namespace NUMINAMATH_CALUDE_otimes_inequality_solutions_l590_59094

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a * (a - b) + 1

-- Define the set of non-negative integers
def NonNegIntegers : Set ℤ := {x : ℤ | x ≥ 0}

-- Theorem statement
theorem otimes_inequality_solutions :
  {x ∈ NonNegIntegers | otimes 2 x ≥ 3} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_otimes_inequality_solutions_l590_59094


namespace NUMINAMATH_CALUDE_total_legs_in_park_l590_59011

/-- The total number of legs in a park with various animals, some with missing legs -/
def total_legs : ℕ :=
  let num_dogs : ℕ := 109
  let num_cats : ℕ := 37
  let num_birds : ℕ := 52
  let num_spiders : ℕ := 19
  let dogs_missing_legs : ℕ := 4
  let cats_missing_legs : ℕ := 3
  let spiders_missing_legs : ℕ := 2
  let dog_legs : ℕ := 4
  let cat_legs : ℕ := 4
  let bird_legs : ℕ := 2
  let spider_legs : ℕ := 8
  (num_dogs * dog_legs - dogs_missing_legs) +
  (num_cats * cat_legs - cats_missing_legs) +
  (num_birds * bird_legs) +
  (num_spiders * spider_legs - 2 * spiders_missing_legs)

theorem total_legs_in_park : total_legs = 829 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_in_park_l590_59011


namespace NUMINAMATH_CALUDE_quadratic_has_minimum_l590_59098

theorem quadratic_has_minimum (a b : ℝ) (ha : a > 0) :
  let g (x : ℝ) := a * x^2 + 2 * b * x + ((4 * b^2) / a - 3)
  ∃ (m : ℝ), ∀ (x : ℝ), g x ≥ m := by
  sorry

end NUMINAMATH_CALUDE_quadratic_has_minimum_l590_59098


namespace NUMINAMATH_CALUDE_product_first_three_eq_960_l590_59003

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  -- The seventh term is 20
  seventh_term : ℕ
  seventh_term_eq : seventh_term = 20
  -- The common difference is 2
  common_diff : ℕ
  common_diff_eq : common_diff = 2

/-- The product of the first three terms of the arithmetic sequence -/
def product_first_three (seq : ArithmeticSequence) : ℕ :=
  let a := seq.seventh_term - 6 * seq.common_diff -- First term
  let a2 := a + seq.common_diff -- Second term
  let a3 := a + 2 * seq.common_diff -- Third term
  a * a2 * a3

/-- Theorem stating that the product of the first three terms is 960 -/
theorem product_first_three_eq_960 (seq : ArithmeticSequence) :
  product_first_three seq = 960 := by
  sorry

end NUMINAMATH_CALUDE_product_first_three_eq_960_l590_59003


namespace NUMINAMATH_CALUDE_mail_order_cost_l590_59066

/-- The total cost of mail ordering books with a shipping fee -/
def total_cost (unit_price : ℝ) (shipping_rate : ℝ) (num_books : ℝ) : ℝ :=
  unit_price * num_books * (1 + shipping_rate)

/-- Theorem: The total cost of mail ordering 'a' books with a unit price of 8 yuan and a 10% shipping fee is 8(1+10%)a yuan -/
theorem mail_order_cost (a : ℝ) : 
  total_cost 8 0.1 a = 8 * (1 + 0.1) * a := by
  sorry

end NUMINAMATH_CALUDE_mail_order_cost_l590_59066


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l590_59063

-- Problem 1
theorem simplify_expression_1 (x : ℝ) :
  (2*x - 1) * (2*x - 3) - (1 - 2*x) * (2 - x) = 2*x^2 - 3*x + 1 := by sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) (ha : a ≠ 0) (ha' : a ≠ 1) :
  (a^2 - 1) / a * (1 - (2*a + 1) / (a^2 + 2*a + 1)) / (a - 1) = a / (a + 1) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l590_59063


namespace NUMINAMATH_CALUDE_exponent_and_logarithm_equalities_l590_59000

theorem exponent_and_logarithm_equalities :
  (3 : ℝ) ^ 64 = 4 ∧ (4 : ℝ) ^ (Real.log 3 / Real.log 2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_exponent_and_logarithm_equalities_l590_59000


namespace NUMINAMATH_CALUDE_equation_solution_exists_l590_59031

theorem equation_solution_exists : ∃ x : ℝ, -x^3 + 555^3 = x^2 - x * 555 + 555^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l590_59031


namespace NUMINAMATH_CALUDE_f_shifted_l590_59072

def f (x : ℝ) : ℝ := 2 * x + 1

theorem f_shifted (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 3) :
  ∃ y, f (y - 1) = 2 * y - 1 ∧ 2 ≤ y ∧ y ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_f_shifted_l590_59072


namespace NUMINAMATH_CALUDE_gcd_119_153_l590_59007

theorem gcd_119_153 : Nat.gcd 119 153 = 17 := by
  -- The proof would go here, but we'll use sorry as instructed
  sorry

end NUMINAMATH_CALUDE_gcd_119_153_l590_59007


namespace NUMINAMATH_CALUDE_valid_a_values_l590_59008

def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2 - a + 1}

theorem valid_a_values :
  ∀ a : ℝ, (A a ⊇ B a) → (a = -1 ∨ a = 2) := by
  sorry

end NUMINAMATH_CALUDE_valid_a_values_l590_59008


namespace NUMINAMATH_CALUDE_real_estate_investment_l590_59037

theorem real_estate_investment
  (total_investment : ℝ)
  (real_estate_ratio : ℝ)
  (h1 : total_investment = 250000)
  (h2 : real_estate_ratio = 3)
  : real_estate_ratio * (total_investment / (1 + real_estate_ratio)) = 187500 :=
by
  sorry

end NUMINAMATH_CALUDE_real_estate_investment_l590_59037


namespace NUMINAMATH_CALUDE_basketball_conference_games_l590_59029

/-- The number of teams in the basketball conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team in the conference -/
def games_per_pair : ℕ := 3

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 2

/-- The total number of games in a season for the basketball conference -/
def total_games : ℕ := (num_teams.choose 2 * games_per_pair) + (num_teams * non_conference_games)

theorem basketball_conference_games :
  total_games = 155 := by sorry

end NUMINAMATH_CALUDE_basketball_conference_games_l590_59029


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l590_59090

theorem smallest_common_multiple_of_8_and_6 : 
  ∃ (n : ℕ), n > 0 ∧ 8 ∣ n ∧ 6 ∣ n ∧ ∀ (m : ℕ), m > 0 ∧ 8 ∣ m ∧ 6 ∣ m → n ≤ m :=
by
  use 24
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l590_59090


namespace NUMINAMATH_CALUDE_two_real_roots_iff_m_nonpositive_m_values_given_roots_relationship_l590_59057

/-- Given a quadratic equation x^2 - 2x + m + 1 = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 - 2*x + m + 1 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  (-2)^2 - 4*(m + 1)

/-- The condition for two real roots -/
def has_two_real_roots (m : ℝ) : Prop :=
  discriminant m ≥ 0

/-- The relationship between the roots and m -/
def roots_relationship (x₁ x₂ m : ℝ) : Prop :=
  x₁ + 3*x₂ = 2*m + 8

/-- Theorem 1: The equation has two real roots iff m ≤ 0 -/
theorem two_real_roots_iff_m_nonpositive (m : ℝ) :
  has_two_real_roots m ↔ m ≤ 0 :=
sorry

/-- Theorem 2: If the roots satisfy the given relationship, then m = -1 or m = -2 -/
theorem m_values_given_roots_relationship (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic_equation x₁ m ∧ quadratic_equation x₂ m ∧ roots_relationship x₁ x₂ m) →
  (m = -1 ∨ m = -2) :=
sorry

end NUMINAMATH_CALUDE_two_real_roots_iff_m_nonpositive_m_values_given_roots_relationship_l590_59057


namespace NUMINAMATH_CALUDE_digit_sum_of_four_digit_number_divisible_by_109_l590_59047

def is_four_digit_nonzero (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ n % 10 ≠ 0

def reverse_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem digit_sum_of_four_digit_number_divisible_by_109 (A : ℕ) :
  is_four_digit_nonzero A →
  (A + reverse_digits A) % 109 = 0 →
  digit_sum A = 14 ∨ digit_sum A = 23 ∨ digit_sum A = 28 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_of_four_digit_number_divisible_by_109_l590_59047


namespace NUMINAMATH_CALUDE_hannah_stocking_stuffers_l590_59025

/-- The number of candy canes per stocking -/
def candy_canes_per_stocking : ℕ := 4

/-- The number of beanie babies per stocking -/
def beanie_babies_per_stocking : ℕ := 2

/-- The number of books per stocking -/
def books_per_stocking : ℕ := 1

/-- The number of kids Hannah has -/
def number_of_kids : ℕ := 3

/-- The total number of stocking stuffers Hannah buys -/
def total_stocking_stuffers : ℕ := 
  (candy_canes_per_stocking + beanie_babies_per_stocking + books_per_stocking) * number_of_kids

theorem hannah_stocking_stuffers : total_stocking_stuffers = 21 := by
  sorry

end NUMINAMATH_CALUDE_hannah_stocking_stuffers_l590_59025


namespace NUMINAMATH_CALUDE_luke_fillets_l590_59089

/-- Calculates the total number of fish fillets Luke has after fishing for a given number of days. -/
def total_fillets (fish_per_day : ℕ) (days : ℕ) (fillets_per_fish : ℕ) : ℕ :=
  fish_per_day * days * fillets_per_fish

/-- Proves that Luke has 120 fish fillets after fishing for 30 days. -/
theorem luke_fillets : total_fillets 2 30 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_luke_fillets_l590_59089


namespace NUMINAMATH_CALUDE_bob_cleaning_time_l590_59022

def alice_time : ℝ := 30

theorem bob_cleaning_time :
  let bob_time := (3 / 4 : ℝ) * alice_time
  bob_time = 22.5 := by sorry

end NUMINAMATH_CALUDE_bob_cleaning_time_l590_59022


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_99_l590_59002

theorem sum_of_powers_equals_99 :
  3^4 + (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_99_l590_59002


namespace NUMINAMATH_CALUDE_donut_selection_problem_l590_59085

theorem donut_selection_problem :
  let n : ℕ := 6  -- number of donuts to select
  let k : ℕ := 4  -- number of donut types
  Nat.choose (n + k - 1) (k - 1) = 84 :=
by sorry

end NUMINAMATH_CALUDE_donut_selection_problem_l590_59085


namespace NUMINAMATH_CALUDE_value_of_M_l590_59061

theorem value_of_M : ∃ M : ℝ, (0.25 * M = 0.35 * 1500) ∧ (M = 2100) := by sorry

end NUMINAMATH_CALUDE_value_of_M_l590_59061


namespace NUMINAMATH_CALUDE_rectangular_box_diagonal_sum_l590_59015

theorem rectangular_box_diagonal_sum (a b c : ℝ) 
  (h_surface_area : 2 * (a * b + b * c + c * a) = 112)
  (h_edge_sum : 4 * (a + b + c) = 60) :
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 113 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_diagonal_sum_l590_59015


namespace NUMINAMATH_CALUDE_light_ray_reflection_angle_l590_59054

/-- Regular hexagon with mirrored inner surface -/
structure RegularHexagon :=
  (side : ℝ)
  (A B C D E F : ℝ × ℝ)

/-- Light ray path in the hexagon -/
structure LightRayPath (hex : RegularHexagon) :=
  (M N : ℝ × ℝ)
  (start_at_A : M.1 = hex.A.1 ∨ M.2 = hex.A.2)
  (end_at_D : N.1 = hex.D.1 ∨ N.2 = hex.D.2)
  (on_sides : (M.1 = hex.A.1 ∨ M.1 = hex.B.1 ∨ M.2 = hex.A.2 ∨ M.2 = hex.B.2) ∧
              (N.1 = hex.B.1 ∨ N.1 = hex.C.1 ∨ N.2 = hex.B.2 ∨ N.2 = hex.C.2))

/-- Main theorem -/
theorem light_ray_reflection_angle (hex : RegularHexagon) (path : LightRayPath hex) :
  let tan_EAM := (hex.E.2 - hex.A.2) / (hex.E.1 - hex.A.1)
  tan_EAM = 1 / (3 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_light_ray_reflection_angle_l590_59054


namespace NUMINAMATH_CALUDE_yogurt_and_clothes_cost_l590_59048

/-- The total cost of buying a yogurt and a set of clothes -/
def total_cost (yogurt_price : ℕ) (clothes_price_multiplier : ℕ) : ℕ :=
  yogurt_price + yogurt_price * clothes_price_multiplier

/-- Theorem: The total cost of buying a yogurt priced at 120 yuan and a set of clothes
    priced at 6 times the yogurt's price is equal to 840 yuan. -/
theorem yogurt_and_clothes_cost :
  total_cost 120 6 = 840 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_and_clothes_cost_l590_59048


namespace NUMINAMATH_CALUDE_set_equality_proof_l590_59005

theorem set_equality_proof (A B : Set α) (h : A ∩ B = A) : A ∪ B = B := by
  sorry

end NUMINAMATH_CALUDE_set_equality_proof_l590_59005


namespace NUMINAMATH_CALUDE_pentagon_count_l590_59077

/-- The number of points on the circumference of the circle -/
def n : ℕ := 15

/-- The number of vertices in each pentagon -/
def k : ℕ := 5

/-- The number of different convex pentagons that can be formed -/
def num_pentagons : ℕ := n.choose k

theorem pentagon_count : num_pentagons = 3003 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_count_l590_59077


namespace NUMINAMATH_CALUDE_quadratic_root_implies_b_value_l590_59075

theorem quadratic_root_implies_b_value (b : ℝ) : 
  (Complex.I ^ 2 = -1) →
  ((3 : ℂ) + Complex.I) ^ 2 - 6 * ((3 : ℂ) + Complex.I) + b = 0 →
  b = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_b_value_l590_59075


namespace NUMINAMATH_CALUDE_equation_solution_l590_59021

theorem equation_solution : ∃ x : ℚ, (3 / 7 + 7 / x = 10 / x + 1 / 10) ∧ x = 210 / 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l590_59021


namespace NUMINAMATH_CALUDE_different_course_selections_eq_30_l590_59038

/-- The number of ways two people can choose 2 courses each from 4 courses with at least one course different -/
def different_course_selections : ℕ :=
  Nat.choose 4 2 * Nat.choose 2 2 + Nat.choose 4 1 * Nat.choose 3 1 * Nat.choose 2 1

/-- Theorem stating that the number of different course selections is 30 -/
theorem different_course_selections_eq_30 : different_course_selections = 30 := by
  sorry

end NUMINAMATH_CALUDE_different_course_selections_eq_30_l590_59038


namespace NUMINAMATH_CALUDE_derek_dogs_at_16_l590_59051

/-- Represents Derek's possessions at different ages --/
structure DereksPossessions where
  dogs_at_6 : ℕ
  cars_at_6 : ℕ
  dogs_at_16 : ℕ
  cars_at_16 : ℕ

/-- Theorem stating the number of dogs Derek has at age 16 --/
theorem derek_dogs_at_16 (d : DereksPossessions) 
  (h1 : d.dogs_at_6 = 3 * d.cars_at_6)
  (h2 : d.dogs_at_6 = 90)
  (h3 : d.cars_at_16 = d.cars_at_6 + 210)
  (h4 : d.cars_at_16 = 2 * d.dogs_at_16) :
  d.dogs_at_16 = 120 := by
  sorry

#check derek_dogs_at_16

end NUMINAMATH_CALUDE_derek_dogs_at_16_l590_59051


namespace NUMINAMATH_CALUDE_max_sections_five_lines_l590_59099

/-- The maximum number of sections a rectangle can be divided into by n line segments -/
def max_sections (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- Theorem: The maximum number of sections a rectangle can be divided into by 5 line segments is 16 -/
theorem max_sections_five_lines :
  max_sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sections_five_lines_l590_59099


namespace NUMINAMATH_CALUDE_isabel_albums_l590_59050

theorem isabel_albums (phone_pics camera_pics pics_per_album : ℕ) 
  (h1 : phone_pics = 2)
  (h2 : camera_pics = 4)
  (h3 : pics_per_album = 2)
  : (phone_pics + camera_pics) / pics_per_album = 3 := by
  sorry

end NUMINAMATH_CALUDE_isabel_albums_l590_59050


namespace NUMINAMATH_CALUDE_parking_spaces_available_l590_59024

theorem parking_spaces_available (front_spaces back_spaces parked_cars : ℕ) :
  front_spaces = 52 →
  back_spaces = 38 →
  parked_cars = 39 →
  (front_spaces + back_spaces) - (parked_cars + back_spaces / 2) = 32 := by
  sorry

end NUMINAMATH_CALUDE_parking_spaces_available_l590_59024


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l590_59027

/-- The distance between the foci of an ellipse given by the equation
    9x^2 - 36x + 4y^2 + 16y + 16 = 0 is 2√5 -/
theorem ellipse_foci_distance : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, 9*x^2 - 36*x + 4*y^2 + 16*y + 16 = 0 ↔ 
      (x - 2)^2 / a^2 + (y + 2)^2 / b^2 = 1) ∧
    a^2 - b^2 = c^2 ∧
    2 * c = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l590_59027


namespace NUMINAMATH_CALUDE_stock_price_return_l590_59016

theorem stock_price_return (original_price : ℝ) (h : original_price > 0) :
  let increased_price := original_price * 1.3
  let decrease_rate := 1 - 1 / 1.3
  increased_price * (1 - decrease_rate) = original_price :=
by
  sorry

#eval (1 - 1 / 1.3) * 100 -- This will output approximately 23.08

end NUMINAMATH_CALUDE_stock_price_return_l590_59016


namespace NUMINAMATH_CALUDE_parabola_directrix_l590_59030

/-- Given a parabola y² = 2px where p > 0, if a point M(1, m) on the parabola
    is at a distance of 5 from the focus, then the directrix is x = -4 -/
theorem parabola_directrix (p : ℝ) (m : ℝ) (h1 : p > 0) (h2 : m^2 = 2*p) 
  (h3 : (1 - p/2)^2 + m^2 = 5^2) : 
  ∃ (x : ℝ), x = -4 ∧ ∀ (y : ℝ), (x + p/2)^2 = (1 - x)^2 + m^2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l590_59030


namespace NUMINAMATH_CALUDE_odd_integer_divisibility_l590_59012

theorem odd_integer_divisibility (n : Int) (h : Odd n) :
  ∀ k : Nat, 2 ∣ k * (n - k) :=
sorry

end NUMINAMATH_CALUDE_odd_integer_divisibility_l590_59012


namespace NUMINAMATH_CALUDE_remaining_money_l590_59062

def initial_amount : ℚ := 3
def purchase_amount : ℚ := 1

theorem remaining_money :
  initial_amount - purchase_amount = 2 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l590_59062


namespace NUMINAMATH_CALUDE_simple_interest_problem_l590_59013

/-- Given a principal amount P and an unknown interest rate R,
    if increasing the rate by 1% results in Rs. 72 more interest over 3 years,
    then P must be Rs. 2400. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 1) * 3) / 100 - (P * R * 3) / 100 = 72 → P = 2400 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l590_59013


namespace NUMINAMATH_CALUDE_water_flow_difference_l590_59091

/-- Given a water flow restrictor problem, prove the difference between 0.6 times
    the original flow rate and the reduced flow rate. -/
theorem water_flow_difference (original_rate reduced_rate : ℝ) 
    (h1 : original_rate = 5)
    (h2 : reduced_rate = 2) :
  0.6 * original_rate - reduced_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_water_flow_difference_l590_59091


namespace NUMINAMATH_CALUDE_derivative_x_exp_x_l590_59036

theorem derivative_x_exp_x (x : ℝ) : deriv (fun x => x * Real.exp x) x = (1 + x) * Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_exp_x_l590_59036


namespace NUMINAMATH_CALUDE_inequality_proof_l590_59040

theorem inequality_proof (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ Real.exp x * (x^2 - x + 1) * (a*x + 3*a - 1) < 1) → 
  a < 2/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l590_59040


namespace NUMINAMATH_CALUDE_expression_result_l590_59083

theorem expression_result : (3.242 * 16) / 100 = 0.51872 := by
  sorry

end NUMINAMATH_CALUDE_expression_result_l590_59083


namespace NUMINAMATH_CALUDE_M_reflected_y_axis_l590_59034

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), p.2)

/-- The coordinates of point M -/
def M : ℝ × ℝ := (1, 2)

theorem M_reflected_y_axis :
  reflect_y_axis M = (-1, 2) := by sorry

end NUMINAMATH_CALUDE_M_reflected_y_axis_l590_59034


namespace NUMINAMATH_CALUDE_apple_pear_equivalence_l590_59064

/-- Given that 3/4 of 12 apples are worth as much as 6 pears,
    prove that 1/3 of 9 apples are worth as much as 2 pears. -/
theorem apple_pear_equivalence (apple pear : ℝ) 
    (h : (3/4 : ℝ) * 12 * apple = 6 * pear) : 
    (1/3 : ℝ) * 9 * apple = 2 * pear := by
  sorry

end NUMINAMATH_CALUDE_apple_pear_equivalence_l590_59064


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l590_59088

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 2 + 35 / 99) ∧ (x = 233 / 99) := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l590_59088


namespace NUMINAMATH_CALUDE_truncated_hexahedron_property_l590_59070

-- Define the structure of our polyhedron
structure Polyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  H : ℕ  -- number of hexagonal faces
  T : ℕ  -- number of triangular faces

-- Define the properties of our specific polyhedron
def truncated_hexahedron : Polyhedron where
  V := 20
  E := 36
  F := 18
  H := 6
  T := 12

-- Theorem statement
theorem truncated_hexahedron_property (p : Polyhedron) 
  (euler : p.V - p.E + p.F = 2)
  (faces : p.F = 18)
  (hex_tri : p.H + p.T = p.F)
  (vertex_config : 2 * p.V = 3 * p.T + 6 * p.H) :
  100 * 2 + 10 * 2 + p.V = 240 := by
  sorry

#check truncated_hexahedron_property

end NUMINAMATH_CALUDE_truncated_hexahedron_property_l590_59070


namespace NUMINAMATH_CALUDE_order_of_abc_l590_59026

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem order_of_abc : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l590_59026


namespace NUMINAMATH_CALUDE_max_cookies_eaten_l590_59041

theorem max_cookies_eaten (total : ℕ) (andy bella : ℕ) : 
  total = 30 →
  bella = 2 * andy →
  total = andy + bella →
  andy ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_cookies_eaten_l590_59041


namespace NUMINAMATH_CALUDE_expression_properties_l590_59067

def expression_result (signs : List Bool) : Int :=
  let nums := List.range 9
  List.foldl (λ acc (n, sign) => if sign then acc + (n + 1) else acc - (n + 1)) 0 (List.zip nums signs)

theorem expression_properties :
  (∀ signs : List Bool, expression_result signs ≠ 0) ∧
  (∃ signs : List Bool, expression_result signs = 1) ∧
  (∀ n : Int, (n % 2 = 1 ∧ -45 ≤ n ∧ n ≤ 45) ↔ ∃ signs : List Bool, expression_result signs = n) := by
  sorry

end NUMINAMATH_CALUDE_expression_properties_l590_59067
