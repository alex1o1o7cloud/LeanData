import Mathlib

namespace NUMINAMATH_CALUDE_product_expansion_l150_15080

theorem product_expansion (x : ℝ) : 
  (x^2 - 3*x + 3) * (x^2 + x + 1) = x^4 - 2*x^3 + x^2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l150_15080


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l150_15075

-- Define the function f(x) = x^3 - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the closed interval [-3, 0]
def I : Set ℝ := Set.Icc (-3) 0

-- Statement of the theorem
theorem f_max_min_on_interval :
  (∃ x ∈ I, ∀ y ∈ I, f y ≤ f x) ∧
  (∃ x ∈ I, ∀ y ∈ I, f x ≤ f y) ∧
  (∃ x ∈ I, f x = 3) ∧
  (∃ x ∈ I, f x = -17) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l150_15075


namespace NUMINAMATH_CALUDE_girls_count_l150_15081

theorem girls_count (total : ℕ) (difference : ℕ) (girls : ℕ) : 
  total = 600 → 
  difference = 30 → 
  girls + (girls - difference) = total → 
  girls = 315 := by
sorry

end NUMINAMATH_CALUDE_girls_count_l150_15081


namespace NUMINAMATH_CALUDE_last_digit_of_special_number_l150_15006

/-- A function that returns the last element of a list -/
def lastDigit (digits : List Nat) : Nat :=
  match digits.reverse with
  | [] => 0  -- Default value for empty list
  | d :: _ => d

/-- Check if a two-digit number is divisible by 13 -/
def isDivisibleBy13 (n : Nat) : Prop :=
  n % 13 = 0

theorem last_digit_of_special_number :
  ∀ (digits : List Nat),
    digits.length = 2019 →
    digits.head? = some 6 →
    (∀ i, i < digits.length - 1 →
      isDivisibleBy13 (digits[i]! * 10 + digits[i+1]!)) →
    lastDigit digits = 2 := by
  sorry

#check last_digit_of_special_number

end NUMINAMATH_CALUDE_last_digit_of_special_number_l150_15006


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l150_15032

theorem reciprocal_of_negative_three :
  ∃ x : ℚ, x * (-3) = 1 ∧ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l150_15032


namespace NUMINAMATH_CALUDE_systematic_sampling_sum_l150_15049

/-- Systematic sampling function -/
def systematicSample (n : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => (start + i * (n / sampleSize)) % n + 1)

theorem systematic_sampling_sum (n : ℕ) (sampleSize : ℕ) (start : ℕ) :
  n = 50 →
  sampleSize = 5 →
  start ≤ n →
  systematicSample n sampleSize start = [4, a, 24, b, 44] →
  a + b = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_sum_l150_15049


namespace NUMINAMATH_CALUDE_tank_inflow_rate_l150_15093

theorem tank_inflow_rate (capacity : ℝ) (time_diff : ℝ) (slow_rate : ℝ) : 
  capacity > 0 → time_diff > 0 → slow_rate > 0 →
  let slow_time := capacity / slow_rate
  let fast_time := slow_time - time_diff
  fast_time > 0 →
  capacity / fast_time = 2 * slow_rate := by
  sorry

-- Example usage with given values
example : 
  let capacity := 20
  let time_diff := 5
  let slow_rate := 2
  let slow_time := capacity / slow_rate
  let fast_time := slow_time - time_diff
  capacity / fast_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_tank_inflow_rate_l150_15093


namespace NUMINAMATH_CALUDE_cold_virus_diameter_scientific_notation_l150_15002

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem cold_virus_diameter_scientific_notation :
  to_scientific_notation 0.00000036 = ScientificNotation.mk 3.6 (-7) sorry := by
  sorry

end NUMINAMATH_CALUDE_cold_virus_diameter_scientific_notation_l150_15002


namespace NUMINAMATH_CALUDE_product_B_percentage_l150_15056

theorem product_B_percentage (X : ℝ) : 
  X ≥ 0 → X ≤ 100 →
  ∃ (total : ℕ), total ≥ 100 ∧
  ∃ (A B both neither : ℕ),
    A + B + neither = total ∧
    both ≤ A ∧ both ≤ B ∧
    (X : ℝ) = (A : ℝ) / total * 100 ∧
    (23 : ℝ) = (both : ℝ) / total * 100 ∧
    (23 : ℝ) = (neither : ℝ) / total * 100 →
  (B : ℝ) / total * 100 = 100 - X :=
by sorry

end NUMINAMATH_CALUDE_product_B_percentage_l150_15056


namespace NUMINAMATH_CALUDE_computer_table_price_l150_15038

/-- The selling price of an item given its cost price and markup percentage -/
def sellingPrice (costPrice : ℚ) (markupPercentage : ℚ) : ℚ :=
  costPrice * (1 + markupPercentage / 100)

/-- Theorem: The selling price of a computer table with cost price 3840 and markup 25% is 4800 -/
theorem computer_table_price : sellingPrice 3840 25 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_price_l150_15038


namespace NUMINAMATH_CALUDE_interest_rate_proof_l150_15001

/-- Given a principal sum P, if the simple interest on P for 4 years is one-fifth of P,
    then the rate of interest per annum is 25%. -/
theorem interest_rate_proof (P : ℝ) (P_pos : P > 0) : 
  (P * 25 * 4) / 100 = P / 5 → 25 = 100 * (P / 5) / (P * 4) := by
sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l150_15001


namespace NUMINAMATH_CALUDE_bus_ride_cost_l150_15026

theorem bus_ride_cost (train_cost bus_cost : ℝ) : 
  train_cost = bus_cost + 6.85 →
  (train_cost * 0.85 + (bus_cost + 1.25)) = 10.50 →
  bus_cost = 1.85 := by
  sorry

end NUMINAMATH_CALUDE_bus_ride_cost_l150_15026


namespace NUMINAMATH_CALUDE_isabel_homework_problem_l150_15052

/-- The total number of homework problems Isabel had -/
def total_problems (finished : ℕ) (pages_left : ℕ) (problems_per_page : ℕ) : ℕ :=
  finished + pages_left * problems_per_page

/-- Theorem stating that Isabel had 72 homework problems in total -/
theorem isabel_homework_problem :
  total_problems 32 5 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_isabel_homework_problem_l150_15052


namespace NUMINAMATH_CALUDE_james_living_room_cost_l150_15073

def couch_price : ℝ := 2500
def sectional_price : ℝ := 3500
def entertainment_center_price : ℝ := 1500
def rug_price : ℝ := 800
def coffee_table_price : ℝ := 700
def accessories_price : ℝ := 500

def couch_discount : ℝ := 0.10
def sectional_discount : ℝ := 0.10
def entertainment_center_discount : ℝ := 0.05
def rug_discount : ℝ := 0.05
def coffee_table_discount : ℝ := 0.12
def accessories_discount : ℝ := 0.15

def sales_tax_rate : ℝ := 0.0825
def service_fee : ℝ := 250

def total_cost : ℝ := 9587.65

theorem james_living_room_cost : 
  (couch_price * (1 - couch_discount) + 
   sectional_price * (1 - sectional_discount) + 
   entertainment_center_price * (1 - entertainment_center_discount) + 
   rug_price * (1 - rug_discount) + 
   coffee_table_price * (1 - coffee_table_discount) + 
   accessories_price * (1 - accessories_discount)) * 
  (1 + sales_tax_rate) + service_fee = total_cost := by
  sorry

end NUMINAMATH_CALUDE_james_living_room_cost_l150_15073


namespace NUMINAMATH_CALUDE_wall_bricks_count_l150_15019

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 192

/-- Represents Beth's individual rate in bricks per hour -/
def beth_rate : ℚ := total_bricks / 8

/-- Represents Ben's individual rate in bricks per hour -/
def ben_rate : ℚ := total_bricks / 12

/-- Represents the reduction in combined output due to chatting, in bricks per hour -/
def chat_reduction : ℕ := 8

/-- Represents the time taken to complete the wall when working together, in hours -/
def time_together : ℕ := 6

theorem wall_bricks_count :
  (beth_rate + ben_rate - chat_reduction) * time_together = total_bricks := by
  sorry

#check wall_bricks_count

end NUMINAMATH_CALUDE_wall_bricks_count_l150_15019


namespace NUMINAMATH_CALUDE_w_over_y_value_l150_15014

theorem w_over_y_value (w x y : ℝ) 
  (h1 : w / x = 1 / 3) 
  (h2 : (x + y) / y = 3.25) : 
  w / y = 0.75 := by
sorry

end NUMINAMATH_CALUDE_w_over_y_value_l150_15014


namespace NUMINAMATH_CALUDE_impossible_average_l150_15033

theorem impossible_average (test1 test2 test3 test4 test5 test6 : ℕ) 
  (h1 : test1 = 85)
  (h2 : test2 = 79)
  (h3 : test3 = 92)
  (h4 : test4 = 84)
  (h5 : test5 = 88)
  (h6 : test6 = 7)
  : ¬ ∃ (test7 test8 : ℕ), (test1 + test2 + test3 + test4 + test5 + test6 + test7 + test8) / 8 = 87 :=
sorry

end NUMINAMATH_CALUDE_impossible_average_l150_15033


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l150_15025

theorem largest_triangle_perimeter :
  ∀ x : ℤ,
  (8 : ℝ) + 11 > (x : ℝ) →
  (8 : ℝ) + (x : ℝ) > 11 →
  (11 : ℝ) + (x : ℝ) > 8 →
  (8 : ℝ) + 11 + (x : ℝ) ≤ 37 :=
by sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l150_15025


namespace NUMINAMATH_CALUDE_x_over_y_is_negative_two_l150_15092

theorem x_over_y_is_negative_two (x y : ℝ) 
  (h1 : 1 < (x - y) / (x + y) ∧ (x - y) / (x + y) < 4)
  (h2 : (x + y) / (x - y) ≠ 1)
  (h3 : ∃ (n : ℤ), x / y = n) : 
  x / y = -2 := by
sorry

end NUMINAMATH_CALUDE_x_over_y_is_negative_two_l150_15092


namespace NUMINAMATH_CALUDE_exists_min_n_all_rows_shaded_l150_15011

/-- Calculates the square number of the nth shaded square -/
def shadedSquareNumber (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Calculates the row number for a given square number -/
def squareToRow (square : ℕ) : ℕ :=
  (square - 1) / 5 + 1

/-- Checks if all rows are shaded up to the nth shaded square -/
def allRowsShaded (n : ℕ) : Prop :=
  ∀ row : ℕ, row ≤ 10 → ∃ k : ℕ, k ≤ n ∧ squareToRow (shadedSquareNumber k) = row

/-- The main theorem stating the existence of a minimum n that shades all rows -/
theorem exists_min_n_all_rows_shaded :
  ∃ n : ℕ, allRowsShaded n ∧ ∀ m : ℕ, m < n → ¬allRowsShaded m :=
sorry

end NUMINAMATH_CALUDE_exists_min_n_all_rows_shaded_l150_15011


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l150_15027

/-- Converts a list of binary digits to a natural number -/
def binaryToNat (digits : List Bool) : Nat :=
  digits.foldl (fun acc d => 2 * acc + if d then 1 else 0) 0

/-- Converts a natural number to a list of binary digits -/
def natToBinary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
  toBinary n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [false, true, true]        -- 110₂
  let expected := [false, true, true, true, true, false, true]  -- 1011110₂
  binaryToNat a * binaryToNat b = binaryToNat expected := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l150_15027


namespace NUMINAMATH_CALUDE_base12_remainder_theorem_l150_15098

/-- Converts a base-12 integer to decimal --/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

/-- The base-12 representation of 2543₁₂ --/
def base12Number : List Nat := [2, 5, 4, 3]

/-- The theorem stating that the remainder of 2543₁₂ divided by 9 is 8 --/
theorem base12_remainder_theorem :
  (base12ToDecimal base12Number) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_base12_remainder_theorem_l150_15098


namespace NUMINAMATH_CALUDE_power_of_two_with_ones_and_twos_l150_15085

theorem power_of_two_with_ones_and_twos (N : ℕ) : 
  ∃ k : ℕ, ∃ m : ℕ, 2^k ≡ m [ZMOD 10^N] ∧ 
  ∀ d : ℕ, d < N → (m / 10^d % 10 = 1 ∨ m / 10^d % 10 = 2) :=
sorry

end NUMINAMATH_CALUDE_power_of_two_with_ones_and_twos_l150_15085


namespace NUMINAMATH_CALUDE_min_transport_time_l150_15009

/-- The minimum time required for transporting goods between two cities --/
theorem min_transport_time (distance : ℝ) (num_trains : ℕ) (speed : ℝ) 
  (h1 : distance = 400)
  (h2 : num_trains = 17)
  (h3 : speed > 0) :
  (distance / speed + (num_trains - 1) * (speed / 20)^2 / speed) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_transport_time_l150_15009


namespace NUMINAMATH_CALUDE_line_plane_relationships_l150_15050

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem line_plane_relationships 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  ((perpendicular_line_plane m α ∧ 
    parallel_line_plane n β ∧ 
    parallel_planes α β) → 
   perpendicular_lines m n) ∧
  ((perpendicular_line_plane m α ∧ 
    perpendicular_line_plane n β ∧ 
    perpendicular_planes α β) → 
   perpendicular_lines m n) :=
by sorry

end NUMINAMATH_CALUDE_line_plane_relationships_l150_15050


namespace NUMINAMATH_CALUDE_triangle_tangent_product_l150_15045

theorem triangle_tangent_product (A B C : Real) (h1 : C = 2 * Real.pi / 3) 
  (h2 : Real.tan A + Real.tan B = 2 * Real.sqrt 3 / 3) : 
  Real.tan A * Real.tan B = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_product_l150_15045


namespace NUMINAMATH_CALUDE_square_inequality_condition_l150_15048

theorem square_inequality_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_square_inequality_condition_l150_15048


namespace NUMINAMATH_CALUDE_xy_sum_l150_15086

theorem xy_sum (x y : ℕ) (hx : x < 15) (hy : y < 25) (hxy : x + y + x * y = 119) :
  x + y = 20 ∨ x + y = 21 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_l150_15086


namespace NUMINAMATH_CALUDE_trapezoid_area_theorem_l150_15042

/-- Represents a trapezoid with given diagonals and height -/
structure Trapezoid where
  diagonal1 : ℝ
  diagonal2 : ℝ
  height : ℝ

/-- Calculates the possible areas of a trapezoid given its diagonals and height -/
def trapezoid_areas (t : Trapezoid) : Set ℝ :=
  {900, 780}

/-- Theorem stating that a trapezoid with diagonals 17 and 113, and height 15 has an area of either 900 or 780 -/
theorem trapezoid_area_theorem (t : Trapezoid) 
    (h1 : t.diagonal1 = 17) 
    (h2 : t.diagonal2 = 113) 
    (h3 : t.height = 15) : 
  ∃ (area : ℝ), area ∈ trapezoid_areas t ∧ (area = 900 ∨ area = 780) := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_area_theorem_l150_15042


namespace NUMINAMATH_CALUDE_f_and_g_are_even_and_increasing_l150_15037

-- Define the functions
def f (x : ℝ) : ℝ := |2 * x|
def g (x : ℝ) : ℝ := 2 * x^2 + 3

-- Define evenness
def is_even (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)

-- Define monotonically increasing on an interval
def is_monotone_increasing_on (h : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → h x ≤ h y

-- Theorem statement
theorem f_and_g_are_even_and_increasing :
  (is_even f ∧ is_monotone_increasing_on f 0 1) ∧
  (is_even g ∧ is_monotone_increasing_on g 0 1) :=
sorry

end NUMINAMATH_CALUDE_f_and_g_are_even_and_increasing_l150_15037


namespace NUMINAMATH_CALUDE_car_value_reduction_l150_15040

theorem car_value_reduction (original_price current_value : ℝ) : 
  current_value = 0.7 * original_price → 
  current_value = 2800 → 
  original_price = 4000 := by
sorry

end NUMINAMATH_CALUDE_car_value_reduction_l150_15040


namespace NUMINAMATH_CALUDE_son_age_proof_l150_15079

theorem son_age_proof (father_age son_age : ℝ) : 
  father_age = son_age + 35 →
  father_age + 5 = 3 * (son_age + 5) →
  son_age = 12.5 := by
sorry

end NUMINAMATH_CALUDE_son_age_proof_l150_15079


namespace NUMINAMATH_CALUDE_fundraiser_result_l150_15083

def fundraiser (num_students : ℕ) (initial_needed : ℕ) (additional_needed : ℕ) 
               (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) (num_half_days : ℕ) : ℕ :=
  let total_per_student := initial_needed + additional_needed
  let total_needed := num_students * total_per_student
  let first_three_days := day1 + day2 + day3
  let half_day_amount := first_three_days / 2
  let total_raised := first_three_days + num_half_days * half_day_amount
  total_raised - total_needed

theorem fundraiser_result : 
  fundraiser 6 450 475 600 900 400 4 = 150 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_result_l150_15083


namespace NUMINAMATH_CALUDE_bread_cost_l150_15015

/-- Prove that the cost of the bread is $1.25 given the conditions --/
theorem bread_cost (total_cost change_nickels : ℚ) 
  (h1 : total_cost = 205/100)  -- Total cost is $2.05
  (h2 : change_nickels = 8 * 5/100)  -- 8 nickels in change
  (h3 : ∃ (change_quarter change_dime : ℚ), 
    change_quarter = 25/100 ∧ 
    change_dime = 10/100 ∧ 
    700/100 - total_cost = change_quarter + change_dime + change_nickels + 420/100) 
  : ∃ (bread_cost cheese_cost : ℚ), 
    bread_cost = 125/100 ∧ 
    cheese_cost = 80/100 ∧ 
    bread_cost + cheese_cost = total_cost := by
  sorry

end NUMINAMATH_CALUDE_bread_cost_l150_15015


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l150_15008

theorem marble_fraction_after_tripling (total : ℚ) (h : total > 0) :
  let initial_blue : ℚ := (4 / 7) * total
  let initial_red : ℚ := total - initial_blue
  let new_red : ℚ := 3 * initial_red
  let new_total : ℚ := initial_blue + new_red
  new_red / new_total = 9 / 13 :=
by sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l150_15008


namespace NUMINAMATH_CALUDE_coloring_arrangements_l150_15077

/-- The number of ways to arrange n distinct objects into n distinct positions -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of parts to be colored -/
def num_parts : ℕ := 4

/-- The number of colors available -/
def num_colors : ℕ := 4

/-- Theorem: The number of ways to color 4 distinct parts with 4 distinct colors, 
    where each part must have a different color, is equal to 24 -/
theorem coloring_arrangements : permutations num_parts = 24 := by
  sorry

end NUMINAMATH_CALUDE_coloring_arrangements_l150_15077


namespace NUMINAMATH_CALUDE_cards_added_l150_15091

theorem cards_added (initial_cards final_cards : ℕ) 
  (h1 : initial_cards = 9) 
  (h2 : final_cards = 13) : 
  final_cards - initial_cards = 4 := by
  sorry

end NUMINAMATH_CALUDE_cards_added_l150_15091


namespace NUMINAMATH_CALUDE_inequality_proof_l150_15058

theorem inequality_proof (a b : ℝ) (h : a < b ∧ b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l150_15058


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l150_15060

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ (623 * n) % 32 = (1319 * n) % 32 ∧ ∀ (m : ℕ), m > 0 → m < n → (623 * m) % 32 ≠ (1319 * m) % 32 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l150_15060


namespace NUMINAMATH_CALUDE_surface_area_ratio_l150_15090

/-- A regular tetrahedron with its inscribed sphere -/
structure RegularTetrahedronWithInscribedSphere where
  /-- The surface area of the regular tetrahedron -/
  S₁ : ℝ
  /-- The surface area of the inscribed sphere -/
  S₂ : ℝ
  /-- The surface area of the tetrahedron is positive -/
  h_S₁_pos : 0 < S₁
  /-- The surface area of the sphere is positive -/
  h_S₂_pos : 0 < S₂

/-- The ratio of the surface area of a regular tetrahedron to its inscribed sphere -/
theorem surface_area_ratio (t : RegularTetrahedronWithInscribedSphere) :
  t.S₁ / t.S₂ = 6 * Real.sqrt 3 / Real.pi := by sorry

end NUMINAMATH_CALUDE_surface_area_ratio_l150_15090


namespace NUMINAMATH_CALUDE_investment_difference_l150_15084

def compound_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (1 + rate)

def emma_investment (initial : ℝ) (rate1 rate2 rate3 : ℝ) : ℝ :=
  compound_interest (compound_interest (compound_interest initial rate1) rate2) rate3

def briana_investment (initial : ℝ) (rate1 rate2 rate3 : ℝ) : ℝ :=
  compound_interest (compound_interest (compound_interest initial rate1) rate2) rate3

theorem investment_difference :
  let emma_initial := 300
  let briana_initial := 500
  let emma_rate1 := 0.15
  let emma_rate2 := 0.12
  let emma_rate3 := 0.18
  let briana_rate1 := 0.10
  let briana_rate2 := 0.08
  let briana_rate3 := 0.14
  briana_investment briana_initial briana_rate1 briana_rate2 briana_rate3 -
  emma_investment emma_initial emma_rate1 emma_rate2 emma_rate3 = 220.808 := by
  sorry

end NUMINAMATH_CALUDE_investment_difference_l150_15084


namespace NUMINAMATH_CALUDE_birch_trees_not_adjacent_probability_l150_15044

def total_trees : ℕ := 17
def birch_trees : ℕ := 6
def non_birch_trees : ℕ := total_trees - birch_trees

theorem birch_trees_not_adjacent_probability : 
  (Nat.choose (non_birch_trees + 1) birch_trees) / (Nat.choose total_trees birch_trees) = 77 / 1033 := by
  sorry

end NUMINAMATH_CALUDE_birch_trees_not_adjacent_probability_l150_15044


namespace NUMINAMATH_CALUDE_sum_smallest_largest_consecutive_odds_l150_15018

/-- Given an even number of consecutive odd integers with arithmetic mean y + 1,
    the sum of the smallest and largest integers is 2y. -/
theorem sum_smallest_largest_consecutive_odds (y : ℝ) (n : ℕ) (h : n > 0) :
  let a := y - 2 * n + 2
  let sequence := fun i => a + 2 * i
  let mean := (sequence 0 + sequence (2 * n - 1)) / 2
  (mean = y + 1) → (sequence 0 + sequence (2 * n - 1) = 2 * y) :=
by sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_consecutive_odds_l150_15018


namespace NUMINAMATH_CALUDE_pentagon_hexagon_side_difference_l150_15028

theorem pentagon_hexagon_side_difference (e : ℕ) : 
  (∃ (p h : ℝ), 5 * p - 6 * h = 1240 ∧ p - h = e ∧ 5 * p > 0 ∧ 6 * h > 0) ↔ e > 248 :=
sorry

end NUMINAMATH_CALUDE_pentagon_hexagon_side_difference_l150_15028


namespace NUMINAMATH_CALUDE_domain_of_g_l150_15069

/-- The domain of f(x) -/
def DomainF : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 6}

/-- The function g(x) -/
noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2 * x) / (x - 2)

/-- The domain of g(x) -/
def DomainG : Set ℝ := {x : ℝ | (0 ≤ x ∧ x < 2) ∨ (2 < x ∧ x ≤ 3)}

/-- Theorem: The domain of g(x) is correct given the domain of f(x) -/
theorem domain_of_g (f : ℝ → ℝ) (hf : ∀ x, x ∈ DomainF → f x ≠ 0) :
  ∀ x, x ∈ DomainG ↔ (2 * x ∈ DomainF ∧ x ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l150_15069


namespace NUMINAMATH_CALUDE_correct_placement_l150_15067

/-- Represents the participants in the competition -/
inductive Participant
| Olya
| Oleg
| Polya
| Pasha

/-- Represents the possible placements in the competition -/
inductive Place
| First
| Second
| Third
| Fourth

/-- Represents whether a participant is a boy or a girl -/
inductive Gender
| Boy
| Girl

/-- Defines the gender of each participant -/
def participantGender (p : Participant) : Gender :=
  match p with
  | Participant.Olya => Gender.Girl
  | Participant.Oleg => Gender.Boy
  | Participant.Polya => Gender.Girl
  | Participant.Pasha => Gender.Boy

/-- Defines whether a participant's name starts with 'O' -/
def nameStartsWithO (p : Participant) : Prop :=
  match p with
  | Participant.Olya => true
  | Participant.Oleg => true
  | Participant.Polya => false
  | Participant.Pasha => false

/-- Defines whether a place is odd-numbered -/
def isOddPlace (p : Place) : Prop :=
  match p with
  | Place.First => true
  | Place.Second => false
  | Place.Third => true
  | Place.Fourth => false

/-- Defines whether two places are consecutive -/
def areConsecutivePlaces (p1 p2 : Place) : Prop :=
  (p1 = Place.First ∧ p2 = Place.Second) ∨
  (p1 = Place.Second ∧ p2 = Place.Third) ∨
  (p1 = Place.Third ∧ p2 = Place.Fourth) ∨
  (p2 = Place.First ∧ p1 = Place.Second) ∨
  (p2 = Place.Second ∧ p1 = Place.Third) ∨
  (p2 = Place.Third ∧ p1 = Place.Fourth)

/-- Represents the final placement of participants -/
def Placement := Participant → Place

/-- Theorem stating the correct placement given the conditions -/
theorem correct_placement (placement : Placement) : 
  (∃! p : Participant, placement p = Place.First) ∧
  (∃! p : Participant, placement p = Place.Second) ∧
  (∃! p : Participant, placement p = Place.Third) ∧
  (∃! p : Participant, placement p = Place.Fourth) ∧
  (∃! p : Participant, (placement p = Place.First → 
    (∀ p' : Place, isOddPlace p' → ∃ p'' : Participant, placement p'' = p' ∧ participantGender p'' = Gender.Boy) ∧
    (areConsecutivePlaces (placement Participant.Oleg) (placement Participant.Olya)) ∧
    (∀ p' : Place, isOddPlace p' → ∃ p'' : Participant, placement p'' = p' ∧ nameStartsWithO p''))) →
  placement Participant.Oleg = Place.First ∧
  placement Participant.Olya = Place.Second ∧
  placement Participant.Polya = Place.Third ∧
  placement Participant.Pasha = Place.Fourth :=
by sorry

end NUMINAMATH_CALUDE_correct_placement_l150_15067


namespace NUMINAMATH_CALUDE_bobby_candy_consumption_l150_15094

theorem bobby_candy_consumption (initial : ℕ) (additional : ℕ) : 
  initial = 26 → additional = 17 → initial + additional = 43 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_consumption_l150_15094


namespace NUMINAMATH_CALUDE_difference_of_squares_l150_15024

theorem difference_of_squares (m n : ℝ) : (3*m + n) * (3*m - n) = (3*m)^2 - n^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l150_15024


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l150_15097

theorem gcd_of_three_numbers : Nat.gcd 9240 (Nat.gcd 12240 33720) = 240 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l150_15097


namespace NUMINAMATH_CALUDE_new_average_is_75_l150_15054

/-- Calculates the new average daily production after adding a new day's production. -/
def new_average_production (past_days : ℕ) (past_average : ℚ) (today_production : ℕ) : ℚ :=
  (past_average * past_days + today_production) / (past_days + 1)

/-- Theorem stating that given the conditions, the new average daily production is 75 units. -/
theorem new_average_is_75 :
  let past_days : ℕ := 3
  let past_average : ℚ := 70
  let today_production : ℕ := 90
  new_average_production past_days past_average today_production = 75 := by
sorry

end NUMINAMATH_CALUDE_new_average_is_75_l150_15054


namespace NUMINAMATH_CALUDE_not_right_triangle_1_5_2_3_l150_15095

/-- A function that checks if three numbers can form the sides of a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

/-- Theorem stating that 1.5, 2, and 3 cannot form the sides of a right triangle -/
theorem not_right_triangle_1_5_2_3 : ¬ is_right_triangle 1.5 2 3 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_1_5_2_3_l150_15095


namespace NUMINAMATH_CALUDE_mathematics_players_count_l150_15010

def total_players : ℕ := 15
def physics_players : ℕ := 9
def both_subjects : ℕ := 3

theorem mathematics_players_count :
  ∃ (math_players : ℕ),
    math_players = total_players - physics_players + both_subjects ∧
    math_players = 9 :=
by sorry

end NUMINAMATH_CALUDE_mathematics_players_count_l150_15010


namespace NUMINAMATH_CALUDE_mitchell_gum_chewing_l150_15004

theorem mitchell_gum_chewing (packets : ℕ) (pieces_per_packet : ℕ) (unchewed_pieces : ℕ) :
  packets = 8 →
  pieces_per_packet = 7 →
  unchewed_pieces = 2 →
  packets * pieces_per_packet - unchewed_pieces = 54 :=
by sorry

end NUMINAMATH_CALUDE_mitchell_gum_chewing_l150_15004


namespace NUMINAMATH_CALUDE_rias_initial_savings_l150_15072

theorem rias_initial_savings (r f : ℚ) : 
  r / f = 5 / 3 →  -- Initial ratio
  (r - 160) / f = 3 / 5 →  -- New ratio after withdrawal
  r = 250 := by
sorry

end NUMINAMATH_CALUDE_rias_initial_savings_l150_15072


namespace NUMINAMATH_CALUDE_go_stones_problem_l150_15076

theorem go_stones_problem (total : ℕ) (difference_result : ℕ) 
  (h_total : total = 6000)
  (h_difference : difference_result = 4800) :
  ∃ (white black : ℕ), 
    white + black = total ∧ 
    white > black ∧ 
    total - (white - black) = difference_result ∧
    white = 3600 := by
  sorry

end NUMINAMATH_CALUDE_go_stones_problem_l150_15076


namespace NUMINAMATH_CALUDE_symmetric_complex_division_l150_15013

/-- Two complex numbers are symmetric about y = x if their real and imaginary parts are swapped -/
def symmetric_about_y_eq_x (z₁ z₂ : ℂ) : Prop :=
  z₁.re = z₂.im ∧ z₁.im = z₂.re

theorem symmetric_complex_division (z₁ z₂ : ℂ) : 
  symmetric_about_y_eq_x z₁ z₂ → z₁ = 1 + 2*I → z₁ / z₂ = 4/5 + 3/5*I := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_division_l150_15013


namespace NUMINAMATH_CALUDE_attention_index_properties_l150_15020

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 10 then 100 * a^(x/10) - 60
  else if 10 < x ∧ x ≤ 20 then 340
  else if 20 < x ∧ x ≤ 40 then 640 - 15*x
  else 0

theorem attention_index_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 5 = 140) :
  a = 4 ∧ 
  f a 5 > f a 35 ∧ 
  (Set.Icc 5 (100/3) : Set ℝ) = {x | 0 ≤ x ∧ x ≤ 40 ∧ f a x ≥ 140} :=
by sorry

end NUMINAMATH_CALUDE_attention_index_properties_l150_15020


namespace NUMINAMATH_CALUDE_min_value_of_f_l150_15047

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2000

-- Theorem statement
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = 1973 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l150_15047


namespace NUMINAMATH_CALUDE_black_beads_fraction_l150_15096

/-- Proves that the fraction of black beads pulled out is 1/6 given the initial conditions -/
theorem black_beads_fraction (total_white : ℕ) (total_black : ℕ) (total_pulled : ℕ) :
  total_white = 51 →
  total_black = 90 →
  total_pulled = 32 →
  (total_pulled - (total_white / 3)) / total_black = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_black_beads_fraction_l150_15096


namespace NUMINAMATH_CALUDE_rectangle_z_value_l150_15034

/-- A rectangle with given vertices and area -/
structure Rectangle where
  z : ℝ
  area : ℝ
  h_vertices : z > 5
  h_area : area = 64

/-- The value of z for the given rectangle is 13 -/
theorem rectangle_z_value (rect : Rectangle) : rect.z = 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_z_value_l150_15034


namespace NUMINAMATH_CALUDE_x_plus_y_equals_four_l150_15078

theorem x_plus_y_equals_four (x y : ℝ) 
  (h1 : |x| + x + y = 12) 
  (h2 : x + |y| - y = 16) : 
  x + y = 4 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_four_l150_15078


namespace NUMINAMATH_CALUDE_tangent_half_angle_sum_l150_15039

theorem tangent_half_angle_sum (α β γ : Real) (h : α + β + γ = Real.pi) :
  Real.tan (α/2) * Real.tan (β/2) + Real.tan (β/2) * Real.tan (γ/2) + Real.tan (γ/2) * Real.tan (α/2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_half_angle_sum_l150_15039


namespace NUMINAMATH_CALUDE_system_solution_l150_15035

theorem system_solution (x y : ℝ) (h1 : 3 * x + y = 21) (h2 : x + 3 * y = 1) : 2 * x + 2 * y = 11 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l150_15035


namespace NUMINAMATH_CALUDE_quadratic_roots_reciprocal_l150_15005

theorem quadratic_roots_reciprocal (b : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - b*x₁ + 1 = 0 ∧ x₂^2 - b*x₂ + 1 = 0 →
  (x₂ = 1 / x₁ ∨ (b = 2 ∧ x₁ = 1 ∧ x₂ = 1) ∨ (b = -2 ∧ x₁ = -1 ∧ x₂ = -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_reciprocal_l150_15005


namespace NUMINAMATH_CALUDE_rectangle_area_l150_15099

/-- The length of the shorter side of the smaller rectangles -/
def short_side : ℝ := 7

/-- The length of the longer side of the smaller rectangles -/
def long_side : ℝ := 3 * short_side

/-- The width of the larger rectangle EFGH -/
def width : ℝ := long_side

/-- The length of the larger rectangle EFGH -/
def length : ℝ := long_side + short_side

/-- The area of the larger rectangle EFGH -/
def area : ℝ := length * width

theorem rectangle_area : area = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l150_15099


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l150_15059

theorem rectangle_perimeter (area : ℝ) (side_ratio : ℝ) (perimeter : ℝ) : 
  area = 500 →
  side_ratio = 2 →
  let shorter_side := Real.sqrt (area / side_ratio)
  let longer_side := side_ratio * shorter_side
  perimeter = 2 * (shorter_side + longer_side) →
  perimeter = 30 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l150_15059


namespace NUMINAMATH_CALUDE_heptagon_coloring_l150_15036

-- Define the color type
inductive Color
| Red
| Blue
| Yellow
| Green

-- Define the heptagon type
def Heptagon := Fin 7 → Color

-- Define the coloring conditions
def validColoring (h : Heptagon) : Prop :=
  ∀ i : Fin 7,
    (h i = Color.Red ∨ h i = Color.Blue →
      h ((i + 1) % 7) ≠ Color.Blue ∧ h ((i + 1) % 7) ≠ Color.Green ∧
      h ((i + 4) % 7) ≠ Color.Blue ∧ h ((i + 4) % 7) ≠ Color.Green) ∧
    (h i = Color.Yellow ∨ h i = Color.Green →
      h ((i + 1) % 7) ≠ Color.Red ∧ h ((i + 1) % 7) ≠ Color.Yellow ∧
      h ((i + 4) % 7) ≠ Color.Red ∧ h ((i + 4) % 7) ≠ Color.Yellow)

-- Theorem statement
theorem heptagon_coloring (h : Heptagon) (hvalid : validColoring h) :
  ∃ c : Color, ∀ i : Fin 7, h i = c :=
sorry

end NUMINAMATH_CALUDE_heptagon_coloring_l150_15036


namespace NUMINAMATH_CALUDE_product_and_sum_of_consecutive_integers_l150_15016

theorem product_and_sum_of_consecutive_integers : 
  ∃ (a b c d e : ℤ), 
    (b = a + 1) ∧ 
    (d = c + 1) ∧ 
    (e = d + 1) ∧ 
    (a > 0) ∧ 
    (a * b = 198) ∧ 
    (c * d * e = 198) ∧ 
    (a + b + c + d + e = 39) := by
  sorry

end NUMINAMATH_CALUDE_product_and_sum_of_consecutive_integers_l150_15016


namespace NUMINAMATH_CALUDE_train_travel_time_l150_15065

/-- Represents the problem of calculating the travel time of two trains --/
theorem train_travel_time 
  (cattle_speed : ℝ) 
  (speed_difference : ℝ) 
  (head_start : ℝ) 
  (total_distance : ℝ) 
  (h1 : cattle_speed = 56) 
  (h2 : speed_difference = 33) 
  (h3 : head_start = 6) 
  (h4 : total_distance = 1284) :
  ∃ t : ℝ, 
    t > 0 ∧ 
    cattle_speed * (t + head_start) + (cattle_speed - speed_difference) * t = total_distance ∧ 
    t = 12 := by
  sorry


end NUMINAMATH_CALUDE_train_travel_time_l150_15065


namespace NUMINAMATH_CALUDE_basketball_shots_mode_and_median_l150_15087

def data_set : List Nat := [6, 7, 6, 9, 8]

def mode (l : List Nat) : Nat := sorry

def median (l : List Nat) : Nat := sorry

theorem basketball_shots_mode_and_median :
  mode data_set = 6 ∧ median data_set = 7 := by sorry

end NUMINAMATH_CALUDE_basketball_shots_mode_and_median_l150_15087


namespace NUMINAMATH_CALUDE_square_sum_theorem_l150_15089

theorem square_sum_theorem (x y z a b c : ℝ) 
  (h1 : x * y = a) 
  (h2 : x * z = b) 
  (h3 : y * z = c) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  x^2 + y^2 + z^2 = ((a*b)^2 + (a*c)^2 + (b*c)^2) / (a*b*c) := by
sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l150_15089


namespace NUMINAMATH_CALUDE_specific_prism_volume_max_prism_volume_max_volume_achievable_l150_15021

/-- Regular quadrangular pyramid with inscribed regular triangular prism -/
structure PyramidWithPrism where
  /-- Volume of the pyramid -/
  V : ℝ
  /-- Angle between lateral edge and base plane (in radians) -/
  angle : ℝ
  /-- Ratio of the division of the pyramid's height by the prism's face -/
  ratio : ℝ × ℝ
  /-- Volume of the inscribed prism -/
  prismVolume : ℝ
  /-- Constraint: angle is 30 degrees (π/6 radians) -/
  angle_is_30_deg : angle = Real.pi / 6
  /-- Constraint: ratio is valid (both parts positive, sum > 0) -/
  ratio_valid : ratio.1 > 0 ∧ ratio.2 > 0 ∧ ratio.1 + ratio.2 > 0
  /-- Constraint: prism volume is positive and less than pyramid volume -/
  volume_valid : 0 < prismVolume ∧ prismVolume < V

/-- Theorem for the volume of the specific prism -/
theorem specific_prism_volume (p : PyramidWithPrism) (h : p.ratio = (2, 3)) :
  p.prismVolume = 9/250 * p.V := by sorry

/-- Theorem for the maximum volume of any such prism -/
theorem max_prism_volume (p : PyramidWithPrism) :
  p.prismVolume ≤ 1/12 * p.V := by sorry

/-- Theorem that 1/12 is achievable -/
theorem max_volume_achievable (V : ℝ) (h : V > 0) :
  ∃ p : PyramidWithPrism, p.V = V ∧ p.prismVolume = 1/12 * V := by sorry

end NUMINAMATH_CALUDE_specific_prism_volume_max_prism_volume_max_volume_achievable_l150_15021


namespace NUMINAMATH_CALUDE_always_returns_to_present_max_stations_visited_l150_15000

/-- Represents the time machine's movement on a circular track of 2009 stations. -/
def TimeMachine :=
  { s : ℕ // s ≤ 2009 }

/-- Moves the time machine to the next station according to the rules. -/
def nextStation (s : TimeMachine) : TimeMachine :=
  sorry

/-- Checks if a number is a power of 2. -/
def isPowerOfTwo (n : ℕ) : Bool :=
  sorry

/-- Returns the sequence of stations visited by the time machine starting from a given station. -/
def stationSequence (start : TimeMachine) : List TimeMachine :=
  sorry

/-- Theorem stating that the time machine always returns to station 1. -/
theorem always_returns_to_present (start : TimeMachine) :
  1 ∈ (stationSequence start).map (fun s => s.val) := by
  sorry

/-- Theorem stating the maximum number of stations the time machine can stop at. -/
theorem max_stations_visited :
  ∃ (start : TimeMachine), (stationSequence start).length = 812 ∧
  ∀ (s : TimeMachine), (stationSequence s).length ≤ 812 := by
  sorry

end NUMINAMATH_CALUDE_always_returns_to_present_max_stations_visited_l150_15000


namespace NUMINAMATH_CALUDE_replaced_person_weight_l150_15051

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℚ) (new_person_weight : ℚ) : ℚ :=
  new_person_weight - (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the replaced person is 67 kg -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 (5/2) 87 = 67 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l150_15051


namespace NUMINAMATH_CALUDE_right_triangle_and_multiplicative_inverse_l150_15023

theorem right_triangle_and_multiplicative_inverse :
  (30^2 + 272^2 = 278^2) ∧
  ((550 * 6) % 4079 = 1) ∧
  (0 ≤ 6 ∧ 6 < 4079) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_and_multiplicative_inverse_l150_15023


namespace NUMINAMATH_CALUDE_jennifer_book_expense_l150_15029

theorem jennifer_book_expense (total : ℚ) (sandwich_fraction : ℚ) (ticket_fraction : ℚ) (leftover : ℚ) :
  total = 180 →
  sandwich_fraction = 1 / 5 →
  ticket_fraction = 1 / 6 →
  leftover = 24 →
  ∃ (book_fraction : ℚ),
    book_fraction = 1 / 2 ∧
    total * sandwich_fraction + total * ticket_fraction + total * book_fraction + leftover = total :=
by sorry

end NUMINAMATH_CALUDE_jennifer_book_expense_l150_15029


namespace NUMINAMATH_CALUDE_a_minus_b_equals_seven_l150_15088

theorem a_minus_b_equals_seven (a b : ℝ) 
  (ha : a^2 = 9)
  (hb : |b| = 4)
  (hgt : a > b) : 
  a - b = 7 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_seven_l150_15088


namespace NUMINAMATH_CALUDE_nested_average_equals_25_18_l150_15003

/-- Average of two numbers -/
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

/-- Average of three numbers -/
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- The main theorem to prove -/
theorem nested_average_equals_25_18 :
  avg3 (avg3 2 2 1) (avg2 1 2) 1 = 25 / 18 := by sorry

end NUMINAMATH_CALUDE_nested_average_equals_25_18_l150_15003


namespace NUMINAMATH_CALUDE_ring_area_l150_15043

theorem ring_area (r : ℝ) (h : r > 0) :
  let outer_radius : ℝ := 3 * r
  let inner_radius : ℝ := r
  let width : ℝ := 3
  outer_radius - inner_radius = width →
  (π * outer_radius^2 - π * inner_radius^2) = 72 * π :=
by
  sorry

end NUMINAMATH_CALUDE_ring_area_l150_15043


namespace NUMINAMATH_CALUDE_fathers_sons_age_product_l150_15061

theorem fathers_sons_age_product (father_age son_age : ℕ) : 
  father_age > 0 ∧ son_age > 0 ∧
  father_age = 7 * (son_age / 3) ∧
  (father_age + 6) = 2 * (son_age + 6) →
  father_age * son_age = 756 := by
sorry

end NUMINAMATH_CALUDE_fathers_sons_age_product_l150_15061


namespace NUMINAMATH_CALUDE_smallest_coefficient_value_l150_15041

-- Define the ratio condition
def ratio_condition (n : ℕ) : Prop :=
  (6^n) / (2^n) = 729

-- Define the function to get the coefficient of the term with the smallest coefficient
def smallest_coefficient (n : ℕ) : ℤ :=
  (-1)^(n - 3) * (Nat.choose n 3)

-- Theorem statement
theorem smallest_coefficient_value :
  ∃ n : ℕ, ratio_condition n ∧ smallest_coefficient n = -20 := by
  sorry

end NUMINAMATH_CALUDE_smallest_coefficient_value_l150_15041


namespace NUMINAMATH_CALUDE_other_factor_power_of_two_l150_15053

def w : ℕ := 144

theorem other_factor_power_of_two :
  (∃ (k : ℕ), 936 * w = k * (3^3) * (12^2)) →
  (∀ (m : ℕ), m < w → ¬(∃ (l : ℕ), 936 * m = l * (3^3) * (12^2))) →
  (∃ (x : ℕ), 2^x ∣ (936 * w) ∧ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_other_factor_power_of_two_l150_15053


namespace NUMINAMATH_CALUDE_frog_jump_probability_l150_15066

-- Define the frog's jump
structure Jump where
  length : ℝ
  direction : ℝ × ℝ

-- Define the frog's position
def Position := ℝ × ℝ

-- Define a function to calculate the final position after n jumps
def finalPosition (jumps : List Jump) : Position :=
  sorry

-- Define a function to calculate the distance between two positions
def distance (p1 p2 : Position) : ℝ :=
  sorry

-- Define the probability function
def probability (n : ℕ) (jumpLength : ℝ) (maxDistance : ℝ) : ℝ :=
  sorry

-- Theorem statement
theorem frog_jump_probability :
  probability 5 1 1.5 = 1/8 :=
sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l150_15066


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l150_15031

theorem quadratic_equation_problem : 
  (∀ m : ℝ, ∃ x : ℝ, x^2 - m*x - 1 = 0) ∧ 
  (∃ x₀ : ℕ, x₀^2 - 2*x₀ - 1 ≤ 0) → 
  ¬((∀ m : ℝ, ∃ x : ℝ, x^2 - m*x - 1 = 0) ∧ 
    ¬(∃ x₀ : ℕ, x₀^2 - 2*x₀ - 1 ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l150_15031


namespace NUMINAMATH_CALUDE_teacher_selection_theorem_l150_15082

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to select 6 teachers out of 10, where two specific teachers cannot be selected together -/
def selectTeachers (totalTeachers invitedTeachers : ℕ) : ℕ :=
  binomial totalTeachers invitedTeachers - binomial (totalTeachers - 2) (invitedTeachers - 2)

theorem teacher_selection_theorem :
  selectTeachers 10 6 = 140 := by sorry

end NUMINAMATH_CALUDE_teacher_selection_theorem_l150_15082


namespace NUMINAMATH_CALUDE_student_average_greater_than_true_average_l150_15063

theorem student_average_greater_than_true_average (x y z : ℝ) (h : x < z ∧ z < y) :
  (x + z) / 2 / 2 + y / 2 > (x + y + z) / 3 := by
  sorry

end NUMINAMATH_CALUDE_student_average_greater_than_true_average_l150_15063


namespace NUMINAMATH_CALUDE_rals_age_is_26_l150_15070

/-- Ral's current age -/
def rals_age : ℕ := 26

/-- Suri's current age -/
def suris_age : ℕ := 13

/-- Ral is twice as old as Suri -/
axiom ral_twice_suri : rals_age = 2 * suris_age

/-- In 3 years, Suri's current age will be 16 -/
axiom suri_age_in_3_years : suris_age + 3 = 16

/-- Theorem: Ral's current age is 26 years old -/
theorem rals_age_is_26 : rals_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_rals_age_is_26_l150_15070


namespace NUMINAMATH_CALUDE_tom_green_marbles_l150_15062

/-- The number of green marbles Sara has -/
def sara_green : ℕ := 3

/-- The total number of green marbles Sara and Tom have together -/
def total_green : ℕ := 7

/-- The number of green marbles Tom has -/
def tom_green : ℕ := total_green - sara_green

theorem tom_green_marbles : tom_green = 4 := by
  sorry

end NUMINAMATH_CALUDE_tom_green_marbles_l150_15062


namespace NUMINAMATH_CALUDE_cafe_customers_l150_15022

/-- The number of sandwiches ordered by offices -/
def office_sandwiches : ℕ := 30

/-- The number of sandwiches each ordering customer in the group ordered -/
def sandwiches_per_customer : ℕ := 4

/-- The total number of sandwiches made by the café -/
def total_sandwiches : ℕ := 54

/-- The fraction of the group that ordered sandwiches -/
def ordering_fraction : ℚ := 1/2

theorem cafe_customers : ℕ :=
  let group_sandwiches := total_sandwiches - office_sandwiches
  let ordering_customers := group_sandwiches / sandwiches_per_customer
  let total_customers := ordering_customers / ordering_fraction
  12

#check cafe_customers

end NUMINAMATH_CALUDE_cafe_customers_l150_15022


namespace NUMINAMATH_CALUDE_matrix_pattern_l150_15074

/-- Given a 2x2 matrix [[a, 2], [5, 6]] where a is unknown, 
    if (5 * 6) = (a * 2) * 3, then a = 5 -/
theorem matrix_pattern (a : ℝ) : (5 * 6 : ℝ) = (a * 2) * 3 → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_matrix_pattern_l150_15074


namespace NUMINAMATH_CALUDE_smallest_games_for_score_l150_15064

theorem smallest_games_for_score (win_points loss_points final_score : ℤ)
  (win_points_pos : win_points > 0)
  (loss_points_pos : loss_points > 0)
  (final_score_pos : final_score > 0)
  (h : win_points = 25 ∧ loss_points = 13 ∧ final_score = 2007) :
  ∃ (wins losses : ℕ),
    wins * win_points - losses * loss_points = final_score ∧
    wins + losses = 87 ∧
    ∀ (w l : ℕ), w * win_points - l * loss_points = final_score →
      w + l ≥ 87 := by
sorry

end NUMINAMATH_CALUDE_smallest_games_for_score_l150_15064


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_l150_15007

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 2} := by sorry

-- Theorem for the complement of the union of A and B
theorem complement_union_A_B : (A ∪ B)ᶜ = {x : ℝ | x ≤ -1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_l150_15007


namespace NUMINAMATH_CALUDE_equation_solution_l150_15057

theorem equation_solution :
  ∃ x : ℚ, x ≠ 1 ∧ (1 / (x - 1) + 1 = 3 / (2 * x - 2)) ∧ x = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l150_15057


namespace NUMINAMATH_CALUDE_drop_is_negative_of_rise_is_positive_l150_15030

/-- Represents the change in water level -/
structure WaterLevelChange where
  magnitude : ℝ
  isRise : Bool

/-- Records a water level change as a signed real number -/
def recordChange (change : WaterLevelChange) : ℝ :=
  if change.isRise then change.magnitude else -change.magnitude

theorem drop_is_negative_of_rise_is_positive 
  (h : ∀ (rise : WaterLevelChange), rise.isRise → recordChange rise = rise.magnitude) :
  ∀ (drop : WaterLevelChange), ¬drop.isRise → recordChange drop = -drop.magnitude :=
by sorry

end NUMINAMATH_CALUDE_drop_is_negative_of_rise_is_positive_l150_15030


namespace NUMINAMATH_CALUDE_smaller_pyramid_volume_theorem_l150_15012

/-- A right square pyramid with given dimensions -/
structure RightSquarePyramid where
  base_edge : ℝ
  slant_edge : ℝ

/-- A plane cutting the pyramid parallel to its base -/
structure CuttingPlane where
  height : ℝ

/-- The volume of the smaller pyramid cut off by the plane -/
def smaller_pyramid_volume (p : RightSquarePyramid) (c : CuttingPlane) : ℝ :=
  sorry

/-- Theorem stating the volume of the smaller pyramid -/
theorem smaller_pyramid_volume_theorem (p : RightSquarePyramid) (c : CuttingPlane) :
  p.base_edge = 12 * Real.sqrt 2 →
  p.slant_edge = 15 →
  c.height = 5 →
  smaller_pyramid_volume p c = 24576 / 507 :=
sorry

end NUMINAMATH_CALUDE_smaller_pyramid_volume_theorem_l150_15012


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l150_15071

theorem binomial_expansion_problem (n : ℕ) (a : ℕ → ℝ) : 
  (∀ k, 0 ≤ k ∧ k ≤ n → a k = (-1)^k * (n.choose k)) →
  (2 * (n.choose 2) - a (n - 5) = 0) →
  n = 8 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l150_15071


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l150_15068

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b, a = 0 → a * b = 0) ∧
  (∃ a b, a * b = 0 ∧ a ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l150_15068


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l150_15046

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Proof that for an arithmetic sequence and distinct positive integers m, n, and p,
    the equation m(a_p - a_n) + n(a_m - a_p) + p(a_n - a_m) = 0 holds -/
theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (m n p : ℕ) (h_arith : ArithmeticSequence a) (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p) :
  m * (a p - a n) + n * (a m - a p) + p * (a n - a m) = 0 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l150_15046


namespace NUMINAMATH_CALUDE_power_function_through_point_l150_15017

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * (x ^ b)

-- Define the theorem
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2) : 
  f 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l150_15017


namespace NUMINAMATH_CALUDE_remaining_soup_feeds_20_adults_l150_15055

/-- Represents the number of adults a can of soup can feed -/
def adults_per_can : ℕ := 4

/-- Represents the number of children a can of soup can feed -/
def children_per_can : ℕ := 6

/-- Represents the total number of cans of soup -/
def total_cans : ℕ := 8

/-- Represents the number of children fed -/
def children_fed : ℕ := 20

/-- Represents the fraction of soup left in a can after feeding children -/
def leftover_fraction : ℚ := 1/3

/-- Calculates the number of adults that can be fed with the remaining soup -/
def adults_fed (adults_per_can : ℕ) (children_per_can : ℕ) (total_cans : ℕ) (children_fed : ℕ) (leftover_fraction : ℚ) : ℕ :=
  sorry

/-- Theorem stating that the remaining soup can feed 20 adults -/
theorem remaining_soup_feeds_20_adults : 
  adults_fed adults_per_can children_per_can total_cans children_fed leftover_fraction = 20 :=
sorry

end NUMINAMATH_CALUDE_remaining_soup_feeds_20_adults_l150_15055
