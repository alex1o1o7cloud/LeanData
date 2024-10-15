import Mathlib

namespace NUMINAMATH_CALUDE_hotel_moves_2_8_l1318_131897

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of ways guests can move in a 2 × n grid hotel -/
def hotelMoves (n : ℕ) : ℕ := (fib (n + 1)) ^ 2

/-- Theorem: The number of ways guests can move in a 2 × 8 grid hotel is 3025 -/
theorem hotel_moves_2_8 : hotelMoves 8 = 3025 := by
  sorry

end NUMINAMATH_CALUDE_hotel_moves_2_8_l1318_131897


namespace NUMINAMATH_CALUDE_distribute_balls_eq_partitions_six_balls_four_boxes_l1318_131876

/-- Number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Number of partitions of n into at most k parts -/
def partitions (n k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing n indistinguishable balls into k indistinguishable boxes
    is equivalent to finding partitions of n into at most k parts -/
theorem distribute_balls_eq_partitions (n k : ℕ) :
  distribute_balls n k = partitions n k := by sorry

/-- The specific case for 6 balls and 4 boxes -/
theorem six_balls_four_boxes :
  distribute_balls 6 4 = 9 := by sorry

end NUMINAMATH_CALUDE_distribute_balls_eq_partitions_six_balls_four_boxes_l1318_131876


namespace NUMINAMATH_CALUDE_negative_two_less_than_negative_three_halves_l1318_131883

theorem negative_two_less_than_negative_three_halves : -2 < -(3/2) := by
  sorry

end NUMINAMATH_CALUDE_negative_two_less_than_negative_three_halves_l1318_131883


namespace NUMINAMATH_CALUDE_average_attendance_theorem_l1318_131862

/-- Represents the attendance data for a week -/
structure WeekAttendance where
  totalStudents : ℕ
  mondayAbsence : ℚ
  tuesdayAbsence : ℚ
  wednesdayAbsence : ℚ
  thursdayAbsence : ℚ
  fridayAbsence : ℚ

/-- Calculates the average number of students present in a week -/
def averageAttendance (w : WeekAttendance) : ℚ :=
  let mondayPresent := w.totalStudents * (1 - w.mondayAbsence)
  let tuesdayPresent := w.totalStudents * (1 - w.tuesdayAbsence)
  let wednesdayPresent := w.totalStudents * (1 - w.wednesdayAbsence)
  let thursdayPresent := w.totalStudents * (1 - w.thursdayAbsence)
  let fridayPresent := w.totalStudents * (1 - w.fridayAbsence)
  (mondayPresent + tuesdayPresent + wednesdayPresent + thursdayPresent + fridayPresent) / 5

theorem average_attendance_theorem (w : WeekAttendance) 
    (h1 : w.totalStudents = 50)
    (h2 : w.mondayAbsence = 1/10)
    (h3 : w.tuesdayAbsence = 3/25)
    (h4 : w.wednesdayAbsence = 3/20)
    (h5 : w.thursdayAbsence = 1/12.5)
    (h6 : w.fridayAbsence = 1/20) : 
  averageAttendance w = 45 := by
sorry

end NUMINAMATH_CALUDE_average_attendance_theorem_l1318_131862


namespace NUMINAMATH_CALUDE_power_of_power_l1318_131817

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1318_131817


namespace NUMINAMATH_CALUDE_cats_on_ship_l1318_131865

/-- Represents the passengers on the Queen Mary II luxury liner -/
structure Passengers where
  sailors : ℕ
  cats : ℕ

/-- The total number of heads on the ship -/
def total_heads (p : Passengers) : ℕ := p.sailors + 1 + 1 + p.cats

/-- The total number of legs on the ship -/
def total_legs (p : Passengers) : ℕ := 2 * p.sailors + 2 + 1 + 4 * p.cats

/-- Theorem stating that there are 7 cats on the ship -/
theorem cats_on_ship : 
  ∃ (p : Passengers), total_heads p = 15 ∧ total_legs p = 43 ∧ p.cats = 7 := by
  sorry

end NUMINAMATH_CALUDE_cats_on_ship_l1318_131865


namespace NUMINAMATH_CALUDE_age_ratio_proof_l1318_131819

/-- Given three people a, b, and c with ages satisfying certain conditions,
    prove that the ratio of b's age to c's age is 2:1. -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →                  -- a is two years older than b
  b = 18 →                     -- b is 18 years old
  a + b + c = 47 →             -- total of ages is 47
  ∃ (k : ℕ), b = k * c →       -- b is some times as old as c
  b = 2 * c                    -- ratio of b's age to c's age is 2:1
  := by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l1318_131819


namespace NUMINAMATH_CALUDE_xy_inequality_l1318_131861

theorem xy_inequality (x y : ℝ) (n : ℕ) (hx : x > 0) (hy : y > 0) :
  x * y ≤ (x^(n+2) + y^(n+2)) / (x^n + y^n) := by
  sorry

end NUMINAMATH_CALUDE_xy_inequality_l1318_131861


namespace NUMINAMATH_CALUDE_max_profit_at_two_l1318_131802

noncomputable section

-- Define the sales volume function
def sales_volume (x : ℝ) : ℝ :=
  if 1 < x ∧ x ≤ 3 then (x - 4)^2 + 6 / (x - 1)
  else if 3 < x ∧ x ≤ 5 then -x + 7
  else 0

-- Define the profit function
def profit (x : ℝ) : ℝ :=
  (sales_volume x) * (x - 1)

-- Main theorem
theorem max_profit_at_two :
  ∀ x, 1 < x ∧ x ≤ 5 → profit x ≤ profit 2 :=
by sorry

end

end NUMINAMATH_CALUDE_max_profit_at_two_l1318_131802


namespace NUMINAMATH_CALUDE_xyz_equals_27_l1318_131855

theorem xyz_equals_27 
  (a b c x y z : ℂ)
  (nonzero_a : a ≠ 0)
  (nonzero_b : b ≠ 0)
  (nonzero_c : c ≠ 0)
  (nonzero_x : x ≠ 0)
  (nonzero_y : y ≠ 0)
  (nonzero_z : z ≠ 0)
  (eq_a : a = b * c * (x - 2))
  (eq_b : b = a * c * (y - 2))
  (eq_c : c = a * b * (z - 2))
  (sum_product : x * y + x * z + y * z = 10)
  (sum : x + y + z = 6) :
  x * y * z = 27 := by
  sorry


end NUMINAMATH_CALUDE_xyz_equals_27_l1318_131855


namespace NUMINAMATH_CALUDE_fermat_coprime_and_infinite_primes_l1318_131828

def fermat (n : ℕ) : ℕ := 2^(2^n) + 1

theorem fermat_coprime_and_infinite_primes :
  (∀ n m : ℕ, n ≠ m → Nat.gcd (fermat n) (fermat m) = 1) ∧
  (¬ ∃ N : ℕ, ∀ p : ℕ, Prime p → p ≤ N) :=
sorry

end NUMINAMATH_CALUDE_fermat_coprime_and_infinite_primes_l1318_131828


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l1318_131891

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (2, 4)
  let b : ℝ × ℝ := (x, 6)
  collinear a b → x = 3 :=
by sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l1318_131891


namespace NUMINAMATH_CALUDE_concert_revenue_is_955000_l1318_131815

/-- Calculates the total revenue of a concert given the following parameters:
  * total_seats: Total number of seats in the arena
  * main_seat_cost: Cost of a main seat ticket
  * back_seat_cost: Cost of a back seat ticket
  * back_seats_sold: Number of back seat tickets sold
-/
def concert_revenue (total_seats : ℕ) (main_seat_cost back_seat_cost : ℕ) (back_seats_sold : ℕ) : ℕ :=
  let main_seats_sold := total_seats - back_seats_sold
  let main_seat_revenue := main_seats_sold * main_seat_cost
  let back_seat_revenue := back_seats_sold * back_seat_cost
  main_seat_revenue + back_seat_revenue

/-- Theorem stating that the concert revenue is $955,000 given the specific conditions -/
theorem concert_revenue_is_955000 :
  concert_revenue 20000 55 45 14500 = 955000 := by
  sorry

#eval concert_revenue 20000 55 45 14500

end NUMINAMATH_CALUDE_concert_revenue_is_955000_l1318_131815


namespace NUMINAMATH_CALUDE_population_difference_is_167_l1318_131820

/-- Represents a tribe with male and female populations -/
structure Tribe where
  males : Nat
  females : Nat

/-- The Gaga tribe -/
def gaga : Tribe := ⟨204, 468⟩

/-- The Nana tribe -/
def nana : Tribe := ⟨334, 516⟩

/-- The Dada tribe -/
def dada : Tribe := ⟨427, 458⟩

/-- The Lala tribe -/
def lala : Tribe := ⟨549, 239⟩

/-- The list of all tribes on the couple continent -/
def tribes : List Tribe := [gaga, nana, dada, lala]

/-- The total number of males on the couple continent -/
def totalMales : Nat := (tribes.map (·.males)).sum

/-- The total number of females on the couple continent -/
def totalFemales : Nat := (tribes.map (·.females)).sum

/-- The difference between females and males on the couple continent -/
def populationDifference : Nat := totalFemales - totalMales

theorem population_difference_is_167 : populationDifference = 167 := by
  sorry

end NUMINAMATH_CALUDE_population_difference_is_167_l1318_131820


namespace NUMINAMATH_CALUDE_equation_solution_l1318_131898

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - |x| - 1
  ∀ x : ℝ, f x = 0 ↔ x = (-1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1318_131898


namespace NUMINAMATH_CALUDE_base_2_representation_of_96_l1318_131844

theorem base_2_representation_of_96 :
  ∃ (a b c d e f g : Nat),
    96 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 ∧
    a = 1 ∧ b = 1 ∧ c = 0 ∧ d = 0 ∧ e = 0 ∧ f = 0 ∧ g = 0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_96_l1318_131844


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2014_l1318_131833

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_2014 :
  ∃ n : ℕ, arithmetic_sequence 1 3 n = 2014 ∧ n = 672 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2014_l1318_131833


namespace NUMINAMATH_CALUDE_broth_per_serving_is_two_point_five_l1318_131879

/-- Represents the number of cups in one pint -/
def cups_per_pint : ℚ := 2

/-- Represents the number of servings -/
def num_servings : ℕ := 8

/-- Represents the number of pints of vegetables and broth combined for all servings -/
def total_pints : ℚ := 14

/-- Represents the number of cups of vegetables in one serving -/
def vegetables_per_serving : ℚ := 1

/-- Calculates the number of cups of broth in one serving of soup -/
def broth_per_serving : ℚ :=
  (total_pints * cups_per_pint - num_servings * vegetables_per_serving) / num_servings

theorem broth_per_serving_is_two_point_five :
  broth_per_serving = 2.5 := by sorry

end NUMINAMATH_CALUDE_broth_per_serving_is_two_point_five_l1318_131879


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l1318_131830

theorem purely_imaginary_z (z : ℂ) : 
  (∃ b : ℝ, z = Complex.I * b) → -- z is purely imaginary
  (∃ c : ℝ, (z + 1)^2 - 2*Complex.I = Complex.I * c) → -- (z+1)^2 - 2i is purely imaginary
  z = -Complex.I := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l1318_131830


namespace NUMINAMATH_CALUDE_distinct_collections_biology_l1318_131877

def biology : Finset Char := {'B', 'I', 'O', 'L', 'O', 'G', 'Y'}

def vowels : Finset Char := {'I', 'O'}
def consonants : Finset Char := {'B', 'L', 'G', 'Y'}

def num_o : ℕ := (biology.filter (· = 'O')).card

theorem distinct_collections_biology :
  let total_selections := (Finset.powerset biology).filter (λ s => 
    (s.filter (λ c => c ∈ vowels)).card = 3 ∧ 
    (s.filter (λ c => c ∈ consonants)).card = 2)
  (Finset.powerset total_selections).card = 18 :=
sorry

end NUMINAMATH_CALUDE_distinct_collections_biology_l1318_131877


namespace NUMINAMATH_CALUDE_positive_number_problem_l1318_131832

theorem positive_number_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x - 4 = 21 / x)
  (eq2 : x + y^2 = 45)
  (eq3 : y * z = x^3) :
  x = 7 ∧ y = Real.sqrt 38 ∧ z = 343 * Real.sqrt 38 / 38 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_problem_l1318_131832


namespace NUMINAMATH_CALUDE_sphere_volume_equal_surface_area_cube_l1318_131842

/-- The volume of a sphere with surface area equal to a cube of side length 2 -/
theorem sphere_volume_equal_surface_area_cube (r : ℝ) : 
  (4 * Real.pi * r^2 = 6 * 2^2) → 
  ((4 / 3) * Real.pi * r^3 = (8 * Real.sqrt 6) / Real.sqrt Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_equal_surface_area_cube_l1318_131842


namespace NUMINAMATH_CALUDE_marble_drawing_probability_l1318_131860

/-- The probability of drawing marbles consecutively by color --/
theorem marble_drawing_probability : 
  let total_marbles : ℕ := 12
  let blue_marbles : ℕ := 4
  let orange_marbles : ℕ := 3
  let green_marbles : ℕ := 5
  let favorable_outcomes : ℕ := Nat.factorial 3 * Nat.factorial blue_marbles * 
                                 Nat.factorial orange_marbles * Nat.factorial green_marbles
  let total_outcomes : ℕ := Nat.factorial total_marbles
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4620 := by
  sorry

end NUMINAMATH_CALUDE_marble_drawing_probability_l1318_131860


namespace NUMINAMATH_CALUDE_average_increase_l1318_131804

theorem average_increase (numbers : Finset ℕ) (sum : ℕ) (added_value : ℕ) :
  numbers.card = 15 →
  sum = numbers.sum id →
  sum / numbers.card = 40 →
  added_value = 10 →
  (sum + numbers.card * added_value) / numbers.card = 50 := by
sorry

end NUMINAMATH_CALUDE_average_increase_l1318_131804


namespace NUMINAMATH_CALUDE_max_ab_tangent_circles_l1318_131867

/-- Two externally tangent circles -/
structure TangentCircles where
  a : ℝ
  b : ℝ
  c1 : (x : ℝ) → (y : ℝ) → (x - a)^2 + (y + 2)^2 = 4
  c2 : (x : ℝ) → (y : ℝ) → (x + b)^2 + (y + 2)^2 = 1
  tangent : a + b = 3

/-- The maximum value of ab for externally tangent circles -/
theorem max_ab_tangent_circles (tc : TangentCircles) : 
  ∃ (max : ℝ), max = 9/4 ∧ tc.a * tc.b ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_ab_tangent_circles_l1318_131867


namespace NUMINAMATH_CALUDE_m_less_than_n_min_sum_a_b_l1318_131858

-- Define the variables and conditions
variables (a b : ℝ) (m n : ℝ)

-- Define the relationships between variables
def m_def : m = a * b + 1 := by sorry
def n_def : n = a + b := by sorry

-- Part 1: Prove m < n when a > 1 and b < 1
theorem m_less_than_n (ha : a > 1) (hb : b < 1) : m < n := by sorry

-- Part 2: Prove minimum value of a + b is 16 when a > 1, b > 1, and m - n = 49
theorem min_sum_a_b (ha : a > 1) (hb : b > 1) (h_diff : m - n = 49) :
  ∃ (min_sum : ℝ), min_sum = 16 ∧ a + b ≥ min_sum := by sorry

end NUMINAMATH_CALUDE_m_less_than_n_min_sum_a_b_l1318_131858


namespace NUMINAMATH_CALUDE_candy_division_l1318_131821

theorem candy_division (total_candies : ℕ) (num_groups : ℕ) (candies_per_group : ℕ) 
  (h1 : total_candies = 30)
  (h2 : num_groups = 10)
  (h3 : candies_per_group = total_candies / num_groups) :
  candies_per_group = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_division_l1318_131821


namespace NUMINAMATH_CALUDE_johns_donation_l1318_131874

/-- Given 10 initial contributions, if a new donation causes the average
    contribution to increase by 80% to $90, then the new donation must be $490. -/
theorem johns_donation (initial_count : ℕ) (increase_percentage : ℚ) (new_average : ℚ) :
  initial_count = 10 →
  increase_percentage = 80 / 100 →
  new_average = 90 →
  let initial_average := new_average / (1 + increase_percentage)
  let initial_total := initial_count * initial_average
  let new_total := (initial_count + 1) * new_average
  new_total - initial_total = 490 := by
sorry

end NUMINAMATH_CALUDE_johns_donation_l1318_131874


namespace NUMINAMATH_CALUDE_coffee_blend_price_l1318_131823

/-- Given two coffee blends, this theorem proves the price of the second blend
    given the conditions of the problem. -/
theorem coffee_blend_price
  (price_blend1 : ℝ)
  (total_weight : ℝ)
  (total_price_per_pound : ℝ)
  (weight_blend2 : ℝ)
  (h1 : price_blend1 = 9)
  (h2 : total_weight = 20)
  (h3 : total_price_per_pound = 8.4)
  (h4 : weight_blend2 = 12)
  : ∃ (price_blend2 : ℝ),
    price_blend2 * weight_blend2 + price_blend1 * (total_weight - weight_blend2) =
    total_price_per_pound * total_weight ∧
    price_blend2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_coffee_blend_price_l1318_131823


namespace NUMINAMATH_CALUDE_greatest_odd_integer_below_sqrt_50_l1318_131822

theorem greatest_odd_integer_below_sqrt_50 :
  ∀ x : ℕ, x % 2 = 1 → x^2 < 50 → x ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_odd_integer_below_sqrt_50_l1318_131822


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_composite_l1318_131894

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ

/-- The roots of a quadratic polynomial -/
structure Roots where
  x₁ : ℤ
  x₂ : ℤ

/-- Predicate to check if a number is composite -/
def IsComposite (n : ℤ) : Prop :=
  ∃ m k : ℤ, m ≠ 1 ∧ m ≠ -1 ∧ k ≠ 1 ∧ k ≠ -1 ∧ n = m * k

/-- Main theorem -/
theorem quadratic_roots_imply_composite
  (p : QuadraticPolynomial)
  (r : Roots)
  (h₁ : r.x₁ * r.x₁ + p.a * r.x₁ + p.b = 0)
  (h₂ : r.x₂ * r.x₂ + p.a * r.x₂ + p.b = 0)
  (h₃ : |r.x₁| > 2)
  (h₄ : |r.x₂| > 2) :
  IsComposite (p.a + p.b + 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_composite_l1318_131894


namespace NUMINAMATH_CALUDE_worker_original_wage_l1318_131806

/-- Calculates the worker's original daily wage given the conditions of the problem -/
def calculate_original_wage (increase_percentage : ℚ) (final_take_home : ℚ) 
  (tax_rate : ℚ) (fixed_deduction : ℚ) : ℚ :=
  let increased_wage := (1 + increase_percentage) * (final_take_home + fixed_deduction) / (1 - tax_rate)
  increased_wage / (1 + increase_percentage)

/-- Theorem stating that the worker's original daily wage is $37.50 -/
theorem worker_original_wage :
  calculate_original_wage (1/2) 42 (1/5) 3 = 75/2 :=
sorry

end NUMINAMATH_CALUDE_worker_original_wage_l1318_131806


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l1318_131826

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 5^n + 1
  let r : ℕ := 3^s - 3*s
  r = 3^126 - 378 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l1318_131826


namespace NUMINAMATH_CALUDE_polynomial_product_equality_l1318_131857

theorem polynomial_product_equality (x y z : ℝ) :
  (3 * x^4 - 4 * y^3 - 6 * z^2) * (9 * x^8 + 16 * y^6 + 36 * z^4 + 12 * x^4 * y^3 + 18 * x^4 * z^2 + 24 * y^3 * z^2) =
  27 * x^12 - 64 * y^9 - 216 * z^6 - 216 * x^4 * y^3 * z^2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_equality_l1318_131857


namespace NUMINAMATH_CALUDE_square_vertex_B_l1318_131839

/-- A square in a 2D Cartesian plane -/
structure Square where
  O : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Predicate to check if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

/-- Theorem: Given a square OABC with O(0,0) and A(4,3), and C in the fourth quadrant, B is at (7,-1) -/
theorem square_vertex_B (s : Square) : 
  s.O = (0, 0) → 
  s.A = (4, 3) → 
  isInFourthQuadrant s.C → 
  s.B = (7, -1) := by
  sorry


end NUMINAMATH_CALUDE_square_vertex_B_l1318_131839


namespace NUMINAMATH_CALUDE_max_value_w_l1318_131871

theorem max_value_w (p q : ℝ) (w : ℝ) 
  (hw : w = Real.sqrt (2 * p - q) + Real.sqrt (3 * q - 2 * p) + Real.sqrt (6 - 2 * q))
  (h1 : 2 * p - q ≥ 0)
  (h2 : 3 * q - 2 * p ≥ 0)
  (h3 : 6 - 2 * q ≥ 0) :
  w ≤ 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_w_l1318_131871


namespace NUMINAMATH_CALUDE_donation_sum_l1318_131868

theorem donation_sum : 
  let donation1 : ℝ := 245.00
  let donation2 : ℝ := 225.00
  let donation3 : ℝ := 230.00
  donation1 + donation2 + donation3 = 700.00 := by
  sorry

end NUMINAMATH_CALUDE_donation_sum_l1318_131868


namespace NUMINAMATH_CALUDE_regular_polygon_distance_sum_l1318_131841

theorem regular_polygon_distance_sum (n : ℕ) (h : ℝ) (h_list : List ℝ) :
  n > 2 →
  h > 0 →
  h_list.length = n →
  (∀ x ∈ h_list, x > 0) →
  h_list.sum = n * h :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_distance_sum_l1318_131841


namespace NUMINAMATH_CALUDE_circles_intersection_l1318_131873

def circle1_center : ℝ × ℝ := (0, 0)
def circle2_center : ℝ × ℝ := (-3, 4)
def circle2_radius : ℝ := 2

theorem circles_intersection (m : ℝ) :
  (∃ (x y : ℝ), (x - circle1_center.1)^2 + (y - circle1_center.2)^2 = m ∧
                (x - circle2_center.1)^2 + (y - circle2_center.2)^2 = circle2_radius^2) ↔
  9 < m ∧ m < 49 := by
  sorry

end NUMINAMATH_CALUDE_circles_intersection_l1318_131873


namespace NUMINAMATH_CALUDE_second_range_lower_limit_l1318_131896

theorem second_range_lower_limit (x y : ℝ) 
  (h1 : 3 < x) (h2 : x < 8) (h3 : x > y) (h4 : x < 10) (h5 : x = 7) : 
  3 < y ∧ y ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_second_range_lower_limit_l1318_131896


namespace NUMINAMATH_CALUDE_no_real_solutions_l1318_131870

theorem no_real_solutions :
  ¬∃ (x : ℝ), (x^4 + 3*x^3)/(x^2 + 3*x + 1) + x = -7 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1318_131870


namespace NUMINAMATH_CALUDE_male_to_total_ratio_l1318_131852

/-- Represents the population of alligators on Lagoon island -/
structure AlligatorPopulation where
  maleCount : ℕ
  adultFemaleCount : ℕ
  juvenileFemaleRatio : ℚ

/-- The ratio of male alligators to total alligators is 1:2 -/
theorem male_to_total_ratio (pop : AlligatorPopulation)
    (h1 : pop.maleCount = 25)
    (h2 : pop.adultFemaleCount = 15)
    (h3 : pop.juvenileFemaleRatio = 2/5) :
    pop.maleCount / (pop.maleCount + pop.adultFemaleCount / (1 - pop.juvenileFemaleRatio)) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_male_to_total_ratio_l1318_131852


namespace NUMINAMATH_CALUDE_prime_power_sum_l1318_131811

theorem prime_power_sum (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → 2^p + 3^p = a^n → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_l1318_131811


namespace NUMINAMATH_CALUDE_additional_sleep_january_l1318_131851

def sleep_december : ℝ := 6.5
def sleep_january : ℝ := 8.5
def days_in_month : ℕ := 31

theorem additional_sleep_january : 
  (sleep_january - sleep_december) * days_in_month = 62 := by
  sorry

end NUMINAMATH_CALUDE_additional_sleep_january_l1318_131851


namespace NUMINAMATH_CALUDE_simplify_expression_l1318_131846

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1318_131846


namespace NUMINAMATH_CALUDE_tire_price_problem_l1318_131864

theorem tire_price_problem (total_cost : ℝ) (discount_tire_price : ℝ) 
  (h1 : total_cost = 250)
  (h2 : discount_tire_price = 10) : 
  ∃ (regular_price : ℝ), 3 * regular_price + discount_tire_price = total_cost ∧ regular_price = 80 := by
  sorry

end NUMINAMATH_CALUDE_tire_price_problem_l1318_131864


namespace NUMINAMATH_CALUDE_profit_difference_l1318_131838

def chocolate_cakes_made : ℕ := 40
def vanilla_cakes_made : ℕ := 35
def strawberry_cakes_made : ℕ := 28
def pastries_made : ℕ := 153

def chocolate_cake_price : ℕ := 10
def vanilla_cake_price : ℕ := 12
def strawberry_cake_price : ℕ := 15
def pastry_price : ℕ := 5

def chocolate_cakes_sold : ℕ := 30
def vanilla_cakes_sold : ℕ := 25
def strawberry_cakes_sold : ℕ := 20
def pastries_sold : ℕ := 106

def total_cake_revenue : ℕ := 
  chocolate_cakes_sold * chocolate_cake_price +
  vanilla_cakes_sold * vanilla_cake_price +
  strawberry_cakes_sold * strawberry_cake_price

def total_pastry_revenue : ℕ := pastries_sold * pastry_price

theorem profit_difference : total_cake_revenue - total_pastry_revenue = 370 := by
  sorry

end NUMINAMATH_CALUDE_profit_difference_l1318_131838


namespace NUMINAMATH_CALUDE_remainder_of_190_div_18_l1318_131803

theorem remainder_of_190_div_18 :
  let g := Nat.gcd 60 190
  g = 18 → 190 % 18 = 10 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_190_div_18_l1318_131803


namespace NUMINAMATH_CALUDE_c_value_satisfies_equation_l1318_131834

/-- Definition of function F -/
def F (a b c d : ℝ) : ℝ := a * b^2 + c * d

/-- Theorem stating that c = 16 satisfies the equation when a = 2 -/
theorem c_value_satisfies_equation :
  ∃ c : ℝ, F 2 3 c 5 = F 2 5 c 3 ∧ c = 16 := by sorry

end NUMINAMATH_CALUDE_c_value_satisfies_equation_l1318_131834


namespace NUMINAMATH_CALUDE_team_improvements_minimum_days_team_a_l1318_131829

-- Define the problem parameters
def team_a_rate : ℝ := 15
def team_b_rate : ℝ := 10
def total_days : ℝ := 25
def total_length : ℝ := 300
def team_a_cost : ℝ := 0.6
def team_b_cost : ℝ := 0.8
def max_total_cost : ℝ := 18

-- Theorem for part 1
theorem team_improvements :
  ∃ (x y : ℝ),
    x + y = total_length ∧
    x / team_a_rate + y / team_b_rate = total_days ∧
    x = 150 ∧ y = 150 := by sorry

-- Theorem for part 2
theorem minimum_days_team_a :
  ∃ (m : ℝ),
    m ≥ 10 ∧
    ∀ (n : ℝ),
      n < 10 →
      team_a_cost * n + team_b_cost * ((total_length - team_a_rate * n) / team_b_rate) > max_total_cost := by sorry

end NUMINAMATH_CALUDE_team_improvements_minimum_days_team_a_l1318_131829


namespace NUMINAMATH_CALUDE_range_of_r_l1318_131856

-- Define the function r(x)
def r (x : ℝ) : ℝ := x^4 + 6*x^2 + 9 - 2*x

-- State the theorem
theorem range_of_r :
  ∀ y : ℝ, y ≥ 9 ↔ ∃ x : ℝ, x ≥ 0 ∧ r x = y :=
by sorry

end NUMINAMATH_CALUDE_range_of_r_l1318_131856


namespace NUMINAMATH_CALUDE_jace_road_trip_distance_l1318_131814

/-- Represents a driving segment with speed in miles per hour and duration in hours -/
structure DrivingSegment where
  speed : ℝ
  duration : ℝ

/-- Calculates the total distance covered given a list of driving segments -/
def totalDistance (segments : List DrivingSegment) : ℝ :=
  segments.foldl (fun acc segment => acc + segment.speed * segment.duration) 0

/-- Jace's road trip theorem -/
theorem jace_road_trip_distance :
  let segments : List DrivingSegment := [
    { speed := 50, duration := 3 },
    { speed := 65, duration := 4.5 },
    { speed := 60, duration := 2.75 },
    { speed := 75, duration := 1.8333 },
    { speed := 55, duration := 2.6667 }
  ]
  ∃ ε > 0, |totalDistance segments - 891.67| < ε :=
by sorry

end NUMINAMATH_CALUDE_jace_road_trip_distance_l1318_131814


namespace NUMINAMATH_CALUDE_luisa_apples_taken_l1318_131849

/-- Proves that Luisa took out 2 apples from the bag -/
theorem luisa_apples_taken (initial_apples initial_oranges initial_mangoes : ℕ)
  (remaining_fruits : ℕ) :
  initial_apples = 7 →
  initial_oranges = 8 →
  initial_mangoes = 15 →
  remaining_fruits = 14 →
  ∃ (apples_taken : ℕ),
    apples_taken + 2 * apples_taken + (2 * initial_mangoes / 3) =
      initial_apples + initial_oranges + initial_mangoes - remaining_fruits ∧
    apples_taken = 2 :=
by sorry

end NUMINAMATH_CALUDE_luisa_apples_taken_l1318_131849


namespace NUMINAMATH_CALUDE_problem_statement_l1318_131854

theorem problem_statement (x y : ℝ) 
  (h1 : 1/x + 1/y = 5)
  (h2 : x*y + 2*x + 2*y = 10) :
  x^2*y + x*y^2 = 500/121 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1318_131854


namespace NUMINAMATH_CALUDE_sundae_cost_l1318_131878

theorem sundae_cost (cherry_jubilee : ℝ) (peanut_butter : ℝ) (royal_banana : ℝ) 
  (tip_percentage : ℝ) (final_bill : ℝ) :
  cherry_jubilee = 9 →
  peanut_butter = 7.5 →
  royal_banana = 10 →
  tip_percentage = 0.2 →
  final_bill = 42 →
  ∃ (death_by_chocolate : ℝ),
    death_by_chocolate = 8.5 ∧
    (cherry_jubilee + peanut_butter + royal_banana + death_by_chocolate) * (1 + tip_percentage) = final_bill :=
by sorry

end NUMINAMATH_CALUDE_sundae_cost_l1318_131878


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_problem_l1318_131890

-- Define a monic quartic polynomial
def is_monic_quartic (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

-- Define the polynomial p with given conditions
def p : ℝ → ℝ := sorry

-- State the theorem
theorem monic_quartic_polynomial_problem :
  is_monic_quartic p ∧ 
  p 1 = 2 ∧ 
  p 2 = 7 ∧ 
  p 3 = 10 ∧ 
  p 4 = 17 → 
  p 5 = 26 := by sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_problem_l1318_131890


namespace NUMINAMATH_CALUDE_residue_calculation_l1318_131863

theorem residue_calculation : (207 * 13 - 18 * 8 + 5) % 16 = 8 := by
  sorry

end NUMINAMATH_CALUDE_residue_calculation_l1318_131863


namespace NUMINAMATH_CALUDE_new_person_weight_l1318_131813

theorem new_person_weight (n : ℕ) (initial_weight replaced_weight new_average : ℝ) :
  n = 10 ∧ 
  replaced_weight = 45 ∧
  new_average = initial_weight + 3 →
  (n * new_average - (n * initial_weight - replaced_weight)) = 75 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1318_131813


namespace NUMINAMATH_CALUDE_segment_sum_bound_l1318_131801

/-- A convex polygon in a 2D plane. -/
structure ConvexPolygon where
  -- We don't need to define the structure completely, just declare it
  area : ℝ

/-- A set of parallel lines in a 2D plane. -/
structure ParallelLines where
  -- Again, we don't need to define this completely
  count : ℕ
  spacing : ℝ

/-- The sum of lengths of segments cut by a polygon on parallel lines. -/
def sumOfSegments (polygon : ConvexPolygon) (lines : ParallelLines) : ℝ :=
  sorry -- Definition not provided, just declared

/-- Theorem statement -/
theorem segment_sum_bound
  (polygon : ConvexPolygon)
  (lines : ParallelLines)
  (h_area : polygon.area = 9)
  (h_lines : lines.count = 9)
  (h_spacing : lines.spacing = 1) :
  sumOfSegments polygon lines ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_segment_sum_bound_l1318_131801


namespace NUMINAMATH_CALUDE_white_balls_count_l1318_131818

theorem white_balls_count (total : ℕ) (p_yellow : ℚ) : 
  total = 10 → p_yellow = 6/10 → (total : ℚ) * (1 - p_yellow) = 4 := by sorry

end NUMINAMATH_CALUDE_white_balls_count_l1318_131818


namespace NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l1318_131800

/-- Represents the total number of athletes -/
def total_athletes : ℕ := 30

/-- Represents the number of male athletes -/
def male_athletes : ℕ := 20

/-- Represents the number of female athletes -/
def female_athletes : ℕ := 10

/-- Represents the sample size -/
def sample_size : ℕ := 6

/-- Represents the number of male athletes to be sampled -/
def male_sample_size : ℚ := (male_athletes : ℚ) * (sample_size : ℚ) / (total_athletes : ℚ)

theorem stratified_sampling_male_athletes :
  male_sample_size = 4 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l1318_131800


namespace NUMINAMATH_CALUDE_two_m_squared_eq_three_n_cubed_l1318_131816

theorem two_m_squared_eq_three_n_cubed (m n : ℕ+) :
  2 * m ^ 2 = 3 * n ^ 3 ↔ ∃ k : ℕ+, m = 18 * k ^ 3 ∧ n = 6 * k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_two_m_squared_eq_three_n_cubed_l1318_131816


namespace NUMINAMATH_CALUDE_first_positive_term_is_seventh_l1318_131827

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

theorem first_positive_term_is_seventh :
  let a₁ := -1
  let d := 1/5
  (∀ k < 7, arithmetic_sequence a₁ d k ≤ 0) ∧
  (arithmetic_sequence a₁ d 7 > 0) :=
by sorry

end NUMINAMATH_CALUDE_first_positive_term_is_seventh_l1318_131827


namespace NUMINAMATH_CALUDE_inclination_angle_of_line_l1318_131866

/-- The inclination angle of a line is the angle between the positive x-axis and the line, 
    measured counterclockwise. -/
def inclination_angle (a b c : ℝ) : ℝ := sorry

/-- The equation of the line is ax + by + c = 0 -/
def is_line_equation (a b c : ℝ) : Prop := sorry

theorem inclination_angle_of_line :
  let a : ℝ := 1
  let b : ℝ := 1
  let c : ℝ := -5
  is_line_equation a b c →
  inclination_angle a b c = 135 * (π / 180) := by sorry

end NUMINAMATH_CALUDE_inclination_angle_of_line_l1318_131866


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1318_131881

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℤ, 2 * X^6 - X^4 + 4 * X^2 - 7 = (X^2 + 4*X + 3) * q + (-704*X - 706) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1318_131881


namespace NUMINAMATH_CALUDE_a_value_in_set_l1318_131810

theorem a_value_in_set (A : Set ℝ) (a : ℝ) (h1 : A = {0, a, a^2})
  (h2 : 1 ∈ A) : a = -1 := by
  sorry

end NUMINAMATH_CALUDE_a_value_in_set_l1318_131810


namespace NUMINAMATH_CALUDE_union_of_intervals_l1318_131843

open Set

theorem union_of_intervals (M N : Set ℝ) : 
  M = {x : ℝ | -1 < x ∧ x < 3} → 
  N = {x : ℝ | x ≥ 1} → 
  M ∪ N = {x : ℝ | x > -1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_intervals_l1318_131843


namespace NUMINAMATH_CALUDE_f_non_monotonic_iff_l1318_131888

/-- A piecewise function f(x) depending on parameters a and t -/
noncomputable def f (a t x : ℝ) : ℝ :=
  if x ≤ t then (4*a - 3)*x + 2*a - 4 else 2*x^3 - 6*x

/-- The theorem stating the condition for f to be non-monotonic for all t -/
theorem f_non_monotonic_iff (a : ℝ) :
  (∀ t : ℝ, ¬Monotone (f a t)) ↔ a ≤ 3/4 := by sorry

end NUMINAMATH_CALUDE_f_non_monotonic_iff_l1318_131888


namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l1318_131812

theorem perfect_square_divisibility (a b : ℕ) 
  (h : (a^2 + b^2 + a) % (a * b) = 0) : 
  ∃ k : ℕ, a = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l1318_131812


namespace NUMINAMATH_CALUDE_stock_market_value_l1318_131886

theorem stock_market_value 
  (face_value : ℝ) 
  (dividend_rate : ℝ) 
  (market_yield : ℝ) 
  (h1 : dividend_rate = 0.07) 
  (h2 : market_yield = 0.10) : 
  (dividend_rate * face_value) / market_yield = 0.7 * face_value := by
  sorry

end NUMINAMATH_CALUDE_stock_market_value_l1318_131886


namespace NUMINAMATH_CALUDE_sqrt_64_minus_neg_2_cubed_equals_16_l1318_131899

theorem sqrt_64_minus_neg_2_cubed_equals_16 : 
  Real.sqrt 64 - (-2)^3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_64_minus_neg_2_cubed_equals_16_l1318_131899


namespace NUMINAMATH_CALUDE_number_of_boys_in_class_l1318_131859

/-- The number of boys in a class given specific height conditions -/
theorem number_of_boys_in_class (n : ℕ) 
  (h1 : (n : ℝ) * 182 = (n : ℝ) * 182 + 166 - 106)
  (h2 : (n : ℝ) * 180 = (n : ℝ) * 182 + 106 - 166) : n = 30 := by
  sorry


end NUMINAMATH_CALUDE_number_of_boys_in_class_l1318_131859


namespace NUMINAMATH_CALUDE_bookstore_problem_l1318_131805

theorem bookstore_problem (total_books : ℕ) (unsold_books : ℕ) (customers : ℕ) :
  total_books = 40 →
  unsold_books = 4 →
  customers = 4 →
  (total_books - unsold_books) % customers = 0 →
  (total_books - unsold_books) / customers = 9 :=
by sorry

end NUMINAMATH_CALUDE_bookstore_problem_l1318_131805


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l1318_131880

/-- Given 6 moles of a compound with a total weight of 252 grams, 
    the molecular weight of the compound is 42 grams/mole. -/
theorem molecular_weight_calculation (moles : ℝ) (total_weight : ℝ) :
  moles = 6 →
  total_weight = 252 →
  total_weight / moles = 42 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l1318_131880


namespace NUMINAMATH_CALUDE_car_speed_l1318_131893

/-- Represents a kilometer marker with two digits -/
structure Marker where
  tens : ℕ
  ones : ℕ
  h_digits : tens < 10 ∧ ones < 10

/-- Represents the time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : hours < 24 ∧ minutes < 60

/-- Represents an observation of a marker at a specific time -/
structure Observation where
  time : Time
  marker : Marker

def speed_kmh (start_obs end_obs : Observation) : ℚ :=
  let time_diff := (end_obs.time.hours - start_obs.time.hours : ℚ) + 
                   ((end_obs.time.minutes - start_obs.time.minutes : ℚ) / 60)
  let distance := (end_obs.marker.tens * 10 + end_obs.marker.ones) - 
                  (start_obs.marker.tens * 10 + start_obs.marker.ones)
  distance / time_diff

theorem car_speed 
  (obs1 obs2 obs3 : Observation)
  (h_time1 : obs1.time = ⟨12, 0, by norm_num⟩)
  (h_time2 : obs2.time = ⟨12, 42, by norm_num⟩)
  (h_time3 : obs3.time = ⟨13, 0, by norm_num⟩)
  (h_marker1 : obs1.marker = ⟨obs1.marker.tens, obs1.marker.ones, by sorry⟩)
  (h_marker2 : obs2.marker = ⟨obs1.marker.ones, obs1.marker.tens, by sorry⟩)
  (h_marker3 : obs3.marker = ⟨obs1.marker.tens, obs1.marker.ones, by sorry⟩)
  (h_constant_speed : speed_kmh obs1 obs2 = speed_kmh obs2 obs3) :
  speed_kmh obs1 obs3 = 90 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_l1318_131893


namespace NUMINAMATH_CALUDE_third_student_number_l1318_131845

theorem third_student_number (A B C D : ℤ) 
  (sum_eq : A + B + C + D = 531)
  (diff_eq : A + B = C + D + 31)
  (third_fourth_diff : C = D + 22) :
  C = 136 := by
  sorry

end NUMINAMATH_CALUDE_third_student_number_l1318_131845


namespace NUMINAMATH_CALUDE_negation_equivalence_l1318_131808

theorem negation_equivalence (a b c : ℝ) :
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1318_131808


namespace NUMINAMATH_CALUDE_no_same_line_l1318_131850

/-- Two lines are the same if and only if they have the same slope and y-intercept -/
def same_line (m1 m2 b1 b2 : ℝ) : Prop := m1 = m2 ∧ b1 = b2

/-- The first line equation: ax + 3y + d = 0 -/
def line1 (a d : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + d = 0

/-- The second line equation: 4x - ay + 8 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := 4 * x - a * y + 8 = 0

/-- Theorem: There are no real values of a and d such that ax+3y+d=0 and 4x-ay+8=0 represent the same line -/
theorem no_same_line : ¬∃ (a d : ℝ), ∀ (x y : ℝ), line1 a d x y ↔ line2 a x y := by
  sorry

end NUMINAMATH_CALUDE_no_same_line_l1318_131850


namespace NUMINAMATH_CALUDE_andrew_toast_count_l1318_131887

/-- The cost of breakfast for Dale and Andrew -/
def total_cost : ℕ := 15

/-- The cost of a slice of toast -/
def toast_cost : ℕ := 1

/-- The cost of an egg -/
def egg_cost : ℕ := 3

/-- The number of slices of toast Dale had -/
def dale_toast : ℕ := 2

/-- The number of eggs Dale had -/
def dale_eggs : ℕ := 2

/-- The number of eggs Andrew had -/
def andrew_eggs : ℕ := 2

/-- The number of slices of toast Andrew had -/
def andrew_toast : ℕ := 1

theorem andrew_toast_count :
  total_cost = 
    dale_toast * toast_cost + dale_eggs * egg_cost + 
    andrew_toast * toast_cost + andrew_eggs * egg_cost :=
by sorry

end NUMINAMATH_CALUDE_andrew_toast_count_l1318_131887


namespace NUMINAMATH_CALUDE_carl_driving_hours_l1318_131837

/-- 
Given Carl's initial daily driving hours and additional weekly hours after promotion, 
prove that the total hours he will drive in two weeks is equal to 40 hours.
-/
theorem carl_driving_hours (initial_daily_hours : ℝ) (additional_weekly_hours : ℝ) : 
  initial_daily_hours = 2 ∧ additional_weekly_hours = 6 → 
  (initial_daily_hours * 7 + additional_weekly_hours) * 2 = 40 := by
sorry

end NUMINAMATH_CALUDE_carl_driving_hours_l1318_131837


namespace NUMINAMATH_CALUDE_sum_in_base6_l1318_131840

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- The main theorem --/
theorem sum_in_base6 :
  let a := base6ToBase10 [4, 3, 2, 1]  -- 1234₆
  let b := base6ToBase10 [4, 5, 6]     -- 654₆
  let c := base6ToBase10 [2, 1]        -- 12₆
  base10ToBase6 (a + b + c) = [4, 4, 3, 2] -- 2344₆
:= by sorry

end NUMINAMATH_CALUDE_sum_in_base6_l1318_131840


namespace NUMINAMATH_CALUDE_product_sum_fractions_l1318_131807

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l1318_131807


namespace NUMINAMATH_CALUDE_system_solution_existence_l1318_131836

theorem system_solution_existence (k : ℝ) :
  (∃ (x y : ℝ), y = k * x + 4 ∧ y = (3 * k - 2) * x + 5) ↔ k ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_existence_l1318_131836


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1318_131889

-- Define the triangle sides
def a : ℝ := 10
def b : ℝ := 6
def c : ℝ := 7

-- Define the perimeter
def perimeter : ℝ := a + b + c

-- Theorem statement
theorem triangle_perimeter : perimeter = 23 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1318_131889


namespace NUMINAMATH_CALUDE_michael_crates_tuesday_l1318_131875

/-- The number of crates Michael bought on Tuesday -/
def T : ℕ := sorry

/-- The number of crates Michael gave out -/
def crates_given_out : ℕ := 2

/-- The number of crates Michael bought on Thursday -/
def crates_bought_thursday : ℕ := 5

/-- The number of eggs each crate holds -/
def eggs_per_crate : ℕ := 30

/-- The total number of eggs Michael has now -/
def total_eggs : ℕ := 270

theorem michael_crates_tuesday : T = 6 := by
  sorry

end NUMINAMATH_CALUDE_michael_crates_tuesday_l1318_131875


namespace NUMINAMATH_CALUDE_complement_A_union_B_eq_greater_than_neg_one_l1318_131831

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}
def B : Set ℝ := {x | x ≥ 1}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- State the theorem
theorem complement_A_union_B_eq_greater_than_neg_one :
  (Set.compl A) ∪ B = {x : ℝ | x > -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_eq_greater_than_neg_one_l1318_131831


namespace NUMINAMATH_CALUDE_smallest_difference_is_one_l1318_131869

/-- Represents a triangle with integer side lengths -/
structure IntegerTriangle where
  de : ℕ
  ef : ℕ
  df : ℕ

/-- Checks if the given side lengths form a valid triangle -/
def is_valid_triangle (t : IntegerTriangle) : Prop :=
  t.de + t.ef > t.df ∧ t.de + t.df > t.ef ∧ t.ef + t.df > t.de

/-- Theorem: The smallest possible difference between EF and DE in the given conditions is 1 -/
theorem smallest_difference_is_one :
  ∃ (t : IntegerTriangle),
    t.de + t.ef + t.df = 3005 ∧
    t.de < t.ef ∧
    t.ef ≤ t.df ∧
    is_valid_triangle t ∧
    (∀ (u : IntegerTriangle),
      u.de + u.ef + u.df = 3005 →
      u.de < u.ef →
      u.ef ≤ u.df →
      is_valid_triangle u →
      u.ef - u.de ≥ t.ef - t.de) ∧
    t.ef - t.de = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_difference_is_one_l1318_131869


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l1318_131885

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (angle_a angle_b angle_c : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (positive_angles : 0 < angle_a ∧ 0 < angle_b ∧ 0 < angle_c)
  (angle_sum : angle_a + angle_b + angle_c = Real.pi)
  (law_of_sines : a / Real.sin angle_a = b / Real.sin angle_b)

-- State the theorem
theorem triangle_angle_relation (t : Triangle) 
  (h : t.angle_a = 3 * t.angle_b) :
  (t.a^2 - t.b^2) * (t.a - t.b) = t.b * t.c^2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_relation_l1318_131885


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1318_131853

/-- Time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (bridge_length : ℝ) : 
  train_length = 110 → 
  train_speed_kmph = 36 → 
  bridge_length = 170 → 
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 28 := by
  sorry


end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1318_131853


namespace NUMINAMATH_CALUDE_max_submerged_cubes_is_five_l1318_131835

/-- Represents the properties of the cylinder and cubes -/
structure CylinderAndCubes where
  cylinder_diameter : ℝ
  initial_water_height : ℝ
  cube_edge_length : ℝ

/-- Calculates the maximum number of cubes that can be submerged -/
def max_submerged_cubes (props : CylinderAndCubes) : ℕ :=
  -- Implementation details omitted
  sorry

/-- The main theorem stating the maximum number of submerged cubes -/
theorem max_submerged_cubes_is_five (props : CylinderAndCubes) 
  (h1 : props.cylinder_diameter = 2.9)
  (h2 : props.initial_water_height = 4)
  (h3 : props.cube_edge_length = 2) :
  max_submerged_cubes props = 5 := by
  sorry

#check max_submerged_cubes_is_five

end NUMINAMATH_CALUDE_max_submerged_cubes_is_five_l1318_131835


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1318_131872

/-- Given that x and y are inversely proportional, prove that when x + y = 60, x = 3y, 
    and x = -6, then y = -112.5 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) : 
  (x * y = k) →  -- x and y are inversely proportional
  (x + y = 60) →  -- sum condition
  (x = 3 * y) →  -- proportion condition
  (x = -6) →  -- given x value
  y = -112.5 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1318_131872


namespace NUMINAMATH_CALUDE_number_puzzle_l1318_131882

theorem number_puzzle : ∃! (N : ℕ), N > 0 ∧ ∃ (Q : ℕ), N = 11 * Q ∧ Q + N + 11 = 71 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1318_131882


namespace NUMINAMATH_CALUDE_rectangle_circle_square_area_l1318_131884

theorem rectangle_circle_square_area :
  ∀ (rectangle_length rectangle_breadth rectangle_area circle_radius square_side : ℝ),
    rectangle_length = 5 * circle_radius →
    rectangle_breadth = 11 →
    rectangle_area = 220 →
    rectangle_area = rectangle_length * rectangle_breadth →
    circle_radius = square_side →
    square_side ^ 2 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_square_area_l1318_131884


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l1318_131895

/-- Given three identical circles with circumference 48, where each circle touches the other two,
    and the arcs in the shaded region each subtend an angle of 90 degrees at the center of their
    respective circles, the perimeter of the shaded region is equal to 36. -/
theorem shaded_region_perimeter (circle_circumference : ℝ) (arc_angle : ℝ) : 
  circle_circumference = 48 → 
  arc_angle = 90 →
  (3 * (arc_angle / 360) * circle_circumference) = 36 := by
  sorry

#check shaded_region_perimeter

end NUMINAMATH_CALUDE_shaded_region_perimeter_l1318_131895


namespace NUMINAMATH_CALUDE_remaining_integers_l1318_131847

theorem remaining_integers (T : Finset ℕ) : 
  T = Finset.range 100 →
  (Finset.filter (fun n => ¬(n % 2 = 0 ∨ n % 3 = 0 ∨ n % 5 = 0)) T).card = 26 :=
by sorry

end NUMINAMATH_CALUDE_remaining_integers_l1318_131847


namespace NUMINAMATH_CALUDE_document_word_count_l1318_131824

/-- Calculates the number of words in a document based on typing speed and time --/
def document_words (original_speed : ℕ) (speed_reduction : ℕ) (typing_time : ℕ) : ℕ :=
  (original_speed - speed_reduction) * typing_time

/-- Proves that the number of words in the document is 810 --/
theorem document_word_count : document_words 65 20 18 = 810 := by
  sorry

end NUMINAMATH_CALUDE_document_word_count_l1318_131824


namespace NUMINAMATH_CALUDE_pyramid_edge_length_l1318_131848

/-- A pyramid with 8 edges of equal length -/
structure Pyramid where
  edge_count : ℕ
  edge_length : ℝ
  total_length : ℝ
  edge_count_eq : edge_count = 8
  total_eq : total_length = edge_count * edge_length

/-- Theorem: If a pyramid has 8 edges of equal length, and the sum of all edges is 14.8 meters,
    then the length of each edge is 1.85 meters. -/
theorem pyramid_edge_length (p : Pyramid) (h : p.total_length = 14.8) :
  p.edge_length = 1.85 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_edge_length_l1318_131848


namespace NUMINAMATH_CALUDE_complement_of_P_l1318_131825

def U : Set ℝ := Set.univ

def P : Set ℝ := {x : ℝ | x^2 ≤ 1}

theorem complement_of_P : (Set.univ \ P) = {x : ℝ | x < -1 ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_P_l1318_131825


namespace NUMINAMATH_CALUDE_factorial_ratio_l1318_131892

theorem factorial_ratio : Nat.factorial 52 / Nat.factorial 50 = 2652 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1318_131892


namespace NUMINAMATH_CALUDE_point_on_line_l1318_131809

/-- A point (x, y) lies on a line passing through two points (x₁, y₁) and (x₂, y₂) if it satisfies the equation of the line. -/
def lies_on_line (x y x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y - y₁ = ((y₂ - y₁) / (x₂ - x₁)) * (x - x₁)

/-- The point (0,3) lies on the line passing through (-2,1) and (2,5). -/
theorem point_on_line : lies_on_line 0 3 (-2) 1 2 5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1318_131809
