import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l566_56683

theorem problem_solution (x y m n a b : ℝ) : 
  x = (Real.sqrt 3 - 1) / 2 →
  y = (Real.sqrt 3 + 1) / 2 →
  m = 1 / x - 1 / y →
  n = y / x + x / y →
  Real.sqrt a - Real.sqrt b = n + 2 →
  Real.sqrt (a * b) = m →
  m = 2 ∧ n = 4 ∧ Real.sqrt a + Real.sqrt b = 2 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l566_56683


namespace NUMINAMATH_CALUDE_sqrt_40000_l566_56679

theorem sqrt_40000 : Real.sqrt 40000 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_40000_l566_56679


namespace NUMINAMATH_CALUDE_rajs_house_area_l566_56635

/-- The total area of Raj's house given the specified room dimensions and counts -/
theorem rajs_house_area : 
  let bedroom_count : ℕ := 4
  let bedroom_side : ℕ := 11
  let bathroom_count : ℕ := 2
  let bathroom_length : ℕ := 8
  let bathroom_width : ℕ := 6
  let kitchen_area : ℕ := 265
  
  bedroom_count * (bedroom_side * bedroom_side) +
  bathroom_count * (bathroom_length * bathroom_width) +
  kitchen_area +
  kitchen_area = 1110 := by
sorry

end NUMINAMATH_CALUDE_rajs_house_area_l566_56635


namespace NUMINAMATH_CALUDE_cans_per_box_is_ten_l566_56676

/-- The number of cans in each box of soda at a family reunion --/
def cans_per_box (people : ℕ) (cans_per_person : ℕ) (box_cost : ℕ) 
                 (family_members : ℕ) (cost_per_member : ℕ) : ℕ :=
  let total_people := 5 * 12  -- five dozens
  let total_cans := total_people * cans_per_person
  let total_cost := family_members * cost_per_member
  let num_boxes := total_cost / box_cost
  total_cans / num_boxes

/-- Theorem stating that the number of cans per box is 10 --/
theorem cans_per_box_is_ten : 
  cans_per_box 60 2 2 6 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cans_per_box_is_ten_l566_56676


namespace NUMINAMATH_CALUDE_greatest_product_of_digits_divisible_by_35_l566_56662

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  tens_single_digit : tens < 10
  units_single_digit : units < 10

/-- Check if a number is divisible by another number -/
def isDivisibleBy (n m : Nat) : Prop := ∃ k, n = m * k

theorem greatest_product_of_digits_divisible_by_35 :
  ∀ n : TwoDigitNumber,
    isDivisibleBy (10 * n.tens + n.units) 35 →
    ∀ m : TwoDigitNumber,
      isDivisibleBy (10 * m.tens + m.units) 35 →
      n.units * n.tens ≤ 40 ∧
      (m.units * m.tens = 40 → n.units * n.tens = 40) :=
sorry

end NUMINAMATH_CALUDE_greatest_product_of_digits_divisible_by_35_l566_56662


namespace NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l566_56653

/-- The equation x^2 - xy - 6y^2 = 0 represents a pair of straight lines -/
theorem equation_represents_pair_of_lines : ∃ (m₁ m₂ : ℝ),
  ∀ (x y : ℝ), x^2 - x*y - 6*y^2 = 0 ↔ (x = m₁*y ∨ x = m₂*y) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l566_56653


namespace NUMINAMATH_CALUDE_floor_equation_solution_l566_56674

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊3*x⌋ - 1/3⌋ = ⌊x + 3⌋ ↔ 4/3 ≤ x ∧ x < 5/3 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l566_56674


namespace NUMINAMATH_CALUDE_robot_number_difference_l566_56634

def largest_three_digit (a b c : Nat) : Nat :=
  100 * max a (max b c) + 10 * max (min (max a b) (max b c)) (min a (min b c)) + min a (min b c)

def smallest_three_digit (a b c : Nat) : Nat :=
  if min a (min b c) = 0
  then 100 * min (max a b) (max b c) + 10 * max (min a (min b c)) (min (max a b) (max b c)) + 0
  else 100 * min a (min b c) + 10 * min (max a b) (max b c) + max a (max b c)

theorem robot_number_difference :
  largest_three_digit 2 3 5 - smallest_three_digit 4 0 6 = 126 := by
  sorry

end NUMINAMATH_CALUDE_robot_number_difference_l566_56634


namespace NUMINAMATH_CALUDE_angela_puzzle_palace_spending_l566_56639

/-- The amount of money Angela got to spend at Puzzle Palace -/
def total_amount : ℕ := sorry

/-- The amount of money Angela spent at Puzzle Palace -/
def amount_spent : ℕ := 78

/-- The amount of money Angela had left after shopping -/
def amount_left : ℕ := 12

/-- Theorem stating that the total amount Angela got to spend at Puzzle Palace is $90 -/
theorem angela_puzzle_palace_spending :
  total_amount = amount_spent + amount_left :=
sorry

end NUMINAMATH_CALUDE_angela_puzzle_palace_spending_l566_56639


namespace NUMINAMATH_CALUDE_parabola_tangent_to_circle_l566_56698

/-- Given a parabola and a circle, if the parabola's axis is tangent to the circle, 
    then the parameter p of the parabola equals 2. -/
theorem parabola_tangent_to_circle (p : ℝ) (h1 : p > 0) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- parabola equation
  (∀ x y : ℝ, x^2 + y^2 - 8*x - 9 = 0) →  -- circle equation
  (∃ x : ℝ, x = -p/2 ∧ (x-4)^2 = 25) →  -- parabola's axis is tangent to the circle
  p = 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_circle_l566_56698


namespace NUMINAMATH_CALUDE_pastry_combinations_l566_56654

/-- The number of ways to distribute n indistinguishable items into k distinguishable bins -/
def combinations_with_repetition (n k : ℕ) : ℕ := 
  Nat.choose (n + k - 1) k

/-- The number of pastry types available -/
def num_pastry_types : ℕ := 3

/-- The total number of pastries to be bought -/
def total_pastries : ℕ := 9

/-- Theorem stating that the number of ways to buy 9 pastries from 3 types is 55 -/
theorem pastry_combinations : 
  combinations_with_repetition total_pastries num_pastry_types = 55 := by
  sorry

end NUMINAMATH_CALUDE_pastry_combinations_l566_56654


namespace NUMINAMATH_CALUDE_exponential_function_condition_l566_56692

theorem exponential_function_condition (x₁ x₂ : ℝ) :
  (x₁ + x₂ > 0) ↔ ((1/2 : ℝ)^x₁ * (1/2 : ℝ)^x₂ < 1) := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_condition_l566_56692


namespace NUMINAMATH_CALUDE_am_gm_difference_bound_l566_56672

theorem am_gm_difference_bound (a : ℝ) (h : 0 < a) :
  let b := a + 1
  let am := (a + b) / 2
  let gm := Real.sqrt (a * b)
  am - gm < (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_am_gm_difference_bound_l566_56672


namespace NUMINAMATH_CALUDE_second_section_has_180_cars_l566_56678

-- Define the given information
def section_g_rows : ℕ := 15
def section_g_cars_per_row : ℕ := 10
def second_section_rows : ℕ := 20
def nate_cars_per_minute : ℕ := 11
def nate_search_time : ℕ := 30

-- Define the total number of cars Nate walked past
def total_cars_walked : ℕ := nate_cars_per_minute * nate_search_time

-- Define the number of cars in Section G
def section_g_cars : ℕ := section_g_rows * section_g_cars_per_row

-- Define the number of cars in the second section
def second_section_cars : ℕ := total_cars_walked - section_g_cars

-- Theorem to prove
theorem second_section_has_180_cars :
  second_section_cars = 180 :=
sorry

end NUMINAMATH_CALUDE_second_section_has_180_cars_l566_56678


namespace NUMINAMATH_CALUDE_simplify_expression_l566_56670

theorem simplify_expression (a b : ℝ) (h1 : 2*b - a < 3) (h2 : 2*a - b < 5) :
  -|2*b - a - 7| - |b - 2*a + 8| + |a + b - 9| = -6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l566_56670


namespace NUMINAMATH_CALUDE_turner_amusement_park_tickets_l566_56606

/-- Calculates the total number of tickets needed for a multi-day amusement park visit -/
def total_tickets (days : ℕ) 
                  (rollercoaster_rides_per_day : ℕ) 
                  (catapult_rides_per_day : ℕ) 
                  (ferris_wheel_rides_per_day : ℕ) 
                  (rollercoaster_tickets_per_ride : ℕ) 
                  (catapult_tickets_per_ride : ℕ) 
                  (ferris_wheel_tickets_per_ride : ℕ) : ℕ :=
  days * (rollercoaster_rides_per_day * rollercoaster_tickets_per_ride +
          catapult_rides_per_day * catapult_tickets_per_ride +
          ferris_wheel_rides_per_day * ferris_wheel_tickets_per_ride)

theorem turner_amusement_park_tickets : 
  total_tickets 3 3 2 1 4 4 1 = 63 := by
  sorry

end NUMINAMATH_CALUDE_turner_amusement_park_tickets_l566_56606


namespace NUMINAMATH_CALUDE_quadratic_equation_single_solution_sum_l566_56644

theorem quadratic_equation_single_solution_sum (b : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ 3 * x^2 + b * x + 6 * x + 10
  (∃! x, f x = 0) → 
  ∃ b₁ b₂, b = b₁ ∨ b = b₂ ∧ b₁ + b₂ = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_single_solution_sum_l566_56644


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_21_19_l566_56607

theorem half_abs_diff_squares_21_19 : (1/2 : ℝ) * |21^2 - 19^2| = 40 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_21_19_l566_56607


namespace NUMINAMATH_CALUDE_expression_simplification_l566_56619

theorem expression_simplification (a b : ℝ) (h1 : a = Real.sqrt 3 - 3) (h2 : b = 3) :
  1 - (a - b) / (a + 2*b) / ((a^2 - b^2) / (a^2 + 4*a*b + 4*b^2)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l566_56619


namespace NUMINAMATH_CALUDE_greatest_common_piece_length_l566_56610

theorem greatest_common_piece_length :
  let rope_lengths : List Nat := [45, 60, 75, 90]
  Nat.gcd (Nat.gcd (Nat.gcd 45 60) 75) 90 = 15 := by sorry

end NUMINAMATH_CALUDE_greatest_common_piece_length_l566_56610


namespace NUMINAMATH_CALUDE_solve_lawn_mowing_problem_l566_56628

/-- Edward's lawn mowing business finances -/
def lawn_mowing_problem (spring_earnings summer_earnings final_amount : ℕ) : Prop :=
  let total_earnings := spring_earnings + summer_earnings
  let supplies_cost := total_earnings - final_amount
  supplies_cost = total_earnings - final_amount

theorem solve_lawn_mowing_problem :
  lawn_mowing_problem 2 27 24 = true :=
by sorry

end NUMINAMATH_CALUDE_solve_lawn_mowing_problem_l566_56628


namespace NUMINAMATH_CALUDE_paramon_solomon_meeting_time_l566_56637

/- Define the total distance between A and B -/
variable (S : ℝ) (S_pos : S > 0)

/- Define the speeds of Paramon, Solomon, and Agafon -/
variable (x y z : ℝ) (x_pos : x > 0) (y_pos : y > 0) (z_pos : z > 0)

/- Define the time when Paramon and Solomon meet -/
def meeting_time : ℝ := 1

/- Theorem stating that Paramon and Solomon meet at 13:00 (1 hour after 12:00) -/
theorem paramon_solomon_meeting_time :
  (S / (2 * x) = 1) ∧                   /- Paramon travels half the distance in 1 hour -/
  (2 * z = S / 2 + 2 * x) ∧             /- Agafon catches up with Paramon at 14:00 -/
  (4 / 3 * (y + z) = S) ∧               /- Agafon meets Solomon at 13:20 -/
  (S / 2 + x * meeting_time = y * meeting_time) /- Paramon and Solomon meet -/
  → meeting_time = 1 := by sorry

end NUMINAMATH_CALUDE_paramon_solomon_meeting_time_l566_56637


namespace NUMINAMATH_CALUDE_difference_of_trailing_zeros_l566_56618

def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem difference_of_trailing_zeros : trailingZeros 300 - trailingZeros 280 = 5 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_trailing_zeros_l566_56618


namespace NUMINAMATH_CALUDE_distance_from_p_to_ad_l566_56699

/-- Square with side length 6 -/
structure Square :=
  (side : ℝ)
  (is_six : side = 6)

/-- Point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Circle in 2D space -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Given a square ABCD, find the distance from point P to side AD, where P is an intersection
    point of two circles: one centered at M (midpoint of CD) with radius 3, and another centered
    at A with radius 5. -/
def distance_to_side (s : Square) : ℝ :=
  let a := Point.mk 0 s.side
  let d := Point.mk 0 0
  let m := Point.mk (s.side / 2) 0
  let circle_m := Circle.mk m 3
  let circle_a := Circle.mk a 5
  -- The actual calculation of the distance would go here
  sorry

/-- The theorem stating that the distance from P to AD is equal to some specific value -/
theorem distance_from_p_to_ad (s : Square) : ∃ x : ℝ, distance_to_side s = x :=
  sorry

end NUMINAMATH_CALUDE_distance_from_p_to_ad_l566_56699


namespace NUMINAMATH_CALUDE_import_tax_calculation_l566_56660

theorem import_tax_calculation (total_value : ℝ) (tax_rate : ℝ) (threshold : ℝ) (tax_amount : ℝ) : 
  total_value = 2580 →
  tax_rate = 0.07 →
  threshold = 1000 →
  tax_amount = (total_value - threshold) * tax_rate →
  tax_amount = 110.60 := by
sorry

end NUMINAMATH_CALUDE_import_tax_calculation_l566_56660


namespace NUMINAMATH_CALUDE_multiply_72519_9999_l566_56694

theorem multiply_72519_9999 : 72519 * 9999 = 725117481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72519_9999_l566_56694


namespace NUMINAMATH_CALUDE_triangle_side_length_l566_56640

theorem triangle_side_length (a b : ℝ) (A B : ℝ) : 
  a = 4 →
  A = π / 3 →  -- 60° in radians
  B = π / 4 →  -- 45° in radians
  b = (4 * Real.sqrt 6) / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l566_56640


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l566_56608

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^2 - 3*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                                a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l566_56608


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l566_56686

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (4 + Real.sqrt (3 * y - 7)) = 3 → y = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l566_56686


namespace NUMINAMATH_CALUDE_original_number_proof_l566_56646

theorem original_number_proof (e : ℝ) : 
  (e + 0.125 * e) - (e - 0.25 * e) = 30 → e = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l566_56646


namespace NUMINAMATH_CALUDE_max_f_sum_l566_56695

/-- A permutation of 4n letters consisting of n occurrences each of A, B, C, and D -/
def Permutation (n : ℕ) := Fin (4 * n) → Fin 4

/-- The number of B's to the right of each A in the permutation -/
def f_AB (σ : Permutation n) : ℕ := sorry

/-- The number of C's to the right of each B in the permutation -/
def f_BC (σ : Permutation n) : ℕ := sorry

/-- The number of D's to the right of each C in the permutation -/
def f_CD (σ : Permutation n) : ℕ := sorry

/-- The number of A's to the right of each D in the permutation -/
def f_DA (σ : Permutation n) : ℕ := sorry

/-- The sum of f_AB, f_BC, f_CD, and f_DA for a given permutation -/
def f_sum (σ : Permutation n) : ℕ := f_AB σ + f_BC σ + f_CD σ + f_DA σ

theorem max_f_sum (n : ℕ) : (∀ σ : Permutation n, f_sum σ ≤ 3 * n^2) ∧ (∃ σ : Permutation n, f_sum σ = 3 * n^2) := by sorry

end NUMINAMATH_CALUDE_max_f_sum_l566_56695


namespace NUMINAMATH_CALUDE_square_sum_plus_quadruple_product_l566_56687

theorem square_sum_plus_quadruple_product (x y : ℝ) 
  (h1 : x + y = 8) (h2 : x * y = 15) : 
  x^2 + 6*x*y + y^2 = 124 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_plus_quadruple_product_l566_56687


namespace NUMINAMATH_CALUDE_closest_to_division_l566_56659

def options : List ℝ := [0.2, 2, 20, 200, 2000]

theorem closest_to_division (x y : ℝ) (h1 : y ≠ 0) :
  ∃ z ∈ options, ∀ w ∈ options, |x / y - z| ≤ |x / y - w| :=
sorry

end NUMINAMATH_CALUDE_closest_to_division_l566_56659


namespace NUMINAMATH_CALUDE_arthur_walking_distance_l566_56668

/-- Calculates the total distance walked given the number of blocks and block length -/
def total_distance (blocks_south : ℕ) (blocks_west : ℕ) (block_length : ℚ) : ℚ :=
  (blocks_south + blocks_west : ℚ) * block_length

/-- Theorem: Arthur's total walking distance is 4.5 miles -/
theorem arthur_walking_distance :
  let blocks_south : ℕ := 8
  let blocks_west : ℕ := 10
  let block_length : ℚ := 1/4
  total_distance blocks_south blocks_west block_length = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walking_distance_l566_56668


namespace NUMINAMATH_CALUDE_circle_symmetry_theorem_l566_56609

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 4*y = 0

-- Define the line of symmetry
def symmetry_line (k b : ℝ) (x y : ℝ) : Prop := y = k*x + b

-- Define the symmetric circle centered at the origin
def symmetric_circle (x y : ℝ) : Prop := x^2 + y^2 = 20

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
  symmetric_circle A.1 A.2 ∧ symmetric_circle B.1 B.2

-- Define the angle ACB
def angle_ACB (A B : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem circle_symmetry_theorem :
  ∃ (k b : ℝ) (A B : ℝ × ℝ),
    (∀ x y : ℝ, circle_C x y ↔ symmetric_circle (2*x - k*y + b) (2*y + k*x - k*b)) →
    k = 2 ∧ b = 5 ∧
    intersection_points A B ∧
    angle_ACB A B = 120 := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_theorem_l566_56609


namespace NUMINAMATH_CALUDE_card_collection_solution_l566_56689

/-- Represents the card collection problem --/
structure CardCollection where
  total_cards : Nat
  damaged_cards : Nat
  full_box_capacity : Nat
  damaged_box_capacity : Nat

/-- Calculates the number of cards in the last partially filled box of undamaged cards --/
def last_box_count (cc : CardCollection) : Nat :=
  (cc.total_cards - cc.damaged_cards) % cc.full_box_capacity

/-- Theorem stating the solution to the card collection problem --/
theorem card_collection_solution (cc : CardCollection) 
  (h1 : cc.total_cards = 120)
  (h2 : cc.damaged_cards = 18)
  (h3 : cc.full_box_capacity = 10)
  (h4 : cc.damaged_box_capacity = 5) :
  last_box_count cc = 2 := by
  sorry

#eval last_box_count { total_cards := 120, damaged_cards := 18, full_box_capacity := 10, damaged_box_capacity := 5 }

end NUMINAMATH_CALUDE_card_collection_solution_l566_56689


namespace NUMINAMATH_CALUDE_positive_real_solution_floor_product_l566_56615

theorem positive_real_solution_floor_product (x : ℝ) : 
  x > 0 → x * ⌊x⌋ = 72 → x = 9 := by sorry

end NUMINAMATH_CALUDE_positive_real_solution_floor_product_l566_56615


namespace NUMINAMATH_CALUDE_usual_walking_time_l566_56617

/-- Given a constant distance and the fact that walking at 40% of usual speed takes 24 minutes more, 
    the usual time to cover the distance is 16 minutes. -/
theorem usual_walking_time (distance : ℝ) (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0)
  (h2 : usual_time > 0)
  (h3 : distance = usual_speed * usual_time)
  (h4 : distance = (0.4 * usual_speed) * (usual_time + 24)) :
  usual_time = 16 := by
sorry

end NUMINAMATH_CALUDE_usual_walking_time_l566_56617


namespace NUMINAMATH_CALUDE_factorial_of_factorial_l566_56648

theorem factorial_of_factorial (n : ℕ) : (n.factorial.factorial) / n.factorial = (n.factorial - 1).factorial := by
  sorry

end NUMINAMATH_CALUDE_factorial_of_factorial_l566_56648


namespace NUMINAMATH_CALUDE_johns_final_push_time_l566_56625

/-- The time of John's final push in a race, given the initial and final distances between
    John and Steve, and their respective speeds. -/
theorem johns_final_push_time 
  (initial_distance : ℝ) 
  (john_speed : ℝ) 
  (steve_speed : ℝ) 
  (final_distance : ℝ) : 
  initial_distance = 16 →
  john_speed = 4.2 →
  steve_speed = 3.7 →
  final_distance = 2 →
  ∃ t : ℝ, t = 15 / 7 ∧ john_speed * t = initial_distance + final_distance :=
by
  sorry

#check johns_final_push_time

end NUMINAMATH_CALUDE_johns_final_push_time_l566_56625


namespace NUMINAMATH_CALUDE_abc_product_l566_56632

theorem abc_product (a b c : ℕ+) 
  (h1 : a * b = 13)
  (h2 : b * c = 52)
  (h3 : c * a = 4) :
  a * b * c = 52 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l566_56632


namespace NUMINAMATH_CALUDE_expression_equals_75_l566_56671

-- Define the expression
def expression : ℚ := 150 / (10 / 5)

-- State the theorem
theorem expression_equals_75 : expression = 75 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_75_l566_56671


namespace NUMINAMATH_CALUDE_max_value_expression_l566_56641

theorem max_value_expression (x y : ℝ) :
  (3 * x + 4 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 50 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l566_56641


namespace NUMINAMATH_CALUDE_g_neg_one_value_l566_56645

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of y = f(x) + x^2 being an odd function
def is_odd_composite (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) + (-x)^2 = -(f x + x^2)

-- Define g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- State the theorem
theorem g_neg_one_value (f : ℝ → ℝ) 
  (h1 : is_odd_composite f) 
  (h2 : f 1 = 1) : 
  g f (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_one_value_l566_56645


namespace NUMINAMATH_CALUDE_cannot_form_70_cents_l566_56658

/-- Represents the types of coins available in the piggy bank -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents -/
def coinValue (c : Coin) : Nat :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Represents a combination of coins -/
def CoinCombination := List Coin

/-- Calculates the total value of a coin combination in cents -/
def totalValue (comb : CoinCombination) : Nat :=
  comb.map coinValue |>.sum

/-- Predicate to check if a coin combination has exactly six coins -/
def hasSixCoins (comb : CoinCombination) : Prop :=
  comb.length = 6

theorem cannot_form_70_cents :
  ¬∃ (comb : CoinCombination), hasSixCoins comb ∧ totalValue comb = 70 :=
sorry

end NUMINAMATH_CALUDE_cannot_form_70_cents_l566_56658


namespace NUMINAMATH_CALUDE_black_pens_count_l566_56652

theorem black_pens_count (green_pens red_pens : ℕ) 
  (prob_neither_red_nor_green : ℚ) :
  green_pens = 5 →
  red_pens = 7 →
  prob_neither_red_nor_green = 1/3 →
  ∃ (total_pens black_pens : ℕ),
    total_pens = green_pens + red_pens + black_pens ∧
    (black_pens : ℚ) / total_pens = prob_neither_red_nor_green ∧
    black_pens = 6 :=
by sorry

end NUMINAMATH_CALUDE_black_pens_count_l566_56652


namespace NUMINAMATH_CALUDE_expression_one_evaluation_l566_56624

theorem expression_one_evaluation : 8 / (-2) - (-4) * (-3) = -16 := by sorry

end NUMINAMATH_CALUDE_expression_one_evaluation_l566_56624


namespace NUMINAMATH_CALUDE_product_from_sum_and_difference_l566_56688

theorem product_from_sum_and_difference :
  ∀ x y : ℝ, x + y = 72 ∧ x - y = 20 → x * y = 1196 := by
sorry

end NUMINAMATH_CALUDE_product_from_sum_and_difference_l566_56688


namespace NUMINAMATH_CALUDE_min_students_forgot_all_items_l566_56602

theorem min_students_forgot_all_items (total : ℕ) (forgot_gloves : ℕ) (forgot_scarves : ℕ) (forgot_hats : ℕ) 
  (h1 : total = 60)
  (h2 : forgot_gloves = 55)
  (h3 : forgot_scarves = 52)
  (h4 : forgot_hats = 50) :
  total - ((total - forgot_gloves) + (total - forgot_scarves) + (total - forgot_hats)) = 37 := by
  sorry

end NUMINAMATH_CALUDE_min_students_forgot_all_items_l566_56602


namespace NUMINAMATH_CALUDE_brave_children_count_l566_56631

/-- Represents the arrangement of children on a bench -/
structure BenchArrangement where
  total_children : ℕ
  boy_girl_pairs : ℕ

/-- The initial arrangement with 2 children -/
def initial_arrangement : BenchArrangement :=
  { total_children := 2, boy_girl_pairs := 1 }

/-- The final arrangement with 22 children alternating boy-girl -/
def final_arrangement : BenchArrangement :=
  { total_children := 22, boy_girl_pairs := 21 }

/-- A child is brave if they create two new boy-girl pairs when sitting down -/
def brave_children (initial final : BenchArrangement) : ℕ :=
  (final.boy_girl_pairs - initial.boy_girl_pairs) / 2

theorem brave_children_count :
  brave_children initial_arrangement final_arrangement = 10 := by
  sorry

end NUMINAMATH_CALUDE_brave_children_count_l566_56631


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_linear_l566_56629

theorem unique_solution_quadratic_linear (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = p.1^2 ∧ p.2 = 2*p.1 - k) ↔ k = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_linear_l566_56629


namespace NUMINAMATH_CALUDE_multiply_72514_99999_l566_56682

theorem multiply_72514_99999 : 72514 * 99999 = 7250675486 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72514_99999_l566_56682


namespace NUMINAMATH_CALUDE_justin_tim_games_count_l566_56647

def total_players : ℕ := 12
def players_per_game : ℕ := 6

theorem justin_tim_games_count :
  (total_players : ℕ) = 12 →
  (players_per_game : ℕ) = 6 →
  (Nat.choose total_players players_per_game : ℕ) = 
    (Nat.choose (total_players - 2) (players_per_game - 2) : ℕ) := by
  sorry

#eval Nat.choose 10 4

end NUMINAMATH_CALUDE_justin_tim_games_count_l566_56647


namespace NUMINAMATH_CALUDE_bessonov_tax_refund_l566_56665

def income_tax : ℝ := 156000
def education_expense : ℝ := 130000
def medical_expense : ℝ := 10000
def tax_rate : ℝ := 0.13

def total_deductible_expenses : ℝ := education_expense + medical_expense

def max_refund : ℝ := tax_rate * total_deductible_expenses

theorem bessonov_tax_refund :
  min max_refund income_tax = 18200 :=
sorry

end NUMINAMATH_CALUDE_bessonov_tax_refund_l566_56665


namespace NUMINAMATH_CALUDE_fourth_root_equation_l566_56614

theorem fourth_root_equation (x : ℝ) :
  (x * (x^4)^(1/2))^(1/4) = 2 → x = 16^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_l566_56614


namespace NUMINAMATH_CALUDE_distance_point_to_line_l566_56636

/-- The distance from a point to a vertical line -/
def distance_point_to_vertical_line (point : ℝ × ℝ) (line_x : ℝ) : ℝ :=
  |point.1 - line_x|

/-- Theorem: The distance from point (1, 2) to the line x = -2 is 3 -/
theorem distance_point_to_line : distance_point_to_vertical_line (1, 2) (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l566_56636


namespace NUMINAMATH_CALUDE_min_value_expression_l566_56696

theorem min_value_expression (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 27 / n ≥ 6 ∧
  ((n : ℝ) / 3 + 27 / n = 6 ↔ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l566_56696


namespace NUMINAMATH_CALUDE_train_time_theorem_l566_56633

/-- The time in minutes for a train to travel between two platforms --/
def train_travel_time (X : ℝ) : Prop :=
  0 < X ∧ X < 60 ∧
  ∀ (start_hour start_minute end_hour end_minute : ℝ),
    -- Angle between hour and minute hands at start
    |30 * start_hour - 5.5 * start_minute| = X →
    -- Angle between hour and minute hands at end
    |30 * end_hour - 5.5 * end_minute| = X →
    -- Time difference between start and end
    (end_hour - start_hour) * 60 + (end_minute - start_minute) = X →
    X = 48

theorem train_time_theorem :
  ∀ X, train_travel_time X → X = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_train_time_theorem_l566_56633


namespace NUMINAMATH_CALUDE_woody_writing_time_l566_56685

/-- Proves that Woody spent 1.5 years writing his book given the conditions -/
theorem woody_writing_time :
  ∀ (woody_months ivanka_months : ℕ),
  ivanka_months = woody_months + 3 →
  woody_months + ivanka_months = 39 →
  (woody_months : ℚ) / 12 = 3/2 := by
sorry

end NUMINAMATH_CALUDE_woody_writing_time_l566_56685


namespace NUMINAMATH_CALUDE_time_conversions_and_difference_l566_56655

/-- Converts 12-hour time (PM) to 24-hour format -/
def convert_pm_to_24h (hour : Nat) : Nat :=
  hour + 12

/-- Calculates the time difference in minutes between two times in 24-hour format -/
def time_diff_minutes (start_hour start_min end_hour end_min : Nat) : Nat :=
  (end_hour * 60 + end_min) - (start_hour * 60 + start_min)

theorem time_conversions_and_difference :
  (convert_pm_to_24h 5 = 17) ∧
  (convert_pm_to_24h 10 = 22) ∧
  (time_diff_minutes 16 40 17 20 = 40) :=
by sorry

end NUMINAMATH_CALUDE_time_conversions_and_difference_l566_56655


namespace NUMINAMATH_CALUDE_cost_of_candies_in_dollars_l566_56627

-- Define the cost of one piece of candy in cents
def cost_per_candy : ℕ := 2

-- Define the number of pieces of candy
def number_of_candies : ℕ := 500

-- Define the conversion rate from cents to dollars
def cents_per_dollar : ℕ := 100

-- Theorem to prove
theorem cost_of_candies_in_dollars :
  (number_of_candies * cost_per_candy) / cents_per_dollar = 10 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_candies_in_dollars_l566_56627


namespace NUMINAMATH_CALUDE_like_terms_exponent_difference_l566_56638

theorem like_terms_exponent_difference (a b : ℝ) (m n : ℤ) : 
  (∃ (k : ℝ), a^(m-2) * b^(n+7) = k * a^4 * b^4) → m - n = 9 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_difference_l566_56638


namespace NUMINAMATH_CALUDE_solve_timmys_orange_problem_l566_56693

/-- Represents the problem of calculating Timmy's remaining money after buying oranges --/
def timmys_orange_problem (calories_per_orange : ℕ) (oranges_per_pack : ℕ) 
  (price_per_orange : ℚ) (initial_money : ℚ) (calorie_goal : ℕ) (tax_rate : ℚ) : Prop :=
  let packs_needed : ℕ := ((calorie_goal + calories_per_orange - 1) / calories_per_orange + oranges_per_pack - 1) / oranges_per_pack
  let total_cost : ℚ := price_per_orange * (packs_needed * oranges_per_pack : ℚ)
  let tax_amount : ℚ := total_cost * tax_rate
  let final_cost : ℚ := total_cost + tax_amount
  let remaining_money : ℚ := initial_money - final_cost
  remaining_money = 244/100

/-- Theorem stating the solution to Timmy's orange problem --/
theorem solve_timmys_orange_problem : 
  timmys_orange_problem 80 3 (120/100) 10 400 (5/100) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_timmys_orange_problem_l566_56693


namespace NUMINAMATH_CALUDE_smallest_four_digit_palindrome_div_by_3_proof_l566_56656

/-- A function that checks if a number is a four-digit palindrome -/
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n / 1000 = n % 10) ∧ ((n / 100) % 10 = (n / 10) % 10)

/-- The smallest four-digit palindrome divisible by 3 -/
def smallest_four_digit_palindrome_div_by_3 : ℕ := 2112

theorem smallest_four_digit_palindrome_div_by_3_proof :
  is_four_digit_palindrome smallest_four_digit_palindrome_div_by_3 ∧
  smallest_four_digit_palindrome_div_by_3 % 3 = 0 ∧
  ∀ n : ℕ, is_four_digit_palindrome n ∧ n % 3 = 0 → n ≥ smallest_four_digit_palindrome_div_by_3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_palindrome_div_by_3_proof_l566_56656


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l566_56616

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 2*x = 9 ↔ (x - 1)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l566_56616


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l566_56651

def team_size : ℕ := 12
def offensive_linemen : ℕ := 4
def positions : ℕ := 5

theorem starting_lineup_combinations :
  (offensive_linemen) *
  (team_size - 1) *
  (team_size - 2) *
  (team_size - 3) *
  (team_size - 4) = 31680 :=
by sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l566_56651


namespace NUMINAMATH_CALUDE_lamp_cost_ratio_l566_56677

/-- The ratio of the cost of the most expensive lamp to the cheapest lamp -/
theorem lamp_cost_ratio 
  (cheapest_lamp : ℕ) 
  (frank_money : ℕ) 
  (remaining_money : ℕ) 
  (h1 : cheapest_lamp = 20)
  (h2 : frank_money = 90)
  (h3 : remaining_money = 30) :
  (frank_money - remaining_money) / cheapest_lamp = 3 := by
sorry

end NUMINAMATH_CALUDE_lamp_cost_ratio_l566_56677


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l566_56649

/-- A quadratic function f(x) = kx^2 - 4x - 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 4 * x - 2

/-- The discriminant of the quadratic function f(x) = kx^2 - 4x - 2 -/
def discriminant (k : ℝ) : ℝ := 16 + 8 * k

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f k x = 0 ∧ f k y = 0) ↔ k > -2 ∧ k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l566_56649


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l566_56691

def complex_to_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem z_in_fourth_quadrant (z : ℂ) 
  (h : (2 - 3*I)/(3 + 2*I) + z = 2 - 2*I) : 
  in_fourth_quadrant (complex_to_point z) := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l566_56691


namespace NUMINAMATH_CALUDE_MON_is_right_angle_l566_56666

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define point E
def E : ℝ × ℝ := (2, 2)

-- Define the line l
def line_l (x y : ℝ) : Prop := ∃ k, y = k*(x - 2)

-- Define that l passes through (2,0)
axiom l_through_2_0 : line_l 2 0

-- Define points A and B on the parabola and line l
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
axiom A_on_parabola : parabola A.1 A.2
axiom B_on_parabola : parabola B.1 B.2
axiom A_on_l : line_l A.1 A.2
axiom B_on_l : line_l B.1 B.2
axiom A_not_E : A ≠ E
axiom B_not_E : B ≠ E

-- Define points M and N
def M : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry
axiom M_on_EA : ∃ t, M = (1 - t) • E + t • A
axiom N_on_EB : ∃ t, N = (1 - t) • E + t • B
axiom M_on_x_neg2 : M.1 = -2
axiom N_on_x_neg2 : N.1 = -2

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem to prove
theorem MON_is_right_angle : 
  let OM := M - O
  let ON := N - O
  OM.1 * ON.1 + OM.2 * ON.2 = 0 := by sorry

end NUMINAMATH_CALUDE_MON_is_right_angle_l566_56666


namespace NUMINAMATH_CALUDE_complex_equation_solution_l566_56667

theorem complex_equation_solution (i : ℂ) (m : ℝ) : 
  i * i = -1 → (1 - m * i) / (i^3) = 1 + i → m = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l566_56667


namespace NUMINAMATH_CALUDE_pig_count_l566_56603

theorem pig_count (initial_pigs additional_pigs : ℝ) 
  (h1 : initial_pigs = 2465.25)
  (h2 : additional_pigs = 5683.75) : 
  initial_pigs + additional_pigs = 8149 :=
by sorry

end NUMINAMATH_CALUDE_pig_count_l566_56603


namespace NUMINAMATH_CALUDE_cards_in_basketball_box_dexter_basketball_cards_l566_56622

/-- The number of cards in each basketball card box -/
def cards_per_basketball_box (total_cards : ℕ) (basketball_boxes : ℕ) (football_boxes : ℕ) (cards_per_football_box : ℕ) : ℕ :=
  (total_cards - football_boxes * cards_per_football_box) / basketball_boxes

/-- Theorem stating the number of cards in each basketball card box -/
theorem cards_in_basketball_box :
  cards_per_basketball_box 255 9 6 20 = 15 := by
  sorry

/-- Main theorem proving the problem statement -/
theorem dexter_basketball_cards :
  ∃ (total_cards basketball_boxes football_boxes cards_per_football_box : ℕ),
    total_cards = 255 ∧
    basketball_boxes = 9 ∧
    football_boxes = basketball_boxes - 3 ∧
    cards_per_football_box = 20 ∧
    cards_per_basketball_box total_cards basketball_boxes football_boxes cards_per_football_box = 15 := by
  sorry

end NUMINAMATH_CALUDE_cards_in_basketball_box_dexter_basketball_cards_l566_56622


namespace NUMINAMATH_CALUDE_simplify_expression_l566_56642

theorem simplify_expression (w : ℝ) : 
  4*w + 6*w + 8*w + 10*w + 12*w + 14*w + 16 = 54*w + 16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l566_56642


namespace NUMINAMATH_CALUDE_area_of_specific_rectangle_l566_56643

/-- A rectangle with a diagonal divided into four equal segments -/
structure DividedRectangle where
  /-- The length of each segment of the diagonal -/
  segment_length : ℝ
  /-- The diagonal is divided into four equal segments -/
  diagonal_length : ℝ := 4 * segment_length
  /-- The parallel lines are perpendicular to the diagonal -/
  perpendicular_lines : Bool

/-- The area of a rectangle with a divided diagonal -/
def area (rect : DividedRectangle) : ℝ :=
  sorry

/-- Theorem: The area of the specific rectangle is 16√3 -/
theorem area_of_specific_rectangle :
  let rect : DividedRectangle := {
    segment_length := 2,
    perpendicular_lines := true
  }
  area rect = 16 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_area_of_specific_rectangle_l566_56643


namespace NUMINAMATH_CALUDE_total_arrangements_l566_56613

def num_students : ℕ := 8
def max_participants : ℕ := 5
def num_activities : ℕ := 2

def valid_distribution (dist : List ℕ) : Prop :=
  dist.length = num_activities ∧
  dist.sum = num_students ∧
  ∀ x ∈ dist, x ≤ max_participants

def num_arrangements : ℕ := sorry

theorem total_arrangements :
  num_arrangements = 182 :=
sorry

end NUMINAMATH_CALUDE_total_arrangements_l566_56613


namespace NUMINAMATH_CALUDE_dandelion_puffs_to_dog_l566_56605

/-- The number of dandelion puffs Caleb gave to his dog -/
def puffs_to_dog (total : ℕ) (to_mom : ℕ) (to_sister : ℕ) (to_grandmother : ℕ) 
                 (num_friends : ℕ) (to_each_friend : ℕ) : ℕ :=
  total - (to_mom + to_sister + to_grandmother + num_friends * to_each_friend)

/-- Theorem stating the number of dandelion puffs Caleb gave to his dog -/
theorem dandelion_puffs_to_dog : 
  puffs_to_dog 40 3 3 5 3 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_puffs_to_dog_l566_56605


namespace NUMINAMATH_CALUDE_ratio_sum_theorem_l566_56600

theorem ratio_sum_theorem (a b c : ℕ+) 
  (h1 : (a : ℚ) / b = 3 / 4)
  (h2 : (b : ℚ) / c = 5 / 6)
  (h3 : a + b + c = 1680) :
  a = 426 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_theorem_l566_56600


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l566_56650

def f (x : ℝ) := x^2 + 3*x - 5

theorem root_exists_in_interval :
  ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ f x = 0 :=
by
  have h1 : f 1.1 < 0 := by sorry
  have h2 : f 1.2 > 0 := by sorry
  sorry

#check root_exists_in_interval

end NUMINAMATH_CALUDE_root_exists_in_interval_l566_56650


namespace NUMINAMATH_CALUDE_store_profit_theorem_l566_56601

/-- Represents the selling price and number of items sold -/
structure SaleInfo where
  price : ℝ
  quantity : ℝ

/-- The profit function given the cost, price, and quantity -/
def profit (cost : ℝ) (info : SaleInfo) : ℝ :=
  (info.price - cost) * info.quantity

/-- The demand function given the base price, base quantity, and price sensitivity -/
def demand (basePrice baseQuantity priceSensitivity : ℝ) (price : ℝ) : ℝ :=
  baseQuantity - priceSensitivity * (price - basePrice)

theorem store_profit_theorem (cost basePrice baseQuantity priceSensitivity targetProfit : ℝ) :
  cost = 40 ∧
  basePrice = 50 ∧
  baseQuantity = 150 ∧
  priceSensitivity = 5 ∧
  targetProfit = 1500 →
  ∃ (info1 info2 : SaleInfo),
    info1.price = 50 ∧
    info1.quantity = 150 ∧
    info2.price = 70 ∧
    info2.quantity = 50 ∧
    profit cost info1 = targetProfit ∧
    profit cost info2 = targetProfit ∧
    info1.quantity = demand basePrice baseQuantity priceSensitivity info1.price ∧
    info2.quantity = demand basePrice baseQuantity priceSensitivity info2.price ∧
    ∀ (info : SaleInfo),
      profit cost info = targetProfit ∧
      info.quantity = demand basePrice baseQuantity priceSensitivity info.price →
      (info = info1 ∨ info = info2) := by
  sorry


end NUMINAMATH_CALUDE_store_profit_theorem_l566_56601


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l566_56675

theorem system_of_equations_solution (u v : ℚ) 
  (eq1 : 5 * u - 6 * v = 28)
  (eq2 : 3 * u + 5 * v = -13) :
  2 * u + 3 * v = -7767 / 645 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l566_56675


namespace NUMINAMATH_CALUDE_min_questions_correct_l566_56680

/-- Represents a company with N people, where one person knows everyone but is known by no one. -/
structure Company (N : ℕ) where
  -- The number of people in the company is at least 2
  people_count : N ≥ 2
  -- The function that determines if person i knows person j
  knows : Fin N → Fin N → Bool
  -- There exists a person who knows everyone else but is known by no one
  exists_z : ∃ z : Fin N, (∀ i : Fin N, i ≠ z → knows z i) ∧ (∀ i : Fin N, i ≠ z → ¬knows i z)

/-- The minimum number of questions needed to identify the person Z -/
def min_questions (N : ℕ) (c : Company N) : ℕ := N - 1

/-- Theorem stating that the minimum number of questions needed is N - 1 -/
theorem min_questions_correct (N : ℕ) (c : Company N) :
  ∀ strategy : (Fin N → Fin N → Bool) → Fin N,
  (∀ knows : Fin N → Fin N → Bool, 
   ∃ z : Fin N, (∀ i : Fin N, i ≠ z → knows z i) ∧ (∀ i : Fin N, i ≠ z → ¬knows i z) →
   ∃ questions : Finset (Fin N × Fin N),
     questions.card ≥ min_questions N c ∧
     strategy knows = z) :=
by
  sorry

end NUMINAMATH_CALUDE_min_questions_correct_l566_56680


namespace NUMINAMATH_CALUDE_part_one_part_two_l566_56669

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 3| + |x - a|

-- Part 1
theorem part_one : 
  {x : ℝ | f 4 x = 7} = Set.Icc (-3) 4 := by sorry

-- Part 2
theorem part_two (h : a > 0) :
  {x : ℝ | f a x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} → a = 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l566_56669


namespace NUMINAMATH_CALUDE_factorial_345_trailing_zeros_l566_56621

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_345_trailing_zeros :
  trailing_zeros 345 = 84 := by
  sorry

end NUMINAMATH_CALUDE_factorial_345_trailing_zeros_l566_56621


namespace NUMINAMATH_CALUDE_relation_between_x_and_y_l566_56623

theorem relation_between_x_and_y (p : ℝ) :
  let x : ℝ := 3 + 3^p
  let y : ℝ := 3 + 3^(-p)
  y = (3*x - 8) / (x - 3) :=
by sorry

end NUMINAMATH_CALUDE_relation_between_x_and_y_l566_56623


namespace NUMINAMATH_CALUDE_mindmaster_secret_codes_l566_56663

/-- The number of colors available for the pegs. -/
def num_colors : ℕ := 7

/-- The number of slots in each code. -/
def code_length : ℕ := 5

/-- The total number of possible codes without restrictions. -/
def total_codes : ℕ := num_colors ^ code_length

/-- The number of colors excluding red. -/
def non_red_colors : ℕ := num_colors - 1

/-- The number of codes without any red pegs. -/
def codes_without_red : ℕ := non_red_colors ^ code_length

/-- The number of valid secret codes in Mindmaster. -/
def valid_secret_codes : ℕ := total_codes - codes_without_red

theorem mindmaster_secret_codes : valid_secret_codes = 9031 := by
  sorry

end NUMINAMATH_CALUDE_mindmaster_secret_codes_l566_56663


namespace NUMINAMATH_CALUDE_odd_decreasing_function_theorem_l566_56690

/-- A function is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is decreasing if f(x₁) > f(x₂) for all x₁ < x₂ in its domain -/
def IsDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b → f x₁ > f x₂

theorem odd_decreasing_function_theorem (f : ℝ → ℝ) (a : ℝ) 
    (h_odd : IsOdd f)
    (h_decreasing : IsDecreasing f (-1) 1)
    (h_condition : f (1 + a) + f (1 - a^2) < 0) :
    a ∈ Set.Ioo (-1) 0 := by
  sorry


end NUMINAMATH_CALUDE_odd_decreasing_function_theorem_l566_56690


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l566_56697

/-- Proves that for a parallelogram with adjacent sides of lengths s and 2s units forming a 60-degree angle, if the area of the parallelogram is 12√3 square units, then s = 2√3. -/
theorem parallelogram_side_length (s : ℝ) :
  s > 0 →
  let side1 : ℝ := s
  let side2 : ℝ := 2 * s
  let angle : ℝ := π / 3  -- 60 degrees in radians
  let area : ℝ := 12 * Real.sqrt 3
  side2 * (side1 * Real.sin angle) = area →
  s = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l566_56697


namespace NUMINAMATH_CALUDE_lori_marbles_l566_56681

/-- The number of friends Lori shares her marbles with -/
def num_friends : ℕ := 5

/-- The number of marbles each friend gets when Lori shares her marbles -/
def marbles_per_friend : ℕ := 6

/-- The total number of marbles Lori has -/
def total_marbles : ℕ := num_friends * marbles_per_friend

theorem lori_marbles : total_marbles = 30 := by
  sorry

end NUMINAMATH_CALUDE_lori_marbles_l566_56681


namespace NUMINAMATH_CALUDE_B_equals_C_equals_A_union_complement_B_l566_56620

-- Define the sets A, B, C, and U
def A : Set ℝ := {x | x^2 ≥ 9}
def B : Set ℝ := {x | (x - 7) / (x + 1) ≤ 0}
def C : Set ℝ := {x | |x - 2| < 4}
def U : Set ℝ := Set.univ

-- Theorem statements
theorem B_equals : B = {x | -1 < x ∧ x ≤ 7} := by sorry

theorem C_equals : C = {x | -2 < x ∧ x < 6} := by sorry

theorem A_union_complement_B :
  A ∪ (U \ B) = {x | x ≥ 3 ∨ x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_B_equals_C_equals_A_union_complement_B_l566_56620


namespace NUMINAMATH_CALUDE_sqrt_equation_condition_l566_56626

theorem sqrt_equation_condition (a : ℝ) : 
  Real.sqrt (a^2 - 4*a + 4) = 2 - a ↔ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_condition_l566_56626


namespace NUMINAMATH_CALUDE_part1_part2_part3_l566_56664

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 2*x + 2*a

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ :=
  if f a x ≥ f' a x then f' a x else f a x

-- Part 1: Condition for f(x) ≤ f'(x) when x ∈ [-2, -1]
theorem part1 (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) (-1), f a x ≤ f' a x) → a ≥ 3/2 :=
sorry

-- Part 2: Solutions to f(x) = |f'(x)|
theorem part2 (a : ℝ) (x : ℝ) :
  f a x = |f' a x| →
  ((a < -1 ∧ (x = -1 ∨ x = 1 - 2*a)) ∨
   (-1 ≤ a ∧ a ≤ 1 ∧ (x = 1 ∨ x = -1 ∨ x = 1 - 2*a ∨ x = -(1 + 2*a))) ∨
   (a > 1 ∧ (x = 1 ∨ x = -(1 + 2*a)))) :=
sorry

-- Part 3: Minimum value of g(x) for x ∈ [2, 4]
theorem part3 (a : ℝ) :
  (∃ m : ℝ, ∀ x ∈ Set.Icc 2 4, g a x ≥ m) ∧
  (a ≤ -4 → ∃ x ∈ Set.Icc 2 4, g a x = 8*a + 17) ∧
  (-4 < a ∧ a < -2 → ∃ x ∈ Set.Icc 2 4, g a x = 1 - a^2) ∧
  (-2 ≤ a ∧ a < -1/2 → ∃ x ∈ Set.Icc 2 4, g a x = 4*a + 5) ∧
  (a ≥ -1/2 → ∃ x ∈ Set.Icc 2 4, g a x = 2*a + 4) :=
sorry

end

end NUMINAMATH_CALUDE_part1_part2_part3_l566_56664


namespace NUMINAMATH_CALUDE_right_triangle_tan_y_l566_56612

theorem right_triangle_tan_y (X Y Z : ℝ × ℝ) :
  -- Right triangle condition
  (Y.1 - X.1) * (Z.2 - X.2) = (Z.1 - X.1) * (Y.2 - X.2) →
  -- XY = 30 condition
  Real.sqrt ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) = 30 →
  -- XZ = 40 condition (derived from the solution)
  Real.sqrt ((Z.1 - X.1)^2 + (Z.2 - X.2)^2) = 40 →
  -- Conclusion: tan Y = 4/3
  (Z.2 - X.2) / (Y.1 - X.1) = 4 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_right_triangle_tan_y_l566_56612


namespace NUMINAMATH_CALUDE_domino_reconstruction_theorem_l566_56684

/-- Represents a 2x1 domino with color information -/
inductive Domino
| WhiteWhite
| BlueBlue
| WhiteBlue
| BlueWhite

/-- Represents an 8x8 grid -/
def Grid := List (List Bool)

/-- Counts the number of blue cells in a grid -/
def countBlue (g : Grid) : Nat := sorry

/-- Divides a grid into 2x1 dominoes -/
def divideToDominoes (g : Grid) : List Domino := sorry

/-- Reconstructs an 8x8 grid from a list of dominoes -/
def reconstructGrid (dominoes : List Domino) : Grid := sorry

/-- Checks if two grids have the same blue pattern -/
def samePattern (g1 g2 : Grid) : Bool := sorry

theorem domino_reconstruction_theorem (g1 g2 : Grid) 
  (h : countBlue g1 = countBlue g2) :
  ∃ (d1 d2 : List Domino), 
    d1 = divideToDominoes g1 ∧ 
    d2 = divideToDominoes g2 ∧ 
    samePattern (reconstructGrid (d1 ++ d2)) g1 ∧
    samePattern (reconstructGrid (d1 ++ d2)) g2 := by
  sorry

end NUMINAMATH_CALUDE_domino_reconstruction_theorem_l566_56684


namespace NUMINAMATH_CALUDE_prudence_sleep_is_200_l566_56673

/-- Represents Prudence's sleep schedule and calculates total sleep over 4 weeks -/
def prudence_sleep : ℕ :=
  let weekday_sleep : ℕ := 5 * 6  -- 5 nights of 6 hours each
  let weekend_sleep : ℕ := 2 * 9  -- 2 nights of 9 hours each
  let nap_sleep : ℕ := 2 * 1      -- 2 days of 1 hour nap each
  let weekly_sleep : ℕ := weekday_sleep + weekend_sleep + nap_sleep
  4 * weekly_sleep                -- 4 weeks

/-- Theorem stating that Prudence's total sleep over 4 weeks is 200 hours -/
theorem prudence_sleep_is_200 : prudence_sleep = 200 := by
  sorry

#eval prudence_sleep  -- This will evaluate to 200

end NUMINAMATH_CALUDE_prudence_sleep_is_200_l566_56673


namespace NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l566_56630

theorem polynomial_root_implies_coefficients 
  (a b : ℝ) 
  (h : (2 - 3*I : ℂ) ^ 3 + a * (2 - 3*I : ℂ) ^ 2 - (2 - 3*I : ℂ) + b = 0) : 
  a = -1/2 ∧ b = 91/2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l566_56630


namespace NUMINAMATH_CALUDE_cannot_make_24_l566_56657

/-- Represents the four basic arithmetic operations -/
inductive Operation
| Add
| Sub
| Mul
| Div

/-- Applies an operation to two rational numbers -/
def applyOp (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => if b ≠ 0 then a / b else 0

/-- Checks if it's possible to get 24 using the given numbers and operations -/
def canMake24 (a b c d : ℚ) : Prop :=
  ∃ (op1 op2 op3 : Operation),
    (applyOp op3 (applyOp op2 (applyOp op1 a b) c) d = 24) ∨
    (applyOp op3 (applyOp op2 (applyOp op1 a b) d) c = 24) ∨
    (applyOp op3 (applyOp op2 (applyOp op1 a c) b) d = 24) ∨
    (applyOp op3 (applyOp op2 (applyOp op1 a c) d) b = 24) ∨
    (applyOp op3 (applyOp op2 (applyOp op1 a d) b) c = 24) ∨
    (applyOp op3 (applyOp op2 (applyOp op1 a d) c) b = 24)

theorem cannot_make_24 : ¬ canMake24 1 6 8 7 := by
  sorry

end NUMINAMATH_CALUDE_cannot_make_24_l566_56657


namespace NUMINAMATH_CALUDE_total_marbles_l566_56604

theorem total_marbles (jar_a jar_b jar_c : ℕ) : 
  jar_a = 28 →
  jar_b = jar_a + 12 →
  jar_c = 2 * jar_b →
  jar_a + jar_b + jar_c = 148 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l566_56604


namespace NUMINAMATH_CALUDE_smallest_next_divisor_l566_56661

theorem smallest_next_divisor (m : ℕ) : 
  m % 2 = 0 ∧ 
  1000 ≤ m ∧ m < 10000 ∧ 
  m % 391 = 0 → 
  (∃ (d : ℕ), d ∣ m ∧ d > 391 ∧ d ≤ 782 ∧ ∀ (x : ℕ), x ∣ m ∧ x > 391 → x ≥ d) ∧
  782 ∣ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_l566_56661


namespace NUMINAMATH_CALUDE_cos_sin_identity_l566_56611

theorem cos_sin_identity : 
  Real.cos (14 * π / 180) * Real.cos (59 * π / 180) + 
  Real.sin (14 * π / 180) * Real.sin (121 * π / 180) = 
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_sin_identity_l566_56611
