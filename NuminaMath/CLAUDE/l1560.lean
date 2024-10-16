import Mathlib

namespace NUMINAMATH_CALUDE_card_sum_problem_l1560_156033

theorem card_sum_problem (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
sorry

end NUMINAMATH_CALUDE_card_sum_problem_l1560_156033


namespace NUMINAMATH_CALUDE_log_function_passes_through_point_l1560_156004

-- Define the logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (4 - x) + 1

-- State the theorem
theorem log_function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_function_passes_through_point_l1560_156004


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l1560_156049

/-- The measure of one interior angle of a regular octagon is 135 degrees -/
theorem regular_octagon_interior_angle : ℝ :=
  135

#check regular_octagon_interior_angle

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l1560_156049


namespace NUMINAMATH_CALUDE_exam_marks_percentage_l1560_156013

theorem exam_marks_percentage (full_marks A_marks B_marks C_marks D_marks : ℝ) : 
  full_marks = 500 →
  A_marks = B_marks * 0.9 →
  B_marks = C_marks * 1.25 →
  C_marks = D_marks * 0.8 →
  A_marks = 360 →
  D_marks / full_marks = 0.8 :=
by sorry

end NUMINAMATH_CALUDE_exam_marks_percentage_l1560_156013


namespace NUMINAMATH_CALUDE_complex_equation_result_l1560_156051

theorem complex_equation_result (a b : ℝ) (h : (1 + Complex.I) * (1 - b * Complex.I) = a) : a / b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_result_l1560_156051


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l1560_156058

def f (x : ℝ) : ℝ := |x| + 1

theorem f_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l1560_156058


namespace NUMINAMATH_CALUDE_tommys_coin_collection_l1560_156021

theorem tommys_coin_collection (nickels dimes quarters pennies : ℕ) : 
  nickels = 100 →
  nickels = 2 * dimes →
  quarters = 4 →
  pennies = 10 * quarters →
  dimes - pennies = 10 := by
  sorry

end NUMINAMATH_CALUDE_tommys_coin_collection_l1560_156021


namespace NUMINAMATH_CALUDE_shirt_sale_price_l1560_156006

theorem shirt_sale_price (original_price : ℝ) (original_price_pos : original_price > 0) :
  let first_sale_price := original_price * (1 - 0.5)
  let final_price := first_sale_price * (1 - 0.1)
  final_price / original_price = 0.45 := by
sorry

end NUMINAMATH_CALUDE_shirt_sale_price_l1560_156006


namespace NUMINAMATH_CALUDE_geometry_theorem_l1560_156025

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (intersection : Plane → Plane → Line → Prop)

-- Theorem statement
theorem geometry_theorem 
  (α β : Plane) (m n l : Line) : 
  (∀ m n α β, perpendicular m n → perpendicular_plane_line α m → perpendicular_plane_line β n → perpendicular_planes α β) ∧
  (∀ m α β, contained_in m α → parallel_planes α β → parallel_line_plane m β) ∧
  (∀ α β m l, intersection α β l → parallel_line_plane m α → parallel_line_plane m β → parallel_lines m l) ∧
  ¬(∀ m n α β, perpendicular m n → perpendicular_plane_line α m → parallel_line_plane n β → perpendicular_planes α β) :=
by sorry

end NUMINAMATH_CALUDE_geometry_theorem_l1560_156025


namespace NUMINAMATH_CALUDE_total_oranges_picked_l1560_156084

def monday_pick : ℕ := 100
def tuesday_pick : ℕ := 3 * monday_pick
def wednesday_pick : ℕ := 70

theorem total_oranges_picked : monday_pick + tuesday_pick + wednesday_pick = 470 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_picked_l1560_156084


namespace NUMINAMATH_CALUDE_sequence_formula_l1560_156041

def sequence_sum (a : ℕ+ → ℚ) (n : ℕ+) : ℚ :=
  (Finset.range n.val).sum (fun i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_formula (a : ℕ+ → ℚ) 
    (h : ∀ n : ℕ+, sequence_sum a n = 2 * n.val - a n + 1) :
    ∀ n : ℕ+, a n = 2 - 1 / (2 ^ n.val) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l1560_156041


namespace NUMINAMATH_CALUDE_max_x_squared_y_l1560_156046

theorem max_x_squared_y (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + 3 * y = 9) :
  x^2 * y ≤ 36 :=
sorry

end NUMINAMATH_CALUDE_max_x_squared_y_l1560_156046


namespace NUMINAMATH_CALUDE_min_value_of_x_l1560_156012

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x / Real.log 3 ≥ Real.log 9 / Real.log 3 + (1/3) * (Real.log x / Real.log 3)) :
  x ≥ 27 ∧ ∀ y : ℝ, y > 0 → Real.log y / Real.log 3 ≥ Real.log 9 / Real.log 3 + (1/3) * (Real.log y / Real.log 3) → y ≥ x → y ≥ 27 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_x_l1560_156012


namespace NUMINAMATH_CALUDE_digit_equation_solution_l1560_156069

theorem digit_equation_solution :
  ∀ x y z : ℕ,
    x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 →
    (10 * x + 5) * (300 + 10 * y + z) = 7850 →
    x = 2 ∧ y = 1 ∧ z = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l1560_156069


namespace NUMINAMATH_CALUDE_problem_solution_l1560_156039

theorem problem_solution : ∃ x : ℚ, (x + x/4 = 80 * 3/4) ∧ (x = 48) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1560_156039


namespace NUMINAMATH_CALUDE_square_triangle_circle_perimeter_l1560_156076

theorem square_triangle_circle_perimeter (x : ℝ) : 
  (4 * x) + (3 * x) = 2 * π * 4 → x = (8 * π) / 7 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_circle_perimeter_l1560_156076


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1560_156065

theorem closest_integer_to_cube_root (x : ℝ := (7^3 + 9^3) ^ (1/3)) : 
  ∃ (n : ℤ), ∀ (m : ℤ), |x - n| ≤ |x - m| ∧ n = 10 := by
sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1560_156065


namespace NUMINAMATH_CALUDE_geometry_book_pages_difference_l1560_156017

/-- Given that a new edition of a Geometry book has 450 pages and the old edition has 340 pages,
    prove that the new edition has 230 pages less than twice the number of pages in the old edition. -/
theorem geometry_book_pages_difference (new_edition : ℕ) (old_edition : ℕ)
  (h1 : new_edition = 450)
  (h2 : old_edition = 340) :
  2 * old_edition - new_edition = 230 := by
  sorry

end NUMINAMATH_CALUDE_geometry_book_pages_difference_l1560_156017


namespace NUMINAMATH_CALUDE_train_speed_difference_l1560_156072

theorem train_speed_difference (v : ℝ) 
  (cattle_speed : ℝ) (head_start : ℝ) (diesel_time : ℝ) (total_distance : ℝ)
  (h1 : v < cattle_speed)
  (h2 : cattle_speed = 56)
  (h3 : head_start = 6)
  (h4 : diesel_time = 12)
  (h5 : total_distance = 1284)
  (h6 : cattle_speed * head_start + cattle_speed * diesel_time + v * diesel_time = total_distance) :
  cattle_speed - v = 33 := by
sorry

end NUMINAMATH_CALUDE_train_speed_difference_l1560_156072


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1560_156074

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 10th term is 3 and the 12th term is 9,
    the 5th term is -12. -/
theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_10th : a 10 = 3)
  (h_12th : a 12 = 9) :
  a 5 = -12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1560_156074


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l1560_156096

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 3*x

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2) 2 ∧
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≤ f c) ∧
  f c = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l1560_156096


namespace NUMINAMATH_CALUDE_fractional_inequality_l1560_156067

theorem fractional_inequality (x : ℝ) : (2*x - 1) / (x + 1) < 0 ↔ -1 < x ∧ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_inequality_l1560_156067


namespace NUMINAMATH_CALUDE_three_digit_five_times_smaller_l1560_156002

/-- A three-digit number -/
def ThreeDigitNumber (a b c : ℕ) : Prop :=
  100 ≤ a * 100 + b * 10 + c ∧ a * 100 + b * 10 + c < 1000

/-- The condition that a number becomes five times smaller when the first digit is removed -/
def FiveTimesSmallerWithoutFirstDigit (a b c : ℕ) : Prop :=
  5 * (b * 10 + c) = a * 100 + b * 10 + c

/-- The theorem stating that 125, 250, and 375 are the only three-digit numbers
    that become five times smaller when the first digit is removed -/
theorem three_digit_five_times_smaller :
  ∀ a b c : ℕ,
  ThreeDigitNumber a b c ∧ FiveTimesSmallerWithoutFirstDigit a b c ↔
  (a = 1 ∧ b = 2 ∧ c = 5) ∨ (a = 2 ∧ b = 5 ∧ c = 0) ∨ (a = 3 ∧ b = 7 ∧ c = 5) :=
by sorry


end NUMINAMATH_CALUDE_three_digit_five_times_smaller_l1560_156002


namespace NUMINAMATH_CALUDE_impossible_to_cover_modified_chessboard_l1560_156018

/-- Represents a chessboard with some squares removed -/
structure ModifiedChessboard where
  size : Nat
  removed : Finset (Nat × Nat)

/-- Represents a domino that covers two squares -/
structure Domino where
  square1 : Nat × Nat
  square2 : Nat × Nat

/-- Checks if a given set of dominos covers the modified chessboard -/
def covers (board : ModifiedChessboard) (dominos : Finset Domino) : Prop :=
  sorry

/-- The color of a square on a chessboard (assuming top-left is white) -/
def squareColor (pos : Nat × Nat) : Bool :=
  (pos.1 + pos.2) % 2 == 0

theorem impossible_to_cover_modified_chessboard :
  ∀ (dominos : Finset Domino),
    let board := ModifiedChessboard.mk 8 {(0, 0), (7, 7)}
    ¬ covers board dominos := by
  sorry

end NUMINAMATH_CALUDE_impossible_to_cover_modified_chessboard_l1560_156018


namespace NUMINAMATH_CALUDE_direction_vector_proof_l1560_156082

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if a vector is a direction vector of a line -/
def isDirectionVector (l : Line2D) (v : Vector2D) : Prop :=
  v.x * l.b = -v.y * l.a

/-- The given line 4x - 3y + m = 0 -/
def givenLine : Line2D :=
  { a := 4, b := -3, c := 0 }  -- We set c to 0 as 'm' is arbitrary

/-- The vector (3, 4) -/
def givenVector : Vector2D :=
  { x := 3, y := 4 }

/-- Theorem: (3, 4) is a direction vector of the line 4x - 3y + m = 0 -/
theorem direction_vector_proof : 
  isDirectionVector givenLine givenVector := by
  sorry

end NUMINAMATH_CALUDE_direction_vector_proof_l1560_156082


namespace NUMINAMATH_CALUDE_value_of_a_l1560_156008

theorem value_of_a : ∀ (a b c d : ℤ),
  a = b + 7 →
  b = c + 15 →
  c = d + 25 →
  d = 90 →
  a = 137 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l1560_156008


namespace NUMINAMATH_CALUDE_total_price_after_tax_l1560_156020

def original_price : ℝ := 200
def tax_rate : ℝ := 0.15

theorem total_price_after_tax :
  original_price * (1 + tax_rate) = 230 := by sorry

end NUMINAMATH_CALUDE_total_price_after_tax_l1560_156020


namespace NUMINAMATH_CALUDE_absolute_value_plus_reciprocal_zero_l1560_156028

theorem absolute_value_plus_reciprocal_zero (x : ℝ) :
  x ≠ 0 ∧ |x| + 1/x = 0 → x = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_plus_reciprocal_zero_l1560_156028


namespace NUMINAMATH_CALUDE_distance_city_A_to_C_l1560_156056

/-- Prove the distance between city A and city C given travel times and speeds -/
theorem distance_city_A_to_C 
  (time_Eddy : ℝ) 
  (time_Freddy : ℝ) 
  (distance_AB : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : time_Eddy = 3) 
  (h2 : time_Freddy = 4) 
  (h3 : distance_AB = 600) 
  (h4 : speed_ratio = 1.7391304347826086) : 
  ∃ distance_AC : ℝ, distance_AC = 460 := by
  sorry

end NUMINAMATH_CALUDE_distance_city_A_to_C_l1560_156056


namespace NUMINAMATH_CALUDE_clock_partition_exists_l1560_156091

/-- A partition of the set {1, 2, ..., 12} into three subsets -/
structure ClockPartition where
  part1 : Finset Nat
  part2 : Finset Nat
  part3 : Finset Nat
  partition_complete : part1 ∪ part2 ∪ part3 = Finset.range 12
  partition_disjoint1 : Disjoint part1 part2
  partition_disjoint2 : Disjoint part1 part3
  partition_disjoint3 : Disjoint part2 part3

/-- The theorem stating that there exists a partition of the clock numbers
    into three parts with equal sums -/
theorem clock_partition_exists : ∃ (p : ClockPartition),
  (p.part1.sum id = p.part2.sum id) ∧ (p.part2.sum id = p.part3.sum id) :=
sorry

end NUMINAMATH_CALUDE_clock_partition_exists_l1560_156091


namespace NUMINAMATH_CALUDE_smallest_divisible_number_l1560_156075

theorem smallest_divisible_number : ∃ (n : ℕ), 
  (n > 2014) ∧ 
  (∀ k : ℕ, k < 10 → n % k = 0) ∧
  (∀ m : ℕ, m > 2014 ∧ m < n → ∃ j : ℕ, j < 10 ∧ m % j ≠ 0) ∧
  n = 2014506 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisible_number_l1560_156075


namespace NUMINAMATH_CALUDE_function_properties_l1560_156061

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x
noncomputable def g (x : ℝ) : ℝ := x^3

theorem function_properties :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f x ≤ f (Real.exp 1)) ∧
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f 1 ≤ f x) ∧
  (∀ x ≥ 1, f x ≤ g x) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1560_156061


namespace NUMINAMATH_CALUDE_discount_composition_l1560_156090

theorem discount_composition (original_price : ℝ) (h : original_price > 0) :
  let first_discount := 0.3
  let second_discount := 0.4
  let price_after_first := original_price * (1 - first_discount)
  let price_after_second := price_after_first * (1 - second_discount)
  let total_discount := 1 - (price_after_second / original_price)
  total_discount = 0.58 := by
sorry

end NUMINAMATH_CALUDE_discount_composition_l1560_156090


namespace NUMINAMATH_CALUDE_comic_arrangement_count_l1560_156024

/-- The number of different Spiderman comic books --/
def spiderman_comics : ℕ := 8

/-- The number of different Archie comic books --/
def archie_comics : ℕ := 6

/-- The number of different Garfield comic books --/
def garfield_comics : ℕ := 7

/-- The number of ways to arrange the comic books --/
def arrange_comics : ℕ := spiderman_comics.factorial * (archie_comics - 1).factorial * garfield_comics.factorial * 2

theorem comic_arrangement_count :
  arrange_comics = 4864460800 :=
by sorry

end NUMINAMATH_CALUDE_comic_arrangement_count_l1560_156024


namespace NUMINAMATH_CALUDE_circle_area_l1560_156060

theorem circle_area (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 55 ∧ 
   A = Real.pi * (Real.sqrt ((x + 3)^2 + (y + 8)^2))^2 ∧
   x^2 + y^2 + 6*x + 16*y + 18 = 0) := by
sorry

end NUMINAMATH_CALUDE_circle_area_l1560_156060


namespace NUMINAMATH_CALUDE_power_76_mod_7_l1560_156031

theorem power_76_mod_7 (n : ℕ) (h : Odd n) : 76^n % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_76_mod_7_l1560_156031


namespace NUMINAMATH_CALUDE_circle_centers_distance_bound_l1560_156043

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Sum of reciprocals of distances between circle centers -/
def sum_reciprocal_distances (circles : List Circle) : ℝ := sorry

/-- No line meets more than two circles -/
def no_line_meets_more_than_two (circles : List Circle) : Prop := sorry

theorem circle_centers_distance_bound (n : ℕ) (circles : List Circle) 
  (h1 : circles.length = n)
  (h2 : ∀ c ∈ circles, c.radius = 1)
  (h3 : no_line_meets_more_than_two circles) :
  sum_reciprocal_distances circles ≤ (n - 1 : ℝ) * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_centers_distance_bound_l1560_156043


namespace NUMINAMATH_CALUDE_first_train_speed_is_40_l1560_156086

/-- The speed of the first train in km/h -/
def first_train_speed : ℝ := sorry

/-- The speed of the second train in km/h -/
def second_train_speed : ℝ := 50

/-- The time difference between the departure of the two trains in hours -/
def time_difference : ℝ := 1

/-- The distance at which the two trains meet in km -/
def meeting_distance : ℝ := 200

/-- Theorem stating that given the conditions, the speed of the first train is 40 km/h -/
theorem first_train_speed_is_40 : first_train_speed = 40 := by sorry

end NUMINAMATH_CALUDE_first_train_speed_is_40_l1560_156086


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1560_156000

-- Define the property that f must satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x * f y) = f x * y

-- Define the three possible functions
def ZeroFunction : ℝ → ℝ := λ _ => 0
def IdentityFunction : ℝ → ℝ := λ x => x
def NegativeIdentityFunction : ℝ → ℝ := λ x => -x

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesProperty f →
    (f = ZeroFunction ∨ f = IdentityFunction ∨ f = NegativeIdentityFunction) :=
by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1560_156000


namespace NUMINAMATH_CALUDE_pens_per_student_after_split_l1560_156087

/-- The number of students --/
def num_students : ℕ := 3

/-- The number of red pens each student initially received --/
def red_pens_per_student : ℕ := 62

/-- The number of black pens each student initially received --/
def black_pens_per_student : ℕ := 43

/-- The total number of pens taken after the first month --/
def pens_taken_first_month : ℕ := 37

/-- The total number of pens taken after the second month --/
def pens_taken_second_month : ℕ := 41

/-- Theorem stating that each student will receive 79 pens when the remaining pens are split equally --/
theorem pens_per_student_after_split : 
  let total_pens := num_students * (red_pens_per_student + black_pens_per_student)
  let remaining_pens := total_pens - pens_taken_first_month - pens_taken_second_month
  remaining_pens / num_students = 79 := by
  sorry


end NUMINAMATH_CALUDE_pens_per_student_after_split_l1560_156087


namespace NUMINAMATH_CALUDE_celine_initial_amount_l1560_156054

/-- The price of a laptop in dollars -/
def laptop_price : ℕ := 600

/-- The price of a smartphone in dollars -/
def smartphone_price : ℕ := 400

/-- The number of laptops Celine bought -/
def laptops_bought : ℕ := 2

/-- The number of smartphones Celine bought -/
def smartphones_bought : ℕ := 4

/-- The amount of change Celine received in dollars -/
def change_received : ℕ := 200

/-- Celine's initial amount of money in dollars -/
def initial_amount : ℕ := laptop_price * laptops_bought + smartphone_price * smartphones_bought + change_received

theorem celine_initial_amount : initial_amount = 3000 := by
  sorry

end NUMINAMATH_CALUDE_celine_initial_amount_l1560_156054


namespace NUMINAMATH_CALUDE_barbed_wire_height_l1560_156032

/-- Calculates the height of a barbed wire fence around a square field. -/
theorem barbed_wire_height 
  (field_area : ℝ) 
  (wire_cost_per_meter : ℝ) 
  (gate_width : ℝ) 
  (num_gates : ℕ) 
  (total_cost : ℝ) 
  (h : field_area = 3136) 
  (h1 : wire_cost_per_meter = 3.5) 
  (h2 : gate_width = 1) 
  (h3 : num_gates = 2) 
  (h4 : total_cost = 2331) : 
  Real.sqrt field_area * 4 - (gate_width * num_gates) * wire_cost_per_meter * 
    (total_cost / (Real.sqrt field_area * 4 - gate_width * num_gates) / wire_cost_per_meter) = 2331 :=
sorry

end NUMINAMATH_CALUDE_barbed_wire_height_l1560_156032


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l1560_156066

/-- Given a geometric sequence {a_n} where a_4 = 7 and a_8 = 63, prove that a_6 = 21 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
    (h_a4 : a 4 = 7) (h_a8 : a 8 = 63) : a 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l1560_156066


namespace NUMINAMATH_CALUDE_min_cuts_for_hendecagons_l1560_156036

/-- Represents a polygon on the table --/
structure Polygon :=
  (sides : ℕ)

/-- Represents the state of the table after some cuts --/
structure TableState :=
  (polygons : List Polygon)

/-- Performs a single straight cut on the table --/
def makeCut (state : TableState) : TableState :=
  sorry

/-- Checks if the table state contains at least 252 hendecagons --/
def hasEnoughHendecagons (state : TableState) : Prop :=
  (state.polygons.filter (λ p => p.sides = 11)).length ≥ 252

/-- The minimum number of cuts needed to create at least 252 hendecagons --/
def minCuts : ℕ := 2015

theorem min_cuts_for_hendecagons :
  ∀ (initialState : TableState),
    initialState.polygons = [Polygon.mk 4] →
    ∀ (n : ℕ),
      (∃ (finalState : TableState),
        (Nat.iterate makeCut n initialState = finalState) ∧
        hasEnoughHendecagons finalState) →
      n ≥ minCuts :=
sorry

end NUMINAMATH_CALUDE_min_cuts_for_hendecagons_l1560_156036


namespace NUMINAMATH_CALUDE_onion_piece_per_student_l1560_156089

/-- Represents the pizza distribution problem --/
structure PizzaDistribution where
  students : ℕ
  pizzas : ℕ
  slices_per_pizza : ℕ
  cheese_per_student : ℕ
  leftover_cheese : ℕ
  leftover_onion : ℕ

/-- Calculates the number of onion pieces per student --/
def onion_per_student (pd : PizzaDistribution) : ℕ :=
  let total_slices := pd.pizzas * pd.slices_per_pizza
  let total_cheese := pd.students * pd.cheese_per_student
  let used_slices := total_slices - pd.leftover_cheese - pd.leftover_onion
  let onion_slices := used_slices - total_cheese
  onion_slices / pd.students

/-- Theorem stating that each student gets 1 piece of onion pizza --/
theorem onion_piece_per_student (pd : PizzaDistribution) 
  (h1 : pd.students = 32)
  (h2 : pd.pizzas = 6)
  (h3 : pd.slices_per_pizza = 18)
  (h4 : pd.cheese_per_student = 2)
  (h5 : pd.leftover_cheese = 8)
  (h6 : pd.leftover_onion = 4) :
  onion_per_student pd = 1 := by
  sorry

end NUMINAMATH_CALUDE_onion_piece_per_student_l1560_156089


namespace NUMINAMATH_CALUDE_pony_daily_food_cost_l1560_156097

def annual_expenses : ℕ := 15890
def monthly_pasture_rent : ℕ := 500
def weekly_lessons : ℕ := 2
def lesson_cost : ℕ := 60
def months_per_year : ℕ := 12
def weeks_per_year : ℕ := 52
def days_per_year : ℕ := 365

theorem pony_daily_food_cost :
  (annual_expenses - (monthly_pasture_rent * months_per_year + weekly_lessons * lesson_cost * weeks_per_year)) / days_per_year = 10 :=
by sorry

end NUMINAMATH_CALUDE_pony_daily_food_cost_l1560_156097


namespace NUMINAMATH_CALUDE_special_pentagon_exists_l1560_156080

/-- A pentagon that can be divided into three parts by one straight cut,
    such that two of the parts can be combined to form the third part. -/
structure SpecialPentagon where
  /-- The vertices of the pentagon -/
  vertices : Fin 5 → ℝ × ℝ
  /-- The cut line that divides the pentagon -/
  cut_line : ℝ × ℝ → ℝ × ℝ → Prop
  /-- The three parts resulting from the cut -/
  parts : Fin 3 → Set (ℝ × ℝ)
  /-- Proof that the cut line divides the pentagon into exactly three parts -/
  valid_division : sorry
  /-- Proof that two of the parts can be combined to form the third part -/
  recombination : sorry

/-- Theorem stating the existence of a special pentagon -/
theorem special_pentagon_exists : ∃ (p : SpecialPentagon), True := by
  sorry

end NUMINAMATH_CALUDE_special_pentagon_exists_l1560_156080


namespace NUMINAMATH_CALUDE_train_length_l1560_156011

/-- The length of a train given its crossing time, bridge length, and speed -/
theorem train_length (crossing_time : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) :
  crossing_time = 26.997840172786177 →
  bridge_length = 170 →
  train_speed_kmph = 36 →
  ∃ (train_length : ℝ), abs (train_length - 99.978) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1560_156011


namespace NUMINAMATH_CALUDE_barbed_wire_rate_l1560_156092

/-- Given a square field with area 3136 sq m, barbed wire drawn 3 m around it,
    two 1 m wide gates, and a total cost of 1332 Rs, prove that the rate of
    drawing barbed wire per meter is 6 Rs/m. -/
theorem barbed_wire_rate (area : ℝ) (wire_distance : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ)
    (h_area : area = 3136)
    (h_wire_distance : wire_distance = 3)
    (h_gate_width : gate_width = 1)
    (h_num_gates : num_gates = 2)
    (h_total_cost : total_cost = 1332) :
    total_cost / (4 * Real.sqrt area - num_gates * gate_width) = 6 := by
  sorry

end NUMINAMATH_CALUDE_barbed_wire_rate_l1560_156092


namespace NUMINAMATH_CALUDE_impossibleOneLight_l1560_156044

/- Define the grid size -/
def gridSize : Nat := 8

/- Define the state of the grid as a function from coordinates to bool -/
def GridState := Fin gridSize → Fin gridSize → Bool

/- Define the initial state where all bulbs are on -/
def initialState : GridState := fun _ _ => true

/- Define the toggle operation for a row -/
def toggleRow (state : GridState) (row : Fin gridSize) : GridState :=
  fun i j => if i = row then !state i j else state i j

/- Define the toggle operation for a column -/
def toggleColumn (state : GridState) (col : Fin gridSize) : GridState :=
  fun i j => if j = col then !state i j else state i j

/- Define a property that checks if exactly one bulb is on -/
def exactlyOneBulbOn (state : GridState) : Prop :=
  ∃! i j, state i j = true

/- The main theorem -/
theorem impossibleOneLight : 
  ¬∃ (toggleSequence : List (Bool × Fin gridSize)), 
    let finalState := toggleSequence.foldl 
      (fun acc (toggle) => 
        match toggle with
        | (true, n) => toggleRow acc n
        | (false, n) => toggleColumn acc n) 
      initialState
    exactlyOneBulbOn finalState :=
by
  sorry

end NUMINAMATH_CALUDE_impossibleOneLight_l1560_156044


namespace NUMINAMATH_CALUDE_factorization_a4_plus_4_l1560_156095

theorem factorization_a4_plus_4 (a : ℝ) : a^4 + 4 = (a^2 + 2*a + 2)*(a^2 - 2*a + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a4_plus_4_l1560_156095


namespace NUMINAMATH_CALUDE_colonization_theorem_l1560_156014

def blue_planets : ℕ := 7
def orange_planets : ℕ := 8
def blue_cost : ℕ := 3
def orange_cost : ℕ := 2
def total_units : ℕ := 21

def colonization_ways (b o bc oc t : ℕ) : ℕ :=
  (Nat.choose b 7 * Nat.choose o 0) +
  (Nat.choose b 5 * Nat.choose o 3) +
  (Nat.choose b 3 * Nat.choose o 6)

theorem colonization_theorem :
  colonization_ways blue_planets orange_planets blue_cost orange_cost total_units = 2157 := by
  sorry

end NUMINAMATH_CALUDE_colonization_theorem_l1560_156014


namespace NUMINAMATH_CALUDE_contractor_payment_result_l1560_156048

def contractor_payment (total_days : ℕ) (working_pay : ℚ) (absence_fine : ℚ) (absent_days : ℕ) : ℚ :=
  let working_days := total_days - absent_days
  let total_earnings := working_days * working_pay
  let total_fines := absent_days * absence_fine
  total_earnings - total_fines

theorem contractor_payment_result :
  contractor_payment 30 25 7.5 12 = 360 := by
  sorry

end NUMINAMATH_CALUDE_contractor_payment_result_l1560_156048


namespace NUMINAMATH_CALUDE_equation_transformation_l1560_156094

theorem equation_transformation (x y : ℝ) : x - 3 = y - 3 → x - y = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l1560_156094


namespace NUMINAMATH_CALUDE_range_of_sum_of_squares_l1560_156047

theorem range_of_sum_of_squares (x y : ℝ) (h : x^2 - 2*x*y + 5*y^2 = 4) :
  3 - Real.sqrt 5 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 3 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_of_squares_l1560_156047


namespace NUMINAMATH_CALUDE_max_stone_value_l1560_156070

/-- Represents the types of stones --/
inductive StoneType
| FivePound
| FourPound
| OnePound

/-- Returns the weight of a stone type in pounds --/
def weight (s : StoneType) : ℕ :=
  match s with
  | StoneType.FivePound => 5
  | StoneType.FourPound => 4
  | StoneType.OnePound => 1

/-- Returns the value of a stone type in dollars --/
def value (s : StoneType) : ℕ :=
  match s with
  | StoneType.FivePound => 14
  | StoneType.FourPound => 11
  | StoneType.OnePound => 2

/-- Represents a combination of stones --/
structure StoneCombination where
  fivePound : ℕ
  fourPound : ℕ
  onePound : ℕ

/-- Calculates the total weight of a stone combination --/
def totalWeight (c : StoneCombination) : ℕ :=
  c.fivePound * weight StoneType.FivePound +
  c.fourPound * weight StoneType.FourPound +
  c.onePound * weight StoneType.OnePound

/-- Calculates the total value of a stone combination --/
def totalValue (c : StoneCombination) : ℕ :=
  c.fivePound * value StoneType.FivePound +
  c.fourPound * value StoneType.FourPound +
  c.onePound * value StoneType.OnePound

/-- Defines a valid stone combination --/
def isValidCombination (c : StoneCombination) : Prop :=
  totalWeight c ≤ 18 ∧ c.fivePound ≤ 20 ∧ c.fourPound ≤ 20 ∧ c.onePound ≤ 20

theorem max_stone_value :
  ∃ (c : StoneCombination), isValidCombination c ∧
    totalValue c = 50 ∧
    ∀ (c' : StoneCombination), isValidCombination c' → totalValue c' ≤ 50 :=
by sorry

end NUMINAMATH_CALUDE_max_stone_value_l1560_156070


namespace NUMINAMATH_CALUDE_solution_value_l1560_156077

theorem solution_value (a b : ℝ) (h : 2 * (-3) - a + 2 * b = 0) : 
  2 * a - 4 * b + 1 = -11 := by
sorry

end NUMINAMATH_CALUDE_solution_value_l1560_156077


namespace NUMINAMATH_CALUDE_count_valid_permutations_l1560_156063

def alphabet : List Char := ['a', 'b', 'c', 'd', 'e']

def is_adjacent (c1 c2 : Char) : Bool :=
  let idx1 := alphabet.indexOf c1
  let idx2 := alphabet.indexOf c2
  (idx1 + 1 = idx2) || (idx2 + 1 = idx1)

def is_valid_permutation (perm : List Char) : Bool :=
  List.zip perm (List.tail perm) |>.all (fun (c1, c2) => !is_adjacent c1 c2)

def valid_permutations : List (List Char) :=
  List.permutations alphabet |>.filter is_valid_permutation

theorem count_valid_permutations : valid_permutations.length = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_permutations_l1560_156063


namespace NUMINAMATH_CALUDE_range_of_a_l1560_156016

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 12, x^2 - a ≥ 0) ∨ 
  (∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0) →
  ¬((∀ x ∈ Set.Icc 1 12, x^2 - a ≥ 0) ∧ 
    (∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0)) →
  (-1 ≤ a ∧ a ≤ 1) ∨ a > 3 :=
by sorry


end NUMINAMATH_CALUDE_range_of_a_l1560_156016


namespace NUMINAMATH_CALUDE_floor_plus_self_equals_seventeen_fourths_l1560_156068

theorem floor_plus_self_equals_seventeen_fourths :
  ∃ x : ℚ, (⌊x⌋ : ℚ) + x = 17/4 ∧ x = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_equals_seventeen_fourths_l1560_156068


namespace NUMINAMATH_CALUDE_right_triangle_area_perimeter_relation_l1560_156088

theorem right_triangle_area_perimeter_relation (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  a * b = 3 * (a + b + c) →
  ((a = 7 ∧ b = 24 ∧ c = 25) ∨
   (a = 8 ∧ b = 15 ∧ c = 17) ∨
   (a = 9 ∧ b = 12 ∧ c = 15) ∨
   (b = 7 ∧ a = 24 ∧ c = 25) ∨
   (b = 8 ∧ a = 15 ∧ c = 17) ∨
   (b = 9 ∧ a = 12 ∧ c = 15)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_perimeter_relation_l1560_156088


namespace NUMINAMATH_CALUDE_new_person_weight_l1560_156055

/-- The weight of a new person who replaces one person in a group, given the change in average weight -/
def weight_of_new_person (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + n * avg_increase

/-- Theorem stating the weight of the new person in the given scenario -/
theorem new_person_weight :
  weight_of_new_person 10 6.3 65 = 128 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1560_156055


namespace NUMINAMATH_CALUDE_pythagorean_triple_identification_l1560_156038

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_identification :
  is_pythagorean_triple 3 4 5 ∧
  is_pythagorean_triple 5 12 13 ∧
  ¬(is_pythagorean_triple 4 5 6) ∧
  is_pythagorean_triple 8 15 17 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_identification_l1560_156038


namespace NUMINAMATH_CALUDE_pq_length_l1560_156005

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 1)

-- Define the intersection points
def intersection_points (P Q : ℝ × ℝ) : Prop :=
  parabola P.1 P.2 ∧ parabola Q.1 Q.2 ∧ line P.1 P.2 ∧ line Q.1 Q.2

-- Theorem statement
theorem pq_length (P Q : ℝ × ℝ) :
  intersection_points P Q →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 16/3 := by sorry

end NUMINAMATH_CALUDE_pq_length_l1560_156005


namespace NUMINAMATH_CALUDE_remainder_2023_times_7_div_45_l1560_156064

theorem remainder_2023_times_7_div_45 : (2023 * 7) % 45 = 31 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2023_times_7_div_45_l1560_156064


namespace NUMINAMATH_CALUDE_coefficient_of_x4_l1560_156037

theorem coefficient_of_x4 (x : ℝ) : 
  let expr := 5*(x^4 - 2*x^5) + 3*(2*x^2 - x^6 + x^3) - (2*x^6 - 3*x^4 + x^2)
  ∃ (a b c d e f : ℝ), expr = 8*x^4 + a*x^6 + b*x^5 + c*x^3 + d*x^2 + e*x + f :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x4_l1560_156037


namespace NUMINAMATH_CALUDE_wage_increase_result_l1560_156019

/-- Calculates the new wage after a percentage increase -/
def new_wage (original_wage : ℝ) (percent_increase : ℝ) : ℝ :=
  original_wage * (1 + percent_increase)

/-- Theorem stating that a 50% increase on a $28 wage results in $42 -/
theorem wage_increase_result :
  new_wage 28 0.5 = 42 := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_result_l1560_156019


namespace NUMINAMATH_CALUDE_amusement_park_admission_difference_l1560_156062

theorem amusement_park_admission_difference :
  let students : ℕ := 194
  let adults : ℕ := 235
  let free_admission : ℕ := 68
  let total_visitors : ℕ := students + adults
  let paid_admission : ℕ := total_visitors - free_admission
  paid_admission - free_admission = 293 :=
by
  sorry

end NUMINAMATH_CALUDE_amusement_park_admission_difference_l1560_156062


namespace NUMINAMATH_CALUDE_square_of_one_plus_i_l1560_156030

theorem square_of_one_plus_i :
  let z : ℂ := 1 + Complex.I
  z^2 = 2 * Complex.I := by sorry

end NUMINAMATH_CALUDE_square_of_one_plus_i_l1560_156030


namespace NUMINAMATH_CALUDE_triangle_hypotenuse_length_l1560_156015

-- Define the triangle and points
def Triangle (P Q R : ℝ × ℝ) : Prop := sorry

def RightTriangle (P Q R : ℝ × ℝ) : Prop := 
  Triangle P Q R ∧ sorry -- Add condition for right angle

def PointOnLine (P Q M : ℝ × ℝ) : Prop := sorry

-- Define the ratio condition
def RatioCondition (P M Q : ℝ × ℝ) : Prop := 
  ∃ (k : ℝ), k = 1/3 ∧ sorry -- Add condition for PM:MQ = 1:3

-- Define the distance function
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem triangle_hypotenuse_length 
  (P Q R M N : ℝ × ℝ) 
  (h1 : RightTriangle P Q R) 
  (h2 : PointOnLine P Q M) 
  (h3 : PointOnLine P R N) 
  (h4 : RatioCondition P M Q) 
  (h5 : RatioCondition P N R) 
  (h6 : distance Q N = 20) 
  (h7 : distance M R = 36) : 
  distance Q R = 2 * Real.sqrt 399 := by
  sorry

end NUMINAMATH_CALUDE_triangle_hypotenuse_length_l1560_156015


namespace NUMINAMATH_CALUDE_direction_vector_b_l1560_156026

/-- Prove that for a line passing through points (-6, 0) and (-3, 3), its direction vector (3, b) has b = 3. -/
theorem direction_vector_b (b : ℝ) : 
  let p1 : ℝ × ℝ := (-6, 0)
  let p2 : ℝ × ℝ := (-3, 3)
  let direction_vector : ℝ × ℝ := (3, b)
  (p2.1 - p1.1 = direction_vector.1 ∧ p2.2 - p1.2 = direction_vector.2) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_direction_vector_b_l1560_156026


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l1560_156034

theorem complex_modulus_sqrt_two (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l1560_156034


namespace NUMINAMATH_CALUDE_snickers_bars_proof_l1560_156050

/-- The number of points needed to win the Nintendo Switch -/
def total_points_needed : ℕ := 2000

/-- The number of chocolate bunnies sold -/
def chocolate_bunnies_sold : ℕ := 8

/-- The number of points earned per chocolate bunny -/
def points_per_bunny : ℕ := 100

/-- The number of points earned per Snickers bar -/
def points_per_snickers : ℕ := 25

/-- Calculates the number of Snickers bars needed to win the Nintendo Switch -/
def snickers_bars_needed : ℕ :=
  (total_points_needed - chocolate_bunnies_sold * points_per_bunny) / points_per_snickers

theorem snickers_bars_proof :
  snickers_bars_needed = 48 := by
  sorry

end NUMINAMATH_CALUDE_snickers_bars_proof_l1560_156050


namespace NUMINAMATH_CALUDE_geometric_ratio_sum_condition_l1560_156083

theorem geometric_ratio_sum_condition (a b c d a' b' c' d' : ℝ) 
  (h1 : a / b = c / d) (h2 : a' / b' = c' / d') :
  (a + a') / (b + b') = (c + c') / (d + d') ↔ a / a' = b / b' ∧ b / b' = c / c' ∧ c / c' = d / d' :=
by sorry

end NUMINAMATH_CALUDE_geometric_ratio_sum_condition_l1560_156083


namespace NUMINAMATH_CALUDE_train_problem_l1560_156022

/-- Calculates the number of people who got on a train given the initial count, 
    the number who got off, and the final count. -/
def peopleGotOn (initial : ℕ) (gotOff : ℕ) (final : ℕ) : ℕ :=
  final - (initial - gotOff)

theorem train_problem : peopleGotOn 78 27 63 = 12 := by
  sorry

end NUMINAMATH_CALUDE_train_problem_l1560_156022


namespace NUMINAMATH_CALUDE_complex_product_square_l1560_156053

/-- Given complex numbers Q, E, and D, prove that (Q * E * D)² equals 8400 + 8000i -/
theorem complex_product_square (Q E D : ℂ) 
  (hQ : Q = 7 + 3*I) 
  (hE : E = 1 + I) 
  (hD : D = 7 - 3*I) : 
  (Q * E * D)^2 = 8400 + 8000*I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_square_l1560_156053


namespace NUMINAMATH_CALUDE_center_cell_value_l1560_156042

theorem center_cell_value (a b c d e f g h i : ℝ) 
  (positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0 ∧ i > 0)
  (row_products : a * b * c = 1 ∧ d * e * f = 1 ∧ g * h * i = 1)
  (col_products : a * d * g = 1 ∧ b * e * h = 1 ∧ c * f * i = 1)
  (square_products : a * d * e * b = 2 ∧ b * e * f * c = 2 ∧ d * e * g * h = 2 ∧ e * f * h * i = 2) :
  e = 1 := by
sorry

end NUMINAMATH_CALUDE_center_cell_value_l1560_156042


namespace NUMINAMATH_CALUDE_repeating_decimal_properties_l1560_156052

/-- Represents a repeating decimal with a 3-digit non-repeating part and a 4-digit repeating part -/
structure RepeatingDecimal where
  N : ℕ  -- Non-repeating part (3 digits)
  M : ℕ  -- Repeating part (4 digits)

variable (R : RepeatingDecimal)

/-- The decimal expansion of R -/
noncomputable def decimal_expansion (R : RepeatingDecimal) : ℝ := sorry

theorem repeating_decimal_properties (R : RepeatingDecimal) :
  -- 1. R = 0.NMM... is a correct representation
  decimal_expansion R = (R.N : ℝ) / 1000 + (R.M : ℝ) / 10000 / (1 - 1 / 10000) ∧
  -- 2. 10^3R = N.MMM... is a correct representation
  1000 * decimal_expansion R = R.N + (R.M : ℝ) / 10000 / (1 - 1 / 10000) ∧
  -- 3. 10^7R ≠ NMN.MMM...
  10000000 * decimal_expansion R ≠ (R.N * 1000000 + R.M * 100 + R.N) + (R.M : ℝ) / 10000 / (1 - 1 / 10000) ∧
  -- 4. 10^3(10^4-1)R ≠ 10^4N - M
  1000 * (10000 - 1) * decimal_expansion R ≠ 10000 * R.N - R.M :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_properties_l1560_156052


namespace NUMINAMATH_CALUDE_min_sum_with_log_condition_l1560_156009

theorem min_sum_with_log_condition (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_log : Real.log a + Real.log b = Real.log (a + b)) : 
  ∀ x y : ℝ, x > 0 → y > 0 → Real.log x + Real.log y = Real.log (x + y) → a + b ≤ x + y ∧ a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_log_condition_l1560_156009


namespace NUMINAMATH_CALUDE_students_walking_home_l1560_156027

theorem students_walking_home (bus auto bike scooter : ℚ) 
  (h_bus : bus = 1/3)
  (h_auto : auto = 1/5)
  (h_bike : bike = 1/6)
  (h_scooter : scooter = 1/10)
  (h_total : bus + auto + bike + scooter < 1) :
  1 - (bus + auto + bike + scooter) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_students_walking_home_l1560_156027


namespace NUMINAMATH_CALUDE_age_of_twentieth_student_l1560_156099

theorem age_of_twentieth_student (total_students : Nat) (total_avg_age : Nat)
  (group1_count : Nat) (group1_avg_age : Nat)
  (group2_count : Nat) (group2_avg_age : Nat)
  (group3_count : Nat) (group3_avg_age : Nat) :
  total_students = 20 →
  total_avg_age = 18 →
  group1_count = 6 →
  group1_avg_age = 16 →
  group2_count = 8 →
  group2_avg_age = 17 →
  group3_count = 5 →
  group3_avg_age = 21 →
  (total_students * total_avg_age) - 
  (group1_count * group1_avg_age + group2_count * group2_avg_age + group3_count * group3_avg_age) = 23 := by
  sorry

end NUMINAMATH_CALUDE_age_of_twentieth_student_l1560_156099


namespace NUMINAMATH_CALUDE_green_apples_count_l1560_156081

/-- Given a basket with red and green apples, prove the number of green apples. -/
theorem green_apples_count (total : ℕ) (red : ℕ) (green : ℕ) : 
  total = 9 → red = 7 → green = total - red → green = 2 := by sorry

end NUMINAMATH_CALUDE_green_apples_count_l1560_156081


namespace NUMINAMATH_CALUDE_total_kids_l1560_156078

theorem total_kids (girls : ℕ) (boys : ℕ) 
  (h1 : girls = 3) (h2 : boys = 6) : 
  girls + boys = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_kids_l1560_156078


namespace NUMINAMATH_CALUDE_integral_sqrt_4_minus_x_squared_l1560_156071

theorem integral_sqrt_4_minus_x_squared : ∫ x in (-2)..2, Real.sqrt (4 - x^2) = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_4_minus_x_squared_l1560_156071


namespace NUMINAMATH_CALUDE_jelly_bean_distribution_l1560_156045

/-- Represents the number of jelly beans each person has -/
structure JellyBeans :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Performs the first distribution: A gives to B and C -/
def firstDistribution (jb : JellyBeans) : JellyBeans :=
  ⟨jb.a - jb.b - jb.c, jb.b + jb.b, jb.c + jb.c⟩

/-- Performs the second distribution: B gives to A and C -/
def secondDistribution (jb : JellyBeans) : JellyBeans :=
  ⟨jb.a + jb.a, jb.b - jb.a - jb.c, jb.c + jb.c⟩

/-- Performs the third distribution: C gives to A and B -/
def thirdDistribution (jb : JellyBeans) : JellyBeans :=
  ⟨jb.a + jb.a, jb.b + jb.b, jb.c - jb.a - jb.b⟩

theorem jelly_bean_distribution :
  let initial := JellyBeans.mk 104 56 32
  let final := thirdDistribution (secondDistribution (firstDistribution initial))
  final.a = 64 ∧ final.b = 64 ∧ final.c = 64 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_distribution_l1560_156045


namespace NUMINAMATH_CALUDE_male_students_bound_l1560_156023

/-- Represents the arrangement of students in a grid -/
structure StudentArrangement where
  rows : ℕ
  columns : ℕ
  total_students : ℕ
  same_gender_pairs_bound : ℕ

/-- Counts the number of male students in a given arrangement -/
def count_male_students (arrangement : StudentArrangement) : ℕ := sorry

/-- The main theorem to be proved -/
theorem male_students_bound (arrangement : StudentArrangement) 
  (h1 : arrangement.rows = 22)
  (h2 : arrangement.columns = 75)
  (h3 : arrangement.total_students = 1650)
  (h4 : arrangement.same_gender_pairs_bound = 11) :
  count_male_students arrangement ≤ 928 := by sorry

end NUMINAMATH_CALUDE_male_students_bound_l1560_156023


namespace NUMINAMATH_CALUDE_equation_solution_l1560_156093

theorem equation_solution : ∃ x : ℝ, 24 - 4 * 2 = 3 + x ∧ x = 13 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1560_156093


namespace NUMINAMATH_CALUDE_solution_set_of_system_l1560_156057

theorem solution_set_of_system (x y : ℝ) :
  x - 2 * y = 1 →
  x^3 - 6 * x * y - 8 * y^3 = 1 →
  y = (x - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_of_system_l1560_156057


namespace NUMINAMATH_CALUDE_min_triangle_area_l1560_156098

/-- The minimum area of a triangle with vertices A(0,0), B(30,10), and C(p,q) where p and q are integers -/
theorem min_triangle_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (30, 10)
  ∃ (min_area : ℝ), min_area = 5/2 ∧ 
    ∀ (p q : ℤ), 
      let C : ℝ × ℝ := (p, q)
      let area := (1/2) * |(-p : ℝ) + 3*q|
      area ≥ min_area :=
by sorry

end NUMINAMATH_CALUDE_min_triangle_area_l1560_156098


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l1560_156035

theorem mans_speed_with_current 
  (current_speed : ℝ) 
  (speed_against_current : ℝ) 
  (h1 : current_speed = 2.5)
  (h2 : speed_against_current = 10) : 
  ∃ (speed_with_current : ℝ), speed_with_current = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_mans_speed_with_current_l1560_156035


namespace NUMINAMATH_CALUDE_tom_payment_proof_l1560_156007

/-- Represents the purchase of a fruit with its quantity and price per kg -/
structure FruitPurchase where
  quantity : Float
  pricePerKg : Float

/-- Calculates the total cost of a fruit purchase -/
def calculateCost (purchase : FruitPurchase) : Float :=
  purchase.quantity * purchase.pricePerKg

/-- Represents Tom's fruit shopping trip -/
def tomShopping : List FruitPurchase := [
  { quantity := 15.3, pricePerKg := 1.85 },  -- apples
  { quantity := 12.7, pricePerKg := 2.45 },  -- mangoes
  { quantity := 10.5, pricePerKg := 3.20 },  -- grapes
  { quantity := 6.2,  pricePerKg := 4.50 }   -- strawberries
]

/-- The discount rate applied to the total bill -/
def discountRate : Float := 0.10

/-- The sales tax rate applied to the discounted amount -/
def taxRate : Float := 0.06

/-- Calculates the final amount Tom pays after discount and tax -/
def calculateFinalAmount (purchases : List FruitPurchase) (discount : Float) (tax : Float) : Float :=
  let totalCost := purchases.map calculateCost |>.sum
  let discountedCost := totalCost * (1 - discount)
  let finalCost := discountedCost * (1 + tax)
  (finalCost * 100).round / 100  -- Round to nearest cent

theorem tom_payment_proof :
  calculateFinalAmount tomShopping discountRate taxRate = 115.36 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_proof_l1560_156007


namespace NUMINAMATH_CALUDE_max_distance_complex_numbers_l1560_156079

theorem max_distance_complex_numbers (z : ℂ) (h : Complex.abs z = 3) :
  Complex.abs ((1 + 2*Complex.I)*z - z^2) ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_complex_numbers_l1560_156079


namespace NUMINAMATH_CALUDE_pencil_cost_solution_l1560_156003

/-- Calculates the cost of a pencil given the cost of a magazine, coupon discount, and total spent amount. -/
def pencil_cost (magazine_cost coupon_discount total_spent : ℚ) : ℚ :=
  total_spent + coupon_discount - magazine_cost

/-- Theorem stating that given the specific values in the problem, the pencil cost is $0.50 -/
theorem pencil_cost_solution :
  pencil_cost 0.85 0.35 1 = 0.5 := by sorry

end NUMINAMATH_CALUDE_pencil_cost_solution_l1560_156003


namespace NUMINAMATH_CALUDE_intersection_line_equation_distance_line_equation_l1560_156059

/-- Line represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def l₁ : Line := { a := 1, b := -2, c := 3 }
def l₂ : Line := { a := 2, b := 3, c := -8 }
def l₃ : Line := { a := 1, b := 3, c := 1 }
def P : Point := { x := 0, y := 4 }

/-- Intersection point of two lines -/
def intersection (l1 l2 : Line) : Point := sorry

/-- Check if two lines are parallel -/
def is_parallel (l1 l2 : Line) : Prop := sorry

/-- Distance from a point to a line -/
def distance_point_to_line (p : Point) (l : Line) : ℝ := sorry

/-- The intersection point of l₁ and l₂ -/
def M : Point := intersection l₁ l₂

theorem intersection_line_equation :
  ∃ (l : Line), l.a = 1 ∧ l.b = 3 ∧ l.c = -7 ∧
  (M.x * l.a + M.y * l.b + l.c = 0) ∧
  is_parallel l l₃ :=
sorry

theorem distance_line_equation :
  ∃ (l : Line), 
  ((l.a = 1 ∧ l.b = 0 ∧ l.c = -1) ∨ (l.a = 3 ∧ l.b = 4 ∧ l.c = -11)) ∧
  (M.x * l.a + M.y * l.b + l.c = 0) ∧
  distance_point_to_line P l = 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_line_equation_distance_line_equation_l1560_156059


namespace NUMINAMATH_CALUDE_m_range_theorem_l1560_156085

open Set

noncomputable def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

noncomputable def Q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

def p_set : Set ℝ := {x | P x}
def q_set (m : ℝ) : Set ℝ := {x | Q x m}

theorem m_range_theorem :
  ∀ m : ℝ, (0 < m ∧ m ≤ 3) ↔ 
    (m > 0 ∧ q_set m ⊂ p_set ∧ q_set m ≠ p_set) :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l1560_156085


namespace NUMINAMATH_CALUDE_inclination_angle_tangent_l1560_156073

theorem inclination_angle_tangent (α : ℝ) : 
  (∃ (x y : ℝ), 2 * x + y + 1 = 0 ∧ α = Real.arctan (-2)) → 
  Real.tan (α - π / 4) = 3 := by
sorry

end NUMINAMATH_CALUDE_inclination_angle_tangent_l1560_156073


namespace NUMINAMATH_CALUDE_complex_root_magnitude_l1560_156001

theorem complex_root_magnitude (n : ℕ) (a : ℝ) (z : ℂ) 
  (h1 : n ≥ 2) 
  (h2 : 0 < a) 
  (h3 : a < (n + 1 : ℝ) / (n - 1 : ℝ)) 
  (h4 : z^(n+1) - a * z^n + a * z - 1 = 0) : 
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_magnitude_l1560_156001


namespace NUMINAMATH_CALUDE_lee_fruit_loading_l1560_156029

/-- Represents the capacity of a large truck in tons -/
def large_truck_capacity : ℕ := 15

/-- Represents the number of large trucks used -/
def num_large_trucks : ℕ := 8

/-- Represents the total amount of fruits to be loaded in tons -/
def total_fruits : ℕ := num_large_trucks * large_truck_capacity

theorem lee_fruit_loading :
  total_fruits = 120 :=
by sorry

end NUMINAMATH_CALUDE_lee_fruit_loading_l1560_156029


namespace NUMINAMATH_CALUDE_stick_ratio_proof_l1560_156010

/-- Prove that the ratio of the uncovered portion of Pat's stick to Sarah's stick is 1/2 -/
theorem stick_ratio_proof (pat_stick : ℕ) (pat_covered : ℕ) (jane_stick : ℕ) (sarah_stick : ℕ) : 
  pat_stick = 30 →
  pat_covered = 7 →
  jane_stick = 22 →
  sarah_stick = jane_stick + 24 →
  (pat_stick - pat_covered : ℚ) / sarah_stick = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_stick_ratio_proof_l1560_156010


namespace NUMINAMATH_CALUDE_division_algorithm_l1560_156040

theorem division_algorithm (x y : ℤ) (hx : x ≥ 0) (hy : y > 0) :
  ∃! (q r : ℤ), x = q * y + r ∧ 0 ≤ r ∧ r < y := by
  sorry

end NUMINAMATH_CALUDE_division_algorithm_l1560_156040
