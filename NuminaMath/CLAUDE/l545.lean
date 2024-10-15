import Mathlib

namespace NUMINAMATH_CALUDE_units_digit_of_cube_minus_square_l545_54573

def n : ℕ := 9867

theorem units_digit_of_cube_minus_square :
  (n^3 - n^2) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_cube_minus_square_l545_54573


namespace NUMINAMATH_CALUDE_valid_pairs_eq_expected_l545_54578

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

def valid_pairs : Finset (ℕ × ℕ) :=
  (divisors 660).product (divisors 72) |>.filter (λ (a, b) => a - b = 4)

theorem valid_pairs_eq_expected : valid_pairs = {(6, 2), (10, 6), (12, 8), (22, 18)} := by
  sorry

end NUMINAMATH_CALUDE_valid_pairs_eq_expected_l545_54578


namespace NUMINAMATH_CALUDE_max_profit_allocation_l545_54560

/-- Represents the profit function for Project A -/
def p (a : ℝ) (t : ℝ) : ℝ := a * t^3 + 21 * t

/-- Represents the profit function for Project B -/
def g (a : ℝ) (b : ℝ) (t : ℝ) : ℝ := -2 * a * (t - b)^2

/-- Represents the total profit function -/
def f (a : ℝ) (b : ℝ) (x : ℝ) : ℝ := p a x + g a b (200 - x)

/-- Theorem stating the maximum profit and optimal investment allocation -/
theorem max_profit_allocation (a b : ℝ) :
  (∀ t, p a t = -1/60 * t^3 + 21 * t) →
  (∀ t, g a b t = 1/30 * (t - 110)^2) →
  (p a 30 = 180) →
  (g a b 170 = 120) →
  (b < 200) →
  (∃ x₀, x₀ ∈ Set.Icc 10 190 ∧ 
    f a b x₀ = 453.6 ∧
    (∀ x, x ∈ Set.Icc 10 190 → f a b x ≤ f a b x₀) ∧
    x₀ = 18) := by
  sorry


end NUMINAMATH_CALUDE_max_profit_allocation_l545_54560


namespace NUMINAMATH_CALUDE_number_value_l545_54527

theorem number_value (tens : ℕ) (ones : ℕ) (tenths : ℕ) (hundredths : ℕ) :
  tens = 21 →
  ones = 8 →
  tenths = 5 →
  hundredths = 34 →
  (tens * 10 : ℚ) + ones + (tenths : ℚ) / 10 + (hundredths : ℚ) / 100 = 218.84 :=
by sorry

end NUMINAMATH_CALUDE_number_value_l545_54527


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l545_54510

/-- An isosceles triangle with given altitude and perimeter has area 75 -/
theorem isosceles_triangle_area (b s h : ℝ) : 
  h = 10 →                -- altitude is 10
  2 * s + 2 * b = 40 →    -- perimeter is 40
  s^2 = b^2 + h^2 →       -- Pythagorean theorem
  (1/2) * (2*b) * h = 75  -- area is 75
  := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l545_54510


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l545_54575

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- The problem statement -/
theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2, m^2)
  parallel a b → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l545_54575


namespace NUMINAMATH_CALUDE_sin_shift_l545_54520

theorem sin_shift (x : ℝ) : 
  Real.sin (2 * x - π / 3) = Real.sin (2 * (x - π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l545_54520


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l545_54523

theorem inscribed_squares_ratio (a b c x y : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  x * (a + b - x) = a * b →
  y * (a + b - y) = (c - y)^2 →
  x / y = 37 / 35 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l545_54523


namespace NUMINAMATH_CALUDE_correct_factorization_l545_54515

theorem correct_factorization (x : ℝ) : x^2 - 4*x + 4 = (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l545_54515


namespace NUMINAMATH_CALUDE_x_y_negative_l545_54530

theorem x_y_negative (x y : ℝ) (h1 : x - y > x) (h2 : 3 * x + 2 * y < 2 * y) : x < 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_x_y_negative_l545_54530


namespace NUMINAMATH_CALUDE_arithmetic_sum_2_to_20_l545_54507

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  (a₁ + aₙ) * n / 2

theorem arithmetic_sum_2_to_20 :
  arithmetic_sum 2 20 2 = 110 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_2_to_20_l545_54507


namespace NUMINAMATH_CALUDE_triangle_perimeter_l545_54570

/-- Given a right-angled triangle with hypotenuse 5000 km and one other side 4000 km,
    the sum of all sides is 12000 km. -/
theorem triangle_perimeter (a b c : ℝ) (h1 : a = 5000) (h2 : b = 4000) 
    (h3 : a^2 = b^2 + c^2) : a + b + c = 12000 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l545_54570


namespace NUMINAMATH_CALUDE_f_properties_l545_54537

-- Define the function f
def f (x : ℝ) : ℝ := -x - x^3

-- State the theorem
theorem f_properties (x₁ x₂ : ℝ) (h : x₁ + x₂ ≤ 0) :
  (f x₁ * f (-x₁) ≤ 0) ∧ (f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l545_54537


namespace NUMINAMATH_CALUDE_tangent_line_at_P_l545_54518

/-- The curve function -/
def f (x : ℝ) : ℝ := x^2 + 2

/-- The point of tangency -/
def P : ℝ × ℝ := (1, 3)

/-- The tangent line function -/
def tangent_line (x : ℝ) : ℝ := 2*x - 1

theorem tangent_line_at_P : 
  (∀ x : ℝ, tangent_line x = 2*x - 1) ∧ 
  (tangent_line P.1 = P.2) ∧
  (HasDerivAt f 2 P.1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_P_l545_54518


namespace NUMINAMATH_CALUDE_kendras_change_l545_54506

/-- Calculates the change received after a purchase -/
def calculate_change (toy_price hat_price : ℕ) (num_toys num_hats : ℕ) (total_money : ℕ) : ℕ :=
  total_money - (toy_price * num_toys + hat_price * num_hats)

/-- Proves that Kendra's change is $30 -/
theorem kendras_change :
  let toy_price : ℕ := 20
  let hat_price : ℕ := 10
  let num_toys : ℕ := 2
  let num_hats : ℕ := 3
  let total_money : ℕ := 100
  calculate_change toy_price hat_price num_toys num_hats total_money = 30 := by
  sorry

#eval calculate_change 20 10 2 3 100

end NUMINAMATH_CALUDE_kendras_change_l545_54506


namespace NUMINAMATH_CALUDE_chest_contents_l545_54569

-- Define the types of coins
inductive CoinType
| Gold
| Silver
| Copper

-- Define the chests
structure Chest where
  inscription : CoinType → Prop
  content : CoinType

-- Define the problem setup
def chestProblem (c1 c2 c3 : Chest) : Prop :=
  -- All inscriptions are incorrect
  (¬c1.inscription c1.content) ∧
  (¬c2.inscription c2.content) ∧
  (¬c3.inscription c3.content) ∧
  -- Each chest contains a different type of coin
  (c1.content ≠ c2.content) ∧
  (c2.content ≠ c3.content) ∧
  (c3.content ≠ c1.content) ∧
  -- Inscriptions on the chests
  (c1.inscription = fun c => c = CoinType.Gold) ∧
  (c2.inscription = fun c => c = CoinType.Silver) ∧
  (c3.inscription = fun c => c = CoinType.Gold ∨ c = CoinType.Silver)

-- The theorem to prove
theorem chest_contents (c1 c2 c3 : Chest) 
  (h : chestProblem c1 c2 c3) : 
  c1.content = CoinType.Silver ∧ 
  c2.content = CoinType.Gold ∧ 
  c3.content = CoinType.Copper := by
  sorry

end NUMINAMATH_CALUDE_chest_contents_l545_54569


namespace NUMINAMATH_CALUDE_cube_shape_product_l545_54557

/-- Represents a 3D shape constructed from identical cubes. -/
structure CubeShape where
  /-- The number of cubes in the shape. -/
  num_cubes : ℕ
  /-- Predicate that returns true if the shape satisfies the given views. -/
  satisfies_views : Bool

/-- The minimum number of cubes that can form the shape satisfying the given views. -/
def min_cubes : ℕ := 8

/-- The maximum number of cubes that can form the shape satisfying the given views. -/
def max_cubes : ℕ := 16

/-- Theorem stating that the product of the maximum and minimum number of cubes is 128. -/
theorem cube_shape_product :
  min_cubes * max_cubes = 128 ∧
  ∀ shape : CubeShape, shape.satisfies_views →
    min_cubes ≤ shape.num_cubes ∧ shape.num_cubes ≤ max_cubes :=
by sorry

end NUMINAMATH_CALUDE_cube_shape_product_l545_54557


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l545_54554

theorem unique_solution_for_equation : 
  ∃! (p n : ℕ), 
    n > 0 ∧ 
    Nat.Prime p ∧ 
    17^n * 2^(n^2) - p = (2^(n^2 + 3) + 2^(n^2) - 1) * n^2 ∧ 
    p = 17 ∧ 
    n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l545_54554


namespace NUMINAMATH_CALUDE_jorge_corn_yield_ratio_l545_54576

/-- Represents the yield ratio problem for Jorge's corn fields --/
theorem jorge_corn_yield_ratio :
  let total_acres : ℚ := 60
  let good_soil_yield : ℚ := 400
  let clay_rich_proportion : ℚ := 1/3
  let total_yield : ℚ := 20000
  let clay_rich_acres : ℚ := total_acres * clay_rich_proportion
  let good_soil_acres : ℚ := total_acres - clay_rich_acres
  let good_soil_total_yield : ℚ := good_soil_acres * good_soil_yield
  let clay_rich_total_yield : ℚ := total_yield - good_soil_total_yield
  let clay_rich_yield : ℚ := clay_rich_total_yield / clay_rich_acres
  clay_rich_yield / good_soil_yield = 1/2 :=
by sorry


end NUMINAMATH_CALUDE_jorge_corn_yield_ratio_l545_54576


namespace NUMINAMATH_CALUDE_anyas_initial_seat_l545_54511

/-- Represents the seat numbers in the theater --/
inductive Seat
| one
| two
| three
| four
| five

/-- Represents the friends --/
inductive Friend
| Anya
| Varya
| Galya
| Diana
| Ella

/-- Represents the seating arrangement before and after Anya left --/
structure SeatingArrangement where
  seats : Friend → Seat

/-- Moves a seat to the right --/
def moveRight (s : Seat) : Seat :=
  match s with
  | Seat.one => Seat.two
  | Seat.two => Seat.three
  | Seat.three => Seat.four
  | Seat.four => Seat.five
  | Seat.five => Seat.five

/-- Moves a seat to the left --/
def moveLeft (s : Seat) : Seat :=
  match s with
  | Seat.one => Seat.one
  | Seat.two => Seat.one
  | Seat.three => Seat.two
  | Seat.four => Seat.three
  | Seat.five => Seat.four

/-- Theorem stating Anya's initial seat was four --/
theorem anyas_initial_seat (initial final : SeatingArrangement) :
  (final.seats Friend.Varya = moveRight (initial.seats Friend.Varya)) →
  (final.seats Friend.Galya = moveLeft (moveLeft (initial.seats Friend.Galya))) →
  (final.seats Friend.Diana = initial.seats Friend.Ella) →
  (final.seats Friend.Ella = initial.seats Friend.Diana) →
  (final.seats Friend.Anya = Seat.five) →
  (initial.seats Friend.Anya = Seat.four) :=
by
  sorry


end NUMINAMATH_CALUDE_anyas_initial_seat_l545_54511


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l545_54586

/-- Sum of a geometric sequence -/
def geometric_sum (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a₀ : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a₀ r n = 3280/6561 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l545_54586


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_sum_fifteen_minus_sqrt500_and_conjugate_l545_54533

/-- The sum of a number and its radical conjugate is twice the real part of the number. -/
theorem sum_with_radical_conjugate (a : ℝ) (b : ℝ) (h : 0 ≤ b) :
  (a - Real.sqrt b) + (a + Real.sqrt b) = 2 * a := by
  sorry

/-- The sum of 15 - √500 and its radical conjugate is 30. -/
theorem sum_fifteen_minus_sqrt500_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_sum_fifteen_minus_sqrt500_and_conjugate_l545_54533


namespace NUMINAMATH_CALUDE_smaller_integer_problem_l545_54583

theorem smaller_integer_problem (x y : ℤ) 
  (sum_eq : x + y = 30)
  (relation : 2 * y = 5 * x - 10) :
  x = 10 ∧ x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smaller_integer_problem_l545_54583


namespace NUMINAMATH_CALUDE_probability_of_selecting_girl_l545_54524

-- Define the total number of candidates
def total_candidates : ℕ := 3 + 1

-- Define the number of girls
def number_of_girls : ℕ := 1

-- Theorem statement
theorem probability_of_selecting_girl :
  (number_of_girls : ℚ) / total_candidates = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selecting_girl_l545_54524


namespace NUMINAMATH_CALUDE_solve_parking_ticket_problem_l545_54580

def parking_ticket_problem (first_two_ticket_cost : ℚ) (third_ticket_fraction : ℚ) (james_remaining_money : ℚ) : Prop :=
  let total_cost := 2 * first_two_ticket_cost + third_ticket_fraction * first_two_ticket_cost
  let james_paid := total_cost - james_remaining_money
  let roommate_paid := total_cost - james_paid
  (roommate_paid / total_cost) = 13 / 14

theorem solve_parking_ticket_problem :
  parking_ticket_problem 150 (1/3) 325 := by
  sorry

end NUMINAMATH_CALUDE_solve_parking_ticket_problem_l545_54580


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_5_and_24_l545_54543

theorem smallest_number_divisible_by_5_and_24 : ∃ n : ℕ+, 
  (∀ m : ℕ+, 5 ∣ m ∧ 24 ∣ m → n ≤ m) ∧ 5 ∣ n ∧ 24 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_5_and_24_l545_54543


namespace NUMINAMATH_CALUDE_left_handed_classical_music_lovers_l545_54585

theorem left_handed_classical_music_lovers (total : ℕ) (left_handed : ℕ) (classical_music : ℕ) (right_handed_non_classical : ℕ) 
  (h1 : total = 25)
  (h2 : left_handed = 10)
  (h3 : classical_music = 18)
  (h4 : right_handed_non_classical = 3)
  (h5 : left_handed + (total - left_handed) = total) :
  ∃ y : ℕ, y = 6 ∧ 
    y + (left_handed - y) + (classical_music - y) + right_handed_non_classical = total :=
by sorry

end NUMINAMATH_CALUDE_left_handed_classical_music_lovers_l545_54585


namespace NUMINAMATH_CALUDE_log_expression_equals_negative_one_l545_54565

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_negative_one :
  log10 (5 / 2) + 2 * log10 2 - (1 / 2)⁻¹ = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_negative_one_l545_54565


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l545_54501

def small_radius : ℝ := 3
def large_radius : ℝ := 5

def left_rectangle_width : ℝ := small_radius
def left_rectangle_height : ℝ := 2 * small_radius
def right_rectangle_width : ℝ := large_radius
def right_rectangle_height : ℝ := 2 * large_radius

def isosceles_triangle_leg : ℝ := small_radius

theorem shaded_area_calculation :
  let left_rectangle_area := left_rectangle_width * left_rectangle_height
  let right_rectangle_area := right_rectangle_width * right_rectangle_height
  let left_semicircle_area := (1/2) * Real.pi * small_radius^2
  let right_semicircle_area := (1/2) * Real.pi * large_radius^2
  let triangle_area := (1/2) * isosceles_triangle_leg^2
  let total_shaded_area := (left_rectangle_area - left_semicircle_area - triangle_area) + 
                           (right_rectangle_area - right_semicircle_area)
  total_shaded_area = 63.5 - 17 * Real.pi := by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l545_54501


namespace NUMINAMATH_CALUDE_sphere_volume_from_tetrahedron_surface_l545_54525

theorem sphere_volume_from_tetrahedron_surface (s : ℝ) (V : ℝ) : 
  s = 3 →
  (4 * π * (V / ((4/3) * π))^((1:ℝ)/3)^2) = (4 * s^2 * Real.sqrt 3) →
  V = (27 * Real.sqrt 2) / Real.sqrt π :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_tetrahedron_surface_l545_54525


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l545_54539

theorem sum_of_reciprocals_of_roots (p q : ℝ) : 
  p^2 - 20*p + 9 = 0 → q^2 - 20*q + 9 = 0 → p ≠ q → (1/p + 1/q) = 20/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l545_54539


namespace NUMINAMATH_CALUDE_maisys_current_wage_l545_54532

/-- Proves that Maisy's current wage is $10 per hour given the job conditions --/
theorem maisys_current_wage (current_hours new_hours : ℕ) 
  (new_wage new_bonus difference : ℚ) :
  current_hours = 8 →
  new_hours = 4 →
  new_wage = 15 →
  new_bonus = 35 →
  difference = 15 →
  (new_hours : ℚ) * new_wage + new_bonus = 
    (current_hours : ℚ) * (10 : ℚ) + difference →
  10 = 10 := by
  sorry

#check maisys_current_wage

end NUMINAMATH_CALUDE_maisys_current_wage_l545_54532


namespace NUMINAMATH_CALUDE_power_of_square_l545_54500

theorem power_of_square (b : ℝ) : (b^2)^3 = b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_square_l545_54500


namespace NUMINAMATH_CALUDE_continuous_piecewise_sum_l545_54584

/-- A piecewise function f(x) defined on the real line. -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then a * x + 6
  else if x ≥ -3 then x - 7
  else 3 * x - b

/-- The function f is continuous on the real line. -/
def is_continuous (a b : ℝ) : Prop :=
  Continuous (f a b)

/-- If f is continuous, then a + b = -7/3. -/
theorem continuous_piecewise_sum (a b : ℝ) :
  is_continuous a b → a + b = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_continuous_piecewise_sum_l545_54584


namespace NUMINAMATH_CALUDE_fractional_equation_root_l545_54503

theorem fractional_equation_root (n : ℤ) : 
  (∃ x : ℝ, x > 0 ∧ (x - 2) / (x - 3) = (n + 1) / (3 - x)) → n = -2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l545_54503


namespace NUMINAMATH_CALUDE_business_value_calculation_l545_54504

theorem business_value_calculation (man_share : ℚ) (sold_portion : ℚ) (sale_price : ℕ) :
  man_share = 1/3 →
  sold_portion = 3/5 →
  sale_price = 2000 →
  ∃ (total_value : ℕ), total_value = 10000 ∧ 
    (sold_portion * man_share * total_value : ℚ) = sale_price := by
  sorry

end NUMINAMATH_CALUDE_business_value_calculation_l545_54504


namespace NUMINAMATH_CALUDE_c_range_l545_54579

theorem c_range (c : ℝ) (h_c_pos : c > 0) : 
  (((∀ x y : ℝ, x < y → c^x > c^y) ↔ ¬(∀ x : ℝ, x + c > 0)) ∧ 
   ((∀ x : ℝ, x + c > 0) ↔ ¬(∀ x y : ℝ, x < y → c^x > c^y))) → 
  (c > 0 ∧ c ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_c_range_l545_54579


namespace NUMINAMATH_CALUDE_greatest_valid_partition_l545_54531

/-- A partition of positive integers into k subsets satisfying the sum property -/
def ValidPartition (k : ℕ) : Prop :=
  ∃ (A : Fin k → Set ℕ), 
    (∀ i j, i ≠ j → A i ∩ A j = ∅) ∧ 
    (⋃ i, A i) = {n : ℕ | n > 0} ∧
    ∀ (n : ℕ) (i : Fin k), n ≥ 15 → 
      ∃ (x y : ℕ), x ∈ A i ∧ y ∈ A i ∧ x ≠ y ∧ x + y = n

/-- The main theorem: 3 is the greatest positive integer satisfying the property -/
theorem greatest_valid_partition : 
  ValidPartition 3 ∧ ∀ k > 3, ¬ValidPartition k :=
sorry

end NUMINAMATH_CALUDE_greatest_valid_partition_l545_54531


namespace NUMINAMATH_CALUDE_pizza_slices_remaining_l545_54548

theorem pizza_slices_remaining (initial_slices : ℕ) 
  (breakfast_slices : ℕ) (lunch_slices : ℕ) (snack_slices : ℕ) (dinner_slices : ℕ) :
  initial_slices = 15 →
  breakfast_slices = 4 →
  lunch_slices = 2 →
  snack_slices = 2 →
  dinner_slices = 5 →
  initial_slices - (breakfast_slices + lunch_slices + snack_slices + dinner_slices) = 2 :=
by sorry

end NUMINAMATH_CALUDE_pizza_slices_remaining_l545_54548


namespace NUMINAMATH_CALUDE_network_connections_l545_54577

/-- Given a network of switches where each switch is connected to exactly
    four others, this function calculates the total number of connections. -/
def calculate_connections (num_switches : ℕ) : ℕ :=
  (num_switches * 4) / 2

/-- Theorem stating that in a network of 30 switches, where each switch
    is directly connected to exactly 4 other switches, the total number
    of connections is 60. -/
theorem network_connections :
  calculate_connections 30 = 60 := by
  sorry

#eval calculate_connections 30

end NUMINAMATH_CALUDE_network_connections_l545_54577


namespace NUMINAMATH_CALUDE_strawberry_harvest_l545_54598

/-- Calculates the expected strawberry harvest for a rectangular garden. -/
theorem strawberry_harvest (length width plants_per_sqft avg_yield : ℕ) : 
  length = 10 →
  width = 12 →
  plants_per_sqft = 5 →
  avg_yield = 10 →
  length * width * plants_per_sqft * avg_yield = 6000 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_harvest_l545_54598


namespace NUMINAMATH_CALUDE_total_balls_bought_l545_54552

/-- Represents the amount of money Mr. Li has -/
def total_money : ℚ := 1

/-- The cost of one plastic ball -/
def plastic_ball_cost : ℚ := 1 / 60

/-- The cost of one glass ball -/
def glass_ball_cost : ℚ := 1 / 36

/-- The cost of one wooden ball -/
def wooden_ball_cost : ℚ := 1 / 45

/-- The number of plastic balls Mr. Li buys -/
def plastic_balls_bought : ℕ := 10

/-- The number of glass balls Mr. Li buys -/
def glass_balls_bought : ℕ := 10

/-- Theorem stating the total number of balls Mr. Li buys -/
theorem total_balls_bought : 
  ∃ (wooden_balls : ℕ), 
    (plastic_balls_bought * plastic_ball_cost + 
     glass_balls_bought * glass_ball_cost + 
     wooden_balls * wooden_ball_cost = total_money) ∧
    (plastic_balls_bought + glass_balls_bought + wooden_balls = 45) :=
by sorry

end NUMINAMATH_CALUDE_total_balls_bought_l545_54552


namespace NUMINAMATH_CALUDE_inequality_preserved_under_subtraction_l545_54535

theorem inequality_preserved_under_subtraction (a b c : ℝ) : 
  a < b → a - 2*c < b - 2*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preserved_under_subtraction_l545_54535


namespace NUMINAMATH_CALUDE_complementary_probability_l545_54521

theorem complementary_probability (P_snow : ℚ) (h : P_snow = 2/5) :
  1 - P_snow = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_complementary_probability_l545_54521


namespace NUMINAMATH_CALUDE_division_4863_by_97_l545_54597

theorem division_4863_by_97 : ∃ (q r : ℤ), 4863 = 97 * q + r ∧ 0 ≤ r ∧ r < 97 ∧ q = 50 ∧ r = 40 := by
  sorry

end NUMINAMATH_CALUDE_division_4863_by_97_l545_54597


namespace NUMINAMATH_CALUDE_evaluate_expression_l545_54593

/-- Given x = 4 and z = -2, prove that z(z - 4x) = 36 -/
theorem evaluate_expression (x z : ℝ) (hx : x = 4) (hz : z = -2) : z * (z - 4 * x) = 36 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l545_54593


namespace NUMINAMATH_CALUDE_houses_with_pool_l545_54526

theorem houses_with_pool (total : ℕ) (garage : ℕ) (neither : ℕ) (both : ℕ) 
  (h_total : total = 85)
  (h_garage : garage = 50)
  (h_neither : neither = 30)
  (h_both : both = 35) :
  ∃ pool : ℕ, pool = 40 ∧ total = (garage - both) + (pool - both) + both + neither :=
by sorry

end NUMINAMATH_CALUDE_houses_with_pool_l545_54526


namespace NUMINAMATH_CALUDE_max_value_constraint_l545_54592

theorem max_value_constraint (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (M : ℝ), M = 19 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 = 4 → x'^2 + 8*y' + 3 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l545_54592


namespace NUMINAMATH_CALUDE_yonder_license_plates_l545_54514

/-- The number of possible letters in a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible symbols in a license plate. -/
def num_symbols : ℕ := 5

/-- The number of possible digits in a license plate. -/
def num_digits : ℕ := 10

/-- The number of letter positions in a license plate. -/
def letter_positions : ℕ := 2

/-- The number of symbol positions in a license plate. -/
def symbol_positions : ℕ := 1

/-- The number of digit positions in a license plate. -/
def digit_positions : ℕ := 4

/-- The total number of valid license plates in Yonder. -/
def total_license_plates : ℕ := 33800000

/-- Theorem stating the total number of valid license plates in Yonder. -/
theorem yonder_license_plates :
  (num_letters ^ letter_positions) * (num_symbols ^ symbol_positions) * (num_digits ^ digit_positions) = total_license_plates :=
by sorry

end NUMINAMATH_CALUDE_yonder_license_plates_l545_54514


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l545_54599

theorem imaginary_part_of_z (z : ℂ) : z = (2 - Complex.I) * Complex.I → z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l545_54599


namespace NUMINAMATH_CALUDE_min_xy_l545_54563

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 2 * x + 8 * y) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' * y' = 2 * x' + 8 * y' → x * y ≤ x' * y') →
  x = 16 ∧ y = 4 := by
sorry

end NUMINAMATH_CALUDE_min_xy_l545_54563


namespace NUMINAMATH_CALUDE_parabola_directrix_l545_54550

/-- The directrix of the parabola y = -3x^2 + 6x - 5 is y = -23/12 -/
theorem parabola_directrix : ∀ x y : ℝ, 
  y = -3 * x^2 + 6 * x - 5 → 
  ∃ (k : ℝ), k = -23/12 ∧ (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
    (p.1 - 1)^2 + (p.2 - k)^2 = (p.2 + 2)^2 / 12) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l545_54550


namespace NUMINAMATH_CALUDE_mark_collection_amount_l545_54519

/-- Calculates the total amount collected by Mark for the homeless -/
def totalAmountCollected (householdsPerDay : ℕ) (days : ℕ) (donationAmount : ℕ) : ℕ :=
  let totalHouseholds := householdsPerDay * days
  let donatingHouseholds := totalHouseholds / 2
  donatingHouseholds * donationAmount

/-- Proves that Mark collected $2000 given the problem conditions -/
theorem mark_collection_amount :
  totalAmountCollected 20 5 40 = 2000 := by
  sorry

#eval totalAmountCollected 20 5 40

end NUMINAMATH_CALUDE_mark_collection_amount_l545_54519


namespace NUMINAMATH_CALUDE_cone_height_from_sphere_l545_54536

/-- The height of a cone formed by melting and reshaping a sphere -/
theorem cone_height_from_sphere (r_sphere : ℝ) (r_cone h_cone : ℝ) : 
  r_sphere = 5 * 3^2 →
  (2 * π * r_cone * (3 * r_cone)) = 3 * (π * r_cone^2) →
  (4/3) * π * r_sphere^3 = (1/3) * π * r_cone^2 * h_cone →
  h_cone = 20 := by
  sorry

#check cone_height_from_sphere

end NUMINAMATH_CALUDE_cone_height_from_sphere_l545_54536


namespace NUMINAMATH_CALUDE_trapezoid_theorem_l545_54502

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midline_ratio : ℝ
  equal_area_segment : ℝ
  longer_base_condition : longer_base = shorter_base + 150
  midline_ratio_condition : midline_ratio = 3 / 4
  equal_area_condition : equal_area_segment > shorter_base ∧ equal_area_segment < longer_base

/-- The main theorem about the trapezoid -/
theorem trapezoid_theorem (t : Trapezoid) : 
  ⌊(t.equal_area_segment^2) / 150⌋ = 416 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_theorem_l545_54502


namespace NUMINAMATH_CALUDE_m_divided_by_8_l545_54555

theorem m_divided_by_8 (m : ℕ) (h : m = 16^500) : m / 8 = 2^1997 := by
  sorry

end NUMINAMATH_CALUDE_m_divided_by_8_l545_54555


namespace NUMINAMATH_CALUDE_purchase_total_cost_l545_54572

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℚ := 2.44

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 2

/-- The cost of a single soda in dollars -/
def soda_cost : ℚ := 0.87

/-- The number of sodas purchased -/
def num_sodas : ℕ := 4

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := 8.36

theorem purchase_total_cost : 
  (num_sandwiches : ℚ) * sandwich_cost + (num_sodas : ℚ) * soda_cost = total_cost := by
  sorry

end NUMINAMATH_CALUDE_purchase_total_cost_l545_54572


namespace NUMINAMATH_CALUDE_faith_change_is_ten_l545_54509

/-- The change Faith receives from her purchase at the baking shop. -/
def faith_change : ℕ :=
  let flour_cost : ℕ := 5
  let cake_stand_cost : ℕ := 28
  let total_cost : ℕ := flour_cost + cake_stand_cost
  let bill_payment : ℕ := 2 * 20
  let coin_payment : ℕ := 3
  let total_payment : ℕ := bill_payment + coin_payment
  total_payment - total_cost

/-- Theorem stating that Faith receives $10 in change. -/
theorem faith_change_is_ten : faith_change = 10 := by
  sorry

end NUMINAMATH_CALUDE_faith_change_is_ten_l545_54509


namespace NUMINAMATH_CALUDE_computer_price_increase_l545_54516

theorem computer_price_increase (original_price : ℝ) : 
  original_price * 1.3 = 351 → 2 * original_price = 540 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l545_54516


namespace NUMINAMATH_CALUDE_range_of_expression_l545_54551

theorem range_of_expression (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) :
  -π/6 < 2*α - β/2 ∧ 2*α - β/2 < π :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l545_54551


namespace NUMINAMATH_CALUDE_multiply_101_by_101_l545_54505

theorem multiply_101_by_101 : 101 * 101 = 10201 := by sorry

end NUMINAMATH_CALUDE_multiply_101_by_101_l545_54505


namespace NUMINAMATH_CALUDE_oyster_ratio_proof_l545_54545

/-- Proves that the ratio of oysters on the second day to the first day is 1:2 -/
theorem oyster_ratio_proof (oysters_day1 crabs_day1 total_count : ℕ) 
  (h1 : oysters_day1 = 50)
  (h2 : crabs_day1 = 72)
  (h3 : total_count = 195)
  (h4 : ∃ (oysters_day2 crabs_day2 : ℕ), 
    crabs_day2 = 2 * crabs_day1 / 3 ∧ 
    oysters_day1 + crabs_day1 + oysters_day2 + crabs_day2 = total_count) :
  ∃ (oysters_day2 : ℕ), oysters_day2 * 2 = oysters_day1 :=
sorry

end NUMINAMATH_CALUDE_oyster_ratio_proof_l545_54545


namespace NUMINAMATH_CALUDE_current_at_6_seconds_l545_54574

/-- The charge function Q(t) representing the amount of electricity flowing through a conductor. -/
def Q (t : ℝ) : ℝ := 3 * t^2 - 3 * t + 4

/-- The current function I(t) derived from Q(t). -/
def I (t : ℝ) : ℝ := 6 * t - 3

/-- Theorem stating that the current at t = 6 seconds is 33 amperes. -/
theorem current_at_6_seconds :
  I 6 = 33 := by sorry

end NUMINAMATH_CALUDE_current_at_6_seconds_l545_54574


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_seven_l545_54538

theorem consecutive_integers_sqrt_seven (a b : ℤ) : 
  (b = a + 1) →  -- a and b are consecutive integers
  (a < Real.sqrt 7) →  -- a < √7
  (Real.sqrt 7 < b) →  -- √7 < b
  a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_seven_l545_54538


namespace NUMINAMATH_CALUDE_slope_is_constant_l545_54541

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define a point in the first quadrant
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the line l
def line (k m x y : ℝ) : Prop := y = k * x + m

-- Define the condition for the areas
def area_condition (x₁ y₁ x₂ y₂ m k : ℝ) : Prop :=
  (y₁^2 + y₂^2) / (y₁ * y₂) = (x₁^2 + x₂^2) / (x₁ * x₂)

-- Main theorem
theorem slope_is_constant
  (k m x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : ellipse x₁ y₁)
  (h₂ : ellipse x₂ y₂)
  (h₃ : in_first_quadrant x₁ y₁)
  (h₄ : in_first_quadrant x₂ y₂)
  (h₅ : line k m x₁ y₁)
  (h₆ : line k m x₂ y₂)
  (h₇ : m ≠ 0)
  (h₈ : area_condition x₁ y₁ x₂ y₂ m k) :
  k = -1/2 := by sorry

end NUMINAMATH_CALUDE_slope_is_constant_l545_54541


namespace NUMINAMATH_CALUDE_geometric_sequence_17th_term_l545_54561

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_17th_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_5th : a 5 = 9)
  (h_13th : a 13 = 1152) :
  a 17 = 36864 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_17th_term_l545_54561


namespace NUMINAMATH_CALUDE_seokjin_drank_least_l545_54534

/-- Represents the amount of milk drunk by each person in liters -/
structure MilkConsumption where
  jungkook : ℝ
  seokjin : ℝ
  yoongi : ℝ

/-- Given the milk consumption of Jungkook, Seokjin, and Yoongi, 
    proves that Seokjin drank the least amount of milk -/
theorem seokjin_drank_least (m : MilkConsumption) 
  (h1 : m.jungkook = 1.3)
  (h2 : m.seokjin = 11/10)
  (h3 : m.yoongi = 7/5) : 
  m.seokjin < m.jungkook ∧ m.seokjin < m.yoongi := by
  sorry

#check seokjin_drank_least

end NUMINAMATH_CALUDE_seokjin_drank_least_l545_54534


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l545_54558

/-- The repeating decimal 0.868686... -/
def repeating_decimal : ℚ := 0.868686

/-- The fraction 86/99 -/
def fraction : ℚ := 86 / 99

/-- Theorem stating that the repeating decimal 0.868686... equals the fraction 86/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l545_54558


namespace NUMINAMATH_CALUDE_seedling_problem_l545_54588

/-- Represents the unit price and quantity of seedlings --/
structure Seedling where
  price : ℚ
  quantity : ℚ

/-- Represents the total cost of a purchase --/
def totalCost (a b : Seedling) : ℚ :=
  a.price * a.quantity + b.price * b.quantity

/-- Represents the discounted price of a seedling --/
def discountedPrice (s : Seedling) (discount : ℚ) : ℚ :=
  s.price * (1 - discount)

theorem seedling_problem :
  ∃ (a b : Seedling),
    (totalCost ⟨a.price, 15⟩ ⟨b.price, 5⟩ = 190) ∧
    (totalCost ⟨a.price, 25⟩ ⟨b.price, 15⟩ = 370) ∧
    (a.price = 10) ∧
    (b.price = 8) ∧
    (∀ m : ℚ,
      m ≤ 100 ∧
      (discountedPrice a 0.1) * m + (discountedPrice b 0.1) * (100 - m) ≤ 828 →
      m ≤ 60) ∧
    (∃ m : ℚ,
      m = 60 ∧
      (discountedPrice a 0.1) * m + (discountedPrice b 0.1) * (100 - m) ≤ 828) :=
by
  sorry


end NUMINAMATH_CALUDE_seedling_problem_l545_54588


namespace NUMINAMATH_CALUDE_total_distance_theorem_l545_54564

/-- Calculates the total distance covered by two cyclists in a week -/
def total_distance_in_week (
  onur_speed : ℝ
  ) (hanil_speed : ℝ
  ) (onur_hours : ℝ
  ) (onur_rest_day : ℕ
  ) (hanil_rest_day : ℕ
  ) (hanil_extra_distance : ℝ
  ) (days_in_week : ℕ
  ) : ℝ :=
  let onur_daily_distance := onur_speed * onur_hours
  let hanil_daily_distance := onur_daily_distance + hanil_extra_distance
  let onur_biking_days := days_in_week - (days_in_week / onur_rest_day)
  let hanil_biking_days := days_in_week - (days_in_week / hanil_rest_day)
  let onur_total_distance := onur_daily_distance * onur_biking_days
  let hanil_total_distance := hanil_daily_distance * hanil_biking_days
  onur_total_distance + hanil_total_distance

/-- Theorem stating the total distance covered by Onur and Hanil in a week -/
theorem total_distance_theorem :
  total_distance_in_week 35 45 7 3 4 40 7 = 2935 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_theorem_l545_54564


namespace NUMINAMATH_CALUDE_tangent_line_at_P_l545_54508

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 1

-- Define the point of tangency
def P : ℝ × ℝ := (1, 0)

-- Define the slope of the tangent line at P
def m : ℝ := 3

-- Define the equation of the tangent line
def tangent_line (x : ℝ) : ℝ := m * (x - P.1) + P.2

theorem tangent_line_at_P : 
  ∀ x : ℝ, tangent_line x = 3*x - 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_P_l545_54508


namespace NUMINAMATH_CALUDE_P_properties_l545_54517

/-- P k l n denotes the number of partitions of n into no more than k summands, 
    each not exceeding l -/
def P (k l n : ℕ) : ℕ :=
  sorry

/-- The four properties of P as stated in the problem -/
theorem P_properties (k l n : ℕ) :
  (P k l n - P k (l-1) n = P (k-1) l (n-l)) ∧
  (P k l n - P (k-1) l n = P k (l-1) (n-k)) ∧
  (P k l n = P l k n) ∧
  (P k l n = P k l (k*l - n)) :=
by sorry

end NUMINAMATH_CALUDE_P_properties_l545_54517


namespace NUMINAMATH_CALUDE_student_count_l545_54596

theorem student_count (cost_per_student : ℕ) (total_cost : ℕ) (h1 : cost_per_student = 8) (h2 : total_cost = 184) :
  total_cost / cost_per_student = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_student_count_l545_54596


namespace NUMINAMATH_CALUDE_simplify_expression_l545_54553

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l545_54553


namespace NUMINAMATH_CALUDE_solution_correctness_l545_54512

-- Define the equation
def equation (x : ℝ) : Prop := 2 * (x + 3) = 5 * x

-- Define the solution steps
def step1 (x : ℝ) : Prop := 2 * x + 6 = 5 * x
def step2 (x : ℝ) : Prop := 2 * x - 5 * x = -6
def step3 (x : ℝ) : Prop := -3 * x = -6
def step4 : ℝ := 2

-- Theorem stating the correctness of the solution and that step3 is not based on associative property
theorem solution_correctness :
  ∀ x : ℝ,
  equation x →
  step1 x ∧
  step2 x ∧
  step3 x ∧
  step4 = x ∧
  ¬(∃ a b c : ℝ, step3 x ↔ (a + b) + c = a + (b + c)) :=
by sorry

end NUMINAMATH_CALUDE_solution_correctness_l545_54512


namespace NUMINAMATH_CALUDE_cylinder_volume_from_lateral_surface_l545_54594

/-- Given a cylinder whose lateral surface unfolds to a square with side length 2,
    prove that its volume is 2/π. -/
theorem cylinder_volume_from_lateral_surface (r h : ℝ) : 
  (2 * π * r = 2) → (h = 2) → (π * r^2 * h = 2/π) := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_lateral_surface_l545_54594


namespace NUMINAMATH_CALUDE_willie_had_48_bananas_l545_54556

/-- Given the total number of bananas and Charles' initial bananas, 
    calculate Willie's initial bananas. -/
def willies_bananas (total : ℝ) (charles_initial : ℝ) : ℝ :=
  total - charles_initial

/-- Theorem stating that Willie had 48.0 bananas given the problem conditions. -/
theorem willie_had_48_bananas : 
  willies_bananas 83 35 = 48 := by
  sorry

#eval willies_bananas 83 35

end NUMINAMATH_CALUDE_willie_had_48_bananas_l545_54556


namespace NUMINAMATH_CALUDE_problem_solution_l545_54591

theorem problem_solution : ∃ x : ℝ, (0.65 * x = 0.20 * 747.50) ∧ (x = 230) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l545_54591


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l545_54562

def U : Set ℕ := {1, 2, 3, 4}

def M : Set ℕ := {x | (x - 1) * (x - 4) = 0}

theorem complement_of_M_in_U : U \ M = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l545_54562


namespace NUMINAMATH_CALUDE_triangle_angle_range_l545_54587

theorem triangle_angle_range (a b : ℝ) (h_a : a = 2) (h_b : b = 2 * Real.sqrt 2) :
  ∃ (A : ℝ), 0 < A ∧ A ≤ π / 4 ∧
  ∀ (c : ℝ), c > 0 → a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_range_l545_54587


namespace NUMINAMATH_CALUDE_f_bounded_by_four_l545_54581

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 3|

-- State the theorem
theorem f_bounded_by_four : ∀ x : ℝ, |f x| ≤ 4 := by sorry

end NUMINAMATH_CALUDE_f_bounded_by_four_l545_54581


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_m_range_l545_54529

theorem ellipse_eccentricity_m_range :
  ∀ m : ℝ,
  m > 0 →
  (∃ e : ℝ, 1/2 < e ∧ e < 1 ∧
    (∀ x y : ℝ, x^2 + m*y^2 = 1 →
      e = Real.sqrt (1 - min m (1/m)))) →
  (m ∈ Set.Ioo 0 (3/4) ∪ Set.Ioi (4/3)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_m_range_l545_54529


namespace NUMINAMATH_CALUDE_min_value_sum_of_roots_l545_54566

theorem min_value_sum_of_roots (x : ℝ) :
  let y := Real.sqrt (x^2 - 2*x + 2) + Real.sqrt (x^2 - 10*x + 34)
  y ≥ 4 * Real.sqrt 2 ∧ ∃ x₀ : ℝ, Real.sqrt (x₀^2 - 2*x₀ + 2) + Real.sqrt (x₀^2 - 10*x₀ + 34) = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_roots_l545_54566


namespace NUMINAMATH_CALUDE_expression_value_l545_54546

theorem expression_value (x y z : ℚ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  (3 * x - 4 * y) / z = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l545_54546


namespace NUMINAMATH_CALUDE_min_correct_answers_for_score_l545_54567

/-- Represents the scoring system and conditions of the math competition --/
structure MathCompetition where
  total_questions : Nat
  attempted_questions : Nat
  correct_points : Nat
  incorrect_deduction : Nat
  unanswered_points : Nat
  min_required_score : Nat

/-- Calculates the score based on the number of correct answers --/
def calculate_score (comp : MathCompetition) (correct_answers : Nat) : Int :=
  let incorrect_answers := comp.attempted_questions - correct_answers
  let unanswered := comp.total_questions - comp.attempted_questions
  (correct_answers * comp.correct_points : Int) -
  (incorrect_answers * comp.incorrect_deduction) +
  (unanswered * comp.unanswered_points)

/-- Theorem stating the minimum number of correct answers needed to achieve the required score --/
theorem min_correct_answers_for_score (comp : MathCompetition)
  (h1 : comp.total_questions = 25)
  (h2 : comp.attempted_questions = 20)
  (h3 : comp.correct_points = 8)
  (h4 : comp.incorrect_deduction = 2)
  (h5 : comp.unanswered_points = 2)
  (h6 : comp.min_required_score = 150) :
  ∃ n : Nat, (∀ m : Nat, calculate_score comp m ≥ comp.min_required_score → m ≥ n) ∧
             calculate_score comp n ≥ comp.min_required_score ∧
             n = 18 := by
  sorry


end NUMINAMATH_CALUDE_min_correct_answers_for_score_l545_54567


namespace NUMINAMATH_CALUDE_investment_comparison_l545_54568

/-- Represents the value of an investment over time -/
structure Investment where
  initial : ℝ
  year1_change : ℝ
  year2_change : ℝ

/-- Calculates the final value of an investment after two years -/
def final_value (inv : Investment) : ℝ :=
  inv.initial * (1 + inv.year1_change) * (1 + inv.year2_change)

/-- The problem setup -/
def problem_setup : (Investment × Investment × Investment) :=
  ({ initial := 150, year1_change := 0.1, year2_change := 0.15 },
   { initial := 150, year1_change := -0.3, year2_change := 0.5 },
   { initial := 150, year1_change := 0, year2_change := -0.1 })

theorem investment_comparison :
  let (a, b, c) := problem_setup
  final_value a > final_value b ∧ final_value b > final_value c :=
by sorry

end NUMINAMATH_CALUDE_investment_comparison_l545_54568


namespace NUMINAMATH_CALUDE_max_n_value_l545_54522

theorem max_n_value (a b c : ℝ) (n : ℕ) (h1 : a > b) (h2 : b > c)
  (h3 : ∀ (a b c : ℝ), a > b → b > c → (a - b)⁻¹ + (b - c)⁻¹ ≥ n^2 * (a - c)⁻¹) :
  n ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_n_value_l545_54522


namespace NUMINAMATH_CALUDE_complex_magnitude_eval_l545_54595

theorem complex_magnitude_eval (ω : ℂ) (h : ω = 7 + 3 * I) :
  Complex.abs (ω^2 + 8*ω + 85) = Real.sqrt 30277 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_eval_l545_54595


namespace NUMINAMATH_CALUDE_ball_reaches_top_left_corner_l545_54589

/-- Represents a rectangular billiard table -/
structure BilliardTable where
  width : ℕ
  length : ℕ

/-- Represents the path of a ball on the billiard table -/
def ball_path (table : BilliardTable) : ℕ :=
  Nat.lcm table.width table.length

/-- Theorem: A ball launched at 45° from the bottom-left corner of a 26x1965 table
    will reach the top-left corner after traveling the LCM of 26 and 1965 in both directions -/
theorem ball_reaches_top_left_corner (table : BilliardTable) 
    (h1 : table.width = 26) (h2 : table.length = 1965) :
    ball_path table = 50990 ∧ 
    50990 % table.width = 0 ∧ 
    50990 % table.length = 0 := by
  sorry

#eval ball_path { width := 26, length := 1965 }

end NUMINAMATH_CALUDE_ball_reaches_top_left_corner_l545_54589


namespace NUMINAMATH_CALUDE_andrew_flooring_theorem_l545_54571

def andrew_flooring_problem (bedroom living_room kitchen guest_bedroom hallway leftover : ℕ) : Prop :=
  let total_used := bedroom + living_room + kitchen + guest_bedroom + 2 * hallway
  let total_original := total_used + leftover
  let ruined_per_bedroom := total_original - total_used
  (bedroom = 8) ∧
  (living_room = 20) ∧
  (kitchen = 11) ∧
  (guest_bedroom = bedroom - 2) ∧
  (hallway = 4) ∧
  (leftover = 6) ∧
  (ruined_per_bedroom = 6)

theorem andrew_flooring_theorem :
  ∀ bedroom living_room kitchen guest_bedroom hallway leftover,
  andrew_flooring_problem bedroom living_room kitchen guest_bedroom hallway leftover :=
by
  sorry

end NUMINAMATH_CALUDE_andrew_flooring_theorem_l545_54571


namespace NUMINAMATH_CALUDE_farmer_purchase_problem_l545_54559

theorem farmer_purchase_problem :
  ∃ (p ch : ℕ), 
    p > 0 ∧ 
    ch > 0 ∧ 
    30 * p + 24 * ch = 1200 ∧ 
    p = 4 ∧ 
    ch = 45 := by
  sorry

end NUMINAMATH_CALUDE_farmer_purchase_problem_l545_54559


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l545_54528

theorem quadratic_no_real_roots : 
  {x : ℝ | x^2 + x + 1 = 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l545_54528


namespace NUMINAMATH_CALUDE_jean_stuffies_fraction_l545_54547

theorem jean_stuffies_fraction (total : ℕ) (kept_fraction : ℚ) (janet_received : ℕ) :
  total = 60 →
  kept_fraction = 1/3 →
  janet_received = 10 →
  (janet_received : ℚ) / (total - total * kept_fraction) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_jean_stuffies_fraction_l545_54547


namespace NUMINAMATH_CALUDE_parallelogram_area_l545_54513

def v : Fin 2 → ℝ := ![7, -4]
def w : Fin 2 → ℝ := ![13, -3]

theorem parallelogram_area : 
  abs (Matrix.det !![v 0, v 1; w 0, w 1]) = 31 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l545_54513


namespace NUMINAMATH_CALUDE_beka_jackson_flight_difference_l545_54544

/-- 
Given that Beka flew 873 miles and Jackson flew 563 miles,
prove that Beka flew 310 miles more than Jackson.
-/
theorem beka_jackson_flight_difference :
  let beka_miles : ℕ := 873
  let jackson_miles : ℕ := 563
  beka_miles - jackson_miles = 310 := by sorry

end NUMINAMATH_CALUDE_beka_jackson_flight_difference_l545_54544


namespace NUMINAMATH_CALUDE_probability_five_blue_marbles_l545_54540

def total_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 4
def total_draws : ℕ := 8
def blue_draws : ℕ := 5

def probability_blue : ℚ := blue_marbles / total_marbles
def probability_red : ℚ := red_marbles / total_marbles

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem probability_five_blue_marbles :
  (binomial_coefficient total_draws blue_draws : ℚ) * 
  (probability_blue ^ blue_draws) * 
  (probability_red ^ (total_draws - blue_draws)) = 1792 / 6561 := by
sorry

end NUMINAMATH_CALUDE_probability_five_blue_marbles_l545_54540


namespace NUMINAMATH_CALUDE_sons_age_l545_54542

/-- Proves that given the conditions, the son's age is 26 years -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 28 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 26 := by
sorry


end NUMINAMATH_CALUDE_sons_age_l545_54542


namespace NUMINAMATH_CALUDE_sequence_divisibility_l545_54582

def u : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 2 * u (n + 1) - 3 * u n

def v (a b c : ℤ) : ℕ → ℤ
  | 0 => a
  | 1 => b
  | 2 => c
  | (n + 3) => v a b c (n + 2) - 3 * v a b c (n + 1) + 27 * v a b c n

theorem sequence_divisibility (a b c : ℤ) :
  (∃ N : ℕ, ∀ n > N, ∃ k : ℤ, v a b c n = k * u n) →
  3 * a = 2 * b + c := by
  sorry

end NUMINAMATH_CALUDE_sequence_divisibility_l545_54582


namespace NUMINAMATH_CALUDE_seven_digit_number_exists_l545_54549

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem seven_digit_number_exists : ∃ n : ℕ, 
  (1000000 ≤ n ∧ n < 9000000) ∧ 
  (sum_of_digits n = 53) ∧ 
  (n % 13 = 0) ∧ 
  (n = 8999990) :=
sorry

end NUMINAMATH_CALUDE_seven_digit_number_exists_l545_54549


namespace NUMINAMATH_CALUDE_oranges_picked_total_l545_54590

theorem oranges_picked_total (mary_oranges jason_oranges : ℕ) 
  (h1 : mary_oranges = 122) 
  (h2 : jason_oranges = 105) : 
  mary_oranges + jason_oranges = 227 := by
  sorry

end NUMINAMATH_CALUDE_oranges_picked_total_l545_54590
