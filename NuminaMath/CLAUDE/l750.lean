import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_ratio_l750_75071

theorem system_solution_ratio (x y a b : ℝ) (h1 : 4*x - 2*y = a) (h2 : 6*y - 12*x = b) (h3 : b ≠ 0) :
  a / b = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l750_75071


namespace NUMINAMATH_CALUDE_max_a_value_l750_75049

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

-- State the theorem
theorem max_a_value (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 2, f a x ≤ 6) →
  a ≤ -1 ∧ ∃ x ∈ Set.Ioo 0 2, f (-1) x = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l750_75049


namespace NUMINAMATH_CALUDE_intersection_A_B_when_m_3_range_of_m_when_A_subset_B_l750_75074

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - 2*x - x^2)}

-- Define set B
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 ≤ 0}

-- Part 1
theorem intersection_A_B_when_m_3 : 
  A ∩ B 3 = {x | -2 ≤ x ∧ x ≤ 1} := by sorry

-- Part 2
theorem range_of_m_when_A_subset_B : 
  ∀ m > 0, A ⊆ B m → m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_m_3_range_of_m_when_A_subset_B_l750_75074


namespace NUMINAMATH_CALUDE_least_number_divisibility_l750_75033

theorem least_number_divisibility (x : ℕ) : 
  (x > 0) →
  (x / 5 = (x % 34) + 8) →
  (∀ y : ℕ, y > 0 → y / 5 = (y % 34) + 8 → y ≥ x) →
  x = 160 := by
sorry

end NUMINAMATH_CALUDE_least_number_divisibility_l750_75033


namespace NUMINAMATH_CALUDE_original_cyclists_l750_75034

theorem original_cyclists (total_bill : ℕ) (left_cyclists : ℕ) (extra_payment : ℕ) :
  total_bill = 80 ∧ left_cyclists = 2 ∧ extra_payment = 2 →
  ∃ x : ℕ, x > 0 ∧ (total_bill / (x - left_cyclists) = total_bill / x + extra_payment) ∧ x = 10 :=
by
  sorry

#check original_cyclists

end NUMINAMATH_CALUDE_original_cyclists_l750_75034


namespace NUMINAMATH_CALUDE_least_x_for_even_prime_quotient_l750_75075

theorem least_x_for_even_prime_quotient :
  ∃ (x p q : ℕ),
    x > 0 ∧
    Prime p ∧
    Prime q ∧
    p ≠ q ∧
    q - p = 3 ∧
    x / (11 * p * q) = 2 ∧
    (∀ y, y > 0 → y / (11 * p * q) = 2 → y ≥ x) ∧
    x = 770 :=
by sorry

end NUMINAMATH_CALUDE_least_x_for_even_prime_quotient_l750_75075


namespace NUMINAMATH_CALUDE_inequalities_not_necessarily_true_l750_75057

theorem inequalities_not_necessarily_true
  (x y a b : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hxa : x < a)
  (hyb : y ≠ b) :
  ∃ (x' y' a' b' : ℝ),
    x' ≠ 0 ∧ y' ≠ 0 ∧ a' ≠ 0 ∧ b' ≠ 0 ∧
    x' < a' ∧ y' ≠ b' ∧
    ¬(x' + y' < a' + b') ∧
    ¬(x' - y' < a' - b') ∧
    ¬(x' * y' < a' * b') ∧
    ¬(x' / y' < a' / b') :=
by sorry

end NUMINAMATH_CALUDE_inequalities_not_necessarily_true_l750_75057


namespace NUMINAMATH_CALUDE_joes_lifts_l750_75098

theorem joes_lifts (first_lift second_lift total_weight : ℕ) : 
  first_lift = 400 →
  2 * first_lift = second_lift + 300 →
  total_weight = first_lift + second_lift →
  total_weight = 900 := by
sorry

end NUMINAMATH_CALUDE_joes_lifts_l750_75098


namespace NUMINAMATH_CALUDE_largest_four_digit_negative_congruent_to_one_mod_seventeen_l750_75051

theorem largest_four_digit_negative_congruent_to_one_mod_seventeen :
  ∃ (n : ℤ), n = -1002 ∧ 
  n ≡ 1 [ZMOD 17] ∧
  n < 0 ∧ 
  -9999 ≤ n ∧ 
  n ≥ -999 ∧
  ∀ (m : ℤ), m ≡ 1 [ZMOD 17] → m < 0 → -9999 ≤ m → m ≥ -999 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_negative_congruent_to_one_mod_seventeen_l750_75051


namespace NUMINAMATH_CALUDE_crayon_selection_theorem_l750_75096

/-- The number of different-colored crayons in the box -/
def total_crayons : ℕ := 18

/-- The number of crayons Karl needs to choose -/
def crayons_to_choose : ℕ := 6

/-- The number of crayons that are either red or blue -/
def red_or_blue_crayons : ℕ := 2

/-- The number of ways to select crayons with at least one red or blue -/
def ways_to_select : ℕ := Nat.choose total_crayons crayons_to_choose - 
                           Nat.choose (total_crayons - red_or_blue_crayons) crayons_to_choose

theorem crayon_selection_theorem : ways_to_select = 10556 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_theorem_l750_75096


namespace NUMINAMATH_CALUDE_divisibility_implication_l750_75052

theorem divisibility_implication (x y : ℤ) (h : 17 ∣ (2 * x + 3 * y)) : 17 ∣ (9 * x + 5 * y) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l750_75052


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l750_75053

theorem sum_of_a_and_b (a b : ℕ+) : 
  (∀ k : ℕ+, k ≠ 1776 → (∃ m n : ℕ, k = m * a + n * b)) →
  (¬ ∃ m n : ℕ, 1776 = m * a + n * b) →
  (∃! k : ℕ+, ∀ x : ℕ+, x > k → ∃ m n : ℕ, x = m * a + n * b) →
  (Nat.card {x : ℕ+ | ¬∃ m n : ℕ, x = m * a + n * b} = 2009) →
  a + b = 133 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l750_75053


namespace NUMINAMATH_CALUDE_long_jump_competition_l750_75099

/-- The long jump competition problem -/
theorem long_jump_competition (first second third fourth : ℝ) : 
  first = 22 →
  second = first + 1 →
  third = second - 2 →
  fourth = 24 →
  fourth - third = 3 := by
  sorry

end NUMINAMATH_CALUDE_long_jump_competition_l750_75099


namespace NUMINAMATH_CALUDE_aaron_erasers_l750_75020

theorem aaron_erasers (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  initial = 81 → given_away = 34 → remaining = initial - given_away → remaining = 47 := by
  sorry

end NUMINAMATH_CALUDE_aaron_erasers_l750_75020


namespace NUMINAMATH_CALUDE_solve_inequality_one_solve_inequality_two_l750_75069

namespace InequalitySolver

-- Part 1
theorem solve_inequality_one (x : ℝ) :
  (3 * x - 2) / (x - 1) > 1 ↔ x > 1 :=
sorry

-- Part 2
theorem solve_inequality_two (x a : ℝ) :
  (a = 0 → ¬∃x, x^2 - a*x - 2*a^2 < 0) ∧
  (a > 0 → (x^2 - a*x - 2*a^2 < 0 ↔ -a < x ∧ x < 2*a)) ∧
  (a < 0 → (x^2 - a*x - 2*a^2 < 0 ↔ 2*a < x ∧ x < -a)) :=
sorry

end InequalitySolver

end NUMINAMATH_CALUDE_solve_inequality_one_solve_inequality_two_l750_75069


namespace NUMINAMATH_CALUDE_average_letters_per_day_l750_75067

def letters_monday : ℕ := 7
def letters_tuesday : ℕ := 10
def letters_wednesday : ℕ := 3
def letters_thursday : ℕ := 5
def letters_friday : ℕ := 12
def total_days : ℕ := 5

theorem average_letters_per_day :
  (letters_monday + letters_tuesday + letters_wednesday + letters_thursday + letters_friday : ℚ) / total_days = 37 / 5 := by
  sorry

end NUMINAMATH_CALUDE_average_letters_per_day_l750_75067


namespace NUMINAMATH_CALUDE_largest_difference_even_odd_three_digit_l750_75079

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 3 ∧ digits.toFinset.card = 3

def all_even_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 0

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 1

theorem largest_difference_even_odd_three_digit : 
  ∃ (a b : ℕ), 
    is_three_digit_number a ∧
    is_three_digit_number b ∧
    has_distinct_digits a ∧
    has_distinct_digits b ∧
    all_even_digits a ∧
    all_odd_digits b ∧
    (∀ (x y : ℕ), 
      is_three_digit_number x ∧
      is_three_digit_number y ∧
      has_distinct_digits x ∧
      has_distinct_digits y ∧
      all_even_digits x ∧
      all_odd_digits y →
      x - y ≤ a - b) ∧
    a - b = 729 :=
sorry

end NUMINAMATH_CALUDE_largest_difference_even_odd_three_digit_l750_75079


namespace NUMINAMATH_CALUDE_megan_bottles_left_l750_75065

/-- Calculates the number of bottles Megan has left after drinking and giving away some bottles. -/
def bottles_left (initial : ℕ) (drank : ℕ) (given_away : ℕ) : ℕ :=
  initial - (drank + given_away)

/-- Theorem stating that Megan has 25 bottles left after starting with 45, drinking 8, and giving away 12. -/
theorem megan_bottles_left : bottles_left 45 8 12 = 25 := by
  sorry

end NUMINAMATH_CALUDE_megan_bottles_left_l750_75065


namespace NUMINAMATH_CALUDE_jerry_added_six_figures_l750_75061

/-- Given that Jerry initially had 4 action figures and ended up with 10 action figures in total,
    prove that he added 6 action figures. -/
theorem jerry_added_six_figures (initial : ℕ) (total : ℕ) (added : ℕ)
    (h1 : initial = 4)
    (h2 : total = 10)
    (h3 : total = initial + added) :
  added = 6 := by
  sorry

end NUMINAMATH_CALUDE_jerry_added_six_figures_l750_75061


namespace NUMINAMATH_CALUDE_working_days_count_l750_75085

/-- Represents a day of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a day in the month -/
structure DayInMonth where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Determines if a given day is a holiday -/
def isHoliday (d : DayInMonth) : Bool :=
  match d.dayOfWeek with
  | DayOfWeek.Sunday => true
  | DayOfWeek.Saturday => d.day % 14 == 8  -- Every second Saturday
  | _ => false

/-- Theorem: In a 30-day month starting on a Saturday, with every second Saturday 
    and all Sundays as holidays, there are 23 working days -/
theorem working_days_count : 
  let month : List DayInMonth := sorry  -- List of 30 days starting from Saturday
  (month.length = 30) →
  (month.head?.map (fun d => d.dayOfWeek) = some DayOfWeek.Saturday) →
  (month.filter (fun d => ¬isHoliday d)).length = 23 :=
by sorry

end NUMINAMATH_CALUDE_working_days_count_l750_75085


namespace NUMINAMATH_CALUDE_choir_size_choir_size_is_30_l750_75047

/-- The number of singers in a school choir, given the initial number of robes,
    the cost per new robe, and the total amount spent on new robes. -/
theorem choir_size (initial_robes : ℕ) (cost_per_robe : ℕ) (total_spent : ℕ) : ℕ :=
  initial_robes + total_spent / cost_per_robe

/-- Proof that the number of singers in the choir is 30. -/
theorem choir_size_is_30 :
  choir_size 12 2 36 = 30 := by
  sorry

end NUMINAMATH_CALUDE_choir_size_choir_size_is_30_l750_75047


namespace NUMINAMATH_CALUDE_triangle_max_area_l750_75093

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where a = 2 and 3b*sin(C) - 5c*sin(B)*cos(A) = 0, 
    the maximum area of the triangle is 10/3. -/
theorem triangle_max_area (a b c A B C : ℝ) : 
  a = 2 → 
  3 * b * Real.sin C - 5 * c * Real.sin B * Real.cos A = 0 → 
  (∃ (S : ℝ), S = (1/2) * a * b * Real.sin C ∧ 
    ∀ (S' : ℝ), S' = (1/2) * a * b * Real.sin C → S' ≤ S) →
  (1/2) * a * b * Real.sin C ≤ 10/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l750_75093


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l750_75041

theorem quadratic_inequality_solution_condition (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ |a| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l750_75041


namespace NUMINAMATH_CALUDE_hex_BF02_eq_48898_l750_75013

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'B' => 11
  | 'F' => 15
  | '0' => 0
  | '2' => 2
  | _ => 0  -- Default case, should not be reached for this problem

/-- Converts a hexadecimal number represented as a string to its decimal value -/
def hex_to_decimal (s : String) : ℕ :=
  s.foldr (fun c acc => hex_to_dec c + 16 * acc) 0

/-- The hexadecimal number BF02 is equal to 48898 in decimal -/
theorem hex_BF02_eq_48898 : hex_to_decimal "BF02" = 48898 := by
  sorry

end NUMINAMATH_CALUDE_hex_BF02_eq_48898_l750_75013


namespace NUMINAMATH_CALUDE_white_towels_count_l750_75001

theorem white_towels_count (green_towels : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  green_towels = 35 → given_away = 34 → remaining = 22 → 
  ∃ white_towels : ℕ, white_towels = 21 ∧ green_towels + white_towels - given_away = remaining :=
by
  sorry

end NUMINAMATH_CALUDE_white_towels_count_l750_75001


namespace NUMINAMATH_CALUDE_root_reciprocal_relation_l750_75029

theorem root_reciprocal_relation (p m q n : ℝ) : 
  (∃ x : ℝ, x^2 + p*x + q = 0 ∧ (1/x)^2 + m*(1/x) + n = 0) → 
  (p*n - m)*(q*m - p) = (q*n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_root_reciprocal_relation_l750_75029


namespace NUMINAMATH_CALUDE_point_transformation_l750_75017

-- Define the point type
def Point := ℝ × ℝ × ℝ

-- Define the transformations
def reflect_yz (p : Point) : Point :=
  let (x, y, z) := p
  (-x, y, z)

def rotate_z_90 (p : Point) : Point :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xy (p : Point) : Point :=
  let (x, y, z) := p
  (x, y, -z)

def rotate_x_180 (p : Point) : Point :=
  let (x, y, z) := p
  (x, -y, -z)

def reflect_xz (p : Point) : Point :=
  let (x, y, z) := p
  (x, -y, z)

-- Define the composition of all transformations
def transform (p : Point) : Point :=
  p |> reflect_yz
    |> rotate_z_90
    |> reflect_xy
    |> rotate_x_180
    |> reflect_xz
    |> rotate_z_90

-- Theorem statement
theorem point_transformation :
  transform (2, 2, 2) = (2, 2, -2) := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l750_75017


namespace NUMINAMATH_CALUDE_rational_equation_power_l750_75039

theorem rational_equation_power (x y : ℚ) 
  (h : |x + 5| + (y - 5)^2 = 0) : (x / y)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_power_l750_75039


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_a_equals_one_l750_75035

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_pure_imaginary_a_equals_one (a : ℝ) :
  is_pure_imaginary ((1 + a * Complex.I) / (1 - Complex.I)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_a_equals_one_l750_75035


namespace NUMINAMATH_CALUDE_parabola_hyperbola_equations_l750_75048

/-- Represents a parabola with vertex at the origin -/
structure Parabola where
  c : ℝ
  eq : ℝ × ℝ → Prop := fun (x, y) ↦ y^2 = 4 * c * x

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : ℝ × ℝ → Prop := fun (x, y) ↦ x^2 / a^2 - y^2 / b^2 = 1

/-- The main theorem -/
theorem parabola_hyperbola_equations 
  (p : Parabola) 
  (h : Hyperbola) 
  (h_a_pos : h.a > 0)
  (h_b_pos : h.b > 0)
  (directrix_passes_focus : ∃ (f : ℝ × ℝ), h.eq f ∧ p.c = 2 * h.a)
  (intersection_point : p.eq (3/2, Real.sqrt 6) ∧ h.eq (3/2, Real.sqrt 6)) :
  p.eq = fun (x, y) ↦ y^2 = 4 * x ∧ 
  h.eq = fun (x, y) ↦ 4 * x^2 - 4 * y^2 / 3 = 1 := by
  sorry


end NUMINAMATH_CALUDE_parabola_hyperbola_equations_l750_75048


namespace NUMINAMATH_CALUDE_boxes_needed_proof_l750_75088

/-- The number of chocolate bars Tom needs to sell -/
def total_bars : ℕ := 849

/-- The number of chocolate bars in each box -/
def bars_per_box : ℕ := 5

/-- The minimum number of boxes needed to contain all the bars -/
def min_boxes_needed : ℕ := (total_bars + bars_per_box - 1) / bars_per_box

theorem boxes_needed_proof : min_boxes_needed = 170 := by
  sorry

end NUMINAMATH_CALUDE_boxes_needed_proof_l750_75088


namespace NUMINAMATH_CALUDE_store_constraints_equivalence_l750_75040

/-- Represents the constraints on product purchases in a store. -/
def StoreConstraints (x : ℝ) : Prop :=
  let productACost : ℝ := 8
  let productBCost : ℝ := 2
  let productBQuantity : ℝ := 2 * x - 4
  let totalItems : ℝ := x + productBQuantity
  let totalCost : ℝ := productACost * x + productBCost * productBQuantity
  (totalItems ≥ 32) ∧ (totalCost ≤ 148)

/-- Theorem stating that the given system of inequalities correctly represents the store constraints. -/
theorem store_constraints_equivalence (x : ℝ) :
  StoreConstraints x ↔ (x + (2 * x - 4) ≥ 32 ∧ 8 * x + 2 * (2 * x - 4) ≤ 148) :=
by sorry

end NUMINAMATH_CALUDE_store_constraints_equivalence_l750_75040


namespace NUMINAMATH_CALUDE_unique_function_property_l750_75054

theorem unique_function_property (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 1) = f x + 1)
  (h2 : ∀ x, f (x^2) = (f x)^2) :
  ∀ x, f x = x :=
by sorry

end NUMINAMATH_CALUDE_unique_function_property_l750_75054


namespace NUMINAMATH_CALUDE_dollar_operation_theorem_l750_75062

/-- Define the dollar operation -/
def dollar (k : ℝ) (a b : ℝ) : ℝ := k * (a - b)^2

/-- Theorem stating that (2x - 3y)² $₃ (3y - 2x)² = 0 for any real x and y -/
theorem dollar_operation_theorem (x y : ℝ) : 
  dollar 3 ((2*x - 3*y)^2) ((3*y - 2*x)^2) = 0 := by
  sorry

#check dollar_operation_theorem

end NUMINAMATH_CALUDE_dollar_operation_theorem_l750_75062


namespace NUMINAMATH_CALUDE_triangle_sine_problem_l750_75046

theorem triangle_sine_problem (D E F : ℝ) (h_area : (1/2) * D * E * Real.sin F = 72) 
  (h_geometric_mean : Real.sqrt (D * E) = 15) : Real.sin F = 16/25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_problem_l750_75046


namespace NUMINAMATH_CALUDE_mikes_typing_speed_l750_75055

/-- Mike's original typing speed in words per minute -/
def original_speed : ℕ := 65

/-- Mike's reduced typing speed in words per minute -/
def reduced_speed : ℕ := original_speed - 20

/-- Number of words in the document -/
def document_words : ℕ := 810

/-- Time taken to type the document at reduced speed, in minutes -/
def typing_time : ℕ := 18

theorem mikes_typing_speed :
  (reduced_speed * typing_time = document_words) ∧
  (original_speed = 65) := by
  sorry

end NUMINAMATH_CALUDE_mikes_typing_speed_l750_75055


namespace NUMINAMATH_CALUDE_constant_term_in_expansion_l750_75004

theorem constant_term_in_expansion (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (x + a / x) * (2 * x - 1 / x)^5 = 2) →
  ∃ coeffs : List ℝ, 
    (∀ x : ℝ, x ≠ 0 → (x + a / x) * (2 * x - 1 / x)^5 = coeffs.sum) ∧
    coeffs.sum = 2 ∧
    (∃ const_term : ℝ, const_term = 40 ∧ 
      ∀ x : ℝ, x ≠ 0 → (x + a / x) * (2 * x - 1 / x)^5 = 
        const_term + x * (coeffs.sum - const_term - a / x) + 
        1 / x * (coeffs.sum - const_term - a * x)) :=
by
  sorry

end NUMINAMATH_CALUDE_constant_term_in_expansion_l750_75004


namespace NUMINAMATH_CALUDE_intersection_A_B_l750_75028

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

def B : Set ℝ := {x | |x - 2| < 2}

theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l750_75028


namespace NUMINAMATH_CALUDE_uncoverable_3x7_and_7x3_other_boards_coverable_l750_75025

/-- A board configuration -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)
  (removed : ℕ)

/-- Checks if a board can be completely covered by dominoes -/
def can_cover (b : Board) : Prop :=
  (b.rows * b.cols - b.removed) % 2 = 0

/-- Theorem: A 3x7 or 7x3 board cannot be completely covered by dominoes -/
theorem uncoverable_3x7_and_7x3 :
  ¬(can_cover ⟨3, 7, 0⟩) ∧ ¬(can_cover ⟨7, 3, 0⟩) :=
sorry

/-- Theorem: All other given board configurations can be covered by dominoes -/
theorem other_boards_coverable :
  can_cover ⟨2, 3, 0⟩ ∧
  can_cover ⟨4, 4, 4⟩ ∧
  can_cover ⟨5, 5, 1⟩ :=
sorry

end NUMINAMATH_CALUDE_uncoverable_3x7_and_7x3_other_boards_coverable_l750_75025


namespace NUMINAMATH_CALUDE_inlet_pipe_fill_rate_l750_75000

/-- Given a tank with specified properties, prove the inlet pipe's fill rate --/
theorem inlet_pipe_fill_rate 
  (tank_capacity : ℝ) 
  (leak_empty_time : ℝ) 
  (combined_empty_time : ℝ) 
  (h1 : tank_capacity = 4320)
  (h2 : leak_empty_time = 6)
  (h3 : combined_empty_time = 8) :
  let leak_rate := tank_capacity / leak_empty_time
  let net_empty_rate := tank_capacity / combined_empty_time
  let inlet_rate := net_empty_rate + leak_rate
  inlet_rate / 60 = 21 := by sorry

end NUMINAMATH_CALUDE_inlet_pipe_fill_rate_l750_75000


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_220_l750_75032

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_is_220 :
  rectangle_area 3025 10 = 220 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_220_l750_75032


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l750_75087

theorem largest_constant_inequality (C : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 3 ≥ C*(x + y + z)) ↔ C ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l750_75087


namespace NUMINAMATH_CALUDE_percentage_problem_l750_75077

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 1280 = (20 / 100) * 650 + 190 → P = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l750_75077


namespace NUMINAMATH_CALUDE_congruent_rectangle_perimeter_l750_75003

/-- Given a rectangle of dimensions a × b units divided into a smaller rectangle of dimensions p × q units
    and four congruent rectangles, the perimeter of one of the four congruent rectangles is 2(a + b - p - q) units. -/
theorem congruent_rectangle_perimeter
  (a b p q : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : p > 0)
  (h4 : q > 0)
  (h5 : p < a)
  (h6 : q < b) :
  ∃ (l1 l2 : ℝ), l1 = b - q ∧ l2 = a - p ∧ 2 * (l1 + l2) = 2 * (a + b - p - q) :=
sorry

end NUMINAMATH_CALUDE_congruent_rectangle_perimeter_l750_75003


namespace NUMINAMATH_CALUDE_mets_to_red_sox_ratio_l750_75006

/-- Proves that the ratio of NY Mets fans to Boston Red Sox fans is 4:5 given the conditions -/
theorem mets_to_red_sox_ratio :
  ∀ (yankees mets red_sox : ℕ),
  yankees + mets + red_sox = 360 →
  3 * mets = 2 * yankees →
  mets = 96 →
  5 * mets = 4 * red_sox :=
by
  sorry

end NUMINAMATH_CALUDE_mets_to_red_sox_ratio_l750_75006


namespace NUMINAMATH_CALUDE_min_comparisons_correct_l750_75045

/-- Represents a deck of cards numbered from 1 to n -/
def Deck (n : ℕ) := Fin n

/-- Checks if two numbers are consecutive -/
def are_consecutive (a b : ℕ) : Prop := (a + 1 = b) ∨ (b + 1 = a)

/-- The minimum number of comparisons needed to guarantee finding a consecutive pair -/
def min_comparisons (n : ℕ) := n - 2

theorem min_comparisons_correct (n : ℕ) (h : n ≥ 100) :
  ∀ (d : Deck n), 
    ∃ (f : Fin (min_comparisons n) → Deck n × Deck n),
      ∀ (g : Deck n × Deck n → Bool),
        (∀ (i j : Deck n), g (i, j) = true ↔ are_consecutive i.val j.val) →
        ∃ (i : Fin (min_comparisons n)), g (f i) = true :=
sorry

#check min_comparisons_correct

end NUMINAMATH_CALUDE_min_comparisons_correct_l750_75045


namespace NUMINAMATH_CALUDE_average_sale_proof_l750_75072

def sales_first_five : List Int := [2500, 6500, 9855, 7230, 7000]
def sales_sixth : Int := 11915
def num_months : Int := 6

theorem average_sale_proof :
  (sales_first_five.sum + sales_sixth) / num_months = 7500 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_proof_l750_75072


namespace NUMINAMATH_CALUDE_school_pens_problem_l750_75008

theorem school_pens_problem (pencils : ℕ) (pen_cost pencil_cost total_cost : ℚ) :
  pencils = 38 →
  pencil_cost = 5/2 →
  pen_cost = 7/2 →
  total_cost = 291 →
  ∃ (pens : ℕ), pens * pen_cost + pencils * pencil_cost = total_cost ∧ pens = 56 := by
  sorry

end NUMINAMATH_CALUDE_school_pens_problem_l750_75008


namespace NUMINAMATH_CALUDE_intersection_point_l750_75050

-- Define the rectangle ABCD
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (6, 4)
def D : ℝ × ℝ := (6, 0)

-- Define the lines from A and B
def lineA1 (x : ℝ) : ℝ := x -- y = x (45° line from A)
def lineA2 (x : ℝ) : ℝ := -x -- y = -x (135° line from A)
def lineB1 (x : ℝ) : ℝ := 4 - x -- y = 4 - x (-45° line from B)
def lineB2 (x : ℝ) : ℝ := 4 + x -- y = 4 + x (-135° line from B)

-- Theorem statement
theorem intersection_point : 
  ∃! p : ℝ × ℝ, 
    (lineA1 p.1 = p.2 ∧ lineB1 p.1 = p.2) ∧ 
    (lineA2 p.1 = p.2 ∧ lineB2 p.1 = p.2) ∧
    p = (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l750_75050


namespace NUMINAMATH_CALUDE_danny_bottle_caps_count_l750_75078

/-- The number of bottle caps Danny has in his collection now -/
def danny_bottle_caps : ℕ := 56

/-- The number of wrappers Danny found at the park -/
def wrappers_found : ℕ := 46

/-- The number of bottle caps Danny found at the park -/
def bottle_caps_found : ℕ := 50

/-- The number of wrappers Danny has in his collection now -/
def wrappers_in_collection : ℕ := 52

theorem danny_bottle_caps_count :
  danny_bottle_caps = wrappers_in_collection + (bottle_caps_found - wrappers_found) :=
by sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_count_l750_75078


namespace NUMINAMATH_CALUDE_x_plus_y_equals_22_l750_75064

theorem x_plus_y_equals_22 (x y : ℝ) 
  (h1 : (4 : ℝ) ^ x = 16 ^ (y + 2))
  (h2 : (25 : ℝ) ^ y = 5 ^ (x - 16)) : 
  x + y = 22 := by sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_22_l750_75064


namespace NUMINAMATH_CALUDE_hives_needed_for_candles_twenty_four_hives_for_ninety_six_candles_l750_75082

/-- Given that 3 beehives make enough wax for 12 candles, 
    prove that 24 hives are needed to make 96 candles. -/
theorem hives_needed_for_candles : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun hives_given candles_given hives_needed candles_needed =>
    (hives_given * (candles_needed / candles_given) = hives_needed) →
    (3 * (96 / 12) = 24)

/-- The main theorem stating that 24 hives are needed for 96 candles. -/
theorem twenty_four_hives_for_ninety_six_candles :
  hives_needed_for_candles 3 12 24 96 := by
  sorry

end NUMINAMATH_CALUDE_hives_needed_for_candles_twenty_four_hives_for_ninety_six_candles_l750_75082


namespace NUMINAMATH_CALUDE_mary_earnings_theorem_l750_75027

/-- Calculates the maximum weekly earnings for Mary --/
def mary_max_earnings : ℝ :=
  let max_hours : ℝ := 60
  let regular_hours : ℝ := 20
  let regular_rate : ℝ := 8
  let overtime_rate : ℝ := regular_rate * 1.25
  let overtime_hours : ℝ := max_hours - regular_hours
  (regular_hours * regular_rate) + (overtime_hours * overtime_rate)

/-- Theorem stating Mary's maximum weekly earnings --/
theorem mary_earnings_theorem : mary_max_earnings = 560 := by
  sorry

end NUMINAMATH_CALUDE_mary_earnings_theorem_l750_75027


namespace NUMINAMATH_CALUDE_simplify_cube_root_l750_75097

theorem simplify_cube_root (a b c : ℝ) : ∃ x y z w : ℝ, 
  (54 * a^5 * b^9 * c^14)^(1/3) = x * a^y * b^z * c^w ∧ y + z + w = 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_root_l750_75097


namespace NUMINAMATH_CALUDE_egg_grouping_l750_75063

theorem egg_grouping (total_eggs : ℕ) (group_size : ℕ) (h1 : total_eggs = 9) (h2 : group_size = 3) :
  total_eggs / group_size = 3 := by
  sorry

end NUMINAMATH_CALUDE_egg_grouping_l750_75063


namespace NUMINAMATH_CALUDE_nina_unanswered_questions_l750_75005

/-- Represents the scoring details for a math test -/
structure ScoringSystem where
  initialPoints : ℕ
  correctPoints : ℕ
  wrongPoints : ℤ
  unansweredPoints : ℕ

/-- Represents the test results -/
structure TestResult where
  totalQuestions : ℕ
  score : ℕ

theorem nina_unanswered_questions
  (oldSystem : ScoringSystem)
  (newSystem : ScoringSystem)
  (oldResult : TestResult)
  (newResult : TestResult)
  (h1 : oldSystem = {
    initialPoints := 40,
    correctPoints := 5,
    wrongPoints := -2,
    unansweredPoints := 0
  })
  (h2 : newSystem = {
    initialPoints := 0,
    correctPoints := 6,
    wrongPoints := 0,
    unansweredPoints := 3
  })
  (h3 : oldResult = {totalQuestions := 35, score := 95})
  (h4 : newResult = {totalQuestions := 35, score := 120})
  (h5 : oldResult.totalQuestions = newResult.totalQuestions) :
  ∃ (correct wrong unanswered : ℕ),
    correct + wrong + unanswered = oldResult.totalQuestions ∧
    oldSystem.initialPoints + oldSystem.correctPoints * correct + oldSystem.wrongPoints * wrong = oldResult.score ∧
    newSystem.correctPoints * correct + newSystem.unansweredPoints * unanswered = newResult.score ∧
    unanswered = 10 :=
by sorry

end NUMINAMATH_CALUDE_nina_unanswered_questions_l750_75005


namespace NUMINAMATH_CALUDE_valid_pairs_count_l750_75010

/-- A function that checks if a positive integer has any zero digits -/
def has_zero_digit (n : ℕ+) : Bool :=
  sorry

/-- The set of positive integers less than or equal to 500 without zero digits -/
def valid_numbers : Set ℕ+ :=
  {n : ℕ+ | n ≤ 500 ∧ ¬(has_zero_digit n)}

/-- The number of ordered pairs (a, b) of positive integers where a + b = 500 
    and neither a nor b has a zero digit -/
def count_valid_pairs : ℕ :=
  sorry

theorem valid_pairs_count : count_valid_pairs = 93196 := by
  sorry

end NUMINAMATH_CALUDE_valid_pairs_count_l750_75010


namespace NUMINAMATH_CALUDE_system_solution_l750_75021

theorem system_solution (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + x + y = 80) : 
  x^2 + y^2 = 3980 / 121 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l750_75021


namespace NUMINAMATH_CALUDE_cake_recipe_difference_l750_75024

theorem cake_recipe_difference (total_flour total_sugar flour_added : ℕ) : 
  total_flour = 9 → 
  total_sugar = 11 → 
  flour_added = 4 → 
  total_sugar - (total_flour - flour_added) = 6 :=
by sorry

end NUMINAMATH_CALUDE_cake_recipe_difference_l750_75024


namespace NUMINAMATH_CALUDE_large_fries_cost_l750_75095

-- Define the costs and quantities
def cheeseburger_cost : ℚ := 365/100
def milkshake_cost : ℚ := 2
def coke_cost : ℚ := 1
def cookie_cost : ℚ := 1/2
def cookie_quantity : ℕ := 3
def tax : ℚ := 1/5
def toby_initial : ℚ := 15
def toby_change : ℚ := 7

-- Define the theorem
theorem large_fries_cost (fries_cost : ℚ) : 
  2 * cheeseburger_cost + milkshake_cost + coke_cost + 
  cookie_cost * cookie_quantity + tax + fries_cost = 
  2 * (toby_initial - toby_change) → fries_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_large_fries_cost_l750_75095


namespace NUMINAMATH_CALUDE_luna_has_seventeen_badges_l750_75015

/-- The number of spelling badges Luna has, given the total number of badges and the number of badges Hermione and Celestia have. -/
def luna_badges (total : ℕ) (hermione : ℕ) (celestia : ℕ) : ℕ :=
  total - (hermione + celestia)

/-- Theorem stating that Luna has 17 spelling badges given the conditions in the problem. -/
theorem luna_has_seventeen_badges :
  luna_badges 83 14 52 = 17 := by
  sorry

end NUMINAMATH_CALUDE_luna_has_seventeen_badges_l750_75015


namespace NUMINAMATH_CALUDE_fish_sales_profit_maximization_l750_75089

-- Define the linear relationship between y and x
def linear_relationship (k b x : ℝ) : ℝ := k * x + b

-- Define the daily sales profit function
def daily_sales_profit (x : ℝ) : ℝ := (x - 30) * (linear_relationship (-10) 600 x)

theorem fish_sales_profit_maximization :
  -- Given conditions
  let y₁ := linear_relationship (-10) 600 50
  let y₂ := linear_relationship (-10) 600 40
  -- Theorem statements
  y₁ = 100 ∧
  y₂ = 200 ∧
  (∀ x : ℝ, 30 ≤ x → x < 60 → daily_sales_profit x ≤ daily_sales_profit 45) ∧
  daily_sales_profit 45 = 2250 :=
by sorry


end NUMINAMATH_CALUDE_fish_sales_profit_maximization_l750_75089


namespace NUMINAMATH_CALUDE_least_possible_smallest_integer_l750_75016

theorem least_possible_smallest_integer
  (a b c d : ℤ) -- Four different integers
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) -- Integers are different
  (h_avg : (a + b + c + d) / 4 = 68) -- Average is 68
  (h_max : d = 90) -- Largest integer is 90
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d) -- Order of integers
  : a ≥ 5 := -- Least possible value of smallest integer is 5
by sorry

end NUMINAMATH_CALUDE_least_possible_smallest_integer_l750_75016


namespace NUMINAMATH_CALUDE_mia_nia_difference_l750_75091

/-- Represents a driving scenario with three drivers: Leo, Nia, and Mia. -/
structure DrivingScenario where
  /-- Leo's driving time in hours -/
  t : ℝ
  /-- Leo's driving speed in miles per hour -/
  s : ℝ
  /-- Leo's total distance driven in miles -/
  d : ℝ
  /-- Nia's total distance driven in miles -/
  nia_d : ℝ
  /-- Mia's total distance driven in miles -/
  mia_d : ℝ
  /-- Leo's distance equals speed times time -/
  leo_distance : d = s * t
  /-- Nia drove 2 hours longer than Leo at 10 mph faster -/
  nia_equation : nia_d = (s + 10) * (t + 2)
  /-- Mia drove 3 hours longer than Leo at 15 mph faster -/
  mia_equation : mia_d = (s + 15) * (t + 3)
  /-- Nia drove 110 miles more than Leo -/
  nia_leo_diff : nia_d = d + 110

/-- Theorem stating that Mia drove 100 miles more than Nia -/
theorem mia_nia_difference (scenario : DrivingScenario) : 
  scenario.mia_d - scenario.nia_d = 100 := by
  sorry

end NUMINAMATH_CALUDE_mia_nia_difference_l750_75091


namespace NUMINAMATH_CALUDE_max_bisections_for_zero_approximation_l750_75026

/-- Theorem: Maximum number of bisections for approximating a zero --/
theorem max_bisections_for_zero_approximation 
  (f : ℝ → ℝ) 
  (zero_exists : ∃ x, x ∈ (Set.Ioo 0 1) ∧ f x = 0) 
  (accuracy : ℝ := 0.01) :
  (∃ n : ℕ, n ≤ 7 ∧ 
    (1 : ℝ) / (2 ^ n) < accuracy ∧ 
    ∀ m : ℕ, m < n → (1 : ℝ) / (2 ^ m) ≥ accuracy) :=
sorry

end NUMINAMATH_CALUDE_max_bisections_for_zero_approximation_l750_75026


namespace NUMINAMATH_CALUDE_sector_central_angle_l750_75086

/-- The central angle of a sector with radius R and circumference 3R is 1 radian. -/
theorem sector_central_angle (R : ℝ) (R_pos : R > 0) : 
  let circumference := 3 * R
  let central_angle := circumference / R - 2
  central_angle = 1 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l750_75086


namespace NUMINAMATH_CALUDE_HNO3_calculation_l750_75044

-- Define the chemical equation
def chemical_equation : String := "CaO + 2 HNO₃ → Ca(NO₃)₂ + H₂O"

-- Define the initial amount of CaO in moles
def initial_CaO : ℝ := 7

-- Define the stoichiometric ratio of HNO₃ to CaO
def stoichiometric_ratio : ℝ := 2

-- Define atomic weights
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Theorem to prove
theorem HNO3_calculation (chemical_equation : String) (initial_CaO : ℝ) 
  (stoichiometric_ratio : ℝ) (atomic_weight_H : ℝ) (atomic_weight_N : ℝ) 
  (atomic_weight_O : ℝ) :
  let moles_HNO3 : ℝ := initial_CaO * stoichiometric_ratio
  let molecular_weight_HNO3 : ℝ := atomic_weight_H + atomic_weight_N + 3 * atomic_weight_O
  (moles_HNO3 = 14 ∧ molecular_weight_HNO3 = 63.02) :=
by
  sorry

end NUMINAMATH_CALUDE_HNO3_calculation_l750_75044


namespace NUMINAMATH_CALUDE_mrs_heine_biscuits_l750_75030

theorem mrs_heine_biscuits (num_dogs : ℕ) (biscuits_per_dog : ℕ) 
  (h1 : num_dogs = 2) (h2 : biscuits_per_dog = 3) : 
  num_dogs * biscuits_per_dog = 6 := by
  sorry

end NUMINAMATH_CALUDE_mrs_heine_biscuits_l750_75030


namespace NUMINAMATH_CALUDE_cylinder_radius_l750_75056

theorem cylinder_radius (h : ℝ) (r : ℝ) : 
  h = 4 → 
  π * (r + 10)^2 * h = π * r^2 * (h + 10) → 
  r = 4 + 2 * Real.sqrt 14 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_radius_l750_75056


namespace NUMINAMATH_CALUDE_wednesday_speed_l750_75059

/-- Jonathan's exercise routine -/
structure ExerciseRoutine where
  monday_speed : ℝ
  wednesday_speed : ℝ
  friday_speed : ℝ
  daily_distance : ℝ
  weekly_time : ℝ

/-- Theorem: Jonathan's walking speed on Wednesdays is 3 miles per hour -/
theorem wednesday_speed (routine : ExerciseRoutine)
  (h1 : routine.monday_speed = 2)
  (h2 : routine.friday_speed = 6)
  (h3 : routine.daily_distance = 6)
  (h4 : routine.weekly_time = 6) :
  routine.wednesday_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_speed_l750_75059


namespace NUMINAMATH_CALUDE_octahedron_theorem_l750_75042

/-- A point in 3D space -/
structure Point3D where
  x : ℚ
  y : ℚ
  z : ℚ

/-- Checks if a point lies on any plane defined by x ± y ± z = n for integer n -/
def liesOnPlane (p : Point3D) : Prop :=
  ∃ n : ℤ, (p.x + p.y + p.z = n) ∨ (p.x + p.y - p.z = n) ∨
           (p.x - p.y + p.z = n) ∨ (p.x - p.y - p.z = n) ∨
           (-p.x + p.y + p.z = n) ∨ (-p.x + p.y - p.z = n) ∨
           (-p.x - p.y + p.z = n) ∨ (-p.x - p.y - p.z = n)

/-- Checks if a point lies strictly inside an octahedron -/
def insideOctahedron (p : Point3D) : Prop :=
  ∃ n : ℤ, (n < p.x + p.y + p.z) ∧ (p.x + p.y + p.z < n + 1) ∧
           (n < p.x + p.y - p.z) ∧ (p.x + p.y - p.z < n + 1) ∧
           (n < p.x - p.y + p.z) ∧ (p.x - p.y + p.z < n + 1) ∧
           (n < -p.x + p.y + p.z) ∧ (-p.x + p.y + p.z < n + 1)

theorem octahedron_theorem (p : Point3D) (h : ¬ liesOnPlane p) :
  ∃ k : ℕ, insideOctahedron ⟨k * p.x, k * p.y, k * p.z⟩ := by
  sorry

end NUMINAMATH_CALUDE_octahedron_theorem_l750_75042


namespace NUMINAMATH_CALUDE_converse_not_hold_for_naturals_l750_75090

theorem converse_not_hold_for_naturals : 
  ∃ (a b c d : ℕ), a + d = b + c ∧ (a < c ∨ b < d) :=
sorry

end NUMINAMATH_CALUDE_converse_not_hold_for_naturals_l750_75090


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l750_75012

theorem arithmetic_geometric_inequality (a b : ℝ) :
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧
  ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l750_75012


namespace NUMINAMATH_CALUDE_polynomial_Q_l750_75014

/-- Given a polynomial Q(x) = Q(0) + Q(1)x + Q(3)x³ where Q(-1) = 2,
    prove that Q(x) = -2x + (2/9)x³ - 2/9 -/
theorem polynomial_Q (Q : ℝ → ℝ) : 
  (∀ x, Q x = Q 0 + Q 1 * x + Q 3 * x^3) → 
  Q (-1) = 2 → 
  ∀ x, Q x = -2 * x + (2/9) * x^3 - 2/9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_Q_l750_75014


namespace NUMINAMATH_CALUDE_round_trip_speed_calculation_l750_75002

/-- Represents a round trip journey between two cities -/
structure RoundTrip where
  initial_speed : ℝ
  initial_time : ℝ
  return_time : ℝ
  average_speed : ℝ
  distance : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem round_trip_speed_calculation (trip : RoundTrip) 
  (h1 : trip.return_time = 2 * trip.initial_time)
  (h2 : trip.average_speed = 34)
  (h3 : trip.distance > 0)
  (h4 : trip.initial_speed > 0) :
  trip.initial_speed = 51 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_speed_calculation_l750_75002


namespace NUMINAMATH_CALUDE_pat_stickers_l750_75073

def stickers_problem (initial_stickers end_stickers : ℝ) : Prop :=
  initial_stickers - end_stickers = 22

theorem pat_stickers : stickers_problem 39 17 := by
  sorry

end NUMINAMATH_CALUDE_pat_stickers_l750_75073


namespace NUMINAMATH_CALUDE_a_in_A_sufficient_not_necessary_for_a_in_B_l750_75060

def A : Set ℝ := {1, 2, 3}
def B : Set ℝ := {x | 0 < x ∧ x < 4}

theorem a_in_A_sufficient_not_necessary_for_a_in_B :
  (∀ a, a ∈ A → a ∈ B) ∧ (∃ a, a ∈ B ∧ a ∉ A) := by sorry

end NUMINAMATH_CALUDE_a_in_A_sufficient_not_necessary_for_a_in_B_l750_75060


namespace NUMINAMATH_CALUDE_work_completion_time_l750_75094

theorem work_completion_time (original_men : ℕ) (original_days : ℕ) (absent_men : ℕ) 
  (h1 : original_men = 180)
  (h2 : original_days = 55)
  (h3 : absent_men = 15) :
  (original_men * original_days) / (original_men - absent_men) = 60 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l750_75094


namespace NUMINAMATH_CALUDE_min_intersection_cardinality_l750_75031

-- Define the sets A, B, and C
variable (A B C : Set α)

-- Define the cardinality function
variable (card : Set α → ℕ)

-- Define the conditions
variable (h1 : card A = 50)
variable (h2 : card B = 50)
variable (h3 : card (A ∩ B) = 45)
variable (h4 : card (B ∩ C) = 40)
variable (h5 : card A + card B + card C = card (A ∪ B ∪ C))

-- State the theorem
theorem min_intersection_cardinality :
  ∃ (x : ℕ), x = card (A ∩ B ∩ C) ∧ 
  (∀ (y : ℕ), y = card (A ∩ B ∩ C) → x ≤ y) ∧
  x = 21 := by
sorry

end NUMINAMATH_CALUDE_min_intersection_cardinality_l750_75031


namespace NUMINAMATH_CALUDE_stock_price_change_l750_75080

theorem stock_price_change (total_stocks : ℕ) 
  (h1 : total_stocks = 1980)
  (h2 : ∃ (higher lower : ℕ), 
    higher + lower = total_stocks ∧ 
    higher = lower + (lower / 5)) :
  ∃ (higher : ℕ), higher = 1080 ∧ 
    ∃ (lower : ℕ), higher + lower = total_stocks ∧ 
    higher = lower + (lower / 5) := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l750_75080


namespace NUMINAMATH_CALUDE_trucks_sold_l750_75037

theorem trucks_sold (total : ℕ) (car_truck_diff : ℕ) (h1 : total = 69) (h2 : car_truck_diff = 27) :
  ∃ trucks : ℕ, trucks * 2 + car_truck_diff = total ∧ trucks = 21 :=
by sorry

end NUMINAMATH_CALUDE_trucks_sold_l750_75037


namespace NUMINAMATH_CALUDE_triangle_not_acute_l750_75022

theorem triangle_not_acute (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 30) (h3 : B = 50) :
  ¬ (A < 90 ∧ B < 90 ∧ C < 90) :=
sorry

end NUMINAMATH_CALUDE_triangle_not_acute_l750_75022


namespace NUMINAMATH_CALUDE_augmented_matrix_problem_l750_75023

/-- Given a system of linear equations with augmented matrix
    ⎛ 3 2 1 ⎞
    ⎝ 1 1 m ⎠
    where Dx = 5, prove that m = -2 -/
theorem augmented_matrix_problem (m : ℝ) : 
  let A : Matrix (Fin 2) (Fin 3) ℝ := ![![3, 2, 1], ![1, 1, m]]
  let Dx := (A 0 2 * A 1 1 - A 0 1 * A 1 2) / (A 0 0 * A 1 1 - A 0 1 * A 1 0)
  Dx = 5 → m = -2 := by
  sorry


end NUMINAMATH_CALUDE_augmented_matrix_problem_l750_75023


namespace NUMINAMATH_CALUDE_marble_problem_l750_75036

theorem marble_problem (x y : ℕ) : 
  (y - 4 = 2 * (x + 4)) → 
  (y + 2 = 11 * (x - 2)) → 
  (y = 20 ∧ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_marble_problem_l750_75036


namespace NUMINAMATH_CALUDE_smallest_a_value_l750_75058

/-- Given a polynomial x^3 - ax^2 + bx - 1890 with three positive integer roots,
    prove that the smallest possible value of a is 41 -/
theorem smallest_a_value (a b : ℤ) (x₁ x₂ x₃ : ℤ) : 
  (∀ x, x^3 - a*x^2 + b*x - 1890 = (x - x₁) * (x - x₂) * (x - x₃)) →
  x₁ > 0 → x₂ > 0 → x₃ > 0 →
  x₁ * x₂ * x₃ = 1890 →
  a = x₁ + x₂ + x₃ →
  ∀ a' : ℤ, (∃ b' x₁' x₂' x₃' : ℤ, 
    (∀ x, x^3 - a'*x^2 + b'*x - 1890 = (x - x₁') * (x - x₂') * (x - x₃')) ∧
    x₁' > 0 ∧ x₂' > 0 ∧ x₃' > 0 ∧
    x₁' * x₂' * x₃' = 1890) →
  a' ≥ 41 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l750_75058


namespace NUMINAMATH_CALUDE_tens_digit_of_23_to_2045_l750_75068

theorem tens_digit_of_23_to_2045 : 23^2045 ≡ 43 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_23_to_2045_l750_75068


namespace NUMINAMATH_CALUDE_total_distance_is_55km_l750_75018

/-- Represents the distances Ivan ran on each day of the week -/
structure RunningDistances where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- The conditions of Ivan's running schedule -/
def validRunningSchedule (d : RunningDistances) : Prop :=
  d.tuesday = 2 * d.monday ∧
  d.wednesday = d.tuesday / 2 ∧
  d.thursday = d.wednesday / 2 ∧
  d.friday = 2 * d.thursday ∧
  d.thursday = 5 -- The shortest distance is 5 km, which occurs on Thursday

/-- The theorem to prove -/
theorem total_distance_is_55km (d : RunningDistances) 
  (h : validRunningSchedule d) : 
  d.monday + d.tuesday + d.wednesday + d.thursday + d.friday = 55 := by
  sorry


end NUMINAMATH_CALUDE_total_distance_is_55km_l750_75018


namespace NUMINAMATH_CALUDE_equation_solution_l750_75007

theorem equation_solution (m n : ℝ) : 
  (∀ x : ℝ, (2*x - 5)*(x + m) = 2*x^2 - 3*x + n) → 
  (m = 1 ∧ n = -5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l750_75007


namespace NUMINAMATH_CALUDE_monkey_arrangements_l750_75009

theorem monkey_arrangements :
  (Finset.range 6).prod (λ i => 6 - i) = 720 := by
  sorry

end NUMINAMATH_CALUDE_monkey_arrangements_l750_75009


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l750_75038

/-- The volume of a cone with the same radius and height as a cylinder with volume 54π cm³ is 18π cm³ -/
theorem cone_volume_from_cylinder (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  π * r^2 * h = 54 * π → (1/3) * π * r^2 * h = 18 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l750_75038


namespace NUMINAMATH_CALUDE_equal_roots_condition_l750_75084

theorem equal_roots_condition (m : ℝ) : 
  (∃ (x : ℝ), (x * (x - 3) - (m + 2)) / ((x - 3) * (m - 2)) = x / m ∧ 
   ∀ (y : ℝ), (y * (y - 3) - (m + 2)) / ((y - 3) * (m - 2)) = y / m → y = x) ↔ 
  (m = 2 ∨ m = (9 + Real.sqrt 57) / 8 ∨ m = (9 - Real.sqrt 57) / 8) := by
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l750_75084


namespace NUMINAMATH_CALUDE_carnation_percentage_is_67_point_5_l750_75070

/-- Represents a flower display with pink and red flowers, either roses or carnations -/
structure FlowerDisplay where
  total : ℝ
  pink_ratio : ℝ
  red_carnation_ratio : ℝ
  pink_rose_ratio : ℝ

/-- Calculates the percentage of carnations in the flower display -/
def carnation_percentage (display : FlowerDisplay) : ℝ :=
  let red_ratio := 1 - display.pink_ratio
  let pink_carnation_ratio := display.pink_ratio * (1 - display.pink_rose_ratio)
  let red_carnation_ratio := red_ratio * display.red_carnation_ratio
  (pink_carnation_ratio + red_carnation_ratio) * 100

/-- Theorem stating that under given conditions, 67.5% of flowers are carnations -/
theorem carnation_percentage_is_67_point_5
  (display : FlowerDisplay)
  (h_pink_ratio : display.pink_ratio = 7/10)
  (h_red_carnation_ratio : display.red_carnation_ratio = 1/2)
  (h_pink_rose_ratio : display.pink_rose_ratio = 1/4) :
  carnation_percentage display = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_carnation_percentage_is_67_point_5_l750_75070


namespace NUMINAMATH_CALUDE_completing_square_transform_l750_75011

theorem completing_square_transform (x : ℝ) : 
  (x^2 - 2*x - 7 = 0) ↔ ((x - 1)^2 = 8) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_transform_l750_75011


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_2_l750_75092

theorem no_solution_implies_a_leq_2 (a : ℝ) : 
  (∀ x : ℝ, ¬(2*x - 4 > 0 ∧ x - a < 0)) → a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_2_l750_75092


namespace NUMINAMATH_CALUDE_range_of_a_l750_75081

-- Define the function representing |x-2|+|x+3|
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Define the condition that the solution set is ℝ
def solution_set_is_reals (a : ℝ) : Prop :=
  ∀ x : ℝ, f x ≥ a

-- Theorem statement
theorem range_of_a (a : ℝ) :
  solution_set_is_reals a ↔ a ∈ Set.Iic 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l750_75081


namespace NUMINAMATH_CALUDE_tangent_lines_parallel_to_4x_minus_1_l750_75076

/-- The curve function f(x) = x³ + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_lines_parallel_to_4x_minus_1 :
  ∃! (a b : ℝ), 
    (∃ (x : ℝ), f' x = 4 ∧ 
      (∀ y : ℝ, y = 4 * x + a ↔ y - f x = f' x * (y - x))) ∧
    (∃ (x : ℝ), f' x = 4 ∧ 
      (∀ y : ℝ, y = 4 * x + b ↔ y - f x = f' x * (y - x))) ∧
    a ≠ b ∧ 
    ({a, b} : Set ℝ) = {-4, 0} :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_parallel_to_4x_minus_1_l750_75076


namespace NUMINAMATH_CALUDE_paint_mixture_theorem_l750_75066

/-- Proves that mixing 5 gallons of 20% yellow paint with 5/3 gallons of 40% yellow paint 
    results in a mixture that is 25% yellow -/
theorem paint_mixture_theorem (x : ℝ) :
  let light_green_volume : ℝ := 5
  let light_green_yellow_percent : ℝ := 0.2
  let dark_green_yellow_percent : ℝ := 0.4
  let target_yellow_percent : ℝ := 0.25
  x = 5/3 →
  (light_green_volume * light_green_yellow_percent + x * dark_green_yellow_percent) / 
  (light_green_volume + x) = target_yellow_percent :=
by sorry

end NUMINAMATH_CALUDE_paint_mixture_theorem_l750_75066


namespace NUMINAMATH_CALUDE_glenda_skating_speed_l750_75019

/-- Prove Glenda's skating speed given the conditions of the problem -/
theorem glenda_skating_speed 
  (ann_speed : ℝ) 
  (time : ℝ) 
  (total_distance : ℝ) 
  (h1 : ann_speed = 6)
  (h2 : time = 3)
  (h3 : total_distance = 42) :
  ∃ (glenda_speed : ℝ), 
    glenda_speed = 8 ∧ 
    ann_speed * time + glenda_speed * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_glenda_skating_speed_l750_75019


namespace NUMINAMATH_CALUDE_car_trip_distance_l750_75043

theorem car_trip_distance (D : ℝ) : 
  (1/2 : ℝ) * D + (1/4 : ℝ) * ((1/2 : ℝ) * D) + 105 = D → D = 280 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_distance_l750_75043


namespace NUMINAMATH_CALUDE_base_conversion_568_to_octal_l750_75083

theorem base_conversion_568_to_octal :
  (1 * 8^3 + 0 * 8^2 + 7 * 8^1 + 0 * 8^0 : ℕ) = 568 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_568_to_octal_l750_75083
