import Mathlib

namespace NUMINAMATH_CALUDE_inequality_holds_l3203_320360

theorem inequality_holds (φ : Real) (h : φ > 0 ∧ φ < Real.pi / 2) :
  Real.sin (Real.cos φ) < Real.cos φ ∧ Real.cos φ < Real.cos (Real.sin φ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3203_320360


namespace NUMINAMATH_CALUDE_boy_running_speed_l3203_320317

/-- The speed of a boy running around a square field -/
theorem boy_running_speed (side_length : Real) (time : Real) : 
  side_length = 60 → time = 72 → (4 * side_length) / time * (3600 / 1000) = 12 := by
  sorry

end NUMINAMATH_CALUDE_boy_running_speed_l3203_320317


namespace NUMINAMATH_CALUDE_one_cow_drinking_time_l3203_320309

/-- Represents the drinking rate of cows and the spring inflow rate -/
structure PondSystem where
  /-- Amount of water one cow drinks per day -/
  cow_drink_rate : ℝ
  /-- Amount of water springs add to the pond per day -/
  spring_rate : ℝ
  /-- Total volume of the pond -/
  pond_volume : ℝ

/-- Given the conditions, proves that one cow will take 75 days to drink the pond -/
theorem one_cow_drinking_time (sys : PondSystem)
  (h1 : sys.pond_volume + 3 * sys.spring_rate = 3 * 17 * sys.cow_drink_rate)
  (h2 : sys.pond_volume + 30 * sys.spring_rate = 30 * 2 * sys.cow_drink_rate) :
  sys.pond_volume + 75 * sys.spring_rate = 75 * sys.cow_drink_rate :=
by sorry


end NUMINAMATH_CALUDE_one_cow_drinking_time_l3203_320309


namespace NUMINAMATH_CALUDE_geometric_series_sum_specific_geometric_series_sum_l3203_320316

def geometric_series (a : ℝ) (r : ℝ) : ℕ → ℝ := fun n => a * r^n

theorem geometric_series_sum (a : ℝ) (r : ℝ) (h : |r| < 1) :
  ∑' n, geometric_series a r n = a / (1 - r) :=
sorry

theorem specific_geometric_series_sum :
  ∑' n, geometric_series 1 (1/3) n = 3/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_specific_geometric_series_sum_l3203_320316


namespace NUMINAMATH_CALUDE_log_equality_implies_relation_l3203_320315

theorem log_equality_implies_relation (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  Real.log p + Real.log q + Real.log r = Real.log (p * q * r + p + q) → p = -q := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_relation_l3203_320315


namespace NUMINAMATH_CALUDE_amy_tickets_l3203_320320

/-- The number of tickets Amy started with -/
def initial_tickets : ℕ := 33

/-- The number of tickets Amy bought -/
def bought_tickets : ℕ := 21

/-- The total number of tickets Amy had -/
def total_tickets : ℕ := 54

theorem amy_tickets : initial_tickets + bought_tickets = total_tickets := by
  sorry

end NUMINAMATH_CALUDE_amy_tickets_l3203_320320


namespace NUMINAMATH_CALUDE_reciprocal_greater_than_one_l3203_320353

theorem reciprocal_greater_than_one (x : ℝ) : 
  (x ≠ 0 ∧ (1 / x) > 1) ↔ (0 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_greater_than_one_l3203_320353


namespace NUMINAMATH_CALUDE_digit_sum_problem_l3203_320399

theorem digit_sum_problem (P Q : ℕ) : 
  P < 10 → Q < 10 → 77 * P + 77 * Q = 1000 * P + 100 * P + 10 * P + 7 → P + Q = 14 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l3203_320399


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l3203_320346

def I : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 3, 4}
def B : Set Nat := {3, 4, 5, 6}

theorem complement_intersection_A_B :
  (I \ (A ∩ B)) = {1, 2, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l3203_320346


namespace NUMINAMATH_CALUDE_function_equality_l3203_320370

theorem function_equality (f : ℝ → ℝ) 
  (h1 : ∀ x, f x ≤ x) 
  (h2 : ∀ x y, f (x + y) ≤ f x + f y) : 
  ∀ x, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l3203_320370


namespace NUMINAMATH_CALUDE_odd_product_units_digit_l3203_320367

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_between (n a b : ℕ) : Prop := a < n ∧ n < b

def units_digit (n : ℕ) : ℕ := n % 10

theorem odd_product_units_digit :
  ∃ (prod : ℕ),
    (∀ n : ℕ, is_odd n ∧ is_between n 20 130 → n ∣ prod) ∧
    units_digit prod = 5 :=
by sorry

end NUMINAMATH_CALUDE_odd_product_units_digit_l3203_320367


namespace NUMINAMATH_CALUDE_eric_marbles_l3203_320375

theorem eric_marbles (total : ℕ) (white : ℕ) (blue : ℕ) (green : ℕ) 
  (h1 : total = 20)
  (h2 : white = 12)
  (h3 : blue = 6)
  (h4 : green = total - (white + blue)) :
  green = 2 := by
sorry

end NUMINAMATH_CALUDE_eric_marbles_l3203_320375


namespace NUMINAMATH_CALUDE_derivative_even_implies_a_zero_l3203_320352

/-- Given a real number a and a function f(x) = x³ + ax² + (a-2)x,
    if f'(x) is an even function, then a = 0 -/
theorem derivative_even_implies_a_zero (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + (a-2)*x
  (∀ x, (deriv f) x = (deriv f) (-x)) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_derivative_even_implies_a_zero_l3203_320352


namespace NUMINAMATH_CALUDE_sqrt_12_plus_sqrt_27_l3203_320349

theorem sqrt_12_plus_sqrt_27 : Real.sqrt 12 + Real.sqrt 27 = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_plus_sqrt_27_l3203_320349


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_and_5_l3203_320384

theorem smallest_perfect_square_divisible_by_2_and_5 :
  ∀ n : ℕ, n > 0 → (∃ m : ℕ, n = m^2) → n % 2 = 0 → n % 5 = 0 → n ≥ 100 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_and_5_l3203_320384


namespace NUMINAMATH_CALUDE_elmo_sandwich_jam_cost_l3203_320311

/-- The cost of blackberry jam used in Elmo's sandwiches -/
theorem elmo_sandwich_jam_cost :
  ∀ (N B J : ℕ),
    N > 1 →
    B > 0 →
    J > 0 →
    N * (6 * B + 7 * J) = 396 →
    (N * J * 7 : ℚ) / 100 = 378 / 100 := by
  sorry

end NUMINAMATH_CALUDE_elmo_sandwich_jam_cost_l3203_320311


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3203_320313

theorem binomial_expansion_coefficient (x : ℝ) : 
  let expansion := (x - 2 / Real.sqrt x) ^ 5
  ∃ c : ℝ, c = 40 ∧ 
    ∃ other_terms : ℝ → ℝ, 
      expansion = c * x^2 + other_terms x :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3203_320313


namespace NUMINAMATH_CALUDE_a_1000_equals_divisors_of_1000_l3203_320339

/-- A sequence of real numbers satisfying the given power series equality -/
def PowerSeriesSequence (a : ℕ → ℝ) : Prop :=
  ∀ x : ℝ, abs x < 1 →
    (∑' n : ℕ, x^n / (1 - x^n)) = ∑' i : ℕ, a i * x^i

/-- The number of divisors of a natural number -/
def numberOfDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem a_1000_equals_divisors_of_1000 (a : ℕ → ℝ) (h : PowerSeriesSequence a) :
    a 1000 = numberOfDivisors 1000 := by
  sorry

end NUMINAMATH_CALUDE_a_1000_equals_divisors_of_1000_l3203_320339


namespace NUMINAMATH_CALUDE_triangle_properties_l3203_320308

noncomputable section

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def triangle (a b c : ℝ) := true

theorem triangle_properties (a b c : ℝ) (h : triangle a b c) 
  (h1 : a^2 + 11*b^2 = 2 * Real.sqrt 3 * a * b)
  (h2 : Real.sin c = 2 * Real.sqrt 3 * Real.sin b)
  (h3 : Real.cos b * a * c = Real.tan b) :
  Real.cos b = 1/2 ∧ (1/2 * a * c * Real.sin b = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3203_320308


namespace NUMINAMATH_CALUDE_salary_reduction_percentage_l3203_320300

theorem salary_reduction_percentage 
  (original : ℝ) 
  (reduced : ℝ) 
  (increase_percentage : ℝ) 
  (h1 : increase_percentage = 38.88888888888889)
  (h2 : reduced * (1 + increase_percentage / 100) = original) :
  ∃ (reduction_percentage : ℝ), 
    reduction_percentage = 28 ∧ 
    reduced = original * (1 - reduction_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_salary_reduction_percentage_l3203_320300


namespace NUMINAMATH_CALUDE_no_100_digit_page_numbering_l3203_320366

theorem no_100_digit_page_numbering :
  ¬ ∃ (n : ℕ), n > 0 ∧ (
    let single_digit_sum := min n 9
    let double_digit_sum := if n > 9 then 2 * (n - 9) else 0
    single_digit_sum + double_digit_sum = 100
  ) := by
  sorry

end NUMINAMATH_CALUDE_no_100_digit_page_numbering_l3203_320366


namespace NUMINAMATH_CALUDE_percentage_increase_l3203_320373

theorem percentage_increase (original : ℝ) (new : ℝ) (percentage : ℝ) : 
  original = 80 →
  new = 88.8 →
  percentage = 11 →
  (new - original) / original * 100 = percentage :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_l3203_320373


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l3203_320396

theorem sum_of_x_and_y_equals_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 20) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l3203_320396


namespace NUMINAMATH_CALUDE_probability_of_specific_arrangement_l3203_320395

def total_tiles : ℕ := 8
def x_tiles : ℕ := 5
def o_tiles : ℕ := 3

def specific_arrangement : List Char := ['X', 'X', 'X', 'O', 'O', 'X', 'O', 'X']

theorem probability_of_specific_arrangement :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles) = 1 / 56 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_arrangement_l3203_320395


namespace NUMINAMATH_CALUDE_square_odd_implies_odd_l3203_320336

theorem square_odd_implies_odd (n : ℤ) : Odd (n^2) → Odd n := by
  sorry

end NUMINAMATH_CALUDE_square_odd_implies_odd_l3203_320336


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l3203_320319

theorem gcd_special_numbers : Nat.gcd 3333333 666666666 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l3203_320319


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l3203_320379

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.exp (x^2) - Real.cos x) / x else 0

theorem f_derivative_at_zero : 
  deriv f 0 = (3/2) := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l3203_320379


namespace NUMINAMATH_CALUDE_expression_evaluation_l3203_320354

theorem expression_evaluation (a : ℝ) 
  (h1 : a ≥ 0) 
  (h2 : a ≠ 1) 
  (h3 : a ≠ 1 + Real.sqrt 2) 
  (h4 : a ≠ 1 - Real.sqrt 2) :
  (1 + 2 * a^(1/4) - a^(1/2)) / (1 - a + 4 * a^(3/4) - 4 * a^(1/2)) +
  (a^(1/4) - 2) / ((a^(1/4) - 1)^2) = 1 / (a^(1/4) - 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3203_320354


namespace NUMINAMATH_CALUDE_glass_piece_coloring_l3203_320359

/-- Represents the count of glass pieces for each color -/
structure GlassPieces where
  red : ℕ
  yellow : ℕ
  blue : ℕ
  sum_is_2005 : red + yellow + blue = 2005

/-- Represents a single operation on the glass pieces -/
inductive Operation
  | RedYellowToBlue
  | RedBlueToYellow
  | YellowBlueToRed

/-- Applies an operation to the glass pieces -/
def apply_operation (gp : GlassPieces) (op : Operation) : GlassPieces :=
  match op with
  | Operation.RedYellowToBlue => 
      { red := gp.red - 1, yellow := gp.yellow - 1, blue := gp.blue + 2, 
        sum_is_2005 := by sorry }
  | Operation.RedBlueToYellow => 
      { red := gp.red - 1, yellow := gp.yellow + 2, blue := gp.blue - 1, 
        sum_is_2005 := by sorry }
  | Operation.YellowBlueToRed => 
      { red := gp.red + 2, yellow := gp.yellow - 1, blue := gp.blue - 1, 
        sum_is_2005 := by sorry }

/-- A sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to the glass pieces -/
def apply_sequence (gp : GlassPieces) (seq : OperationSequence) : GlassPieces :=
  match seq with
  | [] => gp
  | op :: rest => apply_sequence (apply_operation gp op) rest

/-- Predicate to check if all pieces are the same color -/
def all_same_color (gp : GlassPieces) : Prop :=
  (gp.red = 2005 ∧ gp.yellow = 0 ∧ gp.blue = 0) ∨
  (gp.red = 0 ∧ gp.yellow = 2005 ∧ gp.blue = 0) ∨
  (gp.red = 0 ∧ gp.yellow = 0 ∧ gp.blue = 2005)

theorem glass_piece_coloring
  (gp : GlassPieces) :
  (∃ (seq : OperationSequence), all_same_color (apply_sequence gp seq)) ∧
  (∀ (seq1 seq2 : OperationSequence),
    all_same_color (apply_sequence gp seq1) →
    all_same_color (apply_sequence gp seq2) →
    apply_sequence gp seq1 = apply_sequence gp seq2) := by
  sorry

end NUMINAMATH_CALUDE_glass_piece_coloring_l3203_320359


namespace NUMINAMATH_CALUDE_perseverance_arrangement_count_l3203_320329

/-- The number of letters in the word "PERSEVERANCE" -/
def total_letters : ℕ := 12

/-- The number of times the letter 'E' appears in "PERSEVERANCE" -/
def e_count : ℕ := 3

/-- The number of times the letter 'R' appears in "PERSEVERANCE" -/
def r_count : ℕ := 2

/-- The number of unique arrangements of the letters in "PERSEVERANCE" -/
def perseverance_arrangements : ℕ := Nat.factorial total_letters / (Nat.factorial e_count * Nat.factorial r_count)

theorem perseverance_arrangement_count : perseverance_arrangements = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_perseverance_arrangement_count_l3203_320329


namespace NUMINAMATH_CALUDE_book_distribution_l3203_320374

theorem book_distribution (n : ℕ) (k : ℕ) : 
  n = 5 → k = 4 → (k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n) = 292 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_l3203_320374


namespace NUMINAMATH_CALUDE_congruence_properties_l3203_320394

theorem congruence_properties (a b c d : ℤ) (h : b ≡ c [ZMOD d]) :
  (a + b ≡ a + c [ZMOD d]) ∧ 
  (a - b ≡ a - c [ZMOD d]) ∧ 
  (a * b ≡ a * c [ZMOD d]) := by
  sorry

end NUMINAMATH_CALUDE_congruence_properties_l3203_320394


namespace NUMINAMATH_CALUDE_monkey_percentage_after_eating_l3203_320356

/-- The percentage of monkeys among animals after two monkeys each eat one bird -/
theorem monkey_percentage_after_eating (initial_monkeys initial_birds : ℕ) 
  (h1 : initial_monkeys = 6)
  (h2 : initial_birds = 6)
  (h3 : initial_monkeys > 0)
  (h4 : initial_birds ≥ 2) : 
  (initial_monkeys : ℚ) / (initial_monkeys + initial_birds - 2 : ℚ) = 3/5 := by
  sorry

#check monkey_percentage_after_eating

end NUMINAMATH_CALUDE_monkey_percentage_after_eating_l3203_320356


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l3203_320341

-- Define what it means for an angle to be in the third quadrant
def in_third_quadrant (α : Real) : Prop :=
  ∃ k : Int, k * 2 * Real.pi + Real.pi < α ∧ α < k * 2 * Real.pi + 3 * Real.pi / 2

-- Define what it means for an angle to be in the second or fourth quadrant
def in_second_or_fourth_quadrant (α : Real) : Prop :=
  ∃ k : Int, (k * Real.pi + Real.pi / 2 < α ∧ α < k * Real.pi + Real.pi) ∨
             (k * Real.pi + 3 * Real.pi / 2 < α ∧ α < (k + 1) * Real.pi)

-- State the theorem
theorem half_angle_quadrant (α : Real) :
  in_third_quadrant α → in_second_or_fourth_quadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l3203_320341


namespace NUMINAMATH_CALUDE_customer_equation_l3203_320376

theorem customer_equation (X Y Z : ℕ) 
  (h1 : X - Y = 10)
  (h2 : (X - Y) - Z = 4) : 
  X - (X - 10) - 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_customer_equation_l3203_320376


namespace NUMINAMATH_CALUDE_student_selection_methods_l3203_320383

theorem student_selection_methods (n : ℕ) (h : n = 5) : 
  (n.choose 2) * ((n - 2).choose 1) * ((n - 3).choose 1) = 60 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_methods_l3203_320383


namespace NUMINAMATH_CALUDE_advance_agency_fees_calculation_l3203_320337

/-- Proof of advance agency fees calculation -/
theorem advance_agency_fees_calculation 
  (C : ℕ) -- Commission
  (I : ℕ) -- Incentive
  (G : ℕ) -- Amount given to John
  (h1 : C = 25000)
  (h2 : I = 1780)
  (h3 : G = 18500)
  : C + I - G = 8280 := by
  sorry

end NUMINAMATH_CALUDE_advance_agency_fees_calculation_l3203_320337


namespace NUMINAMATH_CALUDE_discounted_price_theorem_l3203_320326

/-- The original price of an article before discounts -/
def original_price : ℝ := 150

/-- The first discount rate -/
def discount1 : ℝ := 0.1

/-- The second discount rate -/
def discount2 : ℝ := 0.2

/-- The final sale price after discounts -/
def final_price : ℝ := 108

/-- Theorem stating that the original price results in the final price after discounts -/
theorem discounted_price_theorem :
  final_price = original_price * (1 - discount1) * (1 - discount2) := by
  sorry

#check discounted_price_theorem

end NUMINAMATH_CALUDE_discounted_price_theorem_l3203_320326


namespace NUMINAMATH_CALUDE_raritet_encounters_l3203_320389

/-- Represents the number of days it takes for a ferry to travel between Dzerzhinsk and Lvov --/
def travel_time : ℕ := 8

/-- Represents the number of ferries departing from Dzerzhinsk during Raritet's journey --/
def ferries_during_journey : ℕ := travel_time

/-- Represents the number of ferries already en route when Raritet departs --/
def ferries_en_route : ℕ := travel_time

/-- Represents the ferry arriving in Lvov when Raritet departs --/
def arriving_ferry : ℕ := 1

/-- Theorem stating the total number of ferries Raritet meets --/
theorem raritet_encounters :
  ferries_during_journey + ferries_en_route + arriving_ferry = 17 :=
sorry

end NUMINAMATH_CALUDE_raritet_encounters_l3203_320389


namespace NUMINAMATH_CALUDE_max_abs_z_l3203_320324

theorem max_abs_z (z : ℂ) (θ : ℝ) (h : z - 1 = Complex.cos θ + Complex.I * Complex.sin θ) :
  Complex.abs z ≤ 2 ∧ ∃ θ₀ : ℝ, Complex.abs (1 + Complex.cos θ₀ + Complex.I * Complex.sin θ₀) = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_z_l3203_320324


namespace NUMINAMATH_CALUDE_line_parallel_plane_relationship_l3203_320306

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Defines when a line is parallel to a plane -/
def parallel_line_plane (l : Line) (α : Plane) : Prop := sorry

/-- Defines when a line is contained within a plane -/
def line_in_plane (a : Line) (α : Plane) : Prop := sorry

/-- Defines when two lines are parallel -/
def parallel_lines (l1 l2 : Line) : Prop := sorry

/-- Defines when two lines are skew -/
def skew_lines (l1 l2 : Line) : Prop := sorry

/-- Theorem: If a line is parallel to a plane, and another line is contained within that plane,
    then the two lines are either parallel or skew -/
theorem line_parallel_plane_relationship (l a : Line) (α : Plane) 
  (h1 : parallel_line_plane l α) (h2 : line_in_plane a α) :
  parallel_lines l a ∨ skew_lines l a := by sorry

end NUMINAMATH_CALUDE_line_parallel_plane_relationship_l3203_320306


namespace NUMINAMATH_CALUDE_selection_with_at_least_one_girl_l3203_320340

def total_students : ℕ := 6
def boys : ℕ := 4
def girls : ℕ := 2
def students_to_select : ℕ := 4

theorem selection_with_at_least_one_girl :
  (Nat.choose total_students students_to_select) - (Nat.choose boys students_to_select) = 14 :=
by sorry

end NUMINAMATH_CALUDE_selection_with_at_least_one_girl_l3203_320340


namespace NUMINAMATH_CALUDE_distances_product_bound_l3203_320381

/-- Given an equilateral triangle with side length 1 and a point P inside it,
    the distances from P to the three sides satisfy 0 < ab + bc + ca ≤ 1/4 -/
theorem distances_product_bound (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = Real.sqrt 3 / 2 → 
  0 < a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_distances_product_bound_l3203_320381


namespace NUMINAMATH_CALUDE_gold_award_winners_possibly_all_freshmen_l3203_320391

theorem gold_award_winners_possibly_all_freshmen 
  (total_winners : ℕ) 
  (selected_students : ℕ) 
  (selected_freshmen : ℕ) 
  (selected_gold : ℕ) 
  (h1 : total_winners = 120)
  (h2 : selected_students = 24)
  (h3 : selected_freshmen = 6)
  (h4 : selected_gold = 4) :
  ∃ (total_freshmen : ℕ) (total_gold : ℕ),
    total_freshmen ≤ total_winners ∧
    total_gold ≤ total_winners ∧
    total_gold ≤ total_freshmen :=
by sorry

end NUMINAMATH_CALUDE_gold_award_winners_possibly_all_freshmen_l3203_320391


namespace NUMINAMATH_CALUDE_smallest_enclosing_sphere_radius_l3203_320302

-- Define a sphere with center and radius
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define the eight spheres
def octantSpheres : List Sphere :=
  [⟨(2, 2, 2), 2⟩, ⟨(-2, 2, 2), 2⟩, ⟨(2, -2, 2), 2⟩, ⟨(2, 2, -2), 2⟩,
   ⟨(-2, -2, 2), 2⟩, ⟨(-2, 2, -2), 2⟩, ⟨(2, -2, -2), 2⟩, ⟨(-2, -2, -2), 2⟩]

-- Function to check if a sphere is tangent to coordinate planes
def isTangentToCoordinatePlanes (s : Sphere) : Prop :=
  let (x, y, z) := s.center
  (|x| = s.radius ∨ |y| = s.radius ∨ |z| = s.radius)

-- Function to check if a sphere contains another sphere
def containsSphere (outer : Sphere) (inner : Sphere) : Prop :=
  let (x₁, y₁, z₁) := outer.center
  let (x₂, y₂, z₂) := inner.center
  ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)^(1/2) + inner.radius ≤ outer.radius

-- Theorem statement
theorem smallest_enclosing_sphere_radius :
  ∃ (r : ℝ), r = 2 + 2 * Real.sqrt 3 ∧
  (∀ s ∈ octantSpheres, isTangentToCoordinatePlanes s) ∧
  (∀ r' : ℝ, r' < r →
    ∃ s ∈ octantSpheres, ¬containsSphere ⟨(0, 0, 0), r'⟩ s) ∧
  (∀ s ∈ octantSpheres, containsSphere ⟨(0, 0, 0), r⟩ s) := by
  sorry

end NUMINAMATH_CALUDE_smallest_enclosing_sphere_radius_l3203_320302


namespace NUMINAMATH_CALUDE_expression_value_l3203_320377

theorem expression_value (m n x y : ℤ) 
  (h1 : m - n = 100) 
  (h2 : x + y = -1) : 
  (n + x) - (m - y) = -101 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3203_320377


namespace NUMINAMATH_CALUDE_unique_quadratic_function_l3203_320372

/-- A quadratic function satisfying specific conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, 
    (∀ x, f x = a * x^2 + b * x + c) ∧ 
    (f (-1) = 0) ∧ 
    (∀ x, x ≤ f x) ∧
    (∀ x, f x ≤ (1 + x^2) / 2)

/-- The unique quadratic function satisfying the given conditions -/
theorem unique_quadratic_function (f : ℝ → ℝ) (hf : QuadraticFunction f) : 
  ∀ x, f x = (1/4) * x^2 + (1/2) * x + 1/4 :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_function_l3203_320372


namespace NUMINAMATH_CALUDE_factor_proof_l3203_320331

theorem factor_proof :
  (∃ n : ℤ, 24 = 4 * n) ∧ (∃ m : ℤ, 180 = 9 * m) := by
  sorry

end NUMINAMATH_CALUDE_factor_proof_l3203_320331


namespace NUMINAMATH_CALUDE_equation_solution_exists_l3203_320358

theorem equation_solution_exists : ∃ c : ℝ, 
  Real.sqrt (4 + Real.sqrt (12 + 6 * c)) + Real.sqrt (6 + Real.sqrt (3 + c)) = 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l3203_320358


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l3203_320390

/-- The area of a square with vertices P(2, 3), Q(-3, 4), R(-2, -1), and S(3, 0) is 26 square units -/
theorem square_area_from_vertices : 
  let P : ℝ × ℝ := (2, 3)
  let Q : ℝ × ℝ := (-3, 4)
  let R : ℝ × ℝ := (-2, -1)
  let S : ℝ × ℝ := (3, 0)
  let square_area := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)^2
  square_area = 26 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l3203_320390


namespace NUMINAMATH_CALUDE_smallest_n_for_equations_l3203_320350

theorem smallest_n_for_equations :
  (∃ (n : ℕ), 
    (∀ (m : ℕ), m < n → ¬∃ (x y : ℕ), x * (x + m) = y^2) ∧
    (∃ (x y : ℕ), x * (x + n) = y^2) ∧
    n = 3) ∧
  (∃ (n : ℕ),
    (∀ (m : ℕ), m < n → ¬∃ (x y : ℕ), x * (x + m) = y^3) ∧
    (∃ (x y : ℕ), x * (x + n) = y^3) ∧
    n = 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_equations_l3203_320350


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l3203_320369

theorem rectangular_parallelepiped_volume 
  (x y z : ℝ) 
  (h1 : (x^2 + y^2) * z^2 = 13) 
  (h2 : (y^2 + z^2) * x^2 = 40) 
  (h3 : (x^2 + z^2) * y^2 = 45) : 
  x * y * z = 6 := by
sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l3203_320369


namespace NUMINAMATH_CALUDE_max_value_expression_l3203_320363

theorem max_value_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  4*x*y*Real.sqrt 2 + 5*y*z + 3*x*z*Real.sqrt 3 ≤ (44*Real.sqrt 2 + 110 + 9*Real.sqrt 3) / 3 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ ≥ 0 ∧ y₀ ≥ 0 ∧ z₀ ≥ 0 ∧ x₀^2 + y₀^2 + z₀^2 = 1 ∧
    4*x₀*y₀*Real.sqrt 2 + 5*y₀*z₀ + 3*x₀*z₀*Real.sqrt 3 = (44*Real.sqrt 2 + 110 + 9*Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3203_320363


namespace NUMINAMATH_CALUDE_math_contest_participants_l3203_320301

theorem math_contest_participants : ∃ n : ℕ, 
  n > 0 ∧ 
  n = n / 3 + n / 4 + n / 5 + 26 ∧ 
  n = 120 := by
sorry

end NUMINAMATH_CALUDE_math_contest_participants_l3203_320301


namespace NUMINAMATH_CALUDE_max_advancing_teams_l3203_320386

/-- The number of teams in the tournament -/
def num_teams : ℕ := 8

/-- The minimum number of points required to advance -/
def min_points_to_advance : ℕ := 15

/-- The number of points awarded for a win -/
def win_points : ℕ := 3

/-- The number of points awarded for a draw -/
def draw_points : ℕ := 1

/-- The number of points awarded for a loss -/
def loss_points : ℕ := 0

/-- The total number of games played in the tournament -/
def total_games : ℕ := (num_teams * (num_teams - 1)) / 2

/-- The maximum total points possible in the tournament -/
def max_total_points : ℕ := total_games * win_points

/-- The maximum number of teams that can advance to the next round -/
theorem max_advancing_teams :
  ∃ (n : ℕ), n ≤ max_total_points / min_points_to_advance ∧
             n = 5 ∧
             (∀ m : ℕ, m > n → m * min_points_to_advance > max_total_points) :=
by sorry

end NUMINAMATH_CALUDE_max_advancing_teams_l3203_320386


namespace NUMINAMATH_CALUDE_extreme_value_negative_a_one_zero_positive_a_l3203_320323

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1/2) * x^2 - (a + 1) * x

-- Theorem for the case when a < 0
theorem extreme_value_negative_a (a : ℝ) (ha : a < 0) :
  (∃ x : ℝ, ∀ y : ℝ, f a y ≥ f a x) ∧
  (∀ x : ℝ, f a x ≥ -a - 1/2) ∧
  (¬∃ x : ℝ, ∀ y : ℝ, f a y ≤ f a x) :=
sorry

-- Theorem for the case when a > 0
theorem one_zero_positive_a (a : ℝ) (ha : a > 0) :
  ∃! x : ℝ, f a x = 0 :=
sorry

end

end NUMINAMATH_CALUDE_extreme_value_negative_a_one_zero_positive_a_l3203_320323


namespace NUMINAMATH_CALUDE_lucy_paid_correct_l3203_320332

/-- Calculate the total amount Lucy paid for fruits with discounts applied -/
def total_paid (grapes_kg : ℝ) (grapes_price : ℝ) (mangoes_kg : ℝ) (mangoes_price : ℝ)
                (apples_kg : ℝ) (apples_price : ℝ) (oranges_kg : ℝ) (oranges_price : ℝ)
                (grapes_apples_discount : ℝ) (mangoes_oranges_discount : ℝ) : ℝ :=
  let grapes_cost := grapes_kg * grapes_price
  let mangoes_cost := mangoes_kg * mangoes_price
  let apples_cost := apples_kg * apples_price
  let oranges_cost := oranges_kg * oranges_price
  let grapes_apples_total := grapes_cost + apples_cost
  let mangoes_oranges_total := mangoes_cost + oranges_cost
  let grapes_apples_discounted := grapes_apples_total * (1 - grapes_apples_discount)
  let mangoes_oranges_discounted := mangoes_oranges_total * (1 - mangoes_oranges_discount)
  grapes_apples_discounted + mangoes_oranges_discounted

theorem lucy_paid_correct :
  total_paid 6 74 9 59 4 45 12 32 0.07 0.05 = 1449.57 := by
  sorry

end NUMINAMATH_CALUDE_lucy_paid_correct_l3203_320332


namespace NUMINAMATH_CALUDE_special_triangle_common_area_l3203_320357

/-- A triangle with side lengths 18, 24, and 30 -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 18
  hb : b = 24
  hc : c = 30

/-- The common region of two overlapping triangles -/
def CommonRegion (t1 t2 : SpecialTriangle) : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Two triangles share the same circumcircle -/
def ShareCircumcircle (t1 t2 : SpecialTriangle) : Prop := sorry

/-- Two triangles share the same inscribed circle -/
def ShareInscribedCircle (t1 t2 : SpecialTriangle) : Prop := sorry

/-- Two triangles do not completely overlap -/
def NotCompletelyOverlap (t1 t2 : SpecialTriangle) : Prop := sorry

theorem special_triangle_common_area 
  (t1 t2 : SpecialTriangle) 
  (h_circ : ShareCircumcircle t1 t2) 
  (h_insc : ShareInscribedCircle t1 t2) 
  (h_overlap : NotCompletelyOverlap t1 t2) : 
  area (CommonRegion t1 t2) = 132 := by sorry

end NUMINAMATH_CALUDE_special_triangle_common_area_l3203_320357


namespace NUMINAMATH_CALUDE_arcsin_arccos_inequality_l3203_320365

theorem arcsin_arccos_inequality (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 1 →
  (Real.arcsin ((5 / (2 * Real.pi)) * Real.arccos x) > Real.arccos ((10 / (3 * Real.pi)) * Real.arcsin x)) ↔
  (x ∈ Set.Icc (Real.cos (2 * Real.pi / 5)) (Real.cos (8 * Real.pi / 25)) ∪
   Set.Ioo (Real.cos (8 * Real.pi / 25)) (Real.cos (Real.pi / 5))) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_arccos_inequality_l3203_320365


namespace NUMINAMATH_CALUDE_monochromatic_unit_area_triangle_exists_l3203_320304

-- Define a color type
inductive Color
| Red
| Green
| Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring of the plane
def Coloring := Point → Color

-- Define a triangle
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ := sorry

-- Define what it means for a triangle to be monochromatic
def isMonochromatic (t : Triangle) (coloring : Coloring) : Prop :=
  coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c

-- The main theorem
theorem monochromatic_unit_area_triangle_exists (coloring : Coloring) :
  ∃ t : Triangle, triangleArea t = 1 ∧ isMonochromatic t coloring := by sorry

end NUMINAMATH_CALUDE_monochromatic_unit_area_triangle_exists_l3203_320304


namespace NUMINAMATH_CALUDE_log_x_16_eq_0_8_implies_x_eq_32_l3203_320314

-- Define the logarithm function for our specific base
noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_x_16_eq_0_8_implies_x_eq_32 :
  ∀ x : ℝ, x > 0 → log_base x 16 = 0.8 → x = 32 := by
  sorry

end NUMINAMATH_CALUDE_log_x_16_eq_0_8_implies_x_eq_32_l3203_320314


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l3203_320303

def line (x y : ℝ) : Prop := y = 2 * x + 1

theorem distance_to_x_axis (k : ℝ) (h : line (-2) k) : 
  |k| = 3 := by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l3203_320303


namespace NUMINAMATH_CALUDE_first_class_students_l3203_320393

/-- The number of students in the first class -/
def x : ℕ := 24

/-- The number of students in the second class -/
def second_class_students : ℕ := 50

/-- The average marks of the first class -/
def first_class_avg : ℚ := 40

/-- The average marks of the second class -/
def second_class_avg : ℚ := 60

/-- The average marks of all students combined -/
def total_avg : ℚ := 53513513513513516 / 1000000000000000

theorem first_class_students :
  (x * first_class_avg + second_class_students * second_class_avg) / (x + second_class_students) = total_avg := by
  sorry

end NUMINAMATH_CALUDE_first_class_students_l3203_320393


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3203_320333

theorem trigonometric_identity (α β : Real) :
  (Real.cos α)^2 + (Real.cos β)^2 - 2 * (Real.cos α) * (Real.cos β) * Real.cos (α + β) =
  (Real.sin α)^2 + (Real.sin β)^2 + 2 * (Real.sin α) * (Real.sin β) * Real.sin (α + β) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3203_320333


namespace NUMINAMATH_CALUDE_spadesuit_example_l3203_320312

def spadesuit (a b : ℝ) : ℝ := |a - b|

theorem spadesuit_example : spadesuit 3 (spadesuit 5 8) = 0 := by
  sorry

end NUMINAMATH_CALUDE_spadesuit_example_l3203_320312


namespace NUMINAMATH_CALUDE_photocopy_discount_is_25_percent_l3203_320392

/-- The discount percentage for bulk photocopy orders -/
def discount_percentage (cost_per_copy : ℚ) (copies_for_discount : ℕ) 
  (steve_copies : ℕ) (dinley_copies : ℕ) (individual_savings : ℚ) : ℚ :=
  let total_copies := steve_copies + dinley_copies
  let total_cost_without_discount := cost_per_copy * total_copies
  let total_savings := individual_savings * 2
  let total_cost_with_discount := total_cost_without_discount - total_savings
  (total_cost_without_discount - total_cost_with_discount) / total_cost_without_discount * 100

theorem photocopy_discount_is_25_percent :
  discount_percentage 0.02 100 80 80 0.40 = 25 := by
  sorry

end NUMINAMATH_CALUDE_photocopy_discount_is_25_percent_l3203_320392


namespace NUMINAMATH_CALUDE_rectangle_area_l3203_320322

/-- A rectangle ABCD is divided into four identical squares and has a perimeter of 160 cm. -/
structure Rectangle :=
  (side : ℝ)
  (perimeter_eq : 10 * side = 160)

/-- The area of the rectangle ABCD is 1024 square centimeters. -/
theorem rectangle_area (rect : Rectangle) : 4 * rect.side^2 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3203_320322


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_3_mod_17_l3203_320355

theorem smallest_five_digit_congruent_to_3_mod_17 : ∃ (n : ℕ), 
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit positive integer
  (n % 17 = 3) ∧              -- congruent to 3 modulo 17
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ (m % 17 = 3) → m ≥ n) ∧ 
  n = 10012 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_3_mod_17_l3203_320355


namespace NUMINAMATH_CALUDE_sharmila_hourly_wage_l3203_320351

/-- Sharmila's work schedule and earnings -/
structure WorkSchedule where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculate the total hours worked in a week -/
def total_hours (schedule : WorkSchedule) : ℕ :=
  schedule.monday_hours + schedule.tuesday_hours + schedule.wednesday_hours +
  schedule.thursday_hours + schedule.friday_hours

/-- Calculate the hourly wage -/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sharmila's actual work schedule -/
def sharmila_schedule : WorkSchedule :=
  { monday_hours := 10
  , tuesday_hours := 8
  , wednesday_hours := 10
  , thursday_hours := 8
  , friday_hours := 10
  , weekly_earnings := 460 }

theorem sharmila_hourly_wage :
  hourly_wage sharmila_schedule = 10 := by
  sorry

end NUMINAMATH_CALUDE_sharmila_hourly_wage_l3203_320351


namespace NUMINAMATH_CALUDE_banana_arrangements_l3203_320343

def word_length : ℕ := 6
def a_count : ℕ := 3
def n_count : ℕ := 2
def b_count : ℕ := 1

theorem banana_arrangements : 
  (word_length.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l3203_320343


namespace NUMINAMATH_CALUDE_trench_digging_time_l3203_320321

theorem trench_digging_time (a b c d : ℝ) : 
  (a + b + c + d = 1/6) →
  (2*a + (1/2)*b + c + d = 1/6) →
  ((1/2)*a + 2*b + c + d = 1/4) →
  (a + b + c = 1/6) :=
by sorry

end NUMINAMATH_CALUDE_trench_digging_time_l3203_320321


namespace NUMINAMATH_CALUDE_donuts_left_l3203_320335

def initial_donuts : ℕ := 50
def bill_eats : ℕ := 2
def secretary_takes : ℕ := 4

def remaining_donuts : ℕ := 
  let after_bill := initial_donuts - bill_eats
  let after_secretary := after_bill - secretary_takes
  after_secretary / 2

theorem donuts_left : remaining_donuts = 22 := by sorry

end NUMINAMATH_CALUDE_donuts_left_l3203_320335


namespace NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l3203_320398

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l3203_320398


namespace NUMINAMATH_CALUDE_group_frequency_problem_l3203_320318

/-- A problem about frequency and relative frequency in a grouped sample -/
theorem group_frequency_problem (total_sample : ℕ) (num_groups : ℕ) 
  (group_frequencies : Fin 8 → ℕ) :
  total_sample = 100 →
  num_groups = 8 →
  group_frequencies 0 = 10 →
  group_frequencies 1 = 13 →
  group_frequencies 3 = 14 →
  group_frequencies 4 = 15 →
  group_frequencies 5 = 13 →
  group_frequencies 6 = 12 →
  group_frequencies 7 = 9 →
  group_frequencies 2 = 14 ∧ 
  (group_frequencies 2 : ℚ) / total_sample = 14 / 100 :=
by sorry

end NUMINAMATH_CALUDE_group_frequency_problem_l3203_320318


namespace NUMINAMATH_CALUDE_equal_cost_layover_l3203_320380

/-- Represents a city in the country -/
structure City where
  id : Nat

/-- Represents an airline operating in the country -/
structure Airline where
  id : Nat

/-- Represents the transportation network of the country -/
structure CountryNetwork where
  cities : Finset City
  airlines : Finset Airline
  connections : City → City → Finset Airline
  cost : City → City → ℚ

/-- The conditions of the problem -/
def ProblemConditions (network : CountryNetwork) : Prop :=
  (network.cities.card = 100) ∧
  (network.airlines.card = 146) ∧
  (∀ c1 c2 : City, c1 ∈ network.cities → c2 ∈ network.cities → c1 ≠ c2 → 
    network.connections c1 c2 ≠ ∅) ∧
  (∀ c1 c2 : City, c1 ∈ network.cities → c2 ∈ network.cities → c1 ≠ c2 → 
    network.cost c1 c2 = 1 / (network.connections c1 c2).card) ∧
  (∀ c1 c2 c3 : City, c1 ∈ network.cities → c2 ∈ network.cities → c3 ∈ network.cities → 
    c1 ≠ c2 → c2 ≠ c3 → c1 ≠ c3 → 
    network.cost c1 c2 + network.cost c2 c3 ≥ network.cost c1 c3)

/-- The theorem to be proved -/
theorem equal_cost_layover (network : CountryNetwork) 
  (h : ProblemConditions network) : 
  ∃ c1 c2 c3 : City, c1 ∈ network.cities ∧ c2 ∈ network.cities ∧ c3 ∈ network.cities ∧
  c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
  network.cost c1 c2 = network.cost c2 c3 :=
sorry

end NUMINAMATH_CALUDE_equal_cost_layover_l3203_320380


namespace NUMINAMATH_CALUDE_addition_commutative_example_l3203_320305

theorem addition_commutative_example : 73 + 93 + 27 = 73 + 27 + 93 := by
  sorry

end NUMINAMATH_CALUDE_addition_commutative_example_l3203_320305


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3203_320338

-- Define the left-hand side of the equation
def lhs (p : ℝ) : ℝ := (7*p^5 - 4*p^3 + 8*p^2 - 5*p + 3) + (-p^5 + 3*p^3 - 7*p^2 + 6*p + 2)

-- Define the right-hand side of the equation
def rhs (p : ℝ) : ℝ := 6*p^5 - p^3 + p^2 + p + 5

-- Theorem statement
theorem polynomial_simplification (p : ℝ) : lhs p = rhs p := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3203_320338


namespace NUMINAMATH_CALUDE_xyz_inequality_l3203_320334

theorem xyz_inequality (x y z : ℝ) (n : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_sum : x + y + z = 1) (h_pos_n : n > 0) :
  x^n + y^n + z^n ≥ 1 / 3^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l3203_320334


namespace NUMINAMATH_CALUDE_quadratic_equation_theorem_l3203_320347

theorem quadratic_equation_theorem (m : ℝ) (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m - 1 = 0 ∧ y^2 - 2*y + m - 1 = 0) →  -- two real roots condition
  (p^2 - 2*p + m - 1 = 0) →  -- p is a root
  ((p^2 - 2*p + 3)*(m + 4) = 7) →  -- given equation
  (m = -3 ∧ m ≤ 2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_theorem_l3203_320347


namespace NUMINAMATH_CALUDE_sequence_limit_property_l3203_320325

theorem sequence_limit_property (a : ℕ → ℝ) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |a (n + 2) - a n| < ε) →
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |((a (n + 1) - a n) : ℝ) / n| < ε) :=
by sorry

end NUMINAMATH_CALUDE_sequence_limit_property_l3203_320325


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3203_320378

/-- An arithmetic sequence is monotonically increasing if its common difference is positive -/
def IsMonoIncreasingArithmeticSeq (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h_mono : IsMonoIncreasingArithmeticSeq a)
  (h_a3 : a 3 = 1)
  (h_a2a4 : a 2 * a 4 = 3/4) :
  a 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3203_320378


namespace NUMINAMATH_CALUDE_train_length_l3203_320361

/-- The length of a train that crosses two platforms of different lengths in given times. -/
theorem train_length 
  (platform1_length : ℝ) 
  (platform1_time : ℝ) 
  (platform2_length : ℝ) 
  (platform2_time : ℝ) 
  (h1 : platform1_length = 170)
  (h2 : platform1_time = 15)
  (h3 : platform2_length = 250)
  (h4 : platform2_time = 20) :
  ∃ (train_length : ℝ), 
    (train_length + platform1_length) / platform1_time = 
    (train_length + platform2_length) / platform2_time ∧ 
    train_length = 70 := by
sorry


end NUMINAMATH_CALUDE_train_length_l3203_320361


namespace NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_one_l3203_320371

/-- A system of linear equations parameterized by n -/
def LinearSystem (n : ℝ) :=
  ∃ (x y z : ℝ), (n * x + y = 1) ∧ ((1/2) * n * y + z = 1) ∧ (x + (1/2) * n * z = 2)

/-- The theorem stating that the system has no solution if and only if n = -1 -/
theorem no_solution_iff_n_eq_neg_one :
  ∀ n : ℝ, ¬(LinearSystem n) ↔ n = -1 := by sorry

end NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_one_l3203_320371


namespace NUMINAMATH_CALUDE_sally_pens_ratio_l3203_320364

def sally_pens_problem (initial_pens : ℕ) (num_students : ℕ) (pens_per_student : ℕ) (pens_taken_home : ℕ) : Prop :=
  let pens_distributed := num_students * pens_per_student
  let pens_remaining := initial_pens - pens_distributed
  let pens_in_locker := pens_remaining - pens_taken_home
  pens_in_locker = pens_taken_home

theorem sally_pens_ratio : sally_pens_problem 342 44 7 17 := by
  sorry

end NUMINAMATH_CALUDE_sally_pens_ratio_l3203_320364


namespace NUMINAMATH_CALUDE_infinite_solutions_exist_l3203_320382

theorem infinite_solutions_exist : 
  ∀ n : ℕ, n > 0 → 
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 
  x = (1 + 1 / n) * y ∧
  (3 * x^3 + x * y^2) * (x^2 * y + 3 * y^3) = (x - y)^7 :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_exist_l3203_320382


namespace NUMINAMATH_CALUDE_problem_statement_l3203_320310

theorem problem_statement (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_eq : a^2 + b^2 = a + b) : 
  ((a + b)^2 ≤ 2*(a^2 + b^2)) ∧ ((a + 1)*(b + 1) ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3203_320310


namespace NUMINAMATH_CALUDE_village_population_original_inhabitants_l3203_320387

theorem village_population (final_population : ℕ) : ℕ :=
  let initial_reduction := 0.9
  let secondary_reduction := 0.75
  let total_reduction := initial_reduction * secondary_reduction
  (final_population : ℝ) / total_reduction
    |> round
    |> Int.toNat

/-- The original number of inhabitants in a village, given the final population after two reductions -/
theorem original_inhabitants : village_population 5265 = 7800 := by
  sorry

end NUMINAMATH_CALUDE_village_population_original_inhabitants_l3203_320387


namespace NUMINAMATH_CALUDE_rectangle_shading_convergence_l3203_320368

theorem rectangle_shading_convergence :
  let initial_shaded : ℚ := 1/2
  let subsequent_shading_ratio : ℚ := 1/16
  let shaded_series : ℕ → ℚ := λ n => initial_shaded * subsequent_shading_ratio^n
  let total_shaded : ℚ := ∑' n, shaded_series n
  total_shaded = 17/30 := by
sorry

end NUMINAMATH_CALUDE_rectangle_shading_convergence_l3203_320368


namespace NUMINAMATH_CALUDE_rice_distribution_l3203_320344

theorem rice_distribution (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) : 
  total_weight = 29 / 4 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight * ounces_per_pound / num_containers : ℚ) = 29 := by
  sorry

end NUMINAMATH_CALUDE_rice_distribution_l3203_320344


namespace NUMINAMATH_CALUDE_square_plot_area_l3203_320330

/-- Given a square plot with a fence, prove that the area is 289 square feet
    when the price per foot is 59 and the total cost is 4012. -/
theorem square_plot_area (side_length : ℝ) (perimeter : ℝ) (price_per_foot : ℝ) (total_cost : ℝ) :
  price_per_foot = 59 →
  total_cost = 4012 →
  perimeter = 4 * side_length →
  total_cost = perimeter * price_per_foot →
  side_length ^ 2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_square_plot_area_l3203_320330


namespace NUMINAMATH_CALUDE_tiffany_cans_l3203_320345

theorem tiffany_cans (monday_bags : ℕ) (next_day_bags : ℕ) 
  (h1 : monday_bags = 8) 
  (h2 : monday_bags = next_day_bags + 1) : 
  next_day_bags = 7 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_cans_l3203_320345


namespace NUMINAMATH_CALUDE_min_sum_given_product_l3203_320388

theorem min_sum_given_product (a b : ℝ) : 
  a > 0 → b > 0 → a * b = a + b + 1 → a + b ≥ 2 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l3203_320388


namespace NUMINAMATH_CALUDE_same_color_probability_l3203_320385

def total_balls : ℕ := 8 + 5 + 3

def prob_blue : ℚ := 8 / total_balls
def prob_green : ℚ := 5 / total_balls
def prob_red : ℚ := 3 / total_balls

theorem same_color_probability : 
  prob_blue * prob_blue + prob_green * prob_green + prob_red * prob_red = 49 / 128 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l3203_320385


namespace NUMINAMATH_CALUDE_mortgage_loan_amount_l3203_320362

/-- The mortgage loan problem -/
theorem mortgage_loan_amount 
  (initial_payment : ℝ) 
  (loan_percentage : ℝ) 
  (h1 : initial_payment = 2000000)
  (h2 : loan_percentage = 0.75) : 
  ∃ (total_cost : ℝ), 
    total_cost = initial_payment + loan_percentage * total_cost ∧ 
    loan_percentage * total_cost = 6000000 :=
by sorry

end NUMINAMATH_CALUDE_mortgage_loan_amount_l3203_320362


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l3203_320307

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ  -- length of equal sides
  base : ℕ  -- length of the base
  is_isosceles : leg > base / 2

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 2 : ℝ) * Real.sqrt ((t.leg : ℝ)^2 - (t.base / 2 : ℝ)^2)

/-- Theorem: The minimum possible common perimeter of two noncongruent
    integer-sided isosceles triangles with the same area and a base ratio of 5:4 is 840 -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    area t1 = area t2 ∧
    5 * t1.base = 4 * t2.base ∧
    perimeter t1 = perimeter t2 ∧
    perimeter t1 = 840 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      area s1 = area s2 →
      5 * s1.base = 4 * s2.base →
      perimeter s1 = perimeter s2 →
      perimeter s1 ≥ 840) := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l3203_320307


namespace NUMINAMATH_CALUDE_sin_alpha_minus_pi_third_l3203_320348

theorem sin_alpha_minus_pi_third (α : Real) 
  (h1 : -π/2 < α) (h2 : α < 0) (h3 : 2 * Real.tan α * Real.sin α = 3) : 
  Real.sin (α - π/3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_pi_third_l3203_320348


namespace NUMINAMATH_CALUDE_range_of_m_l3203_320342

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → m ∈ Set.Icc (-3) 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3203_320342


namespace NUMINAMATH_CALUDE_prime_squared_minus_five_not_divisible_by_eight_l3203_320397

theorem prime_squared_minus_five_not_divisible_by_eight (p : ℕ) 
  (h_prime : Nat.Prime p) (h_gt_two : p > 2) : ¬ (8 ∣ (p^2 - 5)) := by
  sorry

end NUMINAMATH_CALUDE_prime_squared_minus_five_not_divisible_by_eight_l3203_320397


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3203_320328

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_of_M_and_N : M ∪ N = {0, 1, 3, 9} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3203_320328


namespace NUMINAMATH_CALUDE_inscribed_circle_area_l3203_320327

/-- The area of a circle inscribed in an equilateral triangle with side length 24 cm is 48π cm². -/
theorem inscribed_circle_area (s : ℝ) (h : s = 24) : 
  let r := s * Real.sqrt 3 / 6
  π * r^2 = 48 * π := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_l3203_320327
