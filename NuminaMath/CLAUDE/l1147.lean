import Mathlib

namespace yard_to_stride_l1147_114772

-- Define the units of measurement
variable (step stride leap yard : ℚ)

-- Define the relationships between units
axiom step_stride_relation : 3 * step = 4 * stride
axiom leap_step_relation : 5 * leap = 2 * step
axiom leap_yard_relation : 7 * leap = 6 * yard

-- Theorem to prove
theorem yard_to_stride : yard = 28/45 * stride := by
  sorry

end yard_to_stride_l1147_114772


namespace sin_difference_l1147_114790

theorem sin_difference (A B : Real) 
  (h1 : Real.tan A = 2 * Real.tan B) 
  (h2 : Real.sin (A + B) = 1/4) : 
  Real.sin (A - B) = 1/12 := by
sorry

end sin_difference_l1147_114790


namespace henrys_initial_income_l1147_114761

theorem henrys_initial_income (initial_income : ℝ) : 
  initial_income * 1.5 = 180 → initial_income = 120 := by
  sorry

end henrys_initial_income_l1147_114761


namespace parallelogram_d_not_two_neg_two_l1147_114732

/-- Definition of a point in 2D space -/
def Point := ℝ × ℝ

/-- Definition of a parallelogram -/
def is_parallelogram (A B C D : Point) : Prop :=
  (A.1 + C.1 = B.1 + D.1) ∧ (A.2 + C.2 = B.2 + D.2)

/-- Theorem: If ABCD is a parallelogram with A(0,0), B(2,2), C(3,0), then D cannot be (2,-2) -/
theorem parallelogram_d_not_two_neg_two :
  let A : Point := (0, 0)
  let B : Point := (2, 2)
  let C : Point := (3, 0)
  let D : Point := (2, -2)
  ¬(is_parallelogram A B C D) := by
  sorry


end parallelogram_d_not_two_neg_two_l1147_114732


namespace number_difference_l1147_114775

theorem number_difference (L S : ℕ) (h1 : L = 1614) (h2 : L = 6 * S + 15) : L - S = 1348 := by
  sorry

end number_difference_l1147_114775


namespace subtract_negative_three_and_one_l1147_114702

theorem subtract_negative_three_and_one : -3 - 1 = -4 := by sorry

end subtract_negative_three_and_one_l1147_114702


namespace students_in_two_classes_l1147_114725

theorem students_in_two_classes
  (total_students : ℕ)
  (history : ℕ)
  (math : ℕ)
  (english : ℕ)
  (science : ℕ)
  (geography : ℕ)
  (all_five : ℕ)
  (history_and_math : ℕ)
  (english_and_science : ℕ)
  (math_and_geography : ℕ)
  (h_total : total_students = 500)
  (h_history : history = 120)
  (h_math : math = 105)
  (h_english : english = 145)
  (h_science : science = 133)
  (h_geography : geography = 107)
  (h_all_five : all_five = 15)
  (h_history_and_math : history_and_math = 40)
  (h_english_and_science : english_and_science = 35)
  (h_math_and_geography : math_and_geography = 25)
  (h_at_least_one : total_students ≤ history + math + english + science + geography) :
  (history_and_math - all_five) + (english_and_science - all_five) + (math_and_geography - all_five) = 55 := by
  sorry

end students_in_two_classes_l1147_114725


namespace vector_arrangement_exists_l1147_114752

theorem vector_arrangement_exists : ∃ (a b c : ℝ × ℝ),
  (‖a + b‖ = 1) ∧
  (‖b + c‖ = 1) ∧
  (‖c + a‖ = 1) ∧
  (a + b + c = (0, 0)) := by
  sorry

end vector_arrangement_exists_l1147_114752


namespace exp_ln_eight_l1147_114703

theorem exp_ln_eight : Real.exp (Real.log 8) = 8 := by sorry

end exp_ln_eight_l1147_114703


namespace book_selection_theorem_l1147_114750

theorem book_selection_theorem (n m k : ℕ) (h1 : n = 8) (h2 : m = 5) (h3 : k = 2) :
  (Nat.choose (n - k) (m - k)) = (Nat.choose 6 3) :=
sorry

end book_selection_theorem_l1147_114750


namespace cos_75_cos_15_minus_sin_435_sin_15_eq_zero_l1147_114798

theorem cos_75_cos_15_minus_sin_435_sin_15_eq_zero :
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) - 
  Real.sin (435 * π / 180) * Real.sin (15 * π / 180) = 0 := by
  sorry

end cos_75_cos_15_minus_sin_435_sin_15_eq_zero_l1147_114798


namespace power_multiplication_l1147_114731

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end power_multiplication_l1147_114731


namespace fib_identity_fib_1094_1096_minus_1095_squared_l1147_114709

/-- The Fibonacci sequence -/
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The identity for Fibonacci numbers -/
theorem fib_identity (n : ℕ) :
  fib (n + 2) * fib n - fib (n + 1)^2 = (-1)^(n + 1) := by sorry

/-- The main theorem to prove -/
theorem fib_1094_1096_minus_1095_squared :
  fib 1094 * fib 1096 - fib 1095^2 = -1 := by sorry

end fib_identity_fib_1094_1096_minus_1095_squared_l1147_114709


namespace regression_line_equation_l1147_114743

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  point : ℝ × ℝ

/-- Checks if a given equation represents the regression line -/
def is_regression_line_equation (line : RegressionLine) (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = line.slope * (x - line.point.1) + line.point.2) ∧
  (f line.point.1 = line.point.2)

/-- The theorem stating the equation of the specific regression line -/
theorem regression_line_equation (line : RegressionLine) 
  (h1 : line.slope = 6.5)
  (h2 : line.point = (2, 3)) :
  is_regression_line_equation line (λ x => -10 + 6.5 * x) := by
  sorry

end regression_line_equation_l1147_114743


namespace inconsistent_game_statistics_l1147_114760

theorem inconsistent_game_statistics :
  ∀ (total_games : ℕ) (first_part_games : ℕ) (win_percentage : ℚ),
  total_games = 75 →
  first_part_games = 100 →
  (0 : ℚ) ≤ win_percentage ∧ win_percentage ≤ 1 →
  ¬(∃ (first_part_win_percentage : ℚ) (remaining_win_percentage : ℚ),
    first_part_win_percentage * (first_part_games : ℚ) / (total_games : ℚ) +
    remaining_win_percentage * ((total_games - first_part_games) : ℚ) / (total_games : ℚ) = win_percentage ∧
    remaining_win_percentage = 1/2 ∧
    (0 : ℚ) ≤ first_part_win_percentage ∧ first_part_win_percentage ≤ 1) :=
by
  sorry


end inconsistent_game_statistics_l1147_114760


namespace sum_first_20_triangular_numbers_l1147_114767

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the first n triangular numbers -/
def sum_triangular_numbers (n : ℕ) : ℕ :=
  (List.range n).map triangular_number |>.sum

/-- Theorem: The sum of the first 20 triangular numbers is 1540 -/
theorem sum_first_20_triangular_numbers :
  sum_triangular_numbers 20 = 1540 := by
  sorry

end sum_first_20_triangular_numbers_l1147_114767


namespace sum_of_fractions_l1147_114757

theorem sum_of_fractions : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + 
  (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) = 5 / 14 := by
  sorry

end sum_of_fractions_l1147_114757


namespace gold_bars_lost_l1147_114791

theorem gold_bars_lost (initial_bars : ℕ) (num_friends : ℕ) (bars_per_friend : ℕ) : 
  initial_bars = 100 →
  num_friends = 4 →
  bars_per_friend = 20 →
  initial_bars - (num_friends * bars_per_friend) = 20 := by
  sorry

end gold_bars_lost_l1147_114791


namespace book_pages_theorem_l1147_114762

theorem book_pages_theorem (P : ℕ) 
  (h1 : P / 2 + P / 4 + P / 6 + 20 = P) : P = 240 := by
  sorry

end book_pages_theorem_l1147_114762


namespace remainder_polynomial_division_l1147_114741

theorem remainder_polynomial_division (x : ℝ) : 
  let g (x : ℝ) := x^5 + x^4 + x^3 + x^2 + x + 1
  (g (x^12)) % (g x) = 6 := by sorry

end remainder_polynomial_division_l1147_114741


namespace min_omega_value_l1147_114733

theorem min_omega_value (ω : ℕ+) : 
  (∀ k : ℕ+, 2 * Real.sin (2 * Real.pi * ↑k + Real.pi / 3) = Real.sqrt 3 → ω ≤ k) →
  2 * Real.sin (2 * Real.pi * ↑ω + Real.pi / 3) = Real.sqrt 3 →
  ω = 1 := by sorry

end min_omega_value_l1147_114733


namespace fraction_simplification_l1147_114769

theorem fraction_simplification :
  (270 : ℚ) / 18 * 7 / 140 * 9 / 4 = 27 / 16 := by
  sorry

end fraction_simplification_l1147_114769


namespace smallest_overlap_percentage_l1147_114756

theorem smallest_overlap_percentage (coffee_drinkers tea_drinkers : ℝ) 
  (h1 : coffee_drinkers = 75)
  (h2 : tea_drinkers = 80) :
  coffee_drinkers + tea_drinkers - 100 = 55 :=
by sorry

end smallest_overlap_percentage_l1147_114756


namespace book_arrangement_count_l1147_114727

/-- The number of ways to arrange math and history books -/
def arrange_books (math_books : ℕ) (history_books : ℕ) : ℕ :=
  let math_groupings := (math_books + 2 - 1).choose 2
  let math_permutations := Nat.factorial math_books
  let history_placements := history_books.choose 3 * history_books.choose 3
  math_groupings * math_permutations * history_placements

/-- Theorem stating the number of arrangements for 4 math books and 6 history books -/
theorem book_arrangement_count :
  arrange_books 4 6 = 96000 := by
  sorry

end book_arrangement_count_l1147_114727


namespace average_waiting_time_is_nineteen_sixths_l1147_114778

/-- Represents a bus schedule with a given departure interval -/
structure BusSchedule where
  interval : ℕ

/-- Calculates the average waiting time for a set of bus schedules -/
def averageWaitingTime (schedules : List BusSchedule) : ℚ :=
  sorry

/-- Theorem stating that the average waiting time for the given bus schedules is 19/6 minutes -/
theorem average_waiting_time_is_nineteen_sixths :
  let schedules := [
    BusSchedule.mk 10,  -- Bus A
    BusSchedule.mk 12,  -- Bus B
    BusSchedule.mk 15   -- Bus C
  ]
  averageWaitingTime schedules = 19 / 6 := by
  sorry

end average_waiting_time_is_nineteen_sixths_l1147_114778


namespace intersection_distance_l1147_114792

/-- The distance between the intersection points of the line y = x and the circle (x-2)^2 + (y-1)^2 = 1 is √2. -/
theorem intersection_distance :
  ∃ (P Q : ℝ × ℝ),
    (P.1 = P.2 ∧ (P.1 - 2)^2 + (P.2 - 1)^2 = 1) ∧
    (Q.1 = Q.2 ∧ (Q.1 - 2)^2 + (Q.2 - 1)^2 = 1) ∧
    P ≠ Q ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = Real.sqrt 2 := by
  sorry

end intersection_distance_l1147_114792


namespace divisibility_rule_37_l1147_114797

/-- Given a positive integer n, returns a list of its three-digit segments from right to left -/
def threeDigitSegments (n : ℕ+) : List ℕ :=
  sorry

/-- The divisibility rule for 37 -/
theorem divisibility_rule_37 (n : ℕ+) : 
  37 ∣ n ↔ 37 ∣ (threeDigitSegments n).sum :=
sorry

end divisibility_rule_37_l1147_114797


namespace power_three_mod_thirteen_l1147_114779

theorem power_three_mod_thirteen : 3^21 % 13 = 1 := by
  sorry

end power_three_mod_thirteen_l1147_114779


namespace gcd_factorial_problem_l1147_114758

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 5)) = 2520 := by
  sorry

end gcd_factorial_problem_l1147_114758


namespace part_one_part_two_l1147_114706

-- Define proposition p
def p (k : ℝ) : Prop := ∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 + 2*x - k ≤ 0

-- Define proposition q
def q (k : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*k*x + 3*k + 4 = 0

-- Theorem for part 1
theorem part_one (k : ℝ) : p k → k ∈ Set.Ici 3 := by sorry

-- Theorem for part 2
theorem part_two (k : ℝ) : 
  (p k ∧ ¬q k) ∨ (¬p k ∧ q k) → k ∈ Set.Iic (-1) ∪ Set.Ico 3 4 := by sorry

end part_one_part_two_l1147_114706


namespace greatest_of_three_consecutive_integers_l1147_114764

theorem greatest_of_three_consecutive_integers (x y z : ℤ) : 
  (y = x + 1) → (z = y + 1) → (x + y + z = 33) → (max x (max y z) = 12) := by
  sorry

end greatest_of_three_consecutive_integers_l1147_114764


namespace average_daily_low_temperature_l1147_114782

theorem average_daily_low_temperature (temperatures : List ℝ) : 
  temperatures = [40, 47, 45, 41, 39] → 
  (temperatures.sum / temperatures.length : ℝ) = 42.4 := by
  sorry

end average_daily_low_temperature_l1147_114782


namespace inequality_proof_l1147_114744

theorem inequality_proof (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hxy : x * y < 0) :
  (b / a + a / b ≥ 2) ∧ (x / y + y / x ≤ -2) := by sorry

end inequality_proof_l1147_114744


namespace bisector_line_equation_chord_length_at_pi_over_4_l1147_114736

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 24 + y^2 / 12 = 1

-- Define point M
def M : ℝ × ℝ := (3, 1)

-- Define a line passing through M
def line_through_M (m : ℝ) (x y : ℝ) : Prop :=
  y - M.2 = m * (x - M.1)

-- Define the intersection points of a line with the ellipse
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ line_through_M m p.1 p.2}

-- Part I: Equation of line when M bisects AB
theorem bisector_line_equation :
  ∃ (A B : ℝ × ℝ) (m : ℝ),
    A ∈ intersection_points m ∧
    B ∈ intersection_points m ∧
    M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
    line_through_M (-3/2) M.1 M.2 :=
sorry

-- Part II: Length of AB when angle of inclination is π/4
theorem chord_length_at_pi_over_4 :
  ∃ (A B : ℝ × ℝ),
    A ∈ intersection_points 1 ∧
    B ∈ intersection_points 1 →
    ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2) = 16/(3*2^(1/2)) :=
sorry

end bisector_line_equation_chord_length_at_pi_over_4_l1147_114736


namespace second_digit_of_n_l1147_114737

theorem second_digit_of_n (n : ℕ) : 
  (10^99 ≤ 8*n ∧ 8*n < 10^100) ∧ 
  (10^101 ≤ 81*n - 102 ∧ 81*n - 102 < 10^102) →
  ∃ k : ℕ, 12 * 10^97 ≤ n ∧ n < 13 * 10^97 ∧ n = 2 * 10^97 + k :=
by sorry

end second_digit_of_n_l1147_114737


namespace robin_additional_cupcakes_l1147_114795

/-- Calculates the number of additional cupcakes made given the initial number,
    the number sold, and the final number of cupcakes. -/
def additional_cupcakes (initial : ℕ) (sold : ℕ) (final : ℕ) : ℕ :=
  final - (initial - sold)

/-- Proves that Robin made 39 additional cupcakes given the problem conditions. -/
theorem robin_additional_cupcakes :
  additional_cupcakes 42 22 59 = 39 := by
  sorry

end robin_additional_cupcakes_l1147_114795


namespace cost_23_days_l1147_114796

/-- Calculate the cost of staying in a student youth hostel for a given number of days. -/
def hostelCost (days : ℕ) : ℚ :=
  let firstWeekRate : ℚ := 18
  let additionalWeekRate : ℚ := 12
  let firstWeekDays : ℕ := min 7 days
  let additionalDays : ℕ := days - firstWeekDays
  let firstWeekCost : ℚ := firstWeekRate * firstWeekDays
  let additionalCost : ℚ := additionalWeekRate * additionalDays
  firstWeekCost + additionalCost

/-- Theorem stating that the cost for a 23-day stay is $318.00 -/
theorem cost_23_days :
  hostelCost 23 = 318 := by
  sorry

#eval hostelCost 23

end cost_23_days_l1147_114796


namespace apples_per_box_l1147_114786

/-- Given the following conditions:
    - There are 180 apples in each crate
    - 12 crates of apples were delivered
    - 160 apples were rotten and thrown away
    - The remaining apples were packed into 100 boxes
    Prove that there are 20 apples in each box -/
theorem apples_per_box :
  ∀ (apples_per_crate crates_delivered rotten_apples total_boxes : ℕ),
    apples_per_crate = 180 →
    crates_delivered = 12 →
    rotten_apples = 160 →
    total_boxes = 100 →
    (apples_per_crate * crates_delivered - rotten_apples) / total_boxes = 20 := by
  sorry

end apples_per_box_l1147_114786


namespace y₁_less_than_y₂_l1147_114726

/-- A linear function f(x) = -4x + 3 -/
def f (x : ℝ) : ℝ := -4 * x + 3

/-- Point P₁ is on the graph of f -/
def P₁_on_graph (y₁ : ℝ) : Prop := f 1 = y₁

/-- Point P₂ is on the graph of f -/
def P₂_on_graph (y₂ : ℝ) : Prop := f (-3) = y₂

/-- Theorem stating the relationship between y₁ and y₂ -/
theorem y₁_less_than_y₂ (y₁ y₂ : ℝ) (h₁ : P₁_on_graph y₁) (h₂ : P₂_on_graph y₂) : y₁ < y₂ := by
  sorry

end y₁_less_than_y₂_l1147_114726


namespace oscar_coco_difference_l1147_114722

/-- The number of strides Coco takes between consecutive poles -/
def coco_strides : ℕ := 22

/-- The number of leaps Oscar takes between consecutive poles -/
def oscar_leaps : ℕ := 6

/-- The number of strides Elmer takes between consecutive poles -/
def elmer_strides : ℕ := 11

/-- The number of poles -/
def num_poles : ℕ := 31

/-- The total distance in feet between the first and last pole -/
def total_distance : ℕ := 7920

/-- The length of Coco's stride in feet -/
def coco_stride_length : ℚ := total_distance / (coco_strides * (num_poles - 1))

/-- The length of Oscar's leap in feet -/
def oscar_leap_length : ℚ := total_distance / (oscar_leaps * (num_poles - 1))

theorem oscar_coco_difference :
  oscar_leap_length - coco_stride_length = 32 := by sorry

end oscar_coco_difference_l1147_114722


namespace intersection_of_A_and_B_l1147_114715

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x^2 - 1)}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = x^2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x ≥ 1} := by
  sorry

end intersection_of_A_and_B_l1147_114715


namespace smallest_number_l1147_114712

-- Define the numbers in their respective bases
def binary_num : ℕ := 63  -- 111111₍₂₎
def base_6_num : ℕ := 66  -- 150₍₆₎
def base_4_num : ℕ := 64  -- 1000₍₄₎
def octal_num : ℕ := 65   -- 101₍₈₎

-- Theorem statement
theorem smallest_number :
  binary_num < base_6_num ∧ 
  binary_num < base_4_num ∧ 
  binary_num < octal_num :=
by sorry

end smallest_number_l1147_114712


namespace parabola_symmetry_l1147_114751

/-- Represents a quadratic function of the form ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a quadratic function vertically by a given amount -/
def shift_vertical (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, b := f.b, c := f.c + shift }

/-- Checks if two quadratic functions are symmetric about the y-axis -/
def symmetric_about_y_axis (f g : QuadraticFunction) : Prop :=
  f.a = g.a ∧ f.b = -g.b ∧ f.c = g.c

theorem parabola_symmetry (a b : ℝ) (h_a : a ≠ 0) :
  let f : QuadraticFunction := { a := a, b := b, c := -2 }
  let g : QuadraticFunction := { a := 1/2, b := 1, c := -4 }
  symmetric_about_y_axis (shift_vertical f (-2)) g →
  a = 1/2 ∧ b = -1 := by
  sorry

end parabola_symmetry_l1147_114751


namespace total_cats_l1147_114780

/-- The number of cats that can jump -/
def jump : ℕ := 60

/-- The number of cats that can fetch -/
def fetch : ℕ := 35

/-- The number of cats that can meow on command -/
def meow : ℕ := 40

/-- The number of cats that can jump and fetch -/
def jump_fetch : ℕ := 20

/-- The number of cats that can fetch and meow -/
def fetch_meow : ℕ := 15

/-- The number of cats that can jump and meow -/
def jump_meow : ℕ := 25

/-- The number of cats that can do all three tricks -/
def all_three : ℕ := 11

/-- The number of cats that can do none of the tricks -/
def no_tricks : ℕ := 10

/-- Theorem stating the total number of cats in the training center -/
theorem total_cats : 
  jump + fetch + meow - jump_fetch - fetch_meow - jump_meow + all_three + no_tricks = 96 := by
  sorry

end total_cats_l1147_114780


namespace percentage_of_360_equals_108_l1147_114759

theorem percentage_of_360_equals_108 : 
  ∃ (p : ℝ), p * 360 / 100 = 108.0 ∧ p = 30 := by sorry

end percentage_of_360_equals_108_l1147_114759


namespace arithmetic_sequence_property_l1147_114766

/-- An arithmetic sequence with the given property -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  property : a 2 + 4 * a 7 + a 12 = 96
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  2 * seq.a 3 + seq.a 15 = 48 := by
  sorry

end arithmetic_sequence_property_l1147_114766


namespace goods_train_length_goods_train_length_approx_280m_l1147_114730

/-- The length of a goods train passing a passenger train in opposite directions --/
theorem goods_train_length (v_passenger : ℝ) (v_goods : ℝ) (t_pass : ℝ) : ℝ :=
  let v_relative : ℝ := v_passenger + v_goods
  let v_relative_ms : ℝ := v_relative * 1000 / 3600
  let length : ℝ := v_relative_ms * t_pass
  by
    -- Proof goes here
    sorry

/-- The length of the goods train is approximately 280 meters --/
theorem goods_train_length_approx_280m :
  ∃ ε > 0, |goods_train_length 70 42 9 - 280| < ε :=
by
  -- Proof goes here
  sorry

end goods_train_length_goods_train_length_approx_280m_l1147_114730


namespace min_value_squared_distance_l1147_114777

theorem min_value_squared_distance (a b c d : ℝ) 
  (h : |b - (Real.log a) / a| + |c - d + 2| = 0) : 
  ∃ (min : ℝ), min = (9 : ℝ) / 2 ∧ 
  ∀ (x y : ℝ), (x - y)^2 + (b - d)^2 ≥ min :=
sorry

end min_value_squared_distance_l1147_114777


namespace average_weight_increase_l1147_114753

theorem average_weight_increase (initial_count : ℕ) (initial_weight : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 65 →
  new_weight = 89 →
  (initial_count * initial_weight - old_weight + new_weight) / initial_count - initial_weight = 3 :=
by sorry

end average_weight_increase_l1147_114753


namespace plane_speed_against_wind_l1147_114735

/-- Calculates the ground speed of a plane flying against a tailwind, given its ground speed with the tailwind and the wind speed. -/
def ground_speed_against_wind (ground_speed_with_wind wind_speed : ℝ) : ℝ :=
  2 * ground_speed_with_wind - 2 * wind_speed - wind_speed

/-- Theorem stating that a plane with a ground speed of 460 mph with a 75 mph tailwind
    will have a ground speed of 310 mph against the same tailwind. -/
theorem plane_speed_against_wind :
  ground_speed_against_wind 460 75 = 310 := by
  sorry

end plane_speed_against_wind_l1147_114735


namespace ferris_wheel_line_l1147_114721

theorem ferris_wheel_line (capacity : ℕ) (not_riding : ℕ) (total : ℕ) : 
  capacity = 56 → not_riding = 36 → total = capacity + not_riding → total = 92 := by
  sorry

end ferris_wheel_line_l1147_114721


namespace count_valid_assignments_five_l1147_114749

/-- Represents a valid assignment of students to tests -/
def ValidAssignment (n : ℕ) := Fin n → Fin n → Prop

/-- The number of valid assignments for n students and n tests -/
def CountValidAssignments (n : ℕ) : ℕ := sorry

/-- The condition that each student takes exactly 2 distinct tests -/
def StudentTakesTwoTests (assignment : ValidAssignment 5) : Prop :=
  ∀ s : Fin 5, ∃! t1 t2 : Fin 5, t1 ≠ t2 ∧ assignment s t1 ∧ assignment s t2

/-- The condition that each test is taken by exactly 2 students -/
def TestTakenByTwoStudents (assignment : ValidAssignment 5) : Prop :=
  ∀ t : Fin 5, ∃! s1 s2 : Fin 5, s1 ≠ s2 ∧ assignment s1 t ∧ assignment s2 t

theorem count_valid_assignments_five :
  (∀ assignment : ValidAssignment 5,
    StudentTakesTwoTests assignment ∧ TestTakenByTwoStudents assignment) →
  CountValidAssignments 5 = 2040 := by
  sorry

end count_valid_assignments_five_l1147_114749


namespace min_scabs_per_day_l1147_114740

def total_scabs : ℕ := 220
def days_in_week : ℕ := 7

theorem min_scabs_per_day :
  ∃ (n : ℕ), n * days_in_week ≥ total_scabs ∧
  ∀ (m : ℕ), m * days_in_week ≥ total_scabs → m ≥ n :=
by sorry

end min_scabs_per_day_l1147_114740


namespace f_max_and_g_dominance_l1147_114720

open Real

noncomputable def f (x : ℝ) : ℝ := log x - 2 * x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1/2) * m * x^2 + (m - 3) * x - 1

theorem f_max_and_g_dominance :
  (∃ (c : ℝ), c = -log 2 - 1 ∧ ∀ x > 0, f x ≤ c) ∧
  (∀ m : ℤ, (∀ x > 0, f x ≤ g m x) → m ≥ 2) ∧
  (∀ x > 0, f x ≤ g 2 x) :=
sorry

end f_max_and_g_dominance_l1147_114720


namespace number_of_benches_l1147_114724

/-- Converts a base 6 number to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- The number of people that can be seated in the shop -/
def totalSeats : ℕ := base6ToBase10 204

/-- The number of people that sit on one bench -/
def peoplePerBench : ℕ := 2

/-- Theorem: The number of benches in the shop is 38 -/
theorem number_of_benches :
  totalSeats / peoplePerBench = 38 := by
  sorry

end number_of_benches_l1147_114724


namespace quadratic_equation_solution_l1147_114704

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end quadratic_equation_solution_l1147_114704


namespace complex_modulus_problem_l1147_114746

theorem complex_modulus_problem (b : ℝ) (z : ℂ) : 
  z = (b * Complex.I) / (4 + 3 * Complex.I) → 
  Complex.abs z = 5 → 
  b = 25 ∨ b = -25 := by
sorry

end complex_modulus_problem_l1147_114746


namespace tank_capacity_l1147_114723

theorem tank_capacity : 
  ∀ (C : ℝ),
    (C / 6 + C / 12 = (2.5 * 60 + 1.5 * 60) * 8) →
    C = 640 :=
by
  sorry

end tank_capacity_l1147_114723


namespace inequality_chain_l1147_114787

theorem inequality_chain (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (2 : ℝ) / ((1 / a) + (1 / b)) < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 := by
  sorry

end inequality_chain_l1147_114787


namespace larger_number_proof_l1147_114738

theorem larger_number_proof (x y : ℝ) (sum_eq : x + y = 40) (diff_eq : x - y = 10) :
  max x y = 25 := by
  sorry

end larger_number_proof_l1147_114738


namespace fish_cost_l1147_114710

/-- Given that 530 pesos can buy 4 kg of fish and 2 kg of pork,
    and 875 pesos can buy 7 kg of fish and 3 kg of pork,
    prove that the cost of 1 kg of fish is 80 pesos. -/
theorem fish_cost (fish_price pork_price : ℝ) 
  (h1 : 4 * fish_price + 2 * pork_price = 530)
  (h2 : 7 * fish_price + 3 * pork_price = 875) : 
  fish_price = 80 := by
  sorry

end fish_cost_l1147_114710


namespace axis_of_symmetry_sine_curve_l1147_114747

/-- The axis of symmetry for the sine curve y = sin(2πx - π/3) is x = 5/12 -/
theorem axis_of_symmetry_sine_curve (x : ℝ) : 
  (∃ (k : ℤ), x = k / 2 + 5 / 12) ↔ 
  (∃ (n : ℤ), 2 * π * x - π / 3 = n * π + π / 2) :=
sorry

end axis_of_symmetry_sine_curve_l1147_114747


namespace expression_factorization_l1147_114793

theorem expression_factorization (x y z : ℤ) :
  x^2 - (y + z)^2 + 2*x + y - z = (x - y - z) * (x + 2*y + 2) := by
  sorry

end expression_factorization_l1147_114793


namespace lucia_dance_cost_l1147_114713

/-- Represents the cost of dance classes for a week -/
structure DanceClassesCost where
  hip_hop_classes : Nat
  ballet_classes : Nat
  jazz_classes : Nat
  hip_hop_cost : Nat
  ballet_cost : Nat
  jazz_cost : Nat

/-- Calculates the total cost of dance classes for a week -/
def total_cost (c : DanceClassesCost) : Nat :=
  c.hip_hop_classes * c.hip_hop_cost +
  c.ballet_classes * c.ballet_cost +
  c.jazz_classes * c.jazz_cost

/-- Theorem stating that Lucia's total dance class cost for a week is $52 -/
theorem lucia_dance_cost :
  let c : DanceClassesCost := {
    hip_hop_classes := 2,
    ballet_classes := 2,
    jazz_classes := 1,
    hip_hop_cost := 10,
    ballet_cost := 12,
    jazz_cost := 8
  }
  total_cost c = 52 := by
  sorry


end lucia_dance_cost_l1147_114713


namespace triangle_side_length_l1147_114763

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →
  (B = Real.pi / 3) →
  (a^2 + c^2 = 3 * a * c) →
  (b = 2 * Real.sqrt 2) := by
  sorry

end triangle_side_length_l1147_114763


namespace consecutive_shots_count_l1147_114700

/-- The number of ways to arrange 3 successful shots out of 8 attempts, 
    with exactly 2 consecutive successful shots. -/
def consecutiveShots : ℕ := 30

/-- The total number of attempts. -/
def totalAttempts : ℕ := 8

/-- The number of successful shots. -/
def successfulShots : ℕ := 3

/-- The number of consecutive successful shots required. -/
def consecutiveHits : ℕ := 2

theorem consecutive_shots_count :
  consecutiveShots = 
    (totalAttempts - successfulShots + 1).choose 2 := by
  sorry

end consecutive_shots_count_l1147_114700


namespace probability_neither_red_nor_purple_l1147_114717

theorem probability_neither_red_nor_purple (total : ℕ) (red : ℕ) (purple : ℕ) 
  (h1 : total = 60) (h2 : red = 5) (h3 : purple = 7) :
  (total - (red + purple)) / total = 4 / 5 := by
  sorry

end probability_neither_red_nor_purple_l1147_114717


namespace floor_width_is_twenty_l1147_114770

/-- Represents a rectangular floor with a rug -/
structure FloorWithRug where
  length : ℝ
  width : ℝ
  strip_width : ℝ
  rug_area : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem floor_width_is_twenty
  (floor : FloorWithRug)
  (h1 : floor.length = 25)
  (h2 : floor.strip_width = 4)
  (h3 : floor.rug_area = 204)
  (h4 : floor.rug_area = (floor.length - 2 * floor.strip_width) * (floor.width - 2 * floor.strip_width)) :
  floor.width = 20 := by
  sorry

#check floor_width_is_twenty

end floor_width_is_twenty_l1147_114770


namespace baseball_average_runs_l1147_114799

/-- Represents the scoring pattern of a baseball team over a series of games -/
structure ScoringPattern where
  games : ℕ
  oneRun : ℕ
  fourRuns : ℕ
  fiveRuns : ℕ

/-- Calculates the average runs per game given a scoring pattern -/
def averageRuns (pattern : ScoringPattern) : ℚ :=
  (pattern.oneRun * 1 + pattern.fourRuns * 4 + pattern.fiveRuns * 5) / pattern.games

/-- Theorem stating that for the given scoring pattern, the average runs per game is 4 -/
theorem baseball_average_runs :
  let pattern : ScoringPattern := {
    games := 6,
    oneRun := 1,
    fourRuns := 2,
    fiveRuns := 3
  }
  averageRuns pattern = 4 := by sorry

end baseball_average_runs_l1147_114799


namespace selection_schemes_count_l1147_114707

def number_of_people : ℕ := 6
def number_of_places : ℕ := 4
def number_of_restricted_people : ℕ := 2
def number_of_restricted_places : ℕ := 1

theorem selection_schemes_count :
  (number_of_people.choose number_of_places) *
  (number_of_places - number_of_restricted_places).choose 1 *
  ((number_of_people - number_of_restricted_people).choose (number_of_places - 1)) *
  (number_of_places - 1).factorial = 240 := by
  sorry

end selection_schemes_count_l1147_114707


namespace power_of_power_l1147_114785

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l1147_114785


namespace sales_tax_difference_specific_sales_tax_difference_l1147_114711

/-- The difference between state and local sales taxes on a discounted sweater --/
theorem sales_tax_difference (original_price : ℝ) (discount_rate : ℝ) 
  (state_tax_rate : ℝ) (local_tax_rate : ℝ) : ℝ :=
by
  -- Define the discounted price
  let discounted_price := original_price * (1 - discount_rate)
  
  -- Calculate state and local taxes
  let state_tax := discounted_price * state_tax_rate
  let local_tax := discounted_price * local_tax_rate
  
  -- Calculate the difference
  exact state_tax - local_tax

/-- The specific case for the given problem --/
theorem specific_sales_tax_difference : 
  sales_tax_difference 50 0.1 0.075 0.07 = 0.225 :=
by
  sorry

end sales_tax_difference_specific_sales_tax_difference_l1147_114711


namespace linear_systems_solutions_l1147_114781

theorem linear_systems_solutions :
  -- System 1
  let system1 (x y : ℚ) := (y = x - 5) ∧ (3 * x - y = 8)
  let solution1 := (3/2, -7/2)
  -- System 2
  let system2 (x y : ℚ) := (3 * x - 2 * y = 1) ∧ (7 * x + 4 * y = 11)
  let solution2 := (1, 1)
  -- Proof statements
  (∃! p : ℚ × ℚ, system1 p.1 p.2 ∧ p = solution1) ∧
  (∃! q : ℚ × ℚ, system2 q.1 q.2 ∧ q = solution2) :=
by
  sorry

end linear_systems_solutions_l1147_114781


namespace six_ring_clock_interval_l1147_114701

/-- A clock that rings a certain number of times per day at equal intervals -/
structure RingingClock where
  rings_per_day : ℕ
  rings_per_day_pos : rings_per_day > 0

/-- The number of minutes in a day -/
def minutes_per_day : ℕ := 24 * 60

/-- Calculate the time interval between rings in minutes -/
def interval_between_rings (clock : RingingClock) : ℚ :=
  minutes_per_day / (clock.rings_per_day - 1)

/-- Theorem: For a clock that rings 6 times a day, the interval between rings is 288 minutes -/
theorem six_ring_clock_interval :
  let clock : RingingClock := ⟨6, by norm_num⟩
  interval_between_rings clock = 288 := by sorry

end six_ring_clock_interval_l1147_114701


namespace fifteenths_in_fraction_l1147_114783

theorem fifteenths_in_fraction : 
  let whole_number : ℚ := 82
  let fraction : ℚ := 3 / 5
  let divisor : ℚ := 1 / 15
  let multiplier : ℕ := 3
  let subtrahend_whole : ℕ := 42
  let subtrahend_fraction : ℚ := 7 / 10
  
  ((whole_number + fraction) / divisor * multiplier) - 
  (subtrahend_whole + subtrahend_fraction) = 3674.3 := by sorry

end fifteenths_in_fraction_l1147_114783


namespace smallest_k_sum_squares_div_180_l1147_114776

/-- Sum of squares from 1 to n -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Predicate for a number being divisible by 180 -/
def divisible_by_180 (n : ℕ) : Prop := ∃ m : ℕ, n = 180 * m

theorem smallest_k_sum_squares_div_180 :
  (∀ k < 216, ¬(divisible_by_180 (sum_of_squares k))) ∧
  (divisible_by_180 (sum_of_squares 216)) := by sorry

end smallest_k_sum_squares_div_180_l1147_114776


namespace equivalent_discount_equivalent_discount_proof_l1147_114784

theorem equivalent_discount : ℝ → Prop :=
  fun x => 
    let first_discount := 0.15
    let second_discount := 0.10
    let third_discount := 0.05
    let price_after_discounts := (1 - first_discount) * (1 - second_discount) * (1 - third_discount) * x
    let equivalent_single_discount := 0.273
    price_after_discounts = (1 - equivalent_single_discount) * x

-- The proof is omitted
theorem equivalent_discount_proof : ∀ x : ℝ, equivalent_discount x :=
  sorry

end equivalent_discount_equivalent_discount_proof_l1147_114784


namespace rectangle_area_l1147_114708

/-- The area of a rectangle with perimeter equal to a triangle with sides 7, 9, and 10,
    and length twice its width, is 338/9 square centimeters. -/
theorem rectangle_area (w : ℝ) (h : 2 * (2 * w + w) = 7 + 9 + 10) :
  2 * w * w = 338 / 9 := by
  sorry

end rectangle_area_l1147_114708


namespace emily_calculation_l1147_114754

theorem emily_calculation (x y z : ℝ) 
  (h1 : 2*x - 3*y + z = 14) 
  (h2 : 2*x - 3*y - z = 6) : 
  2*x - 3*y = 10 := by
sorry

end emily_calculation_l1147_114754


namespace min_distance_to_line_l1147_114718

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x + 2*y + 1 = 0

/-- The line equation -/
def line_equation (a b x y : ℝ) : Prop :=
  a*x + b*y + 1 = 0

/-- The line bisects the circle's circumference -/
def line_bisects_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y → line_equation a b x y

/-- The theorem to be proved -/
theorem min_distance_to_line (a b : ℝ) :
  line_bisects_circle a b →
  (∃ min : ℝ, min = 5 ∧ ∀ a' b' : ℝ, line_bisects_circle a' b' →
    (a' - 2)^2 + (b' - 2)^2 ≥ min) :=
by sorry

end min_distance_to_line_l1147_114718


namespace ratio_x_to_y_l1147_114705

theorem ratio_x_to_y (x y : ℝ) (h : y = x * (1 - 0.9166666666666666)) :
  x / y = 12 :=
sorry

end ratio_x_to_y_l1147_114705


namespace hyperbola_center_l1147_114729

/-- The equation of a hyperbola in general form -/
def HyperbolaEquation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 16 * y^2 + 128 * y - 400 = 0

/-- The center of a hyperbola -/
def HyperbolaCenter : ℝ × ℝ := (3, 4)

/-- Theorem: The center of the hyperbola defined by the given equation is (3, 4) -/
theorem hyperbola_center :
  ∀ (x y : ℝ), HyperbolaEquation x y →
  ∃ (a b : ℝ), (x - HyperbolaCenter.1)^2 / a^2 - (y - HyperbolaCenter.2)^2 / b^2 = 1 :=
by sorry

end hyperbola_center_l1147_114729


namespace number_of_clients_l1147_114728

/-- Proves that the number of clients who visited the garage is 15, given the specified conditions. -/
theorem number_of_clients (num_cars : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) : 
  num_cars = 10 → selections_per_client = 2 → selections_per_car = 3 → 
  (num_cars * selections_per_car) / selections_per_client = 15 := by
  sorry

end number_of_clients_l1147_114728


namespace quadratic_pair_sum_zero_l1147_114789

/-- A quadratic function and its inverse -/
def QuadraticPair (a b c : ℝ) : Prop :=
  ∃ (g : ℝ → ℝ) (g_inv : ℝ → ℝ),
    (∀ x, g x = a * x^2 + b * x + c) ∧
    (∀ x, g_inv x = c * x^2 + b * x + a) ∧
    (∀ x, g (g_inv x) = x) ∧
    (∀ x, g_inv (g x) = x)

theorem quadratic_pair_sum_zero (a b c : ℝ) (h : QuadraticPair a b c) :
  a + b + c = 0 := by
  sorry

end quadratic_pair_sum_zero_l1147_114789


namespace book_arrangement_l1147_114748

theorem book_arrangement (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 3) :
  (Nat.factorial n) / ((Nat.factorial (n / k))^k * Nat.factorial k) =
  (Nat.factorial 30) / ((Nat.factorial 10)^3 * Nat.factorial 3) :=
by sorry

end book_arrangement_l1147_114748


namespace tangent_line_t_value_l1147_114734

/-- A line in polar coordinates defined by ρcosθ = t, where t > 0 -/
structure PolarLine where
  t : ℝ
  t_pos : t > 0

/-- A curve in polar coordinates defined by ρ = 2sinθ -/
def PolarCurve : Type := Unit

/-- Predicate to check if a line is tangent to the curve -/
def is_tangent (l : PolarLine) (c : PolarCurve) : Prop := sorry

theorem tangent_line_t_value (l : PolarLine) (c : PolarCurve) :
  is_tangent l c → l.t = 1 := by sorry

end tangent_line_t_value_l1147_114734


namespace jasmine_books_pages_l1147_114773

theorem jasmine_books_pages (books : Set ℕ) 
  (shortest longest middle : ℕ) 
  (h1 : shortest ∈ books) 
  (h2 : longest ∈ books) 
  (h3 : middle ∈ books)
  (h4 : shortest = longest / 4)
  (h5 : middle = 297)
  (h6 : middle = 3 * shortest)
  (h7 : ∀ b ∈ books, b ≤ longest) :
  longest = 396 := by
  sorry

end jasmine_books_pages_l1147_114773


namespace vector_at_negative_seven_l1147_114765

/-- A parametric line in 2D space -/
structure ParametricLine where
  /-- The position vector at t = 0 -/
  a : ℝ × ℝ
  /-- The direction vector of the line -/
  d : ℝ × ℝ

/-- Get the vector on the line at a given t -/
def ParametricLine.vectorAt (line : ParametricLine) (t : ℝ) : ℝ × ℝ :=
  (line.a.1 + t * line.d.1, line.a.2 + t * line.d.2)

/-- The main theorem -/
theorem vector_at_negative_seven
  (line : ParametricLine)
  (h1 : line.vectorAt 2 = (1, 4))
  (h2 : line.vectorAt 3 = (3, -4)) :
  line.vectorAt (-7) = (-17, 76) := by
  sorry


end vector_at_negative_seven_l1147_114765


namespace martha_savings_l1147_114771

/-- Martha's daily allowance in dollars -/
def daily_allowance : ℚ := 12

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Number of days Martha saved half her allowance -/
def days_half_saved : ℕ := days_in_week - 1

/-- Amount saved when Martha saves half her allowance -/
def half_savings : ℚ := daily_allowance / 2

/-- Amount saved when Martha saves a quarter of her allowance -/
def quarter_savings : ℚ := daily_allowance / 4

/-- Martha's total savings for the week -/
def total_savings : ℚ := days_half_saved * half_savings + quarter_savings

theorem martha_savings : total_savings = 39 := by
  sorry

end martha_savings_l1147_114771


namespace sqrt_plus_reciprocal_inequality_l1147_114716

theorem sqrt_plus_reciprocal_inequality (x : ℝ) (h : x > 0) : Real.sqrt x + 1 / Real.sqrt x ≥ 2 := by
  sorry

end sqrt_plus_reciprocal_inequality_l1147_114716


namespace f_monotonicity_l1147_114739

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * Real.log x - 1 / x

theorem f_monotonicity (k : ℝ) :
  (∀ x > 0, HasDerivAt (f k) ((k * x + 1) / (x^2)) x) →
  (k ≥ 0 → ∀ x > 0, (k * x + 1) / (x^2) > 0) ∧
  (k < 0 → (∀ x, 0 < x ∧ x < -1/k → (k * x + 1) / (x^2) > 0) ∧
           (∀ x > -1/k, (k * x + 1) / (x^2) < 0)) :=
by sorry

end f_monotonicity_l1147_114739


namespace bottles_remaining_l1147_114788

theorem bottles_remaining (cases : ℕ) (bottles_per_case : ℕ) (used_first_game : ℕ) (used_second_game : ℕ) :
  cases = 10 →
  bottles_per_case = 20 →
  used_first_game = 70 →
  used_second_game = 110 →
  cases * bottles_per_case - used_first_game - used_second_game = 20 :=
by
  sorry

end bottles_remaining_l1147_114788


namespace angle_A_value_min_side_a_value_l1147_114742

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom angle_side_relation : (2 * c - b) * Real.cos A = a * Real.cos B
axiom triangle_area : (1 / 2) * b * c * Real.sin A = 2 * Real.sqrt 3

-- Theorem statements
theorem angle_A_value : A = π / 3 := by sorry

theorem min_side_a_value : ∃ (a_min : ℝ), a_min = 2 * Real.sqrt 2 ∧ ∀ (a : ℝ), a ≥ a_min := by sorry

end angle_A_value_min_side_a_value_l1147_114742


namespace min_value_exponential_sum_l1147_114755

theorem min_value_exponential_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : 2 * x + y = 6) :
  ∃ (m : ℝ), m = 54 ∧ ∀ (z : ℝ), 9^x + 3^y ≥ z → z ≤ m :=
by sorry

end min_value_exponential_sum_l1147_114755


namespace trees_on_road_l1147_114719

/-- Calculates the number of trees that can be planted along a road -/
def numTrees (roadLength : ℕ) (interval : ℕ) : ℕ :=
  roadLength / interval + 1

/-- Theorem stating the number of trees that can be planted on a 100-meter road with 5-meter intervals -/
theorem trees_on_road :
  numTrees 100 5 = 21 := by
  sorry

end trees_on_road_l1147_114719


namespace max_value_f_monotonic_condition_inequality_condition_l1147_114714

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := (-x^2 + 2*a*x) * Real.exp x
def g (x : ℝ) : ℝ := (x - 1) * Real.exp (2*x)

-- Theorem for part (I)
theorem max_value_f (a : ℝ) (h : a ≥ 0) :
  ∃ x : ℝ, x = a - 1 + Real.sqrt (a^2 + 1) ∨ x = a - 1 - Real.sqrt (a^2 + 1) ∧
  ∀ y : ℝ, f a y ≤ f a x :=
sorry

-- Theorem for part (II)
theorem monotonic_condition (a : ℝ) :
  (∀ x y : ℝ, -1 ≤ x ∧ x < y ∧ y ≤ 1 → f a x < f a y) ↔ a ≥ 3/4 :=
sorry

-- Theorem for part (III)
theorem inequality_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≤ g x) ↔ 0 ≤ a ∧ a ≤ 1/2 :=
sorry

end max_value_f_monotonic_condition_inequality_condition_l1147_114714


namespace kindWizardCanAchieveGoal_l1147_114768

/-- Represents a gnome -/
structure Gnome where
  id : Nat

/-- Represents a friendship between two gnomes -/
structure Friendship where
  gnome1 : Gnome
  gnome2 : Gnome

/-- Represents a round table with gnomes -/
structure RoundTable where
  gnomes : List Gnome

/-- The kind wizard's action of making gnomes friends -/
def makeGnomesFriends (pairs : List (Gnome × Gnome)) : List Friendship :=
  sorry

/-- The evil wizard's action of making gnomes unfriends -/
def makeGnomesUnfriends (friendships : List Friendship) (n : Nat) : List Friendship :=
  sorry

/-- Check if a seating arrangement is valid (all adjacent gnomes are friends) -/
def isValidSeating (seating : List Gnome) (friendships : List Friendship) : Prop :=
  sorry

theorem kindWizardCanAchieveGoal (n : Nat) (hn : n > 1 ∧ Odd n) :
  ∃ (table1 table2 : RoundTable),
    table1.gnomes.length = n ∧
    table2.gnomes.length = n ∧
    (∀ (pairs : List (Gnome × Gnome)),
      pairs.length = 2 * n →
      ∀ (evilAction : List Friendship → List Friendship),
        ∃ (finalSeating : List Gnome),
          finalSeating.length = 2 * n ∧
          isValidSeating finalSeating (evilAction (makeGnomesFriends pairs))) :=
by sorry

end kindWizardCanAchieveGoal_l1147_114768


namespace student_line_arrangements_l1147_114794

def number_of_arrangements (n : ℕ) : ℕ := n.factorial

def arrangements_with_two_together (n : ℕ) : ℕ := (n - 1).factorial * 2

theorem student_line_arrangements :
  let total_students : ℕ := 5
  let total_arrangements := number_of_arrangements total_students
  let arrangements_with_specific_two_together := arrangements_with_two_together total_students
  total_arrangements - arrangements_with_specific_two_together = 72 := by
sorry

end student_line_arrangements_l1147_114794


namespace streaming_bill_fixed_fee_l1147_114774

/-- Represents the billing structure for a streaming service -/
structure StreamingBill where
  fixedFee : ℝ
  movieCharge : ℝ

/-- Calculates the total bill given number of movies watched -/
def StreamingBill.totalBill (bill : StreamingBill) (movies : ℝ) : ℝ :=
  bill.fixedFee + bill.movieCharge * movies

theorem streaming_bill_fixed_fee (bill : StreamingBill) :
  bill.totalBill 1 = 15.30 →
  bill.totalBill 1.5 = 20.55 →
  bill.fixedFee = 4.80 := by
  sorry

end streaming_bill_fixed_fee_l1147_114774


namespace max_area_inscribed_quadrilateral_l1147_114745

/-- A quadrilateral inscribed in a semi-circle -/
structure InscribedQuadrilateral (r : ℝ) where
  vertices : Fin 4 → ℝ × ℝ
  inside_semicircle : ∀ i, (vertices i).1^2 + (vertices i).2^2 ≤ r^2 ∧ (vertices i).2 ≥ 0

/-- The area of a quadrilateral -/
def area (q : InscribedQuadrilateral r) : ℝ :=
  sorry

/-- The shape of a half regular hexagon -/
def half_regular_hexagon (r : ℝ) : InscribedQuadrilateral r :=
  sorry

theorem max_area_inscribed_quadrilateral (r : ℝ) (hr : r > 0) :
  (∀ q : InscribedQuadrilateral r, area q ≤ (3 * Real.sqrt 3 / 4) * r^2) ∧
  area (half_regular_hexagon r) = (3 * Real.sqrt 3 / 4) * r^2 :=
sorry

end max_area_inscribed_quadrilateral_l1147_114745
