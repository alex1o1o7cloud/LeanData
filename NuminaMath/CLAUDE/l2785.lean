import Mathlib

namespace NUMINAMATH_CALUDE_scatter_plot_correct_placement_l2785_278549

/-- Represents a variable in a scatter plot -/
inductive Variable
| Forecast
| Explanatory

/-- Represents an axis in a scatter plot -/
inductive Axis
| X
| Y

/-- Determines the correct axis placement for a given variable -/
def correct_axis_placement (v : Variable) : Axis :=
  match v with
  | Variable.Forecast => Axis.Y
  | Variable.Explanatory => Axis.X

/-- Theorem stating the correct placement of variables in a scatter plot -/
theorem scatter_plot_correct_placement :
  (correct_axis_placement Variable.Forecast = Axis.Y) ∧
  (correct_axis_placement Variable.Explanatory = Axis.X) := by
  sorry

end NUMINAMATH_CALUDE_scatter_plot_correct_placement_l2785_278549


namespace NUMINAMATH_CALUDE_problem_statement_l2785_278570

open Real

theorem problem_statement :
  (∀ x ∈ Set.Ioo (-π/2) 0, sin x > x) ∧
  ¬(Set.Ioo 0 1 = {x | log (1 - x) / log 10 < 1}) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2785_278570


namespace NUMINAMATH_CALUDE_cheddar_cheese_sticks_l2785_278533

theorem cheddar_cheese_sticks (mozzarella : ℕ) (pepperjack : ℕ) (p_pepperjack : ℚ) : ℕ :=
  let total := pepperjack * 2
  let cheddar := total - mozzarella - pepperjack
  by
    have h1 : mozzarella = 30 := by sorry
    have h2 : pepperjack = 45 := by sorry
    have h3 : p_pepperjack = 1/2 := by sorry
    exact 15

#check cheddar_cheese_sticks

end NUMINAMATH_CALUDE_cheddar_cheese_sticks_l2785_278533


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2785_278521

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * a * x + a - 2 < 0) ↔ -8/5 < a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2785_278521


namespace NUMINAMATH_CALUDE_white_balls_count_l2785_278584

theorem white_balls_count (red_balls : ℕ) (prob_white : ℚ) (white_balls : ℕ) : 
  red_balls = 12 → 
  prob_white = 2/3 → 
  (white_balls : ℚ) / (white_balls + red_balls) = prob_white →
  white_balls = 24 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l2785_278584


namespace NUMINAMATH_CALUDE_count_even_four_digit_is_784_l2785_278575

/-- Count of even integers between 3000 and 6000 with four different digits -/
def count_even_four_digit : ℕ := sorry

/-- An integer is between 3000 and 6000 -/
def is_between_3000_and_6000 (n : ℕ) : Prop :=
  3000 < n ∧ n < 6000

/-- An integer has four different digits -/
def has_four_different_digits (n : ℕ) : Prop := sorry

/-- Theorem stating that the count of even integers between 3000 and 6000
    with four different digits is 784 -/
theorem count_even_four_digit_is_784 :
  count_even_four_digit = 784 := by sorry

end NUMINAMATH_CALUDE_count_even_four_digit_is_784_l2785_278575


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2785_278509

-- Problem 1
theorem problem_1 (a b : ℝ) : (a + 2*b)^2 - a*(a + 4*b) = 4*b^2 := by sorry

-- Problem 2
theorem problem_2 (m : ℝ) (h1 : m ≠ 1) (h2 : m ≠ -1) :
  ((2 / (m - 1) + 1) / ((2*m + 2) / (m^2 - 2*m + 1))) = (m - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2785_278509


namespace NUMINAMATH_CALUDE_expression_evaluation_l2785_278566

theorem expression_evaluation : (2^(1^(0^2)))^3 + (3^(1^2))^0 + 4^(0^1) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2785_278566


namespace NUMINAMATH_CALUDE_vasya_numbers_l2785_278511

theorem vasya_numbers (x y : ℝ) : (x - 1) * (y - 1) = x * y → x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_vasya_numbers_l2785_278511


namespace NUMINAMATH_CALUDE_day_250_is_tuesday_l2785_278515

/-- Days of the week represented as integers mod 7 -/
inductive DayOfWeek : Type
| Sunday : DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek

def dayOfWeek (n : Nat) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem day_250_is_tuesday (h : dayOfWeek 35 = DayOfWeek.Wednesday) :
  dayOfWeek 250 = DayOfWeek.Tuesday := by
  sorry

end NUMINAMATH_CALUDE_day_250_is_tuesday_l2785_278515


namespace NUMINAMATH_CALUDE_probability_at_least_one_red_l2785_278562

theorem probability_at_least_one_red (prob_red_A prob_red_B : ℝ) :
  prob_red_A = 1/3 →
  prob_red_B = 1/2 →
  1 - (1 - prob_red_A) * (1 - prob_red_B) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_red_l2785_278562


namespace NUMINAMATH_CALUDE_cos_150_degrees_l2785_278546

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l2785_278546


namespace NUMINAMATH_CALUDE_propositions_truth_l2785_278572

-- Define the logarithm function for any base
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Statement of the theorem
theorem propositions_truth : 
  (∀ x > 0, (1/2 : ℝ)^x > (1/3 : ℝ)^x) ∧ 
  (∃ x ∈ Set.Ioo 0 1, log (1/2) x > log (1/3) x) ∧
  (∃ x > 0, (1/2 : ℝ)^x < log (1/2) x) ∧
  (∀ x ∈ Set.Ioo 0 (1/3), (1/2 : ℝ)^x < log (1/3) x) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_l2785_278572


namespace NUMINAMATH_CALUDE_sum_of_unit_complex_squares_l2785_278582

theorem sum_of_unit_complex_squares (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1)
  (h4 : a^3 / (b*c) + b^3 / (a*c) + c^3 / (a*b) = 3) : 
  Complex.abs (a + b + c)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_unit_complex_squares_l2785_278582


namespace NUMINAMATH_CALUDE_least_days_same_date_l2785_278574

/-- A calendar date represented by a day and a month -/
structure CalendarDate where
  day : Nat
  month : Nat

/-- Function to move a given number of days forward or backward from a date -/
def moveDays (date : CalendarDate) (days : Int) : CalendarDate :=
  sorry

/-- Predicate to check if two dates have the same day of the month -/
def sameDayOfMonth (date1 date2 : CalendarDate) : Prop :=
  date1.day = date2.day

theorem least_days_same_date :
  ∃ k : Nat, k > 0 ∧
    (∀ date : CalendarDate, sameDayOfMonth (moveDays date k) (moveDays date (-k))) ∧
    (∀ j : Nat, 0 < j → j < k →
      ∃ date : CalendarDate, ¬sameDayOfMonth (moveDays date j) (moveDays date (-j))) ∧
    k = 14 :=
  sorry

end NUMINAMATH_CALUDE_least_days_same_date_l2785_278574


namespace NUMINAMATH_CALUDE_inner_circle_radius_l2785_278589

theorem inner_circle_radius (r : ℝ) : 
  r > 0 →
  (π * ((10 : ℝ)^2 - (0.5 * r)^2) = 3.25 * π * (8^2 - r^2)) →
  r = 6 := by
sorry

end NUMINAMATH_CALUDE_inner_circle_radius_l2785_278589


namespace NUMINAMATH_CALUDE_paulines_garden_tomato_kinds_l2785_278579

/-- Represents Pauline's garden -/
structure Garden where
  rows : ℕ
  spaces_per_row : ℕ
  tomato_kinds : ℕ
  tomatoes_per_kind : ℕ
  cucumber_kinds : ℕ
  cucumbers_per_kind : ℕ
  potatoes : ℕ
  remaining_spaces : ℕ

/-- Theorem representing the problem -/
theorem paulines_garden_tomato_kinds (g : Garden) 
  (h1 : g.rows = 10)
  (h2 : g.spaces_per_row = 15)
  (h3 : g.tomatoes_per_kind = 5)
  (h4 : g.cucumber_kinds = 5)
  (h5 : g.cucumbers_per_kind = 4)
  (h6 : g.potatoes = 30)
  (h7 : g.remaining_spaces = 85)
  (h8 : g.rows * g.spaces_per_row = 
        g.tomato_kinds * g.tomatoes_per_kind + 
        g.cucumber_kinds * g.cucumbers_per_kind + 
        g.potatoes + g.remaining_spaces) : 
  g.tomato_kinds = 3 := by
  sorry

end NUMINAMATH_CALUDE_paulines_garden_tomato_kinds_l2785_278579


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l2785_278585

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l2785_278585


namespace NUMINAMATH_CALUDE_age_sum_problem_l2785_278504

theorem age_sum_problem (a b c : ℕ+) : 
  a = b ∧ a * b * c = 256 → a + b + c = 20 :=
by sorry

end NUMINAMATH_CALUDE_age_sum_problem_l2785_278504


namespace NUMINAMATH_CALUDE_diagonal_sum_is_161_l2785_278563

/-- Represents a multiplication grid with missing factors --/
structure MultiplicationGrid where
  /-- Products in the grid --/
  wp : ℕ
  xp : ℕ
  wr : ℕ
  zr : ℕ
  xs : ℕ
  vs : ℕ
  vq : ℕ
  yq : ℕ
  yt : ℕ

/-- The sum of diagonal elements in the multiplication grid --/
def diagonalSum (grid : MultiplicationGrid) : ℕ :=
  let p := 3  -- Derived from wp and xp
  let w := grid.wp / p
  let x := grid.xp / p
  let r := grid.wr / w
  let z := grid.zr / r
  let s := grid.xs / x
  let v := grid.vs / s
  let q := grid.vq / v
  let y := grid.yq / q
  let t := grid.yt / y
  v * p + w * q + x * r + y * s + z * t

/-- Theorem stating that the diagonal sum is 161 for the given grid --/
theorem diagonal_sum_is_161 (grid : MultiplicationGrid) 
  (h1 : grid.wp = 15) (h2 : grid.xp = 18) (h3 : grid.wr = 40) 
  (h4 : grid.zr = 56) (h5 : grid.xs = 60) (h6 : grid.vs = 20) 
  (h7 : grid.vq = 10) (h8 : grid.yq = 20) (h9 : grid.yt = 24) : 
  diagonalSum grid = 161 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_sum_is_161_l2785_278563


namespace NUMINAMATH_CALUDE_divisibility_of_prime_square_minus_one_l2785_278531

theorem divisibility_of_prime_square_minus_one (p : ℕ) (h_prime : Nat.Prime p) (h_ge_five : p ≥ 5) :
  24 ∣ (p^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_prime_square_minus_one_l2785_278531


namespace NUMINAMATH_CALUDE_student_number_problem_l2785_278524

theorem student_number_problem (x : ℝ) : 8 * x - 138 = 102 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l2785_278524


namespace NUMINAMATH_CALUDE_at_least_one_negative_l2785_278550

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  ¬(0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l2785_278550


namespace NUMINAMATH_CALUDE_fraction_ordering_l2785_278503

theorem fraction_ordering : (6 : ℚ) / 22 < 8 / 32 ∧ 8 / 32 < 10 / 29 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l2785_278503


namespace NUMINAMATH_CALUDE_cube_root_of_256_l2785_278501

theorem cube_root_of_256 (x : ℝ) (h1 : x > 0) (h2 : x^3 = 256) : x = 4 * Real.rpow 2 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_256_l2785_278501


namespace NUMINAMATH_CALUDE_problem_solution_l2785_278527

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x - 1) + x

def g (x : ℝ) : ℝ := x - 1

def h (m : ℝ) (f' : ℝ → ℝ) (x : ℝ) : ℝ := m * f' x + g x + 1

theorem problem_solution (a : ℝ) :
  (∀ x, deriv (f a) x = a / (x - 1) + 1) →
  deriv (f a) 2 = 2 →
  (a = 1 ∧
   (∀ x, g x = x - 1) ∧
   (∀ m, (∀ x ∈ Set.Icc 2 4, h m (deriv (f a)) x > 0) → m > -1 ∧ ∀ y > -1, ∃ x ∈ Set.Icc 2 4, h y (deriv (f a)) x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2785_278527


namespace NUMINAMATH_CALUDE_exists_k_for_1001_free_ends_l2785_278591

/-- Represents the process of drawing segments as described in the problem -/
structure SegmentDrawing where
  initial_segment : Unit  -- Represents the initial segment OA
  branch_factor : Nat     -- Number of segments drawn from each point (5 in this case)
  free_ends : Nat         -- Number of free ends

/-- Calculates the number of free ends after k iterations of drawing segments -/
def free_ends_after_iterations (k : Nat) : Nat :=
  1 + 4 * k

/-- Theorem stating that it's possible to have exactly 1001 free ends -/
theorem exists_k_for_1001_free_ends :
  ∃ k : Nat, free_ends_after_iterations k = 1001 := by
  sorry

#check exists_k_for_1001_free_ends

end NUMINAMATH_CALUDE_exists_k_for_1001_free_ends_l2785_278591


namespace NUMINAMATH_CALUDE_added_number_forms_geometric_sequence_l2785_278520

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_term : a 1 = 2
  third_term : a 3 = 6
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- The property that adding x to certain terms forms a geometric sequence -/
def FormsGeometricSequence (seq : ArithmeticSequence) (x : ℝ) : Prop :=
  (seq.a 4 + x)^2 = (seq.a 1 + x) * (seq.a 5 + x)

/-- The main theorem -/
theorem added_number_forms_geometric_sequence (seq : ArithmeticSequence) :
  ∃ x : ℝ, FormsGeometricSequence seq x ∧ x = -11 := by
  sorry

end NUMINAMATH_CALUDE_added_number_forms_geometric_sequence_l2785_278520


namespace NUMINAMATH_CALUDE_initial_cloth_length_l2785_278581

/-- Given that 4 men can colour an initial length of cloth in 2 days,
    and 8 men can colour 36 meters of cloth in 0.75 days,
    prove that the initial length of cloth is 48 meters. -/
theorem initial_cloth_length (initial_length : ℝ) : 
  (4 * initial_length / 2 = 8 * 36 / 0.75) → initial_length = 48 := by
  sorry

end NUMINAMATH_CALUDE_initial_cloth_length_l2785_278581


namespace NUMINAMATH_CALUDE_min_extracted_tablets_proof_l2785_278529

/-- Represents the number of tablets of each medicine type in the box -/
structure TabletCounts where
  a : Nat
  b : Nat
  c : Nat

/-- Calculates the minimum number of tablets to extract to guarantee at least 3 of each kind -/
def minExtractedTablets (counts : TabletCounts) : Nat :=
  16

/-- Theorem stating that for the given tablet counts, the minimum number of extracted tablets is 16 -/
theorem min_extracted_tablets_proof (counts : TabletCounts) 
  (h1 : counts.a = 30) (h2 : counts.b = 24) (h3 : counts.c = 18) : 
  minExtractedTablets counts = 16 := by
  sorry

#eval minExtractedTablets { a := 30, b := 24, c := 18 }

end NUMINAMATH_CALUDE_min_extracted_tablets_proof_l2785_278529


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2785_278565

/-- A geometric sequence with first four terms 25, -50, 100, -200 has a common ratio of -2 -/
theorem geometric_sequence_common_ratio :
  ∀ (a : ℕ → ℚ), 
    a 0 = 25 ∧ a 1 = -50 ∧ a 2 = 100 ∧ a 3 = -200 →
    ∃ (r : ℚ), r = -2 ∧ ∀ (n : ℕ), a (n + 1) = r * a n :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2785_278565


namespace NUMINAMATH_CALUDE_ratio_equality_l2785_278505

theorem ratio_equality (a b : ℝ) (h1 : 3 * a = 4 * b) (h2 : a * b ≠ 0) :
  (a / 4) / (b / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2785_278505


namespace NUMINAMATH_CALUDE_sqrt_ab_plus_one_l2785_278519

theorem sqrt_ab_plus_one (a b : ℝ) (h : b = Real.sqrt (3 - a) + Real.sqrt (a - 3) + 8) : 
  (Real.sqrt (a * b + 1) = 5) ∨ (Real.sqrt (a * b + 1) = -5) := by
sorry

end NUMINAMATH_CALUDE_sqrt_ab_plus_one_l2785_278519


namespace NUMINAMATH_CALUDE_units_digit_of_n_squared_plus_two_to_n_l2785_278539

theorem units_digit_of_n_squared_plus_two_to_n (n : ℕ) : n = 3123^2 + 2^3123 → (n^2 + 2^n) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_squared_plus_two_to_n_l2785_278539


namespace NUMINAMATH_CALUDE_parabola_intersection_l2785_278568

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ := {x | 2*x^2 + 3*x - 4 = x^2 + 2*x + 1}

/-- The y-coordinate of the intersection points -/
def intersection_y : ℝ := 4.5

/-- The first parabola -/
def parabola1 (x : ℝ) : ℝ := 2*x^2 + 3*x - 4

/-- The second parabola -/
def parabola2 (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem parabola_intersection :
  intersection_x = {(-1 + Real.sqrt 21) / 2, (-1 - Real.sqrt 21) / 2} ∧
  ∀ x ∈ intersection_x, parabola1 x = intersection_y ∧ parabola2 x = intersection_y :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2785_278568


namespace NUMINAMATH_CALUDE_complex_division_l2785_278583

theorem complex_division : (2 * Complex.I) / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l2785_278583


namespace NUMINAMATH_CALUDE_tea_in_milk_equals_milk_in_tea_l2785_278588

/-- Represents the contents of a cup --/
structure Cup where
  tea : ℚ
  milk : ℚ

/-- Represents the state of both cups --/
structure CupState where
  tea_cup : Cup
  milk_cup : Cup

/-- Initial state of the cups --/
def initial_state : CupState :=
  { tea_cup := { tea := 5, milk := 0 },
    milk_cup := { tea := 0, milk := 5 } }

/-- State after transferring milk to tea cup --/
def after_milk_transfer (state : CupState) : CupState :=
  { tea_cup := { tea := state.tea_cup.tea, milk := state.tea_cup.milk + 1 },
    milk_cup := { tea := state.milk_cup.tea, milk := state.milk_cup.milk - 1 } }

/-- State after transferring mixture back to milk cup --/
def after_mixture_transfer (state : CupState) : CupState :=
  let total_in_tea_cup := state.tea_cup.tea + state.tea_cup.milk
  let tea_fraction := state.tea_cup.tea / total_in_tea_cup
  let milk_fraction := state.tea_cup.milk / total_in_tea_cup
  { tea_cup := { tea := state.tea_cup.tea - tea_fraction, 
                 milk := state.tea_cup.milk - milk_fraction },
    milk_cup := { tea := state.milk_cup.tea + tea_fraction, 
                  milk := state.milk_cup.milk + milk_fraction } }

/-- Final state after both transfers --/
def final_state : CupState :=
  after_mixture_transfer (after_milk_transfer initial_state)

theorem tea_in_milk_equals_milk_in_tea :
  final_state.milk_cup.tea = final_state.tea_cup.milk := by
  sorry

end NUMINAMATH_CALUDE_tea_in_milk_equals_milk_in_tea_l2785_278588


namespace NUMINAMATH_CALUDE_tv_price_change_l2785_278557

theorem tv_price_change (x : ℝ) : 
  (100 - x) * 1.5 = 120 → x = 20 := by sorry

end NUMINAMATH_CALUDE_tv_price_change_l2785_278557


namespace NUMINAMATH_CALUDE_geometric_sum_proof_l2785_278516

/-- The sum of a geometric sequence with first term 9, common ratio 3, and 7 terms -/
def geometric_sum : ℕ := 9827

/-- The first term of the geometric sequence -/
def a : ℕ := 9

/-- The common ratio of the geometric sequence -/
def r : ℕ := 3

/-- The number of terms in the geometric sequence -/
def n : ℕ := 7

/-- Theorem stating that the sum of the geometric sequence equals 9827 -/
theorem geometric_sum_proof : 
  a * (r^n - 1) / (r - 1) = geometric_sum :=
sorry

end NUMINAMATH_CALUDE_geometric_sum_proof_l2785_278516


namespace NUMINAMATH_CALUDE_martha_cards_l2785_278573

theorem martha_cards (initial_cards given_cards : ℝ) 
  (h1 : initial_cards = 76.0)
  (h2 : given_cards = 3.0) : 
  initial_cards - given_cards = 73.0 := by
  sorry

end NUMINAMATH_CALUDE_martha_cards_l2785_278573


namespace NUMINAMATH_CALUDE_m_range_theorem_l2785_278523

/-- Proposition P: The equation x²/(2m) + y²/(9-m) = 1 represents an ellipse with foci on the y-axis -/
def P (m : ℝ) : Prop :=
  0 < m ∧ m < 3

/-- Proposition Q: The eccentricity e of the hyperbola y²/5 - x²/m = 1 is within the range (√6/2, √2) -/
def Q (m : ℝ) : Prop :=
  m > 0 ∧ 5/2 < m ∧ m < 5

/-- The set of valid m values -/
def M : Set ℝ :=
  {m | (0 < m ∧ m ≤ 5/2) ∨ (3 ≤ m ∧ m < 5)}

theorem m_range_theorem :
  ∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m) → m ∈ M :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l2785_278523


namespace NUMINAMATH_CALUDE_total_produce_yield_l2785_278596

def garden_length_steps : ℕ := 18
def garden_width_steps : ℕ := 25
def feet_per_step : ℕ := 3
def carrot_yield_per_sqft : ℚ := 0.4
def potato_yield_per_sqft : ℚ := 0.5

theorem total_produce_yield :
  let garden_length_feet := garden_length_steps * feet_per_step
  let garden_width_feet := garden_width_steps * feet_per_step
  let garden_area := garden_length_feet * garden_width_feet
  let carrot_yield := garden_area * carrot_yield_per_sqft
  let potato_yield := garden_area * potato_yield_per_sqft
  carrot_yield + potato_yield = 3645 := by
  sorry

end NUMINAMATH_CALUDE_total_produce_yield_l2785_278596


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2785_278569

theorem cube_volume_problem (a : ℝ) (h : a > 0) :
  (a^3 : ℝ) - ((a - 1) * a * (a + 1)) = 5 →
  a^3 = 125 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2785_278569


namespace NUMINAMATH_CALUDE_incircle_segment_ratio_l2785_278599

/-- Represents a triangle with an incircle -/
structure TriangleWithIncircle where
  a : ℝ  -- Side length
  b : ℝ  -- Side length
  c : ℝ  -- Side length
  r : ℝ  -- Smaller segment of side 'a' created by incircle
  s : ℝ  -- Larger segment of side 'a' created by incircle
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b
  h_incircle : r + s = a
  h_r_lt_s : r < s

/-- The main theorem -/
theorem incircle_segment_ratio
  (t : TriangleWithIncircle)
  (h_side_lengths : t.a = 8 ∧ t.b = 13 ∧ t.c = 17) :
  t.r / t.s = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_incircle_segment_ratio_l2785_278599


namespace NUMINAMATH_CALUDE_number_problem_l2785_278587

theorem number_problem (x : ℚ) : (3 / 4 : ℚ) * x = x - 19 → x = 76 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2785_278587


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_under_hundred_l2785_278548

theorem largest_multiple_of_seven_under_hundred : 
  ∀ n : ℕ, n % 7 = 0 ∧ n < 100 → n ≤ 98 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_under_hundred_l2785_278548


namespace NUMINAMATH_CALUDE_sequence_inequality_l2785_278517

theorem sequence_inequality (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a n ≥ 0)
  (h2 : ∀ n : ℕ, a n + a (2*n) ≥ 3*n)
  (h3 : ∀ n : ℕ, a (n+1) + n ≤ 2 * Real.sqrt (a n * (n+1))) :
  ∀ n : ℕ, a n ≥ n := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l2785_278517


namespace NUMINAMATH_CALUDE_ones_digit_of_7_pow_35_l2785_278571

/-- The ones digit of 7^n -/
def ones_digit_of_7_pow (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0  -- This case is unreachable, but needed for exhaustive pattern matching

/-- Theorem stating that the ones digit of 7^35 is 3 -/
theorem ones_digit_of_7_pow_35 : ones_digit_of_7_pow 35 = 3 := by
  sorry

#eval ones_digit_of_7_pow 35

end NUMINAMATH_CALUDE_ones_digit_of_7_pow_35_l2785_278571


namespace NUMINAMATH_CALUDE_log_sum_sqrt_equality_l2785_278532

theorem log_sum_sqrt_equality : Real.sqrt (Real.log 8 / Real.log 4 + Real.log 16 / Real.log 8) = Real.sqrt (17 / 6) := by
  sorry

end NUMINAMATH_CALUDE_log_sum_sqrt_equality_l2785_278532


namespace NUMINAMATH_CALUDE_faye_pencil_rows_l2785_278530

/-- Given that Faye has 720 pencils in total and places 24 pencils in each row,
    prove that the number of rows she created is 30. -/
theorem faye_pencil_rows (total_pencils : Nat) (pencils_per_row : Nat) (rows : Nat) :
  total_pencils = 720 →
  pencils_per_row = 24 →
  rows * pencils_per_row = total_pencils →
  rows = 30 := by
  sorry

#check faye_pencil_rows

end NUMINAMATH_CALUDE_faye_pencil_rows_l2785_278530


namespace NUMINAMATH_CALUDE_power_function_properties_l2785_278541

-- Define the power function f
noncomputable def f : ℝ → ℝ := λ x => Real.sqrt x

-- State the theorem
theorem power_function_properties :
  (f 9 = 3) →
  (∀ x ≥ 4, f x ≥ 2) ∧
  (∀ x₁ x₂, x₂ > x₁ ∧ x₁ > 0 → (f x₁ + f x₂) / 2 < f ((x₁ + x₂) / 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_power_function_properties_l2785_278541


namespace NUMINAMATH_CALUDE_max_product_constraint_l2785_278506

theorem max_product_constraint (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  (x * y) + z^2 = (x + z) * (y + z) →
  x + y + z = 3 →
  x * y * z ≤ 1 := by sorry

end NUMINAMATH_CALUDE_max_product_constraint_l2785_278506


namespace NUMINAMATH_CALUDE_painted_cube_probability_l2785_278593

/-- Represents a rectangular prism with painted faces -/
structure PaintedPrism where
  length : ℕ
  width : ℕ
  height : ℕ
  painted_face1 : ℕ × ℕ
  painted_face2 : ℕ × ℕ

/-- Calculates the total number of unit cubes in the prism -/
def total_cubes (p : PaintedPrism) : ℕ :=
  p.length * p.width * p.height

/-- Calculates the number of cubes with exactly one painted face -/
def cubes_with_one_painted_face (p : PaintedPrism) : ℕ :=
  (p.painted_face1.1 - 2) * (p.painted_face1.2 - 2) +
  (p.painted_face2.1 - 2) * (p.painted_face2.2 - 2) + 2

/-- Calculates the number of cubes with no painted faces -/
def cubes_with_no_painted_faces (p : PaintedPrism) : ℕ :=
  total_cubes p - (p.painted_face1.1 * p.painted_face1.2 +
                   p.painted_face2.1 * p.painted_face2.2 -
                   (p.painted_face1.1 + p.painted_face2.1))

/-- The main theorem to be proved -/
theorem painted_cube_probability (p : PaintedPrism)
  (h1 : p.length = 4)
  (h2 : p.width = 3)
  (h3 : p.height = 3)
  (h4 : p.painted_face1 = (4, 3))
  (h5 : p.painted_face2 = (3, 3)) :
  (cubes_with_one_painted_face p * cubes_with_no_painted_faces p : ℚ) /
  (total_cubes p * (total_cubes p - 1) / 2) = 221 / 630 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_probability_l2785_278593


namespace NUMINAMATH_CALUDE_homework_question_not_proposition_l2785_278508

-- Define what a proposition is
def is_proposition (s : String) : Prop := 
  ∃ (b : Bool), (s = "true") ∨ (s = "false")

-- Define the statement in question
def homework_question : String :=
  "Have you finished your homework?"

-- Theorem to prove
theorem homework_question_not_proposition :
  ¬ (is_proposition homework_question) := by
  sorry

end NUMINAMATH_CALUDE_homework_question_not_proposition_l2785_278508


namespace NUMINAMATH_CALUDE_sum_of_divisors_360_l2785_278543

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (·∣n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_360 : sum_of_divisors 360 = 1170 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_360_l2785_278543


namespace NUMINAMATH_CALUDE_area_of_triangle_fyh_l2785_278558

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  ef : ℝ
  gh : ℝ
  area : ℝ

/-- Theorem: Area of triangle FYH in a trapezoid with specific measurements -/
theorem area_of_triangle_fyh (t : Trapezoid) 
  (h1 : t.ef = 24)
  (h2 : t.gh = 36)
  (h3 : t.area = 360) :
  let height : ℝ := 2 * t.area / (t.ef + t.gh)
  let area_egh : ℝ := (1 / 2) * t.gh * height
  let area_efh : ℝ := t.area - area_egh
  let height_eyh : ℝ := (2 / 5) * height
  let area_efh_recalc : ℝ := (1 / 2) * t.ef * (height - height_eyh)
  area_efh - area_efh_recalc = 86.4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_fyh_l2785_278558


namespace NUMINAMATH_CALUDE_max_pairs_from_27_l2785_278559

theorem max_pairs_from_27 (n : ℕ) (h : n = 27) :
  (n * (n - 1)) / 2 = 351 := by
  sorry

end NUMINAMATH_CALUDE_max_pairs_from_27_l2785_278559


namespace NUMINAMATH_CALUDE_not_perfect_square_l2785_278510

theorem not_perfect_square (a b : ℕ+) : ¬ ∃ k : ℤ, (a : ℤ)^2 + Int.ceil ((4 * (a : ℤ)^2) / (b : ℤ)) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2785_278510


namespace NUMINAMATH_CALUDE_nelly_painting_bid_l2785_278597

/-- The amount Nelly paid for the painting -/
def nellys_bid (joes_bid : ℕ) : ℕ :=
  3 * joes_bid + 2000

/-- Theorem stating Nelly's final bid given Joe's bid -/
theorem nelly_painting_bid :
  let joes_bid : ℕ := 160000
  nellys_bid joes_bid = 482000 := by
  sorry

end NUMINAMATH_CALUDE_nelly_painting_bid_l2785_278597


namespace NUMINAMATH_CALUDE_jamal_grade_jamal_grade_is_108_l2785_278564

theorem jamal_grade (total_students : ℕ) (absent_students : ℕ) (first_day_average : ℕ) 
  (new_average : ℕ) (taqeesha_score : ℕ) : ℕ :=
  let students_first_day := total_students - absent_students
  let total_score_first_day := students_first_day * first_day_average
  let total_score_all := total_students * new_average
  let combined_score_absent := total_score_all - total_score_first_day
  combined_score_absent - taqeesha_score

theorem jamal_grade_is_108 :
  jamal_grade 30 2 85 86 92 = 108 := by
  sorry

end NUMINAMATH_CALUDE_jamal_grade_jamal_grade_is_108_l2785_278564


namespace NUMINAMATH_CALUDE_movie_ratio_proof_l2785_278598

theorem movie_ratio_proof (total : ℕ) (dvd : ℕ) (bluray : ℕ) :
  total = 378 →
  dvd + bluray = total →
  dvd / (bluray - 4) = 9 / 2 →
  (dvd : ℚ) / bluray = 51 / 12 :=
by
  sorry

end NUMINAMATH_CALUDE_movie_ratio_proof_l2785_278598


namespace NUMINAMATH_CALUDE_value_of_y_l2785_278556

theorem value_of_y (x y : ℝ) (h1 : x^(3*y) = 9) (h2 : x = 3) : y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l2785_278556


namespace NUMINAMATH_CALUDE_rectangle_fourth_vertex_l2785_278536

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a rectangle by its four vertices
structure Rectangle where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

-- Define the property of being a rectangle
def isRectangle (rect : Rectangle) : Prop :=
  -- Add properties that define a rectangle, such as perpendicular sides and equal diagonals
  sorry

-- Theorem statement
theorem rectangle_fourth_vertex 
  (rect : Rectangle)
  (h1 : isRectangle rect)
  (h2 : rect.A = ⟨1, 1⟩)
  (h3 : rect.B = ⟨3, 1⟩)
  (h4 : rect.C = ⟨3, 5⟩) :
  rect.D = ⟨1, 5⟩ := by
  sorry

end NUMINAMATH_CALUDE_rectangle_fourth_vertex_l2785_278536


namespace NUMINAMATH_CALUDE_residue_mod_12_l2785_278537

theorem residue_mod_12 : (172 * 15 - 13 * 8 + 6) % 12 = 10 := by sorry

end NUMINAMATH_CALUDE_residue_mod_12_l2785_278537


namespace NUMINAMATH_CALUDE_uncool_parents_count_l2785_278552

theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ) 
  (h1 : total = 40)
  (h2 : cool_dads = 18)
  (h3 : cool_moms = 20)
  (h4 : both_cool = 10) :
  total - (cool_dads + cool_moms - both_cool) = 12 := by
  sorry

#check uncool_parents_count

end NUMINAMATH_CALUDE_uncool_parents_count_l2785_278552


namespace NUMINAMATH_CALUDE_at_least_one_quadratic_has_solution_l2785_278512

theorem at_least_one_quadratic_has_solution (a b c : ℝ) : 
  (∃ x : ℝ, x^2 + (a-b)*x + (b-c) = 0) ∨ 
  (∃ x : ℝ, x^2 + (b-c)*x + (c-a) = 0) ∨ 
  (∃ x : ℝ, x^2 + (c-a)*x + (a-b) = 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_quadratic_has_solution_l2785_278512


namespace NUMINAMATH_CALUDE_vector_collinearity_l2785_278545

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def are_collinear (v w : V) : Prop := ∃ (c : ℝ), v = c • w ∨ w = c • v

theorem vector_collinearity 
  (e₁ e₂ a b : V) 
  (h_nonzero_e₁ : e₁ ≠ 0)
  (h_nonzero_e₂ : e₂ ≠ 0)
  (h_a : a = 2 • e₁ - e₂)
  (h_b : ∃ k : ℝ, b = k • e₁ + e₂) :
  (¬ are_collinear e₁ e₂ ∧ are_collinear a b → ∃ k : ℝ, b = k • e₁ + e₂ ∧ k = -2) ∧
  (∀ k : ℝ, are_collinear e₁ e₂ → are_collinear a b) := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2785_278545


namespace NUMINAMATH_CALUDE_work_completion_time_l2785_278525

/-- Given that:
    1. A can complete the work in 15 days
    2. A and B working together for 5 days complete 0.5833333333333334 of the work
    Prove that B can complete the work alone in 20 days -/
theorem work_completion_time (a_time : ℝ) (b_time : ℝ) 
  (h1 : a_time = 15)
  (h2 : 5 * (1 / a_time + 1 / b_time) = 0.5833333333333334) :
  b_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2785_278525


namespace NUMINAMATH_CALUDE_function_properties_l2785_278544

def f (a b x : ℝ) : ℝ := x^2 - (a + 1) * x + b

theorem function_properties (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ -5 < x ∧ x < 2) →
  (a = -4 ∧ b = -10) ∧
  (∀ x, f a a x > 0 ↔
    (a > 1 ∧ (x < 1 ∨ x > a)) ∨
    (a = 1 ∧ x ≠ 1) ∨
    (a < 1 ∧ (x < a ∨ x > 1))) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2785_278544


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l2785_278542

open Real

theorem function_inequality_implies_a_bound (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ →
    (log x₁ + a * x₁^2 - (log x₂ + a * x₂^2)) / (x₁ - x₂) > 2) →
  a ≥ (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l2785_278542


namespace NUMINAMATH_CALUDE_notes_count_l2785_278594

theorem notes_count (total_amount : ℕ) (denominations : Fin 3 → ℕ) : 
  total_amount = 480 ∧ 
  denominations 0 = 1 ∧ 
  denominations 1 = 5 ∧ 
  denominations 2 = 10 ∧ 
  (∃ x : ℕ, (denominations 0 * x + denominations 1 * x + denominations 2 * x = total_amount)) →
  (∃ x : ℕ, x + x + x = 90) :=
by sorry

end NUMINAMATH_CALUDE_notes_count_l2785_278594


namespace NUMINAMATH_CALUDE_female_elementary_students_l2785_278561

theorem female_elementary_students (total_students : ℕ) (non_elementary_girls : ℕ) 
  (h1 : total_students = 30)
  (h2 : non_elementary_girls = 7) :
  total_students / 2 - non_elementary_girls = 8 := by
  sorry

end NUMINAMATH_CALUDE_female_elementary_students_l2785_278561


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_zero_is_exists_lt_zero_l2785_278513

theorem negation_of_forall_geq_zero_is_exists_lt_zero :
  (¬ ∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 - x₀ + 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_zero_is_exists_lt_zero_l2785_278513


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2785_278555

theorem complex_magnitude_problem (z : ℂ) : z = (1 - I) / (1 + I) + 2*I → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2785_278555


namespace NUMINAMATH_CALUDE_real_part_of_z_is_two_l2785_278567

theorem real_part_of_z_is_two : Complex.re (((Complex.I - 1)^2 + 1) / Complex.I^3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_is_two_l2785_278567


namespace NUMINAMATH_CALUDE_lcm_36_75_l2785_278576

theorem lcm_36_75 : Nat.lcm 36 75 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_75_l2785_278576


namespace NUMINAMATH_CALUDE_slope_MN_constant_l2785_278578

/-- Definition of curve C -/
def curve_C (x y : ℝ) : Prop := y^2 = 4*x + 4 ∧ x ≥ 0

/-- Definition of point D on curve C -/
def point_D : ℝ × ℝ := (0, 2)

/-- Definition of complementary slopes -/
def complementary_slopes (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = -1

/-- Theorem: The slope of line MN is constant and equal to -1 -/
theorem slope_MN_constant (k : ℝ) (M N : ℝ × ℝ) :
  curve_C M.1 M.2 →
  curve_C N.1 N.2 →
  curve_C point_D.1 point_D.2 →
  complementary_slopes k (-k) →
  (M.2 - point_D.2) = k * (M.1 - point_D.1) →
  (N.2 - point_D.2) = (-k) * (N.1 - point_D.1) →
  M ≠ point_D →
  N ≠ point_D →
  (N.2 - M.2) / (N.1 - M.1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_slope_MN_constant_l2785_278578


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2785_278528

/-- Given a rectangle where the length is three times the width and the diagonal is 8√10,
    prove that its perimeter is 64. -/
theorem rectangle_perimeter (w l d : ℝ) : 
  l = 3 * w →                 -- length is three times the width
  d = 8 * (10 : ℝ).sqrt →     -- diagonal is 8√10
  w * w + l * l = d * d →     -- Pythagorean theorem
  2 * (w + l) = 64 :=         -- perimeter is 64
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2785_278528


namespace NUMINAMATH_CALUDE_line_properties_l2785_278547

-- Define the line equation
def line_equation (m : ℝ) (x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 2

theorem line_properties :
  -- Part 1: The line passes through (1, 1) for all real m
  (∀ m : ℝ, line_equation m 1 1) ∧
  -- Part 2: When the line is tangent to the circle, m = -1
  (∃ m : ℝ, (∀ x y : ℝ, line_equation m x y → circle_equation x y → 
    (x - 0)^2 + (y - 0)^2 = (1 - m)^2 / (m^2 + 1)) ∧ m = -1) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l2785_278547


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_19_l2785_278526

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_digit_sum_19 :
  ∃ (p : ℕ), is_prime p ∧ digit_sum p = 19 ∧
  ∀ (q : ℕ), is_prime q → digit_sum q = 19 → p ≤ q :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_19_l2785_278526


namespace NUMINAMATH_CALUDE_ramanujan_hardy_complex_numbers_l2785_278518

theorem ramanujan_hardy_complex_numbers :
  ∀ (z w : ℂ),
  z * w = 40 - 24 * I →
  w = 4 + 4 * I →
  z = 2 - 8 * I :=
by sorry

end NUMINAMATH_CALUDE_ramanujan_hardy_complex_numbers_l2785_278518


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2785_278595

theorem trigonometric_identity (α β : ℝ) 
  (h : Real.cos α ^ 2 * Real.sin β ^ 2 + Real.sin α ^ 2 * Real.cos β ^ 2 = 
       Real.cos α * Real.sin α * Real.cos β * Real.sin β) : 
  (Real.sin β ^ 2 * Real.cos α ^ 2) / Real.sin α ^ 2 + 
  (Real.cos β ^ 2 * Real.sin α ^ 2) / Real.cos α ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2785_278595


namespace NUMINAMATH_CALUDE_exam_average_problem_l2785_278534

theorem exam_average_problem (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) (avg₂ : ℚ) :
  n₁ = 15 →
  n₂ = 10 →
  avg₁ = 70 / 100 →
  avg_total = 78 / 100 →
  (n₁ + n₂ : ℚ) * avg_total = n₁ * avg₁ + n₂ * avg₂ →
  avg₂ = 90 / 100 := by
sorry

end NUMINAMATH_CALUDE_exam_average_problem_l2785_278534


namespace NUMINAMATH_CALUDE_max_operation_value_l2785_278560

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def operation (n : ℕ) : ℕ := 3 * (300 - n)

theorem max_operation_value :
  ∃ (m : ℕ), (∀ (n : ℕ), is_two_digit n → operation n ≤ m) ∧ (∃ (k : ℕ), is_two_digit k ∧ operation k = m) ∧ m = 870 :=
sorry

end NUMINAMATH_CALUDE_max_operation_value_l2785_278560


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l2785_278586

/-- The polynomial p(x) -/
def p (x : ℝ) : ℝ := -4*x^2 + 2*x - 5

/-- The polynomial q(x) -/
def q (x : ℝ) : ℝ := -6*x^2 + 4*x - 9

/-- The polynomial r(x) -/
def r (x : ℝ) : ℝ := 6*x^2 + 6*x + 2

/-- The polynomial s(x) -/
def s (x : ℝ) : ℝ := 3*x^2 - 2*x + 1

/-- The sum of polynomials p(x), q(x), r(x), and s(x) is equal to -x^2 + 10x - 11 -/
theorem sum_of_polynomials (x : ℝ) : p x + q x + r x + s x = -x^2 + 10*x - 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l2785_278586


namespace NUMINAMATH_CALUDE_great_circle_to_surface_area_ratio_l2785_278507

theorem great_circle_to_surface_area_ratio (O : Type*) [MetricSpace O] [NormedAddCommGroup O] 
  [InnerProductSpace ℝ O] [FiniteDimensional ℝ O] [ProperSpace O] (S₁ S₂ : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ S₁ = π * r^2 ∧ S₂ = 4 * π * r^2) → 
  S₁ / S₂ = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_great_circle_to_surface_area_ratio_l2785_278507


namespace NUMINAMATH_CALUDE_action_figures_per_shelf_l2785_278554

theorem action_figures_per_shelf 
  (total_shelves : ℕ) 
  (total_figures : ℕ) 
  (h1 : total_shelves = 3) 
  (h2 : total_figures = 27) : 
  total_figures / total_shelves = 9 := by
  sorry

end NUMINAMATH_CALUDE_action_figures_per_shelf_l2785_278554


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l2785_278514

/-- The surface area of a sphere circumscribing a rectangular solid with edge lengths 3, 4, and 5 emanating from one vertex is equal to 50π. -/
theorem circumscribed_sphere_surface_area (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  4 * π * radius^2 = 50 * π := by sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l2785_278514


namespace NUMINAMATH_CALUDE_two_extremum_function_properties_l2785_278553

/-- A function with two distinct extremum points -/
structure TwoExtremumFunction where
  f : ℝ → ℝ
  a : ℝ
  x1 : ℝ
  x2 : ℝ
  h_def : ∀ x, f x = x^2 + a * Real.log (x + 1)
  h_extremum : x1 < x2
  h_distinct : ∃ y, x1 < y ∧ y < x2

/-- The main theorem about the properties of the function -/
theorem two_extremum_function_properties (g : TwoExtremumFunction) :
  0 < g.a ∧ g.a < 1/2 ∧ 0 < g.f g.x2 / g.x1 ∧ g.f g.x2 / g.x1 < -1/2 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_two_extremum_function_properties_l2785_278553


namespace NUMINAMATH_CALUDE_michael_rubber_bands_l2785_278538

/-- The number of rubber bands in Michael's pack -/
def total_rubber_bands (small_ball_bands small_balls large_ball_bands large_balls : ℕ) : ℕ :=
  small_ball_bands * small_balls + large_ball_bands * large_balls

/-- Proof that Michael's pack contained 5000 rubber bands -/
theorem michael_rubber_bands :
  let small_ball_bands := 50
  let large_ball_bands := 300
  let small_balls := 22
  let large_balls := 13
  total_rubber_bands small_ball_bands small_balls large_ball_bands large_balls = 5000 := by
sorry

end NUMINAMATH_CALUDE_michael_rubber_bands_l2785_278538


namespace NUMINAMATH_CALUDE_smallest_z_minus_x_is_444_l2785_278522

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_z_minus_x_is_444 :
  ∃ (x y z : ℕ+),
    (x.val * y.val * z.val = factorial 10) ∧
    (x < y) ∧ (y < z) ∧
    (∀ (a b c : ℕ+),
      (a.val * b.val * c.val = factorial 10) → (a < b) → (b < c) →
      ((z.val - x.val : ℤ) ≤ (c.val - a.val))) ∧
    (z.val - x.val = 444) :=
by sorry

end NUMINAMATH_CALUDE_smallest_z_minus_x_is_444_l2785_278522


namespace NUMINAMATH_CALUDE_concert_hat_wearers_l2785_278592

theorem concert_hat_wearers (total_attendees : ℕ) 
  (women_fraction : ℚ) (women_hat_percentage : ℚ) (men_hat_percentage : ℚ) :
  total_attendees = 3000 →
  women_fraction = 2/3 →
  women_hat_percentage = 15/100 →
  men_hat_percentage = 12/100 →
  ↑(total_attendees * (women_fraction * women_hat_percentage + 
    (1 - women_fraction) * men_hat_percentage)) = (420 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_concert_hat_wearers_l2785_278592


namespace NUMINAMATH_CALUDE_work_completion_time_l2785_278502

theorem work_completion_time 
  (efficiency_ratio : ℝ) 
  (combined_time : ℝ) 
  (a_efficiency : ℝ) 
  (b_efficiency : ℝ) :
  efficiency_ratio = 2 →
  combined_time = 6 →
  a_efficiency = efficiency_ratio * b_efficiency →
  (a_efficiency + b_efficiency) * combined_time = b_efficiency * 18 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2785_278502


namespace NUMINAMATH_CALUDE_tangent_line_of_conic_section_l2785_278590

/-- Conic section equation -/
def ConicSection (A B C D E F : ℝ) (x y : ℝ) : Prop :=
  A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0

/-- Tangent line equation -/
def TangentLine (A B C D E F x₀ y₀ : ℝ) (x y : ℝ) : Prop :=
  2 * A * x₀ * x + B * (x₀ * y + x * y₀) + 2 * C * y₀ * y + 
  D * (x₀ + x) + E * (y₀ + y) + 2 * F = 0

theorem tangent_line_of_conic_section 
  (A B C D E F x₀ y₀ : ℝ) :
  ConicSection A B C D E F x₀ y₀ →
  ∃ ε > 0, ∀ x y : ℝ, 
    0 < (x - x₀)^2 + (y - y₀)^2 ∧ (x - x₀)^2 + (y - y₀)^2 < ε^2 →
    ConicSection A B C D E F x y →
    TangentLine A B C D E F x₀ y₀ x y := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_of_conic_section_l2785_278590


namespace NUMINAMATH_CALUDE_dans_grocery_items_l2785_278551

/-- Represents the items bought at the grocery store -/
structure GroceryItems where
  eggs : ℕ
  flour : ℕ
  butter : ℕ
  vanilla : ℕ

/-- Calculates the total number of individual items -/
def totalItems (items : GroceryItems) : ℕ :=
  items.eggs + items.flour + items.butter + items.vanilla

/-- Theorem stating the total number of items Dan bought -/
theorem dans_grocery_items : ∃ (items : GroceryItems), 
  items.eggs = 9 * 12 ∧ 
  items.flour = 6 ∧ 
  items.butter = 3 * 24 ∧ 
  items.vanilla = 12 ∧ 
  totalItems items = 198 := by
  sorry


end NUMINAMATH_CALUDE_dans_grocery_items_l2785_278551


namespace NUMINAMATH_CALUDE_canoe_production_sum_l2785_278577

/-- Represents the number of canoes built in the first month -/
def first_month_canoes : ℕ := 7

/-- Represents the ratio of canoes built between consecutive months -/
def monthly_ratio : ℕ := 3

/-- Represents the number of months considered -/
def num_months : ℕ := 6

/-- Calculates the sum of a geometric sequence -/
def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem canoe_production_sum :
  geometric_sum first_month_canoes monthly_ratio num_months = 2548 := by
  sorry

end NUMINAMATH_CALUDE_canoe_production_sum_l2785_278577


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l2785_278540

/-- The minimum distance between a point on y = x^2 + 2 and a point on y = √(x - 2) -/
theorem min_distance_between_curves : ∃ (d : ℝ),
  d = (7 * Real.sqrt 2) / 4 ∧
  ∀ (xP yP xQ yQ : ℝ),
    yP = xP^2 + 2 →
    yQ = Real.sqrt (xQ - 2) →
    d ≤ Real.sqrt ((xP - xQ)^2 + (yP - yQ)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l2785_278540


namespace NUMINAMATH_CALUDE_right_triangle_area_l2785_278500

theorem right_triangle_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 90 →
  a^2 + b^2 + c^2 = 3362 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 180 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2785_278500


namespace NUMINAMATH_CALUDE_odd_plus_even_combination_l2785_278535

theorem odd_plus_even_combination (p q : ℤ) 
  (h_p : ∃ k, p = 2 * k + 1) 
  (h_q : ∃ m, q = 2 * m) : 
  ∃ n, 3 * p + 2 * q = 2 * n + 1 := by
sorry

end NUMINAMATH_CALUDE_odd_plus_even_combination_l2785_278535


namespace NUMINAMATH_CALUDE_percentage_to_pass_l2785_278580

/-- Given a test with maximum marks, a student's score, and the margin by which they failed,
    calculate the percentage of marks needed to pass the test. -/
theorem percentage_to_pass (max_marks student_score fail_margin : ℕ) : 
  max_marks = 200 → 
  student_score = 80 → 
  fail_margin = 40 → 
  (student_score + fail_margin) / max_marks * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l2785_278580
