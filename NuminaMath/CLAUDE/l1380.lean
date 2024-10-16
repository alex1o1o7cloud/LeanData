import Mathlib

namespace NUMINAMATH_CALUDE_nine_possible_values_for_E_l1380_138066

def is_digit (n : ℕ) : Prop := n < 10

theorem nine_possible_values_for_E :
  ∀ (A B C D E : ℕ),
    is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ is_digit E →
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E →
    A + B = E →
    (C + D = E ∨ C + D = E + 10) →
    ∃! (count : ℕ), count = 9 ∧ 
      ∃ (possible_E : Finset ℕ), 
        possible_E.card = count ∧
        (∀ e, e ∈ possible_E ↔ 
          ∃ (A' B' C' D' : ℕ),
            is_digit A' ∧ is_digit B' ∧ is_digit C' ∧ is_digit D' ∧ is_digit e ∧
            A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ A' ≠ e ∧
            B' ≠ C' ∧ B' ≠ D' ∧ B' ≠ e ∧
            C' ≠ D' ∧ C' ≠ e ∧
            D' ≠ e ∧
            A' + B' = e ∧
            (C' + D' = e ∨ C' + D' = e + 10)) :=
by
  sorry

end NUMINAMATH_CALUDE_nine_possible_values_for_E_l1380_138066


namespace NUMINAMATH_CALUDE_ruler_cost_l1380_138003

theorem ruler_cost (total_spent : ℕ) (notebook_cost : ℕ) (num_pencils : ℕ) (pencil_cost : ℕ) 
  (h1 : total_spent = 74)
  (h2 : notebook_cost = 35)
  (h3 : num_pencils = 3)
  (h4 : pencil_cost = 7) :
  total_spent - (notebook_cost + num_pencils * pencil_cost) = 18 := by
  sorry

end NUMINAMATH_CALUDE_ruler_cost_l1380_138003


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1380_138073

theorem other_root_of_quadratic (a : ℝ) : 
  (3 : ℝ) ^ 2 - a * 3 - 2 * a = 0 → 
  ((-6 / 5 : ℝ) ^ 2 - a * (-6 / 5) - 2 * a = 0) ∧ 
  (3 + (-6 / 5) : ℝ) = a ∧ 
  (3 * (-6 / 5) : ℝ) = -2 * a := by
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1380_138073


namespace NUMINAMATH_CALUDE_total_steps_three_days_l1380_138026

def steps_day1 : ℕ := 200 + 300

def steps_day2 (day1 : ℕ) : ℕ := 2 * day1

def steps_day3 (day2 : ℕ) : ℕ := day2 + 100

theorem total_steps_three_days :
  steps_day1 + steps_day2 steps_day1 + steps_day3 (steps_day2 steps_day1) = 2600 :=
by sorry

end NUMINAMATH_CALUDE_total_steps_three_days_l1380_138026


namespace NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l1380_138055

-- Define an H function
def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

-- Define a strictly increasing function
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Theorem: A function is an H function if and only if it is strictly increasing
theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ strictly_increasing f :=
sorry

end NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l1380_138055


namespace NUMINAMATH_CALUDE_scissors_in_drawer_l1380_138099

theorem scissors_in_drawer (initial_scissors : ℕ) : initial_scissors = 54 →
  initial_scissors + 22 = 76 := by
  sorry

end NUMINAMATH_CALUDE_scissors_in_drawer_l1380_138099


namespace NUMINAMATH_CALUDE_union_A_B_when_a_neg_four_complement_A_intersect_B_eq_B_l1380_138075

-- Define the sets A and B
def A : Set ℝ := {x | (1 - 2*x) / (x - 3) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a ≤ 0}

-- Theorem 1: Union of A and B when a = -4
theorem union_A_B_when_a_neg_four :
  A ∪ B (-4) = {x : ℝ | -2 ≤ x ∧ x < 3} := by sorry

-- Theorem 2: Condition for (CᵣA) ∩ B = B
theorem complement_A_intersect_B_eq_B (a : ℝ) :
  (Aᶜ ∩ B a = B a) ↔ a > -1/4 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_neg_four_complement_A_intersect_B_eq_B_l1380_138075


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l1380_138045

/-- The minimum distance from a point on the line 5x + 12y = 60 to the origin (0, 0) is 60/13 -/
theorem min_distance_to_origin (x y : ℝ) : 
  5 * x + 12 * y = 60 → 
  (∃ (d : ℝ), d = 60 / 13 ∧ 
    ∀ (p : ℝ × ℝ), p.1 * 5 + p.2 * 12 = 60 → 
      d ≤ Real.sqrt (p.1^2 + p.2^2)) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l1380_138045


namespace NUMINAMATH_CALUDE_nested_abs_ratio_values_l1380_138082

/-- Recursive function representing nested absolute value operations -/
def nestedAbs (n : ℕ) (x y : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => |nestedAbs n x y - y|

/-- The equation condition from the problem -/
def equationCondition (x y : ℝ) : Prop :=
  nestedAbs 2019 x y = nestedAbs 2019 y x

/-- The theorem statement -/
theorem nested_abs_ratio_values (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : equationCondition x y) :
  x / y = 1/3 ∨ x / y = 1 ∨ x / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_nested_abs_ratio_values_l1380_138082


namespace NUMINAMATH_CALUDE_mary_stools_chopped_l1380_138060

/-- The number of sticks of wood produced by chopping up a chair -/
def sticks_per_chair : ℕ := 6

/-- The number of sticks of wood produced by chopping up a table -/
def sticks_per_table : ℕ := 9

/-- The number of sticks of wood produced by chopping up a stool -/
def sticks_per_stool : ℕ := 2

/-- The number of sticks of wood Mary needs to burn per hour to stay warm -/
def sticks_per_hour : ℕ := 5

/-- The number of chairs Mary chopped up -/
def chairs_chopped : ℕ := 18

/-- The number of tables Mary chopped up -/
def tables_chopped : ℕ := 6

/-- The number of hours Mary can keep warm -/
def hours_warm : ℕ := 34

/-- The number of stools Mary chopped up -/
def stools_chopped : ℕ := 4

theorem mary_stools_chopped :
  stools_chopped * sticks_per_stool + 
  chairs_chopped * sticks_per_chair + 
  tables_chopped * sticks_per_table = 
  hours_warm * sticks_per_hour :=
by sorry

end NUMINAMATH_CALUDE_mary_stools_chopped_l1380_138060


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l1380_138096

theorem quadratic_form_equivalence :
  ∀ x : ℝ, x^2 + 2*x - 2 = (x + 1)^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l1380_138096


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l1380_138032

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y + 5 = 0
def line2 (x y : ℝ) (b c : ℝ) : Prop := 6 * x + b * y + c = 0

-- Define the distance between lines
def distance_between_lines (b c : ℝ) : ℝ := 3

-- Define the parallelism condition
def parallel_lines (b : ℝ) : Prop := b = 8

theorem parallel_lines_distance (b c : ℝ) :
  parallel_lines b → distance_between_lines b c = 3 →
  (b + c = -12 ∨ b + c = 48) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l1380_138032


namespace NUMINAMATH_CALUDE_octal_subtraction_l1380_138069

/-- Represents a number in base 8 --/
def OctalNum := ℕ

/-- Converts a natural number to its octal representation --/
def toOctal (n : ℕ) : OctalNum :=
  sorry

/-- Performs subtraction in base 8 --/
def octalSub (a b : OctalNum) : OctalNum :=
  sorry

theorem octal_subtraction :
  octalSub (toOctal 126) (toOctal 57) = toOctal 47 :=
sorry

end NUMINAMATH_CALUDE_octal_subtraction_l1380_138069


namespace NUMINAMATH_CALUDE_binomial_coefficient_seven_two_l1380_138024

theorem binomial_coefficient_seven_two : 
  Nat.choose 7 2 = 21 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_seven_two_l1380_138024


namespace NUMINAMATH_CALUDE_min_side_length_l1380_138089

theorem min_side_length (PQ PR SR SQ : ℝ) (h1 : PQ = 7) (h2 : PR = 15) (h3 : SR = 10) (h4 : SQ = 25) :
  ∀ QR : ℝ, (QR > PR - PQ ∧ QR > SQ - SR) → QR ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_min_side_length_l1380_138089


namespace NUMINAMATH_CALUDE_mediant_inequality_l1380_138040

theorem mediant_inequality (a b p q r s : ℕ) 
  (h1 : q * r - p * s = 1) 
  (h2 : (p : ℚ) / q < (a : ℚ) / b) 
  (h3 : (a : ℚ) / b < (r : ℚ) / s) : 
  b ≥ q + s := by
  sorry

end NUMINAMATH_CALUDE_mediant_inequality_l1380_138040


namespace NUMINAMATH_CALUDE_distance_to_focus_of_parabola_l1380_138037

/-- The distance from a point on a parabola to its focus -/
theorem distance_to_focus_of_parabola (y : ℝ) :
  y^2 = 8 →  -- Point (1, y) satisfies the parabola equation
  Real.sqrt ((1 - 2)^2 + y^2) = 3 :=  -- Distance from (1, y) to focus (2, 0) is 3
by sorry

end NUMINAMATH_CALUDE_distance_to_focus_of_parabola_l1380_138037


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1380_138008

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, (1 < x ∧ x < 4) ↔ (a * x^2 + b * x - 2 > 0)) → 
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1380_138008


namespace NUMINAMATH_CALUDE_last_two_digits_2018_power_2018_base_7_l1380_138084

theorem last_two_digits_2018_power_2018_base_7 : 
  (2018^2018 : ℕ) % 49 = 32 :=
sorry

end NUMINAMATH_CALUDE_last_two_digits_2018_power_2018_base_7_l1380_138084


namespace NUMINAMATH_CALUDE_hundredth_odd_and_following_even_l1380_138047

theorem hundredth_odd_and_following_even :
  (∃ n : ℕ, n = 100 ∧ 2 * n - 1 = 199) ∧
  (∃ m : ℕ, m = 200 ∧ m = 199 + 1 ∧ Even m) := by
  sorry

end NUMINAMATH_CALUDE_hundredth_odd_and_following_even_l1380_138047


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1380_138070

theorem quadratic_max_value :
  ∃ (M : ℝ), M = 34 ∧ ∀ (q : ℝ), -3 * q^2 + 18 * q + 7 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1380_138070


namespace NUMINAMATH_CALUDE_abc_sum_l1380_138015

theorem abc_sum (a b c : ℕ+) 
  (eq1 : a * b + c = 55)
  (eq2 : b * c + a = 55)
  (eq3 : a * c + b = 55) :
  a + b + c = 40 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_l1380_138015


namespace NUMINAMATH_CALUDE_distance_between_given_lines_l1380_138035

def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 6 = 0

def line2 (x y : ℝ) : Prop := 6 * x + 8 * y + 3 = 0

def are_parallel (l1 l2 : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), l1 x y ↔ l2 (k * x) (k * y)

def distance_between_lines (l1 l2 : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem distance_between_given_lines :
  are_parallel line1 line2 →
  distance_between_lines line1 line2 = 1.5 := by sorry

end NUMINAMATH_CALUDE_distance_between_given_lines_l1380_138035


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1380_138097

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧ 
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1380_138097


namespace NUMINAMATH_CALUDE_dana_hourly_wage_l1380_138019

/-- Given a person who worked for a certain number of hours and earned a total amount,
    calculate their hourly wage. -/
def hourly_wage (hours_worked : ℕ) (total_earned : ℕ) : ℚ :=
  total_earned / hours_worked

theorem dana_hourly_wage :
  hourly_wage 22 286 = 13 := by sorry

end NUMINAMATH_CALUDE_dana_hourly_wage_l1380_138019


namespace NUMINAMATH_CALUDE_half_of_one_point_six_million_l1380_138012

theorem half_of_one_point_six_million (x : ℝ) : 
  x = 1.6 * (10 : ℝ)^6 → (1/2 : ℝ) * x = 8 * (10 : ℝ)^5 := by
  sorry

end NUMINAMATH_CALUDE_half_of_one_point_six_million_l1380_138012


namespace NUMINAMATH_CALUDE_john_soap_cost_l1380_138067

/-- The amount of money spent on soap given the number of bars, weight per bar, and price per pound -/
def soap_cost (num_bars : ℕ) (weight_per_bar : ℚ) (price_per_pound : ℚ) : ℚ :=
  num_bars * weight_per_bar * price_per_pound

/-- Theorem stating that John spent $15 on soap -/
theorem john_soap_cost : soap_cost 20 (3/2) (1/2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_john_soap_cost_l1380_138067


namespace NUMINAMATH_CALUDE_sum_of_three_squares_149_l1380_138051

theorem sum_of_three_squares_149 : ∃ (a b c : ℕ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b + c = 21 ∧
  a^2 + b^2 + c^2 = 149 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_149_l1380_138051


namespace NUMINAMATH_CALUDE_toy_factory_production_l1380_138039

/-- Represents the production constraints and goal for a toy factory --/
theorem toy_factory_production :
  ∃ (x y : ℕ),
    15 * x + 10 * y ≤ 450 ∧  -- Labor constraint
    20 * x + 5 * y ≤ 400 ∧   -- Raw material constraint
    80 * x + 45 * y = 2200   -- Total selling price
    := by sorry

end NUMINAMATH_CALUDE_toy_factory_production_l1380_138039


namespace NUMINAMATH_CALUDE_no_right_triangle_with_specific_medians_l1380_138054

theorem no_right_triangle_with_specific_medians : ∀ (m b a_leg b_leg : ℝ),
  a_leg > 0 → b_leg > 0 →
  (∃ (x y : ℝ), y = m * x + b) →  -- hypotenuse parallel to y = mx + b
  (∃ (x y : ℝ), y = 2 * x + 1) →  -- one median on y = 2x + 1
  (∃ (x y : ℝ), y = 5 * x + 2) →  -- another median on y = 5x + 2
  ¬ (
    -- Right triangle condition
    a_leg^2 + b_leg^2 = (a_leg^2 + b_leg^2) ∧
    -- Hypotenuse parallel to y = mx + b
    m = -b_leg / a_leg ∧
    -- One median on y = 2x + 1
    (2 * b_leg / a_leg = 2 ∨ b_leg / (2 * a_leg) = 2) ∧
    -- Another median on y = 5x + 2
    (2 * b_leg / a_leg = 5 ∨ b_leg / (2 * a_leg) = 5)
  ) := by
  sorry

end NUMINAMATH_CALUDE_no_right_triangle_with_specific_medians_l1380_138054


namespace NUMINAMATH_CALUDE_weight_of_brand_a_l1380_138095

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 800

/-- The ratio of brand 'a' to brand 'b' in the mixture -/
def ratio_a : ℝ := 3
def ratio_b : ℝ := 2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3440

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 580

theorem weight_of_brand_a :
  weight_a * (ratio_a / (ratio_a + ratio_b)) * total_volume +
  weight_b * (ratio_b / (ratio_a + ratio_b)) * total_volume = total_weight :=
by sorry

end NUMINAMATH_CALUDE_weight_of_brand_a_l1380_138095


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1380_138062

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 - y^2 = 12) : 
  |x - y| = 2 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1380_138062


namespace NUMINAMATH_CALUDE_speed_conversion_l1380_138087

-- Define the conversion factors
def km_to_m : ℚ := 1000
def hour_to_sec : ℚ := 3600

-- Define the given speed in km/h
def speed_kmh : ℚ := 72

-- Define the conversion function
def kmh_to_ms (speed : ℚ) : ℚ :=
  speed * km_to_m / hour_to_sec

-- Theorem statement
theorem speed_conversion :
  kmh_to_ms speed_kmh = 20 := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l1380_138087


namespace NUMINAMATH_CALUDE_dot_no_line_count_l1380_138093

/-- Represents an alphabet with letters containing dots and straight lines -/
structure Alphabet :=
  (total : ℕ)
  (both : ℕ)
  (line_no_dot : ℕ)
  (has_dot_or_line : total = both + line_no_dot + (total - (both + line_no_dot)))

/-- The number of letters containing a dot but not a straight line -/
def dot_no_line (α : Alphabet) : ℕ :=
  α.total - (α.both + α.line_no_dot)

theorem dot_no_line_count (α : Alphabet) 
  (h1 : α.total = 40)
  (h2 : α.both = 11)
  (h3 : α.line_no_dot = 24) :
  dot_no_line α = 5 := by
  sorry

end NUMINAMATH_CALUDE_dot_no_line_count_l1380_138093


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1380_138071

theorem quadratic_root_difference : 
  let a : ℝ := 5 + 3 * Real.sqrt 2
  let b : ℝ := 2 + Real.sqrt 2
  let c : ℝ := -1
  let discriminant := b^2 - 4*a*c
  let root_difference := (2 * Real.sqrt discriminant) / (2 * a)
  root_difference = (2 * Real.sqrt (24 * Real.sqrt 2 + 180)) / 7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1380_138071


namespace NUMINAMATH_CALUDE_rudy_running_time_l1380_138056

-- Define the running segments
def segment1_distance : ℝ := 5
def segment1_rate : ℝ := 10
def segment2_distance : ℝ := 4
def segment2_rate : ℝ := 9.5
def segment3_distance : ℝ := 3
def segment3_rate : ℝ := 8.5
def segment4_distance : ℝ := 2
def segment4_rate : ℝ := 12

-- Define the rest times
def rest1 : ℝ := 15
def rest2 : ℝ := 10
def rest3 : ℝ := 5

-- Define the total time function
def total_time : ℝ :=
  segment1_distance * segment1_rate +
  segment2_distance * segment2_rate +
  segment3_distance * segment3_rate +
  segment4_distance * segment4_rate +
  rest1 + rest2 + rest3

-- Theorem statement
theorem rudy_running_time : total_time = 167.5 := by
  sorry

end NUMINAMATH_CALUDE_rudy_running_time_l1380_138056


namespace NUMINAMATH_CALUDE_factorial_difference_l1380_138016

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l1380_138016


namespace NUMINAMATH_CALUDE_triangle_problem_l1380_138053

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧
  -- Given condition
  2 * b * Real.cos C = a * Real.cos C + c * Real.cos A →
  -- Part 1: Prove C = π/3
  C = π/3 ∧
  -- Part 2: Given additional conditions
  (b = 2 ∧ c = Real.sqrt 7 →
    -- Prove a = 3
    a = 3 ∧
    -- Prove area = 3√3/2
    1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1380_138053


namespace NUMINAMATH_CALUDE_jeff_tennis_games_l1380_138043

/-- Calculates the number of games Jeff wins in tennis given the playing time, scoring rate, and points needed to win a match. -/
theorem jeff_tennis_games (playing_time : ℕ) (scoring_rate : ℕ) (points_per_match : ℕ) : 
  playing_time = 120 ∧ scoring_rate = 5 ∧ points_per_match = 8 → 
  (playing_time / scoring_rate) / points_per_match = 3 := by sorry

end NUMINAMATH_CALUDE_jeff_tennis_games_l1380_138043


namespace NUMINAMATH_CALUDE_revenue_decrease_l1380_138057

theorem revenue_decrease (T C : ℝ) (h1 : T > 0) (h2 : C > 0) :
  let new_tax := 0.8 * T
  let new_consumption := 1.1 * C
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  (original_revenue - new_revenue) / original_revenue = 0.12 :=
by sorry

end NUMINAMATH_CALUDE_revenue_decrease_l1380_138057


namespace NUMINAMATH_CALUDE_first_positive_term_is_26_l1380_138041

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 4 * n - 102

-- Define the property of being the first positive term
def is_first_positive (k : ℕ) : Prop :=
  a k > 0 ∧ ∀ m : ℕ, m < k → a m ≤ 0

-- Theorem statement
theorem first_positive_term_is_26 : is_first_positive 26 := by
  sorry

end NUMINAMATH_CALUDE_first_positive_term_is_26_l1380_138041


namespace NUMINAMATH_CALUDE_batsman_average_17th_inning_l1380_138064

def batsman_average (total_innings : ℕ) (last_inning_score : ℕ) (average_increase : ℚ) : ℚ :=
  (total_innings - 1 : ℚ) * (average_increase + last_inning_score / total_innings) + last_inning_score / total_innings

theorem batsman_average_17th_inning :
  batsman_average 17 92 3 = 44 := by sorry

end NUMINAMATH_CALUDE_batsman_average_17th_inning_l1380_138064


namespace NUMINAMATH_CALUDE_rational_fraction_implies_integer_sum_squares_l1380_138004

theorem rational_fraction_implies_integer_sum_squares (a b c : ℕ+) 
  (h : ∃ (q : ℚ), (a.val : ℝ) * Real.sqrt 3 + b.val = q * ((b.val : ℝ) * Real.sqrt 3 + c.val)) :
  ∃ (n : ℤ), (a.val^2 + b.val^2 + c.val^2 : ℝ) / (a.val + b.val + c.val) = n := by
sorry

end NUMINAMATH_CALUDE_rational_fraction_implies_integer_sum_squares_l1380_138004


namespace NUMINAMATH_CALUDE_problem_statement_l1380_138052

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a + b) * Real.sqrt (a * b) = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * Real.sqrt (x * y) = 1 → 1 / x^3 + 1 / y^3 ≥ 1 / a^3 + 1 / b^3) ∧
  (1 / a^3 + 1 / b^3 = 4 * Real.sqrt 2) ∧
  ¬∃ (p q : ℝ), p > 0 ∧ q > 0 ∧ 1 / (2 * p) + 1 / (3 * q) = Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_problem_statement_l1380_138052


namespace NUMINAMATH_CALUDE_triangle_problem_l1380_138042

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  -- Given conditions
  t.c = 2 ∧ 
  t.A = π / 3 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A) = Real.sqrt 3 / 2 →
  -- Conclusion
  t.a = Real.sqrt 3 ∧ 
  t.b = 1 ∧ 
  t.C = π / 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l1380_138042


namespace NUMINAMATH_CALUDE_wednesday_profit_l1380_138002

/-- The profit made by a beadshop over three days -/
def BeadshopProfit (total : ℝ) (monday : ℝ) (tuesday : ℝ) (wednesday : ℝ) : Prop :=
  total = 1200 ∧
  monday = (1/3) * total ∧
  tuesday = (1/4) * total ∧
  wednesday = total - monday - tuesday

/-- The profit made on Wednesday is $500 -/
theorem wednesday_profit (total monday tuesday wednesday : ℝ) :
  BeadshopProfit total monday tuesday wednesday →
  wednesday = 500 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_profit_l1380_138002


namespace NUMINAMATH_CALUDE_probability_at_least_one_girl_l1380_138013

theorem probability_at_least_one_girl (total : ℕ) (boys : ℕ) (girls : ℕ) (select : ℕ) :
  total = boys + girls →
  boys = 4 →
  girls = 2 →
  select = 3 →
  (1 - (Nat.choose boys select / Nat.choose total select : ℚ)) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_girl_l1380_138013


namespace NUMINAMATH_CALUDE_unique_equation_solution_l1380_138031

/-- A function that checks if a list of integers contains exactly the digits 1 to 9 --/
def isValidDigitList (lst : List Int) : Prop :=
  lst.length = 9 ∧ (∀ n, n ∈ lst → 1 ≤ n ∧ n ≤ 9) ∧ lst.toFinset.card = 9

/-- A function that converts a list of three integers to a three-digit number --/
def toThreeDigitNumber (a b c : Int) : Int :=
  100 * a + 10 * b + c

/-- A function that converts a list of two integers to a two-digit number --/
def toTwoDigitNumber (a b : Int) : Int :=
  10 * a + b

theorem unique_equation_solution :
  ∃! (digits : List Int),
    isValidDigitList digits ∧
    7 ∈ digits ∧
    let abc := toThreeDigitNumber (digits[0]!) (digits[1]!) (digits[2]!)
    let de := toTwoDigitNumber (digits[3]!) (digits[4]!)
    let f := digits[5]!
    let h := digits[8]!
    abc / de = f ∧ f = h - 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_equation_solution_l1380_138031


namespace NUMINAMATH_CALUDE_characterize_satisfying_functions_l1380_138083

/-- A function satisfying the given inequality condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ (x y u v : ℝ), x > 1 → y > 1 → u > 0 → v > 0 →
    f (x^u * y^v) ≤ f x^(1/(4*u)) * f y^(1/(4*v))

/-- The main theorem stating the form of functions satisfying the condition -/
theorem characterize_satisfying_functions :
  ∀ (f : ℝ → ℝ), (∀ x, x > 1 → f x > 1) →
    SatisfiesCondition f →
    ∃ (c : ℝ), c > 1 ∧ ∀ x, x > 1 → f x = c^(1/Real.log x) :=
by sorry

end NUMINAMATH_CALUDE_characterize_satisfying_functions_l1380_138083


namespace NUMINAMATH_CALUDE_unique_intersection_point_l1380_138061

theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, 
    (3 * p.1 - 2 * p.2 - 9 = 0) ∧
    (6 * p.1 + 4 * p.2 - 12 = 0) ∧
    (p.1 = 3) ∧
    (p.2 = -1) := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l1380_138061


namespace NUMINAMATH_CALUDE_orange_count_l1380_138033

theorem orange_count (initial : ℕ) (thrown_away : ℕ) (added : ℕ) : 
  initial = 5 → thrown_away = 2 → added = 28 → 
  initial - thrown_away + added = 31 :=
by sorry

end NUMINAMATH_CALUDE_orange_count_l1380_138033


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l1380_138059

/-- Given two lines intersecting at (3,3) with slopes 1/3 and 3, and a third line x + y = 12,
    the area of the triangle formed by their intersections is 8.625 -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    let line1 := fun x => (1/3) * x + 2  -- y = (1/3)x + 2
    let line2 := fun x => 3 * x - 6      -- y = 3x - 6
    let line3 := fun x => 12 - x         -- y = 12 - x
    let A := (3, 3)
    let B := (4.5, 7.5)  -- Intersection of line2 and line3
    let C := (7.5, 4.5)  -- Intersection of line1 and line3
    (line1 3 = 3 ∧ line2 3 = 3) →  -- Lines intersect at (3,3)
    (∀ x, line3 x + x = 12) →      -- Third line equation
    area = 8.625

-- The proof is omitted
theorem triangle_area_proof : triangle_area 8.625 := by sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l1380_138059


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1380_138028

theorem repeating_decimal_sum : 
  (1 / 3 : ℚ) + (4 / 99 : ℚ) + (5 / 999 : ℚ) = (14 / 37 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1380_138028


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1380_138044

theorem quadratic_inequality_equivalence (x : ℝ) :
  x^2 - 9*x + 14 < 0 ↔ 2 < x ∧ x < 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1380_138044


namespace NUMINAMATH_CALUDE_stating_popsicle_count_l1380_138090

/-- The number of popsicles in a box with specific melting rate properties -/
def num_popsicles : ℕ := 6

/-- The melting rate factor between consecutive popsicles -/
def melting_rate_factor : ℕ := 2

/-- The relative melting rate of the last popsicle compared to the first -/
def last_to_first_rate : ℕ := 32

/-- 
Theorem stating that the number of popsicles in the box is 6, given the melting rate properties
-/
theorem popsicle_count :
  (melting_rate_factor ^ (num_popsicles - 1) = last_to_first_rate) →
  num_popsicles = 6 := by
sorry

end NUMINAMATH_CALUDE_stating_popsicle_count_l1380_138090


namespace NUMINAMATH_CALUDE_sum_of_three_cubes_not_2002_l1380_138036

theorem sum_of_three_cubes_not_2002 : ¬∃ (a b c : ℤ), a^3 + b^3 + c^3 = 2002 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_cubes_not_2002_l1380_138036


namespace NUMINAMATH_CALUDE_supplementary_angle_of_39_23_l1380_138058

-- Define the angle type with degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the supplementary angle function
def supplementaryAngle (a : Angle) : Angle :=
  let totalMinutes := 180 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- Theorem statement
theorem supplementary_angle_of_39_23 :
  let a : Angle := ⟨39, 23⟩
  supplementaryAngle a = ⟨140, 37⟩ := by
  sorry

end NUMINAMATH_CALUDE_supplementary_angle_of_39_23_l1380_138058


namespace NUMINAMATH_CALUDE_concert_ticket_sales_l1380_138034

/-- Proves that the total number of tickets sold is 45 given the specified conditions --/
theorem concert_ticket_sales : 
  let ticket_price : ℕ := 20
  let first_group_size : ℕ := 10
  let second_group_size : ℕ := 20
  let first_group_discount : ℚ := 40 / 100
  let second_group_discount : ℚ := 15 / 100
  let total_revenue : ℕ := 760
  ∃ (full_price_tickets : ℕ),
    (first_group_size * (ticket_price * (1 - first_group_discount)).floor + 
     second_group_size * (ticket_price * (1 - second_group_discount)).floor + 
     full_price_tickets * ticket_price = total_revenue) ∧
    (first_group_size + second_group_size + full_price_tickets = 45) :=
by sorry

end NUMINAMATH_CALUDE_concert_ticket_sales_l1380_138034


namespace NUMINAMATH_CALUDE_cubic_factorization_l1380_138088

theorem cubic_factorization (m : ℝ) : m^3 - 9*m = m*(m+3)*(m-3) := by sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1380_138088


namespace NUMINAMATH_CALUDE_salary_comparison_l1380_138065

theorem salary_comparison (a b : ℝ) (h : a = 0.8 * b) :
  b = 1.25 * a := by sorry

end NUMINAMATH_CALUDE_salary_comparison_l1380_138065


namespace NUMINAMATH_CALUDE_antiderivative_derivative_l1380_138078

/-- The derivative of the antiderivative is equal to the original function -/
theorem antiderivative_derivative (x : ℝ) :
  let f : ℝ → ℝ := λ x => (2*x^3 + 3*x^2 + 3*x + 2) / ((x^2 + x + 1)*(x^2 + 1))
  let F : ℝ → ℝ := λ x => (1/2) * Real.log (abs (x^2 + x + 1)) +
                          (1/Real.sqrt 3) * Real.arctan ((2*x + 1)/Real.sqrt 3) +
                          (1/2) * Real.log (abs (x^2 + 1)) +
                          Real.arctan x
  (deriv F) x = f x :=
by sorry

end NUMINAMATH_CALUDE_antiderivative_derivative_l1380_138078


namespace NUMINAMATH_CALUDE_race_length_l1380_138001

/-- The race between Nicky and Cristina -/
def race (cristina_speed nicky_speed : ℝ) (head_start catch_up_time : ℝ) : Prop :=
  let nicky_distance := nicky_speed * catch_up_time
  let cristina_time := catch_up_time - head_start
  let cristina_distance := cristina_speed * cristina_time
  nicky_distance = cristina_distance ∧ nicky_distance = 90

/-- The race length is 90 meters -/
theorem race_length :
  race 5 3 12 30 :=
by
  sorry

end NUMINAMATH_CALUDE_race_length_l1380_138001


namespace NUMINAMATH_CALUDE_four_digit_number_problem_l1380_138086

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem four_digit_number_problem (N : ℕ) 
  (h1 : is_four_digit N) 
  (h2 : (70000 + N) - (10 * N + 7) = 53208) : 
  N = 1865 := by sorry

end NUMINAMATH_CALUDE_four_digit_number_problem_l1380_138086


namespace NUMINAMATH_CALUDE_kim_sweater_count_l1380_138091

/-- The number of sweaters Kim knit on Monday -/
def monday_sweaters : ℕ := 8

/-- The total number of sweaters Kim knit in the week -/
def total_sweaters : ℕ := 34

/-- The maximum number of sweaters Kim can knit in a day -/
def max_daily_sweaters : ℕ := 10

theorem kim_sweater_count :
  monday_sweaters ≤ max_daily_sweaters ∧
  monday_sweaters +
  (monday_sweaters + 2) +
  ((monday_sweaters + 2) - 4) +
  ((monday_sweaters + 2) - 4) +
  (monday_sweaters / 2) = total_sweaters :=
by sorry

end NUMINAMATH_CALUDE_kim_sweater_count_l1380_138091


namespace NUMINAMATH_CALUDE_lola_cupcakes_count_l1380_138074

/-- The number of mini cupcakes Lola baked -/
def lola_cupcakes : ℕ := sorry

/-- The number of pop tarts Lola baked -/
def lola_poptarts : ℕ := 10

/-- The number of blueberry pies Lola baked -/
def lola_pies : ℕ := 8

/-- The number of mini cupcakes Lulu made -/
def lulu_cupcakes : ℕ := 16

/-- The number of pop tarts Lulu made -/
def lulu_poptarts : ℕ := 12

/-- The number of blueberry pies Lulu made -/
def lulu_pies : ℕ := 14

/-- The total number of pastries made by Lola and Lulu -/
def total_pastries : ℕ := 73

theorem lola_cupcakes_count : lola_cupcakes = 13 := by
  sorry

end NUMINAMATH_CALUDE_lola_cupcakes_count_l1380_138074


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1380_138027

theorem cos_alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α + π / 6) = 3 / 5) : 
  Real.cos α = (3 * Real.sqrt 3 + 4) / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1380_138027


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l1380_138046

/-- A polynomial satisfies P(x+1) = P(x) + 2x + 1 for all x if and only if it is of the form x^2 + c for some constant c. -/
theorem polynomial_functional_equation (P : ℝ → ℝ) :
  (∀ x, P (x + 1) = P x + 2 * x + 1) ↔
  (∃ c, ∀ x, P x = x^2 + c) :=
sorry

end NUMINAMATH_CALUDE_polynomial_functional_equation_l1380_138046


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1380_138025

/-- Given a principal amount and an interest rate, if increasing the rate by 4% for 2 years
    yields Rs. 60 more in interest, then the principal amount is Rs. 750. -/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (k : R > 0) :
  (P * (R + 4) * 2) / 100 = (P * R * 2) / 100 + 60 → P = 750 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1380_138025


namespace NUMINAMATH_CALUDE_ellipse_iff_range_l1380_138063

/-- The equation of an ellipse with parameter m -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (2 - m) + y^2 / (m + 1) = 1

/-- The range of m for which the equation represents an ellipse -/
def ellipse_range (m : ℝ) : Prop :=
  (m > -1 ∧ m < 1/2) ∨ (m > 1/2 ∧ m < 2)

/-- Theorem stating that the equation represents an ellipse if and only if m is in the specified range -/
theorem ellipse_iff_range (m : ℝ) : is_ellipse m ↔ ellipse_range m := by
  sorry

end NUMINAMATH_CALUDE_ellipse_iff_range_l1380_138063


namespace NUMINAMATH_CALUDE_heart_op_ratio_l1380_138010

def heart_op (n m : ℕ) : ℕ := n^3 * m^2

theorem heart_op_ratio : 
  (heart_op 3 5 : ℚ) / (heart_op 5 3) = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_heart_op_ratio_l1380_138010


namespace NUMINAMATH_CALUDE_aquarium_fish_count_l1380_138007

theorem aquarium_fish_count (total : ℕ) (blue orange green : ℕ) : 
  total = 80 ∧ 
  blue = total / 2 ∧ 
  orange = blue - 15 ∧ 
  total = blue + orange + green → 
  green = 15 := by
sorry

end NUMINAMATH_CALUDE_aquarium_fish_count_l1380_138007


namespace NUMINAMATH_CALUDE_number_of_divisor_pairs_l1380_138072

theorem number_of_divisor_pairs : ∃ (count : ℕ), count = 480 ∧
  count = (Finset.filter (fun p : ℕ × ℕ => 
    (p.1 * p.2 ∣ (2008 * 2009 * 2010)) ∧ 
    p.1 > 0 ∧ p.2 > 0
  ) (Finset.product (Finset.range (2008 * 2009 * 2010 + 1)) (Finset.range (2008 * 2009 * 2010 + 1)))).card :=
by
  sorry

#check number_of_divisor_pairs

end NUMINAMATH_CALUDE_number_of_divisor_pairs_l1380_138072


namespace NUMINAMATH_CALUDE_inequality_proof_l1380_138014

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_two : a + b + c = 2) :
  (1 / (1 + a*b)) + (1 / (1 + b*c)) + (1 / (1 + c*a)) ≥ 27/13 ∧
  ((1 / (1 + a*b)) + (1 / (1 + b*c)) + (1 / (1 + c*a)) = 27/13 ↔ a = 2/3 ∧ b = 2/3 ∧ c = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1380_138014


namespace NUMINAMATH_CALUDE_proposition_intersection_l1380_138092

-- Define the propositions p and q
def p (a : ℝ) : Prop := a^2 - 5*a ≥ 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 4 ≠ 0

-- Define the range of a
def range_a (a : ℝ) : Prop := -4 < a ∧ a ≤ 0

-- Theorem statement
theorem proposition_intersection (a : ℝ) : p a ∧ q a ↔ range_a a := by sorry

end NUMINAMATH_CALUDE_proposition_intersection_l1380_138092


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1380_138009

universe u

def U : Finset ℕ := {1, 2, 3, 4}
def A : Finset ℕ := {1, 2}
def B : Finset ℕ := {2, 3}

theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1380_138009


namespace NUMINAMATH_CALUDE_sum_first_seven_primes_mod_eighth_prime_l1380_138022

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]
def eighth_prime : Nat := 19

theorem sum_first_seven_primes_mod_eighth_prime : 
  (first_seven_primes.sum) % eighth_prime = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_seven_primes_mod_eighth_prime_l1380_138022


namespace NUMINAMATH_CALUDE_boys_together_arrangements_l1380_138018

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of students -/
def total_students : ℕ := 5

/-- The number of boys -/
def num_boys : ℕ := 3

/-- The number of girls -/
def num_girls : ℕ := 2

/-- The number of arrangements where all boys stand together -/
def arrangements_with_boys_together : ℕ := factorial num_boys * factorial (total_students - num_boys + 1)

theorem boys_together_arrangements :
  arrangements_with_boys_together = 36 :=
sorry

end NUMINAMATH_CALUDE_boys_together_arrangements_l1380_138018


namespace NUMINAMATH_CALUDE_rectangle_toothpicks_l1380_138006

/-- Calculates the number of toothpicks needed to form a rectangle --/
def toothpicks_in_rectangle (length width : ℕ) : ℕ :=
  let horizontal_rows := width + 1
  let vertical_columns := length + 1
  horizontal_rows * length + vertical_columns * width

/-- Theorem: A rectangle with length 20 and width 10 requires 430 toothpicks --/
theorem rectangle_toothpicks :
  toothpicks_in_rectangle 20 10 = 430 := by
  sorry

#eval toothpicks_in_rectangle 20 10

end NUMINAMATH_CALUDE_rectangle_toothpicks_l1380_138006


namespace NUMINAMATH_CALUDE_min_value_cyclic_fraction_l1380_138049

theorem min_value_cyclic_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / b + b / c + c / a ≥ 3 ∧ 
  (a / b + b / c + c / a = 3 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_cyclic_fraction_l1380_138049


namespace NUMINAMATH_CALUDE_sufficient_condition_for_not_p_l1380_138011

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x ∈ Set.Icc 1 4, log_half x < 2*x + a

-- Theorem statement
theorem sufficient_condition_for_not_p (a : ℝ) :
  a < -11 → ∀ x ∈ Set.Icc 1 4, log_half x ≥ 2*x + a :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_not_p_l1380_138011


namespace NUMINAMATH_CALUDE_quadratic_roots_and_k_l1380_138021

/-- The quadratic equation x^2 - (k+2)x + 2k - 1 = 0 -/
def quadratic (k x : ℝ) : Prop :=
  x^2 - (k+2)*x + 2*k - 1 = 0

theorem quadratic_roots_and_k :
  (∀ k : ℝ, ∃ x y : ℝ, x ≠ y ∧ quadratic k x ∧ quadratic k y) ∧
  (∃ k : ℝ, quadratic k 3 ∧ quadratic k 1 ∧ k = 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_k_l1380_138021


namespace NUMINAMATH_CALUDE_homework_difference_is_two_l1380_138020

/-- The number of pages of reading homework Rachel has to complete -/
def reading_pages : ℕ := 2

/-- The number of pages of math homework Rachel has to complete -/
def math_pages : ℕ := 4

/-- The difference in pages between math and reading homework -/
def homework_difference : ℕ := math_pages - reading_pages

theorem homework_difference_is_two : homework_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_homework_difference_is_two_l1380_138020


namespace NUMINAMATH_CALUDE_cymbal_triangle_tambourine_sync_l1380_138017

theorem cymbal_triangle_tambourine_sync (cymbal_beats : Nat) (triangle_beats : Nat) (tambourine_beats : Nat)
  (h1 : cymbal_beats = 13)
  (h2 : triangle_beats = 17)
  (h3 : tambourine_beats = 19) :
  Nat.lcm (Nat.lcm cymbal_beats triangle_beats) tambourine_beats = 4199 := by
  sorry

end NUMINAMATH_CALUDE_cymbal_triangle_tambourine_sync_l1380_138017


namespace NUMINAMATH_CALUDE_ratio_of_squares_to_products_l1380_138081

theorem ratio_of_squares_to_products (x y z : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (h_sum : x + 2*y + 3*z = 0) : 
  (x^2 + y^2 + z^2) / (x*y + y*z + z*x) = -4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_squares_to_products_l1380_138081


namespace NUMINAMATH_CALUDE_sum_faces_edges_vertices_l1380_138030

/-- A rectangular prism is a three-dimensional shape with specific properties. -/
structure RectangularPrism where
  -- We don't need to define the specific properties here

/-- The number of faces in a rectangular prism -/
def num_faces (rp : RectangularPrism) : ℕ := 6

/-- The number of edges in a rectangular prism -/
def num_edges (rp : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (rp : RectangularPrism) : ℕ := 8

/-- The sum of faces, edges, and vertices in a rectangular prism is 26 -/
theorem sum_faces_edges_vertices (rp : RectangularPrism) :
  num_faces rp + num_edges rp + num_vertices rp = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_faces_edges_vertices_l1380_138030


namespace NUMINAMATH_CALUDE_popsicle_problem_l1380_138048

/-- Popsicle Making Problem -/
theorem popsicle_problem (total_money : ℕ) (mold_cost : ℕ) (stick_pack_cost : ℕ) 
  (juice_cost : ℕ) (total_sticks : ℕ) (remaining_sticks : ℕ) :
  total_money = 10 →
  mold_cost = 3 →
  stick_pack_cost = 1 →
  juice_cost = 2 →
  total_sticks = 100 →
  remaining_sticks = 40 →
  (total_money - mold_cost - stick_pack_cost) / juice_cost * 
    ((total_sticks - remaining_sticks) / ((total_money - mold_cost - stick_pack_cost) / juice_cost)) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_popsicle_problem_l1380_138048


namespace NUMINAMATH_CALUDE_smallest_prime_six_less_than_square_l1380_138079

theorem smallest_prime_six_less_than_square : 
  ∃ (n : ℕ), 
    (n > 0) ∧ 
    (Nat.Prime n) ∧ 
    (∃ (m : ℕ), n = m^2 - 6) ∧
    (∀ (k : ℕ), k > 0 → Nat.Prime k → (∃ (j : ℕ), k = j^2 - 6) → k ≥ n) ∧
    n = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_six_less_than_square_l1380_138079


namespace NUMINAMATH_CALUDE_vector_cosine_and_projection_l1380_138029

/-- Given vectors a and b with their components, prove the cosine of the angle between them
    and the scalar projection of a onto b. -/
theorem vector_cosine_and_projection (a b : ℝ × ℝ) (h : a = (3, 1) ∧ b = (-2, 4)) :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  (Real.cos θ = -Real.sqrt 2 / 10) ∧
  ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2) * Real.sqrt (b.1^2 + b.2^2) = -Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_vector_cosine_and_projection_l1380_138029


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1380_138023

theorem min_value_quadratic (x : ℝ) : 
  ∃ (y_min : ℝ), ∀ (y : ℝ), y = 4 * x^2 + 8 * x + 16 → y ≥ y_min ∧ y_min = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1380_138023


namespace NUMINAMATH_CALUDE_david_average_marks_l1380_138038

def david_marks : List ℝ := [72, 45, 72, 77, 75]

theorem david_average_marks :
  (david_marks.sum / david_marks.length : ℝ) = 68.2 := by sorry

end NUMINAMATH_CALUDE_david_average_marks_l1380_138038


namespace NUMINAMATH_CALUDE_ball_bounce_count_l1380_138005

/-- The number of bounces required for a ball to reach a height less than 2 feet -/
theorem ball_bounce_count (initial_height : ℝ) (bounce_ratio : ℝ) (target_height : ℝ) :
  initial_height = 20 →
  bounce_ratio = 2/3 →
  target_height = 2 →
  (∀ k : ℕ, k < 6 → initial_height * bounce_ratio^k ≥ target_height) ∧
  initial_height * bounce_ratio^6 < target_height :=
by sorry

end NUMINAMATH_CALUDE_ball_bounce_count_l1380_138005


namespace NUMINAMATH_CALUDE_complex_subtraction_and_multiplication_l1380_138094

theorem complex_subtraction_and_multiplication (i : ℂ) :
  (7 - 3 * i) - 3 * (2 + 5 * i) = 1 - 18 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_and_multiplication_l1380_138094


namespace NUMINAMATH_CALUDE_two_zeros_iff_a_positive_l1380_138068

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x + a * (x - 1)^2

theorem two_zeros_iff_a_positive (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ 
   ∀ x₃ : ℝ, f a x₃ = 0 → x₃ = x₁ ∨ x₃ = x₂) ↔ 
  a > 0 :=
sorry

end NUMINAMATH_CALUDE_two_zeros_iff_a_positive_l1380_138068


namespace NUMINAMATH_CALUDE_basketball_team_selection_l1380_138000

theorem basketball_team_selection (total_students : Nat) 
  (total_girls total_boys : Nat)
  (junior_girls senior_girls : Nat)
  (junior_boys senior_boys : Nat)
  (callback_junior_girls callback_senior_girls : Nat)
  (callback_junior_boys callback_senior_boys : Nat) :
  total_students = 56 →
  total_girls = 33 →
  total_boys = 23 →
  junior_girls = 15 →
  senior_girls = 18 →
  junior_boys = 12 →
  senior_boys = 11 →
  callback_junior_girls = 8 →
  callback_senior_girls = 9 →
  callback_junior_boys = 5 →
  callback_senior_boys = 6 →
  total_students - (callback_junior_girls + callback_senior_girls + callback_junior_boys + callback_senior_boys) = 28 := by
  sorry

#check basketball_team_selection

end NUMINAMATH_CALUDE_basketball_team_selection_l1380_138000


namespace NUMINAMATH_CALUDE_students_in_line_l1380_138076

theorem students_in_line (total : ℕ) (behind : ℕ) (h1 : total = 25) (h2 : behind = 13) :
  total - (behind + 1) = 11 := by
  sorry

end NUMINAMATH_CALUDE_students_in_line_l1380_138076


namespace NUMINAMATH_CALUDE_triangle_angle_c_l1380_138098

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_c (t : Triangle) :
  t.a = 1 ∧ t.A = Real.pi / 3 ∧ t.c = Real.sqrt 3 / 3 → t.C = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l1380_138098


namespace NUMINAMATH_CALUDE_problem_statement_l1380_138080

theorem problem_statement :
  (¬ ∃ x : ℝ, 0 < x ∧ x < 2 ∧ x^3 - x^2 - x + 2 < 0) ∧
  (¬ ∀ x y : ℝ, x + y > 4 → x > 2 ∧ y > 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1380_138080


namespace NUMINAMATH_CALUDE_m_plus_n_squared_l1380_138077

theorem m_plus_n_squared (m n : ℤ) (h1 : |m| = 4) (h2 : |n| = 3) (h3 : m - n < 0) :
  (m + n)^2 = 1 ∨ (m + n)^2 = 49 := by
sorry

end NUMINAMATH_CALUDE_m_plus_n_squared_l1380_138077


namespace NUMINAMATH_CALUDE_rebus_puzzle_solution_l1380_138085

theorem rebus_puzzle_solution :
  ∃! (A B C : Nat),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧
    A = 4 ∧ B = 7 ∧ C = 6 := by
  sorry

end NUMINAMATH_CALUDE_rebus_puzzle_solution_l1380_138085


namespace NUMINAMATH_CALUDE_intersection_points_equality_l1380_138050

/-- Theorem: For a quadratic function y = ax^2 and two parallel lines intersecting
    this function, the difference between the x-coordinates of the intersection points
    satisfies (x3 - x1) = (x2 - x4). -/
theorem intersection_points_equality 
  (a : ℝ) 
  (x1 x2 x3 x4 : ℝ) 
  (h1 : x1 < x2) 
  (h2 : x3 < x4) 
  (h_parallel : ∃ (k b c : ℝ), 
    a * x1^2 = k * x1 + b ∧ 
    a * x2^2 = k * x2 + b ∧ 
    a * x3^2 = k * x3 + c ∧ 
    a * x4^2 = k * x4 + c) :
  x3 - x1 = x2 - x4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_equality_l1380_138050
