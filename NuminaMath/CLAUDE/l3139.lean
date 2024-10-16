import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l3139_313973

theorem quadratic_roots_reciprocal_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 5 = 0 → 
  x₂^2 - 2*x₂ - 5 = 0 → 
  x₁ ≠ 0 → 
  x₂ ≠ 0 → 
  1/x₁ + 1/x₂ = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l3139_313973


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l3139_313993

theorem quadratic_equation_problem (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0) →
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 9) →
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x * y = 15) →
  m = 3 * n →
  m + n = 180 := by
sorry


end NUMINAMATH_CALUDE_quadratic_equation_problem_l3139_313993


namespace NUMINAMATH_CALUDE_gamma_less_than_delta_l3139_313944

open Real

theorem gamma_less_than_delta (α β γ δ : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π/2)
  (h4 : 0 < γ) (h5 : γ < π/2)
  (h6 : 0 < δ) (h7 : δ < π/2)
  (h8 : tan γ = (tan α + tan β) / 2)
  (h9 : 1/cos δ = (1/cos α + 1/cos β) / 2) :
  γ < δ := by
sorry


end NUMINAMATH_CALUDE_gamma_less_than_delta_l3139_313944


namespace NUMINAMATH_CALUDE_iced_coffee_consumption_ratio_l3139_313900

/-- Proves that the ratio of daily servings consumed to servings per container is 1:2 -/
theorem iced_coffee_consumption_ratio 
  (servings_per_bottle : ℕ) 
  (cost_per_bottle : ℚ) 
  (total_cost : ℚ) 
  (duration_weeks : ℕ) 
  (h1 : servings_per_bottle = 6)
  (h2 : cost_per_bottle = 3)
  (h3 : total_cost = 21)
  (h4 : duration_weeks = 2) :
  (total_cost / cost_per_bottle * servings_per_bottle) / (duration_weeks * 7) / servings_per_bottle = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_iced_coffee_consumption_ratio_l3139_313900


namespace NUMINAMATH_CALUDE_total_students_in_line_l3139_313949

/-- The number of students standing in a line with given conditions -/
def number_of_students (people_in_front_of_seokjin : ℕ) 
                       (people_behind_jimin : ℕ) 
                       (people_between_seokjin_and_jimin : ℕ) : ℕ :=
  people_in_front_of_seokjin + 1 + people_between_seokjin_and_jimin + 1 + people_behind_jimin

/-- Theorem stating that the total number of students in line is 16 -/
theorem total_students_in_line : 
  number_of_students 4 7 3 = 16 := by
  sorry


end NUMINAMATH_CALUDE_total_students_in_line_l3139_313949


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_squares_l3139_313955

theorem arithmetic_geometric_mean_sum_squares (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20)
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 120) : 
  x^2 + y^2 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_squares_l3139_313955


namespace NUMINAMATH_CALUDE_m_fourth_plus_n_fourth_l3139_313983

theorem m_fourth_plus_n_fourth (m n : ℝ) 
  (h1 : m - n = -5)
  (h2 : m^2 + n^2 = 13) :
  m^4 + n^4 = 97 := by
sorry

end NUMINAMATH_CALUDE_m_fourth_plus_n_fourth_l3139_313983


namespace NUMINAMATH_CALUDE_max_value_of_cosine_function_l3139_313910

theorem max_value_of_cosine_function :
  ∀ x : ℝ, 4 * (Real.cos x)^3 - 3 * (Real.cos x)^2 - 6 * (Real.cos x) + 5 ≤ 27/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_cosine_function_l3139_313910


namespace NUMINAMATH_CALUDE_function_property_l3139_313908

def is_positive_integer (x : ℝ) : Prop := ∃ n : ℕ, x = n ∧ n > 0

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x y, x < y → f x < f y)  -- monotonically increasing
  (h2 : ∀ n : ℕ, n > 0 → is_positive_integer (f n))  -- f(n) is a positive integer for positive integer n
  (h3 : ∀ n : ℕ, n > 0 → f (f n) = 2 * n + 1)  -- f(f(n)) = 2n + 1 for positive integer n
  : f 1 = 2 ∧ f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3139_313908


namespace NUMINAMATH_CALUDE_percentage_of_sheet_used_for_typing_l3139_313923

/-- Calculates the percentage of a rectangular sheet used for typing, given its dimensions and margins. -/
theorem percentage_of_sheet_used_for_typing 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (side_margin : ℝ) 
  (top_bottom_margin : ℝ) 
  (h1 : sheet_length = 30)
  (h2 : sheet_width = 20)
  (h3 : side_margin = 2)
  (h4 : top_bottom_margin = 3)
  : (((sheet_width - 2 * side_margin) * (sheet_length - 2 * top_bottom_margin)) / (sheet_width * sheet_length)) * 100 = 64 := by
  sorry

#check percentage_of_sheet_used_for_typing

end NUMINAMATH_CALUDE_percentage_of_sheet_used_for_typing_l3139_313923


namespace NUMINAMATH_CALUDE_rectangle_cut_theorem_l3139_313940

theorem rectangle_cut_theorem (m : ℤ) (hm : m > 12) :
  ∃ (x y : ℕ+), (x.val : ℤ) * (y.val : ℤ) > m ∧ (x.val : ℤ) * ((y.val : ℤ) - 1) < m :=
sorry

end NUMINAMATH_CALUDE_rectangle_cut_theorem_l3139_313940


namespace NUMINAMATH_CALUDE_largest_remainder_2015_l3139_313941

theorem largest_remainder_2015 : 
  ∀ d : ℕ, 1 ≤ d ∧ d ≤ 1000 → (2015 % d) ≤ 671 ∧ ∃ d₀ : ℕ, 1 ≤ d₀ ∧ d₀ ≤ 1000 ∧ 2015 % d₀ = 671 :=
by sorry

end NUMINAMATH_CALUDE_largest_remainder_2015_l3139_313941


namespace NUMINAMATH_CALUDE_binary_multiplication_l3139_313970

def binary_to_nat : List Bool → Nat
  | [] => 0
  | b::bs => (if b then 1 else 0) + 2 * binary_to_nat bs

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then
    []
  else
    (n % 2 = 1) :: nat_to_binary (n / 2)

def binary_num1 : List Bool := [true, true, false, true, true]
def binary_num2 : List Bool := [true, true, true, true]
def binary_result : List Bool := [true, false, true, true, true, true, false, true]

theorem binary_multiplication :
  binary_to_nat binary_num1 * binary_to_nat binary_num2 = binary_to_nat binary_result := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_l3139_313970


namespace NUMINAMATH_CALUDE_divisibility_of_fifth_powers_l3139_313914

theorem divisibility_of_fifth_powers (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (y - z) * (z - x) * (x - y)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_fifth_powers_l3139_313914


namespace NUMINAMATH_CALUDE_oliver_initial_money_l3139_313994

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- Oliver's initial problem -/
theorem oliver_initial_money 
  (initial_quarters : ℕ) 
  (given_dollars : ℚ) 
  (given_quarters : ℕ) 
  (remaining_total : ℚ) :
  initial_quarters = 200 →
  given_dollars = 5 →
  given_quarters = 120 →
  remaining_total = 55 →
  (initial_quarters : ℚ) * quarter_value + 
    (given_dollars + (given_quarters : ℚ) * quarter_value + remaining_total) = 120 := by
  sorry

#eval quarter_value -- This line is to check if the definition is correct

end NUMINAMATH_CALUDE_oliver_initial_money_l3139_313994


namespace NUMINAMATH_CALUDE_prob_five_largest_l3139_313943

def card_set : Finset ℕ := Finset.range 6

def selection_size : ℕ := 4

def prob_not_select_6 : ℚ :=
  (5 : ℚ) / 6 * 4 / 5 * 3 / 4 * 2 / 3

def prob_not_select_5_or_6 : ℚ :=
  (4 : ℚ) / 6 * 3 / 5 * 2 / 4 * 1 / 3

theorem prob_five_largest (card_set : Finset ℕ) (selection_size : ℕ) 
  (prob_not_select_6 : ℚ) (prob_not_select_5_or_6 : ℚ) :
  card_set = Finset.range 6 →
  selection_size = 4 →
  prob_not_select_6 = (5 : ℚ) / 6 * 4 / 5 * 3 / 4 * 2 / 3 →
  prob_not_select_5_or_6 = (4 : ℚ) / 6 * 3 / 5 * 2 / 4 * 1 / 3 →
  prob_not_select_6 - prob_not_select_5_or_6 = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_five_largest_l3139_313943


namespace NUMINAMATH_CALUDE_malcolm_lights_problem_l3139_313995

theorem malcolm_lights_problem (initial_white : ℕ) (red : ℕ) (green : ℕ) 
  (h1 : initial_white = 59)
  (h2 : red = 12)
  (h3 : green = 6) :
  initial_white - (red + 3 * red + green) = 5 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_lights_problem_l3139_313995


namespace NUMINAMATH_CALUDE_red_probability_both_jars_l3139_313934

/-- Represents a jar containing buttons -/
structure Jar :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents the state of both jars -/
structure JarState :=
  (jarA : Jar)
  (jarB : Jar)

/-- Initial state of the jars -/
def initialState : JarState :=
  { jarA := { red := 6, blue := 10 },
    jarB := { red := 2, blue := 3 } }

/-- Function to transfer buttons between jars -/
def transfer (s : JarState) (n : ℕ) : JarState :=
  { jarA := { red := s.jarA.red - n, blue := s.jarA.blue - n },
    jarB := { red := s.jarB.red + n, blue := s.jarB.blue + n } }

/-- Final state after transfer -/
def finalState : JarState :=
  transfer initialState 3

/-- Probability of selecting a red button from a jar -/
def redProbability (j : Jar) : ℚ :=
  j.red / (j.red + j.blue)

/-- Theorem: The probability of selecting red buttons from both jars is 3/22 -/
theorem red_probability_both_jars :
  redProbability finalState.jarA * redProbability finalState.jarB = 3 / 22 := by
  sorry

end NUMINAMATH_CALUDE_red_probability_both_jars_l3139_313934


namespace NUMINAMATH_CALUDE_sum_of_ages_is_fifty_l3139_313919

/-- The sum of ages of 5 children born at intervals of 3 years -/
def sum_of_ages (youngest_age : ℕ) (interval : ℕ) (num_children : ℕ) : ℕ :=
  let ages := List.range num_children
  ages.map (fun i => youngest_age + i * interval) |> List.sum

/-- Theorem stating the sum of ages for the given conditions -/
theorem sum_of_ages_is_fifty :
  sum_of_ages 4 3 5 = 50 := by
  sorry

#eval sum_of_ages 4 3 5

end NUMINAMATH_CALUDE_sum_of_ages_is_fifty_l3139_313919


namespace NUMINAMATH_CALUDE_range_of_a_l3139_313952

/-- The function f(x) = x^2 - 4x -/
def f (x : ℝ) : ℝ := x^2 - 4*x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4) a, f x ∈ Set.Icc (-4) 32) →
  (Set.Icc (-4) a = f ⁻¹' (Set.Icc (-4) 32)) →
  a ∈ Set.Icc 2 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3139_313952


namespace NUMINAMATH_CALUDE_piecewise_function_proof_l3139_313971

theorem piecewise_function_proof (x : ℝ) : 
  let a : ℝ → ℝ := λ x => (3 * x + 3) / 2
  let b : ℝ → ℝ := λ x => 5 * x / 2
  let c : ℝ → ℝ := λ x => -x + 1/2
  (x < -1 → |a x| - |b x| + c x = -1) ∧
  (-1 ≤ x ∧ x ≤ 0 → |a x| - |b x| + c x = 3 * x + 2) ∧
  (0 < x → |a x| - |b x| + c x = -2 * x + 2) :=
by sorry

end NUMINAMATH_CALUDE_piecewise_function_proof_l3139_313971


namespace NUMINAMATH_CALUDE_square_of_complex_l3139_313974

theorem square_of_complex (z : ℂ) (i : ℂ) : z = 5 + 3 * i → i^2 = -1 → z^2 = 16 + 30 * i := by
  sorry

end NUMINAMATH_CALUDE_square_of_complex_l3139_313974


namespace NUMINAMATH_CALUDE_total_wrappers_collected_l3139_313904

theorem total_wrappers_collected (andy_wrappers max_wrappers : ℕ) 
  (h1 : andy_wrappers = 34) 
  (h2 : max_wrappers = 15) : 
  andy_wrappers + max_wrappers = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_wrappers_collected_l3139_313904


namespace NUMINAMATH_CALUDE_problem_1_l3139_313953

theorem problem_1 (x : ℝ) : (x + 2) * (-3 * x + 4) = -3 * x^2 - 2 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3139_313953


namespace NUMINAMATH_CALUDE_sum_of_five_digit_binary_numbers_l3139_313916

/-- The set of all positive integers with five digits in base 2 -/
def T : Set Nat :=
  {n | 16 ≤ n ∧ n ≤ 31}

/-- The sum of all elements in T -/
def sum_T : Nat :=
  (Finset.range 16).sum (fun i => i + 16)

theorem sum_of_five_digit_binary_numbers :
  sum_T = 248 :=
sorry

end NUMINAMATH_CALUDE_sum_of_five_digit_binary_numbers_l3139_313916


namespace NUMINAMATH_CALUDE_quadratic_point_relationship_l3139_313912

/-- A quadratic function of the form y = -(x-1)² + c -/
def quadratic_function (x c : ℝ) : ℝ := -(x - 1)^2 + c

/-- Three points on the quadratic function -/
structure Points where
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ

/-- The theorem stating the relationship between y₁, y₂, and y₃ -/
theorem quadratic_point_relationship (c : ℝ) (p : Points) :
  p.y₁ = quadratic_function (-3) c →
  p.y₂ = quadratic_function (-1) c →
  p.y₃ = quadratic_function 5 c →
  p.y₂ > p.y₁ ∧ p.y₁ = p.y₃ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_point_relationship_l3139_313912


namespace NUMINAMATH_CALUDE_susan_homework_start_time_l3139_313988

def volleyball_practice_start : Nat := 16 * 60 -- 4:00 p.m. in minutes since midnight

def homework_duration : Nat := 96 -- in minutes

def time_between_homework_and_practice : Nat := 25 -- in minutes

def homework_start_time : Nat := volleyball_practice_start - homework_duration - time_between_homework_and_practice

theorem susan_homework_start_time :
  homework_start_time = 13 * 60 + 59 -- 1:59 p.m. in minutes since midnight
  := by sorry

end NUMINAMATH_CALUDE_susan_homework_start_time_l3139_313988


namespace NUMINAMATH_CALUDE_marilyn_initial_caps_l3139_313930

/-- The number of bottle caps Marilyn has initially -/
def initial_caps : ℕ := sorry

/-- The number of bottle caps Nancy gives to Marilyn -/
def nancy_caps : ℕ := 36

/-- The total number of bottle caps Marilyn has after receiving Nancy's caps -/
def total_caps : ℕ := 87

/-- Theorem stating that Marilyn's initial number of bottle caps is 51 -/
theorem marilyn_initial_caps : 
  initial_caps + nancy_caps = total_caps → initial_caps = 51 := by sorry

end NUMINAMATH_CALUDE_marilyn_initial_caps_l3139_313930


namespace NUMINAMATH_CALUDE_triangle_side_length_l3139_313969

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 60 * π / 180 →  -- Convert 60° to radians
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  c = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3139_313969


namespace NUMINAMATH_CALUDE_tom_annual_cost_l3139_313918

/-- Calculates the annual cost of medication and doctor visits for Tom --/
def annual_cost (pills_per_day : ℕ) (doctor_visit_interval_months : ℕ) (doctor_visit_cost : ℕ) 
  (pill_cost : ℕ) (insurance_coverage_percent : ℕ) : ℕ :=
  let daily_medication_cost := pills_per_day * (pill_cost * (100 - insurance_coverage_percent) / 100)
  let annual_medication_cost := daily_medication_cost * 365
  let annual_doctor_visits := 12 / doctor_visit_interval_months
  let annual_doctor_cost := annual_doctor_visits * doctor_visit_cost
  annual_medication_cost + annual_doctor_cost

/-- Theorem stating that Tom's annual cost is $1530 --/
theorem tom_annual_cost : 
  annual_cost 2 6 400 5 80 = 1530 := by
  sorry

end NUMINAMATH_CALUDE_tom_annual_cost_l3139_313918


namespace NUMINAMATH_CALUDE_triangle_3_5_7_l3139_313985

/-- A set of three line segments can form a triangle if and only if the sum of the lengths of any two sides is greater than the length of the remaining side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Prove that the set of line segments (3cm, 5cm, 7cm) can form a triangle. -/
theorem triangle_3_5_7 : can_form_triangle 3 5 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_3_5_7_l3139_313985


namespace NUMINAMATH_CALUDE_one_cow_one_bag_days_l3139_313902

def num_cows : ℕ := 45
def num_bags : ℕ := 90
def num_days : ℕ := 60

theorem one_cow_one_bag_days : 
  (num_days * num_cows) / num_bags = 30 := by
  sorry

end NUMINAMATH_CALUDE_one_cow_one_bag_days_l3139_313902


namespace NUMINAMATH_CALUDE_equal_distribution_of_chicken_wings_l3139_313935

def chicken_wings_per_person (num_friends : ℕ) (initial_wings : ℕ) (additional_wings : ℕ) : ℕ :=
  (initial_wings + additional_wings) / num_friends

theorem equal_distribution_of_chicken_wings :
  chicken_wings_per_person 5 20 25 = 9 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_chicken_wings_l3139_313935


namespace NUMINAMATH_CALUDE_counterexample_exists_l3139_313922

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 2)) ∧ n = 27 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3139_313922


namespace NUMINAMATH_CALUDE_quiche_volume_l3139_313987

theorem quiche_volume (raw_spinach : ℝ) (cooked_spinach_ratio : ℝ) (cream_cheese : ℝ) (eggs : ℝ)
  (h1 : raw_spinach = 40)
  (h2 : cooked_spinach_ratio = 0.2)
  (h3 : cream_cheese = 6)
  (h4 : eggs = 4) :
  raw_spinach * cooked_spinach_ratio + cream_cheese + eggs = 18 :=
by sorry

end NUMINAMATH_CALUDE_quiche_volume_l3139_313987


namespace NUMINAMATH_CALUDE_cost_price_of_ball_l3139_313903

/-- Given that selling 11 balls at Rs. 720 results in a loss equal to the cost price of 5 balls,
    prove that the cost price of one ball is Rs. 120. -/
theorem cost_price_of_ball (cost : ℕ) : 
  (11 * cost - 720 = 5 * cost) → cost = 120 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_ball_l3139_313903


namespace NUMINAMATH_CALUDE_max_value_with_constraint_l3139_313948

theorem max_value_with_constraint (x y z : ℝ) (h : x^2 + y^2 + z^2 = 9) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 14 ∧ ∀ (a b c : ℝ), a^2 + b^2 + c^2 = 9 → x + 2*y + 3*z ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_with_constraint_l3139_313948


namespace NUMINAMATH_CALUDE_circle_area_diameter_increase_l3139_313945

theorem circle_area_diameter_increase : 
  ∀ (A D A' D' : ℝ), 
  A > 0 → D > 0 →
  A = π * (D / 2)^2 →
  A' = 4 * A →
  A' = π * (D' / 2)^2 →
  D' / D - 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_area_diameter_increase_l3139_313945


namespace NUMINAMATH_CALUDE_a_range_when_p_and_q_false_l3139_313901

/-- Proposition p: y = a^x is monotonically decreasing on ℝ -/
def p (a : ℝ) : Prop := a > 0 ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → a^x > a^y

/-- Proposition q: y = log(ax^2 - x + a) has range ℝ -/
def q (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, a * x^2 - x + a > 0 ∧ Real.log (a * x^2 - x + a) = y

/-- If "p and q" is false, then a is in (0, 1/2] ∪ (1, ∞) -/
theorem a_range_when_p_and_q_false (a : ℝ) : ¬(p a ∧ q a) → (0 < a ∧ a ≤ 1/2) ∨ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_when_p_and_q_false_l3139_313901


namespace NUMINAMATH_CALUDE_obtuse_triangle_count_l3139_313997

/-- A function that determines if a triangle with sides a, b, and c is obtuse -/
def is_obtuse (a b c : ℕ) : Prop :=
  (a ^ 2 > b ^ 2 + c ^ 2) ∨ (b ^ 2 > a ^ 2 + c ^ 2) ∨ (c ^ 2 > a ^ 2 + b ^ 2)

/-- A function that determines if three lengths can form a valid triangle -/
def is_valid_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

/-- The main theorem stating that there are exactly 14 positive integer values of k
    for which a triangle with side lengths 13, 17, and k is obtuse -/
theorem obtuse_triangle_count :
  (∃! (s : Finset ℕ), s.card = 14 ∧ 
    (∀ k, k ∈ s ↔ (k > 0 ∧ is_valid_triangle 13 17 k ∧ is_obtuse 13 17 k))) :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_count_l3139_313997


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l3139_313921

/-- A line passing through (-2, 3) with slope 2 has the equation 2x - y - 7 = 0 -/
theorem line_equation_through_point_with_slope :
  let point : ℝ × ℝ := (-2, 3)
  let slope : ℝ := 2
  let line_equation (x y : ℝ) := 2 * x - y - 7 = 0
  (∀ x y, line_equation x y ↔ y - point.2 = slope * (x - point.1)) ∧
  line_equation point.1 point.2 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l3139_313921


namespace NUMINAMATH_CALUDE_rectangle_area_with_hole_l3139_313906

theorem rectangle_area_with_hole (x : ℝ) (h : x > 1.5) :
  (x + 10) * (x + 8) - (2 * x) * (x + 1) = -x^2 + 16*x + 80 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_hole_l3139_313906


namespace NUMINAMATH_CALUDE_gcd_increase_l3139_313978

theorem gcd_increase (m n : ℕ) (h : Nat.gcd (m + 6) n = 9 * Nat.gcd m n) :
  Nat.gcd m n = 3 ∨ Nat.gcd m n = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_increase_l3139_313978


namespace NUMINAMATH_CALUDE_prob_both_white_l3139_313938

def box_A_white : ℕ := 3
def box_A_black : ℕ := 2
def box_B_white : ℕ := 2
def box_B_black : ℕ := 3

def prob_white_from_A : ℚ := box_A_white / (box_A_white + box_A_black)
def prob_white_from_B : ℚ := box_B_white / (box_B_white + box_B_black)

theorem prob_both_white :
  prob_white_from_A * prob_white_from_B = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_white_l3139_313938


namespace NUMINAMATH_CALUDE_f_geq_2_solution_set_f_minus_abs_geq_0_t_range_l3139_313913

-- Define the function f
def f (x : ℝ) : ℝ := |x| - 2*|x + 3|

-- Theorem for the first part of the problem
theorem f_geq_2_solution_set :
  {x : ℝ | f x ≥ 2} = {x : ℝ | -4 ≤ x ∧ x ≤ -8/3} := by sorry

-- Theorem for the second part of the problem
theorem f_minus_abs_geq_0_t_range :
  {t : ℝ | ∃ x, f x - |3*t - 2| ≥ 0} = {t : ℝ | -1/3 ≤ t ∧ t ≤ 5/3} := by sorry

end NUMINAMATH_CALUDE_f_geq_2_solution_set_f_minus_abs_geq_0_t_range_l3139_313913


namespace NUMINAMATH_CALUDE_square_of_binomial_l3139_313991

theorem square_of_binomial (a : ℝ) : 
  (∃ b c : ℝ, ∀ x, 9*x^2 + 18*x + a = (b*x + c)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l3139_313991


namespace NUMINAMATH_CALUDE_jennifer_spending_l3139_313976

theorem jennifer_spending (total : ℝ) (sandwich_fraction : ℝ) (museum_fraction : ℝ) (book_fraction : ℝ)
  (h_total : total = 150)
  (h_sandwich : sandwich_fraction = 1/5)
  (h_museum : museum_fraction = 1/6)
  (h_book : book_fraction = 1/2) :
  total - (sandwich_fraction * total + museum_fraction * total + book_fraction * total) = 20 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_spending_l3139_313976


namespace NUMINAMATH_CALUDE_arithmetic_mean_ge_geometric_mean_l3139_313986

theorem arithmetic_mean_ge_geometric_mean (a b : ℝ) : (a + b) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_ge_geometric_mean_l3139_313986


namespace NUMINAMATH_CALUDE_tan_negative_255_degrees_l3139_313907

theorem tan_negative_255_degrees : Real.tan (-(255 * π / 180)) = Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_255_degrees_l3139_313907


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l3139_313999

theorem rectangular_plot_breadth (length width area : ℝ) : 
  length = 3 * width → 
  area = length * width → 
  area = 2028 → 
  width = 26 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l3139_313999


namespace NUMINAMATH_CALUDE_martian_age_conversion_l3139_313917

/-- Converts a single digit from base 9 to base 10 -/
def base9ToBase10Digit (d : Nat) : Nat := d

/-- Converts a 3-digit number from base 9 to base 10 -/
def base9ToBase10 (d₂ d₁ d₀ : Nat) : Nat :=
  base9ToBase10Digit d₂ * 9^2 + base9ToBase10Digit d₁ * 9^1 + base9ToBase10Digit d₀ * 9^0

/-- The age of the Martian robot's manufacturing facility in base 9 -/
def martianAge : Nat := 376

theorem martian_age_conversion :
  base9ToBase10 3 7 6 = 312 := by
  sorry

end NUMINAMATH_CALUDE_martian_age_conversion_l3139_313917


namespace NUMINAMATH_CALUDE_first_movie_length_is_correct_l3139_313977

/-- The length of the first movie in minutes -/
def first_movie_length : ℕ := 90

/-- The length of the second movie in minutes -/
def second_movie_length : ℕ := first_movie_length + 30

/-- The time spent making popcorn in minutes -/
def popcorn_time : ℕ := 10

/-- The time spent making fries in minutes -/
def fries_time : ℕ := 2 * popcorn_time

/-- The total time spent cooking and watching movies in minutes -/
def total_time : ℕ := 4 * 60

theorem first_movie_length_is_correct : 
  first_movie_length + second_movie_length + popcorn_time + fries_time = total_time := by
  sorry

end NUMINAMATH_CALUDE_first_movie_length_is_correct_l3139_313977


namespace NUMINAMATH_CALUDE_geometry_propositions_l3139_313933

theorem geometry_propositions (p₁ p₂ p₃ p₄ : Prop) 
  (h₁ : p₁) (h₂ : ¬p₂) (h₃ : ¬p₃) (h₄ : p₄) : 
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) ∧ ¬(p₁ ∧ p₂) := by
  sorry

end NUMINAMATH_CALUDE_geometry_propositions_l3139_313933


namespace NUMINAMATH_CALUDE_square_difference_l3139_313937

theorem square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 49) 
  (h2 : x * y = 6) : 
  (x - y)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3139_313937


namespace NUMINAMATH_CALUDE_hyperbola_point_distance_to_origin_l3139_313961

/-- Given points F₁ and F₂ on the x-axis, and a point P satisfying the hyperbola equation,
    prove that the distance from P to the origin is √6/2 when P's y-coordinate is 1/2. -/
theorem hyperbola_point_distance_to_origin :
  ∀ (P : ℝ × ℝ),
  let F₁ : ℝ × ℝ := (-Real.sqrt 2, 0)
  let F₂ : ℝ × ℝ := (Real.sqrt 2, 0)
  let dist (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  P.2 = 1/2 →
  dist P F₂ - dist P F₁ = 2 →
  dist P (0, 0) = Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_point_distance_to_origin_l3139_313961


namespace NUMINAMATH_CALUDE_john_volunteer_hours_l3139_313980

/-- Represents John's volunteering schedule for a year -/
structure VolunteerSchedule where
  first_six_months_frequency : Nat
  first_six_months_hours : Nat
  next_five_months_frequency : Nat
  next_five_months_hours : Nat
  december_days : Nat
  december_total_hours : Nat

/-- Calculates the total volunteering hours for a year given a schedule -/
def total_volunteer_hours (schedule : VolunteerSchedule) : Nat :=
  (schedule.first_six_months_frequency * schedule.first_six_months_hours * 6) +
  (schedule.next_five_months_frequency * schedule.next_five_months_hours * 4 * 5) +
  schedule.december_total_hours

/-- Theorem stating that John's volunteering schedule results in 82 hours for the year -/
theorem john_volunteer_hours :
  ∃ (schedule : VolunteerSchedule),
    schedule.first_six_months_frequency = 2 ∧
    schedule.first_six_months_hours = 3 ∧
    schedule.next_five_months_frequency = 1 ∧
    schedule.next_five_months_hours = 2 ∧
    schedule.december_days = 3 ∧
    schedule.december_total_hours = 6 ∧
    total_volunteer_hours schedule = 82 := by
  sorry

end NUMINAMATH_CALUDE_john_volunteer_hours_l3139_313980


namespace NUMINAMATH_CALUDE_buying_goods_equations_l3139_313981

/-- Represents the problem of buying goods collectively --/
def BuyingGoods (x y : ℤ) : Prop :=
  (∃ (leftover : ℤ), 8 * x - y = leftover ∧ leftover = 3) ∧
  (∃ (shortage : ℤ), y - 7 * x = shortage ∧ shortage = 4)

/-- The correct system of equations for the buying goods problem --/
theorem buying_goods_equations (x y : ℤ) :
  BuyingGoods x y ↔ (8 * x - 3 = y ∧ 7 * x + 4 = y) :=
sorry

end NUMINAMATH_CALUDE_buying_goods_equations_l3139_313981


namespace NUMINAMATH_CALUDE_garden_ratio_l3139_313979

/-- Given a rectangular garden with perimeter 180 yards and length 60 yards,
    prove that the ratio of length to width is 2:1 -/
theorem garden_ratio (perimeter : ℝ) (length : ℝ) (width : ℝ)
    (h1 : perimeter = 180)
    (h2 : length = 60)
    (h3 : perimeter = 2 * length + 2 * width) :
    length / width = 2 := by
  sorry

end NUMINAMATH_CALUDE_garden_ratio_l3139_313979


namespace NUMINAMATH_CALUDE_scientific_notation_5690_l3139_313968

theorem scientific_notation_5690 : 
  5690 = 5.69 * (10 : ℝ)^3 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_5690_l3139_313968


namespace NUMINAMATH_CALUDE_four_points_planes_l3139_313965

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A set of four points in space -/
def FourPoints : Type := Fin 4 → Point3D

/-- Three points are not collinear if they don't lie on the same line -/
def NotCollinear (p q r : Point3D) : Prop :=
  ∀ t : ℝ, (q.x - p.x, q.y - p.y, q.z - p.z) ≠ t • (r.x - p.x, r.y - p.y, r.z - p.z)

/-- The number of planes determined by any three points from a set of four points -/
def NumberOfPlanes (points : FourPoints) : ℕ :=
  sorry

/-- Theorem: Given four points in space where any three are not collinear,
    the number of planes determined by any three of these points is either 1 or 4 -/
theorem four_points_planes (points : FourPoints)
    (h : ∀ i j k : Fin 4, i ≠ j → j ≠ k → i ≠ k → NotCollinear (points i) (points j) (points k)) :
    NumberOfPlanes points = 1 ∨ NumberOfPlanes points = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_points_planes_l3139_313965


namespace NUMINAMATH_CALUDE_at_most_two_sides_equal_to_longest_diagonal_l3139_313932

-- Define a convex polygon
def ConvexPolygon : Type := sorry

-- Define the concept of a diagonal in a polygon
def diagonal (p : ConvexPolygon) : Type := sorry

-- Define the length of a side or diagonal
def length {T : Type} (x : T) : ℝ := sorry

-- Define the longest diagonal of a polygon
def longest_diagonal (p : ConvexPolygon) : diagonal p := sorry

-- Define a function that counts the number of sides equal to the longest diagonal
def count_sides_equal_to_longest_diagonal (p : ConvexPolygon) : ℕ := sorry

-- Theorem statement
theorem at_most_two_sides_equal_to_longest_diagonal (p : ConvexPolygon) :
  count_sides_equal_to_longest_diagonal p ≤ 2 := by sorry

end NUMINAMATH_CALUDE_at_most_two_sides_equal_to_longest_diagonal_l3139_313932


namespace NUMINAMATH_CALUDE_simplify_power_of_power_l3139_313984

theorem simplify_power_of_power (x : ℝ) : (2 * x^3)^3 = 8 * x^9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_of_power_l3139_313984


namespace NUMINAMATH_CALUDE_tower_lights_problem_l3139_313947

theorem tower_lights_problem (n : ℕ) (r : ℝ) (sum : ℝ) (h1 : n = 7) (h2 : r = 2) (h3 : sum = 381) :
  let first_term := sum * (r - 1) / (r^n - 1)
  first_term = 3 := by sorry

end NUMINAMATH_CALUDE_tower_lights_problem_l3139_313947


namespace NUMINAMATH_CALUDE_pyramid_volume_change_l3139_313967

theorem pyramid_volume_change (s h : ℝ) : 
  s > 0 → h > 0 → (1/3 : ℝ) * s^2 * h = 60 → 
  (1/3 : ℝ) * (3*s)^2 * (2*h) = 1080 := by
sorry

end NUMINAMATH_CALUDE_pyramid_volume_change_l3139_313967


namespace NUMINAMATH_CALUDE_investment_problem_l3139_313905

/-- Represents the investment scenario described in the problem -/
structure Investment where
  total : ℝ
  interest : ℝ
  known_rate : ℝ
  unknown_amount : ℝ

/-- The theorem statement representing the problem -/
theorem investment_problem (inv : Investment) 
  (h1 : inv.total = 15000)
  (h2 : inv.interest = 1023)
  (h3 : inv.known_rate = 0.075)
  (h4 : inv.unknown_amount = 8200)
  (h5 : inv.unknown_amount + (inv.total - inv.unknown_amount) * inv.known_rate = inv.interest) :
  inv.unknown_amount = 8200 := by
  sorry

#check investment_problem

end NUMINAMATH_CALUDE_investment_problem_l3139_313905


namespace NUMINAMATH_CALUDE_sequence_difference_theorem_l3139_313924

def is_valid_sequence (x : ℕ → ℕ) : Prop :=
  x 1 = 1 ∧ 
  (∀ n, x n < x (n + 1)) ∧
  (∀ n, x (2 * n + 1) ≤ 2 * n)

theorem sequence_difference_theorem (x : ℕ → ℕ) (h : is_valid_sequence x) :
  ∀ k : ℕ, ∃ r s : ℕ, x r - x s = k :=
sorry

end NUMINAMATH_CALUDE_sequence_difference_theorem_l3139_313924


namespace NUMINAMATH_CALUDE_gcd_204_85_l3139_313929

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_204_85_l3139_313929


namespace NUMINAMATH_CALUDE_relatively_prime_dates_february_leap_year_count_l3139_313998

/-- The number of days in February during a leap year -/
def leap_year_february_days : ℕ := 29

/-- The month number for February -/
def february_number : ℕ := 2

/-- A function that returns the number of relatively prime dates in February of a leap year -/
def relatively_prime_dates_february_leap_year : ℕ := 
  leap_year_february_days - (leap_year_february_days / february_number)

/-- Theorem stating that the number of relatively prime dates in February of a leap year is 15 -/
theorem relatively_prime_dates_february_leap_year_count : 
  relatively_prime_dates_february_leap_year = 15 := by sorry

end NUMINAMATH_CALUDE_relatively_prime_dates_february_leap_year_count_l3139_313998


namespace NUMINAMATH_CALUDE_tea_mixture_price_l3139_313954

/-- Given two types of tea mixed in a 1:1 ratio, where one tea costs 62 rupees per kg
    and the mixture is worth 67 rupees per kg, prove that the price of the second tea
    is 72 rupees per kg. -/
theorem tea_mixture_price (price_tea1 price_mixture : ℚ) (ratio : ℚ × ℚ) :
  price_tea1 = 62 →
  price_mixture = 67 →
  ratio = (1, 1) →
  ∃ price_tea2 : ℚ, price_tea2 = 72 ∧
    (price_tea1 * ratio.1 + price_tea2 * ratio.2) / (ratio.1 + ratio.2) = price_mixture :=
by sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l3139_313954


namespace NUMINAMATH_CALUDE_journey_times_equal_l3139_313939

-- Define the variables
def distance1 : ℝ := 120
def distance2 : ℝ := 240

-- Define the theorem
theorem journey_times_equal (speed1 : ℝ) (h1 : speed1 > 0) :
  distance1 / speed1 = distance2 / (2 * speed1) :=
by sorry

end NUMINAMATH_CALUDE_journey_times_equal_l3139_313939


namespace NUMINAMATH_CALUDE_symmetric_functions_property_l3139_313927

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the theorem
theorem symmetric_functions_property (h1 : ∀ x, f (x - 1) = g⁻¹ x) (h2 : g 2 = 0) : f (-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_functions_property_l3139_313927


namespace NUMINAMATH_CALUDE_complex_multiplication_l3139_313925

theorem complex_multiplication (z : ℂ) (h : z = 1 + I) : (1 + z) * z = 1 + 3*I := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3139_313925


namespace NUMINAMATH_CALUDE_inverse_variation_l3139_313958

/-- Given that a and b vary inversely, prove that when a = 800 and b = 0.5, 
    then b = 0.125 when a = 3200 -/
theorem inverse_variation (a b : ℝ) (h : a * b = 800 * 0.5) :
  3200 * (1 / 8) = 800 * 0.5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_l3139_313958


namespace NUMINAMATH_CALUDE_oil_depth_calculation_l3139_313928

/-- Represents a right cylindrical tank -/
structure Tank where
  height : ℝ
  base_diameter : ℝ

/-- Calculates the volume of oil in the tank when lying on its side -/
def oil_volume_side (tank : Tank) (depth : ℝ) : ℝ :=
  sorry

/-- Calculates the depth of oil when the tank is standing upright -/
def oil_depth_upright (tank : Tank) (volume : ℝ) : ℝ :=
  sorry

/-- Theorem: For the given tank dimensions and side oil depth, 
    the upright oil depth is approximately 2.2 feet -/
theorem oil_depth_calculation (tank : Tank) (side_depth : ℝ) :
  tank.height = 20 →
  tank.base_diameter = 6 →
  side_depth = 4 →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
    |oil_depth_upright tank (oil_volume_side tank side_depth) - 2.2| < ε :=
sorry

end NUMINAMATH_CALUDE_oil_depth_calculation_l3139_313928


namespace NUMINAMATH_CALUDE_inequality_proof_l3139_313936

theorem inequality_proof (x y z : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) :
  (1 + 2*x + 2*y + 2*z) * (1 + 2*y + 2*z) * (1 + 2*x + 2*z) * (1 + 2*x + 2*y) ≥ 
  (1 + 3*x + 3*y) * (1 + 3*y + 3*z) * (1 + 3*x) * (1 + 3*z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3139_313936


namespace NUMINAMATH_CALUDE_real_part_of_z_l3139_313960

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z - 3) = -1 + 3 * Complex.I) : 
  z.re = 6 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3139_313960


namespace NUMINAMATH_CALUDE_worksheets_graded_l3139_313990

theorem worksheets_graded (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (remaining_problems : ℕ) : 
  total_worksheets = 17 →
  problems_per_worksheet = 7 →
  remaining_problems = 63 →
  total_worksheets * problems_per_worksheet - remaining_problems = 8 * problems_per_worksheet :=
by sorry

end NUMINAMATH_CALUDE_worksheets_graded_l3139_313990


namespace NUMINAMATH_CALUDE_teachers_survey_l3139_313962

theorem teachers_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h_total : total = 150)
  (h_high_bp : high_bp = 80)
  (h_heart_trouble : heart_trouble = 50)
  (h_both : both = 30) :
  (((total - (high_bp + heart_trouble - both)) : ℚ) / total) * 100 = 100/3 := by
sorry

end NUMINAMATH_CALUDE_teachers_survey_l3139_313962


namespace NUMINAMATH_CALUDE_solve_y_l3139_313966

theorem solve_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 10) : y = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_y_l3139_313966


namespace NUMINAMATH_CALUDE_deceased_member_income_l3139_313931

theorem deceased_member_income
  (initial_members : ℕ)
  (final_members : ℕ)
  (initial_average : ℚ)
  (final_average : ℚ)
  (h1 : initial_members = 4)
  (h2 : final_members = 3)
  (h3 : initial_average = 782)
  (h4 : final_average = 650)
  : (initial_members : ℚ) * initial_average - (final_members : ℚ) * final_average = 1178 := by
  sorry

end NUMINAMATH_CALUDE_deceased_member_income_l3139_313931


namespace NUMINAMATH_CALUDE_imaginary_part_of_inverse_one_plus_i_squared_l3139_313975

theorem imaginary_part_of_inverse_one_plus_i_squared (i : ℂ) (h : i * i = -1) :
  Complex.im (1 / ((1 : ℂ) + i)^2) = -(1/2) := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_inverse_one_plus_i_squared_l3139_313975


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3139_313946

/-- An arithmetic sequence starting at -58 with common difference 7 -/
def arithmeticSequence (n : ℕ) : ℤ := -58 + (n - 1) * 7

/-- The property that the sequence ends at or before 44 -/
def sequenceEndsBeforeOrAt44 (n : ℕ) : Prop := arithmeticSequence n ≤ 44

theorem arithmetic_sequence_length :
  ∃ (n : ℕ), n = 15 ∧ sequenceEndsBeforeOrAt44 n ∧ ¬sequenceEndsBeforeOrAt44 (n + 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3139_313946


namespace NUMINAMATH_CALUDE_brenda_spay_problem_l3139_313942

/-- The number of cats Brenda needs to spay -/
def num_cats : ℕ := 7

/-- The number of dogs Brenda needs to spay -/
def num_dogs : ℕ := 2 * num_cats

/-- The total number of animals Brenda needs to spay -/
def total_animals : ℕ := 21

theorem brenda_spay_problem :
  num_cats = 7 ∧ num_dogs = 2 * num_cats ∧ num_cats + num_dogs = total_animals :=
sorry

end NUMINAMATH_CALUDE_brenda_spay_problem_l3139_313942


namespace NUMINAMATH_CALUDE_unclaimed_candy_l3139_313982

/-- Represents the order of arrival of the winners -/
inductive Winner : Type
  | Al | Bert | Carl | Dana

/-- The ratio of candy each winner should receive -/
def candy_ratio (w : Winner) : ℚ :=
  match w with
  | Winner.Al => 4 / 10
  | Winner.Bert => 3 / 10
  | Winner.Carl => 2 / 10
  | Winner.Dana => 1 / 10

/-- The amount of candy each winner actually takes -/
def candy_taken (w : Winner) : ℚ :=
  match w with
  | Winner.Al => 4 / 10
  | Winner.Bert => 9 / 50
  | Winner.Carl => 21 / 250
  | Winner.Dana => 19 / 250

theorem unclaimed_candy :
  1 - (candy_taken Winner.Al + candy_taken Winner.Bert + candy_taken Winner.Carl + candy_taken Winner.Dana) = 46 / 125 := by
  sorry

end NUMINAMATH_CALUDE_unclaimed_candy_l3139_313982


namespace NUMINAMATH_CALUDE_victoria_shopping_theorem_l3139_313926

def shopping_and_dinner_problem (initial_amount : ℝ) 
  (jacket_price : ℝ) (jacket_quantity : ℕ)
  (trouser_price : ℝ) (trouser_quantity : ℕ)
  (purse_price : ℝ) (purse_quantity : ℕ)
  (discount_rate : ℝ) (dinner_bill : ℝ) : Prop :=
  let jacket_cost := jacket_price * jacket_quantity
  let trouser_cost := trouser_price * trouser_quantity
  let purse_cost := purse_price * purse_quantity
  let discountable_cost := jacket_cost + trouser_cost
  let discount_amount := discountable_cost * discount_rate
  let shopping_cost := discountable_cost - discount_amount + purse_cost
  let dinner_cost := dinner_bill / 1.15
  let total_spent := shopping_cost + dinner_cost
  let remaining_amount := initial_amount - total_spent
  remaining_amount = 3725

theorem victoria_shopping_theorem : 
  shopping_and_dinner_problem 10000 250 8 180 15 450 4 0.15 552.50 :=
by sorry

end NUMINAMATH_CALUDE_victoria_shopping_theorem_l3139_313926


namespace NUMINAMATH_CALUDE_power_of_three_plus_five_mod_eight_l3139_313920

theorem power_of_three_plus_five_mod_eight : (3^101 + 5) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_plus_five_mod_eight_l3139_313920


namespace NUMINAMATH_CALUDE_integer_representation_l3139_313992

theorem integer_representation (n : ℤ) : ∃ (x y z : ℕ+), (n : ℤ) = x^2 + y^2 - z^2 := by
  sorry

end NUMINAMATH_CALUDE_integer_representation_l3139_313992


namespace NUMINAMATH_CALUDE_original_holes_additional_holes_l3139_313950

-- Define the circumference of the circular road
def circumference : ℕ := 400

-- Define the original interval between streetlamps
def original_interval : ℕ := 50

-- Define the new interval between streetlamps
def new_interval : ℕ := 40

-- Theorem for the number of holes in the original plan
theorem original_holes : circumference / original_interval = 8 := by sorry

-- Theorem for the number of additional holes in the new plan
theorem additional_holes : 
  circumference / new_interval - (circumference / (Nat.lcm original_interval new_interval)) = 8 := by sorry

end NUMINAMATH_CALUDE_original_holes_additional_holes_l3139_313950


namespace NUMINAMATH_CALUDE_min_n_for_infinite_moves_l3139_313972

/-- A move in the card game -/
structure Move where
  cards : Finset ℕ
  sum_equals_index : ℕ

/-- The card game setup -/
structure CardGame where
  n : ℕ
  card_count : ℕ → ℕ
  card_count_eq_n : ∀ l : ℕ, card_count l = n

/-- An infinite sequence of moves in the game -/
def InfiniteMoveSequence (game : CardGame) : Type :=
  ℕ → Move

/-- The theorem statement -/
theorem min_n_for_infinite_moves :
  ∀ n : ℕ,
  n ≥ 10000 →
  ∃ (game : CardGame) (moves : InfiniteMoveSequence game),
    (∀ k : ℕ, k > 0 → 
      (moves k).cards.card = 100 ∧
      (moves k).sum_equals_index = k) ∧
  ∀ m : ℕ,
  m < 10000 →
  ¬∃ (game : CardGame) (moves : InfiniteMoveSequence game),
    (∀ k : ℕ, k > 0 → 
      (moves k).cards.card = 100 ∧
      (moves k).sum_equals_index = k) :=
sorry

end NUMINAMATH_CALUDE_min_n_for_infinite_moves_l3139_313972


namespace NUMINAMATH_CALUDE_parallelogram_area_l3139_313964

/-- The area of a parallelogram with base 3.6 and height 2.5 times the base is 32.4 -/
theorem parallelogram_area : 
  let base : ℝ := 3.6
  let height : ℝ := 2.5 * base
  let area : ℝ := base * height
  area = 32.4 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3139_313964


namespace NUMINAMATH_CALUDE_eight_distinct_lengths_l3139_313956

/-- Represents an isosceles right triangle with side length 24 -/
structure IsoscelesRightTriangle :=
  (side : ℝ)
  (is_24 : side = 24)

/-- Counts the number of distinct integer lengths of line segments from a vertex to the hypotenuse -/
def count_distinct_integer_lengths (t : IsoscelesRightTriangle) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 8 distinct integer lengths -/
theorem eight_distinct_lengths (t : IsoscelesRightTriangle) : 
  count_distinct_integer_lengths t = 8 := by sorry

end NUMINAMATH_CALUDE_eight_distinct_lengths_l3139_313956


namespace NUMINAMATH_CALUDE_interior_point_is_center_of_gravity_l3139_313915

/-- A lattice point represented by its x and y coordinates -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle represented by its three vertices -/
structure LatticeTriangle where
  v1 : LatticePoint
  v2 : LatticePoint
  v3 : LatticePoint

/-- Checks if a point is on the boundary of a triangle -/
def isOnBoundary (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Checks if a point is in the interior of a triangle -/
def isInterior (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Calculates the center of gravity of a triangle -/
def centerOfGravity (t : LatticeTriangle) : LatticePoint := sorry

/-- The main theorem -/
theorem interior_point_is_center_of_gravity 
  (t : LatticeTriangle) 
  (h1 : t.v1 = ⟨0, 0⟩) 
  (h2 : ∀ p : LatticePoint, p ≠ t.v1 ∧ p ≠ t.v2 ∧ p ≠ t.v3 → ¬isOnBoundary p t) 
  (p : LatticePoint) 
  (h3 : isInterior p t) 
  (h4 : ∀ q : LatticePoint, q ≠ p → ¬isInterior q t) : 
  p = centerOfGravity t := by
  sorry

end NUMINAMATH_CALUDE_interior_point_is_center_of_gravity_l3139_313915


namespace NUMINAMATH_CALUDE_chips_note_taking_schedule_l3139_313989

/-- Chip's note-taking schedule --/
theorem chips_note_taking_schedule 
  (pages_per_class : ℕ) 
  (num_classes : ℕ) 
  (sheets_per_pack : ℕ) 
  (num_weeks : ℕ) 
  (packs_used : ℕ) 
  (h1 : pages_per_class = 2)
  (h2 : num_classes = 5)
  (h3 : sheets_per_pack = 100)
  (h4 : num_weeks = 6)
  (h5 : packs_used = 3) :
  (packs_used * sheets_per_pack) / (pages_per_class * num_classes * num_weeks) = 5 := by
  sorry

#check chips_note_taking_schedule

end NUMINAMATH_CALUDE_chips_note_taking_schedule_l3139_313989


namespace NUMINAMATH_CALUDE_committee_combinations_l3139_313963

theorem committee_combinations : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_committee_combinations_l3139_313963


namespace NUMINAMATH_CALUDE_special_function_sum_negative_l3139_313909

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  odd : ∀ x, f (x + 2) = -f (-x + 2)
  mono : ∀ x y, x > 2 → y > 2 → x < y → f x < f y

/-- The main theorem -/
theorem special_function_sum_negative (F : SpecialFunction) 
  (x₁ x₂ : ℝ) (h1 : x₁ + x₂ < 4) (h2 : (x₁ - 2) * (x₂ - 2) < 0) :
  F.f x₁ + F.f x₂ < 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_sum_negative_l3139_313909


namespace NUMINAMATH_CALUDE_product_of_roots_l3139_313957

theorem product_of_roots (x : ℂ) : 
  (2 * x^3 - 3 * x^2 - 8 * x + 10 = 0) → 
  (∃ r₁ r₂ r₃ : ℂ, (x - r₁) * (x - r₂) * (x - r₃) = 2 * x^3 - 3 * x^2 - 8 * x + 10 ∧ r₁ * r₂ * r₃ = -5) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l3139_313957


namespace NUMINAMATH_CALUDE_river_rectification_l3139_313951

theorem river_rectification 
  (total_length : ℝ) 
  (rate_A : ℝ) 
  (rate_B : ℝ) 
  (total_time : ℝ) 
  (h1 : total_length = 180)
  (h2 : rate_A = 8)
  (h3 : rate_B = 12)
  (h4 : total_time = 20) :
  ∃ (length_A length_B : ℝ),
    length_A + length_B = total_length ∧
    length_A / rate_A + length_B / rate_B = total_time ∧
    length_A = 120 ∧
    length_B = 60 :=
by sorry

end NUMINAMATH_CALUDE_river_rectification_l3139_313951


namespace NUMINAMATH_CALUDE_jessica_apple_pie_servings_l3139_313911

/-- Calculates the number of apples per serving in Jessica's apple pies. -/
def apples_per_serving (num_guests : ℕ) (num_pies : ℕ) (servings_per_pie : ℕ) (apples_per_guest : ℚ) : ℚ :=
  let total_apples := num_guests * apples_per_guest
  let total_servings := num_pies * servings_per_pie
  total_apples / total_servings

/-- Theorem stating that given Jessica's conditions, each serving requires 1.5 apples. -/
theorem jessica_apple_pie_servings :
  apples_per_serving 12 3 8 3 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_apple_pie_servings_l3139_313911


namespace NUMINAMATH_CALUDE_triangle_ABC_is_right_l3139_313959

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line passing through (5,-2)
def line_through_point (x y : ℝ) : Prop := ∃ (m b : ℝ), y = m*x + b ∧ -2 = m*5 + b

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  let (x₁, y₁) := t.A
  let (x₂, y₂) := t.B
  let (x₃, y₃) := t.C
  (x₂ - x₁) * (x₃ - x₁) + (y₂ - y₁) * (y₃ - y₁) = 0

-- Theorem statement
theorem triangle_ABC_is_right :
  ∀ (B C : ℝ × ℝ),
  parabola B.1 B.2 →
  parabola C.1 C.2 →
  line_through_point B.1 B.2 →
  line_through_point C.1 C.2 →
  is_right_triangle { A := (1, 2), B := B, C := C } :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_is_right_l3139_313959


namespace NUMINAMATH_CALUDE_total_spent_theorem_l3139_313996

/-- Calculates the total amount spent on pens by Dorothy, Julia, and Robert --/
def total_spent_on_pens (robert_pens : ℕ) (julia_factor : ℕ) (dorothy_factor : ℚ) (cost_per_pen : ℚ) : ℚ :=
  let julia_pens := julia_factor * robert_pens
  let dorothy_pens := dorothy_factor * julia_pens
  let total_pens := robert_pens + julia_pens + dorothy_pens
  total_pens * cost_per_pen

/-- Theorem stating the total amount spent on pens by Dorothy, Julia, and Robert --/
theorem total_spent_theorem :
  total_spent_on_pens 4 3 (1/2) (3/2) = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_theorem_l3139_313996
