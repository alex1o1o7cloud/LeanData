import Mathlib

namespace NUMINAMATH_CALUDE_performance_arrangements_l3598_359897

-- Define the number of singers
def n : ℕ := 6

-- Define the number of singers with specific order requirements (A, B, C)
def k : ℕ := 3

-- Define the number of valid orders for B and C relative to A
def valid_orders : ℕ := 4

-- Theorem statement
theorem performance_arrangements : 
  (valid_orders : ℕ) * (n.factorial / k.factorial) = 480 := by
  sorry

end NUMINAMATH_CALUDE_performance_arrangements_l3598_359897


namespace NUMINAMATH_CALUDE_cherry_pits_sprouted_percentage_l3598_359811

theorem cherry_pits_sprouted_percentage (total_pits : ℕ) (saplings_sold : ℕ) (saplings_left : ℕ) :
  total_pits = 80 →
  saplings_sold = 6 →
  saplings_left = 14 →
  (((saplings_sold + saplings_left : ℚ) / total_pits) * 100 : ℚ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pits_sprouted_percentage_l3598_359811


namespace NUMINAMATH_CALUDE_x_is_perfect_square_l3598_359848

theorem x_is_perfect_square (x y : ℕ+) (h : (2 * x * y) ∣ (x^2 + y^2 - x)) : 
  ∃ n : ℕ+, x = n^2 := by
sorry

end NUMINAMATH_CALUDE_x_is_perfect_square_l3598_359848


namespace NUMINAMATH_CALUDE_modular_inverse_17_mod_500_l3598_359833

theorem modular_inverse_17_mod_500 :
  ∃ x : ℕ, x < 500 ∧ (17 * x) % 500 = 1 :=
by
  use 295
  sorry

end NUMINAMATH_CALUDE_modular_inverse_17_mod_500_l3598_359833


namespace NUMINAMATH_CALUDE_next_birthday_age_is_56_l3598_359810

/-- Represents a person's age in years, months, weeks, and days -/
structure Age where
  years : ℕ
  months : ℕ
  weeks : ℕ
  days : ℕ

/-- Calculates the age on the next birthday given a current age -/
def nextBirthdayAge (currentAge : Age) : ℕ :=
  sorry

/-- Theorem stating that given the specific age, the next birthday age will be 56 -/
theorem next_birthday_age_is_56 :
  let currentAge : Age := { years := 50, months := 50, weeks := 50, days := 50 }
  nextBirthdayAge currentAge = 56 := by
  sorry

end NUMINAMATH_CALUDE_next_birthday_age_is_56_l3598_359810


namespace NUMINAMATH_CALUDE_not_monotone_decreasing_l3598_359896

theorem not_monotone_decreasing (f : ℝ → ℝ) (h : f 2 > f 1) : 
  ¬(∀ x y : ℝ, x ≤ y → f x ≥ f y) := by sorry

end NUMINAMATH_CALUDE_not_monotone_decreasing_l3598_359896


namespace NUMINAMATH_CALUDE_lisa_quiz_goal_l3598_359891

theorem lisa_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (completed_as : ℕ) 
  (h1 : total_quizzes = 40)
  (h2 : goal_percentage = 9/10)
  (h3 : completed_quizzes = 25)
  (h4 : completed_as = 20) : 
  (total_quizzes - completed_quizzes : ℤ) - 
  (↑(total_quizzes * goal_percentage.num) / goal_percentage.den - completed_as : ℚ) = 0 :=
by sorry

end NUMINAMATH_CALUDE_lisa_quiz_goal_l3598_359891


namespace NUMINAMATH_CALUDE_train_passenger_count_l3598_359846

theorem train_passenger_count (round_trips : ℕ) (return_passengers : ℕ) (total_passengers : ℕ) :
  round_trips = 4 →
  return_passengers = 60 →
  total_passengers = 640 →
  ∃ (one_way_passengers : ℕ),
    one_way_passengers = 100 ∧
    total_passengers = round_trips * (one_way_passengers + return_passengers) :=
by sorry

end NUMINAMATH_CALUDE_train_passenger_count_l3598_359846


namespace NUMINAMATH_CALUDE_special_hexagon_perimeter_l3598_359883

/-- An equilateral hexagon with specified angle measures and area -/
structure SpecialHexagon where
  -- Side length of the hexagon
  side : ℝ
  -- Assertion that the hexagon is equilateral
  is_equilateral : True
  -- Assertion about the interior angles
  angle_condition : True
  -- Area of the hexagon
  area : ℝ
  -- The area is 12
  area_is_twelve : area = 12

/-- The perimeter of a SpecialHexagon is 12 -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : h.side * 6 = 12 := by
  sorry

#check special_hexagon_perimeter

end NUMINAMATH_CALUDE_special_hexagon_perimeter_l3598_359883


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_implies_divisibility_l3598_359863

theorem gcd_lcm_sum_implies_divisibility (a b : ℤ) 
  (h : Nat.gcd a.natAbs b.natAbs + Nat.lcm a.natAbs b.natAbs = a.natAbs + b.natAbs) : 
  a ∣ b ∨ b ∣ a := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_implies_divisibility_l3598_359863


namespace NUMINAMATH_CALUDE_decimal_place_150_of_5_6_l3598_359887

/-- The decimal representation of 5/6 -/
def decimal_rep_5_6 : ℚ := 5/6

/-- The period of the repeating decimal representation of 5/6 -/
def period : ℕ := 2

/-- The nth digit in the decimal representation of 5/6 -/
def nth_digit (n : ℕ) : ℕ :=
  if n % period = 1 then 8 else 3

/-- The 150th decimal place in the representation of 5/6 is 3 -/
theorem decimal_place_150_of_5_6 : nth_digit 150 = 3 := by
  sorry

end NUMINAMATH_CALUDE_decimal_place_150_of_5_6_l3598_359887


namespace NUMINAMATH_CALUDE_candy_probability_l3598_359839

def total_candies : ℕ := 20
def red_candies : ℕ := 10
def blue_candies : ℕ := 10

def probability_same_combination : ℚ := 118 / 323

theorem candy_probability : 
  total_candies = red_candies + blue_candies →
  probability_same_combination = 
    (2 * (red_candies * (red_candies - 1) * (red_candies - 2) * (red_candies - 3) + 
          blue_candies * (blue_candies - 1) * (blue_candies - 2) * (blue_candies - 3)) + 
     6 * red_candies * (red_candies - 1) * blue_candies * (blue_candies - 1)) / 
    (total_candies * (total_candies - 1) * (total_candies - 2) * (total_candies - 3)) :=
by sorry

end NUMINAMATH_CALUDE_candy_probability_l3598_359839


namespace NUMINAMATH_CALUDE_complex_imaginary_solution_l3598_359827

theorem complex_imaginary_solution (a : ℝ) : 
  (a - (5 : ℂ) / (2 - Complex.I)).im = (a - (5 : ℂ) / (2 - Complex.I)).re → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_imaginary_solution_l3598_359827


namespace NUMINAMATH_CALUDE_crackers_distribution_l3598_359854

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) :
  total_crackers = 8 →
  num_friends = 4 →
  total_crackers = num_friends * crackers_per_friend →
  crackers_per_friend = 2 := by
sorry

end NUMINAMATH_CALUDE_crackers_distribution_l3598_359854


namespace NUMINAMATH_CALUDE_circle_radius_zero_l3598_359870

/-- The radius of the circle described by the equation 4x^2 - 8x + 4y^2 - 16y + 20 = 0 is 0 -/
theorem circle_radius_zero (x y : ℝ) : 
  (4*x^2 - 8*x + 4*y^2 - 16*y + 20 = 0) → 
  ∃ (h k : ℝ), ∀ (x y : ℝ), 4*x^2 - 8*x + 4*y^2 - 16*y + 20 = 0 ↔ (x - h)^2 + (y - k)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_zero_l3598_359870


namespace NUMINAMATH_CALUDE_remainder_71_73_mod_8_l3598_359843

theorem remainder_71_73_mod_8 : (71 * 73) % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_71_73_mod_8_l3598_359843


namespace NUMINAMATH_CALUDE_king_not_right_mind_queen_indeterminate_l3598_359885

-- Define the mental states
inductive MentalState
| RightMind
| NotRightMind

-- Define the royals
structure Royal where
  name : String
  state : MentalState

-- Define the belief function
def believes (r : Royal) (p : Prop) : Prop := sorry

-- Define the King and Queen of Spades
def King : Royal := ⟨"King of Spades", MentalState.NotRightMind⟩
def Queen : Royal := ⟨"Queen of Spades", MentalState.NotRightMind⟩

-- The main theorem
theorem king_not_right_mind_queen_indeterminate :
  believes Queen (believes King (Queen.state = MentalState.NotRightMind)) →
  (King.state = MentalState.NotRightMind) ∧
  ((Queen.state = MentalState.RightMind) ∨ (Queen.state = MentalState.NotRightMind)) :=
by sorry

end NUMINAMATH_CALUDE_king_not_right_mind_queen_indeterminate_l3598_359885


namespace NUMINAMATH_CALUDE_students_who_got_on_correct_l3598_359866

/-- The number of students who got on the bus at the first stop -/
def students_who_got_on (initial_students final_students : ℝ) : ℝ :=
  final_students - initial_students

theorem students_who_got_on_correct (initial_students final_students : ℝ) 
  (h1 : initial_students = 10.0) (h2 : final_students = 13) :
  students_who_got_on initial_students final_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_students_who_got_on_correct_l3598_359866


namespace NUMINAMATH_CALUDE_min_bushes_for_zucchinis_l3598_359884

/-- The number of containers of blueberries yielded by one bush -/
def containers_per_bush : ℕ := 11

/-- The number of containers of blueberries needed to trade for one zucchini -/
def containers_per_zucchini : ℕ := 3

/-- The number of zucchinis Natalie wants to obtain -/
def target_zucchinis : ℕ := 60

/-- The function to calculate the number of bushes needed for a given number of zucchinis -/
def bushes_needed (zucchinis : ℕ) : ℕ :=
  (zucchinis * containers_per_zucchini + containers_per_bush - 1) / containers_per_bush

theorem min_bushes_for_zucchinis :
  bushes_needed target_zucchinis = 17 := by sorry

end NUMINAMATH_CALUDE_min_bushes_for_zucchinis_l3598_359884


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l3598_359804

/-- The equation of the common chord of two circles -/
theorem common_chord_of_circles (x y : ℝ) :
  (x^2 + y^2 - 4*x - 3 = 0) ∧ (x^2 + y^2 - 4*y - 3 = 0) → (x - y = 0) := by
  sorry

#check common_chord_of_circles

end NUMINAMATH_CALUDE_common_chord_of_circles_l3598_359804


namespace NUMINAMATH_CALUDE_tangent_line_proofs_l3598_359805

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_line_proofs :
  let e := Real.exp 1
  -- Tangent line at (e, e^e)
  ∃ (m : ℝ), ∀ x y : ℝ,
    (y = f x) → (x = e ∧ y = f e) →
    (m * (x - e) + f e = y ∧ m * x - y - m * e + f e = 0) →
    (Real.exp e * x - y - Real.exp (e + 1) = 0) ∧
  -- Tangent line from origin
  ∃ (k : ℝ), ∀ x y : ℝ,
    (y = f x) → (y = k * x) →
    (k = f x ∧ k = (f x) / x) →
    (e * x - y = 0) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_proofs_l3598_359805


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l3598_359809

theorem tangent_line_to_parabola (x y : ℝ) :
  let f : ℝ → ℝ := λ t => t^2
  let tangent_point : ℝ × ℝ := (1, 1)
  let slope : ℝ := 2 * tangent_point.1
  2 * x - y - 1 = 0 ↔ y = slope * (x - tangent_point.1) + tangent_point.2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l3598_359809


namespace NUMINAMATH_CALUDE_largest_divisor_of_P_l3598_359838

def P (n : ℕ) : ℕ := (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9)

theorem largest_divisor_of_P (n : ℕ) (h : Even n) (k : ℕ) :
  (∀ m : ℕ, Even m → k ∣ P m) → k ≤ 15 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_P_l3598_359838


namespace NUMINAMATH_CALUDE_max_value_polynomial_l3598_359857

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 ≤ 656^2 / 18 :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l3598_359857


namespace NUMINAMATH_CALUDE_candle_ratio_l3598_359868

/-- Proves the ratio of candles Alyssa used to total candles -/
theorem candle_ratio :
  ∀ (total candles_used_by_alyssa : ℕ) (chelsea_usage_percent : ℚ),
  total = 40 →
  chelsea_usage_percent = 70 / 100 →
  candles_used_by_alyssa + 
    (chelsea_usage_percent * (total - candles_used_by_alyssa)).floor + 6 = total →
  candles_used_by_alyssa * 2 = total := by
  sorry

#check candle_ratio

end NUMINAMATH_CALUDE_candle_ratio_l3598_359868


namespace NUMINAMATH_CALUDE_most_economical_cost_l3598_359879

/-- Represents the problem of finding the most economical cost for purchasing warm reminder signs and garbage bins. -/
theorem most_economical_cost
  (price_difference : ℕ)
  (price_ratio : ℕ)
  (total_items : ℕ)
  (max_cost : ℕ)
  (bin_sign_ratio : ℚ)
  (h1 : price_difference = 350)
  (h2 : price_ratio = 3)
  (h3 : total_items = 3000)
  (h4 : max_cost = 350000)
  (h5 : bin_sign_ratio = 3/2)
  : ∃ (sign_price bin_price : ℕ) (num_signs num_bins : ℕ) (total_cost : ℕ),
    -- Price relationship between bins and signs
    4 * bin_price - 5 * sign_price = price_difference ∧
    bin_price = price_ratio * sign_price ∧
    -- Total number of items constraint
    num_signs + num_bins = total_items ∧
    -- Cost constraint
    num_signs * sign_price + num_bins * bin_price ≤ max_cost ∧
    -- Ratio constraint
    (num_bins : ℚ) ≥ bin_sign_ratio * (num_signs : ℚ) ∧
    -- Most economical solution
    num_signs = 1200 ∧
    total_cost = 330000 ∧
    -- No cheaper solution exists
    ∀ (other_signs : ℕ), 
      other_signs ≠ num_signs →
      other_signs + (total_items - other_signs) = total_items →
      (total_items - other_signs : ℚ) ≥ bin_sign_ratio * (other_signs : ℚ) →
      other_signs * sign_price + (total_items - other_signs) * bin_price ≥ total_cost :=
by sorry


end NUMINAMATH_CALUDE_most_economical_cost_l3598_359879


namespace NUMINAMATH_CALUDE_estimate_N_l3598_359830

-- Define f(n) as the largest prime factor of n
def f (n : ℕ) : ℕ := sorry

-- Define the sum of f(n^2-1) for n from 2 to 10^6
def sum_f_nsquared_minus_one : ℕ := sorry

-- Define the sum of f(n) for n from 2 to 10^6
def sum_f_n : ℕ := sorry

-- Theorem statement
theorem estimate_N : 
  ⌊(10^4 : ℝ) * (sum_f_nsquared_minus_one : ℝ) / (sum_f_n : ℝ)⌋ = 18215 := by sorry

end NUMINAMATH_CALUDE_estimate_N_l3598_359830


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l3598_359819

theorem largest_number_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b / a = 5 / 4 →
  c / a = 6 / 4 →
  (a + b + c) / 3 = 20 →
  c = 24 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l3598_359819


namespace NUMINAMATH_CALUDE_paco_salty_cookies_left_l3598_359873

/-- Represents the number of cookies Paco has --/
structure PacoCookies where
  initial_salty : ℕ
  eaten_salty : ℕ

/-- Calculates the number of salty cookies Paco has left --/
def salty_cookies_left (cookies : PacoCookies) : ℕ :=
  cookies.initial_salty - cookies.eaten_salty

/-- Theorem stating that Paco has 17 salty cookies left --/
theorem paco_salty_cookies_left :
  ∃ (cookies : PacoCookies),
    cookies.initial_salty = 26 ∧
    cookies.eaten_salty = 9 ∧
    salty_cookies_left cookies = 17 := by
  sorry

end NUMINAMATH_CALUDE_paco_salty_cookies_left_l3598_359873


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3598_359823

theorem quadratic_equation_solution (x : ℝ) : 
  x^2 - 2*x - 4 = 0 ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3598_359823


namespace NUMINAMATH_CALUDE_problem_statement_l3598_359845

def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {x | x < a}
def M (m : ℝ) : Set ℝ := {x | x^2 - (1+m)*x + m = 0}

theorem problem_statement (a m : ℝ) (h : m > 1) :
  (A ∩ B a = A → a > 2) ∧
  (m ≠ 2 → A ∪ M m = {1, 2, m}) ∧
  (m = 2 → A ∪ M m = {1, 2}) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3598_359845


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l3598_359867

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 2, c = 2√3, and C = π/3, then b = 4 -/
theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) : 
  (a = 2) → (c = 2 * Real.sqrt 3) → (C = π / 3) → (b = 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l3598_359867


namespace NUMINAMATH_CALUDE_joan_initial_dimes_l3598_359858

/-- The number of dimes Joan spent -/
def dimes_spent : ℕ := 2

/-- The number of dimes Joan has left -/
def dimes_left : ℕ := 3

/-- The initial number of dimes Joan had -/
def initial_dimes : ℕ := dimes_spent + dimes_left

theorem joan_initial_dimes : initial_dimes = 5 := by sorry

end NUMINAMATH_CALUDE_joan_initial_dimes_l3598_359858


namespace NUMINAMATH_CALUDE_parallelogram_height_l3598_359859

theorem parallelogram_height (b h_t : ℝ) (h_t_pos : h_t > 0) :
  let a_t := b * h_t / 2
  let h_p := h_t / 2
  let a_p := b * h_p
  h_t = 10 → a_t = a_p → h_p = 5 := by sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3598_359859


namespace NUMINAMATH_CALUDE_ab_length_l3598_359851

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
def collinear (A B C D : ℝ × ℝ) : Prop := sorry
def distance (P Q : ℝ × ℝ) : ℝ := sorry
def perimeter (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem ab_length
  (h_collinear : collinear A B C D)
  (h_ab_cd : distance A B = distance C D)
  (h_bc : distance B C = 8)
  (h_be : distance B E = 12)
  (h_ce : distance C E = 12)
  (h_perimeter : perimeter A E D = 3 * perimeter B E C) :
  distance A B = 18 := by sorry

end NUMINAMATH_CALUDE_ab_length_l3598_359851


namespace NUMINAMATH_CALUDE_increasing_linear_function_positive_slope_l3598_359880

/-- A linear function f(x) = mx + b -/
def LinearFunction (m b : ℝ) : ℝ → ℝ := fun x ↦ m * x + b

/-- A function is increasing if for any x₁ < x₂, f(x₁) < f(x₂) -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

theorem increasing_linear_function_positive_slope (m b : ℝ) :
  IsIncreasing (LinearFunction m b) → m > 0 := by
  sorry

end NUMINAMATH_CALUDE_increasing_linear_function_positive_slope_l3598_359880


namespace NUMINAMATH_CALUDE_shortest_paths_count_julia_paths_count_l3598_359807

theorem shortest_paths_count : Nat → Nat → Nat
| m, n => Nat.choose (m + n) m

theorem julia_paths_count : shortest_paths_count 8 5 = 1287 := by
  sorry

end NUMINAMATH_CALUDE_shortest_paths_count_julia_paths_count_l3598_359807


namespace NUMINAMATH_CALUDE_bathroom_visit_interval_l3598_359844

/-- Calculates the time between bathroom visits during a movie -/
theorem bathroom_visit_interval (movie_duration : Real) (visit_count : Nat) : 
  movie_duration = 2.5 ∧ visit_count = 3 → 
  (movie_duration * 60) / (visit_count + 1) = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_visit_interval_l3598_359844


namespace NUMINAMATH_CALUDE_inequality_proof_l3598_359847

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^2 + a*b + b^2 = 3*c^2) (h2 : a^3 + a^2*b + a*b^2 + b^3 = 4*d^3) :
  a + b + d ≤ 3*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3598_359847


namespace NUMINAMATH_CALUDE_even_sequence_sum_l3598_359874

theorem even_sequence_sum (n : ℕ) (sum : ℕ) : sum = n * (n + 1) → 2 * n = 30 :=
  sorry

#check even_sequence_sum

end NUMINAMATH_CALUDE_even_sequence_sum_l3598_359874


namespace NUMINAMATH_CALUDE_solution_volume_proof_l3598_359898

-- Define the volume of pure acid in liters
def pure_acid_volume : ℝ := 1.6

-- Define the concentration of the solution as a percentage
def solution_concentration : ℝ := 20

-- Define the total volume of the solution in liters
def total_solution_volume : ℝ := 8

-- Theorem to prove
theorem solution_volume_proof :
  pure_acid_volume = (solution_concentration / 100) * total_solution_volume :=
by sorry

end NUMINAMATH_CALUDE_solution_volume_proof_l3598_359898


namespace NUMINAMATH_CALUDE_initial_bureaus_correct_l3598_359820

/-- The number of offices -/
def num_offices : ℕ := 14

/-- The additional bureaus needed for equal distribution -/
def additional_bureaus : ℕ := 10

/-- The initial number of bureaus -/
def initial_bureaus : ℕ := 8

/-- Theorem stating that the initial number of bureaus is correct -/
theorem initial_bureaus_correct :
  ∃ (x : ℕ), (initial_bureaus + additional_bureaus = num_offices * x) ∧
             (∀ y : ℕ, initial_bureaus ≠ num_offices * y) :=
by sorry

end NUMINAMATH_CALUDE_initial_bureaus_correct_l3598_359820


namespace NUMINAMATH_CALUDE_helga_extra_hours_thursday_l3598_359861

/-- Represents Helga's work schedule and article production --/
structure HelgaWorkSchedule where
  articles_per_30min : ℕ
  normal_hours_per_day : ℕ
  normal_days_per_week : ℕ
  articles_this_week : ℕ
  extra_hours_friday : ℕ

/-- Calculates the number of extra hours Helga worked on Thursday --/
def extra_hours_thursday (schedule : HelgaWorkSchedule) : ℕ :=
  sorry

/-- Theorem stating that given Helga's work schedule, she worked 2 extra hours on Thursday --/
theorem helga_extra_hours_thursday 
  (schedule : HelgaWorkSchedule)
  (h1 : schedule.articles_per_30min = 5)
  (h2 : schedule.normal_hours_per_day = 4)
  (h3 : schedule.normal_days_per_week = 5)
  (h4 : schedule.articles_this_week = 250)
  (h5 : schedule.extra_hours_friday = 3) :
  extra_hours_thursday schedule = 2 :=
sorry

end NUMINAMATH_CALUDE_helga_extra_hours_thursday_l3598_359861


namespace NUMINAMATH_CALUDE_sms_authenticity_l3598_359836

/-- Represents an SMS message -/
structure SMS where
  content : String
  sender : String

/-- Represents a bank card -/
structure BankCard where
  number : String
  bank : String
  officialPhoneNumber : String

/-- Represents a bank's SMS characteristics -/
structure BankSMSCharacteristics where
  shortNumber : String
  messageFormat : String → Bool

/-- Determines if an SMS is genuine based on comparison and bank confirmation -/
def isGenuineSMS (message : SMS) (card : BankCard) (prevMessages : List SMS) 
                 (bankCharacteristics : BankSMSCharacteristics) : Prop :=
  (∃ prev ∈ prevMessages, message.sender = prev.sender ∧ 
                          bankCharacteristics.messageFormat message.content) ∧
  (∃ confirmation : Bool, confirmation = true)

/-- Main theorem: An SMS is genuine iff it matches previous messages and is confirmed by the bank -/
theorem sms_authenticity 
  (message : SMS) 
  (card : BankCard) 
  (prevMessages : List SMS) 
  (bankCharacteristics : BankSMSCharacteristics) :
  isGenuineSMS message card prevMessages bankCharacteristics ↔ 
  (∃ prev ∈ prevMessages, message.sender = prev.sender ∧ 
                          bankCharacteristics.messageFormat message.content) ∧
  (∃ confirmation : Bool, confirmation = true) :=
by sorry


end NUMINAMATH_CALUDE_sms_authenticity_l3598_359836


namespace NUMINAMATH_CALUDE_simplify_expression_l3598_359840

theorem simplify_expression (a : ℝ) (ha : a > 0) :
  a^2 / (a * (a^3)^(1/2))^(1/3) = a^(7/6) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3598_359840


namespace NUMINAMATH_CALUDE_salt_mixture_proof_l3598_359865

theorem salt_mixture_proof :
  let initial_amount : ℝ := 150
  let initial_concentration : ℝ := 0.35
  let added_amount : ℝ := 120
  let added_concentration : ℝ := 0.80
  let final_concentration : ℝ := 0.55
  
  (initial_amount * initial_concentration + added_amount * added_concentration) / (initial_amount + added_amount) = final_concentration :=
by
  sorry

end NUMINAMATH_CALUDE_salt_mixture_proof_l3598_359865


namespace NUMINAMATH_CALUDE_no_friendly_triplet_in_small_range_exists_friendly_triplet_in_large_range_l3598_359895

-- Define friendly integers
def friendly (a b c : ℤ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a ∣ b * c ∨ b ∣ a * c ∨ c ∣ a * b)

theorem no_friendly_triplet_in_small_range (n : ℕ) :
  ¬∃ a b c : ℤ, n^2 < a ∧ a < b ∧ b < c ∧ c < n^2 + n ∧ friendly a b c := by
  sorry

theorem exists_friendly_triplet_in_large_range (n : ℕ) :
  ∃ a b c : ℤ, n^2 < a ∧ a < b ∧ b < c ∧ c < n^2 + n + 3 * Real.sqrt n ∧ friendly a b c := by
  sorry

end NUMINAMATH_CALUDE_no_friendly_triplet_in_small_range_exists_friendly_triplet_in_large_range_l3598_359895


namespace NUMINAMATH_CALUDE_digit_one_more_frequent_than_zero_l3598_359802

def concatenated_sequence (n : ℕ) : String :=
  String.join (List.map toString (List.range n))

def count_digit (s : String) (d : Char) : ℕ :=
  s.toList.filter (· = d) |>.length

theorem digit_one_more_frequent_than_zero (n : ℕ) :
  count_digit (concatenated_sequence n) '1' > count_digit (concatenated_sequence n) '0' :=
sorry

end NUMINAMATH_CALUDE_digit_one_more_frequent_than_zero_l3598_359802


namespace NUMINAMATH_CALUDE_sandy_final_position_l3598_359860

-- Define the coordinate system
def Position := ℝ × ℝ

-- Define the starting position
def start : Position := (0, 0)

-- Define Sandy's movements
def move_south (p : Position) (distance : ℝ) : Position :=
  (p.1, p.2 - distance)

def move_east (p : Position) (distance : ℝ) : Position :=
  (p.1 + distance, p.2)

def move_north (p : Position) (distance : ℝ) : Position :=
  (p.1, p.2 + distance)

-- Define Sandy's final position after her movements
def final_position : Position :=
  move_east (move_north (move_east (move_south start 20) 20) 20) 20

-- Theorem to prove
theorem sandy_final_position :
  final_position = (40, 0) := by sorry

end NUMINAMATH_CALUDE_sandy_final_position_l3598_359860


namespace NUMINAMATH_CALUDE_max_xy_value_l3598_359808

theorem max_xy_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : 6 * x + 8 * y = 72) :
  x * y ≤ 27 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 6 * x₀ + 8 * y₀ = 72 ∧ x₀ * y₀ = 27 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l3598_359808


namespace NUMINAMATH_CALUDE_train_length_proof_l3598_359877

def train_problem (distance_apart : ℝ) (train2_length : ℝ) (speed1 : ℝ) (speed2 : ℝ) (time_to_meet : ℝ) : Prop :=
  let speed1_ms := speed1 * 1000 / 3600
  let speed2_ms := speed2 * 1000 / 3600
  let relative_speed := speed1_ms + speed2_ms
  let distance_covered := relative_speed * time_to_meet
  let train1_length := distance_covered - train2_length
  train1_length = 430

theorem train_length_proof :
  train_problem 630 200 90 72 13.998880089592832 :=
sorry

end NUMINAMATH_CALUDE_train_length_proof_l3598_359877


namespace NUMINAMATH_CALUDE_sin_cos_power_inequality_l3598_359864

theorem sin_cos_power_inequality (x : Real) (h : 0 < x ∧ x < Real.pi / 4) :
  (Real.sin x) ^ (Real.sin x) < (Real.cos x) ^ (Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_power_inequality_l3598_359864


namespace NUMINAMATH_CALUDE_min_ratio_case1_min_ratio_case2_min_ratio_case3_min_ratio_case4_l3598_359899

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the ratio function
def ratio (n : ℕ) : ℚ := n / (sumOfDigits n)

-- Theorem for case (i)
theorem min_ratio_case1 :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → ratio n ≥ 19/10 := by sorry

-- Theorem for case (ii)
theorem min_ratio_case2 :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → ratio n ≥ 119/11 := by sorry

-- Theorem for case (iii)
theorem min_ratio_case3 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → ratio n ≥ 1119/12 := by sorry

-- Theorem for case (iv)
theorem min_ratio_case4 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 → ratio n ≥ 11119/13 := by sorry

end NUMINAMATH_CALUDE_min_ratio_case1_min_ratio_case2_min_ratio_case3_min_ratio_case4_l3598_359899


namespace NUMINAMATH_CALUDE_barry_vitamin_d3_days_l3598_359822

/-- Calculates the number of days Barry was told to take vitamin D3 -/
def vitaminD3Days (capsules_per_bottle : ℕ) (capsules_per_day : ℕ) (bottles_needed : ℕ) : ℕ :=
  (capsules_per_bottle / capsules_per_day) * bottles_needed

theorem barry_vitamin_d3_days :
  vitaminD3Days 60 2 6 = 180 := by
  sorry

end NUMINAMATH_CALUDE_barry_vitamin_d3_days_l3598_359822


namespace NUMINAMATH_CALUDE_inequality_proof_l3598_359849

theorem inequality_proof (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1)^2) :
  y * (y - 1) ≤ x^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3598_359849


namespace NUMINAMATH_CALUDE_product_xy_on_line_k_l3598_359862

/-- A line passing through the origin with slope 1/4 -/
def line_k (x y : ℝ) : Prop := y = (1/4) * x

theorem product_xy_on_line_k :
  ∀ x y : ℝ,
  line_k x 8 → line_k 20 y →
  x * y = 160 := by
sorry

end NUMINAMATH_CALUDE_product_xy_on_line_k_l3598_359862


namespace NUMINAMATH_CALUDE_x_over_y_is_negative_two_l3598_359826

theorem x_over_y_is_negative_two (x y : ℝ) (h1 : 1 < (x - y) / (x + y))
  (h2 : (x - y) / (x + y) < 3) (h3 : ∃ (n : ℤ), x / y = n) :
  x / y = -2 := by
  sorry

end NUMINAMATH_CALUDE_x_over_y_is_negative_two_l3598_359826


namespace NUMINAMATH_CALUDE_fiction_books_count_l3598_359872

theorem fiction_books_count (total : ℕ) (picture_books : ℕ) : 
  total = 35 → picture_books = 11 → ∃ (fiction : ℕ), 
    fiction + (fiction + 4) + 2 * fiction + picture_books = total ∧ fiction = 5 := by
  sorry

end NUMINAMATH_CALUDE_fiction_books_count_l3598_359872


namespace NUMINAMATH_CALUDE_tangent_points_coordinates_fixed_points_on_circle_l3598_359853

/-- Circle M with equation x^2 + (y-2)^2 = 1 -/
def circle_M (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

/-- Line l with equation x - 2y = 0 -/
def line_l (x y : ℝ) : Prop := x - 2*y = 0

/-- Point P lies on line l -/
def P_on_line_l (x y : ℝ) : Prop := line_l x y

/-- PA and PB are tangents to circle M -/
def tangents_to_M (xp yp xa ya xb yb : ℝ) : Prop :=
  circle_M xa ya ∧ circle_M xb yb ∧
  ((xp - xa) * xa + (yp - ya) * (ya - 2) = 0) ∧
  ((xp - xb) * xb + (yp - yb) * (yb - 2) = 0)

/-- Angle APB is 60 degrees -/
def angle_APB_60 (xp yp xa ya xb yb : ℝ) : Prop :=
  let v1x := xa - xp
  let v1y := ya - yp
  let v2x := xb - xp
  let v2y := yb - yp
  (v1x * v2x + v1y * v2y)^2 = 3 * ((v1x^2 + v1y^2) * (v2x^2 + v2y^2)) / 4

theorem tangent_points_coordinates :
  ∀ (xp yp xa ya xb yb : ℝ),
  P_on_line_l xp yp →
  tangents_to_M xp yp xa ya xb yb →
  angle_APB_60 xp yp xa ya xb yb →
  (xp = 0 ∧ yp = 0) ∨ (xp = 8/5 ∧ yp = 4/5) :=
sorry

theorem fixed_points_on_circle :
  ∀ (xp yp xa ya : ℝ),
  P_on_line_l xp yp →
  tangents_to_M xp yp xa ya xp yp →
  ∃ (t : ℝ),
  (1 - t) * xp + t * xa = 0 ∧ (1 - t) * yp + t * ya = 2 ∨
  (1 - t) * xp + t * xa = 4/5 ∧ (1 - t) * yp + t * ya = 2/5 :=
sorry

end NUMINAMATH_CALUDE_tangent_points_coordinates_fixed_points_on_circle_l3598_359853


namespace NUMINAMATH_CALUDE_candied_apple_price_l3598_359878

/-- Given the conditions of candy production and sales, prove the price of each candied apple. -/
theorem candied_apple_price :
  ∀ (num_apples num_grapes : ℕ) (grape_price total_earnings : ℚ),
    num_apples = 15 →
    num_grapes = 12 →
    grape_price = 3/2 →
    total_earnings = 48 →
    ∃ (apple_price : ℚ),
      apple_price * num_apples + grape_price * num_grapes = total_earnings ∧
      apple_price = 2 := by
sorry

end NUMINAMATH_CALUDE_candied_apple_price_l3598_359878


namespace NUMINAMATH_CALUDE_john_total_distance_l3598_359829

-- Define the driving segments
def segment1_speed : ℝ := 55
def segment1_time : ℝ := 2.5
def segment2_speed : ℝ := 65
def segment2_time : ℝ := 3.25
def segment3_speed : ℝ := 50
def segment3_time : ℝ := 4

-- Define the total distance function
def total_distance : ℝ :=
  segment1_speed * segment1_time +
  segment2_speed * segment2_time +
  segment3_speed * segment3_time

-- Theorem statement
theorem john_total_distance :
  total_distance = 548.75 := by
  sorry

end NUMINAMATH_CALUDE_john_total_distance_l3598_359829


namespace NUMINAMATH_CALUDE_helga_usual_work_hours_l3598_359855

/-- Helga's work schedule and article writing capacity -/
structure HelgaWork where
  articles_per_30min : ℕ
  days_per_week : ℕ
  extra_hours_thursday : ℕ
  extra_hours_friday : ℕ
  total_articles_this_week : ℕ

/-- Calculate Helga's usual daily work hours -/
def usual_daily_hours (hw : HelgaWork) : ℚ :=
  let articles_per_hour : ℚ := (hw.articles_per_30min : ℚ) * 2
  let total_hours_this_week : ℚ := (hw.total_articles_this_week : ℚ) / articles_per_hour
  let usual_hours_this_week : ℚ := total_hours_this_week - (hw.extra_hours_thursday + hw.extra_hours_friday)
  usual_hours_this_week / (hw.days_per_week : ℚ)

/-- Theorem: Helga usually works 4 hours each day -/
theorem helga_usual_work_hours (hw : HelgaWork)
  (h1 : hw.articles_per_30min = 5)
  (h2 : hw.days_per_week = 5)
  (h3 : hw.extra_hours_thursday = 2)
  (h4 : hw.extra_hours_friday = 3)
  (h5 : hw.total_articles_this_week = 250) :
  usual_daily_hours hw = 4 := by
  sorry

end NUMINAMATH_CALUDE_helga_usual_work_hours_l3598_359855


namespace NUMINAMATH_CALUDE_final_depth_calculation_l3598_359816

/-- Calculates the final depth aimed to dig given initial and new working conditions -/
theorem final_depth_calculation 
  (initial_men : ℕ) 
  (initial_hours : ℕ) 
  (initial_depth : ℕ) 
  (extra_men : ℕ) 
  (new_hours : ℕ) : 
  initial_men = 75 → 
  initial_hours = 8 → 
  initial_depth = 50 → 
  extra_men = 65 → 
  new_hours = 6 → 
  (initial_men + extra_men) * new_hours * initial_depth = initial_men * initial_hours * 70 := by
  sorry

#check final_depth_calculation

end NUMINAMATH_CALUDE_final_depth_calculation_l3598_359816


namespace NUMINAMATH_CALUDE_knight_position_proof_l3598_359803

/-- The total number of people in the line -/
def total_people : ℕ := 2022

/-- The position of the knight from the left -/
def knight_position : ℕ := 48

/-- The ratio of liars to the right compared to the left for each person (except the ends) -/
def liar_ratio : ℕ := 42

theorem knight_position_proof :
  ∀ k : ℕ, 
  1 < k ∧ k < total_people →
  (total_people - k = liar_ratio * (k - 1)) ↔ 
  k = knight_position :=
sorry

end NUMINAMATH_CALUDE_knight_position_proof_l3598_359803


namespace NUMINAMATH_CALUDE_sector_radius_l3598_359834

theorem sector_radius (r : ℝ) (h1 : r > 0) : 
  (r = r) →  -- radius equals arc length
  ((3 * r) / ((1/2) * r^2) = 2) →  -- ratio of perimeter to area is 2
  r = 3 := by
sorry

end NUMINAMATH_CALUDE_sector_radius_l3598_359834


namespace NUMINAMATH_CALUDE_suitcase_weight_problem_l3598_359815

/-- Proves that given the initial ratio of books to clothes to electronics as 7:4:3, 
    and the fact that removing 6 pounds of clothing doubles the ratio of books to clothes, 
    the weight of electronics is 9 pounds. -/
theorem suitcase_weight_problem (B C E : ℝ) : 
  (B / C = 7 / 4) →  -- Initial ratio of books to clothes
  (C / E = 4 / 3) →  -- Initial ratio of clothes to electronics
  (B / (C - 6) = 7 / 2) →  -- New ratio after removing 6 pounds of clothes
  E = 9 := by
sorry

end NUMINAMATH_CALUDE_suitcase_weight_problem_l3598_359815


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3598_359806

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the condition for any point P on the ellipse
def point_condition (PF1 PF2 : ℝ) : Prop :=
  PF1 + PF2 = 2 * Real.sqrt 2

-- Define the focal distance
def focal_distance : ℝ := 2

-- Define the intersecting line
def intersecting_line (x y t : ℝ) : Prop :=
  x - y + t = 0

-- Define the circle condition for the midpoint of AB
def midpoint_condition (x y : ℝ) : Prop :=
  x^2 + y^2 > 10/9

theorem ellipse_and_line_intersection :
  ∀ (a b : ℝ),
  (∀ (x y : ℝ), ellipse_C x y a b → ∃ (PF1 PF2 : ℝ), point_condition PF1 PF2) →
  (a^2 - b^2 = focal_distance^2) →
  (∀ (x y : ℝ), ellipse_C x y a b ↔ x^2 / 2 + y^2 = 1) ∧
  (∀ (t : ℝ),
    (∃ (x1 y1 x2 y2 : ℝ),
      ellipse_C x1 y1 a b ∧
      ellipse_C x2 y2 a b ∧
      intersecting_line x1 y1 t ∧
      intersecting_line x2 y2 t ∧
      x1 ≠ x2 ∧
      midpoint_condition ((x1 + x2) / 2) ((y1 + y2) / 2)) →
    (-Real.sqrt 3 < t ∧ t ≤ -Real.sqrt 2) ∨ (Real.sqrt 2 ≤ t ∧ t < Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3598_359806


namespace NUMINAMATH_CALUDE_f_theorem_l3598_359890

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂) ∧
  (∀ x₁ x₂, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → f x₁ < f x₂) ∧
  (f 1 = 2)

theorem f_theorem (f : ℝ → ℝ) (h : f_properties f) :
  (∀ x₁ x₂, 0 ≤ x₁ → x₁ < x₂ → -(f x₁)^2 > -(f x₂)^2) ∧
  (∀ x₁ x₂, x₁ < x₂ → x₂ ≤ 0 → -(f x₁)^2 < -(f x₂)^2) ∧
  (∀ a, f (2 * a^2 - 1) + 2 * f a - 6 < 0 ↔ -2 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_f_theorem_l3598_359890


namespace NUMINAMATH_CALUDE_eagles_score_l3598_359886

theorem eagles_score (total_points hawks_margin : ℕ) 
  (h1 : total_points = 50)
  (h2 : hawks_margin = 6) : 
  (total_points - hawks_margin) / 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_eagles_score_l3598_359886


namespace NUMINAMATH_CALUDE_system_solution_l3598_359818

theorem system_solution (x y : ℝ) : 
  (x - y = 2 ∧ 3 * x + y = 4) ↔ (x = 1.5 ∧ y = -0.5) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3598_359818


namespace NUMINAMATH_CALUDE_car_stopping_distance_l3598_359817

/-- Represents the distance traveled by a car in a given second -/
def distance_per_second (n : ℕ) : ℕ :=
  max (40 - 10 * n) 0

/-- Calculates the total distance traveled by the car -/
def total_distance : ℕ :=
  (List.range 5).map distance_per_second |>.sum

/-- Theorem: The total distance traveled by the car is 100 feet -/
theorem car_stopping_distance : total_distance = 100 := by
  sorry

#eval total_distance

end NUMINAMATH_CALUDE_car_stopping_distance_l3598_359817


namespace NUMINAMATH_CALUDE_diary_theorem_l3598_359812

def diary_problem (initial_diaries : ℕ) : ℕ :=
  let doubled := initial_diaries * 2
  let total := initial_diaries + doubled
  total - (total / 4)

theorem diary_theorem : diary_problem 8 = 18 := by
  sorry

end NUMINAMATH_CALUDE_diary_theorem_l3598_359812


namespace NUMINAMATH_CALUDE_circle_ratio_l3598_359881

theorem circle_ratio (α : Real) (r R x : Real) : 
  r > 0 → R > 0 → x > 0 → r < R →
  (R - r) = (R + r) * Real.sin α →
  x = (r * R) / ((Real.sqrt r + Real.sqrt R)^2) →
  (r / x) = 2 * (1 + Real.cos α) / (1 + Real.sin α) := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_l3598_359881


namespace NUMINAMATH_CALUDE_complex_square_one_plus_i_l3598_359800

theorem complex_square_one_plus_i (i : ℂ) : i * i = -1 → (1 + i)^2 = 2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_square_one_plus_i_l3598_359800


namespace NUMINAMATH_CALUDE_spring_sales_calculation_l3598_359871

-- Define the total annual sandwich sales
def total_sales : ℝ := 15

-- Define the seasonal sales
def winter_sales : ℝ := 3
def summer_sales : ℝ := 4
def fall_sales : ℝ := 5

-- Define the winter sales percentage
def winter_percentage : ℝ := 0.2

-- Theorem to prove
theorem spring_sales_calculation :
  ∃ (spring_sales : ℝ),
    winter_percentage * total_sales = winter_sales ∧
    spring_sales + summer_sales + fall_sales + winter_sales = total_sales ∧
    spring_sales = 3 := by
  sorry


end NUMINAMATH_CALUDE_spring_sales_calculation_l3598_359871


namespace NUMINAMATH_CALUDE_tank_capacity_l3598_359893

theorem tank_capacity (initial_fraction : Rat) (added_amount : Rat) (final_fraction : Rat) :
  initial_fraction = 3/4 →
  added_amount = 9 →
  final_fraction = 9/10 →
  ∃ (capacity : Rat), capacity = 60 ∧
    final_fraction * capacity - initial_fraction * capacity = added_amount :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l3598_359893


namespace NUMINAMATH_CALUDE_oranges_added_correct_l3598_359842

/-- The number of oranges added to make apples 50% of the total fruit -/
def oranges_added (initial_apples initial_oranges : ℕ) : ℕ :=
  let total := initial_apples + initial_oranges
  (2 * total) - initial_oranges

theorem oranges_added_correct (initial_apples initial_oranges : ℕ) :
  initial_apples = 10 →
  initial_oranges = 5 →
  oranges_added initial_apples initial_oranges = 5 :=
by
  sorry

#eval oranges_added 10 5

end NUMINAMATH_CALUDE_oranges_added_correct_l3598_359842


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l3598_359856

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number we're working with (1.12 million) -/
def number : ℝ := 1.12e6

/-- The claimed scientific notation of the number -/
def claimed_notation : ScientificNotation := {
  coefficient := 1.12,
  exponent := 6,
  is_valid := by sorry
}

/-- Theorem stating that the claimed notation is correct for the given number -/
theorem scientific_notation_correct : 
  number = claimed_notation.coefficient * (10 : ℝ) ^ claimed_notation.exponent := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l3598_359856


namespace NUMINAMATH_CALUDE_shaded_region_area_l3598_359828

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- A line defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- The shaded region formed by two intersecting lines and a right triangle -/
structure ShadedRegion where
  line1 : Line
  line2 : Line

def area_of_shaded_region (region : ShadedRegion) : ℚ :=
  sorry

theorem shaded_region_area :
  let line1 := Line.mk (Point.mk 0 5) (Point.mk 10 2)
  let line2 := Line.mk (Point.mk 2 6) (Point.mk 9 0)
  let region := ShadedRegion.mk line1 line2
  area_of_shaded_region region = 151425 / 3136 := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_area_l3598_359828


namespace NUMINAMATH_CALUDE_library_checkout_false_implication_l3598_359821

-- Define the universe of books in the library
variable (Book : Type)

-- Define a predicate for books available for checkout
variable (available_for_checkout : Book → Prop)

-- The main theorem
theorem library_checkout_false_implication 
  (h : ¬ ∀ (b : Book), available_for_checkout b) :
  (∃ (b : Book), ¬ available_for_checkout b) ∧ 
  (¬ ∀ (b : Book), available_for_checkout b) := by
  sorry

end NUMINAMATH_CALUDE_library_checkout_false_implication_l3598_359821


namespace NUMINAMATH_CALUDE_factorization_equality_l3598_359814

theorem factorization_equality (x : ℝ) : 5*x*(x+2) + 9*(x+2) = (x+2)*(5*x+9) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3598_359814


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3598_359801

theorem fractional_equation_solution :
  ∀ x : ℝ, x ≠ 0 → x ≠ 3 → (2 / (x - 3) = 3 / x) ↔ x = 9 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3598_359801


namespace NUMINAMATH_CALUDE_eight_digit_non_decreasing_integers_mod_1000_l3598_359894

/-- The number of ways to distribute n indistinguishable objects into k distinguishable bins -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) k

/-- The number of 8-digit positive integers with non-decreasing digits -/
def M : ℕ := stars_and_bars 8 9

theorem eight_digit_non_decreasing_integers_mod_1000 : M % 1000 = 870 := by sorry

end NUMINAMATH_CALUDE_eight_digit_non_decreasing_integers_mod_1000_l3598_359894


namespace NUMINAMATH_CALUDE_picture_fit_count_l3598_359889

-- Define the number of pictures for each category for Ralph and Derrick
def ralph_wild_animals : ℕ := 75
def ralph_landscapes : ℕ := 36
def ralph_family_events : ℕ := 45
def ralph_cars : ℕ := 20

def derrick_wild_animals : ℕ := 95
def derrick_landscapes : ℕ := 42
def derrick_family_events : ℕ := 55
def derrick_cars : ℕ := 25
def derrick_airplanes : ℕ := 10

-- Calculate total pictures for Ralph and Derrick
def ralph_total : ℕ := ralph_wild_animals + ralph_landscapes + ralph_family_events + ralph_cars
def derrick_total : ℕ := derrick_wild_animals + derrick_landscapes + derrick_family_events + derrick_cars + derrick_airplanes

-- Calculate the combined total of pictures
def combined_total : ℕ := ralph_total + derrick_total

-- Calculate the difference in wild animal pictures
def wild_animals_difference : ℕ := derrick_wild_animals - ralph_wild_animals

-- Theorem to prove
theorem picture_fit_count : (combined_total / wild_animals_difference : ℕ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_picture_fit_count_l3598_359889


namespace NUMINAMATH_CALUDE_ted_blue_mushrooms_l3598_359841

theorem ted_blue_mushrooms :
  let bill_red : ℕ := 12
  let bill_brown : ℕ := 6
  let ted_green : ℕ := 14
  let ted_blue : ℕ := x
  let white_spotted_total : ℕ := 17
  let white_spotted_bill_red : ℕ := bill_red * 2 / 3
  let white_spotted_bill_brown : ℕ := bill_brown
  let white_spotted_ted_blue : ℕ := ted_blue / 2
  white_spotted_total = white_spotted_bill_red + white_spotted_bill_brown + white_spotted_ted_blue →
  ted_blue = 10 := by
sorry

end NUMINAMATH_CALUDE_ted_blue_mushrooms_l3598_359841


namespace NUMINAMATH_CALUDE_absolute_value_difference_l3598_359850

theorem absolute_value_difference (a b : ℝ) : 
  (|a| = 2) → (|b| = 5) → (a < b) → ((a - b = -3) ∨ (a - b = -7)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_difference_l3598_359850


namespace NUMINAMATH_CALUDE_fourth_month_sale_l3598_359875

theorem fourth_month_sale 
  (sale1 sale2 sale3 sale5 sale6 average_sale : ℕ)
  (h1 : sale1 = 5435)
  (h2 : sale2 = 5927)
  (h3 : sale3 = 5855)
  (h5 : sale5 = 5562)
  (h6 : sale6 = 3991)
  (h_avg : average_sale = 5500)
  : ∃ sale4 : ℕ, sale4 = 6230 ∧ 
    (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = average_sale := by
  sorry

#check fourth_month_sale

end NUMINAMATH_CALUDE_fourth_month_sale_l3598_359875


namespace NUMINAMATH_CALUDE_rebecca_gave_two_caps_l3598_359869

def initial_caps : ℕ := 7
def final_caps : ℕ := 9

theorem rebecca_gave_two_caps : final_caps - initial_caps = 2 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_gave_two_caps_l3598_359869


namespace NUMINAMATH_CALUDE_sum_first_100_odd_integers_l3598_359825

theorem sum_first_100_odd_integers : 
  (Finset.range 100).sum (fun i => 2 * (i + 1) - 1) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_100_odd_integers_l3598_359825


namespace NUMINAMATH_CALUDE_partitioned_square_theorem_main_theorem_l3598_359892

/-- A square with interior points and partitioned into triangles -/
structure PartitionedSquare where
  /-- The number of interior points in the square -/
  num_interior_points : ℕ
  /-- The number of line segments drawn -/
  num_segments : ℕ
  /-- The number of triangles formed -/
  num_triangles : ℕ
  /-- Ensures that the number of interior points is 1965 -/
  h_points : num_interior_points = 1965

/-- Theorem stating the relationship between the number of interior points,
    line segments, and triangles in a partitioned square -/
theorem partitioned_square_theorem (ps : PartitionedSquare) :
  ps.num_segments = 5896 ∧ ps.num_triangles = 3932 := by
  sorry

/-- Main theorem proving the specific case for 1965 interior points -/
theorem main_theorem : 
  ∃ ps : PartitionedSquare, ps.num_segments = 5896 ∧ ps.num_triangles = 3932 := by
  sorry

end NUMINAMATH_CALUDE_partitioned_square_theorem_main_theorem_l3598_359892


namespace NUMINAMATH_CALUDE_union_of_A_and_I_minus_B_l3598_359876

def I : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 1, 2}

theorem union_of_A_and_I_minus_B : A ∪ (I \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_I_minus_B_l3598_359876


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3598_359882

theorem simplify_and_evaluate (m : ℝ) (h : m = 2 - Real.sqrt 2) :
  (3 / (m + 1) + 1 - m) / ((m + 2) / (m + 1)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3598_359882


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l3598_359813

theorem simplify_nested_expression (x : ℝ) : 1 - (2 - (1 + (2 - (3 - x)))) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l3598_359813


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l3598_359837

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l3598_359837


namespace NUMINAMATH_CALUDE_women_count_correct_l3598_359852

/-- The number of women working with men to complete a job -/
def num_women : ℕ := 15

/-- The number of men working on the job -/
def num_men : ℕ := 10

/-- The number of days it takes for the group to complete the job -/
def group_days : ℕ := 6

/-- The number of days it takes for one man to complete the job -/
def man_days : ℕ := 100

/-- The number of days it takes for one woman to complete the job -/
def woman_days : ℕ := 225

/-- Theorem stating that the number of women working with the men is correct -/
theorem women_count_correct :
  (num_men : ℚ) / man_days + (num_women : ℚ) / woman_days = 1 / group_days :=
sorry


end NUMINAMATH_CALUDE_women_count_correct_l3598_359852


namespace NUMINAMATH_CALUDE_P_equals_Q_l3598_359831

-- Define set P
def P : Set ℝ := {m : ℝ | -1 < m ∧ m ≤ 0}

-- Define set Q
def Q : Set ℝ := {m : ℝ | ∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0}

-- Theorem statement
theorem P_equals_Q : P = Q := by sorry

end NUMINAMATH_CALUDE_P_equals_Q_l3598_359831


namespace NUMINAMATH_CALUDE_count_even_not_divisible_by_3_or_11_l3598_359832

theorem count_even_not_divisible_by_3_or_11 : 
  (Finset.filter (fun n => n % 2 = 0 ∧ n % 3 ≠ 0 ∧ n % 11 ≠ 0) (Finset.range 1000)).card = 108 :=
by sorry

end NUMINAMATH_CALUDE_count_even_not_divisible_by_3_or_11_l3598_359832


namespace NUMINAMATH_CALUDE_box_surface_area_and_volume_l3598_359888

/-- Represents the dimensions of a rectangular sheet and the size of square corners to be removed --/
structure BoxParameters where
  length : ℕ
  width : ℕ
  corner_size : ℕ

/-- Calculates the surface area of the interior of the box --/
def calculate_surface_area (params : BoxParameters) : ℕ :=
  params.length * params.width - 4 * params.corner_size * params.corner_size

/-- Calculates the volume of the box --/
def calculate_volume (params : BoxParameters) : ℕ :=
  (params.length - 2 * params.corner_size) * (params.width - 2 * params.corner_size) * params.corner_size

/-- Theorem stating the surface area and volume of the box --/
theorem box_surface_area_and_volume :
  let params : BoxParameters := { length := 25, width := 35, corner_size := 6 }
  calculate_surface_area params = 731 ∧ calculate_volume params = 1794 :=
by sorry

end NUMINAMATH_CALUDE_box_surface_area_and_volume_l3598_359888


namespace NUMINAMATH_CALUDE_stickers_given_to_alex_l3598_359835

theorem stickers_given_to_alex (initial_stickers : ℕ) (stickers_to_lucy : ℕ) (remaining_stickers : ℕ)
  (h1 : initial_stickers = 99)
  (h2 : stickers_to_lucy = 42)
  (h3 : remaining_stickers = 31) :
  initial_stickers - remaining_stickers - stickers_to_lucy = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_stickers_given_to_alex_l3598_359835


namespace NUMINAMATH_CALUDE_marble_count_theorem_l3598_359824

/-- Represents the count of marbles of each color in a bag -/
structure MarbleCount where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- The ratio of marbles in the bag -/
def marbleRatio : MarbleCount := { red := 2, blue := 4, green := 6 }

/-- The number of green marbles in the bag -/
def greenMarbleCount : ℕ := 42

/-- Theorem stating the correct count of marbles given the ratio and green marble count -/
theorem marble_count_theorem (ratio : MarbleCount) (green_count : ℕ) :
  ratio = marbleRatio →
  green_count = greenMarbleCount →
  ∃ (count : MarbleCount),
    count.red = 14 ∧
    count.blue = 28 ∧
    count.green = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_count_theorem_l3598_359824
