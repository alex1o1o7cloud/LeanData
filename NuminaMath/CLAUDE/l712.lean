import Mathlib

namespace NUMINAMATH_CALUDE_smallest_group_size_l712_71237

theorem smallest_group_size (n : ℕ) : n = 154 ↔ 
  n > 0 ∧ 
  n % 6 = 1 ∧ 
  n % 8 = 2 ∧ 
  n % 9 = 4 ∧ 
  ∀ m : ℕ, m > 0 → m % 6 = 1 → m % 8 = 2 → m % 9 = 4 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_group_size_l712_71237


namespace NUMINAMATH_CALUDE_expression_simplification_l712_71244

theorem expression_simplification :
  (Real.sqrt 3) ^ 0 + 2⁻¹ + Real.sqrt (1/2) - |-1/2| = 1 + (Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l712_71244


namespace NUMINAMATH_CALUDE_smallest_divisor_after_subtraction_l712_71202

theorem smallest_divisor_after_subtraction (n : ℕ) (m : ℕ) (d : ℕ) : 
  n = 378461 →
  m = 5 →
  d = 47307 →
  (n - m) % d = 0 ∧
  ∀ k : ℕ, 5 < k → k < d → (n - m) % k ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_after_subtraction_l712_71202


namespace NUMINAMATH_CALUDE_system_solution_l712_71266

theorem system_solution :
  ∀ x y z : ℚ,
  (x * y + 1 = 2 * z ∧
   y * z + 1 = 2 * x ∧
   z * x + 1 = 2 * y) →
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨
   (x = -2 ∧ y = -2 ∧ z = 5/2) ∨
   (x = -2 ∧ y = 5/2 ∧ z = -2) ∨
   (x = 5/2 ∧ y = -2 ∧ z = -2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l712_71266


namespace NUMINAMATH_CALUDE_mary_cake_flour_l712_71294

/-- Given a recipe that requires a certain amount of flour and the amount already added,
    calculate the remaining amount of flour needed. -/
def remaining_flour (required : ℕ) (added : ℕ) : ℕ :=
  required - added

/-- The problem statement -/
theorem mary_cake_flour : remaining_flour 9 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mary_cake_flour_l712_71294


namespace NUMINAMATH_CALUDE_video_streaming_cost_theorem_l712_71264

/-- Calculates the total cost for one person's share of a video streaming subscription over a year -/
theorem video_streaming_cost_theorem 
  (monthly_cost : ℝ) 
  (num_people_sharing : ℕ) 
  (months_in_year : ℕ) 
  (h1 : monthly_cost = 14) 
  (h2 : num_people_sharing = 2) 
  (h3 : months_in_year = 12) :
  (monthly_cost / num_people_sharing) * months_in_year = 84 := by
  sorry

end NUMINAMATH_CALUDE_video_streaming_cost_theorem_l712_71264


namespace NUMINAMATH_CALUDE_problem_statement_l712_71295

theorem problem_statement : (-1)^2023 - Real.tan (π/3) + (Real.sqrt 5 - 1)^0 + |-(Real.sqrt 3)| = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l712_71295


namespace NUMINAMATH_CALUDE_season_games_count_l712_71257

/-- The number of baseball games in a month -/
def games_per_month : ℕ := 7

/-- The number of months in a season -/
def months_in_season : ℕ := 2

/-- The total number of baseball games in a season -/
def total_games : ℕ := games_per_month * months_in_season

theorem season_games_count : total_games = 14 := by
  sorry

end NUMINAMATH_CALUDE_season_games_count_l712_71257


namespace NUMINAMATH_CALUDE_football_games_per_month_l712_71289

theorem football_games_per_month 
  (total_games : ℕ) 
  (num_months : ℕ) 
  (h1 : total_games = 323) 
  (h2 : num_months = 17) 
  (h3 : total_games % num_months = 0) : 
  total_games / num_months = 19 := by
sorry


end NUMINAMATH_CALUDE_football_games_per_month_l712_71289


namespace NUMINAMATH_CALUDE_spiral_strip_length_l712_71245

/-- The length of a spiral strip on a right circular cylinder -/
theorem spiral_strip_length (base_circumference height : ℝ) 
  (h_base : base_circumference = 18)
  (h_height : height = 8) :
  Real.sqrt (height^2 + base_circumference^2) = Real.sqrt 388 := by
  sorry

end NUMINAMATH_CALUDE_spiral_strip_length_l712_71245


namespace NUMINAMATH_CALUDE_price_change_l712_71296

theorem price_change (original_price : ℝ) (h : original_price > 0) :
  original_price * (1 + 0.02) * (1 - 0.02) < original_price :=
by
  sorry

end NUMINAMATH_CALUDE_price_change_l712_71296


namespace NUMINAMATH_CALUDE_product_198_202_l712_71206

theorem product_198_202 : 198 * 202 = 39996 := by
  sorry

end NUMINAMATH_CALUDE_product_198_202_l712_71206


namespace NUMINAMATH_CALUDE_percentage_problem_l712_71204

theorem percentage_problem (n : ℝ) : 0.15 * 0.30 * 0.50 * n = 90 → n = 4000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l712_71204


namespace NUMINAMATH_CALUDE_selection_with_at_least_one_boy_l712_71270

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of boys in the group -/
def num_boys : ℕ := 8

/-- The number of girls in the group -/
def num_girls : ℕ := 6

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of people to be selected -/
def selection_size : ℕ := 3

theorem selection_with_at_least_one_boy :
  choose total_people selection_size - choose num_girls selection_size = 344 := by
  sorry

end NUMINAMATH_CALUDE_selection_with_at_least_one_boy_l712_71270


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l712_71261

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- Theorem: In a geometric sequence, if a_7 · a_19 = 8, then a_3 · a_23 = 8 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_prod : a 7 * a 19 = 8) : a 3 * a 23 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l712_71261


namespace NUMINAMATH_CALUDE_smallest_difference_l712_71260

theorem smallest_difference (a b : ℤ) (h1 : a + b < 11) (h2 : a > 6) :
  ∃ (m : ℤ), m = 4 ∧ ∀ (c d : ℤ), c + d < 11 → c > 6 → c - d ≥ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_l712_71260


namespace NUMINAMATH_CALUDE_b_work_alone_days_l712_71269

/-- The number of days A takes to finish the work alone -/
def A_days : ℝ := 5

/-- The number of days A and B work together -/
def together_days : ℝ := 2

/-- The number of days B takes to finish the remaining work after A leaves -/
def B_remaining_days : ℝ := 7

/-- The number of days B takes to finish the work alone -/
def B_days : ℝ := 15

/-- Theorem stating that given the conditions, B can finish the work alone in 15 days -/
theorem b_work_alone_days :
  (together_days * (1 / A_days + 1 / B_days) + B_remaining_days * (1 / B_days) = 1) :=
sorry

end NUMINAMATH_CALUDE_b_work_alone_days_l712_71269


namespace NUMINAMATH_CALUDE_password_probability_l712_71215

-- Define the set of possible last digits
def LastDigits : Finset Char := {'A', 'a', 'B', 'b'}

-- Define the set of possible second-to-last digits
def SecondLastDigits : Finset Nat := {4, 5, 6}

-- Define the type for a password
def Password := Nat × Char

-- Define the set of all possible passwords
def AllPasswords : Finset Password :=
  SecondLastDigits.product LastDigits

-- Theorem statement
theorem password_probability :
  (Finset.card AllPasswords : ℚ) = 12 ∧
  (1 : ℚ) / (Finset.card AllPasswords : ℚ) = 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_password_probability_l712_71215


namespace NUMINAMATH_CALUDE_project_profit_analysis_l712_71240

/-- Represents the net profit of a project in millions of yuan -/
def net_profit (n : ℕ+) : ℚ :=
  100 * n - (4 * n^2 + 40 * n) - 144

/-- Represents the average annual profit of a project in millions of yuan -/
def avg_annual_profit (n : ℕ+) : ℚ :=
  net_profit n / n

theorem project_profit_analysis :
  ∀ n : ℕ+,
  (net_profit n = -4 * (n - 3) * (n - 12)) ∧
  (net_profit n > 0 ↔ 3 < n ∧ n < 12) ∧
  (∀ m : ℕ+, avg_annual_profit m ≤ avg_annual_profit 6) := by
  sorry

#check project_profit_analysis

end NUMINAMATH_CALUDE_project_profit_analysis_l712_71240


namespace NUMINAMATH_CALUDE_ratio_c_over_a_l712_71292

theorem ratio_c_over_a (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_arithmetic_seq : 2 * Real.log (a * c) = Real.log (a * b) + Real.log (b * c))
  (h_relation : 4 * (a + c) = 17 * b) :
  c / a = 16 ∨ c / a = 1 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_c_over_a_l712_71292


namespace NUMINAMATH_CALUDE_equation_solution_l712_71288

theorem equation_solution :
  ∀ x : ℝ, x + 36 / (x - 4) = -9 ↔ x = 0 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l712_71288


namespace NUMINAMATH_CALUDE_total_stars_shelby_and_alex_l712_71278

/-- Gold stars earned by a student for each day of the week -/
structure WeeklyStars :=
  (monday : ℕ)
  (tuesday : ℕ)
  (wednesday : ℕ)
  (thursday : ℕ)
  (friday : ℕ)
  (saturday : ℕ)
  (sunday : ℕ)

/-- Calculate the total number of stars for a week -/
def totalStars (w : WeeklyStars) : ℕ :=
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday + w.sunday

/-- Shelby's stars for the week -/
def shelbysStars : WeeklyStars :=
  { monday := 4
  , tuesday := 6
  , wednesday := 3
  , thursday := 5
  , friday := 2
  , saturday := 3
  , sunday := 7 }

/-- Alex's stars for the week -/
def alexsStars : WeeklyStars :=
  { monday := 5
  , tuesday := 3
  , wednesday := 6
  , thursday := 4
  , friday := 7
  , saturday := 2
  , sunday := 5 }

/-- Theorem: The total number of stars earned by Shelby and Alex together is 62 -/
theorem total_stars_shelby_and_alex :
  totalStars shelbysStars + totalStars alexsStars = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_stars_shelby_and_alex_l712_71278


namespace NUMINAMATH_CALUDE_largest_angle_cosine_in_triangle_l712_71267

theorem largest_angle_cosine_in_triangle (a b c : ℝ) (h1 : a = 3) (h2 : b = 5) (h3 : c = 7) :
  let cos_largest_angle := min (min ((a^2 + b^2 - c^2) / (2*a*b)) ((b^2 + c^2 - a^2) / (2*b*c))) ((c^2 + a^2 - b^2) / (2*c*a))
  cos_largest_angle = -1/2 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_cosine_in_triangle_l712_71267


namespace NUMINAMATH_CALUDE_diminished_value_proof_diminished_value_l712_71284

theorem diminished_value_proof (n : Nat) (divisors : List Nat) : Prop :=
  let smallest := 1013
  let value := 5
  let lcm := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 12 16) 18) 21) 28
  (∀ d ∈ divisors, (smallest - value) % d = 0) ∧
  (smallest = lcm + value) ∧
  (∀ m < smallest, ∃ d ∈ divisors, (m - value) % d ≠ 0)

/-- The value that needs to be diminished from 1013 to make it divisible by 12, 16, 18, 21, and 28 is 5. -/
theorem diminished_value :
  diminished_value_proof 1013 [12, 16, 18, 21, 28] :=
by sorry

end NUMINAMATH_CALUDE_diminished_value_proof_diminished_value_l712_71284


namespace NUMINAMATH_CALUDE_bridge_crossing_time_l712_71291

/-- Proves that a man walking at 5 km/hr takes 15 minutes to cross a 1250-meter bridge -/
theorem bridge_crossing_time :
  let walking_speed : ℝ := 5  -- km/hr
  let bridge_length : ℝ := 1250  -- meters
  let crossing_time : ℝ := 15  -- minutes
  
  walking_speed * 1000 / 60 * crossing_time = bridge_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_crossing_time_l712_71291


namespace NUMINAMATH_CALUDE_inequality_solution_range_l712_71283

theorem inequality_solution_range (m : ℝ) : 
  (∃ (a b : ℤ), ∀ (x : ℤ), (x : ℝ)^2 + (m + 1) * (x : ℝ) + m < 0 ↔ x = a ∨ x = b) →
  (-2 ≤ m ∧ m < -1) ∨ (3 < m ∧ m ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l712_71283


namespace NUMINAMATH_CALUDE_no_perfect_square_in_range_l712_71231

theorem no_perfect_square_in_range : 
  ¬ ∃ (n : ℤ), 5 ≤ n ∧ n ≤ 15 ∧ ∃ (m : ℤ), 2 * n^2 + n + 2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_in_range_l712_71231


namespace NUMINAMATH_CALUDE_absolute_difference_simplification_l712_71273

theorem absolute_difference_simplification (a b : ℝ) 
  (ha : a < 0) (hab : a * b < 0) : 
  |a - b - 3| - |4 + b - a| = -1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_simplification_l712_71273


namespace NUMINAMATH_CALUDE_frank_candy_total_l712_71249

/-- Given that Frank puts 11 pieces of candy in each bag and makes 2 bags,
    prove that the total number of candy pieces is 22. -/
theorem frank_candy_total (pieces_per_bag : ℕ) (num_bags : ℕ) 
    (h1 : pieces_per_bag = 11) (h2 : num_bags = 2) : 
    pieces_per_bag * num_bags = 22 := by
  sorry

end NUMINAMATH_CALUDE_frank_candy_total_l712_71249


namespace NUMINAMATH_CALUDE_price_increase_percentage_l712_71259

theorem price_increase_percentage (price_B : ℝ) (price_A : ℝ) : 
  price_A = price_B * 0.8 → 
  (price_B - price_A) / price_A * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l712_71259


namespace NUMINAMATH_CALUDE_min_value_cyclic_fraction_l712_71252

theorem min_value_cyclic_fraction (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  a / b + b / c + c / d + d / a ≥ 4 ∧ 
  (a / b + b / c + c / d + d / a = 4 ↔ a = b ∧ b = c ∧ c = d) := by
  sorry

end NUMINAMATH_CALUDE_min_value_cyclic_fraction_l712_71252


namespace NUMINAMATH_CALUDE_subtraction_division_problem_l712_71207

theorem subtraction_division_problem : 3550 - (1002 / 20.04) = 3499.9501 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_division_problem_l712_71207


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l712_71220

theorem cubic_roots_sum (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - 2*x - 2 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  a*(b-c)^2 + b*(c-a)^2 + c*(a-b)^2 = -2 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l712_71220


namespace NUMINAMATH_CALUDE_committee_selection_count_l712_71246

theorem committee_selection_count : Nat.choose 30 5 = 142506 := by sorry

end NUMINAMATH_CALUDE_committee_selection_count_l712_71246


namespace NUMINAMATH_CALUDE_intersection_P_complement_Q_l712_71214

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 < 1}
def Q : Set ℝ := {x | x ≥ 0}

-- State the theorem
theorem intersection_P_complement_Q : 
  P ∩ (Set.univ \ Q) = {x : ℝ | -1 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_P_complement_Q_l712_71214


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l712_71290

/-- A line passing through points (4, -5) and (k, 23) is parallel to the line 3x - 4y = 12 -/
theorem parallel_line_k_value (k : ℚ) : 
  (∃ (m b : ℚ), (m * 4 + b = -5) ∧ (m * k + b = 23) ∧ (m = 3/4)) → k = 124/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l712_71290


namespace NUMINAMATH_CALUDE_proposition_false_range_l712_71205

open Set

theorem proposition_false_range (a : ℝ) : 
  (¬∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ a ∈ Iio (-3) ∪ Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_proposition_false_range_l712_71205


namespace NUMINAMATH_CALUDE_quadratic_necessary_not_sufficient_l712_71200

theorem quadratic_necessary_not_sufficient :
  (∀ x : ℝ, x > 2 → x^2 + 5*x - 6 > 0) ∧
  (∃ x : ℝ, x^2 + 5*x - 6 > 0 ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_necessary_not_sufficient_l712_71200


namespace NUMINAMATH_CALUDE_daves_coins_l712_71218

theorem daves_coins (n : ℕ) : n > 0 ∧ 
  n % 7 = 2 ∧ 
  n % 5 = 3 ∧ 
  n % 3 = 1 ∧ 
  (∀ m : ℕ, m > 0 → m % 7 = 2 → m % 5 = 3 → m % 3 = 1 → n ≤ m) → 
  n = 58 := by
sorry

end NUMINAMATH_CALUDE_daves_coins_l712_71218


namespace NUMINAMATH_CALUDE_triangle_angles_sum_l712_71297

theorem triangle_angles_sum (x y : ℕ) : 
  x > 0 ∧ y > 0 ∧ 8 * x + 13 * y = 130 → x + y = 1289 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_sum_l712_71297


namespace NUMINAMATH_CALUDE_machine_working_time_l712_71225

/-- The number of shirts made by the machine -/
def total_shirts : ℕ := 196

/-- The number of shirts the machine can make per minute -/
def shirts_per_minute : ℕ := 7

/-- The time worked by the machine in minutes -/
def time_worked : ℕ := total_shirts / shirts_per_minute

theorem machine_working_time : time_worked = 28 := by
  sorry

end NUMINAMATH_CALUDE_machine_working_time_l712_71225


namespace NUMINAMATH_CALUDE_susan_cats_proof_l712_71234

/-- The number of cats Bob has -/
def bob_cats : ℕ := 3

/-- The number of cats Susan gives away -/
def cats_given_away : ℕ := 4

/-- The difference in cats between Susan and Bob after Susan gives some away -/
def cat_difference : ℕ := 14

/-- Susan's initial number of cats -/
def susan_initial_cats : ℕ := 25

theorem susan_cats_proof :
  susan_initial_cats = bob_cats + cats_given_away + cat_difference := by
  sorry

end NUMINAMATH_CALUDE_susan_cats_proof_l712_71234


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l712_71201

theorem sqrt_product_plus_one : 
  Real.sqrt ((41:ℝ) * 40 * 39 * 38 + 1) = 1559 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l712_71201


namespace NUMINAMATH_CALUDE_franks_books_l712_71241

theorem franks_books (a b c : ℤ) (n : ℕ) (p d t : ℕ) :
  p = 2 * a →
  d = 3 * b →
  t = 2 * c * (3 * b) →
  n * p = t →
  n * d = t →
  ∃ (k : ℤ), n = 2 * k ∧ k = c := by
  sorry

end NUMINAMATH_CALUDE_franks_books_l712_71241


namespace NUMINAMATH_CALUDE_problem_solution_l712_71212

def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

theorem problem_solution :
  (∃ a : ℝ, 
    (A ∩ B a = {x : ℝ | 1/2 ≤ x ∧ x < 2} ∧
     A ∪ B a = {x : ℝ | -2 < x ∧ x ≤ 3})) ∧
  (∀ a : ℝ, (Aᶜ ∩ B a = B a) ↔ a ≥ -1/4) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l712_71212


namespace NUMINAMATH_CALUDE_inequality_equivalence_l712_71208

def f (x : ℝ) : ℝ := 5 * x + 3

theorem inequality_equivalence (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, |x + 2| < b → |f x + 7| < a) ↔ b ≤ a / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l712_71208


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l712_71226

theorem tangent_sum_simplification :
  (Real.tan (10 * π / 180) + Real.tan (30 * π / 180) + 
   Real.tan (50 * π / 180) + Real.tan (70 * π / 180)) / 
  Real.cos (30 * π / 180) = 
  4 * Real.sin (40 * π / 180) + 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l712_71226


namespace NUMINAMATH_CALUDE_ratio_equivalence_l712_71251

theorem ratio_equivalence : ∃ (x y : ℚ) (z : ℕ),
  (4 : ℚ) / 5 = 20 / x ∧
  (4 : ℚ) / 5 = y / 20 ∧
  (4 : ℚ) / 5 = (z : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equivalence_l712_71251


namespace NUMINAMATH_CALUDE_min_cubes_in_block_l712_71293

theorem min_cubes_in_block (l m n : ℕ) : 
  (l - 1) * (m - 1) * (n - 1) = 252 → 
  l * m * n ≥ 392 ∧ 
  (∃ (l' m' n' : ℕ), (l' - 1) * (m' - 1) * (n' - 1) = 252 ∧ l' * m' * n' = 392) :=
by sorry

end NUMINAMATH_CALUDE_min_cubes_in_block_l712_71293


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l712_71235

theorem simplify_complex_fraction (y : ℝ) 
  (h1 : y ≠ 4) (h2 : y ≠ 2) (h3 : y ≠ 5) (h4 : y ≠ 7) (h5 : y ≠ 1) :
  (y^2 - 4*y + 3) / (y^2 - 6*y + 8) / ((y^2 - 9*y + 20) / (y^2 - 9*y + 14)) = 
  ((y - 3) * (y - 7)) / ((y - 1) * (y - 5)) :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l712_71235


namespace NUMINAMATH_CALUDE_integral_equals_ten_l712_71268

theorem integral_equals_ten (k : ℝ) : 
  (∫ x in (0 : ℝ)..2, 3 * x^2 + k) = 10 → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_ten_l712_71268


namespace NUMINAMATH_CALUDE_muffin_price_is_four_l712_71228

/-- Represents the number of muffins made by each person and the total contribution --/
structure MuffinSale where
  sasha : ℕ
  melissa : ℕ
  tiffany : ℕ
  contribution : ℕ

/-- Calculates the price per muffin given the sale information --/
def price_per_muffin (sale : MuffinSale) : ℚ :=
  sale.contribution / (sale.sasha + sale.melissa + sale.tiffany)

/-- Theorem stating that the price per muffin is $4 given the conditions --/
theorem muffin_price_is_four :
  ∀ (sale : MuffinSale),
    sale.sasha = 30 →
    sale.melissa = 4 * sale.sasha →
    sale.tiffany = (sale.sasha + sale.melissa) / 2 →
    sale.contribution = 900 →
    price_per_muffin sale = 4 := by
  sorry

end NUMINAMATH_CALUDE_muffin_price_is_four_l712_71228


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l712_71236

theorem quadratic_equal_roots (c : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + c = 0 ∧ (∀ y : ℝ, y^2 - 4*y + c = 0 → y = x)) → c = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l712_71236


namespace NUMINAMATH_CALUDE_midpoint_one_sixth_to_five_sixths_l712_71229

theorem midpoint_one_sixth_to_five_sixths :
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 6
  (a + b) / 2 = (1 : ℚ) / 2 := by
sorry

end NUMINAMATH_CALUDE_midpoint_one_sixth_to_five_sixths_l712_71229


namespace NUMINAMATH_CALUDE_subtraction_result_l712_71282

theorem subtraction_result : 2014 - 4102 = -2088 := by sorry

end NUMINAMATH_CALUDE_subtraction_result_l712_71282


namespace NUMINAMATH_CALUDE_dereks_lowest_score_l712_71232

theorem dereks_lowest_score (test1 test2 test3 test4 : ℕ) : 
  test1 = 85 →
  test2 = 78 →
  test1 ≤ 100 →
  test2 ≤ 100 →
  test3 ≤ 100 →
  test4 ≤ 100 →
  test3 ≥ 60 →
  test4 ≥ 60 →
  (test1 + test2 + test3 + test4) / 4 = 84 →
  (min test3 test4 = 73 ∨ min test3 test4 > 73) :=
by sorry

end NUMINAMATH_CALUDE_dereks_lowest_score_l712_71232


namespace NUMINAMATH_CALUDE_initial_solution_volume_l712_71258

/-- Proves that the initial amount of a 26% alcohol solution is 15 liters
    when 5 liters of water are added to create a 19.5% alcohol mixture -/
theorem initial_solution_volume : 
  ∀ (x : ℝ),
  (0.26 * x = 0.195 * (x + 5)) →
  x = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_solution_volume_l712_71258


namespace NUMINAMATH_CALUDE_darnels_scooping_rate_l712_71254

/-- Proves Darrel's scooping rate given the problem conditions -/
theorem darnels_scooping_rate 
  (steven_rate : ℝ) 
  (total_time : ℝ) 
  (total_load : ℝ) 
  (h1 : steven_rate = 75)
  (h2 : total_time = 30)
  (h3 : total_load = 2550) :
  ∃ (darrel_rate : ℝ), 
    (steven_rate + darrel_rate) * total_time = total_load ∧ 
    darrel_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_darnels_scooping_rate_l712_71254


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l712_71217

theorem inequality_system_solution_range (a : ℝ) : 
  (∀ x : ℝ, (2 * x - 1 < 3 ∧ x - a < 0) ↔ x < a) → 
  a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l712_71217


namespace NUMINAMATH_CALUDE_solution_product_theorem_l712_71216

theorem solution_product_theorem (a b : ℝ) : 
  a ≠ b → 
  (a^4 + a^3 - 1 = 0) → 
  (b^4 + b^3 - 1 = 0) → 
  ((a*b)^6 + (a*b)^4 + (a*b)^3 - (a*b)^2 - 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_solution_product_theorem_l712_71216


namespace NUMINAMATH_CALUDE_ratio_equality_l712_71272

theorem ratio_equality (a b : ℝ) (h : 7 * a = 8 * b) : (a / 8) / (b / 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l712_71272


namespace NUMINAMATH_CALUDE_solve_for_n_l712_71230

theorem solve_for_n (s m k r P : ℝ) (h : P = (s + m) / ((1 + k)^n + r)) :
  n = Real.log ((s + m - P * r) / P) / Real.log (1 + k) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_n_l712_71230


namespace NUMINAMATH_CALUDE_coin_flip_probability_l712_71262

def num_flips : ℕ := 12

def favorable_outcomes : ℕ := (
  Nat.choose num_flips 7 + 
  Nat.choose num_flips 8 + 
  Nat.choose num_flips 9 + 
  Nat.choose num_flips 10 + 
  Nat.choose num_flips 11 + 
  Nat.choose num_flips 12
)

def total_outcomes : ℕ := 2^num_flips

theorem coin_flip_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 793 / 2048 :=
by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l712_71262


namespace NUMINAMATH_CALUDE_jumping_competition_result_l712_71287

/-- The difference in average jumps per minute between two competitors -/
def jump_difference (total_time : ℕ) (jumps_a : ℕ) (jumps_b : ℕ) : ℚ :=
  (jumps_a - jumps_b : ℚ) / total_time

theorem jumping_competition_result :
  jump_difference 5 480 420 = 12 := by
  sorry

end NUMINAMATH_CALUDE_jumping_competition_result_l712_71287


namespace NUMINAMATH_CALUDE_two_digit_cube_l712_71276

theorem two_digit_cube (x : ℕ) : x = 93 ↔ 
  (x ≥ 10 ∧ x < 100) ∧ 
  (∃ n : ℕ, 101010 * x + 1 = n^3) ∧
  (101010 * x + 1 ≥ 1000000 ∧ 101010 * x + 1 < 10000000) := by
sorry

end NUMINAMATH_CALUDE_two_digit_cube_l712_71276


namespace NUMINAMATH_CALUDE_percentage_unsold_bags_l712_71280

/-- Given the initial stock and daily sales of bags in a bookshop,
    prove that the percentage of unsold bags is 25%. -/
theorem percentage_unsold_bags
  (initial_stock : ℕ)
  (monday_sales tuesday_sales wednesday_sales thursday_sales friday_sales : ℕ)
  (h_initial : initial_stock = 600)
  (h_monday : monday_sales = 25)
  (h_tuesday : tuesday_sales = 70)
  (h_wednesday : wednesday_sales = 100)
  (h_thursday : thursday_sales = 110)
  (h_friday : friday_sales = 145) :
  (initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales)) / initial_stock * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_unsold_bags_l712_71280


namespace NUMINAMATH_CALUDE_greatest_three_digit_non_divisor_l712_71256

theorem greatest_three_digit_non_divisor : ∃ n : ℕ, 
  n = 998 ∧ 
  n ≥ 100 ∧ n < 1000 ∧ 
  ∀ m : ℕ, m > n → m < 1000 → 
    (m * (m + 1) / 2 ∣ Nat.factorial (m - 1)) ∧
  ¬(n * (n + 1) / 2 ∣ Nat.factorial (n - 1)) := by
  sorry

#check greatest_three_digit_non_divisor

end NUMINAMATH_CALUDE_greatest_three_digit_non_divisor_l712_71256


namespace NUMINAMATH_CALUDE_adams_books_before_shopping_l712_71279

/-- Calculates the number of books Adam had before his shopping trip -/
theorem adams_books_before_shopping
  (shelves : ℕ)
  (books_per_shelf : ℕ)
  (new_books : ℕ)
  (leftover_books : ℕ)
  (h1 : shelves = 4)
  (h2 : books_per_shelf = 20)
  (h3 : new_books = 26)
  (h4 : leftover_books = 2) :
  shelves * books_per_shelf - (new_books - leftover_books) = 56 :=
by sorry

end NUMINAMATH_CALUDE_adams_books_before_shopping_l712_71279


namespace NUMINAMATH_CALUDE_bottles_taken_back_l712_71285

/-- The number of bottles Debby takes back home is equal to the number of bottles she brought minus the number of bottles drunk. -/
theorem bottles_taken_back (bottles_brought bottles_drunk : ℕ) :
  bottles_brought ≥ bottles_drunk →
  bottles_brought - bottles_drunk = bottles_brought - bottles_drunk :=
by sorry

end NUMINAMATH_CALUDE_bottles_taken_back_l712_71285


namespace NUMINAMATH_CALUDE_op_properties_l712_71281

-- Define the @ operation
def op (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

-- Theorem statement
theorem op_properties :
  (op 1 (-2) = -8) ∧ 
  (∀ a b : ℝ, op a b = op b a) ∧
  (∀ a b : ℝ, a + b = 0 → op a a + op b b = 8 * a^2) := by
sorry

end NUMINAMATH_CALUDE_op_properties_l712_71281


namespace NUMINAMATH_CALUDE_apple_orange_ratio_l712_71223

theorem apple_orange_ratio (num_oranges : ℕ) : 
  (15 : ℚ) + num_oranges = 50 * (3/2) → 
  (15 : ℚ) / num_oranges = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_apple_orange_ratio_l712_71223


namespace NUMINAMATH_CALUDE_calculation_result_l712_71286

theorem calculation_result : (0.0048 * 3.5) / (0.05 * 0.1 * 0.004) = 840 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l712_71286


namespace NUMINAMATH_CALUDE_sausages_theorem_l712_71238

def sausages_left (initial : ℕ) : ℕ :=
  let after_monday := initial - (2 * initial / 5)
  let after_tuesday := after_monday - (after_monday / 2)
  let after_wednesday := after_tuesday - (after_tuesday / 4)
  let after_thursday := after_wednesday - (after_wednesday / 3)
  let after_sharing := after_thursday - (after_thursday / 5)
  after_sharing - ((3 * after_sharing) / 5)

theorem sausages_theorem :
  sausages_left 1200 = 58 := by
  sorry

end NUMINAMATH_CALUDE_sausages_theorem_l712_71238


namespace NUMINAMATH_CALUDE_iron_cubes_melting_l712_71253

theorem iron_cubes_melting (s1 s2 s3 s_large : ℝ) : 
  s1 = 1 ∧ s2 = 6 ∧ s3 = 8 → 
  s_large^3 = s1^3 + s2^3 + s3^3 →
  s_large = 9 := by
sorry

end NUMINAMATH_CALUDE_iron_cubes_melting_l712_71253


namespace NUMINAMATH_CALUDE_root_product_of_equation_l712_71271

theorem root_product_of_equation : ∃ (x y : ℝ), 
  (Real.sqrt (2 * x^2 + 8 * x + 1) - x = 3) ∧
  (Real.sqrt (2 * y^2 + 8 * y + 1) - y = 3) ∧
  (x ≠ y) ∧ (x * y = -8) := by
  sorry

end NUMINAMATH_CALUDE_root_product_of_equation_l712_71271


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l712_71224

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem zero_point_in_interval :
  ∃ (c : ℝ), 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_point_in_interval_l712_71224


namespace NUMINAMATH_CALUDE_initial_average_height_l712_71211

/-- Given a class of boys with an incorrect height measurement, prove the initially calculated average height. -/
theorem initial_average_height
  (n : ℕ) -- number of boys
  (height_difference : ℝ) -- difference between incorrect and correct height
  (actual_average : ℝ) -- actual average height after correction
  (h_n : n = 35) -- there are 35 boys
  (h_diff : height_difference = 60) -- the height difference is 60 cm
  (h_actual : actual_average = 183) -- the actual average height is 183 cm
  : ∃ (initial_average : ℝ), initial_average = 181 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_height_l712_71211


namespace NUMINAMATH_CALUDE_peanuts_added_l712_71275

theorem peanuts_added (initial_peanuts final_peanuts : ℕ) 
  (h1 : initial_peanuts = 4)
  (h2 : final_peanuts = 12) : 
  final_peanuts - initial_peanuts = 8 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_added_l712_71275


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l712_71222

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l712_71222


namespace NUMINAMATH_CALUDE_apples_left_ella_apples_left_l712_71210

theorem apples_left (bags_20 : Nat) (apples_per_bag_20 : Nat) 
                    (bags_25 : Nat) (apples_per_bag_25 : Nat) 
                    (sold : Nat) : Nat :=
  let total_20 := bags_20 * apples_per_bag_20
  let total_25 := bags_25 * apples_per_bag_25
  let total := total_20 + total_25
  total - sold

theorem ella_apples_left : apples_left 4 20 6 25 200 = 30 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_ella_apples_left_l712_71210


namespace NUMINAMATH_CALUDE_compaction_percentage_is_twenty_l712_71203

/-- Represents the compaction problem with cans -/
structure CanCompaction where
  num_cans : ℕ
  space_before : ℕ
  total_space_after : ℕ

/-- Calculates the percentage of original space each can takes up after compaction -/
def compaction_percentage (c : CanCompaction) : ℚ :=
  (c.total_space_after : ℚ) / ((c.num_cans * c.space_before) : ℚ) * 100

/-- Theorem stating that for the given conditions, the compaction percentage is 20% -/
theorem compaction_percentage_is_twenty (c : CanCompaction) 
  (h1 : c.num_cans = 60)
  (h2 : c.space_before = 30)
  (h3 : c.total_space_after = 360) : 
  compaction_percentage c = 20 := by
  sorry

end NUMINAMATH_CALUDE_compaction_percentage_is_twenty_l712_71203


namespace NUMINAMATH_CALUDE_library_repacking_l712_71274

theorem library_repacking (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) :
  initial_boxes = 1584 →
  books_per_initial_box = 45 →
  books_per_new_box = 47 →
  (initial_boxes * books_per_initial_box) % books_per_new_box = 28 := by
  sorry

end NUMINAMATH_CALUDE_library_repacking_l712_71274


namespace NUMINAMATH_CALUDE_value_of_X_l712_71243

theorem value_of_X : ∃ X : ℚ, (1/4 : ℚ) * (1/8 : ℚ) * X = (1/2 : ℚ) * (1/6 : ℚ) * 120 ∧ X = 320 := by
  sorry

end NUMINAMATH_CALUDE_value_of_X_l712_71243


namespace NUMINAMATH_CALUDE_handshake_count_l712_71233

theorem handshake_count (num_gremlins num_imps : ℕ) (h1 : num_gremlins = 30) (h2 : num_imps = 20) :
  let gremlin_handshakes := num_gremlins.choose 2
  let gremlin_imp_handshakes := num_gremlins * num_imps
  gremlin_handshakes + gremlin_imp_handshakes = 1035 := by
  sorry

#check handshake_count

end NUMINAMATH_CALUDE_handshake_count_l712_71233


namespace NUMINAMATH_CALUDE_min_value_cubic_quadratic_l712_71209

theorem min_value_cubic_quadratic (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ineq : 57 * a + 88 * b + 125 * c ≥ 1148) : 
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 57 * x + 88 * y + 125 * z ≥ 1148 →
  a^3 + b^3 + c^3 + 5*a^2 + 5*b^2 + 5*c^2 ≤ x^3 + y^3 + z^3 + 5*x^2 + 5*y^2 + 5*z^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_cubic_quadratic_l712_71209


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_third_l712_71265

theorem reciprocal_of_negative_one_third :
  let x : ℚ := -1/3
  let y : ℚ := -3
  (x * y = 1) → (∀ z : ℚ, x * z = 1 → z = y) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_third_l712_71265


namespace NUMINAMATH_CALUDE_simplify_expression_l712_71277

theorem simplify_expression (w : ℝ) : w + 2 - 3*w - 4 + 5*w + 6 - 7*w - 8 = -4*w - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l712_71277


namespace NUMINAMATH_CALUDE_square_sheet_area_l712_71248

theorem square_sheet_area (x : ℝ) : 
  x > 0 → x * (x - 3) = 40 → x^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_sheet_area_l712_71248


namespace NUMINAMATH_CALUDE_decreasing_interval_of_f_decreasing_interval_is_open_interval_l712_71298

-- Define the function
def f (x : ℝ) := x^3 - 3*x

-- Define the derivative of the function
def f' (x : ℝ) := 3*x^2 - 3

-- Theorem statement
theorem decreasing_interval_of_f :
  ∀ x : ℝ, (f' x < 0) ↔ (-1 < x ∧ x < 1) :=
sorry

-- Main theorem
theorem decreasing_interval_is_open_interval :
  {x : ℝ | ∀ y : ℝ, -1 < y ∧ y < x → f y > f x} = Set.Ioo (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_f_decreasing_interval_is_open_interval_l712_71298


namespace NUMINAMATH_CALUDE_star_two_three_star_two_neg_six_neg_two_thirds_l712_71219

-- Define the operation *
def star (a b : ℚ) : ℚ := (a + b) / 3

-- Theorem for 2 * 3 = 5/3
theorem star_two_three : star 2 3 = 5/3 := by sorry

-- Theorem for 2 * (-6) * (-2/3) = -2/3
theorem star_two_neg_six_neg_two_thirds : star (star 2 (-6)) (-2/3) = -2/3 := by sorry

end NUMINAMATH_CALUDE_star_two_three_star_two_neg_six_neg_two_thirds_l712_71219


namespace NUMINAMATH_CALUDE_complement_of_A_l712_71250

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x + 1 > 0}

theorem complement_of_A (x : ℝ) : x ∈ (Set.univ \ A) ↔ x ≤ -1 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l712_71250


namespace NUMINAMATH_CALUDE_train_station_distance_l712_71255

/-- The distance to the train station -/
def distance : ℝ := 4

/-- The speed of the man in the first scenario (km/h) -/
def speed1 : ℝ := 4

/-- The speed of the man in the second scenario (km/h) -/
def speed2 : ℝ := 5

/-- The time difference between the man's arrival and the train's arrival in the first scenario (minutes) -/
def time_diff1 : ℝ := 6

/-- The time difference between the man's arrival and the train's arrival in the second scenario (minutes) -/
def time_diff2 : ℝ := -6

theorem train_station_distance :
  (distance / speed1 - distance / speed2) * 60 = time_diff1 - time_diff2 := by sorry

end NUMINAMATH_CALUDE_train_station_distance_l712_71255


namespace NUMINAMATH_CALUDE_brown_mm_averages_l712_71242

def brown_smiley_counts : List Nat := [9, 12, 8, 8, 3]
def brown_star_counts : List Nat := [7, 14, 11, 6, 10]

theorem brown_mm_averages :
  let smiley_avg := (brown_smiley_counts.sum : ℚ) / brown_smiley_counts.length
  let star_avg := (brown_star_counts.sum : ℚ) / brown_star_counts.length
  smiley_avg = 8 ∧ star_avg = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_brown_mm_averages_l712_71242


namespace NUMINAMATH_CALUDE_minimum_score_needed_l712_71299

def current_scores : List ℕ := [90, 80, 70, 60, 85]
def score_count : ℕ := current_scores.length
def current_average : ℚ := (current_scores.sum : ℚ) / score_count
def target_increase : ℚ := 3
def new_score_count : ℕ := score_count + 1

theorem minimum_score_needed (x : ℕ) : 
  (((current_scores.sum + x) : ℚ) / new_score_count ≥ current_average + target_increase) ↔ 
  (x ≥ 95) :=
sorry

end NUMINAMATH_CALUDE_minimum_score_needed_l712_71299


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l712_71221

theorem geometric_series_ratio (a r : ℝ) (h1 : r ≠ 1) : 
  (∃ (S : ℝ), S = a / (1 - r) ∧ S = 18) →
  (∃ (S_odd : ℝ), S_odd = a * r / (1 - r^2) ∧ S_odd = 6) →
  r = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l712_71221


namespace NUMINAMATH_CALUDE_factorization_equality_l712_71239

theorem factorization_equality (a : ℝ) : 3 * a^2 - 6 * a + 3 = 3 * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l712_71239


namespace NUMINAMATH_CALUDE_at_least_two_primes_of_form_l712_71213

theorem at_least_two_primes_of_form (n : ℕ) : ∃ (a b : ℕ), 2 ≤ a ∧ 2 ≤ b ∧ a ≠ b ∧ 
  Nat.Prime (a^3 + a + 1) ∧ Nat.Prime (b^3 + b + 1) :=
sorry

end NUMINAMATH_CALUDE_at_least_two_primes_of_form_l712_71213


namespace NUMINAMATH_CALUDE_point_in_region_l712_71227

-- Define the plane region
def in_region (x y : ℝ) : Prop := 2 * x + y - 6 < 0

-- Theorem to prove
theorem point_in_region : in_region 0 1 := by
  sorry

end NUMINAMATH_CALUDE_point_in_region_l712_71227


namespace NUMINAMATH_CALUDE_total_blue_balloons_l712_71263

/-- The number of blue balloons Joan and Melanie have in total -/
def total_balloons (joan_balloons melanie_balloons : ℕ) : ℕ :=
  joan_balloons + melanie_balloons

/-- Theorem stating that Joan and Melanie have 81 blue balloons in total -/
theorem total_blue_balloons :
  total_balloons 40 41 = 81 := by
  sorry

end NUMINAMATH_CALUDE_total_blue_balloons_l712_71263


namespace NUMINAMATH_CALUDE_recurrence_sequence_a8_l712_71247

/-- A strictly increasing sequence of positive integers satisfying the given recurrence relation. -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, n ≥ 1 → a (n + 2) = a n + a (n + 1))

theorem recurrence_sequence_a8 (a : ℕ → ℕ) (h : RecurrenceSequence a) (h7 : a 7 = 120) : 
  a 8 = 194 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_a8_l712_71247
