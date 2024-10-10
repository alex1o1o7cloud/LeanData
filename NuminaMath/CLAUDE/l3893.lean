import Mathlib

namespace least_three_digit_with_digit_product_12_l3893_389323

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_digit_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → 134 ≤ n :=
sorry

end least_three_digit_with_digit_product_12_l3893_389323


namespace first_place_points_l3893_389333

def second_place_points : Nat := 7
def third_place_points : Nat := 5
def fourth_place_points : Nat := 2
def total_participations : Nat := 7
def product_of_points : Nat := 38500

theorem first_place_points (first_place : Nat) 
  (h1 : ∃ (a b c d : Nat), a + b + c + d = total_participations ∧ 
                           a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
                           first_place^a * second_place_points^b * 
                           third_place_points^c * fourth_place_points^d = product_of_points) : 
  first_place = 11 := by
sorry

end first_place_points_l3893_389333


namespace square_diff_over_hundred_l3893_389328

theorem square_diff_over_hundred : (2200 - 2100)^2 / 100 = 100 := by
  sorry

end square_diff_over_hundred_l3893_389328


namespace rotation_of_point_A_l3893_389338

-- Define the rotation function
def rotate_clockwise_90 (x y : ℝ) : ℝ × ℝ := (y, -x)

-- Define the theorem
theorem rotation_of_point_A : 
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := rotate_clockwise_90 A.1 A.2
  B = (1, -2) := by sorry

end rotation_of_point_A_l3893_389338


namespace brad_read_more_books_l3893_389340

/-- Proves that Brad read 4 more books than William across two months --/
theorem brad_read_more_books (william_last_month : ℕ) (brad_this_month : ℕ) : 
  william_last_month = 6 →
  brad_this_month = 8 →
  (3 * william_last_month + brad_this_month) - (william_last_month + 2 * brad_this_month) = 4 := by
sorry

end brad_read_more_books_l3893_389340


namespace smallest_n_square_and_cube_l3893_389334

theorem smallest_n_square_and_cube : 
  (∀ m : ℕ, m > 0 ∧ m < 1875 → ¬(∃ a : ℕ, 3 * m = a ^ 2) ∨ ¬(∃ b : ℕ, 5 * m = b ^ 3)) ∧ 
  (∃ a : ℕ, 3 * 1875 = a ^ 2) ∧ 
  (∃ b : ℕ, 5 * 1875 = b ^ 3) := by
  sorry

end smallest_n_square_and_cube_l3893_389334


namespace zions_dad_age_difference_l3893_389388

/-- Proves that Zion's dad's age is 3 years more than 4 times Zion's age given the conditions. -/
theorem zions_dad_age_difference (zion_age : ℕ) (dad_age : ℕ) : 
  zion_age = 8 →
  dad_age > 4 * zion_age →
  dad_age + 10 = (zion_age + 10) + 27 →
  dad_age = 4 * zion_age + 3 := by
sorry

end zions_dad_age_difference_l3893_389388


namespace sin_alpha_for_point_l3893_389366

theorem sin_alpha_for_point (α : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 3 ∧ r * Real.sin α = -4) →
  Real.sin α = -4/5 := by
sorry

end sin_alpha_for_point_l3893_389366


namespace copper_content_bounds_l3893_389318

/-- Represents the composition of an alloy --/
structure Alloy where
  nickel : ℝ
  copper : ℝ
  manganese : ℝ
  sum_to_one : nickel + copper + manganese = 1

/-- The three initial alloys --/
def alloy1 : Alloy := ⟨0.3, 0.7, 0, by norm_num⟩
def alloy2 : Alloy := ⟨0, 0.1, 0.9, by norm_num⟩
def alloy3 : Alloy := ⟨0.15, 0.25, 0.6, by norm_num⟩

/-- The fraction of each initial alloy in the new alloy --/
structure Fractions where
  x1 : ℝ
  x2 : ℝ
  x3 : ℝ
  sum_to_one : x1 + x2 + x3 = 1
  manganese_constraint : 0.9 * x2 + 0.6 * x3 = 0.4

/-- The copper content in the new alloy --/
def copper_content (f : Fractions) : ℝ :=
  0.7 * f.x1 + 0.1 * f.x2 + 0.25 * f.x3

/-- The main theorem stating the bounds of copper content --/
theorem copper_content_bounds (f : Fractions) : 
  0.4 ≤ copper_content f ∧ copper_content f ≤ 13/30 := by sorry

end copper_content_bounds_l3893_389318


namespace perimeter_ratio_specific_triangle_l3893_389357

/-- Right triangle DEF with altitude FG and external point J -/
structure RightTriangleWithAltitude where
  /-- Length of side DF -/
  df : ℝ
  /-- Length of side EF -/
  ef : ℝ
  /-- Length of hypotenuse DE -/
  de : ℝ
  /-- Length of altitude FG -/
  fg : ℝ
  /-- Length of tangent from J to circle with diameter FG -/
  tj : ℝ
  /-- de² = df² + ef² (Pythagorean theorem) -/
  pythagoras : de^2 = df^2 + ef^2
  /-- fg² = df * ef (geometric mean property of altitude) -/
  altitude_property : fg^2 = df * ef
  /-- tj² = df * (de - df) (tangent-secant theorem) -/
  tangent_secant : tj^2 = df * (de - df)

/-- Theorem: Perimeter ratio for specific right triangle -/
theorem perimeter_ratio_specific_triangle :
  ∀ t : RightTriangleWithAltitude,
  t.df = 9 →
  t.ef = 40 →
  (t.de + 2 * t.tj) / t.de = 49 / 41 := by
  sorry

end perimeter_ratio_specific_triangle_l3893_389357


namespace product_of_positive_real_part_roots_l3893_389325

theorem product_of_positive_real_part_roots : ∃ (roots : Finset ℂ),
  (∀ z ∈ roots, z^6 = -64) ∧
  (∀ z ∈ roots, (z.re : ℝ) > 0) ∧
  (roots.prod id = 4) := by
sorry

end product_of_positive_real_part_roots_l3893_389325


namespace inequality_equivalence_l3893_389345

theorem inequality_equivalence (x : ℝ) : (x - 3) / (x^2 + 4*x + 10) ≥ 0 ↔ x ≥ 3 := by
  sorry

end inequality_equivalence_l3893_389345


namespace cube_inequality_iff_l3893_389344

theorem cube_inequality_iff (a b : ℝ) : a > b ↔ a^3 > b^3 := by
  sorry

end cube_inequality_iff_l3893_389344


namespace quadratic_form_k_value_l3893_389343

theorem quadratic_form_k_value :
  ∃ (a h k : ℚ), ∀ x, x^2 - 5*x = a*(x - h)^2 + k ∧ k = -25/4 := by
  sorry

end quadratic_form_k_value_l3893_389343


namespace unique_positive_number_l3893_389362

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x - 4 = 21 / x := by
  sorry

end unique_positive_number_l3893_389362


namespace max_mondays_in_51_days_l3893_389332

theorem max_mondays_in_51_days : ∀ (start_day : Nat),
  (start_day < 7) →
  (∃ (monday_count : Nat),
    monday_count = (51 / 7 : Nat) + (if start_day ≤ 1 then 1 else 0) ∧
    monday_count ≤ 8 ∧
    ∀ (other_count : Nat),
      (∃ (other_start : Nat), other_start < 7 ∧
        other_count = (51 / 7 : Nat) + (if other_start ≤ 1 then 1 else 0)) →
      other_count ≤ monday_count) :=
by sorry

end max_mondays_in_51_days_l3893_389332


namespace classroom_size_l3893_389300

theorem classroom_size (x : ℕ) 
  (h1 : (11 * x : ℝ) = (10 * (x - 1) + 30 : ℝ)) : x = 20 := by
  sorry

end classroom_size_l3893_389300


namespace rhombus_closeness_range_l3893_389350

-- Define the closeness function
def closeness (α β : ℝ) : ℝ := 180 - |α - β|

-- Theorem statement
theorem rhombus_closeness_range :
  ∀ α β : ℝ, 0 < α ∧ α < 180 → 0 < β ∧ β < 180 →
  0 < closeness α β ∧ closeness α β ≤ 180 :=
by sorry

end rhombus_closeness_range_l3893_389350


namespace smallest_c_for_max_at_zero_l3893_389359

/-- Given a function y = a * cos(b * x + c) where a, b, and c are positive constants,
    and the graph reaches a maximum at x = 0, the smallest possible value of c is 0. -/
theorem smallest_c_for_max_at_zero (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, a * Real.cos (b * x + c) ≤ a * Real.cos c) →
  (∀ ε > 0, ∃ x, a * Real.cos (b * x + (c - ε)) > a * Real.cos (c - ε)) →
  c = 0 := by
  sorry

end smallest_c_for_max_at_zero_l3893_389359


namespace value_k_std_dev_below_mean_l3893_389356

-- Define the properties of the normal distribution
def mean : ℝ := 12
def std_dev : ℝ := 1.2

-- Define the range for k
def k_range (k : ℝ) : Prop := 2 < k ∧ k < 3 ∧ k ≠ ⌊k⌋

-- Theorem statement
theorem value_k_std_dev_below_mean (k : ℝ) (h : k_range k) :
  ∃ (value : ℝ), value = mean - k * std_dev :=
sorry

end value_k_std_dev_below_mean_l3893_389356


namespace proposition_truth_values_l3893_389331

theorem proposition_truth_values (p q : Prop) (h1 : ¬p) (h2 : q) :
  ¬p ∧ ¬(p ∧ q) ∧ ¬(¬q) ∧ (p ∨ q) :=
by sorry

end proposition_truth_values_l3893_389331


namespace watch_cost_price_l3893_389380

/-- Proves that the cost price of a watch is 3000, given the conditions of the problem -/
theorem watch_cost_price (loss_percentage : ℚ) (gain_percentage : ℚ) (price_difference : ℚ) :
  loss_percentage = 10 / 100 →
  gain_percentage = 8 / 100 →
  price_difference = 540 →
  ∃ (cost_price : ℚ),
    cost_price * (1 - loss_percentage) + price_difference = cost_price * (1 + gain_percentage) ∧
    cost_price = 3000 :=
by sorry

end watch_cost_price_l3893_389380


namespace pentagon_sum_l3893_389336

/-- Pentagon with specific properties -/
structure Pentagon where
  u : ℤ
  v : ℤ
  h1 : 1 ≤ v
  h2 : v < u
  A : ℝ × ℝ := (u, v)
  B : ℝ × ℝ := (v, u)
  C : ℝ × ℝ := (-v, u)
  D : ℝ × ℝ := (-u, v)
  E : ℝ × ℝ := (-u, -v)
  h3 : (D.1 - E.1) * (A.1 - E.1) + (D.2 - E.2) * (A.2 - E.2) = 0  -- ∠DEA = 90°
  h4 : (u^2 : ℝ) + v^2 = 500  -- Area of pentagon ABCDE is 500

/-- Theorem stating the sum of u and v -/
theorem pentagon_sum (p : Pentagon) : p.u + p.v = 20 := by
  sorry

end pentagon_sum_l3893_389336


namespace winnie_yesterday_repetitions_l3893_389363

/-- The number of repetitions Winnie completed yesterday -/
def yesterday_repetitions : ℕ := 86

/-- The number of repetitions Winnie completed today -/
def today_repetitions : ℕ := 73

/-- The number of repetitions Winnie fell behind by today -/
def difference : ℕ := 13

/-- Theorem: Winnie completed 86 repetitions yesterday -/
theorem winnie_yesterday_repetitions :
  yesterday_repetitions = today_repetitions + difference :=
by sorry

end winnie_yesterday_repetitions_l3893_389363


namespace quadratic_roots_condition_l3893_389324

/-- A quadratic equation ax^2 + bx + c = 0 has two distinct real roots if and only if its discriminant is positive -/
axiom quadratic_two_roots (a b c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔ b^2 - 4*a*c > 0

/-- If c < 1/4, then the quadratic equation x^2 + 2x + 4c = 0 has two distinct real roots -/
theorem quadratic_roots_condition (c : ℝ) (h : c < 1/4) :
  ∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + 4*c = 0 ∧ y^2 + 2*y + 4*c = 0 := by
sorry

end quadratic_roots_condition_l3893_389324


namespace correct_average_calculation_l3893_389329

/-- Proves that the correct average is 22 given the conditions of the problem -/
theorem correct_average_calculation (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) 
  (hn : n = 10) 
  (hinitial : initial_avg = 18) 
  (hincorrect : incorrect_num = 26)
  (hcorrect : correct_num = 66) :
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 22 :=
by sorry

end correct_average_calculation_l3893_389329


namespace pencil_price_l3893_389304

theorem pencil_price (price : ℝ) : 
  price = 5000 - 20 → price / 10000 = 0.5 := by sorry

end pencil_price_l3893_389304


namespace poem_line_increase_l3893_389352

theorem poem_line_increase (initial_lines : ℕ) (target_lines : ℕ) (lines_per_month : ℕ) (months : ℕ) : 
  initial_lines = 24 →
  target_lines = 90 →
  lines_per_month = 3 →
  initial_lines + months * lines_per_month = target_lines →
  months = 22 := by
sorry

end poem_line_increase_l3893_389352


namespace faculty_reduction_l3893_389385

theorem faculty_reduction (original : ℝ) (reduction_percentage : ℝ) : 
  original = 253.25 → 
  reduction_percentage = 0.23 →
  ⌊original - (original * reduction_percentage)⌋ = 195 := by
sorry

end faculty_reduction_l3893_389385


namespace constrained_words_count_l3893_389358

/-- The number of possible letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- A five-letter word with the given constraints -/
structure ConstrainedWord :=
  (first : Fin alphabet_size)
  (second : Fin alphabet_size)
  (third : Fin alphabet_size)

/-- The total number of constrained words -/
def total_constrained_words : ℕ := alphabet_size ^ 3

theorem constrained_words_count :
  total_constrained_words = 17576 := by
  sorry

end constrained_words_count_l3893_389358


namespace inequality_solution_set_l3893_389392

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) :
  (1 / x ≤ 1 / 3) ↔ (x ≥ 3 ∨ x < 0) := by
  sorry

end inequality_solution_set_l3893_389392


namespace prime_sum_product_l3893_389398

/-- Given prime numbers a, b, and c satisfying abc + a + b + c = 99,
    prove that two of the numbers are 2 and the other is 19 -/
theorem prime_sum_product (a b c : ℕ) : 
  Prime a → Prime b → Prime c → a * b * c + a + b + c = 99 →
  ((a = 2 ∧ b = 2 ∧ c = 19) ∨ (a = 2 ∧ b = 19 ∧ c = 2) ∨ (a = 19 ∧ b = 2 ∧ c = 2)) :=
by sorry

end prime_sum_product_l3893_389398


namespace swan_count_l3893_389347

/-- The number of swans in a lake that has "a pair plus two more" -/
def pair_plus_two (x : ℕ) : Prop := ∃ n : ℕ, x = 2 * n + 2

/-- The number of swans in a lake that has "three minus three" -/
def three_minus_three (x : ℕ) : Prop := ∃ m : ℕ, x = 3 * m - 3

/-- The total number of swans satisfies both conditions -/
theorem swan_count : ∃ x : ℕ, pair_plus_two x ∧ three_minus_three x ∧ x = 12 := by
  sorry

end swan_count_l3893_389347


namespace correct_sunset_time_l3893_389381

-- Define a custom time type
structure Time where
  hours : Nat
  minutes : Nat

-- Define addition for Time
def Time.add (t1 t2 : Time) : Time :=
  let totalMinutes := t1.hours * 60 + t1.minutes + t2.hours * 60 + t2.minutes
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

-- Convert 24-hour format to 12-hour format
def to12HourFormat (t : Time) : Time :=
  if t.hours ≥ 12 then
    { hours := if t.hours = 12 then 12 else t.hours - 12, minutes := t.minutes }
  else
    { hours := if t.hours = 0 then 12 else t.hours, minutes := t.minutes }

theorem correct_sunset_time :
  let sunrise : Time := { hours := 6, minutes := 57 }
  let daylight : Time := { hours := 10, minutes := 24 }
  let sunset := to12HourFormat (Time.add sunrise daylight)
  sunset = { hours := 5, minutes := 21 } := by sorry

end correct_sunset_time_l3893_389381


namespace fold_sum_l3893_389399

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a fold on a piece of graph paper -/
structure Fold where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- Theorem: If a piece of graph paper is folded such that (0,3) matches with (5,0) 
    and (8,4) matches with (p,q), then p + q = 10 -/
theorem fold_sum (f : Fold) (h1 : f.p1 = ⟨0, 3⟩) (h2 : f.p2 = ⟨5, 0⟩) 
    (h3 : f.p3 = ⟨8, 4⟩) (h4 : f.p4 = ⟨f.p4.x, f.p4.y⟩) : 
    f.p4.x + f.p4.y = 10 := by
  sorry

end fold_sum_l3893_389399


namespace product_sum_squares_l3893_389377

theorem product_sum_squares (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) = (a*c + b*d)^2 + (a*d - b*c)^2 := by
  sorry

end product_sum_squares_l3893_389377


namespace inequality_solution_l3893_389339

theorem inequality_solution (x : ℝ) :
  (x^3 / (x + 2) ≥ 3 / (x - 2) + 1) ↔ (x < -2 ∨ x ≥ 2) :=
by sorry

end inequality_solution_l3893_389339


namespace train_speed_l3893_389335

/-- The speed of a train passing through a tunnel -/
theorem train_speed (train_length : Real) (tunnel_length : Real) (time_minutes : Real) :
  train_length = 0.1 →
  tunnel_length = 2.9 →
  time_minutes = 2.5 →
  ∃ (speed : Real), abs (speed - 71.94) < 0.01 ∧ 
    speed = (tunnel_length + train_length) / (time_minutes / 60) := by
  sorry


end train_speed_l3893_389335


namespace ab_nonzero_sufficient_not_necessary_for_a_nonzero_l3893_389375

theorem ab_nonzero_sufficient_not_necessary_for_a_nonzero :
  (∀ a b : ℝ, a * b ≠ 0 → a ≠ 0) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ a * b = 0) :=
by sorry

end ab_nonzero_sufficient_not_necessary_for_a_nonzero_l3893_389375


namespace jug_fills_ten_large_glasses_l3893_389391

/-- Represents the volume of a glass -/
structure Glass :=
  (volume : ℚ)

/-- Represents a jug with a certain capacity -/
structure Jug :=
  (capacity : ℚ)

/-- Represents the problem setup -/
structure JugProblem :=
  (small_glass : Glass)
  (large_glass : Glass)
  (jug : Jug)
  (condition1 : 9 * small_glass.volume + 4 * large_glass.volume = jug.capacity)
  (condition2 : 6 * small_glass.volume + 6 * large_glass.volume = jug.capacity)

theorem jug_fills_ten_large_glasses (problem : JugProblem) :
  problem.jug.capacity = 10 * problem.large_glass.volume :=
sorry

end jug_fills_ten_large_glasses_l3893_389391


namespace triangle_properties_l3893_389322

/-- Triangle ABC with given points and conditions -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  M : ℝ × ℝ
  h_A : A = (-2, 1)
  h_B : B = (4, 3)

/-- The equation of a line in general form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point lies on a line -/
def lies_on (p : ℝ × ℝ) (l : LineEquation) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Predicate to check if a line is perpendicular to another line -/
def perpendicular (l1 l2 : LineEquation) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem triangle_properties (t : Triangle) :
  (t.C = (3, -2) →
    ∃ (l : LineEquation), l.a = 1 ∧ l.b = 5 ∧ l.c = -3 ∧
    lies_on t.A l ∧
    ∃ (bc : LineEquation), lies_on t.B bc ∧ lies_on t.C bc ∧ perpendicular l bc) ∧
  (t.M = (3, 1) ∧ t.M.1 = (t.A.1 + t.C.1) / 2 ∧ t.M.2 = (t.A.2 + t.C.2) / 2 →
    ∃ (l : LineEquation), l.a = 1 ∧ l.b = 2 ∧ l.c = -10 ∧
    lies_on t.B l ∧ lies_on t.C l) :=
sorry

end triangle_properties_l3893_389322


namespace geometric_sequence_second_term_l3893_389342

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_second_term 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 1/4) 
  (h_a3a5 : a 3 * a 5 = 4 * (a 4 - 1)) :
  a 2 = 1/2 := by
sorry

end geometric_sequence_second_term_l3893_389342


namespace missy_yells_84_times_l3893_389365

/-- The number of times Missy yells at her obedient dog -/
def obedient_yells : ℕ := 12

/-- The ratio of yells at the stubborn dog compared to the obedient dog -/
def stubborn_ratio : ℕ := 4

/-- The ratio of yells at the mischievous dog compared to the obedient dog -/
def mischievous_ratio : ℕ := 2

/-- The total number of times Missy yells at all three dogs -/
def total_yells : ℕ := obedient_yells + stubborn_ratio * obedient_yells + mischievous_ratio * obedient_yells

theorem missy_yells_84_times : total_yells = 84 := by
  sorry

end missy_yells_84_times_l3893_389365


namespace luke_total_score_l3893_389348

def total_points (points_per_round : ℕ) (num_rounds : ℕ) : ℕ :=
  points_per_round * num_rounds

theorem luke_total_score :
  let points_per_round : ℕ := 42
  let num_rounds : ℕ := 2
  total_points points_per_round num_rounds = 84 := by sorry

end luke_total_score_l3893_389348


namespace sum_x_y_equals_three_halves_l3893_389314

theorem sum_x_y_equals_three_halves (x y : ℝ) : 
  y = Real.sqrt (3 - 2*x) + Real.sqrt (2*x - 3) → x + y = 3/2 :=
by sorry

end sum_x_y_equals_three_halves_l3893_389314


namespace chord_length_unit_circle_specific_chord_length_l3893_389364

/-- The length of the chord cut by a line on a unit circle -/
theorem chord_length_unit_circle (a b c : ℝ) (h : a^2 + b^2 ≠ 0) :
  let line := {(x, y) : ℝ × ℝ | a * x + b * y + c = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let d := |c| / Real.sqrt (a^2 + b^2)
  2 * Real.sqrt (1 - d^2) = 8/5 :=
by sorry

/-- The specific case for the given problem -/
theorem specific_chord_length :
  let line := {(x, y) : ℝ × ℝ | 3 * x - 4 * y + 3 = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let d := 3 / 5
  2 * Real.sqrt (1 - d^2) = 8/5 :=
by sorry

end chord_length_unit_circle_specific_chord_length_l3893_389364


namespace paulines_convertibles_l3893_389386

/-- Calculates the number of convertibles in Pauline's car collection --/
theorem paulines_convertibles (total : ℕ) (regular_percent trucks_percent sedans_percent sports_percent suvs_percent : ℚ) :
  total = 125 →
  regular_percent = 38/100 →
  trucks_percent = 12/100 →
  sedans_percent = 17/100 →
  sports_percent = 22/100 →
  suvs_percent = 6/100 →
  ∃ (regular trucks sedans sports suvs convertibles : ℕ),
    regular = ⌊(regular_percent * total : ℚ)⌋ ∧
    trucks = ⌊(trucks_percent * total : ℚ)⌋ ∧
    sedans = ⌊(sedans_percent * total : ℚ)⌋ ∧
    sports = ⌊(sports_percent * total : ℚ)⌋ ∧
    suvs = ⌊(suvs_percent * total : ℚ)⌋ ∧
    convertibles = total - (regular + trucks + sedans + sports + suvs) ∧
    convertibles = 8 :=
by
  sorry

end paulines_convertibles_l3893_389386


namespace max_books_with_23_dollars_l3893_389341

/-- Represents the available book purchasing options -/
inductive BookOption
  | Single
  | Set4
  | Set7

/-- Returns the cost of a given book option -/
def cost (option : BookOption) : ℕ :=
  match option with
  | BookOption.Single => 2
  | BookOption.Set4 => 7
  | BookOption.Set7 => 12

/-- Returns the number of books in a given book option -/
def books (option : BookOption) : ℕ :=
  match option with
  | BookOption.Single => 1
  | BookOption.Set4 => 4
  | BookOption.Set7 => 7

/-- Represents a combination of book purchases -/
structure Purchase where
  singles : ℕ
  sets4 : ℕ
  sets7 : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.singles * cost BookOption.Single +
  p.sets4 * cost BookOption.Set4 +
  p.sets7 * cost BookOption.Set7

/-- Calculates the total number of books in a purchase -/
def totalBooks (p : Purchase) : ℕ :=
  p.singles * books BookOption.Single +
  p.sets4 * books BookOption.Set4 +
  p.sets7 * books BookOption.Set7

/-- Theorem: The maximum number of books that can be purchased with $23 is 13 -/
theorem max_books_with_23_dollars :
  ∃ (p : Purchase), totalCost p ≤ 23 ∧
  totalBooks p = 13 ∧
  ∀ (q : Purchase), totalCost q ≤ 23 → totalBooks q ≤ 13 := by
  sorry


end max_books_with_23_dollars_l3893_389341


namespace certain_part_of_number_l3893_389354

theorem certain_part_of_number (x y : ℝ) : 
  x = 1925 → 
  (1 / 7) * x = y + 100 → 
  y = 175 := by
  sorry

end certain_part_of_number_l3893_389354


namespace vasya_reading_time_difference_l3893_389327

/-- Represents the number of books Vasya planned to read each week -/
def planned_books_per_week : ℕ := sorry

/-- Represents the total number of books in the reading list -/
def total_books : ℕ := 12 * planned_books_per_week

/-- Represents the number of weeks it took Vasya to finish when reading one less book per week -/
def actual_weeks : ℕ := 12 + 3

theorem vasya_reading_time_difference :
  (total_books / (planned_books_per_week + 1) = 10) ∧
  (10 = 12 - 2) :=
by sorry

end vasya_reading_time_difference_l3893_389327


namespace value_added_to_numbers_l3893_389395

theorem value_added_to_numbers (n : ℕ) (original_avg new_avg x : ℝ) 
  (h1 : n = 15)
  (h2 : original_avg = 40)
  (h3 : new_avg = 54)
  (h4 : n * new_avg = n * original_avg + n * x) :
  x = 14 := by
  sorry

end value_added_to_numbers_l3893_389395


namespace prob_no_red_square_is_127_128_l3893_389321

/-- Represents a 4-by-4 grid where each cell can be colored red or blue -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Returns true if the grid has a 3-by-3 red square starting at (i, j) -/
def has_red_square (g : Grid) (i j : Fin 2) : Prop :=
  ∀ (x y : Fin 3), g (i + x) (j + y) = true

/-- The probability of a grid not having any 3-by-3 red square -/
def prob_no_red_square : ℚ :=
  1 - (4 : ℚ) / 2^9

theorem prob_no_red_square_is_127_128 :
  prob_no_red_square = 127 / 128 := by sorry

#check prob_no_red_square_is_127_128

end prob_no_red_square_is_127_128_l3893_389321


namespace total_children_l3893_389368

theorem total_children (happy_children sad_children neutral_children boys girls happy_boys sad_girls neutral_boys : ℕ) : 
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  boys = 22 →
  girls = 38 →
  happy_boys = 6 →
  sad_girls = 4 →
  neutral_boys = 10 →
  happy_children + sad_children + neutral_children = boys + girls :=
by
  sorry

end total_children_l3893_389368


namespace expression_simplification_l3893_389316

theorem expression_simplification (b : ℝ) (h : b ≠ -1) :
  1 - (1 / (1 - b / (1 + b))) = -b :=
by sorry

end expression_simplification_l3893_389316


namespace test_questions_count_l3893_389360

theorem test_questions_count :
  ∀ (total_questions : ℕ),
    total_questions % 5 = 0 →
    32 > (70 * total_questions) / 100 →
    32 < (77 * total_questions) / 100 →
    total_questions = 45 :=
by
  sorry

end test_questions_count_l3893_389360


namespace prob_heart_then_ace_is_one_ninety_eighth_l3893_389384

/-- Represents a standard deck of 51 cards (missing the Ace of Spades) -/
def StandardDeck : ℕ := 51

/-- Number of hearts in the deck -/
def NumHearts : ℕ := 13

/-- Number of aces in the deck -/
def NumAces : ℕ := 3

/-- Probability of drawing a heart as the first card and an ace as the second card -/
def prob_heart_then_ace : ℚ := NumHearts / StandardDeck * NumAces / (StandardDeck - 1)

theorem prob_heart_then_ace_is_one_ninety_eighth :
  prob_heart_then_ace = 1 / 98 := by
  sorry

end prob_heart_then_ace_is_one_ninety_eighth_l3893_389384


namespace fourth_term_of_geometric_sequence_l3893_389394

/-- Represents a geometric sequence with a given first term and common ratio -/
def GeometricSequence (a : ℝ) (r : ℝ) : ℕ → ℝ := fun n => a * r ^ (n - 1)

/-- The fourth term of a geometric sequence given its first and sixth terms -/
theorem fourth_term_of_geometric_sequence (a₁ : ℝ) (a₆ : ℝ) :
  a₁ > 0 → a₆ > 0 →
  ∃ (r : ℝ), r > 0 ∧ GeometricSequence a₁ r 6 = a₆ →
  GeometricSequence a₁ r 4 = 1536 :=
by
  sorry

#check fourth_term_of_geometric_sequence 512 125

end fourth_term_of_geometric_sequence_l3893_389394


namespace bill_per_person_l3893_389371

def total_bill : ℚ := 139
def num_people : ℕ := 8
def tip_percentage : ℚ := 1 / 10

theorem bill_per_person : 
  ∃ (bill_share : ℚ), 
    (bill_share * num_people).ceil = 
      ((total_bill * (1 + tip_percentage)).ceil) ∧ 
    bill_share = 1911 / 100 := by
  sorry

end bill_per_person_l3893_389371


namespace problem_statement_l3893_389353

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function has period p if f(x + p) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem problem_statement (f : ℝ → ℝ) (α : ℝ) 
    (h_odd : IsOdd f)
    (h_period : HasPeriod f 5)
    (h_f_neg_three : f (-3) = 1)
    (h_tan_α : Real.tan α = 2) :
    f (20 * Real.sin α * Real.cos α) = -1 := by
  sorry

end problem_statement_l3893_389353


namespace down_payment_calculation_l3893_389315

/-- Proves that the down payment is $4 given the specified conditions -/
theorem down_payment_calculation (purchase_price : ℝ) (monthly_payment : ℝ) 
  (num_payments : ℕ) (interest_rate : ℝ) (down_payment : ℝ) : 
  purchase_price = 112 →
  monthly_payment = 10 →
  num_payments = 12 →
  interest_rate = 0.10714285714285714 →
  down_payment + num_payments * monthly_payment = purchase_price * (1 + interest_rate) →
  down_payment = 4 := by
sorry

end down_payment_calculation_l3893_389315


namespace quadratic_inequality_l3893_389361

theorem quadratic_inequality (x : ℝ) : x ≥ 1 → x^2 + 3*x - 2 ≥ 0 := by
  sorry

end quadratic_inequality_l3893_389361


namespace shopkeeper_loss_percent_l3893_389397

/-- Calculates the loss percent for a shopkeeper given profit margin and theft percentage -/
theorem shopkeeper_loss_percent 
  (profit_margin : ℝ) 
  (theft_percent : ℝ) 
  (hprofit : profit_margin = 0.1) 
  (htheft : theft_percent = 0.4) : 
  (1 - (1 - theft_percent) * (1 + profit_margin)) * 100 = 40 := by
  sorry

end shopkeeper_loss_percent_l3893_389397


namespace inequality_system_solutions_l3893_389301

theorem inequality_system_solutions (m : ℝ) : 
  (∃ x y : ℤ, x ≠ y ∧ 
   3 - 2 * (x : ℝ) ≥ 0 ∧ (x : ℝ) ≥ m ∧
   3 - 2 * (y : ℝ) ≥ 0 ∧ (y : ℝ) ≥ m ∧
   (∀ z : ℤ, z ≠ x ∧ z ≠ y → ¬(3 - 2 * (z : ℝ) ≥ 0 ∧ (z : ℝ) ≥ m))) →
  -1 < m ∧ m ≤ 0 :=
sorry

end inequality_system_solutions_l3893_389301


namespace book_pages_count_l3893_389378

/-- The number of pages Cora read on Monday -/
def monday_pages : ℕ := 23

/-- The number of pages Cora read on Tuesday -/
def tuesday_pages : ℕ := 38

/-- The number of pages Cora read on Wednesday -/
def wednesday_pages : ℕ := 61

/-- The number of pages Cora will read on Thursday -/
def thursday_pages : ℕ := 12

/-- The number of pages Cora will read on Friday -/
def friday_pages : ℕ := 2 * thursday_pages

/-- The total number of pages in the book -/
def total_pages : ℕ := monday_pages + tuesday_pages + wednesday_pages + thursday_pages + friday_pages

theorem book_pages_count : total_pages = 158 := by
  sorry

end book_pages_count_l3893_389378


namespace megan_homework_time_l3893_389373

/-- The time it takes to complete all problems given the number of math problems,
    spelling problems, and problems that can be finished per hour. -/
def time_to_complete (math_problems : ℕ) (spelling_problems : ℕ) (problems_per_hour : ℕ) : ℕ :=
  (math_problems + spelling_problems) / problems_per_hour

/-- Theorem stating that with 36 math problems, 28 spelling problems,
    and the ability to finish 8 problems per hour, it takes 8 hours to complete all problems. -/
theorem megan_homework_time :
  time_to_complete 36 28 8 = 8 := by
  sorry

end megan_homework_time_l3893_389373


namespace optimal_investment_l3893_389387

/-- Represents an investment project with maximum profit and loss rates. -/
structure Project where
  max_profit_rate : ℝ
  max_loss_rate : ℝ

/-- Represents the investment scenario with two projects and constraints. -/
structure InvestmentScenario where
  project_a : Project
  project_b : Project
  total_investment : ℝ
  max_potential_loss : ℝ

/-- Calculates the potential loss for a given investment allocation. -/
def potential_loss (scenario : InvestmentScenario) (invest_a : ℝ) (invest_b : ℝ) : ℝ :=
  invest_a * scenario.project_a.max_loss_rate + invest_b * scenario.project_b.max_loss_rate

/-- Calculates the potential profit for a given investment allocation. -/
def potential_profit (scenario : InvestmentScenario) (invest_a : ℝ) (invest_b : ℝ) : ℝ :=
  invest_a * scenario.project_a.max_profit_rate + invest_b * scenario.project_b.max_profit_rate

/-- Theorem stating that the given investment allocation maximizes potential profits
    while satisfying all constraints. -/
theorem optimal_investment (scenario : InvestmentScenario)
    (h_project_a : scenario.project_a = { max_profit_rate := 1, max_loss_rate := 0.3 })
    (h_project_b : scenario.project_b = { max_profit_rate := 0.5, max_loss_rate := 0.1 })
    (h_total_investment : scenario.total_investment = 100000)
    (h_max_potential_loss : scenario.max_potential_loss = 18000) :
    ∀ (x y : ℝ),
      x + y ≤ scenario.total_investment →
      potential_loss scenario x y ≤ scenario.max_potential_loss →
      potential_profit scenario x y ≤ potential_profit scenario 40000 60000 :=
  sorry

end optimal_investment_l3893_389387


namespace subset_M_l3893_389320

def M : Set ℕ := {x : ℕ | (1 : ℚ) / (x - 2 : ℚ) ≤ 0}

theorem subset_M : {1} ⊆ M := by sorry

end subset_M_l3893_389320


namespace shirts_washed_l3893_389337

theorem shirts_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (unwashed : ℕ) : 
  short_sleeve = 9 → long_sleeve = 21 → unwashed = 1 →
  short_sleeve + long_sleeve - unwashed = 29 := by
sorry

end shirts_washed_l3893_389337


namespace equation_equivalence_l3893_389312

theorem equation_equivalence (x : ℝ) (Q : ℝ) (h : 5 * (3 * x + 7 * Real.pi) = Q) :
  10 * (6 * x + 14 * Real.pi) = 4 * Q := by
  sorry

end equation_equivalence_l3893_389312


namespace square_sum_eq_double_product_implies_zero_l3893_389305

theorem square_sum_eq_double_product_implies_zero (x y z : ℤ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end square_sum_eq_double_product_implies_zero_l3893_389305


namespace jake_roll_combinations_l3893_389349

/-- The number of different combinations of rolls Jake could buy -/
def num_combinations : ℕ := 3

/-- The number of types of rolls available -/
def num_roll_types : ℕ := 3

/-- The total number of rolls Jake needs to purchase -/
def total_rolls : ℕ := 7

/-- The minimum number of each type of roll Jake must purchase -/
def min_per_type : ℕ := 2

theorem jake_roll_combinations :
  num_combinations = 3 ∧
  num_roll_types = 3 ∧
  total_rolls = 7 ∧
  min_per_type = 2 ∧
  total_rolls = num_roll_types * min_per_type + 1 →
  num_combinations = num_roll_types :=
by sorry

end jake_roll_combinations_l3893_389349


namespace solve_equation_l3893_389389

theorem solve_equation : ∃ x : ℝ, (x - 5)^4 = (1/16)⁻¹ ∧ x = 7 := by
  sorry

end solve_equation_l3893_389389


namespace ceiling_floor_difference_l3893_389390

theorem ceiling_floor_difference : 
  ⌈(15 : ℚ) / 8 * (-34 : ℚ) / 4⌉ - ⌊(15 : ℚ) / 8 * ⌊(-34 : ℚ) / 4⌋⌋ = 2 := by
  sorry

end ceiling_floor_difference_l3893_389390


namespace power_equality_implies_exponent_l3893_389319

theorem power_equality_implies_exponent (p : ℕ) : 16^10 = 4^p → p = 20 := by
  sorry

end power_equality_implies_exponent_l3893_389319


namespace sheets_from_jane_l3893_389396

theorem sheets_from_jane (initial_sheets final_sheets given_sheets : ℕ) 
  (h1 : initial_sheets = 212)
  (h2 : given_sheets = 156)
  (h3 : final_sheets = 363) :
  initial_sheets + (final_sheets + given_sheets - initial_sheets) - given_sheets = final_sheets := by
  sorry

#check sheets_from_jane

end sheets_from_jane_l3893_389396


namespace students_without_A_l3893_389306

theorem students_without_A (total : ℕ) (chem : ℕ) (phys : ℕ) (both : ℕ) : 
  total = 40 → chem = 10 → phys = 18 → both = 6 →
  total - (chem + phys - both) = 18 := by sorry

end students_without_A_l3893_389306


namespace wanda_blocks_count_l3893_389313

/-- The total number of blocks Wanda has after receiving more blocks from Theresa -/
def total_blocks (initial : ℕ) (additional : ℕ) : ℕ := initial + additional

/-- Theorem stating that given the initial and additional blocks, Wanda has 83 blocks in total -/
theorem wanda_blocks_count : total_blocks 4 79 = 83 := by
  sorry

end wanda_blocks_count_l3893_389313


namespace olivia_paper_usage_l3893_389317

/-- The number of pieces of paper Olivia initially had -/
def initial_pieces : ℕ := 81

/-- The number of pieces of paper Olivia has left -/
def remaining_pieces : ℕ := 25

/-- The number of pieces of paper Olivia used -/
def used_pieces : ℕ := initial_pieces - remaining_pieces

theorem olivia_paper_usage :
  used_pieces = 56 :=
sorry

end olivia_paper_usage_l3893_389317


namespace students_walking_home_l3893_389308

theorem students_walking_home (total : ℚ) (bus carpool scooter walk : ℚ) : 
  bus = 1/3 * total →
  carpool = 1/5 * total →
  scooter = 1/8 * total →
  walk = total - (bus + carpool + scooter) →
  walk = 41/120 * total := by
sorry

end students_walking_home_l3893_389308


namespace angle_measure_in_triangle_l3893_389374

theorem angle_measure_in_triangle (P Q R : ℝ) (h : P + Q = 60) : P + Q + R = 180 → R = 120 := by
  sorry

end angle_measure_in_triangle_l3893_389374


namespace point_on_parabola_l3893_389355

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 1

-- Define the theorem
theorem point_on_parabola (y w : ℝ) :
  parabola 3 = y → w = 2 → y = 4 * w := by
  sorry

end point_on_parabola_l3893_389355


namespace airplane_seats_theorem_l3893_389310

/-- Represents the total number of seats in an airplane -/
def total_seats : ℕ := 180

/-- Represents the number of seats in First Class -/
def first_class_seats : ℕ := 36

/-- Represents the fraction of total seats in Business Class -/
def business_class_fraction : ℚ := 1/5

/-- Represents the fraction of total seats in Economy Class -/
def economy_class_fraction : ℚ := 3/5

/-- Theorem stating that the total number of seats is correct given the conditions -/
theorem airplane_seats_theorem :
  (first_class_seats : ℚ) + 
  business_class_fraction * total_seats + 
  economy_class_fraction * total_seats = total_seats := by sorry

end airplane_seats_theorem_l3893_389310


namespace incorrect_height_calculation_l3893_389311

theorem incorrect_height_calculation (n : ℕ) (initial_avg real_avg actual_height : ℝ) 
  (h1 : n = 35)
  (h2 : initial_avg = 185)
  (h3 : real_avg = 183)
  (h4 : actual_height = 106) :
  ∃ (incorrect_height : ℝ),
    incorrect_height = n * initial_avg - (n * real_avg - actual_height) ∧
    incorrect_height = 176 := by
  sorry

end incorrect_height_calculation_l3893_389311


namespace soap_box_height_is_five_l3893_389370

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Theorem: Given the carton and soap box dimensions, and the maximum number of soap boxes,
    the height of the soap box must be 5 inches -/
theorem soap_box_height_is_five
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (max_boxes : ℕ)
  (h_carton_length : carton.length = 25)
  (h_carton_width : carton.width = 48)
  (h_carton_height : carton.height = 60)
  (h_soap_length : soap.length = 8)
  (h_soap_width : soap.width = 6)
  (h_max_boxes : max_boxes = 300)
  (h_fit : max_boxes * boxVolume soap = boxVolume carton) :
  soap.height = 5 := by
  sorry

end soap_box_height_is_five_l3893_389370


namespace area_relation_l3893_389302

/-- A triangle is acute-angled if all its angles are less than 90 degrees. -/
def IsAcuteAngledTriangle (A B C : ℝ × ℝ) : Prop := sorry

/-- The orthocentre of a triangle is the point where all three altitudes intersect. -/
def Orthocentre (A B C H : ℝ × ℝ) : Prop := sorry

/-- The centroid of a triangle is the arithmetic mean position of all points in the triangle. -/
def Centroid (A B C G : ℝ × ℝ) : Prop := sorry

/-- The area of a triangle given its vertices. -/
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem area_relation (A B C H G₁ G₂ G₃ : ℝ × ℝ) :
  IsAcuteAngledTriangle A B C →
  Orthocentre A B C H →
  Centroid H B C G₁ →
  Centroid H C A G₂ →
  Centroid H A B G₃ →
  TriangleArea G₁ G₂ G₃ = 7 →
  TriangleArea A B C = 63 := by
  sorry

end area_relation_l3893_389302


namespace polynomial_has_non_real_root_l3893_389376

def is_valid_polynomial (P : Polynomial ℝ) : Prop :=
  (P.degree ≥ 4) ∧
  (∀ i, P.coeff i ∈ ({-1, 0, 1} : Set ℝ)) ∧
  (P.eval 0 ≠ 0)

theorem polynomial_has_non_real_root (P : Polynomial ℝ) 
  (h : is_valid_polynomial P) : 
  ∃ z : ℂ, z.im ≠ 0 ∧ P.eval (z.re : ℝ) = 0 :=
sorry

end polynomial_has_non_real_root_l3893_389376


namespace smallest_operation_between_sqrt18_and_sqrt8_l3893_389367

theorem smallest_operation_between_sqrt18_and_sqrt8 :
  let a := Real.sqrt 18
  let b := Real.sqrt 8
  (a - b < a + b) ∧ (a - b < a * b) ∧ (a - b < a / b) := by
  sorry

end smallest_operation_between_sqrt18_and_sqrt8_l3893_389367


namespace v_1010_proof_l3893_389369

/-- Represents the last term of the nth group in the sequence -/
def f (n : ℕ) : ℕ := (5 * n^2 - 3 * n + 2) / 2

/-- Represents the total number of terms up to and including the nth group -/
def total_terms (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 1010th term of the sequence -/
def v_1010 : ℕ := 4991

theorem v_1010_proof : 
  ∃ (group : ℕ), 
    total_terms group ≥ 1010 ∧ 
    total_terms (group - 1) < 1010 ∧
    v_1010 = f group - (total_terms group - 1010) :=
sorry

end v_1010_proof_l3893_389369


namespace parity_of_D_2021_2022_2023_l3893_389382

def D : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | n+3 => D (n+2) + D n

theorem parity_of_D_2021_2022_2023 :
  (D 2021 % 2 = 0) ∧ (D 2022 % 2 = 1) ∧ (D 2023 % 2 = 0) := by
  sorry

end parity_of_D_2021_2022_2023_l3893_389382


namespace mark_donation_shelters_l3893_389393

/-- The number of shelters Mark donates soup to -/
def num_shelters (people_per_shelter : ℕ) (cans_per_person : ℕ) (total_cans : ℕ) : ℕ :=
  total_cans / (people_per_shelter * cans_per_person)

theorem mark_donation_shelters :
  num_shelters 30 10 1800 = 6 := by
  sorry

end mark_donation_shelters_l3893_389393


namespace integral_exp_2x_l3893_389303

theorem integral_exp_2x : ∫ x in (0)..(1/2), Real.exp (2*x) = (1/2) * (Real.exp 1 - 1) := by sorry

end integral_exp_2x_l3893_389303


namespace dog_roaming_area_l3893_389372

/-- The area a dog can roam when tied to the corner of a rectangular shed --/
theorem dog_roaming_area (shed_length shed_width leash_length : ℝ) 
  (h1 : shed_length = 4)
  (h2 : shed_width = 3)
  (h3 : leash_length = 4) : 
  let area := (3/4) * Real.pi * leash_length^2 + (1/4) * Real.pi * 1^2
  area = 12.25 * Real.pi := by
  sorry

end dog_roaming_area_l3893_389372


namespace smallest_equal_gum_pieces_l3893_389351

theorem smallest_equal_gum_pieces (n : ℕ) : n > 0 ∧ n % 6 = 0 ∧ n % 5 = 0 ∧ n % 8 = 0 → n ≥ 120 := by
  sorry

end smallest_equal_gum_pieces_l3893_389351


namespace rectangle_sides_l3893_389309

theorem rectangle_sides (x y : ℝ) : 
  (2 * (x + y) = 124) →  -- Perimeter of rectangle is 124 cm
  (4 * Real.sqrt ((x/2)^2 + ((124/2 - x)/2)^2) = 100) →  -- Perimeter of rhombus is 100 cm
  ((x = 48 ∧ y = 14) ∨ (x = 14 ∧ y = 48)) :=
by sorry

end rectangle_sides_l3893_389309


namespace total_seeds_equals_45_l3893_389383

/-- The number of flowerbeds -/
def num_flowerbeds : ℕ := 9

/-- The number of seeds planted in each flowerbed -/
def seeds_per_flowerbed : ℕ := 5

/-- The total number of seeds planted -/
def total_seeds : ℕ := num_flowerbeds * seeds_per_flowerbed

theorem total_seeds_equals_45 : total_seeds = 45 := by
  sorry

end total_seeds_equals_45_l3893_389383


namespace tangent_line_to_circle_l3893_389346

/-- The value of a when a line is tangent to a circle --/
theorem tangent_line_to_circle (a : ℝ) : 
  a > 0 →
  (∃ (x y : ℝ), x^2 + y^2 - a*x = 0 ∧ x - y - 1 = 0) →
  (∀ (x y : ℝ), x^2 + y^2 - a*x = 0 → x - y - 1 ≠ 0 ∨ 
    (∃ (x' y' : ℝ), x' ≠ x ∧ y' ≠ y ∧ x'^2 + y'^2 - a*x' = 0 ∧ x' - y' - 1 = 0)) →
  a = 2*(Real.sqrt 2 - 1) := by
sorry

end tangent_line_to_circle_l3893_389346


namespace no_non_multiple_ghosts_l3893_389330

/-- Definition of the sequence S -/
def S (p : ℕ) : ℕ → ℕ
  | n => if n < p then n else sorry

/-- A number is a ghost if it doesn't appear in S -/
def is_ghost (p : ℕ) (k : ℕ) : Prop :=
  ∀ n, S p n ≠ k

/-- Main theorem: There are no ghosts that are not multiples of p -/
theorem no_non_multiple_ghosts (p : ℕ) (hp : Prime p) (hp_odd : Odd p) :
  ∀ k, ¬(p ∣ k) → ¬(is_ghost p k) := by sorry

end no_non_multiple_ghosts_l3893_389330


namespace fraction_decomposition_sum_l3893_389379

theorem fraction_decomposition_sum : ∃ (C D : ℝ), 
  (∀ x : ℝ, x ≠ 2 → x ≠ 4 → (D * x - 17) / ((x - 2) * (x - 4)) = C / (x - 2) + 4 / (x - 4)) ∧
  C + D = 8.5 := by
sorry

end fraction_decomposition_sum_l3893_389379


namespace add_negative_two_and_two_equals_zero_l3893_389307

theorem add_negative_two_and_two_equals_zero : (-2) + 2 = 0 := by
  sorry

end add_negative_two_and_two_equals_zero_l3893_389307


namespace symmetry_point_xOy_l3893_389326

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOy plane in 3D space -/
def xOyPlane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry with respect to the xOy plane -/
def symmetryXOy (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

theorem symmetry_point_xOy :
  let P : Point3D := { x := -3, y := 2, z := -1 }
  symmetryXOy P = { x := -3, y := 2, z := 1 } := by
  sorry

end symmetry_point_xOy_l3893_389326
