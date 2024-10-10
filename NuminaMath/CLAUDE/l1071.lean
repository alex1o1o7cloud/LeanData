import Mathlib

namespace reciprocal_sum_fourths_sixths_l1071_107136

theorem reciprocal_sum_fourths_sixths : (1 / (1/4 + 1/6) : ℚ) = 12/5 := by
  sorry

end reciprocal_sum_fourths_sixths_l1071_107136


namespace multiple_of_10_average_l1071_107168

theorem multiple_of_10_average (N : ℕ) : 
  N % 10 = 0 → -- N is a multiple of 10
  (10 + N) / 2 = 305 → -- The average of multiples of 10 from 10 to N inclusive is 305
  N = 600 := by
sorry

end multiple_of_10_average_l1071_107168


namespace r_value_when_n_is_3_l1071_107197

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let t : ℕ := 2^n + n
  let r : ℕ := 3^t - t
  r = 177136 := by sorry

end r_value_when_n_is_3_l1071_107197


namespace fraction_calculation_l1071_107191

theorem fraction_calculation : (1 / 3 : ℚ) * (4 / 7 : ℚ) * (9 / 13 : ℚ) + (1 / 2 : ℚ) = 49 / 78 := by
  sorry

end fraction_calculation_l1071_107191


namespace division_subtraction_problem_l1071_107188

theorem division_subtraction_problem (x : ℝ) : 
  (800 / x) - 154 = 6 → x = 5 := by
  sorry

end division_subtraction_problem_l1071_107188


namespace investment_growth_l1071_107162

/-- Proves that an initial investment of $400, when compounded annually at 12% interest for 5 years, results in a final value of $704.98. -/
theorem investment_growth (initial_investment : ℝ) (interest_rate : ℝ) (years : ℕ) (final_value : ℝ) :
  initial_investment = 400 →
  interest_rate = 0.12 →
  years = 5 →
  final_value = 704.98 →
  final_value = initial_investment * (1 + interest_rate) ^ years := by
  sorry


end investment_growth_l1071_107162


namespace residue_products_l1071_107132

theorem residue_products (n k : ℕ+) : 
  (∃ (a : Fin n → ℤ) (b : Fin k → ℤ), 
    ∀ (i j i' j' : ℕ) (hi : i < n) (hj : j < k) (hi' : i' < n) (hj' : j' < k),
      (i ≠ i' ∨ j ≠ j') → 
      (a ⟨i, hi⟩ * b ⟨j, hj⟩) % (n * k : ℕ) ≠ (a ⟨i', hi'⟩ * b ⟨j', hj'⟩) % (n * k : ℕ)) ↔ 
  Nat.gcd n k = 1 :=
sorry

end residue_products_l1071_107132


namespace decimal_to_fraction_l1071_107140

theorem decimal_to_fraction : (3.56 : ℚ) = 89 / 25 := by sorry

end decimal_to_fraction_l1071_107140


namespace first_three_squares_s_3_equals_149_l1071_107182

/-- s(n) is the n-digit number formed by attaching the first n perfect squares in order -/
def s (n : ℕ) : ℕ :=
  sorry

/-- The first three perfect squares are 1, 4, and 9 -/
theorem first_three_squares : List ℕ := [1, 4, 9]

/-- s(3) is equal to 149 -/
theorem s_3_equals_149 : s 3 = 149 := by
  sorry

end first_three_squares_s_3_equals_149_l1071_107182


namespace negation_equivalence_l1071_107152

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by
  sorry

end negation_equivalence_l1071_107152


namespace min_squares_to_exceed_1000_l1071_107104

/-- Represents the squaring operation on a calculator --/
def square (n : ℕ) : ℕ := n * n

/-- Applies the squaring operation n times to the initial value --/
def repeated_square (initial : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initial
  | n + 1 => square (repeated_square initial n)

/-- The theorem to be proved --/
theorem min_squares_to_exceed_1000 :
  (∀ k < 3, repeated_square 3 k ≤ 1000) ∧
  repeated_square 3 3 > 1000 :=
sorry

end min_squares_to_exceed_1000_l1071_107104


namespace a_range_l1071_107193

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x, a^x > 1 ↔ x < 0

def q (a : ℝ) : Prop := ∀ x, x^2 - x + a > 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := (0 < a ∧ a ≤ 1/4) ∨ (a > 1)

-- Theorem statement
theorem a_range (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → range_of_a a :=
sorry

end a_range_l1071_107193


namespace blocks_used_for_first_building_l1071_107108

/-- Given the number of building blocks Jesse started with, used for farmhouse and fenced-in area, and left at the end, 
    calculate the number of blocks used for the first building. -/
theorem blocks_used_for_first_building 
  (total_blocks : ℕ) 
  (farmhouse_blocks : ℕ) 
  (fenced_area_blocks : ℕ) 
  (blocks_left : ℕ) 
  (h1 : total_blocks = 344) 
  (h2 : farmhouse_blocks = 123) 
  (h3 : fenced_area_blocks = 57) 
  (h4 : blocks_left = 84) :
  total_blocks - farmhouse_blocks - fenced_area_blocks - blocks_left = 80 :=
by sorry

end blocks_used_for_first_building_l1071_107108


namespace greatest_integer_with_gcd_six_one_forty_satisfies_conditions_one_forty_is_greatest_l1071_107156

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 150 ∧ n.gcd 18 = 6 → n ≤ 140 :=
by sorry

theorem one_forty_satisfies_conditions : 140 < 150 ∧ Nat.gcd 140 18 = 6 :=
by sorry

theorem one_forty_is_greatest : 
  ∀ m : ℕ, m < 150 ∧ m.gcd 18 = 6 → m ≤ 140 :=
by sorry

end greatest_integer_with_gcd_six_one_forty_satisfies_conditions_one_forty_is_greatest_l1071_107156


namespace first_divisor_problem_l1071_107130

theorem first_divisor_problem :
  ∃ (d : ℕ+) (x k m : ℤ),
    x = k * d.val + 11 ∧
    x = 9 * m + 2 ∧
    d.val < 11 ∧
    9 % d.val = 0 ∧
    d = 3 := by
  sorry

end first_divisor_problem_l1071_107130


namespace point_A_coordinates_l1071_107169

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 4 = 0

-- Define a point on the circle
def point_on_circle (P : ℝ × ℝ) : Prop := circle_O P.1 P.2

-- Define a point on the line
def point_on_line (A : ℝ × ℝ) : Prop := line_l A.1 A.2

-- Define the angle PAQ
def angle_PAQ (A P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem point_A_coordinates :
  ∀ A : ℝ × ℝ,
  point_on_line A →
  (∀ P Q : ℝ × ℝ, point_on_circle P → point_on_circle Q → angle_PAQ A P Q ≤ 90) →
  (∃ P Q : ℝ × ℝ, point_on_circle P ∧ point_on_circle Q ∧ angle_PAQ A P Q = 90) →
  A = (1, 3) :=
sorry

end point_A_coordinates_l1071_107169


namespace specific_book_arrangement_l1071_107172

/-- The number of arrangements for placing math and English books on a shelf. -/
def book_arrangements (n_math : ℕ) (n_english : ℕ) (adjacent_math : ℕ) : ℕ :=
  Nat.factorial n_english * 
  (n_english - 1) * 
  Nat.choose (n_english + adjacent_math - 1) (n_math - adjacent_math)

/-- Theorem stating the number of arrangements for the specific book problem. -/
theorem specific_book_arrangement : book_arrangements 6 5 2 = 2400 := by
  sorry

#eval book_arrangements 6 5 2

end specific_book_arrangement_l1071_107172


namespace chord_count_l1071_107147

/-- The number of different chords that can be drawn by connecting any two of ten points 
    on the circumference of a circle, where four of these points form a square. -/
def num_chords : ℕ := 45

/-- The total number of points on the circumference of the circle. -/
def total_points : ℕ := 10

/-- The number of points that form a square. -/
def square_points : ℕ := 4

theorem chord_count : 
  num_chords = (total_points * (total_points - 1)) / 2 :=
sorry

end chord_count_l1071_107147


namespace projection_matrix_values_l1071_107144

def is_projection_matrix (Q : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  Q ^ 2 = Q

theorem projection_matrix_values :
  ∀ (x y : ℝ),
  let Q : Matrix (Fin 2) (Fin 2) ℝ := !![x, 1/5; y, 4/5]
  is_projection_matrix Q ↔ x = 1 ∧ y = 0 := by
sorry


end projection_matrix_values_l1071_107144


namespace puzzle_solution_l1071_107159

theorem puzzle_solution :
  ∀ (F I V T E N : ℕ),
    F = 8 →
    N % 2 = 1 →
    F ≠ I ∧ F ≠ V ∧ F ≠ T ∧ F ≠ E ∧ F ≠ N ∧
    I ≠ V ∧ I ≠ T ∧ I ≠ E ∧ I ≠ N ∧
    V ≠ T ∧ V ≠ E ∧ V ≠ N ∧
    T ≠ E ∧ T ≠ N ∧
    E ≠ N →
    F < 10 ∧ I < 10 ∧ V < 10 ∧ T < 10 ∧ E < 10 ∧ N < 10 →
    100 * F + 10 * I + V + 100 * F + 10 * I + V = 1000 * T + 100 * E + 10 * N →
    I = 4 :=
by sorry

end puzzle_solution_l1071_107159


namespace seminar_discount_percentage_l1071_107170

/-- Calculates the discount percentage for early registration of a seminar --/
theorem seminar_discount_percentage
  (regular_fee : ℝ)
  (num_teachers : ℕ)
  (food_allowance : ℝ)
  (total_spent : ℝ)
  (h1 : regular_fee = 150)
  (h2 : num_teachers = 10)
  (h3 : food_allowance = 10)
  (h4 : total_spent = 1525)
  : (1 - (total_spent - num_teachers * food_allowance) / (num_teachers * regular_fee)) * 100 = 5 := by
  sorry

#check seminar_discount_percentage

end seminar_discount_percentage_l1071_107170


namespace ben_votes_l1071_107151

/-- Given a total of 60 votes and a ratio of 2:3 between Ben's and Matt's votes,
    prove that Ben received 24 votes. -/
theorem ben_votes (total_votes : ℕ) (ben_votes : ℕ) (matt_votes : ℕ) :
  total_votes = 60 →
  ben_votes + matt_votes = total_votes →
  3 * ben_votes = 2 * matt_votes →
  ben_votes = 24 := by
sorry

end ben_votes_l1071_107151


namespace hyperbolic_and_linear_functions_l1071_107103

/-- The hyperbolic and linear functions with their properties -/
theorem hyperbolic_and_linear_functions (k : ℝ) (h : |k| < 1) :
  (∀ x y : ℝ, y = (k - 1) / x → (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) ∧
  (k * (-1 + 1) = 0) := by
  sorry

end hyperbolic_and_linear_functions_l1071_107103


namespace negative_values_range_l1071_107119

/-- A quadratic function that takes negative values for some x -/
def takes_negative_values (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - a*x + 1 < 0

/-- The theorem stating the range of a for which f(x) takes negative values -/
theorem negative_values_range (a : ℝ) :
  takes_negative_values a ↔ a > 2 ∨ a < -2 :=
sorry

end negative_values_range_l1071_107119


namespace limit_proof_l1071_107116

open Real

theorem limit_proof (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧
  ∀ x : ℝ, 0 < |x - 1/2| ∧ |x - 1/2| < δ →
    |(2*x^2 - 5*x + 2)/(x - 1/2) + 3| < ε :=
by
  sorry

end limit_proof_l1071_107116


namespace fixed_point_on_line_l1071_107148

/-- The line equation passing through a fixed point for all values of m -/
def line_equation (m x y : ℝ) : Prop :=
  (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

/-- The fixed point P -/
def P : ℝ × ℝ := (3, 1)

/-- Theorem stating that P lies on the line for all real m -/
theorem fixed_point_on_line : ∀ m : ℝ, line_equation m P.1 P.2 := by
  sorry

end fixed_point_on_line_l1071_107148


namespace smallest_integer_divisible_by_24_and_8_l1071_107122

theorem smallest_integer_divisible_by_24_and_8 : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → (¬(24 ∣ m^2) ∨ ¬(8 ∣ m))) ∧ 
  24 ∣ n^2 ∧ 
  8 ∣ n ∧
  ∀ d : ℕ+, d ∣ n → d ≤ 8 :=
by sorry

end smallest_integer_divisible_by_24_and_8_l1071_107122


namespace jens_ducks_l1071_107138

theorem jens_ducks (chickens ducks : ℕ) : 
  ducks = 4 * chickens + 10 →
  chickens + ducks = 185 →
  ducks = 150 := by
sorry

end jens_ducks_l1071_107138


namespace total_dolls_l1071_107186

theorem total_dolls (jazmin_dolls geraldine_dolls : ℕ) 
  (h1 : jazmin_dolls = 1209) 
  (h2 : geraldine_dolls = 2186) : 
  jazmin_dolls + geraldine_dolls = 3395 := by
  sorry

end total_dolls_l1071_107186


namespace decreasing_g_implies_a_nonpositive_l1071_107112

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x

-- Define what it means for g to be decreasing on ℝ
def isDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- Theorem statement
theorem decreasing_g_implies_a_nonpositive :
  ∀ a : ℝ, isDecreasing (g a) → a ≤ 0 :=
sorry

end decreasing_g_implies_a_nonpositive_l1071_107112


namespace max_m_value_l1071_107149

theorem max_m_value (b a m : ℝ) (h_b : b > 0) :
  (∀ a, (b - (a - 2))^2 + (Real.log b - (a - 1))^2 ≥ m^2 - m) →
  m ≤ 2 :=
by sorry

end max_m_value_l1071_107149


namespace reflect_point_1_2_l1071_107174

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The theorem states that reflecting the point (1,2) across the x-axis results in (1,-2) -/
theorem reflect_point_1_2 : reflect_x (1, 2) = (1, -2) := by sorry

end reflect_point_1_2_l1071_107174


namespace minimum_m_value_l1071_107146

theorem minimum_m_value (a x : ℝ) (ha : |a| ≤ 1) (hx : |x| ≤ 1) :
  ∃ m : ℝ, (∀ a x, |a| ≤ 1 → |x| ≤ 1 → |x^2 - a*x - a^2| ≤ m) ∧ 
  (∀ m' : ℝ, m' < m → ∃ a x, |a| ≤ 1 ∧ |x| ≤ 1 ∧ |x^2 - a*x - a^2| > m') ∧
  m = 5/4 :=
sorry

end minimum_m_value_l1071_107146


namespace x_bounds_and_sqrt2_inequality_l1071_107145

theorem x_bounds_and_sqrt2_inequality :
  ∃ x : ℝ,
    (x = (x^2 + 1) / 198) ∧
    (1/198 < x) ∧
    (x < 197.99494949) ∧
    (Real.sqrt 2 < 1.41421356) := by
  sorry

end x_bounds_and_sqrt2_inequality_l1071_107145


namespace sqrt_equation_solution_l1071_107179

theorem sqrt_equation_solution (x : ℚ) : 
  (Real.sqrt (3 * x + 5) / Real.sqrt (6 * x + 5) = Real.sqrt 5 / 3) → x = 20 / 3 := by
  sorry

end sqrt_equation_solution_l1071_107179


namespace dragon_poker_ways_l1071_107199

/-- The number of points to be scored -/
def target_points : ℕ := 2018

/-- The number of suits in the deck -/
def num_suits : ℕ := 4

/-- Calculates the number of ways to partition a given number into a specified number of parts -/
def partition_ways (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The main theorem: The number of ways to score exactly 2018 points in Dragon Poker -/
theorem dragon_poker_ways : partition_ways target_points num_suits = 1373734330 := by
  sorry

end dragon_poker_ways_l1071_107199


namespace batsman_matches_l1071_107124

theorem batsman_matches (total_matches : ℕ) (first_set_matches : ℕ) (first_set_avg : ℝ) 
                         (second_set_avg : ℝ) (total_avg : ℝ) :
  total_matches = 30 →
  first_set_matches = 20 →
  first_set_avg = 30 →
  second_set_avg = 15 →
  total_avg = 25 →
  (total_matches - first_set_matches : ℝ) = 10 := by
  sorry


end batsman_matches_l1071_107124


namespace lawrence_county_summer_break_l1071_107165

/-- The number of kids who stayed home during summer break in Lawrence county -/
theorem lawrence_county_summer_break (total_kids : ℕ) (camp_kids : ℕ) (h1 : total_kids = 1538832) (h2 : camp_kids = 893835) :
  total_kids - camp_kids = 644997 := by
  sorry

#check lawrence_county_summer_break

end lawrence_county_summer_break_l1071_107165


namespace correct_time_per_lap_l1071_107175

/-- The time in minutes for one lap around the playground -/
def time_per_lap : ℝ := 19.2

/-- The number of laps cycled -/
def num_laps : ℕ := 5

/-- The total time in minutes for cycling the given number of laps -/
def total_time : ℝ := 96

theorem correct_time_per_lap : 
  time_per_lap * num_laps = total_time := by sorry

end correct_time_per_lap_l1071_107175


namespace fun_run_ratio_l1071_107153

def runners_last_year : ℕ := 200 - 40
def runners_this_year : ℕ := 320

theorem fun_run_ratio : 
  (runners_this_year : ℚ) / (runners_last_year : ℚ) = 2 := by sorry

end fun_run_ratio_l1071_107153


namespace h_is_even_l1071_107131

-- Define g as an odd function
def g : ℝ → ℝ := sorry

-- Axiom stating that g is an odd function
axiom g_odd : ∀ x : ℝ, g (-x) = -g x

-- Define h using g
def h (x : ℝ) : ℝ := |g (x^4)|

-- Theorem stating that h is an even function
theorem h_is_even : ∀ x : ℝ, h (-x) = h x := by
  sorry

end h_is_even_l1071_107131


namespace expression_evaluation_l1071_107109

theorem expression_evaluation : 
  8^(1/4) * 42 + (32 * Real.sqrt 3)^6 + Real.log 2 / Real.log 3 * (Real.log (Real.log 27 / Real.log 3) / Real.log 2) = 111 := by
  sorry

end expression_evaluation_l1071_107109


namespace sum_of_equal_expressions_l1071_107177

theorem sum_of_equal_expressions (a b c d : ℝ) :
  a + 2 = b + 3 ∧ 
  b + 3 = c + 4 ∧ 
  c + 4 = d + 5 ∧ 
  d + 5 = a + b + c + d + 10 →
  a + b + c + d = -26/3 := by
sorry

end sum_of_equal_expressions_l1071_107177


namespace max_cubes_in_box_l1071_107181

/-- The maximum number of cubes that can fit in a rectangular box -/
theorem max_cubes_in_box (box_length box_width box_height cube_volume : ℕ) :
  box_length = 8 →
  box_width = 9 →
  box_height = 12 →
  cube_volume = 27 →
  (box_length * box_width * box_height) / cube_volume = 32 := by
  sorry

#check max_cubes_in_box

end max_cubes_in_box_l1071_107181


namespace first_quadrant_iff_sin_cos_sum_gt_one_l1071_107123

theorem first_quadrant_iff_sin_cos_sum_gt_one (α : Real) :
  (0 < α ∧ α < Real.pi / 2) ↔ (Real.sin α + Real.cos α > 1) := by
  sorry

end first_quadrant_iff_sin_cos_sum_gt_one_l1071_107123


namespace andrews_age_l1071_107115

theorem andrews_age (a g s : ℝ) 
  (h1 : g = 10 * a)
  (h2 : g - s = a + 45)
  (h3 : s = 5) :
  a = 50 / 9 := by
sorry

end andrews_age_l1071_107115


namespace max_value_xyz_expression_l1071_107157

theorem max_value_xyz_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x * y * z * (x + y + z) / ((x + y)^2 * (y + z)^2) ≤ 1 / 4 := by
  sorry

end max_value_xyz_expression_l1071_107157


namespace triangle_cosine_rule_l1071_107183

/-- Given a triangle ABC where 6 sin A = 4 sin B = 3 sin C, prove that cos C = -1/4 -/
theorem triangle_cosine_rule (A B C : ℝ) (h : 6 * Real.sin A = 4 * Real.sin B ∧ 4 * Real.sin B = 3 * Real.sin C) :
  Real.cos C = -1/4 := by
  sorry

end triangle_cosine_rule_l1071_107183


namespace repeating_decimal_inequality_l1071_107128

/-- Represents a repeating decimal with non-repeating part P and repeating part Q -/
structure RepeatingDecimal where
  P : ℕ  -- non-repeating part
  Q : ℕ  -- repeating part
  r : ℕ  -- number of digits in P
  s : ℕ  -- number of digits in Q

/-- The value of the repeating decimal as a real number -/
noncomputable def decimal_value (D : RepeatingDecimal) : ℝ :=
  sorry

/-- Statement: The equation 10^r(10^s + 1)D = Q(P + 1) is false for repeating decimals -/
theorem repeating_decimal_inequality (D : RepeatingDecimal) :
  (10^D.r * (10^D.s + 1)) * (decimal_value D) ≠ D.Q * (D.P + 1) :=
sorry

end repeating_decimal_inequality_l1071_107128


namespace sin_seven_pi_thirds_l1071_107187

theorem sin_seven_pi_thirds : Real.sin (7 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_seven_pi_thirds_l1071_107187


namespace candy_bar_consumption_l1071_107176

theorem candy_bar_consumption (calories_per_bar : ℕ) (total_calories : ℕ) (num_bars : ℕ) : 
  calories_per_bar = 8 → total_calories = 24 → num_bars = total_calories / calories_per_bar → num_bars = 3 :=
by
  sorry

end candy_bar_consumption_l1071_107176


namespace airplane_trip_people_count_l1071_107194

/-- Represents the airplane trip scenario --/
structure AirplaneTrip where
  bagsPerPerson : ℕ
  weightPerBag : ℕ
  currentCapacity : ℕ
  additionalBags : ℕ

/-- Calculate the number of people on the trip --/
def numberOfPeople (trip : AirplaneTrip) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the number of people on the trip --/
theorem airplane_trip_people_count :
  let trip := AirplaneTrip.mk 5 50 6000 90
  numberOfPeople trip = 42 := by
  sorry

end airplane_trip_people_count_l1071_107194


namespace half_pond_fill_time_l1071_107129

/-- Represents the growth of water hyacinth in a pond -/
def WaterHyacinthGrowth :=
  {growth : ℕ → ℝ // 
    (∀ n, growth (n + 1) = 2 * growth n) ∧ 
    (growth 10 = 1)}

theorem half_pond_fill_time (g : WaterHyacinthGrowth) : 
  g.val 9 = 1/2 := by sorry

end half_pond_fill_time_l1071_107129


namespace problem1_l1071_107196

theorem problem1 (x y : ℝ) : x^2 * (-2*x*y^2)^3 = -8*x^5*y^6 := by sorry

end problem1_l1071_107196


namespace birthday_problem_solution_l1071_107141

/-- Represents a person's age -/
structure Age :=
  (value : ℕ)

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Represents the ages of Alice, Bob, and Carl -/
structure FamilyAges :=
  (alice : Age)
  (bob : Age)
  (carl : Age)

/-- Checks if one age is a multiple of another -/
def isMultipleOf (a b : Age) : Prop :=
  ∃ k : ℕ, a.value = k * b.value

/-- Represents the conditions of the problem -/
structure BirthdayProblem :=
  (ages : FamilyAges)
  (aliceOlderThanBob : ages.alice.value = ages.bob.value + 2)
  (carlAgeToday : ages.carl.value = 3)
  (bobMultipleOfCarl : isMultipleOf ages.bob ages.carl)
  (firstOfFourBirthdays : ∀ n : ℕ, n < 4 → isMultipleOf ⟨ages.bob.value + n⟩ ⟨ages.carl.value + n⟩)

/-- The main theorem to prove -/
theorem birthday_problem_solution (problem : BirthdayProblem) :
  ∃ (futureAliceAge : ℕ),
    futureAliceAge > problem.ages.alice.value ∧
    isMultipleOf ⟨futureAliceAge⟩ ⟨problem.ages.carl.value + (futureAliceAge - problem.ages.alice.value)⟩ ∧
    sumOfDigits futureAliceAge = 6 :=
  sorry

end birthday_problem_solution_l1071_107141


namespace carol_ate_twelve_cakes_l1071_107117

/-- The number of cakes Sara bakes per day -/
def cakes_per_day : ℕ := 10

/-- The number of days Sara bakes cakes -/
def baking_days : ℕ := 5

/-- The number of cans of frosting needed to frost a single cake -/
def cans_per_cake : ℕ := 2

/-- The number of cans of frosting Sara needs for the remaining cakes -/
def cans_needed : ℕ := 76

/-- The number of cakes Carol ate -/
def cakes_eaten_by_carol : ℕ := cakes_per_day * baking_days - cans_needed / cans_per_cake

theorem carol_ate_twelve_cakes : cakes_eaten_by_carol = 12 := by
  sorry

end carol_ate_twelve_cakes_l1071_107117


namespace runs_ratio_l1071_107125

/-- Represents the runs scored by each player -/
structure Runs where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The conditions of the cricket match -/
def cricket_match (r : Runs) : Prop :=
  r.a + r.b + r.c = 95 ∧
  r.c = 75 ∧
  r.a * 3 = r.b * 1

/-- The theorem to prove -/
theorem runs_ratio (r : Runs) (h : cricket_match r) : 
  r.b * 5 = r.c * 1 := by
sorry


end runs_ratio_l1071_107125


namespace multiply_54_46_l1071_107161

theorem multiply_54_46 : 54 * 46 = 2484 := by
  sorry

end multiply_54_46_l1071_107161


namespace diagonal_passes_through_600_cubes_l1071_107137

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (l w h : ℕ) : ℕ :=
  l + w + h - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l) + Nat.gcd l (Nat.gcd w h)

/-- Theorem: An internal diagonal of a 120 × 280 × 360 rectangular solid passes through 600 cubes -/
theorem diagonal_passes_through_600_cubes :
  cubes_passed_by_diagonal 120 280 360 = 600 := by
  sorry

end diagonal_passes_through_600_cubes_l1071_107137


namespace only_234_not_right_triangle_l1071_107142

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that among the given sets, only (2, 3, 4) is not a right triangle --/
theorem only_234_not_right_triangle :
  ¬(is_right_triangle 2 3 4) ∧
  (is_right_triangle 1 1 (Real.sqrt 2)) ∧
  (is_right_triangle (Real.sqrt 2) (Real.sqrt 3) (Real.sqrt 5)) ∧
  (is_right_triangle 3 4 5) :=
by sorry


end only_234_not_right_triangle_l1071_107142


namespace unique_solution_for_exponential_equation_l1071_107139

theorem unique_solution_for_exponential_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (9 * x)^18 = (27 * x)^9 ∧ x = 1/3 := by
  sorry

end unique_solution_for_exponential_equation_l1071_107139


namespace unique_solution_is_identity_l1071_107127

open Set
open Function
open Real

/-- The functional equation that f must satisfy for all positive real numbers x, y, z -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y z, x > 0 → y > 0 → z > 0 →
    (z + 1) * f (x + y) = f (x * f z + y) + f (y * f z + x)

/-- The main theorem stating that the only function satisfying the equation is the identity function -/
theorem unique_solution_is_identity :
  ∀ f : ℝ → ℝ, (∀ x, x > 0 → f x > 0) →
    satisfies_equation f →
    ∀ x, x > 0 → f x = x :=
by sorry


end unique_solution_is_identity_l1071_107127


namespace girls_count_in_school_l1071_107106

/-- Represents the number of students in a school with a given boy-to-girl ratio. -/
structure School where
  total : ℕ
  ratio : ℚ
  boys : ℕ
  girls : ℕ
  ratio_def : ratio = boys / girls
  total_def : total = boys + girls

/-- Theorem: In a school with 90 students and a 1:2 boy-to-girl ratio, there are 60 girls. -/
theorem girls_count_in_school (s : School) 
    (h_total : s.total = 90)
    (h_ratio : s.ratio = 1/2) : 
    s.girls = 60 := by
  sorry

end girls_count_in_school_l1071_107106


namespace arithmetic_sequence_equivalence_l1071_107134

theorem arithmetic_sequence_equivalence
  (a b c : ℕ → ℝ)
  (h1 : ∀ n, b n = a (n + 1) - a n)
  (h2 : ∀ n, c n = a n + 2 * a (n + 1)) :
  (∃ d, ∀ n, a (n + 1) - a n = d) ↔
  ((∃ D, ∀ n, c (n + 1) - c n = D) ∧ (∀ n, b n ≤ b (n + 1))) :=
by sorry

end arithmetic_sequence_equivalence_l1071_107134


namespace certain_number_problem_l1071_107163

theorem certain_number_problem (x : ℝ) : 0.7 * x = 0.6 * 80 + 22 → x = 100 := by
  sorry

end certain_number_problem_l1071_107163


namespace opposite_of_negative_2023_l1071_107120

theorem opposite_of_negative_2023 : -(-(2023 : ℤ)) = 2023 := by
  sorry

end opposite_of_negative_2023_l1071_107120


namespace sum_of_four_digit_numbers_eq_93324_l1071_107154

def digits : List Nat := [2, 4, 5, 3]

/-- The sum of all four-digit numbers formed by using the digits 2, 4, 5, and 3 once each -/
def sum_of_four_digit_numbers : Nat :=
  let sum_of_digits := digits.sum
  let count_per_place := Nat.factorial 4 / 4
  sum_of_digits * count_per_place * (1000 + 100 + 10 + 1)

theorem sum_of_four_digit_numbers_eq_93324 :
  sum_of_four_digit_numbers = 93324 := by
  sorry

end sum_of_four_digit_numbers_eq_93324_l1071_107154


namespace sum_of_i_powers_2021_to_2024_l1071_107126

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Theorem: The sum of i^2021, i^2022, i^2023, and i^2024 is equal to 0 -/
theorem sum_of_i_powers_2021_to_2024 : i^2021 + i^2022 + i^2023 + i^2024 = 0 := by
  sorry

end sum_of_i_powers_2021_to_2024_l1071_107126


namespace min_value_quadratic_l1071_107102

theorem min_value_quadratic (x y : ℝ) : 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 := by
  sorry

end min_value_quadratic_l1071_107102


namespace luncheon_invitees_l1071_107160

/-- The number of people who didn't show up -/
def no_shows : ℕ := 7

/-- The number of people each table can hold -/
def people_per_table : ℕ := 5

/-- The number of tables needed -/
def tables_needed : ℕ := 8

/-- The original number of invited people -/
def original_invitees : ℕ := (tables_needed * people_per_table) + no_shows

theorem luncheon_invitees : original_invitees = 47 := by
  sorry

end luncheon_invitees_l1071_107160


namespace sara_marbles_l1071_107198

def marble_problem (initial : ℕ) (given : ℕ) (lost : ℕ) (traded : ℕ) : Prop :=
  initial + given - lost - traded = 5

theorem sara_marbles : marble_problem 10 5 7 3 := by
  sorry

end sara_marbles_l1071_107198


namespace factor_t_squared_minus_144_l1071_107114

theorem factor_t_squared_minus_144 (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end factor_t_squared_minus_144_l1071_107114


namespace min_value_trig_expression_greatest_lower_bound_trig_expression_l1071_107101

theorem min_value_trig_expression (x : ℝ) : 
  (4 * Real.sin x * Real.cos x + 3) / (Real.cos x)^2 ≥ 5/3 :=
by sorry

theorem greatest_lower_bound_trig_expression :
  ∃ x : ℝ, (4 * Real.sin x * Real.cos x + 3) / (Real.cos x)^2 = 5/3 :=
by sorry

end min_value_trig_expression_greatest_lower_bound_trig_expression_l1071_107101


namespace systematic_sampling_correspondence_l1071_107192

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : Nat
  num_groups : Nat
  students_per_group : Nat
  selected_student : Nat
  selected_group : Nat

/-- Theorem stating the relationship between selected students in different groups. -/
theorem systematic_sampling_correspondence
  (s : SystematicSampling)
  (h1 : s.total_students = 60)
  (h2 : s.num_groups = 5)
  (h3 : s.students_per_group = s.total_students / s.num_groups)
  (h4 : s.selected_student = 16)
  (h5 : s.selected_group = 2)
  : (s.selected_student - (s.selected_group - 1) * s.students_per_group) + 3 * s.students_per_group = 40 := by
  sorry

end systematic_sampling_correspondence_l1071_107192


namespace y1_less_than_y2_l1071_107150

/-- Given a linear function y = 3x - b and two points P₁(3, y₁) and P₂(4, y₂) on its graph,
    prove that y₁ < y₂. -/
theorem y1_less_than_y2 (b : ℝ) (y₁ y₂ : ℝ) 
    (h₁ : y₁ = 3 * 3 - b) 
    (h₂ : y₂ = 3 * 4 - b) : 
  y₁ < y₂ := by
  sorry

end y1_less_than_y2_l1071_107150


namespace nearest_multiple_of_11_to_457_l1071_107135

theorem nearest_multiple_of_11_to_457 :
  ∃ (n : ℤ), n % 11 = 0 ∧ 
  ∀ (m : ℤ), m % 11 = 0 → |n - 457| ≤ |m - 457| ∧
  n = 462 := by
  sorry

end nearest_multiple_of_11_to_457_l1071_107135


namespace dynamic_number_sum_divisible_by_three_l1071_107185

/-- A dynamic number is a four-digit positive integer where each digit is not 0,
    and the two-digit number formed by the tenth and unit places is twice
    the two-digit number formed by the thousandth and hundredth places. -/
def isDynamicNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∀ a b c d : ℕ,
    n = 1000 * a + 100 * b + 10 * c + d →
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    10 * c + d = 2 * (10 * a + b)

theorem dynamic_number_sum_divisible_by_three (a : ℕ) (h : 10 ≤ a ∧ a < 100) :
  ∃ k : ℕ, 102 * a + (200 * a + a) = 3 * k := by
  sorry

end dynamic_number_sum_divisible_by_three_l1071_107185


namespace fraction_subtraction_l1071_107107

theorem fraction_subtraction (d : ℝ) : (6 - 5 * d) / 9 - 3 = (-21 - 5 * d) / 9 := by
  sorry

end fraction_subtraction_l1071_107107


namespace trigonometric_identities_l1071_107133

theorem trigonometric_identities (α : ℝ) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  (Real.sin (2*α) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 1) := by
  sorry

end trigonometric_identities_l1071_107133


namespace quadratic_equation_roots_l1071_107180

theorem quadratic_equation_roots (a : ℝ) : 
  ((a + 1) * (-1)^2 + (-1) - 1 = 0) → 
  (a = 1 ∧ ∃ x : ℝ, x ≠ -1 ∧ (2 * x^2 + x - 1 = 0)) :=
by sorry

end quadratic_equation_roots_l1071_107180


namespace rectangular_plot_area_l1071_107155

theorem rectangular_plot_area 
  (L B : ℝ) 
  (h_ratio : L / B = 7 / 5) 
  (h_perimeter : 2 * (L + B) = 288) : 
  L * B = 5040 := by sorry

end rectangular_plot_area_l1071_107155


namespace part_one_part_two_l1071_107190

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a - 1 < x ∧ x < -a}
def B : Set ℝ := {x | |x - 1| < 2}

-- Part 1
theorem part_one : (Aᶜ (-1) ∪ B) = {x | x ≤ -3 ∨ x > -1} := by sorry

-- Part 2
theorem part_two : ∀ a : ℝ, (A a ⊆ B ∧ A a ≠ B) ↔ a ≥ 0 := by sorry

end part_one_part_two_l1071_107190


namespace inequality_solution_set_l1071_107173

theorem inequality_solution_set (x : ℝ) :
  (x^2 - 5 * abs x + 6 < 0) ↔ ((-3 < x ∧ x < -2) ∨ (2 < x ∧ x < 3)) := by sorry

end inequality_solution_set_l1071_107173


namespace smallest_gcd_of_multiples_l1071_107111

theorem smallest_gcd_of_multiples (a b : ℕ+) (h : Nat.gcd a b = 18) :
  (Nat.gcd (12 * a) (20 * b)).min = 72 := by
  sorry

end smallest_gcd_of_multiples_l1071_107111


namespace line_intersects_y_axis_l1071_107171

/-- A line in 2D space defined by two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The y-axis, represented as a vertical line with x-coordinate 0 -/
def yAxis : Line := { point1 := (0, 0), point2 := (0, 1) }

/-- Function to determine if a point lies on a given line -/
def pointOnLine (l : Line) (p : ℝ × ℝ) : Prop :=
  let (x1, y1) := l.point1
  let (x2, y2) := l.point2
  let (x, y) := p
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

/-- Function to determine if a point lies on the y-axis -/
def pointOnYAxis (p : ℝ × ℝ) : Prop :=
  p.1 = 0

/-- The main theorem to be proved -/
theorem line_intersects_y_axis :
  let l : Line := { point1 := (2, 3), point2 := (6, -9) }
  pointOnLine l (0, 9) ∧ pointOnYAxis (0, 9) := by sorry

end line_intersects_y_axis_l1071_107171


namespace sqrt_meaningful_range_l1071_107178

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) → x ≥ 2 := by
  sorry

end sqrt_meaningful_range_l1071_107178


namespace charlie_fruit_bags_l1071_107100

theorem charlie_fruit_bags : 0.17 + 0.17 + 0.33 = 0.67 := by sorry

end charlie_fruit_bags_l1071_107100


namespace probability_at_least_one_strike_l1071_107105

theorem probability_at_least_one_strike (p : ℝ) (h : p = 2/5) :
  1 - (1 - p)^2 = 16/25 := by sorry

end probability_at_least_one_strike_l1071_107105


namespace geometric_arithmetic_sequence_ratio_l1071_107113

theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (a 1 + a 3 = 2 * (2 * a 2)) →  -- arithmetic sequence condition
  q = 2 - Real.sqrt 3 ∨ q = 2 + Real.sqrt 3 := by
sorry

end geometric_arithmetic_sequence_ratio_l1071_107113


namespace balanced_distribution_exists_l1071_107118

-- Define the weights of each letter
def weight (letter : Char) : ℕ :=
  match letter with
  | 'O' => 300
  | 'B' => 300
  | 'M' => 200
  | 'E' => 200
  | 'P' => 100
  | _ => 0

-- Define the initial setup
def initial_left : List Char := ['M', 'B']
def initial_right : List Char := ['P', 'E']
def top : Char := 'P'

-- Define the remaining letters to be placed
def remaining_letters : List Char := ['O', 'O', 'B', 'B', 'M', 'M', 'E', 'E', 'P', 'P']

-- Function to calculate the total weight of a list of letters
def total_weight (letters : List Char) : ℕ :=
  letters.map weight |>.sum

-- Theorem stating that a balanced distribution exists
theorem balanced_distribution_exists :
  ∃ (left right : List Char),
    left.length + right.length = remaining_letters.length ∧
    (left ++ initial_left).toFinset ∪ (right ++ initial_right).toFinset ∪ {top} = remaining_letters.toFinset ∪ initial_left.toFinset ∪ initial_right.toFinset ∪ {top} ∧
    total_weight (left ++ initial_left) = total_weight (right ++ initial_right) :=
  sorry

end balanced_distribution_exists_l1071_107118


namespace gcd_78_36_l1071_107195

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := by
  sorry

end gcd_78_36_l1071_107195


namespace sufficient_questions_sufficient_questions_10n_l1071_107164

/-- Represents the origin of a scientist -/
inductive Origin
| Piripocs
| Nekeresd

/-- Represents a scientist at the congress -/
structure Scientist where
  origin : Origin

/-- Represents the congress of scientists -/
structure Congress where
  n : ℕ
  scientists : Fin n → Scientist
  more_piripocs : ∃ (p : ℕ), 2 * p > n ∧ (∀ i : Fin n, (scientists i).origin = Origin.Piripocs → i.val < p)

/-- Function to ask a question about a scientist's origin -/
def ask_question (c : Congress) (asker : Fin c.n) (about : Fin c.n) : Origin :=
  match (c.scientists asker).origin with
  | Origin.Piripocs => (c.scientists about).origin
  | Origin.Nekeresd => sorry  -- This can be either true or false

/-- Theorem stating that n^2 / 2 questions are sufficient -/
theorem sufficient_questions (c : Congress) :
  ∃ (strategy : (Fin c.n → Fin c.n → Origin) → Fin c.n → Origin),
    (∀ f : Fin c.n → Fin c.n → Origin, 
      (∀ i j : Fin c.n, f i j = ask_question c i j) → 
      (∀ i : Fin c.n, strategy f i = (c.scientists i).origin)) ∧
    (∃ m : ℕ, 2 * m ≤ c.n * c.n ∧ 
      ∀ f : Fin c.n → Fin c.n → Origin, 
        (∃ s : Finset (Fin c.n × Fin c.n), s.card ≤ m ∧ 
          ∀ i j : Fin c.n, f i j = ask_question c i j → (i, j) ∈ s)) :=
sorry

/-- Theorem stating that 10n questions are also sufficient -/
theorem sufficient_questions_10n (c : Congress) :
  ∃ (strategy : (Fin c.n → Fin c.n → Origin) → Fin c.n → Origin),
    (∀ f : Fin c.n → Fin c.n → Origin, 
      (∀ i j : Fin c.n, f i j = ask_question c i j) → 
      (∀ i : Fin c.n, strategy f i = (c.scientists i).origin)) ∧
    (∃ s : Finset (Fin c.n × Fin c.n), s.card ≤ 10 * c.n ∧ 
      ∀ f : Fin c.n → Fin c.n → Origin, 
        (∀ i j : Fin c.n, f i j = ask_question c i j → (i, j) ∈ s)) :=
sorry

end sufficient_questions_sufficient_questions_10n_l1071_107164


namespace angle_ABC_is_30_degrees_l1071_107189

theorem angle_ABC_is_30_degrees (BA BC : ℝ × ℝ) : 
  BA = (1/2, Real.sqrt 3/2) → 
  BC = (Real.sqrt 3/2, 1/2) → 
  Real.arccos ((BA.1 * BC.1 + BA.2 * BC.2) / (Real.sqrt (BA.1^2 + BA.2^2) * Real.sqrt (BC.1^2 + BC.2^2))) = π/6 := by
  sorry

end angle_ABC_is_30_degrees_l1071_107189


namespace problem_1_problem_2_l1071_107167

-- Problem 1
theorem problem_1 (a b : ℝ) (h1 : a > b) (h2 : a^2 + b^2 = 13) (h3 : a * b = 6) :
  a - b = 1 := by sorry

-- Problem 2
theorem problem_2 (a b c : ℝ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : a^2 + b^2 = c^2)  -- Pythagorean theorem for right triangle
  (h3 : a^2 + b^2 + 25 = 6*a + 8*b) :
  a + b + c = 12 ∨ a + b + c = 7 + Real.sqrt 7 := by sorry

end problem_1_problem_2_l1071_107167


namespace monkey_climb_time_l1071_107121

/-- A monkey climbing a tree problem -/
theorem monkey_climb_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) 
  (h1 : tree_height = 20)
  (h2 : hop_distance = 3)
  (h3 : slip_distance = 2) :
  ∃ (time : ℕ), time = 17 ∧ 
  time * (hop_distance - slip_distance) + hop_distance ≥ tree_height :=
by
  sorry

end monkey_climb_time_l1071_107121


namespace unique_solution_l1071_107166

def a : Fin 3 → ℝ := ![2, 2, 2]
def b : Fin 3 → ℝ := ![3, -2, 1]
def c : Fin 3 → ℝ := ![3, 3, -4]

def orthogonal (u v : Fin 3 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) + (u 2) * (v 2) = 0

theorem unique_solution :
  orthogonal a b ∧ orthogonal b c ∧ orthogonal a c →
  ∃! (p q r : ℝ), ∀ i : Fin 3,
    (![3, -1, 8] i) = p * (a i) + q * (b i) + r * (c i) ∧
    p = 5/3 ∧ q = 0 ∧ r = -10/17 := by
  sorry

end unique_solution_l1071_107166


namespace john_money_left_l1071_107110

/-- The amount of money John has left after buying pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let drink_cost := q
  let small_pizza_cost := q
  let large_pizza_cost := 4 * q
  let total_cost := 4 * drink_cost + 2 * small_pizza_cost + large_pizza_cost
  50 - total_cost

/-- Theorem stating that John will have 50 - 10q dollars left -/
theorem john_money_left (q : ℝ) : money_left q = 50 - 10 * q := by
  sorry

end john_money_left_l1071_107110


namespace inequality_range_l1071_107158

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 := by
  sorry

end inequality_range_l1071_107158


namespace rice_seedling_stats_l1071_107143

def dataset : List Nat := [25, 26, 27, 26, 27, 28, 29, 26, 29]

def mode (l : List Nat) : Nat := sorry

def median (l : List Nat) : Nat := sorry

theorem rice_seedling_stats :
  mode dataset = 26 ∧ median dataset = 27 := by sorry

end rice_seedling_stats_l1071_107143


namespace unique_assignment_l1071_107184

-- Define the type for tables
inductive Table
| T1 | T2 | T3 | T4

-- Define the type for students
inductive Student
| Albert | Bogdan | Vadim | Denis

-- Define a function to represent the assignment of tables to students
def assignment : Student → Table
| Student.Albert => Table.T4
| Student.Bogdan => Table.T2
| Student.Vadim => Table.T1
| Student.Denis => Table.T3

-- Define a predicate for table intersection
def intersects (t1 t2 : Table) : Prop := sorry

-- Albert and Bogdan colored some cells
axiom albert_bogdan_colored : ∀ (t : Table), t ≠ Table.T1 → intersects (assignment Student.Albert) t ∨ intersects (assignment Student.Bogdan) t

-- Vadim's table doesn't intersect with Albert's or Bogdan's
axiom vadim_condition : ¬(intersects (assignment Student.Vadim) (assignment Student.Albert)) ∧ 
                        ¬(intersects (assignment Student.Vadim) (assignment Student.Bogdan))

-- Denis's table doesn't intersect with Bogdan's or Vadim's
axiom denis_condition : ¬(intersects (assignment Student.Denis) (assignment Student.Bogdan)) ∧ 
                        ¬(intersects (assignment Student.Denis) (assignment Student.Vadim))

-- Theorem stating that the given assignment is the only valid solution
theorem unique_assignment : 
  ∀ (f : Student → Table), 
    (∀ (s1 s2 : Student), s1 ≠ s2 → f s1 ≠ f s2) →
    (∀ (t : Table), t ≠ Table.T1 → intersects (f Student.Albert) t ∨ intersects (f Student.Bogdan) t) →
    (¬(intersects (f Student.Vadim) (f Student.Albert)) ∧ ¬(intersects (f Student.Vadim) (f Student.Bogdan))) →
    (¬(intersects (f Student.Denis) (f Student.Bogdan)) ∧ ¬(intersects (f Student.Denis) (f Student.Vadim))) →
    f = assignment := by sorry

end unique_assignment_l1071_107184
