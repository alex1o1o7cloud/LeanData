import Mathlib

namespace one_integral_root_l1289_128906

theorem one_integral_root :
  ∃! (x : ℤ), x - 9 / (x + 4 : ℚ) = 2 - 9 / (x + 4 : ℚ) :=
by sorry

end one_integral_root_l1289_128906


namespace complex_equation_solution_l1289_128995

theorem complex_equation_solution (x y : ℝ) : 
  (2 * x - y + 1 : ℂ) + (y - 2 : ℂ) * I = 0 → y = 2 := by
  sorry

end complex_equation_solution_l1289_128995


namespace fourth_hexagon_dots_l1289_128942

/-- Calculates the number of dots in the nth layer of the hexagonal pattern. -/
def layerDots (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n % 2 = 0 then 7 * (n - 1)
  else 7 * n

/-- Calculates the total number of dots in the nth hexagon of the sequence. -/
def totalDots (n : ℕ) : ℕ :=
  (List.range n).map layerDots |> List.sum

/-- The fourth hexagon in the sequence contains 50 dots. -/
theorem fourth_hexagon_dots : totalDots 4 = 50 := by
  sorry

end fourth_hexagon_dots_l1289_128942


namespace towel_bleaching_l1289_128903

theorem towel_bleaching (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let L' := L * (1 - x)
  let B' := B * 0.85
  L' * B' = (L * B) * 0.595
  →
  x = 0.3
  := by sorry

end towel_bleaching_l1289_128903


namespace pizza_toppings_count_l1289_128934

theorem pizza_toppings_count (n : ℕ) (h : n = 8) : 
  (n) + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end pizza_toppings_count_l1289_128934


namespace weekly_pie_sales_l1289_128996

/-- The number of pies sold daily by the restaurant -/
def daily_pies : ℕ := 8

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pies sold in a week -/
def total_pies : ℕ := daily_pies * days_in_week

theorem weekly_pie_sales : total_pies = 56 := by
  sorry

end weekly_pie_sales_l1289_128996


namespace f_monotone_decreasing_f_min_max_on_interval_l1289_128908

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- Theorem for monotonically decreasing intervals
theorem f_monotone_decreasing (a : ℝ) :
  ∀ x, (x < -1 ∨ x > 3) → (∀ y, y > x → f a y < f a x) :=
sorry

-- Theorem for minimum and maximum values when a = -2
theorem f_min_max_on_interval :
  let a := -2
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f a x ≤ f a y) ∧
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f a y ≤ f a x) ∧
  (∀ x ∈ Set.Icc (-2) 2, f a x ≥ -7) ∧
  (∀ x ∈ Set.Icc (-2) 2, f a x ≤ 20) ∧
  (∃ x ∈ Set.Icc (-2) 2, f a x = -7) ∧
  (∃ x ∈ Set.Icc (-2) 2, f a x = 20) :=
sorry

end f_monotone_decreasing_f_min_max_on_interval_l1289_128908


namespace batsman_average_theorem_l1289_128914

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an additional inning -/
def newAverage (stats : BatsmanStats) (newInningScore : ℕ) : ℚ :=
  (stats.totalRuns + newInningScore) / (stats.innings + 1)

/-- Theorem: If a batsman's average increases by 5 after scoring 110 in the 11th inning, 
    then the new average is 60 -/
theorem batsman_average_theorem (stats : BatsmanStats) 
  (h1 : stats.innings = 10)
  (h2 : newAverage stats 110 = stats.average + 5) :
  newAverage stats 110 = 60 := by
  sorry

#check batsman_average_theorem

end batsman_average_theorem_l1289_128914


namespace not_heart_zero_sum_property_l1289_128969

def heart (x y : ℝ) : ℝ := |x + y|

theorem not_heart_zero_sum_property : ¬ ∀ x y : ℝ, (heart x 0 + heart 0 y = heart x y) := by
  sorry

end not_heart_zero_sum_property_l1289_128969


namespace no_prime_arrangement_with_natural_expression_l1289_128955

theorem no_prime_arrangement_with_natural_expression :
  ¬ ∃ (p : ℕ → ℕ),
    (∀ n, Prime (p n)) ∧
    (∀ q : ℕ, Prime q → ∃ n, p n = q) ∧
    (∀ i : ℕ, ∃ k : ℕ, (p i * p (i + 1) - p (i + 2)^2) / (p i + p (i + 1)) = k) :=
by sorry

end no_prime_arrangement_with_natural_expression_l1289_128955


namespace oranges_picked_theorem_l1289_128911

/-- The total number of oranges picked over three days --/
def total_oranges (monday : ℕ) (tuesday_multiplier : ℕ) (wednesday : ℕ) : ℕ :=
  monday + tuesday_multiplier * monday + wednesday

/-- Theorem: Given the conditions, the total number of oranges picked is 470 --/
theorem oranges_picked_theorem (monday : ℕ) (tuesday_multiplier : ℕ) (wednesday : ℕ)
  (h1 : monday = 100)
  (h2 : tuesday_multiplier = 3)
  (h3 : wednesday = 70) :
  total_oranges monday tuesday_multiplier wednesday = 470 := by
  sorry

end oranges_picked_theorem_l1289_128911


namespace square_point_B_coordinates_l1289_128997

/-- A square in a 2D plane -/
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Predicate to check if a line is parallel to the x-axis -/
def parallelToXAxis (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) : Prop :=
  p1.2 = p2.2

theorem square_point_B_coordinates :
  ∀ (s : Square),
    s.A = (1, -2) →
    s.C = (4, 1) →
    parallelToXAxis s.A s.B →
    s.B = (4, -2) := by
  sorry

end square_point_B_coordinates_l1289_128997


namespace simplify_sqrt_neg_five_squared_l1289_128901

theorem simplify_sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by
  sorry

end simplify_sqrt_neg_five_squared_l1289_128901


namespace decimal_numbers_less_than_one_infinite_l1289_128943

theorem decimal_numbers_less_than_one_infinite :
  Set.Infinite {x : ℝ | x < 1 ∧ ∃ (n : ℕ), x = ↑n / (10 ^ n)} :=
sorry

end decimal_numbers_less_than_one_infinite_l1289_128943


namespace popped_kernel_probability_l1289_128992

/-- Given a bag of popping corn with white and blue kernels, calculate the probability
    that a randomly selected kernel that popped was white. -/
theorem popped_kernel_probability (total_kernels : ℝ) (h_total_pos : 0 < total_kernels) : 
  let white_ratio : ℝ := 3/4
  let blue_ratio : ℝ := 1/4
  let white_pop_prob : ℝ := 3/5
  let blue_pop_prob : ℝ := 3/4
  let white_kernels := white_ratio * total_kernels
  let blue_kernels := blue_ratio * total_kernels
  let popped_white := white_pop_prob * white_kernels
  let popped_blue := blue_pop_prob * blue_kernels
  let total_popped := popped_white + popped_blue
  (popped_white / total_popped) = 12/13 :=
by sorry

end popped_kernel_probability_l1289_128992


namespace expression_value_l1289_128944

theorem expression_value
  (a b x y : ℝ)
  (m : ℤ)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = -1) :
  2023 * (a + b) + 3 * |m| - 2 * x * y = 1 := by
sorry

end expression_value_l1289_128944


namespace overlapping_area_of_strips_l1289_128973

theorem overlapping_area_of_strips (total_length width : ℝ) 
  (left_length right_length : ℝ) (left_only_area right_only_area : ℝ) :
  total_length = 16 →
  left_length = 9 →
  right_length = 7 →
  left_length + right_length = total_length →
  left_only_area = 27 →
  right_only_area = 18 →
  ∃ (overlap_area : ℝ),
    overlap_area = (left_length * width - left_only_area) ∧
    overlap_area = (right_length * width - right_only_area) ∧
    overlap_area = 13.5 :=
by sorry

end overlapping_area_of_strips_l1289_128973


namespace union_of_A_and_B_l1289_128975

def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 4} := by
  sorry

end union_of_A_and_B_l1289_128975


namespace repeating_decimal_to_fraction_l1289_128917

/-- 
Given a repeating decimal 0.6̄13̄ (where 13 repeats infinitely after 6),
prove that it is equal to the fraction 362/495.
-/
theorem repeating_decimal_to_fraction : 
  (6/10 : ℚ) + (13/99 : ℚ) = 362/495 := by sorry

end repeating_decimal_to_fraction_l1289_128917


namespace wire_ratio_square_octagon_l1289_128954

/-- The ratio of wire lengths for equal-area square and octagon -/
theorem wire_ratio_square_octagon (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a / 4)^2 = (1 + Real.sqrt 2) * (b / 8)^2 → a / b = Real.sqrt (2 * (1 + Real.sqrt 2)) / 2 := by
  sorry

#check wire_ratio_square_octagon

end wire_ratio_square_octagon_l1289_128954


namespace smallest_x_value_l1289_128987

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (240 + x)) :
  x ≥ 4 :=
sorry

end smallest_x_value_l1289_128987


namespace smallest_opposite_l1289_128991

theorem smallest_opposite (a b c d : ℝ) (ha : a = -1) (hb : b = 0) (hc : c = Real.sqrt 5) (hd : d = -1/3) :
  min (-a) (min (-b) (min (-c) (-d))) = -c :=
by sorry

end smallest_opposite_l1289_128991


namespace line_trig_identity_l1289_128923

/-- Given a line with direction vector (-1, 2) and inclination angle α, 
    prove that sin(2α) - cos²(α) - 1 = -2 -/
theorem line_trig_identity (α : Real) (h : Real.tan α = -2) : 
  Real.sin (2 * α) - Real.cos α ^ 2 - 1 = -2 := by
  sorry

end line_trig_identity_l1289_128923


namespace triangle_ratio_l1289_128900

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  b * (Real.cos C) + c * (Real.cos B) = 2 * b →
  a / b = 2 := by
sorry


end triangle_ratio_l1289_128900


namespace inequality_abc_l1289_128977

theorem inequality_abc (a b c : ℝ) (ha : a = Real.log 2.1) (hb : b = Real.exp 0.1) (hc : c = 1.1) :
  a < c ∧ c < b := by
  sorry

end inequality_abc_l1289_128977


namespace work_done_stretching_spring_l1289_128964

/-- Work done by stretching a spring -/
theorem work_done_stretching_spring
  (force : ℝ) (compression : ℝ) (stretch : ℝ)
  (hf : force = 10)
  (hc : compression = 0.1)
  (hs : stretch = 0.06)
  : (1/2) * (force / compression) * stretch^2 = 0.18 := by
  sorry

end work_done_stretching_spring_l1289_128964


namespace purely_imaginary_m_eq_3_second_quadrant_m_range_l1289_128989

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := (m^2 - m - 6 : ℝ) + (m^2 + 5*m + 6 : ℝ) * Complex.I

/-- Theorem: If z is purely imaginary, then m = 3 -/
theorem purely_imaginary_m_eq_3 :
  (∀ m : ℝ, z m = Complex.I * Complex.im (z m)) → (∃ m : ℝ, m = 3) :=
sorry

/-- Theorem: If z is in the second quadrant, then -2 < m < 3 -/
theorem second_quadrant_m_range :
  (∀ m : ℝ, Complex.re (z m) < 0 ∧ Complex.im (z m) > 0) → (∀ m : ℝ, -2 < m ∧ m < 3) :=
sorry

end purely_imaginary_m_eq_3_second_quadrant_m_range_l1289_128989


namespace inheritance_calculation_l1289_128947

theorem inheritance_calculation (x : ℝ) : 
  x > 0 →
  (0.25 * x + 0.15 * (x - 0.25 * x) = 18000) →
  x = 50000 := by
sorry

end inheritance_calculation_l1289_128947


namespace extremum_at_negative_three_l1289_128967

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 5*x^2 + a*x

-- State the theorem
theorem extremum_at_negative_three (a : ℝ) : 
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, x ≠ -3 ∧ |x + 3| < ε → f a x ≤ f a (-3)) ∨
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, x ≠ -3 ∧ |x + 3| < ε → f a x ≥ f a (-3)) →
  a = 3 := by
sorry


end extremum_at_negative_three_l1289_128967


namespace problem_solution_l1289_128960

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) :
  (∀ a : ℝ, a < 1/2 → 1/x + 1/y ≥ |a + 2| - |a - 1|) ∧
  x^2 + 2*y^2 ≥ 8/3 := by sorry

end problem_solution_l1289_128960


namespace total_fish_count_l1289_128965

/-- The number of tuna in the sea -/
def num_tuna : ℕ := 5

/-- The number of spearfish in the sea -/
def num_spearfish : ℕ := 2

/-- The total number of fish in the sea -/
def total_fish : ℕ := num_tuna + num_spearfish

theorem total_fish_count : total_fish = 7 := by sorry

end total_fish_count_l1289_128965


namespace chess_tournament_wins_l1289_128993

theorem chess_tournament_wins (total_games : ℕ) (total_points : ℚ)
  (h1 : total_games = 20)
  (h2 : total_points = 12.5) :
  ∃ (wins losses draws : ℕ),
    wins + losses + draws = total_games ∧
    wins - losses = 5 ∧
    wins + draws / 2 = total_points := by
  sorry

end chess_tournament_wins_l1289_128993


namespace barium_chloride_weight_l1289_128957

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of chlorine in g/mol -/
def atomic_weight_Cl : ℝ := 35.45

/-- The number of moles of barium chloride -/
def moles_BaCl2 : ℝ := 4

/-- The molecular weight of barium chloride in g/mol -/
def molecular_weight_BaCl2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Cl

/-- The total weight of barium chloride in grams -/
def total_weight_BaCl2 : ℝ := moles_BaCl2 * molecular_weight_BaCl2

theorem barium_chloride_weight :
  total_weight_BaCl2 = 832.92 := by sorry

end barium_chloride_weight_l1289_128957


namespace positive_numbers_inequalities_l1289_128962

theorem positive_numbers_inequalities 
  (a b c : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_pos_c : c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end positive_numbers_inequalities_l1289_128962


namespace arithmetic_calculation_l1289_128972

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 30 / 3 = 61 := by
  sorry

end arithmetic_calculation_l1289_128972


namespace point_in_first_quadrant_l1289_128904

/-- Given a point M(2,a) on the graph of y = k/x where k > 0, prove that the coordinates of M are both positive. -/
theorem point_in_first_quadrant (k a : ℝ) (h1 : k > 0) (h2 : a = k / 2) : 2 > 0 ∧ a > 0 := by
  sorry

end point_in_first_quadrant_l1289_128904


namespace sewing_time_proof_l1289_128968

/-- The time it takes to sew one dress -/
def time_per_dress (num_dresses : ℕ) (weekly_sewing_time : ℕ) (total_weeks : ℕ) : ℕ :=
  (weekly_sewing_time * total_weeks) / num_dresses

/-- Theorem stating that the time to sew one dress is 12 hours -/
theorem sewing_time_proof (num_dresses : ℕ) (weekly_sewing_time : ℕ) (total_weeks : ℕ) 
  (h1 : num_dresses = 5)
  (h2 : weekly_sewing_time = 4)
  (h3 : total_weeks = 15) :
  time_per_dress num_dresses weekly_sewing_time total_weeks = 12 := by
  sorry

end sewing_time_proof_l1289_128968


namespace least_months_to_triple_debt_l1289_128950

theorem least_months_to_triple_debt (interest_rate : ℝ) (n : ℕ) : 
  interest_rate = 0.03 →
  n = 37 →
  (∀ m : ℕ, m < n → (1 + interest_rate)^m ≤ 3) ∧
  (1 + interest_rate)^n > 3 :=
sorry

end least_months_to_triple_debt_l1289_128950


namespace gcd_lcm_examples_l1289_128966

theorem gcd_lcm_examples : 
  (Nat.gcd 17 51 = 17) ∧ 
  (Nat.lcm 17 51 = 51) ∧ 
  (Nat.gcd 6 8 = 2) ∧ 
  (Nat.lcm 8 9 = 72) := by
  sorry

end gcd_lcm_examples_l1289_128966


namespace change_calculation_l1289_128930

def bracelet_price : ℚ := 15
def necklace_price : ℚ := 10
def mug_price : ℚ := 20
def keychain_price : ℚ := 5

def bracelet_quantity : ℕ := 3
def necklace_quantity : ℕ := 2
def mug_quantity : ℕ := 1
def keychain_quantity : ℕ := 4

def discount_rate : ℚ := 12 / 100
def payment : ℚ := 100

def total_before_discount : ℚ :=
  bracelet_price * bracelet_quantity +
  necklace_price * necklace_quantity +
  mug_price * mug_quantity +
  keychain_price * keychain_quantity

def discount_amount : ℚ := total_before_discount * discount_rate
def final_amount : ℚ := total_before_discount - discount_amount

theorem change_calculation :
  payment - final_amount = 760 / 100 := by sorry

end change_calculation_l1289_128930


namespace multiply_fractions_l1289_128929

theorem multiply_fractions : (7 : ℚ) * (1 / 17) * 34 = 14 := by sorry

end multiply_fractions_l1289_128929


namespace complement_of_M_wrt_U_l1289_128941

def U : Finset Int := {1, -2, 3, -4, 5, -6}
def M : Finset Int := {1, -2, 3, -4}

theorem complement_of_M_wrt_U :
  U \ M = {5, -6} := by sorry

end complement_of_M_wrt_U_l1289_128941


namespace arithmetic_sequence_problem_l1289_128938

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  first_term : ℝ
  common_difference : ℝ

/-- Get the nth term of an arithmetic sequence. -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1 : ℝ) * seq.common_difference

/-- Our specific arithmetic sequence with given conditions. -/
def our_sequence : ArithmeticSequence :=
  { first_term := 0,  -- We don't know the first term yet, so we use a placeholder
    common_difference := 0 }  -- We don't know the common difference yet, so we use a placeholder

theorem arithmetic_sequence_problem :
  our_sequence.nthTerm 3 = 10 ∧
  our_sequence.nthTerm 20 = 65 →
  our_sequence.nthTerm 32 = 103.8235294118 := by
  sorry


end arithmetic_sequence_problem_l1289_128938


namespace smallest_divisible_by_1_to_9_l1289_128920

def is_divisible_by_range (n : ℕ) (a b : ℕ) : Prop :=
  ∀ i : ℕ, a ≤ i → i ≤ b → n % i = 0

theorem smallest_divisible_by_1_to_9 :
  ∃ (n : ℕ), n > 0 ∧ is_divisible_by_range n 1 9 ∧
  ∀ (m : ℕ), m > 0 → is_divisible_by_range m 1 9 → n ≤ m :=
by
  use 2520
  sorry

end smallest_divisible_by_1_to_9_l1289_128920


namespace cos_B_value_triangle_area_l1289_128990

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_condition_1 (t : Triangle) : Prop :=
  (Real.sin t.B) ^ 2 = (Real.sin t.A) * (Real.sin t.C) ∧ t.a = Real.sqrt 2 * t.b

def satisfies_condition_2 (t : Triangle) : Prop :=
  Real.cos t.B = 3 / 4 ∧ t.a = 2

-- Define the theorems to prove
theorem cos_B_value (t : Triangle) (h : satisfies_condition_1 t) :
  Real.cos t.B = 3 / 4 := by sorry

theorem triangle_area (t : Triangle) (h : satisfies_condition_2 t) :
  let area := 1 / 2 * t.a * t.c * Real.sin t.B
  area = Real.sqrt 7 / 4 ∨ area = Real.sqrt 7 := by sorry

end cos_B_value_triangle_area_l1289_128990


namespace function_value_at_negative_l1289_128902

theorem function_value_at_negative (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = x^5 + x^3 + 1) →
  f m = 10 →
  f (-m) = -8 := by
sorry

end function_value_at_negative_l1289_128902


namespace vector_sum_proof_l1289_128918

theorem vector_sum_proof :
  let v₁ : Fin 3 → ℝ := ![3, -2, 7]
  let v₂ : Fin 3 → ℝ := ![-1, 5, -3]
  v₁ + v₂ = ![2, 3, 4] := by
  sorry

end vector_sum_proof_l1289_128918


namespace right_triangle_hypotenuse_l1289_128978

theorem right_triangle_hypotenuse (x y : ℝ) :
  x > 0 ∧ y > 0 →
  (1/3) * π * x * y^2 = 800 * π →
  (1/3) * π * y * x^2 = 1920 * π →
  Real.sqrt (x^2 + y^2) = 26 := by
  sorry

end right_triangle_hypotenuse_l1289_128978


namespace max_value_sum_products_l1289_128984

theorem max_value_sum_products (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_constraint : a + b + c + d = 200) : 
  ∃ (max : ℝ), max = 10000 ∧ ∀ (x y z w : ℝ), 
    0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ 0 ≤ w ∧ x + y + z + w = 200 →
    x * y + y * z + z * w ≤ max :=
by sorry

end max_value_sum_products_l1289_128984


namespace sequence_decreasing_two_equal_max_terms_l1289_128916

-- Define the sequence aₙ
def a (k : ℝ) (n : ℕ) : ℝ := n * k^n

-- Proposition ②
theorem sequence_decreasing (k : ℝ) (h1 : 0 < k) (h2 : k < 1/2) :
  ∀ n : ℕ, n > 0 → a k (n + 1) < a k n :=
sorry

-- Proposition ④
theorem two_equal_max_terms (k : ℝ) (h : ∃ m : ℕ, m > 0 ∧ k / (1 - k) = m) :
  ∃ n : ℕ, n > 0 ∧ a k n = a k (n + 1) ∧ ∀ m : ℕ, m > 0 → a k m ≤ a k n :=
sorry

end sequence_decreasing_two_equal_max_terms_l1289_128916


namespace milly_fold_count_l1289_128927

/-- Represents the croissant-making process with given time constraints. -/
structure CroissantProcess where
  fold_time : ℕ         -- Time to fold dough once (in minutes)
  rest_time : ℕ         -- Time to rest dough once (in minutes)
  mix_time : ℕ          -- Time to mix ingredients (in minutes)
  bake_time : ℕ         -- Time to bake (in minutes)
  total_time : ℕ        -- Total time for the whole process (in minutes)

/-- Calculates the number of times the dough needs to be folded. -/
def fold_count (process : CroissantProcess) : ℕ :=
  ((process.total_time - process.mix_time - process.bake_time) / 
   (process.fold_time + process.rest_time))

/-- Theorem stating that for the given process, the dough needs to be folded 4 times. -/
theorem milly_fold_count : 
  let process : CroissantProcess := {
    fold_time := 5,
    rest_time := 75,
    mix_time := 10,
    bake_time := 30,
    total_time := 6 * 60  -- 6 hours in minutes
  }
  fold_count process = 4 := by
  sorry

end milly_fold_count_l1289_128927


namespace total_people_present_l1289_128976

/-- Represents the number of associate professors -/
def associate_profs : ℕ := sorry

/-- Represents the number of assistant professors -/
def assistant_profs : ℕ := sorry

/-- Total number of pencils brought to the meeting -/
def total_pencils : ℕ := 10

/-- Total number of charts brought to the meeting -/
def total_charts : ℕ := 14

/-- Theorem stating the total number of people present at the meeting -/
theorem total_people_present : associate_profs + assistant_profs = 8 :=
  sorry

end total_people_present_l1289_128976


namespace polynomial_simplification_l1289_128970

theorem polynomial_simplification (x : ℝ) : (3 * x^2 - 4 * x + 5) - (2 * x^2 - 6 * x - 8) = x^2 + 2 * x + 13 := by
  sorry

end polynomial_simplification_l1289_128970


namespace juice_dispenser_capacity_l1289_128921

/-- A cylindrical juice dispenser with capacity x cups -/
structure JuiceDispenser where
  capacity : ℝ
  cylindrical : Bool

/-- Theorem: A cylindrical juice dispenser that contains 60 cups when 48% full has a total capacity of 125 cups -/
theorem juice_dispenser_capacity (d : JuiceDispenser) 
  (h_cylindrical : d.cylindrical = true) 
  (h_partial : 0.48 * d.capacity = 60) : 
  d.capacity = 125 := by
  sorry

end juice_dispenser_capacity_l1289_128921


namespace complex_square_one_plus_i_l1289_128926

theorem complex_square_one_plus_i (i : ℂ) : i * i = -1 → (1 + i)^2 = 2*i := by
  sorry

end complex_square_one_plus_i_l1289_128926


namespace unique_valid_number_l1289_128924

def is_valid_product (a b : Nat) : Prop :=
  ∃ (x y : Nat), x < 10 ∧ y < 10 ∧ a * 10 + b = x * y

def is_valid_number (n : Nat) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧
  (∀ i : Fin 9, (n / 10^i.val % 10) ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧
  (∀ i : Fin 9, ∀ j : Fin 9, i ≠ j → (n / 10^i.val % 10) ≠ (n / 10^j.val % 10)) ∧
  (∀ i : Fin 8, is_valid_product (n / 10^(i+1).val % 10) (n / 10^i.val % 10))

theorem unique_valid_number : 
  ∃! n : Nat, is_valid_number n ∧ n = 728163549 :=
sorry

end unique_valid_number_l1289_128924


namespace unique_negative_solution_implies_positive_a_l1289_128922

theorem unique_negative_solution_implies_positive_a (a : ℝ) : 
  (∃! x : ℝ, (abs x = 2 * x + a) ∧ (x < 0)) → a > 0 := by
sorry

end unique_negative_solution_implies_positive_a_l1289_128922


namespace infinitely_many_primes_in_differences_l1289_128945

/-- Definition of the sequence a_n -/
def a (k : ℕ) : ℕ → ℕ
  | n => if n < k then 0  -- arbitrary value for n < k
         else if n = k then 2 * k
         else if Nat.gcd (a k (n-1)) n = 1 then a k (n-1) + 1
         else 2 * n

/-- The theorem statement -/
theorem infinitely_many_primes_in_differences (k : ℕ) (h : k ≥ 3) :
  ∀ M : ℕ, ∃ n > k, ∃ p : ℕ, p.Prime ∧ p > M ∧ p ∣ (a k n - a k (n-1)) :=
sorry

end infinitely_many_primes_in_differences_l1289_128945


namespace arithmetic_sequence_property_l1289_128994

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  S : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem stating the property of the arithmetic sequence -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
  (h1 : seq.a 2 + seq.S 3 = 4)
  (h2 : seq.a 3 + seq.S 5 = 12) :
  seq.a 4 + seq.S 7 = 24 := by
  sorry

end arithmetic_sequence_property_l1289_128994


namespace unique_divisibility_condition_l1289_128946

theorem unique_divisibility_condition (n : ℕ) : n > 1 → (
  (∃! a : ℕ, 0 < a ∧ a ≤ Nat.factorial n ∧ (Nat.factorial n ∣ a^n + 1)) ↔ n = 2
) := by sorry

end unique_divisibility_condition_l1289_128946


namespace forint_bill_solution_exists_l1289_128963

def is_valid_solution (x y z : ℕ) : Prop :=
  10 * x + 5 * y + z = 682 ∧ x = y + z

def is_one_of_solutions (x y z : ℕ) : Prop :=
  (x = 58 ∧ y = 11 ∧ z = 47) ∨
  (x = 54 ∧ y = 22 ∧ z = 32) ∨
  (x = 50 ∧ y = 33 ∧ z = 17) ∨
  (x = 46 ∧ y = 44 ∧ z = 2)

theorem forint_bill_solution_exists :
  ∃ x y z : ℕ, is_valid_solution x y z ∧ is_one_of_solutions x y z := by
  sorry

end forint_bill_solution_exists_l1289_128963


namespace largest_integer_with_remainder_l1289_128999

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
by sorry

end largest_integer_with_remainder_l1289_128999


namespace sum_of_root_products_l1289_128915

theorem sum_of_root_products (p q r : ℂ) : 
  (4 * p^3 - 2 * p^2 + 13 * p - 9 = 0) →
  (4 * q^3 - 2 * q^2 + 13 * q - 9 = 0) →
  (4 * r^3 - 2 * r^2 + 13 * r - 9 = 0) →
  p * q + p * r + q * r = 13 / 4 := by
  sorry

end sum_of_root_products_l1289_128915


namespace problem_solution_l1289_128974

theorem problem_solution :
  (∀ x : ℝ, (x + 1) * (x - 3) > (x + 2) * (x - 4)) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * (x + y) = 36 →
    x * y ≤ 81 ∧
    (x * y = 81 ↔ x = 9 ∧ y = 9)) :=
by sorry


end problem_solution_l1289_128974


namespace subset_intersection_bound_l1289_128931

/-- Given a set S of n elements and a family of b subsets of S, each containing k elements,
    with the property that any two subsets intersect in at most one element,
    the number of subsets b is bounded above by ⌊(n/k)⌊(n-1)/(k-1)⌋⌋. -/
theorem subset_intersection_bound (n k b : ℕ) (S : Finset (Fin n)) (B : Fin b → Finset (Fin n)) 
  (h1 : ∀ i, (B i).card = k)
  (h2 : ∀ i j, i < j → (B i ∩ B j).card ≤ 1)
  (h3 : k > 0)
  (h4 : n > 0)
  : b ≤ ⌊(n : ℝ) / k * ⌊(n - 1 : ℝ) / (k - 1)⌋⌋ :=
sorry

end subset_intersection_bound_l1289_128931


namespace consecutive_blue_gumballs_probability_l1289_128925

theorem consecutive_blue_gumballs_probability :
  let p_pink : ℝ := 0.1428571428571428
  let p_blue : ℝ := 1 - p_pink
  p_blue * p_blue = 0.7346938775510203 := by
  sorry

end consecutive_blue_gumballs_probability_l1289_128925


namespace count_numerators_T_l1289_128953

/-- The set of rational numbers with repeating decimal expansion 0.overline(ab) -/
def T : Set ℚ :=
  {r | 0 < r ∧ r < 1 ∧ ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ r = (10 * a + b : ℚ) / 99}

/-- The number of different numerators required to express all elements of T in lowest terms -/
def num_different_numerators : ℕ := 53

/-- Theorem stating that the number of different numerators for T is 53 -/
theorem count_numerators_T : num_different_numerators = 53 := by
  sorry

end count_numerators_T_l1289_128953


namespace whitewashing_cost_calculation_l1289_128971

/-- Calculates the cost of white washing a room with given specifications. -/
def whitewashingCost (roomLength roomWidth roomHeight : ℝ)
                     (doorCount doorLength doorWidth : ℝ)
                     (windowCount windowLength windowWidth : ℝ)
                     (costPerSqFt additionalPaintPercentage : ℝ) : ℝ :=
  let wallArea := 2 * (roomLength + roomWidth) * roomHeight
  let doorArea := doorCount * doorLength * doorWidth
  let windowArea := windowCount * windowLength * windowWidth
  let paintableArea := wallArea - doorArea - windowArea
  let totalPaintArea := paintableArea * (1 + additionalPaintPercentage)
  totalPaintArea * costPerSqFt

/-- Theorem stating the cost of white washing the room with given specifications. -/
theorem whitewashing_cost_calculation :
  whitewashingCost 25 15 12 2 6 3 5 4 3 7 0.1 = 6652.8 := by
  sorry

end whitewashing_cost_calculation_l1289_128971


namespace fraction_calculation_l1289_128928

theorem fraction_calculation : 
  (8 / 17) / (7 / 5) + (5 / 7) * (9 / 17) = 5 / 7 := by sorry

end fraction_calculation_l1289_128928


namespace opposite_numbers_sum_l1289_128949

theorem opposite_numbers_sum (a b : ℝ) : a + b = 0 → 3*a + 3*b - 2 = -2 := by
  sorry

end opposite_numbers_sum_l1289_128949


namespace base_13_conversion_l1289_128986

/-- Represents a digit in base 13 -/
inductive Base13Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a Base13Digit to its numerical value -/
def base13DigitToNat (d : Base13Digit) : ℕ :=
  match d with
  | Base13Digit.D0 => 0
  | Base13Digit.D1 => 1
  | Base13Digit.D2 => 2
  | Base13Digit.D3 => 3
  | Base13Digit.D4 => 4
  | Base13Digit.D5 => 5
  | Base13Digit.D6 => 6
  | Base13Digit.D7 => 7
  | Base13Digit.D8 => 8
  | Base13Digit.D9 => 9
  | Base13Digit.A => 10
  | Base13Digit.B => 11
  | Base13Digit.C => 12

/-- Converts a two-digit number in base 13 to its decimal (base 10) equivalent -/
def base13ToDecimal (d1 d2 : Base13Digit) : ℕ :=
  13 * (base13DigitToNat d1) + (base13DigitToNat d2)

theorem base_13_conversion :
  base13ToDecimal Base13Digit.C Base13Digit.D1 = 157 := by
  sorry

end base_13_conversion_l1289_128986


namespace f_of_g_of_3_l1289_128933

/-- Given functions f and g, prove that f(2 + g(3)) = 44 -/
theorem f_of_g_of_3 (f g : ℝ → ℝ) 
    (hf : ∀ x, f x = 3 * x - 4)
    (hg : ∀ x, g x = x^2 + 2 * x - 1) : 
  f (2 + g 3) = 44 := by
  sorry

end f_of_g_of_3_l1289_128933


namespace computer_arrangements_l1289_128982

theorem computer_arrangements : Nat.factorial 8 = 40320 := by
  sorry

end computer_arrangements_l1289_128982


namespace cars_sold_last_three_days_l1289_128937

/-- Represents the number of cars sold by a salesman over 6 days -/
structure CarSales where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ
  day5 : ℕ
  day6 : ℕ

/-- Calculates the mean of car sales over 6 days -/
def meanSales (sales : CarSales) : ℚ :=
  (sales.day1 + sales.day2 + sales.day3 + sales.day4 + sales.day5 + sales.day6 : ℚ) / 6

/-- Theorem stating the number of cars sold in the last three days -/
theorem cars_sold_last_three_days (sales : CarSales) 
  (h1 : sales.day1 = 8)
  (h2 : sales.day2 = 3)
  (h3 : sales.day3 = 10)
  (h_mean : meanSales sales = 5.5) :
  sales.day4 + sales.day5 + sales.day6 = 12 := by
  sorry

end cars_sold_last_three_days_l1289_128937


namespace solve_proportion_l1289_128980

theorem solve_proportion (y : ℝ) (h : 9 / y^2 = y / 81) : y = 9 := by
  sorry

end solve_proportion_l1289_128980


namespace solution_of_equation_l1289_128951

theorem solution_of_equation (x : ℝ) :
  x ≠ 3 →
  ((2 - x) / (x - 3) = 0) ↔ (x = 2) :=
by sorry

end solution_of_equation_l1289_128951


namespace unequal_outcome_probability_l1289_128907

theorem unequal_outcome_probability : 
  let n : ℕ := 12  -- number of grandchildren
  let p : ℝ := 1/2 -- probability of each gender
  let total_outcomes : ℕ := 2^n -- total number of possible gender combinations
  let equal_outcomes : ℕ := n.choose (n/2) -- number of combinations with equal boys and girls
  
  (total_outcomes - equal_outcomes : ℝ) / total_outcomes = 793/1024 := by
  sorry

end unequal_outcome_probability_l1289_128907


namespace happy_island_parrots_l1289_128939

theorem happy_island_parrots (total_birds : ℕ) (yellow_fraction : ℚ) (red_parrots : ℕ) :
  total_birds = 120 →
  yellow_fraction = 2/3 →
  red_parrots = total_birds - (yellow_fraction * total_birds).floor →
  red_parrots = 40 := by
sorry

end happy_island_parrots_l1289_128939


namespace circle_properties_l1289_128935

/-- Given a circle with polar equation ρ²-4√2ρcos(θ-π/4)+6=0, prove its properties -/
theorem circle_properties (ρ θ : ℝ) :
  ρ^2 - 4 * Real.sqrt 2 * ρ * Real.cos (θ - π/4) + 6 = 0 →
  ∃ (x y : ℝ),
    -- Standard equation
    x^2 + y^2 - 4*x - 4*y + 6 = 0 ∧
    -- Parametric equations
    x = 2 + Real.sqrt 2 * Real.cos θ ∧
    y = 2 + Real.sqrt 2 * Real.sin θ ∧
    -- Maximum and minimum values of x⋅y
    (∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 4*y' + 6 = 0 → x'*y' ≤ 9) ∧
    (∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 4*y' + 6 = 0 → x'*y' ≥ 1) ∧
    (∃ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 4*y' + 6 = 0 ∧ x'*y' = 9) ∧
    (∃ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 4*y' + 6 = 0 ∧ x'*y' = 1) :=
by sorry

end circle_properties_l1289_128935


namespace remainder_of_sum_first_six_primes_div_seventh_prime_l1289_128998

-- Define the first seven prime numbers
def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]

-- Define the sum of the first six primes
def sum_first_six_primes : Nat := (first_seven_primes.take 6).sum

-- Define the seventh prime
def seventh_prime : Nat := first_seven_primes[6]

-- Theorem statement
theorem remainder_of_sum_first_six_primes_div_seventh_prime :
  sum_first_six_primes % seventh_prime = 7 := by
  sorry

end remainder_of_sum_first_six_primes_div_seventh_prime_l1289_128998


namespace circus_crowns_l1289_128956

theorem circus_crowns (total_feathers : ℕ) (feathers_per_crown : ℕ) (h1 : total_feathers = 6538) (h2 : feathers_per_crown = 7) :
  total_feathers / feathers_per_crown = 934 := by
  sorry

end circus_crowns_l1289_128956


namespace isosceles_triangle_vertex_angle_l1289_128948

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define two angles, as the third is determined by these two
  base_angle : Real
  vertex_angle : Real
  -- Condition that the sum of angles in a triangle is 180°
  angle_sum : base_angle * 2 + vertex_angle = 180

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (triangle : IsoscelesTriangle) 
  (h : triangle.base_angle = 80 ∨ triangle.vertex_angle = 80) :
  triangle.vertex_angle = 80 ∨ triangle.vertex_angle = 20 := by
sorry


end isosceles_triangle_vertex_angle_l1289_128948


namespace additional_sugar_needed_l1289_128912

/-- The amount of additional sugar needed for a cake -/
theorem additional_sugar_needed (total_required sugar_available : ℕ) : 
  total_required = 450 → sugar_available = 287 → total_required - sugar_available = 163 := by
  sorry

end additional_sugar_needed_l1289_128912


namespace ellipse_equation_eccentricity_range_l1289_128959

noncomputable section

-- Define the ellipse parameters
def m : ℝ := 1  -- We know m = 1 from the solution, but we keep it as a parameter

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the focus
def focus : ℝ × ℝ := (m, 0)

-- Define the directrices
def left_directrix (x : ℝ) : Prop := x = -m - 1
def right_directrix (x : ℝ) : Prop := x = m + 1

-- Define the line y = x
def diagonal_line (x y : ℝ) : Prop := y = x

-- Define points A and B
def point_A : ℝ × ℝ := (-m - 1, -m - 1)
def point_B : ℝ × ℝ := (m + 1, m + 1)

-- Define vectors AF and FB
def vector_AF : ℝ × ℝ := (2*m + 1, m + 1)
def vector_FB : ℝ × ℝ := (1, m + 1)

-- Define dot product of AF and FB
def dot_product_AF_FB : ℝ := (2*m + 1) * 1 + (m + 1) * (m + 1)

-- Define eccentricity
def eccentricity : ℝ := 1 / Real.sqrt (1 + 1/m)

-- Theorem 1: Prove the equation of the ellipse
theorem ellipse_equation : 
  ∀ x y : ℝ, ellipse x y ↔ x^2 / 2 + y^2 = 1 :=
sorry

-- Theorem 2: Prove the range of eccentricity
theorem eccentricity_range :
  dot_product_AF_FB < 7 → 0 < eccentricity ∧ eccentricity < Real.sqrt 2 / 2 :=
sorry

end ellipse_equation_eccentricity_range_l1289_128959


namespace simplify_nested_expression_l1289_128919

theorem simplify_nested_expression (x : ℝ) : 1 - (2 - (1 + (2 - (3 - x)))) = x - 1 := by
  sorry

end simplify_nested_expression_l1289_128919


namespace arithmetic_equation_l1289_128983

theorem arithmetic_equation : (26.3 * 12 * 20) / 3 + 125 = 2229 := by
  sorry

end arithmetic_equation_l1289_128983


namespace smallest_number_proof_l1289_128961

def digits : List Nat := [0, 2, 4, 6, 8, 9]

def is_valid_number (n : Nat) : Prop :=
  let digits_used := n.digits 10
  (digits_used.toFinset = digits.toFinset) ∧ 
  (digits_used.length = digits.length) ∧
  (n ≥ 100000)

theorem smallest_number_proof :
  (is_valid_number 204689) ∧ 
  (∀ m : Nat, is_valid_number m → m ≥ 204689) :=
sorry

end smallest_number_proof_l1289_128961


namespace binomial_coefficient_16_4_l1289_128909

theorem binomial_coefficient_16_4 : Nat.choose 16 4 = 1820 := by
  sorry

end binomial_coefficient_16_4_l1289_128909


namespace expression_evaluation_l1289_128981

theorem expression_evaluation (x y : ℚ) (hx : x = 1/3) (hy : y = -1/2) :
  (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 1/2 := by
  sorry

end expression_evaluation_l1289_128981


namespace root_in_interval_l1289_128952

-- Define the function f(x) = x³ - 4
def f (x : ℝ) : ℝ := x^3 - 4

-- State the theorem
theorem root_in_interval :
  (f 1 < 0) → (f 2 > 0) → ∃ x : ℝ, x ∈ (Set.Ioo 1 2) ∧ f x = 0 :=
by sorry

end root_in_interval_l1289_128952


namespace geometric_sequence_property_l1289_128910

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 2)^2 - 34*(a 2) + 64 = 0 →
  (a 6)^2 - 34*(a 6) + 64 = 0 →
  a 4 = 8 :=
by sorry

end geometric_sequence_property_l1289_128910


namespace quadratic_equation_solution_l1289_128940

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 + 4*x = 5 ↔ x = 1 ∨ x = -5 := by sorry

end quadratic_equation_solution_l1289_128940


namespace optimal_sequence_l1289_128958

theorem optimal_sequence (x₁ x₂ x₃ x₄ : ℝ) 
  (h_order : 0 ≤ x₄ ∧ x₄ ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1) 
  (h_eq : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + (x₃ - x₄)^2 + x₄^2 = 1/5) :
  x₁ = 4/5 ∧ x₂ = 3/5 ∧ x₃ = 2/5 ∧ x₄ = 1/5 := by
sorry

end optimal_sequence_l1289_128958


namespace novel_pages_per_hour_l1289_128913

-- Define the reading time in hours
def total_reading_time : ℚ := 1/6 * 24

-- Define the reading time for each type of book
def reading_time_per_type : ℚ := total_reading_time / 3

-- Define the pages read per hour for comic books and graphic novels
def comic_pages_per_hour : ℕ := 45
def graphic_pages_per_hour : ℕ := 30

-- Define the total pages read
def total_pages_read : ℕ := 128

-- Theorem to prove
theorem novel_pages_per_hour : 
  ∃ (n : ℕ), 
    (n : ℚ) * reading_time_per_type + 
    (comic_pages_per_hour : ℚ) * reading_time_per_type + 
    (graphic_pages_per_hour : ℚ) * reading_time_per_type = total_pages_read ∧ 
    n = 21 := by
  sorry

end novel_pages_per_hour_l1289_128913


namespace inequality_condition_l1289_128979

theorem inequality_condition (a b : ℝ) (h1 : a * b ≠ 0) :
  (a < b ∧ b < 0) → (1 / a^2 > 1 / b^2) ∧
  ¬(∀ a b : ℝ, a * b ≠ 0 → (1 / a^2 > 1 / b^2) → (a < b ∧ b < 0)) :=
by sorry

end inequality_condition_l1289_128979


namespace regression_line_change_l1289_128905

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the y-value for a given x-value on the regression line -/
def RegressionLine.predict (line : RegressionLine) (x : ℝ) : ℝ :=
  line.intercept + line.slope * x

/-- Theorem: For the regression line y = 2 - 1.5x, 
    when x increases by 1 unit, y decreases by 1.5 units -/
theorem regression_line_change 
  (line : RegressionLine) 
  (h1 : line.intercept = 2) 
  (h2 : line.slope = -1.5) 
  (x : ℝ) : 
  line.predict (x + 1) = line.predict x - 1.5 := by
  sorry


end regression_line_change_l1289_128905


namespace eighth_term_geometric_sequence_l1289_128936

theorem eighth_term_geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) :
  a₁ = 27 ∧ r = 1/3 ∧ n = 8 →
  a₁ * r^(n - 1) = 1/81 := by
  sorry

end eighth_term_geometric_sequence_l1289_128936


namespace smallest_distance_between_complex_numbers_l1289_128932

theorem smallest_distance_between_complex_numbers
  (z w : ℂ)
  (hz : Complex.abs (z + 2 + 4 * Complex.I) = 2)
  (hw : Complex.abs (w - 6 - 7 * Complex.I) = 4) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 185 - 6 ∧
    ∀ (z' w' : ℂ), Complex.abs (z' + 2 + 4 * Complex.I) = 2 →
      Complex.abs (w' - 6 - 7 * Complex.I) = 4 →
        Complex.abs (z' - w') ≥ min_dist :=
by sorry

end smallest_distance_between_complex_numbers_l1289_128932


namespace rocket_launch_l1289_128985

/-- Rocket launch problem -/
theorem rocket_launch (a : ℝ) (g : ℝ) (t : ℝ) (h_object : ℝ) : 
  a = 20 → g = 10 → t = 40 → h_object = 45000 →
  let v₀ : ℝ := a * t
  let h₀ : ℝ := (1/2) * a * t^2
  let t_max : ℝ := v₀ / g
  let h_max : ℝ := h₀ + v₀ * t_max - (1/2) * g * t_max^2
  h_max = 48000 ∧ h_max > h_object :=
by sorry

end rocket_launch_l1289_128985


namespace average_sequence_problem_l1289_128988

theorem average_sequence_problem (a b c d e : ℝ) : 
  a = 8 ∧ 
  d = 26 ∧
  b = (a + c) / 2 ∧
  c = (b + d) / 2 ∧
  d = (c + e) / 2 
  → e = 32 := by sorry

end average_sequence_problem_l1289_128988
