import Mathlib

namespace container_capacity_l609_60904

theorem container_capacity (C : ℝ) : 
  C > 0 → 
  0.30 * C + 36 = 0.75 * C → 
  C = 80 := by
sorry

end container_capacity_l609_60904


namespace division_problem_l609_60910

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 176 →
  quotient = 12 →
  remainder = 8 →
  dividend = divisor * quotient + remainder →
  divisor = 14 := by
sorry

end division_problem_l609_60910


namespace skew_parameter_calculation_l609_60912

/-- Dilation matrix -/
def D (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

/-- Skew transformation matrix -/
def S (a : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, a; 0, 1]

/-- The problem statement -/
theorem skew_parameter_calculation (k : ℝ) (a : ℝ) (h1 : k > 0) :
  S a * D k = !![10, 5; 0, 10] →
  a = 1/2 := by
sorry

end skew_parameter_calculation_l609_60912


namespace speeding_ticket_percentage_l609_60932

/-- The percentage of motorists who exceed the speed limit -/
def exceed_limit : ℝ := 16.666666666666664

/-- The percentage of speeding motorists who do not receive tickets -/
def no_ticket_rate : ℝ := 40

/-- The percentage of motorists who receive speeding tickets -/
def receive_ticket : ℝ := 10

theorem speeding_ticket_percentage :
  receive_ticket = exceed_limit * (1 - no_ticket_rate / 100) := by sorry

end speeding_ticket_percentage_l609_60932


namespace combination_permutation_problem_l609_60999

-- Define the combination function
def C (n k : ℕ) : ℕ := n.choose k

-- Define the permutation function
def A (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

-- State the theorem
theorem combination_permutation_problem (n : ℕ) :
  C n 2 * A 2 2 = 42 → n.factorial / (3 * (n - 3).factorial) = 35 := by
  sorry

end combination_permutation_problem_l609_60999


namespace large_rectangle_perimeter_l609_60948

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle given its dimensions -/
def perimeter (d : Dimensions) : ℕ := 2 * (d.length + d.width)

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the tiling pattern of the large rectangle -/
structure TilingPattern where
  inner : Dimensions
  redTiles : ℕ

theorem large_rectangle_perimeter 
  (pattern : TilingPattern)
  (h1 : pattern.redTiles = 2900) :
  ∃ (large : Dimensions), 
    area large = area pattern.inner + 2900 + 2 * area { length := pattern.inner.length + 20, width := pattern.inner.width + 20 } ∧ 
    perimeter large = 350 := by
  sorry


end large_rectangle_perimeter_l609_60948


namespace compare_negative_fractions_l609_60967

theorem compare_negative_fractions : -2/3 > -3/4 := by sorry

end compare_negative_fractions_l609_60967


namespace num_intersection_values_correct_l609_60957

/-- The number of different possible values for the count of intersection points
    formed by 5 distinct lines on a plane. -/
def num_intersection_values : ℕ := 9

/-- The set of possible values for the count of intersection points
    formed by 5 distinct lines on a plane. -/
def possible_intersection_values : Finset ℕ :=
  {0, 1, 4, 5, 6, 7, 8, 9, 10}

/-- Theorem stating that the number of different possible values for the count
    of intersection points formed by 5 distinct lines on a plane is correct. -/
theorem num_intersection_values_correct :
    num_intersection_values = Finset.card possible_intersection_values := by
  sorry

end num_intersection_values_correct_l609_60957


namespace functional_equation_solution_l609_60958

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) + f (x - y) = x^2 + y^2) →
  (∀ x : ℝ, f x = x^2 / 2) :=
by sorry

end functional_equation_solution_l609_60958


namespace phone_contract_cost_l609_60995

/-- The total cost of buying a phone with a contract -/
def total_cost (phone_price : ℕ) (monthly_fee : ℕ) (contract_months : ℕ) : ℕ :=
  phone_price + monthly_fee * contract_months

/-- Theorem: The total cost of buying 1 phone with a 4-month contract is $30 -/
theorem phone_contract_cost :
  total_cost 2 7 4 = 30 := by
  sorry

end phone_contract_cost_l609_60995


namespace leaves_collected_first_day_l609_60987

/-- Represents the number of leaves collected by Bronson -/
def total_leaves : ℕ := 25

/-- Represents the number of leaves collected on the second day -/
def second_day_leaves : ℕ := 13

/-- Represents the percentage of brown leaves -/
def brown_percent : ℚ := 1/5

/-- Represents the percentage of green leaves -/
def green_percent : ℚ := 1/5

/-- Represents the number of yellow leaves -/
def yellow_leaves : ℕ := 15

/-- Theorem stating the number of leaves collected on the first day -/
theorem leaves_collected_first_day : 
  total_leaves - second_day_leaves = 12 :=
sorry

end leaves_collected_first_day_l609_60987


namespace find_a_l609_60926

theorem find_a (a x y : ℝ) (h1 : a^(3*x - 1) * 3^(4*y - 3) = 49^x * 27^y) (h2 : x + y = 4) : a = 7 := by
  sorry

end find_a_l609_60926


namespace problem_1_l609_60920

theorem problem_1 : Real.sqrt 8 - 4 * Real.sin (45 * π / 180) + (1/3)^0 = 1 := by
  sorry

end problem_1_l609_60920


namespace winnie_keeps_lollipops_l609_60955

/-- The number of cherry lollipops Winnie has -/
def cherry : ℕ := 45

/-- The number of wintergreen lollipops Winnie has -/
def wintergreen : ℕ := 116

/-- The number of grape lollipops Winnie has -/
def grape : ℕ := 4

/-- The number of shrimp cocktail lollipops Winnie has -/
def shrimp : ℕ := 229

/-- The number of Winnie's friends -/
def friends : ℕ := 11

/-- The total number of lollipops Winnie has -/
def total : ℕ := cherry + wintergreen + grape + shrimp

/-- The theorem stating how many lollipops Winnie keeps for herself -/
theorem winnie_keeps_lollipops : total % friends = 9 := by
  sorry

end winnie_keeps_lollipops_l609_60955


namespace calculation_proof_l609_60938

theorem calculation_proof : (((2207 - 2024) ^ 2 * 4) : ℚ) / 144 = 930.25 := by
  sorry

end calculation_proof_l609_60938


namespace complex_modulus_product_l609_60973

theorem complex_modulus_product : 
  Complex.abs ((7 - 4 * Complex.I) * (5 + 12 * Complex.I)) = 13 * Real.sqrt 65 := by
  sorry

end complex_modulus_product_l609_60973


namespace geometric_arithmetic_inequality_l609_60917

/-- A geometric sequence with positive integer terms -/
def geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (r : ℚ), r > 0 ∧ ∀ n, a (n + 1) = a n * ⌊r⌋

/-- An arithmetic sequence -/
def arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

/-- The main theorem -/
theorem geometric_arithmetic_inequality
  (a : ℕ → ℕ) (b : ℕ → ℤ)
  (h_geo : geometric_sequence a)
  (h_arith : arithmetic_sequence b)
  (h_eq : a 6 = b 7) :
  a 3 + a 9 ≥ b 4 + b 10 := by
  sorry

end geometric_arithmetic_inequality_l609_60917


namespace total_money_calculation_l609_60928

theorem total_money_calculation (total_notes : ℕ) 
  (denominations : Fin 3 → ℕ) 
  (h1 : total_notes = 75) 
  (h2 : denominations 0 = 1 ∧ denominations 1 = 5 ∧ denominations 2 = 10) : 
  (total_notes / 3) * (denominations 0 + denominations 1 + denominations 2) = 400 :=
by
  sorry

#check total_money_calculation

end total_money_calculation_l609_60928


namespace second_half_speed_l609_60949

/-- Proves that given a journey of 300 km completed in 11 hours, where the first half of the distance
    is traveled at 30 kmph, the speed for the second half of the journey is 25 kmph. -/
theorem second_half_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (first_half_speed : ℝ) 
  (h1 : total_distance = 300)
  (h2 : total_time = 11)
  (h3 : first_half_speed = 30)
  : ∃ second_half_speed : ℝ, 
    second_half_speed = 25 ∧ 
    total_distance / 2 / first_half_speed + total_distance / 2 / second_half_speed = total_time :=
by sorry

end second_half_speed_l609_60949


namespace isosceles_triangle_perimeter_l609_60950

/-- An isosceles triangle with side lengths 2 and 4 has perimeter 10 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ), 
  a = 4 → b = 4 → c = 2 →  -- Two sides are 4, one side is 2
  a = b →  -- It's an isosceles triangle
  a + b + c = 10  -- The perimeter is 10
  := by sorry

end isosceles_triangle_perimeter_l609_60950


namespace mathematics_letter_probability_l609_60947

theorem mathematics_letter_probability : 
  let alphabet_size : ℕ := 26
  let unique_letters_in_mathematics : ℕ := 8
  let probability : ℚ := unique_letters_in_mathematics / alphabet_size
  probability = 4 / 13 := by
sorry

end mathematics_letter_probability_l609_60947


namespace basketball_score_proof_l609_60942

/-- 
Given a basketball team's scoring pattern:
- 4 games with 10t points each
- g games with 20 points each
- Average score of 28 points per game
Prove that g = 16
-/
theorem basketball_score_proof (t : ℕ) (g : ℕ) : 
  (40 * t + 20 * g) / (4 + g) = 28 → g = 16 := by
  sorry

end basketball_score_proof_l609_60942


namespace parabola_p_value_l609_60974

/-- Given a parabola with equation y^2 = 2px and axis of symmetry x = -1, prove that p = 2 -/
theorem parabola_p_value (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x) → 
  (∀ y : ℝ, y^2 = -2*p) → 
  p = 2 := by
sorry

end parabola_p_value_l609_60974


namespace jo_alan_sum_equal_l609_60946

def jo_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (x : ℕ) : ℕ :=
  5 * ((x + 2) / 5)

def alan_sum (n : ℕ) : ℕ :=
  List.sum (List.map round_to_nearest_five (List.range n))

theorem jo_alan_sum_equal :
  jo_sum 120 = alan_sum 120 :=
sorry

end jo_alan_sum_equal_l609_60946


namespace triangle_area_l609_60952

theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 3 * Real.sqrt 2) (h2 : b = 2 * Real.sqrt 3) (h3 : Real.cos C = 1/3) :
  (1/2) * a * b * Real.sin C = 4 * Real.sqrt 3 := by
  sorry

end triangle_area_l609_60952


namespace base3_product_theorem_l609_60905

/-- Converts a base 3 number to decimal --/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- Converts a decimal number to base 3 --/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- Multiplies two base 3 numbers --/
def multiplyBase3 (a b : List Nat) : List Nat :=
  decimalToBase3 (base3ToDecimal a * base3ToDecimal b)

theorem base3_product_theorem :
  multiplyBase3 [2, 0, 1] [2, 1] = [2, 0, 2, 1] := by sorry

end base3_product_theorem_l609_60905


namespace sum_possibilities_l609_60976

theorem sum_possibilities (a b c d : ℕ) : 
  0 < a ∧ a < 4 ∧ 
  0 < b ∧ b < 4 ∧ 
  0 < c ∧ c < 4 ∧ 
  0 < d ∧ d < 4 ∧ 
  b / c = 1 →
  4^a + 3^b + 2^c + 1^d = 10 ∨ 
  4^a + 3^b + 2^c + 1^d = 22 ∨ 
  4^a + 3^b + 2^c + 1^d = 70 :=
by sorry

end sum_possibilities_l609_60976


namespace remainder_theorem_l609_60983

theorem remainder_theorem (n : ℤ) (h : ∃ k : ℤ, n = 25 * k - 1) :
  (n^2 + 3*n + 5) % 25 = 3 := by
  sorry

end remainder_theorem_l609_60983


namespace ice_cream_cones_l609_60923

theorem ice_cream_cones (cost_per_cone : ℕ) (total_cost : ℕ) (h1 : cost_per_cone = 99) (h2 : total_cost = 198) :
  total_cost / cost_per_cone = 2 :=
by sorry

end ice_cream_cones_l609_60923


namespace quadratic_equation_roots_l609_60975

theorem quadratic_equation_roots (p q : ℝ) : 
  (∃ α β : ℝ, α ≠ β ∧ 
   α^2 + p*α + q = 0 ∧ 
   β^2 + p*β + q = 0 ∧ 
   ({α, β} : Set ℝ) ⊆ {1, 2, 3, 4} ∧ 
   ({α, β} : Set ℝ) ∩ {2, 4, 5, 6} = ∅) →
  p = -4 ∧ q = 3 := by
sorry

end quadratic_equation_roots_l609_60975


namespace orange_delivery_problem_l609_60930

def bag_weights : List ℕ := [22, 25, 28, 31, 34, 36, 38, 40, 45]

def total_weight : ℕ := bag_weights.sum

theorem orange_delivery_problem (weights_A B : ℕ) (weight_C : ℕ) :
  weights_A = 2 * weights_B →
  weights_A + weights_B + weight_C = total_weight →
  weight_C ∈ bag_weights →
  weight_C % 3 = 2 →
  weight_C = 38 := by
  sorry

end orange_delivery_problem_l609_60930


namespace finite_n_with_prime_factors_in_A_l609_60919

theorem finite_n_with_prime_factors_in_A (A : Finset Nat) (a : Nat) 
  (h_A : ∀ p ∈ A, Nat.Prime p) (h_a : a ≥ 2) :
  ∃ S : Finset Nat, ∀ n : Nat, (∀ p : Nat, p ∣ (a^n - 1) → p ∈ A) → n ∈ S :=
by sorry

end finite_n_with_prime_factors_in_A_l609_60919


namespace quadratic_equation_proof_l609_60915

theorem quadratic_equation_proof (m : ℝ) (h1 : m < 0) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x + m
  (∃ x : ℝ, f x = 0 ∧ x = -1) →
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) ∧  -- two distinct real roots
  m = -3 ∧                                  -- value of m
  (∃ x : ℝ, f x = 0 ∧ x = 3)                -- other root
:= by sorry

end quadratic_equation_proof_l609_60915


namespace f_range_implies_a_values_l609_60956

/-- The function f(x) defined as x^2 - 2ax + 2a + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2*a + 4

/-- The property that the range of f is [1, +∞) -/
def range_property (a : ℝ) : Prop :=
  ∀ x, f a x ≥ 1 ∧ ∀ y ≥ 1, ∃ x, f a x = y

/-- Theorem stating that the only values of a satisfying the conditions are -1 and 3 -/
theorem f_range_implies_a_values :
  ∀ a : ℝ, range_property a ↔ (a = -1 ∨ a = 3) :=
sorry

end f_range_implies_a_values_l609_60956


namespace vertical_strips_count_l609_60940

/-- Represents a grid rectangle with a hole -/
structure GridRectangleWithHole where
  outer_perimeter : ℕ
  hole_perimeter : ℕ
  horizontal_strips : ℕ

/-- Theorem: Given a grid rectangle with a hole, if cutting horizontally yields 20 strips,
    then cutting vertically yields 21 strips -/
theorem vertical_strips_count
  (rect : GridRectangleWithHole)
  (h_outer : rect.outer_perimeter = 50)
  (h_hole : rect.hole_perimeter = 32)
  (h_horizontal : rect.horizontal_strips = 20) :
  ∃ (vertical_strips : ℕ), vertical_strips = 21 :=
by
  sorry

end vertical_strips_count_l609_60940


namespace imaginary_unit_power_l609_60971

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2015 = -i := by
  sorry

end imaginary_unit_power_l609_60971


namespace white_balls_count_l609_60909

theorem white_balls_count (total : ℕ) (p_red p_black : ℚ) (h_total : total = 50)
  (h_red : p_red = 15/100) (h_black : p_black = 45/100) :
  (total : ℚ) * (1 - p_red - p_black) = 20 := by
  sorry

end white_balls_count_l609_60909


namespace tetris_single_ratio_is_eight_to_one_l609_60934

/-- The ratio of points for a tetris to points for a single line -/
def tetris_to_single_ratio (single_points tetris_points : ℕ) : ℚ :=
  tetris_points / single_points

/-- The total score given the number of singles, number of tetrises, points for a single, and points for a tetris -/
def total_score (num_singles num_tetrises single_points tetris_points : ℕ) : ℕ :=
  num_singles * single_points + num_tetrises * tetris_points

theorem tetris_single_ratio_is_eight_to_one :
  ∃ (tetris_points : ℕ),
    single_points = 1000 ∧
    num_singles = 6 ∧
    num_tetrises = 4 ∧
    total_score num_singles num_tetrises single_points tetris_points = 38000 ∧
    tetris_to_single_ratio single_points tetris_points = 8 := by
  sorry

end tetris_single_ratio_is_eight_to_one_l609_60934


namespace inequality_proof_l609_60964

theorem inequality_proof (a b c : ℝ) : 
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 2) / 2 := by
  sorry

end inequality_proof_l609_60964


namespace parallel_iff_a_eq_neg_one_l609_60980

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  line1 : ℝ × ℝ → Prop := fun (x, y) ↦ a * x + 2 * y + 2 = 0
  line2 : ℝ × ℝ → Prop := fun (x, y) ↦ x + (a - 1) * y + 1 = 0

/-- The lines are parallel -/
def areParallel (lines : TwoLines) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
    (∀ (x y : ℝ), lines.line1 (x, y) ↔ lines.line2 (k * x + lines.a, k * y + 2))

/-- The main theorem -/
theorem parallel_iff_a_eq_neg_one (lines : TwoLines) :
  areParallel lines ↔ lines.a = -1 :=
sorry

end parallel_iff_a_eq_neg_one_l609_60980


namespace gold_cube_value_scaling_l609_60903

-- Define the properties of the 4-inch gold cube
def gold_cube_4inch_value : ℝ := 500
def gold_cube_4inch_side : ℝ := 4

-- Define the side length of the 5-inch gold cube
def gold_cube_5inch_side : ℝ := 5

-- Function to calculate the volume of a cube
def cube_volume (side : ℝ) : ℝ := side ^ 3

-- Theorem statement
theorem gold_cube_value_scaling :
  let v4 := cube_volume gold_cube_4inch_side
  let v5 := cube_volume gold_cube_5inch_side
  let scale_factor := v5 / v4
  let scaled_value := gold_cube_4inch_value * scale_factor
  ⌊scaled_value + 0.5⌋ = 977 := by sorry

end gold_cube_value_scaling_l609_60903


namespace board_cut_theorem_l609_60933

theorem board_cut_theorem (total_length : ℝ) (x : ℝ) 
  (h1 : total_length = 120)
  (h2 : x = 1.5) : 
  let shorter_piece := total_length / (1 + (2 * x + 1/3))
  let longer_piece := shorter_piece * (2 * x + 1/3)
  longer_piece = 92 + 4/13 := by
  sorry

end board_cut_theorem_l609_60933


namespace prob_10_or_7_prob_below_7_l609_60991

/-- Probability of hitting the 10 ring -/
def P10 : ℝ := 0.21

/-- Probability of hitting the 9 ring -/
def P9 : ℝ := 0.23

/-- Probability of hitting the 8 ring -/
def P8 : ℝ := 0.25

/-- Probability of hitting the 7 ring -/
def P7 : ℝ := 0.28

/-- The probability of hitting either the 10 or 7 ring is 0.49 -/
theorem prob_10_or_7 : P10 + P7 = 0.49 := by sorry

/-- The probability of scoring below 7 rings is 0.03 -/
theorem prob_below_7 : 1 - (P10 + P9 + P8 + P7) = 0.03 := by sorry

end prob_10_or_7_prob_below_7_l609_60991


namespace inequality_solution_set_l609_60951

theorem inequality_solution_set 
  (h : ∀ x : ℝ, x^2 - 2*a*x + a > 0) :
  {t : ℝ | a^(t^2 + 2*t - 3) < 1} = {t : ℝ | t < -3 ∨ t > 1} :=
by sorry

end inequality_solution_set_l609_60951


namespace dave_initial_apps_l609_60978

/-- Represents the number of apps on Dave's phone at different stages -/
structure AppCount where
  initial : ℕ
  afterAdding : ℕ
  afterDeleting : ℕ
  final : ℕ

/-- Represents the number of apps added and deleted -/
structure AppChanges where
  added : ℕ
  deleted : ℕ

/-- The theorem stating Dave's initial app count based on the given conditions -/
theorem dave_initial_apps (ac : AppCount) (ch : AppChanges) : 
  ch.added = 89 ∧ 
  ac.afterDeleting = 24 ∧ 
  ch.added = ch.deleted + 3 ∧
  ac.afterAdding = ac.initial + ch.added ∧
  ac.afterDeleting = ac.afterAdding - ch.deleted ∧
  ac.final = ac.afterDeleting + (ch.added - ch.deleted) →
  ac.initial = 21 := by
  sorry


end dave_initial_apps_l609_60978


namespace tangent_line_at_x_1_l609_60989

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem stating that the tangent line equation is correct
theorem tangent_line_at_x_1 : 
  ∀ x y : ℝ, (y = f 1 + f' 1 * (x - 1)) ↔ (2*x - y + 1 = 0) :=
by sorry

-- Note: The proof is omitted as per the instructions

end tangent_line_at_x_1_l609_60989


namespace line_vector_at_zero_l609_60908

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_at_zero : 
  (∀ (t : ℝ), line_vector t = line_vector 0 + t • (line_vector 1 - line_vector 0)) →
  line_vector (-2) = (2, 4, 10) →
  line_vector 1 = (-1, -3, -5) →
  line_vector 0 = (0, -2/3, 0) := by sorry

end line_vector_at_zero_l609_60908


namespace coefficient_is_three_l609_60901

/-- The derivative function for our equation -/
noncomputable def derivative (q : ℝ) : ℝ := 3 * q - 3

/-- The second derivative of 6 -/
def second_derivative_of_six : ℝ := 210

/-- The coefficient of q in the equation q' = 3q - 3 -/
def coefficient : ℝ := 3

theorem coefficient_is_three : coefficient = 3 := by sorry

end coefficient_is_three_l609_60901


namespace alpha_squared_gt_beta_squared_l609_60943

theorem alpha_squared_gt_beta_squared 
  (α β : Real) 
  (h1 : α ∈ Set.Icc (-π/2) (π/2)) 
  (h2 : β ∈ Set.Icc (-π/2) (π/2)) 
  (h3 : α * Real.sin α - β * Real.sin β > 0) : 
  α^2 > β^2 := by
  sorry

end alpha_squared_gt_beta_squared_l609_60943


namespace geometric_sequence_property_l609_60924

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  a 3 * a 7 = 8 → a 5 = 2 * Real.sqrt 2 ∨ a 5 = -2 * Real.sqrt 2 := by
  sorry

end geometric_sequence_property_l609_60924


namespace number_line_mark_distance_not_always_1cm_l609_60961

/-- Represents a number line -/
structure NumberLine where
  origin : ℝ
  positive_direction : Bool
  unit_length : ℝ

/-- Properties of a number line -/
def valid_number_line (nl : NumberLine) : Prop :=
  nl.unit_length > 0

theorem number_line_mark_distance_not_always_1cm 
  (h1 : ∀ (x : ℝ), ∃ (nl : NumberLine), nl.origin = x ∧ valid_number_line nl)
  (h2 : ∀ (nl : NumberLine), valid_number_line nl → nl.positive_direction = true)
  (h3 : ∀ (l : ℝ), l > 0 → ∃ (nl : NumberLine), nl.unit_length = l ∧ valid_number_line nl) :
  ¬(∀ (nl : NumberLine), valid_number_line nl → nl.unit_length = 1) :=
sorry

end number_line_mark_distance_not_always_1cm_l609_60961


namespace inequality_solution_set_l609_60929

theorem inequality_solution_set (x : ℝ) :
  (2 < (1 / (x - 1)) ∧ (1 / (x - 1)) < 3 ∧ 0 < x - 1) ↔ (4/3 < x ∧ x < 3/2) := by
  sorry

end inequality_solution_set_l609_60929


namespace banana_arrangements_l609_60982

theorem banana_arrangements : 
  let total_letters : ℕ := 6
  let repeated_letter1_count : ℕ := 3
  let repeated_letter2_count : ℕ := 2
  let unique_letter_count : ℕ := 1
  (total_letters = repeated_letter1_count + repeated_letter2_count + unique_letter_count) →
  (Nat.factorial total_letters / (Nat.factorial repeated_letter1_count * Nat.factorial repeated_letter2_count) = 60) := by
  sorry

end banana_arrangements_l609_60982


namespace probability_is_one_over_432_l609_60907

/-- Represents a fair die with 6 faces -/
def Die := Fin 6

/-- Represents the outcome of tossing four dice -/
def FourDiceOutcome := (Die × Die × Die × Die)

/-- Checks if a sequence of four numbers forms an arithmetic progression with common difference 1 -/
def isArithmeticProgression (a b c d : ℕ) : Prop :=
  b - a = 1 ∧ c - b = 1 ∧ d - c = 1

/-- The set of all possible outcomes when tossing four fair dice -/
def allOutcomes : Finset FourDiceOutcome := sorry

/-- The set of favorable outcomes (forming an arithmetic progression) -/
def favorableOutcomes : Finset FourDiceOutcome := sorry

/-- The probability of getting an arithmetic progression when tossing four fair dice -/
def probabilityOfArithmeticProgression : ℚ :=
  (favorableOutcomes.card : ℚ) / (allOutcomes.card : ℚ)

/-- Theorem stating that the probability of getting an arithmetic progression is 1/432 -/
theorem probability_is_one_over_432 :
  probabilityOfArithmeticProgression = 1 / 432 := by sorry

end probability_is_one_over_432_l609_60907


namespace quadratic_inequality_problem_l609_60911

/-- Given that the solution set of ax^2 + (a-5)x - 2 > 0 is {x | -2 < x < -1/4},
    prove the following statements. -/
theorem quadratic_inequality_problem (a : ℝ) 
  (h : ∀ x, ax^2 + (a-5)*x - 2 > 0 ↔ -2 < x ∧ x < -1/4) :
  /- 1. a = -4 -/
  (a = -4) ∧ 
  /- 2. The solution set of 2x^2 + (2-a)x - a > 0 is (-∞, -2) ∪ (-1, ∞) -/
  (∀ x, 2*x^2 + (2-a)*x - a > 0 ↔ x < -2 ∨ x > -1) ∧
  /- 3. The range of b such that -ax^2 + bx + 3 ≥ 0 for all real x 
        is [-4√3, 4√3] -/
  (∀ b, (∀ x, -a*x^2 + b*x + 3 ≥ 0) ↔ -4*Real.sqrt 3 ≤ b ∧ b ≤ 4*Real.sqrt 3) :=
by sorry

end quadratic_inequality_problem_l609_60911


namespace input_is_input_statement_l609_60965

-- Define an enumeration for different types of statements
inductive StatementType
  | Print
  | Input
  | If
  | End

-- Define a function to classify statements
def classifyStatement (s : StatementType) : String :=
  match s with
  | StatementType.Print => "output"
  | StatementType.Input => "input"
  | StatementType.If => "conditional"
  | StatementType.End => "end"

-- Theorem to prove
theorem input_is_input_statement :
  classifyStatement StatementType.Input = "input" := by
  sorry

end input_is_input_statement_l609_60965


namespace system_solution_l609_60954

theorem system_solution (x y z : ℝ) 
  (eq1 : y + z = 17 - 2*x)
  (eq2 : x + z = 1 - 2*y)
  (eq3 : x + y = 8 - 2*z) :
  x + y + z = 6.5 := by
sorry

end system_solution_l609_60954


namespace least_eight_binary_digits_l609_60969

/-- The number of binary digits required to represent a positive integer -/
def binaryDigits (n : ℕ+) : ℕ :=
  (Nat.log2 n.val) + 1

/-- Theorem: 128 is the least positive integer that requires 8 binary digits -/
theorem least_eight_binary_digits :
  (∀ m : ℕ+, m < 128 → binaryDigits m < 8) ∧ binaryDigits 128 = 8 := by
  sorry

end least_eight_binary_digits_l609_60969


namespace line_l_and_symmetrical_line_l609_60959

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def line3 (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Define line l
def l (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define the symmetrical line
def symmetrical_l (x y : ℝ) : Prop := 2 * x + y - 2 = 0

theorem line_l_and_symmetrical_line : 
  (∀ x y : ℝ, line1 x y ∧ line2 x y → (x, y) = P) → 
  (∀ x y : ℝ, l x y → line3 (y + 2) (-x - 1)) →
  (∀ x y : ℝ, l x y ↔ 2 * x + y + 2 = 0) ∧
  (∀ x y : ℝ, symmetrical_l x y ↔ 2 * x + y - 2 = 0) := by sorry

end line_l_and_symmetrical_line_l609_60959


namespace removing_2013th_digit_increases_one_seventh_l609_60996

-- Define the decimal representation of 1/7
def one_seventh_decimal : ℚ := 1 / 7

-- Define the period of the repeating decimal
def period : ℕ := 6

-- Define the position of the digit to be removed
def removed_digit_position : ℕ := 2013

-- Define the function that removes the nth digit after the decimal point
def remove_nth_digit (q : ℚ) (n : ℕ) : ℚ := sorry

-- Theorem statement
theorem removing_2013th_digit_increases_one_seventh :
  remove_nth_digit one_seventh_decimal removed_digit_position > one_seventh_decimal := by
  sorry

end removing_2013th_digit_increases_one_seventh_l609_60996


namespace triangle_inequality_from_sum_product_l609_60935

theorem triangle_inequality_from_sum_product (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  c < a + b ∧ a < b + c ∧ b < c + a := by
sorry

end triangle_inequality_from_sum_product_l609_60935


namespace inequality_is_linear_one_var_l609_60966

/-- A linear inequality with one variable is an inequality of the form ax + b ≤ c or ax + b ≥ c,
    where a, b, and c are constants and x is a variable. -/
def is_linear_inequality_one_var (f : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
  ((∀ x, f x ↔ a * x + b ≤ c) ∨ (∀ x, f x ↔ a * x + b ≥ c))

/-- The inequality 2 - x ≤ 4 -/
def inequality (x : ℝ) : Prop := 2 - x ≤ 4

theorem inequality_is_linear_one_var : is_linear_inequality_one_var inequality := by
  sorry

end inequality_is_linear_one_var_l609_60966


namespace family_members_count_l609_60936

/-- Represents the number of family members -/
def n : ℕ := sorry

/-- The average age of family members in years -/
def average_age : ℕ := 29

/-- The present age of the youngest member in years -/
def youngest_age : ℕ := 5

/-- The average age of the remaining members at the time of birth of the youngest member in years -/
def average_age_at_birth : ℕ := 28

/-- The sum of ages of all family members -/
def sum_of_ages : ℕ := n * average_age

/-- The sum of ages of the remaining members at present -/
def sum_of_remaining_ages : ℕ := (n - 1) * (average_age_at_birth + youngest_age)

theorem family_members_count :
  sum_of_ages = sum_of_remaining_ages + youngest_age → n = 7 := by
  sorry

end family_members_count_l609_60936


namespace not_all_isosceles_congruent_l609_60979

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  is_isosceles : side1 = side2

/-- Congruence of triangles -/
def are_congruent (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.side1 = t2.side1 ∧ t1.side2 = t2.side2 ∧ t1.base = t2.base

/-- Theorem: Not all isosceles triangles are congruent -/
theorem not_all_isosceles_congruent : 
  ∃ t1 t2 : IsoscelesTriangle, ¬(are_congruent t1 t2) :=
sorry

end not_all_isosceles_congruent_l609_60979


namespace sum_of_selected_numbers_l609_60941

def numbers : List ℚ := [14/10, 9/10, 12/10, 5/10, 13/10]
def threshold : ℚ := 11/10

theorem sum_of_selected_numbers :
  (numbers.filter (λ x => x ≥ threshold)).sum = 39/10 := by
  sorry

end sum_of_selected_numbers_l609_60941


namespace range_of_x_no_solution_exists_l609_60998

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = a * b

-- Theorem for the first part of the problem
theorem range_of_x (a b : ℝ) (h : conditions a b) :
  (∀ x : ℝ, |x| + |x - 2| ≤ a + b) ↔ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 :=
sorry

-- Theorem for the second part of the problem
theorem no_solution_exists :
  ¬∃ a b : ℝ, conditions a b ∧ 4 * a + b = 8 :=
sorry

end range_of_x_no_solution_exists_l609_60998


namespace residue_11_1201_mod_19_l609_60931

theorem residue_11_1201_mod_19 :
  (11 : ℤ) ^ 1201 ≡ 1 [ZMOD 19] := by sorry

end residue_11_1201_mod_19_l609_60931


namespace solution_sets_equality_l609_60918

theorem solution_sets_equality (a b : ℝ) : 
  (∀ x : ℝ, |8*x + 9| < 7 ↔ a*x^2 + b*x > 2) → 
  (a = -4 ∧ b = -9) := by
sorry

end solution_sets_equality_l609_60918


namespace mark_and_carolyn_money_sum_l609_60988

theorem mark_and_carolyn_money_sum : (5 : ℚ) / 8 + (7 : ℚ) / 20 = 0.975 := by
  sorry

end mark_and_carolyn_money_sum_l609_60988


namespace coefficient_x5y2_is_90_l609_60913

/-- The coefficient of x^5y^2 in the expansion of (x^2 + 3x - y)^5 -/
def coefficient_x5y2 : ℕ :=
  let n : ℕ := 5
  let k : ℕ := 3
  let binomial_coeff : ℕ := n.choose k
  let x_coeff : ℕ := 9  -- Coefficient of x^5 in (x^2 + 3x)^3
  binomial_coeff * x_coeff

/-- The coefficient of x^5y^2 in the expansion of (x^2 + 3x - y)^5 is 90 -/
theorem coefficient_x5y2_is_90 : coefficient_x5y2 = 90 := by
  sorry

end coefficient_x5y2_is_90_l609_60913


namespace equalize_guppies_l609_60902

/-- Represents a fish tank -/
structure Tank where
  guppies : ℕ
  swordtails : ℕ
  angelfish : ℕ

/-- The problem setup -/
def problem : Prop :=
  let tankA : Tank := { guppies := 180, swordtails := 32, angelfish := 0 }
  let tankB : Tank := { guppies := 120, swordtails := 45, angelfish := 15 }
  let tankC : Tank := { guppies := 80, swordtails := 15, angelfish := 33 }
  let totalFish : ℕ := tankA.guppies + tankA.swordtails + tankA.angelfish +
                       tankB.guppies + tankB.swordtails + tankB.angelfish +
                       tankC.guppies + tankC.swordtails + tankC.angelfish

  totalFish = 520 ∧
  (tankA.guppies + tankB.guppies - (tankA.guppies + tankB.guppies) / 2) = 30

theorem equalize_guppies (tankA tankB : Tank) :
  tankA.guppies = 180 →
  tankB.guppies = 120 →
  (tankA.guppies + tankB.guppies - (tankA.guppies + tankB.guppies) / 2) = 30 :=
by sorry

end equalize_guppies_l609_60902


namespace total_fish_caught_l609_60977

def blaine_fish : ℕ := 5

def keith_fish (blaine : ℕ) : ℕ := 2 * blaine

theorem total_fish_caught : blaine_fish + keith_fish blaine_fish = 15 := by
  sorry

end total_fish_caught_l609_60977


namespace hyperbola_parameters_l609_60984

-- Define the hyperbola parameters
variable (a b : ℝ)

-- Define the conditions
def hyperbola_condition := a > 0 ∧ b > 0
def focus_condition := ∃ (y : ℝ), (6^2 : ℝ) = a^2 + b^2
def asymptote_condition := b / a = Real.sqrt 3

-- State the theorem
theorem hyperbola_parameters
  (h1 : hyperbola_condition a b)
  (h2 : focus_condition a b)
  (h3 : asymptote_condition a b) :
  a^2 = 9 ∧ b^2 = 27 := by sorry

end hyperbola_parameters_l609_60984


namespace pencils_given_eq_difference_l609_60981

/-- The number of pencils Jesse gave to Joshua -/
def pencils_given : ℕ := sorry

/-- The initial number of pencils Jesse had -/
def initial_pencils : ℕ := 78

/-- The remaining number of pencils Jesse has -/
def remaining_pencils : ℕ := 34

/-- Theorem stating that the number of pencils given is equal to the difference between initial and remaining pencils -/
theorem pencils_given_eq_difference : 
  pencils_given = initial_pencils - remaining_pencils := by sorry

end pencils_given_eq_difference_l609_60981


namespace committee_selection_count_club_committee_count_l609_60925

theorem committee_selection_count : Nat → Nat → Nat
  | n, k => (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem club_committee_count :
  committee_selection_count 30 5 = 142506 := by
  sorry

end committee_selection_count_club_committee_count_l609_60925


namespace problem_solution_l609_60953

theorem problem_solution (a b : ℝ) (ha : a = 2 + Real.sqrt 3) (hb : b = 2 - Real.sqrt 3) :
  (a - b = 2 * Real.sqrt 3) ∧ (a * b = 1) ∧ (a^2 + b^2 - 5*a*b = 9) := by
  sorry

end problem_solution_l609_60953


namespace min_value_expression_l609_60990

theorem min_value_expression (a b : ℝ) (h1 : b = 1 + a) (h2 : 0 < b) (h3 : b < 1) :
  ∀ x y : ℝ, x = 1 + y → 0 < x → x < 1 → 
    (2023 / b - (a + 1) / (2023 * a)) ≤ (2023 / x - (y + 1) / (2023 * y)) →
    2023 / b - (a + 1) / (2023 * a) ≥ 2025 :=
by sorry

end min_value_expression_l609_60990


namespace order_total_parts_l609_60968

theorem order_total_parts (total_cost : ℕ) (cost_cheap : ℕ) (cost_expensive : ℕ) (num_expensive : ℕ) :
  total_cost = 2380 →
  cost_cheap = 20 →
  cost_expensive = 50 →
  num_expensive = 40 →
  ∃ (num_cheap : ℕ), num_cheap * cost_cheap + num_expensive * cost_expensive = total_cost ∧
                      num_cheap + num_expensive = 59 :=
by sorry

end order_total_parts_l609_60968


namespace cube_volume_from_surface_area_l609_60921

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 54 → s^3 = 27 :=
by
  sorry

end cube_volume_from_surface_area_l609_60921


namespace largest_three_digit_multiple_of_7_with_digit_sum_21_l609_60939

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_7_with_digit_sum_21 :
  ∃ (n : ℕ), is_three_digit n ∧ n % 7 = 0 ∧ digit_sum n = 21 ∧
  ∀ (m : ℕ), is_three_digit m ∧ m % 7 = 0 ∧ digit_sum m = 21 → m ≤ n :=
by
  use 966
  sorry

end largest_three_digit_multiple_of_7_with_digit_sum_21_l609_60939


namespace min_value_theorem_l609_60962

theorem min_value_theorem (x : ℝ) (h : x > 4) :
  (x + 10) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 14 ∧
  ((x + 10) / Real.sqrt (x - 4) = 2 * Real.sqrt 14 ↔ x = 22) :=
by sorry

end min_value_theorem_l609_60962


namespace charles_discount_l609_60963

/-- The discount given to a customer, given the total cost before discount and the amount paid after discount. -/
def discount (total_cost : ℝ) (amount_paid : ℝ) : ℝ :=
  total_cost - amount_paid

/-- Theorem: The discount given to Charles is $2. -/
theorem charles_discount : discount 45 43 = 2 := by
  sorry

end charles_discount_l609_60963


namespace largest_factor_of_consecutive_product_l609_60970

theorem largest_factor_of_consecutive_product (n : ℕ) : 
  n % 10 = 4 → 120 ∣ n * (n + 1) * (n + 2) ∧ 
  ∀ m : ℕ, m > 120 → ∃ k : ℕ, k % 10 = 4 ∧ ¬(m ∣ k * (k + 1) * (k + 2)) := by
  sorry

end largest_factor_of_consecutive_product_l609_60970


namespace evaluation_of_expression_l609_60906

theorem evaluation_of_expression : (4^4 - 4*(4-1)^4)^4 = 21381376 := by
  sorry

end evaluation_of_expression_l609_60906


namespace constant_sum_l609_60994

theorem constant_sum (x y : ℝ) (h : x + y = 4) : 5 * x + 5 * y = 20 := by
  sorry

end constant_sum_l609_60994


namespace power_zero_fraction_l609_60937

theorem power_zero_fraction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a / b) ^ (0 : ℝ) = 1 := by sorry

end power_zero_fraction_l609_60937


namespace omega_range_l609_60985

theorem omega_range (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.cos (ω * x + π / 6)) →
  ω > 0 →
  (∀ x ∈ Set.Icc 0 π, f x ∈ Set.Icc (-1) (Real.sqrt 3 / 2)) →
  ω ∈ Set.Icc (5 / 6) (5 / 3) :=
sorry

end omega_range_l609_60985


namespace line_parallel_perp_implies_planes_perp_l609_60916

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Parallel relation between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicular relation between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicular relation between two planes -/
def planesPerpendicular (p1 : Plane3D) (p2 : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line is parallel to one plane and perpendicular to another,
    then the two planes are perpendicular -/
theorem line_parallel_perp_implies_planes_perp
  (c : Line3D) (α β : Plane3D)
  (h1 : parallel c α)
  (h2 : perpendicular c β) :
  planesPerpendicular α β :=
sorry

end line_parallel_perp_implies_planes_perp_l609_60916


namespace diagonal_intersection_l609_60945

/-- A regular 18-sided polygon -/
structure RegularPolygon18 where
  vertices : Fin 18 → ℝ × ℝ
  is_regular : ∀ i j : Fin 18, 
    dist (vertices i) (vertices ((i + 1) % 18)) = 
    dist (vertices j) (vertices ((j + 1) % 18))

/-- A diagonal of the polygon -/
def diagonal (p : RegularPolygon18) (i j : Fin 18) : Set (ℝ × ℝ) :=
  {x | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = (1 - t) • p.vertices i + t • p.vertices j}

/-- The statement to be proved -/
theorem diagonal_intersection (p : RegularPolygon18) :
  ∃ x : ℝ × ℝ, x ∈ diagonal p 1 11 ∩ diagonal p 7 17 ∩ diagonal p 4 15 ∧
  (∀ i : Fin 18, x ∉ diagonal p i ((i + 9) % 18)) :=
sorry

end diagonal_intersection_l609_60945


namespace min_omega_for_overlapping_sine_graphs_l609_60993

/-- Given a function f(x) = sin(ωx + π/3) where ω > 0, if the graph of y = f(x) is shifted
    to the right by 2π/3 units and overlaps with the original graph, then the minimum
    value of ω is 3. -/
theorem min_omega_for_overlapping_sine_graphs (ω : ℝ) (f : ℝ → ℝ) :
  ω > 0 →
  (∀ x, f x = Real.sin (ω * x + π / 3)) →
  (∀ x, f (x + 2 * π / 3) = f x) →
  3 ≤ ω ∧ ∀ ω', (ω' > 0 ∧ ∀ x, f (x + 2 * π / 3) = f x) → ω ≤ ω' :=
by sorry

end min_omega_for_overlapping_sine_graphs_l609_60993


namespace triangle_problem_l609_60900

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
  (2 * t.b * (2 * t.b - t.c) * Real.cos t.A = t.a^2 + t.b^2 - t.c^2) →
  ((1/2) * t.b * t.c * Real.sin t.A = 25 * Real.sqrt 3 / 4) →
  (t.a = 5) →
  -- Conclusions
  (t.A = π/3 ∧ t.b + t.c = 10) :=
by sorry


end triangle_problem_l609_60900


namespace cauliflower_increase_l609_60914

theorem cauliflower_increase (n : ℕ) (h : n^2 = 12544) : n^2 - (n-1)^2 = 223 := by
  sorry

end cauliflower_increase_l609_60914


namespace sally_found_thirteen_l609_60997

/-- The number of seashells Tim found -/
def tim_seashells : ℕ := 37

/-- The total number of seashells Tim and Sally found together -/
def total_seashells : ℕ := 50

/-- The number of seashells Sally found -/
def sally_seashells : ℕ := total_seashells - tim_seashells

theorem sally_found_thirteen : sally_seashells = 13 := by
  sorry

end sally_found_thirteen_l609_60997


namespace chocolate_count_correct_l609_60992

/-- The number of small boxes in the large box -/
def total_small_boxes : ℕ := 17

/-- The number of small boxes containing medium boxes -/
def boxes_with_medium : ℕ := 10

/-- The number of medium boxes in each of the first 10 small boxes -/
def medium_boxes_per_small : ℕ := 4

/-- The number of chocolate bars in each medium box -/
def chocolates_per_medium : ℕ := 26

/-- The number of chocolate bars in each of the first two of the remaining small boxes -/
def chocolates_in_first_two : ℕ := 18

/-- The number of chocolate bars in each of the next three of the remaining small boxes -/
def chocolates_in_next_three : ℕ := 22

/-- The number of chocolate bars in each of the last two of the remaining small boxes -/
def chocolates_in_last_two : ℕ := 30

/-- The total number of chocolate bars in the large box -/
def total_chocolates : ℕ := 1202

theorem chocolate_count_correct : 
  (boxes_with_medium * medium_boxes_per_small * chocolates_per_medium) +
  (2 * chocolates_in_first_two) +
  (3 * chocolates_in_next_three) +
  (2 * chocolates_in_last_two) = total_chocolates :=
by sorry

end chocolate_count_correct_l609_60992


namespace test_passing_difference_l609_60944

theorem test_passing_difference (total : ℕ) (arithmetic : ℕ) (algebra : ℕ) (geometry : ℕ)
  (arithmetic_correct : ℚ) (algebra_correct : ℚ) (geometry_correct : ℚ) (passing_grade : ℚ)
  (h1 : total = 90)
  (h2 : arithmetic = 20)
  (h3 : algebra = 40)
  (h4 : geometry = 30)
  (h5 : arithmetic_correct = 60 / 100)
  (h6 : algebra_correct = 50 / 100)
  (h7 : geometry_correct = 70 / 100)
  (h8 : passing_grade = 65 / 100)
  (h9 : total = arithmetic + algebra + geometry) :
  ⌈total * passing_grade⌉ - (⌊arithmetic * arithmetic_correct⌋ + ⌊algebra * algebra_correct⌋ + ⌊geometry * geometry_correct⌋) = 6 := by
  sorry

end test_passing_difference_l609_60944


namespace divisibility_in_ones_sequence_l609_60986

theorem divisibility_in_ones_sequence (k : ℕ) (hprime : Nat.Prime k) (h2 : k ≠ 2) (h5 : k ≠ 5) :
  ∃ n : ℕ, n ≤ k ∧ k ∣ ((10^n - 1) / 9) := by
  sorry

end divisibility_in_ones_sequence_l609_60986


namespace train_length_proof_l609_60972

/-- Proves that the length of each train is 150 meters given the specified conditions -/
theorem train_length_proof (faster_speed slower_speed : ℝ) (passing_time : ℝ) : 
  faster_speed = 46 →
  slower_speed = 36 →
  passing_time = 108 →
  let relative_speed := (faster_speed - slower_speed) * (5 / 18)
  let train_length := relative_speed * passing_time / 2
  train_length = 150 := by
  sorry

#check train_length_proof

end train_length_proof_l609_60972


namespace sum_of_proportions_l609_60960

theorem sum_of_proportions (a b c d e f : ℝ) 
  (h1 : a / b = 2) 
  (h2 : c / d = 2) 
  (h3 : e / f = 2) 
  (h4 : b + d + f = 4) : 
  a + c + e = 8 := by
sorry

end sum_of_proportions_l609_60960


namespace car_speed_problem_l609_60927

theorem car_speed_problem (highway_length : ℝ) (meeting_time : ℝ) (car2_speed : ℝ) : 
  highway_length = 333 ∧ 
  meeting_time = 3 ∧ 
  car2_speed = 57 →
  ∃ car1_speed : ℝ, 
    car1_speed * meeting_time + car2_speed * meeting_time = highway_length ∧ 
    car1_speed = 54 :=
by sorry

end car_speed_problem_l609_60927


namespace annas_size_l609_60922

theorem annas_size (anna_size : ℕ) 
  (becky_size : ℕ) (ginger_size : ℕ) 
  (h1 : becky_size = 3 * anna_size)
  (h2 : ginger_size = 2 * becky_size - 4)
  (h3 : ginger_size = 8) : 
  anna_size = 2 := by
  sorry

end annas_size_l609_60922
