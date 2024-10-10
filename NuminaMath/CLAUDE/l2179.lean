import Mathlib

namespace distributive_property_l2179_217996

theorem distributive_property (a : ℝ) : 2 * (a - 1) = 2 * a - 2 := by
  sorry

end distributive_property_l2179_217996


namespace no_linear_term_implies_p_value_l2179_217913

theorem no_linear_term_implies_p_value (p : ℝ) : 
  (∀ x : ℝ, ∃ a b c : ℝ, (x - 3) * (x^2 + p*x - 1) = a*x^3 + b*x^2 + c) → 
  p = -1/3 :=
by sorry

end no_linear_term_implies_p_value_l2179_217913


namespace newspaper_subscription_cost_l2179_217949

theorem newspaper_subscription_cost (discount_rate : ℝ) (discounted_price : ℝ) (normal_price : ℝ) : 
  discount_rate = 0.45 →
  discounted_price = 44 →
  normal_price * (1 - discount_rate) = discounted_price →
  normal_price = 80 := by
sorry

end newspaper_subscription_cost_l2179_217949


namespace no_formula_matches_l2179_217998

def x_values : List ℕ := [1, 2, 3, 4, 5]
def y_values : List ℕ := [4, 12, 28, 52, 84]

def formula_a (x : ℕ) : ℕ := 4 * x^2
def formula_b (x : ℕ) : ℕ := 3 * x^2 + 3 * x + 1
def formula_c (x : ℕ) : ℕ := 5 * x^3 - 2 * x
def formula_d (x : ℕ) : ℕ := 4 * x^2 + 4 * x

theorem no_formula_matches : 
  ∀ (i : Fin 5), 
    (formula_a (x_values.get i) ≠ y_values.get i) ∧
    (formula_b (x_values.get i) ≠ y_values.get i) ∧
    (formula_c (x_values.get i) ≠ y_values.get i) ∧
    (formula_d (x_values.get i) ≠ y_values.get i) := by
  sorry

end no_formula_matches_l2179_217998


namespace octal_minus_septenary_in_decimal_l2179_217983

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The problem statement -/
theorem octal_minus_septenary_in_decimal : 
  let octal := [2, 1, 3]
  let septenary := [1, 4, 2]
  to_base_10 octal 8 - to_base_10 septenary 7 = 60 := by
  sorry


end octal_minus_septenary_in_decimal_l2179_217983


namespace salem_women_count_l2179_217968

/-- Proves the number of women in Salem after population change -/
theorem salem_women_count (leesburg_population : ℕ) (salem_multiplier : ℕ) (people_moving_out : ℕ) :
  leesburg_population = 58940 →
  salem_multiplier = 15 →
  people_moving_out = 130000 →
  (salem_multiplier * leesburg_population - people_moving_out) / 2 = 377050 := by
sorry

end salem_women_count_l2179_217968


namespace sequence_property_l2179_217973

/-- Given an arithmetic sequence {aₙ} and a geometric sequence {bₙ} where
    a₃ = b₃ = a, a₆ = b₆ = b, and a > b, prove that if (a₄-b₄)(a₅-b₅) < 0, then ab < 0. -/
theorem sequence_property (a b : ℝ) (aₙ : ℕ → ℝ) (bₙ : ℕ → ℝ) 
    (h_arithmetic : ∀ n : ℕ, aₙ (n + 1) - aₙ n = aₙ 2 - aₙ 1)
    (h_geometric : ∀ n : ℕ, bₙ (n + 1) / bₙ n = bₙ 2 / bₙ 1)
    (h_a3 : aₙ 3 = a) (h_b3 : bₙ 3 = a)
    (h_a6 : aₙ 6 = b) (h_b6 : bₙ 6 = b)
    (h_a_gt_b : a > b) :
  (aₙ 4 - bₙ 4) * (aₙ 5 - bₙ 5) < 0 → a * b < 0 := by
  sorry

end sequence_property_l2179_217973


namespace row_swap_property_l2179_217935

def row_swap_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, 0]

theorem row_swap_property (A : Matrix (Fin 2) (Fin 2) ℝ) :
  row_swap_matrix * A = Matrix.of (λ i j => A (1 - i) j) := by
  sorry

end row_swap_property_l2179_217935


namespace sum_of_seventh_powers_squared_l2179_217902

theorem sum_of_seventh_powers_squared (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (sum_zero : a + b + c = 0) :
  (a^7 + b^7 + c^7)^2 / ((a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) * (a^4 + b^4 + c^4) * (a^5 + b^5 + c^5)) = 49/60 := by
  sorry

end sum_of_seventh_powers_squared_l2179_217902


namespace circle_fits_in_triangle_l2179_217964

theorem circle_fits_in_triangle (a b c : ℝ) (S : ℝ) : 
  a = 3 ∧ b = 4 ∧ c = 5 → S = 25 / 8 →
  ∃ (r R : ℝ), r = (a + b - c) / 2 ∧ S = π * R^2 ∧ R < r := by
  sorry

end circle_fits_in_triangle_l2179_217964


namespace triangle_division_theorem_l2179_217925

theorem triangle_division_theorem (A B C : ℝ) :
  A + B + C = 180 →
  B = 120 →
  (∃ D : ℝ, (A = D ∧ B / 2 = D) ∨ (C = D ∧ B / 2 = D) ∨ (A = D ∧ C = D)) →
  ((A = 40 ∧ C = 20) ∨ (A = 45 ∧ C = 15) ∨ (A = 20 ∧ C = 40) ∨ (A = 15 ∧ C = 45)) :=
by sorry

end triangle_division_theorem_l2179_217925


namespace circle_area_equality_l2179_217937

theorem circle_area_equality (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 15) (h₂ : r₂ = 25) (h₃ : r₃ = 20) :
  π * r₃^2 = π * r₂^2 - π * r₁^2 :=
by sorry

end circle_area_equality_l2179_217937


namespace sin_405_degrees_l2179_217916

theorem sin_405_degrees : Real.sin (405 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_degrees_l2179_217916


namespace kaili_circle_method_l2179_217922

theorem kaili_circle_method (S : ℝ) (V : ℝ) (h : S = 4 * Real.pi / 9) :
  (2/3)^3 = 16 * V / 9 :=
sorry

end kaili_circle_method_l2179_217922


namespace museum_ticket_cost_l2179_217952

def entrance_ticket_cost (num_students : ℕ) (num_teachers : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (num_students + num_teachers)

theorem museum_ticket_cost :
  entrance_ticket_cost 20 3 115 = 5 := by
sorry

end museum_ticket_cost_l2179_217952


namespace quadratic_equation_b_value_l2179_217967

theorem quadratic_equation_b_value 
  (b : ℝ) 
  (h1 : 2 * (5 : ℝ)^2 + b * 5 - 65 = 0) : 
  b = 3 := by
sorry

end quadratic_equation_b_value_l2179_217967


namespace range_of_a_l2179_217900

theorem range_of_a (p q : Prop) (hp : ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0)
  (hq : ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) (hpq : p ∧ q) :
  a ≤ -2 ∨ a = 1 := by
  sorry

end range_of_a_l2179_217900


namespace division_theorem_l2179_217950

theorem division_theorem (A B : ℕ) : 23 = 6 * A + B ∧ B < 6 → A = 3 := by
  sorry

end division_theorem_l2179_217950


namespace no_real_roots_x_squared_plus_four_l2179_217966

theorem no_real_roots_x_squared_plus_four :
  ¬ ∃ (x : ℝ), x^2 + 4 = 0 := by
sorry

end no_real_roots_x_squared_plus_four_l2179_217966


namespace programmer_work_hours_l2179_217975

theorem programmer_work_hours (flow_chart_time : ℚ) (coding_time : ℚ) (debug_time : ℚ) 
  (h1 : flow_chart_time = 1/4)
  (h2 : coding_time = 3/8)
  (h3 : debug_time = 1 - (flow_chart_time + coding_time))
  (h4 : debug_time * 48 = 18) :
  48 = 48 := by sorry

end programmer_work_hours_l2179_217975


namespace monster_count_theorem_l2179_217910

/-- Calculates the total number of monsters after 5 days given the initial count and daily growth factors -/
def total_monsters (initial : ℕ) (factor2 factor3 factor4 factor5 : ℕ) : ℕ :=
  initial + 
  initial * factor2 + 
  initial * factor2 * factor3 + 
  initial * factor2 * factor3 * factor4 + 
  initial * factor2 * factor3 * factor4 * factor5

/-- Theorem stating that given the specific initial count and growth factors, the total number of monsters after 5 days is 872 -/
theorem monster_count_theorem : total_monsters 2 3 4 5 6 = 872 := by
  sorry

end monster_count_theorem_l2179_217910


namespace distance_from_origin_to_point_l2179_217965

theorem distance_from_origin_to_point :
  let x : ℝ := 8
  let y : ℝ := 15
  Real.sqrt (x^2 + y^2) = 17 := by sorry

end distance_from_origin_to_point_l2179_217965


namespace min_perimeter_isosceles_triangles_l2179_217908

-- Define the structure for an isosceles triangle
structure IsoscelesTriangle where
  side : ℕ  -- Equal sides
  base : ℕ  -- Base

-- Define the theorem
theorem min_perimeter_isosceles_triangles 
  (t1 t2 : IsoscelesTriangle) 
  (h1 : t1 ≠ t2)  -- Noncongruent triangles
  (h2 : 2 * t1.side + t1.base = 2 * t2.side + t2.base)  -- Same perimeter
  (h3 : t1.side * t1.base = t2.side * t2.base)  -- Same area (simplified)
  (h4 : 9 * t1.base = 8 * t2.base)  -- Ratio of bases
  : 2 * t1.side + t1.base ≥ 868 :=
by sorry

end min_perimeter_isosceles_triangles_l2179_217908


namespace sean_purchase_cost_l2179_217921

/-- The cost of items in Sean's purchase -/
def CostCalculation (soda_price : ℝ) : Prop :=
  let soup_price := 3 * soda_price
  let sandwich_price := 3 * soup_price
  (3 * soda_price) + (2 * soup_price) + sandwich_price = 18

/-- Theorem stating the total cost of Sean's purchase -/
theorem sean_purchase_cost :
  CostCalculation 1 := by
  sorry

end sean_purchase_cost_l2179_217921


namespace child_growth_proof_l2179_217946

/-- Calculates the growth in height given current and previous heights -/
def heightGrowth (currentHeight previousHeight : Float) : Float :=
  currentHeight - previousHeight

theorem child_growth_proof :
  let currentHeight : Float := 41.5
  let previousHeight : Float := 38.5
  heightGrowth currentHeight previousHeight = 3 := by
  sorry

end child_growth_proof_l2179_217946


namespace binomial_coefficient_8_4_l2179_217909

theorem binomial_coefficient_8_4 : Nat.choose 8 4 = 70 := by
  sorry

end binomial_coefficient_8_4_l2179_217909


namespace least_common_multiple_plus_one_l2179_217914

def divisors : List Nat := [2, 3, 5, 7, 8, 9, 10]

theorem least_common_multiple_plus_one : 
  ∃ (n : Nat), n > 1 ∧ 
  (∀ d ∈ divisors, n % d = 1) ∧
  (∀ m : Nat, m > 1 → (∀ d ∈ divisors, m % d = 1) → m ≥ n) ∧
  n = 2521 := by
  sorry

end least_common_multiple_plus_one_l2179_217914


namespace license_plate_palindrome_probability_l2179_217979

/-- The probability of a license plate containing at least one palindrome -/
theorem license_plate_palindrome_probability :
  let total_arrangements : ℕ := 26^4 * 10^4
  let letter_palindromes : ℕ := 26^2
  let digit_palindromes : ℕ := 10^2
  let both_palindromes : ℕ := letter_palindromes * digit_palindromes
  let palindrome_probability : ℚ := (letter_palindromes * 10^4 + digit_palindromes * 26^4 - both_palindromes) / total_arrangements
  palindrome_probability = 775 / 67600 :=
by sorry

end license_plate_palindrome_probability_l2179_217979


namespace river_depth_l2179_217930

/-- The depth of a river given its width, flow rate, and volume of water per minute. -/
theorem river_depth (width : ℝ) (flow_rate : ℝ) (volume_per_minute : ℝ) :
  width = 45 →
  flow_rate = 5 →
  volume_per_minute = 7500 →
  (volume_per_minute / (width * (flow_rate * 1000 / 60))) = 2 := by
  sorry


end river_depth_l2179_217930


namespace arithmetic_sequence_tangent_l2179_217915

/-- Given an arithmetic sequence {a_n} where a_1 + a_7 + a_13 = 4π, 
    prove that tan(a_2 + a_12) = -√3 -/
theorem arithmetic_sequence_tangent (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  a 1 + a 7 + a 13 = 4 * Real.pi →                  -- given condition
  Real.tan (a 2 + a 12) = -Real.sqrt 3 := by
sorry

end arithmetic_sequence_tangent_l2179_217915


namespace second_pipe_filling_time_l2179_217944

/-- Given a pool that can be filled by one pipe in 10 hours and by both pipes in 3.75 hours,
    prove that the second pipe alone takes 6 hours to fill the pool. -/
theorem second_pipe_filling_time
  (time_pipe1 : ℝ) (time_both : ℝ) (time_pipe2 : ℝ)
  (h1 : time_pipe1 = 10)
  (h2 : time_both = 3.75)
  (h3 : 1 / time_pipe1 + 1 / time_pipe2 = 1 / time_both) :
  time_pipe2 = 6 :=
sorry

end second_pipe_filling_time_l2179_217944


namespace gcf_of_30_90_75_l2179_217945

theorem gcf_of_30_90_75 : Nat.gcd 30 (Nat.gcd 90 75) = 15 := by sorry

end gcf_of_30_90_75_l2179_217945


namespace sum_of_s_r_at_points_l2179_217961

def r (x : ℝ) : ℝ := |x| + 3

def s (x : ℝ) : ℝ := -|x|

def evaluation_points : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_of_s_r_at_points :
  (evaluation_points.map (λ x => s (r x))).sum = -63 := by
  sorry

end sum_of_s_r_at_points_l2179_217961


namespace stone_splitting_properties_l2179_217903

/-- Represents the state of stone piles -/
structure PileState :=
  (piles : List Nat)
  (valid : piles.sum = 100)

/-- Represents a single move in the stone-splitting process -/
def split_move (s : PileState) : PileState → Prop :=
  sorry

/-- Represents the complete process of splitting stones -/
def splitting_process (initial : PileState) (final : PileState) : Prop :=
  sorry

theorem stone_splitting_properties 
  (initial : PileState)
  (final : PileState)
  (h_initial : initial.piles = [100])
  (h_final : final.piles.all (· = 1) ∧ final.piles.length = 100)
  (h_process : splitting_process initial final) :
  (∃ s : PileState, splitting_process initial s ∧ 
    (∃ sub : List Nat, sub ⊆ s.piles ∧ sub.length = 30 ∧ sub.sum = 60)) ∧
  (∃ s : PileState, splitting_process initial s ∧ 
    (∃ sub : List Nat, sub ⊆ s.piles ∧ sub.length = 20 ∧ sub.sum = 60)) ∧
  (∃ f : PileState → PileState, 
    splitting_process initial (f final) ∧
    ∀ s, splitting_process initial s → splitting_process s (f final) →
      ¬∃ sub : List Nat, sub ⊆ s.piles ∧ sub.length = 19 ∧ sub.sum = 60) :=
sorry

end stone_splitting_properties_l2179_217903


namespace ratio_x_to_y_is_eight_l2179_217956

theorem ratio_x_to_y_is_eight (x y : ℝ) (h : y = 0.125 * x) : x / y = 8 := by
  sorry

end ratio_x_to_y_is_eight_l2179_217956


namespace sphere_surface_area_doubling_l2179_217951

theorem sphere_surface_area_doubling (r : ℝ) :
  (4 * Real.pi * r^2 = 2464) →
  (4 * Real.pi * (2*r)^2 = 39376) :=
by sorry

end sphere_surface_area_doubling_l2179_217951


namespace mia_average_first_four_days_l2179_217938

theorem mia_average_first_four_days 
  (total_distance : ℝ) 
  (race_days : ℕ) 
  (jesse_avg_first_three : ℝ) 
  (jesse_day_four : ℝ) 
  (combined_avg_last_three : ℝ) 
  (h1 : total_distance = 30)
  (h2 : race_days = 7)
  (h3 : jesse_avg_first_three = 2/3)
  (h4 : jesse_day_four = 10)
  (h5 : combined_avg_last_three = 6) :
  ∃ mia_avg_first_four : ℝ,
    mia_avg_first_four = 3 ∧
    mia_avg_first_four * 4 + combined_avg_last_three * 3 = total_distance ∧
    jesse_avg_first_three * 3 + jesse_day_four + combined_avg_last_three * 3 = total_distance :=
by sorry

end mia_average_first_four_days_l2179_217938


namespace pauls_crayons_given_to_friends_l2179_217960

/-- Given information about Paul's crayons --/
structure CrayonInfo where
  initial : ℕ  -- Initial number of crayons
  lost_difference : ℕ  -- Difference between lost and given crayons
  total_gone : ℕ  -- Total number of crayons no longer in possession

/-- Calculate the number of crayons given to friends --/
def crayons_given_to_friends (info : CrayonInfo) : ℕ :=
  (info.total_gone - info.lost_difference) / 2

/-- Theorem stating the number of crayons Paul gave to his friends --/
theorem pauls_crayons_given_to_friends :
  let info : CrayonInfo := {
    initial := 110,
    lost_difference := 322,
    total_gone := 412
  }
  crayons_given_to_friends info = 45 := by
  sorry

end pauls_crayons_given_to_friends_l2179_217960


namespace geometric_sequence_special_case_l2179_217948

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_special_case (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 3)^2 - 4*(a 3) + 3 = 0 →
  (a 7)^2 - 4*(a 7) + 3 = 0 →
  a 5 = Real.sqrt 3 :=
by sorry

end geometric_sequence_special_case_l2179_217948


namespace dice_prob_same_color_l2179_217982

def prob_same_color (d1_sides d2_sides : ℕ)
  (d1_maroon d1_teal d1_cyan d1_sparkly : ℕ)
  (d2_maroon d2_teal d2_cyan d2_sparkly : ℕ) : ℚ :=
  let p_maroon := (d1_maroon : ℚ) / d1_sides * (d2_maroon : ℚ) / d2_sides
  let p_teal := (d1_teal : ℚ) / d1_sides * (d2_teal : ℚ) / d2_sides
  let p_cyan := (d1_cyan : ℚ) / d1_sides * (d2_cyan : ℚ) / d2_sides
  let p_sparkly := (d1_sparkly : ℚ) / d1_sides * (d2_sparkly : ℚ) / d2_sides
  p_maroon + p_teal + p_cyan + p_sparkly

theorem dice_prob_same_color :
  prob_same_color 20 16 5 8 6 1 4 6 5 1 = 99 / 320 := by
  sorry

end dice_prob_same_color_l2179_217982


namespace smallest_multiple_of_2_3_4_5_7_l2179_217963

theorem smallest_multiple_of_2_3_4_5_7 : ∃ n : ℕ, 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(2 ∣ m ∧ 3 ∣ m ∧ 4 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m)) ∧ 
  (2 ∣ n ∧ 3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) :=
by
  use 420
  sorry

#eval 420 % 2
#eval 420 % 3
#eval 420 % 4
#eval 420 % 5
#eval 420 % 7

end smallest_multiple_of_2_3_4_5_7_l2179_217963


namespace floor_painting_theorem_l2179_217947

/-- The number of integer pairs (a, b) satisfying the floor painting conditions -/
def floor_painting_solutions : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let a := p.1
    let b := p.2
    b > a ∧ b % 3 = 0 ∧ (a - 6) * (b - 6) = 36 ∧ a > 6 ∧ b > 6)
    (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- Theorem stating that there are exactly 2 solutions to the floor painting problem -/
theorem floor_painting_theorem : floor_painting_solutions = 2 := by
  sorry

end floor_painting_theorem_l2179_217947


namespace select_five_from_eight_l2179_217933

/-- The number of combinations of n items taken r at a time -/
def combination (n r : ℕ) : ℕ := sorry

/-- Theorem stating that selecting 5 items from 8 items results in 56 combinations -/
theorem select_five_from_eight : combination 8 5 = 56 := by sorry

end select_five_from_eight_l2179_217933


namespace sin_sum_of_complex_exponentials_l2179_217970

theorem sin_sum_of_complex_exponentials (θ φ : ℝ) :
  Complex.exp (θ * I) = (1 / 5 : ℂ) + (2 * Real.sqrt 6 / 5 : ℂ) * I ∧
  Complex.exp (φ * I) = (-5 / 13 : ℂ) - (12 / 13 : ℂ) * I →
  Real.sin (θ + φ) = -(12 - 10 * Real.sqrt 6) / 65 := by
  sorry

end sin_sum_of_complex_exponentials_l2179_217970


namespace sqrt_x_minus_two_defined_l2179_217969

theorem sqrt_x_minus_two_defined (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end sqrt_x_minus_two_defined_l2179_217969


namespace triangle_3_7_triangle_3_neg4_triangle_neg4_3_triangle_not_commutative_l2179_217940

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := -2 * a * b - b + 1

-- Theorem statements
theorem triangle_3_7 : triangle 3 7 = -48 := by sorry

theorem triangle_3_neg4 : triangle 3 (-4) = 29 := by sorry

theorem triangle_neg4_3 : triangle (-4) 3 = 22 := by sorry

theorem triangle_not_commutative : ∃ a b : ℚ, triangle a b ≠ triangle b a := by sorry

end triangle_3_7_triangle_3_neg4_triangle_neg4_3_triangle_not_commutative_l2179_217940


namespace remainder_of_n_l2179_217981

theorem remainder_of_n (n : ℕ) (h1 : n^2 % 5 = 1) (h2 : n^4 % 5 = 1) :
  n % 5 = 1 ∨ n % 5 = 4 := by
sorry

end remainder_of_n_l2179_217981


namespace fraction_product_equality_l2179_217907

theorem fraction_product_equality : 
  (3 / 4) * (36 / 60) * (10 / 4) * (14 / 28) * (9 / 3)^2 * (45 / 15) * (12 / 18) * (20 / 40)^3 = 27 / 32 := by
  sorry

end fraction_product_equality_l2179_217907


namespace inverse_expression_equals_one_thirteenth_l2179_217941

theorem inverse_expression_equals_one_thirteenth :
  (3 - 5 * (3 - 4)⁻¹ * 2)⁻¹ = (1 : ℚ) / 13 := by
  sorry

end inverse_expression_equals_one_thirteenth_l2179_217941


namespace line_equation_through_point_with_slope_point_satisfies_equation_l2179_217986

/-- Proves that the equation of a line passing through the point (1, 2) with a slope of 3 is y = 3x - 1 -/
theorem line_equation_through_point_with_slope (x y : ℝ) : 
  (y - 2 = 3 * (x - 1)) ↔ (y = 3 * x - 1) := by
  sorry

/-- Verifies that the point (1, 2) satisfies the equation y = 3x - 1 -/
theorem point_satisfies_equation : 
  2 = 3 * 1 - 1 := by
  sorry

end line_equation_through_point_with_slope_point_satisfies_equation_l2179_217986


namespace max_area_MPNQ_l2179_217987

noncomputable section

-- Define the curves C₁ and C₂ in polar coordinates
def C₁ (θ : Real) : Real := 2 * Real.sqrt 2

def C₂ (θ : Real) : Real := 4 * Real.sqrt 2 * (Real.cos θ + Real.sin θ)

-- Define the area of quadrilateral MPNQ as a function of α
def area_MPNQ (α : Real) : Real :=
  4 * Real.sqrt 2 * Real.sin (2 * α + Real.pi / 4) + 4 - 2 * Real.sqrt 2

-- Theorem statement
theorem max_area_MPNQ :
  ∃ α, 0 < α ∧ α < Real.pi / 2 ∧
  ∀ β, 0 < β → β < Real.pi / 2 →
  area_MPNQ β ≤ area_MPNQ α ∧
  area_MPNQ α = 4 + 2 * Real.sqrt 2 :=
sorry

end

end max_area_MPNQ_l2179_217987


namespace binomial_9_choose_3_l2179_217943

theorem binomial_9_choose_3 : Nat.choose 9 3 = 84 := by
  sorry

end binomial_9_choose_3_l2179_217943


namespace quadratic_minimum_l2179_217929

theorem quadratic_minimum : 
  (∀ x : ℝ, x^2 + 10*x + 3 ≥ -22) ∧ (∃ x : ℝ, x^2 + 10*x + 3 = -22) := by
  sorry

end quadratic_minimum_l2179_217929


namespace triangle_angle_proof_l2179_217984

theorem triangle_angle_proof (a b c A B C : ℝ) (S_ABC : ℝ) : 
  b = 2 →
  S_ABC = 2 * Real.sqrt 3 →
  c * Real.cos B + b * Real.cos C - 2 * a * Real.cos A = 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  S_ABC = (1 / 2) * a * b * Real.sin C →
  S_ABC = (1 / 2) * b * c * Real.sin A →
  S_ABC = (1 / 2) * c * a * Real.sin B →
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos C →
  C = π / 2 := by sorry

end triangle_angle_proof_l2179_217984


namespace positive_sqrt_1024_l2179_217994

theorem positive_sqrt_1024 : Real.sqrt 1024 = 32 := by sorry

end positive_sqrt_1024_l2179_217994


namespace math_score_calculation_l2179_217997

theorem math_score_calculation (initial_average : ℝ) (num_initial_subjects : ℕ) (average_drop : ℝ) :
  initial_average = 95 →
  num_initial_subjects = 3 →
  average_drop = 3 →
  let total_initial_score := initial_average * num_initial_subjects
  let new_average := initial_average - average_drop
  let new_total_score := new_average * (num_initial_subjects + 1)
  new_total_score - total_initial_score = 83 := by
  sorry

end math_score_calculation_l2179_217997


namespace subset_condition_l2179_217992

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem subset_condition (a : ℝ) : B a ⊆ A ↔ a = 0 ∨ a = 1/3 ∨ a = 1/5 := by
  sorry

end subset_condition_l2179_217992


namespace overtake_scenario_l2179_217977

/-- Represents the scenario where three people travel at different speeds and overtake each other -/
structure TravelScenario where
  speed_a : ℝ
  speed_b : ℝ
  speed_k : ℝ
  b_delay : ℝ
  overtake_time : ℝ
  k_start_time : ℝ

/-- The theorem statement based on the given problem -/
theorem overtake_scenario (s : TravelScenario) 
  (h1 : s.speed_a = 30)
  (h2 : s.speed_b = 40)
  (h3 : s.speed_k = 60)
  (h4 : s.b_delay = 5)
  (h5 : s.speed_a * s.overtake_time = s.speed_b * (s.overtake_time - s.b_delay))
  (h6 : s.speed_a * s.overtake_time = s.speed_k * s.k_start_time) :
  s.k_start_time = 10 := by
  sorry

end overtake_scenario_l2179_217977


namespace vasya_tolya_winning_strategy_l2179_217985

/-- Represents a player in the game -/
inductive Player : Type
| Petya : Player
| Vasya : Player
| Tolya : Player

/-- Represents a cell on the board -/
structure Cell :=
(index : Nat)

/-- Represents the game board -/
structure Board :=
(size : Nat)
(boundary_cells : Nat)

/-- Represents the game state -/
structure GameState :=
(board : Board)
(painted_cells : List Cell)
(current_player : Player)

/-- Checks if two cells are adjacent -/
def are_adjacent (c1 c2 : Cell) (board : Board) : Prop :=
  (c1.index + 1) % board.boundary_cells = c2.index ∨
  (c2.index + 1) % board.boundary_cells = c1.index

/-- Checks if two cells are symmetrical with respect to the board center -/
def are_symmetrical (c1 c2 : Cell) (board : Board) : Prop :=
  (c1.index + board.boundary_cells / 2) % board.boundary_cells = c2.index

/-- Determines if a move is valid -/
def is_valid_move (cell : Cell) (state : GameState) : Prop :=
  cell.index < state.board.boundary_cells ∧
  cell ∉ state.painted_cells ∧
  (∀ c ∈ state.painted_cells, ¬(are_adjacent cell c state.board)) ∧
  (∀ c ∈ state.painted_cells, ¬(are_symmetrical cell c state.board))

/-- Theorem: There exists a winning strategy for Vasya and Tolya -/
theorem vasya_tolya_winning_strategy :
  ∃ (strategy : GameState → Cell),
    ∀ (initial_state : GameState),
      initial_state.board.size = 100 ∧
      initial_state.board.boundary_cells = 396 ∧
      initial_state.current_player = Player.Petya →
        ∃ (final_state : GameState),
          final_state.current_player = Player.Petya ∧
          ¬∃ (move : Cell), is_valid_move move final_state :=
sorry

end vasya_tolya_winning_strategy_l2179_217985


namespace geometric_sequence_sum_l2179_217957

/-- Given a geometric sequence {a_n} where the sum of the first n terms
    S_n = (1/2)3^(n+1) - a, prove that a = 3/2 -/
theorem geometric_sequence_sum (n : ℕ) (a_n : ℕ → ℝ) (S : ℕ → ℝ) (a : ℝ) :
  (∀ k, S k = (1/2) * 3^(k+1) - a) →
  (∀ k, a_n (k+1) = S (k+1) - S k) →
  (∀ k, a_n (k+2) * a_n k = (a_n (k+1))^2) →
  a = 3/2 := by
  sorry

end geometric_sequence_sum_l2179_217957


namespace x_asymptotics_l2179_217905

/-- The Lambert W function -/
noncomputable def W : ℝ → ℝ := sorry

/-- Asymptotic equivalence -/
def asymptotic_equiv (f g : ℕ → ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ) (N : ℕ), c₁ > 0 ∧ c₂ > 0 ∧ ∀ n ≥ N, c₁ * g n ≤ f n ∧ f n ≤ c₂ * g n

theorem x_asymptotics (n : ℕ) (x : ℝ) (h : x^x = n) :
  asymptotic_equiv (λ n => x) (λ n => Real.log n / Real.log (Real.log n)) :=
sorry

end x_asymptotics_l2179_217905


namespace composition_equation_solution_l2179_217976

theorem composition_equation_solution (δ φ : ℝ → ℝ) (h1 : ∀ x, δ x = 2 * x + 5) 
  (h2 : ∀ x, φ x = 9 * x + 6) (h3 : δ (φ x) = 3) : x = -7/9 := by
  sorry

end composition_equation_solution_l2179_217976


namespace sum_greater_than_8_probability_l2179_217988

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of outcomes where the sum of two dice is 8 or less -/
def outcomes_8_or_less : ℕ := 26

/-- The probability that the sum of two dice is greater than 8 -/
def prob_sum_greater_than_8 : ℚ := 5 / 18

theorem sum_greater_than_8_probability :
  prob_sum_greater_than_8 = 1 - (outcomes_8_or_less : ℚ) / total_outcomes :=
sorry

end sum_greater_than_8_probability_l2179_217988


namespace video_game_lives_l2179_217919

theorem video_game_lives (initial_lives gained_lives final_lives : ℕ) 
  (h1 : initial_lives = 47)
  (h2 : gained_lives = 46)
  (h3 : final_lives = 70) :
  initial_lives - (final_lives - gained_lives) = 23 := by
  sorry

end video_game_lives_l2179_217919


namespace reciprocal_sum_equality_l2179_217978

theorem reciprocal_sum_equality (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (1 / x + 1 / y = 1 / z) → z = (x * y) / (x + y) := by
  sorry

end reciprocal_sum_equality_l2179_217978


namespace power_function_k_values_l2179_217906

/-- A function is a power function if it has the form f(x) = ax^n where a is a non-zero constant and n is a real number. -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^n

/-- The main theorem: if y=(k^2-k-5)x^3 is a power function, then k = 3 or k = -2 -/
theorem power_function_k_values (k : ℝ) :
  is_power_function (λ x => (k^2 - k - 5) * x^3) → k = 3 ∨ k = -2 := by
  sorry

end power_function_k_values_l2179_217906


namespace symmetric_points_sum_l2179_217936

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

theorem symmetric_points_sum (a b : ℝ) :
  symmetric_wrt_origin (-1, a) (b, 2) → a + b = -1 := by
  sorry

end symmetric_points_sum_l2179_217936


namespace subtraction_of_fractions_simplest_form_l2179_217990

theorem subtraction_of_fractions : 
  (9 : ℚ) / 23 - (5 : ℚ) / 69 = (22 : ℚ) / 69 := by
  sorry

theorem simplest_form : 
  ∀ (a b : ℤ), a ≠ 0 → b > 0 → (22 : ℚ) / 69 = (a : ℚ) / b → a = 22 ∧ b = 69 := by
  sorry

end subtraction_of_fractions_simplest_form_l2179_217990


namespace opposite_sign_sum_three_l2179_217999

theorem opposite_sign_sum_three (x y : ℝ) :
  (|x^2 - 4*x + 4| * (2*x - y - 3).sqrt < 0) →
  x + y = 3 := by
sorry

end opposite_sign_sum_three_l2179_217999


namespace custom_mult_identity_l2179_217931

/-- Custom multiplication operation -/
noncomputable def customMult (a b c : ℝ) (x y : ℝ) : ℝ := a * x + b * y + c * x * y

theorem custom_mult_identity {a b c : ℝ} (h1 : customMult a b c 1 2 = 4) (h2 : customMult a b c 2 3 = 6) :
  ∃ m : ℝ, m ≠ 0 ∧ (∀ x : ℝ, customMult a b c x m = x) → m = 5 := by
  sorry


end custom_mult_identity_l2179_217931


namespace lcm_132_315_l2179_217928

theorem lcm_132_315 : Nat.lcm 132 315 = 13860 := by sorry

end lcm_132_315_l2179_217928


namespace floor_plus_x_eq_seventeen_fourths_l2179_217971

theorem floor_plus_x_eq_seventeen_fourths :
  ∃! (x : ℚ), ⌊x⌋ + x = 17/4 :=
by
  -- The proof goes here
  sorry

end floor_plus_x_eq_seventeen_fourths_l2179_217971


namespace platform_length_platform_length_proof_l2179_217904

/-- Calculates the length of a platform given train parameters -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let total_distance := train_speed_mps * crossing_time
  total_distance - train_length

/-- Proves that the platform length is 50 meters given the specific conditions -/
theorem platform_length_proof : 
  platform_length 250 72 15 = 50 := by
  sorry

end platform_length_platform_length_proof_l2179_217904


namespace polynomial_inverse_property_l2179_217974

-- Define the polynomials p and P
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def P (A B C : ℝ) (x : ℝ) : ℝ := A * x^2 + B * x + C

-- State the theorem
theorem polynomial_inverse_property 
  (a b c A B C : ℝ) : 
  (∀ x : ℝ, P A B C (p a b c x) = x) → 
  (∀ x : ℝ, p a b c (P A B C x) = x) :=
by
  sorry

end polynomial_inverse_property_l2179_217974


namespace symmetric_points_sum_l2179_217942

/-- Two points are symmetric about the origin if their coordinates have opposite signs -/
def symmetric_about_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

/-- If points M(3,a-2) and N(b,a) are symmetric about the origin, then a + b = -2 -/
theorem symmetric_points_sum (a b : ℝ) :
  symmetric_about_origin 3 (a - 2) b a → a + b = -2 := by
sorry

end symmetric_points_sum_l2179_217942


namespace clothes_expenditure_fraction_l2179_217939

def salary : ℝ := 190000

theorem clothes_expenditure_fraction 
  (food_fraction : ℝ) 
  (rent_fraction : ℝ) 
  (remaining : ℝ) 
  (h1 : food_fraction = 1/5)
  (h2 : rent_fraction = 1/10)
  (h3 : remaining = 19000)
  (h4 : ∃ (clothes_fraction : ℝ), 
    salary * (1 - food_fraction - rent_fraction - clothes_fraction) = remaining) :
  ∃ (clothes_fraction : ℝ), clothes_fraction = 3/5 := by
sorry

end clothes_expenditure_fraction_l2179_217939


namespace ab_value_l2179_217991

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end ab_value_l2179_217991


namespace equation_solution_l2179_217959

theorem equation_solution :
  ∃! x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ∧ x = -7/6 := by
  sorry

end equation_solution_l2179_217959


namespace A_equals_one_two_l2179_217924

def A : Set ℤ := {x | 0 < x ∧ x ≤ 2}

theorem A_equals_one_two : A = {1, 2} := by
  sorry

end A_equals_one_two_l2179_217924


namespace power_inequality_l2179_217927

def S : Set ℤ := {-2, -1, 0, 1, 2, 3}

theorem power_inequality (n : ℤ) :
  n ∈ S → ((-1/2 : ℚ)^n > (-1/5 : ℚ)^n ↔ n = -1 ∨ n = 2) :=
by sorry

end power_inequality_l2179_217927


namespace range_a_theorem_l2179_217980

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := a ≤ -2 ∨ a = 1

-- State the theorem
theorem range_a_theorem : 
  ∀ a : ℝ, (p a ∧ q a) → range_of_a a := by
  sorry

end range_a_theorem_l2179_217980


namespace point_coordinates_l2179_217923

/-- The coordinates of a point A(a,b) satisfying given conditions -/
theorem point_coordinates :
  ∀ (a b : ℝ),
    (|b| = 3) →  -- Distance from A to x-axis is 3
    (|a| = 4) →  -- Distance from A to y-axis is 4
    (a > b) →    -- Given condition a > b
    ((a = 4 ∧ b = -3) ∨ (a = 4 ∧ b = 3)) := by
  sorry


end point_coordinates_l2179_217923


namespace problem_1_l2179_217958

theorem problem_1 : 3^2 * (-1 + 3) - (-16) / 8 = 20 := by sorry

end problem_1_l2179_217958


namespace boat_current_rate_l2179_217932

/-- Proves that given a boat with a speed of 42 km/hr in still water,
    traveling 35.2 km downstream in 44 minutes, the rate of the current is 6 km/hr. -/
theorem boat_current_rate 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 42)
  (h2 : distance = 35.2)
  (h3 : time = 44 / 60) : 
  ∃ (current_rate : ℝ), 
    current_rate = 6 ∧ 
    distance = (boat_speed + current_rate) * time :=
by sorry

end boat_current_rate_l2179_217932


namespace sqrt_3a_plus_2b_l2179_217918

theorem sqrt_3a_plus_2b (a b : ℝ) 
  (h1 : (2*a + 3)^2 = 3^2) 
  (h2 : (5*a + 2*b - 1)^2 = 4^2) : 
  (3*a + 2*b)^2 = 4^2 := by
sorry

end sqrt_3a_plus_2b_l2179_217918


namespace continuous_stripe_probability_is_two_81ths_l2179_217917

/-- A regular tetrahedron with stripes painted on its faces -/
structure StripedTetrahedron where
  /-- The number of faces in a tetrahedron -/
  num_faces : ℕ
  /-- The number of possible stripe orientations per face -/
  orientations_per_face : ℕ
  /-- The total number of possible stripe combinations -/
  total_combinations : ℕ
  /-- The number of favorable outcomes (continuous stripes) -/
  favorable_outcomes : ℕ
  /-- Constraint: num_faces is 4 for a tetrahedron -/
  face_constraint : num_faces = 4
  /-- Constraint: orientations_per_face is 3 -/
  orientation_constraint : orientations_per_face = 3
  /-- Constraint: total_combinations is orientations_per_face^num_faces -/
  combination_constraint : total_combinations = orientations_per_face ^ num_faces
  /-- Constraint: favorable_outcomes is 2 -/
  outcome_constraint : favorable_outcomes = 2

/-- The probability of having a continuous stripe connecting all vertices -/
def continuous_stripe_probability (t : StripedTetrahedron) : ℚ :=
  t.favorable_outcomes / t.total_combinations

/-- Theorem: The probability of a continuous stripe is 2/81 -/
theorem continuous_stripe_probability_is_two_81ths (t : StripedTetrahedron) :
  continuous_stripe_probability t = 2 / 81 := by
  sorry

end continuous_stripe_probability_is_two_81ths_l2179_217917


namespace problem_solution_l2179_217911

theorem problem_solution (x : ℝ) (h1 : Real.sqrt ((3 * x) / 7) = x) (h2 : x ≠ 0) : x = 3 / 7 := by
  sorry

end problem_solution_l2179_217911


namespace min_value_of_g_l2179_217972

/-- The function f as defined in the problem -/
def f (x₁ x₂ x₃ : ℝ) : ℝ :=
  -2 * (x₁^3 + x₂^3 + x₃^3) + 3 * (x₁^2*(x₂ + x₃) + x₂^2*(x₁ + x₃) + x₃^2*(x₁ + x₂)) - 12*x₁*x₂*x₃

/-- The function g as defined in the problem -/
noncomputable def g (r s t : ℝ) : ℝ :=
  ⨆ (x₃ : ℝ) (h : t ≤ x₃ ∧ x₃ ≤ t + 2), |f r (r + 2) x₃ + s|

/-- The main theorem stating the minimum value of g -/
theorem min_value_of_g :
  (∀ r s t : ℝ, g r s t ≥ 12 * Real.sqrt 3) ∧
  (∃ r₀ s₀ t₀ : ℝ, g r₀ s₀ t₀ = 12 * Real.sqrt 3) :=
sorry

end min_value_of_g_l2179_217972


namespace weight_of_replaced_person_l2179_217912

/-- Proves that the weight of a replaced person is 66 kg given the conditions of the problem -/
theorem weight_of_replaced_person
  (n : ℕ) -- number of people in the initial group
  (avg_increase : ℝ) -- increase in average weight
  (new_weight : ℝ) -- weight of the new person
  (h1 : n = 8) -- there are 8 persons initially
  (h2 : avg_increase = 2.5) -- the average weight increases by 2.5 kg
  (h3 : new_weight = 86) -- the weight of the new person is 86 kg
  : ∃ (replaced_weight : ℝ), replaced_weight = 66 := by
  sorry

end weight_of_replaced_person_l2179_217912


namespace widget_production_difference_l2179_217962

/-- Represents the number of widgets produced by David on Tuesday and Wednesday -/
def widget_difference (t : ℝ) : ℝ :=
  let w := 3 * t  -- Tuesday's production rate
  let tuesday_production := w * t
  let wednesday_production := (w + 3) * (t - 3) * 0.9
  tuesday_production - wednesday_production

/-- Theorem stating the difference in widget production between Tuesday and Wednesday -/
theorem widget_production_difference (t : ℝ) :
  widget_difference t = 0.3 * t^2 + 5.4 * t + 8.1 := by
  sorry


end widget_production_difference_l2179_217962


namespace students_liking_new_menu_l2179_217953

theorem students_liking_new_menu (total_students : ℕ) (disliking_students : ℕ) 
  (h1 : total_students = 400) 
  (h2 : disliking_students = 165) : 
  total_students - disliking_students = 235 := by
  sorry

end students_liking_new_menu_l2179_217953


namespace geometric_sequence_sixth_term_l2179_217934

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (a₁ : ℝ), r ≠ 0 ∧ ∀ n : ℕ, a n = a₁ * r^(n-1)

theorem geometric_sequence_sixth_term 
  (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_sum1 : a 2 + a 4 = 20) 
  (h_sum2 : a 3 + a 5 = 40) : 
  a 6 = 64 := by
  sorry


end geometric_sequence_sixth_term_l2179_217934


namespace total_envelopes_l2179_217955

def blue_envelopes : ℕ := 120
def yellow_envelopes : ℕ := blue_envelopes - 25
def green_envelopes : ℕ := 5 * yellow_envelopes

theorem total_envelopes : blue_envelopes + yellow_envelopes + green_envelopes = 690 := by
  sorry

end total_envelopes_l2179_217955


namespace same_root_implies_a_equals_three_l2179_217954

theorem same_root_implies_a_equals_three (a : ℝ) : 
  (∃ x : ℝ, 3 * x - 2 * a = 0 ∧ 2 * x + 3 * a - 13 = 0) → a = 3 :=
by
  sorry

end same_root_implies_a_equals_three_l2179_217954


namespace smallest_positive_angle_solution_l2179_217920

/-- The equation that needs to be satisfied -/
def equation (y : ℝ) : Prop :=
  6 * Real.sin y * (Real.cos y)^3 - 6 * (Real.sin y)^3 * Real.cos y = 1

/-- The smallest positive angle in degrees that satisfies the equation -/
def smallest_angle : ℝ := 10.4525

theorem smallest_positive_angle_solution :
  equation (smallest_angle * π / 180) ∧
  ∀ y, 0 < y ∧ y < smallest_angle * π / 180 → ¬equation y :=
sorry

end smallest_positive_angle_solution_l2179_217920


namespace no_all_ones_sum_l2179_217989

def has_no_zero_digit (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0

def is_rearrangement (n m : ℕ) : Prop :=
  n.digits 10 ≠ [] ∧ Multiset.ofList (n.digits 10) = Multiset.ofList (m.digits 10)

def all_ones (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 1

theorem no_all_ones_sum (N : ℕ) (hN : has_no_zero_digit N) :
  ∀ M : ℕ, is_rearrangement N M → ¬ all_ones (N + M) :=
sorry

end no_all_ones_sum_l2179_217989


namespace trajectory_equation_of_P_l2179_217926

/-- The trajectory equation of point P on the xOy plane, given its distance from A(0,0,4) -/
theorem trajectory_equation_of_P (P : ℝ × ℝ) (d : ℝ → ℝ → ℝ → ℝ → ℝ) :
  (∀ z, d P.1 P.2 0 z = d P.1 P.2 0 0) →  -- P is on the xOy plane
  d P.1 P.2 0 4 = 5 →                     -- distance between P and A is 5
  P.1^2 + P.2^2 = 9 :=                    -- trajectory equation
by sorry

end trajectory_equation_of_P_l2179_217926


namespace solution_set_f_geq_2_min_value_f_f_equals_one_condition_l2179_217993

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for part I
theorem solution_set_f_geq_2 :
  {x : ℝ | f (x + 2) ≥ 2} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 1/2} := by sorry

-- Theorem for part II
theorem min_value_f :
  ∀ x : ℝ, f x ≥ 1 := by sorry

-- Theorem for the condition when f(x) = 1
theorem f_equals_one_condition (x : ℝ) :
  f x = 1 ↔ 1 ≤ x ∧ x ≤ 2 := by sorry

end solution_set_f_geq_2_min_value_f_f_equals_one_condition_l2179_217993


namespace complex_modulus_l2179_217901

theorem complex_modulus (z : ℂ) : z = -1 + Complex.I * Real.sqrt 3 → Complex.abs z = 2 := by
  sorry

end complex_modulus_l2179_217901


namespace toy_football_sales_performance_toy_football_sales_performance_equality_l2179_217995

/-- Represents the sales performance of two students selling toy footballs --/
theorem toy_football_sales_performance
  (x y z : ℝ)  -- Prices of toy footballs in three sessions
  (hx : x > 0) (hy : y > 0) (hz : z > 0)  -- Prices are positive
  : (x + y + z) / 3 ≥ 3 / (1/x + 1/y + 1/z) := by
  sorry

/-- Equality condition for the sales performance --/
theorem toy_football_sales_performance_equality
  (x y z : ℝ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  : (x + y + z) / 3 = 3 / (1/x + 1/y + 1/z) ↔ x = y ∧ y = z := by
  sorry

end toy_football_sales_performance_toy_football_sales_performance_equality_l2179_217995
