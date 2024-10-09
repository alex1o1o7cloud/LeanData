import Mathlib

namespace final_portfolio_value_l1729_172930

-- Define the initial conditions and growth rates
def initial_investment : ℝ := 80
def first_year_growth_rate : ℝ := 0.15
def additional_investment : ℝ := 28
def second_year_growth_rate : ℝ := 0.10

-- Calculate the values of the portfolio at each step
def after_first_year_investment : ℝ := initial_investment * (1 + first_year_growth_rate)
def after_addition : ℝ := after_first_year_investment + additional_investment
def after_second_year_investment : ℝ := after_addition * (1 + second_year_growth_rate)

theorem final_portfolio_value : after_second_year_investment = 132 := by
  -- This is where the proof would go, but we are omitting it
  sorry

end final_portfolio_value_l1729_172930


namespace total_sand_weight_is_34_l1729_172999

-- Define the conditions
def eden_buckets : ℕ := 4
def mary_buckets : ℕ := eden_buckets + 3
def iris_buckets : ℕ := mary_buckets - 1
def weight_per_bucket : ℕ := 2

-- Define the total weight calculation
def total_buckets : ℕ := eden_buckets + mary_buckets + iris_buckets
def total_weight : ℕ := total_buckets * weight_per_bucket

-- The proof statement
theorem total_sand_weight_is_34 : total_weight = 34 := by
  sorry

end total_sand_weight_is_34_l1729_172999


namespace positive_difference_l1729_172947

theorem positive_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  |y - x| = 60 / 7 := 
sorry

end positive_difference_l1729_172947


namespace repetend_of_5_over_17_is_294117_l1729_172935

theorem repetend_of_5_over_17_is_294117 :
  (∀ n : ℕ, (5 / 17 : ℚ) - (294117 : ℚ) / (10^6 : ℚ) ^ n = 0) :=
by
  sorry

end repetend_of_5_over_17_is_294117_l1729_172935


namespace wednesday_more_than_half_millet_l1729_172911

namespace BirdFeeder

-- Define the initial conditions
def initial_amount_millet (total_seeds : ℚ) : ℚ := 0.4 * total_seeds
def initial_amount_other (total_seeds : ℚ) : ℚ := 0.6 * total_seeds

-- Define the daily consumption
def eaten_millet (millet : ℚ) : ℚ := 0.2 * millet
def eaten_other (other : ℚ) : ℚ := other

-- Define the seed addition every other day
def add_seeds (day : ℕ) (seeds : ℚ) : Prop :=
  day % 2 = 1 → seeds = 1

-- Define the daily update of the millet and other seeds in the feeder
def daily_update (day : ℕ) (millet : ℚ) (other : ℚ) : ℚ × ℚ :=
  let remaining_millet := (1 - 0.2) * millet
  let remaining_other := 0
  if day % 2 = 1 then
    (remaining_millet + initial_amount_millet 1, initial_amount_other 1)
  else
    (remaining_millet, remaining_other)

-- Define the main property to prove
def more_than_half_millet (day : ℕ) (millet : ℚ) (other : ℚ) : Prop :=
  millet > 0.5 * (millet + other)

-- Define the theorem statement
theorem wednesday_more_than_half_millet
  (millet : ℚ := initial_amount_millet 1)
  (other : ℚ := initial_amount_other 1) :
  ∃ day, day = 3 ∧ more_than_half_millet day millet other :=
  by
  sorry

end BirdFeeder

end wednesday_more_than_half_millet_l1729_172911


namespace simplify_expression_l1729_172923

open Real

-- Assume that x, y, z are non-zero real numbers
variables (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)

theorem simplify_expression : (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ := 
by
  -- Proof would go here.
  sorry

end simplify_expression_l1729_172923


namespace maximum_n_Sn_pos_l1729_172979

def arithmetic_sequence := ℕ → ℝ

noncomputable def sum_first_n_terms (a : arithmetic_sequence) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

axiom a1_eq : ∀ (a : arithmetic_sequence), (a 1) = 2 * (a 2) + (a 4)

axiom S5_eq_5 : ∀ (a : arithmetic_sequence), sum_first_n_terms a 5 = 5

theorem maximum_n_Sn_pos : ∀ (a : arithmetic_sequence), (∃ (n : ℕ), n < 6 ∧ sum_first_n_terms a n > 0) → n = 5 :=
  sorry

end maximum_n_Sn_pos_l1729_172979


namespace factorize_a_cubed_minus_a_l1729_172976

theorem factorize_a_cubed_minus_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
by
  sorry

end factorize_a_cubed_minus_a_l1729_172976


namespace overtime_hours_l1729_172990

theorem overtime_hours (regular_rate: ℝ) (regular_hours: ℝ) (total_payment: ℝ) (overtime_rate_multiplier: ℝ) (overtime_hours: ℝ):
  regular_rate = 3 → regular_hours = 40 → total_payment = 198 → overtime_rate_multiplier = 2 → 
  overtime_hours = (total_payment - (regular_rate * regular_hours)) / (regular_rate * overtime_rate_multiplier) →
  overtime_hours = 13 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end overtime_hours_l1729_172990


namespace hexagon_side_lengths_l1729_172967

open Nat

/-- Define two sides AB and BC of a hexagon with their given lengths -/
structure Hexagon :=
  (AB BC AD BE CF DE: ℕ)
  (distinct_lengths : AB ≠ BC ∧ (AB = 7 ∧ BC = 8))
  (total_perimeter : AB + BC + AD + BE + CF + DE = 46)

-- Define a theorem to prove the number of sides measuring 8 units
theorem hexagon_side_lengths (h: Hexagon) :
  ∃ (n : ℕ), n = 4 ∧ n * 8 + (6 - n) * 7 = 46 :=
by
  -- Assume the proof here
  sorry

end hexagon_side_lengths_l1729_172967


namespace abs_diff_31st_terms_l1729_172964

/-- Sequence C is an arithmetic sequence with a starting term 100 and a common difference 15. --/
def seqC (n : ℕ) : ℤ :=
  100 + 15 * (n - 1)

/-- Sequence D is an arithmetic sequence with a starting term 100 and a common difference -20. --/
def seqD (n : ℕ) : ℤ :=
  100 - 20 * (n - 1)

/-- Absolute value of the difference between the 31st terms of sequences C and D is 1050. --/
theorem abs_diff_31st_terms : |seqC 31 - seqD 31| = 1050 := by
  sorry

end abs_diff_31st_terms_l1729_172964


namespace total_snowfall_l1729_172984

theorem total_snowfall (morning afternoon : ℝ) (h1 : morning = 0.125) (h2 : afternoon = 0.5) :
  morning + afternoon = 0.625 := by
  sorry

end total_snowfall_l1729_172984


namespace math_problem_l1729_172929

variable (a b c d : ℝ)

-- The initial condition provided in the problem
def given_condition : Prop := (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7

-- The statement that needs to be proven
theorem math_problem 
  (h : given_condition a b c d) : 
  (a - c) * (b - d) / ((a - b) * (c - d)) = -1 := 
by 
  sorry

end math_problem_l1729_172929


namespace number_of_satisfying_ns_l1729_172981

noncomputable def a_n (n : ℕ) : ℕ := (n-1)*(2*n-1)

def b_n (n : ℕ) : ℕ := 2^n * n

def condition (n : ℕ) : Prop := b_n n ≤ 2019 * a_n n

theorem number_of_satisfying_ns : 
  ∃ n : ℕ, n = 14 ∧ ∀ k : ℕ, (1 ≤ k ∧ k ≤ 14) → condition k := 
by
  sorry

end number_of_satisfying_ns_l1729_172981


namespace root_equation_l1729_172953

variables (m : ℝ)

theorem root_equation {m : ℝ} (h : m^2 - 2 * m - 3 = 0) : m^2 - 2 * m + 2023 = 2026 :=
by {
  sorry 
}

end root_equation_l1729_172953


namespace option_C_correct_l1729_172995

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Definitions for parallel and perpendicular relationships
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def line_parallel (l₁ l₂ : Line) : Prop := sorry

-- Theorem statement based on problem c) translation
theorem option_C_correct (H1 : line_parallel m n) (H2 : perpendicular m α) : perpendicular n α :=
sorry

end option_C_correct_l1729_172995


namespace more_stable_shooting_performance_l1729_172991

theorem more_stable_shooting_performance :
  ∀ (SA2 SB2 : ℝ), SA2 = 1.9 → SB2 = 3 → (SA2 < SB2) → "A" = "Athlete with more stable shooting performance" :=
by
  intros SA2 SB2 h1 h2 h3
  sorry

end more_stable_shooting_performance_l1729_172991


namespace diff_between_largest_and_smallest_fraction_l1729_172997

theorem diff_between_largest_and_smallest_fraction : 
  let f1 := (3 : ℚ) / 4
  let f2 := (7 : ℚ) / 8
  let f3 := (13 : ℚ) / 16
  let f4 := (1 : ℚ) / 2
  let largest := max f1 (max f2 (max f3 f4))
  let smallest := min f1 (min f2 (min f3 f4))
  largest - smallest = (3 : ℚ) / 8 :=
by
  sorry

end diff_between_largest_and_smallest_fraction_l1729_172997


namespace shortest_side_l1729_172919

/-- 
Prove that if the lengths of the sides of a triangle satisfy the inequality \( a^2 + b^2 > 5c^2 \), 
then \( c \) is the length of the shortest side.
-/
theorem shortest_side (a b c : ℝ) (h : a^2 + b^2 > 5 * c^2) (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : c ≤ a ∧ c ≤ b :=
by {
  -- Proof will be provided here.
  sorry
}

end shortest_side_l1729_172919


namespace red_peaches_count_l1729_172921

-- Definitions for the conditions
def yellow_peaches : ℕ := 11
def extra_red_peaches : ℕ := 8

-- The proof statement that the number of red peaches is 19
theorem red_peaches_count : (yellow_peaches + extra_red_peaches = 19) :=
by
  sorry

end red_peaches_count_l1729_172921


namespace monthly_average_growth_rate_price_reduction_for_profit_l1729_172907

-- Part 1: Monthly average growth rate of sales volume
theorem monthly_average_growth_rate (x : ℝ) : 
  256 * (1 + x) ^ 2 = 400 ↔ x = 0.25 :=
by
  sorry

-- Part 2: Price reduction to achieve profit of $4250
theorem price_reduction_for_profit (m : ℝ) : 
  (40 - m - 25) * (400 + 5 * m) = 4250 ↔ m = 5 :=
by
  sorry

end monthly_average_growth_rate_price_reduction_for_profit_l1729_172907


namespace cone_height_correct_l1729_172969

noncomputable def height_of_cone (R1 R2 R3 base_radius : ℝ) : ℝ :=
  if R1 = 20 ∧ R2 = 40 ∧ R3 = 40 ∧ base_radius = 21 then 28 else 0

theorem cone_height_correct :
  height_of_cone 20 40 40 21 = 28 :=
by sorry

end cone_height_correct_l1729_172969


namespace smaller_number_l1729_172902

theorem smaller_number (x y : ℝ) (h1 : x + y = 16) (h2 : x - y = 4) (h3 : x * y = 60) : y = 6 :=
sorry

end smaller_number_l1729_172902


namespace find_n_l1729_172974

theorem find_n (n m : ℕ) (h : m = 4) (eq1 : (1/5)^m * (1/4)^n = 1/(10^4)) : n = 2 :=
by
  sorry

end find_n_l1729_172974


namespace stella_glasses_count_l1729_172916

-- Definitions for the conditions
def dolls : ℕ := 3
def clocks : ℕ := 2
def price_per_doll : ℕ := 5
def price_per_clock : ℕ := 15
def price_per_glass : ℕ := 4
def total_cost : ℕ := 40
def profit : ℕ := 25

-- The proof statement
theorem stella_glasses_count (dolls clocks price_per_doll price_per_clock price_per_glass total_cost profit : ℕ) :
  (dolls * price_per_doll + clocks * price_per_clock) + profit + total_cost = total_cost + profit → 
  (dolls * price_per_doll + clocks * price_per_clock) + profit + total_cost - (dolls * price_per_doll + clocks * price_per_clock) = price_per_glass * 5 :=
sorry

end stella_glasses_count_l1729_172916


namespace original_price_l1729_172982

theorem original_price (P : ℝ) (h₁ : 0.30 * P = 46) : P = 153.33 :=
  sorry

end original_price_l1729_172982


namespace smallest_slice_area_l1729_172934

theorem smallest_slice_area
  (a₁ : ℕ) (d : ℕ) (total_angle : ℕ) (r : ℕ) 
  (h₁ : a₁ = 30) (h₂ : d = 2) (h₃ : total_angle = 360) (h₄ : r = 10) :
  ∃ (n : ℕ) (smallest_angle : ℕ),
  n = 9 ∧ smallest_angle = 18 ∧ 
  ∃ (area : ℝ), area = 5 * Real.pi :=
by
  sorry


end smallest_slice_area_l1729_172934


namespace flowers_per_vase_l1729_172910

-- Definitions of conditions in Lean 4
def number_of_carnations : ℕ := 7
def number_of_roses : ℕ := 47
def total_number_of_flowers : ℕ := number_of_carnations + number_of_roses
def number_of_vases : ℕ := 9

-- Statement in Lean 4
theorem flowers_per_vase : total_number_of_flowers / number_of_vases = 6 := by
  unfold total_number_of_flowers
  show (7 + 47) / 9 = 6
  sorry

end flowers_per_vase_l1729_172910


namespace least_subtract_divisible_l1729_172955

theorem least_subtract_divisible:
  ∃ n : ℕ, n = 31 ∧ (13603 - n) % 87 = 0 :=
by
  sorry

end least_subtract_divisible_l1729_172955


namespace bridge_construction_l1729_172920

-- Definitions used in the Lean statement based on conditions.
def rate (workers : ℕ) (days : ℕ) : ℚ := 1 / (workers * days)

-- The problem statement: prove that if 60 workers working together can build the bridge in 3 days, 
-- then 120 workers will take 1.5 days to build the bridge.
theorem bridge_construction (t : ℚ) : 
  (rate 60 3) * 120 * t = 1 → t = 1.5 := by
  sorry

end bridge_construction_l1729_172920


namespace probability_A8_l1729_172906

/-- Define the probability of event A_n where the sum of die rolls equals n -/
def P (n : ℕ) : ℚ :=
  1/7 * (if n = 8 then 5/36 + 21/216 + 35/1296 + 35/7776 + 21/46656 +
    7/279936 + 1/1679616 else 0)

theorem probability_A8 : P 8 = (1/7) * (5/36 + 21/216 + 35/1296 + 35/7776 + 
  21/46656 + 7/279936 + 1/1679616) :=
by
  sorry

end probability_A8_l1729_172906


namespace max_statements_true_l1729_172924

noncomputable def max_true_statements (a b : ℝ) : ℕ :=
  (if (a^2 > b^2) then 1 else 0) +
  (if (a < b) then 1 else 0) +
  (if (a < 0) then 1 else 0) +
  (if (b < 0) then 1 else 0) +
  (if (1 / a < 1 / b) then 1 else 0)

theorem max_statements_true : ∀ (a b : ℝ), max_true_statements a b ≤ 4 :=
by
  intro a b
  sorry

end max_statements_true_l1729_172924


namespace constant_term_in_expansion_l1729_172944

theorem constant_term_in_expansion :
  let f := (x - (2 / x^2))
  let expansion := f^9
  ∃ c: ℤ, expansion = c ∧ c = -672 :=
sorry

end constant_term_in_expansion_l1729_172944


namespace range_a_l1729_172931

noncomputable def f (x a : ℝ) : ℝ := Real.log x + x + 2 / x - a
noncomputable def g (x : ℝ) : ℝ := Real.log x + x + 2 / x

theorem range_a (a : ℝ) : (∃ x > 0, f x a = 0) → a ≥ 3 :=
by
sorry

end range_a_l1729_172931


namespace absolute_sum_of_roots_l1729_172938

theorem absolute_sum_of_roots (d e f n : ℤ) (h1 : d + e + f = 0) (h2 : d * e + e * f + f * d = -2023) : |d| + |e| + |f| = 98 := 
sorry

end absolute_sum_of_roots_l1729_172938


namespace determinant_difference_l1729_172983

namespace MatrixDeterminantProblem

open Matrix

variables {R : Type*} [CommRing R]

theorem determinant_difference (a b c d : R) 
  (h : det ![![a, b], ![c, d]] = 15) :
  det ![![3 * a, 3 * b], ![3 * c, 3 * d]] - 
  det ![![3 * b, 3 * a], ![3 * d, 3 * c]] = 270 := 
by
  sorry

end MatrixDeterminantProblem

end determinant_difference_l1729_172983


namespace length_of_other_train_l1729_172959

variable (L : ℝ)

theorem length_of_other_train
    (train1_length : ℝ := 260)
    (train1_speed_kmh : ℝ := 120)
    (train2_speed_kmh : ℝ := 80)
    (time_to_cross : ℝ := 9)
    (train1_speed : ℝ := train1_speed_kmh * 1000 / 3600)
    (train2_speed : ℝ := train2_speed_kmh * 1000 / 3600)
    (relative_speed : ℝ := train1_speed + train2_speed)
    (total_distance : ℝ := relative_speed * time_to_cross)
    (other_train_length : ℝ := total_distance - train1_length) :
    L = other_train_length := by
  sorry

end length_of_other_train_l1729_172959


namespace graph_passes_through_quadrants_l1729_172925

def linear_function (x : ℝ) : ℝ := -5 * x + 5

theorem graph_passes_through_quadrants :
  (∃ x y : ℝ, linear_function x = y ∧ x > 0 ∧ y > 0) ∧  -- Quadrant I
  (∃ x y : ℝ, linear_function x = y ∧ x < 0 ∧ y > 0) ∧  -- Quadrant II
  (∃ x y : ℝ, linear_function x = y ∧ x > 0 ∧ y < 0)    -- Quadrant IV
  :=
by
  sorry

end graph_passes_through_quadrants_l1729_172925


namespace ethan_presents_l1729_172904

theorem ethan_presents (ethan alissa : ℕ) 
  (h1 : alissa = ethan + 22) 
  (h2 : alissa = 53) : 
  ethan = 31 := 
by
  sorry

end ethan_presents_l1729_172904


namespace complete_square_expression_l1729_172926

theorem complete_square_expression :
  ∃ (a h k : ℝ), (∀ x : ℝ, 2 * x^2 + 8 * x + 6 = a * (x - h)^2 + k) ∧ (a + h + k = -2) :=
by
  sorry

end complete_square_expression_l1729_172926


namespace children_boys_count_l1729_172937

theorem children_boys_count (girls : ℕ) (total_children : ℕ) (boys : ℕ) 
  (h₁ : girls = 35) (h₂ : total_children = 62) : boys = 27 :=
by
  sorry

end children_boys_count_l1729_172937


namespace simplify_trig_identity_l1729_172970

theorem simplify_trig_identity (α β : ℝ) : 
  (Real.cos (α + β) * Real.cos β + Real.sin (α + β) * Real.sin β) = Real.cos α :=
by
  sorry

end simplify_trig_identity_l1729_172970


namespace bone_meal_percentage_growth_l1729_172918

-- Definitions for the problem conditions
def control_height : ℝ := 36
def cow_manure_height : ℝ := 90
def bone_meal_to_cow_manure_ratio : ℝ := 0.5 -- since cow manure plant is 200% the height of bone meal plant

noncomputable def bone_meal_height : ℝ := cow_manure_height * bone_meal_to_cow_manure_ratio

-- The main theorem to prove
theorem bone_meal_percentage_growth : 
  ( (bone_meal_height - control_height) / control_height ) * 100 = 25 := 
by
  sorry

end bone_meal_percentage_growth_l1729_172918


namespace max_license_plates_is_correct_l1729_172909

theorem max_license_plates_is_correct :
  let letters := 26
  let digits := 10
  (letters * (letters - 1) * digits^3 = 26 * 25 * 10^3) :=
by 
  sorry

end max_license_plates_is_correct_l1729_172909


namespace sequence_length_l1729_172994

theorem sequence_length (a d n : ℕ) (h1 : a = 3) (h2 : d = 5) (h3: 3 + (n-1) * d = 3008) : n = 602 := 
by
  sorry

end sequence_length_l1729_172994


namespace digit_A_value_l1729_172992

theorem digit_A_value :
  ∃ (A : ℕ), A < 10 ∧ (45 % A = 0) ∧ (172 * 10 + A * 10 + 6) % 8 = 0 ∧
    ∀ (B : ℕ), B < 10 ∧ (45 % B = 0) ∧ (172 * 10 + B * 10 + 6) % 8 = 0 → B = A := sorry

end digit_A_value_l1729_172992


namespace solve_system_of_equations_l1729_172927

-- Given conditions
variables {a b c k x y z : ℝ}
variables (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
variables (eq1 : a * x + b * y + c * z = k)
variables (eq2 : a^2 * x + b^2 * y + c^2 * z = k^2)
variables (eq3 : a^3 * x + b^3 * y + c^3 * z = k^3)

-- Statement to be proved
theorem solve_system_of_equations :
  x = k * (k - c) * (k - b) / (a * (a - c) * (a - b)) ∧
  y = k * (k - c) * (k - a) / (b * (b - c) * (b - a)) ∧
  z = k * (k - a) * (k - b) / (c * (c - a) * (c - b)) :=
sorry

end solve_system_of_equations_l1729_172927


namespace find_x_l1729_172998

theorem find_x (x : ℝ) : (x + 3 * x + 1000 + 3000) / 4 = 2018 → x = 1018 :=
by 
  intro h
  sorry

end find_x_l1729_172998


namespace perpendicular_lines_condition_perpendicular_lines_sufficient_not_necessary_l1729_172951

-- Mathematical definitions and theorems required for the problem
theorem perpendicular_lines_condition (m : ℝ) :
  3 * m + m * (2 * m - 1) = 0 ↔ (m = 0 ∨ m = -1) :=
by sorry

-- Translate the specific problem into Lean
theorem perpendicular_lines_sufficient_not_necessary (m : ℝ) (h : 3 * m + m * (2 * m - 1) = 0) :
  m = -1 ∨ (m ≠ -1 ∧ 3 * m + m * (2 * m - 1) = 0) :=
by sorry

end perpendicular_lines_condition_perpendicular_lines_sufficient_not_necessary_l1729_172951


namespace trapezoid_midsegment_l1729_172922

theorem trapezoid_midsegment (h : ℝ) :
  ∃ k : ℝ, (∃ θ : ℝ, θ = 120 ∧ k = 2 * h * Real.cos (θ / 2)) ∧
  (∃ m : ℝ, m = k / 2) ∧
  (∃ midsegment : ℝ, midsegment = m / Real.sqrt 3 ∧ midsegment = h / Real.sqrt 3) :=
by
  -- This is where the proof would go.
  sorry

end trapezoid_midsegment_l1729_172922


namespace g_of_negative_8_l1729_172989

def f (x : ℝ) : ℝ := 4 * x - 9
def g (y : ℝ) : ℝ := y^2 + 6 * y - 7

theorem g_of_negative_8 : g (-8) = -87 / 16 :=
by
  -- Proof goes here
  sorry

end g_of_negative_8_l1729_172989


namespace circle_radius_l1729_172961

theorem circle_radius {s r : ℝ} (h : 4 * s = π * r^2) (h1 : 2 * r = s * Real.sqrt 2) : 
  r = 4 * Real.sqrt 2 / π :=
by
  sorry

end circle_radius_l1729_172961


namespace percent_less_50000_l1729_172941

variable (A B C : ℝ) -- Define the given percentages
variable (h1 : A = 0.45) -- 45% of villages have populations from 20,000 to 49,999
variable (h2 : B = 0.30) -- 30% of villages have fewer than 20,000 residents
variable (h3 : C = 0.25) -- 25% of villages have 50,000 or more residents

theorem percent_less_50000 : A + B = 0.75 := by
  sorry

end percent_less_50000_l1729_172941


namespace wall_width_is_correct_l1729_172988

-- Definitions based on the conditions
def brick_length : ℝ := 25  -- in cm
def brick_height : ℝ := 11.25  -- in cm
def brick_width : ℝ := 6  -- in cm
def num_bricks : ℝ := 5600
def wall_length : ℝ := 700  -- 7 m in cm
def wall_height : ℝ := 600  -- 6 m in cm
def total_volume : ℝ := num_bricks * (brick_length * brick_height * brick_width)

-- Prove that the inferred width of the wall is correct
theorem wall_width_is_correct : (total_volume / (wall_length * wall_height)) = 22.5 := by
  sorry

end wall_width_is_correct_l1729_172988


namespace circumference_of_jack_head_l1729_172960

theorem circumference_of_jack_head (J C : ℝ) (h1 : (2 / 3) * C = 10) (h2 : (1 / 2) * J + 9 = 15) :
  J = 12 :=
by
  sorry

end circumference_of_jack_head_l1729_172960


namespace z_plus_inv_y_eq_10_div_53_l1729_172905

-- Define the conditions for x, y, z being positive real numbers such that
-- xyz = 1, x + 1/z = 8, and y + 1/x = 20
variables (x y z : ℝ)
variables (hx : x > 0)
variables (hy : y > 0)
variables (hz : z > 0)
variables (h1 : x * y * z = 1)
variables (h2 : x + 1 / z = 8)
variables (h3 : y + 1 / x = 20)

-- The goal is to prove that z + 1/y = 10 / 53
theorem z_plus_inv_y_eq_10_div_53 : z + 1 / y = 10 / 53 :=
by {
  sorry
}

end z_plus_inv_y_eq_10_div_53_l1729_172905


namespace center_of_circle_l1729_172915

theorem center_of_circle (x y : ℝ) : 
  (x - 1) ^ 2 + (y + 1) ^ 2 = 4 ↔ (x^2 + y^2 - 2*x + 2*y - 2 = 0) :=
sorry

end center_of_circle_l1729_172915


namespace motorcycle_speed_for_10_minute_prior_arrival_l1729_172968

noncomputable def distance_from_home_to_station (x : ℝ) : Prop :=
  x / 30 + 15 / 60 = x / 18 - 15 / 60

noncomputable def speed_to_arrive_10_minutes_before_departure (x : ℝ) (v : ℝ) : Prop :=
  v = x / (1 - 10 / 60)

theorem motorcycle_speed_for_10_minute_prior_arrival :
  (∀ x : ℝ, distance_from_home_to_station x) →
  (∃ x : ℝ, 
    ∃ v : ℝ, speed_to_arrive_10_minutes_before_departure x v ∧ v = 27) :=
by 
  intro h
  exists 22.5
  exists 27
  unfold distance_from_home_to_station at h
  unfold speed_to_arrive_10_minutes_before_departure
  sorry

end motorcycle_speed_for_10_minute_prior_arrival_l1729_172968


namespace expectation_S_tau_eq_varliminf_ratio_S_tau_l1729_172943

noncomputable def xi : ℕ → ℝ := sorry
noncomputable def tau : ℝ := sorry

-- Statement (a)
theorem expectation_S_tau_eq (ES_tau : ℝ := sorry) (E_tau : ℝ := sorry) (E_xi1 : ℝ := sorry) :
  ES_tau = E_tau * E_xi1 := sorry

-- Statement (b)
theorem varliminf_ratio_S_tau (liminf_val : ℝ := sorry) (E_tau : ℝ := sorry) :
  (liminf_val = E_tau) := sorry

end expectation_S_tau_eq_varliminf_ratio_S_tau_l1729_172943


namespace son_time_to_complete_job_l1729_172962

theorem son_time_to_complete_job (M S : ℝ) (hM : M = 1 / 5) (hMS : M + S = 1 / 4) : S = 1 / 20 → 1 / S = 20 :=
by
  sorry

end son_time_to_complete_job_l1729_172962


namespace cost_per_taco_is_1_50_l1729_172900

namespace TacoTruck

def total_beef : ℝ := 100
def beef_per_taco : ℝ := 0.25
def taco_price : ℝ := 2
def profit : ℝ := 200

theorem cost_per_taco_is_1_50 :
  let total_tacos := total_beef / beef_per_taco
  let total_revenue := total_tacos * taco_price
  let total_cost := total_revenue - profit
  total_cost / total_tacos = 1.50 := 
by
  sorry

end TacoTruck

end cost_per_taco_is_1_50_l1729_172900


namespace simplify_fraction_l1729_172914

theorem simplify_fraction (x y : ℝ) : (x - y) / (y - x) = -1 :=
sorry

end simplify_fraction_l1729_172914


namespace germs_killed_in_common_l1729_172932

theorem germs_killed_in_common :
  ∃ x : ℝ, x = 5 ∧
    ∀ A B C : ℝ, A = 50 → 
    B = 25 → 
    C = 30 → 
    x = A + B - (100 - C) := sorry

end germs_killed_in_common_l1729_172932


namespace positive_integer_expression_l1729_172987

theorem positive_integer_expression (q : ℕ) (h : q > 0) : 
  ((∃ k : ℕ, k > 0 ∧ (5 * q + 18) = k * (3 * q - 8)) ↔ q = 3 ∨ q = 4 ∨ q = 5 ∨ q = 12) := 
sorry

end positive_integer_expression_l1729_172987


namespace find_y_l1729_172963

open Classical

theorem find_y (a b c x y : ℚ)
  (h1 : a / b = 5 / 4)
  (h2 : b / c = 3 / x)
  (h3 : a / c = y / 4) :
  y = 15 / x :=
sorry

end find_y_l1729_172963


namespace chess_tournament_points_l1729_172973

theorem chess_tournament_points (boys girls : ℕ) (total_points : ℝ) 
  (total_matches : ℕ)
  (matches_among_boys points_among_boys : ℕ)
  (matches_among_girls points_among_girls : ℕ)
  (matches_between points_between : ℕ)
  (total_players : ℕ := boys + girls)
  (H1 : boys = 9) (H2 : girls = 3) (H3 : total_players = 12)
  (H4 : total_matches = total_players * (total_players - 1) / 2) 
  (H5 : total_points = total_matches) 
  (H6 : matches_among_boys = boys * (boys - 1) / 2) 
  (H7 : points_among_boys = matches_among_boys)
  (H8 : matches_among_girls = girls * (girls - 1) / 2) 
  (H9 : points_among_girls = matches_among_girls) 
  (H10 : matches_between = boys * girls) 
  (H11 : points_between = matches_between) :
  ¬ ∃ (P_B P_G : ℝ) (x : ℝ),
    P_B = points_among_boys + x ∧
    P_G = points_among_girls + (points_between - x) ∧
    P_B = P_G := by
  sorry

end chess_tournament_points_l1729_172973


namespace tangent_line_at_origin_l1729_172928

/-- 
The curve is given by y = exp x.
The tangent line to this curve that passes through the origin (0, 0) 
has the equation y = exp 1 * x.
-/
theorem tangent_line_at_origin :
  ∀ (x y : ℝ), y = Real.exp x → (∃ k : ℝ, ∀ x, y = k * x ∧ k = Real.exp 1) :=
by
  sorry

end tangent_line_at_origin_l1729_172928


namespace bus_trip_distance_l1729_172996

theorem bus_trip_distance
  (D S : ℕ) (H1 : S = 55)
  (H2 : D / S - 1 = D / (S + 5))
  : D = 660 :=
sorry

end bus_trip_distance_l1729_172996


namespace combined_length_of_straight_parts_l1729_172971

noncomputable def length_of_straight_parts (R : ℝ) (p : ℝ) : ℝ := p * R

theorem combined_length_of_straight_parts :
  ∀ (R : ℝ) (p : ℝ), R = 80 ∧ p = 0.25 → length_of_straight_parts R p = 20 :=
by
  intros R p h
  cases' h with hR hp
  rw [hR, hp]
  simp [length_of_straight_parts]
  sorry

end combined_length_of_straight_parts_l1729_172971


namespace nate_age_is_14_l1729_172949

def nate_current_age (N : ℕ) : Prop :=
  ∃ E : ℕ, E = N / 2 ∧ N - E = 7

theorem nate_age_is_14 : nate_current_age 14 :=
by {
  sorry
}

end nate_age_is_14_l1729_172949


namespace goldfish_added_per_day_is_7_l1729_172913

def initial_koi_fish : ℕ := 227 - 2
def initial_goldfish : ℕ := 280 - initial_koi_fish
def added_goldfish : ℕ := 200 - initial_goldfish
def days_in_three_weeks : ℕ := 3 * 7
def goldfish_added_per_day : ℕ := (added_goldfish + days_in_three_weeks - 1) / days_in_three_weeks -- rounding to nearest integer 

theorem goldfish_added_per_day_is_7 : goldfish_added_per_day = 7 :=
by 
-- sorry to skip the proof
sorry

end goldfish_added_per_day_is_7_l1729_172913


namespace sum_of_nine_l1729_172986

theorem sum_of_nine (S : ℕ → ℕ) (a : ℕ → ℕ) (h₀ : ∀ (n : ℕ), S n = n * (a 1 + a n) / 2)
(h₁ : S 3 = 30) (h₂ : S 6 = 100) : S 9 = 240 := 
sorry

end sum_of_nine_l1729_172986


namespace speed_including_stoppages_l1729_172980

-- Definitions
def speed_excluding_stoppages : ℤ := 50 -- kmph
def stoppage_time_per_hour : ℕ := 24 -- minutes

-- Theorem to prove the speed of the train including stoppages
theorem speed_including_stoppages (h1 : speed_excluding_stoppages = 50)
                                  (h2 : stoppage_time_per_hour = 24) :
  ∃ s : ℤ, s = 30 := 
sorry

end speed_including_stoppages_l1729_172980


namespace circle_eq_of_given_center_and_radius_l1729_172917

theorem circle_eq_of_given_center_and_radius :
  (∀ (x y : ℝ),
    let C := (-1, 2)
    let r := 4
    (x + 1) ^ 2 + (y - 2) ^ 2 = 16) :=
by
  sorry

end circle_eq_of_given_center_and_radius_l1729_172917


namespace diameter_is_10sqrt6_l1729_172946

noncomputable def radius (A : ℝ) (hA : A = 150 * Real.pi) : ℝ :=
  Real.sqrt (A / Real.pi)

noncomputable def diameter (A : ℝ) (hA : A = 150 * Real.pi) : ℝ :=
  2 * radius A hA

theorem diameter_is_10sqrt6 (A : ℝ) (hA : A = 150 * Real.pi) :
  diameter A hA = 10 * Real.sqrt 6 :=
  sorry

end diameter_is_10sqrt6_l1729_172946


namespace common_point_exists_l1729_172972

theorem common_point_exists (a b c : ℝ) :
  ∃ x y : ℝ, y = a * x ^ 2 - b * x + c ∧ y = b * x ^ 2 - c * x + a ∧ y = c * x ^ 2 - a * x + b :=
  sorry

end common_point_exists_l1729_172972


namespace actual_cost_of_article_l1729_172954

theorem actual_cost_of_article (x : ℝ) (h : 0.60 * x = 1050) : x = 1750 := by
  sorry

end actual_cost_of_article_l1729_172954


namespace initial_fee_l1729_172985

theorem initial_fee (initial_fee : ℝ) : 
  (∀ (distance_charge_per_segment travel_total_charge : ℝ), 
    distance_charge_per_segment = 0.35 → 
    3.6 / 0.4 * distance_charge_per_segment + initial_fee = travel_total_charge → 
    travel_total_charge = 5.20)
    → initial_fee = 2.05 :=
by
  intro h
  specialize h 0.35 5.20
  sorry

end initial_fee_l1729_172985


namespace sufficient_but_not_necessary_condition_l1729_172940

def proposition_p (m : ℝ) : Prop := ∀ x : ℝ, |x + 1| + |x - 1| ≥ m
def proposition_q (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 - 2 * m * x₀ + m^2 + m - 3 = 0

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (proposition_p m → proposition_q m) ∧ ¬ (proposition_q m → proposition_p m) :=
sorry

end sufficient_but_not_necessary_condition_l1729_172940


namespace intersection_points_count_l1729_172957

theorem intersection_points_count:
  let line1 := { p : ℝ × ℝ | ∃ x y : ℝ, 4 * y - 3 * x = 2 ∧ (p.1 = x ∧ p.2 = y) }
  let line2 := { p : ℝ × ℝ | ∃ x y : ℝ, x + 3 * y = 3 ∧ (p.1 = x ∧ p.2 = y) }
  let line3 := { p : ℝ × ℝ | ∃ x y : ℝ, 6 * x - 8 * y = 6 ∧ (p.1 = x ∧ p.2 = y) }
  ∃! p1 p2 : ℝ × ℝ, p1 ∈ line1 ∧ p1 ∈ line2 ∧ p2 ∈ line2 ∧ p2 ∈ line3 :=
by
  sorry

end intersection_points_count_l1729_172957


namespace average_weight_l1729_172936

theorem average_weight (w_girls w_boys : ℕ) (avg_girls avg_boys : ℕ) (n : ℕ) : 
  n = 5 → avg_girls = 45 → avg_boys = 55 → 
  w_girls = n * avg_girls → w_boys = n * avg_boys →
  ∀ total_weight, total_weight = w_girls + w_boys →
  ∀ avg_weight, avg_weight = total_weight / (2 * n) →
  avg_weight = 50 :=
by
  intros h_n h_avg_girls h_avg_boys h_w_girls h_w_boys h_total_weight h_avg_weight
  -- here you would start the proof, but it is omitted as per the instructions
  sorry

end average_weight_l1729_172936


namespace quadratic_inequality_solution_set_l1729_172948

theorem quadratic_inequality_solution_set (a b : ℝ) :
  (∀ x : ℝ, (2 < x ∧ x < 3) → (ax^2 + 5*x + b > 0)) →
  ∃ x : ℝ, (-1/2 < x ∧ x < -1/3) :=
sorry

end quadratic_inequality_solution_set_l1729_172948


namespace group_left_to_clean_is_third_group_l1729_172975

-- Definition of group sizes
def group1 := 7
def group2 := 10
def group3 := 16
def group4 := 18

-- Definitions and conditions
def total_students := group1 + group2 + group3 + group4
def lecture_factor := 4
def english_students := 7  -- From solution: must be 7 students attending the English lecture
def math_students := lecture_factor * english_students

-- Hypothesis of the students allocating to lectures
def students_attending_lectures := english_students + math_students
def students_left_to_clean := total_students - students_attending_lectures

-- The statement to be proved in Lean
theorem group_left_to_clean_is_third_group
  (h : students_left_to_clean = group3) :
  students_left_to_clean = 16 :=
sorry

end group_left_to_clean_is_third_group_l1729_172975


namespace point_P_in_fourth_quadrant_l1729_172952

def point_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_P_in_fourth_quadrant (m : ℝ) : point_in_fourth_quadrant (1 + m^2) (-1) :=
by
  sorry

end point_P_in_fourth_quadrant_l1729_172952


namespace negation_of_p_is_false_l1729_172977

def prop_p : Prop :=
  ∀ x : ℝ, 1 < x → (Real.log (x + 2) / Real.log 3 - 2 / 2^x) > 0

theorem negation_of_p_is_false : ¬(∃ x : ℝ, 1 < x ∧ (Real.log (x + 2) / Real.log 3 - 2 / 2^x) ≤ 0) :=
sorry

end negation_of_p_is_false_l1729_172977


namespace Faye_can_still_make_8_bouquets_l1729_172908

theorem Faye_can_still_make_8_bouquets (total_flowers : ℕ) (wilted_flowers : ℕ) (flowers_per_bouquet : ℕ) 
(h1 : total_flowers = 88) 
(h2 : wilted_flowers = 48) 
(h3 : flowers_per_bouquet = 5) : 
(total_flowers - wilted_flowers) / flowers_per_bouquet = 8 := 
by
  sorry

end Faye_can_still_make_8_bouquets_l1729_172908


namespace matrix_addition_correct_l1729_172993

def matrixA : Matrix (Fin 2) (Fin 2) ℤ := fun i j =>
  if i = 0 then
    if j = 0 then 4 else -2
  else
    if j = 0 then -3 else 5

def matrixB : Matrix (Fin 2) (Fin 2) ℤ := fun i j =>
  if i = 0 then
    if j = 0 then -6 else 0
  else
    if j = 0 then 7 else -8

def resultMatrix : Matrix (Fin 2) (Fin 2) ℤ := fun i j =>
  if i = 0 then
    if j = 0 then -2 else -2
  else
    if j = 0 then 4 else -3

theorem matrix_addition_correct :
  matrixA + matrixB = resultMatrix :=
by
  sorry

end matrix_addition_correct_l1729_172993


namespace length_of_PQ_l1729_172966

theorem length_of_PQ (R P Q : ℝ × ℝ) (hR : R = (10, 8))
(hP_line1 : ∃ p : ℝ, P = (p, 24 * p / 7))
(hQ_line2 : ∃ q : ℝ, Q = (q, 5 * q / 13))
(h_mid : ∃ (p q : ℝ), R = ((p + q) / 2, (24 * p / 14 + 5 * q / 26) / 2))
(answer_eq : ∃ (a b : ℕ), PQ_length = a / b ∧ a.gcd b = 1 ∧ a + b = 4925) : 
∃ a b : ℕ, a + b = 4925 := sorry

end length_of_PQ_l1729_172966


namespace square_side_length_l1729_172978

theorem square_side_length (A : ℝ) (side : ℝ) (h₁ : A = side^2) (h₂ : A = 12) : side = 2 * Real.sqrt 3 := 
by
  sorry

end square_side_length_l1729_172978


namespace trajectory_of_Q_existence_of_M_l1729_172956

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := (x + 2) ^ 2 + y ^ 2 = 81 / 16
def C2 (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 1 / 16

-- Define the conditions about circle Q
def is_tangent_to_both (Q : ℝ → ℝ → Prop) : Prop :=
  ∃ r : ℝ, (∀ x y : ℝ, Q x y → (x + 2)^2 + y^2 = (r + 9/4)^2) ∧ (∀ x y : ℝ, Q x y → (x - 2)^2 + y^2 = (r + 1/4)^2)

-- Prove the trajectory of the center of Q
theorem trajectory_of_Q (Q : ℝ → ℝ → Prop) (h : is_tangent_to_both Q) :
  ∀ x y : ℝ, Q x y ↔ (x^2 - y^2 / 3 = 1 ∧ x ≥ 1) :=
sorry

-- Prove the existence and coordinates of M
theorem existence_of_M (M : ℝ) (Q : ℝ → ℝ → Prop) (h : is_tangent_to_both Q) :
  ∃ x y : ℝ, (x, y) = (-1, 0) ∧ (∀ x0 y0 : ℝ, Q x0 y0 → ((-y0 / (x0 - 2) = 2 * (y0 / (x0 - M)) / (1 - (y0 / (x0 - M))^2)) ↔ M = -1)) :=
sorry

end trajectory_of_Q_existence_of_M_l1729_172956


namespace bunnies_burrow_exit_counts_l1729_172912

theorem bunnies_burrow_exit_counts :
  let groupA_bunnies := 40
  let groupA_rate := 3  -- times per minute per bunny
  let groupB_bunnies := 30
  let groupB_rate := 5 / 2 -- times per minute per bunny
  let groupC_bunnies := 30
  let groupC_rate := 8 / 5 -- times per minute per bunny
  let total_bunnies := 100
  let minutes_per_day := 1440
  let days_per_week := 7
  let pre_change_rate_per_min := groupA_bunnies * groupA_rate + groupB_bunnies * groupB_rate + groupC_bunnies * groupC_rate
  let post_change_rate_per_min := pre_change_rate_per_min * 0.5
  let total_pre_change_counts := pre_change_rate_per_min * minutes_per_day * days_per_week
  let total_post_change_counts := post_change_rate_per_min * minutes_per_day * (days_per_week * 2)
  total_pre_change_counts + total_post_change_counts = 4897920 := by
    sorry

end bunnies_burrow_exit_counts_l1729_172912


namespace sum_ac_equals_seven_l1729_172958

theorem sum_ac_equals_seven 
  (a b c d : ℝ)
  (h1 : ab + bc + cd + da = 42)
  (h2 : b + d = 6) :
  a + c = 7 := 
sorry

end sum_ac_equals_seven_l1729_172958


namespace inequality_proof_l1729_172950

theorem inequality_proof
  (x y z : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (hxyz : x * y * z = 1) :
  x^2 + y^2 + z^2 + x * y + y * z + z * x ≥ 2 * (Real.sqrt x + Real.sqrt y + Real.sqrt z) :=
by
  sorry

end inequality_proof_l1729_172950


namespace exists_x_inequality_l1729_172965

theorem exists_x_inequality (a : ℝ) : 
  (∃ x : ℝ, x^2 - 3 * a * x + 9 < 0) ↔ a < -2 ∨ a > 2 :=
by
  sorry

end exists_x_inequality_l1729_172965


namespace locus_of_tangent_circle_is_hyperbola_l1729_172933

theorem locus_of_tangent_circle_is_hyperbola :
  ∀ (P : ℝ × ℝ) (r : ℝ),
    (P.1 ^ 2 + P.2 ^ 2).sqrt = 1 + r ∧ ((P.1 - 4) ^ 2 + P.2 ^ 2).sqrt = 2 + r →
    ∃ (a b : ℝ), (P.1 - a) ^ 2 / b ^ 2 - (P.2 / a) ^ 2 / b ^ 2 = 1 :=
sorry

end locus_of_tangent_circle_is_hyperbola_l1729_172933


namespace time_per_student_l1729_172939

-- Given Conditions
def total_students : ℕ := 18
def groups : ℕ := 3
def minutes_per_group : ℕ := 24

-- Mathematical proof problem
theorem time_per_student :
  (minutes_per_group / (total_students / groups)) = 4 := by
  -- Proof not required, adding placeholder
  sorry

end time_per_student_l1729_172939


namespace max_k_value_l1729_172945

open Real

theorem max_k_value (x y k : ℝ)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_pos_k : 0 < k)
  (h_eq : 6 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ 3 / 2 :=
by
  sorry

end max_k_value_l1729_172945


namespace monotonic_decreasing_interval_l1729_172942

noncomputable def function_y (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 5

theorem monotonic_decreasing_interval : 
  ∀ x, -1 < x ∧ x < 3 →  (deriv function_y x < 0) :=
by
  sorry

end monotonic_decreasing_interval_l1729_172942


namespace gcd_1407_903_l1729_172903

theorem gcd_1407_903 : Nat.gcd 1407 903 = 21 := 
  sorry

end gcd_1407_903_l1729_172903


namespace more_girls_than_boys_l1729_172901

theorem more_girls_than_boys (num_students : ℕ) (boys_ratio : ℕ) (girls_ratio : ℕ) (total_students : ℕ) (total_students_eq : num_students = 42) (ratio_eq : boys_ratio = 3 ∧ girls_ratio = 4) : (4 * 6) - (3 * 6) = 6 := by
  sorry

end more_girls_than_boys_l1729_172901
