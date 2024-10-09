import Mathlib

namespace ordered_pair_sqrt_l520_52051

/-- Problem statement: Given positive integers a and b such that a < b, prove that:
sqrt (1 + sqrt (40 + 24 * sqrt 5)) = sqrt a + sqrt b, if (a, b) = (1, 6). -/
theorem ordered_pair_sqrt (a b : ℕ) (h1 : a = 1) (h2 : b = 6) (h3 : a < b) :
  Real.sqrt (1 + Real.sqrt (40 + 24 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b :=
by
  sorry -- The proof is not required in this task.

end ordered_pair_sqrt_l520_52051


namespace amount_subtracted_l520_52055

theorem amount_subtracted (N A : ℝ) (h1 : N = 100) (h2 : 0.80 * N - A = 60) : A = 20 :=
by 
  sorry

end amount_subtracted_l520_52055


namespace line_passes_through_first_and_fourth_quadrants_l520_52045

theorem line_passes_through_first_and_fourth_quadrants (b k : ℝ) (H : b * k < 0) :
  (∃x₁, k * x₁ + b > 0) ∧ (∃x₂, k * x₂ + b < 0) :=
by
  sorry

end line_passes_through_first_and_fourth_quadrants_l520_52045


namespace boat_downstream_distance_l520_52042

variable (speed_still_water : ℤ) (speed_stream : ℤ) (time_downstream : ℤ)

theorem boat_downstream_distance
    (h₁ : speed_still_water = 24)
    (h₂ : speed_stream = 4)
    (h₃ : time_downstream = 4) :
    (speed_still_water + speed_stream) * time_downstream = 112 := by
  sorry

end boat_downstream_distance_l520_52042


namespace cleaner_steps_l520_52041

theorem cleaner_steps (a b c : ℕ) (h1 : a < 10 ∧ b < 10 ∧ c < 10) (h2 : 100 * a + 10 * b + c > 100 * c + 10 * b + a) (h3 : 100 * a + 10 * b + c + 100 * c + 10 * b + a = 746) :
  (100 * a + 10 * b + c) * 2 = 944 ∨ (100 * a + 10 * b + c) * 2 = 1142 :=
by
  sorry

end cleaner_steps_l520_52041


namespace total_miles_l520_52034

theorem total_miles (miles_Katarina miles_Harriet miles_Tomas miles_Tyler : ℕ)
  (hK : miles_Katarina = 51)
  (hH : miles_Harriet = 48)
  (hT : miles_Tomas = 48)
  (hTy : miles_Tyler = 48) :
  miles_Katarina + miles_Harriet + miles_Tomas + miles_Tyler = 195 :=
  by
    sorry

end total_miles_l520_52034


namespace numerator_equals_denominator_l520_52058

theorem numerator_equals_denominator (x : ℝ) (h : 4 * x - 3 = 5 * x + 2) : x = -5 :=
  by
    sorry

end numerator_equals_denominator_l520_52058


namespace ice_cream_bar_price_l520_52056

theorem ice_cream_bar_price 
  (num_bars num_sundaes : ℕ)
  (total_cost : ℝ)
  (sundae_price ice_cream_bar_price : ℝ)
  (h1 : num_bars = 125)
  (h2 : num_sundaes = 125)
  (h3 : total_cost = 250.00)
  (h4 : sundae_price = 1.40)
  (total_price_condition : num_bars * ice_cream_bar_price + num_sundaes * sundae_price = total_cost) :
  ice_cream_bar_price = 0.60 :=
sorry

end ice_cream_bar_price_l520_52056


namespace lines_parallel_m_value_l520_52038

theorem lines_parallel_m_value (m : ℝ) : 
  (∀ (x y : ℝ), (x + 2 * m * y - 1 = 0) → ((m - 2) * x - m * y + 2 = 0)) → m = 3 / 2 :=
by
  -- placeholder for mathematical proof
  sorry

end lines_parallel_m_value_l520_52038


namespace line_intersects_ellipse_l520_52013

theorem line_intersects_ellipse
  (m : ℝ) :
  ∃ P : ℝ × ℝ, P = (3, 2) ∧ ((m + 2) * P.1 - (m + 4) * P.2 + 2 - m = 0) ∧ 
  (P.1^2 / 25 + P.2^2 / 9 < 1) :=
by 
  sorry

end line_intersects_ellipse_l520_52013


namespace milton_zoology_books_l520_52029

theorem milton_zoology_books
  (z b : ℕ)
  (h1 : z + b = 80)
  (h2 : b = 4 * z) :
  z = 16 :=
by sorry

end milton_zoology_books_l520_52029


namespace num_distinct_x_intercepts_l520_52068

def f (x : ℝ) : ℝ := (x - 5) * (x^3 + 5*x^2 + 9*x + 9)

theorem num_distinct_x_intercepts : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x1 ∨ x = x2) :=
sorry

end num_distinct_x_intercepts_l520_52068


namespace number_of_cars_l520_52061

variable (C B : ℕ)

-- Define the conditions
def number_of_bikes : Prop := B = 2
def total_number_of_wheels : Prop := 4 * C + 2 * B = 44

-- State the theorem
theorem number_of_cars (hB : number_of_bikes B) (hW : total_number_of_wheels C B) : C = 10 := 
by 
  sorry

end number_of_cars_l520_52061


namespace compute_seventy_five_squared_minus_thirty_five_squared_l520_52026

theorem compute_seventy_five_squared_minus_thirty_five_squared :
  75^2 - 35^2 = 4400 := by
  sorry

end compute_seventy_five_squared_minus_thirty_five_squared_l520_52026


namespace allocation_schemes_correct_l520_52032

def numWaysToAllocateVolunteers : ℕ :=
  let n := 5 -- number of volunteers
  let m := 4 -- number of events
  Nat.choose n 2 * Nat.factorial m

theorem allocation_schemes_correct :
  numWaysToAllocateVolunteers = 240 :=
by
  sorry

end allocation_schemes_correct_l520_52032


namespace least_positive_integer_condition_l520_52094

theorem least_positive_integer_condition :
  ∃ (n : ℕ), n > 0 ∧ (n % 2 = 1) ∧ (n % 5 = 4) ∧ (n % 7 = 6) ∧ n = 69 :=
by
  sorry

end least_positive_integer_condition_l520_52094


namespace quadratic_rewriting_l520_52065

theorem quadratic_rewriting (d e : ℤ) (f : ℤ) : 
  (16 * x^2 - 40 * x - 24) = (d * x + e)^2 + f → 
  d^2 = 16 → 
  2 * d * e = -40 → 
  d * e = -20 := 
by
  intros h1 h2 h3
  sorry

end quadratic_rewriting_l520_52065


namespace sum_of_x_coordinates_on_parabola_l520_52007

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x ^ 2 - 2 * x + 1

-- Define the points P and Q on the parabola
variables {x1 x2 : ℝ}

-- The Lean theorem statement: 
theorem sum_of_x_coordinates_on_parabola 
  (h1 : parabola x1 = 1) 
  (h2 : parabola x2 = 1) : 
  x1 + x2 = 2 :=
sorry

end sum_of_x_coordinates_on_parabola_l520_52007


namespace add_base8_l520_52085

/-- Define the numbers in base 8 --/
def base8_add (a b : Nat) : Nat := 
  sorry

theorem add_base8 : base8_add 0o12 0o157 = 0o171 := 
  sorry

end add_base8_l520_52085


namespace right_triangle_sum_of_squares_l520_52014

   theorem right_triangle_sum_of_squares {AB AC BC : ℝ} (h_right: AB^2 + AC^2 = BC^2) (h_hypotenuse: BC = 1) :
     AB^2 + AC^2 + BC^2 = 2 :=
   by
     sorry
   
end right_triangle_sum_of_squares_l520_52014


namespace no_triangle_sum_of_any_two_angles_lt_120_no_triangle_sum_of_any_two_angles_gt_120_l520_52003

theorem no_triangle_sum_of_any_two_angles_lt_120 (α β γ : ℝ) (h : α + β + γ = 180) :
  ¬ (α + β < 120 ∧ β + γ < 120 ∧ γ + α < 120) :=
by
  sorry

theorem no_triangle_sum_of_any_two_angles_gt_120 (α β γ : ℝ) (h : α + β + γ = 180) :
  ¬ (α + β > 120 ∧ β + γ > 120 ∧ γ + α > 120) :=
by
  sorry

end no_triangle_sum_of_any_two_angles_lt_120_no_triangle_sum_of_any_two_angles_gt_120_l520_52003


namespace calc_expression_l520_52088

theorem calc_expression : (900^2) / (264^2 - 256^2) = 194.711 := by
  sorry

end calc_expression_l520_52088


namespace functional_equation_solution_l520_52025

theorem functional_equation_solution {f : ℝ → ℝ}
  (h : ∀ x y : ℝ, f (x^2 + f x * f y) = x * f (x + y)) :
  (f = fun x => 0) ∨ (f = id) ∨ (f = fun x => -x) :=
sorry

end functional_equation_solution_l520_52025


namespace total_discount_is_58_percent_l520_52075

-- Definitions and conditions
def sale_discount : ℝ := 0.4
def coupon_discount : ℝ := 0.3

-- Given an original price, the sale discount price and coupon discount price
def sale_price (original_price : ℝ) : ℝ := (1 - sale_discount) * original_price
def final_price (original_price : ℝ) : ℝ := (1 - coupon_discount) * (sale_price original_price)

-- Theorem statement: final discount is 58%
theorem total_discount_is_58_percent (original_price : ℝ) : (original_price - final_price original_price) / original_price = 0.58 :=
by intros; sorry

end total_discount_is_58_percent_l520_52075


namespace least_possible_perimeter_l520_52019

theorem least_possible_perimeter (x : ℕ) (h1 : 27 < x) (h2 : x < 75) :
  24 + 51 + x = 103 :=
by
  sorry

end least_possible_perimeter_l520_52019


namespace racers_in_final_segment_l520_52027

def initial_racers := 200

def racers_after_segment_1 (initial: ℕ) := initial - 10
def racers_after_segment_2 (after_segment_1: ℕ) := after_segment_1 - after_segment_1 / 3
def racers_after_segment_3 (after_segment_2: ℕ) := after_segment_2 - after_segment_2 / 4
def racers_after_segment_4 (after_segment_3: ℕ) := after_segment_3 - after_segment_3 / 3
def racers_after_segment_5 (after_segment_4: ℕ) := after_segment_4 - after_segment_4 / 2
def racers_after_segment_6 (after_segment_5: ℕ) := after_segment_5 - (3 * after_segment_5 / 4)

theorem racers_in_final_segment : racers_after_segment_6 (racers_after_segment_5 (racers_after_segment_4 (racers_after_segment_3 (racers_after_segment_2 (racers_after_segment_1 initial_racers))))) = 8 :=
  by
  sorry

end racers_in_final_segment_l520_52027


namespace correction_amount_l520_52009

variable (x : ℕ)

def half_dollar := 50
def quarter := 25
def nickel := 5
def dime := 10

theorem correction_amount : 
  ∀ x, (x * (half_dollar - quarter)) - (x * (dime - nickel)) = 20 * x := by
  intros x 
  sorry

end correction_amount_l520_52009


namespace arithmetic_sequence_sum_l520_52070

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Definition of the sum of the first n terms
def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

-- Problem statement in Lean 4
theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_first_n_terms a S)
  (h3 : S 9 = a 4 + a 5 + a 6 + 66) :
  a 2 + a 8 = 22 := by
  sorry

end arithmetic_sequence_sum_l520_52070


namespace mows_in_summer_l520_52090

theorem mows_in_summer (S : ℕ) (h1 : 8 - S = 3) : S = 5 :=
sorry

end mows_in_summer_l520_52090


namespace min_surface_area_l520_52099

/-- Defining the conditions and the problem statement -/
def solid (volume : ℝ) (face1 face2 : ℝ) : Prop := 
  ∃ x y z, x * y * z = volume ∧ (x * y = face1 ∨ y * z = face1 ∨ z * x = face1)
                      ∧ (x * y = face2 ∨ y * z = face2 ∨ z * x = face2)

def juan_solids (face1 face2 face3 face4 face5 face6 : ℝ) : Prop :=
  solid 128 4 32 ∧ solid 128 64 16 ∧ solid 128 8 32

theorem min_surface_area {volume : ℝ} {face1 face2 face3 face4 face5 face6 : ℝ} 
  (h : juan_solids 4 32 64 16 8 32) : 
  ∃ area : ℝ, area = 688 :=
sorry

end min_surface_area_l520_52099


namespace find_a_minus_b_l520_52083

theorem find_a_minus_b (a b : ℝ) (h1: ∀ x : ℝ, (ax^2 + bx - 2 = 0 → x = -2 ∨ x = -1/4)) : (a - b = 5) :=
sorry

end find_a_minus_b_l520_52083


namespace num_trailing_zeroes_500_factorial_l520_52046

-- Define the function to count factors of a prime p in n!
def count_factors_in_factorial (n p : ℕ) : ℕ :=
  if p = 0 then 0 else
    (n / p) + (n / (p ^ 2)) + (n / (p ^ 3)) + (n / (p ^ 4))

theorem num_trailing_zeroes_500_factorial : 
  count_factors_in_factorial 500 5 = 124 :=
sorry

end num_trailing_zeroes_500_factorial_l520_52046


namespace eq_has_unique_solution_l520_52050

theorem eq_has_unique_solution : 
  ∃! x : ℝ, (x ≠ 0)
    ∧ ((x < 0 → false) ∧ 
      (x > 0 → (x^18 + 1) * (x^16 + x^14 + x^12 + x^10 + x^8 + x^6 + x^4 + x^2 + 1) = 18 * x^9)) :=
by sorry

end eq_has_unique_solution_l520_52050


namespace kenya_peanuts_eq_133_l520_52017

def num_peanuts_jose : Nat := 85
def additional_peanuts_kenya : Nat := 48

def peanuts_kenya (jose_peanuts : Nat) (additional_peanuts : Nat) : Nat :=
  jose_peanuts + additional_peanuts

theorem kenya_peanuts_eq_133 : peanuts_kenya num_peanuts_jose additional_peanuts_kenya = 133 := by
  sorry

end kenya_peanuts_eq_133_l520_52017


namespace arithmetic_expression_l520_52012

theorem arithmetic_expression :
  (30 / (10 + 2 - 5) + 4) * 7 = 58 :=
by
  sorry

end arithmetic_expression_l520_52012


namespace maximum_abc_value_l520_52000

theorem maximum_abc_value:
  (∀ (a b c : ℝ), (0 < a ∧ a < 3) ∧ (0 < b ∧ b < 3) ∧ (0 < c ∧ c < 3) ∧ (∀ x : ℝ, (x^4 + a * x^3 + b * x^2 + c * x + 1) ≠ 0) → (abc ≤ 18.75)) :=
sorry

end maximum_abc_value_l520_52000


namespace domain_sqrt_tan_x_sub_sqrt3_l520_52044

open Real

noncomputable def domain := {x : ℝ | ∃ k : ℤ, k * π + π / 3 ≤ x ∧ x < k * π + π / 2}

theorem domain_sqrt_tan_x_sub_sqrt3 :
  {x | ∃ y : ℝ, y = sqrt (tan x - sqrt 3)} = domain :=
by
  sorry

end domain_sqrt_tan_x_sub_sqrt3_l520_52044


namespace race_distance_correct_l520_52043

noncomputable def solve_race_distance : ℝ :=
  let Vq := 1          -- assume Vq as some positive real number (could be normalized to 1 for simplicity)
  let Vp := 1.20 * Vq  -- P runs 20% faster
  let head_start := 300 -- Q's head start in meters
  let Dp := 1800       -- distance P runs

  -- time taken by P
  let time_p := Dp / Vp
  -- time taken by Q, given it has a 300 meter head start
  let Dq := Dp - head_start
  let time_q := Dq / Vq

  Dp

theorem race_distance_correct :
  let Vq := 1          -- assume Vq as some positive real number (could be normalized to 1 for simplicity)
  let Vp := 1.20 * Vq  -- P runs 20% faster
  let head_start := 300 -- Q's head start in meters
  let Dp := 1800       -- distance P runs
  -- time taken by P
  let time_p := Dp / Vp
  -- time taken by Q, given it has a 300 meter head start
  let Dq := Dp - head_start
  let time_q := Dq / Vq

  time_p = time_q := by
  sorry

end race_distance_correct_l520_52043


namespace problem_statement_l520_52002

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem problem_statement :
  let l := { p : ℝ × ℝ | p.1 - p.2 - 2 = 0 }
  let C := { p : ℝ × ℝ | ∃ θ : ℝ, p = (2 * Real.sqrt 3 * Real.cos θ, 2 * Real.sin θ) }
  let A := (-4, -6)
  let B := (4, 2)
  let P := (-2 * Real.sqrt 3, 2)
  let d := (|2 * Real.sqrt 3 * Real.cos (5 * Real.pi / 6) - 2|) / Real.sqrt 2
  distance A B = 8 * Real.sqrt 2 ∧ d = 3 * Real.sqrt 2 ∧
  let max_area := 1 / 2 * 8 * Real.sqrt 2 * 3 * Real.sqrt 2
  P ∈ C ∧ max_area = 24 := by
sorry

end problem_statement_l520_52002


namespace correct_union_l520_52028

universe u

-- Definitions
def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2}
def C_I (A : Set ℕ) : Set ℕ := {x ∈ I | x ∉ A}

-- Theorem statement
theorem correct_union : B ∪ C_I A = {2, 4, 5} :=
by
  sorry

end correct_union_l520_52028


namespace total_people_waiting_in_line_l520_52092

-- Conditions
def people_fitting_in_ferris_wheel : ℕ := 56
def people_not_getting_on : ℕ := 36

-- Definition: Number of people waiting in line
def number_of_people_waiting_in_line : ℕ := people_fitting_in_ferris_wheel + people_not_getting_on

-- Theorem to prove
theorem total_people_waiting_in_line : number_of_people_waiting_in_line = 92 := by
  -- This is a placeholder for the actual proof
  sorry

end total_people_waiting_in_line_l520_52092


namespace number_of_liars_l520_52004

/-- Definition of conditions -/
def total_islands : Nat := 17
def population_per_island : Nat := 119

-- Conditions based on the problem description
def islands_yes_first_question : Nat := 7
def islands_no_first_question : Nat := total_islands - islands_yes_first_question

def islands_no_second_question : Nat := 7
def islands_yes_second_question : Nat := total_islands - islands_no_second_question

def minimum_knights_for_no_second_question : Nat := 60  -- At least 60 knights

/-- Main theorem -/
theorem number_of_liars : 
  ∃ x y: Nat, 
    (x + (islands_no_first_question - y) = islands_yes_first_question ∧ 
     y - x = 3 ∧ 
     60 * x + 59 * y + 119 * (islands_no_first_question - y) = 1010 ∧
     (total_islands * population_per_island - 1010 = 1013)) := by
  sorry

end number_of_liars_l520_52004


namespace probability_red_or_black_probability_red_black_or_white_l520_52039

-- We define the probabilities of events A, B, and C
def P_A : ℚ := 5 / 12
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 6

-- Define the probability of event D for completeness
def P_D : ℚ := 1 / 12

-- 1. Statement for the probability of drawing a red or black ball (P(A ⋃ B))
theorem probability_red_or_black :
  (P_A + P_B = 3 / 4) :=
by
  sorry

-- 2. Statement for the probability of drawing a red, black, or white ball (P(A ⋃ B ⋃ C))
theorem probability_red_black_or_white :
  (P_A + P_B + P_C = 11 / 12) :=
by
  sorry

end probability_red_or_black_probability_red_black_or_white_l520_52039


namespace problem1_problem2_l520_52033

namespace ProofProblems

-- Problem 1: Prove the inequality
theorem problem1 (x : ℝ) (h : x + |2 * x - 1| < 3) : -2 < x ∧ x < 4 / 3 := 
sorry

-- Problem 2: Prove the value of x + y + z 
theorem problem2 (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 1) 
  (h2 : x + 2 * y + 3 * z = Real.sqrt 14) : 
  x + y + z = 3 * Real.sqrt 14 / 7 := 
sorry

end ProofProblems

end problem1_problem2_l520_52033


namespace sum_of_common_ratios_l520_52005

variable (m x y : ℝ)
variable (h₁ : x ≠ y)
variable (h₂ : a2 = m * x)
variable (h₃ : a3 = m * x^2)
variable (h₄ : b2 = m * y)
variable (h₅ : b3 = m * y^2)
variable (h₆ : a3 - b3 = 3 * (a2 - b2))

theorem sum_of_common_ratios : x + y = 3 :=
by
  sorry

end sum_of_common_ratios_l520_52005


namespace range_of_a_l520_52049

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → (x^2 - 2 * a * x + 2) ≥ a) ↔ (-3 ≤ a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l520_52049


namespace larger_number_225_l520_52072

theorem larger_number_225 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a - b = 120) 
  (h4 : Nat.lcm a b = 105 * Nat.gcd a b) : 
  max a b = 225 :=
by
  sorry

end larger_number_225_l520_52072


namespace simplify_expression_l520_52030

-- Define the algebraic expression
def algebraic_expr (x : ℚ) : ℚ := (3 / (x - 1) - x - 1) * (x - 1) / (x^2 - 4 * x + 4)

theorem simplify_expression : algebraic_expr 0 = 1 :=
by
  -- The proof is skipped using sorry
  sorry

end simplify_expression_l520_52030


namespace sean_bought_two_soups_l520_52098

theorem sean_bought_two_soups :
  ∃ (number_of_soups : ℕ),
    let soda_cost := 1
    let total_soda_cost := 3 * soda_cost
    let soup_cost := total_soda_cost
    let sandwich_cost := 3 * soup_cost
    let total_cost := 3 * soda_cost + sandwich_cost + soup_cost * number_of_soups
    total_cost = 18 ∧ number_of_soups = 2 :=
by
  sorry

end sean_bought_two_soups_l520_52098


namespace initial_salty_cookies_l520_52060

theorem initial_salty_cookies
  (initial_sweet_cookies : ℕ) 
  (ate_sweet_cookies : ℕ) 
  (ate_salty_cookies : ℕ) 
  (ate_diff : ℕ) 
  (H1 : initial_sweet_cookies = 39)
  (H2 : ate_sweet_cookies = 32)
  (H3 : ate_salty_cookies = 23)
  (H4 : ate_diff = 9) :
  initial_sweet_cookies - ate_diff = 30 :=
by sorry

end initial_salty_cookies_l520_52060


namespace triangle_cross_section_l520_52037

-- Definitions for the given conditions
inductive Solid
| Prism
| Pyramid
| Frustum
| Cylinder
| Cone
| TruncatedCone
| Sphere

-- The theorem statement of the proof problem
theorem triangle_cross_section (s : Solid) (cross_section_is_triangle : Prop) : 
  cross_section_is_triangle →
  (s = Solid.Prism ∨ s = Solid.Pyramid ∨ s = Solid.Frustum ∨ s = Solid.Cone) :=
sorry

end triangle_cross_section_l520_52037


namespace purely_imaginary_number_eq_l520_52001

theorem purely_imaginary_number_eq (z : ℂ) (a : ℝ) (i : ℂ) (h_imag : z.im = 0 ∧ z = 0 ∧ (3 - i) * z = a + i + i) :
  a = 1 / 3 :=
  sorry

end purely_imaginary_number_eq_l520_52001


namespace simplify_expr1_simplify_expr2_l520_52064

-- Problem 1
theorem simplify_expr1 (x y : ℝ) : x^2 - 5 * y - 4 * x^2 + y - 1 = -3 * x^2 - 4 * y - 1 :=
by sorry

-- Problem 2
theorem simplify_expr2 (a b : ℝ) : 7 * a + 3 * (a - 3 * b) - 2 * (b - 3 * a) = 16 * a - 11 * b :=
by sorry

end simplify_expr1_simplify_expr2_l520_52064


namespace guard_team_size_l520_52059

theorem guard_team_size (b n s : ℕ) (h_total : b * s * n = 1001) (h_condition : s < n ∧ n < b) : s = 7 := 
by
  sorry

end guard_team_size_l520_52059


namespace candy_bar_profit_l520_52082

theorem candy_bar_profit
  (bars_bought : ℕ)
  (cost_per_six : ℝ)
  (bars_sold : ℕ)
  (price_per_three : ℝ)
  (tax_rate : ℝ)
  (h1 : bars_bought = 800)
  (h2 : cost_per_six = 3)
  (h3 : bars_sold = 800)
  (h4 : price_per_three = 2)
  (h5 : tax_rate = 0.1) :
  let cost_per_bar := cost_per_six / 6
  let total_cost := bars_bought * cost_per_bar
  let price_per_bar := price_per_three / 3
  let total_revenue := bars_sold * price_per_bar
  let tax := tax_rate * total_revenue
  let after_tax_revenue := total_revenue - tax
  let profit_after_tax := after_tax_revenue - total_cost
  profit_after_tax = 80.02 := by
    sorry

end candy_bar_profit_l520_52082


namespace exists_n_gt_1958_l520_52018

noncomputable def polyline_path (n : ℕ) : ℝ := sorry
noncomputable def distance_to_origin (n : ℕ) : ℝ := sorry 
noncomputable def sum_lengths (n : ℕ) : ℝ := sorry

theorem exists_n_gt_1958 :
  ∃ (n : ℕ), n > 1958 ∧ (sum_lengths n) / (distance_to_origin n) > 1958 := 
sorry

end exists_n_gt_1958_l520_52018


namespace even_ngon_parallel_edges_odd_ngon_no_two_parallel_edges_l520_52087

theorem even_ngon_parallel_edges (n : ℕ) (h : n % 2 = 0) :
  ∃ i j, i ≠ j ∧ (i + 1) % n + i % n = (j + 1) % n + j % n :=
sorry

theorem odd_ngon_no_two_parallel_edges (n : ℕ) (h : n % 2 = 1) :
  ¬ ∃ i j, i ≠ j ∧ (i + 1) % n + i % n = (j + 1) % n + j % n :=
sorry

end even_ngon_parallel_edges_odd_ngon_no_two_parallel_edges_l520_52087


namespace students_still_in_school_l520_52097

def total_students := 5000
def students_to_beach := total_students / 2
def remaining_after_beach := total_students - students_to_beach
def students_to_art_museum := remaining_after_beach / 3
def remaining_after_art_museum := remaining_after_beach - students_to_art_museum
def students_to_science_fair := remaining_after_art_museum / 4
def remaining_after_science_fair := remaining_after_art_museum - students_to_science_fair
def students_to_music_workshop := 200
def remaining_students := remaining_after_science_fair - students_to_music_workshop

theorem students_still_in_school : remaining_students = 1051 := by
  sorry

end students_still_in_school_l520_52097


namespace expand_polynomial_correct_l520_52011

open Polynomial

noncomputable def expand_polynomial : Polynomial ℤ :=
  (C 3 * X^3 - C 2 * X^2 + X - C 4) * (C 4 * X^2 - C 2 * X + C 5)

theorem expand_polynomial_correct :
  expand_polynomial = C 12 * X^5 - C 14 * X^4 + C 23 * X^3 - C 28 * X^2 + C 13 * X - C 20 :=
by sorry

end expand_polynomial_correct_l520_52011


namespace focus_of_parabola_l520_52069

theorem focus_of_parabola (a : ℝ) (h1 : a > 0)
  (h2 : ∀ x, y = 3 * x → 3 / a = 3) :
  ∃ (focus : ℝ × ℝ), focus = (0, 1 / 8) :=
by
  -- The proof goes here
  sorry

end focus_of_parabola_l520_52069


namespace total_nephews_l520_52020

noncomputable def Alden_past_nephews : ℕ := 50
noncomputable def Alden_current_nephews : ℕ := 2 * Alden_past_nephews
noncomputable def Vihaan_current_nephews : ℕ := Alden_current_nephews + 60

theorem total_nephews :
  Alden_current_nephews + Vihaan_current_nephews = 260 := 
by
  sorry

end total_nephews_l520_52020


namespace bob_tiller_swath_width_l520_52023

theorem bob_tiller_swath_width
  (plot_width plot_length : ℕ)
  (tilling_rate_seconds_per_foot : ℕ)
  (total_tilling_minutes : ℕ)
  (total_area : ℕ)
  (tilled_length : ℕ)
  (swath_width : ℕ) :
  plot_width = 110 →
  plot_length = 120 →
  tilling_rate_seconds_per_foot = 2 →
  total_tilling_minutes = 220 →
  total_area = plot_width * plot_length →
  tilled_length = (total_tilling_minutes * 60) / tilling_rate_seconds_per_foot →
  swath_width = total_area / tilled_length →
  swath_width = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end bob_tiller_swath_width_l520_52023


namespace smallest_number_conditions_l520_52031

theorem smallest_number_conditions :
  ∃ m : ℕ, (∀ k ∈ [3, 4, 5, 6, 7], m % k = 2) ∧ (m % 8 = 0) ∧ ( ∀ n : ℕ, (∀ k ∈ [3, 4, 5, 6, 7], n % k = 2) ∧ (n % 8 = 0) → m ≤ n ) :=
sorry

end smallest_number_conditions_l520_52031


namespace perpendicular_vectors_l520_52084

def vector_a (m : ℝ) : ℝ × ℝ := (m, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (1, m + 1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (m : ℝ) (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by sorry

end perpendicular_vectors_l520_52084


namespace makeup_exam_probability_l520_52096

theorem makeup_exam_probability (total_students : ℕ) (students_in_makeup_exam : ℕ)
  (h1 : total_students = 42) (h2 : students_in_makeup_exam = 3) :
  (students_in_makeup_exam : ℚ) / total_students = 1 / 14 := by
  sorry

end makeup_exam_probability_l520_52096


namespace motorist_routes_birmingham_to_sheffield_l520_52022

-- Definitions for the conditions
def routes_bristol_to_birmingham : ℕ := 6
def routes_sheffield_to_carlisle : ℕ := 2
def total_routes_bristol_to_carlisle : ℕ := 36

-- The proposition that should be proven
theorem motorist_routes_birmingham_to_sheffield : 
  ∃ x : ℕ, routes_bristol_to_birmingham * x * routes_sheffield_to_carlisle = total_routes_bristol_to_carlisle ∧ x = 3 :=
sorry

end motorist_routes_birmingham_to_sheffield_l520_52022


namespace probability_of_selecting_green_ball_l520_52006

-- Declare the probability of selecting each container
def prob_of_selecting_container := (1 : ℚ) / 4

-- Declare the number of balls in each container
def balls_in_container_A := 10
def balls_in_container_B := 14
def balls_in_container_C := 14
def balls_in_container_D := 10

-- Declare the number of green balls in each container
def green_balls_in_A := 6
def green_balls_in_B := 6
def green_balls_in_C := 6
def green_balls_in_D := 7

-- Calculate the probability of drawing a green ball from each container
def prob_green_from_A := (green_balls_in_A : ℚ) / balls_in_container_A
def prob_green_from_B := (green_balls_in_B : ℚ) / balls_in_container_B
def prob_green_from_C := (green_balls_in_C : ℚ) / balls_in_container_C
def prob_green_from_D := (green_balls_in_D : ℚ) / balls_in_container_D

-- Calculate the total probability of drawing a green ball
def total_prob_green :=
  prob_of_selecting_container * prob_green_from_A +
  prob_of_selecting_container * prob_green_from_B +
  prob_of_selecting_container * prob_green_from_C +
  prob_of_selecting_container * prob_green_from_D

theorem probability_of_selecting_green_ball : total_prob_green = 13 / 28 :=
by sorry

end probability_of_selecting_green_ball_l520_52006


namespace biloca_path_proof_l520_52057

def diagonal_length := 5 -- Length of one diagonal as deduced from Pipoca's path
def tile_width := 3 -- Width of one tile as deduced from Tonica's path
def tile_length := 4 -- Length of one tile as deduced from Cotinha's path

def Biloca_path_length : ℝ :=
  3 * diagonal_length + 4 * tile_width + 2 * tile_length

theorem biloca_path_proof :
  Biloca_path_length = 43 :=
by
  sorry

end biloca_path_proof_l520_52057


namespace simplify_expression_l520_52080

theorem simplify_expression (x : ℕ) : (5 * x^4)^3 = 125 * x^(12) := by
  sorry

end simplify_expression_l520_52080


namespace percent_of_y_eq_l520_52071

theorem percent_of_y_eq (y : ℝ) (h : y ≠ 0) : (0.3 * 0.7 * y) = (0.21 * y) := by
  sorry

end percent_of_y_eq_l520_52071


namespace value_of_m_minus_n_l520_52016

variables {a b : ℕ}
variables {m n : ℤ}

def are_like_terms (m n : ℤ) : Prop :=
  (m - 2 = 4) ∧ (n + 7 = 4)

theorem value_of_m_minus_n (h : are_like_terms m n) : m - n = 9 :=
by
  sorry

end value_of_m_minus_n_l520_52016


namespace find_m_if_z_is_pure_imaginary_l520_52076

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem find_m_if_z_is_pure_imaginary (m : ℝ) (z : ℂ) (i : ℂ) (h_i_unit : i^2 = -1) (h_z : z = (1 + i) / (1 - i) + m * (1 - i)) :
  is_pure_imaginary z → m = 0 := 
by
  sorry

end find_m_if_z_is_pure_imaginary_l520_52076


namespace simplify_expression_l520_52073

theorem simplify_expression (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -3) :
    (x + 2 - 5 / (x - 2)) / ((x + 3) / (x - 2)) = x - 3 :=
sorry

end simplify_expression_l520_52073


namespace height_of_picture_frame_l520_52078

-- Define the given conditions
def width : ℕ := 6
def perimeter : ℕ := 30
def perimeter_formula (w h : ℕ) : ℕ := 2 * (w + h)

-- Prove that the height of the picture frame is 9 inches
theorem height_of_picture_frame : ∃ height : ℕ, height = 9 ∧ perimeter_formula width height = perimeter :=
by
  -- Proof goes here
  sorry

end height_of_picture_frame_l520_52078


namespace original_commercial_length_l520_52095

theorem original_commercial_length (x : ℝ) (h : 0.70 * x = 21) : x = 30 := sorry

end original_commercial_length_l520_52095


namespace product_ge_half_l520_52063

theorem product_ge_half (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : 0 ≤ x3) (h_sum : x1 + x2 + x3 ≤ 1/2) :
  (1 - x1) * (1 - x2) * (1 - x3) ≥ 1/2 :=
by
  sorry

end product_ge_half_l520_52063


namespace relatively_prime_positive_integers_l520_52079

theorem relatively_prime_positive_integers (a b : ℕ) (h1 : a > b) (h2 : gcd a b = 1) (h3 : (a^3 - b^3) / (a - b)^3 = 91 / 7) : a - b = 1 := 
by 
  sorry

end relatively_prime_positive_integers_l520_52079


namespace find_a_l520_52067

theorem find_a (a b c : ℕ) (h₁ : a + b = c) (h₂ : b + 2 * c = 10) (h₃ : c = 4) : a = 2 := by
  sorry

end find_a_l520_52067


namespace james_has_43_oreos_l520_52091

def james_oreos (jordan : ℕ) : ℕ := 7 + 4 * jordan

theorem james_has_43_oreos (jordan : ℕ) (total : ℕ) (h1 : total = jordan + james_oreos jordan) (h2 : total = 52) : james_oreos jordan = 43 :=
by
  sorry

end james_has_43_oreos_l520_52091


namespace find_m_of_power_fn_and_increasing_l520_52040

theorem find_m_of_power_fn_and_increasing (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m^2 - m - 5) * x^(m - 1) > 0) →
  m^2 - m - 5 = 1 →
  1 < m →
  m = 3 :=
sorry

end find_m_of_power_fn_and_increasing_l520_52040


namespace sequence_converges_l520_52047

theorem sequence_converges (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n) (h_condition : ∀ m n, a (n + m) ≤ a n * a m) : 
    ∃ l : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |(a n)^ (1/n) - l| < ε :=
by
  sorry

end sequence_converges_l520_52047


namespace maurice_rides_before_visit_l520_52053

-- Defining all conditions in Lean
variables
  (M : ℕ) -- Number of times Maurice had been horseback riding before visiting Matt
  (Matt_rides_with_M : ℕ := 8 * 2) -- Number of times Matt rode with Maurice (8 times, 2 horses each time)
  (Matt_rides_alone : ℕ := 16) -- Number of times Matt rode solo
  (total_Matt_rides : ℕ := Matt_rides_with_M + Matt_rides_alone) -- Total rides by Matt
  (three_times_M : ℕ := 3 * M) -- Three times the number of times Maurice rode before visiting
  (unique_horses_M : ℕ := 8) -- Total number of unique horses Maurice rode during his visit

-- Main theorem
theorem maurice_rides_before_visit  
  (h1: total_Matt_rides = three_times_M) 
  (h2: unique_horses_M = M) 
  : M = 10 := sorry

end maurice_rides_before_visit_l520_52053


namespace greater_number_l520_52035

theorem greater_number (x y : ℕ) (h_sum : x + y = 50) (h_diff : x - y = 16) : x = 33 :=
by
  sorry

end greater_number_l520_52035


namespace no_real_a_values_l520_52010

noncomputable def polynomial_with_no_real_root (a : ℝ) : Prop :=
  ∀ x : ℝ, x^4 + a^2 * x^3 - 2 * x^2 + a * x + 4 ≠ 0
  
theorem no_real_a_values :
  ∀ a : ℝ, (∃ x : ℝ, x^4 + a^2 * x^3 - 2 * x^2 + a * x + 4 = 0) → false :=
by sorry

end no_real_a_values_l520_52010


namespace choose_starters_l520_52048

theorem choose_starters :
  let totalPlayers := 16
  let numberOfTwins := 2
  let playersExcludingTwins := totalPlayers - numberOfTwins
  Nat.choose totalPlayers 6 - Nat.choose playersExcludingTwins 6 = 5005 :=
by
  let totalPlayers := 16
  let numberOfTwins := 2
  let playersExcludingTwins := totalPlayers - numberOfTwins
  sorry

end choose_starters_l520_52048


namespace triangle_area_proof_l520_52036

noncomputable def triangle_area (a b c C : ℝ) : ℝ := 0.5 * a * b * Real.sin C

theorem triangle_area_proof:
  ∀ (A B C a b c : ℝ),
  ¬ (C = π/2) ∧
  c = 1 ∧
  C = π/3 ∧
  Real.sin C + Real.sin (A - B) = 3 * Real.sin (2*B) →
  triangle_area a b c C = 3 * Real.sqrt 3 / 28 :=
by
  intros A B C a b c h
  sorry

end triangle_area_proof_l520_52036


namespace difference_of_squares_l520_52015

def a : ℕ := 601
def b : ℕ := 597

theorem difference_of_squares : a^2 - b^2 = 4792 :=
by {
  sorry
}

end difference_of_squares_l520_52015


namespace solve_system_l520_52024

open Real

theorem solve_system :
  (∃ x y : ℝ, (sin x) ^ 2 + (cos y) ^ 2 = y ^ 4 ∧ (sin y) ^ 2 + (cos x) ^ 2 = x ^ 2) → 
  (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) := by
  sorry

end solve_system_l520_52024


namespace sum_of_three_integers_l520_52008

theorem sum_of_three_integers (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a > 0) (h5 : b > 0) (h6 : c > 0) (h7 : a * b * c = 5^3) : a + b + c = 31 :=
by
  sorry

end sum_of_three_integers_l520_52008


namespace arithmetic_sequence_sum_l520_52077

theorem arithmetic_sequence_sum :
  ∀ (x y : ℕ), (∀ (a b c : ℕ), b - a = c - b → c - b = 5) ∧
               (3 + 5 * 1 = 8) ∧
               (8 + 5 * 1 = 13) ∧
               (x + 5 * 1 = y) ∧
               (y + 5 * 1 = 33) →
               x + y = 51 :=
by
  intros x y h
  sorry

end arithmetic_sequence_sum_l520_52077


namespace vans_needed_l520_52093

-- Definitions of conditions
def students : Nat := 2
def adults : Nat := 6
def capacity_per_van : Nat := 4

-- Main theorem to prove
theorem vans_needed : (students + adults) / capacity_per_van = 2 := by
  sorry

end vans_needed_l520_52093


namespace contrapositive_question_l520_52081

theorem contrapositive_question (x : ℝ) :
  (x = 2 → x^2 - 3 * x + 2 = 0) ↔ (x^2 - 3 * x + 2 ≠ 0 → x ≠ 2) := 
sorry

end contrapositive_question_l520_52081


namespace walking_rate_ratio_l520_52074

variables (R R' : ℝ)

theorem walking_rate_ratio (h₁ : R * 21 = R' * 18) : R' / R = 7 / 6 :=
by {
  sorry
}

end walking_rate_ratio_l520_52074


namespace tan_alpha_plus_pi_div_four_l520_52062

theorem tan_alpha_plus_pi_div_four (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  Real.tan (α + Real.pi / 4) = 3 / 22 := 
by
  sorry

end tan_alpha_plus_pi_div_four_l520_52062


namespace exists_polynomial_triangle_property_l520_52089

noncomputable def f (x y z : ℝ) : ℝ :=
  (x + y + z) * (-x + y + z) * (x - y + z) * (x + y - z)

theorem exists_polynomial_triangle_property :
  ∀ (x y z : ℝ), (f x y z > 0 ↔ (|x| + |y| > |z| ∧ |y| + |z| > |x| ∧ |z| + |x| > |y|)) :=
sorry

end exists_polynomial_triangle_property_l520_52089


namespace probability_three_defective_phones_l520_52066

theorem probability_three_defective_phones :
  let total_smartphones := 380
  let defective_smartphones := 125
  let P_def_1 := (defective_smartphones : ℝ) / total_smartphones
  let P_def_2 := (defective_smartphones - 1 : ℝ) / (total_smartphones - 1)
  let P_def_3 := (defective_smartphones - 2 : ℝ) / (total_smartphones - 2)
  let P_all_three_def := P_def_1 * P_def_2 * P_def_3
  abs (P_all_three_def - 0.0351) < 0.001 := 
by
  sorry

end probability_three_defective_phones_l520_52066


namespace european_confidence_95_european_teams_not_face_l520_52086

-- Definitions for the conditions
def european_teams_round_of_16 := 44
def european_teams_not_round_of_16 := 22
def other_regions_round_of_16 := 36
def other_regions_not_round_of_16 := 58
def total_teams := 160

-- Formula for K^2 calculation
def k_value : ℚ := 3.841
def k_squared (n a_d_diff b_c_diff a b c d : ℚ) : ℚ :=
  n * ((a_d_diff - b_c_diff)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Definitions and calculation of K^2
def n1 := (european_teams_round_of_16 + other_regions_round_of_16 : ℚ)
def a_d_diff1 := (european_teams_round_of_16 * other_regions_not_round_of_16 : ℚ)
def b_c_diff1 := (european_teams_not_round_of_16 * other_regions_round_of_16 : ℚ)
def k_squared_result := k_squared n1 a_d_diff1 b_c_diff1
                                 (european_teams_round_of_16 + european_teams_not_round_of_16)
                                 (other_regions_round_of_16 + other_regions_not_round_of_16)
                                 total_teams total_teams

-- Theorem for 95% confidence derived
theorem european_confidence_95 :
  k_squared_result > k_value := sorry

-- Probability calculation setup
def total_ways_to_pair_teams : ℚ := 15
def ways_european_teams_not_face : ℚ := 6
def probability_european_teams_not_face := ways_european_teams_not_face / total_ways_to_pair_teams

-- Theorem for probability
theorem european_teams_not_face :
  probability_european_teams_not_face = 2 / 5 := sorry

end european_confidence_95_european_teams_not_face_l520_52086


namespace geom_series_sum_l520_52021

/-- The sum of the first six terms of the geometric series 
    with first term a = 1 and common ratio r = (1 / 4) is 1365 / 1024. -/
theorem geom_series_sum : 
  let a : ℚ := 1
  let r : ℚ := 1 / 4
  let n : ℕ := 6
  (a * (1 - r^n) / (1 - r)) = 1365 / 1024 :=
by
  sorry

end geom_series_sum_l520_52021


namespace holiday_customers_l520_52054

-- Define the normal rate of customers entering the store (175 people/hour)
def normal_rate : ℕ := 175

-- Define the holiday rate of customers entering the store
def holiday_rate : ℕ := 2 * normal_rate

-- Define the duration for which we are calculating the total number of customers (8 hours)
def duration : ℕ := 8

-- Define the correct total number of customers (2800 people)
def correct_total_customers : ℕ := 2800

-- The theorem that asserts the total number of customers in 8 hours during the holiday season is 2800
theorem holiday_customers : holiday_rate * duration = correct_total_customers := by
  sorry

end holiday_customers_l520_52054


namespace sanoop_initial_tshirts_l520_52052

theorem sanoop_initial_tshirts (n : ℕ) (T : ℕ) 
(avg_initial : T = n * 526) 
(avg_remaining : T - 673 = (n - 1) * 505) 
(avg_returned : 673 = 673) : 
n = 8 := 
by 
  sorry

end sanoop_initial_tshirts_l520_52052
