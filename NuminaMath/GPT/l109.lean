import Mathlib

namespace probability_x_gt_3y_l109_109650

-- Definitions of the conditions
def rect := {(x, y) | 0 ≤ x ∧ x ≤ 2010 ∧ 0 ≤ y ∧ y ≤ 2011}

-- Theorem statement to prove the probability
theorem probability_x_gt_3y : 
  (∃ (x y : ℝ), (x, y) ∈ rect ∧ (1 / (2010 * 2011)) * ((1 / 2) * 2010 * 670) = 335 / 2011) :=
sorry

end probability_x_gt_3y_l109_109650


namespace sara_initial_quarters_l109_109961

theorem sara_initial_quarters (total_quarters: ℕ) (dad_gave: ℕ) (initial_quarters: ℕ) 
  (h1: dad_gave = 49) (h2: total_quarters = 70) (h3: total_quarters = initial_quarters + dad_gave) :
  initial_quarters = 21 := 
by {
  sorry
}

end sara_initial_quarters_l109_109961


namespace ten_times_ten_thousand_ten_times_one_million_ten_times_ten_million_tens_of_thousands_in_hundred_million_l109_109559

theorem ten_times_ten_thousand : 10 * 10000 = 100000 :=
by sorry

theorem ten_times_one_million : 10 * 1000000 = 10000000 :=
by sorry

theorem ten_times_ten_million : 10 * 10000000 = 100000000 :=
by sorry

theorem tens_of_thousands_in_hundred_million : 100000000 / 10000 = 10000 :=
by sorry

end ten_times_ten_thousand_ten_times_one_million_ten_times_ten_million_tens_of_thousands_in_hundred_million_l109_109559


namespace power_function_at_4_l109_109830

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem power_function_at_4 {α : ℝ} :
  power_function α 2 = (Real.sqrt 2) / 2 →
  α = -1/2 →
  power_function α 4 = 1 / 2 :=
by
  intros h1 h2
  rw [h2, power_function]
  sorry

end power_function_at_4_l109_109830


namespace total_number_of_parts_l109_109100

-- Identify all conditions in the problem: sample size and probability
def sample_size : ℕ := 30
def probability : ℝ := 0.25

-- Statement of the proof problem: The total number of parts N is 120 given the conditions
theorem total_number_of_parts (N : ℕ) (h : (sample_size : ℝ) / N = probability) : N = 120 :=
sorry

end total_number_of_parts_l109_109100


namespace simplify_expression_l109_109965

theorem simplify_expression : 
  let x := 2
  let y := -1 / 2
  (2 * x^2 + (-x^2 - 2 * x * y + 2 * y^2) - 3 * (x^2 - x * y + 2 * y^2)) = -10 := by
  sorry

end simplify_expression_l109_109965


namespace geometric_sequence_sum_l109_109335

variable (a : ℕ → ℝ)

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_sum (h1 : geometric_sequence a)
  (h2 : a 1 > 0)
  (h3 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) :
  a 3 + a 5 = 6 :=
sorry

end geometric_sequence_sum_l109_109335


namespace equal_share_of_marbles_l109_109595

-- Define the number of marbles bought by each friend based on the conditions
def wolfgang_marbles : ℕ := 16
def ludo_marbles : ℕ := wolfgang_marbles + (wolfgang_marbles / 4)
def michael_marbles : ℕ := 2 * (wolfgang_marbles + ludo_marbles) / 3
def shania_marbles : ℕ := 2 * ludo_marbles
def gabriel_marbles : ℕ := (wolfgang_marbles + ludo_marbles + michael_marbles + shania_marbles) - 1
def total_marbles : ℕ := wolfgang_marbles + ludo_marbles + michael_marbles + shania_marbles + gabriel_marbles
def marbles_per_friend : ℕ := total_marbles / 5

-- Mathematical equivalent proof problem
theorem equal_share_of_marbles : marbles_per_friend = 39 := by
  sorry

end equal_share_of_marbles_l109_109595


namespace find_integer_l109_109832

theorem find_integer (n : ℤ) 
  (h1 : 50 ≤ n ∧ n ≤ 150)
  (h2 : n % 7 = 0)
  (h3 : n % 9 = 3)
  (h4 : n % 6 = 3) : 
  n = 63 := by 
  sorry

end find_integer_l109_109832


namespace value_of_x_plus_y_squared_l109_109918

theorem value_of_x_plus_y_squared (x y : ℝ) 
  (h₁ : x^2 + y^2 = 20) 
  (h₂ : x * y = 6) : 
  (x + y)^2 = 32 :=
by
  sorry

end value_of_x_plus_y_squared_l109_109918


namespace percentage_of_sikh_boys_l109_109805

theorem percentage_of_sikh_boys (total_boys muslim_percentage hindu_percentage other_boys : ℕ) 
  (h₁ : total_boys = 300) 
  (h₂ : muslim_percentage = 44) 
  (h₃ : hindu_percentage = 28) 
  (h₄ : other_boys = 54) : 
  (10 : ℝ) = 
  (((total_boys - (muslim_percentage * total_boys / 100 + hindu_percentage * total_boys / 100 + other_boys)) * 100) / total_boys : ℝ) :=
by
  sorry

end percentage_of_sikh_boys_l109_109805


namespace each_child_receives_1680_l109_109410

-- Definitions for conditions
def husband_weekly_savings : ℕ := 335
def wife_weekly_savings : ℕ := 225
def weeks_in_month : ℕ := 4
def months_saving : ℕ := 6
def children : ℕ := 4

-- Total savings calculation
def husband_monthly_savings := husband_weekly_savings * weeks_in_month
def wife_monthly_savings := wife_weekly_savings * weeks_in_month
def total_monthly_savings := husband_monthly_savings + wife_monthly_savings
def total_savings := total_monthly_savings * months_saving
def half_savings := total_savings / 2
def amount_per_child := half_savings / children

-- The theorem to prove
theorem each_child_receives_1680 : amount_per_child = 1680 := 
by 
sorriesorry

end each_child_receives_1680_l109_109410


namespace g_of_5_l109_109319

noncomputable def g (x : ℝ) : ℝ := -2 / x

theorem g_of_5 (x : ℝ) : g (g (g (g (g x)))) = -2 / x :=
by
  sorry

end g_of_5_l109_109319


namespace find_compounding_frequency_l109_109669

-- Lean statement defining the problem conditions and the correct answer

theorem find_compounding_frequency (P A : ℝ) (r t : ℝ) (hP : P = 12000) (hA : A = 13230) 
(hri : r = 0.10) (ht : t = 1) 
: ∃ (n : ℕ), A = P * (1 + r / n) ^ (n * t) ∧ n = 2 := 
by
  -- Definitions from the conditions
  have hP := hP
  have hA := hA
  have hr := hri
  have ht := ht
  
  -- Substitute known values
  use 2
  -- Show that the statement holds with n = 2
  sorry

end find_compounding_frequency_l109_109669


namespace karlanna_marble_problem_l109_109641

theorem karlanna_marble_problem : 
  ∃ (m_values : Finset ℕ), 
  (∀ m ∈ m_values, ∃ n : ℕ, m * n = 450 ∧ m > 1 ∧ n > 1) ∧ 
  m_values.card = 16 := 
by
  sorry

end karlanna_marble_problem_l109_109641


namespace number_of_sixth_powers_less_than_1000_l109_109301

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | ∃ m : ℕ, n = m^6 ∧ n < 1000}.to_finset.card = 3 := 
by {
  sorry
}

end number_of_sixth_powers_less_than_1000_l109_109301


namespace food_left_after_bbqs_l109_109099

noncomputable def mushrooms_bought : ℕ := 15
noncomputable def chicken_bought : ℕ := 20
noncomputable def beef_bought : ℕ := 10

noncomputable def mushrooms_consumed : ℕ := 5 * 3
noncomputable def chicken_consumed : ℕ := 4 * 2
noncomputable def beef_consumed : ℕ := 2 * 1

noncomputable def mushrooms_left : ℕ := mushrooms_bought - mushrooms_consumed
noncomputable def chicken_left : ℕ := chicken_bought - chicken_consumed
noncomputable def beef_left : ℕ := beef_bought - beef_consumed

noncomputable def total_food_left : ℕ := mushrooms_left + chicken_left + beef_left

theorem food_left_after_bbqs : total_food_left = 20 :=
  by
    unfold total_food_left mushrooms_left chicken_left beef_left
    unfold mushrooms_consumed chicken_consumed beef_consumed
    unfold mushrooms_bought chicken_bought beef_bought
    sorry

end food_left_after_bbqs_l109_109099


namespace smallest_among_l109_109876

theorem smallest_among {a b c d : ℝ} (h1 : a = Real.pi) (h2 : b = -2) (h3 : c = 0) (h4 : d = -1) : 
  ∃ (x : ℝ), x = b ∧ x < a ∧ x < c ∧ x < d := 
by {
  sorry
}

end smallest_among_l109_109876


namespace digit_150_of_17_div_70_l109_109213

theorem digit_150_of_17_div_70 : (decimal_digit 150 (17 / 70)) = 7 :=
sorry

end digit_150_of_17_div_70_l109_109213


namespace digit_150th_l109_109207

-- Define the fraction and its repeating decimal properties
def fraction := 17 / 70
def repeating_block := "242857"
def block_length : ℕ := 6

-- The 150th digit calculation
def digit_position : ℕ := 150 % block_length

theorem digit_150th : "150th digit of decimal representation" = 7 :=
by {
  have h1 : fraction = 0.242857142857, sorry,
  have h2 : "252857".cycle = repeating_block, sorry,
  have h3 : digit_position = 0, sorry,
  have h4 : repeating_block.get_last = '7', sorry,
  show "150th digit of decimal representation"= 7, sorry,
}

end digit_150th_l109_109207


namespace simplify_expression_l109_109522

theorem simplify_expression : |(-4 : Int)^2 - (3 : Int)^2 + 2| = 9 := by
  sorry

end simplify_expression_l109_109522


namespace max_ab_value_l109_109461

noncomputable def max_ab (a b : ℝ) : ℝ := a * b

theorem max_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 8) : max_ab a b ≤ 4 :=
by
  sorry

end max_ab_value_l109_109461


namespace min_M_for_inequality_l109_109446

noncomputable def M := (9 * Real.sqrt 2) / 32

theorem min_M_for_inequality (a b c : ℝ) : 
  abs (a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)) 
  ≤ M * (a^2 + b^2 + c^2)^2 := 
sorry

end min_M_for_inequality_l109_109446


namespace find_a6_l109_109499

variable {a : ℕ → ℝ}

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
def given_condition (a : ℕ → ℝ) (d : ℝ) : Prop :=
  2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36

theorem find_a6 (d : ℝ) :
  is_arithmetic_sequence a d →
  given_condition a d →
  a 6 = 3 :=
by
  -- The proof would go here
  sorry

end find_a6_l109_109499


namespace find_a7_l109_109928

variable (a : ℕ → ℝ)
variable (r : ℝ)
variable (n : ℕ)

-- Condition 1: The sequence {a_n} is geometric with all positive terms.
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

-- Condition 2: a₄ * a₁₀ = 16
axiom geo_seq_condition : is_geometric_sequence a r ∧ a 4 * a 10 = 16

-- The goal to prove
theorem find_a7 : (is_geometric_sequence a r ∧ a 4 * a 10 = 16) → a 7 = 4 :=
by {
  sorry
}

end find_a7_l109_109928


namespace monotonic_decreasing_interval_l109_109764

def f (x : ℝ) := Real.exp x / x^2

theorem monotonic_decreasing_interval :
  ∀ x, (0 < x ∧ x ≤ 2) → (f' x < 0) :=
by
  sorry

end monotonic_decreasing_interval_l109_109764


namespace probability_of_odd_score_l109_109678

noncomputable def dartboard : Type := sorry

variables (r_inner r_outer : ℝ)
variables (inner_values outer_values : Fin 3 → ℕ)
variables (P_odd : ℚ)

-- Conditions
def dartboard_conditions (r_inner r_outer : ℝ) (inner_values outer_values : Fin 3 → ℕ) : Prop :=
  r_inner = 4 ∧ r_outer = 8 ∧
  inner_values 0 = 3 ∧ inner_values 1 = 1 ∧ inner_values 2 = 1 ∧
  outer_values 0 = 3 ∧ outer_values 1 = 2 ∧ outer_values 2 = 2

-- Correct Answer
def correct_odds_probability (P_odd : ℚ) : Prop :=
  P_odd = 4 / 9

-- Main Statement
theorem probability_of_odd_score (r_inner r_outer : ℝ) (inner_values outer_values : Fin 3 → ℕ) (P_odd : ℚ) :
  dartboard_conditions r_inner r_outer inner_values outer_values →
  correct_odds_probability P_odd :=
sorry

end probability_of_odd_score_l109_109678


namespace popsicle_count_l109_109500

-- Define the number of each type of popsicles
def num_grape_popsicles : Nat := 2
def num_cherry_popsicles : Nat := 13
def num_banana_popsicles : Nat := 2

-- Prove the total number of popsicles
theorem popsicle_count : num_grape_popsicles + num_cherry_popsicles + num_banana_popsicles = 17 := by
  sorry

end popsicle_count_l109_109500


namespace vector_equivalence_l109_109801

-- Define the vectors a and b
noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def vector_b : ℝ × ℝ := (-1, 1)

-- Define the operation 3a - b
noncomputable def vector_operation (a b : ℝ × ℝ) : ℝ × ℝ :=
  (3 * a.1 - b.1, 3 * a.2 - b.2)

-- State that for given vectors a and b, the result of the operation equals (4, 2)
theorem vector_equivalence : vector_operation vector_a vector_b = (4, 2) :=
  sorry

end vector_equivalence_l109_109801


namespace total_cost_is_58_l109_109243

-- Define the conditions
def cost_per_adult : Nat := 22
def cost_per_child : Nat := 7
def number_of_adults : Nat := 2
def number_of_children : Nat := 2

-- Define the theorem to prove the total cost
theorem total_cost_is_58 : number_of_adults * cost_per_adult + number_of_children * cost_per_child = 58 :=
by
  -- Steps of proof will go here
  sorry

end total_cost_is_58_l109_109243


namespace cornbread_pieces_l109_109341

theorem cornbread_pieces :
  let pan_length := 20
  let pan_width := 18
  let piece_length := 2
  let piece_width := 2
  let pan_area := pan_length * pan_width
  let piece_area := piece_length * piece_width
  let num_pieces := pan_area / piece_area
  num_pieces = 90 :=
by
  let pan_length := 20
  let pan_width := 18
  let piece_length := 2
  let piece_width := 2
  let pan_area := pan_length * pan_width
  let piece_area := piece_length * piece_width
  let num_pieces := pan_area / piece_area
  show num_pieces = 90
  from sorry

end cornbread_pieces_l109_109341


namespace count_sixth_powers_less_than_1000_l109_109310

theorem count_sixth_powers_less_than_1000 : 
  {n : ℕ | 0 < n ∧ n < 1000 ∧ (∃ k : ℕ, n = k^6)}.to_finset.card = 3 :=
by 
  sorry

end count_sixth_powers_less_than_1000_l109_109310


namespace find_k_l109_109792

theorem find_k (k : ℝ) :
  let a := (3, 1)
  let b := (1, 3)
  let c := (k, 7)
  ((a.1 - c.1) * b.2 - (a.2 - c.2) * b.1 = 0) → k = 5 := 
by
  sorry

end find_k_l109_109792


namespace completing_square_result_l109_109367

theorem completing_square_result:
  ∀ x : ℝ, (x^2 + 4 * x - 1 = 0) → (x + 2)^2 = 5 :=
by
  sorry

end completing_square_result_l109_109367


namespace greatest_possible_large_chips_l109_109182

theorem greatest_possible_large_chips :
  ∃ l s : ℕ, ∃ p : ℕ, s + l = 61 ∧ s = l + p ∧ Nat.Prime p ∧ l = 29 :=
sorry

end greatest_possible_large_chips_l109_109182


namespace james_training_hours_in_a_year_l109_109135

-- Definitions based on conditions
def trains_twice_a_day : ℕ := 2
def hours_per_training : ℕ := 4
def days_trains_per_week : ℕ := 7 - 2
def weeks_per_year : ℕ := 52

-- Resultant computation
def daily_training_hours : ℕ := trains_twice_a_day * hours_per_training
def weekly_training_hours : ℕ := daily_training_hours * days_trains_per_week
def yearly_training_hours : ℕ := weekly_training_hours * weeks_per_year

-- Statement to prove
theorem james_training_hours_in_a_year : yearly_training_hours = 2080 := by
  -- proof goes here
  sorry

end james_training_hours_in_a_year_l109_109135


namespace find_cos_A_l109_109775

variable {A : Real}

theorem find_cos_A (h1 : 0 < A) (h2 : A < π / 2) (h3 : Real.tan A = 2 / 3) : Real.cos A = 3 * Real.sqrt 13 / 13 :=
by
  sorry

end find_cos_A_l109_109775


namespace remainder_444_pow_444_mod_13_l109_109702

theorem remainder_444_pow_444_mod_13 :
  (444 ^ 444) % 13 = 1 :=
by
  have h1 : 444 % 13 = 2 := by norm_num
  have h2 : 2 ^ 12 % 13 = 1 := by norm_num
  rw [←pow_mul, ←nat.mul_div_cancel_left 444 12] at h1
  have h3 : 444 = 12 * (444 / 12) := nat.mul_div_cancel_left 444 12
  rw h3
  rw pow_mul
  rw h2
  rw pow_one
  exact h2

end remainder_444_pow_444_mod_13_l109_109702


namespace difference_of_solutions_l109_109856

theorem difference_of_solutions (x : ℝ) (h : (x + 3)^2 / (3 * x + 65) = 2) : ∃ a b : ℝ, a ≠ b ∧ (x = a ∨ x = b) ∧ abs (a - b) = 22 :=
by
  sorry

end difference_of_solutions_l109_109856


namespace right_triangle_area_l109_109565

theorem right_triangle_area (a b : ℝ) (H₁ : a = 3) (H₂ : b = 5) : 
  1 / 2 * a * b = 7.5 := by
  rw [H₁, H₂]
  norm_num

end right_triangle_area_l109_109565


namespace number_of_integers_x_l109_109095

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_acute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2

def valid_range_x (x : ℝ) : Prop :=
  13 < x ∧ x < 43

def conditions_for_acute_triangle (x : ℝ) : Prop :=
  (x > 28 ∧ x^2 < 1009) ∨ (x ≤ 28 ∧ x > 23.64)

theorem number_of_integers_x (count : ℤ) :
  (∃ (x : ℤ), valid_range_x x ∧ is_triangle 15 28 x ∧ is_acute_triangle 15 28 x ∧ conditions_for_acute_triangle x) →
  count = 8 :=
sorry

end number_of_integers_x_l109_109095


namespace part_a_l109_109855

theorem part_a (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) : 
  |a - b| + |b - c| + |c - a| ≤ 2 * Real.sqrt 2 :=
sorry

end part_a_l109_109855


namespace new_number_formed_l109_109121

theorem new_number_formed (t u : ℕ) (ht : t < 10) (hu : u < 10) : 3 * 100 + (10 * t + u) = 300 + 10 * t + u := 
by {
  sorry
}

end new_number_formed_l109_109121


namespace parallel_lines_l109_109471

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x - a * y + 1 = 0) ∧ (∀ x y : ℝ, (a-1) * x - y + a = 0) →
  (a = 2 ↔ (∀ x1 y1 x2 y2 : ℝ, 2 * x1 - a * y1 + 1 = 0 ∧ (a-1) * x2 - y2 + a = 0 →
  (2 * x1 = (a * y1 - 1) ∧ (a-1) * x2 = y2 - a))) :=
sorry

end parallel_lines_l109_109471


namespace partition_with_sum_square_l109_109587

def sum_is_square (a b : ℕ) : Prop := ∃ k : ℕ, a + b = k * k

theorem partition_with_sum_square (n : ℕ) (h : n ≥ 15) :
  ∀ (s₁ s₂ : finset ℕ), (∅ ⊂ s₁ ∪ s₂ ∧ s₁ ∩ s₂ = ∅ ∧ (∀ x ∈ s₁ ∪ s₂, x ∈ finset.range (n + 1))) →
  (∃ a b : ℕ, a ≠ b ∧ (a ∈ s₁ ∧ b ∈ s₁ ∨ a ∈ s₂ ∧ b ∈ s₂) ∧ sum_is_square a b) :=
by sorry

end partition_with_sum_square_l109_109587


namespace pentagon_area_l109_109493

/-- Given a convex pentagon ABCDE where BE and CE are angle bisectors at vertices B and C 
respectively, with ∠A = 35 degrees, ∠D = 145 degrees, and the area of triangle BCE is 11, 
prove that the area of the pentagon ABCDE is 22. -/
theorem pentagon_area (ABCDE : Type) (angle_A : ℝ) (angle_D : ℝ) (area_BCE : ℝ)
  (h_A : angle_A = 35) (h_D : angle_D = 145) (h_area_BCE : area_BCE = 11) :
  ∃ (area_ABCDE : ℝ), area_ABCDE = 22 :=
by
  sorry

end pentagon_area_l109_109493


namespace rhombus_diagonal_length_l109_109161

-- Definitions of given conditions
def d1 : ℝ := 10
def Area : ℝ := 60

-- Proof of desired condition
theorem rhombus_diagonal_length (d2 : ℝ) : 
  (Area = d1 * d2 / 2) → d2 = 12 :=
by
  sorry

end rhombus_diagonal_length_l109_109161


namespace complement_intersection_l109_109510

theorem complement_intersection (A B U : Set ℕ) (hA : A = {4, 5, 7}) (hB : B = {3, 4, 7, 8}) (hU : U = A ∪ B) :
  U \ (A ∩ B) = {3, 5, 8} :=
by
  sorry

end complement_intersection_l109_109510


namespace total_price_is_correct_l109_109242

-- Define the cost of an adult ticket
def cost_adult : ℕ := 22

-- Define the cost of a children ticket
def cost_child : ℕ := 7

-- Define the number of adults in the family
def num_adults : ℕ := 2

-- Define the number of children in the family
def num_children : ℕ := 2

-- Define the total price the family will pay
def total_price : ℕ := cost_adult * num_adults + cost_child * num_children

-- The proof to check the total price
theorem total_price_is_correct : total_price = 58 :=
by 
  -- Here we would solve the proof
  sorry

end total_price_is_correct_l109_109242


namespace completing_square_result_l109_109368

theorem completing_square_result:
  ∀ x : ℝ, (x^2 + 4 * x - 1 = 0) → (x + 2)^2 = 5 :=
by
  sorry

end completing_square_result_l109_109368


namespace quotient_of_integers_l109_109851

theorem quotient_of_integers
  (a b : ℤ)
  (h : 1996 * a + b / 96 = a + b) :
  b / a = 2016 ∨ a / b = 2016 := 
sorry

end quotient_of_integers_l109_109851


namespace find_L_l109_109258

-- Conditions definitions
def initial_marbles := 57
def marbles_won_second_game := 25
def final_marbles := 64

-- Definition of L
def L := initial_marbles - 18

theorem find_L (L : ℕ) (H1 : initial_marbles = 57) (H2 : marbles_won_second_game = 25) (H3 : final_marbles = 64) : 
(initial_marbles - L) + marbles_won_second_game = final_marbles -> 
L = 18 :=
by
  sorry

end find_L_l109_109258


namespace bella_earrings_l109_109420

theorem bella_earrings (B M R : ℝ) 
  (h1 : B = 0.25 * M) 
  (h2 : M = 2 * R) 
  (h3 : B + M + R = 70) : 
  B = 10 := by 
  sorry

end bella_earrings_l109_109420


namespace fourth_roots_of_neg_16_l109_109093

theorem fourth_roots_of_neg_16 : 
  { z : ℂ | z^4 = -16 } = { sqrt (2 : ℂ) + sqrt (2 : ℂ) * complex.I, 
                            - sqrt (2 : ℂ) + sqrt (2 : ℂ) * complex.I, 
                            - sqrt (2 : ℂ) - sqrt (2 : ℂ) * complex.I, 
                            sqrt (2 : ℂ) - sqrt (2 : ℂ) * complex.I } :=
by
  sorry

end fourth_roots_of_neg_16_l109_109093


namespace ellipse_condition_l109_109528

theorem ellipse_condition (m : ℝ) :
  (1 < m ∧ m < 3) → ((m > 1 ∧ m < 3 ∧ m ≠ 2) ∨ (m = 2)) :=
by
  sorry

end ellipse_condition_l109_109528


namespace min_value_expression_l109_109846

theorem min_value_expression (x y : ℝ) : (x^2 + y^2 - 6 * x + 4 * y + 18) ≥ 5 :=
sorry

end min_value_expression_l109_109846


namespace number_multiplies_xz_l109_109934

theorem number_multiplies_xz (x y z w A B : ℝ) (h1 : 4 * x * z + y * w = 3) (h2 : x * w + y * z = 6) :
  A * B = 4 :=
sorry

end number_multiplies_xz_l109_109934


namespace parallel_lines_necessary_and_sufficient_l109_109467

-- Define the lines l1 and l2
def line1 (a : ℝ) (x y : ℝ) : Prop := 2 * x - a * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x - y + a = 0

-- State the theorem
theorem parallel_lines_necessary_and_sufficient (a : ℝ) :
  (∀ x y : ℝ, line1 a x y ↔ line2 a x y) ↔ a = 2 :=
by
  -- Proof omitted
  sorry

end parallel_lines_necessary_and_sufficient_l109_109467


namespace solve_for_x_l109_109935

theorem solve_for_x (x : ℤ) (h : (-1) * 2 * x * 4 = 24) : x = -3 := by
  sorry

end solve_for_x_l109_109935


namespace matrix_power_4_l109_109430

def matrix_exp := λ (A : Matrix (Fin 2) (Fin 2) ℤ) (n : ℕ), A ^ n

theorem matrix_power_4 :
  let A : Matrix (Fin 2) (Fin 2) ℤ := ![![2, -1], ![1, 1]]
  matrix_exp A 4 = ![![0, -9], ![9, -9]] :=
by
  sorry

end matrix_power_4_l109_109430


namespace jerry_reaches_3_at_some_time_l109_109640

def jerry_reaches_3_probability (n : ℕ) (k : ℕ) : ℚ :=
  -- This function represents the probability that Jerry reaches 3 at some point during n coin tosses
  if n = 7 ∧ k = 3 then (21 / 64 : ℚ) else 0

theorem jerry_reaches_3_at_some_time :
  jerry_reaches_3_probability 7 3 = (21 / 64 : ℚ) :=
sorry

end jerry_reaches_3_at_some_time_l109_109640


namespace anna_chocolates_l109_109877

theorem anna_chocolates : ∃ (n : ℕ), (5 * 2^(n-1) > 200) ∧ n = 7 :=
by
  sorry

end anna_chocolates_l109_109877


namespace min_distance_racetracks_l109_109518

theorem min_distance_racetracks : 
  ∀ A B : ℝ × ℝ, (A.1 ^ 2 + A.2 ^ 2 = 1) ∧ (((B.1 - 1) ^ 2) / 16 + (B.2 ^ 2) / 4 = 1) → 
  dist A B ≥ (Real.sqrt 33 - 3) / 3 := by
  sorry

end min_distance_racetracks_l109_109518


namespace sum_even_integers_102_to_200_l109_109176

theorem sum_even_integers_102_to_200 : 
  let sequence := list.range' 102 100 
  ∧ (∀ x ∈ sequence, x % 2 = 0) →
  list.sum sequence = 7550 := 
by 
  let sequence := list.range' 102 100 
  have even_sequence : ∀ x ∈ sequence, x % 2 = 0 := 
    sorry 
  have sum_sequence : list.sum sequence = 7550 := 
    sorry 
  exact sum_sequence 

end sum_even_integers_102_to_200_l109_109176


namespace probability_x_greater_3y_l109_109655

theorem probability_x_greater_3y (x y : ℝ) (h1: 0 ≤ x ∧ x ≤ 2010) (h2: 0 ≤ y ∧ y ≤ 2011) : 
  (let area_triangle := (2010 * (2010 / 3)) / 2 in 
   let area_rectangle := 2010 * 2011 in 
   (area_triangle / area_rectangle = 335 / 2011)) := 
sorry

end probability_x_greater_3y_l109_109655


namespace divisible_values_l109_109271

def is_digit (n : ℕ) : Prop :=
  n >= 0 ∧ n <= 9

def N (x y : ℕ) : ℕ :=
  30 * 10^7 + x * 10^6 + 7 * 10^4 + y * 10^3 + 3

def is_divisible_by_37 (n : ℕ) : Prop :=
  n % 37 = 0

theorem divisible_values :
  ∃ (x y : ℕ), is_digit x ∧ is_digit y ∧ is_divisible_by_37 (N x y) ∧ ((x, y) = (8, 1) ∨ (x, y) = (4, 4) ∨ (x, y) = (0, 7)) :=
by {
  sorry
}

end divisible_values_l109_109271


namespace second_race_distance_l109_109498

theorem second_race_distance (Va Vb Vc : ℝ) (D : ℝ)
  (h1 : Va / Vb = 10 / 9)
  (h2 : Va / Vc = 80 / 63)
  (h3 : Vb / Vc = D / (D - 100)) :
  D = 800 :=
sorry

end second_race_distance_l109_109498


namespace parallel_lines_necessary_and_sufficient_l109_109466

-- Define the lines l1 and l2
def line1 (a : ℝ) (x y : ℝ) : Prop := 2 * x - a * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x - y + a = 0

-- State the theorem
theorem parallel_lines_necessary_and_sufficient (a : ℝ) :
  (∀ x y : ℝ, line1 a x y ↔ line2 a x y) ↔ a = 2 :=
by
  -- Proof omitted
  sorry

end parallel_lines_necessary_and_sufficient_l109_109466


namespace linear_function_iff_l109_109849

variable {x : ℝ} (m : ℝ)

def f (m : ℝ) (x : ℝ) : ℝ := (m + 2) * x + 4 * x - 5

theorem linear_function_iff (m : ℝ) : 
  (∃ c d, ∀ x, f m x = c * x + d) ↔ m ≠ -6 :=
by 
  sorry

end linear_function_iff_l109_109849


namespace polygon_sides_from_diagonals_l109_109321

theorem polygon_sides_from_diagonals (n : ℕ) (h : ↑((n * (n - 3)) / 2) = 14) : n = 7 :=
by
  sorry

end polygon_sides_from_diagonals_l109_109321


namespace megan_markers_l109_109236

theorem megan_markers (initial_markers : ℕ) (new_markers : ℕ) (total_markers : ℕ) :
  initial_markers = 217 →
  new_markers = 109 →
  total_markers = 326 →
  initial_markers + new_markers = 326 :=
by
  sorry

end megan_markers_l109_109236


namespace find_x_l109_109838

theorem find_x (x : ℝ) (h : 0.75 * x + 2 = 8) : x = 8 :=
sorry

end find_x_l109_109838


namespace horner_eval_f_at_5_eval_f_at_5_l109_109454

def f (x: ℝ) : ℝ := x^5 - 2*x^4 + x^3 + x^2 - x - 5

theorem horner_eval_f_at_5 :
  f 5 = ((((5 - 2) * 5 + 1) * 5 + 1) * 5 - 1) * 5 - 5 := by
  sorry

theorem eval_f_at_5 : f 5 = 2015 := by 
  have h : f 5 = ((((5 - 2) * 5 + 1) * 5 + 1) * 5 - 1) * 5 - 5 := by
    apply horner_eval_f_at_5
  rw [h]
  norm_num

end horner_eval_f_at_5_eval_f_at_5_l109_109454


namespace solve_for_x_l109_109484

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 * x + 7 * x = 120) : x = 20 :=
by
  sorry

end solve_for_x_l109_109484


namespace prism_volume_l109_109871

open Real

theorem prism_volume :
  ∃ (a b c : ℝ), a * b = 15 ∧ b * c = 10 ∧ c * a = 30 ∧ a * b * c = 30 * sqrt 5 :=
by
  sorry

end prism_volume_l109_109871


namespace digit_150th_in_decimal_of_fraction_l109_109218

theorem digit_150th_in_decimal_of_fraction : 
  (∀ n : ℕ, n > 0 → (let seq := [2, 4, 2, 8, 5, 7] in seq[((n - 1) % 6)] = 
  if n == 150 then 7 else seq[((n - 1) % 6)])) :=
by
  sorry

end digit_150th_in_decimal_of_fraction_l109_109218


namespace find_y_l109_109132

-- Definitions of angles and the given problem.
def angle_ABC : ℝ := 90
def angle_ABD (y : ℝ) : ℝ := 3 * y
def angle_DBC (y : ℝ) : ℝ := 2 * y

-- The theorem stating the problem
theorem find_y (y : ℝ) (h1 : angle_ABC = 90) (h2 : angle_ABD y + angle_DBC y = angle_ABC) : y = 18 :=
  by 
  sorry

end find_y_l109_109132


namespace task1_task2_task3_l109_109788

noncomputable def f (x a : ℝ) := x^2 - 4 * x + a + 3
noncomputable def g (x m : ℝ) := m * x + 5 - 2 * m

theorem task1 (a m : ℝ) (h₁ : a = -3) (h₂ : m = 0) :
  (∃ x : ℝ, f x a - g x m = 0) ↔ x = -1 ∨ x = 5 :=
sorry

theorem task2 (a : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x a = 0) ↔ -8 ≤ a ∧ a ≤ 0 :=
sorry

theorem task3 (m : ℝ) :
  (∀ x₁ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 4 → ∃ x₂ : ℝ, 1 ≤ x₂ ∧ x₂ ≤ 4 ∧ f x₁ 0 = g x₂ m) ↔ m ≤ -3 ∨ 6 ≤ m :=
sorry

end task1_task2_task3_l109_109788


namespace four_faucets_fill_time_correct_l109_109800

-- Define the parameters given in the conditions
def three_faucets_rate (volume : ℕ) (time : ℕ) := volume / time
def one_faucet_rate (rate : ℕ) := rate / 3
def four_faucets_rate (rate : ℕ) := 4 * rate
def fill_time (volume : ℕ) (rate : ℕ) := volume / rate

-- Given problem parameters
def volume_large_tub : ℕ := 100
def time_large_tub : ℕ := 6
def volume_small_tub : ℕ := 50

-- Theorem to be proven
theorem four_faucets_fill_time_correct :
  fill_time volume_small_tub (four_faucets_rate (one_faucet_rate (three_faucets_rate volume_large_tub time_large_tub))) * 60 = 135 :=
sorry

end four_faucets_fill_time_correct_l109_109800


namespace max_m_value_min_value_expression_l109_109962

-- Define the conditions for the inequality where the solution is the entire real line
theorem max_m_value (x m : ℝ) :
  (∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m) → m ≤ 1 :=
sorry

-- Define the conditions for a, b, c > 0 and their sum equal to 1
-- and prove the minimum value of 4a^2 + 9b^2 + c^2
theorem min_value_expression (a b c : ℝ) (hpos1 : a > 0) (hpos2 : b > 0) (hpos3 : c > 0) (hsum : a + b + c = 1) :
  4 * a^2 + 9 * b^2 + c^2 ≥ 36 / 49 ∧ (4 * a^2 + 9 * b^2 + c^2 = 36 / 49 → a = 9 / 49 ∧ b = 4 / 49 ∧ c = 36 / 49) :=
sorry

end max_m_value_min_value_expression_l109_109962


namespace sum_of_squares_eq_two_l109_109618

theorem sum_of_squares_eq_two {a b : ℝ} (h : (a^2 + b^2) * (a^2 + b^2 + 4) = 12) : a^2 + b^2 = 2 := sorry

end sum_of_squares_eq_two_l109_109618


namespace find_second_offset_l109_109771

theorem find_second_offset 
  (diagonal : ℝ) (offset1 : ℝ) (area_quad : ℝ) (offset2 : ℝ)
  (h1 : diagonal = 20) (h2 : offset1 = 9) (h3 : area_quad = 150) :
  offset2 = 6 :=
by
  sorry

end find_second_offset_l109_109771


namespace arc_length_of_circle_l109_109633

theorem arc_length_of_circle (r : ℝ) (θ_peripheral : ℝ) (h_r : r = 5) (h_θ : θ_peripheral = 2/3 * π) :
  r * (2/3 * θ_peripheral) = 20 * π / 3 := 
by sorry

end arc_length_of_circle_l109_109633


namespace unique_N_l109_109087

-- Given conditions and question in the problem
variable (N : Matrix (Fin 2) (Fin 2) ℝ)

-- Problem statement: prove that the matrix defined below is the only matrix satisfying the given condition
theorem unique_N 
  (h : ∀ (w : Fin 2 → ℝ), N.mulVec w = -7 • w) 
  : N = ![![-7, 0], ![0, -7]] := 
sorry

end unique_N_l109_109087


namespace digit_150_of_17_div_70_is_2_l109_109224

def repeating_cycle_17_div_70 : List ℕ := [2, 4, 2, 8, 5, 7, 1]

theorem digit_150_of_17_div_70_is_2 : 
  (repeating_cycle_17_div_70[(150 % 7) - 1] = 2) :=
by
  -- the proof will go here
  sorry

end digit_150_of_17_div_70_is_2_l109_109224


namespace maximum_a_value_l109_109281

theorem maximum_a_value :
  ∀ (a : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → -2022 ≤ (a + 1)*x^2 - (a + 1)*x + 2022 ∧ (a + 1)*x^2 - (a + 1)*x + 2022 ≤ 2022) →
  a ≤ 16175 := 
by {
  sorry
}

end maximum_a_value_l109_109281


namespace common_ratio_is_63_98_l109_109578

/-- Define the terms of the geometric series -/
def term (n : Nat) : ℚ := 
  match n with
  | 0 => 4 / 7
  | 1 => 18 / 49
  | 2 => 162 / 343
  | _ => sorry  -- For simplicity, we can define more terms if needed, but it's irrelevant for our proof

/-- Define the common ratio of the geometric series -/
def common_ratio (a b : ℚ) : ℚ := b / a

/-- The problem states that the common ratio of first two terms of the given series is equal to 63/98 -/
theorem common_ratio_is_63_98 : common_ratio (term 0) (term 1) = 63 / 98 :=
by
  -- leave the proof as sorry for now
  sorry

end common_ratio_is_63_98_l109_109578


namespace six_times_more_coats_l109_109688

/-- The number of lab coats is 6 times the number of uniforms. --/
def coats_per_uniforms (c u : ℕ) : Prop := c = 6 * u

/-- There are 12 uniforms. --/
def uniforms : ℕ := 12

/-- Each lab tech gets 14 coats and uniforms in total. --/
def total_per_tech : ℕ := 14

/-- Show that the number of lab coats is 6 times the number of uniforms. --/
theorem six_times_more_coats (c u : ℕ) (h1 : coats_per_uniforms c u) (h2 : u = 12) :
  c / u = 6 :=
by
  sorry

end six_times_more_coats_l109_109688


namespace standard_lamp_probability_l109_109081

-- Define the given probabilities
def P_A1 : ℝ := 0.45
def P_A2 : ℝ := 0.40
def P_A3 : ℝ := 0.15

def P_B_given_A1 : ℝ := 0.70
def P_B_given_A2 : ℝ := 0.80
def P_B_given_A3 : ℝ := 0.81

-- Define the calculation for the total probability of B
def P_B : ℝ := P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3

-- The statement to prove
theorem standard_lamp_probability : P_B = 0.7565 := by sorry

end standard_lamp_probability_l109_109081


namespace find_angle_C_find_side_c_l109_109612

noncomputable section

-- Definitions and conditions for Part 1
def vectors_dot_product_sin_2C (A B C : ℝ) (m : ℝ × ℝ) (n : ℝ × ℝ) : Prop :=
  m = (Real.sin A, Real.cos A) ∧ n = (Real.cos B, Real.sin B) ∧ 
  ((m.1 * n.1 + m.2 * n.2) = Real.sin (2 * C))

def angles_of_triangle (A B C : ℝ) : Prop := 
  A + B + C = Real.pi

theorem find_angle_C (A B C : ℝ) (m n : ℝ × ℝ) :
  vectors_dot_product_sin_2C A B C m n → angles_of_triangle A B C → C = Real.pi / 3 :=
sorry

-- Definitions and conditions for Part 2
def sin_in_arithmetic_sequence (x y z : ℝ) : Prop :=
  x + z = 2 * y

def product_of_sides_cos_C (a b c : ℝ) (C : ℝ) : Prop :=
  (a * b * Real.cos C = 18) ∧ (Real.cos C = 1 / 2)

theorem find_side_c (A B C a b c : ℝ) (m n : ℝ × ℝ) :
  sin_in_arithmetic_sequence (Real.sin A) (Real.sin C) (Real.sin B) → 
  angles_of_triangle A B C → 
  product_of_sides_cos_C a b c C → 
  C = Real.pi / 3 → 
  c = 6 :=
sorry

end find_angle_C_find_side_c_l109_109612


namespace parallel_lines_necessary_and_sufficient_l109_109468

-- Define the lines l1 and l2
def line1 (a : ℝ) (x y : ℝ) : Prop := 2 * x - a * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x - y + a = 0

-- State the theorem
theorem parallel_lines_necessary_and_sufficient (a : ℝ) :
  (∀ x y : ℝ, line1 a x y ↔ line2 a x y) ↔ a = 2 :=
by
  -- Proof omitted
  sorry

end parallel_lines_necessary_and_sufficient_l109_109468


namespace midpoint_trajectory_of_chord_l109_109785

theorem midpoint_trajectory_of_chord {x y : ℝ} :
  (∃ (A B : ℝ × ℝ), 
    (A.1^2 / 3 + A.2^2 = 1) ∧ 
    (B.1^2 / 3 + B.2^2 = 1) ∧ 
    ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (x, y) ∧ 
    ∃ t : ℝ, ((-1, 0) = ((1 - t) * A.1 + t * B.1, (1 - t) * A.2 + t * B.2))) -> 
  x^2 + x + 3 * y^2 = 0 :=
by sorry

end midpoint_trajectory_of_chord_l109_109785


namespace probability_male_female_ratio_l109_109450

theorem probability_male_female_ratio :
  let total_possibilities := Nat.choose 9 5
  let specific_scenarios := Nat.choose 5 2 * Nat.choose 4 3 + Nat.choose 5 3 * Nat.choose 4 2
  let probability := specific_scenarios / (total_possibilities : ℚ)
  probability = 50 / 63 :=
by 
  sorry

end probability_male_female_ratio_l109_109450


namespace probability_x_gt_3y_l109_109659

/--
Prove that if a point (x, y) is randomly picked from the rectangular region with vertices
at (0,0), (2010,0), (2010,2011), and (0,2011), the probability that x > 3y is 335/2011.
-/
theorem probability_x_gt_3y (x y : ℝ) :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 2010) ∧ (0 ≤ y ∧ y ≤ 2011) ∧ True -> 
  sorry :=
begin
  sorry
end

end probability_x_gt_3y_l109_109659


namespace inequality_solution_l109_109672

theorem inequality_solution (x : ℝ) (hx : x ≥ 0) :
  2021 * (x ^ (2021 / 202.0)) - 1 = 2020 * x → x = 1 :=
by 
  sorry

end inequality_solution_l109_109672


namespace simplify_expression_l109_109152

open Real

theorem simplify_expression (a b : ℝ) 
  (h1 : b ≠ 0) (h2 : b ≠ -3 * a) (h3 : b ≠ a) (h4 : b ≠ -a) : 
  ((2 * b + a - (4 * a ^ 2 - b ^ 2) / a) / (b ^ 3 + 2 * a * b ^ 2 - 3 * a ^ 2 * b)) *
  ((a ^ 3 * b - 2 * a ^ 2 * b ^ 2 + a * b ^ 3) / (a ^ 2 - b ^ 2)) = 
  (a - b) / (a + b) :=
by
  sorry

end simplify_expression_l109_109152


namespace greatest_power_of_2_divides_10_1004_minus_4_502_l109_109690

theorem greatest_power_of_2_divides_10_1004_minus_4_502 :
  ∃ k, 10^1004 - 4^502 = 2^1007 * k :=
sorry

end greatest_power_of_2_divides_10_1004_minus_4_502_l109_109690


namespace multiple_of_24_l109_109020

theorem multiple_of_24 (n : ℕ) (h : n > 0) : 
  ∃ k₁ k₂ : ℕ, (6 * n - 1)^2 - 1 = 24 * k₁ ∧ (6 * n + 1)^2 - 1 = 24 * k₂ :=
by
  sorry

end multiple_of_24_l109_109020


namespace matrix_pow_A4_l109_109429

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -1], ![1, 1]]

-- State the theorem
theorem matrix_pow_A4 :
  A^4 = ![![0, -9], ![9, -9]] :=
by
  sorry -- Proof is omitted

end matrix_pow_A4_l109_109429


namespace first_more_than_200_paperclips_day_l109_109261

-- Definitions based on the conditions:
def paperclips_on_day (k : ℕ) : ℕ :=
  3 * 2^k

-- The theorem stating the solution:
theorem first_more_than_200_paperclips_day :
  ∃ k : ℕ, paperclips_on_day k > 200 ∧ k = 7 :=
by
  use 7
  sorry

end first_more_than_200_paperclips_day_l109_109261


namespace proof_problem_l109_109113

-- Given conditions for propositions p and q
def p (a : ℝ) := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) := ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

-- Combined proposition p and q
def p_and_q (a : ℝ) := p a ∧ q a

-- Statement of the proof problem: Prove that p_and_q a → a ≤ -1
theorem proof_problem (a : ℝ) : p_and_q a → (a ≤ -1) :=
by
  sorry

end proof_problem_l109_109113


namespace simplify_sqrt_is_cos_20_l109_109820

noncomputable def simplify_sqrt : ℝ :=
  let θ : ℝ := 160 * Real.pi / 180
  Real.sqrt (1 - Real.sin θ ^ 2)

theorem simplify_sqrt_is_cos_20 : simplify_sqrt = Real.cos (20 * Real.pi / 180) :=
  sorry

end simplify_sqrt_is_cos_20_l109_109820


namespace turn_all_black_l109_109495

def invertColor (v : Vertex) (G : Graph) : Graph := sorry

theorem turn_all_black (G : Graph) (n : ℕ) (whiteBlack : Vertex → Bool) :
  (∀ v : Vertex, whiteBlack v = false) :=
by
 -- Providing the base case for induction
  induction n with 
  | zero => sorry -- The base case for graphs with one vertex
  | succ n ih =>
    -- Inductive step: assume true for graph with n vertices and prove for graph with n+1 vertices
    sorry

end turn_all_black_l109_109495


namespace ellipse_semi_minor_axis_is_2_sqrt_3_l109_109130

/-- 
  Given an ellipse with the center at (2, -1), 
  one focus at (2, -3), and one endpoint of a semi-major axis at (2, 3), 
  we prove that the semi-minor axis is 2√3.
-/
theorem ellipse_semi_minor_axis_is_2_sqrt_3 :
  let center := (2, -1)
  let focus := (2, -3)
  let endpoint := (2, 3)
  let c := Real.sqrt ((2 - 2)^2 + (-3 + 1)^2)
  let a := Real.sqrt ((2 - 2)^2 + (3 + 1)^2)
  let b2 := a^2 - c^2
  let b := Real.sqrt b2
  c = 2 ∧ a = 4 ∧ b = 2 * Real.sqrt 3 := 
by
  sorry

end ellipse_semi_minor_axis_is_2_sqrt_3_l109_109130


namespace sequence_sum_relation_l109_109783

theorem sequence_sum_relation (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n, 4 * S n = (a n + 1) ^ 2) →
  (S 1 = a 1) →
  a 1 = 1 →
  (∀ n, a (n + 1) = a n + 2) →
  a 2023 = 4045 :=
by
  sorry

end sequence_sum_relation_l109_109783


namespace common_chord_eqn_l109_109791

theorem common_chord_eqn (x y : ℝ) :
  (x^2 + y^2 + 2 * x - 6 * y + 1 = 0) ∧
  (x^2 + y^2 - 4 * x + 2 * y - 11 = 0) →
  3 * x - 4 * y + 6 = 0 :=
by
  intro h
  sorry

end common_chord_eqn_l109_109791


namespace part1_part2_l109_109687

-- Define the conditions for problem (1)
def condition1 := ∀ (arrangement : List ℕ), (arrangement.length = 5 ∧ arrangement.head ≠ A ∧ arrangement.head ≠ E ∧ arrangement.last ≠ A ∧ arrangement.last ≠ E)

-- Define the conditions for problem (2)
def condition2 := ∀ (arrangement : List ℕ), (arrangement.length = 5 ∧ ((A, B) ∈ zip arrangement (tail arrangement) ∨ (B, A) ∈ zip arrangement (tail arrangement)) ∧ ¬(C, D) ∈ zip arrangement (tail arrangement) ∧ ¬(D, C) ∈ zip arrangement (tail arrangement))

-- Part (1) proof statement
theorem part1 : condition1 → ∃ arrangement : List ℕ, arrangement.length = 5 ∧ A ∉ [arrangement.head, arrangement.last] ∧ E ∉ [arrangement.head, arrangement.last] ∧ arrangement.count A = 1 × arrangement.count B = 1 × arrangement.count C = 1 × arrangement.count D = 1 × arrangement.count E = 1 := 
sorry

-- Part (2) proof statement
theorem part2 : condition2 → ∃ arrangement : List ℕ, arrangement.length = 5 ∧ (A, B) ∈ zip arrangement arrangement.tail ∨ (B, A) ∈ zip arrangement arrangement.tail ∧ ¬(C, D) ∈ zip arrangement arrangement.tail ∧ ¬(D, C) ∈ zip arrangement arrangement.tail ∧ arrangement.count A = 1 × arrangement.count B = 1 × arrangement.count C = 1 × arrangement.count D = 1 × arrangement.count E = 1 := 
sorry

end part1_part2_l109_109687


namespace consecutive_reposition_transformation_fixed_point_l109_109516

def num_of_digits (n : ℕ) : ℕ := n.digits 10 |> List.length

def num_of_odd_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.countp (λ d, d % 2 = 1)

def num_of_even_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.countp (λ d, d % 2 = 0)

def reposition_transformation (n : ℕ) : ℕ :=
  let a := num_of_digits n
  let b := num_of_odd_digits n
  let c := num_of_even_digits n
  a * 100 + b * 10 + c

theorem consecutive_reposition_transformation_fixed_point (n : ℕ) (h : 1000 ≤ n ∧ n < 10000) :
  ∃ k : ℕ, ∀ m : ℕ, reposition_transformation^[m] n = k :=
sorry

end consecutive_reposition_transformation_fixed_point_l109_109516


namespace sequence_sum_of_geometric_progressions_l109_109840

theorem sequence_sum_of_geometric_progressions
  (u1 v1 q p : ℝ)
  (h1 : u1 + v1 = 0)
  (h2 : u1 * q + v1 * p = 0) :
  u1 * q^2 + v1 * p^2 = 0 :=
by sorry

end sequence_sum_of_geometric_progressions_l109_109840


namespace characters_per_day_l109_109983

-- Definitions based on conditions
def chars_total_older : ℕ := 8000
def chars_total_younger : ℕ := 6000
def chars_per_day_diff : ℕ := 100

-- Define the main theorem
theorem characters_per_day (x : ℕ) :
  chars_total_older / x = chars_total_younger / (x - chars_per_day_diff) := 
sorry

end characters_per_day_l109_109983


namespace exists_n_not_coprime_l109_109297

theorem exists_n_not_coprime (p q : ℕ) (h1 : Nat.gcd p q = 1) (h2 : q > p) (h3 : q - p > 1) :
  ∃ (n : ℕ), Nat.gcd (p + n) (q + n) ≠ 1 :=
by
  sorry

end exists_n_not_coprime_l109_109297


namespace discount_coupon_value_l109_109057

theorem discount_coupon_value :
  let hamburger_cost := 2 * 5
  let cola_cost := 3 * 2
  let total_cost := hamburger_cost + cola_cost
  let amount_paid := 12
  total_cost - amount_paid = 4 := by
  intros
  let hamburger_cost := 2 * 5
  let cola_cost := 3 * 2
  let total_cost := hamburger_cost + cola_cost
  let amount_paid := 12
  show total_cost - amount_paid = 4
  sorry

end discount_coupon_value_l109_109057


namespace sin_70_given_sin_10_l109_109914

theorem sin_70_given_sin_10 (k : ℝ) (h : Real.sin 10 = k) : Real.sin 70 = 1 - 2 * k^2 := 
by 
  sorry

end sin_70_given_sin_10_l109_109914


namespace translation_of_graph_l109_109164

theorem translation_of_graph (f : ℝ → ℝ) (x : ℝ) :
  f x = 2 ^ x →
  f (x - 1) + 2 = 2 ^ (x - 1) + 2 :=
by
  intro
  sorry

end translation_of_graph_l109_109164


namespace num_sixth_powers_below_1000_l109_109309

theorem num_sixth_powers_below_1000 : 
  {n : ℕ | n^6 < 1000 ∧ n > 0}.to_finset.card = 3 :=
by sorry

end num_sixth_powers_below_1000_l109_109309


namespace apples_per_person_l109_109474

-- Define conditions
def total_apples : ℝ := 45
def number_of_people : ℝ := 3.0

-- Theorem statement: Calculate how many apples each person received.
theorem apples_per_person : 
  (total_apples / number_of_people) = 15 := 
by
  sorry

end apples_per_person_l109_109474


namespace large_circle_radius_l109_109388

noncomputable def radius_of_large_circle (R : ℝ) : Prop :=
  ∃ r : ℝ, (r = 2) ∧
           (R = r + r) ∧
           (r = 2) ∧
           (R - r = 2) ∧
           (R = 4)

theorem large_circle_radius :
  radius_of_large_circle 4 :=
by
  sorry

end large_circle_radius_l109_109388


namespace eta_zero_ae_l109_109810

noncomputable section

variable {Ω : Type*} [ProbabilitySpace Ω]

-- Define the random variables ξ and η
variable (ξ η : Ω → ℝ)

-- Define independence of ξ and η
variable (h_ind : Independency ξ η)

-- Define the condition that the distribution of ξ + η is the same as the distribution of ξ
variable (h_dist : Distribution (ξ + η) = Distribution ξ)

-- Goal: Prove that η = 0 almost surely
theorem eta_zero_ae : η =ᵐ[ProbabilitySpace] (λ _ : Ω, 0) :=
sorry

end eta_zero_ae_l109_109810


namespace quadratic_solution_l109_109780

-- Definition of the quadratic function satisfying the given conditions
def quadraticFunc (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, ∃ a b c : ℝ, f x = a * x^2 + b * x + c) ∧
  (∀ x : ℝ, f x < 0 ↔ 0 < x ∧ x < 5) ∧
  (f (-1) = 12 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 4 → f x ≤ 12)

-- The proof goal: proving the function f(x) is 2x^2 - 10x
theorem quadratic_solution (f : ℝ → ℝ) (h : quadraticFunc f) : ∀ x, f x = 2 * x^2 - 10 * x :=
by
  sorry

end quadratic_solution_l109_109780


namespace backpack_prices_purchasing_plans_backpacks_given_away_l109_109406

-- Part 1: Prices of Type A and Type B backpacks
theorem backpack_prices (x y : ℝ) (h1 : x = 2 * y - 30) (h2 : 2 * x + 3 * y = 255) : x = 60 ∧ y = 45 :=
sorry

-- Part 2: Possible purchasing plans
theorem purchasing_plans (m : ℕ) (h1 : 8900 ≥ 50 * m + 40 * (200 - m)) (h2 : m > 87) : 
  m = 88 ∨ m = 89 ∨ m = 90 :=
sorry

-- Part 3: Number of backpacks given away
theorem backpacks_given_away (m n : ℕ) (total_A : ℕ := 89) (total_B : ℕ := 111) 
(h1 : m + n = 4) 
(h2 : 1250 = (total_A - if total_A > 10 then total_A / 10 else 0) * 60 + (total_B - if total_B > 10 then total_B / 10 else 0) * 45 - (50 * total_A + 40 * total_B)) :
m = 1 ∧ n = 3 := 
sorry

end backpack_prices_purchasing_plans_backpacks_given_away_l109_109406


namespace max_m_plus_n_l109_109789

theorem max_m_plus_n (m n : ℝ) (h : n = -m^2 - 3*m + 3) : m + n ≤ 4 :=
by {
  sorry
}

end max_m_plus_n_l109_109789


namespace second_polygon_sides_l109_109040

theorem second_polygon_sides (s : ℝ) (h : s > 0) :
  let s1 := 3 * s in
  let s2 := s in
  let sides1 := 50 in
  let perimeter1 := sides1 * s1 in
  let perimeter2 := perimeter1 in
  perimeter2 / s2 = 150 := 
by
  let s1 := 3 * s
  let s2 := s
  let sides1 := 50
  let perimeter1 := sides1 * s1
  let perimeter2 := perimeter1
  show perimeter2 / s2 = 150
  sorry

end second_polygon_sides_l109_109040


namespace pages_remaining_l109_109167

def total_pages : ℕ := 120
def science_project_pages : ℕ := (25 * total_pages) / 100
def math_homework_pages : ℕ := 10
def total_used_pages : ℕ := science_project_pages + math_homework_pages
def remaining_pages : ℕ := total_pages - total_used_pages

theorem pages_remaining : remaining_pages = 80 := by
  sorry

end pages_remaining_l109_109167


namespace plates_difference_l109_109538

def num_plates_sunshine := 26^3 * 10^3
def num_plates_prairie := 26^2 * 10^4
def difference := num_plates_sunshine - num_plates_prairie

theorem plates_difference :
  difference = 10816000 := by sorry

end plates_difference_l109_109538


namespace value_of_a_l109_109630

theorem value_of_a (a b : ℚ) (h₁ : b = 3 * a) (h₂ : b = 12 - 5 * a) : a = 3 / 2 :=
by
  sorry

end value_of_a_l109_109630


namespace problem_statement_l109_109735

-- Define line and plane as types
variable (Line Plane : Type)

-- Define the perpendicularity and parallelism relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLPlane : Line → Plane → Prop)
variable (perpendicularPPlane : Plane → Plane → Prop)

-- Distinctness of lines and planes
variable (a b : Line)
variable (α β : Plane)

-- Conditions given in the problem
axiom distinct_lines : a ≠ b
axiom distinct_planes : α ≠ β

-- Statement to be proven
theorem problem_statement :
  perpendicular a b → 
  perpendicularLPlane a α → 
  perpendicularLPlane b β → 
  perpendicularPPlane α β :=
sorry

end problem_statement_l109_109735


namespace contrapositive_of_zero_squared_l109_109827

theorem contrapositive_of_zero_squared {x y : ℝ} :
  (x^2 + y^2 = 0 → x = 0 ∧ y = 0) →
  (x ≠ 0 ∨ y ≠ 0 → x^2 + y^2 ≠ 0) :=
by
  intro h1
  intro h2
  sorry

end contrapositive_of_zero_squared_l109_109827


namespace find_n_mod_60_l109_109480

theorem find_n_mod_60 {x y : ℤ} (hx : x ≡ 45 [ZMOD 60]) (hy : y ≡ 98 [ZMOD 60]) :
  ∃ n, 150 ≤ n ∧ n ≤ 210 ∧ (x - y ≡ n [ZMOD 60]) ∧ n = 187 := by
  sorry

end find_n_mod_60_l109_109480


namespace sin_70_given_sin_10_l109_109915

theorem sin_70_given_sin_10 (k : ℝ) (h : Real.sin 10 = k) : Real.sin 70 = 1 - 2 * k^2 := 
by 
  sorry

end sin_70_given_sin_10_l109_109915


namespace sum_of_integers_ways_l109_109151

theorem sum_of_integers_ways (n : ℕ) (h : n > 0) : 
  ∃ ways : ℕ, ways = 2^(n-1) := sorry

end sum_of_integers_ways_l109_109151


namespace initial_acidic_liquid_quantity_l109_109995

theorem initial_acidic_liquid_quantity
  (A : ℝ) -- initial quantity of the acidic liquid in liters
  (W : ℝ) -- quantity of water to be removed in liters
  (h1 : W = 6)
  (h2 : (0.40 * A) = 0.60 * (A - W)) : 
  A = 18 :=
by sorry

end initial_acidic_liquid_quantity_l109_109995


namespace parrot_initial_phrases_l109_109600

theorem parrot_initial_phrases (current_phrases : ℕ) (days_with_parrot : ℕ) (phrases_per_week : ℕ) (initial_phrases : ℕ) :
  current_phrases = 17 →
  days_with_parrot = 49 →
  phrases_per_week = 2 →
  initial_phrases = current_phrases - phrases_per_week * (days_with_parrot / 7) :=
by
  sorry

end parrot_initial_phrases_l109_109600


namespace largest_consecutive_odd_integer_sum_l109_109181

theorem largest_consecutive_odd_integer_sum
  (x : Real)
  (h_sum : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = -378.5) :
  x + 8 = -79.7 + 8 :=
by
  sorry

end largest_consecutive_odd_integer_sum_l109_109181


namespace number_of_boys_in_second_group_l109_109118

noncomputable def daily_work_done_by_man (M : ℝ) (B : ℝ) : Prop :=
  M = 2 * B

theorem number_of_boys_in_second_group
  (M B : ℝ)
  (h1 : (12 * M + 16 * B) * 5 = (13 * M + 24 * B) * 4)
  (h2 : daily_work_done_by_man M B) :
  24 = 24 :=
by
  -- The proof is omitted.
  sorry

end number_of_boys_in_second_group_l109_109118


namespace emily_initial_marbles_l109_109082

open Nat

theorem emily_initial_marbles (E : ℕ) (h : 3 * E - (3 * E / 2 + 1) = 8) : E = 6 :=
sorry

end emily_initial_marbles_l109_109082


namespace total_cost_is_9_43_l109_109338

def basketball_game_cost : ℝ := 5.20
def racing_game_cost : ℝ := 4.23
def total_cost : ℝ := basketball_game_cost + racing_game_cost

theorem total_cost_is_9_43 : total_cost = 9.43 := by
  sorry

end total_cost_is_9_43_l109_109338


namespace arcsin_sqrt2_div2_l109_109574

theorem arcsin_sqrt2_div2 :
  Real.arcsin (Real.sqrt 2 / 2) = Real.pi / 4 :=
sorry

end arcsin_sqrt2_div2_l109_109574


namespace f_is_increasing_l109_109951

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) + 3 * x

theorem f_is_increasing : ∀ (x : ℝ), (deriv f x) > 0 :=
by
  intro x
  calc
    deriv f x = 2 * Real.exp (2 * x) + 3 := by sorry
    _ > 0 := by sorry

end f_is_increasing_l109_109951


namespace six_circles_distance_relation_l109_109582

/--
Prove that for any pair of non-touching circles (among six circles where each touches four of the remaining five),
their radii \( r_1 \) and \( r_2 \) and the distance \( d \) between their centers satisfy 

\[ d^{2}=r_{1}^{2}+r_{2}^{2} \pm 6r_{1}r_{2} \]

("plus" if the circles do not lie inside one another, "minus" otherwise).
-/
theorem six_circles_distance_relation 
  (r1 r2 d : ℝ) 
  (h : ∀ i : Fin 6, i < 6 → ∃ c : ℝ, (c = r1 ∨ c = r2) ∧ ∀ j : Fin 6, j ≠ i → abs (c - j) ≠ d ) :
  d^2 = r1^2 + r2^2 + 6 * r1 * r2 ∨ d^2 = r1^2 + r2^2 - 6 * r1 * r2 := 
  sorry

end six_circles_distance_relation_l109_109582


namespace g_1986_l109_109320

def g : ℕ → ℤ := sorry

axiom g_def : ∀ n : ℕ, g n ≥ 0
axiom g_one : g 1 = 3
axiom g_func_eq : ∀ (a b : ℕ), g (a + b) = g a + g b - 3 * g (a * b)

theorem g_1986 : g 1986 = 0 :=
by
  sorry

end g_1986_l109_109320


namespace integer_solutions_of_inequality_l109_109970

theorem integer_solutions_of_inequality (x : ℤ) : 
  (-4 < 1 - 3 * (x: ℤ) ∧ 1 - 3 * (x: ℤ) ≤ 4) ↔ (x = -1 ∨ x = 0 ∨ x = 1) := 
by 
  sorry

end integer_solutions_of_inequality_l109_109970


namespace find_d_l109_109346

-- Define AP terms as S_n = a + (n-1)d, sum of first 10 terms, and difference expression
def arithmetic_progression (S : ℕ → ℕ) (a d : ℕ) : Prop :=
  ∀ n, S n = a + (n - 1) * d

def sum_first_ten (S : ℕ → ℕ) : Prop :=
  (S 1) + (S 2) + (S 3) + (S 4) + (S 5) + (S 6) + (S 7) + (S 8) + (S 9) + (S 10) = 55

def difference_expression (S : ℕ → ℕ) (d : ℕ) : Prop :=
  (S 10 - S 8) + (S 9 - S 7) + (S 8 - S 6) + (S 7 - S 5) + (S 6 - S 4) +
  (S 5 - S 3) + (S 4 - S 2) + (S 3 - S 1) = d

theorem find_d : ∃ (d : ℕ) (S : ℕ → ℕ) (a : ℕ), 
  (∀ n, S n = a + (n - 1) * d) ∧ 
  (S 1) + (S 2) + (S 3) + (S 4) + (S 5) + (S 6) + (S 7) + (S 8) + (S 9) + (S 10) = 55 ∧
  (S 10 - S 8) + (S 9 - S 7) + (S 8 - S 6) + (S 7 - S 5) + (S 6 - S 4) +
  (S 5 - S 3) + (S 4 - S 2) + (S 3 - S 1) = 16 :=
by
  sorry  -- proof is not required

end find_d_l109_109346


namespace area_of_curves_l109_109444

noncomputable def enclosed_area : ℝ :=
  ∫ x in (0:ℝ)..1, (Real.sqrt x - x^2)

theorem area_of_curves :
  enclosed_area = 1 / 3 :=
sorry

end area_of_curves_l109_109444


namespace solve_z_pow_eq_neg_sixteen_l109_109092

theorem solve_z_pow_eq_neg_sixteen (z : ℂ) :
  z^4 = -16 ↔ 
  z = complex.sqrt(2) + complex.I * complex.sqrt(2) ∨ 
  z = -complex.sqrt(2) + complex.I * complex.sqrt(2) ∨ 
  z = -complex.sqrt(2) - complex.I * complex.sqrt(2) ∨ 
  z = complex.sqrt(2) - complex.I * complex.sqrt(2) :=
by
  sorry

end solve_z_pow_eq_neg_sixteen_l109_109092


namespace three_by_three_grid_prob_odd_sum_l109_109802

noncomputable def prob_odd_sum_in_rows : ℚ :=
  let total_ways := (9!).toNat
  let valid_ways := 40^3
  (valid_ways : ℚ) / total_ways

theorem three_by_three_grid_prob_odd_sum :
  prob_odd_sum_in_rows = 4 / 227 :=
by
  -- stmt: This part computes the exact values and verifies that the ratio is correct
  sorry

end three_by_three_grid_prob_odd_sum_l109_109802


namespace convex_m_gons_two_acute_angles_l109_109287

noncomputable def count_convex_m_gons_with_two_acute_angles (m n : ℕ) (P : Finset ℕ) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

theorem convex_m_gons_two_acute_angles {m n : ℕ} {P : Finset ℕ}
  (hP : P.card = 2 * n + 1)
  (hmn : 4 < m ∧ m < n) :
  count_convex_m_gons_with_two_acute_angles m n P = 
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1)) :=
sorry

end convex_m_gons_two_acute_angles_l109_109287


namespace digit_150th_in_decimal_of_fraction_l109_109216

theorem digit_150th_in_decimal_of_fraction : 
  (∀ n : ℕ, n > 0 → (let seq := [2, 4, 2, 8, 5, 7] in seq[((n - 1) % 6)] = 
  if n == 150 then 7 else seq[((n - 1) % 6)])) :=
by
  sorry

end digit_150th_in_decimal_of_fraction_l109_109216


namespace find_d_l109_109286

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (a₁ d : ℝ) : Prop :=
∀ n, a n = a₁ + d * (n - 1)

theorem find_d
  (a : ℕ → ℝ)
  (a₁ d : ℝ)
  (h_arith : arithmetic_sequence a a₁ d)
  (h₁ : a₁ = 1)
  (h_geom_mean : a 2 ^ 2 = a 1 * a 4)
  (h_d_neq_zero : d ≠ 0):
  d = 1 :=
sorry

end find_d_l109_109286


namespace count_sixth_powers_below_1000_l109_109304

-- Definition and necessary condition for perfect sixth power
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 6

-- Main statement of the problem
theorem count_sixth_powers_below_1000 :
  {n : ℕ | n < 1000 ∧ is_sixth_power n}.to_finset.card = 3 := 
sorry

end count_sixth_powers_below_1000_l109_109304


namespace max_quarters_l109_109956

/-- Prove that given the conditions for the number of nickels, dimes, and quarters,
    the maximum number of quarters can be 20. --/
theorem max_quarters {a b c : ℕ} (h1 : a + b + c = 120) (h2 : 5 * a + 10 * b + 25 * c = 1000) :
  c ≤ 20 :=
sorry

end max_quarters_l109_109956


namespace sum_and_product_of_roots_l109_109392

theorem sum_and_product_of_roots :
  let a := 1
  let b := -7
  let c := 12
  (∀ x: ℝ, x^2 - 7*x + 12 = 0 → (x = 3 ∨ x = 4)) →
  (-b/a = 7) ∧ (c/a = 12) := 
by
  sorry

end sum_and_product_of_roots_l109_109392


namespace students_in_class_l109_109159

theorem students_in_class (N : ℕ) 
  (avg_age_class : ℕ) (avg_age_4 : ℕ) (avg_age_10 : ℕ) (age_15th : ℕ) 
  (total_age_class : ℕ) (total_age_4 : ℕ) (total_age_10 : ℕ)
  (h1 : avg_age_class = 15)
  (h2 : avg_age_4 = 14)
  (h3 : avg_age_10 = 16)
  (h4 : age_15th = 9)
  (h5 : total_age_class = avg_age_class * N)
  (h6 : total_age_4 = 4 * avg_age_4)
  (h7 : total_age_10 = 10 * avg_age_10)
  (h8 : total_age_class = total_age_4 + total_age_10 + age_15th) :
  N = 15 :=
by
  sorry

end students_in_class_l109_109159


namespace two_numbers_product_l109_109979

theorem two_numbers_product (x y : ℕ) 
  (h1 : x + y = 90) 
  (h2 : x - y = 10) : x * y = 2000 :=
by
  sorry

end two_numbers_product_l109_109979


namespace find_p_l109_109318

theorem find_p 
  (a : ℝ) (p : ℕ) 
  (h1 : 12345 * 6789 = a * 10^p)
  (h2 : 1 ≤ a) (h3 : a < 10) (h4 : 0 < p) 
  : p = 7 := 
sorry

end find_p_l109_109318


namespace P_at_3_l109_109014

noncomputable def P (x : ℝ) : ℝ := 1 * x^5 + 0 * x^4 + 0 * x^3 + 2 * x^2 + 1 * x + 4

theorem P_at_3 : P 3 = 268 := by
  sorry

end P_at_3_l109_109014


namespace pens_solution_exists_l109_109552

-- Definition of the conditions
def pen_cost_eq (x y : ℕ) : Prop :=
  17 * x + 12 * y = 150

-- Proof problem statement that follows from the conditions
theorem pens_solution_exists :
  ∃ x y : ℕ, pen_cost_eq x y :=
by
  existsi (6 : ℕ)
  existsi (4 : ℕ)
  -- Normally the proof would go here, but as stated, we use sorry.
  sorry

end pens_solution_exists_l109_109552


namespace remainder_444_pow_444_mod_13_l109_109704

theorem remainder_444_pow_444_mod_13 :
  (444 ^ 444) % 13 = 1 :=
by
  have h1 : 444 % 13 = 2 := by norm_num
  have h2 : 2 ^ 12 % 13 = 1 := by norm_num
  rw [←pow_mul, ←nat.mul_div_cancel_left 444 12] at h1
  have h3 : 444 = 12 * (444 / 12) := nat.mul_div_cancel_left 444 12
  rw h3
  rw pow_mul
  rw h2
  rw pow_one
  exact h2

end remainder_444_pow_444_mod_13_l109_109704


namespace distance_geologists_probability_l109_109941

theorem distance_geologists_probability :
  let speed := 4 -- km/h
  let n_roads := 6
  let travel_time := 1 -- hour
  let distance_traveled := speed * travel_time -- km
  let distance_threshold := 6 -- km
  let n_outcomes := n_roads * n_roads
  let favorable_outcomes := 18 -- determined from the solution steps
  let probability := favorable_outcomes / n_outcomes
  probability = 0.5 := by
  sorry

end distance_geologists_probability_l109_109941


namespace max_sum_arithmetic_sequence_terms_l109_109826

theorem max_sum_arithmetic_sequence_terms (d : ℝ) (a : ℕ → ℝ) (n : ℕ) 
  (h0 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h1 : d < 0)
  (h2 : a 1 ^ 2 = a 11 ^ 2) : 
  (n = 5) ∨ (n = 6) :=
sorry

end max_sum_arithmetic_sequence_terms_l109_109826


namespace min_value_frac_ineq_l109_109939

theorem min_value_frac_ineq (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) :
  ∃ x, x = (1/a) + (2/b) ∧ x ≥ 9 :=
sorry

end min_value_frac_ineq_l109_109939


namespace slope_condition_l109_109463

theorem slope_condition {m : ℝ} : 
  (4 - m) / (m + 2) = 1 → m = 1 :=
by
  sorry

end slope_condition_l109_109463


namespace least_width_l109_109818

theorem least_width (w : ℝ) (h_nonneg : w ≥ 0) (h_area : w * (w + 10) ≥ 150) : w = 10 :=
sorry

end least_width_l109_109818


namespace M_necessary_for_N_l109_109948

def M (x : ℝ) : Prop := -1 < x ∧ x < 3
def N (x : ℝ) : Prop := 0 < x ∧ x < 3

theorem M_necessary_for_N : (∀ a : ℝ, N a → M a) ∧ (∃ b : ℝ, M b ∧ ¬N b) :=
by sorry

end M_necessary_for_N_l109_109948


namespace solve_for_b_l109_109145

theorem solve_for_b (b : ℝ) (m : ℝ) (h : b > 0)
  (h1 : ∀ x : ℝ, x^2 + b * x + 54 = (x + m) ^ 2 + 18) : b = 12 :=
by
  sorry

end solve_for_b_l109_109145


namespace probability_of_x_greater_than_3y_l109_109662

noncomputable def area_triangle (base height : ℝ) : ℝ := 0.5 * base * height
noncomputable def area_rectangle (width height : ℝ) : ℝ := width * height
noncomputable def probability (area_triangle area_rectangle : ℝ) : ℝ := area_triangle / area_rectangle

theorem probability_of_x_greater_than_3y :
  let base := 2010
  let height_triangle := base / 3
  let height_rectangle := 2011
  let area_triangle := area_triangle base height_triangle
  let area_rectangle := area_rectangle base height_rectangle
  let prob := probability area_triangle area_rectangle
  prob = (335 : ℝ) / 2011 :=
by
  sorry

end probability_of_x_greater_than_3y_l109_109662


namespace cosine_sum_identity_l109_109455

theorem cosine_sum_identity 
  (α : ℝ) 
  (h_sin : Real.sin α = 3 / 5) 
  (h_alpha_first_quad : 0 < α ∧ α < Real.pi / 2) : 
  Real.cos (Real.pi / 3 + α) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end cosine_sum_identity_l109_109455


namespace min_turns_for_route_l109_109492

-- Define the number of parallel and intersecting streets
def num_parallel_streets := 10
def num_intersecting_streets := 10

-- Define the grid as a product of these two numbers
def num_intersections := num_parallel_streets * num_intersecting_streets

-- Define the minimum number of turns necessary for a closed bus route passing through all intersections
def min_turns (grid_size : Nat) : Nat :=
  if grid_size = num_intersections then 20 else 0

-- The main theorem statement
theorem min_turns_for_route : min_turns num_intersections = 20 :=
  sorry

end min_turns_for_route_l109_109492


namespace tony_water_intake_l109_109387

-- Define the constants and conditions
def water_yesterday : ℝ := 48
def percentage_less_yesterday : ℝ := 0.04
def percentage_more_day_before_yesterday : ℝ := 0.05

-- Define the key quantity to find
noncomputable def water_two_days_ago : ℝ := water_yesterday / (1.05 * (1 - percentage_less_yesterday))

-- The proof statement
theorem tony_water_intake :
  water_two_days_ago = 47.62 :=
by
  sorry

end tony_water_intake_l109_109387


namespace total_pics_uploaded_l109_109816

-- Definitions of conditions
def pic_in_first_album : Nat := 14
def albums_with_7_pics : Nat := 3
def pics_per_album : Nat := 7

-- Theorem statement
theorem total_pics_uploaded :
  pic_in_first_album + albums_with_7_pics * pics_per_album = 35 := by
  sorry

end total_pics_uploaded_l109_109816


namespace ratio_of_blue_marbles_l109_109383

theorem ratio_of_blue_marbles {total_marbles red_marbles orange_marbles blue_marbles : ℕ} 
  (h_total : total_marbles = 24)
  (h_red : red_marbles = 6)
  (h_orange : orange_marbles = 6)
  (h_blue : blue_marbles = total_marbles - red_marbles - orange_marbles) : 
  (blue_marbles : ℚ) / (total_marbles : ℚ) = 1 / 2 := 
by
  sorry -- the proof is omitted as per instructions

end ratio_of_blue_marbles_l109_109383


namespace probability_x_gt_3y_l109_109648

-- Definitions of the conditions
def rect := {(x, y) | 0 ≤ x ∧ x ≤ 2010 ∧ 0 ≤ y ∧ y ≤ 2011}

-- Theorem statement to prove the probability
theorem probability_x_gt_3y : 
  (∃ (x y : ℝ), (x, y) ∈ rect ∧ (1 / (2010 * 2011)) * ((1 / 2) * 2010 * 670) = 335 / 2011) :=
sorry

end probability_x_gt_3y_l109_109648


namespace remainder_zero_when_divided_by_condition_l109_109435

noncomputable def remainder_problem (x : ℂ) : ℂ :=
  (2 * x^5 - x^4 + x^2 - 1) * (x^3 - 1)

theorem remainder_zero_when_divided_by_condition (x : ℂ) (h : x^2 - x + 1 = 0) :
  remainder_problem x % (x^2 - x + 1) = 0 := by
  sorry

end remainder_zero_when_divided_by_condition_l109_109435


namespace tabitha_current_age_l109_109891

noncomputable def tabithaAge (currentColors : ℕ) (yearsPassed : ℕ) (startAge : ℕ) (futureYears : ℕ) (futureColors : ℕ) : Prop :=
  (currentColors = (futureColors - futureYears)) ∧
  (yearsPassed = (currentColors - 2)) ∧
  (yearsPassed + startAge = 18)

theorem tabitha_current_age : tabithaAge 5 3 15 3 8 := 
by
  unfold tabithaAge
  split
  all_goals {simp}
  sorry

end tabitha_current_age_l109_109891


namespace rahul_task_days_l109_109354

theorem rahul_task_days (R : ℕ) (h1 : ∀ x : ℤ, x > 0 → 1 / R + 1 / 84 = 1 / 35) : R = 70 := 
by
  -- placeholder for the proof
  sorry

end rahul_task_days_l109_109354


namespace evaluate_expression_l109_109083

theorem evaluate_expression (b : ℤ) (x : ℤ) (h : x = b + 9) : (x - b + 5 = 14) :=
by
  sorry

end evaluate_expression_l109_109083


namespace excircle_problem_l109_109751

-- Define the data structure for a triangle with incenter and excircle properties
structure TriangleWithIncenterAndExcircle (α : Type) [LinearOrderedField α] :=
  (A B C I X : α)
  (is_incenter : Boolean)  -- condition for point I being the incenter
  (is_excircle_center_opposite_A : Boolean)  -- condition for point X being the excircle center opposite A
  (I_A_I : I ≠ A)
  (X_A_X : X ≠ A)

-- Define the problem statement
theorem excircle_problem
  (α : Type) [LinearOrderedField α]
  (T : TriangleWithIncenterAndExcircle α)
  (h_incenter : T.is_incenter)
  (h_excircle_center : T.is_excircle_center_opposite_A)
  (h_not_eq_I : T.I ≠ T.A)
  (h_not_eq_X : T.X ≠ T.A)
  : 
    (T.I * T.X = T.A * T.B) ∧ 
    (T.I * (T.B * T.C) = T.X * (T.B * T.C)) :=
by
  sorry

end excircle_problem_l109_109751


namespace second_polygon_sides_l109_109044

open Nat

theorem second_polygon_sides (s : ℝ) (same_perimeter : ℝ) :
  ∃ n : ℕ, n = 150 :=
by
  let side1 := 3 * s
  have perimeter1 : same_perimeter = 50 * side1 := rfl
  have perimeter2 : same_perimeter = n * s
  have sides : n = 150 :=
  sorry

end second_polygon_sides_l109_109044


namespace correct_operation_l109_109229

theorem correct_operation : 
  (∀ (a b : ℝ), ¬(a^2 * a^3 = a^6) ∧ ¬((a^2)^3 = a^5) ∧ (∀ (a b : ℝ), (a * b)^3 = a^3 * b^3) ∧ ¬(a^8 / a^2 = a^4)) :=
by
  intros a b
  split
  -- proof for ¬(a^2 * a^3 = a^6)
  sorry
  split
  -- proof for ¬((a^2)^3 = a^5)
  sorry
  split
  -- proof for (a * b)^3 = a^3 * b^3
  sorry
  -- proof for ¬(a^8 / a^2 = a^4)
  sorry

end correct_operation_l109_109229


namespace problem_l109_109715

theorem problem (a : ℕ) (h1 : a = 444) : (444 ^ 444) % 13 = 1 :=
by
  have h444 : 444 % 13 = 3 := by sorry
  have h3_pow3 : 3 ^ 3 % 13 = 1 := by sorry
  sorry

end problem_l109_109715


namespace percent_calculation_l109_109403

theorem percent_calculation (Part Whole : ℝ) (hPart : Part = 14) (hWhole : Whole = 70) : 
  (Part / Whole) * 100 = 20 := 
by 
  sorry

end percent_calculation_l109_109403


namespace greatest_drop_june_increase_april_l109_109184

-- January price change
def jan : ℝ := -1.00

-- February price change
def feb : ℝ := 3.50

-- March price change
def mar : ℝ := -3.00

-- April price change
def apr : ℝ := 4.50

-- May price change
def may : ℝ := -1.50

-- June price change
def jun : ℝ := -3.50

def greatest_drop : List (ℝ × String) := [(jan, "January"), (mar, "March"), (may, "May"), (jun, "June")]

def greatest_increase : List (ℝ × String) := [(feb, "February"), (apr, "April")]

theorem greatest_drop_june_increase_april :
  (∀ d ∈ greatest_drop, d.1 ≤ jun) ∧ (∀ i ∈ greatest_increase, i.1 ≤ apr) :=
by
  sorry

end greatest_drop_june_increase_april_l109_109184


namespace solution_set_inequality_l109_109605

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_decreasing_on_non_neg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x
def f_neg_half_eq_zero (f : ℝ → ℝ) : Prop := f (-1/2) = 0

-- Problem statement
theorem solution_set_inequality (f : ℝ → ℝ) 
  (hf_even : is_even_function f) 
  (hf_decreasing : is_decreasing_on_non_neg f) 
  (hf_neg_half_zero : f_neg_half_eq_zero f) : 
  {x : ℝ | f (Real.logb (1/4) x) < 0} = {x | x > 2} ∪ {x | 0 < x ∧ x < 1/2} :=
  sorry

end solution_set_inequality_l109_109605


namespace option_d_is_correct_l109_109724

theorem option_d_is_correct {x y : ℝ} (h : x - 2 = y - 2) : x = y := 
by 
  sorry

end option_d_is_correct_l109_109724


namespace compute_9_times_one_seventh_pow_4_l109_109265

theorem compute_9_times_one_seventh_pow_4 :
  9 * (1 / 7) ^ 4 = 9 / 2401 :=
by
  -- The actual proof would go here.
  sorry

end compute_9_times_one_seventh_pow_4_l109_109265


namespace conditions_for_star_commute_l109_109274

-- Define the operation star
def star (a b : ℝ) : ℝ := a^3 * b^2 - a * b^3

-- Theorem stating the equivalence
theorem conditions_for_star_commute :
  ∀ (x y : ℝ), (star x y = star y x) ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) :=
sorry

end conditions_for_star_commute_l109_109274


namespace find_m_l109_109938

theorem find_m (x : ℝ) (m : ℝ) (h : ∃ x, (x - 2) ≠ 0 ∧ (4 - 2 * x) ≠ 0 ∧ (3 / (x - 2) + 1 = m / (4 - 2 * x))) : m = -6 :=
by
  sorry

end find_m_l109_109938


namespace original_prices_correct_l109_109025

-- Define the problem conditions
def Shirt_A_discount1 := 0.10
def Shirt_A_discount2 := 0.20
def Shirt_A_final_price := 420

def Shirt_B_discount1 := 0.15
def Shirt_B_discount2 := 0.25
def Shirt_B_final_price := 405

def Shirt_C_discount1 := 0.05
def Shirt_C_discount2 := 0.15
def Shirt_C_final_price := 680

def sales_tax := 0.05

-- Define the original prices for each shirt.
def original_price_A := 420 / (0.9 * 0.8)
def original_price_B := 405 / (0.85 * 0.75)
def original_price_C := 680 / (0.95 * 0.85)

-- Prove the original prices of the shirts
theorem original_prices_correct:
  original_price_A = 583.33 ∧ 
  original_price_B = 635 ∧ 
  original_price_C = 842.24 := 
by
  sorry

end original_prices_correct_l109_109025


namespace expand_product_correct_l109_109767

noncomputable def expand_product (x : ℝ) : ℝ :=
  (3 / 7) * (7 / x^2 + 6 * x^3 - 2)

theorem expand_product_correct (x : ℝ) (h : x ≠ 0) :
  expand_product x = (3 / x^2) + (18 * x^3 / 7) - (6 / 7) := by
  unfold expand_product
  -- The proof will go here
  sorry

end expand_product_correct_l109_109767


namespace verify_final_weights_l109_109808

-- Define the initial weights
def initial_bench_press : ℝ := 500
def initial_squat : ℝ := 400
def initial_deadlift : ℝ := 600

-- Define the weight adjustment transformations for each exercise
def transform_bench_press (w : ℝ) : ℝ :=
  let w1 := w * 0.20
  let w2 := w1 * 1.60
  let w3 := w2 * 0.80
  let w4 := w3 * 3
  w4

def transform_squat (w : ℝ) : ℝ :=
  let w1 := w * 0.50
  let w2 := w1 * 1.40
  let w3 := w2 * 2
  w3

def transform_deadlift (w : ℝ) : ℝ :=
  let w1 := w * 0.70
  let w2 := w1 * 1.80
  let w3 := w2 * 0.60
  let w4 := w3 * 1.50
  w4

-- The final calculated weights for verification
def final_bench_press : ℝ := 384
def final_squat : ℝ := 560
def final_deadlift : ℝ := 680.4

-- Statement of the problem: prove that the transformed weights are as calculated
theorem verify_final_weights : 
  transform_bench_press initial_bench_press = final_bench_press ∧ 
  transform_squat initial_squat = final_squat ∧ 
  transform_deadlift initial_deadlift = final_deadlift := 
by 
  sorry

end verify_final_weights_l109_109808


namespace factorization1_factorization2_l109_109770

theorem factorization1 (x y : ℝ) : 4 - 12 * (x - y) + 9 * (x - y)^2 = (2 - 3 * x + 3 * y)^2 :=
by
  sorry

theorem factorization2 (x : ℝ) (a : ℝ) : 2 * a * (x^2 + 1)^2 - 8 * a * x^2 = 2 * a * (x - 1)^2 * (x + 1)^2 :=
by
  sorry

end factorization1_factorization2_l109_109770


namespace tangent_circle_parallel_collinear_l109_109530

theorem tangent_circle_parallel_collinear
  {CD AB : Line}
  {A B C D E F G H : Point}
  (h_tangent_CD : Tangent CD CircleLarger ∧ Tangent CD CircleSmaller)
  (h_parallel : Parallel CD AB)
  (h_points_AC : OnCircle CircleLarger A ∧ OnCircle CircleLarger C)
  (h_tangent_AB : Tangent AB CircleLarger ∧ Tangent AB CircleSmaller)
  (points_of_tangency : OnCircle CircleLarger F ∧ 
                        OnCircle CircleSmaller E ∧ 
                        OnLine CD F ∧ 
                        OnLine CD E ∧ 
                        OnCircle CircleLarger G ∧
                        OnCircle CircleSmaller H ∧
                        OnLine AB G ∧ 
                        OnLine AB H) :
  CFH = CDH :=
begin
  sorry
end

end tangent_circle_parallel_collinear_l109_109530


namespace determine_number_l109_109896

def is_divisible_by_9 (n : ℕ) : Prop :=
  (n.digits 10).sum % 9 = 0

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 5

def ten_power (n p : ℕ) : ℕ :=
  n * 10 ^ p

theorem determine_number (a b : ℕ) (h₁ : b = 0 ∨ b = 5)
  (h₂ : is_divisible_by_9 (7 + 2 + a + 3 + b))
  (h₃ : is_divisible_by_5 (7 * 10000 + 2 * 1000 + a * 100 + 3 * 10 + b)) :
  (7 * 10000 + 2 * 1000 + a * 100 + 3 * 10 + b = 72630 ∨ 
   7 * 10000 + 2 * 1000 + a * 100 + 3 * 10 + b = 72135) :=
by sorry

end determine_number_l109_109896


namespace possible_values_of_cubes_l109_109013

noncomputable def matrix_N (x y z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![x, y, z], ![y, z, x], ![z, x, y]]

def related_conditions (x y z : ℂ) (N : Matrix (Fin 3) (Fin 3) ℂ) : Prop :=
  N^2 = -1 ∧ x * y * z = -1

theorem possible_values_of_cubes (x y z : ℂ) (N : Matrix (Fin 3) (Fin 3) ℂ)
  (hc1 : matrix_N x y z = N) (hc2 : related_conditions x y z N) :
  ∃ w : ℂ, w = x^3 + y^3 + z^3 ∧ (w = -3 + Complex.I ∨ w = -3 - Complex.I) :=
by
  sorry

end possible_values_of_cubes_l109_109013


namespace find_a_plus_b_l109_109489

theorem find_a_plus_b (a b : ℝ) (h₁ : ∀ x, x - b < 0 → x < b) 
  (h₂ : ∀ x, x + a > 0 → x > -a) 
  (h₃ : ∀ x, 2 < x ∧ x < 3 → -a < x ∧ x < b) : 
  a + b = 1 :=
by
  sorry

end find_a_plus_b_l109_109489


namespace perfect_square_partition_l109_109590

open Nat

-- Define the condition of a number being a perfect square
def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

-- Define the main theorem statement
theorem perfect_square_partition (n : ℕ) (h : n ≥ 15) :
  ∀ (A B : Finset ℕ), (A ∪ B = Finset.range (n+1)) → (A ∩ B = ∅) →
  ∃ a b ∈ A, a ≠ b ∧ is_perfect_square (a + b)
:= by
  sorry

end perfect_square_partition_l109_109590


namespace inequality_holds_for_all_x_l109_109086

theorem inequality_holds_for_all_x (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 4) * x - k + 8 > 0) ↔ (-2 < k ∧ k < 6) :=
by
  sorry

end inequality_holds_for_all_x_l109_109086


namespace zoo_individuals_left_l109_109246

/-!
A fifth-grade class went on a field trip to the zoo, and their class of 10 students merged with another class with the same number of students.
5 parents offered to be a chaperone, and 2 teachers from both classes will be there too.
When the school day was over, 10 of the students left. Two of the chaperones, who were parents in that group, also left.
-/

theorem zoo_individuals_left (students_per_class chaperones teachers students_left chaperones_left : ℕ)
  (h1 : students_per_class = 10)
  (h2 : chaperones = 5)
  (h3 : teachers = 2)
  (h4 : students_left = 10)
  (h5 : chaperones_left = 2) : 
  let total_students := students_per_class * 2,
      total_initial := total_students + chaperones + teachers,
      total_remaining := total_initial - students_left - chaperones_left
  in
  total_remaining = 15 := by
  sorry

end zoo_individuals_left_l109_109246


namespace proof_problem_l109_109327

theorem proof_problem (p q : Prop) (hnpq : ¬ (p ∧ q)) (hnp : ¬ p) : ¬ p :=
by
  exact hnp

end proof_problem_l109_109327


namespace bananas_in_each_bunch_l109_109019

theorem bananas_in_each_bunch (x: ℕ) : (6 * x + 5 * 7 = 83) → x = 8 :=
by
  intro h
  sorry

end bananas_in_each_bunch_l109_109019


namespace find_points_and_min_ordinate_l109_109027

noncomputable def pi : Real := Real.pi
noncomputable def sin : Real → Real := Real.sin
noncomputable def cos : Real → Real := Real.cos

def within_square (x y : Real) : Prop :=
  -pi ≤ x ∧ x ≤ pi ∧ 0 ≤ y ∧ y ≤ 2 * pi

def satisfies_system (x y : Real) : Prop :=
  sin x + sin y = sin 2 ∧ cos x + cos y = cos 2

theorem find_points_and_min_ordinate :
  ∃ (points : List (Real × Real)), 
    (∀ (p : Real × Real), p ∈ points → within_square p.1 p.2 ∧ satisfies_system p.1 p.2) ∧
    points.length = 2 ∧
    ∃ (min_point : Real × Real), min_point ∈ points ∧ ∀ (p : Real × Real), p ∈ points → min_point.2 ≤ p.2 ∧ min_point = (2 + Real.pi / 3, 2 - Real.pi / 3) :=
by
  sorry

end find_points_and_min_ordinate_l109_109027


namespace sum_of_squares_eq_two_l109_109617

theorem sum_of_squares_eq_two {a b : ℝ} (h : (a^2 + b^2) * (a^2 + b^2 + 4) = 12) : a^2 + b^2 = 2 := sorry

end sum_of_squares_eq_two_l109_109617


namespace ratio_boys_to_girls_l109_109329

-- Define the given conditions
def G : ℕ := 300
def T : ℕ := 780

-- State the proposition to be proven
theorem ratio_boys_to_girls (B : ℕ) (h : B + G = T) : B / G = 8 / 5 :=
by
  -- Proof placeholder
  sorry

end ratio_boys_to_girls_l109_109329


namespace quadratic_inequality_solution_l109_109442

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, x^2 - 8 * x + c > 0) ↔ (0 < c ∧ c < 16) := 
sorry

end quadratic_inequality_solution_l109_109442


namespace ratio_H_over_G_l109_109680

theorem ratio_H_over_G (G H : ℤ)
  (h : ∀ x : ℝ, x ≠ -5 → x ≠ 0 → x ≠ 4 →
    (G : ℝ)/(x + 5) + (H : ℝ)/(x^2 - 4*x) = (x^2 - 2*x + 10) / (x^3 + x^2 - 20*x)) :
  H / G = 2 :=
  sorry

end ratio_H_over_G_l109_109680


namespace count_unbroken_matches_l109_109502

theorem count_unbroken_matches :
  let n_1 := 5 * 12  -- number of boxes in the first set
  let matches_1 := n_1 * 20  -- total matches in first set of boxes
  let broken_1 := n_1 * 3  -- total broken matches in first set of boxes
  let unbroken_1 := matches_1 - broken_1  -- unbroken matches in first set of boxes

  let n_2 := 4  -- number of extra boxes
  let matches_2 := n_2 * 25  -- total matches in extra boxes
  let broken_2 := (matches_2 / 5)  -- total broken matches in extra boxes (20%)
  let unbroken_2 := matches_2 - broken_2  -- unbroken matches in extra boxes

  let total_unbroken := unbroken_1 + unbroken_2  -- total unbroken matches

  total_unbroken = 1100 := 
by
  sorry

end count_unbroken_matches_l109_109502


namespace greatest_prime_divisor_digits_sum_l109_109831

theorem greatest_prime_divisor_digits_sum (h : 8191 = 2^13 - 1) : (1 + 2 + 7) = 10 :=
by
  sorry

end greatest_prime_divisor_digits_sum_l109_109831


namespace smallest_integer_sum_to_2020_l109_109773

theorem smallest_integer_sum_to_2020 :
  ∃ B : ℤ, (∃ (n : ℤ), (B * (B + 1) / 2) + ((n * (n + 1)) / 2) = 2020) ∧ (∀ C : ℤ, (∃ (m : ℤ), (C * (C + 1) / 2) + ((m * (m + 1)) / 2) = 2020) → B ≤ C) ∧ B = -2019 :=
by
  sorry

end smallest_integer_sum_to_2020_l109_109773


namespace prob_not_snowing_l109_109171

theorem prob_not_snowing (P_snowing : ℚ) (h : P_snowing = 1/4) : 1 - P_snowing = 3/4 := by
  sorry

end prob_not_snowing_l109_109171


namespace tyre_flattening_time_l109_109416

theorem tyre_flattening_time (R1 R2 : ℝ) (hR1 : R1 = 1 / 9) (hR2 : R2 = 1 / 6) : 
  1 / (R1 + R2) = 3.6 :=
by 
  sorry

end tyre_flattening_time_l109_109416


namespace not_right_triangle_l109_109944

theorem not_right_triangle (A B C : ℝ) (hA : A + B = 180 - C) 
  (hB : A = B / 2 ∧ A = C / 3) 
  (hC : A = B / 2 ∧ B = C / 1.5) 
  (hD : A = 2 * B ∧ A = 3 * C):
  (C ≠ 90) :=
by {
  sorry
}

end not_right_triangle_l109_109944


namespace no_solution_1221_l109_109154

def equation_correctness (n : ℤ) : Prop :=
  -n^3 + 555^3 = n^2 - n * 555 + 555^2

-- Prove that the prescribed value 1221 does not satisfy the modified equation by contradiction
theorem no_solution_1221 : ¬ ∃ n : ℤ, equation_correctness n ∧ n = 1221 := by
  sorry

end no_solution_1221_l109_109154


namespace remainder_of_polynomial_division_l109_109847

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 5 * x^3 - 18 * x^2 + 31 * x - 40

-- Define the divisor D(x)
def D (x : ℝ) : ℝ := 5 * x - 10

-- Prove that the remainder when P(x) is divided by D(x) is -10
theorem remainder_of_polynomial_division : (P 2) = -10 := by
  sorry

end remainder_of_polynomial_division_l109_109847


namespace number_of_sixth_powers_less_than_1000_l109_109314

theorem number_of_sixth_powers_less_than_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ (k : ℕ), n = k^6}.finite.card = 3 :=
by sorry

end number_of_sixth_powers_less_than_1000_l109_109314


namespace part1_part2_l109_109465

def f (x m : ℝ) : ℝ := |x - 1| - |2 * x + m|

theorem part1 (x : ℝ) (m : ℝ) (h : m = -4) : 
    f x m < 0 ↔ x < 5 / 3 ∨ x > 3 := 
by 
  sorry

theorem part2 (x : ℝ) (h : 1 < x) (h' : ∀ x, 1 < x → f x m < 0) : 
    m ≥ -2 :=
by 
  sorry

end part1_part2_l109_109465


namespace num_integers_satisfying_abs_leq_bound_l109_109932

theorem num_integers_satisfying_abs_leq_bound : ∃ n : ℕ, n = 19 ∧ ∀ x : ℤ, |x| ≤ 3 * Real.sqrt 10 → (x ≥ -9 ∧ x ≤ 9) := by
  sorry

end num_integers_satisfying_abs_leq_bound_l109_109932


namespace candle_height_problem_l109_109189

-- Define the conditions given in the problem
def same_initial_height (height : ℝ := 1) := height = 1

def burn_rate_first_candle := 1 / 5

def burn_rate_second_candle := 1 / 4

def height_first_candle (t : ℝ) := 1 - (burn_rate_first_candle * t)

def height_second_candle (t : ℝ) := 1 - (burn_rate_second_candle * t)

-- Define the proof problem
theorem candle_height_problem : ∃ t : ℝ, height_first_candle t = 3 * height_second_candle t ∧ t = 40 / 11 :=
by
  sorry

end candle_height_problem_l109_109189


namespace first_percentage_reduction_l109_109683

theorem first_percentage_reduction (P : ℝ) (x : ℝ) :
  (P - (x / 100) * P) * 0.4 = P * 0.3 → x = 25 := by
  sorry

end first_percentage_reduction_l109_109683


namespace number_of_sections_l109_109423

noncomputable def initial_rope : ℕ := 50
noncomputable def rope_for_art := initial_rope / 5
noncomputable def remaining_rope_after_art := initial_rope - rope_for_art
noncomputable def rope_given_to_friend := remaining_rope_after_art / 2
noncomputable def remaining_rope := remaining_rope_after_art - rope_given_to_friend
noncomputable def section_size : ℕ := 2
noncomputable def sections := remaining_rope / section_size

theorem number_of_sections : sections = 10 :=
by
  sorry

end number_of_sections_l109_109423


namespace ski_price_l109_109030

variable {x y : ℕ}

theorem ski_price (h1 : 2 * x + y = 340) (h2 : 3 * x + 2 * y = 570) : x = 110 ∧ y = 120 := by
  sorry

end ski_price_l109_109030


namespace find_tabitha_age_l109_109890

-- Define the conditions
variable (age_started : ℕ) (colors_started : ℕ) (years_future : ℕ) (future_colors : ℕ)

-- Let's specify the given problem's conditions:
axiom h1 : age_started = 15          -- Tabitha started at age 15
axiom h2 : colors_started = 2        -- with 2 colors
axiom h3 : years_future = 3          -- in three years
axiom h4 : future_colors = 8         -- she will have 8 different colors

-- The proof problem we need to state:
theorem find_tabitha_age : ∃ age_now : ℕ, age_now = age_started + (future_colors - colors_started) - years_future := by
  sorry

end find_tabitha_age_l109_109890


namespace billy_age_is_45_l109_109881

variable (Billy_age Joe_age : ℕ)

-- Given conditions
def condition1 := Billy_age = 3 * Joe_age
def condition2 := Billy_age + Joe_age = 60
def condition3 := Billy_age > 60 / 2

-- Prove Billy's age is 45
theorem billy_age_is_45 (h1 : condition1 Billy_age Joe_age) (h2 : condition2 Billy_age Joe_age) (h3 : condition3 Billy_age) : Billy_age = 45 :=
by
  sorry

end billy_age_is_45_l109_109881


namespace volume_of_cube_l109_109980

-- Define the conditions
def surface_area (a : ℝ) : ℝ := 6 * a^2
def side_length (a : ℝ) (SA : ℝ) : Prop := SA = 6 * a^2
def volume (a : ℝ) : ℝ := a^3

-- State the theorem
theorem volume_of_cube (a : ℝ) (SA : surface_area a = 150) : volume a = 125 := 
sorry

end volume_of_cube_l109_109980


namespace parabola_vertex_coordinates_l109_109160

theorem parabola_vertex_coordinates :
  ∀ (x y : ℝ), y = -3 * (x + 1)^2 - 2 → (x, y) = (-1, -2) := by
  sorry

end parabola_vertex_coordinates_l109_109160


namespace difference_in_girls_and_boys_l109_109632

-- Given conditions as definitions
def boys : ℕ := 40
def ratio_boys_to_girls (b g : ℕ) : Prop := 5 * g = 13 * b

-- Statement of the problem
theorem difference_in_girls_and_boys (g : ℕ) (h : ratio_boys_to_girls boys g) : g - boys = 64 :=
by
  sorry

end difference_in_girls_and_boys_l109_109632


namespace compute_fraction_power_l109_109266

theorem compute_fraction_power : 9 * (1 / 7)^4 = 9 / 2401 := by
  sorry

end compute_fraction_power_l109_109266


namespace fixed_point_of_line_minimized_triangle_area_l109_109606

theorem fixed_point_of_line (a : ℝ) : ∃ x y : ℝ, (a + 1) * x + y - 5 - 2 * a = 0 ∧ x = 2 ∧ y = 3 :=
by
  sorry

theorem minimized_triangle_area : ∃ (a b : ℝ) (h1 : a > 0) (h2 : b > 0), (2 / a + 3 / b = 1) ∧ (a * b = 24) ∧ (3 * 4 / a + 2 * 6 / b - 12 = 0) :=
by
  sorry

end fixed_point_of_line_minimized_triangle_area_l109_109606


namespace probability_of_x_gt_3y_in_rectangle_is_335_over_2011_l109_109663

noncomputable def probability_x_gt_3y : ℚ :=
  let rectangle := ((0 : ℚ, 0 : ℚ), (2010 : ℚ), (2010 : ℚ, 2011 : ℚ), (0 : ℚ, 2011 : ℚ)) in
  let area_rectangle := 2010 * 2011 in
  let area_triangle := (1/2) * 2010 * 670 in
  area_triangle / area_rectangle

theorem probability_of_x_gt_3y_in_rectangle_is_335_over_2011 : probability_x_gt_3y = 335 / 2011 :=
begin
  sorry
end

end probability_of_x_gt_3y_in_rectangle_is_335_over_2011_l109_109663


namespace simplify_and_evaluate_expression_l109_109821

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = Real.sqrt 2 - 3) : 
  (a - 3) / (a^2 + 6 * a + 9) / (1 - 6 / (a + 3)) = Real.sqrt 2 / 2 :=
by sorry

end simplify_and_evaluate_expression_l109_109821


namespace power_mod_eq_one_l109_109696

theorem power_mod_eq_one (n : ℕ) (h₁ : 444 ≡ 3 [MOD 13]) (h₂ : 3^12 ≡ 1 [MOD 13]) :
  444^444 ≡ 1 [MOD 13] :=
by
  sorry

end power_mod_eq_one_l109_109696


namespace average_price_of_towels_l109_109874

-- Definitions based on conditions
def cost_towel1 : ℕ := 3 * 100
def cost_towel2 : ℕ := 5 * 150
def cost_towel3 : ℕ := 2 * 600
def total_cost : ℕ := cost_towel1 + cost_towel2 + cost_towel3
def total_towels : ℕ := 3 + 5 + 2
def average_price : ℕ := total_cost / total_towels

-- Statement to be proved
theorem average_price_of_towels :
  average_price = 225 :=
by
  sorry

end average_price_of_towels_l109_109874


namespace max_integer_a_l109_109098

theorem max_integer_a :
  ∀ (a: ℤ), (∀ x: ℝ, (a + 1) * x^2 - 2 * x + 3 = 0 → (a = -2 → (-12 * a - 8) ≥ 0)) → (∀ a ≤ -2, a ≠ -1) :=
by
  sorry

end max_integer_a_l109_109098


namespace max_abc_value_l109_109949

theorem max_abc_value (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_equation : a * b + c = (a + c) * (b + c))
  (h_sum : a + b + c = 2) : abc ≤ 1/27 :=
by sorry

end max_abc_value_l109_109949


namespace inlet_pipe_rate_l109_109248

-- Conditions definitions
def tank_capacity : ℕ := 4320
def leak_empty_time : ℕ := 6
def full_empty_time_with_inlet : ℕ := 8

-- Question translated into a theorem
theorem inlet_pipe_rate : 
  (tank_capacity / leak_empty_time) = 720 →
  (tank_capacity / full_empty_time_with_inlet) = 540 →
  ∀ R : ℕ, 
    R - 720 = 540 →
    (R / 60) = 21 :=
by
  intros h_leak h_net R h_R
  sorry

end inlet_pipe_rate_l109_109248


namespace total_bread_amt_l109_109729

-- Define the conditions
variables (bread_dinner bread_lunch bread_breakfast total_bread : ℕ)
axiom bread_dinner_amt : bread_dinner = 240
axiom dinner_lunch_ratio : bread_dinner = 8 * bread_lunch
axiom dinner_breakfast_ratio : bread_dinner = 6 * bread_breakfast

-- The proof statement
theorem total_bread_amt : total_bread = bread_dinner + bread_lunch + bread_breakfast → total_bread = 310 :=
by
  -- Use the axioms and the given conditions to derive the statement
  sorry

end total_bread_amt_l109_109729


namespace germination_percentage_l109_109280

theorem germination_percentage :
  ∀ (seeds_plot1 seeds_plot2 germination_rate1 germination_rate2 : ℝ),
    seeds_plot1 = 300 →
    seeds_plot2 = 200 →
    germination_rate1 = 0.30 →
    germination_rate2 = 0.35 →
    ((germination_rate1 * seeds_plot1 + germination_rate2 * seeds_plot2) / (seeds_plot1 + seeds_plot2)) * 100 = 32 :=
by
  intros seeds_plot1 seeds_plot2 germination_rate1 germination_rate2 h1 h2 h3 h4
  sorry

end germination_percentage_l109_109280


namespace pow_calculation_l109_109425

-- We assume a is a non-zero real number or just a variable
variable (a : ℝ)

theorem pow_calculation : (2 * a^2)^3 = 8 * a^6 := 
by
  sorry

end pow_calculation_l109_109425


namespace bc_possible_values_l109_109016

theorem bc_possible_values (a b c : ℝ) 
  (h1 : a + b + c = 100) 
  (h2 : ab + bc + ca = 20) 
  (h3 : (a + b) * (a + c) = 24) : 
  bc = -176 ∨ bc = 224 :=
by
  sorry

end bc_possible_values_l109_109016


namespace pow_15_1234_mod_19_l109_109722

theorem pow_15_1234_mod_19 : (15^1234) % 19 = 6 := 
by sorry

end pow_15_1234_mod_19_l109_109722


namespace ratio_of_boys_to_girls_l109_109022

open Nat

theorem ratio_of_boys_to_girls
    (B G : ℕ) 
    (boys_avg : ℕ) 
    (girls_avg : ℕ) 
    (class_avg : ℕ)
    (h1 : boys_avg = 90)
    (h2 : girls_avg = 96)
    (h3 : class_avg = 94)
    (h4 : 94 * (B + G) = 90 * B + 96 * G) :
    2 * B = G :=
by
  sorry

end ratio_of_boys_to_girls_l109_109022


namespace three_friends_visit_exactly_27_days_l109_109475

theorem three_friends_visit_exactly_27_days
  (A B C D : ℕ) (hA : A = 6) (hB : B = 8) (hC : C = 10) (hD : D = 12) :
  let L := Nat.lcm (Nat.lcm A B) (Nat.lcm C D) in
  360 / L * (1 + 360 / (24 * 6)) = 27 := sorry

end three_friends_visit_exactly_27_days_l109_109475


namespace outdoor_section_width_l109_109564

theorem outdoor_section_width (Length Area Width : ℝ) (h1 : Length = 6) (h2 : Area = 24) : Width = 4 :=
by
  -- We'll use "?" to represent the parts that need to be inferred by the proof assistant. 
  sorry

end outdoor_section_width_l109_109564


namespace correct_negation_statement_l109_109433

def Person : Type := sorry

def is_adult (p : Person) : Prop := sorry
def is_teenager (p : Person) : Prop := sorry
def is_responsible (p : Person) : Prop := sorry
def is_irresponsible (p : Person) : Prop := sorry

axiom all_adults_responsible : ∀ p, is_adult p → is_responsible p
axiom some_adults_responsible : ∃ p, is_adult p ∧ is_responsible p
axiom no_teenagers_responsible : ∀ p, is_teenager p → ¬is_responsible p
axiom all_teenagers_irresponsible : ∀ p, is_teenager p → is_irresponsible p
axiom exists_irresponsible_teenager : ∃ p, is_teenager p ∧ is_irresponsible p
axiom all_teenagers_responsible : ∀ p, is_teenager p → is_responsible p

theorem correct_negation_statement
: (∃ p, is_teenager p ∧ ¬is_responsible p) ↔ 
  (∃ p, is_teenager p ∧ is_irresponsible p) :=
sorry

end correct_negation_statement_l109_109433


namespace sequence_convergence_l109_109642

noncomputable def alpha : ℝ := sorry
def bounded (a : ℕ → ℝ) : Prop := ∃ M > 0, ∀ n, ‖a n‖ ≤ M

-- Translation of the math problem
theorem sequence_convergence (a : ℕ → ℝ) (ha : bounded a) (hα : 0 < alpha ∧ alpha ≤ 1) 
  (ineq : ∀ n ≥ 2, a (n+1) ≤ alpha * a n + (1 - alpha) * a (n-1)) : 
  ∃ l, ∀ ε > 0, ∃ N, ∀ n ≥ N, ‖a n - l‖ < ε := 
sorry

end sequence_convergence_l109_109642


namespace sum_of_bases_l109_109333

theorem sum_of_bases (R1 R2 : ℕ)
  (h1 : ∀ F1 : ℚ, F1 = (4 * R1 + 8) / (R1 ^ 2 - 1) → F1 = (5 * R2 + 9) / (R2 ^ 2 - 1))
  (h2 : ∀ F2 : ℚ, F2 = (8 * R1 + 4) / (R1 ^ 2 - 1) → F2 = (9 * R2 + 5) / (R2 ^ 2 - 1)) :
  R1 + R2 = 24 :=
sorry

end sum_of_bases_l109_109333


namespace find_quadruples_l109_109893

open Nat

theorem find_quadruples (a b p n : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n)
    (h : a^3 + b^3 = p^n) :
    (∃ k, a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3 * k + 1) ∨
    (∃ k, a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3 * k + 2) ∨
    (∃ k, a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3 * k + 2) :=
sorry

end find_quadruples_l109_109893


namespace digit_150_is_7_l109_109195

theorem digit_150_is_7 : (0.242857242857242857 : ℚ) = (17/70 : ℚ) ∧ ( (17/70).decimal 150 = 7) := by
  sorry

end digit_150_is_7_l109_109195


namespace starting_elevation_l109_109505

variable (rate time final_elevation : ℝ)
variable (h_rate : rate = 10)
variable (h_time : time = 5)
variable (h_final_elevation : final_elevation = 350)

theorem starting_elevation (start_elevation : ℝ) :
  start_elevation = 400 :=
  by
    sorry

end starting_elevation_l109_109505


namespace find_c_plus_inv_b_l109_109156

variable (a b c : ℝ)

def conditions := 
  (a * b * c = 1) ∧ 
  (a + 1/c = 7) ∧ 
  (b + 1/a = 16)

theorem find_c_plus_inv_b (h : conditions a b c) : 
  c + 1/b = 25 / 111 :=
sorry

end find_c_plus_inv_b_l109_109156


namespace expression_value_l109_109759

open Real

theorem expression_value :
  3 + sqrt 3 + 1 / (3 + sqrt 3) + 1 / (sqrt 3 - 3) = 3 + 2 * sqrt 3 / 3 := 
sorry

end expression_value_l109_109759


namespace barbara_initial_candies_l109_109880

noncomputable def initialCandies (used left: ℝ) := used + left

theorem barbara_initial_candies (used left: ℝ) (h_used: used = 9.0) (h_left: left = 9) : initialCandies used left = 18 := 
by
  rw [h_used, h_left]
  norm_num
  sorry

end barbara_initial_candies_l109_109880


namespace coprime_count_l109_109614

theorem coprime_count (n : ℕ) (h : n = 56700000) : 
  ∃ m, m = 12960000 ∧ ∀ i < n, Nat.gcd i n = 1 → i < m :=
by
  sorry

end coprime_count_l109_109614


namespace part1_solution_set_l109_109097

theorem part1_solution_set (a : ℝ) (x : ℝ) : a = -2 → (a - 2) * x ^ 2 + 2 * (a - 2) * x - 4 < 0 ↔ x ≠ -1 :=
by sorry

end part1_solution_set_l109_109097


namespace work_completion_l109_109875

theorem work_completion (A B : ℝ → ℝ) (h1 : ∀ t, A t = B t) (h3 : A 4 + B 4 = 1) : B 1 = 1/2 :=
by {
  sorry
}

end work_completion_l109_109875


namespace triangle_inequality_l109_109074

theorem triangle_inequality (a b c : ℕ) : 
    a + b > c ∧ a + c > b ∧ b + c > a ↔ 
    (a, b, c) = (2, 3, 4) ∨ (a, b, c) = (3, 4, 7) ∨ (a, b, c) = (4, 6, 2) ∨ (a, b, c) = (7, 10, 2)
    → (a + b > c ∧ a + c > b ∧ b + c > a ↔ (a, b, c) = (2, 3, 4)) ∧
      (a + b = c ∨ a + c = b ∨ b + c = a         ↔ (a, b, c) = (3, 4, 7)) ∧
      (a + b = c ∨ a + c = b ∨ b + c = a        ↔ (a, b, c) = (4, 6, 2)) ∧
      (a + b < c ∨ a + c < b ∨ b + c < a        ↔ (a, b, c) = (7, 10, 2)) :=
sorry

end triangle_inequality_l109_109074


namespace heart_then_club_probability_l109_109190

theorem heart_then_club_probability :
  (13 / 52) * (13 / 51) = 13 / 204 := by
  sorry

end heart_then_club_probability_l109_109190


namespace power_mod_444_444_l109_109698

open Nat

theorem power_mod_444_444 : (444:ℕ)^444 % 13 = 1 := by
  sorry

end power_mod_444_444_l109_109698


namespace find_expression_l109_109859

theorem find_expression (x y : ℝ) : 2 * x * (-3 * x^2 * y) = -6 * x^3 * y := by
  sorry

end find_expression_l109_109859


namespace cube_surface_area_l109_109398

theorem cube_surface_area (V : ℝ) (s : ℝ) (A : ℝ) :
  V = 729 ∧ V = s^3 ∧ A = 6 * s^2 → A = 486 := by
  sorry

end cube_surface_area_l109_109398


namespace sum_valid_fractions_eq_400_l109_109529

open Nat

def isCoprimeWith30 (n r : ℕ) : Prop := Nat.gcd (30 * n + r) 30 = 1

def validNumerators : List ℕ := [1, 7, 11, 13, 17, 19, 23, 29]

-- We define the fractions we are interested in.
def validFractions : List ℚ :=
List.filter (λ f : ℚ, f < 10)
  ((List.range 10).bind (λ n, (validNumerators.map (λ r, (30 * n + r) / 30))))

-- The main proof statement
theorem sum_valid_fractions_eq_400 : (validFractions.sum = 400) :=
sorry

end sum_valid_fractions_eq_400_l109_109529


namespace determine_a_l109_109609

-- Define the function f(x)
def f (x a : ℝ) : ℝ := x^2 - a * x + 3

-- Define the condition that f(x) >= a for all x in the interval [-1, +∞)
def condition (a : ℝ) : Prop := ∀ x : ℝ, x ≥ -1 → f x a ≥ a

-- The theorem to prove:
theorem determine_a : ∀ a : ℝ, condition a ↔ a ≤ 2 :=
by
  sorry

end determine_a_l109_109609


namespace both_square_and_cube_count_both_square_cube_less_than_1000_l109_109307

theorem both_square_and_cube (n : ℕ) (h_positive : n > 0) (h_bound : n < 1000) (h_square : ∃ k : ℕ, n = k^2) (h_cube : ∃ k : ℕ, n = k^3) :
  n ∈ {1, 64, 729} :=
sorry

theorem count_both_square_cube_less_than_1000 :
  {n : ℕ | n > 0 ∧ n < 1000 ∧ ∃ k : ℕ, n = k^2 ∧ ∃ k : ℕ, n = k^3}.to_finset.card = 3 :=
by
  have h_valid : ∀ n ∈ {1, 64, 729}, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) := sorry
  have h_reverse : ∀ n, n > 0 ∧ n < 1000 ∧ (∃ k : ℕ, n = k^2) ∧ (∃ k : ℕ, n = k^3) → n ∈ {1, 64, 729} := sorry
  exact Finset.card_map _ {1, 64, 729}.to_finset
  sorry

end both_square_and_cube_count_both_square_cube_less_than_1000_l109_109307


namespace second_polygon_num_sides_l109_109047

theorem second_polygon_num_sides 
  (s : ℝ) 
  (h1: s ≠ 0) 
  (perimeter_equal: ∀ (side1_len side2_len : ℝ),
    50 * (3 * side1_len) = side2_len * (50 * 3 * side1_len / side1_len)) : 
  ∃ n : ℕ, n = 150 :=
by {
  use 150,
  sorry
}

end second_polygon_num_sides_l109_109047


namespace inequality_l109_109811

theorem inequality (a b : ℝ) : a^2 + b^2 + 7/4 ≥ a * b + 2 * a + b / 2 :=
sorry

end inequality_l109_109811


namespace average_class_size_l109_109378

theorem average_class_size 
  (num_3_year_olds : ℕ) 
  (num_4_year_olds : ℕ) 
  (num_5_year_olds : ℕ) 
  (num_6_year_olds : ℕ) 
  (class_size_3_and_4 : num_3_year_olds = 13 ∧ num_4_year_olds = 20) 
  (class_size_5_and_6 : num_5_year_olds = 15 ∧ num_6_year_olds = 22) :
  (num_3_year_olds + num_4_year_olds + num_5_year_olds + num_6_year_olds) / 2 = 35 :=
by
  sorry

end average_class_size_l109_109378


namespace digit_150th_of_17_div_70_is_7_l109_109205

noncomputable def decimal_representation (n d : ℚ) : ℚ := n / d

-- We define the repeating part of the decimal representation.
def repeating_part : ℕ := 242857

-- Define the function to get the nth digit of the repeating part.
def get_nth_digit_of_repeating (n : ℕ) : ℕ :=
  let sequence := [2, 4, 2, 8, 5, 7]
  sequence.get! (n % sequence.length)

theorem digit_150th_of_17_div_70_is_7 : get_nth_digit_of_repeating 149 = 7 :=
by
  sorry

end digit_150th_of_17_div_70_is_7_l109_109205


namespace original_price_correct_l109_109598

def sale_price_per_tire : ℝ := 75
def total_savings : ℝ := 36
def number_of_tires : ℕ := 4
def saving_per_tire : ℝ := total_savings / number_of_tires
def original_price_per_tire : ℝ := sale_price_per_tire + saving_per_tire

theorem original_price_correct :
  original_price_per_tire = 84 :=
by
  sorry

end original_price_correct_l109_109598


namespace triangle_area_l109_109158

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

def has_perimeter (a b c p : ℝ) : Prop :=
  a + b + c = p

def has_altitude (base side altitude : ℝ) : Prop :=
  (base / 2) ^ 2 + altitude ^ 2 = side ^ 2

def area_of_triangle (a base altitude : ℝ) : ℝ :=
  0.5 * base * altitude

theorem triangle_area (a b c : ℝ)
  (h_iso : is_isosceles a b c)
  (h_p : has_perimeter a b c 40)
  (h_alt : has_altitude (2 * a) b 12) :
  area_of_triangle a (2 * a) 12 = 76.8 :=
by
  sorry

end triangle_area_l109_109158


namespace probability_x_gt_3y_l109_109649

-- Definitions of the conditions
def rect := {(x, y) | 0 ≤ x ∧ x ≤ 2010 ∧ 0 ≤ y ∧ y ≤ 2011}

-- Theorem statement to prove the probability
theorem probability_x_gt_3y : 
  (∃ (x y : ℝ), (x, y) ∈ rect ∧ (1 / (2010 * 2011)) * ((1 / 2) * 2010 * 670) = 335 / 2011) :=
sorry

end probability_x_gt_3y_l109_109649


namespace inequality_bounds_l109_109445

theorem inequality_bounds (a b c d e : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  1 < (a/(a+b) + b/(b+c) + c/(c+d) + d/(d+e) + e/(e+a)) ∧
  (a/(a+b) + b/(b+c) + c/(c+d) + d/(d+e) + e/(e+a)) < 4 :=
sorry

end inequality_bounds_l109_109445


namespace bob_rope_sections_l109_109422

/-- Given a 50-foot rope, where 1/5 is used for art, half of the remaining is given to a friend,
     and the rest is cut into 2-foot sections, prove that the number of sections Bob gets is 10. -/
theorem bob_rope_sections :
  ∀ (total_rope art_fraction remaining_fraction section_length : ℕ),
    total_rope = 50 →
    art_fraction = 5 →
    remaining_fraction = 2 →
    section_length = 2 →
    (total_rope / art_fraction / remaining_fraction / section_length) = 10 :=
by
  intros total_rope art_fraction remaining_fraction section_length
  assume h_total_rope h_art_fraction h_remaining_fraction h_section_length
  rw [h_total_rope, h_art_fraction, h_remaining_fraction, h_section_length]
  have h1 : 50 / 5 = 10 := by norm_num
  have h2 : (50 - 10) / 2 = 20 := by norm_num
  have h3 : 20 / 2 = 10 := by norm_num
  exact h3

end bob_rope_sections_l109_109422


namespace triangle_right_hypotenuse_l109_109927

theorem triangle_right_hypotenuse (c : ℝ) (a : ℝ) (h₀ : c = 4) (h₁ : 0 < a) (h₂ : a^2 + b^2 = c^2) :
  a ≤ 2 * Real.sqrt 2 :=
sorry

end triangle_right_hypotenuse_l109_109927


namespace smallest_q_exists_l109_109143

noncomputable def p_q_r_are_consecutive_terms (p q r : ℝ) : Prop :=
∃ d : ℝ, p = q - d ∧ r = q + d

theorem smallest_q_exists
  (p q r : ℝ)
  (h1 : p_q_r_are_consecutive_terms p q r)
  (h2 : p > 0) 
  (h3 : q > 0) 
  (h4 : r > 0)
  (h5 : p * q * r = 216) :
  q = 6 :=
sorry

end smallest_q_exists_l109_109143


namespace smallest_n_for_terminating_decimal_l109_109226

theorem smallest_n_for_terminating_decimal :
  ∃ (n : ℕ), 0 < n ∧ ∀ m : ℕ, (0 < m ∧ m < n+53) → (∃ a b : ℕ, n + 53 = 2^a * 5^b) → n = 11 :=
by
  sorry

end smallest_n_for_terminating_decimal_l109_109226


namespace hens_count_l109_109743

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 140) : H = 26 := by
  sorry

end hens_count_l109_109743


namespace extraMaterialNeeded_l109_109146

-- Box dimensions
def smallBoxLength (a : ℝ) : ℝ := a
def smallBoxWidth (b : ℝ) : ℝ := 1.5 * b
def smallBoxHeight (c : ℝ) : ℝ := c

def largeBoxLength (a : ℝ) : ℝ := 1.5 * a
def largeBoxWidth (b : ℝ) : ℝ := 2 * b
def largeBoxHeight (c : ℝ) : ℝ := 2 * c

-- Volume calculations
def volumeSmallBox (a b c : ℝ) : ℝ := a * (1.5 * b) * c
def volumeLargeBox (a b c : ℝ) : ℝ := (1.5 * a) * (2 * b) * (2 * c)

-- Surface area calculations
def surfaceAreaSmallBox (a b c : ℝ) : ℝ := 2 * (a * (1.5 * b)) + 2 * (a * c) + 2 * ((1.5 * b) * c)
def surfaceAreaLargeBox (a b c : ℝ) : ℝ := 2 * ((1.5 * a) * (2 * b)) + 2 * ((1.5 * a) * (2 * c)) + 2 * ((2 * b) * (2 * c))

-- Proof statement
theorem extraMaterialNeeded (a b c : ℝ) :
  (volumeSmallBox a b c = 1.5 * a * b * c) ∧ (volumeLargeBox a b c = 6 * a * b * c) ∧ 
  (surfaceAreaLargeBox a b c - surfaceAreaSmallBox a b c = 3 * a * b + 4 * a * c + 5 * b * c) :=
by
  sorry

end extraMaterialNeeded_l109_109146


namespace cookies_per_student_l109_109774

theorem cookies_per_student (students : ℕ) (percent : ℝ) (oatmeal_cookies : ℕ) 
                            (h_students : students = 40)
                            (h_percent : percent = 10 / 100)
                            (h_oatmeal : oatmeal_cookies = 8) :
                            (oatmeal_cookies / percent / students) = 2 := by
  sorry

end cookies_per_student_l109_109774


namespace sum_sequence_value_l109_109777

open Nat

def sequence (a : ℕ → ℕ) := 
  a 1 = 1 ∧ 
  a 2 = 3 ∧ 
  ∀ n : ℕ, n > 0 → a n * (a (n + 2) - 1) = a (n + 1) * (a (n + 1) - 1) 

noncomputable def sum_seq (a : ℕ → ℕ) : ℕ :=
  ∑ k in range 2024, choose 2023 k * a (k + 1)

theorem sum_sequence_value (a : ℕ → ℕ) (h : sequence a) :
  sum_seq a = 2 * 3^2023 - 2^2023 :=
sorry

end sum_sequence_value_l109_109777


namespace square_side_length_l109_109088

theorem square_side_length(area_sq_cm : ℕ) (h : area_sq_cm = 361) : ∃ side_length : ℕ, side_length ^ 2 = area_sq_cm ∧ side_length = 19 := 
by 
  use 19
  sorry

end square_side_length_l109_109088


namespace hyperbola_eccentricity_l109_109458

theorem hyperbola_eccentricity (a b c : ℝ) (h₁ : 2 * a = 16) (h₂ : 2 * b = 12) (h₃ : c = Real.sqrt (a^2 + b^2)) :
  (c / a) = 5 / 4 :=
by
  sorry

end hyperbola_eccentricity_l109_109458


namespace count_squares_cubes_less_than_1000_l109_109300

theorem count_squares_cubes_less_than_1000 : 
  {n : ℕ | ∃ x : ℕ, n = x^6 ∧ n < 1000}.card = 3 :=
sorry

end count_squares_cubes_less_than_1000_l109_109300


namespace certain_number_l109_109119

theorem certain_number (x certain_number : ℕ) (h1 : x = 3327) (h2 : 9873 + x = certain_number) : 
  certain_number = 13200 := 
by
  sorry

end certain_number_l109_109119


namespace digit_150th_l109_109208

-- Define the fraction and its repeating decimal properties
def fraction := 17 / 70
def repeating_block := "242857"
def block_length : ℕ := 6

-- The 150th digit calculation
def digit_position : ℕ := 150 % block_length

theorem digit_150th : "150th digit of decimal representation" = 7 :=
by {
  have h1 : fraction = 0.242857142857, sorry,
  have h2 : "252857".cycle = repeating_block, sorry,
  have h3 : digit_position = 0, sorry,
  have h4 : repeating_block.get_last = '7', sorry,
  show "150th digit of decimal representation"= 7, sorry,
}

end digit_150th_l109_109208


namespace equal_segments_l109_109945

noncomputable def incircle_touches_at (A B C D : Point) (O : Circle) (incircle : Incircle) : Prop :=
  -- Definition representing the incircle touching BC at D in triangle ABC with center O.
  incircle.touches BC D ∧ incircle = Circle.mk O

noncomputable def diameter_of_incircle (D E : Point) (O : Circle) : Prop :=
  -- Definition representing DE as the diameter of incircle with center O.
  diameter DE O

noncomputable def line_intersects (A E F : Point) (BC : Line) : Prop :=
  -- Definition representing the line AE intersects BC at F.
  intersection A E BC = F

theorem equal_segments (A B C D E F : Point) (O : Circle) (incircle : Incircle) (BC : Line) :
  incircle_touches_at A B C D O incircle → 
  diameter_of_incircle D E O →
  line_intersects A E F BC →
  seg_length B D = seg_length C F :=
begin
  intros h1 h2 h3,
  sorry -- Proof goes here
end

end equal_segments_l109_109945


namespace packets_of_chips_l109_109823

variable (P R M : ℕ)

theorem packets_of_chips (h1: P > 0) (h2: R > 0) (h3: M > 0) :
  ((10 * M * P) / R) = (10 * M * P) / R :=
sorry

end packets_of_chips_l109_109823


namespace sum_boundary_values_of_range_l109_109580

noncomputable def f (x : ℝ) : ℝ := 3 / (3 + 3 * x^2 + 6 * x)

theorem sum_boundary_values_of_range : 
  let c := 0
  let d := 1
  c + d = 1 :=
by
  sorry

end sum_boundary_values_of_range_l109_109580


namespace option_A_option_B_option_C_option_D_l109_109790

theorem option_A (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ n, a (n + 1) = a n + n + 1) : a 20 = 211 :=
sorry

theorem option_B (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n, a (n + 1) = 2^n * a n) : a 5 = 2^10 :=
sorry

theorem option_C (S : ℕ → ℝ) (h₀ : ∀ n, S n = 3^n + 1/2) : ¬(∃ r : ℝ, ∀ n, S n = S 1 * r ^ (n - 1)) :=
sorry

theorem option_D (S : ℕ → ℝ) (a : ℕ → ℝ) (h₀ : S 1 = 1) 
  (h₁ : ∀ n, S n = n * (a 1) + (n * (n - 1) / 2) * (a 2 - a 1))
  (h₂ : (S 8) / 8 - (S 4) / 4 = 8) : a 6 = 21 :=
sorry

end option_A_option_B_option_C_option_D_l109_109790


namespace age_ratio_in_2_years_is_2_1_l109_109249

-- Define the ages and conditions
def son_age (current_year : ℕ) : ℕ := 20
def man_age (current_year : ℕ) : ℕ := son_age current_year + 22

def son_age_in_2_years (current_year : ℕ) : ℕ := son_age current_year + 2
def man_age_in_2_years (current_year : ℕ) : ℕ := man_age current_year + 2

-- The theorem stating the ratio of the man's age to the son's age in two years is 2:1
theorem age_ratio_in_2_years_is_2_1 (current_year : ℕ) :
  man_age_in_2_years current_year = 2 * son_age_in_2_years current_year :=
by
  sorry

end age_ratio_in_2_years_is_2_1_l109_109249


namespace second_polygon_sides_l109_109045

open Nat

theorem second_polygon_sides (s : ℝ) (same_perimeter : ℝ) :
  ∃ n : ℕ, n = 150 :=
by
  let side1 := 3 * s
  have perimeter1 : same_perimeter = 50 * side1 := rfl
  have perimeter2 : same_perimeter = n * s
  have sides : n = 150 :=
  sorry

end second_polygon_sides_l109_109045


namespace rabbit_shape_area_l109_109571

theorem rabbit_shape_area (A_ear : ℝ) (h1 : A_ear = 10) (h2 : A_ear = (1/8) * A_total) :
  A_total = 80 :=
by
  sorry

end rabbit_shape_area_l109_109571


namespace train_speed_computed_l109_109256

noncomputable def train_speed_in_kmh (train_length : ℝ) (platform_length : ℝ) (time_in_seconds : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_mps := total_distance / time_in_seconds
  speed_mps * 3.6

theorem train_speed_computed :
  train_speed_in_kmh 250 50.024 15 = 72.006 := by
  sorry

end train_speed_computed_l109_109256


namespace equivalent_statements_l109_109997

theorem equivalent_statements (P Q : Prop) : (¬P → Q) ↔ (¬Q → P) :=
by
  sorry

end equivalent_statements_l109_109997


namespace egyptian_method_percentage_error_l109_109996

theorem egyptian_method_percentage_error :
  let a := 6
  let b := 4
  let c := 20
  let h := Real.sqrt (c^2 - ((a - b) / 2)^2)
  let S := ((a + b) / 2) * h
  let S1 := ((a + b) * c) / 2
  let percentage_error := abs ((20 / Real.sqrt 399) - 1) * 100
  percentage_error = abs ((20 / Real.sqrt 399) - 1) * 100 := by
  sorry

end egyptian_method_percentage_error_l109_109996


namespace circle_radius_l109_109741

theorem circle_radius 
  {XA XB XC r : ℝ}
  (h1 : XA = 3)
  (h2 : XB = 5)
  (h3 : XC = 1)
  (hx : XA * XB = XC * r)
  (hh : 2 * r = CD) :
  r = 8 :=
by
  sorry

end circle_radius_l109_109741


namespace sum_lengths_DE_EF_equals_9_l109_109372

variable (AB BC FA : ℝ)
variable (area_ABCDEF : ℝ)
variable (DE EF : ℝ)

theorem sum_lengths_DE_EF_equals_9 (h1 : area_ABCDEF = 52) (h2 : AB = 8) (h3 : BC = 9) (h4 : FA = 5)
  (h5 : AB * BC - area_ABCDEF = DE * EF) (h6 : BC - FA = DE) : DE + EF = 9 := 
by 
  sorry

end sum_lengths_DE_EF_equals_9_l109_109372


namespace remainder_of_98_pow_50_mod_50_l109_109992

theorem remainder_of_98_pow_50_mod_50 : (98 ^ 50) % 50 = 0 := by
  sorry

end remainder_of_98_pow_50_mod_50_l109_109992


namespace total_volume_of_drink_l109_109554

theorem total_volume_of_drink :
  ∀ (total_ounces : ℝ),
    (∀ orange_juice watermelon_juice grape_juice : ℝ,
      orange_juice = 0.25 * total_ounces →
      watermelon_juice = 0.4 * total_ounces →
      grape_juice = 0.35 * total_ounces →
      grape_juice = 105 →
      total_ounces = 300) :=
by
  intros total_ounces orange_juice watermelon_juice grape_juice ho hw hg hg_eq
  sorry

end total_volume_of_drink_l109_109554


namespace quilt_shaded_fraction_l109_109035

theorem quilt_shaded_fraction :
  let total_squares := 16
  let shaded_full_square := 4
  let shaded_half_triangles_as_square := 2
  let total_area := total_squares
  let shaded_area := shaded_full_square + shaded_half_triangles_as_square
  shaded_area / total_area = 3 / 8 :=
by
  sorry

end quilt_shaded_fraction_l109_109035


namespace euler_school_voting_problem_l109_109075

theorem euler_school_voting_problem :
  let U := 198
  let A := 149
  let B := 119
  let AcBc := 29
  U - AcBc = 169 → 
  A + B - (U - AcBc) = 99 :=
by
  intros h₁
  sorry

end euler_school_voting_problem_l109_109075


namespace sum_even_integers_102_to_200_l109_109178

theorem sum_even_integers_102_to_200 : 
  (finset.sum (finset.filter (λ n, n % 2 = 0) (finset.range' 102 200.succ))) = 7550 :=
sorry

end sum_even_integers_102_to_200_l109_109178


namespace triangle_inradius_exradii_relation_l109_109402

theorem triangle_inradius_exradii_relation
  (a b c : ℝ) (S : ℝ) (r r_a r_b r_c : ℝ)
  (h_inradius : S = (1/2) * r * (a + b + c))
  (h_exradii_a : r_a = 2 * S / (b + c - a))
  (h_exradii_b : r_b = 2 * S / (c + a - b))
  (h_exradii_c : r_c = 2 * S / (a + b - c))
  (h_area : S = (1/2) * (a * r_a + b * r_b + c * r_c - a * r - b * r - c * r)) :
  1 / r = 1 / r_a + 1 / r_b + 1 / r_c := 
  by sorry

end triangle_inradius_exradii_relation_l109_109402


namespace digit_150_of_17_div_70_l109_109211

theorem digit_150_of_17_div_70 : (decimal_digit 150 (17 / 70)) = 7 :=
sorry

end digit_150_of_17_div_70_l109_109211


namespace servings_in_box_l109_109734

-- Define amounts
def total_cereal : ℕ := 18
def per_serving : ℕ := 2

-- Define the statement to prove
theorem servings_in_box : total_cereal / per_serving = 9 :=
by
  sorry

end servings_in_box_l109_109734


namespace sand_loss_l109_109749

variable (initial_sand : ℝ) (final_sand : ℝ)

theorem sand_loss (h1 : initial_sand = 4.1) (h2 : final_sand = 1.7) :
  initial_sand - final_sand = 2.4 := by
  -- With the given conditions we'll prove this theorem
  sorry

end sand_loss_l109_109749


namespace probability_two_slate_rocks_l109_109059

theorem probability_two_slate_rocks 
    (n_slate : ℕ) (n_pumice : ℕ) (n_granite : ℕ)
    (h_slate : n_slate = 12)
    (h_pumice : n_pumice = 16)
    (h_granite : n_granite = 8) :
    (n_slate / (n_slate + n_pumice + n_granite)) * ((n_slate - 1) / (n_slate + n_pumice + n_granite - 1)) = 11 / 105 :=
by
    sorry

end probability_two_slate_rocks_l109_109059


namespace probability_within_three_units_from_origin_l109_109745

-- Define the properties of the square Q is selected from
def isInSquare (Q : ℝ × ℝ) : Prop :=
  Q.1 ≥ -2 ∧ Q.1 ≤ 2 ∧ Q.2 ≥ -2 ∧ Q.2 ≤ 2

-- Define the condition of being within 3 units from the origin
def withinThreeUnits (Q: ℝ × ℝ) : Prop :=
  (Q.1)^2 + (Q.2)^2 ≤ 9

-- State the problem: Proving the probability is 1
theorem probability_within_three_units_from_origin : 
  ∀ (Q : ℝ × ℝ), isInSquare Q → withinThreeUnits Q := 
by 
  sorry

end probability_within_three_units_from_origin_l109_109745


namespace find_x_l109_109850

variable (N x : ℕ)
variable (h1 : N = 500 * x + 20)
variable (h2 : 4 * 500 + 20 = 2020)

theorem find_x : x = 4 := by
  -- The proof code will go here
  sorry

end find_x_l109_109850


namespace second_polygon_sides_l109_109039

theorem second_polygon_sides (s : ℝ) (h : s > 0) :
  let s1 := 3 * s in
  let s2 := s in
  let sides1 := 50 in
  let perimeter1 := sides1 * s1 in
  let perimeter2 := perimeter1 in
  perimeter2 / s2 = 150 := 
by
  let s1 := 3 * s
  let s2 := s
  let sides1 := 50
  let perimeter1 := sides1 * s1
  let perimeter2 := perimeter1
  show perimeter2 / s2 = 150
  sorry

end second_polygon_sides_l109_109039


namespace digit_150_of_17_div_70_l109_109212

theorem digit_150_of_17_div_70 : (decimal_digit 150 (17 / 70)) = 7 :=
sorry

end digit_150_of_17_div_70_l109_109212


namespace quadratic_equation_in_x_l109_109797

theorem quadratic_equation_in_x (m : ℤ) (h1 : abs m = 2) (h2 : m + 2 ≠ 0) : m = 2 :=
sorry

end quadratic_equation_in_x_l109_109797


namespace count_squares_and_cubes_l109_109306

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l109_109306


namespace compute_fraction_power_l109_109268

theorem compute_fraction_power : 9 * (1 / 7)^4 = 9 / 2401 := by
  sorry

end compute_fraction_power_l109_109268


namespace students_in_grades_2_and_3_l109_109382

theorem students_in_grades_2_and_3 (boys_2nd : ℕ) (girls_2nd : ℕ) (third_grade_factor : ℕ) 
  (h_boys_2nd : boys_2nd = 20) (h_girls_2nd : girls_2nd = 11) (h_third_grade_factor : third_grade_factor = 2) :
  (boys_2nd + girls_2nd) + ((boys_2nd + girls_2nd) * third_grade_factor) = 93 := by
  sorry

end students_in_grades_2_and_3_l109_109382


namespace boyfriend_picks_pieces_l109_109753

theorem boyfriend_picks_pieces (initial_pieces : ℕ) (cat_steals : ℕ) 
(boyfriend_fraction : ℚ) (swept_fraction : ℚ) 
(h_initial : initial_pieces = 60) (h_swept : swept_fraction = 1 / 2) 
(h_cat : cat_steals = 3) (h_boyfriend : boyfriend_fraction = 1 / 3) : 
ℕ :=
  let swept_pieces := initial_pieces * swept_fraction
  let remaining_pieces := swept_pieces - cat_steals
  let boyfriend_pieces := remaining_pieces * boyfriend_fraction
  by
    have h_swept_pieces : swept_pieces = 30 := by sorry
    have h_remaining_pieces : remaining_pieces = 27 := by sorry
    have h_boyfriend_pieces : boyfriend_pieces = 9 := by sorry
    exact h_boyfriend_pieces

end boyfriend_picks_pieces_l109_109753


namespace common_ratio_of_geometric_seq_l109_109004

-- Define the arithmetic sequence
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Define the geometric sequence property
def geometric_seq_property (a2 a3 a6 : ℤ) : Prop :=
  a3 * a3 = a2 * a6

-- State the main theorem
theorem common_ratio_of_geometric_seq (a d : ℤ) (h : ¬d = 0) :
  geometric_seq_property (arithmetic_seq a d 2) (arithmetic_seq a d 3) (arithmetic_seq a d 6) →
  ∃ q : ℤ, q = 3 ∨ q = 1 :=
by
  sorry

end common_ratio_of_geometric_seq_l109_109004


namespace find_angle_C_find_side_a_l109_109456

namespace TriangleProof

-- Declare the conditions and the proof promises
variables {A B C : ℝ} {a b c S : ℝ}

-- First part: Prove angle C
theorem find_angle_C (h1 : c^2 = a^2 + b^2 - a * b) : C = 60 :=
sorry

-- Second part: Prove the value of a
theorem find_side_a (h2 : b = 2) (h3 : S = (3 * Real.sqrt 3) / 2) : a = 3 :=
sorry

end TriangleProof

end find_angle_C_find_side_a_l109_109456


namespace base_number_is_five_l109_109120

theorem base_number_is_five (x k : ℝ) (h1 : x^k = 5) (h2 : x^(2 * k + 2) = 400) : x = 5 :=
by
  sorry

end base_number_is_five_l109_109120


namespace solution_set_of_inequality_l109_109109

theorem solution_set_of_inequality (f : ℝ → ℝ) (hf_diff : differentiable ℝ f)
  (hf_ineq : ∀ x, deriv f x > f x) :
  {x : ℝ | e^(f (real.log x)) - x * f 1 < 0} = set.Ioo 0 (real.exp 1) :=
by sorry

end solution_set_of_inequality_l109_109109


namespace polynomial_coefficients_sum_l109_109125

theorem polynomial_coefficients_sum :
  let p := (5 * x^3 - 3 * x^2 + x - 8) * (8 - 3 * x)
  let a := -15
  let b := 49
  let c := -27
  let d := 32
  let e := -64
  16 * a + 8 * b + 4 * c + 2 * d + e = 44 := 
by
  sorry

end polynomial_coefficients_sum_l109_109125


namespace terminating_decimal_count_l109_109899

theorem terminating_decimal_count : 
  {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ (∃ k : ℕ, n = 49 * k)}.card = 10 :=
by
  sorry

end terminating_decimal_count_l109_109899


namespace find_missing_part_l109_109858

variable (x y : ℚ) -- Using rationals as the base field for generality.

theorem find_missing_part :
  2 * x * (-3 * x^2 * y) = -6 * x^3 * y := 
by
  sorry

end find_missing_part_l109_109858


namespace partition_contains_square_sum_l109_109586

-- Define a natural number n
def is_square (x : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = x

theorem partition_contains_square_sum (n : ℕ) (hn : n ≥ 15) :
  ∀ (A B : fin n → Prop), (∀ x, A x ∨ B x) ∧ (∀ x, ¬ (A x ∧ B x)) → (∃ a b, a ≠ b ∧ A a ∧ A b ∧ is_square (a + b)) ∨ (∃ c d, c ≠ d ∧ B c ∧ B d ∧ is_square (c + d)) :=
by
  sorry

end partition_contains_square_sum_l109_109586


namespace probability_x_greater_3y_in_rectangle_l109_109667

theorem probability_x_greater_3y_in_rectangle :
  let rect := set.Icc (0 : ℝ) 2010 ×ˢ set.Icc (0 : ℝ) 2011 in
  let event := {p : ℝ × ℝ | p.1 > 3 * p.2} in
  let area_rect := (2010 : ℝ) * 2011 in
  let area_event := (2010 : ℝ) * (2010/3) / 2 in
  (event ∩ rect).measure / rect.measure = 335 / 2011 :=
by
  -- The proof goes here
  sorry

end probability_x_greater_3y_in_rectangle_l109_109667


namespace min_value_range_l109_109787

theorem min_value_range:
  ∀ (x m n : ℝ), 
    (y = (3 * x + 2) / (x - 1)) → 
    (∀ x ∈ Set.Ioo m n, y ≥ 3 + 5 / (x - 1)) → 
    (y = 8) → 
    n = 2 → 
    (1 ≤ m ∧ m < 2) := by
  sorry

end min_value_range_l109_109787


namespace inequality_abc_l109_109485

theorem inequality_abc {a b c : ℝ} {n : ℕ} 
  (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) (hn : 0 < n) :
  (1 / (1 + a)^(1 / n : ℝ)) + (1 / (1 + b)^(1 / n : ℝ)) + (1 / (1 + c)^(1 / n : ℝ)) 
  ≤ 3 / (1 + (a * b * c)^(1 / 3 : ℝ))^(1 / n : ℝ) := sorry

end inequality_abc_l109_109485


namespace parallel_lines_l109_109470

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x - a * y + 1 = 0) ∧ (∀ x y : ℝ, (a-1) * x - y + a = 0) →
  (a = 2 ↔ (∀ x1 y1 x2 y2 : ℝ, 2 * x1 - a * y1 + 1 = 0 ∧ (a-1) * x2 - y2 + a = 0 →
  (2 * x1 = (a * y1 - 1) ∧ (a-1) * x2 = y2 - a))) :=
sorry

end parallel_lines_l109_109470


namespace power_mod_eq_one_l109_109692

theorem power_mod_eq_one (n : ℕ) (h₁ : 444 ≡ 3 [MOD 13]) (h₂ : 3^12 ≡ 1 [MOD 13]) :
  444^444 ≡ 1 [MOD 13] :=
by
  sorry

end power_mod_eq_one_l109_109692


namespace probability_x_greater_3y_l109_109656

theorem probability_x_greater_3y (x y : ℝ) (h1: 0 ≤ x ∧ x ≤ 2010) (h2: 0 ≤ y ∧ y ≤ 2011) : 
  (let area_triangle := (2010 * (2010 / 3)) / 2 in 
   let area_rectangle := 2010 * 2011 in 
   (area_triangle / area_rectangle = 335 / 2011)) := 
sorry

end probability_x_greater_3y_l109_109656


namespace real_roots_of_quadratics_l109_109819

theorem real_roots_of_quadratics {p1 p2 q1 q2 : ℝ} (h : p1 * p2 = 2 * (q1 + q2)) :
  (∃ x : ℝ, x^2 + p1 * x + q1 = 0) ∨ (∃ x : ℝ, x^2 + p2 * x + q2 = 0) :=
by
  have D1 := p1^2 - 4 * q1
  have D2 := p2^2 - 4 * q2
  sorry

end real_roots_of_quadratics_l109_109819


namespace matt_current_age_is_65_l109_109863

variable (matt_age james_age : ℕ)

def james_current_age := 30
def james_age_in_5_years := james_current_age + 5
def matt_age_in_5_years := 2 * james_age_in_5_years
def matt_current_age := matt_age_in_5_years - 5

theorem matt_current_age_is_65 : matt_current_age = 65 := 
by
  -- sorry is here to skip the proof.
  sorry

end matt_current_age_is_65_l109_109863


namespace compute_fraction_power_l109_109267

theorem compute_fraction_power : 9 * (1 / 7)^4 = 9 / 2401 := by
  sorry

end compute_fraction_power_l109_109267


namespace inscribed_sphere_l109_109415

theorem inscribed_sphere (r_base height : ℝ) (r_sphere b d : ℝ)
  (h_base : r_base = 15)
  (h_height : height = 20)
  (h_sphere : r_sphere = b * Real.sqrt d - b)
  (h_rsphere_eq : r_sphere = 120 / 11) : 
  b + d = 12 := 
sorry

end inscribed_sphere_l109_109415


namespace james_pitbull_count_l109_109946

-- Defining the conditions
def husky_count : ℕ := 5
def retriever_count : ℕ := 4
def retriever_pups_per_retriever (husky_pups_per_husky : ℕ) : ℕ := husky_pups_per_husky + 2
def husky_pups := husky_count * 3
def retriever_pups := retriever_count * (retriever_pups_per_retriever 3)
def pitbull_pups (P : ℕ) : ℕ := P * 3
def total_pups (P : ℕ) : ℕ := husky_pups + retriever_pups + pitbull_pups P
def total_adults (P : ℕ) : ℕ := husky_count + retriever_count + P
def condition (P : ℕ) : Prop := total_pups P = total_adults P + 30

-- The proof objective
theorem james_pitbull_count : ∃ P : ℕ, condition P → P = 2 := by
  sorry

end james_pitbull_count_l109_109946


namespace second_polygon_sides_l109_109041

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0) : 
  ∀ (P1 P2 : ℝ) (n1 n2 : ℕ), 
    n1 = 50 →
    P1 = n1 * (3 * s) →
    P2 = n2 * s →
    P1 = P2 →
    n2 = 150 :=
by
  intros P1 P2 n1 n2 hn1 hp1 hp2 hp
  sorry

end second_polygon_sides_l109_109041


namespace parallel_lines_condition_iff_l109_109929

def line_parallel (a : ℝ) : Prop :=
  let l1_slope := -1 / -a
  let l2_slope := -(a - 1) / -12
  l1_slope = l2_slope

theorem parallel_lines_condition_iff (a : ℝ) :
  (a = 4) ↔ line_parallel a := by
  sorry

end parallel_lines_condition_iff_l109_109929


namespace length_MN_of_circle_l109_109740

def point := ℝ × ℝ

def circle_passing_through (A B C: point) :=
  ∃ (D E F : ℝ), ∀ (p : point), p = A ∨ p = B ∨ p = C →
    (p.1^2 + p.2^2 + D * p.1 + E * p.2 + F = 0)

theorem length_MN_of_circle (A B C : point) (H : circle_passing_through A B C) :
  A = (1, 3) → B = (4, 2) → C = (1, -7) →
  ∃ M N : ℝ, (A.1 * 0 + N^2 + D * 0 + E * N + F = 0) ∧ (A.1 * 0 + M^2 + D * 0 + E * M + F = 0) ∧
  abs (M - N) = 4 * Real.sqrt 6 := 
sorry

end length_MN_of_circle_l109_109740


namespace seq_a_n_a_4_l109_109637

theorem seq_a_n_a_4 :
  ∃ a : ℕ → ℕ, (a 1 = 1) ∧ (∀ n : ℕ, a (n+1) = 2 * a n) ∧ (a 4 = 8) :=
sorry

end seq_a_n_a_4_l109_109637


namespace count_squares_and_cubes_less_than_1000_l109_109302

theorem count_squares_and_cubes_less_than_1000 :
  {n : ℕ | 0 < n ∧ n < 1000 ∧ ∃ x, n = x ^ 6}.to_finset.card = 3 :=
by sorry

end count_squares_and_cubes_less_than_1000_l109_109302


namespace remainder_444_444_mod_13_l109_109721

theorem remainder_444_444_mod_13 :
  444 ≡ 3 [MOD 13] →
  3^3 ≡ 1 [MOD 13] →
  444^444 ≡ 1 [MOD 13] := by
  intros h1 h2
  sorry

end remainder_444_444_mod_13_l109_109721


namespace trader_profit_loss_l109_109557

noncomputable def profit_loss_percentage (sp1 sp2: ℝ) (gain_loss_rate1 gain_loss_rate2: ℝ) : ℝ :=
  let cp1 := sp1 / (1 + gain_loss_rate1)
  let cp2 := sp2 / (1 - gain_loss_rate2)
  let tcp := cp1 + cp2
  let tsp := sp1 + sp2
  let profit_or_loss := tsp - tcp
  profit_or_loss / tcp * 100

theorem trader_profit_loss : 
  profit_loss_percentage 325475 325475 0.15 0.15 = -2.33 := 
by 
  sorry

end trader_profit_loss_l109_109557


namespace num_int_values_n_terminated_l109_109902

theorem num_int_values_n_terminated (N : ℕ) (hN1 : 1 ≤ N) (hN2 : N ≤ 500) :
  ∃ n : ℕ, n = 10 ∧ ∀ k, 0 ≤ k → k < n → ∃ (m : ℕ), N = m * 49 :=
sorry

end num_int_values_n_terminated_l109_109902


namespace expected_distinct_points_l109_109344

open MeasureTheory
open Asymptotics

def bernoulli_rv {Ω : Type*} (p : ℝ) [measurable_space Ω] [probability_space Ω] : Ω → bool :=
λ ω, if ω ≤ p then true else false

noncomputable def Xi (Ω : Type*) (p : ℝ) [measurable_space Ω] [probability_space Ω] (i : ℕ) : Ω → ℤ :=
λ ω, if bernoulli_rv p ω then 1 else -1

noncomputable def Sn (Ω : Type*) (p : ℝ) [measurable_space Ω] [probability_space Ω] (i : ℕ) : (ℕ → ℤ) :=
λ n, ∑ k in finset.range (n + 1), Xi Ω p k

noncomputable def distinct_points {Ω : Type*} (N : ℕ) (p : ℝ) [measurable_space Ω] [probability_space Ω] : ℕ :=
(finset.range (N + 1)).image (λ n, Sn Ω p n).card

theorem expected_distinct_points (p : ℝ) : 
  p = 1/2 → 
  (λ N, (expectation (λ ω, distinct_points N p))) ~ 
  (λ N, sqrt (2 * N / π)) :=
by sorry

end expected_distinct_points_l109_109344


namespace geometric_series_solution_l109_109376

noncomputable def geometric_series_sums (b1 q : ℝ) : Prop :=
  (b1 / (1 - q) = 16) ∧ (b1^2 / (1 - q^2) = 153.6) ∧ (|q| < 1)

theorem geometric_series_solution (b1 q : ℝ) (h : geometric_series_sums b1 q) :
  q = 2 / 3 ∧ b1 * q^3 = 32 / 9 :=
by
  sorry

end geometric_series_solution_l109_109376


namespace sum_zero_of_distinct_and_ratio_l109_109952

noncomputable def distinct (a b c d : ℝ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

theorem sum_zero_of_distinct_and_ratio (x y u v : ℝ) 
  (h_distinct : distinct x y u v)
  (h_ratio : (x + u) / (x + v) = (y + v) / (y + u)) : 
  x + y + u + v = 0 := 
sorry

end sum_zero_of_distinct_and_ratio_l109_109952


namespace round_robin_odd_game_count_l109_109330

theorem round_robin_odd_game_count (n : ℕ) (h17 : n = 17) :
  ∃ p : ℕ, p < n ∧ (p % 2 = 0) :=
by {
  sorry
}

end round_robin_odd_game_count_l109_109330


namespace combined_area_of_three_walls_l109_109185

theorem combined_area_of_three_walls (A : ℝ) :
  (A - 2 * 30 - 3 * 45 = 180) → (A = 375) :=
by
  intro h
  sorry

end combined_area_of_three_walls_l109_109185


namespace minimum_guests_economical_option_l109_109515

theorem minimum_guests_economical_option :
  ∀ (x : ℕ), (150 + 20 * x > 300 + 15 * x) → x > 30 :=
by 
  intro x
  sorry

end minimum_guests_economical_option_l109_109515


namespace remainder_444_pow_444_mod_13_l109_109706

theorem remainder_444_pow_444_mod_13 :
  (444 ^ 444) % 13 = 1 :=
by
  have h1 : 444 % 13 = 2 := by norm_num
  have h2 : 2 ^ 12 % 13 = 1 := by norm_num
  rw [←pow_mul, ←nat.mul_div_cancel_left 444 12] at h1
  have h3 : 444 = 12 * (444 / 12) := nat.mul_div_cancel_left 444 12
  rw h3
  rw pow_mul
  rw h2
  rw pow_one
  exact h2

end remainder_444_pow_444_mod_13_l109_109706


namespace f_2_eq_4_l109_109316

def f (n : ℕ) : ℕ := (List.range (n + 1)).sum + (List.range n).sum

theorem f_2_eq_4 : f 2 = 4 := by
  sorry

end f_2_eq_4_l109_109316


namespace mysterious_division_l109_109351

theorem mysterious_division (d : ℕ) : (8 * d < 1000) ∧ (7 * d < 900) → d = 124 :=
by
  intro h
  sorry

end mysterious_division_l109_109351


namespace diane_faster_than_rhonda_l109_109960

theorem diane_faster_than_rhonda :
  ∀ (rhonda_time sally_time diane_time total_time : ℕ), 
  rhonda_time = 24 →
  sally_time = rhonda_time + 2 →
  total_time = 71 →
  total_time = rhonda_time + sally_time + diane_time →
  (rhonda_time - diane_time) = 3 :=
by
  intros rhonda_time sally_time diane_time total_time
  intros h_rhonda h_sally h_total h_sum
  sorry

end diane_faster_than_rhonda_l109_109960


namespace geometric_sequence_a4_l109_109005

theorem geometric_sequence_a4 {a : ℕ → ℝ} (q : ℝ) (h₁ : q > 0)
  (h₂ : ∀ n, a (n + 1) = a 1 * q ^ (n)) (h₃ : a 1 = 2) 
  (h₄ : a 2 + 4 = (a 1 + a 3) / 2) : a 4 = 54 := 
by
  sorry

end geometric_sequence_a4_l109_109005


namespace eden_stuffed_bears_l109_109761

theorem eden_stuffed_bears 
  (initial_bears : ℕ) 
  (percentage_kept : ℝ) 
  (sisters : ℕ) 
  (eden_initial_bears : ℕ)
  (h1 : initial_bears = 65) 
  (h2 : percentage_kept = 0.40) 
  (h3 : sisters = 4) 
  (h4 : eden_initial_bears = 20) :
  ∃ eden_bears : ℕ, eden_bears = 29 :=
by
  sorry

end eden_stuffed_bears_l109_109761


namespace inequality_solution_set_l109_109034

noncomputable def solution_set := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

theorem inequality_solution_set : {x : ℝ | (x - 1) * (3 - x) ≥ 0} = solution_set := by
  sorry

end inequality_solution_set_l109_109034


namespace correct_option_l109_109349

noncomputable def M : Set ℝ := {x | x > -2}

theorem correct_option : {0} ⊆ M := 
by 
  intros x hx
  simp at hx
  simp [M]
  show x > -2
  linarith

end correct_option_l109_109349


namespace circle_equation_range_of_k_l109_109122

theorem circle_equation_range_of_k (k : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 4*k*x - 2*y + 5*k = 0) ↔ (k > 1 ∨ k < 1/4) :=
by
  sorry

end circle_equation_range_of_k_l109_109122


namespace count_terminating_decimals_l109_109905

theorem count_terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 500) : 
  (nat.floor (500 / 49) = 10) := 
by
  sorry

end count_terminating_decimals_l109_109905


namespace smallest_six_digit_negative_integer_congruent_to_five_mod_17_l109_109848

theorem smallest_six_digit_negative_integer_congruent_to_five_mod_17 :
  ∃ x : ℤ, x < -100000 ∧ x ≥ -999999 ∧ x % 17 = 5 ∧ x = -100011 :=
by
  sorry

end smallest_six_digit_negative_integer_congruent_to_five_mod_17_l109_109848


namespace dylan_ice_cubes_l109_109888

-- Definitions based on conditions
def trays := 2
def spaces_per_tray := 12
def total_tray_ice := trays * spaces_per_tray
def pitcher_multiplier := 2

-- The statement to be proven
theorem dylan_ice_cubes (x : ℕ) : x + pitcher_multiplier * x = total_tray_ice → x = 8 :=
by {
  sorry
}

end dylan_ice_cubes_l109_109888


namespace probability_x_gt_3y_correct_l109_109653

noncomputable def probability_x_gt_3y : ℚ :=
  let rect_area := (2010 : ℚ) * 2011
  let triangle_area := (2010 : ℚ) * (2010 / 3) / 2
  (triangle_area / rect_area)

theorem probability_x_gt_3y_correct :
  probability_x_gt_3y = 670 / 2011 := 
by
  sorry

end probability_x_gt_3y_correct_l109_109653


namespace count_sixth_powers_below_1000_l109_109313

theorem count_sixth_powers_below_1000 : 
  {n : ℕ | (n < 1000) ∧ (∃ m : ℕ, n = m ^ 6)}.to_finset.card = 3 :=
by sorry

end count_sixth_powers_below_1000_l109_109313


namespace tangent_slope_at_one_one_l109_109374

noncomputable def curve (x : ℝ) : ℝ := x * Real.exp (x - 1)

theorem tangent_slope_at_one_one : (deriv curve 1) = 2 := 
sorry

end tangent_slope_at_one_one_l109_109374


namespace jason_car_count_l109_109009

theorem jason_car_count :
  ∀ (red green purple total : ℕ),
  (green = 4 * red) →
  (red = purple + 6) →
  (purple = 47) →
  (total = purple + red + green) →
  total = 312 :=
by
  intros red green purple total h1 h2 h3 h4
  sorry

end jason_car_count_l109_109009


namespace original_price_per_tire_l109_109599

-- Definitions derived from the problem
def number_of_tires : ℕ := 4
def sale_price_per_tire : ℝ := 75
def total_savings : ℝ := 36

-- Goal to prove the original price of each tire
theorem original_price_per_tire :
  (sale_price_per_tire + total_savings / number_of_tires) = 84 :=
by sorry

end original_price_per_tire_l109_109599


namespace matt_current_age_is_65_l109_109862

variable (matt_age james_age : ℕ)

def james_current_age := 30
def james_age_in_5_years := james_current_age + 5
def matt_age_in_5_years := 2 * james_age_in_5_years
def matt_current_age := matt_age_in_5_years - 5

theorem matt_current_age_is_65 : matt_current_age = 65 := 
by
  -- sorry is here to skip the proof.
  sorry

end matt_current_age_is_65_l109_109862


namespace milk_production_l109_109371

variables (x α y z w β v : ℝ)

theorem milk_production :
  (w * v * β * y) / (α^2 * x * z) = β * y * w * v / (α^2 * x * z) := 
by
  sorry

end milk_production_l109_109371


namespace selling_price_eq_100_l109_109250

variable (CP SP : ℝ)

-- Conditions
def gain : ℝ := 20
def gain_percentage : ℝ := 0.25

-- The proof of the selling price
theorem selling_price_eq_100
  (h1 : gain = 20)
  (h2 : gain_percentage = 0.25)
  (h3 : gain = gain_percentage * CP)
  (h4 : SP = CP + gain) :
  SP = 100 := sorry

end selling_price_eq_100_l109_109250


namespace sum_of_angles_l109_109072

theorem sum_of_angles 
    (ABC_isosceles : ∃ (A B C : Type) (angleBAC : ℝ), (AB = AC) ∧ (angleBAC = 25))
    (DEF_isosceles : ∃ (D E F : Type) (angleEDF : ℝ), (DE = DF) ∧ (angleEDF = 40)) 
    (AD_parallel_CE : Prop) : 
    ∃ (angleDAC angleADE : ℝ), angleDAC = 77.5 ∧ angleADE = 70 ∧ (angleDAC + angleADE = 147.5) :=
by {
  sorry
}

end sum_of_angles_l109_109072


namespace find_x_l109_109315

theorem find_x (b x : ℝ) (hb : b > 1) (hx : x > 0)
  (h : (3 * x)^(Real.log 3 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) :
  x = 1 / 5^(Real.log b / Real.log (5 / 3)) :=
by
  sorry

end find_x_l109_109315


namespace evaluate_expression_at_x_l109_109521

theorem evaluate_expression_at_x (x : ℝ) (h : x = Real.sqrt 2 - 3) : 
  (3 * x / (x^2 - 9)) * (1 - 3 / x) - 2 / (x + 3) = Real.sqrt 2 / 2 := by
  sorry

end evaluate_expression_at_x_l109_109521


namespace lock_combination_correct_l109_109670

noncomputable def lock_combination : ℤ := 812

theorem lock_combination_correct :
  ∀ (S T A R : ℕ), S ≠ T → S ≠ A → S ≠ R → T ≠ A → T ≠ R → A ≠ R →
  ((S * 9^4 + T * 9^3 + A * 9^2 + R * 9 + S) + 
   (T * 9^4 + A * 9^3 + R * 9^2 + T * 9 + S) + 
   (S * 9^4 + T * 9^3 + A * 9^2 + R * 9 + T)) % 9^5 = 
  S * 9^4 + T * 9^3 + A * 9^2 + R * 9 + S →
  (S * 9^2 + T * 9^1 + A) = lock_combination := 
by
  intros S T A R hST hSA hSR hTA hTR hAR h_eq
  sorry

end lock_combination_correct_l109_109670


namespace box_cost_is_550_l109_109967

noncomputable def cost_of_dryer_sheets (loads_per_week : ℕ) (sheets_per_load : ℕ)
                                        (sheets_per_box : ℕ) (annual_savings : ℝ) : ℝ :=
  let sheets_per_week := loads_per_week * sheets_per_load
  let sheets_per_year := sheets_per_week * 52
  let boxes_per_year := sheets_per_year / sheets_per_box
  annual_savings / boxes_per_year

theorem box_cost_is_550 (h1 : 4 = 4)
                        (h2 : 1 = 1)
                        (h3 : 104 = 104)
                        (h4 : 11 = 11) :
  cost_of_dryer_sheets 4 1 104 11 = 5.50 :=
by
  sorry

end box_cost_is_550_l109_109967


namespace find_special_four_digit_number_l109_109585

theorem find_special_four_digit_number :
  ∃ (N : ℕ), 
  (N % 131 = 112) ∧ 
  (N % 132 = 98) ∧ 
  (1000 ≤ N) ∧ 
  (N < 10000) ∧ 
  (N = 1946) :=
sorry

end find_special_four_digit_number_l109_109585


namespace exactly_one_wins_probability_l109_109399

theorem exactly_one_wins_probability :
  let P_A := (2 : ℚ) / 3
  let P_B := (3 : ℚ) / 4
  P_A * (1 - P_B) + P_B * (1 - P_A) = (5 : ℚ) / 12 := by
  let P_A := (2 : ℚ) / 3
  let P_B := (3 : ℚ) / 4
  change P_A * (1 - P_B) + P_B * (1 - P_A) = (5 : ℚ) / 12
  sorry

end exactly_one_wins_probability_l109_109399


namespace simplify_expression_l109_109964

theorem simplify_expression (a b : ℝ) (h : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b - a^2) / (b^2 - a^2) =
  (a^3 - 3 * a * b^2 + 2 * b^3) / (a * b * (b + a)) :=
by
  sorry

end simplify_expression_l109_109964


namespace second_number_is_915_l109_109748

theorem second_number_is_915 :
  ∃ (n1 n2 n3 n4 n5 n6 : ℤ), 
    n1 = 3 ∧ 
    n2 = 915 ∧ 
    n3 = 138 ∧ 
    n4 = 1917 ∧ 
    n5 = 2114 ∧ 
    ∃ x: ℤ, 
      (n1 + n2 + n3 + n4 + n5 + x) / 6 = 12 ∧ 
      n2 = 915 :=
by 
  sorry

end second_number_is_915_l109_109748


namespace number_of_women_l109_109405

theorem number_of_women (w1 w2: ℕ) (m1 m2 d1 d2: ℕ)
    (h1: w2 = 5) (h2: m2 = 100) (h3: d2 = 1) 
    (h4: d1 = 3) (h5: m1 = 360)
    (h6: w1 * d1 = m1 * d2 / m2 * w2) : w1 = 6 :=
by
  sorry

end number_of_women_l109_109405


namespace range_of_a_l109_109294

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 2| + |x + 2| ≤ a^2 - 3 * a) ↔ (a ≥ 4 ∨ a ≤ -1) :=
by
  sorry

end range_of_a_l109_109294


namespace sum_even_102_to_200_l109_109175

noncomputable def sum_even_integers (a b : ℕ) :=
  let n := (b - a) / 2 + 1
  in (n * (a + b)) / 2

theorem sum_even_102_to_200 :
  sum_even_integers 102 200 = 7550 := 
by
  have n : ℕ := (200 - 102) / 2 + 1
  have sum : ℕ := (n * (102 + 200)) / 2
  have n_50 : n = 50 := by sorry
  have sum_7550 : sum = 7550 := by sorry
  exact sum_7550 

end sum_even_102_to_200_l109_109175


namespace δ_can_be_arbitrarily_small_l109_109581

-- Define δ(r) as the distance from the circle to the nearest point with integer coordinates.
def δ (r : ℝ) : ℝ := sorry -- exact definition would depend on the implementation details

-- The main theorem to be proven.
theorem δ_can_be_arbitrarily_small (ε : ℝ) (hε : ε > 0) : ∃ r : ℝ, r > 0 ∧ δ r < ε :=
sorry

end δ_can_be_arbitrarily_small_l109_109581


namespace equilateral_is_cute_specific_triangle_is_cute_find_AB_length_l109_109547

-- Definition of a cute triangle
def is_cute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = 2 * c^2 ∨ a^2 + c^2 = 2 * b^2 ∨ b^2 + c^2 = 2 * a^2

-- 1. Prove an equilateral triangle is a cute triangle
theorem equilateral_is_cute (a : ℝ) : is_cute_triangle a a a :=
by
  sorry

-- 2. Prove the triangle with sides 4, 2√6, and 2√5 is a cute triangle
theorem specific_triangle_is_cute : is_cute_triangle 4 (2*Real.sqrt 6) (2*Real.sqrt 5) :=
by
  sorry

-- 3. Prove the length of AB for the given right triangle is 2√6 or 2√3
theorem find_AB_length (AB BC : ℝ) (AC : ℝ := 2*Real.sqrt 2) (h_cute : is_cute_triangle AB BC AC) : AB = 2*Real.sqrt 6 ∨ AB = 2*Real.sqrt 3 :=
by
  sorry

end equilateral_is_cute_specific_triangle_is_cute_find_AB_length_l109_109547


namespace positive_integer_root_k_l109_109110

theorem positive_integer_root_k (k : ℕ) :
  (∃ x : ℕ, x > 0 ∧ x * x - 34 * x + 34 * k - 1 = 0) ↔ k = 1 :=
by
  sorry

end positive_integer_root_k_l109_109110


namespace wrapping_paper_area_correct_l109_109270

-- Define the length, width, and height of the box
variables (l w h : ℝ)

-- Define the function to calculate the area of the wrapping paper
def wrapping_paper_area (l w h : ℝ) : ℝ := 2 * (l + w + h) ^ 2

-- Statement problem that we need to prove
theorem wrapping_paper_area_correct :
  wrapping_paper_area l w h = 2 * (l + w + h) ^ 2 := 
sorry

end wrapping_paper_area_correct_l109_109270


namespace packets_for_dollars_l109_109370

variable (P R C : ℕ)

theorem packets_for_dollars :
  let dimes := 10 * C
  let taxable_dimes := 9 * C
  ∃ x, x = taxable_dimes * P / R :=
sorry

end packets_for_dollars_l109_109370


namespace arithmetic_identity_l109_109269

theorem arithmetic_identity : 72 * 989 - 12 * 989 = 59340 := by
  sorry

end arithmetic_identity_l109_109269


namespace jakes_weight_l109_109558

theorem jakes_weight
  (J K : ℝ)
  (h1 : J - 8 = 2 * K)
  (h2 : J + K = 290) :
  J = 196 :=
by
  sorry

end jakes_weight_l109_109558


namespace digit_150th_in_decimal_of_fraction_l109_109220

theorem digit_150th_in_decimal_of_fraction : 
  (∀ n : ℕ, n > 0 → (let seq := [2, 4, 2, 8, 5, 7] in seq[((n - 1) % 6)] = 
  if n == 150 then 7 else seq[((n - 1) % 6)])) :=
by
  sorry

end digit_150th_in_decimal_of_fraction_l109_109220


namespace conditional_probability_l109_109803

def slips : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_odd (n : ℕ) : Prop := n % 2 ≠ 0

def P_A : ℚ := 5/9

def P_A_and_B : ℚ := 5/9 * 4/8

theorem conditional_probability :
  (5 / 18) / (5 / 9) = 1 / 2 :=
by
  sorry

end conditional_probability_l109_109803


namespace number_of_boys_in_second_group_l109_109117

noncomputable def daily_work_done_by_man (M : ℝ) (B : ℝ) : Prop :=
  M = 2 * B

theorem number_of_boys_in_second_group
  (M B : ℝ)
  (h1 : (12 * M + 16 * B) * 5 = (13 * M + 24 * B) * 4)
  (h2 : daily_work_done_by_man M B) :
  24 = 24 :=
by
  -- The proof is omitted.
  sorry

end number_of_boys_in_second_group_l109_109117


namespace ratio_bisector_circumradius_l109_109008

theorem ratio_bisector_circumradius (h_a h_b h_c : ℝ) (ha_val : h_a = 1/3) (hb_val : h_b = 1/4) (hc_val : h_c = 1/5) :
  ∃ (CD R : ℝ), CD / R = 24 * Real.sqrt 2 / 35 :=
by
  sorry

end ratio_bisector_circumradius_l109_109008


namespace f_2013_value_l109_109937

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

axiom h1 : ∀ x : ℝ, x ≠ 1 → f (2 * x + 1) + g (3 - x) = x
axiom h2 : ∀ x : ℝ, x ≠ 1 → f ((3 * x + 5) / (x + 1)) + 2 * g ((2 * x + 1) / (x + 1)) = x / (x + 1)

theorem f_2013_value : f 2013 = 1010 / 1007 :=
by
  sorry

end f_2013_value_l109_109937


namespace remainder_444_power_444_mod_13_l109_109710

theorem remainder_444_power_444_mod_13 : (444 ^ 444) % 13 = 1 := by
  sorry

end remainder_444_power_444_mod_13_l109_109710


namespace ratio_Pat_Mark_l109_109352

-- Definitions inferred from the conditions
def total_hours : ℕ := 135
def Kate_hours (K : ℕ) : ℕ := K
def Pat_hours (K : ℕ) : ℕ := 2 * K
def Mark_hours (K : ℕ) : ℕ := K + 75

-- The main statement
theorem ratio_Pat_Mark (K : ℕ) (h : Kate_hours K + Pat_hours K + Mark_hours K = total_hours) :
  (Pat_hours K) / (Mark_hours K) = 1 / 3 := by
  sorry

end ratio_Pat_Mark_l109_109352


namespace digit_150_of_17_div_70_is_2_l109_109222

def repeating_cycle_17_div_70 : List ℕ := [2, 4, 2, 8, 5, 7, 1]

theorem digit_150_of_17_div_70_is_2 : 
  (repeating_cycle_17_div_70[(150 % 7) - 1] = 2) :=
by
  -- the proof will go here
  sorry

end digit_150_of_17_div_70_is_2_l109_109222


namespace ned_price_per_game_l109_109954

def number_of_games : Nat := 15
def non_working_games : Nat := 6
def total_earnings : Nat := 63
def number_of_working_games : Nat := number_of_games - non_working_games
def price_per_working_game : Nat := total_earnings / number_of_working_games

theorem ned_price_per_game : price_per_working_game = 7 :=
by
  sorry

end ned_price_per_game_l109_109954


namespace original_number_l109_109723

theorem original_number (N : ℕ) (h : ∃ k : ℕ, N + 1 = 9 * k) : N = 8 :=
sorry

end original_number_l109_109723


namespace sum_distances_l109_109781

def point := ℝ × ℝ

noncomputable def circle_C (ρ θ : ℝ) : Prop := ρ = 4 * real.cos θ
noncomputable def line_l (P : point) (angle: ℝ) (x y: ℝ) : Prop := y = P.2 + real.sin (angle) * (x - P.1) / real.cos (angle)

theorem sum_distances (P : point) (A B : point) (C_center : point) (radius : ℝ) :
  P = (2,1) → ∀ θ ρ, circle_C ρ θ → A ≠ P → B ≠ P →
  line_l P (3*real.pi/4) A.1 A.2 → line_l P (3*real.pi/4) B.1 B.2 →
  A ≠ B ∧ (A.1 - C_center.1)^2 + A.2^2 = radius^2 ∧
  (B.1 - C_center.1)^2 + B.2^2 = radius^2 →
  C_center = (2, 0) → radius = 2 →
  |((P.1 - A.1)^2 + (P.2 - A.2)^2)^0.5 + ((P.1 - B.1)^2 + (P.2 - B.2)^2)^0.5| = real.sqrt 14 :=
by
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ _
  sorry 

end sum_distances_l109_109781


namespace power_mod_eq_one_l109_109694

theorem power_mod_eq_one (n : ℕ) (h₁ : 444 ≡ 3 [MOD 13]) (h₂ : 3^12 ≡ 1 [MOD 13]) :
  444^444 ≡ 1 [MOD 13] :=
by
  sorry

end power_mod_eq_one_l109_109694


namespace tiered_water_pricing_usage_l109_109186

theorem tiered_water_pricing_usage (total_cost : ℤ) (water_used : ℤ) :
  (total_cost = 60) →
  (water_used > 12 ∧ water_used ≤ 18) →
  (3 * 12 + (water_used - 12) * 6 = total_cost) →
  water_used = 16 :=
by
  intros h_cost h_range h_eq
  sorry

end tiered_water_pricing_usage_l109_109186


namespace total_points_scored_l109_109887

-- Definitions based on the conditions
def three_point_shots := 13
def two_point_shots := 20
def free_throws := 5
def missed_free_throws := 2
def points_per_three_point_shot := 3
def points_per_two_point_shot := 2
def points_per_free_throw := 1
def penalty_per_missed_free_throw := 1

-- Main statement proving the total points James scored
theorem total_points_scored :
  three_point_shots * points_per_three_point_shot +
  two_point_shots * points_per_two_point_shot +
  free_throws * points_per_free_throw -
  missed_free_throws * penalty_per_missed_free_throw = 82 :=
by
  sorry

end total_points_scored_l109_109887


namespace degrees_multiplication_proof_l109_109061

/-- Convert a measurement given in degrees and minutes to purely degrees. -/
def degrees (d : Int) (m : Int) : ℚ := d + m / 60

/-- Given conditions: -/
def lhs : ℚ := degrees 21 17
def rhs : ℚ := degrees 106 25

/-- The theorem to prove the mathematical problem. -/
theorem degrees_multiplication_proof : lhs * 5 = rhs := sorry

end degrees_multiplication_proof_l109_109061


namespace solve_for_x_l109_109363

theorem solve_for_x (x : ℚ) : (2/3 : ℚ) - 1/4 = 1/x → x = 12/5 := 
by
  sorry

end solve_for_x_l109_109363


namespace correct_statements_l109_109677

theorem correct_statements : 
    let statement1 := "The regression effect is characterized by the relevant exponent R^{2}. The larger the R^{2}, the better the fitting effect."
    let statement2 := "The properties of a sphere are inferred from the properties of a circle by analogy."
    let statement3 := "Any two complex numbers cannot be compared in size."
    let statement4 := "Flowcharts are often used to represent some dynamic processes, usually with a 'starting point' and an 'ending point'."
    true -> (statement1 = "correct" ∧ statement2 = "correct" ∧ statement3 = "incorrect" ∧ statement4 = "incorrect") :=
by
  -- proof
  sorry

end correct_statements_l109_109677


namespace digit_150th_of_17_div_70_is_7_l109_109202

noncomputable def decimal_representation (n d : ℚ) : ℚ := n / d

-- We define the repeating part of the decimal representation.
def repeating_part : ℕ := 242857

-- Define the function to get the nth digit of the repeating part.
def get_nth_digit_of_repeating (n : ℕ) : ℕ :=
  let sequence := [2, 4, 2, 8, 5, 7]
  sequence.get! (n % sequence.length)

theorem digit_150th_of_17_div_70_is_7 : get_nth_digit_of_repeating 149 = 7 :=
by
  sorry

end digit_150th_of_17_div_70_is_7_l109_109202


namespace three_Z_five_l109_109124

def Z (a b : ℤ) : ℤ := b + 7 * a - 3 * a^2

theorem three_Z_five : Z 3 5 = -1 := by
  sorry

end three_Z_five_l109_109124


namespace find_c_l109_109084

theorem find_c (c : ℝ) (h1 : ∃ x : ℝ, (⌊c⌋ : ℝ) = x ∧ 3 * x^2 + 12 * x - 27 = 0)
                      (h2 : ∃ x : ℝ, (c - ⌊c⌋) = x ∧ 4 * x^2 - 12 * x + 5 = 0) :
                      c = -8.5 :=
by
  sorry

end find_c_l109_109084


namespace min_value_of_expression_l109_109991

theorem min_value_of_expression (x y : ℝ) : 
  ∃ x y, 2 * x^2 + 3 * y^2 - 8 * x + 12 * y + 40 = 20 := 
sorry

end min_value_of_expression_l109_109991


namespace solve_for_x_l109_109360

theorem solve_for_x:
  ∃ x : ℚ, (2 / 3 - 1 / 4 = 1 / x) ∧ (x = 12 / 5) := by
  sorry

end solve_for_x_l109_109360


namespace cost_of_green_lettuce_l109_109136

-- Definitions based on the conditions given in the problem
def cost_per_pound := 2
def weight_red_lettuce := 6 / cost_per_pound
def total_weight := 7
def weight_green_lettuce := total_weight - weight_red_lettuce

-- Problem statement: Prove that the cost of green lettuce is $8
theorem cost_of_green_lettuce : (weight_green_lettuce * cost_per_pound) = 8 :=
by
  sorry

end cost_of_green_lettuce_l109_109136


namespace cos_double_angle_l109_109283

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (2 * θ) = -7 / 9 :=
by
  sorry

end cos_double_angle_l109_109283


namespace probability_x_gt_3y_correct_l109_109651

noncomputable def probability_x_gt_3y : ℚ :=
  let rect_area := (2010 : ℚ) * 2011
  let triangle_area := (2010 : ℚ) * (2010 / 3) / 2
  (triangle_area / rect_area)

theorem probability_x_gt_3y_correct :
  probability_x_gt_3y = 670 / 2011 := 
by
  sorry

end probability_x_gt_3y_correct_l109_109651


namespace max_ab_l109_109459

theorem max_ab (a b : ℝ) (h1 : a + 4 * b = 8) (h2 : a > 0) (h3 : b > 0) : ab ≤ 4 := 
sorry

end max_ab_l109_109459


namespace NES_sale_price_l109_109188

-- Define all the conditions
def SNES_value : ℝ := 150
def trade_in_percentage : ℝ := 0.8
def additional_money : ℝ := 80
def change_received : ℝ := 10
def game_value : ℝ := 30

-- Proving the sale price of the NES
theorem NES_sale_price :
  let trade_in_value := SNES_value * trade_in_percentage in
  let total_spent := trade_in_value + additional_money in
  let total_received := change_received + game_value in
  total_spent - total_received = 160 :=
by
  sorry

end NES_sale_price_l109_109188


namespace profit_calculation_l109_109407

variable (x y : ℝ)

-- Conditions
def fabric_constraints_1 : Prop := (0.5 * x + 0.9 * (50 - x) ≤ 38)
def fabric_constraints_2 : Prop := (x + 0.2 * (50 - x) ≤ 26)
def x_range : Prop := (17.5 ≤ x ∧ x ≤ 20)

-- Goal
def profit_expression : ℝ := 15 * x + 1500

theorem profit_calculation (h1 : fabric_constraints_1 x) (h2 : fabric_constraints_2 x) (h3 : x_range x) : y = profit_expression x :=
by
  sorry

end profit_calculation_l109_109407


namespace part1_part2_l109_109453

theorem part1 (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 3) : a^2 + b^2 = 22 :=
sorry

theorem part2 (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 3) : (a - 2) * (b + 2) = 7 :=
sorry

end part1_part2_l109_109453


namespace seq_eq_exp_l109_109611

theorem seq_eq_exp (a : ℕ → ℕ) 
  (h₀ : a 1 = 2) 
  (h₁ : ∀ n ≥ 2, a n = 2 * a (n - 1) - 1) :
  ∀ n ≥ 2, a n = 2^(n-1) + 1 := 
  by 
  sorry

end seq_eq_exp_l109_109611


namespace jenni_age_l109_109174

theorem jenni_age 
    (B J : ℤ)
    (h1 : B + J = 70)
    (h2 : B - J = 32) : 
    J = 19 :=
by
  sorry

end jenni_age_l109_109174


namespace monkey_climbing_time_l109_109744

-- Define the conditions
def tree_height : ℕ := 20
def hop_distance : ℕ := 3
def slip_distance : ℕ := 2
def net_distance_per_hour : ℕ := hop_distance - slip_distance

-- Define the theorem statement
theorem monkey_climbing_time : ∃ (t : ℕ), t = 18 ∧ (net_distance_per_hour * (t - 1) + hop_distance) >= tree_height :=
by
  sorry

end monkey_climbing_time_l109_109744


namespace top_card_yellow_second_card_not_yellow_l109_109411

-- Definitions based on conditions
def total_cards : Nat := 65

def yellow_cards : Nat := 13

def non_yellow_cards : Nat := total_cards - yellow_cards

-- Total combinations of choosing two cards
def total_combinations : Nat := total_cards * (total_cards - 1)

-- Numerator for desired probability 
def desired_combinations : Nat := yellow_cards * non_yellow_cards

-- Target probability
def desired_probability : Rat := Rat.ofInt (desired_combinations) / Rat.ofInt (total_combinations)

-- Mathematical proof statement
theorem top_card_yellow_second_card_not_yellow :
  desired_probability = Rat.ofInt 169 / Rat.ofInt 1040 :=
by
  sorry

end top_card_yellow_second_card_not_yellow_l109_109411


namespace percentage_increase_in_surface_area_l109_109323

variable (a : ℝ)

theorem percentage_increase_in_surface_area (ha : a > 0) :
  let original_surface_area := 6 * a^2
  let new_edge_length := 1.5 * a
  let new_surface_area := 6 * (new_edge_length)^2
  let area_increase := new_surface_area - original_surface_area
  let percentage_increase := (area_increase / original_surface_area) * 100
  percentage_increase = 125 := 
by 
  let original_surface_area := 6 * a^2
  let new_edge_length := 1.5 * a
  let new_surface_area := 6 * (new_edge_length)^2
  let area_increase := new_surface_area - original_surface_area
  let percentage_increase := (area_increase / original_surface_area) * 100
  sorry

end percentage_increase_in_surface_area_l109_109323


namespace kareem_has_largest_final_number_l109_109809

def jose_final : ℕ := (15 - 2) * 4 + 5
def thuy_final : ℕ := (15 * 3 - 3) - 4
def kareem_final : ℕ := ((20 - 3) + 4) * 3

theorem kareem_has_largest_final_number :
  kareem_final > jose_final ∧ kareem_final > thuy_final := 
by 
  sorry

end kareem_has_largest_final_number_l109_109809


namespace dogwood_trees_after_work_l109_109942

theorem dogwood_trees_after_work 
  (trees_part1 : ℝ) (trees_part2 : ℝ) (trees_part3 : ℝ)
  (trees_cut : ℝ) (trees_planted : ℝ)
  (h1 : trees_part1 = 5.0) (h2 : trees_part2 = 4.0) (h3 : trees_part3 = 6.0)
  (h_cut : trees_cut = 7.0) (h_planted : trees_planted = 3.0) :
  trees_part1 + trees_part2 + trees_part3 - trees_cut + trees_planted = 11.0 :=
by
  sorry

end dogwood_trees_after_work_l109_109942


namespace bookish_campus_reading_l109_109872

-- Definitions of the variables involved in the problem
def reading_hours : List ℝ := [4, 5, 5, 6, 10]

def mean (l: List ℝ) : ℝ := l.sum / l.length

def variance (l: List ℝ) : ℝ :=
  let μ := mean l
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

-- The proof statement
theorem bookish_campus_reading :
  mean reading_hours = 6 ∧ variance reading_hours = 4.4 :=
by
  sorry

end bookish_campus_reading_l109_109872


namespace root_of_f_l109_109165

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

theorem root_of_f (h_inv : f_inv 0 = 2) (h_interval : 1 ≤ (f_inv 0) ∧ (f_inv 0) ≤ 4) : f 2 = 0 := 
sorry

end root_of_f_l109_109165


namespace factor_expression_correct_l109_109769

noncomputable def factor_expression (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem factor_expression_correct (a b c : ℝ) :
  factor_expression a b c = (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
by
  sorry

end factor_expression_correct_l109_109769


namespace total_potatoes_now_l109_109147

def initial_potatoes : ℕ := 8
def uneaten_new_potatoes : ℕ := 3

theorem total_potatoes_now : initial_potatoes + uneaten_new_potatoes = 11 := by
  sorry

end total_potatoes_now_l109_109147


namespace inequality_abc_l109_109517

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) / (a^2 + b^2) + (b + c) / (b^2 + c^2) + (c + a) / (c^2 + a^2) ≤ 1/a + 1/b + 1/c := 
by
  sorry

end inequality_abc_l109_109517


namespace sale_price_relationship_l109_109276

/-- Elaine's Gift Shop increased the original prices of all items by 10% 
  and then offered a 30% discount on these new prices in a clearance sale 
  - proving the relationship between the final sale price and the original price of an item -/

theorem sale_price_relationship (p : ℝ) : 
  (0.7 * (1.1 * p) = 0.77 * p) :=
by 
  sorry

end sale_price_relationship_l109_109276


namespace constant_function_solution_l109_109275

theorem constant_function_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, 2 * f x = f (x + y) + f (x + 2 * y)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end constant_function_solution_l109_109275


namespace number_of_individuals_left_at_zoo_l109_109245

theorem number_of_individuals_left_at_zoo 
  (students_class1 students_class2 students_left : ℕ)
  (initial_chaperones remaining_chaperones teachers : ℕ) :
  students_class1 = 10 ∧
  students_class2 = 10 ∧
  initial_chaperones = 5 ∧
  teachers = 2 ∧
  students_left = 10 ∧
  remaining_chaperones = initial_chaperones - 2 →
  (students_class1 + students_class2 - students_left) + remaining_chaperones + teachers = 15 :=
by
  sorry

end number_of_individuals_left_at_zoo_l109_109245


namespace minimum_daily_expense_l109_109240

-- Defining the context
variables (x y : ℕ)
def total_capacity (x y : ℕ) : ℕ := 24 * x + 30 * y
def cost (x y : ℕ) : ℕ := 320 * x + 504 * y

theorem minimum_daily_expense :
  (total_capacity x y ≥ 180) →
  (x ≤ 8) →
  (y ≤ 4) →
  cost x y = 2560 := sorry

end minimum_daily_expense_l109_109240


namespace belt_length_sufficient_l109_109981

theorem belt_length_sufficient (r O_1O_2 O_1O_3 O_3_plane : ℝ) 
(O_1O_2_eq : O_1O_2 = 12) (O_1O_3_eq : O_1O_3 = 10) (O_3_plane_eq : O_3_plane = 8) (r_eq : r = 2) : 
(∃ L₁ L₂, L₁ = 32 + 4 * Real.pi ∧ L₂ = 22 + 2 * Real.sqrt 97 + 4 * Real.pi ∧ 
L₁ ≠ 54 ∧ L₂ > 54) := 
by 
  sorry

end belt_length_sufficient_l109_109981


namespace initial_numbers_l109_109844

theorem initial_numbers (x : ℕ) (h1 : 2015 > x) (h2 : ∃ (k : ℕ), 2015 - x = 1024 * k) : x = 991 :=
by {
  sorry
}

end initial_numbers_l109_109844


namespace digit_150_of_17_div_70_l109_109215

theorem digit_150_of_17_div_70 : (decimal_digit 150 (17 / 70)) = 7 :=
sorry

end digit_150_of_17_div_70_l109_109215


namespace range_of_x_minus_cos_y_l109_109137

theorem range_of_x_minus_cos_y
  (x y : ℝ)
  (h : x^2 + 2 * Real.cos y = 1) :
  ∃ (A : Set ℝ), A = {z | -1 ≤ z ∧ z ≤ 1 + Real.sqrt 3} ∧ x - Real.cos y ∈ A :=
by
  sorry

end range_of_x_minus_cos_y_l109_109137


namespace second_polygon_sides_l109_109043

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0) : 
  ∀ (P1 P2 : ℝ) (n1 n2 : ℕ), 
    n1 = 50 →
    P1 = n1 * (3 * s) →
    P2 = n2 * s →
    P1 = P2 →
    n2 = 150 :=
by
  intros P1 P2 n1 n2 hn1 hp1 hp2 hp
  sorry

end second_polygon_sides_l109_109043


namespace solve_for_x_l109_109362

theorem solve_for_x : ∃ x : ℚ, (2/3 - 1/4) = 1/x ∧ x = 12/5 :=
by
  use 12/5
  split
  · norm_num
  · norm_num
  · sorry

end solve_for_x_l109_109362


namespace inequality_abc_l109_109144

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
sorry

end inequality_abc_l109_109144


namespace only_correct_option_is_C_l109_109228

-- Definitions of the conditions as per the given problem
def option_A (a : ℝ) : Prop := a^2 * a^3 = a^6
def option_B (a : ℝ) : Prop := (a^2)^3 = a^5
def option_C (a b : ℝ) : Prop := (a * b)^3 = a^3 * b^3
def option_D (a : ℝ) : Prop := a^8 / a^2 = a^4

-- The theorem stating that only option C is correct
theorem only_correct_option_is_C (a b : ℝ) : 
  ¬(option_A a) ∧ ¬(option_B a) ∧ option_C a b ∧ ¬(option_D a) :=
by sorry

end only_correct_option_is_C_l109_109228


namespace gcd_lcm_ratio_l109_109488

theorem gcd_lcm_ratio (A B : ℕ) (k : ℕ) (h1 : Nat.lcm A B = 200) (h2 : 2 * k = A) (h3 : 5 * k = B) : Nat.gcd A B = k :=
by
  sorry

end gcd_lcm_ratio_l109_109488


namespace max_ab_l109_109460

theorem max_ab (a b : ℝ) (h1 : a + 4 * b = 8) (h2 : a > 0) (h3 : b > 0) : ab ≤ 4 := 
sorry

end max_ab_l109_109460


namespace find_other_number_l109_109971

theorem find_other_number (lcm_ab hcf_ab : ℕ) (A : ℕ) (h_lcm: Nat.lcm A (B) = lcm_ab)
  (h_hcf : Nat.gcd A (B) = hcf_ab) (h_a : A = 48) (h_lcm_value: lcm_ab = 192) (h_hcf_value: hcf_ab = 16) :
  B = 64 :=
by
  sorry

end find_other_number_l109_109971


namespace find_missing_number_l109_109973

theorem find_missing_number (x : ℕ) : 
  (1 + 22 + 23 + 24 + 25 + 26 + x + 2) / 8 = 20 → x = 37 := by
  sorry

end find_missing_number_l109_109973


namespace Tim_placed_rulers_l109_109842

variable (initial_rulers final_rulers : ℕ)
variable (placed_rulers : ℕ)

-- Given conditions
def initial_rulers_def : initial_rulers = 11 := sorry
def final_rulers_def : final_rulers = 25 := sorry

-- Goal
theorem Tim_placed_rulers : placed_rulers = final_rulers - initial_rulers :=
  by
  sorry

end Tim_placed_rulers_l109_109842


namespace seating_arrangement_l109_109555

theorem seating_arrangement (students : ℕ) (desks : ℕ) (empty_desks : ℕ) 
  (h_students : students = 2) (h_desks : desks = 5) 
  (h_empty : empty_desks ≥ 1) :
  ∃ ways, ways = 12 := by
  sorry

end seating_arrangement_l109_109555


namespace digit_150th_of_17_div_70_is_7_l109_109203

noncomputable def decimal_representation (n d : ℚ) : ℚ := n / d

-- We define the repeating part of the decimal representation.
def repeating_part : ℕ := 242857

-- Define the function to get the nth digit of the repeating part.
def get_nth_digit_of_repeating (n : ℕ) : ℕ :=
  let sequence := [2, 4, 2, 8, 5, 7]
  sequence.get! (n % sequence.length)

theorem digit_150th_of_17_div_70_is_7 : get_nth_digit_of_repeating 149 = 7 :=
by
  sorry

end digit_150th_of_17_div_70_is_7_l109_109203


namespace odd_power_sum_divisible_l109_109546

theorem odd_power_sum_divisible (x y : ℤ) (n : ℕ) (h_odd : ∃ k : ℕ, n = 2 * k + 1) :
  (x ^ n + y ^ n) % (x + y) = 0 := 
sorry

end odd_power_sum_divisible_l109_109546


namespace initial_pencils_count_l109_109356

-- Define the conditions
def students : ℕ := 25
def pencils_per_student : ℕ := 5

-- Statement of the proof problem
theorem initial_pencils_count : students * pencils_per_student = 125 :=
by
  sorry

end initial_pencils_count_l109_109356


namespace simplify_sqrt_20_minus_sqrt_5_plus_sqrt_one_fifth_simplify_fraction_of_sqrt_12_plus_sqrt_18_minus_sqrt_half_times_sqrt_3_l109_109428

-- Problem (1)
theorem simplify_sqrt_20_minus_sqrt_5_plus_sqrt_one_fifth :
  (Real.sqrt 20 - Real.sqrt 5 + Real.sqrt (1 / 5) = 6 * Real.sqrt 5 / 5) :=
by
  sorry

-- Problem (2)
theorem simplify_fraction_of_sqrt_12_plus_sqrt_18_minus_sqrt_half_times_sqrt_3 :
  (Real.sqrt 12 + Real.sqrt 18) / Real.sqrt 3 - 2 * Real.sqrt (1 / 2) * Real.sqrt 3 = 2 :=
by
  sorry

end simplify_sqrt_20_minus_sqrt_5_plus_sqrt_one_fifth_simplify_fraction_of_sqrt_12_plus_sqrt_18_minus_sqrt_half_times_sqrt_3_l109_109428


namespace seconds_in_3_hours_45_minutes_l109_109116

theorem seconds_in_3_hours_45_minutes :
  let hours := 3
  let minutes := 45
  let minutes_in_hour := 60
  let seconds_in_minute := 60
  (hours * minutes_in_hour + minutes) * seconds_in_minute = 13500 := by
  sorry

end seconds_in_3_hours_45_minutes_l109_109116


namespace area_kappa_l109_109431

open Real

def regular_ngon_area (n : ℕ) (s : ℝ) : ℝ := sorry
def regular_ngon_circumradius_area (n : ℕ) (c : ℝ) : ℝ := sorry
def kappa_enclosed_area (n : ℕ) (s : ℝ) (c : ℝ) : ℝ := sorry

theorem area_kappa (n : ℕ) (A : ℝ) (B : ℝ)
  (hA : A = regular_ngon_area n 1)
  (hB : B = regular_ngon_circumradius_area n 1) :
  kappa_enclosed_area n 1 1 = 6 * A - 2 * B :=
sorry

end area_kappa_l109_109431


namespace geometric_series_solution_l109_109375

noncomputable def geometric_series_sums (b1 q : ℝ) : Prop :=
  (b1 / (1 - q) = 16) ∧ (b1^2 / (1 - q^2) = 153.6) ∧ (|q| < 1)

theorem geometric_series_solution (b1 q : ℝ) (h : geometric_series_sums b1 q) :
  q = 2 / 3 ∧ b1 * q^3 = 32 / 9 :=
by
  sorry

end geometric_series_solution_l109_109375


namespace base7_arithmetic_l109_109279

theorem base7_arithmetic : 
  let b1000 := 343  -- corresponding to 1000_7 in decimal
  let b666 := 342   -- corresponding to 666_7 in decimal
  let b1234 := 466  -- corresponding to 1234_7 in decimal
  let s := b1000 + b666  -- sum in decimal
  let s_base7 := 1421    -- sum back in base7 (1421 corresponds to 685 in decimal)
  let r_base7 := 254     -- result from subtraction in base7 (254 corresponds to 172 in decimal)
  (1000 * 7^0 + 0 * 7^1 + 0 * 7^2 + 1 * 7^3) + (6 * 7^0 + 6 * 7^1 + 6 * 7^2) - (4 * 7^0 + 3 * 7^1 + 2 * 7^2 + 1 * 7^3) = (4 * 7^0 + 5 * 7^1 + 2 * 7^2)
  :=
sorry

end base7_arithmetic_l109_109279


namespace entrance_ticket_cost_l109_109561

theorem entrance_ticket_cost
  (students teachers : ℕ)
  (total_cost : ℕ)
  (students_count : students = 20)
  (teachers_count : teachers = 3)
  (cost : total_cost = 115) :
  total_cost / (students + teachers) = 5 := by
  sorry

end entrance_ticket_cost_l109_109561


namespace total_students_l109_109381

theorem total_students (boys_2nd:int) (girls_2nd:int) (students_2nd:int) (students_3rd:int) (total_students:int):
  (boys_2nd = 20) ->
  (girls_2nd = 11) ->
  (students_2nd = boys_2nd + girls_2nd) ->
  (students_3rd = 2 * students_2nd) ->
  (total_students = students_2nd + students_3rd) ->
  total_students = 93 := 
begin
  intros h1 h2 h3 h4 h5,
  rw [h1, h2] at h3,
  rw h3 at h4,
  rw [h3, h4] at h5,
  exact h5,
end

end total_students_l109_109381


namespace spinner_probability_l109_109742

-- Define the game board conditions
def total_regions : ℕ := 12  -- The triangle is divided into 12 smaller regions
def shaded_regions : ℕ := 3  -- Three regions are shaded

-- Define the probability calculation
def probability (total : ℕ) (shaded : ℕ): ℚ := shaded / total

-- State the proof problem
theorem spinner_probability :
  probability total_regions shaded_regions = 1 / 4 :=
by
  sorry

end spinner_probability_l109_109742


namespace original_price_color_TV_l109_109868

theorem original_price_color_TV (x : ℝ) 
  (h : 1.12 * x - x = 144) : 
  x = 1200 :=
sorry

end original_price_color_TV_l109_109868


namespace cars_in_fourth_store_l109_109231

theorem cars_in_fourth_store
  (mean : ℝ) 
  (a1 a2 a3 a5 : ℝ) 
  (num_stores : ℝ) 
  (mean_value : mean = 20.8) 
  (a1_value : a1 = 30) 
  (a2_value : a2 = 14) 
  (a3_value : a3 = 14) 
  (a5_value : a5 = 25) 
  (num_stores_value : num_stores = 5) :
  ∃ x : ℝ, (a1 + a2 + a3 + x + a5) / num_stores = mean ∧ x = 21 :=
by
  sorry

end cars_in_fourth_store_l109_109231


namespace range_of_a_l109_109509

noncomputable def p (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^2 * x^2 + a * x - 2 = 0

noncomputable def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → (1 < a ∨ -1 < a ∧ a < 1) :=
by sorry

end range_of_a_l109_109509


namespace sum_abs_coeffs_expansion_l109_109594

theorem sum_abs_coeffs_expansion (x : ℝ) :
  (|1 - 0 * x| + |1 - 3 * x| + |1 - 3^2 * x^2| + |1 - 3^3 * x^3| + |1 - 3^4 * x^4| + |1 - 3^5 * x^5| = 1024) :=
sorry

end sum_abs_coeffs_expansion_l109_109594


namespace find_expression_l109_109860

theorem find_expression (x y : ℝ) : 2 * x * (-3 * x^2 * y) = -6 * x^3 * y := by
  sorry

end find_expression_l109_109860


namespace arithmetic_seq_general_term_geometric_seq_general_term_l109_109464

theorem arithmetic_seq_general_term (a : ℕ → ℝ) (h1 : a 1 + a 2 = 10) (h2 : a 4 - a 3 = 2) :
  ∀ n, a n = 2 * n + 2 :=
by sorry

theorem geometric_seq_general_term (a b : ℕ → ℝ) (h1 : a 1 + a 2 = 10) (h2 : a 4 - a 3 = 2)
  (h3 : b 2 = a 3) (h4 : b 3 = a 7) :
  ∀ n, b n = 2 ^ (n + 1) :=
by sorry

end arithmetic_seq_general_term_geometric_seq_general_term_l109_109464


namespace parabola_vertex_l109_109841

theorem parabola_vertex :
  ∃ (x y : ℝ), (∀ t : ℝ, t^2 + 2 * t - 2 ≥ y) ∧ (x^2 + 2 * x - 2 = y) ∧ (x = -1) ∧ (y = -3) :=
by sorry

end parabola_vertex_l109_109841


namespace hands_opposite_22_times_in_day_l109_109556

def clock_hands_opposite_in_day : ℕ := 22

def minute_hand_speed := 12
def opposite_line_minutes := 30

theorem hands_opposite_22_times_in_day (minute_hand_speed: ℕ) (opposite_line_minutes : ℕ) : 
  minute_hand_speed = 12 →
  opposite_line_minutes = 30 →
  clock_hands_opposite_in_day = 22 :=
by
  intros h1 h2
  sorry

end hands_opposite_22_times_in_day_l109_109556


namespace sum_of_roots_l109_109625

theorem sum_of_roots (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hroots : ∀ x : ℝ, x^2 - p*x + 2*q = 0) :
  p + q = p :=
by sorry

end sum_of_roots_l109_109625


namespace linear_function_not_in_second_quadrant_l109_109457

theorem linear_function_not_in_second_quadrant (m : ℤ) (h1 : m + 4 > 0) (h2 : m + 2 ≤ 0) : 
  m = -3 ∨ m = -2 := 
sorry

end linear_function_not_in_second_quadrant_l109_109457


namespace largest_fraction_among_list_l109_109725

theorem largest_fraction_among_list :
  ∃ (f : ℚ), f = 105 / 209 ∧ 
  (f > 5 / 11) ∧ 
  (f > 9 / 20) ∧ 
  (f > 23 / 47) ∧ 
  (f > 205 / 409) := 
by
  sorry

end largest_fraction_among_list_l109_109725


namespace solve_inequality_l109_109674

theorem solve_inequality:
  ∀ x: ℝ, 0 ≤ x → (2021 * (real.rpow (x ^ 2020) (1 / 202)) - 1 ≥ 2020 * x) ↔ (x = 1) := by
sorry

end solve_inequality_l109_109674


namespace probability_x_gt_3y_l109_109658

/--
Prove that if a point (x, y) is randomly picked from the rectangular region with vertices
at (0,0), (2010,0), (2010,2011), and (0,2011), the probability that x > 3y is 335/2011.
-/
theorem probability_x_gt_3y (x y : ℝ) :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 2010) ∧ (0 ≤ y ∧ y ≤ 2011) ∧ True -> 
  sorry :=
begin
  sorry
end

end probability_x_gt_3y_l109_109658


namespace probability_of_x_greater_than_3y_l109_109660

noncomputable def area_triangle (base height : ℝ) : ℝ := 0.5 * base * height
noncomputable def area_rectangle (width height : ℝ) : ℝ := width * height
noncomputable def probability (area_triangle area_rectangle : ℝ) : ℝ := area_triangle / area_rectangle

theorem probability_of_x_greater_than_3y :
  let base := 2010
  let height_triangle := base / 3
  let height_rectangle := 2011
  let area_triangle := area_triangle base height_triangle
  let area_rectangle := area_rectangle base height_rectangle
  let prob := probability area_triangle area_rectangle
  prob = (335 : ℝ) / 2011 :=
by
  sorry

end probability_of_x_greater_than_3y_l109_109660


namespace students_with_all_three_pets_l109_109129

variables (TotalStudents HaveDogs HaveCats HaveOtherPets NoPets x y z w : ℕ)

theorem students_with_all_three_pets :
  TotalStudents = 40 →
  HaveDogs = 20 →
  HaveCats = 16 →
  HaveOtherPets = 8 →
  NoPets = 7 →
  x = 12 →
  y = 3 →
  z = 11 →
  TotalStudents - NoPets = 33 →
  x + y + w = HaveDogs →
  z + w = HaveCats →
  y + w = HaveOtherPets →
  x + y + z + w = 33 →
  w = 5 :=
by
  intros h1 h2 h3 h4 h5 hx hy hz h6 h7 h8 h9
  sorry

end students_with_all_three_pets_l109_109129


namespace hyperbola_asymptote_distance_l109_109107

section
open Function Real

variables (O P : ℝ × ℝ) (C : ℝ × ℝ → Prop) (M : ℝ × ℝ)
          (dist_asymptote : ℝ)

-- Conditions
def is_origin (O : ℝ × ℝ) : Prop := O = (0, 0)
def on_hyperbola (P : ℝ × ℝ) : Prop := P.1 ^ 2 / 9 - P.2 ^ 2 / 16 = 1
def unit_circle (M : ℝ × ℝ) : Prop := sqrt (M.1 ^ 2 + M.2 ^ 2) = 1
def orthogonal (O M P : ℝ × ℝ) : Prop := O.1 * P.1 + O.2 * P.2 = 0
def min_PM (dist : ℝ) : Prop := dist = 1 -- The minimum distance when |PM| is minimized

-- Proof problem
theorem hyperbola_asymptote_distance :
  is_origin O → 
  on_hyperbola P → 
  unit_circle M → 
  orthogonal O M P → 
  min_PM (sqrt ((P.1 - M.1) ^ 2 + (P.2 - M.2) ^ 2)) → 
  dist_asymptote = 12 / 5 :=
sorry
end

end hyperbola_asymptote_distance_l109_109107


namespace second_polygon_num_sides_l109_109048

theorem second_polygon_num_sides 
  (s : ℝ) 
  (h1: s ≠ 0) 
  (perimeter_equal: ∀ (side1_len side2_len : ℝ),
    50 * (3 * side1_len) = side2_len * (50 * 3 * side1_len / side1_len)) : 
  ∃ n : ℕ, n = 150 :=
by {
  use 150,
  sorry
}

end second_polygon_num_sides_l109_109048


namespace total_sugar_l109_109060

theorem total_sugar (sugar_frosting sugar_cake : ℝ) (h1 : sugar_frosting = 0.6) (h2 : sugar_cake = 0.2) :
  sugar_frosting + sugar_cake = 0.8 :=
by {
  -- The proof goes here
  sorry
}

end total_sugar_l109_109060


namespace find_remaining_rectangle_area_l109_109413

-- Definitions of given areas
def S_DEIH : ℝ := 20
def S_HILK : ℝ := 40
def S_ABHG : ℝ := 126
def S_GHKJ : ℝ := 63
def S_DFMK : ℝ := 161

-- Definition of areas of the remaining rectangle
def S_EFML : ℝ := 101

-- Theorem statement to prove the area of the remaining rectangle
theorem find_remaining_rectangle_area :
  S_DFMK - S_DEIH - S_HILK = S_EFML :=
by
  -- This is where the proof would go
  sorry

end find_remaining_rectangle_area_l109_109413


namespace nonneg_for_all_x_iff_a_in_range_l109_109085

def f (x a : ℝ) : ℝ := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

theorem nonneg_for_all_x_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 0) ↔ -2 ≤ a ∧ a ≤ 1 :=
by
  sorry

end nonneg_for_all_x_iff_a_in_range_l109_109085


namespace find_k_of_division_property_l109_109622

theorem find_k_of_division_property (k : ℝ) :
  (3 * (1 / 3)^3 - k * (1 / 3)^2 + 4) % (3 * (1 / 3) - 1) = 5 → k = -8 :=
by sorry

end find_k_of_division_property_l109_109622


namespace calculate_expression_l109_109758

-- Definitions based on conditions
def step1 : Int := 12 - (-18)
def step2 : Int := step1 + (-7)
def final_result : Int := 23

-- Theorem to prove
theorem calculate_expression : step2 = final_result := by
  have h1 : step1 = 12 + 18 := by sorry
  have h2 : step2 = step1 - 7 := by sorry
  rw [h1, h2]
  norm_num
  sorry

end calculate_expression_l109_109758


namespace simplify_expression_l109_109671

theorem simplify_expression
  (cos : ℝ → ℝ)
  (sin : ℝ → ℝ)
  (cos_double_angle : ∀ x, cos (2 * x) = cos x * cos x - sin x * sin x)
  (sin_double_angle : ∀ x, sin (2 * x) = 2 * sin x * cos x)
  (sin_cofunction : ∀ x, sin (Real.pi / 2 - x) = cos x) :
  (cos 5 * cos 5 - sin 5 * sin 5) / (sin 40 * cos 40) = 2 := by
  sorry

end simplify_expression_l109_109671


namespace min_val_z_is_7_l109_109549

noncomputable def min_val_z (x y : ℝ) (h : x + 3 * y = 2) : ℝ := 3^x + 27^y + 1

theorem min_val_z_is_7  : ∃ x y : ℝ, x + 3 * y = 2 ∧ min_val_z x y (by sorry) = 7 := sorry

end min_val_z_is_7_l109_109549


namespace general_term_formula_l109_109784

def Sn (a_n : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a_n n - 2^(n + 1)

theorem general_term_formula (a_n : ℕ → ℕ) (h : ∀ n : ℕ, n > 0 → Sn a_n n = (2 * a_n n - 2^(n + 1))) :
  ∀ n : ℕ, n > 0 → a_n n = (n + 1) * 2^n :=
sorry

end general_term_formula_l109_109784


namespace problem_statement_l109_109479

-- Define the given condition
def cond_1 (x : ℝ) := x + 1/x = 5

-- State the theorem that needs to be proven
theorem problem_statement (x : ℝ) (h : cond_1 x) : x^3 + 1/x^3 = 110 :=
sorry

end problem_statement_l109_109479


namespace combination_sum_l109_109077

theorem combination_sum :
  (Nat.choose 7 4) + (Nat.choose 7 3) = 70 := by
  sorry

end combination_sum_l109_109077


namespace length_of_adult_bed_is_20_decimeters_l109_109533

-- Define the length of an adult bed as per question context
def length_of_adult_bed := 20

-- Prove that the length of an adult bed in decimeters equals 20
theorem length_of_adult_bed_is_20_decimeters : length_of_adult_bed = 20 :=
by
  -- Proof goes here
  sorry

end length_of_adult_bed_is_20_decimeters_l109_109533


namespace average_class_size_l109_109377

theorem average_class_size 
  (num_3_year_olds : ℕ) 
  (num_4_year_olds : ℕ) 
  (num_5_year_olds : ℕ) 
  (num_6_year_olds : ℕ) 
  (class_size_3_and_4 : num_3_year_olds = 13 ∧ num_4_year_olds = 20) 
  (class_size_5_and_6 : num_5_year_olds = 15 ∧ num_6_year_olds = 22) :
  (num_3_year_olds + num_4_year_olds + num_5_year_olds + num_6_year_olds) / 2 = 35 :=
by
  sorry

end average_class_size_l109_109377


namespace algebraic_expression_transformation_l109_109487

theorem algebraic_expression_transformation (a b : ℝ) (h : ∀ x : ℝ, x^2 - 6*x + b = (x - a)^2 - 1) : b - a = 5 :=
by
  sorry

end algebraic_expression_transformation_l109_109487


namespace angle_rotation_acute_l109_109534

theorem angle_rotation_acute (angle_ACB : ℝ) (h : angle_ACB = 50) : 
  let new_angle := (angle_ACB + 540) % 360 - 180 in 
  if new_angle < 0 then new_angle + 360 else new_angle = 50 :=
by
  -- Proof goes here
  sorry

end angle_rotation_acute_l109_109534


namespace power_mod_444_444_l109_109699

open Nat

theorem power_mod_444_444 : (444:ℕ)^444 % 13 = 1 := by
  sorry

end power_mod_444_444_l109_109699


namespace tangent_lines_through_point_l109_109772

theorem tangent_lines_through_point :
  ∃ k : ℚ, ((5  * k - 12 * (36 - k * 2) + 36 = 0) ∨ (2 = 0)) := sorry

end tangent_lines_through_point_l109_109772


namespace solve_inequality_l109_109822

-- Define the inequality
def inequality (x : ℝ) : Prop :=
  -3 * (x^2 - 4 * x + 16) * (x^2 + 6 * x + 8) / ((x^3 + 64) * (Real.sqrt (x^2 + 4 * x + 4))) ≤ x^2 + x - 3

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  x ∈ Set.Iic (-4) ∪ {x : ℝ | -4 < x ∧ x ≤ -3} ∪ {x : ℝ | -2 < x ∧ x ≤ -1} ∪ Set.Ici 0

-- The theorem statement, which we need to prove
theorem solve_inequality : ∀ x : ℝ, inequality x ↔ solution_set x :=
by
  intro x
  sorry

end solve_inequality_l109_109822


namespace second_polygon_sides_l109_109052

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0)
  (perimeter_eq : ∀ (p1 p2 : ℝ), p1 = p2)
  (first_sides : ℕ) (second_sides : ℕ)
  (first_polygon_side_length : ℝ) (second_polygon_side_length : ℝ)
  (first_sides_eq : first_sides = 50)
  (side_length_relation : first_polygon_side_length = 3 * second_polygon_side_length)
  (perimeter_relation : first_sides * first_polygon_side_length = second_sides * second_polygon_side_length) :
  second_sides = 150 := by
  sorry

end second_polygon_sides_l109_109052


namespace radius_of_larger_circle_is_25_over_3_l109_109984

noncomputable def radius_of_larger_circle (r : ℝ) : ℝ := (5 / 2) * r 

theorem radius_of_larger_circle_is_25_over_3
  (rAB rBD : ℝ)
  (h_ratio : 2 * rBD = 5 * rBD / 2)
  (h_ab : rAB = 8)
  (h_tangent : ∀ rBD, (5 * rBD / 2 - 8) ^ 2 = 64 + rBD ^ 2) :
  radius_of_larger_circle (10 / 3) = 25 / 3 :=
  by
  sorry

end radius_of_larger_circle_is_25_over_3_l109_109984


namespace number_of_buses_used_l109_109675

-- Definitions based on the conditions
def total_students : ℕ := 360
def students_per_bus : ℕ := 45

-- The theorem we need to prove
theorem number_of_buses_used : total_students / students_per_bus = 8 := 
by sorry

end number_of_buses_used_l109_109675


namespace ribbon_tape_length_l109_109807

theorem ribbon_tape_length
  (one_ribbon: ℝ)
  (remaining_cm: ℝ)
  (num_ribbons: ℕ)
  (total_used: ℝ)
  (remaining_meters: remaining_cm = 0.50)
  (ribbon_meter: one_ribbon = 0.84)
  (ribbons_made: num_ribbons = 10)
  (used_len: total_used = one_ribbon * num_ribbons):
  total_used + 0.50 = 8.9 :=
by
  sorry

end ribbon_tape_length_l109_109807


namespace solve_problem1_solve_problem2_l109_109523

-- Problem 1
theorem solve_problem1 (x : ℚ) : (3 * x - 1) ^ 2 = 9 ↔ x = 4 / 3 ∨ x = -2 / 3 := 
by sorry

-- Problem 2
theorem solve_problem2 (x : ℚ) : x * (2 * x - 4) = (2 - x) ^ 2 ↔ x = 2 ∨ x = -2 :=
by sorry

end solve_problem1_solve_problem2_l109_109523


namespace new_mean_after_adding_eleven_l109_109012

theorem new_mean_after_adding_eleven (nums : List ℝ) (h_len : nums.length = 15) (h_avg : (nums.sum / 15) = 40) :
  ((nums.map (λ x => x + 11)).sum / 15) = 51 := by
  sorry

end new_mean_after_adding_eleven_l109_109012


namespace valveOperationTime_l109_109563

theorem valveOperationTime (a b t : ℚ) (h1 : a * (1 / 10) + b * (1 / 15) + (t - a - b) * (1 / 6) = 1) (h2 : t = 7) : 
  t - a - b = 5 := 
by
  sorry

end valveOperationTime_l109_109563


namespace constant_expression_l109_109817

variable {x y m n : ℝ}

theorem constant_expression (hx : x^2 = 25) (hy : ∀ y : ℝ, (x + y) * (x - 2 * y) - m * y * (n * x - y) = 25) :
  m = 2 ∧ n = -1/2 ∧ (x = 5 ∨ x = -5) :=
by {
  sorry
}

end constant_expression_l109_109817


namespace count_terminating_decimals_l109_109906

theorem count_terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 500) : 
  (nat.floor (500 / 49) = 10) := 
by
  sorry

end count_terminating_decimals_l109_109906


namespace sequence_general_formula_l109_109105

theorem sequence_general_formula (a : ℕ → ℕ) 
  (h₁ : a 1 = 2)
  (h₂ : ∀ n, a (n + 1) = 2 * a n - 1) :
  ∀ n, a n = 1 + 2^(n - 1) := 
sorry

end sequence_general_formula_l109_109105


namespace percentage_increase_l109_109127

theorem percentage_increase (L : ℕ) (h1 : L + 450 = 1350) :
  (450 / L : ℚ) * 100 = 50 := by
  sorry

end percentage_increase_l109_109127


namespace value_in_half_dollars_percentage_l109_109553

theorem value_in_half_dollars_percentage (n h q : ℕ) (hn : n = 75) (hh : h = 40) (hq : q = 30) : 
  (h * 50 : ℕ) / (n * 5 + h * 50 + q * 25 : ℕ) * 100 = 64 := by
  sorry

end value_in_half_dollars_percentage_l109_109553


namespace product_of_numbers_l109_109179

theorem product_of_numbers (x y z n : ℕ) 
  (h1 : x + y + z = 150)
  (h2 : 7 * x = n)
  (h3 : y - 10 = n)
  (h4 : z + 10 = n) : x * y * z = 48000 := 
by 
  sorry

end product_of_numbers_l109_109179


namespace max_x2_plus_4y_plus_3_l109_109602

theorem max_x2_plus_4y_plus_3 
  (x y : ℝ) 
  (h : x^2 + y^2 = 1) : 
  x^2 + 4*y + 3 ≤ 7 := sorry

end max_x2_plus_4y_plus_3_l109_109602


namespace remaining_pages_l109_109170

theorem remaining_pages (total_pages : ℕ) (science_project_percentage : ℕ) (math_homework_pages : ℕ)
  (h1 : total_pages = 120)
  (h2 : science_project_percentage = 25) 
  (h3 : math_homework_pages = 10) : 
  total_pages - (total_pages * science_project_percentage / 100) - math_homework_pages = 80 := by
  sorry

end remaining_pages_l109_109170


namespace num_trains_encountered_l109_109071

noncomputable def train_travel_encounters : ℕ := 5

theorem num_trains_encountered (start_time : ℕ) (duration : ℕ) (daily_departure : ℕ) 
  (train_journey_duration : ℕ) (daily_start_interval : ℕ) 
  (end_time : ℕ) (number_encountered : ℕ) :
  (train_journey_duration = 3 * 24 * 60 + 30) → -- 3 days and 30 minutes in minutes
  (daily_start_interval = 24 * 60) →             -- interval between daily train starts (in minutes)
  (number_encountered = 5) :=
by
  sorry

end num_trains_encountered_l109_109071


namespace quadratic_real_roots_l109_109799

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k^2 * x^2 - (2 * k + 1) * x + 1 = 0 ∧ ∃ x2 : ℝ, k^2 * x2^2 - (2 * k + 1) * x2 + 1 = 0)
  ↔ (k ≥ -1/4 ∧ k ≠ 0) := 
by 
  sorry

end quadratic_real_roots_l109_109799


namespace each_child_receive_amount_l109_109409

def husband_weekly_contribution : ℕ := 335
def wife_weekly_contribution : ℕ := 225
def weeks_in_month : ℕ := 4
def months : ℕ := 6
def children : ℕ := 4

noncomputable def total_weekly_contribution : ℕ := husband_weekly_contribution + wife_weekly_contribution
noncomputable def total_savings : ℕ := total_weekly_contribution * (weeks_in_month * months)
noncomputable def half_savings : ℕ := total_savings / 2
noncomputable def amount_per_child : ℕ := half_savings / children

theorem each_child_receive_amount :
  amount_per_child = 1680 :=
by
  sorry

end each_child_receive_amount_l109_109409


namespace perimeter_is_27_l109_109036

-- Definitions of equilateral triangles and midpoints.
def is_equilateral (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def is_midpoint (M A B : Point) : Prop :=
  dist A M = dist M B ∧ dist A M + dist M B = dist A B

-- Given points A, B, C, H, I, J, K and distances.
axiom  A B C H I J K : Point
axiom  AB AC BC AH HC AI HI AK KI IJ JK : ℝ
axiom  hABC : is_equilateral A B C
axiom  hAHI : is_equilateral A H I
axiom  hIJK : is_equilateral I J K
axiom  hH : is_midpoint H A C
axiom  hK : is_midpoint K A I
axiom  hAB : dist A B = 6

-- Perimeter of figure ABCHIJK.
noncomputable def perimeter_ABCHIJK : ℝ :=
  dist A B + dist B C + dist C H + dist H I + dist I J + dist J K + dist K A

-- Proof statement.
theorem perimeter_is_27 : perimeter_ABCHIJK = 27 :=
  sorry

end perimeter_is_27_l109_109036


namespace digit_150th_of_17_div_70_is_7_l109_109204

noncomputable def decimal_representation (n d : ℚ) : ℚ := n / d

-- We define the repeating part of the decimal representation.
def repeating_part : ℕ := 242857

-- Define the function to get the nth digit of the repeating part.
def get_nth_digit_of_repeating (n : ℕ) : ℕ :=
  let sequence := [2, 4, 2, 8, 5, 7]
  sequence.get! (n % sequence.length)

theorem digit_150th_of_17_div_70_is_7 : get_nth_digit_of_repeating 149 = 7 :=
by
  sorry

end digit_150th_of_17_div_70_is_7_l109_109204


namespace find_perimeter_ABCD_l109_109958

open Real

def RhombusInscribedInRectangle (ABCD : Type) :=
  ∃ (P Q R S : Point) (PB BQ PR QS : ℝ),
  (PB = 15) ∧ (BQ = 20) ∧ (PR = 30) ∧ (QS = 40) ∧
  inscribed_rhombus PQRS ABCD ABC PQRS PQ 30 40 ABCD contains_rect ABCD.

theorem find_perimeter_ABCD :
  ∃ m n : ℕ, coprime m n ∧ m + n = 677 :=
by
  sorry

end find_perimeter_ABCD_l109_109958


namespace distinct_positive_roots_log_sum_eq_5_l109_109096

theorem distinct_positive_roots_log_sum_eq_5 (a b : ℝ)
  (h : ∀ (x : ℝ), (8 * x ^ 3 + 6 * a * x ^ 2 + 3 * b * x + a = 0) → x > 0) 
  (h_sum : ∀ u v w : ℝ, (8 * u ^ 3 + 6 * a * u ^ 2 + 3 * b * u + a = 0) ∧
                       (8 * v ^ 3 + 6 * a * v ^ 2 + 3 * b * v + a = 0) ∧
                       (8 * w ^ 3 + 6 * a * w ^ 2 + 3 * b * w + a = 0) → 
                       u ≠ v ∧ u ≠ w ∧ v ≠ w ∧ 
                       (Real.log (u) / Real.log (3) + Real.log (v) / Real.log (3) + Real.log (w) / Real.log (3) = 5)) :
  a = -1944 :=
sorry

end distinct_positive_roots_log_sum_eq_5_l109_109096


namespace nes_sale_price_l109_109187

noncomputable def price_of_nes
    (snes_value : ℝ)
    (tradein_rate : ℝ)
    (cash_given : ℝ)
    (change_received : ℝ)
    (game_value : ℝ) : ℝ :=
  let tradein_credit := snes_value * tradein_rate
  let additional_cost := cash_given - change_received
  let total_cost := tradein_credit + additional_cost
  let nes_price := total_cost - game_value
  nes_price

theorem nes_sale_price 
  (snes_value : ℝ)
  (tradein_rate : ℝ)
  (cash_given : ℝ)
  (change_received : ℝ)
  (game_value : ℝ) :
  snes_value = 150 → tradein_rate = 0.80 → cash_given = 80 → change_received = 10 → game_value = 30 →
  price_of_nes snes_value tradein_rate cash_given change_received game_value = 160 := by
  intros
  sorry

end nes_sale_price_l109_109187


namespace number_of_squares_and_cubes_less_than_1000_l109_109312

noncomputable def count_squares_and_cubes : ℕ :=
  (List.range 1000).countp (λ n : ℕ, (∃ k : ℕ, k^6 = n))

theorem number_of_squares_and_cubes_less_than_1000 :
  count_squares_and_cubes = 3 :=
  by
    sorry

end number_of_squares_and_cubes_less_than_1000_l109_109312


namespace find_marks_in_biology_l109_109884

/-- 
David's marks in various subjects and his average marks are given.
This statement proves David's marks in Biology assuming the conditions provided.
--/
theorem find_marks_in_biology
  (english : ℕ) (math : ℕ) (physics : ℕ) (chemistry : ℕ) (avg_marks : ℕ)
  (total_subjects : ℕ)
  (h_english : english = 91)
  (h_math : math = 65)
  (h_physics : physics = 82)
  (h_chemistry : chemistry = 67)
  (h_avg_marks : avg_marks = 78)
  (h_total_subjects : total_subjects = 5)
  : ∃ (biology : ℕ), biology = 85 := 
by
  sorry

end find_marks_in_biology_l109_109884


namespace percentage_of_copper_buttons_l109_109738

-- Definitions for conditions
def total_items : ℕ := 100
def pin_percentage : ℕ := 30
def button_percentage : ℕ := 100 - pin_percentage
def brass_button_percentage : ℕ := 60
def copper_button_percentage : ℕ := 100 - brass_button_percentage

-- Theorem statement proving the question
theorem percentage_of_copper_buttons (h1 : pin_percentage = 30)
  (h2 : button_percentage = total_items - pin_percentage)
  (h3 : brass_button_percentage = 60)
  (h4 : copper_button_percentage = total_items - brass_button_percentage) :
  (button_percentage * copper_button_percentage) / total_items = 28 := 
sorry

end percentage_of_copper_buttons_l109_109738


namespace dodecahedron_has_150_interior_diagonals_l109_109115

def dodecahedron_diagonals (vertices : ℕ) (adjacent : ℕ) : ℕ :=
  let total := vertices * (vertices - adjacent - 1) / 2
  total

theorem dodecahedron_has_150_interior_diagonals :
  dodecahedron_diagonals 20 4 = 150 :=
by
  sorry

end dodecahedron_has_150_interior_diagonals_l109_109115


namespace digit_150_is_7_l109_109191

theorem digit_150_is_7 : (0.242857242857242857 : ℚ) = (17/70 : ℚ) ∧ ( (17/70).decimal 150 = 7) := by
  sorry

end digit_150_is_7_l109_109191


namespace num_real_solutions_l109_109796

theorem num_real_solutions (x : ℝ) (A B : Set ℝ) (hx : x ∈ A) (hx2 : x^2 ∈ A) :
  A = {0, 1, 2, x} → B = {1, x^2} → A ∪ B = A → 
  ∃! y : ℝ, y = -Real.sqrt 2 ∨ y = Real.sqrt 2 :=
by
  intro hA hB hA_union_B
  sorry

end num_real_solutions_l109_109796


namespace rope_length_l109_109157

theorem rope_length (h1 : ∃ x : ℝ, 4 * x = 20) : 
  ∃ l : ℝ, l = 35 := by
sorry

end rope_length_l109_109157


namespace rate_per_meter_l109_109894

theorem rate_per_meter (d : ℝ) (total_cost : ℝ) (rate_per_meter : ℝ) (h_d : d = 30)
    (h_total_cost : total_cost = 188.49555921538757) :
    rate_per_meter = 2 :=
by
  sorry

end rate_per_meter_l109_109894


namespace price_per_glass_second_day_l109_109731

theorem price_per_glass_second_day (O : ℝ) (P : ℝ) 
  (V1 : ℝ := 2 * O) -- Volume on the first day
  (V2 : ℝ := 3 * O) -- Volume on the second day
  (price_first_day : ℝ := 0.30) -- Price per glass on the first day
  (revenue_equal : V1 * price_first_day = V2 * P) :
  P = 0.20 := 
by
  -- skipping the proof
  sorry

end price_per_glass_second_day_l109_109731


namespace solve_for_y_l109_109478

-- Given conditions expressed as a Lean definition
def given_condition (y : ℝ) : Prop :=
  (y / 5) / 3 = 15 / (y / 3)

-- Prove the equivalent statement
theorem solve_for_y (y : ℝ) (h : given_condition y) : y = 15 * Real.sqrt 3 ∨ y = -15 * Real.sqrt 3 :=
sorry

end solve_for_y_l109_109478


namespace base_unit_digit_l109_109898

def unit_digit (n : ℕ) : ℕ := n % 10

theorem base_unit_digit (x : ℕ) :
  unit_digit ((x^41) * (41^14) * (14^87) * (87^76)) = 4 →
  unit_digit x = 1 :=
by
  sorry

end base_unit_digit_l109_109898


namespace solve_for_x_l109_109483

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 * x + 7 * x = 120) : x = 20 :=
by
  sorry

end solve_for_x_l109_109483


namespace completing_square_solution_l109_109366

theorem completing_square_solution (x : ℝ) : x^2 + 4 * x - 1 = 0 → (x + 2)^2 = 5 :=
by
  sorry

end completing_square_solution_l109_109366


namespace all_points_same_value_l109_109878

theorem all_points_same_value {f : ℤ × ℤ → ℕ}
  (h : ∀ x y : ℤ, f (x, y) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4) :
  ∃ k : ℕ, ∀ x y : ℤ, f (x, y) = k :=
sorry

end all_points_same_value_l109_109878


namespace expected_value_binomial_l109_109472

-- Define the parameters for the binomial distribution
def X : Binomial 6 (1/2)

-- Statement of the theorem
theorem expected_value_binomial : E(X) = 3 := by
  sorry

end expected_value_binomial_l109_109472


namespace smallest_x_for_three_digit_product_l109_109975

theorem smallest_x_for_three_digit_product : ∃ x : ℕ, (27 * x >= 100) ∧ (∀ y < x, 27 * y < 100) :=
by
  sorry

end smallest_x_for_three_digit_product_l109_109975


namespace remainder_444_444_mod_13_l109_109718

theorem remainder_444_444_mod_13 :
  444 ≡ 3 [MOD 13] →
  3^3 ≡ 1 [MOD 13] →
  444^444 ≡ 1 [MOD 13] := by
  intros h1 h2
  sorry

end remainder_444_444_mod_13_l109_109718


namespace digit_150_is_7_l109_109192

theorem digit_150_is_7 : (0.242857242857242857 : ℚ) = (17/70 : ℚ) ∧ ( (17/70).decimal 150 = 7) := by
  sorry

end digit_150_is_7_l109_109192


namespace overall_average_marks_l109_109940

theorem overall_average_marks (n P : ℕ) (P_avg F_avg : ℕ) (H_n : n = 120) (H_P : P = 100) (H_P_avg : P_avg = 39) (H_F_avg : F_avg = 15) :
  (P_avg * P + F_avg * (n - P)) / n = 35 := 
by
  sorry

end overall_average_marks_l109_109940


namespace theta_quadrant_l109_109793

theorem theta_quadrant (θ : ℝ) (h : Real.sin (2 * θ) < 0) : 
  (Real.sin θ < 0 ∧ Real.cos θ > 0) ∨ (Real.sin θ > 0 ∧ Real.cos θ < 0) :=
sorry

end theta_quadrant_l109_109793


namespace domain_f_2x_minus_1_l109_109926

theorem domain_f_2x_minus_1 (f : ℝ → ℝ) : 
  (∀ x, 1 ≤ x ∧ x ≤ 2 → 2 ≤ x + 1 ∧ x + 1 ≤ 3) → 
  (∀ z, 2 ≤ 2 * z - 1 ∧ 2 * z - 1 ≤ 3 → ∃ x, 3/2 ≤ x ∧ x ≤ 2 ∧ 2 * x - 1 = z) := 
sorry

end domain_f_2x_minus_1_l109_109926


namespace correct_answer_is_C_l109_109131

structure Point where
  x : ℤ
  y : ℤ

def inSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

def A : Point := ⟨1, -1⟩
def B : Point := ⟨0, 2⟩
def C : Point := ⟨-3, 2⟩
def D : Point := ⟨4, 0⟩

theorem correct_answer_is_C : inSecondQuadrant C := sorry

end correct_answer_is_C_l109_109131


namespace man_speed_l109_109064

theorem man_speed (rest_time_per_km : ℕ := 5) (total_km_covered : ℕ := 5) (total_time_min : ℕ := 50) : 
  (total_time_min - rest_time_per_km * (total_km_covered - 1)) / 60 * total_km_covered = 10 := by
  sorry

end man_speed_l109_109064


namespace messages_on_monday_l109_109737

theorem messages_on_monday (M : ℕ) (h0 : 200 + 500 + 1000 = 1700) (h1 : M + 1700 = 2000) : M = 300 :=
by
  -- Maths proof step here
  sorry

end messages_on_monday_l109_109737


namespace remainder_444_pow_444_mod_13_l109_109705

theorem remainder_444_pow_444_mod_13 :
  (444 ^ 444) % 13 = 1 :=
by
  have h1 : 444 % 13 = 2 := by norm_num
  have h2 : 2 ^ 12 % 13 = 1 := by norm_num
  rw [←pow_mul, ←nat.mul_div_cancel_left 444 12] at h1
  have h3 : 444 = 12 * (444 / 12) := nat.mul_div_cancel_left 444 12
  rw h3
  rw pow_mul
  rw h2
  rw pow_one
  exact h2

end remainder_444_pow_444_mod_13_l109_109705


namespace terminating_fraction_count_l109_109903

theorem terminating_fraction_count :
  (∃ n_values : Finset ℕ, (∀ n ∈ n_values, 1 ≤ n ∧ n ≤ 500 ∧ (∃ k : ℕ, n = k * 49)) ∧ n_values.card = 10) :=
by
  -- Placeholder for the proof, does not contribute to the conditions-direct definitions.
  sorry

end terminating_fraction_count_l109_109903


namespace fuel_remaining_l109_109292

-- Definitions given in the conditions of the original problem
def initial_fuel : ℕ := 48
def fuel_consumption_rate : ℕ := 8

-- Lean 4 statement of the mathematical proof problem
theorem fuel_remaining (x : ℕ) : 
  ∃ y : ℕ, y = initial_fuel - fuel_consumption_rate * x :=
sorry

end fuel_remaining_l109_109292


namespace digit_150th_of_17_div_70_is_7_l109_109201

noncomputable def decimal_representation (n d : ℚ) : ℚ := n / d

-- We define the repeating part of the decimal representation.
def repeating_part : ℕ := 242857

-- Define the function to get the nth digit of the repeating part.
def get_nth_digit_of_repeating (n : ℕ) : ℕ :=
  let sequence := [2, 4, 2, 8, 5, 7]
  sequence.get! (n % sequence.length)

theorem digit_150th_of_17_div_70_is_7 : get_nth_digit_of_repeating 149 = 7 :=
by
  sorry

end digit_150th_of_17_div_70_is_7_l109_109201


namespace intersection_of_sets_l109_109930

def set_M : Set ℝ := { x | x >= 2 }
def set_N : Set ℝ := { x | -1 <= x ∧ x <= 3 }
def set_intersection : Set ℝ := { x | 2 <= x ∧ x <= 3 }

theorem intersection_of_sets : (set_M ∩ set_N) = set_intersection := by
  sorry

end intersection_of_sets_l109_109930


namespace minimum_value_of_expression_l109_109920

noncomputable def monotonic_function_property
    (f : ℝ → ℝ)
    (h_monotonic : ∀ x y, (x ≤ y → f x ≤ f y) ∨ (x ≥ y → f x ≥ f y))
    (h_additive : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂)
    (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : f a + f (2 * b - 1) = 0): Prop :=
    (1 : ℝ) / a + 8 / b = 25

theorem minimum_value_of_expression 
    (f : ℝ → ℝ)
    (h_monotonic : ∀ x y, (x ≤ y → f x ≤ f y) ∨ (x ≥ y → f x ≥ f y))
    (h_additive : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂)
    (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : f a + f (2 * b - 1) = 0) :
    (1 : ℝ) / a + 8 / b = 25 := 
sorry

end minimum_value_of_expression_l109_109920


namespace triangle_inequalities_l109_109400

theorem triangle_inequalities (a b c : ℝ) :
  (∀ n : ℕ, a^n + b^n > c^n ∧ a^n + c^n > b^n ∧ b^n + c^n > a^n) →
  (a = b ∧ a > c) ∨ (a = b ∧ b = c) :=
by
  sorry

end triangle_inequalities_l109_109400


namespace solve_quadratic1_solve_quadratic2_l109_109155

theorem solve_quadratic1 :
  (∀ x, x^2 + x - 4 = 0 → x = ( -1 + Real.sqrt 17 ) / 2 ∨ x = ( -1 - Real.sqrt 17 ) / 2) := sorry

theorem solve_quadratic2 :
  (∀ x, (2*x + 1)^2 + 15 = 8*(2*x + 1) → x = 1 ∨ x = 2) := sorry

end solve_quadratic1_solve_quadratic2_l109_109155


namespace roots_of_z4_plus_16_eq_0_l109_109090

noncomputable def roots_of_quartic_eq : Set ℂ :=
  { z | z^4 + 16 = 0 }

theorem roots_of_z4_plus_16_eq_0 :
  roots_of_quartic_eq = { z | z = complex.sqrt 2 + complex.I * complex.sqrt 2 ∨
                             z = -complex.sqrt 2 + complex.I * complex.sqrt 2 ∨
                             z = -complex.sqrt 2 - complex.I * complex.sqrt 2 ∨
                             z = complex.sqrt 2 - complex.I * complex.sqrt 2 } :=
by
  sorry

end roots_of_z4_plus_16_eq_0_l109_109090


namespace shortest_distance_proof_l109_109497

noncomputable def shortest_distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

theorem shortest_distance_proof : 
  let A : ℝ × ℝ := (0, 250)
  let B : ℝ × ℝ := (800, 1050)
  shortest_distance A B = 1131 :=
by
  sorry

end shortest_distance_proof_l109_109497


namespace projection_is_correct_l109_109776

variables {α : Type*} [inner_product_space ℝ α] 
variables (a e : α) (θ : ℝ)

noncomputable def projection_vector (a e : α) : α :=
  (‖a‖ * real.cos θ) • (e / ‖e‖)

theorem projection_is_correct
  (h₁ : ‖a‖ = 2)
  (h₂ : ‖e‖ = 1)
  (h₃ : θ = 3 * real.pi / 4) :
  projection_vector a e = -real.sqrt 2 • e := 
sorry

end projection_is_correct_l109_109776


namespace arithmetic_sequence_sum_l109_109806

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ)     -- arithmetic sequence
  (d : ℝ)         -- common difference
  (h: ∀ n, a (n + 1) = a n + d)     -- definition of arithmetic sequence
  (h_sum : a 2 + a 4 + a 5 + a 6 + a 8 = 25) : 
  a 2 + a 8 = 10 := 
  sorry

end arithmetic_sequence_sum_l109_109806


namespace solve_for_x_l109_109482

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 * x + 7 * x = 120) : x = 20 := 
by
  sorry

end solve_for_x_l109_109482


namespace digit_150th_l109_109206

-- Define the fraction and its repeating decimal properties
def fraction := 17 / 70
def repeating_block := "242857"
def block_length : ℕ := 6

-- The 150th digit calculation
def digit_position : ℕ := 150 % block_length

theorem digit_150th : "150th digit of decimal representation" = 7 :=
by {
  have h1 : fraction = 0.242857142857, sorry,
  have h2 : "252857".cycle = repeating_block, sorry,
  have h3 : digit_position = 0, sorry,
  have h4 : repeating_block.get_last = '7', sorry,
  show "150th digit of decimal representation"= 7, sorry,
}

end digit_150th_l109_109206


namespace homework_done_l109_109148

theorem homework_done :
  ∃ (D E C Z M : Prop),
    -- Statements of students
    (¬ D ∧ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) ∧
    -- Truth-telling condition
    ((D → D ∧ ¬ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) ∧
    (E → ¬ D ∧ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) ∧
    (C → ¬ D ∧ ¬ E ∧ C ∧ ¬ Z ∧ ¬ M) ∧
    (Z → ¬ D ∧ ¬ E ∧ ¬ C ∧ Z ∧ ¬ M) ∧
    (M → ¬ D ∧ ¬ E ∧ ¬ C ∧ ¬ Z ∧ M)) ∧
    -- Number of students who did their homework condition
    (¬ D ∧ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) := 
sorry

end homework_done_l109_109148


namespace remainder_of_multiple_of_n_mod_7_l109_109628

theorem remainder_of_multiple_of_n_mod_7
  (n m : ℤ)
  (h1 : n % 7 = 1)
  (h2 : m % 7 = 3) :
  (m * n) % 7 = 3 :=
by
  sorry

end remainder_of_multiple_of_n_mod_7_l109_109628


namespace monthly_income_of_p_l109_109058

theorem monthly_income_of_p (P Q R : ℕ) 
    (h1 : (P + Q) / 2 = 5050)
    (h2 : (Q + R) / 2 = 6250)
    (h3 : (P + R) / 2 = 5200) :
    P = 4000 :=
by
  -- proof would go here
  sorry

end monthly_income_of_p_l109_109058


namespace trapezoid_sides_and_height_l109_109331

def trapezoid_base_height (a h A: ℝ) :=
  (h = (2 * a + 3) / 2) ∧
  (A = a^2 + 3 * a + 9 / 4) ∧
  (A = 2 * a^2 - 7.75)

theorem trapezoid_sides_and_height :
  ∃ (a b h : ℝ), (b = a + 3) ∧
  trapezoid_base_height a h 7.75 ∧
  a = 5 ∧ b = 8 ∧ h = 6.5 :=
by
  sorry

end trapezoid_sides_and_height_l109_109331


namespace unique_line_intercept_l109_109334

noncomputable def is_positive_integer (n : ℕ) : Prop := n > 0
noncomputable def is_prime (n : ℕ) : Prop := n = 2 ∨ (n > 2 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem unique_line_intercept (a b : ℕ) :
  ((is_positive_integer a) ∧ (is_prime b) ∧ (6 * b + 5 * a = a * b)) ↔ (a = 11 ∧ b = 11) :=
by
  sorry

end unique_line_intercept_l109_109334


namespace simplify_expression_l109_109358

noncomputable def simplify_fraction (x : ℝ) (h : x ≠ 2) : ℝ :=
  (1 + (1 / (x - 2))) / ((x - x^2) / (x - 2))

theorem simplify_expression (x : ℝ) (h : x ≠ 2) : simplify_fraction x h = -(x - 1) / x :=
  sorry

end simplify_expression_l109_109358


namespace river_trip_longer_than_lake_trip_l109_109262

theorem river_trip_longer_than_lake_trip (v w : ℝ) (h1 : v > w) : 
  (20 * v) / (v^2 - w^2) > 20 / v :=
by {
  sorry
}

end river_trip_longer_than_lake_trip_l109_109262


namespace probability_absolute_difference_l109_109355

noncomputable def fair_coin_flip : MeasureSpace ℝ := sorry
noncomputable def uniform_random_variable : MeasureSpace ℝ := sorry

variables (x y : ℝ) (hx : x ∈ set.Icc 0 1) (hy : y ∈ set.Icc 0 1)

theorem probability_absolute_difference :
  (MeasureTheory.Measure.prob (λ x y, |x - y| > 1/3) 
    [fair_coin_flip, uniform_random_variable, uniform_random_variable, fair_coin_flip]) = 5/9 :=
sorry

end probability_absolute_difference_l109_109355


namespace cubic_no_negative_roots_l109_109432

noncomputable def cubic_eq (x : ℝ) : ℝ := x^3 - 9 * x^2 + 23 * x - 15

theorem cubic_no_negative_roots {x : ℝ} : cubic_eq x = 0 → 0 ≤ x := sorry

end cubic_no_negative_roots_l109_109432


namespace arithmetic_sequence_sum_l109_109293

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℤ) (n : ℕ)
  (h₁ : ∀ n : ℕ, a n = a 1 + (n - 1) * d)
  (h₂ : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h₃ : 3 * a 5 - a 1 = 10) :
  S 13 = 117 := 
sorry

end arithmetic_sequence_sum_l109_109293


namespace count_bijections_with_fixed_points_l109_109592

def setA : Finset ℕ := {1, 2, 3, 4, 5}

noncomputable def fixed_points_bijections (f: {g: ℕ → ℕ // ∀ a ∈ setA, g a ∈ setA ∧ Function.Bijective g}) : ℕ :=
(setA.filter (λ x, f.1 x = x)).card

theorem count_bijections_with_fixed_points :
  (Finset.filter (λ f: {g: ℕ → ℕ // ∀ a ∈ setA, g a ∈ setA ∧ Function.Bijective g}, fixed_points_bijections f = 2)
     (Finset.univ : Finset {g: ℕ → ℕ // ∀ a ∈ setA, g a ∈ setA ∧ Function.Bijective g})).card = 20 := 
sorry

end count_bijections_with_fixed_points_l109_109592


namespace functional_equation_solution_l109_109892

noncomputable def function_nat_nat (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, f (x + y) = f x + f y

theorem functional_equation_solution :
  ∀ f : ℕ → ℕ, function_nat_nat f → ∃ a : ℕ, ∀ x : ℕ, f x = a * x :=
by
  sorry

end functional_equation_solution_l109_109892


namespace boyfriend_picks_up_correct_l109_109754

-- Define the initial condition
def init_pieces : ℕ := 60

-- Define the amount swept by Anne
def swept_pieces (n : ℕ) : ℕ := n / 2

-- Define the number of pieces stolen by the cat
def stolen_pieces : ℕ := 3

-- Define the remaining pieces after the cat steals
def remaining_pieces (n : ℕ) : ℕ := n - stolen_pieces

-- Define how many pieces the boyfriend picks up
def boyfriend_picks_up (n : ℕ) : ℕ := n / 3

-- The main theorem
theorem boyfriend_picks_up_correct : boyfriend_picks_up (remaining_pieces (init_pieces - swept_pieces init_pieces)) = 9 :=
by
  sorry

end boyfriend_picks_up_correct_l109_109754


namespace Kims_final_score_l109_109373

def easy_points : ℕ := 2
def average_points : ℕ := 3
def hard_points : ℕ := 5
def expert_points : ℕ := 7

def easy_correct : ℕ := 6
def average_correct : ℕ := 2
def hard_correct : ℕ := 4
def expert_correct : ℕ := 3

def complex_problems_bonus : ℕ := 1
def complex_problems_solved : ℕ := 2

def penalty_per_incorrect : ℕ := 1
def easy_incorrect : ℕ := 1
def average_incorrect : ℕ := 2
def hard_incorrect : ℕ := 2
def expert_incorrect : ℕ := 3

theorem Kims_final_score : 
  (easy_correct * easy_points + 
   average_correct * average_points + 
   hard_correct * hard_points + 
   expert_correct * expert_points + 
   complex_problems_solved * complex_problems_bonus) - 
   (easy_incorrect * penalty_per_incorrect + 
    average_incorrect * penalty_per_incorrect + 
    hard_incorrect * penalty_per_incorrect + 
    expert_incorrect * penalty_per_incorrect) = 53 :=
by 
  sorry

end Kims_final_score_l109_109373


namespace yard_area_l109_109947

theorem yard_area (posts : Nat) (spacing : Real) (longer_factor : Nat) (shorter_side_posts longer_side_posts : Nat)
  (h1 : posts = 24)
  (h2 : spacing = 3)
  (h3 : longer_factor = 3)
  (h4 : 2 * (shorter_side_posts + longer_side_posts) = posts - 4)
  (h5 : longer_side_posts = 3 * shorter_side_posts + 2) :
  (spacing * (shorter_side_posts - 1)) * (spacing * (longer_side_posts - 1)) = 144 :=
by
  sorry

end yard_area_l109_109947


namespace range_of_ϕ_l109_109112

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := 2 * Real.sin (2 * x + ϕ) + 1

theorem range_of_ϕ (ϕ : ℝ) (h1 : abs ϕ ≤ Real.pi / 2) 
    (h2 : ∀ (x : ℝ), -Real.pi / 12 < x ∧ x < Real.pi / 3 → f x ϕ > 1) :
  Real.pi / 6 ≤ ϕ ∧ ϕ ≤ Real.pi / 3 :=
sorry

end range_of_ϕ_l109_109112


namespace count_distinct_four_digit_numbers_divisible_by_3_ending_in_45_l109_109931

/--
Prove that the total number of distinct four-digit numbers that end with 45 and 
are divisible by 3 is 27.
-/
theorem count_distinct_four_digit_numbers_divisible_by_3_ending_in_45 :
  ∃ n : ℕ, n = 27 ∧ 
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (0 ≤ b ∧ b ≤ 9) → 
  (∃ k : ℕ, a + b + 9 = 3 * k) → 
  (10 * (10 * a + b) + 45) = 1000 * a + 100 * b + 45 → 
  1000 * a + 100 * b + 45 = n := sorry

end count_distinct_four_digit_numbers_divisible_by_3_ending_in_45_l109_109931


namespace factorization_a_squared_minus_3a_l109_109438

theorem factorization_a_squared_minus_3a (a : ℝ) : a^2 - 3 * a = a * (a - 3) := 
by 
  sorry

end factorization_a_squared_minus_3a_l109_109438


namespace inequality_of_abc_l109_109342

variable (a b c : ℝ)

theorem inequality_of_abc (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c ≥ a * b * c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * a * b * c :=
sorry

end inequality_of_abc_l109_109342


namespace remainder_444_444_mod_13_l109_109717

theorem remainder_444_444_mod_13 :
  444 ≡ 3 [MOD 13] →
  3^3 ≡ 1 [MOD 13] →
  444^444 ≡ 1 [MOD 13] := by
  intros h1 h2
  sorry

end remainder_444_444_mod_13_l109_109717


namespace total_splash_width_l109_109543

theorem total_splash_width :
  let pebble_splash := 1 / 4
  let rock_splash := 1 / 2
  let boulder_splash := 2
  let pebbles := 6
  let rocks := 3
  let boulders := 2
  let total_pebble_splash := pebbles * pebble_splash
  let total_rock_splash := rocks * rock_splash
  let total_boulder_splash := boulders * boulder_splash
  let total_splash := total_pebble_splash + total_rock_splash + total_boulder_splash
  total_splash = 7 := by
  sorry

end total_splash_width_l109_109543


namespace complement_union_l109_109601

open Set

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

-- Define the complement relative to U
def complement (A B : Set ℕ) : Set ℕ := { x ∈ B | x ∉ A }

-- The theorem we need to prove
theorem complement_union :
  complement (M ∪ N) U = {4} :=
by
  sorry

end complement_union_l109_109601


namespace total_pieces_10_rows_l109_109079

-- Define the conditions for the rods
def rod_seq (n : ℕ) : ℕ := 3 * n

-- Define the sum of the arithmetic sequence for rods
def sum_rods (n : ℕ) : ℕ := 3 * (n * (n + 1)) / 2

-- Define the conditions for the connectors
def connector_seq (n : ℕ) : ℕ := n + 1

-- Define the sum of the arithmetic sequence for connectors
def sum_connectors (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- Define the total pieces calculation
def total_pieces (n : ℕ) : ℕ := sum_rods n + sum_connectors (n + 1)

-- The target statement
theorem total_pieces_10_rows : total_pieces 10 = 231 :=
by
  sorry

end total_pieces_10_rows_l109_109079


namespace number_of_sixth_powers_lt_1000_l109_109311

theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | ∃ k : ℕ, k^6 = n ∧ n < 1000}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l109_109311


namespace sequence_general_formula_l109_109006

theorem sequence_general_formula (a : ℕ → ℝ) (h1 : a 1 = 3) (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 3 * a n - 4) :
  ∀ n : ℕ, n ≥ 1 → a n = 3^(n - 1) + 2 :=
sorry

end sequence_general_formula_l109_109006


namespace remainder_444_444_mod_13_l109_109720

theorem remainder_444_444_mod_13 :
  444 ≡ 3 [MOD 13] →
  3^3 ≡ 1 [MOD 13] →
  444^444 ≡ 1 [MOD 13] := by
  intros h1 h2
  sorry

end remainder_444_444_mod_13_l109_109720


namespace fermats_little_theorem_l109_109150

theorem fermats_little_theorem (p : ℕ) (hp : Nat.Prime p) (a : ℤ) : (a^p - a) % p = 0 := 
by sorry

end fermats_little_theorem_l109_109150


namespace equation_of_the_line_l109_109562

theorem equation_of_the_line (a b : ℝ) :
    ((a - b = 5) ∧ (9 / a + 4 / b = 1)) → 
    ( (2 * 9 + 3 * 4 - 30 = 0) ∨ (2 * 9 - 3 * 4 - 6 = 0) ∨ (9 - 4 - 5 = 0)) :=
  by
    sorry

end equation_of_the_line_l109_109562


namespace simple_interest_rate_l109_109504

theorem simple_interest_rate :
  ∀ (P T F : ℝ), P = 1000 → T = 3 → F = 1300 → (F - P) = P * 0.1 * T :=
by
  intros P T F hP hT hF
  sorry

end simple_interest_rate_l109_109504


namespace p_or_q_is_false_implies_p_and_q_is_false_l109_109401

theorem p_or_q_is_false_implies_p_and_q_is_false (p q : Prop) :
  (¬ (p ∨ q) → ¬ (p ∧ q)) ∧ ((¬ (p ∧ q) → (p ∨ q ∨ ¬ (p ∨ q)))) := sorry

end p_or_q_is_false_implies_p_and_q_is_false_l109_109401


namespace jill_tax_on_other_items_l109_109647

-- Define the conditions based on the problem statement.
variables (C : ℝ) (x : ℝ)
def tax_on_clothing := 0.04 * 0.60 * C
def tax_on_food := 0
def tax_on_other_items := 0.01 * x * 0.30 * C
def total_tax_paid := 0.048 * C

-- Prove the required percentage tax on other items.
theorem jill_tax_on_other_items :
  tax_on_clothing C + tax_on_food + tax_on_other_items C x = total_tax_paid C →
  x = 8 :=
by
  sorry

end jill_tax_on_other_items_l109_109647


namespace no_digit_satisfies_equations_l109_109232

-- Define the conditions as predicates.
def is_digit (x : ℤ) : Prop := 0 ≤ x ∧ x < 10

-- Formulate the proof problem based on the given problem conditions and conclusion
theorem no_digit_satisfies_equations : 
  ¬ (∃ x : ℤ, is_digit x ∧ (x - (10 * x + x) = 801 ∨ x - (10 * x + x) = 812)) :=
by
  sorry

end no_digit_satisfies_equations_l109_109232


namespace one_serving_weight_l109_109448

-- Outline the main variables
def chicken_weight_pounds : ℝ := 4.5
def stuffing_weight_ounces : ℝ := 24
def num_servings : ℝ := 12
def conversion_factor : ℝ := 16 -- 1 pound = 16 ounces

-- Define the weights in ounces
def chicken_weight_ounces : ℝ := chicken_weight_pounds * conversion_factor

-- Total weight in ounces for all servings
def total_weight_ounces : ℝ := chicken_weight_ounces + stuffing_weight_ounces

-- Prove one serving weight in ounces
theorem one_serving_weight : total_weight_ounces / num_servings = 8 := by
  sorry

end one_serving_weight_l109_109448


namespace system_of_equations_solution_l109_109369

theorem system_of_equations_solution (x y z : ℝ) 
  (h1 : x + 2 * y = 4)
  (h2 : 2 * x + 5 * y - 2 * z = 11)
  (h3 : 3 * x - 5 * y + 2 * z = -1) : 
  x = 2 ∧ y = 1 ∧ z = -1 :=
by {
  sorry
}

end system_of_equations_solution_l109_109369


namespace problem_a_range_l109_109579

theorem problem_a_range (a : ℝ) :
  (∀ x : ℝ, (a - 1) * x^2 - 2 * (a - 1) * x - 2 < 0) ↔ (-1 < a ∧ a ≤ 1) :=
by
  sorry

end problem_a_range_l109_109579


namespace determine_r_l109_109126

theorem determine_r (S : ℕ → ℤ) (r : ℤ) (n : ℕ) (h1 : 2 ≤ n) (h2 : ∀ k, S k = 2^k + r) : 
  r = -1 :=
sorry

end determine_r_l109_109126


namespace calculate_expression_l109_109755

theorem calculate_expression : 2^4 * 5^2 * 3^3 * 7 = 75600 := by
  sorry

end calculate_expression_l109_109755


namespace water_increase_factor_l109_109639

theorem water_increase_factor 
  (initial_koolaid : ℝ := 2) 
  (initial_water : ℝ := 16) 
  (evaporated_water : ℝ := 4) 
  (final_koolaid_percentage : ℝ := 4) : 
  (initial_water - evaporated_water) * (final_koolaid_percentage / 100) * initial_koolaid = 4 := 
by
  sorry

end water_increase_factor_l109_109639


namespace digit_150_of_17_div_70_l109_109214

theorem digit_150_of_17_div_70 : (decimal_digit 150 (17 / 70)) = 7 :=
sorry

end digit_150_of_17_div_70_l109_109214


namespace find_x_l109_109477

def F (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) : ℕ := a^b + c * d

theorem find_x (x : ℕ) : F 3 x 5 9 = 500 → x = 6 := 
by 
  sorry

end find_x_l109_109477


namespace power_mod_444_444_l109_109700

open Nat

theorem power_mod_444_444 : (444:ℕ)^444 % 13 = 1 := by
  sorry

end power_mod_444_444_l109_109700


namespace remainder_444_power_444_mod_13_l109_109709

theorem remainder_444_power_444_mod_13 : (444 ^ 444) % 13 = 1 := by
  sorry

end remainder_444_power_444_mod_13_l109_109709


namespace y_intercept_is_2_l109_109794

def equation_of_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

def point_P : ℝ × ℝ := (-1, 1)

def y_intercept_of_tangent_line (m c x y : ℝ) : Prop :=
  equation_of_circle x y ∧
  ((y = m * x + c) ∧ (point_P.1, point_P.2) ∈ {(x, y) | y = m * x + c})

theorem y_intercept_is_2 :
  ∃ m c : ℝ, y_intercept_of_tangent_line m c 0 2 :=
sorry

end y_intercept_is_2_l109_109794


namespace find_x_l109_109839

theorem find_x (x : ℝ) (h : 0.75 * x + 2 = 8) : x = 8 :=
sorry

end find_x_l109_109839


namespace total_pencils_l109_109397

theorem total_pencils (pencils_per_child : ℕ) (children : ℕ) (h1 : pencils_per_child = 2) (h2 : children = 9) : pencils_per_child * children = 18 :=
sorry

end total_pencils_l109_109397


namespace sqrt_fraction_evaluation_l109_109259

theorem sqrt_fraction_evaluation :
  (Real.sqrt ((2 / 25) + (1 / 49) - (1 / 100)) = 3 / 10) :=
by sorry

end sqrt_fraction_evaluation_l109_109259


namespace least_number_to_add_l109_109994

theorem least_number_to_add (x : ℕ) : (1021 + x) % 25 = 0 ↔ x = 4 := 
by 
  sorry

end least_number_to_add_l109_109994


namespace compute_9_times_one_seventh_pow_4_l109_109263

theorem compute_9_times_one_seventh_pow_4 :
  9 * (1 / 7) ^ 4 = 9 / 2401 :=
by
  -- The actual proof would go here.
  sorry

end compute_9_times_one_seventh_pow_4_l109_109263


namespace find_a_and_b_l109_109681

theorem find_a_and_b (a b : ℕ) :
  42 = a * 6 ∧ 72 = 6 * b ∧ 504 = 42 * 12 → (a, b) = (7, 12) :=
by
  sorry

end find_a_and_b_l109_109681


namespace decimal_to_fraction_equiv_l109_109227

theorem decimal_to_fraction_equiv : (0.38 : ℝ) = 19 / 50 :=
by
  sorry

end decimal_to_fraction_equiv_l109_109227


namespace ella_and_dog_food_l109_109002

theorem ella_and_dog_food (dog_food_per_pound_eaten_by_ella : ℕ) (ella_daily_food_intake : ℕ) (days : ℕ) : 
  dog_food_per_pound_eaten_by_ella = 4 →
  ella_daily_food_intake = 20 →
  days = 10 →
  (days * (ella_daily_food_intake + dog_food_per_pound_eaten_by_ella * ella_daily_food_intake)) = 1000 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end ella_and_dog_food_l109_109002


namespace find_x_l109_109922

def vec_a : ℝ × ℝ × ℝ := (-2, 1, 3)
def vec_b (x : ℝ) : ℝ × ℝ × ℝ := (1, x, -1)

theorem find_x (x : ℝ) (h : (-2) * 1 + 1 * x + 3 * (-1) = 0) : x = 5 :=
by
  sorry

end find_x_l109_109922


namespace polygon_area_correct_l109_109990

-- Define the coordinates of the vertices
def vertex1 := (2, 1)
def vertex2 := (4, 3)
def vertex3 := (6, 1)
def vertex4 := (4, 6)

-- Define a function to calculate the area using the Shoelace Theorem
noncomputable def shoelace_area (vertices : List (ℕ × ℕ)) : ℚ :=
  let xys := vertices ++ [vertices.head!]
  let sum1 := (xys.zip (xys.tail!)).map (fun ((x1, y1), (x2, y2)) => x1 * y2)
  let sum2 := (xys.zip (xys.tail!)).map (fun ((x1, y1), (x2, y2)) => y1 * x2)
  (sum1.sum - sum2.sum : ℚ) / 2

-- Instantiate the specific vertices
def polygon := [vertex1, vertex2, vertex3, vertex4]

-- The theorem statement
theorem polygon_area_correct : shoelace_area polygon = 6 := by
  sorry

end polygon_area_correct_l109_109990


namespace students_exceed_guinea_pigs_and_teachers_l109_109490

def num_students_per_classroom : Nat := 25
def num_guinea_pigs_per_classroom : Nat := 3
def num_teachers_per_classroom : Nat := 1
def num_classrooms : Nat := 5

def total_students : Nat := num_students_per_classroom * num_classrooms
def total_guinea_pigs : Nat := num_guinea_pigs_per_classroom * num_classrooms
def total_teachers : Nat := num_teachers_per_classroom * num_classrooms
def total_guinea_pigs_and_teachers : Nat := total_guinea_pigs + total_teachers

theorem students_exceed_guinea_pigs_and_teachers :
  total_students - total_guinea_pigs_and_teachers = 105 :=
by
  sorry

end students_exceed_guinea_pigs_and_teachers_l109_109490


namespace train_length_l109_109069

theorem train_length (V L : ℝ) 
  (h1 : L = V * 18) 
  (h2 : L + 600.0000000000001 = V * 54) : 
  L = 300.00000000000005 :=
by 
  sorry

end train_length_l109_109069


namespace probability_green_cube_l109_109239

/-- A box contains 36 pink, 18 blue, 9 green, 6 red, and 3 purple cubes that are identical in size.
    Prove that the probability that a randomly selected cube is green is 1/8. -/
theorem probability_green_cube :
  let pink_cubes := 36
  let blue_cubes := 18
  let green_cubes := 9
  let red_cubes := 6
  let purple_cubes := 3
  let total_cubes := pink_cubes + blue_cubes + green_cubes + red_cubes + purple_cubes
  let probability := (green_cubes : ℚ) / total_cubes
  probability = 1 / 8 := 
by
  sorry

end probability_green_cube_l109_109239


namespace fourth_roots_of_neg_16_l109_109094

theorem fourth_roots_of_neg_16 : 
  { z : ℂ | z^4 = -16 } = { sqrt (2 : ℂ) + sqrt (2 : ℂ) * complex.I, 
                            - sqrt (2 : ℂ) + sqrt (2 : ℂ) * complex.I, 
                            - sqrt (2 : ℂ) - sqrt (2 : ℂ) * complex.I, 
                            sqrt (2 : ℂ) - sqrt (2 : ℂ) * complex.I } :=
by
  sorry

end fourth_roots_of_neg_16_l109_109094


namespace age_problem_l109_109328

-- Defining the conditions and the proof problem
variables (B A : ℕ) -- B and A are natural numbers

-- Given conditions
def B_age : ℕ := 38
def A_age (B : ℕ) : ℕ := B + 8
def age_in_10_years (A : ℕ) : ℕ := A + 10
def years_ago (B : ℕ) (X : ℕ) : ℕ := B - X

-- Lean statement of the problem
theorem age_problem (X : ℕ) (hB : B = B_age) (hA : A = A_age B):
  age_in_10_years A = 2 * (years_ago B X) → X = 10 :=
by
  sorry

end age_problem_l109_109328


namespace g_negative_l109_109508

def g (a : ℚ) : ℚ := sorry

theorem g_negative {a b : ℚ} (h₁ : ∀ a b, g (a * b) = g a + g b)
                    (h₂ : ∀ p : ℚ, nat.prime p.natAbs → g p = p)
                    (x : ℚ) : 
                    x = 23/30 → g x < 0 :=
by
  intros hx
  sorry

end g_negative_l109_109508


namespace correct_number_for_question_mark_l109_109978

def first_row := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 200]
def second_row_no_quest := [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
def question_mark (x : ℕ) := first_row.sum = second_row_no_quest.sum + x

theorem correct_number_for_question_mark : question_mark 155 := 
by sorry -- proof to be completed

end correct_number_for_question_mark_l109_109978


namespace x_minus_p_eq_2_minus_2p_l109_109627

theorem x_minus_p_eq_2_minus_2p (x p : ℝ) (h1 : |x - 3| = p + 1) (h2 : x < 3) : x - p = 2 - 2 * p := 
sorry

end x_minus_p_eq_2_minus_2p_l109_109627


namespace partition_perfect_square_l109_109588

theorem partition_perfect_square (n : ℕ) (h : n ≥ 15) :
  ∀ A B : finset ℕ, disjoint A B → A ∪ B = finset.range (n + 1) →
  ∃ x y ∈ A ∨ ∃ x y ∈ B, x ≠ y ∧ (∃ k : ℕ, x + y = k^2) :=
begin
  sorry
end

end partition_perfect_square_l109_109588


namespace investment_compound_half_yearly_l109_109024

theorem investment_compound_half_yearly
  (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) (t : ℝ) 
  (h1 : P = 6000) 
  (h2 : r = 0.10) 
  (h3 : n = 2) 
  (h4 : A = 6615) :
  t = 1 :=
by
  sorry

end investment_compound_half_yearly_l109_109024


namespace no_positive_x_for_volume_l109_109233

noncomputable def volume (x : ℤ) : ℤ :=
  (x + 5) * (x - 7) * (x^2 + x + 30)

theorem no_positive_x_for_volume : ¬ ∃ x : ℕ, 0 < x ∧ volume x < 800 := by
  sorry

end no_positive_x_for_volume_l109_109233


namespace initial_pieces_l109_109596

-- Definitions of the conditions
def pieces_eaten : ℕ := 7
def pieces_given : ℕ := 21
def pieces_now : ℕ := 37

-- The proposition to prove
theorem initial_pieces (C : ℕ) (h : C - pieces_eaten + pieces_given = pieces_now) : C = 23 :=
by
  -- Proof would go here
  sorry

end initial_pieces_l109_109596


namespace tom_age_l109_109541

theorem tom_age (c : ℕ) (h1 : 2 * c - 1 = tom) (h2 : c + 3 = dave) (h3 : c + (2 * c - 1) + (c + 3) = 30) : tom = 13 :=
  sorry

end tom_age_l109_109541


namespace problem1_problem2_l109_109608

-- Definition of the function and given conditions for problem 1
def f (a b c x : ℝ) := (a * x^2 + b * x + c) * Real.exp x

-- Problem 1
theorem problem1 (a b c : ℝ) (h0 : f a b c 0 = 1) (h1 : f a b c 1 = 0) :
  (∀ x y ∈ set.Icc 0 1, x < y → f a b c x ≥ f a b c y) → a ∈ set.Icc 0 1 := sorry

-- Definition of the function and given conditions for problem 2
def f_a0 (b c x : ℝ) := (b * x + c) * Real.exp x

-- Problem 2
theorem problem2 (b c : ℝ) (m : ℝ)
  (h0 : f_a0 b c 0 = 1) (h1 : f_a0 b c 1 = 0)
  (h2 : ∀ x : ℝ, 2 * f_a0 b c x + 4 * x * Real.exp x ≥ m * x + 1 ∧ m * x + 1 ≥ -x^2 + 4*x + 1) :
  m = 4 := sorry

end problem1_problem2_l109_109608


namespace markup_percent_based_on_discounted_price_l109_109066

-- Defining the conditions
def original_price : ℝ := 1
def discount_percent : ℝ := 0.2
def discounted_price : ℝ := original_price * (1 - discount_percent)

-- The proof problem statement
theorem markup_percent_based_on_discounted_price :
  (original_price - discounted_price) / discounted_price = 0.25 :=
sorry

end markup_percent_based_on_discounted_price_l109_109066


namespace fruitseller_apples_l109_109247

theorem fruitseller_apples (x : ℝ) (sold_percent remaining_apples : ℝ) 
  (h_sold : sold_percent = 0.80) 
  (h_remaining : remaining_apples = 500) 
  (h_equation : (1 - sold_percent) * x = remaining_apples) : 
  x = 2500 := 
by 
  sorry

end fruitseller_apples_l109_109247


namespace smallest_number_of_pencils_l109_109548

theorem smallest_number_of_pencils
  (P : ℕ)
  (h5 : P % 5 = 2)
  (h9 : P % 9 = 2)
  (h11 : P % 11 = 2)
  (hP_gt2 : P > 2) :
  P = 497 :=
by
  sorry

end smallest_number_of_pencils_l109_109548


namespace win_probability_l109_109974

theorem win_probability (P_lose : ℚ) (h : P_lose = 5 / 8) : (1 - P_lose = 3 / 8) :=
by
  -- Provide the proof here if needed, but skip it
  sorry

end win_probability_l109_109974


namespace binomial_expansion_constant_term_l109_109325

theorem binomial_expansion_constant_term (x : ℝ) (n : ℕ) :
  (∃ c : ℝ, (3 * x^2 - (1 / (2 * x^3)))^5 = c ∧ c = 135 / 2) :=
by
  sorry

end binomial_expansion_constant_term_l109_109325


namespace dr_jones_remaining_salary_l109_109766

noncomputable def remaining_salary (salary rent food utilities insurances taxes transport emergency loan retirement : ℝ) : ℝ :=
  salary - (rent + food + utilities + insurances + taxes + transport + emergency + loan + retirement)

theorem dr_jones_remaining_salary :
  remaining_salary 6000 640 385 (1/4 * 6000) (1/5 * 6000) (0.10 * 6000) (0.03 * 6000) (0.02 * 6000) 300 (0.05 * 6000) = 1275 :=
by
  sorry

end dr_jones_remaining_salary_l109_109766


namespace probability_xy_odd_l109_109730

-- Definitions for x and y sets
def xSet := {1, 2, 3, 4}
def ySet := {5, 6, 7}

-- Definition of odd elements in the sets
def odd_of_set (s : Set ℕ) : Set ℕ := {n | n ∈ s ∧ n % 2 = 1}

-- Sizes of the respective sets
def size_xSet := (xSet : Set ℕ).card
def size_ySet := (ySet : Set ℕ).card
def size_odd_xSet := (odd_of_set xSet).card
def size_odd_ySet := (odd_of_set ySet).card

theorem probability_xy_odd :
  size_xSet = 4 ∧ size_ySet = 3 ->
  size_odd_xSet = 2 ∧ size_odd_ySet = 2 ->
  (size_odd_xSet * size_odd_ySet) / (size_xSet * size_ySet) = 1 / 3 := by
  sorry

end probability_xy_odd_l109_109730


namespace michelle_gas_left_l109_109815

def gasLeft (initialGas: ℝ) (usedGas: ℝ) : ℝ :=
  initialGas - usedGas

theorem michelle_gas_left :
  gasLeft 0.5 0.3333333333333333 = 0.1666666666666667 :=
by
  -- proof goes here
  sorry

end michelle_gas_left_l109_109815


namespace correct_divisor_l109_109494

-- Definitions of variables and conditions
variables (X D : ℕ)

-- Stating the theorem
theorem correct_divisor (h1 : X = 49 * 12) (h2 : X = 28 * D) : D = 21 :=
by
  sorry

end correct_divisor_l109_109494


namespace max_min_values_of_f_l109_109913

noncomputable def f (x : ℝ) : ℝ :=
  4^x - 2^(x+1) - 3

theorem max_min_values_of_f :
  ∀ x, 0 ≤ x ∧ x ≤ 2 → (∀ y, y = f x → y ≤ 5) ∧ (∃ y, y = f 2 ∧ y = 5) ∧ (∀ y, y = f x → y ≥ -4) ∧ (∃ y, y = f 0 ∧ y = -4) :=
by
  sorry

end max_min_values_of_f_l109_109913


namespace problem_solution_l109_109076

theorem problem_solution (x y : ℝ) (h₁ : (4 * y^2 + 1) * (x^4 + 2 * x^2 + 2) = 8 * |y| * (x^2 + 1))
  (h₂ : y ≠ 0) :
  (x = 0 ∧ (y = 1/2 ∨ y = -1/2)) :=
by {
  sorry -- Proof required
}

end problem_solution_l109_109076


namespace polygon_sides_l109_109065

theorem polygon_sides (side_length perimeter : ℕ) (h1 : side_length = 4) (h2 : perimeter = 24) : 
  perimeter / side_length = 6 :=
by 
  sorry

end polygon_sides_l109_109065


namespace probability_of_x_gt_3y_in_rectangle_is_335_over_2011_l109_109664

noncomputable def probability_x_gt_3y : ℚ :=
  let rectangle := ((0 : ℚ, 0 : ℚ), (2010 : ℚ), (2010 : ℚ, 2011 : ℚ), (0 : ℚ, 2011 : ℚ)) in
  let area_rectangle := 2010 * 2011 in
  let area_triangle := (1/2) * 2010 * 670 in
  area_triangle / area_rectangle

theorem probability_of_x_gt_3y_in_rectangle_is_335_over_2011 : probability_x_gt_3y = 335 / 2011 :=
begin
  sorry
end

end probability_of_x_gt_3y_in_rectangle_is_335_over_2011_l109_109664


namespace correct_statements_for_function_l109_109644

-- Definitions and the problem statement
def f (x b c : ℝ) := x * |x| + b * x + c

theorem correct_statements_for_function (b c : ℝ) :
  (c = 0 → ∀ x, f x b c = -f (-x) b c) ∧
  (b = 0 ∧ c > 0 → ∀ x, f x b c = 0 → x = 0) ∧
  (∀ x, f x b c = f (-x) b (-c)) :=
sorry

end correct_statements_for_function_l109_109644


namespace average_class_size_l109_109379

theorem average_class_size 
  (three_year_olds : ℕ := 13)
  (four_year_olds : ℕ := 20)
  (five_year_olds : ℕ := 15)
  (six_year_olds : ℕ := 22) : 
  ((three_year_olds + four_year_olds + five_year_olds + six_year_olds) / 2) = 35 := 
by
  sorry

end average_class_size_l109_109379


namespace bacteria_colony_growth_l109_109491

theorem bacteria_colony_growth (n : ℕ) : 
  (∀ m: ℕ, 4 * 3^m ≤ 500 → m < n) → n = 5 :=
by
  sorry

end bacteria_colony_growth_l109_109491


namespace asymptotes_of_hyperbola_min_focal_distance_l109_109895

theorem asymptotes_of_hyperbola_min_focal_distance :
  ∀ (x y m : ℝ),
  (m = 1 → 
   (∀ x y : ℝ, (x^2 / (m^2 + 8) - y^2 / (6 - 2 * m) = 1) → 
   (y = 2/3 * x ∨ y = -2/3 * x))) := 
  sorry

end asymptotes_of_hyperbola_min_focal_distance_l109_109895


namespace benches_count_l109_109825

theorem benches_count (num_people_base6 : ℕ) (people_per_bench : ℕ) (num_people_base10 : ℕ) (num_benches : ℕ) :
  num_people_base6 = 204 ∧ people_per_bench = 2 ∧ num_people_base10 = 76 ∧ num_benches = 38 →
  (num_people_base10 = 2 * 6^2 + 0 * 6^1 + 4 * 6^0) ∧
  (num_benches = num_people_base10 / people_per_bench) :=
by
  sorry

end benches_count_l109_109825


namespace number_of_sixth_powers_lt_1000_l109_109305

-- Define the condition for being a sixth power and being less than 1000
def isSixthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^6

-- Define the main statement to be proved
theorem number_of_sixth_powers_lt_1000 : 
  {n : ℕ | n < 1000 ∧ isSixthPower n}.to_finset.card = 3 :=
by
  sorry

end number_of_sixth_powers_lt_1000_l109_109305


namespace dozen_chocolate_bars_cost_l109_109834

theorem dozen_chocolate_bars_cost
  (cost_mag : ℕ → ℝ) (cost_choco_bar : ℕ → ℝ)
  (H1 : cost_mag 1 = 1)
  (H2 : 4 * (cost_choco_bar 1) = 8 * (cost_mag 1)) :
  12 * (cost_choco_bar 1) = 24 := 
sorry

end dozen_chocolate_bars_cost_l109_109834


namespace compass_legs_cannot_swap_l109_109833

-- Define the problem conditions: compass legs on infinite grid, constant distance d.
def on_grid (p q : ℤ × ℤ) : Prop := 
  ∃ d : ℕ, d * d = (p.1 - q.1) * (p.1 - q.1) + (p.2 - q.2) * (p.2 - q.2) ∧ d > 0

-- Define the main theorem as a Lean 4 statement
theorem compass_legs_cannot_swap (p q : ℤ × ℤ) (h : on_grid p q) : 
  ¬ ∃ r s : ℤ × ℤ, on_grid r p ∧ on_grid s p ∧ p ≠ q ∧ r = q ∧ s = p :=
sorry

end compass_legs_cannot_swap_l109_109833


namespace minimum_value_expression_l109_109053

theorem minimum_value_expression (x : ℝ) : ∃ y : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2050 = y ∧ ∀ z : ℝ, ((x + 1) * (x + 3) * (x + 5) * (x + 7) + 2050 ≥ z) ↔ (z = 2034) :=
by
  sorry

end minimum_value_expression_l109_109053


namespace digit_150_is_7_l109_109194

theorem digit_150_is_7 : (0.242857242857242857 : ℚ) = (17/70 : ℚ) ∧ ( (17/70).decimal 150 = 7) := by
  sorry

end digit_150_is_7_l109_109194


namespace sum_of_squares_eq_2_l109_109620

theorem sum_of_squares_eq_2 (a b : ℝ) 
  (h : (a^2 + b^2) * (a^2 + b^2 + 4) = 12) : a^2 + b^2 = 2 :=
by sorry

end sum_of_squares_eq_2_l109_109620


namespace problem_l109_109713

theorem problem (a : ℕ) (h1 : a = 444) : (444 ^ 444) % 13 = 1 :=
by
  have h444 : 444 % 13 = 3 := by sorry
  have h3_pow3 : 3 ^ 3 % 13 = 1 := by sorry
  sorry

end problem_l109_109713


namespace remainder_444_444_mod_13_l109_109719

theorem remainder_444_444_mod_13 :
  444 ≡ 3 [MOD 13] →
  3^3 ≡ 1 [MOD 13] →
  444^444 ≡ 1 [MOD 13] := by
  intros h1 h2
  sorry

end remainder_444_444_mod_13_l109_109719


namespace square_floor_tile_count_l109_109395

/-
A square floor is tiled with congruent square tiles.
The tiles on the two diagonals of the floor are black.
If there are 101 black tiles, then the total number of tiles is 2601.
-/
theorem square_floor_tile_count  
  (s : ℕ) 
  (hs_odd : s % 2 = 1)  -- s is odd
  (h_black_tile_count : 2 * s - 1 = 101) 
  : s^2 = 2601 := 
by 
  sorry

end square_floor_tile_count_l109_109395


namespace area_of_figure_l109_109576

theorem area_of_figure : 
  ∀ (x y : ℝ), |3 * x + 4| + |4 * y - 3| ≤ 12 → area_of_rhombus = 24 := 
by
  sorry

end area_of_figure_l109_109576


namespace least_common_multiple_of_812_and_3214_is_correct_l109_109845

def lcm_812_3214 : ℕ :=
  Nat.lcm 812 3214

theorem least_common_multiple_of_812_and_3214_is_correct :
  lcm_812_3214 = 1304124 := by
  sorry

end least_common_multiple_of_812_and_3214_is_correct_l109_109845


namespace average_class_size_l109_109380

theorem average_class_size 
  (three_year_olds : ℕ := 13)
  (four_year_olds : ℕ := 20)
  (five_year_olds : ℕ := 15)
  (six_year_olds : ℕ := 22) : 
  ((three_year_olds + four_year_olds + five_year_olds + six_year_olds) / 2) = 35 := 
by
  sorry

end average_class_size_l109_109380


namespace quadratic_reciprocal_sum_l109_109345

theorem quadratic_reciprocal_sum :
  ∃ (x1 x2 : ℝ), (x1^2 - 5 * x1 + 4 = 0) ∧ (x2^2 - 5 * x2 + 4 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2 = 5) ∧ (x1 * x2 = 4) ∧ (1 / x1 + 1 / x2 = 5 / 4) :=
sorry

end quadratic_reciprocal_sum_l109_109345


namespace p_sufficient_but_not_necessary_for_q_l109_109102

def condition_p (x : ℝ) : Prop := x^2 - 9 > 0
def condition_q (x : ℝ) : Prop := x^2 - (5 / 6) * x + (1 / 6) > 0

theorem p_sufficient_but_not_necessary_for_q :
  (∀ x, condition_p x → condition_q x) ∧ ¬(∀ x, condition_q x → condition_p x) :=
sorry

end p_sufficient_but_not_necessary_for_q_l109_109102


namespace remainder_444_power_444_mod_13_l109_109711

theorem remainder_444_power_444_mod_13 : (444 ^ 444) % 13 = 1 := by
  sorry

end remainder_444_power_444_mod_13_l109_109711


namespace abc_sum_l109_109679

theorem abc_sum (a b c : ℤ) 
  (h1 : ∀ x : ℤ, (x + a) * (x + b) = x^2 + 21 * x + 110)
  (h2 : ∀ x : ℤ, (x - b) * (x - c) = x^2 - 19 * x + 88) : 
  a + b + c = 29 := 
by
  sorry

end abc_sum_l109_109679


namespace metals_inductive_reasoning_l109_109536

def conducts_electricity (metal : String) : Prop :=
  metal = "Gold" ∨ metal = "Silver" ∨ metal = "Copper" ∨ metal = "Iron"

def all_metals_conduct_electricity (metals : List String) : Prop :=
  ∀ metal, metal ∈ metals → conducts_electricity metal

theorem metals_inductive_reasoning 
  (h1 : conducts_electricity "Gold")
  (h2 : conducts_electricity "Silver")
  (h3 : conducts_electricity "Copper")
  (h4 : conducts_electricity "Iron") :
  (all_metals_conduct_electricity ["Gold", "Silver", "Copper", "Iron"] → 
  all_metals_conduct_electricity ["All metals"]) :=
  sorry -- Proof skipped, as per instructions.

end metals_inductive_reasoning_l109_109536


namespace field_trip_students_l109_109527

theorem field_trip_students (bus_cost admission_per_student budget : ℕ) (students : ℕ)
  (h1 : bus_cost = 100)
  (h2 : admission_per_student = 10)
  (h3 : budget = 350)
  (total_cost : students * admission_per_student + bus_cost ≤ budget) : 
  students = 25 :=
by
  sorry

end field_trip_students_l109_109527


namespace probability_x_gt_3y_correct_l109_109652

noncomputable def probability_x_gt_3y : ℚ :=
  let rect_area := (2010 : ℚ) * 2011
  let triangle_area := (2010 : ℚ) * (2010 / 3) / 2
  (triangle_area / rect_area)

theorem probability_x_gt_3y_correct :
  probability_x_gt_3y = 670 / 2011 := 
by
  sorry

end probability_x_gt_3y_correct_l109_109652


namespace solve_xy_l109_109237

variable (x y : ℝ)

-- Given conditions
def condition1 : Prop := y = (2 / 3) * x
def condition2 : Prop := 0.4 * x = (1 / 3) * y + 110

-- Statement we want to prove
theorem solve_xy (h1 : condition1 x y) (h2 : condition2 x y) : x = 618.75 ∧ y = 412.5 :=
  by sorry

end solve_xy_l109_109237


namespace probability_x_greater_3y_in_rectangle_l109_109666

theorem probability_x_greater_3y_in_rectangle :
  let rect := set.Icc (0 : ℝ) 2010 ×ˢ set.Icc (0 : ℝ) 2011 in
  let event := {p : ℝ × ℝ | p.1 > 3 * p.2} in
  let area_rect := (2010 : ℝ) * 2011 in
  let area_event := (2010 : ℝ) * (2010/3) / 2 in
  (event ∩ rect).measure / rect.measure = 335 / 2011 :=
by
  -- The proof goes here
  sorry

end probability_x_greater_3y_in_rectangle_l109_109666


namespace inequality_solution_l109_109348

theorem inequality_solution (x : ℝ) (hx : x > 0) : 
  x + 1 / x ≥ 2 ∧ (x + 1 / x = 2 ↔ x = 1) := 
by
  sorry

end inequality_solution_l109_109348


namespace tan_2alpha_value_beta_value_l109_109282

variable (α β : ℝ)
variable (h1 : 0 < β ∧ β < α ∧ α < π / 2)
variable (h2 : Real.cos α = 1 / 7)
variable (h3 : Real.cos (α - β) = 13 / 14)

theorem tan_2alpha_value : Real.tan (2 * α) = - (8 * Real.sqrt 3 / 47) :=
by
  sorry

theorem beta_value : β = π / 3 :=
by
  sorry

end tan_2alpha_value_beta_value_l109_109282


namespace perp_line_parallel_plane_perp_line_l109_109923

variable {Line : Type} {Plane : Type}
variable (a b : Line) (α β : Plane)
variable (parallel : Line → Plane → Prop) (perpendicular : Line → Plane → Prop) (parallel_lines : Line → Line → Prop)

-- Conditions
variable (non_coincident_lines : ¬(a = b))
variable (non_coincident_planes : ¬(α = β))
variable (a_perp_α : perpendicular a α)
variable (b_par_α : parallel b α)

-- Prove
theorem perp_line_parallel_plane_perp_line :
  perpendicular a α ∧ parallel b α → parallel_lines a b :=
sorry

end perp_line_parallel_plane_perp_line_l109_109923


namespace range_of_m_l109_109610

-- Definitions of propositions p and q
def p (m : ℝ) : Prop := 
  (2 * m - 3)^2 - 4 > 0

def q (m : ℝ) : Prop := 
  2 * m > 3

-- Theorem statement
theorem range_of_m (m : ℝ) : ¬ (p m ∧ q m) ∧ (p m ∨ q m) ↔ (m < 1 / 2 ∨ 3 / 2 < m ∧ m ≤ 5 / 2) :=
  sorry

end range_of_m_l109_109610


namespace room_width_to_perimeter_ratio_l109_109566

theorem room_width_to_perimeter_ratio (L W : ℕ) (hL : L = 25) (hW : W = 15) :
  let P := 2 * (L + W)
  let ratio := W / P
  ratio = 3 / 16 :=
by
  sorry

end room_width_to_perimeter_ratio_l109_109566


namespace shorten_to_sixth_power_l109_109987

theorem shorten_to_sixth_power (x n m p q r : ℕ) (h1 : x > 1000000)
  (h2 : x / 10 = n^2)
  (h3 : n^2 / 10 = m^3)
  (h4 : m^3 / 10 = p^4)
  (h5 : p^4 / 10 = q^5) :
  q^5 / 10 = r^6 :=
sorry

end shorten_to_sixth_power_l109_109987


namespace roots_of_z4_plus_16_eq_0_l109_109089

noncomputable def roots_of_quartic_eq : Set ℂ :=
  { z | z^4 + 16 = 0 }

theorem roots_of_z4_plus_16_eq_0 :
  roots_of_quartic_eq = { z | z = complex.sqrt 2 + complex.I * complex.sqrt 2 ∨
                             z = -complex.sqrt 2 + complex.I * complex.sqrt 2 ∨
                             z = -complex.sqrt 2 - complex.I * complex.sqrt 2 ∨
                             z = complex.sqrt 2 - complex.I * complex.sqrt 2 } :=
by
  sorry

end roots_of_z4_plus_16_eq_0_l109_109089


namespace perpendicular_vectors_l109_109613

-- Define the vectors a and b.
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (x : ℝ) : ℝ × ℝ := (-2, x)

-- Define the dot product function.
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The condition that a is perpendicular to b.
def perp_condition (x : ℝ) : Prop :=
  dot_product vector_a (vector_b x) = 0

-- Main theorem stating that if a is perpendicular to b, then x = -1.
theorem perpendicular_vectors (x : ℝ) (h : perp_condition x) : x = -1 :=
by sorry

end perpendicular_vectors_l109_109613


namespace ratio_of_two_numbers_l109_109686

theorem ratio_of_two_numbers (a b : ℝ) (h1 : a + b = 7 * (a - b)) (h2 : a > b) (h3 : a > 0) (h4 : b > 0) :
  a / b = 4 / 3 :=
by
  sorry

end ratio_of_two_numbers_l109_109686


namespace slope_of_line_is_neg_one_l109_109289

theorem slope_of_line_is_neg_one (y : ℝ) (h : (y - 5) / (5 - (-3)) = -1) : y = -3 :=
by
  sorry

end slope_of_line_is_neg_one_l109_109289


namespace total_flour_l109_109814

theorem total_flour (original_flour extra_flour : Real) (h_orig : original_flour = 7.0) (h_extra : extra_flour = 2.0) : original_flour + extra_flour = 9.0 :=
sorry

end total_flour_l109_109814


namespace digit_150_of_17_div_70_is_2_l109_109223

def repeating_cycle_17_div_70 : List ℕ := [2, 4, 2, 8, 5, 7, 1]

theorem digit_150_of_17_div_70_is_2 : 
  (repeating_cycle_17_div_70[(150 % 7) - 1] = 2) :=
by
  -- the proof will go here
  sorry

end digit_150_of_17_div_70_is_2_l109_109223


namespace altered_solution_ratio_l109_109172

theorem altered_solution_ratio (initial_bleach : ℕ) (initial_detergent : ℕ) (initial_water : ℕ) :
  initial_bleach / initial_detergent = 2 / 25 ∧
  initial_detergent / initial_water = 25 / 100 →
  (initial_detergent / initial_water) / 2 = 1 / 8 →
  initial_water = 300 →
  (300 / 8) = 37.5 := 
by 
  sorry

end altered_solution_ratio_l109_109172


namespace remainder_444_power_444_mod_13_l109_109708

theorem remainder_444_power_444_mod_13 : (444 ^ 444) % 13 = 1 := by
  sorry

end remainder_444_power_444_mod_13_l109_109708


namespace paving_stone_width_l109_109982

theorem paving_stone_width :
  let courtyard_length := 70
  let courtyard_width := 16.5
  let num_paving_stones := 231
  let paving_stone_length := 2.5
  let courtyard_area := courtyard_length * courtyard_width
  let total_area_covered := courtyard_area
  let paving_stone_width := total_area_covered / (paving_stone_length * num_paving_stones)
  paving_stone_width = 2 :=
by
  sorry

end paving_stone_width_l109_109982


namespace joanna_estimate_is_larger_l109_109824

theorem joanna_estimate_is_larger 
  (u v ε₁ ε₂ : ℝ) 
  (huv : u > v) 
  (hv0 : v > 0) 
  (hε₁ : ε₁ > 0) 
  (hε₂ : ε₂ > 0) : 
  (u + ε₁) - (v - ε₂) > u - v := 
sorry

end joanna_estimate_is_larger_l109_109824


namespace time_to_cross_approx_l109_109853

-- Define train length, tunnel length, speed in km/hr, conversion factors, and the final equation
def length_of_train : ℕ := 415
def length_of_tunnel : ℕ := 285
def speed_in_kmph : ℕ := 63
def km_to_m : ℕ := 1000
def hr_to_sec : ℕ := 3600

-- Convert speed to m/s
def speed_in_mps : ℚ := (speed_in_kmph * km_to_m) / hr_to_sec

-- Calculate total distance
def total_distance : ℕ := length_of_train + length_of_tunnel

-- Calculate the time to cross the tunnel in seconds
def time_to_cross : ℚ := total_distance / speed_in_mps

theorem time_to_cross_approx : abs (time_to_cross - 40) < 0.1 :=
sorry

end time_to_cross_approx_l109_109853


namespace cassie_and_brian_meet_at_1111am_l109_109260

theorem cassie_and_brian_meet_at_1111am :
  ∃ t : ℕ, t = 11*60 + 11 ∧
    (∃ x : ℚ, x = 51/16 ∧ 
      14 * x + 18 * (x - 1) = 84) :=
sorry

end cassie_and_brian_meet_at_1111am_l109_109260


namespace sofa_love_seat_ratio_l109_109414

theorem sofa_love_seat_ratio (L S: ℕ) (h1: L = 148) (h2: S + L = 444): S = 2 * L := by
  sorry

end sofa_love_seat_ratio_l109_109414


namespace solve_garden_width_l109_109067

noncomputable def garden_width_problem (w l : ℕ) :=
  (w + l = 30) ∧ (w * l = 200) ∧ (l = w + 8) → w = 11

theorem solve_garden_width (w l : ℕ) : garden_width_problem w l :=
by
  intro h
  -- Omitting the actual proof
  sorry

end solve_garden_width_l109_109067


namespace smallest_solution_of_equation_l109_109391

theorem smallest_solution_of_equation :
  ∃ x : ℝ, (9 * x^2 - 45 * x + 50 = 0) ∧ (∀ y : ℝ, 9 * y^2 - 45 * y + 50 = 0 → x ≤ y) :=
sorry

end smallest_solution_of_equation_l109_109391


namespace meetings_percentage_l109_109813

/-- Define the total work day in hours -/
def total_work_day_hours : ℕ := 10

/-- Define the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 60 -- 1 hour = 60 minutes

/-- Define the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 75

/-- Define the break duration in minutes -/
def break_minutes : ℕ := 30

/-- Define the effective work minutes -/
def effective_work_minutes : ℕ := (total_work_day_hours * 60) - break_minutes

/-- Define the total meeting minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- The percentage of the effective work day spent in meetings -/
def percent_meetings : ℕ := (total_meeting_minutes * 100) / effective_work_minutes

theorem meetings_percentage : percent_meetings = 24 := by
  sorry

end meetings_percentage_l109_109813


namespace digit_150_is_7_l109_109193

theorem digit_150_is_7 : (0.242857242857242857 : ℚ) = (17/70 : ℚ) ∧ ( (17/70).decimal 150 = 7) := by
  sorry

end digit_150_is_7_l109_109193


namespace box_width_l109_109404

theorem box_width (W S : ℕ) (h1 : 30 * W * 12 = 80 * S^3) (h2 : S ∣ 30 ∧ S ∣ 12) : W = 48 :=
by
  sorry

end box_width_l109_109404


namespace increase_by_fraction_l109_109988

theorem increase_by_fraction (original_value : ℕ) (fraction : ℚ) : original_value = 120 → fraction = 5/6 → original_value + original_value * fraction = 220 :=
by
  intros h1 h2
  sorry

end increase_by_fraction_l109_109988


namespace cabbages_produced_l109_109063

theorem cabbages_produced (x y : ℕ) (h1 : y = x + 1) (h2 : x^2 + 199 = y^2) : y^2 = 10000 :=
by
  sorry

end cabbages_produced_l109_109063


namespace scientific_notation_of_investment_l109_109149

theorem scientific_notation_of_investment : 41800000000 = 4.18 * 10^10 := 
by
  sorry

end scientific_notation_of_investment_l109_109149


namespace probability_of_x_greater_than_3y_l109_109661

noncomputable def area_triangle (base height : ℝ) : ℝ := 0.5 * base * height
noncomputable def area_rectangle (width height : ℝ) : ℝ := width * height
noncomputable def probability (area_triangle area_rectangle : ℝ) : ℝ := area_triangle / area_rectangle

theorem probability_of_x_greater_than_3y :
  let base := 2010
  let height_triangle := base / 3
  let height_rectangle := 2011
  let area_triangle := area_triangle base height_triangle
  let area_rectangle := area_rectangle base height_rectangle
  let prob := probability area_triangle area_rectangle
  prob = (335 : ℝ) / 2011 :=
by
  sorry

end probability_of_x_greater_than_3y_l109_109661


namespace investment_change_l109_109017

theorem investment_change (x : ℝ) :
  (1 : ℝ) > (0 : ℝ) → 
  1.05 * x / x - 1 * 100 = 5 :=
by
  sorry

end investment_change_l109_109017


namespace digit_150_of_17_div_70_is_2_l109_109225

def repeating_cycle_17_div_70 : List ℕ := [2, 4, 2, 8, 5, 7, 1]

theorem digit_150_of_17_div_70_is_2 : 
  (repeating_cycle_17_div_70[(150 % 7) - 1] = 2) :=
by
  -- the proof will go here
  sorry

end digit_150_of_17_div_70_is_2_l109_109225


namespace sin_70_eq_1_minus_2k_squared_l109_109917

theorem sin_70_eq_1_minus_2k_squared (k : ℝ) (h : Real.sin (10 * Real.pi / 180) = k) :
  Real.sin (70 * Real.pi / 180) = 1 - 2 * k^2 :=
by
  sorry

end sin_70_eq_1_minus_2k_squared_l109_109917


namespace tennis_player_games_l109_109873

theorem tennis_player_games (b : ℕ → ℕ) (h1 : ∀ k, b k ≥ k) (h2 : ∀ k, b k ≤ 12 * (k / 7)) :
  ∃ i j : ℕ, i < j ∧ b j - b i = 20 :=
by
  sorry

end tennis_player_games_l109_109873


namespace incorrect_median_l109_109162

/-- 
Given:
- A stem-and-leaf plot representation.
- Player B's scores are mainly between 30 and 40 points.
- Player B has 13 scores.
Prove:
The judgment "The median score of player B is 28" is incorrect.
-/
theorem incorrect_median (scores : List ℕ) (H_len : scores.length = 13) (H_range : ∀ x ∈ scores, 30 ≤ x ∧ x ≤ 40) 
  (H_median : ∃ median, median = scores.nthLe 6 sorry ∧ median = 28) : False := 
sorry

end incorrect_median_l109_109162


namespace range_of_a_l109_109524

-- Definitions related to the conditions in the problem
def polynomial (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * x ^ 5 - 4 * a * x ^ 3 + 2 * b ^ 2 * x ^ 2 + 1

def v_2 (x : ℝ) (a : ℝ) : ℝ := (3 * x + 0) * x - 4 * a

def v_3 (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (((3 * x + 0) * x - 4 * a) * x + 2 * b ^ 2)

-- The main statement to prove
theorem range_of_a (x a b : ℝ) (h1 : x = 2) (h2 : ∀ b : ℝ, (v_2 x a) < (v_3 x a b)) : a < 3 :=
by
  sorry

end range_of_a_l109_109524


namespace digit_150_in_17_div_70_l109_109196

noncomputable def repeating_sequence_170 : List Nat := [2, 4, 2, 8, 5, 7]

theorem digit_150_in_17_div_70 : (repeating_sequence_170.get? (150 % 6 - 1) = some 7) := by
  sorry

end digit_150_in_17_div_70_l109_109196


namespace total_cost_correct_l109_109732

-- Definitions for the costs of items.
def sandwich_cost : ℝ := 3.49
def soda_cost : ℝ := 0.87

-- Definitions for the quantities.
def num_sandwiches : ℝ := 2
def num_sodas : ℝ := 4

-- The calculation for the total cost.
def total_cost : ℝ := (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)

-- The claim that needs to be proved.
theorem total_cost_correct : total_cost = 10.46 := by
  sorry

end total_cost_correct_l109_109732


namespace combined_average_age_of_fifth_graders_teachers_and_parents_l109_109180

theorem combined_average_age_of_fifth_graders_teachers_and_parents
  (num_fifth_graders : ℕ) (avg_age_fifth_graders : ℕ)
  (num_teachers : ℕ) (avg_age_teachers : ℕ)
  (num_parents : ℕ) (avg_age_parents : ℕ)
  (h1 : num_fifth_graders = 40) (h2 : avg_age_fifth_graders = 10)
  (h3 : num_teachers = 4) (h4 : avg_age_teachers = 40)
  (h5 : num_parents = 60) (h6 : avg_age_parents = 34)
  : (num_fifth_graders * avg_age_fifth_graders + num_teachers * avg_age_teachers + num_parents * avg_age_parents) /
    (num_fifth_graders + num_teachers + num_parents) = 25 :=
by sorry

end combined_average_age_of_fifth_graders_teachers_and_parents_l109_109180


namespace percentage_paid_to_X_l109_109389

theorem percentage_paid_to_X (X Y : ℝ) (h1 : X + Y = 880) (h2 : Y = 400) : 
  ((X / Y) * 100) = 120 :=
by
  sorry

end percentage_paid_to_X_l109_109389


namespace average_age_of_family_l109_109867

theorem average_age_of_family :
  let num_grandparents := 2
  let num_parents := 2
  let num_grandchildren := 3
  let avg_age_grandparents := 64
  let avg_age_parents := 39
  let avg_age_grandchildren := 6
  let total_age_grandparents := avg_age_grandparents * num_grandparents
  let total_age_parents := avg_age_parents * num_parents
  let total_age_grandchildren := avg_age_grandchildren * num_grandchildren
  let total_age_family := total_age_grandparents + total_age_parents + total_age_grandchildren
  let num_family_members := num_grandparents + num_parents + num_grandchildren
  let avg_age_family := total_age_family / num_family_members
  avg_age_family = 32 := 
  by 
  repeat { sorry }

end average_age_of_family_l109_109867


namespace matt_age_three_years_ago_l109_109865

theorem matt_age_three_years_ago (james_age_three_years_ago : ℕ) (age_difference : ℕ) (future_factor : ℕ) :
  james_age_three_years_ago = 27 →
  age_difference = 3 →
  future_factor = 2 →
  ∃ matt_age_now : ℕ,
  james_age_now: ℕ,
    james_age_now = james_age_three_years_ago + age_difference ∧
    (matt_age_now + 5) = future_factor * (james_age_now + 5) ∧
    matt_age_now = 65 :=
by
  sorry

end matt_age_three_years_ago_l109_109865


namespace common_real_solution_for_y_l109_109765

theorem common_real_solution_for_y :
  ∃ x : ℝ, x^2 + y^2 = 9 ∧ x^2 - 4*y + 4 = 0 ↔ (y = -4.44 ∨ y = -8.56) :=
by
  sorry -- Proof goes here

end common_real_solution_for_y_l109_109765


namespace nesbitt_inequality_l109_109519

theorem nesbitt_inequality {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) ∧ (a = b ∧ b = c → a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2) :=
sorry

end nesbitt_inequality_l109_109519


namespace find_vanilla_cookies_l109_109394

variable (V : ℕ)

def num_vanilla_cookies_sold (choc_cookies: ℕ) (vanilla_cookies: ℕ) (total_revenue: ℕ) : Prop :=
  choc_cookies * 1 + vanilla_cookies * 2 = total_revenue

theorem find_vanilla_cookies (h : num_vanilla_cookies_sold 220 V 360) : V = 70 :=
by
  sorry

end find_vanilla_cookies_l109_109394


namespace second_polygon_sides_l109_109051

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0)
  (perimeter_eq : ∀ (p1 p2 : ℝ), p1 = p2)
  (first_sides : ℕ) (second_sides : ℕ)
  (first_polygon_side_length : ℝ) (second_polygon_side_length : ℝ)
  (first_sides_eq : first_sides = 50)
  (side_length_relation : first_polygon_side_length = 3 * second_polygon_side_length)
  (perimeter_relation : first_sides * first_polygon_side_length = second_sides * second_polygon_side_length) :
  second_sides = 150 := by
  sorry

end second_polygon_sides_l109_109051


namespace cornbread_pieces_l109_109340

theorem cornbread_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_length : ℕ) (piece_width : ℕ)
  (hl : pan_length = 20) (hw : pan_width = 18) (hp : piece_length = 2) (hq : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 90 :=
by
  sorry

end cornbread_pieces_l109_109340


namespace second_polygon_sides_l109_109046

open Nat

theorem second_polygon_sides (s : ℝ) (same_perimeter : ℝ) :
  ∃ n : ℕ, n = 150 :=
by
  let side1 := 3 * s
  have perimeter1 : same_perimeter = 50 * side1 := rfl
  have perimeter2 : same_perimeter = n * s
  have sides : n = 150 :=
  sorry

end second_polygon_sides_l109_109046


namespace quadratic_range_l109_109295

theorem quadratic_range (x y : ℝ) (h1 : y = -(x - 5) ^ 2 + 1) (h2 : 2 < x ∧ x < 6) :
  -8 < y ∧ y ≤ 1 := 
sorry

end quadratic_range_l109_109295


namespace cost_per_book_eq_three_l109_109080

-- Let T be the total amount spent, B be the number of books, and C be the cost per book
variables (T B C : ℕ)
-- Conditions: Edward spent $6 (T = 6) to buy 2 books (B = 2)
-- Each book costs the same amount (C = T / B)
axiom total_amount : T = 6
axiom number_of_books : B = 2

-- We need to prove that each book cost $3
theorem cost_per_book_eq_three (h1 : T = 6) (h2 : B = 2) : (T / B) = 3 := by
  sorry

end cost_per_book_eq_three_l109_109080


namespace dividend_is_correct_l109_109689

-- Definitions of the given conditions.
def divisor : ℕ := 17
def quotient : ℕ := 4
def remainder : ℕ := 8

-- Define the dividend using the given formula.
def dividend : ℕ := (divisor * quotient) + remainder

-- The theorem to prove.
theorem dividend_is_correct : dividend = 76 := by
  -- The following line contains a placeholder for the actual proof.
  sorry

end dividend_is_correct_l109_109689


namespace avg_temp_l109_109026

theorem avg_temp (M T W Th F : ℝ) (h1 : M = 41) (h2 : F = 33) (h3 : (T + W + Th + F) / 4 = 46) : 
  (M + T + W + Th) / 4 = 48 :=
by
  -- insert proof steps here
  sorry

end avg_temp_l109_109026


namespace toothpick_count_l109_109985

theorem toothpick_count (length width : ℕ) (h_len : length = 20) (h_width : width = 10) : 
  2 * (length * (width + 1) + width * (length + 1)) = 430 :=
by
  sorry

end toothpick_count_l109_109985


namespace sister_sandcastle_height_l109_109879

theorem sister_sandcastle_height (miki_height : ℝ)
                                (height_diff : ℝ)
                                (h_miki : miki_height = 0.8333333333333334)
                                (h_diff : height_diff = 0.3333333333333333) :
  miki_height - height_diff = 0.5 :=
by
  sorry

end sister_sandcastle_height_l109_109879


namespace remaining_pages_l109_109169

theorem remaining_pages (total_pages : ℕ) (science_project_percentage : ℕ) (math_homework_pages : ℕ)
  (h1 : total_pages = 120)
  (h2 : science_project_percentage = 25) 
  (h3 : math_homework_pages = 10) : 
  total_pages - (total_pages * science_project_percentage / 100) - math_homework_pages = 80 := by
  sorry

end remaining_pages_l109_109169


namespace contractor_absent_days_l109_109062

noncomputable def solve_contractor_problem : Prop :=
  ∃ (x y : ℕ), 
    x + y = 30 ∧ 
    25 * x - 750 / 100 * y = 555 ∧
    y = 6

theorem contractor_absent_days : solve_contractor_problem :=
  sorry

end contractor_absent_days_l109_109062


namespace num_int_values_n_terminated_l109_109901

theorem num_int_values_n_terminated (N : ℕ) (hN1 : 1 ≤ N) (hN2 : N ≤ 500) :
  ∃ n : ℕ, n = 10 ∧ ∀ k, 0 ≤ k → k < n → ∃ (m : ℕ), N = m * 49 :=
sorry

end num_int_values_n_terminated_l109_109901


namespace slope_of_perpendicular_line_l109_109054

theorem slope_of_perpendicular_line 
  (x1 y1 x2 y2 : ℤ)
  (h : x1 = 3 ∧ y1 = -4 ∧ x2 = -6 ∧ y2 = 2) : 
∃ m : ℚ, m = 3/2 :=
by
  sorry

end slope_of_perpendicular_line_l109_109054


namespace alchemerion_age_problem_l109_109257

theorem alchemerion_age_problem 
  (A S F : ℕ)
  (h1 : A = 3 * S)
  (h2 : F = 2 * A + 40)
  (h3 : A = 360) :
  A + S + F = 1240 :=
by 
  sorry

end alchemerion_age_problem_l109_109257


namespace find_number_l109_109836

theorem find_number (x : ℝ) : 0.75 * x + 2 = 8 → x = 8 :=
by
  sorry

end find_number_l109_109836


namespace matt_age_three_years_ago_l109_109864

theorem matt_age_three_years_ago (james_age_three_years_ago : ℕ) (age_difference : ℕ) (future_factor : ℕ) :
  james_age_three_years_ago = 27 →
  age_difference = 3 →
  future_factor = 2 →
  ∃ matt_age_now : ℕ,
  james_age_now: ℕ,
    james_age_now = james_age_three_years_ago + age_difference ∧
    (matt_age_now + 5) = future_factor * (james_age_now + 5) ∧
    matt_age_now = 65 :=
by
  sorry

end matt_age_three_years_ago_l109_109864


namespace intersection_right_complement_l109_109921

open Set

def A := {x : ℝ | x - 1 ≥ 0}
def B := {x : ℝ | 3 / x ≤ 1}

theorem intersection_right_complement :
  A ∩ (compl B) = {x : ℝ | 1 ≤ x ∧ x < 3} :=
by
  sorry

end intersection_right_complement_l109_109921


namespace faye_pencils_l109_109584

theorem faye_pencils (rows : ℕ) (pencils_per_row : ℕ) (h_rows : rows = 30) (h_pencils_per_row : pencils_per_row = 24) :
  rows * pencils_per_row = 720 :=
by
  sorry

end faye_pencils_l109_109584


namespace general_equation_of_line_l109_109326

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 1)

-- Define what it means for a line to pass through two points
def line_through_points (l : ℝ → ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  l A.1 A.2 ∧ l B.1 B.2

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := x - 2 * y + 2 = 0

-- The theorem that needs to be proven
theorem general_equation_of_line : line_through_points line_l A B := 
by
  sorry

end general_equation_of_line_l109_109326


namespace find_h_l109_109621

theorem find_h (h : ℤ) (root_condition : (-3)^3 + h * (-3) - 18 = 0) : h = -15 :=
by
  sorry

end find_h_l109_109621


namespace find_A_l109_109550

theorem find_A (A : ℕ) (h : 10 * A + 2 - 23 = 549) : A = 5 :=
by sorry

end find_A_l109_109550


namespace Tonya_spent_on_brushes_l109_109542

section
variable (total_spent : ℝ)
variable (cost_canvases : ℝ)
variable (cost_paints : ℝ)
variable (cost_easel : ℝ)
variable (cost_brushes : ℝ)

def Tonya_total_spent : Prop := total_spent = 90.0
def Cost_of_canvases : Prop := cost_canvases = 40.0
def Cost_of_paints : Prop := cost_paints = cost_canvases / 2
def Cost_of_easel : Prop := cost_easel = 15.0
def Cost_of_brushes : Prop := cost_brushes = total_spent - (cost_canvases + cost_paints + cost_easel)

theorem Tonya_spent_on_brushes : Tonya_total_spent total_spent →
  Cost_of_canvases cost_canvases →
  Cost_of_paints cost_paints cost_canvases →
  Cost_of_easel cost_easel →
  Cost_of_brushes cost_brushes total_spent cost_canvases cost_paints cost_easel →
  cost_brushes = 15.0 := by
  intro h_total_spent h_cost_canvases h_cost_paints h_cost_easel h_cost_brushes
  rw [Tonya_total_spent, Cost_of_canvases, Cost_of_paints, Cost_of_easel, Cost_of_brushes] at *
  sorry
end

end Tonya_spent_on_brushes_l109_109542


namespace shop_owner_pricing_l109_109567

theorem shop_owner_pricing (L C M S : ℝ)
  (h1 : C = 0.75 * L)
  (h2 : S = 1.3 * C)
  (h3 : S = 0.75 * M) : 
  M = 1.3 * L := 
sorry

end shop_owner_pricing_l109_109567


namespace sum_all_products_eq_l109_109907

def group1 : List ℚ := [3/4, 3/20] -- Using 0.15 as 3/20 to work with rationals
def group2 : List ℚ := [4, 2/3]
def group3 : List ℚ := [3/5, 6/5] -- Using 1.2 as 6/5 to work with rationals

def allProducts (a b c : List ℚ) : List ℚ :=
  List.bind a (fun x =>
  List.bind b (fun y =>
  List.map (fun z => x * y * z) c))

theorem sum_all_products_eq :
  (allProducts group1 group2 group3).sum = 7.56 := by
  sorry

end sum_all_products_eq_l109_109907


namespace digit_150_in_17_div_70_l109_109198

noncomputable def repeating_sequence_170 : List Nat := [2, 4, 2, 8, 5, 7]

theorem digit_150_in_17_div_70 : (repeating_sequence_170.get? (150 % 6 - 1) = some 7) := by
  sorry

end digit_150_in_17_div_70_l109_109198


namespace min_value_of_m_l109_109950

noncomputable def g (x : ℝ) := (Real.exp x + Real.exp (-x)) / 2
noncomputable def h (x : ℝ) := (Real.exp (-x) - Real.exp x) / 2

theorem min_value_of_m (m : ℝ) : (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → m * g x + h x ≥ 0) → m ≥ (Real.exp 2 - 1) / (Real.exp 2 + 1) :=
by
  intro h
  have key_ineq : ∀ x, -1 ≤ x ∧ x ≤ 1 → m ≥ 1 - 2 / (Real.exp (2 * x) + 1) := sorry
  sorry

end min_value_of_m_l109_109950


namespace compute_9_times_one_seventh_pow_4_l109_109264

theorem compute_9_times_one_seventh_pow_4 :
  9 * (1 / 7) ^ 4 = 9 / 2401 :=
by
  -- The actual proof would go here.
  sorry

end compute_9_times_one_seventh_pow_4_l109_109264


namespace group_size_l109_109496

-- Define the conditions
variables (N : ℕ)
variable (h1 : (1 / 5 : ℝ) * N = (N : ℝ) * 0.20)
variable (h2 : 128 ≤ N)
variable (h3 : (1 / 5 : ℝ) * N - 128 = 0.04 * (N : ℝ))

-- Prove that the number of people in the group is 800
theorem group_size : N = 800 :=
by
  sorry

end group_size_l109_109496


namespace min_value_of_function_l109_109291

theorem min_value_of_function (x : ℝ) (h : x > 2) : ∃ y, y = (x^2 - 4*x + 8) / (x - 2) ∧ (∀ z, z = (x^2 - 4*x + 8) / (x - 2) → y ≤ z) :=
sorry

end min_value_of_function_l109_109291


namespace placemat_length_correct_l109_109254

noncomputable def placemat_length (r : ℝ) : ℝ :=
  2 * r * Real.sin (Real.pi / 8)

theorem placemat_length_correct (r : ℝ) (n : ℕ) (w : ℝ) (y : ℝ) (h_r : r = 5)
  (h_n : n = 8) (h_w : w = 1)
  (h_y : y = placemat_length r) :
  y = 5 * Real.sqrt (2 - Real.sqrt 2) :=
by
  sorry

end placemat_length_correct_l109_109254


namespace x_add_one_greater_than_x_l109_109726

theorem x_add_one_greater_than_x (x : ℝ) : x + 1 > x :=
by
  sorry

end x_add_one_greater_than_x_l109_109726


namespace toothpicks_in_arithmetic_sequence_l109_109078

theorem toothpicks_in_arithmetic_sequence :
  let a1 := 5
  let d := 3
  let n := 15
  let a_n n := a1 + (n - 1) * d
  let sum_to_n n := n * (2 * a1 + (n - 1) * d) / 2
  sum_to_n n = 390 := by
  sorry

end toothpicks_in_arithmetic_sequence_l109_109078


namespace solve_for_x_l109_109361

theorem solve_for_x : ∃ x : ℚ, (2/3 - 1/4) = 1/x ∧ x = 12/5 :=
by
  use 12/5
  split
  · norm_num
  · norm_num
  · sorry

end solve_for_x_l109_109361


namespace strip_width_l109_109870

theorem strip_width (w : ℝ) (h_floor : ℝ := 10) (b_floor : ℝ := 8) (area_rug : ℝ := 24) :
  (h_floor - 2 * w) * (b_floor - 2 * w) = area_rug → w = 2 := 
by 
  sorry

end strip_width_l109_109870


namespace number_of_functions_l109_109575

open Nat

theorem number_of_functions (f : Fin 15 → Fin 15)
  (h : ∀ x, (f (f x) - 2 * f x + x : Int) % 15 = 0) :
  ∃! n : Nat, n = 375 := sorry

end number_of_functions_l109_109575


namespace solve_for_x_l109_109364

theorem solve_for_x (x : ℚ) : (2/3 : ℚ) - 1/4 = 1/x → x = 12/5 := 
by
  sorry

end solve_for_x_l109_109364


namespace triangle_side_y_values_l109_109569

theorem triangle_side_y_values (y : ℕ) : (4 < y^2 ∧ y^2 < 20) ↔ (y = 3 ∨ y = 4) :=
by
  sorry

end triangle_side_y_values_l109_109569


namespace simplify_expression_l109_109963

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : 
  2 * (1 - (2 * (1 - (1 + (2 * (1 - x)))))) = 8 * x - 10 := 
by sorry

end simplify_expression_l109_109963


namespace perpendicular_condition_l109_109936

def line1 (a : ℝ) (x y : ℝ) : ℝ := a * x + 2 * y - 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := 3 * x - a * y + 1

def perpendicular_lines (a : ℝ) : Prop := 
  ∀ (x y : ℝ), line1 a x y = 0 → line2 a x y = 0 → 3 * a - 2 * a = 0 

theorem perpendicular_condition (a : ℝ) (h : perpendicular_lines a) : a = 0 := sorry

end perpendicular_condition_l109_109936


namespace ratio_of_length_to_width_of_field_is_two_to_one_l109_109972

-- Definitions based on conditions
def lengthOfField : ℕ := 80
def widthOfField (field_area pond_area : ℕ) : ℕ := field_area / lengthOfField
def pondSideLength : ℕ := 8
def pondArea : ℕ := pondSideLength * pondSideLength
def fieldArea : ℕ := pondArea * 50
def lengthMultipleOfWidth (length width : ℕ) := ∃ k : ℕ, length = k * width

-- Main statement to prove the ratio of length to width is 2:1
theorem ratio_of_length_to_width_of_field_is_two_to_one :
  lengthMultipleOfWidth lengthOfField (widthOfField fieldArea pondArea) →
  lengthOfField = 2 * (widthOfField fieldArea pondArea) :=
by
  -- Conditions
  have h1 : pondSideLength = 8 := rfl
  have h2 : pondArea = pondSideLength * pondSideLength := rfl
  have h3 : fieldArea = pondArea * 50 := rfl
  have h4 : lengthOfField = 80 := rfl
  sorry

end ratio_of_length_to_width_of_field_is_two_to_one_l109_109972


namespace ammonium_chloride_potassium_hydroxide_ammonia_l109_109593

theorem ammonium_chloride_potassium_hydroxide_ammonia
  (moles_KOH : ℕ) (moles_NH3 : ℕ) (moles_NH4Cl : ℕ) 
  (reaction : moles_KOH = 3 ∧ moles_NH3 = moles_KOH ∧ moles_NH4Cl >= moles_KOH) : 
  moles_NH3 = 3 :=
by
  sorry

end ammonium_chloride_potassium_hydroxide_ammonia_l109_109593


namespace number_of_combinations_of_planets_is_1141_l109_109616

def number_of_combinations_of_planets : ℕ :=
  (if 7 ≥ 7 ∧ 8 ≥2 then Nat.choose 7 7 * Nat.choose 8 2 else 0) + 
  (if 7 ≥ 6 ∧ 8 ≥ 4 then Nat.choose 7 6 * Nat.choose 8 4 else 0) + 
  (if 7 ≥ 5 ∧ 8 ≥ 6 then Nat.choose 7 5 * Nat.choose 8 6 else 0) +
  (if 7 ≥ 4 ∧ 8 ≥ 8 then Nat.choose 7 4 * Nat.choose 8 8 else 0)

theorem number_of_combinations_of_planets_is_1141 :
  number_of_combinations_of_planets = 1141 :=
by
  sorry

end number_of_combinations_of_planets_is_1141_l109_109616


namespace prob_yellow_is_3_over_5_required_red_balls_is_8_l109_109332

-- Defining the initial conditions
def total_balls : ℕ := 10
def red_balls : ℕ := 4
def yellow_balls : ℕ := 6

-- Part 1: Prove the probability of drawing a yellow ball is 3/5
theorem prob_yellow_is_3_over_5 :
  (yellow_balls : ℚ) / (total_balls : ℚ) = 3 / 5 := sorry

-- Part 2: Prove that adding 8 red balls makes the probability of drawing a red ball 2/3
theorem required_red_balls_is_8 (x : ℕ) :
  (red_balls + x : ℚ) / (total_balls + x : ℚ) = 2 / 3 → x = 8 := sorry

end prob_yellow_is_3_over_5_required_red_balls_is_8_l109_109332


namespace hannah_final_pay_l109_109114

theorem hannah_final_pay : (30 * 18) - (5 * 3) + (15 * 4) - (((30 * 18) - (5 * 3) + (15 * 4)) * 0.10 + ((30 * 18) - (5 * 3) + (15 * 4)) * 0.05) = 497.25 :=
by
  sorry

end hannah_final_pay_l109_109114


namespace possible_values_of_a_l109_109106

def A (a : ℝ) : Set ℝ := { x | 0 < x ∧ x < a }
def B : Set ℝ := { x | 1 < x ∧ x < 2 }
def complement_R (s : Set ℝ) : Set ℝ := { x | x ∉ s }

theorem possible_values_of_a (a : ℝ) :
  (∃ x, x ∈ A a) →
  B ⊆ complement_R (A a) →
  0 < a ∧ a ≤ 1 :=
by 
  sorry

end possible_values_of_a_l109_109106


namespace find_edge_lengths_sum_l109_109104

noncomputable def sum_edge_lengths (a d : ℝ) (volume surface_area : ℝ) : ℝ :=
  if (a - d) * a * (a + d) = volume ∧ 2 * ((a - d) * a + a * (a + d) + (a - d) * (a + d)) = surface_area then
    4 * ((a - d) + a + (a + d))
  else
    0

theorem find_edge_lengths_sum:
  (∃ a d : ℝ, (a - d) * a * (a + d) = 512 ∧ 2 * ((a - d) * a + a * (a + d) + (a - d) * (a + d)) = 352) →
  sum_edge_lengths (Real.sqrt 59) 1 512 352 = 12 * Real.sqrt 59 :=
by
  sorry

end find_edge_lengths_sum_l109_109104


namespace clara_biked_more_l109_109631

def clara_speed : ℕ := 18
def denise_speed : ℕ := 16
def race_duration : ℕ := 5

def clara_distance := clara_speed * race_duration
def denise_distance := denise_speed * race_duration
def distance_difference := clara_distance - denise_distance

theorem clara_biked_more : distance_difference = 10 := by
  sorry

end clara_biked_more_l109_109631


namespace ratio_of_areas_l109_109486

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C ^ 2) / (Real.pi * R_D ^ 2) = 4 / 9 :=
by
  -- The problem is to prove the ratio of the areas is 4/9
  sorry

end ratio_of_areas_l109_109486


namespace range_of_m_l109_109285

variable (x y m : ℝ)
variable (h1 : 0 < x)
variable (h2 : 0 < y)
variable (h3 : 2/x + 1/y = 1)
variable (h4 : ∀ x y : ℝ, x + 2*y > m^2 + 2*m)

theorem range_of_m (h1 : 0 < x) (h2 : 0 < y) (h3 : 2/x + 1/y = 1) (h4 : ∀ x y : ℝ, x + 2*y > m^2 + 2*m) : -4 < m ∧ m < 2 := 
sorry

end range_of_m_l109_109285


namespace problem_statement_l109_109317

-- Define the variables
variables (S T Tie : ℝ)

-- Define the given conditions
def condition1 : Prop := 6 * S + 4 * T + 2 * Tie = 80
def condition2 : Prop := 5 * S + 3 * T + 2 * Tie = 110

-- Define the question to be proved
def target : Prop := 4 * S + 2 * T + 2 * Tie = 50

-- Lean theorem statement
theorem problem_statement (h1 : condition1 S T Tie) (h2 : condition2 S T Tie) : target S T Tie :=
  sorry

end problem_statement_l109_109317


namespace rectangle_measurement_error_l109_109635

theorem rectangle_measurement_error
    (L W : ℝ) -- actual lengths of the sides
    (x : ℝ) -- percentage in excess for the first side
    (h1 : 0 ≤ x) -- ensuring percentage cannot be negative
    (h2 : (L * (1 + x / 100)) * (W * 0.95) = L * W * 1.045) -- given condition on areas
    : x = 10 :=
by
  sorry

end rectangle_measurement_error_l109_109635


namespace choir_grouping_l109_109408

theorem choir_grouping (sopranos altos tenors basses : ℕ)
  (h_sopranos : sopranos = 10)
  (h_altos : altos = 15)
  (h_tenors : tenors = 12)
  (h_basses : basses = 18)
  (ratio : ℕ) :
  ratio = 1 →
  ∃ G : ℕ, G ≤ 10 ∧ G ≤ 15 ∧ G ≤ 12 ∧ 2 * G ≤ 18 ∧ G = 9 :=
by sorry

end choir_grouping_l109_109408


namespace find_x_plus_y_l109_109924

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2008) (h2 : x + 2008 * Real.cos y = 2007) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 :=
by
  sorry

end find_x_plus_y_l109_109924


namespace eq_4_double_prime_l109_109636

-- Define the function f such that f(q) = 3q - 3
def f (q : ℕ) : ℕ := 3 * q - 3

-- Theorem statement to show that f(f(4)) = 24
theorem eq_4_double_prime : f (f 4) = 24 := by
  sorry

end eq_4_double_prime_l109_109636


namespace count_not_divisible_by_2_3_5_l109_109682

theorem count_not_divisible_by_2_3_5 : 
  let count_div_2 := (100 / 2)
  let count_div_3 := (100 / 3)
  let count_div_5 := (100 / 5)
  let count_div_6 := (100 / 6)
  let count_div_10 := (100 / 10)
  let count_div_15 := (100 / 15)
  let count_div_30 := (100 / 30)
  100 - (count_div_2 + count_div_3 + count_div_5) 
      + (count_div_6 + count_div_10 + count_div_15) 
      - count_div_30 = 26 :=
by
  let count_div_2 := 50
  let count_div_3 := 33
  let count_div_5 := 20
  let count_div_6 := 16
  let count_div_10 := 10
  let count_div_15 := 6
  let count_div_30 := 3
  sorry

end count_not_divisible_by_2_3_5_l109_109682


namespace minimum_m_value_l109_109347

theorem minimum_m_value (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_eq : 24 * m = n^4) : m = 54 := sorry

end minimum_m_value_l109_109347


namespace triangle_at_most_one_obtuse_l109_109634

theorem triangle_at_most_one_obtuse 
  (A B C : ℝ)
  (h_sum : A + B + C = 180) 
  (h_obtuse_A : A > 90) 
  (h_obtuse_B : B > 90) 
  (h_obtuse_C : C > 90) :
  false :=
by 
  sorry

end triangle_at_most_one_obtuse_l109_109634


namespace crayons_difference_l109_109023

theorem crayons_difference (total_crayons : ℕ) (given_crayons : ℕ) (lost_crayons : ℕ) (h1 : total_crayons = 589) (h2 : given_crayons = 571) (h3 : lost_crayons = 161) : (given_crayons - lost_crayons) = 410 := by
  sorry

end crayons_difference_l109_109023


namespace quadratic_roots_real_and_equal_l109_109746

open Real

theorem quadratic_roots_real_and_equal :
  ∀ (x : ℝ), x^2 - 4 * x * sqrt 2 + 8 = 0 → ∃ r : ℝ, x = r :=
by
  intro x
  sorry

end quadratic_roots_real_and_equal_l109_109746


namespace multiply_polynomials_l109_109350

variable {x y z : ℝ}

theorem multiply_polynomials :
  (3 * x^4 - 4 * y^3 - 6 * z^2) * (9 * x^8 + 16 * y^6 + 36 * z^4 + 12 * x^4 * y^3 + 18 * x^4 * z^2 + 24 * y^3 * z^2)
  = 27 * x^12 - 64 * y^9 - 216 * z^6 - 216 * x^4 * y^3 * z^2 := by {
  sorry
}

end multiply_polynomials_l109_109350


namespace solve_apples_problem_l109_109513

def apples_problem (marin_apples donald_apples total_apples : ℕ) : Prop :=
  marin_apples = 9 ∧ total_apples = 11 → donald_apples = 2

theorem solve_apples_problem : apples_problem 9 2 11 := by
  sorry

end solve_apples_problem_l109_109513


namespace jeff_total_travel_distance_l109_109011

theorem jeff_total_travel_distance :
  let d1 := 80 * 6 in
  let d2 := 60 * 4 in
  let d3 := 40 * 2 in
  d1 + d2 + d3 = 800 :=
by
  sorry

end jeff_total_travel_distance_l109_109011


namespace friend_spent_more_l109_109393

theorem friend_spent_more (total_spent friend_spent: ℝ) (h_total: total_spent = 15) (h_friend: friend_spent = 10) :
  friend_spent - (total_spent - friend_spent) = 5 :=
by
  sorry

end friend_spent_more_l109_109393


namespace expected_plain_zongzi_picked_l109_109437

-- Definitions and conditions:
def total_zongzi := 10
def red_bean_zongzi := 3
def meat_zongzi := 3
def plain_zongzi := 4

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probabilities
def P_X_0 : ℚ := (choose 6 2 : ℚ) / choose 10 2
def P_X_1 : ℚ := (choose 6 1 * choose 4 1 : ℚ) / choose 10 2
def P_X_2 : ℚ := (choose 4 2 : ℚ) / choose 10 2

-- Expected value of X
def E_X : ℚ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2

theorem expected_plain_zongzi_picked : E_X = 4 / 5 := by
  -- Using the definition of E_X and the respective probabilities
  unfold E_X P_X_0 P_X_1 P_X_2
  -- Use the given formula to calculate the values
  -- Remaining steps would show detailed calculations leading to the answer
  sorry

end expected_plain_zongzi_picked_l109_109437


namespace equivalent_conditions_l109_109138

open Real

theorem equivalent_conditions (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / x + 1 / y + 1 / z ≤ 1) ↔
  (∀ a b c d : ℝ, a + b + c > d → a^2 * x + b^2 * y + c^2 * z > d^2) :=
by
  sorry

end equivalent_conditions_l109_109138


namespace inscribed_square_in_isosceles_triangle_l109_109134

-- Definitions of the triangle and the inscribed square
structure Triangle where
  a b c : ℝ

structure Square where
  side : ℝ

-- Proof problem statement
theorem inscribed_square_in_isosceles_triangle :
  ∀ (T : Triangle) (S : Square),
    T.a = T.b ∧
    T.a = 10 ∧
    T.c = 12 →
    S.side = 4.8 :=
by
  intro T S
  assume h : T.a = T.b ∧ T.a = 10 ∧ T.c = 12
  sorry

end inscribed_square_in_isosceles_triangle_l109_109134


namespace probability_x_greater_3y_l109_109654

theorem probability_x_greater_3y (x y : ℝ) (h1: 0 ≤ x ∧ x ≤ 2010) (h2: 0 ≤ y ∧ y ≤ 2011) : 
  (let area_triangle := (2010 * (2010 / 3)) / 2 in 
   let area_rectangle := 2010 * 2011 in 
   (area_triangle / area_rectangle = 335 / 2011)) := 
sorry

end probability_x_greater_3y_l109_109654


namespace probability_of_at_least_one_3_l109_109551

noncomputable def probability_at_least_one_3 (d1 d2 : ℕ) (h : d1 ≠ d2) : ℚ :=
let total_outcomes := 30 in
let favorable_outcomes := 10 in
favorable_outcomes / total_outcomes

theorem probability_of_at_least_one_3 (d1 d2 : ℕ) (h : d1 ≠ d2) :
  probability_at_least_one_3 d1 d2 h = 1 / 3 :=
sorry

end probability_of_at_least_one_3_l109_109551


namespace select_and_swap_ways_l109_109843

theorem select_and_swap_ways :
  let n := 8
  let k := 3
  Nat.choose n k * 2 = 112 := 
by
  let n := 8
  let k := 3
  sorry

end select_and_swap_ways_l109_109843


namespace inequality_range_of_k_l109_109798

theorem inequality_range_of_k 
  (a b k : ℝ)
  (h : ∀ a b : ℝ, a^2 + b^2 ≥ 2 * k * a * b) : k ∈ Set.Icc (-1 : ℝ) (1 : ℝ) :=
by
  sorry

end inequality_range_of_k_l109_109798


namespace range_of_a_l109_109296

noncomputable def setA : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def setB (a : ℝ) : Set ℝ := {x | x^2 + a * x + a = 0}

theorem range_of_a :
  ∀ a : ℝ, (setA ∪ setB a) = setA ↔ 0 ≤ a ∧ a < 4 :=
by sorry

end range_of_a_l109_109296


namespace total_price_is_correct_l109_109241

-- Define the cost of an adult ticket
def cost_adult : ℕ := 22

-- Define the cost of a children ticket
def cost_child : ℕ := 7

-- Define the number of adults in the family
def num_adults : ℕ := 2

-- Define the number of children in the family
def num_children : ℕ := 2

-- Define the total price the family will pay
def total_price : ℕ := cost_adult * num_adults + cost_child * num_children

-- The proof to check the total price
theorem total_price_is_correct : total_price = 58 :=
by 
  -- Here we would solve the proof
  sorry

end total_price_is_correct_l109_109241


namespace sufficient_but_not_necessary_l109_109919

variable {a b : ℝ}

theorem sufficient_but_not_necessary (ha : a > 0) (hb : b > 0) : 
  (ab > 1) → (a + b > 2) ∧ ¬ (a + b > 2 → ab > 1) :=
by
  sorry

end sufficient_but_not_necessary_l109_109919


namespace problem_l109_109716

theorem problem (a : ℕ) (h1 : a = 444) : (444 ^ 444) % 13 = 1 :=
by
  have h444 : 444 % 13 = 3 := by sorry
  have h3_pow3 : 3 ^ 3 % 13 = 1 := by sorry
  sorry

end problem_l109_109716


namespace second_polygon_sides_l109_109050

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0)
  (perimeter_eq : ∀ (p1 p2 : ℝ), p1 = p2)
  (first_sides : ℕ) (second_sides : ℕ)
  (first_polygon_side_length : ℝ) (second_polygon_side_length : ℝ)
  (first_sides_eq : first_sides = 50)
  (side_length_relation : first_polygon_side_length = 3 * second_polygon_side_length)
  (perimeter_relation : first_sides * first_polygon_side_length = second_sides * second_polygon_side_length) :
  second_sides = 150 := by
  sorry

end second_polygon_sides_l109_109050


namespace second_polygon_sides_l109_109038

theorem second_polygon_sides (s : ℝ) (h : s > 0) :
  let s1 := 3 * s in
  let s2 := s in
  let sides1 := 50 in
  let perimeter1 := sides1 * s1 in
  let perimeter2 := perimeter1 in
  perimeter2 / s2 = 150 := 
by
  let s1 := 3 * s
  let s2 := s
  let sides1 := 50
  let perimeter1 := sides1 * s1
  let perimeter2 := perimeter1
  show perimeter2 / s2 = 150
  sorry

end second_polygon_sides_l109_109038


namespace hyperbola_eccentricity_l109_109163

theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ), a = 3 → b = 4 → c = Real.sqrt (a^2 + b^2) → c / a = 5 / 3 :=
by
  intros a b c ha hb h_eq
  sorry

end hyperbola_eccentricity_l109_109163


namespace find_length_of_second_movie_l109_109299

noncomputable def length_of_second_movie := 1.5

theorem find_length_of_second_movie
  (total_free_time : ℝ)
  (first_movie_duration : ℝ)
  (words_read : ℝ)
  (reading_rate : ℝ) : 
  first_movie_duration = 3.5 → 
  total_free_time = 8 → 
  words_read = 1800 → 
  reading_rate = 10 → 
  length_of_second_movie = 1.5 := 
by
  intros h1 h2 h3 h4
  -- Here should be the proof steps, which are abstracted away.
  sorry

end find_length_of_second_movie_l109_109299


namespace value_of_X_is_one_l109_109273

-- Problem: Given the numbers 28 at the start of a row, 17 in the middle, and -15 in the same column as X,
-- we show the value of X must be 1 because the sequences are arithmetic.

theorem value_of_X_is_one (d : ℤ) (X : ℤ) :
  -- Conditions
  (17 - X = d) ∧ 
  (X - (-15) = d) ∧ 
  (d = 16) →
  -- Conclusion: X must be 1
  X = 1 :=
by 
  sorry

end value_of_X_is_one_l109_109273


namespace sum_zero_implies_product_terms_nonpositive_l109_109337

theorem sum_zero_implies_product_terms_nonpositive (a b c : ℝ) (h : a + b + c = 0) : 
  a * b + a * c + b * c ≤ 0 := 
by 
  sorry

end sum_zero_implies_product_terms_nonpositive_l109_109337


namespace max_value_of_f_range_of_m_l109_109111

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x^2

theorem max_value_of_f (a b : ℝ) (x : ℝ) (h1 : 1 / Real.exp 1 ≤ x ∧ x ≤ Real.exp 1)
  (h_tangent : ∀ (x : ℝ), f a b x - ((-1/2) * x + (Real.log 1 - 1/2)) = 0) : 
  ∃ x_max, f a b x_max = -1/2 := sorry

theorem range_of_m (m : ℝ) 
  (h_ineq : ∀ (a : ℝ) (x : ℝ), 1 ≤ a ∧ a ≤ 3 / 2 ∧ 1 ≤ x ∧ x ≤ Real.exp 2 → a * Real.log x ≥ m + x) : 
  m ≤ 2 - Real.exp 2 := sorry

end max_value_of_f_range_of_m_l109_109111


namespace problem_solution_l109_109757

theorem problem_solution :
  (- (5 : ℚ) / 12) ^ 2023 * (12 / 5) ^ 2023 = -1 := 
by
  sorry

end problem_solution_l109_109757


namespace remainder_division_l109_109028

theorem remainder_division (L S R : ℕ) (h1 : L - S = 1325) (h2 : L = 1650) (h3 : L = 5 * S + R) : 
  R = 25 :=
sorry

end remainder_division_l109_109028


namespace polygon_area_correct_l109_109424

noncomputable def polygonArea : ℝ :=
  let x1 := 1
  let y1 := 1
  let x2 := 4
  let y2 := 3
  let x3 := 5
  let y3 := 1
  let x4 := 6
  let y4 := 4
  let x5 := 3
  let y5 := 6
  (1 / 2 : ℝ) * 
  abs ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y5 + x5 * y1) -
       (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x5 + y5 * x1))

theorem polygon_area_correct : polygonArea = 11.5 := by
  sorry

end polygon_area_correct_l109_109424


namespace length_of_second_train_is_correct_l109_109736

noncomputable def convert_kmph_to_mps (speed_kmph: ℕ) : ℝ :=
  speed_kmph * (1000 / 3600)

def train_lengths_and_time
  (length_first_train : ℝ)
  (speed_first_train_kmph : ℕ)
  (speed_second_train_kmph : ℕ)
  (time_to_cross : ℝ)
  (length_second_train : ℝ) : Prop :=
  let speed_first_train_mps := convert_kmph_to_mps speed_first_train_kmph
  let speed_second_train_mps := convert_kmph_to_mps speed_second_train_kmph
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance := relative_speed * time_to_cross
  total_distance = length_first_train + length_second_train

theorem length_of_second_train_is_correct :
  train_lengths_and_time 260 120 80 9 239.95 :=
by
  sorry

end length_of_second_train_is_correct_l109_109736


namespace digit_150_of_17_div_70_is_2_l109_109221

def repeating_cycle_17_div_70 : List ℕ := [2, 4, 2, 8, 5, 7, 1]

theorem digit_150_of_17_div_70_is_2 : 
  (repeating_cycle_17_div_70[(150 % 7) - 1] = 2) :=
by
  -- the proof will go here
  sorry

end digit_150_of_17_div_70_is_2_l109_109221


namespace number_of_windows_davids_house_l109_109638

theorem number_of_windows_davids_house
  (windows_per_minute : ℕ → ℕ)
  (h1 : ∀ t, windows_per_minute t = (4 * t) / 10)
  (h2 : windows_per_minute 160 = w)
  : w = 64 :=
by
  sorry

end number_of_windows_davids_house_l109_109638


namespace problem_l109_109440

variable (a : ℝ)

def condition : Prop := (a / 3) - (3 / a) = 4

theorem problem (h : condition a) : ((a ^ 8 - 6561) / (81 * a ^ 4)) * (3 * a / (a ^ 2 + 9)) = 72 :=
by
  sorry

end problem_l109_109440


namespace intersection_complement_l109_109473

open Set

noncomputable def N := {x : ℕ | true}

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}
def C_N (B : Set ℕ) : Set ℕ := {n ∈ N | n ∉ B}

theorem intersection_complement :
  A ∩ (C_N B) = {1} :=
by
  sorry

end intersection_complement_l109_109473


namespace complement_A_in_B_l109_109604

-- Define the sets A and B
def A : Set ℕ := {2, 3}
def B : Set ℕ := {0, 1, 2, 3, 4}

-- Define the complement of A in B
def complement (U : Set ℕ) (A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- Statement to prove
theorem complement_A_in_B :
  complement B A = {0, 1, 4} := by
  sorry

end complement_A_in_B_l109_109604


namespace digit_150th_l109_109210

-- Define the fraction and its repeating decimal properties
def fraction := 17 / 70
def repeating_block := "242857"
def block_length : ℕ := 6

-- The 150th digit calculation
def digit_position : ℕ := 150 % block_length

theorem digit_150th : "150th digit of decimal representation" = 7 :=
by {
  have h1 : fraction = 0.242857142857, sorry,
  have h2 : "252857".cycle = repeating_block, sorry,
  have h3 : digit_position = 0, sorry,
  have h4 : repeating_block.get_last = '7', sorry,
  show "150th digit of decimal representation"= 7, sorry,
}

end digit_150th_l109_109210


namespace division_pairs_l109_109441

theorem division_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (ab^2 + b + 7) ∣ (a^2 * b + a + b) →
  (∃ k : ℕ, k ≥ 1 ∧ a = 7 * k^2 ∧ b = 7 * k) ∨ (a, b) = (11, 1) ∨ (a, b) = (49, 1) :=
sorry

end division_pairs_l109_109441


namespace gcd_three_digit_numbers_l109_109540

theorem gcd_three_digit_numbers (a b c : ℕ) (h1 : b = a + 1) (h2 : c = a + 2) :
  ∃ k, (∀ n, n = 100 * a + 10 * b + c + 100 * c + 10 * b + a → n = 212 * k) :=
by
  sorry

end gcd_three_digit_numbers_l109_109540


namespace segment_length_is_15_l109_109615

theorem segment_length_is_15 : 
  ∀ (x : ℝ), 
  ∀ (y1 y2 : ℝ), 
  x = 3 → 
  y1 = 5 → 
  y2 = 20 → 
  abs (y2 - y1) = 15 := by 
sorry

end segment_length_is_15_l109_109615


namespace inequality_true_l109_109015

noncomputable def f : ℝ → ℝ := sorry -- f is a function defined on (0, +∞)

axiom f_derivative (x : ℝ) (hx : 0 < x) : ∃ f'' : ℝ → ℝ, f'' x * x + 2 * f x = 1 / x^2

theorem inequality_true : (f 2) / 9 < (f 3) / 4 :=
  sorry

end inequality_true_l109_109015


namespace first_group_people_count_l109_109966

theorem first_group_people_count (P : ℕ) (W : ℕ) 
  (h1 : P * 3 * W = 3 * W) 
  (h2 : 8 * 3 * W = 8 * W) : 
  P = 3 :=
by
  sorry

end first_group_people_count_l109_109966


namespace prove_collinear_prove_perpendicular_l109_109451

noncomputable def vec_a : ℝ × ℝ := (1, 3)
noncomputable def vec_b : ℝ × ℝ := (3, -4)

def collinear (k : ℝ) : Prop :=
  let v1 := (k * 1 - 3, k * 3 + 4)
  let v2 := (1 + 3, 3 - 4)
  v1.1 * v2.2 = v1.2 * v2.1

def perpendicular (k : ℝ) : Prop :=
  let v1 := (k * 1 - 3, k * 3 + 4)
  let v2 := (1 + 3, 3 - 4)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem prove_collinear : collinear (-1) :=
by
  sorry

theorem prove_perpendicular : perpendicular (16) :=
by
  sorry

end prove_collinear_prove_perpendicular_l109_109451


namespace smallest_root_of_g_l109_109897

noncomputable def g (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

theorem smallest_root_of_g : ∀ x : ℝ, g x = 0 → x = - Real.sqrt (3 / 7) :=
by
  sorry

end smallest_root_of_g_l109_109897


namespace abby_potatoes_peeled_l109_109476

theorem abby_potatoes_peeled (total_potatoes : ℕ) (homers_rate : ℕ) (abbys_rate : ℕ) (time_alone : ℕ) (potatoes_peeled : ℕ) :
  (total_potatoes = 60) →
  (homers_rate = 4) →
  (abbys_rate = 6) →
  (time_alone = 6) →
  (potatoes_peeled = 22) :=
  sorry

end abby_potatoes_peeled_l109_109476


namespace first_platform_length_l109_109070

noncomputable def length_of_first_platform (t1 t2 l_train l_plat2 time1 time2 : ℕ) : ℕ :=
  let s1 := (l_train + t1) / time1
  let s2 := (l_train + l_plat2) / time2
  if s1 = s2 then t1 else 0

theorem first_platform_length:
  ∀ (time1 time2 : ℕ) (l_train l_plat2 : ℕ), time1 = 15 → time2 = 20 → l_train = 350 → l_plat2 = 250 → length_of_first_platform 100 l_plat2 l_train l_plat2 time1 time2 = 100 :=
by
  intros time1 time2 l_train l_plat2 ht1 ht2 ht3 ht4
  rw [ht1, ht2, ht3, ht4]
  dsimp [length_of_first_platform]
  rfl

end first_platform_length_l109_109070


namespace find_original_denominator_l109_109255

theorem find_original_denominator (d : ℕ) (h : (3 + 7) / (d + 7) = 1 / 3) : d = 23 :=
sorry

end find_original_denominator_l109_109255


namespace waiter_gratuity_l109_109511

def price_leticia : ℕ := 10
def price_scarlett : ℕ := 13
def price_percy : ℕ := 17

def total_cost := price_leticia + price_scarlett + price_percy
def tip_percentage := 0.10
def gratuity := (tip_percentage * total_cost.toReal).toNat

theorem waiter_gratuity : gratuity = 4 :=
sorry

end waiter_gratuity_l109_109511


namespace count_valid_n_l109_109272

theorem count_valid_n (n : ℕ) (h₁ : (n % 2015) ≠ 0) :
  (n^3 + 3^n) % 5 = 0 :=
by
  sorry

end count_valid_n_l109_109272


namespace goods_train_speed_l109_109251

noncomputable def passenger_train_speed := 64 -- in km/h
noncomputable def passing_time := 18 -- in seconds
noncomputable def goods_train_length := 420 -- in meters
noncomputable def relative_speed_kmh := 84 -- in km/h (derived from solution)

theorem goods_train_speed :
  (∃ V_g, relative_speed_kmh = V_g + passenger_train_speed) →
  (goods_train_length / (passing_time / 3600): ℝ) = relative_speed_kmh →
  V_g = 20 :=
by
  intro h1 h2
  sorry

end goods_train_speed_l109_109251


namespace cost_price_of_article_l109_109032

theorem cost_price_of_article (x : ℝ) (h : 66 - x = x - 22) : x = 44 :=
sorry

end cost_price_of_article_l109_109032


namespace duration_of_each_class_is_3_l109_109762

theorem duration_of_each_class_is_3
    (weeks : ℕ) 
    (x : ℝ) 
    (weekly_additional_class_hours : ℝ) 
    (homework_hours_per_week : ℝ) 
    (total_hours : ℝ) 
    (h1 : weeks = 24)
    (h2 : weekly_additional_class_hours = 4)
    (h3 : homework_hours_per_week = 4)
    (h4 : total_hours = 336) :
    (2 * x + weekly_additional_class_hours + homework_hours_per_week) * weeks = total_hours → x = 3 := 
by 
  sorry

end duration_of_each_class_is_3_l109_109762


namespace angle_B_range_l109_109943

def range_of_angle_B (a b c : ℝ) (A B C : ℝ) : Prop :=
  (0 < B ∧ B ≤ Real.pi / 3)

theorem angle_B_range
  (a b c A B C : ℝ)
  (h1 : b^2 = a * c)
  (h2 : A + B + C = π)
  (h3 : a > 0)
  (h4 : b > 0)
  (h5 : c > 0)
  (h6 : a + b > c)
  (h7 : a + c > b)
  (h8 : b + c > a) :
  range_of_angle_B a b c A B C :=
sorry

end angle_B_range_l109_109943


namespace spider_distance_l109_109568

/--
A spider crawls along a number line, starting at -3.
It crawls to -7, then turns around and crawls to 8.
--/
def spiderCrawl (start : ℤ) (point1 : ℤ) (point2 : ℤ): ℤ :=
  let dist1 := abs (point1 - start)
  let dist2 := abs (point2 - point1)
  dist1 + dist2

theorem spider_distance :
  spiderCrawl (-3) (-7) 8 = 19 :=
by
  sorry

end spider_distance_l109_109568


namespace power_mod_eq_one_l109_109693

theorem power_mod_eq_one (n : ℕ) (h₁ : 444 ≡ 3 [MOD 13]) (h₂ : 3^12 ≡ 1 [MOD 13]) :
  444^444 ≡ 1 [MOD 13] :=
by
  sorry

end power_mod_eq_one_l109_109693


namespace binom_divisibility_l109_109139

theorem binom_divisibility (k n : ℕ) (p : ℕ) (h1 : k > 1) (h2 : n > 1) 
  (h3 : p = 2 * k - 1) (h4 : Nat.Prime p) (h5 : p ∣ (Nat.choose n 2 - Nat.choose k 2)) : 
  p^2 ∣ (Nat.choose n 2 - Nat.choose k 2) := 
sorry

end binom_divisibility_l109_109139


namespace point_in_fourth_quadrant_l109_109685

def is_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  is_fourth_quadrant 2 (-3) :=
by
  sorry

end point_in_fourth_quadrant_l109_109685


namespace digit_150th_l109_109209

-- Define the fraction and its repeating decimal properties
def fraction := 17 / 70
def repeating_block := "242857"
def block_length : ℕ := 6

-- The 150th digit calculation
def digit_position : ℕ := 150 % block_length

theorem digit_150th : "150th digit of decimal representation" = 7 :=
by {
  have h1 : fraction = 0.242857142857, sorry,
  have h2 : "252857".cycle = repeating_block, sorry,
  have h3 : digit_position = 0, sorry,
  have h4 : repeating_block.get_last = '7', sorry,
  show "150th digit of decimal representation"= 7, sorry,
}

end digit_150th_l109_109209


namespace vasya_filling_time_l109_109539

-- Definition of conditions
def hose_filling_time (x : ℝ) : Prop :=
  ∀ (first_hose_mult second_hose_mult : ℝ), 
    first_hose_mult = x ∧
    second_hose_mult = 5 * x ∧
    (5 * second_hose_mult - 5 * first_hose_mult) = 1

-- Conclusion
theorem vasya_filling_time (x : ℝ) (first_hose_mult second_hose_mult : ℝ) :
  hose_filling_time x → 25 * x = 1 * (60 + 15) := sorry

end vasya_filling_time_l109_109539


namespace remainder_of_division_l109_109756

def num : ℤ := 1346584
def divisor : ℤ := 137
def remainder : ℤ := 5

theorem remainder_of_division 
  (h : 0 <= divisor) (h' : divisor ≠ 0) : 
  num % divisor = remainder := 
sorry

end remainder_of_division_l109_109756


namespace salary_increase_is_57point35_percent_l109_109646

variable (S : ℝ)

-- Assume Mr. Blue receives a 12% raise every year.
def annualRaise : ℝ := 1.12

-- After four years
theorem salary_increase_is_57point35_percent (h : annualRaise ^ 4 = 1.5735):
  ((annualRaise ^ 4 - 1) * S) / S = 0.5735 :=
by
  sorry

end salary_increase_is_57point35_percent_l109_109646


namespace probability_of_x_gt_3y_in_rectangle_is_335_over_2011_l109_109665

noncomputable def probability_x_gt_3y : ℚ :=
  let rectangle := ((0 : ℚ, 0 : ℚ), (2010 : ℚ), (2010 : ℚ, 2011 : ℚ), (0 : ℚ, 2011 : ℚ)) in
  let area_rectangle := 2010 * 2011 in
  let area_triangle := (1/2) * 2010 * 670 in
  area_triangle / area_rectangle

theorem probability_of_x_gt_3y_in_rectangle_is_335_over_2011 : probability_x_gt_3y = 335 / 2011 :=
begin
  sorry
end

end probability_of_x_gt_3y_in_rectangle_is_335_over_2011_l109_109665


namespace largest_n_l109_109278

def a_n (n : ℕ) (d_a : ℤ) : ℤ := 1 + (n-1) * d_a
def b_n (n : ℕ) (d_b : ℤ) : ℤ := 3 + (n-1) * d_b

theorem largest_n (d_a d_b : ℤ) (n : ℕ) :
  (a_n n d_a * b_n n d_b = 2304 ∧ a_n 1 d_a = 1 ∧ b_n 1 d_b = 3) 
  → n ≤ 20 := 
sorry

end largest_n_l109_109278


namespace two_integers_divide_2_pow_96_minus_1_l109_109925

theorem two_integers_divide_2_pow_96_minus_1 : 
  ∃ a b : ℕ, (60 < a ∧ a < 70 ∧ 60 < b ∧ b < 70 ∧ a ≠ b ∧ a ∣ (2^96 - 1) ∧ b ∣ (2^96 - 1) ∧ a = 63 ∧ b = 65) := 
sorry

end two_integers_divide_2_pow_96_minus_1_l109_109925


namespace benjamin_franklin_gathering_handshakes_l109_109572

theorem benjamin_franklin_gathering_handshakes :
  ∃ (n m : ℕ), n = 15 ∧ m = 15 ∧ 
  let total_handshakes := (n * (n - 1)) / 2 + n * (m - 1)
  in total_handshakes = 315 :=
by
  sorry

end benjamin_franklin_gathering_handshakes_l109_109572


namespace digit_150_in_17_div_70_l109_109200

noncomputable def repeating_sequence_170 : List Nat := [2, 4, 2, 8, 5, 7]

theorem digit_150_in_17_div_70 : (repeating_sequence_170.get? (150 % 6 - 1) = some 7) := by
  sorry

end digit_150_in_17_div_70_l109_109200


namespace solve_for_x_l109_109359

theorem solve_for_x:
  ∃ x : ℚ, (2 / 3 - 1 / 4 = 1 / x) ∧ (x = 12 / 5) := by
  sorry

end solve_for_x_l109_109359


namespace fundraiser_goal_eq_750_l109_109977

def bronze_donations := 10 * 25
def silver_donations := 7 * 50
def gold_donations   := 1 * 100
def total_collected  := bronze_donations + silver_donations + gold_donations
def amount_needed    := 50
def total_goal       := total_collected + amount_needed

theorem fundraiser_goal_eq_750 : total_goal = 750 :=
by
  sorry

end fundraiser_goal_eq_750_l109_109977


namespace largest_digit_divisible_by_4_l109_109336

theorem largest_digit_divisible_by_4 :
  ∃ (A : ℕ), A ≤ 9 ∧ (∃ n : ℕ, 100000 * 4 + 10000 * A + 67994 = n * 4) ∧ 
  (∀ B : ℕ, 0 ≤ B ∧ B ≤ 9 ∧ (∃ m : ℕ, 100000 * 4 + 10000 * B + 67994 = m * 4) → B ≤ A) :=
sorry

end largest_digit_divisible_by_4_l109_109336


namespace slope_of_line_through_origin_and_A_l109_109322

theorem slope_of_line_through_origin_and_A :
  ∀ (x1 y1 x2 y2 : ℝ), (x1 = 0) → (y1 = 0) → (x2 = -2) → (y2 = -2) →
  (y2 - y1) / (x2 - x1) = 1 :=
by intros; sorry

end slope_of_line_through_origin_and_A_l109_109322


namespace area_of_largest_square_l109_109007

theorem area_of_largest_square (a b c : ℕ) (h_triangle : c^2 = a^2 + b^2) (h_sum_areas : a^2 + b^2 + c^2 = 450) :
  c^2 = 225 :=
by
  sorry

end area_of_largest_square_l109_109007


namespace quadratic_solution_identity_l109_109976

theorem quadratic_solution_identity {a b c : ℝ} (h1 : a ≠ 0) (h2 : a * (1 : ℝ)^2 + b * (1 : ℝ) + c = 0) : 
  a + b + c = 0 :=
sorry

end quadratic_solution_identity_l109_109976


namespace years_later_l109_109439

variables (R F Y : ℕ)

-- Conditions
def condition1 := F = 4 * R
def condition2 := F + Y = 5 * (R + Y) / 2
def condition3 := F + Y + 8 = 2 * (R + Y + 8)

-- The result to be proved
theorem years_later (R F Y : ℕ) (h1 : condition1 R F) (h2 : condition2 R F Y) (h3 : condition3 R F Y) : 
  Y = 8 := by
  sorry

end years_later_l109_109439


namespace digit_150_in_17_div_70_l109_109199

noncomputable def repeating_sequence_170 : List Nat := [2, 4, 2, 8, 5, 7]

theorem digit_150_in_17_div_70 : (repeating_sequence_170.get? (150 % 6 - 1) = some 7) := by
  sorry

end digit_150_in_17_div_70_l109_109199


namespace count_correct_propositions_l109_109288

def line_parallel_plane (a : Line) (M : Plane) : Prop := sorry
def line_perpendicular_plane (a : Line) (M : Plane) : Prop := sorry
def line_parallel_line (a b : Line) : Prop := sorry
def line_perpendicular_line (a b : Line) : Prop := sorry
def plane_perpendicular_plane (M N : Plane) : Prop := sorry

theorem count_correct_propositions 
  (a b c : Line) 
  (M N : Plane) 
  (h1 : ¬ (line_parallel_plane a M ∧ line_parallel_plane b M → line_parallel_line a b)) 
  (h2 : line_parallel_plane a M ∧ line_perpendicular_plane b M → line_perpendicular_line b a) 
  (h3 : ¬ ((line_parallel_plane a M ∧ line_perpendicular_plane b M ∧ line_perpendicular_line c a ∧ line_perpendicular_line c b) → line_perpendicular_plane c M))
  (h4 : line_perpendicular_plane a M ∧ line_parallel_plane a N → plane_perpendicular_plane M N) :
  (0 + 1 + 0 + 1) = 2 :=
sorry

end count_correct_propositions_l109_109288


namespace sum_even_integers_correct_l109_109177

variable (S1 S2 : ℕ)

-- Definition: The sum of the first 50 positive even integers
def sum_first_50_even_integers : ℕ := 2550

-- Definition: The sum of even integers from 102 to 200 inclusive
def sum_even_integers_from_102_to_200 : ℕ := 7550

-- Condition: The sum of the first 50 positive even integers is 2550
axiom sum_first_50_even_integers_given : S1 = sum_first_50_even_integers

-- Problem statement: Prove that the sum of even integers from 102 to 200 inclusive is 7550
theorem sum_even_integers_correct :
  S1 = sum_first_50_even_integers →
  S2 = sum_even_integers_from_102_to_200 →
  S2 = 7550 :=
by
  intros h1 h2
  rw [h2]
  sorry

end sum_even_integers_correct_l109_109177


namespace sequence_term_general_sequence_sum_term_general_l109_109778

theorem sequence_term_general (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S (n + 1) = 2 * S n + 1) →
  a 1 = 1 →
  (∀ n ≥ 1, a n = 2^(n-1)) :=
  sorry

theorem sequence_sum_term_general (na : ℕ → ℕ) (T : ℕ → ℕ) :
  (∀ k, na k = k * 2^(k-1)) →
  (∀ n, T n = (n - 1) * 2^n + 1) :=
  sorry

end sequence_term_general_sequence_sum_term_general_l109_109778


namespace compute_expression_value_l109_109760

noncomputable def expression := 3 ^ (Real.log 4 / Real.log 3) - 27 ^ (2 / 3) - Real.log 0.01 / Real.log 10 + Real.log (Real.exp 3)

theorem compute_expression_value :
  expression = 0 := 
by
  sorry

end compute_expression_value_l109_109760


namespace power_mod_444_444_l109_109701

open Nat

theorem power_mod_444_444 : (444:ℕ)^444 % 13 = 1 := by
  sorry

end power_mod_444_444_l109_109701


namespace angle_e1_e2_f_properties_l109_109623

variables {R : Type*} [LinearOrder R] [Field R] [NormedSpace ℝ R] [RealVectorSpace ℝ R] {x : ℝ} {n : ℕ+}

-- Define the unit vectors and their conditions
variables (e₁ e₂ e₃ : R)
variables (h1 : ∥e₁∥ = 1) (h2 : ∥e₂∥ = 1) (h3 : ∥e₃∥ = 1)
variables (h4 : e₁ + e₂ + e₃ = 0)
variables (a : R := x • e₁ + (n / x) • e₂ + (x + n / x) • e₃) 

-- Prove the angle between e₁ and e₂ is 2π/3
theorem angle_e1_e2 : 
  real.angle e₁ e₂ = 2 * real.pi / 3 :=
sorry

-- Define f and show its properties
noncomputable def f (x : ℝ) : ℝ := ∥a∥

-- Prove critical points and minimum of f(x)
theorem f_properties :
  ∃ x_min : ℝ, f x_min = sqrt n ∧ 
  (∀ x, |x| = sqrt n → f x = sqrt n) :=
sorry

end angle_e1_e2_f_properties_l109_109623


namespace power_mod_444_444_l109_109697

open Nat

theorem power_mod_444_444 : (444:ℕ)^444 % 13 = 1 := by
  sorry

end power_mod_444_444_l109_109697


namespace initial_capacity_of_bottle_l109_109507

theorem initial_capacity_of_bottle 
  (C : ℝ)
  (h1 : 1/3 * 3/4 * C = 1) : 
  C = 4 :=
by
  sorry

end initial_capacity_of_bottle_l109_109507


namespace correct_statement_l109_109998

theorem correct_statement : 
  (∀ x : ℝ, (x < 0 → x^2 > x)) ∧
  (¬ ∀ x : ℝ, (x^2 > 0 → x > 0)) ∧
  (¬ ∀ x : ℝ, (x^2 > x → x > 0)) ∧
  (¬ ∀ x : ℝ, (x^2 > x → x < 0)) ∧
  (¬ ∀ x : ℝ, (x < 1 → x^2 < x)) :=
by
  sorry

end correct_statement_l109_109998


namespace math_problem_l109_109284

variable (a : ℝ) (m n : ℝ)

theorem math_problem
  (h1 : a^m = 3)
  (h2 : a^n = 2) :
  a^(2*m + 3*n) = 72 := 
  sorry

end math_problem_l109_109284


namespace problem_l109_109712

theorem problem (a : ℕ) (h1 : a = 444) : (444 ^ 444) % 13 = 1 :=
by
  have h444 : 444 % 13 = 3 := by sorry
  have h3_pow3 : 3 ^ 3 % 13 = 1 := by sorry
  sorry

end problem_l109_109712


namespace cosine_distribution_equality_l109_109140


def uniform_distribution_on_2pi (x : ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x < 2 * Real.pi 

theorem cosine_distribution_equality :
  let varphi psi : ℝ := sorry in
  (uniform_distribution_on_2pi varphi) ∧ (uniform_distribution_on_2pi psi) ∧ (Indep varphi psi) →
  (cos (varphi) + cos (psi))/2 = cos (varphi) * cos (psi) :=
by
  sorry

end cosine_distribution_equality_l109_109140


namespace perimeter_of_ABC_HI_IJK_l109_109037

theorem perimeter_of_ABC_HI_IJK (AB AC AH HI AI AK KI IJ JK : ℝ) 
(H_midpoint : H = AC / 2) (K_midpoint : K = AI / 2) 
(equil_triangle_ABC : AB = AC) (equil_triangle_AHI : AH = HI ∧ HI = AI) 
(equil_triangle_IJK : IJ = JK ∧ JK = KI) 
(AB_eq : AB = 6) : 
  AB + AC + AH + HI + IJ + JK + KI = 22.5 :=
by
  sorry

end perimeter_of_ABC_HI_IJK_l109_109037


namespace power_mod_eq_one_l109_109695

theorem power_mod_eq_one (n : ℕ) (h₁ : 444 ≡ 3 [MOD 13]) (h₂ : 3^12 ≡ 1 [MOD 13]) :
  444^444 ≡ 1 [MOD 13] :=
by
  sorry

end power_mod_eq_one_l109_109695


namespace parallel_vectors_m_eq_neg3_l109_109298

theorem parallel_vectors_m_eq_neg3 {m : ℝ} :
  let a := (1, -2)
  let b := (1 + m, 1 - m)
  (a.1 * b.2 =  a.2 * b.1) → m = -3 :=
by 
  let a := (1, -2)
  let b := (1 + m, 1 - m)
  intro h
  sorry

end parallel_vectors_m_eq_neg3_l109_109298


namespace three_a_ge_two_b_plus_two_l109_109812

theorem three_a_ge_two_b_plus_two (a b : ℕ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : (a! * b!) % (a! + b!) = 0) :
  3 * a ≥ 2 * b + 2 :=
sorry

end three_a_ge_two_b_plus_two_l109_109812


namespace olivia_packs_of_basketball_cards_l109_109021

-- Definitions for the given conditions
def pack_cost : ℕ := 3
def deck_cost : ℕ := 4
def number_of_decks : ℕ := 5
def total_money : ℕ := 50
def change_received : ℕ := 24

-- Statement to be proved
theorem olivia_packs_of_basketball_cards (x : ℕ) (hx : pack_cost * x + deck_cost * number_of_decks = total_money - change_received) : x = 2 :=
by 
  sorry

end olivia_packs_of_basketball_cards_l109_109021


namespace four_p_minus_three_is_square_l109_109343

theorem four_p_minus_three_is_square
  (n : ℕ) (p : ℕ)
  (hn_pos : n > 1)
  (hp_prime : Prime p)
  (h1 : n ∣ (p - 1))
  (h2 : p ∣ (n^3 - 1)) : ∃ k : ℕ, 4 * p - 3 = k^2 := sorry

end four_p_minus_three_is_square_l109_109343


namespace sum_of_squares_eq_2_l109_109619

theorem sum_of_squares_eq_2 (a b : ℝ) 
  (h : (a^2 + b^2) * (a^2 + b^2 + 4) = 12) : a^2 + b^2 = 2 :=
by sorry

end sum_of_squares_eq_2_l109_109619


namespace meeting_2015th_at_C_l109_109909

-- Conditions Definitions
variable (A B C D P : Type)
variable (x y t : ℝ)  -- speeds and starting time difference
variable (mw cyclist : ℝ → ℝ)  -- paths of motorist and cyclist

-- Proof statement
theorem meeting_2015th_at_C 
(Given_meeting_pattern: ∀ n : ℕ, odd n → (mw (n * (x + y))) = C):
  (mw (2015 * (x + y))) = C := 
by 
  sorry  -- Proof omitted

end meeting_2015th_at_C_l109_109909


namespace remainder_444_pow_444_mod_13_l109_109703

theorem remainder_444_pow_444_mod_13 :
  (444 ^ 444) % 13 = 1 :=
by
  have h1 : 444 % 13 = 2 := by norm_num
  have h2 : 2 ^ 12 % 13 = 1 := by norm_num
  rw [←pow_mul, ←nat.mul_div_cancel_left 444 12] at h1
  have h3 : 444 = 12 * (444 / 12) := nat.mul_div_cancel_left 444 12
  rw h3
  rw pow_mul
  rw h2
  rw pow_one
  exact h2

end remainder_444_pow_444_mod_13_l109_109703


namespace bags_with_chocolate_hearts_l109_109503

-- Definitions for given conditions
def total_candies : ℕ := 63
def total_bags : ℕ := 9
def candies_per_bag : ℕ := total_candies / total_bags
def chocolate_kiss_bags : ℕ := 3
def not_chocolate_candies : ℕ := 28
def bags_not_chocolate : ℕ := not_chocolate_candies / candies_per_bag
def remaining_bags : ℕ := total_bags - chocolate_kiss_bags - bags_not_chocolate

-- Statement to be proved
theorem bags_with_chocolate_hearts :
  remaining_bags = 2 := by 
  sorry

end bags_with_chocolate_hearts_l109_109503


namespace total_cost_is_58_l109_109244

-- Define the conditions
def cost_per_adult : Nat := 22
def cost_per_child : Nat := 7
def number_of_adults : Nat := 2
def number_of_children : Nat := 2

-- Define the theorem to prove the total cost
theorem total_cost_is_58 : number_of_adults * cost_per_adult + number_of_children * cost_per_child = 58 :=
by
  -- Steps of proof will go here
  sorry

end total_cost_is_58_l109_109244


namespace find_y_l109_109133

theorem find_y (y : ℝ) (h1 : ∠ABC = 90) (h2 : ∠ABD = 3 * y) (h3 : ∠DBC = 2 * y) : y = 18 := by
  sorry

end find_y_l109_109133


namespace arithmetic_sequence_sum_l109_109603

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S_n : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a n = a 0 + n * d) 
  (h2 : ∀ n, S_n n = n * (a 0 + a n) / 2) 
  (h3 : 2 * a 6 = 5 + a 8) :
  S_n 9 = 45 := 
by 
  sorry

end arithmetic_sequence_sum_l109_109603


namespace flour_needed_for_two_loaves_l109_109514

-- Define the amount of flour needed for one loaf.
def flour_per_loaf : ℝ := 2.5

-- Define the number of loaves.
def number_of_loaves : ℕ := 2

-- Define the total amount of flour needed for the given number of loaves.
def total_flour_needed : ℝ := flour_per_loaf * number_of_loaves

-- The theorem statement: Prove that the total amount of flour needed is 5 cups.
theorem flour_needed_for_two_loaves : total_flour_needed = 5 := by
  sorry

end flour_needed_for_two_loaves_l109_109514


namespace probability_exactly_five_blue_marbles_l109_109957

noncomputable def prob_five_blue_marbles : ℚ :=
  (nat.choose 8 5) * ((2/3)^5) * ((1/3)^3)

theorem probability_exactly_five_blue_marbles :
  (prob_five_blue_marbles).to_real ≈ 0.272 :=
begin
  sorry
end

end probability_exactly_five_blue_marbles_l109_109957


namespace find_g1_l109_109108

variables {f g : ℝ → ℝ}

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem find_g1 (hf : odd_function f)
                (hg : even_function g)
                (h1 : f (-1) + g 1 = 2)
                (h2 : f 1 + g (-1) = 4) :
                g 1 = 3 :=
sorry

end find_g1_l109_109108


namespace back_seat_people_l109_109804

/-- Define the number of seats on the left side of the bus --/
def left_side_seats : ℕ := 15

/-- Define the number of seats on the right side of the bus (3 fewer because of the rear exit door) --/
def right_side_seats : ℕ := left_side_seats - 3

/-- Define the number of people each seat can hold --/
def people_per_seat : ℕ := 3

/-- Define the total capacity of the bus --/
def total_capacity : ℕ := 90

/-- Define the total number of people that can sit on the regular seats (left and right sides) --/
def regular_seats_people := (left_side_seats + right_side_seats) * people_per_seat

/-- Theorem stating the number of people that can sit at the back seat --/
theorem back_seat_people : (total_capacity - regular_seats_people) = 9 := by
  sorry

end back_seat_people_l109_109804


namespace evaluate_expression_l109_109889

theorem evaluate_expression : 
  (196 * (1 / 17 - 1 / 21) + 361 * (1 / 21 - 1 / 13) + 529 * (1 / 13 - 1 / 17)) /
    (14 * (1 / 17 - 1 / 21) + 19 * (1 / 21 - 1 / 13) + 23 * (1 / 13 - 1 / 17)) = 56 :=
by
  sorry

end evaluate_expression_l109_109889


namespace solution_l109_109434

-- Define the vectors and their conditions
variables {u v : ℝ}

def vec1 := (3, -2)
def vec2 := (9, -7)
def vec3 := (-1, 2)
def vec4 := (-3, 4)

-- Condition: The linear combination of vec1 and u*vec2 equals the linear combination of vec3 and v*vec4.
axiom H : (3 + 9 * u, -2 - 7 * u) = (-1 - 3 * v, 2 + 4 * v)

-- Statement of the proof problem:
theorem solution : u = -4/15 ∧ v = -8/15 :=
by {
  sorry
}

end solution_l109_109434


namespace meeting_point_2015th_l109_109910

-- Definitions for the conditions
def motorist_speed (x : ℝ) := x
def cyclist_speed (y : ℝ) := y
def initial_delay (t : ℝ) := t 
def first_meeting_point := C
def second_meeting_point := D

-- The main proof problem statement
theorem meeting_point_2015th
  (x y t : ℝ) -- speeds of the motorist and cyclist and the initial delay
  (C D : Point) -- points C and D on the segment AB where meetings occur
  (pattern_alternation : ∀ n: ℤ, n > 0 → ((n % 2 = 1) → n-th_meeting_point = C) ∧ ((n % 2 = 0) → n-th_meeting_point = D))
  (P_A_B_cycle : ∀ n: ℕ, (P → A ∨ B → C ∨ A → B ∨ D → P) holds for each meeting): 
  2015-th_meeting_point = C :=
by
  sorry

end meeting_point_2015th_l109_109910


namespace person_time_to_walk_without_walkway_l109_109869

def time_to_walk_without_walkway 
  (walkway_length : ℝ) 
  (time_with_walkway : ℝ) 
  (time_against_walkway : ℝ) 
  (correct_time : ℝ) : Prop :=
  ∃ (vp vw : ℝ), 
    ((vp + vw) * time_with_walkway = walkway_length) ∧ 
    ((vp - vw) * time_against_walkway = walkway_length) ∧ 
     correct_time = walkway_length / vp

theorem person_time_to_walk_without_walkway : 
  time_to_walk_without_walkway 120 40 160 64 :=
sorry

end person_time_to_walk_without_walkway_l109_109869


namespace digit_150th_in_decimal_of_fraction_l109_109219

theorem digit_150th_in_decimal_of_fraction : 
  (∀ n : ℕ, n > 0 → (let seq := [2, 4, 2, 8, 5, 7] in seq[((n - 1) % 6)] = 
  if n == 150 then 7 else seq[((n - 1) % 6)])) :=
by
  sorry

end digit_150th_in_decimal_of_fraction_l109_109219


namespace transaction_gain_per_year_l109_109253

theorem transaction_gain_per_year
  (principal : ℝ) (borrow_rate : ℝ) (lend_rate : ℝ) (time : ℕ)
  (principal_eq : principal = 5000)
  (borrow_rate_eq : borrow_rate = 0.04)
  (lend_rate_eq : lend_rate = 0.06)
  (time_eq : time = 2) :
  (principal * lend_rate * time - principal * borrow_rate * time) / time = 100 := by
  sorry

end transaction_gain_per_year_l109_109253


namespace meeting_point_2015_l109_109912

/-- 
A motorist starts at point A, and a cyclist starts at point B. They travel towards each other and 
meet for the first time at point C. After meeting, they turn around and travel back to their starting 
points and continue this pattern of meeting, turning around, and traveling back to their starting points. 
Prove that their 2015th meeting point is at point C.
-/
theorem meeting_point_2015 
  (A B C D : Type) 
  (x y t : ℕ)
  (odd_meeting : ∀ n : ℕ, (2 * n + 1) % 2 = 1) : 
  ∃ n, (n = 2015) → odd_meeting n = 1 → (n % 2 = 1 → (C = "C")) := 
sorry

end meeting_point_2015_l109_109912


namespace probability_at_least_one_multiple_of_3_l109_109421

/-- Ben twice chooses a random integer between 1 and 50, inclusive, 
and he may choose the same integer both times. 
The probability that at least one of the numbers Ben chooses is a multiple of 3 is 336/625 -/
theorem probability_at_least_one_multiple_of_3 : 
  (2 * 34 / 50) ^ 2 + (2 * 16 / 50) ^ 2 = 336 / 625 :=
sorry

end probability_at_least_one_multiple_of_3_l109_109421


namespace number_of_possible_values_of_a_l109_109353

noncomputable def problem_statement : Prop :=
  ∃ (a b c d : ℕ),
  a > b ∧ b > c ∧ c > d ∧
  a + b + c + d = 2040 ∧
  a^2 - b^2 + c^2 - d^2 = 2040 ∧
  508 ∈ {a | ∃ b c d, a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 2040 ∧ a^2 - b^2 + c^2 - d^2 = 2040}

theorem number_of_possible_values_of_a : problem_statement :=
  sorry

end number_of_possible_values_of_a_l109_109353


namespace divisible_by_5_l109_109449

-- Problem statement: For which values of \( x \) is \( 2^x - 1 \) divisible by \( 5 \)?
-- Equivalent Proof Problem in Lean 4.

theorem divisible_by_5 (x : ℕ) : 
  (∃ t : ℕ, x = 6 * t + 1) ∨ (∃ t : ℕ, x = 6 * t + 4) ↔ (5 ∣ (2^x - 1)) :=
by sorry

end divisible_by_5_l109_109449


namespace triangle_problem_l109_109128

-- Defining the conditions as Lean constructs
variable (a c : ℝ)
variable (b : ℝ := 3)
variable (cosB : ℝ := 1 / 3)
variable (dotProductBACBC : ℝ := 2)
variable (cosB_minus_C : ℝ := 23 / 27)

-- Define the problem as a theorem in Lean 4
theorem triangle_problem
  (h1 : a > c)
  (h2 : a * c * cosB = dotProductBACBC)
  (h3 : a^2 + c^2 = 13) :
  a = 3 ∧ c = 2 ∧ cosB_minus_C = 23 / 27 := by
  sorry

end triangle_problem_l109_109128


namespace max_ab_value_l109_109462

noncomputable def max_ab (a b : ℝ) : ℝ := a * b

theorem max_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 8) : max_ab a b ≤ 4 :=
by
  sorry

end max_ab_value_l109_109462


namespace pizza_payment_difference_l109_109573

theorem pizza_payment_difference
  (total_slices : ℕ := 12)
  (plain_cost : ℝ := 12)
  (onion_cost : ℝ := 3)
  (jack_onion_slices : ℕ := 4)
  (jack_plain_slices : ℕ := 3)
  (carl_plain_slices : ℕ := 5) :
  let total_cost := plain_cost + onion_cost
  let cost_per_slice := total_cost / total_slices
  let jack_onion_payment := jack_onion_slices * cost_per_slice
  let jack_plain_payment := jack_plain_slices * cost_per_slice
  let jack_total_payment := jack_onion_payment + jack_plain_payment
  let carl_total_payment := carl_plain_slices * cost_per_slice
  jack_total_payment - carl_total_payment = 2.5 :=
by
  sorry

end pizza_payment_difference_l109_109573


namespace heat_required_l109_109173

theorem heat_required (m : ℝ) (c₀ : ℝ) (alpha : ℝ) (t₁ t₂ : ℝ) :
  m = 2 ∧ c₀ = 150 ∧ alpha = 0.05 ∧ t₁ = 20 ∧ t₂ = 100 →
  let Δt := t₂ - t₁
  let c_avg := (c₀ * (1 + alpha * t₁) + c₀ * (1 + alpha * t₂)) / 2
  let Q := c_avg * m * Δt
  Q = 96000 := by
  sorry

end heat_required_l109_109173


namespace find_number_l109_109056

theorem find_number (x : ℝ) : (x * 12) / (180 / 3) + 80 = 81 → x = 5 :=
by
  sorry

end find_number_l109_109056


namespace greatest_possible_sum_of_10_integers_l109_109684

theorem greatest_possible_sum_of_10_integers (a b c d e f g h i j : ℕ) 
  (h_prod : a * b * c * d * e * f * g * h * i * j = 1024) : 
  a + b + c + d + e + f + g + h + i + j ≤ 1033 :=
sorry

end greatest_possible_sum_of_10_integers_l109_109684


namespace parallel_lines_l109_109469

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x - a * y + 1 = 0) ∧ (∀ x y : ℝ, (a-1) * x - y + a = 0) →
  (a = 2 ↔ (∀ x1 y1 x2 y2 : ℝ, 2 * x1 - a * y1 + 1 = 0 ∧ (a-1) * x2 - y2 + a = 0 →
  (2 * x1 = (a * y1 - 1) ∧ (a-1) * x2 = y2 - a))) :=
sorry

end parallel_lines_l109_109469


namespace find_k_l109_109886

theorem find_k (x k : ℝ) :
  (∀ x, x ∈ Set.Ioo (-4 : ℝ) 3 ↔ x * (x^2 - 9) < k) → k = 0 :=
  by
  sorry

end find_k_l109_109886


namespace field_trip_students_l109_109526

theorem field_trip_students (bus_cost admission_per_student budget : ℕ) (students : ℕ)
  (h1 : bus_cost = 100)
  (h2 : admission_per_student = 10)
  (h3 : budget = 350)
  (total_cost : students * admission_per_student + bus_cost ≤ budget) : 
  students = 25 :=
by
  sorry

end field_trip_students_l109_109526


namespace not_divisible_by_97_l109_109624

theorem not_divisible_by_97 (k : ℤ) (h : k ∣ (99^3 - 99)) : k ≠ 97 :=
sorry

end not_divisible_by_97_l109_109624


namespace final_books_is_correct_l109_109986

def initial_books : ℝ := 35.5
def books_bought : ℝ := 12.3
def books_given_to_friends : ℝ := 7.2
def books_donated : ℝ := 20.8

theorem final_books_is_correct :
  (initial_books + books_bought - books_given_to_friends - books_donated) = 19.8 := by
  sorry

end final_books_is_correct_l109_109986


namespace pascal_triangle_fifth_number_twentieth_row_l109_109885

theorem pascal_triangle_fifth_number_twentieth_row : 
  (Nat.choose 20 4) = 4845 :=
by
  sorry

end pascal_triangle_fifth_number_twentieth_row_l109_109885


namespace sequence_formulas_range_of_k_l109_109033

variable {a b : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable {k : ℝ}

-- (1) Prove the general formulas for {a_n} and {b_n}
theorem sequence_formulas (h1 : ∀ n, a n + b n = 2 * n - 1)
  (h2 : ∀ n, S n = 2 * n^2 - n)
  (hS : ∀ n, a (n + 1) = S (n + 1) - S n)
  (hS1 : a 1 = S 1) :
  (∀ n, a n = 4 * n - 3) ∧ (∀ n, b n = -2 * n + 2) :=
sorry

-- (2) Prove the range of k
theorem range_of_k (h3 : ∀ n, a n = k * 2^(n - 1))
  (h4 : ∀ n, b n = 2 * n - 1 - k * 2^(n - 1))
  (h5 : ∀ n, b (n + 1) < b n) :
  k > 2 :=
sorry

end sequence_formulas_range_of_k_l109_109033


namespace quadratic_solution_sum_l109_109537

theorem quadratic_solution_sum
  (x : ℚ)
  (m n p : ℕ)
  (h_eq : (5 * x - 11) * x = -6)
  (h_form : ∃ m n p, x = (m + Real.sqrt n) / p ∧ x = (m - Real.sqrt n) / p)
  (h_gcd : Nat.gcd (Nat.gcd m n) p = 1) :
  m + n + p = 22 := 
sorry

end quadratic_solution_sum_l109_109537


namespace largest_value_l109_109230

def expr_A : ℕ := 3 + 1 + 0 + 5
def expr_B : ℕ := 3 * 1 + 0 + 5
def expr_C : ℕ := 3 + 1 * 0 + 5
def expr_D : ℕ := 3 * 1 + 0 * 5
def expr_E : ℕ := 3 * 1 + 0 * 5 * 3

theorem largest_value :
  expr_A > expr_B ∧
  expr_A > expr_C ∧
  expr_A > expr_D ∧
  expr_A > expr_E :=
by
  sorry

end largest_value_l109_109230


namespace mangoes_total_l109_109417

theorem mangoes_total (Dilan Ashley Alexis : ℕ) (h1 : Alexis = 4 * (Dilan + Ashley)) (h2 : Ashley = 2 * Dilan) (h3 : Alexis = 60) : Dilan + Ashley + Alexis = 75 :=
by
  sorry

end mangoes_total_l109_109417


namespace EFGH_perimeter_l109_109959

noncomputable def perimeter_rectangle_EFGH (WE EX WY XZ : ℕ) : Rat :=
  let WX := Real.sqrt (WE ^ 2 + EX ^ 2)
  let p := 15232
  let q := 100
  p / q

theorem EFGH_perimeter :
  let WE := 12
  let EX := 16
  let WY := 24
  let XZ := 32
  perimeter_rectangle_EFGH WE EX WY XZ = 15232 / 100 :=
by
  sorry

end EFGH_perimeter_l109_109959


namespace solve_equation_l109_109055

theorem solve_equation (x : ℝ) (h1: (6 * x) ^ 18 = (12 * x) ^ 9) (h2 : x ≠ 0) : x = 1 / 3 := by
  sorry

end solve_equation_l109_109055


namespace mitigate_bank_profit_loss_l109_109235

-- We define the terminology and conditions described in the problem
def bank_suboptimal_cashback_behavior (customer_behavior: ℕ → ℕ) : Prop :=
  ∀ (i : ℕ), (customer_behavior i) = category_specific_cashback i ∧ is_financially_savvy customer_behavior

def category_specific_cashback (i : ℕ) : ℕ := -- categorical cashback logic
  sorry 

def is_financially_savvy (customer_behavior: ℕ → ℕ) : Prop :=
  ∃ (cards : List ℕ), ∀ (i : ℕ), customer_behavior i = max_cashback_for_category i cards

def max_cashback_for_category (i : ℕ) (cards : List ℕ) : ℕ := 
  -- Define the logic for max cashback possible using multiple cards
  sorry

def dynamic_cashback_rates (total_cashback: ℕ) : ℕ := 
  -- Logic for dynamic cashback rates decreasing as total cashback increases
  sorry

def categorical_caps_rotating (i : ℕ) (period: ℕ) : ℕ := 
  -- Logic for rotating cashback caps for categories
  sorry

-- Lean statement that banks can avoid problems due to financially savvy customer behavior
theorem mitigate_bank_profit_loss (customer_behavior: ℕ → ℕ)
  (H : bank_suboptimal_cashback_behavior customer_behavior)
  : (∃ f : ℕ → ℕ, f = dynamic_cashback_rates) ∨ (∃ g : ℕ → ℕ × ℕ, g = categorical_caps_rotating) :=
  sorry

end mitigate_bank_profit_loss_l109_109235


namespace stock_worth_is_100_l109_109412

-- Define the number of puppies and kittens
def num_puppies : ℕ := 2
def num_kittens : ℕ := 4

-- Define the cost per puppy and kitten
def cost_per_puppy : ℕ := 20
def cost_per_kitten : ℕ := 15

-- Define the total stock worth function
def stock_worth (num_puppies num_kittens cost_per_puppy cost_per_kitten : ℕ) : ℕ :=
  (num_puppies * cost_per_puppy) + (num_kittens * cost_per_kitten)

-- The theorem to prove that the stock worth is $100
theorem stock_worth_is_100 :
  stock_worth num_puppies num_kittens cost_per_puppy cost_per_kitten = 100 :=
by
  sorry

end stock_worth_is_100_l109_109412


namespace triangle_area_40_l109_109747

noncomputable def area_of_triangle (base height : ℕ) : ℕ :=
  base * height / 2

theorem triangle_area_40
  (a : ℕ) (P B Q : (ℕ × ℕ)) (PB_side : (P.1 = 0 ∧ P.2 = 0) ∧ (B.1 = 10 ∧ B.2 = 0))
  (Q_vert_aboveP : Q.1 = 0 ∧ Q.2 = 8)
  (PQ_perp_PB : P.1 = Q.1)
  (PQ_length : (Q.snd - P.snd) = 8) :
  area_of_triangle 10 8 = 40 := by
  sorry

end triangle_area_40_l109_109747


namespace hula_hoop_radius_l109_109852

theorem hula_hoop_radius (d : ℝ) (hd : d = 14) : d / 2 = 7 :=
by
  rw [hd]
  norm_num

end hula_hoop_radius_l109_109852


namespace train_speed_including_stoppages_l109_109583

noncomputable def trainSpeedExcludingStoppages : ℝ := 45
noncomputable def stoppageTimePerHour : ℝ := 20 / 60 -- 20 minutes per hour converted to hours
noncomputable def runningTimePerHour : ℝ := 1 - stoppageTimePerHour

theorem train_speed_including_stoppages (speed : ℝ) (stoppage : ℝ) (running_time : ℝ) : 
  speed = 45 → stoppage = 20 / 60 → running_time = 1 - stoppage → 
  (speed * running_time) / 1 = 30 :=
by sorry

end train_speed_including_stoppages_l109_109583


namespace second_polygon_sides_l109_109042

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0) : 
  ∀ (P1 P2 : ℝ) (n1 n2 : ℕ), 
    n1 = 50 →
    P1 = n1 * (3 * s) →
    P2 = n2 * s →
    P1 = P2 →
    n2 = 150 :=
by
  intros P1 P2 n1 n2 hn1 hp1 hp2 hp
  sorry

end second_polygon_sides_l109_109042


namespace find_missing_part_l109_109857

variable (x y : ℚ) -- Using rationals as the base field for generality.

theorem find_missing_part :
  2 * x * (-3 * x^2 * y) = -6 * x^3 * y := 
by
  sorry

end find_missing_part_l109_109857


namespace xiao_pang_xiao_ya_books_l109_109728

theorem xiao_pang_xiao_ya_books : 
  ∀ (x y : ℕ), 
    (x + 2 * x = 66) → 
    (y + y / 3 = 92) → 
    (2 * x = 2 * x) → 
    (y = 3 * (y / 3)) → 
    ((22 + 69) - (2 * 22 + 69 / 3) = 24) :=
by
  intros x y h1 h2 h3 h4
  sorry

end xiao_pang_xiao_ya_books_l109_109728


namespace equal_dice_probability_l109_109238

noncomputable def probability_equal_one_and_two_digit (dice : Fin 5 → Fin 20) : ℚ :=
  let one_digit_prob := 9 / 20
  let two_digit_prob := 11 / 20
  let comb_factor : ℚ := Nat.choose 5 2
  let individual_prob : ℚ := (one_digit_prob ^ 2) * (two_digit_prob ^ 3)
  comb_factor * individual_prob

theorem equal_dice_probability :
  probability_equal_one_and_two_digit = 539055 / 1600000 :=
by
  sorry

end equal_dice_probability_l109_109238


namespace minimum_value_of_expression_l109_109591

theorem minimum_value_of_expression (x : ℝ) (hx : x > 0) : 6 * x + 1 / x ^ 6 ≥ 7 :=
sorry

end minimum_value_of_expression_l109_109591


namespace cube_diagonal_length_l109_109866

theorem cube_diagonal_length (V A : ℝ) (hV : V = 384) (hA : A = 384) : 
  ∃ d : ℝ, d = 8 * Real.sqrt 3 :=
by
  sorry

end cube_diagonal_length_l109_109866


namespace probability_of_sum_18_when_four_dice_rolled_l109_109795

noncomputable def probability_sum_18 : ℝ :=
  sorry -- This is a placeholder for the actual calculation

theorem probability_of_sum_18_when_four_dice_rolled :
  probability_sum_18 = 1/72 :=
begin
  sorry -- This is a placeholder for the actual proof
end

end probability_of_sum_18_when_four_dice_rolled_l109_109795


namespace solve_z_pow_eq_neg_sixteen_l109_109091

theorem solve_z_pow_eq_neg_sixteen (z : ℂ) :
  z^4 = -16 ↔ 
  z = complex.sqrt(2) + complex.I * complex.sqrt(2) ∨ 
  z = -complex.sqrt(2) + complex.I * complex.sqrt(2) ∨ 
  z = -complex.sqrt(2) - complex.I * complex.sqrt(2) ∨ 
  z = complex.sqrt(2) - complex.I * complex.sqrt(2) :=
by
  sorry

end solve_z_pow_eq_neg_sixteen_l109_109091


namespace specially_monotonous_count_l109_109577

open Finset

def is_special_monotonous (n : ℕ) : Prop :=
  if n ≤ 8 then True
  else let digits := nat.digits 10 n in
       (∀ i j, i < j ∧ j < digits.length → digits.nth i < digits.nth j) ∨
       (∀ i j, i < j ∧ j < digits.length → digits.nth i > digits.nth j)

theorem specially_monotonous_count : 
  {n : ℕ | is_special_monotonous n}.to_finset.card = 193 := 
sorry

end specially_monotonous_count_l109_109577


namespace jared_popcorn_l109_109384

-- Define the given conditions
def pieces_per_serving := 30
def number_of_friends := 3
def pieces_per_friend := 60
def servings_ordered := 9

-- Define the total pieces of popcorn
def total_pieces := servings_ordered * pieces_per_serving

-- Define the total pieces of popcorn eaten by Jared's friends
def friends_total_pieces := number_of_friends * pieces_per_friend

-- State the theorem
theorem jared_popcorn : total_pieces - friends_total_pieces = 90 :=
by 
  -- The detailed proof would go here.
  sorry

end jared_popcorn_l109_109384


namespace tree_height_l109_109029

theorem tree_height (future_height : ℕ) (growth_per_year : ℕ) (years : ℕ) (inches_per_foot : ℕ) :
  future_height = 1104 →
  growth_per_year = 5 →
  years = 8 →
  inches_per_foot = 12 →
  (future_height / inches_per_foot - growth_per_year * years) = 52 := 
by
  intros h1 h2 h3 h4
  sorry

end tree_height_l109_109029


namespace new_acute_angle_ACB_l109_109535

-- Define the initial condition: the measure of angle ACB is 50 degrees.
def measure_ACB_initial : ℝ := 50

-- Define the rotation: ray CA is rotated by 540 degrees clockwise.
def rotation_CW_degrees : ℝ := 540

-- Theorem statement: The positive measure of the new acute angle ACB.
theorem new_acute_angle_ACB : 
  ∃ (new_angle : ℝ), new_angle = 50 ∧ new_angle < 90 := 
by
  sorry

end new_acute_angle_ACB_l109_109535


namespace abs_eq_zero_sum_is_neg_two_l109_109626

theorem abs_eq_zero_sum_is_neg_two (x y : ℝ) (h : |x - 1| + |y + 3| = 0) : x + y = -2 := 
by 
  sorry

end abs_eq_zero_sum_is_neg_two_l109_109626


namespace simplify_and_evaluate_l109_109520

theorem simplify_and_evaluate :
  ∀ (x : ℝ), x = -3 → 7 * x^2 - 3 * (2 * x^2 - 1) - 4 = 8 :=
by
  intros x hx
  rw [hx]
  sorry

end simplify_and_evaluate_l109_109520


namespace proof_problem_l109_109525

theorem proof_problem (a b : ℤ) (h1 : ∃ k, a = 5 * k) (h2 : ∃ m, b = 10 * m) :
  (∃ n, b = 5 * n) ∧ (∃ p, a - b = 5 * p) :=
by
  sorry

end proof_problem_l109_109525


namespace find_a_l109_109763

-- Define the main inequality condition
def inequality_condition (a x : ℝ) : Prop := |x^2 + a * x + 4 * a| ≤ 3

-- Define the condition that there is exactly one solution to the inequality
def has_exactly_one_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, (inequality_condition a x) ∧ (∀ y : ℝ, x ≠ y → ¬(inequality_condition a y))

-- The theorem that states the specific values of a
theorem find_a (a : ℝ) : has_exactly_one_solution a ↔ a = 8 + 2 * Real.sqrt 13 ∨ a = 8 - 2 * Real.sqrt 13 := 
by
  sorry

end find_a_l109_109763


namespace score_difference_proof_l109_109597

variable (α β γ δ : ℝ)

theorem score_difference_proof
  (h1 : α + β = γ + δ + 17)
  (h2 : α = β - 4)
  (h3 : γ = δ + 5) :
  β - δ = 13 :=
by
  -- proof goes here
  sorry

end score_difference_proof_l109_109597


namespace value_of_x_l109_109993

theorem value_of_x (x : ℤ) : (3000 + x) ^ 2 = x ^ 2 → x = -1500 := 
by
  sorry

end value_of_x_l109_109993


namespace total_splash_width_l109_109544

theorem total_splash_width :
  let pebbles_splash_width := 1 / 4
  let rocks_splash_width := 1 / 2
  let boulders_splash_width := 2
  let num_pebbles := 6
  let num_rocks := 3
  let num_boulders := 2
  let total_width := num_pebbles * pebbles_splash_width + num_rocks * rocks_splash_width + num_boulders * boulders_splash_width
  in total_width = 7 := by
  sorry

end total_splash_width_l109_109544


namespace remainder_444_power_444_mod_13_l109_109707

theorem remainder_444_power_444_mod_13 : (444 ^ 444) % 13 = 1 := by
  sorry

end remainder_444_power_444_mod_13_l109_109707


namespace total_food_in_10_days_l109_109003

theorem total_food_in_10_days :
  (let ella_food_per_day := 20
   let days := 10
   let dog_food_ratio := 4
   let ella_total_food := ella_food_per_day * days
   let dog_total_food := dog_food_ratio * ella_total_food
   ella_total_food + dog_total_food = 1000) :=
by
  sorry

end total_food_in_10_days_l109_109003


namespace solve_for_x_l109_109481

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 * x + 7 * x = 120) : x = 20 := 
by
  sorry

end solve_for_x_l109_109481


namespace inscribed_circle_radius_l109_109101

theorem inscribed_circle_radius :
  ∀ (r : ℝ), 
    (∀ (R : ℝ), R = 12 →
      (∀ (d : ℝ), d = 12 → r = 3)) :=
by sorry

end inscribed_circle_radius_l109_109101


namespace range_of_a_l109_109532

-- Define the quadratic inequality
def quadratic_inequality (a x : ℝ) : ℝ := (a-1)*x^2 + (a-1)*x + 1

theorem range_of_a :
  (∀ x : ℝ, quadratic_inequality a x > 0) ↔ (1 ≤ a ∧ a < 5) :=
by
  sorry

end range_of_a_l109_109532


namespace range_of_a_l109_109123

theorem range_of_a (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (a - 2)^x₁ > (a - 2)^x₂) → (2 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l109_109123


namespace sum_of_powers_of_two_l109_109882

theorem sum_of_powers_of_two : 2^4 + 2^4 + 2^4 + 2^4 = 2^6 :=
by
  sorry

end sum_of_powers_of_two_l109_109882


namespace quadrilateral_area_l109_109989

def vertex1 : ℝ × ℝ := (2, 1)
def vertex2 : ℝ × ℝ := (4, 3)
def vertex3 : ℝ × ℝ := (7, 1)
def vertex4 : ℝ × ℝ := (4, 6)

noncomputable def shoelace_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  abs ((v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v4.2 + v4.1 * v1.2) -
       (v1.2 * v2.1 + v2.2 * v3.1 + v3.2 * v4.1 + v4.2 * v1.1)) / 2

theorem quadrilateral_area :
  shoelace_area vertex1 vertex2 vertex3 vertex4 = 7.5 :=
by
  sorry

end quadrilateral_area_l109_109989


namespace largest_common_divisor_l109_109073

theorem largest_common_divisor (d h m s : ℕ) : 
  40 ∣ (1000000 * d + 10000 * h + 100 * m + s - (86400 * d + 3600 * h + 60 * m + s)) :=
by
  sorry

end largest_common_divisor_l109_109073


namespace problem1_problem2_l109_109103

variables (x y : ℝ)

-- Given Conditions
def given_conditions :=
  (x = 2 + Real.sqrt 3) ∧ (y = 2 - Real.sqrt 3)

-- Problem 1
theorem problem1 (h : given_conditions x y) : x^2 + y^2 = 14 :=
sorry

-- Problem 2
theorem problem2 (h : given_conditions x y) : (x / y) - (y / x) = 8 * Real.sqrt 3 :=
sorry

end problem1_problem2_l109_109103


namespace total_cost_is_9_43_l109_109339

def basketball_game_cost : ℝ := 5.20
def racing_game_cost : ℝ := 4.23
def total_cost : ℝ := basketball_game_cost + racing_game_cost

theorem total_cost_is_9_43 : total_cost = 9.43 := by
  sorry

end total_cost_is_9_43_l109_109339


namespace completing_square_solution_l109_109365

theorem completing_square_solution (x : ℝ) : x^2 + 4 * x - 1 = 0 → (x + 2)^2 = 5 :=
by
  sorry

end completing_square_solution_l109_109365


namespace terminating_fraction_count_l109_109904

theorem terminating_fraction_count :
  (∃ n_values : Finset ℕ, (∀ n ∈ n_values, 1 ≤ n ∧ n ≤ 500 ∧ (∃ k : ℕ, n = k * 49)) ∧ n_values.card = 10) :=
by
  -- Placeholder for the proof, does not contribute to the conditions-direct definitions.
  sorry

end terminating_fraction_count_l109_109904


namespace sin_70_eq_1_minus_2k_squared_l109_109916

theorem sin_70_eq_1_minus_2k_squared (k : ℝ) (h : Real.sin (10 * Real.pi / 180) = k) :
  Real.sin (70 * Real.pi / 180) = 1 - 2 * k^2 :=
by
  sorry

end sin_70_eq_1_minus_2k_squared_l109_109916


namespace meeting_point_2015_is_C_l109_109908

-- Given definitions based on conditions
variable (x y : ℝ) -- Speeds of the motorcycle and the cyclist
variable (A B C D : Point) -- Points on segment AB
variable (meetings : ℕ → Point) -- Function representing the meeting point sequence

-- Conditions stating the alternating meeting pattern
axiom start_at_A (n : ℕ) : meetings (2 * n + 1) = C
axiom start_at_B (n : ℕ) : meetings (2 * n + 2) = D

-- The theorem statement to be proved
theorem meeting_point_2015_is_C : meetings 2015 = C := sorry

end meeting_point_2015_is_C_l109_109908


namespace polynomial_root_fraction_l109_109883

theorem polynomial_root_fraction (p q r s : ℝ) (h : p ≠ 0) 
    (h1 : p * 4^3 + q * 4^2 + r * 4 + s = 0)
    (h2 : p * (-3)^3 + q * (-3)^2 + r * (-3) + s = 0) : 
    (q + r) / p = -13 :=
by
  sorry

end polynomial_root_fraction_l109_109883


namespace swimming_speed_eq_l109_109968

theorem swimming_speed_eq (S R H : ℝ) (h1 : R = 9) (h2 : H = 5) (h3 : H = (2 * S * R) / (S + R)) :
  S = 45 / 13 :=
by
  sorry

end swimming_speed_eq_l109_109968


namespace local_minimum_f_is_1_maximum_local_minimum_g_is_1_l109_109786

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) - Real.log x

def local_minimum_value_f := 1

theorem local_minimum_f_is_1 : 
  ∃ x0 : ℝ, x0 > 0 ∧ (∀ x > 0, f x0 ≤ f x) ∧ f x0 = local_minimum_value_f :=
sorry

noncomputable def g (a x : ℝ) : ℝ := f x - a * (x - 1)

def maximum_value_local_minimum_g := 1

theorem maximum_local_minimum_g_is_1 :
  ∃ a x0 : ℝ, a = 0 ∧ x0 > 0 ∧ (∀ x > 0, g a x0 ≤ g a x) ∧ g a x0 = maximum_value_local_minimum_g :=
sorry

end local_minimum_f_is_1_maximum_local_minimum_g_is_1_l109_109786


namespace coffee_price_l109_109419

theorem coffee_price (C : ℝ) :
  (7 * C) + (8 * 4) = 67 → C = 5 :=
by
  intro h
  sorry

end coffee_price_l109_109419


namespace cost_price_A_l109_109234

-- Establishing the definitions based on the conditions from a)

def profit_A_to_B (CP_A : ℝ) : ℝ := 1.20 * CP_A
def profit_B_to_C (CP_B : ℝ) : ℝ := 1.25 * CP_B
def price_paid_by_C : ℝ := 222

-- Stating the theorem to be proven:
theorem cost_price_A (CP_A : ℝ) (H : profit_B_to_C (profit_A_to_B CP_A) = price_paid_by_C) : CP_A = 148 :=
by 
  sorry

end cost_price_A_l109_109234


namespace mostSuitableSampleSurvey_l109_109727

-- Conditions
def conditionA := "Security check for passengers before boarding a plane"
def conditionB := "Understanding the amount of physical exercise each classmate does per week"
def conditionC := "Interviewing job applicants for a company's recruitment process"
def conditionD := "Understanding the lifespan of a batch of light bulbs"

-- Define a predicate to determine the most suitable for a sample survey
def isMostSuitableForSampleSurvey (s : String) : Prop :=
  s = conditionD

-- Theorem statement
theorem mostSuitableSampleSurvey :
  isMostSuitableForSampleSurvey conditionD :=
by
  -- Skipping the proof for now
  sorry

end mostSuitableSampleSurvey_l109_109727


namespace sum_of_ages_l109_109645

def Maria_age (E : ℕ) : ℕ := E + 7

theorem sum_of_ages (M E : ℕ) (h1 : M = E + 7) (h2 : M + 10 = 3 * (E - 5)) :
  M + E = 39 :=
by
  sorry

end sum_of_ages_l109_109645


namespace trip_assistant_cost_l109_109385

theorem trip_assistant_cost :
  (let 
    hours_one_way := 4
    hours_round_trip := hours_one_way * 2
    cost_per_hour := 10
    total_cost := hours_round_trip * cost_per_hour
  in 
    total_cost = 80) :=
by
  simp only []
  sorry

end trip_assistant_cost_l109_109385


namespace problem1_problem2_l109_109426

-- Problem 1
theorem problem1 : ∀ x : ℝ, 4 * x - 3 * (20 - x) + 4 = 0 → x = 8 :=
by
  intro x
  intro h
  sorry

-- Problem 2
theorem problem2 : ∀ x : ℝ, (2 * x + 1) / 3 = 1 - (x - 1) / 5 → x = 1 :=
by
  intro x
  intro h
  sorry

end problem1_problem2_l109_109426


namespace jason_stacked_bales_l109_109183

theorem jason_stacked_bales (initial_bales : ℕ) (final_bales : ℕ) (stored_bales : ℕ) 
  (h1 : initial_bales = 73) (h2 : final_bales = 96) : stored_bales = final_bales - initial_bales := 
by
  rw [h1, h2]
  sorry

end jason_stacked_bales_l109_109183


namespace no_even_threes_in_circle_l109_109570

theorem no_even_threes_in_circle (arr : ℕ → ℕ) (h1 : ∀ i, 1 ≤ arr i ∧ arr i ≤ 2017)
  (h2 : ∀ i, (arr i + arr ((i + 1) % 2017) + arr ((i + 2) % 2017)) % 2 = 0) : false :=
sorry

end no_even_threes_in_circle_l109_109570


namespace range_of_m_l109_109782

theorem range_of_m (m : ℝ) :
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ 2 ∧ 0 ≤ x2 ∧ x2 ≤ 2 → |((x2^2 - m * x2) - (x1^2 - m * x1))| ≤ 9) →
  -5 / 2 ≤ m ∧ m ≤ 13 / 2 :=
sorry

end range_of_m_l109_109782


namespace initial_number_of_observations_l109_109031

theorem initial_number_of_observations (n : ℕ) 
  (initial_mean : ℝ := 100) 
  (wrong_obs : ℝ := 75) 
  (corrected_obs : ℝ := 50) 
  (corrected_mean : ℝ := 99.075) 
  (h1 : (n:ℝ) * initial_mean = n * corrected_mean + wrong_obs - corrected_obs) 
  (h2 : n = (25 : ℝ) / 0.925) 
  : n = 27 := 
sorry

end initial_number_of_observations_l109_109031


namespace number_of_other_communities_correct_l109_109000

def total_students : ℕ := 1520
def percent_muslims : ℚ := 41 / 100
def percent_hindus : ℚ := 32 / 100
def percent_sikhs : ℚ := 12 / 100
def percent_other_communities : ℚ := 1 - (percent_muslims + percent_hindus + percent_sikhs)
def number_other_communities : ℤ := (percent_other_communities * total_students).nat_abs

theorem number_of_other_communities_correct :
  number_other_communities = 228 :=
by
  sorry

end number_of_other_communities_correct_l109_109000


namespace natural_numbers_partition_l109_109589

def isSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

def subsets_with_square_sum (n : ℕ) : Prop :=
  ∀ (A B : Finset ℕ), (A ∪ B = Finset.range (n + 1) ∧ A ∩ B = ∅) →
  ∃ (a b : ℕ), a ≠ b ∧ isSquare (a + b) ∧ (a ∈ A ∨ a ∈ B) ∧ (b ∈ A ∨ b ∈ B)

theorem natural_numbers_partition (n : ℕ) : n ≥ 15 → subsets_with_square_sum n := 
sorry

end natural_numbers_partition_l109_109589


namespace prod_mod_17_l109_109691

theorem prod_mod_17 : (1520 * 1521 * 1522) % 17 = 11 := sorry

end prod_mod_17_l109_109691


namespace sam_drove_distance_l109_109953

theorem sam_drove_distance (m_distance : ℕ) (m_time : ℕ) (s_time : ℕ) (s_distance : ℕ)
  (m_distance_eq : m_distance = 120) (m_time_eq : m_time = 3) (s_time_eq : s_time = 4) :
  s_distance = (m_distance / m_time) * s_time :=
by
  sorry

end sam_drove_distance_l109_109953


namespace total_surface_area_of_new_solid_l109_109252

-- Define the heights of the pieces using the given conditions
def height_A := 1 / 4
def height_B := 1 / 5
def height_C := 1 / 6
def height_D := 1 / 7
def height_E := 1 / 8
def height_F := 1 - (height_A + height_B + height_C + height_D + height_E)

-- Assembling the pieces back in reverse order (F to A), encapsulate the total surface area calculation
theorem total_surface_area_of_new_solid : 
  (2 * (1 : ℝ)) + (2 * (1 * 1 : ℝ)) + (2 * (1 * 1 : ℝ)) = 6 :=
by
  sorry

end total_surface_area_of_new_solid_l109_109252


namespace weight_of_one_serving_l109_109447

theorem weight_of_one_serving
  (total_servings : ℕ)
  (chicken_weight_pounds : ℝ)
  (stuffing_weight_ounces : ℝ)
  (ounces_per_pound : ℝ)
  (total_servings = 12)
  (chicken_weight_pounds = 4.5)
  (stuffing_weight_ounces = 24)
  (ounces_per_pound = 16) :
  (chicken_weight_pounds * ounces_per_pound + stuffing_weight_ounces) / total_servings = 8 :=
by
  sorry

end weight_of_one_serving_l109_109447


namespace domain_of_f_eq_R_l109_109629

noncomputable def f (x m : ℝ) : ℝ := (x - 4) / (m * x^2 + 4 * m * x + 3)

theorem domain_of_f_eq_R (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 4 * m * x + 3 ≠ 0) ↔ (0 ≤ m ∧ m < 3 / 4) :=
by
  sorry

end domain_of_f_eq_R_l109_109629


namespace least_perimeter_l109_109166

theorem least_perimeter (a b : ℕ) (ha : a = 36) (hb : b = 45) (c : ℕ) (hc1 : c > 9) (hc2 : c < 81) : 
  a + b + c = 91 :=
by
  -- Placeholder for proof
  sorry

end least_perimeter_l109_109166


namespace simplify_expression_l109_109357

open Real

-- Define the given expression as a function of x
noncomputable def given_expression (x : ℝ) (hx : 0 < x) : ℝ :=
  sqrt (2 * (1 + sqrt (1 + ( (x^4 - 1) / (2 * x^2) )^2)))

-- Define the expected simplified expression
noncomputable def expected_expression (x : ℝ) (hx : 0 < x) : ℝ :=
  (x^2 + 1) / x

-- Proof statement to verify the simplification
theorem simplify_expression (x : ℝ) (hx : 0 < x) :
  given_expression x hx = expected_expression x hx :=
sorry

end simplify_expression_l109_109357


namespace total_distance_traveled_eq_l109_109010

-- Define the conditions as speeds and times for each segment of Jeff's trip.
def speed1 : ℝ := 80
def time1 : ℝ := 6
def speed2 : ℝ := 60
def time2 : ℝ := 4
def speed3 : ℝ := 40
def time3 : ℝ := 2

-- Define the distance function given speed and time.
def distance (speed time : ℝ) : ℝ := speed * time

-- Calculate the individual distances for each segment.
def distance1 : ℝ := distance speed1 time1
def distance2 : ℝ := distance speed2 time2
def distance3 : ℝ := distance speed3 time3

-- State the proof problem to show that the total distance is 800 miles.
theorem total_distance_traveled_eq : distance1 + distance2 + distance3 = 800 :=
by
  -- Placeholder for actual proof
  sorry

end total_distance_traveled_eq_l109_109010


namespace problem1_problem2_l109_109607

noncomputable def f (a b c x : ℝ) := (a * x^2 + b * x + c) * Real.exp x

def problem1_statement (a : ℝ) : Prop :=
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → 
    a * x^2 + (a - 1) * x - a ≤ 0) → 0 ≤ a ∧ a ≤ 1

def problem2_statement (a m : ℝ) : Prop :=
  a = 0 → 
  (∃ m, ∀ (x : ℝ), 
    2 * ((1 - x) * Real.exp x) + 4 * x * Real.exp x ≥ m * x + 1 ∧ 
    m * x + 1 ≥ -x^2 + 4 * x + 1) → m = 4

theorem problem1 (a : ℝ) (h0 : f a (-(1 + a)) 1 0 = 1) (h1 : f a (-(1 + a)) 1 1 = 0) : 
  problem1_statement a := 
begin
  intros, 
  sorry
end

theorem problem2 (a : ℝ) (h0 : f 0 (-1) 1 0 = 1) (h1 : f 0 (-1) 1 1 = 0) : 
  problem2_statement 0 4 := 
begin
  intros, 
  sorry
end

end problem1_problem2_l109_109607


namespace geometric_sequence_fourth_term_l109_109829

theorem geometric_sequence_fourth_term (a r T4 : ℝ)
  (h1 : a = 1024)
  (h2 : a * r^5 = 32)
  (h3 : T4 = a * r^3) :
  T4 = 128 :=
by {
  sorry
}

end geometric_sequence_fourth_term_l109_109829


namespace calc_expression_l109_109427

theorem calc_expression :
  (2014 * 2014 + 2012) - 2013 * 2013 = 6039 :=
by
  -- Let 2014 = 2013 + 1 and 2012 = 2013 - 1
  have h2014 : 2014 = 2013 + 1 := by sorry
  have h2012 : 2012 = 2013 - 1 := by sorry
  -- Start the main proof
  sorry

end calc_expression_l109_109427


namespace games_bought_from_friend_is_21_l109_109506

-- Definitions from the conditions
def games_bought_at_garage_sale : ℕ := 8
def non_working_games : ℕ := 23
def good_games : ℕ := 6

-- The total number of games John has is the sum of good and non-working games
def total_games : ℕ := good_games + non_working_games

-- The number of games John bought from his friend
def games_from_friend : ℕ := total_games - games_bought_at_garage_sale

-- Statement to prove
theorem games_bought_from_friend_is_21 : games_from_friend = 21 := by
  sorry

end games_bought_from_friend_is_21_l109_109506


namespace wall_height_l109_109739

noncomputable def brickVolume (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def wallVolume (L W H : ℝ) : ℝ :=
  L * W * H

theorem wall_height (bricks_needed : ℝ) (brick_length_cm brick_width_cm brick_height_cm wall_length wall_width wall_height : ℝ)
  (H1 : bricks_needed = 4094.3396226415093)
  (H2 : brick_length_cm = 20)
  (H3 : brick_width_cm = 13.25)
  (H4 : brick_height_cm = 8)
  (H5 : wall_length = 7)
  (H6 : wall_width = 8)
  (H7 : brickVolume (brick_length_cm / 100) (brick_width_cm / 100) (brick_height_cm / 100) * bricks_needed = wallVolume wall_length wall_width wall_height) :
  wall_height = 0.155 :=
by
  sorry

end wall_height_l109_109739


namespace find_f1_l109_109643

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def condition_on_function (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, x ≤ 0 → f x = 2^x - 3 * x + 2 * m

theorem find_f1 (f : ℝ → ℝ) (m : ℝ)
  (h_odd : is_odd_function f)
  (h_condition : condition_on_function f m) :
  f 1 = -(5 / 2) :=
by
  sorry

end find_f1_l109_109643


namespace meena_sold_to_stone_l109_109018

def total_cookies_baked : ℕ := 5 * 12
def cookies_bought_brock : ℕ := 7
def cookies_bought_katy : ℕ := 2 * cookies_bought_brock
def cookies_left : ℕ := 15
def cookies_sold_total : ℕ := total_cookies_baked - cookies_left
def cookies_bought_friends : ℕ := cookies_bought_brock + cookies_bought_katy
def cookies_sold_stone : ℕ := cookies_sold_total - cookies_bought_friends
def dozens_sold_stone : ℕ := cookies_sold_stone / 12

theorem meena_sold_to_stone : dozens_sold_stone = 2 := by
  sorry

end meena_sold_to_stone_l109_109018


namespace part1_part2_l109_109142

noncomputable def f (x : ℝ) := x * Real.log x
noncomputable def g (x : ℝ) := x^2 - 1

theorem part1 {x : ℝ} (h : 1 ≤ x) : f x ≤ (1 / 2) * g x := by
  sorry

theorem part2 {m : ℝ} : (∀ x, 1 ≤ x → f x - m * g x ≤ 0) → m ≥ (1 / 2) := by
  sorry

end part1_part2_l109_109142


namespace gratuity_is_four_l109_109512

-- Define the prices and tip percentage (conditions)
def a : ℕ := 10
def b : ℕ := 13
def c : ℕ := 17
def p : ℚ := 0.1

-- Define the total bill and gratuity based on the given definitions
def total_bill : ℕ := a + b + c
def gratuity : ℚ := total_bill * p

-- Theorem (proof problem): Prove that the gratuity is $4
theorem gratuity_is_four : gratuity = 4 := by
  sorry

end gratuity_is_four_l109_109512


namespace primes_digit_3_count_l109_109933

open Nat 

def primes_with_digit_3_under_50 : List ℕ := [3, 13, 23, 33, 43]

/-- The number of primes less than 50 with 3 as the ones digit is 4. -/
theorem primes_digit_3_count :
  (primes_with_digit_3_under_50.filter (λ n, Prime n)).length = 4 :=
by 
  sorry

end primes_digit_3_count_l109_109933


namespace pen_cost_price_l109_109750

-- Define the variables and assumptions
variable (x : ℝ)

-- Given conditions
def profit_one_pen (x : ℝ) := 10 - x
def profit_three_pens (x : ℝ) := 20 - 3 * x

-- Statement to prove
theorem pen_cost_price : profit_one_pen x = profit_three_pens x → x = 5 :=
by
  sorry

end pen_cost_price_l109_109750


namespace y_value_solution_l109_109436

theorem y_value_solution (y : ℝ) (h : (3 / y) - ((4 / y) * (2 / y)) = 1.5) : 
  y = 1 + Real.sqrt (19 / 3) := 
sorry

end y_value_solution_l109_109436


namespace probability_x_gt_3y_l109_109657

/--
Prove that if a point (x, y) is randomly picked from the rectangular region with vertices
at (0,0), (2010,0), (2010,2011), and (0,2011), the probability that x > 3y is 335/2011.
-/
theorem probability_x_gt_3y (x y : ℝ) :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 2010) ∧ (0 ≤ y ∧ y ≤ 2011) ∧ True -> 
  sorry :=
begin
  sorry
end

end probability_x_gt_3y_l109_109657


namespace solve_inequality_l109_109673

theorem solve_inequality (x : ℝ) (hx : x ≥ 0) :
  2021 * (x ^ (2020/202)) - 1 ≥ 2020 * x ↔ x = 1 :=
by sorry

end solve_inequality_l109_109673


namespace find_a_l109_109277
open Real

theorem find_a (a : ℝ) (k : ℤ) :
  (∃ x1 y1 x2 y2 : ℝ,
    (x1^2 + y1^2 = 10 * (x1 * cos a + y1 * sin a) ∧
     x2^2 + y2^2 = 10 * (x2 * sin (3 * a) + y2 * cos (3 * a)) ∧
     (x2 - x1)^2 + (y2 - y1)^2 = 64)) ↔
  (∃ k : ℤ, a = π / 8 + k * π / 2) :=
sorry

end find_a_l109_109277


namespace min_value_2a_plus_b_l109_109452

theorem min_value_2a_plus_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : 3 * a + b = a^2 + a * b) :
  2 * a + b ≥ 2 * Real.sqrt 2 + 3 :=
sorry

end min_value_2a_plus_b_l109_109452


namespace no_real_solution_of_fraction_eq_l109_109324

theorem no_real_solution_of_fraction_eq (m : ℝ) :
  (∀ x : ℝ, (x - 1) / (x + 4) ≠ m / (x + 4)) → m = -5 :=
sorry

end no_real_solution_of_fraction_eq_l109_109324


namespace paper_cut_square_l109_109955

noncomputable def proof_paper_cut_square : Prop :=
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ ((2 * x - 2 = 2 - x) ∨ (2 * (2 * x - 2) = 2 - x)) ∧ (x = 1.2 ∨ x = 1.5)

theorem paper_cut_square : proof_paper_cut_square :=
sorry

end paper_cut_square_l109_109955


namespace four_times_angle_triangle_l109_109001

theorem four_times_angle_triangle (A B C : ℕ) 
  (h1 : A + B + C = 180) 
  (h2 : A = 40)
  (h3 : (A = 4 * C) ∨ (B = 4 * C) ∨ (C = 4 * A)) : 
  (B = 130 ∧ C = 10) ∨ (B = 112 ∧ C = 28) :=
by
  sorry

end four_times_angle_triangle_l109_109001


namespace probability_x_greater_3y_in_rectangle_l109_109668

theorem probability_x_greater_3y_in_rectangle :
  let rect := set.Icc (0 : ℝ) 2010 ×ˢ set.Icc (0 : ℝ) 2011 in
  let event := {p : ℝ × ℝ | p.1 > 3 * p.2} in
  let area_rect := (2010 : ℝ) * 2011 in
  let area_event := (2010 : ℝ) * (2010/3) / 2 in
  (event ∩ rect).measure / rect.measure = 335 / 2011 :=
by
  -- The proof goes here
  sorry

end probability_x_greater_3y_in_rectangle_l109_109668


namespace find_number_l109_109837

theorem find_number (x : ℝ) : 0.75 * x + 2 = 8 → x = 8 :=
by
  sorry

end find_number_l109_109837


namespace pages_remaining_l109_109168

def total_pages : ℕ := 120
def science_project_pages : ℕ := (25 * total_pages) / 100
def math_homework_pages : ℕ := 10
def total_used_pages : ℕ := science_project_pages + math_homework_pages
def remaining_pages : ℕ := total_pages - total_used_pages

theorem pages_remaining : remaining_pages = 80 := by
  sorry

end pages_remaining_l109_109168


namespace polynomial_p_l109_109969

variable {a b c : ℝ}

theorem polynomial_p (a b c : ℝ) : 
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = (a - b) * (b - c) * (c - a) * 2 :=
by
  sorry

end polynomial_p_l109_109969


namespace second_polygon_num_sides_l109_109049

theorem second_polygon_num_sides 
  (s : ℝ) 
  (h1: s ≠ 0) 
  (perimeter_equal: ∀ (side1_len side2_len : ℝ),
    50 * (3 * side1_len) = side2_len * (50 * 3 * side1_len / side1_len)) : 
  ∃ n : ℕ, n = 150 :=
by {
  use 150,
  sorry
}

end second_polygon_num_sides_l109_109049


namespace meeting_point_2015th_l109_109911

-- Define the parameters of the problem
variables (A B C D : Type)
variables (x y t : ℝ) -- Speeds and the initial time delay

-- State the problem as a theorem
theorem meeting_point_2015th (start_times_differ : t > 0)
                            (speeds_pos : x > 0 ∧ y > 0)
                            (pattern : ∀ n : ℕ, (odd n → (meeting_point n = C)) ∧ (even n → (meeting_point n = D)))
                            (n = 2015) :
  meeting_point n = C :=
  sorry

end meeting_point_2015th_l109_109911


namespace problem_l109_109714

theorem problem (a : ℕ) (h1 : a = 444) : (444 ^ 444) % 13 = 1 :=
by
  have h444 : 444 % 13 = 3 := by sorry
  have h3_pow3 : 3 ^ 3 % 13 = 1 := by sorry
  sorry

end problem_l109_109714


namespace billiard_ball_radius_unique_l109_109560

noncomputable def radius_of_billiard_balls (r : ℝ) : Prop :=
  let side_length := 292
  let lhs := (8 + 2 * Real.sqrt 3) * r
  lhs = side_length

theorem billiard_ball_radius_unique (r : ℝ) : radius_of_billiard_balls r → r = (146 / 13) * (4 - Real.sqrt 3 / 3) :=
by
  intro h1
  sorry

end billiard_ball_radius_unique_l109_109560


namespace cost_per_ice_cream_l109_109418

theorem cost_per_ice_cream (chapati_count : ℕ)
                           (rice_plate_count : ℕ)
                           (mixed_vegetable_plate_count : ℕ)
                           (ice_cream_cup_count : ℕ)
                           (cost_per_chapati : ℕ)
                           (cost_per_rice_plate : ℕ)
                           (cost_per_mixed_vegetable : ℕ)
                           (amount_paid : ℕ)
                           (total_cost_chapatis : ℕ)
                           (total_cost_rice : ℕ)
                           (total_cost_mixed_vegetable : ℕ)
                           (total_non_ice_cream_cost : ℕ)
                           (total_ice_cream_cost : ℕ)
                           (cost_per_ice_cream_cup : ℕ) :
    chapati_count = 16 →
    rice_plate_count = 5 →
    mixed_vegetable_plate_count = 7 →
    ice_cream_cup_count = 6 →
    cost_per_chapati = 6 →
    cost_per_rice_plate = 45 →
    cost_per_mixed_vegetable = 70 →
    amount_paid = 961 →
    total_cost_chapatis = chapati_count * cost_per_chapati →
    total_cost_rice = rice_plate_count * cost_per_rice_plate →
    total_cost_mixed_vegetable = mixed_vegetable_plate_count * cost_per_mixed_vegetable →
    total_non_ice_cream_cost = total_cost_chapatis + total_cost_rice + total_cost_mixed_vegetable →
    total_ice_cream_cost = amount_paid - total_non_ice_cream_cost →
    cost_per_ice_cream_cup = total_ice_cream_cost / ice_cream_cup_count →
    cost_per_ice_cream_cup = 25 :=
by
    intros; sorry

end cost_per_ice_cream_l109_109418


namespace rancher_no_cows_l109_109999

theorem rancher_no_cows (s c : ℕ) (h1 : 30 * s + 31 * c = 1200) 
  (h2 : 15 ≤ s) (h3 : s ≤ 35) : c = 0 :=
by
  sorry

end rancher_no_cows_l109_109999


namespace markup_is_correct_l109_109835

def purchase_price : ℝ := 48
def overhead_percent : ℝ := 0.25
def net_profit : ℝ := 12

def overhead_cost := overhead_percent * purchase_price
def total_cost := purchase_price + overhead_cost
def selling_price := total_cost + net_profit
def markup := selling_price - purchase_price

theorem markup_is_correct : markup = 24 := by sorry

end markup_is_correct_l109_109835


namespace compare_constants_l109_109141

noncomputable def a := 1 / Real.exp 1
noncomputable def b := Real.log 2 / 2
noncomputable def c := Real.log 3 / 3

theorem compare_constants : b < c ∧ c < a := by
  sorry

end compare_constants_l109_109141


namespace count_squares_and_cubes_less_than_1000_l109_109303

theorem count_squares_and_cubes_less_than_1000 : 
  {n : ℕ // n^6 < 1000}.card = 3 :=
by
  sorry

end count_squares_and_cubes_less_than_1000_l109_109303


namespace amount_made_per_jersey_l109_109676

-- Definitions based on conditions
def total_revenue_from_jerseys : ℕ := 25740
def number_of_jerseys_sold : ℕ := 156

-- Theorem statement
theorem amount_made_per_jersey : 
  total_revenue_from_jerseys / number_of_jerseys_sold = 165 := 
by
  sorry

end amount_made_per_jersey_l109_109676


namespace temperature_conversion_correct_l109_109390

noncomputable def f_to_c (T : ℝ) : ℝ := (T - 32) * (5 / 9)

theorem temperature_conversion_correct :
  f_to_c 104 = 40 :=
by
  sorry

end temperature_conversion_correct_l109_109390


namespace factor_w4_minus_16_l109_109768

theorem factor_w4_minus_16 (w : ℝ) : (w^4 - 16) = (w - 2) * (w + 2) * (w^2 + 4) :=
by
    sorry

end factor_w4_minus_16_l109_109768


namespace digit_150th_in_decimal_of_fraction_l109_109217

theorem digit_150th_in_decimal_of_fraction : 
  (∀ n : ℕ, n > 0 → (let seq := [2, 4, 2, 8, 5, 7] in seq[((n - 1) % 6)] = 
  if n == 150 then 7 else seq[((n - 1) % 6)])) :=
by
  sorry

end digit_150th_in_decimal_of_fraction_l109_109217


namespace find_B_l109_109396

variable (A B : ℝ)

def condition1 : Prop := A + B = 1210
def condition2 : Prop := (4 / 15) * A = (2 / 5) * B

theorem find_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 484 :=
sorry

end find_B_l109_109396


namespace find_roots_l109_109861

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_roots 
  (h_symm : ∀ x : ℝ, f (2 + x) = f (2 - x))
  (h_three_roots : ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ f a = 0 ∧ f b = 0 ∧ f c = 0)
  (h_zero_root : f 0 = 0) :
  ∃ a b : ℝ, a = 2 ∧ b = 4 ∧ f a = 0 ∧ f b = 0 :=
sorry

end find_roots_l109_109861


namespace y_affected_by_other_factors_l109_109501

-- Given the linear regression model
def linear_regression_model (b a e x : ℝ) : ℝ := b * x + a + e

-- Theorem: Prove that the dependent variable \( y \) may be affected by factors other than the independent variable \( x \)
theorem y_affected_by_other_factors (b a e x : ℝ) :
  ∃ y, (y = linear_regression_model b a e x ∧ e ≠ 0) :=
sorry

end y_affected_by_other_factors_l109_109501


namespace exponents_divisible_by_8_l109_109545

theorem exponents_divisible_by_8 (n : ℕ) : 8 ∣ (3^(4 * n + 1) + 5^(2 * n + 1)) :=
by
-- Base case and inductive step will be defined here.
sorry

end exponents_divisible_by_8_l109_109545


namespace count_positive_integers_square_and_cube_lt_1000_l109_109308

theorem count_positive_integers_square_and_cube_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ k : ℕ, n = k^6}.to_finset.card = 3 :=
by
  sorry

end count_positive_integers_square_and_cube_lt_1000_l109_109308


namespace find_x_satisfying_sinx_plus_cosx_eq_one_l109_109443

theorem find_x_satisfying_sinx_plus_cosx_eq_one :
  ∀ x, 0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x + Real.cos x = 1 ↔ x = 0) := by
  sorry

end find_x_satisfying_sinx_plus_cosx_eq_one_l109_109443


namespace number_of_students_l109_109733

noncomputable def is_handshakes_correct (m n : ℕ) : Prop :=
  m ≥ 3 ∧ n ≥ 3 ∧ 
  (1 / 2 : ℚ) * (12 + 10 * (m + n - 4) + 8 * (m - 2) * (n - 2)) = 1020

theorem number_of_students (m n : ℕ) (h : is_handshakes_correct m n) : m * n = 280 := sorry

end number_of_students_l109_109733


namespace total_cost_for_round_trip_l109_109386

def time_to_cross_one_way : ℕ := 4 -- time in hours to cross the lake one way
def cost_per_hour : ℕ := 10 -- cost in dollars per hour

def total_time := time_to_cross_one_way * 2 -- total time in hours for a round trip
def total_cost := total_time * cost_per_hour -- total cost in dollars for the assistant

theorem total_cost_for_round_trip : total_cost = 80 := by
  repeat {sorry} -- Leaving the proof for now

end total_cost_for_round_trip_l109_109386


namespace arithmetic_sequence_a12_l109_109779

def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
∀ n : ℕ, a n = a1 + n * d

theorem arithmetic_sequence_a12 (a : ℕ → ℤ) (a1 d : ℤ) (h : arithmetic_sequence a a1 d) :
  a 11 = 23 :=
by
  -- condtions
  let a1_val := 1
  let d_val := 2
  have ha1 : a1 = a1_val := sorry
  have hd : d = d_val := sorry
  
  -- proof
  rw [ha1, hd] at h
  
  sorry

end arithmetic_sequence_a12_l109_109779


namespace contrapositive_equivalence_l109_109828

-- Define the original proposition and its contrapositive
def original_proposition (q p : Prop) := q → p
def contrapositive (q p : Prop) := ¬q → ¬p

-- The theorem to prove
theorem contrapositive_equivalence (q p : Prop) :
  (original_proposition q p) ↔ (contrapositive q p) :=
by
  sorry

end contrapositive_equivalence_l109_109828


namespace area_of_inscribed_square_l109_109068

-- Define the right triangle with segments m and n on the hypotenuse
variables {m n : ℝ}

-- Noncomputable setting for non-constructive aspects
noncomputable def inscribed_square_area (m n : ℝ) : ℝ :=
  (m * n)

-- Theorem stating that the area of the inscribed square is m * n
theorem area_of_inscribed_square (m n : ℝ) : inscribed_square_area m n = m * n :=
by sorry

end area_of_inscribed_square_l109_109068


namespace net_change_is_12_l109_109854

-- Definitions based on the conditions of the problem

def initial_investment : ℝ := 100
def first_year_increase_percentage : ℝ := 0.60
def second_year_decrease_percentage : ℝ := 0.30

-- Calculate the wealth at the end of the first year
def end_of_first_year_wealth : ℝ := initial_investment * (1 + first_year_increase_percentage)

-- Calculate the wealth at the end of the second year
def end_of_second_year_wealth : ℝ := end_of_first_year_wealth * (1 - second_year_decrease_percentage)

-- Calculate the net change
def net_change : ℝ := end_of_second_year_wealth - initial_investment

-- The target theorem to prove
theorem net_change_is_12 : net_change = 12 := by
  sorry

end net_change_is_12_l109_109854


namespace intersection_A_B_l109_109290

-- Define the sets A and B based on given conditions
def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {x | -1 < x}

-- The statement to prove
theorem intersection_A_B : (A ∩ B) = {x | -1 < x ∧ x < 4} :=
  sorry

end intersection_A_B_l109_109290


namespace terminating_decimal_count_l109_109900

theorem terminating_decimal_count : 
  {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ (∃ k : ℕ, n = 49 * k)}.card = 10 :=
by
  sorry

end terminating_decimal_count_l109_109900


namespace value_of_3a_minus_b_l109_109153
noncomputable def solveEquation : Type := sorry

theorem value_of_3a_minus_b (a b : ℝ) (h1 : a = 3 + Real.sqrt 15) (h2 : b = 3 - Real.sqrt 15) (h3 : a ≥ b) :
  3 * a - b = 6 + 4 * Real.sqrt 15 :=
sorry

end value_of_3a_minus_b_l109_109153


namespace triangle_problem_l109_109752

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

def has_same_area (a b : ℕ) (area : ℝ) : Prop :=
  let s := (2 * a + b) / 2
  let areaT := Real.sqrt (s * (s - a) * (s - a) * (s - b))
  areaT = area

def has_same_perimeter (a b : ℕ) (perimeter : ℕ) : Prop :=
  2 * a + b = perimeter

def correct_b (b : ℕ) : Prop :=
  b = 5

theorem triangle_problem
  (a1 a2 b1 b2 : ℕ)
  (h1 : is_isosceles_triangle a1 a1 b1)
  (h2 : is_isosceles_triangle a2 a2 b2)
  (h3 : has_same_area a1 b1 (Real.sqrt 275))
  (h4 : has_same_perimeter a1 b1 22)
  (h5 : has_same_area a2 b2 (Real.sqrt 275))
  (h6 : has_same_perimeter a2 b2 22)
  (h7 : ¬(a1 = a2 ∧ b1 = b2)) : correct_b b2 :=
by
  sorry

end triangle_problem_l109_109752


namespace find_f8_l109_109531

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f (x + y) = f x * f y
axiom initial_condition : f 2 = 4

theorem find_f8 : f 8 = 256 := by
  sorry

end find_f8_l109_109531


namespace digit_150_in_17_div_70_l109_109197

noncomputable def repeating_sequence_170 : List Nat := [2, 4, 2, 8, 5, 7]

theorem digit_150_in_17_div_70 : (repeating_sequence_170.get? (150 % 6 - 1) = some 7) := by
  sorry

end digit_150_in_17_div_70_l109_109197
