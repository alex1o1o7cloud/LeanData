import Mathlib

namespace josh_money_left_l1001_100109

theorem josh_money_left (initial_amount : ℝ) (first_spend : ℝ) (second_spend : ℝ) 
  (h1 : initial_amount = 9) 
  (h2 : first_spend = 1.75) 
  (h3 : second_spend = 1.25) : 
  initial_amount - first_spend - second_spend = 6 := 
by 
  sorry

end josh_money_left_l1001_100109


namespace negation_of_prop_l1001_100135

def prop (x : ℝ) := x^2 ≥ 0

theorem negation_of_prop:
  ¬ ∀ x : ℝ, prop x ↔ ∃ x : ℝ, x^2 < 0 := by
    sorry

end negation_of_prop_l1001_100135


namespace total_bills_is_126_l1001_100121

noncomputable def F : ℕ := 84  -- number of 5-dollar bills
noncomputable def T : ℕ := (840 - 5 * F) / 10  -- derive T based on the total value and F
noncomputable def total_bills : ℕ := F + T

theorem total_bills_is_126 : total_bills = 126 :=
by
  -- Placeholder for the proof
  sorry

end total_bills_is_126_l1001_100121


namespace markup_percentage_l1001_100185

-- Define the wholesale cost
def wholesale_cost : ℝ := sorry

-- Define the retail cost
def retail_cost : ℝ := sorry

-- Condition given in the problem: selling at 60% discount nets a 20% profit
def discount_condition (W R : ℝ) : Prop :=
  0.40 * R = 1.20 * W

-- We need to prove the markup percentage is 200%
theorem markup_percentage (W R : ℝ) (h : discount_condition W R) : 
  ((R - W) / W) * 100 = 200 :=
by sorry

end markup_percentage_l1001_100185


namespace point_d_lies_on_graph_l1001_100160

theorem point_d_lies_on_graph : (-1 : ℝ) = -2 * (1 : ℝ) + 1 :=
by {
  sorry
}

end point_d_lies_on_graph_l1001_100160


namespace sum_coefficients_l1001_100195

theorem sum_coefficients (a1 a2 a3 a4 a5 : ℤ) (h : ∀ x : ℕ, a1 * (x - 1) ^ 4 + a2 * (x - 1) ^ 3 + a3 * (x - 1) ^ 2 + a4 * (x - 1) + a5 = x ^ 4) :
  a2 + a3 + a4 = 14 :=
  sorry

end sum_coefficients_l1001_100195


namespace combined_width_approximately_8_l1001_100139

noncomputable def C1 := 352 / 7
noncomputable def C2 := 528 / 7
noncomputable def C3 := 704 / 7

noncomputable def r1 := C1 / (2 * Real.pi)
noncomputable def r2 := C2 / (2 * Real.pi)
noncomputable def r3 := C3 / (2 * Real.pi)

noncomputable def W1 := r2 - r1
noncomputable def W2 := r3 - r2

noncomputable def combined_width := W1 + W2

theorem combined_width_approximately_8 :
  |combined_width - 8| < 1 :=
by
  sorry

end combined_width_approximately_8_l1001_100139


namespace sequence_increasing_l1001_100174

theorem sequence_increasing (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ∀ n : ℕ, a^n / n^b < a^(n+1) / (n+1)^b :=
by sorry

end sequence_increasing_l1001_100174


namespace time_after_9999_seconds_l1001_100192

theorem time_after_9999_seconds:
  let initial_hours := 5
  let initial_minutes := 45
  let initial_seconds := 0
  let added_seconds := 9999
  let total_seconds := initial_seconds + added_seconds
  let total_minutes := total_seconds / 60
  let remaining_seconds := total_seconds % 60
  let total_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let final_hours := (initial_hours + total_hours + (initial_minutes + remaining_minutes) / 60) % 24
  let final_minutes := (initial_minutes + remaining_minutes) % 60
  initial_hours = 5 →
  initial_minutes = 45 →
  initial_seconds = 0 →
  added_seconds = 9999 →
  final_hours = 8 ∧ final_minutes = 31 ∧ remaining_seconds = 39 :=
by
  intros
  sorry

end time_after_9999_seconds_l1001_100192


namespace f_zero_f_increasing_on_negative_l1001_100111

noncomputable def f : ℝ → ℝ := sorry
variable {x : ℝ}

-- Assume f is an odd function
axiom odd_f : ∀ x, f (-x) = -f x

-- Assume f is increasing on (0, +∞)
axiom increasing_f_on_positive :
  ∀ ⦃x₁ x₂⦄, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂

-- Prove that f(0) = 0
theorem f_zero : f 0 = 0 := sorry

-- Prove that f is increasing on (-∞, 0)
theorem f_increasing_on_negative :
  ∀ ⦃x₁ x₂⦄, x₁ < x₂ → x₂ < 0 → f x₁ < f x₂ := sorry

end f_zero_f_increasing_on_negative_l1001_100111


namespace total_rent_of_field_is_correct_l1001_100197

namespace PastureRental

def cowMonths (cows : ℕ) (months : ℕ) : ℕ := cows * months

def aCowMonths : ℕ := cowMonths 24 3
def bCowMonths : ℕ := cowMonths 10 5
def cCowMonths : ℕ := cowMonths 35 4
def dCowMonths : ℕ := cowMonths 21 3

def totalCowMonths : ℕ := aCowMonths + bCowMonths + cCowMonths + dCowMonths

def rentPerCowMonth : ℕ := 1440 / aCowMonths

def totalRent : ℕ := rentPerCowMonth * totalCowMonths

theorem total_rent_of_field_is_correct :
  totalRent = 6500 :=
by
  sorry

end PastureRental

end total_rent_of_field_is_correct_l1001_100197


namespace mean_equal_implication_l1001_100112

theorem mean_equal_implication (y : ℝ) :
  (7 + 10 + 15 + 23 = 55) →
  (55 / 4 = 13.75) →
  (18 + y + 30 = 48 + y) →
  (48 + y) / 3 = 13.75 →
  y = -6.75 :=
by 
  intros h1 h2 h3 h4
  -- The steps would be applied here to prove y = -6.75
  sorry

end mean_equal_implication_l1001_100112


namespace square_perimeter_ratio_l1001_100138

theorem square_perimeter_ratio (a₁ a₂ s₁ s₂ : ℝ) 
  (h₁ : a₁ / a₂ = 16 / 25)
  (h₂ : a₁ = s₁^2)
  (h₃ : a₂ = s₂^2) :
  (4 : ℝ) / 5 = s₁ / s₂ :=
by sorry

end square_perimeter_ratio_l1001_100138


namespace six_n_digit_remains_divisible_by_7_l1001_100164

-- Given the conditions
def is_6n_digit_number (N : ℕ) (n : ℕ) : Prop :=
  N < 10^(6*n) ∧ N ≥ 10^(6*(n-1))

def is_divisible_by_7 (N : ℕ) : Prop :=
  N % 7 = 0

-- Define new number M formed by moving the unit digit to the beginning
def new_number (N : ℕ) (n : ℕ) : ℕ :=
  let a_0 := N % 10
  let rest := N / 10
  a_0 * 10^(6*n - 1) + rest

-- The theorem statement
theorem six_n_digit_remains_divisible_by_7 (N : ℕ) (n : ℕ)
  (hN : is_6n_digit_number N n)
  (hDiv7 : is_divisible_by_7 N) : is_divisible_by_7 (new_number N n) :=
sorry

end six_n_digit_remains_divisible_by_7_l1001_100164


namespace yellow_marbles_problem_l1001_100103

variable (Y B R : ℕ)

theorem yellow_marbles_problem
  (h1 : Y + B + R = 19)
  (h2 : B = (3 * R) / 4)
  (h3 : R = Y + 3) :
  Y = 5 :=
by
  sorry

end yellow_marbles_problem_l1001_100103


namespace part_I_part_II_l1001_100159

def f (x a : ℝ) : ℝ := abs (x - a) + abs (2 * x + 1)

-- Part (I)
theorem part_I (x : ℝ) : f x 1 ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1 :=
by sorry

-- Part (II)
theorem part_II (a : ℝ) : (∃ x ∈ Set.Ici a, f x a ≤ 2 * a + x) ↔ a ≥ 1 :=
by sorry

end part_I_part_II_l1001_100159


namespace sin_alpha_cos_2beta_l1001_100141

theorem sin_alpha_cos_2beta :
  ∀ α β : ℝ, 3 * Real.sin α - Real.sin β = Real.sqrt 10 ∧ α + β = Real.pi / 2 →
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 :=
by
  intros α β h
  sorry

end sin_alpha_cos_2beta_l1001_100141


namespace uncovered_side_length_l1001_100156

theorem uncovered_side_length {L W : ℕ} (h1 : L * W = 680) (h2 : 2 * W + L = 74) : L = 40 :=
sorry

end uncovered_side_length_l1001_100156


namespace daily_production_l1001_100125

-- Define the conditions
def bottles_per_case : ℕ := 9
def num_cases : ℕ := 8000

-- State the theorem with the question and the calculated answer
theorem daily_production : bottles_per_case * num_cases = 72000 :=
by
  sorry

end daily_production_l1001_100125


namespace train_passes_man_in_approximately_18_seconds_l1001_100142

noncomputable def length_of_train : ℝ := 330 -- meters
noncomputable def speed_of_train : ℝ := 60 -- kmph
noncomputable def speed_of_man : ℝ := 6 -- kmph

noncomputable def kmph_to_mps (v : ℝ) : ℝ := v * (5/18)

noncomputable def relative_speed_mps : ℝ :=
  kmph_to_mps (speed_of_train + speed_of_man)

noncomputable def time_to_pass (length : ℝ) (speed : ℝ) : ℝ := length / speed

theorem train_passes_man_in_approximately_18_seconds :
  abs (time_to_pass length_of_train relative_speed_mps - 18) < 1 :=
by
  sorry

end train_passes_man_in_approximately_18_seconds_l1001_100142


namespace average_test_score_fifty_percent_l1001_100183

-- Given conditions
def percent1 : ℝ := 15
def avg1 : ℝ := 100
def percent2 : ℝ := 50
def avg3 : ℝ := 63
def overall_average : ℝ := 76.05

-- Intermediate calculations based on given conditions
def total_percent : ℝ := 100
def percent3: ℝ := total_percent - percent1 - percent2
def sum_of_weights: ℝ := overall_average * total_percent

-- Expected average of the group that is 50% of the class
theorem average_test_score_fifty_percent (X: ℝ) :
  sum_of_weights = percent1 * avg1 + percent2 * X + percent3 * avg3 → X = 78 := by
  sorry

end average_test_score_fifty_percent_l1001_100183


namespace max_unbounded_xy_sum_l1001_100110

theorem max_unbounded_xy_sum (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  ∃ M : ℝ, ∀ z : ℝ, z > 0 → ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ (xy + 1)^2 + (x - y)^2 > z := 
  sorry

end max_unbounded_xy_sum_l1001_100110


namespace florist_sold_16_roses_l1001_100102

-- Definitions for initial and final states
def initial_roses : ℕ := 37
def picked_roses : ℕ := 19
def final_roses : ℕ := 40

-- Defining the variable for number of roses sold
variable (x : ℕ)

-- The statement to prove
theorem florist_sold_16_roses
  (h : initial_roses - x + picked_roses = final_roses) : x = 16 := 
by
  -- Placeholder for proof
  sorry

end florist_sold_16_roses_l1001_100102


namespace range_of_sqrt_meaningful_real_l1001_100163

theorem range_of_sqrt_meaningful_real (x : ℝ) : (x - 1 ≥ 0) ↔ (x ≥ 1) :=
by
  sorry

end range_of_sqrt_meaningful_real_l1001_100163


namespace proof_total_distance_l1001_100173

-- Define the total distance
def total_distance (D : ℕ) :=
  let by_foot := (1 : ℚ) / 6
  let by_bicycle := (1 : ℚ) / 4
  let by_bus := (1 : ℚ) / 3
  let by_car := 10
  let by_train := (1 : ℚ) / 12
  D - (by_foot + by_bicycle + by_bus + by_train) * D = by_car

-- Given proof problem
theorem proof_total_distance : ∃ D : ℕ, total_distance D ∧ D = 60 :=
sorry

end proof_total_distance_l1001_100173


namespace problem_solution_l1001_100172

def seq (a : ℕ → ℝ) (a1 : a 1 = 0) (rec : ∀ n, a (n + 1) = (a n - Real.sqrt 3) / (1 + Real.sqrt 3 * a n)) : Prop :=
  a 6 = Real.sqrt 3

theorem problem_solution (a : ℕ → ℝ) (h1 : a 1 = 0) (hrec : ∀ n, a (n + 1) = (a n - Real.sqrt 3) / (1 + Real.sqrt 3 * a n)) : 
  seq a h1 hrec :=
by
  sorry

end problem_solution_l1001_100172


namespace al_sandwiches_correct_l1001_100155

-- Definitions based on the given conditions
def num_breads := 5
def num_meats := 7
def num_cheeses := 6
def total_combinations := num_breads * num_meats * num_cheeses

def turkey_swiss := num_breads -- disallowed turkey/Swiss cheese combinations
def multigrain_turkey := num_cheeses -- disallowed multi-grain bread/turkey combinations

def al_sandwiches := total_combinations - turkey_swiss - multigrain_turkey

-- The theorem to prove
theorem al_sandwiches_correct : al_sandwiches = 199 := 
by sorry

end al_sandwiches_correct_l1001_100155


namespace sum_of_first_9_terms_l1001_100166

variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)
variable (a1 : ℤ)
variable (d : ℤ)

-- Given is that the sequence is arithmetic.
-- Given a1 is the first term, and d is the common difference, we can define properties based on the conditions.
def is_arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = a1 + (n - 1) * d

def sum_first_n_terms (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Given condition: 2a_1 + a_13 = -9.
def given_condition (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ) : Prop :=
  2 * a1 + (a1 + 12 * d) = -9

theorem sum_of_first_9_terms (a : ℕ → ℤ) (S : ℕ → ℤ) (a1 d : ℤ)
  (h_arith : is_arithmetic_sequence a a1 d)
  (h_sum : sum_first_n_terms S a)
  (h_cond : given_condition a a1 d) :
  S 9 = -27 :=
sorry

end sum_of_first_9_terms_l1001_100166


namespace usual_time_to_catch_bus_l1001_100146

theorem usual_time_to_catch_bus (S T : ℝ) (h : S / (4 / 5 * S) = (T + 3) / T) : T = 12 :=
by 
  sorry

end usual_time_to_catch_bus_l1001_100146


namespace bob_more_than_ken_l1001_100140

def ken_situps : ℕ := 20

def nathan_situps : ℕ := 2 * ken_situps

def bob_situps : ℕ := (ken_situps + nathan_situps) / 2

theorem bob_more_than_ken : bob_situps - ken_situps = 10 := 
sorry

end bob_more_than_ken_l1001_100140


namespace general_term_formula_minimum_sum_value_l1001_100101

variable {a : ℕ → ℚ} -- The arithmetic sequence
variable {S : ℕ → ℚ} -- Sum of the first n terms of the sequence

-- Conditions
axiom a_seq_cond1 : a 2 + a 6 = 6
axiom S_sum_cond5 : S 5 = 35 / 3

-- Definitions
def a_n (n : ℕ) : ℚ := (2 / 3) * n + 1 / 3
def S_n (n : ℕ) : ℚ := (1 / 3) * (n^2 + 2 * n)

-- Hypotheses
axiom seq_def : ∀ n, a n = a_n n
axiom sum_def : ∀ n, S n = S_n n

-- Theorems to be proved
theorem general_term_formula : ∀ n, a n = (2 / 3 * n) + 1 / 3 := by sorry
theorem minimum_sum_value : ∀ n, S 1 ≤ S n := by sorry

end general_term_formula_minimum_sum_value_l1001_100101


namespace upper_limit_of_x_l1001_100134

theorem upper_limit_of_x :
  ∀ x : ℤ, (0 < x ∧ x < 7) ∧ (0 < x ∧ x < some_upper_limit) ∧ (5 > x ∧ x > -1) ∧ (3 > x ∧ x > 0) ∧ (x + 2 < 4) →
  some_upper_limit = 2 :=
by
  intros x h
  sorry

end upper_limit_of_x_l1001_100134


namespace triangle_product_l1001_100136

theorem triangle_product (a b c: ℕ) (p: ℕ)
    (h1: ∃ k1 k2 k3: ℕ, a * k1 * k2 = p ∧ k2 * k3 * b = p ∧ k3 * c * a = p) 
    : (1 ≤ c ∧ c ≤ 336) :=
by
  sorry

end triangle_product_l1001_100136


namespace series_sum_eq_one_sixth_l1001_100171

noncomputable def series_sum := 
  ∑' n : ℕ, (3^n) / ((7^ (2^n)) + 1)

theorem series_sum_eq_one_sixth : series_sum = 1 / 6 := 
  sorry

end series_sum_eq_one_sixth_l1001_100171


namespace vegetation_coverage_relationship_l1001_100196

noncomputable def conditions :=
  let n := 20
  let sum_x := 60
  let sum_y := 1200
  let sum_xx := 80
  let sum_xy := 640
  (n, sum_x, sum_y, sum_xx, sum_xy)

theorem vegetation_coverage_relationship
  (n sum_x sum_y sum_xx sum_xy : ℕ)
  (h1 : n = 20)
  (h2 : sum_x = 60)
  (h3 : sum_y = 1200)
  (h4 : sum_xx = 80)
  (h5 : sum_xy = 640) :
  let b1 := sum_xy / sum_xx
  let mean_x := sum_x / n
  let mean_y := sum_y / n
  (b1 = 8) ∧ (b1 * (sum_xx / sum_xy) ≤ 1) ∧ ((3, 60) = (mean_x, mean_y)) :=
by
  sorry

end vegetation_coverage_relationship_l1001_100196


namespace incorrect_reciprocal_quotient_l1001_100124

-- Definitions based on problem conditions
def identity_property (x : ℚ) : x * 1 = x := by sorry
def division_property (a b : ℚ) (h : b ≠ 0) : a / b = 0 → a = 0 := by sorry
def additive_inverse_property (x : ℚ) : x * (-1) = -x := by sorry

-- Statement that needs to be proved
theorem incorrect_reciprocal_quotient (a b : ℚ) (h1 : a ≠ 0) (h2 : b = 1 / a) : a / b ≠ 1 :=
by sorry

end incorrect_reciprocal_quotient_l1001_100124


namespace problem_is_happy_number_512_l1001_100131

/-- A number is a "happy number" if it is the square difference of two consecutive odd numbers. -/
def is_happy_number (x : ℕ) : Prop :=
  ∃ n : ℤ, x = 8 * n

/-- The number 512 is a "happy number". -/
theorem problem_is_happy_number_512 : is_happy_number 512 :=
  sorry

end problem_is_happy_number_512_l1001_100131


namespace perfect_square_iff_l1001_100115

theorem perfect_square_iff (A : ℕ) : (∃ k : ℕ, A = k^2) ↔ (∀ n : ℕ, n > 0 → ∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ n ∣ ((A + k)^2 - A)) :=
by
  sorry

end perfect_square_iff_l1001_100115


namespace problem_part1_problem_part2_l1001_100152

theorem problem_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  a + b + c ≥ 1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c := 
sorry

theorem problem_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  a^2 + b^2 + c^2 ≥ Real.sqrt a + Real.sqrt b + Real.sqrt c :=
sorry

end problem_part1_problem_part2_l1001_100152


namespace cone_volume_in_liters_l1001_100126

theorem cone_volume_in_liters (d h : ℝ) (pi : ℝ) (liters_conversion : ℝ) :
  d = 12 → h = 10 → liters_conversion = 1000 → (1/3) * pi * (d/2)^2 * h * (1 / liters_conversion) = 0.12 * pi :=
by
  intros hd hh hc
  sorry

end cone_volume_in_liters_l1001_100126


namespace intersection_of_A_and_B_l1001_100114

def setA (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1
def setB (x : ℝ) : Prop := 0 < x ∧ x ≤ 2
def setIntersection (x : ℝ) : Prop := 0 < x ∧ x ≤ 1

theorem intersection_of_A_and_B :
  ∀ x, (setA x ∧ setB x) ↔ setIntersection x := 
by sorry

end intersection_of_A_and_B_l1001_100114


namespace polynomial_strictly_monotonic_l1001_100193

variable {P : ℝ → ℝ}

/-- The polynomial P(x) is such that the polynomials P(P(x)) and P(P(P(x))) are strictly monotonic 
on the entire real axis. Prove that P(x) is also strictly monotonic on the entire real axis. -/
theorem polynomial_strictly_monotonic
  (h1 : StrictMono (P ∘ P))
  (h2 : StrictMono (P ∘ P ∘ P)) :
  StrictMono P :=
sorry

end polynomial_strictly_monotonic_l1001_100193


namespace prob_triangle_includes_G_l1001_100117

-- Definitions based on conditions in the problem
def total_triangles : ℕ := 6
def triangles_including_G : ℕ := 4

-- The theorem statement proving the probability
theorem prob_triangle_includes_G : (triangles_including_G : ℚ) / total_triangles = 2 / 3 :=
by
  sorry

end prob_triangle_includes_G_l1001_100117


namespace part_one_costs_part_two_feasible_values_part_three_min_cost_l1001_100162

noncomputable def cost_of_stationery (a b : ℕ) (cost_A_and_B₁ : 2 * a + b = 35) (cost_A_and_B₂ : a + 3 * b = 30): ℕ × ℕ :=
(a, b)

theorem part_one_costs (a b : ℕ) (h₁ : 2 * a + b = 35) (h₂ : a + 3 * b = 30): cost_of_stationery a b h₁ h₂ = (15, 5) :=
sorry

theorem part_two_feasible_values (x : ℕ) (h₁ : x + (120 - x) = 120) (h₂ : 975 ≤ 15 * x + 5 * (120 - x)) (h₃ : 15 * x + 5 * (120 - x) ≤ 1000):
  x = 38 ∨ x = 39 ∨ x = 40 :=
sorry

theorem part_three_min_cost (x : ℕ) (h₁ : x = 38 ∨ x = 39 ∨ x = 40):
  ∃ min_cost, (min_cost = 10 * 38 + 600 ∧ min_cost ≤ 10 * x + 600) :=
sorry

end part_one_costs_part_two_feasible_values_part_three_min_cost_l1001_100162


namespace decompose_fraction1_decompose_fraction2_l1001_100175

-- Define the first problem as a theorem
theorem decompose_fraction1 (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (2 / (x^2 - 1)) = (1 / (x - 1)) - (1 / (x + 1)) :=
sorry  -- Proof required

-- Define the second problem as a theorem
theorem decompose_fraction2 (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (2 * x / (x^2 - 1)) = (1 / (x - 1)) + (1 / (x + 1)) :=
sorry  -- Proof required

end decompose_fraction1_decompose_fraction2_l1001_100175


namespace slices_with_all_toppings_l1001_100120

theorem slices_with_all_toppings (p m o a b c x total : ℕ) 
  (pepperoni_slices : p = 8)
  (mushrooms_slices : m = 12)
  (olives_slices : o = 14)
  (total_slices : total = 16)
  (inclusion_exclusion : p + m + o - a - b - c - 2 * x = total) :
  x = 4 := 
by
  rw [pepperoni_slices, mushrooms_slices, olives_slices, total_slices] at inclusion_exclusion
  sorry

end slices_with_all_toppings_l1001_100120


namespace proof_emails_in_morning_l1001_100190

def emailsInAfternoon : ℕ := 2

def emailsMoreInMorning : ℕ := 4

def emailsInMorning : ℕ := 6

theorem proof_emails_in_morning
  (a : ℕ) (h1 : a = emailsInAfternoon)
  (m : ℕ) (h2 : m = emailsMoreInMorning)
  : emailsInMorning = a + m := by
  sorry

end proof_emails_in_morning_l1001_100190


namespace turtle_hare_race_headstart_l1001_100150

noncomputable def hare_time_muddy (distance speed_reduction hare_speed : ℝ) : ℝ :=
  distance / (hare_speed * speed_reduction)

noncomputable def hare_time_sandy (distance hare_speed : ℝ) : ℝ :=
  distance / hare_speed

noncomputable def hare_time_regular (distance hare_speed : ℝ) : ℝ :=
  distance / hare_speed

noncomputable def turtle_time_muddy (distance turtle_speed : ℝ) : ℝ :=
  distance / turtle_speed

noncomputable def turtle_time_sandy (distance speed_increase turtle_speed : ℝ) : ℝ :=
  distance / (turtle_speed * speed_increase)

noncomputable def turtle_time_regular (distance turtle_speed : ℝ) : ℝ :=
  distance / turtle_speed

noncomputable def hare_total_time (hare_speed : ℝ) : ℝ :=
  hare_time_muddy 20 0.5 hare_speed + hare_time_sandy 10 hare_speed + hare_time_regular 20 hare_speed

noncomputable def turtle_total_time (turtle_speed : ℝ) : ℝ :=
  turtle_time_muddy 20 turtle_speed + turtle_time_sandy 10 1.5 turtle_speed + turtle_time_regular 20 turtle_speed

theorem turtle_hare_race_headstart (hare_speed turtle_speed : ℝ) (t_hs : ℝ) :
  hare_speed = 10 →
  turtle_speed = 1 →
  t_hs = 39.67 →
  hare_total_time hare_speed + t_hs = turtle_total_time turtle_speed :=
by
  intros 
  sorry

end turtle_hare_race_headstart_l1001_100150


namespace rectangle_measurement_error_l1001_100119

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) 
  (h1 : 0 < L) (h2 : 0 < W) 
  (h3 : A = L * W)
  (h4 : A' = L * (1 + x / 100) * W * (1 - 4 / 100))
  (h5 : A' = A * (100.8 / 100)) :
  x = 5 :=
by
  sorry

end rectangle_measurement_error_l1001_100119


namespace simplify_proof_l1001_100167

noncomputable def simplify_expression (a b c d x y : ℝ) (h : c * x ≠ d * y) : ℝ :=
  (c * x * (b^2 * x^2 - 4 * b^2 * y^2 + a^2 * y^2) 
  - d * y * (b^2 * x^2 - 2 * a^2 * x^2 - 3 * a^2 * y^2)) / (c * x - d * y)

theorem simplify_proof (a b c d x y : ℝ) (h : c * x ≠ d * y) :
  simplify_expression a b c d x y h = b^2 * x^2 + a^2 * y^2 :=
by sorry

end simplify_proof_l1001_100167


namespace find_m_l1001_100194

-- Definition and conditions
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

noncomputable def vertex_property (a b c : ℝ) : Prop := 
  (∀ x, quadratic a b c x ≤ quadratic a b c 2) ∧ quadratic a b c 2 = 4

noncomputable def passes_through_origin (a b c : ℝ) : Prop :=
  quadratic a b c 0 = -7

-- Main theorem statement
theorem find_m (a b c m : ℝ) 
  (h1 : vertex_property a b c) 
  (h2 : passes_through_origin a b c) 
  (h3 : quadratic a b c 5 = m) :
  m = -83/4 :=
sorry

end find_m_l1001_100194


namespace initial_discount_percentage_l1001_100170

-- Statement of the problem
theorem initial_discount_percentage (d : ℝ) (x : ℝ)
  (h₁ : d > 0)
  (h_staff_price : d * ((100 - x) / 100) * 0.5 = 0.225 * d) :
  x = 55 := 
sorry

end initial_discount_percentage_l1001_100170


namespace robin_gum_total_l1001_100158

theorem robin_gum_total :
  let original_gum := 18.0
  let given_gum := 44.0
  original_gum + given_gum = 62.0 := by
  sorry

end robin_gum_total_l1001_100158


namespace pool_filling_time_l1001_100187

theorem pool_filling_time :
  (∀ t : ℕ, t >= 6 → ∃ v : ℝ, v = (2^(t-6)) * 0.25) →
  ∃ t : ℕ, t = 8 :=
by
  intros h
  existsi 8
  sorry

end pool_filling_time_l1001_100187


namespace option_B_is_one_variable_quadratic_l1001_100184

theorem option_B_is_one_variable_quadratic :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x : ℝ, 2 * (x - x^2) - 1 = a * x^2 + b * x + c) :=
by
  sorry

end option_B_is_one_variable_quadratic_l1001_100184


namespace simplify_expression_l1001_100116

theorem simplify_expression (a : ℝ) (h : a < 1 / 4) : 4 * (4 * a - 1)^2 = (1 - 4 * a) ^ 2 :=
by
  sorry

end simplify_expression_l1001_100116


namespace horner_evaluation_at_3_l1001_100149

def f (x : ℤ) : ℤ := x^5 + 2 * x^3 + 3 * x^2 + x + 1

theorem horner_evaluation_at_3 : f 3 = 328 := by
  sorry

end horner_evaluation_at_3_l1001_100149


namespace mod_11_residue_l1001_100179

theorem mod_11_residue : 
  ((312 - 3 * 52 + 9 * 165 + 6 * 22) % 11) = 2 :=
by
  sorry

end mod_11_residue_l1001_100179


namespace cycling_route_length_l1001_100133

-- Conditions (segment lengths)
def segment1 : ℝ := 4
def segment2 : ℝ := 7
def segment3 : ℝ := 2
def segment4 : ℝ := 6
def segment5 : ℝ := 7

-- Specify the total length calculation
noncomputable def total_length : ℝ :=
  2 * (segment1 + segment2 + segment3) + 2 * (segment4 + segment5)

-- The theorem we want to prove
theorem cycling_route_length :
  total_length = 52 :=
by
  sorry

end cycling_route_length_l1001_100133


namespace min_performances_l1001_100178

theorem min_performances (n_pairs_per_show m n_singers : ℕ) (h1 : n_singers = 8) (h2 : n_pairs_per_show = 6) 
  (condition : 6 * m = 28 * 3) : m = 14 :=
by
  -- Use the assumptions to prove the statement
  sorry

end min_performances_l1001_100178


namespace first_other_factor_of_lcm_l1001_100108

theorem first_other_factor_of_lcm (A B hcf lcm : ℕ) (h1 : A = 368) (h2 : hcf = 23) (h3 : lcm = hcf * 16 * X) :
  X = 1 :=
by
  sorry

end first_other_factor_of_lcm_l1001_100108


namespace sum_of_edges_96_l1001_100180

noncomputable def volume (a r : ℝ) : ℝ := 
  (a / r) * a * (a * r)

noncomputable def surface_area (a r : ℝ) : ℝ := 
  2 * ((a^2) / r + a^2 + a^2 * r)

noncomputable def sum_of_edges (a r : ℝ) : ℝ := 
  4 * ((a / r) + a + (a * r))

theorem sum_of_edges_96 :
  (∃ (a r : ℝ), volume a r = 512 ∧ surface_area a r = 384 ∧ sum_of_edges a r = 96) :=
by
  have a := 8
  have r := 1
  have h_volume : volume a r = 512 := sorry
  have h_surface_area : surface_area a r = 384 := sorry
  have h_sum_of_edges : sum_of_edges a r = 96 := sorry
  exact ⟨a, r, h_volume, h_surface_area, h_sum_of_edges⟩

end sum_of_edges_96_l1001_100180


namespace tangent_line_right_triangle_l1001_100151

theorem tangent_line_right_triangle {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (tangent_condition : a^2 + b^2 = c^2) : 
  (abs c)^2 = (abs a)^2 + (abs b)^2 :=
by
  sorry

end tangent_line_right_triangle_l1001_100151


namespace length_PQ_is_5_l1001_100129

/-
Given:
- Point P with coordinates (3, 4, 5)
- Point Q is the projection of P onto the xOy plane

Show:
- The length of the segment PQ is 5
-/

def P : ℝ × ℝ × ℝ := (3, 4, 5)
def Q : ℝ × ℝ × ℝ := (3, 4, 0)

theorem length_PQ_is_5 : dist P Q = 5 := by
  sorry

end length_PQ_is_5_l1001_100129


namespace abs_neg_three_l1001_100132

theorem abs_neg_three : |(-3 : ℤ)| = 3 := 
by
  sorry

end abs_neg_three_l1001_100132


namespace least_value_x_y_z_l1001_100148

theorem least_value_x_y_z (x y z : ℕ) (hx : x = 4 * y) (hy : y = 7 * z) (hz : 0 < z) : x - y - z = 19 :=
by
  -- placeholder for actual proof
  sorry

end least_value_x_y_z_l1001_100148


namespace average_speed_of_train_l1001_100191

-- Definitions based on the conditions
def distance1 : ℝ := 325
def distance2 : ℝ := 470
def time1 : ℝ := 3.5
def time2 : ℝ := 4

-- Proof statement
theorem average_speed_of_train :
  (distance1 + distance2) / (time1 + time2) = 106 := 
by 
  sorry

end average_speed_of_train_l1001_100191


namespace rocket_altitude_time_l1001_100198

theorem rocket_altitude_time (a₁ d : ℕ) (n : ℕ) (h₁ : a₁ = 2) (h₂ : d = 2)
  (h₃ : n * a₁ + (n * (n - 1) * d) / 2 = 240) : n = 15 :=
by
  -- The proof is ignored as per instruction.
  sorry

end rocket_altitude_time_l1001_100198


namespace time_difference_leak_l1001_100113

/-- 
The machine usually fills one barrel in 3 minutes. 
However, with a leak, it takes 5 minutes to fill one barrel. 
Given that it takes 24 minutes longer to fill 12 barrels with the leak, prove that it will take 2n minutes longer to fill n barrels with the leak.
-/
theorem time_difference_leak (n : ℕ) : 
  (3 * 12 + 24 = 5 * 12) →
  (5 * n) - (3 * n) = 2 * n :=
by
  intros h
  sorry

end time_difference_leak_l1001_100113


namespace octopus_legs_l1001_100153

-- Definitions of octopus behavior based on the number of legs
def tells_truth (legs: ℕ) : Prop := legs = 6 ∨ legs = 8
def lies (legs: ℕ) : Prop := legs = 7

-- Statements made by the octopuses
def blue_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 28
def green_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 27
def yellow_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 26
def red_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 25

noncomputable def legs_b := 7
noncomputable def legs_g := 6
noncomputable def legs_y := 7
noncomputable def legs_r := 7

-- Main theorem
theorem octopus_legs : 
  (tells_truth legs_g) ∧ 
  (lies legs_b) ∧ 
  (lies legs_y) ∧ 
  (lies legs_r) ∧ 
  blue_statement legs_b legs_g legs_y legs_r ∧ 
  green_statement legs_b legs_g legs_y legs_r ∧ 
  yellow_statement legs_b legs_g legs_y legs_r ∧ 
  red_statement legs_b legs_g legs_y legs_r := 
by 
  sorry

end octopus_legs_l1001_100153


namespace annual_interest_rate_l1001_100137

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) (r : ℝ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

theorem annual_interest_rate :
  compound_interest_rate 150 181.50 2 1 (0.2 : ℝ) :=
by
  unfold compound_interest_rate
  sorry

end annual_interest_rate_l1001_100137


namespace number_of_seats_in_classroom_l1001_100161

theorem number_of_seats_in_classroom 
    (seats_per_row_condition : 7 + 13 = 19) 
    (rows_condition : 8 + 14 = 21) : 
    19 * 21 = 399 := 
by 
    sorry

end number_of_seats_in_classroom_l1001_100161


namespace bruce_purchased_mangoes_l1001_100199

-- Condition definitions
def cost_of_grapes (k_gra kg_cost_gra : ℕ) : ℕ := k_gra * kg_cost_gra
def amount_spent_on_mangoes (total_paid cost_gra : ℕ) : ℕ := total_paid - cost_gra
def quantity_of_mangoes (total_amt_mangoes rate_per_kg_mangoes : ℕ) : ℕ := total_amt_mangoes / rate_per_kg_mangoes

-- Parameters
variable (k_gra rate_per_kg_gra rate_per_kg_mangoes total_paid : ℕ)
variable (kg_gra_total_amt spent_amt_mangoes_qty : ℕ)

-- Given values
axiom A1 : k_gra = 7
axiom A2 : rate_per_kg_gra = 70
axiom A3 : rate_per_kg_mangoes = 55
axiom A4 : total_paid = 985

-- Calculations based on conditions
axiom H1 : cost_of_grapes k_gra rate_per_kg_gra = kg_gra_total_amt
axiom H2 : amount_spent_on_mangoes total_paid kg_gra_total_amt = spent_amt_mangoes_qty
axiom H3 : quantity_of_mangoes spent_amt_mangoes_qty rate_per_kg_mangoes = 9

-- Proof statement to be proven
theorem bruce_purchased_mangoes : quantity_of_mangoes spent_amt_mangoes_qty rate_per_kg_mangoes = 9 := sorry

end bruce_purchased_mangoes_l1001_100199


namespace operation_two_three_l1001_100130

def operation (a b : ℕ) : ℤ := 4 * a ^ 2 - 4 * b ^ 2

theorem operation_two_three : operation 2 3 = -20 :=
by
  sorry

end operation_two_three_l1001_100130


namespace solution_set_of_inequality_l1001_100145

theorem solution_set_of_inequality (x : ℝ) : x^2 - |x| - 2 < 0 ↔ -2 < x ∧ x < 2 := 
by
  sorry

end solution_set_of_inequality_l1001_100145


namespace fraction_sum_eq_neg_one_l1001_100186

theorem fraction_sum_eq_neg_one (p q : ℝ) (hpq : (1 / p) + (1 / q) = (1 / (p + q))) :
  (p / q) + (q / p) = -1 :=
by
  sorry

end fraction_sum_eq_neg_one_l1001_100186


namespace total_number_recruits_l1001_100181

theorem total_number_recruits 
  (x y z : ℕ)
  (h1 : x = 50)
  (h2 : y = 100)
  (h3 : z = 170)
  (h4 : x = 4 * (y - 50) ∨ y = 4 * (z - 170) ∨ x = 4 * (z - 170)) : 
  171 + (z - 170) = 211 :=
by
  sorry

end total_number_recruits_l1001_100181


namespace xiao_ming_actual_sleep_time_l1001_100128

def required_sleep_time : ℝ := 9
def recorded_excess_sleep_time : ℝ := 0.4
def actual_sleep_time (required : ℝ) (excess : ℝ) : ℝ := required + excess

theorem xiao_ming_actual_sleep_time :
  actual_sleep_time required_sleep_time recorded_excess_sleep_time = 9.4 := 
by
  sorry

end xiao_ming_actual_sleep_time_l1001_100128


namespace xy_yz_zx_equal_zero_l1001_100106

noncomputable def side1 (x y z : ℝ) : ℝ := 1 / abs (x^2 + 2 * y * z)
noncomputable def side2 (x y z : ℝ) : ℝ := 1 / abs (y^2 + 2 * z * x)
noncomputable def side3 (x y z : ℝ) : ℝ := 1 / abs (z^2 + 2 * x * y)

def non_degenerate_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem xy_yz_zx_equal_zero
  (x y z : ℝ)
  (h1 : non_degenerate_triangle (side1 x y z) (side2 x y z) (side3 x y z)) :
  xy + yz + zx = 0 := sorry

end xy_yz_zx_equal_zero_l1001_100106


namespace intersection_of_A_and_B_l1001_100107

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | x > 1}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end intersection_of_A_and_B_l1001_100107


namespace flour_masses_l1001_100157

theorem flour_masses (x : ℝ) (h: 
    (x * (1 + x / 100) + (x + 10) * (1 + (x + 10) / 100) = 112.5)) :
    x = 35 ∧ (x + 10) = 45 :=
by 
  sorry

end flour_masses_l1001_100157


namespace unique_shape_determination_l1001_100188

theorem unique_shape_determination (ratio_sides_median : Prop) (ratios_three_sides : Prop) 
                                   (ratio_circumradius_side : Prop) (ratio_two_angles : Prop) 
                                   (length_one_side_heights : Prop) :
  ¬(ratio_circumradius_side → (ratio_sides_median ∧ ratios_three_sides ∧ ratio_two_angles ∧ length_one_side_heights)) := 
sorry

end unique_shape_determination_l1001_100188


namespace intersection_M_N_l1001_100147

-- Defining set M
def M : Set ℕ := {1, 2, 3, 4}

-- Defining the set N based on the condition
def N : Set ℕ := {x | ∃ n ∈ M, x = n^2}

-- Lean statement to prove the intersection
theorem intersection_M_N : M ∩ N = {1, 4} := 
by
  sorry

end intersection_M_N_l1001_100147


namespace pradeep_failed_marks_l1001_100165

theorem pradeep_failed_marks
    (total_marks : ℕ)
    (obtained_marks : ℕ)
    (pass_percentage : ℕ)
    (pass_marks : ℕ)
    (fail_marks : ℕ)
    (total_marks_eq : total_marks = 2075)
    (obtained_marks_eq : obtained_marks = 390)
    (pass_percentage_eq : pass_percentage = 20)
    (pass_marks_eq : pass_marks = (pass_percentage * total_marks) / 100)
    (fail_marks_eq : fail_marks = pass_marks - obtained_marks) :
    fail_marks = 25 :=
by
  rw [total_marks_eq, obtained_marks_eq, pass_percentage_eq] at *
  sorry

end pradeep_failed_marks_l1001_100165


namespace solve_quadratic_eqn_l1001_100169

theorem solve_quadratic_eqn:
  (∃ x: ℝ, (x + 10)^2 = (4 * x + 6) * (x + 8)) ↔ 
  (∀ x: ℝ, x = 2.131 ∨ x = -8.131) := 
by
  sorry

end solve_quadratic_eqn_l1001_100169


namespace man_born_in_1892_l1001_100100

-- Define the conditions and question
def man_birth_year (x : ℕ) : ℕ :=
x^2 - x

-- Conditions:
variable (x : ℕ)
-- 1. The man was born in the first half of the 20th century
variable (h1 : man_birth_year x < 1950)
-- 2. The man's age x and the conditions in the problem
variable (h2 : x^2 - x < 1950)

-- The statement we aim to prove
theorem man_born_in_1892 (x : ℕ) (h1 : man_birth_year x < 1950) (h2 : x = 44) : man_birth_year x = 1892 := by
  sorry

end man_born_in_1892_l1001_100100


namespace find_largest_beta_l1001_100127

theorem find_largest_beta (α : ℝ) (r : ℕ → ℝ) (C : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < 1)
  (h3 : ∀ n, ∀ m ≠ n, dist (r n) (r m) ≥ (r n) ^ α)
  (h4 : ∀ n, r n ≤ r (n + 1)) 
  (h5 : ∀ n, r n ≥ C * n ^ (1 / (2 * (1 - α)))) :
  ∀ β, (∃ C > 0, ∀ n, r n ≥ C * n ^ β) → β ≤ 1 / (2 * (1 - α)) :=
sorry

end find_largest_beta_l1001_100127


namespace math_competition_l1001_100154

theorem math_competition (a b c d e f g : ℕ) (h1 : a + b + c + d + e + f + g = 25)
    (h2 : b = 2 * c + f) (h3 : a = d + e + g + 1) (h4 : a = b + c) :
    b = 6 :=
by
  -- The proof is omitted as the problem requests the statement only.
  sorry

end math_competition_l1001_100154


namespace selected_40th_is_795_l1001_100122

-- Definitions of constants based on the problem conditions
def total_participants : ℕ := 1000
def selections : ℕ := 50
def equal_spacing : ℕ := total_participants / selections
def first_selected_number : ℕ := 15
def nth_selected_number (n : ℕ) : ℕ := (n - 1) * equal_spacing + first_selected_number

-- The theorem to prove the 40th selected number is 795
theorem selected_40th_is_795 : nth_selected_number 40 = 795 := 
by 
  -- Skipping the detailed proof
  sorry

end selected_40th_is_795_l1001_100122


namespace initial_concentration_l1001_100168

theorem initial_concentration (C : ℝ) 
  (hC : (C * 0.2222222222222221) + (0.25 * 0.7777777777777779) = 0.35) :
  C = 0.7 :=
sorry

end initial_concentration_l1001_100168


namespace problem_l1001_100123

theorem problem (r : ℝ) (h : (r + 1/r)^4 = 17) : r^6 + 1/r^6 = 1 * Real.sqrt 17 - 6 :=
sorry

end problem_l1001_100123


namespace minimize_sum_of_reciprocals_l1001_100118

def dataset : List ℝ := [2, 4, 6, 8]

def mean : ℝ := 5
def variance: ℝ := 5

theorem minimize_sum_of_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : mean * a + variance * b = 1) : 
  (1 / a + 1 / b) = 20 :=
sorry

end minimize_sum_of_reciprocals_l1001_100118


namespace domain_of_sqrt_function_l1001_100176

theorem domain_of_sqrt_function (x : ℝ) :
  (x + 4 ≥ 0) ∧ (1 - x ≥ 0) ∧ (x ≠ 0) ↔ (-4 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1) := 
sorry

end domain_of_sqrt_function_l1001_100176


namespace contemporaries_probability_l1001_100177

open Real

noncomputable def probability_of_contemporaries
  (born_within : ℝ) (lifespan : ℝ) : ℝ :=
  let total_area := born_within * born_within
  let side := born_within - lifespan
  let non_overlap_area := 2 * (1/2 * side * side)
  let overlap_area := total_area - non_overlap_area
  overlap_area / total_area

theorem contemporaries_probability :
  probability_of_contemporaries 300 80 = 104 / 225 := 
by
  sorry

end contemporaries_probability_l1001_100177


namespace range_of_m_l1001_100144

noncomputable def f (m x : ℝ) : ℝ := 2 * m * x^2 - 2 * (4 - m) * x + 1
noncomputable def g (m x : ℝ) : ℝ := m * x

theorem range_of_m :
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) → 0 < m ∧ m < 8 :=
sorry

end range_of_m_l1001_100144


namespace set_equality_l1001_100182

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hA : A = {1, 4})
variable (hB : B = {3, 4})

theorem set_equality : ({2, 5} : Set ℕ) = U \ (A ∪ B) :=
by
  sorry

end set_equality_l1001_100182


namespace loop_condition_l1001_100189

theorem loop_condition (b : ℕ) : (b = 10 ∧ ∀ n, b = 10 + 3 * n ∧ b < 16 → n + 1 = 16) → ∀ (condition : ℕ → Prop), condition b → b = 16 :=
by sorry

end loop_condition_l1001_100189


namespace candy_proof_l1001_100105

variable (x s t : ℤ)

theorem candy_proof (H1 : 4 * x - 15 * s = 23)
                    (H2 : 5 * x - 23 * t = 15) :
  x = 302 := by
  sorry

end candy_proof_l1001_100105


namespace ratio_of_M_to_N_l1001_100104

theorem ratio_of_M_to_N 
  (M Q P N : ℝ) 
  (h1 : M = 0.4 * Q) 
  (h2 : Q = 0.25 * P) 
  (h3 : N = 0.75 * P) : 
  M / N = 2 / 15 := 
sorry

end ratio_of_M_to_N_l1001_100104


namespace work_days_l1001_100143

theorem work_days (A B C : ℝ) (h₁ : A + B = 1 / 15) (h₂ : C = 1 / 7.5) : 1 / (A + B + C) = 5 :=
by
  sorry

end work_days_l1001_100143
