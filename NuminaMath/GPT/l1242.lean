import Mathlib

namespace NUMINAMATH_GPT_difference_of_numbers_l1242_124271

variable (x y : ℝ)

theorem difference_of_numbers (h1 : x + y = 10) (h2 : x - y = 19) (h3 : x^2 - y^2 = 190) : x - y = 19 :=
by sorry

end NUMINAMATH_GPT_difference_of_numbers_l1242_124271


namespace NUMINAMATH_GPT_not_forall_abs_ge_zero_l1242_124277

theorem not_forall_abs_ge_zero : (¬(∀ x : ℝ, |x + 1| ≥ 0)) ↔ (∃ x : ℝ, |x + 1| < 0) :=
by
  sorry

end NUMINAMATH_GPT_not_forall_abs_ge_zero_l1242_124277


namespace NUMINAMATH_GPT_exams_in_fourth_year_l1242_124221

variable (a b c d e : ℕ)

theorem exams_in_fourth_year:
  a + b + c + d + e = 31 ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e = 3 * a → d = 8 := by
  sorry

end NUMINAMATH_GPT_exams_in_fourth_year_l1242_124221


namespace NUMINAMATH_GPT_min_sum_of_2x2_grid_l1242_124253

theorem min_sum_of_2x2_grid (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
(h_sum : a * b + c * d + a * c + b * d = 2015) : a + b + c + d = 88 :=
sorry

end NUMINAMATH_GPT_min_sum_of_2x2_grid_l1242_124253


namespace NUMINAMATH_GPT_roots_of_quadratic_equation_are_real_and_distinct_l1242_124217

def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_of_quadratic_equation_are_real_and_distinct :
  quadratic_discriminant 1 (-2) (-6) > 0 :=
by
  norm_num
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_equation_are_real_and_distinct_l1242_124217


namespace NUMINAMATH_GPT_parallel_lines_distance_l1242_124264

theorem parallel_lines_distance (b c : ℝ) 
  (h1: b = 8) 
  (h2: (abs (10 - c) / (Real.sqrt (3^2 + 4^2))) = 3) :
  b + c = -12 ∨ b + c = 48 := by
 sorry

end NUMINAMATH_GPT_parallel_lines_distance_l1242_124264


namespace NUMINAMATH_GPT_fifteenth_term_of_geometric_sequence_l1242_124266

theorem fifteenth_term_of_geometric_sequence :
  let a := 12
  let r := (1:ℚ) / 3
  let n := 15
  (a * r^(n-1)) = (4 / 1594323:ℚ)
:=
  by
    sorry

end NUMINAMATH_GPT_fifteenth_term_of_geometric_sequence_l1242_124266


namespace NUMINAMATH_GPT_cost_of_fencing_per_meter_in_cents_l1242_124249

-- Definitions for the conditions
def ratio_length_width : ℕ := 3
def ratio_width_length : ℕ := 2
def total_area : ℕ := 3750
def total_fencing_cost : ℕ := 175

-- Main theorem statement with proof omitted
theorem cost_of_fencing_per_meter_in_cents :
  (ratio_length_width = 3) →
  (ratio_width_length = 2) →
  (total_area = 3750) →
  (total_fencing_cost = 175) →
  ∃ (cost_per_meter_in_cents : ℕ), cost_per_meter_in_cents = 70 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_cost_of_fencing_per_meter_in_cents_l1242_124249


namespace NUMINAMATH_GPT_tangent_line_to_parabola_l1242_124269

-- Define the line and parabola equations
def line (x y k : ℝ) := 4 * x + 3 * y + k = 0
def parabola (x y : ℝ) := y ^ 2 = 16 * x

-- Prove that if the line is tangent to the parabola, then k = 9
theorem tangent_line_to_parabola (k : ℝ) :
  (∃ (x y : ℝ), line x y k ∧ parabola x y ∧ (y^2 + 12 * y + 4 * k = 0 ∧ 144 - 16 * k = 0)) → k = 9 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_to_parabola_l1242_124269


namespace NUMINAMATH_GPT_problem_solution_l1242_124265

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem problem_solution (f : ℝ → ℝ)
  (H1 : even_function f)
  (H2 : ∀ x, f (x + 4) = -f x)
  (H3 : ∀ ⦃x y⦄, 0 ≤ x → x < y → y ≤ 4 → f y < f x) :
  f 13 < f 10 ∧ f 10 < f 15 :=
  by
    sorry

end NUMINAMATH_GPT_problem_solution_l1242_124265


namespace NUMINAMATH_GPT_largest_part_of_proportional_division_l1242_124250

theorem largest_part_of_proportional_division (sum : ℚ) (a b c largest : ℚ) 
  (prop1 prop2 prop3 : ℚ) 
  (h1 : sum = 156)
  (h2 : prop1 = 2)
  (h3 : prop2 = 1 / 2)
  (h4 : prop3 = 1 / 4)
  (h5 : sum = a + b + c)
  (h6 : a / prop1 = b / prop2 ∧ b / prop2 = c / prop3)
  (h7 : largest = max a (max b c)) :
  largest = 112 + 8 / 11 :=
by
  sorry

end NUMINAMATH_GPT_largest_part_of_proportional_division_l1242_124250


namespace NUMINAMATH_GPT_tan_product_ge_sqrt2_l1242_124227

variable {α β γ : ℝ}

theorem tan_product_ge_sqrt2 (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) 
  (h : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) : 
  Real.tan α * Real.tan β * Real.tan γ ≥ 2 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_tan_product_ge_sqrt2_l1242_124227


namespace NUMINAMATH_GPT_number_in_scientific_notation_l1242_124270

def scientific_notation_form (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ a ∧ a < 10

theorem number_in_scientific_notation : scientific_notation_form 3.7515 7 ∧ 37515000 = 3.7515 * 10^7 :=
by
  sorry

end NUMINAMATH_GPT_number_in_scientific_notation_l1242_124270


namespace NUMINAMATH_GPT_fraction_sent_afternoon_l1242_124205

-- Defining the problem conditions
def total_fliers : ℕ := 1000
def fliers_sent_morning : ℕ := total_fliers * 1/5
def fliers_left_afternoon : ℕ := total_fliers - fliers_sent_morning
def fliers_left_next_day : ℕ := 600
def fliers_sent_afternoon : ℕ := fliers_left_afternoon - fliers_left_next_day

-- Proving the fraction of fliers sent in the afternoon
theorem fraction_sent_afternoon : (fliers_sent_afternoon : ℚ) / fliers_left_afternoon = 1/4 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_fraction_sent_afternoon_l1242_124205


namespace NUMINAMATH_GPT_all_children_receive_candy_iff_power_of_two_l1242_124232

theorem all_children_receive_candy_iff_power_of_two (n : ℕ) : 
  (∀ (k : ℕ), k < n → ∃ (m : ℕ), (m * (m + 1) / 2) % n = k) ↔ ∃ (k : ℕ), n = 2^k :=
by sorry

end NUMINAMATH_GPT_all_children_receive_candy_iff_power_of_two_l1242_124232


namespace NUMINAMATH_GPT_choose_4_out_of_10_l1242_124211

theorem choose_4_out_of_10 : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_GPT_choose_4_out_of_10_l1242_124211


namespace NUMINAMATH_GPT_quadrant_of_theta_l1242_124261

theorem quadrant_of_theta (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.sin θ < 0) : (0 < θ ∧ θ < π/2) ∨ (3*π/2 < θ ∧ θ < 2*π) :=
by
  sorry

end NUMINAMATH_GPT_quadrant_of_theta_l1242_124261


namespace NUMINAMATH_GPT_intersection_of_sets_l1242_124209

open Set

variable {x : ℝ}

theorem intersection_of_sets : 
  let A := {x : ℝ | x^2 - 4*x + 3 < 0}
  let B := {x : ℝ | x > 2}
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1242_124209


namespace NUMINAMATH_GPT_robins_total_pieces_of_gum_l1242_124296

theorem robins_total_pieces_of_gum :
  let initial_packages := 27
  let pieces_per_initial_package := 18
  let additional_packages := 15
  let pieces_per_additional_package := 12
  let more_packages := 8
  let pieces_per_more_package := 25
  (initial_packages * pieces_per_initial_package) +
  (additional_packages * pieces_per_additional_package) +
  (more_packages * pieces_per_more_package) = 866 :=
by
  sorry

end NUMINAMATH_GPT_robins_total_pieces_of_gum_l1242_124296


namespace NUMINAMATH_GPT_problem1_part1_problem1_part2_problem2_l1242_124246

open Set

-- Definitions for sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def U : Set ℝ := univ

-- Part (1) of the problem
theorem problem1_part1 : A ∩ B = {x | 2 < x ∧ x ≤ 3} :=
sorry

theorem problem1_part2 : A ∪ (U \ B) = {x | x ≤ 3} :=
sorry

-- Definitions for set C
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- Part (2) of the problem
theorem problem2 (a : ℝ) (h : C a ⊆ A) : 1 < a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_GPT_problem1_part1_problem1_part2_problem2_l1242_124246


namespace NUMINAMATH_GPT_choir_member_count_l1242_124233

theorem choir_member_count (n : ℕ) : 
  (n ≡ 4 [MOD 7]) ∧ 
  (n ≡ 8 [MOD 6]) ∧ 
  (50 ≤ n ∧ n ≤ 200) 
  ↔ 
  (n = 60 ∨ n = 102 ∨ n = 144 ∨ n = 186) := 
by 
  sorry

end NUMINAMATH_GPT_choir_member_count_l1242_124233


namespace NUMINAMATH_GPT_area_of_circle_B_l1242_124220

theorem area_of_circle_B (rA rB : ℝ) (h : π * rA^2 = 16 * π) (h1 : rB = 2 * rA) : π * rB^2 = 64 * π :=
by
  sorry

end NUMINAMATH_GPT_area_of_circle_B_l1242_124220


namespace NUMINAMATH_GPT_rhyme_around_3_7_l1242_124252

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def rhymes_around (p q m : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p ≠ q ∧ ((p < m ∧ q > m ∧ q - m = m - p) ∨ (p > m ∧ q < m ∧ p - m = m - q))

theorem rhyme_around_3_7 : ∃ m : ℕ, rhymes_around 3 7 m ∧ m = 5 :=
by
  sorry

end NUMINAMATH_GPT_rhyme_around_3_7_l1242_124252


namespace NUMINAMATH_GPT_figure_F10_squares_l1242_124280

def num_squares (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 1 + 3 * (n - 1) * n

theorem figure_F10_squares : num_squares 10 = 271 :=
by sorry

end NUMINAMATH_GPT_figure_F10_squares_l1242_124280


namespace NUMINAMATH_GPT_seq_problem_part1_seq_problem_part2_l1242_124202

def seq (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1) - a n|

theorem seq_problem_part1 (a : ℕ → ℤ) 
  (h1 : a 1 = 2)
  (h2 : a 2 = -1)
  (h_seq : seq a) :
  a 2008 = 0 := 
sorry

theorem seq_problem_part2 (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : a 2 = -1)
  (h_seq : seq a) :
  ∃ (M : ℤ), 
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ a m = 0) ∧ 
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ a m = M) := 
sorry

end NUMINAMATH_GPT_seq_problem_part1_seq_problem_part2_l1242_124202


namespace NUMINAMATH_GPT_compute_expression_l1242_124286

theorem compute_expression : 
  let x := 19
  let y := 15
  (x + y)^2 - (x - y)^2 = 1140 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1242_124286


namespace NUMINAMATH_GPT_probability_of_A_winning_l1242_124201

-- Define the conditions
def p : ℝ := 0.6
def q : ℝ := 1 - p  -- probability of losing a set

-- Formulate the probabilities for each win scenario
def P_WW : ℝ := p * p
def P_LWW : ℝ := q * p * p
def P_WLW : ℝ := p * q * p

-- Calculate the total probability of winning the match
def total_probability : ℝ := P_WW + P_LWW + P_WLW

-- Prove that the total probability of A winning the match is 0.648
theorem probability_of_A_winning : total_probability = 0.648 :=
by
    -- Provide the calculation details
    sorry  -- replace with the actual proof steps if needed, otherwise keep sorry to skip the proof

end NUMINAMATH_GPT_probability_of_A_winning_l1242_124201


namespace NUMINAMATH_GPT_cos_double_angle_l1242_124231

open Real

theorem cos_double_angle (α : Real) (h : tan α = 3) : cos (2 * α) = -4/5 :=
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1242_124231


namespace NUMINAMATH_GPT_actual_time_of_storm_l1242_124234

theorem actual_time_of_storm
  (malfunctioned_hours_tens_digit : ℕ)
  (malfunctioned_hours_units_digit : ℕ)
  (malfunctioned_minutes_tens_digit : ℕ)
  (malfunctioned_minutes_units_digit : ℕ)
  (original_time : ℕ × ℕ)
  (hours_tens_digit : ℕ := 2)
  (hours_units_digit : ℕ := 0)
  (minutes_tens_digit : ℕ := 0)
  (minutes_units_digit : ℕ := 9) :
  (malfunctioned_hours_tens_digit = hours_tens_digit + 1 ∨ malfunctioned_hours_tens_digit = hours_tens_digit - 1) →
  (malfunctioned_hours_units_digit = hours_units_digit + 1 ∨ malfunctioned_hours_units_digit = hours_units_digit - 1) →
  (malfunctioned_minutes_tens_digit = minutes_tens_digit + 1 ∨ malfunctioned_minutes_tens_digit = minutes_tens_digit - 1) →
  (malfunctioned_minutes_units_digit = minutes_units_digit + 1 ∨ malfunctioned_minutes_units_digit = minutes_units_digit - 1) →
  original_time = (11, 18) :=
by
  sorry

end NUMINAMATH_GPT_actual_time_of_storm_l1242_124234


namespace NUMINAMATH_GPT_binary_101111_to_decimal_is_47_decimal_47_to_octal_is_57_l1242_124299

def binary_to_decimal (b : ℕ) : ℕ :=
  32 + 0 + 8 + 4 + 2 + 1 -- Calculated manually for simplicity

def decimal_to_octal (d : ℕ) : ℕ :=
  (5 * 10) + 7 -- Manually converting decimal 47 to octal 57 for simplicity

theorem binary_101111_to_decimal_is_47 : binary_to_decimal 0b101111 = 47 := 
by sorry

theorem decimal_47_to_octal_is_57 : decimal_to_octal 47 = 57 := 
by sorry

end NUMINAMATH_GPT_binary_101111_to_decimal_is_47_decimal_47_to_octal_is_57_l1242_124299


namespace NUMINAMATH_GPT_train_speed_km_per_hr_l1242_124255

theorem train_speed_km_per_hr 
  (length : ℝ) 
  (time : ℝ) 
  (h_length : length = 150) 
  (h_time : time = 9.99920006399488) : 
  length / time * 3.6 = 54.00287976961843 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_km_per_hr_l1242_124255


namespace NUMINAMATH_GPT_evaluate_square_difference_l1242_124244

theorem evaluate_square_difference:
  let a := 70
  let b := 30
  (a^2 - b^2) = 4000 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_square_difference_l1242_124244


namespace NUMINAMATH_GPT_moli_initial_payment_l1242_124257

variable (R C S M : ℕ)

-- Conditions
def condition1 : Prop := 3 * R + 7 * C + 1 * S = M
def condition2 : Prop := 4 * R + 10 * C + 1 * S = 164
def condition3 : Prop := 1 * R + 1 * C + 1 * S = 32

theorem moli_initial_payment : condition1 R C S M ∧ condition2 R C S ∧ condition3 R C S → M = 120 := by
  sorry

end NUMINAMATH_GPT_moli_initial_payment_l1242_124257


namespace NUMINAMATH_GPT_election_total_votes_l1242_124297

theorem election_total_votes
  (V : ℝ)
  (h1 : 0 ≤ V) 
  (h_majority : 0.70 * V - 0.30 * V = 182) :
  V = 455 := 
by 
  sorry

end NUMINAMATH_GPT_election_total_votes_l1242_124297


namespace NUMINAMATH_GPT_jane_uses_40_ribbons_l1242_124213

theorem jane_uses_40_ribbons :
  (∀ dresses1 dresses2 ribbons_per_dress, 
  dresses1 = 2 * 7 ∧ 
  dresses2 = 3 * 2 → 
  ribbons_per_dress = 2 → 
  (dresses1 + dresses2) * ribbons_per_dress = 40)
:= 
by 
  sorry

end NUMINAMATH_GPT_jane_uses_40_ribbons_l1242_124213


namespace NUMINAMATH_GPT_sub_eight_l1242_124215

theorem sub_eight (x y : ℝ) (h1 : 3 = 0.15 * x) (h2 : 3 = 0.25 * y) : x - y = 8 :=
by
  sorry

end NUMINAMATH_GPT_sub_eight_l1242_124215


namespace NUMINAMATH_GPT_inequality_solution_l1242_124236

theorem inequality_solution
  (f : ℝ → ℝ)
  (h_deriv : ∀ x : ℝ, deriv f x > 2 * f x)
  (h_value : f (1/2) = Real.exp 1)
  (x : ℝ)
  (h_pos : 0 < x) :
  f (Real.log x) < x^2 ↔ x < Real.exp (1/2) :=
sorry

end NUMINAMATH_GPT_inequality_solution_l1242_124236


namespace NUMINAMATH_GPT_number_of_dogs_on_boat_l1242_124248

theorem number_of_dogs_on_boat 
  (initial_sheep : ℕ) (initial_cows : ℕ) (initial_dogs : ℕ)
  (drowned_sheep : ℕ) (drowned_cows : ℕ)
  (made_it_to_shore : ℕ)
  (H1 : initial_sheep = 20)
  (H2 : initial_cows = 10)
  (H3 : drowned_sheep = 3)
  (H4 : drowned_cows = 2 * drowned_sheep)
  (H5 : made_it_to_shore = 35)
  : initial_dogs = 14 := 
sorry

end NUMINAMATH_GPT_number_of_dogs_on_boat_l1242_124248


namespace NUMINAMATH_GPT_simplify_expression_l1242_124207

theorem simplify_expression (y : ℝ) :
  4 * y - 8 * y^2 + 6 - (3 - 6 * y - 9 * y^2 + 2 * y^3) = -2 * y^3 + y^2 + 10 * y + 3 := 
by
  -- Proof goes here, but we just state sorry for now
  sorry

end NUMINAMATH_GPT_simplify_expression_l1242_124207


namespace NUMINAMATH_GPT_factorize_m_square_minus_4m_l1242_124230

theorem factorize_m_square_minus_4m (m : ℝ) : m^2 - 4 * m = m * (m - 4) :=
by
  sorry

end NUMINAMATH_GPT_factorize_m_square_minus_4m_l1242_124230


namespace NUMINAMATH_GPT_sum_of_squares_l1242_124210

theorem sum_of_squares (x y z : ℝ)
  (h1 : (x + y + z) / 3 = 10)
  (h2 : (xyz)^(1/3) = 6)
  (h3 : 3 / ((1/x) + (1/y) + (1/z)) = 4) : 
  x^2 + y^2 + z^2 = 576 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l1242_124210


namespace NUMINAMATH_GPT_largest_n_for_triangle_property_l1242_124226

-- Define the triangle property for a set
def triangle_property (S : Set ℕ) : Prop :=
  ∀ {a b c : ℕ}, a ∈ S → b ∈ S → c ∈ S → a < b → b < c → a + b > c

-- Define the smallest subset that violates the triangle property
def violating_subset : Set ℕ := {5, 6, 11, 17, 28, 45, 73, 118, 191, 309}

-- Define the set of consecutive integers from 5 to n
def consecutive_integers (n : ℕ) : Set ℕ := {x : ℕ | 5 ≤ x ∧ x ≤ n}

-- The theorem we want to prove
theorem largest_n_for_triangle_property : ∀ (S : Set ℕ), S = consecutive_integers 308 → triangle_property S := sorry

end NUMINAMATH_GPT_largest_n_for_triangle_property_l1242_124226


namespace NUMINAMATH_GPT_measure_of_one_interior_angle_of_regular_octagon_l1242_124251

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_one_interior_angle_of_regular_octagon_l1242_124251


namespace NUMINAMATH_GPT_smaller_screen_diagonal_l1242_124247

/-- The area of a 20-inch square screen is 38 square inches greater than the area
    of a smaller square screen. Prove that the length of the diagonal of the smaller screen is 18 inches. -/
theorem smaller_screen_diagonal (x : ℝ) (d : ℝ) (A₁ A₂ : ℝ)
  (h₀ : d = x * Real.sqrt 2)
  (h₁ : A₁ = 20 * Real.sqrt 2 * 20 * Real.sqrt 2)
  (h₂ : A₂ = x * x)
  (h₃ : A₁ = A₂ + 38) :
  d = 18 :=
by
  sorry

end NUMINAMATH_GPT_smaller_screen_diagonal_l1242_124247


namespace NUMINAMATH_GPT_find_ordered_pair_l1242_124243

theorem find_ordered_pair (x y : ℝ) 
  (h1 : x + y = (7 - x) + (7 - y)) 
  (h2 : x - y = (x - 2) + (y - 2)) : 
  (x = 5 ∧ y = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_ordered_pair_l1242_124243


namespace NUMINAMATH_GPT_smallest_fraction_numerator_l1242_124273

theorem smallest_fraction_numerator :
  ∃ a b : ℕ, 10 ≤ a ∧ a < b ∧ b ≤ 99 ∧ 6 * a > 5 * b ∧ ∀ c d : ℕ,
    (10 ≤ c ∧ c < d ∧ d ≤ 99 ∧ 6 * c > 5 * d → a ≤ c) ∧ 
    a = 81 :=
sorry

end NUMINAMATH_GPT_smallest_fraction_numerator_l1242_124273


namespace NUMINAMATH_GPT_range_of_a_l1242_124278

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 3 → log (x - 1) + log (3 - x) = log (a - x)) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 1 < x₁ ∧ x₁ < 3 ∧ 1 < x₂ ∧ x₂ < 3) →
  3 < a ∧ a < 13 / 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1242_124278


namespace NUMINAMATH_GPT_tom_and_jerry_same_speed_l1242_124291

noncomputable def speed_of_tom (y : ℝ) : ℝ :=
  y^2 - 14*y + 45

noncomputable def speed_of_jerry (y : ℝ) : ℝ :=
  (y^2 - 2*y - 35) / (y - 5)

theorem tom_and_jerry_same_speed (y : ℝ) (h₁ : y ≠ 5) (h₂ : speed_of_tom y = speed_of_jerry y) :
  speed_of_tom y = 6 :=
by
  sorry

end NUMINAMATH_GPT_tom_and_jerry_same_speed_l1242_124291


namespace NUMINAMATH_GPT_customer_outreach_time_l1242_124262

variable (x : ℝ)

theorem customer_outreach_time
  (h1 : 8 = x + x / 2 + 2) :
  x = 4 :=
by sorry

end NUMINAMATH_GPT_customer_outreach_time_l1242_124262


namespace NUMINAMATH_GPT_part_one_part_two_l1242_124275

def f (a x : ℝ) : ℝ := abs (x - a ^ 2) + abs (x + 2 * a + 3)

theorem part_one (a x : ℝ) : f a x ≥ 2 :=
by 
  sorry

noncomputable def f_neg_three_over_two (a : ℝ) : ℝ := f a (-3/2)

theorem part_two (a : ℝ) (h : f_neg_three_over_two a < 3) : -1 < a ∧ a < 0 :=
by 
  sorry

end NUMINAMATH_GPT_part_one_part_two_l1242_124275


namespace NUMINAMATH_GPT_sum_of_first_three_terms_l1242_124276

theorem sum_of_first_three_terms 
  (a d : ℤ) 
  (h1 : a + 4 * d = 15) 
  (h2 : d = 3) : 
  a + (a + d) + (a + 2 * d) = 18 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_three_terms_l1242_124276


namespace NUMINAMATH_GPT_solve_quadratic_eq_1_solve_quadratic_eq_2_l1242_124204

-- Proof for Equation 1
theorem solve_quadratic_eq_1 : ∀ x : ℝ, x^2 - 4 * x + 1 = 0 ↔ (x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) :=
by sorry

-- Proof for Equation 2
theorem solve_quadratic_eq_2 : ∀ x : ℝ, 5 * x - 2 = (2 - 5 * x) * (3 * x + 4) ↔ (x = 2 / 5 ∨ x = -5 / 3) :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_eq_1_solve_quadratic_eq_2_l1242_124204


namespace NUMINAMATH_GPT_parabola_directrix_l1242_124256

theorem parabola_directrix (x y : ℝ) :
    x^2 = - (1 / 4) * y → y = - (1 / 16) :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_l1242_124256


namespace NUMINAMATH_GPT_goods_train_speed_l1242_124224

theorem goods_train_speed:
  let speed_mans_train := 100   -- in km/h
  let length_goods_train := 280 -- in meters
  let passing_time := 9         -- in seconds
  ∃ speed_goods_train: ℝ, 
  (speed_mans_train + speed_goods_train) * (5 / 18) * passing_time = length_goods_train ↔ speed_goods_train = 12 :=
by
  sorry

end NUMINAMATH_GPT_goods_train_speed_l1242_124224


namespace NUMINAMATH_GPT_symmetric_points_origin_l1242_124228

theorem symmetric_points_origin {a b : ℝ} (h₁ : a = -(-4)) (h₂ : b = -(3)) : a - b = 7 :=
by 
  -- since this is a statement template, the proof is omitted
  sorry

end NUMINAMATH_GPT_symmetric_points_origin_l1242_124228


namespace NUMINAMATH_GPT_solve_a₃_l1242_124225

noncomputable def geom_seq (a₁ a₅ a₃ : ℝ) : Prop :=
a₁ = 1 / 9 ∧ a₅ = 9 ∧ a₁ * a₅ = a₃^2

theorem solve_a₃ : ∃ a₃ : ℝ, geom_seq (1/9) 9 a₃ ∧ a₃ = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_a₃_l1242_124225


namespace NUMINAMATH_GPT_afternoon_to_morning_ratio_l1242_124258

theorem afternoon_to_morning_ratio
  (A : ℕ) (M : ℕ)
  (h1 : A = 340)
  (h2 : A + M = 510) :
  A / M = 2 :=
by
  sorry

end NUMINAMATH_GPT_afternoon_to_morning_ratio_l1242_124258


namespace NUMINAMATH_GPT_solve_system_l1242_124242

theorem solve_system (x y z : ℝ) (h1 : x + y + z = 6) (h2 : x * y + y * z + z * x = 11) (h3 : x * y * z = 6) :
  (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨ 
  (x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) :=
sorry

end NUMINAMATH_GPT_solve_system_l1242_124242


namespace NUMINAMATH_GPT_Vlad_height_feet_l1242_124282

theorem Vlad_height_feet 
  (sister_height_feet : ℕ)
  (sister_height_inches : ℕ)
  (vlad_height_diff : ℕ)
  (vlad_height_inches : ℕ)
  (vlad_height_feet : ℕ)
  (vlad_height_rem : ℕ)
  (sister_height := (sister_height_feet * 12) + sister_height_inches)
  (vlad_height := sister_height + vlad_height_diff)
  (vlad_height_feet_rem := (vlad_height / 12, vlad_height % 12)) 
  (h_sister_height : sister_height_feet = 2)
  (h_sister_height_inches : sister_height_inches = 10)
  (h_vlad_height_diff : vlad_height_diff = 41)
  (h_vlad_height : vlad_height = 75)
  (h_vlad_height_feet : vlad_height_feet = 6)
  (h_vlad_height_rem : vlad_height_rem = 3) :
  vlad_height_feet = 6 := by
  sorry

end NUMINAMATH_GPT_Vlad_height_feet_l1242_124282


namespace NUMINAMATH_GPT_Ara_height_in_inches_l1242_124287

theorem Ara_height_in_inches (Shea_current_height : ℝ) (Shea_growth_percentage : ℝ) (Ara_growth_factor : ℝ) (Shea_growth_amount : ℝ) (Ara_current_height : ℝ) :
  Shea_current_height = 75 →
  Shea_growth_percentage = 0.25 →
  Ara_growth_factor = 1 / 3 →
  Shea_growth_amount = 75 * (1 / (1 + 0.25)) * 0.25 →
  Ara_current_height = 75 * (1 / (1 + 0.25)) + (75 * (1 / (1 + 0.25)) * 0.25) * (1 / 3) →
  Ara_current_height = 65 :=
by sorry

end NUMINAMATH_GPT_Ara_height_in_inches_l1242_124287


namespace NUMINAMATH_GPT_water_tank_capacity_l1242_124289

variable (C : ℝ)  -- Full capacity of the tank in liters

theorem water_tank_capacity (h1 : 0.4 * C = 0.9 * C - 50) : C = 100 := by
  sorry

end NUMINAMATH_GPT_water_tank_capacity_l1242_124289


namespace NUMINAMATH_GPT_probability_sum_is_five_l1242_124293

theorem probability_sum_is_five (m n : ℕ) (h_m : 1 ≤ m ∧ m ≤ 6) (h_n : 1 ≤ n ∧ n ≤ 6)
  (h_total_outcomes : ∃(total_outcomes : ℕ), total_outcomes = 36)
  (h_favorable_outcomes : ∃(favorable_outcomes : ℕ), favorable_outcomes = 4) :
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 9 :=
sorry

end NUMINAMATH_GPT_probability_sum_is_five_l1242_124293


namespace NUMINAMATH_GPT_geometric_series_sum_l1242_124268

theorem geometric_series_sum : 
  ∀ (a r l : ℕ), 
    a = 2 ∧ r = 3 ∧ l = 4374 → 
    ∃ n S, 
      a * r ^ (n - 1) = l ∧ 
      S = a * (r^n - 1) / (r - 1) ∧ 
      S = 6560 :=
by 
  intros a r l h
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1242_124268


namespace NUMINAMATH_GPT_two_b_squared_eq_a_squared_plus_c_squared_l1242_124203

theorem two_b_squared_eq_a_squared_plus_c_squared (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 2 / (c + a)) : 
  2 * b^2 = a^2 + c^2 := 
sorry

end NUMINAMATH_GPT_two_b_squared_eq_a_squared_plus_c_squared_l1242_124203


namespace NUMINAMATH_GPT_find_omitted_angle_l1242_124260

-- Definitions and conditions
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

def omitted_angle (calculated_sum actual_sum : ℝ) : ℝ :=
  actual_sum - calculated_sum

-- The theorem to be proven
theorem find_omitted_angle (n : ℕ) (h₁ : 1958 + 22 = sum_of_interior_angles n) :
  omitted_angle 1958 (sum_of_interior_angles n) = 22 :=
by
  sorry

end NUMINAMATH_GPT_find_omitted_angle_l1242_124260


namespace NUMINAMATH_GPT_triangle_a_eq_5_over_3_triangle_b_plus_c_eq_4_l1242_124281

theorem triangle_a_eq_5_over_3
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a / (Real.cos A) = (3 * c - 2 * b) / (Real.cos B))
  (h2 : b = Real.sqrt 5 * Real.sin B) :
  a = 5 / 3 := sorry

theorem triangle_b_plus_c_eq_4
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a / (Real.cos A) = (3 * c - 2 * b) / (Real.cos B))
  (h2 : a = Real.sqrt 6)
  (h3 : 1 / 2 * b * c * Real.sin A = Real.sqrt 5 / 2) :
  b + c = 4 := sorry

end NUMINAMATH_GPT_triangle_a_eq_5_over_3_triangle_b_plus_c_eq_4_l1242_124281


namespace NUMINAMATH_GPT_verify_magic_square_l1242_124218

-- Define the grid as a 3x3 matrix
def magic_square := Matrix (Fin 3) (Fin 3) ℕ

-- Conditions for the magic square
def is_magic_square (m : magic_square) : Prop :=
  (∀ i : Fin 3, (m i 0) + (m i 1) + (m i 2) = 15) ∧
  (∀ j : Fin 3, (m 0 j) + (m 1 j) + (m 2 j) = 15) ∧
  ((m 0 0) + (m 1 1) + (m 2 2) = 15) ∧
  ((m 0 2) + (m 1 1) + (m 2 0) = 15)

-- Given specific filled numbers in the grid
def given_filled_values (m : magic_square) : Prop :=
  (m 0 1 = 5) ∧
  (m 1 0 = 2) ∧
  (m 2 2 = 8)

-- The complete grid based on the solution
def completed_magic_square : magic_square :=
  ![![4, 9, 2], ![3, 5, 7], ![8, 1, 6]]

-- The main theorem to prove
theorem verify_magic_square : 
  is_magic_square completed_magic_square ∧ 
  given_filled_values completed_magic_square := 
by 
  sorry

end NUMINAMATH_GPT_verify_magic_square_l1242_124218


namespace NUMINAMATH_GPT_du_chin_remaining_money_l1242_124274

noncomputable def du_chin_revenue_over_week : ℝ := 
  let day0_revenue := 200 * 20
  let day0_cost := 3 / 5 * day0_revenue
  let day0_remaining := day0_revenue - day0_cost

  let day1_revenue := day0_remaining * 1.10
  let day1_cost := day0_cost * 1.10
  let day1_remaining := day1_revenue - day1_cost

  let day2_revenue := day1_remaining * 0.95
  let day2_cost := day1_cost * 0.90
  let day2_remaining := day2_revenue - day2_cost

  let day3_revenue := day2_remaining
  let day3_cost := day2_cost
  let day3_remaining := day3_revenue - day3_cost

  let day4_revenue := day3_remaining * 1.15
  let day4_cost := day3_cost * 1.05
  let day4_remaining := day4_revenue - day4_cost

  let day5_revenue := day4_remaining * 0.92
  let day5_cost := day4_cost * 0.95
  let day5_remaining := day5_revenue - day5_cost

  let day6_revenue := day5_remaining * 1.05
  let day6_cost := day5_cost
  let day6_remaining := day6_revenue - day6_cost

  day0_remaining + day1_remaining + day2_remaining + day3_remaining + day4_remaining + day5_remaining + day6_remaining

theorem du_chin_remaining_money : du_chin_revenue_over_week = 13589.08 := 
  sorry

end NUMINAMATH_GPT_du_chin_remaining_money_l1242_124274


namespace NUMINAMATH_GPT_product_of_odd_primes_mod_32_l1242_124279

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_product_of_odd_primes_mod_32_l1242_124279


namespace NUMINAMATH_GPT_percentage_increase_selling_price_l1242_124288

-- Defining the conditions
def original_price : ℝ := 6
def increased_price : ℝ := 8.64
def total_sales_per_hour : ℝ := 216
def max_price : ℝ := 10

-- Statement for Part 1
theorem percentage_increase (x : ℝ) : 6 * (1 + x)^2 = 8.64 → x = 0.2 :=
by
  sorry

-- Statement for Part 2
theorem selling_price (a : ℝ) : (6 + a) * (30 - 2 * a) = 216 → 6 + a ≤ 10 → 6 + a = 9 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_selling_price_l1242_124288


namespace NUMINAMATH_GPT_sum_due_is_correct_l1242_124206

-- Definitions of the given conditions
def BD : ℝ := 78
def TD : ℝ := 66

-- Definition of the sum due (S)
noncomputable def S : ℝ := (TD^2) / (BD - TD) + TD

-- The theorem to be proved
theorem sum_due_is_correct : S = 429 := by
  sorry

end NUMINAMATH_GPT_sum_due_is_correct_l1242_124206


namespace NUMINAMATH_GPT_inequality_1_inequality_2_l1242_124284

-- Define the first inequality proof problem
theorem inequality_1 (x : ℝ) : 5 * x + 3 < 11 + x ↔ x < 2 := by
  sorry

-- Define the second set of inequalities proof problem
theorem inequality_2 (x : ℝ) : 
  (2 * x + 1 < 3 * x + 3) ∧ ((x + 1) / 2 ≤ (1 - x) / 6 + 1) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_GPT_inequality_1_inequality_2_l1242_124284


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_is_e_l1242_124259

-- Definitions and given conditions
variable (a b c : ℝ)
variable (h_a_pos : a > 0) (h_b_pos : b > 0)
variable (h_hyperbola : ∀ x y : ℝ, (x^2)/(a^2) - (y^2)/(b^2) = 1)
variable (h_left_focus : ∀ F : ℝ × ℝ, F = (-c, 0))
variable (h_circle : ∀ E : ℝ × ℝ, E.1^2 + E.2^2 = a^2)
variable (h_parabola : ∀ P : ℝ × ℝ, P.2^2 = 4*c*P.1)
variable (h_midpoint : ∀ E P F : ℝ × ℝ, E = (F.1 + P.1) / 2 ∧ E.2 = (F.2 + P.2) / 2)

-- The statement to be proved
theorem eccentricity_of_hyperbola_is_e :
    ∃ e : ℝ, e = (Real.sqrt 5 + 1) / 2 :=
sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_is_e_l1242_124259


namespace NUMINAMATH_GPT_distinct_convex_polygons_of_four_or_more_sides_l1242_124263

noncomputable def total_subsets (n : Nat) : Nat := 2^n

noncomputable def subsets_with_fewer_than_four_members (n : Nat) : Nat := 
  (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)

noncomputable def valid_subsets (n : Nat) : Nat := 
  total_subsets n - subsets_with_fewer_than_four_members n

theorem distinct_convex_polygons_of_four_or_more_sides (n : Nat) (h : n = 15) : valid_subsets n = 32192 := by
  sorry

end NUMINAMATH_GPT_distinct_convex_polygons_of_four_or_more_sides_l1242_124263


namespace NUMINAMATH_GPT_integer_pairs_sum_product_l1242_124272

theorem integer_pairs_sum_product (x y : ℤ) (h : x + y = x * y) : (x = 2 ∧ y = 2) ∨ (x = 0 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_integer_pairs_sum_product_l1242_124272


namespace NUMINAMATH_GPT_smallest_n_mod_equiv_l1242_124267

theorem smallest_n_mod_equiv (n : ℕ) (h : 0 < n ∧ 2^n ≡ n^5 [MOD 4]) : n = 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_mod_equiv_l1242_124267


namespace NUMINAMATH_GPT_arithmetic_sequence_ninth_term_eq_l1242_124237

theorem arithmetic_sequence_ninth_term_eq :
  let a_1 := (3 : ℚ) / 8
  let a_17 := (2 : ℚ) / 3
  let a_9 := (a_1 + a_17) / 2
  a_9 = (25 : ℚ) / 48 := by
  let a_1 := (3 : ℚ) / 8
  let a_17 := (2 : ℚ) / 3
  let a_9 := (a_1 + a_17) / 2
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ninth_term_eq_l1242_124237


namespace NUMINAMATH_GPT_f_of_72_l1242_124290

theorem f_of_72 (f : ℕ → ℝ) (p q : ℝ) (h1 : ∀ a b : ℕ, f (a * b) = f a + f b)
  (h2 : f 2 = p) (h3 : f 3 = q) : f 72 = 3 * p + 2 * q := 
sorry

end NUMINAMATH_GPT_f_of_72_l1242_124290


namespace NUMINAMATH_GPT_quadratic_has_two_roots_l1242_124240

variables {R : Type*} [LinearOrderedField R]

theorem quadratic_has_two_roots (a1 a2 a3 b1 b2 b3 : R) 
  (h1 : a1 * a2 * a3 = b1 * b2 * b3) (h2 : a1 * a2 * a3 > 1) : 
  (4 * a1^2 - 4 * b1 > 0) ∨ (4 * a2^2 - 4 * b2 > 0) ∨ (4 * a3^2 - 4 * b3 > 0) :=
sorry

end NUMINAMATH_GPT_quadratic_has_two_roots_l1242_124240


namespace NUMINAMATH_GPT_balloons_left_after_distribution_l1242_124295

-- Definitions for the conditions
def red_balloons : ℕ := 23
def blue_balloons : ℕ := 39
def green_balloons : ℕ := 71
def yellow_balloons : ℕ := 89
def total_balloons : ℕ := red_balloons + blue_balloons + green_balloons + yellow_balloons
def number_of_friends : ℕ := 10

-- Statement to prove the correct answer
theorem balloons_left_after_distribution : total_balloons % number_of_friends = 2 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_balloons_left_after_distribution_l1242_124295


namespace NUMINAMATH_GPT_max_checkers_on_board_l1242_124241

-- Define the size of the board.
def board_size : ℕ := 8

-- Define the max number of checkers per row/column.
def max_checkers_per_line : ℕ := 3

-- Define the conditions of the board.
structure BoardConfiguration :=
  (rows : Fin board_size → Fin (max_checkers_per_line + 1))
  (columns : Fin board_size → Fin (max_checkers_per_line + 1))
  (valid : ∀ (i : Fin board_size), rows i ≤ max_checkers_per_line ∧ columns i ≤ max_checkers_per_line)

-- Define the function to calculate the total number of checkers.
def total_checkers (config : BoardConfiguration) : ℕ :=
  Finset.univ.sum (λ i => config.rows i + config.columns i)

-- The theorem which states that the maximum number of checkers is 30.
theorem max_checkers_on_board : ∃ (config : BoardConfiguration), total_checkers config = 30 :=
  sorry

end NUMINAMATH_GPT_max_checkers_on_board_l1242_124241


namespace NUMINAMATH_GPT_inequality_problem_l1242_124283

variables {a b c d : ℝ}

theorem inequality_problem (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : d > 0) :
  (a^3 + b^3 + c^3) / (a + b + c) + (b^3 + c^3 + d^3) / (b + c + d) +
  (c^3 + d^3 + a^3) / (c + d + a) + (d^3 + a^3 + b^3) / (d + a + b) ≥ a^2 + b^2 + c^2 + d^2 := 
by
  sorry

end NUMINAMATH_GPT_inequality_problem_l1242_124283


namespace NUMINAMATH_GPT_mandy_quarters_l1242_124214

theorem mandy_quarters (q : ℕ) : 
  40 < q ∧ q < 400 ∧ 
  q % 6 = 2 ∧ 
  q % 7 = 2 ∧ 
  q % 8 = 2 →
  (q = 170 ∨ q = 338) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_mandy_quarters_l1242_124214


namespace NUMINAMATH_GPT_stickers_remaining_l1242_124212

theorem stickers_remaining (total_stickers : ℕ) (front_page_stickers : ℕ) (other_pages_stickers : ℕ) (num_other_pages : ℕ) (remaining_stickers : ℕ)
  (h0 : total_stickers = 89)
  (h1 : front_page_stickers = 3)
  (h2 : other_pages_stickers = 7)
  (h3 : num_other_pages = 6)
  (h4 : remaining_stickers = total_stickers - (front_page_stickers + other_pages_stickers * num_other_pages)) :
  remaining_stickers = 44 :=
by
  sorry

end NUMINAMATH_GPT_stickers_remaining_l1242_124212


namespace NUMINAMATH_GPT_archer_prob_6_or_less_l1242_124292

noncomputable def prob_event_D (P_A P_B P_C : ℝ) : ℝ :=
  1 - (P_A + P_B + P_C)

theorem archer_prob_6_or_less :
  let P_A := 0.5
  let P_B := 0.2
  let P_C := 0.1
  prob_event_D P_A P_B P_C = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_archer_prob_6_or_less_l1242_124292


namespace NUMINAMATH_GPT_inscribe_circle_in_convex_polygon_l1242_124298

theorem inscribe_circle_in_convex_polygon
  (S P r : ℝ) 
  (hP_pos : P > 0)
  (h_poly_area : S > 0)
  (h_nonneg : r ≥ 0) :
  S / P ≤ r :=
sorry

end NUMINAMATH_GPT_inscribe_circle_in_convex_polygon_l1242_124298


namespace NUMINAMATH_GPT_angle_covered_in_three_layers_l1242_124294

/-- Define the conditions: A 90-degree angle, sum of angles is 290 degrees,
    and prove the angle covered in three layers is 110 degrees. -/
theorem angle_covered_in_three_layers {α β : ℝ}
  (h1 : α + β = 90)
  (h2 : 2*α + 3*β = 290) :
  β = 110 := 
sorry

end NUMINAMATH_GPT_angle_covered_in_three_layers_l1242_124294


namespace NUMINAMATH_GPT_cylinder_volume_l1242_124222

theorem cylinder_volume (r l : ℝ) (h1 : r = 1) (h2 : l = 2 * r) : 
  ∃ V : ℝ, V = 2 * Real.pi := 
by 
  sorry

end NUMINAMATH_GPT_cylinder_volume_l1242_124222


namespace NUMINAMATH_GPT_circles_intersect_l1242_124200

theorem circles_intersect (t : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2 * t * x + t^2 - 4 = 0 ∧ x^2 + y^2 + 2 * x - 4 * t * y + 4 * t^2 - 8 = 0) ↔ 
  (-12 / 5 < t ∧ t < -2 / 5) ∨ (0 < t ∧ t < 2) :=
sorry

end NUMINAMATH_GPT_circles_intersect_l1242_124200


namespace NUMINAMATH_GPT_julia_paid_for_puppy_l1242_124254

theorem julia_paid_for_puppy :
  let dog_food := 20
  let treat := 2.5
  let treats := 2 * treat
  let toys := 15
  let crate := 20
  let bed := 20
  let collar_leash := 15
  let discount_rate := 0.20
  let total_before_discount := dog_food + treats + toys + crate + bed + collar_leash
  let discount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount
  let total_spent := 96
  total_spent - total_after_discount = 20 := 
by 
  sorry

end NUMINAMATH_GPT_julia_paid_for_puppy_l1242_124254


namespace NUMINAMATH_GPT_suitable_survey_method_l1242_124229

-- Definitions based on conditions
def large_population (n : ℕ) : Prop := n > 10000  -- Example threshold for large population
def impractical_comprehensive_survey : Prop := true  -- Given in condition

-- The statement of the problem
theorem suitable_survey_method (n : ℕ) (h1 : large_population n) (h2 : impractical_comprehensive_survey) : 
  ∃ method : String, method = "sampling survey" :=
sorry

end NUMINAMATH_GPT_suitable_survey_method_l1242_124229


namespace NUMINAMATH_GPT_smallest_ratio_l1242_124239

theorem smallest_ratio (r s : ℤ) (h1 : 3 * r ≥ 2 * s - 3) (h2 : 4 * s ≥ r + 12) : 
  (∃ r s, (r : ℚ) / s = 1 / 2) :=
by 
  sorry

end NUMINAMATH_GPT_smallest_ratio_l1242_124239


namespace NUMINAMATH_GPT_sum_of_abs_coeffs_l1242_124208

theorem sum_of_abs_coeffs (a : ℕ → ℤ) :
  (∀ x : ℤ, (1 - x)^5 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5) →
  |a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| = 32 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_abs_coeffs_l1242_124208


namespace NUMINAMATH_GPT_distance_A_beats_B_l1242_124223

theorem distance_A_beats_B
  (time_A time_B : ℝ)
  (dist : ℝ)
  (time_A_eq : time_A = 198)
  (time_B_eq : time_B = 220)
  (dist_eq : dist = 3) :
  (dist / time_A) * time_B - dist = 333 / 1000 :=
by
  sorry

end NUMINAMATH_GPT_distance_A_beats_B_l1242_124223


namespace NUMINAMATH_GPT_line_intersects_ellipse_if_and_only_if_l1242_124245

theorem line_intersects_ellipse_if_and_only_if (k : ℝ) (m : ℝ) :
  (∀ x, ∃ y, y = k * x + 1 ∧ (x^2 / 5 + y^2 / m = 1)) ↔ (m ≥ 1 ∧ m ≠ 5) := 
sorry

end NUMINAMATH_GPT_line_intersects_ellipse_if_and_only_if_l1242_124245


namespace NUMINAMATH_GPT_avg_speed_additional_hours_l1242_124216

/-- Definitions based on the problem conditions -/
def first_leg_speed : ℕ := 30 -- miles per hour
def first_leg_time : ℕ := 6 -- hours
def total_trip_time : ℕ := 8 -- hours
def total_avg_speed : ℕ := 34 -- miles per hour

/-- The theorem that ties everything together -/
theorem avg_speed_additional_hours : 
  ((total_avg_speed * total_trip_time) - (first_leg_speed * first_leg_time)) / (total_trip_time - first_leg_time) = 46 := 
sorry

end NUMINAMATH_GPT_avg_speed_additional_hours_l1242_124216


namespace NUMINAMATH_GPT_ellipse_major_axis_length_l1242_124238

-- Given conditions
variable (radius : ℝ) (h_radius : radius = 2)
variable (minor_axis : ℝ) (h_minor_axis : minor_axis = 2 * radius)
variable (major_axis : ℝ) (h_major_axis : major_axis = 1.4 * minor_axis)

-- Proof problem statement
theorem ellipse_major_axis_length : major_axis = 5.6 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_major_axis_length_l1242_124238


namespace NUMINAMATH_GPT_male_salmon_count_l1242_124235

theorem male_salmon_count (total_salmon : ℕ) (female_salmon : ℕ) (male_salmon : ℕ) 
  (h1 : total_salmon = 971639) 
  (h2 : female_salmon = 259378) 
  (h3 : male_salmon = total_salmon - female_salmon) : 
  male_salmon = 712261 :=
by
  sorry

end NUMINAMATH_GPT_male_salmon_count_l1242_124235


namespace NUMINAMATH_GPT_sum_of_two_numbers_l1242_124219

theorem sum_of_two_numbers (x y : ℕ) (h1 : y = 2 * x - 3) (h2 : x = 14) : x + y = 39 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l1242_124219


namespace NUMINAMATH_GPT_both_subjects_sum_l1242_124285

-- Define the total number of students
def N : ℕ := 1500

-- Define the bounds for students studying Biology (B) and Chemistry (C)
def B_min : ℕ := 900
def B_max : ℕ := 1050

def C_min : ℕ := 600
def C_max : ℕ := 750

-- Let x and y be the smallest and largest number of students studying both subjects
def x : ℕ := B_max + C_max - N
def y : ℕ := B_min + C_min - N

-- Prove that y + x = 300
theorem both_subjects_sum : y + x = 300 := by
  sorry

end NUMINAMATH_GPT_both_subjects_sum_l1242_124285
