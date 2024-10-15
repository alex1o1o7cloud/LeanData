import Mathlib

namespace NUMINAMATH_GPT_max_quadratic_value_l885_88500

def quadratic (x : ℝ) : ℝ :=
  -2 * x^2 + 4 * x + 3

theorem max_quadratic_value : ∃ x : ℝ, ∀ y : ℝ, quadratic x = y → y ≤ 5 ∧ (∀ z : ℝ, quadratic z ≤ y) := 
by
  sorry

end NUMINAMATH_GPT_max_quadratic_value_l885_88500


namespace NUMINAMATH_GPT_comparison_l885_88520

noncomputable def a : ℝ := 7 / 9
noncomputable def b : ℝ := 0.7 * Real.exp 0.1
noncomputable def c : ℝ := Real.cos (2 / 3)

theorem comparison : c > a ∧ a > b :=
by
  -- c > a proof
  have h1 : c > a := sorry
  -- a > b proof
  have h2 : a > b := sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_comparison_l885_88520


namespace NUMINAMATH_GPT_find_quotient_l885_88541

theorem find_quotient (dividend divisor remainder quotient : ℕ) 
  (h_dividend : dividend = 171) 
  (h_divisor : divisor = 21) 
  (h_remainder : remainder = 3) 
  (h_div_eq : dividend = divisor * quotient + remainder) :
  quotient = 8 :=
by sorry

end NUMINAMATH_GPT_find_quotient_l885_88541


namespace NUMINAMATH_GPT_range_of_a_l885_88567

theorem range_of_a (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 4) (h3 : a > b) (h4 : b > c) :
  a ∈ Set.Ioo (2 / 3) 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l885_88567


namespace NUMINAMATH_GPT_james_added_8_fish_l885_88578

theorem james_added_8_fish
  (initial_fish : ℕ := 60)
  (fish_eaten_per_day : ℕ := 2)
  (total_days_with_worm : ℕ := 21)
  (fish_remaining_when_discovered : ℕ := 26) :
  ∃ (additional_fish : ℕ), additional_fish = 8 :=
by
  let total_fish_eaten := total_days_with_worm * fish_eaten_per_day
  let fish_remaining_without_addition := initial_fish - total_fish_eaten
  let additional_fish := fish_remaining_when_discovered - fish_remaining_without_addition
  exact ⟨additional_fish, sorry⟩

end NUMINAMATH_GPT_james_added_8_fish_l885_88578


namespace NUMINAMATH_GPT_max_e_of_conditions_l885_88542

theorem max_e_of_conditions (a b c d e : ℝ) 
  (h1 : a + b + c + d + e = 8) 
  (h2 : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  e ≤ (16 / 5) :=
by 
  sorry

end NUMINAMATH_GPT_max_e_of_conditions_l885_88542


namespace NUMINAMATH_GPT_two_subsets_count_l885_88511

-- Definitions from the problem conditions
def S : Set (Fin 5) := {0, 1, 2, 3, 4}

-- Main statement
theorem two_subsets_count : 
  (∃ A B : Set (Fin 5), A ∪ B = S ∧ A ∩ B = {a, b} ∧ A ≠ B) → 
  (number_of_ways = 40) :=
sorry

end NUMINAMATH_GPT_two_subsets_count_l885_88511


namespace NUMINAMATH_GPT_seat_adjustment_schemes_l885_88561

theorem seat_adjustment_schemes {n k : ℕ} (h1 : n = 7) (h2 : k = 3) :
  (2 * Nat.choose n k) = 70 :=
by
  -- n is the number of people, k is the number chosen
  rw [h1, h2]
  -- the rest is skipped for the statement only
  sorry

end NUMINAMATH_GPT_seat_adjustment_schemes_l885_88561


namespace NUMINAMATH_GPT_geometric_sequence_a9_l885_88588

open Nat

theorem geometric_sequence_a9 (a : ℕ → ℝ) (h1 : a 3 = 20) (h2 : a 6 = 5) 
  (h_geometric : ∀ m n, a ((m + n) / 2) ^ 2 = a m * a n) : 
  a 9 = 5 / 4 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a9_l885_88588


namespace NUMINAMATH_GPT_arithmetic_sequence_third_term_l885_88591

theorem arithmetic_sequence_third_term (b y : ℝ) 
  (h1 : 2 * b + y + 2 = 10) 
  (h2 : b + y + 2 = b + y + 2) : 
  8 - b = 6 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_third_term_l885_88591


namespace NUMINAMATH_GPT_smallest_positive_debt_l885_88565

theorem smallest_positive_debt :
  ∃ (p g : ℤ), 25 = 250 * p + 175 * g :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_debt_l885_88565


namespace NUMINAMATH_GPT_equivalent_polar_coordinates_l885_88570

-- Definitions of given conditions and the problem statement
def polar_point_neg (r : ℝ) (θ : ℝ) : Prop := r = -3 ∧ θ = 5 * Real.pi / 6
def polar_point_pos (r : ℝ) (θ : ℝ) : Prop := r = 3 ∧ θ = 11 * Real.pi / 6
def angle_range (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < 2 * Real.pi

theorem equivalent_polar_coordinates :
  ∃ (r θ : ℝ), polar_point_neg r θ → polar_point_pos 3 (11 * Real.pi / 6) ∧ angle_range (11 * Real.pi / 6) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_polar_coordinates_l885_88570


namespace NUMINAMATH_GPT_positive_difference_g_b_values_l885_88534

noncomputable def g (n : ℤ) : ℤ :=
if n < 0 then n^2 + 5 * n + 6 else 3 * n - 30

theorem positive_difference_g_b_values : 
  let g_neg_3 := g (-3)
  let g_3 := g 3
  g_neg_3 = 0 → g_3 = -21 → 
  ∃ b1 b2 : ℤ, g_neg_3 + g_3 + g b1 = 0 ∧ g_neg_3 + g_3 + g b2 = 0 ∧ 
  b1 ≠ b2 ∧ b1 < b2 ∧ b1 < 0 ∧ b2 > 0 ∧ b2 - b1 = 22 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_g_b_values_l885_88534


namespace NUMINAMATH_GPT_machine_A_produces_1_sprockets_per_hour_l885_88543

namespace SprocketsProduction

variable {A T : ℝ} -- A: sprockets per hour of machine A, T: hours it takes for machine Q to produce 110 sprockets

-- Given conditions
axiom machine_Q_production_rate : 110 / T = 1.10 * A
axiom machine_P_production_rate : 110 / (T + 10) = A

-- The target theorem to prove
theorem machine_A_produces_1_sprockets_per_hour (h1 : 110 / T = 1.10 * A) (h2 : 110 / (T + 10) = A) : A = 1 :=
by sorry

end SprocketsProduction

end NUMINAMATH_GPT_machine_A_produces_1_sprockets_per_hour_l885_88543


namespace NUMINAMATH_GPT_product_11_29_product_leq_20_squared_product_leq_half_m_squared_l885_88569

-- Definition of natural numbers
variable (a b m : ℕ)

-- Statement 1: Prove that 11 × 29 = 20^2 - 9^2
theorem product_11_29 : 11 * 29 = 20^2 - 9^2 := sorry

-- Statement 2: Prove ∀ a, b ∈ ℕ, if a + b = 40, then ab ≤ 20^2.
theorem product_leq_20_squared (a b : ℕ) (h : a + b = 40) : a * b ≤ 20^2 := sorry

-- Statement 3: Prove ∀ a, b ∈ ℕ, if a + b = m, then ab ≤ (m/2)^2.
theorem product_leq_half_m_squared (a b : ℕ) (m : ℕ) (h : a + b = m) : a * b ≤ (m / 2)^2 := sorry

end NUMINAMATH_GPT_product_11_29_product_leq_20_squared_product_leq_half_m_squared_l885_88569


namespace NUMINAMATH_GPT_find_t_l885_88590

theorem find_t (t : ℝ) : (∃ y : ℝ, y = -(t - 1) ∧ 2 * y - 4 = 3 * (y - 2)) ↔ t = -1 :=
by sorry

end NUMINAMATH_GPT_find_t_l885_88590


namespace NUMINAMATH_GPT_scientific_notation_280000_l885_88545

theorem scientific_notation_280000 : (280000 : ℝ) = 2.8 * 10^5 :=
sorry

end NUMINAMATH_GPT_scientific_notation_280000_l885_88545


namespace NUMINAMATH_GPT_similar_triangle_shortest_side_l885_88531

theorem similar_triangle_shortest_side 
  (a₁ : ℝ) (b₁ : ℝ) (c₁ : ℝ) (c₂ : ℝ) (k : ℝ)
  (h₁ : a₁ = 15) 
  (h₂ : c₁ = 39) 
  (h₃ : c₂ = 117) 
  (h₄ : k = c₂ / c₁) 
  (h₅ : k = 3) 
  (h₆ : a₂ = a₁ * k) :
  a₂ = 45 := 
by {
  sorry -- proof is not required
}

end NUMINAMATH_GPT_similar_triangle_shortest_side_l885_88531


namespace NUMINAMATH_GPT_limit_of_sequence_l885_88551

noncomputable def limit_problem := 
  ∀ ε > 0, ∃ N : ℕ, ∀ n > N, |((2 * n - 3) / (n + 2) : ℝ) - 2| < ε

theorem limit_of_sequence : limit_problem :=
sorry

end NUMINAMATH_GPT_limit_of_sequence_l885_88551


namespace NUMINAMATH_GPT_find_a4_l885_88593

noncomputable def quadratic_eq (t : ℝ) := t^2 - 36 * t + 288 = 0

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∃ a1 : ℝ, a 1 = a1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def condition1 (a : ℕ → ℝ) := a 1 + a 2 = -1
def condition2 (a : ℕ → ℝ) := a 1 - a 3 = -3

theorem find_a4 :
  ∃ (a : ℕ → ℝ) (q : ℝ), quadratic_eq q ∧ geometric_sequence a q ∧ condition1 a ∧ condition2 a ∧ a 4 = -8 :=
by
  sorry

end NUMINAMATH_GPT_find_a4_l885_88593


namespace NUMINAMATH_GPT_carrie_pays_l885_88596

/-- Define the costs of different items --/
def shirt_cost : ℕ := 8
def pants_cost : ℕ := 18
def jacket_cost : ℕ := 60

/-- Define the quantities of different items bought by Carrie --/
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2

/-- Define the total cost calculation for Carrie --/
def total_cost : ℕ := (num_shirts * shirt_cost) + (num_pants * pants_cost) + (num_jackets * jacket_cost)

theorem carrie_pays : total_cost / 2 = 94 := 
by
  sorry

end NUMINAMATH_GPT_carrie_pays_l885_88596


namespace NUMINAMATH_GPT_number_of_planes_l885_88521

-- Definitions based on the conditions
def Line (space: Type) := space → space → Prop

variables {space: Type} [MetricSpace space]

-- Given conditions
variable (l1 l2 l3 : Line space)
variable (intersects : ∃ p : space, l1 p p ∧ l2 p p ∧ l3 p p)

-- The theorem stating the conclusion
theorem number_of_planes (h: ∃ p : space, l1 p p ∧ l2 p p ∧ l3 p p) :
  (1 = 1 ∨ 1 = 2 ∨ 1 = 3) ∨ (2 = 1 ∨ 2 = 2 ∨ 2 = 3) ∨ (3 = 1 ∨ 3 = 2 ∨ 3 = 3) := 
sorry

end NUMINAMATH_GPT_number_of_planes_l885_88521


namespace NUMINAMATH_GPT_smallest_n_geometric_seq_l885_88584

noncomputable def geom_seq (a r : ℝ) (n : ℕ) : ℝ :=
  a * r ^ (n - 1)

noncomputable def S_n (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a else a * (1 - r ^ n) / (1 - r)

theorem smallest_n_geometric_seq :
  (∃ n : ℕ, S_n (1/9) 3 n > 2018) ∧ ∀ m : ℕ, m < 10 → S_n (1/9) 3 m ≤ 2018 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_geometric_seq_l885_88584


namespace NUMINAMATH_GPT_amount_spent_l885_88582

-- Definitions
def initial_amount : ℕ := 54
def amount_left : ℕ := 29

-- Proof statement
theorem amount_spent : initial_amount - amount_left = 25 :=
by
  sorry

end NUMINAMATH_GPT_amount_spent_l885_88582


namespace NUMINAMATH_GPT_students_count_l885_88506

theorem students_count (initial: ℕ) (left: ℕ) (new: ℕ) (result: ℕ) 
  (h1: initial = 31)
  (h2: left = 5)
  (h3: new = 11)
  (h4: result = initial - left + new) : result = 37 := by
  sorry

end NUMINAMATH_GPT_students_count_l885_88506


namespace NUMINAMATH_GPT_base_number_equals_2_l885_88524

theorem base_number_equals_2 (x : ℝ) (n : ℕ) (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^26) (h2 : n = 25) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_base_number_equals_2_l885_88524


namespace NUMINAMATH_GPT_find_subtracted_number_l885_88507

theorem find_subtracted_number 
  (a : ℕ) (b : ℕ) (g : ℕ) (n : ℕ) 
  (h1 : a = 2) 
  (h2 : b = 3 * a) 
  (h3 : g = 2 * b - n) 
  (h4 : g = 8) : n = 4 :=
by 
  sorry

end NUMINAMATH_GPT_find_subtracted_number_l885_88507


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l885_88546

-- Given sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- Prove the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l885_88546


namespace NUMINAMATH_GPT_math_problem_l885_88517

variable (a b c m : ℝ)

-- Quadratic equation: y = ax^2 + bx + c
def quadratic (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Opens downward
axiom a_neg : a < 0
-- Passes through A(1, 0)
axiom passes_A : quadratic a b c 1 = 0
-- Passes through B(m, 0) with -2 < m < -1
axiom passes_B : quadratic a b c m = 0
axiom m_range : -2 < m ∧ m < -1

-- Prove the conclusions
theorem math_problem : b < 0 ∧ (a + b + c = 0) ∧ (a * (m+1) - b + c > 0) ∧ ¬(4 * a * c - b^2 > 4 * a) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l885_88517


namespace NUMINAMATH_GPT_total_fruits_l885_88562

def num_papaya_trees : ℕ := 2
def num_mango_trees : ℕ := 3
def papayas_per_tree : ℕ := 10
def mangos_per_tree : ℕ := 20

theorem total_fruits : (num_papaya_trees * papayas_per_tree) + (num_mango_trees * mangos_per_tree) = 80 := 
by
  sorry

end NUMINAMATH_GPT_total_fruits_l885_88562


namespace NUMINAMATH_GPT_kyler_games_won_l885_88575

theorem kyler_games_won (peter_wins peter_losses emma_wins emma_losses kyler_losses : ℕ)
  (h_peter : peter_wins = 5)
  (h_peter_losses : peter_losses = 4)
  (h_emma : emma_wins = 2)
  (h_emma_losses : emma_losses = 5)
  (h_kyler_losses : kyler_losses = 4) : ∃ kyler_wins : ℕ, kyler_wins = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_kyler_games_won_l885_88575


namespace NUMINAMATH_GPT_televisions_sold_this_black_friday_l885_88587

theorem televisions_sold_this_black_friday 
  (T : ℕ) 
  (h1 : ∀ (n : ℕ), n = 3 → (T + (50 * n) = 477)) 
  : T = 327 := 
sorry

end NUMINAMATH_GPT_televisions_sold_this_black_friday_l885_88587


namespace NUMINAMATH_GPT_intersect_curves_l885_88505

theorem intersect_curves (R : ℝ) (hR : R > 0) :
  (∃ (x y : ℝ), x^2 + y^2 = R^2 ∧ x - y - 2 = 0) ↔ R ≥ Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_intersect_curves_l885_88505


namespace NUMINAMATH_GPT_sum_of_coeffs_is_minus_one_l885_88581

theorem sum_of_coeffs_is_minus_one 
  (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℤ) :
  (∀ x : ℤ, (1 - x^3)^3 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8 + a₉ * x^9)
  → a = 1 
  → a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coeffs_is_minus_one_l885_88581


namespace NUMINAMATH_GPT_calculate_expression_l885_88548

theorem calculate_expression : ( (3 / 20 + 5 / 200 + 7 / 2000) * 2 = 0.357 ) :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l885_88548


namespace NUMINAMATH_GPT_find_triples_l885_88559

theorem find_triples (x n p : ℕ) (hp : Nat.Prime p) 
  (hx_pos : x > 0) (hn_pos : n > 0) : 
  x^3 + 3 * x + 14 = 2 * p^n → (x = 1 ∧ n = 2 ∧ p = 3) ∨ (x = 3 ∧ n = 2 ∧ p = 5) :=
by 
  sorry

end NUMINAMATH_GPT_find_triples_l885_88559


namespace NUMINAMATH_GPT_juvy_chives_l885_88525

-- Definitions based on the problem conditions
def total_rows : Nat := 20
def plants_per_row : Nat := 10
def parsley_rows : Nat := 3
def rosemary_rows : Nat := 2
def chive_rows : Nat := total_rows - (parsley_rows + rosemary_rows)

-- The statement we want to prove
theorem juvy_chives : chive_rows * plants_per_row = 150 := by
  sorry

end NUMINAMATH_GPT_juvy_chives_l885_88525


namespace NUMINAMATH_GPT_cone_lateral_area_and_sector_area_l885_88501

theorem cone_lateral_area_and_sector_area 
  (slant_height : ℝ) 
  (height : ℝ) 
  (r : ℝ) 
  (h_slant : slant_height = 1) 
  (h_height : height = 0.8) 
  (h_r : r = Real.sqrt (slant_height^2 - height^2)) :
  (1 / 2 * 2 * Real.pi * r * slant_height = 3 / 5 * Real.pi) ∧
  (1 / 2 * 2 * Real.pi * r * slant_height = 3 / 5 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_cone_lateral_area_and_sector_area_l885_88501


namespace NUMINAMATH_GPT_probability_same_color_is_correct_l885_88528

-- Definitions of the conditions
def num_red : ℕ := 6
def num_blue : ℕ := 5
def total_plates : ℕ := num_red + num_blue
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probability statement
def prob_three_same_color : ℚ :=
  let total_ways := choose total_plates 3
  let ways_red := choose num_red 3
  let ways_blue := choose num_blue 3
  let favorable_ways := ways_red
  favorable_ways / total_ways

theorem probability_same_color_is_correct : prob_three_same_color = (4 : ℚ) / 33 := sorry

end NUMINAMATH_GPT_probability_same_color_is_correct_l885_88528


namespace NUMINAMATH_GPT_find_f_5_l885_88585

-- Define the function f satisfying the given conditions
noncomputable def f : ℝ → ℝ :=
sorry

-- Assert the conditions as hypotheses
axiom f_eq : ∀ x y : ℝ, f (x - y) = f x + f y
axiom f_zero : f 0 = 2

-- State the theorem we need to prove
theorem find_f_5 : f 5 = 1 :=
sorry

end NUMINAMATH_GPT_find_f_5_l885_88585


namespace NUMINAMATH_GPT_opposite_of_2023_is_neg_2023_l885_88515

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end NUMINAMATH_GPT_opposite_of_2023_is_neg_2023_l885_88515


namespace NUMINAMATH_GPT_hire_applicant_A_l885_88574

-- Define the test scores for applicants A and B
def education_A := 7
def experience_A := 8
def attitude_A := 9

def education_B := 10
def experience_B := 7
def attitude_B := 8

-- Define the weights for the test items
def weight_education := 1 / 6
def weight_experience := 2 / 6
def weight_attitude := 3 / 6

-- Define the final scores
def final_score_A := education_A * weight_education + experience_A * weight_experience + attitude_A * weight_attitude
def final_score_B := education_B * weight_education + experience_B * weight_experience + attitude_B * weight_attitude

-- Prove that Applicant A is hired because their final score is higher
theorem hire_applicant_A : final_score_A > final_score_B :=
by sorry

end NUMINAMATH_GPT_hire_applicant_A_l885_88574


namespace NUMINAMATH_GPT_midpoint_of_symmetric_chord_on_ellipse_l885_88539

theorem midpoint_of_symmetric_chord_on_ellipse
  (A B : ℝ × ℝ) -- coordinates of points A and B
  (hA : (A.1^2 / 16) + (A.2^2 / 4) = 1) -- A lies on the ellipse
  (hB : (B.1^2 / 16) + (B.2^2 / 4) = 1) -- B lies on the ellipse
  (symm : 2 * (A.1 + B.1) / 2 - 2 * (A.2 + B.2) / 2 - 3 = 0) -- A and B are symmetric about the line
  : ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (2, 1 / 2) :=
  sorry

end NUMINAMATH_GPT_midpoint_of_symmetric_chord_on_ellipse_l885_88539


namespace NUMINAMATH_GPT_functional_equation_solution_l885_88573

-- Define the functional equation with given conditions
def func_eq (f : ℤ → ℝ) (N : ℕ) : Prop :=
  (∀ k : ℤ, f (2 * k) = 2 * f k) ∧
  (∀ k : ℤ, f (N - k) = f k)

-- State the mathematically equivalent proof problem
theorem functional_equation_solution (N : ℕ) (f : ℤ → ℝ) 
  (h1 : ∀ k : ℤ, f (2 * k) = 2 * f k)
  (h2 : ∀ k : ℤ, f (N - k) = f k) : 
  ∀ a : ℤ, f a = 0 := 
sorry

end NUMINAMATH_GPT_functional_equation_solution_l885_88573


namespace NUMINAMATH_GPT_initial_volume_is_72_l885_88535

noncomputable def initial_volume (V : ℝ) : Prop :=
  let salt_initial : ℝ := 0.10 * V
  let total_volume_new : ℝ := V + 18
  let salt_percentage_new : ℝ := 0.08 * total_volume_new
  salt_initial = salt_percentage_new

theorem initial_volume_is_72 :
  ∃ V : ℝ, initial_volume V ∧ V = 72 :=
by
  sorry

end NUMINAMATH_GPT_initial_volume_is_72_l885_88535


namespace NUMINAMATH_GPT_moles_of_NH4Cl_l885_88547

-- Define what is meant by "mole" and the substances NH3, HCl, and NH4Cl
def NH3 : Type := ℕ -- Use ℕ to represent moles
def HCl : Type := ℕ
def NH4Cl : Type := ℕ

-- Define the stoichiometry of the reaction
def reaction (n_NH3 n_HCl : ℕ) : ℕ :=
n_NH3 + n_HCl

-- Lean 4 statement: given 1 mole of NH3 and 1 mole of HCl, prove the reaction produces 1 mole of NH4Cl
theorem moles_of_NH4Cl (n_NH3 n_HCl : ℕ) (h1 : n_NH3 = 1) (h2 : n_HCl = 1) : 
  reaction n_NH3 n_HCl = 1 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_NH4Cl_l885_88547


namespace NUMINAMATH_GPT_second_fraction_correct_l885_88536

theorem second_fraction_correct : 
  ∃ x : ℚ, (2 / 3) * x * (1 / 3) * (3 / 8) = 0.07142857142857142 ∧ x = 6 / 7 :=
by
  sorry

end NUMINAMATH_GPT_second_fraction_correct_l885_88536


namespace NUMINAMATH_GPT_intersection_eq_l885_88527

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

theorem intersection_eq : A ∩ B = {1, 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l885_88527


namespace NUMINAMATH_GPT_tangential_difference_l885_88519

noncomputable def tan_alpha_minus_beta (α β : ℝ) : ℝ :=
  Real.tan (α - β)

theorem tangential_difference 
  {α β : ℝ}
  (h : 3 / (2 + Real.sin (2 * α)) + 2021 / (2 + Real.sin β) = 2024) : 
  tan_alpha_minus_beta α β = 1 := 
sorry

end NUMINAMATH_GPT_tangential_difference_l885_88519


namespace NUMINAMATH_GPT_minimize_y_l885_88592

noncomputable def y (x a b : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + 3 * x + 5

theorem minimize_y (a b : ℝ) : 
  ∃ x : ℝ, (∀ x' : ℝ, y x a b ≤ y x' a b) → x = (2 * a + 2 * b - 3) / 4 := by
  sorry

end NUMINAMATH_GPT_minimize_y_l885_88592


namespace NUMINAMATH_GPT_solve_for_x_and_y_l885_88549

theorem solve_for_x_and_y (x y : ℝ) 
  (h1 : 0.75 / x = 7 / 8)
  (h2 : x / y = 5 / 6) :
  x = 6 / 7 ∧ y = (6 / 7 * 6) / 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_and_y_l885_88549


namespace NUMINAMATH_GPT_max_value_fraction_l885_88564

theorem max_value_fraction : ∀ x : ℝ, 
  (∃ x : ℝ, max (1 + (16 / (4 * x^2 + 8 * x + 5))) = 17) :=
by
  sorry

end NUMINAMATH_GPT_max_value_fraction_l885_88564


namespace NUMINAMATH_GPT_mark_repayment_l885_88544

noncomputable def totalDebt (days : ℕ) : ℝ :=
  if days < 3 then
    20 + (20 * 0.10 * days)
  else
    35 + (20 * 0.10 * 3) + (35 * 0.10 * (days - 3))

theorem mark_repayment :
  ∃ (x : ℕ), totalDebt x ≥ 70 ∧ x = 12 :=
by
  -- Use this theorem statement to prove the corresponding lean proof
  sorry

end NUMINAMATH_GPT_mark_repayment_l885_88544


namespace NUMINAMATH_GPT_prop1_converse_prop1_inverse_prop1_contrapositive_prop2_converse_prop2_inverse_prop2_contrapositive_l885_88553

theorem prop1_converse (a b c : ℝ) (h : a * c^2 > b * c^2) : a > b := sorry

theorem prop1_inverse (a b c : ℝ) (h : a ≤ b) : a * c^2 ≤ b * c^2 := sorry

theorem prop1_contrapositive (a b c : ℝ) (h : a * c^2 ≤ b * c^2) : a ≤ b := sorry

theorem prop2_converse (a b c : ℝ) (f : ℝ → ℝ) (h : ∃x, f x = 0) : b^2 - 4 * a * c < 0 := sorry

theorem prop2_inverse (a b c : ℝ) (f : ℝ → ℝ) (h : b^2 - 4 * a * c ≥ 0) : ¬∃x, f x = 0 := sorry

theorem prop2_contrapositive (a b c : ℝ) (f : ℝ → ℝ) (h : ¬∃x, f x = 0) : b^2 - 4 * a * c ≥ 0 := sorry

end NUMINAMATH_GPT_prop1_converse_prop1_inverse_prop1_contrapositive_prop2_converse_prop2_inverse_prop2_contrapositive_l885_88553


namespace NUMINAMATH_GPT_felix_can_lift_150_l885_88514

-- Define the weights of Felix and his brother.
variables (F B : ℤ)

-- Given conditions
-- Felix's brother can lift three times his weight off the ground, and this amount is 600 pounds.
def brother_lift (B : ℤ) : Prop := 3 * B = 600
-- Felix's brother weighs twice as much as Felix.
def brother_weight (B F : ℤ) : Prop := B = 2 * F
-- Felix can lift off the ground 1.5 times his weight.
def felix_lift (F : ℤ) : ℤ := 3 * F / 2 -- Note: 1.5F can be represented as 3F/2 in Lean for integer operations.

-- Goal: Prove that Felix can lift 150 pounds.
theorem felix_can_lift_150 (F B : ℤ) (h1 : brother_lift B) (h2 : brother_weight B F) : felix_lift F = 150 := by
  dsimp [brother_lift, brother_weight, felix_lift] at *
  sorry

end NUMINAMATH_GPT_felix_can_lift_150_l885_88514


namespace NUMINAMATH_GPT_cos_seven_theta_l885_88533

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 2 / 5) : Real.cos (7 * θ) = -83728 / 390625 := 
sorry

end NUMINAMATH_GPT_cos_seven_theta_l885_88533


namespace NUMINAMATH_GPT_binary_to_decimal_101101_l885_88509

theorem binary_to_decimal_101101 : 
  (1 * 2^0 + 0 * 2^1 + 1 * 2^2 + 1 * 2^3 + 0 * 2^4 + 1 * 2^5) = 45 := 
by 
  sorry

end NUMINAMATH_GPT_binary_to_decimal_101101_l885_88509


namespace NUMINAMATH_GPT_max_ab_l885_88555

theorem max_ab (a b : ℝ) (h1 : a + 4 * b = 8) (h2 : a > 0) (h3 : b > 0) : ab ≤ 4 := 
sorry

end NUMINAMATH_GPT_max_ab_l885_88555


namespace NUMINAMATH_GPT_main_expr_equals_target_l885_88508

-- Define the improper fractions for the mixed numbers:
def mixed_to_improper (a b : ℕ) (c : ℕ) : ℚ := (a * b + c) / b

noncomputable def mixed_1 := mixed_to_improper 5 7 2
noncomputable def mixed_2 := mixed_to_improper 3 4 3
noncomputable def mixed_3 := mixed_to_improper 4 6 1
noncomputable def mixed_4 := mixed_to_improper 2 5 1

-- Define the main expression
noncomputable def main_expr := 47 * (mixed_1 - mixed_2) / (mixed_3 + mixed_4)

-- Define the target result converted to an improper fraction
noncomputable def target_result : ℚ := (11 * 99 + 13) / 99

-- The theorem to be proved: main_expr == target_result
theorem main_expr_equals_target : main_expr = target_result :=
by sorry

end NUMINAMATH_GPT_main_expr_equals_target_l885_88508


namespace NUMINAMATH_GPT_value_of_expression_l885_88523

theorem value_of_expression (n m : ℤ) (h : m = 2 * n^2 + n + 1) : 8 * n^2 - 4 * m + 4 * n - 3 = -7 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l885_88523


namespace NUMINAMATH_GPT_value_of_f_minus_a_l885_88538

noncomputable def f (x : ℝ) : ℝ := x^3 + x + 1

theorem value_of_f_minus_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by sorry

end NUMINAMATH_GPT_value_of_f_minus_a_l885_88538


namespace NUMINAMATH_GPT_canoe_problem_l885_88530

-- Definitions:
variables (P_L P_R : ℝ)

-- Conditions:
def conditions := 
  (P_L = P_R) ∧ -- Condition that the probabilities for left and right oars working are the same
  (0 ≤ P_L) ∧ (P_L ≤ 1) ∧ -- Probability values must be between 0 and 1
  (1 - (1 - P_L) * (1 - P_R) = 0.84) -- Given the rowing probability is 0.84

-- Theorem that P_L = 0.6 given the conditions:
theorem canoe_problem : conditions P_L P_R → P_L = 0.6 :=
by
  sorry

end NUMINAMATH_GPT_canoe_problem_l885_88530


namespace NUMINAMATH_GPT_chord_intersection_eq_l885_88518

theorem chord_intersection_eq (x y : ℝ) (r : ℝ) : 
  (x + 1)^2 + y^2 = r^2 → 
  (x - 4)^2 + (y - 1)^2 = 4 → 
  (x = 4) → 
  (y = 1) → 
  (r^2 = 26) → (5 * x + y - 19 = 0) :=
by
  sorry

end NUMINAMATH_GPT_chord_intersection_eq_l885_88518


namespace NUMINAMATH_GPT_area_enclosed_by_graph_l885_88597

theorem area_enclosed_by_graph (x y : ℝ) (h : abs (5 * x) + abs (3 * y) = 15) : 
  ∃ (area : ℝ), area = 30 :=
sorry

end NUMINAMATH_GPT_area_enclosed_by_graph_l885_88597


namespace NUMINAMATH_GPT_monthly_interest_payment_l885_88599

theorem monthly_interest_payment (P : ℝ) (R : ℝ) (monthly_payment : ℝ)
  (hP : P = 28800) (hR : R = 0.09) : 
  monthly_payment = (P * R) / 12 :=
by
  sorry

end NUMINAMATH_GPT_monthly_interest_payment_l885_88599


namespace NUMINAMATH_GPT_george_total_coins_l885_88526

-- We'll state the problem as proving the total number of coins George has.
variable (num_nickels num_dimes : ℕ)
variable (value_of_coins : ℝ := 2.60)
variable (value_of_nickels : ℝ := 0.05 * num_nickels)
variable (value_of_dimes : ℝ := 0.10 * num_dimes)

theorem george_total_coins :
  num_nickels = 4 → 
  value_of_coins = value_of_nickels + value_of_dimes → 
  num_nickels + num_dimes = 28 := 
by
  sorry

end NUMINAMATH_GPT_george_total_coins_l885_88526


namespace NUMINAMATH_GPT_remaining_dimes_l885_88579

-- Conditions
def initial_pennies : Nat := 7
def initial_dimes : Nat := 8
def borrowed_dimes : Nat := 4

-- Define the theorem
theorem remaining_dimes : initial_dimes - borrowed_dimes = 4 := by
  -- Use the conditions to state the remaining dimes
  sorry

end NUMINAMATH_GPT_remaining_dimes_l885_88579


namespace NUMINAMATH_GPT_right_triangle_area_l885_88580

theorem right_triangle_area (a b c : ℝ) (h1 : c = 13) (h2 : a = 5) (h3 : a^2 + b^2 = c^2) : 0.5 * a * b = 30 := by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l885_88580


namespace NUMINAMATH_GPT_sack_flour_cost_l885_88558

theorem sack_flour_cost
  (x y : ℝ) 
  (h1 : 10 * x + 800 = 108 * y)
  (h2 : 4 * x - 800 = 36 * y) : x = 1600 := by
  -- Add your proof here
  sorry

end NUMINAMATH_GPT_sack_flour_cost_l885_88558


namespace NUMINAMATH_GPT_slope_of_intersection_line_is_one_l885_88502

open Real

-- Definitions of the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 4 * y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8 * x + 2 * y + 4 = 0

-- The statement to prove that the slope of the line through the intersection points is 1
theorem slope_of_intersection_line_is_one :
  ∃ m : ℝ, (∀ x y : ℝ, circle1 x y → circle2 x y → (y = m * x + b)) ∧ m = 1 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_intersection_line_is_one_l885_88502


namespace NUMINAMATH_GPT_train_speed_l885_88568

theorem train_speed (distance time : ℤ) (h_distance : distance = 500)
    (h_time : time = 3) :
    distance / time = 166 :=
by
  -- Proof steps will be filled in here
  sorry

end NUMINAMATH_GPT_train_speed_l885_88568


namespace NUMINAMATH_GPT_angle_D_in_triangle_DEF_l885_88594

theorem angle_D_in_triangle_DEF 
  (E F D : ℝ) 
  (hEF : F = 3 * E) 
  (hE : E = 15) 
  (h_sum_angles : D + E + F = 180) : D = 120 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_angle_D_in_triangle_DEF_l885_88594


namespace NUMINAMATH_GPT_smallest_sum_of_consecutive_primes_divisible_by_5_l885_88598

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (p1 p2 p3 : ℕ) : Prop :=
  is_prime p1 ∧ is_prime p2 ∧ p2 = p1 + 1 ∧ is_prime p3 ∧ p3 = p2 + 1

def sum_divisible_by_5 (p1 p2 p3 : ℕ) : Prop :=
  (p1 + p2 + p3) % 5 = 0

theorem smallest_sum_of_consecutive_primes_divisible_by_5 :
  ∃ (p1 p2 p3 : ℕ), consecutive_primes p1 p2 p3 ∧ sum_divisible_by_5 p1 p2 p3 ∧ p1 + p2 + p3 = 10 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_of_consecutive_primes_divisible_by_5_l885_88598


namespace NUMINAMATH_GPT_sum_of_powers_l885_88529

-- Here is the statement in Lean 4
theorem sum_of_powers (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  (ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68) = (ω^2 - 1) / (ω^4 - 1) :=
sorry -- Proof is omitted as per instructions.

end NUMINAMATH_GPT_sum_of_powers_l885_88529


namespace NUMINAMATH_GPT_find_f_50_l885_88503

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x * y
axiom f_20 : f 20 = 10

theorem find_f_50 : f 50 = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_f_50_l885_88503


namespace NUMINAMATH_GPT_C_share_l885_88572

-- Definitions based on conditions
def total_sum : ℝ := 164
def ratio_B : ℝ := 0.65
def ratio_C : ℝ := 0.40

-- Statement of the proof problem
theorem C_share : (ratio_C * (total_sum / (1 + ratio_B + ratio_C))) = 32 :=
by
  sorry

end NUMINAMATH_GPT_C_share_l885_88572


namespace NUMINAMATH_GPT_min_r_minus_p_l885_88557

theorem min_r_minus_p : ∃ (p q r : ℕ), p * q * r = 362880 ∧ p < q ∧ q < r ∧ (∀ p' q' r' : ℕ, (p' * q' * r' = 362880 ∧ p' < q' ∧ q' < r') → r - p ≤ r' - p') ∧ r - p = 39 :=
by
  sorry

end NUMINAMATH_GPT_min_r_minus_p_l885_88557


namespace NUMINAMATH_GPT_function_identity_l885_88566

theorem function_identity (f : ℕ+ → ℕ+) (h : ∀ m n : ℕ+, f m + f n ∣ m + n) : ∀ m : ℕ+, f m = m := by
  sorry

end NUMINAMATH_GPT_function_identity_l885_88566


namespace NUMINAMATH_GPT_factorize_expression_l885_88589

theorem factorize_expression (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l885_88589


namespace NUMINAMATH_GPT_ellipse_hyperbola_tangent_n_values_are_62_20625_and_1_66875_l885_88576

theorem ellipse_hyperbola_tangent_n_values_are_62_20625_and_1_66875:
  let is_ellipse (x y n : ℝ) := x^2 + n*(y - 1)^2 = n
  let is_hyperbola (x y : ℝ) := x^2 - 4*(y + 3)^2 = 4
  ∃ (n1 n2 : ℝ),
    n1 = 62.20625 ∧ n2 = 1.66875 ∧
    (∀ (x y : ℝ), is_ellipse x y n1 → is_hyperbola x y → 
       is_ellipse x y n2 → is_hyperbola x y → 
       (4 + n1)*(y^2 - 2*y + 1) = 4*(y^2 + 6*y + 9) + 4 ∧ 
       ((24 - 2*n1)^2 - 4*(4 + n1)*40 = 0) ∧
       (4 + n2)*(y^2 - 2*y + 1) = 4*(y^2 + 6*y + 9) + 4 ∧ 
       ((24 - 2*n2)^2 - 4*(4 + n2)*40 = 0))
:= sorry

end NUMINAMATH_GPT_ellipse_hyperbola_tangent_n_values_are_62_20625_and_1_66875_l885_88576


namespace NUMINAMATH_GPT_final_speed_train_l885_88556

theorem final_speed_train
  (u : ℝ) (a : ℝ) (t : ℕ) :
  u = 0 → a = 1 → t = 20 → u + a * t = 20 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_final_speed_train_l885_88556


namespace NUMINAMATH_GPT_determinant_of_sum_l885_88522

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![5, 6], ![2, 3]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 1], ![1, 0]]

theorem determinant_of_sum : (A + B).det = -3 := 
by 
  sorry

end NUMINAMATH_GPT_determinant_of_sum_l885_88522


namespace NUMINAMATH_GPT_sin_A_equals_4_over_5_l885_88571

variables {A B C : ℝ}
-- Given a right triangle ABC with angle B = 90 degrees
def right_triangle (A B C : ℝ) : Prop :=
  (A + B + C = 180) ∧ (B = 90)

-- We are given 3 * sin(A) = 4 * cos(A)
def given_condition (A : ℝ) : Prop :=
  3 * Real.sin A = 4 * Real.cos A

-- We need to prove that sin(A) = 4/5
theorem sin_A_equals_4_over_5 (A B C : ℝ) 
  (h1 : right_triangle B 90 C)
  (h2 : given_condition A) : 
  Real.sin A = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_A_equals_4_over_5_l885_88571


namespace NUMINAMATH_GPT_number_property_l885_88577

theorem number_property (n : ℕ) (h : n = 7101449275362318840579) :
  n / 7 = 101449275362318840579 :=
sorry

end NUMINAMATH_GPT_number_property_l885_88577


namespace NUMINAMATH_GPT_find_crossed_out_digit_l885_88550

theorem find_crossed_out_digit (n : ℕ) (h_rev : ∀ (k : ℕ), k < n → k % 9 = 0) (remaining_sum : ℕ) 
  (crossed_sum : ℕ) (h_sum : remaining_sum + crossed_sum = 27) : 
  crossed_sum = 8 :=
by
  -- We can incorporate generating the value from digit sum here.
  sorry

end NUMINAMATH_GPT_find_crossed_out_digit_l885_88550


namespace NUMINAMATH_GPT_minimum_arc_length_of_curve_and_line_l885_88504

-- Definition of the curve C and the line x = π/4
def curve (x y α : ℝ) : Prop :=
  (x - Real.arcsin α) * (x - Real.arccos α) + (y - Real.arcsin α) * (y + Real.arccos α) = 0

def line (x : ℝ) : Prop :=
  x = Real.pi / 4

-- Statement of the proof problem: the minimum value of d as α varies
theorem minimum_arc_length_of_curve_and_line : 
  (∀ α : ℝ, ∃ d : ℝ, (∃ y : ℝ, curve (Real.pi / 4) y α) → 
    (d = Real.pi / 2)) :=
sorry

end NUMINAMATH_GPT_minimum_arc_length_of_curve_and_line_l885_88504


namespace NUMINAMATH_GPT_minimize_shelves_books_l885_88510

theorem minimize_shelves_books : 
  ∀ (n : ℕ),
    (n > 0 ∧ 130 % n = 0 ∧ 195 % n = 0) → 
    (n ≤ 65) := sorry

end NUMINAMATH_GPT_minimize_shelves_books_l885_88510


namespace NUMINAMATH_GPT_distance_from_unselected_vertex_l885_88583

-- Define the problem statement
theorem distance_from_unselected_vertex
  (base length : ℝ) (area : ℝ) (h : ℝ) 
  (h_area : area = (base * h) / 2) 
  (h_base : base = 8) 
  (h_area_given : area = 24) : 
  h = 6 :=
by
  -- The proof here is skipped
  sorry

end NUMINAMATH_GPT_distance_from_unselected_vertex_l885_88583


namespace NUMINAMATH_GPT_no_roots_impl_a_neg_l885_88512

theorem no_roots_impl_a_neg {a : ℝ} : (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x - 1/x + a ≠ 0) → a < 0 :=
sorry

end NUMINAMATH_GPT_no_roots_impl_a_neg_l885_88512


namespace NUMINAMATH_GPT_average_percentage_l885_88595

theorem average_percentage (x : ℝ) : (60 + x + 80) / 3 = 70 → x = 70 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_average_percentage_l885_88595


namespace NUMINAMATH_GPT_final_amount_H2O_l885_88552

theorem final_amount_H2O (main_reaction : ∀ (Li3N H2O LiOH NH3 : ℕ), Li3N + 3 * H2O = 3 * LiOH + NH3)
  (side_reaction : ∀ (Li3N LiOH Li2O NH4OH : ℕ), Li3N + LiOH = Li2O + NH4OH)
  (temperature : ℕ) (pressure : ℕ)
  (percentage : ℝ) (init_moles_LiOH : ℕ) (init_moles_Li3N : ℕ)
  (H2O_req_main : ℝ) (H2O_req_side : ℝ) :
  400 = temperature →
  2 = pressure →
  0.05 = percentage →
  9 = init_moles_LiOH →
  3 = init_moles_Li3N →
  H2O_req_main = init_moles_Li3N * 3 →
  H2O_req_side = init_moles_LiOH * percentage →
  H2O_req_main + H2O_req_side = 9.45 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end NUMINAMATH_GPT_final_amount_H2O_l885_88552


namespace NUMINAMATH_GPT_darry_total_steps_l885_88540

def largest_ladder_steps : ℕ := 20
def largest_ladder_times : ℕ := 12

def medium_ladder_steps : ℕ := 15
def medium_ladder_times : ℕ := 8

def smaller_ladder_steps : ℕ := 10
def smaller_ladder_times : ℕ := 10

def smallest_ladder_steps : ℕ := 5
def smallest_ladder_times : ℕ := 15

theorem darry_total_steps :
  (largest_ladder_steps * largest_ladder_times)
  + (medium_ladder_steps * medium_ladder_times)
  + (smaller_ladder_steps * smaller_ladder_times)
  + (smallest_ladder_steps * smallest_ladder_times)
  = 535 := by
  sorry

end NUMINAMATH_GPT_darry_total_steps_l885_88540


namespace NUMINAMATH_GPT_memorial_visits_l885_88563

theorem memorial_visits (x : ℕ) (total_visits : ℕ) (difference : ℕ) 
  (h1 : total_visits = 589) 
  (h2 : difference = 56) 
  (h3 : 2 * x + difference = total_visits - x) : 
  2 * x + 56 = 589 - x :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_memorial_visits_l885_88563


namespace NUMINAMATH_GPT_sum_first_3n_terms_l885_88554

-- Geometric Sequence: Sum of first n terms Sn, first 2n terms S2n, first 3n terms S3n.
variables {n : ℕ} {S : ℕ → ℕ}

-- Conditions
def sum_first_n_terms (S : ℕ → ℕ) (n : ℕ) : Prop := S n = 48
def sum_first_2n_terms (S : ℕ → ℕ) (n : ℕ) : Prop := S (2 * n) = 60

-- Theorem to Prove
theorem sum_first_3n_terms {S : ℕ → ℕ} (h1 : sum_first_n_terms S n) (h2 : sum_first_2n_terms S n) :
  S (3 * n) = 63 :=
sorry

end NUMINAMATH_GPT_sum_first_3n_terms_l885_88554


namespace NUMINAMATH_GPT_radius_of_curvature_correct_l885_88532

open Real

noncomputable def radius_of_curvature_squared (a b t_0 : ℝ) : ℝ :=
  (a^2 * sin t_0^2 + b^2 * cos t_0^2)^3 / (a^2 * b^2)

theorem radius_of_curvature_correct (a b t_0 : ℝ) (h : a > 0) (h₁ : b > 0) :
  radius_of_curvature_squared a b t_0 = (a^2 * sin t_0^2 + b^2 * cos t_0^2)^3 / (a^2 * b^2) :=
sorry

end NUMINAMATH_GPT_radius_of_curvature_correct_l885_88532


namespace NUMINAMATH_GPT_money_last_duration_l885_88513

-- Defining the conditions
def money_from_mowing : ℕ := 14
def money_from_weed_eating : ℕ := 26
def money_spent_per_week : ℕ := 5

-- Theorem statement to prove Mike's money will last 8 weeks
theorem money_last_duration : (money_from_mowing + money_from_weed_eating) / money_spent_per_week = 8 := by
  sorry

end NUMINAMATH_GPT_money_last_duration_l885_88513


namespace NUMINAMATH_GPT_most_likely_units_digit_l885_88537

theorem most_likely_units_digit :
  ∃ m n : Fin 11, ∀ (M N : Fin 11), (∃ k : Nat, k * 11 + M + N = m + n) → 
    (m + n) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_most_likely_units_digit_l885_88537


namespace NUMINAMATH_GPT_q0_r0_eq_three_l885_88586

variable (p q r s : Polynomial ℝ)
variable (hp_const : p.coeff 0 = 2)
variable (hs_eq : s = p * q * r)
variable (hs_const : s.coeff 0 = 6)

theorem q0_r0_eq_three : (q.coeff 0) * (r.coeff 0) = 3 := by
  sorry

end NUMINAMATH_GPT_q0_r0_eq_three_l885_88586


namespace NUMINAMATH_GPT_calories_burned_per_week_l885_88560

-- Definitions of the conditions
def classes_per_week : ℕ := 3
def hours_per_class : ℝ := 1.5
def calories_per_min : ℝ := 7
def minutes_per_hour : ℝ := 60

-- Theorem stating the proof problem
theorem calories_burned_per_week : 
  (classes_per_week * (hours_per_class * minutes_per_hour) * calories_per_min) = 1890 := by
  sorry

end NUMINAMATH_GPT_calories_burned_per_week_l885_88560


namespace NUMINAMATH_GPT_more_stable_performance_l885_88516

theorem more_stable_performance (s_A_sq s_B_sq : ℝ) (hA : s_A_sq = 0.25) (hB : s_B_sq = 0.12) : s_A_sq > s_B_sq :=
by
  rw [hA, hB]
  sorry

end NUMINAMATH_GPT_more_stable_performance_l885_88516
