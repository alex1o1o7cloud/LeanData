import Mathlib

namespace NUMINAMATH_GPT_impossible_sequence_l1010_101056

theorem impossible_sequence (a : ℕ → ℝ) (c : ℝ) (a1 : ℝ)
  (h_periodic : ∀ n, a (n + 3) = a n)
  (h_det : ∀ n, a n * a (n + 3) - a (n + 1) * a (n + 2) = c)
  (ha1 : a 1 = 2) (hc : c = 2) : false :=
by
  sorry

end NUMINAMATH_GPT_impossible_sequence_l1010_101056


namespace NUMINAMATH_GPT_min_value_inequality_l1010_101011

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ( (x^2 + 4*x + 2) * (y^2 + 5*y + 3) * (z^2 + 6*z + 4) ) / (x * y * z) ≥ 336 := 
by
  sorry

end NUMINAMATH_GPT_min_value_inequality_l1010_101011


namespace NUMINAMATH_GPT_area_three_layers_l1010_101091

def total_area_rugs : ℝ := 200
def floor_covered_area : ℝ := 140
def exactly_two_layers_area : ℝ := 24

theorem area_three_layers : (2 * (200 - 140 - 24) / 2 = 2 * 18) := 
by admit -- since we're instructed to skip the proof

end NUMINAMATH_GPT_area_three_layers_l1010_101091


namespace NUMINAMATH_GPT_last_term_arithmetic_progression_eq_62_l1010_101002

theorem last_term_arithmetic_progression_eq_62
  (a : ℕ) (d : ℕ) (n : ℕ) 
  (h_a : a = 2)
  (h_d : d = 2)
  (h_n : n = 31) : 
  a + (n - 1) * d = 62 :=
by
  sorry

end NUMINAMATH_GPT_last_term_arithmetic_progression_eq_62_l1010_101002


namespace NUMINAMATH_GPT_units_digit_of_33_pow_33_mul_7_pow_7_l1010_101092

theorem units_digit_of_33_pow_33_mul_7_pow_7 : (33 ^ (33 * (7 ^ 7))) % 10 = 7 := 
  sorry

end NUMINAMATH_GPT_units_digit_of_33_pow_33_mul_7_pow_7_l1010_101092


namespace NUMINAMATH_GPT_max_negatives_l1010_101077

theorem max_negatives (a b c d e f : ℤ) (h : ab + cdef < 0) : ∃ w : ℤ, w = 4 := 
sorry

end NUMINAMATH_GPT_max_negatives_l1010_101077


namespace NUMINAMATH_GPT_min_value_of_seq_l1010_101071

theorem min_value_of_seq 
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (m a₁ : ℝ)
  (h1 : ∀ n, a n + a (n + 1) = n * (-1) ^ ((n * (n + 1)) / 2))
  (h2 : m + S 2015 = -1007)
  (h3 : a₁ * m > 0) :
  ∃ x, x = (1 / a₁) + (4 / m) ∧ x = 9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_seq_l1010_101071


namespace NUMINAMATH_GPT_inequality_tangents_l1010_101037

def f (x : ℝ) (a b : ℝ) : ℝ := x^3 - a * x - b

theorem inequality_tangents (a b : ℝ) (h1 : 0 < a)
  (h2 : ∃ x0 : ℝ, 2 * x0^3 - 3 * a * x0^2 + a^2 + 2 * b = 0): 
  -a^2 / 2 < b ∧ b < f a a b :=
by
  sorry

end NUMINAMATH_GPT_inequality_tangents_l1010_101037


namespace NUMINAMATH_GPT_jane_trail_mix_chocolate_chips_l1010_101072

theorem jane_trail_mix_chocolate_chips (c₁ : ℝ) (c₂ : ℝ) (c₃ : ℝ) (c₄ : ℝ) (c₅ : ℝ) :
  (c₁ = 0.30) → (c₂ = 0.70) → (c₃ = 0.45) → (c₄ = 0.35) → (c₅ = 0.60) →
  c₄ = 0.35 ∧ (c₅ - c₁) * 2 = 0.40 := 
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end NUMINAMATH_GPT_jane_trail_mix_chocolate_chips_l1010_101072


namespace NUMINAMATH_GPT_intersection_is_correct_l1010_101070

noncomputable def A : Set ℝ := {x | -2 < x ∧ x < 2}

noncomputable def B : Set ℝ := {x | x^2 - 5 * x - 6 < 0}

theorem intersection_is_correct : A ∩ B = {x | -1 < x ∧ x < 2} := 
by { sorry }

end NUMINAMATH_GPT_intersection_is_correct_l1010_101070


namespace NUMINAMATH_GPT_find_x_such_that_fraction_eq_l1010_101086

theorem find_x_such_that_fraction_eq 
  (x : ℚ) (h₁ : x ≠ 1) (h₂ : x ≠ 5) : 
  (x^2 - 4 * x + 3) / (x^2 - 6 * x + 5) = (x^2 - 3 * x - 10) / (x^2 - 2 * x - 15) ↔ 
  x = -19 / 3 :=
sorry

end NUMINAMATH_GPT_find_x_such_that_fraction_eq_l1010_101086


namespace NUMINAMATH_GPT_transformation_matrix_correct_l1010_101089
noncomputable def M : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, 3],
  ![-3, 0]
]

theorem transformation_matrix_correct :
  let R : Matrix (Fin 2) (Fin 2) ℝ := ![
    ![0, 1],
    ![-1, 0]
  ];
  let S : ℝ := 3;
  M = S • R :=
by
  sorry

end NUMINAMATH_GPT_transformation_matrix_correct_l1010_101089


namespace NUMINAMATH_GPT_max_ab_is_nine_l1010_101044

noncomputable def f (a b x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

/-- If a > 0, b > 0, and the function f(x) = 4x^3 - ax^2 - 2bx + 2 has an extremum at x = 1, then the maximum value of ab is 9. -/
theorem max_ab_is_nine {a b : ℝ}
  (ha : a > 0) (hb : b > 0)
  (extremum_x1 : deriv (f a b) 1 = 0) :
  a * b ≤ 9 :=
sorry

end NUMINAMATH_GPT_max_ab_is_nine_l1010_101044


namespace NUMINAMATH_GPT_smallest_value_l1010_101019

theorem smallest_value (x : ℝ) (h : 3 * x^2 + 33 * x - 90 = x * (x + 18)) : x ≥ -10.5 :=
sorry

end NUMINAMATH_GPT_smallest_value_l1010_101019


namespace NUMINAMATH_GPT_tan_difference_of_angle_l1010_101085

noncomputable def point_on_terminal_side (θ : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (2, 3) = (k * Real.cos θ, k * Real.sin θ)

theorem tan_difference_of_angle (θ : ℝ) (hθ : point_on_terminal_side θ) :
  Real.tan (θ - Real.pi / 4) = 1 / 5 :=
sorry

end NUMINAMATH_GPT_tan_difference_of_angle_l1010_101085


namespace NUMINAMATH_GPT_ellipse_and_line_properties_l1010_101032

theorem ellipse_and_line_properties :
  (∃ a b : ℝ, a > b ∧ b > 0 ∧ a * a = 4 ∧ b * b = 3 ∧
  ∀ x y : ℝ, (x, y) = (1, 3/2) → x^2 / a^2 + y^2 / b^2 = 1) ∧
  (∃ k : ℝ, k = 1 / 2 ∧ ∀ x y : ℝ, (x, y) = (2, 1) →
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧
  (x1 - 2) * (x2 - 2) + (k * (x1 - 2) + 1 - 1) * (k * (x2 - 2) + 1 - 1) = 5 / 4) :=
sorry

end NUMINAMATH_GPT_ellipse_and_line_properties_l1010_101032


namespace NUMINAMATH_GPT_same_color_probability_l1010_101048

-- Define the total number of balls
def total_balls : ℕ := 4 + 6 + 5

-- Define the number of each color of balls
def white_balls : ℕ := 4
def black_balls : ℕ := 6
def red_balls : ℕ := 5

-- Define the events and probabilities
def pr_event (n : ℕ) (total : ℕ) : ℚ := n / total
def pr_cond_event (n : ℕ) (total : ℕ) : ℚ := n / total

-- Define the probabilities for each compound event
def pr_C1 : ℚ := pr_event white_balls total_balls * pr_cond_event (white_balls - 1) (total_balls - 1)
def pr_C2 : ℚ := pr_event black_balls total_balls * pr_cond_event (black_balls - 1) (total_balls - 1)
def pr_C3 : ℚ := pr_event red_balls total_balls * pr_cond_event (red_balls - 1) (total_balls - 1)

-- Define the total probability
def pr_C : ℚ := pr_C1 + pr_C2 + pr_C3

-- The goal is to prove that the total probability pr_C is equal to 31 / 105
theorem same_color_probability : pr_C = 31 / 105 := 
  by sorry

end NUMINAMATH_GPT_same_color_probability_l1010_101048


namespace NUMINAMATH_GPT_total_money_difference_l1010_101063

-- Define the number of quarters each sibling has
def quarters_Karen : ℕ := 32
def quarters_Christopher : ℕ := 64
def quarters_Emily : ℕ := 20
def quarters_Michael : ℕ := 12

-- Define the value of each quarter
def value_per_quarter : ℚ := 0.25

-- Prove that the total money difference between the pairs of siblings is $16.00
theorem total_money_difference : 
  (quarters_Karen - quarters_Emily) * value_per_quarter + 
  (quarters_Christopher - quarters_Michael) * value_per_quarter = 16 := by
sorry

end NUMINAMATH_GPT_total_money_difference_l1010_101063


namespace NUMINAMATH_GPT_ratio_of_third_to_second_is_four_l1010_101078

theorem ratio_of_third_to_second_is_four
  (x y z k : ℕ)
  (h1 : y = 2 * x)
  (h2 : z = k * y)
  (h3 : (x + y + z) / 3 = 165)
  (h4 : y = 90) :
  z / y = 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_third_to_second_is_four_l1010_101078


namespace NUMINAMATH_GPT_num_ordered_pairs_eq_seven_l1010_101057

theorem num_ordered_pairs_eq_seven : ∃ n, n = 7 ∧ ∀ (x y : ℕ), (x * y = 64) → (x > 0 ∧ y > 0) → n = 7 :=
by
  sorry

end NUMINAMATH_GPT_num_ordered_pairs_eq_seven_l1010_101057


namespace NUMINAMATH_GPT_carla_wins_one_game_l1010_101029

/-
We are given the conditions:
Alice, Bob, and Carla each play each other twice in a round-robin format.
Alice won 5 games and lost 3 games.
Bob won 6 games and lost 2 games.
Carla lost 5 games.
We need to prove that Carla won 1 game.
-/

theorem carla_wins_one_game (games_per_match : Nat) 
                            (total_players : Nat)
                            (alice_wins : Nat) 
                            (alice_losses : Nat) 
                            (bob_wins : Nat) 
                            (bob_losses : Nat) 
                            (carla_losses : Nat) :
  (games_per_match = 2) → 
  (total_players = 3) → 
  (alice_wins = 5) → 
  (alice_losses = 3) → 
  (bob_wins = 6) → 
  (bob_losses = 2) → 
  (carla_losses = 5) → 
  ∃ (carla_wins : Nat), 
  carla_wins = 1 := 
by
  intros 
    games_match_eq total_players_eq 
    alice_wins_eq alice_losses_eq 
    bob_wins_eq bob_losses_eq 
    carla_losses_eq
  sorry

end NUMINAMATH_GPT_carla_wins_one_game_l1010_101029


namespace NUMINAMATH_GPT_emma_prob_at_least_one_correct_l1010_101068

-- Define the probability of getting a question wrong
def prob_wrong : ℚ := 4 / 5

-- Define the probability of getting all five questions wrong
def prob_all_wrong : ℚ := prob_wrong ^ 5

-- Define the probability of getting at least one question correct
def prob_at_least_one_correct : ℚ := 1 - prob_all_wrong

-- Define the main theorem to be proved
theorem emma_prob_at_least_one_correct : prob_at_least_one_correct = 2101 / 3125 := by
  sorry  -- This is where the proof would go

end NUMINAMATH_GPT_emma_prob_at_least_one_correct_l1010_101068


namespace NUMINAMATH_GPT_value_of_expression_l1010_101000

variable {a : ℝ}

theorem value_of_expression (h : a^2 + 2 * a - 1 = 0) : 2 * a^2 + 4 * a - 2024 = -2022 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1010_101000


namespace NUMINAMATH_GPT_min_sum_ab_72_l1010_101003

theorem min_sum_ab_72 (a b : ℤ) (h : a * b = 72) : a + b ≥ -17 := sorry

end NUMINAMATH_GPT_min_sum_ab_72_l1010_101003


namespace NUMINAMATH_GPT_measure_of_angle_A_l1010_101074

noncomputable def angle_A (angle_B : ℝ) := 3 * angle_B - 40

theorem measure_of_angle_A (x : ℝ) (angle_A_parallel_B : true) (h : ∃ k : ℝ, (k = x ∧ (angle_A x = x ∨ angle_A x + x = 180))) :
  angle_A x = 20 ∨ angle_A x = 125 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_A_l1010_101074


namespace NUMINAMATH_GPT_angle_between_line_and_plane_l1010_101031

variables (α β : ℝ) -- angles in radians
-- Definitions to capture the provided conditions
def dihedral_angle (α : ℝ) : Prop := true -- The angle between the planes γ₁ and γ₂
def angle_with_edge (β : ℝ) : Prop := true -- The angle between line AB and edge l

-- The angle between line AB and the plane γ₂
theorem angle_between_line_and_plane (α β : ℝ) (h1 : dihedral_angle α) (h2 : angle_with_edge β) : 
  ∃ θ : ℝ, θ = Real.arcsin (Real.sin α * Real.sin β) :=
by
  sorry

end NUMINAMATH_GPT_angle_between_line_and_plane_l1010_101031


namespace NUMINAMATH_GPT_intersection_is_correct_l1010_101046

def M : Set ℤ := {-2, 1, 2}
def N : Set ℤ := {1, 2, 4}

theorem intersection_is_correct : M ∩ N = {1, 2} := 
by {
  sorry
}

end NUMINAMATH_GPT_intersection_is_correct_l1010_101046


namespace NUMINAMATH_GPT_abs_discriminant_inequality_l1010_101098

theorem abs_discriminant_inequality 
  (a b c A B C : ℝ) 
  (ha : a ≠ 0) 
  (hA : A ≠ 0) 
  (h : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) : 
  |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| :=
sorry

end NUMINAMATH_GPT_abs_discriminant_inequality_l1010_101098


namespace NUMINAMATH_GPT_inequality_solution_l1010_101083

theorem inequality_solution (x : ℝ) : 
  (x + 10) / (x^2 + 2 * x + 5) ≥ 0 ↔ x ∈ Set.Ici (-10) :=
sorry

end NUMINAMATH_GPT_inequality_solution_l1010_101083


namespace NUMINAMATH_GPT_find_ellipse_equation_l1010_101016

-- Definitions based on conditions
def ellipse_centered_at_origin (x y : ℝ) (m n : ℝ) := m * x ^ 2 + n * y ^ 2 = 1

def passes_through_points_A_and_B (m n : ℝ) := 
  (ellipse_centered_at_origin 0 (-2) m n) ∧ (ellipse_centered_at_origin (3 / 2) (-1) m n)

-- Statement to be proved
theorem find_ellipse_equation : 
  ∃ (m n : ℝ), (m > 0) ∧ (n > 0) ∧ (m ≠ n) ∧ 
  passes_through_points_A_and_B m n ∧ 
  m = 1 / 3 ∧ n = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_find_ellipse_equation_l1010_101016


namespace NUMINAMATH_GPT_find_a_plus_b_l1010_101004

theorem find_a_plus_b (a b : ℕ) 
  (h1 : 2^(2 * a) + 2^b + 5 = k^2) : a + b = 4 ∨ a + b = 5 :=
sorry

end NUMINAMATH_GPT_find_a_plus_b_l1010_101004


namespace NUMINAMATH_GPT_value_of_c_in_base8_perfect_cube_l1010_101043

theorem value_of_c_in_base8_perfect_cube (c : ℕ) (h : 0 ≤ c ∧ c < 8) :
  4 * 8^2 + c * 8 + 3 = x^3 → c = 0 := by
  sorry

end NUMINAMATH_GPT_value_of_c_in_base8_perfect_cube_l1010_101043


namespace NUMINAMATH_GPT_lifespan_represents_sample_l1010_101001

-- Definitions
def survey_population := 2500
def provinces_and_cities := 11

-- Theorem stating that the lifespan of the urban residents surveyed represents a sample
theorem lifespan_represents_sample
  (number_of_residents : ℕ) (num_provinces : ℕ) 
  (h₁ : number_of_residents = survey_population)
  (h₂ : num_provinces = provinces_and_cities) :
  "Sample" = "Sample" :=
by 
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_lifespan_represents_sample_l1010_101001


namespace NUMINAMATH_GPT_sum_non_solutions_l1010_101025

theorem sum_non_solutions (A B C : ℝ) (h : ∀ x, (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9) → x ≠ -12) :
  -12 = -12 := 
sorry

end NUMINAMATH_GPT_sum_non_solutions_l1010_101025


namespace NUMINAMATH_GPT_simplify_fraction_l1010_101024

theorem simplify_fraction (x : ℝ) :
  ((x + 2) / 4) + ((3 - 4 * x) / 3) = (18 - 13 * x) / 12 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1010_101024


namespace NUMINAMATH_GPT_negation_of_neither_even_l1010_101023

variable (a b : Nat)

def is_even (n : Nat) : Prop :=
  n % 2 = 0

theorem negation_of_neither_even 
  (H : ¬ (¬ is_even a ∧ ¬ is_even b)) : is_even a ∨ is_even b :=
sorry

end NUMINAMATH_GPT_negation_of_neither_even_l1010_101023


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1010_101041

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (a4_eq_3 : a 4 = 3) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 21 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1010_101041


namespace NUMINAMATH_GPT_Isabel_subtasks_remaining_l1010_101010

-- Definition of the known quantities
def Total_problems : ℕ := 72
def Completed_problems : ℕ := 32
def Subtasks_per_problem : ℕ := 5

-- Definition of the calculations
def Total_subtasks : ℕ := Total_problems * Subtasks_per_problem
def Completed_subtasks : ℕ := Completed_problems * Subtasks_per_problem
def Remaining_subtasks : ℕ := Total_subtasks - Completed_subtasks

-- The theorem we need to prove
theorem Isabel_subtasks_remaining : Remaining_subtasks = 200 := by
  -- Proof would go here, but we'll use sorry to indicate it's omitted
  sorry

end NUMINAMATH_GPT_Isabel_subtasks_remaining_l1010_101010


namespace NUMINAMATH_GPT_area_of_square_is_25_l1010_101030

-- Define side length of the square
def sideLength : ℝ := 5

-- Define the area of the square
def area_of_square (side : ℝ) : ℝ := side * side

-- Prove the area of the square with side length 5 is 25 square meters
theorem area_of_square_is_25 : area_of_square sideLength = 25 := by
  sorry

end NUMINAMATH_GPT_area_of_square_is_25_l1010_101030


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1010_101058

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 5) : b = 8 := 
by
  /- The proof steps would go here -/
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1010_101058


namespace NUMINAMATH_GPT_lines_intersect_at_point_l1010_101096

noncomputable def line1 (s : ℚ) : ℚ × ℚ :=
  (1 + 2 * s, 4 - 3 * s)

noncomputable def line2 (v : ℚ) : ℚ × ℚ :=
  (3 + 3 * v, 2 - v)

theorem lines_intersect_at_point :
  ∃ s v : ℚ,
    line1 s = (15 / 7, 16 / 7) ∧
    line2 v = (15 / 7, 16 / 7) ∧
    s = 4 / 7 ∧
    v = -2 / 7 := by
  sorry

end NUMINAMATH_GPT_lines_intersect_at_point_l1010_101096


namespace NUMINAMATH_GPT_jerome_contact_list_count_l1010_101038

theorem jerome_contact_list_count :
  (let classmates := 20
   let out_of_school_friends := classmates / 2
   let family := 3 -- two parents and one sister
   let total_contacts := classmates + out_of_school_friends + family
   total_contacts = 33) :=
by
  let classmates := 20
  let out_of_school_friends := classmates / 2
  let family := 3
  let total_contacts := classmates + out_of_school_friends + family
  show total_contacts = 33
  sorry

end NUMINAMATH_GPT_jerome_contact_list_count_l1010_101038


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l1010_101027

theorem no_positive_integer_solutions :
  ¬ ∃ (x1 x2 : ℕ), 903 * x1 + 731 * x2 = 1106 := by
  sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l1010_101027


namespace NUMINAMATH_GPT_range_of_a_l1010_101087

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, ∃ y ∈ Set.Ici a, y = (x^2 + 2*x + a) / (x + 1)) ↔ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1010_101087


namespace NUMINAMATH_GPT_product_neg_int_add_five_l1010_101049

theorem product_neg_int_add_five:
  let x := -11 
  let y := -8 
  x * y + 5 = 93 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_product_neg_int_add_five_l1010_101049


namespace NUMINAMATH_GPT_peter_pizza_fraction_l1010_101099

def pizza_slices : ℕ := 16
def peter_slices_alone : ℕ := 2
def shared_slice : ℚ := 1 / 2

theorem peter_pizza_fraction :
  let fraction_alone := peter_slices_alone * (1 / pizza_slices)
  let fraction_shared := shared_slice * (1 / pizza_slices)
  let total_fraction := fraction_alone + fraction_shared
  total_fraction = 5 / 32 :=
by
  let fraction_alone := peter_slices_alone * (1 / pizza_slices)
  let fraction_shared := shared_slice * (1 / pizza_slices)
  let total_fraction := fraction_alone + fraction_shared
  sorry

end NUMINAMATH_GPT_peter_pizza_fraction_l1010_101099


namespace NUMINAMATH_GPT_harry_books_l1010_101014

theorem harry_books : ∀ (H : ℝ), 
  (H + 2 * H + H / 2 = 175) → 
  H = 50 :=
by
  intros H h_sum
  sorry

end NUMINAMATH_GPT_harry_books_l1010_101014


namespace NUMINAMATH_GPT_inverse_value_l1010_101028

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 25 / (7 + 2 * x)

-- Define the goal of the proof
theorem inverse_value {g : ℝ → ℝ}
  (h : ∀ y, g (g⁻¹ y) = y) :
  ((g⁻¹ 5)⁻¹) = -1 :=
by
  sorry

end NUMINAMATH_GPT_inverse_value_l1010_101028


namespace NUMINAMATH_GPT_range_of_a_l1010_101033

open Real

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x ^ 2 + a * x - 1 < 0) ↔ -4 < a ∧ a ≤ 0 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1010_101033


namespace NUMINAMATH_GPT_ones_digit_exponent_73_l1010_101079

theorem ones_digit_exponent_73 (n : ℕ) : 
  (73 ^ n) % 10 = 7 ↔ n % 4 = 3 := 
sorry

end NUMINAMATH_GPT_ones_digit_exponent_73_l1010_101079


namespace NUMINAMATH_GPT_find_b_l1010_101039

-- Define the conditions as hypotheses
def f (b : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + b*x - 3

theorem find_b (x₁ x₂ b : ℝ) (h₁ : x₁ ≠ x₂)
  (h₂ : 3 * x₁^2 + 4 * x₁ + b = 0)
  (h₃ : 3 * x₂^2 + 4 * x₂ + b = 0)
  (h₄ : x₁^2 + x₂^2 = 34 / 9) :
  b = -3 :=
by
  -- Proof will be inserted here
  sorry

end NUMINAMATH_GPT_find_b_l1010_101039


namespace NUMINAMATH_GPT_jumps_correct_l1010_101073

def R : ℕ := 157
def X : ℕ := 86
def total_jumps (R X : ℕ) : ℕ := R + (R + X)

theorem jumps_correct : total_jumps R X = 400 := by
  sorry

end NUMINAMATH_GPT_jumps_correct_l1010_101073


namespace NUMINAMATH_GPT_positive_difference_eq_six_l1010_101047

theorem positive_difference_eq_six (x y : ℝ) (h1 : x + y = 8) (h2 : x ^ 2 - y ^ 2 = 48) : |x - y| = 6 := by
  sorry

end NUMINAMATH_GPT_positive_difference_eq_six_l1010_101047


namespace NUMINAMATH_GPT_suraya_picked_more_apples_l1010_101022

theorem suraya_picked_more_apples (suraya caleb kayla : ℕ) 
  (h1 : suraya = caleb + 12)
  (h2 : caleb = kayla - 5)
  (h3 : kayla = 20) : suraya - kayla = 7 := by
  sorry

end NUMINAMATH_GPT_suraya_picked_more_apples_l1010_101022


namespace NUMINAMATH_GPT_prove_f_three_eq_neg_three_l1010_101008

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.tan x + 1

theorem prove_f_three_eq_neg_three (a b : ℝ) (h : f (-3) a b = 5) : f 3 a b = -3 := by
  sorry

end NUMINAMATH_GPT_prove_f_three_eq_neg_three_l1010_101008


namespace NUMINAMATH_GPT_audrey_peaches_l1010_101050

variable (A : ℕ)
variable (P : ℕ := 48)
variable (D : ℕ := 22)

theorem audrey_peaches : A - P = D → A = 70 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_audrey_peaches_l1010_101050


namespace NUMINAMATH_GPT_hyperbola_asymptote_distance_l1010_101088

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

end NUMINAMATH_GPT_hyperbola_asymptote_distance_l1010_101088


namespace NUMINAMATH_GPT_coin_probability_l1010_101080

theorem coin_probability :
  let PA := 3/4
  let PB := 1/2
  let PC := 1/4
  (PA * PB * (1 - PC)) = 9/32 :=
by
  sorry

end NUMINAMATH_GPT_coin_probability_l1010_101080


namespace NUMINAMATH_GPT_find_A_l1010_101051

theorem find_A (A : ℝ) (h : 4 * A + 5 = 33) : A = 7 :=
  sorry

end NUMINAMATH_GPT_find_A_l1010_101051


namespace NUMINAMATH_GPT_height_of_box_l1010_101062

theorem height_of_box (h : ℝ) :
  (∃ (h : ℝ),
    (∀ (x y z : ℝ), (x = 3) ∧ (y = 3) ∧ (z = h / 2) → true) ∧
    (∀ (x y z : ℝ), (x = 1) ∧ (y = 1) ∧ (z = 1) → true) ∧
    h = 6) :=
sorry

end NUMINAMATH_GPT_height_of_box_l1010_101062


namespace NUMINAMATH_GPT_problem_statement_l1010_101040

theorem problem_statement (x : ℤ) (h : 3 - x = -2) : x + 1 = 6 := 
by {
  -- Proof would be provided here
  sorry
}

end NUMINAMATH_GPT_problem_statement_l1010_101040


namespace NUMINAMATH_GPT_problem_conditions_l1010_101061

theorem problem_conditions (m : ℝ) (hf_pow : m^2 - m - 1 = 1) (hf_inc : m > 0) : m = 2 :=
sorry

end NUMINAMATH_GPT_problem_conditions_l1010_101061


namespace NUMINAMATH_GPT_ian_investment_percentage_change_l1010_101060

theorem ian_investment_percentage_change :
  let initial_investment := 200
  let first_year_loss := 0.10
  let second_year_gain := 0.25
  let amount_after_loss := initial_investment * (1 - first_year_loss)
  let amount_after_gain := amount_after_loss * (1 + second_year_gain)
  let percentage_change := (amount_after_gain - initial_investment) / initial_investment * 100
  percentage_change = 12.5 := 
by
  sorry

end NUMINAMATH_GPT_ian_investment_percentage_change_l1010_101060


namespace NUMINAMATH_GPT_equivalent_proof_problem_l1010_101013

def op (a b : ℝ) : ℝ := (a + b) ^ 2

theorem equivalent_proof_problem (x y : ℝ) : 
  op ((x + y) ^ 2) ((x - y) ^ 2) = 4 * (x ^ 2 + y ^ 2) ^ 2 := 
by 
  sorry

end NUMINAMATH_GPT_equivalent_proof_problem_l1010_101013


namespace NUMINAMATH_GPT_cat_food_sufficiency_l1010_101066

theorem cat_food_sufficiency (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end NUMINAMATH_GPT_cat_food_sufficiency_l1010_101066


namespace NUMINAMATH_GPT_perpendicular_vectors_m_value_l1010_101064

theorem perpendicular_vectors_m_value
  (a : ℝ × ℝ := (1, 2))
  (b : ℝ × ℝ)
  (h_perpendicular : (a.1 * b.1 + a.2 * b.2) = 0) :
  b = (-2, 1) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_m_value_l1010_101064


namespace NUMINAMATH_GPT_smallest_of_product_and_sum_l1010_101015

theorem smallest_of_product_and_sum (a b c : ℤ) 
  (h1 : a * b * c = 32) 
  (h2 : a + b + c = 3) : 
  a = -4 ∨ b = -4 ∨ c = -4 :=
sorry

end NUMINAMATH_GPT_smallest_of_product_and_sum_l1010_101015


namespace NUMINAMATH_GPT_mul_equiv_l1010_101076

theorem mul_equiv :
  (213 : ℝ) * 16 = 3408 →
  (16 : ℝ) * 21.3 = 340.8 :=
by
  sorry

end NUMINAMATH_GPT_mul_equiv_l1010_101076


namespace NUMINAMATH_GPT_find_k_l1010_101006

theorem find_k
  (angle_C : ℝ)
  (AB : ℝ × ℝ)
  (AC : ℝ × ℝ)
  (h1 : angle_C = 90)
  (h2 : AB = (k, 1))
  (h3 : AC = (2, 3)) :
  k = 5 := by
  sorry

end NUMINAMATH_GPT_find_k_l1010_101006


namespace NUMINAMATH_GPT_find_y_l1010_101052

open Complex

theorem find_y (y : ℝ) (h₁ : (3 : ℂ) + (↑y : ℂ) * I = z₁) 
  (h₂ : (2 : ℂ) - I = z₂) 
  (h₃ : z₁ / z₂ = 1 + I) 
  (h₄ : z₁ = (3 : ℂ) + (↑y : ℂ) * I) 
  (h₅ : z₂ = (2 : ℂ) - I)
  : y = 1 :=
sorry


end NUMINAMATH_GPT_find_y_l1010_101052


namespace NUMINAMATH_GPT_unique_function_satisfying_conditions_l1010_101067

theorem unique_function_satisfying_conditions :
  ∀ f : ℚ → ℚ, (f 1 = 2) → (∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) → (∀ x : ℚ, f x = x + 1) :=
by
  intro f h1 hCond
  sorry

end NUMINAMATH_GPT_unique_function_satisfying_conditions_l1010_101067


namespace NUMINAMATH_GPT_total_movies_seen_l1010_101012

theorem total_movies_seen (d h a c : ℕ) (hd : d = 7) (hh : h = 12) (ha : a = 15) (hc : c = 2) :
  (c + (d - c) + (h - c) + (a - c)) = 30 :=
by
  sorry

end NUMINAMATH_GPT_total_movies_seen_l1010_101012


namespace NUMINAMATH_GPT_bananas_left_l1010_101036

theorem bananas_left (original_bananas : ℕ) (bananas_eaten : ℕ) 
  (h1 : original_bananas = 12) (h2 : bananas_eaten = 4) : 
  original_bananas - bananas_eaten = 8 := 
by
  sorry

end NUMINAMATH_GPT_bananas_left_l1010_101036


namespace NUMINAMATH_GPT_tom_gets_correct_share_l1010_101034

def total_savings : ℝ := 18500.0
def natalie_share : ℝ := 0.35 * total_savings
def remaining_after_natalie : ℝ := total_savings - natalie_share
def rick_share : ℝ := 0.30 * remaining_after_natalie
def remaining_after_rick : ℝ := remaining_after_natalie - rick_share
def lucy_share : ℝ := 0.40 * remaining_after_rick
def remaining_after_lucy : ℝ := remaining_after_rick - lucy_share
def minimum_share : ℝ := 1000.0
def tom_share : ℝ := remaining_after_lucy

theorem tom_gets_correct_share :
  (natalie_share ≥ minimum_share) ∧ (rick_share ≥ minimum_share) ∧ (lucy_share ≥ minimum_share) →
  tom_share = 5050.50 :=
by
  sorry

end NUMINAMATH_GPT_tom_gets_correct_share_l1010_101034


namespace NUMINAMATH_GPT_proof_6_times_15_times_5_eq_2_l1010_101097

noncomputable def given_condition (a b c : ℝ) : Prop :=
  a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

theorem proof_6_times_15_times_5_eq_2 : 
  given_condition 6 15 5 → 6 * 15 * 5 = 2 :=
by
  sorry

end NUMINAMATH_GPT_proof_6_times_15_times_5_eq_2_l1010_101097


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1010_101093

theorem solution_set_of_inequality (x : ℝ) : 
  (3 * x - 4 > 2) → (x > 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1010_101093


namespace NUMINAMATH_GPT_rebecca_groups_of_eggs_l1010_101017

def eggs : Nat := 16
def group_size : Nat := 2

theorem rebecca_groups_of_eggs : (eggs / group_size) = 8 := by
  sorry

end NUMINAMATH_GPT_rebecca_groups_of_eggs_l1010_101017


namespace NUMINAMATH_GPT_certain_event_idiom_l1010_101090

theorem certain_event_idiom : 
  ∃ (idiom : String), idiom = "Catching a turtle in a jar" ∧ 
  ∀ (option : String), 
    option = "Catching a turtle in a jar" ∨ 
    option = "Carving a boat to find a sword" ∨ 
    option = "Waiting by a tree stump for a rabbit" ∨ 
    option = "Fishing for the moon in the water" → 
    (option = idiom ↔ (option = "Catching a turtle in a jar")) := 
by
  sorry

end NUMINAMATH_GPT_certain_event_idiom_l1010_101090


namespace NUMINAMATH_GPT_domain_of_f_intervals_of_monotonicity_extremal_values_l1010_101026

noncomputable def f (x : ℝ) := (1 / 2) * x ^ 2 - 5 * x + 4 * Real.log x 

theorem domain_of_f : ∀ x, 0 < x → f x = (1 / 2) * x ^ 2 - 5 * x + 4 * Real.log x :=
by
  intro x hx
  exact rfl

theorem intervals_of_monotonicity :
  (∀ x, 0 < x ∧ x < 1 → f x < f 1) ∧
  (∀ x, 1 < x ∧ x < 4 → f x > f 1 ∧ f x < f 4) ∧
  (∀ x, 4 < x → f x > f 4) :=
sorry

theorem extremal_values :
  (f 1 = - (9 / 2)) ∧ 
  (f 4 = -12 + 4 * Real.log 4) :=
sorry

end NUMINAMATH_GPT_domain_of_f_intervals_of_monotonicity_extremal_values_l1010_101026


namespace NUMINAMATH_GPT_line_equation_l1010_101054

variable (t : ℝ)
variable (x y : ℝ)

def param_x (t : ℝ) : ℝ := 3 * t + 2
def param_y (t : ℝ) : ℝ := 5 * t - 7

theorem line_equation :
  ∃ m b : ℝ, ∀ t : ℝ, y = param_y t ∧ x = param_x t → y = m * x + b := by
  use (5 / 3)
  use (-31 / 3)
  sorry

end NUMINAMATH_GPT_line_equation_l1010_101054


namespace NUMINAMATH_GPT_sum_of_equal_numbers_l1010_101005

theorem sum_of_equal_numbers (a b : ℝ) (h1 : (12 + 25 + 18 + a + b) / 5 = 20) (h2 : a = b) : a + b = 45 :=
sorry

end NUMINAMATH_GPT_sum_of_equal_numbers_l1010_101005


namespace NUMINAMATH_GPT_total_earnings_correct_l1010_101035

-- Define the earnings of Terrence
def TerrenceEarnings : ℕ := 30

-- Define the difference in earnings between Jermaine and Terrence
def JermaineEarningsDifference : ℕ := 5

-- Define the earnings of Jermaine
def JermaineEarnings : ℕ := TerrenceEarnings + JermaineEarningsDifference

-- Define the earnings of Emilee
def EmileeEarnings : ℕ := 25

-- Define the total earnings
def TotalEarnings : ℕ := TerrenceEarnings + JermaineEarnings + EmileeEarnings

theorem total_earnings_correct : TotalEarnings = 90 := by
  sorry

end NUMINAMATH_GPT_total_earnings_correct_l1010_101035


namespace NUMINAMATH_GPT_tangent_line_at_point_l1010_101018

theorem tangent_line_at_point (x y : ℝ) (h_curve : y = x^3 - 2 * x + 1) (h_point : (x, y) = (1, 0)) :
  y = x - 1 :=
sorry

end NUMINAMATH_GPT_tangent_line_at_point_l1010_101018


namespace NUMINAMATH_GPT_tablets_of_medicine_A_l1010_101053

-- Given conditions as definitions
def B_tablets : ℕ := 16

def min_extracted_tablets : ℕ := 18

-- Question and expected answer encapsulated in proof statement
theorem tablets_of_medicine_A (A_tablets : ℕ) (h : A_tablets + B_tablets - 2 >= min_extracted_tablets) : A_tablets = 3 :=
sorry

end NUMINAMATH_GPT_tablets_of_medicine_A_l1010_101053


namespace NUMINAMATH_GPT_double_inequality_solution_l1010_101081

open Set

theorem double_inequality_solution (x : ℝ) :
  -1 < (x^2 - 16 * x + 24) / (x^2 - 4 * x + 8) ∧
  (x^2 - 16 * x + 24) / (x^2 - 4 * x + 8) < 1 ↔
  x ∈ Ioo (3 / 2) 4 ∪ Ioi 8 :=
by
  sorry

end NUMINAMATH_GPT_double_inequality_solution_l1010_101081


namespace NUMINAMATH_GPT_Jason_toys_correct_l1010_101082

variable (R Jn Js : ℕ)

def Rachel_toys : ℕ := 1

def John_toys (R : ℕ) : ℕ := R + 6

def Jason_toys (Jn : ℕ) : ℕ := 3 * Jn

theorem Jason_toys_correct (hR : R = 1) (hJn : Jn = John_toys R) (hJs : Js = Jason_toys Jn) : Js = 21 :=
by
  sorry

end NUMINAMATH_GPT_Jason_toys_correct_l1010_101082


namespace NUMINAMATH_GPT_rational_solutions_iff_k_equals_8_l1010_101065

theorem rational_solutions_iff_k_equals_8 {k : ℕ} (hk : k > 0) :
  (∃ (x : ℚ), k * x^2 + 16 * x + k = 0) ↔ k = 8 :=
by
  sorry

end NUMINAMATH_GPT_rational_solutions_iff_k_equals_8_l1010_101065


namespace NUMINAMATH_GPT_smallest_positive_divisible_by_111_has_last_digits_2004_l1010_101042

theorem smallest_positive_divisible_by_111_has_last_digits_2004 :
  ∃ (X : ℕ), (∃ (A : ℕ), X = A * 10^4 + 2004) ∧ 111 ∣ X ∧ X = 662004 := by
  sorry

end NUMINAMATH_GPT_smallest_positive_divisible_by_111_has_last_digits_2004_l1010_101042


namespace NUMINAMATH_GPT_jeans_cost_before_sales_tax_l1010_101095

-- Defining conditions
def original_cost : ℝ := 49
def summer_discount : ℝ := 0.50
def wednesday_discount : ℝ := 10

-- The mathematical equivalent proof problem
theorem jeans_cost_before_sales_tax :
  let discount_price := original_cost * (1 - summer_discount)
  let wednesday_price := discount_price - wednesday_discount
  wednesday_price = 14.50 :=
by
  let discount_price := original_cost * (1 - summer_discount)
  let wednesday_price := discount_price - wednesday_discount
  sorry

end NUMINAMATH_GPT_jeans_cost_before_sales_tax_l1010_101095


namespace NUMINAMATH_GPT_eval_expression_l1010_101084

theorem eval_expression (x : ℝ) (h₀ : x = 3) :
  let initial_expr : ℝ := (2 * x + 2) / (x - 2)
  let replaced_expr : ℝ := (2 * initial_expr + 2) / (initial_expr - 2)
  replaced_expr = 8 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1010_101084


namespace NUMINAMATH_GPT_initial_quantity_l1010_101009

variables {A : ℝ} -- initial quantity of acidic liquid
variables {W : ℝ} -- quantity of water removed

theorem initial_quantity (h1: A * 0.6 = W + 25) (h2: W = 9) : A = 27 :=
by
  sorry

end NUMINAMATH_GPT_initial_quantity_l1010_101009


namespace NUMINAMATH_GPT_volume_of_rectangular_prism_l1010_101020

theorem volume_of_rectangular_prism (l w h : ℕ) (x : ℕ) 
  (h_ratio : l = 3 * x ∧ w = 2 * x ∧ h = x)
  (h_edges : 4 * l + 4 * w + 4 * h = 72) : 
  l * w * h = 162 := 
by
  sorry

end NUMINAMATH_GPT_volume_of_rectangular_prism_l1010_101020


namespace NUMINAMATH_GPT_sum_reciprocals_of_roots_l1010_101045

-- Problem statement: Prove that the sum of the reciprocals of the roots of the quadratic equation x^2 - 11x + 6 = 0 is 11/6.
theorem sum_reciprocals_of_roots : 
  ∀ (p q : ℝ), p + q = 11 → p * q = 6 → (1 / p + 1 / q = 11 / 6) :=
by
  intro p q hpq hprod
  sorry

end NUMINAMATH_GPT_sum_reciprocals_of_roots_l1010_101045


namespace NUMINAMATH_GPT_sum_f_eq_28743_l1010_101059

def f (n : ℕ) : ℕ := 4 * n ^ 3 - 6 * n ^ 2 + 4 * n + 13

theorem sum_f_eq_28743 : (Finset.range 13).sum (λ n => f (n + 1)) = 28743 :=
by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_sum_f_eq_28743_l1010_101059


namespace NUMINAMATH_GPT_sum_of_consecutive_page_numbers_l1010_101094

theorem sum_of_consecutive_page_numbers (n : ℕ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_page_numbers_l1010_101094


namespace NUMINAMATH_GPT_debate_team_selections_l1010_101007

theorem debate_team_selections
  (A_selected C_selected B_selected E_selected : Prop)
  (h1: A_selected ∨ C_selected)
  (h2: B_selected ∨ E_selected)
  (h3: ¬ (B_selected ∧ E_selected) ∧ ¬ (C_selected ∧ E_selected))
  (not_B_selected : ¬ B_selected) :
  A_selected ∧ E_selected :=
by
  sorry

end NUMINAMATH_GPT_debate_team_selections_l1010_101007


namespace NUMINAMATH_GPT_alice_average_speed_l1010_101021

/-- Alice cycled 40 miles at 8 miles per hour and 20 miles at 40 miles per hour. 
    The average speed for the entire trip --/
theorem alice_average_speed :
  let distance1 := 40
  let speed1 := 8
  let distance2 := 20
  let speed2 := 40
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  (total_distance / total_time) = (120 / 11) := 
by
  sorry -- proof steps would go here

end NUMINAMATH_GPT_alice_average_speed_l1010_101021


namespace NUMINAMATH_GPT_isosceles_triangle_of_cosine_condition_l1010_101055

theorem isosceles_triangle_of_cosine_condition
  (A B C : ℝ)
  (h : 2 * Real.cos A * Real.cos B = 1 - Real.cos C) :
  A = B ∨ A = π - B :=
  sorry

end NUMINAMATH_GPT_isosceles_triangle_of_cosine_condition_l1010_101055


namespace NUMINAMATH_GPT_compute_f_at_2012_l1010_101069

noncomputable def B := { x : ℚ | x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 2 }

noncomputable def h (x : ℚ) : ℚ := 2 - (1 / x)

noncomputable def f (x : B) : ℝ := sorry  -- As a placeholder since the definition isn't given directly

-- Main theorem
theorem compute_f_at_2012 : 
  (∀ x : B, f x + f ⟨h x, sorry⟩ = Real.log (abs (2 * (x : ℚ)))) →
  f ⟨2012, sorry⟩ = Real.log ((4024 : ℚ) / (4023 : ℚ)) :=
sorry

end NUMINAMATH_GPT_compute_f_at_2012_l1010_101069


namespace NUMINAMATH_GPT_complement_intersection_empty_l1010_101075

open Set

-- Given definitions and conditions
def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3}

-- Complement operation with respect to U
def C_U (X : Set ℕ) : Set ℕ := U \ X

-- The proof statement to be shown
theorem complement_intersection_empty :
  (C_U A ∩ C_U B) = ∅ := by sorry

end NUMINAMATH_GPT_complement_intersection_empty_l1010_101075
