import Mathlib

namespace NUMINAMATH_GPT_expected_value_decisive_games_l707_70745

/-- According to the rules of a chess match, the winner is the one who gains two victories over the opponent. -/
def winner_conditions (a b : Nat) : Prop :=
  a = 2 ∨ b = 2

/-- A game match where the probabilities of winning for the opponents are equal.-/
def probabilities_equal : Prop :=
  true

/-- Define X as the random variable representing the number of decisive games in the match. -/
def X (a b : Nat) : Nat :=
  a + b

/-- The expected value of the number of decisive games given equal probabilities of winning. -/
theorem expected_value_decisive_games (a b : Nat) (h1 : winner_conditions a b) (h2 : probabilities_equal) : 
  (X a b) / 2 = 4 :=
sorry

end NUMINAMATH_GPT_expected_value_decisive_games_l707_70745


namespace NUMINAMATH_GPT_Sue_button_count_l707_70742

variable (K S : ℕ)

theorem Sue_button_count (H1 : 64 = 5 * K + 4) (H2 : S = K / 2) : S = 6 := 
by
sorry

end NUMINAMATH_GPT_Sue_button_count_l707_70742


namespace NUMINAMATH_GPT_number_of_strawberry_cakes_l707_70719

def number_of_chocolate_cakes := 3
def price_of_chocolate_cake := 12
def price_of_strawberry_cake := 22
def total_payment := 168

theorem number_of_strawberry_cakes (S : ℕ) : 
    number_of_chocolate_cakes * price_of_chocolate_cake + S * price_of_strawberry_cake = total_payment → 
    S = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_strawberry_cakes_l707_70719


namespace NUMINAMATH_GPT_proof_solution_l707_70726

noncomputable def proof_problem : Prop :=
  ∀ (x y z : ℝ), 3 * x - 4 * y - 2 * z = 0 ∧ x - 2 * y - 8 * z = 0 ∧ z ≠ 0 → 
  (x^2 + 3 * x * y) / (y^2 + z^2) = 329 / 61

theorem proof_solution : proof_problem :=
by
  intros x y z h
  sorry

end NUMINAMATH_GPT_proof_solution_l707_70726


namespace NUMINAMATH_GPT_circles_intersect_l707_70708

-- Definitions of the circles
def circle_O1 := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1}
def circle_O2 := {p : ℝ × ℝ | p.1^2 + (p.2 - 3)^2 = 9}

-- Proving the relationship between the circles
theorem circles_intersect : ∀ (p : ℝ × ℝ),
  p ∈ circle_O1 ∧ p ∈ circle_O2 :=
sorry

end NUMINAMATH_GPT_circles_intersect_l707_70708


namespace NUMINAMATH_GPT_transformed_cube_edges_l707_70736

-- Let's define the problem statement
theorem transformed_cube_edges : 
  let original_edges := 12 
  let new_edges_per_edge := 2 
  let additional_edges_per_pyramid := 1 
  let total_edges := original_edges + (original_edges * new_edges_per_edge) + (original_edges * additional_edges_per_pyramid) 
  total_edges = 48 :=
by sorry

end NUMINAMATH_GPT_transformed_cube_edges_l707_70736


namespace NUMINAMATH_GPT_proposition_not_hold_for_4_l707_70780

variable (P : ℕ → Prop)

axiom induction_step (k : ℕ) (hk : k > 0) : P k → P (k + 1)
axiom base_case : ¬ P 5

theorem proposition_not_hold_for_4 : ¬ P 4 :=
sorry

end NUMINAMATH_GPT_proposition_not_hold_for_4_l707_70780


namespace NUMINAMATH_GPT_third_pasture_cows_l707_70772

theorem third_pasture_cows (x y : ℝ) (H1 : x + 27 * y = 18) (H2 : 2 * x + 84 * y = 51) : 
  10 * x + 10 * 3 * y = 60 -> 60 / 3 = 20 :=
by
  sorry

end NUMINAMATH_GPT_third_pasture_cows_l707_70772


namespace NUMINAMATH_GPT_find_n_l707_70797

-- Define x and y
def x : ℕ := 3
def y : ℕ := 1

-- Define n based on the given expression.
def n : ℕ := x - y^(x - (y + 1))

-- State the theorem
theorem find_n : n = 2 := by
  sorry

end NUMINAMATH_GPT_find_n_l707_70797


namespace NUMINAMATH_GPT_tracy_initial_candies_l707_70735

theorem tracy_initial_candies 
  (x : ℕ)
  (h1 : 4 ∣ x)
  (h2 : 5 ≤ ((x / 2) - 24))
  (h3 : ((x / 2) - 24) ≤ 9) 
  : x = 68 :=
sorry

end NUMINAMATH_GPT_tracy_initial_candies_l707_70735


namespace NUMINAMATH_GPT_minimum_value_of_sum_of_squares_l707_70770

noncomputable def minimum_of_sum_of_squares (a b : ℝ) : ℝ :=
  a^2 + b^2

theorem minimum_value_of_sum_of_squares (a b : ℝ) (h : |a * b| = 6) :
  a^2 + b^2 ≥ 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_value_of_sum_of_squares_l707_70770


namespace NUMINAMATH_GPT_value_of_t_for_x_equals_y_l707_70706

theorem value_of_t_for_x_equals_y (t : ℝ) (h1 : x = 1 - 4 * t) (h2 : y = 2 * t - 2) : 
    t = 1 / 2 → x = y :=
by 
  intro ht
  rw [ht] at h1 h2
  sorry

end NUMINAMATH_GPT_value_of_t_for_x_equals_y_l707_70706


namespace NUMINAMATH_GPT_Ursula_hours_per_day_l707_70718

theorem Ursula_hours_per_day (hourly_wage : ℝ) (days_per_month : ℕ) (annual_salary : ℝ) (months_per_year : ℕ) :
  hourly_wage = 8.5 →
  days_per_month = 20 →
  annual_salary = 16320 →
  months_per_year = 12 →
  (annual_salary / months_per_year / days_per_month / hourly_wage) = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Ursula_hours_per_day_l707_70718


namespace NUMINAMATH_GPT_intersection_eq_l707_70790

-- Universal set and its sets M and N
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 > 9}
def N : Set ℝ := {x | -1 < x ∧ x < 4}
def complement_N : Set ℝ := {x | x ≤ -1 ∨ x ≥ 4}

-- Prove the intersection
theorem intersection_eq :
  M ∩ complement_N = {x | x < -3 ∨ x ≥ 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l707_70790


namespace NUMINAMATH_GPT_triangle_side_lengths_relation_l707_70729

-- Given a triangle ABC with side lengths a, b, c
variables (a b c R d : ℝ)
-- Given orthocenter H and circumcenter O, and the radius of the circumcircle is R,
-- and distance between O and H is d.
-- Prove that a² + b² + c² = 9R² - d²

theorem triangle_side_lengths_relation (a b c R d : ℝ) (H O : Type) (orthocenter : H) (circumcenter : O)
  (radius_circumcircle : O → ℝ)
  (distance_OH : O → H → ℝ) :
  a^2 + b^2 + c^2 = 9 * R^2 - d^2 :=
sorry

end NUMINAMATH_GPT_triangle_side_lengths_relation_l707_70729


namespace NUMINAMATH_GPT_quilt_width_l707_70720

-- Definitions according to the conditions
def quilt_length : ℕ := 16
def patch_area : ℕ := 4
def first_10_patches_cost : ℕ := 100
def total_cost : ℕ := 450
def remaining_budget : ℕ := total_cost - first_10_patches_cost
def cost_per_additional_patch : ℕ := 5
def num_additional_patches : ℕ := remaining_budget / cost_per_additional_patch
def total_patches : ℕ := 10 + num_additional_patches
def total_area : ℕ := total_patches * patch_area

-- Theorem statement
theorem quilt_width :
  (total_area / quilt_length) = 20 :=
by
  sorry

end NUMINAMATH_GPT_quilt_width_l707_70720


namespace NUMINAMATH_GPT_find_r_l707_70700

theorem find_r (a b m p r : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a * b = 4)
  (h4 : ∀ x : ℚ, x^2 - m * x + 4 = (x - a) * (x - b)) :
  (a - 1 / b) * (b - 1 / a) = 9 / 4 := by
  sorry

end NUMINAMATH_GPT_find_r_l707_70700


namespace NUMINAMATH_GPT_excursion_min_parents_l707_70763

theorem excursion_min_parents 
  (students : ℕ) 
  (car_capacity : ℕ)
  (h_students : students = 30)
  (h_car_capacity : car_capacity = 5) 
  : ∃ (parents_needed : ℕ), parents_needed = 8 := 
by
  sorry -- proof goes here

end NUMINAMATH_GPT_excursion_min_parents_l707_70763


namespace NUMINAMATH_GPT_average_of_three_l707_70716

-- Definitions of Conditions
variables (A B C : ℝ)
variables (h1 : A + B = 147) (h2 : B + C = 123) (h3 : A + C = 132)

-- The proof problem stating the goal
theorem average_of_three (A B C : ℝ) 
    (h1 : A + B = 147) (h2 : B + C = 123) (h3 : A + C = 132) : 
    (A + B + C) / 3 = 67 := 
sorry

end NUMINAMATH_GPT_average_of_three_l707_70716


namespace NUMINAMATH_GPT_determine_m_minus_n_l707_70783

-- Definitions of the conditions
variables {m n : ℝ}

-- The proof statement
theorem determine_m_minus_n (h_eq : ∀ x y : ℝ, x^(4 - 3 * |m|) + y^(3 * |n|) = 2009 → x + y = 2009)
  (h_prod_lt_zero : m * n < 0)
  (h_sum : 0 < m + n ∧ m + n ≤ 3) : m - n = 4/3 := 
sorry

end NUMINAMATH_GPT_determine_m_minus_n_l707_70783


namespace NUMINAMATH_GPT_train_length_l707_70754

def relative_speed (v_fast v_slow : ℕ) : ℚ :=
  v_fast - v_slow

def convert_speed (speed : ℚ) : ℚ :=
  (speed * 1000) / 3600

def covered_distance (speed : ℚ) (time_seconds : ℚ) : ℚ :=
  speed * time_seconds

theorem train_length (L : ℚ) (v_fast v_slow : ℕ) (time_seconds : ℚ)
    (hf : v_fast = 42) (hs : v_slow = 36) (ht : time_seconds = 36)
    (hc : relative_speed v_fast v_slow * 1000 / 3600 * time_seconds = 2 * L) :
    L = 30 := by
  sorry

end NUMINAMATH_GPT_train_length_l707_70754


namespace NUMINAMATH_GPT_find_alpha_l707_70713

theorem find_alpha
  (α : Real)
  (h1 : α > 0)
  (h2 : α < π)
  (h3 : 1 / Real.sin α + 1 / Real.cos α = 2) :
  α = π + 1 / 2 * Real.arcsin ((1 - Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_GPT_find_alpha_l707_70713


namespace NUMINAMATH_GPT_carlos_laundry_l707_70722

theorem carlos_laundry (n : ℕ) 
  (h1 : 45 * n + 75 = 165) : n = 2 :=
by
  sorry

end NUMINAMATH_GPT_carlos_laundry_l707_70722


namespace NUMINAMATH_GPT_range_of_a_l707_70766

theorem range_of_a (a : ℝ) (x : ℤ) (h1 : ∀ x, x > 0 → ⌊(x + a) / 3⌋ = 2) : a < 8 :=
sorry

end NUMINAMATH_GPT_range_of_a_l707_70766


namespace NUMINAMATH_GPT_wall_width_l707_70778

theorem wall_width (w h l V : ℝ) (h_eq : h = 4 * w) (l_eq : l = 3 * h) (V_eq : V = w * h * l) (v_val : V = 10368) : w = 6 :=
  sorry

end NUMINAMATH_GPT_wall_width_l707_70778


namespace NUMINAMATH_GPT_sin_3pi_div_2_eq_neg_1_l707_70796

theorem sin_3pi_div_2_eq_neg_1 : Real.sin (3 * Real.pi / 2) = -1 := by
  sorry

end NUMINAMATH_GPT_sin_3pi_div_2_eq_neg_1_l707_70796


namespace NUMINAMATH_GPT_min_formula_l707_70748

theorem min_formula (a b : ℝ) : 
  min a b = (a + b - Real.sqrt ((a - b) ^ 2)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_formula_l707_70748


namespace NUMINAMATH_GPT_percentage_who_do_not_have_job_of_choice_have_university_diploma_l707_70775

theorem percentage_who_do_not_have_job_of_choice_have_university_diploma :
  ∀ (total_population university_diploma job_of_choice no_diploma_job_of_choice : ℝ),
    total_population = 100 →
    job_of_choice = 40 →
    no_diploma_job_of_choice = 10 →
    university_diploma = 48 →
    ((university_diploma - (job_of_choice - no_diploma_job_of_choice)) / (total_population - job_of_choice)) * 100 = 30 :=
by
  intros total_population university_diploma job_of_choice no_diploma_job_of_choice h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_percentage_who_do_not_have_job_of_choice_have_university_diploma_l707_70775


namespace NUMINAMATH_GPT_tangent_line_eqn_l707_70760

theorem tangent_line_eqn :
  ∃ k : ℝ, 
  x^2 + y^2 - 4*x + 3 = 0 → 
  (∃ x y : ℝ, (x-2)^2 + y^2 = 1 ∧ x > 2 ∧ y < 0 ∧ y = k*x) → 
  k = - (Real.sqrt 3) / 3 := 
by
  sorry

end NUMINAMATH_GPT_tangent_line_eqn_l707_70760


namespace NUMINAMATH_GPT_stream_speed_l707_70787

theorem stream_speed (C S : ℝ) 
    (h1 : C - S = 8) 
    (h2 : C + S = 12) : 
    S = 2 :=
sorry

end NUMINAMATH_GPT_stream_speed_l707_70787


namespace NUMINAMATH_GPT_problem_statement_l707_70779

def op (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem problem_statement : ((op 7 4) - 12) * 5 = 105 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l707_70779


namespace NUMINAMATH_GPT_negation_of_existence_l707_70793

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ ∀ x : ℝ, x^2 + 2 * a * x + a > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_l707_70793


namespace NUMINAMATH_GPT_total_votes_l707_70743

theorem total_votes (votes_veggies : ℕ) (votes_meat : ℕ) (H1 : votes_veggies = 337) (H2 : votes_meat = 335) : votes_veggies + votes_meat = 672 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_l707_70743


namespace NUMINAMATH_GPT_num_ways_128_as_sum_of_four_positive_perfect_squares_l707_70761

noncomputable def is_positive_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, 0 < m ∧ m * m = n

noncomputable def four_positive_perfect_squares_sum (n : ℕ) : Prop :=
  ∃ a b c d : ℕ,
    is_positive_perfect_square a ∧
    is_positive_perfect_square b ∧
    is_positive_perfect_square c ∧
    is_positive_perfect_square d ∧
    a + b + c + d = n

theorem num_ways_128_as_sum_of_four_positive_perfect_squares :
  (∃! (a b c d : ℕ), four_positive_perfect_squares_sum 128) :=
sorry

end NUMINAMATH_GPT_num_ways_128_as_sum_of_four_positive_perfect_squares_l707_70761


namespace NUMINAMATH_GPT_coin_overlap_black_region_cd_sum_l707_70746

noncomputable def black_region_probability : ℝ := 
  let square_side := 10
  let triangle_leg := 3
  let diamond_side := 3 * Real.sqrt 2
  let coin_diameter := 2
  let coin_radius := coin_diameter / 2
  let reduced_square_side := square_side - coin_diameter
  let reduced_square_area := reduced_square_side * reduced_square_side
  let triangle_area := 4 * ((triangle_leg * triangle_leg) / 2)
  let extra_triangle_area := 4 * (Real.pi / 4 + 3)
  let diamond_area := (diamond_side * diamond_side) / 2
  let extra_diamond_area := Real.pi + 12 * Real.sqrt 2
  let total_black_area := triangle_area + extra_triangle_area + diamond_area + extra_diamond_area

  total_black_area / reduced_square_area

theorem coin_overlap_black_region: 
  black_region_probability = (1 / 64) * (30 + 12 * Real.sqrt 2 + Real.pi) := 
sorry

theorem cd_sum: 
  let c := 30
  let d := 12
  c + d = 42 := 
by
  trivial

end NUMINAMATH_GPT_coin_overlap_black_region_cd_sum_l707_70746


namespace NUMINAMATH_GPT_number_of_solutions_l707_70712

theorem number_of_solutions :
  ∃ n : ℕ,  (1 + ⌊(102 * n : ℚ) / 103⌋ = ⌈(101 * n : ℚ) / 102⌉) ↔ (n < 10506) := 
sorry

end NUMINAMATH_GPT_number_of_solutions_l707_70712


namespace NUMINAMATH_GPT_Danny_bottle_caps_l707_70744

theorem Danny_bottle_caps (r w c : ℕ) (h1 : r = 11) (h2 : c = r + 1) : c = 12 := by
  sorry

end NUMINAMATH_GPT_Danny_bottle_caps_l707_70744


namespace NUMINAMATH_GPT_geometric_sequence_relation_l707_70794

theorem geometric_sequence_relation (a b c : ℝ) (r : ℝ)
  (h1 : -2 * r = a)
  (h2 : a * r = b)
  (h3 : b * r = c)
  (h4 : c * r = -8) :
  b = -4 ∧ a * c = 16 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_relation_l707_70794


namespace NUMINAMATH_GPT_b_plus_c_neg_seven_l707_70733

theorem b_plus_c_neg_seven {A B : Set ℝ} (hA : A = {x : ℝ | x > 3 ∨ x < -1}) (hB : B = {x : ℝ | -1 ≤ x ∧ x ≤ 4})
  (h_union : A ∪ B = Set.univ) (h_inter : A ∩ B = {x : ℝ | 3 < x ∧ x ≤ 4}) :
  ∃ b c : ℝ, (∀ x, x^2 + b * x + c ≤ 0 ↔ x ∈ B) ∧ b + c = -7 :=
by
  sorry

end NUMINAMATH_GPT_b_plus_c_neg_seven_l707_70733


namespace NUMINAMATH_GPT_square_possible_n12_square_possible_n15_l707_70707

-- Define the nature of the problem with condition n = 12
def min_sticks_to_break_for_square_n12 : ℕ :=
  let n := 12
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Define the nature of the problem with condition n = 15
def min_sticks_to_break_for_square_n15 : ℕ :=
  let n := 15
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Statement of the problems in Lean 4 language
theorem square_possible_n12 : min_sticks_to_break_for_square_n12 = 2 := by
  sorry

theorem square_possible_n15 : min_sticks_to_break_for_square_n15 = 0 := by
  sorry

end NUMINAMATH_GPT_square_possible_n12_square_possible_n15_l707_70707


namespace NUMINAMATH_GPT_no_integer_roots_l707_70791

theorem no_integer_roots (x : ℤ) : ¬ (x^2 + 2^2018 * x + 2^2019 = 0) :=
sorry

end NUMINAMATH_GPT_no_integer_roots_l707_70791


namespace NUMINAMATH_GPT_PetesOriginalNumber_l707_70767

-- Define the context and problem
theorem PetesOriginalNumber (x : ℤ) (h : 3 * (2 * x + 12) = 90) : x = 9 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_PetesOriginalNumber_l707_70767


namespace NUMINAMATH_GPT_sqrt_14_plus_2_range_l707_70773

theorem sqrt_14_plus_2_range :
  5 < Real.sqrt 14 + 2 ∧ Real.sqrt 14 + 2 < 6 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_14_plus_2_range_l707_70773


namespace NUMINAMATH_GPT_opposite_neg_half_l707_70739

theorem opposite_neg_half : -(-1/2) = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_opposite_neg_half_l707_70739


namespace NUMINAMATH_GPT_problem1_problem2_l707_70703

theorem problem1 : 6 + (-8) - (-5) = 3 := sorry

theorem problem2 : 18 / (-3) + (-2) * (-4) = 2 := sorry

end NUMINAMATH_GPT_problem1_problem2_l707_70703


namespace NUMINAMATH_GPT_solve_for_x_l707_70776

theorem solve_for_x (x : ℝ) (h : (x / 5) / 3 = 9 / (x / 3)) : x = 15 * Real.sqrt 1.8 ∨ x = -15 * Real.sqrt 1.8 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l707_70776


namespace NUMINAMATH_GPT_tan_beta_minus_2alpha_l707_70727

open Real

-- Given definitions
def condition1 (α : ℝ) : Prop :=
  (sin α * cos α) / (1 - cos (2 * α)) = 1 / 4

def condition2 (α β : ℝ) : Prop :=
  tan (α - β) = 2

-- Proof problem statement
theorem tan_beta_minus_2alpha (α β : ℝ) (h1 : condition1 α) (h2 : condition2 α β) :
  tan (β - 2 * α) = 4 / 3 :=
sorry

end NUMINAMATH_GPT_tan_beta_minus_2alpha_l707_70727


namespace NUMINAMATH_GPT_find_x_l707_70757

theorem find_x (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x^2 / y = 3) (h2 : y^2 / z = 4) (h3 : z^2 / x = 5) : 
  x = (36 * Real.sqrt 5)^(4/11) := 
sorry

end NUMINAMATH_GPT_find_x_l707_70757


namespace NUMINAMATH_GPT_equivalent_statements_l707_70758

variable (P Q : Prop)

theorem equivalent_statements :
  (P → Q) ↔ (P → Q) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q) := by
  sorry

end NUMINAMATH_GPT_equivalent_statements_l707_70758


namespace NUMINAMATH_GPT_count_integers_with_same_remainder_l707_70737

theorem count_integers_with_same_remainder (n : ℤ) : 
  (150 < n ∧ n < 250) ∧ 
  (∃ r : ℤ, 0 ≤ r ∧ r ≤ 6 ∧ ∃ a b : ℤ, n = 7 * a + r ∧ n = 9 * b + r) ↔ n = 7 :=
sorry

end NUMINAMATH_GPT_count_integers_with_same_remainder_l707_70737


namespace NUMINAMATH_GPT_contrapositive_mul_non_zero_l707_70701

variables (a b : ℝ)

theorem contrapositive_mul_non_zero (h : a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) :
  (a = 0 ∨ b = 0) → a * b = 0 :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_mul_non_zero_l707_70701


namespace NUMINAMATH_GPT_tangent_line_at_x1_f_nonnegative_iff_l707_70768

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (x-1) * Real.log x - m * (x+1)

noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := Real.log x + (x-1) / x - m

theorem tangent_line_at_x1 (m : ℝ) (h : m = 1) :
  ∀ x y : ℝ, f x 1 = y → (x = 1) → x + y + 1 = 0 :=
sorry

theorem f_nonnegative_iff (m : ℝ) :
  (∀ x : ℝ, 0 < x → f x m ≥ 0) ↔ m ≤ 0 :=
sorry

end NUMINAMATH_GPT_tangent_line_at_x1_f_nonnegative_iff_l707_70768


namespace NUMINAMATH_GPT_new_mean_correct_l707_70715

-- Define the original condition data
def initial_mean : ℝ := 42
def total_numbers : ℕ := 60
def discard1 : ℝ := 50
def discard2 : ℝ := 60
def increment : ℝ := 2

-- A function representing the new arithmetic mean
noncomputable def new_arithmetic_mean : ℝ :=
  let initial_sum := initial_mean * total_numbers
  let sum_after_discard := initial_sum - (discard1 + discard2)
  let sum_after_increment := sum_after_discard + (increment * (total_numbers - 2))
  sum_after_increment / (total_numbers - 2)

-- The theorem statement
theorem new_mean_correct : new_arithmetic_mean = 43.55 :=
by 
  sorry

end NUMINAMATH_GPT_new_mean_correct_l707_70715


namespace NUMINAMATH_GPT_find_abc_pairs_l707_70750

theorem find_abc_pairs :
  ∀ (a b c : ℕ), 1 < a ∧ a < b ∧ b < c ∧ (a-1)*(b-1)*(c-1) ∣ a*b*c - 1 → 
  (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_abc_pairs_l707_70750


namespace NUMINAMATH_GPT_infinite_solutions_2n_3n_square_n_multiple_of_40_infinite_solutions_general_l707_70781

open Nat

theorem infinite_solutions_2n_3n_square :
  ∃ᶠ n : ℤ in at_top, ∃ a b : ℤ, 2 * n + 1 = a^2 ∧ 3 * n + 1 = b^2 :=
sorry

theorem n_multiple_of_40 :
  ∀ n : ℤ, (∃ a b : ℤ, 2 * n + 1 = a^2 ∧ 3 * n + 1 = b^2) → (40 ∣ n) :=
sorry

theorem infinite_solutions_general (m : ℕ) (hm : 0 < m) :
  ∃ᶠ n : ℤ in at_top, ∃ a b : ℤ, m * n + 1 = a^2 ∧ (m + 1) * n + 1 = b^2 :=
sorry

end NUMINAMATH_GPT_infinite_solutions_2n_3n_square_n_multiple_of_40_infinite_solutions_general_l707_70781


namespace NUMINAMATH_GPT_mathematically_equivalent_proof_l707_70730

noncomputable def proof_problem (a : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ a^x = 2 ∧ a^y = 3 → a^(x - 2 * y) = 2 / 9

theorem mathematically_equivalent_proof (a : ℝ) (x y : ℝ) :
  proof_problem a x y :=
by
  sorry  -- Proof steps will go here

end NUMINAMATH_GPT_mathematically_equivalent_proof_l707_70730


namespace NUMINAMATH_GPT_quadratic_rewriting_l707_70792

theorem quadratic_rewriting (b n : ℝ) (h₁ : 0 < n)
  (h₂ : ∀ x : ℝ, x^2 + b*x + 72 = (x + n)^2 + 20) :
  b = 4 * Real.sqrt 13 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_rewriting_l707_70792


namespace NUMINAMATH_GPT_Sam_has_4_French_Bulldogs_l707_70789

variable (G F : ℕ)

theorem Sam_has_4_French_Bulldogs
  (h1 : G = 3)
  (h2 : 3 * G + 2 * F = 17) :
  F = 4 :=
sorry

end NUMINAMATH_GPT_Sam_has_4_French_Bulldogs_l707_70789


namespace NUMINAMATH_GPT_braids_each_dancer_l707_70751

-- Define the conditions
def num_dancers := 8
def time_per_braid := 30 -- seconds per braid
def total_time := 20 * 60 -- convert 20 minutes into seconds

-- Define the total number of braids Jill makes
def total_braids := total_time / time_per_braid

-- Define the number of braids per dancer
def braids_per_dancer := total_braids / num_dancers

-- Theorem: Prove that each dancer has 5 braids
theorem braids_each_dancer : braids_per_dancer = 5 := 
by sorry

end NUMINAMATH_GPT_braids_each_dancer_l707_70751


namespace NUMINAMATH_GPT_find_C_and_D_l707_70711

noncomputable def C : ℚ := 51 / 10
noncomputable def D : ℚ := 29 / 10

theorem find_C_and_D (x : ℚ) (h1 : x^2 - 4*x - 21 = (x - 7)*(x + 3))
  (h2 : (8*x - 5) / ((x - 7)*(x + 3)) = C / (x - 7) + D / (x + 3)) :
  C = 51 / 10 ∧ D = 29 / 10 :=
by
  sorry

end NUMINAMATH_GPT_find_C_and_D_l707_70711


namespace NUMINAMATH_GPT_am_gm_inequality_l707_70774

-- Definitions of the variables and hypotheses
variables {a b : ℝ}

-- The theorem statement
theorem am_gm_inequality (h : a * b > 0) : a / b + b / a ≥ 2 :=
sorry

end NUMINAMATH_GPT_am_gm_inequality_l707_70774


namespace NUMINAMATH_GPT_new_mean_rent_l707_70704

theorem new_mean_rent (avg_rent : ℕ) (num_friends : ℕ) (rent_increase_pct : ℕ) (initial_rent : ℕ) :
  avg_rent = 800 →
  num_friends = 4 →
  rent_increase_pct = 25 →
  initial_rent = 800 →
  (avg_rent * num_friends + initial_rent * rent_increase_pct / 100) / num_friends = 850 :=
by
  intros h_avg h_num h_pct h_init
  sorry

end NUMINAMATH_GPT_new_mean_rent_l707_70704


namespace NUMINAMATH_GPT_dishwasher_manager_wage_ratio_l707_70786

theorem dishwasher_manager_wage_ratio
  (chef_wage dishwasher_wage manager_wage : ℝ)
  (h1 : chef_wage = 1.22 * dishwasher_wage)
  (h2 : dishwasher_wage = r * manager_wage)
  (h3 : manager_wage = 8.50)
  (h4 : chef_wage = manager_wage - 3.315) :
  r = 0.5 :=
sorry

end NUMINAMATH_GPT_dishwasher_manager_wage_ratio_l707_70786


namespace NUMINAMATH_GPT_primes_infinite_l707_70705

theorem primes_infinite : ∀ (S : Set ℕ), (∀ p, p ∈ S → Nat.Prime p) → (∃ a, a ∉ S ∧ Nat.Prime a) :=
by
  sorry

end NUMINAMATH_GPT_primes_infinite_l707_70705


namespace NUMINAMATH_GPT_min_value_expression_l707_70740

theorem min_value_expression {x y z w : ℝ} 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) 
  (hw : 0 ≤ w ∧ w ≤ 1) : 
  ∃ m, m = 2 ∧ ∀ x y z w, (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧ (0 ≤ w ∧ w ≤ 1) →
  m ≤ (1 / ((1 - x) * (1 - y) * (1 - z) * (1 - w)) + 1 / ((1 + x) * (1 + y) * (1 + z) * (1 + w))) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l707_70740


namespace NUMINAMATH_GPT_line_intercepts_l707_70747

theorem line_intercepts :
  (exists a b : ℝ, (forall x y : ℝ, x - 2*y - 2 = 0 ↔ (x = 2 ∨ y = -1)) ∧ a = 2 ∧ b = -1) :=
by
  sorry

end NUMINAMATH_GPT_line_intercepts_l707_70747


namespace NUMINAMATH_GPT_range_of_m_l707_70765

theorem range_of_m (x m : ℝ) (h1 : x + 3 = 3 * x - m) (h2 : x ≥ 0) : m ≥ -3 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l707_70765


namespace NUMINAMATH_GPT_rearrange_marked_cells_below_diagonal_l707_70771

theorem rearrange_marked_cells_below_diagonal (n : ℕ) (marked_cells : Finset (Fin n × Fin n)) :
  marked_cells.card = n - 1 →
  ∃ row_permutation col_permutation : Equiv (Fin n) (Fin n), ∀ (i j : Fin n),
    (row_permutation i, col_permutation j) ∈ marked_cells → j < i :=
by
  sorry

end NUMINAMATH_GPT_rearrange_marked_cells_below_diagonal_l707_70771


namespace NUMINAMATH_GPT_set_contains_difference_of_elements_l707_70785

variable {A : Set Int}

axiom cond1 (a : Int) (ha : a ∈ A) : 2 * a ∈ A
axiom cond2 (a b : Int) (ha : a ∈ A) (hb : b ∈ A) : a + b ∈ A

theorem set_contains_difference_of_elements 
  (a b : Int) (ha : a ∈ A) (hb : b ∈ A) : a - b ∈ A := by
  sorry

end NUMINAMATH_GPT_set_contains_difference_of_elements_l707_70785


namespace NUMINAMATH_GPT_tan_sum_sin_cos_conditions_l707_70756

theorem tan_sum_sin_cos_conditions {x y : ℝ} 
  (h1 : Real.sin x + Real.sin y = 1 / 2) 
  (h2 : Real.cos x + Real.cos y = Real.sqrt 3 / 2) :
  Real.tan x + Real.tan y = -Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_tan_sum_sin_cos_conditions_l707_70756


namespace NUMINAMATH_GPT_charge_per_person_on_second_day_l707_70788

noncomputable def charge_second_day (k : ℕ) (x : ℝ) :=
  let total_revenue := 30 * k + 5 * k * x + 32.5 * k
  let total_visitors := 20 * k
  (total_revenue / total_visitors = 5)

theorem charge_per_person_on_second_day
  (k : ℕ) (hx : charge_second_day k 7.5) :
  7.5 = 7.5 :=
sorry

end NUMINAMATH_GPT_charge_per_person_on_second_day_l707_70788


namespace NUMINAMATH_GPT_remainder_when_x_squared_divided_by_20_l707_70759

theorem remainder_when_x_squared_divided_by_20
  (x : ℤ)
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 2 * x ≡ 8 [ZMOD 20]) :
  x^2 ≡ 16 [ZMOD 20] :=
sorry

end NUMINAMATH_GPT_remainder_when_x_squared_divided_by_20_l707_70759


namespace NUMINAMATH_GPT_min_soldiers_in_square_formations_l707_70725

theorem min_soldiers_in_square_formations : ∃ (a : ℕ), 
  ∃ (k : ℕ), 
    (a = k^2 ∧ 
    11 * a + 1 = (m : ℕ) ^ 2) ∧ 
    (∀ (b : ℕ), 
      (∃ (j : ℕ), b = j^2 ∧ 11 * b + 1 = (n : ℕ) ^ 2) → a ≤ b) ∧ 
    a = 9 := 
sorry

end NUMINAMATH_GPT_min_soldiers_in_square_formations_l707_70725


namespace NUMINAMATH_GPT_min_people_same_score_l707_70702

theorem min_people_same_score (participants : ℕ) (nA nB : ℕ) (pointsA pointsB : ℕ) (scores : Finset ℕ) :
  participants = 400 →
  nA = 8 →
  nB = 6 →
  pointsA = 4 →
  pointsB = 7 →
  scores.card = (nA + 1) * (nB + 1) - 6 →
  participants / scores.card < 8 :=
by
  intros h_participants h_nA h_nB h_pointsA h_pointsB h_scores_card
  sorry

end NUMINAMATH_GPT_min_people_same_score_l707_70702


namespace NUMINAMATH_GPT_DE_value_l707_70728

theorem DE_value {AG GF FC HJ DE : ℝ} (h1 : AG = 2) (h2 : GF = 13) 
  (h3 : FC = 1) (h4 : HJ = 7) : DE = 2 * Real.sqrt 22 :=
sorry

end NUMINAMATH_GPT_DE_value_l707_70728


namespace NUMINAMATH_GPT_reassemble_black_rectangles_into_1x2_rectangle_l707_70721

theorem reassemble_black_rectangles_into_1x2_rectangle
  (x y : ℝ)
  (h1 : 0 < x ∧ x < 2)
  (h2 : 0 < y ∧ y < 2)
  (black_white_equal : 2*x*y - 2*x - 2*y + 2 = 0) :
  (x = 1 ∨ y = 1) →
  ∃ (z : ℝ), z = 1 :=
by
  sorry

end NUMINAMATH_GPT_reassemble_black_rectangles_into_1x2_rectangle_l707_70721


namespace NUMINAMATH_GPT_min_c_value_l707_70749

theorem min_c_value (c : ℝ) : (-c^2 + 9 * c - 14 >= 0) → (c >= 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_min_c_value_l707_70749


namespace NUMINAMATH_GPT_page_numbers_sum_l707_70782

theorem page_numbers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 136080) : n + (n + 1) + (n + 2) = 144 :=
by
  sorry

end NUMINAMATH_GPT_page_numbers_sum_l707_70782


namespace NUMINAMATH_GPT_probability_even_product_l707_70753

-- Define spinner A and spinner C
def SpinnerA : List ℕ := [1, 2, 3, 4]
def SpinnerC : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define even and odd number sets for Spinner A and Spinner C
def evenNumbersA : List ℕ := [2, 4]
def oddNumbersA : List ℕ := [1, 3]

def evenNumbersC : List ℕ := [2, 4, 6]
def oddNumbersC : List ℕ := [1, 3, 5]

-- Define a function to check if a product is even
def isEven (n : ℕ) : Bool := n % 2 == 0

-- Probability calculation
def evenProductProbability : ℚ :=
  let totalOutcomes := (SpinnerA.length * SpinnerC.length)
  let evenA_outcomes := (evenNumbersA.length * SpinnerC.length)
  let oddA_evenC_outcomes := (oddNumbersA.length * evenNumbersC.length)
  (evenA_outcomes + oddA_evenC_outcomes) / totalOutcomes

theorem probability_even_product :
  evenProductProbability = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_even_product_l707_70753


namespace NUMINAMATH_GPT_distinct_x_intercepts_l707_70723

-- Given conditions
def polynomial (x : ℝ) : ℝ := (x - 4) * (x^2 + 4 * x + 13)

-- Statement of the problem as a Lean theorem
theorem distinct_x_intercepts : 
  (∃ (x : ℝ), polynomial x = 0 ∧ 
    ∀ (y : ℝ), y ≠ x → polynomial y = 0 → False) :=
  sorry

end NUMINAMATH_GPT_distinct_x_intercepts_l707_70723


namespace NUMINAMATH_GPT_findNumberOfItemsSoldByStoreA_l707_70795

variable (P x : ℝ) -- P is the price of the product, x is the number of items Store A sells

-- Total sales amount for Store A (in yuan)
def totalSalesA := P * x = 7200

-- Total sales amount for Store B (in yuan)
def totalSalesB := 0.8 * P * (x + 15) = 7200

-- Same price in both stores
def samePriceInBothStores := (P > 0)

-- Proof Problem Statement
theorem findNumberOfItemsSoldByStoreA (storeASellsAtListedPrice : totalSalesA P x)
  (storeBSells15MoreItemsAndAt80PercentPrice : totalSalesB P x)
  (priceIsPositive : samePriceInBothStores P) :
  x = 60 :=
sorry

end NUMINAMATH_GPT_findNumberOfItemsSoldByStoreA_l707_70795


namespace NUMINAMATH_GPT_recurrence_relation_solution_l707_70764

theorem recurrence_relation_solution (a : ℕ → ℕ) 
  (h_rec : ∀ n ≥ 2, a n = 4 * a (n - 1) - 3 * a (n - 2))
  (h0 : a 0 = 3)
  (h1 : a 1 = 5) :
  ∀ n, a n = 3^n + 2 :=
by
  sorry

end NUMINAMATH_GPT_recurrence_relation_solution_l707_70764


namespace NUMINAMATH_GPT_sum_of_other_endpoint_l707_70731

theorem sum_of_other_endpoint (x y : ℝ) (h1 : (6 + x) / 2 = 3) (h2 : (-2 + y) / 2 = 5) : x + y = 12 := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_other_endpoint_l707_70731


namespace NUMINAMATH_GPT_seq_bn_arithmetic_seq_an_formula_sum_an_terms_l707_70798

-- (1) Prove that the sequence {b_n} is an arithmetic sequence
theorem seq_bn_arithmetic (a : ℕ → ℕ) (b : ℕ → ℤ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = 2 * a n + 2^n)
  (h3 : ∀ n, b n = a n / 2^(n - 1)) :
  ∀ n, b (n + 1) - b n = 1 := by
  sorry

-- (2) Find the general formula for the sequence {a_n}
theorem seq_an_formula (a : ℕ → ℕ) (b : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = 2 * a n + 2^n)
  (h3 : ∀ n, b n = a n / 2^(n - 1)) :
  ∀ n, a n = n * 2^(n - 1) := by
  sorry

-- (3) Find the sum of the first n terms of the sequence {a_n}
theorem sum_an_terms (a : ℕ → ℕ) (S : ℕ → ℤ) (h1 : ∀ n, a n = n * 2^(n - 1)) :
  ∀ n, S n = (n - 1) * 2^n + 1 := by
  sorry

end NUMINAMATH_GPT_seq_bn_arithmetic_seq_an_formula_sum_an_terms_l707_70798


namespace NUMINAMATH_GPT_vertex_farthest_from_origin_l707_70741

theorem vertex_farthest_from_origin (center : ℝ × ℝ) (area : ℝ) (top_side_horizontal : Prop) (dilation_center : ℝ × ℝ) (scale_factor : ℝ) :
  center = (10, -5) ∧ area = 16 ∧ top_side_horizontal ∧ dilation_center = (0, 0) ∧ scale_factor = 3 →
  ∃ (vertex_farthest : ℝ × ℝ), vertex_farthest = (36, -21) :=
by
  sorry

end NUMINAMATH_GPT_vertex_farthest_from_origin_l707_70741


namespace NUMINAMATH_GPT_dogs_running_l707_70784

theorem dogs_running (total_dogs playing_with_toys barking not_doing_anything running : ℕ)
  (h1 : total_dogs = 88)
  (h2 : playing_with_toys = total_dogs / 2)
  (h3 : barking = total_dogs / 4)
  (h4 : not_doing_anything = 10)
  (h5 : running = total_dogs - playing_with_toys - barking - not_doing_anything) :
  running = 12 :=
sorry

end NUMINAMATH_GPT_dogs_running_l707_70784


namespace NUMINAMATH_GPT_evaluate_64_pow_3_div_2_l707_70734

theorem evaluate_64_pow_3_div_2 : (64 : ℝ)^(3/2) = 512 := by
  -- given 64 = 2^6
  have h : (64 : ℝ) = 2^6 := by norm_num
  -- use this substitution and properties of exponents
  rw [h, ←pow_mul]
  norm_num
  sorry -- completing the proof, not needed based on the guidelines

end NUMINAMATH_GPT_evaluate_64_pow_3_div_2_l707_70734


namespace NUMINAMATH_GPT_john_ate_10_chips_l707_70724

variable (c p : ℕ)

/-- Given the total calories from potato chips and the calories increment of cheezits,
prove the number of potato chips John ate. -/
theorem john_ate_10_chips (h₀ : p * c = 60)
  (h₁ : ∃ c_cheezit, (c_cheezit = (4 / 3 : ℝ) * c))
  (h₂ : ∀ c_cheezit, p * c + 6 * c_cheezit = 108) :
  p = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_john_ate_10_chips_l707_70724


namespace NUMINAMATH_GPT_fraction_calculation_l707_70709

theorem fraction_calculation : 
  (1 / 4 + 1 / 6 - 1 / 2) / (-1 / 24) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_calculation_l707_70709


namespace NUMINAMATH_GPT_largest_multiple_of_15_less_than_500_l707_70769

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end NUMINAMATH_GPT_largest_multiple_of_15_less_than_500_l707_70769


namespace NUMINAMATH_GPT_calculate_force_l707_70777

noncomputable def force_on_dam (ρ g a b h : ℝ) : ℝ :=
  ρ * g * h^2 * (b / 2 - (b - a) / 3)

theorem calculate_force : force_on_dam 1000 10 4.8 7.2 3.0 = 252000 := 
  by 
  sorry

end NUMINAMATH_GPT_calculate_force_l707_70777


namespace NUMINAMATH_GPT_line_passes_through_vertex_of_parabola_l707_70732

theorem line_passes_through_vertex_of_parabola : 
  ∃ (a : ℝ), (∀ x y : ℝ, y = 2 * x + a ↔ y = x^2 + a^2) ↔ a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_GPT_line_passes_through_vertex_of_parabola_l707_70732


namespace NUMINAMATH_GPT_minimum_distinct_numbers_l707_70755

theorem minimum_distinct_numbers (a : ℕ → ℕ) (h_pos : ∀ i, 1 ≤ i → a i > 0)
  (h_distinct_ratios : ∀ i j : ℕ, 1 ≤ i ∧ i < 2006 ∧ 1 ≤ j ∧ j < 2006 ∧ i ≠ j → a i / a (i + 1) ≠ a j / a (j + 1)) :
  ∃ (n : ℕ), n = 46 ∧ ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 2006 ∧ 1 ≤ j ∧ j ≤ i ∧ (a i = a j → i = j) :=
sorry

end NUMINAMATH_GPT_minimum_distinct_numbers_l707_70755


namespace NUMINAMATH_GPT_susan_more_cats_than_bob_after_transfer_l707_70717

-- Definitions and conditions
def susan_initial_cats : ℕ := 21
def bob_initial_cats : ℕ := 3
def cats_transferred : ℕ := 4

-- Question statement translated to Lean
theorem susan_more_cats_than_bob_after_transfer :
  (susan_initial_cats - cats_transferred) - bob_initial_cats = 14 :=
by
  sorry

end NUMINAMATH_GPT_susan_more_cats_than_bob_after_transfer_l707_70717


namespace NUMINAMATH_GPT_probability_all_calls_same_probability_two_calls_for_A_l707_70714

theorem probability_all_calls_same (pA pB pC : ℚ) (hA : pA = 1/6) (hB : pB = 1/3) (hC : pC = 1/2) :
  (pA^3 + pB^3 + pC^3) = 1/6 :=
by
  sorry

theorem probability_two_calls_for_A (pA : ℚ) (hA : pA = 1/6) :
  (3 * (pA^2) * (5/6)) = 5/72 :=
by
  sorry

end NUMINAMATH_GPT_probability_all_calls_same_probability_two_calls_for_A_l707_70714


namespace NUMINAMATH_GPT_perfect_square_polynomial_l707_70710

theorem perfect_square_polynomial (m : ℝ) :
  (∃ a b : ℝ, (a * x + b)^2 = m - 10 * x + x^2) → m = 25 :=
sorry

end NUMINAMATH_GPT_perfect_square_polynomial_l707_70710


namespace NUMINAMATH_GPT_determinant_of_A_l707_70752

section
  open Matrix

  -- Define the given matrix
  def A : Matrix (Fin 3) (Fin 3) ℤ :=
    ![ ![0, 2, -4], ![6, -1, 3], ![2, -3, 5] ]

  -- State the theorem for the determinant
  theorem determinant_of_A : det A = 16 :=
  sorry
end

end NUMINAMATH_GPT_determinant_of_A_l707_70752


namespace NUMINAMATH_GPT_value_of_b_l707_70762

theorem value_of_b (a b : ℝ) (h1 : 4 * a^2 + 1 = 1) (h2 : b - a = 3) : b = 3 :=
sorry

end NUMINAMATH_GPT_value_of_b_l707_70762


namespace NUMINAMATH_GPT_ascorbic_acid_weight_l707_70738

def molecular_weight (formula : String) : ℝ :=
  if formula = "C6H8O6" then 176.12 else 0

theorem ascorbic_acid_weight : molecular_weight "C6H8O6" = 176.12 :=
by {
  sorry
}

end NUMINAMATH_GPT_ascorbic_acid_weight_l707_70738


namespace NUMINAMATH_GPT_determine_a_values_l707_70799

theorem determine_a_values (a : ℝ) (A : Set ℝ) (B : Set ℝ)
  (hA : A = { x | abs x = 1 }) 
  (hB : B = { x | a * x = 1 }) 
  (h_superset : A ⊇ B) :
  a = -1 ∨ a = 0 ∨ a = 1 :=
sorry

end NUMINAMATH_GPT_determine_a_values_l707_70799
