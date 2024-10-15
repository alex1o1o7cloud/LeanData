import Mathlib

namespace NUMINAMATH_GPT_fraction_of_population_married_l2006_200618

theorem fraction_of_population_married
  (M W N : ℕ)
  (h1 : (2 / 3 : ℚ) * M = N)
  (h2 : (3 / 5 : ℚ) * W = N)
  : ((2 * N) : ℚ) / (M + W) = 12 / 19 := 
by
  sorry

end NUMINAMATH_GPT_fraction_of_population_married_l2006_200618


namespace NUMINAMATH_GPT_chloe_at_least_85_nickels_l2006_200649

-- Define the given values
def shoe_cost : ℝ := 45.50
def ten_dollars : ℝ := 10.0
def num_ten_dollar_bills : ℕ := 4
def quarter_value : ℝ := 0.25
def num_quarters : ℕ := 5
def nickel_value : ℝ := 0.05

-- Define the statement to be proved
theorem chloe_at_least_85_nickels (n : ℕ) 
  (H1 : shoe_cost = 45.50)
  (H2 : ten_dollars = 10.0)
  (H3 : num_ten_dollar_bills = 4)
  (H4 : quarter_value = 0.25)
  (H5 : num_quarters = 5)
  (H6 : nickel_value = 0.05) :
  4 * ten_dollars + 5 * quarter_value + n * nickel_value >= shoe_cost → n >= 85 :=
by {
  sorry
}

end NUMINAMATH_GPT_chloe_at_least_85_nickels_l2006_200649


namespace NUMINAMATH_GPT_sum_of_possible_values_l2006_200604

theorem sum_of_possible_values (x : ℝ) :
  (x + 3) * (x - 4) = 20 →
  ∃ a b, (a ≠ b) ∧ 
         ((x = a) ∨ (x = b)) ∧ 
         (x^2 - x - 32 = 0) ∧ 
         (a + b = 1) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_l2006_200604


namespace NUMINAMATH_GPT_train_pass_time_eq_4_seconds_l2006_200645

-- Define the length of the train in meters
def train_length : ℕ := 40

-- Define the speed of the train in km/h
def train_speed_kmph : ℕ := 36

-- Conversion factor: 1 kmph = 1000 meters / 3600 seconds
def conversion_factor : ℚ := 1000 / 3600

-- Convert the train's speed from km/h to m/s
def train_speed_mps : ℚ := train_speed_kmph * conversion_factor

-- Calculate the time to pass the telegraph post
def time_to_pass_post : ℚ := train_length / train_speed_mps

-- The goal: prove the actual time is 4 seconds
theorem train_pass_time_eq_4_seconds : time_to_pass_post = 4 := by
  sorry

end NUMINAMATH_GPT_train_pass_time_eq_4_seconds_l2006_200645


namespace NUMINAMATH_GPT_sum_of_squares_divisible_by_7_implies_product_divisible_by_49_l2006_200631

theorem sum_of_squares_divisible_by_7_implies_product_divisible_by_49 (a b : ℕ) 
  (h : (a * a + b * b) % 7 = 0) : (a * b) % 49 = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_divisible_by_7_implies_product_divisible_by_49_l2006_200631


namespace NUMINAMATH_GPT_no_solution_inequality_l2006_200622

theorem no_solution_inequality (a b x : ℝ) (h : |a - b| > 2) : ¬(|x - a| + |x - b| ≤ 2) :=
sorry

end NUMINAMATH_GPT_no_solution_inequality_l2006_200622


namespace NUMINAMATH_GPT_polynomial_perfect_square_l2006_200686

theorem polynomial_perfect_square (k : ℤ) : (∃ b : ℤ, (x + b)^2 = x^2 + 8 * x + k) -> k = 16 := by
  sorry

end NUMINAMATH_GPT_polynomial_perfect_square_l2006_200686


namespace NUMINAMATH_GPT_add_to_make_divisible_by_23_l2006_200660

def least_addend_for_divisibility (n k : ℕ) : ℕ :=
  let remainder := n % k
  k - remainder

theorem add_to_make_divisible_by_23 : least_addend_for_divisibility 1053 23 = 5 :=
by
  sorry

end NUMINAMATH_GPT_add_to_make_divisible_by_23_l2006_200660


namespace NUMINAMATH_GPT_min_value_expression_l2006_200611

theorem min_value_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 4) :
  ∃ c : ℝ, (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x * y * z = 4 → 
  (2 * (x / y) + 3 * (y / z) + 4 * (z / x)) ≥ c) ∧ c = 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l2006_200611


namespace NUMINAMATH_GPT_gina_snake_mice_eaten_in_decade_l2006_200661

-- Define the constants and conditions
def weeks_per_year : ℕ := 52
def years_per_decade : ℕ := 10
def weeks_per_decade : ℕ := years_per_decade * weeks_per_year
def mouse_eating_period : ℕ := 4

-- The problem to prove
theorem gina_snake_mice_eaten_in_decade : (weeks_per_decade / mouse_eating_period) = 130 := 
by
  -- The proof would typically go here, but we skip it
  sorry

end NUMINAMATH_GPT_gina_snake_mice_eaten_in_decade_l2006_200661


namespace NUMINAMATH_GPT_determine_M_l2006_200689

theorem determine_M : ∃ M : ℕ, 36^2 * 75^2 = 30^2 * M^2 ∧ M = 90 := 
by
  sorry

end NUMINAMATH_GPT_determine_M_l2006_200689


namespace NUMINAMATH_GPT_units_digit_base8_l2006_200630

theorem units_digit_base8 (a b : ℕ) (h_a : a = 123) (h_b : b = 57) :
  let product := a * b
  let units_digit := product % 8
  units_digit = 7 := by
  sorry

end NUMINAMATH_GPT_units_digit_base8_l2006_200630


namespace NUMINAMATH_GPT_factory_selection_and_probability_l2006_200602

/-- Total number of factories in districts A, B, and C --/
def factories_A := 18
def factories_B := 27
def factories_C := 18

/-- Total number of factories and sample size --/
def total_factories := factories_A + factories_B + factories_C
def sample_size := 7

/-- Number of factories selected from districts A, B, and C --/
def selected_from_A := factories_A * sample_size / total_factories
def selected_from_B := factories_B * sample_size / total_factories
def selected_from_C := factories_C * sample_size / total_factories

/-- Number of ways to choose 2 factories out of the 7 --/
noncomputable def comb_7_2 := Nat.choose 7 2

/-- Number of favorable outcomes where at least one factory comes from district A --/
noncomputable def favorable_outcomes := 11

/-- Probability that at least one of the 2 factories comes from district A --/
noncomputable def probability := favorable_outcomes / comb_7_2

theorem factory_selection_and_probability :
  selected_from_A = 2 ∧ selected_from_B = 3 ∧ selected_from_C = 2 ∧ probability = 11 / 21 := by
  sorry

end NUMINAMATH_GPT_factory_selection_and_probability_l2006_200602


namespace NUMINAMATH_GPT_apples_per_pie_l2006_200667

-- Definitions of the conditions
def number_of_pies : ℕ := 10
def harvested_apples : ℕ := 50
def to_buy_apples : ℕ := 30
def total_apples_needed : ℕ := harvested_apples + to_buy_apples

-- The theorem to prove
theorem apples_per_pie :
  (total_apples_needed / number_of_pies) = 8 := 
sorry

end NUMINAMATH_GPT_apples_per_pie_l2006_200667


namespace NUMINAMATH_GPT_isosceles_triangle_altitude_l2006_200654

open Real

theorem isosceles_triangle_altitude (DE DF DG EG GF EF : ℝ) (h1 : DE = 5) (h2 : DF = 5) (h3 : EG = 2 * GF)
(h4 : DG = sqrt (DE^2 - GF^2)) (h5 : EF = EG + GF) (h6 : EF = 3 * GF) : EF = 5 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_isosceles_triangle_altitude_l2006_200654


namespace NUMINAMATH_GPT_proof_a_eq_b_pow_n_l2006_200613

theorem proof_a_eq_b_pow_n 
  (a b n : ℕ) 
  (h : ∀ k : ℕ, k ≠ b → b - k ∣ a - k^n) : a = b^n := 
by 
  sorry

end NUMINAMATH_GPT_proof_a_eq_b_pow_n_l2006_200613


namespace NUMINAMATH_GPT_negation_of_existence_l2006_200621

theorem negation_of_existence (p : Prop) : 
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 ≤ x + 2)) ↔ (∀ x : ℝ, x > 0 → x^2 > x + 2) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_l2006_200621


namespace NUMINAMATH_GPT_complete_square_form_l2006_200655

theorem complete_square_form (a b x : ℝ) : 
  ∃ (p : ℝ) (q : ℝ), 
  (p = x ∧ q = 1 ∧ (x^2 + 2*x + 1 = (p + q)^2)) ∧ 
  (¬ ∃ (p q : ℝ), a^2 + 4 = (a + p) * (a + q)) ∧
  (¬ ∃ (p q : ℝ), a^2 + a*b + b^2 = (a + p) * (a + q)) ∧
  (¬ ∃ (p q : ℝ), a^2 + 4*a*b + b^2 = (a + p) * (a + q)) :=
  sorry

end NUMINAMATH_GPT_complete_square_form_l2006_200655


namespace NUMINAMATH_GPT_sixty_percent_of_total_is_960_l2006_200666

-- Definitions from the conditions
def boys : ℕ := 600
def difference : ℕ := 400
def girls : ℕ := boys + difference
def total : ℕ := boys + girls
def sixty_percent_of_total : ℕ := total * 60 / 100

-- The theorem to prove
theorem sixty_percent_of_total_is_960 :
  sixty_percent_of_total = 960 := 
  sorry

end NUMINAMATH_GPT_sixty_percent_of_total_is_960_l2006_200666


namespace NUMINAMATH_GPT_ax_product_zero_l2006_200640

theorem ax_product_zero 
  {a x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ x₁₁ x₁₂ x₁₃ : ℤ} 
  (h1 : a = (1 + x₁) * (1 + x₂) * (1 + x₃) * (1 + x₄) * (1 + x₅) * (1 + x₆) * (1 + x₇) *
           (1 + x₈) * (1 + x₉) * (1 + x₁₀) * (1 + x₁₁) * (1 + x₁₂) * (1 + x₁₃))
  (h2 : a = (1 - x₁) * (1 - x₂) * (1 - x₃) * (1 - x₄) * (1 - x₅) * (1 - x₆) * (1 - x₇) *
           (1 - x₈) * (1 - x₉) * (1 - x₁₀) * (1 - x₁₁) * (1 - x₁₂) * (1 - x₁₃)) :
  a * x₁ * x₂ * x₃ * x₄ * x₅ * x₆ * x₇ * x₈ * x₉ * x₁₀ * x₁₁ * x₁₂ * x₁₃ = 0 := 
sorry

end NUMINAMATH_GPT_ax_product_zero_l2006_200640


namespace NUMINAMATH_GPT_degenerate_ellipse_value_c_l2006_200665

theorem degenerate_ellipse_value_c (c : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 14 * y + c = 0) ∧
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 14 * y + c = 0 → (x+1)^2 + (y-7)^2 = 0) ↔ c = 52 :=
by
  sorry

end NUMINAMATH_GPT_degenerate_ellipse_value_c_l2006_200665


namespace NUMINAMATH_GPT_parity_of_expression_l2006_200671

theorem parity_of_expression (a b c : ℕ) (h_apos : 0 < a) (h_aodd : a % 2 = 1) (h_beven : b % 2 = 0) :
  (3^a + (b+1)^2 * c) % 2 = if c % 2 = 0 then 1 else 0 :=
sorry

end NUMINAMATH_GPT_parity_of_expression_l2006_200671


namespace NUMINAMATH_GPT_integers_sum_eighteen_l2006_200663

theorem integers_sum_eighteen (a b : ℕ) (h₀ : a ≠ b) (h₁ : a < 20) (h₂ : b < 20) (h₃ : Nat.gcd a b = 1) 
(h₄ : a * b + a + b = 95) : a + b = 18 :=
by
  sorry

end NUMINAMATH_GPT_integers_sum_eighteen_l2006_200663


namespace NUMINAMATH_GPT_race_problem_equivalent_l2006_200629

noncomputable def race_track_distance (D_paved D_dirt D_muddy : ℝ) : Prop :=
  let v1 := 100 -- speed on paved section in km/h
  let v2 := 70  -- speed on dirt section in km/h
  let v3 := 15  -- speed on muddy section in km/h
  let initial_distance := 0.5 -- initial distance in km (since 500 meters is 0.5 km)
  
  -- Time to cover paved section
  let t_white_paved := D_paved / v1
  let t_red_paved := (D_paved - initial_distance) / v1

  -- Times to cover dirt section
  let t_white_dirt := D_dirt / v2
  let t_red_dirt := D_dirt / v2 -- same time since both start at the same time on dirt

  -- Times to cover muddy section
  let t_white_muddy := D_muddy / v3
  let t_red_muddy := D_muddy / v3 -- same time since both start at the same time on mud

  -- Distances between cars on dirt and muddy sections
  ((t_white_paved - t_red_paved) * v2 = initial_distance) ∧ 
  ((t_white_paved - t_red_paved) * v3 = initial_distance)

-- Prove the distance between the cars when both are on the dirt and muddy sections is 500 meters
theorem race_problem_equivalent (D_paved D_dirt D_muddy : ℝ) : race_track_distance D_paved D_dirt D_muddy :=
by
  -- Insert proof here, for now we use sorry
  sorry

end NUMINAMATH_GPT_race_problem_equivalent_l2006_200629


namespace NUMINAMATH_GPT_polynomial_divisibility_condition_l2006_200646

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := x^5 - x^4 + x^3 - p * x^2 + q * x - 6

theorem polynomial_divisibility_condition (p q : ℝ) :
  (f (-1) p q = 0) ∧ (f 2 p q = 0) → 
  (p = 0) ∧ (q = -9) := by
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_condition_l2006_200646


namespace NUMINAMATH_GPT_calculate_product_l2006_200608

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end NUMINAMATH_GPT_calculate_product_l2006_200608


namespace NUMINAMATH_GPT_diagonal_of_rectangular_prism_l2006_200614

theorem diagonal_of_rectangular_prism (x y z : ℝ) (d : ℝ)
  (h_surface_area : 2 * x * y + 2 * x * z + 2 * y * z = 22)
  (h_edge_length : x + y + z = 6) :
  d = Real.sqrt 14 :=
by
  sorry

end NUMINAMATH_GPT_diagonal_of_rectangular_prism_l2006_200614


namespace NUMINAMATH_GPT_branches_on_fourth_tree_l2006_200691

theorem branches_on_fourth_tree :
  ∀ (height_1 branches_1 height_2 branches_2 height_3 branches_3 height_4 avg_branches_per_foot : ℕ),
    height_1 = 50 →
    branches_1 = 200 →
    height_2 = 40 →
    branches_2 = 180 →
    height_3 = 60 →
    branches_3 = 180 →
    height_4 = 34 →
    avg_branches_per_foot = 4 →
    (height_4 * avg_branches_per_foot = 136) :=
by
  intros height_1 branches_1 height_2 branches_2 height_3 branches_3 height_4 avg_branches_per_foot
  intros h1_eq_50 b1_eq_200 h2_eq_40 b2_eq_180 h3_eq_60 b3_eq_180 h4_eq_34 avg_eq_4
  -- We assume the conditions of the problem are correct, so add them to the context
  have height1 := h1_eq_50
  have branches1 := b1_eq_200
  have height2 := h2_eq_40
  have branches2 := b2_eq_180
  have height3 := h3_eq_60
  have branches3 := b3_eq_180
  have height4 := h4_eq_34
  have avg_branches := avg_eq_4
  -- Now prove the desired result
  sorry

end NUMINAMATH_GPT_branches_on_fourth_tree_l2006_200691


namespace NUMINAMATH_GPT_ab_bc_ca_abc_inequality_l2006_200690

open Real

theorem ab_bc_ca_abc_inequality :
  ∀ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a^2 + b^2 + c^2 + a * b * c = 4 →
    0 ≤ a * b + b * c + c * a - a * b * c ∧ a * b + b * c + c * a - a * b * c ≤ 2 :=
by
  intro a b c
  intro h
  sorry

end NUMINAMATH_GPT_ab_bc_ca_abc_inequality_l2006_200690


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2006_200681

theorem quadratic_inequality_solution :
  ∀ x : ℝ, (3 * x^2 - 5 * x - 2 < 0) ↔ (-1/3 < x ∧ x < 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2006_200681


namespace NUMINAMATH_GPT_smallest_sum_of_squares_l2006_200668

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 91) : x^2 + y^2 ≥ 109 :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_squares_l2006_200668


namespace NUMINAMATH_GPT_circle_symmetric_line_a_value_l2006_200651

theorem circle_symmetric_line_a_value :
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∀ x y : ℝ, (x, y) = (-1, 2)) →
  (∀ x y : ℝ, ax + y + 1 = 0) →
  a = 3 :=
by
  sorry

end NUMINAMATH_GPT_circle_symmetric_line_a_value_l2006_200651


namespace NUMINAMATH_GPT_polynomial_transformation_l2006_200642

theorem polynomial_transformation (x y : ℂ) (h : y = x + 1/x) : x^4 + x^3 - 4*x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 6) = 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_transformation_l2006_200642


namespace NUMINAMATH_GPT_find_f4_l2006_200636

variable (a b : ℝ)
variable (f : ℝ → ℝ)
variable (h1 : f 1 = 5)
variable (h2 : f 2 = 8)
variable (h3 : f 3 = 11)
variable (h4 : ∀ x, f x = a * x + b)

theorem find_f4 : f 4 = 14 := by
  sorry

end NUMINAMATH_GPT_find_f4_l2006_200636


namespace NUMINAMATH_GPT_hanoi_moves_minimal_l2006_200675

theorem hanoi_moves_minimal (n : ℕ) : ∃ m, 
  (∀ move : ℕ, move = 2^n - 1 → move = m) := 
by
  sorry

end NUMINAMATH_GPT_hanoi_moves_minimal_l2006_200675


namespace NUMINAMATH_GPT_polygon_sides_l2006_200659

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l2006_200659


namespace NUMINAMATH_GPT_length_of_first_square_flag_l2006_200677

theorem length_of_first_square_flag
  (x : ℝ)
  (h1x : x * 5 + 10 * 7 + 5 * 5 = 15 * 9) : 
  x = 8 :=
by
  sorry

end NUMINAMATH_GPT_length_of_first_square_flag_l2006_200677


namespace NUMINAMATH_GPT_price_increase_percentage_l2006_200619

theorem price_increase_percentage (original_price new_price : ℝ) (h₁ : original_price = 300) (h₂ : new_price = 360) : 
  (new_price - original_price) / original_price * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_price_increase_percentage_l2006_200619


namespace NUMINAMATH_GPT_numberOfTermsArithmeticSequence_l2006_200606

theorem numberOfTermsArithmeticSequence (a1 d l : ℕ) (h1 : a1 = 3) (h2 : d = 4) (h3 : l = 2012) :
  ∃ n : ℕ, 3 + (n - 1) * 4 ≤ 2012 ∧ (n : ℕ) = 502 :=
by {
  sorry
}

end NUMINAMATH_GPT_numberOfTermsArithmeticSequence_l2006_200606


namespace NUMINAMATH_GPT_diana_wins_l2006_200615

noncomputable def probability_diana_wins : ℚ :=
  45 / 100

theorem diana_wins (d : ℕ) (a : ℕ) (hd : 1 ≤ d ∧ d ≤ 10) (ha : 1 ≤ a ∧ a ≤ 10) :
  probability_diana_wins = 9 / 20 :=
by
  sorry

end NUMINAMATH_GPT_diana_wins_l2006_200615


namespace NUMINAMATH_GPT_eq_satisfies_exactly_four_points_l2006_200638

theorem eq_satisfies_exactly_four_points : ∀ (x y : ℝ), 
  (x^2 - 4)^2 + (y^2 - 4)^2 = 0 ↔ 
  (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) ∨ (x = 2 ∧ y = -2) ∨ (x = -2 ∧ y = -2) := 
by
  sorry

end NUMINAMATH_GPT_eq_satisfies_exactly_four_points_l2006_200638


namespace NUMINAMATH_GPT_total_number_of_people_l2006_200684

theorem total_number_of_people (num_cannoneers num_women num_men total_people : ℕ)
  (h1 : num_women = 2 * num_cannoneers)
  (h2 : num_cannoneers = 63)
  (h3 : num_men = 2 * num_women)
  (h4 : total_people = num_women + num_men) : 
  total_people = 378 := by
  sorry

end NUMINAMATH_GPT_total_number_of_people_l2006_200684


namespace NUMINAMATH_GPT_divisors_of_30240_l2006_200647

theorem divisors_of_30240 : 
  ∃ s : Finset ℕ, (s = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (∀ d ∈ s, (30240 % d = 0)) ∧ (s.card = 9) :=
by
  sorry

end NUMINAMATH_GPT_divisors_of_30240_l2006_200647


namespace NUMINAMATH_GPT_sin_double_angle_l2006_200625

theorem sin_double_angle (theta : ℝ) 
  (h : Real.sin (theta + Real.pi / 4) = 2 / 5) :
  Real.sin (2 * theta) = -17 / 25 := by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l2006_200625


namespace NUMINAMATH_GPT_corner_contains_same_color_cells_l2006_200670

theorem corner_contains_same_color_cells (colors : Finset (Fin 120)) :
  ∀ (coloring : Fin 2017 × Fin 2017 → Fin 120),
  ∃ (corner : Fin 2017 × Fin 2017 → Prop), 
    (∃ cell1 cell2, corner cell1 ∧ corner cell2 ∧ coloring cell1 = coloring cell2) := 
by 
  sorry

end NUMINAMATH_GPT_corner_contains_same_color_cells_l2006_200670


namespace NUMINAMATH_GPT_find_lost_bowls_l2006_200676

def bowls_problem (L : ℕ) : Prop :=
  let total_bowls := 638
  let broken_bowls := 15
  let payment := 1825
  let fee := 100
  let safe_bowl_payment := 3
  let lost_broken_bowl_cost := 4
  100 + 3 * (total_bowls - L - broken_bowls) - 4 * (L + broken_bowls) = payment

theorem find_lost_bowls : ∃ L : ℕ, bowls_problem L ∧ L = 26 :=
  by
  sorry

end NUMINAMATH_GPT_find_lost_bowls_l2006_200676


namespace NUMINAMATH_GPT_eval_expression_at_neg3_l2006_200617

def evaluate_expression (x : ℤ) : ℚ :=
  (5 + x * (5 + x) - 4 ^ 2 : ℤ) / (x - 4 + x ^ 3 : ℤ)

theorem eval_expression_at_neg3 :
  evaluate_expression (-3) = -17 / 20 := by
  sorry

end NUMINAMATH_GPT_eval_expression_at_neg3_l2006_200617


namespace NUMINAMATH_GPT_recreation_proof_l2006_200648

noncomputable def recreation_percentage_last_week (W : ℝ) (P : ℝ) :=
  let last_week_spent := (P/100) * W
  let this_week_wages := (70/100) * W
  let this_week_spent := (20/100) * this_week_wages
  this_week_spent = (70/100) * last_week_spent

theorem recreation_proof :
  ∀ (W : ℝ), recreation_percentage_last_week W 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_recreation_proof_l2006_200648


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2006_200699

variable (x : ℝ)

theorem necessary_but_not_sufficient (h : x > 2) : x > 1 ∧ ¬ (x > 1 → x > 2) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2006_200699


namespace NUMINAMATH_GPT_diamond_value_l2006_200674

def diamond (a b : Int) : Int :=
  a * b^2 - b + 1

theorem diamond_value : diamond (-1) 6 = -41 := by
  sorry

end NUMINAMATH_GPT_diamond_value_l2006_200674


namespace NUMINAMATH_GPT_division_quotient_proof_l2006_200697

theorem division_quotient_proof :
  (300324 / 29 = 10356) →
  (100007892 / 333 = 300324) :=
by
  intros h1
  sorry

end NUMINAMATH_GPT_division_quotient_proof_l2006_200697


namespace NUMINAMATH_GPT_binary_11101_to_decimal_l2006_200626

theorem binary_11101_to_decimal : 
  (1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 29) := by
  sorry

end NUMINAMATH_GPT_binary_11101_to_decimal_l2006_200626


namespace NUMINAMATH_GPT_bicentric_quad_lemma_l2006_200644

-- Define the properties and radii of the bicentric quadrilateral
variables (KLMN : Type) (r ρ h : ℝ)

-- Assuming quadrilateral KLMN is bicentric with given radii
def is_bicentric (KLMN : Type) := true

-- State the theorem we wish to prove
theorem bicentric_quad_lemma (br : is_bicentric KLMN) : 
  (1 / (ρ + h) ^ 2) + (1 / (ρ - h) ^ 2) = (1 / r ^ 2) :=
sorry

end NUMINAMATH_GPT_bicentric_quad_lemma_l2006_200644


namespace NUMINAMATH_GPT_tom_seashells_now_l2006_200694

def original_seashells : ℕ := 5
def given_seashells : ℕ := 2

theorem tom_seashells_now : original_seashells - given_seashells = 3 :=
by
  sorry

end NUMINAMATH_GPT_tom_seashells_now_l2006_200694


namespace NUMINAMATH_GPT_unique_integer_solution_range_l2006_200605

theorem unique_integer_solution_range (a : ℝ) : 
  (∀ x : ℝ, (2 * x + 3 > 5) ∧ (x - a ≤ 0) → (x = 2)) ↔ (2 ≤ a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_unique_integer_solution_range_l2006_200605


namespace NUMINAMATH_GPT_geometric_sequence_150th_term_l2006_200688

noncomputable def geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem geometric_sequence_150th_term :
  geometric_sequence 8 (-1 / 2) 150 = -8 * (1 / 2) ^ 149 :=
by
  -- This is the proof placeholder
  sorry

end NUMINAMATH_GPT_geometric_sequence_150th_term_l2006_200688


namespace NUMINAMATH_GPT_value_of_y_l2006_200682

theorem value_of_y (x y : ℤ) (h1 : x + y = 270) (h2 : x - y = 200) : y = 35 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l2006_200682


namespace NUMINAMATH_GPT_sum_le_square_l2006_200628

theorem sum_le_square (m n : ℕ) (h: (m * n) % (m + n) = 0) : m + n ≤ n^2 :=
by sorry

end NUMINAMATH_GPT_sum_le_square_l2006_200628


namespace NUMINAMATH_GPT_quadratic_sum_constants_l2006_200696

-- Define the quadratic expression
def quadratic (x : ℝ) : ℝ := -3 * x^2 + 27 * x + 135

-- Define the representation of the quadratic in the form a(x + b)^2 + c
def quadratic_rewritten (a b c : ℝ) (x : ℝ) : ℝ := a * (x + b)^2 + c

-- Theorem statement
theorem quadratic_sum_constants :
  ∃ a b c, (∀ x, quadratic x = quadratic_rewritten a b c x) ∧ a + b + c = 197.75 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_sum_constants_l2006_200696


namespace NUMINAMATH_GPT_part_1_part_2_l2006_200634

def p (a x : ℝ) : Prop :=
a * x - 2 ≤ 0 ∧ a * x + 1 > 0

def q (x : ℝ) : Prop :=
x^2 - x - 2 < 0

theorem part_1 (a : ℝ) :
  (∃ x : ℝ, (1/2 < x ∧ x < 3) ∧ p a x) → 
  (-2 < a ∧ a < 4) :=
sorry

theorem part_2 (a : ℝ) :
  (∀ x, p a x → q x) ∧ 
  (∃ x, q x ∧ ¬p a x) → 
  (-1/2 ≤ a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_GPT_part_1_part_2_l2006_200634


namespace NUMINAMATH_GPT_remainder_when_3m_divided_by_5_l2006_200609

theorem remainder_when_3m_divided_by_5 (m : ℤ) (hm : m % 5 = 2) : (3 * m) % 5 = 1 := 
sorry

end NUMINAMATH_GPT_remainder_when_3m_divided_by_5_l2006_200609


namespace NUMINAMATH_GPT_angle_of_inclination_of_line_l2006_200603

theorem angle_of_inclination_of_line (x y : ℝ) (h : x - y - 1 = 0) : 
  ∃ α : ℝ, α = π / 4 := 
sorry

end NUMINAMATH_GPT_angle_of_inclination_of_line_l2006_200603


namespace NUMINAMATH_GPT_person_A_number_is_35_l2006_200693

theorem person_A_number_is_35
    (A B : ℕ)
    (h1 : A + B = 8)
    (h2 : 10 * B + A - (10 * A + B) = 18) :
    10 * A + B = 35 :=
by
    sorry

end NUMINAMATH_GPT_person_A_number_is_35_l2006_200693


namespace NUMINAMATH_GPT_exchange_ways_count_l2006_200678

theorem exchange_ways_count : ∃ n : ℕ, n = 46 ∧ ∀ x y z : ℕ, x + 2 * y + 5 * z = 20 → n = 46 :=
by
  sorry

end NUMINAMATH_GPT_exchange_ways_count_l2006_200678


namespace NUMINAMATH_GPT_book_pages_total_l2006_200698

theorem book_pages_total
  (days_in_week : ℕ)
  (daily_read_times : ℕ)
  (pages_per_time : ℕ)
  (additional_pages_per_day : ℕ)
  (num_days : days_in_week = 7)
  (times_per_day : daily_read_times = 3)
  (pages_each_time : pages_per_time = 6)
  (extra_pages : additional_pages_per_day = 2) :
  daily_read_times * pages_per_time + additional_pages_per_day * days_in_week = 140 := 
sorry

end NUMINAMATH_GPT_book_pages_total_l2006_200698


namespace NUMINAMATH_GPT_inequality_solution_l2006_200643

theorem inequality_solution (x : ℝ) (h : 1 / (x - 2) < 4) : x < 2 ∨ x > 9 / 4 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l2006_200643


namespace NUMINAMATH_GPT_find_a_b_l2006_200600

theorem find_a_b (a b : ℝ) (h₁ : a^2 = 64 * b) (h₂ : a^2 = 4 * b) : a = 0 ∧ b = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_l2006_200600


namespace NUMINAMATH_GPT_find_n_l2006_200632

theorem find_n (n : ℕ) : (16 : ℝ)^(1/4) = 2^n ↔ n = 1 := by
  sorry

end NUMINAMATH_GPT_find_n_l2006_200632


namespace NUMINAMATH_GPT_andy_tomatoes_left_l2006_200637

theorem andy_tomatoes_left :
  let plants := 50
  let tomatoes_per_plant := 15
  let total_tomatoes := plants * tomatoes_per_plant
  let tomatoes_dried := (2 / 3) * total_tomatoes
  let tomatoes_left_after_drying := total_tomatoes - tomatoes_dried
  let tomatoes_for_marinara := (1 / 2) * tomatoes_left_after_drying
  let tomatoes_left := tomatoes_left_after_drying - tomatoes_for_marinara
  tomatoes_left = 125 := sorry

end NUMINAMATH_GPT_andy_tomatoes_left_l2006_200637


namespace NUMINAMATH_GPT_find_certain_number_l2006_200692

theorem find_certain_number (a : ℤ) (certain_number : ℤ) (h₁ : a = 105) (h₂ : a^3 = 21 * 25 * 45 * certain_number) : certain_number = 49 := 
sorry

end NUMINAMATH_GPT_find_certain_number_l2006_200692


namespace NUMINAMATH_GPT_magic_square_sum_l2006_200664

-- Given conditions
def magic_square (S : ℕ) (a b c d e : ℕ) :=
  (30 + b + 27 = S) ∧
  (30 + 33 + a = S) ∧
  (33 + c + d = S) ∧
  (a + 18 + e = S) ∧
  (30 + c + e = S)

-- Prove that the sum a + d is 38 given the sums of the 3x3 magic square are equivalent
theorem magic_square_sum (a b c d e S : ℕ) (h : magic_square S a b c d e) : a + d = 38 :=
  sorry

end NUMINAMATH_GPT_magic_square_sum_l2006_200664


namespace NUMINAMATH_GPT_largest_square_test_plots_l2006_200687

/-- 
  A fenced, rectangular field measures 30 meters by 45 meters. 
  An agricultural researcher has 1500 meters of fence that can be used for internal fencing to partition 
  the field into congruent, square test plots. 
  The entire field must be partitioned, and the sides of the squares must be parallel to the edges of the field. 
  What is the largest number of square test plots into which the field can be partitioned using all or some of the 1500 meters of fence?
 -/
theorem largest_square_test_plots
  (field_length : ℕ := 30)
  (field_width : ℕ := 45)
  (total_fence_length : ℕ := 1500):
  ∃ (n : ℕ), n = 576 := 
sorry

end NUMINAMATH_GPT_largest_square_test_plots_l2006_200687


namespace NUMINAMATH_GPT_a_is_perfect_square_l2006_200616

theorem a_is_perfect_square (a b : ℕ) (h : ∃ (k : ℕ), a^2 + b^2 + a = k * a * b) : ∃ n : ℕ, a = n^2 := by
  sorry

end NUMINAMATH_GPT_a_is_perfect_square_l2006_200616


namespace NUMINAMATH_GPT_selfish_subsets_equals_fibonacci_l2006_200627

noncomputable def fibonacci : ℕ → ℕ
| 0           => 0
| 1           => 1
| (n + 2)     => fibonacci (n + 1) + fibonacci n

noncomputable def selfish_subsets_count (n : ℕ) : ℕ := 
sorry -- This will be replaced with the correct recursive function

theorem selfish_subsets_equals_fibonacci (n : ℕ) : 
  selfish_subsets_count n = fibonacci n :=
sorry

end NUMINAMATH_GPT_selfish_subsets_equals_fibonacci_l2006_200627


namespace NUMINAMATH_GPT_remainder_5_pow_100_div_18_l2006_200658

theorem remainder_5_pow_100_div_18 : (5 ^ 100) % 18 = 13 := 
  sorry

end NUMINAMATH_GPT_remainder_5_pow_100_div_18_l2006_200658


namespace NUMINAMATH_GPT_bacteria_growth_relation_l2006_200610

variable (w1: ℝ := 10.0) (w2: ℝ := 16.0) (w3: ℝ := 25.6)

theorem bacteria_growth_relation :
  (w2 / w1) = (w3 / w2) :=
by
  sorry

end NUMINAMATH_GPT_bacteria_growth_relation_l2006_200610


namespace NUMINAMATH_GPT_problem_l2006_200662

variable {x : ℝ}

theorem problem (h : x + 1/x = 5) : x^4 + 1/x^4 = 527 :=
by
  sorry

end NUMINAMATH_GPT_problem_l2006_200662


namespace NUMINAMATH_GPT_average_income_l2006_200680

-- Lean statement to express the given mathematical problem
theorem average_income (A B C : ℝ) 
  (h1 : (A + B) / 2 = 4050)
  (h2 : (B + C) / 2 = 5250)
  (h3 : A = 3000) :
  (A + C) / 2 = 4200 :=
by
  sorry

end NUMINAMATH_GPT_average_income_l2006_200680


namespace NUMINAMATH_GPT_least_number_divisible_l2006_200657

-- Define the numbers as given in the conditions
def given_number : ℕ := 3072
def divisor1 : ℕ := 57
def divisor2 : ℕ := 29
def least_number_to_add : ℕ := 234

-- Define the LCM
noncomputable def lcm_57_29 : ℕ := Nat.lcm divisor1 divisor2

-- Prove that adding least_number_to_add to given_number makes it divisible by both divisors
theorem least_number_divisible :
  (given_number + least_number_to_add) % divisor1 = 0 ∧ 
  (given_number + least_number_to_add) % divisor2 = 0 := 
by
  -- Proof should be provided here
  sorry

end NUMINAMATH_GPT_least_number_divisible_l2006_200657


namespace NUMINAMATH_GPT_golden_ratio_problem_l2006_200633

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

theorem golden_ratio_problem (m : ℝ) (x : ℝ) :
  (1000 ≤ m) → (1000 ≤ x) → (x ≤ m) →
  ((m - 1000) / (x - 1000) = phi ∧ (x - 1000) / (m - x) = phi) →
  (m = 2000 ∨ m = 2618) :=
by
  sorry

end NUMINAMATH_GPT_golden_ratio_problem_l2006_200633


namespace NUMINAMATH_GPT_average_annual_growth_rate_l2006_200641

variable (a b : ℝ)

theorem average_annual_growth_rate :
  ∃ x : ℝ, (1 + x)^2 = (1 + a) * (1 + b) ∧ x = Real.sqrt ((1 + a) * (1 + b)) - 1 := by
  sorry

end NUMINAMATH_GPT_average_annual_growth_rate_l2006_200641


namespace NUMINAMATH_GPT_reflection_problem_l2006_200650

theorem reflection_problem 
  (m b : ℝ)
  (h : ∀ (P Q : ℝ × ℝ), 
        P = (2,2) ∧ Q = (8,4) → 
        ∃ mid : ℝ × ℝ, 
        mid = ((P.fst + Q.fst) / 2, (P.snd + Q.snd) / 2) ∧ 
        ∃ m' : ℝ, m' ≠ 0 ∧ P.snd - m' * P.fst = Q.snd - m' * Q.fst) :
  m + b = 15 := 
sorry

end NUMINAMATH_GPT_reflection_problem_l2006_200650


namespace NUMINAMATH_GPT_expected_accidents_no_overtime_l2006_200669

noncomputable def accidents_with_no_overtime_hours 
    (hours1 hours2 : ℕ) (accidents1 accidents2 : ℕ) : ℕ :=
  let slope := (accidents2 - accidents1) / (hours2 - hours1)
  let intercept := accidents1 - slope * hours1
  intercept

theorem expected_accidents_no_overtime : 
    accidents_with_no_overtime_hours 1000 400 8 5 = 3 :=
by
  sorry

end NUMINAMATH_GPT_expected_accidents_no_overtime_l2006_200669


namespace NUMINAMATH_GPT_solve_for_t_l2006_200635

theorem solve_for_t (p t : ℝ) (h1 : 5 = p * 3^t) (h2 : 45 = p * 9^t) : t = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_t_l2006_200635


namespace NUMINAMATH_GPT_solve_fractional_equation_l2006_200601

theorem solve_fractional_equation (x : ℝ) (h : (3 * x + 6) / (x ^ 2 + 5 * x - 6) = (3 - x) / (x - 1)) (hx : x ≠ 1) : x = -4 := 
sorry

end NUMINAMATH_GPT_solve_fractional_equation_l2006_200601


namespace NUMINAMATH_GPT_final_state_of_marbles_after_operations_l2006_200639

theorem final_state_of_marbles_after_operations :
  ∃ (b w : ℕ), b + w = 2 ∧ w = 2 ∧ (∀ n : ℕ, n % 2 = 0 → n = 100 - k * 2) :=
sorry

end NUMINAMATH_GPT_final_state_of_marbles_after_operations_l2006_200639


namespace NUMINAMATH_GPT_camp_weights_l2006_200653

theorem camp_weights (m_e_w : ℕ) (m_e_w1 : ℕ) (c_w : ℕ) (m_e_w2 : ℕ) (d : ℕ)
  (h1 : m_e_w = 30) 
  (h2 : m_e_w1 = 28) 
  (h3 : c_w = 56)
  (h4 : m_e_w = m_e_w1 + d)
  (h5 : m_e_w1 = m_e_w2 + d)
  (h6 : c_w = m_e_w + m_e_w1 + d) :
  m_e_w = 28 ∧ m_e_w2 = 26 := 
by {
    sorry
}

end NUMINAMATH_GPT_camp_weights_l2006_200653


namespace NUMINAMATH_GPT_bathroom_area_is_eight_l2006_200695

def bathroomArea (length width : ℕ) : ℕ :=
  length * width

theorem bathroom_area_is_eight : bathroomArea 4 2 = 8 := 
by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_bathroom_area_is_eight_l2006_200695


namespace NUMINAMATH_GPT_dogs_in_shelter_l2006_200679

theorem dogs_in_shelter (D C : ℕ) (h1 : D * 7 = 15 * C) (h2 : D * 11 = 15 * (C + 8)) :
  D = 30 :=
sorry

end NUMINAMATH_GPT_dogs_in_shelter_l2006_200679


namespace NUMINAMATH_GPT_values_of_a2_add_b2_l2006_200673

theorem values_of_a2_add_b2 (a b : ℝ) (h1 : a^3 - 3 * a * b^2 = 11) (h2 : b^3 - 3 * a^2 * b = 2) : a^2 + b^2 = 5 := 
by
  sorry

end NUMINAMATH_GPT_values_of_a2_add_b2_l2006_200673


namespace NUMINAMATH_GPT_average_water_per_day_l2006_200612

-- Define the given conditions as variables/constants
def day1 := 318
def day2 := 312
def day3_morning := 180
def day3_afternoon := 162

-- Define the total water added on day 3
def day3 := day3_morning + day3_afternoon

-- Define the total water added over three days
def total_water := day1 + day2 + day3

-- Define the number of days
def days := 3

-- The proof statement: the average water added per day is 324 liters
theorem average_water_per_day : total_water / days = 324 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_average_water_per_day_l2006_200612


namespace NUMINAMATH_GPT_find_a_l2006_200685

noncomputable def f (a x : ℝ) := a * x + 1 / Real.sqrt 2

theorem find_a (a : ℝ) (h_pos : 0 < a) (h : f a (f a (1 / Real.sqrt 2)) = f a 0) : a = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2006_200685


namespace NUMINAMATH_GPT_imaginary_unit_div_l2006_200652

open Complex

theorem imaginary_unit_div (i : ℂ) (hi : i * i = -1) : (i / (1 + i) = (1 / 2) + (1 / 2) * i) :=
by
  sorry

end NUMINAMATH_GPT_imaginary_unit_div_l2006_200652


namespace NUMINAMATH_GPT_opposite_of_neg_2023_l2006_200672

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_2023_l2006_200672


namespace NUMINAMATH_GPT_next_number_in_sequence_is_131_l2006_200620

/-- Define the sequence increments between subsequent numbers -/
def sequencePattern : List ℕ := [1, 2, 2, 4, 2, 4, 2, 4, 6, 2]

-- Function to apply a sequence of increments starting from an initial value
def computeNext (initial : ℕ) (increments : List ℕ) : ℕ :=
  increments.foldl (λ acc inc => acc + inc) initial

-- Function to get the sequence's nth element 
def sequenceNthElement (n : ℕ) : ℕ :=
  (computeNext 12 (sequencePattern.take n))

-- Proof that the next number in the sequence is 131 
theorem next_number_in_sequence_is_131 :
  sequenceNthElement 10 = 131 :=
  by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_next_number_in_sequence_is_131_l2006_200620


namespace NUMINAMATH_GPT_remainder_2n_div_9_l2006_200624

theorem remainder_2n_div_9 (n : ℤ) (h : n % 18 = 10) : (2 * n) % 9 = 2 := 
sorry

end NUMINAMATH_GPT_remainder_2n_div_9_l2006_200624


namespace NUMINAMATH_GPT_carton_height_is_60_l2006_200683

-- Definitions
def carton_length : ℕ := 30
def carton_width : ℕ := 42
def soap_length : ℕ := 7
def soap_width : ℕ := 6
def soap_height : ℕ := 5
def max_soap_boxes : ℕ := 360

-- Theorem Statement
theorem carton_height_is_60 (h : ℕ) (H : ∀ (layers : ℕ), layers = max_soap_boxes / ((carton_length / soap_length) * (carton_width / soap_width)) → h = layers * soap_height) : h = 60 :=
  sorry

end NUMINAMATH_GPT_carton_height_is_60_l2006_200683


namespace NUMINAMATH_GPT_measure_angle_ABC_l2006_200623

theorem measure_angle_ABC (x : ℝ) (h1 : ∃ θ, θ = 180 - x ∧ x / 2 = (180 - x) / 3) : x = 72 :=
by
  sorry

end NUMINAMATH_GPT_measure_angle_ABC_l2006_200623


namespace NUMINAMATH_GPT_farmer_farm_size_l2006_200607

theorem farmer_farm_size 
  (sunflowers flax : ℕ)
  (h1 : flax = 80)
  (h2 : sunflowers = flax + 80) :
  (sunflowers + flax = 240) :=
by
  sorry

end NUMINAMATH_GPT_farmer_farm_size_l2006_200607


namespace NUMINAMATH_GPT_train_speed_l2006_200656

theorem train_speed (train_length : ℝ) (man_speed_kmph : ℝ) (passing_time : ℝ) : 
  train_length = 160 → man_speed_kmph = 6 →
  passing_time = 6 → (train_length / passing_time + man_speed_kmph * 1000 / 3600) * 3600 / 1000 = 90 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- further proof steps are omitted
  sorry

end NUMINAMATH_GPT_train_speed_l2006_200656
