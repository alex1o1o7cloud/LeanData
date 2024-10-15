import Mathlib

namespace NUMINAMATH_GPT_dot_not_line_l71_7160

variable (D S DS T : Nat)
variable (h1 : DS = 20) (h2 : S = 36) (h3 : T = 60)
variable (h4 : T = D + S - DS)

theorem dot_not_line : (D - DS) = 24 :=
by
  sorry

end NUMINAMATH_GPT_dot_not_line_l71_7160


namespace NUMINAMATH_GPT_find_positive_number_l71_7151

-- The definition to state the given condition
def condition1 (n : ℝ) : Prop := n > 0 ∧ n^2 + n = 245

-- The theorem stating the problem and its solution
theorem find_positive_number (n : ℝ) (h : condition1 n) : n = 14 :=
by sorry

end NUMINAMATH_GPT_find_positive_number_l71_7151


namespace NUMINAMATH_GPT_company_picnic_attendance_l71_7124

theorem company_picnic_attendance :
  ∀ (employees men women men_attending women_attending : ℕ)
  (h_employees : employees = 100)
  (h_men : men = 55)
  (h_women : women = 45)
  (h_men_attending: men_attending = 11)
  (h_women_attending: women_attending = 18),
  (100 * (men_attending + women_attending) / employees) = 29 := 
by
  intros employees men women men_attending women_attending 
         h_employees h_men h_women h_men_attending h_women_attending
  sorry

end NUMINAMATH_GPT_company_picnic_attendance_l71_7124


namespace NUMINAMATH_GPT_tetrad_does_not_have_four_chromosomes_l71_7166

noncomputable def tetrad_has_two_centromeres : Prop := -- The condition: a tetrad has two centromeres
  sorry

noncomputable def tetrad_contains_four_dna_molecules : Prop := -- The condition: a tetrad contains four DNA molecules
  sorry

noncomputable def tetrad_consists_of_two_pairs_of_sister_chromatids : Prop := -- The condition: a tetrad consists of two pairs of sister chromatids
  sorry

theorem tetrad_does_not_have_four_chromosomes 
  (h1: tetrad_has_two_centromeres)
  (h2: tetrad_contains_four_dna_molecules)
  (h3: tetrad_consists_of_two_pairs_of_sister_chromatids) 
  : ¬ (tetrad_has_four_chromosomes : Prop) :=
sorry

end NUMINAMATH_GPT_tetrad_does_not_have_four_chromosomes_l71_7166


namespace NUMINAMATH_GPT_number_of_eighth_graders_l71_7104

theorem number_of_eighth_graders (x y : ℕ) :
  (x > 0) ∧ (y > 0) ∧ (8 + x * y = (x * (x + 3) - 14) / 2) →
  x = 7 ∨ x = 14 :=
by
  sorry

end NUMINAMATH_GPT_number_of_eighth_graders_l71_7104


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l71_7177

variable (a : ℕ → ℕ)
variable (d : ℕ) -- common difference for the arithmetic sequence
variable (h1 : ∀ n : ℕ, a (n + 1) = a n + d)
variable (h2 : a 1 - a 9 + a 17 = 7)

theorem arithmetic_sequence_problem : a 3 + a 15 = 14 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l71_7177


namespace NUMINAMATH_GPT_remainder_of_sum_l71_7125

theorem remainder_of_sum :
  ((88134 + 88135 + 88136 + 88137 + 88138 + 88139) % 9) = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_l71_7125


namespace NUMINAMATH_GPT_best_is_man_l71_7163

structure Competitor where
  name : String
  gender : String
  age : Int
  is_twin : Bool

noncomputable def participants : List Competitor := [
  ⟨"man", "male", 30, false⟩,
  ⟨"sister", "female", 30, true⟩,
  ⟨"son", "male", 30, true⟩,
  ⟨"niece", "female", 25, false⟩
]

def are_different_gender (c1 c2 : Competitor) : Bool := c1.gender ≠ c2.gender
def has_same_age (c1 c2 : Competitor) : Bool := c1.age = c2.age

noncomputable def best_competitor : Competitor :=
  let best_candidate := participants[0] -- assuming "man" is the best for example's sake
  let worst_candidate := participants[2] -- assuming "son" is the worst for example's sake
  best_candidate

theorem best_is_man : best_competitor.name = "man" :=
by
  have h1 : are_different_gender (participants[0]) (participants[2]) := by sorry
  have h2 : has_same_age (participants[0]) (participants[2]) := by sorry
  exact sorry

end NUMINAMATH_GPT_best_is_man_l71_7163


namespace NUMINAMATH_GPT_banana_count_l71_7111

theorem banana_count : (2 + 7) = 9 := by
  rfl

end NUMINAMATH_GPT_banana_count_l71_7111


namespace NUMINAMATH_GPT_part1_part2_l71_7147

variable (f : ℝ → ℝ)

-- Conditions
axiom h1 : ∀ x y : ℝ, f (x - y) = f x / f y
axiom h2 : ∀ x : ℝ, f x > 0
axiom h3 : ∀ x y : ℝ, x < y → f x > f y

-- First part: f(0) = 1 and proving f(x + y) = f(x) * f(y)
theorem part1 : f 0 = 1 ∧ (∀ x y : ℝ, f (x + y) = f x * f y) :=
sorry

-- Second part: Given f(-1) = 3, solve the inequality
axiom h4 : f (-1) = 3

theorem part2 : {x : ℝ | (x ≤ 3) ∨ (x ≥ 4)} = {x : ℝ | f (x^2 - 7*x + 10) ≤ f (-2)} :=
sorry

end NUMINAMATH_GPT_part1_part2_l71_7147


namespace NUMINAMATH_GPT_total_weight_loss_l71_7181

def seth_loss : ℝ := 17.53
def jerome_loss : ℝ := 3 * seth_loss
def veronica_loss : ℝ := seth_loss + 1.56
def seth_veronica_loss : ℝ := seth_loss + veronica_loss
def maya_loss : ℝ := seth_veronica_loss - 0.25 * seth_veronica_loss
def total_loss : ℝ := seth_loss + jerome_loss + veronica_loss + maya_loss

theorem total_weight_loss : total_loss = 116.675 := by
  sorry

end NUMINAMATH_GPT_total_weight_loss_l71_7181


namespace NUMINAMATH_GPT_eccentricity_range_l71_7113

-- Definitions and conditions
variable (a b c e : ℝ) (A B: ℝ × ℝ)
variable (d1 d2 : ℝ)

variable (a_pos : a > 2)
variable (b_pos : b > 0)
variable (c_pos : c > 0)
variable (c_eq : c = Real.sqrt (a ^ 2 + b ^ 2))
variable (A_def : A = (a, 0))
variable (B_def : B = (0, b))
variable (d1_def : d1 = abs (b * 2 + a * 0 - a * b ) / Real.sqrt (a^2 + b^2))
variable (d2_def : d2 = abs (b * (-2) + a * 0 - a * b) / Real.sqrt (a^2 + b^2))
variable (d_ineq : d1 + d2 ≥ (4 / 5) * c)
variable (eccentricity : e = c / a)

-- Theorem statement
theorem eccentricity_range : (Real.sqrt 5 / 2 ≤ e) ∧ (e ≤ Real.sqrt 5) :=
by sorry

end NUMINAMATH_GPT_eccentricity_range_l71_7113


namespace NUMINAMATH_GPT_asian_games_discount_equation_l71_7171

variable (a : ℝ)

theorem asian_games_discount_equation :
  168 * (1 - a / 100)^2 = 128 :=
sorry

end NUMINAMATH_GPT_asian_games_discount_equation_l71_7171


namespace NUMINAMATH_GPT_create_proper_six_sided_figure_l71_7120

-- Definition of a matchstick configuration
structure MatchstickConfig where
  sides : ℕ
  matchsticks : ℕ

-- Initial configuration: a regular hexagon with 6 matchsticks
def initialConfig : MatchstickConfig := ⟨6, 6⟩

-- Condition: Cannot lay any stick on top of another, no free ends
axiom no_overlap (cfg : MatchstickConfig) : Prop
axiom no_free_ends (cfg : MatchstickConfig) : Prop

-- New configuration after adding 3 matchsticks
def newConfig : MatchstickConfig := ⟨6, 9⟩

-- Theorem stating the possibility to create a proper figure with six sides
theorem create_proper_six_sided_figure : no_overlap newConfig → no_free_ends newConfig → newConfig.sides = 6 :=
by
  sorry

end NUMINAMATH_GPT_create_proper_six_sided_figure_l71_7120


namespace NUMINAMATH_GPT_simplify_evaluate_expression_l71_7188

noncomputable def a : ℝ := 2 * Real.cos (60 * Real.pi / 180) + 1

theorem simplify_evaluate_expression : (a - (a^2) / (a + 1)) / ((a^2) / ((a^2) - 1)) = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_simplify_evaluate_expression_l71_7188


namespace NUMINAMATH_GPT_equation_is_hyperbola_l71_7135

-- Define the equation
def equation (x y : ℝ) : Prop :=
  4 * x^2 - 9 * y^2 + 3 * x = 0

-- Theorem stating that the given equation represents a hyperbola
theorem equation_is_hyperbola : ∀ x y : ℝ, equation x y → (∃ A B : ℝ, A * x^2 - B * y^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_equation_is_hyperbola_l71_7135


namespace NUMINAMATH_GPT_scott_monthly_miles_l71_7156

theorem scott_monthly_miles :
  let miles_per_mon_wed := 3
  let mon_wed_days := 3
  let thur_fri_factor := 2
  let thur_fri_days := 2
  let weeks_per_month := 4
  let miles_mon_wed := miles_per_mon_wed * mon_wed_days
  let miles_thur_fri_per_day := thur_fri_factor * miles_per_mon_wed
  let miles_thur_fri := miles_thur_fri_per_day * thur_fri_days
  let miles_per_week := miles_mon_wed + miles_thur_fri
  let total_miles_in_month := miles_per_week * weeks_per_month
  total_miles_in_month = 84 := 
  by
    sorry

end NUMINAMATH_GPT_scott_monthly_miles_l71_7156


namespace NUMINAMATH_GPT_arithmetic_expression_equals_47_l71_7107

-- Define the arithmetic expression
def arithmetic_expression : ℕ :=
  2 + 5 * 3^2 - 4 + 6 * 2 / 3

-- The proof goal: arithmetic_expression equals 47
theorem arithmetic_expression_equals_47 : arithmetic_expression = 47 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_equals_47_l71_7107


namespace NUMINAMATH_GPT_mouse_lives_difference_l71_7178

-- Definitions of variables and conditions
def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := 13

-- Theorem to prove
theorem mouse_lives_difference : mouse_lives - dog_lives = 7 := by
  -- This is where the proof would go, but we use sorry to skip it.
  sorry

end NUMINAMATH_GPT_mouse_lives_difference_l71_7178


namespace NUMINAMATH_GPT_find_a1_a7_l71_7190

variable {a n : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ k n, a (k + n) = a k + n * d

theorem find_a1_a7 
  (a1 : ℝ) (d : ℝ)
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a d)
  (h1 : a 3 + a 5 = 14)
  (h2 : a 2 * a 6 = 33) :
  a 1 * a 7 = 13 := 
sorry

end NUMINAMATH_GPT_find_a1_a7_l71_7190


namespace NUMINAMATH_GPT_ana_wins_l71_7193

-- Define the game conditions and state
def game_conditions (n : ℕ) (m : ℕ) : Prop :=
  n < m ∧ m < n^2 ∧ Nat.gcd n m = 1

-- Define the losing condition
def losing_condition (n : ℕ) : Prop :=
  n >= 2016

-- Define the predicate for Ana having a winning strategy
def ana_winning_strategy : Prop :=
  ∃ (strategy : ℕ → ℕ), strategy 3 = 5 ∧
  (∀ n, (¬ losing_condition n) → (losing_condition (strategy n)))

theorem ana_wins : ana_winning_strategy :=
  sorry

end NUMINAMATH_GPT_ana_wins_l71_7193


namespace NUMINAMATH_GPT_remainder_17_pow_2037_mod_20_l71_7186

theorem remainder_17_pow_2037_mod_20:
      (17^1) % 20 = 17 ∧
      (17^2) % 20 = 9 ∧
      (17^3) % 20 = 13 ∧
      (17^4) % 20 = 1 → 
      (17^2037) % 20 = 17 := sorry

end NUMINAMATH_GPT_remainder_17_pow_2037_mod_20_l71_7186


namespace NUMINAMATH_GPT_maintain_income_with_new_demand_l71_7174

variable (P D : ℝ) -- Original Price and Demand
def new_price := 1.20 * P -- New Price after 20% increase
def new_demand := 1.12 * D -- New Demand after 12% increase due to advertisement
def original_income := P * D -- Original income
def new_income := new_price * new_demand -- New income after changes

theorem maintain_income_with_new_demand :
  ∀ P D : ℝ, P * D = 1.20 * P * 1.12 * (D_new : ℝ) → (D_new = 14/15 * D) :=
by
  intro P D h
  sorry

end NUMINAMATH_GPT_maintain_income_with_new_demand_l71_7174


namespace NUMINAMATH_GPT_frank_eats_each_day_l71_7115

theorem frank_eats_each_day :
  ∀ (cookies_per_tray cookies_per_day days ted_eats remaining_cookies : ℕ),
  cookies_per_tray = 12 →
  cookies_per_day = 2 →
  days = 6 →
  ted_eats = 4 →
  remaining_cookies = 134 →
  (2 * cookies_per_tray * days) - (ted_eats + remaining_cookies) / days = 1 :=
  by
    intros cookies_per_tray cookies_per_day days ted_eats remaining_cookies ht hc hd hted hr
    sorry

end NUMINAMATH_GPT_frank_eats_each_day_l71_7115


namespace NUMINAMATH_GPT_angle_B_degrees_l71_7143

theorem angle_B_degrees (A B C : ℕ) (h1 : A < B) (h2 : B < C) (h3 : 4 * C = 7 * A) (h4 : A + B + C = 180) : B = 59 :=
sorry

end NUMINAMATH_GPT_angle_B_degrees_l71_7143


namespace NUMINAMATH_GPT_num_of_three_digit_integers_greater_than_217_l71_7189

theorem num_of_three_digit_integers_greater_than_217 : 
  ∃ n : ℕ, n = 82 ∧ ∀ x : ℕ, (217 < x ∧ x < 300) → 200 ≤ x ∧ x ≤ 299 → n = 82 := 
by
  sorry

end NUMINAMATH_GPT_num_of_three_digit_integers_greater_than_217_l71_7189


namespace NUMINAMATH_GPT_integer_solutions_of_equation_l71_7195

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ (x = -7 ∧ y = -99) ∨ (x = -1 ∧ y = -9) ∨ (x = 1 ∧ y = 5) ∨ (x = 7 ∧ y = -97) := by 
  sorry

end NUMINAMATH_GPT_integer_solutions_of_equation_l71_7195


namespace NUMINAMATH_GPT_square_area_l71_7130

theorem square_area (side_length : ℝ) (h : side_length = 10) : side_length * side_length = 100 := by
  sorry

end NUMINAMATH_GPT_square_area_l71_7130


namespace NUMINAMATH_GPT_Jackson_game_time_l71_7196

/-- Jackson's grade increases by 15 points for every hour he spends studying, 
    and his grade is 45 points, prove that he spends 9 hours playing video 
    games when he spends 3 hours studying and 1/3 of his study time on 
    playing video games. -/
theorem Jackson_game_time (S G : ℕ) (h1 : 15 * S = 45) (h2 : G = 3 * S) : G = 9 :=
by
  sorry

end NUMINAMATH_GPT_Jackson_game_time_l71_7196


namespace NUMINAMATH_GPT_longest_side_of_similar_triangle_l71_7199

-- Define the sides of the original triangle
def a : ℕ := 8
def b : ℕ := 10
def c : ℕ := 12

-- Define the perimeter of the similar triangle
def perimeter_similar_triangle : ℕ := 150

-- Formalize the problem using Lean statement
theorem longest_side_of_similar_triangle :
  ∃ x : ℕ, 8 * x + 10 * x + 12 * x = 150 ∧ 12 * x = 60 :=
by
  sorry

end NUMINAMATH_GPT_longest_side_of_similar_triangle_l71_7199


namespace NUMINAMATH_GPT_darius_age_is_8_l71_7159

def age_of_darius (jenna_age darius_age : ℕ) : Prop :=
  jenna_age = darius_age + 5

theorem darius_age_is_8 (jenna_age darius_age : ℕ) (h1 : jenna_age = darius_age + 5) (h2: jenna_age = 13) : 
  darius_age = 8 :=
by
  sorry

end NUMINAMATH_GPT_darius_age_is_8_l71_7159


namespace NUMINAMATH_GPT_min_value_of_expr_l71_7197

def expr (x y : ℝ) : ℝ := 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10

theorem min_value_of_expr : ∃ x y : ℝ, expr x y = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expr_l71_7197


namespace NUMINAMATH_GPT_ken_pencils_kept_l71_7131

-- Define the known quantities and conditions
def initial_pencils : ℕ := 250
def manny_pencils : ℕ := 25
def nilo_pencils : ℕ := manny_pencils * 2
def carlos_pencils : ℕ := nilo_pencils / 2
def tina_pencils : ℕ := carlos_pencils + 10
def rina_pencils : ℕ := tina_pencils - 20

-- Formulate the total pencils given away
def total_given_away : ℕ :=
  manny_pencils + nilo_pencils + carlos_pencils + tina_pencils + rina_pencils

-- Prove the final number of pencils Ken kept.
theorem ken_pencils_kept : initial_pencils - total_given_away = 100 :=
by
  sorry

end NUMINAMATH_GPT_ken_pencils_kept_l71_7131


namespace NUMINAMATH_GPT_min_ab_min_inv_a_plus_2_inv_b_max_sqrt_2a_plus_sqrt_b_not_max_a_plus_1_times_b_plus_1_l71_7150

-- Condition definitions
variable {a b : ℝ}
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1)

-- Minimum value of ab is 1/8
theorem min_ab (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = (a * b) ∧ y = 1 / 8 := by
  sorry

-- Minimum value of 1/a + 2/b is 8
theorem min_inv_a_plus_2_inv_b (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = (1 / a + 2 / b) ∧ y = 8 := by
  sorry

-- Maximum value of sqrt(2a) + sqrt(b) is sqrt(2)
theorem max_sqrt_2a_plus_sqrt_b (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = (Real.sqrt (2 * a) + Real.sqrt b) ∧ y = Real.sqrt 2 := by
  sorry

-- Maximum value of (a+1)(b+1) is not 2
theorem not_max_a_plus_1_times_b_plus_1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = ((a + 1) * (b + 1)) ∧ y ≠ 2 := by
  sorry


end NUMINAMATH_GPT_min_ab_min_inv_a_plus_2_inv_b_max_sqrt_2a_plus_sqrt_b_not_max_a_plus_1_times_b_plus_1_l71_7150


namespace NUMINAMATH_GPT_integer_roots_polynomial_l71_7106

theorem integer_roots_polynomial (a : ℤ) :
  (∃ x : ℤ, x^3 + 3 * x^2 + a * x + 9 = 0) ↔ 
  (a = -109 ∨ a = -21 ∨ a = -13 ∨ a = 3 ∨ a = 11 ∨ a = 53) :=
by
  sorry

end NUMINAMATH_GPT_integer_roots_polynomial_l71_7106


namespace NUMINAMATH_GPT_chemistry_class_size_l71_7180

theorem chemistry_class_size
  (total_students : ℕ)
  (chem_bio_both : ℕ)
  (bio_students : ℕ)
  (chem_students : ℕ)
  (both_students : ℕ)
  (H1 : both_students = 8)
  (H2 : bio_students + chem_students + both_students = total_students)
  (H3 : total_students = 70)
  (H4 : chem_students = 2 * (bio_students + both_students)) :
  chem_students + both_students = 52 :=
by
  sorry

end NUMINAMATH_GPT_chemistry_class_size_l71_7180


namespace NUMINAMATH_GPT_find_prices_max_sets_of_go_compare_options_l71_7169

theorem find_prices (x y : ℕ) (h1 : 2 * x + 3 * y = 140) (h2 : 4 * x + y = 130) :
  x = 25 ∧ y = 30 :=
by sorry

theorem max_sets_of_go (m : ℕ) (h3 : 25 * (80 - m) + 30 * m ≤ 2250) :
  m ≤ 50 :=
by sorry

theorem compare_options (a : ℕ) :
  (a < 10 → 27 * a < 21 * a + 60) ∧ (a = 10 → 27 * a = 21 * a + 60) ∧ (a > 10 → 27 * a > 21 * a + 60) :=
by sorry

end NUMINAMATH_GPT_find_prices_max_sets_of_go_compare_options_l71_7169


namespace NUMINAMATH_GPT_problem_l71_7179

theorem problem (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 5) 
: (5 * m * r - 2 * n * t) / (7 * n * t - 10 * m * r) = -31 / 56 := 
sorry

end NUMINAMATH_GPT_problem_l71_7179


namespace NUMINAMATH_GPT_quadratic_completion_l71_7118

theorem quadratic_completion (a b : ℤ) (h_eq : (x : ℝ) → x^2 - 10 * x + 25 = 0) :
  (∃ a b : ℤ, ∀ x : ℝ, (x + a) ^ 2 = b) → a + b = -5 := by
  sorry

end NUMINAMATH_GPT_quadratic_completion_l71_7118


namespace NUMINAMATH_GPT_cloth_total_selling_price_l71_7191

theorem cloth_total_selling_price
    (meters : ℕ) (profit_per_meter cost_price_per_meter : ℝ) :
    meters = 92 →
    profit_per_meter = 24 →
    cost_price_per_meter = 83.5 →
    (cost_price_per_meter + profit_per_meter) * meters = 9890 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cloth_total_selling_price_l71_7191


namespace NUMINAMATH_GPT_range_x_l71_7129

variable {R : Type*} [LinearOrderedField R]

def monotone_increasing_on (f : R → R) (s : Set R) := ∀ ⦃a b⦄, a ≤ b → f a ≤ f b

theorem range_x 
    (f : R → R) 
    (h_mono : monotone_increasing_on f Set.univ) 
    (h_zero : f 1 = 0) 
    (h_ineq : ∀ x, f (x^2 + 3 * x - 3) < 0) :
  ∀ x, -4 < x ∧ x < 1 :=
by 
  sorry

end NUMINAMATH_GPT_range_x_l71_7129


namespace NUMINAMATH_GPT_cost_per_trip_l71_7142

theorem cost_per_trip
  (pass_cost : ℝ)
  (oldest_trips : ℕ)
  (youngest_trips : ℕ)
  (h_pass_cost : pass_cost = 100.0)
  (h_oldest_trips : oldest_trips = 35)
  (h_youngest_trips : youngest_trips = 15) :
  (2 * pass_cost) / (oldest_trips + youngest_trips) = 4.0 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_trip_l71_7142


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l71_7164

theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
                              (h3 : 2 * a = 4) (h4 : 2 * b = 6) : 
                              ∀ x y : ℝ, (y = (3 / 2) * x) ∨ (y = - (3 / 2) * x) := by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l71_7164


namespace NUMINAMATH_GPT_find_p_l71_7182

-- Define the conditions for the problem.
-- Random variable \xi follows binomial distribution B(n, p).
axiom binomial_distribution (n : ℕ) (p : ℝ) : Type
variables (ξ : binomial_distribution n p)

-- Given conditions: Eξ = 300 and Dξ = 200.
axiom Eξ (ξ : binomial_distribution n p) : ℝ
axiom Dξ (ξ : binomial_distribution n p) : ℝ

-- Given realizations of expectations and variance.
axiom h1 : Eξ ξ = 300
axiom h2 : Dξ ξ = 200

-- Prove that p = 1/3
theorem find_p (n : ℕ) (p : ℝ) (ξ : binomial_distribution n p)
  (h1 : Eξ ξ = 300) (h2 : Dξ ξ = 200) : p = 1 / 3 :=
sorry

end NUMINAMATH_GPT_find_p_l71_7182


namespace NUMINAMATH_GPT_problem_statement_l71_7134

variables (u v w : ℝ)

theorem problem_statement (h₁: u + v + w = 3) : 
  (1 / (u^2 + 7) + 1 / (v^2 + 7) + 1 / (w^2 + 7) ≤ 3 / 8) :=
sorry

end NUMINAMATH_GPT_problem_statement_l71_7134


namespace NUMINAMATH_GPT_sequence_n_500_l71_7152

theorem sequence_n_500 (a : ℕ → ℤ) 
  (h1 : a 1 = 1010) 
  (h2 : a 2 = 1011) 
  (h3 : ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = 2 * n + 3) : 
  a 500 = 3003 := 
sorry

end NUMINAMATH_GPT_sequence_n_500_l71_7152


namespace NUMINAMATH_GPT_sufficient_p_wages_l71_7149

variable (S P Q : ℕ)

theorem sufficient_p_wages (h1 : S = 40 * Q) (h2 : S = 15 * (P + Q))  :
  ∃ D : ℕ, S = D * P ∧ D = 24 := 
by
  use 24
  sorry

end NUMINAMATH_GPT_sufficient_p_wages_l71_7149


namespace NUMINAMATH_GPT_gcd_factorial_8_10_l71_7145

theorem gcd_factorial_8_10 : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_GPT_gcd_factorial_8_10_l71_7145


namespace NUMINAMATH_GPT_total_matches_l71_7172

theorem total_matches (home_wins home_draws home_losses rival_wins rival_draws rival_losses : ℕ)
  (H_home_wins : home_wins = 3)
  (H_home_draws : home_draws = 4)
  (H_home_losses : home_losses = 0)
  (H_rival_wins : rival_wins = 2 * home_wins)
  (H_rival_draws : rival_draws = 4)
  (H_rival_losses : rival_losses = 0) :
  home_wins + home_draws + home_losses + rival_wins + rival_draws + rival_losses = 17 :=
by
  sorry

end NUMINAMATH_GPT_total_matches_l71_7172


namespace NUMINAMATH_GPT_goose_eggs_count_l71_7168

theorem goose_eggs_count (E : ℕ) (h1 : E % 3 = 0) 
(h2 : ((4 / 5) * (1 / 3) * E) * (2 / 5) = 120) : E = 1125 := 
sorry

end NUMINAMATH_GPT_goose_eggs_count_l71_7168


namespace NUMINAMATH_GPT_solve_for_n_l71_7165

theorem solve_for_n (n : ℤ) (h : (1 : ℤ) / (n + 2) + 2 / (n + 2) + n / (n + 2) = 2) : n = 2 :=
sorry

end NUMINAMATH_GPT_solve_for_n_l71_7165


namespace NUMINAMATH_GPT_prove_equation_C_l71_7136

theorem prove_equation_C (m : ℝ) : -(m - 2) = -m + 2 := 
  sorry

end NUMINAMATH_GPT_prove_equation_C_l71_7136


namespace NUMINAMATH_GPT_min_negative_numbers_l71_7117

theorem min_negative_numbers (a b c d : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a + b + c < d) (h6 : a + b + d < c) (h7 : a + c + d < b) (h8 : b + c + d < a) :
  3 ≤ (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) + (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0) := 
sorry

end NUMINAMATH_GPT_min_negative_numbers_l71_7117


namespace NUMINAMATH_GPT_range_of_a_for_monotonically_decreasing_l71_7128

noncomputable def f (a x: ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - 2 * x

theorem range_of_a_for_monotonically_decreasing (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (1/x - a*x - 2 < 0)) ↔ (a ∈ Set.Ioi (-1)) := 
sorry

end NUMINAMATH_GPT_range_of_a_for_monotonically_decreasing_l71_7128


namespace NUMINAMATH_GPT_wang_pens_purchase_l71_7184

theorem wang_pens_purchase :
  ∀ (total_money spent_on_albums pen_cost : ℝ)
  (number_of_pens : ℕ),
  total_money = 80 →
  spent_on_albums = 45.6 →
  pen_cost = 2.5 →
  number_of_pens = 13 →
  (total_money - spent_on_albums) / pen_cost ≥ number_of_pens ∧ 
  (total_money - spent_on_albums) / pen_cost < number_of_pens + 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_wang_pens_purchase_l71_7184


namespace NUMINAMATH_GPT_range_of_a_l71_7101

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → x^2 - 2*x + a ≥ 0) → a ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l71_7101


namespace NUMINAMATH_GPT_subset_intersection_exists_l71_7173

theorem subset_intersection_exists {n : ℕ} (A : Fin (n + 1) → Finset (Fin n)) 
    (h_distinct : ∀ i j : Fin (n + 1), i ≠ j → A i ≠ A j)
    (h_size : ∀ i : Fin (n + 1), (A i).card = 3) : 
    ∃ (i j : Fin (n + 1)), i ≠ j ∧ (A i ∩ A j).card = 1 :=
by
  sorry

end NUMINAMATH_GPT_subset_intersection_exists_l71_7173


namespace NUMINAMATH_GPT_half_sum_of_squares_l71_7154

theorem half_sum_of_squares (n m : ℕ) (h : n ≠ m) :
  ∃ a b : ℕ, ( (2 * n)^2 + (2 * m)^2) / 2 = a^2 + b^2 := by
  sorry

end NUMINAMATH_GPT_half_sum_of_squares_l71_7154


namespace NUMINAMATH_GPT_max_expr_on_circle_l71_7140

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 4 * x - 6 * y + 4 = 0

noncomputable def expr (x y : ℝ) : ℝ :=
  3 * x - 4 * y

theorem max_expr_on_circle : 
  ∃ (x y : ℝ), circle_eq x y ∧ ∀ (x' y' : ℝ), circle_eq x' y' → expr x y ≤ expr x' y' :=
sorry

end NUMINAMATH_GPT_max_expr_on_circle_l71_7140


namespace NUMINAMATH_GPT_factorize_expression_l71_7170

theorem factorize_expression (a b : ℤ) (h1 : 3 * b + a = -1) (h2 : a * b = -18) : a - b = -11 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l71_7170


namespace NUMINAMATH_GPT_total_receipts_l71_7121

theorem total_receipts 
  (x y : ℕ) 
  (h1 : x + y = 64)
  (h2 : y ≥ 8) 
  : 3 * x + 4 * y = 200 := 
by
  sorry

end NUMINAMATH_GPT_total_receipts_l71_7121


namespace NUMINAMATH_GPT_taxi_ride_cost_l71_7105

-- Definitions
def initial_fee : Real := 2
def distance_in_miles : Real := 4
def cost_per_mile : Real := 2.5

-- Theorem statement
theorem taxi_ride_cost (initial_fee : Real) (distance_in_miles : Real) (cost_per_mile : Real) : 
  initial_fee + distance_in_miles * cost_per_mile = 12 := 
by
  sorry

end NUMINAMATH_GPT_taxi_ride_cost_l71_7105


namespace NUMINAMATH_GPT_min_value_of_m_cauchy_schwarz_inequality_l71_7122

theorem min_value_of_m (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m = a + 1 / ((a - b) * b)) : 
  ∃ t, t = 3 ∧ ∀ a b : ℝ, a > b → b > 0 → m = a + 1 / ((a - b) * b) → m ≥ t :=
sorry

theorem cauchy_schwarz_inequality (x y z : ℝ) :
  (x^2 + 4 * y^2 + z^2 = 3) → |x + 2 * y + z| ≤ 3 :=
sorry

end NUMINAMATH_GPT_min_value_of_m_cauchy_schwarz_inequality_l71_7122


namespace NUMINAMATH_GPT_problem_l71_7198

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def m : ℝ := sorry
noncomputable def p : ℝ := sorry
noncomputable def r : ℝ := sorry

theorem problem
  (h1 : a^2 - m*a + 3 = 0)
  (h2 : b^2 - m*b + 3 = 0)
  (h3 : a * b = 3)
  (h4 : ∀ x, x^2 - p * x + r = (x - (a + 1 / b)) * (x - (b + 1 / a))) :
  r = 16 / 3 :=
sorry

end NUMINAMATH_GPT_problem_l71_7198


namespace NUMINAMATH_GPT_min_value_at_2_l71_7102

noncomputable def min_value (x : ℝ) := x + 4 / x + 5

theorem min_value_at_2 (x : ℝ) (h : x > 0) : min_value x ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_at_2_l71_7102


namespace NUMINAMATH_GPT_closest_point_on_line_is_correct_l71_7157

theorem closest_point_on_line_is_correct :
  ∃ (p : ℝ × ℝ), p = (-0.04, -0.28) ∧
  ∃ x : ℝ, p = (x, (3 * x - 1) / 4) ∧
  ∀ q : ℝ × ℝ, (q = (x, (3 * x - 1) / 4) → 
  (dist (2, -3) p) ≤ (dist (2, -3) q)) :=
sorry

end NUMINAMATH_GPT_closest_point_on_line_is_correct_l71_7157


namespace NUMINAMATH_GPT_smallest_n_for_quadratic_factorization_l71_7116

theorem smallest_n_for_quadratic_factorization :
  ∃ (n : ℤ), (∀ A B : ℤ, A * B = 50 → n = 5 * B + A) ∧ (∀ m : ℤ, 
    (∀ A B : ℤ, A * B = 50 → m ≤ 5 * B + A) → n ≤ m) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_quadratic_factorization_l71_7116


namespace NUMINAMATH_GPT_find_value_of_f_l71_7161

axiom f : ℝ → ℝ

theorem find_value_of_f :
  (∀ x : ℝ, f (Real.cos x) = Real.sin (3 * x)) →
  f (Real.sin (Real.pi / 9)) = -1 / 2 :=
sorry

end NUMINAMATH_GPT_find_value_of_f_l71_7161


namespace NUMINAMATH_GPT_cos_beta_eq_neg_16_over_65_l71_7112

variable (α β : ℝ)
variable (h1 : 0 < α ∧ α < π)
variable (h2 : 0 < β ∧ β < π)
variable (h3 : Real.sin β = 5 / 13)
variable (h4 : Real.tan (α / 2) = 1 / 2)

theorem cos_beta_eq_neg_16_over_65 : Real.cos β = -16 / 65 := by
  sorry

end NUMINAMATH_GPT_cos_beta_eq_neg_16_over_65_l71_7112


namespace NUMINAMATH_GPT_solve_arithmetic_sequence_l71_7139

theorem solve_arithmetic_sequence (y : ℝ) (h : 0 < y) (h_arith : ∃ (d : ℝ), 4 + d = y^2 ∧ y^2 + d = 16 ∧ 16 + d = 36) :
  y = Real.sqrt 10 := by
  sorry

end NUMINAMATH_GPT_solve_arithmetic_sequence_l71_7139


namespace NUMINAMATH_GPT_point_equidistant_x_axis_y_axis_line_l71_7100

theorem point_equidistant_x_axis_y_axis_line (x y : ℝ) (h1 : abs y = abs x) (h2 : abs (x + y - 2) / Real.sqrt 2 = abs x) :
  x = 1 :=
  sorry

end NUMINAMATH_GPT_point_equidistant_x_axis_y_axis_line_l71_7100


namespace NUMINAMATH_GPT_value_range_of_sum_difference_l71_7137

theorem value_range_of_sum_difference (a b c : ℝ) (h₁ : a < b)
  (h₂ : a + b = b / a) (h₃ : a * b = c / a) (h₄ : a + b > c)
  (h₅ : a + c > b) (h₆ : b + c > a) : 
  ∃ x y, x = 7 / 8 ∧ y = Real.sqrt 5 - 1 ∧ x < a + b - c ∧ a + b - c < y := sorry

end NUMINAMATH_GPT_value_range_of_sum_difference_l71_7137


namespace NUMINAMATH_GPT_revenue_increase_l71_7110

theorem revenue_increase
  (P Q : ℝ)
  (h : 0 < P)
  (hQ : 0 < Q)
  (price_decrease : 0.90 = 0.90)
  (unit_increase : 2 = 2) :
  (0.90 * P) * (2 * Q) = 1.80 * (P * Q) :=
by
  sorry

end NUMINAMATH_GPT_revenue_increase_l71_7110


namespace NUMINAMATH_GPT_sum_of_sides_eq_l71_7167

open Real

theorem sum_of_sides_eq (a h : ℝ) (α : ℝ) (ha : a > 0) (hh : h > 0) (hα : 0 < α ∧ α < π) :
  ∃ b c : ℝ, b + c = sqrt (a^2 + 2 * a * h * (cos (α / 2) / sin (α / 2))) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_sides_eq_l71_7167


namespace NUMINAMATH_GPT_numPeopleToLeftOfKolya_l71_7175

-- Definitions based on the conditions.
def peopleToRightOfKolya := 12
def peopleToLeftOfSasha := 20
def peopleToRightOfSasha := 8

-- Theorem statement with the given conditions and conclusion.
theorem numPeopleToLeftOfKolya 
  (h1 : peopleToRightOfKolya = 12)
  (h2 : peopleToLeftOfSasha = 20)
  (h3 : peopleToRightOfSasha = 8) :
  ∃ n, n = 16 :=
by
  -- Proving the theorem will be done here.
  sorry

end NUMINAMATH_GPT_numPeopleToLeftOfKolya_l71_7175


namespace NUMINAMATH_GPT_find_q_l71_7141

theorem find_q (p q : ℚ) (h1 : 5 * p + 7 * q = 20) (h2 : 7 * p + 5 * q = 26) : q = 5 / 12 := by
  sorry

end NUMINAMATH_GPT_find_q_l71_7141


namespace NUMINAMATH_GPT_arithmetic_sequence_l71_7119

theorem arithmetic_sequence (n : ℕ) (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n + 1) : 
  ∀ n, a (n + 1) - a n = 3 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_l71_7119


namespace NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l71_7176

theorem quadratic_has_distinct_real_roots (q : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x^2 + 8 * x + q = 0) ↔ q < 16 :=
by
  -- only the statement is provided, the proof is omitted
  sorry

end NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l71_7176


namespace NUMINAMATH_GPT_dana_total_earnings_l71_7155

-- Define the constants for Dana's hourly rate and hours worked each day
def hourly_rate : ℝ := 13
def friday_hours : ℝ := 9
def saturday_hours : ℝ := 10
def sunday_hours : ℝ := 3

-- Define the total earnings calculation function
def total_earnings (rate : ℝ) (hours1 hours2 hours3 : ℝ) : ℝ :=
  rate * hours1 + rate * hours2 + rate * hours3

-- The main statement
theorem dana_total_earnings : total_earnings hourly_rate friday_hours saturday_hours sunday_hours = 286 := by
  sorry

end NUMINAMATH_GPT_dana_total_earnings_l71_7155


namespace NUMINAMATH_GPT_comparison_of_square_roots_l71_7146

theorem comparison_of_square_roots (P Q : ℝ) (hP : P = Real.sqrt 2) (hQ : Q = Real.sqrt 6 - Real.sqrt 2) : P > Q :=
by
  sorry

end NUMINAMATH_GPT_comparison_of_square_roots_l71_7146


namespace NUMINAMATH_GPT_workbook_problems_l71_7153

theorem workbook_problems (P : ℕ)
  (h1 : (1/2 : ℚ) * P = (1/2 : ℚ) * P)
  (h2 : (1/4 : ℚ) * P = (1/4 : ℚ) * P)
  (h3 : (1/6 : ℚ) * P = (1/6 : ℚ) * P)
  (h4 : ((1/2 : ℚ) * P + (1/4 : ℚ) * P + (1/6 : ℚ) * P + 20 = P)) : 
  P = 240 :=
sorry

end NUMINAMATH_GPT_workbook_problems_l71_7153


namespace NUMINAMATH_GPT_mila_needs_48_hours_to_earn_as_much_as_agnes_l71_7192

/-- Definition of the hourly wage for the babysitters and the working hours of Agnes. -/
def mila_hourly_wage : ℝ := 10
def agnes_hourly_wage : ℝ := 15
def agnes_weekly_hours : ℝ := 8
def weeks_in_month : ℝ := 4

/-- Mila needs to work 48 hours in a month to earn as much as Agnes. -/
theorem mila_needs_48_hours_to_earn_as_much_as_agnes :
  ∃ (mila_monthly_hours : ℝ), mila_monthly_hours = 48 ∧ 
  mila_hourly_wage * mila_monthly_hours = agnes_hourly_wage * agnes_weekly_hours * weeks_in_month := 
sorry

end NUMINAMATH_GPT_mila_needs_48_hours_to_earn_as_much_as_agnes_l71_7192


namespace NUMINAMATH_GPT_petri_dishes_count_l71_7194

def germs_total : ℕ := 5400000
def germs_per_dish : ℕ := 500
def petri_dishes : ℕ := germs_total / germs_per_dish

theorem petri_dishes_count : petri_dishes = 10800 := by
  sorry

end NUMINAMATH_GPT_petri_dishes_count_l71_7194


namespace NUMINAMATH_GPT_compare_f_values_max_f_value_l71_7103

noncomputable def f (x : ℝ) : ℝ := -2 * Real.sin x - Real.cos (2 * x)

theorem compare_f_values :
  f (Real.pi / 4) > f (Real.pi / 6) :=
sorry

theorem max_f_value :
  ∃ x : ℝ, f x = 3 :=
sorry

end NUMINAMATH_GPT_compare_f_values_max_f_value_l71_7103


namespace NUMINAMATH_GPT_delivery_driver_stops_l71_7108

theorem delivery_driver_stops (initial_stops more_stops total_stops : ℕ)
  (h_initial : initial_stops = 3)
  (h_more : more_stops = 4)
  (h_total : total_stops = initial_stops + more_stops) : total_stops = 7 := by
  sorry

end NUMINAMATH_GPT_delivery_driver_stops_l71_7108


namespace NUMINAMATH_GPT_not_solvable_equations_l71_7183

theorem not_solvable_equations :
  ¬(∃ x : ℝ, (x - 5) ^ 2 = -1) ∧ ¬(∃ x : ℝ, |2 * x| + 3 = 0) :=
by
  sorry

end NUMINAMATH_GPT_not_solvable_equations_l71_7183


namespace NUMINAMATH_GPT_value_of_af_over_cd_l71_7138

variable (a b c d e f : ℝ)

theorem value_of_af_over_cd :
  a * b * c = 130 ∧
  b * c * d = 65 ∧
  c * d * e = 500 ∧
  d * e * f = 250 →
  (a * f) / (c * d) = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_af_over_cd_l71_7138


namespace NUMINAMATH_GPT_puppy_sleep_duration_l71_7132

-- Definitions based on conditions
def connor_sleep : ℕ := 6
def luke_sleep : ℕ := connor_sleep + 2
def puppy_sleep : ℕ := 2 * luke_sleep

-- Theorem stating that the puppy sleeps for 16 hours
theorem puppy_sleep_duration : puppy_sleep = 16 := by
  sorry

end NUMINAMATH_GPT_puppy_sleep_duration_l71_7132


namespace NUMINAMATH_GPT_solution_set_l71_7187

-- Defining the condition and inequalities:
variable (a x : Real)

-- Condition that a < 0
def condition_a : Prop := a < 0

-- Inequalities in the system
def inequality1 : Prop := x > -2 * a
def inequality2 : Prop := x > 3 * a

-- The solution set we need to prove
theorem solution_set (h : condition_a a) : (inequality1 a x) ∧ (inequality2 a x) ↔ x > -2 * a :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l71_7187


namespace NUMINAMATH_GPT_remaining_movie_duration_l71_7133

/--
Given:
1. The laptop was fully charged at 3:20 pm.
2. Hannah started watching a 3-hour series.
3. The laptop turned off at 5:44 pm (fully discharged).

Prove:
The remaining duration of the movie Hannah needs to watch is 36 minutes.
-/
theorem remaining_movie_duration
    (start_full_charge : ℕ := 200)  -- representing 3:20 pm as 200 (20 minutes past 3:00)
    (end_discharge : ℕ := 344)  -- representing 5:44 pm as 344 (44 minutes past 5:00)
    (total_duration_minutes : ℕ := 180)  -- 3 hours in minutes
    (start_time_minutes : ℕ := 200)  -- convert 3:20 pm to minutes past noon
    (end_time_minutes : ℕ := 344)  -- convert 5:44 pm to minutes past noon
    : (total_duration_minutes - (end_time_minutes - start_time_minutes)) = 36 :=
by
  sorry

end NUMINAMATH_GPT_remaining_movie_duration_l71_7133


namespace NUMINAMATH_GPT_cone_volume_ratio_l71_7162

theorem cone_volume_ratio (rC hC rD hD : ℝ) (h_rC : rC = 10) (h_hC : hC = 20) (h_rD : rD = 20) (h_hD : hD = 10) :
  ((1/3) * π * rC^2 * hC) / ((1/3) * π * rD^2 * hD) = 1/2 :=
by 
  sorry

end NUMINAMATH_GPT_cone_volume_ratio_l71_7162


namespace NUMINAMATH_GPT_socks_total_l71_7114

def socks_lisa (initial: Nat) := initial + 0

def socks_sandra := 20

def socks_cousin (sandra: Nat) := sandra / 5

def socks_mom (initial: Nat) := 8 + 3 * initial

theorem socks_total (initial: Nat) (sandra: Nat) :
  initial = 12 → sandra = 20 → 
  socks_lisa initial + socks_sandra + socks_cousin sandra + socks_mom initial = 80 :=
by
  intros h_initial h_sandra
  rw [h_initial, h_sandra]
  sorry

end NUMINAMATH_GPT_socks_total_l71_7114


namespace NUMINAMATH_GPT_problem_solution_l71_7123

theorem problem_solution (x : ℝ) (N : ℝ) (h1 : 625 ^ (-x) + N ^ (-2 * x) + 5 ^ (-4 * x) = 11) (h2 : x = 0.25) :
  N = 25 / 2809 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l71_7123


namespace NUMINAMATH_GPT_kristin_reading_time_l71_7185

-- Definitions
def total_books : Nat := 20
def peter_time_per_book : ℕ := 18
def reading_speed_ratio : Nat := 3

-- Derived Definitions
def kristin_time_per_book : ℕ := peter_time_per_book * reading_speed_ratio
def kristin_books_to_read : Nat := total_books / 2
def kristin_total_time : ℕ := kristin_time_per_book * kristin_books_to_read

-- Statement to be proved
theorem kristin_reading_time :
  kristin_total_time = 540 :=
  by 
    -- Proof would go here, but we are only required to state the theorem
    sorry

end NUMINAMATH_GPT_kristin_reading_time_l71_7185


namespace NUMINAMATH_GPT_smallest_n_value_existence_l71_7148

-- Define a three-digit positive integer n such that the conditions hold
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def problem_conditions (n : ℕ) : Prop :=
  n % 9 = 3 ∧ n % 6 = 3

-- Main statement: There exists a three-digit positive integer n satisfying the conditions and is equal to 111
theorem smallest_n_value_existence : ∃ n : ℕ, is_three_digit n ∧ problem_conditions n ∧ n = 111 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_value_existence_l71_7148


namespace NUMINAMATH_GPT_prove_a_plus_b_l71_7127

-- Defining the function f(x)
def f (a b x: ℝ) : ℝ := a * x^2 + b * x

-- The given conditions
variable (a b : ℝ)
variable (h1 : f a b (a - 1) = f a b (2 * a))
variable (h2 : ∀ x : ℝ, f a b x = f a b (-x))

-- The objective is to show a + b = 1/3
theorem prove_a_plus_b (a b : ℝ) (h1 : f a b (a - 1) = f a b (2 * a)) (h2 : ∀ x : ℝ, f a b x = f a b (-x)) :
  a + b = 1 / 3 := 
sorry

end NUMINAMATH_GPT_prove_a_plus_b_l71_7127


namespace NUMINAMATH_GPT_cannon_hit_probability_l71_7109

theorem cannon_hit_probability
  (P1 P2 P3 : ℝ)
  (h1 : P1 = 0.2)
  (h3 : P3 = 0.3)
  (h_none_hit : (1 - P1) * (1 - P2) * (1 - P3) = 0.27999999999999997) :
  P2 = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_cannon_hit_probability_l71_7109


namespace NUMINAMATH_GPT_correct_population_growth_pattern_statement_l71_7158

-- Definitions based on the conditions provided
def overall_population_growth_modern (world_population : ℕ) : Prop :=
  -- The overall pattern of population growth worldwide is already in the modern stage
  sorry

def transformation_synchronized (world_population : ℕ) : Prop :=
  -- The transformation of population growth patterns in countries or regions around the world is synchronized
  sorry

def developed_countries_transformed (world_population : ℕ) : Prop :=
  -- Developed countries have basically completed the transformation of population growth patterns
  sorry

def transformation_determined_by_population_size (world_population : ℕ) : Prop :=
  -- The process of transformation in population growth patterns is determined by the population size of each area
  sorry

-- The statement to be proven
theorem correct_population_growth_pattern_statement (world_population : ℕ) :
  developed_countries_transformed world_population := sorry

end NUMINAMATH_GPT_correct_population_growth_pattern_statement_l71_7158


namespace NUMINAMATH_GPT_g_1200_value_l71_7126

noncomputable def g : ℝ → ℝ := sorry

-- Assume the given condition as a definition
axiom functional_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y

-- Assume the given value of g(1000)
axiom g_1000_value : g 1000 = 4

-- Prove that g(1200) = 10/3
theorem g_1200_value : g 1200 = 10 / 3 := by
  sorry

end NUMINAMATH_GPT_g_1200_value_l71_7126


namespace NUMINAMATH_GPT_additional_number_is_31_l71_7144

theorem additional_number_is_31
(six_numbers_sum : ℕ)
(seven_numbers_avg : ℕ)
(h1 : six_numbers_sum = 144)
(h2 : seven_numbers_avg = 25)
: ∃ x : ℕ, ((six_numbers_sum + x) / 7 = 25) ∧ x = 31 := 
by
  sorry

end NUMINAMATH_GPT_additional_number_is_31_l71_7144
