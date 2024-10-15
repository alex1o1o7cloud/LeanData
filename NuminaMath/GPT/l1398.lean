import Mathlib

namespace NUMINAMATH_GPT_boys_or_girls_rink_l1398_139855

variables (Class : Type) (is_boy : Class → Prop) (is_girl : Class → Prop) (visited_rink : Class → Prop) (met_at_rink : Class → Class → Prop)

-- Every student in the class visited the rink at least once.
axiom all_students_visited : ∀ (s : Class), visited_rink s

-- Every boy met every girl at the rink.
axiom boys_meet_girls : ∀ (b g : Class), is_boy b → is_girl g → met_at_rink b g

-- Prove that there exists a time when all the boys, or all the girls were simultaneously on the rink.
theorem boys_or_girls_rink : ∃ (t : Prop), (∀ b, is_boy b → visited_rink b) ∨ (∀ g, is_girl g → visited_rink g) :=
sorry

end NUMINAMATH_GPT_boys_or_girls_rink_l1398_139855


namespace NUMINAMATH_GPT_part_I_part_II_l1398_139842

noncomputable def f (a : ℝ) (x : ℝ) := a * x - Real.log x
noncomputable def F (a : ℝ) (x : ℝ) := Real.exp x + a * x
noncomputable def g (a : ℝ) (x : ℝ) := x * Real.exp (a * x - 1) - 2 * a * x + f a x

def monotonicity_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < b ∧ a < y ∧ y < b ∧ x < y → f x < f y

theorem part_I (a : ℝ) : 
  monotonicity_in_interval (f a) 0 (Real.log 3) = monotonicity_in_interval (F a) 0 (Real.log 3) ↔ a ≤ -3 :=
sorry

theorem part_II (a : ℝ) (ha : a ∈ Set.Iic (-1 / Real.exp 2)) : 
  (∃ x, x > 0 ∧ g a x = M) → M ≥ 0 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l1398_139842


namespace NUMINAMATH_GPT_watch_cost_price_l1398_139849

theorem watch_cost_price (CP : ℝ) (H1 : 0.90 * CP = CP - 0.10 * CP)
(H2 : 1.04 * CP = CP + 0.04 * CP)
(H3 : 1.04 * CP - 0.90 * CP = 168) : CP = 1200 := by
sorry

end NUMINAMATH_GPT_watch_cost_price_l1398_139849


namespace NUMINAMATH_GPT_complex_identity_l1398_139893

open Complex

noncomputable def z := 1 + 2 * I
noncomputable def z_inv := (1 - 2 * I) / 5
noncomputable def z_conj := 1 - 2 * I

theorem complex_identity : 
  (z + z_inv) * z_conj = (22 / 5 : ℂ) - (4 / 5) * I := 
by
  sorry

end NUMINAMATH_GPT_complex_identity_l1398_139893


namespace NUMINAMATH_GPT_solve_linear_equation_l1398_139806

theorem solve_linear_equation (x : ℝ) (h : 2 * x - 1 = 1) : x = 1 :=
sorry

end NUMINAMATH_GPT_solve_linear_equation_l1398_139806


namespace NUMINAMATH_GPT_value_of_business_l1398_139863

theorem value_of_business 
  (ownership : ℚ)
  (sale_fraction : ℚ)
  (sale_value : ℚ) 
  (h_ownership : ownership = 2/3) 
  (h_sale_fraction : sale_fraction = 3/4) 
  (h_sale_value : sale_value = 6500) : 
  2 * sale_value = 13000 := 
by
  -- mathematical equivalent proof here
  -- This is a placeholder.
  sorry

end NUMINAMATH_GPT_value_of_business_l1398_139863


namespace NUMINAMATH_GPT_probability_YW_correct_l1398_139807

noncomputable def probability_YW_greater_than_six_sqrt_three (XY YZ XZ YW : ℝ) : ℝ :=
  if H : XY = 12 ∧ YZ = 6 ∧ XZ = 6 * Real.sqrt 3 then 
    if YW > 6 * Real.sqrt 3 then (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3
    else 0
  else 0

theorem probability_YW_correct : probability_YW_greater_than_six_sqrt_three 12 6 (6 * Real.sqrt 3) (6 * Real.sqrt 3) = (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_probability_YW_correct_l1398_139807


namespace NUMINAMATH_GPT_real_root_if_and_only_if_l1398_139829

theorem real_root_if_and_only_if (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end NUMINAMATH_GPT_real_root_if_and_only_if_l1398_139829


namespace NUMINAMATH_GPT_layla_more_than_nahima_l1398_139835

-- Definitions for the conditions
def layla_points : ℕ := 70
def total_points : ℕ := 112
def nahima_points : ℕ := total_points - layla_points

-- Statement of the theorem
theorem layla_more_than_nahima : layla_points - nahima_points = 28 :=
by
  sorry

end NUMINAMATH_GPT_layla_more_than_nahima_l1398_139835


namespace NUMINAMATH_GPT_jude_age_today_l1398_139881
-- Import the necessary libraries

-- Define the conditions as hypotheses and then state the required proof
theorem jude_age_today (heath_age_today : ℕ) (heath_age_in_5_years : ℕ) (jude_age_in_5_years : ℕ) 
  (H1 : heath_age_today = 16)
  (H2 : heath_age_in_5_years = heath_age_today + 5)
  (H3 : heath_age_in_5_years = 3 * jude_age_in_5_years) :
  jude_age_in_5_years - 5 = 2 :=
by
  -- Given conditions imply Jude's age today is 2. Proof is omitted.
  sorry

end NUMINAMATH_GPT_jude_age_today_l1398_139881


namespace NUMINAMATH_GPT_problem1_problem2_l1398_139891

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

-- Problem 1: (0 < m < 1/e) implies g(x) = f(x) - m has two zeros
theorem problem1 (m : ℝ) (h1 : 0 < m) (h2 : m < 1 / Real.exp 1) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = m ∧ f x2 = m :=
sorry

-- Problem 2: (2/e^2 ≤ a < 1/e) implies f^2(x) - af(x) > 0 has only one integer solution
theorem problem2 (a : ℝ) (h1 : 2 / (Real.exp 2) ≤ a) (h2 : a < 1 / Real.exp 1) :
  ∃! x : ℤ, ∀ y : ℤ, (f y)^2 - a * (f y) > 0 → y = x :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1398_139891


namespace NUMINAMATH_GPT_probability_yellow_second_l1398_139866

section MarbleProbabilities

def bag_A := (5, 6)     -- (white marbles, black marbles)
def bag_B := (3, 7)     -- (yellow marbles, blue marbles)
def bag_C := (5, 6)     -- (yellow marbles, blue marbles)

def P_white_A := 5 / 11
def P_black_A := 6 / 11
def P_yellow_given_B := 3 / 10
def P_yellow_given_C := 5 / 11

theorem probability_yellow_second :
  P_white_A * P_yellow_given_B + P_black_A * P_yellow_given_C = 33 / 121 :=
by
  -- Proof would be provided here
  sorry

end MarbleProbabilities

end NUMINAMATH_GPT_probability_yellow_second_l1398_139866


namespace NUMINAMATH_GPT_equation_elliptic_and_canonical_form_l1398_139877

-- Defining the necessary conditions and setup
def a11 := 1
def a12 := 1
def a22 := 2

def is_elliptic (a11 a12 a22 : ℝ) : Prop :=
  a12^2 - a11 * a22 < 0

def canonical_form (u_xx u_xy u_yy u_x u_y u x y : ℝ) : Prop :=
  let ξ := y - x
  let η := x
  let u_ξξ := u_xx -- Assuming u_xx represents u_ξξ after change of vars
  let u_ξη := u_xy
  let u_ηη := u_yy
  let u_ξ := u_x -- Assuming u_x represents u_ξ after change of vars
  let u_η := u_y
  u_ξξ + u_ηη = -2 * u_η + u + η + (ξ + η)^2

theorem equation_elliptic_and_canonical_form (u_xx u_xy u_yy u_x u_y u x y : ℝ) :
  is_elliptic a11 a12 a22 ∧
  canonical_form u_xx u_xy u_yy u_x u_y u x y :=
by
  sorry -- Proof to be completed

end NUMINAMATH_GPT_equation_elliptic_and_canonical_form_l1398_139877


namespace NUMINAMATH_GPT_gray_eyed_black_haired_students_l1398_139861

theorem gray_eyed_black_haired_students :
  ∀ (students : ℕ)
    (green_eyed_red_haired : ℕ)
    (black_haired : ℕ)
    (gray_eyed : ℕ),
    students = 60 →
    green_eyed_red_haired = 20 →
    black_haired = 40 →
    gray_eyed = 25 →
    (gray_eyed - (students - black_haired - green_eyed_red_haired)) = 25 := by
  intros students green_eyed_red_haired black_haired gray_eyed
  intros h_students h_green h_black h_gray
  sorry

end NUMINAMATH_GPT_gray_eyed_black_haired_students_l1398_139861


namespace NUMINAMATH_GPT_total_cost_toys_l1398_139826

variable (c_e_actionfigs : ℕ := 60) -- number of action figures for elder son
variable (cost_e_actionfig : ℕ := 5) -- cost per action figure for elder son
variable (c_y_actionfigs : ℕ := 3 * c_e_actionfigs) -- number of action figures for younger son
variable (cost_y_actionfig : ℕ := 4) -- cost per action figure for younger son
variable (c_y_cars : ℕ := 20) -- number of cars for younger son
variable (cost_car : ℕ := 3) -- cost per car
variable (c_y_animals : ℕ := 10) -- number of stuffed animals for younger son
variable (cost_animal : ℕ := 7) -- cost per stuffed animal

theorem total_cost_toys (c_e_actionfigs c_y_actionfigs c_y_cars c_y_animals : ℕ)
                         (cost_e_actionfig cost_y_actionfig cost_car cost_animal : ℕ) :
  (c_e_actionfigs * cost_e_actionfig + c_y_actionfigs * cost_y_actionfig + 
  c_y_cars * cost_car + c_y_animals * cost_animal) = 1150 := by
  sorry

end NUMINAMATH_GPT_total_cost_toys_l1398_139826


namespace NUMINAMATH_GPT_july_husband_current_age_l1398_139878

-- Define the initial ages and the relationship between Hannah and July's age
def hannah_initial_age : ℕ := 6
def hannah_july_age_relation (hannah_age july_age : ℕ) : Prop := hannah_age = 2 * july_age

-- Define the time that has passed and the age difference between July and her husband
def time_passed : ℕ := 20
def july_husband_age_relation (july_age husband_age : ℕ) : Prop := husband_age = july_age + 2

-- Lean statement to prove July's husband's current age
theorem july_husband_current_age : ∃ (july_age husband_age : ℕ),
  hannah_july_age_relation hannah_initial_age july_age ∧
  july_husband_age_relation (july_age + time_passed) husband_age ∧
  husband_age = 25 :=
by
  sorry

end NUMINAMATH_GPT_july_husband_current_age_l1398_139878


namespace NUMINAMATH_GPT_find_five_digit_number_l1398_139814

theorem find_five_digit_number
  (x y : ℕ)
  (h1 : 10 * y + x - (10000 * x + y) = 34767)
  (h2 : 10 * y + x + (10000 * x + y) = 86937) :
  10000 * x + y = 26035 := by
  sorry

end NUMINAMATH_GPT_find_five_digit_number_l1398_139814


namespace NUMINAMATH_GPT_option_C_qualified_l1398_139888

-- Define the acceptable range
def lower_bound : ℝ := 25 - 0.2
def upper_bound : ℝ := 25 + 0.2

-- Define the option to be checked
def option_C : ℝ := 25.1

-- The theorem stating that option C is within the acceptable range
theorem option_C_qualified : lower_bound ≤ option_C ∧ option_C ≤ upper_bound := 
by 
  sorry

end NUMINAMATH_GPT_option_C_qualified_l1398_139888


namespace NUMINAMATH_GPT_solve_inequality_l1398_139889

theorem solve_inequality (x : ℝ) : 2 * x ^ 2 - 7 * x - 30 < 0 ↔ - (5 / 2) < x ∧ x < 6 := 
sorry

end NUMINAMATH_GPT_solve_inequality_l1398_139889


namespace NUMINAMATH_GPT_average_weight_increase_l1398_139804

theorem average_weight_increase
  (initial_weight replaced_weight : ℝ)
  (num_persons : ℕ)
  (avg_increase : ℝ)
  (h₁ : num_persons = 5)
  (h₂ : replaced_weight = 65)
  (h₃ : avg_increase = 1.5)
  (total_increase : ℝ)
  (new_weight : ℝ)
  (h₄ : total_increase = num_persons * avg_increase)
  (h₅ : total_increase = new_weight - replaced_weight) :
  new_weight = 72.5 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_increase_l1398_139804


namespace NUMINAMATH_GPT_expected_length_after_2012_repetitions_l1398_139819

noncomputable def expected_length_remaining (n : ℕ) := (11/18 : ℚ)^n

theorem expected_length_after_2012_repetitions :
  expected_length_remaining 2012 = (11 / 18 : ℚ) ^ 2012 :=
by
  sorry

end NUMINAMATH_GPT_expected_length_after_2012_repetitions_l1398_139819


namespace NUMINAMATH_GPT_average_population_is_1000_l1398_139874

-- Define the populations of the villages.
def populations : List ℕ := [803, 900, 1100, 1023, 945, 980, 1249]

-- Define the number of villages.
def num_villages : ℕ := 7

-- Define the total population.
def total_population (pops : List ℕ) : ℕ :=
  pops.foldl (λ acc x => acc + x) 0

-- Define the average population computation.
def average_population (pops : List ℕ) (n : ℕ) : ℕ :=
  total_population pops / n

-- Prove that the average population of the 7 villages is 1000.
theorem average_population_is_1000 :
  average_population populations num_villages = 1000 := by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_average_population_is_1000_l1398_139874


namespace NUMINAMATH_GPT_solve_equation_l1398_139886

-- Define the given equation
def equation (x : ℝ) : Prop := (x^3 - 3 * x^2) / (x^2 - 4 * x + 4) + x = -3

-- State the theorem indicating the solutions to the equation
theorem solve_equation (x : ℝ) (h : x ≠ 2) : 
  equation x ↔ x = -2 ∨ x = 3 / 2 :=
sorry

end NUMINAMATH_GPT_solve_equation_l1398_139886


namespace NUMINAMATH_GPT_base_11_arithmetic_l1398_139872

-- Define the base and the numbers in base 11
def base := 11

def a := 6 * base^2 + 7 * base + 4  -- 674 in base 11
def b := 2 * base^2 + 7 * base + 9  -- 279 in base 11
def c := 1 * base^2 + 4 * base + 3  -- 143 in base 11
def result := 5 * base^2 + 5 * base + 9  -- 559 in base 11

theorem base_11_arithmetic :
  (a - b + c) = result :=
sorry

end NUMINAMATH_GPT_base_11_arithmetic_l1398_139872


namespace NUMINAMATH_GPT_tower_total_surface_area_l1398_139845

/-- Given seven cubes with volumes 1, 8, 27, 64, 125, 216, and 343 cubic units each, stacked vertically
    with volumes decreasing from bottom to top, compute their total surface area including the bottom. -/
theorem tower_total_surface_area :
  let volumes := [1, 8, 27, 64, 125, 216, 343]
  let side_lengths := volumes.map (fun v => v ^ (1 / 3))
  let surface_area (n : ℝ) (visible_faces : ℕ) := visible_faces * (n ^ 2)
  let total_surface_area := surface_area 7 5 + surface_area 6 4 + surface_area 5 4 + surface_area 4 4
                            + surface_area 3 4 + surface_area 2 4 + surface_area 1 5
  total_surface_area = 610 := sorry

end NUMINAMATH_GPT_tower_total_surface_area_l1398_139845


namespace NUMINAMATH_GPT_total_salmons_caught_l1398_139825

theorem total_salmons_caught :
  let hazel_salmons := 24
  let dad_salmons := 27
  hazel_salmons + dad_salmons = 51 :=
by
  sorry

end NUMINAMATH_GPT_total_salmons_caught_l1398_139825


namespace NUMINAMATH_GPT_negation_of_universal_abs_nonneg_l1398_139802

theorem negation_of_universal_abs_nonneg :
  (¬ (∀ x : ℝ, |x| ≥ 0)) ↔ (∃ x : ℝ, |x| < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_abs_nonneg_l1398_139802


namespace NUMINAMATH_GPT_solve_indeterminate_equation_l1398_139808

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem solve_indeterminate_equation (x y : ℕ) (hx : is_prime x) (hy : is_prime y) :
  x^2 - y^2 = x * y^2 - 19 ↔ (x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 7) :=
by
  sorry

end NUMINAMATH_GPT_solve_indeterminate_equation_l1398_139808


namespace NUMINAMATH_GPT_problem_proof_l1398_139858

theorem problem_proof:
  (∃ n : ℕ, 25 = n ^ 2) ∧
  (Prime 31) ∧
  (¬ ∀ p : ℕ, Prime p → p >= 3 → p = 2) ∧
  (∃ m : ℕ, 8 = m ^ 3) ∧
  (∃ a b : ℕ, Prime a ∧ Prime b ∧ 15 = a * b) :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l1398_139858


namespace NUMINAMATH_GPT_find_point_on_line_l1398_139818

theorem find_point_on_line (x y : ℝ) (h : 4 * x + 7 * y = 28) (hx : x = 3) : y = 16 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_point_on_line_l1398_139818


namespace NUMINAMATH_GPT_quadratic_function_value_at_2_l1398_139892

theorem quadratic_function_value_at_2 
  (a b c : ℝ) (h_a : a ≠ 0) 
  (h1 : 7 = a * (-3)^2 + b * (-3) + c)
  (h2 : 7 = a * (5)^2 + b * 5 + c)
  (h3 : -8 = c) :
  a * 2^2 + b * 2 + c = -8 := by 
  sorry

end NUMINAMATH_GPT_quadratic_function_value_at_2_l1398_139892


namespace NUMINAMATH_GPT_ratio_of_products_l1398_139895

theorem ratio_of_products (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  ((a - c) * (b - d)) / ((a - b) * (c - d)) = -4 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_products_l1398_139895


namespace NUMINAMATH_GPT_no_equal_refereed_matches_l1398_139805

theorem no_equal_refereed_matches {k : ℕ} (h1 : ∀ {n : ℕ}, n > k → n = 2 * k) 
    (h2 : ∀ {n : ℕ}, n > k → ∃ m, m = k * (2 * k - 1))
    (h3 : ∀ {n : ℕ}, n > k → ∃ r, r = (2 * k - 1) / 2): 
    False := 
by
  sorry

end NUMINAMATH_GPT_no_equal_refereed_matches_l1398_139805


namespace NUMINAMATH_GPT_inequality_has_no_solutions_l1398_139875

theorem inequality_has_no_solutions (x : ℝ) : ¬ (3 * x^2 + 9 * x + 12 ≤ 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_has_no_solutions_l1398_139875


namespace NUMINAMATH_GPT_find_three_numbers_l1398_139880

theorem find_three_numbers :
  ∃ (a₁ a₄ a₂₅ : ℕ), a₁ + a₄ + a₂₅ = 114 ∧
    ( ∃ r ≠ 1, a₄ = a₁ * r ∧ a₂₅ = a₄ * r * r ) ∧
    ( ∃ d, a₄ = a₁ + 3 * d ∧ a₂₅ = a₁ + 24 * d ) ∧
    a₁ = 2 ∧ a₄ = 14 ∧ a₂₅ = 98 :=
by
  sorry

end NUMINAMATH_GPT_find_three_numbers_l1398_139880


namespace NUMINAMATH_GPT_expression_value_is_241_l1398_139831

noncomputable def expression_value : ℕ :=
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2

theorem expression_value_is_241 : expression_value = 241 := 
by
  sorry

end NUMINAMATH_GPT_expression_value_is_241_l1398_139831


namespace NUMINAMATH_GPT_change_in_y_when_x_increases_l1398_139824

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 3 - 5 * x

-- State the theorem
theorem change_in_y_when_x_increases (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -5 :=
by
  sorry

end NUMINAMATH_GPT_change_in_y_when_x_increases_l1398_139824


namespace NUMINAMATH_GPT_S_17_33_50_sum_l1398_139801

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    - (n / 2)
  else
    (n + 1) / 2

theorem S_17_33_50_sum : S 17 + S 33 + S 50 = 1 :=
by
  sorry

end NUMINAMATH_GPT_S_17_33_50_sum_l1398_139801


namespace NUMINAMATH_GPT_undefined_value_l1398_139883

theorem undefined_value (x : ℝ) : (x^2 - 16 * x + 64 = 0) → (x = 8) := by
  sorry

end NUMINAMATH_GPT_undefined_value_l1398_139883


namespace NUMINAMATH_GPT_connie_blue_markers_l1398_139884

theorem connie_blue_markers :
  ∀ (total_markers red_markers blue_markers : ℕ),
    total_markers = 105 →
    red_markers = 41 →
    blue_markers = total_markers - red_markers →
    blue_markers = 64 :=
by
  intros total_markers red_markers blue_markers htotal hred hblue
  rw [htotal, hred] at hblue
  exact hblue

end NUMINAMATH_GPT_connie_blue_markers_l1398_139884


namespace NUMINAMATH_GPT_least_integer_square_double_condition_l1398_139813

theorem least_integer_square_double_condition : ∃ x : ℤ, x^2 = 2 * x + 75 ∧ ∀ y : ℤ, y^2 = 2 * y + 75 → x ≤ y :=
by
  use -8
  sorry

end NUMINAMATH_GPT_least_integer_square_double_condition_l1398_139813


namespace NUMINAMATH_GPT_negation_of_existence_l1398_139890

theorem negation_of_existence :
  ¬ (∃ x : ℝ, x^2 + 3*x + 2 < 0) ↔ ∀ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
sorry

end NUMINAMATH_GPT_negation_of_existence_l1398_139890


namespace NUMINAMATH_GPT_f_value_plus_deriv_l1398_139837

noncomputable def f : ℝ → ℝ := sorry

-- Define the function f and its derivative at x = 1
axiom f_deriv_at_1 : deriv f 1 = 1 / 2

-- Define the value of the function f at x = 1
axiom f_value_at_1 : f 1 = 5 / 2

-- Prove that f(1) + f'(1) = 3
theorem f_value_plus_deriv : f 1 + deriv f 1 = 3 :=
by
  rw [f_value_at_1, f_deriv_at_1]
  norm_num

end NUMINAMATH_GPT_f_value_plus_deriv_l1398_139837


namespace NUMINAMATH_GPT_Douglas_weight_correct_l1398_139843

def Anne_weight : ℕ := 67
def weight_diff : ℕ := 15
def Douglas_weight : ℕ := 52

theorem Douglas_weight_correct : Douglas_weight = Anne_weight - weight_diff := by
  sorry

end NUMINAMATH_GPT_Douglas_weight_correct_l1398_139843


namespace NUMINAMATH_GPT_candy_count_after_giving_l1398_139816

def numKitKats : ℕ := 5
def numHersheyKisses : ℕ := 3 * numKitKats
def numNerds : ℕ := 8
def numLollipops : ℕ := 11
def numBabyRuths : ℕ := 10
def numReeseCups : ℕ := numBabyRuths / 2
def numLollipopsGivenAway : ℕ := 5

def totalCandyBefore : ℕ := numKitKats + numHersheyKisses + numNerds + numLollipops + numBabyRuths + numReeseCups
def totalCandyAfter : ℕ := totalCandyBefore - numLollipopsGivenAway

theorem candy_count_after_giving : totalCandyAfter = 49 := by
  sorry

end NUMINAMATH_GPT_candy_count_after_giving_l1398_139816


namespace NUMINAMATH_GPT_no_solution_for_inequality_l1398_139856

theorem no_solution_for_inequality (x : ℝ) : ¬ (3 * x^2 + 9 * x ≤ -12) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_inequality_l1398_139856


namespace NUMINAMATH_GPT_integer_solution_l1398_139840

theorem integer_solution (x : ℕ) (h : (4 * x)^2 - 2 * x = 3178) : x = 226 :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_l1398_139840


namespace NUMINAMATH_GPT_integer_roots_condition_l1398_139862

noncomputable def has_integer_roots (n : ℕ) : Prop :=
  ∃ x : ℤ, x * x - 4 * x + n = 0

theorem integer_roots_condition (n : ℕ) (h : n > 0) :
  has_integer_roots n ↔ n = 3 ∨ n = 4 :=
by 
  sorry

end NUMINAMATH_GPT_integer_roots_condition_l1398_139862


namespace NUMINAMATH_GPT_base_seven_sum_of_product_l1398_139834

def base_seven_to_decimal (d1 d0 : ℕ) : ℕ :=
  7 * d1 + d0

def decimal_to_base_seven (n : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let d3 := n / (7 ^ 3)
  let r3 := n % (7 ^ 3)
  let d2 := r3 / (7 ^ 2)
  let r2 := r3 % (7 ^ 2)
  let d1 := r2 / 7
  let d0 := r2 % 7
  (d3, d2, d1, d0)

def sum_of_base_seven_digits (d3 d2 d1 d0 : ℕ) : ℕ :=
  d3 + d2 + d1 + d0

theorem base_seven_sum_of_product :
  let n1 := base_seven_to_decimal 3 5
  let n2 := base_seven_to_decimal 4 2
  let product := n1 * n2
  let (d3, d2, d1, d0) := decimal_to_base_seven product
  sum_of_base_seven_digits d3 d2 d1 d0 = 18 :=
  by
    sorry

end NUMINAMATH_GPT_base_seven_sum_of_product_l1398_139834


namespace NUMINAMATH_GPT_probability_accurate_forecast_l1398_139885

theorem probability_accurate_forecast (p q : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : 0 ≤ q ∧ q ≤ 1) : 
  p * (1 - q) = p * (1 - q) :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_accurate_forecast_l1398_139885


namespace NUMINAMATH_GPT_convert_246_octal_to_decimal_l1398_139896

theorem convert_246_octal_to_decimal : 2 * (8^2) + 4 * (8^1) + 6 * (8^0) = 166 := 
by
  -- We skip the proof part as it is not required in the task
  sorry

end NUMINAMATH_GPT_convert_246_octal_to_decimal_l1398_139896


namespace NUMINAMATH_GPT_find_polygon_sides_l1398_139823

theorem find_polygon_sides (n : ℕ) (h : n - 3 = 5) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_polygon_sides_l1398_139823


namespace NUMINAMATH_GPT_total_marbles_l1398_139846

theorem total_marbles (marbles_per_row_8 : ℕ) (rows_of_9 : ℕ) (marbles_per_row_1 : ℕ) (rows_of_4 : ℕ) 
  (h1 : marbles_per_row_8 = 9) 
  (h2 : rows_of_9 = 8) 
  (h3 : marbles_per_row_1 = 4) 
  (h4 : rows_of_4 = 1) : 
  (marbles_per_row_8 * rows_of_9 + marbles_per_row_1 * rows_of_4) = 76 :=
by
  sorry

end NUMINAMATH_GPT_total_marbles_l1398_139846


namespace NUMINAMATH_GPT_range_of_a_l1398_139828

-- Definition of sets A and B
def set_A := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def set_B (a : ℝ) := {x : ℝ | 0 < x ∧ x < a}

-- Statement that if A ⊆ B, then a > 3
theorem range_of_a (a : ℝ) (h : set_A ⊆ set_B a) : 3 < a :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1398_139828


namespace NUMINAMATH_GPT_correct_transformation_l1398_139844

variable {a b c : ℝ}

-- A: \frac{a+3}{b+3} = \frac{a}{b}
def transformation_A (a b : ℝ) : Prop := (a + 3) / (b + 3) = a / b

-- B: \frac{a}{b} = \frac{ac}{bc}
def transformation_B (a b c : ℝ) : Prop := a / b = (a * c) / (b * c)

-- C: \frac{3a}{3b} = \frac{a}{b}
def transformation_C (a b : ℝ) : Prop := (3 * a) / (3 * b) = a / b

-- D: \frac{a}{b} = \frac{a^2}{b^2}
def transformation_D (a b : ℝ) : Prop := a / b = (a ^ 2) / (b ^ 2)

-- The main theorem to prove
theorem correct_transformation : transformation_C a b :=
by
  sorry

end NUMINAMATH_GPT_correct_transformation_l1398_139844


namespace NUMINAMATH_GPT_intersection_eq_l1398_139897

def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 < 0}
def intersection : Set ℝ := {x | 2 < x ∧ x < 3}

theorem intersection_eq : M ∩ N = intersection := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l1398_139897


namespace NUMINAMATH_GPT_number_of_players_l1398_139898

-- Definitions based on conditions
def socks_price : ℕ := 6
def tshirt_price : ℕ := socks_price + 7
def total_cost_per_player : ℕ := 2 * (socks_price + tshirt_price)
def total_expenditure : ℕ := 4092

-- Lean theorem statement
theorem number_of_players : total_expenditure / total_cost_per_player = 108 := 
by
  sorry

end NUMINAMATH_GPT_number_of_players_l1398_139898


namespace NUMINAMATH_GPT_twenty_five_billion_scientific_notation_l1398_139868

theorem twenty_five_billion_scientific_notation :
  (25 * 10^9 : ℝ) = 2.5 * 10^10 := 
by simp only [←mul_assoc, ←@pow_add ℝ, pow_one, two_mul];
   norm_num

end NUMINAMATH_GPT_twenty_five_billion_scientific_notation_l1398_139868


namespace NUMINAMATH_GPT_f_at_8_l1398_139865

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition1 : f 1 = 1
axiom f_condition2 : ∀ x y : ℝ, f (x + y) + f (x - y) = f x * f y

theorem f_at_8 : f 8 = -1 := 
by
-- The following will be filled with the proof, hence sorry for now.
sorry

end NUMINAMATH_GPT_f_at_8_l1398_139865


namespace NUMINAMATH_GPT_remainder_3211_div_103_l1398_139899

theorem remainder_3211_div_103 :
  3211 % 103 = 18 :=
by
  sorry

end NUMINAMATH_GPT_remainder_3211_div_103_l1398_139899


namespace NUMINAMATH_GPT_simplify_fraction_fraction_c_over_d_l1398_139867

-- Define necessary constants and variables
variable (k : ℤ)

/-- Original expression -/
def original_expr := (6 * k + 12 + 3 : ℤ)

/-- Simplified expression -/
def simplified_expr := (2 * k + 5 : ℤ)

/-- The main theorem to prove the equivalent mathematical proof problem -/
theorem simplify_fraction : (original_expr / 3) = simplified_expr :=
by
  sorry

-- The final fraction to prove the answer
theorem fraction_c_over_d : (2 / 5 : ℚ) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_fraction_c_over_d_l1398_139867


namespace NUMINAMATH_GPT_farthest_vertex_label_l1398_139852

-- The vertices and their labeling
def cube_faces : List (List Nat) := [
  [1, 2, 5, 8],
  [3, 4, 6, 7],
  [2, 4, 5, 7],
  [1, 3, 6, 8],
  [2, 3, 7, 8],
  [1, 4, 5, 6]
]

-- Define the cube vertices labels
def vertices : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

-- Statement of the problem in Lean 4
theorem farthest_vertex_label (h : true) : 
  ∃ v : Nat, v ∈ vertices ∧ ∀ face ∈ cube_faces, v ∉ face → v = 6 := 
sorry

end NUMINAMATH_GPT_farthest_vertex_label_l1398_139852


namespace NUMINAMATH_GPT_product_is_58_l1398_139833

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def p := 2
def q := 29

-- Conditions based on the problem
axiom prime_p : is_prime p
axiom prime_q : is_prime q
axiom sum_eq_31 : p + q = 31

-- Theorem to be proven
theorem product_is_58 : p * q = 58 :=
by
  sorry

end NUMINAMATH_GPT_product_is_58_l1398_139833


namespace NUMINAMATH_GPT_base_number_pow_19_mod_10_l1398_139873

theorem base_number_pow_19_mod_10 (x : ℕ) (h : x ^ 19 % 10 = 7) : x % 10 = 3 :=
sorry

end NUMINAMATH_GPT_base_number_pow_19_mod_10_l1398_139873


namespace NUMINAMATH_GPT_work_combined_days_l1398_139859

theorem work_combined_days (A B C : ℝ) (hA : A = 1 / 4) (hB : B = 1 / 12) (hC : C = 1 / 6) :
  1 / (A + B + C) = 2 :=
by
  sorry

end NUMINAMATH_GPT_work_combined_days_l1398_139859


namespace NUMINAMATH_GPT_initial_population_is_10000_l1398_139854

def population_growth (P : ℝ) : Prop :=
  let growth_rate := 0.20
  let final_population := 12000
  final_population = P * (1 + growth_rate)

theorem initial_population_is_10000 : population_growth 10000 :=
by
  unfold population_growth
  sorry

end NUMINAMATH_GPT_initial_population_is_10000_l1398_139854


namespace NUMINAMATH_GPT_find_fiona_experience_l1398_139853

namespace Experience

variables (d e f : ℚ)

def avg_experience_equation : Prop := d + e + f = 36
def fiona_david_equation : Prop := f - 5 = d
def emma_david_future_equation : Prop := e + 4 = (3/4) * (d + 4)

theorem find_fiona_experience (h1 : avg_experience_equation d e f) (h2 : fiona_david_equation d f) (h3 : emma_david_future_equation d e) :
  f = 183 / 11 :=
by
  sorry

end Experience

end NUMINAMATH_GPT_find_fiona_experience_l1398_139853


namespace NUMINAMATH_GPT_min_x2_plus_y2_l1398_139810

noncomputable def min_val (x y : ℝ) : ℝ :=
  if h : (x + 1)^2 + y^2 = 1/4 then x^2 + y^2 else 0

theorem min_x2_plus_y2 : 
  ∃ x y : ℝ, (x + 1)^2 + y^2 = 1/4 ∧ x^2 + y^2 = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_min_x2_plus_y2_l1398_139810


namespace NUMINAMATH_GPT_cent_piece_value_l1398_139860

theorem cent_piece_value (Q P : ℕ) 
  (h1 : Q + P = 29)
  (h2 : 25 * Q + P = 545)
  (h3 : Q = 17) : 
  P = 120 := by
  sorry

end NUMINAMATH_GPT_cent_piece_value_l1398_139860


namespace NUMINAMATH_GPT_sphere_radius_twice_cone_volume_l1398_139830

theorem sphere_radius_twice_cone_volume :
  ∀ (r_cone h_cone : ℝ) (r_sphere : ℝ), 
    r_cone = 2 → h_cone = 8 → 2 * (1 / 3 * Real.pi * r_cone^2 * h_cone) = (4/3 * Real.pi * r_sphere^3) → 
    r_sphere = 2^(4/3) :=
by
  intros r_cone h_cone r_sphere h_r_cone h_h_cone h_volume_equiv
  sorry

end NUMINAMATH_GPT_sphere_radius_twice_cone_volume_l1398_139830


namespace NUMINAMATH_GPT_chickens_and_rabbits_l1398_139841

theorem chickens_and_rabbits (total_animals : ℕ) (total_legs : ℕ) (chickens : ℕ) (rabbits : ℕ) 
    (h1 : total_animals = 40) 
    (h2 : total_legs = 108) 
    (h3 : total_animals = chickens + rabbits) 
    (h4 : total_legs = 2 * chickens + 4 * rabbits) : 
    chickens = 26 ∧ rabbits = 14 :=
by
  sorry

end NUMINAMATH_GPT_chickens_and_rabbits_l1398_139841


namespace NUMINAMATH_GPT_households_soap_usage_l1398_139850

theorem households_soap_usage
  (total_households : ℕ)
  (neither : ℕ)
  (both : ℕ)
  (only_B_ratio : ℕ)
  (B := only_B_ratio * both) :
  total_households = 200 →
  neither = 80 →
  both = 40 →
  only_B_ratio = 3 →
  (total_households - neither - both - B = 40) :=
by
  intros
  sorry

end NUMINAMATH_GPT_households_soap_usage_l1398_139850


namespace NUMINAMATH_GPT_solve_for_x_l1398_139864

theorem solve_for_x (x y z : ℚ) (h1 : x * y = 2 * (x + y)) (h2 : y * z = 4 * (y + z)) (h3 : x * z = 8 * (x + z)) (hx0 : x ≠ 0) (hy0 : y ≠ 0) (hz0 : z ≠ 0) : x = 16 / 3 := 
sorry

end NUMINAMATH_GPT_solve_for_x_l1398_139864


namespace NUMINAMATH_GPT_catch_bus_probability_within_5_minutes_l1398_139821

theorem catch_bus_probability_within_5_minutes :
  (Pbus3 : ℝ) → (Pbus6 : ℝ) → (Pbus3 = 0.20) → (Pbus6 = 0.60) → (Pcatch : ℝ) → (Pcatch = Pbus3 + Pbus6) → (Pcatch = 0.80) :=
by
  intros Pbus3 Pbus6 hPbus3 hPbus6 Pcatch hPcatch
  sorry

end NUMINAMATH_GPT_catch_bus_probability_within_5_minutes_l1398_139821


namespace NUMINAMATH_GPT_translated_line_eqn_l1398_139847

theorem translated_line_eqn
  (c : ℝ) :
  ∀ (y_eqn : ℝ → ℝ), 
    (∀ x, y_eqn x = 2 * x + 1) →
    (∀ x, (y_eqn (x - 2) - 3) = (2 * x - 6)) :=
by
  sorry

end NUMINAMATH_GPT_translated_line_eqn_l1398_139847


namespace NUMINAMATH_GPT_cistern_filled_in_12_hours_l1398_139851

def fill_rate := 1 / 6
def empty_rate := 1 / 12
def net_rate := fill_rate - empty_rate

theorem cistern_filled_in_12_hours :
  (1 / net_rate) = 12 :=
by
  -- Proof omitted for clarity
  sorry

end NUMINAMATH_GPT_cistern_filled_in_12_hours_l1398_139851


namespace NUMINAMATH_GPT_lattice_midpoint_l1398_139809

theorem lattice_midpoint (points : Fin 5 → ℤ × ℤ) :
  ∃ (i j : Fin 5), i ≠ j ∧ 
  let (x1, y1) := points i 
  let (x2, y2) := points j
  (x1 + x2) % 2 = 0 ∧ (y1 + y2) % 2 = 0 := 
sorry

end NUMINAMATH_GPT_lattice_midpoint_l1398_139809


namespace NUMINAMATH_GPT_oliver_remaining_dishes_l1398_139838

def num_dishes := 36
def dishes_with_mango_salsa := 3
def dishes_with_fresh_mango := num_dishes / 6
def dishes_with_mango_jelly := 1
def oliver_picks_two := 2

theorem oliver_remaining_dishes : 
  num_dishes - (dishes_with_mango_salsa + dishes_with_fresh_mango + dishes_with_mango_jelly - oliver_picks_two) = 28 := by
  sorry

end NUMINAMATH_GPT_oliver_remaining_dishes_l1398_139838


namespace NUMINAMATH_GPT_solve_problem_l1398_139876

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) :
  f (x^3 + y^3) = (x + y) * ((f x)^2 - (f x) * (f y) + (f (f y))^2)

theorem solve_problem (x : ℝ) : f (1996 * x) = 1996 * f x :=
sorry

end NUMINAMATH_GPT_solve_problem_l1398_139876


namespace NUMINAMATH_GPT_opposite_sides_of_line_l1398_139811

theorem opposite_sides_of_line (m : ℝ) (h : (2 * (-2 : ℝ) + m - 2) * (2 * m + 4 - 2) < 0) : -1 < m ∧ m < 6 :=
sorry

end NUMINAMATH_GPT_opposite_sides_of_line_l1398_139811


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1398_139800

theorem quadratic_inequality_solution (k : ℝ) :
  (-1 < k ∧ k < 7) ↔ ∀ x : ℝ, x^2 - (k - 5) * x - k + 8 > 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1398_139800


namespace NUMINAMATH_GPT_expected_value_of_winnings_l1398_139832

noncomputable def expected_value : ℝ :=
  (1 / 8) * (1 / 2) + (1 / 8) * (3 / 2) + (1 / 8) * (5 / 2) + (1 / 8) * (7 / 2) +
  (1 / 8) * 2 + (1 / 8) * 4 + (1 / 8) * 6 + (1 / 8) * 8

theorem expected_value_of_winnings : expected_value = 3.5 :=
by
  -- the proof steps will go here
  sorry

end NUMINAMATH_GPT_expected_value_of_winnings_l1398_139832


namespace NUMINAMATH_GPT_cos_two_pi_over_three_plus_two_alpha_l1398_139848

theorem cos_two_pi_over_three_plus_two_alpha 
  (α : ℝ)
  (h : Real.sin (π / 6 - α) = 1 / 3) :
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 := 
by
  sorry

end NUMINAMATH_GPT_cos_two_pi_over_three_plus_two_alpha_l1398_139848


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1398_139857

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h0 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2) 
  (h1 : S 10 = 12)
  (h2 : S 20 = 17) :
  S 30 = 15 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1398_139857


namespace NUMINAMATH_GPT_louisa_second_day_miles_l1398_139870

theorem louisa_second_day_miles (T1 T2 : ℕ) (speed miles_first_day miles_second_day : ℕ)
  (h1 : speed = 25) 
  (h2 : miles_first_day = 100)
  (h3 : T1 = miles_first_day / speed) 
  (h4 : T2 = T1 + 3) 
  (h5 : miles_second_day = speed * T2) :
  miles_second_day = 175 := 
by
  -- We can add the necessary calculations here, but for now, sorry is used to skip the proof.
  sorry

end NUMINAMATH_GPT_louisa_second_day_miles_l1398_139870


namespace NUMINAMATH_GPT_large_font_pages_l1398_139887

theorem large_font_pages (L S : ℕ) (h1 : L + S = 21) (h2 : 3 * L = 2 * S) : L = 8 :=
by {
  sorry -- Proof can be filled in Lean; this ensures the statement aligns with problem conditions.
}

end NUMINAMATH_GPT_large_font_pages_l1398_139887


namespace NUMINAMATH_GPT_smallest_number_of_pencils_l1398_139882

theorem smallest_number_of_pencils 
  (p : ℕ) 
  (h1 : p % 6 = 5)
  (h2 : p % 7 = 3)
  (h3 : p % 8 = 7) :
  p = 35 := 
sorry

end NUMINAMATH_GPT_smallest_number_of_pencils_l1398_139882


namespace NUMINAMATH_GPT_krishan_money_l1398_139839

theorem krishan_money
  (R G K : ℕ)
  (h1 : R / G = 7 / 17)
  (h2 : G / K = 7 / 17)
  (h3 : R = 637) : 
  K = 3774 := 
by
  sorry

end NUMINAMATH_GPT_krishan_money_l1398_139839


namespace NUMINAMATH_GPT_tree_heights_l1398_139871

theorem tree_heights (T S : ℕ) (h1 : T - S = 20) (h2 : T - 10 = 3 * (S - 10)) : T = 40 := 
by
  sorry

end NUMINAMATH_GPT_tree_heights_l1398_139871


namespace NUMINAMATH_GPT_number_of_people_on_boats_l1398_139815

def boats := 5
def people_per_boat := 3

theorem number_of_people_on_boats : boats * people_per_boat = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_on_boats_l1398_139815


namespace NUMINAMATH_GPT_wendy_tooth_extraction_cost_eq_290_l1398_139822

def dentist_cleaning_cost : ℕ := 70
def dentist_filling_cost : ℕ := 120
def wendy_dentist_bill : ℕ := 5 * dentist_filling_cost
def wendy_cleaning_and_fillings_cost : ℕ := dentist_cleaning_cost + 2 * dentist_filling_cost
def wendy_tooth_extraction_cost : ℕ := wendy_dentist_bill - wendy_cleaning_and_fillings_cost

theorem wendy_tooth_extraction_cost_eq_290 : wendy_tooth_extraction_cost = 290 := by
  sorry

end NUMINAMATH_GPT_wendy_tooth_extraction_cost_eq_290_l1398_139822


namespace NUMINAMATH_GPT_perpendicular_slope_l1398_139827

theorem perpendicular_slope (x y : ℝ) (h : 3 * x - 4 * y = 8) :
  ∃ m : ℝ, m = - (4 / 3) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_slope_l1398_139827


namespace NUMINAMATH_GPT_probability_white_ball_l1398_139820

theorem probability_white_ball (num_black_balls num_white_balls : ℕ) 
  (black_balls : num_black_balls = 6) 
  (white_balls : num_white_balls = 5) : 
  (num_white_balls / (num_black_balls + num_white_balls) : ℚ) = 5 / 11 :=
by
  sorry

end NUMINAMATH_GPT_probability_white_ball_l1398_139820


namespace NUMINAMATH_GPT_classroom_gpa_l1398_139879

theorem classroom_gpa (x : ℝ) (h1 : (1 / 3) * x + (2 / 3) * 18 = 17) : x = 15 := 
by 
    sorry

end NUMINAMATH_GPT_classroom_gpa_l1398_139879


namespace NUMINAMATH_GPT_difference_between_largest_and_smallest_quarters_l1398_139803

noncomputable def coin_collection : Prop :=
  ∃ (n d q : ℕ), 
    (n + d + q = 150) ∧ 
    (5 * n + 10 * d + 25 * q = 2000) ∧ 
    (forall (q1 q2 : ℕ), (n + d + q1 = 150) ∧ (5 * n + 10 * d + 25 * q1 = 2000) → 
     (n + d + q2 = 150) ∧ (5 * n + 10 * d + 25 * q2 = 2000) → 
     (q1 = q2))

theorem difference_between_largest_and_smallest_quarters : coin_collection :=
  sorry

end NUMINAMATH_GPT_difference_between_largest_and_smallest_quarters_l1398_139803


namespace NUMINAMATH_GPT_division_of_powers_l1398_139817

theorem division_of_powers (a : ℝ) (h : a ≠ 0) : a^10 / a^9 = a := 
by sorry

end NUMINAMATH_GPT_division_of_powers_l1398_139817


namespace NUMINAMATH_GPT_polygon_properties_l1398_139894

-- Assume n is the number of sides of the polygon
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180
def sum_of_exterior_angles : ℝ := 360

-- Given the condition
def given_condition (n : ℕ) : Prop := sum_of_interior_angles n = 5 * sum_of_exterior_angles

theorem polygon_properties (n : ℕ) (h1 : given_condition n) :
  n = 12 ∧ (n * (n - 3)) / 2 = 54 :=
by
  sorry

end NUMINAMATH_GPT_polygon_properties_l1398_139894


namespace NUMINAMATH_GPT_p_necessary_not_sufficient_for_q_l1398_139836

def vec (a b : ℝ) : ℝ × ℝ := (a, b)

def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def p (a : ℝ) : Prop :=
  collinear (vec a (a^2)) (vec 1 2)

def q (a : ℝ) : Prop := a = 2

theorem p_necessary_not_sufficient_for_q :
  (∀ a : ℝ, q a → p a) ∧ ¬(∀ a : ℝ, p a → q a) :=
sorry

end NUMINAMATH_GPT_p_necessary_not_sufficient_for_q_l1398_139836


namespace NUMINAMATH_GPT_tautology_a_tautology_b_tautology_c_tautology_d_l1398_139812

variable (p q : Prop)

theorem tautology_a : p ∨ ¬ p := by
  sorry

theorem tautology_b : ¬ ¬ p ↔ p := by
  sorry

theorem tautology_c : ((p → q) → p) → p := by
  sorry

theorem tautology_d : ¬ (p ∧ ¬ p) := by
  sorry

end NUMINAMATH_GPT_tautology_a_tautology_b_tautology_c_tautology_d_l1398_139812


namespace NUMINAMATH_GPT_min_sum_abc_l1398_139869

theorem min_sum_abc (a b c : ℕ) (h1 : a * b * c = 3960) : a + b + c ≥ 150 :=
sorry

end NUMINAMATH_GPT_min_sum_abc_l1398_139869
