import Mathlib

namespace NUMINAMATH_GPT_polynomial_not_divisible_by_x_minus_5_l157_15798

theorem polynomial_not_divisible_by_x_minus_5 (m : ℝ) :
  (∀ x, x = 4 → (4 * x^3 - 16 * x^2 + m * x - 20) = 0) →
  ¬(∀ x, x = 5 → (4 * x^3 - 16 * x^2 + m * x - 20) = 0) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_not_divisible_by_x_minus_5_l157_15798


namespace NUMINAMATH_GPT_speed_of_B_l157_15724

theorem speed_of_B 
  (A_speed : ℝ)
  (t1 : ℝ)
  (t2 : ℝ)
  (d1 := A_speed * t1)
  (d2 := A_speed * t2)
  (total_distance := d1 + d2)
  (B_speed := total_distance / t2) :
  A_speed = 7 → 
  t1 = 0.5 → 
  t2 = 1.8 →
  B_speed = 8.944 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  exact sorry

end NUMINAMATH_GPT_speed_of_B_l157_15724


namespace NUMINAMATH_GPT_order_of_a_b_c_l157_15728

noncomputable def a : ℝ := (Real.log 5) / 5
noncomputable def b : ℝ := 1 / Real.exp 1
noncomputable def c : ℝ := (Real.log 4) / 4

theorem order_of_a_b_c : a < c ∧ c < b := by
  sorry

end NUMINAMATH_GPT_order_of_a_b_c_l157_15728


namespace NUMINAMATH_GPT_lara_cookies_l157_15758

theorem lara_cookies (total_cookies trays rows_per_row : ℕ)
  (h_total : total_cookies = 120)
  (h_trays : trays = 4)
  (h_rows_per_row : rows_per_row = 6) :
  total_cookies / rows_per_row / trays = 5 :=
by
  sorry

end NUMINAMATH_GPT_lara_cookies_l157_15758


namespace NUMINAMATH_GPT_find_coef_of_quadratic_l157_15791

-- Define the problem conditions
def solutions_of_abs_eq : Set ℤ := {x | abs (x - 3) = 4}

-- Given that the solutions are 7 and -1
def paul_solutions : Set ℤ := {7, -1}

-- The problem translates to proving the equivalence of two sets
def equivalent_equation_solutions (d e : ℤ) : Prop :=
  ∀ x, x ∈ solutions_of_abs_eq ↔ x^2 + d * x + e = 0

theorem find_coef_of_quadratic :
  equivalent_equation_solutions (-6) (-7) :=
by
  sorry

end NUMINAMATH_GPT_find_coef_of_quadratic_l157_15791


namespace NUMINAMATH_GPT_contribution_amount_l157_15763

-- Definitions based on conditions
variable (x : ℝ)

-- Total amount needed
def total_needed := 200

-- Contributions from different families
def contribution_two_families := 2 * x
def contribution_eight_families := 8 * 10 -- 80
def contribution_ten_families := 10 * 5 -- 50
def total_contribution := contribution_two_families + contribution_eight_families + contribution_ten_families

-- Amount raised so far given they need 30 more to reach the target
def raised_so_far := total_needed - 30 -- 170

-- Statement to prove
theorem contribution_amount :
  total_contribution x = raised_so_far →
  x = 20 := by 
  sorry

end NUMINAMATH_GPT_contribution_amount_l157_15763


namespace NUMINAMATH_GPT_vasya_numbers_l157_15744

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_vasya_numbers_l157_15744


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l157_15731

-- Variables and conditions
variables (a : ℕ) (A B : ℝ)
variable (positive_a : 0 < a)

-- System of equations
def system_has_positive_integer_solutions (x y z : ℕ) : Prop :=
  (x^2 + y^2 + z^2 = (13 * a)^2) ∧ 
  (x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) = 
    (1 / 4) * (2 * A + B) * (13 * a)^4)

-- Statement of the theorem
theorem necessary_and_sufficient_condition:
  (∃ (x y z : ℕ), system_has_positive_integer_solutions a A B x y z) ↔ B = 2 * A :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l157_15731


namespace NUMINAMATH_GPT_circle_symmetric_line_l157_15797
-- Importing the entire Math library

-- Define the statement
theorem circle_symmetric_line (a : ℝ) :
  (∀ (A B : ℝ × ℝ), 
    (A.1)^2 + (A.2)^2 = 2 * a * (A.1) 
    ∧ (B.1)^2 + (B.2)^2 = 2 * a * (B.1) 
    ∧ A.2 = 2 * A.1 + 1 
    ∧ B.2 = 2 * B.1 + 1 
    ∧ A.2 = B.2) 
  → a = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_circle_symmetric_line_l157_15797


namespace NUMINAMATH_GPT_prevent_white_cube_n2_prevent_white_cube_n3_l157_15701

def min_faces_to_paint (n : ℕ) : ℕ :=
  if n = 2 then 2 else if n = 3 then 12 else sorry

theorem prevent_white_cube_n2 : min_faces_to_paint 2 = 2 := by
  sorry

theorem prevent_white_cube_n3 : min_faces_to_paint 3 = 12 := by
  sorry

end NUMINAMATH_GPT_prevent_white_cube_n2_prevent_white_cube_n3_l157_15701


namespace NUMINAMATH_GPT_percentage_profits_revenues_previous_year_l157_15730

noncomputable def companyProfits (R P R2009 P2009 : ℝ) : Prop :=
  (R2009 = 0.8 * R) ∧ (P2009 = 0.15 * R2009) ∧ (P2009 = 1.5 * P)

theorem percentage_profits_revenues_previous_year (R P : ℝ) (h : companyProfits R P (0.8 * R) (0.12 * R)) : 
  (P / R * 100) = 8 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_profits_revenues_previous_year_l157_15730


namespace NUMINAMATH_GPT_negative_only_option_B_l157_15727

theorem negative_only_option_B :
  (0 > -3) ∧ 
  (|-3| = 3) ∧ 
  (0 < 3) ∧
  (0 < (1/3)) ∧
  ∀ x, x = -3 → x < 0 :=
by
  sorry

end NUMINAMATH_GPT_negative_only_option_B_l157_15727


namespace NUMINAMATH_GPT_inequality_system_solution_l157_15734

theorem inequality_system_solution (x : ℝ) :
  (3 * x > x + 6) ∧ ((1 / 2) * x < -x + 5) ↔ (3 < x) ∧ (x < 10 / 3) :=
by
  sorry

end NUMINAMATH_GPT_inequality_system_solution_l157_15734


namespace NUMINAMATH_GPT_largest_beverage_amount_l157_15743

theorem largest_beverage_amount :
  let Milk := (3 / 8 : ℚ)
  let Cider := (7 / 10 : ℚ)
  let OrangeJuice := (11 / 15 : ℚ)
  OrangeJuice > Milk ∧ OrangeJuice > Cider :=
by
  have Milk := (3 / 8 : ℚ)
  have Cider := (7 / 10 : ℚ)
  have OrangeJuice := (11 / 15 : ℚ)
  sorry

end NUMINAMATH_GPT_largest_beverage_amount_l157_15743


namespace NUMINAMATH_GPT_sum_A_k_div_k_l157_15789

noncomputable def A (k : ℕ) : ℕ :=
  (Finset.filter (fun d => d % 2 = 1 ∧ d ≤ Nat.sqrt (2 * k - 1)) (Finset.range k)).card

noncomputable def sumExpression : ℝ :=
  ∑' k, (-1)^(k-1) * (A k / k : ℝ)

theorem sum_A_k_div_k : sumExpression = Real.pi^2 / 8 :=
  sorry

end NUMINAMATH_GPT_sum_A_k_div_k_l157_15789


namespace NUMINAMATH_GPT_problem_statement_l157_15752

theorem problem_statement : (6^3 + 4^2) * 7^5 = 3897624 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l157_15752


namespace NUMINAMATH_GPT_find_first_number_l157_15794

theorem find_first_number (x : ℝ) (h1 : 2994 / x = 175) (h2 : 29.94 / 1.45 = 17.5) : x = 17.1 :=
by
  sorry

end NUMINAMATH_GPT_find_first_number_l157_15794


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l157_15748

theorem necessary_but_not_sufficient_condition (a c : ℝ) (h : c ≠ 0) : ¬ ((∀ (a : ℝ) (h : c ≠ 0), (ax^2 + y^2 = c) → ((ax^2 + y^2 = c) → ( (c ≠ 0) ))) ∧ ¬ ((∀ (a : ℝ), ¬ (ax^2 + y^2 ≠ c) → ( (ax^2 + y^2 = c) → ((c = 0) ))) )) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l157_15748


namespace NUMINAMATH_GPT_max_quotient_l157_15714

theorem max_quotient (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) : ∃ q, q = b / a ∧ q ≤ 16 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_max_quotient_l157_15714


namespace NUMINAMATH_GPT_find_value_of_a_l157_15750

theorem find_value_of_a (a : ℝ) (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 < a → a^x ≥ 1)
  (h_sum : (a^1) + (a^0) = 3) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_value_of_a_l157_15750


namespace NUMINAMATH_GPT_factorize_1_factorize_2_factorize_3_l157_15796

-- Problem 1: Factorize 3a^3 - 6a^2 + 3a
theorem factorize_1 (a : ℝ) : 3 * a^3 - 6 * a^2 + 3 * a = 3 * a * (a - 1)^2 :=
sorry

-- Problem 2: Factorize a^2(x - y) + b^2(y - x)
theorem factorize_2 (a b x y : ℝ) : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a - b) * (a + b) :=
sorry

-- Problem 3: Factorize 16(a + b)^2 - 9(a - b)^2
theorem factorize_3 (a b : ℝ) : 16 * (a + b)^2 - 9 * (a - b)^2 = (a + 7 * b) * (7 * a + b) :=
sorry

end NUMINAMATH_GPT_factorize_1_factorize_2_factorize_3_l157_15796


namespace NUMINAMATH_GPT_sum_of_powers_of_i_l157_15774

theorem sum_of_powers_of_i : 
  ∀ (i : ℂ), i^2 = -1 → 1 + i + i^2 + i^3 + i^4 + i^5 + i^6 = i :=
by
  intro i h
  sorry

end NUMINAMATH_GPT_sum_of_powers_of_i_l157_15774


namespace NUMINAMATH_GPT_Diana_total_earnings_l157_15772

def July : ℝ := 150
def August : ℝ := 3 * July
def September : ℝ := 2 * August
def October : ℝ := September + 0.1 * September
def November : ℝ := 0.95 * October
def Total_earnings : ℝ := July + August + September + October + November

theorem Diana_total_earnings : Total_earnings = 3430.50 := by
  sorry

end NUMINAMATH_GPT_Diana_total_earnings_l157_15772


namespace NUMINAMATH_GPT_prob_twins_street_l157_15761

variable (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1)

theorem prob_twins_street : p ≠ 1 → real := sorry

end NUMINAMATH_GPT_prob_twins_street_l157_15761


namespace NUMINAMATH_GPT_area_is_12_5_l157_15745

-- Define the triangle XYZ
structure Triangle := 
  (X Y Z : Type) 
  (XZ YZ : ℝ) 
  (angleX angleY angleZ : ℝ)

-- Provided conditions in the problem
def triangleXYZ : Triangle := {
  X := ℝ, 
  Y := ℝ, 
  Z := ℝ, 
  XZ := 5,
  YZ := 5,
  angleX := 45,
  angleY := 45,
  angleZ := 90
}

-- Lean statement to prove the area of triangle XYZ
theorem area_is_12_5 (t : Triangle) 
  (h1 : t.angleZ = 90)
  (h2 : t.angleX = 45)
  (h3 : t.angleY = 45)
  (h4 : t.XZ = 5)
  (h5 : t.YZ = 5) : 
  (1/2 * t.XZ * t.YZ) = 12.5 :=
sorry

end NUMINAMATH_GPT_area_is_12_5_l157_15745


namespace NUMINAMATH_GPT_abs_fraction_eq_sqrt_seven_thirds_l157_15737

open Real

theorem abs_fraction_eq_sqrt_seven_thirds {a b : ℝ} 
  (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h : a^2 + b^2 = 5 * a * b) : 
  abs ((a + b) / (a - b)) = sqrt (7 / 3) :=
by
  sorry

end NUMINAMATH_GPT_abs_fraction_eq_sqrt_seven_thirds_l157_15737


namespace NUMINAMATH_GPT_total_frames_l157_15771

def frames_per_page : ℝ := 143.0

def pages : ℝ := 11.0

theorem total_frames : frames_per_page * pages = 1573.0 :=
by
  sorry

end NUMINAMATH_GPT_total_frames_l157_15771


namespace NUMINAMATH_GPT_min_value_of_T_l157_15706

noncomputable def T_min_value (a b c : ℝ) : ℝ :=
  (5 + 2*a*b + 4*a*c) / (a*b + 1)

theorem min_value_of_T :
  ∀ (a b c : ℝ),
  a < 0 →
  b > 0 →
  b^2 ≤ (4 * c) / a →
  c ≤ (1/4) * a * b^2 →
  T_min_value a b c ≥ 4 ∧ (T_min_value a b c = 4 ↔ a * b = -3) :=
by
  intros
  sorry

end NUMINAMATH_GPT_min_value_of_T_l157_15706


namespace NUMINAMATH_GPT_sin_angle_identity_l157_15769

theorem sin_angle_identity : 
  (Real.sin (Real.pi / 4) * Real.sin (7 * Real.pi / 12) + Real.sin (Real.pi / 4) * Real.sin (Real.pi / 12)) = Real.sqrt 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_sin_angle_identity_l157_15769


namespace NUMINAMATH_GPT_average_licks_to_center_l157_15787

theorem average_licks_to_center (Dan_lcks Michael_lcks Sam_lcks David_lcks Lance_lcks : ℕ)
  (h1 : Dan_lcks = 58) 
  (h2 : Michael_lcks = 63) 
  (h3 : Sam_lcks = 70) 
  (h4 : David_lcks = 70) 
  (h5 : Lance_lcks = 39) :
  (Dan_lcks + Michael_lcks + Sam_lcks + David_lcks + Lance_lcks) / 5 = 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_licks_to_center_l157_15787


namespace NUMINAMATH_GPT_votes_cast_is_750_l157_15715

-- Define the conditions as Lean statements
def initial_score : ℤ := 0
def score_increase (likes : ℕ) : ℤ := likes
def score_decrease (dislikes : ℕ) : ℤ := -dislikes
def observed_score : ℤ := 150
def percent_likes : ℚ := 0.60

-- Express the proof
theorem votes_cast_is_750 (total_votes : ℕ) (likes : ℕ) (dislikes : ℕ) 
  (h1 : total_votes = likes + dislikes) 
  (h2 : percent_likes * total_votes = likes) 
  (h3 : dislikes = (1 - percent_likes) * total_votes)
  (h4 : observed_score = score_increase likes + score_decrease dislikes) :
  total_votes = 750 := 
sorry

end NUMINAMATH_GPT_votes_cast_is_750_l157_15715


namespace NUMINAMATH_GPT_calculate_total_income_l157_15717

/-- Total income calculation proof for a person with given distributions and remaining amount -/
theorem calculate_total_income
  (I : ℝ) -- total income
  (leftover : ℝ := 40000) -- leftover amount after distribution and donation
  (c1_percentage : ℝ := 3 * 0.15) -- percentage given to children
  (c2_percentage : ℝ := 0.30) -- percentage given to wife
  (c3_percentage : ℝ := 0.05) -- percentage donated to orphan house
  (remaining_percentage : ℝ := 1 - (c1_percentage + c2_percentage)) -- remaining percentage after children and wife
  (R : ℝ := remaining_percentage * I) -- remaining amount after children and wife
  (donation : ℝ := c3_percentage * R) -- amount donated to orphan house)
  (left_amount : ℝ := R - donation) -- final remaining amount
  (income : ℝ := (leftover / (1 - remaining_percentage * (1 - c3_percentage)))) -- calculation of the actual income
  : I = income := sorry

end NUMINAMATH_GPT_calculate_total_income_l157_15717


namespace NUMINAMATH_GPT_diameter_of_circular_field_l157_15720

theorem diameter_of_circular_field :
  ∀ (π : ℝ) (cost_per_meter total_cost circumference diameter : ℝ),
    π = Real.pi → 
    cost_per_meter = 1.50 → 
    total_cost = 94.24777960769379 → 
    circumference = total_cost / cost_per_meter →
    circumference = π * diameter →
    diameter = 20 := 
by
  intros π cost_per_meter total_cost circumference diameter hπ hcp ht cutoff_circ hcirc
  sorry

end NUMINAMATH_GPT_diameter_of_circular_field_l157_15720


namespace NUMINAMATH_GPT_star_polygon_x_value_l157_15702

theorem star_polygon_x_value
  (a b c d e p q r s t : ℝ)
  (h1 : p + q + r + s + t = 500)
  (h2 : a + b + c + d + e = x)
  :
  x = 140 :=
sorry

end NUMINAMATH_GPT_star_polygon_x_value_l157_15702


namespace NUMINAMATH_GPT_inequality_proof_l157_15795

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 0.3 / Real.log 2

theorem inequality_proof : (a * b < a + b ∧ a + b < 0) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l157_15795


namespace NUMINAMATH_GPT_no_x4_term_implies_a_zero_l157_15742

theorem no_x4_term_implies_a_zero (a : ℝ) :
  ¬ (∃ (x : ℝ), -5 * x^3 * (x^2 + a * x + 5) = -5 * x^5 - 5 * a * x^4 - 25 * x^3 + 5 * a * x^4) →
  a = 0 :=
by
  -- Step through the proof process to derive this conclusion
  sorry

end NUMINAMATH_GPT_no_x4_term_implies_a_zero_l157_15742


namespace NUMINAMATH_GPT_coordinates_of_C_l157_15775

theorem coordinates_of_C (A B : ℝ × ℝ) (C : ℝ × ℝ) 
    (hA : A = (1, 3)) (hB : B = (9, -3)) (hBC_AB : dist B C = 1/2 * dist A B) : 
    C = (13, -6) :=
sorry

end NUMINAMATH_GPT_coordinates_of_C_l157_15775


namespace NUMINAMATH_GPT_choose_student_B_l157_15732

-- Define the scores for students A and B
def scores_A : List ℕ := [72, 85, 86, 90, 92]
def scores_B : List ℕ := [76, 83, 85, 87, 94]

-- Function to calculate the average of scores
def average (scores : List ℕ) : ℚ :=
  scores.sum / scores.length

-- Function to calculate the variance of scores
def variance (scores : List ℕ) : ℚ :=
  let mean := average scores
  (scores.map (λ x => (x - mean) * (x - mean))).sum / scores.length

-- Calculate the average scores for A and B
def avg_A : ℚ := average scores_A
def avg_B : ℚ := average scores_B

-- Calculate the variances for A and B
def var_A : ℚ := variance scores_A
def var_B : ℚ := variance scores_B

-- The theorem to be proved
theorem choose_student_B : var_B < var_A :=
  by sorry

end NUMINAMATH_GPT_choose_student_B_l157_15732


namespace NUMINAMATH_GPT_functional_eq_f800_l157_15783

theorem functional_eq_f800
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y)
  (h2 : f 1000 = 6)
  : f 800 = 7.5 := by
  sorry

end NUMINAMATH_GPT_functional_eq_f800_l157_15783


namespace NUMINAMATH_GPT_ball_reaches_less_than_5_l157_15799

noncomputable def height_after_bounces (initial_height : ℕ) (ratio : ℝ) (bounces : ℕ) : ℝ :=
  initial_height * (ratio ^ bounces)

theorem ball_reaches_less_than_5 (initial_height : ℕ) (ratio : ℝ) (k : ℕ) (target_height : ℝ) (stop_height : ℝ) 
  (h_initial : initial_height = 500) (h_ratio : ratio = 0.6) (h_target : target_height = 5) (h_stop : stop_height = 0.1) :
  ∃ n, height_after_bounces initial_height ratio n < target_height ∧ 500 * (0.6 ^ 17) < stop_height := by
  sorry

end NUMINAMATH_GPT_ball_reaches_less_than_5_l157_15799


namespace NUMINAMATH_GPT_max_minute_hands_l157_15768

theorem max_minute_hands (m n : ℕ) (h : m * n = 27) : m + n ≤ 28 :=
  sorry

end NUMINAMATH_GPT_max_minute_hands_l157_15768


namespace NUMINAMATH_GPT_ratio_triangle_BFD_to_square_ABCE_l157_15726

-- Defining necessary components for the mathematical problem
def square_ABCE (x : ℝ) : ℝ := 16 * x^2
def triangle_BFD_area (x : ℝ) : ℝ := 7 * x^2

-- The theorem that needs to be proven, stating the ratio of the areas
theorem ratio_triangle_BFD_to_square_ABCE (x : ℝ) (hx : x > 0) :
  (triangle_BFD_area x) / (square_ABCE x) = 7 / 16 :=
by
  sorry

end NUMINAMATH_GPT_ratio_triangle_BFD_to_square_ABCE_l157_15726


namespace NUMINAMATH_GPT_smallest_rel_prime_to_180_l157_15709

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Int.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Int.gcd y 180 = 1 → x ≤ y := 
sorry

end NUMINAMATH_GPT_smallest_rel_prime_to_180_l157_15709


namespace NUMINAMATH_GPT_brownies_shared_l157_15756

theorem brownies_shared
  (total_brownies : ℕ)
  (tina_brownies : ℕ)
  (husband_brownies : ℕ)
  (remaining_brownies : ℕ)
  (shared_brownies : ℕ)
  (h1 : total_brownies = 24)
  (h2 : tina_brownies = 10)
  (h3 : husband_brownies = 5)
  (h4 : remaining_brownies = 5) :
  shared_brownies = total_brownies - (tina_brownies + husband_brownies + remaining_brownies) → shared_brownies = 4 :=
by
  sorry

end NUMINAMATH_GPT_brownies_shared_l157_15756


namespace NUMINAMATH_GPT_set_union_intersection_example_l157_15792

open Set

theorem set_union_intersection_example :
  let A := {1, 3, 4, 5}
  let B := {2, 4, 6}
  let C := {0, 1, 2, 3, 4}
  (A ∪ B) ∩ C = ({1, 2, 3, 4} : Set ℕ) :=
by
  sorry

end NUMINAMATH_GPT_set_union_intersection_example_l157_15792


namespace NUMINAMATH_GPT_profit_percentage_with_discount_l157_15754

theorem profit_percentage_with_discount
    (P M : ℝ)
    (h1 : M = 1.27 * P)
    (h2 : 0 < P) :
    ((0.95 * M - P) / P) * 100 = 20.65 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_with_discount_l157_15754


namespace NUMINAMATH_GPT_solve_inequality_system_l157_15710

theorem solve_inequality_system (x : ℝ) (h1 : 3 * x - 2 < x) (h2 : (1 / 3) * x < -2) : x < -6 :=
sorry

end NUMINAMATH_GPT_solve_inequality_system_l157_15710


namespace NUMINAMATH_GPT_volume_of_solid_l157_15785

noncomputable def s : ℝ := 2 * Real.sqrt 2

noncomputable def h : ℝ := 3 * s

noncomputable def base_area (a b : ℝ) : ℝ := 1 / 2 * a * b

noncomputable def volume (base_area height : ℝ) : ℝ := base_area * height

theorem volume_of_solid : volume (base_area s s) h = 24 * Real.sqrt 2 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_volume_of_solid_l157_15785


namespace NUMINAMATH_GPT_min_value_x_plus_2y_l157_15723

theorem min_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = x * y) : x + 2 * y ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_x_plus_2y_l157_15723


namespace NUMINAMATH_GPT_log_ordering_l157_15757

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 8 / Real.log 4
noncomputable def c : ℝ := Real.log 10 / Real.log 5

theorem log_ordering : a > b ∧ b > c :=
by {
  sorry
}

end NUMINAMATH_GPT_log_ordering_l157_15757


namespace NUMINAMATH_GPT_sum_of_15th_set_l157_15704

def first_element_of_set (n : ℕ) : ℕ :=
  3 + (n * (n - 1)) / 2

def sum_of_elements_in_set (n : ℕ) : ℕ :=
  let a_n := first_element_of_set n
  let l_n := a_n + n - 1
  n * (a_n + l_n) / 2

theorem sum_of_15th_set :
  sum_of_elements_in_set 15 = 1725 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_15th_set_l157_15704


namespace NUMINAMATH_GPT_cost_of_coat_eq_l157_15779

-- Define the given conditions
def total_cost : ℕ := 110
def cost_of_shoes : ℕ := 30
def cost_per_jeans : ℕ := 20
def num_of_jeans : ℕ := 2

-- Define the cost calculation for the jeans
def cost_of_jeans : ℕ := num_of_jeans * cost_per_jeans

-- Define the known total cost (shoes and jeans)
def known_total_cost : ℕ := cost_of_shoes + cost_of_jeans

-- Prove James' coat cost
theorem cost_of_coat_eq :
  (total_cost - known_total_cost) = 40 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_coat_eq_l157_15779


namespace NUMINAMATH_GPT_min_value_functions_l157_15711

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 1 / x^2
noncomputable def f_B (x : ℝ) : ℝ := 2 * x + 2 / x
noncomputable def f_C (x : ℝ) : ℝ := (x - 1) / (x + 1)
noncomputable def f_D (x : ℝ) : ℝ := Real.log (Real.sqrt x + 1)

theorem min_value_functions :
  (∃ x : ℝ, ∀ y : ℝ, f_A x ≤ f_A y) ∧
  (∃ x : ℝ, ∀ y : ℝ, f_D x ≤ f_D y) ∧
  ¬ (∃ x : ℝ, ∀ y : ℝ, f_B x ≤ f_B y) ∧
  ¬ (∃ x : ℝ, ∀ y : ℝ, f_C x ≤ f_C y) :=
by
  sorry

end NUMINAMATH_GPT_min_value_functions_l157_15711


namespace NUMINAMATH_GPT_triangle_square_side_ratio_l157_15766

theorem triangle_square_side_ratio :
  (∀ (a : ℝ), (a * 3 = 60) → (∀ (b : ℝ), (b * 4 = 60) → (a / b = 4 / 3))) :=
by
  intros a h1 b h2
  sorry

end NUMINAMATH_GPT_triangle_square_side_ratio_l157_15766


namespace NUMINAMATH_GPT_correct_factorization_A_l157_15784

-- Define the polynomial expressions
def expression_A : Prop :=
  (x : ℝ) → x^2 - x - 6 = (x + 2) * (x - 3)

def expression_B : Prop :=
  (x : ℝ) → x^2 - 1 = x * (x - 1 / x)

def expression_C : Prop :=
  (x y : ℝ) → 7 * x^2 * y^5 = x * y * 7 * x * y^4

def expression_D : Prop :=
  (x : ℝ) → x^2 + 4 * x + 4 = x * (x + 4) + 4

-- The correct factorization from left to right
theorem correct_factorization_A : expression_A := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_correct_factorization_A_l157_15784


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l157_15719

-- Define the set A
def A : Set ℝ := {-1, 0, 1}

-- Define the set B based on the given conditions
def B : Set ℝ := {y | ∃ x ∈ A, y = Real.cos (Real.pi * x)}

-- The main theorem to prove that A ∩ B is {-1, 1}
theorem intersection_of_A_and_B : A ∩ B = {-1, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l157_15719


namespace NUMINAMATH_GPT_total_number_of_workers_l157_15762

theorem total_number_of_workers 
  (W : ℕ) 
  (h_all_avg : W * 8000 = 10 * 12000 + (W - 10) * 6000) : 
  W = 30 := 
by
  sorry

end NUMINAMATH_GPT_total_number_of_workers_l157_15762


namespace NUMINAMATH_GPT_thirty_two_not_sum_consecutive_natural_l157_15780

theorem thirty_two_not_sum_consecutive_natural (n k : ℕ) : 
  (n > 0) → (32 ≠ (n * (2 * k + n - 1)) / 2) :=
by
  sorry

end NUMINAMATH_GPT_thirty_two_not_sum_consecutive_natural_l157_15780


namespace NUMINAMATH_GPT_angela_insects_l157_15753

theorem angela_insects (A J D : ℕ) (h1 : A = J / 2) (h2 : J = 5 * D) (h3 : D = 30) : A = 75 :=
by
  sorry

end NUMINAMATH_GPT_angela_insects_l157_15753


namespace NUMINAMATH_GPT_polynomial_non_negative_l157_15760

theorem polynomial_non_negative (a : ℝ) : a^2 * (a^2 - 1) - a^2 + 1 ≥ 0 := by
  -- we would include the proof steps here
  sorry

end NUMINAMATH_GPT_polynomial_non_negative_l157_15760


namespace NUMINAMATH_GPT_distance_between_points_l157_15736

theorem distance_between_points {A B : ℝ}
  (hA : abs A = 3)
  (hB : abs B = 9) :
  abs (A - B) = 6 ∨ abs (A - B) = 12 :=
sorry

end NUMINAMATH_GPT_distance_between_points_l157_15736


namespace NUMINAMATH_GPT_people_and_carriages_condition_l157_15781

-- Definitions corresponding to the conditions
def num_people_using_carriages (x : ℕ) : ℕ := 3 * (x - 2)
def num_people_sharing_carriages (x : ℕ) : ℕ := 2 * x + 9

-- The theorem statement we need to prove
theorem people_and_carriages_condition (x : ℕ) : 
  num_people_using_carriages x = num_people_sharing_carriages x ↔ 3 * (x - 2) = 2 * x + 9 :=
by sorry

end NUMINAMATH_GPT_people_and_carriages_condition_l157_15781


namespace NUMINAMATH_GPT_smallest_n_power_mod_5_l157_15770

theorem smallest_n_power_mod_5 :
  ∃ N : ℕ, 100 ≤ N ∧ N ≤ 999 ∧ (2^N + 1) % 5 = 0 ∧ ∀ M : ℕ, 100 ≤ M ∧ M ≤ 999 ∧ (2^M + 1) % 5 = 0 → N ≤ M := 
sorry

end NUMINAMATH_GPT_smallest_n_power_mod_5_l157_15770


namespace NUMINAMATH_GPT_probability_interval_l157_15716

theorem probability_interval (P_A P_B P_A_inter_P_B : ℝ) (h1 : P_A = 3 / 4) (h2 : P_B = 2 / 3) : 
  5/12 ≤ P_A_inter_P_B ∧ P_A_inter_P_B ≤ 2/3 :=
sorry

end NUMINAMATH_GPT_probability_interval_l157_15716


namespace NUMINAMATH_GPT_not_necessarily_true_l157_15746

theorem not_necessarily_true (x y : ℝ) (h : x > y) : ¬ (x^2 > y^2) :=
sorry

end NUMINAMATH_GPT_not_necessarily_true_l157_15746


namespace NUMINAMATH_GPT_alpha_numeric_puzzle_l157_15790

theorem alpha_numeric_puzzle : 
  ∀ (a b c d e f g h i : ℕ),
  (∀ x y : ℕ, x ≠ 0 → y ≠ 0 → x ≠ y) →
  100 * a + 10 * b + c + 100 * d + 10 * e + f + 100 * g + 10 * h + i = 1665 → 
  c + f + i = 15 →
  b + e + h = 15 :=
by
  intros a b c d e f g h i distinct nonzero_sum unit_digits_sum
  sorry

end NUMINAMATH_GPT_alpha_numeric_puzzle_l157_15790


namespace NUMINAMATH_GPT_general_term_sequence_l157_15713

theorem general_term_sequence (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n + 3^n) :
  ∀ n, a n = (3^n - 1) / 2 := 
by
  sorry

end NUMINAMATH_GPT_general_term_sequence_l157_15713


namespace NUMINAMATH_GPT_train_length_calculation_l157_15782

noncomputable def length_of_train (speed : ℝ) (time_in_sec : ℝ) : ℝ :=
  let time_in_hr := time_in_sec / 3600
  let distance_in_km := speed * time_in_hr
  distance_in_km * 1000

theorem train_length_calculation : 
  length_of_train 60 30 = 500 :=
by
  -- The proof would go here, but we provide a stub with sorry.
  sorry

end NUMINAMATH_GPT_train_length_calculation_l157_15782


namespace NUMINAMATH_GPT_Nord_Stream_pipeline_payment_l157_15738

/-- Suppose Russia, Germany, and France decided to build the "Nord Stream 2" pipeline,
     which is 1200 km long, agreeing to finance this project equally.
     Russia built 650 kilometers of the pipeline.
     Germany built 550 kilometers of the pipeline.
     France contributed its share in money and did not build any kilometers.
     Germany received 1.2 billion euros from France.
     Prove that Russia should receive 2 billion euros from France.
--/
theorem Nord_Stream_pipeline_payment
  (total_km : ℝ)
  (russia_km : ℝ)
  (germany_km : ℝ)
  (total_countries : ℝ)
  (payment_to_germany : ℝ)
  (germany_additional_payment : ℝ)
  (france_km : ℝ)
  (france_payment_ratio : ℝ)
  (russia_payment : ℝ) :
  total_km = 1200 ∧
  russia_km = 650 ∧
  germany_km = 550 ∧
  total_countries = 3 ∧
  payment_to_germany = 1.2 ∧
  france_km = 0 ∧
  germany_additional_payment = germany_km - (total_km / total_countries) ∧
  france_payment_ratio = 5 / 3 ∧
  russia_payment = payment_to_germany * (5 / 3) →
  russia_payment = 2 := by sorry

end NUMINAMATH_GPT_Nord_Stream_pipeline_payment_l157_15738


namespace NUMINAMATH_GPT_half_abs_diff_squares_l157_15749

theorem half_abs_diff_squares (a b : ℝ) (h₁ : a = 25) (h₂ : b = 20) :
  (1 / 2) * |a^2 - b^2| = 112.5 :=
sorry

end NUMINAMATH_GPT_half_abs_diff_squares_l157_15749


namespace NUMINAMATH_GPT_Joey_downhill_speed_l157_15788

theorem Joey_downhill_speed
  (Route_length : ℝ) (Time_uphill : ℝ) (Speed_uphill : ℝ) (Overall_average_speed : ℝ) (Extra_time_due_to_rain : ℝ) :
  Route_length = 5 →
  Time_uphill = 1.25 →
  Speed_uphill = 4 →
  Overall_average_speed = 6 →
  Extra_time_due_to_rain = 0.25 →
  ((2 * Route_length) / Overall_average_speed - Time_uphill - Extra_time_due_to_rain) * (Route_length / (2 * Route_length / Overall_average_speed - Time_uphill - Extra_time_due_to_rain)) = 30 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_Joey_downhill_speed_l157_15788


namespace NUMINAMATH_GPT_avg_weight_B_correct_l157_15705

-- Definitions of the conditions
def students_A : ℕ := 24
def students_B : ℕ := 16
def avg_weight_A : ℝ := 40
def avg_weight_class : ℝ := 38

-- Definition of the total weight calculation for sections A and B
def total_weight_A : ℝ := students_A * avg_weight_A
def total_weight_class : ℝ := (students_A + students_B) * avg_weight_class

-- Defining the average weight of section B as the unknown to be proven
noncomputable def avg_weight_B : ℝ := 35

-- The theorem to prove that the average weight of section B is 35 kg
theorem avg_weight_B_correct : 
  total_weight_A + students_B * avg_weight_B = total_weight_class :=
by
  sorry

end NUMINAMATH_GPT_avg_weight_B_correct_l157_15705


namespace NUMINAMATH_GPT_sum_of_digits_Joey_age_twice_Max_next_l157_15764

noncomputable def Joey_is_two_years_older (C : ℕ) : ℕ := C + 2

noncomputable def Max_age_today := 2

noncomputable def Eight_multiples_of_Max (C : ℕ) := 
  ∃ n : ℕ, C = 24 + n

noncomputable def Next_Joey_age_twice_Max (C J M n : ℕ): Prop := J + n = 2 * (M + n)

theorem sum_of_digits_Joey_age_twice_Max_next (C J M n : ℕ) 
  (h1: J = Joey_is_two_years_older C)
  (h2: M = Max_age_today)
  (h3: Eight_multiples_of_Max C)
  (h4: Next_Joey_age_twice_Max C J M n) 
  : ∃ s, s = 7 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_Joey_age_twice_Max_next_l157_15764


namespace NUMINAMATH_GPT_range_of_f_l157_15741

noncomputable def f (t : ℝ) : ℝ := (t^2 + 2 * t) / (t^2 + 2)

theorem range_of_f : Set.range f = Set.Icc (-1 : ℝ) 2 :=
sorry

end NUMINAMATH_GPT_range_of_f_l157_15741


namespace NUMINAMATH_GPT_four_units_away_l157_15776

theorem four_units_away (x : ℤ) (h : abs (x + 2) = 4) : x = 2 ∨ x = -6 :=
by
  sorry

end NUMINAMATH_GPT_four_units_away_l157_15776


namespace NUMINAMATH_GPT_find_x_l157_15747

variable (a b c d e f g h x : ℤ)

def cell_relationships (a b c d e f g h x : ℤ) : Prop :=
  (a = 10) ∧
  (h = 3) ∧
  (a = 10 + b) ∧
  (b = c + a) ∧
  (c = b + d) ∧
  (d = c + h) ∧
  (e = 10 + f) ∧
  (f = e + g) ∧
  (g = d + h) ∧
  (h = g + x)

theorem find_x : cell_relationships a b c d e f g h x → x = 7 :=
sorry

end NUMINAMATH_GPT_find_x_l157_15747


namespace NUMINAMATH_GPT_amber_josh_departure_time_l157_15707

def latest_departure_time (flight_time : ℕ) (check_in_time : ℕ) (drive_time : ℕ) (parking_time : ℕ) :=
  flight_time - check_in_time - drive_time - parking_time

theorem amber_josh_departure_time :
  latest_departure_time 20 2 (45 / 60) (15 / 60) = 17 :=
by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_amber_josh_departure_time_l157_15707


namespace NUMINAMATH_GPT_x_squared_minus_y_squared_l157_15751

theorem x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 3) : x^2 - y^2 = 27 := by
  sorry

end NUMINAMATH_GPT_x_squared_minus_y_squared_l157_15751


namespace NUMINAMATH_GPT_inequality_abc_l157_15739

theorem inequality_abc {a b c : ℝ} {n : ℕ} 
  (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) (hn : 0 < n) :
  (1 / (1 + a)^(1 / n : ℝ)) + (1 / (1 + b)^(1 / n : ℝ)) + (1 / (1 + c)^(1 / n : ℝ)) 
  ≤ 3 / (1 + (a * b * c)^(1 / 3 : ℝ))^(1 / n : ℝ) := sorry

end NUMINAMATH_GPT_inequality_abc_l157_15739


namespace NUMINAMATH_GPT_david_older_than_rosy_l157_15777

theorem david_older_than_rosy
  (R D : ℕ) 
  (h1 : R = 12) 
  (h2 : D + 6 = 2 * (R + 6)) : 
  D - R = 18 := 
by
  sorry

end NUMINAMATH_GPT_david_older_than_rosy_l157_15777


namespace NUMINAMATH_GPT_regression_equation_represents_real_relationship_maximized_l157_15733

-- Definitions from the conditions
def regression_equation (y x : ℝ) := ∃ (a b : ℝ), y = a * x + b

def represents_real_relationship_maximized (y x : ℝ) := regression_equation y x

-- The proof problem statement
theorem regression_equation_represents_real_relationship_maximized 
: ∀ (y x : ℝ), regression_equation y x → represents_real_relationship_maximized y x :=
by
  sorry

end NUMINAMATH_GPT_regression_equation_represents_real_relationship_maximized_l157_15733


namespace NUMINAMATH_GPT_words_with_at_least_one_consonant_l157_15718

-- Define the letters available and classify them as vowels and consonants
def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'C', 'D', 'F']

-- Define the total number of 5-letter words using the given letters
def total_words : ℕ := 6^5

-- Define the total number of 5-letter words composed exclusively of vowels
def vowel_words : ℕ := 2^5

-- Define the number of 5-letter words that contain at least one consonant
noncomputable def words_with_consonant : ℕ := total_words - vowel_words

-- The theorem to prove
theorem words_with_at_least_one_consonant : words_with_consonant = 7744 := by
  sorry

end NUMINAMATH_GPT_words_with_at_least_one_consonant_l157_15718


namespace NUMINAMATH_GPT_original_fraction_eq_two_thirds_l157_15767

theorem original_fraction_eq_two_thirds (a b : ℕ) (h : (a^3 : ℚ) / (b + 3) = 2 * (a / b)) : a = 2 ∧ b = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_original_fraction_eq_two_thirds_l157_15767


namespace NUMINAMATH_GPT_solve_z_squared_eq_l157_15703

open Complex

theorem solve_z_squared_eq : 
  ∀ z : ℂ, z^2 = -100 - 64 * I → (z = 4 - 8 * I ∨ z = -4 + 8 * I) :=
by
  sorry

end NUMINAMATH_GPT_solve_z_squared_eq_l157_15703


namespace NUMINAMATH_GPT_find_m_when_lines_parallel_l157_15793

theorem find_m_when_lines_parallel (m : ℝ) :
  (∀ x y : ℝ, x + (1 + m) * y = 2 - m) ∧ (∀ x y : ℝ, 2 * m * x + 4 * y = -16) →
  ∃ m : ℝ, m = 1 :=
sorry

end NUMINAMATH_GPT_find_m_when_lines_parallel_l157_15793


namespace NUMINAMATH_GPT_total_flour_needed_l157_15712

-- Definitions of flour needed by Katie and Sheila
def katie_flour : ℕ := 3
def sheila_flour : ℕ := katie_flour + 2

-- Statement of the theorem
theorem total_flour_needed : katie_flour + sheila_flour = 8 := by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_total_flour_needed_l157_15712


namespace NUMINAMATH_GPT_speed_of_j_l157_15755

theorem speed_of_j (j p : ℝ) 
  (h_faster : j > p)
  (h_distance_j : 24 / j = 24 / j)
  (h_distance_p : 24 / p = 24 / p)
  (h_sum_speeds : j + p = 7)
  (h_sum_times : 24 / j + 24 / p = 14) : j = 4 := 
sorry

end NUMINAMATH_GPT_speed_of_j_l157_15755


namespace NUMINAMATH_GPT_paint_rate_l157_15721

theorem paint_rate (l b : ℝ) (cost : ℕ) (rate_per_sq_m : ℝ) 
  (h1 : l = 3 * b) 
  (h2 : cost = 300) 
  (h3 : l = 13.416407864998739) 
  (area : ℝ := l * b) : 
  rate_per_sq_m = 5 :=
by
  sorry

end NUMINAMATH_GPT_paint_rate_l157_15721


namespace NUMINAMATH_GPT_work_schedules_lcm_l157_15786

theorem work_schedules_lcm : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9) = 360 := 
by 
  sorry

end NUMINAMATH_GPT_work_schedules_lcm_l157_15786


namespace NUMINAMATH_GPT_find_common_difference_find_max_sum_find_max_n_l157_15773

-- Condition for the sequence
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Problem statement (1): Find the common difference
theorem find_common_difference (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 = 23)
  (h2 : is_arithmetic_sequence a d)
  (h6 : a 6 > 0)
  (h7 : a 7 < 0) : d = -4 :=
sorry

-- Problem statement (2): Find the maximum value of the sum S₆
theorem find_max_sum (d : ℤ) (h : d = -4) : 6 * 23 + (6 * 5 / 2) * d = 78 :=
sorry

-- Problem statement (3): Find the maximum value of n when S_n > 0
theorem find_max_n (d : ℤ) (h : d = -4) : ∀ n : ℕ, (n > 0 ∧ (23 * n + (n * (n - 1) / 2) * d > 0)) → n ≤ 12 :=
sorry

end NUMINAMATH_GPT_find_common_difference_find_max_sum_find_max_n_l157_15773


namespace NUMINAMATH_GPT_Scarlett_adds_correct_amount_l157_15708

-- Define the problem with given conditions
def currentOilAmount : ℝ := 0.17
def desiredOilAmount : ℝ := 0.84

-- Prove that the amount of oil Scarlett needs to add is 0.67 cup
theorem Scarlett_adds_correct_amount : (desiredOilAmount - currentOilAmount) = 0.67 := by
  sorry

end NUMINAMATH_GPT_Scarlett_adds_correct_amount_l157_15708


namespace NUMINAMATH_GPT_factorization_4x2_minus_144_l157_15700

theorem factorization_4x2_minus_144 (x : ℝ) : 4 * x^2 - 144 = 4 * (x - 6) * (x + 6) := 
  sorry

end NUMINAMATH_GPT_factorization_4x2_minus_144_l157_15700


namespace NUMINAMATH_GPT_height_percentage_increase_l157_15725

theorem height_percentage_increase (B A : ℝ) 
  (hA : A = B * 0.8) : ((B - A) / A) * 100 = 25 := by
--   Given the condition that A's height is 20% less than B's height
--   translate into A = B * 0.8
--   We need to show ((B - A) / A) * 100 = 25
sorry

end NUMINAMATH_GPT_height_percentage_increase_l157_15725


namespace NUMINAMATH_GPT_a_b_c_at_least_one_not_less_than_one_third_l157_15729

theorem a_b_c_at_least_one_not_less_than_one_third (a b c : ℝ) (h : a + b + c = 1) :
  ¬ (a < 1/3 ∧ b < 1/3 ∧ c < 1/3) :=
by
  sorry

end NUMINAMATH_GPT_a_b_c_at_least_one_not_less_than_one_third_l157_15729


namespace NUMINAMATH_GPT_num_valid_a_values_l157_15778

theorem num_valid_a_values : 
  ∃ S : Finset ℕ, (∀ a ∈ S, a < 100 ∧ (a^3 + 23) % 24 = 0) ∧ S.card = 5 :=
sorry

end NUMINAMATH_GPT_num_valid_a_values_l157_15778


namespace NUMINAMATH_GPT_speed_of_stream_l157_15765

theorem speed_of_stream (b s : ℝ) 
  (H1 : b + s = 10)
  (H2 : b - s = 4) : 
  s = 3 :=
sorry

end NUMINAMATH_GPT_speed_of_stream_l157_15765


namespace NUMINAMATH_GPT_impossible_triangle_angle_sum_l157_15722

theorem impossible_triangle_angle_sum (x y z : ℝ) (h : x + y + z = 180) : x + y + z ≠ 360 :=
by
sorry

end NUMINAMATH_GPT_impossible_triangle_angle_sum_l157_15722


namespace NUMINAMATH_GPT_domain_of_log_l157_15735

def log_domain := {x : ℝ | x > 1}

theorem domain_of_log : {x : ℝ | ∃ y, y = log_domain} = {x : ℝ | x > 1} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_log_l157_15735


namespace NUMINAMATH_GPT_area_gray_region_correct_l157_15740

def center_C : ℝ × ℝ := (3, 5)
def radius_C : ℝ := 3
def center_D : ℝ × ℝ := (9, 5)
def radius_D : ℝ := 3

noncomputable def area_gray_region : ℝ :=
  let rectangle_area := (center_D.1 - center_C.1) * (center_C.2 - (center_C.2 - radius_C))
  let sector_area := (1 / 4) * radius_C ^ 2 * Real.pi
  rectangle_area - 2 * sector_area

theorem area_gray_region_correct :
  area_gray_region = 18 - 9 / 2 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_gray_region_correct_l157_15740


namespace NUMINAMATH_GPT_solve_equation_l157_15759

theorem solve_equation (x : ℚ) (h : x ≠ 3) : (x + 5) / (x - 3) = 4 ↔ x = 17 / 3 :=
sorry

end NUMINAMATH_GPT_solve_equation_l157_15759
