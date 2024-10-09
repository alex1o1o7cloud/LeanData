import Mathlib

namespace triangle_isosceles_or_right_angled_l696_69607

theorem triangle_isosceles_or_right_angled
  (β γ : ℝ)
  (h : Real.tan β * Real.sin γ ^ 2 = Real.tan γ * Real.sin β ^ 2) :
  (β = γ ∨ β + γ = Real.pi / 2) :=
sorry

end triangle_isosceles_or_right_angled_l696_69607


namespace inequality_mn_l696_69617

theorem inequality_mn (m n : ℤ)
  (h : ∃ x : ℤ, (x + m) * (x + n) = x + m + n) : 
  2 * (m^2 + n^2) < 5 * m * n := 
sorry

end inequality_mn_l696_69617


namespace cos_neg_300_eq_positive_half_l696_69612

theorem cos_neg_300_eq_positive_half : Real.cos (-300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_neg_300_eq_positive_half_l696_69612


namespace horizontal_distance_is_0_65_l696_69671

def parabola (x : ℝ) : ℝ := 2 * x^2 - 3 * x - 4

-- Calculate the horizontal distance between two points on the parabola given their y-coordinates and prove it equals to 0.65
theorem horizontal_distance_is_0_65 :
  ∃ (x1 x2 : ℝ), 
    parabola x1 = 10 ∧ parabola x2 = 0 ∧ abs (x1 - x2) = 0.65 :=
sorry

end horizontal_distance_is_0_65_l696_69671


namespace fraction_spent_at_toy_store_l696_69614

theorem fraction_spent_at_toy_store 
  (total_allowance : ℝ)
  (arcade_fraction : ℝ)
  (candy_store_amount : ℝ) 
  (remaining_allowance : ℝ)
  (toy_store_amount : ℝ)
  (H1 : total_allowance = 2.40)
  (H2 : arcade_fraction = 3 / 5)
  (H3 : candy_store_amount = 0.64)
  (H4 : remaining_allowance = total_allowance - (arcade_fraction * total_allowance))
  (H5 : toy_store_amount = remaining_allowance - candy_store_amount) :
  toy_store_amount / remaining_allowance = 1 / 3 := 
sorry

end fraction_spent_at_toy_store_l696_69614


namespace sarahs_score_l696_69660

theorem sarahs_score (g s : ℕ) (h₁ : s = g + 30) (h₂ : (s + g) / 2 = 95) : s = 110 := by
  sorry

end sarahs_score_l696_69660


namespace books_sold_to_used_bookstore_l696_69601

-- Conditions
def initial_books := 72
def books_from_club := 1 * 12
def books_from_bookstore := 5
def books_from_yardsales := 2
def books_from_daughter := 1
def books_from_mother := 4
def books_donated := 12
def books_end_of_year := 81

-- Proof problem
theorem books_sold_to_used_bookstore :
  initial_books
  + books_from_club
  + books_from_bookstore
  + books_from_yardsales
  + books_from_daughter
  + books_from_mother
  - books_donated
  - books_end_of_year
  = 3 := by
  -- calculation omitted
  sorry

end books_sold_to_used_bookstore_l696_69601


namespace problem_I4_1_l696_69682

theorem problem_I4_1 (a : ℝ) : ((∃ y : ℝ, x + 2 * y + 3 = 0) ∧ (∃ y : ℝ, 4 * x - a * y + 5 = 0) ∧ 
  (∃ m1 m2 : ℝ, m1 = -(1 / 2) ∧ m2 = 4 / a ∧ m1 * m2 = -1)) → a = 2 :=
sorry

end problem_I4_1_l696_69682


namespace xyz_value_l696_69611

theorem xyz_value
  (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19)
  (h3 : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 = 11) :
  x * y * z = 26 / 3 :=
sorry

end xyz_value_l696_69611


namespace goldfish_growth_solution_l696_69634

def goldfish_growth_problem : Prop :=
  ∃ n : ℕ, 
    (∀ k, (k < n → 3 * (5:ℕ)^k ≠ 243 * (3:ℕ)^k)) ∧
    3 * (5:ℕ)^n = 243 * (3:ℕ)^n

theorem goldfish_growth_solution : goldfish_growth_problem :=
sorry

end goldfish_growth_solution_l696_69634


namespace wheel_moves_distance_in_one_hour_l696_69684

-- Definition of the given conditions
def rotations_per_minute : ℕ := 10
def distance_per_rotation : ℕ := 20
def minutes_per_hour : ℕ := 60

-- Theorem statement to prove the wheel moves 12000 cm in one hour
theorem wheel_moves_distance_in_one_hour : 
  rotations_per_minute * minutes_per_hour * distance_per_rotation = 12000 := 
by
  sorry

end wheel_moves_distance_in_one_hour_l696_69684


namespace arithmetic_geometric_sequence_sum_l696_69651

theorem arithmetic_geometric_sequence_sum 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ x y z : ℝ, (x = a ∧ y = -4 ∧ z = b ∨ x = b ∧ y = -4 ∧ z = a) 
                   ∧ (x + z = 2 * y) ∧ (x * z = y^2)) : 
  a + b = 10 :=
by sorry

end arithmetic_geometric_sequence_sum_l696_69651


namespace baseball_games_per_month_l696_69688

theorem baseball_games_per_month (total_games : ℕ) (season_length : ℕ) (games_per_month : ℕ) :
  total_games = 14 → season_length = 2 → games_per_month = total_games / season_length → games_per_month = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end baseball_games_per_month_l696_69688


namespace simplify_expression_l696_69615

variable (y : ℝ)

theorem simplify_expression : 
  (3 * y - 2) * (5 * y ^ 12 + 3 * y ^ 11 + 5 * y ^ 9 + 3 * y ^ 8) = 
  15 * y ^ 13 - y ^ 12 + 3 * y ^ 11 + 15 * y ^ 10 - y ^ 9 - 6 * y ^ 8 :=
by
  sorry

end simplify_expression_l696_69615


namespace count_males_not_in_orchestra_l696_69638

variable (females_band females_orchestra females_choir females_all
          males_band males_orchestra males_choir males_all total_students : ℕ)
variable (males_band_not_in_orchestra : ℕ)

theorem count_males_not_in_orchestra :
  females_band = 120 ∧ females_orchestra = 90 ∧ females_choir = 50 ∧ females_all = 30 ∧
  males_band = 90 ∧ males_orchestra = 120 ∧ males_choir = 40 ∧ males_all = 20 ∧
  total_students = 250 ∧ males_band_not_in_orchestra = (males_band - (males_band + males_orchestra + males_choir - males_all - total_students)) 
  → males_band_not_in_orchestra = 20 :=
by
  intros
  sorry

end count_males_not_in_orchestra_l696_69638


namespace sum_of_squares_of_roots_l696_69608

theorem sum_of_squares_of_roots :
  ∀ r1 r2 : ℝ, (r1 + r2 = 14) ∧ (r1 * r2 = 8) → (r1^2 + r2^2 = 180) := by
  sorry

end sum_of_squares_of_roots_l696_69608


namespace supplement_greater_than_complement_l696_69641

variable (angle1 : ℝ)

def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

theorem supplement_greater_than_complement (h : is_acute angle1) :
  180 - angle1 = 90 + (90 - angle1) :=
by {
  sorry
}

end supplement_greater_than_complement_l696_69641


namespace value_of_f_sum_l696_69692

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (h_odd : ∀ x, f (-x) = -f x) : Prop
axiom period_9 (h_period : ∀ x, f (x + 9) = f x) : Prop
axiom f_one (h_f1 : f 1 = 5) : Prop

theorem value_of_f_sum (h_odd : ∀ x, f (-x) = -f x)
                       (h_period : ∀ x, f (x + 9) = f x)
                       (h_f1 : f 1 = 5) :
  f 2007 + f 2008 = 5 :=
sorry

end value_of_f_sum_l696_69692


namespace inequality_solution_set_l696_69647

theorem inequality_solution_set {m n : ℝ} (h : ∀ x : ℝ, -3 < x ∧ x < 6 ↔ x^2 - m * x - 6 * n < 0) : m + n = 6 :=
by
  sorry

end inequality_solution_set_l696_69647


namespace ratio_lt_one_l696_69674

def product_sequence (k j : ℕ) := List.prod (List.range' k j)

theorem ratio_lt_one :
  let a := product_sequence 2020 4
  let b := product_sequence 2120 4
  a / b < 1 :=
by
  sorry

end ratio_lt_one_l696_69674


namespace smaller_of_two_digit_numbers_with_product_2210_l696_69600

theorem smaller_of_two_digit_numbers_with_product_2210 :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 2210 ∧ a ≤ b ∧ a = 26 :=
by
  sorry

end smaller_of_two_digit_numbers_with_product_2210_l696_69600


namespace percentage_difference_l696_69605

theorem percentage_difference:
  let x1 := 0.4 * 60
  let x2 := 0.8 * 25
  x1 - x2 = 4 :=
by
  sorry

end percentage_difference_l696_69605


namespace servings_required_l696_69695

/-- Each serving of cereal is 2.0 cups, and 36 cups are needed. Prove that the number of servings required is 18. -/
theorem servings_required (cups_per_serving : ℝ) (total_cups : ℝ) (h1 : cups_per_serving = 2.0) (h2 : total_cups = 36.0) :
  total_cups / cups_per_serving = 18 :=
by
  sorry

end servings_required_l696_69695


namespace find_x_l696_69631

theorem find_x
  (x : ℝ)
  (h1 : (x - 2)^2 + (15 - 5)^2 = 13^2)
  (h2 : x > 0) : 
  x = 2 + Real.sqrt 69 :=
sorry

end find_x_l696_69631


namespace propositions_correct_l696_69645

def vertical_angles (α β : ℝ) : Prop := ∃ γ, α = γ ∧ β = γ

def problem_statement : Prop :=
  (∀ α β, vertical_angles α β → α = β) ∧
  ¬(∀ α β, α = β → vertical_angles α β) ∧
  ¬(∀ α β, ¬vertical_angles α β → ¬(α = β)) ∧
  (∀ α β, ¬(α = β) → ¬vertical_angles α β)

theorem propositions_correct :
  problem_statement :=
by
  sorry

end propositions_correct_l696_69645


namespace sufficient_but_not_necessary_condition_l696_69678

theorem sufficient_but_not_necessary_condition (f : ℝ → ℝ) (h : ∀ x, f x = x⁻¹) :
  ∀ x, (x > 1 → f (x + 2) > f (2*x + 1)) ∧ (¬ (x > 1) → ¬ (f (x + 2) > f (2*x + 1))) :=
by
  sorry

end sufficient_but_not_necessary_condition_l696_69678


namespace percent_decrease_l696_69639

theorem percent_decrease(call_cost_1980 call_cost_2010 : ℝ) (h₁ : call_cost_1980 = 50) (h₂ : call_cost_2010 = 5) :
  ((call_cost_1980 - call_cost_2010) / call_cost_1980 * 100) = 90 :=
by
  sorry

end percent_decrease_l696_69639


namespace theta_plus_2phi_eq_pi_div_4_l696_69655

noncomputable def theta (θ : ℝ) (φ : ℝ) : Prop := 
  ((Real.tan θ = 5 / 12) ∧ 
   (Real.sin φ = 1 / 2) ∧ 
   (0 < θ ∧ θ < Real.pi / 2) ∧ 
   (0 < φ ∧ φ < Real.pi / 2)  )

theorem theta_plus_2phi_eq_pi_div_4 (θ φ : ℝ) (h : theta θ φ) : 
    θ + 2 * φ = Real.pi / 4 :=
by 
  sorry

end theta_plus_2phi_eq_pi_div_4_l696_69655


namespace find_son_l696_69672

variable (SonAge ManAge : ℕ)

def age_relationship (SonAge ManAge : ℕ) : Prop :=
  ManAge = SonAge + 20 ∧ ManAge + 2 = 2 * (SonAge + 2)

theorem find_son's_age (S M : ℕ) (h : age_relationship S M) : S = 18 :=
by
  unfold age_relationship at h
  obtain ⟨h1, h2⟩ := h
  sorry

end find_son_l696_69672


namespace investment_in_scheme_B_l696_69693

theorem investment_in_scheme_B 
    (yieldA : ℝ) (yieldB : ℝ) (investmentA : ℝ) (difference : ℝ) (totalA : ℝ) (totalB : ℝ):
    yieldA = 0.30 → yieldB = 0.50 → investmentA = 300 → difference = 90 
    → totalA = investmentA + (yieldA * investmentA) 
    → totalB = (1 + yieldB) * totalB 
    → totalA = totalB + difference 
    → totalB = 200 :=
by sorry

end investment_in_scheme_B_l696_69693


namespace solve_equation_l696_69690

theorem solve_equation (x : ℝ) (h : x ≠ -2) : (x = -1/2) ↔ (x / (x + 2) + 1 = 1 / (x + 2)) :=
by
  sorry

end solve_equation_l696_69690


namespace sum_of_a_and_c_l696_69642

variable {R : Type} [LinearOrderedField R]

theorem sum_of_a_and_c
    (ha hb hc hd : R) 
    (h_intersect : (1, 7) ∈ {p | p.2 = -2 * abs (p.1 - ha) + hb} ∧ (1, 7) ∈ {p | p.2 = 2 * abs (p.1 - hc) + hd}
                 ∧ (9, 1) ∈ {p | p.2 = -2 * abs (p.1 - ha) + hb} ∧ (9, 1) ∈ {p | p.2 = 2 * abs (p.1 - hc) + hd}) :
  ha + hc = 10 :=
by
  sorry

end sum_of_a_and_c_l696_69642


namespace archie_touchdown_passes_l696_69630

-- Definitions based on the conditions
def richard_avg_first_14_games : ℕ := 6
def richard_avg_last_2_games : ℕ := 3
def richard_games_first : ℕ := 14
def richard_games_last : ℕ := 2

-- Total touchdowns Richard made in the first 14 games
def touchdowns_first_14 := richard_games_first * richard_avg_first_14_games

-- Total touchdowns Richard needs in the final 2 games
def touchdowns_last_2 := richard_games_last * richard_avg_last_2_games

-- Total touchdowns Richard made in the season
def richard_touchdowns_season := touchdowns_first_14 + touchdowns_last_2

-- Archie's record is one less than Richard's total touchdowns for the season
def archie_record := richard_touchdowns_season - 1

-- Proposition to prove Archie's touchdown passes in a season
theorem archie_touchdown_passes : archie_record = 89 := by
  sorry

end archie_touchdown_passes_l696_69630


namespace min_value_fraction_l696_69640

open Real

theorem min_value_fraction (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 3) :
  (∃ x, x = (a + b) / (a * b * c) ∧ x = 16 / 9) :=
by
  sorry

end min_value_fraction_l696_69640


namespace max_abs_z_2_2i_l696_69604

open Complex

theorem max_abs_z_2_2i (z : ℂ) (h : abs (z + 2 - 2 * I) = 1) : 
  ∃ w : ℂ, abs (w - 2 - 2 * I) = 5 :=
sorry

end max_abs_z_2_2i_l696_69604


namespace correct_value_l696_69694

theorem correct_value (x : ℝ) (h : x + 2.95 = 9.28) : x - 2.95 = 3.38 :=
by
  sorry

end correct_value_l696_69694


namespace obtuse_triangles_in_17_gon_l696_69649

noncomputable def number_of_obtuse_triangles (n : ℕ): ℕ := 
  if h : n ≥ 3 then (n * (n - 1) * (n - 2)) / 6 else 0

theorem obtuse_triangles_in_17_gon : number_of_obtuse_triangles 17 = 476 := sorry

end obtuse_triangles_in_17_gon_l696_69649


namespace cubes_sum_is_214_5_l696_69635

noncomputable def r_plus_s_plus_t : ℝ := 12
noncomputable def rs_plus_rt_plus_st : ℝ := 47
noncomputable def rst : ℝ := 59.5

theorem cubes_sum_is_214_5 :
    (r_plus_s_plus_t * ((r_plus_s_plus_t)^2 - 3 * rs_plus_rt_plus_st) + 3 * rst) = 214.5 := by
    sorry

end cubes_sum_is_214_5_l696_69635


namespace new_sales_volume_monthly_profit_maximize_profit_l696_69616

-- Define assumptions and variables
variables (x : ℝ) (p : ℝ) (v : ℝ) (profit : ℝ)

-- Part 1: New sales volume after price increase
theorem new_sales_volume (h : 0 < x ∧ x < 20) : v = 600 - 10 * x :=
sorry

-- Part 2: Price and quantity for a monthly profit of 10,000 yuan
theorem monthly_profit (h : profit = (40 + x - 30) * (600 - 10 * x)) (h2: profit = 10000) : p = 50 ∧ v = 500 :=
sorry

-- Part 3: Price for maximizing monthly sales profit
theorem maximize_profit (h : profit = (40 + x - 30) * (600 - 10 * x)) : (∃ x_max: ℝ, x_max < 20 ∧ ∀ x, x < 20 → profit ≤ -10 * (x - 25)^2 + 12250 ∧ p = 59 ∧ profit = 11890) :=
sorry

end new_sales_volume_monthly_profit_maximize_profit_l696_69616


namespace initial_bottle_caps_l696_69670

theorem initial_bottle_caps (X : ℕ) (h1 : X - 60 + 58 = 67) : X = 69 := by
  sorry

end initial_bottle_caps_l696_69670


namespace real_solution_l696_69686

theorem real_solution (x : ℝ) (h : x ≠ 3) :
  (x * (x + 2)) / ((x - 3)^2) ≥ 8 ↔ (2 ≤ x ∧ x < 3) ∨ (3 < x ∧ x ≤ 48) :=
by
  sorry

end real_solution_l696_69686


namespace smallest_positive_period_l696_69621

noncomputable def tan_period (a b x : ℝ) : ℝ := 
  Real.tan ((a + b) * x / 2)

theorem smallest_positive_period 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  ∃ p > 0, ∀ x, tan_period a b (x + p) = tan_period a b x ∧ p = 2 * Real.pi :=
by
  sorry

end smallest_positive_period_l696_69621


namespace true_proposition_l696_69650

-- Define the propositions p and q
def p : Prop := ∃ x0 : ℝ, x0 ^ 2 - x0 + 1 ≥ 0

def q : Prop := ∀ (a b : ℝ), a < b → 1 / a > 1 / b

-- Prove that p ∧ ¬q is true
theorem true_proposition : p ∧ ¬q :=
by
  sorry

end true_proposition_l696_69650


namespace interior_angle_ratio_l696_69613

theorem interior_angle_ratio (exterior_angle1 exterior_angle2 exterior_angle3 : ℝ)
  (h_ratio : 3 * exterior_angle1 = 4 * exterior_angle2 ∧ 
             4 * exterior_angle1 = 5 * exterior_angle3 ∧ 
             3 * exterior_angle1 + 4 * exterior_angle2 + 5 * exterior_angle3 = 360 ) : 
  3 * (180 - exterior_angle1) = 2 * (180 - exterior_angle2) ∧ 
  2 * (180 - exterior_angle2) = 1 * (180 - exterior_angle3) :=
sorry

end interior_angle_ratio_l696_69613


namespace volume_of_inscribed_sphere_l696_69697

theorem volume_of_inscribed_sphere (a : ℝ) (π : ℝ) (h : a = 6) : 
  (4 / 3 * π * (a / 2) ^ 3) = 36 * π :=
by
  sorry

end volume_of_inscribed_sphere_l696_69697


namespace min_value_ineq_l696_69699

theorem min_value_ineq (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_point_on_chord : ∃ x y : ℝ, x = 4 * a ∧ y = 2 * b ∧ (x + y = 2) ∧ (x^2 + y^2 = 4) ∧ ((x - 2)^2 + (y - 2)^2 = 4)) :
  1 / a + 2 / b ≥ 8 :=
by
  sorry

end min_value_ineq_l696_69699


namespace chocolate_bars_produced_per_minute_l696_69625

theorem chocolate_bars_produced_per_minute
  (sugar_per_bar : ℝ)
  (total_sugar : ℝ)
  (time_in_minutes : ℝ) 
  (bars_per_min : ℝ) :
  sugar_per_bar = 1.5 →
  total_sugar = 108 →
  time_in_minutes = 2 →
  bars_per_min = 36 :=
sorry

end chocolate_bars_produced_per_minute_l696_69625


namespace math_proof_problem_l696_69663

-- Definitions for conditions:
def condition1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 3 / 2) = -f x
def condition2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x - 3 / 4) = -f (- (x - 3 / 4))

-- Statements to prove:
def statement1 (f : ℝ → ℝ) : Prop := ∃ p, p ≠ 0 ∧ ∀ x, f (x + p) = f x
def statement2 (f : ℝ → ℝ) : Prop := ∀ x, f (-(3 / 4) - x) = f (-(3 / 4) + x)
def statement3 (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def statement4 (f : ℝ → ℝ) : Prop := ¬(∀ x y : ℝ, x < y → f x ≤ f y)

theorem math_proof_problem (f : ℝ → ℝ) (h1 : condition1 f) (h2 : condition2 f) :
  statement1 f ∧ statement2 f ∧ statement3 f ∧ statement4 f :=
by
  sorry

end math_proof_problem_l696_69663


namespace necessary_but_not_sufficient_condition_geometric_sequence_l696_69657

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * a (n - 1) / a n 

theorem necessary_but_not_sufficient_condition_geometric_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2) :
  (is_geometric_sequence a → (∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2)) ∧ (∃ b : ℕ → ℝ, (b n = 0 ∨ b n = b (n - 1) ∨ b n = b (n + 1)) ∧ ¬ is_geometric_sequence b) := 
sorry

end necessary_but_not_sufficient_condition_geometric_sequence_l696_69657


namespace total_percent_decrease_baseball_card_l696_69659

theorem total_percent_decrease_baseball_card
  (original_value : ℝ)
  (first_year_decrease : ℝ := 0.20)
  (second_year_decrease : ℝ := 0.30)
  (value_after_first_year : ℝ := original_value * (1 - first_year_decrease))
  (final_value : ℝ := value_after_first_year * (1 - second_year_decrease))
  (total_percent_decrease : ℝ := ((original_value - final_value) / original_value) * 100) :
  total_percent_decrease = 44 :=
by 
  sorry

end total_percent_decrease_baseball_card_l696_69659


namespace average_weight_l696_69619

theorem average_weight {w : ℝ} 
  (h1 : 62 < w) 
  (h2 : w < 72) 
  (h3 : 60 < w) 
  (h4 : w < 70) 
  (h5 : w ≤ 65) : w = 63.5 :=
by
  sorry

end average_weight_l696_69619


namespace determine_continuous_function_l696_69606

open Real

theorem determine_continuous_function (f : ℝ → ℝ) 
  (h_continuous : Continuous f)
  (h_initial : f 0 = 1)
  (h_inequality : ∀ x y : ℝ, f (x + y) ≥ f x * f y) : 
  ∃ k : ℝ, ∀ x : ℝ, f x = exp (k * x) :=
sorry

end determine_continuous_function_l696_69606


namespace first_candidate_percentage_l696_69643

theorem first_candidate_percentage (total_votes : ℕ) (invalid_percentage : ℕ) (second_candidate_votes : ℕ) 
  (h_total_votes : total_votes = 7500) 
  (h_invalid_percentage : invalid_percentage = 20) 
  (h_second_candidate_votes : second_candidate_votes = 2700) : 
  (100 * (total_votes * (1 - (invalid_percentage / 100)) - second_candidate_votes) / (total_votes * (1 - (invalid_percentage / 100)))) = 55 :=
by
  sorry

end first_candidate_percentage_l696_69643


namespace total_weight_on_scale_l696_69610

-- Define the weights of Alexa and Katerina
def alexa_weight : ℕ := 46
def katerina_weight : ℕ := 49

-- State the theorem to prove the total weight on the scale
theorem total_weight_on_scale : alexa_weight + katerina_weight = 95 := by
  sorry

end total_weight_on_scale_l696_69610


namespace minimum_value_y_l696_69683

noncomputable def y (x : ℝ) : ℝ := x + 1 / (x - 1)

theorem minimum_value_y (x : ℝ) (hx : x > 1) : ∃ A, (A = 3) ∧ (∀ y', y' = y x → y' ≥ A) := sorry

end minimum_value_y_l696_69683


namespace michael_twenty_dollar_bills_l696_69602

theorem michael_twenty_dollar_bills (total_amount : ℕ) (denomination : ℕ) 
  (h_total : total_amount = 280) (h_denom : denomination = 20) : 
  total_amount / denomination = 14 := by
  sorry

end michael_twenty_dollar_bills_l696_69602


namespace find_number_l696_69691

theorem find_number (x : ℕ) (h : x - 263 + 419 = 725) : x = 569 :=
sorry

end find_number_l696_69691


namespace trajectory_midpoints_parabola_l696_69622

theorem trajectory_midpoints_parabola {k : ℝ} (hk : k ≠ 0) :
  ∀ (x1 x2 y1 y2 : ℝ), 
    y1 = 2 * x1^2 → 
    y2 = 2 * x2^2 → 
    y2 - y1 = 2 * (x2 + x1) * (x2 - x1) → 
    x = (x1 + x2) / 2 → 
    k = (y2 - y1) / (x2 - x1) → 
    x = 1 / (4 * k) := 
sorry

end trajectory_midpoints_parabola_l696_69622


namespace find_b_value_l696_69644

-- Definitions based on the problem conditions
def line_bisects_circle (b : ℝ) : Prop :=
  ∃ c : ℝ × ℝ, (c.fst = 4 ∧ c.snd = -1) ∧
                (c.snd = c.fst + b)

-- Theorem statement for the problem
theorem find_b_value : line_bisects_circle (-5) :=
by
  sorry

end find_b_value_l696_69644


namespace chairs_made_after_tables_l696_69658

def pieces_of_wood : Nat := 672
def wood_per_table : Nat := 12
def wood_per_chair : Nat := 8
def number_of_tables : Nat := 24

theorem chairs_made_after_tables (pieces_of_wood wood_per_table wood_per_chair number_of_tables : Nat) :
  wood_per_table * number_of_tables <= pieces_of_wood ->
  (pieces_of_wood - wood_per_table * number_of_tables) / wood_per_chair = 48 :=
by
  sorry

end chairs_made_after_tables_l696_69658


namespace sum_of_solutions_eq_35_over_3_l696_69653

theorem sum_of_solutions_eq_35_over_3 (a b : ℝ) 
  (h1 : 2 * a + b = 14) (h2 : a + 2 * b = 21) : 
  a + b = 35 / 3 := 
by
  sorry

end sum_of_solutions_eq_35_over_3_l696_69653


namespace image_digit_sum_l696_69664

theorem image_digit_sum 
  (cat chicken crab bear goat: ℕ)
  (h1 : 5 * crab = 10)
  (h2 : 4 * crab + goat = 11)
  (h3 : 2 * goat + crab + 2 * bear = 16)
  (h4 : cat + bear + 2 * goat + crab = 13)
  (h5 : 2 * crab + 2 * chicken + goat = 17) :
  cat = 1 ∧ chicken = 5 ∧ crab = 2 ∧ bear = 4 ∧ goat = 3 := by
  sorry

end image_digit_sum_l696_69664


namespace ratio_a_b_equals_sqrt2_l696_69654

variable (A B C a b c : ℝ) -- Define the variables representing the angles and sides.

-- Assuming the sides a, b, c are positive and a triangle is formed (non-degenerate)
axiom triangle_ABC : 0 < a ∧ 0 < b ∧ 0 < c

-- Assuming the sum of the angles in a triangle equals 180 degrees (π radians)
axiom sum_angles_triangle : A + B + C = Real.pi

-- Given condition
axiom given_condition : b * Real.cos C + c * Real.cos B = Real.sqrt 2 * b

-- Problem statement to be proven
theorem ratio_a_b_equals_sqrt2 : (a / b) = Real.sqrt 2 :=
by
  -- Assume the problem statement is correct
  sorry

end ratio_a_b_equals_sqrt2_l696_69654


namespace angle_C_magnitude_area_triangle_l696_69696

variable {a b c A B C : ℝ}

namespace triangle

-- Conditions and variable declarations
axiom condition1 : 2 * b * Real.cos C = a * Real.cos C + c * Real.cos A
axiom triangle_sides : a = 3 ∧ b = 2 ∧ c = Real.sqrt 7

-- Prove the magnitude of angle C is π/3
theorem angle_C_magnitude : C = Real.pi / 3 :=
by sorry

-- Prove that given b = 2 and c = sqrt(7), a = 3 and the area of triangle ABC is 3*sqrt(3)/2
theorem area_triangle :
  (b = 2 ∧ c = Real.sqrt 7 ∧ C = Real.pi / 3) → 
  (a = 3 ∧ (1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2)) :=
by sorry

end triangle

end angle_C_magnitude_area_triangle_l696_69696


namespace molecular_weight_of_one_mole_l696_69609

noncomputable def molecular_weight (total_weight : ℝ) (moles : ℕ) : ℝ :=
total_weight / moles

theorem molecular_weight_of_one_mole (h : molecular_weight 252 6 = 42) : molecular_weight 252 6 = 42 := by
  exact h

end molecular_weight_of_one_mole_l696_69609


namespace magic_square_expression_l696_69668

theorem magic_square_expression : 
  let a := 8
  let b := 6
  let c := 14
  let d := 10
  let e := 11
  let f := 5
  let g := 3
  a - b - c + d + e + f - g = 11 :=
by
  sorry

end magic_square_expression_l696_69668


namespace problem_solution_l696_69669

theorem problem_solution (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : a + b = 1) :
  (a + 1 / b) ^ 2 + (b + 1 / a) ^ 2 ≥ 25 / 2 :=
sorry

end problem_solution_l696_69669


namespace cost_per_bracelet_l696_69637

/-- Each friend and the number of their name's letters -/
def friends_letters_counts : List (String × Nat) :=
  [("Jessica", 7), ("Tori", 4), ("Lily", 4), ("Patrice", 7)]

/-- Total cost spent by Robin -/
def total_cost : Nat := 44

/-- Calculate the total number of bracelets -/
def total_bracelets : Nat :=
  friends_letters_counts.foldr (λ p acc => p.snd + acc) 0

theorem cost_per_bracelet : (total_cost / total_bracelets) = 2 :=
  by
    sorry

end cost_per_bracelet_l696_69637


namespace length_of_platform_l696_69624

theorem length_of_platform (v t_m t_p L_t L_p : ℝ)
    (h1 : v = 33.3333333)
    (h2 : t_m = 22)
    (h3 : t_p = 45)
    (h4 : L_t = v * t_m)
    (h5 : L_t + L_p = v * t_p) :
    L_p = 766.666666 :=
by
  sorry

end length_of_platform_l696_69624


namespace find_number_l696_69675

theorem find_number (X a b : ℕ) (hX : X = 10 * a + b) 
  (h1 : a * b = 24) (h2 : 10 * b + a = X + 18) : X = 46 :=
by
  sorry

end find_number_l696_69675


namespace minimum_a_l696_69665

noncomputable def f (x a : ℝ) := Real.exp x * (x^3 - 3 * x + 3) - a * Real.exp x - x

theorem minimum_a (a : ℝ) : (∃ x, x ≥ -2 ∧ f x a ≤ 0) ↔ a ≥ 1 - 1 / Real.exp 1 :=
by
  sorry

end minimum_a_l696_69665


namespace rectangle_width_decrease_proof_l696_69603

def rectangle_width_decreased_percentage (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : ℝ := 
  28.57

theorem rectangle_width_decrease_proof (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : 
  rectangle_width_decreased_percentage L W h = 28.57 := 
by
  sorry

end rectangle_width_decrease_proof_l696_69603


namespace sum_of_coefficients_l696_69662

-- Define the polynomial
def polynomial (x : ℝ) : ℝ :=
  2 * (4 * x ^ 8 + 7 * x ^ 6 - 9 * x ^ 3 + 3) + 6 * (x ^ 7 - 2 * x ^ 4 + 8 * x ^ 2 - 2)

-- State the theorem to prove the sum of the coefficients
theorem sum_of_coefficients : polynomial 1 = 40 :=
by
  sorry

end sum_of_coefficients_l696_69662


namespace ratio_of_red_to_total_simplified_l696_69677

def number_of_red_haired_children := 9
def total_number_of_children := 48

theorem ratio_of_red_to_total_simplified:
  (number_of_red_haired_children: ℚ) / (total_number_of_children: ℚ) = (3 : ℚ) / (16 : ℚ) := 
by
  sorry

end ratio_of_red_to_total_simplified_l696_69677


namespace smallest_x_plus_y_l696_69629

theorem smallest_x_plus_y 
  (x y : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy_neq : x ≠ y)
  (h_eq : (1/x + 1/y = 1/10)) : x + y = 45 :=
sorry

end smallest_x_plus_y_l696_69629


namespace initial_people_count_l696_69687

theorem initial_people_count (left remaining total : ℕ) (h1 : left = 6) (h2 : remaining = 5) : total = 11 :=
  by
  sorry

end initial_people_count_l696_69687


namespace slips_drawn_l696_69620

theorem slips_drawn (P : ℚ) (P_value : P = 24⁻¹) :
  ∃ n : ℕ, (n ≤ 5 ∧ P = (Nat.choose 5 n) / (Nat.choose 10 n) ∧ n = 4) := by
{
  sorry
}

end slips_drawn_l696_69620


namespace max_sum_n_value_l696_69676

open Nat

-- Definitions for the problem
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a 0 + (n - 1) * (a 1 - a 0))) / 2

-- Statement of the theorem
theorem max_sum_n_value (a : ℕ → ℤ) (d : ℤ) (h_arith_seq : arithmetic_sequence a d) 
  (h_initial : a 0 > 0) (h_condition : 8 * a 4 = 13 * a 10) : 
  ∃ n, sum_of_first_n_terms a n = max (sum_of_first_n_terms a n) ∧ n = 20 :=
sorry

end max_sum_n_value_l696_69676


namespace avg_age_initial_group_l696_69618

theorem avg_age_initial_group (N : ℕ) (A avg_new_persons avg_entire_group : ℝ) (hN : N = 15)
  (h_avg_new_persons : avg_new_persons = 15) (h_avg_entire_group : avg_entire_group = 15.5) :
  (A * (N : ℝ) + 15 * avg_new_persons) = ((N + 15) : ℝ) * avg_entire_group → A = 16 :=
by
  intro h
  have h_initial : N = 15 := hN
  have h_new : avg_new_persons = 15 := h_avg_new_persons
  have h_group : avg_entire_group = 15.5 := h_avg_entire_group
  sorry

end avg_age_initial_group_l696_69618


namespace prime_factors_of_difference_l696_69698

theorem prime_factors_of_difference (A B : ℕ) (h_neq : A ≠ B) : 
  ∃ p, Nat.Prime p ∧ p ∣ (Nat.gcd (9 * A - 9 * B + 10) (9 * B - 9 * A - 10)) :=
by
  sorry

end prime_factors_of_difference_l696_69698


namespace sum_of_cubes_l696_69623

theorem sum_of_cubes (x y z : ℝ) (h1 : x + y + z = 7) (h2 : xy + xz + yz = 9) (h3 : xyz = -18) :
  x^3 + y^3 + z^3 = 100 :=
by
  sorry

end sum_of_cubes_l696_69623


namespace inf_arith_seq_contains_inf_geo_seq_l696_69626

-- Condition: Infinite arithmetic sequence of natural numbers
variable (a d : ℕ) (h : ∀ n : ℕ, n ≥ 1 → ∃ k : ℕ, k = a + (n - 1) * d)

-- Theorem: There exists an infinite geometric sequence within the arithmetic sequence
theorem inf_arith_seq_contains_inf_geo_seq :
  ∃ r : ℕ, ∀ n : ℕ, ∃ k : ℕ, k = a * r ^ (n - 1) := sorry

end inf_arith_seq_contains_inf_geo_seq_l696_69626


namespace sequence_value_l696_69685

theorem sequence_value (a : ℕ → ℕ) (h₁ : ∀ n, a (2 * n) = a (2 * n - 1) + (-1 : ℤ)^n) 
                        (h₂ : ∀ n, a (2 * n + 1) = a (2 * n) + n)
                        (h₃ : a 1 = 1) : a 20 = 46 :=
by 
  sorry

end sequence_value_l696_69685


namespace Apollonian_Circle_Range_l696_69679

def range_of_m := Set.Icc (Real.sqrt 5 / 2) (Real.sqrt 21 / 2)

theorem Apollonian_Circle_Range :
  ∃ P : ℝ × ℝ, ∃ m > 0, ((P.1 - 2) ^ 2 + (P.2 - m) ^ 2 = 1 / 4) ∧ 
            (Real.sqrt ((P.1 + 1) ^ 2 + P.2 ^ 2) = 2 * Real.sqrt ((P.1 - 2) ^ 2 + P.2 ^ 2)) →
            m ∈ range_of_m :=
  sorry

end Apollonian_Circle_Range_l696_69679


namespace circumscribedCircleDiameter_is_10sqrt2_l696_69666

noncomputable def circumscribedCircleDiameter (a : ℝ) (A : ℝ) : ℝ :=
  a / Real.sin A

theorem circumscribedCircleDiameter_is_10sqrt2 :
  circumscribedCircleDiameter 10 (Real.pi / 4) = 10 * Real.sqrt 2 :=
by
  sorry

end circumscribedCircleDiameter_is_10sqrt2_l696_69666


namespace percentage_mike_has_l696_69680
-- Definitions and conditions
variables (phone_cost : ℝ) (additional_needed : ℝ)
def amount_mike_has := phone_cost - additional_needed

-- Main statement
theorem percentage_mike_has (phone_cost : ℝ) (additional_needed : ℝ) (h1 : phone_cost = 1300) (h2 : additional_needed = 780) : 
  (amount_mike_has phone_cost additional_needed) * 100 / phone_cost = 40 :=
by
  sorry

end percentage_mike_has_l696_69680


namespace increasing_order_magnitudes_l696_69632

variable (x : ℝ)

noncomputable def y := x^x
noncomputable def z := x^(x^x)

theorem increasing_order_magnitudes (h1 : 1 < x) (h2 : x < 1.1) : x < y x ∧ y x < z x :=
by
  have h3 : y x = x^x := rfl
  have h4 : z x = x^(x^x) := rfl
  sorry

end increasing_order_magnitudes_l696_69632


namespace determinant_expression_l696_69627

noncomputable def matrixDet (α β : ℝ) : ℝ :=
  Matrix.det ![
    ![Real.sin α * Real.cos β, -Real.sin α * Real.sin β, Real.cos α],
    ![-Real.sin β, -Real.cos β, 0],
    ![Real.cos α * Real.cos β, Real.cos α * Real.sin β, Real.sin α]]

theorem determinant_expression (α β: ℝ) : matrixDet α β = Real.sin α ^ 3 := 
by 
  sorry

end determinant_expression_l696_69627


namespace monomial_properties_l696_69648

noncomputable def monomial_coeff : ℚ := -(3/5 : ℚ)

def monomial_degree (x y : ℤ) : ℕ :=
  1 + 2

theorem monomial_properties (x y : ℤ) :
  monomial_coeff = -(3/5) ∧ monomial_degree x y = 3 :=
by
  -- Proof is to be filled here
  sorry

end monomial_properties_l696_69648


namespace sum_of_three_numbers_l696_69636

theorem sum_of_three_numbers (a b c : ℤ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : a + 15 = (a + b + c) / 3) (h4 : (a + b + c) / 3 = c - 20) (h5 : b = 7) :
  a + b + c = 36 :=
sorry

end sum_of_three_numbers_l696_69636


namespace taylor_probability_l696_69661

open Nat Real

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1 - p)^(n - k)

theorem taylor_probability :
  (binomial_probability 5 2 (3/5) = 144 / 625) :=
by
  sorry

end taylor_probability_l696_69661


namespace max_m_for_factored_polynomial_l696_69689

theorem max_m_for_factored_polynomial :
  ∃ m, (∀ A B : ℤ, (5 * x ^ 2 + m * x + 45 = (5 * x + A) * (x + B) → AB = 45) → 
    m = 226) :=
sorry

end max_m_for_factored_polynomial_l696_69689


namespace factor_expression_l696_69633

theorem factor_expression (b : ℝ) : 45 * b^2 + 135 * b^3 = 45 * b^2 * (1 + 3 * b) :=
by
  sorry

end factor_expression_l696_69633


namespace paul_initial_savings_l696_69656

theorem paul_initial_savings (additional_allowance: ℕ) (cost_per_toy: ℕ) (number_of_toys: ℕ) (total_savings: ℕ) :
  additional_allowance = 7 →
  cost_per_toy = 5 →
  number_of_toys = 2 →
  total_savings + additional_allowance = cost_per_toy * number_of_toys →
  total_savings = 3 :=
by
  intros h_additional h_cost h_number h_total
  sorry

end paul_initial_savings_l696_69656


namespace value_of_a_l696_69646

noncomputable def A : Set ℝ := { x | abs x = 1 }
def B (a : ℝ) : Set ℝ := { x | a * x = 1 }
def is_superset (A B : Set ℝ) : Prop := ∀ x, x ∈ B → x ∈ A

theorem value_of_a (a : ℝ) (h : is_superset A (B a)) : a = 1 ∨ a = 0 ∨ a = -1 :=
  sorry

end value_of_a_l696_69646


namespace number_of_ways_to_choose_students_l696_69681

theorem number_of_ways_to_choose_students :
  let female_students := 4
  let male_students := 3
  (female_students * male_students) = 12 :=
by
  sorry

end number_of_ways_to_choose_students_l696_69681


namespace sum_angles_bisected_l696_69652

theorem sum_angles_bisected (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h₁ : 0 < θ₁) (h₂ : 0 < θ₂) (h₃ : 0 < θ₃) (h₄ : 0 < θ₄)
  (h_sum : θ₁ + θ₂ + θ₃ + θ₄ = 360) :
  (θ₁ / 2 + θ₃ / 2 = 180 ∨ θ₂ / 2 + θ₄ / 2 = 180) ∧ (θ₂ / 2 + θ₄ / 2 = 180 ∨ θ₁ / 2 + θ₃ / 2 = 180) := 
by 
  sorry

end sum_angles_bisected_l696_69652


namespace cubic_has_three_natural_roots_l696_69628

theorem cubic_has_three_natural_roots (p : ℝ) :
  (∃ (x1 x2 x3 : ℕ), 5 * (x1:ℝ)^3 - 5 * (p + 1) * (x1:ℝ)^2 + (71 * p - 1) * (x1:ℝ) + 1 = 66 * p ∧
                     5 * (x2:ℝ)^3 - 5 * (p + 1) * (x2:ℝ)^2 + (71 * p - 1) * (x2:ℝ) + 1 = 66 * p ∧
                     5 * (x3:ℝ)^3 - 5 * (p + 1) * (x3:ℝ)^2 + (71 * p - 1) * (x3:ℝ) + 1 = 66 * p) ↔ p = 76 :=
by sorry

end cubic_has_three_natural_roots_l696_69628


namespace inequality_proof_l696_69667

variable {x : ℝ}
variable {n : ℕ}
variable {a : ℝ}

theorem inequality_proof (h1 : x > 0) (h2 : n > 0) (h3 : x + a / x^n ≥ n + 1) : a = n^n := 
sorry

end inequality_proof_l696_69667


namespace inverse_proportion_l696_69673

variable {x y x1 x2 y1 y2 : ℝ}
variable {k : ℝ}

theorem inverse_proportion {h1 : x1 ≠ 0} {h2 : x2 ≠ 0} {h3 : y1 ≠ 0} {h4 : y2 ≠ 0}
  (h5 : (∃ k, ∀ (x y : ℝ), x * y = k))
  (h6 : x1 / x2 = 4 / 5) : 
  y1 / y2 = 5 / 4 :=
sorry

end inverse_proportion_l696_69673
