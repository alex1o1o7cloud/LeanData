import Mathlib

namespace min_value_reciprocal_sum_l1973_197303

theorem min_value_reciprocal_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 3) :
  3 ≤ (1 / a) + (1 / b) + (1 / c) :=
by sorry

end min_value_reciprocal_sum_l1973_197303


namespace total_playtime_l1973_197312

noncomputable def lena_playtime_minutes : ℕ := 210
noncomputable def brother_playtime_minutes (lena_playtime: ℕ) : ℕ := lena_playtime + 17
noncomputable def sister_playtime_minutes (brother_playtime: ℕ) : ℕ := 2 * brother_playtime

theorem total_playtime
  (lena_playtime : ℕ)
  (brother_playtime : ℕ)
  (sister_playtime : ℕ)
  (h_lena : lena_playtime = lena_playtime_minutes)
  (h_brother : brother_playtime = brother_playtime_minutes lena_playtime)
  (h_sister : sister_playtime = sister_playtime_minutes brother_playtime) :
  lena_playtime + brother_playtime + sister_playtime = 891 := 
  by sorry

end total_playtime_l1973_197312


namespace simplify_fraction_l1973_197380

theorem simplify_fraction
  (a b c : ℝ)
  (h : 2 * a - 3 * c - 4 - b ≠ 0)
  : (6 * a ^ 2 - 2 * b ^ 2 + 6 * c ^ 2 + a * b - 13 * a * c - 4 * b * c - 18 * a - 5 * b + 17 * c + 12) /
    (4 * a ^ 2 - b ^ 2 + 9 * c ^ 2 - 12 * a * c - 16 * a + 24 * c + 16) =
    (3 * a - 2 * c - 3 + 2 * b) / (2 * a - 3 * c - 4 + b) :=
  sorry

end simplify_fraction_l1973_197380


namespace distinct_cyclic_quadrilaterals_perimeter_36_l1973_197331

noncomputable def count_distinct_cyclic_quadrilaterals : Nat :=
  1026

theorem distinct_cyclic_quadrilaterals_perimeter_36 :
  (∃ (a b c d : ℕ), a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ a + b + c + d = 36 ∧ a < b + c + d) → count_distinct_cyclic_quadrilaterals = 1026 :=
by
  rintro ⟨a, b, c, d, hab, hbc, hcd, hsum, hlut⟩
  sorry

end distinct_cyclic_quadrilaterals_perimeter_36_l1973_197331


namespace maximum_x_plus_2y_l1973_197315

theorem maximum_x_plus_2y 
  (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x^2 + 8 * y^2 + x * y = 2) :
  x + 2 * y ≤ 4 / 3 :=
sorry

end maximum_x_plus_2y_l1973_197315


namespace original_card_count_l1973_197344

theorem original_card_count
  (r b : ℕ)
  (initial_prob_red : (r : ℚ) / (r + b) = 2 / 5)
  (prob_red_after_adding_black : (r : ℚ) / (r + (b + 6)) = 1 / 3) :
  r + b = 30 := sorry

end original_card_count_l1973_197344


namespace smallest_prime_factor_in_C_l1973_197316

def smallest_prime_factor_def (n : Nat) : Nat :=
  if n % 2 = 0 then 2 else
  if n % 3 = 0 then 3 else
  sorry /- Define a function to find the smallest prime factor of a number n -/

def is_prime (p : Nat) : Prop :=
  2 ≤ p ∧ ∀ d : Nat, 2 ≤ d → d ∣ p → d = p

def in_set (x : Nat) : Prop :=
  x = 64 ∨ x = 66 ∨ x = 67 ∨ x = 68 ∨ x = 71

theorem smallest_prime_factor_in_C : ∀ x, in_set x → 
  (smallest_prime_factor_def x = 2 ∨ smallest_prime_factor_def x = 67 ∨ smallest_prime_factor_def x = 71) :=
by
  intro x hx
  cases hx with
  | inl hx  => sorry
  | inr hx  => sorry

end smallest_prime_factor_in_C_l1973_197316


namespace abs_eq_condition_l1973_197308

theorem abs_eq_condition (x : ℝ) (h : |x - 1| + |x - 5| = 4) : 1 ≤ x ∧ x ≤ 5 :=
by 
  sorry

end abs_eq_condition_l1973_197308


namespace simplify_fraction_l1973_197311

theorem simplify_fraction :
  ( (3 * 5 * 7 : ℚ) / (9 * 11 * 13) ) * ( (7 * 9 * 11 * 15) / (3 * 5 * 14) ) = 15 / 26 :=
by
  sorry

end simplify_fraction_l1973_197311


namespace distinct_nonzero_digits_sum_l1973_197347

theorem distinct_nonzero_digits_sum (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0) 
  (h7 : 100*a + 10*b + c + 100*a + 10*c + b + 100*b + 10*a + c + 100*b + 10*c + a + 100*c + 10*a + b + 100*c + 10*b + a = 1776) : 
  (a = 1 ∧ b = 2 ∧ c = 5) ∨ (a = 1 ∧ b = 3 ∧ c = 4) ∨ (a = 1 ∧ b = 4 ∧ c = 3) ∨ (a = 1 ∧ b = 5 ∧ c = 2) ∨ (a = 2 ∧ b = 1 ∧ c = 5) ∨
  (a = 2 ∧ b = 5 ∧ c = 1) ∨ (a = 3 ∧ b = 1 ∧ c = 4) ∨ (a = 3 ∧ b = 4 ∧ c = 1) ∨ (a = 4 ∧ b = 1 ∧ c = 3) ∨ (a = 4 ∧ b = 3 ∧ c = 1) ∨
  (a = 5 ∧ b = 1 ∧ c = 2) ∨ (a = 5 ∧ b = 2 ∧ c = 1) :=
sorry

end distinct_nonzero_digits_sum_l1973_197347


namespace square_field_area_l1973_197356

theorem square_field_area (speed time perimeter : ℕ) (h1 : speed = 20) (h2 : time = 4) (h3 : perimeter = speed * time) :
  ∃ s : ℕ, perimeter = 4 * s ∧ s * s = 400 :=
by
  -- All conditions and definitions are stated, proof is skipped using sorry
  sorry

end square_field_area_l1973_197356


namespace cloud_height_l1973_197339

/--
Given:
- α : ℝ (elevation angle from the top of a tower)
- β : ℝ (depression angle seen in the lake)
- m : ℝ (height of the tower)
Prove:
- The height of the cloud hovering above the observer (h - m) is given by
 2 * m * cos β * sin α / sin (β - α)
-/
theorem cloud_height (α β m : ℝ) :
  (∃ h : ℝ, h - m = 2 * m * Real.cos β * Real.sin α / Real.sin (β - α)) :=
by
  sorry

end cloud_height_l1973_197339


namespace total_strawberry_weight_l1973_197349

def MarcosStrawberries : ℕ := 3
def DadsStrawberries : ℕ := 17

theorem total_strawberry_weight : MarcosStrawberries + DadsStrawberries = 20 := by
  sorry

end total_strawberry_weight_l1973_197349


namespace census_survey_is_suitable_l1973_197309

def suitable_for_census (s: String) : Prop :=
  s = "Understand the vision condition of students in a class"

theorem census_survey_is_suitable :
  suitable_for_census "Understand the vision condition of students in a class" :=
by
  sorry

end census_survey_is_suitable_l1973_197309


namespace hospital_cost_minimization_l1973_197394

theorem hospital_cost_minimization :
  ∃ (x y : ℕ), (5 * x + 6 * y = 50) ∧ (10 * x + 20 * y = 140) ∧ (2 * x + 3 * y = 23) :=
by
  sorry

end hospital_cost_minimization_l1973_197394


namespace sequence_general_term_l1973_197307

theorem sequence_general_term (a : ℕ → ℤ) : 
  (∀ n, a n = (-1)^(n + 1) * (3 * n - 2)) ↔ 
  (a 1 = 1 ∧ a 2 = -4 ∧ a 3 = 7 ∧ a 4 = -10 ∧ a 5 = 13) :=
by
  sorry

end sequence_general_term_l1973_197307


namespace apples_sold_fresh_l1973_197386

-- Definitions per problem conditions
def total_production : Float := 8.0
def initial_percentage_mixed : Float := 0.30
def percentage_increase_per_million : Float := 0.05
def percentage_for_apple_juice : Float := 0.60
def percentage_sold_fresh : Float := 0.40

-- We need to prove that given the conditions, the amount of apples sold fresh is 2.24 million tons
theorem apples_sold_fresh :
  ( (total_production - (initial_percentage_mixed * total_production)) * percentage_sold_fresh = 2.24 ) :=
by
  sorry

end apples_sold_fresh_l1973_197386


namespace problem1_problem2_problem3_l1973_197325

/-- Problem 1: Calculate 25 * 26 * 8 and show it equals 5200 --/
theorem problem1 : 25 * 26 * 8 = 5200 := 
sorry

/-- Problem 2: Calculate 340 * 40 / 17 and show it equals 800 --/
theorem problem2 : 340 * 40 / 17 = 800 := 
sorry

/-- Problem 3: Calculate 440 * 15 + 480 * 15 + 79 * 15 + 15 and show it equals 15000 --/
theorem problem3 : 440 * 15 + 480 * 15 + 79 * 15 + 15 = 15000 := 
sorry

end problem1_problem2_problem3_l1973_197325


namespace max_min_sum_zero_l1973_197395

def cubic_function (x : ℝ) : ℝ :=
  x^3 - 3 * x

def first_derivative (x : ℝ) : ℝ :=
  3 * x^2 - 3

theorem max_min_sum_zero :
  let m := cubic_function (-1);
  let n := cubic_function 1;
  m + n = 0 :=
by
  sorry

end max_min_sum_zero_l1973_197395


namespace even_parts_impossible_odd_parts_possible_l1973_197390

theorem even_parts_impossible (n m : ℕ) (h₁ : n = 1) (h₂ : ∀ k, m = n + 2 * k) : n + 2 * m ≠ 100 := by
  -- Proof omitted
  sorry

theorem odd_parts_possible (n m : ℕ) (h₁ : n = 1) (h₂ : ∀ k, m = n + 2 * k) : ∃ k, n + 2 * k = 2017 := by
  -- Proof omitted
  sorry

end even_parts_impossible_odd_parts_possible_l1973_197390


namespace cone_volume_calc_l1973_197371

noncomputable def cone_volume (diameter slant_height: ℝ) : ℝ :=
  let r := diameter / 2
  let h := Real.sqrt (slant_height^2 - r^2)
  (1 / 3) * Real.pi * r^2 * h

theorem cone_volume_calc :
  cone_volume 12 10 = 96 * Real.pi :=
by
  sorry

end cone_volume_calc_l1973_197371


namespace forming_n_and_m_l1973_197362

def is_created_by_inserting_digit (n: ℕ) (base: ℕ): Prop :=
  ∃ d1 d2 d3 d: ℕ, n = d1 * 1000 + d * 100 + d2 * 10 + d3 ∧ base = d1 * 100 + d2 * 10 + d3

theorem forming_n_and_m (a b: ℕ) (base: ℕ) (sum: ℕ) 
  (h1: is_created_by_inserting_digit a base)
  (h2: is_created_by_inserting_digit b base) 
  (h3: a + b = sum):
  (a = 2195 ∧ b = 2165) 
  ∨ (a = 2185 ∧ b = 2175) 
  ∨ (a = 2215 ∧ b = 2145) 
  ∨ (a = 2165 ∧ b = 2195) 
  ∨ (a = 2175 ∧ b = 2185) 
  ∨ (a = 2145 ∧ b = 2215) := 
sorry

end forming_n_and_m_l1973_197362


namespace polar_conversion_equiv_l1973_197326

noncomputable def polar_convert (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
if r < 0 then (-r, θ + Real.pi) else (r, θ)

theorem polar_conversion_equiv : polar_convert (-3) (Real.pi / 4) = (3, 5 * Real.pi / 4) :=
by
  sorry

end polar_conversion_equiv_l1973_197326


namespace exist_two_quadrilaterals_l1973_197324

-- Define the structure of a quadrilateral with four sides and two diagonals
structure Quadrilateral :=
  (s1 : ℝ) -- side 1
  (s2 : ℝ) -- side 2
  (s3 : ℝ) -- side 3
  (s4 : ℝ) -- side 4
  (d1 : ℝ) -- diagonal 1
  (d2 : ℝ) -- diagonal 2

-- The theorem stating the existence of two quadrilaterals satisfying the given conditions
theorem exist_two_quadrilaterals :
  ∃ (quad1 quad2 : Quadrilateral),
  quad1.s1 < quad2.s1 ∧ quad1.s2 < quad2.s2 ∧ quad1.s3 < quad2.s3 ∧ quad1.s4 < quad2.s4 ∧
  quad1.d1 > quad2.d1 ∧ quad1.d2 > quad2.d2 :=
by
  sorry

end exist_two_quadrilaterals_l1973_197324


namespace discount_is_15_point_5_percent_l1973_197369

noncomputable def wholesale_cost (W : ℝ) := W
noncomputable def retail_price (W : ℝ) := 1.5384615384615385 * W
noncomputable def selling_price (W : ℝ) := 1.3 * W
noncomputable def discount_percentage (W : ℝ) := 
  let D := retail_price W - selling_price W
  (D / retail_price W) * 100

theorem discount_is_15_point_5_percent (W : ℝ) (hW : W > 0) : 
  discount_percentage W = 15.5 := 
by 
  sorry

end discount_is_15_point_5_percent_l1973_197369


namespace expression_value_eq_3084_l1973_197353

theorem expression_value_eq_3084 (x : ℤ) (hx : x = -3007) :
  (abs (abs (Real.sqrt (abs x - x) - x) - x) - Real.sqrt (abs (x - x^2)) = 3084) :=
by
  sorry

end expression_value_eq_3084_l1973_197353


namespace evaluate_pow_l1973_197306

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end evaluate_pow_l1973_197306


namespace cube_path_count_l1973_197389

noncomputable def numberOfWaysToMoveOnCube : Nat :=
  20

theorem cube_path_count :
  ∀ (cube : Type) (top bottom side1 side2 side3 side4 : cube),
    (∀ (p : cube → cube → Prop), 
      (p top side1 ∨ p top side2 ∨ p top side3 ∨ p top side4) ∧ 
      (p side1 bottom ∨ p side2 bottom ∨ p side3 bottom ∨ p side4 bottom)) →
    numberOfWaysToMoveOnCube = 20 :=
by
  intros
  sorry

end cube_path_count_l1973_197389


namespace maximize_area_l1973_197318

-- Define the variables and constants
variables {x y p : ℝ}

-- Define the conditions
def perimeter (x y p : ℝ) := (2 * x + 2 * y = p)
def area (x y : ℝ) := x * y

-- The theorem statement with conditions
theorem maximize_area (h : perimeter x y p) : x = y → x = p / 4 :=
by
  sorry

end maximize_area_l1973_197318


namespace arrangement_count_example_l1973_197361

theorem arrangement_count_example 
  (teachers : Finset String) 
  (students : Finset String) 
  (locations : Finset String) 
  (h_teachers : teachers.card = 2) 
  (h_students : students.card = 4) 
  (h_locations : locations.card = 2)
  : ∃ n : ℕ, n = 12 := 
sorry

end arrangement_count_example_l1973_197361


namespace animal_shelter_l1973_197379

theorem animal_shelter : ∃ D C : ℕ, (D = 75) ∧ (D / C = 15 / 7) ∧ (D / (C + 20) = 15 / 11) :=
by
  sorry

end animal_shelter_l1973_197379


namespace calc_area_of_quadrilateral_l1973_197393

-- Define the terms and conditions using Lean definitions
noncomputable def triangle_areas : ℕ × ℕ × ℕ := (6, 9, 15)

-- State the theorem
theorem calc_area_of_quadrilateral (a b c d : ℕ) (area1 area2 area3 : ℕ):
  area1 = 6 →
  area2 = 9 →
  area3 = 15 →
  a + b + c + d = area1 + area2 + area3 →
  d = 65 :=
  sorry

end calc_area_of_quadrilateral_l1973_197393


namespace range_of_m_l1973_197355

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x1 : ℝ, 0 < x1 ∧ x1 < 3 / 2 → ∃ x2 : ℝ, 0 < x2 ∧ x2 < 3 / 2 ∧ f x1 > g x2) →
  (∀ x : ℝ, f x = -x + x * Real.log x + m) →
  (∀ x : ℝ, g x = -3 * Real.exp x / (3 + 4 * x ^ 2)) →
  m > 1 - 3 / 4 * Real.sqrt (Real.exp 1) :=
by
  sorry

end range_of_m_l1973_197355


namespace polynomial_roots_l1973_197354

theorem polynomial_roots : ∀ x : ℝ, (x^3 - 4*x^2 - x + 4) * (x - 3) * (x + 2) = 0 ↔ 
  (x = -2 ∨ x = -1 ∨ x = 1 ∨ x = 3 ∨ x = 4) :=
by 
  sorry

end polynomial_roots_l1973_197354


namespace total_cost_is_15_75_l1973_197310

def price_sponge : ℝ := 4.20
def price_shampoo : ℝ := 7.60
def price_soap : ℝ := 3.20
def tax_rate : ℝ := 0.05
def total_cost_before_tax : ℝ := price_sponge + price_shampoo + price_soap
def tax_amount : ℝ := tax_rate * total_cost_before_tax
def total_cost_including_tax : ℝ := total_cost_before_tax + tax_amount

theorem total_cost_is_15_75 : total_cost_including_tax = 15.75 :=
by sorry

end total_cost_is_15_75_l1973_197310


namespace boat_speed_in_still_water_l1973_197376

variable (x : ℝ) -- Speed of the boat in still water
variable (r : ℝ) -- Rate of the stream
variable (d : ℝ) -- Distance covered downstream
variable (t : ℝ) -- Time taken downstream

theorem boat_speed_in_still_water (h_rate : r = 5) (h_distance : d = 168) (h_time : t = 8) :
  x = 16 :=
by
  -- Substitute conditions into the equation.
  -- Calculate the effective speed downstream.
  -- Solve x from the resulting equation.
  sorry

end boat_speed_in_still_water_l1973_197376


namespace angle_B_value_l1973_197314

noncomputable def degree_a (A : ℝ) : Prop := A = 30 ∨ A = 60

noncomputable def degree_b (A B : ℝ) : Prop := B = 3 * A - 60

theorem angle_B_value (A B : ℝ) 
  (h1 : B = 3 * A - 60)
  (h2 : A = 30 ∨ A = 60) :
  B = 30 ∨ B = 120 :=
by
  sorry

end angle_B_value_l1973_197314


namespace ratio_of_group_average_l1973_197360

theorem ratio_of_group_average
  (d l e : ℕ)
  (avg_group_age : ℕ := 45) 
  (avg_doctors_age : ℕ := 40) 
  (avg_lawyers_age : ℕ := 55) 
  (avg_engineers_age : ℕ := 35)
  (h : (40 * d + 55 * l + 35 * e) / (d + l + e) = avg_group_age)
  : d = 2 * l - e ∧ l = 2 * e :=
sorry

end ratio_of_group_average_l1973_197360


namespace football_points_difference_l1973_197381

theorem football_points_difference :
  let points_per_touchdown := 7
  let brayden_gavin_touchdowns := 7
  let cole_freddy_touchdowns := 9
  let brayden_gavin_points := brayden_gavin_touchdowns * points_per_touchdown
  let cole_freddy_points := cole_freddy_touchdowns * points_per_touchdown
  cole_freddy_points - brayden_gavin_points = 14 :=
by sorry

end football_points_difference_l1973_197381


namespace intersection_A_B_l1973_197378

def set_A (x : ℝ) : Prop := x^2 - 4 * x - 5 < 0
def set_B (x : ℝ) : Prop := 2 < x ∧ x < 4

theorem intersection_A_B (x : ℝ) :
  (set_A x ∧ set_B x) ↔ 2 < x ∧ x < 4 :=
by sorry

end intersection_A_B_l1973_197378


namespace sum_of_first_12_terms_l1973_197330

noncomputable def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

def Sn (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

theorem sum_of_first_12_terms (a d : ℤ) (h1 : a + d * 4 = 3 * (a + d * 2))
                             (h2 : a + d * 9 = 14) : Sn a d 12 = 84 := 
by
  sorry

end sum_of_first_12_terms_l1973_197330


namespace merchant_profit_percentage_l1973_197305

-- Given
def initial_cost_price : ℝ := 100
def marked_price : ℝ := initial_cost_price + 0.50 * initial_cost_price
def discount_percentage : ℝ := 0.20
def discount : ℝ := discount_percentage * marked_price
def selling_price : ℝ := marked_price - discount

-- Prove
theorem merchant_profit_percentage :
  ((selling_price - initial_cost_price) / initial_cost_price) * 100 = 20 :=
by
  sorry

end merchant_profit_percentage_l1973_197305


namespace chess_tournament_third_place_wins_l1973_197399

theorem chess_tournament_third_place_wins :
  ∀ (points : Fin 8 → ℕ)
  (total_games : ℕ)
  (total_points : ℕ),
  (total_games = 28) →
  (∀ i j : Fin 8, i ≠ j → points i ≠ points j) →
  ((points 1) = (points 4 + points 5 + points 6 + points 7)) →
  (points 2 > points 4) →
  ∃ (games_won : Fin 8 → Fin 8 → Prop),
  (games_won 2 4) :=
by
  sorry

end chess_tournament_third_place_wins_l1973_197399


namespace parade_team_people_count_min_l1973_197365

theorem parade_team_people_count_min (n : ℕ) :
  n ≥ 1000 ∧ n % 5 = 0 ∧ n % 4 = 3 ∧ n % 3 = 2 ∧ n % 2 = 1 → n = 1045 :=
by
  sorry

end parade_team_people_count_min_l1973_197365


namespace length_of_platform_l1973_197319

theorem length_of_platform (length_of_train speed_of_train time_to_cross : ℕ) 
    (h1 : length_of_train = 450) (h2 : speed_of_train = 126) (h3 : time_to_cross = 20) :
    ∃ length_of_platform : ℕ, length_of_platform = 250 := 
by 
  sorry

end length_of_platform_l1973_197319


namespace cookies_per_person_l1973_197352

/-- Brenda's mother made cookies for 5 people. She prepared 35 cookies, 
    and each of them had the same number of cookies. 
    We aim to prove that each person had 7 cookies. --/
theorem cookies_per_person (total_cookies : ℕ) (number_of_people : ℕ) 
  (h1 : total_cookies = 35) (h2 : number_of_people = 5) : total_cookies / number_of_people = 7 := 
by
  sorry

end cookies_per_person_l1973_197352


namespace calculate_expression_l1973_197340

theorem calculate_expression : 2 * (-3)^3 - 4 * (-3) + 15 = -27 := 
by
  sorry

end calculate_expression_l1973_197340


namespace permutations_of_six_digit_number_l1973_197338

/-- 
Theorem: The number of distinct permutations of the digits 1, 1, 3, 3, 3, 8 
to form six-digit positive integers is 60. 
-/
theorem permutations_of_six_digit_number : 
  (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 3)) = 60 := 
by 
  sorry

end permutations_of_six_digit_number_l1973_197338


namespace determine_d_and_vertex_l1973_197367

-- Definition of the quadratic equation
def g (x d : ℝ) : ℝ := 3 * x^2 + 12 * x + d

-- The proof problem
theorem determine_d_and_vertex (d : ℝ) :
  (∃ x : ℝ, g x d = 0 ∧ ∀ y : ℝ, g y d ≥ g x d) ↔ (d = 12 ∧ ∀ x : ℝ, 3 > 0 ∧ (g x d ≥ g 0 d)) := 
by 
  sorry

end determine_d_and_vertex_l1973_197367


namespace sara_staircase_steps_l1973_197374

-- Define the problem statement and conditions
theorem sara_staircase_steps (n : ℕ) :
  (3 * n * (n + 1) / 2 = 270) → n = 12 := 
by
  intro h
  sorry

end sara_staircase_steps_l1973_197374


namespace problem1_solution_problem2_solution_problem3_solution_l1973_197333

noncomputable def problem1 : Real :=
  3 * Real.sqrt 3 + Real.sqrt 8 - Real.sqrt 2 + Real.sqrt 27

theorem problem1_solution : problem1 = 6 * Real.sqrt 3 + Real.sqrt 2 := by
  sorry

noncomputable def problem2 : Real :=
  (1/2) * (Real.sqrt 3 + Real.sqrt 5) - (3/4) * (Real.sqrt 5 - Real.sqrt 12)

theorem problem2_solution : problem2 = 2 * Real.sqrt 3 - (1/4) * Real.sqrt 5 := by
  sorry

noncomputable def problem3 : Real :=
  (2 * Real.sqrt 5 + Real.sqrt 6) * (2 * Real.sqrt 5 - Real.sqrt 6) - (Real.sqrt 5 - Real.sqrt 6) ^ 2

theorem problem3_solution : problem3 = 3 + 2 * Real.sqrt 30 := by
  sorry

end problem1_solution_problem2_solution_problem3_solution_l1973_197333


namespace additional_sugar_is_correct_l1973_197383

def sugar_needed : ℝ := 450
def sugar_in_house : ℝ := 287
def sugar_in_basement_kg : ℝ := 50
def kg_to_lbs : ℝ := 2.20462

def sugar_in_basement : ℝ := sugar_in_basement_kg * kg_to_lbs
def total_sugar : ℝ := sugar_in_house + sugar_in_basement
def additional_sugar_needed : ℝ := sugar_needed - total_sugar

theorem additional_sugar_is_correct : additional_sugar_needed = 52.769 := by
  sorry

end additional_sugar_is_correct_l1973_197383


namespace inequalities_quadrants_l1973_197397

theorem inequalities_quadrants :
  (∀ x y : ℝ, y > 2 * x → y > 4 - x → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)) := sorry

end inequalities_quadrants_l1973_197397


namespace jill_salary_l1973_197391

-- Defining the conditions
variables (S : ℝ) -- Jill's net monthly salary
variables (discretionary_income : ℝ) -- One fifth of her net monthly salary
variables (vacation_fund : ℝ) -- 30% of discretionary income into a vacation fund
variables (savings : ℝ) -- 20% of discretionary income into savings
variables (eating_out_socializing : ℝ) -- 35% of discretionary income on eating out and socializing
variables (leftover : ℝ) -- The remaining amount, which is $99

-- Given Conditions
-- One fifth of her net monthly salary left as discretionary income
def one_fifth_of_salary : Prop := discretionary_income = (1/5) * S

-- 30% into a vacation fund
def vacation_allocation : Prop := vacation_fund = 0.30 * discretionary_income

-- 20% into savings
def savings_allocation : Prop := savings = 0.20 * discretionary_income

-- 35% on eating out and socializing
def socializing_allocation : Prop := eating_out_socializing = 0.35 * discretionary_income

-- This leaves her with $99
def leftover_amount : Prop := leftover = 99

-- Eqution considering all conditions results her leftover being $99
def income_allocation : Prop := 
  vacation_fund + savings + eating_out_socializing + leftover = discretionary_income

-- The main proof goal: given all the conditions, Jill's net monthly salary is $3300
theorem jill_salary : 
  one_fifth_of_salary S discretionary_income → 
  vacation_allocation discretionary_income vacation_fund → 
  savings_allocation discretionary_income savings → 
  socializing_allocation discretionary_income eating_out_socializing → 
  leftover_amount leftover → 
  income_allocation discretionary_income vacation_fund savings eating_out_socializing leftover → 
  S = 3300 := by sorry

end jill_salary_l1973_197391


namespace twice_a_minus_4_nonnegative_l1973_197301

theorem twice_a_minus_4_nonnegative (a : ℝ) : 2 * a - 4 ≥ 0 ↔ 2 * a - 4 = 0 ∨ 2 * a - 4 > 0 := 
by
  sorry

end twice_a_minus_4_nonnegative_l1973_197301


namespace negation_of_exists_x_squared_gt_one_l1973_197377

-- Negation of the proposition
theorem negation_of_exists_x_squared_gt_one :
  ¬ (∃ x : ℝ, x^2 > 1) ↔ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 :=
by
  sorry

end negation_of_exists_x_squared_gt_one_l1973_197377


namespace set_complement_intersection_l1973_197351

theorem set_complement_intersection
  (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)
  (hU : U = {0, 1, 2, 3, 4})
  (hM : M = {0, 1, 2})
  (hN : N = {2, 3}) :
  ((U \ M) ∩ N) = {3} :=
  by sorry

end set_complement_intersection_l1973_197351


namespace least_add_to_divisible_by_17_l1973_197392

/-- Given that the remainder when 433124 is divided by 17 is 2,
    prove that the least number that must be added to 433124 to make 
    it divisible by 17 is 15. -/
theorem least_add_to_divisible_by_17: 
  (433124 % 17 = 2) → 
  (∃ n, n ≥ 0 ∧ (433124 + n) % 17 = 0 ∧ n = 15) := 
by
  sorry

end least_add_to_divisible_by_17_l1973_197392


namespace cookie_ratio_l1973_197387

theorem cookie_ratio (K : ℕ) (h1 : K / 2 + K + 24 = 33) : 24 / K = 4 :=
by {
  sorry
}

end cookie_ratio_l1973_197387


namespace lines_perpendicular_l1973_197343

-- Define the lines l1 and l2
def line1 (m x y : ℝ) := m * x + y - 1 = 0
def line2 (m x y : ℝ) := x + (m - 1) * y + 2 = 0

-- State the problem: Find the value of m such that the lines l1 and l2 are perpendicular.
theorem lines_perpendicular (m : ℝ) (h₁ : line1 m x y) (h₂ : line2 m x y) : m = 1/2 := 
sorry

end lines_perpendicular_l1973_197343


namespace students_preferring_windows_is_correct_l1973_197332

-- Define the total number of students surveyed
def total_students : ℕ := 210

-- Define the number of students preferring Mac
def students_preferring_mac : ℕ := 60

-- Define the number of students preferring both Mac and Windows equally
def students_preferring_both : ℕ := students_preferring_mac / 3

-- Define the number of students with no preference
def students_no_preference : ℕ := 90

-- Calculate the total number of students with a preference
def students_with_preference : ℕ := total_students - students_no_preference

-- Calculate the number of students preferring Windows
def students_preferring_windows : ℕ := students_with_preference - (students_preferring_mac + students_preferring_both)

-- State the theorem to prove that the number of students preferring Windows is 40
theorem students_preferring_windows_is_correct : students_preferring_windows = 40 :=
by
  -- calculations based on definitions
  unfold students_preferring_windows students_with_preference students_preferring_mac students_preferring_both students_no_preference total_students
  sorry

end students_preferring_windows_is_correct_l1973_197332


namespace time_to_pass_bridge_l1973_197372

noncomputable def train_length : Real := 357
noncomputable def speed_km_per_hour : Real := 42
noncomputable def bridge_length : Real := 137

noncomputable def speed_m_per_s : Real := speed_km_per_hour * (1000 / 3600)

noncomputable def total_distance : Real := train_length + bridge_length

noncomputable def time_to_pass : Real := total_distance / speed_m_per_s

theorem time_to_pass_bridge : abs (time_to_pass - 42.33) < 0.01 :=
sorry

end time_to_pass_bridge_l1973_197372


namespace proof_problem_l1973_197302

theorem proof_problem (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a^3 + b^3 = 2 * a * b) : a^2 + b^2 ≤ 1 + a * b := 
sorry

end proof_problem_l1973_197302


namespace candy_eaten_l1973_197323

theorem candy_eaten (x : ℕ) (initial_candy eaten_more remaining : ℕ) (h₁ : initial_candy = 22) (h₂ : eaten_more = 5) (h₃ : remaining = 8) (h₄ : initial_candy - x - eaten_more = remaining) : x = 9 :=
by
  -- proof
  sorry

end candy_eaten_l1973_197323


namespace initial_fliers_l1973_197350

theorem initial_fliers (F : ℕ) (morning_sent afternoon_sent remaining : ℕ) :
  morning_sent = F / 5 → 
  afternoon_sent = (F - morning_sent) / 4 → 
  remaining = F - morning_sent - afternoon_sent → 
  remaining = 1800 → 
  F = 3000 := 
by 
  sorry

end initial_fliers_l1973_197350


namespace number_of_welders_left_l1973_197375

-- Define the constants and variables
def welders_total : ℕ := 36
def days_to_complete : ℕ := 5
def rate : ℝ := 1  -- Assume the rate per welder is 1 for simplicity
def total_work : ℝ := welders_total * days_to_complete * rate

def days_after_first : ℕ := 6
def work_done_in_first_day : ℝ := welders_total * 1 * rate
def remaining_work : ℝ := total_work - work_done_in_first_day

-- Define the theorem to solve for the number of welders x that started to work on another project
theorem number_of_welders_left (x : ℕ) : (welders_total - x) * days_after_first * rate = remaining_work → x = 12 := by
  intros h
  sorry

end number_of_welders_left_l1973_197375


namespace problem1_problem2_problem3_problem4_l1973_197304

-- Problem 1
theorem problem1 (x : ℤ) (h : 4 * x = 20) : x = 5 :=
sorry

-- Problem 2
theorem problem2 (x : ℤ) (h : x - 18 = 40) : x = 58 :=
sorry

-- Problem 3
theorem problem3 (x : ℤ) (h : x / 7 = 12) : x = 84 :=
sorry

-- Problem 4
theorem problem4 (n : ℚ) (h : 8 * n / 2 = 15) : n = 15 / 4 :=
sorry

end problem1_problem2_problem3_problem4_l1973_197304


namespace max_n_for_factorization_l1973_197335

theorem max_n_for_factorization (A B n : ℤ) (AB_cond : A * B = 48) (n_cond : n = 5 * B + A) :
  n ≤ 241 :=
by
  sorry

end max_n_for_factorization_l1973_197335


namespace other_root_of_quadratic_l1973_197342

theorem other_root_of_quadratic (m t : ℝ) : (∀ (x : ℝ),
    (3 * x^2 - m * x - 3 = 0) → 
    (x = 1)) → 
    (1 * t = -1) := 
sorry

end other_root_of_quadratic_l1973_197342


namespace cubic_difference_l1973_197348

theorem cubic_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 50) : a^3 - b^3 = 353.5 := by
  sorry

end cubic_difference_l1973_197348


namespace simplify_expression_l1973_197359

theorem simplify_expression (w x : ℝ) :
  3 * w + 6 * w + 9 * w + 12 * w + 15 * w - 2 * x - 4 * x - 6 * x - 8 * x - 10 * x + 24 = 
  45 * w - 30 * x + 24 :=
by sorry

end simplify_expression_l1973_197359


namespace bruce_total_amount_paid_l1973_197327

-- Definitions for quantities and rates
def quantity_of_grapes : Nat := 8
def rate_per_kg_grapes : Nat := 70
def quantity_of_mangoes : Nat := 11
def rate_per_kg_mangoes : Nat := 55

-- Calculate individual costs
def cost_of_grapes := quantity_of_grapes * rate_per_kg_grapes
def cost_of_mangoes := quantity_of_mangoes * rate_per_kg_mangoes

-- Calculate total amount paid
def total_amount_paid := cost_of_grapes + cost_of_mangoes

-- Statement to prove
theorem bruce_total_amount_paid : total_amount_paid = 1165 := by
  -- Proof is intentionally left as a placeholder
  sorry

end bruce_total_amount_paid_l1973_197327


namespace tile_in_center_l1973_197385

-- Define the coloring pattern of the grid
inductive Color
| A | B | C

-- Predicates for grid, tile placement, and colors
def Grid := Fin 5 × Fin 5

def is_1x3_tile (t : Grid × Grid × Grid) : Prop :=
  -- Ensure each tuple t represents three cells that form a $1 \times 3$ tile
  sorry

def is_tiling (g : Grid → Option Color) : Prop :=
  -- Ensure the entire grid is correctly tiled with the given tiles and within the coloring pattern
  sorry

def center : Grid := (Fin.mk 2 (by decide), Fin.mk 2 (by decide))

-- The theorem statement
theorem tile_in_center (g : Grid → Option Color) : is_tiling g → 
  (∃! tile : Grid, g tile = some Color.B) :=
sorry

end tile_in_center_l1973_197385


namespace john_drinks_42_quarts_per_week_l1973_197328

def gallons_per_day : ℝ := 1.5
def quarts_per_gallon : ℝ := 4
def days_per_week : ℕ := 7

theorem john_drinks_42_quarts_per_week :
  gallons_per_day * quarts_per_gallon * days_per_week = 42 := sorry

end john_drinks_42_quarts_per_week_l1973_197328


namespace fraction_ordering_l1973_197300

theorem fraction_ordering :
  (6:ℚ)/29 < (8:ℚ)/25 ∧ (8:ℚ)/25 < (10:ℚ)/31 :=
by
  sorry

end fraction_ordering_l1973_197300


namespace digit_one_not_in_mean_l1973_197313

def seq : List ℕ := [5, 55, 555, 5555, 55555, 555555, 5555555, 55555555, 555555555]

noncomputable def arithmetic_mean (l : List ℕ) : ℕ := l.sum / l.length

theorem digit_one_not_in_mean :
  ¬(∃ d, d ∈ (arithmetic_mean seq).digits 10 ∧ d = 1) :=
sorry

end digit_one_not_in_mean_l1973_197313


namespace positive_expression_with_b_l1973_197373

-- Defining the conditions and final statement
open Real

theorem positive_expression_with_b (a : ℝ) : (a + 2) * (a + 5) * (a + 8) * (a + 11) + 82 > 0 := 
sorry

end positive_expression_with_b_l1973_197373


namespace find_smallest_number_l1973_197388

theorem find_smallest_number (x y z : ℝ) 
  (h1 : x + y + z = 150) 
  (h2 : y = 3 * x + 10) 
  (h3 : z = x^2 - 5) 
  : x = 10.21 :=
sorry

end find_smallest_number_l1973_197388


namespace complex_pure_imaginary_l1973_197322

theorem complex_pure_imaginary (a : ℂ) : (∃ (b : ℂ), (2 - I) * (a + 2 * I) = b * I) → a = -1 :=
by
  sorry

end complex_pure_imaginary_l1973_197322


namespace product_div_sum_eq_5_quotient_integer_condition_next_consecutive_set_l1973_197370

theorem product_div_sum_eq_5 (x : ℤ) (h : (x^3 - x) / (3 * x) = 5) : x = 4 := by
  sorry

theorem quotient_integer_condition (x : ℤ) : ((∃ k : ℤ, x = 3 * k + 1) ∨ (∃ k : ℤ, x = 3 * k - 1)) ↔ ∃ q : ℤ, (x^3 - x) / (3 * x) = q := by
  sorry

theorem next_consecutive_set (x : ℤ) (h : x = 4) : x - 1 = 3 ∧ x = 4 ∧ x + 1 = 5 := by
  sorry

end product_div_sum_eq_5_quotient_integer_condition_next_consecutive_set_l1973_197370


namespace counterexample_disproves_statement_l1973_197346

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem counterexample_disproves_statement :
  ∃ n : ℕ, ¬ is_prime n ∧ is_prime (n + 3) :=
  by
    use 8
    -- Proof that 8 is not prime
    -- Proof that 11 (8 + 3) is prime
    sorry

end counterexample_disproves_statement_l1973_197346


namespace sin_pi_minus_alpha_cos_2pi_minus_alpha_sin_minus_cos_l1973_197396

-- Problem 1: Given that tan(α) = 3, prove that sin(π - α) * cos(2π - α) = 3 / 10.
theorem sin_pi_minus_alpha_cos_2pi_minus_alpha (α : ℝ) (h : Real.tan α = 3) : 
  Real.sin (Real.pi - α) * Real.cos (2 * Real.pi - α) = 3 / 10 :=
by
  sorry

-- Problem 2: Given that sin(α) * cos(α) = 1/4 and 0 < α < π/4, prove that sin(α) - cos(α) = - sqrt(2) / 2.
theorem sin_minus_cos (α : ℝ) (h₁ : Real.sin α * Real.cos α = 1 / 4) (h₂ : 0 < α) (h₃ : α < Real.pi / 4) :
  Real.sin α - Real.cos α = - (Real.sqrt 2) / 2 :=
by
  sorry

end sin_pi_minus_alpha_cos_2pi_minus_alpha_sin_minus_cos_l1973_197396


namespace runners_meet_time_l1973_197398

theorem runners_meet_time :
  let time_runner_1 := 2
  let time_runner_2 := 4
  let time_runner_3 := 11 / 2
  Nat.lcm time_runner_1 (Nat.lcm time_runner_2 (Nat.lcm (11) 2)) = 44 := by
  sorry

end runners_meet_time_l1973_197398


namespace smaller_balloon_radius_is_correct_l1973_197357

-- Condition: original balloon radius
def original_balloon_radius : ℝ := 2

-- Condition: number of smaller balloons
def num_smaller_balloons : ℕ := 64

-- Question (to be proved): Radius of each smaller balloon
theorem smaller_balloon_radius_is_correct :
  ∃ r : ℝ, (4/3) * Real.pi * (original_balloon_radius^3) = num_smaller_balloons * (4/3) * Real.pi * (r^3) ∧ r = 1/2 := 
by {
  sorry
}

end smaller_balloon_radius_is_correct_l1973_197357


namespace three_tenths_of_number_l1973_197366

theorem three_tenths_of_number (N : ℝ) (h : (1/3) * (1/4) * N = 15) : (3/10) * N = 54 :=
sorry

end three_tenths_of_number_l1973_197366


namespace negated_proposition_l1973_197320

theorem negated_proposition : ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0 := by
  sorry

end negated_proposition_l1973_197320


namespace piggy_bank_total_l1973_197364

def amount_added_in_january: ℕ := 19
def amount_added_in_february: ℕ := 19
def amount_added_in_march: ℕ := 8

theorem piggy_bank_total:
  amount_added_in_january + amount_added_in_february + amount_added_in_march = 46 := by
  sorry

end piggy_bank_total_l1973_197364


namespace interval_monotonically_decreasing_l1973_197384

noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2 * x + 3)

theorem interval_monotonically_decreasing :
  ∀ x y : ℝ, 1 < x → x < 3 → 1 < y → y < 3 → x < y → f y < f x := 
by sorry

end interval_monotonically_decreasing_l1973_197384


namespace total_coins_l1973_197345

-- Defining the conditions
def stack1 : Nat := 4
def stack2 : Nat := 8

-- Statement of the proof problem
theorem total_coins : stack1 + stack2 = 12 :=
by
  sorry

end total_coins_l1973_197345


namespace company_percentage_increase_l1973_197363

theorem company_percentage_increase (employees_jan employees_dec : ℝ) (P_increase : ℝ) 
  (h_jan : employees_jan = 391.304347826087)
  (h_dec : employees_dec = 450)
  (h_P : P_increase = 15) : 
  (employees_dec - employees_jan) / employees_jan * 100 = P_increase :=
by 
  sorry

end company_percentage_increase_l1973_197363


namespace profit_percentage_B_l1973_197337

theorem profit_percentage_B (cost_price_A : ℝ) (sell_price_C : ℝ) 
  (profit_A_percent : ℝ) (profit_B_percent : ℝ) 
  (cost_price_A_eq : cost_price_A = 148) 
  (sell_price_C_eq : sell_price_C = 222) 
  (profit_A_percent_eq : profit_A_percent = 0.2) :
  profit_B_percent = 0.25 := 
by
  have cost_price_B := cost_price_A * (1 + profit_A_percent)
  have profit_B := sell_price_C - cost_price_B
  have profit_B_percent := (profit_B / cost_price_B) * 100 
  sorry

end profit_percentage_B_l1973_197337


namespace relatively_prime_perfect_squares_l1973_197317

theorem relatively_prime_perfect_squares (a b c : ℤ) (h_gcd : Int.gcd (Int.gcd a b) c = 1) 
    (h_eq : (1:ℚ) / a + (1:ℚ) / b = (1:ℚ) / c) :
    ∃ x y z : ℤ, (a + b = x^2 ∧ a - c = y^2 ∧ b - c = z^2) :=
  sorry

end relatively_prime_perfect_squares_l1973_197317


namespace jenny_research_time_l1973_197382

noncomputable def time_spent_on_research (total_hours : ℕ) (proposal_hours : ℕ) (report_hours : ℕ) : ℕ :=
  total_hours - proposal_hours - report_hours

theorem jenny_research_time : time_spent_on_research 20 2 8 = 10 := by
  sorry

end jenny_research_time_l1973_197382


namespace range_of_a_l1973_197321

theorem range_of_a (a : ℝ) :
  (1 ∉ {x : ℝ | x^2 - 2 * x + a > 0}) → a ≤ 1 :=
by
  sorry

end range_of_a_l1973_197321


namespace simplify_trig_expression_l1973_197358

open Real

theorem simplify_trig_expression (theta : ℝ) (h : 0 < theta ∧ theta < π / 4) :
  sqrt (1 - 2 * sin (π + theta) * sin (3 * π / 2 - theta)) = cos theta - sin theta :=
sorry

end simplify_trig_expression_l1973_197358


namespace sum_seq_equals_2_pow_n_minus_1_l1973_197334

-- Define the sequences a_n and b_n with given conditions
def a (n : ℕ) : ℕ := if n = 0 then 2 else if n = 1 then 4 else sorry
def b (n : ℕ) : ℕ := if n = 0 then 2 else if n = 1 then 4 else sorry

-- Relation for a_n: 2a_{n+1} = a_n + a_{n+2}
axiom a_relation (n : ℕ) : 2 * a (n + 1) = a n + a (n + 2)

-- Inequalities for b_n
axiom b_inequality_1 (n : ℕ) : b (n + 1) - b n < 2^n + 1 / 2
axiom b_inequality_2 (n : ℕ) : b (n + 2) - b n > 3 * 2^n - 1

-- Note that b_n ∈ ℤ is implied by the definition being in ℕ

-- Prove that the sum of the first n terms of the sequence { n * b_n / a_n }
theorem sum_seq_equals_2_pow_n_minus_1 (n : ℕ) : 
  (Finset.range n).sum (λ k => k * b k / a k) = 2^n - 1 := 
sorry

end sum_seq_equals_2_pow_n_minus_1_l1973_197334


namespace remainder_proof_l1973_197336

theorem remainder_proof (x y u v : ℕ) (h1 : x = u * y + v) (h2 : 0 ≤ v ∧ v < y) : 
  (x + y * u^2 + 3 * v) % y = 4 * v % y :=
by
  sorry

end remainder_proof_l1973_197336


namespace transfer_people_eq_l1973_197341

theorem transfer_people_eq : ∃ x : ℕ, 22 + x = 2 * (26 - x) := 
by 
  -- hypothesis and equation statement
  sorry

end transfer_people_eq_l1973_197341


namespace cos_tan_quadrant_l1973_197329

theorem cos_tan_quadrant (α : ℝ) 
  (hcos : Real.cos α < 0) 
  (htan : Real.tan α > 0) : 
  (2 * π / 2 < α ∧ α < π) :=
by
  sorry

end cos_tan_quadrant_l1973_197329


namespace find_numbers_l1973_197368

theorem find_numbers (x y a : ℕ) (h1 : x = 6 * y - a) (h2 : x + y = 38) : 7 * x = 228 - a → y = 38 - x :=
by
  sorry

end find_numbers_l1973_197368
