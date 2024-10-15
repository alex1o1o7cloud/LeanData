import Mathlib

namespace NUMINAMATH_GPT_shipping_cost_per_unit_l539_53927

noncomputable def fixed_monthly_costs : ℝ := 16500
noncomputable def production_cost_per_component : ℝ := 80
noncomputable def production_quantity : ℝ := 150
noncomputable def selling_price_per_component : ℝ := 193.33

theorem shipping_cost_per_unit :
  ∀ (S : ℝ), (production_quantity * production_cost_per_component + production_quantity * S + fixed_monthly_costs) ≤ (production_quantity * selling_price_per_component) → S ≤ 3.33 :=
by
  intro S
  sorry

end NUMINAMATH_GPT_shipping_cost_per_unit_l539_53927


namespace NUMINAMATH_GPT_minValue_l539_53929

noncomputable def minValueOfExpression (a b c : ℝ) : ℝ :=
  (1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a))

theorem minValue (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 2 * a + 2 * b + 2 * c = 3) : 
  minValueOfExpression a b c = 2 :=
  sorry

end NUMINAMATH_GPT_minValue_l539_53929


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l539_53979

noncomputable def A : Set ℝ := {x | x^2 - 1 ≤ 0}

noncomputable def B : Set ℝ := {x | (x - 2) / x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l539_53979


namespace NUMINAMATH_GPT_shot_put_surface_area_l539_53913

noncomputable def radius (d : ℝ) : ℝ := d / 2

noncomputable def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem shot_put_surface_area :
  surface_area (radius 5) = 25 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_shot_put_surface_area_l539_53913


namespace NUMINAMATH_GPT_sophia_age_in_three_years_l539_53999

def current_age_jeremy : Nat := 40
def current_age_sebastian : Nat := current_age_jeremy + 4

def sum_ages_in_three_years (age_jeremy age_sebastian age_sophia : Nat) : Nat :=
  (age_jeremy + 3) + (age_sebastian + 3) + (age_sophia + 3)

theorem sophia_age_in_three_years (age_sophia : Nat) 
  (h1 : sum_ages_in_three_years current_age_jeremy current_age_sebastian age_sophia = 150) :
  age_sophia + 3 = 60 := by
  sorry

end NUMINAMATH_GPT_sophia_age_in_three_years_l539_53999


namespace NUMINAMATH_GPT_race_min_distance_l539_53935

noncomputable def min_distance : ℝ :=
  let A : ℝ × ℝ := (0, 300)
  let B : ℝ × ℝ := (1200, 500)
  let wall_length : ℝ := 1200
  let B' : ℝ × ℝ := (1200, -500)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance A B'

theorem race_min_distance :
  min_distance = 1442 := sorry

end NUMINAMATH_GPT_race_min_distance_l539_53935


namespace NUMINAMATH_GPT_trajectory_equation_l539_53950

theorem trajectory_equation 
  (P : ℝ × ℝ)
  (h : (P.2 / (P.1 + 4)) * (P.2 / (P.1 - 4)) = -4 / 9) :
  P.1 ≠ 4 ∧ P.1 ≠ -4 → P.1^2 / 64 + P.2^2 / (64 / 9) = 1 :=
by
  sorry

end NUMINAMATH_GPT_trajectory_equation_l539_53950


namespace NUMINAMATH_GPT_santiago_more_roses_l539_53909

def red_roses_santiago := 58
def red_roses_garrett := 24
def red_roses_difference := red_roses_santiago - red_roses_garrett

theorem santiago_more_roses : red_roses_difference = 34 := by
  sorry

end NUMINAMATH_GPT_santiago_more_roses_l539_53909


namespace NUMINAMATH_GPT_smaller_variance_stability_l539_53960

variable {α : Type*}
variable [Nonempty α]

def same_average (X Y : α → ℝ) (avg : ℝ) : Prop := 
  (∀ x, X x = avg) ∧ (∀ y, Y y = avg)

def smaller_variance_is_stable (X Y : α → ℝ) : Prop := 
  (X = Y)

theorem smaller_variance_stability {X Y : α → ℝ} (avg : ℝ) :
  same_average X Y avg → smaller_variance_is_stable X Y :=
by sorry

end NUMINAMATH_GPT_smaller_variance_stability_l539_53960


namespace NUMINAMATH_GPT_alloy_problem_l539_53924

theorem alloy_problem (x y : ℝ) 
  (h1 : x + y = 1000) 
  (h2 : 0.25 * x + 0.50 * y = 450) 
  (hx : 0 ≤ x) 
  (hy : 0 ≤ y) :
  x = 200 ∧ y = 800 := 
sorry

end NUMINAMATH_GPT_alloy_problem_l539_53924


namespace NUMINAMATH_GPT_sqrt_720_simplified_l539_53980

theorem sqrt_720_simplified : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  have h1 : 720 = 2^4 * 3^2 * 5 := by norm_num
  -- Here we use another proven fact or logic per original conditions and definition
  sorry

end NUMINAMATH_GPT_sqrt_720_simplified_l539_53980


namespace NUMINAMATH_GPT_eight_child_cotton_l539_53941

theorem eight_child_cotton {a_1 a_8 d S_8 : ℕ} 
  (h1 : d = 17)
  (h2 : S_8 = 996)
  (h3 : 8 * a_1 + 28 * d = S_8) :
  a_8 = a_1 + 7 * d → a_8 = 184 := by
  intro h4
  subst_vars
  sorry

end NUMINAMATH_GPT_eight_child_cotton_l539_53941


namespace NUMINAMATH_GPT_solve_for_x_l539_53965

def δ (x : ℝ) : ℝ := 4 * x + 5
def φ (x : ℝ) : ℝ := 5 * x + 4

theorem solve_for_x (x : ℝ) (h : δ (φ x) = 4) : x = -17 / 20 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l539_53965


namespace NUMINAMATH_GPT_percentage_female_guests_from_jay_family_l539_53930

def total_guests : ℕ := 240
def female_guests_percentage : ℕ := 60
def female_guests_from_jay_family : ℕ := 72

theorem percentage_female_guests_from_jay_family :
  (female_guests_from_jay_family : ℚ) / (total_guests * (female_guests_percentage / 100) : ℚ) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_female_guests_from_jay_family_l539_53930


namespace NUMINAMATH_GPT_problem1_problem2_l539_53972

-- Define the sets A and B
def setA (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < 2 * a + 1 }
def setB : Set ℝ := { x | 0 < x ∧ x < 1 }

-- Prove that for a = 1/2, A ∩ B = { x | 0 < x ∧ x < 1 }
theorem problem1 : setA (1/2) ∩ setB = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

-- Prove that if A ∩ B = ∅, then a ≤ -1/2 or a ≥ 2
theorem problem2 (a : ℝ) (h : setA a ∩ setB = ∅) : a ≤ -1/2 ∨ a ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l539_53972


namespace NUMINAMATH_GPT_number_of_2_dollar_socks_l539_53962

theorem number_of_2_dollar_socks :
  ∃ (a b c : ℕ), (a + b + c = 15) ∧ (2 * a + 3 * b + 5 * c = 40) ∧ (a ≥ 1) ∧ (b ≥ 1) ∧ (c ≥ 1) ∧ (a = 7 ∨ a = 9 ∨ a = 11) :=
by {
  -- The details of the proof will go here, but we skip it for our requirements
  sorry
}

end NUMINAMATH_GPT_number_of_2_dollar_socks_l539_53962


namespace NUMINAMATH_GPT_ordering_of_exponentials_l539_53923

theorem ordering_of_exponentials :
  let A := 3^20
  let B := 6^10
  let C := 2^30
  B < A ∧ A < C :=
by
  -- Definitions and conditions
  have h1 : 6^10 = 3^10 * 2^10 := by sorry
  have h2 : 3^10 = 59049 := by sorry
  have h3 : 2^10 = 1024 := by sorry
  have h4 : 2^30 = (2^10)^3 := by sorry
  
  -- We know 3^20, 6^10, 2^30 by definition and conditions
  -- Comparison
  have h5 : 3^20 = (3^10)^2 := by sorry
  have h6 : 2^30 = 1024^3 := by sorry
  
  -- Combine to get results
  have h7 : (3^10)^2 > 6^10 := by sorry
  have h8 : 1024^3 > 6^10 := by sorry
  have h9 : 1024^3 > (3^10)^2 := by sorry

  exact ⟨h7, h9⟩

end NUMINAMATH_GPT_ordering_of_exponentials_l539_53923


namespace NUMINAMATH_GPT_monthly_energy_consumption_l539_53904

-- Defining the given conditions
def power_fan_kW : ℝ := 0.075 -- kilowatts
def hours_per_day : ℝ := 8 -- hours per day
def days_per_month : ℝ := 30 -- days per month

-- The math proof statement with conditions and the expected answer
theorem monthly_energy_consumption : (power_fan_kW * hours_per_day * days_per_month) = 18 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_monthly_energy_consumption_l539_53904


namespace NUMINAMATH_GPT_amrita_bakes_cake_next_thursday_l539_53967

theorem amrita_bakes_cake_next_thursday (n m : ℕ) (h1 : n = 5) (h2 : m = 7) : Nat.lcm n m = 35 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_amrita_bakes_cake_next_thursday_l539_53967


namespace NUMINAMATH_GPT_no_two_obtuse_angles_in_triangle_l539_53958

theorem no_two_obtuse_angles_in_triangle (A B C : ℝ) 
  (h1 : 0 < A) (h2 : A < 180) 
  (h3 : 0 < B) (h4 : B < 180) 
  (h5 : 0 < C) (h6 : C < 180)
  (h7 : A + B + C = 180) 
  (h8 : A > 90) (h9 : B > 90) : false :=
by
  sorry

end NUMINAMATH_GPT_no_two_obtuse_angles_in_triangle_l539_53958


namespace NUMINAMATH_GPT_derivative_of_y_l539_53990

open Real

noncomputable def y (x : ℝ) : ℝ := (cos (2 * x)) ^ ((log (cos (2 * x))) / 4)

theorem derivative_of_y (x : ℝ) :
  deriv y x = -((cos (2 * x)) ^ ((log (cos (2 * x))) / 4)) * (tan (2 * x)) * (log (cos (2 * x))) := by
    sorry

end NUMINAMATH_GPT_derivative_of_y_l539_53990


namespace NUMINAMATH_GPT_sport_vs_std_ratio_comparison_l539_53937

/-- Define the ratios for the standard formulation. -/
def std_flavor_syrup_ratio := 1 / 12
def std_flavor_water_ratio := 1 / 30

/-- Define the conditions for the sport formulation. -/
def sport_water := 15 -- ounces of water in the sport formulation
def sport_syrup := 1 -- ounce of corn syrup in the sport formulation

/-- The ratio of flavoring to water in the sport formulation is half that of the standard formulation. -/
def sport_flavor_water_ratio := std_flavor_water_ratio / 2

/-- Calculate the amount of flavoring in the sport formulation. -/
def sport_flavor := sport_water * sport_flavor_water_ratio

/-- The ratio of flavoring to corn syrup in the sport formulation. -/
def sport_flavor_syrup_ratio := sport_flavor / sport_syrup

/-- The proof problem statement. -/
theorem sport_vs_std_ratio_comparison : sport_flavor_syrup_ratio = 3 * std_flavor_syrup_ratio := 
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_sport_vs_std_ratio_comparison_l539_53937


namespace NUMINAMATH_GPT_num_triangles_with_perimeter_20_l539_53975

theorem num_triangles_with_perimeter_20 : 
  ∃ (triangles : List (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ triangles → a + b + c = 20 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ b + c > a ∧ c + a > b) ∧ 
    triangles.length = 8 :=
sorry

end NUMINAMATH_GPT_num_triangles_with_perimeter_20_l539_53975


namespace NUMINAMATH_GPT_perfect_square_proof_l539_53944

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem perfect_square_proof :
  isPerfectSquare (factorial 22 * factorial 23 * factorial 24 / 12) :=
sorry

end NUMINAMATH_GPT_perfect_square_proof_l539_53944


namespace NUMINAMATH_GPT_line_slope_intercept_l539_53926

theorem line_slope_intercept (a b : ℝ) 
  (h1 : (7 : ℝ) = a * 3 + b) 
  (h2 : (13 : ℝ) = a * (9/2) + b) : 
  a - b = 9 := 
sorry

end NUMINAMATH_GPT_line_slope_intercept_l539_53926


namespace NUMINAMATH_GPT_min_tries_to_get_blue_and_yellow_l539_53976

theorem min_tries_to_get_blue_and_yellow 
  (purple blue yellow : ℕ) 
  (h_purple : purple = 7) 
  (h_blue : blue = 5)
  (h_yellow : yellow = 11) :
  ∃ n, n = 9 ∧ (∀ tries, tries ≥ n → (∃ i j, (i ≤ purple ∧ j ≤ tries - i ∧ j ≤ blue) → (∃ k, k = tries - i - j ∧ k ≤ yellow))) :=
by sorry

end NUMINAMATH_GPT_min_tries_to_get_blue_and_yellow_l539_53976


namespace NUMINAMATH_GPT_magnitude_diff_l539_53918

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions
def condition_1 : ‖a‖ = 2 := sorry
def condition_2 : ‖b‖ = 2 := sorry
def condition_3 : ‖a + b‖ = Real.sqrt 7 := sorry

-- Proof statement
theorem magnitude_diff (a b : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 2) 
  (h3 : ‖a + b‖ = Real.sqrt 7) : 
  ‖a - b‖ = 3 :=
sorry

end NUMINAMATH_GPT_magnitude_diff_l539_53918


namespace NUMINAMATH_GPT_lines_per_stanza_l539_53977

-- Define the number of stanzas
def num_stanzas : ℕ := 20

-- Define the number of words per line
def words_per_line : ℕ := 8

-- Define the total number of words in the poem
def total_words : ℕ := 1600

-- Theorem statement to prove the number of lines per stanza
theorem lines_per_stanza : 
  (total_words / words_per_line) / num_stanzas = 10 := 
by sorry

end NUMINAMATH_GPT_lines_per_stanza_l539_53977


namespace NUMINAMATH_GPT_bestCompletion_is_advantage_l539_53945

-- Defining the phrase and the list of options
def phrase : String := "British students have a language ____ for jobs in the USA and Australia"

def options : List (String × String) := 
  [("A", "chance"), ("B", "ability"), ("C", "possibility"), ("D", "advantage")]

-- Defining the best completion function (using a placeholder 'sorry' for the logic which is not the focus here)
noncomputable def bestCompletion (phrase : String) (options : List (String × String)) : String :=
  "advantage"  -- We assume given the problem that this function correctly identifies 'advantage'

-- Lean theorem stating the desired property
theorem bestCompletion_is_advantage : bestCompletion phrase options = "advantage" :=
by sorry

end NUMINAMATH_GPT_bestCompletion_is_advantage_l539_53945


namespace NUMINAMATH_GPT_common_difference_of_common_terms_l539_53946

def sequence_a (n : ℕ) : ℕ := 4 * n - 3
def sequence_b (k : ℕ) : ℕ := 3 * k - 1

theorem common_difference_of_common_terms :
  ∃ (d : ℕ), (∀ (m : ℕ), 12 * m + 5 ∈ { x | ∃ (n k : ℕ), sequence_a n = x ∧ sequence_b k = x }) ∧ d = 12 := 
sorry

end NUMINAMATH_GPT_common_difference_of_common_terms_l539_53946


namespace NUMINAMATH_GPT_some_number_is_five_l539_53964

theorem some_number_is_five (x : ℕ) (some_number : ℕ) (h1 : x = 5) (h2 : x / some_number + 3 = 4) : some_number = 5 := by
  sorry

end NUMINAMATH_GPT_some_number_is_five_l539_53964


namespace NUMINAMATH_GPT_nursing_home_received_boxes_l539_53981

-- Each condition will be a definition in Lean 4.
def vitamins := 472
def supplements := 288
def total_boxes := 760

-- Statement of the proof problem in Lean
theorem nursing_home_received_boxes : vitamins + supplements = total_boxes := by
  sorry

end NUMINAMATH_GPT_nursing_home_received_boxes_l539_53981


namespace NUMINAMATH_GPT_no_real_roots_l539_53989

noncomputable def polynomial (p : ℝ) (x : ℝ) : ℝ :=
  x^4 + 4 * p * x^3 + 6 * x^2 + 4 * p * x + 1

theorem no_real_roots (p : ℝ) :
  (p > -Real.sqrt 5 / 2) ∧ (p < Real.sqrt 5 / 2) ↔ ¬(∃ x : ℝ, polynomial p x = 0) := by
  sorry

end NUMINAMATH_GPT_no_real_roots_l539_53989


namespace NUMINAMATH_GPT_fraction_numerator_exceeds_denominator_l539_53916

theorem fraction_numerator_exceeds_denominator (x : ℝ) (h1 : -1 ≤ x) (h2 : x ≤ 3) :
  4 * x + 5 > 10 - 3 * x ↔ (5 / 7) < x ∧ x ≤ 3 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_numerator_exceeds_denominator_l539_53916


namespace NUMINAMATH_GPT_find_P_plus_Q_l539_53949

theorem find_P_plus_Q (P Q : ℝ) (h : ∃ b c : ℝ, (x^2 + 3 * x + 4) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) : 
P + Q = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_P_plus_Q_l539_53949


namespace NUMINAMATH_GPT_cost_of_eraser_pencil_l539_53996

-- Define the cost of regular and short pencils
def cost_regular_pencil : ℝ := 0.5
def cost_short_pencil : ℝ := 0.4

-- Define the quantities sold
def quantity_eraser_pencils : ℕ := 200
def quantity_regular_pencils : ℕ := 40
def quantity_short_pencils : ℕ := 35

-- Define the total revenue
def total_revenue : ℝ := 194

-- Problem statement: Prove that the cost of a pencil with an eraser is 0.8
theorem cost_of_eraser_pencil (P : ℝ)
  (h : 200 * P + 40 * cost_regular_pencil + 35 * cost_short_pencil = total_revenue) :
  P = 0.8 := by
  sorry

end NUMINAMATH_GPT_cost_of_eraser_pencil_l539_53996


namespace NUMINAMATH_GPT_find_z_l539_53910

def M (z : ℂ) : Set ℂ := {1, 2, z * Complex.I}
def N : Set ℂ := {3, 4}

theorem find_z (z : ℂ) (h : M z ∩ N = {4}) : z = -4 * Complex.I := by
  sorry

end NUMINAMATH_GPT_find_z_l539_53910


namespace NUMINAMATH_GPT_only_linear_equation_with_two_variables_l539_53915

def is_linear_equation_with_two_variables (eqn : String) : Prop :=
  eqn = "4x-5y=5"

def equation_A := "4x-5y=5"
def equation_B := "xy-y=1"
def equation_C := "4x+5y"
def equation_D := "2/x+5/y=1/7"

theorem only_linear_equation_with_two_variables :
  is_linear_equation_with_two_variables equation_A ∧
  ¬ is_linear_equation_with_two_variables equation_B ∧
  ¬ is_linear_equation_with_two_variables equation_C ∧
  ¬ is_linear_equation_with_two_variables equation_D :=
by
  sorry

end NUMINAMATH_GPT_only_linear_equation_with_two_variables_l539_53915


namespace NUMINAMATH_GPT_palmer_first_week_photos_l539_53925

theorem palmer_first_week_photos :
  ∀ (X : ℕ), 
    100 + X + 2 * X + 80 = 380 →
    X = 67 :=
by
  intros X h
  -- h represents the condition 100 + X + 2 * X + 80 = 380
  sorry

end NUMINAMATH_GPT_palmer_first_week_photos_l539_53925


namespace NUMINAMATH_GPT_strawberries_per_box_l539_53993

-- Define the initial conditions
def initial_strawberries : ℕ := 42
def additional_strawberries : ℕ := 78
def number_of_boxes : ℕ := 6

-- Define the total strawberries based on the given conditions
def total_strawberries : ℕ := initial_strawberries + additional_strawberries

-- The theorem to prove the number of strawberries per box
theorem strawberries_per_box : total_strawberries / number_of_boxes = 20 :=
by
  -- Proof steps would go here, but we use sorry since it's not required
  sorry

end NUMINAMATH_GPT_strawberries_per_box_l539_53993


namespace NUMINAMATH_GPT_winner_votes_percentage_l539_53906

-- Define the total votes as V
def total_votes (winner_votes : ℕ) (winning_margin : ℕ) : ℕ :=
  winner_votes + (winner_votes - winning_margin)

-- Define the percentage function
def percentage_of_votes (part : ℕ) (total : ℕ) : ℕ :=
  (part * 100) / total

-- Lean statement to prove the result
theorem winner_votes_percentage
  (winner_votes : ℕ)
  (winning_margin : ℕ)
  (H_winner_votes : winner_votes = 550)
  (H_winning_margin : winning_margin = 100) :
  percentage_of_votes winner_votes (total_votes winner_votes winning_margin) = 55 := by
  sorry

end NUMINAMATH_GPT_winner_votes_percentage_l539_53906


namespace NUMINAMATH_GPT_ratio_twice_width_to_length_l539_53942

theorem ratio_twice_width_to_length (L W : ℝ) (k : ℤ)
  (h1 : L = 24)
  (h2 : W = 13.5)
  (h3 : L = k * W - 3) :
  2 * W / L = 9 / 8 := by
  sorry

end NUMINAMATH_GPT_ratio_twice_width_to_length_l539_53942


namespace NUMINAMATH_GPT_number_of_tacos_l539_53994

-- Define the conditions and prove the statement
theorem number_of_tacos (T : ℕ) :
  (4 * 7 + 9 * T = 37) → T = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_of_tacos_l539_53994


namespace NUMINAMATH_GPT_math_problem_l539_53959

noncomputable def problem : Real :=
  (2 * Real.sqrt 2 - 1) ^ 2 + (1 + Real.sqrt 5) * (1 - Real.sqrt 5)

theorem math_problem :
  problem = 5 - 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l539_53959


namespace NUMINAMATH_GPT_problem1_problem2_l539_53905

-- Problem 1: Remainder of 2011-digit number with each digit 2 when divided by 9 is 8

theorem problem1 : (4022 % 9 = 8) := by
  sorry

-- Problem 2: Remainder of n-digit number with each digit 7 when divided by 9 and n % 9 = 3 is 3

theorem problem2 (n : ℕ) (h : n % 9 = 3) : ((7 * n) % 9 = 3) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l539_53905


namespace NUMINAMATH_GPT_part1_solution_set_part2_values_of_a_l539_53992

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end NUMINAMATH_GPT_part1_solution_set_part2_values_of_a_l539_53992


namespace NUMINAMATH_GPT_amount_b_l539_53987

variable {a b : ℚ} -- a and b are rational numbers

theorem amount_b (h1 : a + b = 1210) (h2 : (4 / 15) * a = (2 / 5) * b) : b = 484 :=
sorry

end NUMINAMATH_GPT_amount_b_l539_53987


namespace NUMINAMATH_GPT_complex_right_triangle_l539_53917

open Complex

theorem complex_right_triangle {z1 z2 a b : ℂ}
  (h1 : z2 = I * z1)
  (h2 : z1 + z2 = -a)
  (h3 : z1 * z2 = b) :
  a^2 / b = 2 :=
by sorry

end NUMINAMATH_GPT_complex_right_triangle_l539_53917


namespace NUMINAMATH_GPT_sequence_ratio_l539_53914

theorem sequence_ratio :
  ∀ {a : ℕ → ℝ} (h₁ : a 1 = 1/2) (h₂ : ∀ n, a n = (a (n + 1)) * (a (n + 1))),
  (a 200 / a 300) = (301 / 201) :=
by
  sorry

end NUMINAMATH_GPT_sequence_ratio_l539_53914


namespace NUMINAMATH_GPT_august_first_problem_answer_l539_53988

theorem august_first_problem_answer (A : ℕ)
  (h1 : 2 * A = B)
  (h2 : 3 * A - 400 = C)
  (h3 : A + B + C = 3200) : A = 600 :=
sorry

end NUMINAMATH_GPT_august_first_problem_answer_l539_53988


namespace NUMINAMATH_GPT_greatest_perfect_power_sum_l539_53985

def sum_c_d_less_500 : ℕ :=
  let c := 22
  let d := 2
  c + d

theorem greatest_perfect_power_sum :
  ∃ c d : ℕ, 0 < c ∧ 1 < d ∧ c^d < 500 ∧
  ∀ x y : ℕ, 0 < x ∧ 1 < y ∧ x^y < 500 → x^y ≤ c^d ∧ (c + d = 24) :=
by
  sorry

end NUMINAMATH_GPT_greatest_perfect_power_sum_l539_53985


namespace NUMINAMATH_GPT_train_length_l539_53903

theorem train_length
  (train_speed_kmph : ℝ)
  (person_speed_kmph : ℝ)
  (time_seconds : ℝ)
  (h_train_speed : train_speed_kmph = 80)
  (h_person_speed : person_speed_kmph = 16)
  (h_time : time_seconds = 15)
  : (train_speed_kmph - person_speed_kmph) * (5/18) * time_seconds = 266.67 := 
by
  rw [h_train_speed, h_person_speed, h_time]
  norm_num
  sorry

end NUMINAMATH_GPT_train_length_l539_53903


namespace NUMINAMATH_GPT_solve_quadratic_eq_l539_53974

theorem solve_quadratic_eq {x : ℝ} :
  (x = 3 ∨ x = -1) ↔ x^2 - 2 * x - 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l539_53974


namespace NUMINAMATH_GPT_minimum_value_of_f_l539_53984

noncomputable def f (x : ℝ) : ℝ := x^2 + (1 / x^2) + (1 / (x^2 + 1 / x^2))

theorem minimum_value_of_f (x : ℝ) (hx : x > 0) : ∃ y : ℝ, y = f x ∧ y >= 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l539_53984


namespace NUMINAMATH_GPT_single_burger_cost_l539_53998

-- Conditions
def total_cost : ℝ := 74.50
def total_burgers : ℕ := 50
def cost_double_burger : ℝ := 1.50
def double_burgers : ℕ := 49

-- Derived information
def cost_single_burger : ℝ := total_cost - (double_burgers * cost_double_burger)

-- Theorem: Prove the cost of a single burger
theorem single_burger_cost : cost_single_burger = 1.00 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_single_burger_cost_l539_53998


namespace NUMINAMATH_GPT_ninth_number_l539_53919

theorem ninth_number (S1 S2 Total N : ℕ)
  (h1 : S1 = 9 * 56)
  (h2 : S2 = 9 * 63)
  (h3 : Total = 17 * 59)
  (h4 : Total = S1 + S2 - N) :
  N = 68 :=
by 
  -- The proof is omitted, only the statement is needed.
  sorry

end NUMINAMATH_GPT_ninth_number_l539_53919


namespace NUMINAMATH_GPT_range_of_a_l539_53939

variable (x a : ℝ)

def p (x : ℝ) := x^2 + x - 2 > 0
def q (x a : ℝ) := x > a

theorem range_of_a (h : ∀ x, q x a → p x)
  (h_not : ∃ x, ¬ q x a ∧ p x) : 1 ≤ a :=
sorry

end NUMINAMATH_GPT_range_of_a_l539_53939


namespace NUMINAMATH_GPT_candy_remainder_l539_53971

theorem candy_remainder :
  38759863 % 6 = 1 :=
by
  sorry

end NUMINAMATH_GPT_candy_remainder_l539_53971


namespace NUMINAMATH_GPT_maximize_perimeter_l539_53907

theorem maximize_perimeter 
  (l : ℝ) (c_f : ℝ) (C : ℝ) (b : ℝ)
  (hl: l = 400) (hcf: c_f = 5) (hC: C = 1500) :
  ∃ (y : ℝ), y = 180 :=
by
  sorry

end NUMINAMATH_GPT_maximize_perimeter_l539_53907


namespace NUMINAMATH_GPT_seventh_term_of_arithmetic_sequence_l539_53995

theorem seventh_term_of_arithmetic_sequence (a d : ℤ) 
  (h1 : 5 * a + 10 * d = 15) 
  (h2 : a + 5 * d = 6) : 
  a + 6 * d = 7 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_seventh_term_of_arithmetic_sequence_l539_53995


namespace NUMINAMATH_GPT_problem_l539_53933

def T := {n : ℤ | ∃ (k : ℤ), n = 4 * (2*k + 1)^2 + 13}

theorem problem :
  (∀ n ∈ T, ¬ 2 ∣ n) ∧ (∀ n ∈ T, ¬ 5 ∣ n) :=
by
  sorry

end NUMINAMATH_GPT_problem_l539_53933


namespace NUMINAMATH_GPT_sequence_value_2016_l539_53901

theorem sequence_value_2016 :
  ∀ (a : ℕ → ℤ),
    a 1 = 3 →
    a 2 = 6 →
    (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) →
    a 2016 = -3 :=
by
  sorry

end NUMINAMATH_GPT_sequence_value_2016_l539_53901


namespace NUMINAMATH_GPT_complex_ratio_of_cubes_l539_53952

theorem complex_ratio_of_cubes (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 10) (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 8 :=
by
  sorry

end NUMINAMATH_GPT_complex_ratio_of_cubes_l539_53952


namespace NUMINAMATH_GPT_divisible_iff_l539_53938

theorem divisible_iff (m n k : ℕ) (h : m > n) : 
  (3^(k+1)) ∣ (4^m - 4^n) ↔ (3^k) ∣ (m - n) := 
sorry

end NUMINAMATH_GPT_divisible_iff_l539_53938


namespace NUMINAMATH_GPT_physicist_imons_no_entanglement_l539_53940

theorem physicist_imons_no_entanglement (G : SimpleGraph V) :
  (∃ ops : ℕ, ∀ v₁ v₂ : V, ¬G.Adj v₁ v₂) :=
by
  sorry

end NUMINAMATH_GPT_physicist_imons_no_entanglement_l539_53940


namespace NUMINAMATH_GPT_pancake_cut_l539_53947

theorem pancake_cut (n : ℕ) (h : 3 ≤ n) :
  ∃ (cut_piece : ℝ), cut_piece > 0 :=
sorry

end NUMINAMATH_GPT_pancake_cut_l539_53947


namespace NUMINAMATH_GPT_ball_distribution_l539_53955

theorem ball_distribution (balls boxes : ℕ) (hballs : balls = 7) (hboxes : boxes = 4) :
  (∃ (ways : ℕ), ways = (Nat.choose (balls - 1) (boxes - 1)) ∧ ways = 20) :=
by
  sorry

end NUMINAMATH_GPT_ball_distribution_l539_53955


namespace NUMINAMATH_GPT_count_lattice_points_on_hyperbola_l539_53991

theorem count_lattice_points_on_hyperbola : 
  (∃ (S : Finset (ℤ × ℤ)), (∀ (p : ℤ × ℤ), p ∈ S ↔ (p.1 ^ 2 - p.2 ^ 2 = 1800 ^ 2)) ∧ S.card = 150) :=
sorry

end NUMINAMATH_GPT_count_lattice_points_on_hyperbola_l539_53991


namespace NUMINAMATH_GPT_train_speed_l539_53900

theorem train_speed (time_seconds : ℕ) (length_meters : ℕ) (speed_kmph : ℕ)
  (h1 : time_seconds = 9) (h2 : length_meters = 135) : speed_kmph = 54 :=
sorry

end NUMINAMATH_GPT_train_speed_l539_53900


namespace NUMINAMATH_GPT_chinese_team_wins_gold_l539_53920

noncomputable def prob_player_a_wins : ℚ := 3 / 7
noncomputable def prob_player_b_wins : ℚ := 1 / 4

theorem chinese_team_wins_gold : prob_player_a_wins + prob_player_b_wins = 19 / 28 := by
  sorry

end NUMINAMATH_GPT_chinese_team_wins_gold_l539_53920


namespace NUMINAMATH_GPT_finitely_many_negative_terms_l539_53911

theorem finitely_many_negative_terms (A : ℝ) :
  (∀ (x : ℕ → ℝ), (∀ n, x n ≠ 0) ∧ (∀ n, x (n+1) = A - 1 / x n) →
  (∃ N, ∀ n ≥ N, x n ≥ 0)) ↔ A ≥ 2 :=
sorry

end NUMINAMATH_GPT_finitely_many_negative_terms_l539_53911


namespace NUMINAMATH_GPT_krikor_speed_increase_l539_53931

/--
Krikor traveled to work on two consecutive days, Monday and Tuesday, at different speeds.
Both days, he covered the same distance. On Monday, he traveled for 0.5 hours, and on
Tuesday, he traveled for \( \frac{5}{12} \) hours. Prove that the percentage increase in his speed 
from Monday to Tuesday is 20%.
-/
theorem krikor_speed_increase :
  ∀ (v1 v2 : ℝ), (0.5 * v1 = (5 / 12) * v2) → (v2 = (6 / 5) * v1) → 
  ((v2 - v1) / v1 * 100 = 20) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_krikor_speed_increase_l539_53931


namespace NUMINAMATH_GPT_nina_spends_70_l539_53969

-- Definitions of the quantities and prices
def toys := 3
def toy_price := 10
def basketball_cards := 2
def card_price := 5
def shirts := 5
def shirt_price := 6

-- Calculate the total amount spent
def total_spent := (toys * toy_price) + (basketball_cards * card_price) + (shirts * shirt_price)

-- Problem statement: Prove that the total amount spent is $70
theorem nina_spends_70 : total_spent = 70 := by
  sorry

end NUMINAMATH_GPT_nina_spends_70_l539_53969


namespace NUMINAMATH_GPT_correct_operation_l539_53953

theorem correct_operation (a : ℝ) : 2 * a^3 / a^2 = 2 * a := 
sorry

end NUMINAMATH_GPT_correct_operation_l539_53953


namespace NUMINAMATH_GPT_cars_per_day_l539_53921

noncomputable def paul_rate : ℝ := 2
noncomputable def jack_rate : ℝ := 3
noncomputable def paul_jack_rate : ℝ := paul_rate + jack_rate
noncomputable def hours_per_day : ℝ := 8
noncomputable def total_cars : ℝ := paul_jack_rate * hours_per_day

theorem cars_per_day : total_cars = 40 := by
  sorry

end NUMINAMATH_GPT_cars_per_day_l539_53921


namespace NUMINAMATH_GPT_overall_profit_or_loss_l539_53948

def price_USD_to_INR(price_usd : ℝ) : ℝ := price_usd * 75
def price_EUR_to_INR(price_eur : ℝ) : ℝ := price_eur * 80
def price_GBP_to_INR(price_gbp : ℝ) : ℝ := price_gbp * 100
def price_JPY_to_INR(price_jpy : ℝ) : ℝ := price_jpy * 0.7

def CP_grinder : ℝ := price_USD_to_INR (150 + 0.1 * 150)
def SP_grinder : ℝ := price_USD_to_INR (165 - 0.04 * 165)

def CP_mobile_phone : ℝ := price_EUR_to_INR ((100 - 0.05 * 100) + 0.15 * (100 - 0.05 * 100))
def SP_mobile_phone : ℝ := price_EUR_to_INR ((109.25 : ℝ) + 0.1 * 109.25)

def CP_laptop : ℝ := price_GBP_to_INR (200 + 0.08 * 200)
def SP_laptop : ℝ := price_GBP_to_INR (216 - 0.08 * 216)

def CP_camera : ℝ := price_JPY_to_INR ((12000 - 0.12 * 12000) + 0.05 * (12000 - 0.12 * 12000))
def SP_camera : ℝ := price_JPY_to_INR (11088 + 0.15 * 11088)

def total_CP : ℝ := CP_grinder + CP_mobile_phone + CP_laptop + CP_camera
def total_SP : ℝ := SP_grinder + SP_mobile_phone + SP_laptop + SP_camera

theorem overall_profit_or_loss :
  (total_SP - total_CP) = -184.76 := 
sorry

end NUMINAMATH_GPT_overall_profit_or_loss_l539_53948


namespace NUMINAMATH_GPT_segment_distance_sum_l539_53954

theorem segment_distance_sum
  (AB_len : ℝ) (A'B'_len : ℝ) (D_midpoint : AB_len / 2 = 4)
  (D'_midpoint : A'B'_len / 2 = 6) (x : ℝ) (y : ℝ)
  (x_val : x = 3) :
  x + y = 10 :=
by sorry

end NUMINAMATH_GPT_segment_distance_sum_l539_53954


namespace NUMINAMATH_GPT_marcus_savings_l539_53966

def MarcusMaxPrice : ℝ := 130
def ShoeInitialPrice : ℝ := 120
def DiscountPercentage : ℝ := 0.30
def FinalPrice : ℝ := ShoeInitialPrice - (DiscountPercentage * ShoeInitialPrice)
def Savings : ℝ := MarcusMaxPrice - FinalPrice

theorem marcus_savings : Savings = 46 := by
  sorry

end NUMINAMATH_GPT_marcus_savings_l539_53966


namespace NUMINAMATH_GPT_final_score_correct_l539_53970

def innovation_score : ℕ := 88
def comprehensive_score : ℕ := 80
def language_score : ℕ := 75

def weight_innovation : ℕ := 5
def weight_comprehensive : ℕ := 3
def weight_language : ℕ := 2

def final_score : ℕ :=
  (innovation_score * weight_innovation + comprehensive_score * weight_comprehensive +
   language_score * weight_language) /
  (weight_innovation + weight_comprehensive + weight_language)

theorem final_score_correct :
  final_score = 83 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_final_score_correct_l539_53970


namespace NUMINAMATH_GPT_find_n_l539_53922

theorem find_n (n : ℕ) (d : ℕ) (h_pos : n > 0) (h_digit : d < 10) (h_equiv : n * 999 = 810 * (100 * d + 25)) : n = 750 :=
  sorry

end NUMINAMATH_GPT_find_n_l539_53922


namespace NUMINAMATH_GPT_azalea_wool_price_l539_53983

noncomputable def sheep_count : ℕ := 200
noncomputable def wool_per_sheep : ℕ := 10
noncomputable def shearing_cost : ℝ := 2000
noncomputable def profit : ℝ := 38000

-- Defining total wool and total revenue based on these definitions
noncomputable def total_wool : ℕ := sheep_count * wool_per_sheep
noncomputable def total_revenue : ℝ := profit + shearing_cost
noncomputable def price_per_pound : ℝ := total_revenue / total_wool

-- Problem statement: Proving that the price per pound of wool is equal to $20
theorem azalea_wool_price :
  price_per_pound = 20 := 
sorry

end NUMINAMATH_GPT_azalea_wool_price_l539_53983


namespace NUMINAMATH_GPT_Albert_has_more_rocks_than_Jose_l539_53932

noncomputable def Joshua_rocks : ℕ := 80
noncomputable def Jose_rocks : ℕ := Joshua_rocks - 14
noncomputable def Albert_rocks : ℕ := Joshua_rocks + 6

theorem Albert_has_more_rocks_than_Jose :
  Albert_rocks - Jose_rocks = 20 := by
  sorry

end NUMINAMATH_GPT_Albert_has_more_rocks_than_Jose_l539_53932


namespace NUMINAMATH_GPT_min_value_expression_l539_53963

theorem min_value_expression : ∃ x : ℝ, x = 300 ∧ ∀ y : ℝ, (y^2 - 600*y + 369) ≥ (300^2 - 600*300 + 369) := by
  use 300
  sorry

end NUMINAMATH_GPT_min_value_expression_l539_53963


namespace NUMINAMATH_GPT_two_times_sum_of_fourth_power_is_perfect_square_l539_53956

theorem two_times_sum_of_fourth_power_is_perfect_square (a b c : ℤ) 
  (h : a + b + c = 0) : 2 * (a^4 + b^4 + c^4) = (a^2 + b^2 + c^2)^2 := 
by sorry

end NUMINAMATH_GPT_two_times_sum_of_fourth_power_is_perfect_square_l539_53956


namespace NUMINAMATH_GPT_cafeteria_extra_fruits_l539_53982

def red_apples_ordered : ℕ := 43
def green_apples_ordered : ℕ := 32
def oranges_ordered : ℕ := 25
def red_apples_chosen : ℕ := 7
def green_apples_chosen : ℕ := 5
def oranges_chosen : ℕ := 4

def extra_red_apples : ℕ := red_apples_ordered - red_apples_chosen
def extra_green_apples : ℕ := green_apples_ordered - green_apples_chosen
def extra_oranges : ℕ := oranges_ordered - oranges_chosen

def total_extra_fruits : ℕ := extra_red_apples + extra_green_apples + extra_oranges

theorem cafeteria_extra_fruits : total_extra_fruits = 84 := by
  sorry

end NUMINAMATH_GPT_cafeteria_extra_fruits_l539_53982


namespace NUMINAMATH_GPT_my_problem_l539_53943

-- Definitions and conditions from the problem statement
variables (p q r u v w : ℝ)

-- Conditions
axiom h1 : 17 * u + q * v + r * w = 0
axiom h2 : p * u + 29 * v + r * w = 0
axiom h3 : p * u + q * v + 56 * w = 0
axiom h4 : p ≠ 17
axiom h5 : u ≠ 0

-- Problem statement to prove
theorem my_problem : (p / (p - 17)) + (q / (q - 29)) + (r / (r - 56)) = 0 :=
sorry

end NUMINAMATH_GPT_my_problem_l539_53943


namespace NUMINAMATH_GPT_remainder_calculation_l539_53934

theorem remainder_calculation :
  ((2367 * 1023) % 500) = 41 := by
  sorry

end NUMINAMATH_GPT_remainder_calculation_l539_53934


namespace NUMINAMATH_GPT_smallest_p_l539_53902

theorem smallest_p (n p : ℕ) (h1 : n % 2 = 1) (h2 : n % 7 = 5) (h3 : (n + p) % 10 = 0) : p = 1 := 
sorry

end NUMINAMATH_GPT_smallest_p_l539_53902


namespace NUMINAMATH_GPT_product_of_numbers_is_178_5_l539_53912

variables (a b c d : ℚ)

def sum_eq_36 := a + b + c + d = 36
def first_num_cond := a = 3 * (b + c + d)
def second_num_cond := b = 5 * c
def fourth_num_cond := d = (1 / 2) * c

theorem product_of_numbers_is_178_5 (h1 : sum_eq_36 a b c d)
  (h2 : first_num_cond a b c d) (h3 : second_num_cond b c) (h4 : fourth_num_cond d c) :
  a * b * c * d = 178.5 :=
by
  sorry

end NUMINAMATH_GPT_product_of_numbers_is_178_5_l539_53912


namespace NUMINAMATH_GPT_larger_number_of_two_l539_53961

theorem larger_number_of_two (x y : ℝ) (h1 : x * y = 30) (h2 : x + y = 13) : max x y = 10 :=
sorry

end NUMINAMATH_GPT_larger_number_of_two_l539_53961


namespace NUMINAMATH_GPT_find_tan_alpha_plus_pi_div_12_l539_53973

theorem find_tan_alpha_plus_pi_div_12 (α : ℝ) (h : Real.sin α = 3 * Real.sin (α + Real.pi / 6)) :
  Real.tan (α + Real.pi / 12) = 2 * Real.sqrt 3 - 4 :=
by
  sorry

end NUMINAMATH_GPT_find_tan_alpha_plus_pi_div_12_l539_53973


namespace NUMINAMATH_GPT_tens_digit_of_expression_l539_53928

theorem tens_digit_of_expression :
  (2023 ^ 2024 - 2025) % 100 / 10 % 10 = 1 :=
by sorry

end NUMINAMATH_GPT_tens_digit_of_expression_l539_53928


namespace NUMINAMATH_GPT_range_of_m_l539_53978

theorem range_of_m :
  (∀ x : ℝ, (x > 0) → (x^2 - m * x + 4 ≥ 0)) ∧ (¬∃ x : ℝ, (x^2 - 2 * m * x + 7 * m - 10 = 0)) ↔ (2 < m ∧ m ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l539_53978


namespace NUMINAMATH_GPT_find_angle_l539_53957

def complementary (x : ℝ) := 90 - x
def supplementary (x : ℝ) := 180 - x

theorem find_angle (x : ℝ) (h : supplementary x = 3 * complementary x) : x = 45 :=
by 
  sorry

end NUMINAMATH_GPT_find_angle_l539_53957


namespace NUMINAMATH_GPT_arithmetic_sequence_seventh_term_l539_53936

theorem arithmetic_sequence_seventh_term (a d : ℝ) 
  (h1 : a + (a + d) + (a + 2 * d) + (a + 3 * d) = 14) 
  (h2 : a + 4 * d = 9) : 
  a + 6 * d = 13.4 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_seventh_term_l539_53936


namespace NUMINAMATH_GPT_problem_l539_53997

theorem problem (p q r : ℝ) (h1 : p + q + r = 5000) (h2 : r = (2 / 3) * (p + q)) : r = 2000 :=
by
  sorry

end NUMINAMATH_GPT_problem_l539_53997


namespace NUMINAMATH_GPT_value_of_A_l539_53951

def random_value (c : Char) : ℤ := sorry

-- Given conditions
axiom H_value : random_value 'H' = 12
axiom MATH_value : random_value 'M' + random_value 'A' + random_value 'T' + random_value 'H' = 40
axiom TEAM_value : random_value 'T' + random_value 'E' + random_value 'A' + random_value 'M' = 50
axiom MEET_value : random_value 'M' + random_value 'E' + random_value 'E' + random_value 'T' = 44

-- Prove that A = 28
theorem value_of_A : random_value 'A' = 28 := by
  sorry

end NUMINAMATH_GPT_value_of_A_l539_53951


namespace NUMINAMATH_GPT_cows_gift_by_friend_l539_53986

-- Define the base conditions
def initial_cows : Nat := 39
def cows_died : Nat := 25
def cows_sold : Nat := 6
def cows_increase : Nat := 24
def cows_bought : Nat := 43
def final_cows : Nat := 83

-- Define the computation to get the number of cows after each event
def cows_after_died : Nat := initial_cows - cows_died
def cows_after_sold : Nat := cows_after_died - cows_sold
def cows_after_increase : Nat := cows_after_sold + cows_increase
def cows_after_bought : Nat := cows_after_increase + cows_bought

-- Define the proof problem
theorem cows_gift_by_friend : (final_cows - cows_after_bought) = 8 := by
  sorry

end NUMINAMATH_GPT_cows_gift_by_friend_l539_53986


namespace NUMINAMATH_GPT_pure_imaginary_z_squared_l539_53968

-- Formalization in Lean 4
theorem pure_imaginary_z_squared (a : ℝ) (h : a + (1 + a) * I = (1 + a) * I) : (a + (1 + a) * I)^2 = -1 :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_z_squared_l539_53968


namespace NUMINAMATH_GPT_largest_int_less_100_remainder_5_l539_53908

theorem largest_int_less_100_remainder_5 (a : ℕ) (h1 : a < 100) (h2 : a % 9 = 5) :
  a = 95 :=
sorry

end NUMINAMATH_GPT_largest_int_less_100_remainder_5_l539_53908
