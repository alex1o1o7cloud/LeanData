import Mathlib

namespace Patrick_hours_less_than_twice_Greg_l131_131438

def J := 18
def G := J - 6
def total_hours := 50
def P : ℕ := sorry -- To be defined, we need to establish the proof later with the condition J + G + P = 50
def X : ℕ := sorry -- To be defined, we need to establish the proof later with the condition P = 2 * G - X

theorem Patrick_hours_less_than_twice_Greg : X = 4 := by
  -- Placeholder definitions for P and X based on the given conditions
  let P := total_hours - (J + G)
  let X := 2 * G - P
  sorry -- Proof details to be filled in

end Patrick_hours_less_than_twice_Greg_l131_131438


namespace max_value_inequality_l131_131120

theorem max_value_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 3) :
  (x^2 + x * y + y^2) * (y^2 + y * z + z^2) * (z^2 + z * x + x^2) ≤ 27 := 
sorry

end max_value_inequality_l131_131120


namespace find_k_l131_131949

theorem find_k (k : ℝ) (h : (2 * (7:ℝ)^2) + 3 * 7 - k = 0) : k = 119 := by
  sorry

end find_k_l131_131949


namespace number_of_boys_at_reunion_l131_131678

theorem number_of_boys_at_reunion (n : ℕ) (h : (n * (n - 1)) / 2 = 21) : n = 7 :=
by
  sorry

end number_of_boys_at_reunion_l131_131678


namespace find_value_of_expression_l131_131794

variables (a b : ℝ)

-- Given the condition that 2a - 3b = 5, prove that 2a - 3b + 3 = 8.
theorem find_value_of_expression
  (h : 2 * a - 3 * b = 5) : 2 * a - 3 * b + 3 = 8 :=
by sorry

end find_value_of_expression_l131_131794


namespace marciaHairLengthProof_l131_131764

noncomputable def marciaHairLengthAtEndOfSchoolYear : Float :=
  let L0 := 24.0                           -- initial length
  let L1 := L0 - 0.3 * L0                  -- length after September cut
  let L2 := L1 + 3.0 * 1.5                 -- length after three months of growth (Sept - Dec)
  let L3 := L2 - 0.2 * L2                  -- length after January cut
  let L4 := L3 + 5.0 * 1.8                 -- length after five months of growth (Jan - May)
  let L5 := L4 - 4.0                       -- length after June cut
  L5

theorem marciaHairLengthProof : marciaHairLengthAtEndOfSchoolYear = 22.04 :=
by
  sorry

end marciaHairLengthProof_l131_131764


namespace min_x1_x2_squared_l131_131470

theorem min_x1_x2_squared (x1 x2 m : ℝ) (hm : (m + 3)^2 ≥ 0) 
  (h_sum : x1 + x2 = -(m + 1)) 
  (h_prod : x1 * x2 = 2 * m - 2) : 
  (x1^2 + x2^2 = (m - 1)^2 + 4) ∧ ∃ m, m = 1 → x1^2 + x2^2 = 4 :=
by {
  sorry
}

end min_x1_x2_squared_l131_131470


namespace mathieu_plot_area_l131_131706

def total_area (x y : ℕ) : ℕ := x * x

theorem mathieu_plot_area :
  ∃ (x y : ℕ), (x^2 - y^2 = 464) ∧ (x - y = 8) ∧ (total_area x y = 1089) :=
by sorry

end mathieu_plot_area_l131_131706


namespace wheel_revolutions_l131_131154

theorem wheel_revolutions (x y : ℕ) (h1 : y = x + 300)
  (h2 : 10 / (x : ℝ) = 10 / (y : ℝ) + 1 / 60) : 
  x = 300 ∧ y = 600 := 
by sorry

end wheel_revolutions_l131_131154


namespace exist_positive_real_x_l131_131524

theorem exist_positive_real_x (x : ℝ) (hx1 : 0 < x) (hx2 : Nat.floor x * x = 90) : x = 10 := 
sorry

end exist_positive_real_x_l131_131524


namespace no_hexagon_cross_section_l131_131205

-- Define the shape of the cross-section resulting from cutting a triangular prism with a plane
inductive Shape
| triangle
| quadrilateral
| pentagon
| hexagon

-- Define the condition of cutting a triangular prism
structure TriangularPrism where
  cut : Shape

-- The theorem stating that cutting a triangular prism with a plane cannot result in a hexagon
theorem no_hexagon_cross_section (P : TriangularPrism) : P.cut ≠ Shape.hexagon :=
by
  sorry

end no_hexagon_cross_section_l131_131205


namespace rowed_upstream_distance_l131_131637

def distance_downstream := 120
def time_downstream := 2
def distance_upstream := 2
def speed_stream := 15

def speed_boat (V_b : ℝ) := V_b

theorem rowed_upstream_distance (V_b : ℝ) (D_u : ℝ) :
  (distance_downstream = (V_b + speed_stream) * time_downstream) ∧
  (D_u = (V_b - speed_stream) * time_upstream) →
  D_u = 60 :=
by 
  sorry

end rowed_upstream_distance_l131_131637


namespace cordelia_bleach_time_l131_131509

theorem cordelia_bleach_time
    (H : ℕ)
    (total_time : H + 2 * H = 9) :
    H = 3 :=
by
  sorry

end cordelia_bleach_time_l131_131509


namespace solution_set_for_composed_function_l131_131534

theorem solution_set_for_composed_function :
  ∀ x : ℝ, (∀ y : ℝ, y = 2 * x - 1 → (2 * y - 1) ≥ 1) ↔ x ≥ 1 := by
  sorry

end solution_set_for_composed_function_l131_131534


namespace fraction_of_hidden_sea_is_five_over_eight_l131_131664

noncomputable def cloud_fraction := 1 / 2
noncomputable def island_uncovered_fraction := 1 / 4 
noncomputable def island_covered_fraction := island_uncovered_fraction / (1 - cloud_fraction)

-- The total island area is the sum of covered and uncovered.
noncomputable def total_island_fraction := island_uncovered_fraction + island_covered_fraction 

-- The sea area covered by the cloud is half minus the fraction of the island covered by the cloud.
noncomputable def sea_covered_by_cloud := cloud_fraction - island_covered_fraction 

-- The sea occupies the remainder of the landscape not taken by the uncoveed island.
noncomputable def total_sea_fraction := 1 - island_uncovered_fraction - cloud_fraction + island_covered_fraction 

-- The sea fraction visible and not covered by clouds
noncomputable def sea_visible_not_covered := total_sea_fraction - sea_covered_by_cloud 

-- The fraction of the sea hidden by the cloud
noncomputable def sea_fraction_hidden_by_cloud := sea_covered_by_cloud / total_sea_fraction 

theorem fraction_of_hidden_sea_is_five_over_eight : sea_fraction_hidden_by_cloud = 5 / 8 := 
by
  sorry

end fraction_of_hidden_sea_is_five_over_eight_l131_131664


namespace total_students_l131_131323

-- Define the conditions
def ratio_boys_to_girls (boys girls : ℕ) : Prop := boys = 3 * (girls / 2)
def boys_girls_difference (boys girls : ℕ) : Prop := boys = girls + 20

-- Define the property to be proved
theorem total_students (boys girls : ℕ) 
  (h1 : ratio_boys_to_girls boys girls)
  (h2 : boys_girls_difference boys girls) :
  boys + girls = 100 :=
sorry

end total_students_l131_131323


namespace stephen_speed_l131_131993

theorem stephen_speed (v : ℝ) 
  (time : ℝ := 0.25)
  (speed_second_third : ℝ := 12)
  (speed_last_third : ℝ := 20)
  (total_distance : ℝ := 12) :
  (v * time + speed_second_third * time + speed_last_third * time = total_distance) → 
  v = 16 :=
by
  intro h
  -- introducing the condition h: v * 0.25 + 3 + 5 = 12
  sorry

end stephen_speed_l131_131993


namespace discount_percentage_l131_131759

theorem discount_percentage (cp mp pm : ℤ) (x : ℤ) 
    (Hcp : cp = 160) 
    (Hmp : mp = 240) 
    (Hpm : pm = 20) 
    (Hcondition : mp * (100 - x) = cp * (100 + pm)) : 
  x = 20 := 
  sorry

end discount_percentage_l131_131759


namespace find_units_digit_of_n_l131_131783

-- Define the problem conditions
def units_digit (a : ℕ) : ℕ := a % 10

theorem find_units_digit_of_n (m n : ℕ) (h1 : units_digit m = 3) (h2 : units_digit (m * n) = 6) (h3 : units_digit (14^8) = 6) :
  units_digit n = 2 :=
  sorry

end find_units_digit_of_n_l131_131783


namespace part1_part2_l131_131409

theorem part1 (m : ℝ) : 
  (∀ x y : ℝ, (x^2 + y^2 - 2 * x + 4 * y - 4 = 0 ∧ y = x + m) → -3 - 3 * Real.sqrt 2 < m ∧ m < -3 + 3 * Real.sqrt 2) :=
sorry

theorem part2 (m x1 x2 y1 y2 : ℝ) (h1 : x1 + x2 = -(m + 1)) (h2 : x1 * x2 = (m^2 + 4 * m - 4) / 2) 
(h3 : (x - x1) * (x - x2) + (x1 + m) * (x2 + m) = 0) : 
  m = -4 ∨ m = 1 →
  (∀ x y : ℝ, y = x + m ↔ x - y - 4 = 0 ∨ x - y + 1 = 0) :=
sorry

end part1_part2_l131_131409


namespace original_cookies_l131_131923

noncomputable def initial_cookies (final_cookies : ℝ) (ratio : ℝ) (days : ℕ) : ℝ :=
  final_cookies / ratio^days

theorem original_cookies :
  ∀ (final_cookies : ℝ) (ratio : ℝ) (days : ℕ),
  final_cookies = 28 →
  ratio = 0.7 →
  days = 3 →
  initial_cookies final_cookies ratio days = 82 :=
by
  intros final_cookies ratio days h_final h_ratio h_days
  rw [initial_cookies, h_final, h_ratio, h_days]
  norm_num
  sorry

end original_cookies_l131_131923


namespace triangle_area_ratio_l131_131039

theorem triangle_area_ratio (a n m : ℕ) (h1 : 0 < a) (h2 : 0 < n) (h3 : 0 < m) :
  let area_A := (a^2 : ℝ) / (4 * n^2)
  let area_B := (a^2 : ℝ) / (4 * m^2)
  (area_A / area_B) = (m^2 : ℝ) / (n^2 : ℝ) :=
by
  sorry

end triangle_area_ratio_l131_131039


namespace transformed_roots_polynomial_l131_131700

-- Given conditions
variables {a b c : ℝ}
variables (h : ∀ x, (x - a) * (x - b) * (x - c) = x^3 - 4 * x + 6)

-- Prove the equivalent polynomial with the transformed roots
theorem transformed_roots_polynomial :
  (∀ x, (x - (a - 3)) * (x - (b - 3)) * (x - (c - 3)) = x^3 + 9 * x^2 + 23 * x + 21) :=
sorry

end transformed_roots_polynomial_l131_131700


namespace divide_square_into_smaller_squares_l131_131849

def P (n : ℕ) : Prop := sorry /- Define the property of dividing a square into n smaller squares -/

theorem divide_square_into_smaller_squares (n : ℕ) (h : n > 5) : P n :=
  sorry

end divide_square_into_smaller_squares_l131_131849


namespace cordelia_bleach_time_l131_131510

theorem cordelia_bleach_time
    (H : ℕ)
    (total_time : H + 2 * H = 9) :
    H = 3 :=
by
  sorry

end cordelia_bleach_time_l131_131510


namespace quadratic_equation_solutions_l131_131321

theorem quadratic_equation_solutions :
  ∀ x : ℝ, x^2 + 7 * x = 0 ↔ (x = 0 ∨ x = -7) := 
by 
  intro x
  sorry

end quadratic_equation_solutions_l131_131321


namespace number_of_roof_tiles_l131_131031

def land_cost : ℝ := 50
def bricks_cost_per_1000 : ℝ := 100
def roof_tile_cost : ℝ := 10
def land_required : ℝ := 2000
def bricks_required : ℝ := 10000
def total_construction_cost : ℝ := 106000

theorem number_of_roof_tiles :
  let land_total := land_cost * land_required
  let bricks_total := (bricks_required / 1000) * bricks_cost_per_1000
  let remaining_cost := total_construction_cost - (land_total + bricks_total)
  let roof_tiles := remaining_cost / roof_tile_cost
  roof_tiles = 500 := by
  sorry

end number_of_roof_tiles_l131_131031


namespace determine_sum_of_digits_l131_131413

theorem determine_sum_of_digits (x y : ℕ) (hx : x < 10) (hy : y < 10)
  (h : ∃ a b c d : ℕ, 
       a = 30 + x ∧ b = 10 * y + 4 ∧
       c = (a * (b % 10)) % 100 ∧ 
       d = (a * (b % 10)) / 100 ∧ 
       10 * d + c = 156) :
  x + y = 13 :=
by
  sorry

end determine_sum_of_digits_l131_131413


namespace average_age_before_new_students_joined_l131_131717

theorem average_age_before_new_students_joined 
  (A : ℝ) 
  (N : ℕ) 
  (new_students_average_age : ℝ) 
  (average_age_drop : ℝ) 
  (original_class_strength : ℕ)
  (hN : N = 17) 
  (h_new_students : new_students_average_age = 32)
  (h_age_drop : average_age_drop = 4)
  (h_strength : original_class_strength = 17)
  (h_equation : 17 * A + 17 * new_students_average_age = (2 * original_class_strength) * (A - average_age_drop)) :
  A = 40 :=
by sorry

end average_age_before_new_students_joined_l131_131717


namespace girls_tried_out_l131_131609

-- Definitions for conditions
def boys_trying_out : ℕ := 4
def students_called_back : ℕ := 26
def students_did_not_make_cut : ℕ := 17

-- Definition to calculate total students who tried out
def total_students_who_tried_out : ℕ := students_called_back + students_did_not_make_cut

-- Proof statement
theorem girls_tried_out : ∀ (G : ℕ), G + boys_trying_out = total_students_who_tried_out → G = 39 :=
by
  intro G
  intro h
  rw [total_students_who_tried_out, boys_trying_out] at h
  sorry

end girls_tried_out_l131_131609


namespace geometric_sequence_increasing_l131_131294

theorem geometric_sequence_increasing {a : ℕ → ℝ} (r : ℝ) (h_pos : 0 < r) (h_geometric : ∀ n, a (n + 1) = r * a n) :
  (a 0 < a 1 ∧ a 1 < a 2) ↔ ∀ n m, n < m → a n < a m :=
by sorry

end geometric_sequence_increasing_l131_131294


namespace find_x_l131_131521

-- Define that x is a real number and positive
variable (x : ℝ)
variable (hx_pos : 0 < x)

-- Define the floor function and the main equation
variable (hx_eq : ⌊x⌋ * x = 90)

theorem find_x (h : ⌊x⌋ * x = 90) (hx_pos : 0 < x) : ⌊x⌋ = 9 ∧ x = 10 :=
by
  sorry

end find_x_l131_131521


namespace seating_arrangements_l131_131822

-- Number of ways to arrange a block of n items
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Groups
def dodgers : ℕ := 4
def marlins : ℕ := 3
def phillies : ℕ := 2

-- Total number of players
def total_players : ℕ := dodgers + marlins + phillies

-- Number of ways to arrange the blocks
def blocks_arrangements : ℕ := factorial 3

-- Internal arrangements within each block
def dodgers_arrangements : ℕ := factorial dodgers
def marlins_arrangements : ℕ := factorial marlins
def phillies_arrangements : ℕ := factorial phillies

-- Total number of ways to seat the players
def total_arrangements : ℕ :=
  blocks_arrangements * dodgers_arrangements * marlins_arrangements * phillies_arrangements

-- Prove that the total arrangements is 1728
theorem seating_arrangements : total_arrangements = 1728 := by
  sorry

end seating_arrangements_l131_131822


namespace polygon_quadrilateral_l131_131817

theorem polygon_quadrilateral {n : ℕ} (h : (n - 2) * 180 = 360) : n = 4 := by
  sorry

end polygon_quadrilateral_l131_131817


namespace number_of_lattice_points_l131_131708

theorem number_of_lattice_points (A B : ℝ) (h : B - A = 10) :
  ∃ n, n = 10 ∨ n = 11 :=
sorry

end number_of_lattice_points_l131_131708


namespace fg_of_2_l131_131408

-- Define the functions f and g
def f (x : ℝ) : ℝ := 5 - 4 * x
def g (x : ℝ) : ℝ := x^2 + 2

-- Prove the specific property
theorem fg_of_2 : f (g 2) = -19 := by
  -- Placeholder for the proof
  sorry

end fg_of_2_l131_131408


namespace total_sections_l131_131732

theorem total_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 288) : 
  (boys / Nat.gcd boys girls) + (girls / Nat.gcd boys girls) = 29 :=
by
  sorry

end total_sections_l131_131732


namespace cordelia_bleach_time_l131_131508

theorem cordelia_bleach_time
    (H : ℕ)
    (total_time : H + 2 * H = 9) :
    H = 3 :=
by
  sorry

end cordelia_bleach_time_l131_131508


namespace mosquito_feedings_to_death_l131_131175

theorem mosquito_feedings_to_death 
  (drops_per_feeding : ℕ := 20) 
  (drops_per_liter : ℕ := 5000) 
  (lethal_blood_loss_liters : ℝ := 3) 
  (drops_per_feeding_liters : ℝ := drops_per_feeding / drops_per_liter) 
  (lethal_feedings : ℝ := lethal_blood_loss_liters / drops_per_feeding_liters) :
  lethal_feedings = 750 := 
by
  sorry

end mosquito_feedings_to_death_l131_131175


namespace arithmetic_sequence_term_number_l131_131747

-- Given:
def first_term : ℕ := 1
def common_difference : ℕ := 3
def target_term : ℕ := 2011

-- To prove:
theorem arithmetic_sequence_term_number :
    ∃ n : ℕ, target_term = first_term + (n - 1) * common_difference ∧ n = 671 := 
by
  -- The proof is omitted
  sorry

end arithmetic_sequence_term_number_l131_131747


namespace crackers_eaten_by_Daniel_and_Elsie_l131_131300

theorem crackers_eaten_by_Daniel_and_Elsie :
  ∀ (initial_crackers remaining_crackers eaten_by_Ally eaten_by_Bob eaten_by_Clair: ℝ),
    initial_crackers = 27.5 →
    remaining_crackers = 10.5 →
    eaten_by_Ally = 3.5 →
    eaten_by_Bob = 4.0 →
    eaten_by_Clair = 5.5 →
    initial_crackers - remaining_crackers = (eaten_by_Ally + eaten_by_Bob + eaten_by_Clair) + (4 : ℝ) :=
by sorry

end crackers_eaten_by_Daniel_and_Elsie_l131_131300


namespace solution_set_of_inequality_l131_131672

theorem solution_set_of_inequality (a : ℝ) (h : a < 0) :
  {x : ℝ | (x - 1) * (a * x - 4) < 0} = {x : ℝ | x > 1 ∨ x < 4 / a} :=
sorry

end solution_set_of_inequality_l131_131672


namespace matt_total_score_l131_131982

-- Definitions from the conditions
def num_2_point_shots : ℕ := 4
def num_3_point_shots : ℕ := 2
def score_per_2_point_shot : ℕ := 2
def score_per_3_point_shot : ℕ := 3

-- Proof statement
theorem matt_total_score : 
  (num_2_point_shots * score_per_2_point_shot) + 
  (num_3_point_shots * score_per_3_point_shot) = 14 := 
by 
  sorry  -- placeholder for the actual proof

end matt_total_score_l131_131982


namespace mildred_initial_oranges_l131_131985

theorem mildred_initial_oranges (final_oranges : ℕ) (added_oranges : ℕ) 
  (final_oranges_eq : final_oranges = 79) (added_oranges_eq : added_oranges = 2) : 
  final_oranges - added_oranges = 77 :=
by
  -- proof steps would go here
  sorry

end mildred_initial_oranges_l131_131985


namespace probability_solution_l131_131972

noncomputable def binom_10_7 := Nat.choose 10 7
noncomputable def binom_10_6 := Nat.choose 10 6

theorem probability_solution (p q : ℝ) (h₁ : q = 1 - p) (h₂ : binom_10_7 = 120) (h₃ : binom_10_6 = 210)
  (h₄ : 120 * p ^ 7 * q ^ 3 = 210 * p ^ 6 * q ^ 4) : p = 7 / 11 := 
sorry

end probability_solution_l131_131972


namespace point_P_lies_on_x_axis_l131_131827

noncomputable def point_on_x_axis (x : ℝ) : Prop :=
  (0 = (0 : ℝ)) -- This is a placeholder definition stating explicitly that point lies on the x-axis

theorem point_P_lies_on_x_axis (x : ℝ) : point_on_x_axis x :=
by
  sorry

end point_P_lies_on_x_axis_l131_131827


namespace train_cross_bridge_time_l131_131476

-- Length of the train in meters
def train_length : ℕ := 165

-- Length of the bridge in meters
def bridge_length : ℕ := 660

-- Speed of the train in kmph
def train_speed_kmph : ℕ := 54

-- Conversion factor from kmph to m/s
def kmph_to_mps : ℚ := 5 / 18

-- Total distance to be traveled by the train to cross the bridge
def total_distance : ℕ := train_length + bridge_length

-- Speed of the train in meters per second (m/s)
def train_speed_mps : ℚ := train_speed_kmph * kmph_to_mps

-- Time taken for the train to cross the bridge (in seconds)
def time_to_cross_bridge : ℚ := total_distance / train_speed_mps

-- Prove that the time taken for the train to cross the bridge is 55 seconds
theorem train_cross_bridge_time : time_to_cross_bridge = 55 := by
  -- Proof goes here
  sorry

end train_cross_bridge_time_l131_131476


namespace original_population_l131_131358

-- Define the conditions
def population_increase (n : ℕ) : ℕ := n + 1200
def population_decrease (p : ℕ) : ℕ := (89 * p) / 100
def final_population (n : ℕ) : ℕ := population_decrease (population_increase n)

-- Claim that needs to be proven
theorem original_population (n : ℕ) (H : final_population n = n - 32) : n = 10000 :=
by
  sorry

end original_population_l131_131358


namespace binomial_10_10_binomial_10_9_l131_131047

-- Prove that \(\binom{10}{10} = 1\)
theorem binomial_10_10 : Nat.choose 10 10 = 1 :=
by sorry

-- Prove that \(\binom{10}{9} = 10\)
theorem binomial_10_9 : Nat.choose 10 9 = 10 :=
by sorry

end binomial_10_10_binomial_10_9_l131_131047


namespace find_x_l131_131530

theorem find_x (x y z : ℕ) 
  (h1 : x + y = 74) 
  (h2 : (x + y) + y + z = 164) 
  (h3 : z - y = 16) : 
  x = 37 :=
sorry

end find_x_l131_131530


namespace jogging_time_l131_131675

theorem jogging_time (distance : ℝ) (speed : ℝ) (h1 : distance = 25) (h2 : speed = 5) : (distance / speed) = 5 :=
by
  rw [h1, h2]
  norm_num

end jogging_time_l131_131675


namespace percentage_of_sikhs_l131_131563

theorem percentage_of_sikhs
  (total_boys : ℕ := 400)
  (percent_muslims : ℕ := 44)
  (percent_hindus : ℕ := 28)
  (other_boys : ℕ := 72) :
  ((total_boys - (percent_muslims * total_boys / 100 + percent_hindus * total_boys / 100 + other_boys)) * 100 / total_boys) = 10 :=
by
  -- proof goes here
  sorry

end percentage_of_sikhs_l131_131563


namespace sum_four_terms_eq_40_l131_131557

def sequence_sum (S_n : ℕ → ℕ) (n : ℕ) : ℕ := n^2 + 2 * n + 5

theorem sum_four_terms_eq_40 (S_n : ℕ → ℕ) (h : ∀ n : ℕ, S_n n = sequence_sum S_n n) :
  (S_n 6 - S_n 2) = 40 :=
by
  sorry

end sum_four_terms_eq_40_l131_131557


namespace percentage_increase_in_rectangle_area_l131_131887

theorem percentage_increase_in_rectangle_area (L W : ℝ) :
  (1.35 * 1.35 * L * W - L * W) / (L * W) * 100 = 82.25 :=
by sorry

end percentage_increase_in_rectangle_area_l131_131887


namespace imaginary_part_of_z_l131_131665

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + I) = 1 - 3 * I) : z.im = -2 := by
  sorry

end imaginary_part_of_z_l131_131665


namespace compute_expression_l131_131916

theorem compute_expression (x : ℕ) (h : x = 3) : (x^8 + 8 * x^4 + 16) / (x^4 - 4) = 93 :=
by
  rw [h]
  sorry

end compute_expression_l131_131916


namespace runs_scored_by_c_l131_131743

-- Definitions
variables (A B C : ℕ)

-- Conditions as hypotheses
theorem runs_scored_by_c (h1 : B = 3 * A) (h2 : C = 5 * B) (h3 : A + B + C = 95) : C = 75 :=
by
  -- Proof will be here
  sorry

end runs_scored_by_c_l131_131743


namespace midpoint_of_complex_numbers_l131_131426

theorem midpoint_of_complex_numbers :
  let A := (1 - 1*I) / (1 + 1)
  let B := (1 + 1*I) / (1 + 1)
  (A + B) / 2 = 1 / 2 := by
sorry

end midpoint_of_complex_numbers_l131_131426


namespace coordinates_of_point_l131_131398

theorem coordinates_of_point (x : ℝ) (P : ℝ × ℝ) (h : P = (1 - x, 2 * x + 1)) (y_axis : P.1 = 0) : P = (0, 3) :=
by
  sorry

end coordinates_of_point_l131_131398


namespace parallel_lines_condition_suff_not_nec_l131_131386

theorem parallel_lines_condition_suff_not_nec 
  (a : ℝ) : (a = -2) → 
  (∀ x y : ℝ, ax + 2 * y - 1 = 0) → 
  (∀ x y : ℝ, x + (a + 1) * y + 4 = 0) → 
  (∀ x1 y1 x2 y2 : ℝ, ((a = -2) → (2 * y1 - 2 * x1 = 1) → (y2 - x2 = -4) → (x1 = x2 → y1 = y2))) ∧ 
  (∃ b : ℝ, ¬ (b = -2) ∧ ((2 * y1 - b * x1 = 1) → (x2 - (b + 1) * y2 = -4) → ¬(x1 = x2 → y1 = y2)))
   :=
by
  sorry

end parallel_lines_condition_suff_not_nec_l131_131386


namespace sin_2x_value_l131_131402

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x)

theorem sin_2x_value (x : ℝ) (h1 : f x = 5 / 3) (h2 : -Real.pi / 6 < x) (h3 : x < Real.pi / 6) :
  Real.sin (2 * x) = (Real.sqrt 3 - 2 * Real.sqrt 2) / 6 := 
sorry

end sin_2x_value_l131_131402


namespace circle_equation_tangent_to_x_axis_l131_131450

/--
The standard equation of a circle with center (-5, 4) and tangent to the x-axis
is given by (x + 5)² + (y - 4)² = 16.
-/
theorem circle_equation_tangent_to_x_axis :
  ∀ x y : ℝ, (x + 5) ^ 2 + (y - 4) ^ 2 = 16 ↔
    (x, y) ∈ {p : ℝ × ℝ | (p.1 + 5) ^ 2 + (p.2 - 4) ^ 2 = 16} :=
by 
  sorry

end circle_equation_tangent_to_x_axis_l131_131450


namespace solve_s_l131_131373

theorem solve_s (s : ℝ) (h_pos : 0 < s) (h_eq : s^3 = 256) : s = 4 :=
sorry

end solve_s_l131_131373


namespace gold_copper_alloy_ratio_l131_131019

theorem gold_copper_alloy_ratio 
  (water : ℝ) 
  (G : ℝ) 
  (C : ℝ) 
  (H1 : G = 10 * water)
  (H2 : C = 6 * water)
  (H3 : 10 * G + 6 * C = 8 * (G + C)) : 
  G / C = 1 :=
by
  sorry

end gold_copper_alloy_ratio_l131_131019


namespace sum_of_cubes_l131_131967

open Real

theorem sum_of_cubes (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
(h_eq : (a^3 + 6) / a = (b^3 + 6) / b ∧ (b^3 + 6) / b = (c^3 + 6) / c) :
  a^3 + b^3 + c^3 = -18 := 
by sorry

end sum_of_cubes_l131_131967


namespace number_of_carbon_atoms_l131_131483

-- Definitions and Conditions
def hydrogen_atoms : ℕ := 6
def molecular_weight : ℕ := 78
def hydrogen_atomic_weight : ℕ := 1
def carbon_atomic_weight : ℕ := 12

-- Theorem Statement: Number of Carbon Atoms
theorem number_of_carbon_atoms 
  (H_atoms : ℕ := hydrogen_atoms)
  (M_weight : ℕ := molecular_weight)
  (H_weight : ℕ := hydrogen_atomic_weight)
  (C_weight : ℕ := carbon_atomic_weight) : 
  (M_weight - H_atoms * H_weight) / C_weight = 6 :=
sorry

end number_of_carbon_atoms_l131_131483


namespace water_added_l131_131169

theorem water_added (capacity : ℝ) (percentage_initial : ℝ) (percentage_final : ℝ) :
  capacity = 120 →
  percentage_initial = 0.30 →
  percentage_final = 0.75 →
  ((percentage_final * capacity) - (percentage_initial * capacity)) = 54 :=
by intros
   sorry

end water_added_l131_131169


namespace rectangle_area_function_relationship_l131_131593

theorem rectangle_area_function_relationship (x y : ℝ) (h : x * y = 12) : y = 12 / x :=
by
  sorry

end rectangle_area_function_relationship_l131_131593


namespace count_perfect_squares_l131_131546

theorem count_perfect_squares (N : Nat) :
  ∃ k : Nat, k = 1666 ∧ ∀ m, (∃ n, m = n * n ∧ m < 10^8 ∧ 36 ∣ m) ↔ (m = 36 * k ^ 2 ∧ k < 10^4) :=
sorry

end count_perfect_squares_l131_131546


namespace Tim_weekly_earnings_l131_131617

theorem Tim_weekly_earnings :
  let tasks_per_day := 100
  let pay_per_task := 1.2
  let days_per_week := 6
  let daily_earnings := tasks_per_day * pay_per_task
  let weekly_earnings := daily_earnings * days_per_week
  weekly_earnings = 720 := by
  sorry

end Tim_weekly_earnings_l131_131617


namespace mushroom_collection_l131_131762

variable (a b v g : ℕ)

theorem mushroom_collection : 
  (a / 2 + 2 * b = v + g) ∧ (a + b = v / 2 + 2 * g) → (v = 2 * b) ∧ (a = 2 * g) :=
by
  sorry

end mushroom_collection_l131_131762


namespace smallest_rel_prime_to_180_l131_131235

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Int.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Int.gcd y 180 = 1 → x ≤ y := 
sorry

end smallest_rel_prime_to_180_l131_131235


namespace decimal_equivalent_l131_131484

theorem decimal_equivalent (x : ℚ) (h : x = 16 / 50) : x = 32 / 100 :=
by
  sorry

end decimal_equivalent_l131_131484


namespace find_a_b_find_k_range_l131_131083

-- Define the conditions for part 1
def quad_inequality (a x : ℝ) : Prop :=
  a * x^2 - 3 * x + 2 > 0

def solution_set (x b : ℝ) : Prop :=
  x < 1 ∨ x > b

theorem find_a_b (a b : ℝ) :
  (∀ x, quad_inequality a x ↔ solution_set x b) → (a = 1 ∧ b = 2) :=
sorry

-- Define the conditions for part 2
def valid_x_y (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

def equation1 (a b x y : ℝ) : Prop :=
  a / x + b / y = 1

def inequality1 (x y k : ℝ) : Prop :=
  2 * x + y ≥ k^2 + k + 2

theorem find_k_range (a b : ℝ) (x y k : ℝ) :
  a = 1 → b = 2 → valid_x_y x y → equation1 a b x y → inequality1 x y k →
  (-3 ≤ k ∧ k ≤ 2) :=
sorry

end find_a_b_find_k_range_l131_131083


namespace smallest_rel_prime_to_180_is_7_l131_131245

theorem smallest_rel_prime_to_180_is_7 :
  ∃ x : ℕ, x > 1 ∧ gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial, },
  split,
  { exact dec_trivial, },
  intros y hy,
  cases hy with hy1 hy2,
  exact dec_trivial,
end

end smallest_rel_prime_to_180_is_7_l131_131245


namespace no_real_solutions_for_equation_l131_131781

theorem no_real_solutions_for_equation:
  ∀ x : ℝ, (3 * x / (x^2 + 2 * x + 4) + 4 * x / (x^2 - 4 * x + 5) = 1) →
  (¬(∃ x : ℝ, 3 * x / (x^2 + 2 * x + 4) + 4 * x / (x^2 - 4 * x + 5) = 1)) :=
by
  sorry

end no_real_solutions_for_equation_l131_131781


namespace prasanna_speed_l131_131292

variable (v_L : ℝ) (d t : ℝ)

theorem prasanna_speed (hLaxmiSpeed : v_L = 18) (htime : t = 1) (hdistance : d = 45) : 
  ∃ v_P : ℝ, v_P = 27 :=
  sorry

end prasanna_speed_l131_131292


namespace problem_part1_problem_part2_l131_131079

-- Statement for Part (1)
theorem problem_part1 (a b : ℝ) (h_sol : {x : ℝ | ax^2 - 3 * x + 2 > 0} = {x : ℝ | x < 1 ∨ x > 2}) :
  a = 1 ∧ b = 2 := sorry

-- Statement for Part (2)
theorem problem_part2 (x y k : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) 
  (h_eq : (1 / x) + (2 / y) = 1) (h_ineq : 2 * x + y ≥ k^2 + k + 2) :
  -3 ≤ k ∧ k ≤ 2 := sorry

end problem_part1_problem_part2_l131_131079


namespace smallest_rel_prime_to_180_l131_131249

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ x ≤ 7 ∧ (∀ y : ℕ, y > 1 ∧ y < x → y.gcd 180 ≠ 1) ∧ x.gcd 180 = 1 :=
by
  sorry

end smallest_rel_prime_to_180_l131_131249


namespace ryan_total_commuting_time_l131_131459

def biking_time : ℕ := 30
def bus_time : ℕ := biking_time + 10
def bus_commutes : ℕ := 3
def total_bus_time : ℕ := bus_time * bus_commutes
def friend_time : ℕ := biking_time - (2 * biking_time / 3)
def total_commuting_time : ℕ := biking_time + total_bus_time + friend_time

theorem ryan_total_commuting_time :
  total_commuting_time = 160 :=
by
  sorry

end ryan_total_commuting_time_l131_131459


namespace obtuse_triangle_count_l131_131261

-- Definitions based on conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  2 * b = a + c

def is_obtuse_triangle (a b c : ℕ) : Prop :=
  a * a + b * b < c * c ∨ b * b + c * c < a * a ∨ c * c + a * a < b * b

-- Main conjecture to prove
theorem obtuse_triangle_count :
  ∃ (n : ℕ), n = 157 ∧
    ∀ (a b c : ℕ), 
      a <= 50 ∧ b <= 50 ∧ c <= 50 ∧ 
      is_arithmetic_sequence a b c ∧ 
      is_triangle a b c ∧ 
      is_obtuse_triangle a b c → 
    true := sorry

end obtuse_triangle_count_l131_131261


namespace purchase_price_of_grinder_l131_131114

theorem purchase_price_of_grinder (G : ℝ) (H : 0.95 * G + 8800 - (G + 8000) = 50) : G = 15000 := 
sorry

end purchase_price_of_grinder_l131_131114


namespace combined_PPC_correct_l131_131605

noncomputable def combined_PPC (K : ℝ) : ℝ :=
  if K ≤ 2 then 168 - 0.5 * K^2
  else if K ≤ 22 then 170 - 2 * K
  else if K ≤ 36 then 20 * K - 0.5 * K^2 - 72
  else 0

theorem combined_PPC_correct (K : ℝ) :
  (K ≤ 2 → combined_PPC K = 168 - 0.5 * K^2) ∧
  (2 < K ∧ K ≤ 22 → combined_PPC K = 170 - 2 * K) ∧
  (22 < K ∧ K ≤ 36 → combined_PPC K = 20 * K - 0.5 * K^2 - 72) :=
by 
  split
  all_goals
  intros 
  unfold combined_PPC
  try {simp [if_pos]}
  try {simp [if_neg, if_pos]}
  try {simp [if_neg, if_neg, if_pos]}
  sorry

end combined_PPC_correct_l131_131605


namespace range_of_a_l131_131556

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, 2^(2 * x) + 2^x * a + a + 1 = 0) : a ≤ 2 - 2 * Real.sqrt 2 :=
sorry

end range_of_a_l131_131556


namespace min_value_of_f_l131_131650

noncomputable def f (x : ℝ) := max (3 - x) (x^2 - 4*x + 3)

theorem min_value_of_f : ∃ x : ℝ, f x = -1 :=
by {
  use 2,
  sorry
}

end min_value_of_f_l131_131650


namespace complement_of_P_union_Q_in_Z_is_M_l131_131835

-- Definitions of the sets M, P, Q
def M : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}
def P : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def Q : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}

-- Theorem statement
theorem complement_of_P_union_Q_in_Z_is_M : (Set.univ \ (P ∪ Q)) = M :=
by 
  sorry

end complement_of_P_union_Q_in_Z_is_M_l131_131835


namespace all_plants_diseased_l131_131306

theorem all_plants_diseased (n : ℕ) (h : n = 1007) : 
  n * 2 = 2014 := by
  sorry

end all_plants_diseased_l131_131306


namespace length_of_platform_is_180_l131_131359

-- Define the train passing a platform and a man with given speeds and times
def train_pass_platform (speed : ℝ) (time_man time_platform : ℝ) (length_train length_platform : ℝ) :=
  time_man = length_train / speed ∧ 
  time_platform = (length_train + length_platform) / speed

-- Given conditions
noncomputable def train_length_platform :=
  ∃ length_platform,
    train_pass_platform 15 20 32 300 length_platform ∧
    length_platform = 180

-- The main theorem we want to prove
theorem length_of_platform_is_180 : train_length_platform :=
sorry

end length_of_platform_is_180_l131_131359


namespace count_perfect_squares_divisible_by_36_l131_131547

theorem count_perfect_squares_divisible_by_36 :
  let N := 10000
  let max_square := 10^8
  let multiple := 36
  let valid_divisor := 1296
  let count_multiples := 277
  (∀ N : ℕ, N^2 < max_square → (∃ k : ℕ, N = k * multiple ∧ k < N)) → 
  ∃ cnt : ℕ, cnt = count_multiples := 
by {
  sorry
}

end count_perfect_squares_divisible_by_36_l131_131547


namespace stamp_exhibition_l131_131635

def total_number_of_stamps (x : ℕ) : ℕ := 3 * x + 24

theorem stamp_exhibition : ∃ x : ℕ, total_number_of_stamps x = 174 ∧ (4 * x - 26) = 174 :=
by
  sorry

end stamp_exhibition_l131_131635


namespace find_b_given_a_l131_131988

-- Definitions based on the conditions
def varies_inversely (a b : ℝ) (k : ℝ) : Prop := a * b = k
def k_value : ℝ := 400

-- The proof statement
theorem find_b_given_a (a b : ℝ) (h1 : varies_inversely 800 0.5 k_value) (h2 : a = 3200) : b = 0.125 :=
by
  -- skipped proof
  sorry

end find_b_given_a_l131_131988


namespace probability_none_given_no_A_l131_131771

/-- The probability that an individual has none of the four risk factors given they 
do not have risk factor A is 31/46 in simplest form, where 31 and 46 are relatively prime. -/
theorem probability_none_given_no_A : 
  (∀ p q : ℕ, (p * q ≠ 0) → p.gcd q = 1 → Rat.mk p q = 31 / 46 → p + q = 77) :=
sorry

end probability_none_given_no_A_l131_131771


namespace jenny_eggs_per_basket_l131_131831

theorem jenny_eggs_per_basket :
  ∃ n, (30 % n = 0 ∧ 42 % n = 0 ∧ 18 % n = 0 ∧ n >= 6) → n = 6 :=
by
  sorry

end jenny_eggs_per_basket_l131_131831


namespace quadrilateral_diagonals_perpendicular_l131_131303

def convex_quadrilateral (A B C D : Type) : Prop := sorry -- Assume it’s defined elsewhere 
def tangent_to_all_sides (circle : Type) (A B C D : Type) : Prop := sorry -- Assume it’s properly specified with its conditions elsewhere
def tangent_to_all_extensions (circle : Type) (A B C D : Type) : Prop := sorry -- Same as above

theorem quadrilateral_diagonals_perpendicular
  (A B C D : Type)
  (h_convex : convex_quadrilateral A B C D)
  (incircle excircle : Type)
  (h_incircle : tangent_to_all_sides incircle A B C D)
  (h_excircle : tangent_to_all_extensions excircle A B C D) : 
  (⊥ : Prop) :=  -- statement indicating perpendicularity 
sorry

end quadrilateral_diagonals_perpendicular_l131_131303


namespace geometric_loci_l131_131942

noncomputable def quadratic_discriminant (x y : ℝ) : ℝ :=
  x^2 + 4 * y^2 - 4

-- Conditions:
def real_and_distinct (x y : ℝ) := 
  ((x^2) / 4 + y^2 > 1) 

def equal_and_real (x y : ℝ) := 
  ((x^2) / 4 + y^2 = 1) 

def complex_roots (x y : ℝ) := 
  ((x^2) / 4 + y^2 < 1)

def both_roots_positive (x y : ℝ) := 
  (x < 0) ∧ (-1 < y) ∧ (y < 1)

def both_roots_negative (x y : ℝ) := 
  (x > 0) ∧ (-1 < y) ∧ (y < 1)

def opposite_sign_roots (x y : ℝ) := 
  (y > 1) ∨ (y < -1)

theorem geometric_loci (x y : ℝ) :
  (real_and_distinct x y ∨ equal_and_real x y ∨ complex_roots x y) ∧ 
  ((real_and_distinct x y ∧ both_roots_positive x y) ∨
   (real_and_distinct x y ∧ both_roots_negative x y) ∨
   (real_and_distinct x y ∧ opposite_sign_roots x y)) := 
sorry

end geometric_loci_l131_131942


namespace megan_folders_l131_131578

def filesOnComputer : Nat := 93
def deletedFiles : Nat := 21
def filesPerFolder : Nat := 8

theorem megan_folders:
  let remainingFiles := filesOnComputer - deletedFiles
  (remainingFiles / filesPerFolder) = 9 := by
    sorry

end megan_folders_l131_131578


namespace smallest_rel_prime_to_180_l131_131250

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  { intro y,
    intro h,
    cases h with h1 h2,
    repeat { try { apply dec_trivial,
                   apply lt_or_eq_of_le,
                   norm_num,
                   apply Nat.prime_not_dvd_mul,
                   norm_num,
                   apply not_or_distrib.mpr,
                   split,
                   norm_cast,
                   intro,
                   exact le_antisymm _ },
           sorry }
end

end smallest_rel_prime_to_180_l131_131250


namespace shell_placements_l131_131963

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem shell_placements : factorial 14 / 7 = 10480142147302400 := by
  sorry

end shell_placements_l131_131963


namespace exists_sequences_l131_131130

theorem exists_sequences (m n : Nat → Nat) (h₁ : ∀ k, m k = 2 * k) (h₂ : ∀ k, n k = 5 * k * k)
  (h₃ : ∀ (i j : Nat), (i ≠ j) → (m i ≠ m j) ∧ (n i ≠ n j)) :
  (∀ k, Nat.sqrt (n k + (m k) * (m k)) = 3 * k) ∧
  (∀ k, Nat.sqrt (n k - (m k) * (m k)) = k) :=
by 
  sorry

end exists_sequences_l131_131130


namespace binomial_coeffs_odd_iff_l131_131663

def binomial_expansion (a b : ℕ) (n : ℕ) := (a + b) ^ n

theorem binomial_coeffs_odd_iff (n : ℕ) :
  (∀ k, 0 ≤ k ∧ k ≤ n → Nat.choose n k % 2 = 1) ↔ (∃ k, n = 2 ^ k - 1) := 
sorry

end binomial_coeffs_odd_iff_l131_131663


namespace front_view_l131_131649

def first_column_heights := [3, 2]
def middle_column_heights := [1, 4, 2]
def third_column_heights := [5]

theorem front_view (h1 : first_column_heights = [3, 2])
                   (h2 : middle_column_heights = [1, 4, 2])
                   (h3 : third_column_heights = [5]) :
    [3, 4, 5] = [
        first_column_heights.foldr max 0,
        middle_column_heights.foldr max 0,
        third_column_heights.foldr max 0
    ] :=
    sorry

end front_view_l131_131649


namespace general_formula_for_sequence_l131_131528

def sequence_terms (a : ℕ → ℚ) : Prop :=
∀ n : ℕ, n > 0 → a n = 1 / (n * (n + 1))

def seq_conditions (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
a 1 = 1 / 2 ∧ (∀ n : ℕ, n > 0 → S n = n^2 * a n)

theorem general_formula_for_sequence :
  ∃ a S : ℕ → ℚ, seq_conditions a S ∧ sequence_terms a := by
  sorry

end general_formula_for_sequence_l131_131528


namespace length_of_first_train_is_correct_l131_131734

noncomputable def length_of_first_train (speed1_km_hr speed2_km_hr : ℝ) (time_cross_sec : ℝ) (length2_m : ℝ) : ℝ :=
  let speed1_m_s := speed1_km_hr * (5 / 18)
  let speed2_m_s := speed2_km_hr * (5 / 18)
  let relative_speed_m_s := speed1_m_s + speed2_m_s
  let total_distance_m := relative_speed_m_s * time_cross_sec
  total_distance_m - length2_m

theorem length_of_first_train_is_correct : 
  length_of_first_train 60 40 11.879049676025918 160 = 170 := by
  sorry

end length_of_first_train_is_correct_l131_131734


namespace find_unknown_gift_l131_131567

def money_from_aunt : ℝ := 9
def money_from_uncle : ℝ := 9
def money_from_bestfriend1 : ℝ := 22
def money_from_bestfriend2 : ℝ := 22
def money_from_bestfriend3 : ℝ := 22
def money_from_sister : ℝ := 7
def mean_money : ℝ := 16.3
def number_of_gifts : ℕ := 7

theorem find_unknown_gift (X : ℝ)
  (h1: money_from_aunt = 9)
  (h2: money_from_uncle = 9)
  (h3: money_from_bestfriend1 = 22)
  (h4: money_from_bestfriend2 = 22)
  (h5: money_from_bestfriend3 = 22)
  (h6: money_from_sister = 7)
  (h7: mean_money = 16.3)
  (h8: number_of_gifts = 7)
  : X = 23.1 := sorry

end find_unknown_gift_l131_131567


namespace prove_angle_A_l131_131113

-- Definitions and conditions in triangle ABC
variables (A B C : ℝ) (a b c : ℝ) (h₁ : a^2 - b^2 = 3 * b * c) (h₂ : sin C = 2 * sin B)

-- Objective: Prove that angle A is 120 degrees
theorem prove_angle_A : A = 120 :=
sorry

end prove_angle_A_l131_131113


namespace raised_bed_height_l131_131913

theorem raised_bed_height : 
  ∀ (total_planks : ℕ) (num_beds : ℕ) (planks_per_bed : ℕ) (height : ℚ),
  total_planks = 50 →
  num_beds = 10 →
  planks_per_bed = 4 * height →
  (total_planks = num_beds * planks_per_bed) →
  height = 5 / 4 :=
by
  intros total_planks num_beds planks_per_bed H
  intros h1 h2 h3 h4
  sorry

end raised_bed_height_l131_131913


namespace dodecagon_diagonals_l131_131804

/--
The formula for the number of diagonals in a convex n-gon is given by (n * (n - 3)) / 2.
-/
def number_of_diagonals (n : Nat) : Nat := (n * (n - 3)) / 2

/--
A dodecagon has 12 sides.
-/
def dodecagon_sides : Nat := 12

/--
The number of diagonals in a convex dodecagon is 54.
-/
theorem dodecagon_diagonals : number_of_diagonals dodecagon_sides = 54 := by
  sorry

end dodecagon_diagonals_l131_131804


namespace part1_part2_l131_131526

open Function

def Tournament (α : Type _) [Fintype α] [DecidableEq α] :=
  { G : SimpleGraph α // ∀ v, ∃ v', G.Adj v v' ∧ G.Adj v' v }

def good_tournament (G : SimpleGraph (Fin 8)) : Prop :=
  ∀ v, ¬ (∃ p, @SimpleGraph.Walk.is_cycle _ G v p)

def bad_tournament (G : SimpleGraph (Fin 8)) : Prop :=
  ¬ (good_tournament G)

theorem part1 : ∃ G : SimpleGraph (Fin 8), bad_tournament G ∧ 
  ∀ G', (∀ v w, G.Adj v w ↔ G'.Adj v w ∨ G'.Adj w v) → bad_tournament G' := 
sorry

theorem part2 : ∀ G : SimpleGraph (Fin 8), ∃ G' : SimpleGraph (Fin 8),
  (∃ edges_to_reorient : Finset (Fin 8 × Fin 8), edges_to_reorient.card ≤ 8 ∧
    ∀ v w, (v, w) ∈ edges_to_reorient → G'.Adj v w ↔ ¬ G.Adj v w) ∧
    good_tournament G' := 
sorry

end part1_part2_l131_131526


namespace ratio_of_rooms_l131_131372

theorem ratio_of_rooms (rooms_danielle : ℕ) (rooms_grant : ℕ) (ratio_grant_heidi : ℚ)
  (h1 : rooms_danielle = 6)
  (h2 : rooms_grant = 2)
  (h3 : ratio_grant_heidi = 1/9) :
  (18 : ℚ) / rooms_danielle = 3 :=
by
  sorry

end ratio_of_rooms_l131_131372


namespace trig_identity_l131_131936

theorem trig_identity (α : ℝ) (h : Real.sin (π + α) = 1 / 2) : Real.cos (α - 3 / 2 * π) = 1 / 2 :=
  sorry

end trig_identity_l131_131936


namespace smallest_n_l131_131229

theorem smallest_n (n : ℕ) (h1 : n > 2016) (h2 : (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0) : n = 2020 :=
sorry

end smallest_n_l131_131229


namespace sum_squares_of_roots_of_quadratic_l131_131928

theorem sum_squares_of_roots_of_quadratic:
  ∀ (s_1 s_2 : ℝ),
  (s_1 + s_2 = 20) ∧ (s_1 * s_2 = 32) →
  (s_1^2 + s_2^2 = 336) :=
by
  intros s_1 s_2 h
  sorry

end sum_squares_of_roots_of_quadratic_l131_131928


namespace inequality_holds_for_positive_x_l131_131440

theorem inequality_holds_for_positive_x (x : ℝ) (h : 0 < x) :
  (1 + x + x^2) * (1 + x + x^2 + x^3 + x^4) ≤ (1 + x + x^2 + x^3)^2 :=
sorry

end inequality_holds_for_positive_x_l131_131440


namespace geometric_progression_fourth_term_l131_131866

theorem geometric_progression_fourth_term (a b c : ℝ) (r : ℝ) 
  (h1 : a = 2) (h2 : b = 2 * Real.sqrt 2) (h3 : c = 4) (h4 : r = Real.sqrt 2)
  (h5 : b = a * r) (h6 : c = b * r) :
  c * r = 4 * Real.sqrt 2 := 
sorry

end geometric_progression_fourth_term_l131_131866


namespace compare_values_l131_131531

-- Define that f(x) is an even function, periodic and satisfies decrease and increase conditions as given
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

noncomputable def f : ℝ → ℝ := sorry -- the exact definition of f is unknown, so we use sorry for now

-- The conditions of the problem
axiom f_even : is_even_function f
axiom f_period : periodic_function f 2
axiom f_decreasing : decreasing_on_interval f (-1) 0
axiom f_transformation : ∀ x, f (x + 1) = 1 / f x

-- Prove the comparison between a, b, and c under the given conditions
theorem compare_values (a b c : ℝ) (h1 : a = f (Real.log 2 / Real.log 5)) (h2 : b = f (Real.log 4 / Real.log 2)) (h3 : c = f (Real.sqrt 2)) :
  a > c ∧ c > b :=
by
  sorry

end compare_values_l131_131531


namespace revenue_percentage_change_l131_131143

theorem revenue_percentage_change (P S : ℝ) (hP : P > 0) (hS : S > 0) :
  let P_new := 1.30 * P
  let S_new := 0.80 * S
  let R := P * S
  let R_new := P_new * S_new
  (R_new - R) / R * 100 = 4 := by
  sorry

end revenue_percentage_change_l131_131143


namespace solution_l131_131604

theorem solution
  (a b c d : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (H : (1 / a + 1 / b) * (1 / c + 1 / d) + 1 / (a * b) + 1 / (c * d) = 6 / Real.sqrt (a * b * c * d)) :
  (a^2 + a * c + c^2) / (b^2 - b * d + d^2) = 3 :=
sorry

end solution_l131_131604


namespace largest_lcm_l131_131333

theorem largest_lcm: 
  let lcm182 := Nat.lcm 18 2;
      lcm184 := Nat.lcm 18 4;
      lcm186 := Nat.lcm 18 6;
      lcm189 := Nat.lcm 18 9;
      lcm1812 := Nat.lcm 18 12;
      lcm1815 := Nat.lcm 18 15 
  in max (max (max (max (max lcm182 lcm184) lcm186) lcm189) lcm1812) lcm1815 = 90 :=
by
  sorry

end largest_lcm_l131_131333


namespace no_zero_position_l131_131876

-- Define the concept of regular pentagon vertex assignments and operations
def pentagon_arith_mean (x y : ℝ) : ℝ := (x + y) / 2

-- Define the condition for the initial sum of numbers on the vertices being zero
def initial_sum_zero (a b c d e : ℝ) : Prop := a + b + c + d + e = 0

-- Define the main theorem statement
theorem no_zero_position (a b c d e : ℝ) (h : initial_sum_zero a b c d e) :
  ¬ ∃ a' b' c' d' e' : ℝ, ∀ v w : ℝ, pentagon_arith_mean v w = 0 :=
sorry

end no_zero_position_l131_131876


namespace must_be_true_if_not_all_electric_l131_131901

variable (P : Type) (ElectricCar : P → Prop)

theorem must_be_true_if_not_all_electric (h : ¬ ∀ x : P, ElectricCar x) : 
  ∃ x : P, ¬ ElectricCar x :=
by 
sorry

end must_be_true_if_not_all_electric_l131_131901


namespace letter_puzzle_solutions_l131_131215

theorem letter_puzzle_solutions (A B : ℕ) : 
  (1 ≤ A ∧ A < 10) ∧ (1 ≤ B ∧ B < 10) ∧ (A ≠ B) ∧ (A^B = 10 * B + A) → 
  (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l131_131215


namespace negation_proposition_l131_131721

theorem negation_proposition (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * a * x + a > 0) :=
sorry

end negation_proposition_l131_131721


namespace tileability_condition_l131_131703

theorem tileability_condition (a b k m n : ℕ) (h₁ : k ∣ a) (h₂ : k ∣ b) (h₃ : ∃ (t : Nat), t * (a * b) = m * n) : 
  2 * k ∣ m ∨ 2 * k ∣ n := 
sorry

end tileability_condition_l131_131703


namespace per_capita_income_growth_l131_131185

noncomputable def income2020 : ℝ := 3.2
noncomputable def income2022 : ℝ := 3.7
variable (x : ℝ)

/--
Prove the per capita disposable income model.
-/
theorem per_capita_income_growth :
  income2020 * (1 + x)^2 = income2022 :=
sorry

end per_capita_income_growth_l131_131185


namespace z_value_l131_131412

theorem z_value (x y z : ℝ) (h : 1 / x + 1 / y = 2 / z) : z = (x * y) / 2 :=
by
  sorry

end z_value_l131_131412


namespace accurate_to_ten_thousandth_l131_131996

/-- Define the original number --/
def original_number : ℕ := 580000

/-- Define the accuracy of the number represented by 5.8 * 10^5 --/
def is_accurate_to_ten_thousandth_place (n : ℕ) : Prop :=
  n = 5 * 100000 + 8 * 10000

/-- The statement to be proven --/
theorem accurate_to_ten_thousandth : is_accurate_to_ten_thousandth_place original_number :=
by
  sorry

end accurate_to_ten_thousandth_l131_131996


namespace otimes_self_twice_l131_131049

def otimes (x y : ℝ) := x^2 - y^2

theorem otimes_self_twice (a : ℝ) : (otimes (otimes a a) (otimes a a)) = 0 :=
  sorry

end otimes_self_twice_l131_131049


namespace distance_between_A_and_B_l131_131176

-- Given conditions as definitions

def total_time : ℝ := 4
def boat_speed : ℝ := 7.5
def stream_speed : ℝ := 2.5
def distance_AC : ℝ := 10

-- Define the possible solutions for the distance between A and B
def distance_AB (x : ℝ) := 
  (x / (boat_speed + stream_speed) + (x + distance_AC) / (boat_speed - stream_speed) = total_time) 
  ∨ 
  (x / (boat_speed + stream_speed) + (x - distance_AC) / (boat_speed - stream_speed) = total_time)

-- Problem statement
theorem distance_between_A_and_B :
  ∃ x : ℝ, (distance_AB x) ∧ (x = 20 ∨ x = 20 / 3) :=
sorry

end distance_between_A_and_B_l131_131176


namespace time_of_same_distance_l131_131492

theorem time_of_same_distance (m : ℝ) (h_m : 0 ≤ m ∧ m ≤ 60) : 180 - 6 * m = 90 + 0.5 * m :=
by
  sorry

end time_of_same_distance_l131_131492


namespace edge_length_of_cube_l131_131539

/--
Given:
1. A cuboid with base width of 70 cm, base length of 40 cm, and height of 150 cm.
2. A cube-shaped cabinet whose volume is 204,000 cm³ smaller than that of the cuboid.

Prove that one edge of the cube-shaped cabinet is 60 cm.
-/
theorem edge_length_of_cube (W L H V_diff : ℝ) (cuboid_vol : ℝ) (cube_vol : ℝ) (edge : ℝ) :
  W = 70 ∧ L = 40 ∧ H = 150 ∧ V_diff = 204000 ∧ 
  cuboid_vol = W * L * H ∧ cube_vol = cuboid_vol - V_diff ∧ edge ^ 3 = cube_vol -> 
  edge = 60 :=
by
  sorry

end edge_length_of_cube_l131_131539


namespace Anya_walks_to_school_l131_131813

theorem Anya_walks_to_school
  (t_f t_b : ℝ)
  (h1 : t_f + t_b = 1.5)
  (h2 : 2 * t_b = 0.5) :
  2 * t_f = 2.5 :=
by
  -- The proof details will go here eventually.
  sorry

end Anya_walks_to_school_l131_131813


namespace independence_of_events_l131_131091

noncomputable def is_independent (A B : Prop) (chi_squared : ℝ) := 
  chi_squared ≤ 3.841

theorem independence_of_events (A B : Prop) (chi_squared : ℝ) : 
  is_independent A B chi_squared → A ↔ B :=
by
  sorry

end independence_of_events_l131_131091


namespace binomial_16_4_l131_131200

theorem binomial_16_4 : Nat.choose 16 4 = 1820 :=
  sorry

end binomial_16_4_l131_131200


namespace composite_quadratic_l131_131075

theorem composite_quadratic (m n : ℤ) (x1 x2 : ℤ)
  (h1 : 2 * x1^2 + m * x1 + 2 - n = 0)
  (h2 : 2 * x2^2 + m * x2 + 2 - n = 0)
  (h3 : x1 ≠ 0) 
  (h4 : x2 ≠ 0) :
  ∃ (k : ℕ), ∃ (l : ℕ), 
    (k > 1) ∧ (l > 1) ∧ (k * l = (m^2 + n^2) / 4) := sorry

end composite_quadratic_l131_131075


namespace quadratic_inequality_solution_l131_131264

theorem quadratic_inequality_solution (a b c : ℝ) (h_solution_set : ∀ x, ax^2 + bx + c < 0 ↔ x < -1 ∨ x > 3) :
  (a < 0) ∧
  (a + b + c > 0) ∧
  (∀ x, cx^2 - bx + a < 0 ↔ -1/3 < x ∧ x < 1) :=
by
  sorry

end quadratic_inequality_solution_l131_131264


namespace total_tbs_of_coffee_l131_131133

theorem total_tbs_of_coffee (guests : ℕ) (weak_drinkers : ℕ) (medium_drinkers : ℕ) (strong_drinkers : ℕ) 
                           (cups_per_weak_drinker : ℕ) (cups_per_medium_drinker : ℕ) (cups_per_strong_drinker : ℕ) 
                           (tbsp_per_cup_weak : ℕ) (tbsp_per_cup_medium : ℝ) (tbsp_per_cup_strong : ℕ) :
  guests = 18 ∧ 
  weak_drinkers = 6 ∧ 
  medium_drinkers = 6 ∧ 
  strong_drinkers = 6 ∧ 
  cups_per_weak_drinker = 2 ∧ 
  cups_per_medium_drinker = 3 ∧ 
  cups_per_strong_drinker = 1 ∧ 
  tbsp_per_cup_weak = 1 ∧ 
  tbsp_per_cup_medium = 1.5 ∧ 
  tbsp_per_cup_strong = 2 →
  (weak_drinkers * cups_per_weak_drinker * tbsp_per_cup_weak + 
   medium_drinkers * cups_per_medium_drinker * tbsp_per_cup_medium + 
   strong_drinkers * cups_per_strong_drinker * tbsp_per_cup_strong) = 51 :=
by
  sorry

end total_tbs_of_coffee_l131_131133


namespace integer_solutions_positive_product_l131_131782

theorem integer_solutions_positive_product :
  {a : ℤ | (5 + a) * (3 - a) > 0} = {-4, -3, -2, -1, 0, 1, 2} :=
by
  sorry

end integer_solutions_positive_product_l131_131782


namespace fraction_of_value_l131_131151

def value_this_year : ℝ := 16000
def value_last_year : ℝ := 20000

theorem fraction_of_value : (value_this_year / value_last_year) = 4 / 5 := by
  sorry

end fraction_of_value_l131_131151


namespace bleaching_takes_3_hours_l131_131504

-- Define the total time and the relationship between dyeing and bleaching.
def total_time : ℕ := 9
def dyeing_takes_twice (H : ℕ) : Prop := 2 * H + H = total_time

-- Prove that bleaching takes 3 hours.
theorem bleaching_takes_3_hours : ∃ H : ℕ, dyeing_takes_twice H ∧ H = 3 := 
by 
  sorry

end bleaching_takes_3_hours_l131_131504


namespace G5_units_digit_is_0_l131_131202

def power_mod (base : ℕ) (exp : ℕ) (modulus : ℕ) : ℕ :=
  (base ^ exp) % modulus

def G (n : ℕ) : ℕ := 2 ^ (3 ^ n) + 2

theorem G5_units_digit_is_0 : (G 5) % 10 = 0 :=
by
  sorry

end G5_units_digit_is_0_l131_131202


namespace nicky_speed_l131_131842

theorem nicky_speed
  (head_start : ℕ := 36)
  (cristina_speed : ℕ := 6)
  (time_to_catch_up : ℕ := 12)
  (distance_cristina_runs : ℕ := cristina_speed * time_to_catch_up)
  (distance_nicky_runs : ℕ := distance_cristina_runs - head_start)
  (nicky_speed : ℕ := distance_nicky_runs / time_to_catch_up) :
  nicky_speed = 3 :=
by
  sorry

end nicky_speed_l131_131842


namespace alster_caught_two_frogs_l131_131587

-- Definitions and conditions
variables (alster quinn bret : ℕ)

-- Condition 1: Quinn catches twice the amount of frogs as Alster
def quinn_catches_twice_as_alster : Prop := quinn = 2 * alster

-- Condition 2: Bret catches three times the amount of frogs as Quinn
def bret_catches_three_times_as_quinn : Prop := bret = 3 * quinn

-- Condition 3: Bret caught 12 frogs
def bret_caught_twelve : Prop := bret = 12

-- Theorem: How many frogs did Alster catch? Alster caught 2 frogs
theorem alster_caught_two_frogs (h1 : quinn_catches_twice_as_alster alster quinn)
                                (h2 : bret_catches_three_times_as_quinn quinn bret)
                                (h3 : bret_caught_twelve bret) :
                                alster = 2 :=
by sorry

end alster_caught_two_frogs_l131_131587


namespace find_f_42_div_17_l131_131778

def f : ℚ → ℤ := sorry

theorem find_f_42_div_17 : 
  (∀ x y : ℚ, x ≠ y → (x * y = 1 ∨ x + y = 1) → f x * f y = -1) → 
  f 0 = 1 →
  f (42 / 17) = -1 :=
sorry

end find_f_42_div_17_l131_131778


namespace kendra_shirts_needed_l131_131832

def school_shirts_per_week : Nat := 5
def club_shirts_per_week : Nat := 3
def spirit_day_shirt_per_week : Nat := 1
def saturday_shirts_per_week : Nat := 3
def sunday_shirts_per_week : Nat := 3
def family_reunion_shirt_per_month : Nat := 1

def total_shirts_needed_per_week : Nat :=
  school_shirts_per_week + club_shirts_per_week + spirit_day_shirt_per_week +
  saturday_shirts_per_week + sunday_shirts_per_week

def total_shirts_needed_per_four_weeks : Nat :=
  total_shirts_needed_per_week * 4 + family_reunion_shirt_per_month

theorem kendra_shirts_needed : total_shirts_needed_per_four_weeks = 61 := by
  sorry

end kendra_shirts_needed_l131_131832


namespace negative_movement_south_l131_131810

noncomputable def movement_interpretation (x : ℤ) : String :=
if x > 0 then 
  "moving " ++ toString x ++ "m north"
else 
  "moving " ++ toString (-x) ++ "m south"

theorem negative_movement_south : movement_interpretation (-50) = "moving 50m south" := 
by 
  sorry

end negative_movement_south_l131_131810


namespace gcd_36745_59858_l131_131380

theorem gcd_36745_59858 : Nat.gcd 36745 59858 = 7 :=
sorry

end gcd_36745_59858_l131_131380


namespace positive_perfect_squares_multiples_of_36_lt_10_pow_8_l131_131551

theorem positive_perfect_squares_multiples_of_36_lt_10_pow_8 :
  ∃ (count : ℕ), count = 1666 ∧ 
    (∀ n : ℕ, (n * n < 10^8 → n % 6 = 0) ↔ (n < 10^4 ∧ n % 6 = 0)) :=
sorry

end positive_perfect_squares_multiples_of_36_lt_10_pow_8_l131_131551


namespace bus_capacity_l131_131685

def left_side_seats : ℕ := 15
def seats_difference : ℕ := 3
def people_per_seat : ℕ := 3
def back_seat_capacity : ℕ := 7

theorem bus_capacity : left_side_seats + (left_side_seats - seats_difference) * people_per_seat + back_seat_capacity = 88 := 
by
  sorry

end bus_capacity_l131_131685


namespace ryan_total_commuting_time_l131_131458

def biking_time : ℕ := 30
def bus_time : ℕ := biking_time + 10
def bus_commutes : ℕ := 3
def total_bus_time : ℕ := bus_time * bus_commutes
def friend_time : ℕ := biking_time - (2 * biking_time / 3)
def total_commuting_time : ℕ := biking_time + total_bus_time + friend_time

theorem ryan_total_commuting_time :
  total_commuting_time = 160 :=
by
  sorry

end ryan_total_commuting_time_l131_131458


namespace jaden_toy_cars_problem_l131_131693

theorem jaden_toy_cars_problem :
  let initial := 14
  let bought := 28
  let birthday := 12
  let to_vinnie := 3
  let left := 43
  let total := initial + bought + birthday
  let after_vinnie := total - to_vinnie
  (after_vinnie - left = 8) :=
by
  sorry

end jaden_toy_cars_problem_l131_131693


namespace find_amount_l131_131748

-- Given conditions
variables (x A : ℝ)

theorem find_amount :
  (0.65 * x = 0.20 * A) → (x = 190) → (A = 617.5) :=
by
  intros h1 h2
  sorry

end find_amount_l131_131748


namespace factor_polynomial_l131_131377

theorem factor_polynomial (x : ℤ) :
  36 * x ^ 6 - 189 * x ^ 12 + 81 * x ^ 9 = 9 * x ^ 6 * (4 + 9 * x ^ 3 - 21 * x ^ 6) := 
sorry

end factor_polynomial_l131_131377


namespace probability_all_five_dice_even_l131_131880

-- Definitions of conditions
def standard_six_sided_die : Set ℕ := {1, 2, 3, 4, 5, 6}
def even_numbers : Set ℕ := {2, 4, 6}

-- The statement to be proven
theorem probability_all_five_dice_even : 
  (∀ die ∈ standard_six_sided_die, (∃ n ∈ even_numbers, die = n)) → (1 / 32) = (1 / 2) ^ 5 :=
by
  intro h
  sorry

end probability_all_five_dice_even_l131_131880


namespace smallest_rel_prime_to_180_l131_131237

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Int.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Int.gcd y 180 = 1 → x ≤ y := 
sorry

end smallest_rel_prime_to_180_l131_131237


namespace linear_inequality_m_eq_zero_l131_131275

theorem linear_inequality_m_eq_zero (m : ℝ) (x : ℝ) : 
  ((m - 2) * x ^ |m - 1| - 3 > 6) → abs (m - 1) = 1 → m ≠ 2 → m = 0 := by
  intros h1 h2 h3
  -- Proof of m = 0 based on given conditions
  sorry

end linear_inequality_m_eq_zero_l131_131275


namespace total_floor_area_covered_l131_131153

-- Definitions for the given problem
def combined_area : ℕ := 204
def overlap_two_layers : ℕ := 24
def overlap_three_layers : ℕ := 20
def total_floor_area : ℕ := 140

-- Theorem to prove the total floor area covered by the rugs
theorem total_floor_area_covered :
  combined_area - overlap_two_layers - 2 * overlap_three_layers = total_floor_area := by
  sorry

end total_floor_area_covered_l131_131153


namespace number_of_true_propositions_l131_131787

noncomputable def f : ℝ → ℝ := sorry -- since it's not specified, we use sorry here

-- Definitions for the conditions
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Original proposition
def original_proposition (f : ℝ → ℝ) :=
  ∀ x : ℝ, is_odd f → f 0 = 0

-- Converse proposition
def converse_proposition (f : ℝ → ℝ) :=
  f 0 = 0 → ∀ x : ℝ, is_odd f

-- Inverse proposition (logically equivalent to the converse)
def inverse_proposition (f : ℝ → ℝ) :=
  ∀ x : ℝ, ¬(is_odd f) → f 0 ≠ 0

-- Contrapositive proposition (logically equivalent to the original)
def contrapositive_proposition (f : ℝ → ℝ) :=
  f 0 ≠ 0 → ∀ x : ℝ, ¬(is_odd f)

-- Theorem statement
theorem number_of_true_propositions (f : ℝ → ℝ) :
  (original_proposition f → true) ∧
  (converse_proposition f → false) ∧
  (inverse_proposition f → false) ∧
  (contrapositive_proposition f → true) →
  2 = 2 := 
by 
  sorry -- proof to be inserted

end number_of_true_propositions_l131_131787


namespace baking_powder_now_l131_131007

def baking_powder_yesterday : ℝ := 0.4
def baking_powder_used : ℝ := 0.1

theorem baking_powder_now : 
  baking_powder_yesterday - baking_powder_used = 0.3 :=
by
  sorry

end baking_powder_now_l131_131007


namespace travel_agency_choice_l131_131513

noncomputable def cost_A (x : ℕ) : ℝ :=
  350 * x + 1000

noncomputable def cost_B (x : ℕ) : ℝ :=
  400 * x + 800

theorem travel_agency_choice (x : ℕ) :
  if x < 4 then cost_A x > cost_B x
  else if x = 4 then cost_A x = cost_B x
  else cost_A x < cost_B x :=
by sorry

end travel_agency_choice_l131_131513


namespace find_a_l131_131107

theorem find_a (a : ℝ) : 
  (∃ r : ℕ, (10 - 3 * r = 1 ∧ (-a)^r * (Nat.choose 5 r) *  x^(10 - 2 * r - r) = x ∧ -10 = (-a)^3 * (Nat.choose 5 3)))
  → a = 1 :=
sorry

end find_a_l131_131107


namespace eval_f_function_l131_131430

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1 else if x = 0 then Real.pi else 0

theorem eval_f_function : f (f (f (-1))) = Real.pi + 1 :=
  sorry

end eval_f_function_l131_131430


namespace trees_planted_tomorrow_l131_131731

-- Definitions from the conditions
def current_trees := 39
def trees_planted_today := 41
def total_trees := 100

-- Theorem statement matching the proof problem
theorem trees_planted_tomorrow : 
  ∃ (trees_planted_tomorrow : ℕ), current_trees + trees_planted_today + trees_planted_tomorrow = total_trees ∧ trees_planted_tomorrow = 20 := 
by
  sorry

end trees_planted_tomorrow_l131_131731


namespace perfect_square_condition_l131_131382

theorem perfect_square_condition (n : ℤ) : 
    ∃ k : ℤ, n^2 + 6*n + 1 = k^2 ↔ n = 0 ∨ n = -6 := by
  sorry

end perfect_square_condition_l131_131382


namespace parker_added_dumbbells_l131_131710

def initial_dumbbells : Nat := 4
def weight_per_dumbbell : Nat := 20
def total_weight_used : Nat := 120

theorem parker_added_dumbbells :
  (total_weight_used - (initial_dumbbells * weight_per_dumbbell)) / weight_per_dumbbell = 2 := by
  sorry

end parker_added_dumbbells_l131_131710


namespace instantaneous_velocity_at_t_eq_2_l131_131739

variable (t : ℝ)

def displacement (t : ℝ) : ℝ := 2 * (1 - t) ^ 2 

theorem instantaneous_velocity_at_t_eq_2 :
  (deriv (displacement) 2) = 4 :=
sorry

end instantaneous_velocity_at_t_eq_2_l131_131739


namespace sum_of_other_two_angles_is_108_l131_131692

theorem sum_of_other_two_angles_is_108 (A B C : Type) (angleA angleB angleC : ℝ) 
  (h_angle_sum : angleA + angleB + angleC = 180) (h_angleB : angleB = 72) :
  angleA + angleC = 108 := 
by
  sorry

end sum_of_other_two_angles_is_108_l131_131692


namespace min_value_expr_least_is_nine_l131_131660

noncomputable def minimum_value_expression (a b c d : ℝ) : ℝ :=
  ((a + b)^2 + (b - c)^2 + (d - c)^2 + (c - a)^2) / b^2

theorem min_value_expr_least_is_nine (a b c d : ℝ)
  (h1 : b > d) (h2 : d > c) (h3 : c > a) (h4 : b ≠ 0) :
  minimum_value_expression a b c d = 9 := 
sorry

end min_value_expr_least_is_nine_l131_131660


namespace equivalent_problem_l131_131662

def f (x : ℤ) : ℤ := 9 - x

def g (x : ℤ) : ℤ := x - 9

theorem equivalent_problem : g (f 15) = -15 := sorry

end equivalent_problem_l131_131662


namespace order_of_m_n_p_q_l131_131947

variable {m n p q : ℝ} -- Define the variables as real numbers

theorem order_of_m_n_p_q (h1 : m < n) 
                         (h2 : p < q) 
                         (h3 : (p - m) * (p - n) < 0) 
                         (h4 : (q - m) * (q - n) < 0) : 
    m < p ∧ p < q ∧ q < n := 
by
  sorry

end order_of_m_n_p_q_l131_131947


namespace original_cost_of_dolls_l131_131918

theorem original_cost_of_dolls 
  (x : ℝ) -- original cost of each Russian doll
  (savings : ℝ) -- total savings of Daniel
  (h1 : savings = 15 * x) -- Daniel saves enough to buy 15 dolls at original price
  (h2 : savings = 20 * 3) -- with discounted price, he can buy 20 dolls
  : x = 4 :=
by
  sorry

end original_cost_of_dolls_l131_131918


namespace problem_I4_1_l131_131688

variable (A D E B C : Type) [Field A] [Field D] [Field E] [Field B] [Field C]
variable (AD DB DE BC : ℚ)
variable (a : ℚ)
variable (h1 : DE = BC) -- DE parallel to BC
variable (h2 : AD = 4)
variable (h3 : DB = 6)
variable (h4 : DE = 6)

theorem problem_I4_1 : a = 15 :=
  by
  sorry

end problem_I4_1_l131_131688


namespace max_height_l131_131640

-- Define the parabolic function h(t) representing the height of the soccer ball.
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 11

-- State that the maximum height of the soccer ball is 136 feet.
theorem max_height : ∃ t : ℝ, h t = 136 :=
by
  sorry

end max_height_l131_131640


namespace annual_decrease_rate_l131_131320

theorem annual_decrease_rate
  (P0 : ℕ := 8000)
  (P2 : ℕ := 6480) :
  ∃ r : ℝ, 8000 * (1 - r / 100)^2 = 6480 ∧ r = 10 :=
by
  use 10
  sorry

end annual_decrease_rate_l131_131320


namespace sum_of_conjugates_eq_30_l131_131201

theorem sum_of_conjugates_eq_30 :
  (15 - Real.sqrt 2023) + (15 + Real.sqrt 2023) = 30 :=
sorry

end sum_of_conjugates_eq_30_l131_131201


namespace incorrect_major_premise_l131_131033

-- Define a structure for Line and Plane
structure Line : Type :=
  (name : String)

structure Plane : Type :=
  (name : String)

-- Define relationships: parallel and contains
def parallel (l1 l2 : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Conditions
variables 
  (a b : Line) 
  (α : Plane)
  (H1 : line_in_plane a α) 
  (H2 : parallel_line_plane b α)

-- Major premise to disprove
def major_premise (l : Line) (p : Plane) : Prop :=
  ∀ (l_in : Line), line_in_plane l_in p → parallel l l_in

-- State the problem
theorem incorrect_major_premise : ¬major_premise b α :=
sorry

end incorrect_major_premise_l131_131033


namespace borrowed_nickels_l131_131125

-- Define the initial and remaining number of nickels
def initial_nickels : ℕ := 87
def remaining_nickels : ℕ := 12

-- Prove that the number of nickels borrowed is 75
theorem borrowed_nickels : initial_nickels - remaining_nickels = 75 := by
  sorry

end borrowed_nickels_l131_131125


namespace number_of_monkeys_l131_131310

theorem number_of_monkeys (N : ℕ)
  (h1 : N * 1 * 8 = 8)
  (h2 : 3 * 1 * 8 = 3 * 8) :
  N = 8 :=
sorry

end number_of_monkeys_l131_131310


namespace problem_f_l131_131958

/-- Coefficient of the term x^m * y^n in the expansion of (1+x)^6 * (1+y)^4 --/
def f (m n : ℕ) : ℕ := Nat.choose 6 m * Nat.choose 4 n

theorem problem_f :
  f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := by
  -- We skip the proof
  sorry

end problem_f_l131_131958


namespace no_integer_n_gte_1_where_9_divides_7n_plus_n3_l131_131364

theorem no_integer_n_gte_1_where_9_divides_7n_plus_n3 :
  ∀ n : ℕ, 1 ≤ n → ¬ (7^n + n^3) % 9 = 0 := 
by
  intros n hn
  sorry

end no_integer_n_gte_1_where_9_divides_7n_plus_n3_l131_131364


namespace range_of_4x_2y_l131_131097

theorem range_of_4x_2y (x y : ℝ) 
  (h1 : 1 ≤ x + y) (h2 : x + y ≤ 3) 
  (h3 : -1 ≤ x - y) (h4 : x - y ≤ 1) :
  2 ≤ 4 * x + 2 * y ∧ 4 * x + 2 * y ≤ 10 := 
sorry

end range_of_4x_2y_l131_131097


namespace ryan_weekly_commuting_time_l131_131456

-- Define Ryan's commuting conditions
def bike_time (biking_days: Nat) : Nat := biking_days * 30
def bus_time (bus_days: Nat) : Nat := bus_days * 40
def friend_time (friend_days: Nat) : Nat := friend_days * 10

-- Calculate total commuting time per week
def total_commuting_time (biking_days bus_days friend_days: Nat) : Nat := 
  bike_time biking_days + bus_time bus_days + friend_time friend_days

-- Given conditions
def biking_days : Nat := 1
def bus_days : Nat := 3
def friend_days : Nat := 1

-- Formal statement to prove
theorem ryan_weekly_commuting_time : 
  total_commuting_time biking_days bus_days friend_days = 160 := by 
  sorry

end ryan_weekly_commuting_time_l131_131456


namespace problem1_problem2_problem3_l131_131209

-- Define the functions f and g
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def g (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- Problem statements in Lean
theorem problem1 (a b c : ℝ) (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |f a b c x| ≤ 1) : |c| ≤ 1 :=
sorry

theorem problem2 (a b c : ℝ) (h₁ : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |f a b c x| ≤ 1) : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |g a b x| ≤ 2 :=
sorry

theorem problem3 (a b c : ℝ) (ha : a > 0) (hx : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → g a b x ≤ 2) (hf : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |f a b c x| ≤ 1) :
  ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ g a b x = 2 :=
sorry

end problem1_problem2_problem3_l131_131209


namespace max_height_reached_by_rocket_l131_131179

def h (t : ℝ) : ℝ := -12 * t^2 + 72 * t + 36

theorem max_height_reached_by_rocket : ∃ t : ℝ, h t = 144 ∧ ∀ t' : ℝ, h t' ≤ 144 := sorry

end max_height_reached_by_rocket_l131_131179


namespace terrier_to_poodle_grooming_ratio_l131_131723

-- Definitions and conditions
def time_to_groom_poodle : ℕ := 30
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8
def total_grooming_time : ℕ := 210
def time_to_groom_terrier := total_grooming_time - (num_poodles * time_to_groom_poodle) / num_terriers

-- Theorem statement
theorem terrier_to_poodle_grooming_ratio :
  time_to_groom_terrier / time_to_groom_poodle = 1 / 2 :=
by
  sorry

end terrier_to_poodle_grooming_ratio_l131_131723


namespace tire_radius_increase_l131_131128

noncomputable def radius_increase (initial_radius : ℝ) (odometer_initial : ℝ) (odometer_winter : ℝ) : ℝ :=
  let rotations := odometer_initial / ((2 * Real.pi * initial_radius) / 63360)
  let winter_circumference := (odometer_winter / rotations) * 63360
  let new_radius := winter_circumference / (2 * Real.pi)
  new_radius - initial_radius

theorem tire_radius_increase : radius_increase 16 520 505 = 0.32 := by
  sorry

end tire_radius_increase_l131_131128


namespace fraction_product_l131_131329

theorem fraction_product (a b c d e : ℝ) (h1 : a = 1/2) (h2 : b = 1/3) (h3 : c = 1/4) (h4 : d = 1/6) (h5 : e = 144) :
  a * b * c * d * e = 1 := 
by
  -- Given the conditions h1 to h5, we aim to prove the product is 1
  sorry

end fraction_product_l131_131329


namespace fair_coin_heads_probability_l131_131441

theorem fair_coin_heads_probability
  (fair_coin : ∀ n : ℕ, (∀ (heads tails : ℕ), heads + tails = n → (heads / n = 1 / 2) ∧ (tails / n = 1 / 2)))
  (n : ℕ)
  (heads : ℕ)
  (tails : ℕ)
  (h1 : n = 20)
  (h2 : heads = 8)
  (h3 : tails = 12)
  (h4 : heads + tails = n)
  : heads / n = 1 / 2 :=
by
  sorry

end fair_coin_heads_probability_l131_131441


namespace b_divisible_by_8_l131_131965

variable (b : ℕ) (n : ℕ)
variable (hb_even : b % 2 = 0) (hb_pos : b > 0) (hn_gt1 : n > 1)
variable (h_square : ∃ k : ℕ, k^2 = (b^n - 1) / (b - 1))

theorem b_divisible_by_8 : b % 8 = 0 :=
by
  sorry

end b_divisible_by_8_l131_131965


namespace total_savings_percentage_l131_131351

theorem total_savings_percentage :
  let coat_price := 100
  let hat_price := 50
  let shoes_price := 75
  let coat_discount := 0.30
  let hat_discount := 0.40
  let shoes_discount := 0.25
  let original_total := coat_price + hat_price + shoes_price
  let coat_savings := coat_price * coat_discount
  let hat_savings := hat_price * hat_discount
  let shoes_savings := shoes_price * shoes_discount
  let total_savings := coat_savings + hat_savings + shoes_savings
  let savings_percentage := (total_savings / original_total) * 100
  savings_percentage = 30.556 :=
by
  sorry

end total_savings_percentage_l131_131351


namespace arithmetic_series_first_term_l131_131929

theorem arithmetic_series_first_term :
  ∃ a d : ℚ, 
    (30 * (2 * a + 59 * d) = 240) ∧
    (30 * (2 * a + 179 * d) = 3240) ∧
    a = - (247 / 12) :=
by
  sorry

end arithmetic_series_first_term_l131_131929


namespace students_behind_minyoung_l131_131025

-- Definition of the initial conditions
def total_students : ℕ := 35
def students_in_front_of_minyoung : ℕ := 27

-- The question we want to prove
theorem students_behind_minyoung : (total_students - (students_in_front_of_minyoung + 1) = 7) := 
by 
  sorry

end students_behind_minyoung_l131_131025


namespace smallest_rel_prime_to_180_is_7_l131_131242

theorem smallest_rel_prime_to_180_is_7 :
  ∃ x : ℕ, x > 1 ∧ gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial, },
  split,
  { exact dec_trivial, },
  intros y hy,
  cases hy with hy1 hy2,
  exact dec_trivial,
end

end smallest_rel_prime_to_180_is_7_l131_131242


namespace alice_journey_duration_l131_131195
noncomputable def journey_duration (start_hour start_minute end_hour end_minute : ℕ) : ℕ :=
  let start_in_minutes := start_hour * 60 + start_minute
  let end_in_minutes := end_hour * 60 + end_minute
  if end_in_minutes >= start_in_minutes then end_in_minutes - start_in_minutes
  else end_in_minutes + 24 * 60 - start_in_minutes
  
theorem alice_journey_duration :
  ∃ start_hour start_minute end_hour end_minute,
  (7 ≤ start_hour ∧ start_hour < 8 ∧ start_minute = 38) ∧
  (16 ≤ end_hour ∧ end_hour < 17 ∧ end_minute = 35) ∧
  journey_duration start_hour start_minute end_hour end_minute = 537 :=
by {
  sorry
}

end alice_journey_duration_l131_131195


namespace smallest_number_of_eggs_l131_131011

-- Define the conditions given in the problem
def total_containers (c : ℕ) : ℕ := 15 * c - 3

-- Prove that given the conditions, the smallest number of eggs you could have is 162
theorem smallest_number_of_eggs (h : ∃ c : ℕ, total_containers c > 150) : ∃ c : ℕ, total_containers c = 162 :=
by
  sorry

end smallest_number_of_eggs_l131_131011


namespace calculation_is_zero_l131_131769

theorem calculation_is_zero : 
  20062006 * 2007 + 20072007 * 2008 - 2006 * 20072007 - 2007 * 20082008 = 0 := 
by 
  sorry

end calculation_is_zero_l131_131769


namespace total_albums_l131_131841

-- Defining the initial conditions
def albumsAdele : ℕ := 30
def albumsBridget : ℕ := albumsAdele - 15
def albumsKatrina : ℕ := 6 * albumsBridget
def albumsMiriam : ℕ := 7 * albumsKatrina
def albumsCarlos : ℕ := 3 * albumsMiriam
def albumsDiane : ℕ := 2 * albumsKatrina

-- Proving the total number of albums
theorem total_albums :
  albumsAdele + albumsBridget + albumsKatrina + albumsMiriam + albumsCarlos + albumsDiane = 2835 :=
by
  sorry

end total_albums_l131_131841


namespace jaime_saves_enough_l131_131924

-- Definitions of the conditions
def weekly_savings : ℕ := 50
def bi_weekly_expense : ℕ := 46
def target_savings : ℕ := 135

-- The proof goal
theorem jaime_saves_enough : ∃ weeks : ℕ, 2 * ((weeks * weekly_savings - bi_weekly_expense) / 2) = target_savings := 
sorry

end jaime_saves_enough_l131_131924


namespace direct_variation_y_value_l131_131035

theorem direct_variation_y_value (x y : ℝ) (hx1 : x ≤ 10 → y = 3 * x)
  (hx2 : x > 10 → y = 6 * x) : 
  x = 20 → y = 120 := by
  sorry

end direct_variation_y_value_l131_131035


namespace sqrt_of_1_5625_eq_1_25_l131_131882

theorem sqrt_of_1_5625_eq_1_25 : Real.sqrt 1.5625 = 1.25 :=
  sorry

end sqrt_of_1_5625_eq_1_25_l131_131882


namespace dynaco_shares_sold_l131_131761

-- Define the conditions
def MicrotronPrice : ℝ := 36
def DynacoPrice : ℝ := 44
def TotalShares : ℕ := 300
def AvgPrice : ℝ := 40
def TotalValue : ℝ := TotalShares * AvgPrice

-- Define unknown variables
variables (M D : ℕ)

-- Express conditions in Lean
def total_shares_eq : Prop := M + D = TotalShares
def total_value_eq : Prop := MicrotronPrice * M + DynacoPrice * D = TotalValue

-- Define the problem statement
theorem dynaco_shares_sold : ∃ D : ℕ, 
  (∃ M : ℕ, total_shares_eq M D ∧ total_value_eq M D) ∧ D = 150 :=
by
  sorry

end dynaco_shares_sold_l131_131761


namespace estimate_probability_l131_131315

noncomputable def freq_20 : ℝ := 0.300
noncomputable def freq_50 : ℝ := 0.360
noncomputable def freq_100 : ℝ := 0.350
noncomputable def freq_300 : ℝ := 0.350
noncomputable def freq_500 : ℝ := 0.352
noncomputable def freq_1000 : ℝ := 0.351
noncomputable def freq_5000 : ℝ := 0.351

theorem estimate_probability : (|0.35 - ((freq_20 + freq_50 + freq_100 + freq_300 + freq_500 + freq_1000 + freq_5000) / 7)| < 0.01) :=
by sorry

end estimate_probability_l131_131315


namespace base7_sub_base5_to_base10_l131_131776

def base7to10 (n : Nat) : Nat :=
  match n with
  | 52403 => 5 * 7^4 + 2 * 7^3 + 4 * 7^2 + 0 * 7^1 + 3 * 7^0
  | _ => 0

def base5to10 (n : Nat) : Nat :=
  match n with
  | 20345 => 2 * 5^4 + 0 * 5^3 + 3 * 5^2 + 4 * 5^1 + 5 * 5^0
  | _ => 0

theorem base7_sub_base5_to_base10 :
  base7to10 52403 - base5to10 20345 = 11540 :=
by
  sorry

end base7_sub_base5_to_base10_l131_131776


namespace consecutive_odds_base_eqn_l131_131500

-- Given conditions
def isOdd (n : ℕ) : Prop := n % 2 = 1

variables {C D : ℕ}

theorem consecutive_odds_base_eqn (C_odd : isOdd C) (D_odd : isOdd D) (consec : D = C + 2)
    (base_eqn : 2 * C^2 + 4 * C + 3 + 6 * D + 5 = 10 * (C + D) + 7) :
    C + D = 16 :=
sorry

end consecutive_odds_base_eqn_l131_131500


namespace students_enrolled_in_only_english_l131_131423

theorem students_enrolled_in_only_english (total_students both_english_german total_german : ℕ) (h1 : total_students = 40) (h2 : both_english_german = 12) (h3 : total_german = 22) (h4 : ∀ s, s < 40) :
  (total_students - (total_german - both_english_german) - both_english_german) = 18 := 
by {
  sorry
}

end students_enrolled_in_only_english_l131_131423


namespace find_number_l131_131898

theorem find_number (N : ℕ) (h : N / 16 = 16 * 8) : N = 2048 :=
sorry

end find_number_l131_131898


namespace probability_adjacent_vertices_of_octagon_l131_131632

theorem probability_adjacent_vertices_of_octagon :
  let num_vertices := 8;
  let adjacent_vertices (v1 v2 : Fin num_vertices) : Prop := 
    (v2 = (v1 + 1) % num_vertices) ∨ (v2 = (v1 - 1 + num_vertices) % num_vertices);
  let total_vertices := num_vertices - 1;
  (2 : ℚ) / total_vertices = (2 / 7 : ℚ) :=
by
  -- Proof goes here
  sorry

end probability_adjacent_vertices_of_octagon_l131_131632


namespace bf_length_l131_131772

noncomputable def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
  {p | (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1}

noncomputable def focus (a b : ℝ) : ℝ × ℝ :=
  (real.sqrt (a ^ 2 - b ^ 2), 0)

theorem bf_length {a b : ℝ} (h : a = 4) (h1 : b = 3) (AF_length : ℝ) (x1 y1 : ℝ) :
  let F := focus a b in
  let E := ellipse a b in
  AF_length = 2 →
  F ∈ E →
  F = (real.sqrt (a ^ 2 - b ^ 2), 0) →
  ∃ BF_length : ℝ,
    (x1 - real.sqrt (a ^ 2 - b ^ 2)) ^ 2 + y1 ^ 2 = 4 →
    AF_length = 2 →
    BF_length = (real.sqrt ((- x1 - real.sqrt (a ^ 2 - b ^ 2)) ^ 2 + y1 ^ 2)) := 
sorry

end bf_length_l131_131772


namespace dodecagon_diagonals_l131_131802

def numDiagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem dodecagon_diagonals : numDiagonals 12 = 54 := by
  sorry

end dodecagon_diagonals_l131_131802


namespace company_max_revenue_l131_131352

structure Conditions where
  max_total_time : ℕ -- maximum total time in minutes
  max_total_cost : ℕ -- maximum total cost in yuan
  rate_A : ℕ -- rate per minute for TV A in yuan
  rate_B : ℕ -- rate per minute for TV B in yuan
  revenue_A : ℕ -- revenue per minute for TV A in million yuan
  revenue_B : ℕ -- revenue per minute for TV B in million yuan

def company_conditions : Conditions :=
  { max_total_time := 300,
    max_total_cost := 90000,
    rate_A := 500,
    rate_B := 200,
    revenue_A := 3, -- as 0.3 million yuan converted to 3 tenths (integer representation)
    revenue_B := 2  -- as 0.2 million yuan converted to 2 tenths (integer representation)
  }

def advertising_strategy
  (conditions : Conditions)
  (time_A : ℕ) (time_B : ℕ) : Prop :=
  time_A + time_B ≤ conditions.max_total_time ∧
  time_A * conditions.rate_A + time_B * conditions.rate_B ≤ conditions.max_total_cost

def revenue
  (conditions : Conditions)
  (time_A : ℕ) (time_B : ℕ) : ℕ :=
  time_A * conditions.revenue_A + time_B * conditions.revenue_B

theorem company_max_revenue (time_A time_B : ℕ)
  (h : advertising_strategy company_conditions time_A time_B) :
  revenue company_conditions time_A time_B = 70 := 
  by
  have h1 : time_A = 100 := sorry
  have h2 : time_B = 200 := sorry
  sorry

end company_max_revenue_l131_131352


namespace total_peaches_l131_131152

variable (numberOfBaskets : ℕ)
variable (redPeachesPerBasket : ℕ)
variable (greenPeachesPerBasket : ℕ)

theorem total_peaches (h1 : numberOfBaskets = 1) 
                      (h2 : redPeachesPerBasket = 4)
                      (h3 : greenPeachesPerBasket = 3) :
  numberOfBaskets * (redPeachesPerBasket + greenPeachesPerBasket) = 7 := 
by
  sorry

end total_peaches_l131_131152


namespace Tim_weekly_earnings_l131_131616

theorem Tim_weekly_earnings :
  let tasks_per_day := 100
  let pay_per_task := 1.2
  let days_per_week := 6
  let daily_earnings := tasks_per_day * pay_per_task
  let weekly_earnings := daily_earnings * days_per_week
  weekly_earnings = 720 := by
  sorry

end Tim_weekly_earnings_l131_131616


namespace student_arrangement_l131_131892

theorem student_arrangement (students : Fin 6 → Prop)
  (A : (students 0) ∨ (students 5) → False)
  (females_adj : ∃ (i : Fin 6), i < 5 ∧ students i → students (i + 1))
  : ∃! n, n = 96 := by
  sorry

end student_arrangement_l131_131892


namespace Matthew_initial_cakes_l131_131984

theorem Matthew_initial_cakes (n_cakes : ℕ) (n_crackers : ℕ) (n_friends : ℕ) (crackers_per_person : ℕ) :
  n_friends = 4 →
  n_crackers = 32 →
  crackers_per_person = 8 →
  n_crackers = n_friends * crackers_per_person →
  n_cakes = n_friends * crackers_per_person →
  n_cakes = 32 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  rw [h1, h3] at h5
  exact h5

end Matthew_initial_cakes_l131_131984


namespace smallest_number_of_eggs_proof_l131_131015

noncomputable def smallest_number_of_eggs (c : ℕ) : ℕ := 15 * c - 3

theorem smallest_number_of_eggs_proof :
  ∃ c : ℕ, c ≥ 11 ∧ smallest_number_of_eggs c = 162 ∧ smallest_number_of_eggs c > 150 :=
by
  sorry

end smallest_number_of_eggs_proof_l131_131015


namespace cab_income_third_day_l131_131027

noncomputable def cab_driver_income (day1 day2 day3 day4 day5 : ℕ) : ℕ := 
day1 + day2 + day3 + day4 + day5

theorem cab_income_third_day 
  (day1 day2 day4 day5 avg_income total_income day3 : ℕ)
  (h1 : day1 = 45)
  (h2 : day2 = 50)
  (h3 : day4 = 65)
  (h4 : day5 = 70)
  (h_avg : avg_income = 58)
  (h_total : total_income = 5 * avg_income)
  (h_day_sum : day1 + day2 + day4 + day5 = 230) :
  total_income - 230 = 60 :=
sorry

end cab_income_third_day_l131_131027


namespace mimi_shells_l131_131579

theorem mimi_shells (Kyle_shells Mimi_shells Leigh_shells : ℕ) 
  (h₀ : Kyle_shells = 2 * Mimi_shells) 
  (h₁ : Leigh_shells = Kyle_shells / 3) 
  (h₂ : Leigh_shells = 16) 
  : Mimi_shells = 24 := by 
  sorry

end mimi_shells_l131_131579


namespace days_needed_to_wash_all_towels_l131_131711

def towels_per_hour : ℕ := 7
def hours_per_day : ℕ := 2
def total_towels : ℕ := 98

theorem days_needed_to_wash_all_towels :
  (total_towels / (towels_per_hour * hours_per_day)) = 7 :=
by
  sorry

end days_needed_to_wash_all_towels_l131_131711


namespace full_house_plus_two_probability_l131_131737

def total_ways_to_choose_7_cards_from_52 : ℕ :=
  Nat.choose 52 7

def ways_for_full_house_plus_two : ℕ :=
  13 * 4 * 12 * 6 * 55 * 16

def probability_full_house_plus_two : ℚ :=
  (ways_for_full_house_plus_two : ℚ) / (total_ways_to_choose_7_cards_from_52 : ℚ)

theorem full_house_plus_two_probability :
  probability_full_house_plus_two = 13732 / 3344614 :=
by
  sorry

end full_house_plus_two_probability_l131_131737


namespace find_a_l131_131533

-- We define the conditions given in the problem
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- The expression defined as per the problem statement
def expansion_coeff_x2 (a : ℝ) : ℝ :=
  (binom 4 2) * 4 - 2 * (binom 4 1) * (binom 5 1) * a + (binom 5 2) * a^2

-- We now express the proof statement in Lean 4. 
-- We need to prove that given the coefficient of x^2 is -16, then a = 2
theorem find_a (a : ℝ) (h : expansion_coeff_x2 a = -16) : a = 2 :=
  by sorry

end find_a_l131_131533


namespace marble_weight_l131_131302

-- Define the weights of marbles and waffle irons
variables (m w : ℝ)

-- Given conditions
def condition1 : Prop := 9 * m = 4 * w
def condition2 : Prop := 3 * w = 75 

-- The theorem we want to prove
theorem marble_weight (h1 : condition1 m w) (h2 : condition2 w) : m = 100 / 9 :=
by
  sorry

end marble_weight_l131_131302


namespace eval_expression_l131_131514

theorem eval_expression : 15 * 30 + 45 * 15 - 15 * 10 = 975 :=
by 
  sorry

end eval_expression_l131_131514


namespace remaining_lemon_heads_after_eating_l131_131115

-- Assume initial number of lemon heads is given
variables (initial_lemon_heads : ℕ)

-- Patricia eats 15 lemon heads
def remaining_lemon_heads (initial_lemon_heads : ℕ) : ℕ :=
  initial_lemon_heads - 15

theorem remaining_lemon_heads_after_eating :
  ∀ (initial_lemon_heads : ℕ), remaining_lemon_heads initial_lemon_heads = initial_lemon_heads - 15 :=
by
  intros
  rfl

end remaining_lemon_heads_after_eating_l131_131115


namespace more_likely_to_return_to_initial_count_l131_131583

noncomputable def P_A (a b c d : ℕ) : ℚ :=
(b * (d + 1) + a * (c + 1)) / (50 * 51)

noncomputable def P_A_bar (a b c d : ℕ) : ℚ :=
(b * c + a * d) / (50 * 51)

theorem more_likely_to_return_to_initial_count (a b c d : ℕ) (h1 : a + b = 50) (h2 : c + d = 50) 
  (h3 : b ≥ a) (h4 : d ≥ c - 1) (h5 : a > 0) :
P_A a b c d > P_A_bar a b c d := by
  sorry

end more_likely_to_return_to_initial_count_l131_131583


namespace doughnuts_per_box_l131_131581

theorem doughnuts_per_box (total_doughnuts : ℕ) (boxes : ℕ) (h_doughnuts : total_doughnuts = 48) (h_boxes : boxes = 4) : 
  total_doughnuts / boxes = 12 :=
by
  -- This is a placeholder for the proof
  sorry

end doughnuts_per_box_l131_131581


namespace total_red_stripes_l131_131568

theorem total_red_stripes 
  (flagA_stripes : ℕ := 30) 
  (flagB_stripes : ℕ := 45) 
  (flagC_stripes : ℕ := 60)
  (flagA_count : ℕ := 20) 
  (flagB_count : ℕ := 30) 
  (flagC_count : ℕ := 40)
  (flagA_red : ℕ := 15)
  (flagB_red : ℕ := 15)
  (flagC_red : ℕ := 14) : 
  300 + 450 + 560 = 1310 := 
by
  have flagA_red_stripes : 15 = 15 := by rfl
  have flagB_red_stripes : 15 = 15 := by rfl
  have flagC_red_stripes : 14 = 14 := by rfl
  have total_A_red_stripes : 15 * 20 = 300 := by norm_num
  have total_B_red_stripes : 15 * 30 = 450 := by norm_num
  have total_C_red_stripes : 14 * 40 = 560 := by norm_num
  exact add_assoc 300 450 560 ▸ rfl

end total_red_stripes_l131_131568


namespace picture_size_l131_131987

theorem picture_size (total_pics_A : ℕ) (size_A : ℕ) (total_pics_B : ℕ) (C : ℕ)
  (hA : total_pics_A * size_A = C) (hB : total_pics_B = 3000) : 
  (C / total_pics_B = 8) :=
by
  sorry

end picture_size_l131_131987


namespace rowing_upstream_speed_l131_131485

-- Define the speed of the man in still water
def V_m : ℝ := 45

-- Define the speed of the man rowing downstream
def V_downstream : ℝ := 65

-- Define the speed of the stream
def V_s : ℝ := V_downstream - V_m

-- Define the speed of the man rowing upstream
def V_upstream : ℝ := V_m - V_s

-- Prove that the speed of the man rowing upstream is 25 kmph
theorem rowing_upstream_speed :
  V_upstream = 25 := by
  sorry

end rowing_upstream_speed_l131_131485


namespace cost_first_third_hour_l131_131565

theorem cost_first_third_hour 
  (c : ℝ) 
  (h1 : 0 < c) 
  (h2 : ∀ t : ℝ, t > 1/4 → (t - 1/4) * 12 + c = 31)
  : c = 5 :=
by
  sorry

end cost_first_third_hour_l131_131565


namespace boat_capacity_problem_l131_131455

variables (L S : ℕ)

theorem boat_capacity_problem
  (h1 : L + 4 * S = 46)
  (h2 : 2 * L + 3 * S = 57) :
  3 * L + 6 * S = 96 :=
sorry

end boat_capacity_problem_l131_131455


namespace find_S3_l131_131875

noncomputable def geometric_sum (n : ℕ) : ℕ := sorry  -- Placeholder for the sum function.

theorem find_S3 (S : ℕ → ℕ) (hS6 : S 6 = 30) (hS9 : S 9 = 70) : S 3 = 10 :=
by
  -- Establish the needed conditions and equation 
  have h : (S 6 - S 3) ^ 2 = (S 9 - S 6) * S 3 := sorry
  -- Substitute given S6 and S9 into the equation and solve
  exact sorry

end find_S3_l131_131875


namespace adults_tickets_sold_eq_1200_l131_131855

variable (A : ℕ)
variable (S : ℕ := 300) -- Number of student tickets
variable (P_adult : ℕ := 12) -- Price per adult ticket
variable (P_student : ℕ := 6) -- Price per student ticket
variable (total_tickets : ℕ := 1500) -- Total tickets sold
variable (total_amount : ℕ := 16200) -- Total amount collected

theorem adults_tickets_sold_eq_1200
  (h1 : S = 300)
  (h2 : A + S = total_tickets)
  (h3 : P_adult * A + P_student * S = total_amount) :
  A = 1200 := by
  sorry

end adults_tickets_sold_eq_1200_l131_131855


namespace max_area_rectangle_l131_131178

theorem max_area_rectangle (P : ℝ) (x : ℝ) (h1 : P = 40) (h2 : 6 * x = P) : 
  2 * (x ^ 2) = 800 / 9 :=
by
  sorry

end max_area_rectangle_l131_131178


namespace letter_puzzle_solution_l131_131219

theorem letter_puzzle_solution :
  ∃ (A B : ℕ), (A ≠ B) ∧ (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ A ^ B = B * 10 + A :=
by {
  use 2, 5,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 2 ^ 5 = 5 * 10 + 2),
  sorry
} ∨
by {
  use 6, 2,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 6 ^ 2 = 2 * 10 + 6),
  sorry
} ∨
by {
  use 4, 3,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 4 ^ 3 = 3 * 10 + 4),
  sorry
}

end letter_puzzle_solution_l131_131219


namespace regular_polygon_sides_l131_131858

-- Define the central angle and number of sides of a regular polygon
def central_angle (θ : ℝ) := θ = 30
def number_of_sides (n : ℝ) := 360 / 30 = n

-- Theorem to prove that the number of sides of the regular polygon is 12 given the central angle is 30 degrees
theorem regular_polygon_sides (θ n : ℝ) (hθ : central_angle θ) : number_of_sides n → n = 12 :=
sorry

end regular_polygon_sides_l131_131858


namespace problem_part1_problem_part2_l131_131080

-- Statement for Part (1)
theorem problem_part1 (a b : ℝ) (h_sol : {x : ℝ | ax^2 - 3 * x + 2 > 0} = {x : ℝ | x < 1 ∨ x > 2}) :
  a = 1 ∧ b = 2 := sorry

-- Statement for Part (2)
theorem problem_part2 (x y k : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) 
  (h_eq : (1 / x) + (2 / y) = 1) (h_ineq : 2 * x + y ≥ k^2 + k + 2) :
  -3 ≤ k ∧ k ≤ 2 := sorry

end problem_part1_problem_part2_l131_131080


namespace triangle_inequality_proof_l131_131995

theorem triangle_inequality_proof 
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end triangle_inequality_proof_l131_131995


namespace find_x_l131_131520

-- Define that x is a real number and positive
variable (x : ℝ)
variable (hx_pos : 0 < x)

-- Define the floor function and the main equation
variable (hx_eq : ⌊x⌋ * x = 90)

theorem find_x (h : ⌊x⌋ * x = 90) (hx_pos : 0 < x) : ⌊x⌋ = 9 ∧ x = 10 :=
by
  sorry

end find_x_l131_131520


namespace ratio_of_areas_l131_131319

theorem ratio_of_areas (b : ℝ) (h1 : 0 < b) (h2 : b < 4) 
  (h3 : (9 : ℝ) / 25 = (4 - b) / b * (4 : ℝ)) : b = 2.5 := 
sorry

end ratio_of_areas_l131_131319


namespace height_difference_l131_131847

def burj_khalifa_height : ℝ := 830
def sears_tower_height : ℝ := 527

theorem height_difference : burj_khalifa_height - sears_tower_height = 303 := 
by
  sorry

end height_difference_l131_131847


namespace sum_of_integers_l131_131861

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 :=
sorry

end sum_of_integers_l131_131861


namespace sum_of_nonnegative_numbers_eq_10_l131_131420

theorem sum_of_nonnegative_numbers_eq_10 (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 48)
  (h2 : ab + bc + ca = 26)
  (h3 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) : a + b + c = 10 := 
by
  sorry

end sum_of_nonnegative_numbers_eq_10_l131_131420


namespace central_angle_eq_one_l131_131073

noncomputable def radian_measure_of_sector (α r : ℝ) : Prop :=
  α * r = 2 ∧ (1 / 2) * α * r^2 = 2

-- Theorem stating the radian measure of the central angle is 1
theorem central_angle_eq_one (α r : ℝ) (h : radian_measure_of_sector α r) : α = 1 :=
by
  -- provide proof steps here
  sorry

end central_angle_eq_one_l131_131073


namespace time_comparison_l131_131494

variable (s : ℝ) (h_pos : s > 0)

noncomputable def t1 : ℝ := 120 / s
noncomputable def t2 : ℝ := 480 / (4 * s)

theorem time_comparison : t1 s = t2 s := by
  rw [t1, t2]
  field_simp [h_pos]
  norm_num
  sorry

end time_comparison_l131_131494


namespace negation_of_p_l131_131431

variable (p : Prop) (n : ℕ)

def proposition_p := ∃ n : ℕ, n^2 > 2^n

theorem negation_of_p : ¬ proposition_p ↔ ∀ n : ℕ, n^2 <= 2^n :=
by
  sorry

end negation_of_p_l131_131431


namespace arithmetic_sequence_eightieth_term_l131_131865

open BigOperators

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem arithmetic_sequence_eightieth_term :
  ∀ (d : ℝ),
  arithmetic_sequence 3 d 21 = 41 →
  arithmetic_sequence 3 d 80 = 153.1 :=
by
  intros
  sorry

end arithmetic_sequence_eightieth_term_l131_131865


namespace Tim_weekly_earnings_l131_131618

theorem Tim_weekly_earnings :
  let tasks_per_day := 100
  let pay_per_task := 1.2
  let days_per_week := 6
  let daily_earnings := tasks_per_day * pay_per_task
  let weekly_earnings := daily_earnings * days_per_week
  weekly_earnings = 720 := by
  sorry

end Tim_weekly_earnings_l131_131618


namespace quadratic_equation_solution_l131_131725

theorem quadratic_equation_solution : ∀ x : ℝ, x^2 - 9 = 0 ↔ (x = 3 ∨ x = -3) :=
by
  sorry

end quadratic_equation_solution_l131_131725


namespace intern_knows_same_number_l131_131589

theorem intern_knows_same_number (n : ℕ) (h : n > 1) : 
  ∃ (a b : fin n), a ≠ b ∧ 
  ∃ (f : fin n → ℕ), f a = f b ∧ ∀ i, 0 ≤ f i ∧ f i < n - 1 :=
begin
  sorry,
end

end intern_knows_same_number_l131_131589


namespace combined_PP_curve_l131_131606

-- Definitions based on the given conditions
def M1 (K : ℝ) : ℝ := 40 - 2 * K
def M2 (K : ℝ) : ℝ := 64 - K ^ 2
def combinedPPC (K1 K2 : ℝ) : ℝ := 128 - 0.5 * K1^2 + 40 - 2 * K2

theorem combined_PP_curve (K : ℝ) :
  (K ≤ 2 → combinedPPC K 0 = 168 - 0.5 * K^2) ∧
  (2 < K ∧ K ≤ 22 → combinedPPC 2 (K - 2) = 170 - 2 * K) ∧
  (22 < K ∧ K ≤ 36 → combinedPPC (K - 20) 20 = 20 * K - 0.5 * K^2 - 72) :=
by
  sorry

end combined_PP_curve_l131_131606


namespace find_number_of_students_l131_131744

theorem find_number_of_students (N : ℕ) (T : ℕ) (hN : N ≠ 0) (hT : T = 80 * N) 
  (h_avg_excluded : (T - 200) / (N - 5) = 90) : N = 25 :=
by
  sorry

end find_number_of_students_l131_131744


namespace smallest_coprime_gt_one_l131_131233

theorem smallest_coprime_gt_one (x : ℕ) (h1 : 1 < x) (h2 : Nat.gcd x 180 = 1) : x = 7 := sorry

end smallest_coprime_gt_one_l131_131233


namespace evaluate_seventy_five_squared_minus_twenty_five_squared_l131_131515

theorem evaluate_seventy_five_squared_minus_twenty_five_squared :
  75^2 - 25^2 = 5000 :=
by
  sorry

end evaluate_seventy_five_squared_minus_twenty_five_squared_l131_131515


namespace probability_no_defective_pencils_l131_131021

theorem probability_no_defective_pencils :
  let total_pencils := 9
  let defective_pencils := 2
  let total_ways_choose_3 := Nat.choose total_pencils 3
  let non_defective_pencils := total_pencils - defective_pencils
  let ways_choose_3_non_defective := Nat.choose non_defective_pencils 3
  (ways_choose_3_non_defective : ℚ) / total_ways_choose_3 = 5 / 12 :=
by
  sorry

end probability_no_defective_pencils_l131_131021


namespace letter_puzzle_solutions_l131_131211

noncomputable def is_solution (A B : ℕ) : Prop :=
A ≠ B ∧ A ∈ finset.range (10) ∧ B ∈ finset.range (10) ∧ 10 ≤ B * 10 + A ∧ B * 10 + A ≤ 99 ∧ A^B = B * 10 + A

theorem letter_puzzle_solutions :
  ∃ A B : ℕ, is_solution A B ∧ ((A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3)) :=
by
  sorry

end letter_puzzle_solutions_l131_131211


namespace fill_tank_time_l131_131439

-- Define the rates at which the pipes fill or empty the tank
def rateA : ℚ := 1 / 16
def rateB : ℚ := - (1 / 24)  -- Since pipe B empties the tank, it's negative.

-- Define the time after which pipe B is closed
def timeBClosed : ℚ := 21

-- Define the initial combined rate of both pipes
def combinedRate : ℚ := rateA + rateB

-- Define the proportion of the tank filled in the initial 21 minutes
def filledIn21Minutes : ℚ := combinedRate * timeBClosed

-- Define the remaining tank to be filled after pipe B is closed
def remainingTank : ℚ := 1 - filledIn21Minutes

-- Define the additional time required to fill the remaining part of the tank with only pipe A
def additionalTime : ℚ := remainingTank / rateA

-- Total time is the sum of the initial time and additional time
def totalTime : ℚ := timeBClosed + additionalTime

theorem fill_tank_time : totalTime = 30 :=
by
  -- Proof omitted
  sorry

end fill_tank_time_l131_131439


namespace cannot_determine_exact_insect_l131_131171

-- Defining the conditions as premises
def insect_legs : ℕ := 6

def total_legs_two_insects (legs_per_insect : ℕ) (num_insects : ℕ) : ℕ :=
  legs_per_insect * num_insects

-- Statement: Proving that given just the number of legs, we cannot determine the exact type of insect
theorem cannot_determine_exact_insect (legs : ℕ) (num_insects : ℕ) (h1 : legs = 6) (h2 : num_insects = 2) (h3 : total_legs_two_insects legs num_insects = 12) :
  ∃ insect_type, insect_type :=
by
  sorry

end cannot_determine_exact_insect_l131_131171


namespace correct_operation_l131_131163

/-- Proving that among the given mathematical operations, only the second option is correct. -/
theorem correct_operation (m : ℝ) : ¬ (m^3 - m^2 = m) ∧ (3 * m^2 * 2 * m^3 = 6 * m^5) ∧ ¬ (3 * m^2 + 2 * m^3 = 5 * m^5) ∧ ¬ ((2 * m^2)^3 = 8 * m^5) :=
by
  -- These are the conditions, proof is omitted using sorry
  sorry

end correct_operation_l131_131163


namespace c_share_l131_131475

theorem c_share (x : ℕ) (a b c d : ℕ) 
  (h1: a = 5 * x)
  (h2: b = 3 * x)
  (h3: c = 2 * x)
  (h4: d = 3 * x)
  (h5: a = b + 1000): 
  c = 1000 := 
by 
  sorry

end c_share_l131_131475


namespace avg_annual_growth_rate_l131_131192
-- Import the Mathlib library

-- Define the given conditions
def initial_income : ℝ := 32000
def final_income : ℝ := 37000
def period : ℝ := 2
def initial_income_ten_thousands : ℝ := initial_income / 10000
def final_income_ten_thousands : ℝ := final_income / 10000

-- Define the growth rate
variable (x : ℝ)

-- Define the theorem
theorem avg_annual_growth_rate :
  3.2 * (1 + x) ^ 2 = 3.7 :=
sorry

end avg_annual_growth_rate_l131_131192


namespace probability_heads_equals_7_over_11_l131_131977

theorem probability_heads_equals_7_over_11 (p : ℝ) (q : ℝ)
  (h1 : q = 1 - p)
  (h2 : 120 * p^7 * q^3 = 210 * p^6 * q^4) :
  p = 7 / 11 :=
by {
  sorry
}

end probability_heads_equals_7_over_11_l131_131977


namespace gcd_fact8_fact7_l131_131226

noncomputable def fact8 : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
noncomputable def fact7 : ℕ := 7 * 6 * 5 * 4 * 3 * 2 * 1

theorem gcd_fact8_fact7 : Nat.gcd fact8 fact7 = fact7 := by
  unfold fact8 fact7
  exact sorry

end gcd_fact8_fact7_l131_131226


namespace division_of_neg6_by_3_l131_131367

theorem division_of_neg6_by_3 : (-6 : ℤ) / 3 = -2 := 
by
  sorry

end division_of_neg6_by_3_l131_131367


namespace exist_positive_real_x_l131_131523

theorem exist_positive_real_x (x : ℝ) (hx1 : 0 < x) (hx2 : Nat.floor x * x = 90) : x = 10 := 
sorry

end exist_positive_real_x_l131_131523


namespace range_of_a_l131_131262

variables {f : ℝ → ℝ} (a : ℝ)

-- Even function definition
def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

-- Monotonically increasing on (-∞, 0)
def mono_increasing_on_neg (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → y < 0 → f x ≤ f y

-- Problem statement
theorem range_of_a
  (h_even : even_function f)
  (h_mono_neg : mono_increasing_on_neg f)
  (h_inequality : f (2 ^ |a - 1|) > f 4) :
  -1 < a ∧ a < 3 :=
sorry

end range_of_a_l131_131262


namespace true_statements_count_l131_131994

def reciprocal (n : ℕ) : ℚ := 1 / n

theorem true_statements_count :
  let s1 := reciprocal 4 + reciprocal 8 = reciprocal 12
  let s2 := reciprocal 8 - reciprocal 5 = reciprocal 3
  let s3 := reciprocal 3 * reciprocal 9 = reciprocal 27
  let s4 := reciprocal 15 / reciprocal 3 = reciprocal 5
  (if s1 then 1 else 0) + 
  (if s2 then 1 else 0) + 
  (if s3 then 1 else 0) + 
  (if s4 then 1 else 0) = 2 :=
by
  sorry

end true_statements_count_l131_131994


namespace max_length_segment_l131_131159

theorem max_length_segment (p b : ℝ) (h : b = p / 2) : (b * (p - b)) / p = p / 4 :=
by
  sorry

end max_length_segment_l131_131159


namespace dodecagon_diagonals_l131_131803

def numDiagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem dodecagon_diagonals : numDiagonals 12 = 54 := by
  sorry

end dodecagon_diagonals_l131_131803


namespace shuai_fen_ratio_l131_131149

theorem shuai_fen_ratio 
  (C : ℕ) (B_and_D : ℕ) (a : ℕ) (x : ℚ) 
  (hC : C = 36) (hB_and_D : B_and_D = 75) :
  (x = 0.25) ∧ (a = 175) := 
by {
  -- This is where the proof steps would go
  sorry
}

end shuai_fen_ratio_l131_131149


namespace range_of_g_l131_131296

noncomputable def g (x : ℝ) : ℤ :=
if x > -3 then
  ⌈1 / ((x + 3)^2)⌉
else
  ⌊1 / ((x + 3)^2)⌋

theorem range_of_g :
  ∀ y : ℤ, (∃ x : ℝ, g x = y) ↔ (∃ n : ℕ, y = n + 1) :=
by sorry

end range_of_g_l131_131296


namespace weight_of_new_person_l131_131135

-- Define the problem conditions
variables (W : ℝ) -- Weight of the new person
variable (initial_weight : ℝ := 65) -- Weight of the person being replaced
variable (increase_in_avg : ℝ := 4) -- Increase in average weight
variable (num_persons : ℕ := 8) -- Number of persons

-- Define the total increase in weight due to the new person
def total_increase : ℝ := num_persons * increase_in_avg

-- The Lean statement to prove
theorem weight_of_new_person (W : ℝ) (h : total_increase = W - initial_weight) : W = 97 := sorry

end weight_of_new_person_l131_131135


namespace total_bedrooms_is_correct_l131_131644

def bedrooms_second_floor : Nat := 2
def bedrooms_first_floor : Nat := 8
def total_bedrooms (b1 b2 : Nat) : Nat := b1 + b2

theorem total_bedrooms_is_correct : total_bedrooms bedrooms_second_floor bedrooms_first_floor = 10 := 
by
  sorry

end total_bedrooms_is_correct_l131_131644


namespace part_a_area_of_square_l131_131742

theorem part_a_area_of_square {s : ℝ} (h : s = 9) : s ^ 2 = 81 := 
sorry

end part_a_area_of_square_l131_131742


namespace smallest_rel_prime_to_180_l131_131246

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ x ≤ 7 ∧ (∀ y : ℕ, y > 1 ∧ y < x → y.gcd 180 ≠ 1) ∧ x.gcd 180 = 1 :=
by
  sorry

end smallest_rel_prime_to_180_l131_131246


namespace positive_integer_solutions_l131_131516

theorem positive_integer_solutions :
  ∀ (a b c : ℕ), (8 * a - 5 * b)^2 + (3 * b - 2 * c)^2 + (3 * c - 7 * a)^2 = 2 → 
  (a = 3 ∧ b = 5 ∧ c = 7) ∨ (a = 12 ∧ b = 19 ∧ c = 28) :=
by
  sorry

end positive_integer_solutions_l131_131516


namespace turnips_in_mashed_potatoes_l131_131648

theorem turnips_in_mashed_potatoes:
  ∀ (turnips_prev potatoes_prev : ℝ) (potatoes_curr : ℝ),
    turnips_prev = 2 →
    potatoes_prev = 5 →
    potatoes_curr = 20 →
    (potatoes_curr / (potatoes_prev / turnips_prev) = 8) :=
begin
  intros,
  sorry,
end

end turnips_in_mashed_potatoes_l131_131648


namespace geometric_sequence_a6_l131_131281

theorem geometric_sequence_a6 (a : ℕ → ℝ) (geometric_seq : ∀ n, a (n + 1) = a n * a 1)
  (h1 : (a 4) * (a 8) = 9) (h2 : (a 4) + (a 8) = -11) : a 6 = -3 := by
  sorry

end geometric_sequence_a6_l131_131281


namespace rectangle_height_l131_131561

-- Define the given right-angled triangle with its legs and hypotenuse
variables {a b c d : ℝ}

-- Define the conditions: Right-angled triangle with legs a, b and hypotenuse c
def right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Define the height of the inscribed rectangle is d
def height_of_rectangle (a b d : ℝ) : Prop :=
  d = a + b

-- The problem statement: Prove that the height of the rectangle is the sum of the heights of the squares
theorem rectangle_height (a b c d : ℝ) (ht : right_angled_triangle a b c) : height_of_rectangle a b d :=
by
  sorry

end rectangle_height_l131_131561


namespace area_of_largest_circle_l131_131181

theorem area_of_largest_circle (side_length : ℝ) (h : side_length = 2) : 
  (Real.pi * (side_length / 2)^2 = 3.14) :=
by
  sorry

end area_of_largest_circle_l131_131181


namespace technicians_count_l131_131443

def avg_salary_all := 9500
def avg_salary_technicians := 12000
def avg_salary_rest := 6000
def total_workers := 12

theorem technicians_count : 
  ∃ (T R : ℕ), 
  (T + R = total_workers) ∧ 
  ((T * avg_salary_technicians + R * avg_salary_rest) / total_workers = avg_salary_all) ∧ 
  (T = 7) :=
by sorry

end technicians_count_l131_131443


namespace part_I_part_II_l131_131790

noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| + |2 * x - 1|

theorem part_I (x : ℝ) : 
  (f x > f 1) ↔ (x < -3/2 ∨ x > 1) :=
by
  -- We leave the proof as sorry to indicate it needs to be filled in
  sorry

theorem part_II (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x : ℝ, f x ≥ 1/m + 1/n) → m + n ≥ 4/3 :=
by
  -- We leave the proof as sorry to indicate it needs to be filled in
  sorry

end part_I_part_II_l131_131790


namespace polygon_sides_l131_131361

theorem polygon_sides (n : ℕ) (h : n ≥ 3) (sum_angles : (n - 2) * 180 = 1620) :
  n = 10 ∨ n = 11 ∨ n = 12 :=
sorry

end polygon_sides_l131_131361


namespace find_number_of_valid_subsets_l131_131227

noncomputable def numValidSubsets : ℕ :=
  let count := 25.choose 9
  count

theorem find_number_of_valid_subsets :
  numValidSubsets = 177100 := by
  sorry

end find_number_of_valid_subsets_l131_131227


namespace pagoda_top_story_lanterns_l131_131957

/--
Given a 7-story pagoda where each story has twice as many lanterns as the one above it, 
and a total of 381 lanterns across all stories, prove the number of lanterns on the top (7th) story is 3.
-/
theorem pagoda_top_story_lanterns (a : ℕ) (n : ℕ) (r : ℚ) (sum_lanterns : ℕ) :
  n = 7 → r = 1 / 2 → sum_lanterns = 381 →
  (a * (1 - r^n) / (1 - r) = sum_lanterns) → (a * r^(n - 1) = 3) :=
by
  intros h_n h_r h_sum h_geo_sum
  let a_val := 192 -- from the solution steps
  rw [h_n, h_r, h_sum] at h_geo_sum
  have h_a : a = a_val := by sorry
  rw [h_a, h_n, h_r]
  exact sorry

end pagoda_top_story_lanterns_l131_131957


namespace woman_wait_time_l131_131343
noncomputable def time_for_man_to_catch_up (man_speed woman_speed distance: ℝ) : ℝ :=
  distance / man_speed

theorem woman_wait_time 
    (man_speed : ℝ)
    (woman_speed : ℝ)
    (wait_time_minutes : ℝ) 
    (woman_time : ℝ)
    (distance : ℝ)
    (man_time : ℝ) :
    man_speed = 5 -> 
    woman_speed = 15 -> 
    wait_time_minutes = 2 -> 
    woman_time = woman_speed * (1 / 60) * wait_time_minutes -> 
    woman_time = distance -> 
    man_speed * (1 / 60) = 0.0833 -> 
    man_time = distance / 0.0833 -> 
    man_time = 6 :=
by
  intros
  sorry

end woman_wait_time_l131_131343


namespace area_of_region_l131_131207

-- Definitions from the problem's conditions.
def equation (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 10*y = -9

-- Statement of the theorem.
theorem area_of_region : 
  ∃ (area : ℝ), (∀ x y : ℝ, equation x y → True) ∧ area = 32 * Real.pi :=
by
  sorry

end area_of_region_l131_131207


namespace smallest_number_of_eggs_proof_l131_131016

noncomputable def smallest_number_of_eggs (c : ℕ) : ℕ := 15 * c - 3

theorem smallest_number_of_eggs_proof :
  ∃ c : ℕ, c ≥ 11 ∧ smallest_number_of_eggs c = 162 ∧ smallest_number_of_eggs c > 150 :=
by
  sorry

end smallest_number_of_eggs_proof_l131_131016


namespace eval_g_at_8_l131_131814

def g (x : ℚ) : ℚ := (3 * x + 2) / (x - 2)

theorem eval_g_at_8 : g 8 = 13 / 3 := by
  sorry

end eval_g_at_8_l131_131814


namespace g_at_8_eq_13_over_3_l131_131815

def g (x : ℝ) : ℝ := (3 * x + 2) / (x - 2)

theorem g_at_8_eq_13_over_3 : g 8 = 13 / 3 := by
  sorry

end g_at_8_eq_13_over_3_l131_131815


namespace ratio_b_to_c_l131_131473

variable (a b c k : ℕ)

-- Conditions
def condition1 : Prop := a = b + 2
def condition2 : Prop := b = k * c
def condition3 : Prop := a + b + c = 32
def condition4 : Prop := b = 12

-- Question: Prove that ratio of b to c is 2:1
theorem ratio_b_to_c
  (h1 : condition1 a b)
  (h2 : condition2 b k c)
  (h3 : condition3 a b c)
  (h4 : condition4 b) :
  b = 2 * c := 
sorry

end ratio_b_to_c_l131_131473


namespace smallest_rel_prime_to_180_l131_131239

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, (y > 1 ∧ Nat.gcd y 180 = 1) → x ≤ y :=
begin
  sorry
end

end smallest_rel_prime_to_180_l131_131239


namespace number_of_women_l131_131453

variable (W : ℕ) (x : ℝ)

-- Conditions
def daily_wage_men_and_women (W : ℕ) (x : ℝ) : Prop :=
  24 * 350 + W * x = 11600

def half_men_and_37_women (W : ℕ) (x : ℝ) : Prop :=
  12 * 350 + 37 * x = 24 * 350 + W * x

def daily_wage_man := (350 : ℝ)

-- Proposition to prove
theorem number_of_women (W : ℕ) (x : ℝ) (h1 : daily_wage_men_and_women W x)
  (h2 : half_men_and_37_women W x) : W = 16 := 
  by
  sorry

end number_of_women_l131_131453


namespace max_gcd_11n_3_6n_1_l131_131908

theorem max_gcd_11n_3_6n_1 : ∃ n : ℕ+, ∀ k : ℕ+,  11 * n + 3 = 7 * k + 1 ∧ 6 * n + 1 = 7 * k + 2 → ∃ d : ℕ, d = Nat.gcd (11 * n + 3) (6 * n + 1) ∧ d = 7 :=
by
  sorry

end max_gcd_11n_3_6n_1_l131_131908


namespace prove_inequality_l131_131396

theorem prove_inequality
  (a b c d : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : c > 0)
  (h₃ : d > 0)
  (h₄ : a ≤ b)
  (h₅ : b ≤ c)
  (h₆ : c ≤ d)
  (h₇ : a + b + c + d ≥ 1) :
  a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 ≥ 1 :=
by
  sorry

end prove_inequality_l131_131396


namespace divisor_value_l131_131886

theorem divisor_value :
  ∃ D : ℕ, 
    (242 % D = 11) ∧
    (698 % D = 18) ∧
    (365 % D = 15) ∧
    (527 % D = 13) ∧
    ((242 + 698 + 365 + 527) % D = 9) ∧
    (D = 48) :=
sorry

end divisor_value_l131_131886


namespace roger_current_money_l131_131304

noncomputable def roger_initial_money : ℕ := 16
noncomputable def roger_birthday_money : ℕ := 28
noncomputable def roger_game_spending : ℕ := 25

theorem roger_current_money : roger_initial_money + roger_birthday_money - roger_game_spending = 19 := by
  sorry

end roger_current_money_l131_131304


namespace percentage_difference_l131_131945

theorem percentage_difference : 0.70 * 100 - 0.60 * 80 = 22 := 
by
  sorry

end percentage_difference_l131_131945


namespace length_of_train_l131_131040

variable (L : ℝ) (S : ℝ)

-- Condition 1: The train crosses a 120 meters platform in 15 seconds
axiom condition1 : S = (L + 120) / 15

-- Condition 2: The train crosses a 250 meters platform in 20 seconds
axiom condition2 : S = (L + 250) / 20

-- The theorem to be proved
theorem length_of_train : L = 270 :=
by
  sorry

end length_of_train_l131_131040


namespace bianca_total_pictures_l131_131765

def album1_pictures : Nat := 27
def album2_3_4_pictures : Nat := 3 * 2

theorem bianca_total_pictures : album1_pictures + album2_3_4_pictures = 33 := by
  sorry

end bianca_total_pictures_l131_131765


namespace largest_tan_B_l131_131960

-- The context of the problem involves a triangle with given side lengths
variables (ABC : Triangle) -- A triangle ABC

-- Define the lengths of sides AB and BC
variables (AB BC : ℝ) 
-- Define the value of tan B
variable (tanB : ℝ)

-- The given conditions
def condition_1 := AB = 25
def condition_2 := BC = 20

-- Define the actual statement we need to prove
theorem largest_tan_B (ABC : Triangle) (AB BC tanB : ℝ) : 
  AB = 25 → BC = 20 → tanB = 3 / 4 := sorry

end largest_tan_B_l131_131960


namespace hex_B3F_to_decimal_l131_131204

-- Define the hexadecimal values of B, 3, F
def hex_B : ℕ := 11
def hex_3 : ℕ := 3
def hex_F : ℕ := 15

-- Prove the conversion of B3F_{16} to a base 10 integer equals 2879
theorem hex_B3F_to_decimal : (hex_B * 16^2 + hex_3 * 16^1 + hex_F * 16^0) = 2879 := 
by 
  -- calculation details skipped
  sorry

end hex_B3F_to_decimal_l131_131204


namespace max_m_value_l131_131527

theorem max_m_value (a b : ℝ) (m : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : ∀ a b, 0 < a → 0 < b → (m / (3 * a + b) - 3 / a - 1 / b ≤ 0)) :
  m ≤ 16 :=
sorry

end max_m_value_l131_131527


namespace sum_roots_of_quadratic_l131_131788

theorem sum_roots_of_quadratic (a b : ℝ) (h₁ : a^2 - a - 6 = 0) (h₂ : b^2 - b - 6 = 0) (h₃ : a ≠ b) :
  a + b = 1 :=
sorry

end sum_roots_of_quadratic_l131_131788


namespace sequence_general_term_l131_131669

/-- 
  Define the sequence a_n recursively as:
  a_1 = 2
  a_n = 2 * a_(n-1) - 1

  Prove that the general term of the sequence is:
  a_n = 2^(n-1) + 1
-/
theorem sequence_general_term {a : ℕ → ℕ} 
  (h₁ : a 1 = 2) 
  (h₂ : ∀ n, a (n + 1) = 2 * a n - 1) 
  (n : ℕ) : 
  a n = 2^(n-1) + 1 := by
  sorry

end sequence_general_term_l131_131669


namespace odd_nat_existence_l131_131088

theorem odd_nat_existence (a b : ℕ) (h1 : a % 2 = 1) (h2 : b % 2 = 1) (n : ℕ) :
  ∃ m : ℕ, (a^m * b^2 - 1) % 2^n = 0 ∨ (b^m * a^2 - 1) % 2^n = 0 := 
by
  sorry

end odd_nat_existence_l131_131088


namespace min_shaded_triangles_l131_131760

-- Definitions (conditions) directly from the problem
def Triangle (n : ℕ) := { x : ℕ // x ≤ n }
def side_length := 8
def smaller_side_length := 1

-- Goal (question == correct answer)
theorem min_shaded_triangles : ∃ (shaded : ℕ), shaded = 15 :=
by {
  sorry
}

end min_shaded_triangles_l131_131760


namespace area_of_circle_below_line_l131_131463

theorem area_of_circle_below_line (x y : ℝ) :
  (x - 3)^2 + (y - 5)^2 = 9 →
  y ≤ 8 →
  ∃ (A : ℝ), A = 9 * Real.pi :=
sorry

end area_of_circle_below_line_l131_131463


namespace waiter_earnings_l131_131889

theorem waiter_earnings
  (total_customers : ℕ)
  (no_tip_customers : ℕ)
  (tip_amount : ℕ)
  (customers_tipped : total_customers - no_tip_customers = 3)
  (tips_per_customer : tip_amount = 9) :
  (total_customers - no_tip_customers) * tip_amount = 27 := by
  sorry

end waiter_earnings_l131_131889


namespace length_of_intersection_segment_l131_131289

-- Definitions
def line_theta_eq_pi_over_3 (theta: ℝ) (rho: ℝ) : Prop := theta = π / 3
def circle_polar_equation (theta: ℝ) (rho: ℝ) : Prop := rho = 4 * cos theta + 4 * (sqrt 3) * sin theta

-- Problem Statement
theorem length_of_intersection_segment : 
  ∀ A B : EuclideanSpace ℝ (Fin 2), 
  (∃ (rho_A theta_A rho_B theta_B: ℝ), line_theta_eq_pi_over_3 theta_A rho_A ∧ circle_polar_equation theta_A rho_A ∧ A = ⟨rho_A * cos theta_A, rho_A * sin theta_A⟩ ∧
  line_theta_eq_pi_over_3 theta_B rho_B ∧ circle_polar_equation theta_B rho_B ∧ B = ⟨rho_B * cos theta_B, rho_B * sin theta_B⟩) →
  dist A B = 8 :=
by
  sorry

end length_of_intersection_segment_l131_131289


namespace find_number_l131_131768

theorem find_number (number : ℝ) : 469138 * number = 4690910862 → number = 10000.1 :=
by
  sorry

end find_number_l131_131768


namespace sufficient_but_not_necessary_l131_131654

theorem sufficient_but_not_necessary (x : ℝ) (h : 1 / x < 1 / 2) : x > 2 ∨ x < 0 :=
by
  sorry

end sufficient_but_not_necessary_l131_131654


namespace bill_profit_difference_l131_131912

theorem bill_profit_difference 
  (SP : ℝ) 
  (hSP : SP = 1.10 * (SP / 1.10)) 
  (hSP_val : SP = 989.9999999999992) 
  (NP : ℝ) 
  (hNP : NP = 0.90 * (SP / 1.10)) 
  (NSP : ℝ) 
  (hNSP : NSP = 1.30 * NP) 
  : NSP - SP = 63.0000000000008 := 
by 
  sorry

end bill_profit_difference_l131_131912


namespace ground_beef_per_package_l131_131839

-- Declare the given conditions and the expected result.
theorem ground_beef_per_package (num_people : ℕ) (weight_per_burger : ℕ) (total_packages : ℕ) 
    (h1 : num_people = 10) 
    (h2 : weight_per_burger = 2) 
    (h3 : total_packages = 4) : 
    (num_people * weight_per_burger) / total_packages = 5 := 
by 
  sorry

end ground_beef_per_package_l131_131839


namespace smallest_four_digit_integer_l131_131465

theorem smallest_four_digit_integer (n : ℕ) (h1 : n ≥ 1000 ∧ n < 10000) 
  (h2 : ∀ d ∈ [1, 5, 6], n % d = 0)
  (h3 : ∀ d1 d2, d1 ≠ d2 → d1 ∈ [1, 5, 6] → d2 ∈ [1, 5, 6] → d1 ≠ d2) :
  n = 1560 :=
by
  sorry

end smallest_four_digit_integer_l131_131465


namespace triangle_area_l131_131621

theorem triangle_area {x y : ℝ} :

  (∀ a:ℝ, y = a ↔ a = x) ∧
  (∀ b:ℝ, y = -b ↔ b = x) ∧
  ( y = 10 )
  → 1 / 2 * abs (10 - (-10)) * 10 = 100 :=
by
  sorry

end triangle_area_l131_131621


namespace set_intersection_l131_131433

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def A : Set ℕ := {1, 2, 5}
noncomputable def B : Set ℕ := {x ∈ U | (3 / (2 - x) + 1 ≤ 0)}
noncomputable def C_U_B : Set ℕ := U \ B

theorem set_intersection : A ∩ C_U_B = {1, 2} :=
by {
  sorry
}

end set_intersection_l131_131433


namespace prove_angle_A_l131_131112

-- Definitions and conditions in triangle ABC
variables (A B C : ℝ) (a b c : ℝ) (h₁ : a^2 - b^2 = 3 * b * c) (h₂ : sin C = 2 * sin B)

-- Objective: Prove that angle A is 120 degrees
theorem prove_angle_A : A = 120 :=
sorry

end prove_angle_A_l131_131112


namespace number_of_pieces_of_string_l131_131017

theorem number_of_pieces_of_string (total_length piece_length : ℝ) (h1 : total_length = 60) (h2 : piece_length = 0.6) :
    total_length / piece_length = 100 := by
  sorry

end number_of_pieces_of_string_l131_131017


namespace max_value_of_x2_plus_y2_l131_131970

noncomputable def max_value (x y : ℝ) : ℝ := x^2 + y^2

theorem max_value_of_x2_plus_y2 {x y : ℝ} (h : 5*x^2 + 4*y^2 = 10*x) : max_value x y ≤ 4 := sorry

end max_value_of_x2_plus_y2_l131_131970


namespace Tim_weekly_earnings_l131_131615

def number_of_tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def working_days_per_week : ℕ := 6

theorem Tim_weekly_earnings :
  (number_of_tasks_per_day * pay_per_task) * working_days_per_week = 720 := by
  sorry

end Tim_weekly_earnings_l131_131615


namespace ryan_weekly_commuting_time_l131_131457

-- Define Ryan's commuting conditions
def bike_time (biking_days: Nat) : Nat := biking_days * 30
def bus_time (bus_days: Nat) : Nat := bus_days * 40
def friend_time (friend_days: Nat) : Nat := friend_days * 10

-- Calculate total commuting time per week
def total_commuting_time (biking_days bus_days friend_days: Nat) : Nat := 
  bike_time biking_days + bus_time bus_days + friend_time friend_days

-- Given conditions
def biking_days : Nat := 1
def bus_days : Nat := 3
def friend_days : Nat := 1

-- Formal statement to prove
theorem ryan_weekly_commuting_time : 
  total_commuting_time biking_days bus_days friend_days = 160 := by 
  sorry

end ryan_weekly_commuting_time_l131_131457


namespace triangle_is_isosceles_if_median_bisects_perimeter_l131_131139

-- Defining the sides of the triangle
variables {a b c : ℝ}

-- Defining the median condition
def median_bisects_perimeter (a b c : ℝ) : Prop :=
  a + b + c = 2 * (a/2 + b)

-- The main theorem stating that the triangle is isosceles if the median bisects the perimeter
theorem triangle_is_isosceles_if_median_bisects_perimeter (a b c : ℝ) 
  (h : median_bisects_perimeter a b c) : b = c :=
by
  sorry

end triangle_is_isosceles_if_median_bisects_perimeter_l131_131139


namespace ultramindmaster_secret_codes_count_l131_131101

/-- 
In the game UltraMindmaster, we need to find the total number of possible secret codes 
formed by placing pegs of any of eight different colors into five slots.
Colors may be repeated, and each slot must be filled.
-/
theorem ultramindmaster_secret_codes_count :
  let colors := 8
  let slots := 5
  colors ^ slots = 32768 := by
    sorry

end ultramindmaster_secret_codes_count_l131_131101


namespace shadow_change_sequence_l131_131005

-- Define the height of the person and the height of the street lamp.
variables (h_person h_lamp : ℝ) (H_height : h_person < h_lamp)

-- Define the distance function for when a person is far from, under, and moving away
def shadow_length := λ d : ℝ, if d = 0 then h_person else ((h_lamp - h_person) / d) * h_person

-- Define conditions for person being 'far from', 'under', and 'away from' the street lamp
def far_from_lamp (d : ℝ) : Prop := d > 1
def under_lamp (d : ℝ) : Prop := d = 0
def away_from_lamp (d : ℝ) : Prop := d < 1 ∧ d > 0

-- Define the sequence of shadow length changes as a person walks under a street lamp.
theorem shadow_change_sequence :
  ∀ d : ℝ, (far_from_lamp d → shadow_length d > h_person) ∧
           (under_lamp d → shadow_length d = h_person) ∧
           (away_from_lamp d → shadow_length d > h_person) :=
by 
  sorry

end shadow_change_sequence_l131_131005


namespace P_roots_implies_Q_square_roots_l131_131428

noncomputable def P (x : ℝ) : ℝ := x^3 - 2 * x + 1

noncomputable def Q (x : ℝ) : ℝ := x^3 - 4 * x^2 + 4 * x - 1

theorem P_roots_implies_Q_square_roots (r : ℝ) (h : P r = 0) : Q (r^2) = 0 := sorry

end P_roots_implies_Q_square_roots_l131_131428


namespace revenue_ratio_l131_131020

variable (R_d : ℝ) (R_n : ℝ) (R_j : ℝ)

theorem revenue_ratio
  (nov_cond : R_n = 2 / 5 * R_d)
  (jan_cond : R_j = 1 / 2 * R_n) :
  R_d = 10 / 3 * ((R_n + R_j) / 2) := by
  -- Proof steps go here
  sorry

end revenue_ratio_l131_131020


namespace even_and_odd_functions_satisfying_equation_l131_131071

theorem even_and_odd_functions_satisfying_equation :
  ∀ (f g : ℝ → ℝ),
    (∀ x : ℝ, f (-x) = f x) →                      -- condition 1: f is even
    (∀ x : ℝ, g (-x) = -g x) →                    -- condition 2: g is odd
    (∀ x : ℝ, f x - g x = x^3 + x^2 + 1) →        -- condition 3: f(x) - g(x) = x^3 + x^2 + 1
    f 1 + g 1 = 1 :=                              -- question: proof of f(1) + g(1) = 1
by
  intros f g h_even h_odd h_eqn
  sorry

end even_and_odd_functions_satisfying_equation_l131_131071


namespace smallest_rel_prime_to_180_l131_131247

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ x ≤ 7 ∧ (∀ y : ℕ, y > 1 ∧ y < x → y.gcd 180 ≠ 1) ∧ x.gcd 180 = 1 :=
by
  sorry

end smallest_rel_prime_to_180_l131_131247


namespace choose_5_from_12_l131_131103

theorem choose_5_from_12 : Nat.choose 12 5 = 792 := by
  sorry

end choose_5_from_12_l131_131103


namespace min_positive_period_condition_1_min_value_condition_2_min_value_condition_3_not_unique_l131_131797

-- Function definition
def f (x m : ℝ) : ℝ := Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 + m

-- Minimum period proof
theorem min_positive_period : ∀ m : ℝ, ∃ T > 0, (∀ x : ℝ, f (x + T) m = f x m) ∧ T = Real.pi :=
by sorry

-- Condition ①: The maximum value of f(x) is 1
theorem condition_1_min_value : ∃ m : ℝ, ∀ x : ℝ, (m = -Real.sqrt 2) → (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x m ≥ -Real.sqrt 2) :=
by sorry

-- Condition ②: Symmetry point (3π/8, 0)
theorem condition_2_min_value : ∃ m : ℝ, (f (3*Real.pi/8) m = 0) → (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x (-1) ≥ -1) :=
by sorry

-- Condition ③: Line of symmetry x = π/8 does not determine m uniquely
theorem condition_3_not_unique : ∀ m : ℝ, ¬(∃ x : ℝ, ∀ x, (f x m = f (Real.pi/8 - x) m)) :=
by sorry

end min_positive_period_condition_1_min_value_condition_2_min_value_condition_3_not_unique_l131_131797


namespace inequality_proof_l131_131119

variable (a b c : ℝ)

theorem inequality_proof (a b c : ℝ) :
    a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤
    1 + (1 / 3) * (a + b + c) ^ 2 :=
by
  sorry

end inequality_proof_l131_131119


namespace number_of_terminating_decimals_l131_131255

theorem number_of_terminating_decimals (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 150) :
  ∃ m, m = 50 ∧ 
  ∀ n, (1 ≤ n ∧ n ≤ 150) → (∃ k, n = 3 * k) →
  m = 50 :=
by 
  sorry

end number_of_terminating_decimals_l131_131255


namespace area_product_is_2_l131_131689

open Real

-- Definitions for parabola, points, and the condition of dot product
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 * B.1 + A.2 * B.2) = -4

def area (O F P : ℝ × ℝ) : ℝ :=
  0.5 * abs (O.1 * (F.2 - P.2) + F.1 * (P.2 - O.2) + P.1 * (O.2 - F.2))

-- Points A and B are on the parabola and the dot product condition holds
variables (A B : ℝ × ℝ)
variable (H_A_on_parabola : parabola A.1 A.2)
variable (H_B_on_parabola : parabola B.1 B.2)
variable (H_dot_product : dot_product_condition A B)

-- Focus of the parabola
def F : ℝ × ℝ := (1, 0)

-- Origin
def O : ℝ × ℝ := (0, 0)

-- Prove that the product of areas is 2
theorem area_product_is_2 : 
  area O F A * area O F B = 2 :=
sorry

end area_product_is_2_l131_131689


namespace area_of_sector_l131_131597

theorem area_of_sector (l : ℝ) (α : ℝ) (r : ℝ) (S : ℝ) 
  (h1 : l = 3)
  (h2 : α = 1)
  (h3 : l = α * r) : 
  S = 9 / 2 :=
by
  sorry

end area_of_sector_l131_131597


namespace quadratic_intersects_once_l131_131931

theorem quadratic_intersects_once (c : ℝ) : (∀ x : ℝ, x^2 - 6 * x + c = 0 → x = 3 ) ↔ c = 9 :=
by
  sorry

end quadratic_intersects_once_l131_131931


namespace song_distribution_l131_131282

theorem song_distribution (five_songs : Finset α)
  (like_AB_Beth_Amy_not_Jo : ∃ s, s ∈ five_songs)
  (like_BC_Beth_Jo_not_Amy : ∃ s, s ∈ five_songs)
  (like_CA_Jo_Amy_not_Beth : ∃ s, s ∈ five_songs)
  (no_song_liked_by_all : ∀ s, s ∉ (like_AB_Beth_Amy_not_Jo ∩ like_BC_Beth_Jo_not_Amy ∩ like_CA_Jo_Amy_not_Beth)) :
  ∃ arrangements, arrangements.card = 168 :=
sorry

end song_distribution_l131_131282


namespace frank_remaining_money_l131_131933

theorem frank_remaining_money
  (cheapest_lamp : ℕ)
  (most_expensive_factor : ℕ)
  (frank_money : ℕ)
  (cheapest_lamp_cost : cheapest_lamp = 20)
  (most_expensive_lamp_cost : most_expensive_factor = 3)
  (frank_current_money : frank_money = 90) :
  frank_money - (most_expensive_factor * cheapest_lamp) = 30 :=
by {
  sorry
}

end frank_remaining_money_l131_131933


namespace inequality_proof_l131_131394

theorem inequality_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
(h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d) (h_sum : a + b + c + d ≥ 1) : 
  a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 ≥ 1 := 
by 
  sorry

end inequality_proof_l131_131394


namespace find_k_l131_131279

theorem find_k (t k : ℤ) (h1 : t = 35) (h2 : t = 5 * (k - 32) / 9) : k = 95 :=
sorry

end find_k_l131_131279


namespace number_of_cows_l131_131173

theorem number_of_cows (n : ℝ) (h1 : n / 2 + n / 4 + n / 5 + 7 = n) : n = 140 := 
sorry

end number_of_cows_l131_131173


namespace research_development_success_l131_131029

theorem research_development_success 
  (P_A : ℝ)  -- probability of Team A successfully developing a product
  (P_B : ℝ)  -- probability of Team B successfully developing a product
  (independent : Bool)  -- independence condition (dummy for clarity)
  (h1 : P_A = 2/3)
  (h2 : P_B = 3/5) 
  (h3 : independent = true) :
  (1 - (1 - P_A) * (1 - P_B) = 13/15) :=
by
  sorry

end research_development_success_l131_131029


namespace rate_of_interest_l131_131641

theorem rate_of_interest (SI P T R : ℝ) 
  (hSI : SI = 4016.25) 
  (hP : P = 6693.75) 
  (hT : T = 5) 
  (h : SI = (P * R * T) / 100) : 
  R = 12 :=
by 
  sorry

end rate_of_interest_l131_131641


namespace range_of_a_l131_131680

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
sorry

end range_of_a_l131_131680


namespace universal_quantifiers_are_true_l131_131042

-- Declare the conditions as hypotheses
theorem universal_quantifiers_are_true :
  (∀ x : ℝ, x^2 - x + 0.25 ≥ 0) ∧ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) :=
by
  sorry -- Proof skipped

end universal_quantifiers_are_true_l131_131042


namespace problem_part_1_problem_part_2_l131_131535

theorem problem_part_1 (a b : ℝ) (h1 : a * 1^2 - 3 * 1 + 2 = 0) (h2 : a * b^2 - 3 * b + 2 = 0) (h3 : 1 + b = 3 / a) (h4 : 1 * b = 2 / a) : a = 1 ∧ b = 2 :=
sorry

theorem problem_part_2 (m : ℝ) (h5 : a = 1) (h6 : b = 2) : 
  (m = 2 → ∀ x, ¬ (x^2 - (m + 2) * x + 2 * m < 0)) ∧
  (m < 2 → ∀ x, x ∈ Set.Ioo m 2 ↔ x^2 - (m + 2) * x + 2 * m < 0) ∧
  (m > 2 → ∀ x, x ∈ Set.Ioo 2 m ↔ x^2 - (m + 2) * x + 2 * m < 0) :=
sorry

end problem_part_1_problem_part_2_l131_131535


namespace problem_equivalence_l131_131954

noncomputable def parametric_circle_point (t : ℝ) : ℝ × ℝ :=
  (-5 + (Real.sqrt 2) * Real.cos t, 3 + (Real.sqrt 2) * Real.sin t)

def polar_to_cartesian_point (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def point_A : ℝ × ℝ := polar_to_cartesian_point 2 (Real.pi / 2)

def point_B : ℝ × ℝ := polar_to_cartesian_point 2 Real.pi

def line_l_cartesian (ρ θ : ℝ) : ℝ :=
  ρ * Real.cos θ - ρ * Real.sin θ

theorem problem_equivalence :
  (∀ t : ℝ, let (x, y) := parametric_circle_point t in (x + 5)^2 + (y - 3)^2 = 2) ∧
  (line_l_cartesian (2 * Real.sqrt 2) (Real.pi / 4) = -2) ∧
  (let (Aₓ, Aᵧ) := point_A, (Bₓ, Bᵧ) := point_B,
       d_min := (4 / Real.sqrt 2) in
     1 / 2 * 2 * Real.sqrt 2 * d_min = 4) :=
by
  sorry

end problem_equivalence_l131_131954


namespace intersect_point_one_l131_131774

theorem intersect_point_one (k : ℝ) : 
  (∀ y : ℝ, (x = -3 * y^2 - 2 * y + 4 ↔ x = k)) ↔ k = 13 / 3 := 
by
  sorry

end intersect_point_one_l131_131774


namespace corrected_mean_l131_131446

theorem corrected_mean (mean_initial : ℝ) (num_obs : ℕ) (obs_incorrect : ℝ) (obs_correct : ℝ) :
  mean_initial = 36 → num_obs = 50 → obs_incorrect = 23 → obs_correct = 30 →
  (mean_initial * ↑num_obs + (obs_correct - obs_incorrect)) / ↑num_obs = 36.14 :=
by
  intros h1 h2 h3 h4
  sorry

end corrected_mean_l131_131446


namespace genuine_coin_remains_l131_131170

theorem genuine_coin_remains (n : ℕ) (g f : ℕ) (h : n = 2022) (h_g : g > n/2) (h_f : f = n - g) : 
  (after_moves : ℕ) -> after_moves = n - 1 -> ∃ remaining_g : ℕ, remaining_g > 0 :=
by
  intros
  sorry

end genuine_coin_remains_l131_131170


namespace directrix_of_parabola_l131_131059

def parabola_directrix (x_y_eqn : ℝ → ℝ) : ℝ := by
  -- Assuming the parabola equation x = -(1/4) y^2
  sorry

theorem directrix_of_parabola : parabola_directrix (fun y => -(1/4) * y^2) = 1 := by
  sorry

end directrix_of_parabola_l131_131059


namespace two_hours_charge_l131_131349

def charge_condition_1 (F A : ℕ) : Prop :=
  F = A + 35

def charge_condition_2 (F A : ℕ) : Prop :=
  F + 4 * A = 350

theorem two_hours_charge (F A : ℕ) (h1 : charge_condition_1 F A) (h2 : charge_condition_2 F A) : 
  F + A = 161 := 
sorry

end two_hours_charge_l131_131349


namespace sum_of_squares_pattern_l131_131401

theorem sum_of_squares_pattern (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^6 + b^6 = 18 :=
sorry

end sum_of_squares_pattern_l131_131401


namespace weights_problem_l131_131184

theorem weights_problem
  (a b c d : ℕ)
  (h1 : a + b = 280)
  (h2 : b + c = 255)
  (h3 : c + d = 290) 
  : a + d = 315 := 
  sorry

end weights_problem_l131_131184


namespace marcy_cat_time_l131_131124

theorem marcy_cat_time (petting_time combing_ratio : ℝ) :
  petting_time = 12 ∧ combing_ratio = 1/3 → (petting_time + petting_time * combing_ratio) = 16 :=
by
  intros h
  cases h with petting_eq combing_eq
  rw [petting_eq, combing_eq]
  norm_num


end marcy_cat_time_l131_131124


namespace salary_january_l131_131998

theorem salary_january
  (J F M A May : ℝ)  -- declare the salaries as real numbers
  (h1 : (J + F + M + A) / 4 = 8000)  -- condition 1
  (h2 : (F + M + A + May) / 4 = 9500)  -- condition 2
  (h3 : May = 6500) :  -- condition 3
  J = 500 := 
by
  sorry

end salary_january_l131_131998


namespace marcy_total_time_l131_131123

theorem marcy_total_time 
    (petting_time : ℝ)
    (fraction_combing : ℝ)
    (H1 : petting_time = 12)
    (H2 : fraction_combing = 1/3) :
    (petting_time + (fraction_combing * petting_time) = 16) :=
  sorry

end marcy_total_time_l131_131123


namespace angle_B_eq_pi_over_3_range_of_area_l131_131284

-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively
-- And given vectors m and n represented as stated and are collinear
-- Prove that angle B is π/3
theorem angle_B_eq_pi_over_3
  (ABC_acute : True) -- placeholder condition indicating acute triangle
  (a b c A B C : ℝ)
  (m := (2 * Real.sin (A + C), - Real.sqrt 3))
  (n := (Real.cos (2 * B), 2 * Real.cos (B / 2) ^ 2 - 1))
  (collinear_m_n : m.1 * n.2 = m.2 * n.1) :
  B = Real.pi / 3 :=
sorry

-- Given side b = 1, find the range of the area S of triangle ABC
theorem range_of_area
  (a c A B C : ℝ)
  (triangle_area : ℝ)
  (ABC_acute : True) -- placeholder condition indicating acute triangle
  (hB : B = Real.pi / 3)
  (hb : b = 1)
  (cosine_theorem : 1 = a^2 + c^2 - a*c)
  (area_formula : triangle_area = (1/2) * a * c * Real.sin B) :
  0 < triangle_area ∧ triangle_area ≤ (Real.sqrt 3) / 4 :=
sorry

end angle_B_eq_pi_over_3_range_of_area_l131_131284


namespace angle_measure_l131_131950

-- Defining the type of angles in degrees
def is_complementary (a b : ℝ) : Prop := a + b = 90
def is_supplementary (a b : ℝ) : Prop := a + b = 180

-- Defining the conditions 
def conditions (x : ℝ) : Prop :=
  is_complementary (90 - x) (180 - x)

-- Main theorem statement
theorem angle_measure (x : ℝ) (h : conditions x) : x = 45 :=
  sorry

end angle_measure_l131_131950


namespace terminating_decimal_fractions_l131_131254

theorem terminating_decimal_fractions :
  let n_count := (finset.range 151).filter (λ n, n % 3 = 0),
  n_count.card = 50 :=
by
  sorry

end terminating_decimal_fractions_l131_131254


namespace each_group_has_145_bananas_l131_131454

theorem each_group_has_145_bananas (total_bananas : ℕ) (groups_bananas : ℕ) : 
  total_bananas = 290 ∧ groups_bananas = 2 → total_bananas / groups_bananas = 145 := 
by 
  sorry

end each_group_has_145_bananas_l131_131454


namespace point_on_y_axis_l131_131687

theorem point_on_y_axis (y : ℝ) :
  let A := (1, 0, 2)
  let B := (1, -3, 1)
  let M := (0, y, 0)
  dist A M = dist B M → y = -1 :=
by sorry

end point_on_y_axis_l131_131687


namespace common_ratio_geometric_sequence_l131_131288

theorem common_ratio_geometric_sequence (a₃ S₃ : ℝ) (q : ℝ)
  (h1 : a₃ = 7) (h2 : S₃ = 21)
  (h3 : ∃ a₁ : ℝ, a₃ = a₁ * q^2)
  (h4 : ∃ a₁ : ℝ, S₃ = a₁ * (1 + q + q^2)) :
  q = -1/2 ∨ q = 1 :=
sorry

end common_ratio_geometric_sequence_l131_131288


namespace adam_played_rounds_l131_131164

theorem adam_played_rounds (total_points points_per_round : ℕ) (h_total : total_points = 283) (h_per_round : points_per_round = 71) : total_points / points_per_round = 4 := by
  -- sorry is a placeholder for the actual proof
  sorry

end adam_played_rounds_l131_131164


namespace correct_statement_about_Digital_Earth_l131_131468

-- Definitions of the statements
def statement_A : Prop :=
  "Digital Earth is a reflection of the real Earth through digital means" = "Correct statement about Digital Earth"

def statement_B : Prop :=
  "Digital Earth is an extension of GIS technology" = "Correct statement about Digital Earth"

def statement_C : Prop :=
  "Digital Earth can only achieve global information sharing through the internet" = "Correct statement about Digital Earth"

def statement_D : Prop :=
  "The core idea of Digital Earth is to use digital means to uniformly address Earth's issues" = "Correct statement about Digital Earth"

-- Theorem that needs to be proved 
theorem correct_statement_about_Digital_Earth : statement_C :=
by 
  sorry

end correct_statement_about_Digital_Earth_l131_131468


namespace intersection_A_B_l131_131024

-- Define the sets A and B
def A : Set ℤ := {1, 3, 5, 7}
def B : Set ℤ := {x | 2 ≤ x ∧ x ≤ 5}

-- The goal is to prove that A ∩ B = {3, 5}
theorem intersection_A_B : A ∩ B = {3, 5} :=
  sorry

end intersection_A_B_l131_131024


namespace equilateral_division_l131_131907

theorem equilateral_division (k : ℕ) :
  (k = 1 ∨ k = 3 ∨ k = 4 ∨ k = 9 ∨ k = 12 ∨ k = 36) ↔
  (k ∣ 36 ∧ ¬ (k = 2 ∨ k = 6 ∨ k = 18)) := by
  sorry

end equilateral_division_l131_131907


namespace unique_cubic_coefficients_l131_131314

noncomputable def cubic_function (a b c : ℝ) (x : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + c

theorem unique_cubic_coefficients
  (a b c : ℝ)
  (h1 : ∀ x, -1 ≤ x ∧ x ≤ 1 → -1 ≤ cubic_function a b c x ∧ cubic_function a b c x ≤ 1) :
  (a = 0 ∧ b = -3 ∧ c = 0) :=
by
  sorry

end unique_cubic_coefficients_l131_131314


namespace geometric_sequence_angles_l131_131652

noncomputable def theta_conditions (θ : ℝ) :=
  0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ θ ≠ Real.pi / 2 ∧ θ ≠ Real.pi ∧ θ ≠ 3 * Real.pi / 2 ∧
  (∀ {a b c : ℝ}, Set {a, b, c} = {Real.sin θ, Real.cos θ, Real.cot θ} → a * c = b * b)

theorem geometric_sequence_angles : 
  ∃! θ1 θ2 : ℝ, theta_conditions θ1 ∧ theta_conditions θ2 ∧ θ1 ≠ θ2 :=
  sorry

end geometric_sequence_angles_l131_131652


namespace range_of_a_l131_131668

theorem range_of_a (hP : ¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) : 0 < a ∧ a < 1 :=
sorry

end range_of_a_l131_131668


namespace binomial_sum_l131_131499

theorem binomial_sum :
  (Nat.choose 10 3) + (Nat.choose 10 4) = 330 :=
by
  sorry

end binomial_sum_l131_131499


namespace speed_ratio_l131_131890

variable (v_A v_B : ℝ)

def equidistant_3min : Prop := 3 * v_A = abs (-800 + 3 * v_B)
def equidistant_8min : Prop := 8 * v_A = abs (-800 + 8 * v_B)
def speed_ratio_correct : Prop := v_A / v_B = 1 / 2

theorem speed_ratio (h1 : equidistant_3min v_A v_B) (h2 : equidistant_8min v_A v_B) : speed_ratio_correct v_A v_B :=
by
  sorry

end speed_ratio_l131_131890


namespace balls_into_boxes_l131_131806

theorem balls_into_boxes :
  let n := 7 -- number of balls
  let k := 3 -- number of boxes
  let ways := Nat.choose (n + k - 1) (k - 1)
  ways = 36 :=
by
  sorry

end balls_into_boxes_l131_131806


namespace ax5_by5_eq_28616_l131_131309

variables (a b x y : ℝ)

theorem ax5_by5_eq_28616
  (h1 : a * x + b * y = 1)
  (h2 : a * x^2 + b * y^2 = 9)
  (h3 : a * x^3 + b * y^3 = 28)
  (h4 : a * x^4 + b * y^4 = 96) :
  a * x^5 + b * y^5 = 28616 :=
sorry

end ax5_by5_eq_28616_l131_131309


namespace pencils_profit_goal_l131_131487

theorem pencils_profit_goal (n : ℕ) (price_purchase price_sale cost_goal : ℚ) (purchase_quantity : ℕ) 
  (h1 : price_purchase = 0.10) 
  (h2 : price_sale = 0.25) 
  (h3 : cost_goal = 100) 
  (h4 : purchase_quantity = 1500) 
  (h5 : n * price_sale ≥ purchase_quantity * price_purchase + cost_goal) :
  n ≥ 1000 :=
sorry

end pencils_profit_goal_l131_131487


namespace sn_values_l131_131429

noncomputable def s (x1 x2 x3 : ℂ) (n : ℕ) : ℂ :=
  x1^n + x2^n + x3^n

theorem sn_values (p q x1 x2 x3 : ℂ) (h_root1 : x1^3 + p * x1 + q = 0)
                    (h_root2 : x2^3 + p * x2 + q = 0)
                    (h_root3 : x3^3 + p * x3 + q = 0) :
  s x1 x2 x3 2 = -3 * q ∧
  s x1 x2 x3 3 = 3 * q^2 ∧
  s x1 x2 x3 4 = 2 * p^2 ∧
  s x1 x2 x3 5 = 5 * p * q ∧
  s x1 x2 x3 6 = -2 * p^3 + 3 * q^2 ∧
  s x1 x2 x3 7 = -7 * p^2 * q ∧
  s x1 x2 x3 8 = 2 * p^4 - 8 * p * q^2 ∧
  s x1 x2 x3 9 = 9 * p^3 * q - 3 * q^3 ∧
  s x1 x2 x3 10 = -2 * p^5 + 15 * p^2 * q^2 :=
by {
  sorry
}

end sn_values_l131_131429


namespace find_y_l131_131818

open Real

variable {x y : ℝ}

theorem find_y (h1 : x * y = 25) (h2 : x / y = 36) (hx : 0 < x) (hy : 0 < y) :
  y = 5 / 6 :=
by
  sorry

end find_y_l131_131818


namespace possible_case_l131_131096

-- Define the logical propositions P and Q
variables (P Q : Prop)

-- State the conditions given in the problem
axiom h1 : P ∨ Q     -- P ∨ Q is true
axiom h2 : ¬ (P ∧ Q) -- P ∧ Q is false

-- Formulate the proof problem in Lean
theorem possible_case : P ∧ ¬Q :=
by
  sorry -- Proof to be filled in later

end possible_case_l131_131096


namespace find_number_l131_131375

theorem find_number (x : ℝ) (h : x / 0.05 = 900) : x = 45 :=
by sorry

end find_number_l131_131375


namespace Tim_weekly_earnings_l131_131614

def number_of_tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def working_days_per_week : ℕ := 6

theorem Tim_weekly_earnings :
  (number_of_tasks_per_day * pay_per_task) * working_days_per_week = 720 := by
  sorry

end Tim_weekly_earnings_l131_131614


namespace middle_term_is_35_l131_131826

-- Define the arithmetic sequence condition
def arithmetic_sequence (a b c d e f : ℤ) : Prop :=
  b - a = c - b ∧ c - b = d - c ∧ d - c = e - d ∧ e - d = f - e

-- Given sequence values
def seq1 := 23
def seq6 := 47

-- Theorem stating that the middle term y in the sequence is 35
theorem middle_term_is_35 (x y z w : ℤ) :
  arithmetic_sequence seq1 x y z w seq6 → y = 35 :=
by
  sorry

end middle_term_is_35_l131_131826


namespace jimmy_fill_bucket_time_l131_131290

-- Definitions based on conditions
def pool_volume : ℕ := 84
def bucket_volume : ℕ := 2
def total_time_minutes : ℕ := 14
def total_time_seconds : ℕ := total_time_minutes * 60
def trips : ℕ := pool_volume / bucket_volume

-- Theorem statement
theorem jimmy_fill_bucket_time : (total_time_seconds / trips) = 20 := by
  sorry

end jimmy_fill_bucket_time_l131_131290


namespace age_ratio_l131_131472

theorem age_ratio 
  (a b c : ℕ)
  (h1 : a = b + 2)
  (h2 : a + b + c = 32)
  (h3 : b = 12) :
  b = 2 * c :=
by
  sorry

end age_ratio_l131_131472


namespace base9_perfect_square_l131_131416

theorem base9_perfect_square (a b d : ℕ) (h1 : a ≠ 0) (h2 : ∃ k : ℕ, (729 * a + 81 * b + 36 + d) = k * k) :
    d = 0 ∨ d = 1 ∨ d = 4 ∨ d = 7 :=
sorry

end base9_perfect_square_l131_131416


namespace smallest_coprime_gt_one_l131_131230

theorem smallest_coprime_gt_one (x : ℕ) (h1 : 1 < x) (h2 : Nat.gcd x 180 = 1) : x = 7 := sorry

end smallest_coprime_gt_one_l131_131230


namespace find_the_number_l131_131134

theorem find_the_number : ∃ x : ℝ, (10 + x + 50) / 3 = (20 + 40 + 6) / 3 + 8 ∧ x = 30 := 
by
  sorry

end find_the_number_l131_131134


namespace gcd_equivalence_l131_131517

theorem gcd_equivalence : 
  let m := 2^2100 - 1
  let n := 2^2091 + 31
  gcd m n = gcd (2^2091 + 31) 511 :=
by
  sorry

end gcd_equivalence_l131_131517


namespace part1_part2_l131_131084

open Classical

theorem part1 (x : ℝ) (a : ℝ) (b : ℝ) :
  (a = 1) ∧ (b = 2) ∧ (∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :=
by
  sorry

theorem part2 (x y k : ℝ) (a b : ℝ) :
  a = 1 ∧ b = 2 ∧ (x > 0) ∧ (y > 0) ∧ (1 / x + 2 / y = 1) ∧ (2 * x + y ≥ k^2 + k + 2) → -3 ≤ k ∧ k ≤ 2 :=
by
  sorry

end part1_part2_l131_131084


namespace division_of_neg_six_by_three_l131_131369

theorem division_of_neg_six_by_three : (-6) / 3 = -2 := by
  sorry

end division_of_neg_six_by_three_l131_131369


namespace expand_product_l131_131054

theorem expand_product (x : ℝ) :
  (2 * x^2 - 3 * x + 5) * (x^2 + 4 * x + 3) = 2 * x^4 + 5 * x^3 - x^2 + 11 * x + 15 :=
by
  -- Proof to be filled in
  sorry

end expand_product_l131_131054


namespace parabola_focus_distance_l131_131108

theorem parabola_focus_distance (p : ℝ) : 
  (∀ (y : ℝ), y^2 = 2 * p * 4 → abs (4 + p / 2) = 5) → 
  p = 2 :=
by
  sorry

end parabola_focus_distance_l131_131108


namespace clock_angle_8_15_l131_131334

theorem clock_angle_8_15:
  ∃ angle : ℝ, time_on_clock = 8.25 → angle = 157.5 := sorry

end clock_angle_8_15_l131_131334


namespace haley_tv_total_hours_l131_131800

theorem haley_tv_total_hours (h_sat : Nat) (h_sun : Nat) (H_sat : h_sat = 6) (H_sun : h_sun = 3) :
  h_sat + h_sun = 9 := by
  sorry

end haley_tv_total_hours_l131_131800


namespace minimum_value_of_a2b_l131_131414

noncomputable def minimum_value (a b : ℝ) := a + 2 * b

theorem minimum_value_of_a2b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 1 / (2 * a + b) + 1 / (b + 1) = 1) :
  minimum_value a b = (2 * Real.sqrt 3 + 1) / 2 :=
sorry

end minimum_value_of_a2b_l131_131414


namespace number_of_recipes_needed_l131_131909

def numStudents : ℕ := 150
def avgCookiesPerStudent : ℕ := 3
def cookiesPerRecipe : ℕ := 18
def attendanceDrop : ℝ := 0.40

theorem number_of_recipes_needed (n : ℕ) (c : ℕ) (r : ℕ) (d : ℝ) : 
  n = numStudents →
  c = avgCookiesPerStudent →
  r = cookiesPerRecipe →
  d = attendanceDrop →
  ∃ (recipes : ℕ), recipes = 15 :=
by
  intros
  sorry

end number_of_recipes_needed_l131_131909


namespace avg_annual_growth_rate_l131_131191
-- Import the Mathlib library

-- Define the given conditions
def initial_income : ℝ := 32000
def final_income : ℝ := 37000
def period : ℝ := 2
def initial_income_ten_thousands : ℝ := initial_income / 10000
def final_income_ten_thousands : ℝ := final_income / 10000

-- Define the growth rate
variable (x : ℝ)

-- Define the theorem
theorem avg_annual_growth_rate :
  3.2 * (1 + x) ^ 2 = 3.7 :=
sorry

end avg_annual_growth_rate_l131_131191


namespace distinct_roots_difference_l131_131117

theorem distinct_roots_difference (r s : ℝ) (h₀ : r ≠ s) (h₁ : r > s) (h₂ : ∀ x, (5 * x - 20) / (x^2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) :
  r - s = Real.sqrt 29 :=
by
  sorry

end distinct_roots_difference_l131_131117


namespace smallest_m_l131_131466

theorem smallest_m (m : ℕ) (p q : ℤ) (h_eq : 12 * p * p - m * p + 432 = 0) (h_sum : p + q = m / 12) (h_prod : p * q = 36) :
  m = 144 :=
by
  sorry

end smallest_m_l131_131466


namespace smallest_number_of_eggs_proof_l131_131014

noncomputable def smallest_number_of_eggs (c : ℕ) : ℕ := 15 * c - 3

theorem smallest_number_of_eggs_proof :
  ∃ c : ℕ, c ≥ 11 ∧ smallest_number_of_eggs c = 162 ∧ smallest_number_of_eggs c > 150 :=
by
  sorry

end smallest_number_of_eggs_proof_l131_131014


namespace paintable_sum_l131_131801

theorem paintable_sum :
  ∃ (h t u v : ℕ), h > 0 ∧ t > 0 ∧ u > 0 ∧ v > 0 ∧
  (∀ k, k % h = 1 ∨ k % t = 2 ∨ k % u = 3 ∨ k % v = 4) ∧
  (∀ k k', k ≠ k' → (k % h ≠ k' % h ∧ k % t ≠ k' % t ∧ k % u ≠ k' % u ∧ k % v ≠ k' % v)) ∧
  1000 * h + 100 * t + 10 * u + v = 4536 :=
by
  sorry

end paintable_sum_l131_131801


namespace average_mark_of_excluded_students_l131_131600

theorem average_mark_of_excluded_students
  (N : ℕ) (A A_remaining : ℕ)
  (num_excluded : ℕ)
  (hN : N = 9)
  (hA : A = 60)
  (hA_remaining : A_remaining = 80)
  (h_excluded : num_excluded = 5) :
  (N * A - (N - num_excluded) * A_remaining) / num_excluded = 44 :=
by
  sorry

end average_mark_of_excluded_students_l131_131600


namespace pen_distribution_l131_131922

theorem pen_distribution:
  (∃ (fountain: ℕ) (ballpoint: ℕ), fountain = 2 ∧ ballpoint = 3) ∧
  (∃ (students: ℕ), students = 4) →
  (∀ (s: ℕ), s ≥ 1 → s ≤ 4) →
  ∃ (ways: ℕ), ways = 28 :=
by
  sorry

end pen_distribution_l131_131922


namespace proof_theorem_l131_131763

noncomputable def proof_problem (y1 y2 y3 y4 y5 : ℝ) :=
  y1 + 8*y2 + 27*y3 + 64*y4 + 125*y5 = 7 ∧
  8*y1 + 27*y2 + 64*y3 + 125*y4 + 216*y5 = 100 ∧
  27*y1 + 64*y2 + 125*y3 + 216*y4 + 343*y5 = 1000 →
  64*y1 + 125*y2 + 216*y3 + 343*y4 + 512*y5 = -5999

theorem proof_theorem : ∀ (y1 y2 y3 y4 y5 : ℝ), proof_problem y1 y2 y3 y4 y5 :=
  by intros y1 y2 y3 y4 y5
     unfold proof_problem
     intro h
     sorry

end proof_theorem_l131_131763


namespace find_k_l131_131691

def geom_seq (c : ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = c * (a n)

def sum_first_n_terms (S : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n, S n = 3^n + k

theorem find_k {c : ℝ} {a : ℕ → ℝ} {S : ℕ → ℝ} {k : ℝ} (hGeom : geom_seq c a) (hSum : sum_first_n_terms S k) :
  k = -1 :=
by
  sorry

end find_k_l131_131691


namespace hyperbola_eccentricity_l131_131417

-- Definitions of conditions
variables {a b c : ℝ}
variables (h : a > 0) (h' : b > 0)
variables (hyp : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
variables (parab : ∀ y : ℝ, y^2 = 4 * b * y)
variables (ratio_cond : (b + c) / (c - b) = 5 / 3)

-- Proof statement
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 4 * Real.sqrt 15 / 15 :=
by
  have hyp_foci_distance : ∃ c : ℝ, c^2 = a^2 + b^2 := sorry
  have e := (4 * Real.sqrt 15) / 15
  use e
  sorry

end hyperbola_eccentricity_l131_131417


namespace total_cost_of_4_sandwiches_and_6_sodas_is_37_4_l131_131588

def sandwiches_cost (s: ℕ) : ℝ := 4 * s
def sodas_cost (d: ℕ) : ℝ := 3 * d
def total_cost_before_tax (s: ℕ) (d: ℕ) : ℝ := sandwiches_cost s + sodas_cost d
def tax (amount: ℝ) : ℝ := 0.10 * amount
def total_cost (s: ℕ) (d: ℕ) : ℝ := total_cost_before_tax s d + tax (total_cost_before_tax s d)

theorem total_cost_of_4_sandwiches_and_6_sodas_is_37_4 :
    total_cost 4 6 = 37.4 :=
sorry

end total_cost_of_4_sandwiches_and_6_sodas_is_37_4_l131_131588


namespace sin_alpha_value_l131_131385

theorem sin_alpha_value (α : ℝ) (h1 : Real.tan α = 2) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.sin α = -2 * Real.sqrt 5 / 5 :=
by
  sorry

end sin_alpha_value_l131_131385


namespace mark_buttons_l131_131837

theorem mark_buttons (initial_buttons : ℕ) (shane_buttons : ℕ) (sam_buttons : ℕ) :
  initial_buttons = 14 →
  shane_buttons = 3 * initial_buttons →
  sam_buttons = (initial_buttons + shane_buttons) / 2 →
  final_buttons = (initial_buttons + shane_buttons) - sam_buttons →
  final_buttons = 28 :=
by
  sorry

end mark_buttons_l131_131837


namespace symmetric_line_equation_l131_131537

theorem symmetric_line_equation :
  (∃ l : ℝ × ℝ × ℝ, (∀ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ + x₂ = -4 → y₁ + y₂ = 2 → 
    ∃ a b c : ℝ, l = (a, b, c) ∧ x₁ * a + y₁ * b + c = 0 ∧ x₂ * a + y₂ * b + c = 0) → 
  l = (2, -1, 5)) :=
sorry

end symmetric_line_equation_l131_131537


namespace frank_remaining_money_l131_131932

theorem frank_remaining_money
  (cheapest_lamp : ℕ)
  (most_expensive_factor : ℕ)
  (frank_money : ℕ)
  (cheapest_lamp_cost : cheapest_lamp = 20)
  (most_expensive_lamp_cost : most_expensive_factor = 3)
  (frank_current_money : frank_money = 90) :
  frank_money - (most_expensive_factor * cheapest_lamp) = 30 :=
by {
  sorry
}

end frank_remaining_money_l131_131932


namespace problem_statement_l131_131069

variable {a : ℕ → ℝ} 
variable {a1 d : ℝ}
variable (h_arith : ∀ n, a (n + 1) = a n + d)  -- Arithmetic sequence condition
variable (h_d_nonzero : d ≠ 0)  -- d ≠ 0
variable (h_a1_nonzero : a1 ≠ 0)  -- a1 ≠ 0
variable (h_geom : (a 1) * (a 7) = (a 3) ^ 2)  -- Geometric sequence condition a2 = a 1, a4 = a 3, a8 = a 7

theorem problem_statement :
  (a 0 + a 4 + a 8) / (a 1 + a 2) = 3 :=
by
  sorry

end problem_statement_l131_131069


namespace find_arith_seq_params_l131_131467

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- The conditions given in the problem
theorem find_arith_seq_params :
  ∃ a d : ℤ, 
  (arithmetic_sequence a d 8) = 5 * (arithmetic_sequence a d 1) ∧
  (arithmetic_sequence a d 12) = 2 * (arithmetic_sequence a d 5) + 5 ∧
  a = 3 ∧
  d = 4 :=
by
  sorry

end find_arith_seq_params_l131_131467


namespace area_BCD_sixteen_area_BCD_with_new_ABD_l131_131260

-- Define the conditions and parameters of the problem.
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Given conditions from part (a)
variable (AB_length : Real) (BC_length : Real) (area_ABD : Real)

-- Define the lengths and areas in our problem.
axiom AB_eq_five : AB_length = 5
axiom BC_eq_eight : BC_length = 8
axiom area_ABD_eq_ten : area_ABD = 10

-- Part (a) problem statement
theorem area_BCD_sixteen (AB_length BC_length area_ABD : Real) :
  AB_length = 5 → BC_length = 8 → area_ABD = 10 → (∃ area_BCD : Real, area_BCD = 16) :=
by
  sorry

-- Given conditions from part (b)
variable (new_area_ABD : Real)

-- Define the new area.
axiom new_area_ABD_eq_hundred : new_area_ABD = 100

-- Part (b) problem statement
theorem area_BCD_with_new_ABD (AB_length BC_length new_area_ABD : Real) :
  AB_length = 5 → BC_length = 8 → new_area_ABD = 100 → (∃ area_BCD : Real, area_BCD = 160) :=
by
  sorry

end area_BCD_sixteen_area_BCD_with_new_ABD_l131_131260


namespace pipe_length_l131_131755

theorem pipe_length (S L : ℕ) (h1: S = 28) (h2: L = S + 12) : S + L = 68 := 
by
  sorry

end pipe_length_l131_131755


namespace probability_drop_l131_131953

open Real

noncomputable def probability_of_oil_drop_falling_in_hole (c : ℝ) : ℝ :=
  (0.25 * c^2) / (π * (c^2 / 4))

theorem probability_drop (c : ℝ) (hc : c > 0) : 
  probability_of_oil_drop_falling_in_hole c = 0.25 / π :=
by
  sorry

end probability_drop_l131_131953


namespace parity_equivalence_l131_131891

theorem parity_equivalence (p q : ℕ) :
  (Even (p^3 - q^3)) ↔ (Even (p + q)) :=
by
  sorry

end parity_equivalence_l131_131891


namespace inequality_solution_l131_131607

theorem inequality_solution (x : ℝ) : (4 + 2 * x > -6) → (x > -5) :=
by sorry

end inequality_solution_l131_131607


namespace arithmetic_seq_problem_l131_131956

theorem arithmetic_seq_problem (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a n = a 1 + (n - 1) * d)
  (h_cond : a 1 + 3 * a 8 + a 15 = 60) :
  2 * a 9 - a 10 = 12 := 
sorry

end arithmetic_seq_problem_l131_131956


namespace math_problem_l131_131342

-- Define constants and conversions from decimal/mixed numbers to fractions
def thirteen_and_three_quarters : ℚ := 55 / 4
def nine_and_sixth : ℚ := 55 / 6
def one_point_two : ℚ := 1.2
def ten_point_three : ℚ := 103 / 10
def eight_and_half : ℚ := 17 / 2
def six_point_eight : ℚ := 34 / 5
def three_and_three_fifths : ℚ := 18 / 5
def five_and_five_sixths : ℚ := 35 / 6
def three_and_two_thirds : ℚ := 11 / 3
def three_and_one_sixth : ℚ := 19 / 6
def fifty_six : ℚ := 56
def twenty_seven_and_sixth : ℚ := 163 / 6

def E : ℚ := 
  ((thirteen_and_three_quarters + nine_and_sixth) * one_point_two) / ((ten_point_three - eight_and_half) * (5 / 9)) + 
  ((six_point_eight - three_and_three_fifths) * five_and_five_sixths) / ((three_and_two_thirds - three_and_one_sixth) * fifty_six) - 
  twenty_seven_and_sixth

theorem math_problem : E = 29 / 3 := by
  sorry

end math_problem_l131_131342


namespace quadrilateral_area_l131_131899

theorem quadrilateral_area (a b : ℤ) (h1 : a > b) (h2 : b > 0) 
  (h3 : 2 * |a - b| * |a + b| = 32) : a + b = 8 :=
by
  sorry

end quadrilateral_area_l131_131899


namespace count_perfect_squares_divisible_by_36_l131_131548

theorem count_perfect_squares_divisible_by_36 :
  let N := 10000
  let max_square := 10^8
  let multiple := 36
  let valid_divisor := 1296
  let count_multiples := 277
  (∀ N : ℕ, N^2 < max_square → (∃ k : ℕ, N = k * multiple ∧ k < N)) → 
  ∃ cnt : ℕ, cnt = count_multiples := 
by {
  sorry
}

end count_perfect_squares_divisible_by_36_l131_131548


namespace find_x_l131_131522

-- Define that x is a real number and positive
variable (x : ℝ)
variable (hx_pos : 0 < x)

-- Define the floor function and the main equation
variable (hx_eq : ⌊x⌋ * x = 90)

theorem find_x (h : ⌊x⌋ * x = 90) (hx_pos : 0 < x) : ⌊x⌋ = 9 ∧ x = 10 :=
by
  sorry

end find_x_l131_131522


namespace smallest_number_of_eggs_l131_131013

-- Define the conditions given in the problem
def total_containers (c : ℕ) : ℕ := 15 * c - 3

-- Prove that given the conditions, the smallest number of eggs you could have is 162
theorem smallest_number_of_eggs (h : ∃ c : ℕ, total_containers c > 150) : ∃ c : ℕ, total_containers c = 162 :=
by
  sorry

end smallest_number_of_eggs_l131_131013


namespace min_contribution_proof_l131_131631

noncomputable def min_contribution (total_contribution : ℕ) (num_people : ℕ) (max_contribution: ℕ) :=
  ∃ (min_each_person: ℕ), num_people * min_each_person ≤ total_contribution ∧ max_contribution * (num_people - 1) + min_each_person ≥ total_contribution ∧ min_each_person = 2

theorem min_contribution_proof :
  min_contribution 30 15 16 :=
sorry

end min_contribution_proof_l131_131631


namespace relationship_between_x_and_z_l131_131978

-- Definitions of the given conditions
variable {x y z : ℝ}

-- Statement of the theorem
theorem relationship_between_x_and_z (h1 : x = 1.027 * y) (h2 : y = 0.45 * z) : x = 0.46215 * z :=
by
  sorry

end relationship_between_x_and_z_l131_131978


namespace compute_expression_l131_131048

theorem compute_expression :
  (3 + 3 / 8) ^ (2 / 3) - (5 + 4 / 9) ^ (1 / 2) + 0.008 ^ (2 / 3) / 0.02 ^ (1 / 2) * 0.32 ^ (1 / 2) / 0.0625 ^ (1 / 4) = 43 / 150 := 
sorry

end compute_expression_l131_131048


namespace factorize_one_factorize_two_l131_131925

variable (m x y : ℝ)

-- Problem statement for Question 1
theorem factorize_one (m : ℝ) : 
  2 * m^2 - 8 = 2 * (m + 2) * (m - 2) := 
sorry

-- Problem statement for Question 2
theorem factorize_two (x y : ℝ) : 
  (x + y)^2 - 4 * (x + y) + 4 = (x + y - 2)^2 := 
sorry

end factorize_one_factorize_two_l131_131925


namespace wall_width_is_correct_l131_131896

-- Definitions based on the conditions
def brick_length : ℝ := 25  -- in cm
def brick_height : ℝ := 11.25  -- in cm
def brick_width : ℝ := 6  -- in cm
def num_bricks : ℝ := 5600
def wall_length : ℝ := 700  -- 7 m in cm
def wall_height : ℝ := 600  -- 6 m in cm
def total_volume : ℝ := num_bricks * (brick_length * brick_height * brick_width)

-- Prove that the inferred width of the wall is correct
theorem wall_width_is_correct : (total_volume / (wall_length * wall_height)) = 22.5 := by
  sorry

end wall_width_is_correct_l131_131896


namespace solution_set_of_inequality_l131_131724

theorem solution_set_of_inequality :
  {x : ℝ | (x - 1) * (2 - x) ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l131_131724


namespace digit_B_l131_131656

def is_valid_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 7

def unique_digits (A B C D E F G : ℕ) : Prop :=
  is_valid_digit A ∧ is_valid_digit B ∧ is_valid_digit C ∧ is_valid_digit D ∧ 
  is_valid_digit E ∧ is_valid_digit F ∧ is_valid_digit G ∧ 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ 
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ 
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ 
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ 
  E ≠ F ∧ E ≠ G ∧ 
  F ≠ G

def total_sum (A B C D E F G : ℕ) : ℕ :=
  (A + B + C) + (A + E + F) + (C + D + E) + (B + D + G) + (B + F) + (G + E)

theorem digit_B (A B C D E F G : ℕ) 
  (h1 : unique_digits A B C D E F G)
  (h2 : total_sum A B C D E F G = 65) : B = 7 := 
sorry

end digit_B_l131_131656


namespace letter_puzzle_solution_l131_131217

theorem letter_puzzle_solution :
  ∃ (A B : ℕ), (A ≠ B) ∧ (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ A ^ B = B * 10 + A :=
by {
  use 2, 5,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 2 ^ 5 = 5 * 10 + 2),
  sorry
} ∨
by {
  use 6, 2,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 6 ^ 2 = 2 * 10 + 6),
  sorry
} ∨
by {
  use 4, 3,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 4 ^ 3 = 3 * 10 + 4),
  sorry
}

end letter_puzzle_solution_l131_131217


namespace count_perfect_squares_multiple_of_36_l131_131541

theorem count_perfect_squares_multiple_of_36 (N : ℕ) (M : ℕ := 36) :
  (∃ n, n ^ 2 < 10^8 ∧ (M ∣ n ^ 2) ∧ (1 ≤ n) ∧ (n < 10^4)) →
  (M = 36 ∧ ∃ C, ∑ k in finset.range (N + 1), if (M * k < 10^4) then 1 else 0 = C ∧ C = 277) :=
begin
  sorry
end

end count_perfect_squares_multiple_of_36_l131_131541


namespace Carrie_can_add_turnips_l131_131647

-- Define the variables and conditions
def potatoToTurnipRatio (potatoes turnips : ℕ) : ℚ :=
  potatoes / turnips

def pastPotato : ℕ := 5
def pastTurnip : ℕ := 2
def currentPotato : ℕ := 20
def allowedTurnipAddition : ℕ := 8

-- Define the main theorem to prove, given the conditions.
theorem Carrie_can_add_turnips (past_p_ratio : potatoToTurnipRatio pastPotato pastTurnip = 2.5)
                                : potatoToTurnipRatio currentPotato allowedTurnipAddition = 2.5 :=
sorry

end Carrie_can_add_turnips_l131_131647


namespace find_p_l131_131121

theorem find_p
  (p : ℝ)
  (h1 : ∃ (x y : ℝ), p * (x^2 - y^2) = (p^2 - 1) * x * y ∧ |x - 1| + |y| = 1)
  (h2 : ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
         x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
         p * (x₁^2 - y₁^2) = (p^2 - 1) * x₁ * y₁ ∧ |x₁ - 1| + |y₁| = 1 ∧
         p * (x₂^2 - y₂^2) = (p^2 - 1) * x₂ * y₂ ∧ |x₂ - 1| + |y₂| = 1 ∧
         p * (x₃^2 - y₃^2) = (p^2 - 1) * x₃ * y₃ ∧ |x₃ - 1| + |y₃| = 1) :
  p = 1 ∨ p = -1 :=
by sorry

end find_p_l131_131121


namespace range_of_m_l131_131208

def A := { x : ℝ | x^2 - 2 * x - 15 ≤ 0 }
def B (m : ℝ) := { x : ℝ | m - 2 < x ∧ x < 2 * m - 3 }

theorem range_of_m : ∀ m : ℝ, (B m ⊆ A) ↔ (m ≤ 4) :=
by sorry

end range_of_m_l131_131208


namespace letter_puzzle_solutions_l131_131214

theorem letter_puzzle_solutions (A B : ℕ) : 
  (1 ≤ A ∧ A < 10) ∧ (1 ≤ B ∧ B < 10) ∧ (A ≠ B) ∧ (A^B = 10 * B + A) → 
  (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l131_131214


namespace polygon_sides_l131_131566

theorem polygon_sides (n : ℕ) 
  (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
sorry

end polygon_sides_l131_131566


namespace mindy_messages_total_l131_131435

theorem mindy_messages_total (P : ℕ) (h1 : 83 = 9 * P - 7) : 83 + P = 93 :=
  by
    sorry

end mindy_messages_total_l131_131435


namespace per_capita_income_growth_l131_131187

noncomputable def income2020 : ℝ := 3.2
noncomputable def income2022 : ℝ := 3.7
variable (x : ℝ)

/--
Prove the per capita disposable income model.
-/
theorem per_capita_income_growth :
  income2020 * (1 + x)^2 = income2022 :=
sorry

end per_capita_income_growth_l131_131187


namespace ways_to_distribute_items_l131_131491

/-- The number of ways to distribute 5 different items into 4 identical bags, with some bags possibly empty, is 36. -/
theorem ways_to_distribute_items : ∃ (n : ℕ), n = 36 := by
  sorry

end ways_to_distribute_items_l131_131491


namespace determine_a_zeros_l131_131939

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x = 3 then a else 2 / |x - 3|

noncomputable def y (x : ℝ) (a : ℝ) : ℝ := f x a - 4

theorem determine_a_zeros (a : ℝ) : (∃ c d, c ≠ 3 ∧ d ≠ 3 ∧ c ≠ d ∧ y c a = 0 ∧ y d a = 0 ∧ y 3 a = 0) → a = 4 :=
sorry

end determine_a_zeros_l131_131939


namespace are_names_possible_l131_131906

-- Define the structure to hold names
structure Person where
  first_name  : String
  middle_name : String
  last_name   : String

-- List of 4 people
def people : List Person :=
  [{ first_name := "Ivan", middle_name := "Ivanovich", last_name := "Ivanov" },
   { first_name := "Ivan", middle_name := "Petrovich", last_name := "Petrov" },
   { first_name := "Petr", middle_name := "Ivanovich", last_name := "Petrov" },
   { first_name := "Petr", middle_name := "Petrovich", last_name := "Ivanov" }]

-- Define the problem theorem
theorem are_names_possible :
  ∃ (people : List Person), 
    (∀ (p1 p2 p3 : Person), p1 ≠ p2 → p1 ≠ p3 → p2 ≠ p3 → (p1.first_name ≠ p2.first_name ∨ p1.first_name ≠ p3.first_name ∨ p2.first_name ≠ p3.first_name) ∧
    (p1.middle_name ≠ p2.middle_name ∨ p1.middle_name ≠ p3.middle_name ∨ p2.middle_name ≠ p3.middle_name) ∧
    (p1.last_name ≠ p2.last_name ∨ p1.last_name ≠ p3.last_name ∨ p2.last_name ≠ p3.last_name)) ∧
    (∀ (p1 p2 : Person), p1 ≠ p2 → (p1.first_name = p2.first_name ∨ p1.middle_name = p2.middle_name ∨ p1.last_name = p2.last_name)) :=
by
  -- Place proof here
  sorry

end are_names_possible_l131_131906


namespace sphere_radius_l131_131357

theorem sphere_radius (x y z r : ℝ) (h1 : 2 * x * y + 2 * y * z + 2 * z * x = 384)
  (h2 : x + y + z = 28) (h3 : (2 * r)^2 = x^2 + y^2 + z^2) : r = 10 := sorry

end sphere_radius_l131_131357


namespace remainder_six_n_mod_four_l131_131339

theorem remainder_six_n_mod_four (n : ℤ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := 
by sorry

end remainder_six_n_mod_four_l131_131339


namespace order_of_A_B_C_D_l131_131845

def A := Nat.factorial 8 ^ Nat.factorial 8
def B := 8 ^ (8 ^ 8)
def C := 8 ^ 88
def D := 8 ^ 64

theorem order_of_A_B_C_D : D < C ∧ C < B ∧ B < A := by
  sorry

end order_of_A_B_C_D_l131_131845


namespace largest_val_is_E_l131_131625

noncomputable def A : ℚ := 4 / (2 - 1/4)
noncomputable def B : ℚ := 4 / (2 + 1/4)
noncomputable def C : ℚ := 4 / (2 - 1/3)
noncomputable def D : ℚ := 4 / (2 + 1/3)
noncomputable def E : ℚ := 4 / (2 - 1/2)

theorem largest_val_is_E : E > A ∧ E > B ∧ E > C ∧ E > D := 
by sorry

end largest_val_is_E_l131_131625


namespace upper_limit_l131_131952

noncomputable def upper_limit_Arun (w : ℝ) (X : ℝ) : Prop :=
  (w > 66 ∧ w < X) ∧ (w > 60 ∧ w < 70) ∧ (w ≤ 69) ∧ ((66 + X) / 2 = 68)

theorem upper_limit (w : ℝ) (X : ℝ) (h : upper_limit_Arun w X) : X = 69 :=
by sorry

end upper_limit_l131_131952


namespace train_speed_l131_131642

def distance := 11.67 -- distance in km
def time := 10.0 / 60.0 -- time in hours (10 minutes is 10/60 hours)

theorem train_speed : (distance / time) = 70.02 := by
  sorry

end train_speed_l131_131642


namespace part1_part2_l131_131388

-- Part (1): Proving the range of x when a = 1
theorem part1 (x : ℝ) : (x^2 - 6 * 1 * x + 8 < 0) ∧ (x^2 - 4 * x + 3 ≤ 0) ↔ 2 < x ∧ x ≤ 3 := 
by sorry

-- Part (2): Proving the range of a when p is a sufficient but not necessary condition for q
theorem part2 (a : ℝ) : (∀ x : ℝ, (x^2 - 6 * a * x + 8 * a^2 < 0 → x^2 - 4 * x + 3 ≤ 0) 
  ∧ (∃ x : ℝ, x^2 - 4 * x + 3 ≤ 0 ∧ x^2 - 6 * a * x + 8 * a^2 ≥ 0)) ↔ (1/2 ≤ a ∧ a ≤ 3/4) :=
by sorry

end part1_part2_l131_131388


namespace tan_half_product_values_l131_131946

theorem tan_half_product_values (a b : ℝ) (h : 3 * (Real.sin a + Real.sin b) + 2 * (Real.sin a * Real.sin b + 1) = 0) : 
  ∃ x : ℝ, x = Real.tan (a / 2) * Real.tan (b / 2) ∧ (x = -4 ∨ x = -1) := sorry

end tan_half_product_values_l131_131946


namespace competition_results_l131_131585

namespace Competition

-- Define the probabilities for each game
def prob_win_game_A : ℚ := 2 / 3
def prob_win_game_B : ℚ := 1 / 2

-- Define the probability of winning each project (best of five format)
def prob_win_project_A : ℚ := (8 / 27) + (8 / 27) + (16 / 81)
def prob_win_project_B : ℚ := (1 / 8) + (3 / 16) + (3 / 16)

-- Define the distribution of the random variable X (number of projects won by player A)
def P_X_0 : ℚ := (17 / 81) * (1 / 2)
def P_X_2 : ℚ := (64 / 81) * (1 / 2)
def P_X_1 : ℚ := 1 - P_X_0 - P_X_2

-- Define the mathematical expectation of X
def E_X : ℚ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2

-- Theorem stating the results
theorem competition_results :
  prob_win_project_A = 64 / 81 ∧
  prob_win_project_B = 1 / 2 ∧
  P_X_0 = 17 / 162 ∧
  P_X_1 = 81 / 162 ∧
  P_X_2 = 64 / 162 ∧
  E_X = 209 / 162 :=
by sorry

end Competition

end competition_results_l131_131585


namespace maxwell_meets_brad_l131_131577

-- Define the given conditions
def distance_between_homes : ℝ := 94
def maxwell_speed : ℝ := 4
def brad_speed : ℝ := 6
def time_delay : ℝ := 1

-- Define the total time it takes Maxwell to meet Brad
theorem maxwell_meets_brad : ∃ t : ℝ, maxwell_speed * (t + time_delay) + brad_speed * t = distance_between_homes ∧ (t + time_delay = 10) :=
by
  sorry

end maxwell_meets_brad_l131_131577


namespace maria_towels_l131_131165

theorem maria_towels (green_towels white_towels given_towels : ℕ) (h1 : green_towels = 35) (h2 : white_towels = 21) (h3 : given_towels = 34) :
  green_towels + white_towels - given_towels = 22 :=
by
  sorry

end maria_towels_l131_131165


namespace Tim_weekly_earnings_l131_131613

def number_of_tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def working_days_per_week : ℕ := 6

theorem Tim_weekly_earnings :
  (number_of_tasks_per_day * pay_per_task) * working_days_per_week = 720 := by
  sorry

end Tim_weekly_earnings_l131_131613


namespace coefficient_of_pi_x_over_5_l131_131999

-- Definition of the function where we find the coefficient
def coefficient_of_fraction (expr : ℝ) : ℝ := sorry

-- Statement with proof obligation
theorem coefficient_of_pi_x_over_5 :
  coefficient_of_fraction (π * x / 5) = π / 5 :=
sorry

end coefficient_of_pi_x_over_5_l131_131999


namespace cordelia_bleach_time_l131_131506

theorem cordelia_bleach_time (B D : ℕ) (h1 : B + D = 9) (h2 : D = 2 * B) : B = 3 :=
by
  sorry

end cordelia_bleach_time_l131_131506


namespace animath_interns_pigeonhole_l131_131592

theorem animath_interns_pigeonhole (n : ℕ) (knows : Fin n → Finset (Fin n)) :
  ∃ (i j : Fin n), i ≠ j ∧ (knows i).card = (knows j).card :=
by
  sorry

end animath_interns_pigeonhole_l131_131592


namespace factorial_multiple_l131_131129

theorem factorial_multiple (m n : ℕ) : 
  ∃ k : ℕ, k * (m! * n! * (m + n)!) = (2 * m)! * (2 * n)! :=
sorry

end factorial_multiple_l131_131129


namespace projectile_height_35_l131_131601

theorem projectile_height_35 (t : ℝ) : 
  (∃ t : ℝ, -4.9 * t ^ 2 + 30 * t = 35 ∧ t > 0) → t = 10 / 7 := 
sorry

end projectile_height_35_l131_131601


namespace area_rectangle_relation_l131_131595

theorem area_rectangle_relation (x y : ℝ) (h : x * y = 12) : y = 12 / x :=
by
  sorry

end area_rectangle_relation_l131_131595


namespace larger_number_is_391_l131_131868

theorem larger_number_is_391 (A B : ℕ) 
  (hcf : ∀ n : ℕ, n ∣ A ∧ n ∣ B ↔ n = 23)
  (lcm_factors : ∃ C D : ℕ, lcm A B = 23 * 13 * 17 ∧ C = 13 ∧ D = 17) :
  max A B = 391 :=
sorry

end larger_number_is_391_l131_131868


namespace sixth_root_binomial_expansion_l131_131651

theorem sixth_root_binomial_expansion :
  (2748779069441 = 1 * 150^6 + 6 * 150^5 + 15 * 150^4 + 20 * 150^3 + 15 * 150^2 + 6 * 150 + 1) →
  (2748779069441 = Nat.choose 6 6 * 150^6 + Nat.choose 6 5 * 150^5 + Nat.choose 6 4 * 150^4 + Nat.choose 6 3 * 150^3 + Nat.choose 6 2 * 150^2 + Nat.choose 6 1 * 150 + Nat.choose 6 0) →
  (Real.sqrt (2748779069441 : ℝ) = 151) :=
by
  intros h1 h2
  sorry

end sixth_root_binomial_expansion_l131_131651


namespace reciprocal_neg4_l131_131145

def reciprocal (x : ℝ) : ℝ :=
  1 / x

theorem reciprocal_neg4 : reciprocal (-4) = -1 / 4 := by
  sorry

end reciprocal_neg4_l131_131145


namespace store_owner_marked_price_l131_131757

theorem store_owner_marked_price (L M : ℝ) (h1 : M = (56 / 45) * L) : M / L = 124.44 / 100 :=
by
  sorry

end store_owner_marked_price_l131_131757


namespace triangular_pyramid_volume_l131_131265

theorem triangular_pyramid_volume (a b c : ℝ) 
  (h1 : 1 / 2 * a * b = 6) 
  (h2 : 1 / 2 * a * c = 4) 
  (h3 : 1 / 2 * b * c = 3) : 
  (1 / 3) * (1 / 2) * a * b * c = 4 := by 
  sorry

end triangular_pyramid_volume_l131_131265


namespace division_of_neg_six_by_three_l131_131370

theorem division_of_neg_six_by_three : (-6) / 3 = -2 := by
  sorry

end division_of_neg_six_by_three_l131_131370


namespace least_six_digit_divisible_by_198_l131_131464

/-- The least 6-digit natural number that is divisible by 198 is 100188. -/
theorem least_six_digit_divisible_by_198 : 
  ∃ n : ℕ, n ≥ 100000 ∧ n % 198 = 0 ∧ n = 100188 :=
by
  use 100188
  sorry

end least_six_digit_divisible_by_198_l131_131464


namespace problem_2002_multiples_l131_131944

theorem problem_2002_multiples :
  ∃ (n : ℕ), 
    n = 1800 ∧
    (∀ (i j : ℕ), 0 ≤ i ∧ i < j ∧ j ≤ 149 →
      2002 ∣ (10^j - 10^i) ↔ j - i ≡ 0 [MOD 6]) :=
sorry

end problem_2002_multiples_l131_131944


namespace point_equidistant_l131_131427

def A : ℝ × ℝ × ℝ := (10, 0, 0)
def B : ℝ × ℝ × ℝ := (0, -6, 0)
def C : ℝ × ℝ × ℝ := (0, 0, 8)
def D : ℝ × ℝ × ℝ := (0, 0, 0)
def P : ℝ × ℝ × ℝ := (5, -3, 4)

theorem point_equidistant : dist A P = dist B P ∧ dist B P = dist C P ∧ dist C P = dist D P :=
by
  sorry

end point_equidistant_l131_131427


namespace solve_for_y_l131_131812

theorem solve_for_y (y : ℝ) (h : 1 / 4 - 1 / 5 = 4 / y) : y = 80 :=
by
  sorry

end solve_for_y_l131_131812


namespace sum_of_integers_l131_131863

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : 
  x + y = 32 :=
by
  sorry

end sum_of_integers_l131_131863


namespace income_growth_rate_l131_131189

noncomputable def income_growth_eq (x : ℝ) : Prop :=
  3.2 * (1 + x)^2 = 3.7

theorem income_growth_rate :
  ∃ x : ℝ, income_growth_eq x :=
sorry

end income_growth_rate_l131_131189


namespace find_m_l131_131271

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 5
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x - 9

theorem find_m (m : ℝ) : f 5 - g 5 m = 20 → m = -16.8 :=
by
  -- Given f(x) and g(x, m) definitions, we want to prove m = -16.8 given f 5 - g 5 m = 20.
  sorry

end find_m_l131_131271


namespace first_reduction_percentage_l131_131449

theorem first_reduction_percentage (P : ℝ) (x : ℝ) :
  P * (1 - x / 100) * 0.6 = P * 0.45 → x = 25 :=
by
  sorry

end first_reduction_percentage_l131_131449


namespace complex_sum_abs_eq_1_or_3_l131_131298

open Complex

theorem complex_sum_abs_eq_1_or_3
  (a b c : ℂ)
  (ha : abs a = 1)
  (hb : abs b = 1)
  (hc : abs c = 1)
  (h : a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = 1) :
  ∃ r : ℝ, (r = 1 ∨ r = 3) ∧ abs (a + b + c) = r :=
by {
  -- Proof goes here
  sorry
}

end complex_sum_abs_eq_1_or_3_l131_131298


namespace hyperbola_equation_l131_131602

-- Define the hyperbola and its properties
def hyperbola (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :=
  { x y : ℝ // x^2 / a^2 - y^2 / b^2 = 1 }

-- Define the properties of the foci, distance, and slope
variable (a b c : ℝ)
variable (F2 : ℝ × ℝ) (P : ℝ × ℝ)
variable (PF2_dist : ℝ)
variable (slope_PF2 : ℝ)

-- Assume the given conditions
axiom h_a_pos : a > 0
axiom h_b_pos : b > 0
axiom asymptote_slope : slope_PF2 = -1 / 2
axiom distance_PF2 : PF2_dist = 2

-- Assume the focus coordinates and distance to point P
axiom F2_coords : F2 = (c, 0)
axiom P_coords : P = (a^2 / c, ab / c)

-- The theorem statement
theorem hyperbola_equation 
  (h1 : F2 = (sqrt(a^2 + b^2), 0))
  (h2 : PF2_dist = 2)
  (h3 : slope_PF2 = -1 / 2) :
  b = 2 ∧ x^2 - y^2 / 4 = 1 ∧ P = (sqrt(5) / 5, 2sqrt(5) / 5) :=
  sorry

end hyperbola_equation_l131_131602


namespace length_of_faster_train_proof_l131_131480

-- Definitions based on the given conditions
def faster_train_speed_kmh := 72 -- in km/h
def slower_train_speed_kmh := 36 -- in km/h
def time_to_cross_seconds := 18 -- in seconds

-- Conversion factor from km/h to m/s
def kmh_to_ms := 5 / 18

-- Define the relative speed in m/s
def relative_speed_ms := (faster_train_speed_kmh - slower_train_speed_kmh) * kmh_to_ms

-- Length of the faster train in meters
def length_of_faster_train := relative_speed_ms * time_to_cross_seconds

-- The theorem statement for the Lean prover
theorem length_of_faster_train_proof : length_of_faster_train = 180 := by
  sorry

end length_of_faster_train_proof_l131_131480


namespace eccentricity_of_ellipse_l131_131399

theorem eccentricity_of_ellipse {a b c e : ℝ} 
  (h1 : b^2 = 3) 
  (h2 : c = 1 / 4)
  (h3 : a^2 = b^2 + c^2)
  (h4 : a = 7 / 4) 
  : e = c / a → e = 1 / 7 :=
by 
  intros
  sorry

end eccentricity_of_ellipse_l131_131399


namespace cars_meet_after_5_hours_l131_131620

theorem cars_meet_after_5_hours :
  ∀ (t : ℝ), (40 * t + 60 * t = 500) → t = 5 := 
by
  intro t
  intro h
  sorry

end cars_meet_after_5_hours_l131_131620


namespace min_sum_equals_nine_l131_131937

theorem min_sum_equals_nine (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 4 * a + b - a * b = 0) : a + b = 9 :=
by
  sorry

end min_sum_equals_nine_l131_131937


namespace division_of_neg6_by_3_l131_131368

theorem division_of_neg6_by_3 : (-6 : ℤ) / 3 = -2 := 
by
  sorry

end division_of_neg6_by_3_l131_131368


namespace exists_xy_such_that_x2_add_y2_eq_n_mod_p_p_mod_4_eq_1_implies_n_can_be_0_p_mod_4_eq_3_implies_n_cannot_be_0_l131_131658

theorem exists_xy_such_that_x2_add_y2_eq_n_mod_p
  (p : ℕ) [Fact (Nat.Prime p)] (n : ℤ)
  (hp1 : p > 5) :
  (∃ x y : ℤ, x ≠ 0 ∧ y ≠ 0 ∧ (x^2 + y^2) % p = n % p) :=
sorry

theorem p_mod_4_eq_1_implies_n_can_be_0
  (p : ℕ) [Fact (Nat.Prime p)] (hp1 : p % 4 = 1) : 
  (∃ x y : ℤ, x ≠ 0 ∧ y ≠ 0 ∧ (x^2 + y^2) % p = 0) :=
sorry

theorem p_mod_4_eq_3_implies_n_cannot_be_0
  (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 4 = 3) :
  ¬(∃ x y : ℤ, x ≠ 0 ∧ y ≠ 0 ∧ (x^2 + y^2) % p = 0) :=
sorry

end exists_xy_such_that_x2_add_y2_eq_n_mod_p_p_mod_4_eq_1_implies_n_can_be_0_p_mod_4_eq_3_implies_n_cannot_be_0_l131_131658


namespace angle_is_120_degrees_l131_131671

-- Define the magnitudes of vectors a and b and their dot product
def magnitude_a : ℝ := 10
def magnitude_b : ℝ := 12
def dot_product_ab : ℝ := -60

-- Define the angle between vectors a and b
def angle_between_vectors (θ : ℝ) : Prop :=
  magnitude_a * magnitude_b * Real.cos θ = dot_product_ab

-- Prove that the angle θ is 120 degrees
theorem angle_is_120_degrees : angle_between_vectors (2 * Real.pi / 3) :=
by 
  unfold angle_between_vectors
  sorry

end angle_is_120_degrees_l131_131671


namespace sum_consecutive_evens_l131_131586

theorem sum_consecutive_evens (n k : ℕ) (hn : 2 < n) (hk : 2 < k) : 
  ∃ (m : ℕ), n * (n - 1)^(k - 1) = n * (2 * m + (n - 1)) :=
by
  sorry

end sum_consecutive_evens_l131_131586


namespace smallest_rel_prime_to_180_l131_131240

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, (y > 1 ∧ Nat.gcd y 180 = 1) → x ≤ y :=
begin
  sorry
end

end smallest_rel_prime_to_180_l131_131240


namespace distance_to_valley_l131_131348

theorem distance_to_valley (car_speed_kph : ℕ) (time_seconds : ℕ) (sound_speed_mps : ℕ) 
  (car_speed_mps : ℕ) (distance_by_car : ℕ) (distance_by_sound : ℕ) 
  (total_distance_equation : 2 * x + distance_by_car = distance_by_sound) : x = 640 :=
by
  have car_speed_kph := 72
  have time_seconds := 4
  have sound_speed_mps := 340
  have car_speed_mps := car_speed_kph * 1000 / 3600
  have distance_by_car := time_seconds * car_speed_mps
  have distance_by_sound := time_seconds * sound_speed_mps
  have total_distance_equation := (2 * x + distance_by_car = distance_by_sound)
  exact sorry

end distance_to_valley_l131_131348


namespace inequality_sum_l131_131397

variables {a b c : ℝ}

theorem inequality_sum (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a^2 + b^2 + c^2 :=
sorry

end inequality_sum_l131_131397


namespace letter_puzzle_solutions_l131_131224

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def to_number (B A : ℕ) : ℕ :=
  10 * B + A

theorem letter_puzzle_solutions (A B : ℕ) (h_diff : A ≠ B) (h_digits : 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9)
  (h_two_digit : is_two_digit (to_number B A)) :
  A^B = to_number B A ↔ (A = 2 ∧ B = 5 ∨ A = 6 ∧ B = 2 ∨ A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l131_131224


namespace kite_area_is_28_l131_131785

noncomputable def area_of_kite : ℝ :=
  let base_upper := 8
  let height_upper := 2
  let base_lower := 8
  let height_lower := 5
  let area_upper := (1 / 2 : ℝ) * base_upper * height_upper
  let area_lower := (1 / 2 : ℝ) * base_lower * height_lower
  area_upper + area_lower

theorem kite_area_is_28 :
  area_of_kite = 28 :=
by
  simp [area_of_kite]
  sorry

end kite_area_is_28_l131_131785


namespace sequence_a_n_is_n_l131_131432

-- Definitions and statements based on the conditions
def sequence_cond (a : ℕ → ℕ) (n : ℕ) : ℕ := 
1 / 2 * (a n) ^ 2 + n / 2

theorem sequence_a_n_is_n :
  ∀ (a : ℕ → ℕ), (∀ n, n > 0 → ∃ (S_n : ℕ), S_n = sequence_cond a n) → 
  (∀ n, n > 0 → a n = n) :=
by
  sorry

end sequence_a_n_is_n_l131_131432


namespace mass_ratio_speed_ratio_l131_131167

variable {m1 m2 : ℝ} -- masses of the two balls
variable {V0 V : ℝ} -- velocities before and after collision
variable (h1 : V = 4 * V0) -- speed of m2 is four times that of m1 after collision

theorem mass_ratio (h2 :  m1 * V0^2 = m1 * V^2 + 16 * m2 * V^2)
                   (h3 : m1 * V0 = m1 * V + 4 * m2 * V) :
  m2 / m1 = 1 / 2 := sorry

theorem speed_ratio (h2 :  m1 * V0^2 = m1 * V^2 + 16 * m2 * V^2)
                    (h3 : m1 * V0 = m1 * V + 4 * m2 * V)
                    (h4 : m2 / m1 = 1 / 2) :
  V0 / V = 3 := sorry

end mass_ratio_speed_ratio_l131_131167


namespace baseball_football_difference_is_five_l131_131829

-- Define the conditions
def total_cards : ℕ := 125
def baseball_cards : ℕ := 95
def some_more : ℕ := baseball_cards - 3 * (total_cards - baseball_cards)

-- Define the number of football cards
def football_cards : ℕ := total_cards - baseball_cards

-- Define the difference between the number of baseball cards and three times the number of football cards
def difference : ℕ := baseball_cards - 3 * football_cards

-- Statement of the proof
theorem baseball_football_difference_is_five : difference = 5 := 
by
  sorry

end baseball_football_difference_is_five_l131_131829


namespace greatest_integer_solution_l131_131331

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 12 * n + 28 ≤ 0) : 6 ≤ n :=
sorry

end greatest_integer_solution_l131_131331


namespace letter_puzzle_solutions_l131_131213

noncomputable def is_solution (A B : ℕ) : Prop :=
A ≠ B ∧ A ∈ finset.range (10) ∧ B ∈ finset.range (10) ∧ 10 ≤ B * 10 + A ∧ B * 10 + A ≤ 99 ∧ A^B = B * 10 + A

theorem letter_puzzle_solutions :
  ∃ A B : ℕ, is_solution A B ∧ ((A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3)) :=
by
  sorry

end letter_puzzle_solutions_l131_131213


namespace distinct_banners_count_l131_131482

def colors : Finset String := 
  {"red", "white", "blue", "green", "yellow"}

def valid_banners (strip1 strip2 strip3 : String) : Prop :=
  strip1 ∈ colors ∧ strip2 ∈ colors ∧ strip3 ∈ colors ∧
  strip1 ≠ strip2 ∧ strip2 ≠ strip3 ∧ strip3 ≠ strip1

theorem distinct_banners_count : 
  ∃ (banners : Finset (String × String × String)), 
    (∀ s1 s2 s3, (s1, s2, s3) ∈ banners ↔ valid_banners s1 s2 s3) ∧
    banners.card = 60 :=
by
  sorry

end distinct_banners_count_l131_131482


namespace sum_of_coefficients_eq_zero_l131_131726

theorem sum_of_coefficients_eq_zero :
  polynomial.sum_of_coefficients (polynomial.expand (λ x, (x - 2) * (x - 1) ^ 5)) = 0 := by
sorry

end sum_of_coefficients_eq_zero_l131_131726


namespace possible_values_for_D_l131_131690

def distinct_digits (A B C D E : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E

def digits_range (A B C D E : ℕ) : Prop :=
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧ 0 ≤ E ∧ E ≤ 9

def addition_equation (A B C D E : ℕ) : Prop :=
  A * 10000 + B * 1000 + C * 100 + D * 10 + B +
  B * 10000 + C * 1000 + A * 100 + D * 10 + E = 
  E * 10000 + D * 1000 + D * 100 + E * 10 + E

theorem possible_values_for_D : 
  ∀ (A B C D E : ℕ),
  distinct_digits A B C D E →
  digits_range A B C D E →
  addition_equation A B C D E →
  ∃ (S : Finset ℕ), (∀ d ∈ S, 0 ≤ d ∧ d ≤ 9) ∧ (S.card = 2) :=
by
  -- Proof omitted
  sorry

end possible_values_for_D_l131_131690


namespace sum_of_interior_angles_10th_polygon_l131_131564

theorem sum_of_interior_angles_10th_polygon (n : ℕ) (h1 : n = 10) : 
  180 * (n - 2) = 1440 :=
by
  sorry

end sum_of_interior_angles_10th_polygon_l131_131564


namespace count_terminating_decimals_l131_131256

theorem count_terminating_decimals :
  {n : ℕ | 1 ≤ n ∧ n ≤ 150 ∧ ∃ k : ℕ, n = 3 * k}.to_finset.card = 50 := by
sorry

end count_terminating_decimals_l131_131256


namespace angle_A_120_l131_131110

variable {a b c : ℝ}
variable {A B C : ℝ}

theorem angle_A_120 
  (h₁ : a^2 - b^2 = 3 * b * c)
  (h₂ : sin C = 2 * sin B) :
  A = 120 :=
sorry

end angle_A_120_l131_131110


namespace share_difference_l131_131488

theorem share_difference 
  (S : ℝ) -- Total sum of money
  (A B C D : ℝ) -- Shares of a, b, c, d respectively
  (h_proportion : A = 5 / 14 * S)
  (h_proportion : B = 2 / 14 * S)
  (h_proportion : C = 4 / 14 * S)
  (h_proportion : D = 3 / 14 * S)
  (h_d_share : D = 1500) :
  C - D = 500 :=
sorry

end share_difference_l131_131488


namespace intern_knows_same_number_l131_131590

theorem intern_knows_same_number (n : ℕ) (h : n > 1) : 
  ∃ (a b : fin n), a ≠ b ∧ 
  ∃ (f : fin n → ℕ), f a = f b ∧ ∀ i, 0 ≤ f i ∧ f i < n - 1 :=
begin
  sorry,
end

end intern_knows_same_number_l131_131590


namespace center_of_the_hyperbola_l131_131920

def hyperbola_eq (x y : ℝ) : Prop := 9 * x^2 - 54 * x - 36 * y^2 + 288 * y - 576 = 0

structure Point where
  x : ℝ
  y : ℝ

def center_of_hyperbola_is (p : Point) : Prop :=
  hyperbola_eq (p.x + 3) (p.y + 4)

theorem center_of_the_hyperbola :
  ∀ x y : ℝ, hyperbola_eq x y → center_of_hyperbola_is {x := 3, y := 4} :=
by
  intros x y h
  sorry

end center_of_the_hyperbola_l131_131920


namespace cos_sin_225_deg_l131_131199

theorem cos_sin_225_deg : (Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2) ∧ (Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2) :=
by
  -- Lean proof steps would go here
  sorry

end cos_sin_225_deg_l131_131199


namespace train_speed_l131_131182

/-- A train that crosses a pole in a certain time of 7 seconds and is 210 meters long has a speed of 108 kilometers per hour. -/
theorem train_speed (time_to_cross: ℝ) (length_of_train: ℝ) (speed_kmh : ℝ) 
  (H_time: time_to_cross = 7) (H_length: length_of_train = 210) 
  (conversion_factor: ℝ := 3.6) : speed_kmh = 108 :=
by
  have speed_mps : ℝ := length_of_train / time_to_cross
  have speed_kmh_calc : ℝ := speed_mps * conversion_factor
  sorry

end train_speed_l131_131182


namespace leaves_decrease_by_four_fold_l131_131197

theorem leaves_decrease_by_four_fold (x y : ℝ) (h1 : y ≤ x / 4) : 
  9 * y ≤ (9 * x) / 4 := by 
  sorry

end leaves_decrease_by_four_fold_l131_131197


namespace range_of_a_l131_131070

-- Definitions of propositions p and q

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem stating the range of values for a given p ∧ q is true

theorem range_of_a (a : ℝ) : (p a ∧ q a) → (a ≤ -2 ∨ a = 1) :=
by sorry

end range_of_a_l131_131070


namespace harris_carrot_cost_l131_131270

-- Definitions stemming from the conditions
def carrots_per_day : ℕ := 1
def days_per_year : ℕ := 365
def carrots_per_bag : ℕ := 5
def cost_per_bag : ℕ := 2

-- Prove that Harris's total cost for carrots in one year is $146
theorem harris_carrot_cost : (days_per_year * carrots_per_day / carrots_per_bag) * cost_per_bag = 146 := by
  sorry

end harris_carrot_cost_l131_131270


namespace find_d_l131_131777

theorem find_d (d : ℝ) (h₁ : ∃ x, x = ⌊d⌋ ∧ 3 * x^2 + 19 * x - 84 = 0)
                (h₂ : ∃ y, y = d - ⌊d⌋ ∧ 5 * y^2 - 28 * y + 12 = 0 ∧ 0 ≤ y ∧ y < 1) :
  d = 3.2 :=
by
  sorry

end find_d_l131_131777


namespace find_a_b_find_k_range_l131_131082

-- Define the conditions for part 1
def quad_inequality (a x : ℝ) : Prop :=
  a * x^2 - 3 * x + 2 > 0

def solution_set (x b : ℝ) : Prop :=
  x < 1 ∨ x > b

theorem find_a_b (a b : ℝ) :
  (∀ x, quad_inequality a x ↔ solution_set x b) → (a = 1 ∧ b = 2) :=
sorry

-- Define the conditions for part 2
def valid_x_y (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

def equation1 (a b x y : ℝ) : Prop :=
  a / x + b / y = 1

def inequality1 (x y k : ℝ) : Prop :=
  2 * x + y ≥ k^2 + k + 2

theorem find_k_range (a b : ℝ) (x y k : ℝ) :
  a = 1 → b = 2 → valid_x_y x y → equation1 a b x y → inequality1 x y k →
  (-3 ≤ k ∧ k ≤ 2) :=
sorry

end find_a_b_find_k_range_l131_131082


namespace cylinder_surface_area_l131_131900

/-- A right cylinder with radius 3 inches and height twice the radius has a total surface area of 54π square inches. -/
theorem cylinder_surface_area (r : ℝ) (h : ℝ) (A_total : ℝ) (π : ℝ) : r = 3 → h = 2 * r → π = Real.pi → A_total = 54 * π :=
by
  sorry

end cylinder_surface_area_l131_131900


namespace Jason_age_l131_131830

theorem Jason_age : ∃ J K : ℕ, (J = 7 * K) ∧ (J + 4 = 3 * (2 * (K + 2))) ∧ (J = 56) :=
by
  sorry

end Jason_age_l131_131830


namespace product_of_roots_quadratic_l131_131228

noncomputable def product_of_roots (a b c : ℚ) : ℚ :=
  c / a

theorem product_of_roots_quadratic : product_of_roots 14 21 (-250) = -125 / 7 :=
by
  sorry

end product_of_roots_quadratic_l131_131228


namespace minimum_racing_stripes_l131_131821

variable 
  (totalCars : ℕ) (carsWithoutAirConditioning : ℕ) 
  (maxCarsWithAirConditioningWithoutStripes : ℕ)

-- Defining specific problem conditions
def conditions (totalCars carsWithoutAirConditioning maxCarsWithAirConditioningWithoutStripes : ℕ) : Prop :=
  totalCars = 100 ∧ 
  carsWithoutAirConditioning = 37 ∧ 
  maxCarsWithAirConditioningWithoutStripes = 59

-- The statement to be proved
theorem minimum_racing_stripes (h : conditions totalCars carsWithoutAirConditioning maxCarsWithAirConditioningWithoutStripes) :
   exists (R : ℕ ), R = 4 :=
by
  rcases h with ⟨h1, h2, h3⟩
  sorry

end minimum_racing_stripes_l131_131821


namespace solve_equation_l131_131307

theorem solve_equation : 
  ∀ x : ℝ, (x - 3 ≠ 0) → (x + 6) / (x - 3) = 4 → x = 6 :=
by
  intros x h1 h2
  sorry

end solve_equation_l131_131307


namespace square_of_real_not_always_positive_l131_131327

theorem square_of_real_not_always_positive (a : ℝ) : ¬(a^2 > 0) := 
sorry

end square_of_real_not_always_positive_l131_131327


namespace quadratic_inequality_solutions_l131_131511

theorem quadratic_inequality_solutions (k : ℝ) :
  (0 < k ∧ k < 16) ↔ ∃ x : ℝ, x^2 - 8*x + k < 0 :=
sorry

end quadratic_inequality_solutions_l131_131511


namespace min_blocks_for_wall_l131_131026

theorem min_blocks_for_wall (len height : ℕ) (blocks : ℕ → ℕ → ℕ)
  (block_1 : ℕ) (block_2 : ℕ) (block_3 : ℕ) :
  len = 120 → height = 9 →
  block_3 = 1 → block_2 = 2 → block_1 = 3 →
  blocks 5 41 + blocks 4 40 = 365 :=
by
  sorry

end min_blocks_for_wall_l131_131026


namespace minimum_value_of_reciprocals_l131_131267

theorem minimum_value_of_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hline : a - b = 1) :
  (1 / a) + (1 / b) ≥ 4 :=
sorry

end minimum_value_of_reciprocals_l131_131267


namespace probability_gt_2_of_dice_roll_l131_131477

theorem probability_gt_2_of_dice_roll : 
  (∃ outcomes : finset ℕ, 
   outcomes = {1, 2, 3, 4, 5, 6} ∧ 
   (∃ favorable : finset ℕ, 
    favorable = {3, 4, 5, 6} ∧ 
    (favorable.card / outcomes.card : ℚ) = (2 / 3 : ℚ))) := 
sorry

end probability_gt_2_of_dice_roll_l131_131477


namespace field_width_l131_131869

theorem field_width (W L : ℝ) (h1 : L = (7 / 5) * W) (h2 : 2 * L + 2 * W = 288) : W = 60 :=
by
  sorry

end field_width_l131_131869


namespace find_sum_of_xy_l131_131655

theorem find_sum_of_xy (x y : ℝ) (hx_ne_y : x ≠ y) (hx_nonzero : x ≠ 0) (hy_nonzero : y ≠ 0)
  (h_equation : x^4 - 2018 * x^3 - 2018 * y^2 * x = y^4 - 2018 * y^3 - 2018 * y * x^2) :
  x + y = 2018 :=
sorry

end find_sum_of_xy_l131_131655


namespace final_points_l131_131376

-- Definitions of the points in each round
def first_round_points : Int := 16
def second_round_points : Int := 33
def last_round_points : Int := -48

-- The theorem to prove Emily's final points
theorem final_points :
  first_round_points + second_round_points + last_round_points = 1 :=
by
  sorry

end final_points_l131_131376


namespace find_stream_speed_l131_131874

variable (boat_speed dist_downstream dist_upstream : ℝ)
variable (stream_speed : ℝ)

noncomputable def speed_of_stream (boat_speed dist_downstream dist_upstream : ℝ) : ℝ :=
  let t_downstream := dist_downstream / (boat_speed + stream_speed)
  let t_upstream := dist_upstream / (boat_speed - stream_speed)
  if t_downstream = t_upstream then stream_speed else 0

theorem find_stream_speed
  (h : speed_of_stream 20 26 14 stream_speed = stream_speed) :
  stream_speed = 6 :=
sorry

end find_stream_speed_l131_131874


namespace monotonicity_f_f_gt_lower_bound_l131_131404

-- Definition of the function
def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

-- Statement 1: Monotonicity discussion
theorem monotonicity_f (a : ℝ) :
  (a ≤ 0 → ∀ x : ℝ, (f a)' x < 0) ∧ 
  (a > 0 → ∀ x : ℝ,
    (f a)' x < 0 ∧ x < Real.log (1 / a) ∨
    (f a)' x > 0 ∧ x > Real.log (1 / a)) :=
sorry

-- Statement 2: Proof for f(x) > 2 ln a + 3/2 for a > 0
theorem f_gt_lower_bound (a x : ℝ) (ha : 0 < a) :
  f a x > 2 * Real.log a + 3 / 2 :=
sorry

end monotonicity_f_f_gt_lower_bound_l131_131404


namespace factorize_expression_l131_131210

-- Define the variables m and n
variables (m n : ℝ)

-- The statement to prove
theorem factorize_expression : -8 * m^2 + 2 * m * n = -2 * m * (4 * m - n) :=
sorry

end factorize_expression_l131_131210


namespace math_problem_proof_l131_131670

-- Define the system of equations
structure equations :=
  (x y m : ℝ)
  (eq1 : x + 2*y - 6 = 0)
  (eq2 : x - 2*y + m*x + 5 = 0)

-- Define the problem conditions and prove the required solutions in Lean 4
theorem math_problem_proof :
  -- Part 1: Positive integer solutions for x + 2y - 6 = 0
  (∀ x y : ℕ, x + 2*y = 6 → (x, y) = (2, 2) ∨ (x, y) = (4, 1)) ∧
  -- Part 2: Given x + y = 0, find m
  (∀ x y : ℝ, x + y = 0 → x + 2*y - 6 = 0 → x - 2*y - (13/6)*x + 5 = 0) ∧
  -- Part 3: Fixed solution for x - 2y + mx + 5 = 0
  (∀ m : ℝ, 0 - 2*2.5 + m*0 + 5 = 0) :=
sorry

end math_problem_proof_l131_131670


namespace letter_puzzle_solutions_l131_131212

noncomputable def is_solution (A B : ℕ) : Prop :=
A ≠ B ∧ A ∈ finset.range (10) ∧ B ∈ finset.range (10) ∧ 10 ≤ B * 10 + A ∧ B * 10 + A ≤ 99 ∧ A^B = B * 10 + A

theorem letter_puzzle_solutions :
  ∃ A B : ℕ, is_solution A B ∧ ((A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3)) :=
by
  sorry

end letter_puzzle_solutions_l131_131212


namespace bleaching_takes_3_hours_l131_131503

-- Define the total time and the relationship between dyeing and bleaching.
def total_time : ℕ := 9
def dyeing_takes_twice (H : ℕ) : Prop := 2 * H + H = total_time

-- Prove that bleaching takes 3 hours.
theorem bleaching_takes_3_hours : ∃ H : ℕ, dyeing_takes_twice H ∧ H = 3 := 
by 
  sorry

end bleaching_takes_3_hours_l131_131503


namespace not_right_triangle_A_l131_131741

def is_right_triangle (a b c : Real) : Prop :=
  a^2 + b^2 = c^2

theorem not_right_triangle_A : ¬ (is_right_triangle 1.5 2 3) :=
by sorry

end not_right_triangle_A_l131_131741


namespace martha_painting_rate_l131_131838

noncomputable def martha_square_feet_per_hour
  (width1 : ℕ) (width2 : ℕ) (height : ℕ) (coats : ℕ) (total_hours : ℕ) 
  (pair1_walls : ℕ) (pair2_walls : ℕ) : ℕ :=
  let pair1_total_area := width1 * height * pair1_walls
  let pair2_total_area := width2 * height * pair2_walls
  let total_area := pair1_total_area + pair2_total_area
  let total_paint_area := total_area * coats
  total_paint_area / total_hours

theorem martha_painting_rate :
  martha_square_feet_per_hour 12 16 10 3 42 2 2 = 40 :=
by
  -- Proof goes here
  sorry

end martha_painting_rate_l131_131838


namespace degree_poly_product_l131_131854

open Polynomial

-- Given conditions: p and q are polynomials with specified degrees
variables {R : Type*} [CommRing R]
variable (p q : R[X])
variable (hp : degree p = 3)
variable (hq : degree q = 6)

-- Proposition: The degree of p(x^2) * q(x^4) is 30
theorem degree_poly_product : degree (p.comp ((X : R[X])^2) * (q.comp ((X : R[X])^4))) = 30 :=
by sorry

end degree_poly_product_l131_131854


namespace income_growth_rate_l131_131188

noncomputable def income_growth_eq (x : ℝ) : Prop :=
  3.2 * (1 + x)^2 = 3.7

theorem income_growth_rate :
  ∃ x : ℝ, income_growth_eq x :=
sorry

end income_growth_rate_l131_131188


namespace range_of_expression_l131_131792

theorem range_of_expression (x y : ℝ) (h1 : x * y = 1) (h2 : 3 ≥ x ∧ x ≥ 4 * y ∧ 4 * y > 0) :
  ∃ A B, A = 4 ∧ B = 5 ∧ ∀ z, z = (x^2 + 4 * y^2) / (x - 2 * y) → 4 ≤ z ∧ z ≤ 5 :=
by
  sorry

end range_of_expression_l131_131792


namespace omega_terms_sum_to_zero_l131_131966

theorem omega_terms_sum_to_zero {ω : ℂ} (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^12 + ω^15 + ω^18 + ω^21 + ω^24 = 0 :=
by sorry

end omega_terms_sum_to_zero_l131_131966


namespace complete_the_square_l131_131619

theorem complete_the_square (x : ℝ) : 
  (x^2 - 8 * x + 10 = 0) → 
  ((x - 4)^2 = 6) :=
sorry

end complete_the_square_l131_131619


namespace total_gold_coins_l131_131362

/--
An old man distributed all the gold coins he had to his two sons into 
two different numbers such that the difference between the squares 
of the two numbers is 49 times the difference between the two numbers. 
Prove that the total number of gold coins the old man had is 49.
-/
theorem total_gold_coins (x y : ℕ) (h : x ≠ y) (h1 : x^2 - y^2 = 49 * (x - y)) : x + y = 49 :=
sorry

end total_gold_coins_l131_131362


namespace tangent_line_at_x_equals_1_monotonic_intervals_range_of_a_l131_131076

noncomputable def f (a x : ℝ) := a * x + Real.log x

theorem tangent_line_at_x_equals_1 (a : ℝ) (x : ℝ) (h₀ : a = 2) (h₁ : x = 1) : 
  3 * x - (f a 1) - 1 = 0 := 
sorry

theorem monotonic_intervals (a x : ℝ) (h₀ : x > 0) :
  ((a >= 0 ∧ ∀ (x : ℝ), x > 0 → (f a x) > (f a (x - 1))) ∨ 
  (a < 0 ∧ ∀ (x : ℝ), 0 < x ∧ x < -1/a → (f a x) > (f a (x - 1)) ∧ ∀ (x : ℝ), x > -1/a → (f a x) < (f a (x - 1)))) :=
sorry

theorem range_of_a (a x : ℝ) (h₀ : 0 < x) (h₁ : f a x < 2) : a < -1 / Real.exp (3) :=
sorry

end tangent_line_at_x_equals_1_monotonic_intervals_range_of_a_l131_131076


namespace range_of_t_l131_131701

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (a t : ℝ) := 2 * a * t - t^2

theorem range_of_t (t : ℝ) (a : ℝ) (x : ℝ) (h₁ : ∀ x : ℝ, f (-x) = -f x)
                   (h₂ : ∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ 1 → f x₁ ≤ f x₂)
                   (h₃ : f (-1) = -1) (h₄ : -1 ≤ x ∧ x ≤ 1 → f x ≤ t^2 - 2 * a * t + 1)
                   (h₅ : -1 ≤ a ∧ a ≤ 1) :
  t ≥ 2 ∨ t = 0 ∨ t ≤ -2 := sorry

end range_of_t_l131_131701


namespace triple_hash_72_eq_7_25_l131_131206

def hash (N : ℝ) : ℝ := 0.5 * N - 1

theorem triple_hash_72_eq_7_25 : hash (hash (hash 72)) = 7.25 :=
by
  sorry

end triple_hash_72_eq_7_25_l131_131206


namespace displacement_representation_l131_131808

def represents_north (d : ℝ) : Prop := d > 0

theorem displacement_representation (d : ℝ) (h : represents_north 80) : represents_north d ↔ d > 0 :=
by trivial

example (h : represents_north 80) : 
  ∀ d, d = -50 → ¬ represents_north d ∧ abs d = 50 → ∃ s, s = "south" :=
sorry

end displacement_representation_l131_131808


namespace probability_equality_l131_131968

variables {Ω : Type*} [ProbabilitySpace Ω]
variables {N : ℕ} (p q : ℝ) (hp : p > 0) (hq : p + q = 1)

-- Defining the Bernoulli random variables
noncomputable def xi (n : ℕ) : Ω → Bool := 
  λ ω, (Bernoulli (MeasureTheory.probMeasure p)).val ω

-- Sum of i.i.d Bernoulli random variables
noncomputable def S (n : ℕ) : Ω → ℕ
| 0 := 0
| (n+1) := S n + if xi (n+1) then 1 else 0

-- Probability of S_n being equal to k
noncomputable def P_n (n k : ℕ) : ℝ :=
  MeasureTheory.prob (λ ω, S n ω = k)

-- Statement to be proven
theorem probability_equality (n k : ℕ) (hn : n < N) (hk : k ≥ 1) :
  P_n (n+1) k = p * P_n n (k-1) + q * P_n n k :=
sorry

end probability_equality_l131_131968


namespace f_inequality_l131_131405

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_inequality (a : ℝ) (h : a > 0) : ∀ x : ℝ, f x a > 2 * Real.log a + 3 / 2 :=
sorry

end f_inequality_l131_131405


namespace no_integer_solutions_for_equation_l131_131780

theorem no_integer_solutions_for_equation : ¬∃ (a b c : ℤ), a^4 + b^4 = c^4 + 3 := 
  by sorry

end no_integer_solutions_for_equation_l131_131780


namespace largest_angle_in_triangle_l131_131421

theorem largest_angle_in_triangle (A B C : ℝ) 
  (a b c : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hABC : A + B + C = 180) (h_sin : Real.sin A + Real.sin C = Real.sqrt 2 * Real.sin B)
  : B = 90 :=
by
  sorry

end largest_angle_in_triangle_l131_131421


namespace ratio_james_paid_l131_131694

-- Define the parameters of the problem
def packs : ℕ := 4
def stickers_per_pack : ℕ := 30
def cost_per_sticker : ℚ := 0.10
def james_paid : ℚ := 6

-- Total number of stickers
def total_stickers : ℕ := packs * stickers_per_pack
-- Total cost of stickers
def total_cost : ℚ := total_stickers * cost_per_sticker

-- Theorem stating that the ratio of the amount James paid to the total cost of the stickers is 1:2
theorem ratio_james_paid : james_paid / total_cost = 1 / 2 :=
by 
  -- proof goes here
  sorry

end ratio_james_paid_l131_131694


namespace infinite_primes_congruent_3_mod_4_infinite_primes_congruent_5_mod_6_l131_131345

-- Problem 1: Infinitely many primes congruent to 3 modulo 4
theorem infinite_primes_congruent_3_mod_4 :
  ∀ (ps : Finset ℕ), (∀ p ∈ ps, Nat.Prime p ∧ p % 4 = 3) → ∃ q, Nat.Prime q ∧ q % 4 = 3 ∧ q ∉ ps :=
by
  sorry

-- Problem 2: Infinitely many primes congruent to 5 modulo 6
theorem infinite_primes_congruent_5_mod_6 :
  ∀ (ps : Finset ℕ), (∀ p ∈ ps, Nat.Prime p ∧ p % 6 = 5) → ∃ q, Nat.Prime q ∧ q % 6 = 5 ∧ q ∉ ps :=
by
  sorry

end infinite_primes_congruent_3_mod_4_infinite_primes_congruent_5_mod_6_l131_131345


namespace smallest_a_l131_131698

theorem smallest_a 
  (a : ℤ) (P : ℤ → ℤ) 
  (h_pos : 0 < a) 
  (hP1 : P 1 = a) (hP5 : P 5 = a) (hP7 : P 7 = a) (hP9 : P 9 = a) 
  (hP2 : P 2 = -a) (hP4 : P 4 = -a) (hP6 : P 6 = -a) (hP8 : P 8 = -a) : 
  a ≥ 336 :=
by
  sorry

end smallest_a_l131_131698


namespace total_cases_l131_131052

-- Define the number of boys' high schools and girls' high schools
def boys_high_schools : Nat := 4
def girls_high_schools : Nat := 3

-- Theorem to be proven
theorem total_cases (B G : Nat) (hB : B = boys_high_schools) (hG : G = girls_high_schools) : 
  B + G = 7 :=
by
  rw [hB, hG]
  exact rfl

end total_cases_l131_131052


namespace pump_A_time_l131_131646

theorem pump_A_time (B C A : ℝ) (hB : B = 1/3) (hC : C = 1/6)
(h : (A + B - C) * 0.75 = 0.5) : 1 / A = 2 :=
by
sorry

end pump_A_time_l131_131646


namespace original_cost_of_dress_l131_131493

theorem original_cost_of_dress (x : ℝ) 
  (h1 : x / 2 - 10 < x)
  (h2 : x - (x / 2 - 10) = 80) : 
  x = 140 := 
sorry

end original_cost_of_dress_l131_131493


namespace elaine_earnings_l131_131964

variable (E P : ℝ)
variable (H1 : 0.30 * E * (1 + P / 100) = 2.025 * 0.20 * E)

theorem elaine_earnings : P = 35 :=
by
  -- We assume the conditions here and the proof is skipped by sorry.
  sorry

end elaine_earnings_l131_131964


namespace equilateral_triangle_sum_l131_131004

theorem equilateral_triangle_sum (side_length : ℚ) (h_eq : side_length = 13 / 12) :
  3 * side_length = 13 / 4 :=
by
  -- Proof omitted
  sorry

end equilateral_triangle_sum_l131_131004


namespace even_property_of_f_when_a_zero_non_even_odd_property_of_f_when_a_nonzero_minimum_value_of_f_l131_131699

open Real

def f (a x : ℝ) : ℝ := x^2 + |x - a| - 1

theorem even_property_of_f_when_a_zero : 
  ∀ x : ℝ, f 0 x = f 0 (-x) :=
by sorry

theorem non_even_odd_property_of_f_when_a_nonzero : 
  ∀ (a x : ℝ), a ≠ 0 → (f a x ≠ f a (-x) ∧ f a x ≠ -f a (-x)) :=
by sorry

theorem minimum_value_of_f :
  ∀ (a : ℝ), 
    (a ≤ -1/2 → ∃ x : ℝ, f a x = -a - 5/4) ∧ 
    (-1/2 < a ∧ a ≤ 1/2 → ∃ x : ℝ, f a x = a^2 - 1) ∧ 
    (a > 1/2 → ∃ x : ℝ, f a x = a - 5/4) :=
by sorry

end even_property_of_f_when_a_zero_non_even_odd_property_of_f_when_a_nonzero_minimum_value_of_f_l131_131699


namespace two_pow_n_minus_one_prime_imp_n_prime_l131_131969

theorem two_pow_n_minus_one_prime_imp_n_prime (n : ℕ) (h : Nat.Prime (2^n - 1)) : Nat.Prime n := 
sorry

end two_pow_n_minus_one_prime_imp_n_prime_l131_131969


namespace divisor_is_seven_l131_131683

theorem divisor_is_seven 
  (d x : ℤ)
  (h1 : x % d = 5)
  (h2 : 4 * x % d = 6) :
  d = 7 := 
sorry

end divisor_is_seven_l131_131683


namespace prob_yellow_straight_l131_131820

variable {P : ℕ → ℕ → ℚ}
-- Defining the probabilities of the given events
def prob_green : ℚ := 2 / 3
def prob_straight : ℚ := 1 / 2
def prob_rose : ℚ := 1 / 4
def prob_daffodil : ℚ := 1 / 2
def prob_tulip : ℚ := 1 / 4
def prob_rose_straight : ℚ := 1 / 6
def prob_daffodil_curved : ℚ := 1 / 3
def prob_tulip_straight : ℚ := 1 / 8

/-- The probability of picking a yellow and straight-petaled flower is 1/6 -/
theorem prob_yellow_straight : P 1 1 = 1 / 6 := sorry

end prob_yellow_straight_l131_131820


namespace find_x_l131_131885

theorem find_x
  (a b c d k : ℝ)
  (h1 : a ≠ b)
  (h2 : b ≠ 0)
  (h3 : d ≠ 0)
  (h4 : k ≠ 0)
  (h5 : k ≠ 1)
  (h_frac_change : (a + k * x) / (b + x) = c / d) :
  x = (b * c - a * d) / (k * d - c) := by
  sorry

end find_x_l131_131885


namespace verify_b_c_sum_ten_l131_131276

theorem verify_b_c_sum_ten (a b c : ℕ) (ha : 1 ≤ a ∧ a < 10) (hb : 1 ≤ b ∧ b < 10) (hc : 1 ≤ c ∧ c < 10) 
    (h_eq : (10 * b + a) * (10 * c + a) = 100 * b * c + 100 * a + a ^ 2) : b + c = 10 :=
by
  sorry

end verify_b_c_sum_ten_l131_131276


namespace rowing_speed_in_still_water_l131_131032

theorem rowing_speed_in_still_water (speed_of_current : ℝ) (time_seconds : ℝ) (distance_meters : ℝ) (S : ℝ)
  (h_current : speed_of_current = 3) 
  (h_time : time_seconds = 9.390553103577801) 
  (h_distance : distance_meters = 60) 
  (h_S : S = 20) : 
  (distance_meters / 1000) / (time_seconds / 3600) - speed_of_current = S :=
by 
  sorry

end rowing_speed_in_still_water_l131_131032


namespace no_consecutive_primes_sum_65_l131_131914

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (p q : ℕ) : Prop := 
  is_prime p ∧ is_prime q ∧ (q = p + 2 ∨ q = p - 2)

theorem no_consecutive_primes_sum_65 : 
  ¬ ∃ p q : ℕ, consecutive_primes p q ∧ p + q = 65 :=
by 
  sorry

end no_consecutive_primes_sum_65_l131_131914


namespace probability_solution_l131_131974

noncomputable def binom_10_7 := Nat.choose 10 7
noncomputable def binom_10_6 := Nat.choose 10 6

theorem probability_solution (p q : ℝ) (h₁ : q = 1 - p) (h₂ : binom_10_7 = 120) (h₃ : binom_10_6 = 210)
  (h₄ : 120 * p ^ 7 * q ^ 3 = 210 * p ^ 6 * q ^ 4) : p = 7 / 11 := 
sorry

end probability_solution_l131_131974


namespace gcd_X_Y_Z_l131_131293

-- Define the conditions
variables {a b c : ℕ}
variables (X Y Z : ℕ)
def is_digit (n : ℕ) := 0 < n ∧ n < 10

-- Define X, Y, and Z
def X := 10 * a + b
def Y := 10 * b + c
def Z := 10 * c + a

-- The main theorem
theorem gcd_X_Y_Z {a b c : ℕ} (ha : is_digit a) (hb : is_digit b) (hc : is_digit c) (h_distinct: a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  gcd (gcd X Y) Z ∈ {1, 2, 3, 4, 7, 13, 14} :=
sorry

end gcd_X_Y_Z_l131_131293


namespace orange_ring_weight_correct_l131_131571

-- Define the weights as constants
def purple_ring_weight := 0.3333333333333333
def white_ring_weight := 0.4166666666666667
def total_weight := 0.8333333333
def orange_ring_weight := 0.0833333333

-- Theorem statement
theorem orange_ring_weight_correct :
  total_weight - purple_ring_weight - white_ring_weight = orange_ring_weight :=
by
  -- Sorry is added to skip the proof part as per the instruction
  sorry

end orange_ring_weight_correct_l131_131571


namespace probability_diff_faces_l131_131353

def total_lines : ℕ := 15

def total_pairs : ℕ := Nat.choose total_lines 2

def different_faces_pairs : ℕ := 36

def expected_probability : ℚ := 12 / 35

theorem probability_diff_faces : 
  (different_faces_pairs : ℚ) / total_pairs = expected_probability := by
  sorry

end probability_diff_faces_l131_131353


namespace max_sum_abs_coeff_l131_131553

theorem max_sum_abs_coeff (a b c : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : |f 1| ≤ 1)
  (h3 : |f (1/2)| ≤ 1)
  (h4 : |f 0| ≤ 1) :
  |a| + |b| + |c| ≤ 17 :=
sorry

end max_sum_abs_coeff_l131_131553


namespace Amy_crumbs_l131_131166

variable (z : ℕ)

theorem Amy_crumbs (T C : ℕ) (h1 : T * C = z)
  (h2 : ∃ T_A : ℕ, T_A = 2 * T)
  (h3 : ∃ C_A : ℕ, C_A = (3 * C) / 2) :
  ∃ z_A : ℕ, z_A = 3 * z :=
by
  sorry

end Amy_crumbs_l131_131166


namespace value_of_x_l131_131883

theorem value_of_x : 
  ∀ (x : ℕ), x = (2011^2 + 2011) / 2011 → x = 2012 :=
by
  intro x
  intro h
  sorry

end value_of_x_l131_131883


namespace intersection_of_A_and_B_l131_131087

open Set

variable {α : Type*} [LinearOrder α] [ArchimedeanOrderedAddCommGroup α]

def A : Set α := {x | 2^x > 1}
def B : Set α := {x | x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x < 1} :=
by {
  sorry
}

end intersection_of_A_and_B_l131_131087


namespace confidence_interval_for_mean_l131_131036

noncomputable def confidence_interval (samples : Fin 100 → ℝ) (sample_mean : ℝ) (variance : ℝ) (p : ℝ) : set ℝ :=
  {a : ℝ | (sample_mean - 0.233 < a) ∧ (a < sample_mean + 0.233)}

theorem confidence_interval_for_mean
  (a : ℝ)
  (ξ : ℝ → ℝ)
  (sample : Fin 100 → ℝ)
  (sample_mean : ℝ)
  (p : ℝ)
  (hξ : ∀ x, ξ x = Normal a 1)
  (h_sample_mean : sample_mean = 1.3)
  (hp : p = 0.98)
  (hsample_mean : (Finset.univ.sum (λ i, sample i)) / 100 = sample_mean) :
  confidence_interval sample sample_mean 1 p = { x | 1.067 < x ∧ x < 1.533 } :=
by
  sorry

end confidence_interval_for_mean_l131_131036


namespace complex_division_l131_131673

-- Define complex numbers and imaginary unit
def i : ℂ := Complex.I

theorem complex_division : (3 + 4 * i) / (1 + i) = (7 / 2) + (1 / 2) * i :=
by
  sorry

end complex_division_l131_131673


namespace average_monthly_balance_l131_131905

theorem average_monthly_balance :
  let balances := [100, 200, 250, 50, 300, 300]
  (balances.sum / balances.length : ℕ) = 200 :=
by
  sorry

end average_monthly_balance_l131_131905


namespace sum_of_integers_l131_131862

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : 
  x + y = 32 :=
by
  sorry

end sum_of_integers_l131_131862


namespace speed_conversion_l131_131180

theorem speed_conversion (speed_kmh : ℚ) (speed_kmh = 1.5428571428571427) : (speed_kmh / 3.6 = 3 / 7) :=
by
  have conversion_factor : ℚ := 1 / 3.6
  have speed_mps : ℚ := speed_kmh * conversion_factor
  show speed_mps = 3 / 7, from sorry

end speed_conversion_l131_131180


namespace sum_digits_350_1350_base2_l131_131624

def binary_sum_digits (n : ℕ) : ℕ :=
  (Nat.digits 2 n).sum

theorem sum_digits_350_1350_base2 :
  binary_sum_digits 350 + binary_sum_digits 1350 = 20 :=
by
  sorry

end sum_digits_350_1350_base2_l131_131624


namespace length_of_major_axis_l131_131138

def ellipse_length_major_axis (a b : ℝ) : ℝ := 2 * a

theorem length_of_major_axis : ellipse_length_major_axis 4 1 = 8 :=
by
  unfold ellipse_length_major_axis
  norm_num

end length_of_major_axis_l131_131138


namespace man_age_twice_son_age_l131_131753

theorem man_age_twice_son_age (S M X : ℕ) (h1 : S = 28) (h2 : M = S + 30) (h3 : M + X = 2 * (S + X)) : X = 2 :=
by
  sorry

end man_age_twice_son_age_l131_131753


namespace exists_tiling_5x6_no_gaps_no_tiling_5x6_with_gaps_no_tiling_6x6_l131_131512

/-- There exists a way to completely tile a 5x6 board with dominos without leaving any gaps. -/
theorem exists_tiling_5x6_no_gaps :
  ∃ (tiling : List (Set (Fin 5 × Fin 6))), True := 
sorry

/-- It is not possible to tile a 5x6 board with dominos such that gaps are left. -/
theorem no_tiling_5x6_with_gaps :
  ¬ ∃ (tiling : List (Set (Fin 5 × Fin 6))), False := 
sorry

/-- It is impossible to tile a 6x6 board with dominos. -/
theorem no_tiling_6x6 :
  ¬ ∃ (tiling : List (Set (Fin 6 × Fin 6))), True := 
sorry

end exists_tiling_5x6_no_gaps_no_tiling_5x6_with_gaps_no_tiling_6x6_l131_131512


namespace quadratic_function_properties_l131_131066

noncomputable def quadratic_function := 
  -((8:ℝ) / (3:ℝ)) * (Polynomial.X - 1) * (Polynomial.X - 5)

theorem quadratic_function_properties :
  (quadratic_function.eval 1 = 0) ∧ 
  (quadratic_function.eval 5 = 0) ∧ 
  (quadratic_function.eval 2 = 8) :=
by
  sorry

end quadratic_function_properties_l131_131066


namespace frank_money_remaining_l131_131934

-- Define the conditions
def cost_cheapest_lamp : ℕ := 20
def factor_most_expensive : ℕ := 3
def initial_money : ℕ := 90

-- Define the cost of the most expensive lamp
def cost_most_expensive_lamp : ℕ := cost_cheapest_lamp * factor_most_expensive

-- Define the money remaining after purchase
def money_remaining : ℕ := initial_money - cost_most_expensive_lamp

-- The theorem we need to prove
theorem frank_money_remaining : money_remaining = 30 := by
  sorry

end frank_money_remaining_l131_131934


namespace solve_inequality_l131_131308

theorem solve_inequality (x : ℝ) :
  |(3 * x - 2) / (x ^ 2 - x - 2)| > 3 ↔ (x ∈ Set.Ioo (-1) (-2 / 3) ∪ Set.Ioo (1 / 3) 4) :=
by sorry

end solve_inequality_l131_131308


namespace positive_multiples_of_11_ending_with_7_l131_131540

-- Definitions for conditions
def is_multiple_of_11 (n : ℕ) : Prop := (n % 11 = 0)
def ends_with_7 (n : ℕ) : Prop := (n % 10 = 7)

-- Main theorem statement
theorem positive_multiples_of_11_ending_with_7 :
  ∃ n, (n = 13) ∧ ∀ k, is_multiple_of_11 k ∧ ends_with_7 k ∧ 0 < k ∧ k < 1500 → k = 77 + (k / 110) * 110 := 
sorry

end positive_multiples_of_11_ending_with_7_l131_131540


namespace wall_length_correct_l131_131634

noncomputable def length_of_wall : ℝ :=
  let volume_of_one_brick := 25 * 11.25 * 6
  let total_volume_of_bricks := volume_of_one_brick * 6800
  let wall_width := 600
  let wall_height := 22.5
  total_volume_of_bricks / (wall_width * wall_height)

theorem wall_length_correct : length_of_wall = 850 := by
  sorry

end wall_length_correct_l131_131634


namespace count_perfect_squares_multiple_of_36_l131_131542

theorem count_perfect_squares_multiple_of_36 (N : ℕ) (M : ℕ := 36) :
  (∃ n, n ^ 2 < 10^8 ∧ (M ∣ n ^ 2) ∧ (1 ≤ n) ∧ (n < 10^4)) →
  (M = 36 ∧ ∃ C, ∑ k in finset.range (N + 1), if (M * k < 10^4) then 1 else 0 = C ∧ C = 277) :=
begin
  sorry
end

end count_perfect_squares_multiple_of_36_l131_131542


namespace brownies_pieces_l131_131569

theorem brownies_pieces (pan_length pan_width piece_length piece_width : ℕ)
  (h_pan_dims : pan_length = 15) (h_pan_width : pan_width = 25)
  (h_piece_length : piece_length = 3) (h_piece_width : piece_width = 5) :
  (pan_length * pan_width) / (piece_length * piece_width) = 25 :=
by
  sorry

end brownies_pieces_l131_131569


namespace carol_is_inviting_friends_l131_131498

theorem carol_is_inviting_friends :
  ∀ (invitations_per_pack packs_needed friends_invited : ℕ), 
  invitations_per_pack = 2 → 
  packs_needed = 5 → 
  friends_invited = invitations_per_pack * packs_needed → 
  friends_invited = 10 :=
by
  intros invitations_per_pack packs_needed friends_invited h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end carol_is_inviting_friends_l131_131498


namespace probability_heads_equals_7_over_11_l131_131976

theorem probability_heads_equals_7_over_11 (p : ℝ) (q : ℝ)
  (h1 : q = 1 - p)
  (h2 : 120 * p^7 * q^3 = 210 * p^6 * q^4) :
  p = 7 / 11 :=
by {
  sorry
}

end probability_heads_equals_7_over_11_l131_131976


namespace intersection_A_B_l131_131067

def A (x : ℝ) : Prop := (x ≥ 2 ∧ x ≠ 3)
def B (x : ℝ) : Prop := (3 ≤ x ∧ x ≤ 5)
def C := {x : ℝ | 3 < x ∧ x ≤ 5}

theorem intersection_A_B : {x : ℝ | A x} ∩ {x : ℝ | B x} = C :=
  by sorry

end intersection_A_B_l131_131067


namespace sum_two_numbers_eq_twelve_l131_131727

theorem sum_two_numbers_eq_twelve (x y : ℕ) (h1 : x^2 + y^2 = 90) (h2 : x * y = 27) : x + y = 12 :=
by
  sorry

end sum_two_numbers_eq_twelve_l131_131727


namespace tangent_alpha_l131_131786

open Real

noncomputable def a (α : ℝ) : ℝ × ℝ := (sin α, 2)
noncomputable def b (α : ℝ) : ℝ × ℝ := (-cos α, 1)

theorem tangent_alpha (α : ℝ) (h : ∀ k : ℝ, a α = (k • b α)) : tan α = -2 := by
  have h1 : sin α / -cos α = 2 := by sorry
  have h2 : tan α = -2 := by sorry
  exact h2

end tangent_alpha_l131_131786


namespace malvina_correct_l131_131962
noncomputable def angle (x : ℝ) : Prop := 0 < x ∧ x < 180
noncomputable def malvina_identifies (x : ℝ) : Prop := x > 90

noncomputable def sum_of_values := (Real.sqrt 5 + Real.sqrt 2) / 2

theorem malvina_correct (x : ℝ) (h1 : angle x) (h2 : malvina_identifies x) :
  sum_of_values = (Real.sqrt 5 + Real.sqrt 2) / 2 :=
by sorry

end malvina_correct_l131_131962


namespace avg_annual_growth_rate_l131_131193
-- Import the Mathlib library

-- Define the given conditions
def initial_income : ℝ := 32000
def final_income : ℝ := 37000
def period : ℝ := 2
def initial_income_ten_thousands : ℝ := initial_income / 10000
def final_income_ten_thousands : ℝ := final_income / 10000

-- Define the growth rate
variable (x : ℝ)

-- Define the theorem
theorem avg_annual_growth_rate :
  3.2 * (1 + x) ^ 2 = 3.7 :=
sorry

end avg_annual_growth_rate_l131_131193


namespace number_of_cars_in_train_l131_131043

theorem number_of_cars_in_train
  (constant_speed : Prop)
  (cars_in_12_seconds : ℕ)
  (time_to_clear : ℕ)
  (cars_per_second : ℕ → ℕ → ℚ)
  (total_time_seconds : ℕ) :
  cars_in_12_seconds = 8 →
  time_to_clear = 180 →
  cars_per_second cars_in_12_seconds 12 = 2 / 3 →
  total_time_seconds = 180 →
  cars_per_second cars_in_12_seconds 12 * total_time_seconds = 120 :=
by
  sorry

end number_of_cars_in_train_l131_131043


namespace regular_polygon_sides_l131_131857

-- Conditions
def central_angle (θ : ℝ) := θ = 30
def sum_of_central_angles (sumθ : ℝ) := sumθ = 360

-- The proof problem
theorem regular_polygon_sides (θ sumθ : ℝ) (h₁ : central_angle θ) (h₂ : sum_of_central_angles sumθ) :
  sumθ / θ = 12 := by
  sorry

end regular_polygon_sides_l131_131857


namespace probability_at_least_one_girl_l131_131065

theorem probability_at_least_one_girl (total_students boys girls k : ℕ) (h_total: total_students = 5) (h_boys: boys = 3) (h_girls: girls = 2) (h_k: k = 3) : 
  (1 - ((Nat.choose boys k) / (Nat.choose total_students k))) = 9 / 10 :=
by
  sorry

end probability_at_least_one_girl_l131_131065


namespace find_m_l131_131661

noncomputable def curve (x : ℝ) : ℝ := (1 / 4) * x^2
noncomputable def line (x : ℝ) : ℝ := 1 - 2 * x

theorem find_m (m n : ℝ) (h_curve : curve m = n) (h_perpendicular : (1 / 2) * m * (-2) = -1) : m = 1 := 
  sorry

end find_m_l131_131661


namespace range_of_x_range_of_a_l131_131391

-- Part (1): 
theorem range_of_x (x : ℝ) : 
  (a = 1) → (x^2 - 6 * a * x + 8 * a^2 < 0) → (x^2 - 4 * x + 3 ≤ 0) → (2 < x ∧ x ≤ 3) := sorry

-- Part (2):
theorem range_of_a (a : ℝ) : 
  (a ≠ 0) → (∀ x, (x^2 - 4 * x + 3 ≤ 0) → (x^2 - 6 * a * x + 8 * a^2 < 0)) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 4) := sorry

end range_of_x_range_of_a_l131_131391


namespace bottles_recycled_l131_131384

theorem bottles_recycled (start_bottles : ℕ) (recycle_ratio : ℕ) (answer : ℕ)
  (h_start : start_bottles = 256) (h_recycle : recycle_ratio = 4) : answer = 85 :=
sorry

end bottles_recycled_l131_131384


namespace average_annual_reduction_10_percent_l131_131354

theorem average_annual_reduction_10_percent :
  ∀ x : ℝ, (1 - x) ^ 2 = 1 - 0.19 → x = 0.1 :=
by
  intros x h
  -- Proof to be filled in
  sorry

end average_annual_reduction_10_percent_l131_131354


namespace estimate_event_probability_l131_131316

/-- 
Given the frequencies of a random event occurring during an experiment for various numbers of trials,
we estimate the probability of this event occurring through the experiment and prove that it is approximately 0.35 
when rounded to 0.01. 
-/
theorem estimate_event_probability :
  let freq := [0.300, 0.360, 0.350, 0.350, 0.352, 0.351, 0.351]
  let approx_prob := 0.35
  ∀ n : ℕ, n ∈ [20, 50, 100, 300, 500, 1000, 5000] →
  freq.get! n ≈ approx_prob := 
  sorry

end estimate_event_probability_l131_131316


namespace part1_part2_l131_131077

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * (Real.sin x) * (Real.cos x)

theorem part1 : f (Real.pi / 8) = Real.sqrt 2 + 1 := sorry

theorem part2 : (∀ x1 x2 : ℝ, f (x1 + Real.pi) = f x1) ∧ (∀ x : ℝ, f x ≥ 1 - Real.sqrt 2) := 
  sorry

-- Explanation:
-- part1 is for proving f(π/8) = √2 + 1
-- part2 handles proving the smallest positive period and the minimum value of the function.

end part1_part2_l131_131077


namespace gift_options_l131_131183

theorem gift_options (n : ℕ) (h : n = 10) : (2^n - 1) = 1023 :=
by {
  rw h,
  norm_num,
  sorry
}

end gift_options_l131_131183


namespace find_y_l131_131676

theorem find_y (x y : ℝ) : x - y = 8 ∧ x + y = 14 → y = 3 := by
  sorry

end find_y_l131_131676


namespace function_relationship_minimize_total_cost_l131_131172

noncomputable def y (a x : ℕ) : ℕ :=
6400 * x + 50 * a + 100 * a^2 / (x - 1)

theorem function_relationship (a : ℕ) (hx : 2 ≤ x) : 
  y a x = 6400 * x + 50 * a + 100 * a^2 / (x - 1) :=
by sorry

theorem minimize_total_cost (a : ℕ) (hx : 2 ≤ x) (ha : a = 56) : 
  y a x ≥ 1650 * a + 6400 ∧ (x = 8) :=
by sorry

end function_relationship_minimize_total_cost_l131_131172


namespace regular_polygon_sides_l131_131856

-- Conditions
def central_angle (θ : ℝ) := θ = 30
def sum_of_central_angles (sumθ : ℝ) := sumθ = 360

-- The proof problem
theorem regular_polygon_sides (θ sumθ : ℝ) (h₁ : central_angle θ) (h₂ : sum_of_central_angles sumθ) :
  sumθ / θ = 12 := by
  sorry

end regular_polygon_sides_l131_131856


namespace percent_students_both_correct_l131_131278

def percent_answered_both_questions (total_students first_correct second_correct neither_correct : ℕ) : ℕ :=
  let at_least_one_correct := total_students - neither_correct
  let total_individual_correct := first_correct + second_correct
  total_individual_correct - at_least_one_correct

theorem percent_students_both_correct
  (total_students : ℕ)
  (first_question_correct : ℕ)
  (second_question_correct : ℕ)
  (neither_question_correct : ℕ) 
  (h_total_students : total_students = 100)
  (h_first_correct : first_question_correct = 80)
  (h_second_correct : second_question_correct = 55)
  (h_neither_correct : neither_question_correct = 20) :
  percent_answered_both_questions total_students first_question_correct second_question_correct neither_question_correct = 55 :=
by
  rw [h_total_students, h_first_correct, h_second_correct, h_neither_correct]
  sorry


end percent_students_both_correct_l131_131278


namespace smallest_number_of_eggs_l131_131008

theorem smallest_number_of_eggs (c : ℕ) (eggs_total : ℕ) :
  eggs_total = 15 * c - 3 ∧ eggs_total > 150 → eggs_total = 162 :=
by
  sorry

end smallest_number_of_eggs_l131_131008


namespace dodecagon_diagonals_l131_131805

/--
The formula for the number of diagonals in a convex n-gon is given by (n * (n - 3)) / 2.
-/
def number_of_diagonals (n : Nat) : Nat := (n * (n - 3)) / 2

/--
A dodecagon has 12 sides.
-/
def dodecagon_sides : Nat := 12

/--
The number of diagonals in a convex dodecagon is 54.
-/
theorem dodecagon_diagonals : number_of_diagonals dodecagon_sides = 54 := by
  sorry

end dodecagon_diagonals_l131_131805


namespace find_second_number_l131_131997

theorem find_second_number 
  (x : ℕ)
  (h1 : (55 + x + 507 + 2 + 684 + 42) / 6 = 223)
  : x = 48 := 
by 
  sorry

end find_second_number_l131_131997


namespace range_of_m_for_nonempty_solution_set_l131_131419

theorem range_of_m_for_nonempty_solution_set :
  {m : ℝ | ∃ x : ℝ, m * x^2 - m * x + 1 < 0} = {m : ℝ | m < 0} ∪ {m : ℝ | m > 4} :=
by sorry

end range_of_m_for_nonempty_solution_set_l131_131419


namespace positive_correlation_not_proportional_l131_131325

/-- Two quantities x and y depend on each other, and when one increases, the other also increases.
    This general relationship is denoted as a function g such that for any x₁, x₂,
    if x₁ < x₂ then g(x₁) < g(x₂). This implies a positive correlation but not necessarily proportionality. 
    We will prove that this does not imply a proportional relationship (y = kx). -/
theorem positive_correlation_not_proportional (g : ℝ → ℝ) 
(h_increasing: ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂) :
¬ ∃ k : ℝ, ∀ x : ℝ, g x = k * x :=
sorry

end positive_correlation_not_proportional_l131_131325


namespace problem_part1_problem_part2_l131_131078

-- Statement for Part (1)
theorem problem_part1 (a b : ℝ) (h_sol : {x : ℝ | ax^2 - 3 * x + 2 > 0} = {x : ℝ | x < 1 ∨ x > 2}) :
  a = 1 ∧ b = 2 := sorry

-- Statement for Part (2)
theorem problem_part2 (x y k : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) 
  (h_eq : (1 / x) + (2 / y) = 1) (h_ineq : 2 * x + y ≥ k^2 + k + 2) :
  -3 ≤ k ∧ k ≤ 2 := sorry

end problem_part1_problem_part2_l131_131078


namespace paul_sold_11_books_l131_131986

variable (initial_books : ℕ) (books_given : ℕ) (books_left : ℕ) (books_sold : ℕ)

def number_of_books_sold (initial_books books_given books_left books_sold : ℕ) : Prop :=
  initial_books - books_given - books_left = books_sold

theorem paul_sold_11_books : number_of_books_sold 108 35 62 11 :=
by
  sorry

end paul_sold_11_books_l131_131986


namespace neither_sufficient_nor_necessary_l131_131451

theorem neither_sufficient_nor_necessary (x : ℝ) : 
  ¬(-1 < x ∧ x < 2 → |x - 2| < 1) ∧ ¬(|x - 2| < 1 → -1 < x ∧ x < 2) :=
by
  sorry

end neither_sufficient_nor_necessary_l131_131451


namespace proof_problem_l131_131497

def from_base (b : ℕ) (digits : List ℕ) : ℕ :=
digits.foldr (λ (d acc) => d + b * acc) 0

def problem : Prop :=
  let a := from_base 8 [2, 3, 4, 5] -- 2345 base 8
  let b := from_base 5 [1, 4, 0]    -- 140 base 5
  let c := from_base 4 [1, 0, 3, 2] -- 1032 base 4
  let d := from_base 8 [2, 9, 1, 0] -- 2910 base 8
  let result := (a / b + c - d : ℤ)
  result = -1502

theorem proof_problem : problem :=
by
  sorry

end proof_problem_l131_131497


namespace special_number_exists_l131_131884

theorem special_number_exists (a b c d e : ℕ) (h1 : a < b ∧ b < c ∧ c < d ∧ d < e)
    (h2 : a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e) 
    (h_num : a * 10 + b = 13 ∧ c = 4 ∧ d * 10 + e = 52) :
    (10 * a + b) * c = 10 * d + e :=
by
  sorry

end special_number_exists_l131_131884


namespace find_x_l131_131378

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 135) (h2 : x > 0) : x = 11.25 :=
by
  sorry

end find_x_l131_131378


namespace augmented_matrix_determinant_l131_131598

theorem augmented_matrix_determinant (m : ℝ) 
  (h : (1 - 2 * m) / (3 - 2) = 5) : 
  m = -2 :=
  sorry

end augmented_matrix_determinant_l131_131598


namespace exponent_of_two_gives_n_l131_131555

theorem exponent_of_two_gives_n (x: ℝ) (n: ℝ) (b: ℝ)
  (h1: n = 2 ^ x)
  (h2: n ^ b = 8)
  (h3: b = 12) : x = 3 / 12 :=
by
  sorry

end exponent_of_two_gives_n_l131_131555


namespace find_a_l131_131536

noncomputable def slope1 (a : ℝ) : ℝ := -3 / (3^a - 3)
noncomputable def slope2 : ℝ := 2

theorem find_a (a : ℝ) (h : slope1 a * slope2 = -1) : a = 2 :=
sorry

end find_a_l131_131536


namespace min_n_plus_d_l131_131285

theorem min_n_plus_d (a : ℕ → ℕ) (n d : ℕ) (h1 : a 1 = 1) (h2 : a n = 51)
  (h3 : ∀ i, a i = a 1 + (i-1) * d) : n + d = 16 :=
by
  sorry

end min_n_plus_d_l131_131285


namespace part_a_part_b_l131_131864

def triangle := Type
def point := Type

structure TriangleInCircle (ABC : triangle) where
  A : point
  B : point
  C : point
  A1 : point
  B1 : point
  C1 : point
  M : point
  r : Real
  R : Real

theorem part_a (ABC : triangle) (t : TriangleInCircle ABC) :
  ∃ MA MC MB_1, (MA * MC) / MB_1 = 2 * t.r := sorry
  
theorem part_b (ABC : triangle) (t : TriangleInCircle ABC) :
  ∃ MA_1 MC_1 MB, ( (MA_1 * MC_1) / MB) = t.R := sorry

end part_a_part_b_l131_131864


namespace express_y_in_terms_of_x_l131_131297

variable (x y p : ℝ)

-- Conditions
def condition1 := x = 1 + 3^p
def condition2 := y = 1 + 3^(-p)

-- The theorem to be proven
theorem express_y_in_terms_of_x (h1 : condition1 x p) (h2 : condition2 y p) : y = x / (x - 1) :=
sorry

end express_y_in_terms_of_x_l131_131297


namespace smallest_k_exists_l131_131335

open Nat

theorem smallest_k_exists (n m k : ℕ) (hn : n > 0) (hm : 0 < m ∧ m ≤ 5) (hk : k % 3 = 0) :
  (64^k + 32^m > 4^(16 + n^2)) ↔ k = 6 :=
by
  sorry

end smallest_k_exists_l131_131335


namespace cordelia_bleach_time_l131_131507

theorem cordelia_bleach_time (B D : ℕ) (h1 : B + D = 9) (h2 : D = 2 * B) : B = 3 :=
by
  sorry

end cordelia_bleach_time_l131_131507


namespace bill_project_days_l131_131766

theorem bill_project_days (naps: ℕ) (hours_per_nap: ℕ) (working_hours: ℕ) : 
  (naps = 6) → (hours_per_nap = 7) → (working_hours = 54) → 
  (naps * hours_per_nap + working_hours) / 24 = 4 := 
by
  intros h1 h2 h3
  sorry

end bill_project_days_l131_131766


namespace letter_puzzle_l131_131220

theorem letter_puzzle (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (h_diff : A ≠ B) :
  A^B = 10 * B + A ↔ (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_l131_131220


namespace bisection_method_termination_condition_l131_131155

theorem bisection_method_termination_condition (x1 x2 e : ℝ) (h : e > 0) :
  |x1 - x2| < e → true :=
sorry

end bisection_method_termination_condition_l131_131155


namespace part_I_part_II_l131_131403

noncomputable def f_I (x : ℝ) : ℝ := abs (3*x - 1) + abs (x + 3)

theorem part_I :
  ∀ x : ℝ, f_I x ≥ 4 ↔ x ≤ 0 ∨ x ≥ 1/2 :=
by sorry

noncomputable def f_II (x b c : ℝ) : ℝ := abs (x - b) + abs (x + c)

theorem part_II :
  ∀ b c : ℝ, b > 0 → c > 0 → b + c = 1 → 
  (∀ x : ℝ, f_II x b c ≥ 1) → (1 / b + 1 / c = 4) :=
by sorry

end part_I_part_II_l131_131403


namespace avg_gpa_8th_graders_l131_131599

theorem avg_gpa_8th_graders :
  ∀ (GPA_6th GPA_8th : ℝ),
    GPA_6th = 93 →
    (∀ GPA_7th : ℝ, GPA_7th = GPA_6th + 2 →
    (GPA_6th + GPA_7th + GPA_8th) / 3 = 93 →
    GPA_8th = 91) :=
by
  intros GPA_6th GPA_8th h1 GPA_7th h2 h3
  sorry

end avg_gpa_8th_graders_l131_131599


namespace arithmetic_square_root_of_4_l131_131716

theorem arithmetic_square_root_of_4 : ∃ y : ℝ, y^2 = 4 ∧ y = 2 := 
  sorry

end arithmetic_square_root_of_4_l131_131716


namespace luke_earning_problem_l131_131299

variable (WeedEarning Weeks SpendPerWeek MowingEarning : ℤ)

theorem luke_earning_problem
  (h1 : WeedEarning = 18)
  (h2 : Weeks = 9)
  (h3 : SpendPerWeek = 3)
  (h4 : MowingEarning + WeedEarning = Weeks * SpendPerWeek) :
  MowingEarning = 9 := by
  sorry

end luke_earning_problem_l131_131299


namespace prime_saturated_96_l131_131754

def is_prime_saturated (d : ℕ) : Prop :=
  let prime_factors := [2, 3]  -- list of the different positive prime factors of 96
  prime_factors.prod < d       -- the product of prime factors should be less than d

theorem prime_saturated_96 : is_prime_saturated 96 :=
by
  sorry

end prime_saturated_96_l131_131754


namespace monotonicity_case1_monotonicity_case2_lower_bound_fx_l131_131406

noncomputable def f (x a : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_case1 {x a : ℝ} (h : a ≤ 0) : 
  ∀ x, (differentiable ℝ (λ x, f x a) ∧ deriv (λ x, f x a) x ≤ -1) :=
sorry

theorem monotonicity_case2 {x a : ℝ} (h : 0 < a) : 
  ∀ x, (x < Real.log (1 / a) → (f x a) < (f (Real.log (1 / a)) a)) ∧ (Real.log (1 / a) < x → (f (Real.log (1 / a)) a) < f x a) :=
sorry

theorem lower_bound_fx {x a : ℝ} (h : 0 < a) : f x a > 2 * Real.log a + 3 / 2 :=
sorry

end monotonicity_case1_monotonicity_case2_lower_bound_fx_l131_131406


namespace find_a_b_find_k_range_l131_131081

-- Define the conditions for part 1
def quad_inequality (a x : ℝ) : Prop :=
  a * x^2 - 3 * x + 2 > 0

def solution_set (x b : ℝ) : Prop :=
  x < 1 ∨ x > b

theorem find_a_b (a b : ℝ) :
  (∀ x, quad_inequality a x ↔ solution_set x b) → (a = 1 ∧ b = 2) :=
sorry

-- Define the conditions for part 2
def valid_x_y (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

def equation1 (a b x y : ℝ) : Prop :=
  a / x + b / y = 1

def inequality1 (x y k : ℝ) : Prop :=
  2 * x + y ≥ k^2 + k + 2

theorem find_k_range (a b : ℝ) (x y k : ℝ) :
  a = 1 → b = 2 → valid_x_y x y → equation1 a b x y → inequality1 x y k →
  (-3 ≤ k ∧ k ≤ 2) :=
sorry

end find_a_b_find_k_range_l131_131081


namespace extreme_point_inequality_l131_131407

open Real

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*x - 1 - a * log x

theorem extreme_point_inequality (a x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) (h3 : x2 < 1) (h4 : -1/2 < a) (h5 : a < 0)
  (h6 : (2 * x1^2 - 2 * x1 - a = 0)) (h7 : (2 * x2^2 - 2 * x2 - a = 0)) :
  (f x1 a) / x2 > -7/2 - log 2 :=
sorry

end extreme_point_inequality_l131_131407


namespace john_paid_correct_amount_l131_131696

theorem john_paid_correct_amount : 
  let upfront_fee := 1000
  let hourly_rate := 100
  let court_hours := 50
  let prep_hours := 2 * court_hours
  let total_hours_fee := (court_hours + prep_hours) * hourly_rate
  let paperwork_fee := 500
  let transportation_costs := 300
  let total_fee := total_hours_fee + upfront_fee + paperwork_fee + transportation_costs
  let john_share := total_fee / 2
  john_share = 8400 :=
by
  let upfront_fee := 1000
  let hourly_rate := 100
  let court_hours := 50
  let prep_hours := 2 * court_hours
  let total_hours_fee := (court_hours + prep_hours) * hourly_rate
  let paperwork_fee := 500
  let transportation_costs := 300
  let total_fee := total_hours_fee + upfront_fee + paperwork_fee + transportation_costs
  let john_share := total_fee / 2
  show john_share = 8400
  sorry

end john_paid_correct_amount_l131_131696


namespace combined_annual_income_eq_correct_value_l131_131720

theorem combined_annual_income_eq_correct_value :
  let A_income := 5 / 2 * 17000
  let B_income := 1.12 * 17000
  let C_income := 17000
  let D_income := 0.85 * A_income
  (A_income + B_income + C_income + D_income) * 12 = 1375980 :=
by
  sorry

end combined_annual_income_eq_correct_value_l131_131720


namespace lisa_eats_correct_number_of_pieces_l131_131836

variable (M A K R L : ℚ) -- All variables are rational numbers (real numbers could also be used)
variable (n : ℕ) -- n is a natural number (the number of pieces of lasagna)

-- Let's define the conditions succinctly
def manny_wants_one_piece := M = 1
def aaron_eats_nothing := A = 0
def kai_eats_twice_manny := K = 2 * M
def raphael_eats_half_manny := R = 0.5 * M
def lasagna_is_cut_into_6_pieces := n = 6

-- The proof goal is to show Lisa eats 2.5 pieces
theorem lisa_eats_correct_number_of_pieces (M A K R L : ℚ) (n : ℕ) :
  manny_wants_one_piece M →
  aaron_eats_nothing A →
  kai_eats_twice_manny M K →
  raphael_eats_half_manny M R →
  lasagna_is_cut_into_6_pieces n →
  L = n - (M + K + R) →
  L = 2.5 :=
by
  intros hM hA hK hR hn hL
  sorry  -- Proof omitted

end lisa_eats_correct_number_of_pieces_l131_131836


namespace option_c_is_correct_l131_131490

variable (x : ℝ)

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f(x)

def is_increasing (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → f(x) < f(y)

def f := λ x : ℝ, 2^x - 2^(-x)

theorem option_c_is_correct : is_odd f ∧ is_increasing f :=
by
  sorry

end option_c_is_correct_l131_131490


namespace shaded_fraction_eighth_triangle_l131_131681

def triangular_number (n : Nat) : Nat := n * (n + 1) / 2
def square_number (n : Nat) : Nat := n * n

theorem shaded_fraction_eighth_triangle :
  let shaded_triangles := triangular_number 7
  let total_triangles := square_number 8
  shaded_triangles / total_triangles = 7 / 16 := 
by
  sorry

end shaded_fraction_eighth_triangle_l131_131681


namespace inequality_solution_l131_131919

theorem inequality_solution (x : ℝ) (h_pos : 0 < x) :
  (3 / 8 + |x - 14 / 24| < 8 / 12) ↔ x ∈ Set.Ioo (7 / 24) (7 / 8) :=
by
  sorry

end inequality_solution_l131_131919


namespace average_rainfall_february_1964_l131_131684

theorem average_rainfall_february_1964 :
  let total_rainfall := 280
  let days_february := 29
  let hours_per_day := 24
  (total_rainfall / (days_february * hours_per_day)) = (280 / (29 * 24)) :=
by
  sorry

end average_rainfall_february_1964_l131_131684


namespace mod_remainder_l131_131336

theorem mod_remainder (n : ℕ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := 
by
  sorry

end mod_remainder_l131_131336


namespace percentage_of_men_l131_131559

theorem percentage_of_men (E M W : ℝ) 
  (h1 : M + W = E)
  (h2 : 0.5 * M + 0.1666666666666669 * W = 0.4 * E)
  (h3 : W = E - M) : 
  (M / E = 0.70) :=
by
  sorry

end percentage_of_men_l131_131559


namespace problem_a5_value_l131_131798

def Sn (n : ℕ) : ℕ := 2 * n^2 + 3 * n - 1

theorem problem_a5_value : Sn 5 - Sn 4 = 21 := by
  sorry

end problem_a5_value_l131_131798


namespace solver_inequality_l131_131992

theorem solver_inequality (x : ℝ) :
  (2 * x - 1 ≥ x + 2) ∧ (x + 5 < 4 * x - 1) → (x ≥ 3) :=
by
  intro h
  sorry

end solver_inequality_l131_131992


namespace tim_weekly_earnings_l131_131610

-- Define the constants based on the conditions
def tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def days_per_week : ℕ := 6

-- Statement to prove Tim's weekly earnings
theorem tim_weekly_earnings : tasks_per_day * pay_per_task * days_per_week = 720 := by
  sorry

end tim_weekly_earnings_l131_131610


namespace tennis_tournament_l131_131823

theorem tennis_tournament (n : ℕ) (w m : ℕ) 
  (total_matches : ℕ)
  (women_wins men_wins : ℕ) :
  n + 2 * n = 3 * n →
  total_matches = (3 * n * (3 * n - 1)) / 2 →
  women_wins + men_wins = total_matches →
  women_wins / men_wins = 7 / 5 →
  n = 3 :=
by sorry

end tennis_tournament_l131_131823


namespace cups_of_flour_required_l131_131981

/-- Define the number of cups of sugar and salt required by the recipe. --/
def sugar := 14
def salt := 7
/-- Define the number of cups of flour already added. --/
def flour_added := 2
/-- Define the additional requirement of flour being 3 more cups than salt. --/
def additional_flour_requirement := 3

/-- Main theorem to prove the total amount of flour the recipe calls for. --/
theorem cups_of_flour_required : total_flour = 10 :=
by
  sorry

end cups_of_flour_required_l131_131981


namespace max_side_length_triangle_l131_131643

theorem max_side_length_triangle (a b c : ℕ) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_perimeter : a + b + c = 20) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : max a (max b c) = 9 := 
sorry

end max_side_length_triangle_l131_131643


namespace solve_quadratic_equation_l131_131062

theorem solve_quadratic_equation (x : ℝ) : 4 * (x - 1)^2 = 36 ↔ (x = 4 ∨ x = -2) :=
by sorry

end solve_quadratic_equation_l131_131062


namespace same_color_probability_l131_131636

-- Given conditions
def totalChairs : Nat := 33
def blackChairs : Nat := 15
def brownChairs : Nat := 18

-- Define the probability calculation
noncomputable def probability_same_color : Rat :=
  let prob_both_black := (blackChairs : Rat) * (blackChairs - 1) / (totalChairs * (totalChairs - 1))
  let prob_both_brown := (brownChairs : Rat) * (brownChairs - 1) / (totalChairs * (totalChairs - 1))
  prob_both_black + prob_both_brown

-- The statement to prove
theorem same_color_probability : probability_same_color = 43 / 88 :=
  sorry

end same_color_probability_l131_131636


namespace smallest_coprime_gt_one_l131_131231

theorem smallest_coprime_gt_one (x : ℕ) (h1 : 1 < x) (h2 : Nat.gcd x 180 = 1) : x = 7 := sorry

end smallest_coprime_gt_one_l131_131231


namespace prove_inequality_l131_131395

theorem prove_inequality
  (a b c d : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : c > 0)
  (h₃ : d > 0)
  (h₄ : a ≤ b)
  (h₅ : b ≤ c)
  (h₆ : c ≤ d)
  (h₇ : a + b + c + d ≥ 1) :
  a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 ≥ 1 :=
by
  sorry

end prove_inequality_l131_131395


namespace first_number_percentage_of_second_l131_131094

theorem first_number_percentage_of_second {X : ℝ} (H1 : ℝ) (H2 : ℝ) 
  (H1_def : H1 = 0.05 * X) (H2_def : H2 = 0.25 * X) : 
  (H1 / H2) * 100 = 20 :=
by
  sorry

end first_number_percentage_of_second_l131_131094


namespace stamp_distribution_correct_l131_131840

variables {W : ℕ} -- We use ℕ (natural numbers) for simplicity but this can be any type representing weight.

-- Number of envelopes that weigh less than W and need 2 stamps each
def envelopes_lt_W : ℕ := 6

-- Number of stamps per envelope if the envelope weighs less than W
def stamps_lt_W : ℕ := 2

-- Number of envelopes in total
def total_envelopes : ℕ := 14

-- Number of stamps for the envelopes that weigh less
def total_stamps_lt_W : ℕ := envelopes_lt_W * stamps_lt_W

-- Total stamps bought by Micah
def total_stamps_bought : ℕ := 52

-- Stamps left for envelopes that weigh more than W
def stamps_remaining : ℕ := total_stamps_bought - total_stamps_lt_W

-- Remaining envelopes that need stamps (those that weigh more than W)
def envelopes_gt_W : ℕ := total_envelopes - envelopes_lt_W

-- Number of stamps required per envelope that weighs more than W
def stamps_gt_W : ℕ := 5

-- Total stamps needed for the envelopes that weigh more than W
def total_stamps_needed_gt_W : ℕ := envelopes_gt_W * stamps_gt_W

theorem stamp_distribution_correct :
  total_stamps_bought = (total_stamps_lt_W + total_stamps_needed_gt_W) :=
by
  sorry

end stamp_distribution_correct_l131_131840


namespace area_rectangle_relation_l131_131596

theorem area_rectangle_relation (x y : ℝ) (h : x * y = 12) : y = 12 / x :=
by
  sorry

end area_rectangle_relation_l131_131596


namespace line_equation_through_points_and_area_l131_131752

variable (a b S : ℝ)
variable (h_b_gt_a : b > a)
variable (h_area : S = 1/2 * (b - a) * (2 * S / (b - a)))

theorem line_equation_through_points_and_area :
  0 = -2 * S * x + (b - a)^2 * y + 2 * S * a - 2 * S * b := sorry

end line_equation_through_points_and_area_l131_131752


namespace fragment_probability_l131_131736

noncomputable def probability_fragment_in_21_digit_code : ℚ :=
  (12 * 10^11 - 30) / 10^21

theorem fragment_probability:
  ∀ (code : Fin 10 → Fin 21 → Fin 10),
  (∃ (i : Fin 12), ∀ (j : Fin 10), code (i + j) = j) → 
  probability_fragment_in_21_digit_code = (12 * 10^11 - 30) / 10^21 :=
sorry

end fragment_probability_l131_131736


namespace arithmetic_sequence_sum_mul_three_eq_3480_l131_131915

theorem arithmetic_sequence_sum_mul_three_eq_3480 :
  let a := 50
  let d := 3
  let l := 95
  let n := ((l - a) / d + 1 : ℕ)
  let sum := n * (a + l) / 2
  3 * sum = 3480 := by
  sorry

end arithmetic_sequence_sum_mul_three_eq_3480_l131_131915


namespace bold_o_lit_cells_l131_131041

-- Define the conditions
def grid_size : ℕ := 5
def original_o_lit_cells : ℕ := 12 -- Number of cells lit in the original 'o'
def additional_lit_cells : ℕ := 12 -- Additional cells lit in the bold 'o'

-- Define the property to be proved
theorem bold_o_lit_cells : (original_o_lit_cells + additional_lit_cells) = 24 :=
by
  -- computation skipped
  sorry

end bold_o_lit_cells_l131_131041


namespace mean_values_are_two_l131_131442

noncomputable def verify_means (a b : ℝ) : Prop :=
  (a + b) / 2 = 2 ∧ 2 / ((1 / a) + (1 / b)) = 2

theorem mean_values_are_two (a b : ℝ) (h : verify_means a b) : a = 2 ∧ b = 2 :=
  sorry

end mean_values_are_two_l131_131442


namespace smallest_rel_prime_to_180_l131_131241

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, (y > 1 ∧ Nat.gcd y 180 = 1) → x ≤ y :=
begin
  sorry
end

end smallest_rel_prime_to_180_l131_131241


namespace find_missing_number_l131_131095

theorem find_missing_number 
  (x : ℝ) (y : ℝ)
  (h1 : (12 + x + 42 + 78 + 104) / 5 = 62)
  (h2 : (128 + 255 + y + 1023 + x) / 5 = 398.2) :
  y = 511 := 
sorry

end find_missing_number_l131_131095


namespace geom_seq_min_m_l131_131280

def initial_capital : ℝ := 50
def growth_rate : ℝ := 0.5
def annual_payment (t : ℝ) : Prop := t ≤ 2500
def capital_remaining (aₙ : ℕ → ℝ) (n : ℕ) (t : ℝ) : ℝ := aₙ n * (1 + growth_rate) - t

theorem geom_seq (aₙ : ℕ → ℝ) (t : ℝ) (h₁ : annual_payment t) :
  (∀ n, aₙ (n + 1) = 3 / 2 * aₙ n - t) →
  (t ≠ 2500) →
  ∃ r : ℝ, ∀ n, aₙ n - 2 * t = (aₙ 0 - 2 * t) * r ^ n :=
sorry

theorem min_m (t : ℝ) (h₁ : t = 1500) (aₙ : ℕ → ℝ) :
  (∀ n, aₙ (n + 1) = 3 / 2 * aₙ n - t) →
  (aₙ 0 = initial_capital * (1 + growth_rate) - t) →
  ∃ m : ℕ, aₙ m > 21000 ∧ ∀ k < m, aₙ k ≤ 21000 :=
sorry

end geom_seq_min_m_l131_131280


namespace graph_of_equation_pair_of_lines_l131_131051

theorem graph_of_equation_pair_of_lines (x y : ℝ) : x^2 - 9 * y^2 = 0 ↔ (x = 3 * y ∨ x = -3 * y) :=
by
  sorry

end graph_of_equation_pair_of_lines_l131_131051


namespace slopes_and_angles_l131_131445

theorem slopes_and_angles (m n : ℝ) (θ₁ θ₂ : ℝ)
  (h1 : θ₁ = 3 * θ₂)
  (h2 : m = 5 * n)
  (h3 : m = Real.tan θ₁)
  (h4 : n = Real.tan θ₂)
  (h5 : m ≠ 0) :
  m * n = 5 / 7 :=
by {
  sorry
}

end slopes_and_angles_l131_131445


namespace zero_function_unique_l131_131481

theorem zero_function_unique 
  (f : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), f (x ^ (42 ^ 42) + y) = f (x ^ 3 + 2 * y) + f (x ^ 12)) : 
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end zero_function_unique_l131_131481


namespace assign_teachers_l131_131198

theorem assign_teachers : 
  let teachers := {t1, t2, t3, t4, t5}
  let classes := {c1, c2, c3}
  -- Number of ways to group 5 teachers into 3 classes with at least one teacher in each class:
  let num_groupings := (choose 5 2 * choose 3 2 * choose 1 1 / 2!) + choose 5 3
  -- Number of ways to permute the 3 groups among the 3 classes:
  let num_permutations := 3!
  -- Total number of assignments:
  (num_groupings * num_permutations) = 150
:= sorry

end assign_teachers_l131_131198


namespace unknown_number_is_7_l131_131356

theorem unknown_number_is_7 (x : ℤ) (hx : x > 0)
  (h : (1 / 4 : ℚ) * (10 * x + 7 - x ^ 2) - x = 0) : x = 7 :=
  sorry

end unknown_number_is_7_l131_131356


namespace longest_boat_length_l131_131580

-- Definitions of the conditions
def total_savings : ℤ := 20000
def cost_per_foot : ℤ := 1500
def license_registration : ℤ := 500
def docking_fees := 3 * license_registration

-- Calculate the reserved amount for license, registration, and docking fees
def reserved_amount := license_registration + docking_fees

-- Calculate the amount left for the boat
def amount_left := total_savings - reserved_amount

-- Calculate the maximum length of the boat Mitch can afford
def max_boat_length := amount_left / cost_per_foot

-- Theorem to prove the longest boat Mitch can buy
theorem longest_boat_length : max_boat_length = 12 :=
by
  sorry

end longest_boat_length_l131_131580


namespace inequality_must_hold_l131_131006

theorem inequality_must_hold (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
sorry

end inequality_must_hold_l131_131006


namespace fraction_simplify_l131_131056

theorem fraction_simplify:
  (1/5 + 1/7) / (3/8 - 1/9) = 864 / 665 :=
by
  sorry

end fraction_simplify_l131_131056


namespace base_conversion_min_sum_l131_131722

theorem base_conversion_min_sum (a b : ℕ) (h : 3 * a + 5 = 5 * b + 3)
    (h_mod: 3 * a - 2 ≡ 0 [MOD 5])
    (valid_base_a : a >= 2)
    (valid_base_b : b >= 2):
  a + b = 14 := sorry

end base_conversion_min_sum_l131_131722


namespace part1_part2_l131_131085

open Classical

theorem part1 (x : ℝ) (a : ℝ) (b : ℝ) :
  (a = 1) ∧ (b = 2) ∧ (∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :=
by
  sorry

theorem part2 (x y k : ℝ) (a b : ℝ) :
  a = 1 ∧ b = 2 ∧ (x > 0) ∧ (y > 0) ∧ (1 / x + 2 / y = 1) ∧ (2 * x + y ≥ k^2 + k + 2) → -3 ≤ k ∧ k ≤ 2 :=
by
  sorry

end part1_part2_l131_131085


namespace letter_puzzle_solutions_l131_131216

theorem letter_puzzle_solutions (A B : ℕ) : 
  (1 ≤ A ∧ A < 10) ∧ (1 ≤ B ∧ B < 10) ∧ (A ≠ B) ∧ (A^B = 10 * B + A) → 
  (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l131_131216


namespace problem_solution_l131_131411

-- Define the sets and the conditions given in the problem
def setA : Set ℝ := 
  {y | ∃ (x : ℝ), (x ∈ Set.Icc (3 / 4) 2) ∧ (y = x^2 - (3 / 2) * x + 1)}

def setB (m : ℝ) : Set ℝ := 
  {x | x + m^2 ≥ 1}

-- The proof statement contains two parts
theorem problem_solution (m : ℝ) :
  -- Part (I) - Prove the set A
  setA = Set.Icc (7 / 16) 2
  ∧
  -- Part (II) - Prove the range for m
  (∀ x, x ∈ setA → x ∈ setB m) → (m ≥ 3 / 4 ∨ m ≤ -3 / 4) :=
by
  sorry

end problem_solution_l131_131411


namespace min_inv_sum_l131_131266

theorem min_inv_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : 2 * a * 1 + b * 2 = 2) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ (1/a) + (1/b) = 4 :=
sorry

end min_inv_sum_l131_131266


namespace newOp_of_M_and_N_l131_131050

def newOp (A B : Set ℕ) : Set ℕ :=
  {x | x ∈ A ∨ x ∈ B ∧ x ∉ (A ∩ B)}

theorem newOp_of_M_and_N (M N : Set ℕ) :
  M = {0, 2, 4, 6, 8, 10} →
  N = {0, 3, 6, 9, 12, 15} →
  newOp (newOp M N) M = N :=
by
  intros hM hN
  sorry

end newOp_of_M_and_N_l131_131050


namespace jovana_added_shells_l131_131570

theorem jovana_added_shells (initial_amount final_amount added_amount : ℕ) 
  (h_initial : initial_amount = 5) 
  (h_final : final_amount = 17) 
  (h_equation : final_amount = initial_amount + added_amount) : 
  added_amount = 12 := 
by 
  sorry

end jovana_added_shells_l131_131570


namespace range_of_m_plus_n_l131_131387

theorem range_of_m_plus_n (m n : ℝ)
  (tangent_condition : (∀ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 → (x - 1)^2 + (y - 1)^2 = 1)) :
  m + n ∈ (Set.Iic (2 - 2*Real.sqrt 2) ∪ Set.Ici (2 + 2*Real.sqrt 2)) :=
sorry

end range_of_m_plus_n_l131_131387


namespace minimal_odd_sum_is_1683_l131_131156

/-!
# Proof Problem:
Prove that the minimal odd sum of two three-digit numbers and one four-digit number 
formed using the digits 0 through 9 exactly once is 1683.
-/
theorem minimal_odd_sum_is_1683 :
  ∃ (a b : ℕ) (c : ℕ), 
    100 ≤ a ∧ a < 1000 ∧ 
    100 ≤ b ∧ b < 1000 ∧ 
    1000 ≤ c ∧ c < 10000 ∧ 
    a + b + c % 2 = 1 ∧ 
    (∀ d e f : ℕ, 
      100 ≤ d ∧ d < 1000 ∧ 
      100 ≤ e ∧ e < 1000 ∧ 
      1000 ≤ f ∧ f < 10000 ∧ 
      d + e + f % 2 = 1 → a + b + c ≤ d + e + f) ∧ 
    a + b + c = 1683 := 
sorry

end minimal_odd_sum_is_1683_l131_131156


namespace tim_weekly_earnings_l131_131611

-- Define the constants based on the conditions
def tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def days_per_week : ℕ := 6

-- Statement to prove Tim's weekly earnings
theorem tim_weekly_earnings : tasks_per_day * pay_per_task * days_per_week = 720 := by
  sorry

end tim_weekly_earnings_l131_131611


namespace quadrilateral_diagonals_l131_131392

-- Define the points of the quadrilateral
variables {A B C D P Q R S : ℝ × ℝ}

-- Define the midpoints condition
def is_midpoint (M : ℝ × ℝ) (X Y : ℝ × ℝ) := M = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)

-- Define the lengths squared condition
def dist_sq (X Y : ℝ × ℝ) := (X.1 - Y.1)^2 + (X.2 - Y.2)^2

-- Main theorem to prove
theorem quadrilateral_diagonals (hP : is_midpoint P A B) (hQ : is_midpoint Q B C)
  (hR : is_midpoint R C D) (hS : is_midpoint S D A) :
  dist_sq A C + dist_sq B D = 2 * (dist_sq P R + dist_sq Q S) :=
by
  sorry

end quadrilateral_diagonals_l131_131392


namespace intersection_of_A_and_B_l131_131799

-- Definitions of the sets A and B
def A : Set ℝ := { x | x^2 + 2*x - 3 < 0 }
def B : Set ℝ := { x | |x - 1| < 2 }

-- The statement to prove their intersection
theorem intersection_of_A_and_B : A ∩ B = { x | -1 < x ∧ x < 1 } :=
by 
  sorry

end intersection_of_A_and_B_l131_131799


namespace charlotte_age_l131_131366

theorem charlotte_age : 
  ∀ (B C E : ℝ), 
    (B = 4 * C) → 
    (E = C + 5) → 
    (B = E) → 
    C = 5 / 3 :=
by
  intros B C E h1 h2 h3
  /- start of the proof -/
  sorry

end charlotte_age_l131_131366


namespace pentagon_zero_impossible_l131_131877

theorem pentagon_zero_impossible
  (x : Fin 5 → ℝ)
  (h_sum : x 0 + x 1 + x 2 + x 3 + x 4 = 0)
  (operation : ∀ i : Fin 5, ∀ y : Fin 5 → ℝ,
    y i = (x i + x ((i + 1) % 5)) / 2 ∧ y ((i + 1) % 5) = (x i + x ((i + 1) % 5)) / 2) :
  ¬ ∃ (y : ℕ → (Fin 5 → ℝ)), ∃ N : ℕ, y N = 0 := 
sorry

end pentagon_zero_impossible_l131_131877


namespace part1_part2_l131_131089

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def b : ℝ × ℝ := (3, -Real.sqrt 3)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

noncomputable def f (x : ℝ) : ℝ := dot_product (a x) b

theorem part1 (hx : x ∈ Set.Icc 0 Real.pi) (h_perp : dot_product (a x) b = 0) : x = 5 * Real.pi / 6 :=
sorry

theorem part2 (hx : x ∈ Set.Icc 0 Real.pi) :
  (f x ≤ 2 * Real.sqrt 3) ∧ (f x = 2 * Real.sqrt 3 → x = 0) ∧
  (f x ≥ -2 * Real.sqrt 3) ∧ (f x = -2 * Real.sqrt 3 → x = 5 * Real.pi / 6) :=
sorry

end part1_part2_l131_131089


namespace unoccupied_volume_correct_l131_131990

-- Define the conditions given in the problem
def tank_length := 12 -- inches
def tank_width := 8 -- inches
def tank_height := 10 -- inches
def water_fraction := 1 / 3
def ice_cube_side := 1 -- inches
def num_ice_cubes := 12

-- Calculate the occupied volume
noncomputable def tank_volume : ℝ := tank_length * tank_width * tank_height
noncomputable def water_volume : ℝ := tank_volume * water_fraction
noncomputable def ice_cube_volume : ℝ := ice_cube_side^3
noncomputable def total_ice_volume : ℝ := ice_cube_volume * num_ice_cubes
noncomputable def total_occupied_volume : ℝ := water_volume + total_ice_volume

-- Calculate the unoccupied volume
noncomputable def unoccupied_volume : ℝ := tank_volume - total_occupied_volume

-- State the problem
theorem unoccupied_volume_correct : unoccupied_volume = 628 := by
  sorry

end unoccupied_volume_correct_l131_131990


namespace part1_part2_l131_131796

theorem part1 (m : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = m - |x - 2|) 
  (h₂ : ∀ x, f (x + 2) ≥ 0 ↔ x ∈ [-1, 1]) : m = 1 := 
sorry

theorem part2 (a b c : ℝ) (Z : ℝ) 
  (h₁ : ∀ x, 0 < x) 
  (h₂ : (1 / a) + (1 / (2 * b)) + (1 / (3 * c)) = 1) : 
  Z = a + 2 * b + 3 * c → Z ≥ 9 := 
sorry

end part1_part2_l131_131796


namespace white_marbles_count_l131_131346

theorem white_marbles_count (total_marbles blue_marbles red_marbles : ℕ) (probability_red_or_white : ℚ)
    (h_total : total_marbles = 60)
    (h_blue : blue_marbles = 5)
    (h_red : red_marbles = 9)
    (h_probability : probability_red_or_white = 0.9166666666666666) :
    ∃ W : ℕ, W = total_marbles - blue_marbles - red_marbles ∧ probability_red_or_white = (red_marbles + W)/(total_marbles) ∧ W = 46 :=
by
  sorry

end white_marbles_count_l131_131346


namespace triangle_coordinates_sum_l131_131462

noncomputable def coordinates_of_triangle_A (p q : ℚ) : Prop :=
  let B := (12, 19)
  let C := (23, 20)
  let area := ((B.1 * C.2 + C.1 * q + p * B.2) - (B.2 * C.1 + C.2 * p + q * B.1)) / 2 
  let M := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let median_slope := (q - M.2) / (p - M.1)
  area = 60 ∧ median_slope = 3 

theorem triangle_coordinates_sum (p q : ℚ) 
(h : coordinates_of_triangle_A p q) : p + q = 52 := 
sorry

end triangle_coordinates_sum_l131_131462


namespace coloring_ways_of_circle_l131_131825

noncomputable def num_ways_to_color_circle (n : ℕ) (k : ℕ) : ℕ :=
  if h : n % 2 = 1 then -- There are 13 parts; n must be odd (since adjacent matching impossible in even n)
    (k * (k - 1)^(n - 1) : ℕ)
  else
    0

theorem coloring_ways_of_circle :
  num_ways_to_color_circle 13 3 = 6 :=
by
  sorry

end coloring_ways_of_circle_l131_131825


namespace depth_of_water_is_60_l131_131194

def dean_height : ℕ := 6
def depth_multiplier : ℕ := 10
def water_depth : ℕ := depth_multiplier * dean_height

theorem depth_of_water_is_60 : water_depth = 60 := by
  -- mathematical equivalent proof problem
  sorry

end depth_of_water_is_60_l131_131194


namespace investment_a_l131_131627

/-- Given:
  * b's profit share is Rs. 1800,
  * the difference between a's and c's profit shares is Rs. 720,
  * b invested Rs. 10000,
  * c invested Rs. 12000,
  prove that a invested Rs. 16000. -/
theorem investment_a (P_b : ℝ) (P_a : ℝ) (P_c : ℝ) (B : ℝ) (C : ℝ) (A : ℝ)
  (h1 : P_b = 1800)
  (h2 : P_a - P_c = 720)
  (h3 : B = 10000)
  (h4 : C = 12000)
  (h5 : P_b / B = P_c / C)
  (h6 : P_a / A = P_b / B) : A = 16000 :=
sorry

end investment_a_l131_131627


namespace count_perfect_squares_multiple_of_36_l131_131543

theorem count_perfect_squares_multiple_of_36 (N : ℕ) (M : ℕ := 36) :
  (∃ n, n ^ 2 < 10^8 ∧ (M ∣ n ^ 2) ∧ (1 ≤ n) ∧ (n < 10^4)) →
  (M = 36 ∧ ∃ C, ∑ k in finset.range (N + 1), if (M * k < 10^4) then 1 else 0 = C ∧ C = 277) :=
begin
  sorry
end

end count_perfect_squares_multiple_of_36_l131_131543


namespace distance_between_bakery_and_butcher_shop_l131_131910

variables (v1 v2 : ℝ) -- speeds of the butcher's and baker's son respectively
variables (x : ℝ) -- distance covered by the baker's son by the time they meet
variable (distance : ℝ) -- distance between the bakery and the butcher shop

-- Given conditions
def butcher_walks_500_more := x + 0.5
def butcher_time_left := 10 / 60
def baker_time_left := 22.5 / 60

-- Equivalent relationships
def v1_def := v1 = 6 * x
def v2_def := v2 = (8/3) * (x + 0.5)

-- Final proof problem
theorem distance_between_bakery_and_butcher_shop :
  (x + 0.5 + x) = 2.5 :=
sorry

end distance_between_bakery_and_butcher_shop_l131_131910


namespace mrs_generous_jelly_beans_l131_131126

-- Define necessary terms and state the problem
def total_children (x : ℤ) : ℤ := x + (x + 3)

theorem mrs_generous_jelly_beans :
  ∃ x : ℤ, x^2 + (x + 3)^2 = 490 ∧ total_children x = 31 :=
by {
  sorry
}

end mrs_generous_jelly_beans_l131_131126


namespace fraction_ordering_l131_131738

theorem fraction_ordering 
  (a : ℚ) (b : ℚ) (c : ℚ) 
  (h1 : a = 6 / 29) 
  (h2 : b = 10 / 31) 
  (h3 : c = 8 / 25) : 
  a < c ∧ c < b := 
by 
  sorry

end fraction_ordering_l131_131738


namespace bonus_implies_completion_l131_131560

variable (John : Type)
variable (completes_all_tasks_perfectly : John → Prop)
variable (receives_bonus : John → Prop)

theorem bonus_implies_completion :
  (∀ e : John, completes_all_tasks_perfectly e → receives_bonus e) →
  (∀ e : John, receives_bonus e → completes_all_tasks_perfectly e) :=
by
  intros h e
  sorry

end bonus_implies_completion_l131_131560


namespace positive_perfect_squares_multiples_of_36_lt_10_pow_8_l131_131552

theorem positive_perfect_squares_multiples_of_36_lt_10_pow_8 :
  ∃ (count : ℕ), count = 1666 ∧ 
    (∀ n : ℕ, (n * n < 10^8 → n % 6 = 0) ↔ (n < 10^4 ∧ n % 6 = 0)) :=
sorry

end positive_perfect_squares_multiples_of_36_lt_10_pow_8_l131_131552


namespace nth_term_206_l131_131959

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 0 = 10 ∧ a 1 = -10 ∧ ∀ n, a (n + 2) = -a n

theorem nth_term_206 (a : ℕ → ℝ) (h : geometric_sequence a) : a 205 = -10 :=
by
  -- Utilizing the sequence property to determine the 206th term
  sorry

end nth_term_206_l131_131959


namespace jordan_field_area_l131_131291

theorem jordan_field_area
  (s l : ℕ)
  (h1 : 2 * (s + l) = 24)
  (h2 : l + 1 = 2 * (s + 1)) :
  3 * s * 3 * l = 189 := 
by
  sorry

end jordan_field_area_l131_131291


namespace ratio_b_to_c_l131_131474

variable (a b c k : ℕ)

-- Conditions
def condition1 : Prop := a = b + 2
def condition2 : Prop := b = k * c
def condition3 : Prop := a + b + c = 32
def condition4 : Prop := b = 12

-- Question: Prove that ratio of b to c is 2:1
theorem ratio_b_to_c
  (h1 : condition1 a b)
  (h2 : condition2 b k c)
  (h3 : condition3 a b c)
  (h4 : condition4 b) :
  b = 2 * c := 
sorry

end ratio_b_to_c_l131_131474


namespace unique_nonneg_sequence_l131_131712

theorem unique_nonneg_sequence (a : List ℝ) (h_sum : 0 < a.sum) :
  ∃ b : List ℝ, (∀ x ∈ b, 0 ≤ x) ∧ 
                (∃ f : List ℝ → List ℝ, (f a = b) ∧ (∀ x y z, f (x :: y :: z :: tl) = (x + y) :: (-y) :: (z + y) :: tl)) :=
sorry

end unique_nonneg_sequence_l131_131712


namespace additional_people_needed_to_mow_lawn_l131_131064

theorem additional_people_needed_to_mow_lawn :
  (∀ (k : ℕ), (∀ (n t : ℕ), n * t = k) → (4 * 6 = k) → (∃ (n : ℕ), n * 3 = k) → (8 - 4 = 4)) :=
by sorry

end additional_people_needed_to_mow_lawn_l131_131064


namespace arithmetic_sequence_term_2011_l131_131746

theorem arithmetic_sequence_term_2011 :
  ∃ (n : ℕ), 1 + (n - 1) * 3 = 2011 ∧ n = 671 :=
by
  existsi 671
  split
  ·  sorry
  ·  refl

end arithmetic_sequence_term_2011_l131_131746


namespace max_value_of_3x_plus_4y_on_curve_C_l131_131410

theorem max_value_of_3x_plus_4y_on_curve_C :
  ∀ (x y : ℝ),
  (∃ (ρ θ : ℝ), ρ^2 = 36 / (4 * (Real.cos θ)^2 + 9 * (Real.sin θ)^2) ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (P : ℝ × ℝ) →
  (P = (x, y)) →
  3 * x + 4 * y ≤ Real.sqrt 145 ∧ ∃ (α : ℝ), 0 ≤ α ∧ α < 2 * Real.pi ∧ 3 * x + 4 * y = Real.sqrt 145 := 
by
  intros x y h_exists P hP
  sorry

end max_value_of_3x_plus_4y_on_curve_C_l131_131410


namespace problem1_problem2_l131_131938

namespace ArithmeticSequence

-- Part (1)
theorem problem1 (a1 : ℚ) (d : ℚ) (S_n : ℚ) (n : ℕ) (a_n : ℚ) 
  (h1 : a1 = 5 / 6) 
  (h2 : d = -1 / 6) 
  (h3 : S_n = -5) 
  (h4 : S_n = n * (2 * a1 + (n - 1) * d) / 2) 
  (h5 : a_n = a1 + (n - 1) * d) : 
  (n = 15) ∧ (a_n = -3 / 2) :=
sorry

-- Part (2)
theorem problem2 (d : ℚ) (n : ℕ) (a_n : ℚ) (a1 : ℚ) (S_n : ℚ)
  (h1 : d = 2) 
  (h2 : n = 15) 
  (h3 : a_n = -10) 
  (h4 : a_n = a1 + (n - 1) * d) 
  (h5 : S_n = n * (2 * a1 + (n - 1) * d) / 2) : 
  (a1 = -38) ∧ (S_n = -360) :=
sorry

end ArithmeticSequence

end problem1_problem2_l131_131938


namespace share_of_y_l131_131758

-- Define the conditions as hypotheses
variables (n : ℝ) (x y z : ℝ)

-- The main theorem we need to prove
theorem share_of_y (h1 : x = n) 
                   (h2 : y = 0.45 * n) 
                   (h3 : z = 0.50 * n) 
                   (h4 : x + y + z = 78) : 
  y = 18 :=
by 
  -- insert proof here (not required as per instructions)
  sorry

end share_of_y_l131_131758


namespace scheduled_conference_games_l131_131311

-- Definitions based on conditions
def num_divisions := 3
def teams_per_division := 4
def games_within_division := 3
def games_across_divisions := 2

-- Proof statement
theorem scheduled_conference_games :
  let teams_in_division := teams_per_division
  let div_game_count := games_within_division * (teams_in_division * (teams_in_division - 1) / 2) 
  let total_within_division := div_game_count * num_divisions
  let cross_div_game_count := (teams_in_division * games_across_divisions * (num_divisions - 1) * teams_in_division * num_divisions) / 2
  total_within_division + cross_div_game_count = 102 := 
by {
  sorry
}

end scheduled_conference_games_l131_131311


namespace greatest_possible_value_of_x_l131_131332

-- Define the function based on the given equation
noncomputable def f (x : ℝ) : ℝ := (4 * x - 16) / (3 * x - 4)

-- Statement to be proved
theorem greatest_possible_value_of_x : 
  (∀ x : ℝ, (f x)^2 + (f x) = 20) → 
  ∃ x : ℝ, (f x)^2 + (f x) = 20 ∧ x = 36 / 19 :=
by
  sorry

end greatest_possible_value_of_x_l131_131332


namespace minimum_value_of_expression_l131_131519

noncomputable def min_value_expression (x y : ℝ) : ℝ :=
  x^2 + x * y + y^2 + 7

theorem minimum_value_of_expression :
  ∃ x y : ℝ, min_value_expression x y = 7 :=
by
  use 0, 0
  sorry

end minimum_value_of_expression_l131_131519


namespace sufficient_but_not_necessary_condition_converse_not_true_cond_x_gt_2_iff_sufficient_not_necessary_l131_131136

theorem sufficient_but_not_necessary_condition 
  (x : ℝ) : (x + 1) * (x - 2) > 0 → x > 2 :=
by sorry

theorem converse_not_true 
  (x : ℝ) : x > 2 → (x + 1) * (x - 2) > 0 :=
by sorry

theorem cond_x_gt_2_iff_sufficient_not_necessary 
  (x : ℝ) : (x > 2 → (x + 1) * (x - 2) > 0) ∧ 
            ((x + 1) * (x - 2) > 0 → x > 2) :=
by sorry

end sufficient_but_not_necessary_condition_converse_not_true_cond_x_gt_2_iff_sufficient_not_necessary_l131_131136


namespace no_primes_sum_to_53_l131_131824

open Nat

def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem no_primes_sum_to_53 :
  ¬ ∃ (p q : Nat), p + q = 53 ∧ isPrime p ∧ isPrime q ∧ (p < 30 ∨ q < 30) :=
by
  sorry

end no_primes_sum_to_53_l131_131824


namespace relationship_between_coefficients_l131_131001

theorem relationship_between_coefficients
  (b c : ℝ)
  (h_discriminant : b^2 - 4 * c ≥ 0)
  (h_root_condition : ∃ x1 x2 : ℝ, x1^2 = -x2 ∧ x1 + x2 = -b ∧ x1 * x2 = c):
  b^3 - 3 * b * c - c^2 - c = 0 :=
by
  sorry

end relationship_between_coefficients_l131_131001


namespace smallest_angle_satisfying_trig_eqn_l131_131917

theorem smallest_angle_satisfying_trig_eqn :
  ∃ x : ℝ, 0 < x ∧ 8 * (Real.sin x)^2 * (Real.cos x)^4 - 8 * (Real.sin x)^4 * (Real.cos x)^2 = 1 ∧ x = 10 :=
by
  sorry

end smallest_angle_satisfying_trig_eqn_l131_131917


namespace wall_height_correct_l131_131538

noncomputable def brick_volume : ℝ := 25 * 11.25 * 6

noncomputable def wall_total_volume (num_bricks : ℕ) (brick_vol : ℝ) : ℝ := num_bricks * brick_vol

noncomputable def wall_height (total_volume : ℝ) (length : ℝ) (thickness : ℝ) : ℝ :=
  total_volume / (length * thickness)

theorem wall_height_correct :
  wall_height (wall_total_volume 7200 brick_volume) 900 22.5 = 600 := by
  sorry

end wall_height_correct_l131_131538


namespace find_d_values_l131_131379

open Set

theorem find_d_values :
  ∀ {f : ℝ → ℝ}, ContinuousOn f (Icc 0 1) → (f 0 = f 1) →
  ∃ (d : ℝ), d ∈ Ioo 0 1 ∧ (∀ x₀, x₀ ∈ Icc 0 (1 - d) → (f x₀ = f (x₀ + d))) ↔
  ∃ k : ℕ, d = 1 / k :=
by
  sorry

end find_d_values_l131_131379


namespace bob_expected_difference_l131_131495

-- Required definitions and conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

def probability_of_event_s : ℚ := 4 / 7
def probability_of_event_u : ℚ := 2 / 7
def probability_of_event_s_and_u : ℚ := 1 / 7
def number_of_days : ℕ := 365

noncomputable def expected_days_sweetened : ℚ :=
   (probability_of_event_s - (1 / 2) * probability_of_event_s_and_u) * number_of_days

noncomputable def expected_days_unsweetened : ℚ :=
   (probability_of_event_u - (1 / 2) * probability_of_event_s_and_u) * number_of_days

noncomputable def expected_difference : ℚ :=
   expected_days_sweetened - expected_days_unsweetened

theorem bob_expected_difference : expected_difference = 135.45 := sorry

end bob_expected_difference_l131_131495


namespace intersection_of_sets_l131_131268

-- Define the sets A and B
def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | 3 * x - 2 ≥ 1}

-- Prove that A ∩ B = {x | 1 ≤ x ∧ x ≤ 2}
theorem intersection_of_sets : A ∩ B = {x | 1 ≤ x ∧ x ≤ 2} :=
by sorry

end intersection_of_sets_l131_131268


namespace big_al_ate_40_bananas_on_june_7_l131_131911

-- Given conditions
def bananas_eaten_on_day (initial_bananas : ℕ) (day : ℕ) : ℕ :=
  initial_bananas + 4 * (day - 1)

def total_bananas_eaten (initial_bananas : ℕ) : ℕ :=
  bananas_eaten_on_day initial_bananas 1 +
  bananas_eaten_on_day initial_bananas 2 +
  bananas_eaten_on_day initial_bananas 3 +
  bananas_eaten_on_day initial_bananas 4 +
  bananas_eaten_on_day initial_bananas 5 +
  bananas_eaten_on_day initial_bananas 6 +
  bananas_eaten_on_day initial_bananas 7

noncomputable def final_bananas_on_june_7 (initial_bananas : ℕ) : ℕ :=
  bananas_eaten_on_day initial_bananas 7

-- Theorem to be proved
theorem big_al_ate_40_bananas_on_june_7 :
  ∃ initial_bananas, total_bananas_eaten initial_bananas = 196 ∧ final_bananas_on_june_7 initial_bananas = 40 :=
sorry

end big_al_ate_40_bananas_on_june_7_l131_131911


namespace find_missing_surface_area_l131_131489

noncomputable def total_surface_area (areas : List ℕ) : ℕ :=
  areas.sum

def known_areas : List ℕ := [148, 46, 72, 28, 88, 126, 58]

def missing_surface_area : ℕ := 22

theorem find_missing_surface_area (areas : List ℕ) (total : ℕ) (missing : ℕ) :
  total_surface_area areas + missing = total →
  missing = 22 :=
by
  sorry

end find_missing_surface_area_l131_131489


namespace sum_of_digits_is_3_l131_131709

-- We introduce variables for the digits a and b, and the number
variables (a b : ℕ)

-- Conditions: a and b must be digits, and the number must satisfy the given equation
-- One half of (10a + b) exceeds its one fourth by 3
def valid_digits (a b : ℕ) : Prop := a < 10 ∧ b < 10
def equation_condition (a b : ℕ) : Prop := 2 * (10 * a + b) = (10 * a + b) + 12

-- The number is two digits number
def two_digits_number (a b : ℕ) : ℕ := 10 * a + b

-- Final statement combining all conditions and proving the desired sum of digits
theorem sum_of_digits_is_3 : 
  ∀ (a b : ℕ), valid_digits a b → equation_condition a b → a + b = 3 := 
by
  intros a b h1 h2
  sorry

end sum_of_digits_is_3_l131_131709


namespace abs_ineq_cond_l131_131718

theorem abs_ineq_cond (a : ℝ) : 
  (-3 < a ∧ a < 1) ↔ (∃ x : ℝ, |x - a| + |x + 1| < 2) := sorry

end abs_ineq_cond_l131_131718


namespace quadratic_symmetry_l131_131313

def quadratic (b c : ℝ) (x : ℝ) : ℝ :=
  x^2 + b * x + c

theorem quadratic_symmetry (b c : ℝ) :
  let f := quadratic b c
  (f 2) < (f 1) ∧ (f 1) < (f 4) :=
by
  sorry

end quadratic_symmetry_l131_131313


namespace unique_solution_3x_4y_5z_l131_131926

theorem unique_solution_3x_4y_5z (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3 ^ x + 4 ^ y = 5 ^ z → x = 2 ∧ y = 2 ∧ z = 2 :=
by
  intro h
  sorry

end unique_solution_3x_4y_5z_l131_131926


namespace sum_arithmetic_series_l131_131045

theorem sum_arithmetic_series : 
    let a₁ := 1
    let d := 2
    let n := 9
    let a_n := a₁ + (n - 1) * d
    let S_n := n * (a₁ + a_n) / 2
    a_n = 17 → S_n = 81 :=
by intros
   sorry

end sum_arithmetic_series_l131_131045


namespace probability_B_and_C_exactly_two_out_of_A_B_C_l131_131558

variables (A B C : Prop)
noncomputable def P : Prop → ℚ := sorry

axiom hA : P A = 3 / 4
axiom hAC : P (¬ A ∧ ¬ C) = 1 / 12
axiom hBC : P (B ∧ C) = 1 / 4

theorem probability_B_and_C : P B = 3 / 8 ∧ P C = 2 / 3 :=
sorry

theorem exactly_two_out_of_A_B_C : 
  P (A ∧ B ∧ ¬ C) + P (A ∧ ¬ B ∧ C) + P (¬ A ∧ B ∧ C) = 15 / 32 :=
sorry

end probability_B_and_C_exactly_two_out_of_A_B_C_l131_131558


namespace john_paid_l131_131695

def upfront : ℤ := 1000
def hourly_rate : ℤ := 100
def court_hours : ℤ := 50
def prep_multiplier : ℤ := 2
def brother_share : ℚ := 1/2
def paperwork_fee : ℤ := 500
def transport_costs : ℤ := 300

theorem john_paid :
  let court_cost := court_hours * hourly_rate,
      prep_hours := prep_multiplier * court_hours,
      prep_cost := prep_hours * hourly_rate,
      total_cost := upfront + court_cost + prep_cost + paperwork_fee + transport_costs,
      john_share := total_cost * brother_share
  in john_share = 8400 :=
by {
  sorry
}

end john_paid_l131_131695


namespace algebra_problem_l131_131277

theorem algebra_problem 
  (x : ℝ) 
  (h : x^2 - 2 * x = 3) : 
  2 * x^2 - 4 * x + 3 = 9 := 
by 
  sorry

end algebra_problem_l131_131277


namespace minimum_pawns_remaining_l131_131707

-- Define the initial placement and movement conditions
structure Chessboard :=
  (white_pawns : ℕ)
  (black_pawns : ℕ)
  (on_board : ℕ)

def valid_placement (cb : Chessboard) : Prop :=
  cb.white_pawns = 32 ∧ cb.black_pawns = 32 ∧ cb.on_board = 64

def can_capture (player_pawn : ℕ → ℕ → Prop) := 
  ∀ (wp bp : ℕ), 
  wp ≥ 0 ∧ bp ≥ 0 ∧ wp + bp = 64 →
  ∀ (p_wp p_bp : ℕ), 
  player_pawn wp p_wp ∧ player_pawn bp p_bp →
  p_wp + p_bp ≥ 2
  
-- Our theorem to prove
theorem minimum_pawns_remaining (cb : Chessboard) (player_pawn : ℕ → ℕ → Prop) :
  valid_placement cb →
  can_capture player_pawn →
  ∃ min_pawns : ℕ, min_pawns = 2 :=
by
  sorry

end minimum_pawns_remaining_l131_131707


namespace evaluate_expression_equals_three_plus_sqrt_three_l131_131770

noncomputable def tan_sixty_squared_plus_one := Real.tan (60 * Real.pi / 180) ^ 2 + 1
noncomputable def tan_fortyfive_minus_twocos_thirty := Real.tan (45 * Real.pi / 180) - 2 * Real.cos (30 * Real.pi / 180)
noncomputable def expression (x y : ℝ) : ℝ := (x - (2 * x * y - y ^ 2) / x) / ((x ^ 2 - y ^ 2) / (x ^ 2 + x * y))

theorem evaluate_expression_equals_three_plus_sqrt_three :
  expression tan_sixty_squared_plus_one tan_fortyfive_minus_twocos_thirty = 3 + Real.sqrt 3 :=
sorry

end evaluate_expression_equals_three_plus_sqrt_three_l131_131770


namespace letter_puzzle_solutions_l131_131223

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def to_number (B A : ℕ) : ℕ :=
  10 * B + A

theorem letter_puzzle_solutions (A B : ℕ) (h_diff : A ≠ B) (h_digits : 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9)
  (h_two_digit : is_two_digit (to_number B A)) :
  A^B = to_number B A ↔ (A = 2 ∧ B = 5 ∨ A = 6 ∧ B = 2 ∨ A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l131_131223


namespace determine_days_l131_131326

-- Define the problem
def team_repair_time (x y : ℕ) : Prop :=
  ((1 / (x:ℝ)) + (1 / (y:ℝ)) = 1 / 18) ∧ 
  ((2 / 3 * x + 1 / 3 * y = 40))

theorem determine_days : ∃ x y : ℕ, team_repair_time x y :=
by
    use 45
    use 30
    have h1: (1/(45:ℝ) + 1/(30:ℝ)) = 1/18 := by
        sorry
    have h2: (2/3*45 + 1/3*30 = 40) := by
        sorry 
    exact ⟨h1, h2⟩

end determine_days_l131_131326


namespace cordelia_bleach_time_l131_131505

theorem cordelia_bleach_time (B D : ℕ) (h1 : B + D = 9) (h2 : D = 2 * B) : B = 3 :=
by
  sorry

end cordelia_bleach_time_l131_131505


namespace syrup_cost_per_week_l131_131897

theorem syrup_cost_per_week (gallons_per_week : ℕ) (gallons_per_box : ℕ) (cost_per_box : ℕ) 
  (h1 : gallons_per_week = 180) 
  (h2 : gallons_per_box = 30) 
  (h3 : cost_per_box = 40) : 
  (gallons_per_week / gallons_per_box) * cost_per_box = 240 := 
by
  sorry

end syrup_cost_per_week_l131_131897


namespace vector_addition_l131_131269

def a : ℝ × ℝ := (5, -3)
def b : ℝ × ℝ := (-6, 4)

theorem vector_addition : a + b = (-1, 1) := by
  rw [a, b]
  sorry

end vector_addition_l131_131269


namespace B_starts_6_hours_after_A_l131_131904

theorem B_starts_6_hours_after_A 
    (A_walk_speed : ℝ) (B_cycle_speed : ℝ) (catch_up_distance : ℝ)
    (hA : A_walk_speed = 10) (hB : B_cycle_speed = 20) (hD : catch_up_distance = 120) :
    ∃ t : ℝ, t = 6 :=
by
  sorry

end B_starts_6_hours_after_A_l131_131904


namespace cross_platform_time_l131_131317

-- Definitions of the conditions
def length_train : ℝ := 750
def length_platform := length_train
def speed_train_kmh : ℝ := 90
def speed_train_ms := speed_train_kmh * (1000 / 3600)

-- The proof problem: Prove that the time to cross the platform is 60 seconds
theorem cross_platform_time : 
  (length_train + length_platform) / speed_train_ms = 60 := 
by
  sorry

end cross_platform_time_l131_131317


namespace smallest_rel_prime_to_180_l131_131252

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  { intro y,
    intro h,
    cases h with h1 h2,
    repeat { try { apply dec_trivial,
                   apply lt_or_eq_of_le,
                   norm_num,
                   apply Nat.prime_not_dvd_mul,
                   norm_num,
                   apply not_or_distrib.mpr,
                   split,
                   norm_cast,
                   intro,
                   exact le_antisymm _ },
           sorry }
end

end smallest_rel_prime_to_180_l131_131252


namespace smallest_number_of_eggs_l131_131009

theorem smallest_number_of_eggs (c : ℕ) (eggs_total : ℕ) :
  eggs_total = 15 * c - 3 ∧ eggs_total > 150 → eggs_total = 162 :=
by
  sorry

end smallest_number_of_eggs_l131_131009


namespace lowest_point_on_graph_l131_131058

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2 * x + 2) / (x + 1)

theorem lowest_point_on_graph : ∃ (x y : ℝ), x = 0 ∧ y = 2 ∧ ∀ z > -1, f z ≥ y ∧ f x = y := by
  sorry

end lowest_point_on_graph_l131_131058


namespace bricks_in_chimney_l131_131496

-- Define the conditions
def brenda_rate (h : ℕ) : ℚ := h / 8
def brandon_rate (h : ℕ) : ℚ := h / 12
def combined_rate (h : ℕ) : ℚ := (brenda_rate h + brandon_rate h) - 15
def total_bricks_in_6_hours (h : ℕ) : ℚ := 6 * combined_rate h

-- The proof statement
theorem bricks_in_chimney : ∃ h : ℕ, total_bricks_in_6_hours h = h ∧ h = 360 :=
by
  -- Proof goes here
  sorry

end bricks_in_chimney_l131_131496


namespace coordinates_of_a_l131_131793

theorem coordinates_of_a
  (a : ℝ × ℝ)
  (b : ℝ × ℝ := (1, 2))
  (h1 : (a.1)^2 + (a.2)^2 = 5)
  (h2 : ∃ k : ℝ, a = (k, 2 * k))
  : a = (1, 2) ∨ a = (-1, -2) :=
  sorry

end coordinates_of_a_l131_131793


namespace fraction_of_loss_is_correct_l131_131639

-- Definitions based on the conditions
def selling_price : ℕ := 18
def cost_price : ℕ := 19

-- Calculating the loss
def loss : ℕ := cost_price - selling_price

-- Fraction of the loss compared to the cost price
def fraction_of_loss : ℚ := loss / cost_price

-- The theorem we want to prove
theorem fraction_of_loss_is_correct : fraction_of_loss = 1 / 19 := by
  sorry

end fraction_of_loss_is_correct_l131_131639


namespace arithmetic_sequence_common_difference_l131_131955

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (h1 : a 1 = 1) (h3 : a 3 = 4) :
  ∃ d : ℤ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ d = 3 / 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l131_131955


namespace sum_of_angles_around_point_l131_131287

theorem sum_of_angles_around_point (x : ℝ) (h : 6 * x + 3 * x + 4 * x + x + 2 * x = 360) : x = 22.5 :=
by
  sorry

end sum_of_angles_around_point_l131_131287


namespace smallest_coprime_gt_one_l131_131232

theorem smallest_coprime_gt_one (x : ℕ) (h1 : 1 < x) (h2 : Nat.gcd x 180 = 1) : x = 7 := sorry

end smallest_coprime_gt_one_l131_131232


namespace smallest_rel_prime_to_180_is_7_l131_131244

theorem smallest_rel_prime_to_180_is_7 :
  ∃ x : ℕ, x > 1 ∧ gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial, },
  split,
  { exact dec_trivial, },
  intros y hy,
  cases hy with hy1 hy2,
  exact dec_trivial,
end

end smallest_rel_prime_to_180_is_7_l131_131244


namespace part_I_part_II_l131_131940

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x + 5|
def g (x : ℝ) : ℝ := |x - 1| - |2 * x|

-- Part I
theorem part_I : ∀ x : ℝ, g x > -4 → -5 < x ∧ x < -3 :=
by
  sorry

-- Part II
theorem part_II : 
  (∃ x1 x2 : ℝ, f x1 a = g x2) → -6 ≤ a ∧ a ≤ -4 :=
by
  sorry

end part_I_part_II_l131_131940


namespace correct_option_l131_131161

theorem correct_option :
  (3 * a^2 - a^2 = 2 * a^2) ∧
  (¬ (a^2 * a^3 = a^6)) ∧
  (¬ ((3 * a)^2 = 6 * a^2)) ∧
  (¬ (a^6 / a^3 = a^2)) :=
by
  -- We only need to state the theorem; the proof details are omitted per the instructions.
  sorry

end correct_option_l131_131161


namespace animath_interns_pigeonhole_l131_131591

theorem animath_interns_pigeonhole (n : ℕ) (knows : Fin n → Finset (Fin n)) :
  ∃ (i j : Fin n), i ≠ j ∧ (knows i).card = (knows j).card :=
by
  sorry

end animath_interns_pigeonhole_l131_131591


namespace prob_of_multiples_l131_131098

-- Define the set of numbers
def S : Set ℕ := {4, 10, 20, 25, 40, 50, 100}

-- Define the predicate for a number being a multiple of 200
def is_multiple_of_200 (n : ℕ) : Prop := 200 ∣ n

-- Define the probability of two distinct members of the set having a product that is a multiple of 200
def prob_multiple_of_200 (s : Set ℕ) := let size := (s.to_finset.card.choose 2 : ℚ) in
  (s.to_finset.powerset.card.filter (fun t => t.card = 2 ∧ is_multiple_of_200 (t.to_finset.prod)) : ℚ) / size

-- Finally, the statement that needs to be proved
theorem prob_of_multiples :
  prob_multiple_of_200 S = 8 / 21 := 
sorry

end prob_of_multiples_l131_131098


namespace correct_options_A_and_D_l131_131340

noncomputable def problem_statement :=
  ∃ A B C D : Prop,
  (A = (∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0)) ∧ 
  (B = ∀ (a b c d : ℝ), a > b → c > d → ¬(a * c > b * d)) ∧
  (C = ∀ m : ℝ, ¬((∀ x : ℝ, x > 0 → (m^2 - m - 1) * x^(m^2 - 2 * m - 3) < 0) → (-1 < m ∧ m < 2))) ∧
  (D = ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ x₁ + x₂ = 3 - a ∧ x₁ * x₂ = a) → a < 0)

-- We need to prove that only A and D are true
theorem correct_options_A_and_D : problem_statement :=
  sorry

end correct_options_A_and_D_l131_131340


namespace displacement_representation_l131_131807

def represents_north (d : ℝ) : Prop := d > 0

theorem displacement_representation (d : ℝ) (h : represents_north 80) : represents_north d ↔ d > 0 :=
by trivial

example (h : represents_north 80) : 
  ∀ d, d = -50 → ¬ represents_north d ∧ abs d = 50 → ∃ s, s = "south" :=
sorry

end displacement_representation_l131_131807


namespace sqrt_eight_simplify_l131_131850

theorem sqrt_eight_simplify : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end sqrt_eight_simplify_l131_131850


namespace smallest_rel_prime_to_180_l131_131236

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Int.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Int.gcd y 180 = 1 → x ≤ y := 
sorry

end smallest_rel_prime_to_180_l131_131236


namespace isosceles_triangle_relation_range_l131_131871

-- Definitions of the problem conditions and goal
variables (x y : ℝ)

-- Given conditions
def isosceles_triangle (x y : ℝ) :=
  x + x + y = 10

-- Prove the relationship and range 
theorem isosceles_triangle_relation_range (h : isosceles_triangle x y) :
  y = 10 - 2 * x ∧ (5 / 2 < x ∧ x < 5) :=
  sorry

end isosceles_triangle_relation_range_l131_131871


namespace expand_polynomial_l131_131055

noncomputable def polynomial_expression (x : ℝ) : ℝ := -2 * (x - 3) * (x + 4) * (2 * x - 1)

theorem expand_polynomial (x : ℝ) :
  polynomial_expression x = -4 * x^3 - 2 * x^2 + 50 * x - 24 :=
sorry

end expand_polynomial_l131_131055


namespace max_x_plus_2y_l131_131834

theorem max_x_plus_2y (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 3 * x + 5 * y ≤ 15) : 
  x + 2 * y ≤ 6 :=
sorry

end max_x_plus_2y_l131_131834


namespace find_x_l131_131018

theorem find_x (x : ℝ) (h : 5.76 = 0.12 * 0.40 * x) : x = 120 := 
sorry

end find_x_l131_131018


namespace find_a_l131_131575

theorem find_a
  (x1 x2 a : ℝ)
  (h1 : x1^2 + 4 * x1 - 3 = 0)
  (h2 : x2^2 + 4 * x2 - 3 = 0)
  (h3 : 2 * x1 * (x2^2 + 3 * x2 - 3) + a = 2) :
  a = -4 :=
sorry

end find_a_l131_131575


namespace not_hyperbola_condition_l131_131903

theorem not_hyperbola_condition (m : ℝ) (x y : ℝ) (h1 : 1 ≤ m) (h2 : m ≤ 3) :
  (m - 1) * x^2 + (3 - m) * y^2 = (m - 1) * (3 - m) :=
sorry

end not_hyperbola_condition_l131_131903


namespace find_base_l131_131106

theorem find_base (r : ℕ) : 
  (2 * r^2 + 1 * r + 0) + (2 * r^2 + 6 * r + 0) = 5 * r^2 + 0 * r + 0 → r = 7 :=
by
  sorry

end find_base_l131_131106


namespace choose_5_person_committee_l131_131104

theorem choose_5_person_committee : nat.choose 12 5 = 792 := 
by
  sorry

end choose_5_person_committee_l131_131104


namespace probability_one_head_two_tails_l131_131740

-- Define an enumeration for Coin with two possible outcomes: heads and tails.
inductive Coin
| heads
| tails

-- Function to count the number of heads in a list of Coin.
def countHeads : List Coin → Nat
| [] => 0
| Coin.heads :: xs => 1 + countHeads xs
| Coin.tails :: xs => countHeads xs

-- Function to calculate the probability of a specific event given the total outcomes.
def probability (specific_events total_outcomes : Nat) : Rat :=
  (specific_events : Rat) / (total_outcomes : Rat)

-- The main theorem
theorem probability_one_head_two_tails : probability 3 8 = (3 / 8 : Rat) :=
sorry

end probability_one_head_two_tails_l131_131740


namespace inequality_proof_l131_131393

theorem inequality_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
(h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d) (h_sum : a + b + c + d ≥ 1) : 
  a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 ≥ 1 := 
by 
  sorry

end inequality_proof_l131_131393


namespace height_of_model_l131_131989

noncomputable def original_monument_height : ℝ := 100
noncomputable def original_monument_radius : ℝ := 20
noncomputable def original_monument_volume : ℝ := 125600
noncomputable def model_volume : ℝ := 1.256

theorem height_of_model : original_monument_height / (original_monument_volume / model_volume)^(1/3) = 1 :=
by
  sorry

end height_of_model_l131_131989


namespace range_of_a_l131_131093

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1^2 * Real.exp (-x1) = a) 
    ∧ (x2^2 * Real.exp (-x2) = a) ∧ (x3^2 * Real.exp (-x3) = a)) ↔ (0 < a ∧ a < 4 * Real.exp (-2)) :=
sorry

end range_of_a_l131_131093


namespace necessary_but_not_sufficient_l131_131068

def p (a : ℝ) : Prop := (a - 1) * (a - 2) = 0
def q (a : ℝ) : Prop := a = 1

theorem necessary_but_not_sufficient (a : ℝ) : 
  (q a → p a) ∧ (p a → q a → False) :=
by
  sorry

end necessary_but_not_sufficient_l131_131068


namespace car_speed_second_hour_l131_131148

theorem car_speed_second_hour
  (v1 : ℕ) (avg_speed : ℕ) (time : ℕ) (v2 : ℕ)
  (h1 : v1 = 90)
  (h2 : avg_speed = 70)
  (h3 : time = 2) :
  v2 = 50 :=
by
  sorry

end car_speed_second_hour_l131_131148


namespace min_seats_occupied_l131_131501

theorem min_seats_occupied (total_seats : ℕ) (h_total_seats : total_seats = 180) : 
  ∃ min_occupied : ℕ, 
    min_occupied = 90 ∧ 
    (∀ num_occupied : ℕ, num_occupied < min_occupied -> 
      ∃ next_seat : ℕ, (next_seat ≤ total_seats ∧ 
      num_occupied + next_seat < total_seats ∧ 
      (next_seat + 1 ≤ total_seats → ∃ a b: ℕ, a = next_seat ∧ b = next_seat + 1 ∧ 
      num_occupied + 1 < min_occupied ∧ 
      (a = b ∨ b = a + 1)))) :=
sorry

end min_seats_occupied_l131_131501


namespace transformed_graph_area_l131_131715

theorem transformed_graph_area (g : ℝ → ℝ) (a b : ℝ)
  (h_area_g : ∫ x in a..b, g x = 15) :
  ∫ x in a..b, 2 * g (x + 3) = 30 := 
sorry

end transformed_graph_area_l131_131715


namespace vehicle_value_fraction_l131_131150

theorem vehicle_value_fraction (V_this_year V_last_year : ℕ)
  (h_this_year : V_this_year = 16000)
  (h_last_year : V_last_year = 20000) :
  (V_this_year : ℚ) / V_last_year = 4 / 5 := by 
  rw [h_this_year, h_last_year]
  norm_num 
  sorry

end vehicle_value_fraction_l131_131150


namespace find_num_20_paise_coins_l131_131023

def num_20_paise_coins (x y : ℕ) : Prop :=
  x + y = 334 ∧ 20 * x + 25 * y = 7100

theorem find_num_20_paise_coins (x y : ℕ) (h : num_20_paise_coins x y) : x = 250 :=
by
  sorry

end find_num_20_paise_coins_l131_131023


namespace letter_puzzle_solutions_l131_131225

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def to_number (B A : ℕ) : ℕ :=
  10 * B + A

theorem letter_puzzle_solutions (A B : ℕ) (h_diff : A ≠ B) (h_digits : 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9)
  (h_two_digit : is_two_digit (to_number B A)) :
  A^B = to_number B A ↔ (A = 2 ∧ B = 5 ∨ A = 6 ∧ B = 2 ∨ A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l131_131225


namespace initial_pieces_l131_131046

-- Definitions based on given conditions
variable (left : ℕ) (used : ℕ)
axiom cond1 : left = 93
axiom cond2 : used = 4

-- The mathematical proof problem statement
theorem initial_pieces (left used : ℕ) (cond1 : left = 93) (cond2 : used = 4) : left + used = 97 :=
by
  sorry

end initial_pieces_l131_131046


namespace proof_problem_l131_131811

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x - 1)^2

theorem proof_problem : f (g (-3)) = 67 := 
by 
  sorry

end proof_problem_l131_131811


namespace problem_l131_131295

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 12 / Real.log 6

theorem problem : a > b ∧ b > c := by
  sorry

end problem_l131_131295


namespace inequality_positive_integers_l131_131383

theorem inequality_positive_integers (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  |n * Real.sqrt (n^2 + 1) - m| ≥ Real.sqrt 2 - 1 :=
sorry

end inequality_positive_integers_l131_131383


namespace slope_of_tangent_line_at_zero_l131_131873

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem slope_of_tangent_line_at_zero : (deriv f 0) = 1 :=
by
  sorry 

end slope_of_tangent_line_at_zero_l131_131873


namespace sum_of_digits_of_largest_n_l131_131574

def is_prime (p : ℕ) : Prop := Nat.Prime p

def is_single_digit_prime (p : ℕ) : Prop := is_prime p ∧ p < 10

noncomputable def required_n (d e : ℕ) : ℕ := d * e * (d^2 + 10 * e)

def sum_of_digits (n : ℕ) : ℕ := 
  if n < 10 then n 
  else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_largest_n : 
  ∃ (d e : ℕ), 
    is_single_digit_prime d ∧ is_single_digit_prime e ∧ 
    is_prime (d^2 + 10 * e) ∧ 
    (∀ d' e' : ℕ, is_single_digit_prime d' ∧ is_single_digit_prime e' ∧ is_prime (d'^2 + 10 * e') → required_n d e ≥ required_n d' e') ∧ 
    sum_of_digits (required_n d e) = 9 :=
sorry

end sum_of_digits_of_largest_n_l131_131574


namespace square_area_given_equal_perimeters_l131_131486

theorem square_area_given_equal_perimeters 
  (a b c : ℝ) (a_eq : a = 7.5) (b_eq : b = 9.5) (c_eq : c = 12) 
  (sq_perimeter_eq_tri : 4 * s = a + b + c) : 
  s^2 = 52.5625 :=
by
  sorry

end square_area_given_equal_perimeters_l131_131486


namespace smallest_rel_prime_to_180_l131_131253

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  { intro y,
    intro h,
    cases h with h1 h2,
    repeat { try { apply dec_trivial,
                   apply lt_or_eq_of_le,
                   norm_num,
                   apply Nat.prime_not_dvd_mul,
                   norm_num,
                   apply not_or_distrib.mpr,
                   split,
                   norm_cast,
                   intro,
                   exact le_antisymm _ },
           sorry }
end

end smallest_rel_prime_to_180_l131_131253


namespace arithmetic_seq_term_six_l131_131930

theorem arithmetic_seq_term_six {a : ℕ → ℝ} (a1 : ℝ) (S3 : ℝ) (h1 : a1 = 2) (h2 : S3 = 12) :
  a 6 = 12 :=
sorry

end arithmetic_seq_term_six_l131_131930


namespace count_distinct_special_sums_l131_131371

noncomputable def special_fractions : Finset ℚ :=
Finset.filter (λ q, ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b = 18 ∧ q = a / b) (Finset.image (λ p : ℕ × ℕ, p.1 / p.2) { p : ℕ × ℕ | p.1 + p.2 = 18 ∧ 0 < p.1 ∧ 0 < p.2}.to_finset)

noncomputable def special_sums : Finset ℤ :=
Finset.image (λ p : ℚ × ℚ, (p.1 + p.2).num) (special_fractions.product special_fractions)

theorem count_distinct_special_sums : special_sums.card = 15 :=
sorry

end count_distinct_special_sums_l131_131371


namespace paper_cut_count_incorrect_l131_131852

theorem paper_cut_count_incorrect (n : ℕ) (h : n = 1961) : 
  ∀ i, (∃ k, i = 7 ∨ i = 7 + 6 * k) → i % 6 = 1 → n ≠ i :=
by
  sorry

end paper_cut_count_incorrect_l131_131852


namespace binomial_coefficient_x3_expansion_l131_131374

theorem binomial_coefficient_x3_expansion :
  let x := @X ℚ _ 
  let term := (Polynomial.X - (Polynomial.C (1 : ℚ) * Polynomial.X ^ -2)) ^ 6
  (term.coeff 3) = -6 := by
  sorry

end binomial_coefficient_x3_expansion_l131_131374


namespace part1_part2_l131_131971

noncomputable def f (x a : ℝ) : ℝ := |x - a|

theorem part1 (x : ℝ) : (f x 2 ≥ 7 - |x - 1|) ↔ (x ≤ -2 ∨ x ≥ 5) :=
by sorry

theorem part2 (a : ℝ) (h : ∀ x, f x a ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) : a = 1 :=
by sorry

end part1_part2_l131_131971


namespace exists_eulerian_circuit_without_small_cycles_l131_131132

open Nat

noncomputable def has_eulerian_circuit_without_small_cycles (p : ℕ) (hp : Nat.Prime p) (h_large : p > 2023) : Prop :=
  ∃ (g : ℕ), (∀ (a b : ℕ), a ≤ 2023 → b ≤ 2023 → g ≠ (-a) / b) ∧
    ∃ (circuit : List (Fin p)), 
    (∀ (v : Fin p), v ∈ circuit) ∧
    (∃ (seqs : List (List (Fin p))), 
      (∀ seq ∈ seqs, ∀ i j, i ≠ j → seq[i] ≠ seq[j]) ∧
      seqs.chain' (λ s t, t.head = s.last) ∧ 
      (circuit = seqs.join) ∧
      (∀ k ≤ 2023, ¬ list_cycle_length circuit k))

theorem exists_eulerian_circuit_without_small_cycles (p : ℕ) (hp : Nat.Prime p) (h_large : p > 2023) :
  has_eulerian_circuit_without_small_cycles p hp h_large :=
sorry

end exists_eulerian_circuit_without_small_cycles_l131_131132


namespace absolute_value_simplification_l131_131674

theorem absolute_value_simplification (x : ℝ) (h : x > 3) : 
  |x - Real.sqrt ((x - 3)^2)| = 3 := 
by 
  sorry

end absolute_value_simplification_l131_131674


namespace eq_abs_distinct_solution_count_l131_131773

theorem eq_abs_distinct_solution_count :
  ∃! x : ℝ, |x - 10| = |x + 5| + 2 := 
sorry

end eq_abs_distinct_solution_count_l131_131773


namespace factor_of_quadratic_polynomial_l131_131057

theorem factor_of_quadratic_polynomial (t : ℚ) :
  (8 * t^2 + 22 * t + 5 = 0) ↔ (t = -1/4) ∨ (t = -5/2) :=
by sorry

end factor_of_quadratic_polynomial_l131_131057


namespace ellipse_eccentricity_l131_131400

noncomputable def eccentricity_of_ellipse (a c : ℝ) : ℝ :=
  c / a

theorem ellipse_eccentricity (F1 A : ℝ) (v : ℝ) (a c : ℝ)
  (h1 : 4 * a = 10 * (a - c))
  (h2 : F1 = 0 ∧ A = 0 ∧ v ≠ 0) :
  eccentricity_of_ellipse a c = 3 / 5 := by
sorry

end ellipse_eccentricity_l131_131400


namespace overall_percentage_loss_l131_131038

noncomputable def original_price : ℝ := 100
noncomputable def increased_price : ℝ := original_price * 1.36
noncomputable def first_discount_price : ℝ := increased_price * 0.90
noncomputable def second_discount_price : ℝ := first_discount_price * 0.85
noncomputable def third_discount_price : ℝ := second_discount_price * 0.80
noncomputable def final_price_with_tax : ℝ := third_discount_price * 1.05
noncomputable def percentage_change : ℝ := ((final_price_with_tax - original_price) / original_price) * 100

theorem overall_percentage_loss : percentage_change = -12.6064 :=
by
  sorry

end overall_percentage_loss_l131_131038


namespace sum_of_reciprocals_l131_131728

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : (1/x) + (1/y) = 3/8 :=
by
  sorry

end sum_of_reciprocals_l131_131728


namespace ways_to_choose_providers_l131_131697

theorem ways_to_choose_providers : (25 * 24 * 23 * 22 = 303600) :=
by
  sorry

end ways_to_choose_providers_l131_131697


namespace interest_rate_is_five_percent_l131_131682

-- Define the principal amount P and the interest rate r.
variables (P : ℝ) (r : ℝ)

-- Define the conditions given in the problem
def simple_interest_condition : Prop := P * r * 2 = 40
def compound_interest_condition : Prop := P * (1 + r)^2 - P = 41

-- Define the goal statement to prove
theorem interest_rate_is_five_percent (h1 : simple_interest_condition P r) (h2 : compound_interest_condition P r) : r = 0.05 :=
sorry

end interest_rate_is_five_percent_l131_131682


namespace non_neg_integer_solutions_for_inequality_l131_131851

theorem non_neg_integer_solutions_for_inequality :
  {x : ℤ | 5 * x - 1 < 3 * (x + 1) ∧ (1 - x) / 3 ≤ 1 ∧ 0 ≤ x } = {0, 1} := 
by {
  sorry
}

end non_neg_integer_solutions_for_inequality_l131_131851


namespace expected_worth_coin_flip_l131_131645

def prob_head : ℚ := 2 / 3
def prob_tail : ℚ := 1 / 3
def gain_head : ℚ := 5
def loss_tail : ℚ := -12

theorem expected_worth_coin_flip : ∃ E : ℚ, E = round (((prob_head * gain_head) + (prob_tail * loss_tail)) * 100) / 100 ∧ E = - (2 / 3) :=
by
  sorry

end expected_worth_coin_flip_l131_131645


namespace smallest_number_of_eggs_l131_131010

theorem smallest_number_of_eggs (c : ℕ) (eggs_total : ℕ) :
  eggs_total = 15 * c - 3 ∧ eggs_total > 150 → eggs_total = 162 :=
by
  sorry

end smallest_number_of_eggs_l131_131010


namespace probability_heads_equals_7_over_11_l131_131975

theorem probability_heads_equals_7_over_11 (p : ℝ) (q : ℝ)
  (h1 : q = 1 - p)
  (h2 : 120 * p^7 * q^3 = 210 * p^6 * q^4) :
  p = 7 / 11 :=
by {
  sorry
}

end probability_heads_equals_7_over_11_l131_131975


namespace initial_bottles_l131_131705

theorem initial_bottles (x : ℕ) (h1 : x - 8 + 45 = 51) : x = 14 :=
by
  -- Proof goes here
  sorry

end initial_bottles_l131_131705


namespace arithmetic_mean_of_two_digit_multiples_of_5_l131_131000

theorem arithmetic_mean_of_two_digit_multiples_of_5:
  let smallest := 10
  let largest := 95
  let num_terms := 18
  let sum := 945
  let mean := (sum : ℝ) / (num_terms : ℝ)
  mean = 52.5 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_5_l131_131000


namespace smallest_b_for_perfect_square_l131_131003

theorem smallest_b_for_perfect_square : ∃ (b : ℕ), b > 4 ∧ (∃ k, (2 * b + 4) = k * k) ∧
                                             ∀ (b' : ℕ), b' > 4 ∧ (∃ k, (2 * b' + 4) = k * k) → b ≤ b' :=
by
  sorry

end smallest_b_for_perfect_square_l131_131003


namespace negation_true_l131_131870

theorem negation_true (a : ℝ) : ¬ (∀ a : ℝ, a ≤ 2 → a^2 < 4) :=
sorry

end negation_true_l131_131870


namespace area_of_shaded_region_l131_131196

-- Definitions of given conditions
def octagon_side_length : ℝ := 5
def arc_radius : ℝ := 4

-- Theorem statement
theorem area_of_shaded_region : 
  let octagon_area := 50
  let sectors_area := 16 * Real.pi
  octagon_area - sectors_area = 50 - 16 * Real.pi :=
by
  sorry

end area_of_shaded_region_l131_131196


namespace largest_divisor_of_n4_sub_4n2_is_4_l131_131921

theorem largest_divisor_of_n4_sub_4n2_is_4 (n : ℤ) : 4 ∣ (n^4 - 4 * n^2) :=
sorry

end largest_divisor_of_n4_sub_4n2_is_4_l131_131921


namespace village_population_rate_l131_131157

theorem village_population_rate
    (population_X : ℕ := 68000)
    (population_Y : ℕ := 42000)
    (increase_Y : ℕ := 800)
    (years : ℕ := 13) :
  ∃ R : ℕ, population_X - years * R = population_Y + years * increase_Y ∧ R = 1200 :=
by
  exists 1200
  sorry

end village_population_rate_l131_131157


namespace sector_area_proof_l131_131666

-- Define the sector with its characteristics
structure sector :=
  (r : ℝ)            -- radius
  (theta : ℝ)        -- central angle

-- Given conditions
def sector_example : sector := {r := 1, theta := 2}

-- Definition of perimeter for a sector
def perimeter (sec : sector) : ℝ :=
  2 * sec.r + sec.theta * sec.r

-- Definition of area for a sector
def area (sec : sector) : ℝ :=
  0.5 * sec.r * (sec.theta * sec.r)

-- Theorem statement based on the problem statement
theorem sector_area_proof (sec : sector) (h1 : perimeter sec = 4) (h2 : sec.theta = 2) : area sec = 1 := 
  sorry

end sector_area_proof_l131_131666


namespace triangle_ineq_sqrt_triangle_l131_131158

open Real

theorem triangle_ineq_sqrt_triangle (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a):
  (∃ u v w : ℝ, u > 0 ∧ v > 0 ∧ w > 0 ∧ a = v + w ∧ b = u + w ∧ c = u + v) ∧ 
  (sqrt (a * b) + sqrt (b * c) + sqrt (c * a) ≤ a + b + c ∧ a + b + c ≤ 2 * sqrt (a * b) + 2 * sqrt (b * c) + 2 * sqrt (c * a)) :=
  sorry

end triangle_ineq_sqrt_triangle_l131_131158


namespace isosceles_right_triangle_hypotenuse_length_l131_131381

theorem isosceles_right_triangle_hypotenuse_length (A B C : ℝ) (h1 : (A = 0) ∧ (B = 0) ∧ (C = 1)) (h2 : AC = 5) (h3 : BC = 5) : 
  AB = 5 * Real.sqrt 2 := 
sorry

end isosceles_right_triangle_hypotenuse_length_l131_131381


namespace matt_total_score_l131_131983

-- Definitions from the conditions
def num_2_point_shots : ℕ := 4
def num_3_point_shots : ℕ := 2
def score_per_2_point_shot : ℕ := 2
def score_per_3_point_shot : ℕ := 3

-- Proof statement
theorem matt_total_score : 
  (num_2_point_shots * score_per_2_point_shot) + 
  (num_3_point_shots * score_per_3_point_shot) = 14 := 
by 
  sorry  -- placeholder for the actual proof

end matt_total_score_l131_131983


namespace elena_fraction_left_l131_131053

variable (M : ℝ) -- Total amount of money
variable (B : ℝ) -- Total cost of all the books

-- Condition: Elena spends one-third of her money to buy half of the books
def condition : Prop := (1 / 3) * M = (1 / 2) * B

-- Goal: Fraction of the money left after buying all the books is one-third
theorem elena_fraction_left (h : condition M B) : (M - B) / M = 1 / 3 :=
by
  sorry

end elena_fraction_left_l131_131053


namespace grandma_red_bacon_bits_l131_131365

theorem grandma_red_bacon_bits:
  ∀ (mushrooms cherryTomatoes pickles baconBits redBaconBits : ℕ),
    mushrooms = 3 →
    cherryTomatoes = 2 * mushrooms →
    pickles = 4 * cherryTomatoes →
    baconBits = 4 * pickles →
    redBaconBits = 1 / 3 * baconBits →
    redBaconBits = 32 := 
by
  intros mushrooms cherryTomatoes pickles baconBits redBaconBits
  intros h1 h2 h3 h4 h5
  sorry

end grandma_red_bacon_bits_l131_131365


namespace time_to_cross_platform_l131_131318

-- Definitions for the length of the train, the length of the platform, and the speed of the train
def length_train : ℕ := 750
def length_platform : ℕ := 750
def speed_train_kmh : ℕ := 90

-- Conversion constants
def meters_per_kilometer : ℕ := 1000
def seconds_per_hour : ℕ := 3600

-- Convert speed from km/hr to m/s
def speed_train_ms : ℚ := speed_train_kmh * meters_per_kilometer / seconds_per_hour

-- Total distance the train covers to cross the platform
def total_distance : ℕ := length_train + length_platform

-- Proof problem: To prove that the time taken to cross the platform is 60 seconds
theorem time_to_cross_platform : total_distance / speed_train_ms = 60 := by
  sorry

end time_to_cross_platform_l131_131318


namespace quadratic_equal_roots_relation_l131_131816

theorem quadratic_equal_roots_relation (a b c : ℝ) (h₁ : b ≠ c) 
  (h₂ : ∀ x : ℝ, (b - c) * x^2 + (a - b) * x + (c - a) = 0 → 
          (a - b)^2 - 4 * (b - c) * (c - a) = 0) : 
  c = (a + b) / 2 := sorry

end quadratic_equal_roots_relation_l131_131816


namespace remainder_when_y_squared_divided_by_30_l131_131274

theorem remainder_when_y_squared_divided_by_30 (y : ℤ) :
  6 * y ≡ 12 [ZMOD 30] → 5 * y ≡ 25 [ZMOD 30] → y ^ 2 ≡ 19 [ZMOD 30] :=
  by
  intro h1 h2
  sorry

end remainder_when_y_squared_divided_by_30_l131_131274


namespace stephen_total_distance_l131_131714

-- Define the conditions
def trips : ℕ := 10
def mountain_height : ℝ := 40000
def fraction_of_height_reached : ℝ := 3 / 4

-- Calculate the total distance covered
def total_distance_covered : ℝ :=
  2 * (fraction_of_height_reached * mountain_height) * trips

-- Prove the total distance covered is 600,000 feet
theorem stephen_total_distance :
  total_distance_covered = 600000 := by
  sorry

end stephen_total_distance_l131_131714


namespace choir_girls_count_l131_131730

noncomputable def number_of_girls_in_choir (o b t c b_boys : ℕ) : ℕ :=
  c - b_boys

theorem choir_girls_count (o b t b_boys : ℕ) (h1 : o = 20) (h2 : b = 2 * o) (h3 : t = 88)
  (h4 : b_boys = 12) : number_of_girls_in_choir o b t (t - (o + b)) b_boys = 16 :=
by
  sorry

end choir_girls_count_l131_131730


namespace hall_reunion_attendance_l131_131044

/-- At the Taj Hotel, two family reunions are happening: the Oates reunion and the Hall reunion.
All 150 guests at the hotel attend at least one of the reunions.
70 people attend the Oates reunion.
28 people attend both reunions.
Prove that 108 people attend the Hall reunion. -/
theorem hall_reunion_attendance (total oates both : ℕ) (h_total : total = 150) (h_oates : oates = 70) (h_both : both = 28) :
  ∃ hall : ℕ, total = oates + hall - both ∧ hall = 108 :=
by
  -- Proof will be skipped and not considered for this task
  sorry

end hall_reunion_attendance_l131_131044


namespace find_value_of_2a_plus_c_l131_131258

theorem find_value_of_2a_plus_c (a b c : ℝ) (h1 : 3 * a + b + 2 * c = 3) (h2 : a + 3 * b + 2 * c = 1) :
  2 * a + c = 2 :=
sorry

end find_value_of_2a_plus_c_l131_131258


namespace Zachary_did_47_pushups_l131_131626

-- Define the conditions and the question
def Zachary_pushups (David_pushups difference : ℕ) : ℕ :=
  David_pushups - difference

theorem Zachary_did_47_pushups :
  Zachary_pushups 62 15 = 47 :=
by
  -- Provide the proof here (we'll use sorry for now)
  sorry

end Zachary_did_47_pushups_l131_131626


namespace find_smaller_number_l131_131452

-- Define the conditions as hypotheses and the goal as a proposition
theorem find_smaller_number (x y : ℕ) (h1 : x + y = 77) (h2 : x = 42 ∨ y = 42) (h3 : 5 * x = 6 * y) : x = 35 :=
sorry

end find_smaller_number_l131_131452


namespace per_capita_income_growth_l131_131186

noncomputable def income2020 : ℝ := 3.2
noncomputable def income2022 : ℝ := 3.7
variable (x : ℝ)

/--
Prove the per capita disposable income model.
-/
theorem per_capita_income_growth :
  income2020 * (1 + x)^2 = income2022 :=
sorry

end per_capita_income_growth_l131_131186


namespace age_ratio_l131_131471

theorem age_ratio 
  (a b c : ℕ)
  (h1 : a = b + 2)
  (h2 : a + b + c = 32)
  (h3 : b = 12) :
  b = 2 * c :=
by
  sorry

end age_ratio_l131_131471


namespace smallest_rel_prime_to_180_l131_131234

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Int.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Int.gcd y 180 = 1 → x ≤ y := 
sorry

end smallest_rel_prime_to_180_l131_131234


namespace fire_brigade_allocation_l131_131350

theorem fire_brigade_allocation (sites fire_brigades : ℕ) (h_sites : sites = 3) (h_fire_brigades : fire_brigades = 4) :
  ∃ (allocations : ℕ), (∀ (site : ℕ), 1 ≤ site ∧ site ≤ sites → ∃ (brigades : ℕ), brigades ≥ 1) ∧ allocations = 36 := 
by {
  sorry
}

end fire_brigade_allocation_l131_131350


namespace maximize_revenue_l131_131437

-- Define the problem conditions
def is_valid (x y : ℕ) : Prop :=
  x + y ≤ 60 ∧ 6 * x + 30 * y ≤ 600

-- Define the objective function
def revenue (x y : ℕ) : ℚ :=
  2.5 * x + 7.5 * y

-- State the theorem with the given conditions
theorem maximize_revenue : 
  (∃ x y : ℕ, is_valid x y ∧ ∀ a b : ℕ, is_valid a b → revenue x y >= revenue a b) ∧
  ∃ x y, is_valid x y ∧ revenue x y = revenue 50 10 := 
sorry

end maximize_revenue_l131_131437


namespace solve_rational_numbers_l131_131573

theorem solve_rational_numbers 
  (a b c d : ℚ)
  (h₁ : a + b + c = -1)
  (h₂ : a + b + d = -3)
  (h₃ : a + c + d = 2)
  (h₄ : b + c + d = 17) :
  a = 6 ∧ b = 8 ∧ c = 3 ∧ d = -12 := 
by
  sorry

end solve_rational_numbers_l131_131573


namespace find_9b_l131_131090

variable (a b : ℚ)

theorem find_9b (h1 : 7 * a + 3 * b = 0) (h2 : a = b - 4) : 9 * b = 126 / 5 := 
by
  sorry

end find_9b_l131_131090


namespace hundredth_term_sequence_l131_131582

def numerators (n : ℕ) : ℕ := 1 + (n - 1) * 2
def denominators (n : ℕ) : ℕ := 2 + (n - 1) * 3

theorem hundredth_term_sequence : numerators 100 / denominators 100 = 199 / 299 := by
  sorry

end hundredth_term_sequence_l131_131582


namespace maria_dozen_flowers_l131_131063

theorem maria_dozen_flowers (x : ℕ) (h : 12 * x + 2 * x = 42) : x = 3 :=
by
  sorry

end maria_dozen_flowers_l131_131063


namespace closest_perfect_square_to_314_l131_131622

theorem closest_perfect_square_to_314 :
  ∃ n : ℤ, n^2 = 324 ∧ ∀ m : ℤ, m^2 ≠ 324 → |m^2 - 314| > |324 - 314| :=
by
  sorry

end closest_perfect_square_to_314_l131_131622


namespace carol_name_tag_l131_131324

theorem carol_name_tag (a b c : ℕ) (ha : Prime a ∧ a ≥ 10 ∧ a < 100) (hb : Prime b ∧ b ≥ 10 ∧ b < 100) (hc : Prime c ∧ c ≥ 10 ∧ c < 100) 
  (h1 : b + c = 14) (h2 : a + c = 20) (h3 : a + b = 18) : c = 11 := 
by 
  sorry

end carol_name_tag_l131_131324


namespace min_value_expression_l131_131702

theorem min_value_expression (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z = 1) :
  (x^2 + 8 * x * y + 25 * y^2 + 16 * y * z + 9 * z^2) ≥ 403 / 9 := by
  sorry

end min_value_expression_l131_131702


namespace income_growth_rate_l131_131190

noncomputable def income_growth_eq (x : ℝ) : Prop :=
  3.2 * (1 + x)^2 = 3.7

theorem income_growth_rate :
  ∃ x : ℝ, income_growth_eq x :=
sorry

end income_growth_rate_l131_131190


namespace distance_between_A_and_B_l131_131028

-- Define speeds, times, and distances as real numbers
def speed_A_to_B := 42.5
def time_travelled := 1.5
def remaining_to_midpoint := 26.0

-- Define the total distance between A and B as a variable
def distance_A_to_B : ℝ := 179.5

-- Prove that the distance between locations A and B is 179.5 kilometers given the conditions
theorem distance_between_A_and_B : (42.5 * 1.5 + 26) * 2 = 179.5 :=
by 
  sorry

end distance_between_A_and_B_l131_131028


namespace sufficient_condition_range_k_l131_131789

theorem sufficient_condition_range_k {x k : ℝ} (h : ∀ x, x > k → (3 / (x + 1) < 1)) : k ≥ 2 :=
sorry

end sufficient_condition_range_k_l131_131789


namespace percentage_of_Luccas_balls_are_basketballs_l131_131122

-- Defining the variables and their conditions 
variables (P : ℝ) (Lucca_Balls : ℕ := 100) (Lucien_Balls : ℕ := 200)
variable (Total_Basketballs : ℕ := 50)

-- Condition that Lucien has 20% basketballs
def Lucien_Basketballs := (20 / 100) * Lucien_Balls

-- We need to prove that percentage of Lucca's balls that are basketballs is 10%
theorem percentage_of_Luccas_balls_are_basketballs :
  (P / 100) * Lucca_Balls + Lucien_Basketballs = Total_Basketballs → P = 10 :=
by
  sorry

end percentage_of_Luccas_balls_are_basketballs_l131_131122


namespace problem_am_gm_inequality_l131_131118

theorem problem_am_gm_inequality
  (a b c : ℝ)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_sq : a^2 + b^2 + c^2 = 3) : 
  (1 / (1 + a * b)) + (1 / (1 + b * c)) + (1 / (1 + c * a)) ≥ 3 / 2 :=
by
  sorry

end problem_am_gm_inequality_l131_131118


namespace fraction_taken_by_kiley_l131_131322

-- Define the constants and conditions
def total_crayons : ℕ := 48
def remaining_crayons_after_joe : ℕ := 18

-- Define the main statement to be proven
theorem fraction_taken_by_kiley (f : ℚ) : 
  (48 - (48 * f)) / 2 = 18 → f = 1 / 4 :=
by 
  intro h
  sorry

end fraction_taken_by_kiley_l131_131322


namespace similar_triangles_iff_l131_131131

variables {a b c a' b' c' : ℂ}

theorem similar_triangles_iff :
  (∃ (z w : ℂ), a' = a * z + w ∧ b' = b * z + w ∧ c' = c * z + w) ↔
  a' * (b - c) + b' * (c - a) + c' * (a - b) = 0 :=
sorry

end similar_triangles_iff_l131_131131


namespace angle_triple_complement_l131_131330

theorem angle_triple_complement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 := 
by
  sorry

end angle_triple_complement_l131_131330


namespace estimate_red_balls_l131_131633

-- Define the conditions in Lean 4
def total_balls : ℕ := 15
def freq_red_ball : ℝ := 0.4

-- Define the proof statement without proving it
theorem estimate_red_balls (x : ℕ) 
  (h1 : x ≤ total_balls) 
  (h2 : ∃ (p : ℝ), p = x / total_balls ∧ p = freq_red_ball) :
  x = 6 :=
sorry

end estimate_red_balls_l131_131633


namespace sqrt_2_plus_x_nonnegative_l131_131137

theorem sqrt_2_plus_x_nonnegative (x : ℝ) : (2 + x ≥ 0) → (x ≥ -2) :=
by
  sorry

end sqrt_2_plus_x_nonnegative_l131_131137


namespace perfect_square_quotient_l131_131072

theorem perfect_square_quotient (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h : (a * b + 1) ∣ (a * a + b * b)) : 
  ∃ k : ℕ, (a * a + b * b) = (a * b + 1) * (k * k) := 
sorry

end perfect_square_quotient_l131_131072


namespace find_values_of_a_l131_131779

theorem find_values_of_a :
  ∃ (a : ℝ), 
    (∀ x y, (|y + 2| + |x - 11| - 3) * (x^2 + y^2 - 13) = 0 ∧ 
             (x - 5)^2 + (y + 2)^2 = a) ↔ 
    a = 9 ∨ a = 42 + 2 * Real.sqrt 377 :=
sorry

end find_values_of_a_l131_131779


namespace larger_segment_length_l131_131872

theorem larger_segment_length (a b c : ℕ) (h : ℝ) (x : ℝ)
  (ha : a = 50) (hb : b = 90) (hc : c = 110)
  (hyp1 : a^2 = x^2 + h^2)
  (hyp2 : b^2 = (c - x)^2 + h^2) :
  110 - x = 80 :=
by {
  sorry
}

end larger_segment_length_l131_131872


namespace smallest_rel_prime_to_180_l131_131248

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ x ≤ 7 ∧ (∀ y : ℕ, y > 1 ∧ y < x → y.gcd 180 ≠ 1) ∧ x.gcd 180 = 1 :=
by
  sorry

end smallest_rel_prime_to_180_l131_131248


namespace range_of_a_l131_131941

-- Define the inequality condition
def inequality (x a : ℝ) : Prop :=
  2 * x^2 + a * x - a^2 > 0

-- State the main problem
theorem range_of_a (a: ℝ) : 
  inequality 2 a -> (-2 < a) ∧ (a < 4) :=
by
  sorry

end range_of_a_l131_131941


namespace prize_distribution_correct_l131_131879

def probability_A_correct : ℚ := 3 / 4
def probability_B_correct : ℚ := 4 / 5
def total_prize : ℚ := 190

-- Calculation of expected prizes
def probability_A_only_correct : ℚ := probability_A_correct * (1 - probability_B_correct)
def probability_B_only_correct : ℚ := probability_B_correct * (1 - probability_A_correct)
def probability_both_correct : ℚ := probability_A_correct * probability_B_correct

def normalized_probability : ℚ := probability_A_only_correct + probability_B_only_correct + probability_both_correct

def expected_prize_A : ℚ := (probability_A_only_correct / normalized_probability * total_prize) + (probability_both_correct / normalized_probability * (total_prize / 2))
def expected_prize_B : ℚ := (probability_B_only_correct / normalized_probability * total_prize) + (probability_both_correct / normalized_probability * (total_prize / 2))

theorem prize_distribution_correct :
  expected_prize_A = 90 ∧ expected_prize_B = 100 := 
by
  sorry

end prize_distribution_correct_l131_131879


namespace problem_b_l131_131341

theorem problem_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b + a * b = 3) : a + b ≥ 2 :=
sorry

end problem_b_l131_131341


namespace mod_remainder_l131_131337

theorem mod_remainder (n : ℕ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := 
by
  sorry

end mod_remainder_l131_131337


namespace sector_area_l131_131286

theorem sector_area (α r : ℝ) (hα : α = 2) (h_r : r = 1 / Real.sin 1) : 
  (1 / 2) * r^2 * α = 1 / (Real.sin 1)^2 :=
by
  sorry

end sector_area_l131_131286


namespace letter_puzzle_solution_l131_131218

theorem letter_puzzle_solution :
  ∃ (A B : ℕ), (A ≠ B) ∧ (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ A ^ B = B * 10 + A :=
by {
  use 2, 5,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 2 ^ 5 = 5 * 10 + 2),
  sorry
} ∨
by {
  use 6, 2,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 6 ^ 2 = 2 * 10 + 6),
  sorry
} ∨
by {
  use 4, 3,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 4 ^ 3 = 3 * 10 + 4),
  sorry
}

end letter_puzzle_solution_l131_131218


namespace solve_equation_1_solve_equation_2_l131_131991

theorem solve_equation_1 (x : ℝ) (h : 0.5 * x + 1.1 = 6.5 - 1.3 * x) : x = 3 :=
  by sorry

theorem solve_equation_2 (x : ℝ) (h : (1 / 6) * (3 * x - 9) = (2 / 5) * x - 3) : x = -15 :=
  by sorry

end solve_equation_1_solve_equation_2_l131_131991


namespace average_probable_weight_l131_131478

-- Define the conditions
def Arun_opinion (w : ℝ) : Prop := 64 < w ∧ w < 72
def Brother_opinion (w : ℝ) : Prop := 60 < w ∧ w < 70
def Mother_opinion (w : ℝ) : Prop := w ≤ 67

-- The proof problem statement
theorem average_probable_weight :
  ∃ (w : ℝ), Arun_opinion w ∧ Brother_opinion w ∧ Mother_opinion w →
  (64 + 67) / 2 = 65.5 :=
by
  sorry

end average_probable_weight_l131_131478


namespace difference_of_two_numbers_l131_131888

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 :=
sorry

end difference_of_two_numbers_l131_131888


namespace count_perfect_squares_divisible_by_36_l131_131549

theorem count_perfect_squares_divisible_by_36 :
  let N := 10000
  let max_square := 10^8
  let multiple := 36
  let valid_divisor := 1296
  let count_multiples := 277
  (∀ N : ℕ, N^2 < max_square → (∃ k : ℕ, N = k * multiple ∧ k < N)) → 
  ∃ cnt : ℕ, cnt = count_multiples := 
by {
  sorry
}

end count_perfect_squares_divisible_by_36_l131_131549


namespace math_problem_l131_131572

open ProbabilityTheory

variables {X : ℕ → ℝ}

/-- S_n is the sum of the first n X_i's --/
def S (n : ℕ) : ℝ := ∑ i in Finset.range n, X i

theorem math_problem (h_indep : ∀ i j, i ≠ j → indep_fun (X i) (X j))
  (h_ident_dist : ∀ i j, ident_distrib (X i) (X j)) (n : ℕ) (c : ℝ) :
  P (λ ω, |(S (2 * n) ω) / (2 * n) - c| ≤ |(S n ω) / n - c|) ≥ 1 / 2 :=
sorry

end math_problem_l131_131572


namespace find_h_for_expression_l131_131679

theorem find_h_for_expression (a k : ℝ) (h : ℝ) :
  (∃ a k : ℝ, ∀ x : ℝ, x^2 - 6*x + 1 = a*(x - h)^3 + k) ↔ h = 2 :=
by
  sorry

end find_h_for_expression_l131_131679


namespace product_of_a_values_l131_131653

/--
Let a be a real number and consider the points P = (3 * a, a - 5) and Q = (5, -2).
Given that the distance between P and Q is 3 * sqrt 10, prove that the product
of all possible values of a is -28 / 5.
-/
theorem product_of_a_values :
  ∀ (a : ℝ),
  (dist (3 * a, a - 5) (5, -2) = 3 * Real.sqrt 10) →
  ∃ (a₁ a₂ : ℝ), (5 * a₁ * a₁ - 18 * a₁ - 28 = 0) ∧ 
                 (5 * a₂ * a₂ - 18 * a₂ - 28 = 0) ∧ 
                 (a₁ * a₂ = -28 / 5) := 
by
  sorry

end product_of_a_values_l131_131653


namespace count_perfect_squares_l131_131544

theorem count_perfect_squares (N : Nat) :
  ∃ k : Nat, k = 1666 ∧ ∀ m, (∃ n, m = n * n ∧ m < 10^8 ∧ 36 ∣ m) ↔ (m = 36 * k ^ 2 ∧ k < 10^4) :=
sorry

end count_perfect_squares_l131_131544


namespace sample_size_l131_131751

theorem sample_size (T : ℕ) (f_C : ℚ) (samples_C : ℕ) (n : ℕ) 
    (hT : T = 260)
    (hfC : f_C = 3 / 13)
    (hsamples_C : samples_C = 3) : n = 13 :=
by
  -- Proof goes here
  sorry

end sample_size_l131_131751


namespace Stephen_total_distance_l131_131713

theorem Stephen_total_distance 
  (round_trips : ℕ := 10) 
  (mountain_height : ℕ := 40000) 
  (fraction_of_height : ℚ := 3/4) :
  (round_trips * (2 * (fraction_of_height * mountain_height))) = 600000 :=
by
  sorry

end Stephen_total_distance_l131_131713


namespace sets_are_equal_l131_131704

-- Defining sets A and B as per the given conditions
def setA : Set ℕ := {x | ∃ a : ℕ, x = a^2 + 1}
def setB : Set ℕ := {y | ∃ b : ℕ, y = b^2 - 4 * b + 5}

-- Proving that set A is equal to set B
theorem sets_are_equal : setA = setB :=
by
  sorry

end sets_are_equal_l131_131704


namespace smallest_rel_prime_to_180_is_7_l131_131243

theorem smallest_rel_prime_to_180_is_7 :
  ∃ x : ℕ, x > 1 ∧ gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial, },
  split,
  { exact dec_trivial, },
  intros y hy,
  cases hy with hy1 hy2,
  exact dec_trivial,
end

end smallest_rel_prime_to_180_is_7_l131_131243


namespace sum_of_integers_l131_131860

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 :=
sorry

end sum_of_integers_l131_131860


namespace proof_of_expression_l131_131415

theorem proof_of_expression (a : ℝ) (h : a^2 + a + 1 = 2) : (5 - a) * (6 + a) = 29 :=
by {
  sorry
}

end proof_of_expression_l131_131415


namespace paige_folders_l131_131745

-- Definitions derived from the conditions
def initial_files : Nat := 27
def deleted_files : Nat := 9
def files_per_folder : Nat := 6

-- Define the remaining files after deletion
def remaining_files : Nat := initial_files - deleted_files

-- The theorem: Prove that the number of folders is 3
theorem paige_folders : remaining_files / files_per_folder = 3 := by
  sorry

end paige_folders_l131_131745


namespace ninety_percent_of_population_is_expected_number_l131_131894

/-- Define the total population of the village -/
def total_population : ℕ := 9000

/-- Define the percentage rate as a fraction -/
def percentage_rate : ℕ := 90

/-- Define the expected number of people representing 90% of the population -/
def expected_number : ℕ := 8100

/-- The proof problem: Prove that 90% of the total population is 8100 -/
theorem ninety_percent_of_population_is_expected_number :
  (percentage_rate * total_population / 100) = expected_number :=
by
  sorry

end ninety_percent_of_population_is_expected_number_l131_131894


namespace relationship_of_y_values_l131_131418

theorem relationship_of_y_values (m : ℝ) (y1 y2 y3 : ℝ) :
  (∀ x y, (x = -2 ∧ y = y1 ∨ x = -1 ∧ y = y2 ∨ x = 1 ∧ y = y3) → (y = (m^2 + 1) / x)) →
  y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relationship_of_y_values_l131_131418


namespace exist_positive_real_x_l131_131525

theorem exist_positive_real_x (x : ℝ) (hx1 : 0 < x) (hx2 : Nat.floor x * x = 90) : x = 10 := 
sorry

end exist_positive_real_x_l131_131525


namespace initial_pens_l131_131469

theorem initial_pens (P : ℤ) (INIT : 2 * (P + 22) - 19 = 39) : P = 7 :=
by
  sorry

end initial_pens_l131_131469


namespace delta_is_59_degrees_l131_131518

noncomputable def angleDelta : ℝ :=
  Real.arccos ((∑ i in (3271:ℕ)..(6871:ℕ), Real.sin (i * Real.pi / 180)) ^ (∑ j in (3240:ℕ)..(6840:ℕ), Real.cos (j * Real.pi / 180)) +
               ∑ k in (3241:ℕ)..(6840:ℕ), Real.cos (k * Real.pi / 180))

theorem delta_is_59_degrees :
  angleDelta = 59 * Real.pi / 180 :=
sorry

end delta_is_59_degrees_l131_131518


namespace conceived_number_is_seven_l131_131638

theorem conceived_number_is_seven (x : ℕ) (h1 : x > 0) (h2 : (1 / 4 : ℚ) * (10 * x + 7 - x * x) - x = 0) : x = 7 := by
  sorry

end conceived_number_is_seven_l131_131638


namespace prime_roots_quadratic_l131_131554

theorem prime_roots_quadratic (p q : ℕ) (x1 x2 : ℕ) 
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (h_prime_x1 : Nat.Prime x1)
  (h_prime_x2 : Nat.Prime x2)
  (h_eq : p * x1 * x1 + p * x2 * x2 - q * x1 * x2 + 1985 = 0) :
  12 * p * p + q = 414 :=
sorry

end prime_roots_quadratic_l131_131554


namespace red_paint_amount_l131_131283

theorem red_paint_amount (r w : ℕ) (hrw : r / w = 5 / 7) (hwhite : w = 21) : r = 15 :=
by {
  sorry
}

end red_paint_amount_l131_131283


namespace toms_total_score_l131_131461

def points_per_enemy : ℕ := 10
def enemies_killed : ℕ := 175

def base_score (enemies : ℕ) : ℝ := enemies * points_per_enemy

def bonus_percentage (enemies : ℕ) : ℝ :=
  if 100 ≤ enemies ∧ enemies < 150 then 0.50
  else if 150 ≤ enemies ∧ enemies < 200 then 0.75
  else if enemies ≥ 200 then 1.00
  else 0.0

def total_score (enemies : ℕ) : ℝ :=
  let base := base_score enemies
  let bonus := base * bonus_percentage enemies
  base + bonus

theorem toms_total_score :
  total_score enemies_killed = 3063 :=
by
  -- The proof will show the computed total score
  -- matches the expected value
  sorry

end toms_total_score_l131_131461


namespace wildcats_points_l131_131312

theorem wildcats_points (panthers_points wildcats_additional_points wildcats_points : ℕ)
  (h_panthers : panthers_points = 17)
  (h_wildcats : wildcats_additional_points = 19)
  (h_wildcats_points : wildcats_points = panthers_points + wildcats_additional_points) :
  wildcats_points = 36 :=
by
  have h1 : panthers_points = 17 := h_panthers
  have h2 : wildcats_additional_points = 19 := h_wildcats
  have h3 : wildcats_points = panthers_points + wildcats_additional_points := h_wildcats_points
  sorry

end wildcats_points_l131_131312


namespace joshua_finishes_after_malcolm_l131_131980

def time_difference_between_runners
  (race_length : ℕ)
  (malcolm_speed : ℕ)
  (joshua_speed : ℕ)
  (malcolm_finish_time : ℕ := malcolm_speed * race_length)
  (joshua_finish_time : ℕ := joshua_speed * race_length) : ℕ :=
joshua_finish_time - malcolm_finish_time

theorem joshua_finishes_after_malcolm
  (race_length : ℕ)
  (malcolm_speed : ℕ)
  (joshua_speed : ℕ)
  (h_race_length : race_length = 12)
  (h_malcolm_speed : malcolm_speed = 7)
  (h_joshua_speed : joshua_speed = 9) : time_difference_between_runners race_length malcolm_speed joshua_speed = 24 :=
by 
  subst h_race_length
  subst h_malcolm_speed
  subst h_joshua_speed
  rfl

#print joshua_finishes_after_malcolm

end joshua_finishes_after_malcolm_l131_131980


namespace elephant_entry_rate_l131_131735

-- Define the variables and constants
def initial_elephants : ℕ := 30000
def exit_rate : ℕ := 2880
def exit_time : ℕ := 4
def enter_time : ℕ := 7
def final_elephants : ℕ := 28980

-- Prove the rate of new elephants entering the park
theorem elephant_entry_rate :
  (final_elephants - (initial_elephants - exit_rate * exit_time)) / enter_time = 1500 :=
by
  sorry -- placeholder for the proof

end elephant_entry_rate_l131_131735


namespace find_N_l131_131828

theorem find_N (
    A B : ℝ) (N : ℕ) (r : ℝ) (hA : A = N * π * r^2 / 2) 
    (hB : B = (π * r^2 / 2) * (N^2 - N)) 
    (ratio : A / B = 1 / 18) : 
    N = 19 :=
by
  sorry

end find_N_l131_131828


namespace find_angle_and_perimeter_l131_131109

open Real

variables {A B C a b c : ℝ}

/-- If (2a - c)sinA + (2c - a)sinC = 2bsinB in triangle ABC -/
theorem find_angle_and_perimeter
  (h1 : (2 * a - c) * sin A + (2 * c - a) * sin C = 2 * b * sin B)
  (acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (b_eq : b = 1) :
  B = π / 3 ∧ (sqrt 3 + 1 < a + b + c ∧ a + b + c ≤ 3) :=
sorry

end find_angle_and_perimeter_l131_131109


namespace total_students_in_school_l131_131686

variable (TotalStudents : ℕ)
variable (num_students_8_years_old : ℕ := 48)
variable (percent_students_below_8 : ℝ := 0.20)
variable (num_students_above_8 : ℕ := (2 / 3) * num_students_8_years_old)

theorem total_students_in_school :
  percent_students_below_8 * TotalStudents + (num_students_8_years_old + num_students_above_8) = TotalStudents :=
by
  sorry

end total_students_in_school_l131_131686


namespace maria_scored_33_points_l131_131422

-- Defining constants and parameters
def num_shots := 40
def equal_distribution : ℕ := num_shots / 3 -- each type of shot

-- Given success rates
def success_rate_three_point : ℚ := 0.25
def success_rate_two_point : ℚ := 0.50
def success_rate_free_throw : ℚ := 0.80

-- Defining the points per successful shot
def points_per_successful_three_point_shot : ℕ := 3
def points_per_successful_two_point_shot : ℕ := 2
def points_per_successful_free_throw_shot : ℕ := 1

-- Calculating total points scored
def total_points_scored :=
  (success_rate_three_point * points_per_successful_three_point_shot * equal_distribution) +
  (success_rate_two_point * points_per_successful_two_point_shot * equal_distribution) +
  (success_rate_free_throw * points_per_successful_free_throw_shot * equal_distribution)

theorem maria_scored_33_points :
  total_points_scored = 33 := 
sorry

end maria_scored_33_points_l131_131422


namespace solution_set_of_fraction_inequality_l131_131263

theorem solution_set_of_fraction_inequality
  (a b : ℝ) (h₀ : ∀ x : ℝ, x > 1 → ax - b > 0) :
  {x : ℝ | (ax + b) / (x - 2) > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 2} :=
by
  sorry

end solution_set_of_fraction_inequality_l131_131263


namespace canoe_speed_downstream_l131_131347

theorem canoe_speed_downstream (V_upstream V_s V_c V_downstream : ℝ) 
    (h1 : V_upstream = 6) 
    (h2 : V_s = 2) 
    (h3 : V_upstream = V_c - V_s) 
    (h4 : V_downstream = V_c + V_s) : 
  V_downstream = 10 := 
by 
  sorry

end canoe_speed_downstream_l131_131347


namespace banana_pie_angle_l131_131100

theorem banana_pie_angle
  (total_students : ℕ := 48)
  (chocolate_students : ℕ := 15)
  (apple_students : ℕ := 10)
  (blueberry_students : ℕ := 9)
  (remaining_students := total_students - (chocolate_students + apple_students + blueberry_students))
  (banana_students := remaining_students / 2) :
  (banana_students : ℝ) / total_students * 360 = 52.5 :=
by
  sorry

end banana_pie_angle_l131_131100


namespace max_value_of_symmetric_function_l131_131951

noncomputable def f (x a b : ℝ) : ℝ := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function (a b : ℝ) :
  (∀ x : ℝ, f (-2 - x) a b = f (-2 + x) a b) → ∃ x : ℝ, ∀ y : ℝ, f x a b ≥ f y a b ∧ f x a b = 16 :=
sorry

end max_value_of_symmetric_function_l131_131951


namespace part1_part2_l131_131667

noncomputable def f (x : ℝ) : ℝ := |3 * x + 2|

theorem part1 (x : ℝ): f x < 6 - |x - 2| ↔ (-3/2 < x ∧ x < 1) :=
by sorry

theorem part2 (a : ℝ) (m n : ℝ) (h₁ : 0 < m) (h₂ : 0 < n) (h₃ : m + n = 4) (h₄ : 0 < a) (h₅ : ∀ x, |x - a| - f x ≤ 1/m + 1/n) :
    0 < a ∧ a ≤ 1/3 :=
by sorry

end part1_part2_l131_131667


namespace frank_money_remaining_l131_131935

-- Define the conditions
def cost_cheapest_lamp : ℕ := 20
def factor_most_expensive : ℕ := 3
def initial_money : ℕ := 90

-- Define the cost of the most expensive lamp
def cost_most_expensive_lamp : ℕ := cost_cheapest_lamp * factor_most_expensive

-- Define the money remaining after purchase
def money_remaining : ℕ := initial_money - cost_most_expensive_lamp

-- The theorem we need to prove
theorem frank_money_remaining : money_remaining = 30 := by
  sorry

end frank_money_remaining_l131_131935


namespace combined_resistance_parallel_l131_131022

theorem combined_resistance_parallel (R1 R2 R3 R : ℝ)
  (h1 : R1 = 2) (h2 : R2 = 5) (h3 : R3 = 6)
  (h4 : 1/R = 1/R1 + 1/R2 + 1/R3) :
  R = 15/13 := 
by
  sorry

end combined_resistance_parallel_l131_131022


namespace find_number_l131_131893

theorem find_number (x : ℝ) : 8050 * x = 80.5 → x = 0.01 :=
by
  sorry

end find_number_l131_131893


namespace salary_decrease_increase_l131_131848

theorem salary_decrease_increase (S : ℝ) (x : ℝ) (h : (S * (1 - x / 100) * (1 + x / 100) = 0.51 * S)) : x = 70 := 
by sorry

end salary_decrease_increase_l131_131848


namespace Nicky_wait_time_l131_131843

theorem Nicky_wait_time (x : ℕ) (h1 : x + (4 * x + 14) = 114) : x = 20 :=
by {
  sorry
}

end Nicky_wait_time_l131_131843


namespace age_difference_l131_131479

variable (A B C : ℕ)

theorem age_difference : A + B = B + C + 11 → A - C = 11 := by
  sorry

end age_difference_l131_131479


namespace sum_of_edges_of_rectangular_solid_l131_131608

theorem sum_of_edges_of_rectangular_solid 
(volume : ℝ) (surface_area : ℝ) (a b c : ℝ)
(h1 : volume = a * b * c)
(h2 : surface_area = 2 * (a * b + b * c + c * a))
(h3 : ∃ s : ℝ, s ≠ 0 ∧ a = b / s ∧ c = b * s)
(h4 : volume = 512)
(h5 : surface_area = 384) :
a + b + c = 24 := 
sorry

end sum_of_edges_of_rectangular_solid_l131_131608


namespace find_y_l131_131677

theorem find_y (x y : ℝ) : x - y = 8 ∧ x + y = 14 → y = 3 := by
  sorry

end find_y_l131_131677


namespace quadratic_smaller_solution_l131_131061

theorem quadratic_smaller_solution : ∀ (x : ℝ), x^2 - 9 * x + 20 = 0 → x = 4 ∨ x = 5 :=
by
  sorry

end quadratic_smaller_solution_l131_131061


namespace slope_of_line_l131_131160

theorem slope_of_line (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (1, 2)) (h2 : (x2, y2) = (4, 8)) :
  (y2 - y1) / (x2 - x1) = 2 := 
by
  sorry

end slope_of_line_l131_131160


namespace notecard_area_new_dimension_l131_131434

theorem notecard_area_new_dimension :
  ∀ (length : ℕ) (width : ℕ) (shortened : ℕ),
    length = 7 →
    width = 5 →
    shortened = 2 →
    (width - shortened) * length = 21 →
    (length - shortened) * (width - shortened + shortened) = 25 :=
by
  intros length width shortened h_length h_width h_shortened h_area
  sorry

end notecard_area_new_dimension_l131_131434


namespace emma_average_speed_last_segment_l131_131775

open Real

theorem emma_average_speed_last_segment :
  ∀ (d1 d2 d3 : ℝ) (t1 t2 t3 : ℝ),
    d1 + d2 + d3 = 120 →
    t1 + t2 + t3 = 2 →
    t1 = 2 / 3 → t2 = 2 / 3 → 
    t1 = d1 / 50 → t2 = d2 / 55 → 
    ∃ x : ℝ, t3 = d3 / x ∧ x = 75 := 
by
  intros d1 d2 d3 t1 t2 t3 h1 h2 ht1 ht2 hs1 hs2
  use 75 / (2 / 3)
  -- skipped proof for simplicity
  sorry

end emma_average_speed_last_segment_l131_131775


namespace number_of_roots_l131_131659

noncomputable def roots_eq (a : ℝ) (ha : 0 < a ∧ a < real.exp (-1)) : ℕ := 
  let f : ℂ → ℂ := λ z, z^2
  let phi : ℂ → ℂ := λ z, -a * complex.exp z
  if h : ∀ z, complex.abs z = 1 → complex.abs (f z) > complex.abs (phi z)
  then 2
  else 0

theorem number_of_roots (a : ℝ) (ha : 0 < a ∧ a < real.exp (-1)) : roots_eq a ha = 2 := by
  sorry

end number_of_roots_l131_131659


namespace ab_ac_plus_bc_range_l131_131116

theorem ab_ac_plus_bc_range (a b c : ℝ) (h : a + b + 2 * c = 0) :
  ∃ (k : ℝ), k ≤ 0 ∧ k = ab + ac + bc :=
sorry

end ab_ac_plus_bc_range_l131_131116


namespace earnings_correct_l131_131363

-- Define the initial number of roses, the number of roses left, and the price per rose.
def initial_roses : ℕ := 13
def roses_left : ℕ := 4
def price_per_rose : ℕ := 4

-- Calculate the number of roses sold.
def roses_sold : ℕ := initial_roses - roses_left

-- Calculate the total earnings.
def earnings : ℕ := roses_sold * price_per_rose

-- Prove that the earnings are 36 dollars.
theorem earnings_correct : earnings = 36 := by
  sorry

end earnings_correct_l131_131363


namespace smallest_rel_prime_to_180_l131_131238

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, (y > 1 ∧ Nat.gcd y 180 = 1) → x ≤ y :=
begin
  sorry
end

end smallest_rel_prime_to_180_l131_131238


namespace part1_part2_l131_131389

-- Part (1): Proving the range of x when a = 1
theorem part1 (x : ℝ) : (x^2 - 6 * 1 * x + 8 < 0) ∧ (x^2 - 4 * x + 3 ≤ 0) ↔ 2 < x ∧ x ≤ 3 := 
by sorry

-- Part (2): Proving the range of a when p is a sufficient but not necessary condition for q
theorem part2 (a : ℝ) : (∀ x : ℝ, (x^2 - 6 * a * x + 8 * a^2 < 0 → x^2 - 4 * x + 3 ≤ 0) 
  ∧ (∃ x : ℝ, x^2 - 4 * x + 3 ≤ 0 ∧ x^2 - 6 * a * x + 8 * a^2 ≥ 0)) ↔ (1/2 ≤ a ∧ a ≤ 3/4) :=
by sorry

end part1_part2_l131_131389


namespace find_roots_l131_131060

theorem find_roots (x : ℝ) (h : 21 / (x^2 - 9) - 3 / (x - 3) = 1) : x = -3 ∨ x = 7 :=
by {
  sorry
}

end find_roots_l131_131060


namespace state_a_selection_percentage_l131_131424

-- Definitions based on the conditions
variables {P : ℕ} -- percentage of candidates selected in State A

theorem state_a_selection_percentage 
  (candidates : ℕ) 
  (state_b_percentage : ℕ) 
  (extra_selected_in_b : ℕ) 
  (total_selected_in_b : ℕ) 
  (total_selected_in_a : ℕ)
  (appeared_in_each_state : ℕ) 
  (H1 : appeared_in_each_state = 8200)
  (H2 : state_b_percentage = 7)
  (H3 : extra_selected_in_b = 82)
  (H4 : total_selected_in_b = (state_b_percentage * appeared_in_each_state) / 100)
  (H5 : total_selected_in_a = total_selected_in_b - extra_selected_in_b)
  (H6 : total_selected_in_a = (P * appeared_in_each_state) / 100)
  : P = 6 :=
by {
  sorry
}

end state_a_selection_percentage_l131_131424


namespace a_wins_by_200_meters_l131_131819

-- Define the conditions
def race_distance : ℕ := 600
def speed_ratio_a_to_b := 5 / 4
def head_start_a : ℕ := 100

-- Define the proof statement
theorem a_wins_by_200_meters (x : ℝ) (ha_speed : ℝ := 5 * x) (hb_speed : ℝ := 4 * x)
  (ha_distance_to_win : ℝ := race_distance - head_start_a) :
  (ha_distance_to_win / ha_speed) = (400 / hb_speed) → 
  600 - (400) = 200 :=
by
  -- For now, skip the proof, focus on the statement.
  sorry

end a_wins_by_200_meters_l131_131819


namespace hyperbola_properties_l131_131603

theorem hyperbola_properties (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (c := Real.sqrt (a^2 + b^2))
  (F2 := (c, 0)) (P : ℝ × ℝ)
  (h_perpendicular : ∃ (x y : ℝ), P = (x, y) ∧ y = -a/b * (x - c))
  (h_distance : Real.sqrt ((P.1 - c)^2 + P.2^2) = 2)
  (h_slope : P.2 / (P.1 - c) = -1/2) :
  
  b = 2 ∧
  (∀ x y, x^2 - y^2 / 4 = 1 ↔ x^2 - y^2 / b^2 = 1) ∧
  P = (Real.sqrt (5) / 5, 2 * Real.sqrt (5) / 5) :=
sorry

end hyperbola_properties_l131_131603


namespace radar_placement_and_coverage_area_l131_131460

theorem radar_placement_and_coverage_area (r : ℝ) (w : ℝ) (n : ℕ) (h_radars : n = 5) (h_radius : r = 13) (h_width : w = 10) :
  let max_dist := 12 / Real.sin (Real.pi / 5)
  let area_ring := (240 * Real.pi) / Real.tan (Real.pi / 5)
  max_dist = 12 / Real.sin (Real.pi / 5) ∧ area_ring = (240 * Real.pi) / Real.tan (Real.pi / 5) :=
by
  sorry

end radar_placement_and_coverage_area_l131_131460


namespace ptarmigan_environmental_capacity_l131_131844

theorem ptarmigan_environmental_capacity (predators_eradicated : Prop) (mass_deaths : Prop) : 
  (after_predator_eradication : predators_eradicated → mass_deaths) →
  (environmental_capacity_increased : Prop) → environmental_capacity_increased :=
by
  intros h1 h2
  sorry

end ptarmigan_environmental_capacity_l131_131844


namespace cost_per_component_l131_131750

theorem cost_per_component (C : ℝ) : 
  (150 * C + 150 * 4 + 16500 = 150 * 193.33) → 
  C = 79.33 := 
by
  intro h
  sorry

end cost_per_component_l131_131750


namespace percentage_of_seniors_is_90_l131_131436

-- Definitions of the given conditions
def total_students : ℕ := 120
def students_in_statistics : ℕ := total_students / 2
def seniors_in_statistics : ℕ := 54

-- Statement to prove
theorem percentage_of_seniors_is_90 : 
  ( seniors_in_statistics / students_in_statistics : ℚ ) * 100 = 90 := 
by
  sorry  -- Proof will be provided here.

end percentage_of_seniors_is_90_l131_131436


namespace probability_solution_l131_131973

noncomputable def binom_10_7 := Nat.choose 10 7
noncomputable def binom_10_6 := Nat.choose 10 6

theorem probability_solution (p q : ℝ) (h₁ : q = 1 - p) (h₂ : binom_10_7 = 120) (h₃ : binom_10_6 = 210)
  (h₄ : 120 * p ^ 7 * q ^ 3 = 210 * p ^ 6 * q ^ 4) : p = 7 / 11 := 
sorry

end probability_solution_l131_131973


namespace negation_of_universal_proposition_l131_131447

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by
  sorry

end negation_of_universal_proposition_l131_131447


namespace sequence_an_l131_131576

theorem sequence_an (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * (a n - 1)) : a 2 = 4 := 
by
  sorry

end sequence_an_l131_131576


namespace letter_puzzle_l131_131222

theorem letter_puzzle (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (h_diff : A ≠ B) :
  A^B = 10 * B + A ↔ (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_l131_131222


namespace two_bacteria_fill_time_l131_131846

-- Define the conditions
def one_bacterium_fills_bottle_in (a : Nat) (t : Nat) : Prop :=
  (2^t = 2^a)

def two_bacteria_fill_bottle_in (a : Nat) (x : Nat) : Prop :=
  (2 * 2^x = 2^a)

-- State the theorem
theorem two_bacteria_fill_time (a : Nat) : ∃ x, two_bacteria_fill_bottle_in a x ∧ x = a - 1 :=
by
  -- Use the given conditions
  sorry

end two_bacteria_fill_time_l131_131846


namespace subcommittee_has_teacher_l131_131142

def total_combinations (n k : ℕ) : ℕ := Nat.choose n k

def teacher_subcommittee_count : ℕ := total_combinations 12 5 - total_combinations 7 5

theorem subcommittee_has_teacher : teacher_subcommittee_count = 771 := 
by
  sorry

end subcommittee_has_teacher_l131_131142


namespace quadratic_properties_l131_131034

theorem quadratic_properties (d e f : ℝ)
  (h1 : d * 1^2 + e * 1 + f = 3)
  (h2 : d * 2^2 + e * 2 + f = 0)
  (h3 : d * 9 + e * 3 + f = -3) :
  d + e + 2 * f = 19.5 :=
sorry

end quadratic_properties_l131_131034


namespace total_value_of_item_l131_131344

theorem total_value_of_item
  (import_tax : ℝ)
  (V : ℝ)
  (h₀ : import_tax = 110.60)
  (h₁ : import_tax = 0.07 * (V - 1000)) :
  V = 2579.43 := 
sorry

end total_value_of_item_l131_131344


namespace repeatingDecimal_as_fraction_l131_131272

def repeatingDecimal : ℚ := 0.136513513513

theorem repeatingDecimal_as_fraction : repeatingDecimal = 136377 / 999000 := 
by 
  sorry

end repeatingDecimal_as_fraction_l131_131272


namespace roja_alone_time_l131_131979

theorem roja_alone_time (W : ℝ) (R : ℝ) :
  (1 / 60 + 1 / R = 1 / 35) → (R = 210) :=
by
  intros
  -- Proof goes here
  sorry

end roja_alone_time_l131_131979


namespace range_of_m_l131_131259

theorem range_of_m (m : ℝ) (h : 1 < m) :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 1 → -m ≤ x ∧ x ≤ m - 1) → (3 ≤ m) :=
by
  sorry  -- The proof will be constructed here.

end range_of_m_l131_131259


namespace nba_conference_division_impossible_l131_131729

theorem nba_conference_division_impossible :
  let teams := 30
  let games_per_team := 82
  let total_games := teams * games_per_team
  let unique_games := total_games / 2
  let inter_conference_games := unique_games / 2
  ¬∃ (A B : ℕ), A + B = teams ∧ A * B = inter_conference_games := 
by
  let teams := 30
  let games_per_team := 82
  let total_games := teams * games_per_team
  let unique_games := total_games / 2
  let inter_conference_games := unique_games / 2
  sorry

end nba_conference_division_impossible_l131_131729


namespace number_of_ways_to_fill_l131_131360

-- Definitions and conditions
def triangular_array (row : ℕ) (col : ℕ) : Prop :=
  -- Placeholder definition for the triangular array structure
  sorry 

def sum_based (row : ℕ) (col : ℕ) : Prop :=
  -- Placeholder definition for the sum-based condition
  sorry 

def valid_filling (x : Fin 13 → ℕ) :=
  (∀ i, x i = 0 ∨ x i = 1) ∧
  (x 0 + x 12) % 5 = 0

theorem number_of_ways_to_fill (x : Fin 13 → ℕ) :
  triangular_array 13 1 → sum_based 13 1 →
  valid_filling x → 
  (∃ (count : ℕ), count = 4096) :=
sorry

end number_of_ways_to_fill_l131_131360


namespace sum_of_factorials_is_perfect_square_l131_131784

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_factorials (n : ℕ) : ℕ :=
  (List.range (n + 1)).map factorial |>.sum

theorem sum_of_factorials_is_perfect_square (n : ℕ) (h : n > 0) :
  (∃ m : ℕ, m * m = sum_of_factorials n) ↔ (n = 1 ∨ n = 3) := 
sorry

end sum_of_factorials_is_perfect_square_l131_131784


namespace chi_square_test_not_reject_l131_131037

theorem chi_square_test_not_reject 
  (n : ℕ) (s2 : ℝ) (sigma0_2 : ℝ) (alpha : ℝ) (k : ℕ) (chi2_crit : ℝ)
  (h_n : n = 21)
  (h_s2 : s2 = 16.2)
  (h_sigma0_2 : sigma0_2 = 15)
  (h_alpha : alpha = 0.01)
  (h_k : k = 20)
  (h_chi2crit : chi2_crit = 37.6) :
  ((n - 1 : ℝ) * s2) / sigma0_2 < chi2_crit :=
by
  -- The actual proof would go here.
  sorry

end chi_square_test_not_reject_l131_131037


namespace sum_of_15th_set_l131_131203

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

end sum_of_15th_set_l131_131203


namespace fraction_inequality_solution_l131_131141

theorem fraction_inequality_solution (x : ℝ) :
  -1 ≤ x ∧ x ≤ 3 ∧ (4 * x + 3 > 2 * (8 - 3 * x)) → (13 / 10) < x ∧ x ≤ 3 :=
by
  sorry

end fraction_inequality_solution_l131_131141


namespace permutation_20th_l131_131328

noncomputable theory
open List

def is_20th_permutation (n : ℕ) : Prop :=
  let digits := [1, 2, 3, 4, 5]
  let permutations := permutations digits
  permutations.get! (n - 1) = [1, 2, 5, 4, 3]

theorem permutation_20th : is_20th_permutation 20 :=
by
  sorry

end permutation_20th_l131_131328


namespace math_problem_l131_131355

theorem math_problem
  (N O : ℝ)
  (h₁ : 96 / 100 = |(O - 5 * N) / (5 * N)|)
  (h₂ : 5 * N ≠ 0) :
  O = 0.2 * N :=
by
  sorry

end math_problem_l131_131355


namespace smallest_rel_prime_to_180_l131_131251

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  { intro y,
    intro h,
    cases h with h1 h2,
    repeat { try { apply dec_trivial,
                   apply lt_or_eq_of_le,
                   norm_num,
                   apply Nat.prime_not_dvd_mul,
                   norm_num,
                   apply not_or_distrib.mpr,
                   split,
                   norm_cast,
                   intro,
                   exact le_antisymm _ },
           sorry }
end

end smallest_rel_prime_to_180_l131_131251


namespace number_of_girls_l131_131144

variable (g b : ℕ) -- Number of girls (g) and boys (b) in the class
variable (h_ratio : g / b = 4 / 3) -- The ratio condition
variable (h_total : g + b = 63) -- The total number of students condition

theorem number_of_girls (g b : ℕ) (h_ratio : g / b = 4 / 3) (h_total : g + b = 63) :
    g = 36 :=
sorry

end number_of_girls_l131_131144


namespace degree_at_least_three_l131_131833

noncomputable def p : Polynomial ℤ := sorry
noncomputable def q : Polynomial ℤ := sorry

theorem degree_at_least_three (h1 : p.degree ≥ 1)
                              (h2 : q.degree ≥ 1)
                              (h3 : (∃ xs : Fin 33 → ℤ, ∀ i, p.eval (xs i) * q.eval (xs i) - 2015 = 0)) :
  p.degree ≥ 3 ∧ q.degree ≥ 3 := 
sorry

end degree_at_least_three_l131_131833


namespace simplify_expression_l131_131629

variable (a : ℝ) (ha : a ≠ -3)

theorem simplify_expression : (a^2) / (a + 3) - 9 / (a + 3) = a - 3 :=
by
  sorry

end simplify_expression_l131_131629


namespace find_CP_A_l131_131756

noncomputable def CP_A : Float := 173.41
def SP_B (CP_A : Float) : Float := 1.20 * CP_A
def SP_C (SP_B : Float) : Float := 1.25 * SP_B
def TC_C (SP_C : Float) : Float := 1.15 * SP_C
def SP_D1 (TC_C : Float) : Float := 1.30 * TC_C
def SP_D2 (SP_D1 : Float) : Float := 0.90 * SP_D1
def SP_D2_actual : Float := 350

theorem find_CP_A : 
  (SP_D2 (SP_D1 (TC_C (SP_C (SP_B CP_A))))) = SP_D2_actual → 
  CP_A = 173.41 := sorry

end find_CP_A_l131_131756


namespace fraction_product_l131_131767

theorem fraction_product : (1/2) * (3/5) * (7/11) * (4/13) = 84/1430 := by
  sorry

end fraction_product_l131_131767


namespace solve_the_problem_l131_131105

noncomputable def solve_problem : Prop :=
  ∀ (θ t α : ℝ),
    (∃ x y : ℝ, x = 2 * Real.cos θ ∧ y = 4 * Real.sin θ) → 
    (∃ x y : ℝ, x = 1 + t * Real.cos α ∧ y = 2 + t * Real.sin α) →
    (∃ m n : ℝ, m = 1 ∧ n = 2) →
    (-2 = Real.tan α)

theorem solve_the_problem : solve_problem := by
  sorry

end solve_the_problem_l131_131105


namespace find_abs_sum_roots_l131_131257

noncomputable def polynomial_root_abs_sum (n p q r : ℤ) : Prop :=
(p + q + r = 0) ∧
(p * q + q * r + r * p = -2009) ∧
(p * q * r = -n) →
(|p| + |q| + |r| = 102)

theorem find_abs_sum_roots (n p q r : ℤ) :
  polynomial_root_abs_sum n p q r :=
sorry

end find_abs_sum_roots_l131_131257


namespace find_a_for_tangent_l131_131948

theorem find_a_for_tangent (a : ℤ) (x : ℝ) (h : ∀ x, 3*x^2 - 4*a*x + 2*a > 0) : a = 1 :=
sorry

end find_a_for_tangent_l131_131948


namespace tim_weekly_earnings_l131_131612

-- Define the constants based on the conditions
def tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def days_per_week : ℕ := 6

-- Statement to prove Tim's weekly earnings
theorem tim_weekly_earnings : tasks_per_day * pay_per_task * days_per_week = 720 := by
  sorry

end tim_weekly_earnings_l131_131612


namespace probability_of_350_germinating_out_of_400_l131_131867

noncomputable def germination_probability 
  (n : ℝ) (p : ℝ) (k : ℝ) : ℝ :=
  let q := 1 - p in
  let sqrt_npq := real.sqrt (n * p * q) in
  let x := (k - n * p) / sqrt_npq in
  let phi_x := 1 / (real.sqrt (2 * real.pi)) * real.exp (-(x^2) / 2) in
  phi_x / sqrt_npq

theorem probability_of_350_germinating_out_of_400 :
  germination_probability 400 0.9 350 ≈ 0.0167 := 
by sorry

end probability_of_350_germinating_out_of_400_l131_131867


namespace goods_train_speed_l131_131174

noncomputable def speed_of_goods_train (train_speed : ℝ) (goods_length : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed_mps := goods_length / passing_time
  let relative_speed_kmph := relative_speed_mps * 3.6
  (relative_speed_kmph - train_speed)

theorem goods_train_speed :
  speed_of_goods_train 30 280 9 = 82 :=
by
  sorry

end goods_train_speed_l131_131174


namespace intersection_A_B_l131_131943

open Set

def universal_set : Set ℕ := {0, 1, 3, 5, 7, 9}
def complement_A : Set ℕ := {0, 5, 9}
def B : Set ℕ := {3, 5, 7}
def A : Set ℕ := universal_set \ complement_A

theorem intersection_A_B :
  A ∩ B = {3, 7} :=
by
  sorry

end intersection_A_B_l131_131943


namespace last_number_nth_row_sum_of_nth_row_position_of_2008_l131_131127

theorem last_number_nth_row (n : ℕ) : 
  ∃ last_number, last_number = 2^n - 1 := by
  sorry

theorem sum_of_nth_row (n : ℕ) : 
  ∃ sum_nth_row, sum_nth_row = 2^(2*n-2) + 2^(2*n-3) - 2^(n-2) := by
  sorry

theorem position_of_2008 : 
  ∃ (row : ℕ) (position : ℕ), row = 11 ∧ position = 2008 - 2^10 + 1 :=
  by sorry

end last_number_nth_row_sum_of_nth_row_position_of_2008_l131_131127


namespace midpoint_coords_product_l131_131881

def midpoint_prod (x1 y1 x2 y2 : ℤ) : ℤ :=
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  mx * my

theorem midpoint_coords_product :
  midpoint_prod 4 (-7) (-8) 9 = -2 := by
  sorry

end midpoint_coords_product_l131_131881


namespace evaluate_expression_l131_131657

theorem evaluate_expression (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) : 2 * x ^ y + 5 * y ^ x - z ^ 2 = 42 :=
by
  sorry

end evaluate_expression_l131_131657


namespace initial_minutes_planA_equivalence_l131_131030

-- Conditions translated into Lean:
variable (x : ℝ)

-- Definitions for costs
def planA_cost_12 : ℝ := 0.60 + 0.06 * (12 - x)
def planB_cost_12 : ℝ := 0.08 * 12

-- Theorem we want to prove
theorem initial_minutes_planA_equivalence :
  (planA_cost_12 x = planB_cost_12) → x = 6 :=
by
  intro h
  -- complete proof is skipped with sorry
  sorry

end initial_minutes_planA_equivalence_l131_131030


namespace bleaching_takes_3_hours_l131_131502

-- Define the total time and the relationship between dyeing and bleaching.
def total_time : ℕ := 9
def dyeing_takes_twice (H : ℕ) : Prop := 2 * H + H = total_time

-- Prove that bleaching takes 3 hours.
theorem bleaching_takes_3_hours : ∃ H : ℕ, dyeing_takes_twice H ∧ H = 3 := 
by 
  sorry

end bleaching_takes_3_hours_l131_131502


namespace correct_option_d_l131_131162

-- Define variables and constants.
variable (a : ℝ)

-- State the conditions as definitions.
def optionA := a^2 * a^3 = a^5
def optionB := (3 * a)^2 = 9 * a^2
def optionC := a^6 / a^3 = a^3
def optionD := 3 * a^2 - a^2 = 2 * a^2

-- The theorem states that the correct option is D.
theorem correct_option_d : optionD := by
  sorry

end correct_option_d_l131_131162


namespace trajectory_of_P_l131_131074
-- Import entire library for necessary definitions and theorems.

-- Define the properties of the conic sections.
def ellipse (x y : ℝ) (n : ℝ) : Prop :=
  x^2 / 4 + y^2 / n = 1

def hyperbola (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / 8 - y^2 / m = 1

-- Define the condition where the conic sections share the same foci.
def shared_foci (n m : ℝ) : Prop :=
  4 - n = 8 + m

-- The main theorem stating the relationship between m and n forming a straight line.
theorem trajectory_of_P : ∀ (n m : ℝ), shared_foci n m → (m + n + 4 = 0) :=
by
  intros n m h
  sorry

end trajectory_of_P_l131_131074


namespace input_equals_output_l131_131795

theorem input_equals_output (x : ℝ) :
  (x ≤ 1 → 2 * x - 3 = x) ∨ (x > 1 → x^2 - 3 * x + 3 = x) ↔ x = 3 :=
by
  sorry

end input_equals_output_l131_131795


namespace charge_y1_charge_y2_cost_effective_range_call_duration_difference_l131_131425

def y1 (x : ℕ) : ℝ :=
  if x ≤ 600 then 30 else 0.1 * x - 30

def y2 (x : ℕ) : ℝ :=
  if x ≤ 1200 then 50 else 0.1 * x - 70

theorem charge_y1 (x : ℕ) :
  (x ≤ 600 → y1 x = 30) ∧ (x > 600 → y1 x = 0.1 * x - 30) :=
by sorry

theorem charge_y2 (x : ℕ) :
  (x ≤ 1200 → y2 x = 50) ∧ (x > 1200 → y2 x = 0.1 * x - 70) :=
by sorry

theorem cost_effective_range (x : ℕ) :
  (0 ≤ x) ∧ (x < 800) → y1 x < y2 x :=
by sorry

noncomputable def call_time_xiaoming : ℕ := 1300
noncomputable def call_time_xiaohua : ℕ := 900

theorem call_duration_difference :
  call_time_xiaoming = call_time_xiaohua + 400 :=
by sorry

end charge_y1_charge_y2_cost_effective_range_call_duration_difference_l131_131425


namespace sum_of_center_coordinates_l131_131448

theorem sum_of_center_coordinates (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 7) (h2 : y1 = -6) (h3 : x2 = -5) (h4 : y2 = 4) :
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  midpoint_x + midpoint_y = 0 := by
  -- Definitions and setup
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  sorry

end sum_of_center_coordinates_l131_131448


namespace intersection_points_of_line_l131_131444

theorem intersection_points_of_line (x y : ℝ) :
  ((y = 2 * x - 1) → (y = 0 → x = 0.5)) ∧
  ((y = 2 * x - 1) → (x = 0 → y = -1)) :=
by sorry

end intersection_points_of_line_l131_131444


namespace remainder_six_n_mod_four_l131_131338

theorem remainder_six_n_mod_four (n : ℤ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := 
by sorry

end remainder_six_n_mod_four_l131_131338


namespace no_quad_term_l131_131878

theorem no_quad_term (x m : ℝ) : 
  (2 * x^2 - 2 * (7 + 3 * x - 2 * x^2) + m * x^2) = -6 * x - 14 → m = -6 := 
by 
  sorry

end no_quad_term_l131_131878


namespace count_perfect_squares_l131_131545

theorem count_perfect_squares (N : Nat) :
  ∃ k : Nat, k = 1666 ∧ ∀ m, (∃ n, m = n * n ∧ m < 10^8 ∧ 36 ∣ m) ↔ (m = 36 * k ^ 2 ∧ k < 10^4) :=
sorry

end count_perfect_squares_l131_131545


namespace sector_radius_cone_l131_131733

theorem sector_radius_cone {θ R r : ℝ} (sector_angle : θ = 120) (cone_base_radius : r = 2) :
  (R * θ / 360) * 2 * π = 2 * π * r → R = 6 :=
by
  intros h
  sorry

end sector_radius_cone_l131_131733


namespace smallest_n_for_abc_factorials_l131_131853

theorem smallest_n_for_abc_factorials (a b c : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a + b + c = 2006) :
  ∃ m n : ℕ, (¬ ∃ k : ℕ, m = 10 * k) ∧ a.factorial * b.factorial * c.factorial = m * 10^n ∧ n = 492 :=
sorry

end smallest_n_for_abc_factorials_l131_131853


namespace rectangle_area_function_relationship_l131_131594

theorem rectangle_area_function_relationship (x y : ℝ) (h : x * y = 12) : y = 12 / x :=
by
  sorry

end rectangle_area_function_relationship_l131_131594


namespace smallest_number_of_eggs_l131_131012

-- Define the conditions given in the problem
def total_containers (c : ℕ) : ℕ := 15 * c - 3

-- Prove that given the conditions, the smallest number of eggs you could have is 162
theorem smallest_number_of_eggs (h : ∃ c : ℕ, total_containers c > 150) : ∃ c : ℕ, total_containers c = 162 :=
by
  sorry

end smallest_number_of_eggs_l131_131012


namespace drummer_difference_l131_131146

def flute_players : Nat := 5
def trumpet_players : Nat := 3 * flute_players
def trombone_players : Nat := trumpet_players - 8
def clarinet_players : Nat := 2 * flute_players
def french_horn_players : Nat := trombone_players + 3
def total_seats_needed : Nat := 65
def total_seats_taken : Nat := flute_players + trumpet_players + trombone_players + clarinet_players + french_horn_players
def drummers : Nat := total_seats_needed - total_seats_taken

theorem drummer_difference : drummers - trombone_players = 11 := by
  sorry

end drummer_difference_l131_131146


namespace solution_set_of_inequality_l131_131147

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 5 * x - 14 < 0} = {x : ℝ | -2 < x ∧ x < 7} :=
by
  sorry

end solution_set_of_inequality_l131_131147


namespace find_divisor_of_115_l131_131927

theorem find_divisor_of_115 (x : ℤ) (N : ℤ)
  (hN : N = 115)
  (h1 : N % 38 = 1)
  (h2 : N % x = 1) :
  x = 57 :=
by
  sorry

end find_divisor_of_115_l131_131927


namespace ordered_pairs_unique_solution_l131_131273

theorem ordered_pairs_unique_solution :
  ∃! (b c : ℕ), (b > 0) ∧ (c > 0) ∧ (b^2 - 4 * c = 0) ∧ (c^2 - 4 * b = 0) :=
sorry

end ordered_pairs_unique_solution_l131_131273


namespace true_proposition_among_choices_l131_131092

theorem true_proposition_among_choices (p q : Prop) (hp : p) (hq : ¬ q) :
  p ∧ ¬ q :=
by
  sorry

end true_proposition_among_choices_l131_131092


namespace graph_symmetric_l131_131532

noncomputable def f (x : ℝ) : ℝ := sorry

theorem graph_symmetric (f : ℝ → ℝ) :
  (∀ x y, y = f x ↔ (∃ y₁, y₁ = f (2 - x) ∧ y = - (1 / (y₁ + 1)))) →
  ∀ x, f x = 1 / (x - 3) := 
by
  intro h x
  sorry

end graph_symmetric_l131_131532


namespace rows_of_seats_l131_131902

theorem rows_of_seats (students sections_per_row students_per_section : ℕ) (h1 : students_per_section = 2) (h2 : sections_per_row = 2) (h3 : students = 52) :
  (students / students_per_section / sections_per_row) = 13 :=
sorry

end rows_of_seats_l131_131902


namespace part1_part2_l131_131086

open Classical

theorem part1 (x : ℝ) (a : ℝ) (b : ℝ) :
  (a = 1) ∧ (b = 2) ∧ (∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :=
by
  sorry

theorem part2 (x y k : ℝ) (a b : ℝ) :
  a = 1 ∧ b = 2 ∧ (x > 0) ∧ (y > 0) ∧ (1 / x + 2 / y = 1) ∧ (2 * x + y ≥ k^2 + k + 2) → -3 ≤ k ∧ k ≤ 2 :=
by
  sorry

end part1_part2_l131_131086


namespace range_of_x_range_of_a_l131_131390

-- Part (1): 
theorem range_of_x (x : ℝ) : 
  (a = 1) → (x^2 - 6 * a * x + 8 * a^2 < 0) → (x^2 - 4 * x + 3 ≤ 0) → (2 < x ∧ x ≤ 3) := sorry

-- Part (2):
theorem range_of_a (a : ℝ) : 
  (a ≠ 0) → (∀ x, (x^2 - 4 * x + 3 ≤ 0) → (x^2 - 6 * a * x + 8 * a^2 < 0)) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 4) := sorry

end range_of_x_range_of_a_l131_131390


namespace percent_of_pizza_not_crust_l131_131177

theorem percent_of_pizza_not_crust (total_weight crust_weight : ℝ) (h_total : total_weight = 800) (h_crust : crust_weight = 200) :
  (total_weight - crust_weight) / total_weight * 100 = 75 :=
by
  sorry

end percent_of_pizza_not_crust_l131_131177


namespace positive_perfect_squares_multiples_of_36_lt_10_pow_8_l131_131550

theorem positive_perfect_squares_multiples_of_36_lt_10_pow_8 :
  ∃ (count : ℕ), count = 1666 ∧ 
    (∀ n : ℕ, (n * n < 10^8 → n % 6 = 0) ↔ (n < 10^4 ∧ n % 6 = 0)) :=
sorry

end positive_perfect_squares_multiples_of_36_lt_10_pow_8_l131_131550


namespace letter_puzzle_l131_131221

theorem letter_puzzle (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (h_diff : A ≠ B) :
  A^B = 10 * B + A ↔ (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_l131_131221


namespace value_of_x_l131_131719

theorem value_of_x 
    (r : ℝ) (a : ℝ) (x : ℝ) (shaded_area : ℝ)
    (h1 : r = 2)
    (h2 : a = 2)
    (h3 : shaded_area = 2) :
  x = (Real.pi / 3) + (Real.sqrt 3 / 2) - 1 :=
sorry

end value_of_x_l131_131719


namespace ladder_slip_l131_131749

theorem ladder_slip (l : ℝ) (d1 d2 : ℝ) (h1 h2 : ℝ) :
  l = 30 → d1 = 8 → h1^2 + d1^2 = l^2 → h2 = h1 - 4 → 
  (h2^2 + (d1 + d2)^2 = l^2) → d2 = 2 :=
by
  intros h_l h_d1 h_h1_eq h_h2 h2_eq_l   
  sorry

end ladder_slip_l131_131749


namespace variance_of_set_l131_131529

theorem variance_of_set (x : ℝ) (h : (-1 + x + 0 + 1 - 1)/5 = 0) : 
  (1/5) * ( (-1)^2 + (x)^2 + 0^2 + 1^2 + (-1)^2 ) = 0.8 :=
by
  -- placeholder for the proof
  sorry

end variance_of_set_l131_131529


namespace regular_polygon_sides_l131_131859

-- Define the central angle and number of sides of a regular polygon
def central_angle (θ : ℝ) := θ = 30
def number_of_sides (n : ℝ) := 360 / 30 = n

-- Theorem to prove that the number of sides of the regular polygon is 12 given the central angle is 30 degrees
theorem regular_polygon_sides (θ n : ℝ) (hθ : central_angle θ) : number_of_sides n → n = 12 :=
sorry

end regular_polygon_sides_l131_131859


namespace find_n_l131_131628

theorem find_n (n : ℕ) : 2^(2 * n) + 2^(2 * n) + 2^(2 * n) + 2^(2 * n) = 4^22 → n = 21 :=
by
  sorry

end find_n_l131_131628


namespace donut_selection_l131_131584

-- Lean statement for the proof problem
theorem donut_selection (n k : ℕ) (h1 : n = 5) (h2 : k = 4) : (n + k - 1).choose (k - 1) = 56 :=
by
  rw [h1, h2]
  sorry

end donut_selection_l131_131584


namespace negation_of_existential_statement_l131_131140

theorem negation_of_existential_statement (x : ℚ) :
  ¬ (∃ x : ℚ, x^2 = 3) ↔ ∀ x : ℚ, x^2 ≠ 3 :=
by sorry

end negation_of_existential_statement_l131_131140


namespace range_of_a_satisfies_l131_131791

noncomputable def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-(x + 1)) = -f (x + 1)) ∧
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ 0 ≤ x2 → (f x1 - f x2) / (x1 - x2) > -1) ∧
  (∀ a : ℝ, f (a^2 - 1) + f (a - 1) + a^2 + a > 2)

theorem range_of_a_satisfies (f : ℝ → ℝ) (hf_conditions : satisfies_conditions f) :
  {a : ℝ | f (a^2 - 1) + f (a - 1) + a^2 + a > 2} = {a | a < -2 ∨ a > 1} :=
by
  sorry

end range_of_a_satisfies_l131_131791


namespace triangle_side_lengths_l131_131562

noncomputable def side_lengths (a b c : ℝ) : Prop :=
  a = 10 ∧ (a^2 + b^2 + c^2 = 2050) ∧ (c^2 = a^2 + b^2)

theorem triangle_side_lengths :
  ∃ b c : ℝ, side_lengths 10 b c ∧ b = Real.sqrt 925 ∧ c = Real.sqrt 1025 :=
by
  sorry

end triangle_side_lengths_l131_131562


namespace absolute_value_bound_l131_131623

theorem absolute_value_bound (x : ℝ) (hx : |x| ≤ 2) : |3 * x - x^3| ≤ 2 := 
by
  sorry

end absolute_value_bound_l131_131623


namespace negative_movement_south_l131_131809

noncomputable def movement_interpretation (x : ℤ) : String :=
if x > 0 then 
  "moving " ++ toString x ++ "m north"
else 
  "moving " ++ toString (-x) ++ "m south"

theorem negative_movement_south : movement_interpretation (-50) = "moving 50m south" := 
by 
  sorry

end negative_movement_south_l131_131809


namespace factorial_div_l131_131002

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_div : (factorial 4) / (factorial (4 - 3)) = 24 := by
  sorry

end factorial_div_l131_131002


namespace batsman_percentage_running_between_wickets_l131_131168

def boundaries : Nat := 6
def runs_per_boundary : Nat := 4
def sixes : Nat := 4
def runs_per_six : Nat := 6
def no_balls : Nat := 8
def runs_per_no_ball : Nat := 1
def wide_balls : Nat := 5
def runs_per_wide_ball : Nat := 1
def leg_byes : Nat := 2
def runs_per_leg_bye : Nat := 1
def total_score : Nat := 150

def runs_from_boundaries : Nat := boundaries * runs_per_boundary
def runs_from_sixes : Nat := sixes * runs_per_six
def runs_not_off_bat : Nat := no_balls * runs_per_no_ball + wide_balls * runs_per_wide_ball + leg_byes * runs_per_leg_bye

def runs_running_between_wickets : Nat := total_score - runs_not_off_bat - runs_from_boundaries - runs_from_sixes

def percentage_runs_running_between_wickets : Float := 
  (runs_running_between_wickets.toFloat / total_score.toFloat) * 100

theorem batsman_percentage_running_between_wickets : percentage_runs_running_between_wickets = 58 := sorry

end batsman_percentage_running_between_wickets_l131_131168


namespace local_minimum_f_when_k2_l131_131630

noncomputable def f (k : ℕ) (x : ℝ) : ℝ := (Real.exp x - 1) * (x - 1) ^ k

theorem local_minimum_f_when_k2 : ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f 2 x ≥ f 2 1 :=
by
  -- the question asks to prove that the function attains a local minimum at x = 1 when k = 2
  sorry

end local_minimum_f_when_k2_l131_131630


namespace angle_A_120_l131_131111

variable {a b c : ℝ}
variable {A B C : ℝ}

theorem angle_A_120 
  (h₁ : a^2 - b^2 = 3 * b * c)
  (h₂ : sin C = 2 * sin B) :
  A = 120 :=
sorry

end angle_A_120_l131_131111


namespace percentage_died_by_bombardment_l131_131102

def initial_population : ℕ := 4675
def remaining_population : ℕ := 3553
def left_percentage : ℕ := 20

theorem percentage_died_by_bombardment (x : ℕ) (h : initial_population * (100 - x) / 100 * 8 / 10 = remaining_population) : 
  x = 5 :=
by
  sorry

end percentage_died_by_bombardment_l131_131102


namespace cousin_reading_time_l131_131301

theorem cousin_reading_time (my_time_hours : ℕ) (speed_ratio : ℕ) (my_time_minutes := my_time_hours * 60) :
  (my_time_hours = 3) ∧ (speed_ratio = 5) → 
  (my_time_minutes / speed_ratio = 36) :=
by
  sorry

end cousin_reading_time_l131_131301


namespace jodi_walked_miles_per_day_l131_131961

theorem jodi_walked_miles_per_day (x : ℕ) 
  (h1 : 6 * x + 12 + 18 + 24 = 60) : 
  x = 1 :=
by
  sorry

end jodi_walked_miles_per_day_l131_131961


namespace ratio_B_to_C_l131_131305

theorem ratio_B_to_C (A_share B_share C_share : ℝ) 
  (total : A_share + B_share + C_share = 510) 
  (A_share_val : A_share = 360) 
  (B_share_val : B_share = 90)
  (C_share_val : C_share = 60)
  (A_cond : A_share = (2 / 3) * B_share) 
  : B_share / C_share = 3 / 2 := 
by 
  sorry

end ratio_B_to_C_l131_131305


namespace triangle_ABC_l131_131099

theorem triangle_ABC (a b c : ℝ) (A B C : ℝ)
  (h1 : a + b = 5)
  (h2 : c = Real.sqrt 7)
  (h3 : 4 * (Real.sin ((A + B) / 2))^2 - Real.cos (2 * C) = 7 / 2) :
  (C = Real.pi / 3)
  ∧ (1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :=
by
  sorry

end triangle_ABC_l131_131099


namespace sugar_for_cake_l131_131895

-- Definitions of given values
def sugar_for_frosting : ℝ := 0.6
def total_sugar_required : ℝ := 0.8

-- Proof statement
theorem sugar_for_cake : (total_sugar_required - sugar_for_frosting) = 0.2 :=
by
  sorry

end sugar_for_cake_l131_131895
