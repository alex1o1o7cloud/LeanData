import Mathlib

namespace average_rainfall_feb_1983_l161_16102

theorem average_rainfall_feb_1983 (total_rainfall : ℕ) (days_in_february : ℕ) (hours_per_day : ℕ) 
  (H1 : total_rainfall = 789) (H2 : days_in_february = 28) (H3 : hours_per_day = 24) : 
  total_rainfall / (days_in_february * hours_per_day) = 789 / 672 :=
by
  sorry

end average_rainfall_feb_1983_l161_16102


namespace find_cd_l161_16181

noncomputable def period := (3 / 4) * Real.pi
noncomputable def x_value := (1 / 8) * Real.pi
noncomputable def y_value := 3
noncomputable def tangent_value := Real.tan (Real.pi / 6) -- which is 1 / sqrt(3)
noncomputable def c_value := 3 * Real.sqrt 3

theorem find_cd (c d : ℝ) 
  (h_period : d = 4 / 3) 
  (h_point : y_value = c * Real.tan (d * x_value)) :
  c * d = 4 * Real.sqrt 3 := 
sorry

end find_cd_l161_16181


namespace arithmetic_geometric_progressions_l161_16190

theorem arithmetic_geometric_progressions (a b : ℕ → ℕ) (d r : ℕ) 
  (ha : ∀ n, a (n + 1) = a n + d)
  (hb : ∀ n, b (n + 1) = r * b n)
  (h_comm_ratio : r = 2)
  (h_eq1 : a 1 + d - 2 * (b 1) = a 1 + 2 * d - 4 * (b 1))
  (h_eq2 : a 1 + d - 2 * (b 1) = 8 * (b 1) - (a 1 + 3 * d)) :
  (a 1 = b 1) ∧ (∃ n, ∀ k, 1 ≤ k ∧ k ≤ 10 → (b (k + 1) = a (1 + n * d) + a 1)) := by
  sorry

end arithmetic_geometric_progressions_l161_16190


namespace scientific_notation_26_billion_l161_16184

theorem scientific_notation_26_billion :
  ∃ (a : ℝ) (n : ℤ), (26 * 10^8 : ℝ) = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 2.6 ∧ n = 9 :=
sorry

end scientific_notation_26_billion_l161_16184


namespace magnitude_z1_condition_z2_range_condition_l161_16140

-- Define and set up the conditions and problem statements
open Complex

def complex_number_condition (z₁ : ℂ) (m : ℝ) : Prop :=
  z₁ = 1 + m * I ∧ ((z₁ * (1 - I)).re = 0)

def z₂_condition (z₂ z₁ : ℂ) (n : ℝ) : Prop :=
  z₂ = z₁ * (n - I) ∧ z₂.re < 0 ∧ z₂.im < 0

-- Prove that if z₁ = 1 + m * I and z₁ * (1 - I) is pure imaginary, then |z₁| = sqrt 2
theorem magnitude_z1_condition (m : ℝ) (z₁ : ℂ) 
  (h₁ : complex_number_condition z₁ m) : abs z₁ = Real.sqrt 2 :=
by sorry

-- Prove that if z₂ = z₁ * (n + i^3) is in the third quadrant, then n is in the range (-1, 1)
theorem z2_range_condition (n : ℝ) (m : ℝ) (z₁ z₂ : ℂ)
  (h₁ : complex_number_condition z₁ m)
  (h₂ : z₂_condition z₂ z₁ n) : -1 < n ∧ n < 1 :=
by sorry

end magnitude_z1_condition_z2_range_condition_l161_16140


namespace max_S_n_l161_16132

theorem max_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h1 : ∀ n, a (n + 1) = a n + d) (h2 : d < 0) (h3 : S 6 = 5 * (a 1) + 10 * d) :
  ∃ n, (n = 5 ∨ n = 6) ∧ (∀ m, S m ≤ S n) :=
by
  sorry

end max_S_n_l161_16132


namespace miniVanTankCapacity_is_65_l161_16103

noncomputable def miniVanTankCapacity : ℝ :=
  let serviceCostPerVehicle := 2.10
  let fuelCostPerLiter := 0.60
  let numMiniVans := 3
  let numTrucks := 2
  let totalCost := 299.1
  let truckFactor := 1.2
  let V := (totalCost - serviceCostPerVehicle * (numMiniVans + numTrucks)) /
            (fuelCostPerLiter * (numMiniVans + numTrucks * (1 + truckFactor)))
  V

theorem miniVanTankCapacity_is_65 : miniVanTankCapacity = 65 :=
  sorry

end miniVanTankCapacity_is_65_l161_16103


namespace distance_to_convenience_store_l161_16192

def distance_work := 6
def days_work := 5
def distance_dog_walk := 2
def times_dog_walk := 2
def days_week := 7
def distance_friend_house := 1
def times_friend_visit := 1
def total_miles := 95
def trips_convenience_store := 2

theorem distance_to_convenience_store :
  ∃ x : ℝ,
    (distance_work * 2 * days_work) +
    (distance_dog_walk * times_dog_walk * days_week) +
    (distance_friend_house * 2 * times_friend_visit) +
    (x * trips_convenience_store) = total_miles
    → x = 2.5 :=
by
  sorry

end distance_to_convenience_store_l161_16192


namespace shipping_cost_correct_l161_16169

-- Definitions of given conditions
def total_weight_of_fish : ℕ := 540
def weight_of_each_crate : ℕ := 30
def total_shipping_cost : ℚ := 27

-- Calculating the number of crates
def number_of_crates : ℕ := total_weight_of_fish / weight_of_each_crate

-- Definition of the target shipping cost per crate
def shipping_cost_per_crate : ℚ := total_shipping_cost / number_of_crates

-- Lean statement to prove the given problem
theorem shipping_cost_correct :
  shipping_cost_per_crate = 1.50 := by
  sorry

end shipping_cost_correct_l161_16169


namespace max_value_of_cubes_l161_16166

theorem max_value_of_cubes (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 + ab + ac + ad + bc + bd + cd = 10) :
  a^3 + b^3 + c^3 + d^3 ≤ 4 * Real.sqrt 10 :=
sorry

end max_value_of_cubes_l161_16166


namespace system1_solution_system2_solution_l161_16156

-- System (1)
theorem system1_solution (x y : ℝ) (h1 : x + y = 1) (h2 : 3 * x + y = 5) : x = 2 ∧ y = -1 := sorry

-- System (2)
theorem system2_solution (x y : ℝ) (h1 : 3 * (x - 1) + 4 * y = 1) (h2 : 2 * x + 3 * (y + 1) = 2) : x = 16 ∧ y = -11 := sorry

end system1_solution_system2_solution_l161_16156


namespace determine_num_chickens_l161_16100

def land_acres : ℕ := 30
def land_cost_per_acre : ℕ := 20
def house_cost : ℕ := 120000
def num_cows : ℕ := 20
def cow_cost_per_cow : ℕ := 1000
def install_hours : ℕ := 6
def install_cost_per_hour : ℕ := 100
def equipment_cost : ℕ := 6000
def total_expenses : ℕ := 147700
def chicken_cost_per_chicken : ℕ := 5

def total_cost_before_chickens : ℕ := 
  (land_acres * land_cost_per_acre) + 
  house_cost + 
  (num_cows * cow_cost_per_cow) + 
  (install_hours * install_cost_per_hour) + 
  equipment_cost

def chickens_cost : ℕ := total_expenses - total_cost_before_chickens

def num_chickens : ℕ := chickens_cost / chicken_cost_per_chicken

theorem determine_num_chickens : num_chickens = 100 := by
  sorry

end determine_num_chickens_l161_16100


namespace value_of_expression_l161_16150

theorem value_of_expression (p q r s : ℝ) (h : -27 * p + 9 * q - 3 * r + s = -7) : 
  4 * p - 2 * q + r - s = 7 :=
by
  sorry

end value_of_expression_l161_16150


namespace inequality_proof_l161_16143

variable (a b c d : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) (h_nonneg_d : 0 ≤ d)
variable (h_sum : a + b + c + d = 1)

theorem inequality_proof :
  a * b * c + b * c * d + c * d * a + d * a * b ≤ (1 / 27) + (176 / 27) * a * b * c * d :=
by
  sorry

end inequality_proof_l161_16143


namespace Robert_GRE_exam_l161_16139

/-- Robert started preparation for GRE entrance examination in the month of January and prepared for 5 months. Prove that he could write the examination any date after the end of May.-/
theorem Robert_GRE_exam (start_month : ℕ) (prep_duration : ℕ) : 
  start_month = 1 → prep_duration = 5 → ∃ exam_date, exam_date > 5 :=
by
  sorry

end Robert_GRE_exam_l161_16139


namespace sum_of_coefficients_equals_28_l161_16170

def P (x : ℝ) : ℝ :=
  2 * (4 * x^8 - 5 * x^5 + 9 * x^3 - 6) + 8 * (x^6 - 4 * x^3 + 6)

theorem sum_of_coefficients_equals_28 : P 1 = 28 := by
  sorry

end sum_of_coefficients_equals_28_l161_16170


namespace sum_pattern_l161_16172

theorem sum_pattern (a b : ℕ) : (6 + 7 = 13) ∧ (8 + 9 = 17) ∧ (5 + 6 = 11) ∧ (7 + 8 = 15) ∧ (3 + 3 = 6) → (6 + 7 = 12) :=
by
  sorry

end sum_pattern_l161_16172


namespace acme_cheaper_min_shirts_l161_16154

theorem acme_cheaper_min_shirts :
  ∃ x : ℕ, 60 + 11 * x < 10 + 16 * x ∧ x = 11 :=
by {
  sorry
}

end acme_cheaper_min_shirts_l161_16154


namespace friday_can_determine_arrival_date_l161_16128

-- Define the conditions
def Robinson_crusoe (day : ℕ) : Prop := day % 365 = 0

-- Goal: Within 183 days, Friday can determine his arrival date.
theorem friday_can_determine_arrival_date : 
  (∀ day : ℕ, day < 183 → (Robinson_crusoe day ↔ ¬ Robinson_crusoe (day + 1)) ∨ (day % 365 = 0)) :=
sorry

end friday_can_determine_arrival_date_l161_16128


namespace tens_digit_19_2021_l161_16134

theorem tens_digit_19_2021 : (19^2021 % 100) / 10 % 10 = 1 :=
by sorry

end tens_digit_19_2021_l161_16134


namespace solve_fraction_eq_l161_16137

theorem solve_fraction_eq : 
  ∀ x : ℝ, (x - 3) ≠ 0 → (x + 6) / (x - 3) = 4 → x = 6 := by
  intros x h_ne_zero h_eq
  sorry

end solve_fraction_eq_l161_16137


namespace system1_solution_system2_solution_l161_16104

-- Definition and proof for System (1)
theorem system1_solution (x y : ℝ) (h1 : x - y = 2) (h2 : 2 * x + y = 7) : x = 3 ∧ y = 1 := 
by 
  sorry

-- Definition and proof for System (2)
theorem system2_solution (x y : ℝ) (h1 : x - 2 * y = 3) (h2 : (1 / 2) * x + (3 / 4) * y = 13 / 4) : x = 5 ∧ y = 1 :=
by 
  sorry

end system1_solution_system2_solution_l161_16104


namespace brick_width_is_correct_l161_16171

-- Defining conditions
def wall_length : ℝ := 200 -- wall length in cm
def wall_width : ℝ := 300 -- wall width in cm
def wall_height : ℝ := 2   -- wall height in cm
def brick_length : ℝ := 25 -- brick length in cm
def brick_height : ℝ := 6  -- brick height in cm
def num_bricks : ℝ := 72.72727272727273

-- Total volume of wall
def vol_wall : ℝ := wall_length * wall_width * wall_height

-- Volume of one brick
def vol_brick (width : ℝ) : ℝ := brick_length * width * brick_height

-- Proof statement
theorem brick_width_is_correct : ∃ width : ℝ, vol_wall = vol_brick width * num_bricks ∧ width = 11 :=
by
  sorry

end brick_width_is_correct_l161_16171


namespace inequality_solution_set_l161_16109

theorem inequality_solution_set (x : ℝ) : (2 * x + 1 ≥ 3) ∧ (4 * x - 1 < 7) ↔ (1 ≤ x ∧ x < 2) :=
by
  sorry

end inequality_solution_set_l161_16109


namespace find_t_l161_16141

variables (s t : ℚ)

theorem find_t (h1 : 12 * s + 7 * t = 154) (h2 : s = 2 * t - 3) : t = 190 / 31 :=
by
  sorry

end find_t_l161_16141


namespace remainder_mod_29_l161_16177

-- Definitions of the given conditions
def N (k : ℕ) := 899 * k + 63

-- The proof statement to be proved
theorem remainder_mod_29 (k : ℕ) : (N k) % 29 = 5 := 
by {
  sorry
}

end remainder_mod_29_l161_16177


namespace city_partition_exists_l161_16183

-- Define a market and street as given
structure City where
  markets : Type
  street : markets → markets → Prop
  leaves_exactly_two : ∀ (m : markets), ∃ (m1 m2 : markets), street m m1 ∧ street m m2

-- Our formal proof statement
theorem city_partition_exists (C : City) : 
  ∃ (partition : C.markets → Fin 1014), 
    (∀ (m1 m2 : C.markets), C.street m1 m2 → partition m1 ≠ partition m2) ∧
    (∀ (d1 d2 : Fin 1014) (m1 m2 : C.markets), (partition m1 = d1) ∧ (partition m2 = d2) → 
     (C.street m1 m2 ∨ C.street m2 m1) →  (∀ (k l : Fin 1014), (k = d1) → (l = d2) → (∀ (a b : C.markets), (partition a = k) → (partition b = l) → (C.street a b ∨ C.street b a)))) :=
sorry

end city_partition_exists_l161_16183


namespace xy_eq_zero_l161_16145

theorem xy_eq_zero (x y : ℝ) (h1 : x - y = 3) (h2 : x^3 - y^3 = 27) : x * y = 0 := by
  sorry

end xy_eq_zero_l161_16145


namespace mike_peaches_eq_120_l161_16131

def original_peaches : ℝ := 34.0
def picked_peaches : ℝ := 86.0
def total_peaches (orig : ℝ) (picked : ℝ) : ℝ := orig + picked

theorem mike_peaches_eq_120 : total_peaches original_peaches picked_peaches = 120.0 := 
by
  sorry

end mike_peaches_eq_120_l161_16131


namespace negation_of_universal_statement_l161_16178

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, x^2 - 2*x + 4 ≤ 0)) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by
  sorry

end negation_of_universal_statement_l161_16178


namespace gcd_1821_2993_l161_16110

theorem gcd_1821_2993 : Nat.gcd 1821 2993 = 1 := 
by 
  sorry

end gcd_1821_2993_l161_16110


namespace total_handshakes_l161_16121

theorem total_handshakes (players_team1 players_team2 referees : ℕ) 
  (h1 : players_team1 = 11) (h2 : players_team2 = 11) (h3 : referees = 3) : 
  players_team1 * players_team2 + (players_team1 + players_team2) * referees = 187 := 
by
  sorry

end total_handshakes_l161_16121


namespace right_triangle_perimeter_l161_16123

theorem right_triangle_perimeter (a b : ℕ) (h : a^2 + b^2 = 100) (r : ℕ := 1) :
  (a + b + 10) = 24 :=
sorry

end right_triangle_perimeter_l161_16123


namespace right_triangle_side_length_l161_16158

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (h1 : a = 5) 
  (h2 : c = 12) 
  (h_right : a^2 + b^2 = c^2) : 
  b = Real.sqrt 119 :=
by
  sorry

end right_triangle_side_length_l161_16158


namespace no_odd_total_rows_columns_l161_16122

open Function

def array_odd_column_row_count (n : ℕ) (array : ℕ → ℕ → ℤ) : Prop :=
  n % 2 = 1 ∧
  (∀ i j, 0 ≤ array i j ∧ array i j ≤ 1 ∧ array i j = -1 ∨ array i j = 1) →
  (∃ (rows cols : Finset ℕ),
    rows.card + cols.card = n ∧
    ∀ r ∈ rows, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array r) k = -1 ∧
    ∀ c ∈ cols, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array c) k = -1
    )

theorem no_odd_total_rows_columns (n : ℕ) (array : ℕ → ℕ → ℤ) :
  n % 2 = 1 →
  (∀ i j, 0 ≤ array i j ∧ array i j ≤ 1 ∧ (array i j = -1 ∨ array i j = 1)) →
  ¬ (∃ rows cols : Finset ℕ,
       rows.card + cols.card = n ∧
       ∀ r ∈ rows, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array r k = -1) ∧
       ∀ c ∈ cols, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array c k = -1)) :=
by
  intros h_array
  sorry

end no_odd_total_rows_columns_l161_16122


namespace inverse_function_value_l161_16196

-- Defining the function g as a list of pairs
def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 3
  | 2 => 6
  | 3 => 1
  | 4 => 5
  | 5 => 4
  | 6 => 2
  | _ => 0 -- default case which should not be used

-- Defining the inverse function g_inv using the values determined from g
def g_inv (y : ℕ) : ℕ :=
  match y with
  | 3 => 1
  | 6 => 2
  | 1 => 3
  | 5 => 4
  | 4 => 5
  | 2 => 6
  | _ => 0 -- default case which should not be used

theorem inverse_function_value :
  g_inv (g_inv (g_inv 6)) = 2 :=
by
  sorry

end inverse_function_value_l161_16196


namespace hands_per_hoopit_l161_16114

-- Defining conditions
def num_hoopits := 7
def num_neglarts := 8
def total_toes := 164
def toes_per_hand_hoopit := 3
def toes_per_hand_neglart := 2
def hands_per_neglart := 5

-- The statement to prove
theorem hands_per_hoopit : 
  ∃ (H : ℕ), (H * toes_per_hand_hoopit * num_hoopits + hands_per_neglart * toes_per_hand_neglart * num_neglarts = total_toes) → H = 4 :=
sorry

end hands_per_hoopit_l161_16114


namespace rectangular_C₁_general_C₂_intersection_and_sum_l161_16108

-- Definition of curve C₁ in polar coordinates
def C₁_polar (ρ θ : ℝ) : Prop := ρ * Real.cos θ ^ 2 = Real.sin θ

-- Definition of curve C₂ in parametric form
def C₂_param (k x y : ℝ) : Prop := 
  x = 8 * k / (1 + k^2) ∧ y = 2 * (1 - k^2) / (1 + k^2)

-- Rectangular coordinate equation of curve C₁ is x² = y
theorem rectangular_C₁ (ρ θ : ℝ) (x y : ℝ) (h₁ : ρ * Real.cos θ ^ 2 = Real.sin θ)
  (h₂ : x = ρ * Real.cos θ) (h₃ : y = ρ * Real.sin θ) : x^2 = y :=
sorry

-- General equation of curve C₂ is x² / 16 + y² / 4 = 1 with y ≠ -2
theorem general_C₂ (k x y : ℝ) (h₁ : x = 8 * k / (1 + k^2))
  (h₂ : y = 2 * (1 - k^2) / (1 + k^2)) : x^2 / 16 + y^2 / 4 = 1 ∧ y ≠ -2 :=
sorry

-- Given point M and parametric line l, prove the value of sum reciprocals of distances to points of intersection with curve C₁ is √7
theorem intersection_and_sum (t m₁ m₂ x y : ℝ) 
  (M : ℝ × ℝ) (hM : M = (0, 1/2))
  (hline : x = Real.sqrt 3 * t ∧ y = 1/2 + t)
  (hintersect1 : 3 * m₁^2 - 2 * m₁ - 2 = 0)
  (hintersect2 : 3 * m₂^2 - 2 * m₂ - 2 = 0)
  (hroot1_2 : m₁ + m₂ = 2/3 ∧ m₁ * m₂ = -2/3) : 
  1 / abs (M.fst - x) + 1 / abs (M.snd - y) = Real.sqrt 7 :=
sorry

end rectangular_C₁_general_C₂_intersection_and_sum_l161_16108


namespace find_a_l161_16162

variable {x y a : ℤ}

theorem find_a (h1 : 3 * x + y = 1 + 3 * a) (h2 : x + 3 * y = 1 - a) (h3 : x + y = 0) : a = -1 := 
sorry

end find_a_l161_16162


namespace largest_whole_number_l161_16115

theorem largest_whole_number (x : ℕ) (h : 6 * x + 3 < 150) : x ≤ 24 :=
sorry

end largest_whole_number_l161_16115


namespace exponent_problem_l161_16151

theorem exponent_problem (m : ℕ) : 8^2 = 4^2 * 2^m → m = 2 := by
  intro h
  sorry

end exponent_problem_l161_16151


namespace find_number_l161_16182

theorem find_number (x : ℤ) (h : (x + 305) / 16 = 31) : x = 191 :=
sorry

end find_number_l161_16182


namespace difference_SP_l161_16101

-- Definitions for amounts
variables (P Q R S : ℕ)

-- Conditions given in the problem
def total_amount := P + Q + R + S = 1000
def P_condition := P = 2 * Q
def S_condition := S = 4 * R
def Q_R_equal := Q = R

-- Statement of the problem that needs to be proven
theorem difference_SP (P Q R S : ℕ) (h1 : total_amount P Q R S) 
  (h2 : P_condition P Q) (h3 : S_condition S R) (h4 : Q_R_equal Q R) : 
  S - P = 250 :=
by 
  sorry

end difference_SP_l161_16101


namespace brian_traveled_correct_distance_l161_16138

def miles_per_gallon : Nat := 20
def gallons_used : Nat := 3
def expected_miles : Nat := 60

theorem brian_traveled_correct_distance : (miles_per_gallon * gallons_used) = expected_miles := by
  sorry

end brian_traveled_correct_distance_l161_16138


namespace ratio_of_ages_in_two_years_l161_16189

theorem ratio_of_ages_in_two_years
    (S : ℕ) (M : ℕ) 
    (h1 : M = S + 32)
    (h2 : S = 30) : 
    (M + 2) / (S + 2) = 2 :=
by
  sorry

end ratio_of_ages_in_two_years_l161_16189


namespace geometric_sequence_sum_l161_16161

theorem geometric_sequence_sum 
  (a r : ℝ) 
  (h1 : a + a * r = 8)
  (h2 : a + a * r + a * r^2 + a * r^3 + a * r^4 + a * r^5 = 120) :
  a * (1 + r + r^2 + r^3) = 30 := 
by
  sorry

end geometric_sequence_sum_l161_16161


namespace rectangle_base_length_l161_16188

theorem rectangle_base_length
  (h : ℝ) (b : ℝ)
  (common_height_nonzero : h ≠ 0)
  (triangle_base : ℝ := 24)
  (same_area : (1/2) * triangle_base * h = b * h) :
  b = 12 :=
by
  sorry

end rectangle_base_length_l161_16188


namespace area_triangle_QCA_l161_16142

/--
  Given:
  - θ (θ is acute) is the angle at Q between QA and QC
  - Q is at the coordinates (0, 12)
  - A is at the coordinates (3, 12)
  - C is at the coordinates (0, p)

  Prove that the area of triangle QCA is (3/2) * (12 - p) * sin(θ).
-/
theorem area_triangle_QCA (p θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) :
  let Q := (0, 12)
  let A := (3, 12)
  let C := (0, p)
  let base := 3
  let height := (12 - p) * Real.sin θ
  let area := (1 / 2) * base * height
  area = (3 / 2) * (12 - p) * Real.sin θ := by
  sorry

end area_triangle_QCA_l161_16142


namespace inequality_solution_set_l161_16186

theorem inequality_solution_set 
  (c : ℝ) (a : ℝ) (b : ℝ) (h : c > 0) (hb : b = (5 / 2) * c) (ha : a = - (3 / 2) * c) :
  ∀ x : ℝ, (a * x^2 + b * x + c ≥ 0) ↔ (- (1 / 3) ≤ x ∧ x ≤ 2) :=
sorry

end inequality_solution_set_l161_16186


namespace range_of_a_for_monotonic_function_l161_16187

theorem range_of_a_for_monotonic_function (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → 0 ≤ (1 / x) + a) → a ≥ -1 / 2 := 
by
  sorry

end range_of_a_for_monotonic_function_l161_16187


namespace num_ordered_pairs_l161_16155

theorem num_ordered_pairs : ∃ (n : ℕ), n = 24 ∧ ∀ (a b : ℂ), a^4 * b^6 = 1 ∧ a^8 * b^3 = 1 → n = 24 :=
by
  sorry

end num_ordered_pairs_l161_16155


namespace fractional_inequality_solution_l161_16193

theorem fractional_inequality_solution (x : ℝ) :
  (x - 2) / (x + 1) < 0 ↔ -1 < x ∧ x < 2 :=
sorry

end fractional_inequality_solution_l161_16193


namespace min_value_2a_minus_ab_l161_16107

theorem min_value_2a_minus_ab (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (ha_lt_11 : a < 11) (hb_lt_11 : b < 11) : 
  ∃ (min_val : ℤ), min_val = -80 ∧ ∀ x y : ℕ, 0 < x → 0 < y → x < 11 → y < 11 → 2 * x - x * y ≥ min_val :=
by
  use -80
  sorry

end min_value_2a_minus_ab_l161_16107


namespace Q_div_P_l161_16157

theorem Q_div_P (P Q : ℚ) (h : ∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 5 →
  P / (x + 3) + Q / (x * (x - 5)) = (x^2 - 3 * x + 8) / (x * (x + 3) * (x - 5))) :
  Q / P = 1 / 3 :=
by
  sorry

end Q_div_P_l161_16157


namespace unique_valid_quintuple_l161_16148

theorem unique_valid_quintuple :
  ∃! (a b c d e : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧
    a^2 + b^2 + c^3 + d^3 + e^3 = 5 ∧
    (a + b + c + d + e) * (a^3 + b^3 + c^2 + d^2 + e^2) = 25 :=
sorry

end unique_valid_quintuple_l161_16148


namespace pseudo_code_output_l161_16111

theorem pseudo_code_output (a b c : Int)
  (h1 : a = 3)
  (h2 : b = -5)
  (h3 : c = 8)
  (ha : a = -5)
  (hb : b = 8)
  (hc : c = -5) : 
  a = -5 ∧ b = 8 ∧ c = -5 :=
by
  sorry

end pseudo_code_output_l161_16111


namespace cookies_yesterday_l161_16119

theorem cookies_yesterday (cookies_today : ℕ) (difference : ℕ)
  (h1 : cookies_today = 140)
  (h2 : difference = 30) :
  cookies_today - difference = 110 :=
by
  sorry

end cookies_yesterday_l161_16119


namespace ratio_of_parallel_vectors_l161_16116

theorem ratio_of_parallel_vectors (m n : ℝ) 
  (h1 : ∃ k : ℝ, (m, 1, 3) = (k * 2, k * n, k)) : (m / n) = 18 :=
by
  sorry

end ratio_of_parallel_vectors_l161_16116


namespace show_revenue_and_vacancies_l161_16136

theorem show_revenue_and_vacancies:
  let total_seats := 600
  let vip_seats := 50
  let general_seats := 400
  let balcony_seats := 150
  let vip_price := 40
  let general_price := 25
  let balcony_price := 15
  let vip_filled_rate := 0.80
  let general_filled_rate := 0.70
  let balcony_filled_rate := 0.50
  let vip_filled := vip_filled_rate * vip_seats
  let general_filled := general_filled_rate * general_seats
  let balcony_filled := balcony_filled_rate * balcony_seats
  let vip_revenue := vip_filled * vip_price
  let general_revenue := general_filled * general_price
  let balcony_revenue := balcony_filled * balcony_price
  let overall_revenue := vip_revenue + general_revenue + balcony_revenue
  let vip_vacant := vip_seats - vip_filled
  let general_vacant := general_seats - general_filled
  let balcony_vacant := balcony_seats - balcony_filled
  vip_revenue = 1600 ∧
  general_revenue = 7000 ∧
  balcony_revenue = 1125 ∧
  overall_revenue = 9725 ∧
  vip_vacant = 10 ∧
  general_vacant = 120 ∧
  balcony_vacant = 75 :=
by
  sorry

end show_revenue_and_vacancies_l161_16136


namespace correct_factorization_l161_16120

-- Definitions for the given conditions of different options
def condition_A (a : ℝ) : Prop := 2 * a^2 - 2 * a + 1 = 2 * a * (a - 1) + 1
def condition_B (x y : ℝ) : Prop := (x + y) * (x - y) = x^2 - y^2
def condition_C (x y : ℝ) : Prop := x^2 - 4 * x * y + 4 * y^2 = (x - 2 * y)^2
def condition_D (x : ℝ) : Prop := x^2 + 1 = x * (x + 1 / x)

-- The theorem to prove that option C is correct
theorem correct_factorization (x y : ℝ) : condition_C x y :=
by sorry

end correct_factorization_l161_16120


namespace minimum_questions_to_determine_village_l161_16125

-- Step 1: Define the types of villages
inductive Village
| A : Village
| B : Village
| C : Village

-- Step 2: Define the properties of residents in each village
def tells_truth (v : Village) (p : Prop) : Prop :=
  match v with
  | Village.A => p
  | Village.B => ¬p
  | Village.C => p ∨ ¬p

-- Step 3: Define the problem context in Lean
theorem minimum_questions_to_determine_village :
    ∀ (tourist_village person_village : Village), ∃ (n : ℕ), n = 4 := by
  sorry

end minimum_questions_to_determine_village_l161_16125


namespace set_representation_l161_16159

def A (x : ℝ) := -3 < x ∧ x < 1
def B (x : ℝ) := x ≤ -1
def C (x : ℝ) := -2 < x ∧ x ≤ 2

theorem set_representation :
  (∀ x, A x ↔ (A x ∧ (B x ∨ C x))) ∧
  (∀ x, A x ↔ (A x ∨ (B x ∧ C x))) ∧
  (∀ x, A x ↔ ((A x ∧ B x) ∨ (A x ∧ C x))) :=
by
  sorry

end set_representation_l161_16159


namespace price_cashews_l161_16199

noncomputable def price_per_pound_cashews 
  (price_mixed_nuts_per_pound : ℝ) 
  (weight_mixed_nuts : ℕ) 
  (weight_peanuts : ℕ) 
  (price_peanuts_per_pound : ℝ) 
  (weight_cashews : ℕ) : ℝ := 
  (price_mixed_nuts_per_pound * weight_mixed_nuts - price_peanuts_per_pound * weight_peanuts) / weight_cashews

open Real

theorem price_cashews 
  (price_mixed_nuts_per_pound : ℝ) 
  (weight_mixed_nuts : ℕ) 
  (weight_peanuts : ℕ) 
  (price_peanuts_per_pound : ℝ) 
  (weight_cashews : ℕ)
  (h1 : price_mixed_nuts_per_pound = 2.50) 
  (h2 : weight_mixed_nuts = 100) 
  (h3 : weight_peanuts = 40) 
  (h4 : price_peanuts_per_pound = 3.50) 
  (h5 : weight_cashews = 60) : 
  price_per_pound_cashews price_mixed_nuts_per_pound weight_mixed_nuts weight_peanuts price_peanuts_per_pound weight_cashews = 11 / 6 := by 
  sorry

end price_cashews_l161_16199


namespace find_m_range_l161_16163

noncomputable def quadratic_inequality_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x + 2 > 0

theorem find_m_range :
  { m : ℝ | quadratic_inequality_condition m } = { m : ℝ | 1 ≤ m ∧ m < 9 } :=
sorry

end find_m_range_l161_16163


namespace smallest_value_3a_2_l161_16168

theorem smallest_value_3a_2 (a : ℝ) (h : 8 * a^2 + 6 * a + 5 = 2) : 3 * a + 2 = - (5 / 2) := sorry

end smallest_value_3a_2_l161_16168


namespace eccentricity_of_hyperbola_l161_16118

variable {a b c e : ℝ}
variable (h_hyperbola : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
variable (ha_pos : a > 0)
variable (hb_pos : b > 0)
variable (h_vertices : A1 = (-a, 0) ∧ A2 = (a, 0))
variable (h_imaginary_axis : B1 = (0, b) ∧ B2 = (0, -b))
variable (h_foci : F1 = (-c, 0) ∧ F2 = (c, 0))
variable (h_relation : a^2 + b^2 = c^2)
variable (h_tangent_circle : ∀ d, (d = 2*a) → (tangent (circle d) (rhombus F1 B1 F2 B2)))

theorem eccentricity_of_hyperbola : e = (1 + Real.sqrt 5) / 2 :=
sorry

end eccentricity_of_hyperbola_l161_16118


namespace fractional_expression_value_l161_16195

theorem fractional_expression_value (x y z : ℝ) (hz : z ≠ 0) 
  (h1 : 2 * x - 3 * y - z = 0)
  (h2 : x + 3 * y - 14 * z = 0) :
  (x^2 + 3 * x * y) / (y^2 + z^2) = 7 := 
by sorry

end fractional_expression_value_l161_16195


namespace general_formula_of_geometric_seq_term_in_arithmetic_seq_l161_16144

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Condition: Geometric sequence {a_n} with a_1 = 2 and a_4 = 16
def geometric_seq (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n, a (n + 1) = a n * q

-- General formula for the sequence {a_n}
theorem general_formula_of_geometric_seq 
  (ha : geometric_seq a) (h1 : a 1 = 2) (h4 : a 4 = 16) :
  ∀ n, a n = 2^n :=
sorry

-- Condition: Arithmetic sequence {b_n} with b_3 = a_3 and b_5 = a_5
def arithmetic_seq (b : ℕ → ℝ) := ∃ d : ℝ, ∀ n, b (n + 1) = b n + d

-- Check if a_9 is a term in the sequence {b_n} and find its term number
theorem term_in_arithmetic_seq 
  (ha : geometric_seq a) (hb : arithmetic_seq b)
  (h1 : a 1 = 2) (h4 : a 4 = 16)
  (hb3 : b 3 = a 3) (hb5 : b 5 = a 5) :
  ∃ n, b n = a 9 ∧ n = 45 :=
sorry

end general_formula_of_geometric_seq_term_in_arithmetic_seq_l161_16144


namespace total_columns_l161_16165

variables (N L : ℕ)

theorem total_columns (h1 : L > 1500) (h2 : L = 30 * (N - 70)) : N = 180 :=
by
  sorry

end total_columns_l161_16165


namespace region_diff_correct_l161_16147

noncomputable def hexagon_area : ℝ := (3 * Real.sqrt 3) / 2
noncomputable def one_triangle_area : ℝ := (Real.sqrt 3) / 4
noncomputable def triangles_area : ℝ := 18 * one_triangle_area
noncomputable def R_area : ℝ := hexagon_area + triangles_area
noncomputable def S_area : ℝ := 4 * (1 + Real.sqrt 2)
noncomputable def region_diff : ℝ := S_area - R_area

theorem region_diff_correct :
  region_diff = 4 + 4 * Real.sqrt 2 - 6 * Real.sqrt 3 :=
by
  sorry

end region_diff_correct_l161_16147


namespace total_surface_area_with_holes_l161_16185

def cube_edge_length : ℝ := 5
def hole_side_length : ℝ := 2

/-- Calculate the total surface area of a modified cube with given edge length and holes -/
theorem total_surface_area_with_holes 
  (l : ℝ) (h : ℝ)
  (hl_pos : l > 0) (hh_pos : h > 0) (hh_lt_hl : h < l) : 
  (6 * l^2 - 6 * h^2 + 6 * 4 * h^2) = 222 :=
by sorry

end total_surface_area_with_holes_l161_16185


namespace cost_price_of_table_l161_16152

theorem cost_price_of_table (CP : ℝ) (SP : ℝ) (h1 : SP = CP * 1.10) (h2 : SP = 8800) : CP = 8000 :=
by
  sorry

end cost_price_of_table_l161_16152


namespace find_x_y_sum_squared_l161_16194

theorem find_x_y_sum_squared (x y : ℝ) (h1 : x * y = 6) (h2 : (1 / x^2) + (1 / y^2) = 7) (h3 : x - y = Real.sqrt 10) :
  (x + y)^2 = 264 := sorry

end find_x_y_sum_squared_l161_16194


namespace max_sin_A_plus_sin_C_l161_16164

variables {a b c S : ℝ}
variables {A B C : ℝ}

-- Assume the sides of the triangle
variables (ha : a > 0) (hb : b > 0) (hc : c > 0)

-- Assume the angles of the triangle
variables (hA : A > 0) (hB : B > (Real.pi / 2)) (hC : C > 0)
variables (hSumAngles : A + B + C = Real.pi)

-- Assume the relationship between the area and the sides
variables (hArea : S = (1/2) * a * c * Real.sin B)

-- Assume the given equation holds
variables (hEquation : 4 * b * S = a * (b^2 + c^2 - a^2))

-- The statement to prove
theorem max_sin_A_plus_sin_C : (Real.sin A + Real.sin C) ≤ 9 / 8 :=
sorry

end max_sin_A_plus_sin_C_l161_16164


namespace units_digit_sum_2_pow_a_5_pow_b_l161_16133

theorem units_digit_sum_2_pow_a_5_pow_b (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 100)
  (h2 : 1 ≤ b ∧ b ≤ 100) :
  (2 ^ a + 5 ^ b) % 10 ≠ 8 :=
sorry

end units_digit_sum_2_pow_a_5_pow_b_l161_16133


namespace edward_books_bought_l161_16197

def money_spent : ℕ := 6
def cost_per_book : ℕ := 3

theorem edward_books_bought : money_spent / cost_per_book = 2 :=
by
  sorry

end edward_books_bought_l161_16197


namespace speed_in_still_water_l161_16176

variable (upstream downstream : ℝ)

-- Conditions
def upstream_speed : Prop := upstream = 26
def downstream_speed : Prop := downstream = 40

-- Question and correct answer
theorem speed_in_still_water (h1 : upstream_speed upstream) (h2 : downstream_speed downstream) :
  (upstream + downstream) / 2 = 33 := by
  sorry

end speed_in_still_water_l161_16176


namespace circle_radius_3_l161_16146

theorem circle_radius_3 :
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 2 * y - 7 = 0) → (∃ r : ℝ, r = 3) :=
by
  sorry

end circle_radius_3_l161_16146


namespace simple_interest_sum_l161_16198

theorem simple_interest_sum (SI R T : ℝ) (hSI : SI = 4016.25) (hR : R = 0.01) (hT : T = 3) :
  SI / (R * T) = 133875 := by
  sorry

end simple_interest_sum_l161_16198


namespace max_sum_of_squares_70_l161_16106

theorem max_sum_of_squares_70 :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  a^2 + b^2 + c^2 + d^2 = 70 ∧ a + b + c + d = 16 :=
by
  sorry

end max_sum_of_squares_70_l161_16106


namespace younger_brother_age_l161_16167

variable (x y : ℕ)

-- Conditions
axiom sum_of_ages : x + y = 46
axiom younger_is_third_plus_ten : y = x / 3 + 10

theorem younger_brother_age : y = 19 := 
by
  sorry

end younger_brother_age_l161_16167


namespace measure_angle_PSR_is_40_l161_16129

noncomputable def isosceles_triangle (P Q R : Point) (PQ PR : ℝ) (hPQ_PR : PQ = PR) : Triangle := sorry
noncomputable def square (D R S T : Point) : Square := sorry
noncomputable def angle (A B C : Point) (θ : ℝ) : Prop := sorry

def angle_PQR (P Q R : Point) (PQ PR : ℝ) (hPQ_PR : PQ = PR) : ℝ := sorry
def angle_PRQ (P Q R : Point) (PQ PR : ℝ) (hPQ_PR : PQ = PR) : ℝ := sorry

theorem measure_angle_PSR_is_40
  (P Q R S T D : Point)
  (PQ PR : ℝ)
  (hPQ_PR : PQ = PR)
  (hQ_eq_D : Q = D)
  (hQPS : angle P Q S 100)
  (hDRST_square : square D R S T) : angle P S R 40 :=
by
  -- Proof omitted for brevity
  sorry

end measure_angle_PSR_is_40_l161_16129


namespace water_consumption_and_bill_34_7_l161_16173

noncomputable def calculate_bill (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 20.8 * x
  else if 1 < x ∧ x ≤ (5 / 3) then 27.8 * x - 7
  else 32 * x - 14

theorem water_consumption_and_bill_34_7 (x : ℝ) :
  calculate_bill 1.5 = 34.7 ∧ 5 * 1.5 = 7.5 ∧ 3 * 1.5 = 4.5 ∧ 
  5 * 2.6 + (5 * 1.5 - 5) * 4 = 23 ∧ 
  4.5 * 2.6 = 11.7 :=
  sorry

end water_consumption_and_bill_34_7_l161_16173


namespace quadratic_real_roots_l161_16124

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 4 * x - 1 = 0) ↔ m ≥ -3 ∧ m ≠ 1 := 
by 
  sorry

end quadratic_real_roots_l161_16124


namespace y_z_add_x_eq_160_l161_16126

theorem y_z_add_x_eq_160 (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x * (y + z) = 132) (h5 : z * (x + y) = 180) (h6 : x * y * z = 160) :
  y * (z + x) = 160 := 
by 
  sorry

end y_z_add_x_eq_160_l161_16126


namespace inequality_neg_mul_l161_16113

theorem inequality_neg_mul (a b : ℝ) (h : a > b) : -3 * a < -3 * b :=
sorry

end inequality_neg_mul_l161_16113


namespace correct_operation_l161_16149

theorem correct_operation (a b : ℝ) :
  (3 * a^2 - a^2 ≠ 3) ∧
  ((a + b)^2 ≠ a^2 + b^2) ∧
  ((-3 * a * b^2)^2 ≠ -6 * a^2 * b^4) →
  a^3 / a^2 = a :=
by
sorry

end correct_operation_l161_16149


namespace proposition_d_correct_l161_16179

theorem proposition_d_correct (a b c : ℝ) (h : a > b) : a - c > b - c := 
by
  sorry

end proposition_d_correct_l161_16179


namespace sector_central_angle_in_radians_l161_16105

/-- 
Given a sector of a circle where the perimeter is 4 cm 
and the area is 1 cm², prove that the central angle 
of the sector in radians is 2.
-/
theorem sector_central_angle_in_radians 
  (r l : ℝ) 
  (h_perimeter : 2 * r + l = 4) 
  (h_area : (1 / 2) * l * r = 1) : 
  l / r = 2 :=
by
  sorry

end sector_central_angle_in_radians_l161_16105


namespace find_number_l161_16112

theorem find_number (number : ℝ) (h : 0.003 * number = 0.15) : number = 50 :=
by
  sorry

end find_number_l161_16112


namespace find_p5_l161_16175

-- Definitions based on conditions from the problem
def p (x : ℝ) : ℝ :=
  x^4 - 10 * x^3 + 35 * x^2 - 50 * x + 18  -- this construction ensures it's a quartic monic polynomial satisfying provided conditions

-- The main theorem we want to prove
theorem find_p5 :
  p 1 = 3 ∧ p 2 = 7 ∧ p 3 = 13 ∧ p 4 = 21 → p 5 = 51 :=
by
  -- The proof will be inserted here later
  sorry

end find_p5_l161_16175


namespace line_always_passes_through_fixed_point_l161_16160

theorem line_always_passes_through_fixed_point (k : ℝ) : 
  ∀ x y, y + 2 = k * (x + 1) → (x = -1 ∧ y = -2) :=
by
  sorry

end line_always_passes_through_fixed_point_l161_16160


namespace shortest_distance_between_circles_l161_16127

theorem shortest_distance_between_circles :
  let c1 := (1, -3)
  let r1 := 2 * Real.sqrt 2
  let c2 := (-3, 1)
  let r2 := 1
  let distance_centers := Real.sqrt ((1 - -3)^2 + (-3 - 1)^2)
  let shortest_distance := distance_centers - (r1 + r2)
  shortest_distance = 2 * Real.sqrt 2 - 1 :=
by
  sorry

end shortest_distance_between_circles_l161_16127


namespace original_radius_eq_n_div_3_l161_16153

theorem original_radius_eq_n_div_3 (r n : ℝ) (h : (r + n)^2 = 4 * r^2) : r = n / 3 :=
by
  sorry

end original_radius_eq_n_div_3_l161_16153


namespace geometric_sequence_sum_l161_16174

/-- Let {a_n} be a geometric sequence with positive common ratio, a_1 = 2, and a_3 = a_2 + 4.
    Prove the general formula for a_n is 2^n, and the sum of the first n terms, S_n, of the sequence { (2n+1)a_n }
    is (2n-1) * 2^(n+1) + 2. -/
theorem geometric_sequence_sum
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h3 : a 3 = a 2 + 4) :
  (∀ n, a n = 2^n) ∧
  (∀ S : ℕ → ℕ, ∀ n, S n = (2 * n - 1) * 2 ^ (n + 1) + 2) :=
by sorry

end geometric_sequence_sum_l161_16174


namespace time_to_complete_job_l161_16180

-- Define the conditions
variables {A B : ℕ} -- Efficiencies of A and B

-- Assume B's efficiency is 100 units, and A is 130 units.
def efficiency_A : ℕ := 130
def efficiency_B : ℕ := 100

-- Given: A can complete the job in 23 days
def days_A : ℕ := 23

-- Compute total work W. Since A can complete the job in 23 days and its efficiency is 130 units/day:
def total_work : ℕ := efficiency_A * days_A

-- Combined efficiency of A and B
def combined_efficiency : ℕ := efficiency_A + efficiency_B

-- Determine the time taken by A and B working together
def time_A_B_together : ℕ := total_work / combined_efficiency

-- Prove that the time A and B working together is 13 days
theorem time_to_complete_job : time_A_B_together = 13 :=
by
  sorry -- Proof is omitted as per instructions

end time_to_complete_job_l161_16180


namespace sum_is_3600_l161_16191

variables (P R T : ℝ)
variables (CI SI : ℝ)

theorem sum_is_3600
  (hR : R = 10)
  (hT : T = 2)
  (hCI : CI = P * (1 + R / 100) ^ T - P)
  (hSI : SI = P * R * T / 100)
  (h_diff : CI - SI = 36) :
  P = 3600 :=
sorry

end sum_is_3600_l161_16191


namespace trains_total_distance_l161_16130

theorem trains_total_distance (speedA_kmph speedB_kmph time_min : ℕ)
                             (hA : speedA_kmph = 70)
                             (hB : speedB_kmph = 90)
                             (hT : time_min = 15) :
    let speedA_kmpm := (speedA_kmph : ℝ) / 60
    let speedB_kmpm := (speedB_kmph : ℝ) / 60
    let distanceA := speedA_kmpm * (time_min : ℝ)
    let distanceB := speedB_kmpm * (time_min : ℝ)
    distanceA + distanceB = 40 := 
by 
  sorry

end trains_total_distance_l161_16130


namespace simplify_expression_l161_16135

theorem simplify_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 - 2) / x) * ((y^2 - 2) / y) - ((x^2 + 2) / y) * ((y^2 + 2) / x) = -4 * (x / y + y / x) :=
by
  sorry

end simplify_expression_l161_16135


namespace probability_of_log_ge_than_1_l161_16117

noncomputable def probability_log_greater_than_one : ℝ := sorry

theorem probability_of_log_ge_than_1 :
  probability_log_greater_than_one = 1 / 2 :=
sorry

end probability_of_log_ge_than_1_l161_16117
