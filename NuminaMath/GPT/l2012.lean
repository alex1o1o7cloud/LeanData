import Mathlib

namespace train_ride_length_l2012_201225

noncomputable def totalMinutesUntil0900 (leaveTime : Nat) (arrivalTime : Nat) : Nat :=
  arrivalTime - leaveTime

noncomputable def walkTime : Nat := 10

noncomputable def rideTime (totalTime : Nat) (walkTime : Nat) : Nat :=
  totalTime - walkTime

theorem train_ride_length (leaveTime : Nat) (arrivalTime : Nat) :
  leaveTime = 450 → arrivalTime = 540 → rideTime (totalMinutesUntil0900 leaveTime arrivalTime) walkTime = 80 :=
by
  intros h_leaveTime h_arrivalTime
  rw [h_leaveTime, h_arrivalTime]
  unfold totalMinutesUntil0900
  unfold rideTime
  unfold walkTime
  sorry

end train_ride_length_l2012_201225


namespace average_income_eq_58_l2012_201284

def income_day1 : ℕ := 45
def income_day2 : ℕ := 50
def income_day3 : ℕ := 60
def income_day4 : ℕ := 65
def income_day5 : ℕ := 70
def number_of_days : ℕ := 5

theorem average_income_eq_58 :
  (income_day1 + income_day2 + income_day3 + income_day4 + income_day5) / number_of_days = 58 := by
  sorry

end average_income_eq_58_l2012_201284


namespace coins_after_10_hours_l2012_201286

def numberOfCoinsRemaining : Nat :=
  let hour1_coins := 20
  let hour2_coins := hour1_coins + 30
  let hour3_coins := hour2_coins + 30
  let hour4_coins := hour3_coins + 40
  let hour5_coins := hour4_coins - (hour4_coins * 20 / 100)
  let hour6_coins := hour5_coins + 50
  let hour7_coins := hour6_coins + 60
  let hour8_coins := hour7_coins - (hour7_coins / 5)
  let hour9_coins := hour8_coins + 70
  let hour10_coins := hour9_coins - (hour9_coins * 15 / 100)
  hour10_coins

theorem coins_after_10_hours : numberOfCoinsRemaining = 200 := by
  sorry

end coins_after_10_hours_l2012_201286


namespace maurice_rides_l2012_201206

theorem maurice_rides (M : ℕ) 
    (h1 : ∀ m_attended : ℕ, m_attended = 8)
    (h2 : ∀ matt_other : ℕ, matt_other = 16)
    (h3 : ∀ total_matt : ℕ, total_matt = matt_other + m_attended)
    (h4 : total_matt = 3 * M) : M = 8 :=
by 
  sorry

end maurice_rides_l2012_201206


namespace sum_of_digits_is_21_l2012_201204

theorem sum_of_digits_is_21 :
  ∃ (a b c d : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) ∧ 
  ((10 * a + b) * (10 * c + b) = 111 * d) ∧ 
  (d = 9) ∧ 
  (a + b + c + d = 21) := by
  sorry

end sum_of_digits_is_21_l2012_201204


namespace jake_present_weight_l2012_201261

theorem jake_present_weight :
  ∃ (J K L : ℕ), J = 194 ∧ J + K = 287 ∧ J - L = 2 * K ∧ J = 194 := by
  sorry

end jake_present_weight_l2012_201261


namespace crackers_given_to_friends_l2012_201207

theorem crackers_given_to_friends (crackers_per_friend : ℕ) (number_of_friends : ℕ) (h1 : crackers_per_friend = 6) (h2 : number_of_friends = 6) : (crackers_per_friend * number_of_friends) = 36 :=
by
  sorry

end crackers_given_to_friends_l2012_201207


namespace algebraic_fraction_l2012_201233

theorem algebraic_fraction (x : ℝ) (h1 : 1 / 3 = 1 / 3) 
(h2 : x / Real.pi = x / Real.pi) 
(h3 : 2 / (x + 3) = 2 / (x + 3))
(h4 : (x + 2) / 3 = (x + 2) / 3) 
: 
2 / (x + 3) = 2 / (x + 3) := sorry

end algebraic_fraction_l2012_201233


namespace smallest_a_plus_b_l2012_201293

theorem smallest_a_plus_b 
  (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : 2^10 * 3^5 = a^b) : a + b = 248833 :=
sorry

end smallest_a_plus_b_l2012_201293


namespace find_f_l2012_201205

def f : ℝ → ℝ := sorry

theorem find_f (x : ℝ) : f (x + 2) = 2 * x + 3 → f x = 2 * x - 1 :=
by
  intro h
  -- Proof goes here 
  sorry

end find_f_l2012_201205


namespace lcm_18_24_l2012_201278

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l2012_201278


namespace angle_bc_l2012_201229

variables (a b c : ℝ → ℝ → Prop) (theta : ℝ)

-- Definitions of parallelism and angle conditions
def parallel (x y : ℝ → ℝ → Prop) : Prop := ∀ p q r s : ℝ, x p q → y r s → p - q = r - s

def angle_between (x y : ℝ → ℝ → Prop) (θ : ℝ) : Prop := sorry  -- Assume we have a definition for angle between lines

-- Given conditions
axiom parallel_ab : parallel a b
axiom angle_ac : angle_between a c theta

-- Theorem statement
theorem angle_bc : angle_between b c theta :=
sorry

end angle_bc_l2012_201229


namespace total_drivers_l2012_201211

theorem total_drivers (N : ℕ) (A : ℕ) (sA sB sC sD : ℕ) (total_sampled : ℕ)
  (hA : A = 96) (hsA : sA = 12) (hsB : sB = 21) (hsC : sC = 25) (hsD : sD = 43) (htotal : total_sampled = sA + sB + sC + sD)
  (hsA_proportion : (sA : ℚ) / A = (total_sampled : ℚ) / N) : N = 808 := by
  sorry

end total_drivers_l2012_201211


namespace largest_fraction_l2012_201212

noncomputable def compare_fractions : List ℚ :=
  [5 / 11, 7 / 16, 9 / 20, 11 / 23, 111 / 245, 145 / 320, 185 / 409, 211 / 465, 233 / 514]

theorem largest_fraction :
  max (5 / 11) (max (7 / 16) (max (9 / 20) (max (11 / 23) (max (111 / 245) (max (145 / 320) (max (185 / 409) (max (211 / 465) (233 / 514)))))))) = 11 / 23 := 
  sorry

end largest_fraction_l2012_201212


namespace product_and_divisibility_l2012_201245

theorem product_and_divisibility (n : ℕ) (h : n = 3) :
  (n-1) * n * (n+1) * (n+2) * (n+3) = 720 ∧ ¬ (720 % 11 = 0) :=
by
  sorry

end product_and_divisibility_l2012_201245


namespace largest_possible_percent_error_l2012_201274

theorem largest_possible_percent_error
  (d : ℝ) (error_percent : ℝ) (actual_area : ℝ)
  (h_d : d = 30) (h_error_percent : error_percent = 0.1)
  (h_actual_area : actual_area = 225 * Real.pi) :
  ∃ max_error_percent : ℝ,
    (max_error_percent = 21) :=
by
  sorry

end largest_possible_percent_error_l2012_201274


namespace min_value_expression_l2012_201226

theorem min_value_expression : ∀ x : ℝ, (x + 1)^2 * (x + 2)^2 * (x + 3)^2 * (x + 4)^2 + 2025 ≥ 3625 :=
by
  sorry

end min_value_expression_l2012_201226


namespace building_height_l2012_201242

noncomputable def height_of_building (flagpole_height shadow_of_flagpole shadow_of_building : ℝ) : ℝ :=
  (flagpole_height / shadow_of_flagpole) * shadow_of_building

theorem building_height : height_of_building 18 45 60 = 24 := by {
  sorry
}

end building_height_l2012_201242


namespace union_complement_l2012_201298

open Set

-- Definitions for the universal set U and subsets A, B
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 3, 4}
def B : Set ℕ := {1, 3}

-- Definition for the complement of A with respect to U
def CuA : Set ℕ := U \ A

-- Proof statement
theorem union_complement (U_def : U = {0, 1, 2, 3, 4})
                         (A_def : A = {0, 3, 4})
                         (B_def : B = {1, 3}) :
  (CuA ∪ B) = {1, 2, 3} := by
  sorry

end union_complement_l2012_201298


namespace evaluate_expression_l2012_201209

variable (a b c : ℝ)

theorem evaluate_expression 
  (h : a / (20 - a) + b / (75 - b) + c / (55 - c) = 8) :
  4 / (20 - a) + 15 / (75 - b) + 11 / (55 - c) = 8.8 :=
sorry

end evaluate_expression_l2012_201209


namespace rationalize_fraction_l2012_201252

theorem rationalize_fraction :
  (5 : ℚ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18 + Real.sqrt 32) = 
  (5 * Real.sqrt 2) / 36 :=
by
  sorry

end rationalize_fraction_l2012_201252


namespace students_without_A_l2012_201266

theorem students_without_A 
  (total_students : ℕ) 
  (A_in_literature : ℕ) 
  (A_in_science : ℕ) 
  (A_in_both : ℕ) 
  (h_total_students : total_students = 35)
  (h_A_in_literature : A_in_literature = 10)
  (h_A_in_science : A_in_science = 15)
  (h_A_in_both : A_in_both = 5) :
  total_students - (A_in_literature + A_in_science - A_in_both) = 15 :=
by {
  sorry
}

end students_without_A_l2012_201266


namespace whipped_cream_needed_l2012_201203

/- Problem conditions -/
def pies_per_day : ℕ := 3
def days : ℕ := 11
def pies_total : ℕ := pies_per_day * days
def pies_eaten_by_tiffany : ℕ := 4
def pies_remaining : ℕ := pies_total - pies_eaten_by_tiffany
def whipped_cream_per_pie : ℕ := 2

/- Proof statement -/
theorem whipped_cream_needed : whipped_cream_per_pie * pies_remaining = 58 := by
  sorry

end whipped_cream_needed_l2012_201203


namespace range_of_a_l2012_201263

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x = 1 → x > a) : a < 1 := 
by
  sorry

end range_of_a_l2012_201263


namespace joe_travel_time_l2012_201215

theorem joe_travel_time
  (d : ℝ) -- Total distance
  (rw : ℝ) (rr : ℝ) -- Walking and running rates
  (tw : ℝ) -- Walking time
  (tr : ℝ) -- Running time
  (h1 : tw = 9)
  (h2 : rr = 4 * rw)
  (h3 : rw * tw = d / 3)
  (h4 : rr * tr = 2 * d / 3) :
  tw + tr = 13.5 :=
by 
  sorry

end joe_travel_time_l2012_201215


namespace number_of_propositions_is_4_l2012_201243

def is_proposition (s : String) : Prop :=
  s = "The Earth is a planet in the solar system" ∨ 
  s = "{0} ∈ ℕ" ∨ 
  s = "1+1 > 2" ∨ 
  s = "Elderly people form a set"

theorem number_of_propositions_is_4 : 
  (is_proposition "The Earth is a planet in the solar system" ∨ 
   is_proposition "{0} ∈ ℕ" ∨ 
   is_proposition "1+1 > 2" ∨ 
   is_proposition "Elderly people form a set") → 
  4 = 4 :=
by
  sorry

end number_of_propositions_is_4_l2012_201243


namespace math_books_count_l2012_201244

theorem math_books_count (M H : ℕ) (h1 : M + H = 80) (h2 : 4 * M + 5 * H = 373) : M = 27 :=
by
  sorry

end math_books_count_l2012_201244


namespace least_possible_value_m_n_l2012_201231

theorem least_possible_value_m_n :
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ Nat.gcd (m + n) 330 = 1 ∧ n ∣ m^m ∧ ¬(m % n = 0) ∧ (m + n = 377) :=
by
  sorry

end least_possible_value_m_n_l2012_201231


namespace smallest_denominator_of_sum_of_irreducible_fractions_l2012_201246

theorem smallest_denominator_of_sum_of_irreducible_fractions :
  ∀ (a b : ℕ),
  Nat.Coprime a 600 → Nat.Coprime b 700 →
  (∃ c d : ℕ, Nat.Coprime c d ∧ d < 168 ∧ (7 * a + 6 * b) / Nat.gcd (7 * a + 6 * b) 4200 = c / d) →
  False :=
by
  sorry

end smallest_denominator_of_sum_of_irreducible_fractions_l2012_201246


namespace camping_trip_percentage_l2012_201260

theorem camping_trip_percentage (t : ℕ) (h1 : 22 / 100 * t > 0) (h2 : 75 / 100 * (22 / 100 * t) ≤ t) :
  (88 / 100 * t) = t :=
by
  sorry

end camping_trip_percentage_l2012_201260


namespace distance_after_rest_l2012_201214

-- Define the conditions
def distance_before_rest := 0.75
def total_distance := 1.0

-- State the theorem
theorem distance_after_rest :
  total_distance - distance_before_rest = 0.25 :=
by sorry

end distance_after_rest_l2012_201214


namespace geometric_sum_n_eq_4_l2012_201291

theorem geometric_sum_n_eq_4 :
  ∃ n : ℕ, (n = 4) ∧ 
  ((1 : ℚ) * (1 - (1 / 4 : ℚ) ^ n) / (1 - (1 / 4 : ℚ)) = (85 / 64 : ℚ)) :=
by
  use 4
  simp
  sorry

end geometric_sum_n_eq_4_l2012_201291


namespace Merrill_and_Elliot_have_fewer_marbles_than_Selma_l2012_201289

variable (Merrill_marbles Elliot_marbles Selma_marbles total_marbles fewer_marbles : ℕ)

-- Conditions
def Merrill_has_30_marbles : Merrill_marbles = 30 := by sorry

def Elliot_has_half_of_Merrill's_marbles : Elliot_marbles = Merrill_marbles / 2 := by sorry

def Selma_has_50_marbles : Selma_marbles = 50 := by sorry

def Merrill_and_Elliot_together_total_marbles : total_marbles = Merrill_marbles + Elliot_marbles := by sorry

def number_of_fewer_marbles : fewer_marbles = Selma_marbles - total_marbles := by sorry

-- Goal
theorem Merrill_and_Elliot_have_fewer_marbles_than_Selma :
  fewer_marbles = 5 := by
  sorry

end Merrill_and_Elliot_have_fewer_marbles_than_Selma_l2012_201289


namespace a_range_l2012_201248

open Set

variable (A B : Set Real) (a : Real)

def A_def : Set Real := {x | 3 * x + 1 < 4}
def B_def : Set Real := {x | x - a < 0}
def intersection_eq : A ∩ B = A := sorry

theorem a_range : a ≥ 1 :=
  by
  have hA : A = {x | x < 1} := sorry
  have hB : B = {x | x < a} := sorry
  have h_intersection : (A ∩ B) = A := sorry
  sorry

end a_range_l2012_201248


namespace total_exercise_hours_l2012_201256

theorem total_exercise_hours (natasha_minutes_per_day : ℕ) (natasha_days : ℕ)
  (esteban_minutes_per_day : ℕ) (esteban_days : ℕ)
  (h_n : natasha_minutes_per_day = 30) (h_nd : natasha_days = 7)
  (h_e : esteban_minutes_per_day = 10) (h_ed : esteban_days = 9) :
  (natasha_minutes_per_day * natasha_days + esteban_minutes_per_day * esteban_days) / 60 = 5 :=
by
  sorry

end total_exercise_hours_l2012_201256


namespace ratio_of_tetrahedron_to_cube_volume_l2012_201258

theorem ratio_of_tetrahedron_to_cube_volume (x : ℝ) (hx : 0 < x) :
  let V_cube := x^3
  let a_tetrahedron := (x * Real.sqrt 3) / 2
  let V_tetrahedron := (a_tetrahedron^3 * Real.sqrt 2) / 12
  (V_tetrahedron / V_cube) = (Real.sqrt 6 / 32) :=
by
  sorry

end ratio_of_tetrahedron_to_cube_volume_l2012_201258


namespace independent_sum_of_projections_l2012_201255

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem independent_sum_of_projections (A1 A2 A3 P P1 P2 P3 : ℝ × ℝ) 
  (h_eq_triangle : distance A1 A2 = distance A2 A3 ∧ distance A2 A3 = distance A3 A1)
  (h_proj_P1 : P1 = (P.1, A2.2))
  (h_proj_P2 : P2 = (P.1, A3.2))
  (h_proj_P3 : P3 = (P.1, A1.2)) :
  distance A1 P2 + distance A2 P3 + distance A3 P1 = (3 / 2) * distance A1 A2 := 
sorry

end independent_sum_of_projections_l2012_201255


namespace species_below_threshold_in_year_2019_l2012_201272

-- Definitions based on conditions in the problem.
def initial_species (N : ℝ) : ℝ := N
def yearly_decay_rate : ℝ := 0.70
def threshold : ℝ := 0.05

-- The problem statement to prove.
theorem species_below_threshold_in_year_2019 (N : ℝ) (hN : N > 0):
  ∃ k : ℕ, k ≥ 9 ∧ yearly_decay_rate ^ k * initial_species N < threshold * initial_species N :=
sorry

end species_below_threshold_in_year_2019_l2012_201272


namespace ratio_of_x_y_l2012_201294

theorem ratio_of_x_y (x y : ℚ) (h : (2 * x - y) / (x + y) = 2 / 3) : x / y = 5 / 4 :=
sorry

end ratio_of_x_y_l2012_201294


namespace simplify_expression_l2012_201259

theorem simplify_expression :
  2 * Real.sqrt (1 + Real.sin 8) + Real.sqrt (2 + 2 * Real.cos 8) = -2 * Real.sin 4 - 4 * Real.cos 4 :=
by
  sorry

end simplify_expression_l2012_201259


namespace no_common_real_root_l2012_201216

theorem no_common_real_root (a b : ℚ) : 
  ¬ ∃ (r : ℝ), (r^5 - r - 1 = 0) ∧ (r^2 + a * r + b = 0) :=
by
  sorry

end no_common_real_root_l2012_201216


namespace parametric_circle_eqn_l2012_201290

variables (t x y : ℝ)

theorem parametric_circle_eqn (h1 : y = t * x) (h2 : x^2 + y^2 - 4 * y = 0) :
  x = 4 * t / (1 + t^2) ∧ y = 4 * t^2 / (1 + t^2) :=
by
  sorry

end parametric_circle_eqn_l2012_201290


namespace milton_books_l2012_201236

theorem milton_books (Z B : ℕ) (h1 : B = 4 * Z) (h2 : Z + B = 80) : Z = 16 :=
sorry

end milton_books_l2012_201236


namespace factorization_of_expression_l2012_201271

theorem factorization_of_expression (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) :=
by 
  sorry

end factorization_of_expression_l2012_201271


namespace trigonometric_identity_l2012_201227

variable {α : ℝ}

theorem trigonometric_identity (h : Real.tan α = 3) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5 / 6 :=
by
  sorry

end trigonometric_identity_l2012_201227


namespace possible_m_values_l2012_201296

theorem possible_m_values (m : ℝ) :
  let A := {x : ℝ | mx - 1 = 0}
  let B := {2, 3}
  (A ⊆ B) → (m = 0 ∨ m = 1 / 2 ∨ m = 1 / 3) :=
by
  intro A B h
  sorry

end possible_m_values_l2012_201296


namespace shapeB_is_symmetric_to_original_l2012_201235

-- Assume a simple type to represent our shapes
inductive Shape
| shapeA
| shapeB
| shapeC
| shapeD
| shapeE
| originalShape

-- Define the symmetry condition
def is_symmetric (s1 s2 : Shape) : Prop := sorry  -- this would be the condition to check symmetry

-- The theorem to prove that shapeB is symmetric to the original shape
theorem shapeB_is_symmetric_to_original :
  is_symmetric Shape.shapeB Shape.originalShape :=
sorry

end shapeB_is_symmetric_to_original_l2012_201235


namespace symmetric_point_R_l2012_201285

variable (a b : ℝ) 

def symmetry_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def symmetry_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

theorem symmetric_point_R :
  let M := (a, b)
  let N := symmetry_x M
  let P := symmetry_y N
  let Q := symmetry_x P
  let R := symmetry_y Q
  R = (a, b) := by
  unfold symmetry_x symmetry_y
  sorry

end symmetric_point_R_l2012_201285


namespace necessary_and_sufficient_condition_l2012_201232

theorem necessary_and_sufficient_condition (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : (a + b > a * b) ↔ (a = 1 ∨ b = 1) := 
sorry

end necessary_and_sufficient_condition_l2012_201232


namespace cubed_multiplication_identity_l2012_201267

theorem cubed_multiplication_identity : 3^3 * 6^3 = 5832 := by
  sorry

end cubed_multiplication_identity_l2012_201267


namespace ratio_x_y_l2012_201230

variable (x y : ℝ)

-- Conditions:
-- 1. lengths of pieces
def is_square (x : ℝ) : Prop := ∃ s, x = 4 * s
def is_pentagon (y : ℝ) : Prop := ∃ t, y = 5 * t
def equal_perimeter (x y : ℝ) : Prop := x = y

-- Theorem to prove
theorem ratio_x_y (hx : is_square x) (hy : is_pentagon y) (h_perimeter : equal_perimeter x y) : x / y = 1 :=
by {
  -- Implementation of the proof
  sorry
}

end ratio_x_y_l2012_201230


namespace number_of_packages_l2012_201277

-- Given conditions
def totalMarkers : ℕ := 40
def markersPerPackage : ℕ := 5

-- Theorem: Calculate the number of packages
theorem number_of_packages (totalMarkers: ℕ) (markersPerPackage: ℕ) : totalMarkers / markersPerPackage = 8 :=
by 
  sorry

end number_of_packages_l2012_201277


namespace train_return_time_l2012_201224

open Real

theorem train_return_time
  (C_small : Real := 1.5)
  (C_large : Real := 3)
  (speed : Real := 10)
  (initial_connection : String := "A to C")
  (switch_interval : Real := 1) :
  (126 = 2.1 * 60) :=
sorry

end train_return_time_l2012_201224


namespace total_votes_proof_l2012_201268

noncomputable def total_votes (A : ℝ) (T : ℝ) := 0.40 * T = A
noncomputable def votes_in_favor (A : ℝ) := A + 68
noncomputable def total_votes_calc (T : ℝ) (Favor : ℝ) (A : ℝ) := T = Favor + A

theorem total_votes_proof (A T : ℝ) (Favor : ℝ) 
  (hA : total_votes A T) 
  (hFavor : votes_in_favor A = Favor) 
  (hT : total_votes_calc T Favor A) : 
  T = 340 :=
by
  sorry

end total_votes_proof_l2012_201268


namespace quadrilateral_circumscribed_l2012_201213

structure ConvexQuad (A B C D : Type) := 
  (is_convex : True)
  (P : Type)
  (interior : True)
  (angle_APB_angle_CPD_eq_angle_BPC_angle_DPA : True)
  (angle_PAD_angle_PCD_eq_angle_PAB_angle_PCB : True)
  (angle_PDC_angle_PBC_eq_angle_PDA_angle_PBA : True)

theorem quadrilateral_circumscribed (A B C D : Type) (quad : ConvexQuad A B C D) : True := 
sorry

end quadrilateral_circumscribed_l2012_201213


namespace three_sum_eq_nine_seven_five_l2012_201210

theorem three_sum_eq_nine_seven_five {a b c : ℝ} 
    (h1 : b + c = 15 - 2 * a)
    (h2 : a + c = -10 - 4 * b)
    (h3 : a + b = 8 - 2 * c) : 
    3 * a + 3 * b + 3 * c = 9.75 := 
by
    sorry

end three_sum_eq_nine_seven_five_l2012_201210


namespace Barbara_Mike_ratio_is_one_half_l2012_201241

-- Define the conditions
def Mike_age_current : ℕ := 16
def Mike_age_future : ℕ := 24
def Barbara_age_future : ℕ := 16

-- Define Barbara's current age based on the conditions
def Barbara_age_current : ℕ := Mike_age_current - (Mike_age_future - Barbara_age_future)

-- Define the ratio of Barbara's age to Mike's age
def ratio_Barbara_Mike : ℚ := Barbara_age_current / Mike_age_current

-- Prove that the ratio is 1:2
theorem Barbara_Mike_ratio_is_one_half : ratio_Barbara_Mike = 1 / 2 := by
  sorry

end Barbara_Mike_ratio_is_one_half_l2012_201241


namespace find_x_l2012_201237

def perpendicular_vectors_solution (x : ℝ) : Prop :=
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (3, -2)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 2 / 3

theorem find_x (x : ℝ) : perpendicular_vectors_solution x := sorry

end find_x_l2012_201237


namespace trigonometric_expression_l2012_201279

noncomputable def cosθ (θ : ℝ) := 1 / Real.sqrt 10
noncomputable def sinθ (θ : ℝ) := 3 / Real.sqrt 10
noncomputable def tanθ (θ : ℝ) := 3

theorem trigonometric_expression (θ : ℝ) (h : tanθ θ = 3) :
  (1 + cosθ θ) / sinθ θ + sinθ θ / (1 - cosθ θ) = (10 * Real.sqrt 10 + 10) / 9 := 
  sorry

end trigonometric_expression_l2012_201279


namespace kem_hourly_wage_l2012_201202

theorem kem_hourly_wage (shem_total_earnings: ℝ) (shem_hours_worked: ℝ) (ratio: ℝ)
  (h1: shem_total_earnings = 80)
  (h2: shem_hours_worked = 8)
  (h3: ratio = 2.5) :
  (shem_total_earnings / shem_hours_worked) / ratio = 4 :=
by 
  sorry

end kem_hourly_wage_l2012_201202


namespace translate_function_l2012_201280

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (4 * x + 13 * Real.pi / 12) - 1

theorem translate_function :
  ∀ x : ℝ, f (x) = 2 * Real.sin (4 * x + 13 * Real.pi / 12) - 1 :=
by
  intro x
  sorry

end translate_function_l2012_201280


namespace determine_position_correct_l2012_201295

def determine_position (option : String) : Prop :=
  option = "East longitude 120°, North latitude 30°"

theorem determine_position_correct :
  determine_position "East longitude 120°, North latitude 30°" :=
by
  sorry

end determine_position_correct_l2012_201295


namespace percentage_below_50000_l2012_201273

-- Define all the conditions
def cities_between_50000_and_100000 := 35 -- percentage
def cities_below_20000 := 45 -- percentage
def cities_between_20000_and_50000 := 10 -- percentage
def cities_above_100000 := 10 -- percentage

-- The proof statement
theorem percentage_below_50000 : 
    cities_below_20000 + cities_between_20000_and_50000 = 55 :=
by
    unfold cities_below_20000 cities_between_20000_and_50000
    sorry

end percentage_below_50000_l2012_201273


namespace curve_is_line_l2012_201254

-- Define the polar equation as a condition
def polar_eq (r θ : ℝ) : Prop := r = 2 / (2 * Real.sin θ - Real.cos θ)

-- Define what it means for a curve to be a line
def is_line (x y : ℝ) : Prop := x + 2 * y = 2

-- The main statement to prove
theorem curve_is_line (r θ : ℝ) (x y : ℝ) (hr : polar_eq r θ) (hx : x = r * Real.cos θ) (hy : y = r * Real.sin θ) :
  is_line x y :=
sorry

end curve_is_line_l2012_201254


namespace lcm_of_two_numbers_l2012_201270

-- Define the numbers involved
def a : ℕ := 28
def b : ℕ := 72

-- Define the expected LCM result
def lcm_ab : ℕ := 504

-- State the problem as a theorem
theorem lcm_of_two_numbers : Nat.lcm a b = lcm_ab :=
by sorry

end lcm_of_two_numbers_l2012_201270


namespace k_value_if_perfect_square_l2012_201275

theorem k_value_if_perfect_square (a k : ℝ) (h : ∃ b : ℝ, a^2 + 2*k*a + 1 = (a + b)^2) : k = 1 ∨ k = -1 :=
sorry

end k_value_if_perfect_square_l2012_201275


namespace smallest_positive_integer_congruence_l2012_201200

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, 5 * x ≡ 14 [MOD 31] ∧ 0 < x ∧ x < 31 := 
sorry

end smallest_positive_integer_congruence_l2012_201200


namespace nobody_but_angela_finished_9_problems_l2012_201221

theorem nobody_but_angela_finished_9_problems :
  ∀ (total_problems martha_problems : ℕ)
    (jenna_problems : ℕ → ℕ)
    (mark_problems : ℕ → ℕ),
    total_problems = 20 →
    martha_problems = 2 →
    jenna_problems martha_problems = 4 * martha_problems - 2 →
    mark_problems (jenna_problems martha_problems) = (jenna_problems martha_problems) / 2 →
    total_problems - (martha_problems + jenna_problems martha_problems + mark_problems (jenna_problems martha_problems)) = 9 :=
by
  intros total_problems martha_problems jenna_problems mark_problems h_total h_martha h_jenna h_mark
  sorry

end nobody_but_angela_finished_9_problems_l2012_201221


namespace add_hex_numbers_l2012_201257

theorem add_hex_numbers : (7 * 16^2 + 10 * 16^1 + 3) + (1 * 16^2 + 15 * 16^1 + 4) = 9 * 16^2 + 9 * 16^1 + 7 := by sorry

end add_hex_numbers_l2012_201257


namespace max_traffic_flow_at_v_40_traffic_flow_at_least_10_thousand_l2012_201262

-- Define the traffic flow function
noncomputable def traffic_flow (v : ℝ) : ℝ := 920 * v / (v^2 + 3 * v + 1600)

-- Condition: v > 0
axiom v_pos (v : ℝ) : v > 0 → traffic_flow v ≥ 0

-- Prove that the average speed v = 40 results in the maximum traffic flow y = 920/83 ≈ 11.08
theorem max_traffic_flow_at_v_40 : traffic_flow 40 = 920 / 83 :=
sorry

-- Prove that to ensure the traffic flow is at least 10 thousand vehicles per hour,
-- the average speed v should be in the range [25, 64]
theorem traffic_flow_at_least_10_thousand (v : ℝ) (h : traffic_flow v ≥ 10) : 25 ≤ v ∧ v ≤ 64 :=
sorry

end max_traffic_flow_at_v_40_traffic_flow_at_least_10_thousand_l2012_201262


namespace thabo_total_books_l2012_201287

noncomputable def total_books (H PNF PF : ℕ) : ℕ := H + PNF + PF

theorem thabo_total_books :
  ∀ (H PNF PF : ℕ),
    H = 30 →
    PNF = H + 20 →
    PF = 2 * PNF →
    total_books H PNF PF = 180 :=
by
  intros H PNF PF hH hPNF hPF
  sorry

end thabo_total_books_l2012_201287


namespace solution_l2012_201282

-- Given conditions in the problem
def F (x : ℤ) : ℤ := sorry -- Placeholder for the polynomial with integer coefficients
variables (a : ℕ → ℤ) (m : ℕ)

-- Given that: ∀ n, ∃ k, F(n) is divisible by a(k) for some k in {1, 2, ..., m}
axiom forall_n_exists_k : ∀ n : ℤ, ∃ k : ℕ, k < m ∧ a k ∣ F n

-- Desired conclusion: ∃ k, ∀ n, F(n) is divisible by a(k)
theorem solution : ∃ k : ℕ, k < m ∧ (∀ n : ℤ, a k ∣ F n) :=
sorry

end solution_l2012_201282


namespace chessboard_cover_l2012_201281

open Nat

/-- 
  For an m × n chessboard, after removing any one small square, it can always be completely covered
  with L-shaped tiles if and only if 3 divides (mn - 1) and min(m,n) is not equal to 1, 2, 5 or m=n=2.
-/
theorem chessboard_cover (m n : ℕ) :
  (∃ k : ℕ, 3 * k = m * n - 1) ∧ (min m n ≠ 1 ∧ min m n ≠ 2 ∧ min m n ≠ 5 ∨ m = 2 ∧ n = 2) :=
sorry

end chessboard_cover_l2012_201281


namespace maximum_profit_l2012_201238

def radioactive_marble_problem : ℕ :=
    let total_marbles := 100
    let radioactive_marbles := 1
    let non_radioactive_profit := 1
    let measurement_cost := 1
    let max_profit := 92 
    max_profit

theorem maximum_profit 
    (total_marbles : ℕ := 100) 
    (radioactive_marbles : ℕ := 1) 
    (non_radioactive_profit : ℕ := 1) 
    (measurement_cost : ℕ := 1) :
    radioactive_marble_problem = 92 :=
by sorry

end maximum_profit_l2012_201238


namespace problem_inequality_l2012_201297

variables (a b c : ℝ)
open Real

theorem problem_inequality (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 :=
sorry

end problem_inequality_l2012_201297


namespace problem_condition_l2012_201223

theorem problem_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x : ℝ, f x = 5 * x - 3) → (∀ x : ℝ, |x + 0.4| < b → |f x + 1| < a) ↔ (0 < a ∧ 0 < b ∧ b ≤ a / 5) := by
  sorry

end problem_condition_l2012_201223


namespace function_properties_l2012_201234

variable (f : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, 2 * f x * f y = f (x + y) + f (x - y))
variable (h2 : f 1 = -1)

theorem function_properties :
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f x + f (1 - x) = 0) :=
sorry

end function_properties_l2012_201234


namespace nancy_small_gardens_l2012_201276

theorem nancy_small_gardens (total_seeds big_garden_seeds small_garden_seed_count : ℕ) 
    (h1 : total_seeds = 52) 
    (h2 : big_garden_seeds = 28) 
    (h3 : small_garden_seed_count = 4) : 
    (total_seeds - big_garden_seeds) / small_garden_seed_count = 6 := by 
    sorry

end nancy_small_gardens_l2012_201276


namespace roots_polynomial_sum_l2012_201292

theorem roots_polynomial_sum :
  ∀ (p q r : ℂ), (p^3 - 3*p^2 - p + 3 = 0) ∧ (q^3 - 3*q^2 - q + 3 = 0) ∧ (r^3 - 3*r^2 - r + 3 = 0) →
  (1 / (p - 2) + 1 / (q - 2) + 1 / (r - 2) = 1) :=
by
  intros p q r h
  sorry

end roots_polynomial_sum_l2012_201292


namespace more_perfect_squares_with_7_digit_17th_l2012_201219

noncomputable def seventeenth_digit (n : ℕ) : ℕ :=
  (n / 10^16) % 10

theorem more_perfect_squares_with_7_digit_17th
  (h_bound : ∀ n, n < 10^10 → (n * n) < 10^20)
  (h_representation : ∀ m, m < 10^20 → ∃ n, n < 10^10 ∧ m = n * n) :
  (∃ majority_digit_7 : ℕ,
    (∃ majority_digit_8 : ℕ,
      ∀ n, seventeenth_digit (n * n) = 7 → majority_digit_7 > majority_digit_8)
  ) :=
sorry

end more_perfect_squares_with_7_digit_17th_l2012_201219


namespace John_needs_more_days_l2012_201217

theorem John_needs_more_days (days_worked : ℕ) (amount_earned : ℕ) :
  days_worked = 10 ∧ amount_earned = 250 ∧ 
  (∀ d : ℕ, d < days_worked → amount_earned / days_worked = amount_earned / 10) →
  ∃ more_days : ℕ, more_days = 10 ∧ amount_earned * 2 = (days_worked + more_days) * (amount_earned / days_worked) :=
sorry

end John_needs_more_days_l2012_201217


namespace product_sin_eq_one_eighth_l2012_201288

theorem product_sin_eq_one_eighth (h1 : Real.sin (3 * Real.pi / 8) = Real.cos (Real.pi / 8))
                                  (h2 : Real.sin (Real.pi / 8) = Real.cos (3 * Real.pi / 8)) :
  ((1 - Real.sin (Real.pi / 8)) * (1 - Real.sin (3 * Real.pi / 8)) * 
   (1 + Real.sin (Real.pi / 8)) * (1 + Real.sin (3 * Real.pi / 8)) = 1 / 8) :=
by {
  sorry
}

end product_sin_eq_one_eighth_l2012_201288


namespace exists_triang_and_square_le_50_l2012_201250

def is_triang_num (n : ℕ) : Prop := ∃ m : ℕ, n = m * (m + 1) / 2
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem exists_triang_and_square_le_50 : ∃ n : ℕ, n ≤ 50 ∧ is_triang_num n ∧ is_perfect_square n :=
by
  sorry

end exists_triang_and_square_le_50_l2012_201250


namespace joe_initial_paint_amount_l2012_201218

theorem joe_initial_paint_amount (P : ℝ) 
  (h1 : (2/3) * P + (1/15) * P = 264) : P = 360 :=
sorry

end joe_initial_paint_amount_l2012_201218


namespace ratio_of_pats_stick_not_covered_to_sarah_stick_l2012_201269

-- Defining the given conditions
def pat_stick_length : ℕ := 30
def dirt_covered : ℕ := 7
def jane_stick_length : ℕ := 22
def two_feet : ℕ := 24

-- Computing Sarah's stick length from Jane's stick length and additional two feet
def sarah_stick_length : ℕ := jane_stick_length + two_feet

-- Computing the portion of Pat's stick not covered in dirt
def portion_not_covered_in_dirt : ℕ := pat_stick_length - dirt_covered

-- The statement we need to prove
theorem ratio_of_pats_stick_not_covered_to_sarah_stick : 
  (portion_not_covered_in_dirt : ℚ) / (sarah_stick_length : ℚ) = 1 / 2 := 
by sorry

end ratio_of_pats_stick_not_covered_to_sarah_stick_l2012_201269


namespace triangle_inequality_l2012_201220

variable (a b c : ℝ) -- sides of the triangle
variable (h_a h_b h_c S r R : ℝ) -- heights, area of the triangle, inradius, circumradius

-- Definitions of conditions
axiom h_def : h_a + h_b + h_c = (a + b + c) -- express heights sum in terms of sides sum (for illustrative purposes)
axiom S_def : S = 0.5 * a * h_a  -- area definition (adjust as needed)
axiom r_def : 9 * r ≤ h_a + h_b + h_c -- given in solution
axiom R_def : h_a + h_b + h_c ≤ 9 * R / 2 -- given in solution

theorem triangle_inequality :
  9 * r / (2 * S) ≤ (1 / a) + (1 / b) + (1 / c) ∧ (1 / a) + (1 / b) + (1 / c) ≤ 9 * R / (4 * S) :=
by
  sorry

end triangle_inequality_l2012_201220


namespace negation_of_p_l2012_201253

theorem negation_of_p : (¬ ∃ x : ℕ, x^2 > 4^x) ↔ (∀ x : ℕ, x^2 ≤ 4^x) :=
by
  sorry

end negation_of_p_l2012_201253


namespace length_of_each_piece_l2012_201208

theorem length_of_each_piece :
  ∀ (ribbon_length remaining_length pieces : ℕ),
  ribbon_length = 51 →
  remaining_length = 36 →
  pieces = 100 →
  (ribbon_length - remaining_length) / pieces * 100 = 15 :=
by
  intros ribbon_length remaining_length pieces h1 h2 h3
  sorry

end length_of_each_piece_l2012_201208


namespace find_numbers_l2012_201283

theorem find_numbers (p q x : ℝ) (h : (p ≠ 1)) :
  ((p * x) ^ 2 - x ^ 2) / (p * x + x) = q ↔ x = q / (p - 1) ∧ p * x = (p * q) / (p - 1) := 
by
  sorry

end find_numbers_l2012_201283


namespace largest_part_of_proportional_division_l2012_201228

theorem largest_part_of_proportional_division :
  ∀ (x y z : ℝ),
    x + y + z = 120 ∧
    x / (1 / 2) = y / (1 / 4) ∧
    x / (1 / 2) = z / (1 / 6) →
    max x (max y z) = 60 :=
by sorry

end largest_part_of_proportional_division_l2012_201228


namespace joan_balloons_l2012_201222

-- Defining the condition
def melanie_balloons : ℕ := 41
def total_balloons : ℕ := 81

-- Stating the theorem
theorem joan_balloons :
  ∃ (joan_balloons : ℕ), joan_balloons = total_balloons - melanie_balloons ∧ joan_balloons = 40 :=
by
  -- Placeholder for the proof
  sorry

end joan_balloons_l2012_201222


namespace garden_area_increase_l2012_201201

theorem garden_area_increase :
    let length := 60
    let width := 20
    let perimeter := 2 * (length + width)
    let side_of_square := perimeter / 4
    let area_rectangular := length * width
    let area_square := side_of_square * side_of_square
    area_square - area_rectangular = 400 :=
by
  sorry

end garden_area_increase_l2012_201201


namespace find_m_l2012_201251

def A : Set ℤ := {-1, 1}
def B (m : ℤ) : Set ℤ := {x | m * x = 1}

theorem find_m (m : ℤ) (h : B m ⊆ A) : m = 0 ∨ m = 1 ∨ m = -1 := 
sorry

end find_m_l2012_201251


namespace max_sector_area_l2012_201265

theorem max_sector_area (r θ : ℝ) (S : ℝ) (h_perimeter : 2 * r + θ * r = 16)
  (h_max_area : S = 1 / 2 * θ * r^2) :
  r = 4 ∧ θ = 2 ∧ S = 16 := by
  -- sorry, the proof is expected to go here
  sorry

end max_sector_area_l2012_201265


namespace equilateral_triangle_perimeter_l2012_201240

theorem equilateral_triangle_perimeter (x : ℕ) (h : 2 * x = x + 15) : 
  3 * (2 * x) = 90 :=
by
  -- Definitions & hypothesis
  sorry

end equilateral_triangle_perimeter_l2012_201240


namespace next_meeting_time_l2012_201299

noncomputable def perimeter (AB BC CD DA : ℝ) : ℝ :=
  AB + BC + CD + DA

theorem next_meeting_time 
  (AB BC CD AD : ℝ) 
  (v_human v_dog : ℝ) 
  (initial_meeting_time : ℝ) :
  AB = 100 → BC = 200 → CD = 100 → AD = 200 →
  initial_meeting_time = 2 →
  v_human + v_dog = 300 →
  ∃ next_time : ℝ, next_time = 14 := 
by
  sorry

end next_meeting_time_l2012_201299


namespace defect_rate_product_l2012_201247

theorem defect_rate_product (P1_defect P2_defect : ℝ) (h1 : P1_defect = 0.10) (h2 : P2_defect = 0.03) : 
  ((1 - P1_defect) * (1 - P2_defect)) = 0.873 → (1 - ((1 - P1_defect) * (1 - P2_defect)) = 0.127) :=
by
  intro h
  sorry

end defect_rate_product_l2012_201247


namespace total_fish_l2012_201239

variable (L A : ℕ)

theorem total_fish (h1 : L = 40) (h2 : A = L + 20) : L + A = 100 := by 
  sorry

end total_fish_l2012_201239


namespace max_xy_l2012_201249

theorem max_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 5 * x + 6 * y < 90) :
  xy * (90 - 5 * x - 6 * y) ≤ 900 := by
  sorry

end max_xy_l2012_201249


namespace find_third_number_in_proportion_l2012_201264

theorem find_third_number_in_proportion (x : ℝ) (third_number : ℝ) (h1 : x = 0.9) (h2 : 0.75 / 6 = x / third_number) : third_number = 5 := by
  sorry

end find_third_number_in_proportion_l2012_201264
