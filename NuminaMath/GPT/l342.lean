import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Analysis.Calculus.SpecificFunctions
import Mathlib.Analysis.Probability.ProbabilityMeasure
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Complex
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Fib
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.GroupTheory.Subgroup.Basic
import Mathlib.Probability.MassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Real

namespace final_result_l342_342575

-- Define the operation ⊕
def has_oplus (x y : ℝ) : ℝ := 1 / (x * y)

-- Given assumption a = 2
def a := 2

-- Derived definitions of a ⊕ 1540 and b 
def a_oplus_1540 := has_oplus a 1540
def b := has_oplus 4 a_oplus_1540

-- The theorem to prove
theorem final_result : b = 770 := 
by
  -- Intermediate simplifications based on given definitions.
  -- (Note that the intermediate proof steps are skipped and 'sorry' is used in place)
  sorry

end final_result_l342_342575


namespace map_scale_l342_342266

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l342_342266


namespace map_distance_l342_342339

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l342_342339


namespace function_seven_zeros_l342_342586

theorem function_seven_zeros (ω : ℝ) (h_pos : ω > 0) :
  has_seven_distinct_zeros (λ x : ℝ, if x > 0 then x + abs (Real.log x) - 2 else Real.sin (ω * x + Real.pi / 4) - 1 / 2) ↔
  ω ∈ Set.Ico (49 / 12) (65 / 12) :=
sorry

end function_seven_zeros_l342_342586


namespace limit_cos_pi_x_pow_tan_x_minus_2_l342_342043

open Real

theorem limit_cos_pi_x_pow_tan_x_minus_2 : 
  tendsto (λ x : ℝ, (cos (π * x)) ^ (tan (x - 2))) (𝓝 2) (𝓝 1) := 
by sorry

end limit_cos_pi_x_pow_tan_x_minus_2_l342_342043


namespace probability_divisible_by_5_or_2_l342_342798

theorem probability_divisible_by_5_or_2 :
  let nums := [1, 2, 3, 4, 5]
  let total_permutations := (nums.permutations.length : ℚ)
  let favorable_outcomes := (nums.filter (λ x, x = 2 ∨ x = 4 ∨ x = 5)).length
  (favorable_outcomes / nums.length : ℚ) = 0.6 :=
by
  sorry

end probability_divisible_by_5_or_2_l342_342798


namespace train_speed_l342_342968

theorem train_speed
  (length : ℝ) (time_seconds : ℝ) (distance_km : length = 80 / 1000)
  (time_hours : time_seconds = 1.9998400127989762 / 3600) :
  (length / (time_seconds / 3600)) ≈ 144 :=
by sorry

end train_speed_l342_342968


namespace fraction_white_surface_area_of_larger_cube_l342_342918

theorem fraction_white_surface_area_of_larger_cube :
  (let total_surface_area := 6 * (4^2),
       total_faces_filled_with_black := 3 * 16,
       total_faces_white := total_surface_area - total_faces_filled_with_black
   in total_faces_white / total_surface_area = 1 / 2) :=
  sorry

end fraction_white_surface_area_of_larger_cube_l342_342918


namespace range_of_a_l342_342613

theorem range_of_a (a : ℝ) (h : log a (3/5) < 1) : ((0 < a ∧ a < 3/5) ∨ (1 < a)) := 
sorry

end range_of_a_l342_342613


namespace math_problem_l342_342184

def median (l : List Int) : Int :=
  let sorted_l := l.qsort (· < ·)
  if h : sorted_l.length % 2 = 0 then
    let mid := sorted_l.length / 2
    (sorted_l.get! (mid - 1) + sorted_l.get! mid) / 2
  else
    sorted_l.get! (sorted_l.length / 2)

def mode (l : List Int) : Int :=
  let freq_map := l.foldl (λm x => m.insert x ((m.find x).getD 0 + 1)) (RBMap.empty)
  let max_pair := freq_map.toList.qsort (Ord.compare ·.2 ·.2).reverse.head!
  max_pair.1

def median_of_list_i := 8

theorem math_problem :
  let i := [9, 2, 4, 7, 10, 11]
  let ii := [3, 3, 4, 6, 7, 10]
  median i = median_of_list_i := by
  let med_ii := median ii
  let mod_ii := mode ii
  have h : median_of_list_i = med_ii + mod_ii := sorry
  exact sorry

end math_problem_l342_342184


namespace abs_expression_eq_l342_342562

def ω := 5 + 3 * complex.i

theorem abs_expression_eq : complex.abs (ω^2 + 10 * ω + 40) = 4 * real.sqrt 1066 := 
by 
  sorry

end abs_expression_eq_l342_342562


namespace isosceles_triangle_perimeter_l342_342686

def is_isosceles_triangle (A B C : ℝ) : Prop :=
  (A = B ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (A = C ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (B = C ∧ A + B > C ∧ A + C > B ∧ B + C > A)

theorem isosceles_triangle_perimeter {A B C : ℝ} 
  (h : is_isosceles_triangle A B C) 
  (h1 : A = 3 ∨ A = 7) 
  (h2 : B = 3 ∨ B = 7) 
  (h3 : C = 3 ∨ C = 7) : 
  A + B + C = 17 := 
sorry

end isosceles_triangle_perimeter_l342_342686


namespace num_of_distinct_m_values_l342_342234

theorem num_of_distinct_m_values : 
  (∃ (x1 x2 : ℤ), x1 * x2 = 36 ∧ m = x1 + x2) → 
  (finset.card (finset.image (λ (p : ℤ × ℤ), p.1 + p.2) 
    {p | p.1 * p.2 = 36})) = 10 :=
sorry

end num_of_distinct_m_values_l342_342234


namespace chess_tournament_games_l342_342488

theorem chess_tournament_games (n : ℕ) (h : n = 16) :
  (n * (n - 1) * 2) / 2 = 480 :=
by
  rw [h]
  simp
  norm_num
  sorry

end chess_tournament_games_l342_342488


namespace joanne_collected_in_fourth_hour_l342_342563

structure CoinCollection :=
  (first_hour : ℕ)
  (next_two_hours : ℕ)
  (coins_given : ℕ)
  (coins_after_fourth_hour : ℕ)

def joanne : CoinCollection :=
  { first_hour := 15,
    next_two_hours := 35,
    coins_given := 15,
    coins_after_fourth_hour := 120 }

theorem joanne_collected_in_fourth_hour :
  ∃ coins, joanne.first_hour + 2 * joanne.next_two_hours + coins = joanne.coins_after_fourth_hour + joanne.coins_given ∧ coins = 50 :=
by {
  let total_first_three_hours := joanne.first_hour + 2 * joanne.next_two_hours,
  have h : 120 + 15 = total_first_three_hours + 50,
  {
    -- Numerical calculation can be verified here
    linarith
  },
  use 50,
  split,
  exact h,
  refl
}

end joanne_collected_in_fourth_hour_l342_342563


namespace domain_of_log_l342_342389

def log_domain := {x : ℝ | x > 1}

theorem domain_of_log : {x : ℝ | ∃ y, y = log_domain} = {x : ℝ | x > 1} :=
by
  sorry

end domain_of_log_l342_342389


namespace map_length_representation_l342_342315

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l342_342315


namespace matrix_power_solution_l342_342154

theorem matrix_power_solution (b m : ℕ) :
    let C := !![ [1, 1, b], [0, 1, 3], [0, 0, 1] ]
    let A := !![ [1, 15, 2017], [0, 1, 30], [0, 0, 1] ]
    C^m = A → b + m = 23 :=
by sorry

end matrix_power_solution_l342_342154


namespace second_player_wins_with_optimal_play_l342_342857

-- Definitions based on the conditions provided
structure Grid :=
  (rows : Fin 51)
  (cols : Fin 51)
  (occupied : Fin 51 → Fin 51 → Nat) -- Function to denote number of pieces in a cell, we restrict it to 0, 1, or 2.

def valid_placement (grid : Grid) (r : Fin 51) (c : Fin 51) : Prop :=
  grid.occupied r c < 2 ∧
  (∀ i, (Nat.sum (grid.occupied i)) ≤ 2) ∧
  (∀ j, (Nat.sum (flip grid.occupied j)) ≤ 2)

def player1_cannot_move (grid : Grid) : Prop :=
  ∀ r c, ¬ valid_placement grid r c

def player2_has_symmetric_strategy (grid : Grid) : Prop :=
  ∀ r c, valid_placement grid r c → valid_placement grid (⟨25 - r.val, sorry⟩) (⟨25 - c.val, sorry⟩)

theorem second_player_wins_with_optimal_play : ∃ strategy : Grid → (Fin 51 × Fin 51), 
  ∀ grid, ¬player1_cannot_move grid → player2_has_symmetric_strategy (strategy grid) :=
sorry

end second_player_wins_with_optimal_play_l342_342857


namespace map_length_representation_l342_342275

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l342_342275


namespace elastic_collision_inelastic_collision_l342_342422

-- Given conditions for Case A and Case B
variables (L V : ℝ) (m : ℝ) -- L is length of the rods, V is the speed, m is mass of each sphere

-- Prove Case A: The dumbbells separate maintaining their initial velocities
theorem elastic_collision (h1 : L > 0) (h2 : V > 0) (h3 : m > 0) :
  -- After a perfectly elastic collision, the dumbbells separate maintaining their initial velocities
  true := sorry

-- Prove Case B: The dumbbells start rotating around the collision point with angular velocity V / (2 * L)
theorem inelastic_collision (h1 : L > 0) (h2 : V > 0) (h3 : m > 0) :
  -- After a perfectly inelastic collision, the dumbbells start rotating around the collision point with angular velocity V / (2 * L)
  true := sorry

end elastic_collision_inelastic_collision_l342_342422


namespace fewest_students_l342_342001

theorem fewest_students (n : ℕ) : 
  (n % 6 = 1) ∧ (n % 8 = 3) ∧ (n % 9 = 5) → n = 187 :=
begin
  sorry
end

end fewest_students_l342_342001


namespace yards_after_8_marathons_l342_342006

-- Define the constants and conditions
def marathon_miles := 26
def marathon_yards := 395
def yards_per_mile := 1760

-- Definition for total distance covered after 8 marathons
def total_miles := marathon_miles * 8
def total_yards := marathon_yards * 8

-- Convert the total yards into miles with remainder
def extra_miles := total_yards / yards_per_mile
def remainder_yards := total_yards % yards_per_mile

-- Prove the remainder yards is 1400
theorem yards_after_8_marathons : remainder_yards = 1400 := by
  -- Proof steps would go here
  sorry

end yards_after_8_marathons_l342_342006


namespace spending_ratio_l342_342791

theorem spending_ratio 
  (lisa_tshirts : Real)
  (lisa_jeans : Real)
  (lisa_coats : Real)
  (carly_tshirts : Real)
  (carly_jeans : Real)
  (carly_coats : Real)
  (total_spent : Real)
  (hl1 : lisa_tshirts = 40)
  (hl2 : lisa_jeans = lisa_tshirts / 2)
  (hl3 : lisa_coats = 2 * lisa_tshirts)
  (hc1 : carly_tshirts = lisa_tshirts / 4)
  (hc2 : carly_coats = lisa_coats / 4)
  (htotal : total_spent = lisa_tshirts + lisa_jeans + lisa_coats + carly_tshirts + carly_jeans + carly_coats)
  (h_total_spent_val : total_spent = 230) :
  carly_jeans = 3 * lisa_jeans :=
by
  -- Placeholder for theorem's proof
  sorry

end spending_ratio_l342_342791


namespace sin_double_angle_l342_342707

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_l342_342707


namespace sin_double_angle_l342_342710

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_l342_342710


namespace citizen_income_l342_342172

-- Define the constants for the problem
def tax_rate_first := 0.12
def tax_rate_excess := 0.20
def first_income_threshold := 40000
def total_tax_paid := 8000
def fixed_tax := tax_rate_first * first_income_threshold
def excess_tax (income_over_threshold : ℝ) := tax_rate_excess * income_over_threshold

-- Define the income as a variable
variable (I : ℝ)

-- The condition that the total tax paid is equal to $8,000
def tax_condition := fixed_tax + excess_tax (I - first_income_threshold) = total_tax_paid

-- The goal is to prove that the income is $56,000 given the conditions
theorem citizen_income : tax_condition I → I = 56000 :=
by
  intros h
  -- The proof would go here
  sorry

end citizen_income_l342_342172


namespace find_point_M_l342_342197

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def distance (P Q : Point3D) : ℝ :=
real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)

theorem find_point_M :
  ∃ M : Point3D, M.x = 0 ∧ M.y = 0 ∧ distance M ⟨2, 1, 1⟩ = distance M ⟨1, -3, 2⟩ ∧ M.z = 4 :=
sorry

end find_point_M_l342_342197


namespace binom_1500_1_l342_342996

-- Define binomial coefficient
def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

-- Theorem statement
theorem binom_1500_1 : binom 1500 1 = 1500 :=
by
  sorry

end binom_1500_1_l342_342996


namespace remainder_17_pow_2047_mod_23_l342_342867

theorem remainder_17_pow_2047_mod_23 : (17 ^ 2047) % 23 = 11 := 
by
  sorry

end remainder_17_pow_2047_mod_23_l342_342867


namespace find_constants_l342_342552

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 3
  else if x ≤ 3 then -x + 3
  else 0

def h (x : ℝ) (a b d : ℝ) : ℝ :=
  a * f (b * x) + d

theorem find_constants : (a b d : ℝ) → 
  (∀ x, h x 2 (1/3) 5 = 2 * f (x / 3) + 5) → 
  (a = 2 ∧ b = 1/3 ∧ d = 5) :=
by
  intros a b d h_equation
  sorry

end find_constants_l342_342552


namespace sqrt_three_irrational_l342_342878

theorem sqrt_three_irrational : ¬ (∃ (p q : ℤ), q ≠ 0 ∧ (sqrt 3 : ℝ) = p / q) := 
by
  sorry

end sqrt_three_irrational_l342_342878


namespace five_b_value_l342_342161

theorem five_b_value (a b : ℚ) (h1 : 3 * a + 4 * b = 0) (h2 : a = b - 3) : 5 * b = 45 / 7 :=
by
  sorry

end five_b_value_l342_342161


namespace smallest_three_digit_palindromic_prime_with_2s_l342_342449

def is_palindrome (n : ℕ) : Prop :=
  let s := toDigits 10 n in s = s.reverse

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ x, x ∣ n → x = 1 ∨ x = n)

theorem smallest_three_digit_palindromic_prime_with_2s (n : ℕ) :
  100 ≤ n ∧ n < 1000 ∧ is_palindrome n ∧ is_prime n ∧
  ((n / 100) = 2) ∧ ((n % 10) = 2) ↔ n = 202 :=
sorry

end smallest_three_digit_palindromic_prime_with_2s_l342_342449


namespace map_representation_l342_342257

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l342_342257


namespace xyz_plus_54_l342_342158

theorem xyz_plus_54 (x y z : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x * y + z = 53) (h2 : y * z + x = 53) (h3 : z * x + y = 53) : 
  x + y + z = 54 := by
  sorry

end xyz_plus_54_l342_342158


namespace employee_discount_percentage_l342_342916

theorem employee_discount_percentage:
  let purchase_price := 500
  let markup_percentage := 0.15
  let savings := 57.5
  let retail_price := purchase_price * (1 + markup_percentage)
  let discount_percentage := (savings / retail_price) * 100
  discount_percentage = 10 :=
by
  sorry

end employee_discount_percentage_l342_342916


namespace inequality_problem_l342_342766

variable {R : Type*} [LinearOrderedField R]

theorem inequality_problem
  (a b : R) 
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hab : a + b = 1) :
  (a + 1 / a)^2 + (b + 1 / b)^2 ≥ 25 / 2 := 
sorry

end inequality_problem_l342_342766


namespace distinct_m_count_l342_342237

noncomputable def countDistinctMValues : Nat :=
  let pairs := [(1, 36), (2, 18), (3, 12), (4, 9), (6, 6), 
                (-1, -36), (-2, -18), (-3, -12), (-4, -9), (-6, -6)]
  let ms := pairs.map (λ p => p.1 + p.2)
  ms.eraseDups.length

theorem distinct_m_count :
  countDistinctMValues = 10 := sorry

end distinct_m_count_l342_342237


namespace weaving_increase_each_day_l342_342822

/-- 
There is a woman skilled in weaving who weaves faster day by day. Starting from the second day,
she weaves an additional constant amount of fabric each day compared to the previous day. 
If she weaves 5 meters of fabric on the first day and a total of 390 meters of fabric in a month 
(considering a month as 30 days), then the additional amount of fabric she weaves each day 
compared to the previous day is 16/29 meters. 
-/
theorem weaving_increase_each_day (d : ℚ) :
  (∑ i in finset.range 30, (5 + i * d)) = 390 → d = 16 / 29 :=
by
  sorry

end weaving_increase_each_day_l342_342822


namespace length_of_MC_l342_342692

variables (A B C D E M : Type)
variables (rectangle_ABCD : Rectangle A B C D)
variables (point_E_on_AC : Point E A C)
variables (point_M_on_BC : Point M B C)
variables (BC_eq_EC : BC = EC)
variables (EM_eq_MC : EM = MC)
variables (BM_eq_5 : BM = 5)
variables (AE_eq_2 : AE = 2)

theorem length_of_MC : MC = 7 := sorry

end length_of_MC_l342_342692


namespace gravitational_force_solution_l342_342393

noncomputable def gravitational_force_proportionality (d d' : ℕ) (f f' k : ℝ) : Prop :=
  (f * (d:ℝ)^2 = k) ∧
  d = 6000 ∧
  f = 800 ∧
  d' = 36000 ∧
  f' * (d':ℝ)^2 = k

theorem gravitational_force_solution : ∃ k, gravitational_force_proportionality 6000 36000 800 (1/45) k :=
by
  sorry

end gravitational_force_solution_l342_342393


namespace map_representation_l342_342293

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l342_342293


namespace find_sam_current_age_l342_342687

def Drew_current_age : ℕ := 12

def Drew_age_in_five_years : ℕ := Drew_current_age + 5

def Sam_age_in_five_years : ℕ := 3 * Drew_age_in_five_years

def Sam_current_age : ℕ := Sam_age_in_five_years - 5

theorem find_sam_current_age : Sam_current_age = 46 := by
  sorry

end find_sam_current_age_l342_342687


namespace map_representation_l342_342284

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l342_342284


namespace map_distance_l342_342347

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l342_342347


namespace island_count_l342_342699

-- Defining the conditions
def lakes := 7
def canals := 10

-- Euler's formula for connected planar graph
def euler_characteristic (V E F : ℕ) := V - E + F = 2

-- Determine the number of faces using Euler's formula
def faces (V E : ℕ) :=
  let F := V - E + 2
  F

-- The number of islands is the number of faces minus one for the outer face
def number_of_islands (F : ℕ) :=
  F - 1

-- The given proof problem to be converted to Lean
theorem island_count :
  number_of_islands (faces lakes canals) = 4 :=
by
  unfold lakes canals faces number_of_islands
  sorry

end island_count_l342_342699


namespace find_x_from_w_condition_l342_342467

theorem find_x_from_w_condition :
  ∀ (x u y z w : ℕ), 
  (x = u + 7) → 
  (u = y + 5) → 
  (y = z + 12) → 
  (z = w + 25) → 
  (w = 100) → 
  x = 149 :=
by intros x u y z w h1 h2 h3 h4 h5
   sorry

end find_x_from_w_condition_l342_342467


namespace employees_in_january_l342_342475

theorem employees_in_january
  (january_employees december_employees : ℕ)
  (h1 : december_employees = january_employees + (0.15 * january_employees))
  (h2 : december_employees = 480) :
  january_employees = 417 :=
by sorry

end employees_in_january_l342_342475


namespace least_five_digit_congruent_to_6_mod_17_l342_342439

theorem least_five_digit_congruent_to_6_mod_17 :
  ∃ (x : ℕ), 10000 ≤ x ∧ x ≤ 99999 ∧ x % 17 = 6 ∧
  ∀ (y : ℕ), 10000 ≤ y ∧ y ≤ 99999 ∧ y % 17 = 6 → x ≤ y := 
sorry

end least_five_digit_congruent_to_6_mod_17_l342_342439


namespace number_of_sides_l342_342954

-- Define the conditions as variables/constants
def exterior_angle (n : ℕ) : ℝ := 18         -- Each exterior angle is 18 degrees
def sum_of_exterior_angles : ℝ := 360        -- Sum of exterior angles of any polygon is 360 degrees

-- Prove the number of sides is equal to 20 given the conditions
theorem number_of_sides : 
  ∃ n : ℕ, (exterior_angle n) * (n : ℝ) = sum_of_exterior_angles → n = 20 := 
by
  sorry

end number_of_sides_l342_342954


namespace map_length_scale_l342_342299

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l342_342299


namespace determine_a_b_l342_342893

-- Step d) The Lean 4 statement for the transformed problem
theorem determine_a_b (a b : ℝ) (h : ∀ t : ℝ, (t^2 + t + 1) * 1^2 - 2 * (a + t)^2 * 1 + t^2 + 3 * a * t + b = 0) : 
  a = 1 ∧ b = 1 := 
sorry

end determine_a_b_l342_342893


namespace solution_to_inverse_eq_zero_l342_342223

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := 2 / (a * x + b)

-- Define the condition that a and b are nonzero constants
axiom a_ne_zero (a : ℝ) : a ≠ 0
axiom b_ne_zero (b : ℝ) : b ≠ 0

-- Define the proposition to be proven
theorem solution_to_inverse_eq_zero (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : 
  let f_inv_0_solution := 2 / b in
  f_inv_0_solution = f 0 := sorry

end solution_to_inverse_eq_zero_l342_342223


namespace suitable_communication_l342_342843

def is_suitable_to_communicate (beijing_time : Nat) (sydney_difference : Int) (los_angeles_difference : Int) : Bool :=
  let sydney_time := beijing_time + sydney_difference
  let los_angeles_time := beijing_time - los_angeles_difference
  sydney_time >= 8 ∧ sydney_time <= 22 -- let's assume suitable time is between 8:00 to 22:00

theorem suitable_communication:
  let beijing_time := 18
  let sydney_difference := 2
  let los_angeles_difference := 15
  is_suitable_to_communicate beijing_time sydney_difference los_angeles_difference = true :=
by
  sorry

end suitable_communication_l342_342843


namespace course_choice_gender_related_l342_342413
open scoped Real

theorem course_choice_gender_related :
  let a := 40 -- Males choosing Calligraphy
  let b := 10 -- Males choosing Paper Cutting
  let c := 30 -- Females choosing Calligraphy
  let d := 20 -- Females choosing Paper Cutting
  let n := a + b + c + d -- Total number of students
  let χ_squared := (n * (a*d - b*c)^2) / ((a+b) * (c+d) * (a+c) * (b+d))
  χ_squared > 3.841 := 
by
  sorry

end course_choice_gender_related_l342_342413


namespace recurring_decimal_sum_is_13_over_33_l342_342066

noncomputable def recurring_decimal_sum : ℚ :=
  let x := 1/3 -- 0.\overline{3}
  let y := 2/33 -- 0.\overline{06}
  x + y

theorem recurring_decimal_sum_is_13_over_33 : recurring_decimal_sum = 13/33 := by
  sorry

end recurring_decimal_sum_is_13_over_33_l342_342066


namespace geo_seq_fifth_term_l342_342113

theorem geo_seq_fifth_term (a : ℕ → ℝ) (q : ℝ) (h1 : q = 2) (h2 : a 3 = 3) :
  a 5 = 12 := 
sorry

end geo_seq_fifth_term_l342_342113


namespace map_length_scale_l342_342301

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l342_342301


namespace total_amount_in_euros_l342_342518

theorem total_amount_in_euros (w x y z: ℝ) (w_share_in_euros: ℝ) (exchange_rate: ℝ) (h1: x = 0.75 * w) (h2: y = 0.50 * w) (h3: z = 0.25 * w) (h4: w_share_in_euros = 15) (h5: exchange_rate = 0.85) :
  let w_dollars := w_share_in_euros / exchange_rate in
  let total_dollars := w_dollars + x + y + z in
  total_dollars * exchange_rate = 37.5 :=
by
  sorry

end total_amount_in_euros_l342_342518


namespace distance_between_x_intercepts_l342_342511

/-- Prove that the distance between the x-intercepts of two lines given their equations. -/
theorem distance_between_x_intercepts : 
  ∀ (slope1 slope2 y_intercept : ℝ), 
  slope1 = 2 → slope2 = -4 → y_intercept = 6 → 
  let x_intercept1 := -y_intercept / slope1,
      x_intercept2 := -y_intercept / slope2 in
  abs (x_intercept1 - x_intercept2) = 9 / 2 :=
begin
  intros slope1 slope2 y_intercept h_slope1 h_slope2 h_y_intercept,
  have x_intercept1_def : x_intercept1 = -y_intercept / slope1 := by auto [x_intercept1],
  have x_intercept2_def : x_intercept2 = -y_intercept / slope2 := by auto [x_intercept2],
  sorry, -- Proof goes here
end

end distance_between_x_intercepts_l342_342511


namespace intersection_length_of_sphere_and_tetrahedron_l342_342671

theorem intersection_length_of_sphere_and_tetrahedron (O : Point) (R : ℝ) (r : ℝ) (edge_length : ℝ) : 
  radius = √2 ∧ edge_length = 2*√6 ∧ R = √3 →
  total_length = 8*√2*π :=
by
  sorry

end intersection_length_of_sphere_and_tetrahedron_l342_342671


namespace sin_2phi_l342_342729

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end sin_2phi_l342_342729


namespace find_triples_l342_342073

theorem find_triples (a b c : ℕ) (h₁ : a ≥ b) (h₂ : b ≥ c) (h₃ : a^3 + 9 * b^2 + 9 * c + 7 = 1997) :
  (a = 10 ∧ b = 10 ∧ c = 10) :=
by sorry

end find_triples_l342_342073


namespace log_base_2_of_14_l342_342612

variables (a b : ℝ)

-- Given that log base 2 of 3 is a and log base 3 of 7 is b
def log_base_2_of_3_is_a := log 2 3 = a
def log_base_3_of_7_is_b := log 3 7 = b

-- Prove that log base 2 of 14 is 1 + ab
theorem log_base_2_of_14 (h₁ : log_base_2_of_3_is_a a) (h₂ : log_base_3_of_7_is_b b) : log 2 14 = 1 + a * b :=
by
  have h₃ : log 2 7 = a * b := sorry
  -- use intermediate steps to prove the theorem 
  sorry

end log_base_2_of_14_l342_342612


namespace map_representation_l342_342291

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l342_342291


namespace expression_C_is_fraction_l342_342532

def is_fraction (expr : ℕ → Prop) : Prop :=
  ∃ x, ∃ y, expr x ∧ (y ≠ 0 ∧ y = x + 1)

theorem expression_C_is_fraction : 
  is_fraction (λ x, x / (x + 1) = x / (x + 1)) :=
sorry

end expression_C_is_fraction_l342_342532


namespace number_of_nonnegative_solutions_l342_342653

theorem number_of_nonnegative_solutions : 
    ∃! (x : ℝ), x^2 + 5 * x = 0 ∧ x ≥ 0 :=
begin
  sorry
end

end number_of_nonnegative_solutions_l342_342653


namespace system_of_two_linear_equations_l342_342470

theorem system_of_two_linear_equations :
  ((∃ x y z, x + z = 5 ∧ x - 2 * y = 6) → False) ∧
  ((∃ x y, x * y = 5 ∧ x - 4 * y = 2) → False) ∧
  ((∃ x y, x + y = 5 ∧ 3 * x - 4 * y = 12) → True) ∧
  ((∃ x y, x^2 + y = 2 ∧ x - y = 9) → False) :=
by {
  sorry
}

end system_of_two_linear_equations_l342_342470


namespace range_frequency_025_l342_342638

def sample_data : List ℕ := [12, 7, 11, 12, 11, 12, 10, 10, 9, 8, 13, 12, 10, 9, 6, 11, 8, 9, 8, 10]

def total_samples : ℕ := 20

def frequency (n : ℕ) : ℕ -> ℚ
| x => x / total_samples

theorem range_frequency_025 :
  ∃ lower upper : ℚ, (lower, upper) = (11.5, 13.5) ∧ frequency sample_data.count (λ x, 11.5 ≤ x ∧ x < 13.5) = 0.25 := sorry

end range_frequency_025_l342_342638


namespace sequence_operations_correct_l342_342891

noncomputable def x_value_makes_operations_correct : ℤ :=
  let x := 4 in
  let operation1 (x : ℤ) : ℤ := (x + 2) / 2
  let operation2 (y : ℤ) : ℤ := (y + 2) / 2 + 1
  let final_value := operation2 (operation1 x)
  final_value

theorem sequence_operations_correct (x : ℤ) (h : x = 4) :
  x = (x + 2) / 2 + 1 ↔ x = 4 :=
by
  sorry

end sequence_operations_correct_l342_342891


namespace kim_total_distance_traveled_l342_342978

-- Definitions based on given conditions
def column_length := 1 -- km
def infantry_march_distance := 4 / 3 -- km
def column_speed := 0.5 -- km/h, arbitrary for defining k later

-- Factor by which Kim’s speed is greater
def speed_factor : ℝ := 2 -- As derived in the solution, k = 2

-- Speed of Kim
def kim_speed := speed_factor * column_speed

theorem kim_total_distance_traveled :
  kim_speed * (infantry_march_distance / column_speed) = 8 / 3 :=
sorry

end kim_total_distance_traveled_l342_342978


namespace sin_double_angle_l342_342709

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_l342_342709


namespace sum_of_first_4_terms_arithmetic_sequence_l342_342695

theorem sum_of_first_4_terms_arithmetic_sequence (a : ℕ → ℤ) 
  (h1 : a 1 = 1) (h4 : a 4 = 7) (h_arith : ∀ n, a (n + 1) = a n + d) :
  (1 / 2 : ℝ) * (4 * ((a 1) + (a 4))) = 16 :=
by 
  norm_num
  rw [h1, h4]
  norm_num
  sorry

end sum_of_first_4_terms_arithmetic_sequence_l342_342695


namespace least_five_digit_congruent_6_mod_17_l342_342442

theorem least_five_digit_congruent_6_mod_17 : ∃ n: ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 6 ∧ ∀ m: ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 17 = 6 → n ≤ m :=
sorry

end least_five_digit_congruent_6_mod_17_l342_342442


namespace map_distance_l342_342348

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l342_342348


namespace find_original_denominator_l342_342964

variable (d : ℕ)

theorem find_original_denominator
  (h1 : ∀ n : ℕ, n = 3)
  (h2 : 3 + 7 = 10)
  (h3 : (10 : ℕ) = 1 * (d + 7) / 3) :
  d = 23 := by
  sorry

end find_original_denominator_l342_342964


namespace stack_of_logs_total_l342_342016

-- Definition of the problem based on given conditions
def total_logs (bottom_row : ℕ) (top_row : ℕ) (decrement : ℕ) : ℕ :=
  let number_of_terms := bottom_row - top_row + 1 in
  let average_terms := (top_row + bottom_row) / 2 in
  number_of_terms * average_terms

-- The main statement that needs to be proved
theorem stack_of_logs_total :
  total_logs 15 5 1 = 110 :=
by
  -- A placeholder for the proof to ensure the theorem statement is correct
  sorry

end stack_of_logs_total_l342_342016


namespace ratio_gold_to_green_horses_l342_342385

theorem ratio_gold_to_green_horses (blue_horses purple_horses green_horses gold_horses : ℕ)
    (h1 : blue_horses = 3)
    (h2 : purple_horses = 3 * blue_horses)
    (h3 : green_horses = 2 * purple_horses)
    (h4 : blue_horses + purple_horses + green_horses + gold_horses = 33) :
  gold_horses / gcd gold_horses green_horses = 1 / 6 :=
by
  sorry

end ratio_gold_to_green_horses_l342_342385


namespace original_volume_l342_342973

variable (V : ℝ)

theorem original_volume (h1 : (1/4) * V = V₁)
                       (h2 : (1/4) * V₁ = V₂)
                       (h3 : (1/3) * V₂ = 0.4) : 
                       V = 19.2 := 
by 
  sorry

end original_volume_l342_342973


namespace measure_of_angle_y_l342_342191

variables {m n : ℝ}
variables {y : ℝ}

/- Define the angles adjacent to line m -/
def angle_adjacent_m_left := 40
def angle_adjacent_m_right := 120
def angle_opposite_m_right := 180 - angle_adjacent_m_right

/- Parallel lines condition -/
def parallel (a b : ℝ) := ∀ (x y : ℝ), a = b

/- Given the parallel condition and given angles, prove that angle y is 80 degrees -/
theorem measure_of_angle_y (h_parallel: parallel m n) (h1: angle_adjacent_m_left = 40) (h2: angle_adjacent_m_right = 120):
  y = 180 - (angle_adjacent_m_left + angle_opposite_m_right) :=
begin
  sorry
end

end measure_of_angle_y_l342_342191


namespace sin_2phi_l342_342737

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_2phi_l342_342737


namespace median_unchanged_after_removal_l342_342789

open List

theorem median_unchanged_after_removal (scores : List ℝ) (h : scores.length = 9) :
  let sorted_scores := scores.sort (≤),
      median := sorted_scores.nth_le (sorted_scores.length / 2) (by simp [h])
  in 
    let new_scores := sorted_scores.drop 1 |>.init
    in new_scores.nth_le (new_scores.length / 2) (by simp [h]) = median :=
by sorry

end median_unchanged_after_removal_l342_342789


namespace probability_of_including_seed_l342_342668

-- Define the volume of the batch and the volume of the sample
def batch_volume : ℝ := 2000 -- in mL, since 2L = 2000 mL
def sample_volume : ℝ := 10 -- in mL

-- Define the event A: the sample includes the seed with powdery mildew
def event_A (sample_volume batch_volume : ℝ) : Prop :=
  (sample_volume / batch_volume) = 1 / 200

-- Define the probability function
def probability_of_event_A : ℝ :=
  sample_volume / batch_volume

-- State the theorem: the probability of event A is 1/200
theorem probability_of_including_seed (h1 : sample_volume = 10) (h2 : batch_volume = 2000) :
  probability_of_event_A = 1 / 200 :=
by
  sorry

end probability_of_including_seed_l342_342668


namespace estimated_shadow_area_l342_342035

theorem estimated_shadow_area (total_area : ℝ) (total_beans : ℕ) (beans_outside : ℕ) 
  (h1 : total_area = 10)
  (h2 : total_beans = 200) 
  (h3 : beans_outside = 114) 
: (total_area - (beans_outside / total_beans) * total_area) ≈ 4.3 :=
  sorry

end estimated_shadow_area_l342_342035


namespace varphi_value_l342_342831

theorem varphi_value (varphi : ℝ) (k : ℤ) (hp : -Real.pi ≤ varphi ∧ varphi ≤ Real.pi)
  (h_translation : ∀ x, Math.cos (2 * x + varphi + Real.pi) = Math.cos (2 * x - Real.pi / 6)) :
  varphi = 5 * Real.pi / 6 := 
by
  -- Sorry is used to skip the actual proof
  sorry

end varphi_value_l342_342831


namespace distribute_pencils_l342_342183

theorem distribute_pencils (number_of_pencils : ℕ) (number_of_people : ℕ)
  (h_pencils : number_of_pencils = 2) (h_people : number_of_people = 5) :
  number_of_distributions = 15 := by
  sorry

end distribute_pencils_l342_342183


namespace variety_show_team_combinations_l342_342914

theorem variety_show_team_combinations :
  let male_guests := 3
  let female_guests := 2
  let total_guests := male_guests + female_guests
  let team_size := 3
  -- Calculate the total number of ways to choose any 3 guests from 5
  let total_combinations := Nat.choose total_guests team_size
  -- Calculate the number of ways to choose 3 male guests (which is invalid under given conditions)
  let invalid_combinations := Nat.choose male_guests team_size
  -- The valid combinations are the total combinations minus the invalid ones
  total_combinations - invalid_combinations = 9 := 
by
  have male_guests := 3
  have female_guests := 2
  have total_guests := male_guests + female_guests
  have team_size := 3
  -- Calculate total combinations
  have total_combinations := Nat.choose total_guests team_size
  -- Calculate invalid all-male team combinations
  have invalid_combinations := Nat.choose male_guests team_size
  have valid_combinations := total_combinations - invalid_combinations
  -- The correct answer should be 9
  show valid_combinations = 9 from sorry

end variety_show_team_combinations_l342_342914


namespace variance_shifted_l342_342598

variable {n : ℕ} (x : Fin n → ℝ)

def variance (x : Fin n → ℝ) : ℝ :=
  let mean := (∑ i, x i) / n
  (∑ i, (x i - mean)^2) / n

theorem variance_shifted {n : ℕ} (x : Fin n → ℝ) (h : variance x = 7) :
  variance (λ i => x i - 1) = 7 :=
by
  sorry

end variance_shifted_l342_342598


namespace a_5_is_9_l342_342594

-- Definition of the sequence sum S_n
def S : ℕ → ℕ
| n => n^2 - 1

-- Define the specific term in the sequence
def a (n : ℕ) :=
  if n = 1 then S 1 else S n - S (n - 1)

-- Theorem to prove
theorem a_5_is_9 : a 5 = 9 :=
sorry

end a_5_is_9_l342_342594


namespace map_scale_representation_l342_342330

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l342_342330


namespace constant_term_of_expansion_l342_342624

theorem constant_term_of_expansion (a : ℕ) (h : (a + 2) * 2^5 = 96) : 
  let expansion_sum_coef := (1 + x + a * x^3) * (x + 1/x)^5,
      constant_term := (binom 5 3) + (binom 5 4) 
  in constant_term = 15 :=
by
  -- Step of identifying the value of a from h (here, a is solved to be 1)
  let a := 1
  
  -- Proof that the constant term is 15 based on expansion
  let expansion_sum_coef := (1 + x + x^3) * (x + 1/x)^5
  let constant_term := (binom 5 3) + (binom 5 4)

  have h_constant_term : constant_term = 15, by {sorry},
  exact h_constant_term

end constant_term_of_expansion_l342_342624


namespace ellipse_line_slope_ratio_l342_342390

theorem ellipse_line_slope_ratio (a b : ℝ) (hab: a > 0) (hbb: b > 0) (hne: a ≠ b) 
  (mid_slope: (a * (2 * (1 - 2 * ((4*b) / (a + 4*b))) / (2*b) = sqrt 3) / 2)) : 
  a / b = sqrt 3 :=
sorry

end ellipse_line_slope_ratio_l342_342390


namespace option_A_option_B_option_D_l342_342114

noncomputable theory

variables {R : Type*} [Real R] {f : R → R} (f' : R → R)

-- Given conditions
def domain_f : Prop := ∀ x : R, f x ∈ R
def derivative_f : Prop := ∀ x : R, deriv f x = f' x
def functional_eq : Prop := ∀ x : R, f (4 + x) = f (-x)
def odd_function : Prop := ∀ x : R, f (-x + 1) = -f (x + 1)
def derivative_at_1 : Prop := f' 1 = -1

-- Theorem statements
theorem option_A (h1 : domain_f) (h2 : derivative_f) (h3 : functional_eq) (h4 : odd_function) (h5 : derivative_at_1) : f 1 = 0 := sorry
theorem option_B (h1 : domain_f) (h2 : derivative_f) (h3 : functional_eq) (h4 : odd_function) (h5 : derivative_at_1) : ∀ x : R, f (x + 4) = f x := sorry
theorem option_D (h1 : domain_f) (h2 : derivative_f) (h3 : functional_eq) (h4 : odd_function) (h5 : derivative_at_1) : f' 2023 = 1 := sorry

end option_A_option_B_option_D_l342_342114


namespace coeff_linear_term_quadratic_l342_342386

theorem coeff_linear_term_quadratic (x : ℝ) : 
  ∃ a b c : ℝ, (a ≠ 0) ∧ (b = -1) ∧ (c = 0) ∧ (a * x^2 + b * x + c = x^2 - x) :=
begin
  use [1, -1, 0],
  split,
  { exact one_ne_zero, },
  split,
  { refl, },
  split,
  { refl, },
  { simp, },
end

end coeff_linear_term_quadratic_l342_342386


namespace polynomial_q_value_l342_342057

theorem polynomial_q_value :
  ∀ (p q d : ℝ),
    (d = 6) →
    (-p / 3 = -d) →
    (1 + p + q + d = - d) →
    q = -31 :=
by sorry

end polynomial_q_value_l342_342057


namespace number_of_distinct_m_values_l342_342229

theorem number_of_distinct_m_values (m : ℤ) :
  (∃ x1 x2 : ℤ, x1 * x2 = 36 ∧ x1 + x2 = m) →
  set.card {m | ∃ x1 x2 : ℤ, x1 * x2 = 36 ∧ x1 + x2 = m} = 10 :=
by
  sorry

end number_of_distinct_m_values_l342_342229


namespace any_number_of_inhabitants_l342_342800

def is_vegetarian (person : ℕ) : Prop := sorry
def is_prime (n : ℕ) : Prop := sorry
def standing_in_prime_distance (lineup : ℕ → ℕ) (person : ℕ) : Prop := 
  ∀ v, is_vegetarian v → is_prime (lineup v - lineup person)

theorem any_number_of_inhabitants (n : ℕ) : 
  ∃ (lineup : ℕ → ℕ), 
  (∀ i, i ∈ (finset.range n) → (is_vegetarian i ∨ ¬ is_vegetarian i)) ∧
  (∀ i, i ∈ (finset.range n) → is_vegetarian i → standing_in_prime_distance lineup i) ∧
  (∀ j, j ∈ (finset.range n) → ¬ is_vegetarian j → ¬ standing_in_prime_distance lineup j) :=
sorry

end any_number_of_inhabitants_l342_342800


namespace sin_double_angle_l342_342713

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l342_342713


namespace sin_2phi_l342_342739

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_2phi_l342_342739


namespace mouse_jump_vs_grasshopper_l342_342392

-- Definitions for jumps
def grasshopper_jump : ℕ := 14
def frog_jump : ℕ := grasshopper_jump + 37
def mouse_jump : ℕ := frog_jump - 16

-- Theorem stating the result
theorem mouse_jump_vs_grasshopper : mouse_jump - grasshopper_jump = 21 :=
by
  -- Skip the proof
  sorry

end mouse_jump_vs_grasshopper_l342_342392


namespace painters_work_days_l342_342203

noncomputable def job_constant (number_painters : ℝ) (work_days : ℝ) : ℝ :=
number_painters * work_days

theorem painters_work_days :
  (job_constant 4 1.25) = 5 →
  ∀ (D : ℝ), (job_constant 3 D = 5) → D = 5 / 3 :=
by
  intro h_const
  intro D h_eq
  calc
    D = 5 / 3 : by sorry

end painters_work_days_l342_342203


namespace number_of_sides_l342_342955

-- Define the conditions as variables/constants
def exterior_angle (n : ℕ) : ℝ := 18         -- Each exterior angle is 18 degrees
def sum_of_exterior_angles : ℝ := 360        -- Sum of exterior angles of any polygon is 360 degrees

-- Prove the number of sides is equal to 20 given the conditions
theorem number_of_sides : 
  ∃ n : ℕ, (exterior_angle n) * (n : ℝ) = sum_of_exterior_angles → n = 20 := 
by
  sorry

end number_of_sides_l342_342955


namespace map_scale_l342_342272

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l342_342272


namespace f_f_5_eq_1_l342_342135

def f (x : ℕ) : ℕ :=
if x < 3 then x^2 else f (x - 2)

theorem f_f_5_eq_1 : f(f(5)) = 1 := 
by 
  sorry 

end f_f_5_eq_1_l342_342135


namespace tan_sin_sum_lt_zero_l342_342617

theorem tan_sin_sum_lt_zero {α : ℝ} (hα : α ∈ set.Ioo π (3 * π / 2)) : 
  let cos_alpha := Real.cos α
  let sin_alpha := Real.sin α
  let tan_alpha := Real.tan α
  (tan_alpha + sin_alpha) < 0 :=
by
  sorry

end tan_sin_sum_lt_zero_l342_342617


namespace linear_function_change_l342_342224

-- Define a linear function g
variable (g : ℝ → ℝ)

-- Define and assume the conditions
def linear_function (g : ℝ → ℝ) : Prop := ∀ x y, g (x + y) = g x + g y ∧ g (x - y) = g x - g y
def condition_g_at_points : Prop := g 3 - g (-1) = 20

-- Prove that g(10) - g(2) = 40
theorem linear_function_change (g : ℝ → ℝ) 
  (linear_g : linear_function g) 
  (cond_g : condition_g_at_points g) : 
  g 10 - g 2 = 40 :=
sorry

end linear_function_change_l342_342224


namespace problem1_problem2_l342_342084

-- Proof Problem 1
theorem problem1 (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  (a ^ (2/3) * b ^ (1/2)) / (a ^ (-1/2) * b ^ (1/3)) / ((a ^ (-1) * b ^ (-1/2)) / (b * a ^ (1/2))) ^ (-2/3) = 
  a ^ (1/6) * b ^ (-5/6) :=
  sorry

-- Proof Problem 2
theorem problem2 : 
  log 3 (3 ^ (3/4) / 3) * log 5 (4 ^ (1/2 * log 2 10) - (3 * sqrt 3) ^ (2/3) - 7 ^ log 7 2) = -1 / 4 :=
  sorry

end problem1_problem2_l342_342084


namespace interval_for_f_l342_342164

noncomputable def f (x : ℝ) : ℝ :=
-0.5 * x ^ 2 + 13 / 2

theorem interval_for_f (a b : ℝ) :
  f a = 2 * b ∧ f b = 2 * a ∧ (a ≤ 0 ∨ 0 ≤ b) → 
  ([a, b] = [1, 3] ∨ [a, b] = [-2 - Real.sqrt 17, 13 / 4]) :=
by sorry

end interval_for_f_l342_342164


namespace sum_of_digits_7_pow_11_l342_342454

theorem sum_of_digits_7_pow_11 : 
  let n := 7 in
  let power := 11 in
  let last_two_digits := (n ^ power) % 100 in
  let tens_digit := last_two_digits / 10 in
  let ones_digit := last_two_digits % 10 in
  tens_digit + ones_digit = 7 :=
by {
  sorry
}

end sum_of_digits_7_pow_11_l342_342454


namespace number_of_distinct_m_values_l342_342239

theorem number_of_distinct_m_values :
  let roots (x_1 x_2 : ℤ) := x_1 * x_2 = 36 ∧ x_2 = x_2
  let m_values := {m : ℤ | ∃ (x_1 x_2 : ℤ), x_1 * x_2 = 36 ∧ m = x_1 + x_2}
  m_values.card = 10 :=
sorry

end number_of_distinct_m_values_l342_342239


namespace greater_grazing_area_l342_342002

/--
A homeowner wishes to secure a goat with a 12-foot rope to a circular water tank that has a radius of 10 feet.
The rope is attached such that it allows the goat to roam around the tank.
We prove that Arrangement 1 gives the goat the greater area to graze by 35π square feet compared to Arrangement 2.

Arrangement 1: The rope allows the goat to roam in a full circle around the tank.
Arrangement 2: The rope is attached halfway along the radius of the tank, allowing the goat to roam around part of the tank.
-/
theorem greater_grazing_area (r_rope r_tank : ℝ) (h_rope : r_rope = 12) (h_tank : r_tank = 10) :
  let area1 := π * (r_rope ^ 2)
  let area2 := (3 / 4) * π * (r_rope ^ 2) + (1 / 4) * π * (2 ^ 2)
  area1 - area2 = 35 * π :=
by
  let area1 := π * (r_rope ^ 2)
  let area2 := (3 / 4) * π * (r_rope ^ 2) + (1 / 4) * π * (2 ^ 2)
  have h_area1 : area1 = π * (12 ^ 2) := by rw h_rope; rfl
  have h_area2 : area2 = (3 / 4) * π * (12 ^ 2) + (1 / 4) * π * (2 ^ 2) := by rw h_rope; rfl
  show area1 - area2 = 35 * π
  rw [h_area1, h_area2]
  linarith

end greater_grazing_area_l342_342002


namespace jones_children_probability_l342_342796

/-- Assuming that the gender of each of Mr. Jones' 8 children is determined independently with an equal probability of being male or female, the probability that Mr. Jones has more sons than daughters or more daughters than sons is 93 / 128. -/
theorem jones_children_probability :
  (let p := 1 / 2 in 
   let n := 8 in 
   let total := 2^n in 
   let binom := Nat.choose n (n / 2) in
   let favorable := total - binom in
   favorable / total = 93 / 128) :=
by
  sorry

end jones_children_probability_l342_342796


namespace sum_of_coefficients_eq_128_coeff_of_x3_in_expansion_l342_342119

theorem sum_of_coefficients_eq_128 (n : ℕ) (h : (x + x⁻¹) ^ n).sum_coefficients = 128 : n = 7 :=
by sorry

theorem coeff_of_x3_in_expansion (n : ℕ) (h : n = 7) : coeff (expand (x + x⁻¹) ^ n) 3 = 21 :=
by sorry

end sum_of_coefficients_eq_128_coeff_of_x3_in_expansion_l342_342119


namespace perimeter_of_triangle_OED_eq_BC_l342_342827

theorem perimeter_of_triangle_OED_eq_BC
  (A B C O D E : Point)
  (angle_bisectors_intersect_at_O : IsAngleBisector (angleA A B C) O ∧
                                    IsAngleBisector (angleB A B C) O ∧
                                    IsAngleBisector (angleC A B C) O)
  (lines_through_O_parallel_AB_AC : Parallel (Line O D) (Line A B) ∧
                                    Parallel (Line O E) (Line A C))
  (intersect_BC_at_D_E : Intersect (Line O D) (Line B C) D ∧
                         Intersect (Line O E) (Line B C) E) :
  Perimeter (Triangle O E D) = Segment B C :=
by
  sorry

end perimeter_of_triangle_OED_eq_BC_l342_342827


namespace percent_increase_march_to_june_l342_342480

def profit_increase_percent (P : ℝ) : ℝ :=
  let april_profit := 1.20 * P
  let may_profit := 0.96 * P 
  let june_profit := 1.44 * P 
  let increase := june_profit - P
  (increase / P) * 100

theorem percent_increase_march_to_june (P : ℝ) (hP : 0 < P) :
  profit_increase_percent P = 44 := 
by
  sorry -- proof not required

end percent_increase_march_to_june_l342_342480


namespace sum_of_possible_x_values_l342_342555

theorem sum_of_possible_x_values : 
  (∑ x in { x : ℝ | |3 * x - 6| = 9 }, x) = 4 :=
sorry

end sum_of_possible_x_values_l342_342555


namespace map_scale_l342_342271

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l342_342271


namespace original_number_exists_l342_342523

theorem original_number_exists : 
  ∃ (t o : ℕ), (10 * t + o = 74) ∧ (t = o * o - 9) ∧ (10 * o + t = 10 * t + o - 27) :=
by
  sorry

end original_number_exists_l342_342523


namespace initial_number_of_trees_l342_342408

theorem initial_number_of_trees (trees_removed remaining_trees initial_trees : ℕ) 
  (h1 : trees_removed = 4) 
  (h2 : remaining_trees = 2) 
  (h3 : remaining_trees + trees_removed = initial_trees) : 
  initial_trees = 6 :=
by
  sorry

end initial_number_of_trees_l342_342408


namespace variance_translation_invariant_l342_342596

noncomputable def variance (data : List ℝ) : ℝ :=
  let mean := (data.sum / data.length)
  (data.map (λ x, (x - mean) ^ 2)).sum / data.length

theorem variance_translation_invariant
  (data : List ℝ)
  (h : variance data = 7)
  (translated_data : List ℝ := data.map (λ x, x - 1)) :
  variance translated_data = 7 := by
  sorry

end variance_translation_invariant_l342_342596


namespace adam_spent_money_on_ferris_wheel_l342_342539

def tickets_bought : ℕ := 13
def tickets_left : ℕ := 4
def ticket_cost : ℕ := 9
def tickets_used : ℕ := tickets_bought - tickets_left

theorem adam_spent_money_on_ferris_wheel :
  tickets_used * ticket_cost = 81 :=
by
  sorry

end adam_spent_money_on_ferris_wheel_l342_342539


namespace variance_translation_invariant_l342_342595

noncomputable def variance (data : List ℝ) : ℝ :=
  let mean := (data.sum / data.length)
  (data.map (λ x, (x - mean) ^ 2)).sum / data.length

theorem variance_translation_invariant
  (data : List ℝ)
  (h : variance data = 7)
  (translated_data : List ℝ := data.map (λ x, x - 1)) :
  variance translated_data = 7 := by
  sorry

end variance_translation_invariant_l342_342595


namespace coordinates_of_D_l342_342093
-- Importing the necessary library

-- Defining the conditions as given in the problem
def AB : ℝ × ℝ := (5, 3)
def C : ℝ × ℝ := (-1, 3)
def CD : ℝ × ℝ := (2 * 5, 2 * 3)

-- The target proof statement
theorem coordinates_of_D :
  ∃ D : ℝ × ℝ, CD = D - C ∧ D = (9, -3) :=
by
  sorry

end coordinates_of_D_l342_342093


namespace orange_juice_serving_size_l342_342972

theorem orange_juice_serving_size :
  ∀ (cans_of_concentrate water_per_concentrate total_servings concentrate_volume serving_volume : ℕ),
    water_per_concentrate = 3 →
    concentrate_volume = 12 →
    cans_of_concentrate = 45 →
    total_servings = 360 →
    serving_volume = (cans_of_concentrate * (1 + water_per_concentrate) * concentrate_volume) / total_servings →
    serving_volume = 6 := 
by
  intros,
  sorry

end orange_juice_serving_size_l342_342972


namespace map_length_representation_l342_342312

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l342_342312


namespace min_value_quotient_l342_342103

noncomputable def a : ℕ → ℕ 
| 1 := 12
| (n + 1) := a n + 2 * n

theorem min_value_quotient (n : ℕ) (h : n > 0) : 
  (∃ a : ℕ → ℕ, a 1 = 12 ∧ ∀ n : ℕ, a (n + 1) = a n + 2 * n) → 
  ∃ m, m = 6 :=
by
  intro ha
  sorry

end min_value_quotient_l342_342103


namespace greatest_integer_less_than_11_over_3_l342_342435

theorem greatest_integer_less_than_11_over_3 :
  ∃ n : ℤ, n < 11 / 3 ∧ ∀ m : ℤ, m < 11 / 3 → m ≤ n := 
by {
  use 3,
  sorry
}

end greatest_integer_less_than_11_over_3_l342_342435


namespace factorize_expression_l342_342471

variable {a b x y : ℝ}

theorem factorize_expression :
  (x^2 - y^2) * a^2 - (x^2 - y^2) * b^2 = (x + y) * (x - y) * (a + b) * (a - b) :=
by
  sorry

end factorize_expression_l342_342471


namespace map_scale_representation_l342_342335

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l342_342335


namespace surface_area_invisible_block_l342_342969

-- Define the given areas of the seven blocks
def A1 := 148
def A2 := 46
def A3 := 72
def A4 := 28
def A5 := 88
def A6 := 126
def A7 := 58

-- Define total surface areas of the black and white blocks
def S_black := A1 + A2 + A3 + A4
def S_white := A5 + A6 + A7

-- Define the proof problem
theorem surface_area_invisible_block : S_black - S_white = 22 :=
by
  -- This sorry allows the Lean statement to build successfully
  sorry

end surface_area_invisible_block_l342_342969


namespace amount_added_to_doubled_number_l342_342853

theorem amount_added_to_doubled_number (N A : ℝ) (h1 : N = 6.0) (h2 : 2 * N + A = 17) : A = 5.0 :=
by
  sorry

end amount_added_to_doubled_number_l342_342853


namespace map_representation_l342_342261

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l342_342261


namespace probability_x_plus_y_lt_4_inside_square_l342_342928

def square_area : ℝ := 9
def triangle_area : ℝ := 2
def probability : ℝ := 7 / 9

theorem probability_x_plus_y_lt_4_inside_square :
  ∀ (x y : ℝ), 
  (0 ≤ x ∧ x ≤ 3) ∧ (0 ≤ y ∧ y ≤ 3) ∧ (x + y < 4) → 
  (triangle_area = 2) ∧ (square_area = 9) ∧ (probability = 7 / 9) :=
by
  intros x y h
  sorry

end probability_x_plus_y_lt_4_inside_square_l342_342928


namespace correct_options_l342_342611

variables {P : ℝ × ℝ} {F1 F2 : ℝ × ℝ}
variables {a b c : ℝ} (x y : ℝ)
def ellipse := x^2 / 25 + y^2 / 16 = 1
def foc_dist := sqrt (a^2 - b^2)
def a_val := a = 5
def b_val := b = 4
def c_val := c = sqrt (25 - 16)
def cos_angle := cos (angle F1 P F2) = 1 / 2

theorem correct_options
  (H1 : P ∈ ellipse)
  (H2 : F1 = (-3, 0) ∧ F2 = (3, 0))
  (H3 : cos_angle) :
  (perimeter_of_triangle F1 P F2 = 16) ∧ 
  (area_of_triangle F1 P F2 = (16 * sqrt 3) / 3) ∧ 
  (distance P (0, y) = (16 * sqrt 3) / 9) :=
sorry

end correct_options_l342_342611


namespace price_per_glass_first_day_l342_342799

theorem price_per_glass_first_day 
(O G : ℝ) (H : 2 * O * G * P₁ = 3 * O * G * 0.5466666666666666 ) : 
  P₁ = 0.82 :=
by
  sorry

end price_per_glass_first_day_l342_342799


namespace sin_double_angle_l342_342714

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l342_342714


namespace regular_polygon_sides_l342_342943

-- Definitions based on conditions in the problem
def exterior_angle (n : ℕ) : ℝ := 360 / n

theorem regular_polygon_sides (n : ℕ) (h : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l342_342943


namespace max_sum_ineq_l342_342609

theorem max_sum_ineq (x y : ℝ) (h : 2^x + 2^y = 1) : x + y ≤ -2 := 
sorry

end max_sum_ineq_l342_342609


namespace bridgette_total_baths_l342_342985

def bridgette_baths (dogs baths_per_dog_per_month cats baths_per_cat_per_month birds baths_per_bird_per_month : ℕ) : ℕ :=
  (dogs * baths_per_dog_per_month * 12) + (cats * baths_per_cat_per_month * 12) + (birds * (12 / baths_per_bird_per_month))

theorem bridgette_total_baths :
  bridgette_baths 2 2 3 1 4 4 = 96 :=
by
  -- Proof omitted
  sorry

end bridgette_total_baths_l342_342985


namespace Mira_model_height_l342_342185

theorem Mira_model_height 
  (H_actual : ℝ) 
  (P_actual : ℝ)
  (P_model : ℝ)
  (H_actual_eq : H_actual = 320)
  (P_actual_eq : P_actual = 800)
  (P_model_eq : P_model = 0.8) 
: ∃ H_model : ℝ, H_model ≈ 10.12 :=
by {
  let ratio_people := P_actual / P_model,
  let scale_ratio := Real.sqrt ratio_people,
  let H_model := H_actual / scale_ratio,
  use H_model,
  have approx_height : H_model ≈ 10.12 := 
    by sorry, -- Proof skipped
  exact approx_height
}

end Mira_model_height_l342_342185


namespace sum_is_two_l342_342771

noncomputable def compute_sum (x : ℂ) (hx : x^7 = 1) (hx_ne : x ≠ 1) : ℂ :=
  (x^2 / (x - 1)) + (x^4 / (x^2 - 1)) + (x^6 / (x^3 - 1)) + (x^8 / (x^4 - 1)) + (x^10 / (x^5 - 1)) + (x^12 / (x^6 - 1))

theorem sum_is_two (x : ℂ) (hx : x^7 = 1) (hx_ne : x ≠ 1) : compute_sum x hx hx_ne = 2 :=
by
  sorry

end sum_is_two_l342_342771


namespace carlson_fraction_jam_l342_342757

-- Definitions and conditions.
def total_time (T : ℕ) := T > 0
def time_maloish_cookies (t : ℕ) := t > 0
def equal_cookies (c : ℕ) := c > 0
def carlson_rate := 3

-- Let j_k and j_m be the amounts of jam eaten by Carlson and Maloish respectively.
def fraction_jam_carlson (j_k j_m : ℕ) : ℚ := j_k / (j_k + j_m)

-- The problem statement
theorem carlson_fraction_jam (T t c j_k j_m : ℕ)
  (hT : total_time T)
  (ht : time_maloish_cookies t)
  (hc : equal_cookies c)
  (h_carlson_rate : carlson_rate = 3)
  (h_equal_cookies : c > 0)  -- Both ate equal cookies
  (h_jam : j_k + j_m = j_k * 9 / 10 + j_m / 10) :
  fraction_jam_carlson j_k j_m = 9 / 10 :=
by
  sorry

end carlson_fraction_jam_l342_342757


namespace sequence_non_periodic_l342_342391

def is_close_to_floor (S : ℕ → ℕ) (a : ℝ) (ε : ℝ) (n : ℕ) : Prop :=
  abs (S n - n * a) ≤ ε

def is_aperiodic_payments (S : ℕ → ℕ) : Prop :=
  ∀ T N, ∃ n > N, S (n + T) ≠ S n
  
def sequence := λ n : ℕ, if n % 2 = 0 then 1 else 2

theorem sequence_non_periodic :
  (∀ n : ℕ, abs (sequence n - n * (Real.sqrt 2 : ℝ)) ≤ 0.5) ∧
  (sequence 0 = 1) ∧ (sequence 1 = 2) ∧ 
  (sequence 2 = 1) ∧ (sequence 3 = 2) ∧ (sequence 4 = 1) →
  is_aperiodic_payments sequence :=
by
  sorry

end sequence_non_periodic_l342_342391


namespace number_of_distinct_m_values_l342_342240

theorem number_of_distinct_m_values :
  let roots (x_1 x_2 : ℤ) := x_1 * x_2 = 36 ∧ x_2 = x_2
  let m_values := {m : ℤ | ∃ (x_1 x_2 : ℤ), x_1 * x_2 = 36 ∧ m = x_1 + x_2}
  m_values.card = 10 :=
sorry

end number_of_distinct_m_values_l342_342240


namespace sampling_interval_no_exclusion_l342_342844

theorem sampling_interval_no_exclusion (N : ℕ) (intervals : list ℕ)
  (hN : N = 203)
  (h_intervals : intervals = [4, 5, 6, 7]) :
  7 ∈ intervals ∧ N % 7 = 0 :=
by
  sorry

end sampling_interval_no_exclusion_l342_342844


namespace sin_2phi_l342_342740

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_2phi_l342_342740


namespace total_expenditure_l342_342561

def Emma_spent : ℤ := 58
def Elsa_spent := 2 * Emma_spent
def Elizabeth_spent := 4 * Elsa_spent
def total_spent := Emma_spent + Elsa_spent + Elizabeth_spent

theorem total_expenditure : total_spent = 638 := by
  -- Conditions
  have h1 : Emma_spent = 58 := rfl
  have h2 : Elsa_spent = 2 * Emma_spent := rfl
  have h3 : Elizabeth_spent = 4 * Elsa_spent := rfl
  
  -- Since total_spent is a combination of the three expenditures
  -- And using the conditions given, let's calculate
  rw [h2, h1] at h3 -- Substitute values
  rw [h3, h2, h1] at total_spent
  norm_num -- Compute the total
  
  sorry

end total_expenditure_l342_342561


namespace image_of_square_is_irregular_l342_342217

-- Definitions for points and transformations
structure Point (α : Type) :=
(x : α)
(y : α)

def P := Point.mk 0 0
def Q := Point.mk 1 0
def R := Point.mk 1 1
def S := Point.mk 0 1

-- Transformation functions
def u (x y : ℝ) : ℝ := x^2 + 2*x*y - y^2
def v (x y : ℝ) : ℝ := x^2*y - y^3

-- The proof problem
theorem image_of_square_is_irregular :
  let P' := Point.mk (u P.x P.y) (v P.x P.y),
      Q' := Point.mk (u Q.x Q.y) (v Q.x Q.y),
      R' := Point.mk (u R.x R.y) (v R.x R.y),
      S' := Point.mk (u S.x S.y) (v S.x S.y),
      image_points := {P', Q', R', S'} in
  -- image_points form an irregular shape defined by parabolas and curves
  (P' = Point.mk 0 0) ∧
  (Q' = Point.mk 1 0) ∧
  (R' = Point.mk 3 0) ∧
  (S' = Point.mk (-1) (-1)) →
  image_points ≠ set.univ ∧
  ∃ parabolic_curve, parabolic_curve ∈ image_points ∧ image_points ≠ {} :=
by
  sorry

end image_of_square_is_irregular_l342_342217


namespace percentage_return_l342_342691

theorem percentage_return (income investment : ℝ) (h_income : income = 680) (h_investment : investment = 8160) :
  (income / investment) * 100 = 8.33 :=
by
  rw [h_income, h_investment]
  -- The rest of the proof is omitted.
  sorry

end percentage_return_l342_342691


namespace sum_of_tens_and_ones_digits_of_seven_eleven_l342_342453

theorem sum_of_tens_and_ones_digits_of_seven_eleven :
  let n := (3 + 4) ^ 11 in 
  (let ones := n % 10 in
   let tens := (n / 10) % 10 in
   ones + tens = 7) := 
by sorry

end sum_of_tens_and_ones_digits_of_seven_eleven_l342_342453


namespace faye_homework_problems_left_l342_342067

-- Defining the problem conditions
def M : ℕ := 46
def S : ℕ := 9
def A : ℕ := 40

-- The statement to prove
theorem faye_homework_problems_left : M + S - A = 15 := by
  sorry

end faye_homework_problems_left_l342_342067


namespace minimum_value_of_quadratic_function_l342_342743

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem minimum_value_of_quadratic_function 
  (f : ℝ → ℝ)
  (n : ℕ)
  (h1 : f n = 6)
  (h2 : f (n + 1) = 5)
  (h3 : f (n + 2) = 5)
  (hf : ∃ a b c : ℝ, f = quadratic_function a b c) :
  ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = 5 :=
by
  sorry

end minimum_value_of_quadratic_function_l342_342743


namespace smallest_Y_74_l342_342218

def isDigitBin (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d = 0 ∨ d = 1

def smallest_Y (Y : ℕ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ isDigitBin T ∧ T % 15 = 0 ∧ Y = T / 15

theorem smallest_Y_74 : smallest_Y 74 := by
  sorry

end smallest_Y_74_l342_342218


namespace avg_high_correct_avg_low_correct_l342_342124

def high_temps : List ℝ := [51, 63, 59, 56, 48]
def low_temps : List ℝ := [42, 48, 46, 43, 40]

def avg (temps : List ℝ) : ℝ :=
  (List.sum temps) / (temps.length)

theorem avg_high_correct : avg high_temps = 55.4 := by
  sorry

theorem avg_low_correct : avg low_temps = 43.8 := by
  sorry

end avg_high_correct_avg_low_correct_l342_342124


namespace trajectory_and_area_l342_342643

noncomputable def point : Type := ℝ × ℝ

def distance (P Q : point) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def fixedPointA : point := (-2, 0)
def fixedPointB : point := (1, 0)

def satisfies_condition (P : point) : Prop :=
  distance P fixedPointA = 2 * distance P fixedPointB

theorem trajectory_and_area (P : point) (A B : point) :
  A = fixedPointA → B = fixedPointB → satisfies_condition P →
  (P.1 - 2)^2 + P.2^2 = 4 ∧ π * 2 * 2 = 4 * π :=
by
  intros hA hB hc
  sorry

end trajectory_and_area_l342_342643


namespace prob_A_prob_B_prob_two_computers_l342_342495

open ProbabilityTheory

def repair_events :=
  { A0 := 0.75, A1 := 0.15, A2 := 0.06, A3 := 0.04 }

def P (A : String) : ℝ :=
  match A with
  | "A0" => repair_events.A0
  | "A1" => repair_events.A1
  | "A2" => repair_events.A2
  | "A3" => repair_events.A3
  | _ => 0

theorem prob_A : P("A1") + P("A2") + P("A3") = 0.25 :=
by sorry

theorem prob_B : P("A0") + P("A1") = 0.9 :=
by sorry

def two_computers_prob (X Y : String) : ℝ :=
  P(X) * P(Y)

theorem prob_two_computers : 
  two_computers_prob "A0" "A0"
  + 2 * two_computers_prob "A0" "A1"
  + 2 * two_computers_prob "A0" "A2"
  + two_computers_prob "A1" "A1" = 0.9 :=
by sorry

end prob_A_prob_B_prob_two_computers_l342_342495


namespace christen_potatoes_l342_342648

def rate_Homer : ℕ := 4
def rate_Christen : ℕ := 6
def initial_pile : ℕ := 60
def time_alone : ℕ := 6

theorem christen_potatoes : 
  let peeled_by_homer_alone := rate_Homer * time_alone,
      remaining_potatoes := initial_pile - peeled_by_homer_alone,
      combined_rate := rate_Homer + rate_Christen,
      time_together := remaining_potatoes / combined_rate,
      christen_potatoes := rate_Christen * time_together in
  christen_potatoes = 21 :=
by
  sorry

end christen_potatoes_l342_342648


namespace exists_special_set_l342_342063

-- Definition of the condition
def satisfies_condition (A : Set ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → ∃! x ∈ A, x / n ∈ Finset.image (λ k : ℕ, (k : ℕ) * n) (Finset.range 15)

-- The statement of the theorem
theorem exists_special_set : ∃ A : Set ℕ, satisfies_condition A :=
by
  sorry

end exists_special_set_l342_342063


namespace suzhou_visitors_accuracy_l342_342483

/--
In Suzhou, during the National Day holiday in 2023, the city received 17.815 million visitors.
Given that number, prove that it is accurate to the thousands place.
-/
theorem suzhou_visitors_accuracy :
  (17.815 : ℝ) * 10^6 = 17815000 ∧ true := 
by
sorry

end suzhou_visitors_accuracy_l342_342483


namespace inlet_rate_l342_342508

def leak_rate (volume : ℝ) (time_to_empty : ℝ) : ℝ := volume / time_to_empty

def net_emptying_rate (volume : ℝ) (time_to_empty_with_inlet : ℝ) : ℝ := volume / time_to_empty_with_inlet

theorem inlet_rate (volume : ℝ) (time_to_empty : ℝ) (time_to_empty_with_inlet : ℝ) (L R : ℝ) 
  (hL : L = leak_rate volume time_to_empty)
  (hNR : L - R = net_emptying_rate volume time_to_empty_with_inlet) :
  R = 360 :=
by
  have L_val : L = 6048 / 7 := hL
  have R_val : L - R = 6048 / 12 := hNR
  sorry

end inlet_rate_l342_342508


namespace total_cost_model_minimum_average_cost_at_10_l342_342545

-- Define the necessary functions based on the conditions
def initial_purchase_cost : ℕ := 100000
def annual_cost (n : ℕ) : ℕ := 9000 * n
def maintenance_cost (n : ℕ) : ℕ := 2000 * n
def total_maintenance_cost (n : ℕ) : ℕ := n * (1 + n) / 2 * 2000

-- Total cost after n years
def S_n (n : ℕ) : ℕ := initial_purchase_cost + annual_cost n + total_maintenance_cost n

-- Define the total cost model based on the given condition
theorem total_cost_model (n : ℕ) : S_n n = 100000 + 9000 * n + 2000 * n * (n + 1) / 2 := by
  rw [S_n, initial_purchase_cost, annual_cost, total_maintenance_cost]
  norm_num

-- Define the minimum average annual cost condition
def average_annual_cost (n : ℕ) : ℚ := S_n n / n

-- Prove that the minimum average annual cost happens at n = 10
theorem minimum_average_cost_at_10 (n : ℕ) : n = 10 → average_annual_cost n = 30000 := by
  intro h
  rw [h, average_annual_cost]
  have h₁ : (S_n 10 / 10).toRat = 30000.toRat := by
    rw [S_n, initial_purchase_cost, annual_cost, total_maintenance_cost]
    norm_num
  exact h₁

end total_cost_model_minimum_average_cost_at_10_l342_342545


namespace number_of_sides_l342_342957

-- Define the conditions as variables/constants
def exterior_angle (n : ℕ) : ℝ := 18         -- Each exterior angle is 18 degrees
def sum_of_exterior_angles : ℝ := 360        -- Sum of exterior angles of any polygon is 360 degrees

-- Prove the number of sides is equal to 20 given the conditions
theorem number_of_sides : 
  ∃ n : ℕ, (exterior_angle n) * (n : ℝ) = sum_of_exterior_angles → n = 20 := 
by
  sorry

end number_of_sides_l342_342957


namespace first_digit_after_decimal_correct_l342_342778

noncomputable def first_digit_after_decimal (n: ℕ) : ℕ :=
  if n % 2 = 0 then 9 else 4

theorem first_digit_after_decimal_correct (n : ℕ) :
  (first_digit_after_decimal n = 9 ↔ n % 2 = 0) ∧ (first_digit_after_decimal n = 4 ↔ n % 2 = 1) :=
by
  sorry

end first_digit_after_decimal_correct_l342_342778


namespace tennis_tournament_matches_l342_342959

theorem tennis_tournament_matches (num_players : ℕ) (total_days : ℕ) (rest_days : ℕ)
  (num_matches_per_day : ℕ) (matches_per_player : ℕ)
  (h1 : num_players = 10)
  (h2 : total_days = 9)
  (h3 : rest_days = 1)
  (h4 : num_matches_per_day = 5)
  (h5 : matches_per_player = 1)
  : (num_players * (num_players - 1) / 2) - (num_matches_per_day * (total_days - rest_days)) = 40 :=
by
  sorry

end tennis_tournament_matches_l342_342959


namespace sin_2phi_l342_342738

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_2phi_l342_342738


namespace problem_inequality_l342_342221

-- Conditions formulated as variables in Lean
def a : ℝ := real.log 4 / real.log 3
def b : ℝ := (1 / 3) ^ 0.3
def c : ℝ := real.log ((real.log (2 ^ 0.5) / real.log 2)) / real.log 2

-- The theorem to prove the desired inequality
theorem problem_inequality : a < c ∧ c < b := by
  -- Proof will go here
  sorry

end problem_inequality_l342_342221


namespace range_of_a_for_increasing_l342_342138

noncomputable def f (a : ℝ) : (ℝ → ℝ) := λ x => x^3 + a * x^2 + 3 * x

theorem range_of_a_for_increasing (a : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + 2 * a * x + 3) ≥ 0) ↔ (-3 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_for_increasing_l342_342138


namespace quadratic_two_distinct_real_roots_find_other_root_and_m_value_l342_342646

theorem quadratic_two_distinct_real_roots (m : ℝ) : ∀ x : ℝ, (x^2 - m * x - 1 = 0) →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x^2 - m * x - 1 = 0) :=
by sorry

theorem find_other_root_and_m_value :
  ∀ (m x₁ : ℝ), x₁ = real.sqrt 2 → (x^2 - m * x - 1 = 0) →
  ∀ x₂, x₂ = -real.sqrt(2)/2 ∧ m = real.sqrt(2)/2 :=
by sorry

end quadratic_two_distinct_real_roots_find_other_root_and_m_value_l342_342646


namespace sin_double_angle_l342_342715

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l342_342715


namespace slope_angle_of_line_l342_342701

-- Definitions from the conditions
def line := λ x y : ℝ, x + (real.sqrt 3) * y - 3 = 0

-- Statement of the theorem
theorem slope_angle_of_line :
  ∃ α : ℝ, (∀ (x y : ℝ), line x y → ∃ (m : ℝ), m = -1 / (real.sqrt 3)) ∧ (α = 150) := by
  sorry

end slope_angle_of_line_l342_342701


namespace fractions_product_equals_54_l342_342548

theorem fractions_product_equals_54 :
  (4 / 5) * (9 / 6) * (12 / 4) * (20 / 15) * (14 / 21) * (35 / 28) * (48 / 32) * (24 / 16) = 54 :=
by
  -- Add the proof here
  sorry

end fractions_product_equals_54_l342_342548


namespace smallest_12_digit_proof_l342_342082

def is_12_digit_number (n : ℕ) : Prop :=
  n >= 10^11 ∧ n < 10^12

def contains_each_digit_0_to_9 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] → d ∈ n.digits 10

def is_divisible_by_36 (n : ℕ) : Prop :=
  n % 36 = 0

noncomputable def smallest_12_digit_divisible_by_36_and_contains_each_digit : ℕ :=
  100023457896

theorem smallest_12_digit_proof :
  is_12_digit_number smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  contains_each_digit_0_to_9 smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  is_divisible_by_36 smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  ∀ m : ℕ, is_12_digit_number m ∧ contains_each_digit_0_to_9 m ∧ is_divisible_by_36 m →
  m >= smallest_12_digit_divisible_by_36_and_contains_each_digit :=
by
  sorry

end smallest_12_digit_proof_l342_342082


namespace magnitude_of_b_l342_342112

variables (a b : ℝ) (θ : ℝ) (angle_ab : θ = 60) (norm_a : a = 3) (norm_a_plus_b : a + b = sqrt 13)

noncomputable def vector_b_magnitude : ℝ := by
  sorry

theorem magnitude_of_b : vector_b_magnitude a b θ = 1 :=
by
  -- Use the dot product properties and given conditions to prove the magnitude of b
  sorry

end magnitude_of_b_l342_342112


namespace perpendicular_EF_FB_l342_342781

variables (A B C D E F : Point)
variables (rectangle : Rectangle A B C D)
variables (perpendicular_from_C : ∃ E, Perp (Line C E) (Line B D) ∧ Collinear A B E)
variables (circle_cent_B_radius_BC : Circle B (dist B C))
variables (F_on_AD : ∃ F, OnCircle F circle_cent_B_radius_B)

theorem perpendicular_EF_FB :
  ∀ {A B C D E F : Point}
    (rectangle : Rectangle A B C D)
    (perpendicular_from_C : ∃ E, Perp (Line C E) (Line B D) ∧ Collinear A B E)
    (circle_cent_B_radius_BC : Circle B (dist B C))
    (F_on_AD : ∃ F, OnCircle F circle_cent_B_radius_B),
  ∠ F B E = 90 :=
by sorry

end perpendicular_EF_FB_l342_342781


namespace minimum_value_of_function_l342_342838

noncomputable def function_y (x : ℝ) : ℝ := 1 / (Real.sqrt (x - x^2))

theorem minimum_value_of_function : (∀ x : ℝ, 0 < x ∧ x < 1 → function_y x ≥ 2) ∧ (∃ x : ℝ, 0 < x ∧ x < 1 ∧ function_y x = 2) :=
by
  sorry

end minimum_value_of_function_l342_342838


namespace triangle_area_proof_l342_342211

theorem triangle_area_proof (a b c h_a h_b h_c a' b' c': ℝ) 
  (T : (ℝ, ℝ, ℝ)) (T' : (ℝ, ℝ, ℝ)) (T'' : (ℝ, ℝ, ℝ)) 
  (A B C : ℝ)
  (h1 : T = (a, b, c))
  (h2 : T' = (a', b', c'))
  (h3 : T'' = (h_a', h_b', h_c'))
  (h4 : T' = (h_a, h_b, h_c))
  (h5 : ∀A B C, A = 30 ∧ B = 20):
  A = 45 :=
begin
  sorry,
end

end triangle_area_proof_l342_342211


namespace map_length_representation_l342_342310

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l342_342310


namespace function_is_convex_l342_342143

variable (a k : ℝ) (ha : 0 < a) (hk : 0 < k)

def f (x : ℝ) := 1 / x ^ k + a

theorem function_is_convex (x : ℝ) (hx : 0 < x) : ConvexOn ℝ Set.Ioi { x : ℝ | 0 < x } f := 
sorry

end function_is_convex_l342_342143


namespace runners_meet_at_2pm_l342_342984

theorem runners_meet_at_2pm :
  let ben_start := 0
  let emily_start := 2
  let nick_start := 4
  let ben_lap := 5
  let emily_lap := 8
  let nick_lap := 9 in
  let lcm_laps := Nat.lcm ben_lap (Nat.lcm emily_lap nick_lap) in
  let result_time := lcm_laps + ben_start in
  result_time = 360 ∧
  result_time / 60 = 6 ∧
  (ben_start + results_time) % ben_lap = 0 ∧
  (emily_start + results_time) % emily_lap = 0 ∧
  (nick_start + results_time) % nick_lap = 0 :=
by
  let ben_start := 0
  let emily_start := 2
  let nick_start := 4
  let ben_lap := 5
  let emily_lap := 8
  let nick_lap := 9
  let lcm_laps := Nat.lcm ben_lap (Nat.lcm emily_lap nick_lap)
  have h_lcm : lcm_laps = 360 := sorry
  let result_time := lcm_laps + ben_start
  have h_result_time : result_time = 360 := sorry
  have h_time_divisible : result_time / 60 = 6 := by rw [h_result_time]; exact sorry
  have h_ben : (ben_start + result_time) % ben_lap = 0 := sorry
  have h_emily : (emily_start + result_time) % emily_lap = 0 := sorry
  have h_nick : (nick_start + result_time) % nick_lap = 0 := sorry
  exact ⟨h_lcm, h_result_time, h_time_divisible, h_ben, h_emily, h_nick⟩

end runners_meet_at_2pm_l342_342984


namespace maximum_radius_inscribed_circle_point_coordinates_isosceles_triangle_l342_342194

noncomputable def ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + (p.2^2 / 3) = 1}

def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

-- Prove that the maximum radius of the inscribed circle of triangle PF1F2 is √3 / 3
theorem maximum_radius_inscribed_circle (P : ℝ × ℝ) (hP : P ∈ ellipse) : 
  let r := (Real.sqrt 3) / 3 in 
  ∃ r_max, r_max = r :=
sorry

-- Prove the coordinates of P given triangle POF2 is isosceles
theorem point_coordinates_isosceles_triangle 
  (P : ℝ × ℝ) (hP : P ∈ ellipse) (h_isosceles : (P.1 = 1/2)) :
  P = (1/2, (3 * Real.sqrt 5) / 4) ∨ P = (1/2, -(3 * Real.sqrt 5) / 4) :=
sorry

end maximum_radius_inscribed_circle_point_coordinates_isosceles_triangle_l342_342194


namespace find_first_month_sale_l342_342507

theorem find_first_month_sale(
  (sale_2 : ℝ) (sale_2 = 6927) 
  (sale_3 : ℝ) (sale_3 = 6855) 
  (sale_4 : ℝ) (sale_4 = 7230)
  (sale_5 : ℝ) (sale_5 = 6562) 
  (sale_6 : ℝ) (sale_6 = 5191) 
  (avg_6_months : ℝ) (avg_6_months = 6500)
): ∃ (sale_1 : ℝ), sale_1 = 6235 := 
sorry

end find_first_month_sale_l342_342507


namespace length_of_QR_l342_342425

-- Definitions for the problem
variable (P Q R : Type) 
variable [MetricSpace P] [MetricSpace Q] [MetricSpace R]
variable (radius_P : ℝ) (radius_Q : ℝ)
variable (tangent_line : Set P) (ray_PQ : Set P)

-- Conditions from the problem
def extern_tangent_circles (P Q : Type) [MetricSpace P] [MetricSpace Q] (radius_P : ℝ) (radius_Q : ℝ) :=
  dist P Q = radius_P + radius_Q

def intersects_ray (tangent_line : Set P) (ray_PQ : Set P) (R : P) :=
  R ∈ tangent_line ∧ R ∈ ray_PQ

def vertical_tangent (Q : P) (radius_P : ℝ) :=
  ∃ U, (U ∈ tangent_line ∧ U ∈ circle P radius_P ∧ is_perpendicular U Q)

-- Problem statement to prove
theorem length_of_QR 
  (h_tangent : extern_tangent_circles P Q 7 4)
  (h_intersect : intersects_ray tangent_line ray_PQ R)
  (h_vertical : vertical_tangent Q 7) : dist Q R = 4 := 
  sorry

end length_of_QR_l342_342425


namespace map_representation_l342_342251

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l342_342251


namespace exists_equal_sum_pairs_l342_342481

theorem exists_equal_sum_pairs (n : ℕ) (hn : n > 2009) :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ≤ 2009 ∧ b ≤ 2009 ∧ c ≤ 2009 ∧ d ≤ 2009 ∧
  (1 / a + 1 / b : ℝ) = 1 / c + 1 / d :=
sorry

end exists_equal_sum_pairs_l342_342481


namespace number_of_distinct_m_values_l342_342242

theorem number_of_distinct_m_values :
  let roots (x_1 x_2 : ℤ) := x_1 * x_2 = 36 ∧ x_2 = x_2
  let m_values := {m : ℤ | ∃ (x_1 x_2 : ℤ), x_1 * x_2 = 36 ∧ m = x_1 + x_2}
  m_values.card = 10 :=
sorry

end number_of_distinct_m_values_l342_342242


namespace partition_exists_l342_342762

theorem partition_exists (k : ℕ) (hk : k > 0) :
  ∃ (A B : set ℕ), 
    A ∪ B = {x | x ∈ set.Ico 0 (2^(k+1))} ∧
    A ∩ B = ∅ ∧
    (∀ m ∈ (set.range (k + 1)).erase 0, 
      (∑ x in A, x^m = ∑ y in B, y^m)) :=
begin
  sorry
end

end partition_exists_l342_342762


namespace sum_ineq_l342_342086

noncomputable theory

def finite_nonzero {α : Type*} (s : α → ℝ) := set.finite {a : α | s a ≠ 0}

theorem sum_ineq (a : ℤ × ℤ → ℝ)
  (h_zero : a (0,0) = 0)
  (h_finite : finite_nonzero a) :
  (∑ x in set.range (λ x : ℤ × ℤ, x.1), ∑ y in set.range (λ y : ℤ × ℤ, y.2),
    a (x, y) * (a (x, 2 * x + y) + a (x + 2 * y, y))) ≤ 
  real.sqrt 3 * ∑ x in set.range (λ x : ℤ × ℤ, x.1), ∑ y in set.range (λ y : ℤ × ℤ, y.2), a (x, y) ^ 2 :=
begin
  sorry
end

end sum_ineq_l342_342086


namespace max_value_sqrt_sum_l342_342772

theorem max_value_sqrt_sum (x y z : ℝ) (h₀ : x ≥ 0) (h₁ : y ≥ 0) (h₂ : z ≥ 0) (h₃ : x + y + z = 7) : 
    sqrt (3 * x + 2) + sqrt (3 * y + 2) + sqrt (3 * z + 2) ≤ 9 :=
sorry

end max_value_sqrt_sum_l342_342772


namespace convex_polygon_inequality_l342_342807

noncomputable def arithmeticMean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem convex_polygon_inequality (n : ℕ) (hn : 4 ≤ n) :
  let sides := Polygon.sides n
  let diagonals := Polygon.diagonals n
  arithmeticMean sides < arithmeticMean diagonals :=
by
  sorry

end convex_polygon_inequality_l342_342807


namespace find_y_coordinate_l342_342963

theorem find_y_coordinate (m b x : ℝ) (h1 : m = 4) (h2 : b = 100) (h3 : x = 50) : 
  let y := m * x + b in y = 300 :=
by
  sorry

end find_y_coordinate_l342_342963


namespace coeff_x4_l342_342569

theorem coeff_x4 :
  let term1 := 5 * (2 * x^4 - x^6)
  let term2 := -4 * (x^3 - x^4 + x^7)
  let term3 := 3 * (3 * x^4 - x^{11})
  ∀ x : ℝ, (term1 + term2 + term3).coeff 4 = 15 :=
by
  sorry

end coeff_x4_l342_342569


namespace rate_of_current_is_8_5_l342_342005

-- Define the constants for the problem
def downstream_speed : ℝ := 24
def upstream_speed : ℝ := 7
def rate_still_water : ℝ := 15.5

-- Define the rate of the current calculation
def rate_of_current : ℝ := downstream_speed - rate_still_water

-- Define the rate of the current proof statement
theorem rate_of_current_is_8_5 :
  rate_of_current = 8.5 :=
by
  -- This skip the actual proof
  sorry

end rate_of_current_is_8_5_l342_342005


namespace balls_in_boxes_l342_342655

def num_ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  if boxes = 1 then 1
  else if boxes = 2 then (balls + 1) // 2
  else if boxes = 3 then match balls with
    | 0 => 1
    | 1 => 1
    | 2 => 1
    | _ => 1 + 7 + 21 + 35 + 7 + (35 / 3).toNat + 12
  else 0

theorem balls_in_boxes :
  num_ways_to_distribute_balls 7 3 = 95 :=
by
  sorry

end balls_in_boxes_l342_342655


namespace triangle_AMN_to_square_ABCD_ratio_l342_342186

noncomputable def ratio_areas_triangle_to_square (x : ℝ) : ℝ := (1 / 12)

theorem triangle_AMN_to_square_ABCD_ratio :
  ∀ (x : ℝ) (ABCD : ℝ) (M N : ℝ),
    ABCD = x * x →
    M = x / 2 →
    N = x / 3 →
    ratio_areas_triangle_to_square x = (1 : ℝ) / 12 :=
by {
  intros x ABCD M N,
  intro h1,
  intro h2,
  intro h3,
  rw ratio_areas_triangle_to_square,
  refl,
}

end triangle_AMN_to_square_ABCD_ratio_l342_342186


namespace number_of_points_l342_342225

-- Definitions based on conditions in the problem
variables (A B P : Type) [metric_space A B P]
variable (r : ℝ)
variable (AP BP : ℝ → ℝ)
variable (angleAPB : ℝ)
variable (sum_of_squares_AP_BP : ℝ)

-- The proof statement
theorem number_of_points (r = 2) (sum_of_squares_AP_BP = 10) (angleAPB = 60) :
  ∃! P, (↑r = 2 ∧ sum_of_squares_AP_BP = 10 ∧ angleAPB = 60) :=
sorry

end number_of_points_l342_342225


namespace approximate_number_l342_342033

theorem approximate_number (x : ℝ) (y : ℕ) (h : x = 1.20 ∧ y = 10000) :
  (x * y).to_nat = 12000 → ← true sorry
  by sorry

end approximate_number_l342_342033


namespace max_abc_l342_342244

theorem max_abc : ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (a * b + b * c = 518) ∧ 
  (a * b - a * c = 360) ∧ 
  (a * b * c = 1008) := 
by {
  -- Definitions of a, b, c satisfying the given conditions.
  -- Proof of the maximum value will be placed here (not required as per instructions).
  sorry
}

end max_abc_l342_342244


namespace find_line_eq_l342_342101

theorem find_line_eq
  (P : ℝ × ℝ)
  (hP : P = (2, -1))
  (parallel_line_eq : ∀ x y, 2*x + 3*y - 4 = 0) :
  ∃ c, 2*x + 3*y + c = 0 ∧ (2*x + 3*y + c) = 2*x + 3*y - 1 := 
begin
  have hc : c = -1,
  sorry,
  use [c],
  split,
  { exact parallel_line_eq x y },
  { exact hc }
end

end find_line_eq_l342_342101


namespace angle_between_tangents_l342_342825

theorem angle_between_tangents (x y : ℝ) : 
  (x^2 + y^2 - 12 * x + 27 = 0) → 
  (angle_between_tangents_from_origin x y = π / 3) := 
sorry

end angle_between_tangents_l342_342825


namespace quadrilaterals_same_area_and_perimeter_l342_342420

noncomputable def distance (p q : ℝ × ℝ) : ℝ := 
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

def area_of_quadrilateral (a b c d : ℝ × ℝ) : ℝ := 
  -- Using the Shoelace formula
  (0.5) * |(b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2) + 
           (c.1 - a.1) * (d.2 - a.2) - (d.1 - a.1) * (c.2 - a.2) + 
           (d.1 - a.1) * (a.2 - d.2) - (a.1 - d.1) * (d.2 - a.2)|
 
def perimeter_of_quadrilateral (a b c d : ℝ × ℝ) : ℝ :=
  distance a b + distance b c + distance c d + distance d a

theorem quadrilaterals_same_area_and_perimeter :
  let A₁ := (0, 0) 
  let A₂ := (3, 0)
  let A₃ := (3, 2)
  let A₄ := (0, 3)
  let B₁ := (0, 0)
  let B₂ := (3, 0)
  let B₃ := (3, 3)
  let B₄ := (0, 2)
  area_of_quadrilateral A₁ A₂ A₃ A₄ = area_of_quadrilateral B₁ B₂ B₃ B₄ ∧
  perimeter_of_quadrilateral A₁ A₂ A₃ A₄ = perimeter_of_quadrilateral B₁ B₂ B₃ B₄ := 
by {
  sorry
}

end quadrilaterals_same_area_and_perimeter_l342_342420


namespace value_of_a_l342_342122

noncomputable def is_pure_imaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = complex.i * b

theorem value_of_a (a : ℝ) (h : is_pure_imaginary ((a + a * complex.i) / (2 - a * complex.i))) : a = 2 :=
by sorry

end value_of_a_l342_342122


namespace regular_polygon_has_20_sides_l342_342951

-- Definition of a regular polygon with an exterior angle of 18 degrees
def regular_polygon_sides (external_angle : ℝ) : ℕ :=
  if external_angle > 0 then (360 / external_angle).toInt else 0

theorem regular_polygon_has_20_sides :
  regular_polygon_sides 18 = 20 :=
by
  sorry

end regular_polygon_has_20_sides_l342_342951


namespace radius_of_cone_l342_342661

theorem radius_of_cone (A : ℝ) (g : ℝ) (R : ℝ) (hA : A = 15 * Real.pi) (hg : g = 5) : R = 3 :=
sorry

end radius_of_cone_l342_342661


namespace basketball_points_l342_342174

variable {f t : ℕ}

theorem basketball_points (h₁ : f + t = 40) 
  (h₂ : ∃ k1 k2 : ℕ, k1 = 0.25 * f ∧ k2 = 0.4 * t ∧ k1 * 1 + k2 * 3 = 48)
  : 0.25 * f + 1.2 * t = 48 :=
sorry

end basketball_points_l342_342174


namespace boats_distance_one_minute_before_collision_l342_342419

theorem boats_distance_one_minute_before_collision :
  let speedA := 5  -- miles/hr
  let speedB := 21 -- miles/hr
  let initial_distance := 20 -- miles
  let combined_speed := speedA + speedB -- combined speed in miles/hr
  let speed_per_minute := combined_speed / 60 -- convert to miles/minute
  let time_to_collision := initial_distance / speed_per_minute -- time in minutes until collision
  initial_distance - (time_to_collision - 1) * speed_per_minute = 0.4333 :=
by
  sorry

end boats_distance_one_minute_before_collision_l342_342419


namespace calculate_expression_l342_342542

theorem calculate_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end calculate_expression_l342_342542


namespace find_reflecting_point_l342_342930

-- Definitions of points and the plane equation
def A : ℝ × ℝ × ℝ := (2, -1, 3)
def plane (x y z : ℝ) : Prop := 2 * x - y + z = 3
def C : ℝ × ℝ × ℝ := (4, 1, 5)

-- Proof stating the condition of the reflecting point B
theorem find_reflecting_point (B : ℝ × ℝ × ℝ) :
  (∃ x y z : ℝ, plane B.1 B.2 B.3 ∧ 
    B = ( (8 - x) / 2 , (1 / 2 - y) , (17 - z) / 2 )) :=
sorry

end find_reflecting_point_l342_342930


namespace carson_runs_at_8_mph_l342_342205

/-- Jerry's and Carson's running speeds. -/
variables (jerry_speed carson_speed : ℝ)

/-- Time in hours for Jerry to run one way to school. -/
def jerry_time : ℝ := 15 / 60 -- 15 minutes converted to hours

/-- Distance from Jerry's house to his school in miles. -/
def distance_to_school : ℝ := 4

/-- Jerry's running speed in miles per hour. -/
def jerry_speed : ℝ := distance_to_school / jerry_time

/-- Jerry's round trip distance and time in hours. -/
def jerry_round_trip_distance : ℝ := 2 * distance_to_school
def jerry_round_trip_time : ℝ := 2 * jerry_time

/-- Carson's running speed calculation. -/
noncomputable def carson_speed : ℝ := distance_to_school / jerry_round_trip_time

/-- Prove Carson's running speed is 8 mph. -/
theorem carson_runs_at_8_mph :
  carson_speed = 8 := by
  sorry

end carson_runs_at_8_mph_l342_342205


namespace regular_polygon_sides_l342_342947

theorem regular_polygon_sides (θ : ℝ) (n : ℕ) (h1 : θ = 18) (h2 : 360 = θ * n) : n = 20 :=
by {
  sorry
}

end regular_polygon_sides_l342_342947


namespace sum_first_seven_terms_arithmetic_sequence_l342_342677

variable (a : ℕ → ℚ)
variable (d : ℚ)

def is_arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_first_seven_terms_arithmetic_sequence (h : is_arithmetic_sequence a d) 
  (h2 : a 2 = 5 / 3) (h6 : a 6 = -7 / 3)
  : (∑ i in Finset.range 7, a i) = -7 / 3 :=
by
  sorry

end sum_first_seven_terms_arithmetic_sequence_l342_342677


namespace sin_2phi_l342_342735

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_2phi_l342_342735


namespace necessary_but_not_sufficient_condition_l342_342902

theorem necessary_but_not_sufficient_condition 
  (α β : ℝ) :
  ¬ (∀ α β, (sin α + cos β = 0) ↔ (sin^2 α + sin^2 β = 1)) ∧ 
    (∀ α β, (sin α + cos β = 0) → (sin^2 α + sin^2 β = 1)) := 
  sorry

end necessary_but_not_sufficient_condition_l342_342902


namespace problem_solution_l342_342703

-- Definition of the conditions
variables {a b c : ℝ} {A B C : ℝ}
variables (cosA cosB sinB sinC : ℝ)

-- Known angles and side lengths
def A_eq_two_pi_over_three (a_eq : a = 4 * real.sqrt 3) (b_plus_c_eq : b + c = 8) : Prop :=
  ∀ (cosA cosB : ℝ),
    let m := (cosA, cosB)
    let n := (b + 2 * c, a)
    vector.dot_product m n = 0 →
    cosA = -1 / 2 →
    A = 2 * π / 3

-- Height from vertex A to side BC
def height_from_A_to_BC {h : ℝ} (a_eq : a = 4 * real.sqrt 3) (b_plus_c_eq : b + c = 8) : Prop :=
  (b = 4) ∧ (c = 4) → h = 2 * real.sqrt 3

theorem problem_solution (a_eq : a = 4 * real.sqrt 3) (b_plus_c_eq : b + c = 8) 
  (cosA_def : cos A = -1/2) (height_eq : height_from_A_to_BC a_eq b_plus_c_eq) :
  A_eq_two_pi_over_three a_eq b_plus_c_eq ∧ height_eq := 
by
  split
  · intro cosA cosB dot_product_eq cosA_eq
    sorry  -- Prove that A = 2 * π / 3 using the provided equations and conditions
  · sorry  -- Prove that h = 2 * real.sqrt 3 using the provided equations and conditions

end problem_solution_l342_342703


namespace map_representation_l342_342290

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l342_342290


namespace Ronald_phone_selling_price_l342_342372

theorem Ronald_phone_selling_price :
  ∀ (InitialInvestment ProfitPerPhone CostPerPhone TotalPhones SellingPricePerPhone : ℕ),
  InitialInvestment = 3000 → 
  TotalPhones = 200 → 
  ProfitPerPhone = InitialInvestment / 3 / TotalPhones → 
  CostPerPhone = InitialInvestment / TotalPhones → 
  SellingPricePerPhone = CostPerPhone + ProfitPerPhone → 
  SellingPricePerPhone = 20 :=
by
  intros InitialInvestment ProfitPerPhone CostPerPhone TotalPhones SellingPricePerPhone
  intros h_initial h_total h_profit h_cost h_selling
  rw h_initial at h_profit h_cost
  rw h_total at h_profit h_cost
  sorry

end Ronald_phone_selling_price_l342_342372


namespace bus_rental_cost_l342_342829

theorem bus_rental_cost :
  let budget : ℕ := 350
  let students : ℕ := 25
  let admission_per_student : ℕ := 10
  cost_to_rent_bus = budget - students * admission_per_student := 100 :=
by
  let budget : ℕ := 350
  let students : ℕ := 25
  let admission_per_student : ℕ := 10
  let admission_fees := students * admission_per_student
  let cost_to_rent_bus := budget - admission_fees
  show cost_to_rent_bus = 100
  sorry

end bus_rental_cost_l342_342829


namespace ratio_M_N_l342_342156

theorem ratio_M_N (P Q M N : ℝ) (h1 : M = 0.30 * Q) (h2 : Q = 0.20 * P) (h3 : N = 0.50 * P) (hP_nonzero : P ≠ 0) :
  M / N = 3 / 25 := 
by 
  sorry

end ratio_M_N_l342_342156


namespace find_triplets_solutions_l342_342909

theorem find_triplets_solutions :
  ∀ (x y z : ℕ),
    x > y > z ∧ (1 / (x : ℝ) + 2 / (y : ℝ) + 3 / (z : ℝ) = 1) ↔
      (x, y, z) = (36, 9, 4) ∨ (x, y, z) = (20, 10, 4) ∨ (x, y, z) = (15, 6, 5) :=
sorry

end find_triplets_solutions_l342_342909


namespace reading_homework_pages_eq_three_l342_342809

-- Define the conditions
def pages_of_math_homework : ℕ := 7
def difference : ℕ := 4

-- Define what we need to prove
theorem reading_homework_pages_eq_three (x : ℕ) (h : x + difference = pages_of_math_homework) : x = 3 := by
  sorry

end reading_homework_pages_eq_three_l342_342809


namespace map_length_representation_l342_342281

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l342_342281


namespace problem1_problem2_problem3_problem4_problem5_problem6_l342_342543

theorem problem1 : -10 + 2 = -8 := 
by 
  sorry

theorem problem2 : -6 - 3 = -9 := 
by 
  sorry

theorem problem3 : (-4) * 6 = -24 := 
by 
  sorry

theorem problem4 : (-15) / 5 = -3 := 
by 
  sorry

theorem problem5 : (-4)^2 / 2 = 8 := 
by 
  sorry

theorem problem6 : | -2 | - 2 = 0 := 
by 
  sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l342_342543


namespace map_length_represents_distance_l342_342323

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l342_342323


namespace profit_percent_l342_342842

/-- 
  Prove that the profit percent is 25%
  given that the ratio of cost price (CP) to selling price (SP) is 4:5.
-/
theorem profit_percent (CP SP : ℝ) (h : CP / SP = 4 / 5) : ((SP - CP) / CP) * 100 = 25 := by
  -- Converting the ratio to an equality
  have h1 : CP = 4 * (SP / 5) := by
    field_simp
    rw [mul_comm (SP / 5), ← mul_div_assoc]
    rw [div_eq_mul_inv, ← mul_inv, mul_comm 5⁻¹, mul_assoc, mul_inv_cancel_left]
    exact h
  -- Substitute CP in the profit percent formula
  rw [h1, sub_mul, div_mul_eq_mul_div, mul_div_assoc, div_self, mul_one, mul_add, ← div_eq_mul_inv, ← mul_div_assoc]
  norm_num
  -- Simplify the resulting expression
  norm_num
  triv

end profit_percent_l342_342842


namespace regular_polygon_sides_l342_342934

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (k : ℕ), (k : ℕ) * 18 = 360) : n = 20 :=
by
  -- Proof body here
  sorry

end regular_polygon_sides_l342_342934


namespace prime_factorization_2020_prime_factorization_2021_l342_342092

theorem prime_factorization_2020 : 2020 = 2^2 * 5 * 101 := by
  sorry

theorem prime_factorization_2021 : 2021 = 43 * 47 := by
  sorry

end prime_factorization_2020_prime_factorization_2021_l342_342092


namespace limit_of_sequence_l342_342044

theorem limit_of_sequence :
  (tendsto (λ (n : ℕ), (1 : ℝ/n + 2n -> +1 --ldots λ (2 ) := ( limit sumodd /den) tendsto ( \lim 
{ sorry }

end limit_of_sequence_l342_342044


namespace map_scale_l342_342355

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l342_342355


namespace largest_x_exists_x_l342_342438

theorem largest_x (x : ℝ) (h : x ≠ 0) : (x / 5 + 1 / (5 * x) = 1 / 2) → x ≤ 2 :=
begin
  sorry
end

theorem exists_x : ∃ x : ℝ, x / 5 + 1 / (5 * x) = 1 / 2 ∧ x = 2 :=
begin
  use 2,
  split,
  { -- Prove the condition
    norm_num,
    linarith,
  },
  { -- Prove the maximum value
    refl
  }
end

end largest_x_exists_x_l342_342438


namespace johns_number_l342_342206

theorem johns_number : 
  (∃ x : ℝ, (5 / 7) * x + 123 = 984) → 
  (∀ y : ℝ, y = (0.7396 * x - 45) → y ≈ 844.85) :=
by
  sorry

end johns_number_l342_342206


namespace seq_a_geometric_sum_seq_b_property_l342_342788

noncomputable def seq_a : ℕ+ → ℝ
| ⟨1, _⟩        => 1
| ⟨n+1, h⟩ => (n + 2) / n * (seq_a ⟨n, Nat.succ_pos n⟩) / (n + 1)

noncomputable def sum_seq_a (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), seq_a ⟨i + 1, Nat.succ_pos i⟩

def geom_seq_property (n : ℕ+) : Prop :=
  (sum_seq_a n / n) = 2^(n - 1)

theorem seq_a_geometric :
  ∀ n : ℕ+, geom_seq_property n :=
sorry

noncomputable def seq_b (n : ℕ) : ℝ :=
  log 2 (seq_a ⟨n + 1, Nat.succ_pos n⟩ / (n + 1))

noncomputable def sum_seq_b (n : ℕ) : ℝ :=
  (∑ i in Finset.range n, seq_b i)

theorem sum_seq_b_property (n : ℕ) :
  sum_seq_b n = (n - 3) * n / 2 :=
sorry

end seq_a_geometric_sum_seq_b_property_l342_342788


namespace num_of_distinct_m_values_l342_342231

theorem num_of_distinct_m_values : 
  (∃ (x1 x2 : ℤ), x1 * x2 = 36 ∧ m = x1 + x2) → 
  (finset.card (finset.image (λ (p : ℤ × ℤ), p.1 + p.2) 
    {p | p.1 * p.2 = 36})) = 10 :=
sorry

end num_of_distinct_m_values_l342_342231


namespace quadratic_min_value_l342_342745

theorem quadratic_min_value (f : ℕ → ℚ) (n : ℕ)
  (h₁ : f n = 6)
  (h₂ : f (n + 1) = 5)
  (h₃ : f (n + 2) = 5) :
  ∃ c : ℚ, c = 39 / 8 ∧ ∀ x : ℕ, f x ≥ c :=
by
  sorry

end quadratic_min_value_l342_342745


namespace sam_age_l342_342689

theorem sam_age (drew_current_age : ℕ) (drew_future_age : ℕ) (sam_future_age : ℕ) : 
  (drew_current_age = 12) → 
  (drew_future_age = drew_current_age + 5) → 
  (sam_future_age = 3 * drew_future_age) → 
  (sam_future_age - 5 = 46) := 
by sorry

end sam_age_l342_342689


namespace paintings_total_l342_342535

def june_paintings : ℕ := 2
def july_paintings : ℕ := 2 * june_paintings
def august_paintings : ℕ := 3 * july_paintings
def total_paintings : ℕ := june_paintings + july_paintings + august_paintings

theorem paintings_total : total_paintings = 18 :=
by {
  sorry
}

end paintings_total_l342_342535


namespace area_of_ADC_l342_342182

open_locale classical

variables {A B C D : Type}

structure TriangleArea (A B C D : Type) :=
(BD : ℝ)
(DC : ℝ)
(AreaABD : ℝ)
(areaRatio : ℝ)
(AreaADC : ℝ)

theorem area_of_ADC {A B C D : Type} 
  (h1 : TriangleArea A B C D)
  (h2 : h1.BD / h1.DC = 3 / 2)
  (h3 : h1.AreaABD = 30) : 
  h1.AreaADC = 20 :=
sorry

end area_of_ADC_l342_342182


namespace necessary_not_sufficient_cond_l342_342899

theorem necessary_not_sufficient_cond (α β : ℝ) :
  (sin α)^2 + (sin β)^2 = 1 → (∀ α β, (sin α + cos β = 0) → (sin α)^2 + (sin β)^2 = 1) ∧ ¬ (∀ α β, ((sin α)^2 + (sin β)^2 = 1) → (sin α + cos β = 0)) :=
by
  sorry

end necessary_not_sufficient_cond_l342_342899


namespace percentage_of_600_eq_half_of_900_l342_342162

theorem percentage_of_600_eq_half_of_900 : 
  ∃ P : ℝ, (P / 100) * 600 = 0.5 * 900 ∧ P = 75 := by
  -- Proof goes here
  sorry

end percentage_of_600_eq_half_of_900_l342_342162


namespace sum_of_tens_and_ones_digits_of_seven_eleven_l342_342450

theorem sum_of_tens_and_ones_digits_of_seven_eleven :
  let n := (3 + 4) ^ 11 in 
  (let ones := n % 10 in
   let tens := (n / 10) % 10 in
   ones + tens = 7) := 
by sorry

end sum_of_tens_and_ones_digits_of_seven_eleven_l342_342450


namespace percentage_decrease_hours_worked_l342_342749

theorem percentage_decrease_hours_worked (B H : ℝ) (h₁ : H > 0) (h₂ : B > 0)
  (h_assistant1 : (1.8 * B) = B * 1.8) (h_assistant2 : (2 * (B / H)) = (1.8 * B) / (0.9 * H)) : 
  ((H - (0.9 * H)) / H) * 100 = 10 := 
by
  sorry

end percentage_decrease_hours_worked_l342_342749


namespace number_of_combinations_l342_342014

theorem number_of_combinations (n k : ℕ) (h_n : n = 6) (h_k : k = 4) :
  (Nat.choose n k) * (Nat.choose n k) * (Nat.factorial k) = 5400 :=
by
  -- provide numeric equivalence for n and k
  rw [h_n, h_k]
  -- calculate the binomial coefficient and factorial
  have h₁ : Nat.choose 6 4 = 15 := by norm_num
  have h₂ : Nat.factorial 4 = 24 := by norm_num
  -- compute the product
  calc
    (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4)
        = 15 * 15 * 24 := by rw [h₁, h₂]
    ... = 5400 := by norm_num

end number_of_combinations_l342_342014


namespace max_product_sum_25_l342_342839

theorem max_product_sum_25 :
  ∃ (x : Fin 25 → ℕ), (∃ (k : ℕ), k ≤ 25 ∧ ∑ i in Finset.finRange k, x i = 25) ∧ (∀ (y : Fin 25 → ℕ) (l : ℕ), 
    (l ≤ 25 ∧ ∑ i in Finset.finRange l, y i = 25) → (∏ i in Finset.finRange l, y i) ≤ (∏ i in Finset.finRange k, x i) :=
begin
  sorry
end

end max_product_sum_25_l342_342839


namespace company_KW_price_l342_342052

theorem company_KW_price (A B : ℝ) (x : ℝ) (h1 : P = x * A) (h2 : P = 2 * B) (h3 : P = (6 / 7) * (A + B)) : x = 1.666666666666667 := 
sorry

end company_KW_price_l342_342052


namespace number_of_elements_in_A_is_power_of_2_l342_342209

variables {k : ℕ} {a : Fin k → ℕ} (h : ∀i, a i ∈ {0, 1, 2, 3})

def p (z : ℕ) : ℕ := ∑ i in Finset.range k, a i * (4 ^ i)

def is_base4_expansion (x : ℕ) (l : List ℕ) (k : ℕ) : Prop :=
  ∃ (x' : ℕ), (x' < 4^k) ∧ (x'.digits 4 = l)

def A : Finset ℕ :=
  (Finset.range (4 ^ k)).filter (λ z, p z = z)

theorem number_of_elements_in_A_is_power_of_2 :
  ∃ (n : ℕ), A.card = 2 ^ n :=
sorry

end number_of_elements_in_A_is_power_of_2_l342_342209


namespace number_of_sides_l342_342958

-- Define the conditions as variables/constants
def exterior_angle (n : ℕ) : ℝ := 18         -- Each exterior angle is 18 degrees
def sum_of_exterior_angles : ℝ := 360        -- Sum of exterior angles of any polygon is 360 degrees

-- Prove the number of sides is equal to 20 given the conditions
theorem number_of_sides : 
  ∃ n : ℕ, (exterior_angle n) * (n : ℝ) = sum_of_exterior_angles → n = 20 := 
by
  sorry

end number_of_sides_l342_342958


namespace final_answer_l342_342790

noncomputable def point := (ℝ × ℝ)

def point1 : point := (1, 2)
def point2 : point := (7, -4)
def trisect_points (p1 p2 : point) : list point :=
  let (x1, y1) := p1 in
  let (x2, y2) := p2 in
  let dx := x2 - x1 in
  let dy := y2 - y1 in
  [(x1 + dx/3, y1 + dy/3), (x1 + 2*dx/3, y1 + 2*dy/3)]

def line_eq (a b c : ℝ) (point : point) : Prop :=
  let (x, y) := point in
  a * x + b * y + c = 0

def through_points (line : ℝ × ℝ × ℝ) (pts : list point) : Prop :=
  pts.any (λ p, line_eq line.fst (line.snd).fst (line.snd).snd p)

theorem final_answer :
  let point1 := (1, 2) in
  let point2 := (7, -4) in
  let lines_to_check := [(2, -3, 1), (3, -4, 7), (7, -2, -18), (2, 8, -22), (4, -9, 15)] in
  let trisect_pts := trisect_points point1 point2 in
  (4, -9, 15) = (4, -9, 15) ∧ through_points (2, 3) [(4, -9, 15)]
:= sorry

end final_answer_l342_342790


namespace products_not_all_for_sale_l342_342906

variable (P : Type) (D : P → Prop) (S : P → Prop)

def products_sale_statement : Prop :=
  ∀ x, D x → S x

theorem products_not_all_for_sale :
  ¬ products_sale_statement D S →
  (∃ x, D x ∧ ¬ S x) ∧ (∃ x, D x ∧ ¬ S x) :=
by
  sorry

end products_not_all_for_sale_l342_342906


namespace selling_prices_max_profit_strategy_l342_342496

theorem selling_prices (x y : ℕ) (hx : y - x = 30) (hy : 2 * x + 3 * y = 740) : x = 130 ∧ y = 160 :=
by
  sorry

theorem max_profit_strategy (m : ℕ) (hm : 20 ≤ m ∧ m ≤ 80) 
(hcost : 90 * m + 110 * (80 - m) ≤ 8400) : m = 20 ∧ (80 - m) = 60 :=
by
  sorry

end selling_prices_max_profit_strategy_l342_342496


namespace cricketer_running_percentage_l342_342502

theorem cricketer_running_percentage :
  ∀ (total_runs runs_from_boundaries runs_from_sixes runs_by_running runs_percentage : ℝ),
  total_runs = 134 ∧
  runs_from_boundaries = 12 * 4 ∧
  runs_from_sixes = 2 * 6 ∧
  runs_by_running = total_runs - (runs_from_boundaries + runs_from_sixes) ∧
  runs_percentage = (runs_by_running / total_runs) * 100 →
  runs_percentage ≈ 55.22 :=
by
  intros; sorry

end cricketer_running_percentage_l342_342502


namespace necessary_condition_not_sufficient_condition_l342_342904

theorem necessary_condition (α β : ℝ) (h : sin α + cos β = 0) : sin^2 α + sin^2 β = 1 :=
by sorry

theorem not_sufficient_condition (α β : ℝ) (h : sin α + cos β ≠ 0) : sin^2 α + sin^2 β = 1 → false :=
by sorry

end necessary_condition_not_sufficient_condition_l342_342904


namespace work_completion_in_days_l342_342492

noncomputable def work_days_needed : ℕ :=
  let A_rate := 1 / 9
  let B_rate := 1 / 18
  let C_rate := 1 / 12
  let D_rate := 1 / 24
  let AB_rate := A_rate + B_rate
  let CD_rate := C_rate + D_rate
  let two_day_work := AB_rate + CD_rate
  let total_cycles := 24 / 7
  let total_days := (if total_cycles % 1 = 0 then total_cycles else total_cycles + 1) * 2
  total_days

theorem work_completion_in_days :
  work_days_needed = 8 :=
by
  sorry

end work_completion_in_days_l342_342492


namespace map_representation_l342_342253

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l342_342253


namespace smallest_integer_with_eight_minimal_fibonacci_ones_l342_342861

open Nat

-- Define the minimal Fibonacci representation condition
def is_minimal_fibonacci_representation (k : ℕ) (a : ℕ → ℕ) : Prop :=
  k = ∑ i in finset.range (nat.succ (finset.sup (finset.filter (λ i, a i = 1) finset.univ))), a i * fib i ∧ 
  ∀ i, a i ∈ {0, 1} ∧ a (nat.succ (finset.sup (finset.filter (λ i, a i = 1) finset.univ))) = 1 ∧ 
  -- Ensure non-consecutiveness (Zeckendorf's condition)
  ∀ i, (a i = 1 → a (nat.succ i) = 0) ∧ (a i = 1 → a (nat.succ (nat.succ i)) = 0)

-- Define the condition of exactly eight ones in the representation
def exactly_eight_ones (a : ℕ → ℕ) : Prop :=
  (finset.filter (λ i, a i = 1) finset.univ).card = 8

noncomputable def minimal_fibonacci_representation_eight_ones : ℕ :=
  ∑ i in finset.range 8, fib (2 * i + 2)

theorem smallest_integer_with_eight_minimal_fibonacci_ones : minimal_fibonacci_representation_eight_ones = 1596 :=
by 
  sorry

end smallest_integer_with_eight_minimal_fibonacci_ones_l342_342861


namespace min_value_of_expression_l342_342615

theorem min_value_of_expression (m n : ℕ) (hm : 0 < m) (hn : 0 < n)
  (hpar : ∀ x y : ℝ, 2 * x + (n - 1) * y - 2 = 0 → ∃ c : ℝ, mx + ny + c = 0) :
  2 * m + n = 9 :=
by
  sorry

end min_value_of_expression_l342_342615


namespace sandy_savings_percentage_l342_342208

theorem sandy_savings_percentage
  (S : ℝ) -- Sandy's salary last year
  (H1 : 0.10 * S = saved_last_year) -- Last year, Sandy saved 10% of her salary.
  (H2 : 1.10 * S = salary_this_year) -- This year, Sandy made 10% more than last year.
  (H3 : 0.15 * salary_this_year = saved_this_year) -- This year, Sandy saved 15% of her salary.
  : (saved_this_year / saved_last_year) * 100 = 165 := 
by 
  sorry

end sandy_savings_percentage_l342_342208


namespace length_of_curve_eq_l342_342077

noncomputable def length_of_parametric_curve : ℝ :=
  let x (t : ℝ) := 3 * Real.sin t
  let y (t : ℝ) := 3 * Real.cos t
  let dx_dt (t : ℝ) := Real.deriv x t
  let dy_dt (t : ℝ) := Real.deriv y t
  ∫ t in 0 .. (3 * Real.pi / 2), Real.sqrt ((dx_dt t) ^ 2 + (dy_dt t) ^ 2)

theorem length_of_curve_eq : length_of_parametric_curve = 4.5 * Real.pi :=
sorry

end length_of_curve_eq_l342_342077


namespace number_of_nonnegative_solutions_l342_342654

theorem number_of_nonnegative_solutions : 
    ∃! (x : ℝ), x^2 + 5 * x = 0 ∧ x ≥ 0 :=
begin
  sorry
end

end number_of_nonnegative_solutions_l342_342654


namespace find_point_on_ellipse_range_of_slope_l342_342216

-- Definitions based on the conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

def foci : set (ℝ × ℝ) := {(-√3, 0), (√3, 0)}

-- Part I
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  ∃ x y > 0, P = (x, y) ∧ ellipse x y ∧
  let PF1 := (x + √3, y) in
  let PF2 := (x - √3, y) in
  (PF1.1 * PF2.1 + PF1.2 * PF2.2) = -5/4

theorem find_point_on_ellipse :
  ∃ P : ℝ × ℝ, point_on_ellipse P ∧ P = (1, √3 / 2) := sorry

-- Part II
def acute_angle_condition (A B : ℝ × ℝ) : Prop :=
  let O := (0, 0) in
  (A.1 * B.1 + A.2 * B.2) > 0

def line_through_M (k : ℝ) : set (ℝ × ℝ) :=
  {P | ∃ x, P = (x, k * x + 2)}

theorem range_of_slope :
  {k : ℝ | ∃ A B ∈ (line_through_M k), ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ acute_angle_condition A B} = 
    {k | -2 < k ∧ k < -√3 / 2 ∨ √3 / 2 < k ∧ k < 2} := sorry

end find_point_on_ellipse_range_of_slope_l342_342216


namespace distances_equal_l342_342698

variables (A B C D A1 A1' A1'' B1 B1' B1'' C1 C1' C1'' D1 D1' D1'') (AB BC CD DA BD : ℝ)

-- The incircle of triangle ABD touches BD, AD, and AB at points A1, A1', and A1'' respectively.
-- The incircle of triangle BCD touches BC, CD, and DB at points C1', C1'', and C1 respectively.
-- Points B1 and D1 are on the diagonal AC of quadrilateral ABCD.
-- Points B1'', B1', D1'', and D1' are on AB, BC, CD, and DA respectively.
-- Quadrilateral ABCD is convex.

theorem distances_equal 
  (h_inc_ABD_BD : A1 ∈ segment ABD)
  (h_inc_ABD_AD : A1' ∈ segment AAD)
  (h_inc_ABD_AB : A1'' ∈ segment AAB)
  (h_inc_BCD_BC : C1' ∈ segment BCD)
  (h_inc_BCD_CD : C1'' ∈ segment CCD)
  (h_inc_BCD_DB : C1 ∈ segment BDB)
  (h_B1_on_AC : B1 ∈ segment A C)
  (h_D1_on_AC : D1 ∈ segment A C)
  (h_B1''_on_AB : B1'' ∈ segment AAB)
  (h_B1'_on_BC : B1' ∈ segment BBC)
  (h_D1''_on_CD : D1'' ∈ segment CCD)
  (h_D1'_on_DA : D1' ∈ segment DDA)
  (ABCD_convex : convex_quadrilateral A B C D) :
  A1C1 = B1D1 ∧ A1'C1' = B1'D1' ∧ A1''C1'' = B1''D1'' :=
sorry

end distances_equal_l342_342698


namespace domain_length_correct_l342_342060

noncomputable def domain_length : ℚ :=
(3^320 - 1) / 3^324

theorem domain_length_correct :
  ∃ m n : ℕ, nat.coprime m n ∧ m + n = 3^320 * 10 ∧ domain_length = m / n :=
begin
  -- Definitions and domain analysis steps
  let g := λ x : ℝ, log (3⁻¹) (log 9 (log (9⁻¹) (log 81 (log (81⁻¹) x)))),
  -- Correct answer
  sorry
end

end domain_length_correct_l342_342060


namespace map_length_representation_l342_342276

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l342_342276


namespace percentage_decrease_hours_worked_l342_342748

theorem percentage_decrease_hours_worked (B H : ℝ) (h₁ : H > 0) (h₂ : B > 0)
  (h_assistant1 : (1.8 * B) = B * 1.8) (h_assistant2 : (2 * (B / H)) = (1.8 * B) / (0.9 * H)) : 
  ((H - (0.9 * H)) / H) * 100 = 10 := 
by
  sorry

end percentage_decrease_hours_worked_l342_342748


namespace range_of_omega_l342_342589

noncomputable def f : ℝ → ℝ
| x => if x > 0 then x + abs (log x) - 2 else sin (ω * x + π / 4) - 1 / 2

theorem range_of_omega (ω : ℝ) : 
  (∀ x, f x = 0 → (x > 0 ∧ x = real.log x + 2) ∨ (x ≤ 0 ∧ sin(ω * x + π / 4) = 1 / 2)) 
  ∧ (7 = (set.univ.filter (λ x, f x = 0)).count) 
  → ω ∈ set.Ico (49 / 12 : ℝ) (65 / 12 : ℝ) :=
sorry

end range_of_omega_l342_342589


namespace profit_percentage_60_l342_342960

theorem profit_percentage_60 (total_apples : Real) (percent_sold_40 : Real) 
(percent_sold_60 : Real) (profit_40 : Real) (total_profit_percent : Real)
(weight_40 : Real) (weight_60 : Real) (profit_40_kg : Real) 
(total_profit_kg : Real) (profit_60_kg : Real) :
  total_apples = 280 ∧
  percent_sold_40 = 0.40 ∧
  percent_sold_60 = 0.60 ∧
  profit_40 = 0.10 ∧
  total_profit_percent = 0.22 ∧
  weight_40 = percent_sold_40 * total_apples ∧
  weight_60 = percent_sold_60 * total_apples ∧
  profit_40_kg = profit_40 * weight_40 ∧
  total_profit_kg = total_profit_percent * total_apples ∧
  profit_60_kg = total_profit_kg - profit_40_kg →
  (profit_60_kg / weight_60) * 100 = 30 :=
begin
  intro h,
  -- Skipping the proof as instructed
  sorry
end

end profit_percentage_60_l342_342960


namespace values_of_a_for_equation_l342_342090

theorem values_of_a_for_equation :
  ∃ S : Finset ℤ, (∀ a ∈ S, |3 * a + 7| + |3 * a - 5| = 12) ∧ S.card = 4 :=
by
  sorry

end values_of_a_for_equation_l342_342090


namespace map_representation_l342_342255

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l342_342255


namespace minimum_value_of_quadratic_function_l342_342742

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem minimum_value_of_quadratic_function 
  (f : ℝ → ℝ)
  (n : ℕ)
  (h1 : f n = 6)
  (h2 : f (n + 1) = 5)
  (h3 : f (n + 2) = 5)
  (hf : ∃ a b c : ℝ, f = quadratic_function a b c) :
  ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = 5 :=
by
  sorry

end minimum_value_of_quadratic_function_l342_342742


namespace arcsin_equation_solution_l342_342375

theorem arcsin_equation_solution (x : ℝ) 
  (h1 : abs (x * sqrt 11 / (2 * sqrt 21)) ≤ 1) 
  (h2 : abs (x * sqrt 11 / (4 * sqrt 21)) ≤ 1) 
  (h3 : abs (5 * x * sqrt 11 / (8 * sqrt 21)) ≤ 1) :
  (arcsin (x * sqrt 11 / (2 * sqrt 21)) + arcsin (x * sqrt 11 / (4 * sqrt 21)) = arcsin (5 * x * sqrt 11 / (8 * sqrt 21))) ↔ (x = 0 ∨ x = 21 / 10 ∨ x = -21 / 10) :=
by
  sorry

end arcsin_equation_solution_l342_342375


namespace sin_double_angle_l342_342712

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_l342_342712


namespace bisect_CT_l342_342212

variables {k : Type*} [metric_space k] {A B C D T : k}
  (diameter_AB : segment k A B)
  (C_on_AB : C ∈ linek (A, B) ∧ is_between B A C)
  (T_on_k : T ∈ circle k (diameter_AB)) 
  (CT_tangent : is_tangent (linek (C, T)) (circle k (diameter_AB)))
  (l_parallel_CT : parallel (linek (A, D)) (linek (C, T)))
  (D_intersection : D = l_parallel_CT ∩ (perpendicular (linek (A, B)) (T)))

-- Prove that line DB bisects segment CT
theorem bisect_CT (h : is_bisector (linek (D, B)) (segment k C T)) : 
  segment_midpoint (linek (D, B)) C T := 
sorry

end bisect_CT_l342_342212


namespace convex_f_l342_342140

variable {α : Type*}
variable [LinearOrder α] [OrderedAddCommGroup α] [Module ℝ α] [OrderedSMul ℝ α] [OrderedAddCommMonoid α]

noncomputable def f (x : ℝ) (a k : ℝ) : ℝ := (1 / x^k) + a

theorem convex_f (a k : ℝ) (ha : 0 < a) (hk : 0 < k) :
  ConvexOn ℝ (Ioi 0) (λ x => (1 / x^k) + a) :=
  sorry

end convex_f_l342_342140


namespace find_ages_l342_342910

theorem find_ages (P J G : ℕ)
  (h1 : P - 10 = 1 / 3 * (J - 10))
  (h2 : J = P + 12)
  (h3 : G = 1 / 2 * (P + J)) :
  P = 16 ∧ G = 22 :=
by
  sorry

end find_ages_l342_342910


namespace dawn_wash_dishes_time_l342_342979

theorem dawn_wash_dishes_time (D : ℕ) : 2 * D + 6 = 46 → D = 20 :=
by
  intro h
  sorry

end dawn_wash_dishes_time_l342_342979


namespace increasing_interval_of_f_l342_342630

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.sin (2 * x + ϕ)

theorem increasing_interval_of_f (ϕ : ℝ) (k : ℤ) :
  (π / 2 < |ϕ| ∧ | f (π / 6) ϕ | = 1 ∧ ∀ x, f x ϕ ≤ | f (π / 6) ϕ |) →
  ∀ x, k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3 :=
sorry

end increasing_interval_of_f_l342_342630


namespace angle_MOP_eq_angle_NOP_eq_angle_NOM_l342_342895

open EuclideanGeometry -- Assumes necessary Euclidean geometry constructs are available

noncomputable def tetrahedron (A B C D : Point) := True -- Abstract representation

-- Assume Point, Midpoint, distance, angle, and equality constructs would be provided by Mathlib

variables {A B C D O M N P : Point}
variables (h1 : distance O A = distance O B) 
          (h2 : distance O B = distance O C) 
          (h3 : distance O C = distance O D)
          (h4 : distance O D = distance O A)
          (h5 : distance A D + distance B D = distance A C + distance B C)
          (h6 : distance B D + distance C D = distance B A + distance C A)
          (h7 : distance C D + distance A D = distance C B + distance A B)
          (hM : M = Midpoint B C)
          (hN : N = Midpoint C A)
          (hP : P = Midpoint A B)

theorem angle_MOP_eq_angle_NOP_eq_angle_NOM :
  ∠ M O P = ∠ N O P ∧ ∠ N O P = ∠ N O M ∧ ∠ N O M = ∠ M O P :=
by
  sorry

end angle_MOP_eq_angle_NOP_eq_angle_NOM_l342_342895


namespace map_scale_l342_342360

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l342_342360


namespace map_length_represents_distance_l342_342326

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l342_342326


namespace rhombus_area_fraction_l342_342250

theorem rhombus_area_fraction :
  let grid_area := 36
  let vertices := [(2, 2), (4, 2), (3, 3), (3, 1)]
  let rhombus_area := 2
  rhombus_area / grid_area = 1 / 18 :=
by
  sorry

end rhombus_area_fraction_l342_342250


namespace sum_of_first_fifty_terms_l342_342767

theorem sum_of_first_fifty_terms :
  let a : ℕ → ℝ := λ n, 15 + (n - 1) * d_a in
  let b : ℕ → ℝ := λ n, 85 + (n - 1) * d_b in
  (∀ d_a d_b, a 1 = 15 ∧ b 1 = 85 ∧ a 50 + b 50 = 200 →
  ∑ i in finset.range 50, a (i + 1) + b (i + 1) = 7500) :=
begin
  intros d_a d_b h,
  let a : ℕ → ℝ := λ n, 15 + (n - 1) * d_a,
  let b : ℕ → ℝ := λ n, 85 + (n - 1) * d_b,
  sorry
end

end sum_of_first_fifty_terms_l342_342767


namespace compute_AB_squared_l342_342852

variable (A B C O H D : Type*)
variables [Coord A B C O H D] -- Implicitly assume A B C O H D to be points in coordinate plane

-- Conditions
variable (triangle_ABC : Triangle A B C)
variable (circumcenter_O : is_circumcenter O triangle_ABC)
variable (orthocenter_H : is_orthocenter H triangle_ABC)
variable (foot_of_altitude_D : is_foot_of_altitude D A B C)
variable (length_AD : distance A D = 12)
variable (ratio_BD_BC : distance B D = 1/4 * distance B C)
variable (parallel_OH_BC : is_parallel (line_through O H) (line_through B C))

-- Statement to be proven
theorem compute_AB_squared : (distance A B)^2 = 160 := sorry

end compute_AB_squared_l342_342852


namespace x_intercept_correct_l342_342477

def x_intercept_of_line (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let slope := (y2 - y1) / (x2 - x1)
  (0 - y1) / slope + x1

theorem x_intercept_correct :
  x_intercept_of_line (10, 3) (-6, -5) = 4 :=
by sorry

end x_intercept_correct_l342_342477


namespace y_coordinate_of_vertex_C_l342_342367

theorem y_coordinate_of_vertex_C : ∃ x : ℝ, ∃ h : ℝ, 
  (let A := (0, 0) in
  let B := (0, 5) in
  let D := (5, 5) in
  let E := (5, 0) in
  let pentagon_area := 50 in
  let square_area := 25 in
  let triangle_area := pentagon_area - square_area in
  let base_BD := 5 in
  let h' := h - 5 in
  5 * h' = 50 ∧ h = 15) :=
sorry

end y_coordinate_of_vertex_C_l342_342367


namespace imaginary_part_of_complex_z_l342_342585

noncomputable theory
open Complex

-- Define the condition
def condition : Prop :=
  ∀ z : ℂ, (1 / z) = (1 / (1 + 2 * I)) + (1 / (1 - I))

-- Define the proof problem
theorem imaginary_part_of_complex_z (z : ℂ) (h : condition z) : (z.im = -1/5) :=
by
  sorry

end imaginary_part_of_complex_z_l342_342585


namespace minimum_distinct_numbers_sum_l342_342866

/-- Minimum distinct numbers sum to others -/
theorem minimum_distinct_numbers_sum :
  ∃ (S : Finset ℝ), S.card = 7 ∧
  ∀ x ∈ S, ∃ a b c ∈ S, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ x = a + b + c :=
by 
  sorry

end minimum_distinct_numbers_sum_l342_342866


namespace map_distance_l342_342343

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l342_342343


namespace intersection_M_N_l342_342486

def M : Set ℕ := {3, 5, 6, 8}
def N : Set ℕ := {4, 5, 7, 8}

theorem intersection_M_N : M ∩ N = {5, 8} :=
  sorry

end intersection_M_N_l342_342486


namespace map_length_representation_l342_342273

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l342_342273


namespace pie_cut_minimum_pieces_l342_342003

theorem pie_cut_minimum_pieces : ∃ n, n = 20 ∧ 
  (∃ p1 p2 : ℕ, p1 = 10 ∧ p2 = 11 ∧
   n % p1 = 0 ∧ n % p2 = 0) :=
by
  use 20
  split
  · rfl
  · use 10, 11
    repeat { split; try {norm_num} }
    sorry

end pie_cut_minimum_pieces_l342_342003


namespace part1_part2_l342_342645

-- Definitions of vectors a and b
def vector_a (α : ℝ) : ℝ × ℝ :=
  (√2 * sin α, 1)

def vector_b (α : ℝ) : ℝ × ℝ :=
  (1, sin (α + π / 4))

-- Given conditions
variables {α : ℝ}
axiom α_condition (hx : 3^2 + 4^2 = 5^2) : sin α = 4 / 5 ∧ cos α = 3 / 5

-- First part of the problem
theorem part1 (h₀ : 3^2 + 4^2 = 5^2) : 
  let a := vector_a α in let b := vector_b α in a.1 * b.1 + a.2 * b.2 = 3 * √2 / 2 :=
sorry

-- Given condition for parallel vectors
axiom parallel_condition : ∀ α : ℝ, (√2 * sin α) * sin (α + π / 4) = 1 → sin α * cos α = cos α * cos α

-- Second part of the problem
theorem part2 (h₁ : (√2 * sin α) * sin (α + π / 4) = 1) : α = π / 4 :=
sorry

end part1_part2_l342_342645


namespace problem_1_problem_2_l342_342988

-- Problem 1

theorem problem_1 :
  ((real.sqrt (cbrt 1.5) * real.sqrt (sqrt 12))² + 8 * 1 ^ (3/4) - (-1/4) ^ (-2) - 5 * (0.12 * 1)) = 9 :=
by
  sorry

-- Problem 2

theorem problem_2 :
  real.log 25 + real.log 2 * real.log 50 + (real.log 2) ^ 2 - real.exp (3 * real.log 2) = -6 :=
by
  sorry

end problem_1_problem_2_l342_342988


namespace least_five_digit_congruent_to_six_mod_seventeen_l342_342447

theorem least_five_digit_congruent_to_six_mod_seventeen : ∃ x : ℕ, x ≥ 10000 ∧ x < 100000 ∧ x % 17 = 6 ∧ ∀ y : ℕ, y ≥ 10000 ∧ y < 100000 ∧ y % 17 = 6 → x ≤ y :=
  by
    sorry

end least_five_digit_congruent_to_six_mod_seventeen_l342_342447


namespace problem_statement_l342_342614

def unit_vectors (e1 e2 e3 : ℝ × ℝ × ℝ) : Prop :=
  e1 ≠ (0, 0, 0) ∧ e2 ≠ (0, 0, 0) ∧ e3 ≠ (0, 0, 0) ∧
  (e1.1^2 + e1.2^2 + e1.3^2 = 1) ∧ (e2.1^2 + e2.2^2 + e2.3^2 = 1) ∧ (e3.1^2 + e3.2^2 + e3.3^2 = 1) ∧
  (e1.1 + e2.1 + e3.1 = 0) ∧ (e1.2 + e2.2 + e3.2 = 0) ∧ (e1.3 + e2.3 + e3.3 = 0)

noncomputable def vector_a (e1 e2 e3 : ℝ × ℝ × ℝ) (x : ℝ) (n : ℕ) : ℝ × ℝ × ℝ :=
  (x * e1.1 + (n / x) * e2.1 + (x + (n / x)) * e3.1,
   x * e1.2 + (n / x) * e2.2 + (x + (n / x)) * e3.2,
   x * e1.3 + (n / x) * e2.3 + (x + (n / x)) * e3.3)

noncomputable def f (e1 e2 e3 : ℝ × ℝ × ℝ) (x : ℝ) (n : ℕ) : ℝ :=
  let a := vector_a e1 e2 e3 x n in
  real.sqrt (a.1^2 + a.2^2 + a.3^2)

theorem problem_statement (e1 e2 e3 : ℝ × ℝ × ℝ) (x : ℝ) (n : ℕ) :
  unit_vectors e1 e2 e3 →
  (real.arccos ((e1.1 * e2.1 + e1.2 * e2.2 + e1.3 * e2.3) / (real.sqrt (e1.1^2 + e1.2^2 + e1.3^2) * real.sqrt (e2.1^2 + e2.2^2 + e2.3^2)))) = 2 * real.pi / 3 ∧
  f e1 e2 e3 x n = real.sqrt (x^2 + (n / x)^2 - n) →
  (f e1 e2 e3 is_increasing_on Ioo (-sqrt n) 0 ∧ f e1 e2 e3 is_decreasing_on Ioo (-∞) (-sqrt n) ∧ f e1 e2 e3 is_decreasing_on Ioo 0 (sqrt n) ∧ f e1 e2 e3 is_increasing_on Ioo (sqrt n) ⊤) ∧
  (real.sqrt (min f) = real.sqrt n) :=
sorry

end problem_statement_l342_342614


namespace a_general_term_T_sum_first_n_l342_342696

-- Definitions based on given conditions
def a_1 : ℕ := 2
axiom d_nonzero : ∀ d : ℤ, d ≠ 0
axiom S_n (n : ℕ) : ℤ
axiom S_2n_eq_4S_n : ∀ (n : ℕ), S_n (2 * n) = 4 * S_n n

-- The sequence a_n
def a (n : ℕ) : ℤ := a_1 + (n - 1) * 4

-- The sequence b_n
def b (n : ℕ) : ℚ := 4 / (Real.sqrt (a n) + Real.sqrt (a (n + 1)))

-- The sum T_n of the first n terms of b_n
def T (n : ℕ) : ℚ := (Nat.sqrt (4 * n + 2).to_nat - Nat.sqrt 2)

-- Problem 1: Find the general term formula for the sequence a_n
theorem a_general_term (n : ℕ) : a n = 4 * n - 2 := sorry

-- Problem 2: Find the sum of the first n terms of the sequence b_n
theorem T_sum_first_n (n : ℕ) : 
  (∑ k in Finset.range n, b k) = T n := sorry

end a_general_term_T_sum_first_n_l342_342696


namespace matthew_younger_than_freddy_l342_342553

variables (M R F : ℕ)

-- Define the conditions
def sum_of_ages : Prop := M + R + F = 35
def matthew_older_than_rebecca : Prop := M = R + 2
def freddy_age : Prop := F = 15

-- Prove the statement "Matthew is 4 years younger than Freddy."
theorem matthew_younger_than_freddy (h1 : sum_of_ages M R F) (h2 : matthew_older_than_rebecca M R) (h3 : freddy_age F) :
    F - M = 4 := by
  sorry

end matthew_younger_than_freddy_l342_342553


namespace analytic_expression_on_1_2_l342_342768

noncomputable def f : ℝ → ℝ :=
  sorry

theorem analytic_expression_on_1_2 (x : ℝ) (h1 : 1 < x) (h2 : x < 2) :
  f x = Real.logb (1 / 2) (x - 1) :=
sorry

end analytic_expression_on_1_2_l342_342768


namespace inclination_angle_l342_342163

theorem inclination_angle (P Q : ℝ × ℝ) (hP : P = (1, 2)) (hQ : Q = (4, 2 + sqrt 3)) :
  ∃ α : ℝ, 0 ≤ α ∧ α ≤ π ∧ tan α = (2 + sqrt 3 - 2) / (4 - 1) ∧ α = π / 6 :=
by
  sorry

end inclination_angle_l342_342163


namespace map_length_representation_l342_342309

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l342_342309


namespace inscribed_rectangle_max_area_l342_342841

open Real

theorem inscribed_rectangle_max_area 
  (p a : ℝ) (h_p : 0 < p) (h_a : 0 < a) 
  (f : ℝ → ℝ) 
  (h_parabola : ∀ x, f x = sqrt (2 * p * x)) :
  ∃ x : ℝ, 
  (0 < x ∧ x ≤ a) ∧ 
  area (x : ℝ) = 2 * (a - x) * sqrt (2 * p * x) ∧
  (x = a / 3) ∧
  area (a / 3) = 4 / 3 * a * sqrt (2 / 3 * a * p) :=
by
  sorry

end inscribed_rectangle_max_area_l342_342841


namespace child_ticket_cost_l342_342410

theorem child_ticket_cost
  (C : ℕ) -- The cost of a child ticket
  (total_tickets : ℕ := 130) -- Total number of tickets sold
  (total_receipts : ℕ := 840) -- Total receipts in dollars
  (adult_ticket_cost : ℕ := 12) -- Cost of an adult ticket
  (adult_tickets_sold : ℕ := 40) -- Number of adult tickets sold
  (child_tickets_sold : ℕ := total_tickets - adult_tickets_sold) -- Number of child tickets sold
  (adult_revenue : ℕ := adult_tickets_sold * adult_ticket_cost) -- Revenue from adult tickets
  (total_expense : ℕ := adult_revenue + child_tickets_sold * C) -- Total expense from tickets sold
: total_expense = total_receipts → C = 4 :=
begin
  intro h,
  sorry
end

end child_ticket_cost_l342_342410


namespace sum_f_lt_zero_l342_342629

noncomputable def f (x : ℝ) : ℝ := 2^(-x) - 2^x

theorem sum_f_lt_zero (a b c : ℝ) (h1 : a + b > 0) (h2 : b + c > 0) (h3 : c + a > 0) : 
  f(a) + f(b) + f(c) < 0 := 
sorry

end sum_f_lt_zero_l342_342629


namespace jason_commute_with_detour_l342_342754

theorem jason_commute_with_detour (d1 d2 d3 d4 d5 : ℝ) 
  (h1 : d1 = 4)     -- Distance from house to first store
  (h2 : d2 = 6)     -- Distance between first and second store
  (h3 : d3 = d2 + (2/3) * d2) -- Distance between second and third store without detour
  (h4 : d4 = 3)     -- Additional distance due to detour
  (h5 : d5 = d1)    -- Distance from third store to work
  : d1 + d2 + (d3 + d4) + d5 = 27 :=
by
  sorry

end jason_commute_with_detour_l342_342754


namespace no_function_exists_f_l342_342783

noncomputable def alpha : ℝ := sorry
axiom alpha_gt_half : 1 / 2 < alpha

theorem no_function_exists_f : 
  ¬ ∃ f : ℝ → ℝ, (∀ x, 0 ≤ x ∧ x ≤ 1 → (f x = 1 + alpha * ∫ t in set.Icc (x:ℝ) 1, f t * f (t - x))) :=
by {
  sorry
}

end no_function_exists_f_l342_342783


namespace solve_vec_decomposition_l342_342885

noncomputable def vec_decomposition (x p q r : ℝ × ℝ × ℝ) : Prop :=
x = (-3 : ℝ) * p + 2 * q + 5 * r

theorem solve_vec_decomposition :
  ∃ α β γ : ℝ, let p := (1, 1, 0) in
                let q := (0, 1, -2) in
                let r := (1, 0, 3) in
                let x := (2, -1, 11) in
                x = α * p + β * q + γ * r ∧ α = -3 ∧ β = 2 ∧ γ = 5 := 
begin
  use [-3, 2, 5],
  simp,
end

#print axioms solve_vec_decomposition

end solve_vec_decomposition_l342_342885


namespace map_scale_l342_342359

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l342_342359


namespace incorrect_statement_D_l342_342883

theorem incorrect_statement_D :
  ¬ (∀ (x : ℝ) (y : ℝ), 
    (R_squared : ℝ) (r : ℝ) (SSR : ℝ),
    (R_squared > 0 → (r = -0.982 → abs r = 0.982 → SSR > 0 → 
    (x → y = -3 * x + 0.8 → y = y - 3 * x))))

sorrry

end incorrect_statement_D_l342_342883


namespace martin_family_ice_cream_cost_l342_342823

def price_kiddie_scoop := 3
def price_regular_scoop := 4
def price_double_scoop := 6
def price_sprinkles := 1
def price_nuts := 1.50
def discount := 0.10

def mr_and_mrs_martin := 2 * (price_regular_scoop + price_nuts)
def young_children := 2 * (price_kiddie_scoop + price_sprinkles)
def teenage_children := 3 * (price_double_scoop + price_nuts + price_sprinkles)
def elderly_grandparents := (price_regular_scoop + price_sprinkles) + (price_regular_scoop + price_nuts)

def total_cost := mr_and_mrs_martin + young_children + teenage_children + elderly_grandparents
def discounted_cost := total_cost * (1 - discount)

theorem martin_family_ice_cream_cost : discounted_cost = 49.50 :=
by
  -- proof will go here
  sorry

end martin_family_ice_cream_cost_l342_342823


namespace f_is_odd_f_is_decreasing_l342_342134

noncomputable def f (x : ℝ) := x / (x^2 - 1)

-- Definition of the domain
def inDomain (x : ℝ) : Prop := x ≠ 1 ∧ x ≠ -1

-- Ⅰ. Prove that f(x) is an odd function
theorem f_is_odd (x : ℝ) (h : inDomain x) : f (-x) = -f(x) := sorry

-- Ⅱ. Prove that f(x) is decreasing on the interval (-1, 1)
theorem f_is_decreasing (x1 x2 : ℝ) (h1 : -1 < x1 ∧ x1 < 1) (h2 : -1 < x2 ∧ x2 < 1) (h3 : x1 < x2) : f x1 > f x2 := sorry

end f_is_odd_f_is_decreasing_l342_342134


namespace triangle_symmetry_vertex_in_polygon_l342_342522

theorem triangle_symmetry_vertex_in_polygon (T M : Set Point) (P : Point)
  (convex_M : convex M) (symm_M : centrally_symmetric M) 
  (T_in_M : T ⊆ M)
  (P_in_T : P ∈ T) 
  (T' : Set Point := symmetry P T):
  ∃ v ∈ vertices T', v ∈ M := 
by
  sorry

end triangle_symmetry_vertex_in_polygon_l342_342522


namespace map_distance_l342_342349

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l342_342349


namespace subset_count_excluding_element_2_l342_342840

def original_set : Set ℕ := {1, 2, 3, 4, 5}
def reduced_set : Set ℕ := {1, 3, 4, 5}

theorem subset_count_excluding_element_2 : 
  (set.powerset reduced_set).card = 16 := by 
sorry

end subset_count_excluding_element_2_l342_342840


namespace solve_equation_l342_342817

noncomputable def equation (x : ℝ) : ℝ :=
  (|Real.sin x| + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x))

theorem solve_equation (x : ℝ) (k : ℤ) :
  (equation x = 2 / Real.sqrt 3) ↔
  (∃ k : ℤ, x = π / 12 + 2 * k * π ∨ x = 7 * π / 12 + 2 * k * π ∨ x = -5 * π / 6 + 2 * k * π) :=
sorry

end solve_equation_l342_342817


namespace locus_of_points_l342_342078

-- Definitions of angle, point, segment, circle, and triangle are assumed to exist in the Mathlib library.

-- Define the locus theorem
theorem locus_of_points (A B : Point) (alpha : Angle) (C C' : Point) :
  (angle A C B = alpha) ∧ (angle A C' B = alpha) ∧ (C ≠ C') ∧ (C, C' are on the opposite sides of line (line_through A B)) ↔
  (forall M : Point, (angle A M B = alpha) → (M lies on the arc (circle_circumscribed_around A B C) excluding A and B ∨ M lies on the arc (circle_circumscribed_around A B C') excluding A and B)) :=
sorry

end locus_of_points_l342_342078


namespace regular_polygon_sides_l342_342939

-- Definitions based on conditions in the problem
def exterior_angle (n : ℕ) : ℝ := 360 / n

theorem regular_polygon_sides (n : ℕ) (h : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l342_342939


namespace correct_answer_is_ln_abs_l342_342026

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_monotonically_increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, (0 < x ∧ x < y) → f x ≤ f y

theorem correct_answer_is_ln_abs :
  is_even_function (fun x => Real.log (abs x)) ∧ is_monotonically_increasing_on_pos (fun x => Real.log (abs x)) ∧
  ¬ is_even_function (fun x => x^3) ∧
  ¬ is_monotonically_increasing_on_pos (fun x => Real.cos x) :=
by
  sorry

end correct_answer_is_ln_abs_l342_342026


namespace equilateral_triangle_in_ellipse_l342_342536

theorem equilateral_triangle_in_ellipse :
  ∃ (p q : ℕ), Nat.coprime p q ∧ (0 < p ∧ 0 < q) ∧ p + q = 133 ∧
  (∃ A B C : ℝ × ℝ,
    -- Points and conditions regarding the triangle.
    A = (0, 1) ∧
    B.1 = -C.1 ∧ B.2 = C.2 ∧
    -- Triangle is equilateral and symmetrically distributed along y-axis.
    (B.2 - 1 = sqrt 3 * B.1) ∧
    -- Length calculation of the sides.
    let side_square := ((B.1 - A.1)^2 + (B.2 - A.2)^2) in
    p / q = side_square ∧
    -- Conditions related to the ellipse.
    (A.1^2 + 3 * A.2^2 = 3) ∧
    (B.1^2 + 3 * B.2^2 = 3) ∧
    (C.1^2 + 3 * C.2^2 = 3)) :=
sorry

end equilateral_triangle_in_ellipse_l342_342536


namespace irrational_sqrt3_l342_342880

theorem irrational_sqrt3 : 
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ (sqrt 3) = (p / q : ℝ)) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (-2 : ℝ) = (p / q : ℝ)) ∧ 
  (∃ (p q : ℤ), q ≠ 0 ∧ (0 : ℝ) = (p / q : ℝ)) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (-1 / 2 : ℝ) = (p / q : ℝ)) := 
by
  sorry

end irrational_sqrt3_l342_342880


namespace probability_two_mismatches_l342_342377

theorem probability_two_mismatches : 
  let total_events := 6
  let mismatch_events := 3
  let probability := mismatch_events / total_events
  probability = 1 / 2 :=
by
  let total_events : ℕ := 6
  let mismatch_events : ℕ := 3
  let probability : ℚ := mismatch_events / total_events
  show probability = 1 / 2
  sorry

end probability_two_mismatches_l342_342377


namespace base_six_equals_base_b_l342_342384

theorem base_six_equals_base_b (b : ℕ) (h1 : 3 * 6 ^ 1 + 4 * 6 ^ 0 = 22)
  (h2 : b ^ 2 + 2 * b + 1 = 22) : b = 3 :=
sorry

end base_six_equals_base_b_l342_342384


namespace map_scale_l342_342262

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l342_342262


namespace sum_first_7_terms_l342_342100

noncomputable def a (n : ℕ) := a1 + (n - 1) * d

noncomputable def S (n : ℕ) := (n * (2 * a1 + (n - 1) * d)) / 2

axiom a3_eq_neg1 : a 3 = -1
axiom geom_mean_condition : (a 4)^2 = -a 1 * (a 1 + 5 * d)

-- We need to prove that S 7 = -14 under the given conditions
theorem sum_first_7_terms :
  S 7 = -14 :=
by
  sorry

end sum_first_7_terms_l342_342100


namespace map_length_representation_l342_342307

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l342_342307


namespace max_abc_value_l342_342247

theorem max_abc_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b + b * c = 518) (h2 : a * b - a * c = 360) : 
  a * b * c ≤ 1008 :=
sorry

end max_abc_value_l342_342247


namespace star_cell_value_l342_342538

theorem star_cell_value : 
  ∀ (grid : vector (vector ℕ 5) 5),
  (∀ i, ∃! (c : ℕ), c ∈ {1,2,3,4,5} ∧ (∀ j, grid i j ∈ {1,2,3,4,5})) →
  (∀ j, ∃! (r : ℕ), r ∈ {1,2,3,4,5} ∧ (∀ i, grid i j ∈ {1,2,3,4,5})) →
  (∀ i j, let idx_r := (i / 2) * 2 + 1, idx_c := (j / 2) * 2 + 1 in 
           ∃! (v : ℕ), v ∈ {1,2,3,4,5} ∧ vector.nth (vector.nth grid idx_r) idx_c = v) →
  vector.nth (vector.nth grid 3) 3 = 1 :=
by sorry

end star_cell_value_l342_342538


namespace problem_1_problem_2_l342_342047

-- Problem 1
theorem problem_1 : (3 - Real.pi)^0 - 2^2 + (1/2)^(-2) = 1 := by
  sorry

-- Problem 2
variables (a b : ℝ)

theorem problem_2 : ((a * b^2)^2 - 2 * a * b^4) / (a * b^4) = a - 2 := by
  sorry

end problem_1_problem_2_l342_342047


namespace find_m_l342_342835

theorem find_m (m : ℝ) (h : ∀ A B : ℝ × ℝ, A = (-2, m) → B = (m, 4) → ∃ θ : ℝ, θ = 45 → tan θ = (B.snd - A.snd) / (B.fst - A.fst)) :
  m = 1 :=
by
  -- Place the statement and assumptions here
  sorry

end find_m_l342_342835


namespace map_scale_l342_342264

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l342_342264


namespace age_of_b_l342_342887

-- Define the conditions as per the problem statement
variables (A B C D E : ℚ)

axiom cond1 : A = B + 2
axiom cond2 : B = 2 * C
axiom cond3 : D = A - 3
axiom cond4 : E = D / 2 + 3
axiom cond5 : A + B + C + D + E = 70

theorem age_of_b : B = 16.625 :=
by {
  -- Placeholder for the proof
  sorry
}

end age_of_b_l342_342887


namespace percentage_decrease_in_sale_l342_342801

theorem percentage_decrease_in_sale (P Q : ℝ) (D : ℝ)
  (h1 : 1.80 * P * Q * (1 - D / 100) = 1.44 * P * Q) : 
  D = 20 :=
by
  -- Proof goes here
  sorry

end percentage_decrease_in_sale_l342_342801


namespace solve_IvanTsarevich_problem_l342_342201

variable (BabaYaga : ℕ → Prop)
variable (Truthful : ℕ → Prop)
variable (Lying : ℕ → Prop)
variable (correct_road : ℕ)
variable (incorrect_road : ℕ)

-- Baba Yaga answers truthfully every other day
axiom alternate_day : ∀ n : ℕ, (BabaYaga n → Truthful n ↔ Truthful (n - 1) ∨ Lying (n - 1))

-- Ivan can ask only one question
def IvanTsarevich_one_question : Prop :=
  ∀ (today : ℕ), (∃ answer : ℕ, 
    (Truthful today → BabaYaga today = correct_road) → 
    (Lying today → BabaYaga today = incorrect_road) →
     ((Truthful today ∨ Lying today) → 
       (BabaYaga (today - 1) = correct_road ↔ BabaYaga today = incorrect_road) →
       (correct_road ≠ incorrect_road) →
         true))

theorem solve_IvanTsarevich_problem : IvanTsarevich_one_question BabaYaga correct_road incorrect_road :=
by 
  -- The statement says that Ivan can determine the correct road by asking:
  -- "What would you have answered me yesterday if I had asked which road leads to Koschei's kingdom?"
  sorry

end solve_IvanTsarevich_problem_l342_342201


namespace T_gt_2_l342_342758

noncomputable def T : ℝ :=
  1 / (3 - real.sqrt 8) -
  1 / (real.sqrt 8 - real.sqrt 7) +
  1 / (real.sqrt 7 - real.sqrt 6) -
  1 / (real.sqrt 6 - real.sqrt 5) +
  1 / (real.sqrt 5 - 2)

theorem T_gt_2 : T > 2 :=
sorry

end T_gt_2_l342_342758


namespace table_length_is_77_l342_342009

theorem table_length_is_77 :
  ∀ (x : ℕ), (∀ (sheets: ℕ), sheets = 72 → x = (5 + sheets)) → x = 77 :=
by {
  sorry
}

end table_length_is_77_l342_342009


namespace elastic_collision_inelastic_collision_l342_342421

-- Given conditions for Case A and Case B
variables (L V : ℝ) (m : ℝ) -- L is length of the rods, V is the speed, m is mass of each sphere

-- Prove Case A: The dumbbells separate maintaining their initial velocities
theorem elastic_collision (h1 : L > 0) (h2 : V > 0) (h3 : m > 0) :
  -- After a perfectly elastic collision, the dumbbells separate maintaining their initial velocities
  true := sorry

-- Prove Case B: The dumbbells start rotating around the collision point with angular velocity V / (2 * L)
theorem inelastic_collision (h1 : L > 0) (h2 : V > 0) (h3 : m > 0) :
  -- After a perfectly inelastic collision, the dumbbells start rotating around the collision point with angular velocity V / (2 * L)
  true := sorry

end elastic_collision_inelastic_collision_l342_342421


namespace map_scale_representation_l342_342329

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l342_342329


namespace symmetry_probability_l342_342961

open_locale big_operators
open set

noncomputable def point := (ℤ × ℤ)

def is_center (p : point) : Prop :=
  p = (6, 6)

def is_symmetry_line (p q : point) : Prop :=
  (q.1 = 6) ∨ (q.2 = 6) ∨ (q.1 - q.2 = 0) ∨ (q.1 + q.2 = 12)

def probability_symmetry_line := 
  let grid_points : finset point := (finset.Icc (0, 0) (10, 10)) in
  let remaining_points := grid_points \ {(6, 6)},
      symmetric_points := { q ∈ remaining_points | is_symmetry_line (6, 6) q } in
  (symmetric_points.card : ℝ) / (remaining_points.card : ℝ)

theorem symmetry_probability : probability_symmetry_line = 1 / 3 :=
  sorry

end symmetry_probability_l342_342961


namespace quadrilateral_area_correct_l342_342568

noncomputable def areaQuadrilateral (d₁ d₂ : ℝ) (θ : ℝ) : ℝ :=
  0.5 * d₁ * d₂ * Real.sin θ

theorem quadrilateral_area_correct :
  let d₁ := 40
  let d₂ := 25
  let θ := 75 * Real.pi / 180
  areaQuadrilateral d₁ d₂ θ ≈ 482.95 :=
by
  let d₁ := 40
  let d₂ := 25
  let θ := 75 * Real.pi / 180
  sorry

end quadrilateral_area_correct_l342_342568


namespace bisection_method_next_interval_l342_342858

noncomputable def f (x : ℝ) := x^3 - 2*x - 5

theorem bisection_method_next_interval :
  [2, 3] → 
  let mid := (2 + 3) / 2 in 
  (f 2) * (f mid) < 0 ↔ ′(interval → ℝ → Prop) :=  
begin
  let f := λ x, x^ 3 - 2 * x - 5,
  by interval_start_2( f, 2, 3 ),
  sorry
  -- further steps to evaluate
end

end bisection_method_next_interval_l342_342858


namespace sin_double_angle_solution_l342_342720

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_solution_l342_342720


namespace amount_spent_l342_342528

-- Definitions
def initial_amount : ℕ := 54
def amount_left : ℕ := 29

-- Proof statement
theorem amount_spent : initial_amount - amount_left = 25 :=
by
  sorry

end amount_spent_l342_342528


namespace richard_remaining_distance_l342_342812

noncomputable def remaining_distance : ℝ :=
  let d1 := 45
  let d2 := d1 / 2 - 8
  let d3 := 2 * d2 - 4
  let d4 := (d1 + d2 + d3) / 3 + 3
  let d5 := 0.7 * d4
  let total_walked := d1 + d2 + d3 + d4 + d5
  635 - total_walked

theorem richard_remaining_distance : abs (remaining_distance - 497.5166) < 0.0001 :=
by
  sorry

end richard_remaining_distance_l342_342812


namespace sum_tens_ones_digit_of_7_pow_11_l342_342459

/--
The sum of the tens digit and the ones digit of (3+4)^{11} is 7.
-/
theorem sum_tens_ones_digit_of_7_pow_11 : 
  let number := (3 + 4)^11
  let tens_digit := (number / 10) % 10
  let ones_digit := number % 10
  tens_digit + ones_digit = 7 :=
by
  sorry

end sum_tens_ones_digit_of_7_pow_11_l342_342459


namespace binom_1500_1_eq_1500_l342_342998

theorem binom_1500_1_eq_1500 : binom 1500 1 = 1500 :=
by
  sorry

end binom_1500_1_eq_1500_l342_342998


namespace sum_of_circumferences_eq_28pi_l342_342500

theorem sum_of_circumferences_eq_28pi (R r : ℝ) (h1 : r = (1:ℝ)/3 * R) (h2 : R - r = 7) : 
  2 * Real.pi * R + 2 * Real.pi * r = 28 * Real.pi :=
by
  sorry

end sum_of_circumferences_eq_28pi_l342_342500


namespace solve_trig_eq_l342_342816

theorem solve_trig_eq (x : ℝ) (k : ℤ) :
  (x = (π / 12) + 2 * k * π ∨
   x = (7 * π / 12) + 2 * k * π ∨
   x = (7 * π / 6) + 2 * k * π ∨
   x = -(5 * π / 6) + 2 * k * π) →
  (|Real.sin x| + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x)) = 2 / Real.sqrt 3 :=
sorry

end solve_trig_eq_l342_342816


namespace solution_set_inequality_l342_342028

theorem solution_set_inequality (x : ℝ) : (-2 * x + 3 < 0) ↔ (x > 3 / 2) := by 
  sorry

end solution_set_inequality_l342_342028


namespace jane_percentage_decrease_l342_342753

theorem jane_percentage_decrease
  (B H : ℝ) -- Number of bears Jane makes per week and hours she works per week
  (H' : ℝ) -- Number of hours Jane works per week with assistant
  (h1 : H' = 0.9 * H) -- New working hours when Jane works with an assistant
  (h2 : H ≠ 0) -- Ensure H is not zero to avoid division by zero
  : ((H - H') / H) * 100 = 10 := 
by calc
  ((H - H') / H) * 100
      = ((H - 0.9 * H) / H) * 100 : by rw [h1]
  ... = (0.1 * H / H) * 100 : by simp
  ... = 0.1 * 100 : by rw [div_self h2]
  ... = 10 : by norm_num

end jane_percentage_decrease_l342_342753


namespace number_of_sides_is_15_l342_342167

variable {n : ℕ} -- n is the number of sides

-- Define the conditions
def sum_of_all_but_one_angle (n : ℕ) : Prop :=
  180 * (n - 2) - 2190 > 0 ∧ 180 * (n - 2) - 2190 < 180

-- State the theorem to be proven
theorem number_of_sides_is_15 (n : ℕ) (h : sum_of_all_but_one_angle n) : n = 15 :=
sorry

end number_of_sides_is_15_l342_342167


namespace geometric_sequence_sum_l342_342109

variables (a : ℕ → ℤ) (q : ℤ)

-- assumption that the sequence is geometric
def geometric_sequence (a : ℕ → ℤ) (q : ℤ) : Prop := 
  ∀ n, a (n + 1) = a n * q

noncomputable def a2 := a 2
noncomputable def a3 := a 3
noncomputable def a4 := a 4
noncomputable def a5 := a 5
noncomputable def a6 := a 6
noncomputable def a7 := a 7

theorem geometric_sequence_sum
  (h_geom : geometric_sequence a q)
  (h1 : a2 + a3 = 1)
  (h2 : a3 + a4 = -2) :
  a5 + a6 + a7 = 24 :=
sorry

end geometric_sequence_sum_l342_342109


namespace sum_of_factors_36_l342_342869

theorem sum_of_factors_36 : (∑ i in (finset.filter (λ d, 36 % d = 0) (finset.range (36 + 1))), i) = 91 :=
by {
  sorry
}

end sum_of_factors_36_l342_342869


namespace regular_polygons_from_cube_cut_l342_342884

theorem regular_polygons_from_cube_cut (n : ℕ) :
  (∃ (plane : ℝ × ℝ × ℝ → Prop), is_regular_polygon (plane ∩ cube) n) ↔ (n = 3 ∨ n = 4 ∨ n = 6) := 
sorry

end regular_polygons_from_cube_cut_l342_342884


namespace necessary_not_sufficient_cond_l342_342897

theorem necessary_not_sufficient_cond (α β : ℝ) :
  (sin α)^2 + (sin β)^2 = 1 → (∀ α β, (sin α + cos β = 0) → (sin α)^2 + (sin β)^2 = 1) ∧ ¬ (∀ α β, ((sin α)^2 + (sin β)^2 = 1) → (sin α + cos β = 0)) :=
by
  sorry

end necessary_not_sufficient_cond_l342_342897


namespace hyperbola_asymptote_angle_l342_342636

open Real

theorem hyperbola_asymptote_angle (a : ℝ) (h : 0 < a) :
  (∃ (a : ℝ), (1 / a = tan(π / 6)) ∧ (a > 0)) → a = sqrt 3 :=
by
  sorry

end hyperbola_asymptote_angle_l342_342636


namespace distinct_m_count_l342_342238

noncomputable def countDistinctMValues : Nat :=
  let pairs := [(1, 36), (2, 18), (3, 12), (4, 9), (6, 6), 
                (-1, -36), (-2, -18), (-3, -12), (-4, -9), (-6, -6)]
  let ms := pairs.map (λ p => p.1 + p.2)
  ms.eraseDups.length

theorem distinct_m_count :
  countDistinctMValues = 10 := sorry

end distinct_m_count_l342_342238


namespace Jean_cards_per_grandkid_l342_342755

theorem Jean_cards_per_grandkid (g c A : ℕ) (h1 : g = 3) (h2 : c = 80) (h3 : A = 480) : A / (g * c) = 2 :=
by
  -- Using given conditions
  rw [h1, h2, h3]
  -- Compute the number of cards per grandkid
  sorry

end Jean_cards_per_grandkid_l342_342755


namespace triangle_ABC_area_l342_342863

def point : Type := ℚ × ℚ

def triangle_area (A B C : point) : ℚ :=
  let (x1, y1) := A;
  let (x2, y2) := B;
  let (x3, y3) := C;
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem triangle_ABC_area :
  let A : point := (-5, 4)
  let B : point := (1, 7)
  let C : point := (4, -3)
  triangle_area A B C = 34.5 :=
by
  sorry

end triangle_ABC_area_l342_342863


namespace categorize_sets_l342_342068

def given_numbers : Set ℝ := {2021, -1.7, 2 / 5, 0, -6, 23 / 8, Real.pi / 2}

def positive_numbers : Set ℝ := {x | 0 < x}
def integers : Set ℤ := {x | Real.ofInt x ∈ given_numbers}
def negative_fractions : Set ℝ := {x | ∃ (p q : ℤ), p < 0 ∧ q > 0 ∧ x = p / q}
def positive_rationals : Set ℚ := {x | 0 < x ∧ Real.ofRat x ∈ given_numbers}

theorem categorize_sets :
  positive_numbers ∩ given_numbers = {2021, 2 / 5, 23 / 8, Real.pi / 2} ∧
  integers ∩ given_numbers = {2021, 0, -6} ∧
  negative_fractions ∩ given_numbers = {-1.7} ∧
  positive_rationals ∩ given_numbers = {2021, 2 / 5, 23 / 8} :=
by
  sorry

end categorize_sets_l342_342068


namespace three_power_not_square_l342_342777

theorem three_power_not_square (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) : ¬ ∃ k : ℕ, k * k = 3^m + 3^n + 1 := by 
  sorry

end three_power_not_square_l342_342777


namespace explain_why_curved_road_distance_shortens_l342_342468

-- A condition: When a curved road is straightened, the distance is shortened
def curved_road_distance_shortens: ∀ (A B: Point) (curve_path: Path A B) (straight_path: LineSegment A B), 
  length curve_path > length straight_path :=
begin
  sorry
end

-- B condition: The shortest distance between two points is a line segment
axiom shortest_distance_is_line_segment (A B: Point): ∀ (p: Path A B), 
  length p ≥ length (LineSegment.mk A B)

-- Main theorem: Use the geometric principle from condition B to prove the problem statement
theorem explain_why_curved_road_distance_shortens
    (A B: Point) (curve_path: Path A B) (straight_path: LineSegment A B)
    (h: length curve_path > length straight_path):
  shortest_distance_is_line_segment A B curve_path :=
begin
  sorry
end

end explain_why_curved_road_distance_shortens_l342_342468


namespace distance_is_20_sqrt_6_l342_342850

-- Definitions for problem setup
def distance_between_parallel_lines (r d : ℝ) : Prop :=
  ∃ O C D E F P Q : ℝ, 
  40^2 * 40 + (d / 2)^2 * 40 = 40 * r^2 ∧ 
  15^2 * 30 + (d / 2)^2 * 30 = 30 * r^2

-- The main statement to be proved
theorem distance_is_20_sqrt_6 :
  ∀ r d : ℝ,
  distance_between_parallel_lines r d →
  d = 20 * Real.sqrt 6 :=
sorry

end distance_is_20_sqrt_6_l342_342850


namespace four_digit_numbers_gt_3000_l342_342152

theorem four_digit_numbers_gt_3000 (d1 d2 d3 d4 : ℕ) (h_digits : (d1, d2, d3, d4) = (2, 0, 5, 5)) (h_distinct_4digit : (d1 * 1000 + d2 * 100 + d3 * 10 + d4) > 3000) :
  ∃ count, count = 3 := sorry

end four_digit_numbers_gt_3000_l342_342152


namespace tom_drives_12_miles_before_karen_wins_l342_342479

theorem tom_drives_12_miles_before_karen_wins (
  karen_speed : ℝ,
  tom_speed : ℝ,
  karen_late : ℝ,
  distance_to_beat : ℝ
) : 
  (karen_speed = 60 ∧ tom_speed = 45 ∧ karen_late = 4 / 60 ∧ distance_to_beat = 4) →
  ∃ y : ℝ, y = 12 :=
by
  sorry

end tom_drives_12_miles_before_karen_wins_l342_342479


namespace remainder_sum_of_integers_division_l342_342775

theorem remainder_sum_of_integers_division (n S : ℕ) (hn_cond : n > 0) (hn_sq : n^2 + 12 * n - 3007 ≥ 0) (hn_square : ∃ m : ℕ, n^2 + 12 * n - 3007 = m^2):
  S = n → S % 1000 = 516 := 
sorry

end remainder_sum_of_integers_division_l342_342775


namespace quadratic_min_value_l342_342744

theorem quadratic_min_value (f : ℕ → ℚ) (n : ℕ)
  (h₁ : f n = 6)
  (h₂ : f (n + 1) = 5)
  (h₃ : f (n + 2) = 5) :
  ∃ c : ℚ, c = 39 / 8 ∧ ∀ x : ℕ, f x ≥ c :=
by
  sorry

end quadratic_min_value_l342_342744


namespace sequence_inequality_l342_342146

noncomputable def a_seq : ℕ+ → ℝ
| ⟨1, _⟩ := 1
| ⟨n + 1, h⟩ := let a_n := a_seq ⟨n, Nat.succ_pos' _⟩
                in a_n + a_n^2 / (n + 1)^2

theorem sequence_inequality (n : ℕ+) : 
  (2 * n + 2) / (n + 3) < a_seq (n + 1) ∧ a_seq (n + 1) < n + 1 := 
sorry

end sequence_inequality_l342_342146


namespace journey_time_l342_342525

-- Conditions
variables (speed1 speed2 distance total_distance : ℕ)
variables (T T1 T2 : ℚ)

-- Constants from the problem
def speed1 : ℕ := 21
def speed2 : ℕ := 24
def total_distance : ℕ := 672
def half_distance : ℕ := total_distance / 2

-- Definitions from the problem
def T1 : ℚ := half_distance / speed1
def T2 : ℚ := half_distance / speed2
def T : ℚ := T1 + T2

-- The problem statement to prove
theorem journey_time : T = 30 := by
  sorry

end journey_time_l342_342525


namespace map_scale_l342_342354

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l342_342354


namespace book_distribution_methods_l342_342848

theorem book_distribution_methods :
  let novels := 2
  let picture_books := 2
  let students := 3
  (number_ways : ℕ) = 12 :=
by
  sorry

end book_distribution_methods_l342_342848


namespace sum_of_digits_7_pow_11_l342_342462

/-
  Let's define the problem with the given conditions and expected answer.
  We need to prove that the sum of the tens digit and the ones digit of 7^11 is 7.
-/

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_7_pow_11 : tens_digit (7 ^ 11) + ones_digit (7 ^ 11) = 7 :=
sorry

end sum_of_digits_7_pow_11_l342_342462


namespace triangle_area_range_l342_342602

variable (a b c : ℝ)
variable (S : ℝ)
variable (A B C : ℝ)
variable [fact (0 < a)]
variable [fact (0 < b)]
variable [fact (0 < c)]
variable [fact (a = sqrt 3)]
variable [fact (b^2 + c^2 - b * c = 3)]
variable [fact (0 < A)]
variable [fact (0 < B)]
variable [fact (0 < C)]
variable [fact (A + B + C = π)]
variable [fact (A < π / 2)]
variable [fact (B < π / 2)]
variable [fact (C < π / 2)]
variable [fact (S = 1/2 * b * c * sin A)]

theorem triangle_area_range : sqrt 3 / 2 < S ∧ S ≤ 3 * sqrt 3 / 4 :=
sorry

end triangle_area_range_l342_342602


namespace find_a_l342_342139

noncomputable def f (x : ℝ) : ℝ := x + 1 + Real.log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

theorem find_a (a : ℝ) : 
  (∃ x : ℝ, g a x = 2 * x) ∧ (deriv f 1 = 2) ∧ f 1 = 2 → a = 4 :=
by
  -- Math proof goes here
  sorry

end find_a_l342_342139


namespace trig_expression_value_l342_342050

theorem trig_expression_value :
    (3 * (Real.tan (30 * Real.pi / 180)) ^ 2 + 
     (Real.tan (60 * Real.pi / 180)) ^ 2 - 
     (Real.cos (30 * Real.pi / 180)) * 
     (Real.sin (60 * Real.pi / 180)) * 
     (Real.cot (45 * Real.pi / 180))) = 7 / 4 := 
by
  sorry

end trig_expression_value_l342_342050


namespace angle_between_plane_base_l342_342932

def angle_between_planes (a b c : ℝ) : ℝ :=
  Real.arctan (Real.sqrt (a^2 + b^2) / c)

theorem angle_between_plane_base (a b c θ : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0):
  θ = angle_between_planes a b c ↔ θ = Real.arctan (Real.sqrt (a^2 + b^2) / c) :=
by
  sorry

end angle_between_plane_base_l342_342932


namespace function_seven_zeros_l342_342587

theorem function_seven_zeros (ω : ℝ) (h_pos : ω > 0) :
  has_seven_distinct_zeros (λ x : ℝ, if x > 0 then x + abs (Real.log x) - 2 else Real.sin (ω * x + Real.pi / 4) - 1 / 2) ↔
  ω ∈ Set.Ico (49 / 12) (65 / 12) :=
sorry

end function_seven_zeros_l342_342587


namespace complex_number_quadrant_l342_342627

noncomputable def given_z (m : ℝ) : ℂ :=
  (1 : ℂ) - (m * complex.I)

theorem complex_number_quadrant
  (z : ℂ) (h1 : ∃ m : ℝ, z * complex.I = complex.I + m)
  (h2 : z.im = 1) :
  0 < z.re ∧ 0 < z.im :=
by
  sorry

end complex_number_quadrant_l342_342627


namespace add_base6_numbers_l342_342022

def base6_to_base10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6^1 + c * 6^0

def base10_to_base6 (n : ℕ) : (ℕ × ℕ × ℕ) := 
  (n / 6^2, (n % 6^2) / 6^1, (n % 6^2) % 6^1)

theorem add_base6_numbers : 
  let n1 := 3 * 6^1 + 5 * 6^0
  let n2 := 2 * 6^1 + 5 * 6^0
  let sum := n1 + n2
  base10_to_base6 sum = (1, 0, 4) :=
by
  -- Proof steps would go here
  sorry

end add_base6_numbers_l342_342022


namespace largest_spherical_ball_radius_l342_342966

-- Give definitions based on conditions
def inner_radius : ℝ := 3
def outer_radius : ℝ := 5
def circle_center : ℝ × ℝ × ℝ := (4, 0, 1)
def circle_radius : ℝ := 1

-- Statement for the proof problem
theorem largest_spherical_ball_radius :
  ∃ r : ℝ, r = 4 ∧ 
  (let O := (0, 0, r) in
   let P := (4, 0, 1) in
   let horizontal_distance := (4 : ℝ) in
   let vertical_distance := r - 1 in
   let hypotenuse := r + 1 in
   (horizontal_distance^2 + vertical_distance^2 = hypotenuse^2)) :=
begin
  use 4,
  split,
  { refl, },
  sorry
end

end largest_spherical_ball_radius_l342_342966


namespace route_Y_saves_time_l342_342797

-- Definitions based on conditions
def distance_X : ℝ := 8 -- Route X Distance in miles
def speed_X : ℝ := 40 -- Route X Speed in miles per hour
def distance_Y1 : ℝ := 6.5 -- Route Y normal distance in miles
def speed_Y1 : ℝ := 50 -- Route Y normal speed in miles per hour
def distance_Y2 : ℝ := 0.5 -- Route Y construction zone distance in miles
def speed_Y2 : ℝ := 10 -- Route Y construction zone speed in miles per hour

-- Calculating time
def time_X : ℝ := distance_X / speed_X * 60 -- Time in minutes
def time_Y1 : ℝ := distance_Y1 / speed_Y1 * 60 -- Time for part of Route Y in minutes
def time_Y2 : ℝ := distance_Y2 / speed_Y2 * 60 -- Time for construction zone in Route Y in minutes
def total_time_Y : ℝ := time_Y1 + time_Y2 -- Total time for Route Y in minutes

-- Goal: Prove Route Y saves 1.2 minutes compared to Route X
theorem route_Y_saves_time : (time_X - total_time_Y = 1.2) :=
by
  -- Outline of proof without completing
  sorry

end route_Y_saves_time_l342_342797


namespace candy_bars_division_l342_342380

theorem candy_bars_division (chocolate_total caramel_total nougat_total bags : ℕ) 
  (h_chocolate : chocolate_total = 12) 
  (h_caramel : caramel_total = 18) 
  (h_nougat : nougat_total = 15) 
  (h_bags : bags = 5) : 
  ∃ (chocolates_per_bag caramels_per_bag nougats_per_bag : ℕ), 
    chocolates_per_bag = 2 ∧ caramels_per_bag = 3 ∧ nougats_per_bag = 3 :=
begin
  sorry
end

end candy_bars_division_l342_342380


namespace median_lengths_l342_342042

theorem median_lengths {a b c : ℝ} (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) 
  (triangle_ineq : a + b > c ∧ a + c > b ∧ b + c > a) : 
  let s_a := (2 * b^2 + 2 * c^2 - a^2) / 4, s_b := (2 * a^2 + 2 * c^2 - b^2) / 4, s_c := (2 * a^2 + 2 * b^2 - c^2) / 4 in
  sqrt s_a / 2 = (sqrt (2 * b^2 + 2 * c^2 - a^2)) / 2 ∧
  sqrt s_b / 2 = (sqrt (2 * a^2 + 2 * c^2 - b^2)) / 2 ∧
  sqrt s_c / 2 = (sqrt (2 * a^2 + 2 * b^2 - c^2)) / 2 :=
by
  sorry

end median_lengths_l342_342042


namespace angle_MAN_eq_45_l342_342362

-- Define the isosceles right triangle with right angle at B
variables {A B C M N : Type}
variable [inner_product_space ℝ A]

-- Define the points B, C, M, N on an appropriate type
variables {b c m n : A}
variable {bc_angle : ∠ B A C = 90}
variable {ha : is_iso_right_triangle B A C bc_angle}

-- Define the condition BM^2 - MN^2 + NC^2 = 0
variable {h_condition : dist B M ^ 2 - dist M N ^ 2 + dist N C ^ 2 = 0}

-- Prove the angle MAN equals 45 degrees
theorem angle_MAN_eq_45
  (h_am : is_iso_right_triangle B A C bc_angle)
  (h_condition : dist B M ^ 2 - dist M N ^ 2 + dist N C ^ 2 = 0) :
  ∠ M A N = 45 :=
sorry

end angle_MAN_eq_45_l342_342362


namespace sum_of_tens_and_ones_digits_of_seven_eleven_l342_342452

theorem sum_of_tens_and_ones_digits_of_seven_eleven :
  let n := (3 + 4) ^ 11 in 
  (let ones := n % 10 in
   let tens := (n / 10) % 10 in
   ones + tens = 7) := 
by sorry

end sum_of_tens_and_ones_digits_of_seven_eleven_l342_342452


namespace map_scale_representation_l342_342333

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l342_342333


namespace digits_difference_divisible_by_2_l342_342248

-- Definition of number of digits in a number
def num_digits (A : ℝ) : ℝ := ⌊log10 A⌋ + 1

-- Conditions
axiom log10_2_and_5_sum : log10 2 + log10 5 = 1
noncomputable def A : ℝ := 5 ^ 1090701
noncomputable def B : ℝ := 2 ^ 1090701

-- The theorem to be proven
theorem digits_difference_divisible_by_2 :
  (num_digits A - num_digits B) % 2 = 0 := sorry

end digits_difference_divisible_by_2_l342_342248


namespace units_digit_fraction_l342_342872

theorem units_digit_fraction :
  let n := 30 * 31 * 32 * 33 * 34 * 35
  let d := 1500
  let fraction := n / d
  (fraction % 10) = 4 := by
  let n := 30 * 31 * 32 * 33 * 34 * 35
  let d := 1500
  let fraction := n / d
  have h₁ : 30 = 2 * 3 * 5 := by rfl
  have h₂ : 31 = 31 := by rfl
  have h₃ : 32 = 2 ^ 5 := by rfl
  have h₄ : 33 = 3 * 11 := by rfl
  have h₅ : 34 = 2 * 17 := by rfl
  have h₆ : 35 = 5 * 7 := by rfl
  have h₇ : 1500 = 2 ^ 2 * 3 * 5 ^ 3 := by rfl
  have num_factorization : n = 2 ^ 7 * 3 ^ 2 * 5 ^ 2 * 31 * 11 * 17 * 7 := by
    rw [← h₁, ← h₂, ← h₃, ← h₄, ← h₅, ← h₆]
    ring
  have den_factorization : d = 2 ^ 2 * 3 * 5 ^ 3 := by rw h₇
  have simplified_fraction : fraction = 2 ^ 5 * 3 * 31 * 11 * 17 * 7 := by
    rw [num_factorization, den_factorization]
    field_simp
    ring
  have : (2 ^ 5 * 3 * 31 * 11 * 17 * 7 % 10) = 4 := by sorry
  exact this

end units_digit_fraction_l342_342872


namespace polygon_enclosure_l342_342516

theorem polygon_enclosure (m n : ℕ) (h1 : m = 8) (h2 : ∀ k, k < m → regular_polygon n k)
  (h3 : ∀ i, i < m → vertex_match m n i) : n = 8 :=
by
  sorry -- Proof not required, just the statement.

end polygon_enclosure_l342_342516


namespace tangent_line_equation_monotonicity_of_f_g_has_two_distinct_zeros_l342_342634

-- Define the function f(x) as given
def f (a : ℝ) (x : ℝ) : ℝ := log (a * x) - (1 / 3) * x^3

-- Define the theorem for Part I
theorem tangent_line_equation (a : ℝ) (ha : a = 2) (x : ℝ) (hx : x = 1 / 2) :
    21 * x - 12 * (f a x) - 11 = 0 :=
sorry

-- Define the theorem for Part II
theorem monotonicity_of_f (a : ℝ) (ha : a ≠ 0) :
    (a < 0 → ∀ x, x < 0 → f' a x < 0) ∧
    (a > 0 →
      ∀ x, (0 < x ∧ x < 1 → f' a x > 0) ∧
           (1 < x → f' a x < 0)) :=
sorry

-- Define the function g(x) as given in Part III
def g (x : ℝ) (t : ℝ) : ℝ := f 1 x + t

-- Define the theorem for Part III
theorem g_has_two_distinct_zeros (t : ℝ) : 
    (∃ x₁ x₂, x₁ ≠ x₂ ∧ g x₁ t = 0 ∧ g x₂ t = 0) ↔ t ∈ Ioc (1 / 3) ∞ :=
sorry

end tangent_line_equation_monotonicity_of_f_g_has_two_distinct_zeros_l342_342634


namespace exact_recovery_probability_exceed_recovery_probability_l342_342559

-- Define the given probabilities as conditions
def first_year_probs : List (ℚ × ℚ) := [(1.0, 0.2), (0.9, 0.4), (0.8, 0.4)]
def second_year_probs : List (ℚ × ℚ) := [(1.5, 0.3), (1.25, 0.3), (1.0, 0.4)]

-- Function to calculate the combined probability for exact recovery
noncomputable def probability_exact_recovery :=
  let p1 := (first_year_probs.find? (λ p => p.fst = 1.0)).map (λ p => p.snd) |>.getOrElse 0
  let p2 := (second_year_probs.find? (λ p => p.fst = 1.0)).map (λ p => p.snd) |>.getOrElse 0
  let p3 := (first_year_probs.find? (λ p => p.fst = 0.8)).map (λ p => p.snd) |>.getOrElse 0
  let p4 := (second_year_probs.find? (λ p => p.fst = 1.25)).map (λ p => p.snd) |>.getOrElse 0
  p1 * p2 + p3 * p4

-- Function to calculate the combined probability for exceeding recovery
noncomputable def probability_exceed_recovery :=
  let p1 := (first_year_probs.find? (λ p => p.fst = 1.0)).map (λ p => p.snd) |>.getOrElse 0
  let p2 := (second_year_probs.find? (λ p => p.fst = 1.5)).map (λ p => p.snd) |>.getOrElse 0
  let p3 := (first_year_probs.find? (λ p => p.fst = 0.9)).map (λ p => p.snd) |>.getOrElse 0
  let p4 := (second_year_probs.find? (λ p => p.fst = 1.5)).map (λ p => p.snd) |>.getOrElse 0
  let p5 := (second_year_probs.find? (λ p => p.fst = 1.25)).map (λ p => p.snd) |>.getOrElse 0
  p1 * p2 + p3 * p4 + p3 * p5

-- Prove that probability of exact recovery is 0.2
theorem exact_recovery_probability : probability_exact_recovery = 0.20 :=
  by
    sorry

-- Prove that probability of exceeding recovery is 0.3
theorem exceed_recovery_probability : probability_exceed_recovery = 0.30 :=
  by
    sorry

end exact_recovery_probability_exceed_recovery_probability_l342_342559


namespace isosceles_triangle_perimeter_l342_342680

theorem isosceles_triangle_perimeter
  (a b : ℕ)
  (ha : a = 3)
  (hb : b = 7)
  (h_iso : ∃ (c : ℕ), (c = a ∨ c = b) ∧ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a))) :
  a + b + b = 17 :=
by
  sorry

end isosceles_triangle_perimeter_l342_342680


namespace find_eccentricty_of_conic_l342_342116

noncomputable def eccentricity_of_conic_section (m : ℝ) : ℝ :=
  if h : m = 6 then (sqrt 5 / sqrt 6) else sqrt 7

theorem find_eccentricty_of_conic
  (m : ℝ)
  (h_geom : 4, m, 9 in_geom_seq)
  : eccentricity_of_conic_section m = sqrt 30 / 6 ∨ eccentricity_of_conic_section m = sqrt 7 := by
  sorry

end find_eccentricty_of_conic_l342_342116


namespace simultaneous_equations_solution_l342_342577

-- Definition of the two equations
def eq1 (m x y : ℝ) : Prop := y = m * x + 5
def eq2 (m x y : ℝ) : Prop := y = (3 * m - 2) * x + 6

-- Lean theorem statement to check if the equations have a solution
theorem simultaneous_equations_solution (m : ℝ) :
  (m ≠ 1) ↔ ∃ x y : ℝ, eq1 m x y ∧ eq2 m x y := 
sorry

end simultaneous_equations_solution_l342_342577


namespace nine_linked_rings_min_moves_4_l342_342396

def seq (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n % 2 = 0 then 2 * seq (n - 1) - 1
  else 2 * seq (n - 1) + 2

theorem nine_linked_rings_min_moves_4 : seq 4 = 7 :=
by {
  unfold seq,
  have h1 : seq 1 = 1 := rfl,
  rw [h1],
  have h2 : seq 2 = 2 * seq 1 - 1 := rfl,
  rw [h1, h2],
  have h3 : seq 3 = 2 * seq 2 + 2 := rfl,
  rw [h1, h2, h3],
  have h4 : seq 4 = 2 * seq 3 - 1 := rfl,
  rw [h1, h2, h3, h4],
  sorry -- This is where we skip the detailed proof.
}

end nine_linked_rings_min_moves_4_l342_342396


namespace isosceles_triangle_perimeter_l342_342683

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 3) (h2 : b = 7) : 
∃ (c : ℕ), 
  (c = 7 ∧ a = 3 ∧ b = 7 ∧ a + b + c = 17) ∨ 
  (c = 3 ∧ a = 7 ∧ b = 7 ∧ a + b + c = 17) :=
  sorry

end isosceles_triangle_perimeter_l342_342683


namespace k_value_l342_342566

open Real

noncomputable def k_from_roots (α β : ℝ) : ℝ := - (α + β)

theorem k_value (k : ℝ) (α β : ℝ) (h1 : α + β = -k) (h2 : α * β = 8) (h3 : (α+3) + (β+3) = k) (h4 : (α+3) * (β+3) = 12) : k = 3 :=
by
  -- Here we skip the proof as instructed.
  sorry

end k_value_l342_342566


namespace circle_geometry_l342_342805

theorem circle_geometry 
  (A M B K : Point)  -- Define points A, M, B, and K
  (on_circle : OnCircle A M B)  -- Points A, M, B are on the same circle
  (AM_eq_BM : |AM| = |BM|)  -- AM congruent to BM
  (K_on_arc_AMB : K_on_arc A M B K)  -- K is on the arc AMB
  (connect_MK_AK_BK : (connected M K A B))  -- M and K are connected to A and B
  :
  |AK| * |BK| = |AM|^2 - |KM|^2 := 
sorry

end circle_geometry_l342_342805


namespace marble_arrangement_l342_342065

theorem marble_arrangement :
  let green_marbles := 6
  let initial_red_marbles := 6
  let additional_red_marbles := 12
  let total_red_marbles := initial_red_marbles + additional_red_marbles
  let total_marbles := green_marbles + total_red_marbles
  let arrangments := Nat.choose total_marbles total_red_marbles
in
  total_red_marbles = 18 ∧ arrangments % 1000 = 564 := 
by
  sorry

end marble_arrangement_l342_342065


namespace original_fraction_l342_342498

theorem original_fraction (n d : ℝ) (h1 : n + d = 5.25) (h2 : (n + 3) / (2 * d) = 1 / 3) : n / d = 2 / 33 :=
by
  sorry

end original_fraction_l342_342498


namespace find_n_l342_342765

-- Definitions of the lengths of the sides of the triangle
def AB : ℝ := 80
def AC : ℝ := 150
def BC : ℝ := 170

-- Definition of the inradius of ΔABC
def r₁ : ℝ := 30

-- Definitions of coordinates for centers O₂ and O₃
def O₂ : ℝ × ℝ := (50, 120 + 24)
def O₃ : ℝ × ℝ := (50 + 18.75, 120)

-- Distance between centers of C₂ and C₃ in terms of sqrt(10n)
def distance_between_centers_C₂_C₃ : ℝ := sqrt (10 * 35.15625)

-- The theorem that needs to be proven
theorem find_n : ∃ n : ℝ, distance_between_centers_C₂_C₃ = sqrt (10 * n) := 
sorry

end find_n_l342_342765


namespace sin_double_angle_solution_l342_342723

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_solution_l342_342723


namespace variance_shifted_l342_342599

variable {n : ℕ} (x : Fin n → ℝ)

def variance (x : Fin n → ℝ) : ℝ :=
  let mean := (∑ i, x i) / n
  (∑ i, (x i - mean)^2) / n

theorem variance_shifted {n : ℕ} (x : Fin n → ℝ) (h : variance x = 7) :
  variance (λ i => x i - 1) = 7 :=
by
  sorry

end variance_shifted_l342_342599


namespace star_example_l342_342590

section star_operation

variables (x y z : ℕ) 

-- Define the star operation as a binary function
def star (a b : ℕ) : ℕ := a * b

-- Given conditions
axiom star_idempotent : ∀ x : ℕ, star x x = 0
axiom star_associative : ∀ x y z : ℕ, star x (star y z) = (star x y) + z

-- Main theorem to be proved
theorem star_example : star 1993 1935 = 58 :=
sorry

end star_operation

end star_example_l342_342590


namespace map_scale_representation_l342_342334

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l342_342334


namespace exact_time_now_l342_342202

/-- Given that it is between 9:00 and 10:00 o'clock,
and nine minutes from now, the minute hand of a watch
will be exactly opposite the place where the hour hand
was six minutes ago, show that the exact time now is 9:06
-/
theorem exact_time_now 
  (t : ℕ)
  (h1 : t < 60)
  (h2 : ∃ t, 6 * (t + 9) - (270 + 0.5 * (t - 6)) = 180 ∨ 6 * (t + 9) - (270 + 0.5 * (t - 6)) = -180) :
  t = 6 := 
sorry

end exact_time_now_l342_342202


namespace cone_base_circumference_l342_342013

theorem cone_base_circumference (r : ℝ) (θ : ℝ) (circ_res : ℝ) :
  r = 4 → θ = 270 → circ_res = 6 * Real.pi :=
by 
  sorry

end cone_base_circumference_l342_342013


namespace sin_2phi_l342_342733

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end sin_2phi_l342_342733


namespace find_guanaco_numbers_l342_342505

-- Define the concept of a four-digit guanaco number
def is_digit (x : ℕ) : Prop := x ≥ 0 ∧ x < 10

def is_four_digit_number (abcd : ℕ) : Prop := 
  let a := abcd / 1000 in
  let b := (abcd % 1000) / 100 in
  let c := (abcd % 100) / 10 in
  let d := abcd % 10 in
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ a ≠ 0

def is_guanaco (n : ℕ) : Prop :=
  let a := n / 1000 in
  let b := (n % 1000) / 100 in
  let c := (n % 100) / 10 in
  let d := n % 10 in
  let x := 10 * a + b in
  let y := 10 * c + d in
  is_four_digit_number n ∧ (x * y > 0 ∧ (100 * x + y) % (x * y) = 0)

theorem find_guanaco_numbers :
  { n : ℕ | is_guanaco n } = {1352, 1734} :=
sorry

end find_guanaco_numbers_l342_342505


namespace sin_double_angle_l342_342717

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l342_342717


namespace isosceles_triangle_perimeter_l342_342679

theorem isosceles_triangle_perimeter
  (a b : ℕ)
  (ha : a = 3)
  (hb : b = 7)
  (h_iso : ∃ (c : ℕ), (c = a ∨ c = b) ∧ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a))) :
  a + b + b = 17 :=
by
  sorry

end isosceles_triangle_perimeter_l342_342679


namespace area_of_quadrilateral_ABDF_l342_342931

-- Definitions for points and dimensions
def AC : ℝ := 40
def AE : ℝ := 24
def B := AC / 3
def F := AE / 2

-- Area calculations for the given conditions
theorem area_of_quadrilateral_ABDF : 
    let AC := 40 in 
    let AE := 24 in
    let area_rect := AC * AE in
    let area_tri_BCD := (1 / 2) * (AC - AC / 3) * AE in
    let area_tri_EFD := (1 / 2) * (AE / 2) * AC in
    area_rect - area_tri_BCD - area_tri_EFD = 400 :=
by
    sorry

end area_of_quadrilateral_ABDF_l342_342931


namespace part_a_l342_342808

theorem part_a
  (A B C : Type)
  (A1 : Type)
  (b c a : ℝ)
  (m_a : ℝ)
  (Apollonius_Theorem : AB^2 + AC^2 = 2 * AA_1^2 + 2 * (a / 2)^2)
  (midpoint : A1 = (B + C) / 2) :
  m_a^2 = (2 * b^2 + 2 * c^2 - a^2) / 4 :=
sorry

end part_a_l342_342808


namespace map_scale_representation_l342_342338

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l342_342338


namespace number_of_possible_committees_l342_342036

theorem number_of_possible_committees :
  let departments := ["Physics", "Chemistry", "Biology"]
  let male_professors_per_department := 3
  let female_professors_per_department := 3
  let committee_size := 6
  ∃ (man_case committee_num: ℕ),
    (man_case = 1 ∧ ∃ (d1 d2 d3: ℕ), 
      d1 = male_professors_per_department 
      ∧ d2 = female_professors_per_department 
      ∧ d3 = 9 ∧ d1 * d2 * d3^3 = 729)
  ∨ 
  (man_case = 2 ∧ ∃ (d1 d2 d3: ℕ),
    d1 = 3
    ∧ d2 = 9 
    ∧ d3 = 6 
    ∧ 3 * d1 * (3 * d1) * 9 * d3 = 486) 
  ∧ committee_num = (729 + 486) :
  1215 = 729 + 486 :=
by {
  sorry
}

end number_of_possible_committees_l342_342036


namespace map_scale_l342_342263

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l342_342263


namespace eight_lines_eleven_points_l342_342804

-- Define the maximum number of intersection points for n lines
def max_intersections (n : ℕ) : ℕ := n * (n - 1) / 2

-- Given condition: 5 lines intersect at exactly 6 points
axiom five_lines_six_points : max_intersections 5 = 10 ∧ ∃ (n : ℕ), n = 6

-- Define the statement: There exists a configuration of 8 lines intersecting at exactly 11 points
theorem eight_lines_eleven_points : ∃ (l : Finset (Set Point)), l.card = 8 ∧ (l.intersections.card = 11) :=
by
  sorry

end eight_lines_eleven_points_l342_342804


namespace sin_2phi_l342_342730

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end sin_2phi_l342_342730


namespace combined_height_of_cylinders_l342_342920

/-- Given three cylinders with perimeters 6 feet, 9 feet, and 11 feet respectively,
    and rolled out on a rectangular plate with a diagonal of 19 feet,
    the combined height of the cylinders is 26 feet. -/
theorem combined_height_of_cylinders
  (p1 p2 p3 : ℝ) (d : ℝ)
  (h_p1 : p1 = 6) (h_p2 : p2 = 9) (h_p3 : p3 = 11) (h_d : d = 19) :
  p1 + p2 + p3 = 26 :=
sorry

end combined_height_of_cylinders_l342_342920


namespace necessary_but_not_sufficient_condition_l342_342901

theorem necessary_but_not_sufficient_condition 
  (α β : ℝ) :
  ¬ (∀ α β, (sin α + cos β = 0) ↔ (sin^2 α + sin^2 β = 1)) ∧ 
    (∀ α β, (sin α + cos β = 0) → (sin^2 α + sin^2 β = 1)) := 
  sorry

end necessary_but_not_sufficient_condition_l342_342901


namespace cassandra_watch_time_loss_l342_342051

theorem cassandra_watch_time_loss 
  (initial_time : ℕ := 8) 
  (watch_time_at_3pm : ℕ := 14) 
  (watch_loss_rate : ℚ := 2 / 60)
  (watch_target_time : ℕ := 11) :
  let actual_time := (watch_target_time * 60 / 58 : ℚ) in 
  actual_time = 683 :=
by
  sorry

end cassandra_watch_time_loss_l342_342051


namespace least_five_digit_congruent_6_mod_17_l342_342443

theorem least_five_digit_congruent_6_mod_17 : ∃ n: ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 6 ∧ ∀ m: ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 17 = 6 → n ≤ m :=
sorry

end least_five_digit_congruent_6_mod_17_l342_342443


namespace regular_polygon_sides_l342_342937

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (k : ℕ), (k : ℕ) * 18 = 360) : n = 20 :=
by
  -- Proof body here
  sorry

end regular_polygon_sides_l342_342937


namespace shortest_distance_skew_lines_l342_342369

theorem shortest_distance_skew_lines
  (a b : Line)
  (skew_a_b : a ∩ b = ∅)
  (A : Point)
  (B : Point)
  (A_on_a : A ∈ a)
  (B_on_b : B ∈ b)
  (AB_perpendicular : ∀ B', B' ∈ b → AB ⊥ b)
  (M : Point)
  (N : Point)
  (M_on_a : M ∈ a)
  (M_ne_A : M ≠ A)
  (N_on_b : N ∈ b)
  (N_ne_B : N ≠ B)
  (α : Plane)
  (α_parallel_a : α ∥ a)
  (M_to_α_perp : ∃ P : Point, P ∈ α ∧ M ⊥ P ∧ distance M P = distance A B)
  : distance M N > distance A B := sorry

end shortest_distance_skew_lines_l342_342369


namespace add_in_base_12_eq_l342_342527

def A85 : ℕ := 10 * 12^2 + 8 * 12 + 5
def 2B4 : ℕ := 2 * 12^2 + 11 * 12 + 4
def result : ℕ := 1 * 12^3 + 1 * 12^2 + 7 * 12 + 9

theorem add_in_base_12_eq :
  A85 + 2B4 = result :=
sorry

end add_in_base_12_eq_l342_342527


namespace necessary_but_not_sufficient_condition_l342_342900

theorem necessary_but_not_sufficient_condition 
  (α β : ℝ) :
  ¬ (∀ α β, (sin α + cos β = 0) ↔ (sin^2 α + sin^2 β = 1)) ∧ 
    (∀ α β, (sin α + cos β = 0) → (sin^2 α + sin^2 β = 1)) := 
  sorry

end necessary_but_not_sufficient_condition_l342_342900


namespace speed_in_still_water_l342_342924

theorem speed_in_still_water (U D : ℝ) (hU : U = 15) (hD : D = 25) : (U + D) / 2 = 20 :=
by
  rw [hU, hD]
  norm_num

end speed_in_still_water_l342_342924


namespace solve_gcd_problem_l342_342433

def gcd_problem : Prop :=
  gcd 153 119 = 17

theorem solve_gcd_problem : gcd_problem :=
  by
    sorry

end solve_gcd_problem_l342_342433


namespace sum_tens_ones_digit_of_7_pow_11_l342_342460

/--
The sum of the tens digit and the ones digit of (3+4)^{11} is 7.
-/
theorem sum_tens_ones_digit_of_7_pow_11 : 
  let number := (3 + 4)^11
  let tens_digit := (number / 10) % 10
  let ones_digit := number % 10
  tens_digit + ones_digit = 7 :=
by
  sorry

end sum_tens_ones_digit_of_7_pow_11_l342_342460


namespace convex_f_l342_342141

variable {α : Type*}
variable [LinearOrder α] [OrderedAddCommGroup α] [Module ℝ α] [OrderedSMul ℝ α] [OrderedAddCommMonoid α]

noncomputable def f (x : ℝ) (a k : ℝ) : ℝ := (1 / x^k) + a

theorem convex_f (a k : ℝ) (ha : 0 < a) (hk : 0 < k) :
  ConvexOn ℝ (Ioi 0) (λ x => (1 / x^k) + a) :=
  sorry

end convex_f_l342_342141


namespace map_length_representation_l342_342278

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l342_342278


namespace exists_students_A_B_l342_342911

-- Define the range for scores
def score_range := {i : ℕ | i ≤ 7 }

-- Define a student as a vector of three scores
structure Student :=
(scores : vector ℕ 3)
(h_scores : ∀ k, scores.to_list.nth k ∈ score_range)

-- N define the number of students
def N := 249

-- Define the main proposition
theorem exists_students_A_B (students : vector Student N) :
  ∃ (A B : Student), ∀ k < 3, (A.scores.to_list.nth k) ≥ (B.scores.to_list.nth k) :=
sorry

end exists_students_A_B_l342_342911


namespace map_scale_l342_342268

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l342_342268


namespace map_representation_l342_342258

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l342_342258


namespace map_length_scale_l342_342302

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l342_342302


namespace map_distance_l342_342340

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l342_342340


namespace sin_double_angle_l342_342706

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_l342_342706


namespace function_is_convex_l342_342142

variable (a k : ℝ) (ha : 0 < a) (hk : 0 < k)

def f (x : ℝ) := 1 / x ^ k + a

theorem function_is_convex (x : ℝ) (hx : 0 < x) : ConvexOn ℝ Set.Ioi { x : ℝ | 0 < x } f := 
sorry

end function_is_convex_l342_342142


namespace variance_shifted_l342_342600

variable {n : ℕ} (x : Fin n → ℝ)

def variance (x : Fin n → ℝ) : ℝ :=
  let mean := (∑ i, x i) / n
  (∑ i, (x i - mean)^2) / n

theorem variance_shifted {n : ℕ} (x : Fin n → ℝ) (h : variance x = 7) :
  variance (λ i => x i - 1) = 7 :=
by
  sorry

end variance_shifted_l342_342600


namespace sin_2phi_l342_342731

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end sin_2phi_l342_342731


namespace max_abc_l342_342245

theorem max_abc : ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (a * b + b * c = 518) ∧ 
  (a * b - a * c = 360) ∧ 
  (a * b * c = 1008) := 
by {
  -- Definitions of a, b, c satisfying the given conditions.
  -- Proof of the maximum value will be placed here (not required as per instructions).
  sorry
}

end max_abc_l342_342245


namespace sandcastle_height_difference_l342_342982

theorem sandcastle_height_difference :
  ∀ (Miki_sandcastle_height Sister_sandcastle_height : ℝ),
    Miki_sandcastle_height = 0.83 →
    Sister_sandcastle_height = 0.5 →
    Miki_sandcastle_height - Sister_sandcastle_height = 0.33 :=
by
  intros Miki_sandcastle_height Sister_sandcastle_height
  intro h1
  intro h2
  rw [h1, h2]
  norm_num
  sorry

end sandcastle_height_difference_l342_342982


namespace area_of_annulus_l342_342029

variables (R r x : ℝ) (hRr : R > r) (h : R^2 - r^2 = x^2)

theorem area_of_annulus : π * R^2 - π * r^2 = π * x^2 :=
by
  sorry

end area_of_annulus_l342_342029


namespace tangents_secant_intersection_ratio_l342_342981

open scoped Classical

variables {α : Type*}
variables [EuclideanGeometry α]

theorem tangents_secant_intersection_ratio (O A B C D P E : α)
  (hPA : tangent P O A) (hPB : tangent P O B)
  (hPCD : secant P C D O)
  (hE : intersection (line A B) (line P D) = E) :
  PC / PD = CE / DE :=
sorry

end tangents_secant_intersection_ratio_l342_342981


namespace median_is_8_l342_342173
-- Lean Statement of the problem

def data_set : List ℕ := [8, 10, 10, 4, 6]

/-- Prove that the median of the data_set is 8 -/
theorem median_is_8 : List.median data_set = 8 := by
  sorry

end median_is_8_l342_342173


namespace base6_addition_correct_l342_342020

-- Define a function to convert a base 6 digit to its base 10 equivalent
def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 4
  | 5 => 5
  | d => 0 -- for illegal digits, fallback to 0

-- Define a function to convert a number in base 6 to base 10
def convert_base6_to_base10 (n : Nat) : Nat :=
  let units := base6_to_base10 (n % 10)
  let tens := base6_to_base10 ((n / 10) % 10)
  let hundreds := base6_to_base10 ((n / 100) % 10)
  units + 6 * tens + 6 * 6 * hundreds

-- Define a function to convert a base 10 number to a base 6 number
def base10_to_base6 (n : Nat) : Nat :=
  (n % 6) + 10 * ((n / 6) % 6) + 100 * ((n / (6 * 6)) % 6)

theorem base6_addition_correct : base10_to_base6 (convert_base6_to_base10 35 + convert_base6_to_base10 25) = 104 := by
  sorry

end base6_addition_correct_l342_342020


namespace map_length_scale_l342_342305

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l342_342305


namespace least_sum_of_exponents_for_2015_l342_342159

theorem least_sum_of_exponents_for_2015 : 
  ∃ (s : Finset ℕ), (2015 = ∑ k in s, 2^k) ∧ s.sum id = 49 :=
by
  use {0, 2, 3, 4, 6, 7, 8, 9, 10}
  split
  · -- proof that sum of 2^s = 2015
    sorry
  · -- proof that sum of exponents = 49
    sorry

end least_sum_of_exponents_for_2015_l342_342159


namespace sin_2phi_l342_342727

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end sin_2phi_l342_342727


namespace rectangular_region_area_l342_342008

theorem rectangular_region_area (a : ℝ) (ha : 0 < a) : 
  (2x - ay)^2 = 25a^2 ∧ (5ax + 2y)^2 = 36a^2 → 
  area = 120 * a^2 / real.sqrt (100 * a^2 + 16 + 100 * a^4) :=
by
  sorry

end rectangular_region_area_l342_342008


namespace average_age_combined_l342_342382

theorem average_age_combined (n₁ n₂ n₃ : ℕ) (a₁ a₂ a₃ : ℕ) (N : ℕ) (A : ℕ) :
  n₁ = 40 → a₁ = 12 →
  n₂ = 60 → a₂ = 35 →
  n₃ = 10 → a₃ = 45 →
  N = n₁ + n₂ + n₃ →
  A = (n₁ * a₁ + n₂ * a₂ + n₃ * a₃) →
  A / N = 275454545 / 10000000 :=
by exactlysorry

end average_age_combined_l342_342382


namespace map_scale_l342_342352

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l342_342352


namespace quadratic_inequality_sufficient_necessary_l342_342018

theorem quadratic_inequality_sufficient_necessary (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + a > 0) ↔ 0 < a ∧ a < 4 :=
by
  -- proof skipped
  sorry

end quadratic_inequality_sufficient_necessary_l342_342018


namespace distinct_values_g_l342_342058

def g (x : ℝ) : ℝ := ∑ k in finset.range 15, (⌊k * x + 3⌋ - (k + 3) * ⌊x⌋)

theorem distinct_values_g (x : ℝ) (hx : x ≥ 0) :
  ∃ n, n = ∑ k in finset.range 15, nat.totient (k + 1) + 1 ∧ n ∈ {52, 48, 55, 60, 65} :=
sorry

end distinct_values_g_l342_342058


namespace find_six_digit_perfect_square_l342_342526

def is_six_digit_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

def digits_distinct (n : ℕ) : Prop :=
  let d := Nat.digits 10 n
  d.nodup

def digits_in_ascending_order (n : ℕ) : Prop :=
  let d := Nat.digits 10 n
  d = d.insertion_sort Nat.ble

theorem find_six_digit_perfect_square :
  ∃ n : ℕ, is_six_digit_number n ∧ digits_distinct n ∧ digits_in_ascending_order n ∧ ∃ k : ℕ, k^2 = n ∧ n = 134689 :=
by
  sorry

end find_six_digit_perfect_square_l342_342526


namespace range_of_omega_l342_342588

noncomputable def f : ℝ → ℝ
| x => if x > 0 then x + abs (log x) - 2 else sin (ω * x + π / 4) - 1 / 2

theorem range_of_omega (ω : ℝ) : 
  (∀ x, f x = 0 → (x > 0 ∧ x = real.log x + 2) ∨ (x ≤ 0 ∧ sin(ω * x + π / 4) = 1 / 2)) 
  ∧ (7 = (set.univ.filter (λ x, f x = 0)).count) 
  → ω ∈ set.Ico (49 / 12 : ℝ) (65 / 12 : ℝ) :=
sorry

end range_of_omega_l342_342588


namespace largest_n_l342_342572

def C (n k : ℕ) : ℕ := nat.choose n k

theorem largest_n (n : ℕ) (answer : ℕ) :
  (C n 1 + 2 * C n 2 + 3 * C n 3 + ... + n * C n n < 2006) →
  n = answer :=
sorry

end largest_n_l342_342572


namespace x_eq_zero_sufficient_not_necessary_l342_342111

theorem x_eq_zero_sufficient_not_necessary (x : ℝ) : 
  (x = 0 → x^2 - 2 * x = 0) ∧ (x^2 - 2 * x = 0 → x = 0 ∨ x = 2) :=
by
  sorry

end x_eq_zero_sufficient_not_necessary_l342_342111


namespace regular_polygon_sides_l342_342936

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (k : ℕ), (k : ℕ) * 18 = 360) : n = 20 :=
by
  -- Proof body here
  sorry

end regular_polygon_sides_l342_342936


namespace sum_of_digits_7_pow_11_l342_342465

/-
  Let's define the problem with the given conditions and expected answer.
  We need to prove that the sum of the tens digit and the ones digit of 7^11 is 7.
-/

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_7_pow_11 : tens_digit (7 ^ 11) + ones_digit (7 ^ 11) = 7 :=
sorry

end sum_of_digits_7_pow_11_l342_342465


namespace circle_radius_tangency_condition_l342_342560

noncomputable def parabola (r : ℝ) : ℝ → ℝ := λ x, x^2 + r

theorem circle_radius_tangency_condition :
  ∃ r : ℝ, (∀ x : ℝ, parabola r x = x → x^2 + r = x ∧ 1 - 4 * r = 0) ∧ r = 1 / 4 :=
by sorry

end circle_radius_tangency_condition_l342_342560


namespace choose_socks_with_blue_l342_342085

-- Define the set of socks
structure Socks :=
  (colors : Finset String)
  (size : Nat)
  (has_blue : Bool)

-- Theorem statement
theorem choose_socks_with_blue : 
  ∀ (sock_set : Socks), 
    sock_set.colors = { "blue", "brown", "black", "red", "purple" } → 
    sock_set.size = 5 → 
    sock_set.has_blue → 
    (Finset.card (Finset.filter (λ c, c ∈ { "blue" }) (Finset.powersetLen 3 sock_set.colors))) = 6 :=
by
  sorry

end choose_socks_with_blue_l342_342085


namespace find_coordinates_l342_342198

namespace CoordinateSystem

def Point := ℝ × ℝ × ℝ

def A : Point := (2, 1, 1)
def B : Point := (1, -3, 2)

def on_z_axis (M : Point) : Prop := M.fst = 0 ∧ M.snd.fst = 0

def distance (P Q : Point) : ℝ :=
  Math.sqrt ((P.fst - Q.fst)^2 + (P.snd.fst - Q.snd.fst)^2 + (P.snd.snd - Q.snd.snd)^2)

theorem find_coordinates (M : Point)
  (h1 : on_z_axis M)
  (h2 : distance M A = distance M B) :
  M = (0, 0, 4) :=
sorry

end CoordinateSystem

end find_coordinates_l342_342198


namespace isosceles_triangle_perimeter_l342_342682

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 3) (h2 : b = 7) : 
∃ (c : ℕ), 
  (c = 7 ∧ a = 3 ∧ b = 7 ∧ a + b + c = 17) ∨ 
  (c = 3 ∧ a = 7 ∧ b = 7 ∧ a + b + c = 17) :=
  sorry

end isosceles_triangle_perimeter_l342_342682


namespace boat_upstream_distance_is_75_l342_342491

noncomputable def boat_upstream_distance
  (D_downstream : ℕ)  -- downstream distance in km
  (T_downstream : ℕ)  -- downstream time in hours
  (T_upstream : ℕ)    -- upstream time in hours
  (V_s : ℕ)           -- speed of the stream in km/h
  : ℕ := 
  let V_downstream := D_downstream / T_downstream in
  let V_b := V_downstream - V_s in
  let V_upstream := V_b - V_s in
  V_upstream * T_upstream

theorem boat_upstream_distance_is_75 :
  boat_upstream_distance 130 10 15 4 = 75 :=
  by 
    -- Conditions are directly used in the definition 
    -- hence the proof will validate the correct distance
    sorry

end boat_upstream_distance_is_75_l342_342491


namespace gcd_153_119_l342_342430

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  have h1 : 153 = 119 * 1 + 34 := by rfl
  have h2 : 119 = 34 * 3 + 17 := by rfl
  have h3 : 34 = 17 * 2 := by rfl
  sorry

end gcd_153_119_l342_342430


namespace value_of_r_l342_342780

theorem value_of_r (n : ℕ) (h : n = 3) : (let s := 2^n - 1 in 2^s + s) = 135 :=
by
  sorry

end value_of_r_l342_342780


namespace volume_of_solid_l342_342547

noncomputable def volume_of_solid_of_revolution :
  ℝ := π * (∫ y in (0:ℝ)..1, (y ^ (2/3) - y)) 

theorem volume_of_solid :
  volume_of_solid_of_revolution = π / 10 :=
by
  -- Proof steps would go here
  sorry

end volume_of_solid_l342_342547


namespace map_length_represents_distance_l342_342325

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l342_342325


namespace tan_angle_point_l342_342626

theorem tan_angle_point (α m : ℝ) (hα : α = 7 * Real.pi / 3) (h_point: (Real.sqrt m, Real.cbrt m) = point_on_terminal_side α) :
  m = 1 / 27 := 
sorry

end tan_angle_point_l342_342626


namespace second_train_speed_l342_342427

theorem second_train_speed :
  ∃ v : ℝ, (10 * v - 10 * 10 = 250) ∧ v = 35 := 
by {
  use 35,
  split,
  sorry,
  refl,
}

end second_train_speed_l342_342427


namespace smallest_12_digit_divisible_by_36_with_all_digits_l342_342080

/-- We want to prove that the smallest 12-digit natural number that is divisible by 36 
    and contains each digit from 0 to 9 at least once is 100023457896. -/
theorem smallest_12_digit_divisible_by_36_with_all_digits :
  ∃ n : ℕ, n = 100023457896 ∧ 
    (nat.digits 10 n).length = 12 ∧ 
    (∀ d ∈ (finset.range 10).val, d ∈ (nat.digits 10 n).val) ∧ 
    n % 36 = 0 :=
begin
  sorry
end

end smallest_12_digit_divisible_by_36_with_all_digits_l342_342080


namespace math_problem_l342_342049

noncomputable def cube_root_8 : ℝ := real.cbrt 8
noncomputable def cube_root_27 : ℝ := real.cbrt 27
noncomputable def sqrt_2 : ℝ := real.sqrt 2

theorem math_problem:
    2 * (cube_root_8 - sqrt_2) - (cube_root_27 - sqrt_2) = 1 - sqrt_2 :=
by {
  have h1 : cube_root_8 = 2 := by sorry,
  have h2 : cube_root_27 = 3 := by sorry,
  rw [h1, h2],
  ring,
}

end math_problem_l342_342049


namespace area_of_figure_l342_342193
   
   -- Definitions of the conditions
   def side_length : ℝ := 2
   def angle_FAB : ℝ := 60
   def angle_BCD : ℝ := 60
   def parallel_AF_CD := true
   def parallel_AB_EF := true
   def parallel_BC_ED := true

   -- The statement to be proven
   theorem area_of_figure : 
     parallel_AF_CD ∧ parallel_AB_EF ∧ parallel_BC_ED ∧
     angle_FAB = 60 ∧ angle_BCD = 60 ∧
     ∀ (s : ℝ), s = side_length → (4 * (√3 / 4) * s^2) = 4 * √3 :=
   by
     intro,
     exact sorry
   
end area_of_figure_l342_342193


namespace PA_over_PB_squared_l342_342761

-- Definitions for the given conditions
def Circle := { center : ℝ × ℝ, radius : ℝ }

def Γ₁ : Circle := { center := (0, 0), radius := 1 }
def Γ₂ : Circle := { center := (3, 0), radius := 2 }
def Γ₃ : Circle := { center := (7, 0), radius := 3 }
def Ω : Circle := { center := (4, 0), radius := 5 }  -- Arbitrarily defined to follow the given collinearity condition, assume radius

def P := (5, 0)  -- Tangency point, assume collinearity as direct line between centers

def A := (3, 0)  -- Center of Γ₂
def B := (7, 0)  -- Center of Γ₃

-- PA and PB distances
noncomputable def PA : ℝ := (P.1 - A.1)
noncomputable def PB : ℝ := (P.1 - B.1)

theorem PA_over_PB_squared :
  (PA^2 / PB^2) = 8 / 15 := by
sorry

end PA_over_PB_squared_l342_342761


namespace find_angle_for_given_conditions_area_range_triangle_l342_342110

theorem find_angle_for_given_conditions (a b c : ℝ) 
  (h_a : a = 2)
  (h_eq : (b + 2) * (Real.sin (2 * Real.pi / 3) - Real.sin B) = c * (Real.sin B + Real.sin C)) :
  angle_A = 2 * Real.pi / 3 :=
sorry

theorem area_range_triangle (a b c A : ℝ) 
  (h_A : A = 2 * Real.pi / 3)
  (h_a : a = 2)
  (h_sines : ∀ (B C : ℝ), (a / Real.sin A) = (b / Real.sin B) ∧ (a / Real.sin A) = (c / Real.sin C))
  (h_cos : Real.cos A = -1 / 2)
  (B_range : ∀ (B : ℝ), 0 < B ∧ B < Real.pi / 3) :
  S ∈ Set.Ioo 0 (sqrt 3 / 3) :=
sorry

end find_angle_for_given_conditions_area_range_triangle_l342_342110


namespace smaller_circle_radius_eq_l342_342195

-- Definitions based on the problem conditions
def largest_circle_radius : ℝ := 10
def largest_circle_diameter : ℝ := 2 * largest_circle_radius
def num_smaller_circles : ℝ := 6

-- Theorem stating the radius of one of the six smaller circles
theorem smaller_circle_radius_eq :
  ∃ r : ℝ, (num_smaller_circles * (2 * r) = largest_circle_diameter) ∧ (r = 5 / 3) :=
begin
  existsi (5 / 3),
  split,
  { rw [largest_circle_diameter], norm_num },
  { refl }
end

end smaller_circle_radius_eq_l342_342195


namespace num_values_P_eq_3_l342_342087

noncomputable def P (x : ℝ) : ℂ := 1 + complex.cos x + complex.sin x * complex.I 
  - complex.cos (2 * x) - complex.sin (2 * x) * complex.I 
  + complex.cos (3 * x) + complex.sin (3 * x) * complex.I 
  - complex.cos (4 * x) - complex.sin (4 * x) * complex.I

theorem num_values_P_eq_3 : 
  (set.count {x ∈ set.Ico 0 (2 * real.pi) | P x = 0} = 3) :=
sorry

end num_values_P_eq_3_l342_342087


namespace find_m_and_circle_equation_l342_342603

-- Assumption and definitions from conditions in (a)
def line (m : ℝ) : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), p.1 - p.2 + m = 0

def rotated_line (Q : ℝ × ℝ) (m : ℝ) : Prop := 
  let P : ℝ × ℝ := (-m, 0)
  in (Q.2 - 0) / (Q.1 + m) = -1

def circle_equation (center : ℝ × ℝ) (radius : ℝ) : ℝ × ℝ → Prop := 
  λ (p : ℝ × ℝ), (p.1 + center.1)^2 + (p.2 + center.2)^2 = radius^2

-- statement only
theorem find_m_and_circle_equation :
  ∃ m, line m (2, -3) ∧ rotated_line (2, -3) m ∧ 
  ∃ center, line 1 center ∧ 
  circle_equation (-3, -2) 5 (1, 1) ∧ 
  circle_equation (-3, -2) 5 (2, -2) :=
by sorry

end find_m_and_circle_equation_l342_342603


namespace find_m_l342_342118

variable (m : ℝ)
axiom slope_condition (h : x + m * y - 3 = 0) : -1 / m = Real.tan (Real.pi / 6)

theorem find_m (h : slope_condition m) : m = -Real.sqrt 3 :=
sorry

end find_m_l342_342118


namespace sin_cos_fraction_l342_342097

theorem sin_cos_fraction (α : ℝ) (h1 : Real.sin α - Real.cos α = 1 / 5) (h2 : α ∈ Set.Ioo (-Real.pi / 2) (Real.pi / 2)) :
    Real.sin α * Real.cos α / (Real.sin α + Real.cos α) = 12 / 35 :=
by
  sorry

end sin_cos_fraction_l342_342097


namespace parabola_focus_line_slope_intersect_l342_342145

theorem parabola_focus (p : ℝ) (hp : 0 < p) 
  (focus : (1/2 : ℝ) = p/2) : p = 1 :=
by sorry

theorem line_slope_intersect (t : ℝ)
  (intersects_parabola : ∃ A B : ℝ × ℝ, A ≠ (0, 0) ∧ B ≠ (0, 0) ∧
    A ≠ B ∧ A.2 = 2 * A.1 + t ∧ B.2 = 2 * B.1 + t ∧ 
    A.2^2 = 2 * p * A.1 ∧ B.2^2 = 2 * p * B.1 ∧ 
    A.1 * B.1 + A.2 * B.2 = 0) : 
  t = -4 :=
by sorry

end parabola_focus_line_slope_intersect_l342_342145


namespace max_radius_in_cone_l342_342177

theorem max_radius_in_cone :
  ∀ (PO PF : ℝ) (hPO : PO = 4) (hPF : PF = 8) (r : ℝ),
    r = (12 / (5 + 2 * real.sqrt 3)) :=
by
  intros PO PF hPO hPF r
  sorry

end max_radius_in_cone_l342_342177


namespace two_hundred_twenty_second_digit_l342_342862

theorem two_hundred_twenty_second_digit:
  let frac := (25 : ℚ) / 350 in
  let simplified_frac := 1 / 14 in
  frac = simplified_frac → 
  ∀ (n : ℕ), n = 222 → 
  let cycle := "071428".to_list in
  (cycle!!(n % 6) = '8') :=
by
  sorry

end two_hundred_twenty_second_digit_l342_342862


namespace least_five_digit_congruent_to_6_mod_17_l342_342441

theorem least_five_digit_congruent_to_6_mod_17 :
  ∃ (x : ℕ), 10000 ≤ x ∧ x ≤ 99999 ∧ x % 17 = 6 ∧
  ∀ (y : ℕ), 10000 ≤ y ∧ y ≤ 99999 ∧ y % 17 = 6 → x ≤ y := 
sorry

end least_five_digit_congruent_to_6_mod_17_l342_342441


namespace sum_of_digits_7_pow_11_l342_342463

/-
  Let's define the problem with the given conditions and expected answer.
  We need to prove that the sum of the tens digit and the ones digit of 7^11 is 7.
-/

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_7_pow_11 : tens_digit (7 ^ 11) + ones_digit (7 ^ 11) = 7 :=
sorry

end sum_of_digits_7_pow_11_l342_342463


namespace wendy_miles_second_day_l342_342361

theorem wendy_miles_second_day (m1 m3 total : ℕ) (h1 : m1 = 125) (h3 : m3 = 145) (htotal : total = 493) :
  ∃ m2 : ℕ, m2 = total - (m1 + m3) ∧ m2 = 223 :=
by
  use (total - (m1 + m3))
  split
  · sorry
  · sorry

end wendy_miles_second_day_l342_342361


namespace find_point_M_l342_342196

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def distance (P Q : Point3D) : ℝ :=
real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)

theorem find_point_M :
  ∃ M : Point3D, M.x = 0 ∧ M.y = 0 ∧ distance M ⟨2, 1, 1⟩ = distance M ⟨1, -3, 2⟩ ∧ M.z = 4 :=
sorry

end find_point_M_l342_342196


namespace Petya_wins_l342_342849

-- Definition of the total number of edges in a complete graph with 2000 vertices.
def number_of_nodes : ℕ := 2000
def total_edges : ℕ := number_of_nodes * (number_of_nodes - 1) / 2

-- Rule for cutting wires: Varia cuts 1 wire, Petya cuts 2 or 3 wires.
def Varia_cuts (remaining_edges : ℕ) : ℕ := remaining_edges - 1
def Petya_cuts (remaining_edges : ℕ) (cuts : ℕ) : ℕ := if cuts = 2 ∨ cuts = 3 then remaining_edges - cuts else remaining_edges

-- Winning condition: The player who cuts the last wire loses.
def game_end (remaining_edges : ℕ) : Prop := remaining_edges = 0

-- Proof that Petya wins the game.
theorem Petya_wins : ∃ (strategy : ℕ → ℕ × (ℕ → ℕ)), 
                    (∀ (remaining_edges : ℕ), strategy remaining_edges = (cut_edges : ℕ, move : ℕ → ℕ) → 
                    (remaining_edges % 4 = 0 → cut_edges = Petya_cuts remaining_edges move) ∨ 
                    (game_end remaining_edges ∧ strategy remaining_edges = (remaining_edges, Varia_cuts remaining_edges) → false)) → 
                    (total_edges % 4 = 0) → 
                    strategy(total_edges) = (remaining_edges, move) → game_end remaining_edges → 
                    move = Varia_cuts remaining_edges → false ∧ Petya wins the game :=
begin
    sorry   -- Proof omitted
end

end Petya_wins_l342_342849


namespace cos_alpha_minus_pi_over_3_l342_342094

-- Given condition
def alpha : ℝ := sorry
axiom sin_condition : Real.sin (alpha + π / 6) = 4 / 5

-- Mathematical proof problem
theorem cos_alpha_minus_pi_over_3 : Real.cos (alpha - π / 3) = 4 / 5 :=
by
    -- We state the theorem here with the main goal stated
    sorry

end cos_alpha_minus_pi_over_3_l342_342094


namespace base6_addition_correct_l342_342019

-- Define a function to convert a base 6 digit to its base 10 equivalent
def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 4
  | 5 => 5
  | d => 0 -- for illegal digits, fallback to 0

-- Define a function to convert a number in base 6 to base 10
def convert_base6_to_base10 (n : Nat) : Nat :=
  let units := base6_to_base10 (n % 10)
  let tens := base6_to_base10 ((n / 10) % 10)
  let hundreds := base6_to_base10 ((n / 100) % 10)
  units + 6 * tens + 6 * 6 * hundreds

-- Define a function to convert a base 10 number to a base 6 number
def base10_to_base6 (n : Nat) : Nat :=
  (n % 6) + 10 * ((n / 6) % 6) + 100 * ((n / (6 * 6)) % 6)

theorem base6_addition_correct : base10_to_base6 (convert_base6_to_base10 35 + convert_base6_to_base10 25) = 104 := by
  sorry

end base6_addition_correct_l342_342019


namespace puppies_sold_l342_342513

theorem puppies_sold (total_puppies sold_puppies puppies_per_cage total_cages : ℕ)
  (h1 : total_puppies = 102)
  (h2 : puppies_per_cage = 9)
  (h3 : total_cages = 9)
  (h4 : total_puppies - sold_puppies = puppies_per_cage * total_cages) :
  sold_puppies = 21 :=
by {
  -- Proof details would go here
  sorry
}

end puppies_sold_l342_342513


namespace indefinite_integral_l342_342041

noncomputable def integrand : ℝ → ℝ := λ x, (x^3 - 6 * x^2 + 14 * x - 6) / ((x + 1) * (x - 2)^3)

theorem indefinite_integral :
  ∫ integrand x dx = λ x, ln |x + 1| - 1 / (x - 2)^2 + C :=
sorry

end indefinite_integral_l342_342041


namespace map_length_representation_l342_342283

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l342_342283


namespace three_digit_number_satisfying_condition_l342_342965

theorem three_digit_number_satisfying_condition :
  ∃ (a b c : ℕ), a = 1 ∧ 100 * a + 10 * b + c = 243 ∧
  let abc := 100 * a + 10 * b + c,
      bca := 100 * b + 10 * c + a,
      cab := 100 * c + 10 * a + b in
  ∃ n : ℕ, (abc * bca * cab) = n^2 :=
by {
  sorry
}

end three_digit_number_satisfying_condition_l342_342965


namespace intersection_length_of_sphere_and_tetrahedron_l342_342672

theorem intersection_length_of_sphere_and_tetrahedron (O : Point) (R : ℝ) (r : ℝ) (edge_length : ℝ) : 
  radius = √2 ∧ edge_length = 2*√6 ∧ R = √3 →
  total_length = 8*√2*π :=
by
  sorry

end intersection_length_of_sphere_and_tetrahedron_l342_342672


namespace astronomical_year_length_l342_342482

-- Define the conditions in Lean
def is_leap_year (n : ℕ) : Prop :=
  (n % 4 = 0 ∧ n % 100 ≠ 0) ∨ (n % 400 = 0)

def days_in_year (n : ℕ) : ℕ :=
  if is_leap_year n then 366 else 365

-- The theorem statement aiming to prove the mean length of an astronomical year.
theorem astronomical_year_length : 
  (∑ n in (0 : ℕ) to 400, days_in_year n) / 400 = 365.2425 := by
sorry

end astronomical_year_length_l342_342482


namespace black_spools_l342_342746

-- Define the given conditions
def spools_per_beret : ℕ := 3
def red_spools : ℕ := 12
def blue_spools : ℕ := 6
def berets_made : ℕ := 11

-- Define the statement to be proved using the defined conditions
theorem black_spools (spools_per_beret red_spools blue_spools berets_made : ℕ) : (spools_per_beret * berets_made) - (red_spools + blue_spools) = 15 :=
by sorry

end black_spools_l342_342746


namespace course_choice_related_to_gender_l342_342415

def contingency_table (a b c d n : ℕ) : Prop :=
  n = a + b + c + d ∧
  a + b = 50 ∧
  c + d = 50 ∧
  a + c = 70 ∧
  b + d = 30

def chi_square_test (a b c d n : ℕ) : ℕ := 
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem course_choice_related_to_gender (a b c d n : ℕ) :
  contingency_table 40 10 30 20 100 →
  chi_square_test 40 10 30 20 100 > 3.841 :=
by
  intros h_table
  sorry

end course_choice_related_to_gender_l342_342415


namespace find_vector_equation_and_rotated_plane_l342_342210

noncomputable def line_intersection_planes : Set (ℝ × ℝ × ℝ) :=
  {l | ∃ λ : ℝ, l = (0, 0, 0) + λ • (-1, 1, -1)}

def plane1 : Set (ℝ × ℝ × ℝ) := {p | p.1 + p.2 = 0}

def plane2 : Set (ℝ × ℝ × ℝ) := {p | p.2 + p.3 = 0}

def rotated_plane (k : ℝ) : Set (ℝ × ℝ × ℝ) := 
  {p | p.1 + (2 + k) * p.2 + (1 + k) * p.3 = 0}

theorem find_vector_equation_and_rotated_plane:
  line_intersection_planes = {l | ∃ λ : ℝ, l = (0, 0, 0) + λ • (-1, 1, -1)} ∧
  (rotated_plane (real.sqrt 3) = {p | p.1 + (2 + real.sqrt 3) * p.2 + (1 + real.sqrt 3) * p.3 = 0} ∨ 
   rotated_plane (-real.sqrt 3) = {p | p.1 + (2 - real.sqrt 3) * p.2 + (1 - real.sqrt 3) * p.3 = 0}) :=
by
  sorry

end find_vector_equation_and_rotated_plane_l342_342210


namespace regular_polygon_sides_l342_342948

theorem regular_polygon_sides (θ : ℝ) (n : ℕ) (h1 : θ = 18) (h2 : 360 = θ * n) : n = 20 :=
by {
  sorry
}

end regular_polygon_sides_l342_342948


namespace cockpit_reinforcement_necessity_l342_342064

-- Definitions based on the conditions
def uniform_bullet_distribution : Prop := 
  ∀ (part : String), part ∈ ["wing", "fuselage", "tail", "cockpit"] → P(part) = 0.25 -- simplified assumption

def observed_plane_damage (part : String) : Prop :=
  part ≠ "cockpit" → ∃ (damage : Bool), damage = True

-- The question translated to a proof problem
theorem cockpit_reinforcement_necessity 
  (h1 : uniform_bullet_distribution)
  (h2 : ∀ (part : String), observed_plane_damage part) :
  ∃ (fragile : Bool), fragile = True :=
by
  sorry -- Proof omitted

end cockpit_reinforcement_necessity_l342_342064


namespace mode_of_scores_l342_342373

def stem_and_leaf_scores : List ℕ := 
  [45, 45, 52, 61, 67, 67, 70, 73, 74, 74, 74, 74, 80, 85, 85, 86, 90, 90]

theorem mode_of_scores (scores : List ℕ) : ∃ mode, 
  mode ∈ scores ∧ (∀ x ∈ scores, count scores mode ≥ count scores x) :=
by
  use 74
  -- Here, we will prove that 74 appears most frequently in the list
  sorry

end mode_of_scores_l342_342373


namespace smallest_seven_binary_digits_is_64_l342_342448

-- Definitions based on conditions
def has_seven_binary_digits (n : ℕ) : Prop :=
  64 ≤ n ∧ n < 128

-- Theorem statement proving the question == answer
theorem smallest_seven_binary_digits_is_64 :
  ∃ n : ℕ, has_seven_binary_digits(n) ∧ ∀ m : ℕ, has_seven_binary_digits(m) → n ≤ m :=
sorry

end smallest_seven_binary_digits_is_64_l342_342448


namespace num_ways_select_with_second_largest_seven_l342_342579

theorem num_ways_select_with_second_largest_seven :
  ∃ (S : Finset ℕ), S.card = 4 ∧ 7 ∈ S ∧ (∀ t ∈ S, t ≤ 10) ∧ (S.erase 7).max' (by simp) = 7 ↔ (S.erase 7).card = 3 ∧ S.count 7 = 1 ∧ S.count > 10 :
    45 :=
sorry

end num_ways_select_with_second_largest_seven_l342_342579


namespace map_representation_l342_342286

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l342_342286


namespace solve_for_q_l342_342105

noncomputable def is_arithmetic_SUM_seq (a₁ q: ℝ) (n: ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem solve_for_q (a₁ q S3 S6 S9: ℝ) (hq: q ≠ 1) (hS3: S3 = is_arithmetic_SUM_seq a₁ q 3) 
(hS6: S6 = is_arithmetic_SUM_seq a₁ q 6) (hS9: S9 = is_arithmetic_SUM_seq a₁ q 9) 
(h_arith: 2 * S9 = S3 + S6) : q^3 = 3 / 2 :=
sorry

end solve_for_q_l342_342105


namespace solve_gcd_problem_l342_342432

def gcd_problem : Prop :=
  gcd 153 119 = 17

theorem solve_gcd_problem : gcd_problem :=
  by
    sorry

end solve_gcd_problem_l342_342432


namespace jane_percentage_decrease_l342_342751

theorem jane_percentage_decrease
  (B H : ℝ) -- Number of bears Jane makes per week and hours she works per week
  (H' : ℝ) -- Number of hours Jane works per week with assistant
  (h1 : H' = 0.9 * H) -- New working hours when Jane works with an assistant
  (h2 : H ≠ 0) -- Ensure H is not zero to avoid division by zero
  : ((H - H') / H) * 100 = 10 := 
by calc
  ((H - H') / H) * 100
      = ((H - 0.9 * H) / H) * 100 : by rw [h1]
  ... = (0.1 * H / H) * 100 : by simp
  ... = 0.1 * 100 : by rw [div_self h2]
  ... = 10 : by norm_num

end jane_percentage_decrease_l342_342751


namespace weight_calculation_l342_342383

def weight_of_replaced_person : Type :=
  { weight : ℕ // let original_average_increase := 4 * 8 in
                 let new_person_weight := 97 in
                 new_person_weight = weight + original_average_increase }

theorem weight_calculation (w : weight_of_replaced_person) : w.weight = 65 :=
  sorry

end weight_calculation_l342_342383


namespace alyssa_cookie_count_l342_342974

variable (Aiyanna_cookies Alyssa_cookies : ℕ)
variable (h1 : Aiyanna_cookies = 140)
variable (h2 : Aiyanna_cookies = Alyssa_cookies + 11)

theorem alyssa_cookie_count : Alyssa_cookies = 129 := by
  -- We can use the given conditions to prove the theorem
  sorry

end alyssa_cookie_count_l342_342974


namespace map_scale_representation_l342_342331

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l342_342331


namespace evaluate_f_at_half_l342_342631

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + a * real.logb 3 x

theorem evaluate_f_at_half (a : ℝ) (h : f 2 a = 6) :
  f (1 / 2) a = 17 / 8 :=
sorry

end evaluate_f_at_half_l342_342631


namespace prove_g_ggg_2_l342_342243

def g (x : ℝ) : ℝ :=
  if x >= 3 then x^3 else x^2 + 1

theorem prove_g_ggg_2 : g (g (g (2))) = 1953125 := by
  sorry

end prove_g_ggg_2_l342_342243


namespace minimum_additional_marbles_l342_342417

-- Definitions corresponding to the conditions
def friends := 12
def initial_marbles := 40

-- Sum of the first n natural numbers definition
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Prove the necessary number of additional marbles
theorem minimum_additional_marbles (h1 : friends = 12) (h2 : initial_marbles = 40) : 
  ∃ additional_marbles, additional_marbles = sum_first_n friends - initial_marbles := by
  sorry

end minimum_additional_marbles_l342_342417


namespace black_car_overtakes_red_car_in_one_hour_l342_342890

def red_car_speed : ℕ := 40
def black_car_speed : ℕ := 50
def initial_gap : ℕ := 10

theorem black_car_overtakes_red_car_in_one_hour (h_red_car_speed : red_car_speed = 40)
                                               (h_black_car_speed : black_car_speed = 50)
                                               (h_initial_gap : initial_gap = 10) :
  initial_gap / (black_car_speed - red_car_speed) = 1 :=
by
  sorry

end black_car_overtakes_red_car_in_one_hour_l342_342890


namespace right_triangle_hypotenuse_l342_342837

theorem right_triangle_hypotenuse (a b : ℝ)
  (h1 : b^2 + (3 * a / 2)^2 = 39)
  (h2 : a^2 + (3 * b / 2)^2 = 36) :
  sqrt (9 * (a^2 + b^2)) = 3 * sqrt 23 :=
by
  sorry

end right_triangle_hypotenuse_l342_342837


namespace arccos_sin3_eq_l342_342055

theorem arccos_sin3_eq :
  ∀ (x : ℝ), x = arccos (sin 3) ↔ x = 3 - (π / 2) :=
by
  sorry

end arccos_sin3_eq_l342_342055


namespace map_representation_l342_342288

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l342_342288


namespace EP_over_PF_eq_3_over_4_m_plus_n_eq_7_l342_342851

open_locale real

noncomputable def equilateral_triangle_coordinates :
  ℝ × ℝ × ℝ × ℝ × ℝ × ℝ :=
  (0, 0, 12, 0, 6, 6 * real.sqrt 3)

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def point_F (B A : ℝ × ℝ) : ℝ × ℝ :=
  ((2 * B.1 + A.1) / 3, (2 * B.2 + A.2) / 3)

def line_eq (A B : ℝ × ℝ) : ℝ → ℝ :=
  λ x, ((B.2 - A.2) / (B.1 - A.1)) * (x - A.1) + A.2

noncomputable def point_P (A D E F : ℝ × ℝ) : ℝ × ℝ :=
  let (xA, yA, xD, yD, xE, yE, xF, yF) :=
    (A.1, A.2, D.1, D.2, E.1, E.2, F.1, F.2) in
  let α := ((yE - yF) / (xE - xF) - (yA - yD) / (xA - xD)) ⁻¹ * 
          ((yA - yD) / (xA - xD) * xA - (yE - yF) / (xE - xF) * xE + yE - yA) in
  (α, line_eq A D α)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

noncomputable def ratio_EP_PF (A B C D E F P : ℝ × ℝ) : ℚ :=
  (distance E P) / (distance P F)

theorem EP_over_PF_eq_3_over_4 :
  let (A, B, C) := equilateral_triangle_coordinates in
  let D := midpoint B C in
  let E := midpoint A C in
  let F := point_F B A in
  let P := point_P A D E F in
  ratio_EP_PF A B C D E F P = 3 / 4 :=
begin
  sorry
end

theorem m_plus_n_eq_7 :
  let (A, B, C) := equilateral_triangle_coordinates in
  let D := midpoint B C in
  let E := midpoint A C in
  let F := point_F B A in
  let P := point_P A D E F in
  let ratio := ratio_EP_PF A B C D E F P in
  let m := ratio.numerator in
  let n := ratio.denominator in
  m + n = 7 :=
begin
  sorry
end

end EP_over_PF_eq_3_over_4_m_plus_n_eq_7_l342_342851


namespace sin_double_angle_solution_l342_342724

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_solution_l342_342724


namespace equal_functions_l342_342975

def f (x : ℝ) : ℝ := abs x

def g (x : ℝ) : ℝ := if x ≥ 0 then x else -x

theorem equal_functions : ∀ x : ℝ, f x = g x := 
by
  intros x
  simp only [f, g]
  split_ifs
  { reflexivity }
  { rw abs_of_neg h }
  { reflexivity }
  sorry

end equal_functions_l342_342975


namespace units_digit_fraction_l342_342873

theorem units_digit_fraction :
  let n := 30 * 31 * 32 * 33 * 34 * 35
  let d := 1500
  let fraction := n / d
  (fraction % 10) = 4 := by
  let n := 30 * 31 * 32 * 33 * 34 * 35
  let d := 1500
  let fraction := n / d
  have h₁ : 30 = 2 * 3 * 5 := by rfl
  have h₂ : 31 = 31 := by rfl
  have h₃ : 32 = 2 ^ 5 := by rfl
  have h₄ : 33 = 3 * 11 := by rfl
  have h₅ : 34 = 2 * 17 := by rfl
  have h₆ : 35 = 5 * 7 := by rfl
  have h₇ : 1500 = 2 ^ 2 * 3 * 5 ^ 3 := by rfl
  have num_factorization : n = 2 ^ 7 * 3 ^ 2 * 5 ^ 2 * 31 * 11 * 17 * 7 := by
    rw [← h₁, ← h₂, ← h₃, ← h₄, ← h₅, ← h₆]
    ring
  have den_factorization : d = 2 ^ 2 * 3 * 5 ^ 3 := by rw h₇
  have simplified_fraction : fraction = 2 ^ 5 * 3 * 31 * 11 * 17 * 7 := by
    rw [num_factorization, den_factorization]
    field_simp
    ring
  have : (2 ^ 5 * 3 * 31 * 11 * 17 * 7 % 10) = 4 := by sorry
  exact this

end units_digit_fraction_l342_342873


namespace map_length_representation_l342_342274

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l342_342274


namespace angle_calculation_l342_342621

noncomputable def angle_between_vectors (a b : ℝ) : ℝ := 
-- Assume necessary vector operations and angle calculation functions are predefined

variables (a b : ℝ^3)
variables (angle_ab : ℝ) (norm_a : ℝ) (norm_b : ℝ)

-- Given Conditions
axiom angle_ab_def : angle_ab = π / 3
axiom norm_a_def : norm_a = 1
axiom norm_b_def : norm_b = 0.5

-- Target Angle Calculation
theorem angle_calculation : angle_between_vectors (a - 2 • b) a = π / 3 := 
sorry

end angle_calculation_l342_342621


namespace parallel_lines_sufficient_not_necessary_l342_342607

-- Definitions of parallelism in the context of geometric lines and planes
def line := Set Point
def plane := Set Point

-- Line m, line n, and plane α are given
variables (m n : line) (α : plane)

-- Conditions given in the problem
-- condition 1: line n is a subset of plane α
def line_in_plane (line : line) (pl : plane) : Prop :=
  ∀ p, p ∈ line → p ∈ pl

-- condition 2: line m is not a subset of plane α
def not_line_in_plane (line : line) (pl : plane) : Prop :=
  ¬ (line_in_plane line pl)

-- Definition of parallelism
def parallel (l1 l2 : line) : Prop := 
  ∃ v, (∀ p1 p2, p1 ∈ l1 → p2 ∈ l2 → v ≠ 0 ∧ (p1 - p2))  -- Simplified for representation

def parallel_to_plane (line : line) (pl : plane) : Prop :=
  ∃ v, (∀ p1 p2, p1 ∈ line → p2 ∈ pl → v ≠ 0 ∧ (p1 - p2))  -- Simplified for representation

-- The proof problem statement in Lean 4
theorem parallel_lines_sufficient_not_necessary (h1 : line_in_plane n α) (h2 : not_line_in_plane m α) :
  (parallel m n → parallel_to_plane m α) ∧ ¬ (parallel_to_plane m α → parallel m n) :=
by
  sorry

end parallel_lines_sufficient_not_necessary_l342_342607


namespace proposition_2_counterexample_proposition_1_counterexample_proposition_3_l342_342220

variables (a b c : ℝ)

-- Proposition ②: If c > 1 and 0 < b < 2, then a^2 + ab + c > 0.
theorem proposition_2 (h1 : c > 1) (h2 : 0 < b ∧ b < 2) : a^2 + ab + c > 0 :=
sorry

-- Counterexample for proposition ①: Refuting If a^2 + ab + c > 0 and c > 1, then 0 < b < 2.
theorem counterexample_proposition_1 : ∃ (a b c : ℝ), (a^2 + ab + c > 0) ∧ (c > 1) ∧ ¬ (0 < b ∧ b < 2) :=
by
  use [0, 4, 5]
  split; {norm_num, linarith}
  sorry

-- Counterexample for proposition ③: Refuting If 0 < b < 2 and a^2 + ab + c > 0, then c > 1.
theorem counterexample_proposition_3 : ∃ (a b c : ℝ), (0 < b ∧ b < 2) ∧ (a^2 + ab + c > 0) ∧ ¬ (c > 1) :=
by
  use [0, 1, 1/4]
  split; {norm_num, linarith}
  sorry

end proposition_2_counterexample_proposition_1_counterexample_proposition_3_l342_342220


namespace map_length_representation_l342_342308

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l342_342308


namespace sin_double_angle_l342_342716

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l342_342716


namespace product_fraction_eq_351_l342_342053

open BigOperators

theorem product_fraction_eq_351 :
  ∏ n in Finset.range 25 |>.map (λ n, n + 1), (n + 2) / n = 351 :=
by
  sorry

end product_fraction_eq_351_l342_342053


namespace map_length_representation_l342_342277

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l342_342277


namespace arthur_bought_hamburgers_on_first_day_l342_342980

-- Define the constants and parameters
def D : ℕ := 1
def H : ℕ := 2
def total_cost_day1 : ℕ := 10
def total_cost_day2 : ℕ := 7

-- Define the equation representing the transactions
def equation_day1 (h : ℕ) := H * h + 4 * D = total_cost_day1
def equation_day2 := 2 * H + 3 * D = total_cost_day2

-- The theorem we need to prove: the number of hamburgers h bought on the first day is 3
theorem arthur_bought_hamburgers_on_first_day (h : ℕ) (hd1 : equation_day1 h) (hd2 : equation_day2) : h = 3 := 
by 
  sorry

end arthur_bought_hamburgers_on_first_day_l342_342980


namespace number_of_distinct_m_values_l342_342230

theorem number_of_distinct_m_values (m : ℤ) :
  (∃ x1 x2 : ℤ, x1 * x2 = 36 ∧ x1 + x2 = m) →
  set.card {m | ∃ x1 x2 : ℤ, x1 * x2 = 36 ∧ x1 + x2 = m} = 10 :=
by
  sorry

end number_of_distinct_m_values_l342_342230


namespace luca_loss_years_l342_342540

variable (months_in_year : ℕ := 12)
variable (barbi_kg_per_month : ℚ := 1.5)
variable (luca_kg_per_year : ℚ := 9)
variable (luca_additional_kg : ℚ := 81)

theorem luca_loss_years (barbi_yearly_loss : ℚ :=
                          barbi_kg_per_month * months_in_year) :
  (81 + barbi_yearly_loss) / luca_kg_per_year = 11 := by
  let total_loss_by_luca := 81 + barbi_yearly_loss
  sorry

end luca_loss_years_l342_342540


namespace money_left_l342_342512

def remaining_money (S : ℝ) : ℝ :=
  S - (2 / 5 * S + 3 / 10 * S + 1 / 8 * S)

theorem money_left (S : ℝ) (h_food_conveyance : 3 / 10 * S + 1 / 8 * S = 3400) :
  S = 8000 → remaining_money S = 1400 :=
by
  intro hS
  rw [hS]
  unfold remaining_money
  norm_num

end money_left_l342_342512


namespace workEfficiencyRatioProof_is_2_1_l342_342473

noncomputable def workEfficiencyRatioProof : Prop :=
  ∃ (A B : ℝ), 
  (1 / B = 21) ∧ 
  (1 / (A + B) = 7) ∧
  (A / B = 2)

theorem workEfficiencyRatioProof_is_2_1 : workEfficiencyRatioProof :=
  sorry

end workEfficiencyRatioProof_is_2_1_l342_342473


namespace map_length_represents_distance_l342_342327

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l342_342327


namespace euler_polynomial_not_prime_consecutive_l342_342665

noncomputable def euler_polynomial (n : ℕ) : ℕ := n^2 + n + 41

theorem euler_polynomial_not_prime_consecutive :
  ∃ k, ∀ j : ℕ, 0 ≤ j ∧ j ≤ 39 → ¬ prime (euler_polynomial (k + j)) :=
by sorry

end euler_polynomial_not_prime_consecutive_l342_342665


namespace distinct_m_count_l342_342235

noncomputable def countDistinctMValues : Nat :=
  let pairs := [(1, 36), (2, 18), (3, 12), (4, 9), (6, 6), 
                (-1, -36), (-2, -18), (-3, -12), (-4, -9), (-6, -6)]
  let ms := pairs.map (λ p => p.1 + p.2)
  ms.eraseDups.length

theorem distinct_m_count :
  countDistinctMValues = 10 := sorry

end distinct_m_count_l342_342235


namespace length_AB_l342_342923

noncomputable def parabola_focus : ℝ × ℝ := (4, 0)

noncomputable def parabola_directrix : ℝ → Prop := λ x, x = -4

variables (x1 y1 x2 y2 : ℝ)

-- Definition of points A and B on the parabola
noncomputable def on_parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Given Condition
axiom sum_x1_x2 : x1 + x2 = 6

-- Axiom stating points A and B are on the parabola
axiom A_on_parabola : on_parabola x1 y1
axiom B_on_parabola : on_parabola x2 y2

-- Theorem to prove the length of |AB|
theorem length_AB (x1 y1 x2 y2 : ℝ) (h1 : on_parabola x1 y1) (h2 : on_parabola x2 y2) (h3 : x1 + x2 = 6) : |AB| = 14 :=
sorry

end length_AB_l342_342923


namespace map_scale_l342_342265

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l342_342265


namespace map_length_scale_l342_342304

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l342_342304


namespace inequality_proof_l342_342811

theorem inequality_proof (x y : ℝ) (h1 : x^2 ≥ y) (h2 : y^2 ≥ x) : 
  (x / (y^2 + 1) + y / (x^2 + 1) ≤ 1) :=
sorry

end inequality_proof_l342_342811


namespace regular_polygon_sides_l342_342941

-- Definitions based on conditions in the problem
def exterior_angle (n : ℕ) : ℝ := 360 / n

theorem regular_polygon_sides (n : ℕ) (h : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l342_342941


namespace map_scale_l342_342358

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l342_342358


namespace cosine_of_angle_C_l342_342693

-- Define the conditions 
structure RightTriangle :=
  (A B C : Type) 
  (angleA : ℝ)
  (AB BC : ℝ)
  (right_angle : angleA = 90)
  (AB_length : AB = 8)
  (BC_length : BC = 17)

-- Hypotenuse calculation and cosine extraction
noncomputable def calculate_cosine_of_C (T : RightTriangle) : ℝ :=
  let AC := real.sqrt (T.BC * T.BC - T.AB * T.AB) in AC / T.BC

-- Prove the theorem given the conditions
theorem cosine_of_angle_C (T : RightTriangle) : calculate_cosine_of_C T = 15 / 17 :=
by
  -- The proof would go here
  sorry

end cosine_of_angle_C_l342_342693


namespace isosceles_triangle_perimeter_l342_342681

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 3) (h2 : b = 7) : 
∃ (c : ℕ), 
  (c = 7 ∧ a = 3 ∧ b = 7 ∧ a + b + c = 17) ∨ 
  (c = 3 ∧ a = 7 ∧ b = 7 ∧ a + b + c = 17) :=
  sorry

end isosceles_triangle_perimeter_l342_342681


namespace suzie_found_two_worms_l342_342379

theorem suzie_found_two_worms
  (w1_length : ℝ) (h1 : w1_length = 0.8)
  (w2_length : ℝ) (h2 : w2_length = 0.1)
  (h3 : w1_length - w2_length = 0.7) :
  ∃ n : ℕ, n = 2 :=
by
  use 2
  sorry

end suzie_found_two_worms_l342_342379


namespace cos_theta_is_one_l342_342510

open Real

def direction_vector_1 := (2, 1)
def direction_vector_2 := (4, 2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2)

noncomputable def cos_theta : ℝ :=
  dot_product direction_vector_1 direction_vector_2 / (magnitude direction_vector_1 * magnitude direction_vector_2)

theorem cos_theta_is_one :
  cos_theta = 1 :=
sorry

end cos_theta_is_one_l342_342510


namespace regular_polygon_sides_l342_342942

-- Definitions based on conditions in the problem
def exterior_angle (n : ℕ) : ℝ := 360 / n

theorem regular_polygon_sides (n : ℕ) (h : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l342_342942


namespace smallest_number_of_tvs_l342_342793

def no_prohibited_digits (n : ℕ) : Prop :=
  ∀ d ∈ (to_string n).to_list, d ≠ '0' ∧ d ≠ '7' ∧ d ≠ '8' ∧ d ≠ '9'

theorem smallest_number_of_tvs (n : ℕ) : n = 56 ↔ 
  (∀ m < n, ¬ no_prohibited_digits (1994 * m)) ∧ no_prohibited_digits (1994 * n) :=
by
  sorry

end smallest_number_of_tvs_l342_342793


namespace problem_1_problem_2_l342_342908

-- (I)
theorem problem_1 (x : ℝ) (a : ℝ := x^2 + 1/2) (b : ℝ := 2 - x) (c : ℝ := x^2 - x + 1) : 
  a ≥ 1 ∨ b ≥ 1 ∨ c ≥ 1 := 
sorry

-- (II)
theorem problem_2 (a : ℝ) (h : a > 0) : 
  sqrt(a^2 + 1 / a^2) + 2 ≥ a + 1 / a + sqrt 2 := 
sorry

end problem_1_problem_2_l342_342908


namespace option_C_correct_l342_342875

theorem option_C_correct : 5 + (-6) - (-7) = 5 - 6 + 7 := 
by
  sorry

end option_C_correct_l342_342875


namespace statement_A_statement_B_statement_C_statement_D_l342_342644

variables {V : Type*} [inner_product_space ℝ V] -- assuming V is the vector space

-- Definitions and conditions
variables 
  (a b c : V) -- non-zero vectors
  (h_nonzero_a : a ≠ 0) 
  (h_nonzero_b : b ≠ 0) 
  (h_nonzero_c : c ≠ 0)

-- Statement A: If \( |\overrightarrow{a} - \overrightarrow{b}| = |\overrightarrow{a}| + |\overrightarrow{b}| \), then \( \overrightarrow{a} \text{ and } \overrightarrow{b} \text{ are collinear and in opposite directions} \)
theorem statement_A 
  (hA : ∥a - b∥ = ∥a∥ + ∥b∥) : 
  ∃ k : ℝ, k < 0 ∧ a = k • b :=
sorry

-- Statement B: If \( \overrightarrow{a} \parallel \overrightarrow{b} \) and \( \overrightarrow{b} \parallel \overrightarrow{c} \), then \( \overrightarrow{a} \parallel \overrightarrow{c} \)
theorem statement_B 
  (hB1 : ∃ λ : ℝ, b = λ • a)
  (hB2 : ∃ μ : ℝ, c = μ • b) : 
  ∃ ν : ℝ, c = ν • a :=
sorry

-- Statement C: If \( \overrightarrow{a} \cdot \overrightarrow{c} = \overrightarrow{b} \cdot \overrightarrow{c} \), then \( \overrightarrow{a} = \overrightarrow{b} \)
theorem statement_C 
  (hC : ⟪a, c⟫ = ⟪b, c⟫) : 
  a = b ↔ ∃ k : ℝ, k ≠ 0 ∧ (a - b) = k • c :=
sorry

-- Statement D: If \( |\overrightarrow{a} + \overrightarrow{b}| = |\overrightarrow{a} - \overrightarrow{b}| \), then \( \overrightarrow{a} \perp \overrightarrow{b} \)
theorem statement_D 
  (hD : ∥a + b∥ = ∥a - b∥) :
  ⟪a, b⟫ = 0 :=
sorry

end statement_A_statement_B_statement_C_statement_D_l342_342644


namespace no_such_P_l342_342557

-- We define a polynomial P in terms of its coefficients
def P (X : ℝ) (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  (X - a 0)^3 * (X - a 1)^2 * (n - 1).finset.prod (λ i, (X - a i.succ.succ))

-- We define another polynomial Q in a similar manner
def Q (X : ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  (X - b 0)^3 * (X - b 1)^2 * (n - 1).finset.prod (λ i, (X - b i.succ.succ))

theorem no_such_P (a b : ℕ → ℝ) (n : ℕ) :
  ¬ ∃ P, (∀ X, P X = P X (a, n)) ∧ (∀ X, P X + 1 = Q X (b, n)) := 
sorry

end no_such_P_l342_342557


namespace quadrilateral_square_l342_342882

theorem quadrilateral_square
  (Q : Type) [quadrilateral Q]
  (d1 d2 : diagonal Q)
  (h1 : perpendicular d1 d2)
  (h2 : bisect d1 d2)
  (h3 : equal_length d1 d2) :
  is_square Q :=
sorry

end quadrilateral_square_l342_342882


namespace part_I_part_II_l342_342582

variables {x : Real}
noncomputable def a : (Real × Real) := (Real.sin x, Real.cos x)
noncomputable def b : (Real × Real) := (Real.sqrt 3, -1)

-- Part I
theorem part_I (h_parallel : -(Real.sin x) = Real.sqrt 3 * Real.cos x) : 
  Real.sin x ^ 2 - 6 * Real.cos x ^ 2 = -3 / 4 :=
begin
  sorry
end

-- Part II
def f (x : Real) : Real := 
  let a := (Real.sin x, Real.cos x)
  let b := (Real.sqrt 3, -1)
  a.fst * b.fst + a.snd * b.snd

theorem part_II :
  ∀ k : Int, ∀ x : Real, (π / 3 + k * π) ≤ x ∧ x ≤ (5 * π / 6 + k * π) →

  ( 2 * Real.sin (2 * x - π / 6 - (Real.sqrt 3 * Real.sin x - Real.cos x) ))≤ 0 :=
begin
  sorry
end

end part_I_part_II_l342_342582


namespace greatest_integer_prime_l342_342864

def is_prime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, m > 0 → m < n → n % m ≠ 0

theorem greatest_integer_prime (x : ℤ) :
  is_prime (|8 * x ^ 2 - 56 * x + 21|) → ∀ y : ℤ, (is_prime (|8 * y ^ 2 - 56 * y + 21|) → y ≤ x) :=
by
  sorry

end greatest_integer_prime_l342_342864


namespace investment_values_order_l342_342820

theorem investment_values_order :
  let I : ℝ := 200
  let rD₁ : ℝ := 1.1
  let rD₂ : ℝ := 0.9
  let rE₁ : ℝ := 0.7
  let rE₂ : ℝ := 1.5
  let rF₂ : ℝ := 0.95
  let D₁ : ℝ := I * rD₁
  let E₁ : ℝ := I * rE₁
  let F₁ : ℝ := I 
  let D₂ : ℝ := D₁ * rD₂
  let E₂ : ℝ := E₁ * rE₂
  let F₂ : ℝ := F₁ * rF₂
  F₂ < D₂ ∧ D₂ < E₂ :=
by {
  let I : ℝ := 200
  let rD₁ : ℝ := 1.1
  let rD₂ : ℝ := 0.9
  let rE₁ : ℝ := 0.7
  let rE₂ : ℝ := 1.5
  let rF₂ : ℝ := 0.95
  let D₁ : ℝ := I * rD₁
  let E₁ : ℝ := I * rE₁
  let F₁ : ℝ := I 
  let D₂ : ℝ := D₁ * rD₂
  let E₂ : ℝ := E₁ * rE₂
  let F₂ : ℝ := F₁ * rF₂
  have : D₁ = 220 := rfl,
  have : E₁ = 140 := rfl,
  have : F₁ = 200 := rfl,
  have : D₂ = 198 := rfl,
  have : E₂ = 210 := rfl,
  have : F₂ = 190 := rfl,
  sorry
}

end investment_values_order_l342_342820


namespace sum_of_tens_and_ones_digits_of_seven_eleven_l342_342451

theorem sum_of_tens_and_ones_digits_of_seven_eleven :
  let n := (3 + 4) ^ 11 in 
  (let ones := n % 10 in
   let tens := (n / 10) % 10 in
   ones + tens = 7) := 
by sorry

end sum_of_tens_and_ones_digits_of_seven_eleven_l342_342451


namespace least_number_remainder_l342_342573

theorem least_number_remainder (n : ℕ) (h1 : n % 34 = 4) (h2 : n % 5 = 4) : n = 174 :=
  sorry

end least_number_remainder_l342_342573


namespace largest_inscribed_triangle_area_l342_342992

theorem largest_inscribed_triangle_area (r : ℝ) (h_r : r = 8) :
  ∃ A : ℝ, A = 64 ∧ (∀ Δ, Δ.isInscribedInCircleOfRadius r → Δ.hasBaseAsDiameter → Δ.area ≤ A) :=
by
  sorry

end largest_inscribed_triangle_area_l342_342992


namespace smallest_12_digit_proof_l342_342083

def is_12_digit_number (n : ℕ) : Prop :=
  n >= 10^11 ∧ n < 10^12

def contains_each_digit_0_to_9 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] → d ∈ n.digits 10

def is_divisible_by_36 (n : ℕ) : Prop :=
  n % 36 = 0

noncomputable def smallest_12_digit_divisible_by_36_and_contains_each_digit : ℕ :=
  100023457896

theorem smallest_12_digit_proof :
  is_12_digit_number smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  contains_each_digit_0_to_9 smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  is_divisible_by_36 smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  ∀ m : ℕ, is_12_digit_number m ∧ contains_each_digit_0_to_9 m ∧ is_divisible_by_36 m →
  m >= smallest_12_digit_divisible_by_36_and_contains_each_digit :=
by
  sorry

end smallest_12_digit_proof_l342_342083


namespace regular_polygon_sides_l342_342946

theorem regular_polygon_sides (θ : ℝ) (n : ℕ) (h1 : θ = 18) (h2 : 360 = θ * n) : n = 20 :=
by {
  sorry
}

end regular_polygon_sides_l342_342946


namespace find_first_number_l342_342089

def bin_op (m n : ℕ) : ℕ := n ^ 2 - m

theorem find_first_number (x : ℕ) : bin_op x 3 = 6 → x = 3 := by
  intro h
  have eq1 : bin_op x 3 = 9 - x := rfl
  rw [eq1] at h
  rst sorry

end find_first_number_l342_342089


namespace elastic_collision_inelastic_collision_l342_342423

-- Definition of conditions
variables {m L V : ℝ}
variables (w1 w2 : ℝ → Prop)

-- Proof problem for Elastic Collision
theorem elastic_collision (w1_at_collision : w1 L) (w2_at_collision : w2 L)  :
  let v1_after := V
      v2_after := -V in
  w1 L ∧ w2 L := sorry

-- Proof problem for Inelastic Collision
theorem inelastic_collision (w1_at_collision : w1 L) (w2_at_collision : w2 L)  :
  let omega := V / (2 * L) in
  w1 L ∧ w2 L := sorry

end elastic_collision_inelastic_collision_l342_342423


namespace smallest_value_abs_diff_l342_342868

theorem smallest_value_abs_diff : ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ (∀ (x y : ℕ), 0 < x → 0 < y → |12 ^ x - 5 ^ y| ≥ 7) ∧ |12 ^ m - 5 ^ n| = 7 :=
sorry

end smallest_value_abs_diff_l342_342868


namespace find_cost_price_l342_342534

theorem find_cost_price (SP : ℤ) (profit_percent : ℚ) (CP : ℤ) (h1 : SP = CP + (profit_percent * CP)) (h2 : SP = 240) (h3 : profit_percent = 0.25) : CP = 192 :=
by
  sorry

end find_cost_price_l342_342534


namespace find_m_value_l342_342580

theorem find_m_value (x m : ℝ)
  (h1 : -3 * x = -5 * x + 4)
  (h2 : m^x - 9 = 0) :
  m = 3 ∨ m = -3 := 
sorry

end find_m_value_l342_342580


namespace complex_number_in_first_quadrant_l342_342123

def z : ℂ := (2 + complex.I) / (1 - complex.I)

theorem complex_number_in_first_quadrant :
  ∃ a b : ℝ, z = a + b * complex.I ∧ 0 < a ∧ 0 < b :=
sorry

end complex_number_in_first_quadrant_l342_342123


namespace sin_B_value_l342_342171

variable {A B C : Real}
variable {a b c : Real}
variable {sin_A sin_B sin_C : Real}

-- Given conditions as hypotheses
axiom h1 : c = 2 * a
axiom h2 : b * sin_B - a * sin_A = (1 / 2) * a * sin_C

-- The statement to prove
theorem sin_B_value : sin_B = Real.sqrt 7 / 4 :=
by
  -- Proof omitted
  sorry

end sin_B_value_l342_342171


namespace sum_of_solutions_eq_five_l342_342466

theorem sum_of_solutions_eq_five :
  (∑ x in {x : ℝ | x^2 - 5*x + 4 = 6}.to_finset) = 5 :=
sorry

end sum_of_solutions_eq_five_l342_342466


namespace map_length_representation_l342_342306

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l342_342306


namespace am_perp_bc_l342_342786

open Locale Classical

noncomputable theory

-- Given definitions
variables (A B C D E F G M : Type)
variables [has_perp A BC] [diameter BC] [semicircle_intersect AB D] [semicircle_intersect AC E]
  [perpendicular_foot D BC F] [perpendicular_foot E BC G] [intersection DG EF M]

-- Required theorem statement
theorem am_perp_bc : AM ⊥ BC := sorry

end am_perp_bc_l342_342786


namespace find_y_when_x_is_7_l342_342845

theorem find_y_when_x_is_7
  (x y : ℝ)
  (h1 : x * y = 384)
  (h2 : x + y = 40)
  (h3 : x - y = 8)
  (h4 : x = 7) :
  y = 384 / 7 :=
by
  sorry

end find_y_when_x_is_7_l342_342845


namespace parallel_conditions_l342_342605

open Set

-- Define the contextual environment
variables {α : Type*} [plane : Set α] {m n : Set α}

-- Definitions based on conditions
def is_line (l : Set α) : Prop := ∃ (a b : α), a ≠ b ∧ ∀ t, l t ↔ t = a ∨ t = b
def is_plane (p : Set α) : Prop := ∃ (u v w : α), u ≠ v ∧ v ≠ w ∧ w ≠ u ∧ ∀ t, p t ↔ t ∈ span {u, v, w}
def is_subset (A B : Set α) : Prop := ∀ ⦃x⦄, A x → B x
def is_parallel_to (l₁ l₂ : Set α) [is_line l₁] [is_line l₂] : Prop := ∀ a ∈ l₁, ∀ b ∈ l₂, a ≠ b ∧ (l₁ = l₂ ∨ l₁ ∩ l₂ = ∅)
def is_parallel_to_plane (l : Set α) (p : Set α) [is_line l] [is_plane p] : Prop := ∀ ⦃x y : α⦄, l x → l y → (span {x, y} ∩ p = ∅)

-- Main theorem statement
theorem parallel_conditions 
  (h₁ : is_line n) 
  (h₂ : is_line m) 
  (h₃ : is_plane α) 
  (h₄ : is_subset n α) 
  (h₅ : ¬ is_subset m α) 
  : (is_parallel_to m n → is_parallel_to_plane m α) 
  ∧ ¬ (is_parallel_to_plane m α → is_parallel_to m n) := 
sorry

end parallel_conditions_l342_342605


namespace count_magic_numbers_l342_342659

def is_magic_number (N : ℕ) : Prop :=
  ∀ (k : ℕ) (P : ℕ), (N < 130) ∧ (N ∣ (P * 10 ^ k + N))

theorem count_magic_numbers : 
  ∑ n in finset.range 130, (if is_magic_number n then 1 else 0) = 9 := 
sorry

end count_magic_numbers_l342_342659


namespace points_in_plane_region_l342_342608

def point_in_region (x y : ℝ) : Prop := 3 * x + 2 * y - 1 ≥ 0

def P1 := (0 : ℝ, 0 : ℝ)
def P2 := (1 : ℝ, 1 : ℝ)
def P3 := (1 / 3 : ℝ, 0 : ℝ)

theorem points_in_plane_region :
  (point_in_region (P2.1) (P2.2)) ∧ (point_in_region (P3.1) (P3.2)) ∧ ¬(point_in_region (P1.1) (P1.2)) :=
by
  sorry

end points_in_plane_region_l342_342608


namespace binom_1500_1_l342_342994

-- Define binomial coefficient
def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

-- Theorem statement
theorem binom_1500_1 : binom 1500 1 = 1500 :=
by
  sorry

end binom_1500_1_l342_342994


namespace monotonically_decreasing_intervals_l342_342079

open Real

-- Define the function f
def f (x: ℝ) : ℝ := 3 * (sin x) ^ 2 + 2 * (sin x) * (cos x) + (cos x) ^ 2 - 2

-- Define the interval where we need to prove the function is monotonically decreasing
def interval (k : ℤ) (x : ℝ) : Prop := k * π + (3/8) * π ≤ x ∧ x ≤ k * π + (7/8) * π

-- The theorem statement
theorem monotonically_decreasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, interval k x → ∀ x₁ x₂ : ℝ, x₁ < x₂ → interval k x₁ → interval k x₂ → f x₂ ≤ f x₁ :=
by
  sorry

end monotonically_decreasing_intervals_l342_342079


namespace map_length_represents_distance_l342_342318

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l342_342318


namespace parallelogram_triangle_area_ratios_l342_342179

/-- In a parallelogram ABCD, points E and F are the midpoints of AB and BC, respectively,
    and P is the intersection of EC and FD. The segments AP, BP, CP, and DP divide the
    parallelogram into four triangles whose areas are in the ratio 1 : 2 : 3 : 4. -/
theorem parallelogram_triangle_area_ratios
  (A B C D E F P : Type)
  [Midpoint E A B]
  [Midpoint F B C]
  [Intersection P E C F D]
  [Parallelogram ABCD] :
  area (triangle A P B) : area (triangle B P C) : area (triangle C P D) : area (triangle D P A) = 1 : 2 : 3 : 4 :=
sorry

end parallelogram_triangle_area_ratios_l342_342179


namespace find_parallelogram_of_XA_XB_XC_XD_l342_342365

noncomputable def is_parallelogram {α : Type*} [AddCommGroup α] (A' B' C' D' : α) : Prop :=
  A' + C' = B' + D'

variables {P : Type*} [AffineSpace ℝ P]
variables {A B C D X : P} 
variables {A' B' C' D' : P}
variables {XA XB XC XD : AffineSubspace ℝ P}

-- Define the condition that A', B', C', D' lie on the respective lines
axiom A'_condition : A' ∈ XA
axiom B'_condition : B' ∈ XB
axiom C'_condition : C' ∈ XC
axiom D'_condition : D' ∈ XD

-- Main theorem stating that A'B'C'D' is a parallelogram
theorem find_parallelogram_of_XA_XB_XC_XD
  (P : AffineSubspace ℝ P)
  (convex_quadrilateral : ∃ (A B C D : P), A ∈ P ∧ B ∈ P ∧ C ∈ P ∧ D ∈ P)
  (X_not_in_P : X ∉ P)
  (XA : Submodule ℝ P = line_through X A)
  (XB : Submodule ℝ P = line_through X B)
  (XC : Submodule ℝ P = line_through X C)
  (XD : Submodule ℝ P = line_through X D)
   : ∃ (A' B' C' D' : P), is_parallelogram A' B' C' D' := sorry

end find_parallelogram_of_XA_XB_XC_XD_l342_342365


namespace negation_of_implication_l342_342395

theorem negation_of_implication (x : ℝ) :
  ¬ (x ≠ 3 ∧ x ≠ 2 → x^2 - 5 * x + 6 ≠ 0) ↔ (x = 3 ∨ x = 2 → x^2 - 5 * x + 6 = 0) := 
by {
  sorry
}

end negation_of_implication_l342_342395


namespace sum_of_interior_diagonals_of_box_l342_342007

theorem sum_of_interior_diagonals_of_box (a b c : ℝ) 
  (h_edges : 4 * (a + b + c) = 60)
  (h_surface_area : 2 * (a * b + b * c + c * a) = 150) : 
  4 * Real.sqrt (a^2 + b^2 + c^2) = 20 * Real.sqrt 3 := 
by
  sorry

end sum_of_interior_diagonals_of_box_l342_342007


namespace catering_budget_l342_342756

/-- Jenny is planning her catering budget for her wedding. 
She's going to have 80 guests. 
3 times as many guests want steak as chicken. 
Each steak entree costs $25 and each chicken entree costs $18. 
Prove that the total catering budget is $1860. -/
theorem catering_budget
  (C S : ℕ) -- Number of guests wanting chicken and steak
  (total_guests : ℕ) (total_guests = 80) -- Total number of guests
  (relation : S = 3 * C) -- 3 times as many guests want steak as chicken
  (cost_steak : ℕ) (cost_steak = 25) -- Cost per steak entree
  (cost_chicken : ℕ) (cost_chicken = 18) -- Cost per chicken entree
  (guest_equation : C + S = 80) -- Equation representing the total number of guests
  : (S * cost_steak + C * cost_chicken = 1860) := 
sorry

end catering_budget_l342_342756


namespace map_length_represents_distance_l342_342319

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l342_342319


namespace find_sam_current_age_l342_342688

def Drew_current_age : ℕ := 12

def Drew_age_in_five_years : ℕ := Drew_current_age + 5

def Sam_age_in_five_years : ℕ := 3 * Drew_age_in_five_years

def Sam_current_age : ℕ := Sam_age_in_five_years - 5

theorem find_sam_current_age : Sam_current_age = 46 := by
  sorry

end find_sam_current_age_l342_342688


namespace quadratic_inequality_solution_set_l342_342400

theorem quadratic_inequality_solution_set (a b c : ℝ) (h₁ : a < 0) (h₂ : b^2 - 4 * a * c < 0) :
  ∀ x : ℝ, a * x^2 + b * x + c < 0 :=
sorry

end quadratic_inequality_solution_set_l342_342400


namespace regular_polygon_has_20_sides_l342_342949

-- Definition of a regular polygon with an exterior angle of 18 degrees
def regular_polygon_sides (external_angle : ℝ) : ℕ :=
  if external_angle > 0 then (360 / external_angle).toInt else 0

theorem regular_polygon_has_20_sides :
  regular_polygon_sides 18 = 20 :=
by
  sorry

end regular_polygon_has_20_sides_l342_342949


namespace triangle_equilateral_l342_342826

variables {A B C K M T O : Type*}

/-- Assume we have a triangle ABC with the following properties:
1. The median from A (AK), the altitude from B (BM), and the angle bisector from C (CT) intersect at point O.
2. These segments divide the triangle into six smaller triangles.
3. The areas of three shaded triangles are equal.
We aim to prove that triangle ABC is equilateral.
-/
theorem triangle_equilateral
  (angle_bisector : ∃ (AK : B → C → Type*), is_angle_bisector AK A)
  (median : ∃ (BM : A → C → Type*), is_median BM B)
  (altitude : ∃ (CT : A → B → Type*), is_altitude CT C)
  (intersect_at_O : ∀ (B' : Type*) (C' : Type*) (AO : O → A → Type*), AO ∩ AK = O ∧ AO ∩ BM = O ∧ AO ∩ CT = O)
  (divide_into_six : ∀ (∆ABC : Type*) (∆small : Type*), ∆ABC → divide_six ∆small)
  (areas_equal : ∀ (∆1 ∆2 ∆3 : Type*) (area : ∀ x : Type*, x → ℝ), area ∆1 = area ∆2 ∧ area ∆2 = area ∆3) 
  : is_equilateral △ABC :=
sorry

end triangle_equilateral_l342_342826


namespace remaining_amount_to_be_paid_l342_342658

-- Define the deposit and total percentage
def deposit := 55
def percentage := 0.1
def total_cost := deposit / percentage
def remaining_amount := total_cost - deposit

-- Proof statement
theorem remaining_amount_to_be_paid : remaining_amount = 495 :=
by
  have total_cost_correct : total_cost = 550 := by 
    unfold total_cost
    ring
  have remain : remaining_amount = total_cost - deposit := by 
    unfold remaining_amount
    ring
  rw [total_cost_correct] at remain
  norm_num at remain
  exact remain

end remaining_amount_to_be_paid_l342_342658


namespace binomial_coeff_sum_l342_342619

theorem binomial_coeff_sum (n : ℕ) (h : n > 0) (h_coeff : -8 * (Nat.choose n 3) = -80) :
  2^n = 32 :=
begin
  sorry
end

end binomial_coeff_sum_l342_342619


namespace angelina_speed_park_to_library_l342_342537

theorem angelina_speed_park_to_library :
  let v := 5 in
  ((6 : ℝ) * v = 30) ∧ 
  (150 / v - 10 = 100 / v) ∧ 
  (250 / (v / 2) = 100) ∧ 
  (300 / (6 * v) = 10) ∧ 
  (100 = 10 + 20) :=
by {
  let v := 5,
  split,
  -- Proving the speed from park to library
  { exact congrArg (λ x, x * (6 : ℝ)) (rfl : v = 5) },
  -- Proving the initial speed based on time difference between grocery to gym and home to grocery.
  { apply (congrArg (λ t, 150 / t - 10 = 100 / t) (rfl : v = 5))},
  -- Proving the time to park from the gym based on speed v
  { apply (congrArg (λ t, 250 / (t / 2) = 100) (rfl : v = 5))},
  -- Proving the time to library from the park based on speed v
  { apply (congrArg (λ t, 300 / (6 * t) = 10) (rfl : v = 5))},
  -- Verifying the total time difference condition
  { exact (100 = 10 + 20) }
} sorry

end angelina_speed_park_to_library_l342_342537


namespace expectation_linear_l342_342663

noncomputable def pdf_X (x : ℝ) : ℝ :=
  (1 / (2 * Real.sqrt (2 * Real.pi))) * Real.exp (-((x + 2) ^ 2) / 8)

theorem expectation_linear:
  let E := λ (X : ℝ → ℝ) (pdf : ℝ → ℝ), ∫ x in set.univ, X x * pdf x :=
  ∀ (X : ℝ → ℝ), (E (λ x, 2 * X x - 1) pdf_X) = -5 :=
sorry

end expectation_linear_l342_342663


namespace necessary_but_not_sufficient_l342_342025

theorem necessary_but_not_sufficient (a b : ℝ) : (a > b) → (a > b - 1) ∧ ¬(a > b - 1 → a > b) := by
  intro h
  split
  · linarith
  · intro h1
    sorry

end necessary_but_not_sufficient_l342_342025


namespace earning_80_yuan_represents_l342_342697

-- Defining the context of the problem
def spending (n : Int) : Int := -n
def earning (n : Int) : Int := n

-- The problem statement as a Lean theorem
theorem earning_80_yuan_represents (x : Int) (hx : earning x = 80) : x = 80 := 
by
  sorry

end earning_80_yuan_represents_l342_342697


namespace complement_intersection_eq_l342_342610

variable (U P Q : Set ℕ)
variable (hU : U = {1, 2, 3})
variable (hP : P = {1, 2})
variable (hQ : Q = {2, 3})

theorem complement_intersection_eq : 
  (U \ (P ∩ Q)) = {1, 3} := by
  sorry

end complement_intersection_eq_l342_342610


namespace evaluate_A_l342_342472

noncomputable def proof_problem (m n : ℝ) (h0 : 0 < m) (h1 : 0 < n) : ℝ :=
  let A := ( (4 * m^2 * n^2 / (4 * m * n - m^2 - 4 * n^2)) - 
             ( (2 + n/m + m/n) / (4/(m * n) - 1/n^2 - 4/m^2) ) ) ^ (1 / 2) *
             ( (sqrt (m * n)) / (m - 2 * n) )
  in if 0 < m / n ∧ m / n ≤ 1 then (m - n)
     else if 1 < m / n ∧ m / n < 2 then (n - m)
     else if 2 < m / n then (m - n)
     else 0 -- should never happen

theorem evaluate_A (m n : ℝ) (h0 : 0 < m) (h1 : 0 < n) (h_ratio : 0 < m / n) :
  proof_problem m n h0 h1 = 
  if 1 < m / n ∧ m / n < 2 then n - m
  else m - n :=
by 
  sorry

end evaluate_A_l342_342472


namespace course_choice_gender_related_l342_342412
open scoped Real

theorem course_choice_gender_related :
  let a := 40 -- Males choosing Calligraphy
  let b := 10 -- Males choosing Paper Cutting
  let c := 30 -- Females choosing Calligraphy
  let d := 20 -- Females choosing Paper Cutting
  let n := a + b + c + d -- Total number of students
  let χ_squared := (n * (a*d - b*c)^2) / ((a+b) * (c+d) * (a+c) * (b+d))
  χ_squared > 3.841 := 
by
  sorry

end course_choice_gender_related_l342_342412


namespace calculate_selling_price_l342_342004

noncomputable def originalPrice : ℝ := 120
noncomputable def firstDiscountRate : ℝ := 0.30
noncomputable def secondDiscountRate : ℝ := 0.15
noncomputable def taxRate : ℝ := 0.08

def discountedPrice1 (originalPrice firstDiscountRate : ℝ) : ℝ :=
  originalPrice * (1 - firstDiscountRate)

def discountedPrice2 (discountedPrice1 secondDiscountRate : ℝ) : ℝ :=
  discountedPrice1 * (1 - secondDiscountRate)

def finalPrice (discountedPrice2 taxRate : ℝ) : ℝ :=
  discountedPrice2 * (1 + taxRate)

theorem calculate_selling_price : 
  finalPrice (discountedPrice2 (discountedPrice1 originalPrice firstDiscountRate) secondDiscountRate) taxRate = 77.112 := 
sorry

end calculate_selling_price_l342_342004


namespace find_b_l342_342834

theorem find_b (b : ℚ) (h : b * (-3) - (b - 1) * 5 = b - 3) : b = 8 / 9 :=
by
  sorry

end find_b_l342_342834


namespace problem_statement_l342_342126

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x

-- Conditions and conclusions
theorem problem_statement :
  (∃ x, is_local_max f x ∧ ∀ y, f y < f x) ∧
  (∀ b, (∀ x, f x = b → ∃! x (h : f x = b), (∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3)) → 0 < b ∧ b < 6 * Real.exp (-3)) :=
by
  sorry

end problem_statement_l342_342126


namespace least_five_digit_congruent_to_six_mod_seventeen_l342_342445

theorem least_five_digit_congruent_to_six_mod_seventeen : ∃ x : ℕ, x ≥ 10000 ∧ x < 100000 ∧ x % 17 = 6 ∧ ∀ y : ℕ, y ≥ 10000 ∧ y < 100000 ∧ y % 17 = 6 → x ≤ y :=
  by
    sorry

end least_five_digit_congruent_to_six_mod_seventeen_l342_342445


namespace minimum_possible_value_P0_l342_342378

theorem minimum_possible_value_P0 {P : ℤ → ℤ} (hP : ∀ x, P x = x^2 + d * x + e)
  (a : ℤ) (h1 : a ≠ 20) (h2 : a ≠ 22)
  (h3 : a * P a = 20 * P 20) 
  (h4 : 20 * P 20 = 22 * P 22) : 
  (∃ d e, P 0 = e ∧ e > 0 ∧ e = 20) :=
sorry

end minimum_possible_value_P0_l342_342378


namespace inscribed_circles_of_triangles_are_equal_l342_342428

theorem inscribed_circles_of_triangles_are_equal
  {A B C D E F : Point} {r : ℝ} 
  (h1 : intersect_along_hexagon A B C D E F)
  (h2 : separates_6_smaller_triangles A B C D E F)
  (h3 : ∀ (X : Point) (Y Z : Point) (in_smaller_triangle : X ∈ {Y, Z}), inscribed_circle_radius (triangle X Y Z) = r):
  ∃ r' : ℝ, ∀ (T₁ T₂ : Triangle), radius_of_inscribed_circle(T₁) = radius_of_inscribed_circle(T₂) :=
sorry

end inscribed_circles_of_triangles_are_equal_l342_342428


namespace sum_of_areas_of_rectangles_l342_342847

theorem sum_of_areas_of_rectangles :
  let width := 2
  let lengths := [1, 4, 9, 16, 25, 36]
  let areas := lengths.map (λ l => l * width)
  let total_area := areas.sum
  total_area = 182 := by
  sorry

end sum_of_areas_of_rectangles_l342_342847


namespace min_h_x1_x2_l342_342144

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log (x + 1) + a * x^2 - x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * log x - 1 / x - log (x + 1) - a * x^2 + 2 * x

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := f a x + g a x

def x2 (x1 : ℝ) : ℝ := 1 / x1

theorem min_h_x1_x2 (a : ℝ) (x1 : ℝ) (hx1 : 0 < x1 ∧ x1 ≤ 1 / Real.exp 1) 
  (critical_points : ∀ x, h a x = 0 → x = x1 ∨ x = x2 x1):
  h a x1 - h a (x2 x1) = 4 / Real.exp 1 := by
  sorry

end min_h_x1_x2_l342_342144


namespace system_of_equations_solution_system_of_inequalities_solution_l342_342484

-- Problem (1): Solve the system of equations
theorem system_of_equations_solution :
  ∃ (x y : ℝ), (3 * (y - 2) = x - 1) ∧ (2 * (x - 1) = 5 * y - 8) ∧ (x = 7) ∧ (y = 4) :=
by
  sorry

-- Problem (2): Solve the system of linear inequalities
theorem system_of_inequalities_solution :
  ∃ (x : ℝ), (3 * x ≤ 2 * x + 3) ∧ ((x + 1) / 6 - 1 < (2 * (x + 1)) / 3) ∧ (-3 < x) ∧ (x ≤ 3) :=
by
  sorry

end system_of_equations_solution_system_of_inequalities_solution_l342_342484


namespace adults_not_wearing_blue_l342_342037

-- Conditions
def children : ℕ := 45
def adults : ℕ := children / 3
def adults_wearing_blue : ℕ := adults / 3

-- Theorem Statement
theorem adults_not_wearing_blue :
  adults - adults_wearing_blue = 10 :=
sorry

end adults_not_wearing_blue_l342_342037


namespace unique_real_root_iff_m_eq_one_l342_342165

-- Define the quadratic polynomial
def quadratic (m : ℝ) : ℝ → ℝ := λ x, x^2 - 6 * m * x + 9 * m

-- Define the discriminant condition for a quadratic to have exactly one real root
def discriminant_zero (a b c : ℝ) : Prop := b^2 - 4 * a * c = 0

-- State the theorem
theorem unique_real_root_iff_m_eq_one (m : ℝ) :
  (∃ x : ℝ, quadratic m x = 0) ∧ (discriminant_zero 1 (-6*m) (9*m)) ↔ m = 1 :=
begin
  sorry
end

end unique_real_root_iff_m_eq_one_l342_342165


namespace solve_for_a_l342_342657

theorem solve_for_a (a : ℚ) (h : a + a/3 + a/4 = 11/4) : a = 33/19 :=
sorry

end solve_for_a_l342_342657


namespace total_packs_of_groceries_l342_342792

-- Definitions based on conditions
def packs_of_cookies : Nat := 4
def packs_of_cake : Nat := 22
def packs_of_chocolate : Nat := 16

-- The proof statement
theorem total_packs_of_groceries : packs_of_cookies + packs_of_cake + packs_of_chocolate = 42 :=
by
  -- Proof skipped using sorry
  sorry

end total_packs_of_groceries_l342_342792


namespace nonnegative_solutions_x_squared_eq_neg5x_l342_342652

theorem nonnegative_solutions_x_squared_eq_neg5x : 
  ∀ x : ℝ, x^2 = -5 * x → (x ≥ 0) → x = 0 :=
by
  intros x h_eq h_nonneg
  have h_eq_Rearranged : x * (x + 5) = 0 := by
    calc
      x * (x + 5)
        = x * x + x * 5 : by ring
        ... = x^2 + 5 * x : by ring
        ... = 0 : by rw [h_eq]
  have h_solutions : x = 0 ∨ x = -5 := by
    apply eq_zero_or_eq_zero_of_mul_eq_zero
    exact h_eq_Rearranged
  cases h_solutions with h_zero h_neg
  · exact h_zero
  · exfalso
    linarith

end nonnegative_solutions_x_squared_eq_neg5x_l342_342652


namespace remainder_of_S_div_1000_l342_342774

theorem remainder_of_S_div_1000 :
  let S := (Finset.filter (λ n : ℕ, ∃ m : ℕ, n^2 + 12 * n - 3007 = m^2)
    (Finset.range 2000)).sum in
  (S % 1000) = 566 := by
  sorry

end remainder_of_S_div_1000_l342_342774


namespace carl_candy_bars_l342_342564

/-- 
Carl earns $0.75 every week for taking out his neighbor's trash. 
Carl buys a candy bar every time he earns $0.50. 
After four weeks, Carl will be able to buy 6 candy bars.
-/
theorem carl_candy_bars :
  (0.75 * 4) / 0.50 = 6 := 
  by
    sorry

end carl_candy_bars_l342_342564


namespace map_distance_l342_342345

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l342_342345


namespace map_length_scale_l342_342296

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l342_342296


namespace total_shaded_area_l342_342856

theorem total_shaded_area (r : ℝ) (h_r : r = 6) (h_perpendicular : True) (h_theta : ∀ sector, sector.angle = 120) : 
  2 * (1/3 * π * r^2) + 4 * r^2 = 24 * π + 144 :=
by
  sorry

end total_shaded_area_l342_342856


namespace cost_of_phone_call_l342_342388

-- Define the ceiling function
def ceil (x : ℝ) : ℝ := ⌈x⌉

-- Define the cost function g(t)
def g (t : ℝ) : ℝ := 1.06 * (0.75 * ceil t + 1)

-- The main theorem stating the cost for 5.5 minutes
theorem cost_of_phone_call : g 5.5 = 5.83 :=
by
  -- Proof omitted, replace by sorry
  sorry

end cost_of_phone_call_l342_342388


namespace sum_first_10_a_b_n_l342_342620

noncomputable def a_n (a1 : ℕ) (n : ℕ) : ℕ := a1 + (n - 1)
noncomputable def b_n (b1 : ℕ) (n : ℕ) : ℕ := b1 + (n - 1)
noncomputable def a_b_n (a1 b1 : ℕ) (n : ℕ) : ℕ := a_n a1 (b_n b1 n)

theorem sum_first_10_a_b_n
  (a1 b1 : ℕ)
  (h1 : a1 + b1 = 5)
  (h2 : a1 > b1)
  (h3 : 0 < a1 ∧ 0 < b1) :
  ∑ i in finset.range 10, a_b_n a1 b1 (i + 1) = 85 := 
by
  sorry

end sum_first_10_a_b_n_l342_342620


namespace proof_problem1_proof_problem2_l342_342544

noncomputable def problem1 : Prop := 
  sqrt 3 * sqrt 2 - sqrt 12 / sqrt 8 = sqrt 6 / 2

noncomputable def problem2 : Prop := 
  (sqrt 2 - 3)^2 - sqrt 2^2 - sqrt (2^2) - sqrt 2 = 7 - 7 * sqrt 2

theorem proof_problem1 : problem1 := 
  by sorry

theorem proof_problem2 : problem2 := 
  by sorry

end proof_problem1_proof_problem2_l342_342544


namespace gcd_153_119_l342_342431

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  have h1 : 153 = 119 * 1 + 34 := by rfl
  have h2 : 119 = 34 * 3 + 17 := by rfl
  have h3 : 34 = 17 * 2 := by rfl
  sorry

end gcd_153_119_l342_342431


namespace f_plus_g_at_1_l342_342485

-- Define f(x) as an even function
def evenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = f(x)

-- Define g(x) as an odd function
def oddFunction (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g(-x) = -g(x)

-- Conditions
variable (f g : ℝ → ℝ)
variable (h_even : evenFunction f)
variable (h_odd : oddFunction g)
variable (h_eq : ∀ x : ℝ, f(x) - g(x) = x^3 + x^2 + 1)

-- The theorem we need to prove
theorem f_plus_g_at_1 : f(1) + g(1) = 1 :=
by
  sorry -- skip the proof

end f_plus_g_at_1_l342_342485


namespace find_beta_l342_342819

noncomputable def proportional_constant : ℝ := 6 / 18

theorem find_beta : (∃ (k : ℝ), α = k * β ∧ α = 6 ∧ β = 18) → α = 15 → β = 45 :=
by
  assume hα h15,
  sorry

end find_beta_l342_342819


namespace four_sq_geq_prod_sum_l342_342892

variable {α : Type*} [LinearOrderedField α]

theorem four_sq_geq_prod_sum (a b c d : α) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) :=
sorry

end four_sq_geq_prod_sum_l342_342892


namespace value_of_composite_function_l342_342633

def f (x : ℝ) : ℝ :=
  if x >= 0 then x + 1 else x^2

theorem value_of_composite_function : f (f (-2)) = 5 := by
  sorry

end value_of_composite_function_l342_342633


namespace no_integer_pairs_satisfy_equation_l342_342061

theorem no_integer_pairs_satisfy_equation :
  ∀ (a b : ℤ), a^3 + 3 * a^2 + 2 * a ≠ 125 * b^3 + 75 * b^2 + 15 * b + 2 :=
by
  intro a b
  sorry

end no_integer_pairs_satisfy_equation_l342_342061


namespace fraction_simplest_form_iff_n_odd_l342_342076

theorem fraction_simplest_form_iff_n_odd (n : ℤ) :
  (Nat.gcd (3 * n + 10) (5 * n + 16) = 1) ↔ (n % 2 ≠ 0) :=
by sorry

end fraction_simplest_form_iff_n_odd_l342_342076


namespace points_concyclic_l342_342368

-- Declare the geometric setup and points
variables {A B C I_a X A' Y Z T: Type}
variable [IsAexcircleCenter I_a A B C]
variable [IsTangent I_a X B C]
variable [IsDiametricallyOppositePoint A' A (Circumcircle A B C)]
variable [OnSegment Y I_a X]
variable [OnSegment Z B A']
variable [OnSegment T C A']
variable [Inradius r (triangle A B C)]
variable h1 : distance I_a Y = r
variable h2 : distance B Z = r
variable h3 : distance C T = r

-- Define the problem statement
theorem points_concyclic (I_a_center : IsAexcircleCenter I_a A B C)
                         (tangent_to_BC : IsTangent I_a X B C)
                         (A_opposite : IsDiametricallyOppositePoint A' A (Circumcircle A B C))
                         (point_Y : OnSegment Y I_a X)
                         (point_Z : OnSegment Z B A')
                         (point_T : OnSegment T C A')
                         (inradius_condition_1 : distance I_a Y = r)
                         (inradius_condition_2 : distance B Z = r)
                         (inradius_condition_3 : distance C T = r) :
  Concyclic X Y Z T :=
begin
  sorry
end

end points_concyclic_l342_342368


namespace hyperbola_standard_equation_l342_342637

theorem hyperbola_standard_equation :
  (∀ (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)
    (h_eccentricity : (sqrt (1 + (b/a)^2)) = sqrt 5),
    ∃ x y : ℝ,
    let C1 := (λ x : ℝ, x^2 = 2*y) in
    let C2 := (λ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) in
    let A := (a, 0) in
    let asymptote := (λ x : ℝ, y = (b/a)*(x - a)) in
    (∀ x, is_tangent at x C1 asymptote) →  (C2 x y) → (x^2 - y^2 / 4 = 1) ) :=
sorry

end hyperbola_standard_equation_l342_342637


namespace combined_error_percentage_l342_342031

theorem combined_error_percentage 
  (S : ℝ) 
  (error_side : ℝ) 
  (error_area : ℝ) 
  (h1 : error_side = 0.20) 
  (h2 : error_area = 0.04) :
  (1.04 * ((1 + error_side) * S) ^ 2 - S ^ 2) / S ^ 2 * 100 = 49.76 := 
by
  sorry

end combined_error_percentage_l342_342031


namespace sin_double_angle_solution_l342_342725

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_solution_l342_342725


namespace map_representation_l342_342260

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l342_342260


namespace b_k_divisible_by_9_count_l342_342222

noncomputable def sum_of_squares (k : ℕ) : ℕ :=
  (k * (k + 1) * (2 * k + 1)) / 6

def is_divisible_by_nine (n : ℕ) : Prop :=
  n % 9 = 0

def count_divisible_by_nine (upper_limit : ℕ) : ℕ :=
  (finset.range upper_limit).filter (λ k, is_divisible_by_nine (sum_of_squares k)).card

theorem b_k_divisible_by_9_count :
  count_divisible_by_nine 101 = 33 :=
by sorry

end b_k_divisible_by_9_count_l342_342222


namespace discarded_cards_correct_last_card_correct_l342_342404

noncomputable theory

-- Conditions
def total_sets : ℕ := 288
def cards_per_set : ℕ := 7
def total_cards : ℕ := total_sets * cards_per_set
def remaining_cards : ℕ := 301
def discarded_cards : ℕ := total_cards - remaining_cards

-- Questions
theorem discarded_cards_correct :
  discarded_cards = 1715 := by
  sorry

theorem last_card_correct :
  ∃ (group_no card_no : ℕ), group_no = 124 ∧ card_no = 3 ∧
    -- Prove that the last remaining card is the 3rd card of the 124th group
    -- This can be interpreted more contextually fixed within operations
    true := by
  sorry

end discarded_cards_correct_last_card_correct_l342_342404


namespace binary_expression_computation_l342_342054

theorem binary_expression_computation :
  (11011₂ + 1010₂ - 10001₂ + 1011₂ - 1110₂) = 001001₂ :=
sorry

end binary_expression_computation_l342_342054


namespace arrange_books_l342_342921

/-- 
A librarian needs to arrange 4 copies of Algebra Basics and 5 copies of Calculus Fundamentals on a bookshelf. 
Prove that the number of ways to arrange these books is 126.
-/
theorem arrange_books : ∀ (algebra_books calculus_books : ℕ), 
  algebra_books = 4 → calculus_books = 5 → 
  (nat.choose (algebra_books + calculus_books) algebra_books) = 126 :=
by
  intros algebra_books calculus_books h1 h2
  rw [h1, h2]
  exact nat.choose_eq_factorial_div_factorial (by norm_num : 9 = 9)
  simpa [mul_assoc, mul_comm, mul_left_comm, nat.factorial] using
    congr_arg (λ x, x / (nat.factorial 4 * nat.factorial 5)) (nat.factorial 9)


end arrange_books_l342_342921


namespace map_scale_l342_342350

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l342_342350


namespace youngest_child_age_l342_342401

-- Define the ages as an arithmetic progression
def arithmetic_progression (n : Nat) (a : Nat) (d : Nat) : List Nat := 
  List.range n |>.map (λ i => a + i * d)

-- Sum of arithmetic progression
def sum_arithmetic_progression (n : Nat) (a : Nat) (d : Nat) := 
  n * (2 * a + (n - 1) * d) / 2

-- Define the conditions
noncomputable
def num_children : Nat := 8

noncomputable
def common_difference : Nat := 3

noncomputable
def total_sum : Nat := 184

-- Define the property to prove
theorem youngest_child_age :
  ∃ a : Nat, 
    let ages := arithmetic_progression num_children a common_difference in
    sum_arithmetic_progression num_children a common_difference = total_sum ∧ List.head ages = 2 :=
begin
  sorry,
end

end youngest_child_age_l342_342401


namespace surface_area_cylinders_l342_342514

def surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r * h + 2 * Real.pi * r^2

theorem surface_area_cylinders :
  let length := 4
  let width := 3
  surface_area (3 / 2) length = 42 * Real.pi ∧
  surface_area (length / 2) width = 56 * Real.pi := 
by
  sorry

end surface_area_cylinders_l342_342514


namespace regular_polygon_has_20_sides_l342_342950

-- Definition of a regular polygon with an exterior angle of 18 degrees
def regular_polygon_sides (external_angle : ℝ) : ℕ :=
  if external_angle > 0 then (360 / external_angle).toInt else 0

theorem regular_polygon_has_20_sides :
  regular_polygon_sides 18 = 20 :=
by
  sorry

end regular_polygon_has_20_sides_l342_342950


namespace local_maximum_no_global_maximum_equation_root_condition_l342_342129

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x
noncomputable def f' (x : ℝ) : ℝ := (x^2 + 2*x - 3) * Real.exp x

theorem local_maximum_no_global_maximum : (∃ x0 : ℝ, f' x0 = 0 ∧ (∀ x < x0, f x < f x0) ∧ (∀ x > x0, f x < f x0))
∧ (f 1 = -2 * Real.exp 1) 
∧ (∀ x : ℝ, ∃ b : ℝ, f x = b ∧ b > 6 * Real.exp (-3) → ¬(f x = f 1))
:= sorry

theorem equation_root_condition (b : ℝ) : (∃ x1 x2 x3 : ℝ, f x1 = b ∧ f x2 = b ∧ f x3 = b ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) 
→ (0 < b ∧ b < 6 * Real.exp (-3))
:= sorry

end local_maximum_no_global_maximum_equation_root_condition_l342_342129


namespace S_squared_arithmetic_sequence_general_formula_for_a_sum_of_sequence_b_math_proof_problem_l342_342102

-- Condition: Definition of the sequence {a_n} and related sums {S_n}
def a (n : ℕ) : ℝ := if n = 1 then 1 else (Real.sqrt n - Real.sqrt (n - 1))
def S (n : ℕ) : ℝ := ∑ i in Finset.range n + 1, a i

-- Condition: S_n is the arithmetic mean of a_n and 1/a_n
def arithmetic_mean_condition (n : ℕ) : Prop :=
  2 * S n = a n + 1 / a n

-- Prove: {S_n^2} is an arithmetic sequence.
theorem S_squared_arithmetic_sequence (n : ℕ) : Prop :=
  S n ^ 2 = n

-- Prove: General formula for {a_n} is a_n = Real.sqrt n - Real.sqrt (n-1)
theorem general_formula_for_a (n : ℕ) : Prop :=
  a n = if n = 1 then 1 else (Real.sqrt n - Real.sqrt (n - 1))

-- Prove: Sum of the first n terms {T_n} of the sequence {b_n} is T_n = (-1)^n * Real.sqrt n
theorem sum_of_sequence_b (n : ℕ) : Prop :=
  let b (n : ℕ) := (-1 : ℝ) ^ n / a n in
  ∑ i in Finset.range n + 1, b i = (-1) ^ n * Real.sqrt n

-- Main theorem combining all proofs
theorem math_proof_problem (n : ℕ) : Prop :=
  S_squared_arithmetic_sequence n ∧ general_formula_for_a n ∧ sum_of_sequence_b n := by
    sorry

end S_squared_arithmetic_sequence_general_formula_for_a_sum_of_sequence_b_math_proof_problem_l342_342102


namespace irrational_sqrt3_l342_342881

theorem irrational_sqrt3 : 
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ (sqrt 3) = (p / q : ℝ)) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (-2 : ℝ) = (p / q : ℝ)) ∧ 
  (∃ (p q : ℤ), q ≠ 0 ∧ (0 : ℝ) = (p / q : ℝ)) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (-1 / 2 : ℝ) = (p / q : ℝ)) := 
by
  sorry

end irrational_sqrt3_l342_342881


namespace isosceles_triangle_perimeter_l342_342684

def is_isosceles_triangle (A B C : ℝ) : Prop :=
  (A = B ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (A = C ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (B = C ∧ A + B > C ∧ A + C > B ∧ B + C > A)

theorem isosceles_triangle_perimeter {A B C : ℝ} 
  (h : is_isosceles_triangle A B C) 
  (h1 : A = 3 ∨ A = 7) 
  (h2 : B = 3 ∨ B = 7) 
  (h3 : C = 3 ∨ C = 7) : 
  A + B + C = 17 := 
sorry

end isosceles_triangle_perimeter_l342_342684


namespace interval_of_increase_l342_342833

theorem interval_of_increase :
  let y := λ x : ℝ, (3 - x^2) * exp x
  ∃ I : set ℝ, I = set.Ioo (-3 : ℝ) 1 ∧ (∀ x ∈ I, deriv y x > 0) :=
by
  let y := λ x : ℝ, (3 - x^2) * exp x
  have y' : ∀ x, deriv y x = (3 - 3 * x^2) * exp x :=
    sorry
  let I := set.Ioo (-3 : ℝ) 1
  use I
  split
  · refl
  · intros x hx
    simp [y']
    sorry

end interval_of_increase_l342_342833


namespace find_sqrt_l342_342226

theorem find_sqrt (x y : ℝ) (h : sqrt (x - 1) + (3 * x + y - 1) ^ 2 = 0) :
  sqrt (5 * x + y ^ 2) = 3 :=
sorry

end find_sqrt_l342_342226


namespace map_distance_l342_342346

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l342_342346


namespace profit_relation_max_profit_at_200_l342_342497

-- Definitions of given conditions
def fixed_cost := 5
def selling_price_per_unit := 2
def additional_cost (x : ℕ) : ℝ :=
if 0 < x ∧ x < 150 then
  (1/2) * x^2 + 128 * x
else if x ≥ 150 then
  210 * x + 400000 / x - 6900
else
  0

-- Definition of profit function
def profit (x : ℕ) : ℝ :=
let total_cost := fixed_cost + additional_cost x in
let total_revenue := selling_price_per_unit * x in
total_revenue - total_cost

-- Relationship between profit y and annual production x
theorem profit_relation (x : ℕ) (hx : x > 0) :
  profit x = if x < 150 then
               -(1/2 : ℝ) * x^2 + 72 * x - 500
             else
               -10 * (x + 40000 / x) + 6400 :=
by
  sorry

-- Proof for production level that achieves maximum profit
theorem max_profit_at_200 :
  ∀x : ℕ, x > 0 → profit x ≤ profit 200 :=
by
  sorry

end profit_relation_max_profit_at_200_l342_342497


namespace solve_gas_cost_l342_342409

noncomputable def gas_cost (x : ℝ) : Prop :=
  (x / 3) - (x / 6) = 9

theorem solve_gas_cost : ∃ x : ℝ, gas_cost x ∧ x = 54 :=
by {
  existsi 54,
  split,
  {
    unfold gas_cost,
    norm_num,
  },
  {
    norm_num,
  }
}

end solve_gas_cost_l342_342409


namespace correct_option_l342_342623

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem correct_option (α : ℝ) (x1 x2 : ℝ) (hα : 2^α = Real.sqrt 2)
  (hx : 0 < x1 ∧ x1 < x2) :
  let f := power_function α in
  x2 * f x1 > x1 * f x2 :=
by
  let f := power_function α
  have h1 : α = 1 / 2, from sorry
  have h2 : f x1 = x1^α, from sorry
  have h3 : f x2 = x2^α, from sorry
  sorry

end correct_option_l342_342623


namespace interest_difference_l342_342519

theorem interest_difference (P r t : ℕ) (hP : P = 2500) (ht : t = 5) (hr : ∃ r, True) : 
  let SI_original := (P * r * t) / 100
  let SI_higher := (P * (r + 2) * t) / 100
  SI_higher - SI_original = 250 :=
by
  sorry

end interest_difference_l342_342519


namespace list_price_l342_342836

theorem list_price (P : ℝ) (h₀ : 0.83817 * P = 56.16) : P = 67 :=
sorry

end list_price_l342_342836


namespace map_length_representation_l342_342280

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l342_342280


namespace regular_polygon_sides_l342_342938

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (k : ℕ), (k : ℕ) * 18 = 360) : n = 20 :=
by
  -- Proof body here
  sorry

end regular_polygon_sides_l342_342938


namespace map_scale_representation_l342_342336

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l342_342336


namespace map_representation_l342_342287

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l342_342287


namespace sum_of_digits_7_pow_11_l342_342455

theorem sum_of_digits_7_pow_11 : 
  let n := 7 in
  let power := 11 in
  let last_two_digits := (n ^ power) % 100 in
  let tens_digit := last_two_digits / 10 in
  let ones_digit := last_two_digits % 10 in
  tens_digit + ones_digit = 7 :=
by {
  sorry
}

end sum_of_digits_7_pow_11_l342_342455


namespace association_population_after_four_years_l342_342976

theorem association_population_after_four_years :
  let a : ℕ → ℕ := λ k =>
    match k with
    | 0 => 20
    | n+1 => 4 * a n - 18
  in a 4 = 3590 :=
by
  -- Proof is omitted
  sorry

end association_population_after_four_years_l342_342976


namespace find_symmetric_point_l342_342694

structure Point := (x : Int) (y : Int)

def translate_right (p : Point) (n : Int) : Point :=
  { x := p.x + n, y := p.y }

def symmetric_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem find_symmetric_point : 
  ∀ (A B C : Point),
  A = ⟨-1, 2⟩ →
  B = translate_right A 2 →
  C = symmetric_x_axis B →
  C = ⟨1, -2⟩ :=
by
  intros A B C hA hB hC
  sorry

end find_symmetric_point_l342_342694


namespace emilia_donut_holes_count_l342_342038

noncomputable def surface_area (r : ℕ) : ℕ := 4 * r^2

def lcm (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

def donut_holes := 5103

theorem emilia_donut_holes_count :
  ∀ (S1 S2 S3 : ℕ), 
  S1 = surface_area 5 → 
  S2 = surface_area 7 → 
  S3 = surface_area 9 → 
  donut_holes = lcm S1 S2 S3 / S1 :=
by
  intros S1 S2 S3 hS1 hS2 hS3
  sorry

end emilia_donut_holes_count_l342_342038


namespace probability_x_plus_y_lt_4_l342_342926

open MeasureTheory

-- Define the vertices of the square
def square : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.1 ≤ 3 ∧ p.2 ≥ 0 ∧ p.2 ≤ 3}

-- Define the predicate x + y < 4
def condition (p : ℝ × ℝ) : Prop := p.1 + p.2 < 4

-- Define the probability measure uniform over the square
noncomputable def uniform_square : Measure (ℝ × ℝ) :=
  MeasureTheory.Measure.Uniform (Icc (0, 0) (3, 3))

-- Define the probability of the condition x + y < 4
noncomputable def prob_condition : ennreal :=
  MeasureTheory.MeasureTheory.toOuterMeasure uniform_square {p | condition p} / 
  MeasureTheory.MeasureTheory.toOuterMeasure uniform_square square

-- Statement to prove
theorem probability_x_plus_y_lt_4 : prob_condition = (7 / 9 : ℝ) :=
  sorry

end probability_x_plus_y_lt_4_l342_342926


namespace map_length_representation_l342_342279

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l342_342279


namespace sum_tens_ones_digit_of_7_pow_11_l342_342461

/--
The sum of the tens digit and the ones digit of (3+4)^{11} is 7.
-/
theorem sum_tens_ones_digit_of_7_pow_11 : 
  let number := (3 + 4)^11
  let tens_digit := (number / 10) % 10
  let ones_digit := number % 10
  tens_digit + ones_digit = 7 :=
by
  sorry

end sum_tens_ones_digit_of_7_pow_11_l342_342461


namespace magnitude_quotient_is_one_l342_342628

noncomputable def z : ℂ := 1 + complex.I
noncomputable def conjugate_z : ℂ := conj z

theorem magnitude_quotient_is_one : abs (conjugate_z / z) = 1 :=
by
  sorry

end magnitude_quotient_is_one_l342_342628


namespace find_angle_A_find_area_triangle_l342_342170

open Real

variables a b c : ℝ
variables A B C : ℝ

-- Conditions
axiom sides_of_triangle : A ∈ Ioo 0 π ∧ B ∈ Ioo 0 π ∧ C ∈ Ioo 0 π
axiom angle_sum_of_triangle : A + B + C = π
axiom opposite_sides_cos : 2 * b * cos A - sqrt 3 * c * cos A = sqrt 3 * a * cos C
axiom specific_angle_B : B = π / 6
axiom median_AM : sqrt (7) = sqrt ((a ^ 2 + a ^ 2) / 4 - ((a ^ 2 + a ^ 2) / 4) * cos (2 * π / 3))

-- Questions
theorem find_angle_A : A = π / 6 := sorry
theorem find_area_triangle : (1 / 2) * (2 ^ 2) * sin (2 * π / 3) = sqrt 3 := sorry

end find_angle_A_find_area_triangle_l342_342170


namespace map_representation_l342_342254

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l342_342254


namespace map_length_scale_l342_342300

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l342_342300


namespace map_representation_l342_342294

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l342_342294


namespace number_of_distinct_m_values_l342_342228

theorem number_of_distinct_m_values (m : ℤ) :
  (∃ x1 x2 : ℤ, x1 * x2 = 36 ∧ x1 + x2 = m) →
  set.card {m | ∃ x1 x2 : ℤ, x1 * x2 = 36 ∧ x1 + x2 = m} = 10 :=
by
  sorry

end number_of_distinct_m_values_l342_342228


namespace regular_polygon_sides_l342_342940

-- Definitions based on conditions in the problem
def exterior_angle (n : ℕ) : ℝ := 360 / n

theorem regular_polygon_sides (n : ℕ) (h : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l342_342940


namespace mark_segments_impossible_l342_342705

/-
  Prove that it is impossible to mark 50 segments with lengths 1, 2, 3, ..., 50
  such that all endpoints lie on integer points from 1 to 100 inclusive.
-/

theorem mark_segments_impossible :
  ¬(∃ (segments : list (ℕ × ℕ)), 
      (∀ i, 1 ≤ i ∧ i ≤ 50 → ∃ (a b : ℕ), (a, b) ∈ segments ∧ (b - a = i)) ∧ 
      (∀ (a b : ℕ), (a, b) ∈ segments → 1 ≤ a ∧ b ≤ 100)) :=
by
  sorry

end mark_segments_impossible_l342_342705


namespace stan_pages_l342_342376

noncomputable def words_per_minute := 50
noncomputable def words_per_page := 400
noncomputable def water_per_hour := 15
noncomputable def total_water_needed := 10

def typing_time_per_page_in_minutes := words_per_page / words_per_minute
def typing_time_per_page_in_hours := typing_time_per_page_in_minutes / 60
def water_per_page := water_per_hour * typing_time_per_page_in_hours
def pages_needed := total_water_needed / water_per_page

theorem stan_pages : pages_needed = 5 :=
by
  sorry

end stan_pages_l342_342376


namespace percentage_increase_is_9_09_l342_342478

-- Define the weekly earnings before and after the raise
def original_earnings : Real := 55
def new_earnings : Real := 60

-- Define the calculation for the percentage increase
def percentage_increase : Real :=
  ((new_earnings - original_earnings) / original_earnings) * 100

-- Theorem: The calculated percentage increase is 9.09%
theorem percentage_increase_is_9_09 :
  percentage_increase = 9.09 :=
by
  -- skipping the actual proof steps here
  sorry

end percentage_increase_is_9_09_l342_342478


namespace necessary_condition_not_sufficient_condition_l342_342903

theorem necessary_condition (α β : ℝ) (h : sin α + cos β = 0) : sin^2 α + sin^2 β = 1 :=
by sorry

theorem not_sufficient_condition (α β : ℝ) (h : sin α + cos β ≠ 0) : sin^2 α + sin^2 β = 1 → false :=
by sorry

end necessary_condition_not_sufficient_condition_l342_342903


namespace solve_trig_eq_l342_342815

theorem solve_trig_eq (x : ℝ) (k : ℤ) :
  (x = (π / 12) + 2 * k * π ∨
   x = (7 * π / 12) + 2 * k * π ∨
   x = (7 * π / 6) + 2 * k * π ∨
   x = -(5 * π / 6) + 2 * k * π) →
  (|Real.sin x| + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x)) = 2 / Real.sqrt 3 :=
sorry

end solve_trig_eq_l342_342815


namespace max_min_ghi_l342_342189

-- Define the distinct digits from 1 to 9
def isDistinctDigits (digits : List ℕ) : Prop :=
  digits.nodup ∧ (∀ d ∈ digits, 1 ≤ d ∧ d ≤ 9) ∧ digits.length = 9

-- Define the condition that the sum of digits is 45
def digitsSumTo45 (digits : List ℕ) : Prop :=
  digits.sum = 45

-- Define the addition of digits forming three-digit numbers 
def threeDigitSum (a b c d e f g h i : ℕ) : Prop :=
  let abc := 100 * a + 10 * b + c
  let def := 100 * d + 10 * e + f
  let ghi := 100 * g + 10 * h + i
  abc + def = ghi

-- Lean statement for the maximum and minimum value proof
theorem max_min_ghi {a b c d e f g h i : ℕ} :
  isDistinctDigits [a, b, c, d, e, f, g, h, i] →
  digitsSumTo45 [a, b, c, d, e, f, g, h, i] →
  threeDigitSum a b c d e f g h i →
  (ghi = 981 ∨ ghi = 459) :=
sorry

end max_min_ghi_l342_342189


namespace am_gm_inequality_application_l342_342779

theorem am_gm_inequality_application (n : ℕ) (a : Fin n → ℝ) 
  (h : (∀ i, 0 < a i) ∧ (∏ i, a i = 1)) : (∏ i, (2 + a i)) ≥ 3^n := 
by 
  sorry

end am_gm_inequality_application_l342_342779


namespace shortest_distance_pyramid_rope_l342_342200

theorem shortest_distance_pyramid_rope 
  (P A B C : Type) [MetricSpace P] 
  (PA PB PC : P) 
  (hPA : dist P PA = 2)
  (hPB : dist P PB = 2)
  (hPC : dist P PC = 2)
  (angleBPA : angle B P A = 30) 
  (angleBPC : angle B P C = 30)
  (angleCPA : angle C P A = 30) : 
  shortest_rope_distance P A B C = 2 * real.sqrt 2 := 
sorry

end shortest_distance_pyramid_rope_l342_342200


namespace find_n_l342_342759

noncomputable def f (x : ℤ) : ℤ := sorry -- f is some polynomial with integer coefficients

theorem find_n (n : ℤ) (h1 : f 1 = -1) (h4 : f 4 = 2) (h8 : f 8 = 34) (hn : f n = n^2 - 4 * n - 18) : n = 3 ∨ n = 6 :=
sorry

end find_n_l342_342759


namespace april_rainfall_correct_l342_342666

-- Define the constants for the rainfalls in March and the difference in April
def march_rainfall : ℝ := 0.81
def rain_difference : ℝ := 0.35

-- Define the expected April rainfall based on the conditions
def april_rainfall : ℝ := march_rainfall - rain_difference

-- Theorem to prove that the April rainfall is 0.46 inches
theorem april_rainfall_correct : april_rainfall = 0.46 :=
by
  -- Placeholder for the proof
  sorry

end april_rainfall_correct_l342_342666


namespace alibaba_can_enter_cave_l342_342530

theorem alibaba_can_enter_cave :
  ∃ (attempts : ℕ → fin 4 → bool → bool), ∀ (s : fin 4 → bool), ∃ k ≤ 10, 
  (∀ i, attempts k i s[i] = s[(i+2)%4]) ∨ (∀ i, attempts k i s[i] = s[(i-2)%4]) :=
sorry

end alibaba_can_enter_cave_l342_342530


namespace relative_error_comparison_l342_342030

-- Definitions for the given conditions
def error_first : ℝ := 0.05
def length_first : ℝ := 15
def error_second : ℝ := 0.25
def length_second : ℝ := 125

-- Calculate relative errors
def relative_error (error length : ℝ) : ℝ := (error / length) * 100

-- Prove the relative error of the first measurement is greater than the second
theorem relative_error_comparison :
  relative_error error_first length_first > relative_error error_second length_second :=
by {
  -- Calculation details here
  sorry
}

end relative_error_comparison_l342_342030


namespace harold_catches_up_at_12_miles_l342_342363

/-- 
Proof Problem: Given that Adrienne starts walking from X to Y at 3 miles per hour and one hour later Harold starts walking from X to Y at 4 miles per hour, prove that Harold covers 12 miles when he catches up to Adrienne.
-/
theorem harold_catches_up_at_12_miles :
  (∀ (T : ℕ), (ad_distance : ℕ) = 3 * (T + 1) → (ha_distance : ℕ) = 4 * T → ad_distance = ha_distance) →
  (∃ T : ℕ, ha_distance = 12) :=
by
  sorry

end harold_catches_up_at_12_miles_l342_342363


namespace find_p_of_five_l342_342919

-- Define the cubic polynomial and the conditions
def cubic_poly (p : ℝ → ℝ) :=
  ∀ x, ∃ a b c d, p x = a * x^3 + b * x^2 + c * x + d

def satisfies_conditions (p : ℝ → ℝ) :=
  p 1 = 1 ^ 2 ∧
  p 2 = 2 ^ 2 ∧
  p 3 = 3 ^ 2 ∧
  p 4 = 4 ^ 2

-- Theorem statement to be proved
theorem find_p_of_five (p : ℝ → ℝ) (hcubic : cubic_poly p) (hconditions : satisfies_conditions p) : p 5 = 25 :=
by
  sorry

end find_p_of_five_l342_342919


namespace sin_double_angle_l342_342708

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_l342_342708


namespace projection_segments_sum_l342_342670

theorem projection_segments_sum (O A B C A1 B1 C1 X Y : Point)
  (R : ℝ)
  (h1 : distance O A = R)
  (h2 : distance O B = R)
  (h3 : distance O C = R)
  (h4 : projection A X Y = A1)
  (h5 : projection B X Y = B1)
  (h6 : projection C X Y = C1) :
  distance X B1 = distance O A1 + distance C1 Y ∨
  distance O A1 = distance X B1 + distance C1 Y ∨
  distance C1 Y = distance X B1 + distance O A1 :=
sorry

end projection_segments_sum_l342_342670


namespace max_autos_on_ferry_l342_342000

noncomputable def ferry_capacity_pounds : ℕ := 50 * 2000  -- The ferry's capacity in pounds
noncomputable def lightest_auto_pounds : ℕ := 1600       -- The lightest possible weight for an automobile

theorem max_autos_on_ferry (ferry_capacity_pounds : ℕ) (lightest_auto_pounds : ℕ) : ℕ :=
  (ferry_capacity_pounds / lightest_auto_pounds : ℕ)

example : max_autos_on_ferry 100000 1600 = 62 := by
  simp [max_autos_on_ferry]
  norm_num
  sorry

end max_autos_on_ferry_l342_342000


namespace find_3m_plus_2n_l342_342407

-- Define the conditions of the problem
def exists_pos_integers (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ 
  log 10 (x^2) + 2 * log 10 (Nat.gcd x y) = 120 ∧ 
  log 10 (y^2) + 2 * log 10 (Nat.lcm x y) = 1140

-- Define prime factorization count
def prime_factors_count (n : ℕ) : ℕ :=
  (Nat.factorization n).values.sum

def m (x : ℕ) : ℕ :=
  prime_factors_count x

def n (y : ℕ) : ℕ :=
  prime_factors_count y

-- Main theorem to be proved
theorem find_3m_plus_2n : 
  ∃ x y : ℕ, exists_pos_integers x y ∧ 3 * m x + 2 * n y = 980 :=
by
  sorry

end find_3m_plus_2n_l342_342407


namespace percentage_of_workers_present_l342_342889

theorem percentage_of_workers_present (total_workers : ℕ) (present_workers : ℕ) (h1 : total_workers = 210) (h2 : present_workers = 198) :
    (real.ceil ((present_workers / total_workers.to_real) * 100 * 10) / 10 = 94.3) :=
by
    have h3 : (present_workers : ℝ) / (total_workers : ℝ) * 100 = 198 / 210 * 100 := by
        rw [h1, h2]
    have h4 : 198 / 210 * 100 = 94.28571428571429 := by norm_num
    rw [h3, h4]
    have h5 : real.ceil (94.28571428571429 * 10) / 10 = 94.3 := by norm_num
    exact h5

end percentage_of_workers_present_l342_342889


namespace solve_equation_l342_342818

noncomputable def equation (x : ℝ) : ℝ :=
  (|Real.sin x| + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x))

theorem solve_equation (x : ℝ) (k : ℤ) :
  (equation x = 2 / Real.sqrt 3) ↔
  (∃ k : ℤ, x = π / 12 + 2 * k * π ∨ x = 7 * π / 12 + 2 * k * π ∨ x = -5 * π / 6 + 2 * k * π) :=
sorry

end solve_equation_l342_342818


namespace isosceles_triangle_perimeter_l342_342678

theorem isosceles_triangle_perimeter
  (a b : ℕ)
  (ha : a = 3)
  (hb : b = 7)
  (h_iso : ∃ (c : ℕ), (c = a ∨ c = b) ∧ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a))) :
  a + b + b = 17 :=
by
  sorry

end isosceles_triangle_perimeter_l342_342678


namespace units_digit_of_3_pow_2020_l342_342040

theorem units_digit_of_3_pow_2020 : (3 ^ 2020) % 10 = 1 :=
by
  -- Recall the repetitive pattern of units digits
  have h1 : (3 ^ 4) % 10 = 1 := by norm_num,
  -- Calculate 2020 modulo 4 to find the position in the cycle
  have h2 : 2020 % 4 = 0 := by norm_num,
  -- Conclude the units digit of 3^2020 is the units digit of 3^4 based on the cycle
  calc (3 ^ 2020) % 10 = (3 ^ (4 * 505)) % 10 : by rw (nat.mul_div_cancel' (nat.div_mul_cancel h2))
                    ... = ((3 ^ 4) ^ 505) % 10 : by rw pow_mul
                    ... = 1 : by rw [pow_right_mod_self, h1, pow_one]
  -- Auxiliary lemmas
  where
    pow_right_mod_self : (3 ^ 4 % 10 ^ (505: ℕ)) % 10 = 1 := by
      rw [nat.pow_modₓ]
      exact h1
    sorry

end units_digit_of_3_pow_2020_l342_342040


namespace add_base6_numbers_l342_342021

def base6_to_base10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6^1 + c * 6^0

def base10_to_base6 (n : ℕ) : (ℕ × ℕ × ℕ) := 
  (n / 6^2, (n % 6^2) / 6^1, (n % 6^2) % 6^1)

theorem add_base6_numbers : 
  let n1 := 3 * 6^1 + 5 * 6^0
  let n2 := 2 * 6^1 + 5 * 6^0
  let sum := n1 + n2
  base10_to_base6 sum = (1, 0, 4) :=
by
  -- Proof steps would go here
  sorry

end add_base6_numbers_l342_342021


namespace solve_euler_totient_problem_l342_342072

noncomputable def phi (n : ℕ) : ℕ :=
  if n = 0 then 0 else ((List.range n).filter (fun k => Nat.gcd n k = 1)).length

theorem solve_euler_totient_problem :
  ∀ (a b : ℕ), 0 < a → 0 < b →
  (a + b = phi a + phi b + Nat.gcd a b) ↔
  (∃ t : ℕ, t ≥ 1 ∧ a = 2^t ∧ b = 2^t) ∨
  (∃ p : ℕ, Nat.prime p ∧ ((a = 1 ∧ b = p) ∨ (a = p ∧ b = 1))) :=
by
  sorry

end solve_euler_totient_problem_l342_342072


namespace fires_ratio_proof_l342_342034

def fires_ratio_problem (K : ℕ) : Prop :=
  let Doug := 20 in
  let Eli := K / 2 in
  (Doug + K + Eli = 110) → 
  (K : Doug) = 3 : 1

theorem fires_ratio_proof : 
  ∃ K : ℕ, fires_ratio_problem K := by
  sorry

end fires_ratio_proof_l342_342034


namespace terry_total_miles_l342_342821

def total_gasoline_used := 9 + 17
def average_gas_mileage := 30

theorem terry_total_miles (M : ℕ) : 
  total_gasoline_used * average_gas_mileage = M → M = 780 :=
by
  intro h
  rw [←h]
  sorry

end terry_total_miles_l342_342821


namespace map_representation_l342_342285

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l342_342285


namespace union_of_sets_l342_342642

theorem union_of_sets (P Q : Set ℝ) 
  (hP : P = {x | 2 ≤ x ∧ x ≤ 3}) 
  (hQ : Q = {x | x^2 ≤ 4}) : 
  P ∪ Q = {x | -2 ≤ x ∧ x ≤ 3} := 
sorry

end union_of_sets_l342_342642


namespace range_of_m_l342_342785

open Set Real

variable (A B : Set ℝ)
variable (m : ℝ)

def setA : Set ℝ := {x | x^2 + 2 * x - 8 < 0}
def setB (m : ℝ) : Set ℝ := {x | (5 - m) < x ∧ x < (2 * m - 1)}

theorem range_of_m : ∀ (m : ℝ), 
  (setA = {x | -4 < x ∧ x < 2}) →
  (A = setA) →
  (B = setB m) →
  (A ∩ (compl (setB m)) = A) →
  -∞ < m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l342_342785


namespace arccos_sin_five_l342_342993

theorem arccos_sin_five :
  arccos (sin 5) = (5 - Real.pi) / 2 :=
sorry

end arccos_sin_five_l342_342993


namespace probability_transform_in_S_l342_342933

def region_S (z : ℂ) : Prop :=
  let x := z.re;
  let y := z.im;
  -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1

def transform (z : ℂ) : ℂ :=
  (1/2 : ℝ) * (z.re - z.im) + (1/2 : ℝ) * (z.re + z.im) * complex.i

theorem probability_transform_in_S (z : ℂ) (hz : region_S z) : 
  region_S (transform z) :=
sorry

end probability_transform_in_S_l342_342933


namespace more_candidates_selected_l342_342176

theorem more_candidates_selected (total_a total_b selected_a selected_b : ℕ)
  (h1 : total_a = 8000)
  (h2 : total_b = 8000)
  (h3 : selected_a = 6 * total_a / 100)
  (h4 : selected_b = 7 * total_b / 100) :
  selected_b - selected_a = 80 :=
  sorry

end more_candidates_selected_l342_342176


namespace regular_polygon_has_20_sides_l342_342952

-- Definition of a regular polygon with an exterior angle of 18 degrees
def regular_polygon_sides (external_angle : ℝ) : ℕ :=
  if external_angle > 0 then (360 / external_angle).toInt else 0

theorem regular_polygon_has_20_sides :
  regular_polygon_sides 18 = 20 :=
by
  sorry

end regular_polygon_has_20_sides_l342_342952


namespace number_of_distinct_m_values_l342_342241

theorem number_of_distinct_m_values :
  let roots (x_1 x_2 : ℤ) := x_1 * x_2 = 36 ∧ x_2 = x_2
  let m_values := {m : ℤ | ∃ (x_1 x_2 : ℤ), x_1 * x_2 = 36 ∧ m = x_1 + x_2}
  m_values.card = 10 :=
sorry

end number_of_distinct_m_values_l342_342241


namespace map_length_represents_distance_l342_342321

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l342_342321


namespace opposite_event_is_at_least_one_hit_l342_342925

def opposite_event_of_missing_both_times (hit1 hit2 : Prop) : Prop :=
  ¬(¬hit1 ∧ ¬hit2)

theorem opposite_event_is_at_least_one_hit (hit1 hit2 : Prop) :
  opposite_event_of_missing_both_times hit1 hit2 = (hit1 ∨ hit2) :=
by
  sorry

end opposite_event_is_at_least_one_hit_l342_342925


namespace units_digit_of_sum_pow_l342_342062

-- Definition of the given variables
def a : ℕ := 5619
def b : ℕ := 2272
def n : ℕ := 124

-- The theorem we want to prove
theorem units_digit_of_sum_pow :
  let s := a + b in
  (s % 10) ^ n % 10 = 1 :=
by
  sorry

end units_digit_of_sum_pow_l342_342062


namespace polynomial_characterization_l342_342071

-- Define a polynomial with complex coefficients
def is_nonconstant_polynomial_with_unit_circle_roots 
(P : Polynomial ℂ) : Prop :=
  ¬ P.is_constant ∧
  (∀ z : ℂ, P.roots z → |z| = 1) ∧
  (∀ z : ℂ, (P - 1).roots z → |z| = 1)

-- Define the final form of the polynomial
def final_polynomial_form (P : Polynomial ℂ) : Prop :=
  ∃ (z1 : ℂ) (z2 : ℂ) (n : ℕ), 
  |z1| = 1 ∧ |z2| = 1 ∧ |z2 - 1| = 1 ∧ 
  P = z1 * X^n + z2

-- Main theorem statement
theorem polynomial_characterization 
(P : Polynomial ℂ) : 
  is_nonconstant_polynomial_with_unit_circle_roots P → 
  final_polynomial_form P := 
sorry

end polynomial_characterization_l342_342071


namespace red_ball_higher_prob_l342_342010

theorem red_ball_higher_prob (prob : ℕ → ℝ) (red_bin blue_bin : ℕ → ℝ) :
  (∀ k, prob k = 1 / (3^k)) →
  (∀ k, red_bin k = prob k) →
  (∀ k, blue_bin k = prob k) →
  let higher_prob := ∑ k, (red_bin k * blue_bin k) in
  let same_bin_prob := ∑ k, (red_bin k * blue_bin k) in
  let red_higher_prob := (1 - same_bin_prob) / 2 in
  red_higher_prob = 7 / 16 :=
by
  sorry

end red_ball_higher_prob_l342_342010


namespace problem_statement_l342_342133

noncomputable def f (x : ℝ) : ℝ := cos (2 * x - 2 * Real.pi / 3) - cos (2 * x)

theorem problem_statement (x B C a b c : ℝ) (h1 : f x = cos (2 * x - 2 * Real.pi / 3) - cos 2 * x)
  (h2 : b = 1) (h3 : c = sqrt 3) (h4 : a > b) (h5 : f (B / 2) = -sqrt 3 / 2) :
  (∃ T, T = Real.pi ∧ (∀ x, f (x + T) = f x) ∧ (∀ x, f x ≥ -sqrt 3)) ∧
  (B = Real.pi / 6) ∧ (C = Real.pi / 3) :=
by
  sorry

end problem_statement_l342_342133


namespace triangle_equality_iff_angle_l342_342601

variables {a b c : ℝ}
variables {A B C : Type*}

theorem triangle_equality_iff_angle (hABC : ∠ B = 60) :
  (1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) ↔ ∠ B = 60 :=
sorry

end triangle_equality_iff_angle_l342_342601


namespace subset_implies_value_l342_342641

theorem subset_implies_value (m : ℝ) (A : Set ℝ) (B : Set ℝ) 
  (hA : A = {-1, 3, 2*m-1}) (hB : B = {3, m}) (hSub : B ⊆ A) : 
  m = -1 ∨ m = 1 := by
  sorry

end subset_implies_value_l342_342641


namespace angle_AGD_is_120_degrees_l342_342104

variables {A B C D E F G : Type*}
variables (line : collinear_points A B C D)
variables (triangle_ABE : equilateral_triangle A B E)
variables (triangle_CDF : equilateral_triangle C D F)
variables (ω1 : circumcircle A E C)
variables (ω2 : circumcircle B F D)
variables (intersection_points : intersection_points ω1 ω2 G C)

theorem angle_AGD_is_120_degrees
    (h1 : collinear_points A B C D)
    (h2 : equilateral_triangle A B E)
    (h3 : equilateral_triangle C D F)
    (h4 : circumcircle A E C = ω1)
    (h5 : circumcircle B F D = ω2)
    (h6 : intersection_points ω1 ω2 G C) :
    angle A G D = 120 :=
sorry

end angle_AGD_is_120_degrees_l342_342104


namespace necessary_but_not_sufficient_l342_342095

theorem necessary_but_not_sufficient (a : ℝ) : 
  (a > 2 → a^2 > 2 * a) ∧ (a^2 > 2 * a → (a > 2 ∨ a < 0)) :=
by
  sorry

end necessary_but_not_sufficient_l342_342095


namespace sin_double_angle_l342_342711

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_l342_342711


namespace find_a_b_find_max_m_l342_342632

-- Define the function
def f (x : ℝ) : ℝ := abs (2 * x + 1) + abs (3 * x - 2)

-- Conditions
def solution_set_condition (x a : ℝ) : Prop := (-4 * a / 5 ≤ x ∧ x ≤ 3 * a / 5)
def eq_five_condition (x : ℝ) : Prop := f x ≤ 5

-- Prove the values of a and b
theorem find_a_b (a b : ℝ) : (∀ x : ℝ, eq_five_condition x ↔ solution_set_condition x a) → (a = 1 ∧ b = 2) :=
by
  sorry

-- Prove that |x - a| + |x + b| >= m^2 - 3m and find the maximum value of m
theorem find_max_m (a b m : ℝ) : (a = 1 ∧ b = 2) →
  (∀ x : ℝ, abs (x - a) + abs (x + b) ≥ m^2 - 3 * m) →
  m ≤ (3 + Real.sqrt 21) / 2 :=
by
  sorry


end find_a_b_find_max_m_l342_342632


namespace fine_increase_per_mph_is_correct_l342_342249

noncomputable def base_fine : ℝ := 50
noncomputable def court_costs : ℝ := 300
noncomputable def lawyer_fees_per_hour : ℝ := 80
noncomputable def lawyer_hours : ℝ := 3
noncomputable def total_owed : ℝ := 820
noncomputable def initial_speed : ℝ := 75
noncomputable def speed_limit : ℝ := 30
noncomputable def additional_penalties : ℝ := 820 - (50 + 300 + 80 * 3)

def speed_difference : ℝ := initial_speed - speed_limit
def doubled_penalties : ℝ := additional_penalties / 2
def fine_increase_per_mph : ℝ := doubled_penalties / speed_difference

theorem fine_increase_per_mph_is_correct :
  fine_increase_per_mph = 2.56 :=
by 
  -- Conditions included in the definition
  sorry

end fine_increase_per_mph_is_correct_l342_342249


namespace find_dick_jane_pair_l342_342204
open Nat

theorem find_dick_jane_pair (d n : ℕ) (prime : ℕ → Prop) (hj : Jane_age = 27) (ho : d > Jane_age) 
  (hd : 10 * a + b = Jane_age + n) (hb : 10 * b + a = d + n) 
  (h3 : 1 ≤ a) (h4 : a < b) (h5 : b ≤ 9) (ip : prime (a + b)) :
    (d, n) = (36, 7) := by
  let Jane_age := 27
  let prime := λ x => ∀ y, y | x → y = 1 ∨ y = x
  sorry

end find_dick_jane_pair_l342_342204


namespace function_decreasing_interval_l342_342830

theorem function_decreasing_interval (a : ℝ) : 
  (∀ x ∈ set.Icc (2 : ℝ) (6 : ℝ), deriv (λ x, -x^2 + 2 * a * x + 3) x ≤ 0) → a ≤ 2 :=
by
  sorry

end function_decreasing_interval_l342_342830


namespace parallel_lines_sufficient_not_necessary_l342_342606

-- Definitions of parallelism in the context of geometric lines and planes
def line := Set Point
def plane := Set Point

-- Line m, line n, and plane α are given
variables (m n : line) (α : plane)

-- Conditions given in the problem
-- condition 1: line n is a subset of plane α
def line_in_plane (line : line) (pl : plane) : Prop :=
  ∀ p, p ∈ line → p ∈ pl

-- condition 2: line m is not a subset of plane α
def not_line_in_plane (line : line) (pl : plane) : Prop :=
  ¬ (line_in_plane line pl)

-- Definition of parallelism
def parallel (l1 l2 : line) : Prop := 
  ∃ v, (∀ p1 p2, p1 ∈ l1 → p2 ∈ l2 → v ≠ 0 ∧ (p1 - p2))  -- Simplified for representation

def parallel_to_plane (line : line) (pl : plane) : Prop :=
  ∃ v, (∀ p1 p2, p1 ∈ line → p2 ∈ pl → v ≠ 0 ∧ (p1 - p2))  -- Simplified for representation

-- The proof problem statement in Lean 4
theorem parallel_lines_sufficient_not_necessary (h1 : line_in_plane n α) (h2 : not_line_in_plane m α) :
  (parallel m n → parallel_to_plane m α) ∧ ¬ (parallel_to_plane m α → parallel m n) :=
by
  sorry

end parallel_lines_sufficient_not_necessary_l342_342606


namespace map_length_representation_l342_342314

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l342_342314


namespace sin_2phi_l342_342732

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end sin_2phi_l342_342732


namespace area_of_circle_irrational_l342_342166

noncomputable def rational (r : ℚ) : Prop := true -- Define rational involving Q

noncomputable def irrational (r : ℝ) : Prop := ¬ ∃ q : ℚ, q = r -- Define irrational

theorem area_of_circle_irrational (q p : ℚ) (h₁ : 0 < q) (h₂ : 0 < p) :
  irrational (π * (q + real.sqrt p)^2) :=
by
  sorry

end area_of_circle_irrational_l342_342166


namespace exists_directed_triangle_l342_342403

structure Tournament (V : Type) :=
  (edges : V → V → Prop)
  (complete : ∀ x y, x ≠ y → edges x y ∨ edges y x)
  (outdegree_at_least_one : ∀ x, ∃ y, edges x y)

theorem exists_directed_triangle {V : Type} [Fintype V] (T : Tournament V) :
  ∃ (a b c : V), T.edges a b ∧ T.edges b c ∧ T.edges c a := by
sorry

end exists_directed_triangle_l342_342403


namespace trigonometric_identity_l342_342045

theorem trigonometric_identity :
  (Real.cos (Real.pi / 3)) - (Real.tan (Real.pi / 4)) + (3 / 4) * (Real.tan (Real.pi / 6))^2 - (Real.sin (Real.pi / 6)) + (Real.cos (Real.pi / 6))^2 = 0 :=
by
  sorry

end trigonometric_identity_l342_342045


namespace all_non_positive_l342_342108

theorem all_non_positive (n : ℕ) (a : ℕ → ℤ) 
  (h₀ : a 0 = 0) 
  (hₙ : a n = 0) 
  (ineq : ∀ k, 1 ≤ k ∧ k ≤ n - 1 → a (k - 1) - 2 * a k + a (k + 1) ≥ 0) : ∀ k, a k ≤ 0 :=
by 
  sorry

end all_non_positive_l342_342108


namespace parallel_conditions_l342_342604

open Set

-- Define the contextual environment
variables {α : Type*} [plane : Set α] {m n : Set α}

-- Definitions based on conditions
def is_line (l : Set α) : Prop := ∃ (a b : α), a ≠ b ∧ ∀ t, l t ↔ t = a ∨ t = b
def is_plane (p : Set α) : Prop := ∃ (u v w : α), u ≠ v ∧ v ≠ w ∧ w ≠ u ∧ ∀ t, p t ↔ t ∈ span {u, v, w}
def is_subset (A B : Set α) : Prop := ∀ ⦃x⦄, A x → B x
def is_parallel_to (l₁ l₂ : Set α) [is_line l₁] [is_line l₂] : Prop := ∀ a ∈ l₁, ∀ b ∈ l₂, a ≠ b ∧ (l₁ = l₂ ∨ l₁ ∩ l₂ = ∅)
def is_parallel_to_plane (l : Set α) (p : Set α) [is_line l] [is_plane p] : Prop := ∀ ⦃x y : α⦄, l x → l y → (span {x, y} ∩ p = ∅)

-- Main theorem statement
theorem parallel_conditions 
  (h₁ : is_line n) 
  (h₂ : is_line m) 
  (h₃ : is_plane α) 
  (h₄ : is_subset n α) 
  (h₅ : ¬ is_subset m α) 
  : (is_parallel_to m n → is_parallel_to_plane m α) 
  ∧ ¬ (is_parallel_to_plane m α → is_parallel_to m n) := 
sorry

end parallel_conditions_l342_342604


namespace probability_of_equal_digit_counts_l342_342489

noncomputable def probability_equal_digit_counts : ℚ :=
  let p_one_digit := (9 : ℚ) / 20
  let p_two_digit := (11 : ℚ) / 20
  let ways := nat.choose 6 3
  ways * p_one_digit^3 * p_two_digit^3

theorem probability_of_equal_digit_counts :
  (probability_equal_digit_counts = (4851495 : ℚ) / 16000000) :=
by
  -- Mathematical proof skipped
  sorry

end probability_of_equal_digit_counts_l342_342489


namespace shaded_area_correct_l342_342181

noncomputable def total_area_shaded_triangles (PQ PR : ℝ) (iterations : ℕ) : ℝ :=
let initial_area := (1 / 2) * PQ * PR in
let reduction_factor := 1 / 4 in
let first_shaded_area := initial_area * reduction_factor in
let sum_series := first_shaded_area / (1 - reduction_factor) in
if iterations = 100 then sum_series else sorry

theorem shaded_area_correct :
  total_area_shaded_triangles 10 10 100 = 16.67 :=
by sorry

end shaded_area_correct_l342_342181


namespace max_page_number_with_twenty_two_twos_l342_342366

def count_twos_in_number (n : Nat) : Nat :=
  n.toString.count('2')

def count_twos_up_to (n : Nat) : Nat :=
  (List.range (n+1)).sum count_twos_in_number

theorem max_page_number_with_twenty_two_twos : ∃ n, count_twos_up_to n = 22 ∧ ∀ m > n, count_twos_up_to m > 22 := by
  sorry

end max_page_number_with_twenty_two_twos_l342_342366


namespace volume_with_margin_l342_342056

theorem volume_with_margin
  (w l h : ℕ) (margin : ℕ)
  (w_pos : w = 5)
  (l_pos : l = 6)
  (h_pos : h = 8)
  (margin_pos : margin = 2) :
  let vol := w * l * h + 2 * (w * l + w * h + l * h) * margin + 
             8 * (2 * ⇑(real.pi)) + 36 * ⇑(real.pi) in
  vol = (2136 + 140 * real.pi) / 3 :=
by
  sorry

end volume_with_margin_l342_342056


namespace divisible_by_7_imp_coefficients_divisible_by_7_l342_342806

theorem divisible_by_7_imp_coefficients_divisible_by_7
  (a0 a1 a2 a3 a4 a5 a6 : ℤ)
  (h : ∀ x : ℤ, 7 ∣ (a6 * x^6 + a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0)) :
  7 ∣ a0 ∧ 7 ∣ a1 ∧ 7 ∣ a2 ∧ 7 ∣ a3 ∧ 7 ∣ a4 ∧ 7 ∣ a5 ∧ 7 ∣ a6 :=
sorry

end divisible_by_7_imp_coefficients_divisible_by_7_l342_342806


namespace jane_percentage_decrease_l342_342752

theorem jane_percentage_decrease
  (B H : ℝ) -- Number of bears Jane makes per week and hours she works per week
  (H' : ℝ) -- Number of hours Jane works per week with assistant
  (h1 : H' = 0.9 * H) -- New working hours when Jane works with an assistant
  (h2 : H ≠ 0) -- Ensure H is not zero to avoid division by zero
  : ((H - H') / H) * 100 = 10 := 
by calc
  ((H - H') / H) * 100
      = ((H - 0.9 * H) / H) * 100 : by rw [h1]
  ... = (0.1 * H / H) * 100 : by simp
  ... = 0.1 * 100 : by rw [div_self h2]
  ... = 10 : by norm_num

end jane_percentage_decrease_l342_342752


namespace area_triangle_FOH_l342_342418

theorem area_triangle_FOH 
(trapezoid E F G H O : Type) 
(base1 base2 : ℝ) 
(area_trap : ℝ) 
(height_trap : ℝ)
(h1 : base1 = 24) 
(h2 : base2 = 40) 
(h3 : area_trap = 384)
(h4 : height_trap = 12)
(isosceles : EFGH ≅ EFGH) 
(diagonals_bisect : ∀ {EG FH : EFGH}, EG ∩ FH = O)
: area (triangle F O H) = 96 := 
sorry

end area_triangle_FOH_l342_342418


namespace smallest_12_digit_divisible_by_36_with_all_digits_l342_342081

/-- We want to prove that the smallest 12-digit natural number that is divisible by 36 
    and contains each digit from 0 to 9 at least once is 100023457896. -/
theorem smallest_12_digit_divisible_by_36_with_all_digits :
  ∃ n : ℕ, n = 100023457896 ∧ 
    (nat.digits 10 n).length = 12 ∧ 
    (∀ d ∈ (finset.range 10).val, d ∈ (nat.digits 10 n).val) ∧ 
    n % 36 = 0 :=
begin
  sorry
end

end smallest_12_digit_divisible_by_36_with_all_digits_l342_342081


namespace asymptote_of_hyperbola_l342_342125

theorem asymptote_of_hyperbola
  (m n : ℝ)
  (h1 : ∀ x y : ℝ, (x^2 / (3 * m^2) + y^2 / (5 * n^2) = 1) → True)
  (h2 : ∀ x y : ℝ, (x^2 / (2 * m^2) - y^2 / (3 * n^2) = 1) → True)
  (h_common_focus : True) :
  ∀ x y : ℝ, (y = sqrt 3 / 4 * x ∨ y = - (sqrt 3 / 4) * x) :=
sorry

end asymptote_of_hyperbola_l342_342125


namespace mean_score_calculation_l342_342667

noncomputable def class_mean_score (total_students students_1 mean_score_1 students_2 mean_score_2 : ℕ) : ℚ :=
  ((students_1 * mean_score_1 + students_2 * mean_score_2) : ℚ) / total_students

theorem mean_score_calculation :
  class_mean_score 60 54 76 6 82 = 76.6 := 
sorry

end mean_score_calculation_l342_342667


namespace number_of_brown_dogs_l342_342669

-- Define the problem conditions and the proof statement
theorem number_of_brown_dogs (T L N LB B : ℕ)
  (ht : T = 45)
  (hl : L = 26)
  (hn : N = 8)
  (hlb : LB = 19)
  (hb : B = 30) : B = 30 :=
by {
  -- Provide a temporary proof placeholder
  sorry,
}

end number_of_brown_dogs_l342_342669


namespace map_scale_l342_342270

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l342_342270


namespace price_change_proof_l342_342962

noncomputable def price_change_percent : ℝ :=
  let P : ℝ := 1.0  -- Assume original price P = 1 for simplicity
  let final_price : ℝ := 0.75 * P
  let x : ℝ := 40.82 / 100
  let new_price : ℝ := P * (1 + x) * (1 - x) * 0.9
  x * 100

theorem price_change_proof :
  ∀ (P : ℝ) (x : ℝ),
  0 < P →
  x = Real.tonnary 40.82 (let original : ℝ := 1.0 in P = original →
  let new_price := P * (1 + x/100) * (1 - x/100) * 0.9
  new_price = 0.75 * P) :=
begin
  sorry
end

end price_change_proof_l342_342962


namespace coefficient_x2_in_binomial_expansion_l342_342074

theorem coefficient_x2_in_binomial_expansion :
  (∃ (c : ℚ), c = 15 / 4 ∧ 
    (∀ T : ℚ → ℚ, 
      let general_term := T
      in general_term = λ r, 
        (binomial 6 r) * ((1 / 2)^(6 - r)) * ((-1)^r) * x^(12 - (5 * r / 2))
      ∧ general_term 4 = c * x^2 )) :=
by
  sorry

end coefficient_x2_in_binomial_expansion_l342_342074


namespace correct_proposition_l342_342877

-- Definitions for the conditions in the problem.
-- These are geometric properties related to quadrilaterals and rectangles.
def quadrilateral (α : Type*) [metric_space α] (a b c d : α) := true
def rectangle (α : Type*) [metric_space α] (a b c d : α) := true
def diagonals_perpendicular (α : Type*) [metric_space α] (a b c d : α) := true
def diagonals_bisect_each_other (α : Type*) [metric_space α] (a b c d : α) := true
def one_right_angle (α : Type*) [metric_space α] (a b c d : α) := true
def square (α : Type*) [metric_space α] (a b c d : α) := true

-- The proof problem stating that the correct proposition is that a rectangle with diagonals 
-- perpendicular to each other is a square.
theorem correct_proposition 
  (α : Type*) [metric_space α] 
  (a b c d : α) : 
  (rectangle α a b c d) ∧ (diagonals_perpendicular α a b c d) → (square α a b c d) :=
sorry

end correct_proposition_l342_342877


namespace gcd_coprime_a_2007_value_l342_342639

-- Given sequence definition
def a : ℕ → ℕ
| 0       := 3
| (n + 1) := 2 + a 0 * a 1 * ... * a n  -- requires representation for the product sequence

-- Part 1: Prove any two terms are relatively prime
theorem gcd_coprime (i j : ℕ) (h : i ≠ j) : gcd (a i) (a j) = 1 :=
sorry

-- Part 2: Find the value of a_2007
theorem a_2007_value : a 2007 = 2 ^ (2 ^ 2007) + 1 :=
sorry

end gcd_coprime_a_2007_value_l342_342639


namespace map_length_scale_l342_342298

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l342_342298


namespace directrix_tangent_circle_l342_342662

-- Definition of the parabola and its directrix
def parabola : Prop := ∀ (x y : ℝ), y^2 = 8 * x
def directrix_of_parabola : Prop := ∀ (x : ℝ), x = -2

-- Definition of the circle
def circle (x y m : ℝ) := ∀ (x y : ℝ), x^2 + y^2 + 6 * x + m = 0

-- The theorem to prove
theorem directrix_tangent_circle (m : ℝ) :
  (∃ (c : ℝ) (x y : ℝ), directrix_of_parabola x ∧ circle x y m ∧ (circle x y m = directrix_of_parabola x)) →
  m = 8 :=
sorry

end directrix_tangent_circle_l342_342662


namespace mark_orb_speed_at_halfway_l342_342795

noncomputable def mark_winning_speed := by sorry

theorem mark_orb_speed_at_halfway :
  ∀ (initial_speed_mark initial_speed_william distance_between_walls : ℝ),
    initial_speed_mark = 1/1000 →
    initial_speed_william = 1 →
    distance_between_walls = 1 →
    mark_winning_speed initial_speed_mark initial_speed_william distance_between_walls = (2^17/125) :=
by 
  intros; 
  sorry

end mark_orb_speed_at_halfway_l342_342795


namespace probability_x_plus_y_lt_4_l342_342927

open MeasureTheory

-- Define the vertices of the square
def square : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.1 ≤ 3 ∧ p.2 ≥ 0 ∧ p.2 ≤ 3}

-- Define the predicate x + y < 4
def condition (p : ℝ × ℝ) : Prop := p.1 + p.2 < 4

-- Define the probability measure uniform over the square
noncomputable def uniform_square : Measure (ℝ × ℝ) :=
  MeasureTheory.Measure.Uniform (Icc (0, 0) (3, 3))

-- Define the probability of the condition x + y < 4
noncomputable def prob_condition : ennreal :=
  MeasureTheory.MeasureTheory.toOuterMeasure uniform_square {p | condition p} / 
  MeasureTheory.MeasureTheory.toOuterMeasure uniform_square square

-- Statement to prove
theorem probability_x_plus_y_lt_4 : prob_condition = (7 / 9 : ℝ) :=
  sorry

end probability_x_plus_y_lt_4_l342_342927


namespace complement_intersection_l342_342581

def U : Set ℝ := Set.univ
def M : Set ℝ := {y | 0 ≤ y ∧ y ≤ 2}
def N : Set ℝ := {x | (x < -3) ∨ (x > 0)}

theorem complement_intersection :
  (Set.univ \ M) ∩ N = {x | x < -3 ∨ x > 2} :=
by
  sorry

end complement_intersection_l342_342581


namespace sum_of_digits_7_pow_11_l342_342464

/-
  Let's define the problem with the given conditions and expected answer.
  We need to prove that the sum of the tens digit and the ones digit of 7^11 is 7.
-/

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_7_pow_11 : tens_digit (7 ^ 11) + ones_digit (7 ^ 11) = 7 :=
sorry

end sum_of_digits_7_pow_11_l342_342464


namespace average_speed_is_48_l342_342493

-- Definitions for the conditions
def speed_with_load : ℝ := 40
def speed_without_load : ℝ := 60
def distance_one_way (d : ℝ) : ℝ := d
def total_distance (d : ℝ) : ℝ := 2 * d
def time_with_load (d : ℝ) : ℝ := d / speed_with_load
def time_without_load (d : ℝ) : ℝ := d / speed_without_load
def total_time (d : ℝ) : ℝ := time_with_load d + time_without_load d

-- Theorem statement
theorem average_speed_is_48 (d : ℝ) : 
  2 * d / (time_with_load d + time_without_load d) = 48 := by 
    sorry

end average_speed_is_48_l342_342493


namespace sand_in_box_is_heavier_l342_342874

noncomputable def weight_of_sand_in_barrel (w_empty_barrel total_weight_barrel : ℕ) : ℕ :=
total_weight_barrel - w_empty_barrel

noncomputable def weight_of_sand_in_box (w_empty_box total_weight_box : ℕ) : ℕ :=
total_weight_box - w_empty_box

theorem sand_in_box_is_heavier (
  w_empty_barrel : ℕ := 250,
  total_weight_barrel : ℕ := 1000 + 780,
  w_empty_box : ℕ := 460,
  total_weight_box : ℕ := 2000 + 250
) : 
  let sand_in_barrel := weight_of_sand_in_barrel w_empty_barrel total_weight_barrel
  let sand_in_box := weight_of_sand_in_box w_empty_box total_weight_box
  sand_in_box = sand_in_barrel + 260 :=
by
  let sand_in_barrel := weight_of_sand_in_barrel w_empty_barrel total_weight_barrel
  let sand_in_box := weight_of_sand_in_box w_empty_box total_weight_box
  have eq1 : sand_in_barrel = 1530 := rfl
  have eq2 : sand_in_box = 1790 := rfl
  rw [eq1, eq2]
  show 1790 = 1530 + 260
  sorry

end sand_in_box_is_heavier_l342_342874


namespace least_five_digit_congruent_to_six_mod_seventeen_l342_342446

theorem least_five_digit_congruent_to_six_mod_seventeen : ∃ x : ℕ, x ≥ 10000 ∧ x < 100000 ∧ x % 17 = 6 ∧ ∀ y : ℕ, y ≥ 10000 ∧ y < 100000 ∧ y % 17 = 6 → x ≤ y :=
  by
    sorry

end least_five_digit_congruent_to_six_mod_seventeen_l342_342446


namespace least_integer_square_condition_l342_342865

theorem least_integer_square_condition (x : ℤ) (h : x^2 = 3 * x + 36) : x = -6 :=
by sorry

end least_integer_square_condition_l342_342865


namespace necessary_not_sufficient_cond_l342_342898

theorem necessary_not_sufficient_cond (α β : ℝ) :
  (sin α)^2 + (sin β)^2 = 1 → (∀ α β, (sin α + cos β = 0) → (sin α)^2 + (sin β)^2 = 1) ∧ ¬ (∀ α β, ((sin α)^2 + (sin β)^2 = 1) → (sin α + cos β = 0)) :=
by
  sorry

end necessary_not_sufficient_cond_l342_342898


namespace population_in_scientific_notation_l342_342990

theorem population_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 1370540000 = a * 10^n ∧ a = 1.37054 ∧ n = 9 :=
by
  sorry

end population_in_scientific_notation_l342_342990


namespace map_length_representation_l342_342316

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l342_342316


namespace toothbrushes_ratio_25_16_l342_342558

theorem toothbrushes_ratio_25_16 :
  ∃ (A B : ℕ), (A + B = 164) ∧ (A - B = 36) ∧ (A / Nat.gcd A B = 25) ∧ (B / Nat.gcd A B = 16) :=
begin
  sorry
end

end toothbrushes_ratio_25_16_l342_342558


namespace marina_blood_expiry_l342_342794

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : 0 < n then List.prod (List.range' 1 n) else 1

def seconds_per_day := 60 * 60 * 24

def blood_expiry_date (donation_day : ℕ) (expiry_seconds : ℕ) : ℕ :=
  donation_day + (expiry_seconds / seconds_per_day).toNat

theorem marina_blood_expiry :
  blood_expiry_date 1 (factorial 9) = 5 :=
by
  sorry

end marina_blood_expiry_l342_342794


namespace prove_rational_l342_342371

noncomputable def x_and_y_rational (x y: ℝ) : Prop :=
  ∀ (p q : ℕ), (Nat.Prime p) → (Nat.Prime q) → (p ≠ q) → (p % 2 = 1) → (q % 2 = 1) →
  x^p + y^q ∈ ℚ

theorem prove_rational (x y: ℝ)
  (h1: x_and_y_rational x y) : (x ∈ ℚ) ∧ (y ∈ ℚ) :=
sorry

end prove_rational_l342_342371


namespace find_parameters_and_monotonic_intervals_l342_342137

noncomputable def f (x : ℝ) (a ω b : ℝ) : ℝ := a * Real.sin (2 * ω * x + π / 6) + a / 6 + b

theorem find_parameters_and_monotonic_intervals (a ω b : ℝ) (ha : a > 0) (hω : ω > 0)
  (h1 : ∀ x, f x a ω b = f (x + π / ω) a ω b)
  (h2 : ∀ x, f x a ω b ≤ 7 / 4 ∧ f x a ω b ≥ 3 / 4) :
  ω = 1 ∧ a = 1 / 2 ∧ b = 1 ∧
  (∀ k : ℤ, ∀ x, k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6 → monotone (f x 1 (1/2) 1)) :=
sorry

end find_parameters_and_monotonic_intervals_l342_342137


namespace binom_1500_1_eq_1500_l342_342999

theorem binom_1500_1_eq_1500 : binom 1500 1 = 1500 :=
by
  sorry

end binom_1500_1_eq_1500_l342_342999


namespace map_distance_l342_342342

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l342_342342


namespace probability_x_plus_y_lt_4_inside_square_l342_342929

def square_area : ℝ := 9
def triangle_area : ℝ := 2
def probability : ℝ := 7 / 9

theorem probability_x_plus_y_lt_4_inside_square :
  ∀ (x y : ℝ), 
  (0 ≤ x ∧ x ≤ 3) ∧ (0 ≤ y ∧ y ≤ 3) ∧ (x + y < 4) → 
  (triangle_area = 2) ∧ (square_area = 9) ∧ (probability = 7 / 9) :=
by
  intros x y h
  sorry

end probability_x_plus_y_lt_4_inside_square_l342_342929


namespace remainder_sum_of_integers_division_l342_342776

theorem remainder_sum_of_integers_division (n S : ℕ) (hn_cond : n > 0) (hn_sq : n^2 + 12 * n - 3007 ≥ 0) (hn_square : ∃ m : ℕ, n^2 + 12 * n - 3007 = m^2):
  S = n → S % 1000 = 516 := 
sorry

end remainder_sum_of_integers_division_l342_342776


namespace total_cost_1500_farm_A_total_cost_1500_farm_B_functional_relationship_farm_A_functional_relationship_farm_B_l342_342416

theorem total_cost_1500_farm_A (x : ℕ) (hx : x = 1500) : y_A = 5900 :=
by
  -- Given conditions
  have ha1 : y_A = 1000 * 4 + (x - 1000) * 3.8,
  {
    sorry,
  },
  -- Compute result
  rw [ha1, hx],
  norm_num,

theorem total_cost_1500_farm_B (x : ℕ) (hx : x = 1500) : y_B = 6000 :=
by
  -- Given conditions
  have hb1 : y_B = x * 4,
  {
    sorry,
  },
  -- Compute result
  rw [hb1, hx],
  norm_num,

theorem functional_relationship_farm_A (x : ℕ) (hx : x > 2000) : y_A = 3.8 * x + 200 :=
by
  -- Given conditions
  have ha2 : y_A = 1000 * 4 + (x - 1000) * 3.8,
  {
    sorry,
  },
  -- Simplify
  rw [ha2],
  ring,

theorem functional_relationship_farm_B (x : ℕ) (hx : x > 2000) : y_B = 3.6 * x + 800 :=
by
  -- Given conditions
  have hb2 : y_B = 2000 * 4 + (x - 2000) * 3.6,
  {
    sorry,
  },
  -- Simplify
  rw [hb2],
  ring,

end total_cost_1500_farm_A_total_cost_1500_farm_B_functional_relationship_farm_A_functional_relationship_farm_B_l342_342416


namespace find_DE_plus_FG_l342_342032

-- Definition of the isosceles triangle ABC with given side lengths
def triangle_ABC_isosceles (A B C : Type*) [metric_space A] (dist: A → A → ℝ)
  (AB AC BC : ℝ) : Prop :=
  dist A B = AB ∧ dist A C = AC ∧ dist B C = BC 

-- The conditions on points D, E, G satisfying the given constraints
def points_on_lines (A B C D E G : Type*) [metric_space A] (dist: A → A → ℝ)
  (AD DE FG : ℝ) : Prop :=
  (∃ x y z : ℝ,
    AD = x ∧ DE = y ∧ FG = y ∧ 
    -- The perimeter conditions:
    (AD + DE + (z + y) + z) + (2 - x + y + (2 - z - y) + z) = 2 * (y + (z + y) + z) ∧
    -- Solving the equations must yield:
    y = 3 - z ∧
    x + z = 2)

-- The Lean statement for the proof problem
theorem find_DE_plus_FG (A B C D E G : Type*) [metric_space A] (dist: A → A → ℝ)
  (AB AC BC AD DE FG : ℝ)
  (h1 : triangle_ABC_isosceles A B C dist 2 2 3)
  (h2 : points_on_lines A B C D E G dist AD DE FG) :
  DE + FG = 4 :=
sorry

end find_DE_plus_FG_l342_342032


namespace similar_triangles_l342_342012

theorem similar_triangles (y : ℝ) 
  (h₁ : 12 / y = 9 / 6) : y = 8 :=
by {
  -- solution here
  -- currently, we just provide the theorem statement as requested
  sorry
}

end similar_triangles_l342_342012


namespace sin_double_angle_solution_l342_342726

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_solution_l342_342726


namespace map_representation_l342_342259

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l342_342259


namespace inequality_proof_l342_342099

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : (x^2)/(1 + x^2) + (y^2)/(1 + y^2) + (z^2)/(1 + z^2) = 2) :
  (x/(1 + x^2) + y/(1 + y^2) + z/(1 + z^2) ≤ real.sqrt 2) :=
sorry

end inequality_proof_l342_342099


namespace no_5_points_with_distances_1_to_10_l342_342556

theorem no_5_points_with_distances_1_to_10 :
  ¬ ∃ (P : Fin 5 → ℝ × ℝ × ℝ), ∀ n ∈ (Finset.range 10).map Nat.succ, 
  ∃ i j : Fin 5, i ≠ j ∧ dist (P i) (P j) = n :=
by
  sorry

end no_5_points_with_distances_1_to_10_l342_342556


namespace soft_elements_subgroup_l342_342213

variable {G : Type*} [Group G]

-- Define what it means for an element to be soft
def is_soft (x : G) : Prop :=
  ∀ (S : Set G), (Group.closure (S ∪ {x}) = ⊤ → Group.closure S = ⊤)

-- Define the set of all soft elements
def soft_elements (G : Type*) [Group G] : Set G :=
  { x | is_soft x }

-- State that the set of soft elements is a subgroup
theorem soft_elements_subgroup : is_subgroup (soft_elements G) :=
  sorry

end soft_elements_subgroup_l342_342213


namespace num_of_distinct_m_values_l342_342232

theorem num_of_distinct_m_values : 
  (∃ (x1 x2 : ℤ), x1 * x2 = 36 ∧ m = x1 + x2) → 
  (finset.card (finset.image (λ (p : ℤ × ℤ), p.1 + p.2) 
    {p | p.1 * p.2 = 36})) = 10 :=
sorry

end num_of_distinct_m_values_l342_342232


namespace smallest_n_partition_l342_342660

theorem smallest_n_partition (A : Finset ℕ) (A_is : ∀ i : Fin 63, ∃ A_i : Finset ℕ, A_i ⊆ A ∧ A_i ≠ ∅ ∧ pairwise_disjoint A_is ∧ A_i (1 ≤ i ≤ 63)):
  ∃ n : ℕ, n >= 2016 ∧ (∀ i : Fin 63, ∃ x y : ℕ, x ∈ A_i ∧ y ∈ A_i ∧ x > y ∧ 31 * x ≤ 32 * y) :=
begin
  sorry
end

lemma pairwise_disjoint {α : Type*} [DecidableEq α] (s : Finset (Finset α)) : 
  ∀ (i j : Fin (s.card)), (i ≠ j ∧ s[i] ∩ s[j] = ∅) :=
sorry

end smallest_n_partition_l342_342660


namespace distance_x_intercepts_l342_342888

theorem distance_x_intercepts {P : Type*} [MetricSpace P] (point : P) (slope1 slope2 x1 x2 y : ℝ) :
  (slope1 = 2) →
  (slope2 = 6) →
  (point = (40 : ℝ, 30 : ℝ)) →
  (line1 : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), p.2 - 30 = 2 * (p.1 - 40)) →
  (line2 : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), p.2 - 30 = 6 * (p.1 - 40)) →
  (intercept1 : ℝ × ℝ := (25, 0)) →
  (intercept2 : ℝ × ℝ := (35, 0)) →
  (dist : ℝ := 10) →
  dist = (intercept2.1 - intercept1.1) →
  dist = 10 :=
by
  intros
  sorry

end distance_x_intercepts_l342_342888


namespace correct_order_steps_l342_342017

theorem correct_order_steps :
  ∃ order: list (ℕ → ℕ),
  order = [2, 4, 1, 3] ∧
  (∀ step: ℕ → ℕ,
    step = 2 → "Collect the math scores of each student who moved from seventh grade to eighth grade" ∧
    step = 4 → "Organize the relevant data from the quality tests conducted in the eighth grade" ∧
    step = 1 → "Draw a line chart to represent the changes in scores" ∧
    step = 3 → "Analyze the changes in scores from the line chart") :=
sorry

end correct_order_steps_l342_342017


namespace find_b_l342_342787

noncomputable def f (b x : ℝ) : ℝ :=
if x < 1 then 2 * x - b else 2 ^ x

theorem find_b (b : ℝ) (h : f b (f b (1 / 2)) = 4) : b = -1 :=
sorry

end find_b_l342_342787


namespace third_median_length_l342_342675

theorem third_median_length (m1 m2 area : ℝ) (h1 : m1 = 5) (h2 : m2 = 10) (h3 : area = 10 * Real.sqrt 10) : 
  ∃ m3 : ℝ, m3 = 3 * Real.sqrt 10 :=
by
  sorry

end third_median_length_l342_342675


namespace find_values_of_a_l342_342567

noncomputable def has_one_real_solution (a : ℝ) : Prop :=
  ∃ x: ℝ, (x^3 - a*x^2 - 3*a*x + a^2 - 1 = 0) ∧ (∀ y: ℝ, (y^3 - a*y^2 - 3*a*y + a^2 - 1 = 0) → y = x)

theorem find_values_of_a : ∀ a: ℝ, has_one_real_solution a ↔ a < -(5 / 4) :=
by
  sorry

end find_values_of_a_l342_342567


namespace right_triangle_area_l342_342190

theorem right_triangle_area (z : ℂ) (hz : z ≠ 0) (hz_angle : ∠(z, z^2, z^4) = real.pi / 2) :
  real.abs (z) = 1 → area (z, z^2, z^4) = 1 / 2 :=
by sorry

noncomputable def area (a b c : ℂ) : ℝ :=
  1 / 2 * real.abs ((b - a) * complex.conj (b - a) * complex.abs (c - b))

end right_triangle_area_l342_342190


namespace percentage_of_girls_taking_lunch_l342_342175

theorem percentage_of_girls_taking_lunch 
  (total_students : ℕ)
  (boys_ratio girls_ratio : ℕ)
  (boys_to_girls_ratio : boys_ratio + girls_ratio = 10)
  (boys : ℕ)
  (girls : ℕ)
  (boys_calc : boys = (boys_ratio * total_students) / 10)
  (girls_calc : girls = (girls_ratio * total_students) / 10)
  (boys_lunch_percentage : ℕ)
  (boys_lunch : ℕ)
  (boys_lunch_calc : boys_lunch = (boys_lunch_percentage * boys) / 100)
  (total_lunch_percentage : ℕ)
  (total_lunch : ℕ)
  (total_lunch_calc : total_lunch = (total_lunch_percentage * total_students) / 100)
  (girls_lunch : ℕ)
  (girls_lunch_calc : girls_lunch = total_lunch - boys_lunch) :
  ((girls_lunch * 100) / girls) = 40 :=
by 
  -- The proof can be filled in here
  sorry

end percentage_of_girls_taking_lunch_l342_342175


namespace find_standard_equation_and_prove_q_fixed_l342_342622

namespace EllipseProblem

-- Define the ellipse passing through point M
def ellipse (a : ℝ) (x y : ℝ) : Prop := (x^2)/(a^2) + (y^2)/(a^2 - 7) = 1

-- Define the ellipse with the specific value of a^2
def standard_ellipse (x y : ℝ) : Prop := (x^2)/16 + (y^2)/9 = 1

-- The point M
def M : ℝ × ℝ := (-2, (3*Real.sqrt 3)/2)

-- The point N
def N : ℝ × ℝ := (0, 6)

-- The fixed line y = 3/2
def fixed_line (x y : ℝ) : Prop := y = 3/2

theorem find_standard_equation_and_prove_q_fixed :
  (∃ a : ℝ, ellipse a M.1 M.2) →
  standard_ellipse M.1 M.2 →
  ∃ A B C D Q : ℝ × ℝ,
    ((Q.2 = (N.2 + 3) / 2) ∧ fixed_line Q.1 Q.2) :=
by
  sorry

end find_standard_equation_and_prove_q_fixed_l342_342622


namespace map_length_represents_distance_l342_342320

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l342_342320


namespace remainder_of_S_div_1000_l342_342773

theorem remainder_of_S_div_1000 :
  let S := (Finset.filter (λ n : ℕ, ∃ m : ℕ, n^2 + 12 * n - 3007 = m^2)
    (Finset.range 2000)).sum in
  (S % 1000) = 566 := by
  sorry

end remainder_of_S_div_1000_l342_342773


namespace bisect_diagonal_BD_l342_342894

-- Define a convex quadrilateral ABCD
variable (A B C D M N : Type) [convex_quadrilateral ABCD]
variable [longest_side (AB : Type)]

-- Define points M on AB and N on BC where AN and CM divide quadrilateral into equal areas
variable (M_on_AB : is_on_side M AB)
variable (N_on_BC : is_on_side N BC)
variable (AN_divides_area : divides_area_equal AN ABCD)
variable (CM_divides_area : divides_area_equal CM ABCD)

-- Property to be proved that MN bisects BD
theorem bisect_diagonal_BD : bisects_segment MN BD :=
sorry

end bisect_diagonal_BD_l342_342894


namespace cupcakes_total_l342_342088

theorem cupcakes_total (initially_made : ℕ) (sold : ℕ) (newly_made : ℕ) (initially_made_eq : initially_made = 42) (sold_eq : sold = 22) (newly_made_eq : newly_made = 39) : initially_made - sold + newly_made = 59 :=
by
  sorry

end cupcakes_total_l342_342088


namespace integral_P_zero_l342_342059

open polynomial

noncomputable def T_k (n k : ℕ) (x : ℝ) : ℝ :=
∏ i in finset.range (n + 1), if i = k then 1 else x - i

noncomputable def P (n : ℕ) : polynomial ℝ :=
∑ k in finset.range (n + 1), polynomial.C (T_k n k k)

theorem integral_P_zero (n : ℕ) (s t : ℕ) (h1 : 1 ≤ s) (h2 : s ≤ n) (h3 : 1 ≤ t) (h4 : t ≤ n) : 
  ∫ x in s.to_real..t.to_real, (P n).eval x = 0 :=
by sorry

end integral_P_zero_l342_342059


namespace rationalize_denominator_l342_342810

theorem rationalize_denominator : 
  let a := 32
  let b := 8
  let c := 2
  let d := 4
  (a / (c * Real.sqrt c) + b / (d * Real.sqrt c)) = (9 * Real.sqrt c) :=
by
  sorry

end rationalize_denominator_l342_342810


namespace ribbon_initial_amount_l342_342207

theorem ribbon_initial_amount (x : ℕ) (gift_count : ℕ) (ribbon_per_gift : ℕ) (ribbon_left : ℕ)
  (H1 : ribbon_per_gift = 2) (H2 : gift_count = 6) (H3 : ribbon_left = 6)
  (H4 : x = gift_count * ribbon_per_gift + ribbon_left) : x = 18 :=
by
  rw [H1, H2, H3] at H4
  exact H4

end ribbon_initial_amount_l342_342207


namespace problem_statement_l342_342576

def greatest_integer_not_exceeding (a : ℝ) : ℤ :=
  (Real.floor a : ℤ)

def question (n : ℤ) : ℤ :=
  greatest_integer_not_exceeding (Real.sqrt n)

def is_divisor (d n : ℤ) : Prop :=
  ∃ k : ℤ, n = d * k

def number_of_satisfying_integers : ℤ :=
  300

theorem problem_statement : 
  ∃ (count : ℤ), 
  (count = number_of_satisfying_integers) ∧ 
  (∀ n : ℤ, 1 ≤ n ∧ n ≤ 10000 → is_divisor (question n) n) :=
begin
  sorry
end

end problem_statement_l342_342576


namespace v2_correct_at_2_l342_342132

def poly (x : ℕ) : ℕ := x^5 + x^4 + 2 * x^3 + 3 * x^2 + 4 * x + 1

def horner_v2 (x : ℕ) : ℕ :=
  let v0 := 1
  let v1 := v0 * x + 4
  let v2 := v1 * x + 3
  v2

theorem v2_correct_at_2 : horner_v2 2 = 15 := by
  sorry

end v2_correct_at_2_l342_342132


namespace find_angle_C_l342_342664

theorem find_angle_C (A B C : ℝ) (h1 : A = 88) (h2 : B - C = 20) (angle_sum : A + B + C = 180) : C = 36 :=
by
  sorry

end find_angle_C_l342_342664


namespace binom_1500_1_l342_342995

-- Define binomial coefficient
def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

-- Theorem statement
theorem binom_1500_1 : binom 1500 1 = 1500 :=
by
  sorry

end binom_1500_1_l342_342995


namespace map_scale_l342_342357

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l342_342357


namespace map_representation_l342_342292

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l342_342292


namespace maximum_matches_l342_342859

theorem maximum_matches (A B C : ℕ) (h1 : A > B) (h2 : B > C) 
    (h3 : A ≥ B + 10) (h4 : B ≥ C + 10) (h5 : B + C > A) : 
    A + B + C - 1 ≤ 62 :=
sorry

end maximum_matches_l342_342859


namespace cost_of_each_square_is_24_l342_342515

-- Define the dimensions of the floor
def length_floor : ℝ := 24
def width_floor : ℝ := 64

-- Define the dimensions of each carpet square
def length_square : ℝ := 8
def width_square : ℝ := 8

-- Total cost to cover the floor with carpet squares
def total_cost : ℝ := 576

-- Calculate the area of the floor and each carpet square
def area_floor : ℝ := length_floor * width_floor
def area_square : ℝ := length_square * width_square

-- Calculate the number of carpet squares needed
def number_of_squares : ℝ := area_floor / area_square

-- Calculate the cost per carpet square
def cost_per_square : ℝ := total_cost / number_of_squares

-- The theorem that the cost of each carpet square is $24
theorem cost_of_each_square_is_24 : cost_per_square = 24 := by sorry

end cost_of_each_square_is_24_l342_342515


namespace maximum_value_inequality_l342_342784

theorem maximum_value_inequality (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 6) :
  sqrt (3 * x + 2) + sqrt (3 * y + 2) + sqrt (3 * z + 2) ≤ 3 * sqrt 20 :=
sorry

end maximum_value_inequality_l342_342784


namespace annual_return_l342_342983

theorem annual_return (initial_price profit : ℝ) (h₁ : initial_price = 5000) (h₂ : profit = 400) : 
  ((profit / initial_price) * 100 = 8) := by
  -- Lean's substitute for proof
  sorry

end annual_return_l342_342983


namespace ratio_of_horns_l342_342989

def charlie_flutes := 1
def charlie_horns := 2
def charlie_harps := 1

def carli_flutes := 2 * charlie_flutes
def carli_harps := 0

def total_instruments := 7

def charlie_instruments := charlie_flutes + charlie_horns + charlie_harps
def carli_instruments := total_instruments - charlie_instruments

def carli_horns := carli_instruments - carli_flutes

theorem ratio_of_horns : (carli_horns : ℚ) / charlie_horns = 1 / 2 := by
  sorry

end ratio_of_horns_l342_342989


namespace algebraic_expression_value_l342_342583

theorem algebraic_expression_value (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 5) : a^2 - b^2 = 15 :=
by
  sorry

end algebraic_expression_value_l342_342583


namespace proof_addition_and_rounding_l342_342023

noncomputable def round_nearest_tenth (x : ℝ) : ℝ :=
  Float.ofInt (Int.round (x * 10)) / 10

theorem proof_addition_and_rounding :
  (round_nearest_tenth (47.32 + 28.659) = 76.0) :=
by
  sorry

end proof_addition_and_rounding_l342_342023


namespace expression_evaluation_l342_342046

theorem expression_evaluation : 
  (3^2 + 3^1 + 3^0) / (3^(-1) + 3^(-2) + 3^(-3)) = 27 := 
by 
  sorry

end expression_evaluation_l342_342046


namespace cosine_of_angle_BHD_l342_342180

noncomputable def cosine_angle_BHD : ℝ :=
let CD := 2 in
let AB := 3 in
let AD := 1 in
let angle_DHG := 30 in -- 30 degrees
let angle_FHB := 45 in -- 45 degrees
  (11 * Real.sqrt 6) / 48

theorem cosine_of_angle_BHD :
  let CD := 2 in
  let AB := 3 in
  let AD := 1 in
  let angle_DHG := 30 in -- 30 degrees
  let angle_FHB := 45 in -- 45 degrees
  cos (angle BHD) = (11 * Real.sqrt 6) / 48 :=
by
  sorry

end cosine_of_angle_BHD_l342_342180


namespace problem_statement_l342_342127

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x

-- Conditions and conclusions
theorem problem_statement :
  (∃ x, is_local_max f x ∧ ∀ y, f y < f x) ∧
  (∀ b, (∀ x, f x = b → ∃! x (h : f x = b), (∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3)) → 0 < b ∧ b < 6 * Real.exp (-3)) :=
by
  sorry

end problem_statement_l342_342127


namespace sin_2phi_l342_342728

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end sin_2phi_l342_342728


namespace eventually_monotonic_of_has_long_monotonic_segment_l342_342107

noncomputable def has_long_monotonic_segment (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, ∃ i : ℕ, (∀ j : ℕ, j < k + 1 → a (i + j) < a (i + j + 1)) ∨
                    (∀ j : ℕ, j < k + 1 → a (i + j) > a (i + j + 1))

theorem eventually_monotonic_of_has_long_monotonic_segment
  (a : ℕ → ℝ) (h : function.injective a) (h_mono : has_long_monotonic_segment a) :
  ∃ N : ℕ, ∀ m n : ℕ, N ≤ m → N ≤ n → (a m ≤ a n ∨ a n ≤ a m) :=
sorry

end eventually_monotonic_of_has_long_monotonic_segment_l342_342107


namespace units_digit_fraction_l342_342870

-- Given conditions
def numerator : ℕ := 30 * 31 * 32 * 33 * 34 * 35
def denominator : ℕ := 1500
def simplified_fraction : ℕ := 2^5 * 3 * 31 * 33 * 17 * 7

-- Statement of the proof goal
theorem units_digit_fraction :
  (simplified_fraction) % 10 = 2 := by
  sorry

end units_digit_fraction_l342_342870


namespace solution_interval_l342_342398

def f (x : ℝ) : ℝ := 2^(x-1) + x - 5

theorem solution_interval : ∃ (c : ℝ), c ∈ Ioo 2 3 ∧ f c = 0 := by
  sorry

end solution_interval_l342_342398


namespace least_five_digit_congruent_to_6_mod_17_l342_342440

theorem least_five_digit_congruent_to_6_mod_17 :
  ∃ (x : ℕ), 10000 ≤ x ∧ x ≤ 99999 ∧ x % 17 = 6 ∧
  ∀ (y : ℕ), 10000 ≤ y ∧ y ≤ 99999 ∧ y % 17 = 6 → x ≤ y := 
sorry

end least_five_digit_congruent_to_6_mod_17_l342_342440


namespace coordinates_change_l342_342846

variable (e1 e2 e3 : ℝ → ℝ → ℝ)
variable (x : ℝ × ℝ × ℝ := (1, 2, 3))

-- Definitions for the new basis vectors
def e1' : ℝ × ℝ × ℝ := (e1 1 0 + 2 * e3 0 0)
def e2' : ℝ × ℝ × ℝ := (e2 0 1 + e3 0 0)
def e3' : ℝ × ℝ × ℝ := (-e1 1 0 - e2 0 1 - 2 * e3 0 0)

-- Definitions for the original basis vectors
def e1 : ℝ × ℝ × ℝ := (1, 0, 0)
def e2 : ℝ × ℝ × ℝ := (0, 1, 0)
def e3 : ℝ × ℝ × ℝ := (0, 0, 1)

-- Transition matrix and its inverse
def C : matrix (fin 3) (fin 3) ℝ :=
  ![![1, 0, -1], ![0, 1, -1], ![2, 1, -2]]

def C_inv : matrix (fin 3) (fin 3) ℝ :=
  ![![-1, -1, 1], ![-2, 0, 1], ![-2, -1, 1]]

-- Proof that the coordinates in the new basis are [0, 1, -1]
theorem coordinates_change (x : ℝ × ℝ × ℝ) :
  let x_new := vecMul C_inv x in
  x_new = (0, 1, -1) := by
    sorry

end coordinates_change_l342_342846


namespace map_representation_l342_342289

theorem map_representation (d1 d2 l1 l2 : ℕ)
  (h1 : l1 = 15)
  (h2 : d1 = 90)
  (h3 : l2 = 20) :
  d2 = 120 :=
by
  sorry

end map_representation_l342_342289


namespace least_five_digit_congruent_6_mod_17_l342_342444

theorem least_five_digit_congruent_6_mod_17 : ∃ n: ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 6 ∧ ∀ m: ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 17 = 6 → n ≤ m :=
sorry

end least_five_digit_congruent_6_mod_17_l342_342444


namespace minimum_vertical_distance_l342_342832

noncomputable def absolute_value (x : ℝ) : ℝ := abs x

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 - 3 * x - 5

theorem minimum_vertical_distance :
  ∃ x : ℝ, (∀ y : ℝ, |absolute_value y - quadratic_function y| ≥ 4) ∧ (|absolute_value x - quadratic_function x| = 4) := 
sorry

end minimum_vertical_distance_l342_342832


namespace find_f2_l342_342096

def f (x : ℝ) (a b : ℝ) := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 3) : f 2 a b = -19 :=
by sorry

end find_f2_l342_342096


namespace largest_number_of_acute_angles_in_convex_hexagon_l342_342437

theorem largest_number_of_acute_angles_in_convex_hexagon :
  ∀ (hexagon : Fin 6 → ℝ),
    (∑ i, hexagon i = 720) →
    (∀ i, 0 < hexagon i ∧ hexagon i < 180) →
    (∀ i, if hexagon i < 90 then true else false) ≤ 3 :=
sorry

end largest_number_of_acute_angles_in_convex_hexagon_l342_342437


namespace find_a_b_and_min_g_l342_342130

-- Define the function f as described in the problem statement
def f (a b : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2 - b * x - 1

-- Define the derivative g of the function f
def g (a b : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a * x - b

-- Define the condition for the tangent line
def tangent_line_condition (a b : ℝ) : Prop :=
  let f' := g a b in
  f' 1 = Real.exp 1 - Real.exp 1 + 1 - b = Real.exp 1 - 1 ∧
  f a b 1 = Real.exp 1 - a - b - 1 = Real.exp 1 - 2

-- Define the problem statement
theorem find_a_b_and_min_g : 
  ∃ (a b : ℝ),
  tangent_line_condition a b ∧
  (if a ≤ 1/2 then ∀ x ∈ Set.Icc 0 1, g a b x ≥ 1 - b else
  if a ≥ Real.exp 1 / 2 then ∀ x ∈ Set.Icc 0 1, g a b x ≥ 1 - 2 * a - b else
  ∀ x ∈ Set.Icc 0 1, g a b x ≥ 2 * a - 2 * a * Real.log (2 * a) - b) :=
  sorry

end find_a_b_and_min_g_l342_342130


namespace map_length_scale_l342_342297

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l342_342297


namespace map_length_scale_l342_342295

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l342_342295


namespace course_choice_related_to_gender_l342_342414

def contingency_table (a b c d n : ℕ) : Prop :=
  n = a + b + c + d ∧
  a + b = 50 ∧
  c + d = 50 ∧
  a + c = 70 ∧
  b + d = 30

def chi_square_test (a b c d n : ℕ) : ℕ := 
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem course_choice_related_to_gender (a b c d n : ℕ) :
  contingency_table 40 10 30 20 100 →
  chi_square_test 40 10 30 20 100 > 3.841 :=
by
  intros h_table
  sorry

end course_choice_related_to_gender_l342_342414


namespace sum_of_digits_7_pow_11_l342_342457

theorem sum_of_digits_7_pow_11 : 
  let n := 7 in
  let power := 11 in
  let last_two_digits := (n ^ power) % 100 in
  let tens_digit := last_two_digits / 10 in
  let ones_digit := last_two_digits % 10 in
  tens_digit + ones_digit = 7 :=
by {
  sorry
}

end sum_of_digits_7_pow_11_l342_342457


namespace no_unique_sums_l342_342704

theorem no_unique_sums (n : ℕ) : ¬ (∃ (table : matrix (fin n) (fin n) ℤ),
  (∀ i, ∑ j, table i j ∈ {-n, -n+1, ..., n} ∧ ∑ j, table i j ≠ ∑ j, table (i+1 % n) j) ∧
  (∀ j, ∑ i, table i j ∈ {-n, -n+1, ..., n} ∧ ∑ i, table i j ≠ ∑ i, table i (j+1 % n)) ∧
  (∑ i in finset.range n, table i i ∈ {-n, -n+1, ..., n} ∧ ∑ i in finset.range n, table i i ≠ ∑ i in finset.range n, table (i+1 % n) (i+1 % n)) ∧
  (∑ i in finset.range n, table i (n-1-i) ∈ {-n, -n+1, ..., n} ∧ ∑ i in finset.range n, table i (n-1-i) ≠ ∑ i in finset.range n, table (i+1 % n) (n-1-(i+1 % n))))
: sorry

end no_unique_sums_l342_342704


namespace map_length_represents_distance_l342_342324

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l342_342324


namespace tan_ratio_sum_l342_342155

theorem tan_ratio_sum (x y : ℝ) (h1 : (sin x / cos y) + (sin y / cos x) = 2)
  (h2 : (cos x / sin y) + (cos y / sin x) = 5) :
  (tan x / tan y) + (tan y / tan x) = 6 / 5 :=
by
  sorry

end tan_ratio_sum_l342_342155


namespace validate_option_B_l342_342876

theorem validate_option_B (a b : ℝ) : 
  (2 * a + 3 * a^2 ≠ 5 * a^3) ∧ 
  ((-a^3)^2 = a^6) ∧ 
  (¬ (-4 * a^3 * b / (2 * a) = -2 * a^2)) ∧ 
  ((5 * a * b)^2 ≠ 10 * a^2 * b^2) := 
by
  sorry

end validate_option_B_l342_342876


namespace find_a_l342_342147

open Set

theorem find_a (a : ℝ) :
  let U := {3, 7, a^2 - 2*a - 3}
  let A := {7, abs (a - 7)}
  let complement_u_a := U \ A
  (complement_u_a = {5}) → a = 4 :=
by
  sorry

end find_a_l342_342147


namespace tea_factory_allocation_l342_342549
theorem tea_factory_allocation:
  (number of different allocation schemes for 6 workers across three processes with given constraints) = 360.

end tea_factory_allocation_l342_342549


namespace original_length_equals_13_l342_342747

-- Definitions based on conditions
def original_width := 18
def increased_length (x : ℕ) := x + 2
def increased_width := 20

-- Total area condition
def total_area (x : ℕ) := 
  4 * ((increased_length x) * increased_width) + 2 * ((increased_length x) * increased_width)

theorem original_length_equals_13 (x : ℕ) (h : total_area x = 1800) : x = 13 := 
by
  sorry

end original_length_equals_13_l342_342747


namespace best_player_total_hits_l342_342913

theorem best_player_total_hits
  (team_avg_hits_per_game : ℕ)
  (games_played : ℕ)
  (total_players : ℕ)
  (other_players_avg_hits_next_6_games : ℕ)
  (correct_answer : ℕ)
  (h1 : team_avg_hits_per_game = 15)
  (h2 : games_played = 5)
  (h3 : total_players = 11)
  (h4 : other_players_avg_hits_next_6_games = 6)
  (h5 : correct_answer = 25) :
  ∃ total_hits_of_best_player : ℕ,
  total_hits_of_best_player = correct_answer := by
  sorry

end best_player_total_hits_l342_342913


namespace subset_implies_value_l342_342214

theorem subset_implies_value (a : ℝ) : (∀ x ∈ ({0, -a} : Set ℝ), x ∈ ({1, -1, 2 * a - 2} : Set ℝ)) → a = 1 := by
  sorry

end subset_implies_value_l342_342214


namespace blake_bought_six_chocolate_packs_l342_342541

-- Defining the conditions as hypotheses
variables (lollipops : ℕ) (lollipopCost : ℕ) (packCost : ℕ)
          (cashGiven : ℕ) (changeReceived : ℕ)
          (totalSpent : ℕ) (totalLollipopCost : ℕ) (amountSpentOnChocolates : ℕ)

-- Assertion of the values based on the conditions
axiom h1 : lollipops = 4
axiom h2 : lollipopCost = 2
axiom h3 : packCost = lollipops * lollipopCost
axiom h4 : cashGiven = 6 * 10
axiom h5 : changeReceived = 4
axiom h6 : totalSpent = cashGiven - changeReceived
axiom h7 : totalLollipopCost = lollipops * lollipopCost
axiom h8 : amountSpentOnChocolates = totalSpent - totalLollipopCost
axiom chocolatePacks : ℕ
axiom h9 : chocolatePacks = amountSpentOnChocolates / packCost

-- The statement to be proved
theorem blake_bought_six_chocolate_packs :
    chocolatePacks = 6 :=
by
  subst_vars
  sorry

end blake_bought_six_chocolate_packs_l342_342541


namespace limit_of_n_sum_div_R_pow_3_2_l342_342760

noncomputable def n (R : ℕ) : ℕ :=
  {p : ℤ × ℤ × ℤ | 2 * p.1 * p.1 + 3 * p.2 * p.2 + 5 * p.3 * p.3 = R}.to_finset.card

theorem limit_of_n_sum_div_R_pow_3_2 :
  (Real.limit (fun R : ℕ => (n 1 + n 2 + ⋯ + n R) / R ^ (3 / 2)) (Filter.at_top) : ℝ) = 4 * Real.pi / (3 * Real.sqrt 30) :=
sorry

end limit_of_n_sum_div_R_pow_3_2_l342_342760


namespace girls_in_club_l342_342517

/-
A soccer club has 30 members. For a recent team meeting, only 18 members could attend:
one-third of the girls attended but all of the boys attended. Prove that the number of 
girls in the soccer club is 18.
-/

variables (B G : ℕ)

-- Conditions
def total_members (B G : ℕ) := B + G = 30
def meeting_attendance (B G : ℕ) := (1/3 : ℚ) * G + B = 18

theorem girls_in_club (B G : ℕ) (h1 : total_members B G) (h2 : meeting_attendance B G) : G = 18 :=
  sorry

end girls_in_club_l342_342517


namespace g_monotonically_decreasing_on_interval_l342_342136

noncomputable def f (x : ℝ) : ℝ := sqrt 2 * sin (2 * x - π / 4)
noncomputable def g (x : ℝ) : ℝ := sqrt 2 * sin (1 / 2 * x - π / 12)

theorem g_monotonically_decreasing_on_interval : 
  ∀ x, x ∈ (Set.Icc (7 * π / 6) (19 * π / 6)) →
    ∀ y, y ∈ (Set.Icc (7 * π / 6) (19 * π / 6)) → 
      x < y → g x > g y :=
sorry

end g_monotonically_decreasing_on_interval_l342_342136


namespace parallel_vector_lambda_l342_342151

theorem parallel_vector_lambda (lambda : ℝ) :
  let a := (1 : ℝ, 2 : ℝ)
  let b := (2 : ℝ, -2 : ℝ)
  let c := (1 : ℝ, lambda)
  let d := (2 * a.1 + b.1, 2 * a.2 + b.2)
  (c.1 / d.1 = c.2 / d.2) → lambda = 1 / 2 :=
by
  let a := (1 : ℝ, 2 : ℝ)
  let b := (2 : ℝ, -2 : ℝ)
  let c := (1 : ℝ, lambda)
  let d := (2 * a.1 + b.1, 2 * a.2 + b.2)
  let parallel_condition := (c.1 / d.1 = c.2 / d.2)
  sorry

end parallel_vector_lambda_l342_342151


namespace four_lines_circumcircles_intersect_at_common_point_l342_342474

theorem four_lines_circumcircles_intersect_at_common_point 
(l1 l2 l3 l4 : line) 
(h1 : ∀ (l : line), l ∉ {l1, l2, l3, l4} → (l ∥ l1)) 
(h2 : ∀ (p : point), (∃ l ∈ {l1, l2, l3, l4}, p ∈ l)) 
(h3 : ∀ (l : line), l ∈ {l1, l2, l3, l4} → ∀ (m : line), m ∈ {l1, l2, l3, l4} → (l ≠ m → ¬(l ∥ m))) : 
∃ P, (P ∈ circumscribed_circle (intersection_points l1 l2 l3) ∧
      P ∈ circumscribed_circle (intersection_points l1 l2 l4) ∧
      P ∈ circumscribed_circle (intersection_points l1 l3 l4) ∧
      P ∈ circumscribed_circle (intersection_points l2 l3 l4)) :=
sorry

end four_lines_circumcircles_intersect_at_common_point_l342_342474


namespace sum_binom_mod_100_l342_342764

theorem sum_binom_mod_100 : 
  (∑ n in Finset.range 433, (-1 : ℤ) ^ n * Nat.choose 1500 (3 * n) % 100) = 66 := 
sorry

end sum_binom_mod_100_l342_342764


namespace map_scale_l342_342267

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l342_342267


namespace binom_1500_1_eq_1500_l342_342997

theorem binom_1500_1_eq_1500 : binom 1500 1 = 1500 :=
by
  sorry

end binom_1500_1_eq_1500_l342_342997


namespace map_scale_representation_l342_342328

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l342_342328


namespace sum_of_slopes_of_dividing_lines_l342_342700

/-- Assuming a geometric figure defined by points O, P, Q, R, S, T with a total area of 48,
    and two lines through O that divide the figure into three equal areas,
    the sum of the slopes of these lines is 11/8. -/
theorem sum_of_slopes_of_dividing_lines
  (area_OPQRST : ℝ)
  (O P Q R S T : ℝ × ℝ)
  (condition1 : area_OPQRST = 48)
  (condition2 : O = (0, 0))
  (condition3 : True) -- additional conditions defining the shape and dimensions
  : let slope_OV := 9/8,
        slope_OW := 1/4
    in slope_OV + slope_OW = 11 / 8 :=
begin
  -- The proof should go here
  sorry
end

end sum_of_slopes_of_dividing_lines_l342_342700


namespace sum_of_x_y_l342_342370

theorem sum_of_x_y (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 48) : x + y = 2 :=
sorry

end sum_of_x_y_l342_342370


namespace map_length_represents_distance_l342_342322

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l342_342322


namespace regular_polygon_has_20_sides_l342_342953

-- Definition of a regular polygon with an exterior angle of 18 degrees
def regular_polygon_sides (external_angle : ℝ) : ℕ :=
  if external_angle > 0 then (360 / external_angle).toInt else 0

theorem regular_polygon_has_20_sides :
  regular_polygon_sides 18 = 20 :=
by
  sorry

end regular_polygon_has_20_sides_l342_342953


namespace first_day_of_month_is_tuesday_l342_342824

theorem first_day_of_month_is_tuesday (day23_is_wednesday : (23 % 7 = 3)) : (1 % 7 = 2) :=
sorry

end first_day_of_month_is_tuesday_l342_342824


namespace sum_a3_a4_a5_a6_l342_342640

theorem sum_a3_a4_a5_a6 (S : ℕ → ℕ) (h : ∀ n, S n = n^2 + 2 * n) : S 6 - S 2 = 40 :=
by
  sorry

end sum_a3_a4_a5_a6_l342_342640


namespace isosceles_triangle_perimeter_l342_342685

def is_isosceles_triangle (A B C : ℝ) : Prop :=
  (A = B ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (A = C ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (B = C ∧ A + B > C ∧ A + C > B ∧ B + C > A)

theorem isosceles_triangle_perimeter {A B C : ℝ} 
  (h : is_isosceles_triangle A B C) 
  (h1 : A = 3 ∨ A = 7) 
  (h2 : B = 3 ∨ B = 7) 
  (h3 : C = 3 ∨ C = 7) : 
  A + B + C = 17 := 
sorry

end isosceles_triangle_perimeter_l342_342685


namespace map_scale_l342_342351

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l342_342351


namespace equal_centers_l342_342402

-- Definitions of the main concepts given in the conditions
variables (M A B C D O1 O2 : Type)
variables (triangles_similar_opposite : Prop)

-- Define the key conditions
axiom condition1 : triangles_similar_opposite = (similar MAB MCD) ∧ opposite_orientation MAB MCD
axiom condition2 : O1 = center_of_rotation 2 (angle (vector A B, vector B M)) A C
axiom condition3 : O2 = center_of_rotation 2 (angle (vector A B, vector A M)) B D

-- The theorem to be proved
theorem equal_centers (M A B C D O1 O2 : Type)
  (triangles_similar_opposite : Prop)
  (condition1 : triangles_similar_opposite)
  (condition2 : O1 = center_of_rotation 2 (angle (vector A B, vector B M)) A C)
  (condition3 : O2 = center_of_rotation 2 (angle (vector A B, vector A M)) B D) :
  O1 = O2 :=
sorry

end equal_centers_l342_342402


namespace first_restaurant_meals_per_day_l342_342647

theorem first_restaurant_meals_per_day
  (second_restaurant_meals_per_day : ℕ)
  (third_restaurant_meals_per_day : ℕ)
  (total_meals_per_week : ℕ)
  (second_restaurant_meals_per_day = 40)
  (third_restaurant_meals_per_day = 50)
  (total_meals_per_week = 770) :
  ∃ first_restaurant_meals_per_day : ℕ, first_restaurant_meals_per_day = 20 :=
by
  let second_restaurant_meals_per_week := second_restaurant_meals_per_day * 7
  let third_restaurant_meals_per_week := third_restaurant_meals_per_day * 7
  let total_two_restaurants_meals_per_week := second_restaurant_meals_per_week + third_restaurant_meals_per_week
  let first_restaurant_meals_per_week := total_meals_per_week - total_two_restaurants_meals_per_week
  let first_restaurant_meals_per_day := first_restaurant_meals_per_week / 7
  exists first_restaurant_meals_per_day
  sorry

end first_restaurant_meals_per_day_l342_342647


namespace triangle_ratio_bounds_l342_342169

theorem triangle_ratio_bounds {A B C : ℝ} {a b c : ℝ} 
    (h1 : b^2 = 8 * a * c) 
    (h2 : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B) 
    (h3 : Real.sin B^2 = 8 * Real.sin A * Real.sin C) 
    (h4 : 0 < Real.cos B ∧ Real.cos B < 1) :
    (√6 / 3) < b / (a + c) ∧ b / (a + c) < 2 * √5 / 5 := 
by
  sorry

end triangle_ratio_bounds_l342_342169


namespace twice_x_minus_one_negative_l342_342565

variable (x : ℝ)

theorem twice_x_minus_one_negative (h : 2 * x - 1 < 0) : 2 * x - 1 < 0 :=
by
  assume h
  exact h

end twice_x_minus_one_negative_l342_342565


namespace map_distance_l342_342344

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l342_342344


namespace cannot_tile_with_hexagon_and_octagon_l342_342469

def internal_angle (n : ℕ) : ℝ := 180 - (360 / n)

theorem cannot_tile_with_hexagon_and_octagon :
  ¬ (∃ (hexagon octagon : Type), regular_polygon hexagon 6 ∧ regular_polygon octagon 8 ∧ can_tile [hexagon, octagon]) :=
sorry

end cannot_tile_with_hexagon_and_octagon_l342_342469


namespace sam_age_l342_342690

theorem sam_age (drew_current_age : ℕ) (drew_future_age : ℕ) (sam_future_age : ℕ) : 
  (drew_current_age = 12) → 
  (drew_future_age = drew_current_age + 5) → 
  (sam_future_age = 3 * drew_future_age) → 
  (sam_future_age - 5 = 46) := 
by sorry

end sam_age_l342_342690


namespace equation_of_ellipse_maximum_area_of_triangle_l342_342977

-- Definitions for the conditions
structure EllipseC where
  a b : ℝ
  (a_gt_b : a > b)
  (b_gt_zero : b > 0)
  (eccentricity : b / a = 1 / 2)
  (segment_length : 2 * b^2 / a = 1)

-- Definition for condition 5
structure Tangent where
  k m : ℝ
  (tangent_radius : |m| / (1 + k^2).sqrt = 2 * (5).sqrt / 5)

-- Mathematical proof problem 1
theorem equation_of_ellipse (C : EllipseC) :
  C.a = 2 ∧ C.b = 1 ∧ (∀ x y, (x * x) / (C.a * C.a) + (y * y) / (C.b * C.b) = 1 → (x*x)/4 + y*y = 1) :=
sorry

-- Mathematical proof problem 2
theorem maximum_area_of_triangle (C : EllipseC) (T : Tangent) :
  (∃ A B : ℝ × ℝ, ∃ O : ℝ × ℝ, let S := 1 / 2 * 2 * (5).sqrt / 5 * (5).sqrt in
  Triangle.area A B O = S) :=
sorry

end equation_of_ellipse_maximum_area_of_triangle_l342_342977


namespace sin_double_angle_l342_342718

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l342_342718


namespace frank_handed_cashier_amount_l342_342578

-- Place conditions as definitions
def cost_chocolate_bar : ℕ := 2
def cost_bag_chip : ℕ := 3
def num_chocolate_bars : ℕ := 5
def num_bag_chips : ℕ := 2
def change_received : ℕ := 4

-- Define the target theorem (Lean 4 statement)
theorem frank_handed_cashier_amount :
  (num_chocolate_bars * cost_chocolate_bar + num_bag_chips * cost_bag_chip + change_received = 20) := 
sorry

end frank_handed_cashier_amount_l342_342578


namespace f_positive_l342_342157

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def f'' : ℝ → ℝ := sorry

axiom f_monotonically_decreasing : ∀ x y : ℝ, x < y → f x > f y
axiom inequality_condition : ∀ x : ℝ, (f x) / (f'' x) + x < 1

theorem f_positive : ∀ x : ℝ, f x > 0 :=
by sorry

end f_positive_l342_342157


namespace cube_volume_l342_342149

noncomputable def volume_of_cube_with_space_diagonal :
    ℝ := 3 * sqrt 3

theorem cube_volume (A B : ℝ × ℝ × ℝ)
  (hA : A = (-1, 2, 1))
  (hB : B = (-2, 0, 3)) :
  ∃ V, V = volume_of_cube_with_space_diagonal :=
begin
  use volume_of_cube_with_space_diagonal,
  sorry
end

end cube_volume_l342_342149


namespace map_scale_l342_342353

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l342_342353


namespace units_digit_fraction_l342_342871

-- Given conditions
def numerator : ℕ := 30 * 31 * 32 * 33 * 34 * 35
def denominator : ℕ := 1500
def simplified_fraction : ℕ := 2^5 * 3 * 31 * 33 * 17 * 7

-- Statement of the proof goal
theorem units_digit_fraction :
  (simplified_fraction) % 10 = 2 := by
  sorry

end units_digit_fraction_l342_342871


namespace sum_of_coefficients_l342_342584

theorem sum_of_coefficients (a : Fin 7 → ℕ) (x : ℕ) : 
  (1 - x) ^ 6 = (a 0) + (a 1) * x + (a 2) * x^2 + (a 3) * x^3 + (a 4) * x^4 + (a 5) * x^5 + (a 6) * x^6 → 
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 0 := 
by
  intro h
  by_cases hx : x = 1
  · rw [hx] at h
    sorry
  · sorry

end sum_of_coefficients_l342_342584


namespace rectangular_prism_diagonal_and_surface_area_l342_342487

variables (a b c : ℝ)

theorem rectangular_prism_diagonal_and_surface_area :
  (a = 12) → (b = 15) → (c = 8) →
  (sqrt (a^2 + b^2 + c^2) = sqrt 433 ∧ 2 * (a * b + a * c + b * c) = 792) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  split
  · simp [pow_two, add_assoc]
  · simp [mul_add, add_assoc]
  sorry

end rectangular_prism_diagonal_and_surface_area_l342_342487


namespace max_distance_on_ellipse_l342_342215

noncomputable def upper_vertex : ℝ × ℝ := (0, 2)

noncomputable def ellipse (θ : ℝ) : ℝ × ℝ := (sqrt 5 * cos θ, 2 * sin θ)

noncomputable def distance (P B : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)

theorem max_distance_on_ellipse :
  ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * real.pi → distance (ellipse θ) upper_vertex ≤ 4 :=
begin
  sorry
end

end max_distance_on_ellipse_l342_342215


namespace problem_1_problem_2_l342_342048

-- Problem 1
theorem problem_1 : (3 - Real.pi)^0 - 2^2 + (1/2)^(-2) = 1 := by
  sorry

-- Problem 2
variables (a b : ℝ)

theorem problem_2 : ((a * b^2)^2 - 2 * a * b^4) / (a * b^4) = a - 2 := by
  sorry

end problem_1_problem_2_l342_342048


namespace sin_double_angle_solution_l342_342721

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_solution_l342_342721


namespace total_questions_completed_l342_342574

theorem total_questions_completed
  (x y z : ℝ)
  (fiona_first_hour : ℝ := 36)
  (shirley_first_hour : ℝ := 2 * fiona_first_hour)
  (kiana_first_hour : ℝ := (fiona_first_hour + shirley_first_hour) / 2) :
  let fiona_second_hour := fiona_first_hour + (x / 100) * fiona_first_hour,
      shirley_second_hour := shirley_first_hour + (y / 100) * shirley_first_hour,
      kiana_second_hour := kiana_first_hour + (z / 100) * kiana_first_hour,
      total_questions := (fiona_first_hour + fiona_second_hour) +
                         (shirley_first_hour + shirley_second_hour) +
                         (kiana_first_hour + kiana_second_hour)
  in 
  total_questions = 324 + 0.36 * x + 0.72 * y + 0.54 * z :=
by
  sorry

end total_questions_completed_l342_342574


namespace alcohol_percentage_new_mixture_l342_342912

def V_water := 3 -- initial water volume in liters
def V_solution := 11 -- initial solution volume in liters
def P_alcohol := 42 / 100 -- initial alcohol percentage in solution

theorem alcohol_percentage_new_mixture :
  let amount_alcohol := P_alcohol * V_solution in
  let total_volume := V_water + V_solution in
  (amount_alcohol / total_volume) * 100 = 33 :=
by
  sorry

end alcohol_percentage_new_mixture_l342_342912


namespace transformed_cos_eq_sin_l342_342374

theorem transformed_cos_eq_sin :
  ∀ x, (let y₀ := (λ x, cos (2 * x)) in
        let y₁ := (λ x, cos (2 * (x - π / 4))) in
        let y₂ := (λ x, sin (2 * x)) in
        let y₃ := (λ x, sin x) in
        y₀ (x - π / 4) = y₁ x ∧ y₁ x = y₂ x ∧ y₂ (x / 2) = y₃ x) :=
by
  sorry

end transformed_cos_eq_sin_l342_342374


namespace train_speed_in_kmh_l342_342520

def length_of_train : ℝ := 156
def length_of_bridge : ℝ := 219.03
def time_to_cross_bridge : ℝ := 30
def speed_of_train_kmh : ℝ := 45.0036

theorem train_speed_in_kmh :
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6 = speed_of_train_kmh :=
by
  sorry

end train_speed_in_kmh_l342_342520


namespace max_abc_value_l342_342246

theorem max_abc_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b + b * c = 518) (h2 : a * b - a * c = 360) : 
  a * b * c ≤ 1008 :=
sorry

end max_abc_value_l342_342246


namespace part_I_part_II_l342_342907

-- Statement for Part I
theorem part_I (x : ℝ) : |6 - |2*x+1|| > 1 ↔ x > 3 ∨ x < -4 ∨ -3 < x ∧ x < 2 := sorry

-- Statement for Part II
theorem part_II (m : ℝ) : (∃ (x : ℝ), |x+1| + |x-1| + 3 + x < m) → m > 4 := 
begin
  intro h,
  cases h with x h1,
  cases lt_or_ge x 1 with hx_lt hx_ge,
  {
    cases lt_or_ge x (-1) with hx_btw hx_lte,
    {
      -- Case: -1 < x < 1
      have h2 := h1,
      linarith,
    },
    {
      -- Case: x ≤ -1
      have h2 := h1,
      linarith,
    }
  },
  {
    -- Case: x ≥ 1
    have h2 := h1,
    linarith,
  }
end

end part_I_part_II_l342_342907


namespace sum_of_squares_l342_342625

-- Define the geometric sequence and its sum property
noncomputable def geometric_sequence (a : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0 else let r := 2 in (2 ^ (n - 1))

-- Define the sum of the first n terms of the geometric sequence
noncomputable def S_n (a : ℝ) (n : ℕ) : ℝ :=
  (2^n - a)

-- Prove that the sum of the squares of the first n terms of the geometric sequence
theorem sum_of_squares (a : ℝ) (n : ℕ) (h_Sn : S_n a n = (2^n - a)) :
  (finset.range n).sum (λ i, geometric_sequence a (i + 1) ^ 2) = (1 / 3) * (4^n - 1) :=
sorry

end sum_of_squares_l342_342625


namespace construct_right_triangle_l342_342499

-- Definition of an equilateral triangle
structure EquilateralTriangle (A B C : Point) : Prop :=
(equilateral : dist A B = dist B C ∧ dist B C = dist C A)

-- Definition of a circle passing through two points
structure CircleThroughPoints (A B : Point) (circle : Set Point) : Prop :=
(passes_through : A ∈ circle ∧ B ∈ circle)

-- Main theorem statement
theorem construct_right_triangle {A B C P : Point}
    (hABC : EquilateralTriangle A B C)
    (circle1 circle2 : Set Point)
    (hCircle1 : CircleThroughPoints A B circle1)
    (hCircle2 : CircleThroughPoints A C circle2)
    (hIntersection : P ∈ circle1 ∧ P ∈ circle2 ∧ ∠BPC = 90) :
    ∃ BP CP, is_right_triangle BP CP (dist A B) :=
sorry

end construct_right_triangle_l342_342499


namespace nonnegative_solutions_x_squared_eq_neg5x_l342_342651

theorem nonnegative_solutions_x_squared_eq_neg5x : 
  ∀ x : ℝ, x^2 = -5 * x → (x ≥ 0) → x = 0 :=
by
  intros x h_eq h_nonneg
  have h_eq_Rearranged : x * (x + 5) = 0 := by
    calc
      x * (x + 5)
        = x * x + x * 5 : by ring
        ... = x^2 + 5 * x : by ring
        ... = 0 : by rw [h_eq]
  have h_solutions : x = 0 ∨ x = -5 := by
    apply eq_zero_or_eq_zero_of_mul_eq_zero
    exact h_eq_Rearranged
  cases h_solutions with h_zero h_neg
  · exact h_zero
  · exfalso
    linarith

end nonnegative_solutions_x_squared_eq_neg5x_l342_342651


namespace solve_problem_l342_342769

variable {α : Type*} [LinearOrderedField α]

def f (x : α) : α := Real.sqrt (x^2 - 4)
def g (x : α) : α := Real.sqrt (x^2 + 4)

theorem solve_problem (a : α) (ha : 0 < a ∧ a < 1) : 
    f (a + 1 / a) + g (a - 1 / a) = 2 / a := 
by
  sorry

end solve_problem_l342_342769


namespace elastic_collision_inelastic_collision_l342_342424

-- Definition of conditions
variables {m L V : ℝ}
variables (w1 w2 : ℝ → Prop)

-- Proof problem for Elastic Collision
theorem elastic_collision (w1_at_collision : w1 L) (w2_at_collision : w2 L)  :
  let v1_after := V
      v2_after := -V in
  w1 L ∧ w2 L := sorry

-- Proof problem for Inelastic Collision
theorem inelastic_collision (w1_at_collision : w1 L) (w2_at_collision : w2 L)  :
  let omega := V / (2 * L) in
  w1 L ∧ w2 L := sorry

end elastic_collision_inelastic_collision_l342_342424


namespace countSequences_equals_300_l342_342650

def countSequences (s : String) : ℕ :=
  let letters := ['E', 'X', 'A', 'M', 'P', 'L', 'E'] -- The letters in "EXAMPLE"
  let first := 'E' -- Each sequence must begin with E
  let forbiddenEnd := 'X' -- Each sequence must not end with X
  let n := 5 -- Length of the sequence
  calc
    1 /* fix first letter */ * 
    (letters.erase first).erase forbiddenEnd.size /* choices for last letter */ * 
    Nat.factorial ((letters.erase first).erase forbiddenEnd.size) / Nat.factorial ((letters.erase first).erase forbiddenEnd.size - (n - 2))

theorem countSequences_equals_300 :
  countSequences "EXAMPLE" = 300 :=
  sorry

end countSequences_equals_300_l342_342650


namespace sin_double_angle_solution_l342_342722

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_solution_l342_342722


namespace find_coordinates_l342_342199

namespace CoordinateSystem

def Point := ℝ × ℝ × ℝ

def A : Point := (2, 1, 1)
def B : Point := (1, -3, 2)

def on_z_axis (M : Point) : Prop := M.fst = 0 ∧ M.snd.fst = 0

def distance (P Q : Point) : ℝ :=
  Math.sqrt ((P.fst - Q.fst)^2 + (P.snd.fst - Q.snd.fst)^2 + (P.snd.snd - Q.snd.snd)^2)

theorem find_coordinates (M : Point)
  (h1 : on_z_axis M)
  (h2 : distance M A = distance M B) :
  M = (0, 0, 4) :=
sorry

end CoordinateSystem

end find_coordinates_l342_342199


namespace A_times_B_is_correct_l342_342763

noncomputable def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
noncomputable def B : Set ℝ := {x : ℝ | x ≥ 0}

noncomputable def A_union_B : Set ℝ := {x : ℝ | x ≥ 0}
noncomputable def A_inter_B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

noncomputable def A_times_B : Set ℝ := {x : ℝ | x ∈ A_union_B ∧ x ∉ A_inter_B}

theorem A_times_B_is_correct :
  A_times_B = {x : ℝ | x > 2} := sorry

end A_times_B_is_correct_l342_342763


namespace value_of_v4_at_x_neg4_l342_342429

noncomputable def polynomial_evaluation (x : ℤ) : ℤ :=
  let coeffs := [12, 35, -8, 79, 6, 5, 3] in
  coeffs.foldr (λ a acc, a + acc * x) 0

theorem value_of_v4_at_x_neg4 : polynomial_evaluation (-4) = 220 :=
  by
    -- Proof by Qin Jiushao's algorithm
    sorry

end value_of_v4_at_x_neg4_l342_342429


namespace quadratic_inequality_solution_l342_342399

-- Define the quadratic equation and the given solution set range 
variables {a b c : ℝ}

-- The conditions provided in the problem
def solution_set := ∀ x : ℝ, (1/2 ≤ x ∧ x ≤ 2) → ax^2 + bx + c ≥ 0

-- Stating the target Lean theorem
theorem quadratic_inequality_solution (h : solution_set):
  b > 0 ∧ a + b + c > 0 :=
sorry

end quadratic_inequality_solution_l342_342399


namespace sum_max_min_expression_l342_342148

open Complex Real Set

noncomputable def vector_a (θ : ℝ) := (cos θ, sin θ)
noncomputable def vector_b := (sqrt 3, -1 : ℝ × ℝ)

def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

def θ_range : Set ℝ := {θ | θ + π / 6 ∈ Icc (π / 6) (7 * π / 6)}

def expression (θ : ℝ) : ℝ :=
  magnitude (2 • vector_a θ - vector_b)

theorem sum_max_min_expression :
  ∃θ₁ θ₂ ∈ θ_range, 
  let max_val := expression θ₁
  let min_val := expression θ₂
  max_val = 4 ∧ min_val = sqrt 6 - sqrt 2 ∧ 
  4 + sqrt 6 - sqrt 2 = max_val + min_val :=
by
  sorry

end sum_max_min_expression_l342_342148


namespace calc_expression_l342_342987

theorem calc_expression : (5 / 3) ^ 2023 * 0.6 ^ 2022 = 5 / 3 := 
by
  have h : 0.6 = 3 / 5 := by norm_num
  rw [h]
  sorry

end calc_expression_l342_342987


namespace map_scale_representation_l342_342337

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l342_342337


namespace even_product_divisible_l342_342501

theorem even_product_divisible (n : ℕ) : 
  (∏ i in (finset.range (n/2 + 1)).map ((*) 2), i) % (1997 * 2011 * 2027) = 0 ↔ n = 4056 := 
by
  sorry

end even_product_divisible_l342_342501


namespace number_of_distinct_m_values_l342_342227

theorem number_of_distinct_m_values (m : ℤ) :
  (∃ x1 x2 : ℤ, x1 * x2 = 36 ∧ x1 + x2 = m) →
  set.card {m | ∃ x1 x2 : ℤ, x1 * x2 = 36 ∧ x1 + x2 = m} = 10 :=
by
  sorry

end number_of_distinct_m_values_l342_342227


namespace max_tan_value_l342_342616

open real

noncomputable def max_tan (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2)
                          (h₂ : 0 < β ∧ β < π / 2)
                          (h₃ : cos (α + β) = sin α / sin β) : ℝ :=
  sup (set_of (λ x : ℝ, 0 < x ∧ x = tan α ∧ x ≤ sqrt 2 / 4))

theorem max_tan_value (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2)
                           (h₂ : 0 < β ∧ β < π / 2)
                           (h₃ : cos (α + β) = sin α / sin β) :
  max_tan α β h₁ h₂ h₃ = sqrt 2 / 4 :=
sorry

end max_tan_value_l342_342616


namespace tetrahedron_sphere_intersection_l342_342674

structure Point (α : Type*) := 
(x : α) 
(y : α) 
(z : α)

structure Tetrahedron (α : Type*) :=
(a b c d : Point α)
(edge_length : α)

noncomputable def sphere_intersection_length (α : Type*) [linear_ordered_field α] 
  (o : Point α) (r : α) (t : Tetrahedron α) : α := 
sorry

theorem tetrahedron_sphere_intersection {α : Type*} [linear_ordered_field α] :
  let o := Point.mk 0 0 0
  let r := (3 : α)^(1/2) -- sqrt(3)
  let a := 2 * (6 : α)^(1/2) -- 2sqrt(6)
  let t : Tetrahedron α := { 
    a := Point.mk a 0 0, 
    b := Point.mk (a / 2) (a * ((3:α)^(1/2) / 2)) 0,
    c := Point.mk (a / 2) ((a * (3:α)^(1/2)) / 6) (a * ((6:α)^(1/2)) / 3),
    d := Point.mk (a / 2) ((a * (3:α)^(1/2)) / 6) (-(a * ((6:α)^(1/2)) / 3)),
    edge_length := a
  } in
  sphere_intersection_length α o r t = 8 * (2 : α)^(1/2) * real.pi := 
sorry

end tetrahedron_sphere_intersection_l342_342674


namespace sequence_expression_l342_342593

open_locale classical

noncomputable theory

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
2 * b = a + c

def a_n (n : ℕ) : ℕ := n

def S_n (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem sequence_expression (n : ℕ) (h : n ≥ 1) :
  ∀ n : ℕ, n ≥ 1 → 2 * S_n n = a_n n + (a_n n) ^ 2 →
  a_n n = n :=
sorry

end sequence_expression_l342_342593


namespace product_is_cube_l342_342741

/-
  Given conditions:
    - a, b, and c are distinct composite natural numbers.
    - None of a, b, and c are divisible by any of the integers from 2 to 100 inclusive.
    - a, b, and c are the smallest possible numbers satisfying the above conditions.

  We need to prove that their product a * b * c is a cube of a natural number.
-/

noncomputable def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ n = p * q

theorem product_is_cube (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : is_composite a) (h5 : is_composite b) (h6 : is_composite c)
  (h7 : ∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ a))
  (h8 : ∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ b))
  (h9 : ∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ c))
  (h10 : ∀ (d e f : ℕ), is_composite d → is_composite e → is_composite f → d ≠ e → e ≠ f → d ≠ f → 
         (∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ d)) → (∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ e)) → (∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ f)) →
         (d * e * f ≥ a * b * c)) :
  ∃ (n : ℕ), a * b * c = n ^ 3 :=
by
  sorry

end product_is_cube_l342_342741


namespace largest_four_digit_negative_congruent_3_mod_29_l342_342436

theorem largest_four_digit_negative_congruent_3_mod_29 : 
  ∃ (n : ℤ), n < 0 ∧ n ≥ -9999 ∧ (n % 29 = 3) ∧ n = -1012 :=
sorry

end largest_four_digit_negative_congruent_3_mod_29_l342_342436


namespace basketball_team_starting_lineups_l342_342490

theorem basketball_team_starting_lineups :
  let total_players := 16
  let starters := 7
  let triplets := 3
  let twins := 2
  let unrestricted_lineups := Nat.choose 16 7
  let invalid_triplets_lineups := Nat.choose 13 4
  let invalid_twins_lineups := Nat.choose 14 5
  let overlap_lineups := Nat.choose 11 2
  unrestricted_lineups - (invalid_triplets_lineups + invalid_twins_lineups - overlap_lineups) = 8778 := by
  sorry

end basketball_team_starting_lineups_l342_342490


namespace range_of_circle_center_x_coordinate_l342_342187

noncomputable def circle_center_x_range : Set ℝ :=
  {a : ℝ | 0 ≤ a ∧ a ≤ 12 / 5}
  
theorem range_of_circle_center_x_coordinate :
  ∀ (a : ℝ),
    (∃ (M : ℝ × ℝ), (M.1^2 + (M.2 - 3)^2 = 4 ∧ M.1^2 + M.2^2 = 1) ∧
                      ∃ (x y : ℝ), y = 2 * x - 4 ∧ (M.1^2 + (M.2 + 1)^2 = 4)) →
    (center_circle_C : ℝ × ℝ)
    (center_circle_C.2 = 2 * center_circle_C.1 - 4) →
    (C: circle center_circle_C 1) →
    a = center_circle_C.1 →
    a ∈ circle_center_x_range :=
sorry

end range_of_circle_center_x_coordinate_l342_342187


namespace number_of_arrangements_l342_342405

theorem number_of_arrangements (boys girls : ℕ) (adjacent : Bool) (not_ends : Bool) : 
  (boys = 5) → (girls = 2) → (adjacent = true) → (not_ends = true) → 
  (∃ n, n = 960) := by
  intros hboys hgirls hadjacent hnot_ends
  exists 960
  sorry

end number_of_arrangements_l342_342405


namespace volume_of_each_hemisphere_container_is_correct_l342_342524

-- Define the given conditions
def Total_volume : ℕ := 10936
def Number_containers : ℕ := 2734

-- Define the volume of each hemisphere container
def Volume_each_container : ℕ := Total_volume / Number_containers

-- The theorem to prove, asserting the volume is correct
theorem volume_of_each_hemisphere_container_is_correct :
  Volume_each_container  = 4 := by
  -- placeholder for the actual proof
  sorry

end volume_of_each_hemisphere_container_is_correct_l342_342524


namespace wall_height_l342_342411

noncomputable def brick_volume : ℝ := 25 * 11.25 * 6

noncomputable def total_brick_volume : ℝ := brick_volume * 6400

noncomputable def wall_length : ℝ := 800

noncomputable def wall_width : ℝ := 600

theorem wall_height :
  ∀ (wall_volume : ℝ), 
  wall_volume = total_brick_volume → 
  wall_volume = wall_length * wall_width * 22.48 :=
by
  sorry

end wall_height_l342_342411


namespace more_not_ten_digit_products_of_two_five_digits_l342_342024

theorem more_not_ten_digit_products_of_two_five_digits :
  let num_ten_digit_numbers := 9 * 10^9,
      num_five_digit_numbers := 90000,
      num_pairs := (num_five_digit_numbers * (num_five_digit_numbers - 1)) / 2 + num_five_digit_numbers,
      num_pairs_in_range := num_pairs / 2
  in num_pairs_in_range < num_ten_digit_numbers / 2 :=
by
  let num_ten_digit_numbers := 9 * 10^9
  let num_five_digit_numbers := 90000
  let num_pairs := (num_five_digit_numbers * (num_five_digit_numbers - 1)) / 2 + num_five_digit_numbers
  let num_pairs_in_range := num_pairs / 2
  have H : num_pairs_in_range < num_ten_digit_numbers / 2 := sorry
  exact H

end more_not_ten_digit_products_of_two_five_digits_l342_342024


namespace count_ordered_triples_l342_342153

theorem count_ordered_triples (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a^2 = b^2 + c^2) (h5 : b^2 = a^2 + c^2) (h6 : c^2 = a^2 + b^2) : 
  (a = b ∧ b = c ∧ a ≠ 0) ∨ (a = -b ∧ b = c ∧ a ≠ 0) ∨ (a = b ∧ b = -c ∧ a ≠ 0) ∨ (a = -b ∧ b = -c ∧ a ≠ 0) :=
sorry

end count_ordered_triples_l342_342153


namespace lehmer_mean_inequality_A_lehmer_mean_inequality_B_l342_342219

variables (a b : ℝ) (p : ℚ)

def arithmetic_mean (a b : ℝ) : ℝ := (a + b) / 2
def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)
def lehmer_mean (p : ℚ) (a b : ℝ) : ℝ := (a^p + b^p) / (a^(p-1) + b^(p-1))

-- Positive condition
variable (h_pos : 0 < a ∧ 0 < b)

theorem lehmer_mean_inequality_A : lehmer_mean (0.5 : ℚ) a b ≤ lehmer_mean (1 : ℚ) a b :=
sorry

theorem lehmer_mean_inequality_B : lehmer_mean (0 : ℚ) a b ≤ geometric_mean a b :=
sorry

end lehmer_mean_inequality_A_lehmer_mean_inequality_B_l342_342219


namespace local_maximum_no_global_maximum_equation_root_condition_l342_342128

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3) * Real.exp x
noncomputable def f' (x : ℝ) : ℝ := (x^2 + 2*x - 3) * Real.exp x

theorem local_maximum_no_global_maximum : (∃ x0 : ℝ, f' x0 = 0 ∧ (∀ x < x0, f x < f x0) ∧ (∀ x > x0, f x < f x0))
∧ (f 1 = -2 * Real.exp 1) 
∧ (∀ x : ℝ, ∃ b : ℝ, f x = b ∧ b > 6 * Real.exp (-3) → ¬(f x = f 1))
:= sorry

theorem equation_root_condition (b : ℝ) : (∃ x1 x2 x3 : ℝ, f x1 = b ∧ f x2 = b ∧ f x3 = b ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) 
→ (0 < b ∧ b < 6 * Real.exp (-3))
:= sorry

end local_maximum_no_global_maximum_equation_root_condition_l342_342128


namespace find_analytical_expression_and_m_range_l342_342115

-- Definitions from the problem conditions
def logarithmic_function (a : ℝ) (h : a > 0 ∧ a ≠ 1) := ∀ x : ℝ, log a x

-- Given conditions and required proofs
theorem find_analytical_expression_and_m_range (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  (logarithmic_function a h 3 = 1 / 2) →
  (∀ x, logarithmic_function a h x = Real.log x / Real.log a) →
  let f := fun x => Real.log x / (2 * Real.log 3) in
  (∀ x, 1 < x ∧ x < 5 → (f x) > 0 ∧ (f x) < Real.log 5 / (2 * Real.log 3)) ∧
  (1 < (Real.exp (m * 2 * Real.log 3)) ∧ (Real.exp (m * 2 * Real.log 3)) < 5 → 
  m > 0 ∧ m < Real.log 5 / (2 * Real.log 3)) :=
by {
  sorry
}

end find_analytical_expression_and_m_range_l342_342115


namespace pumps_empty_pond_l342_342803

def rate_first_pump : ℝ := 1 / 5
def rate_second_pump : ℝ := 1 / (5 / 4)  -- equivalent to 1 / 1.25

def combined_rate : ℝ := rate_first_pump + rate_second_pump

theorem pumps_empty_pond (r1 r2 c : ℝ) (half_pond_time : ℝ) (full_pond_time : ℝ) (initial_time : ℝ) :
  r1 = rate_first_pump →
  r2 = rate_second_pump →
  c = combined_rate →
  half_pond_time = 2.5 →
  full_pond_time = 1.25 →
  initial_time = (0.5:ℝ / c) →
  initial_time = 0.5 :=
by
  intros
  sorry

end pumps_empty_pond_l342_342803


namespace Jeremy_age_l342_342533

noncomputable def A : ℝ := sorry
noncomputable def J : ℝ := sorry
noncomputable def C : ℝ := sorry

-- Conditions
axiom h1 : A + J + C = 132
axiom h2 : A = (1/3) * J
axiom h3 : C = 2 * A

-- The goal is to prove J = 66
theorem Jeremy_age : J = 66 :=
sorry

end Jeremy_age_l342_342533


namespace necessary_condition_not_sufficient_condition_l342_342905

theorem necessary_condition (α β : ℝ) (h : sin α + cos β = 0) : sin^2 α + sin^2 β = 1 :=
by sorry

theorem not_sufficient_condition (α β : ℝ) (h : sin α + cos β ≠ 0) : sin^2 α + sin^2 β = 1 → false :=
by sorry

end necessary_condition_not_sufficient_condition_l342_342905


namespace fried_chicken_dinner_pieces_l342_342860

-- Definitions based on the conditions from the problem statement.
def chicken_pasta := 2
def barbecue_chicken := 3
def fried_chicken_dinner : Nat
def total_orders_pasta := 6
def total_orders_barbecue := 3
def total_orders_fried := 2
def total_pieces := 37

-- The statement to be proved
theorem fried_chicken_dinner_pieces :
  2 * total_orders_fried + 6 * chicken_pasta + 3 * barbecue_chicken = total_pieces → 
  2 * fried_chicken_dinner + (total_orders_pasta * chicken_pasta) + (total_orders_barbecue * barbecue_chicken) = total_pieces → 
  fried_chicken_dinner = 8 :=
by
  -- The conditions and statement have been provided, proof can be filled later
  sorry

end fried_chicken_dinner_pieces_l342_342860


namespace cost_of_article_l342_342160

-- Definitions
variables (C G : ℝ)
def equation1 := 580 = C + G
def equation2 := 600 = C + 1.05 * G

-- Theorem to prove
theorem cost_of_article : C = 180 :=
by
  have h1 : 580 = C + G := equation1,
  have h2 : 600 = C + 1.05 * G := equation2,
  sorry

end cost_of_article_l342_342160


namespace clock_strikes_one_day_l342_342915

theorem clock_strikes_one_day : 
  let S := (12 * (12 + 1)) / 2
  in S + S = 156 :=
by
  -- Definitions based on conditions
  let S := (12 * (12 + 1)) / 2
  -- Show that the total strikes in one day is 156
  show S + S = 156
  sorry

end clock_strikes_one_day_l342_342915


namespace map_length_representation_l342_342311

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l342_342311


namespace least_number_of_cans_l342_342504

noncomputable def gcd (a b : ℕ) : ℕ := Nat.gcd a b

noncomputable def gcd_three (a b c : ℕ) : ℕ := gcd (gcd a b) c

theorem least_number_of_cans (a b c gcd_abc : ℕ) 
  (ha : a = 60) 
  (hb : b = 144) 
  (hc : c = 368) 
  (hg : gcd_abc = gcd_three a b c)
  (h_gcd_value : gcd_abc = 4) :
  (a / gcd_abc + b / gcd_abc + c / gcd_abc = 143) :=
by
  sorry

end least_number_of_cans_l342_342504


namespace part1_solution_set_a_eq_2_part2_range_of_a_l342_342635

noncomputable def f (a x : ℝ) : ℝ := abs (x - a) + abs (2 * x - 2)

theorem part1_solution_set_a_eq_2 :
  { x : ℝ | f 2 x > 2 } = { x | x < (2 / 3) } ∪ { x | x > 2 } :=
by
  sorry

theorem part2_range_of_a :
  { a : ℝ | ∀ x : ℝ, f a x ≥ 2 } = { a | a ≤ -1 } ∪ { a | a ≥ 3 } :=
by
  sorry

end part1_solution_set_a_eq_2_part2_range_of_a_l342_342635


namespace regular_polygon_sides_l342_342944

theorem regular_polygon_sides (θ : ℝ) (n : ℕ) (h1 : θ = 18) (h2 : 360 = θ * n) : n = 20 :=
by {
  sorry
}

end regular_polygon_sides_l342_342944


namespace variance_translation_invariant_l342_342597

noncomputable def variance (data : List ℝ) : ℝ :=
  let mean := (data.sum / data.length)
  (data.map (λ x, (x - mean) ^ 2)).sum / data.length

theorem variance_translation_invariant
  (data : List ℝ)
  (h : variance data = 7)
  (translated_data : List ℝ := data.map (λ x, x - 1)) :
  variance translated_data = 7 := by
  sorry

end variance_translation_invariant_l342_342597


namespace map_length_representation_l342_342282

theorem map_length_representation :
  ∀ (length1 length2 km1 : ℝ), 
  length1 = 15 ∧ km1 = 90 ∧ length2 = 20 →
  (length2 * (km1 / length1) = 120) :=
by
  intros length1 length2 km1
  intro h
  cases h with hl1 h
  cases h with hkm1 hl2
  rw [hl1, hkm1, hl2]
  simp
  sorry

end map_length_representation_l342_342282


namespace total_amount_of_money_l342_342494

def total_money (P₁ P₂ : ℝ) : ℝ := P₁ + P₂

def interest (rate P : ℝ) : ℝ := rate * P / 100

theorem total_amount_of_money (P₁ P₂ : ℝ) (h₁ : P₁ ≈ 1300) (h₂ : interest 3 P₁ + interest 5 P₂ = 144) : total_money P₁ P₂ = 3400 :=
sorry

end total_amount_of_money_l342_342494


namespace distance_and_slope_correct_l342_342570

structure Point :=
(x : ℝ)
(y : ℝ)

def distance (p1 p2 : Point) : ℝ :=
Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

def slope (p1 p2 : Point) : ℝ :=
(p2.y - p1.y) / (p2.x - p1.x)

theorem distance_and_slope_correct :
  let p1 := Point.mk (-3) (1)
  let p2 := Point.mk (1) (-3)
  distance p1 p2 = 4 * Real.sqrt 2 ∧ slope p1 p2 = -1 :=
by
  sorry

end distance_and_slope_correct_l342_342570


namespace ellipsoid_center_and_axes_sum_l342_342551

theorem ellipsoid_center_and_axes_sum :
  let x₀ := -2
  let y₀ := 3
  let z₀ := 1
  let A := 6
  let B := 4
  let C := 2
  x₀ + y₀ + z₀ + A + B + C = 14 := 
by
  sorry

end ellipsoid_center_and_axes_sum_l342_342551


namespace tetrahedron_sphere_intersection_l342_342673

structure Point (α : Type*) := 
(x : α) 
(y : α) 
(z : α)

structure Tetrahedron (α : Type*) :=
(a b c d : Point α)
(edge_length : α)

noncomputable def sphere_intersection_length (α : Type*) [linear_ordered_field α] 
  (o : Point α) (r : α) (t : Tetrahedron α) : α := 
sorry

theorem tetrahedron_sphere_intersection {α : Type*} [linear_ordered_field α] :
  let o := Point.mk 0 0 0
  let r := (3 : α)^(1/2) -- sqrt(3)
  let a := 2 * (6 : α)^(1/2) -- 2sqrt(6)
  let t : Tetrahedron α := { 
    a := Point.mk a 0 0, 
    b := Point.mk (a / 2) (a * ((3:α)^(1/2) / 2)) 0,
    c := Point.mk (a / 2) ((a * (3:α)^(1/2)) / 6) (a * ((6:α)^(1/2)) / 3),
    d := Point.mk (a / 2) ((a * (3:α)^(1/2)) / 6) (-(a * ((6:α)^(1/2)) / 3)),
    edge_length := a
  } in
  sphere_intersection_length α o r t = 8 * (2 : α)^(1/2) * real.pi := 
sorry

end tetrahedron_sphere_intersection_l342_342673


namespace double_sum_value_l342_342546

theorem double_sum_value :
  ∑ i in Finset.range 150, ∑ j in Finset.range 150, (i + 1 + j + 1 + (i + 1) * (j + 1)) = 14542375 :=
by
  sorry

end double_sum_value_l342_342546


namespace product_divisible_by_8_l342_342531

-- Define a standard 8-sided die with values from 1 to 8.
def is_standard_8_sided_die (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 8

-- Define a function that rolls 4 dice and checks if their product is divisible by 8.
def probability_divisible_by_8 : ℚ :=
  let total_ways := 8^4 in
  let ways_product_divisible_by_8 := 53 * 64 in
  (ways_product_divisible_by_8 : ℚ) / total_ways

-- The theorem we aim to prove
theorem product_divisible_by_8 (d1 d2 d3 d4 : ℕ)
  (h1 : is_standard_8_sided_die d1)
  (h2 : is_standard_8_sided_die d2)
  (h3 : is_standard_8_sided_die d3)
  (h4 : is_standard_8_sided_die d4) :
  probability_divisible_by_8 = 53 / 64 :=
sorry

end product_divisible_by_8_l342_342531


namespace problem_l342_342106

def ellipse_foci : Prop :=
  ∃ C : Set (ℝ × ℝ),
    (∀ p ∈ C, dist p (-2, 0) + dist p (2, 0) = 6)

def ellipse_contains_point : Prop :=
  ellipse_foci ∧ ⟨0, sqrt 5⟩ ∈ C

def ellipse_standard_equation : Prop :=
  E : Set (ℝ × ℝ) := { p | p.1^2 / 9 + p.2^2 / 5 = 1 }

def intersection_line_ellipse : Prop :=
  E ∧ ∀ P Q ∈ E,
    (P.2 = P.1 + 2) ∧ (Q.2 = Q.1 + 2) →
      dist P Q = 30 / 7

theorem problem : ellipse_foci → ellipse_contains_point → ellipse_standard_equation → intersection_line_ellipse := by
  intros,
  sorry

end problem_l342_342106


namespace no_such_N_l342_342069

variable {R : Type} [CommRing R]

def transpose_and_mult (N : Matrix (Fin 2) (Fin 2) R) (M : Matrix (Fin 2) (Fin 2) R) : Matrix (Fin 2) (Fin 2) R :=
  N ⬝ M

theorem no_such_N (a b c d : R) :
  ∀ (N : Matrix (Fin 2) (Fin 2) R),
    N ⬝ (λ (i j : Fin 2), match (i, j) with
      | (0, 0) => a
      | (0, 1) => b
      | (1, 0) => c
      | (1, 1) => d
      end) ≠
    (λ (i j : Fin 2), match (i, j) with
      | (0, 0) => 2 * b
      | (0, 1) => a
      | (1, 0) => 2 * d
      | (1, 1) => c
      end) →
  N = (0 : Matrix (Fin 2) (Fin 2) R) := sorry


end no_such_N_l342_342069


namespace final_position_total_distance_traveled_fuel_needed_to_return_l342_342986

def distances : List ℝ := [7, -12, 15, -3.5, 5, 4, -7, -11.5]
def fuelConsumptionRate : ℝ := 0.4

-- Prove that the final position relative to point A is 3 km west (i.e., -3 km)
theorem final_position (d : List ℝ) : (d = distances) → (d.sum = -3) :=
by 
  intro hd 
  rw hd
  simp
  sorry

-- Prove that the total distance traveled is 68 km
theorem total_distance_traveled (d : List ℝ) : (d = distances) → (d.map abs).sum = 68 :=
by 
  intro hd 
  rw hd
  simp
  sorry

-- Prove that the fuel needed to return to point A is 1.2 liters
theorem fuel_needed_to_return (d : List ℝ) (rate : ℝ) : (d = distances) → (rate = fuelConsumptionRate) → abs d.sum * rate = 1.2 :=
by 
  intro hd hr 
  rw [hd, hr]
  simp
  sorry

end final_position_total_distance_traveled_fuel_needed_to_return_l342_342986


namespace find_angle_B_find_AD_l342_342618

-- Define properties of the triangle ABC and the given conditions
variable (A B C : Real) (a b c : Real)

-- Given conditions
axiom acute_triangle_ABC : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
axiom law_of_sines : a / Real.sin A = b / Real.sin B
axiom given_condition1 : b * Real.sin A = a * Real.cos (B - π / 6)

-- Part 1: Find the value of angle B
theorem find_angle_B : B = π / 3 :=
  by
  sorry

-- Part 2: Find the length of AD
variable (D : Real) 
axiom point_D_on_AC : D ∈ Ioo a c  -- D is between A and C on the line segment

variable (S_ABD S_ABC AD : Real)
axiom given_condition2 : b = Real.sqrt 13
axiom given_condition3 : a = 4
axiom area_ABD : S_ABD = 2 * Real.sqrt 3

theorem find_AD : AD = 2 * Real.sqrt 13 / 3 :=
  by
  sorry

end find_angle_B_find_AD_l342_342618


namespace _l342_342192

lemma binomial_integral_theorem :
  let m := (Nat.choose 8 3) * (1 / 2) ^ 3 in
  ∫ x in 0..1, (m * x + sqrt (1 - x^2)) = (7 / 2) + (π / 4) :=
by
  let m := (Nat.choose 8 3) * (1 / 2) ^ 3
  have h_m : m = 7 := by sorry -- This is derived from the binomial coefficient calculation
  calc
    ∫ x in 0..1, (m * x + sqrt (1 - x^2))
       = ∫ x in 0..1, (7 * x + sqrt (1 - x^2)) : by rw [h_m]
   ... = (7 / 2) + (π / 4) : by sorry

end _l342_342192


namespace λ_is_correct_l342_342121

variables {ℝ : Type*} [linear_ordered_field ℝ]

structure Vector (ℝ : Type*) := (e1 e2 : ℝ)

noncomputable def λ_value (e1 e2: ℝ) (H1: e1 ≠ 0 ∨ e2 ≠ 0) (H2: ∃ (m: ℝ), 3 = m ∧ -2 = λ * m) : ℝ :=
- 2 / 3

-- Proof the above definition given the conditions.
theorem λ_is_correct (e1 e2: ℝ) (H1: e1 ≠ 0 ∨ e2 ≠ 0) (λ : ℝ)
  (H2: ∃ (m: ℝ), 3 = m ∧ -2 = λ * m) : λ = -2 / 3 :=
sorry

end λ_is_correct_l342_342121


namespace investment_B_l342_342970

-- Definitions based on conditions
variables (A C B Prof_A Prof_Total Invest_A Invest_C Invest_Total : ℝ)

def Invest_A := 6300
def Invest_C := 10500
def Prof_Total := 13000
def Prof_A := 3900

-- Definition based on the problem context
def Invest_B := B

-- Total investment equation
def Total_Investment := Invest_A + Invest_B + Invest_C

-- Given ratio from the problem statement
def Ratio_A := Invest_A / Total_Investment
def Ratio_Profit := Prof_A / Prof_Total

-- Lean problem statement
theorem investment_B :
  Ratio_A = Ratio_Profit →
  Invest_B = 4200 :=
by
  -- use sorry to skip the proof
  sorry

end investment_B_l342_342970


namespace find_x_such_that_custom_op_neg3_eq_one_l342_342168

def custom_op (x y : Int) : Int := x * y - 2 * (x + y)

theorem find_x_such_that_custom_op_neg3_eq_one :
  ∃ x : Int, custom_op x (-3) = 1 ∧ x = 1 :=
by
  use 1
  sorry

end find_x_such_that_custom_op_neg3_eq_one_l342_342168


namespace prove_correct_statement_l342_342027

-- Define the conditions; we use the negation of incorrect statements
def condition1 (a b : ℝ) : Prop := a ≠ b → ¬((a - b > 0) → (a > 0 ∧ b > 0))
def condition2 (x : ℝ) : Prop := ¬(|x| > 0)
def condition4 (x : ℝ) : Prop := x ≠ 0 → (¬(∃ y, y = 1 / x))

-- Define the statement we want to prove as the correct one
def correct_statement (q : ℚ) : Prop := 0 - q = -q

-- The main theorem that combines conditions and proves the correct statement
theorem prove_correct_statement (a b : ℝ) (q : ℚ) :
  condition1 a b →
  condition2 a →
  condition4 a →
  correct_statement q :=
  by
  intros h1 h2 h4
  unfold correct_statement
  -- Proof goes here
  sorry

end prove_correct_statement_l342_342027


namespace greatest_sum_of_int_pairs_squared_eq_64_l342_342406

theorem greatest_sum_of_int_pairs_squared_eq_64 :
  ∃ (x y : ℤ), x^2 + y^2 = 64 ∧ (∀ (a b : ℤ), a^2 + b^2 = 64 → a + b ≤ 8) ∧ x + y = 8 :=
by 
  sorry

end greatest_sum_of_int_pairs_squared_eq_64_l342_342406


namespace number_of_distinct_permutations_l342_342649

-- Define the given digits list
def digits : List ℕ := [1, 1, 1, 7, 7]

-- Define a function that calculates the factorial
noncomputable def fact (n : ℕ) : ℕ :=
  (Nat.factorial n)

-- Prove that the number of distinct permutations of the given digits is 10
theorem number_of_distinct_permutations :
  List.perm_length digits (fact 5) / ((fact 3) * (fact 2)) = 10 :=
by
  sorry

end number_of_distinct_permutations_l342_342649


namespace sum_tens_ones_digit_of_7_pow_11_l342_342458

/--
The sum of the tens digit and the ones digit of (3+4)^{11} is 7.
-/
theorem sum_tens_ones_digit_of_7_pow_11 : 
  let number := (3 + 4)^11
  let tens_digit := (number / 10) % 10
  let ones_digit := number % 10
  tens_digit + ones_digit = 7 :=
by
  sorry

end sum_tens_ones_digit_of_7_pow_11_l342_342458


namespace triangle_perimeter_l342_342676

theorem triangle_perimeter (a b : ℝ) (x : ℝ) 
  (h₁ : a = 3) 
  (h₂ : b = 5) 
  (h₃ : x ^ 2 - 5 * x + 6 = 0)
  (h₄ : 2 < x ∧ x < 8) : a + b + x = 11 :=
by sorry

end triangle_perimeter_l342_342676


namespace train_half_speed_time_l342_342521

-- Definitions for Lean
variables (S T D : ℝ)

-- Conditions
axiom cond1 : D = S * T
axiom cond2 : D = (1 / 2) * S * (T + 4)

-- Theorem Statement
theorem train_half_speed_time : 
  (T = 4) → (4 + 4 = 8) := 
by 
  intros hT
  simp [hT]

end train_half_speed_time_l342_342521


namespace total_shirts_sold_l342_342011

theorem total_shirts_sold (p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 : ℕ) (h1 : p1 = 20) (h2 : p2 = 22) (h3 : p3 = 25)
(h4 : p4 + p5 + p6 + p7 + p8 + p9 + p10 = 133) (h5 : ((p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10) / 10) > 20)
: p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 = 200 ∧ 10 = 10 := sorry

end total_shirts_sold_l342_342011


namespace part_a_part_b_l342_342896

noncomputable def f : ℕ → ℕ := sorry
def is_finite (f : ℕ → ℕ) := ∃ S : set ℕ, S.finite ∧ ∀ n : ℕ, f n ∈ S
def is_periodic (f : ℕ → ℕ) := ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, f (n + p) = f n
def satisfies_equation (f : ℕ → ℕ) := ∀ n : ℕ, f (n + f n) = f n

theorem part_a (h : satisfies_equation f) (h_fin : is_finite f) : is_periodic f := sorry

-- Example of a non-periodic function satisfying the functional equation
def example_non_periodic : ℕ → ℕ := λ n, nat.log n

theorem part_b : satisfies_equation example_non_periodic ∧ ¬(is_periodic example_non_periodic) :=
begin
  split,
  {
    assume n,
    have h := nat.log_add (example_non_periodic n),
    rw h,
  },
  {
    intro h_periodic,
    obtain ⟨p, hp_pos, hp_periodic⟩ := h_periodic,
    sorry -- Detailed proof showing example_non_periodic is not periodic
  }
end

end part_a_part_b_l342_342896


namespace roots_not_integers_l342_342592

theorem roots_not_integers (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) :
    ¬ ∃ x₁ x₂ : ℤ, a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  sorry

end roots_not_integers_l342_342592


namespace distinct_m_count_l342_342236

noncomputable def countDistinctMValues : Nat :=
  let pairs := [(1, 36), (2, 18), (3, 12), (4, 9), (6, 6), 
                (-1, -36), (-2, -18), (-3, -12), (-4, -9), (-6, -6)]
  let ms := pairs.map (λ p => p.1 + p.2)
  ms.eraseDups.length

theorem distinct_m_count :
  countDistinctMValues = 10 := sorry

end distinct_m_count_l342_342236


namespace jo_age_l342_342381

theorem jo_age (j d g : ℕ) (even_j : 2 * j = j * 2) (even_d : 2 * d = d * 2) (even_g : 2 * g = g * 2)
    (h : 8 * j * d * g = 2024) : 2 * j = 46 :=
sorry

end jo_age_l342_342381


namespace faster_train_pass_time_l342_342426

-- Define conditions
def length_of_train : ℝ := 100 -- in meters
def speed_faster_train : ℝ := 46 -- in km/hr
def speed_slower_train : ℝ := 36 -- in km/hr

-- Convert speeds to m/s
def speed_faster_train_m_per_s : ℝ := speed_faster_train * (1000 / 3600)
def speed_slower_train_m_per_s : ℝ := speed_slower_train * (1000 / 3600)

-- Define relative speed
def relative_speed : ℝ := speed_faster_train_m_per_s - speed_slower_train_m_per_s

-- Define total distance to be covered
def total_distance : ℝ := 2 * length_of_train

-- Define expected time to pass
def expected_time : ℝ := 71.94 -- in seconds

-- Prove the statement
theorem faster_train_pass_time :
  (total_distance / relative_speed) = expected_time :=
sorry

end faster_train_pass_time_l342_342426


namespace constant_term_liam_polynomial_l342_342434

-- Define the polynomials p and q with the given properties
noncomputable def p : ℤ[X] := sorry
noncomputable def q : ℤ[X] := sorry

-- Assume the conditions
axiom h1 : p.natDegree = 3
axiom h2 : q.natDegree = 3
axiom h3 : p.leadingCoeff = 1
axiom h4 : q.leadingCoeff = 1
axiom h5 : (p * q) = (Polynomial.ofNatDegreeAndLCF 6 1).sumCoeffs [1, 4, 6, 6, 5, 8, 9]
axiom h6 : ∃ c : ℕ, c > 0 ∧ p.coeff 0 = c ∧ q.coeff 0 = c

-- The goal is to show that the constant term c is 3
theorem constant_term_liam_polynomial : ∃ c : ℕ, c = 3 ∧
  (∃ (p q : ℤ[X]), 
    p.natDegree = 3 ∧ q.natDegree = 3 ∧ 
    p.leadingCoeff = 1 ∧ q.leadingCoeff = 1 ∧ 
    p.coeff 0 = c ∧ q.coeff 0 = c ∧ 
    p * q = Polynomial.ofNatDegreeAndLCF 6 1).sumCoeffs [1, 4, 6, 6, 5, 8, 9] :=
sorry

end constant_term_liam_polynomial_l342_342434


namespace num_points_right_of_origin_l342_342397

theorem num_points_right_of_origin : 
  let exprs := [(-2 : ℤ)^3, (-3 : ℤ)^6, -((5 : ℤ)^2), 0, (x^2 + 9), (1 : ℤ)^2023]
  (finset.count (λ e, e > 0) (finset.of_list exprs) = 3) :=
by
  -- Definition of expressions
  let exprs := [(-2 : ℤ)^3, (-3 : ℤ)^6, -((5 : ℤ)^2), 0, (x^2 + 9), (1 : ℤ)^2023]
  -- We will use count to count the number of expressions greater than 0
  have num_positive := finset.count (λ e, e > 0) (finset.of_list exprs)
  -- Statement we are proving
  exact num_positive = 3

end num_points_right_of_origin_l342_342397


namespace area_of_frame_and_mirror_l342_342091

def dimensions_frame : ℕ × ℕ := (100, 120)
def frame_width : ℕ := 15
def dimensions_mirror (dw : ℕ) (dl : ℕ) : ℕ × ℕ := (dw - 2 * frame_width, dl - 2 * frame_width)

def area (d : ℕ × ℕ) : ℕ := d.1 * d.2

theorem area_of_frame_and_mirror
  (dw : ℕ) (dl : ℕ)
  (hw : dimensions_frame = (dw, dl)) :
  let mirror := dimensions_mirror dw dl in
  area mirror = 6300 ∧ area dimensions_frame - area mirror = 5700 :=
by
  sorry

end area_of_frame_and_mirror_l342_342091


namespace angle_A_gt_45_l342_342971

theorem angle_A_gt_45
  (ABC : Type)
  [Nonempty ABC]
  [IsTriangle ABC]
  {A B C D M H : ABC}
  (acute_angled_triangle : ∀x: ABC, ∠x < 90°)
  (is_concurrent : areConcurrent A D B M C H) : 
  ∠ A > 45° :=
  sorry

end angle_A_gt_45_l342_342971


namespace sum_of_first_14_terms_l342_342120

theorem sum_of_first_14_terms :
  (∑ n in Finset.range 14, (1 : ℝ) / ((n + 1 + 1) * (n + 2 + 1))) = (7 : ℝ) / 16 := by
-- Define S_n and a_n recursively
let S_n (n : ℕ) := (1 / 2 : ℝ) * n^2 + (3 / 2 : ℝ) * n
let a_n (n : ℕ) := if n = 1 then S_n 1 else S_n n - S_n (n - 1)

-- Use this to show the given sequence sum
let seq_term (n : ℕ) := 1 / (a_n n * a_n (n + 1))
sorry

end sum_of_first_14_terms_l342_342120


namespace area_of_triangle_AGE_l342_342802

theorem area_of_triangle_AGE :
  ∀ (A B C D E G : Point),
  square ABCD →
  E ∈ segment B C →
  B ≠ C → length (segment B E) = 2 → length (segment E C) = 3 →
  is_circumscribed E A B →
  intersects_circumscribed_circle (circumscribed_circle_triangle A B E) (diagonal B D) G →
  intersect_point_count (circumscribed_circle_triangle A B E) (diagonal B D) = 2 →
  area_triangle A G E = 62.5 := 
sorry

end area_of_triangle_AGE_l342_342802


namespace cookies_distribution_l342_342991

def smallest_odd_number_satisfying_condition (a: ℕ) (condition: ℕ → Prop) : ℕ :=
  Inf {x | x % 2 = 1 ∧ condition x ∧ a ≤ x}

theorem cookies_distribution : 
  ∃ (Chris Kenny Glenn Terry Dan Anne : ℕ),
    Glenn = 24 ∧
    Chris = Kenny / 3 ∧
    Glenn = 4 * Chris ∧
    Terry = ↑⌈(real.sqrt Glenn)⌉ + 3 ∧
    Dan = 2 * (Chris + Kenny) ∧
    Anne = Kenny / 2 ∧
    Anne ≥ 7 ∧
    Kenny = smallest_odd_number_satisfying_condition 7 (λ k => Chris = k / 3) ∧
    (Chris + Kenny + Glenn + Terry + Dan + Anne) = 113 :=
sorry

end cookies_distribution_l342_342991


namespace imaginary_part_of_complex_l342_342571

open Complex

theorem imaginary_part_of_complex (i : ℂ) (z : ℂ) (h1 : i^2 = -1) (h2 : z = (3 - 2 * i^3) / (1 + i)) : z.im = -1 / 2 :=
by {
  -- Proof would go here
  sorry
}

end imaginary_part_of_complex_l342_342571


namespace sqrt_three_irrational_l342_342879

theorem sqrt_three_irrational : ¬ (∃ (p q : ℤ), q ≠ 0 ∧ (sqrt 3 : ℝ) = p / q) := 
by
  sorry

end sqrt_three_irrational_l342_342879


namespace curve_c1_polar_eq_intersection_distance_l342_342188

-- Definitions for the curves
def C1_parametric (θ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos θ, 2 * Real.sin θ)
def C2_polar (θ : ℝ) : ℝ := Real.sqrt 3 * Real.sin θ + Real.cos θ
def C3_polar : ℝ := Real.pi / 6

-- Conversion to polar coordinates and finding distance
theorem curve_c1_polar_eq :
  ∃ ρ : ℝ, ρ^2 = 4 * Real.cos θ :=
sorry

theorem intersection_distance :
  let A : ℝ × ℝ := (4 * Real.cos (Real.pi / 6), Real.pi / 6)
  let B : ℝ × ℝ := (Real.sqrt 3, Real.pi / 6)
  |A.1 - B.1| = Real.sqrt 3 :=
sorry

end curve_c1_polar_eq_intersection_distance_l342_342188


namespace map_representation_l342_342252

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l342_342252


namespace regular_polygon_sides_l342_342945

theorem regular_polygon_sides (θ : ℝ) (n : ℕ) (h1 : θ = 18) (h2 : 360 = θ * n) : n = 20 :=
by {
  sorry
}

end regular_polygon_sides_l342_342945


namespace geometric_sequence_common_ratio_l342_342387

/-
Define the geometric sequence {4^(2n+1)} and the common ratio.
Prove that the common ratio of this sequence is 16.
-/
theorem geometric_sequence_common_ratio :
  let geom_seq (n : ℕ) := 4 ^ (2 * n + 1) in
  let q := geom_seq 2 / geom_seq 1 in
  q = 16 :=
by
  -- Define the sequence
  let geom_seq : ℕ → ℝ := λ n, 4 ^ (2 * n + 1)
  -- Calculate the common ratio
  let q := geom_seq 2 / geom_seq 1
  -- Prove the common ratio is 16
  sorry

end geometric_sequence_common_ratio_l342_342387


namespace regular_polygon_sides_l342_342935

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (k : ℕ), (k : ℕ) * 18 = 360) : n = 20 :=
by
  -- Proof body here
  sorry

end regular_polygon_sides_l342_342935


namespace quadratic_function_equal_values_l342_342591

theorem quadratic_function_equal_values (a m n : ℝ) (h : a ≠ 0) (hmn : a * m^2 - 4 * a * m - 3 = a * n^2 - 4 * a * n - 3) : m + n = 4 :=
by
  sorry

end quadratic_function_equal_values_l342_342591


namespace problem_l342_342656

noncomputable def key_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : Real.sqrt (x * y) ≤ 1) 
    : Prop := ∃ z : ℝ, 0 < z ∧ z = 2 * (x + y) / (x + y + 2)^2

theorem problem (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
    (habcd : a + b + c + d = 2) :
    (a^2 / (a^2 + 1)^2) + (b^2 / (b^2 + 1)^2) + (c^2 / (c^2 + 1)^2) + (d^2 / (d^2 + 1)^2) ≤ 16 / 25 := 
sorry

end problem_l342_342656


namespace sum_of_digits_7_pow_11_l342_342456

theorem sum_of_digits_7_pow_11 : 
  let n := 7 in
  let power := 11 in
  let last_two_digits := (n ^ power) % 100 in
  let tens_digit := last_two_digits / 10 in
  let ones_digit := last_two_digits % 10 in
  tens_digit + ones_digit = 7 :=
by {
  sorry
}

end sum_of_digits_7_pow_11_l342_342456


namespace Kishore_education_expense_l342_342529

theorem Kishore_education_expense
  (rent milk groceries petrol misc saved : ℝ) -- expenses
  (total_saved_salary : ℝ) -- percentage of saved salary
  (saving_amount : ℝ) -- actual saving
  (total_salary total_expense_children_education : ℝ) -- total salary and expense on children's education
  (H1 : rent = 5000)
  (H2 : milk = 1500)
  (H3 : groceries = 4500)
  (H4 : petrol = 2000)
  (H5 : misc = 3940)
  (H6 : total_saved_salary = 0.10)
  (H7 : saving_amount = 2160)
  (H8 : total_salary = saving_amount / total_saved_salary)
  (H9 : total_expense_children_education = total_salary - (rent + milk + groceries + petrol + misc) - saving_amount) :
  total_expense_children_education = 2600 :=
by 
  simp only [H1, H2, H3, H4, H5, H6, H7] at *
  norm_num at *
  sorry

end Kishore_education_expense_l342_342529


namespace n_minus_m_is_893_l342_342770

theorem n_minus_m_is_893 : 
  ∃ m n : ℕ, (m = Nat.find (λ k, k ≥ 100 ∧ k % 13 = 4) ∧ 
              n = Nat.find (λ k, k ≥ 1000 ∧ k % 13 = 4) ∧ 
              n - m = 893) :=
by
  let m := Nat.find (λ k, k ≥ 100 ∧ k % 13 = 4)
  let n := Nat.find (λ k, k ≥ 1000 ∧ k % 13 = 4)
  have h_m : m = 108 := sorry
  have h_n : n = 1001 := sorry
  exact ⟨m, n, h_m, h_n, by rw [h_m, h_n]; norm_num⟩

end n_minus_m_is_893_l342_342770


namespace samantha_sleeps_8_hours_per_night_l342_342814

variable {S : ℝ}

-- Given conditions as hypotheses
def baby_sister_sleeps := 2.5 * S
def father_sleeps := 0.5 * baby_sister_sleeps
def father_sleeps_per_week := father_sleeps * 7 = 70

-- The main theorem
theorem samantha_sleeps_8_hours_per_night (h1 : father_sleeps_per_week) : S = 8 := 
by
  sorry

end samantha_sleeps_8_hours_per_night_l342_342814


namespace painted_cubes_l342_342506

theorem painted_cubes (total_cube_size : ℕ) (unpainted_face_count : ℕ) (cut_size : ℕ) :
  total_cube_size = 4 →
  unpainted_face_count = 2 →
  cut_size = 1 →
  let total_small_cubes := (total_cube_size ^ 3) in
  let unpainted_cubes_per_face := (total_cube_size ^ 2) in
  let total_unpainted_cubes := unpainted_face_count * unpainted_cubes_per_face in
  let total_painted_cubes := total_small_cubes - total_unpainted_cubes in
  total_painted_cubes = 48 :=
by
  intros total_cube_size_eq unpainted_face_count_eq cut_size_eq
  let total_small_cubes := (total_cube_size ^ 3)
  let unpainted_cubes_per_face := (total_cube_size ^ 2)
  let total_unpainted_cubes := unpainted_face_count * unpainted_cubes_per_face
  let total_painted_cubes := total_small_cubes - total_unpainted_cubes
  have h1 : total_cube_size ^ 3 = 64 := by sorry -- proof omitted for brevity
  have h2 : unpainted_face_count * (total_cube_size ^ 2) = 32 := by sorry -- proof omitted for brevity
  have h3 : total_painted_cubes = 64 - 32 := by sorry -- proof omitted for brevity
  exact (48 : ℕ)

end painted_cubes_l342_342506


namespace integral_solution_l342_342075

noncomputable def integral_expression : Real → Real :=
  fun x => (1 + (x ^ (3 / 4))) ^ (4 / 5) / (x ^ (47 / 20))

theorem integral_solution :
  ∫ (x : Real), integral_expression x = - (20 / 27) * ((1 + (x ^ (3 / 4)) / (x ^ (3 / 4))) ^ (9 / 5)) + C := 
by 
  sorry

end integral_solution_l342_342075


namespace length_of_AB_l342_342702

theorem length_of_AB {A B C : Type} [Real] 
(H_angle_A : ∠A = 90)
(H_BC : BC = 20)
(H_tan_C : tan C = 2 * sin B) :
AB = 10 * sqrt 3 :=
by sorry

end length_of_AB_l342_342702


namespace minimum_cups_needed_l342_342503

theorem minimum_cups_needed (container_capacity cup_capacity : ℕ) (h1 : container_capacity = 980) (h2 : cup_capacity = 80) : 
  Nat.ceil (container_capacity / cup_capacity : ℚ) = 13 :=
by
  sorry

end minimum_cups_needed_l342_342503


namespace parity_of_magazines_and_celebrities_l342_342967

-- Define the main problem statement using Lean 4

theorem parity_of_magazines_and_celebrities {m c : ℕ}
  (h1 : ∀ i, i < m → ∃ d_i, d_i % 2 = 1)
  (h2 : ∀ j, j < c → ∃ e_j, e_j % 2 = 1) :
  (m % 2 = c % 2) ∧ (∃ ways, ways = 2 ^ ((m - 1) * (c - 1))) :=
by
  sorry

end parity_of_magazines_and_celebrities_l342_342967


namespace map_scale_l342_342269

theorem map_scale (represents_15cm_as_km : 15 * 6 = 90) :
  20 * 6 = 120 :=
by
  sorry

end map_scale_l342_342269


namespace angle_equality_l342_342782

variables {A B C D E F G H: Type}

structure Rectangle (A B C D : Type) : Prop :=
  (is_rectangle : true) -- Placeholder, define more rigorously if necessary

structure PerpendicularFoot (P Q R : Type) : Prop :=
  (is_perpendicular : true) -- Placeholder, define more rigorously if necessary

structure IntersectionPoint (P Q N : Type) : Prop :=
  (is_intersection : true) -- Placeholder, define more rigorously if necessary

axiom angle (A B C : Type) : Type

axiom eq_angle {A B C D : Type} : angle A B C = angle A B D → Prop

variables (ABCD : Rectangle A B C D)
variables (E : A) (F : B) (G : C) (H : D)
variables (hE : PerpendicularFoot A E B)
variables (hF : F ∈ line B D)
variables (hG : IntersectionPoint C F G)
variables (hH : IntersectionPoint B C H)

theorem angle_equality : eq_angle (angle E G B) (angle E H B) :=
by { sorry }

end angle_equality_l342_342782


namespace map_distance_l342_342341

theorem map_distance (fifteen_cm_in_km : ℤ) (cm_to_km : ℕ): 
  fifteen_cm_in_km = 90 ∧ cm_to_km = 6 → 20 * cm_to_km = 120 := 
by 
  sorry

end map_distance_l342_342341


namespace vectors_parallel_sum_l342_342150

theorem vectors_parallel_sum (x y : ℝ) :
  (∃ λ : ℝ, (-1, x, 3) = λ • (2, -4, y)) → x + y = -4 :=
by
  intro h
  sorry

end vectors_parallel_sum_l342_342150


namespace number_of_shirts_l342_342178

theorem number_of_shirts (ratio_pants_shirts: ℕ) (num_pants: ℕ) (S: ℕ) : 
  ratio_pants_shirts = 7 ∧ num_pants = 14 → S = 20 :=
by
  sorry

end number_of_shirts_l342_342178


namespace max_one_line_parallel_l342_342509

theorem max_one_line_parallel 
  (a : Line) (alpha : Plane) (n : ℕ) (lines_in_alpha : Fin n → Line)
  (h1 : a ∥ alpha) (h2 : ∃ p : Point, ∀ i, lines_in_alpha i ∋ p) :
  (∃ (i : Fin n), lines_in_alpha i ∥ a) ∧
  (∀ i j, lines_in_alpha i ∥ a ∧ lines_in_alpha j ∥ a → i = j) :=
by
  sorry

end max_one_line_parallel_l342_342509


namespace map_scale_representation_l342_342332

theorem map_scale_representation :
  (∀ len_km_1 len_cm_1 len_cm_2 : ℕ, len_km_1 = 90 → len_cm_1 = 15 → len_cm_2 = 20 →
    let scale := len_km_1 / len_cm_1 in
    len_cm_2 * scale = 120)
  := sorry

end map_scale_representation_l342_342332


namespace minimal_period_of_f_l342_342394

noncomputable def f (x : ℝ) : ℝ := (Real.tan x) / (1 + (Real.tan x)^2)

theorem minimal_period_of_f : Function.periodic f π :=
by
  sorry

end minimal_period_of_f_l342_342394


namespace angle_A1FB1_is_90_degrees_l342_342922

-- Define the problem setup and proof goal
theorem angle_A1FB1_is_90_degrees 
  (parabola : Set (ℝ × ℝ))
  (F : ℝ × ℝ)
  (A B : ℝ × ℝ)
  (A1 B1 : ℝ × ℝ)
  (is_parabola : ∀ P : ℝ × ℝ, P ∈ parabola ↔ dist P F = dist P (directrix_point parabola P))
  (A_in_parabola : A ∈ parabola)
  (B_in_parabola : B ∈ parabola)
  (line_FAB : ∃ L : Set (ℝ × ℝ), L = line_through F ∧ A ∈ L ∧ B ∈ L)
  (proj_A1 : A1 = projection_on_directrix parabola A)
  (proj_B1 : B1 = projection_on_directrix parabola B) :
  ∠ A1 F B1 = 90 := 
sorry

end angle_A1FB1_is_90_degrees_l342_342922


namespace sin_2phi_l342_342736

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_2phi_l342_342736


namespace map_length_represents_distance_l342_342317

theorem map_length_represents_distance :
  ∀ (s : ℝ), (15 * s = 90) → (20 * s = 120) :=
begin
  intros s h,
  sorry
end

end map_length_represents_distance_l342_342317


namespace map_representation_l342_342256

theorem map_representation (length_1_cm length_2_cm : ℕ) (distance_1_km : ℕ) :
  length_1_cm = 15 → distance_1_km = 90 →
  length_2_cm = 20 →
  distance_2_km = length_2_cm * (distance_1_km / length_1_cm) →
  distance_2_km = 120 :=
by
  intros h_len1 h_distance1 h_len2 h_calculate 
  rw [h_len1, h_distance1, h_len2] at h_calculate
  have scale := 90 / 15
  have distance_2_km := 20 * scale
  have : scale = 6 := by norm_num
  rw [this] at h_calculate
  norm_num at distance_2_km
  assumption
  sorry

end map_representation_l342_342256


namespace total_weight_correct_l342_342039

-- Definitions for the problem conditions
def jug1_capacity : ℝ := 2
def jug2_capacity : ℝ := 3
def jug3_capacity : ℝ := 4

def fill1 : ℝ := 0.7
def fill2 : ℝ := 0.6
def fill3 : ℝ := 0.5

def density1 : ℝ := 5
def density2 : ℝ := 4
def density3 : ℝ := 3

-- The weights of the sand in each jug
def weight1 : ℝ := fill1 * jug1_capacity * density1
def weight2 : ℝ := fill2 * jug2_capacity * density2
def weight3 : ℝ := fill3 * jug3_capacity * density3

-- The total weight of the sand in all jugs
def total_weight : ℝ := weight1 + weight2 + weight3

-- The proof statement
theorem total_weight_correct : total_weight = 20.2 := by
  sorry

end total_weight_correct_l342_342039


namespace sin_double_angle_l342_342719

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l342_342719


namespace quadratic_radicals_of_same_type_l342_342117

theorem quadratic_radicals_of_same_type (m : ℕ) : sqrt (m + 1) = sqrt 8 → m = 1 :=
by
  intro h
  have h₁ : sqrt 8 = 2 * sqrt 2, by sorry
  rw h₁ at h
  sorry

end quadratic_radicals_of_same_type_l342_342117


namespace triangle_probability_is_correct_l342_342098

noncomputable theory

def lengths : List ℕ := [1, 3, 5, 7, 9]

def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def probability_of_forming_triangle : ℚ :=
  let possible_combinations := lengths.combinations 3
  let valid_combinations := possible_combinations.filter (λ l, can_form_triangle l[0] l[1] l[2])
  (valid_combinations.length : ℚ) / (possible_combinations.length : ℚ)

theorem triangle_probability_is_correct :
  probability_of_forming_triangle = 0.3 := 
sorry

end triangle_probability_is_correct_l342_342098


namespace percent_problem_l342_342476

theorem percent_problem (x : ℝ) (hx : 0.60 * 600 = 0.50 * x) : x = 720 :=
by
  sorry

end percent_problem_l342_342476


namespace range_m_if_extremum_at_neg1_l342_342131

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - 3 * a * x - 1

theorem range_m_if_extremum_at_neg1 (a : ℝ) (h : a ≠ 0) (h_ext : f'.re f (-1) = 0) :
  set_of (λ m, ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 a = m ∧ f x2 a = m ∧ f x3 a = m) = set.interval (-3 : ℝ) (1 : ℝ) :=
sorry

end range_m_if_extremum_at_neg1_l342_342131


namespace sin_2phi_l342_342734

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_2phi_l342_342734


namespace shortBingoLastColumn_l342_342554

-- Define the set from which the numbers are chosen
def shortBingoSet : Finset ℕ := {51, 52, 53, 54, 55, 56, 57, 58, 59, 60}

-- Define the condition that there must be 5 distinct numbers chosen and order matters
def lastColumnPossibilities : ℕ := (shortBingoSet.card) * (shortBingoSet.card - 1) * (shortBingoSet.card - 2) * (shortBingoSet.card - 3) * (shortBingoSet.card - 4)

theorem shortBingoLastColumn :
  lastColumnPossibilities = 30240 :=
  by
    -- Define the cardinality of the set
    have h_card : shortBingoSet.card = 10 := by
      simp [shortBingoSet]

    -- Substitute the cardinality into the expression
    simp [lastColumnPossibilities, h_card]
    -- Performed arithmetic calculation
    norm_num
    done

end shortBingoLastColumn_l342_342554


namespace surface_area_approx_l342_342917

-- Define the problem's conditions
structure Cube :=
  (edge_length : ℝ)
  (A B C D A1 B1 C1 D1 F K : Point)

-- Define the cube configuration
structure CubeConfiguration (c : Cube) :=
  (midpoint_A1B1 : c.F = midpoint c.A1 c.B1)
  (center_A1B1C1D1 : c.K = center c.A1 c.B1 c.C1 c.D1)

-- Define the volume of the frustum
constant volume_frustum : Cube → ℝ 
constant surface_area_frustum : Cube → ℝ

-- Given the volume condition and edge length determined from solution
axiom volume_condition (c : Cube) (h : volume_frustum c = 13608)
    (edge_length_condition : c.edge_length = 36)

-- The main theorem statement
theorem surface_area_approx (c : Cube) (h : volume_frustum c = 13608)
    (edge_length_condition : c.edge_length = 36)
    (config : CubeConfiguration c) :
    |surface_area_frustum c - 4243| < 1 :=
  sorry

end surface_area_approx_l342_342917


namespace num_of_distinct_m_values_l342_342233

theorem num_of_distinct_m_values : 
  (∃ (x1 x2 : ℤ), x1 * x2 = 36 ∧ m = x1 + x2) → 
  (finset.card (finset.image (λ (p : ℤ × ℤ), p.1 + p.2) 
    {p | p.1 * p.2 = 36})) = 10 :=
sorry

end num_of_distinct_m_values_l342_342233


namespace claudia_initial_water_l342_342550

theorem claudia_initial_water :
  let water_5_oz := 6 * 5
  let water_8_oz := 4 * 8
  let water_4_oz := 15 * 4
  in water_5_oz + water_8_oz + water_4_oz = 122 :=
by 
  let water_5_oz := 6 * 5
  let water_8_oz := 4 * 8
  let water_4_oz := 15 * 4
  show water_5_oz + water_8_oz + water_4_oz = 122
  sorry

end claudia_initial_water_l342_342550


namespace candle_lighting_time_l342_342854

-- Define the main variables
variables {ℓ t t_s : ℝ}

-- Condition for burn rates and remaining length functions
def burn_rate_candle1 := ℓ / 240
def burn_rate_candle2 := ℓ / 300

def remaining_length_candle1 (t : ℝ) := ℓ - burn_rate_candle1 * t
def remaining_length_candle2 (t : ℝ) := ℓ - burn_rate_candle2 * t

-- Question and condition setup
theorem candle_lighting_time :
  ∀ ℓ t t_s, 
    (t = 300 - t_s) →
    (remaining_length_candle2 t = 3 * remaining_length_candle1 t) →
    t_s = 82 :=
begin
  intros,
  unfold burn_rate_candle1 burn_rate_candle2 remaining_length_candle1 remaining_length_candle2 at *,
  sorry
end

end candle_lighting_time_l342_342854


namespace sally_picked_peaches_l342_342813

variable (p_initial p_current p_picked : ℕ)

theorem sally_picked_peaches (h1 : p_initial = 13) (h2 : p_current = 55) :
  p_picked = p_current - p_initial → p_picked = 42 :=
by
  intros
  sorry

end sally_picked_peaches_l342_342813


namespace map_scale_l342_342356

namespace MapProblem

variables {c1 c2 k1 k2 : ℕ}

def representing_kilometers (c1 c2 k1 k2 : ℕ) : Prop :=
  (c1 * k2 = c2 * k1)

theorem map_scale :
  representing_kilometers 15 20 90 120 :=
by
  unfold representing_kilometers
  calc
    15 * 120 = 1800 := by norm_num
    20 * 90 = 1800 := by norm_num
  sorry

end MapProblem

end map_scale_l342_342356


namespace number_of_sides_l342_342956

-- Define the conditions as variables/constants
def exterior_angle (n : ℕ) : ℝ := 18         -- Each exterior angle is 18 degrees
def sum_of_exterior_angles : ℝ := 360        -- Sum of exterior angles of any polygon is 360 degrees

-- Prove the number of sides is equal to 20 given the conditions
theorem number_of_sides : 
  ∃ n : ℕ, (exterior_angle n) * (n : ℝ) = sum_of_exterior_angles → n = 20 := 
by
  sorry

end number_of_sides_l342_342956


namespace no_special_three_digit_couples_l342_342855

-- Define what it means to be a couple
def is_couple (x y : ℕ) : Prop :=
  let xs := x.digits 10 
  let ys := y.digits 10
  (xs.length = ys.length) ∧ (∀ i, i < xs.length → xs.nth i + ys.nth i = some 9)

-- Part (a): Couple that pairs with 2010
def couple_with_2010 : ℕ :=
  7989

-- Part (b): Number of two-digit couple pairs
def num_two_digit_couples : ℕ :=
  40

-- Define what it means to be a special couple
def is_special_couple (x y : ℕ) : Prop :=
  is_couple x y ∧ (x.digits 10).nodup ∧ (y.digits 10).nodup

-- Part (c): Example of special four-digit couples
def special_four_digit_couples : List (ℕ × ℕ) :=
  [(2376, 7623), (5814, 4185), (8901, 1098)]

-- Part (d): Proof that no special three-digit couple exists
theorem no_special_three_digit_couples (x y : ℕ) (h : is_special_couple x y) (hx : x < 1000) (hy : y < 1000)
  : false :=
sorry

end no_special_three_digit_couples_l342_342855


namespace map_length_scale_l342_342303

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end map_length_scale_l342_342303


namespace complex_solution_l342_342070

noncomputable def z1 : ℂ := 1.97 + 10.15 * Complex.i
noncomputable def z2 : ℂ := -1.97 - 10.15 * Complex.i

theorem complex_solution (z : ℂ) (h : z ^ 2 = -99 + 40 * Complex.i) : z = z1 ∨ z = z2 :=
by sorry

end complex_solution_l342_342070


namespace total_hours_worked_l342_342886

-- Definitions
def regular_rate : ℝ := 16
def overtime_rate : ℝ := regular_rate * 1.75
def weekly_earnings : ℝ := 760
def regular_hours : ℝ := 30
def earnings_from_regular_hours := regular_rate * regular_hours
def earnings_from_overtime := weekly_earnings - earnings_from_regular_hours
def overtime_hours := earnings_from_overtime / overtime_rate

-- Theorem statement
theorem total_hours_worked : regular_hours + overtime_hours = 40 := by
  sorry

end total_hours_worked_l342_342886


namespace football_team_lineup_count_l342_342364

theorem football_team_lineup_count :
  let team_members := 12
  let offensive_lineman_choices := 5
  let remaining_choices := team_members - 1
  let quarterback_choices := remaining_choices
  let running_back_choices := remaining_choices - 1
  let wide_receiver_choices := remaining_choices - 2
  let tight_end_choices := remaining_choices - 3

  offensive_lineman_choices * quarterback_choices * running_back_choices * wide_receiver_choices * tight_end_choices = 39600 :=
by
  let team_members := 12
  let offensive_lineman_choices := 5
  let remaining_choices := team_members - 1
  let quarterback_choices := remaining_choices
  let running_back_choices := remaining_choices - 1
  let wide_receiver_choices := remaining_choices - 2
  let tight_end_choices := remaining_choices - 3
  
  exact sorry

end football_team_lineup_count_l342_342364


namespace map_length_representation_l342_342313

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l342_342313


namespace calculate_new_volume_l342_342015

noncomputable def volume_of_sphere_with_increased_radius
  (initial_surface_area : ℝ) (radius_increase : ℝ) : ℝ :=
  (4 / 3) * Real.pi * ((Real.sqrt (initial_surface_area / (4 * Real.pi)) + radius_increase) ^ 3)

theorem calculate_new_volume :
  volume_of_sphere_with_increased_radius 400 (2) = 2304 * Real.pi :=
by
  sorry

end calculate_new_volume_l342_342015


namespace minimum_volume_sum_l342_342828

section pyramid_volume

variables {R : Type*} [OrderedRing R]
variables {V : Type*} [AddCommGroup V] [Module R V]

-- Define the volumes of the pyramids
variables (V_SABR1 V_SR2P2R3Q2 V_SCDR4 : R)
variables (V_SR1P1R2Q1 V_SR3P3R4Q3 : R)

-- Given condition
axiom volume_condition : V_SR1P1R2Q1 + V_SR3P3R4Q3 = 78

-- The theorem to be proved
theorem minimum_volume_sum : 
  V_SABR1^2 + V_SR2P2R3Q2^2 + V_SCDR4^2 ≥ 2028 :=
sorry

end pyramid_volume

end minimum_volume_sum_l342_342828


namespace percentage_decrease_hours_worked_l342_342750

theorem percentage_decrease_hours_worked (B H : ℝ) (h₁ : H > 0) (h₂ : B > 0)
  (h_assistant1 : (1.8 * B) = B * 1.8) (h_assistant2 : (2 * (B / H)) = (1.8 * B) / (0.9 * H)) : 
  ((H - (0.9 * H)) / H) * 100 = 10 := 
by
  sorry

end percentage_decrease_hours_worked_l342_342750
