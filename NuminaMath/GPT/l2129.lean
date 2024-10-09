import Mathlib

namespace confidence_k_squared_l2129_212951

-- Define the condition for 95% confidence relation between events A and B
def confidence_95 (A B : Prop) : Prop := 
  -- Placeholder for the actual definition, assume 95% confidence implies a specific condition
  True

-- Define the data value and critical value condition
def K_squared : ℝ := sorry  -- Placeholder for the actual K² value

theorem confidence_k_squared (A B : Prop) (h : confidence_95 A B) : K_squared > 3.841 := 
by
  sorry  -- Proof is not required, only the statement

end confidence_k_squared_l2129_212951


namespace find_y_l2129_212923

theorem find_y 
  (α : Real)
  (P : Real × Real)
  (P_coord : P = (-Real.sqrt 3, y))
  (sin_alpha : Real.sin α = Real.sqrt 13 / 13) :
  P.2 = 1 / 2 :=
by
  sorry

end find_y_l2129_212923


namespace average_age_across_rooms_l2129_212994

theorem average_age_across_rooms :
  let room_a_people := 8
  let room_a_average_age := 35
  let room_b_people := 5
  let room_b_average_age := 30
  let room_c_people := 7
  let room_c_average_age := 25
  let total_people := room_a_people + room_b_people + room_c_people
  let total_age := (room_a_people * room_a_average_age) + (room_b_people * room_b_average_age) + (room_c_people * room_c_average_age)
  let average_age := total_age / total_people
  average_age = 30.25 := by
{
  sorry
}

end average_age_across_rooms_l2129_212994


namespace seating_arrangement_l2129_212901

theorem seating_arrangement (M : ℕ) (h1 : 8 * M = 12 * M) : M = 3 :=
by
  sorry

end seating_arrangement_l2129_212901


namespace eccentricity_range_of_hyperbola_l2129_212955

open Real

noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

noncomputable def eccentricity_range :=
  ∀ (a b c : ℝ), 
    ∃ (e : ℝ),
      hyperbola a b (-c) 0 ∧ -- condition for point F
      (a + b > 0) ∧ -- additional conditions due to hyperbola properties
      (1 < e ∧ e < 2)
      
theorem eccentricity_range_of_hyperbola :
  eccentricity_range :=
by
  sorry

end eccentricity_range_of_hyperbola_l2129_212955


namespace largest_digit_to_correct_sum_l2129_212924

theorem largest_digit_to_correct_sum :
  (725 + 864 + 991 = 2570) → (∃ (d : ℕ), d = 9 ∧ 
  (∃ (n1 : ℕ), n1 ∈ [702, 710, 711, 721, 715] ∧ 
  ∃ (n2 : ℕ), n2 ∈ [806, 805, 814, 854, 864] ∧ 
  ∃ (n3 : ℕ), n3 ∈ [918, 921, 931, 941, 981, 991] ∧ 
  n1 + n2 + n3 = n1 + n2 + n3 - 10))
    → d = 9 :=
by
  sorry

end largest_digit_to_correct_sum_l2129_212924


namespace tiles_difference_eighth_sixth_l2129_212973

-- Define the side length of the nth square
def side_length (n : ℕ) : ℕ := n

-- Define the number of tiles given the side length
def number_of_tiles (n : ℕ) : ℕ := n * n

-- State the theorem about the difference in tiles between the 8th and 6th squares
theorem tiles_difference_eighth_sixth :
  number_of_tiles (side_length 8) - number_of_tiles (side_length 6) = 28 :=
by
  -- skipping the proof
  sorry

end tiles_difference_eighth_sixth_l2129_212973


namespace perfect_square_if_integer_l2129_212945

theorem perfect_square_if_integer (n : ℤ) (k : ℤ) 
  (h : k = 2 + 2 * Int.sqrt (28 * n^2 + 1)) : ∃ m : ℤ, k = m^2 :=
by 
  sorry

end perfect_square_if_integer_l2129_212945


namespace theta_in_first_quadrant_l2129_212938

noncomputable def quadrant_of_theta (theta : ℝ) (h1 : Real.sin (Real.pi + theta) < 0) (h2 : Real.cos (Real.pi - theta) < 0) : ℕ :=
  if 0 < Real.sin theta ∧ 0 < Real.cos theta then 1 else sorry

theorem theta_in_first_quadrant (theta : ℝ) (h1 : Real.sin (Real.pi + theta) < 0) (h2 : Real.cos (Real.pi - theta) < 0) :
  quadrant_of_theta theta h1 h2 = 1 :=
by
  sorry

end theta_in_first_quadrant_l2129_212938


namespace solve_eq1_solve_eq2_l2129_212965

-- Define the problem for equation (1)
theorem solve_eq1 (x : Real) : (x - 1)^2 = 2 ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by 
  sorry

-- Define the problem for equation (2)
theorem solve_eq2 (x : Real) : x^2 - 6 * x - 7 = 0 ↔ (x = -1 ∨ x = 7) :=
by 
  sorry

end solve_eq1_solve_eq2_l2129_212965


namespace points_distance_within_rectangle_l2129_212980

theorem points_distance_within_rectangle :
  ∀ (points : Fin 6 → (ℝ × ℝ)), (∀ i, 0 ≤ (points i).1 ∧ (points i).1 ≤ 3 ∧ 0 ≤ (points i).2 ∧ (points i).2 ≤ 4) →
  ∃ (i j : Fin 6), i ≠ j ∧ dist (points i) (points j) ≤ Real.sqrt 2 :=
by
  sorry

end points_distance_within_rectangle_l2129_212980


namespace pqrs_sum_l2129_212978

noncomputable def distinct_real_numbers (p q r s : ℝ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

theorem pqrs_sum (p q r s : ℝ) 
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s)
  (distinct : distinct_real_numbers p q r s) :
  p + q + r + s = -13 :=
  sorry

end pqrs_sum_l2129_212978


namespace john_bought_notebooks_l2129_212950

def pages_per_notebook : ℕ := 40
def pages_per_day : ℕ := 4
def total_days : ℕ := 50

theorem john_bought_notebooks : (pages_per_day * total_days) / pages_per_notebook = 5 :=
by
  sorry

end john_bought_notebooks_l2129_212950


namespace other_endpoint_of_diameter_l2129_212907

theorem other_endpoint_of_diameter (center endpoint : ℝ × ℝ) (hc : center = (1, 2)) (he : endpoint = (4, 6)) :
    ∃ other_endpoint : ℝ × ℝ, other_endpoint = (-2, -2) :=
by
  sorry

end other_endpoint_of_diameter_l2129_212907


namespace prime_sum_eq_14_l2129_212925

theorem prime_sum_eq_14 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : q^5 - 2 * p^2 = 1) : p + q = 14 := 
sorry

end prime_sum_eq_14_l2129_212925


namespace work_together_l2129_212977

theorem work_together (W : ℝ) (Dx Dy : ℝ) (hx : Dx = 15) (hy : Dy = 30) : 
  (Dx * Dy) / (Dx + Dy) = 10 := 
by
  sorry

end work_together_l2129_212977


namespace election_winner_votes_l2129_212947

variable (V : ℝ) (winner_votes : ℝ) (winner_margin : ℝ)
variable (condition1 : V > 0)
variable (condition2 : winner_votes = 0.60 * V)
variable (condition3 : winner_margin = 240)

theorem election_winner_votes (h : winner_votes - 0.40 * V = winner_margin) : winner_votes = 720 := by
  sorry

end election_winner_votes_l2129_212947


namespace quadratic_inequality_l2129_212954

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality
  (a b c : ℝ)
  (h_pos : 0 < a)
  (h_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0)
  (x : ℝ) :
  f a b c x + f a b c (x - 1) - f a b c (x + 1) > -4 * a :=
  sorry

end quadratic_inequality_l2129_212954


namespace family_age_problem_l2129_212993

theorem family_age_problem (T y : ℕ)
  (h1 : T = 5 * 17)
  (h2 : (T + 5 * y + 2) = 6 * 17)
  : y = 3 := by
  sorry

end family_age_problem_l2129_212993


namespace multiples_of_7_between_50_and_200_l2129_212984

theorem multiples_of_7_between_50_and_200 : 
  ∃ n, n = 21 ∧ ∀ k, (k ≥ 50 ∧ k ≤ 200) ↔ ∃ m, k = 7 * m := sorry

end multiples_of_7_between_50_and_200_l2129_212984


namespace percentage_proof_l2129_212904

theorem percentage_proof (a : ℝ) (paise : ℝ) (x : ℝ) (h1: paise = 85) (h2: a = 170) : 
  (x/100) * a = paise ↔ x = 50 := 
by
  -- The setup includes:
  -- paise = 85
  -- a = 170
  -- We prove that x% of 170 equals 85 if and only if x = 50.
  sorry

end percentage_proof_l2129_212904


namespace solve_system_of_equations_solve_algebraic_equation_l2129_212987

-- Problem 1: System of Equations
theorem solve_system_of_equations (x y : ℝ) (h1 : x + 2 * y = 3) (h2 : 2 * x - y = 1) : x = 1 ∧ y = 1 :=
sorry

-- Problem 2: Algebraic Equation
theorem solve_algebraic_equation (x : ℝ) (h : 1 / (x - 1) + 2 = 5 / (1 - x)) : x = -2 :=
sorry

end solve_system_of_equations_solve_algebraic_equation_l2129_212987


namespace minimum_value_l2129_212928

open Real

theorem minimum_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 9) :
  (x ^ 2 + y ^ 2) / (x + y) + (x ^ 2 + z ^ 2) / (x + z) + (y ^ 2 + z ^ 2) / (y + z) ≥ 9 :=
by sorry

end minimum_value_l2129_212928


namespace kate_hair_length_l2129_212962

theorem kate_hair_length :
  ∀ (logans_hair : ℕ) (emilys_hair : ℕ) (kates_hair : ℕ),
  logans_hair = 20 →
  emilys_hair = logans_hair + 6 →
  kates_hair = emilys_hair / 2 →
  kates_hair = 13 :=
by
  intros logans_hair emilys_hair kates_hair
  sorry

end kate_hair_length_l2129_212962


namespace sum_of_smallest_and_largest_l2129_212944

theorem sum_of_smallest_and_largest (n : ℕ) (h : Odd n) (b z : ℤ)
  (h_mean : z = b + n - 1 - 2 / (n : ℤ)) :
  ((b - 2) + (b + 2 * (n - 2))) = 2 * z - 4 + 4 / (n : ℤ) :=
by
  sorry

end sum_of_smallest_and_largest_l2129_212944


namespace cosine_of_3pi_over_2_l2129_212921

theorem cosine_of_3pi_over_2 : Real.cos (3 * Real.pi / 2) = 0 := by
  sorry

end cosine_of_3pi_over_2_l2129_212921


namespace interest_rate_10_percent_l2129_212996

-- Definitions for the problem
variables (P : ℝ) (R : ℝ) (T : ℝ)

-- Condition that the money doubles in 10 years on simple interest
def money_doubles_in_10_years (P R : ℝ) : Prop :=
  P = (P * R * 10) / 100

-- Statement that R is 10% if the money doubles in 10 years
theorem interest_rate_10_percent {P : ℝ} (h : money_doubles_in_10_years P R) : R = 10 :=
by
  sorry

end interest_rate_10_percent_l2129_212996


namespace wade_customers_l2129_212932

theorem wade_customers (F : ℕ) (h1 : 2 * F + 6 * F + 72 = 296) : F = 28 := 
by 
  sorry

end wade_customers_l2129_212932


namespace projectile_reaches_100_feet_l2129_212958

noncomputable def projectile_height (t : ℝ) : ℝ :=
  -16 * t^2 + 80 * t

theorem projectile_reaches_100_feet :
  ∃ t : ℝ, t = 2.5 ∧ projectile_height t = 100 :=
by
  use 2.5
  sorry

end projectile_reaches_100_feet_l2129_212958


namespace circle_ways_l2129_212998

noncomputable def count3ConsecutiveCircles : ℕ :=
  let longSideWays := 1 + 2 + 3 + 4 + 5 + 6
  let perpendicularWays := (4 + 4 + 4 + 3 + 2 + 1) * 2
  longSideWays + perpendicularWays

theorem circle_ways : count3ConsecutiveCircles = 57 := by
  sorry

end circle_ways_l2129_212998


namespace morleys_theorem_l2129_212991

def is_trisector (A B C : Point) (p : Point) : Prop :=
sorry -- Definition that this point p is on one of the trisectors of ∠BAC

def triangle (A B C : Point) : Prop :=
sorry -- Definition that points A, B, C form a triangle

def equilateral (A B C : Point) : Prop :=
sorry -- Definition that triangle ABC is equilateral

theorem morleys_theorem (A B C D E F : Point)
  (hABC : triangle A B C)
  (hD : is_trisector A B C D)
  (hE : is_trisector B C A E)
  (hF : is_trisector C A B F) :
  equilateral D E F :=
sorry

end morleys_theorem_l2129_212991


namespace rest_area_milepost_l2129_212935

theorem rest_area_milepost 
  (milepost_fifth_exit : ℕ) 
  (milepost_fifteenth_exit : ℕ) 
  (rest_area_milepost : ℕ)
  (h1 : milepost_fifth_exit = 50)
  (h2 : milepost_fifteenth_exit = 350)
  (h3 : rest_area_milepost = (milepost_fifth_exit + (milepost_fifteenth_exit - milepost_fifth_exit) / 2)) :
  rest_area_milepost = 200 := 
by
  intros
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end rest_area_milepost_l2129_212935


namespace dog_cat_food_difference_l2129_212912

theorem dog_cat_food_difference :
  let dogFood := 600
  let catFood := 327
  dogFood - catFood = 273 :=
by
  let dogFood := 600
  let catFood := 327
  show dogFood - catFood = 273
  sorry

end dog_cat_food_difference_l2129_212912


namespace opposite_sides_range_l2129_212948

theorem opposite_sides_range (a : ℝ) :
  (3 * (-3) - 2 * (-1) - a) * (3 * 4 - 2 * (-6) - a) < 0 ↔ -7 < a ∧ a < 24 :=
by
  simp
  sorry

end opposite_sides_range_l2129_212948


namespace axisymmetric_triangle_is_isosceles_l2129_212976

-- Define a triangle and its properties
structure Triangle :=
  (a b c : ℝ) -- Triangle sides as real numbers
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)

def is_axisymmetric (T : Triangle) : Prop :=
  -- Here define what it means for a triangle to be axisymmetric
  -- This is often represented as having at least two sides equal
  (T.a = T.b ∨ T.b = T.c ∨ T.c = T.a)

def is_isosceles (T : Triangle) : Prop :=
  -- Definition of an isosceles triangle
  (T.a = T.b ∨ T.b = T.c ∨ T.c = T.a)

-- The theorem to be proven
theorem axisymmetric_triangle_is_isosceles (T : Triangle) (h : is_axisymmetric T) : is_isosceles T :=
by {
  -- Proof would go here
  sorry
}

end axisymmetric_triangle_is_isosceles_l2129_212976


namespace sandy_initial_payment_l2129_212961

variable (P : ℝ) 

theorem sandy_initial_payment
  (h1 : (1.2 : ℝ) * (P + 200) = 1200) :
  P = 800 :=
by
  -- Proof goes here
  sorry

end sandy_initial_payment_l2129_212961


namespace dvds_bought_online_l2129_212981

theorem dvds_bought_online (total_dvds : ℕ) (store_dvds : ℕ) (online_dvds : ℕ) :
  total_dvds = 10 → store_dvds = 8 → online_dvds = total_dvds - store_dvds → online_dvds = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end dvds_bought_online_l2129_212981


namespace houses_built_during_boom_l2129_212906

theorem houses_built_during_boom :
  let original_houses := 20817
  let current_houses := 118558
  let houses_built := current_houses - original_houses
  houses_built = 97741 := by
  sorry

end houses_built_during_boom_l2129_212906


namespace intersection_of_A_and_B_l2129_212933

def setA : Set ℝ := {x | abs (x - 1) < 2}

def setB : Set ℝ := {x | x^2 + x - 2 > 0}

theorem intersection_of_A_and_B :
  (setA ∩ setB) = {x | 1 < x ∧ x < 3} :=
sorry

end intersection_of_A_and_B_l2129_212933


namespace problem_statement_l2129_212917

def class_of_rem (k : ℕ) : Set ℤ := {n | ∃ m : ℤ, n = 4 * m + k}

theorem problem_statement : (2013 ∈ class_of_rem 1) ∧ 
                            (-2 ∈ class_of_rem 2) ∧ 
                            (∀ x : ℤ, x ∈ class_of_rem 0 ∨ x ∈ class_of_rem 1 ∨ x ∈ class_of_rem 2 ∨ x ∈ class_of_rem 3) ∧ 
                            (∀ a b : ℤ, (∃ k : ℕ, (a ∈ class_of_rem k ∧ b ∈ class_of_rem k)) ↔ (a - b) ∈ class_of_rem 0) :=
by
  -- each of the statements should hold true
  sorry

end problem_statement_l2129_212917


namespace sequence_item_l2129_212972

theorem sequence_item (n : ℕ) (a_n : ℕ → Rat) (h : a_n n = 2 / (n^2 + n)) : a_n n = 1 / 15 → n = 5 := by
  sorry

end sequence_item_l2129_212972


namespace quadratic_equation_has_root_l2129_212937

theorem quadratic_equation_has_root (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) :
  ∃ (x : ℝ), (a * x^2 + 2 * b * x + c = 0) ∨
             (b * x^2 + 2 * c * x + a = 0) ∨
             (c * x^2 + 2 * a * x + b = 0) :=
sorry

end quadratic_equation_has_root_l2129_212937


namespace profit_percentage_l2129_212941

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 500) (hSP : SP = 725) : 
  100 * (SP - CP) / CP = 45 :=
by
  sorry

end profit_percentage_l2129_212941


namespace compute_g_neg_101_l2129_212995

noncomputable def g (x : ℝ) : ℝ := sorry

theorem compute_g_neg_101 (g_condition : ∀ x y : ℝ, g (x * y) + x = x * g y + g x)
                         (g1 : g 1 = 7) :
    g (-101) = -95 := 
by 
  sorry

end compute_g_neg_101_l2129_212995


namespace selection_methods_eq_total_students_l2129_212997

def num_boys := 36
def num_girls := 28
def total_students : ℕ := num_boys + num_girls

theorem selection_methods_eq_total_students :
    total_students = 64 :=
by
  -- Placeholder for the proof
  sorry

end selection_methods_eq_total_students_l2129_212997


namespace stock_percent_change_l2129_212929

variable (x : ℝ)

theorem stock_percent_change (h1 : ∀ x, 0.75 * x = x * 0.75)
                             (h2 : ∀ x, 1.05 * x = 0.75 * x + 0.3 * 0.75 * x):
    ((1.05 * x - x) / x) * 100 = 5 :=
by
  sorry

end stock_percent_change_l2129_212929


namespace total_fence_poles_l2129_212931

def num_poles_per_side : ℕ := 27
def sides_of_square : ℕ := 4
def corners_of_square : ℕ := 4

theorem total_fence_poles : 
  (num_poles_per_side * sides_of_square) - corners_of_square = 104 :=
  sorry

end total_fence_poles_l2129_212931


namespace problem_l2129_212949

theorem problem (n : ℝ) (h : (n - 2009)^2 + (2008 - n)^2 = 1) : (n - 2009) * (2008 - n) = 0 := 
by
  sorry

end problem_l2129_212949


namespace point_of_tangency_l2129_212930

theorem point_of_tangency (x y : ℝ) (h : (y = x^3 + x - 2)) (slope : 4 = 3 * x^2 + 1) : (x, y) = (-1, -4) := 
sorry

end point_of_tangency_l2129_212930


namespace domain_of_f_l2129_212911

noncomputable def f (x : ℝ) : ℝ := 1 / Real.log x + Real.sqrt (2 - x)

theorem domain_of_f :
  { x : ℝ | 0 < x ∧ x ≤ 2 ∧ x ≠ 1 } = { x : ℝ | (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2) } :=
by
  sorry

end domain_of_f_l2129_212911


namespace value_a_plus_c_l2129_212920

noncomputable def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c
noncomputable def g (a b c : ℝ) (x : ℝ) := c * x^2 + b * x + a

theorem value_a_plus_c (a b c : ℝ) (h : ∀ x : ℝ, f a b c (g a b c x) = x) : a + c = -1 :=
by
  sorry

end value_a_plus_c_l2129_212920


namespace find_certain_number_l2129_212936

theorem find_certain_number (x : ℝ) : 136 - 0.35 * x = 31 -> x = 300 :=
by
  intro h
  sorry

end find_certain_number_l2129_212936


namespace base_conversion_problem_l2129_212968

theorem base_conversion_problem (b : ℕ) (h : b^2 + b + 3 = 34) : b = 6 :=
sorry

end base_conversion_problem_l2129_212968


namespace total_books_l2129_212985

noncomputable def num_books_on_shelf : ℕ := 8

theorem total_books (p h s : ℕ) (assump1 : p = 2) (assump2 : h = 6) (assump3 : s = 36) :
  p + h = num_books_on_shelf :=
by {
  -- leaving the proof construction out as per instructions
  sorry
}

end total_books_l2129_212985


namespace find_possible_values_of_n_l2129_212979

theorem find_possible_values_of_n (n : ℕ) :
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n + 1 ∧ 
    (2*n*(2*n + 1))/2 - (n*k + (n*(n-1))/2) = 1615) ↔ (n = 34 ∨ n = 38) :=
by
  sorry

end find_possible_values_of_n_l2129_212979


namespace binary_to_decimal_conversion_l2129_212942

theorem binary_to_decimal_conversion : (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 51 :=
by 
  sorry

end binary_to_decimal_conversion_l2129_212942


namespace calculate_final_amount_l2129_212914

def initial_amount : ℝ := 7500
def first_year_rate : ℝ := 0.20
def second_year_rate : ℝ := 0.25

def first_year_interest (p : ℝ) (r : ℝ) : ℝ := p * r
def amount_after_first_year (p : ℝ) (i : ℝ) : ℝ := p + i

def second_year_interest (p : ℝ) (r : ℝ) : ℝ := p * r
def amount_after_second_year (p : ℝ) (i : ℝ) : ℝ := p + i

theorem calculate_final_amount :
  let initial : ℝ := initial_amount
  let interest1 : ℝ := first_year_interest initial first_year_rate
  let amount1 : ℝ := amount_after_first_year initial interest1
  let interest2 : ℝ := second_year_interest amount1 second_year_rate
  let final_amount : ℝ := amount_after_second_year amount1 interest2
  final_amount = 11250 := by
  sorry

end calculate_final_amount_l2129_212914


namespace calculate_fraction_l2129_212908

theorem calculate_fraction : (1 / (1 + 1 / (4 + 1 / 5))) = (21 / 26) :=
by
  sorry

end calculate_fraction_l2129_212908


namespace pump_leak_drain_time_l2129_212916

theorem pump_leak_drain_time {P L : ℝ} (hP : P = 0.25) (hPL : P - L = 0.05) : (1 / L) = 5 :=
by sorry

end pump_leak_drain_time_l2129_212916


namespace sin_square_eq_c_div_a2_plus_b2_l2129_212910

theorem sin_square_eq_c_div_a2_plus_b2 
  (a b c : ℝ) (α β : ℝ)
  (h1 : a * Real.cos α + b * Real.sin α = c)
  (h2 : a * Real.cos β + b * Real.sin β = 0)
  (not_all_zero : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  Real.sin (α - β) ^ 2 = c ^ 2 / (a ^ 2 + b ^ 2) :=
by
  sorry

end sin_square_eq_c_div_a2_plus_b2_l2129_212910


namespace distance_to_focus_l2129_212999

open Real

theorem distance_to_focus {P : ℝ × ℝ} 
  (h₁ : P.2 ^ 2 = 4 * P.1)
  (h₂ : abs (P.1 + 3) = 5) :
  dist P ⟨1, 0⟩ = 3 := 
sorry

end distance_to_focus_l2129_212999


namespace range_of_a_l2129_212953

-- Defining the function f
noncomputable def f (x a : ℝ) : ℝ :=
  (Real.exp x) * (2 * x - 1) - a * x + a

-- Main statement
theorem range_of_a (a : ℝ)
  (h1 : a < 1)
  (h2 : ∃ x0 x1 : ℤ, x0 ≠ x1 ∧ f x0 a ≤ 0 ∧ f x1 a ≤ 0) :
  (5 / (3 * Real.exp 2)) < a ∧ a ≤ (3 / (2 * Real.exp 1)) :=
sorry

end range_of_a_l2129_212953


namespace fill_cistern_time_l2129_212963

theorem fill_cistern_time (R1 R2 R3 : ℝ) (H1 : R1 = 1/10) (H2 : R2 = 1/12) (H3 : R3 = 1/40) : 
  (1 / (R1 + R2 - R3)) = (120 / 19) :=
by
  sorry

end fill_cistern_time_l2129_212963


namespace seeds_distributed_equally_l2129_212975

theorem seeds_distributed_equally (S G n seeds_per_small_garden : ℕ) 
  (hS : S = 42) 
  (hG : G = 36) 
  (hn : n = 3) 
  (h_seeds : seeds_per_small_garden = (S - G) / n) : 
  seeds_per_small_garden = 2 := by
  rw [hS, hG, hn] at h_seeds
  simp at h_seeds
  exact h_seeds

end seeds_distributed_equally_l2129_212975


namespace metal_contest_winner_l2129_212990

theorem metal_contest_winner (x y : ℕ) (hx : 95 * x + 74 * y = 2831) : x = 15 ∧ y = 19 ∧ 95 * 15 > 74 * 19 := by
  sorry

end metal_contest_winner_l2129_212990


namespace maria_workers_problem_l2129_212983

-- Define the initial conditions
def initial_days : ℕ := 40
def days_passed : ℕ := 10
def fraction_completed : ℚ := 2/5
def initial_workers : ℕ := 10

-- Define the required minimum number of workers to complete the job on time
def minimum_workers_required : ℕ := 5

-- The theorem statement
theorem maria_workers_problem 
  (initial_days : ℕ)
  (days_passed : ℕ)
  (fraction_completed : ℚ)
  (initial_workers : ℕ) :
  ( ∀ (total_days remaining_days : ℕ), 
    initial_days = 40 ∧ days_passed = 10 ∧ fraction_completed = 2/5 ∧ initial_workers = 10 → 
    remaining_days = initial_days - days_passed ∧ 
    total_days = initial_days ∧ 
    fraction_completed + (remaining_days / total_days) = 1) →
  minimum_workers_required = 5 := 
sorry

end maria_workers_problem_l2129_212983


namespace count_divisible_by_3_in_range_l2129_212969

theorem count_divisible_by_3_in_range (a b : ℤ) :
  a = 252 → b = 549 → (∃ n : ℕ, (a ≤ 3 * n ∧ 3 * n ≤ b) ∧ (b - a) / 3 = (100 : ℝ)) :=
by
  intros ha hb
  have h1 : ∃ k : ℕ, a = 3 * k := by sorry
  have h2 : ∃ m : ℕ, b = 3 * m := by sorry
  sorry

end count_divisible_by_3_in_range_l2129_212969


namespace amount_left_for_gas_and_maintenance_l2129_212926

def monthly_income : ℤ := 3200
def rent : ℤ := 1250
def utilities : ℤ := 150
def retirement_savings : ℤ := 400
def groceries_eating_out : ℤ := 300
def insurance : ℤ := 200
def miscellaneous : ℤ := 200
def car_payment : ℤ := 350

def total_expenses : ℤ :=
  rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous + car_payment

theorem amount_left_for_gas_and_maintenance : monthly_income - total_expenses = 350 :=
by
  -- Proof is omitted
  sorry

end amount_left_for_gas_and_maintenance_l2129_212926


namespace segments_have_common_point_l2129_212967

-- Define the predicate that checks if two segments intersect
def segments_intersect (seg1 seg2 : ℝ × ℝ) : Prop :=
  let (a1, b1) := seg1
  let (a2, b2) := seg2
  max a1 a2 ≤ min b1 b2

-- Define the main theorem
theorem segments_have_common_point (segments : Fin 2019 → ℝ × ℝ)
  (h_intersect : ∀ (i j : Fin 2019), i ≠ j → segments_intersect (segments i) (segments j)) :
  ∃ p : ℝ, ∀ i : Fin 2019, (segments i).1 ≤ p ∧ p ≤ (segments i).2 :=
by
  sorry

end segments_have_common_point_l2129_212967


namespace infinite_series_sum_l2129_212919

theorem infinite_series_sum :
  ∑' (n : ℕ), (1 / (1 + 3^n : ℝ) - 1 / (1 + 3^(n+1) : ℝ)) = 1/2 := 
sorry

end infinite_series_sum_l2129_212919


namespace find_x_intercept_l2129_212956

variables (a x y : ℝ)
def l1 (a x y : ℝ) : Prop := (a + 2) * x + 3 * y = 5
def l2 (a x y : ℝ) : Prop := (a - 1) * x + 2 * y = 6
def are_parallel (a : ℝ) : Prop := (- (a + 2) / 3) = (- (a - 1) / 2)
def x_intercept_of_l1 (a x : ℝ) : Prop := l1 a x 0

theorem find_x_intercept (h : are_parallel a) : x_intercept_of_l1 7 (5 / 9) := 
sorry

end find_x_intercept_l2129_212956


namespace average_increase_l2129_212970

variable (A : ℕ) -- The batsman's average before the 17th inning
variable (runs_in_17th_inning : ℕ := 86) -- Runs made in the 17th inning
variable (new_average : ℕ := 38) -- The average after the 17th inning
variable (total_runs_16_innings : ℕ := 16 * A) -- Total runs after 16 innings
variable (total_runs_after_17_innings : ℕ := total_runs_16_innings + runs_in_17th_inning) -- Total runs after 17 innings
variable (total_runs_should_be : ℕ := 17 * new_average) -- Theoretical total runs after 17 innings

theorem average_increase :
  total_runs_after_17_innings = total_runs_should_be → (new_average - A) = 3 :=
by
  sorry

end average_increase_l2129_212970


namespace log_eq_condition_pq_l2129_212986

theorem log_eq_condition_pq :
  ∀ (p q : ℝ), p > 0 → q > 0 → (Real.log p + Real.log q = Real.log (2 * p + q)) → p = 3 ∧ q = 3 :=
by
  intros p q hp hq hlog
  sorry

end log_eq_condition_pq_l2129_212986


namespace transform_into_product_l2129_212909

theorem transform_into_product : 447 * (Real.sin (75 * Real.pi / 180) + Real.sin (15 * Real.pi / 180)) = 447 * Real.sqrt 6 / 2 := by
  sorry

end transform_into_product_l2129_212909


namespace remainder_of_4521_l2129_212943

theorem remainder_of_4521 (h1 : ∃ d : ℕ, d = 88)
  (h2 : 3815 % 88 = 31) : 4521 % 88 = 33 :=
sorry

end remainder_of_4521_l2129_212943


namespace youtube_more_than_tiktok_l2129_212971

-- Definitions for followers in different social media platforms
def instagram_followers : ℕ := 240
def facebook_followers : ℕ := 500
def total_followers : ℕ := 3840

-- Number of followers on Twitter is half the sum of followers on Instagram and Facebook
def twitter_followers : ℕ := (instagram_followers + facebook_followers) / 2

-- Number of followers on TikTok is 3 times the followers on Twitter
def tiktok_followers : ℕ := 3 * twitter_followers

-- Calculate the number of followers on all social media except YouTube
def other_followers : ℕ := instagram_followers + facebook_followers + twitter_followers + tiktok_followers

-- Number of followers on YouTube
def youtube_followers : ℕ := total_followers - other_followers

-- Prove the number of followers on YouTube is greater than TikTok by a certain amount
theorem youtube_more_than_tiktok : youtube_followers - tiktok_followers = 510 := by
  -- Sorry is a placeholder for the proof
  sorry

end youtube_more_than_tiktok_l2129_212971


namespace cylinder_ratio_l2129_212940

theorem cylinder_ratio
  (V : ℝ) (R H : ℝ)
  (hV : V = 1000)
  (hVolume : π * R^2 * H = V) :
  H / R = 1 :=
by
  sorry

end cylinder_ratio_l2129_212940


namespace derivative_at_one_l2129_212913

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x

theorem derivative_at_one : deriv f 1 = 2 + Real.exp 1 := by
  sorry

end derivative_at_one_l2129_212913


namespace cos_value_proof_l2129_212957

variable (α : Real)
variable (h1 : -Real.pi / 2 < α ∧ α < 0)
variable (h2 : Real.sin (α + Real.pi / 3) + Real.sin α = -(4 * Real.sqrt 3) / 5)

theorem cos_value_proof : Real.cos (α + 2 * Real.pi / 3) = 4 / 5 :=
by
  sorry

end cos_value_proof_l2129_212957


namespace Cody_total_bill_l2129_212902

-- Definitions for the problem
def cost_per_child : ℝ := 7.5
def cost_per_adult : ℝ := 12.0

variables (A C : ℕ)

-- Conditions
def condition1 : Prop := C = A + 8
def condition2 : Prop := A + C = 12

-- Total bill
def total_cost := (A * cost_per_adult) + (C * cost_per_child)

-- The proof statement
theorem Cody_total_bill (h1 : condition1 A C) (h2 : condition2 A C) : total_cost A C = 99.0 := by
  sorry

end Cody_total_bill_l2129_212902


namespace choir_average_age_l2129_212988

-- Conditions
def women_count : ℕ := 12
def men_count : ℕ := 10
def avg_age_women : ℝ := 25.0
def avg_age_men : ℝ := 40.0

-- Expected Answer
def expected_avg_age : ℝ := 31.82

-- Proof Statement
theorem choir_average_age :
  ((women_count * avg_age_women) + (men_count * avg_age_men)) / (women_count + men_count) = expected_avg_age :=
by
  sorry

end choir_average_age_l2129_212988


namespace MarksScore_l2129_212922

theorem MarksScore (h_highest : ℕ) (h_range : ℕ) (h_relation : h_highest - h_least = h_range) (h_mark_twice : Mark = 2 * h_least) : Mark = 46 :=
by
    let h_highest := 98
    let h_range := 75
    let h_least := h_highest - h_range
    let Mark := 2 * h_least
    have := h_relation
    have := h_mark_twice
    sorry

end MarksScore_l2129_212922


namespace flowers_per_vase_l2129_212927

theorem flowers_per_vase (carnations roses vases total_flowers flowers_per_vase : ℕ)
  (h1 : carnations = 7)
  (h2 : roses = 47)
  (h3 : vases = 9)
  (h4 : total_flowers = carnations + roses)
  (h5 : flowers_per_vase = total_flowers / vases):
  flowers_per_vase = 6 := 
by {
  sorry
}

end flowers_per_vase_l2129_212927


namespace problem_statement_l2129_212934

theorem problem_statement (x y : ℝ) 
  (hA : A = (x + y) * (y - 3 * x))
  (hB : B = (x - y)^4 / (x - y)^2)
  (hCond : 2 * y + A = B - 6) :
  y = 2 * x^2 - 3 ∧ (y + 3)^2 - 2 * x * (x * y - 3) - 6 * x * (x + 1) = 0 :=
by
  sorry

end problem_statement_l2129_212934


namespace smallest_int_rel_prime_150_l2129_212903

theorem smallest_int_rel_prime_150 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 150 = 1 ∧ ∀ y : ℕ, (y > 1 ∧ Nat.gcd y 150 = 1) → x ≤ y :=
by
  sorry

end smallest_int_rel_prime_150_l2129_212903


namespace smallest_d_for_inverse_domain_l2129_212974

noncomputable def g (x : ℝ) : ℝ := 2 * (x + 1)^2 - 7

theorem smallest_d_for_inverse_domain : ∃ d : ℝ, (∀ x1 x2 : ℝ, x1 ≥ d → x2 ≥ d → g x1 = g x2 → x1 = x2) ∧ d = -1 :=
by
  use -1
  constructor
  · sorry
  · rfl

end smallest_d_for_inverse_domain_l2129_212974


namespace percentage_water_mixture_l2129_212915

theorem percentage_water_mixture 
  (volume_A : ℝ) (volume_B : ℝ) (volume_C : ℝ)
  (ratio_A : ℝ := 5) (ratio_B : ℝ := 3) (ratio_C : ℝ := 2)
  (percentage_water_A : ℝ := 0.20) (percentage_water_B : ℝ := 0.35) (percentage_water_C : ℝ := 0.50) :
  (volume_A = ratio_A) → (volume_B = ratio_B) → (volume_C = ratio_C) → 
  ((percentage_water_A * volume_A + percentage_water_B * volume_B + percentage_water_C * volume_C) /
   (ratio_A + ratio_B + ratio_C)) * 100 = 30.5 := 
by 
  intros hA hB hC
  -- Proof steps would go here
  sorry

end percentage_water_mixture_l2129_212915


namespace fixed_points_and_zeros_no_fixed_points_range_b_l2129_212952

def f (b c x : ℝ) : ℝ := x^2 + b * x + c

theorem fixed_points_and_zeros (b c : ℝ) (h1 : f b c (-3) = -3) (h2 : f b c 2 = 2) :
  ∃ x1 x2 : ℝ, f b c x1 = 0 ∧ f b c x2 = 0 ∧ x1 = -1 + Real.sqrt 7 ∧ x2 = -1 - Real.sqrt 7 :=
sorry

theorem no_fixed_points_range_b {b : ℝ} (h : ∀ x : ℝ, f b (b^2 / 4) x ≠ x) : 
  b > 1 / 3 ∨ b < -1 :=
sorry

end fixed_points_and_zeros_no_fixed_points_range_b_l2129_212952


namespace trapezoid_longer_side_length_l2129_212992

theorem trapezoid_longer_side_length (x : ℝ) (h₁ : 4 = 2*2) (h₂ : ∃ AP DQ O : ℝ, ∀ (S : ℝ), 
  S = (1/2) * (x + 2) * 1 → S = 2) : 
  x = 2 :=
by sorry

end trapezoid_longer_side_length_l2129_212992


namespace least_sum_of_exponents_l2129_212918

theorem least_sum_of_exponents (a b c : ℕ) (ha : 2^a ∣ 520) (hb : 2^b ∣ 520) (hc : 2^c ∣ 520) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  a + b + c = 12 :=
by
  sorry

end least_sum_of_exponents_l2129_212918


namespace questions_left_blank_l2129_212946

-- Definitions based on the conditions
def total_questions : Nat := 60
def word_problems : Nat := 20
def add_subtract_problems : Nat := 25
def algebra_problems : Nat := 10
def geometry_problems : Nat := 5
def total_time : Nat := 90

def time_per_word_problem : Nat := 2
def time_per_add_subtract_problem : Float := 1.5
def time_per_algebra_problem : Nat := 3
def time_per_geometry_problem : Nat := 4

def word_problems_answered : Nat := 15
def add_subtract_problems_answered : Nat := 22
def algebra_problems_answered : Nat := 8
def geometry_problems_answered : Nat := 3

-- The final goal is to prove that Steve left 12 questions blank
theorem questions_left_blank :
  total_questions - (word_problems_answered + add_subtract_problems_answered + algebra_problems_answered + geometry_problems_answered) = 12 :=
by
  sorry

end questions_left_blank_l2129_212946


namespace education_expenses_l2129_212989

noncomputable def totalSalary (savings : ℝ) (savingsPercentage : ℝ) : ℝ :=
  savings / savingsPercentage

def totalExpenses (rent milk groceries petrol misc : ℝ) : ℝ :=
  rent + milk + groceries + petrol + misc

def amountSpentOnEducation (totalSalary totalExpenses savings : ℝ) : ℝ :=
  totalSalary - (totalExpenses + savings)

theorem education_expenses :
  let rent := 5000
  let milk := 1500
  let groceries := 4500
  let petrol := 2000
  let misc := 700
  let savings := 1800
  let savingsPercentage := 0.10
  amountSpentOnEducation (totalSalary savings savingsPercentage) 
                          (totalExpenses rent milk groceries petrol misc) 
                          savings = 2500 :=
by
  sorry

end education_expenses_l2129_212989


namespace sector_field_area_l2129_212960

/-- Given a sector field with a circumference of 30 steps and a diameter of 16 steps, prove that its area is 120 square steps. --/
theorem sector_field_area (C : ℝ) (d : ℝ) (A : ℝ) : 
  C = 30 → d = 16 → A = 120 :=
by
  sorry

end sector_field_area_l2129_212960


namespace principal_amount_l2129_212900

theorem principal_amount
  (SI : ℝ) (R : ℝ) (T : ℝ)
  (h1 : SI = 155) (h2 : R = 4.783950617283951) (h3 : T = 4) :
  SI * 100 / (R * T) = 810.13 := 
  by 
    -- proof omitted
    sorry

end principal_amount_l2129_212900


namespace find_Y_l2129_212905

theorem find_Y 
  (a b c d X Y : ℕ)
  (h1 : a + b + c + d = 40)
  (h2 : X + Y + c + b = 40)
  (h3 : a + b + X = 30)
  (h4 : c + d + Y = 30)
  (h5 : X = 9) 
  : Y = 11 := 
by 
  sorry

end find_Y_l2129_212905


namespace gcd_of_45_75_90_l2129_212982

def gcd_three_numbers (a b c : ℕ) : ℕ :=
  Nat.gcd (Nat.gcd a b) c

theorem gcd_of_45_75_90 : gcd_three_numbers 45 75 90 = 15 := by
  sorry

end gcd_of_45_75_90_l2129_212982


namespace egg_rolls_total_l2129_212959

theorem egg_rolls_total (omar_egg_rolls karen_egg_rolls lily_egg_rolls : ℕ) :
  omar_egg_rolls = 219 → karen_egg_rolls = 229 → lily_egg_rolls = 275 → 
  omar_egg_rolls + karen_egg_rolls + lily_egg_rolls = 723 := 
by
  intros h1 h2 h3
  sorry

end egg_rolls_total_l2129_212959


namespace min_value_of_quadratic_l2129_212964

theorem min_value_of_quadratic (x : ℝ) : ∃ m : ℝ, (∀ x, x^2 + 10 * x ≥ m) ∧ m = -25 := by
  sorry

end min_value_of_quadratic_l2129_212964


namespace kim_boxes_on_tuesday_l2129_212966

theorem kim_boxes_on_tuesday
  (sold_on_thursday : ℕ)
  (sold_on_wednesday : ℕ)
  (sold_on_tuesday : ℕ)
  (h1 : sold_on_thursday = 1200)
  (h2 : sold_on_wednesday = 2 * sold_on_thursday)
  (h3 : sold_on_tuesday = 2 * sold_on_wednesday) :
  sold_on_tuesday = 4800 :=
sorry

end kim_boxes_on_tuesday_l2129_212966


namespace smallest_integer_to_make_multiple_of_five_l2129_212939

/-- The smallest positive integer that can be added to 725 to make it a multiple of 5 is 5. -/
theorem smallest_integer_to_make_multiple_of_five : 
  ∃ k : ℕ, k > 0 ∧ (725 + k) % 5 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → k ≤ m :=
sorry

end smallest_integer_to_make_multiple_of_five_l2129_212939
