import Mathlib

namespace solve_for_f_8_l164_164260

noncomputable def f (x : ℝ) : ℝ := (Real.logb 2 x)

theorem solve_for_f_8 {x : ℝ} (h : f (x^3) = Real.logb 2 x) : f 8 = 1 :=
by
sorry

end solve_for_f_8_l164_164260


namespace emily_small_gardens_l164_164807

theorem emily_small_gardens 
  (total_seeds : ℕ)
  (seeds_in_big_garden : ℕ)
  (seeds_per_small_garden : ℕ)
  (remaining_seeds := total_seeds - seeds_in_big_garden)
  (number_of_small_gardens := remaining_seeds / seeds_per_small_garden) :
  total_seeds = 41 → seeds_in_big_garden = 29 → seeds_per_small_garden = 4 → number_of_small_gardens = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end emily_small_gardens_l164_164807


namespace minute_first_catch_hour_l164_164462

theorem minute_first_catch_hour :
  ∃ (t : ℚ), t = 60 * (1 + (5 / 11)) :=
sorry

end minute_first_catch_hour_l164_164462


namespace propane_tank_and_burner_cost_l164_164387

theorem propane_tank_and_burner_cost
(Total_money: ℝ)
(Sheet_cost: ℝ)
(Rope_cost: ℝ)
(Helium_cost_per_oz: ℝ)
(Lift_per_oz: ℝ)
(Max_height: ℝ)
(ht: Total_money = 200)
(hs: Sheet_cost = 42)
(hr: Rope_cost = 18)
(hh: Helium_cost_per_oz = 1.50)
(hlo: Lift_per_oz = 113)
(hm: Max_height = 9492)
:
(Total_money - (Sheet_cost + Rope_cost) 
 - (Max_height / Lift_per_oz * Helium_cost_per_oz) 
 = 14) :=
by
  sorry

end propane_tank_and_burner_cost_l164_164387


namespace anne_speed_ratio_l164_164141

variable (B A A' : ℝ) (hours_to_clean_together : ℝ) (hours_to_clean_with_new_anne : ℝ)

-- Conditions
def cleaning_condition_1 := (A + B) * 4 = 1 -- Combined rate for 4 hours
def cleaning_condition_2 := A = 1 / 12      -- Anne's rate alone
def cleaning_condition_3 := (A' + B) * 3 = 1 -- Combined rate for 3 hours with new Anne's rate

-- Theorem to Prove
theorem anne_speed_ratio (h1 : cleaning_condition_1 B A)
                         (h2 : cleaning_condition_2 A)
                         (h3 : cleaning_condition_3 B A') :
                         (A' / A) = 2 :=
by sorry

end anne_speed_ratio_l164_164141


namespace land_value_moon_l164_164023

-- Define the conditions
def surface_area_earth : ℕ := 200
def surface_area_ratio : ℕ := 5
def value_ratio : ℕ := 6
def total_value_earth : ℕ := 80

-- Define the question and the expected answer
noncomputable def total_value_moon : ℕ := 96

-- State the proof problem
theorem land_value_moon :
  (surface_area_earth / surface_area_ratio * value_ratio) * (surface_area_earth / surface_area_ratio) = total_value_moon := 
sorry

end land_value_moon_l164_164023


namespace distance_between_parallel_lines_eq_2_l164_164391

def line1 (x y : ℝ) : Prop := 3 * x - 4 * y + 2 = 0
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y - 8 = 0

theorem distance_between_parallel_lines_eq_2 :
  let A := 3
  let B := -4
  let c1 := 2
  let c2 := -8
  let d := (|c1 - c2| / Real.sqrt (A^2 + B^2))
  d = 2 :=
by
  sorry

end distance_between_parallel_lines_eq_2_l164_164391


namespace glucose_solution_volume_l164_164198

theorem glucose_solution_volume (V : ℕ) (h : 500 / 10 = V / 20) : V = 1000 :=
sorry

end glucose_solution_volume_l164_164198


namespace jacket_spending_l164_164008

def total_spent : ℝ := 14.28
def spent_on_shorts : ℝ := 9.54
def spent_on_jacket : ℝ := 4.74

theorem jacket_spending :
  spent_on_jacket = total_spent - spent_on_shorts :=
by sorry

end jacket_spending_l164_164008


namespace sum_cubes_div_product_eq_three_l164_164180

-- Given that x, y, z are non-zero real numbers and x + y + z = 3,
-- we need to prove that the possible value of (x^3 + y^3 + z^3) / xyz is 3.

theorem sum_cubes_div_product_eq_three 
  (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (hxyz_sum : x + y + z = 3) : 
  (x^3 + y^3 + z^3) / (x * y * z) = 3 :=
by
  sorry

end sum_cubes_div_product_eq_three_l164_164180


namespace geometric_sequence_sum_l164_164547

theorem geometric_sequence_sum
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (r : ℝ)
  (h1 : r ≠ 1)
  (h2 : ∀ n, S n = a 0 * (1 - r^(n + 1)) / (1 - r))
  (h3 : S 5 = 3)
  (h4 : S 10 = 9) :
  S 15 = 21 :=
sorry

end geometric_sequence_sum_l164_164547


namespace triangle_angles_l164_164478

noncomputable def angle_triangle (E : ℝ) :=
if E = 45 then (90, 45, 45) else if E = 36 then (72, 72, 36) else (0, 0, 0)

theorem triangle_angles (E : ℝ) :
  (∃ E, E = 45 → angle_triangle E = (90, 45, 45))
  ∨
  (∃ E, E = 36 → angle_triangle E = (72, 72, 36)) :=
by
    sorry

end triangle_angles_l164_164478


namespace transformed_point_of_function_l164_164481

theorem transformed_point_of_function (f : ℝ → ℝ) (h : f 1 = -2) : f (-1) + 1 = -1 :=
by
  sorry

end transformed_point_of_function_l164_164481


namespace sufficient_but_not_necessary_for_abs_eq_two_l164_164208

theorem sufficient_but_not_necessary_for_abs_eq_two (a : ℝ) :
  (a = -2 → |a| = 2) ∧ (|a| = 2 → a = 2 ∨ a = -2) :=
by
   sorry

end sufficient_but_not_necessary_for_abs_eq_two_l164_164208


namespace necessary_and_sufficient_condition_l164_164616

def line1 (x y : ℝ) : Prop := x - y - 1 = 0
def line2 (x y a : ℝ) : Prop := x + a * y - 2 = 0

def p (a : ℝ) : Prop := ∀ x y : ℝ, line1 x y → line2 x y a
def q (a : ℝ) : Prop := a = -1

theorem necessary_and_sufficient_condition (a : ℝ) : (p a) ↔ (q a) :=
by
  sorry

end necessary_and_sufficient_condition_l164_164616


namespace typing_difference_l164_164620

theorem typing_difference (initial_speed after_speed : ℕ) (time_interval : ℕ) (h_initial : initial_speed = 10) 
  (h_after : after_speed = 8) (h_time : time_interval = 5) : 
  (initial_speed * time_interval) - (after_speed * time_interval) = 10 := 
by 
  sorry

end typing_difference_l164_164620


namespace max_sum_of_arithmetic_sequence_l164_164195

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
(h1 : 3 * a 8 = 5 * a 13) 
(h2 : a 1 > 0)
(hS : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * d)) :
S 20 > S 21 ∧ S 20 > S 10 ∧ S 20 > S 11 :=
sorry

end max_sum_of_arithmetic_sequence_l164_164195


namespace find_some_ounce_size_l164_164179

variable (x : ℕ)
variable (h_total : 122 = 6 * 5 + 4 * x + 15 * 4)

theorem find_some_ounce_size : x = 8 := by
  sorry

end find_some_ounce_size_l164_164179


namespace work_efficiency_ratio_l164_164880

variables (A_eff B_eff : ℚ) (a b : Type)

theorem work_efficiency_ratio (h1 : B_eff = 1 / 33)
  (h2 : A_eff + B_eff = 1 / 11) :
  A_eff / B_eff = 2 :=
by 
  sorry

end work_efficiency_ratio_l164_164880


namespace polygon_number_of_sides_l164_164273

-- Define the given conditions
def each_interior_angle (n : ℕ) : ℕ := 120

-- Define the property to calculate the number of sides
def num_sides (each_exterior_angle : ℕ) : ℕ := 360 / each_exterior_angle

-- Statement of the problem
theorem polygon_number_of_sides : num_sides (180 - each_interior_angle 6) = 6 :=
by
  -- Proof is omitted
  sorry

end polygon_number_of_sides_l164_164273


namespace find_real_numbers_a_b_l164_164255

noncomputable def f (a b x : ℝ) : ℝ :=
  a * (Real.sin x * Real.cos x) - (Real.sqrt 3) * a * (Real.cos x) ^ 2 + Real.sqrt 3 / 2 * a + b

theorem find_real_numbers_a_b (a b : ℝ) (h1 : 0 < a)
    (h2 : ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), -2 ≤ f a b x ∧ f a b x ≤ Real.sqrt 3)
    : a = 2 ∧ b = -2 + Real.sqrt 3 :=
sorry

end find_real_numbers_a_b_l164_164255


namespace range_of_m_l164_164666

theorem range_of_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) (hineq : 4 / (x + 1) + 1 / y < m^2 + (3 / 2) * m) :
  m < -3 ∨ m > 3 / 2 :=
by sorry

end range_of_m_l164_164666


namespace find_y_value_l164_164218

-- Define the angles in Lean
def angle1 (y : ℕ) : ℕ := 6 * y
def angle2 (y : ℕ) : ℕ := 7 * y
def angle3 (y : ℕ) : ℕ := 3 * y
def angle4 (y : ℕ) : ℕ := 2 * y

-- The condition that the sum of the angles is 360
def angles_sum_to_360 (y : ℕ) : Prop :=
  angle1 y + angle2 y + angle3 y + angle4 y = 360

-- The proof problem statement
theorem find_y_value (y : ℕ) (h : angles_sum_to_360 y) : y = 20 :=
sorry

end find_y_value_l164_164218


namespace car_pass_time_l164_164793

theorem car_pass_time (length : ℝ) (speed_kmph : ℝ) (speed_mps : ℝ) (time : ℝ) :
  length = 10 → 
  speed_kmph = 36 → 
  speed_mps = speed_kmph * (1000 / 3600) → 
  time = length / speed_mps → 
  time = 1 :=
by
  intros h_length h_speed_kmph h_speed_conversion h_time_calculation
  -- Here we would normally construct the proof
  sorry

end car_pass_time_l164_164793


namespace min_satisfies_condition_only_for_x_eq_1_div_4_l164_164816

theorem min_satisfies_condition_only_for_x_eq_1_div_4 (x : ℝ) (hx_nonneg : 0 ≤ x) :
  (min (Real.sqrt x) (min (x^2) x) = 1/16) ↔ (x = 1/4) :=
by sorry

end min_satisfies_condition_only_for_x_eq_1_div_4_l164_164816


namespace value_of_e_is_91_l164_164957

noncomputable def value_of_e (a b c d e : ℤ) (k : ℤ) : Prop :=
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ e % 2 = 1 ∧
  b = a + 2 * k ∧ c = a + 4 * k ∧ d = a + 6 * k ∧ e = a + 8 * k ∧
  a + c = 146 ∧ k > 0 ∧ 2 * k ≥ 4 ∧ k ≠ 2

theorem value_of_e_is_91 (a b c d e k : ℤ) (h : value_of_e a b c d e k) : e = 91 :=
  sorry

end value_of_e_is_91_l164_164957


namespace sum_of_cubes_of_integers_l164_164278

theorem sum_of_cubes_of_integers (n: ℕ) (h1: (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 = 8830) : 
  (n-1)^3 + n^3 + (n+1)^3 + (n+2)^3 = 52264 :=
by
  sorry

end sum_of_cubes_of_integers_l164_164278


namespace remainder_of_product_and_square_is_zero_mod_7_remainder_of_9_pow_100_mod_8_l164_164486

-- Part (a): Remainder of (1989 * 1990 * 1991 + 1992^2) when divided by 7 is 0.
theorem remainder_of_product_and_square_is_zero_mod_7 :
  (1989 * 1990 * 1991 + 1992^2) % 7 = 0 :=
sorry

-- Part (b): Remainder of 9^100 when divided by 8 is 1.
theorem remainder_of_9_pow_100_mod_8 :
  9^100 % 8 = 1 :=
sorry

end remainder_of_product_and_square_is_zero_mod_7_remainder_of_9_pow_100_mod_8_l164_164486


namespace cosine_product_identity_l164_164070

open Real

theorem cosine_product_identity (α : ℝ) (n : ℕ) :
  (List.foldr (· * ·) 1 (List.map (λ k => cos (2^k * α)) (List.range (n + 1)))) =
  sin (2^(n + 1) * α) / (2^(n + 1) * sin α) :=
sorry

end cosine_product_identity_l164_164070


namespace triangular_square_is_triangular_l164_164396

-- Definition of a triangular number
def is_triang_number (n : ℕ) : Prop :=
  ∃ x : ℕ, n = x * (x + 1) / 2

-- The main theorem statement
theorem triangular_square_is_triangular :
  ∃ x : ℕ, 
    is_triang_number x ∧ 
    is_triang_number (x * x) :=
sorry

end triangular_square_is_triangular_l164_164396


namespace measure_angle_PQR_given_conditions_l164_164519

-- Definitions based on conditions
variables {R P Q S : Type} [LinearOrder R] [AddGroup Q] [LinearOrder P] [LinearOrder S]

-- Assume given conditions
def is_straight_line (r s p : ℝ) : Prop := r + p = 2 * s

def is_isosceles_triangle (p s q : ℝ) : Prop := p = q

def angle (q s p : ℝ) := (q - s) - (s - p)

variables (r p q s : ℝ)

-- Define the given angles and equality conditions
def given_conditions : Prop := 
  is_straight_line r s p ∧
  angle q s p = 60 ∧
  is_isosceles_triangle p s q ∧
  r ≠ q 

-- The theorem we want to prove
theorem measure_angle_PQR_given_conditions : given_conditions r p q s → angle p q r = 120 := by
  sorry

end measure_angle_PQR_given_conditions_l164_164519


namespace third_number_correct_l164_164194

-- Given that the row of Pascal's triangle with 51 numbers corresponds to the binomial coefficients of 50.
def third_number_in_51_pascal_row : ℕ := Nat.choose 50 2

-- Prove that the third number in this row of Pascal's triangle is 1225.
theorem third_number_correct : third_number_in_51_pascal_row = 1225 := 
by 
  -- Calculation part can be filled in for the full proof.
  sorry

end third_number_correct_l164_164194


namespace probability_blue_or_green_l164_164058

def faces : Type := {faces : ℕ // faces = 6}
noncomputable def blue_faces : ℕ := 3
noncomputable def red_faces : ℕ := 2
noncomputable def green_faces : ℕ := 1

theorem probability_blue_or_green :
  (blue_faces + green_faces) / 6 = (2 / 3) := by
  sorry

end probability_blue_or_green_l164_164058


namespace ladder_base_distance_l164_164494

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l164_164494


namespace arithmetic_seq_sin_identity_l164_164032

theorem arithmetic_seq_sin_identity:
  ∀ (a : ℕ → ℝ), (a 2 + a 6 = (3/2) * Real.pi) → (Real.sin (2 * a 4 - Real.pi / 3) = -1 / 2) :=
by
  sorry

end arithmetic_seq_sin_identity_l164_164032


namespace inequality_holds_l164_164158

theorem inequality_holds (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  x^4 + y^4 + 2 / (x^2 * y^2) ≥ 4 := 
by
  sorry

end inequality_holds_l164_164158


namespace original_number_l164_164841

theorem original_number (n : ℕ) (h1 : 2319 % 21 = 0) (h2 : 2319 = 21 * (n + 1) - 1) : n = 2318 := 
sorry

end original_number_l164_164841


namespace no_solution_l164_164814

theorem no_solution (x : ℝ) : ¬ (3 * x^2 + 9 * x ≤ -12) :=
sorry

end no_solution_l164_164814


namespace pq_or_l164_164170

def p : Prop := 2 % 2 = 0
def q : Prop := 3 % 2 = 0

theorem pq_or : p ∨ q :=
by
  -- proof goes here
  sorry

end pq_or_l164_164170


namespace sqrt_ceil_eq_sqrt_sqrt_l164_164798

theorem sqrt_ceil_eq_sqrt_sqrt (a : ℝ) (h : a > 1) : 
  (Int.floor (Real.sqrt (Int.floor (Real.sqrt a)))) = (Int.floor (Real.sqrt (Real.sqrt a))) :=
sorry

end sqrt_ceil_eq_sqrt_sqrt_l164_164798


namespace orthocenter_PQR_is_correct_l164_164815

def Point := (ℝ × ℝ × ℝ)

def P : Point := (2, 3, 4)
def Q : Point := (6, 4, 2)
def R : Point := (4, 5, 6)

def orthocenter (P Q R : Point) : Point := sorry

theorem orthocenter_PQR_is_correct : orthocenter P Q R = (3 / 2, 13 / 2, 5) :=
sorry

end orthocenter_PQR_is_correct_l164_164815


namespace number_of_human_family_members_l164_164445

-- Definitions for the problem
def num_birds := 4
def num_dogs := 3
def num_cats := 18
def bird_feet := 2
def dog_feet := 4
def cat_feet := 4
def human_feet := 2
def human_heads := 1

def animal_feet := (num_birds * bird_feet) + (num_dogs * dog_feet) + (num_cats * cat_feet)
def animal_heads := num_birds + num_dogs + num_cats

def total_feet (H : Nat) := animal_feet + (H * human_feet)
def total_heads (H : Nat) := animal_heads + (H * human_heads)

-- The problem statement translated to Lean
theorem number_of_human_family_members (H : Nat) : (total_feet H) = (total_heads H) + 74 → H = 7 :=
by
  sorry

end number_of_human_family_members_l164_164445


namespace P_div_by_Q_iff_l164_164703

def P (x : ℂ) (n : ℕ) : ℂ := x^(4*n) + x^(3*n) + x^(2*n) + x^n + 1
def Q (x : ℂ) : ℂ := x^4 + x^3 + x^2 + x + 1

theorem P_div_by_Q_iff (n : ℕ) : (Q x ∣ P x n) ↔ ¬(5 ∣ n) := sorry

end P_div_by_Q_iff_l164_164703


namespace length_of_arc_correct_l164_164646

open Real

noncomputable def length_of_arc (r θ : ℝ) := θ * r

theorem length_of_arc_correct (A r θ : ℝ) (hA : A = (θ / (2 * π)) * (π * r^2)) (hr : r = 5) (hA_val : A = 13.75) :
  length_of_arc r θ = 5.5 :=
by
  -- Proof steps are omitted
  sorry

end length_of_arc_correct_l164_164646


namespace sum_of_reciprocals_l164_164374

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 16) (h2 : x * y = 48) : (1 / x + 1 / y) = (1 / 3) :=
by
  sorry

end sum_of_reciprocals_l164_164374


namespace correct_exponentiation_l164_164298

theorem correct_exponentiation (a : ℕ) : 
  (a^3 * a^2 = a^5) ∧ ¬(a^3 + a^2 = a^5) ∧ ¬((a^2)^3 = a^5) ∧ ¬(a^10 / a^2 = a^5) :=
by
  -- Proof steps and actual mathematical validation will go here.
  -- For now, we skip the actual proof due to the problem requirements.
  sorry

end correct_exponentiation_l164_164298


namespace current_length_of_highway_l164_164864

def total_length : ℕ := 650
def miles_first_day : ℕ := 50
def miles_second_day : ℕ := 3 * miles_first_day
def miles_still_needed : ℕ := 250
def miles_built : ℕ := miles_first_day + miles_second_day

theorem current_length_of_highway :
  total_length - miles_still_needed = 400 :=
by
  sorry

end current_length_of_highway_l164_164864


namespace brothers_work_rate_l164_164959

theorem brothers_work_rate (A B C : ℝ) :
  (1 / A + 1 / B = 1 / 8) ∧ (1 / A + 1 / C = 1 / 9) ∧ (1 / B + 1 / C = 1 / 10) →
  A = 160 / 19 ∧ B = 160 / 9 ∧ C = 32 / 3 :=
by
  sorry

end brothers_work_rate_l164_164959


namespace carrie_phone_charges_l164_164518

def total_miles (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

def charges_needed (total_miles charge_miles : ℕ) : ℕ :=
  total_miles / charge_miles + if total_miles % charge_miles = 0 then 0 else 1

theorem carrie_phone_charges :
  let d1 := 135
  let d2 := 135 + 124
  let d3 := 159
  let d4 := 189
  let charge_miles := 106
  charges_needed (total_miles d1 d2 d3 d4) charge_miles = 7 :=
by
  sorry

end carrie_phone_charges_l164_164518


namespace find_angle_C_max_area_triangle_l164_164953

-- Part I: Proving angle C
theorem find_angle_C (a b c : ℝ) (A B C : ℝ)
    (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (h4 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C) :
    C = Real.pi / 3 :=
sorry

-- Part II: Finding maximum area of triangle ABC
theorem max_area_triangle (a b : ℝ) (c : ℝ) (h_c : c = 2 * Real.sqrt 3) (A B C : ℝ)
    (h_A : A > 0) (h_B : B > 0) (h_C : C = Real.pi / 3)
    (h : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C) :
    0.5 * a * b * Real.sin C ≤ 3 * Real.sqrt 3 :=
sorry

end find_angle_C_max_area_triangle_l164_164953


namespace find_n_l164_164757

theorem find_n (n : ℕ) (composite_n : n > 1 ∧ ¬Prime n) : 
  ((∃ m : ℕ, ∀ d : ℕ, d ∣ n ∧ 1 < d ∧ d < n → d + 1 ∣ m ∧ 1 < d + 1 ∧ d + 1 < m) ↔ 
    (n = 4 ∨ n = 8)) :=
by sorry

end find_n_l164_164757


namespace cost_of_large_tubs_l164_164934

theorem cost_of_large_tubs (L : ℝ) (h1 : 3 * L + 6 * 5 = 48) : L = 6 :=
by {
  sorry
}

end cost_of_large_tubs_l164_164934


namespace find_a10_l164_164943

theorem find_a10 (a : ℕ → ℝ) 
  (h₁ : a 1 = 1) 
  (h₂ : ∀ n : ℕ, a n - a (n+1) = a n * a (n+1)) : 
  a 10 = 1 / 10 :=
sorry

end find_a10_l164_164943


namespace sum_first_53_odd_numbers_l164_164732

-- Definitions based on the given conditions
def first_odd_number := 1

def nth_odd_number (n : ℕ) : ℕ :=
  1 + (n - 1) * 2

def sum_n_odd_numbers (n : ℕ) : ℕ :=
  (n * n)

-- Theorem statement to prove
theorem sum_first_53_odd_numbers : sum_n_odd_numbers 53 = 2809 := 
by
  sorry

end sum_first_53_odd_numbers_l164_164732


namespace triangle_inequality_1_triangle_inequality_2_l164_164935

variable (a b c : ℝ)

theorem triangle_inequality_1 (h1 : a + b + c = 2) (h2 : 0 ≤ a) (h3 : 0 ≤ b) (h4 : 0 ≤ c) (h5 : a ≤ 1) (h6 : b ≤ 1) (h7 : c ≤ 1) : 
  a * b * c + 28 / 27 ≥ a * b + b * c + c * a :=
by
  sorry

theorem triangle_inequality_2 (h1 : a + b + c = 2) (h2 : 0 ≤ a) (h3 : 0 ≤ b) (h4 : 0 ≤ c) (h5 : a ≤ 1) (h6 : b ≤ 1) (h7 : c ≤ 1) : 
  a * b + b * c + c * a ≥ a * b * c + 1 :=
by
  sorry

end triangle_inequality_1_triangle_inequality_2_l164_164935


namespace complex_sum_equals_one_l164_164676

noncomputable def main (x : ℂ) (h1 : x^7 = 1) (h2 : x ≠ 1) : ℂ :=
  (x^2 / (x - 1)) + (x^4 / (x^2 - 1)) + (x^6 / (x^3 - 1))

theorem complex_sum_equals_one (x : ℂ) (h1 : x^7 = 1) (h2 : x ≠ 1) : main x h1 h2 = 1 := by
  sorry

end complex_sum_equals_one_l164_164676


namespace total_bags_l164_164892

-- Definitions based on the conditions
def bags_on_monday : ℕ := 4
def bags_next_day : ℕ := 8

-- Theorem statement
theorem total_bags : bags_on_monday + bags_next_day = 12 :=
by
  -- Proof will be added here
  sorry

end total_bags_l164_164892


namespace largest_n_divides_l164_164316

theorem largest_n_divides (n : ℕ) (h : 2^n ∣ 5^256 - 1) : n ≤ 10 := sorry

end largest_n_divides_l164_164316


namespace distinct_midpoints_at_least_2n_minus_3_l164_164686

open Set

theorem distinct_midpoints_at_least_2n_minus_3 
  (n : ℕ) 
  (points : Finset (ℝ × ℝ)) 
  (h_points_card : points.card = n) :
  ∃ (midpoints : Finset (ℝ × ℝ)), 
    midpoints.card ≥ 2 * n - 3 := 
sorry

end distinct_midpoints_at_least_2n_minus_3_l164_164686


namespace cloves_used_for_roast_chicken_l164_164979

section
variable (total_cloves : ℕ)
variable (remaining_cloves : ℕ)

theorem cloves_used_for_roast_chicken (h1 : total_cloves = 93) (h2 : remaining_cloves = 7) : total_cloves - remaining_cloves = 86 := 
by 
  have h : total_cloves - remaining_cloves = 93 - 7 := by rw [h1, h2]
  exact h
-- sorry
end

end cloves_used_for_roast_chicken_l164_164979


namespace parallel_planes_imply_l164_164898

variable {Point Line Plane : Type}

-- Definitions of parallelism and perpendicularity between lines and planes
variables {parallel_perpendicular : Line → Plane → Prop}
variables {parallel_lines : Line → Line → Prop}
variables {parallel_planes : Plane → Plane → Prop}

-- Given conditions
variable {m n : Line}
variable {α β : Plane}

-- Conditions
axiom m_parallel_n : parallel_lines m n
axiom m_perpendicular_α : parallel_perpendicular m α
axiom n_perpendicular_β : parallel_perpendicular n β

-- The statement to be proven
theorem parallel_planes_imply (m_parallel_n : parallel_lines m n)
  (m_perpendicular_α : parallel_perpendicular m α)
  (n_perpendicular_β : parallel_perpendicular n β) :
  parallel_planes α β :=
sorry

end parallel_planes_imply_l164_164898


namespace sqrt_quartic_equiv_l164_164909

-- Define x as a positive real number
variable (x : ℝ)
variable (hx : 0 < x)

-- Statement of the problem to prove
theorem sqrt_quartic_equiv (x : ℝ) (hx : 0 < x) : (x^2 * x^(1/2))^(1/4) = x^(5/8) :=
sorry

end sqrt_quartic_equiv_l164_164909


namespace eleven_twelve_divisible_by_133_l164_164726

theorem eleven_twelve_divisible_by_133 (n : ℕ) (h : n > 0) : 133 ∣ (11^(n+2) + 12^(2*n+1)) := 
by 
  sorry

end eleven_twelve_divisible_by_133_l164_164726


namespace car_distance_l164_164220

theorem car_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (distance : ℝ) 
  (h_speed : speed = 160) 
  (h_time : time = 5) 
  (h_dist_formula : distance = speed * time) : 
  distance = 800 :=
by sorry

end car_distance_l164_164220


namespace color_of_face_opposite_silver_is_yellow_l164_164303

def Face : Type := String

def Color : Type := String

variable (B Y O Bl S V : Color)

-- Conditions based on views
variable (cube : Face → Color)
variable (top front_right_1 right_1 front_right_2 front_right_3 : Face)
variable (back : Face)

axiom view1 : cube top = B ∧ cube front_right_1 = Y ∧ cube right_1 = O
axiom view2 : cube top = B ∧ cube front_right_2 = Bl ∧ cube right_1 = O
axiom view3 : cube top = B ∧ cube front_right_3 = V ∧ cube right_1 = O

-- Additional axiom based on the fact that S is not visible and deduced to be on the back face
axiom silver_back : cube back = S

-- The problem: Prove that the color of the face opposite the silver face is yellow.
theorem color_of_face_opposite_silver_is_yellow :
  (∃ front : Face, cube front = Y) :=
by
  sorry

end color_of_face_opposite_silver_is_yellow_l164_164303


namespace proof_complement_union_l164_164018

open Set

variable (U A B: Set Nat)

def complement_equiv_union (U A B: Set Nat) : Prop :=
  (U \ A) ∪ B = {0, 2, 3, 6}

theorem proof_complement_union: 
  U = {0, 1, 3, 5, 6, 8} → 
  A = {1, 5, 8} → 
  B = {2} → 
  complement_equiv_union U A B :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  -- Proof omitted
  sorry

end proof_complement_union_l164_164018


namespace tank_capacity_l164_164301

theorem tank_capacity (x : ℝ) 
  (h1 : 1/4 * x + 180 = 2/3 * x) : 
  x = 432 :=
by
  sorry

end tank_capacity_l164_164301


namespace roundness_of_8000000_l164_164003

def is_prime (n : Nat) : Prop := sorry

def prime_factors_exponents (n : Nat) : List (Nat × Nat) := sorry

def roundness (n : Nat) : Nat := 
  (prime_factors_exponents n).foldr (λ p acc => p.2 + acc) 0

theorem roundness_of_8000000 : roundness 8000000 = 15 :=
sorry

end roundness_of_8000000_l164_164003


namespace king_arthur_round_table_seats_l164_164968

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end king_arthur_round_table_seats_l164_164968


namespace length_of_segment_l164_164677

theorem length_of_segment (x : ℝ) : 
  |x - (27^(1/3))| = 5 →
  ∃ a b : ℝ, a - b = 10 ∧ (|a - (27^(1/3))| = 5 ∧ |b - (27^(1/3))| = 5) :=
by
  sorry

end length_of_segment_l164_164677


namespace max_minute_hands_l164_164708

theorem max_minute_hands (m n : ℕ) (h : m * n = 27) : m + n ≤ 28 :=
sorry

end max_minute_hands_l164_164708


namespace positive_difference_for_6_points_l164_164456

def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def positiveDifferenceTrianglesAndQuadrilaterals (n : ℕ) : ℕ :=
  combinations n 3 - combinations n 4

theorem positive_difference_for_6_points : positiveDifferenceTrianglesAndQuadrilaterals 6 = 5 :=
by
  sorry

end positive_difference_for_6_points_l164_164456


namespace power_multiplication_eq_neg4_l164_164698

theorem power_multiplication_eq_neg4 :
  (-0.25) ^ 11 * (-4) ^ 12 = -4 := 
  sorry

end power_multiplication_eq_neg4_l164_164698


namespace nanometers_to_scientific_notation_l164_164542

theorem nanometers_to_scientific_notation :
  (246 : ℝ) * (10 ^ (-9 : ℝ)) = (2.46 : ℝ) * (10 ^ (-7 : ℝ)) :=
by
  sorry

end nanometers_to_scientific_notation_l164_164542


namespace tax_computation_l164_164232

def income : ℕ := 56000
def first_portion_income : ℕ := 40000
def first_portion_rate : ℝ := 0.12
def remaining_income : ℕ := income - first_portion_income
def remaining_rate : ℝ := 0.20
def expected_tax : ℝ := 8000

theorem tax_computation :
  (first_portion_rate * first_portion_income) +
  (remaining_rate * remaining_income) = expected_tax := by
  sorry

end tax_computation_l164_164232


namespace central_angle_of_sector_l164_164305

theorem central_angle_of_sector :
  ∃ R α : ℝ, (2 * R + α * R = 4) ∧ (1 / 2 * R ^ 2 * α = 1) ∧ α = 2 :=
by
  sorry

end central_angle_of_sector_l164_164305


namespace total_cost_textbooks_l164_164971

theorem total_cost_textbooks :
  let sale_books := 5 * 10
  let online_books := 40
  let bookstore_books := 3 * 40
  sale_books + online_books + bookstore_books = 210 :=
by
  let sale_books := 5 * 10
  let online_books := 40
  let bookstore_books := 3 * 40
  show sale_books + online_books + bookstore_books = 210
  sorry

end total_cost_textbooks_l164_164971


namespace percentage_of_red_shirts_l164_164758

theorem percentage_of_red_shirts
  (Total : ℕ) 
  (P_blue P_green : ℝ) 
  (N_other : ℕ)
  (H_Total : Total = 600)
  (H_P_blue : P_blue = 0.45) 
  (H_P_green : P_green = 0.15) 
  (H_N_other : N_other = 102) :
  ( (Total - (P_blue * Total + P_green * Total + N_other)) / Total ) * 100 = 23 := by
  sorry

end percentage_of_red_shirts_l164_164758


namespace integral_equals_result_l164_164879

noncomputable def integral_value : ℝ :=
  ∫ x in 1.0..2.0, (x^2 + 1) / x

theorem integral_equals_result :
  integral_value = (3 / 2) + Real.log 2 := 
by
  sorry

end integral_equals_result_l164_164879


namespace find_prime_p_l164_164929

noncomputable def concatenate (q r : ℕ) : ℕ :=
q * 10 ^ (r.digits 10).length + r

theorem find_prime_p (q r p : ℕ) (hq : Nat.Prime q) (hr : Nat.Prime r) (hp : Nat.Prime p)
  (h : concatenate q r + 3 = p^2) : p = 5 :=
sorry

end find_prime_p_l164_164929


namespace find_r_x_l164_164501

open Nat

theorem find_r_x (r n : ℕ) (x : ℕ) (h_r_le_70 : r ≤ 70) (repr_x : x = (10 * r + 6) * (r ^ (2 * n) - 1) / (r ^ 2 - 1))
  (repr_x2 : x^2 = (r ^ (4 * n) - 1) / (r - 1)) :
  (r = 7 ∧ x = 26) :=
by
  sorry

end find_r_x_l164_164501


namespace betty_garden_total_plants_l164_164532

theorem betty_garden_total_plants (B O : ℕ) 
  (h1 : B = 5) 
  (h2 : O = 2 + 2 * B) : 
  B + O = 17 := by
  sorry

end betty_garden_total_plants_l164_164532


namespace initial_water_amount_l164_164977

theorem initial_water_amount (W : ℝ) (h1 : 0.006 * 50 = 0.03 * W) : W = 10 :=
by
  -- Proof steps would go here
  sorry

end initial_water_amount_l164_164977


namespace pow_sub_nat_ge_seven_l164_164261

open Nat

theorem pow_sub_nat_ge_seven
  (m n : ℕ) 
  (h1 : m > 1)
  (h2 : 2^(2 * m + 1) - n^2 ≥ 0) : 
  2^(2 * m + 1) - n^2 ≥ 7 :=
sorry

end pow_sub_nat_ge_seven_l164_164261


namespace random_sampling_not_in_proving_methods_l164_164160

inductive Method
| Comparison
| RandomSampling
| SyntheticAndAnalytic
| ProofByContradictionAndScaling

open Method

def proving_methods : List Method :=
  [Comparison, SyntheticAndAnalytic, ProofByContradictionAndScaling]

theorem random_sampling_not_in_proving_methods : 
  RandomSampling ∉ proving_methods :=
sorry

end random_sampling_not_in_proving_methods_l164_164160


namespace person_A_boxes_average_unit_price_after_promotion_l164_164213

-- Definitions based on the conditions.
def unit_price (x: ℕ) (y: ℕ) : ℚ := y / x

def person_A_spent : ℕ := 2400
def person_B_spent : ℕ := 3000
def promotion_discount : ℕ := 20
def boxes_difference : ℕ := 10

-- Main proofs
theorem person_A_boxes (unit_price: ℕ → ℕ → ℚ) 
  (person_A_spent person_B_spent boxes_difference: ℕ): 
  ∃ x, unit_price person_A_spent x = unit_price person_B_spent (x + boxes_difference) 
  ∧ x = 40 := 
by {
  sorry
}

theorem average_unit_price_after_promotion (unit_price: ℕ → ℕ → ℚ) 
  (promotion_discount: ℕ) (person_A_spent person_B_spent: ℕ) 
  (boxes_A_promotion boxes_B: ℕ): 
  person_A_spent / (boxes_A_promotion * 2) + 20 = 48 
  ∧ person_B_spent / (boxes_B * 2) + 20 = 50 :=
by {
  sorry
}

end person_A_boxes_average_unit_price_after_promotion_l164_164213


namespace evaluate_g_sum_l164_164775

def g (a b : ℚ) : ℚ :=
if a + b ≤ 5 then (a^2 * b - a + 3) / (3 * a) 
else (a * b^2 - b - 3) / (-3 * b)

theorem evaluate_g_sum : g 3 2 + g 3 3 = -1 / 3 :=
by
  sorry

end evaluate_g_sum_l164_164775


namespace total_surface_area_prime_rectangular_solid_l164_164204

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Prime n

def prime_edge_lengths (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c

def volume (a b c : ℕ) : ℕ := a * b * c

def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + b * c + c * a)

-- The main theorem statement
theorem total_surface_area_prime_rectangular_solid :
  ∃ (a b c : ℕ), prime_edge_lengths a b c ∧ volume a b c = 105 ∧ surface_area a b c = 142 :=
sorry

end total_surface_area_prime_rectangular_solid_l164_164204


namespace volume_of_prism_l164_164505

theorem volume_of_prism (l w h : ℝ) (h1 : l * w = 15) (h2 : w * h = 10) (h3 : l * h = 6) :
  l * w * h = 30 :=
by
  sorry

end volume_of_prism_l164_164505


namespace all_numbers_are_2007_l164_164171

noncomputable def sequence_five_numbers (a b c d e : ℤ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ 
  (a = 2007 ∨ b = 2007 ∨ c = 2007 ∨ d = 2007 ∨ e = 2007) ∧ 
  (∃ r1, b = r1 * a ∧ c = r1 * b ∧ d = r1 * c ∧ e = r1 * d) ∧
  (∃ r2, a = r2 * b ∧ c = r2 * a ∧ d = r2 * c ∧ e = r2 * d) ∧
  (∃ r3, a = r3 * c ∧ b = r3 * a ∧ d = r3 * b ∧ e = r3 * d) ∧
  (∃ r4, a = r4 * d ∧ b = r4 * a ∧ c = r4 * b ∧ e = r4 * d) ∧
  (∃ r5, a = r5 * e ∧ b = r5 * a ∧ c = r5 * b ∧ d = r5 * c)

theorem all_numbers_are_2007 (a b c d e : ℤ) 
  (h : sequence_five_numbers a b c d e) : 
  a = 2007 ∧ b = 2007 ∧ c = 2007 ∧ d = 2007 ∧ e = 2007 :=
sorry

end all_numbers_are_2007_l164_164171


namespace shaded_region_occupies_32_percent_of_total_area_l164_164911

-- Conditions
def angle_sector := 90
def r_small := 1
def r_large := 3
def r_sector := 4

-- Question: Prove the shaded region occupies 32% of the total area given the conditions
theorem shaded_region_occupies_32_percent_of_total_area :
  let area_large_sector := (1 / 4) * Real.pi * (r_sector ^ 2)
  let area_small_sector := (1 / 4) * Real.pi * (r_large ^ 2)
  let total_area := area_large_sector + area_small_sector
  let shaded_area := (1 / 4) * Real.pi * (r_large ^ 2) - (1 / 4) * Real.pi * (r_small ^ 2)
  let shaded_percent := (shaded_area / total_area) * 100
  shaded_percent = 32 := by
  sorry

end shaded_region_occupies_32_percent_of_total_area_l164_164911


namespace legos_in_box_at_end_l164_164083

def initial_legos : ℕ := 500
def legos_used : ℕ := initial_legos / 2
def missing_legos : ℕ := 5
def remaining_legos := legos_used - missing_legos

theorem legos_in_box_at_end : remaining_legos = 245 := 
by
  sorry

end legos_in_box_at_end_l164_164083


namespace solve_for_x_l164_164370

def star (a b : ℝ) : ℝ := 3 * a - b

theorem solve_for_x :
  ∃ x : ℝ, star 2 (star 5 x) = 1 ∧ x = 10 := by
  sorry

end solve_for_x_l164_164370


namespace smallest_x_value_l164_164665

theorem smallest_x_value {x : ℝ} (h : abs (x + 4) = 15) : x = -19 :=
sorry

end smallest_x_value_l164_164665


namespace count_symmetric_numbers_count_symmetric_divisible_by_4_sum_symmetric_numbers_l164_164319

-- Definitions based on conditions
def is_symmetric_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 8 ∨ d = 6 ∨ d = 9

def symmetric_pair (a b : ℕ) : Prop :=
  (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1) ∨ (a = 8 ∧ b = 8) ∨ (a = 6 ∧ b = 9) ∨ (a = 9 ∧ b = 6)

-- 1. Prove the total number of 7-digit symmetric numbers
theorem count_symmetric_numbers : ∃ n, n = 300 := by
  sorry

-- 2. Prove the number of symmetric numbers divisible by 4
theorem count_symmetric_divisible_by_4 : ∃ n, n = 75 := by
  sorry

-- 3. Prove the total sum of these 7-digit symmetric numbers
theorem sum_symmetric_numbers : ∃ s, s = 1959460200 := by
  sorry

end count_symmetric_numbers_count_symmetric_divisible_by_4_sum_symmetric_numbers_l164_164319


namespace train_speed_l164_164263

theorem train_speed
  (length_m : ℝ)
  (time_s : ℝ)
  (h_length : length_m = 280.0224)
  (h_time : time_s = 25.2) :
  (length_m / 1000) / (time_s / 3600) = 40.0032 :=
by
  sorry

end train_speed_l164_164263


namespace B_days_solve_l164_164877

noncomputable def combined_work_rate (A_rate B_rate C_rate : ℝ) : ℝ := A_rate + B_rate + C_rate
noncomputable def A_rate : ℝ := 1 / 6
noncomputable def C_rate : ℝ := 1 / 7.5
noncomputable def combined_rate : ℝ := 1 / 2

theorem B_days_solve : ∃ (B_days : ℝ), combined_work_rate A_rate (1 / B_days) C_rate = combined_rate ∧ B_days = 5 :=
by
  use 5
  rw [←inv_div] -- simplifying the expression of 1/B_days
  have : ℝ := sorry -- steps to cancel and simplify, proving the equality
  sorry

end B_days_solve_l164_164877


namespace negate_proposition_l164_164308

theorem negate_proposition (x : ℝ) :
  (¬(x > 1 → x^2 > 1)) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by
  sorry

end negate_proposition_l164_164308


namespace cost_of_fencing_per_meter_l164_164829

def rectangular_farm_area : Real := 1200
def short_side_length : Real := 30
def total_cost : Real := 1440

theorem cost_of_fencing_per_meter : (total_cost / (short_side_length + (rectangular_farm_area / short_side_length) + Real.sqrt ((rectangular_farm_area / short_side_length)^2 + short_side_length^2))) = 12 :=
by
  sorry

end cost_of_fencing_per_meter_l164_164829


namespace solution_set_inequality_l164_164809

theorem solution_set_inequality (x : ℝ) (h : x - 3 / x > 2) :
    -1 < x ∧ x < 0 ∨ x > 3 :=
  sorry

end solution_set_inequality_l164_164809


namespace sum_of_midpoints_of_triangle_l164_164785

theorem sum_of_midpoints_of_triangle (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_of_triangle_l164_164785


namespace nonneg_real_inequality_l164_164186

theorem nonneg_real_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
    a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
sorry

end nonneg_real_inequality_l164_164186


namespace missed_both_shots_l164_164121

variables (p q : Prop)

theorem missed_both_shots : (¬p ∧ ¬q) ↔ ¬(p ∨ q) :=
by sorry

end missed_both_shots_l164_164121


namespace length_of_each_reel_l164_164031

theorem length_of_each_reel
  (reels : ℕ)
  (sections : ℕ)
  (length_per_section : ℕ)
  (total_sections : ℕ)
  (h1 : reels = 3)
  (h2 : length_per_section = 10)
  (h3 : total_sections = 30)
  : (total_sections * length_per_section) / reels = 100 := 
by
  sorry

end length_of_each_reel_l164_164031


namespace value_of_K_l164_164071

theorem value_of_K (K: ℕ) : 4^5 * 2^3 = 2^K → K = 13 := by
  sorry

end value_of_K_l164_164071


namespace class_6_1_students_l164_164072

noncomputable def number_of_students : ℕ :=
  let n := 30
  n

theorem class_6_1_students (n : ℕ) (t : ℕ) (h1 : (n + 1) * t = 527) (h2 : n % 5 = 0) : n = 30 :=
  by
  sorry

end class_6_1_students_l164_164072


namespace plant_species_numbering_impossible_l164_164349

theorem plant_species_numbering_impossible :
  ∀ (n m : ℕ), 2 ≤ n ∨ n ≤ 20000 ∧ 2 ≤ m ∨ m ≤ 20000 ∧ n ≠ m → 
  ∃ x y : ℕ, 2 ≤ x ∨ x ≤ 20000 ∧ 2 ≤ y ∨ y ≤ 20000 ∧ x ≠ y ∧
  (∀ k : ℕ, gcd x k = gcd n k ∧ gcd y k = gcd m k) :=
  by sorry

end plant_species_numbering_impossible_l164_164349


namespace slopes_and_angles_l164_164038

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

end slopes_and_angles_l164_164038


namespace num_of_loads_l164_164707

theorem num_of_loads (n : ℕ) (h1 : 7 * n = 42) : n = 6 :=
by
  sorry

end num_of_loads_l164_164707


namespace conditional_probability_of_A_given_target_hit_l164_164587

theorem conditional_probability_of_A_given_target_hit :
  (3 / 5 : ℚ) * ( ( 4 / 5 + 1 / 5) ) = (15 / 23 : ℚ) :=
  sorry

end conditional_probability_of_A_given_target_hit_l164_164587


namespace Z_4_3_eq_neg11_l164_164991

def Z (a b : ℤ) : ℤ := a^2 - 3 * a * b + b^2

theorem Z_4_3_eq_neg11 : Z 4 3 = -11 := 
by
  sorry

end Z_4_3_eq_neg11_l164_164991


namespace necessary_but_not_sufficient_condition_for_x_gt_2_l164_164945

theorem necessary_but_not_sufficient_condition_for_x_gt_2 :
  ∀ (x : ℝ), (2 / x < 1 → x > 2) ∧ (x > 2 → 2 / x < 1) → (¬ (x > 2 → 2 / x < 1) ∨ ¬ (2 / x < 1 → x > 2)) :=
by
  intro x h
  sorry

end necessary_but_not_sufficient_condition_for_x_gt_2_l164_164945


namespace quadratic_discriminant_one_solution_l164_164536

theorem quadratic_discriminant_one_solution (m : ℚ) : 
  (3 * (1 : ℚ))^2 - 12 * m = 0 → m = 49 / 12 := 
by {
  sorry
}

end quadratic_discriminant_one_solution_l164_164536


namespace pythagorean_triple_correct_l164_164901

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_correct :
  is_pythagorean_triple 5 12 13 ∧
  ¬ is_pythagorean_triple 7 9 11 ∧
  ¬ is_pythagorean_triple 6 9 12 ∧
  ¬ is_pythagorean_triple (3/10) (4/10) (5/10) :=
by
  sorry

end pythagorean_triple_correct_l164_164901


namespace sum_possible_values_of_y_l164_164743

theorem sum_possible_values_of_y (y : ℝ) (h : y^2 = 36) : y = 6 ∨ y = -6 → (6 + (-6) = 0) :=
by
  sorry

end sum_possible_values_of_y_l164_164743


namespace total_books_l164_164378

theorem total_books (books_last_month : ℕ) (goal_factor : ℕ) (books_this_month : ℕ) (total_books : ℕ) 
  (h1 : books_last_month = 4) 
  (h2 : goal_factor = 2) 
  (h3 : books_this_month = goal_factor * books_last_month) 
  (h4 : total_books = books_last_month + books_this_month) 
  : total_books = 12 := 
by
  sorry

end total_books_l164_164378


namespace find_two_numbers_l164_164082

noncomputable def quadratic_roots (a b : ℝ) : Prop :=
  a = (5 + Real.sqrt 5) / 2 ∧ b = (5 - Real.sqrt 5) / 2

theorem find_two_numbers (a b : ℝ) (h1 : a * b = 5) (h2 : 2 * (a * b) / (a + b) = 5 / 2) :
  quadratic_roots a b :=
by
  sorry

end find_two_numbers_l164_164082


namespace factorize_expression_l164_164269

variable (a : ℝ) (b : ℝ)

theorem factorize_expression : 2 * a - 8 * a * b^2 = 2 * a * (1 - 2 * b) * (1 + 2 * b) := by
  sorry

end factorize_expression_l164_164269


namespace crescent_moon_falcata_area_l164_164860

/-
Prove that the area of the crescent moon falcata, which is bounded by:
1. A portion of the circle with radius 4 centered at (0,0) in the second quadrant.
2. A portion of the circle with radius 2 centered at (0,2) in the second quadrant.
3. The line segment from (0,0) to (-4,0).
is equal to 6π.
-/
theorem crescent_moon_falcata_area :
  let radius_large := 4
  let radius_small := 2
  let area_large := (1 / 2) * (π * (radius_large ^ 2))
  let area_small := (1 / 2) * (π * (radius_small ^ 2))
  (area_large - area_small) = 6 * π := by
  sorry

end crescent_moon_falcata_area_l164_164860


namespace no_valid_abc_l164_164915

theorem no_valid_abc : 
  ∀ (a b c : ℕ), (100 * a + 10 * b + c) % 15 = 0 → (10 * b + c) % 4 = 0 → a > b → b > c → false :=
by
  intros a b c habc_mod15 hbc_mod4 h_ab_gt h_bc_gt
  sorry

end no_valid_abc_l164_164915


namespace imaginary_part_of_z_l164_164355

theorem imaginary_part_of_z (z : ℂ) (h : (z / (1 - I)) = (3 + I)) : z.im = -2 :=
sorry

end imaginary_part_of_z_l164_164355


namespace tip_percentage_l164_164790

variable (L : ℝ) (T : ℝ)
 
theorem tip_percentage (h : L = 60.50) (h1 : T = 72.6) :
  ((T - L) / L) * 100 = 20 :=
by
  sorry

end tip_percentage_l164_164790


namespace find_number_of_elements_l164_164401

theorem find_number_of_elements (n S : ℕ) (h1 : S + 26 = 19 * n) (h2 : S + 76 = 24 * n) : n = 10 := 
sorry

end find_number_of_elements_l164_164401


namespace plywood_cut_difference_l164_164859

/-- A proof problem to determine the positive difference between the greatest possible
perimeter and the least possible perimeter of congruent pieces resulting from cutting 
a 6-foot by 9-foot rectangular plywood into 6 congruent rectangles with no wood leftover 
or lost --/
theorem plywood_cut_difference :
  ∃ (perimeter_max perimeter_min : ℕ), 
  let piece1 := 1 * 9
  let piece2 := 1 * 6
  let piece3 := 2 * 3
  let perimeter1 := 2 * (1 + 9)
  let perimeter2 := 2 * (1 + 6)
  let perimeter3 := 2 * (2 + 3)
  perimeter_max = perimeter1 ∧
  perimeter_min = perimeter3 ∧
  (perimeter_max - perimeter_min) = 10 :=
sorry

end plywood_cut_difference_l164_164859


namespace number_of_acute_triangles_l164_164049

def num_triangles : ℕ := 7
def right_triangles : ℕ := 2
def obtuse_triangles : ℕ := 3

theorem number_of_acute_triangles :
  num_triangles - right_triangles - obtuse_triangles = 2 := by
  sorry

end number_of_acute_triangles_l164_164049


namespace common_difference_of_arithmetic_sequence_l164_164492

variable {a : ℕ → ℝ} (a2 a5 : ℝ)
variable (h1 : a 2 = 9) (h2 : a 5 = 33)

theorem common_difference_of_arithmetic_sequence :
  ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ d = 8 := by
  sorry

end common_difference_of_arithmetic_sequence_l164_164492


namespace physics_class_size_l164_164952

variable (students : ℕ)
variable (physics math both : ℕ)

-- Conditions
def conditions := students = 75 ∧ physics = 2 * (math - both) + both ∧ both = 9

-- The proof goal
theorem physics_class_size : conditions students physics math both → physics = 56 := 
by 
  sorry

end physics_class_size_l164_164952


namespace penguins_count_l164_164834

variable (P B : ℕ)

theorem penguins_count (h1 : B = 2 * P) (h2 : P + B = 63) : P = 21 :=
by
  sorry

end penguins_count_l164_164834


namespace two_lines_perpendicular_to_same_line_are_parallel_l164_164653

/- Define what it means for two lines to be perpendicular -/
def perpendicular (l m : Line) : Prop :=
  -- A placeholder definition for perpendicularity, replace with the actual definition
  sorry

/- Define what it means for two lines to be parallel -/
def parallel (l m : Line) : Prop :=
  -- A placeholder definition for parallelism, replace with the actual definition
  sorry

/- Given: Two lines l1 and l2 that are perpendicular to the same line l3 -/
variables (l1 l2 l3 : Line)
variable (h1 : perpendicular l1 l3)
variable (h2 : perpendicular l2 l3)

/- Prove: l1 and l2 are parallel to each other -/
theorem two_lines_perpendicular_to_same_line_are_parallel :
  parallel l1 l2 :=
  sorry

end two_lines_perpendicular_to_same_line_are_parallel_l164_164653


namespace certain_number_divided_by_10_l164_164913
-- Broad import to bring in necessary libraries

-- Define the constants and hypotheses
variable (x : ℝ)
axiom condition : 5 * x = 100

-- Theorem to prove the required equality
theorem certain_number_divided_by_10 : (x / 10) = 2 :=
by
  -- The proof is skipped by sorry
  sorry

end certain_number_divided_by_10_l164_164913


namespace none_of_these_valid_l164_164446

variables {x y z w u v : ℝ}

def statement_1 (x y z w : ℝ) := x > y → z < w
def statement_2 (z w u v : ℝ) := z > w → u < v

theorem none_of_these_valid (h₁ : statement_1 x y z w) (h₂ : statement_2 z w u v) :
  ¬ ( (x < y → u < v) ∨ (u < v → x < y) ∨ (u > v → x > y) ∨ (x > y → u > v) ) :=
by {
  sorry
}

end none_of_these_valid_l164_164446


namespace color_crafter_secret_codes_l164_164633

theorem color_crafter_secret_codes :
  8^5 = 32768 := by
  sorry

end color_crafter_secret_codes_l164_164633


namespace office_distance_eq_10_l164_164956

noncomputable def distance_to_office (D T : ℝ) : Prop :=
  D = 10 * (T + 10 / 60) ∧ D = 15 * (T - 10 / 60)

theorem office_distance_eq_10 (D T : ℝ) (h : distance_to_office D T) : D = 10 :=
by
  sorry

end office_distance_eq_10_l164_164956


namespace john_speed_when_runs_alone_l164_164428

theorem john_speed_when_runs_alone (x : ℝ) : 
  (6 * (1/2) + x * (1/2) = 5) → x = 4 :=
by
  intro h
  linarith

end john_speed_when_runs_alone_l164_164428


namespace cuboid_area_correct_l164_164416

def cuboid_surface_area (length breadth height : ℕ) :=
  2 * (length * height) + 2 * (breadth * height) + 2 * (length * breadth)

theorem cuboid_area_correct : cuboid_surface_area 4 6 5 = 148 := by
  sorry

end cuboid_area_correct_l164_164416


namespace division_by_fraction_l164_164315

theorem division_by_fraction :
  (12 : ℝ) / (1 / 6) = 72 :=
by
  sorry

end division_by_fraction_l164_164315


namespace expression_value_l164_164752

theorem expression_value (a b c : ℝ) (h : a * b + b * c + c * a = 3) : 
  (a * (b^2 + 3)) / (a + b) + (b * (c^2 + 3)) / (b + c) + (c * (a^2 + 3)) / (c + a) = 6 := 
by
  sorry

end expression_value_l164_164752


namespace polynomial_satisfies_condition_l164_164964

-- Define P as a real polynomial
def P (a : ℝ) (X : ℝ) : ℝ := a * X

-- Define a statement that needs to be proven
theorem polynomial_satisfies_condition (P : ℝ → ℝ) :
  (∀ X : ℝ, P (2 * X) = 2 * P X) ↔ ∃ a : ℝ, ∀ X : ℝ, P X = a * X :=
by
  sorry

end polynomial_satisfies_condition_l164_164964


namespace fixed_point_exists_line_intersects_circle_shortest_chord_l164_164872

noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25
noncomputable def line_l (x y m : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem fixed_point_exists : ∃ P : ℝ × ℝ, (∀ m : ℝ, line_l P.1 P.2 m) ∧ P = (3, 1) :=
by
  sorry

theorem line_intersects_circle : ∀ m : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ, line_l x₁ y₁ m ∧ line_l x₂ y₂ m ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
by
  sorry

theorem shortest_chord : ∃ m : ℝ, m = -3/4 ∧ (∀ x y, line_l x y m ↔ 2 * x - y - 5 = 0) :=
by
  sorry

end fixed_point_exists_line_intersects_circle_shortest_chord_l164_164872


namespace area_of_square_efgh_proof_l164_164767

noncomputable def area_of_square_efgh : ℝ :=
  let original_square_side_length := 3
  let radius_of_circles := (3 * Real.sqrt 2) / 2
  let efgh_side_length := original_square_side_length + 2 * radius_of_circles 
  efgh_side_length ^ 2

theorem area_of_square_efgh_proof :
  area_of_square_efgh = 27 + 18 * Real.sqrt 2 :=
by
  sorry

end area_of_square_efgh_proof_l164_164767


namespace ten_factorial_mod_thirteen_l164_164285

open Nat

theorem ten_factorial_mod_thirteen :
  (10! % 13) = 6 := by
  sorry

end ten_factorial_mod_thirteen_l164_164285


namespace choir_final_score_l164_164803

theorem choir_final_score (content_score sing_score spirit_score : ℕ)
  (content_weight sing_weight spirit_weight : ℝ)
  (h_content : content_weight = 0.30) 
  (h_sing : sing_weight = 0.50) 
  (h_spirit : spirit_weight = 0.20) 
  (h_content_score : content_score = 90)
  (h_sing_score : sing_score = 94)
  (h_spirit_score : spirit_score = 95) :
  content_weight * content_score + sing_weight * sing_score + spirit_weight * spirit_score = 93 := by
  sorry

end choir_final_score_l164_164803


namespace rows_of_seats_l164_164551

theorem rows_of_seats (students sections_per_row students_per_section : ℕ) (h1 : students_per_section = 2) (h2 : sections_per_row = 2) (h3 : students = 52) :
  (students / students_per_section / sections_per_row) = 13 :=
sorry

end rows_of_seats_l164_164551


namespace numDogsInPetStore_l164_164102

-- Definitions from conditions
variables {D P : Nat}

-- Theorem statement - no proof provided
theorem numDogsInPetStore (h1 : D + P = 15) (h2 : 4 * D + 2 * P = 42) : D = 6 :=
by
  sorry

end numDogsInPetStore_l164_164102


namespace rice_flour_weights_l164_164348

variables (r f : ℝ)

theorem rice_flour_weights :
  (8 * r + 6 * f = 550) ∧ (4 * r + 7 * f = 375) → (r = 50) ∧ (f = 25) :=
by
  intro h
  sorry

end rice_flour_weights_l164_164348


namespace fraction_values_l164_164755

theorem fraction_values (a b c : ℚ) (h1 : a / b = 2) (h2 : b / c = 4 / 3) : c / a = 3 / 8 := 
by
  sorry

end fraction_values_l164_164755


namespace contrapositive_proposition_l164_164406

theorem contrapositive_proposition (x : ℝ) : (x ≤ -1 ∨ x ≥ 1) → (x^2 ≥ 1) :=
by
  sorry

end contrapositive_proposition_l164_164406


namespace scientific_notation_400000000_l164_164449

theorem scientific_notation_400000000 : 400000000 = 4 * 10^8 :=
by
  sorry

end scientific_notation_400000000_l164_164449


namespace four_digit_palindrome_perfect_squares_l164_164588

theorem four_digit_palindrome_perfect_squares : 
  ∃ (count : ℕ), count = 2 ∧ 
  (∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → 
            ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
            n = 1001 * a + 110 * b ∧ 
            ∃ k : ℕ, k * k = n) → count = 2 := by
  sorry

end four_digit_palindrome_perfect_squares_l164_164588


namespace converse_of_posImpPosSquare_l164_164441

-- Let's define the condition proposition first
def posImpPosSquare (x : ℝ) : Prop := x > 0 → x^2 > 0

-- Now, we state the converse we need to prove
theorem converse_of_posImpPosSquare (x : ℝ) (h : posImpPosSquare x) : x^2 > 0 → x > 0 := sorry

end converse_of_posImpPosSquare_l164_164441


namespace total_cups_sold_is_46_l164_164538

-- Define the number of cups sold last week
def cups_sold_last_week : ℕ := 20

-- Define the percentage increase
def percentage_increase : ℕ := 30

-- Calculate the number of cups sold this week
def cups_sold_this_week : ℕ := cups_sold_last_week + (cups_sold_last_week * percentage_increase / 100)

-- Calculate the total number of cups sold over both weeks
def total_cups_sold : ℕ := cups_sold_last_week + cups_sold_this_week

-- State the theorem to prove the total number of cups sold
theorem total_cups_sold_is_46 : total_cups_sold = 46 := sorry

end total_cups_sold_is_46_l164_164538


namespace find_selling_price_l164_164139

def cost_price : ℝ := 59
def selling_price_for_loss : ℝ := 52
def loss := cost_price - selling_price_for_loss

theorem find_selling_price (sp : ℝ) : (sp - cost_price = loss) → sp = 66 :=
by
  sorry

end find_selling_price_l164_164139


namespace sqrt_360000_eq_600_l164_164617

theorem sqrt_360000_eq_600 : Real.sqrt 360000 = 600 := by
  sorry

end sqrt_360000_eq_600_l164_164617


namespace debra_probability_theorem_l164_164334

-- Define event for Debra's coin flipping game starting with "HTT"
def debra_coin_game_event : Prop := 
  let heads_probability : ℝ := 0.5
  let tails_probability : ℝ := 0.5
  let initial_prob : ℝ := heads_probability * tails_probability * tails_probability
  let Q : ℝ := 1 / 3  -- the computed probability of getting HH after HTT
  let final_probability : ℝ := initial_prob * Q
  final_probability = 1 / 24

-- The theorem statement
theorem debra_probability_theorem :
  debra_coin_game_event := 
by
  sorry

end debra_probability_theorem_l164_164334


namespace qin_jiushao_operations_required_l164_164487

def polynomial (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

theorem qin_jiushao_operations_required : 
  (∃ x : ℝ, polynomial x = (((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1)) →
  (∃ m a : ℕ, m = 5 ∧ a = 5) := by
  sorry

end qin_jiushao_operations_required_l164_164487


namespace problem_statement_l164_164771

theorem problem_statement (m : ℝ) (h_m : 0 ≤ m ∧ m ≤ 1) (x : ℝ) :
    (m * x^2 - 2 * x - m ≥ 2) ↔ (x ≤ -1) := sorry

end problem_statement_l164_164771


namespace percent_gain_on_transaction_l164_164609

theorem percent_gain_on_transaction :
  ∀ (x : ℝ), (850 : ℝ) * x + (50 : ℝ) * (1.10 * ((850 : ℝ) * x / 800)) = 850 * x * (1 + 0.06875) := 
by
  intro x
  sorry

end percent_gain_on_transaction_l164_164609


namespace rectangular_plot_dimensions_l164_164984

theorem rectangular_plot_dimensions (a b : ℝ) 
  (h_area : a * b = 800) 
  (h_perimeter_fencing : 2 * a + b = 100) :
  (a = 40 ∧ b = 20) ∨ (a = 10 ∧ b = 80) := 
sorry

end rectangular_plot_dimensions_l164_164984


namespace shaded_to_white_area_ratio_l164_164161

-- Define the problem
theorem shaded_to_white_area_ratio :
  let total_triangles_shaded := 5
  let total_triangles_white := 3
  let ratio_shaded_to_white := total_triangles_shaded / total_triangles_white
  ratio_shaded_to_white = (5 : ℚ)/(3 : ℚ) := by
  -- Proof steps should be provided here, but "sorry" is used to skip the proof.
  sorry

end shaded_to_white_area_ratio_l164_164161


namespace solve_congruence_l164_164112

theorem solve_congruence :
  ∃ a m : ℕ, (8 * (x : ℕ) + 1) % 12 = 5 % 12 ∧ m ≥ 2 ∧ a < m ∧ x ≡ a [MOD m] ∧ a + m = 5 :=
by
  sorry

end solve_congruence_l164_164112


namespace correct_equation_l164_164230

theorem correct_equation (x : ℤ) : 232 + x = 3 * (146 - x) :=
sorry

end correct_equation_l164_164230


namespace isosceles_triangles_count_isosceles_triangles_l164_164291

theorem isosceles_triangles (x : ℕ) (b : ℕ) : 
  (2 * x + b = 29 ∧ b % 2 = 1) ∧ (b < 14) → 
  (b = 1 ∧ x = 14 ∨ b = 3 ∧ x = 13 ∨ b = 5 ∧ x = 12 ∨ b = 7 ∧ x = 11 ∨ b = 9 ∧ x = 10) :=
by sorry

theorem count_isosceles_triangles : 
  (∃ x b, (2 * x + b = 29 ∧ b % 2 = 1) ∧ (b < 14)) → 
  (5 = 5) :=
by sorry

end isosceles_triangles_count_isosceles_triangles_l164_164291


namespace woman_total_coins_l164_164240

theorem woman_total_coins
  (num_each_coin : ℕ)
  (h : 1 * num_each_coin + 5 * num_each_coin + 10 * num_each_coin + 25 * num_each_coin + 100 * num_each_coin = 351)
  : 5 * num_each_coin = 15 :=
by
  sorry

end woman_total_coins_l164_164240


namespace commute_distance_l164_164297

noncomputable def distance_to_work (total_time : ℕ) (speed_to_work : ℕ) (speed_to_home : ℕ) : ℕ :=
  let d := (speed_to_work * speed_to_home * total_time) / (speed_to_work + speed_to_home)
  d

-- Given conditions
def speed_to_work : ℕ := 45
def speed_to_home : ℕ := 30
def total_time : ℕ := 1

-- Proof problem statement
theorem commute_distance : distance_to_work total_time speed_to_work speed_to_home = 18 :=
by
  sorry

end commute_distance_l164_164297


namespace exists_positive_integer_pow_not_integer_l164_164787

theorem exists_positive_integer_pow_not_integer
  (α β : ℝ)
  (hαβ : α ≠ β)
  (h_non_int : ¬(↑⌊α⌋ = α ∧ ↑⌊β⌋ = β)) :
  ∃ n : ℕ, 0 < n ∧ ¬∃ k : ℤ, α^n - β^n = k :=
by
  sorry

end exists_positive_integer_pow_not_integer_l164_164787


namespace number_of_valid_lines_l164_164605

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def lines_passing_through_point (x_int : ℕ) (y_int : ℕ) (p : ℕ × ℕ) : Prop :=
  p.1 * y_int + p.2 * x_int = x_int * y_int

theorem number_of_valid_lines (p : ℕ × ℕ) : 
  ∃! l : ℕ × ℕ, is_prime (l.1) ∧ is_power_of_two (l.2) ∧ lines_passing_through_point l.1 l.2 p :=
sorry

end number_of_valid_lines_l164_164605


namespace sqrt_eq_cubrt_l164_164214

theorem sqrt_eq_cubrt (x : ℝ) (h : Real.sqrt x = x^(1/3)) : x = 0 ∨ x = 1 :=
by
  sorry

end sqrt_eq_cubrt_l164_164214


namespace fruit_baskets_l164_164067

def apple_choices := 8 -- From 0 to 7 apples
def orange_choices := 13 -- From 0 to 12 oranges

theorem fruit_baskets (a : ℕ) (o : ℕ) (ha : a = 7) (ho : o = 12) :
  (apple_choices * orange_choices) - 1 = 103 := by
  sorry

end fruit_baskets_l164_164067


namespace triangle_inequality_l164_164511

theorem triangle_inequality (a b c : ℝ) (h : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
sorry

end triangle_inequality_l164_164511


namespace crayons_taken_out_l164_164739

-- Define the initial and remaining number of crayons
def initial_crayons : ℕ := 7
def remaining_crayons : ℕ := 4

-- Define the proposition to prove
theorem crayons_taken_out : initial_crayons - remaining_crayons = 3 := by
  sorry

end crayons_taken_out_l164_164739


namespace find_divisor_l164_164749

theorem find_divisor : 
  ∀ (dividend quotient remainder divisor : ℕ), 
    dividend = 140 →
    quotient = 9 →
    remainder = 5 →
    dividend = (divisor * quotient) + remainder →
    divisor = 15 :=
by
  intros dividend quotient remainder divisor hd hq hr hdiv
  sorry

end find_divisor_l164_164749


namespace total_miles_l164_164162

theorem total_miles (miles_Darius : Int) (miles_Julia : Int) (h1 : miles_Darius = 679) (h2 : miles_Julia = 998) :
  miles_Darius + miles_Julia = 1677 :=
by
  sorry

end total_miles_l164_164162


namespace line_intersects_curve_equal_segments_l164_164802

theorem line_intersects_curve_equal_segments (k m : ℝ)
  (A B C : ℝ × ℝ)
  (hA_curve : A.2 = A.1^3 - 6 * A.1^2 + 13 * A.1 - 8)
  (hB_curve : B.2 = B.1^3 - 6 * B.1^2 + 13 * B.1 - 8)
  (hC_curve : C.2 = C.1^3 - 6 * C.1^2 + 13 * C.1 - 8)
  (h_lineA : A.2 = k * A.1 + m)
  (h_lineB : B.2 = k * B.1 + m)
  (h_lineC : C.2 = k * C.1 + m)
  (h_midpoint : 2 * B.1 = A.1 + C.1 ∧ 2 * B.2 = A.2 + C.2)
  : 2 * k + m = 2 :=
sorry

end line_intersects_curve_equal_segments_l164_164802


namespace hours_rained_l164_164876

theorem hours_rained (total_hours non_rain_hours rained_hours : ℕ)
 (h_total : total_hours = 8)
 (h_non_rain : non_rain_hours = 6)
 (h_rain_eq : rained_hours = total_hours - non_rain_hours) :
 rained_hours = 2 := 
by
  sorry

end hours_rained_l164_164876


namespace unit_prices_minimize_cost_l164_164362

theorem unit_prices (x y : ℕ) (h1 : x + 2 * y = 40) (h2 : 2 * x + 3 * y = 70) :
  x = 20 ∧ y = 10 :=
by {
  sorry -- proof would go here
}

theorem minimize_cost (total_pieces : ℕ) (cost_A cost_B : ℕ) 
  (total_cost : ℕ → ℕ)
  (h3 : total_pieces = 60) 
  (h4 : ∀ m, cost_A * m + cost_B * (total_pieces - m) = total_cost m) 
  (h5 : ∀ m, cost_A * m + cost_B * (total_pieces - m) ≥ 800) 
  (h6 : ∀ m, m ≥ (total_pieces - m) / 2) :
  total_cost 20 = 800 :=
by {
  sorry -- proof would go here
}

end unit_prices_minimize_cost_l164_164362


namespace twenty_four_points_game_l164_164741

theorem twenty_four_points_game :
  let a := (-6 : ℚ)
  let b := (3 : ℚ)
  let c := (4 : ℚ)
  let d := (10 : ℚ)
  3 * (d - a + c) = 24 := 
by
  sorry

end twenty_four_points_game_l164_164741


namespace coupon_probability_l164_164684

-- We will define our conditions
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Now we state our problem
theorem coupon_probability :
  ∀ (C6_6 C11_3 C17_9 : ℕ),
  C6_6 = combination 6 6 →
  C11_3 = combination 11 3 →
  C17_9 = combination 17 9 →
  (C6_6 * C11_3) / C17_9 = 3 / 442 :=
by
  intros C6_6 C11_3 C17_9 hC6_6 hC11_3 hC17_9
  rw [hC6_6, hC11_3, hC17_9]
  sorry

end coupon_probability_l164_164684


namespace four_digit_divisors_l164_164924

theorem four_digit_divisors :
  ∀ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 →
  (1000 * a + 100 * b + 10 * c + d ∣ 1000 * b + 100 * c + 10 * d + a ∨
   1000 * a + 100 * b + 10 * c + d ∣ 1000 * c + 100 * d + 10 * a + b ∨
   1000 * a + 100 * b + 10 * c + d ∣ 1000 * d + 100 * a + 10 * b + c) →
  ∃ (e f : ℕ), e = a ∧ f = b ∧ (e ≠ 0 ∧ f ≠ 0) ∧ (1000 * e + 100 * e + 10 * f + f = 1000 * a + 100 * b + 10 * a + b) ∧
  (1000 * e + 100 * e + 10 * f + f ∣ 1000 * b + 100 * c + 10 * d + a ∨
   1000 * e + 100 * e + 10 * f + f ∣ 1000 * c + 100 * d + 10 * a + b ∨
   1000 * e + 100 * e + 10 * f + f ∣ 1000 * d + 100 * a + 10 * b + c) := 
by
  sorry

end four_digit_divisors_l164_164924


namespace convex_polygon_sides_l164_164887

theorem convex_polygon_sides (n : ℕ) (h1 : 180 * (n - 2) - 90 = 2790) : n = 18 :=
sorry

end convex_polygon_sides_l164_164887


namespace compound_O_atoms_l164_164080

theorem compound_O_atoms (Cu_weight C_weight O_weight compound_weight : ℝ)
  (Cu_atoms : ℕ) (C_atoms : ℕ) (O_atoms : ℕ)
  (hCu : Cu_weight = 63.55)
  (hC : C_weight = 12.01)
  (hO : O_weight = 16.00)
  (h_compound_weight : compound_weight = 124)
  (h_atoms : Cu_atoms = 1 ∧ C_atoms = 1)
  : O_atoms = 3 :=
sorry

end compound_O_atoms_l164_164080


namespace find_m_l164_164593

theorem find_m (m : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = (m^2 - 5*m + 7)*x^(m-2)) 
  (h2 : ∀ x, f (-x) = - f x) : 
  m = 3 :=
by
  sorry

end find_m_l164_164593


namespace age_of_other_man_l164_164172

theorem age_of_other_man
  (n : ℕ) (average_age_before : ℕ) (average_age_after : ℕ) (age_of_one_man : ℕ) (average_age_women : ℕ) 
  (h1 : n = 9)
  (h2 : average_age_after = average_age_before + 4)
  (h3 : age_of_one_man = 36)
  (h4 : average_age_women = 52) :
  (68 - 36 = 32) := 
by
  sorry

end age_of_other_man_l164_164172


namespace at_least_50_singers_l164_164019

def youth_summer_village (total people_not_working people_with_families max_subset : ℕ) : Prop :=
  total = 100 ∧ 
  people_not_working = 50 ∧ 
  people_with_families = 25 ∧ 
  max_subset = 50

theorem at_least_50_singers (S : ℕ) (h : youth_summer_village 100 50 25 50) : S ≥ 50 :=
by
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end at_least_50_singers_l164_164019


namespace symmetric_about_y_axis_l164_164062

theorem symmetric_about_y_axis (m n : ℝ) (A B : ℝ × ℝ)
  (hA : A = (-3, 2 * m - 1))
  (hB : B = (n + 1, 4))
  (symmetry : A.1 = -B.1)
  : m = 2.5 ∧ n = 2 :=
by
  sorry

end symmetric_about_y_axis_l164_164062


namespace maria_sold_in_first_hour_l164_164129

variable (x : ℕ)

-- Conditions
def sold_in_first_hour := x
def sold_in_second_hour := 2
def average_sold_in_two_hours := 6

-- Proof Goal
theorem maria_sold_in_first_hour :
  (sold_in_first_hour + sold_in_second_hour) / 2 = average_sold_in_two_hours → sold_in_first_hour = 10 :=
by
  sorry

end maria_sold_in_first_hour_l164_164129


namespace cosine_of_acute_angle_l164_164993

theorem cosine_of_acute_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : Real.sin α = 4 / 5) : Real.cos α = 3 / 5 :=
by
  sorry

end cosine_of_acute_angle_l164_164993


namespace stratified_sampling_difference_l164_164922

theorem stratified_sampling_difference
  (male_athletes : ℕ := 56)
  (female_athletes : ℕ := 42)
  (sample_size : ℕ := 28)
  (H_total : male_athletes + female_athletes = 98)
  (H_sample_frac : sample_size = 28)
  : (56 * (sample_size / 98) - 42 * (sample_size / 98) = 4) :=
sorry

end stratified_sampling_difference_l164_164922


namespace sqrt_pos_condition_l164_164515

theorem sqrt_pos_condition (x : ℝ) : (1 - x) ≥ 0 ↔ x ≤ 1 := 
by 
  sorry

end sqrt_pos_condition_l164_164515


namespace semicircle_area_increase_l164_164537

noncomputable def area_semicircle (r : ℝ) : ℝ :=
  (1 / 2) * Real.pi * r^2

noncomputable def percent_increase (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

theorem semicircle_area_increase :
  let r_long := 6
  let r_short := 4
  let area_long := 2 * area_semicircle r_long
  let area_short := 2 * area_semicircle r_short
  percent_increase area_short area_long = 125 :=
by
  let r_long := 6
  let r_short := 4
  let area_long := 2 * area_semicircle r_long
  let area_short := 2 * area_semicircle r_short
  have : area_semicircle r_long = 18 * Real.pi := by sorry
  have : area_semicircle r_short = 8 * Real.pi := by sorry
  have : area_long = 36 * Real.pi := by sorry
  have : area_short = 16 * Real.pi := by sorry
  have : percent_increase area_short area_long = 125 := by sorry
  exact this

end semicircle_area_increase_l164_164537


namespace quadratic_solutions_l164_164043

theorem quadratic_solutions (x : ℝ) : (2 * x^2 + 5 * x + 3 = 0) → (x = -1 ∨ x = -3 / 2) :=
by {
  sorry
}

end quadratic_solutions_l164_164043


namespace sum_of_three_numbers_l164_164950

variable (x y z : ℝ)

theorem sum_of_three_numbers :
  y = 5 → 
  (x + y + z) / 3 = x + 10 →
  (x + y + z) / 3 = z - 15 →
  x + y + z = 30 :=
by
  intros hy h1 h2
  rw [hy] at h1 h2
  sorry

end sum_of_three_numbers_l164_164950


namespace cistern_fill_time_l164_164074

theorem cistern_fill_time (fillA emptyB : ℕ) (hA : fillA = 8) (hB : emptyB = 12) : (24 : ℕ) = 24 :=
by
  sorry

end cistern_fill_time_l164_164074


namespace closed_path_has_even_length_l164_164312

   theorem closed_path_has_even_length 
     (u d r l : ℤ) 
     (hu : u = d) 
     (hr : r = l) : 
     ∃ k : ℤ, 2 * (u + r) = 2 * k :=
   by
     sorry
   
end closed_path_has_even_length_l164_164312


namespace minimum_treasure_buried_l164_164839

def palm_tree (n : Nat) := n < 30

def sign_condition (n : Nat) (k : Nat) : Prop :=
  if n = 15 then palm_tree n ∧ k = 15
  else if n = 8 then palm_tree n ∧ k = 8
  else if n = 4 then palm_tree n ∧ k = 4
  else if n = 3 then palm_tree n ∧ k = 3
  else False

def treasure_condition (n : Nat) (k : Nat) : Prop :=
  (n ≤ k) → ∀ x, palm_tree x → sign_condition x k → x ≠ n

theorem minimum_treasure_buried : ∃ k, k = 15 ∧ ∀ n, treasure_condition n k :=
by
  sorry

end minimum_treasure_buried_l164_164839


namespace total_weight_puffy_muffy_l164_164385

def scruffy_weight : ℕ := 12
def muffy_weight : ℕ := scruffy_weight - 3
def puffy_weight : ℕ := muffy_weight + 5

theorem total_weight_puffy_muffy : puffy_weight + muffy_weight = 23 := 
by
  sorry

end total_weight_puffy_muffy_l164_164385


namespace divisible_by_120_l164_164056

theorem divisible_by_120 (n : ℤ) : 120 ∣ (n ^ 6 + 2 * n ^ 5 - n ^ 2 - 2 * n) :=
by sorry

end divisible_by_120_l164_164056


namespace banks_policies_for_seniors_justified_l164_164557

-- Defining conditions
def better_credit_repayment_reliability : Prop := sorry
def stable_pension_income : Prop := sorry
def indirect_younger_relative_contributions : Prop := sorry
def pensioners_inclination_to_save : Prop := sorry
def regular_monthly_income : Prop := sorry
def preference_for_long_term_deposits : Prop := sorry

-- Lean theorem statement using the conditions
theorem banks_policies_for_seniors_justified :
  better_credit_repayment_reliability →
  stable_pension_income →
  indirect_younger_relative_contributions →
  pensioners_inclination_to_save →
  regular_monthly_income →
  preference_for_long_term_deposits →
  (banks_should_offer_higher_deposit_and_lower_loan_rates_to_seniors : Prop) :=
by
  -- Insert proof here that given all the conditions the conclusion follows
  sorry -- proof not required, so skipping

end banks_policies_for_seniors_justified_l164_164557


namespace jim_total_cars_l164_164848

theorem jim_total_cars (B F C : ℕ) (h1 : B = 4 * F) (h2 : F = 2 * C + 3) (h3 : B = 220) :
  B + F + C = 301 :=
by
  sorry

end jim_total_cars_l164_164848


namespace trains_meet_after_time_l164_164419

/-- Given the lengths of two trains, the initial distance between them, and their speeds,
prove that they will meet after approximately 2.576 seconds. --/
theorem trains_meet_after_time 
  (length_train1 : ℝ) (length_train2 : ℝ) (initial_distance : ℝ)
  (speed_train1_kmph : ℝ) (speed_train2_mps : ℝ) :
  length_train1 = 87.5 →
  length_train2 = 94.3 →
  initial_distance = 273.2 →
  speed_train1_kmph = 65 →
  speed_train2_mps = 88 →
  abs ((initial_distance / ((speed_train1_kmph * 1000 / 3600) + speed_train2_mps)) - 2.576) < 0.001 := by
  sorry

end trains_meet_after_time_l164_164419


namespace percentage_of_600_eq_half_of_900_l164_164237

theorem percentage_of_600_eq_half_of_900 : 
  ∃ P : ℝ, (P / 100) * 600 = 0.5 * 900 ∧ P = 75 := by
  -- Proof goes here
  sorry

end percentage_of_600_eq_half_of_900_l164_164237


namespace lori_earnings_l164_164034

theorem lori_earnings
    (red_cars : ℕ)
    (white_cars : ℕ)
    (cost_red_car : ℕ)
    (cost_white_car : ℕ)
    (rental_time_hours : ℕ)
    (rental_time_minutes : ℕ)
    (correct_earnings : ℕ) :
    red_cars = 3 →
    white_cars = 2 →
    cost_red_car = 3 →
    cost_white_car = 2 →
    rental_time_hours = 3 →
    rental_time_minutes = rental_time_hours * 60 →
    correct_earnings = 2340 →
    (red_cars * cost_red_car + white_cars * cost_white_car) * rental_time_minutes = correct_earnings :=
by
  intros
  sorry

end lori_earnings_l164_164034


namespace minimum_value_quadratic_function_l164_164851

noncomputable def quadratic_function (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

theorem minimum_value_quadratic_function : ∀ x, x ≥ 0 → quadratic_function x ≥ 1 :=
by
  sorry

end minimum_value_quadratic_function_l164_164851


namespace surface_area_of_given_cube_l164_164625

-- Define the edge length condition
def edge_length_of_cube (sum_edge_lengths : ℕ) :=
  sum_edge_lengths / 12

-- Define the surface area of a cube given an edge length
def surface_area_of_cube (edge_length : ℕ) :=
  6 * (edge_length * edge_length)

-- State the theorem
theorem surface_area_of_given_cube : 
  edge_length_of_cube 36 = 3 ∧ surface_area_of_cube 3 = 54 :=
by
  -- We leave the proof as an exercise.
  sorry

end surface_area_of_given_cube_l164_164625


namespace number_of_boxes_l164_164619

-- Definitions based on conditions
def bottles_per_box := 50
def bottle_capacity := 12
def fill_fraction := 3 / 4
def total_water := 4500

-- Question rephrased as a proof problem
theorem number_of_boxes (h1 : bottles_per_box = 50)
                        (h2 : bottle_capacity = 12)
                        (h3 : fill_fraction = 3 / 4)
                        (h4 : total_water = 4500) :
  4500 / ((12 : ℝ) * (3 / 4)) / 50 = 10 := 
by {
  sorry
}

end number_of_boxes_l164_164619


namespace work_completion_days_l164_164828

theorem work_completion_days (A_days B_days : ℕ) (hA : A_days = 3) (hB : B_days = 6) : 
  (1 / ((1 / (A_days : ℚ)) + (1 / (B_days : ℚ)))) = 2 := 
by
  sorry

end work_completion_days_l164_164828


namespace area_of_triangle_ABC_l164_164057

theorem area_of_triangle_ABC 
  (BD DC : ℕ) 
  (h_ratio : BD / DC = 4 / 3)
  (S_BEC : ℕ) 
  (h_BEC : S_BEC = 105) :
  ∃ (S_ABC : ℕ), S_ABC = 315 := 
sorry

end area_of_triangle_ABC_l164_164057


namespace find_c_l164_164187

def p (x : ℝ) : ℝ := 3 * x - 8
def q (x : ℝ) (c : ℝ) : ℝ := 5 * x - c

theorem find_c (c : ℝ) (h : p (q 3 c) = 14) : c = 23 / 3 :=
by
  sorry

end find_c_l164_164187


namespace truck_travel_distance_l164_164415

theorem truck_travel_distance (miles_per_5gallons miles distance gallons rate : ℕ)
  (h1 : miles_per_5gallons = 150) 
  (h2 : gallons = 5) 
  (h3 : rate = miles_per_5gallons / gallons) 
  (h4 : gallons = 7) 
  (h5 : distance = rate * gallons) : 
  distance = 210 := 
by sorry

end truck_travel_distance_l164_164415


namespace yards_after_8_marathons_l164_164193

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

end yards_after_8_marathons_l164_164193


namespace cannot_transform_with_swap_rows_and_columns_l164_164711

def initialTable : Matrix (Fin 3) (Fin 3) ℕ :=
![![1, 2, 3], ![4, 5, 6], ![7, 8, 9]]

def goalTable : Matrix (Fin 3) (Fin 3) ℕ :=
![![1, 4, 7], ![2, 5, 8], ![3, 6, 9]]

theorem cannot_transform_with_swap_rows_and_columns :
  ¬ ∃ (is_transformed_by_swapping : Matrix (Fin 3) (Fin 3) ℕ → Matrix (Fin 3) (Fin 3) ℕ → Prop),
    is_transformed_by_swapping initialTable goalTable :=
by sorry

end cannot_transform_with_swap_rows_and_columns_l164_164711


namespace average_coins_per_day_l164_164999

theorem average_coins_per_day :
  let a := 10
  let d := 10
  let n := 7
  let extra := 20
  let total_coins := a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d) + (a + 6 * d + extra)
  total_coins = 300 →
  total_coins / n = 300 / 7 :=
by
  sorry

end average_coins_per_day_l164_164999


namespace sum_of_rel_prime_ints_l164_164811

theorem sum_of_rel_prime_ints (a b : ℕ) (h1 : a < 15) (h2 : b < 15) (h3 : a * b + a + b = 71)
    (h4 : Nat.gcd a b = 1) : a + b = 16 := by
  sorry

end sum_of_rel_prime_ints_l164_164811


namespace positive_integer_solutions_of_inequality_l164_164658

theorem positive_integer_solutions_of_inequality : 
  {x : ℕ | 3 * x - 1 ≤ 2 * x + 3} = {1, 2, 3, 4} :=
by
  sorry

end positive_integer_solutions_of_inequality_l164_164658


namespace maximal_q_for_broken_line_l164_164585

theorem maximal_q_for_broken_line :
  ∃ q : ℝ, (∀ i : ℕ, 0 ≤ i → i < 5 → ∀ A_i : ℝ, (A_i = q ^ i)) ∧ 
  (q = (1 + Real.sqrt 5) / 2) := sorry

end maximal_q_for_broken_line_l164_164585


namespace solve_for_x_l164_164866

theorem solve_for_x (x : ℝ) (h : 4^x = Real.sqrt 64) : x = 3 / 2 :=
sorry

end solve_for_x_l164_164866


namespace imaginary_part_of_complex_z_l164_164477

noncomputable def complex_z : ℂ := (1 + Complex.I) / (1 - Complex.I) + (1 - Complex.I) ^ 2

theorem imaginary_part_of_complex_z : complex_z.im = -1 := by
  sorry

end imaginary_part_of_complex_z_l164_164477


namespace problem1_problem2_l164_164347

def A := { x : ℝ | -2 < x ∧ x ≤ 4 }
def B := { x : ℝ | 2 - x < 1 }
def U := ℝ
def complement_B := { x : ℝ | x ≤ 1 }

theorem problem1 : { x : ℝ | 1 < x ∧ x ≤ 4 } = { x : ℝ | x ∈ A ∧ x ∈ B } := 
by sorry

theorem problem2 : { x : ℝ | x ≤ 4 } = { x : ℝ | x ∈ A ∨ x ∈ complement_B } := 
by sorry

end problem1_problem2_l164_164347


namespace find_a_to_make_f_odd_l164_164087

noncomputable def f (a : ℝ) (x : ℝ): ℝ := x^3 * (Real.log (Real.exp x + 1) + a * x)

theorem find_a_to_make_f_odd :
  (∃ a : ℝ, ∀ x : ℝ, f a (-x) = -f a x) ↔ a = -1/2 :=
by 
  sorry

end find_a_to_make_f_odd_l164_164087


namespace equal_sum_squares_l164_164075

open BigOperators

-- Definitions
def n := 10

-- Assuming x and y to be arrays that hold the number of victories and losses for each player respectively.
variables {x y : Fin n → ℝ}

-- Conditions
axiom pair_meet_once : ∀ i : Fin n, x i + y i = (n - 1)

-- Theorem to be proved
theorem equal_sum_squares : ∑ i : Fin n, x i ^ 2 = ∑ i : Fin n, y i ^ 2 :=
by
  sorry

end equal_sum_squares_l164_164075


namespace total_shaded_area_l164_164506

theorem total_shaded_area (S T U : ℝ) 
  (h1 : 12 / S = 4)
  (h2 : S / T = 2)
  (h3 : T / U = 2) :
  1 * (S * S) + 4 * (T * T) + 8 * (U * U) = 22.5 := by
sorry

end total_shaded_area_l164_164506


namespace first_range_is_30_l164_164967

theorem first_range_is_30 
  (R2 R3 : ℕ)
  (h1 : R2 = 26)
  (h2 : R3 = 32)
  (h3 : min 26 (min 30 32) = 30) : 
  ∃ R1 : ℕ, R1 = 30 :=
  sorry

end first_range_is_30_l164_164967


namespace fraction_of_field_planted_l164_164910

theorem fraction_of_field_planted : 
  let field_area := 5 * 6
  let triangle_area := (5 * 6) / 2
  let a := (41 * 3) / 33  -- derived from the given conditions
  let square_area := a^2
  let planted_area := triangle_area - square_area
  (planted_area / field_area) = (404 / 841) := 
by
  sorry

end fraction_of_field_planted_l164_164910


namespace soldier_rearrangement_20x20_soldier_rearrangement_21x21_l164_164104

theorem soldier_rearrangement_20x20 (d : ℝ) : d ≤ 10 * Real.sqrt 2 :=
by
  -- Problem (a) setup and conditions
  sorry

theorem soldier_rearrangement_21x21 (d : ℝ) : d ≤ 10 * Real.sqrt 2 :=
by
  -- Problem (b) setup and conditions
  sorry

end soldier_rearrangement_20x20_soldier_rearrangement_21x21_l164_164104


namespace radius_ratio_ge_sqrt2plus1_l164_164346

theorem radius_ratio_ge_sqrt2plus1 (r R a h : ℝ) (h1 : 2 * a ≠ 0) (h2 : h ≠ 0) 
  (hr : r = a * h / (a + Real.sqrt (a ^ 2 + h ^ 2)))
  (hR : R = (2 * a ^ 2 + h ^ 2) / (2 * h)) : 
  R / r ≥ 1 + Real.sqrt 2 := 
sorry

end radius_ratio_ge_sqrt2plus1_l164_164346


namespace parabola_exists_l164_164117

noncomputable def parabola_conditions (a b : ℝ) : Prop :=
  (a + b = -3) ∧ (4 * a - 2 * b = 12)

noncomputable def translated_min_equals_six (m : ℝ) : Prop :=
  (m > 0) ∧ ((-1 - 2 + m)^2 - 3 = 6) ∨ ((3 - 2 - m)^2 - 3 = 6)

theorem parabola_exists (a b m : ℝ) (x y : ℝ) :
  parabola_conditions a b → y = x^2 + b * x + 1 → translated_min_equals_six m →
  (y = x^2 - 4 * x + 1) ∧ (m = 6 ∨ m = 4) := 
by 
  sorry

end parabola_exists_l164_164117


namespace compute_expression_l164_164000

theorem compute_expression (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
by
  sorry

end compute_expression_l164_164000


namespace john_paid_after_tax_l164_164649

-- Definitions based on problem conditions
def original_cost : ℝ := 200
def tax_rate : ℝ := 0.15

-- Definition of the tax amount
def tax_amount : ℝ := tax_rate * original_cost

-- Definition of the total amount paid
def total_amount_paid : ℝ := original_cost + tax_amount

-- Theorem statement for the proof
theorem john_paid_after_tax : total_amount_paid = 230 := by
  sorry

end john_paid_after_tax_l164_164649


namespace find_e_l164_164906

theorem find_e (a e : ℕ) (h1: a = 105) (h2: a ^ 3 = 21 * 25 * 45 * e) : e = 49 :=
sorry

end find_e_l164_164906


namespace Tom_needs_11_25_hours_per_week_l164_164398

theorem Tom_needs_11_25_hours_per_week
  (summer_weeks: ℕ) (summer_weeks_val: summer_weeks = 8)
  (summer_hours_per_week: ℕ) (summer_hours_per_week_val: summer_hours_per_week = 45)
  (summer_earnings: ℝ) (summer_earnings_val: summer_earnings = 3600)
  (rest_weeks: ℕ) (rest_weeks_val: rest_weeks = 40)
  (rest_earnings_goal: ℝ) (rest_earnings_goal_val: rest_earnings_goal = 4500) :
  (rest_earnings_goal / (summer_earnings / (summer_hours_per_week * summer_weeks))) / rest_weeks = 11.25 :=
by
  simp [summer_earnings_val, rest_earnings_goal_val, summer_hours_per_week_val, summer_weeks_val]
  sorry

end Tom_needs_11_25_hours_per_week_l164_164398


namespace ratio_steel_to_tin_l164_164706

def mass_copper (C : ℝ) := C = 90
def total_weight (S C T : ℝ) := 20 * S + 20 * C + 20 * T = 5100
def mass_steel (S C : ℝ) := S = C + 20

theorem ratio_steel_to_tin (S T C : ℝ)
  (hC : mass_copper C)
  (hTW : total_weight S C T)
  (hS : mass_steel S C) :
  S / T = 2 :=
by
  sorry

end ratio_steel_to_tin_l164_164706


namespace range_of_m_l164_164461

open Set Real

def A : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def B (m : ℝ) : Set ℝ := { x | x^2 - (m + 3) * x + m^2 = 0 }

theorem range_of_m (m : ℝ) :
  (A ∪ (univ \ B m)) = univ ↔ m ∈ Iio (-1) ∪ Ici 3 :=
sorry

end range_of_m_l164_164461


namespace range_of_m_l164_164306

theorem range_of_m (m : ℝ) : 
    (∀ x y : ℝ, (x^2 / (4 - m) + y^2 / (m - 3) = 1) → 
    4 - m > 0 ∧ m - 3 > 0 ∧ m - 3 > 4 - m) → 
    (7/2 < m ∧ m < 4) :=
sorry

end range_of_m_l164_164306


namespace volume_of_pyramid_l164_164942

noncomputable def volume_pyramid : ℝ :=
  let a := 9
  let b := 12
  let s := 15
  let base_area := a * b
  let diagonal := Real.sqrt (a^2 + b^2)
  let half_diagonal := diagonal / 2
  let height := Real.sqrt (s^2 - half_diagonal^2)
  (1 / 3) * base_area * height

theorem volume_of_pyramid :
  volume_pyramid = 36 * Real.sqrt 168.75 := by
  sorry

end volume_of_pyramid_l164_164942


namespace big_rectangle_width_l164_164286

theorem big_rectangle_width
  (W : ℝ)
  (h₁ : ∃ l w : ℝ, l = 40 ∧ w = W)
  (h₂ : ∃ l' w' : ℝ, l' = l / 2 ∧ w' = w / 2)
  (h_area : 200 = l' * w') :
  W = 20 :=
by sorry

end big_rectangle_width_l164_164286


namespace correct_operation_l164_164554

theorem correct_operation : 
  ¬(3 * x^2 + 2 * x^2 = 6 * x^4) ∧ 
  ¬((-2 * x^2)^3 = -6 * x^6) ∧ 
  ¬(x^3 * x^2 = x^6) ∧ 
  (-6 * x^2 * y^3 / (2 * x^2 * y^2) = -3 * y) :=
by
  sorry

end correct_operation_l164_164554


namespace different_digits_probability_l164_164522

noncomputable def number_nonidentical_probability : ℚ :=
  let total_numbers := 900
  let identical_numbers := 9
  -- The probability of identical digits.
  let identical_probability := identical_numbers / total_numbers
  -- The probability of non-identical digits.
  1 - identical_probability

theorem different_digits_probability : number_nonidentical_probability = 99 / 100 := by
  sorry

end different_digits_probability_l164_164522


namespace quadratic_expression_l164_164026

theorem quadratic_expression (b c : ℤ) : 
  (∀ x : ℝ, (x^2 - 20*x + 49 = (x + b)^2 + c)) → (b + c = -61) :=
by
  sorry

end quadratic_expression_l164_164026


namespace necessary_but_not_sufficient_condition_for_geometric_sequence_l164_164442

theorem necessary_but_not_sufficient_condition_for_geometric_sequence
  (a b c : ℝ) :
  (∃ (r : ℝ), a = r * b ∧ b = r * c) → (b^2 = a * c) ∧ ¬((b^2 = a * c) → (∃ (r : ℝ), a = r * b ∧ b = r * c)) := 
by
  sorry

end necessary_but_not_sufficient_condition_for_geometric_sequence_l164_164442


namespace arithmetic_sequence_solution_l164_164728

-- Definitions of a, b, c, and d in terms of d and sequence difference
def is_in_arithmetic_sequence (a b c d : ℝ) (diff : ℝ) : Prop :=
  a + diff = b ∧ b + diff = c ∧ c + diff = d

-- Conditions
def pos_real_sequence (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

def product_condition (a b c d : ℝ) (prod : ℝ) : Prop :=
  a * b * c * d = prod

-- The resulting value of d
def d_value_as_fraction (d : ℝ) : Prop :=
  d = (3 + Real.sqrt 95) / (Real.sqrt 2)

-- Proof statement
theorem arithmetic_sequence_solution :
  ∃ a b c d : ℝ, pos_real_sequence a b c d ∧ 
                 is_in_arithmetic_sequence a b c d (Real.sqrt 2) ∧ 
                 product_condition a b c d 2021 ∧ 
                 d_value_as_fraction d :=
sorry

end arithmetic_sequence_solution_l164_164728


namespace arithmetic_sequence_geometric_condition_l164_164846

theorem arithmetic_sequence_geometric_condition (a : ℕ → ℤ) (d : ℤ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d = 3)
  (h3 : ∃ k, a (k+3) * a k = (a (k+1)) * (a (k+2))) :
  a 2 = -9 :=
by
  sorry

end arithmetic_sequence_geometric_condition_l164_164846


namespace tangent_line_curve_l164_164740

theorem tangent_line_curve (a b : ℚ) 
  (h1 : 3 * a + b = 1) 
  (h2 : a + b = 2) : 
  b - a = 3 := 
by 
  sorry

end tangent_line_curve_l164_164740


namespace car_rental_cost_per_mile_l164_164358

def daily_rental_rate := 29.0
def total_amount_paid := 46.12
def miles_driven := 214.0

theorem car_rental_cost_per_mile : 
  (total_amount_paid - daily_rental_rate) / miles_driven = 0.08 := 
by
  sorry

end car_rental_cost_per_mile_l164_164358


namespace fraction_paint_left_after_third_day_l164_164631

noncomputable def original_paint : ℝ := 2
noncomputable def paint_after_first_day : ℝ := original_paint - (1 / 2 * original_paint)
noncomputable def paint_after_second_day : ℝ := paint_after_first_day - (1 / 4 * paint_after_first_day)
noncomputable def paint_after_third_day : ℝ := paint_after_second_day - (1 / 2 * paint_after_second_day)

theorem fraction_paint_left_after_third_day :
  paint_after_third_day / original_ppaint = 3 / 8 :=
sorry

end fraction_paint_left_after_third_day_l164_164631


namespace product_of_numbers_l164_164300

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 404) : x * y = 40 := 
sorry

end product_of_numbers_l164_164300


namespace chucks_team_final_score_l164_164498

variable (RedTeamScore : ℕ) (scoreDifference : ℕ)

-- Given conditions
def red_team_score := RedTeamScore = 76
def score_difference := scoreDifference = 19

-- Question: What was the final score of Chuck's team?
def chucks_team_score (RedTeamScore scoreDifference : ℕ) : ℕ := 
  RedTeamScore + scoreDifference

-- Proof statement
theorem chucks_team_final_score : red_team_score 76 ∧ score_difference 19 → chucks_team_score 76 19 = 95 :=
by
  sorry

end chucks_team_final_score_l164_164498


namespace eagles_win_at_least_three_matches_l164_164296

-- Define the conditions
def n : ℕ := 5
def p : ℝ := 0.5

-- Binomial coefficient function
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Probability function for the binomial distribution
noncomputable def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial n k) * p^k * (1 - p)^(n - k)

-- Theorem stating the main result
theorem eagles_win_at_least_three_matches :
  (binomial_prob n 3 p + binomial_prob n 4 p + binomial_prob n 5 p) = 1 / 2 :=
by
  sorry

end eagles_win_at_least_three_matches_l164_164296


namespace cayli_combinations_l164_164364

theorem cayli_combinations (art_choices sports_choices music_choices : ℕ)
  (h1 : art_choices = 2)
  (h2 : sports_choices = 3)
  (h3 : music_choices = 4) :
  art_choices * sports_choices * music_choices = 24 := by
  sorry

end cayli_combinations_l164_164364


namespace pentagon_angles_sum_l164_164011

theorem pentagon_angles_sum {α β γ δ ε : ℝ} (h1 : α + β + γ + δ + ε = 180) (h2 : α = 50) :
  β + ε = 230 := 
sorry

end pentagon_angles_sum_l164_164011


namespace minimize_x_plus_y_on_circle_l164_164654

theorem minimize_x_plus_y_on_circle (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 1) : x + y ≥ 2 :=
by
  sorry

end minimize_x_plus_y_on_circle_l164_164654


namespace geom_seq_a5_l164_164081

noncomputable def S3 (a1 q : ℚ) : ℚ := a1 + a1 * q^2
noncomputable def a (a1 q : ℚ) (n : ℕ) : ℚ := a1 * q^(n - 1)

theorem geom_seq_a5 (a1 q : ℚ) (hS3 : S3 a1 q = 5 * a1) (ha7 : a a1 q 7 = 2) :
  a a1 q 5 = 1 / 2 :=
by
  sorry

end geom_seq_a5_l164_164081


namespace total_players_count_l164_164875

def kabadi_players : ℕ := 10
def kho_kho_only_players : ℕ := 35
def both_games_players : ℕ := 5

theorem total_players_count : kabadi_players + kho_kho_only_players - both_games_players = 40 :=
by
  sorry

end total_players_count_l164_164875


namespace distance_difference_l164_164946

-- Definitions related to the problem conditions
variables (v D_AB D_BC D_AC : ℝ)

-- Conditions
axiom h1 : D_AB = v * 7
axiom h2 : D_BC = v * 5
axiom h3 : D_AC = 6
axiom h4 : D_AC = D_AB + D_BC

-- Theorem for proof problem
theorem distance_difference : D_AB - D_BC = 1 :=
by sorry

end distance_difference_l164_164946


namespace total_questions_l164_164647

theorem total_questions (S C I : ℕ) (h1 : S = 73) (h2 : C = 91) (h3 : S = C - 2 * I) : C + I = 100 :=
sorry

end total_questions_l164_164647


namespace find_d_l164_164520

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_d (d e : ℝ) (h1 : -(-6) / 3 = 2) (h2 : 3 + d + e - 6 = 9) (h3 : -d / 3 = 6) : d = -18 :=
by
  sorry

end find_d_l164_164520


namespace find_greater_number_l164_164108

theorem find_greater_number (x y : ℕ) 
  (h1 : x + y = 40)
  (h2 : x - y = 12) : x = 26 :=
by
  sorry

end find_greater_number_l164_164108


namespace julian_needs_more_legos_l164_164663

-- Definitions based on the conditions
def legos_julian_has := 400
def legos_per_airplane := 240
def number_of_airplanes := 2

-- Calculate the total number of legos required for two airplane models
def total_legos_needed := legos_per_airplane * number_of_airplanes

-- Calculate the number of additional legos Julian needs
def additional_legos_needed := total_legos_needed - legos_julian_has

-- Statement that needs to be proven
theorem julian_needs_more_legos : additional_legos_needed = 80 := by
  sorry

end julian_needs_more_legos_l164_164663


namespace transform_to_quadratic_l164_164467

theorem transform_to_quadratic :
  (∀ x : ℝ, (x + 1) ^ 2 + (x - 2) * (x + 2) = 1 ↔ 2 * x ^ 2 + 2 * x - 4 = 0) :=
sorry

end transform_to_quadratic_l164_164467


namespace find_a5_l164_164037

theorem find_a5 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = 2 * n * (n + 1))
  (ha : ∀ n ≥ 2, a n = S n - S (n - 1)) : 
  a 5 = 20 := 
sorry

end find_a5_l164_164037


namespace ratio_of_legs_of_triangles_l164_164493

theorem ratio_of_legs_of_triangles (s a b : ℝ) (h1 : 0 < s)
  (h2 : a = s / 2)
  (h3 : b = (s * Real.sqrt 7) / 2) :
  b / a = Real.sqrt 7 := by
  sorry

end ratio_of_legs_of_triangles_l164_164493


namespace megan_final_balance_percentage_l164_164276

noncomputable def initial_balance_usd := 125.0
noncomputable def increase_percentage_babysitting := 0.25
noncomputable def exchange_rate_usd_to_eur_1 := 0.85
noncomputable def decrease_percentage_shoes := 0.20
noncomputable def exchange_rate_eur_to_usd := 1.15
noncomputable def increase_percentage_stocks := 0.15
noncomputable def decrease_percentage_medical := 0.10
noncomputable def exchange_rate_usd_to_eur_2 := 0.88

theorem megan_final_balance_percentage :
  let new_balance_after_babysitting := initial_balance_usd * (1 + increase_percentage_babysitting)
  let balance_in_eur := new_balance_after_babysitting * exchange_rate_usd_to_eur_1
  let balance_after_shoes := balance_in_eur * (1 - decrease_percentage_shoes)
  let balance_back_to_usd := balance_after_shoes * exchange_rate_eur_to_usd
  let balance_after_stocks := balance_back_to_usd * (1 + increase_percentage_stocks)
  let balance_after_medical := balance_after_stocks * (1 - decrease_percentage_medical)
  let final_balance_in_eur := balance_after_medical * exchange_rate_usd_to_eur_2
  let initial_balance_in_eur := initial_balance_usd * exchange_rate_usd_to_eur_1
  (final_balance_in_eur / initial_balance_in_eur) * 100 = 104.75 := by
  sorry

end megan_final_balance_percentage_l164_164276


namespace min_value_inverse_sum_l164_164222

theorem min_value_inverse_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 2 * b = 2) :
  (1 / a + 2 / b) ≥ 9 / 2 :=
sorry

end min_value_inverse_sum_l164_164222


namespace second_car_mileage_l164_164098

theorem second_car_mileage (x : ℝ) : 
  (150 / 50) + (150 / x) + (150 / 15) = 56 / 2 → x = 10 :=
by
  intro h
  sorry

end second_car_mileage_l164_164098


namespace find_positive_number_l164_164891

theorem find_positive_number (x n : ℝ) (h₁ : (x + 1) ^ 2 = n) (h₂ : (x - 5) ^ 2 = n) : n = 9 := 
sorry

end find_positive_number_l164_164891


namespace negation_proposition_real_l164_164339

theorem negation_proposition_real :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
by
  sorry

end negation_proposition_real_l164_164339


namespace calc_mixed_number_expr_l164_164963

theorem calc_mixed_number_expr :
  53 * (3 + 1 / 4 - (3 + 3 / 4)) / (1 + 2 / 3 + (2 + 2 / 5)) = -6 - 57 / 122 := 
by
  sorry

end calc_mixed_number_expr_l164_164963


namespace parallel_perpendicular_implies_l164_164696

variables {Line : Type} {Plane : Type}
variables (m n : Line) (α β : Plane)

-- Conditions
axiom distinct_lines : m ≠ n
axiom distinct_planes : α ≠ β

-- Parallel and Perpendicular relationships
axiom parallel : Line → Plane → Prop
axiom perpendicular : Line → Plane → Prop

-- Given conditions
axiom parallel_mn : parallel m n
axiom perpendicular_mα : perpendicular m α

-- Proof statement
theorem parallel_perpendicular_implies (h1 : parallel m n) (h2 : perpendicular m α) : perpendicular n α :=
sorry

end parallel_perpendicular_implies_l164_164696


namespace students_at_end_l164_164424

def initial_students := 11
def students_left := 6
def new_students := 42

theorem students_at_end (init : ℕ := initial_students) (left : ℕ := students_left) (new : ℕ := new_students) :
    (init - left + new) = 47 := 
by
  sorry

end students_at_end_l164_164424


namespace pair_with_15_l164_164990

theorem pair_with_15 (s : List ℕ) (h : s = [49, 29, 9, 40, 22, 15, 53, 33, 13, 47]) :
  ∃ (t : List (ℕ × ℕ)), (∀ (x y : ℕ), (x, y) ∈ t → x + y = 62) ∧ (15, 47) ∈ t := by
  sorry

end pair_with_15_l164_164990


namespace sufficient_condition_l164_164132

-- Definitions of propositions p and q
variables (p q : Prop)

-- Theorem statement
theorem sufficient_condition (h : ¬(p ∨ q)) : ¬p :=
by sorry

end sufficient_condition_l164_164132


namespace integer_subset_property_l164_164888

theorem integer_subset_property (M : Set ℤ) (h1 : ∃ a ∈ M, a > 0) (h2 : ∃ b ∈ M, b < 0)
(h3 : ∀ {a b : ℤ}, a ∈ M → b ∈ M → 2 * a ∈ M ∧ a + b ∈ M)
: ∀ a b : ℤ, a ∈ M → b ∈ M → a - b ∈ M :=
by
  sorry

end integer_subset_property_l164_164888


namespace sqrt_of_4_equals_2_l164_164097

theorem sqrt_of_4_equals_2 : Real.sqrt 4 = 2 :=
by sorry

end sqrt_of_4_equals_2_l164_164097


namespace arcsin_neg_sqrt3_div_2_l164_164759

theorem arcsin_neg_sqrt3_div_2 : 
  Real.arcsin (- (Real.sqrt 3 / 2)) = - (Real.pi / 3) := 
by sorry

end arcsin_neg_sqrt3_div_2_l164_164759


namespace shaded_area_of_overlap_l164_164788

structure Rectangle where
  width : ℕ
  height : ℕ

structure Parallelogram where
  base : ℕ
  height : ℕ

def area_of_rectangle (r : Rectangle) : ℕ :=
  r.width * r.height

def area_of_parallelogram (p : Parallelogram) : ℕ :=
  p.base * p.height

def overlapping_area_square (side : ℕ) : ℕ :=
  side * side

theorem shaded_area_of_overlap 
  (r : Rectangle)
  (p : Parallelogram)
  (overlapping_side : ℕ)
  (h1 : r.width = 4)
  (h2 : r.height = 12)
  (h3 : p.base = 10)
  (h4 : p.height = 4)
  (h5 : overlapping_side = 4) :
  area_of_rectangle r + area_of_parallelogram p - overlapping_area_square overlapping_side = 72 :=
by
  sorry

end shaded_area_of_overlap_l164_164788


namespace completing_square_correct_l164_164524

theorem completing_square_correct :
  ∀ x : ℝ, (x^2 - 4 * x + 2 = 0) ↔ ((x - 2)^2 = 2) := 
by
  intros x
  sorry

end completing_square_correct_l164_164524


namespace yen_per_cad_l164_164469

theorem yen_per_cad (yen cad : ℝ) (h : yen / cad = 5000 / 60) : yen = 83 := by
  sorry

end yen_per_cad_l164_164469


namespace isosceles_triangle_perimeter_l164_164667

theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 6) (h₂ : b = 3) (h₃ : a > b) : a + a + b = 15 :=
by
  sorry

end isosceles_triangle_perimeter_l164_164667


namespace price_decrease_l164_164893

theorem price_decrease (current_price original_price : ℝ) (h1 : current_price = 684) (h2 : original_price = 900) :
  ((original_price - current_price) / original_price) * 100 = 24 :=
by
  sorry

end price_decrease_l164_164893


namespace maria_money_difference_l164_164941

-- Defining constants for Maria's money when she arrived and left the fair
def money_at_arrival : ℕ := 87
def money_at_departure : ℕ := 16

-- Calculating the expected difference
def expected_difference : ℕ := 71

-- Statement: proving that the difference between money_at_arrival and money_at_departure is expected_difference
theorem maria_money_difference : money_at_arrival - money_at_departure = expected_difference := by
  sorry

end maria_money_difference_l164_164941


namespace batsman_average_after_20th_innings_l164_164835

theorem batsman_average_after_20th_innings 
    (score_20th_innings : ℕ)
    (previous_avg_increase : ℕ)
    (total_innings : ℕ)
    (never_not_out : Prop)
    (previous_avg : ℕ)
    : score_20th_innings = 90 →
      previous_avg_increase = 2 →
      total_innings = 20 →
      previous_avg = (19 * previous_avg + score_20th_innings) / total_innings →
      ((19 * previous_avg + score_20th_innings) / total_innings) + previous_avg_increase = 52 :=
by 
  sorry

end batsman_average_after_20th_innings_l164_164835


namespace correct_average_weight_l164_164113

-- Definitions
def initial_average_weight : ℝ := 58.4
def number_of_boys : ℕ := 20
def misread_weight_initial : ℝ := 56
def misread_weight_correct : ℝ := 68

-- Correct average weight
theorem correct_average_weight : 
  let initial_total_weight := initial_average_weight * (number_of_boys : ℝ)
  let difference := misread_weight_correct - misread_weight_initial
  let correct_total_weight := initial_total_weight + difference
  let correct_average_weight := correct_total_weight / (number_of_boys : ℝ)
  correct_average_weight = 59 :=
by
  -- Insert the proof steps if needed
  sorry

end correct_average_weight_l164_164113


namespace tangent_line_at_1_0_monotonic_intervals_l164_164779

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x + 2 * Real.log x

noncomputable def f_derivative (x : ℝ) (a : ℝ) : ℝ := (2 * x^2 - a * x + 2) / x

theorem tangent_line_at_1_0 (a : ℝ) (h : a = 1) :
  ∀ x y : ℝ, 
  (f x a, f 1 a) = (0, x - 1) → 
  y = 3 * x - 3 := 
sorry

theorem monotonic_intervals (a : ℝ) :
  (∀ x : ℝ, 0 < x → f_derivative x a ≥ 0) ↔ (a ≤ 4) ∧ 
  (∀ x : ℝ, 0 < x → 
    (0 < x ∧ x < (a - Real.sqrt (a^2 - 16)) / 4) ∨ 
    ((a + Real.sqrt (a^2 - 16)) / 4 < x) 
  ) :=
sorry

end tangent_line_at_1_0_monotonic_intervals_l164_164779


namespace min_value_d1_d2_l164_164210

noncomputable def min_distance_sum : ℝ :=
  let d1 (u : ℝ) : ℝ := (1 / 5) * abs (3 * Real.cos u - 4 * Real.sin u - 10)
  let d2 (u : ℝ) : ℝ := 3 - Real.cos u
  let d_sum (u : ℝ) : ℝ := d1 u + d2 u
  ((5 - (4 * Real.sqrt 5 / 5)))

theorem min_value_d1_d2 :
  ∀ (P : ℝ × ℝ) (u : ℝ),
    P = (Real.cos u, Real.sin u) →
    (P.1 ^ 2 + P.2 ^ 2 = 1) →
    let d1 := (1 / 5) * abs (3 * P.1 - 4 * P.2 - 10)
    let d2 := 3 - P.1
    d1 + d2 ≥ (5 - (4 * Real.sqrt 5 / 5)) :=
by
  sorry

end min_value_d1_d2_l164_164210


namespace sum_f_values_l164_164502

theorem sum_f_values (a b c d e f g : ℕ) 
  (h1: 100 * a * b = 100 * d)
  (h2: c * d * e = 100 * d)
  (h3: b * d * f = 100 * d)
  (h4: b * f = 100)
  (h5: 100 * d = 100) : 
  100 + 50 + 25 + 20 + 10 + 5 + 4 + 2 + 1 = 217 :=
by
  sorry

end sum_f_values_l164_164502


namespace BD_distance_16_l164_164975

noncomputable def distanceBD (DA AB : ℝ) (angleBDA : ℝ) : ℝ :=
  (DA^2 + AB^2 - 2 * DA * AB * Real.cos angleBDA).sqrt

theorem BD_distance_16 :
  distanceBD 10 14 (60 * Real.pi / 180) = 16 := by
  sorry

end BD_distance_16_l164_164975


namespace boys_and_girls_total_l164_164734

theorem boys_and_girls_total (c : ℕ) (h_lollipop_fraction : c = 90) 
  (h_one_third_lollipops : c / 3 = 30)
  (h_lollipops_shared : 30 / 3 = 10) 
  (h_candy_caness_shared : 60 / 2 = 30) : 
  10 + 30 = 40 :=
by
  simp [h_one_third_lollipops, h_lollipops_shared, h_candy_caness_shared]

end boys_and_girls_total_l164_164734


namespace average_price_per_book_l164_164717

def books_from_shop1 := 42
def price_from_shop1 := 520
def books_from_shop2 := 22
def price_from_shop2 := 248

def total_books := books_from_shop1 + books_from_shop2
def total_price := price_from_shop1 + price_from_shop2
def average_price := total_price / total_books

theorem average_price_per_book : average_price = 12 := by
  sorry

end average_price_per_book_l164_164717


namespace maximum_k_value_l164_164982

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / (x - 1)
noncomputable def g (x : ℝ) (k : ℕ) : ℝ := k / x

theorem maximum_k_value (c : ℝ) (h_c : c > 1) : 
  (∃ a b : ℝ, 0 < a ∧ a < b ∧ b < c ∧ f c = f a ∧ f a = g b 3) ∧ 
  (∀ k : ℕ, k > 3 → ¬ ∃ a b : ℝ, 0 < a ∧ a < b ∧ b < c ∧ f c = f a ∧ f a = g b k) :=
sorry

end maximum_k_value_l164_164982


namespace percentage_politics_not_local_politics_l164_164527

variables (total_reporters : ℝ) 
variables (reporters_cover_local_politics : ℝ) 
variables (reporters_not_cover_politics : ℝ)

theorem percentage_politics_not_local_politics :
  total_reporters = 100 → 
  reporters_cover_local_politics = 5 → 
  reporters_not_cover_politics = 92.85714285714286 → 
  (total_reporters - reporters_not_cover_politics) - reporters_cover_local_politics = 2.14285714285714 := 
by 
  intros ht hr hn
  rw [ht, hr, hn]
  norm_num


end percentage_politics_not_local_politics_l164_164527


namespace min_value_M_l164_164849

theorem min_value_M 
  (S_n : ℕ → ℝ)
  (T_n : ℕ → ℝ)
  (a : ℕ → ℝ)
  (h1 : ∀ n, S_n n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1)))
  (h2 : a 4 - a 2 = 8)
  (h3 : a 3 + a 5 = 26)
  (h4 : ∀ n, T_n n = S_n n / n^2) :
  ∃ M : ℝ, M = 2 ∧ (∀ n > 0, T_n n ≤ M) :=
by sorry

end min_value_M_l164_164849


namespace four_leaved_clovers_percentage_l164_164610

noncomputable def percentage_of_four_leaved_clovers (clovers total_clovers purple_four_leaved_clovers : ℕ ) : ℝ := 
  (purple_four_leaved_clovers * 4 * 100) / total_clovers 

theorem four_leaved_clovers_percentage :
  percentage_of_four_leaved_clovers 500 500 25 = 20 := 
by
  -- application of conditions and arithmetic simplification.
  sorry

end four_leaved_clovers_percentage_l164_164610


namespace razorback_tshirt_money_l164_164420

noncomputable def money_made_from_texas_tech_game (tshirt_price : ℕ) (total_sold : ℕ) (arkansas_sold : ℕ) : ℕ :=
  tshirt_price * (total_sold - arkansas_sold)

theorem razorback_tshirt_money :
  money_made_from_texas_tech_game 78 186 172 = 1092 := by
  sorry

end razorback_tshirt_money_l164_164420


namespace greatest_divisor_l164_164246

theorem greatest_divisor (d : ℕ) :
  (6215 % d = 23 ∧ 7373 % d = 29 ∧ 8927 % d = 35) → d = 36 :=
by
  sorry

end greatest_divisor_l164_164246


namespace wings_count_total_l164_164434

def number_of_wings (num_planes : Nat) (wings_per_plane : Nat) : Nat :=
  num_planes * wings_per_plane

theorem wings_count_total :
  number_of_wings 45 2 = 90 :=
  by
    sorry

end wings_count_total_l164_164434


namespace difference_in_savings_correct_l164_164066

def S_last_year : ℝ := 45000
def saved_last_year_pct : ℝ := 0.083
def raise_pct : ℝ := 0.115
def saved_this_year_pct : ℝ := 0.056

noncomputable def saved_last_year_amount : ℝ := saved_last_year_pct * S_last_year
noncomputable def S_this_year : ℝ := S_last_year * (1 + raise_pct)
noncomputable def saved_this_year_amount : ℝ := saved_this_year_pct * S_this_year
noncomputable def difference_in_savings : ℝ := saved_last_year_amount - saved_this_year_amount

theorem difference_in_savings_correct :
  difference_in_savings = 925.20 := by
  sorry

end difference_in_savings_correct_l164_164066


namespace polynomial_divisibility_l164_164651

theorem polynomial_divisibility (a b c : ℝ) :
  (∀ x, (x-1)^3 ∣ x^4 + a * x^2 + b * x + c) ↔ (a = -6 ∧ b = 8 ∧ c = -3) :=
by
  sorry

end polynomial_divisibility_l164_164651


namespace freezer_temp_correct_l164_164234

variable (t_refrigeration : ℝ) (t_freezer : ℝ)

-- Given conditions
def refrigeration_temperature := t_refrigeration = 5
def freezer_temperature := t_freezer = -12

-- Goal: Prove that the freezer compartment's temperature is -12 degrees Celsius
theorem freezer_temp_correct : freezer_temperature t_freezer := by
  sorry

end freezer_temp_correct_l164_164234


namespace value_of_b_is_one_l164_164496

open Complex

theorem value_of_b_is_one (a b : ℝ) (h : (1 + I) / (1 - I) = a + b * I) : b = 1 := 
by
  sorry

end value_of_b_is_one_l164_164496


namespace find_f_7_l164_164077

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 4) = f x
axiom piecewise_function (x : ℝ) (h1 : 0 < x) (h2 : x < 2) : f x = 2 * x^3

theorem find_f_7 : f 7 = -2 := by
  sorry

end find_f_7_l164_164077


namespace election_valid_votes_l164_164660

variable (V : ℕ)
variable (invalid_pct : ℝ)
variable (exceed_pct : ℝ)
variable (total_votes : ℕ)
variable (invalid_votes : ℝ)
variable (valid_votes : ℕ)
variable (A_votes : ℕ)
variable (B_votes : ℕ)

theorem election_valid_votes :
  V = 9720 →
  invalid_pct = 0.20 →
  exceed_pct = 0.15 →
  total_votes = V →
  invalid_votes = invalid_pct * V →
  valid_votes = total_votes - invalid_votes →
  A_votes = B_votes + exceed_pct * total_votes →
  A_votes + B_votes = valid_votes →
  B_votes = 3159 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end election_valid_votes_l164_164660


namespace perp_lines_solution_l164_164958

theorem perp_lines_solution (a : ℝ) :
  ((a+2) * (a-1) + (1-a) * (2*a + 3) = 0) → (a = 1 ∨ a = -1) :=
by
  sorry

end perp_lines_solution_l164_164958


namespace Mike_profit_l164_164063

def total_cost (acres : ℕ) (cost_per_acre : ℕ) : ℕ :=
  acres * cost_per_acre

def revenue (acres_sold : ℕ) (price_per_acre : ℕ) : ℕ :=
  acres_sold * price_per_acre

def profit (revenue : ℕ) (cost : ℕ) : ℕ :=
  revenue - cost

theorem Mike_profit :
  let acres := 200
  let cost_per_acre := 70
  let acres_sold := acres / 2
  let price_per_acre := 200
  let cost := total_cost acres cost_per_acre
  let rev := revenue acres_sold price_per_acre
  profit rev cost = 6000 :=
by
  sorry

end Mike_profit_l164_164063


namespace minimal_area_circle_equation_circle_equation_center_on_line_l164_164989

-- Question (1): Prove the equation of the circle with minimal area
theorem minimal_area_circle_equation :
  (∃ (C : ℝ × ℝ) (r : ℝ), (r > 0) ∧ 
  C = (0, -4) ∧ r = Real.sqrt 5 ∧ 
  ∀ (P : ℝ × ℝ), (P = (2, -3) ∨ P = (-2, -5)) → P.1 ^ 2 + (P.2 + 4) ^ 2 = 5) :=
sorry

-- Question (2): Prove the equation of a circle with the center on a specific line
theorem circle_equation_center_on_line :
  (∃ (C : ℝ × ℝ) (r : ℝ), (r > 0) ∧ 
  (C.1 - 2 * C.2 - 3 = 0) ∧
  C = (-1, -2) ∧ r = Real.sqrt 10 ∧ 
  ∀ (P : ℝ × ℝ), (P = (2, -3) ∨ P = (-2, -5)) → (P.1 + 1) ^ 2 + (P.2 + 2) ^ 2 = 10) :=
sorry

end minimal_area_circle_equation_circle_equation_center_on_line_l164_164989


namespace linear_function_not_in_fourth_quadrant_l164_164459

theorem linear_function_not_in_fourth_quadrant (a b : ℝ) (h : a = 2 ∧ b = 1) :
  ∀ (x : ℝ), (2 * x + 1 < 0 → x > 0) := 
sorry

end linear_function_not_in_fourth_quadrant_l164_164459


namespace smallest_number_of_cubes_filling_box_l164_164335
open Nat

theorem smallest_number_of_cubes_filling_box (L W D : ℕ) (hL : L = 27) (hW : W = 15) (hD : D = 6) :
  let gcd := 3
  let cubes_along_length := L / gcd
  let cubes_along_width := W / gcd
  let cubes_along_depth := D / gcd
  cubes_along_length * cubes_along_width * cubes_along_depth = 90 :=
by
  sorry

end smallest_number_of_cubes_filling_box_l164_164335


namespace add_fractions_l164_164064

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l164_164064


namespace shape_of_r_eq_c_in_cylindrical_coords_l164_164603

variable {c : ℝ}

theorem shape_of_r_eq_c_in_cylindrical_coords (h : c > 0) :
  ∀ (r θ z : ℝ), (r = c) ↔ ∃ (cylinder : ℝ), cylinder = r ∧ cylinder = c :=
by
  sorry

end shape_of_r_eq_c_in_cylindrical_coords_l164_164603


namespace percentage_less_than_y_is_70_percent_less_than_z_l164_164047

variable {x y z : ℝ}

theorem percentage_less_than (h1 : x = 1.20 * y) (h2 : x = 0.36 * z) : y = 0.3 * z :=
by
  sorry

theorem y_is_70_percent_less_than_z (h : y = 0.3 * z) : (1 - y / z) * 100 = 70 :=
by
  sorry

end percentage_less_than_y_is_70_percent_less_than_z_l164_164047


namespace smallest_whole_number_greater_than_triangle_perimeter_l164_164005

theorem smallest_whole_number_greater_than_triangle_perimeter 
  (a b : ℝ) (h_a : a = 7) (h_b : b = 23) :
  ∀ c : ℝ, 16 < c ∧ c < 30 → ⌈a + b + c⌉ = 60 :=
by
  intros c h
  rw [h_a, h_b]
  sorry

end smallest_whole_number_greater_than_triangle_perimeter_l164_164005


namespace find_third_sum_l164_164290

def arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) : Prop :=
  (a 1) + (a 4) + (a 7) = 39 ∧ (a 2) + (a 5) + (a 8) = 33

theorem find_third_sum (a : ℕ → ℝ)
                       (d : ℝ)
                       (h_seq : arithmetic_sequence_sum a d)
                       (a_1 : ℝ) :
  a 1 = a_1 ∧ a 2 = a_1 + d ∧ a 3 = a_1 + 2 * d ∧
  a 4 = a_1 + 3 * d ∧ a 5 = a_1 + 4 * d ∧ a 6 = a_1 + 5 * d ∧
  a 7 = a_1 + 6 * d ∧ a 8 = a_1 + 7 * d ∧ a 9 = a_1 + 8 * d →
  a 3 + a 6 + a 9 = 27 :=
by
  sorry

end find_third_sum_l164_164290


namespace common_tangent_intersects_x_axis_at_point_A_l164_164729

-- Define the ellipses using their equations
def ellipse_C1 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def ellipse_C2 (x y : ℝ) : Prop := (x - 2)^2 + 4 * y^2 = 1

-- The theorem stating the coordinates of the point where the common tangent intersects the x-axis
theorem common_tangent_intersects_x_axis_at_point_A :
  (∃ x : ℝ, (ellipse_C1 x 0 ∧ ellipse_C2 x 0) ↔ x = 4) :=
sorry

end common_tangent_intersects_x_axis_at_point_A_l164_164729


namespace noncongruent_integer_tris_l164_164936

theorem noncongruent_integer_tris : 
  ∃ S : Finset (ℕ × ℕ × ℕ), S.card = 18 ∧ 
    ∀ (a b c : ℕ), (a, b, c) ∈ S → 
      (a + b > c ∧ a + b + c < 20 ∧ a < b ∧ b < c ∧ a^2 + b^2 ≠ c^2) :=
sorry

end noncongruent_integer_tris_l164_164936


namespace cordelia_bleach_time_l164_164030

theorem cordelia_bleach_time (B D : ℕ) (h1 : B + D = 9) (h2 : D = 2 * B) : B = 3 :=
by
  sorry

end cordelia_bleach_time_l164_164030


namespace find_solutions_l164_164955

theorem find_solutions :
  ∀ (x n : ℕ), 0 < x → 0 < n → x^(n+1) - (x + 1)^n = 2001 → (x, n) = (13, 2) :=
by
  intros x n hx hn heq
  sorry

end find_solutions_l164_164955


namespace sam_drove_distance_l164_164207

theorem sam_drove_distance (m_distance : ℕ) (m_time : ℕ) (s_time : ℕ) (s_distance : ℕ)
  (m_distance_eq : m_distance = 120) (m_time_eq : m_time = 3) (s_time_eq : s_time = 4) :
  s_distance = (m_distance / m_time) * s_time :=
by
  sorry

end sam_drove_distance_l164_164207


namespace dawns_earnings_per_hour_l164_164962

variable (hours_per_painting : ℕ) (num_paintings : ℕ) (total_earnings : ℕ)

def total_hours (hours_per_painting num_paintings : ℕ) : ℕ :=
  hours_per_painting * num_paintings

def earnings_per_hour (total_earnings total_hours : ℕ) : ℕ :=
  total_earnings / total_hours

theorem dawns_earnings_per_hour :
  hours_per_painting = 2 →
  num_paintings = 12 →
  total_earnings = 3600 →
  earnings_per_hour total_earnings (total_hours hours_per_painting num_paintings) = 150 :=
by
  intros h1 h2 h3
  sorry

end dawns_earnings_per_hour_l164_164962


namespace part1_part2_l164_164294

def f (x a : ℝ) := x^2 + 4 * a * x + 2 * a + 6

theorem part1 (a : ℝ) : (∃ x : ℝ, f x a = 0) ↔ (a = -1 ∨ a = 3 / 2) := 
by 
  sorry

def g (a : ℝ) := 2 - a * |a + 3|

theorem part2 (a : ℝ) :
  (-1 ≤ a ∧ a ≤ 3 / 2) →
  -19 / 4 ≤ g a ∧ g a ≤ 4 :=
by 
  sorry

end part1_part2_l164_164294


namespace books_loaned_out_l164_164580

theorem books_loaned_out (initial_books loaned_books returned_percentage end_books missing_books : ℕ)
  (h1 : initial_books = 75)
  (h2 : end_books = 66)
  (h3 : returned_percentage = 70)
  (h4 : initial_books - end_books = missing_books)
  (h5 : missing_books = (loaned_books * (100 - returned_percentage)) / 100):
  loaned_books = 30 :=
by
  sorry

end books_loaned_out_l164_164580


namespace positive_difference_of_squares_l164_164640

theorem positive_difference_of_squares {x y : ℕ} (hx : x > y) (hxy_sum : x + y = 70) (hxy_diff : x - y = 20) :
  x^2 - y^2 = 1400 :=
by
  sorry

end positive_difference_of_squares_l164_164640


namespace arithmetic_sequence_problem_l164_164292

noncomputable def a_n (n : ℕ) (a d : ℝ) : ℝ := a + (n - 1) * d

theorem arithmetic_sequence_problem (a d : ℝ) 
  (h : a_n 1 a d - a_n 4 a d - a_n 8 a d - a_n 12 a d + a_n 15 a d = 2) :
  a_n 3 a d + a_n 13 a d = -4 :=
by
  sorry

end arithmetic_sequence_problem_l164_164292


namespace decreasing_function_condition_l164_164302

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (4 * a - 1) * x + 4 * a else a ^ x

theorem decreasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y ≤ f a x) ↔ (1 / 7 ≤ a ∧ a < 1 / 4) :=
by
  sorry

end decreasing_function_condition_l164_164302


namespace iPhones_sold_l164_164535

theorem iPhones_sold (x : ℕ) (h1 : (1000 * x + 18000 + 16000) / (x + 100) = 670) : x = 100 :=
by
  sorry

end iPhones_sold_l164_164535


namespace sqrt_of_expression_l164_164238

theorem sqrt_of_expression : Real.sqrt (5^2 * 7^6) = 1715 := 
by
  sorry

end sqrt_of_expression_l164_164238


namespace solve_x2_plus_4y2_l164_164091

theorem solve_x2_plus_4y2 (x y : ℝ) (h₁ : x + 2 * y = 6) (h₂ : x * y = -6) : x^2 + 4 * y^2 = 60 :=
by
  sorry

end solve_x2_plus_4y2_l164_164091


namespace correct_statement_A_l164_164564

-- Declare Avogadro's constant
def Avogadro_constant : ℝ := 6.022e23

-- Given conditions
def gas_mass_ethene : ℝ := 5.6 -- grams of ethylene
def gas_mass_cyclopropane : ℝ := 5.6 -- grams of cyclopropane
def gas_combined_carbon_atoms : ℝ := 0.4 * Avogadro_constant

-- Assertion to prove
theorem correct_statement_A :
    gas_combined_carbon_atoms = 0.4 * Avogadro_constant :=
by
  sorry

end correct_statement_A_l164_164564


namespace minimum_distance_l164_164422

-- Define conditions and problem

def lies_on_line (P : ℝ × ℝ) : Prop :=
  P.1 + P.2 - 4 = 0

theorem minimum_distance (P : ℝ × ℝ) (h : lies_on_line P) : P.1^2 + P.2^2 ≥ 8 :=
sorry

end minimum_distance_l164_164422


namespace probability_of_70th_percentile_is_25_over_56_l164_164685

-- Define the weights of the students
def weights : List ℕ := [90, 100, 110, 120, 140, 150, 150, 160]

-- Define the number of students to select
def n_selected_students : ℕ := 3

-- Define the percentile value
def percentile_value : ℕ := 70

-- Define the corresponding weight for the 70th percentile
def percentile_weight : ℕ := 150

-- Define the combination function
noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability calculation
noncomputable def probability_70th_percentile : ℚ :=
  let total_ways := C 8 3
  let favorable_ways := (C 2 2) * (C 5 1) + (C 2 1) * (C 5 2)
  favorable_ways / total_ways

-- Define the theorem to prove the probability
theorem probability_of_70th_percentile_is_25_over_56 :
  probability_70th_percentile = 25 / 56 := by
  sorry

end probability_of_70th_percentile_is_25_over_56_l164_164685


namespace trigonometric_identity_l164_164600

open Real

theorem trigonometric_identity
  (α : ℝ)
  (h₁ : tan α + 1 / tan α = 10 / 3)
  (h₂ : π / 4 < α ∧ α < π / 2) :
  sin (2 * α + π / 4) + 2 * cos (π / 4) * sin α ^ 2 = 4 * sqrt 2 / 5 :=
by
  sorry

end trigonometric_identity_l164_164600


namespace problem_lean_statement_l164_164331

theorem problem_lean_statement (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 2) = 2 * x ^ 2 + 5 * x + 3)
  (h2 : ∀ x, f x = a * x ^ 2 + b * x + c) : a + b + c = 0 :=
by sorry

end problem_lean_statement_l164_164331


namespace product_of_first_five_terms_l164_164414

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ m n p q : ℕ, m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0 ∧ m + n = p + q → a m * a n = a p * a q

theorem product_of_first_five_terms 
  (h : geometric_sequence a) 
  (h3 : a 3 = 2) : 
  a 1 * a 2 * a 3 * a 4 * a 5 = 32 :=
sorry

end product_of_first_five_terms_l164_164414


namespace olive_needs_two_colours_l164_164784

theorem olive_needs_two_colours (α : Type) [Finite α] (G : SimpleGraph α) (colour : α → Fin 2) :
  (∀ v : α, ∃! w : α, G.Adj v w ∧ colour v = colour w) → ∃ color_map : α → Fin 2, ∀ v, ∃! w, G.Adj v w ∧ color_map v = color_map w :=
sorry

end olive_needs_two_colours_l164_164784


namespace proof_of_problem_statement_l164_164150

noncomputable def problem_statement : Prop :=
  ∀ (k : ℝ) (m : ℝ),
    (0 < m ∧ m < 3/2) → 
    (-3/(4 * m) = k) → 
    (k < -1/2)

theorem proof_of_problem_statement : problem_statement :=
  sorry

end proof_of_problem_statement_l164_164150


namespace compute_x_y_power_sum_l164_164206

noncomputable def pi : ℝ := Real.pi

theorem compute_x_y_power_sum
  (x y : ℝ)
  (h1 : 1 < x)
  (h2 : 1 < y)
  (h3 : (Real.log x / Real.log 2)^5 + (Real.log y / Real.log 3)^5 + 32 = 16 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^pi + y^pi = 2^(pi * (16:ℝ)^(1/5)) + 3^(pi * (16:ℝ)^(1/5)) :=
by
  sorry

end compute_x_y_power_sum_l164_164206


namespace f_decreasing_in_interval_l164_164723

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)

noncomputable def shifted_g (x : ℝ) : ℝ := g (x + Real.pi / 6)

noncomputable def f (x : ℝ) : ℝ := shifted_g (2 * x)

theorem f_decreasing_in_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 4 → f y < f x :=
by
  sorry

end f_decreasing_in_interval_l164_164723


namespace a10_eq_neg12_l164_164131

variable (a_n : ℕ → ℤ)
variable (S_n : ℕ → ℤ)
variable (d a1 : ℤ)

-- Conditions of the problem
axiom arithmetic_sequence : ∀ n : ℕ, a_n n = a1 + (n - 1) * d
axiom sum_of_first_n_terms : ∀ n : ℕ, S_n n = n * (2 * a1 + (n - 1) * d) / 2
axiom a2_eq_4 : a_n 2 = 4
axiom S8_eq_neg8 : S_n 8 = -8

-- The statement to prove
theorem a10_eq_neg12 : a_n 10 = -12 :=
sorry

end a10_eq_neg12_l164_164131


namespace stu_books_count_l164_164223

noncomputable def elmo_books : ℕ := 24
noncomputable def laura_books : ℕ := elmo_books / 3
noncomputable def stu_books : ℕ := laura_books / 2

theorem stu_books_count :
  stu_books = 4 :=
by
  sorry

end stu_books_count_l164_164223


namespace fred_now_has_l164_164961

-- Definitions based on conditions
def original_cards : ℕ := 40
def purchased_cards : ℕ := 22

-- Theorem to prove the number of cards Fred has now
theorem fred_now_has (original_cards : ℕ) (purchased_cards : ℕ) : original_cards - purchased_cards = 18 :=
by
  sorry

end fred_now_has_l164_164961


namespace not_equal_factorial_l164_164432

noncomputable def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem not_equal_factorial (n : ℕ) :
  permutations (n + 1) n ≠ (by apply Nat.factorial n) := by
  sorry

end not_equal_factorial_l164_164432


namespace tan_product_pi_over_6_3_2_undefined_l164_164191

noncomputable def tan_pi_over_6 : ℝ := Real.tan (Real.pi / 6)
noncomputable def tan_pi_over_3 : ℝ := Real.tan (Real.pi / 3)
noncomputable def tan_pi_over_2 : ℝ := Real.tan (Real.pi / 2)

theorem tan_product_pi_over_6_3_2_undefined :
  ∃ (x y : ℝ), Real.tan (Real.pi / 6) = x ∧ Real.tan (Real.pi / 3) = y ∧ Real.tan (Real.pi / 2) = 0 :=
by
  sorry

end tan_product_pi_over_6_3_2_undefined_l164_164191


namespace calculation_correct_l164_164642

theorem calculation_correct :
  (-1 : ℝ)^51 + (2 : ℝ)^(4^2 + 5^2 - 7^2) = -(127 / 128) := 
by
  sorry

end calculation_correct_l164_164642


namespace trees_died_in_typhoon_l164_164878

theorem trees_died_in_typhoon :
  ∀ (original_trees left_trees died_trees : ℕ), 
  original_trees = 20 → 
  left_trees = 4 → 
  died_trees = original_trees - left_trees → 
  died_trees = 16 :=
by
  intros original_trees left_trees died_trees h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end trees_died_in_typhoon_l164_164878


namespace negation_of_exists_l164_164137

theorem negation_of_exists (x : ℝ) : 
  ¬ (∃ x : ℝ, 2 * x^2 + 2 * x - 1 ≤ 0) ↔ ∀ x : ℝ, 2 * x^2 + 2 * x - 1 > 0 :=
by
  sorry

end negation_of_exists_l164_164137


namespace plywood_width_is_5_l164_164949

theorem plywood_width_is_5 (length width perimeter : ℕ) (h1 : length = 6) (h2 : perimeter = 2 * (length + width)) (h3 : perimeter = 22) : width = 5 :=
by {
  -- proof steps would go here, but are omitted per instructions
  sorry
}

end plywood_width_is_5_l164_164949


namespace intercept_sum_l164_164737

theorem intercept_sum (x y : ℤ) (h1 : 0 ≤ x) (h2 : x < 42) (h3 : 0 ≤ y) (h4 : y < 42)
  (h : 5 * x ≡ 3 * y - 2 [ZMOD 42]) : (x + y) = 36 :=
by
  sorry

end intercept_sum_l164_164737


namespace problem_l164_164516

noncomputable def f (x : ℝ) : ℝ := 5 * x - 7
noncomputable def g (x : ℝ) : ℝ := x / 5 + 3

theorem problem : ∀ x : ℝ, f (g x) - g (f x) = 6.4 :=
by
  intro x
  sorry

end problem_l164_164516


namespace length_of_fountain_built_by_20_men_in_6_days_l164_164838

noncomputable def work (workers : ℕ) (days : ℕ) : ℕ :=
  workers * days

theorem length_of_fountain_built_by_20_men_in_6_days :
  (work 35 3) / (work 20 6) * 49 = 56 :=
by
  sorry

end length_of_fountain_built_by_20_men_in_6_days_l164_164838


namespace min_people_in_group_l164_164452

theorem min_people_in_group (B G : ℕ) (h : B / (B + G : ℝ) > 0.94) : B + G ≥ 17 :=
sorry

end min_people_in_group_l164_164452


namespace solution_set_of_inequality_l164_164762

open Set

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - x - 6 < 0} = Ioo (-2 : ℝ) 3 := 
sorry

end solution_set_of_inequality_l164_164762


namespace total_fruit_punch_eq_21_l164_164118

def orange_punch : ℝ := 4.5
def cherry_punch := 2 * orange_punch
def apple_juice := cherry_punch - 1.5

theorem total_fruit_punch_eq_21 : orange_punch + cherry_punch + apple_juice = 21 := by 
  -- This is where the proof would go
  sorry

end total_fruit_punch_eq_21_l164_164118


namespace daniel_total_earnings_l164_164177

-- Definitions of conditions
def fabric_delivered_monday : ℕ := 20
def fabric_delivered_tuesday : ℕ := 2 * fabric_delivered_monday
def fabric_delivered_wednesday : ℕ := fabric_delivered_tuesday / 4
def total_fabric_delivered : ℕ := fabric_delivered_monday + fabric_delivered_tuesday + fabric_delivered_wednesday

def cost_per_yard : ℕ := 2
def total_earnings : ℕ := total_fabric_delivered * cost_per_yard

-- Proposition to be proved
theorem daniel_total_earnings : total_earnings = 140 := by
  sorry

end daniel_total_earnings_l164_164177


namespace smallest_three_digit_multiple_of_17_l164_164626

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 17 ∣ m → n ≤ m := by
  use 102
  sorry

end smallest_three_digit_multiple_of_17_l164_164626


namespace set_A_correct_l164_164006

-- Definition of the sets and conditions
def A : Set ℤ := {-3, 0, 2, 6}
def B : Set ℤ := {-1, 3, 5, 8}

theorem set_A_correct : 
  (∃ a1 a2 a3 a4 : ℤ, A = {a1, a2, a3, a4} ∧ 
  {a1 + a2 + a3, a1 + a2 + a4, a1 + a3 + a4, a2 + a3 + a4} = B) → 
  A = {-3, 0, 2, 6} :=
by 
  sorry

end set_A_correct_l164_164006


namespace inverse_proportion_relation_l164_164761

theorem inverse_proportion_relation :
  let y1 := 3 / (-2)
  let y2 := 3 / (-1)
  let y3 := 3 / 1
  y2 < y1 ∧ y1 < y3 :=
by
  -- Variable definitions according to conditions
  let y1 := 3 / (-2)
  let y2 := 3 / (-1)
  let y3 := 3 / 1
  -- Proof steps go here (not required for the statement)
  -- Since proof steps are omitted, we use sorry to indicate it
  sorry

end inverse_proportion_relation_l164_164761


namespace spring_outing_students_l164_164268

variable (x y : ℕ)

theorem spring_outing_students (hx : x % 10 = 0) (hy : y % 10 = 0) (h1 : x + y = 1008) (h2 : y - x = 133) :
  x = 437 ∧ y = 570 :=
by
  sorry

end spring_outing_students_l164_164268


namespace roots_of_quadratic_eq_l164_164997

theorem roots_of_quadratic_eq : 
    ∃ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0) ∧ (x2^2 - 2 * x2 - 8 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2) / (x1 * x2) = -1 / 4 := 
sorry

end roots_of_quadratic_eq_l164_164997


namespace hyperbola_line_intersection_l164_164591

theorem hyperbola_line_intersection
  (A B m : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) (hm : m ≠ 0) :
  ∃ x y : ℝ, A^2 * x^2 - B^2 * y^2 = 1 ∧ Ax - By = m ∧ Bx + Ay ≠ 0 :=
by
  sorry

end hyperbola_line_intersection_l164_164591


namespace max_OM_ON_value_l164_164756

noncomputable def maximum_OM_ON (a b : ℝ) : ℝ :=
  (1 + Real.sqrt 2) / 2 * (a + b)

-- Given the conditions in triangle ABC with sides BC and AC having fixed lengths a and b respectively,
-- and that AB can vary such that a square is constructed outward on side AB with center O,
-- and M and N are the midpoints of sides BC and AC respectively, prove the maximum value of OM + ON.
theorem max_OM_ON_value (a b : ℝ) : 
  ∃ OM ON : ℝ, OM + ON = maximum_OM_ON a b :=
sorry

end max_OM_ON_value_l164_164756


namespace rectangle_area_relation_l164_164800

theorem rectangle_area_relation (x y : ℝ) (h : x * y = 4) (hx : x > 0) : y = 4 / x := 
sorry

end rectangle_area_relation_l164_164800


namespace area_of_efgh_l164_164399

def small_rectangle_shorter_side : ℝ := 7
def small_rectangle_longer_side : ℝ := 3 * small_rectangle_shorter_side
def larger_rectangle_width : ℝ := small_rectangle_longer_side
def larger_rectangle_length : ℝ := small_rectangle_longer_side + small_rectangle_shorter_side

theorem area_of_efgh :
  larger_rectangle_length * larger_rectangle_width = 588 := by
  sorry

end area_of_efgh_l164_164399


namespace calc_expr_l164_164768

theorem calc_expr :
  (2 * Real.sqrt 2 - 1) ^ 2 + (1 + Real.sqrt 3) * (1 - Real.sqrt 3) = 7 - 4 * Real.sqrt 2 :=
by
  sorry

end calc_expr_l164_164768


namespace remainder_of_95_times_97_div_12_l164_164576

theorem remainder_of_95_times_97_div_12 : 
  (95 * 97) % 12 = 11 := by
  sorry

end remainder_of_95_times_97_div_12_l164_164576


namespace powers_of_2_form_6n_plus_8_l164_164832

noncomputable def is_power_of_two (x : ℕ) : Prop := ∃ k : ℕ, x = 2 ^ k

def of_the_form (n : ℕ) : ℕ := 6 * n + 8

def is_odd_greater_than_one (k : ℕ) : Prop := k % 2 = 1 ∧ k > 1

theorem powers_of_2_form_6n_plus_8 (k : ℕ) (n : ℕ) :
  (2 ^ k = of_the_form n) ↔ is_odd_greater_than_one k :=
sorry

end powers_of_2_form_6n_plus_8_l164_164832


namespace find_rate_percent_l164_164225

-- Define the conditions
def principal : ℝ := 1200
def time : ℝ := 4
def simple_interest : ℝ := 400

-- Define the rate that we need to prove
def rate : ℝ := 8.3333  -- approximately

-- Formalize the proof problem in Lean 4
theorem find_rate_percent
  (P : ℝ) (T : ℝ) (SI : ℝ) (R : ℝ)
  (hP : P = principal) (hT : T = time) (hSI : SI = simple_interest) :
  SI = (P * R * T) / 100 → R = rate :=
by
  intros h
  sorry

end find_rate_percent_l164_164225


namespace final_student_count_is_correct_l164_164455

-- Define the initial conditions
def initial_students : ℕ := 11
def students_left_first_semester : ℕ := 6
def students_joined_first_semester : ℕ := 25
def additional_students_second_semester : ℕ := 15
def students_transferred_second_semester : ℕ := 3
def students_switched_class_second_semester : ℕ := 2

-- Define the final number of students to be proven
def final_number_of_students : ℕ := 
  initial_students - students_left_first_semester + students_joined_first_semester + 
  additional_students_second_semester - students_transferred_second_semester - students_switched_class_second_semester

-- The theorem we need to prove
theorem final_student_count_is_correct : final_number_of_students = 40 := by
  sorry

end final_student_count_is_correct_l164_164455


namespace possible_values_of_x_l164_164450

theorem possible_values_of_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 10) (h2 : y + 1 / x = 5 / 12) : x = 4 ∨ x = 6 :=
by
  sorry

end possible_values_of_x_l164_164450


namespace new_mixture_concentration_l164_164411

def vessel1_capacity : ℝ := 2
def vessel1_concentration : ℝ := 0.30
def vessel2_capacity : ℝ := 6
def vessel2_concentration : ℝ := 0.40
def total_volume : ℝ := 8
def expected_concentration : ℝ := 37.5

theorem new_mixture_concentration :
  ((vessel1_capacity * vessel1_concentration + vessel2_capacity * vessel2_concentration) / total_volume) * 100 = expected_concentration :=
by
  sorry

end new_mixture_concentration_l164_164411


namespace almonds_received_by_amanda_l164_164843

variable (totalAlmonds : ℚ)
variable (numberOfPiles : ℚ)
variable (pilesForAmanda : ℚ)

-- Conditions
def stephanie_has_almonds := totalAlmonds = 66 / 7
def distribute_equally_into_piles := numberOfPiles = 6
def amanda_receives_piles := pilesForAmanda = 3

-- Conclusion to prove
theorem almonds_received_by_amanda :
  stephanie_has_almonds totalAlmonds →
  distribute_equally_into_piles numberOfPiles →
  amanda_receives_piles pilesForAmanda →
  (totalAlmonds / numberOfPiles) * pilesForAmanda = 33 / 7 :=
by
  sorry

end almonds_received_by_amanda_l164_164843


namespace exists_root_in_interval_l164_164468

theorem exists_root_in_interval
    (a b c x₁ x₂ : ℝ)
    (h₁ : a * x₁^2 + b * x₁ + c = 0)
    (h₂ : -a * x₂^2 + b * x₂ + c = 0) :
    ∃ x₃ : ℝ, (a / 2) * x₃^2 + b * x₃ + c = 0 ∧ ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) :=
sorry

end exists_root_in_interval_l164_164468


namespace find_some_number_l164_164279

-- Conditions on operations
axiom plus_means_mult (a b : ℕ) : (a + b) = (a * b)
axiom minus_means_plus (a b : ℕ) : (a - b) = (a + b)
axiom mult_means_div (a b : ℕ) : (a * b) = (a / b)
axiom div_means_minus (a b : ℕ) : (a / b) = (a - b)

-- Problem statement
theorem find_some_number (some_number : ℕ) :
  (6 - 9 + some_number * 3 / 25 = 5 ↔
   6 + 9 * some_number / 3 - 25 = 5) ∧
  some_number = 8 := by
  sorry

end find_some_number_l164_164279


namespace dolphins_score_l164_164980

theorem dolphins_score (S D : ℕ) (h1 : S + D = 48) (h2 : S = D + 20) : D = 14 := by
    sorry

end dolphins_score_l164_164980


namespace scientific_notation_of_218000000_l164_164020

theorem scientific_notation_of_218000000 :
  218000000 = 2.18 * 10^8 :=
sorry

end scientific_notation_of_218000000_l164_164020


namespace absolute_value_of_neg_eight_l164_164437

/-- Absolute value of a number is the distance from 0 on the number line. -/
def absolute_value (x : ℤ) : ℤ :=
  if x >= 0 then x else -x

theorem absolute_value_of_neg_eight : absolute_value (-8) = 8 := by
  -- Proof is omitted
  sorry

end absolute_value_of_neg_eight_l164_164437


namespace calc_f_g_3_minus_g_f_3_l164_164812

def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := x^2 + 2

theorem calc_f_g_3_minus_g_f_3 :
  (f (g 3) - g (f 3)) = -96 :=
by
  sorry

end calc_f_g_3_minus_g_f_3_l164_164812


namespace hamburgers_sold_in_winter_l164_164474

theorem hamburgers_sold_in_winter:
  ∀ (T x : ℕ), 
  (T = 5 * 4) → 
  (5 + 6 + 4 + x = T) →
  (x = 5) :=
by
  intros T x hT hTotal
  sorry

end hamburgers_sold_in_winter_l164_164474


namespace number_of_crocodiles_l164_164782

theorem number_of_crocodiles
  (f : ℕ) -- number of frogs
  (c : ℕ) -- number of crocodiles
  (total_eyes : ℕ) -- total number of eyes
  (frog_eyes : ℕ) -- number of eyes per frog
  (croc_eyes : ℕ) -- number of eyes per crocodile
  (h_f : f = 20) -- condition: there are 20 frogs
  (h_total_eyes : total_eyes = 52) -- condition: total number of eyes is 52
  (h_frog_eyes : frog_eyes = 2) -- condition: each frog has 2 eyes
  (h_croc_eyes : croc_eyes = 2) -- condition: each crocodile has 2 eyes
  :
  c = 6 := -- proof goal: number of crocodiles is 6
by
  sorry

end number_of_crocodiles_l164_164782


namespace no_triangle_tangent_l164_164712

open Real

/-- Given conditions --/
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C2 (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0 ∧ (1 / a^2) + (1 / b^2) = 1

theorem no_triangle_tangent (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : (1 : ℝ) / a^2 + 1 / b^2 = 1) :
  ¬∃ (A B C : ℝ × ℝ), 
    (C1 A.1 A.2) ∧ (C1 B.1 B.2) ∧ (C1 C.1 C.2) ∧
    (∃ (l : ℝ) (m : ℝ) (n : ℝ), C2 l m a b ∧ C2 n l a b) :=
by
  sorry

end no_triangle_tangent_l164_164712


namespace drums_needed_for_profit_l164_164827

def cost_to_enter_contest : ℝ := 10
def money_per_drum : ℝ := 0.025
def money_needed_for_profit (drums_hit : ℝ) : Prop :=
  drums_hit * money_per_drum > cost_to_enter_contest

theorem drums_needed_for_profit : ∃ D : ℝ, money_needed_for_profit D ∧ D = 400 :=
  by
  use 400
  sorry

end drums_needed_for_profit_l164_164827


namespace ratio_division_l164_164624

theorem ratio_division
  (A B C : ℕ)
  (h : (A : ℚ) / B = 3 / 2 ∧ (B : ℚ) / C = 1 / 3) :
  (5 * A + 3 * B) / (5 * C - 2 * A) = 7 / 8 :=
by
  sorry

end ratio_division_l164_164624


namespace range_of_m_value_of_m_l164_164894

-- Define the quadratic equation and the condition for having real roots
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*x - 2*m + 5

-- Condition for the quadratic equation to have real roots
def discriminant_nonnegative (m : ℝ) : Prop := (4^2 - 4*1*(-2*m + 5)) ≥ 0

-- Define Vieta's formulas for the roots of the quadratic equation
def vieta_sum_roots (x1 x2 : ℝ) : Prop := x1 + x2 = 4
def vieta_product_roots (x1 x2 : ℝ) (m : ℝ) : Prop := x1 * x2 = -2*m + 5

-- Given condition with the roots
def condition_on_roots (x1 x2 m : ℝ) : Prop := x1 * x2 + x1 + x2 = m^2 + 6

-- Prove the range of m
theorem range_of_m (m : ℝ) : 
  discriminant_nonnegative m → m ≥ 1/2 := by 
  sorry

-- Prove the value of m based on the given root condition
theorem value_of_m (x1 x2 m : ℝ) : 
  vieta_sum_roots x1 x2 → 
  vieta_product_roots x1 x2 m → 
  condition_on_roots x1 x2 m → 
  m = 1 := by 
  sorry

end range_of_m_value_of_m_l164_164894


namespace sin_neg_thirtyone_sixths_pi_l164_164668

theorem sin_neg_thirtyone_sixths_pi : Real.sin (-31 / 6 * Real.pi) = 1 / 2 :=
by 
  sorry

end sin_neg_thirtyone_sixths_pi_l164_164668


namespace tail_length_10_l164_164004

theorem tail_length_10 (length_body tail_length head_length width height overall_length: ℝ) 
  (h1 : tail_length = (1 / 2) * length_body)
  (h2 : head_length = (1 / 6) * length_body)
  (h3 : height = 1.5 * width)
  (h4 : overall_length = length_body + tail_length)
  (h5 : overall_length = 30)
  (h6 : width = 12) :
  tail_length = 10 :=
by
  sorry

end tail_length_10_l164_164004


namespace greatest_area_difference_l164_164503

def first_rectangle_perimeter (l w : ℕ) : Prop :=
  2 * l + 2 * w = 156

def second_rectangle_perimeter (l w : ℕ) : Prop :=
  2 * l + 2 * w = 144

theorem greatest_area_difference : 
  ∃ (l1 w1 l2 w2 : ℕ), 
  first_rectangle_perimeter l1 w1 ∧ 
  second_rectangle_perimeter l2 w2 ∧ 
  (l1 * (78 - l1) - l2 * (72 - l2) = 225) := 
sorry

end greatest_area_difference_l164_164503


namespace first_stack_height_is_seven_l164_164700

-- Definitions of the conditions
def first_stack (h : ℕ) := h
def second_stack (h : ℕ) := h + 5
def third_stack (h : ℕ) := h + 12

-- Conditions on the blocks falling down
def blocks_fell_first_stack (h : ℕ) := h
def blocks_fell_second_stack (h : ℕ) := (h + 5) - 2
def blocks_fell_third_stack (h : ℕ) := (h + 12) - 3

-- Total blocks fell down
def total_blocks_fell (h : ℕ) := blocks_fell_first_stack h + blocks_fell_second_stack h + blocks_fell_third_stack h

-- Lean statement to prove the height of the first stack
theorem first_stack_height_is_seven (h : ℕ) (h_eq : total_blocks_fell h = 33) : h = 7 :=
by sorry

-- Testing the conditions hold for the solution h = 7
#eval total_blocks_fell 7 -- Expected: 33

end first_stack_height_is_seven_l164_164700


namespace value_of_x_squared_plus_y_squared_l164_164377

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x + y = -4) (h2 : x = 6 / y) : 
  x^2 + y^2 = 4 :=
  sorry

end value_of_x_squared_plus_y_squared_l164_164377


namespace parabola_point_value_l164_164482

variable {x₀ y₀ : ℝ}

theorem parabola_point_value
  (h₁ : y₀^2 = 4 * x₀)
  (h₂ : (Real.sqrt ((x₀ - 1)^2 + y₀^2) = 5/4 * x₀)) :
  x₀ = 4 := by
  sorry

end parabola_point_value_l164_164482


namespace average_age_l164_164106

variable (John Mary Tonya : ℕ)

theorem average_age (h1 : John = 2 * Mary) (h2 : John = Tonya / 2) (h3 : Tonya = 60) : 
  (John + Mary + Tonya) / 3 = 35 :=
by
  sorry

end average_age_l164_164106


namespace man_and_son_together_days_l164_164236

noncomputable def man_days : ℝ := 7
noncomputable def son_days : ℝ := 5.25
noncomputable def combined_days : ℝ := man_days * son_days / (man_days + son_days)

theorem man_and_son_together_days :
  combined_days = 7 / 5 :=
by
  sorry

end man_and_son_together_days_l164_164236


namespace sin_120_eq_sqrt3_div_2_l164_164430

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
sorry

end sin_120_eq_sqrt3_div_2_l164_164430


namespace find_a_value_l164_164242

noncomputable def find_a (a : ℝ) : Prop :=
  (a > 0) ∧ (1 / 3 = 2 / a)

theorem find_a_value (a : ℝ) (h : find_a a) : a = 6 :=
sorry

end find_a_value_l164_164242


namespace houses_without_garage_nor_pool_l164_164645

def total_houses : ℕ := 85
def houses_with_garage : ℕ := 50
def houses_with_pool : ℕ := 40
def houses_with_both : ℕ := 35
def neither_garage_nor_pool : ℕ := 30

theorem houses_without_garage_nor_pool :
  total_houses - (houses_with_garage + houses_with_pool - houses_with_both) = neither_garage_nor_pool :=
by
  sorry

end houses_without_garage_nor_pool_l164_164645


namespace concentric_circles_area_difference_l164_164159

/-- Two concentric circles with radii 12 cm and 7 cm have an area difference of 95π cm² between them. -/
theorem concentric_circles_area_difference :
  let r1 := 12
  let r2 := 7
  let area_larger := Real.pi * r1^2
  let area_smaller := Real.pi * r2^2
  let area_difference := area_larger - area_smaller
  area_difference = 95 * Real.pi := by
sorry

end concentric_circles_area_difference_l164_164159


namespace quarterly_insurance_payment_l164_164573

theorem quarterly_insurance_payment (annual_payment : ℕ) (quarters_in_year : ℕ) (quarterly_payment : ℕ) : 
  annual_payment = 1512 → quarters_in_year = 4 → quarterly_payment * quarters_in_year = annual_payment → quarterly_payment = 378 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  sorry

end quarterly_insurance_payment_l164_164573


namespace total_peanut_cost_l164_164280

def peanut_cost_per_pound : ℝ := 3
def minimum_pounds : ℝ := 15
def extra_pounds : ℝ := 20

theorem total_peanut_cost :
  (minimum_pounds + extra_pounds) * peanut_cost_per_pound = 105 :=
by
  sorry

end total_peanut_cost_l164_164280


namespace find_h_l164_164583

theorem find_h (j k h : ℕ) (h₁ : 2013 = 3 * h^2 + j) (h₂ : 2014 = 2 * h^2 + k)
  (pos_int_x_intercepts_1 : ∃ x1 x2 : ℕ, x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ (3 * (x1 - h)^2 + j = 0 ∧ 3 * (x2 - h)^2 + j = 0))
  (pos_int_x_intercepts_2 : ∃ y1 y2 : ℕ, y1 ≠ y2 ∧ y1 > 0 ∧ y2 > 0 ∧ (2 * (y1 - h)^2 + k = 0 ∧ 2 * (y2 - h)^2 + k = 0)):
  h = 36 :=
by
  sorry

end find_h_l164_164583


namespace second_bus_percentage_full_l164_164575

noncomputable def bus_capacity : ℕ := 150
noncomputable def employees_in_buses : ℕ := 195
noncomputable def first_bus_percentage : ℚ := 0.60

theorem second_bus_percentage_full :
  let employees_first_bus := first_bus_percentage * bus_capacity
  let employees_second_bus := (employees_in_buses : ℚ) - employees_first_bus
  let second_bus_percentage := (employees_second_bus / bus_capacity) * 100
  second_bus_percentage = 70 :=
by
  sorry

end second_bus_percentage_full_l164_164575


namespace intersection_when_m_eq_2_range_of_m_l164_164954

open Set

variables (m x : ℝ)

def A (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}
def B : Set ℝ := {x | -4 ≤ x ∧ x ≤ 2}
def intersection (m : ℝ) : Set ℝ := A m ∩ B

-- First proof: When m = 2, the intersection of A and B is [1,2].
theorem intersection_when_m_eq_2 : intersection 2 = {x | 1 ≤ x ∧ x ≤ 2} :=
sorry

-- Second proof: The range of m such that A ⊆ A ∩ B
theorem range_of_m : {m | A m ⊆ B} = {m | -2 ≤ m ∧ m ≤ 1 / 2} :=
sorry

end intersection_when_m_eq_2_range_of_m_l164_164954


namespace batsman_average_increase_l164_164774

theorem batsman_average_increase
  (prev_avg : ℝ) -- average before the 17th innings
  (total_runs_16 : ℝ := 16 * prev_avg) -- total runs scored in the first 16 innings
  (score_17th : ℝ := 85) -- score in the 17th innings
  (new_avg : ℝ := 37) -- new average after 17 innings
  (total_runs_17 : ℝ := total_runs_16 + score_17th) -- total runs after 17 innings
  (calc_total_runs_17 : ℝ := 17 * new_avg) -- new total runs calculated by the new average
  (h : total_runs_17 = calc_total_runs_17) -- given condition: total_runs_17 = calc_total_runs_17
  : (new_avg - prev_avg) = 3 := 
by
  sorry

end batsman_average_increase_l164_164774


namespace rewrite_equation_to_function_l164_164919

theorem rewrite_equation_to_function (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
by
  sorry

end rewrite_equation_to_function_l164_164919


namespace oil_level_drop_l164_164022

noncomputable def stationary_tank_radius : ℝ := 100
noncomputable def stationary_tank_height : ℝ := 25
noncomputable def truck_tank_radius : ℝ := 7
noncomputable def truck_tank_height : ℝ := 10

noncomputable def π : ℝ := Real.pi
noncomputable def truck_tank_volume := π * truck_tank_radius^2 * truck_tank_height
noncomputable def stationary_tank_area := π * stationary_tank_radius^2

theorem oil_level_drop (volume_truck: ℝ) (area_stationary: ℝ) : volume_truck = 490 * π → area_stationary = π * 10000 → (volume_truck / area_stationary) = 0.049 :=
by
  intros h1 h2
  sorry

end oil_level_drop_l164_164022


namespace revenue_increase_20_percent_l164_164981

variable (P Q : ℝ)

def original_revenue (P Q : ℝ) : ℝ := P * Q
def new_price (P : ℝ) : ℝ := P * 1.5
def new_quantity (Q : ℝ) : ℝ := Q * 0.8
def new_revenue (P Q : ℝ) : ℝ := (new_price P) * (new_quantity Q)

theorem revenue_increase_20_percent (P Q : ℝ) : 
  (new_revenue P Q) = 1.2 * (original_revenue P Q) := by
  sorry

end revenue_increase_20_percent_l164_164981


namespace vector_addition_l164_164101

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-3, 4)

theorem vector_addition :
  2 • a + b = (1, 2) :=
by
  sorry

end vector_addition_l164_164101


namespace M_gt_N_l164_164819

-- Define M and N
def M (x y : ℝ) : ℝ := x^2 + y^2 + 1
def N (x y : ℝ) : ℝ := 2 * (x + y - 1)

-- State the theorem to prove M > N given the conditions
theorem M_gt_N (x y : ℝ) : M x y > N x y := by
  sorry

end M_gt_N_l164_164819


namespace restaurant_dinners_sold_on_Monday_l164_164638

theorem restaurant_dinners_sold_on_Monday (M : ℕ) 
  (h1 : ∀ tues_dinners, tues_dinners = M + 40) 
  (h2 : ∀ wed_dinners, wed_dinners = (M + 40) / 2)
  (h3 : ∀ thurs_dinners, thurs_dinners = ((M + 40) / 2) + 3)
  (h4 : M + (M + 40) + ((M + 40) / 2) + (((M + 40) / 2) + 3) = 203) : 
  M = 40 := 
sorry

end restaurant_dinners_sold_on_Monday_l164_164638


namespace domain_of_function_l164_164912

noncomputable def function_domain := {x : ℝ | 1 + 1 / x > 0 ∧ 1 - x^2 ≥ 0}

theorem domain_of_function : function_domain = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end domain_of_function_l164_164912


namespace seq_general_term_l164_164783

noncomputable def seq (n : ℕ) : ℚ :=
  if n = 0 then 1/2
  else if n = 1 then 1/2
  else seq (n - 1) * 3 / (seq (n - 1) + 3)

theorem seq_general_term : ∀ n : ℕ, seq (n + 1) = 3 / (n + 6) :=
by
  intro n
  induction n with
  | zero => sorry
  | succ k ih => sorry

end seq_general_term_l164_164783


namespace ophelia_age_l164_164404

/-- 
If Lennon is currently 8 years old, 
and in two years Ophelia will be four times as old as Lennon,
then Ophelia is currently 38 years old 
-/
theorem ophelia_age 
  (lennon_age : ℕ) 
  (ophelia_age_in_two_years : ℕ) 
  (h1 : lennon_age = 8)
  (h2 : ophelia_age_in_two_years = 4 * (lennon_age + 2)) : 
  ophelia_age_in_two_years - 2 = 38 :=
by
  sorry

end ophelia_age_l164_164404


namespace tom_age_ratio_l164_164678

theorem tom_age_ratio (T N : ℝ) (h1 : T - N = 3 * (T - 4 * N)) : T / N = 5.5 :=
by
  sorry

end tom_age_ratio_l164_164678


namespace opposite_of_negative_seven_l164_164831

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_negative_seven : opposite (-7) = 7 := 
by 
  sorry

end opposite_of_negative_seven_l164_164831


namespace isometric_curve_l164_164463

noncomputable def Q (a b c x y : ℝ) := a * x^2 + 2 * b * x * y + c * y^2

theorem isometric_curve (a b c d e f : ℝ) (h : a * c - b^2 = 0) :
  ∃ (p : ℝ), (Q a b c x y + 2 * d * x + 2 * e * y = f → 
    (y^2 = 2 * p * x) ∨ 
    (∃ c' : ℝ, y^2 = c'^2) ∨ 
    y^2 = 0 ∨ 
    ∀ x y : ℝ, false) :=
sorry

end isometric_curve_l164_164463


namespace probability_at_least_one_six_is_11_div_36_l164_164777

noncomputable def probability_at_least_one_six : ℚ :=
  let total_outcomes := 36
  let no_six_outcomes := 25
  let favorable_outcomes := total_outcomes - no_six_outcomes
  favorable_outcomes / total_outcomes
  
theorem probability_at_least_one_six_is_11_div_36 : 
  probability_at_least_one_six = 11 / 36 :=
by
  sorry

end probability_at_least_one_six_is_11_div_36_l164_164777


namespace hyperbola_distance_to_foci_l164_164659

theorem hyperbola_distance_to_foci
  (E : ∀ x y : ℝ, (x^2 / 9) - (y^2 / 16) = 1)
  (F1 F2 : ℝ)
  (P : ℝ)
  (dist_PF1 : P = 5)
  (a : ℝ)
  (ha : a = 3): 
  |P - F2| = 11 :=
by
  sorry

end hyperbola_distance_to_foci_l164_164659


namespace infinite_geometric_series_first_term_l164_164235

theorem infinite_geometric_series_first_term 
  (r : ℝ) 
  (S : ℝ) 
  (a : ℝ) 
  (h1 : r = -3/7) 
  (h2 : S = 18) 
  (h3 : S = a / (1 - r)) : 
  a = 180 / 7 := by
  -- omitted proof
  sorry

end infinite_geometric_series_first_term_l164_164235


namespace intersection_complement_l164_164111

open Set

-- Definitions from the problem
def U : Set ℝ := univ
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {y | 0 < y}

-- The proof statement
theorem intersection_complement : A ∩ (compl B) = Ioc (-1 : ℝ) 0 := by
  sorry

end intersection_complement_l164_164111


namespace condition_sufficient_not_necessary_l164_164773

theorem condition_sufficient_not_necessary (x : ℝ) : (1 < x ∧ x < 2) → ((x - 2) ^ 2 < 1) ∧ ¬ ((x - 2) ^ 2 < 1 → (1 < x ∧ x < 2)) :=
by
  sorry

end condition_sufficient_not_necessary_l164_164773


namespace value_of_expression_l164_164608

theorem value_of_expression (a b m n x : ℝ) 
    (hab : a * b = 1) 
    (hmn : m + n = 0) 
    (hxsq : x^2 = 1) : 
    2022 * (m + n) + 2018 * x^2 - 2019 * (a * b) = -1 := 
by 
    sorry

end value_of_expression_l164_164608


namespace winner_lifted_weight_l164_164078

theorem winner_lifted_weight (A B C : ℕ) 
  (h1 : A + B = 220)
  (h2 : A + C = 240) 
  (h3 : B + C = 250) : 
  C = 135 :=
by
  sorry

end winner_lifted_weight_l164_164078


namespace smallest_palindrome_not_five_digit_l164_164688

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.toDigits 10
  s = s.reverse

theorem smallest_palindrome_not_five_digit (n : ℕ) :
  (∃ n, is_palindrome n ∧ 100 ≤ n ∧ n < 1000 ∧ ¬is_palindrome (102 * n)) → n = 101 := by
  sorry

end smallest_palindrome_not_five_digit_l164_164688


namespace weight_first_watermelon_l164_164621

-- We define the total weight and the weight of the second watermelon
def total_weight := 14.02
def second_watermelon := 4.11

-- We need to prove that the weight of the first watermelon is 9.91 pounds
theorem weight_first_watermelon : total_weight - second_watermelon = 9.91 := by
  -- Insert mathematical steps here (omitted in this case)
  sorry

end weight_first_watermelon_l164_164621


namespace side_length_estimate_l164_164655

theorem side_length_estimate (x : ℝ) (h : x^2 = 15) : 3 < x ∧ x < 4 :=
sorry

end side_length_estimate_l164_164655


namespace no_solutions_to_equation_l164_164410

theorem no_solutions_to_equation :
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x ^ 2 - 2 * y ^ 2 = 5 := by
  sorry

end no_solutions_to_equation_l164_164410


namespace digit_7_count_in_range_l164_164271

def count_digit_7 : ℕ :=
  let units_place := (107 - 100) / 10 + 1
  let tens_place := 10
  units_place + tens_place

theorem digit_7_count_in_range : count_digit_7 = 20 := by
  sorry

end digit_7_count_in_range_l164_164271


namespace find_constants_l164_164644

noncomputable section

theorem find_constants (P Q R : ℝ)
  (h : ∀ x : ℝ, x ≠ 2 → x ≠ 4 →
    (5*x^2 + 7*x) / ((x - 2) * (x - 4)^2) =
    P / (x - 2) + Q / (x - 4) + R / (x - 4)^2) :
  P = 3.5 ∧ Q = 1.5 ∧ R = 18 :=
by
  sorry

end find_constants_l164_164644


namespace second_discount_correct_l164_164568

noncomputable def second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : ℝ :=
  let first_discount_amount := first_discount / 100 * original_price
  let price_after_first_discount := original_price - first_discount_amount
  let second_discount_amount := price_after_first_discount - final_price
  (second_discount_amount / price_after_first_discount) * 100

theorem second_discount_correct :
  second_discount_percentage 510 12 381.48 = 15 :=
by
  sorry

end second_discount_correct_l164_164568


namespace monotonically_increasing_interval_l164_164855

noncomputable def f (x : ℝ) : ℝ := 4 * x - x^3

theorem monotonically_increasing_interval : ∀ x1 x2 : ℝ, -2 < x1 ∧ x1 < x2 ∧ x2 < 2 → f x1 < f x2 :=
by
  intros x1 x2 h
  sorry

end monotonically_increasing_interval_l164_164855


namespace lines_connecting_intersections_l164_164357

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem lines_connecting_intersections (n : ℕ) (h : n ≥ 2) :
  let N := binomial n 2
  binomial N 2 = (n * n * (n - 1) * (n - 1) - 2 * n * (n - 1)) / 8 :=
by {
  sorry
}

end lines_connecting_intersections_l164_164357


namespace percentage_of_women_attended_picnic_l164_164857

variable (E : ℝ) -- total number of employees
variable (M : ℝ) -- number of men
variable (W : ℝ) -- number of women

-- 45% of all employees are men
axiom h1 : M = 0.45 * E
-- Rest of employees are women
axiom h2 : W = E - M
-- 20% of men attended the picnic
variable (x : ℝ) -- percentage of women who attended the picnic
axiom h3 : 0.20 * M + (x / 100) * W = 0.31000000000000007 * E

theorem percentage_of_women_attended_picnic : x = 40 :=
by
  sorry

end percentage_of_women_attended_picnic_l164_164857


namespace age_difference_l164_164053

theorem age_difference (P M Mo : ℕ)
  (h1 : P = 3 * M / 5)
  (h2 : Mo = 5 * M / 3)
  (h3 : P + M + Mo = 196) :
  Mo - P = 64 := 
sorry

end age_difference_l164_164053


namespace marathon_yards_l164_164556

theorem marathon_yards (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) (yards_per_mile : ℕ) (marathons_run : ℕ) 
  (total_miles : ℕ) (total_yards : ℕ) (h1 : miles_per_marathon = 26) (h2 : yards_per_marathon = 385)
  (h3 : yards_per_mile = 1760) (h4 : marathons_run = 15) (h5 : 
  total_miles = marathons_run * miles_per_marathon + (marathons_run * yards_per_marathon) / yards_per_mile) 
  (h6 : total_yards = (marathons_run * yards_per_marathon) % yards_per_mile) : 
  total_yards = 495 :=
by
  -- This will be our process to verify the transformation
  sorry

end marathon_yards_l164_164556


namespace ultratown_run_difference_l164_164693

/-- In Ultratown, the streets are all 25 feet wide, 
and the blocks they enclose are rectangular with lengths of 500 feet and widths of 300 feet. 
Hannah runs around the block on the longer 500-foot side of the street, 
while Harry runs on the opposite, outward side of the street. 
Prove that Harry runs 200 more feet than Hannah does for every lap around the block.
-/ 
theorem ultratown_run_difference :
  let street_width : ℕ := 25
  let inner_length : ℕ := 500
  let inner_width : ℕ := 300
  let outer_length := inner_length + 2 * street_width
  let outer_width := inner_width + 2 * street_width
  let inner_perimeter := 2 * (inner_length + inner_width)
  let outer_perimeter := 2 * (outer_length + outer_width)
  (outer_perimeter - inner_perimeter) = 200 :=
by
  sorry

end ultratown_run_difference_l164_164693


namespace Ian_kept_1_rose_l164_164586

def initial_roses : ℕ := 20
def roses_given_to_mother : ℕ := 6
def roses_given_to_grandmother : ℕ := 9
def roses_given_to_sister : ℕ := 4
def total_roses_given : ℕ := roses_given_to_mother + roses_given_to_grandmother + roses_given_to_sister
def roses_kept (initial: ℕ) (given: ℕ) : ℕ := initial - given

theorem Ian_kept_1_rose :
  roses_kept initial_roses total_roses_given = 1 :=
by
  sorry

end Ian_kept_1_rose_l164_164586


namespace least_sum_four_primes_gt_10_l164_164059

theorem least_sum_four_primes_gt_10 : 
  ∃ (p1 p2 p3 p4 : ℕ), 
    p1 > 10 ∧ p2 > 10 ∧ p3 > 10 ∧ p4 > 10 ∧ 
    Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    p1 + p2 + p3 + p4 = 60 ∧
    ∀ (q1 q2 q3 q4 : ℕ), 
      q1 > 10 ∧ q2 > 10 ∧ q3 > 10 ∧ q4 > 10 ∧ 
      Nat.Prime q1 ∧ Nat.Prime q2 ∧ Nat.Prime q3 ∧ Nat.Prime q4 ∧
      q1 ≠ q2 ∧ q1 ≠ q3 ∧ q1 ≠ q4 ∧ q2 ≠ q3 ∧ q2 ≠ q4 ∧ q3 ≠ q4 →
      q1 + q2 + q3 + q4 ≥ 60 :=
by
  sorry

end least_sum_four_primes_gt_10_l164_164059


namespace find_num_female_students_l164_164904

noncomputable def numFemaleStudents (totalAvg maleAvg femaleAvg : ℕ) (numMales : ℕ) : ℕ :=
  let numFemales := (totalAvg * (numMales + (totalAvg * 0)) - (maleAvg * numMales)) / femaleAvg
  numFemales

theorem find_num_female_students :
  (totalAvg maleAvg femaleAvg : ℕ) →
  (numMales : ℕ) →
  totalAvg = 90 →
  maleAvg = 83 →
  femaleAvg = 92 →
  numMales = 8 →
  numFemaleStudents totalAvg maleAvg femaleAvg numMales = 28 := by
    intros
    sorry

end find_num_female_students_l164_164904


namespace smallest_geometric_third_term_l164_164397

theorem smallest_geometric_third_term (d : ℝ) (a₁ a₂ a₃ g₁ g₂ g₃ : ℝ) 
  (h_AP : a₁ = 5 ∧ a₂ = 5 + d ∧ a₃ = 5 + 2 * d)
  (h_GP : g₁ = a₁ ∧ g₂ = a₂ + 3 ∧ g₃ = a₃ + 15)
  (h_geom : (g₂)^2 = g₁ * g₃) : g₃ = -4 := 
by
  -- We would provide the proof here.
  sorry

end smallest_geometric_third_term_l164_164397


namespace common_point_of_arithmetic_progression_lines_l164_164390

theorem common_point_of_arithmetic_progression_lines 
  (a d : ℝ) 
  (h₁ : a ≠ 0)
  (h_d_ne_zero : d ≠ 0) 
  (h₃ : ∀ (x y : ℝ), (x = -1 ∧ y = 1) ↔ (∃ a d : ℝ, a ≠ 0 ∧ d ≠ 0 ∧ a*(x) + (a-d)*y = (a-2*d))) :
  (∀ (x y : ℝ), (a ≠ 0 ∧ d ≠ 0 ∧ a*(x) + (a-d)*y = a-2*d) → x = -1 ∧ y = 1) :=
by 
  sorry

end common_point_of_arithmetic_progression_lines_l164_164390


namespace gcd_459_357_l164_164126

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l164_164126


namespace dice_probability_l164_164776

/-- A standard six-sided die -/
inductive Die : Type
| one | two | three | four | five | six

open Die

/-- Calculates the probability that after re-rolling four dice, at least four out of the six total dice show the same number,
given that initially six dice are rolled and there is no three-of-a-kind, and there is a pair of dice showing the same number
which are then set aside before re-rolling the remaining four dice. -/
theorem dice_probability (h1 : ∀ (d1 d2 d3 d4 d5 d6 : Die), 
  ¬ (d1 = d2 ∧ d2 = d3 ∨ d1 = d2 ∧ d2 = d4 ∨ d1 = d2 ∧ d2 = d5 ∨
     d1 = d2 ∧ d2 = d6 ∨ d1 = d3 ∧ d3 = d4 ∨ d1 = d3 ∧ d3 = d5 ∨
     d1 = d3 ∧ d3 = d6 ∨ d1 = d4 ∧ d4 = d5 ∨ d1 = d4 ∧ d4 = d6 ∨
     d1 = d5 ∧ d5 = d6 ∨ d2 = d3 ∧ d3 = d4 ∨ d2 = d3 ∧ d3 = d5 ∨
     d2 = d3 ∧ d3 = d6 ∨ d2 = d4 ∧ d4 = d5 ∨ d2 = d4 ∧ d4 = d6 ∨
     d2 = d5 ∧ d5 = d6 ∨ d3 = d4 ∧ d4 = d5 ∨ d3 = d4 ∧ d4 = d6 ∨ d3 = d5 ∧ d5 = d6 ∨ d4 = d5 ∧ d5 = d6))
    (h2 : ∃ (d1 d2 : Die) (d3 d4 d5 d6 : Die), d1 = d2 ∧ d3 ≠ d1 ∧ d4 ≠ d1 ∧ d5 ≠ d1 ∧ d6 ≠ d1): 
    ℚ := 
11 / 81

end dice_probability_l164_164776


namespace intersection_M_N_l164_164152

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
sorry

end intersection_M_N_l164_164152


namespace negate_at_most_two_l164_164882

def atMost (n : Nat) : Prop := ∃ k : Nat, k ≤ n
def atLeast (n : Nat) : Prop := ∃ k : Nat, k ≥ n

theorem negate_at_most_two : ¬ atMost 2 ↔ atLeast 3 := by
  sorry

end negate_at_most_two_l164_164882


namespace min_value_of_2x_plus_y_l164_164447

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + 2*x*y - 3 = 0) :
  2*x + y ≥ 3 :=
sorry

end min_value_of_2x_plus_y_l164_164447


namespace white_wash_cost_l164_164122

noncomputable def room_length : ℝ := 25
noncomputable def room_width : ℝ := 15
noncomputable def room_height : ℝ := 12
noncomputable def door_height : ℝ := 6
noncomputable def door_width : ℝ := 3
noncomputable def window_height : ℝ := 4
noncomputable def window_width : ℝ := 3
noncomputable def num_windows : ℕ := 3
noncomputable def cost_per_sqft : ℝ := 3

theorem white_wash_cost :
  let wall_area := 2 * (room_length * room_height + room_width * room_height)
  let door_area := door_height * door_width
  let window_area := window_height * window_width
  let total_non_white_wash_area := door_area + ↑num_windows * window_area
  let white_wash_area := wall_area - total_non_white_wash_area
  let total_cost := white_wash_area * cost_per_sqft
  total_cost = 2718 :=  
by
  sorry

end white_wash_cost_l164_164122


namespace width_of_wall_l164_164836

theorem width_of_wall (l : ℕ) (w : ℕ) (hl : l = 170) (hw : w = 5 * l + 80) : w = 930 := 
by
  sorry

end width_of_wall_l164_164836


namespace infinite_series_sum_l164_164254

/-- The sum of the infinite series ∑ 1/(n(n+3)) for n from 1 to ∞ is 7/9. -/
theorem infinite_series_sum :
  ∑' n, (1 : ℝ) / (n * (n + 3)) = 7 / 9 :=
sorry

end infinite_series_sum_l164_164254


namespace calculate_f_of_f_of_f_30_l164_164742

-- Define the function f (equivalent to $\#N = 0.5N + 2$)
def f (N : ℝ) : ℝ := 0.5 * N + 2

-- The proof statement
theorem calculate_f_of_f_of_f_30 : 
  f (f (f 30)) = 7.25 :=
by
  sorry

end calculate_f_of_f_of_f_30_l164_164742


namespace cricketer_boundaries_l164_164744

theorem cricketer_boundaries (total_runs : ℕ) (sixes : ℕ) (percent_runs_by_running : ℝ)
  (h1 : total_runs = 152)
  (h2 : sixes = 2)
  (h3 : percent_runs_by_running = 60.526315789473685) :
  let runs_by_running := round (total_runs * percent_runs_by_running / 100)
  let runs_from_sixes := sixes * 6
  let runs_from_boundaries := total_runs - runs_by_running - runs_from_sixes
  let boundaries := runs_from_boundaries / 4
  boundaries = 12 :=
by
  sorry

end cricketer_boundaries_l164_164744


namespace find_y_l164_164245

theorem find_y (n x y : ℕ) 
    (h1 : (n + 200 + 300 + x) / 4 = 250)
    (h2 : (300 + 150 + n + x + y) / 5 = 200) :
    y = 50 := 
by
  -- Placeholder for the proof
  sorry

end find_y_l164_164245


namespace number_of_rows_l164_164085

theorem number_of_rows (r : ℕ) (h1 : ∀ bus : ℕ, bus * (4 * r) = 240) : r = 10 :=
sorry

end number_of_rows_l164_164085


namespace dawn_hourly_income_l164_164288

theorem dawn_hourly_income 
  (n : ℕ) (t_s t_p t_f I_p I_s I_f : ℝ)
  (h_n : n = 12)
  (h_t_s : t_s = 1.5)
  (h_t_p : t_p = 2)
  (h_t_f : t_f = 0.5)
  (h_I_p : I_p = 3600)
  (h_I_s : I_s = 1200)
  (h_I_f : I_f = 300) :
  (I_p + I_s + I_f) / (n * (t_s + t_p + t_f)) = 106.25 := 
  by
  sorry

end dawn_hourly_income_l164_164288


namespace sqrt_108_eq_6_sqrt_3_l164_164916

theorem sqrt_108_eq_6_sqrt_3 : Real.sqrt 108 = 6 * Real.sqrt 3 := 
sorry

end sqrt_108_eq_6_sqrt_3_l164_164916


namespace largest_divisor_of_expression_l164_164120

theorem largest_divisor_of_expression (x : ℤ) (h_odd : x % 2 = 1) : 
  1200 ∣ ((10 * x - 4) * (10 * x) * (5 * x + 15)) := 
  sorry

end largest_divisor_of_expression_l164_164120


namespace remaining_bottle_caps_l164_164079

-- Definitions based on conditions
def initial_bottle_caps : ℕ := 65
def eaten_bottle_caps : ℕ := 4

-- Theorem
theorem remaining_bottle_caps : initial_bottle_caps - eaten_bottle_caps = 61 :=
by
  sorry

end remaining_bottle_caps_l164_164079


namespace total_peanuts_is_388_l164_164360

def peanuts_total (jose kenya marcos : ℕ) : ℕ :=
  jose + kenya + marcos

theorem total_peanuts_is_388 :
  ∀ (jose kenya marcos : ℕ),
    (jose = 85) →
    (kenya = jose + 48) →
    (marcos = kenya + 37) →
    peanuts_total jose kenya marcos = 388 := 
by
  intros jose kenya marcos h_jose h_kenya h_marcos
  sorry

end total_peanuts_is_388_l164_164360


namespace net_increase_in_bicycle_stock_l164_164716

-- Definitions for changes in stock over the three days
def net_change_friday : ℤ := 15 - 10
def net_change_saturday : ℤ := 8 - 12
def net_change_sunday : ℤ := 11 - 9

-- Total net increase in stock
def total_net_increase : ℤ := net_change_friday + net_change_saturday + net_change_sunday

-- Theorem statement
theorem net_increase_in_bicycle_stock : total_net_increase = 3 := by
  -- We would provide the detailed proof here.
  sorry

end net_increase_in_bicycle_stock_l164_164716


namespace eliana_steps_total_l164_164115

def eliana_walks_first_day_steps := 200 + 300
def eliana_walks_second_day_steps := 2 * eliana_walks_first_day_steps
def eliana_walks_third_day_steps := eliana_walks_second_day_steps + 100
def eliana_total_steps := eliana_walks_first_day_steps + eliana_walks_second_day_steps + eliana_walks_third_day_steps

theorem eliana_steps_total : eliana_total_steps = 2600 := by
  sorry

end eliana_steps_total_l164_164115


namespace rhombus_triangle_area_l164_164299

theorem rhombus_triangle_area (d1 d2 : ℝ) (h_d1 : d1 = 15) (h_d2 : d2 = 20) :
  ∃ (area : ℝ), area = 75 := 
by
  sorry

end rhombus_triangle_area_l164_164299


namespace common_ratio_solution_l164_164361

-- Define the problem condition
def geometric_sum_condition (a1 : ℝ) (q : ℝ) : Prop :=
  (a1 * (1 - q^3)) / (1 - q) = 3 * a1

-- Define the theorem we want to prove
theorem common_ratio_solution (a1 : ℝ) (q : ℝ) (h : geometric_sum_condition a1 q) :
  q = 1 ∨ q = -2 :=
sorry

end common_ratio_solution_l164_164361


namespace fraction_of_income_from_tips_l164_164427

theorem fraction_of_income_from_tips (S T : ℚ) (h : T = (11/4) * S) : (T / (S + T)) = (11/15) :=
by sorry

end fraction_of_income_from_tips_l164_164427


namespace smallest_nonneg_int_mod_15_l164_164036

theorem smallest_nonneg_int_mod_15 :
  ∃ x : ℕ, x + 7263 ≡ 3507 [MOD 15] ∧ ∀ y : ℕ, y + 7263 ≡ 3507 [MOD 15] → x ≤ y :=
by
  sorry

end smallest_nonneg_int_mod_15_l164_164036


namespace problem1_problem2_problem3_l164_164418

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

end problem1_problem2_problem3_l164_164418


namespace product_xyz_l164_164217

theorem product_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x * y = 30 * (4:ℝ)^(1/3)) (h5 : x * z = 45 * (4:ℝ)^(1/3)) (h6 : y * z = 18 * (4:ℝ)^(1/3)) :
  x * y * z = 540 * Real.sqrt 3 :=
sorry

end product_xyz_l164_164217


namespace vijay_work_alone_in_24_days_l164_164869

theorem vijay_work_alone_in_24_days (ajay_rate vijay_rate combined_rate : ℝ) 
  (h1 : ajay_rate = 1 / 8) 
  (h2 : combined_rate = 1 / 6) 
  (h3 : ajay_rate + vijay_rate = combined_rate) : 
  vijay_rate = 1 / 24 := 
sorry

end vijay_work_alone_in_24_days_l164_164869


namespace sum_a_for_exactly_one_solution_l164_164794

theorem sum_a_for_exactly_one_solution :
  (∀ a : ℝ, ∃ x : ℝ, 3 * x^2 + (a + 6) * x + 7 = 0) →
  ((-6 + 2 * Real.sqrt 21) + (-6 - 2 * Real.sqrt 21) = -12) :=
by
  sorry

end sum_a_for_exactly_one_solution_l164_164794


namespace parabola_intersects_x_axis_l164_164440

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 + 2 * x + m - 1

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := 4 - 4 * (m - 1)

-- Lean statement to prove the range of m
theorem parabola_intersects_x_axis (m : ℝ) : (∃ x : ℝ, quadratic x m = 0) ↔ m ≤ 2 := by
  sorry

end parabola_intersects_x_axis_l164_164440


namespace find_number_l164_164266

theorem find_number (x : ℝ) (h : x - (3 / 5) * x = 64) : x = 160 :=
sorry

end find_number_l164_164266


namespace increasing_function_iff_l164_164371

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a ^ x else (3 - a) * x + (1 / 2) * a

theorem increasing_function_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ 2 ≤ a ∧ a < 3 :=
by
  sorry

end increasing_function_iff_l164_164371


namespace algebraic_expression_value_l164_164914

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 2 + 1) : x^2 - 2 * x + 2 = 3 :=
by
  sorry

end algebraic_expression_value_l164_164914


namespace part1_prob_dist_part1_expectation_part2_prob_A_wins_n_throws_l164_164247

-- Definitions for part (1)
def P_X_1 : ℚ := 1 / 6
def P_X_2 : ℚ := 5 / 36
def P_X_3 : ℚ := 25 / 216
def P_X_4 : ℚ := 125 / 216
def E_X : ℚ := 671 / 216

theorem part1_prob_dist (X : ℚ) :
  (X = 1 → P_X_1 = 1 / 6) ∧
  (X = 2 → P_X_2 = 5 / 36) ∧
  (X = 3 → P_X_3 = 25 / 216) ∧
  (X = 4 → P_X_4 = 125 / 216) := 
by sorry

theorem part1_expectation :
  E_X = 671 / 216 :=
by sorry

-- Definition for part (2)
def P_A_wins_n_throws (n : ℕ) : ℚ := 1 / 6 * (5 / 6) ^ (2 * n - 2)

theorem part2_prob_A_wins_n_throws (n : ℕ) (hn : n ≥ 1) :
  P_A_wins_n_throws n = 1 / 6 * (5 / 6) ^ (2 * n - 2) :=
by sorry

end part1_prob_dist_part1_expectation_part2_prob_A_wins_n_throws_l164_164247


namespace isosceles_triangle_side_length_l164_164341

theorem isosceles_triangle_side_length (base : ℝ) (area : ℝ) (congruent_side : ℝ) 
  (h_base : base = 30) (h_area : area = 60) : congruent_side = Real.sqrt 241 :=
by 
  sorry

end isosceles_triangle_side_length_l164_164341


namespace minimum_value_of_expression_l164_164978

theorem minimum_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : (x / y + y / z + z / x) + (y / x + z / y + x / z) = 10) :
  ∃ P, (P = (x / y + y / z + z / x) * (y / x + z / y + x / z)) ∧ P = 25 := 
by sorry

end minimum_value_of_expression_l164_164978


namespace m_range_l164_164994

variable (a1 b1 : ℝ)

def arithmetic_sequence (n : ℕ) : ℝ := a1 + 2 * (n - 1)
def geometric_sequence (n : ℕ) : ℝ := b1 * 2^(n - 1)

def a2_condition : Prop := arithmetic_sequence a1 2 + geometric_sequence b1 2 < -2
def a1_b1_condition : Prop := a1 + b1 > 0

theorem m_range : a1_b1_condition a1 b1 ∧ a2_condition a1 b1 → 
  let a4 := arithmetic_sequence a1 4 
  let b3 := geometric_sequence b1 3 
  let m := a4 + b3 
  m < 0 := 
by
  sorry

end m_range_l164_164994


namespace solve_system_of_equations_l164_164618

theorem solve_system_of_equations :
  ∃ x y : ℝ, 
  (4 * x - 3 * y = -0.5) ∧ 
  (5 * x + 7 * y = 10.3) ∧ 
  (|x - 0.6372| < 1e-4) ∧ 
  (|y - 1.0163| < 1e-4) :=
by
  sorry

end solve_system_of_equations_l164_164618


namespace value_of_x_l164_164252

-- Define the conditions extracted from problem (a)
def condition1 (x : ℝ) : Prop := x^2 - 1 = 0
def condition2 (x : ℝ) : Prop := x - 1 ≠ 0

-- The statement to be proved
theorem value_of_x : ∀ x : ℝ, condition1 x → condition2 x → x = -1 :=
by
  intros x h1 h2
  sorry

end value_of_x_l164_164252


namespace solve_for_x_l164_164201

theorem solve_for_x (x : ℝ) (h : 6 * x ^ (1 / 3) - 3 * (x / x ^ (2 / 3)) = -1 + 2 * x ^ (1 / 3) + 4) :
  x = 27 :=
by 
  sorry

end solve_for_x_l164_164201


namespace hyperbola_m_range_l164_164561

theorem hyperbola_m_range (m : ℝ) (h_eq : ∀ x y, (x^2 / m) - (y^2 / (2*m - 1)) = 1) : 
  0 < m ∧ m < 1/2 :=
sorry

end hyperbola_m_range_l164_164561


namespace parallel_lines_slope_equal_intercepts_lines_l164_164224

theorem parallel_lines_slope (m : ℝ) :
  (∀ x y, (2 * x - y - 3 = 0 ∧ x - m * y + 1 - 3 * m = 0) → 2 = (1 / m)) → m = 1 / 2 :=
by
  intro h
  sorry

theorem equal_intercepts_lines (m : ℝ) :
  (m ≠ 0 → (∀ x y, (x - m * y + 1 - 3 * m = 0) → (1 - 3 * m) / m = 3 * m - 1)) →
  (m = -1 ∨ m = 1 / 3) →
  ∀ x y, (x - m * y + 1 - 3 * m = 0) →
  (x + y + 4 = 0 ∨ 3 * x - y = 0) :=
by
  intro h hm
  sorry

end parallel_lines_slope_equal_intercepts_lines_l164_164224


namespace fill_time_60_gallons_ten_faucets_l164_164826

-- Define the problem parameters
def rate_of_five_faucets : ℚ := 150 / 8 -- in gallons per minute

def rate_of_one_faucet : ℚ := rate_of_five_faucets / 5

def rate_of_ten_faucets : ℚ := rate_of_one_faucet * 10

def time_to_fill_60_gallons_minutes : ℚ := 60 / rate_of_ten_faucets

def time_to_fill_60_gallons_seconds : ℚ := time_to_fill_60_gallons_minutes * 60

-- The main theorem to prove
theorem fill_time_60_gallons_ten_faucets : time_to_fill_60_gallons_seconds = 96 := by
  sorry

end fill_time_60_gallons_ten_faucets_l164_164826


namespace phillip_initial_marbles_l164_164188

theorem phillip_initial_marbles
  (dilan_marbles : ℕ) (martha_marbles : ℕ) (veronica_marbles : ℕ) 
  (total_after_redistribution : ℕ) 
  (individual_marbles_after : ℕ) :
  dilan_marbles = 14 →
  martha_marbles = 20 →
  veronica_marbles = 7 →
  total_after_redistribution = 4 * individual_marbles_after →
  individual_marbles_after = 15 →
  ∃phillip_marbles : ℕ, phillip_marbles = 19 :=
by
  intro h_dilan h_martha h_veronica h_total_after h_individual
  have total_initial := 60 - (14 + 20 + 7)
  existsi total_initial
  sorry

end phillip_initial_marbles_l164_164188


namespace trigonometric_identities_l164_164842

theorem trigonometric_identities (α : Real) (h1 : 3 * π / 2 < α ∧ α < 2 * π) (h2 : Real.sin α = -3 / 5) :
  Real.tan α = 3 / 4 ∧ Real.tan (α - π / 4) = -1 / 7 ∧ Real.cos (2 * α) = 7 / 25 :=
by
  sorry

end trigonometric_identities_l164_164842


namespace quadratic_roots_are_correct_l164_164748

theorem quadratic_roots_are_correct (x: ℝ) : 
    (x^2 + x - 1 = 0) ↔ (x = (-1 + Real.sqrt 5) / 2) ∨ (x = (-1 - Real.sqrt 5) / 2) := 
by sorry

end quadratic_roots_are_correct_l164_164748


namespace probability_of_exactly_one_red_ball_l164_164109

-- Definitions based on the conditions:
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def draw_count : ℕ := 2

-- Required to calculate combinatory values
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Definitions of probabilities (though we won't use them explicitly for the statement):
def total_events : ℕ := choose total_balls draw_count
def no_red_ball_events : ℕ := choose white_balls draw_count
def one_red_ball_events : ℕ := choose red_balls 1 * choose white_balls 1

-- Probability Functions (for context):
def probability (events : ℕ) (total_events : ℕ) : ℚ := events / total_events

-- Lean 4 statement:
theorem probability_of_exactly_one_red_ball :
  probability one_red_ball_events total_events = 3/5 := by
  sorry

end probability_of_exactly_one_red_ball_l164_164109


namespace inverse_variation_l164_164143

theorem inverse_variation (k : ℝ) : 
  (∀ (x y : ℝ), x * y^2 = k) → 
  (∀ (x y : ℝ), x = 1 → y = 2 → k = 4) → 
  (x = 0.1111111111111111) → 
  (y = 6) :=
by 
  -- Assume the given conditions
  intros h h0 hx
  -- Proof goes here...
  sorry

end inverse_variation_l164_164143


namespace remainder_when_sum_divided_by_30_l164_164529

theorem remainder_when_sum_divided_by_30 {c d : ℕ} (p q : ℕ)
  (hc : c = 60 * p + 58)
  (hd : d = 90 * q + 85) :
  (c + d) % 30 = 23 :=
by
  sorry

end remainder_when_sum_divided_by_30_l164_164529


namespace taco_castle_parking_lot_l164_164010

variable (D F T V : ℕ)

theorem taco_castle_parking_lot (h1 : F = D / 3) (h2 : F = 2 * T) (h3 : V = T / 2) (h4 : V = 5) : D = 60 :=
by
  sorry

end taco_castle_parking_lot_l164_164010


namespace smarties_modulo_l164_164738

theorem smarties_modulo (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 :=
by
  sorry

end smarties_modulo_l164_164738


namespace silk_dyeing_total_correct_l164_164733

open Real

theorem silk_dyeing_total_correct :
  let green := 61921
  let pink := 49500
  let blue := 75678
  let yellow := 34874.5
  let total_without_red := green + pink + blue + yellow
  let red := 0.10 * total_without_red
  let total_with_red := total_without_red + red
  total_with_red = 245270.85 :=
by
  sorry

end silk_dyeing_total_correct_l164_164733


namespace postage_problem_l164_164969

noncomputable def sum_all_positive_integers (n1 n2 : ℕ) : ℕ :=
  n1 + n2

theorem postage_problem : sum_all_positive_integers 21 22 = 43 :=
by
  have h1 : ∀ x y z : ℕ, 7 * x + 21 * y + 23 * z ≠ 120 := sorry
  have h2 : ∀ x y z : ℕ, 7 * x + 22 * y + 24 * z ≠ 120 := sorry
  exact rfl

end postage_problem_l164_164969


namespace acres_used_for_corn_l164_164267

-- Define the conditions
def total_acres : ℝ := 5746
def ratio_beans : ℝ := 7.5
def ratio_wheat : ℝ := 3.2
def ratio_corn : ℝ := 5.6
def total_parts : ℝ := ratio_beans + ratio_wheat + ratio_corn

-- Define the statement to prove
theorem acres_used_for_corn : (total_acres / total_parts) * ratio_corn = 1975.46 :=
by
  -- Placeholder for the proof; to be completed separately
  sorry

end acres_used_for_corn_l164_164267


namespace diet_soda_bottles_l164_164307

def total_bottles : ℕ := 17
def regular_soda_bottles : ℕ := 9

theorem diet_soda_bottles : total_bottles - regular_soda_bottles = 8 := by
  sorry

end diet_soda_bottles_l164_164307


namespace sum_gcd_lcm_l164_164531

theorem sum_gcd_lcm (a b c d : ℕ) (ha : a = 15) (hb : b = 45) (hc : c = 30) :
  Int.gcd a b + Nat.lcm a c = 45 := 
by
  sorry

end sum_gcd_lcm_l164_164531


namespace parallel_lines_iff_l164_164250

theorem parallel_lines_iff (a : ℝ) :
  (∀ x y : ℝ, x - y - 1 = 0 → x + a * y - 2 = 0) ↔ (a = -1) :=
by
  sorry

end parallel_lines_iff_l164_164250


namespace rosy_fish_is_twelve_l164_164344

/-- Let lilly_fish be the number of fish Lilly has. -/
def lilly_fish : ℕ := 10

/-- Let total_fish be the total number of fish Lilly and Rosy have together. -/
def total_fish : ℕ := 22

/-- Prove that the number of fish Rosy has is equal to 12. -/
theorem rosy_fish_is_twelve : (total_fish - lilly_fish) = 12 :=
by sorry

end rosy_fish_is_twelve_l164_164344


namespace find_a_l164_164154

def star (x y : ℤ × ℤ) : ℤ × ℤ := (x.1 - y.1, x.2 + y.2)

theorem find_a :
  ∃ (a b : ℤ), 
  star (5, 2) (1, 1) = (a, b) ∧
  star (a, b) (0, 1) = (2, 5) ∧
  a = 2 :=
sorry

end find_a_l164_164154


namespace cos_alpha_minus_beta_l164_164381

theorem cos_alpha_minus_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (h_cos_add : Real.cos (α + β) = -5 / 13)
  (h_tan_sum : Real.tan α + Real.tan β = 3) :
  Real.cos (α - β) = 1 :=
by
  sorry

end cos_alpha_minus_beta_l164_164381


namespace find_number_l164_164995

theorem find_number (N: ℕ): (N % 131 = 112) ∧ (N % 132 = 98) → 1000 ≤ N ∧ N ≤ 9999 ∧ N = 1946 :=
sorry

end find_number_l164_164995


namespace quadratic_min_value_l164_164015

theorem quadratic_min_value (p q r : ℝ) (h : ∀ x : ℝ, x^2 + p * x + q + r ≥ -r) : q = p^2 / 4 :=
sorry

end quadratic_min_value_l164_164015


namespace opposite_of_neg_2_l164_164799

noncomputable def opposite (a : ℤ) : ℤ := 
  a * (-1)

theorem opposite_of_neg_2 : opposite (-2) = 2 := by
  -- definition of opposite
  unfold opposite
  -- calculation using the definition
  rfl

end opposite_of_neg_2_l164_164799


namespace output_value_is_3_l164_164533

-- Define the variables and the program logic
def program (a b : ℕ) : ℕ :=
  if a > b then a else b

-- The theorem statement
theorem output_value_is_3 (a b : ℕ) (ha : a = 2) (hb : b = 3) : program a b = 3 :=
by
  -- Automatically assume the given conditions and conclude the proof. The actual proof is skipped.
  sorry

end output_value_is_3_l164_164533


namespace Daria_money_l164_164813

theorem Daria_money (num_tickets : ℕ) (price_per_ticket : ℕ) (amount_needed : ℕ) (h1 : num_tickets = 4) (h2 : price_per_ticket = 90) (h3 : amount_needed = 171) : 
  (num_tickets * price_per_ticket) - amount_needed = 189 := 
by 
  sorry

end Daria_money_l164_164813


namespace solve_equation_l164_164562

noncomputable def equation_to_solve (x : ℝ) : ℝ :=
  1 / (4^(3*x) - 13 * 4^(2*x) + 51 * 4^x - 60) + 1 / (4^(2*x) - 7 * 4^x + 12)

theorem solve_equation :
  (equation_to_solve (1/2) = 0) ∧ (equation_to_solve (Real.log 6 / Real.log 4) = 0) :=
by {
  sorry
}

end solve_equation_l164_164562


namespace total_heads_eq_fifteen_l164_164628

-- Definitions for types of passengers and their attributes
def cats_heads : Nat := 7
def cats_legs : Nat := 7 * 4
def total_legs : Nat := 43
def captain_heads : Nat := 1
def captain_legs : Nat := 1

noncomputable def crew_heads (C : Nat) : Nat := C
noncomputable def crew_legs (C : Nat) : Nat := 2 * C

theorem total_heads_eq_fifteen : 
  ∃ (C : Nat),
    cats_legs + crew_legs C + captain_legs = total_legs ∧
    cats_heads + crew_heads C + captain_heads = 15 :=
by
  sorry

end total_heads_eq_fifteen_l164_164628


namespace entree_cost_difference_l164_164823

theorem entree_cost_difference 
  (total_cost : ℕ)
  (entree_cost : ℕ)
  (dessert_cost : ℕ)
  (h1 : total_cost = 23)
  (h2 : entree_cost = 14)
  (h3 : total_cost = entree_cost + dessert_cost) :
  entree_cost - dessert_cost = 5 :=
by
  sorry

end entree_cost_difference_l164_164823


namespace probability_of_X_le_1_l164_164577

noncomputable def C (n k : ℕ) : ℚ := Nat.choose n k

noncomputable def P_X_le_1 := 
  (C 4 3 / C 6 3) + (C 4 2 * C 2 1 / C 6 3)

theorem probability_of_X_le_1 : P_X_le_1 = 4 / 5 := by
  sorry

end probability_of_X_le_1_l164_164577


namespace count_arithmetic_sequence_l164_164900

theorem count_arithmetic_sequence: 
  ∃ n : ℕ, (2 + (n - 1) * 3 = 2014) ∧ n = 671 := 
sorry

end count_arithmetic_sequence_l164_164900


namespace ral_age_is_26_l164_164598

def ral_current_age (suri_age : ℕ) (ral_age : ℕ) : Prop :=
  ral_age = 2 * suri_age

theorem ral_age_is_26 (suri_current_age : ℕ) (ral_current_age : ℕ) (h1 : suri_current_age + 3 = 16) (h2 : ral_age = 2 * suri_age) : ral_current_age = 26 := 
by
  sorry

end ral_age_is_26_l164_164598


namespace find_natural_numbers_with_integer_roots_l164_164451

theorem find_natural_numbers_with_integer_roots :
  ∃ (p q : ℕ), 
    (∀ x : ℤ, x * x - (p * q) * x + (p + q) = 0 → ∃ (x1 x2 : ℤ), x = x1 ∧ x = x2 ∧ x1 + x2 = p * q ∧ x1 * x2 = p + q) ↔
    ((p = 1 ∧ q = 5) ∨ (p = 5 ∧ q = 1) ∨ (p = 2 ∧ q = 2) ∨ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by
-- proof skipped
sorry

end find_natural_numbers_with_integer_roots_l164_164451


namespace years_to_rise_to_chief_l164_164607

-- Definitions based on the conditions
def ageWhenRetired : ℕ := 46
def ageWhenJoined : ℕ := 18
def additionalYearsAsMasterChief : ℕ := 10
def multiplierForChiefToMasterChief : ℚ := 1.25

-- Total years spent in the military
def totalYearsInMilitary : ℕ := ageWhenRetired - ageWhenJoined

-- Given conditions and correct answer
theorem years_to_rise_to_chief (x : ℚ) (h : totalYearsInMilitary = x + multiplierForChiefToMasterChief * x + additionalYearsAsMasterChief) :
  x = 8 := by
  sorry

end years_to_rise_to_chief_l164_164607


namespace stacy_current_height_l164_164725

-- Conditions
def last_year_height_stacy : ℕ := 50
def brother_growth : ℕ := 1
def stacy_growth : ℕ := brother_growth + 6

-- Statement to prove
theorem stacy_current_height : last_year_height_stacy + stacy_growth = 57 :=
by
  sorry

end stacy_current_height_l164_164725


namespace ben_initial_marbles_l164_164940

theorem ben_initial_marbles (B : ℕ) (John_initial_marbles : ℕ) (H1 : John_initial_marbles = 17) (H2 : John_initial_marbles + B / 2 = B / 2 + B / 2 + 17) : B = 34 := by
  sorry

end ben_initial_marbles_l164_164940


namespace janna_sleep_hours_l164_164664

-- Define the sleep hours from Monday to Sunday with the specified conditions
def sleep_hours_monday : ℕ := 7
def sleep_hours_tuesday : ℕ := 7 + 1 / 2
def sleep_hours_wednesday : ℕ := 7
def sleep_hours_thursday : ℕ := 7 + 1 / 2
def sleep_hours_friday : ℕ := 7 + 1
def sleep_hours_saturday : ℕ := 8
def sleep_hours_sunday : ℕ := 8

-- Calculate the total sleep hours in a week
noncomputable def total_sleep_hours : ℕ :=
  sleep_hours_monday +
  sleep_hours_tuesday +
  sleep_hours_wednesday +
  sleep_hours_thursday +
  sleep_hours_friday +
  sleep_hours_saturday +
  sleep_hours_sunday

-- The statement we want to prove
theorem janna_sleep_hours : total_sleep_hours = 53 := by
  sorry

end janna_sleep_hours_l164_164664


namespace weaving_additional_yards_l164_164850

theorem weaving_additional_yards {d : ℝ} :
  (∃ d : ℝ, (30 * 5 + (30 * 29) / 2 * d = 390) → d = 16 / 29) :=
sorry

end weaving_additional_yards_l164_164850


namespace skips_in_one_meter_l164_164488

variable (p q r s t u : ℕ)

theorem skips_in_one_meter (h1 : p * s * u = q * r * t) : 1 = (p * r * t) / (u * s * q) := by
  sorry

end skips_in_one_meter_l164_164488


namespace goldfish_graph_finite_set_of_points_l164_164407

-- Define the cost function for goldfish including the setup fee
def cost (n : ℕ) : ℝ := 20 * n + 5

-- Define the condition
def n_values := {n : ℕ | 1 ≤ n ∧ n ≤ 12}

-- The Lean statement to prove the nature of the graph
theorem goldfish_graph_finite_set_of_points :
  ∀ n ∈ n_values, ∃ k : ℝ, (k = cost n) :=
by
  sorry

end goldfish_graph_finite_set_of_points_l164_164407


namespace required_blue_balls_to_remove_l164_164639

-- Define the constants according to conditions
def total_balls : ℕ := 120
def red_balls : ℕ := 54
def initial_blue_balls : ℕ := total_balls - red_balls
def desired_percentage_red : ℚ := 0.75 -- ℚ is the type for rational numbers

-- Lean theorem statement
theorem required_blue_balls_to_remove (x : ℕ) : 
    (red_balls:ℚ) / (total_balls - x : ℚ) = desired_percentage_red → x = 48 :=
by
  sorry

end required_blue_balls_to_remove_l164_164639


namespace problem_l164_164861

variable (a : ℕ → ℝ) (n m : ℕ)

-- Condition: non-negative sequence and a_{n+m} ≤ a_n + a_m
axiom condition (n m : ℕ) : a n ≥ 0 ∧ a (n + m) ≤ a n + a m

-- Theorem: for any n ≥ m
theorem problem (h : n ≥ m) : a n ≤ m * a 1 + ((n / m) - 1) * a m :=
sorry

end problem_l164_164861


namespace solve_x_l164_164582

theorem solve_x (x : ℝ) (h : 2 - 2 / (1 - x) = 2 / (1 - x)) : x = -2 := 
by
  sorry

end solve_x_l164_164582


namespace correct_exp_identity_l164_164933

variable (a b : ℝ)

theorem correct_exp_identity : ((a^2 * b)^3 / (-a * b)^2 = a^4 * b) := sorry

end correct_exp_identity_l164_164933


namespace number_of_customers_l164_164966

theorem number_of_customers 
    (boxes_opened : ℕ) 
    (samples_per_box : ℕ) 
    (samples_left_over : ℕ) 
    (samples_limit_per_person : ℕ)
    (h1 : boxes_opened = 12)
    (h2 : samples_per_box = 20)
    (h3 : samples_left_over = 5)
    (h4 : samples_limit_per_person = 1) : 
    ∃ customers : ℕ, customers = (boxes_opened * samples_per_box) - samples_left_over ∧ customers = 235 :=
by {
  sorry
}

end number_of_customers_l164_164966


namespace probability_all_even_l164_164095

theorem probability_all_even :
  let die1_even_count := 3
  let die1_total := 6
  let die2_even_count := 3
  let die2_total := 7
  let die3_even_count := 4
  let die3_total := 9
  let prob_die1_even := die1_even_count / die1_total
  let prob_die2_even := die2_even_count / die2_total
  let prob_die3_even := die3_even_count / die3_total
  let probability_all_even := prob_die1_even * prob_die2_even * prob_die3_even
  probability_all_even = 1 / 10.5 :=
by
  sorry

end probability_all_even_l164_164095


namespace investment_period_two_years_l164_164014

theorem investment_period_two_years
  (P : ℝ) (r : ℝ) (A : ℝ) (n : ℕ) (hP : P = 6000) (hr : r = 0.10) (hA : A = 7260) (hn : n = 1) : 
  ∃ t : ℝ, t = 2 ∧ A = P * (1 + r / n) ^ (n * t) :=
by
  sorry

end investment_period_two_years_l164_164014


namespace part_one_part_two_l164_164093

variable {a b c : ℝ}
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a^2 + b^2 + 4*c^2 = 3)

theorem part_one (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  a + b + 2*c ≤ 3 :=
sorry

theorem part_two (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + 4*c^2 = 3) (h_b_eq_2c : b = 2*c) :
  1/a + 1/c ≥ 3 :=
sorry

end part_one_part_two_l164_164093


namespace exponent_subtraction_l164_164068

theorem exponent_subtraction (a : ℝ) (m n : ℕ) (h1 : a^m = 6) (h2 : a^n = 2) : a^(m - n) = 3 := by
  sorry

end exponent_subtraction_l164_164068


namespace soccer_ball_diameter_l164_164612

theorem soccer_ball_diameter 
  (h : ℝ)
  (s : ℝ)
  (d : ℝ)
  (h_eq : h = 1.25)
  (s_eq : s = 1)
  (d_eq : d = 0.23) : 2 * (d * h / (s - h)) = 0.46 :=
by
  sorry

end soccer_ball_diameter_l164_164612


namespace binomial_coefficient_10_3_l164_164579

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l164_164579


namespace f_2015_l164_164539

noncomputable def f : ℝ → ℝ := sorry
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_periodic : ∀ x : ℝ, f (x - 2) = -f x
axiom f_initial_segment : ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 0) → f x = 2^x

theorem f_2015 : f 2015 = 1 / 2 :=
by
  -- Proof goes here
  sorry

end f_2015_l164_164539


namespace hexagon_height_correct_l164_164553

-- Define the dimensions of the original rectangle
def original_rectangle_width := 16
def original_rectangle_height := 9
def original_rectangle_area := original_rectangle_width * original_rectangle_height

-- Define the dimensions of the new rectangle formed by the hexagons
def new_rectangle_width := 12
def new_rectangle_height := 12
def new_rectangle_area := new_rectangle_width * new_rectangle_height

-- Define the parameter x, which is the height of the hexagons
def hexagon_height := 6

-- Theorem stating the equivalence of the areas and the specific height x
theorem hexagon_height_correct :
  original_rectangle_area = new_rectangle_area ∧
  hexagon_height * 2 = new_rectangle_height :=
by
  sorry

end hexagon_height_correct_l164_164553


namespace at_least_two_consecutive_heads_probability_l164_164151

noncomputable def probability_at_least_two_consecutive_heads : ℚ := 
  let total_outcomes := 16
  let unfavorable_outcomes := 8
  1 - (unfavorable_outcomes / total_outcomes)

theorem at_least_two_consecutive_heads_probability :
  probability_at_least_two_consecutive_heads = 1 / 2 := 
by
  sorry

end at_least_two_consecutive_heads_probability_l164_164151


namespace max_writers_and_editors_l164_164806

theorem max_writers_and_editors (total people writers editors y x : ℕ) 
  (h1 : total = 110) 
  (h2 : writers = 45) 
  (h3 : editors = 38 + y) 
  (h4 : y > 0) 
  (h5 : 45 + editors + 2 * x = 110) : 
  x = 13 := 
sorry

end max_writers_and_editors_l164_164806


namespace max_value_real_roots_l164_164443

theorem max_value_real_roots (k x1 x2 : ℝ) :
  (∀ k, k^2 + 3 * k + 5 ≥ 0) →
  (x1 + x2 = k - 2) →
  (x1 * x2 = k^2 + 3 * k + 5) →
  (x1^2 + x2^2 ≤ 18) :=
by
  intro h1 h2 h3
  sorry

end max_value_real_roots_l164_164443


namespace binary_to_decimal_l164_164249

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4 + 1 * 2^5) = 51 := by
  sorry

end binary_to_decimal_l164_164249


namespace frustum_lateral_surface_area_l164_164433

theorem frustum_lateral_surface_area (r1 r2 h : ℝ) (r1_eq : r1 = 10) (r2_eq : r2 = 4) (h_eq : h = 6) :
  let s := Real.sqrt (h^2 + (r1 - r2)^2)
  let A := Real.pi * (r1 + r2) * s
  A = 84 * Real.pi * Real.sqrt 2 :=
by
  sorry

end frustum_lateral_surface_area_l164_164433


namespace minimum_value_exists_l164_164507

noncomputable def min_value (a b c : ℝ) : ℝ :=
  a / (3 * b^2) + b / (4 * c^3) + c / (5 * a^4)

theorem minimum_value_exists :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → abc = 1 → min_value a b c ≥ 1 :=
by
  sorry

end minimum_value_exists_l164_164507


namespace first_house_bottles_l164_164165

theorem first_house_bottles (total_bottles : ℕ) 
  (cider_only : ℕ) (beer_only : ℕ) (half : ℕ → ℕ)
  (mixture : ℕ)
  (half_cider_bottles : ℕ)
  (half_beer_bottles : ℕ)
  (half_mixture_bottles : ℕ) : 
  total_bottles = 180 →
  cider_only = 40 →
  beer_only = 80 →
  mixture = total_bottles - (cider_only + beer_only) →
  half c = c / 2 →
  half_cider_bottles = half cider_only →
  half_beer_bottles = half beer_only →
  half_mixture_bottles = half mixture →
  half_cider_bottles + half_beer_bottles + half_mixture_bottles = 90 :=
by
  intros h_tot h_cid h_beer h_mix h_half half_cid half_beer half_mix
  sorry

end first_house_bottles_l164_164165


namespace identity_element_exists_identity_element_self_commutativity_associativity_l164_164938

noncomputable def star_op (a b : ℤ) : ℤ := a + b + a * b

theorem identity_element_exists : ∃ E : ℤ, ∀ a : ℤ, star_op a E = a :=
by sorry

theorem identity_element_self (E : ℤ) (h1 : ∀ a : ℤ, star_op a E = a) : star_op E E = E :=
by sorry

theorem commutativity (a b : ℤ) : star_op a b = star_op b a :=
by sorry

theorem associativity (a b c : ℤ) : star_op (star_op a b) c = star_op a (star_op b c) :=
by sorry

end identity_element_exists_identity_element_self_commutativity_associativity_l164_164938


namespace average_rate_dan_trip_l164_164233

/-- 
Given:
- Dan runs along a 4-mile stretch of river and then swims back along the same route.
- Dan runs at a rate of 10 miles per hour.
- Dan swims at a rate of 6 miles per hour.

Prove:
Dan's average rate for the entire trip is 0.125 miles per minute.
-/
theorem average_rate_dan_trip :
  let distance := 4 -- miles
  let run_rate := 10 -- miles per hour
  let swim_rate := 6 -- miles per hour
  let time_run_hours := distance / run_rate -- hours
  let time_swim_hours := distance / swim_rate -- hours
  let time_run_minutes := time_run_hours * 60 -- minutes
  let time_swim_minutes := time_swim_hours * 60 -- minutes
  let total_distance := distance + distance -- miles
  let total_time := time_run_minutes + time_swim_minutes -- minutes
  let average_rate := total_distance / total_time -- miles per minute
  average_rate = 0.125 :=
by sorry

end average_rate_dan_trip_l164_164233


namespace solve_system_l164_164970

theorem solve_system :
  ∃ x y : ℝ, (x^2 - 9 * y^2 = 0 ∧ 2 * x - 3 * y = 6) ∧ (x = 6 ∧ y = 2) ∨ (x = 2 ∧ y = -2 / 3) :=
by
  -- The proof will go here
  sorry

end solve_system_l164_164970


namespace catFinishesOnMondayNextWeek_l164_164413

def morningConsumptionDaily (day : String) : ℚ := if day = "Wednesday" then 1 / 3 else 1 / 4
def eveningConsumptionDaily : ℚ := 1 / 6

def totalDailyConsumption (day : String) : ℚ :=
  morningConsumptionDaily day + eveningConsumptionDaily

-- List of days in order
def week : List String := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

-- Total food available initially
def totalInitialFood : ℚ := 8

-- Function to calculate total food consumed until a given day
def foodConsumedUntil (day : String) : ℚ :=
  week.takeWhile (· != day) |>.foldl (λ acc d => acc + totalDailyConsumption d) 0

-- Function to determine the day when 8 cans are completely consumed
def finishingDay : String :=
  match week.find? (λ day => foodConsumedUntil day + totalDailyConsumption day = totalInitialFood) with
  | some day => day
  | none => "Monday"  -- If no exact match is found in the first week, it is Monday of the next week

theorem catFinishesOnMondayNextWeek :
  finishingDay = "Monday" := by
  sorry

end catFinishesOnMondayNextWeek_l164_164413


namespace solve_y_eq_l164_164181

theorem solve_y_eq :
  ∀ y: ℝ, y ≠ -1 → (y^3 - 3 * y^2) / (y^2 + 2 * y + 1) + 2 * y = -1 → 
  y = 1 / Real.sqrt 3 ∨ y = -1 / Real.sqrt 3 :=
by sorry

end solve_y_eq_l164_164181


namespace number_of_sides_of_polygon_l164_164589

theorem number_of_sides_of_polygon (n : ℕ) :
  ((n - 2) * 180 = 3 * 360) → n = 8 := 
by
  sorry

end number_of_sides_of_polygon_l164_164589


namespace value_of_x_l164_164824

theorem value_of_x (y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 := by
  sorry

end value_of_x_l164_164824


namespace gwen_science_problems_l164_164899

theorem gwen_science_problems (math_problems : ℕ) (finished_problems : ℕ) (remaining_problems : ℕ)
  (h1 : math_problems = 18) (h2 : finished_problems = 24) (h3 : remaining_problems = 5) :
  (finished_problems + remaining_problems - math_problems = 11) :=
by
  sorry

end gwen_science_problems_l164_164899


namespace expression_value_l164_164822

theorem expression_value (x y z : ℕ) (hx : x = 5) (hy : y = 4) (hz : z = 3) :
  ( (1 / (y : ℚ)) + (1 / (z : ℚ))) / (1 / (x : ℚ)) = 35 / 12 := by
  sorry

end expression_value_l164_164822


namespace possible_values_of_m_l164_164479

-- Defining sets A and B based on the given conditions
def set_A : Set ℝ := { x | x^2 - 2 * x - 3 = 0 }
def set_B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }

-- The main theorem statement
theorem possible_values_of_m (m : ℝ) :
  (set_A ∪ set_B m = set_A) ↔ (m = 0 ∨ m = -1 / 3 ∨ m = 1) := by
  sorry

end possible_values_of_m_l164_164479


namespace expand_product_l164_164513

-- Define x as a variable within the real numbers
variable (x : ℝ)

-- Statement of the theorem
theorem expand_product : (x + 3) * (x - 4) = x^2 - x - 12 := 
by 
  sorry

end expand_product_l164_164513


namespace infinite_geometric_series_sum_l164_164745

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * (r ^ n) = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l164_164745


namespace negation_exists_implies_forall_l164_164283

theorem negation_exists_implies_forall (x_0 : ℝ) (h : ∃ x_0 : ℝ, x_0^3 - x_0 + 1 > 0) : 
  ¬ (∃ x_0 : ℝ, x_0^3 - x_0 + 1 > 0) ↔ ∀ x : ℝ, x^3 - x + 1 ≤ 0 :=
by 
  sorry

end negation_exists_implies_forall_l164_164283


namespace power_of_i_l164_164674

theorem power_of_i (i : ℂ) (h₀ : i^2 = -1) : i^(2016) = 1 :=
by {
  -- Proof will go here
  sorry
}

end power_of_i_l164_164674


namespace book_has_50_pages_l164_164001

noncomputable def sentences_per_hour : ℕ := 200
noncomputable def hours_to_read : ℕ := 50
noncomputable def sentences_per_paragraph : ℕ := 10
noncomputable def paragraphs_per_page : ℕ := 20

theorem book_has_50_pages :
  (sentences_per_hour * hours_to_read) / sentences_per_paragraph / paragraphs_per_page = 50 :=
by
  sorry

end book_has_50_pages_l164_164001


namespace youth_gathering_l164_164563

theorem youth_gathering (x : ℕ) (h1 : ∃ x, 9 * (2 * x + 12) = 20 * x) : 
  2 * x + 12 = 120 :=
by sorry

end youth_gathering_l164_164563


namespace compare_neg_fractions_l164_164155

theorem compare_neg_fractions : (- (2 / 3) < - (1 / 2)) :=
sorry

end compare_neg_fractions_l164_164155


namespace pump1_half_drain_time_l164_164648

-- Definitions and Conditions
def time_to_drain_half_pump1 (t : ℝ) : Prop :=
  ∃ rate1 rate2 : ℝ, 
    rate1 = 1 / (2 * t) ∧
    rate2 = 1 / 1.25 ∧
    rate1 + rate2 = 2

-- Equivalent Proof Problem
theorem pump1_half_drain_time (t : ℝ) : time_to_drain_half_pump1 t → t = 5 / 12 := sorry

end pump1_half_drain_time_l164_164648


namespace mean_score_l164_164565

theorem mean_score (mu sigma : ℝ) 
  (h1 : 86 = mu - 7 * sigma) 
  (h2 : 90 = mu + 3 * sigma) :
  mu = 88.8 :=
by
  -- skipping the proof
  sorry

end mean_score_l164_164565


namespace max_area_triangle_l164_164623

noncomputable def max_area (QA QB QC BC : ℝ) : ℝ :=
  1 / 2 * ((QA^2 + QB^2 - QC^2) / (2 * BC) + 3) * BC

theorem max_area_triangle (QA QB QC BC : ℝ) (hQA : QA = 3) (hQB : QB = 4) (hQC : QC = 5) (hBC : BC = 6) :
  max_area QA QB QC BC = 19 := by
  sorry

end max_area_triangle_l164_164623


namespace unique_function_solution_l164_164284

theorem unique_function_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 + f y) = y + f x^2) → (∀ x : ℝ, f x = x) :=
by
  sorry

end unique_function_solution_l164_164284


namespace carl_garden_area_l164_164550

theorem carl_garden_area 
  (total_posts : Nat)
  (length_post_distance : Nat)
  (corner_posts : Nat)
  (longer_side_multiplier : Nat)
  (posts_per_shorter_side : Nat)
  (posts_per_longer_side : Nat)
  (shorter_side_distance : Nat)
  (longer_side_distance : Nat) :
  total_posts = 24 →
  length_post_distance = 5 →
  corner_posts = 4 →
  longer_side_multiplier = 2 →
  posts_per_shorter_side = (24 + 4) / 6 →
  posts_per_longer_side = (24 + 4) / 6 * 2 →
  shorter_side_distance = (posts_per_shorter_side - 1) * length_post_distance →
  longer_side_distance = (posts_per_longer_side - 1) * length_post_distance →
  shorter_side_distance * longer_side_distance = 900 :=
by
  intros
  sorry

end carl_garden_area_l164_164550


namespace isabella_original_hair_length_l164_164508

-- Define conditions from the problem
def isabella_current_hair_length : ℕ := 9
def hair_cut_length : ℕ := 9

-- The proof problem to show original hair length equals 18 inches
theorem isabella_original_hair_length 
  (hc : isabella_current_hair_length = 9)
  (ht : hair_cut_length = 9) : 
  isabella_current_hhair_length + hair_cut_length = 18 := 
sorry

end isabella_original_hair_length_l164_164508


namespace math_problem_l164_164870

-- Define the conditions
def a := -6
def b := 2
def c := 1 / 3
def d := 3 / 4
def e := 12
def f := -3

-- Statement of the problem
theorem math_problem : a / b + (c - d) * e + f^2 = 1 :=
by
  sorry

end math_problem_l164_164870


namespace max_discount_benefit_l164_164046

theorem max_discount_benefit {S X : ℕ} (P : ℕ → Prop) :
  S = 1000 →
  X = 99 →
  (∀ s1 s2 s3 s4 : ℕ, s1 ≥ s2 ∧ s2 ≥ s3 ∧ s3 ≥ s4 ∧ s4 ≥ X ∧ s1 + s2 + s3 + s4 = S →
  ∃ N : ℕ, P N ∧ N = 504) := 
by
  intros hS hX
  sorry

end max_discount_benefit_l164_164046


namespace anne_total_bottle_caps_l164_164680

def initial_bottle_caps_anne : ℕ := 10
def found_bottle_caps_anne : ℕ := 5

theorem anne_total_bottle_caps : initial_bottle_caps_anne + found_bottle_caps_anne = 15 := 
by
  sorry

end anne_total_bottle_caps_l164_164680


namespace ellipse_centroid_locus_l164_164657

noncomputable def ellipse_equation (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1
noncomputable def centroid_locus (x y : ℝ) : Prop := (9 * x^2) / 4 + 3 * y^2 = 1 ∧ y ≠ 0

theorem ellipse_centroid_locus (x y : ℝ) (h : ellipse_equation x y) : centroid_locus (x / 3) (y / 3) :=
  sorry

end ellipse_centroid_locus_l164_164657


namespace Jasmine_shoe_size_l164_164426

theorem Jasmine_shoe_size (J A : ℕ) (h1 : A = 2 * J) (h2 : J + A = 21) : J = 7 :=
by 
  sorry

end Jasmine_shoe_size_l164_164426


namespace train_speed_l164_164750

/-- 
Theorem: Given the length of the train L = 1200 meters and the time T = 30 seconds, the speed of the train S is 40 meters per second.
-/
theorem train_speed (L : ℕ) (T : ℕ) (hL : L = 1200) (hT : T = 30) : L / T = 40 := by
  sorry

end train_speed_l164_164750


namespace solution_of_system_l164_164863

noncomputable def system_of_equations (x y : ℝ) :=
  x = 1.12 * y + 52.8 ∧ x = y + 50

theorem solution_of_system : 
  ∃ (x y : ℝ), system_of_equations x y ∧ y = -23.33 ∧ x = 26.67 :=
by
  sorry

end solution_of_system_l164_164863


namespace vector_problem_l164_164544

open Real

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  (v.1 ^ 2 + v.2 ^ 2) ^ (1 / 2)

variables (a b : ℝ × ℝ)
variables (h1 : a ≠ (0, 0)) (h2 : b ≠ (0, 0))
variables (h3 : a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0)
variables (h4 : 2 * magnitude a = magnitude b) (h5 : magnitude b =2)

theorem vector_problem : magnitude (2 * a.1 - b.1, 2 * a.2 - b.2) = 2 :=
sorry

end vector_problem_l164_164544


namespace perfect_square_trinomial_l164_164581

theorem perfect_square_trinomial (k : ℝ) : 
  ∃ a : ℝ, (x^2 - k*x + 1 = (x + a)^2) → (k = 2 ∨ k = -2) :=
by
  sorry

end perfect_square_trinomial_l164_164581


namespace sum_of_distinct_prime_factors_of_2016_l164_164330

-- Define 2016 and the sum of its distinct prime factors
def n : ℕ := 2016
def sumOfDistinctPrimeFactors (n : ℕ) : ℕ :=
  if n = 2016 then 2 + 3 + 7 else 0  -- Capture the problem-specific condition

-- The main theorem to prove the sum of the distinct prime factors of 2016 is 12
theorem sum_of_distinct_prime_factors_of_2016 :
  sumOfDistinctPrimeFactors 2016 = 12 :=
by
  -- Since this is beyond the obvious steps, we use a sorry here
  sorry

end sum_of_distinct_prime_factors_of_2016_l164_164330


namespace maria_savings_l164_164395

variable (S : ℝ) -- Define S as a real number (amount saved initially)

-- Conditions
def bike_cost : ℝ := 600
def additional_money : ℝ := 250 + 230

-- Theorem statement
theorem maria_savings : S + additional_money = bike_cost → S = 120 :=
by
  intro h -- Assume the hypothesis (condition)
  sorry -- Proof will go here

end maria_savings_l164_164395


namespace first_shaded_square_each_column_l164_164028

/-- A rectangular board with 10 columns, numbered starting from 
    1 to the nth square left-to-right and top-to-bottom. The student shades squares 
    that are perfect squares. Prove that the first shaded square ensuring there's at least 
    one shaded square in each of the 10 columns is 400. -/
theorem first_shaded_square_each_column : 
  (∃ n, (∀ k, 1 ≤ k ∧ k ≤ 10 → ∃ m, m^2 ≡ k [MOD 10] ∧ m^2 ≤ n) ∧ n = 400) :=
sorry

end first_shaded_square_each_column_l164_164028


namespace sectionBSeats_l164_164596

-- Definitions from the conditions
def seatsIn60SeatSubsectionA : Nat := 60
def subsectionsIn80SeatA : Nat := 3
def seatsPer80SeatSubsectionA : Nat := 80
def extraSeatsInSectionB : Nat := 20

-- Total seats in 80-seat subsections of Section A
def totalSeatsIn80SeatSubsections : Nat := subsectionsIn80SeatA * seatsPer80SeatSubsectionA

-- Total seats in Section A
def totalSeatsInSectionA : Nat := totalSeatsIn80SeatSubsections + seatsIn60SeatSubsectionA

-- Total seats in Section B
def totalSeatsInSectionB : Nat := 3 * totalSeatsInSectionA + extraSeatsInSectionB

-- The statement to prove
theorem sectionBSeats : totalSeatsInSectionB = 920 := by
  sorry

end sectionBSeats_l164_164596


namespace equation_represents_lines_and_point_l164_164317

theorem equation_represents_lines_and_point:
    (∀ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 0 → (x = 1 ∧ y = -2)) ∧
    (∀ x y : ℝ, x^2 - y^2 = 0 → (x = y) ∨ (x = -y)) → 
    (∀ x y : ℝ, ((x - 1)^2 + (y + 2)^2) * (x^2 - y^2) = 0 → 
    ((x = 1 ∧ y = -2) ∨ (x + y = 0) ∨ (x - y = 0))) :=
by
  intros h1 h2 h3
  sorry

end equation_represents_lines_and_point_l164_164317


namespace sequence_ab_sum_l164_164457

theorem sequence_ab_sum (s a b : ℝ) (h1 : 16 * s = 4) (h2 : 1024 * s = a) (h3 : a * s = b) : a + b = 320 := by
  sorry

end sequence_ab_sum_l164_164457


namespace beautiful_point_coordinates_l164_164874

-- Define a "beautiful point"
def is_beautiful_point (P : ℝ × ℝ) : Prop :=
  P.1 + P.2 = P.1 * P.2

theorem beautiful_point_coordinates (M : ℝ × ℝ) : 
  is_beautiful_point M ∧ abs M.1 = 2 → 
  (M = (2, 2) ∨ M = (-2, 2/3)) :=
by sorry

end beautiful_point_coordinates_l164_164874


namespace point_reflection_example_l164_164985

def point := ℝ × ℝ

def reflect_x_axis (p : point) : point := (p.1, -p.2)

theorem point_reflection_example : reflect_x_axis (1, -2) = (1, 2) := sorry

end point_reflection_example_l164_164985


namespace cryptarithm_solution_exists_l164_164076

theorem cryptarithm_solution_exists :
  ∃ (L E S O : ℕ), L ≠ E ∧ L ≠ S ∧ L ≠ O ∧ E ≠ S ∧ E ≠ O ∧ S ≠ O ∧
  (L < 10) ∧ (E < 10) ∧ (S < 10) ∧ (O < 10) ∧
  (1000 * O + 100 * S + 10 * E + L) +
  (100 * S + 10 * E + L) +
  (10 * E + L) +
  L = 10034 ∧
  ((L = 6 ∧ E = 7 ∧ S = 4 ∧ O = 9) ∨
   (L = 6 ∧ E = 7 ∧ S = 9 ∧ O = 8)) :=
by
  -- The proof is omitted here.
  sorry

end cryptarithm_solution_exists_l164_164076


namespace sector_area_correct_l164_164656

noncomputable def sector_area (θ r : ℝ) : ℝ :=
  (θ / (2 * Real.pi)) * (Real.pi * r^2)

theorem sector_area_correct : 
  sector_area (Real.pi / 3) 3 = (3 / 2) * Real.pi :=
by
  sorry

end sector_area_correct_l164_164656


namespace cos_300_eq_half_l164_164504

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l164_164504


namespace A_infinite_l164_164202

noncomputable def f : ℝ → ℝ := sorry

def A : Set ℝ := { a : ℝ | f a > a ^ 2 }

theorem A_infinite
  (h_f_def : ∀ x : ℝ, ∃ y : ℝ, y = f x)
  (h_inequality: ∀ x : ℝ, (f x) ^ 2 ≤ 2 * x ^ 2 * f (x / 2))
  (h_A_nonempty : A ≠ ∅) :
  Set.Infinite A := 
sorry

end A_infinite_l164_164202


namespace May4th_Sunday_l164_164710

theorem May4th_Sunday (x : ℕ) (h_sum : x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 80) : 
  (4 % 7) = 0 :=
by
  sorry

end May4th_Sunday_l164_164710


namespace frank_initial_candy_l164_164375

theorem frank_initial_candy (n : ℕ) (h1 : n = 21) (h2 : 2 > 0) :
  2 * n = 42 :=
by
  --* Use the hypotheses to establish the required proof
  sorry

end frank_initial_candy_l164_164375


namespace square_area_PS_l164_164275

noncomputable def area_of_square_on_PS : ℕ :=
  sorry

theorem square_area_PS (PQ QR RS PR PS : ℝ)
  (h1 : PQ ^ 2 = 25)
  (h2 : QR ^ 2 = 49)
  (h3 : RS ^ 2 = 64)
  (h4 : PQ^2 + QR^2 = PR^2)
  (h5 : PR^2 + RS^2 = PS^2) :
  PS^2 = 138 :=
by
  -- proof skipping
  sorry


end square_area_PS_l164_164275


namespace boxes_neither_pens_nor_pencils_l164_164089

def total_boxes : ℕ := 10
def pencil_boxes : ℕ := 6
def pen_boxes : ℕ := 3
def both_boxes : ℕ := 2

theorem boxes_neither_pens_nor_pencils : (total_boxes - (pencil_boxes + pen_boxes - both_boxes)) = 3 :=
by
  sorry

end boxes_neither_pens_nor_pencils_l164_164089


namespace parabola_y_axis_intersection_l164_164852

theorem parabola_y_axis_intersection:
  (∀ x y : ℝ, y = -2 * (x - 1)^2 - 3 → x = 0 → y = -5) :=
by
  intros x y h_eq h_x
  sorry

end parabola_y_axis_intersection_l164_164852


namespace integer_part_divisible_by_112_l164_164920

def is_odd (n : ℕ) : Prop := n % 2 = 1
def not_divisible_by_3 (n : ℕ) : Prop := n % 3 ≠ 0

theorem integer_part_divisible_by_112
  (m : ℕ) (hm_pos : 0 < m) (hm_odd : is_odd m) (hm_not_div3 : not_divisible_by_3 m) :
  ∃ n : ℤ, 112 * n = 4^m - (2 + Real.sqrt 2)^m - (2 - Real.sqrt 2)^m :=
by
  sorry

end integer_part_divisible_by_112_l164_164920


namespace total_number_of_birds_l164_164013

def geese : ℕ := 58
def ducks : ℕ := 37
def swans : ℕ := 42

theorem total_number_of_birds : geese + ducks + swans = 137 := by
  sorry

end total_number_of_birds_l164_164013


namespace covered_digits_l164_164403

def four_digit_int (n : ℕ) : Prop := n ≥ 1000 ∧ n < 10000

theorem covered_digits (a b c : ℕ) (n1 n2 n3 : ℕ) :
  four_digit_int n1 → four_digit_int n2 → four_digit_int n3 →
  n1 + n2 + n3 = 10126 →
  (n1 % 10 = 3 ∧ n2 % 10 = 7 ∧ n3 % 10 = 6) →
  (n1 / 10 % 10 = 4 ∧ n2 / 10 % 10 = a ∧ n3 / 10 % 10 = 2) →
  (n1 / 100 % 10 = 2 ∧ n2 / 100 % 10 = 1 ∧ n3 / 100 % 10 = c) →
  (n1 / 1000 = 1 ∧ n2 / 1000 = 2 ∧ n3 / 1000 = b) →
  (a = 5 ∧ b = 6 ∧ c = 7) := 
sorry

end covered_digits_l164_164403


namespace value_of_x_squared_plus_y_squared_l164_164002

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x^2 = 8 * x + y) (h2 : y^2 = x + 8 * y) (h3 : x ≠ y) : 
  x^2 + y^2 = 63 := sorry

end value_of_x_squared_plus_y_squared_l164_164002


namespace minimize_quadratic_l164_164897

theorem minimize_quadratic : ∃ x : ℝ, ∀ y : ℝ, (x^2 - 12*x + 28 ≤ y^2 - 12*y + 28) :=
by
  use 6
  sorry

end minimize_quadratic_l164_164897


namespace find_box_length_l164_164287

theorem find_box_length (width depth : ℕ) (num_cubes : ℕ) (cube_side length : ℕ) 
  (h1 : width = 20)
  (h2 : depth = 10)
  (h3 : num_cubes = 56)
  (h4 : cube_side = 10)
  (h5 : length * width * depth = num_cubes * cube_side * cube_side * cube_side) :
  length = 280 :=
sorry

end find_box_length_l164_164287


namespace octagon_diagonals_l164_164937

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end octagon_diagonals_l164_164937


namespace min_value_fraction_l164_164329

theorem min_value_fraction 
  (a_n : ℕ → ℕ)
  (S_n : ℕ → ℕ)
  (a1 a3 a13 : ℕ)
  (d : ℕ) 
  (h1 : ∀ n, a_n n = a1 + (n - 1) * d)
  (h2 : d ≠ 0)
  (h3 : a1 = 1)
  (h4 : a3 ^ 2 = a1 * a13)
  (h5 : ∀ n, S_n n = n * (a1 + a_n n) / 2) :
  ∃ n, (2 * S_n n + 16) / (a_n n + 3) = 4 := 
sorry

end min_value_fraction_l164_164329


namespace equal_roots_h_l164_164382

theorem equal_roots_h (h : ℝ) : (∀ x : ℝ, 3 * x^2 - 4 * x + (h / 3) = 0) -> h = 4 :=
by 
  sorry

end equal_roots_h_l164_164382


namespace find_smaller_number_l164_164052

theorem find_smaller_number (x y : ℕ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : x = 18 :=
by
  sorry

end find_smaller_number_l164_164052


namespace determine_positions_l164_164196

-- Defining the participants
inductive Participant
| Olya
| Oleg
| Pasha

open Participant

-- Defining the possible places
inductive Place
| First
| Second
| Third

open Place

-- Define the conditions
def condition1 (pos : Participant → Place) : Prop := 
  pos Olya = First ∨ pos Oleg = First ∨ pos Pasha = First

def condition2 (pos : Participant → Place) : Prop :=
  (pos Olya = First ∧ pos Olya = Second ∧ pos Olya = Third) ∨
  (pos Oleg = First ∧ pos Oleg = Second ∧ pos Oleg = Third) ∨
  (pos Pasha = First ∧ pos Pasha = Second ∧ pos Pasha = Third)

def condition3 (pos : Participant → Place) : Prop :=
  ∀ p, pos p ≠ First ∧ pos p ≠ Second ∧ pos p ≠ Third

def condition4 (pos : Participant → Place) : Prop :=
  (pos Olya = First → (pos Oleg = First ∨ pos Pasha = First)) ∧
  (pos Oleg = First → pos Olya ≠ First) ∧
  (pos Pasha = First → (pos Oleg = First ∨ pos Olya = First))

def always_true_or_false : Prop :=
  (∀ p, p = Olya ∨ p = Oleg ∨ p = Pasha )

-- Main theorem
theorem determine_positions (pos : Participant → Place) :
  condition1 pos ∧ condition2 pos ∧ condition3 pos ∧ condition4 pos ∧ always_true_or_false →
  pos Oleg = First ∧ pos Pasha = Second ∧ pos Olya = Third := 
by
  sorry

end determine_positions_l164_164196


namespace standard_equation_of_parabola_l164_164128

theorem standard_equation_of_parabola (F : ℝ × ℝ) (hF : F.1 + 2 * F.2 + 3 = 0) :
  (∃ y₀: ℝ, y₀ < 0 ∧ F = (0, y₀) ∧ ∀ x: ℝ, x ^ 2 = - 6 * y₀ * x) ∨
  (∃ x₀: ℝ, x₀ < 0 ∧ F = (x₀, 0) ∧ ∀ y: ℝ, y ^ 2 = - 12 * x₀ * y) :=
sorry

end standard_equation_of_parabola_l164_164128


namespace present_population_l164_164662

theorem present_population (P : ℝ)
  (h1 : P + 0.10 * P = 242) :
  P = 220 := 
sorry

end present_population_l164_164662


namespace birds_more_than_nests_l164_164164

theorem birds_more_than_nests : 
  let birds := 6 
  let nests := 3 
  (birds - nests) = 3 := 
by 
  sorry

end birds_more_than_nests_l164_164164


namespace mean_value_of_interior_angles_pentagon_l164_164176

def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

theorem mean_value_of_interior_angles_pentagon :
  sum_of_interior_angles 5 / 5 = 108 :=
by
  sorry

end mean_value_of_interior_angles_pentagon_l164_164176


namespace tangent_product_20_40_60_80_l164_164148

theorem tangent_product_20_40_60_80 :
  Real.tan (20 * Real.pi / 180) * Real.tan (40 * Real.pi / 180) * Real.tan (60 * Real.pi / 180) * Real.tan (80 * Real.pi / 180) = 3 :=
by
  sorry

end tangent_product_20_40_60_80_l164_164148


namespace problem_l164_164439

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 8

theorem problem 
  (a b c : ℝ) 
  (h : f a b c (-2) = 10) 
  : f a b c 2 = 6 :=
by
  sorry

end problem_l164_164439


namespace vector_magnitude_parallel_l164_164574

/-- Given two plane vectors a = (1, 2) and b = (-2, y),
if a is parallel to b, then |2a - b| = 4 * sqrt 5. -/
theorem vector_magnitude_parallel (y : ℝ) 
  (h_parallel : (1 : ℝ) / (-2 : ℝ) = (2 : ℝ) / y) : 
  ‖2 • (1, 2) - (-2, y)‖ = 4 * Real.sqrt 5 := 
by
  sorry

end vector_magnitude_parallel_l164_164574


namespace hash_op_8_4_l164_164127

def hash_op (a b : ℕ) : ℕ := a + a / b - 2

theorem hash_op_8_4 : hash_op 8 4 = 8 := 
by 
  -- The proof is left as an exercise, indicated by sorry.
  sorry

end hash_op_8_4_l164_164127


namespace pauls_plumbing_hourly_charge_l164_164764

theorem pauls_plumbing_hourly_charge :
  ∀ P : ℕ,
  (55 + 4 * P = 75 + 4 * 30) → 
  P = 35 :=
by
  intros P h
  sorry

end pauls_plumbing_hourly_charge_l164_164764


namespace contrapositive_of_not_p_implies_q_l164_164821

variable (p q : Prop)

theorem contrapositive_of_not_p_implies_q :
  (¬p → q) → (¬q → p) := by
  sorry

end contrapositive_of_not_p_implies_q_l164_164821


namespace max_sum_of_prices_l164_164837

theorem max_sum_of_prices (R P : ℝ) 
  (h1 : 4 * R + 5 * P ≥ 27) 
  (h2 : 6 * R + 3 * P ≤ 27) : 
  3 * R + 4 * P ≤ 36 :=
by 
  sorry

end max_sum_of_prices_l164_164837


namespace rent_3600_yields_88_optimal_rent_is_4100_and_max_revenue_is_304200_l164_164549

/-
Via conditions:
1. The rental company owns 100 cars.
2. When the monthly rent for each car is set at 3000 yuan, all cars can be rented out.
3. For every 50 yuan increase in the monthly rent per car, there will be one more car that is not rented out.
4. The maintenance cost for each rented car is 200 yuan per month.
-/

noncomputable def num_rented_cars (rent_per_car : ℕ) : ℕ :=
  if rent_per_car < 3000 then 100 else max 0 (100 - (rent_per_car - 3000) / 50)

noncomputable def monthly_revenue (rent_per_car : ℕ) : ℕ :=
  let cars_rented := num_rented_cars rent_per_car
  let maintenance_cost := 200 * cars_rented
  (rent_per_car - maintenance_cost) * cars_rented

theorem rent_3600_yields_88 : num_rented_cars 3600 = 88 :=
  sorry

theorem optimal_rent_is_4100_and_max_revenue_is_304200 :
  ∃ rent_per_car, rent_per_car = 4100 ∧ monthly_revenue rent_per_car = 304200 :=
  sorry

end rent_3600_yields_88_optimal_rent_is_4100_and_max_revenue_is_304200_l164_164549


namespace cost_of_four_dozen_l164_164694

-- Defining the conditions
def cost_of_three_dozen (cost : ℚ) : Prop :=
  cost = 25.20

-- The theorem to prove the cost of four dozen apples at the same rate
theorem cost_of_four_dozen (cost : ℚ) :
  cost_of_three_dozen cost →
  (4 * (cost / 3) = 33.60) :=
by
  sorry

end cost_of_four_dozen_l164_164694


namespace rectangle_breadth_approx_1_1_l164_164523

theorem rectangle_breadth_approx_1_1 (s b : ℝ) (h1 : 4 * s = 2 * (16 + b))
  (h2 : abs ((π * s / 2) + s - 21.99) < 0.01) : abs (b - 1.1) < 0.01 :=
sorry

end rectangle_breadth_approx_1_1_l164_164523


namespace sugar_percentage_l164_164844

theorem sugar_percentage (S : ℝ) (P : ℝ) : 
  (3 / 4 * S * 0.10 + (1 / 4) * S * P / 100 = S * 0.20) → 
  P = 50 := 
by 
  intro h
  sorry

end sugar_percentage_l164_164844


namespace train_stoppages_l164_164594

variables (sA sA' sB sB' sC sC' : ℝ)
variables (x y z : ℝ)

-- Conditions
def conditions : Prop :=
  sA = 80 ∧ sA' = 60 ∧
  sB = 100 ∧ sB' = 75 ∧
  sC = 120 ∧ sC' = 90

-- Goal that we need to prove
def goal : Prop :=
  x = 15 ∧ y = 15 ∧ z = 15

-- Main statement
theorem train_stoppages : conditions sA sA' sB sB' sC sC' → goal x y z :=
by
  sorry

end train_stoppages_l164_164594


namespace remainder_of_power_sums_modulo_seven_l164_164865

theorem remainder_of_power_sums_modulo_seven :
  (9^6 + 8^8 + 7^9) % 7 = 2 := 
by 
  sorry

end remainder_of_power_sums_modulo_seven_l164_164865


namespace sum_of_areas_B_D_l164_164475

theorem sum_of_areas_B_D (area_large_square : ℝ) (area_small_square : ℝ) (B D : ℝ) 
  (h1 : area_large_square = 9) 
  (h2 : area_small_square = 1)
  (h3 : B + D = 4) : 
  B + D = 4 := 
by
  sorry

end sum_of_areas_B_D_l164_164475


namespace compare_a_b_c_l164_164604

noncomputable def a := 0.1 * Real.exp 0.1
def b := 1 / 9
noncomputable def c := -Real.log 0.9

theorem compare_a_b_c : c < a ∧ a < b := by
  have h_c_b : c < b := sorry
  have h_a_b : a < b := sorry
  have h_c_a : c < a := sorry
  exact ⟨h_c_a, h_a_b⟩

end compare_a_b_c_l164_164604


namespace problem_solution_l164_164342

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.rpow 3 (1 / 3)
noncomputable def c : ℝ := Real.log 2 / Real.log 3

theorem problem_solution : c < a ∧ a < b := 
by
  sorry

end problem_solution_l164_164342


namespace exam_combinations_l164_164211

/-- In the "$3+1+2$" examination plan in Hubei Province, 2021,
there are three compulsory subjects: Chinese, Mathematics, and English.
Candidates must choose one subject from Physics and History.
Candidates must choose two subjects from Chemistry, Biology, Ideological and Political Education, and Geography.
Prove that the total number of different combinations of examination subjects is 12.
-/
theorem exam_combinations : exists n : ℕ, n = 12 :=
by
  have compulsory_choice := 1
  have physics_history_choice := 2
  have remaining_subjects_choice := Nat.choose 4 2
  exact Exists.intro (compulsory_choice * physics_history_choice * remaining_subjects_choice) sorry

end exam_combinations_l164_164211


namespace probability_no_three_consecutive_1s_l164_164918

theorem probability_no_three_consecutive_1s (m n : ℕ) (h_relatively_prime : Nat.gcd m n = 1) (h_eq : 2^12 = 4096) :
  let b₁ := 2
  let b₂ := 4
  let b₃ := 7
  let b₄ := b₃ + b₂ + b₁
  let b₅ := b₄ + b₃ + b₂
  let b₆ := b₅ + b₄ + b₃
  let b₇ := b₆ + b₅ + b₄
  let b₈ := b₇ + b₆ + b₅
  let b₉ := b₈ + b₇ + b₆
  let b₁₀ := b₉ + b₈ + b₇
  let b₁₁ := b₁₀ + b₉ + b₈
  let b₁₂ := b₁₁ + b₁₀ + b₉
  (m = 1705 ∧ n = 4096 ∧ b₁₂ = m) →
  m + n = 5801 := 
by
  intros
  sorry

end probability_no_three_consecutive_1s_l164_164918


namespace identity_problem_l164_164029

theorem identity_problem
  (a b : ℝ)
  (h₁ : a * b = 2)
  (h₂ : a + b = 3) :
  (a - b)^2 = 1 :=
by
  sorry

end identity_problem_l164_164029


namespace siblings_ate_two_slices_l164_164760

-- Let slices_after_dinner be the number of slices left after eating one-fourth of 16 slices
def slices_after_dinner : ℕ := 16 - 16 / 4

-- Let slices_after_yves be the number of slices left after Yves ate one-fourth of the remaining pizza
def slices_after_yves : ℕ := slices_after_dinner - slices_after_dinner / 4

-- Let slices_left be the number of slices left after Yves's siblings ate some slices
def slices_left : ℕ := 5

-- Let slices_eaten_by_siblings be the number of slices eaten by Yves's siblings
def slices_eaten_by_siblings : ℕ := slices_after_yves - slices_left

-- Since there are two siblings, each ate half of the slices_eaten_by_siblings
def slices_per_sibling : ℕ := slices_eaten_by_siblings / 2

-- The theorem stating that each sibling ate 2 slices
theorem siblings_ate_two_slices : slices_per_sibling = 2 :=
by
  -- Definition of slices_after_dinner
  have h1 : slices_after_dinner = 12 := by sorry
  -- Definition of slices_after_yves
  have h2 : slices_after_yves = 9 := by sorry
  -- Definition of slices_eaten_by_siblings
  have h3 : slices_eaten_by_siblings = 4 := by sorry
  -- Final assertion of slices_per_sibling
  have h4 : slices_per_sibling = 2 := by sorry
  exact h4

end siblings_ate_two_slices_l164_164760


namespace no_real_roots_l164_164833

def op (m n : ℝ) : ℝ := n^2 - m * n + 1

theorem no_real_roots (x : ℝ) : op 1 x = 0 → ¬ ∃ x : ℝ, x^2 - x + 1 = 0 :=
by {
  sorry
}

end no_real_roots_l164_164833


namespace race_problem_l164_164244

theorem race_problem (a_speed b_speed : ℕ) (A B : ℕ) (finish_dist : ℕ)
  (h1 : finish_dist = 3000)
  (h2 : A = finish_dist - 500)
  (h3 : B = finish_dist - 600)
  (h4 : A / a_speed = B / b_speed)
  (h5 : a_speed / b_speed = 25 / 24) :
  B - ((500 * b_speed) / a_speed) = 120 :=
by
  sorry

end race_problem_l164_164244


namespace usage_difference_correct_l164_164392

def computerUsageLastWeek : ℕ := 91

def computerUsageThisWeek : ℕ :=
  let first4days := 4 * 8
  let last3days := 3 * 10
  first4days + last3days

def computerUsageFollowingWeek : ℕ :=
  let weekdays := 5 * (5 + 3)
  let weekends := 2 * 12
  weekdays + weekends

def differenceThisWeek : ℕ := computerUsageLastWeek - computerUsageThisWeek
def differenceFollowingWeek : ℕ := computerUsageLastWeek - computerUsageFollowingWeek

theorem usage_difference_correct :
  differenceThisWeek = 29 ∧ differenceFollowingWeek = 27 := by
  sorry

end usage_difference_correct_l164_164392


namespace largest_gold_coins_l164_164528

theorem largest_gold_coins (k : ℤ) (h1 : 13 * k + 3 < 100) : 91 ≤ 13 * k + 3 :=
by
  sorry

end largest_gold_coins_l164_164528


namespace Diego_total_stamp_cost_l164_164309

theorem Diego_total_stamp_cost :
  let price_brazil_colombia := 0.07
  let price_peru := 0.05
  let num_brazil_50s := 6
  let num_brazil_60s := 9
  let num_peru_50s := 8
  let num_peru_60s := 5
  let num_colombia_50s := 7
  let num_colombia_60s := 6
  let total_brazil := num_brazil_50s + num_brazil_60s
  let total_peru := num_peru_50s + num_peru_60s
  let total_colombia := num_colombia_50s + num_colombia_60s
  let cost_brazil := total_brazil * price_brazil_colombia
  let cost_peru := total_peru * price_peru
  let cost_colombia := total_colombia * price_brazil_colombia
  cost_brazil + cost_peru + cost_colombia = 2.61 :=
by
  sorry

end Diego_total_stamp_cost_l164_164309


namespace initial_invitation_count_l164_164311

def people_invited (didnt_show : ℕ) (num_tables : ℕ) (people_per_table : ℕ) : ℕ :=
  didnt_show + num_tables * people_per_table

theorem initial_invitation_count (didnt_show : ℕ) (num_tables : ℕ) (people_per_table : ℕ)
    (h1 : didnt_show = 35) (h2 : num_tables = 5) (h3 : people_per_table = 2) :
  people_invited didnt_show num_tables people_per_table = 45 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end initial_invitation_count_l164_164311


namespace find_k_l164_164896

theorem find_k (k : ℕ) : (∃ n : ℕ, 2^k + 8*k + 5 = n^2) ↔ k = 2 := by
  sorry

end find_k_l164_164896


namespace ratio_ashley_mary_l164_164384

-- Definitions based on conditions
def sum_ages (A M : ℕ) := A + M = 22
def ashley_age (A : ℕ) := A = 8

-- Theorem stating the ratio of Ashley's age to Mary's age
theorem ratio_ashley_mary (A M : ℕ) 
  (h1 : sum_ages A M)
  (h2 : ashley_age A) : 
  (A : ℚ) / (M : ℚ) = 4 / 7 :=
by
  -- Skipping the proof as specified
  sorry

end ratio_ashley_mary_l164_164384


namespace initial_ratio_of_milk_to_water_l164_164012

theorem initial_ratio_of_milk_to_water 
  (M W : ℕ) 
  (h1 : M + 10 + W = 30)
  (h2 : (M + 10) * 2 = W * 5)
  (h3 : M + W = 20) : 
  M = 11 ∧ W = 9 := 
by 
  sorry

end initial_ratio_of_milk_to_water_l164_164012


namespace roots_of_cubic_l164_164992

theorem roots_of_cubic (a b c d r s t : ℝ) 
  (h1 : r + s + t = -b / a)
  (h2 : r * s + r * t + s * t = c / a)
  (h3 : r * s * t = -d / a) :
  1 / (r ^ 2) + 1 / (s ^ 2) + 1 / (t ^ 2) = (c ^ 2 - 2 * b * d) / (d ^ 2) := 
sorry

end roots_of_cubic_l164_164992


namespace base_five_product_l164_164124

open Nat

/-- Definition of the base 5 representation of 131 and 21 --/
def n131 := 1 * 5^2 + 3 * 5^1 + 1 * 5^0
def n21 := 2 * 5^1 + 1 * 5^0

/-- Definition of the expected result in base 5 --/
def expected_result := 3 * 5^3 + 2 * 5^2 + 5 * 5^1 + 1 * 5^0

/-- Claim to prove that the product of 131_5 and 21_5 equals 3251_5 --/
theorem base_five_product : n131 * n21 = expected_result := by sorry

end base_five_product_l164_164124


namespace value_of_m_l164_164540

theorem value_of_m (m : ℤ) : (∃ (f : ℤ → ℤ), ∀ x : ℤ, x^2 + m * x + 16 = (f x)^2) ↔ (m = 8 ∨ m = -8) := 
by
  sorry

end value_of_m_l164_164540


namespace wine_ages_l164_164854

-- Define the ages of the wines as variables
variable (C F T B Bo M : ℝ)

-- Define the six conditions
axiom h1 : F = 3 * C
axiom h2 : C = 4 * T
axiom h3 : B = (1 / 2) * T
axiom h4 : Bo = 2 * F
axiom h5 : M^2 = Bo
axiom h6 : C = 40

-- Prove the ages of the wines 
theorem wine_ages : 
  F = 120 ∧ 
  T = 10 ∧ 
  B = 5 ∧ 
  Bo = 240 ∧ 
  M = Real.sqrt 240 :=
by
  sorry

end wine_ages_l164_164854


namespace container_unoccupied_volume_l164_164320

noncomputable def unoccupied_volume (side_length_container : ℝ) (side_length_ice : ℝ) (num_ice_cubes : ℕ) : ℝ :=
  let volume_container := side_length_container ^ 3
  let volume_water := (3 / 4) * volume_container
  let volume_ice := num_ice_cubes / 2 * side_length_ice ^ 3
  volume_container - (volume_water + volume_ice)

theorem container_unoccupied_volume :
  unoccupied_volume 12 1.5 12 = 411.75 :=
by
  sorry

end container_unoccupied_volume_l164_164320


namespace cat_food_insufficient_for_six_days_l164_164436

theorem cat_food_insufficient_for_six_days
  (B S : ℕ)
  (h1 : B > S)
  (h2 : B < 2 * S)
  (h3 : B + 2 * S = 2 * ((B + 2 * S) / 2)) :
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by sorry

end cat_food_insufficient_for_six_days_l164_164436


namespace candles_lit_time_correct_l164_164119

noncomputable def candle_time : String :=
  let initial_length := 1 -- Since the length is uniform, we use 1
  let rateA := initial_length / (6 * 60) -- Rate at which Candle A burns out
  let rateB := initial_length / (8 * 60) -- Rate at which Candle B burns out
  let t := 320 -- The time in minutes that satisfy the condition
  let time_lit := (16 * 60 - t) / 60 -- Convert minutes to hours
  if time_lit = 10 + 40 / 60 then "10:40 AM" else "Unknown"

theorem candles_lit_time_correct :
  candle_time = "10:40 AM" := 
by
  sorry

end candles_lit_time_correct_l164_164119


namespace purple_coincide_pairs_l164_164627

theorem purple_coincide_pairs
    (yellow_triangles_upper : ℕ)
    (yellow_triangles_lower : ℕ)
    (green_triangles_upper : ℕ)
    (green_triangles_lower : ℕ)
    (purple_triangles_upper : ℕ)
    (purple_triangles_lower : ℕ)
    (yellow_coincide_pairs : ℕ)
    (green_coincide_pairs : ℕ)
    (yellow_purple_pairs : ℕ) :
    yellow_triangles_upper = 4 →
    yellow_triangles_lower = 4 →
    green_triangles_upper = 6 →
    green_triangles_lower = 6 →
    purple_triangles_upper = 10 →
    purple_triangles_lower = 10 →
    yellow_coincide_pairs = 3 →
    green_coincide_pairs = 4 →
    yellow_purple_pairs = 3 →
    (∃ purple_coincide_pairs : ℕ, purple_coincide_pairs = 5) :=
by sorry

end purple_coincide_pairs_l164_164627


namespace math_problem_l164_164372

-- Define the mixed numbers as fractions
def mixed_3_1_5 := 16 / 5 -- 3 + 1/5 = 16/5
def mixed_4_1_2 := 9 / 2  -- 4 + 1/2 = 9/2
def mixed_2_3_4 := 11 / 4 -- 2 + 3/4 = 11/4
def mixed_1_2_3 := 5 / 3  -- 1 + 2/3 = 5/3

-- Define the main expression
def main_expr := 53 * (mixed_3_1_5 - mixed_4_1_2) / (mixed_2_3_4 + mixed_1_2_3)

-- Define the expected answer in its fractional form
def expected_result := -78 / 5

-- The theorem to prove the main expression equals the expected mixed number
theorem math_problem : main_expr = expected_result :=
by sorry

end math_problem_l164_164372


namespace sara_total_payment_l164_164332

structure DecorationCosts where
  balloons: ℝ
  tablecloths: ℝ
  streamers: ℝ
  banners: ℝ
  confetti: ℝ
  change_received: ℝ

noncomputable def total_cost (c : DecorationCosts) : ℝ :=
  c.balloons + c.tablecloths + c.streamers + c.banners + c.confetti

noncomputable def amount_given (c : DecorationCosts) : ℝ :=
  total_cost c + c.change_received

theorem sara_total_payment : 
  ∀ (costs : DecorationCosts), 
    costs = ⟨3.50, 18.25, 9.10, 14.65, 7.40, 6.38⟩ →
    amount_given costs = 59.28 :=
by
  intros
  sorry

end sara_total_payment_l164_164332


namespace other_x_intercept_l164_164886

theorem other_x_intercept (a b c : ℝ) 
  (h_eq : ∀ x, a * x^2 + b * x + c = y) 
  (h_vertex: (5, 10) = ((-b / (2 * a)), (4 * a * 10 / (4 * a)))) 
  (h_intercept : ∃ x, a * x * 0 + b * 0 + c = 0) : ∃ x, x = 10 :=
by
  sorry

end other_x_intercept_l164_164886


namespace fundraiser_goal_eq_750_l164_164687

def bronze_donations := 10 * 25
def silver_donations := 7 * 50
def gold_donations   := 1 * 100
def total_collected  := bronze_donations + silver_donations + gold_donations
def amount_needed    := 50
def total_goal       := total_collected + amount_needed

theorem fundraiser_goal_eq_750 : total_goal = 750 :=
by
  sorry

end fundraiser_goal_eq_750_l164_164687


namespace complex_problem_l164_164947

def is_imaginary_unit (x : ℂ) : Prop := x^2 = -1

theorem complex_problem (a b : ℝ) (i : ℂ) (h1 : (a - 2 * i) / i = (b : ℂ) + i) (h2 : is_imaginary_unit i) :
  a - b = 1 := 
sorry

end complex_problem_l164_164947


namespace inequality_proof_l164_164702

theorem inequality_proof (a b c x y z : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : x ≥ y) (h5 : y ≥ z) (h6 : z > 0) :
  (a^2 * x^2 / ((b * y + c * z) * (b * z + c * y)) + 
   b^2 * y^2 / ((a * x + c * z) * (a * z + c * x)) +
   c^2 * z^2 / ((a * x + b * y) * (a * y + b * x))) ≥ 3 / 4 := 
by
  sorry

end inequality_proof_l164_164702


namespace largest_divisor_of_n_given_n_squared_divisible_by_72_l164_164228

theorem largest_divisor_of_n_given_n_squared_divisible_by_72 (n : ℕ) (h1 : 0 < n) (h2 : 72 ∣ n^2) :
  ∃ q, q = 12 ∧ q ∣ n :=
by
  sorry

end largest_divisor_of_n_given_n_squared_divisible_by_72_l164_164228


namespace factorization_correct_l164_164792

theorem factorization_correct (x : ℝ) : x^2 - 6*x + 9 = (x - 3)^2 :=
by
  sorry

end factorization_correct_l164_164792


namespace sin_14pi_div_3_eq_sqrt3_div_2_l164_164974

theorem sin_14pi_div_3_eq_sqrt3_div_2 : Real.sin (14 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_14pi_div_3_eq_sqrt3_div_2_l164_164974


namespace fraction_difference_l164_164925

theorem fraction_difference (a b : ℝ) : 
  (a / (a + 1)) - (b / (b + 1)) = (a - b) / ((a + 1) * (b + 1)) :=
sorry

end fraction_difference_l164_164925


namespace inequality_for_positive_nums_l164_164282

theorem inequality_for_positive_nums 
    (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
    a^2 / b + c^2 / d ≥ (a + c)^2 / (b + d) :=
by
  sorry

end inequality_for_positive_nums_l164_164282


namespace tony_solving_puzzles_time_l164_164116

theorem tony_solving_puzzles_time : ∀ (warm_up_time long_puzzle_ratio num_long_puzzles : ℕ),
  warm_up_time = 10 →
  long_puzzle_ratio = 3 →
  num_long_puzzles = 2 →
  (warm_up_time + long_puzzle_ratio * warm_up_time * num_long_puzzles) = 70 :=
by
  intros
  sorry

end tony_solving_puzzles_time_l164_164116


namespace grid_with_value_exists_possible_values_smallest_possible_value_l164_164973

open Nat

def isGridValuesP (P : ℕ) (a b c d e f g h i : ℕ) : Prop :=
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i) ∧
  (b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
   f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
   g ≠ h ∧ g ≠ i ∧
   h ≠ i) ∧
  (a * b * c = P) ∧ (d * e * f = P) ∧
  (g * h * i = P) ∧ (a * d * g = P) ∧
  (b * e * h = P) ∧ (c * f * i = P)

theorem grid_with_value_exists (P : ℕ) :
  ∃ a b c d e f g h i : ℕ, isGridValuesP P a b c d e f g h i :=
sorry

theorem possible_values (P : ℕ) :
  P ∈ [1992, 1995] ↔ 
  ∃ a b c d e f g h i : ℕ, isGridValuesP P a b c d e f g h i :=
sorry

theorem smallest_possible_value : 
  ∃ P a b c d e f g h i : ℕ, isGridValuesP P a b c d e f g h i ∧ 
  ∀ Q, (∃ w x y z u v s t q : ℕ, isGridValuesP Q w x y z u v s t q) → Q ≥ 120 :=
sorry

end grid_with_value_exists_possible_values_smallest_possible_value_l164_164973


namespace inverse_function_passes_through_point_a_l164_164248

theorem inverse_function_passes_through_point_a
  (a : ℝ) (ha_pos : 0 < a) (ha_neq_one : a ≠ 1) :
  ∃ (A : ℝ × ℝ), A = (2, 3) ∧ (∀ x, (a^(x-3) + 1) = 2 ↔ x = 3) → (2 - 1)/(3-3) = 0 :=
by
  sorry

end inverse_function_passes_through_point_a_l164_164248


namespace boat_speed_in_still_water_l164_164343

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 7) : b = 9 :=
by sorry

end boat_speed_in_still_water_l164_164343


namespace factorize_x_squared_plus_2x_l164_164156

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x + 2) :=
by sorry

end factorize_x_squared_plus_2x_l164_164156


namespace find_radius_l164_164890

noncomputable def radius_from_tangent_circles (AB : ℝ) (r : ℝ) : ℝ :=
  let O1O2 := 2 * r
  let proportion := AB / O1O2
  r + r * proportion

theorem find_radius
  (AB : ℝ) (r : ℝ)
  (hAB : AB = 11) (hr : r = 5) :
  radius_from_tangent_circles AB r = 55 :=
by
  sorry

end find_radius_l164_164890


namespace generate_13121_not_generate_12131_l164_164425

theorem generate_13121 : ∃ n m : ℕ, 13121 + 1 = 2^n * 3^m := by
  sorry

theorem not_generate_12131 : ¬∃ n m : ℕ, 12131 + 1 = 2^n * 3^m := by
  sorry

end generate_13121_not_generate_12131_l164_164425


namespace find_a_l164_164086

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a (x₀ a : ℝ) (h : f x₀ a - g x₀ a = 3) : a = -Real.log 2 - 1 :=
by
  sorry

end find_a_l164_164086


namespace find_room_length_l164_164352

variable (width : ℝ) (cost rate : ℝ) (length : ℝ)

theorem find_room_length (h_width : width = 4.75)
  (h_cost : cost = 34200)
  (h_rate : rate = 900)
  (h_area : cost / rate = length * width) :
  length = 8 :=
sorry

end find_room_length_l164_164352


namespace sum_of_digits_second_smallest_mult_of_lcm_l164_164356

theorem sum_of_digits_second_smallest_mult_of_lcm :
  let lcm12345678 := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8
  let M := 2 * lcm12345678
  (Nat.digits 10 M).sum = 15 := by
    -- Definitions from the problem statement
    let lcm12345678 := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8
    let M := 2 * lcm12345678
    sorry

end sum_of_digits_second_smallest_mult_of_lcm_l164_164356


namespace quadratic_coefficients_l164_164125

theorem quadratic_coefficients :
  ∀ x : ℝ, x * (x + 2) = 5 * (x - 2) → ∃ a b c : ℝ, a = 1 ∧ b = -3 ∧ c = 10 ∧ a * x^2 + b * x + c = 0 := by
  intros x h
  use 1, -3, 10
  sorry

end quadratic_coefficients_l164_164125


namespace text_messages_December_l164_164614

-- Definitions of the number of text messages sent each month
def text_messages_November := 1
def text_messages_January := 4
def text_messages_February := 8
def doubling_pattern (a b : ℕ) : Prop := b = 2 * a

-- Prove that Jared sent 2 text messages in December
theorem text_messages_December : ∃ x : ℕ, 
  doubling_pattern text_messages_November x ∧ 
  doubling_pattern x text_messages_January ∧ 
  doubling_pattern text_messages_January text_messages_February ∧ 
  x = 2 :=
by
  sorry

end text_messages_December_l164_164614


namespace therapist_charge_difference_l164_164369

theorem therapist_charge_difference :
  ∃ F A : ℝ, F + 4 * A = 350 ∧ F + A = 161 ∧ F - A = 35 :=
by {
  -- Placeholder for the actual proof.
  sorry
}

end therapist_charge_difference_l164_164369


namespace time_to_cross_platform_l164_164476

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

end time_to_cross_platform_l164_164476


namespace divisible_by_6_l164_164265

theorem divisible_by_6 (n : ℤ) : 6 ∣ (n^3 - n + 6) :=
by
  sorry

end divisible_by_6_l164_164265


namespace f_monotonic_m_range_l164_164333

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.tan x - 2 * x

theorem f_monotonic {x : ℝ} (h : x ∈ Set.Ioo (-Real.pi / 2) (Real.pi / 2)) :
  Monotone f :=
sorry

theorem m_range {x : ℝ} (h : x ∈ Set.Ioo 0 (Real.pi / 2)) {m : ℝ} (hm : f x ≥ m * x^2) :
  m ≤ 0 :=
sorry

end f_monotonic_m_range_l164_164333


namespace fraction_problem_l164_164421

theorem fraction_problem (a : ℕ) (h1 : (a:ℚ)/(a + 27) = 865/1000) : a = 173 := 
by
  sorry

end fraction_problem_l164_164421


namespace maximize_value_l164_164873

def f (x : ℝ) : ℝ := -3 * x^2 - 8 * x + 18

theorem maximize_value : ∀ x : ℝ, f x ≤ f (-4/3) :=
by sorry

end maximize_value_l164_164873


namespace find_ABC_base10_l164_164713

theorem find_ABC_base10
  (A B C : ℕ)
  (h1 : 0 < A ∧ A < 6)
  (h2 : 0 < B ∧ B < 6)
  (h3 : 0 < C ∧ C < 6)
  (h4 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h5 : B + C = 6)
  (h6 : A + 1 = C)
  (h7 : A + B = C) :
  100 * A + 10 * B + C = 415 :=
by
  sorry

end find_ABC_base10_l164_164713


namespace largest_angle_of_pentagon_l164_164629

theorem largest_angle_of_pentagon (a d : ℝ) (h1 : a = 100) (h2 : d = 2) :
  let angle1 := a
  let angle2 := a + d
  let angle3 := a + 2 * d
  let angle4 := a + 3 * d
  let angle5 := a + 4 * d
  angle1 + angle2 + angle3 + angle4 + angle5 = 540 ∧ angle5 = 116 :=
by
  sorry

end largest_angle_of_pentagon_l164_164629


namespace x_cubed_gt_y_squared_l164_164681

theorem x_cubed_gt_y_squared (x y : ℝ) (h1 : x^5 > y^4) (h2 : y^5 > x^4) : x^3 > y^2 := by
  sorry

end x_cubed_gt_y_squared_l164_164681


namespace right_triangle_set_D_l164_164258

theorem right_triangle_set_D : (5^2 + 12^2 = 13^2) ∧ 
  ((3^2 + 3^2 ≠ 5^2) ∧ (6^2 + 8^2 ≠ 9^2) ∧ (4^2 + 5^2 ≠ 6^2)) :=
by
  sorry

end right_triangle_set_D_l164_164258


namespace sum_first_six_terms_l164_164190

variable {S : ℕ → ℝ}

theorem sum_first_six_terms (h2 : S 2 = 4) (h4 : S 4 = 6) : S 6 = 7 := 
  sorry

end sum_first_six_terms_l164_164190


namespace range_of_a_l164_164327

-- Define the sets A and B
def setA (a : ℝ) : Set ℝ := {x | x - a > 0}
def setB : Set ℝ := {x | x ≤ 0}

-- The main theorem asserting the condition
theorem range_of_a {a : ℝ} (h : setA a ∩ setB = ∅) : a ≥ 0 := by
  sorry

end range_of_a_l164_164327


namespace brad_ate_six_halves_l164_164323

theorem brad_ate_six_halves (total_cookies : ℕ) (total_halves : ℕ) (greg_ate : ℕ) (halves_left : ℕ) (halves_brad_ate : ℕ) 
  (h1 : total_cookies = 14)
  (h2 : total_halves = total_cookies * 2)
  (h3 : greg_ate = 4)
  (h4 : halves_left = 18)
  (h5 : total_halves - greg_ate - halves_brad_ate = halves_left) :
  halves_brad_ate = 6 :=
by
  sorry

end brad_ate_six_halves_l164_164323


namespace second_hand_degrees_per_minute_l164_164367

theorem second_hand_degrees_per_minute (clock_gains_5_minutes_per_hour : true) :
  (360 / 60 = 6) := 
by
  sorry

end second_hand_degrees_per_minute_l164_164367


namespace num_candidates_l164_164408

theorem num_candidates (n : ℕ) (h : n * (n - 1) = 30) : n = 6 :=
sorry

end num_candidates_l164_164408


namespace total_spent_proof_l164_164902

noncomputable def total_spent (cost_pen cost_pencil cost_notebook : ℝ) 
  (pens_robert pencils_robert notebooks_dorothy : ℕ) 
  (julia_pens_ratio robert_pens_ratio dorothy_pens_ratio : ℝ) 
  (julia_pencils_diff notebooks_julia_diff : ℕ) 
  (robert_notebooks_ratio dorothy_pencils_ratio : ℝ) : ℝ :=
    let pens_julia := robert_pens_ratio * pens_robert
    let pens_dorothy := dorothy_pens_ratio * pens_julia
    let total_pens := pens_robert + pens_julia + pens_dorothy
    let cost_pens := total_pens * cost_pen 
    
    let pencils_julia := pencils_robert - julia_pencils_diff
    let pencils_dorothy := dorothy_pencils_ratio * pencils_julia
    let total_pencils := pencils_robert + pencils_julia + pencils_dorothy
    let cost_pencils := total_pencils * cost_pencil 
        
    let notebooks_julia := notebooks_dorothy + notebooks_julia_diff
    let notebooks_robert := robert_notebooks_ratio * notebooks_julia
    let total_notebooks := notebooks_dorothy + notebooks_julia + notebooks_robert
    let cost_notebooks := total_notebooks * cost_notebook
        
    cost_pens + cost_pencils + cost_notebooks

theorem total_spent_proof 
  (cost_pen : ℝ := 1.50)
  (cost_pencil : ℝ := 0.75)
  (cost_notebook : ℝ := 4.00)
  (pens_robert : ℕ := 4)
  (pencils_robert : ℕ := 12)
  (notebooks_dorothy : ℕ := 3)
  (julia_pens_ratio : ℝ := 3)
  (robert_pens_ratio : ℝ := 3)
  (dorothy_pens_ratio : ℝ := 0.5)
  (julia_pencils_diff : ℕ := 5)
  (notebooks_julia_diff : ℕ := 1)
  (robert_notebooks_ratio : ℝ := 0.5)
  (dorothy_pencils_ratio : ℝ := 2) : 
  total_spent cost_pen cost_pencil cost_notebook pens_robert pencils_robert notebooks_dorothy 
    julia_pens_ratio robert_pens_ratio dorothy_pens_ratio julia_pencils_diff notebooks_julia_diff robert_notebooks_ratio dorothy_pencils_ratio 
    = 93.75 := 
by 
  sorry

end total_spent_proof_l164_164902


namespace correct_time_fraction_l164_164209

theorem correct_time_fraction : 
  let hours_with_glitch := [5]
  let minutes_with_glitch := [5, 15, 25, 35, 45, 55]
  let total_hours := 12
  let total_minutes_per_hour := 60
  let correct_hours := total_hours - hours_with_glitch.length
  let correct_minutes := total_minutes_per_hour - minutes_with_glitch.length
  (correct_hours * correct_minutes) / (total_hours * total_minutes_per_hour) = 33 / 40 :=
by
  sorry

end correct_time_fraction_l164_164209


namespace part_I_part_II_l164_164040

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + a + 2

theorem part_I (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≤ 0) →
  -1 < a ∧ a ≤ 11/5 :=
sorry

noncomputable def g (x a : ℝ) : ℝ := 
  if abs x ≥ 1 then 2 * x^2 - 2 * a * x + a + 1 
  else -2 * a * x + a + 3

theorem part_II (a : ℝ) :
  (∃ x1 x2, 0 < x1 ∧ x1 < x2 ∧ x2 < 3 ∧ g x1 a = 0 ∧ g x2 a = 0) →
  1 + Real.sqrt 3 < a ∧ a ≤ 19/5 :=
sorry

end part_I_part_II_l164_164040


namespace geoff_election_l164_164637

theorem geoff_election (Votes: ℝ) (Percent: ℝ) (ExtraVotes: ℝ) (x: ℝ) 
  (h1 : Votes = 6000) 
  (h2 : Percent = 1) 
  (h3 : ExtraVotes = 3000) 
  (h4 : ReceivedVotes = (Percent / 100) * Votes) 
  (h5 : TotalVotesNeeded = ReceivedVotes + ExtraVotes) 
  (h6 : x = (TotalVotesNeeded / Votes) * 100) :
  x = 51 := 
  by 
    sorry

end geoff_election_l164_164637


namespace cost_of_football_correct_l164_164490

-- We define the variables for the costs
def total_amount_spent : ℝ := 20.52
def cost_of_marbles : ℝ := 9.05
def cost_of_baseball : ℝ := 6.52
def cost_of_football : ℝ := total_amount_spent - cost_of_marbles - cost_of_baseball

-- We now state what needs to be proven: that Mike spent $4.95 on the football.
theorem cost_of_football_correct : cost_of_football = 4.95 := by
  sorry

end cost_of_football_correct_l164_164490


namespace distance_center_to_plane_l164_164114

noncomputable def sphere_center_to_plane_distance 
  (volume : ℝ) (AB AC : ℝ) (angleACB : ℝ) : ℝ :=
  let R := (3 * volume / 4 / Real.pi)^(1 / 3);
  let circumradius := AB / (2 * Real.sin (angleACB / 2));
  Real.sqrt (R^2 - circumradius^2)

theorem distance_center_to_plane 
  (volume : ℝ) (AB : ℝ) (angleACB : ℝ)
  (h_volume : volume = 500 * Real.pi / 3)
  (h_AB : AB = 4 * Real.sqrt 3)
  (h_angleACB : angleACB = Real.pi / 3) :
  sphere_center_to_plane_distance volume AB angleACB = 3 :=
by
  sorry

end distance_center_to_plane_l164_164114


namespace midpoint_sum_l164_164867

theorem midpoint_sum :
  let x1 := 8
  let y1 := -4
  let z1 := 10
  let x2 := -2
  let y2 := 10
  let z2 := -6
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  let midpoint_z := (z1 + z2) / 2
  midpoint_x + midpoint_y + midpoint_z = 8 :=
by
  -- We just need to state the theorem, proof is not required
  sorry

end midpoint_sum_l164_164867


namespace soccer_team_games_l164_164530

theorem soccer_team_games (pizzas : ℕ) (slices_per_pizza : ℕ) (average_goals_per_game : ℕ) (total_games : ℕ) 
  (h1 : pizzas = 6) 
  (h2 : slices_per_pizza = 12) 
  (h3 : average_goals_per_game = 9) 
  (h4 : total_games = (pizzas * slices_per_pizza) / average_goals_per_game) :
  total_games = 8 :=
by
  rw [h1, h2, h3] at h4
  exact h4

end soccer_team_games_l164_164530


namespace replace_digits_correct_l164_164472

def digits_eq (a b c d e : ℕ) : Prop :=
  5 * 10 + a + (b * 100) + (c * 10) + 3 = (d * 1000) + (e * 100) + 1

theorem replace_digits_correct :
  ∃ (a b c d e : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧
    digits_eq a b c d e ∧ a = 1 ∧ b = 1 ∧ c = 4 ∧ d = 1 ∧ e = 4 :=
by
  sorry

end replace_digits_correct_l164_164472


namespace toby_steps_needed_l164_164951

noncomputable def total_steps_needed : ℕ := 10000 * 9

noncomputable def first_sunday_steps : ℕ := 10200
noncomputable def first_monday_steps : ℕ := 10400
noncomputable def tuesday_steps : ℕ := 9400
noncomputable def wednesday_steps : ℕ := 9100
noncomputable def thursday_steps : ℕ := 8300
noncomputable def friday_steps : ℕ := 9200
noncomputable def saturday_steps : ℕ := 8900
noncomputable def second_sunday_steps : ℕ := 9500

noncomputable def total_steps_walked := 
  first_sunday_steps + 
  first_monday_steps + 
  tuesday_steps + 
  wednesday_steps + 
  thursday_steps + 
  friday_steps + 
  saturday_steps + 
  second_sunday_steps

noncomputable def remaining_steps_needed := total_steps_needed - total_steps_walked

noncomputable def days_left : ℕ := 3

noncomputable def average_steps_needed := remaining_steps_needed / days_left

theorem toby_steps_needed : average_steps_needed = 5000 := by
  sorry

end toby_steps_needed_l164_164951


namespace rectangle_dimensions_l164_164073

theorem rectangle_dimensions (l w : ℝ) : 
  (∃ x : ℝ, x = l - 3 ∧ x = w - 2 ∧ x^2 = (1 / 2) * l * w) → (l = 9 ∧ w = 8) :=
by
  sorry

end rectangle_dimensions_l164_164073


namespace students_received_B_l164_164675

theorem students_received_B (charles_ratio : ℚ) (dawsons_class : ℕ) 
  (h_charles_ratio : charles_ratio = 3 / 5) (h_dawsons_class : dawsons_class = 30) : 
  ∃ y : ℕ, (charles_ratio = y / dawsons_class) ∧ y = 18 := 
by 
  sorry

end students_received_B_l164_164675


namespace find_a_value_l164_164153

-- Define the problem conditions
theorem find_a_value (a : ℝ) :
  let x_values := [0, 1, 3, 4]
  let y_values := [a, 4.3, 4.8, 6.7]
  let mean_x := (0 + 1 + 3 + 4) / 4
  let mean_y := (a + 4.3 + 4.8 + 6.7) / 4
  (mean_y = 0.95 * mean_x + 2.6) → a = 2.2 :=
by
  -- Let bindings are for convenience to follow the problem statement
  let x_values := [0, 1, 3, 4]
  let y_values := [a, 4.3, 4.8, 6.7]
  let mean_x := (0 + 1 + 3 + 4) / 4
  let mean_y := (a + 4.3 + 4.8 + 6.7) / 4
  intro h
  sorry

end find_a_value_l164_164153


namespace find_longer_parallel_side_length_l164_164328

noncomputable def longer_parallel_side_length_of_trapezoid : ℝ :=
  let square_side_length : ℝ := 2
  let center_to_side_length : ℝ := square_side_length / 2
  let midline_length : ℝ := square_side_length / 2
  let equal_area : ℝ := (square_side_length^2) / 3
  let height_of_trapezoid : ℝ := center_to_side_length
  let shorter_parallel_side_length : ℝ := midline_length
  let longer_parallel_side_length := (2 * equal_area / height_of_trapezoid) - shorter_parallel_side_length
  longer_parallel_side_length

theorem find_longer_parallel_side_length : 
  longer_parallel_side_length_of_trapezoid = 5/3 := 
sorry

end find_longer_parallel_side_length_l164_164328


namespace boxes_left_l164_164772

theorem boxes_left (boxes_saturday boxes_sunday apples_per_box apples_sold : ℕ)
  (h_saturday : boxes_saturday = 50)
  (h_sunday : boxes_sunday = 25)
  (h_apples_per_box : apples_per_box = 10)
  (h_apples_sold : apples_sold = 720) :
  ((boxes_saturday + boxes_sunday) * apples_per_box - apples_sold) / apples_per_box = 3 :=
by
  sorry

end boxes_left_l164_164772


namespace find_D_l164_164801

-- Definitions
def divides (a b : ℕ) : Prop := ∃ k, b = a * k
def remainder (a b r : ℕ) : Prop := ∃ k, a = b * k + r

-- Problem Statement
theorem find_D {N D : ℕ} (h1 : remainder N D 75) (h2 : remainder N 37 1) : 
  D = 112 :=
by
  sorry

end find_D_l164_164801


namespace at_least_two_fail_l164_164084

theorem at_least_two_fail (p q : ℝ) (n : ℕ) (h_p : p = 0.2) (h_q : q = 1 - p) :
  n ≥ 18 → (1 - ((q^n) * (1 + n * p / 4))) ≥ 0.9 :=
by
  sorry

end at_least_two_fail_l164_164084


namespace max_sum_combined_shape_l164_164566

-- Definitions for the initial prism
def faces_prism := 6
def edges_prism := 12
def vertices_prism := 8

-- Definitions for the changes when pyramid is added to a rectangular face
def additional_faces_rect := 4
def additional_edges_rect := 4
def additional_vertices_rect := 1

-- Definition for the maximum sum calculation
def max_sum := faces_prism - 1 + additional_faces_rect + 
               edges_prism + additional_edges_rect + 
               vertices_prism + additional_vertices_rect

-- The theorem to prove the maximum sum
theorem max_sum_combined_shape : max_sum = 34 :=
by
  sorry

end max_sum_combined_shape_l164_164566


namespace arithmetic_sequence_example_l164_164379

theorem arithmetic_sequence_example (a : ℕ → ℝ) (d : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h2 : a 2 = 2) (h14 : a 14 = 18) : a 8 = 10 :=
by
  sorry

end arithmetic_sequence_example_l164_164379


namespace find_A_d_minus_B_d_l164_164205

variable {d : ℕ} (A B : ℕ) (h₁ : d > 6) (h₂ : (d^1 * A + B) + (d^1 * A + A) = 1 * d^2 + 6 * d^1 + 2)

theorem find_A_d_minus_B_d (h₁ : d > 6) (h₂ : (d^1 * A + B) + (d^1 * A + A) = 1 * d^2 + 6 * d^1 + 2) :
  A - B = 3 :=
sorry

end find_A_d_minus_B_d_l164_164205


namespace sum_of_possible_g9_values_l164_164094

def f (x : ℝ) : ℝ := x^2 - 6 * x + 14

def g (y : ℝ) : ℝ := 3 * y + 2

theorem sum_of_possible_g9_values : ∀ {x1 x2 : ℝ}, f x1 = 9 → f x2 = 9 → g x1 + g x2 = 22 := by
  intros
  sorry

end sum_of_possible_g9_values_l164_164094


namespace find_k_values_l164_164845

noncomputable def parallel_vectors (k : ℝ) : Prop :=
  (k^2 / k = (k + 1) / 4)

theorem find_k_values (k : ℝ) : parallel_vectors k ↔ (k = 0 ∨ k = 1 / 3) :=
by sorry

end find_k_values_l164_164845


namespace coloring_15_segments_impossible_l164_164928

theorem coloring_15_segments_impossible :
  ¬ ∃ (colors : Fin 15 → Fin 3) (adj : Fin 15 → Fin 2),
    ∀ i j, adj i = adj j → i ≠ j → colors i ≠ colors j :=
by
  sorry

end coloring_15_segments_impossible_l164_164928


namespace calc_square_difference_and_square_l164_164672

theorem calc_square_difference_and_square (a b : ℤ) (h1 : a = 7) (h2 : b = 3)
  (h3 : a^2 = 49) (h4 : b^2 = 9) : (a^2 - b^2)^2 = 1600 := by
  sorry

end calc_square_difference_and_square_l164_164672


namespace weight_triangle_correct_weight_l164_164778

noncomputable def area_square (side : ℝ) : ℝ := side ^ 2

noncomputable def area_triangle (side : ℝ) : ℝ := (side ^ 2 * Real.sqrt 3) / 4

noncomputable def weight (area : ℝ) (density : ℝ) := area * density

noncomputable def weight_equilateral_triangle (weight_square : ℝ) (side_square : ℝ) (side_triangle : ℝ) : ℝ :=
  let area_s := area_square side_square
  let area_t := area_triangle side_triangle
  let density := weight_square / area_s
  weight area_t density

theorem weight_triangle_correct_weight :
  weight_equilateral_triangle 8 4 6 = 9 * Real.sqrt 3 / 2 := by sorry

end weight_triangle_correct_weight_l164_164778


namespace intersection_A_B_l164_164272

-- Define set A and set B based on the conditions
def set_A : Set ℝ := {x : ℝ | x^2 - 3 * x - 4 < 0}
def set_B : Set ℝ := {-4, 1, 3, 5}

-- State the theorem to prove the intersection of A and B
theorem intersection_A_B : set_A ∩ set_B = {1, 3} :=
by sorry

end intersection_A_B_l164_164272


namespace john_buys_spools_l164_164931

theorem john_buys_spools (spool_length necklace_length : ℕ) 
  (necklaces : ℕ) 
  (total_length := necklaces * necklace_length) 
  (spools := total_length / spool_length) :
  spool_length = 20 → 
  necklace_length = 4 → 
  necklaces = 15 → 
  spools = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end john_buys_spools_l164_164931


namespace arithmetic_sequence_common_diff_l164_164988

noncomputable def variance (s : List ℝ) : ℝ :=
  let mean := (s.sum) / (s.length : ℝ)
  (s.map (λ x => (x - mean) ^ 2)).sum / (s.length : ℝ)

theorem arithmetic_sequence_common_diff (a1 a2 a3 a4 a5 a6 a7 d : ℝ) 
(h_seq : a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d ∧ a5 = a1 + 4 * d ∧ a6 = a1 + 5 * d ∧ a7 = a1 + 6 * d)
(h_var : variance [a1, a2, a3, a4, a5, a6, a7] = 1) : 
d = 1 / 2 ∨ d = -1 / 2 := 
sorry

end arithmetic_sequence_common_diff_l164_164988


namespace gcd_228_1995_l164_164251

theorem gcd_228_1995 :
  Nat.gcd 228 1995 = 21 :=
sorry

end gcd_228_1995_l164_164251


namespace option_d_correct_l164_164847

theorem option_d_correct (a b : ℝ) (h : a * b < 0) : 
  (a / b + b / a) ≤ -2 := by
  sorry

end option_d_correct_l164_164847


namespace simplify_expression_l164_164818

theorem simplify_expression (y : ℝ) : y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 :=
sorry

end simplify_expression_l164_164818


namespace part1_solution_set_part2_values_a_b_part3_range_m_l164_164464

-- Definitions for the given functions
def y1 (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def y2 (x : ℝ) : ℝ := x^2 + x - 2

-- Proof that the solution set for y2 < 0 is (-2, 1)
theorem part1_solution_set : ∀ x : ℝ, y2 x < 0 ↔ (x > -2 ∧ x < 1) :=
sorry

-- Given |y1| ≤ |y2| for all x ∈ ℝ, prove that a = 1 and b = -2
theorem part2_values_a_b (a b : ℝ) : (∀ x : ℝ, |y1 x a b| ≤ |y2 x|) → a = 1 ∧ b = -2 :=
sorry

-- Given y1 > (m-2)x - m for all x > 1 under condition from part 2, prove the range for m is (-∞, 2√2 + 5)
theorem part3_range_m (a b : ℝ) (m : ℝ) : 
  (∀ x : ℝ, |y1 x a b| ≤ |y2 x|) → a = 1 ∧ b = -2 →
  (∀ x : ℝ, x > 1 → y1 x a b > (m-2) * x - m) → m < 2 * Real.sqrt 2 + 5 :=
sorry

end part1_solution_set_part2_values_a_b_part3_range_m_l164_164464


namespace dance_lesson_cost_l164_164016

-- Define the conditions
variable (total_lessons : Nat) (free_lessons : Nat) (paid_lessons_cost : Nat)

-- State the problem with the given conditions
theorem dance_lesson_cost
  (h1 : total_lessons = 10)
  (h2 : free_lessons = 2)
  (h3 : paid_lessons_cost = 80) :
  let number_of_paid_lessons := total_lessons - free_lessons
  number_of_paid_lessons ≠ 0 -> 
  (paid_lessons_cost / number_of_paid_lessons) = 10 := by
  sorry

end dance_lesson_cost_l164_164016


namespace cord_lengths_l164_164435

noncomputable def cordLengthFirstDog (distance : ℝ) : ℝ :=
  distance / 2

noncomputable def cordLengthSecondDog (distance : ℝ) : ℝ :=
  distance / 2

noncomputable def cordLengthThirdDog (radius : ℝ) : ℝ :=
  radius

theorem cord_lengths (d1 d2 r : ℝ) (h1 : d1 = 30) (h2 : d2 = 40) (h3 : r = 20) :
  cordLengthFirstDog d1 = 15 ∧ cordLengthSecondDog d2 = 20 ∧ cordLengthThirdDog r = 20 := by
  sorry

end cord_lengths_l164_164435


namespace compound_interest_amount_l164_164613

theorem compound_interest_amount 
  (P_si : ℝ := 3225) 
  (R_si : ℝ := 8) 
  (T_si : ℝ := 5) 
  (R_ci : ℝ := 15) 
  (T_ci : ℝ := 2) 
  (SI : ℝ := P_si * R_si * T_si / 100) 
  (CI : ℝ := 2 * SI) 
  (CI_formula : ℝ := P_ci * ((1 + R_ci / 100)^T_ci - 1))
  (P_ci := 516 / 0.3225) :
  P_ci = 1600 := 
by
  sorry

end compound_interest_amount_l164_164613


namespace value_of_square_l164_164883

theorem value_of_square (z : ℝ) (h : 3 * z^2 + 2 * z = 5 * z + 11) : (6 * z - 5)^2 = 141 := by
  sorry

end value_of_square_l164_164883


namespace find_expression_value_l164_164185

theorem find_expression_value (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + m^3 + 1/m^3 + 4 = 1072 := 
by 
  sorry

end find_expression_value_l164_164185


namespace solution_set_of_inequality_l164_164489

theorem solution_set_of_inequality : {x : ℝ // |x - 2| > x - 2} = {x : ℝ // x < 2} :=
sorry

end solution_set_of_inequality_l164_164489


namespace most_likely_number_of_cars_l164_164359

theorem most_likely_number_of_cars 
    (cars_in_first_10_seconds : ℕ := 6) 
    (time_for_first_10_seconds : ℕ := 10) 
    (total_time_seconds : ℕ := 165) 
    (constant_speed : Prop := true) : 
    ∃ (num_cars : ℕ), num_cars = 100 :=
by
  sorry

end most_likely_number_of_cars_l164_164359


namespace range_of_a_l164_164243

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x ≤ a → (x + y + 1 ≤ 2 * (x + 1) - 3 * (y + 1))) → a ≤ -2 :=
by 
  intros h
  sorry

end range_of_a_l164_164243


namespace right_triangle_sides_l164_164470

theorem right_triangle_sides (m n : ℝ) (x : ℝ) (a b c : ℝ)
  (h1 : 2 * x < m + n) 
  (h2 : a = Real.sqrt (2 * m * n) - m)
  (h3 : b = Real.sqrt (2 * m * n) - n)
  (h4 : c = m + n - Real.sqrt (2 * m * n))
  (h5 : a^2 + b^2 = c^2)
  (h6 : 4 * x^2 = (m - 2 * x)^2 + (n - 2 * x)^2) :
  a = Real.sqrt (2 * m * n) - m ∧ b = Real.sqrt (2 * m * n) - n ∧ c = m + n - Real.sqrt (2 * m * n) :=
by
  sorry

end right_triangle_sides_l164_164470


namespace parallel_resistors_l164_164354
noncomputable def resistance_R (x y z w : ℝ) : ℝ :=
  1 / (1/x + 1/y + 1/z + 1/w)

theorem parallel_resistors :
  resistance_R 5 7 3 9 = 315 / 248 :=
by
  sorry

end parallel_resistors_l164_164354


namespace find_k_l164_164926

theorem find_k (k : ℕ) (h : (64 : ℕ) / k = 4) : k = 16 := by
  sorry

end find_k_l164_164926


namespace mangoes_count_l164_164578

noncomputable def total_fruits : ℕ := 58
noncomputable def pears : ℕ := 10
noncomputable def pawpaws : ℕ := 12
noncomputable def lemons : ℕ := 9
noncomputable def kiwi : ℕ := 9

theorem mangoes_count (mangoes : ℕ) : 
  (pears + pawpaws + lemons + kiwi + mangoes = total_fruits) → 
  mangoes = 18 :=
by
  sorry

end mangoes_count_l164_164578


namespace jars_of_pickled_mangoes_l164_164683

def total_mangoes := 54
def ratio_ripe := 1/3
def ratio_unripe := 2/3
def kept_unripe_mangoes := 16
def mangoes_per_jar := 4

theorem jars_of_pickled_mangoes : 
  (total_mangoes * ratio_unripe - kept_unripe_mangoes) / mangoes_per_jar = 5 :=
by
  sorry

end jars_of_pickled_mangoes_l164_164683


namespace amount_subtracted_is_30_l164_164661

-- Definitions based on conditions
def N : ℕ := 200
def subtracted_amount (A : ℕ) : Prop := 0.40 * (N : ℝ) - (A : ℝ) = 50

-- The theorem statement
theorem amount_subtracted_is_30 : subtracted_amount 30 :=
by 
  -- proof will be completed here
  sorry

end amount_subtracted_is_30_l164_164661


namespace right_triangle_acute_angle_l164_164810

theorem right_triangle_acute_angle (a b : ℝ) (h1 : a + b = 90) (h2 : a = 55) : b = 35 := 
by sorry

end right_triangle_acute_angle_l164_164810


namespace percentage_weight_loss_measured_l164_164458

variable (W : ℝ)

def weight_after_loss (W : ℝ) := 0.85 * W
def weight_with_clothes (W : ℝ) := weight_after_loss W * 1.02

theorem percentage_weight_loss_measured (W : ℝ) :
  ((W - weight_with_clothes W) / W) * 100 = 13.3 := by
  sorry

end percentage_weight_loss_measured_l164_164458


namespace find_x_l164_164730

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := 
by
  sorry

end find_x_l164_164730


namespace sum_of_solutions_of_quadratic_l164_164270

theorem sum_of_solutions_of_quadratic :
    let a := 1;
    let b := -8;
    let c := -40;
    let discriminant := b * b - 4 * a * c;
    let root_discriminant := Real.sqrt discriminant;
    let sol1 := (-b + root_discriminant) / (2 * a);
    let sol2 := (-b - root_discriminant) / (2 * a);
    sol1 + sol2 = 8 := by
{
  sorry
}

end sum_of_solutions_of_quadratic_l164_164270


namespace solve_system_of_equations_solve_fractional_equation_l164_164021

noncomputable def solution1 (x y : ℚ) := (3 * x - 5 * y = 3) ∧ (x / 2 - y / 3 = 1) ∧ (x = 8 / 3) ∧ (y = 1)

noncomputable def solution2 (x : ℚ) := (x / (x - 1) + 1 = 3 / (2 * x - 2)) ∧ (x = 5 / 4)

theorem solve_system_of_equations (x y : ℚ) : solution1 x y := by
  sorry

theorem solve_fractional_equation (x : ℚ) : solution2 x := by
  sorry

end solve_system_of_equations_solve_fractional_equation_l164_164021


namespace least_number_of_cans_l164_164453

theorem least_number_of_cans 
  (Maaza_volume : ℕ) (Pepsi_volume : ℕ) (Sprite_volume : ℕ) 
  (h1 : Maaza_volume = 80) (h2 : Pepsi_volume = 144) (h3 : Sprite_volume = 368) :
  (Maaza_volume / Nat.gcd (Nat.gcd Maaza_volume Pepsi_volume) Sprite_volume) +
  (Pepsi_volume / Nat.gcd (Nat.gcd Maaza_volume Pepsi_volume) Sprite_volume) +
  (Sprite_volume / Nat.gcd (Nat.gcd Maaza_volume Pepsi_volume) Sprite_volume) = 37 := by
  sorry

end least_number_of_cans_l164_164453


namespace minimize_expression_l164_164215

open Real

theorem minimize_expression (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) :
  4 * p^3 + 6 * q^3 + 24 * r^3 + 8 / (3 * p * q * r) ≥ 16 :=
sorry

end minimize_expression_l164_164215


namespace sum_of_squares_s_comp_r_l164_164679

def r (x : ℝ) : ℝ := x^2 - 4
def s (x : ℝ) : ℝ := -|x + 1|
def s_comp_r (x : ℝ) : ℝ := s (r x)

theorem sum_of_squares_s_comp_r :
  (s_comp_r (-4))^2 + (s_comp_r (-3))^2 + (s_comp_r (-2))^2 + (s_comp_r (-1))^2 +
  (s_comp_r 0)^2 + (s_comp_r 1)^2 + (s_comp_r 2)^2 + (s_comp_r 3)^2 + (s_comp_r 4)^2 = 429 :=
by
  sorry

end sum_of_squares_s_comp_r_l164_164679


namespace original_employees_229_l164_164060

noncomputable def original_number_of_employees (reduced_employees : ℕ) (reduction_percentage : ℝ) : ℝ := 
  reduced_employees / (1 - reduction_percentage)

theorem original_employees_229 : original_number_of_employees 195 0.15 = 229 := 
by
  sorry

end original_employees_229_l164_164060


namespace ratio_of_areas_of_squares_l164_164289

open Real

theorem ratio_of_areas_of_squares :
  let side_length_C := 48
  let side_length_D := 60
  let area_C := side_length_C^2
  let area_D := side_length_D^2
  area_C / area_D = (16 : ℝ) / 25 :=
by
  sorry

end ratio_of_areas_of_squares_l164_164289


namespace max_square_test_plots_l164_164868

theorem max_square_test_plots 
  (length : ℕ) (width : ℕ) (fence_available : ℕ) 
  (side_length : ℕ) (num_plots : ℕ) 
  (h_length : length = 30)
  (h_width : width = 60)
  (h_fencing : fence_available = 2500)
  (h_side_length : side_length = 10)
  (h_num_plots : num_plots = 18) :
  (length * width / side_length^2 = num_plots) ∧
  (30 * (60 / side_length - 1) + 60 * (30 / side_length - 1) ≤ fence_available) := 
sorry

end max_square_test_plots_l164_164868


namespace original_length_of_field_l164_164473

theorem original_length_of_field (L W : ℕ) 
  (h1 : L * W = 144) 
  (h2 : (L + 6) * W = 198) : 
  L = 16 := 
by 
  sorry

end original_length_of_field_l164_164473


namespace probability_no_prize_l164_164570

theorem probability_no_prize : (1 : ℚ) - (1 : ℚ) / (50 * 50) = 2499 / 2500 :=
by
  sorry

end probability_no_prize_l164_164570


namespace percentage_managers_decrease_l164_164389

theorem percentage_managers_decrease
  (employees : ℕ)
  (initial_percentage : ℝ)
  (managers_leave : ℝ)
  (new_percentage : ℝ)
  (h1 : employees = 200)
  (h2 : initial_percentage = 99)
  (h3 : managers_leave = 100)
  (h4 : new_percentage = 98) :
  ((initial_percentage / 100 * employees - managers_leave) / (employees - managers_leave) * 100 = new_percentage) :=
by
  -- To be proven
  sorry

end percentage_managers_decrease_l164_164389


namespace find_n_value_l164_164903

theorem find_n_value (n a b : ℕ) 
    (h1 : n = 12 * b + a)
    (h2 : n = 10 * a + b)
    (h3 : 0 ≤ a ∧ a ≤ 11)
    (h4 : 0 ≤ b ∧ b ≤ 9) : 
    n = 119 :=
by
  sorry

end find_n_value_l164_164903


namespace increased_speed_l164_164017

theorem increased_speed (S : ℝ) : 
  (∀ (usual_speed : ℝ) (usual_time : ℝ) (distance : ℝ), 
    usual_speed = 20 ∧ distance = 100 ∧ usual_speed * usual_time = distance ∧ S * (usual_time - 1) = distance) → 
  S = 25 :=
by
  intros h1
  sorry

end increased_speed_l164_164017


namespace find_x_l164_164138

variables (a b x : ℝ)
variables (pos_a : a > 0) (pos_b : b > 0) (pos_x : x > 0)

theorem find_x : ((2 * a) ^ (2 * b) = (a^2) ^ b * x ^ b) → (x = 4) := by
  sorry

end find_x_l164_164138


namespace find_number_l164_164182

theorem find_number (x: ℝ) (h1: 0.10 * x + 0.15 * 50 = 10.5) : x = 30 :=
by
  sorry

end find_number_l164_164182


namespace solve_system_of_equations_l164_164199

theorem solve_system_of_equations :
  ∃ x y : ℝ, (2 * x - 5 * y = -1) ∧ (-4 * x + y = -7) ∧ (x = 2) ∧ (y = 1) :=
by
  -- proof omitted
  sorry

end solve_system_of_equations_l164_164199


namespace first_thrilling_thursday_after_start_l164_164690

theorem first_thrilling_thursday_after_start (start_date : ℕ) (school_start_month : ℕ) (school_start_day_of_week : ℤ) (month_length : ℕ → ℕ) (day_of_week_on_first_of_month : ℕ → ℤ) : 
    school_start_month = 9 ∧ school_start_day_of_week = 2 ∧ start_date = 12 ∧ month_length 9 = 30 ∧ day_of_week_on_first_of_month 10 = 0 → 
    ∃ day_of_thursday : ℕ, day_of_thursday = 26 :=
by
  sorry

end first_thrilling_thursday_after_start_l164_164690


namespace tan_eq_tan_of_period_for_405_l164_164429

theorem tan_eq_tan_of_period_for_405 (m : ℤ) (h : -180 < m ∧ m < 180) :
  (Real.tan (m * (Real.pi / 180))) = (Real.tan (405 * (Real.pi / 180))) ↔ m = 45 ∨ m = -135 :=
by sorry

end tan_eq_tan_of_period_for_405_l164_164429


namespace solve_x_l164_164256

theorem solve_x (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 8 * x ^ 2 + 16 * x * y = x ^ 3 + 3 * x ^ 2 * y) (h₄ : y = 2 * x) : x = 40 / 7 :=
by
  sorry

end solve_x_l164_164256


namespace number_of_people_l164_164024

theorem number_of_people (x : ℕ) (H : x * (x - 1) = 72) : x = 9 :=
sorry

end number_of_people_l164_164024


namespace coeff_x2_product_l164_164960

open Polynomial

noncomputable def poly1 : Polynomial ℤ := -5 * X^3 - 5 * X^2 - 7 * X + 1
noncomputable def poly2 : Polynomial ℤ := -X^2 - 6 * X + 1

theorem coeff_x2_product : (poly1 * poly2).coeff 2 = 36 := by
  sorry

end coeff_x2_product_l164_164960


namespace solve_for_x_l164_164216

variable (x : ℝ)

theorem solve_for_x (h : 0.05 * x + 0.12 * (30 + x) = 15.6) : x = 12 / 0.17 := by
  sorry

end solve_for_x_l164_164216


namespace perimeter_ratio_l164_164635

def original_paper : ℕ × ℕ := (12, 8)
def folded_paper : ℕ × ℕ := (original_paper.1, original_paper.2 / 2)
def small_rectangle : ℕ × ℕ := (folded_paper.1 / 2, folded_paper.2)

def perimeter (rect : ℕ × ℕ) : ℕ :=
  2 * (rect.1 + rect.2)

theorem perimeter_ratio :
  perimeter small_rectangle = 1 / 2 * perimeter original_paper :=
by
  sorry

end perimeter_ratio_l164_164635


namespace josh_points_l164_164466

variable (x y : ℕ)
variable (three_point_success_rate two_point_success_rate : ℚ)
variable (total_shots : ℕ)
variable (points : ℚ)

theorem josh_points (h1 : three_point_success_rate = 0.25)
                    (h2 : two_point_success_rate = 0.40)
                    (h3 : total_shots = 40)
                    (h4 : x + y = total_shots) :
                    points = 32 :=
by sorry

end josh_points_l164_164466


namespace simplify_expression_l164_164050

theorem simplify_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^y * y^x) / (y^y * x^x) = (x / y) ^ (y - x) :=
by
  sorry

end simplify_expression_l164_164050


namespace coin_flip_probability_l164_164491

noncomputable def probability_successful_outcomes : ℚ :=
  let total_outcomes := 32
  let successful_outcomes := 3
  successful_outcomes / total_outcomes

theorem coin_flip_probability :
  probability_successful_outcomes = 3 / 32 :=
by
  sorry

end coin_flip_probability_l164_164491


namespace percentage_decrease_in_y_when_x_doubles_l164_164310

variable {k x y : ℝ}
variable (h_pos_x : 0 < x) (h_pos_y : 0 < y)
variable (inverse_proportional : x * y = k)

theorem percentage_decrease_in_y_when_x_doubles :
  (x' = 2 * x) →
  (y' = y / 2) →
  (100 * (y - y') / y) = 50 :=
by
  intro h1 h2
  simp [h1, h2]
  sorry

end percentage_decrease_in_y_when_x_doubles_l164_164310


namespace range_of_m_for_quadratic_sol_in_interval_l164_164763

theorem range_of_m_for_quadratic_sol_in_interval :
  {m : ℝ // ∀ x, (x^2 + (m-1)*x + 1 = 0) → (0 ≤ x ∧ x ≤ 2)} = {m : ℝ // m < -1} :=
by
  sorry

end range_of_m_for_quadratic_sol_in_interval_l164_164763


namespace N_positive_l164_164932

def N (a b : ℝ) : ℝ :=
  4 * a^2 - 12 * a * b + 13 * b^2 - 6 * a + 4 * b + 13

theorem N_positive (a b : ℝ) : N a b > 0 :=
by
  sorry

end N_positive_l164_164932


namespace phone_plan_cost_equal_at_2500_l164_164431

-- We define the costs C1 and C2 as described in the problem conditions.
def C1 (x : ℕ) : ℝ :=
  if x <= 500 then 50 else 50 + 0.35 * (x - 500)

def C2 (x : ℕ) : ℝ :=
  if x <= 1000 then 75 else 75 + 0.45 * (x - 1000)

-- We need to prove that the costs are equal when x = 2500.
theorem phone_plan_cost_equal_at_2500 : C1 2500 = C2 2500 := by
  sorry

end phone_plan_cost_equal_at_2500_l164_164431


namespace mural_width_l164_164175

theorem mural_width (l p r c t w : ℝ) (h₁ : l = 6) (h₂ : p = 4) (h₃ : r = 1.5) (h₄ : c = 10) (h₅ : t = 192) :
  4 * 6 * w + 10 * (6 * w / 1.5) = 192 → w = 3 :=
by
  intros
  sorry

end mural_width_l164_164175


namespace birthday_money_l164_164889

theorem birthday_money (x : ℤ) (h₀ : 16 + x - 25 = 19) : x = 28 :=
by
  sorry

end birthday_money_l164_164889


namespace ellipse_graph_equivalence_l164_164747

theorem ellipse_graph_equivalence :
  ∀ x y : ℝ, x^2 + 4 * y^2 - 6 * x + 8 * y + 9 = 0 ↔ (x - 3)^2 / 4 + (y + 1)^2 / 1 = 1 := by
  sorry

end ellipse_graph_equivalence_l164_164747


namespace return_time_possibilities_l164_164939

variables (d v w : ℝ) (t_return : ℝ)

-- Condition 1: Flight against wind takes 84 minutes
axiom flight_against_wind : d / (v - w) = 84

-- Condition 2: Return trip with wind takes 9 minutes less than without wind
axiom return_wind_condition : d / (v + w) = d / v - 9

-- Problem Statement: Find the possible return times
theorem return_time_possibilities :
  t_return = d / (v + w) → t_return = 63 ∨ t_return = 12 :=
sorry

end return_time_possibilities_l164_164939


namespace arithmetic_sequence_problem_l164_164998

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h1 : ∀ k, k ≥ 2 → a (k + 1) - a k^2 + a (k - 1) = 0) (h2 : ∀ k, a k ≠ 0) (h3 : ∀ k ≥ 2, a (k + 1) + a (k - 1) = 2 * a k) :
  S (2 * n - 1) - 4 * n = -2 :=
by
  sorry

end arithmetic_sequence_problem_l164_164998


namespace last_three_digits_of_7_pow_83_l164_164921

theorem last_three_digits_of_7_pow_83 :
  (7 ^ 83) % 1000 = 886 := sorry

end last_three_digits_of_7_pow_83_l164_164921


namespace width_of_lawn_is_30_m_l164_164595

-- Define the conditions
def lawn_length : ℕ := 70
def lawn_width : ℕ := 30
def road_width : ℕ := 5
def gravel_rate_per_sqm : ℕ := 4
def gravel_cost : ℕ := 1900

-- Mathematically equivalent proof problem statement
theorem width_of_lawn_is_30_m 
  (H1 : lawn_length = 70)
  (H2 : road_width = 5)
  (H3 : gravel_rate_per_sqm = 4)
  (H4 : gravel_cost = 1900)
  (H5 : 2*road_width*5 + (lawn_length - road_width) * 5 * gravel_rate_per_sqm = gravel_cost) :
  lawn_width = 30 := 
sorry

end width_of_lawn_is_30_m_l164_164595


namespace speeds_of_bodies_l164_164274

theorem speeds_of_bodies 
  (v1 v2 : ℝ)
  (h1 : 21 * v1 + 10 * v2 = 270)
  (h2 : 51 * v1 + 40 * v2 = 540)
  (h3 : 5 * v2 = 3 * v1): 
  v1 = 10 ∧ v2 = 6 :=
by
  sorry

end speeds_of_bodies_l164_164274


namespace cost_price_of_article_l164_164736

theorem cost_price_of_article 
  (CP SP : ℝ)
  (H1 : SP = 1.13 * CP)
  (H2 : 1.10 * SP = 616) :
  CP = 495.58 :=
by
  sorry

end cost_price_of_article_l164_164736


namespace box_height_is_6_l164_164157

-- Defining the problem setup
variables (h : ℝ) (r_large r_small : ℝ)
variables (box_size : ℝ) (n_spheres : ℕ)

-- The conditions of the problem
def rectangular_box :=
  box_size = 5 ∧ r_large = 3 ∧ r_small = 1.5 ∧ n_spheres = 4 ∧
  (∀ k : ℕ, k < n_spheres → 
   ∃ C : ℝ, 
     (C = r_small) ∧ 
     -- Each smaller sphere is tangent to three sides of the box condition
     (C ≤ box_size))

def sphere_tangency (h r_large r_small : ℝ) :=
  h = 2 * r_large ∧ r_large + r_small = 4.5

def height_of_box (h : ℝ) := 2 * 3 = h

-- The mathematically equivalent proof problem
theorem box_height_is_6 (h : ℝ) (r_large : ℝ) (r_small : ℝ) (box_size : ℝ) (n_spheres : ℕ) 
  (conditions : rectangular_box box_size r_large r_small n_spheres) 
  (tangency : sphere_tangency h r_large r_small) :
  height_of_box h :=
by {
  -- Proof is omitted
  sorry
}

end box_height_is_6_l164_164157


namespace find_a_l164_164373

theorem find_a (a : ℝ) 
  (h1 : a < 0)
  (h2 : a < 1/3)
  (h3 : -2 * a + (1 - 3 * a) = 6) : 
  a = -1 := 
by 
  sorry

end find_a_l164_164373


namespace true_proposition_l164_164167

-- Define propositions p and q
variable (p q : Prop)

-- Assume p is true and q is false
axiom h1 : p
axiom h2 : ¬q

-- Prove that p ∧ ¬q is true
theorem true_proposition (p q : Prop) (h1 : p) (h2 : ¬q) : p ∧ ¬q :=
by
  sorry

end true_proposition_l164_164167


namespace relationship_m_n_l164_164471

theorem relationship_m_n (m n : ℕ) (h : 10 / (m + 10 + n) = (m + n) / (m + 10 + n)) : m + n = 10 := 
by sorry

end relationship_m_n_l164_164471


namespace games_needed_in_single_elimination_l164_164592

theorem games_needed_in_single_elimination (teams : ℕ) (h : teams = 23) : 
  ∃ games : ℕ, games = teams - 1 ∧ games = 22 :=
by
  existsi (teams - 1)
  sorry

end games_needed_in_single_elimination_l164_164592


namespace probability_first_4_second_club_third_2_l164_164927

theorem probability_first_4_second_club_third_2 :
  let deck_size := 52
  let prob_4_first := 4 / deck_size
  let deck_minus_first_card := deck_size - 1
  let prob_club_second := 13 / deck_minus_first_card
  let deck_minus_two_cards := deck_minus_first_card - 1
  let prob_2_third := 4 / deck_minus_two_cards
  prob_4_first * prob_club_second * prob_2_third = 1 / 663 :=
by
  sorry

end probability_first_4_second_club_third_2_l164_164927


namespace comparison_of_a_b_c_l164_164972

theorem comparison_of_a_b_c (a b c : ℝ) (h_a : a = Real.log 2) (h_b : b = 5^(-1/2 : ℝ)) (h_c : c = Real.sin (Real.pi / 6)) : 
  b < c ∧ c < a :=
by
  sorry

end comparison_of_a_b_c_l164_164972


namespace company_fund_initial_amount_l164_164510

-- Let n be the number of employees in the company.
variable (n : ℕ)

-- Conditions from the problem.
def initial_fund := 60 * n - 10
def adjusted_fund := 50 * n + 150
def employees_count := 16

-- Given the conditions, prove that the initial fund amount was $950.
theorem company_fund_initial_amount
    (h1 : adjusted_fund n = initial_fund n)
    (h2 : n = employees_count) : 
    initial_fund n = 950 := by
  sorry

end company_fund_initial_amount_l164_164510


namespace probability_heads_all_three_tosses_l164_164423

theorem probability_heads_all_three_tosses :
  (1 / 2) * (1 / 2) * (1 / 2) = 1 / 8 := 
sorry

end probability_heads_all_three_tosses_l164_164423


namespace sum_of_two_terms_is_term_iff_a_is_multiple_of_d_l164_164705

theorem sum_of_two_terms_is_term_iff_a_is_multiple_of_d
    (a d : ℤ) 
    (n k : ℕ) 
    (h : ∀ (p : ℕ), a + d * n + (a + d * k) = a + d * p)
    : ∃ m : ℤ, a = d * m :=
sorry

end sum_of_two_terms_is_term_iff_a_is_multiple_of_d_l164_164705


namespace inequalities_hold_l164_164239

theorem inequalities_hold (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (b / a > c / a) ∧ (b - a) / c > 0 ∧ (a - c) / (a * c) < 0 :=
by 
  sorry

end inequalities_hold_l164_164239


namespace arithmetic_sequence_terms_l164_164130

theorem arithmetic_sequence_terms (a : ℕ → ℕ) (n : ℕ)
  (h1 : a 1 + a 2 + a 3 = 34)
  (h2 : a n + a (n - 1) + a (n - 2) = 146)
  (h3 : n * (a 1 + a n) = 780) : n = 13 :=
sorry

end arithmetic_sequence_terms_l164_164130


namespace original_number_not_800_l164_164365

theorem original_number_not_800 (x : ℕ) (h : 10 * x = x + 720) : x ≠ 800 :=
by {
  sorry
}

end original_number_not_800_l164_164365


namespace wheel_rotation_angle_l164_164517

-- Define the conditions
def radius : ℝ := 20
def arc_length : ℝ := 40

-- Define the theorem stating the desired proof problem
theorem wheel_rotation_angle (r : ℝ) (l : ℝ) (h_r : r = radius) (h_l : l = arc_length) :
  l / r = 2 := 
by sorry

end wheel_rotation_angle_l164_164517


namespace total_clothing_ironed_l164_164200

-- Definitions based on conditions
def shirts_per_hour := 4
def pants_per_hour := 3
def hours_ironing_shirts := 3
def hours_ironing_pants := 5

-- Theorem statement based on the problem and its solution
theorem total_clothing_ironed : 
  (shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants) = 27 := 
by
  sorry

end total_clothing_ironed_l164_164200


namespace solve_sausage_problem_l164_164055

def sausage_problem (x y : ℕ) (condition1 : y = x + 300) (condition2 : x = y + 500) : Prop :=
  x + y = 2 * 400

theorem solve_sausage_problem (x y : ℕ) (h1 : y = x + 300) (h2 : x = y + 500) :
  sausage_problem x y h1 h2 :=
by
  sorry

end solve_sausage_problem_l164_164055


namespace binom_subtract_l164_164069

theorem binom_subtract :
  (Nat.choose 7 4) - 5 = 30 :=
by
  -- proof goes here
  sorry

end binom_subtract_l164_164069


namespace hyperbola_condition_l164_164366

noncomputable def hyperbola_eccentricity_difference (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  let e_2pi_over_3 := Real.sqrt 3 + 1
  let e_pi_over_3 := (Real.sqrt 3) / 3 + 1
  e_2pi_over_3 - e_pi_over_3

theorem hyperbola_condition (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  hyperbola_eccentricity_difference a b h1 h2 = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end hyperbola_condition_l164_164366


namespace range_of_a_l164_164033

/-- Definitions for propositions p and q --/
def p (a : ℝ) : Prop := a > 0 ∧ a < 1
def q (a : ℝ) : Prop := (2 * a - 3) ^ 2 - 4 > 0

/-- Theorem stating the range of possible values for a given conditions --/
theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : ¬(p a) ∧ ¬(q a) = false) (h4 : p a ∨ q a) :
  (1 / 2 ≤ a ∧ a < 1) ∨ (a ≥ 5 / 2) :=
sorry

end range_of_a_l164_164033


namespace process_can_continue_indefinitely_l164_164808

noncomputable def P (x : ℝ) : ℝ := x^3 - x^2 - x - 1

-- Assume the existence of t > 1 such that P(t) = 0
axiom exists_t : ∃ t : ℝ, t > 1 ∧ P t = 0

def triangle_inequality_fails (a b c : ℝ) : Prop :=
  ¬(a + b > c ∧ b + c > a ∧ c + a > b)

def shorten (a b : ℝ) : ℝ := a + b

def can_continue_indefinitely (a b c : ℝ) : Prop :=
  ∀ t, t > 0 → ∀ a b c, triangle_inequality_fails a b c → 
  (triangle_inequality_fails (shorten b c - shorten a b) b c ∧
   triangle_inequality_fails a (shorten a c - shorten b c) c ∧
   triangle_inequality_fails a b (shorten a b - shorten b c))

theorem process_can_continue_indefinitely (a b c : ℝ) (h : triangle_inequality_fails a b c) :
  can_continue_indefinitely a b c :=
sorry

end process_can_continue_indefinitely_l164_164808


namespace solve_inequality_l164_164692

theorem solve_inequality :
  { x : ℝ | (9 * x^2 + 27 * x - 64) / ((3 * x - 4) * (x + 5) * (x - 1)) < 4 } = 
    { x : ℝ | -5 < x ∧ x < -17 / 3 } ∪ { x : ℝ | 1 < x ∧ x < 4 } :=
by
  sorry

end solve_inequality_l164_164692


namespace problem1_monotonic_decreasing_problem2_monotonic_decreasing_pos_problem2_monotonic_decreasing_neg_l164_164388

-- Problem 1: Monotonicity of f(x) = 1 - 3x on ℝ
theorem problem1_monotonic_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → (1 - 3 * x1) > (1 - 3 * x2) :=
by
  -- Proof (skipped)
  sorry

-- Problem 2: Monotonicity of g(x) = 1/x + 2 on (0, ∞) and (-∞, 0)
theorem problem2_monotonic_decreasing_pos : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → (1 / x1 + 2) > (1 / x2 + 2) :=
by
  -- Proof (skipped)
  sorry

theorem problem2_monotonic_decreasing_neg : ∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → (1 / x1 + 2) > (1 / x2 + 2) :=
by
  -- Proof (skipped)
  sorry

end problem1_monotonic_decreasing_problem2_monotonic_decreasing_pos_problem2_monotonic_decreasing_neg_l164_164388


namespace compare_fractions_l164_164368

theorem compare_fractions (a b m : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : m > 0) : 
  (b / a) < ((b + m) / (a + m)) :=
sorry

end compare_fractions_l164_164368


namespace atomic_weight_of_Calcium_l164_164163

/-- Given definitions -/
def molecular_weight_CaOH₂ : ℕ := 74
def atomic_weight_O : ℕ := 16
def atomic_weight_H : ℕ := 1

/-- Given conditions -/
def total_weight_O_H : ℕ := 2 * atomic_weight_O + 2 * atomic_weight_H

/-- Problem statement -/
theorem atomic_weight_of_Calcium (H1 : molecular_weight_CaOH₂ = 74)
                                   (H2 : atomic_weight_O = 16)
                                   (H3 : atomic_weight_H = 1)
                                   (H4 : total_weight_O_H = 2 * atomic_weight_O + 2 * atomic_weight_H) :
  74 - (2 * 16 + 2 * 1) = 40 :=
by {
  sorry
}

end atomic_weight_of_Calcium_l164_164163


namespace product_of_numbers_l164_164499

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 72) (h2 : x - y = 20) : x * y = 1196 := 
sorry

end product_of_numbers_l164_164499


namespace louisa_second_day_distance_l164_164719

-- Definitions based on conditions
def time_on_first_day (distance : ℕ) (speed : ℕ) : ℕ := distance / speed
def time_on_second_day (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

def condition (distance_first_day : ℕ) (speed : ℕ) (time_difference : ℕ) (x : ℕ) : Prop := 
  time_on_first_day distance_first_day speed + time_difference = time_on_second_day x speed

-- The proof statement
theorem louisa_second_day_distance (distance_first_day : ℕ) (speed : ℕ) (time_difference : ℕ) (x : ℕ) :
  distance_first_day = 240 → 
  speed = 60 → 
  time_difference = 3 → 
  condition distance_first_day speed time_difference x → 
  x = 420 :=
by
  intros h1 h2 h3 h4
  sorry

end louisa_second_day_distance_l164_164719


namespace problem_statement_l164_164107

section

variable {f : ℝ → ℝ}

-- Conditions
axiom even_function (h : ∀ x : ℝ, f (-x) = f x) : ∀ x, f (-x) = f x 
axiom monotonically_increasing (h : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y) :
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

-- Goal
theorem problem_statement 
  (h_even : ∀ x, f (-x) = f x)
  (h_mono : ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y) :
  f (-Real.log 2 / Real.log 3) > f (Real.log 2 / Real.log 3) ∧ f (Real.log 2 / Real.log 3) > f 0 := 
sorry

end

end problem_statement_l164_164107


namespace fourth_person_knight_l164_164380

-- Let P1, P2, P3, and P4 be the statements made by the four people respectively.
def P1 := ∀ x y z w : Prop, x = y ∧ y = z ∧ z = w ∧ w = ¬w
def P2 := ∃! x y z w : Prop, x = true
def P3 := ∀ x y z w : Prop, (x = true ∧ y = true ∧ z = false) ∨ (x = true ∧ y = false ∧ z = true) ∨ (x = false ∧ y = true ∧ z = true)
def P4 := ∀ x : Prop, x = true → x = true

-- Now let's express the requirement of proving that the fourth person is a knight
theorem fourth_person_knight : P4 := by
  sorry

end fourth_person_knight_l164_164380


namespace max_abs_x_minus_2y_plus_1_l164_164545

theorem max_abs_x_minus_2y_plus_1 (x y : ℝ) (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) :
  |x - 2 * y + 1| ≤ 5 :=
sorry

end max_abs_x_minus_2y_plus_1_l164_164545


namespace quadratic_roots_transformation_l164_164555

theorem quadratic_roots_transformation {a b c r s : ℝ}
  (h1 : r + s = -b / a)
  (h2 : r * s = c / a) :
  (∃ p q : ℝ, p = a * r + 2 * b ∧ q = a * s + 2 * b ∧ 
     (∀ x, x^2 - 3 * b * x + 2 * b^2 + a * c = (x - p) * (x - q))) :=
by
  sorry

end quadratic_roots_transformation_l164_164555


namespace arithmetic_sequence_fifth_term_l164_164636

noncomputable def fifth_term (x y : ℝ) : ℝ :=
  let a1 := x^2 + y^2
  let a2 := x^2 - y^2
  let a3 := x^2 * y^2
  let a4 := x^2 / y^2
  let d := -2 * y^2
  a4 + d

theorem arithmetic_sequence_fifth_term (x y : ℝ) (hy : y ≠ 0) (hx2 : x ^ 2 = 3 * y ^ 2 / (y ^ 2 - 1)) :
  fifth_term x y = 3 / (y ^ 2 - 1) - 2 * y ^ 2 :=
by
  sorry

end arithmetic_sequence_fifth_term_l164_164636


namespace emily_and_berengere_contribution_l164_164606

noncomputable def euro_to_usd : ℝ := 1.20
noncomputable def euro_to_gbp : ℝ := 0.85

noncomputable def cake_cost_euros : ℝ := 12
noncomputable def cookies_cost_euros : ℝ := 5
noncomputable def total_cost_euros : ℝ := cake_cost_euros + cookies_cost_euros

noncomputable def emily_usd : ℝ := 10
noncomputable def liam_gbp : ℝ := 10

noncomputable def emily_euros : ℝ := emily_usd / euro_to_usd
noncomputable def liam_euros : ℝ := liam_gbp / euro_to_gbp

noncomputable def total_available_euros : ℝ := emily_euros + liam_euros

theorem emily_and_berengere_contribution : total_available_euros >= total_cost_euros := by
  sorry

end emily_and_berengere_contribution_l164_164606


namespace workers_to_build_cars_l164_164682

theorem workers_to_build_cars (W : ℕ) (hW : W > 0) : 
  (∃ D : ℝ, D = 63 / W) :=
by
  sorry

end workers_to_build_cars_l164_164682


namespace minimum_value_expression_l164_164770

open Real

theorem minimum_value_expression : ∃ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 = 2018 := 
sorry

end minimum_value_expression_l164_164770


namespace mother_l164_164715

theorem mother's_age (D M : ℕ) (h1 : 2 * D + M = 70) (h2 : D + 2 * M = 95) : M = 40 :=
sorry

end mother_l164_164715


namespace erin_serves_all_soup_in_15_minutes_l164_164321

noncomputable def time_to_serve_all_soup
  (ounces_per_bowl : ℕ)
  (bowls_per_minute : ℕ)
  (soup_in_gallons : ℕ)
  (ounces_per_gallon : ℕ) : ℕ :=
  let total_ounces := soup_in_gallons * ounces_per_gallon
  let total_bowls := (total_ounces + ounces_per_bowl - 1) / ounces_per_bowl -- to round up
  let total_minutes := (total_bowls + bowls_per_minute - 1) / bowls_per_minute -- to round up
  total_minutes

theorem erin_serves_all_soup_in_15_minutes :
  time_to_serve_all_soup 10 5 6 128 = 15 :=
sorry

end erin_serves_all_soup_in_15_minutes_l164_164321


namespace maximize_garden_area_length_l164_164753

noncomputable def length_parallel_to_wall (cost_per_foot : ℝ) (fence_cost : ℝ) : ℝ :=
  let total_length := fence_cost / cost_per_foot 
  let y := total_length / 4 
  let length_parallel := total_length - 2 * y
  length_parallel

theorem maximize_garden_area_length :
  ∀ (cost_per_foot fence_cost : ℝ), cost_per_foot = 10 → fence_cost = 1500 → 
  length_parallel_to_wall cost_per_foot fence_cost = 75 :=
by
  intros
  simp [length_parallel_to_wall, *]
  sorry

end maximize_garden_area_length_l164_164753


namespace variables_and_unknowns_l164_164042

theorem variables_and_unknowns (f_1 f_2: ℝ → ℝ → ℝ) (f: ℝ → ℝ → ℝ) :
  (∀ x y, f_1 x y = 0 ∧ f_2 x y = 0 → (x ≠ 0 ∨ y ≠ 0)) ∧
  (∀ x y, f x y = 0 → (∃ a b, x = a ∧ y = b)) :=
by sorry

end variables_and_unknowns_l164_164042


namespace contrapositive_of_square_comparison_l164_164295

theorem contrapositive_of_square_comparison (x y : ℝ) : (x^2 > y^2 → x > y) → (x ≤ y → x^2 ≤ y^2) :=
  by sorry

end contrapositive_of_square_comparison_l164_164295


namespace larger_integer_is_50_l164_164140

-- Definition of the problem conditions.
def is_two_digit (x : ℕ) : Prop := 10 ≤ x ∧ x ≤ 99

def problem_conditions (m n : ℕ) : Prop := 
  is_two_digit m ∧ is_two_digit n ∧
  (m + n) / 2 = m + n / 100

-- Statement of the proof problem.
theorem larger_integer_is_50 (m n : ℕ) (h : problem_conditions m n) : max m n = 50 :=
  sorry

end larger_integer_is_50_l164_164140


namespace percent_divisible_by_six_up_to_120_l164_164571

theorem percent_divisible_by_six_up_to_120 : 
  let total_numbers := 120
  let divisible_by_six := total_numbers / 6
  let percentage := (divisible_by_six * 100) / total_numbers
  percentage = 50 / 3 := sorry

end percent_divisible_by_six_up_to_120_l164_164571


namespace hilltop_high_students_l164_164669

theorem hilltop_high_students : 
  ∀ (n_sophomore n_freshman n_junior : ℕ), 
  (n_sophomore : ℚ) / n_freshman = 7 / 4 ∧ (n_junior : ℚ) / n_sophomore = 6 / 7 → 
  n_sophomore + n_freshman + n_junior = 17 :=
by
  sorry

end hilltop_high_students_l164_164669


namespace algebra_ineq_l164_164622

theorem algebra_ineq (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : a * b + b * c + c * a = 1) : a + b + c ≥ 2 := 
by sorry

end algebra_ineq_l164_164622


namespace part1_part2_part3_l164_164632

noncomputable def a (n : ℕ) : ℝ := 
if n = 1 then 1 else 
if n = 2 then 3/2 else 
if n = 3 then 5/4 else 
sorry

noncomputable def S (n : ℕ) : ℝ := sorry

axiom recurrence {n : ℕ} (h : n ≥ 2) : 4 * S (n + 2) + 5 * S n = 8 * S (n + 1) + S (n - 1)

-- Part 1
theorem part1 : a 4 = 7 / 8 :=
sorry

-- Part 2
theorem part2 : ∃ (r : ℝ) (b : ℕ → ℝ), (r = 1/2) ∧ (∀ n ≥ 1, a (n + 1) - r * a n = b n) :=
sorry

-- Part 3
theorem part3 : ∀ n, a n = (2 * n - 1) / 2^(n - 1) :=
sorry

end part1_part2_part3_l164_164632


namespace solve_inequality_part1_solve_inequality_part2_l164_164590

-- Define the first part of the problem
theorem solve_inequality_part1 (a : ℝ) (x : ℝ) :
  (x^2 - a * x - 2 * a^2 < 0) ↔ 
    (a = 0 ∧ false) ∨ 
    (a > 0 ∧ -a < x ∧ x < 2 * a) ∨ 
    (a < 0 ∧ 2 * a < x ∧ x < -a) := 
sorry

-- Define the second part of the problem
theorem solve_inequality_part2 (a b : ℝ) (x : ℝ) 
  (h : { x | x^2 - a * x - b < 0 } = { x | -1 < x ∧ x < 2 }) :
  { x | a * x^2 + x - b > 0 } = { x | x < -2 } ∪ { x | 1 < x } :=
sorry

end solve_inequality_part1_solve_inequality_part2_l164_164590


namespace possible_galina_numbers_l164_164262

def is_divisible_by (m n : ℕ) : Prop := n % m = 0

def conditions_for_galina_number (n : ℕ) : Prop :=
  let C1 := is_divisible_by 7 n
  let C2 := is_divisible_by 11 n
  let C3 := n < 13
  let C4 := is_divisible_by 77 n
  (C1 ∧ ¬C2 ∧ C3 ∧ ¬C4) ∨ (¬C1 ∧ C2 ∧ C3 ∧ ¬C4)

theorem possible_galina_numbers (n : ℕ) :
  conditions_for_galina_number n ↔ (n = 7 ∨ n = 11) :=
by
  -- Proof to be filled in
  sorry

end possible_galina_numbers_l164_164262


namespace train_length_l164_164405

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (length : ℝ) : 
  speed_kmph = 60 → time_sec = 12 → 
  length = speed_kmph * (1000 / 3600) * time_sec → 
  length = 200.04 :=
by
  intros h_speed h_time h_length
  sorry

end train_length_l164_164405


namespace find_n_plus_m_l164_164862

noncomputable def f (x : ℝ) := abs (Real.log x / Real.log 2)

theorem find_n_plus_m (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m < n)
    (h4 : f m = f n) (h5 : ∀ x, m^2 ≤ x ∧ x ≤ n → f x ≤ 2) :
    n + m = 5 / 2 := sorry

end find_n_plus_m_l164_164862


namespace jars_of_plum_jelly_sold_l164_164314

theorem jars_of_plum_jelly_sold (P R G S : ℕ) (h1 : R = 2 * P) (h2 : G = 3 * R) (h3 : G = 2 * S) (h4 : S = 18) : P = 6 := by
  sorry

end jars_of_plum_jelly_sold_l164_164314


namespace added_number_is_four_l164_164996

theorem added_number_is_four :
  ∃ x y, 2 * x < 3 * x ∧ (3 * x - 2 * x = 8) ∧ 
         ((2 * x + y) * 7 = 5 * (3 * x + y)) ∧ y = 4 :=
  sorry

end added_number_is_four_l164_164996


namespace log_216_eq_3_log_2_add_3_log_3_l164_164689

theorem log_216_eq_3_log_2_add_3_log_3 (log : ℝ → ℝ) (h1 : ∀ x y, log (x * y) = log x + log y)
  (h2 : ∀ x n, log (x^n) = n * log x) :
  log 216 = 3 * log 2 + 3 * log 3 :=
by
  sorry

end log_216_eq_3_log_2_add_3_log_3_l164_164689


namespace math_problem_l164_164908

theorem math_problem (a b c d x : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |x| = 2) :
  x^4 + c * d * x^2 - a - b = 20 :=
sorry

end math_problem_l164_164908


namespace num_real_roots_eq_two_l164_164652

theorem num_real_roots_eq_two : 
  ∀ x : ℝ, (∃ r : ℕ, r = 2 ∧ (abs (x^2 - 1) = 1/10 * (x + 9/10) → x = r)) := sorry

end num_real_roots_eq_two_l164_164652


namespace range_of_a_l164_164853

def A : Set ℝ := { x | x^2 - 2 * x - 3 ≤ 0 }

theorem range_of_a (a : ℝ) (h : a ∈ A) : -1 ≤ a ∧ a ≤ 3 := 
sorry

end range_of_a_l164_164853


namespace fans_received_all_items_l164_164673

theorem fans_received_all_items (n : ℕ) (h1 : (∀ k : ℕ, k * 45 ≤ n → (k * 45) ∣ n))
                                (h2 : (∀ k : ℕ, k * 50 ≤ n → (k * 50) ∣ n))
                                (h3 : (∀ k : ℕ, k * 100 ≤ n → (k * 100) ∣ n))
                                (capacity_full : n = 5000) :
  n / Nat.lcm 45 (Nat.lcm 50 100) = 5 :=
by
  sorry

end fans_received_all_items_l164_164673


namespace solve_inequality_inequality_proof_l164_164338

-- Problem 1: Solve the inequality |2x+1| - |x-4| > 2
theorem solve_inequality (x : ℝ) :
  (|2 * x + 1| - |x - 4| > 2) ↔ (x < -7 ∨ x > (5/3)) :=
sorry

-- Problem 2: Prove the inequality given a > 0 and b > 0
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / Real.sqrt b + b / Real.sqrt a) ≥ (Real.sqrt a + Real.sqrt b) :=
sorry

end solve_inequality_inequality_proof_l164_164338


namespace sin_240_eq_neg_sqrt3_div_2_l164_164345

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l164_164345


namespace all_blue_figures_are_small_l164_164197

variables (Shape : Type) (Large Blue Small Square Triangle : Shape → Prop)

-- Given conditions
axiom h1 : ∀ (x : Shape), Large x → Square x
axiom h2 : ∀ (x : Shape), Blue x → Triangle x

-- The goal to prove
theorem all_blue_figures_are_small : ∀ (x : Shape), Blue x → Small x :=
by
  sorry

end all_blue_figures_are_small_l164_164197


namespace tim_initial_books_l164_164257

def books_problem : Prop :=
  ∃ T : ℕ, 10 + T - 24 = 19 ∧ T = 33

theorem tim_initial_books : books_problem :=
  sorry

end tim_initial_books_l164_164257


namespace monotonicity_of_even_function_l164_164400

-- Define the function and its properties
def f (m : ℝ) (x : ℝ) : ℝ := (m-1)*x^2 + 2*m*x + 3

-- A function is even if f(x) = f(-x) for all x
def is_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = g (-x)

-- The main theorem statement
theorem monotonicity_of_even_function :
  ∀ (m : ℝ), is_even (f m) → (f 0 = 3) ∧ (∀ x : ℝ, f 0 x = - x^2 + 3) →
  (∀ a b, -3 < a ∧ a < b ∧ b < 1 → f 0 a < f 0 b → f 0 b > f 0 a) :=
by
  intro m
  intro h
  intro H
  sorry

end monotonicity_of_even_function_l164_164400


namespace range_of_m_values_l164_164572

theorem range_of_m_values {P Q : ℝ × ℝ} (hP : P = (-1, 1)) (hQ : Q = (2, 2)) (m : ℝ) :
  -3 < m ∧ m < -2 / 3 → (∃ (l : ℝ → ℝ), ∀ x y, y = l x → x + m * y + m = 0) :=
sorry

end range_of_m_values_l164_164572


namespace range_of_a_l164_164135

-- Define the set M
def M : Set ℝ := { x | -1 ≤ x ∧ x < 2 }

-- Define the set N
def N (a : ℝ) : Set ℝ := { x | x ≤ a }

-- The theorem to be proved
theorem range_of_a (a : ℝ) (h : (M ∩ N a).Nonempty) : a ≥ -1 := sorry

end range_of_a_l164_164135


namespace length_of_train_l164_164386

theorem length_of_train :
  ∀ (L : ℝ) (V : ℝ),
  (∀ t p : ℝ, t = 14 → p = 535.7142857142857 → V = L / t) →
  (∀ t p : ℝ, t = 39 → p = 535.7142857142857 → V = (L + p) / t) →
  L = 300 :=
by
  sorry

end length_of_train_l164_164386


namespace no_sensor_in_option_B_l164_164552

/-- Define the technologies and whether they involve sensors --/
def technology_involves_sensor (opt : String) : Prop :=
  opt = "A" ∨ opt = "C" ∨ opt = "D"

theorem no_sensor_in_option_B :
  ¬ technology_involves_sensor "B" :=
by
  -- We assume the proof for the sake of this example.
  sorry

end no_sensor_in_option_B_l164_164552


namespace regular_polygon_sides_l164_164820

theorem regular_polygon_sides (θ : ℝ) (h : θ = 20) : 360 / θ = 18 := by
  sorry

end regular_polygon_sides_l164_164820


namespace initial_production_rate_l164_164336

theorem initial_production_rate 
  (x : ℝ)
  (h1 : 60 <= (60 * x) / 30 - 60 + 1800)
  (h2 : 60 <= 120)
  (h3 : 30 = (120 / (60 / x + 1))) : x = 20 := by
  sorry

end initial_production_rate_l164_164336


namespace product_pass_rate_l164_164325

variable (a b : ℝ)

theorem product_pass_rate (h1 : 0 ≤ a) (h2 : a < 1) (h3 : 0 ≤ b) (h4 : b < 1) : 
  (1 - a) * (1 - b) = 1 - (a + b - a * b) :=
by sorry

end product_pass_rate_l164_164325


namespace sum_of_reciprocals_l164_164791

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 5 * x * y) : 
  (1/x) + (1/y) = 5 :=
by
  sorry

end sum_of_reciprocals_l164_164791


namespace radius_of_tangent_circle_l164_164766

theorem radius_of_tangent_circle (k r : ℝ) (hk : k > 8) (h1 : k - 8 = r) (h2 : r * Real.sqrt 2 = k) : 
  r = 8 * (Real.sqrt 2 + 1) := 
sorry

end radius_of_tangent_circle_l164_164766


namespace students_exceed_guinea_pigs_l164_164099

theorem students_exceed_guinea_pigs :
  let classrooms := 5
  let students_per_classroom := 20
  let guinea_pigs_per_classroom := 3
  let total_students := classrooms * students_per_classroom
  let total_guinea_pigs := classrooms * guinea_pigs_per_classroom
  total_students - total_guinea_pigs = 85 :=
by
  -- using the conditions and correct answer identified above
  let classrooms := 5
  let students_per_classroom := 20
  let guinea_pigs_per_classroom := 3
  let total_students := classrooms * students_per_classroom
  let total_guinea_pigs := classrooms * guinea_pigs_per_classroom
  show total_students - total_guinea_pigs = 85
  sorry

end students_exceed_guinea_pigs_l164_164099


namespace total_goals_proof_l164_164007

-- Definitions based on the conditions
def first_half_team_a := 8
def first_half_team_b := first_half_team_a / 2
def first_half_team_c := first_half_team_b * 2

def second_half_team_a := first_half_team_c
def second_half_team_b := first_half_team_a
def second_half_team_c := second_half_team_b + 3

-- Total scores for each team
def total_team_a := first_half_team_a + second_half_team_a
def total_team_b := first_half_team_b + second_half_team_b
def total_team_c := first_half_team_c + second_half_team_c

-- Total goals for all teams
def total_goals := total_team_a + total_team_b + total_team_c

-- The theorem to be proved
theorem total_goals_proof : total_goals = 47 := by
  sorry

end total_goals_proof_l164_164007


namespace sum_of_largest_and_smallest_l164_164227

theorem sum_of_largest_and_smallest (a b c : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) :
  a + c = 22 :=
by
  sorry

end sum_of_largest_and_smallest_l164_164227


namespace sum_in_range_l164_164731

theorem sum_in_range : 
    let a := (2:ℝ) + 1/8
    let b := (3:ℝ) + 1/3
    let c := (5:ℝ) + 1/18
    10.5 < a + b + c ∧ a + b + c < 11 := 
by 
    sorry

end sum_in_range_l164_164731


namespace sum_of_numbers_l164_164229

theorem sum_of_numbers (x : ℕ) (first_num second_num third_num sum : ℕ) 
  (h1 : 5 * x = first_num) 
  (h2 : 3 * x = second_num)
  (h3 : 4 * x = third_num) 
  (h4 : second_num = 27)
  : first_num + second_num + third_num = 108 :=
by {
  sorry
}

end sum_of_numbers_l164_164229


namespace Jacob_eats_more_calories_than_planned_l164_164351

theorem Jacob_eats_more_calories_than_planned 
  (planned_calories : ℕ) (actual_calories : ℕ)
  (h1 : planned_calories < 1800) 
  (h2 : actual_calories = 400 + 900 + 1100)
  : actual_calories - planned_calories = 600 := by
  sorry

end Jacob_eats_more_calories_than_planned_l164_164351


namespace range_of_a_l164_164103

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 1| - |x - 2| < a^2 + a + 1) →
  (a < -1 ∨ a > 0) :=
by
  sorry

end range_of_a_l164_164103


namespace max_side_of_triangle_exists_max_side_of_elevent_l164_164412

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l164_164412


namespace probability_complement_l164_164313

theorem probability_complement (p : ℝ) (h : p = 0.997) : 1 - p = 0.003 :=
by
  rw [h]
  norm_num

end probability_complement_l164_164313


namespace arithmetic_sequence_n_value_l164_164786

theorem arithmetic_sequence_n_value (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 3) :
  a 672 = 2014 :=
sorry

end arithmetic_sequence_n_value_l164_164786


namespace wrapping_paper_area_l164_164337

theorem wrapping_paper_area (length width : ℕ) (h1 : width = 6) (h2 : 2 * (length + width) = 28) : length * width = 48 :=
by
  sorry

end wrapping_paper_area_l164_164337


namespace sum_divisible_by_15_l164_164858

theorem sum_divisible_by_15 (a : ℤ) : 15 ∣ (9 * a^5 - 5 * a^3 - 4 * a) :=
sorry

end sum_divisible_by_15_l164_164858


namespace cubic_function_decreasing_l164_164169

-- Define the given function
def f (a x : ℝ) : ℝ := a * x^3 - 1

-- Define the condition that the function is decreasing on ℝ
def is_decreasing_on_R (a : ℝ) : Prop :=
  ∀ x : ℝ, 3 * a * x^2 ≤ 0 

-- Main theorem and its statement
theorem cubic_function_decreasing (a : ℝ) (h : is_decreasing_on_R a) : a < 0 :=
sorry

end cubic_function_decreasing_l164_164169


namespace frank_fencemaker_fence_length_l164_164548

theorem frank_fencemaker_fence_length :
  ∃ (L W : ℕ), W = 40 ∧
               (L * W = 200) ∧
               (2 * L + W = 50) :=
by
  sorry

end frank_fencemaker_fence_length_l164_164548


namespace probability_of_B_l164_164724

theorem probability_of_B (P : Set ℕ → ℝ) (A B : Set ℕ) (hA : P A = 0.25) (hAB : P (A ∩ B) = 0.15) (hA_complement_B_complement : P (Aᶜ ∩ Bᶜ) = 0.5) : P B = 0.4 :=
by
  sorry

end probability_of_B_l164_164724


namespace probability_of_drawing_red_ball_l164_164444

/-- Define the colors of the balls in the bag -/
def yellow_balls : ℕ := 2
def red_balls : ℕ := 3
def white_balls : ℕ := 5

/-- Define the total number of balls in the bag -/
def total_balls : ℕ := yellow_balls + red_balls + white_balls

/-- Define the probability of drawing exactly one red ball -/
def probability_of_red_ball : ℚ := red_balls / total_balls

/-- The main theorem to prove the given problem -/
theorem probability_of_drawing_red_ball :
  probability_of_red_ball = 3 / 10 :=
by
  -- Calculation steps would go here, but are omitted
  sorry

end probability_of_drawing_red_ball_l164_164444


namespace original_paint_intensity_l164_164460

theorem original_paint_intensity (I : ℝ) (h1 : 0.5 * I + 0.5 * 20 = 15) : I = 10 :=
sorry

end original_paint_intensity_l164_164460


namespace egg_production_difference_l164_164174

def eggs_last_year : ℕ := 1416
def eggs_this_year : ℕ := 4636
def eggs_difference (a b : ℕ) : ℕ := a - b

theorem egg_production_difference : eggs_difference eggs_this_year eggs_last_year = 3220 := 
by
  sorry

end egg_production_difference_l164_164174


namespace students_neither_l164_164054

-- Define the given conditions
def total_students : Nat := 460
def football_players : Nat := 325
def cricket_players : Nat := 175
def both_players : Nat := 90

-- Define the Lean statement for the proof problem
theorem students_neither (total_students football_players cricket_players both_players : Nat) (h1 : total_students = 460)
  (h2 : football_players = 325) (h3 : cricket_players = 175) (h4 : both_players = 90) :
  total_students - (football_players + cricket_players - both_players) = 50 := by
  sorry

end students_neither_l164_164054


namespace arrange_books_l164_164917

open Nat

theorem arrange_books :
    let german_books := 3
    let spanish_books := 4
    let french_books := 3
    let total_books := german_books + spanish_books + french_books
    (total_books == 10) →
    let units := 2
    let items_to_arrange := units + german_books
    factorial items_to_arrange * factorial spanish_books * factorial french_books = 17280 :=
by 
    intros
    sorry

end arrange_books_l164_164917


namespace all_real_K_have_real_roots_l164_164650

noncomputable def quadratic_discriminant (K : ℝ) : ℝ :=
  let a := K ^ 3
  let b := -(4 * K ^ 3 + 1)
  let c := 3 * K ^ 3
  b ^ 2 - 4 * a * c

theorem all_real_K_have_real_roots : ∀ K : ℝ, quadratic_discriminant K ≥ 0 :=
by
  sorry

end all_real_K_have_real_roots_l164_164650


namespace right_triangle_area_l164_164709

theorem right_triangle_area (h b : ℝ) (hypotenuse : h = 5) (base : b = 3) :
  ∃ a : ℝ, a = 1 / 2 * b * (Real.sqrt (h^2 - b^2)) ∧ a = 6 := 
by
  sorry

end right_triangle_area_l164_164709


namespace irrational_infinitely_many_approximations_l164_164189

theorem irrational_infinitely_many_approximations (x : ℝ) (hx : Irrational x) (hx_pos : 0 < x) :
  ∃ᶠ (q : ℕ) in at_top, ∃ p : ℤ, |x - p / q| < 1 / q^2 :=
sorry

end irrational_infinitely_many_approximations_l164_164189


namespace birthday_paradox_l164_164009

-- Defining the problem conditions
def people (n : ℕ) := n ≥ 367

-- Using the Pigeonhole Principle as a condition
def pigeonhole_principle (pigeonholes pigeons : ℕ) := pigeonholes < pigeons

-- Stating the final proposition
theorem birthday_paradox (n : ℕ) (days_in_year : ℕ) (h1 : days_in_year = 366) (h2 : people n) : pigeonhole_principle days_in_year n :=
sorry

end birthday_paradox_l164_164009


namespace find_ab_sum_eq_42_l164_164100

noncomputable def find_value (a b : ℝ) : ℝ := a^2 * b + a * b^2

theorem find_ab_sum_eq_42 (a b : ℝ) (h1 : a + b = 6) (h2 : a * b = 7) : find_value a b = 42 := by
  sorry

end find_ab_sum_eq_42_l164_164100


namespace olivia_correct_answers_l164_164065

theorem olivia_correct_answers (c w : ℕ) 
  (h1 : c + w = 15) 
  (h2 : 6 * c - 3 * w = 45) : 
  c = 10 := 
  sorry

end olivia_correct_answers_l164_164065


namespace intersection_eq_l164_164560

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1, 2}

-- Prove the intersection of A and B is {0, 1}
theorem intersection_eq : A ∩ B = {0, 1} := by
  sorry

end intersection_eq_l164_164560


namespace initial_amount_l164_164559

def pie_cost : Real := 6.75
def juice_cost : Real := 2.50
def gift : Real := 10.00
def mary_final : Real := 52.00

theorem initial_amount (M : Real) :
  M = mary_final + pie_cost + juice_cost + gift :=
by
  sorry

end initial_amount_l164_164559


namespace symmetric_circle_l164_164340

theorem symmetric_circle (x y : ℝ) :
  let C := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1 }
  let L := { p : ℝ × ℝ | p.1 + p.2 = 1 }
  ∃ C' : ℝ × ℝ → Prop, (∀ p, C' p ↔ (p.1)^2 + (p.2)^2 = 1) :=
sorry

end symmetric_circle_l164_164340


namespace other_number_eq_462_l164_164484

theorem other_number_eq_462 (a b : ℕ) 
  (lcm_ab : Nat.lcm a b = 4620) 
  (gcd_ab : Nat.gcd a b = 21) 
  (a_eq : a = 210) : b = 462 := 
by
  sorry

end other_number_eq_462_l164_164484


namespace new_cost_percentage_l164_164393

variables (t c a x : ℝ) (n : ℕ)

def original_cost (t c a x : ℝ) (n : ℕ) : ℝ :=
  t * c * (a * x) ^ n

def new_cost (t c a x : ℝ) (n : ℕ) : ℝ :=
  t * (2 * c) * ((2 * a) * x) ^ (n + 2)

theorem new_cost_percentage (t c a x : ℝ) (n : ℕ) :
  new_cost t c a x n = 2^(n+1) * original_cost t c a x n * x^2 :=
by
  sorry

end new_cost_percentage_l164_164393


namespace friedas_probability_to_corner_l164_164149

-- Define the grid size and positions
def grid_size : Nat := 4
def start_position : ℕ × ℕ := (3, 3)
def corner_positions : List (ℕ × ℕ) := [(1, 1), (1, 4), (4, 1), (4, 4)]

-- Define the number of hops allowed
def max_hops : Nat := 4

-- Define a function to calculate the probability of reaching a corner square
-- within the given number of hops starting from the initial position.
noncomputable def prob_reach_corner (grid_size : ℕ) (start_position : ℕ × ℕ) 
                                     (corner_positions : List (ℕ × ℕ)) 
                                     (max_hops : ℕ) : ℚ :=
  -- Implementation details skipped
  sorry

-- Define the main theorem that states the desired probability
theorem friedas_probability_to_corner : 
  prob_reach_corner grid_size start_position corner_positions max_hops = 17 / 64 :=
sorry

end friedas_probability_to_corner_l164_164149


namespace sum_of_coefficients_l164_164727

theorem sum_of_coefficients (x : ℝ) : (∃ x : ℝ, 5 * x * (1 - x) = 3) → 5 + (-5) + 3 = 3 :=
by
  intro h
  -- Proof goes here
  sorry

end sum_of_coefficients_l164_164727


namespace count_positive_integers_satisfying_properties_l164_164027

theorem count_positive_integers_satisfying_properties :
  (∃ n : ℕ, ∀ N < 2007,
    (N % 2 = 1) ∧
    (N % 3 = 2) ∧
    (N % 4 = 3) ∧
    (N % 5 = 4) ∧
    (N % 6 = 5) → n = 33) :=
by
  sorry

end count_positive_integers_satisfying_properties_l164_164027


namespace exponent_fraction_simplification_l164_164061

theorem exponent_fraction_simplification : 
  (2 ^ 2016 + 2 ^ 2014) / (2 ^ 2016 - 2 ^ 2014) = 5 / 3 := 
by {
  -- proof steps would go here
  sorry
}

end exponent_fraction_simplification_l164_164061


namespace find_a_plus_b_l164_164569

theorem find_a_plus_b (a b : ℚ) (y : ℚ) (x : ℚ) :
  (y = a + b / x) →
  (2 = a + b / (-2 : ℚ)) →
  (3 = a + b / (-6 : ℚ)) →
  a + b = 13 / 2 :=
by
  intros h₁ h₂ h₃
  sorry

end find_a_plus_b_l164_164569


namespace cheryl_material_used_l164_164350

theorem cheryl_material_used :
  let material1 := (4 / 19 : ℚ)
  let material2 := (2 / 13 : ℚ)
  let bought := material1 + material2
  let leftover := (4 / 26 : ℚ)
  let used := bought - leftover
  used = (52 / 247 : ℚ) :=
by
  let material1 := (4 / 19 : ℚ)
  let material2 := (2 / 13 : ℚ)
  let bought := material1 + material2
  let leftover := (4 / 26 : ℚ)
  let used := bought - leftover
  have : used = (52 / 247 : ℚ) := sorry
  exact this

end cheryl_material_used_l164_164350


namespace megan_numbers_difference_l164_164546

theorem megan_numbers_difference 
  (x1 x2 x3 x4 x5 : ℝ) 
  (h_mean3 : (x1 + x2 + x3) / 3 = -3)
  (h_mean4 : (x1 + x2 + x3 + x4) / 4 = 4)
  (h_mean5 : (x1 + x2 + x3 + x4 + x5) / 5 = -5) :
  x4 - x5 = 66 :=
by
  sorry

end megan_numbers_difference_l164_164546


namespace total_amount_paid_l164_164797

/-- The owner's markup percentage and the cost price are given. 
We need to find out the total amount paid by the customer, which is equivalent to proving the total cost. -/
theorem total_amount_paid (markup_percentage : ℝ) (cost_price : ℝ) (markup : ℝ) (total_paid : ℝ) 
    (h1 : markup_percentage = 0.24) 
    (h2 : cost_price = 6425) 
    (h3 : markup = markup_percentage * cost_price) 
    (h4 : total_paid = cost_price + markup) : 
    total_paid = 7967 := 
sorry

end total_amount_paid_l164_164797


namespace fraction_to_terminanting_decimal_l164_164695

theorem fraction_to_terminanting_decimal : (47 / (5^4 * 2) : ℚ) = 0.0376 := 
by 
  sorry

end fraction_to_terminanting_decimal_l164_164695


namespace squirrel_travel_distance_l164_164671

def squirrel_distance (height : ℕ) (circumference : ℕ) (rise_per_circuit : ℕ) : ℕ :=
  let circuits := height / rise_per_circuit
  let horizontal_distance := circuits * circumference
  Nat.sqrt (height * height + horizontal_distance * horizontal_distance)

theorem squirrel_travel_distance :
  (squirrel_distance 16 3 4) = 20 := by
  sorry

end squirrel_travel_distance_l164_164671


namespace problem_statement_l164_164601

noncomputable def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
noncomputable def B (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

theorem problem_statement (m : ℝ) : (A ∩ (B m) = B m) → (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) :=
by
  sorry

end problem_statement_l164_164601


namespace triangle_area_l164_164048

theorem triangle_area (a b c : ℕ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25)
  (right_triangle : a^2 + b^2 = c^2) : 
  (1/2 : ℚ) * (a * b) = 84 := 
by
  -- Sorry is used as we are only providing the statement, not the full proof.
  sorry

end triangle_area_l164_164048


namespace max_integer_value_of_x_l164_164944

theorem max_integer_value_of_x (x : ℤ) : 3 * x - (1 / 4 : ℚ) ≤ (1 / 3 : ℚ) * x - 2 → x ≤ -1 :=
by
  intro h
  sorry

end max_integer_value_of_x_l164_164944


namespace simplify_expression_l164_164817

theorem simplify_expression :
  (64^(1/3) - 216^(1/3) = -2) :=
by
  have h1 : 64 = 4^3 := by norm_num
  have h2 : 216 = 6^3 := by norm_num
  sorry

end simplify_expression_l164_164817


namespace smallest_four_digit_multiple_of_18_l164_164986

-- Define the concept of a four-digit number
def four_digit (N : ℕ) : Prop := 1000 ≤ N ∧ N < 10000

-- Define the concept of a multiple of 18
def multiple_of_18 (N : ℕ) : Prop := ∃ k : ℕ, N = 18 * k

-- Define the combined condition for N being a four-digit multiple of 18
def four_digit_multiple_of_18 (N : ℕ) : Prop := four_digit N ∧ multiple_of_18 N

-- State that 1008 is the smallest such number
theorem smallest_four_digit_multiple_of_18 : ∀ N : ℕ, four_digit_multiple_of_18 N → 1008 ≤ N := 
by
  intros N H
  sorry

end smallest_four_digit_multiple_of_18_l164_164986


namespace exist_non_zero_function_iff_sum_zero_l164_164804

theorem exist_non_zero_function_iff_sum_zero (a b c : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x y z : ℝ, a * f (x * y + f z) + b * f (y * z + f x) + c * f (z * x + f y) = 0) ∧ ¬ (∀ x : ℝ, f x = 0)) ↔ (a + b + c = 0) :=
by {
  sorry
}

end exist_non_zero_function_iff_sum_zero_l164_164804


namespace cost_of_mixture_verify_cost_of_mixture_l164_164277

variables {C1 C2 Cm : ℝ}

def ratio := 5 / 12

axiom cost_of_rice_1 : C1 = 4.5
axiom cost_of_rice_2 : C2 = 8.75
axiom mix_ratio : ratio = 5 / 12

theorem cost_of_mixture (h1 : C1 = 4.5) (h2 : C2 = 8.75) (r : ratio = 5 / 12) :
  Cm = (8.75 * 5 + 4.5 * 12) / 17 :=
by sorry

-- Prove that the cost of the mixture Cm is indeed 5.75
theorem verify_cost_of_mixture (h1 : C1 = 4.5) (h2 : C2 = 8.75) (r : ratio = 5 / 12) :
  Cm = 5.75 :=
by sorry

end cost_of_mixture_verify_cost_of_mixture_l164_164277


namespace mabel_initial_daisies_l164_164602

theorem mabel_initial_daisies (D: ℕ) (h1: 8 * (D - 2) = 24) : D = 5 :=
by
  sorry

end mabel_initial_daisies_l164_164602


namespace width_of_field_l164_164840

-- Definitions for the conditions
variables (W L : ℝ) (P : ℝ)
axiom length_condition : L = (7 / 5) * W
axiom perimeter_condition : P = 2 * L + 2 * W
axiom perimeter_value : P = 336

-- Theorem to be proved
theorem width_of_field : W = 70 :=
by
  -- Here will be the proof body
  sorry

end width_of_field_l164_164840


namespace total_amount_is_200_l164_164701

-- Given conditions
def sam_amount : ℕ := 75
def billy_amount : ℕ := 2 * sam_amount - 25

-- Theorem to prove
theorem total_amount_is_200 : billy_amount + sam_amount = 200 :=
by
  sorry

end total_amount_is_200_l164_164701


namespace sufficient_not_necessary_condition_l164_164765

variable (a b c : ℝ)

-- Define the condition that the sequence forms a geometric sequence
def geometric_sequence (a1 a2 a3 a4 a5 : ℝ) :=
  ∃ q : ℝ, q ≠ 0 ∧ a1 * q = a2 ∧ a2 * q = a3 ∧ a3 * q = a4 ∧ a4 * q = a5

-- Lean statement proving the problem
theorem sufficient_not_necessary_condition :
  (geometric_sequence 1 a b c 16) → (b = 4) ∧ ¬ (b = 4 → geometric_sequence 1 a b c 16) :=
sorry

end sufficient_not_necessary_condition_l164_164765


namespace vegetables_in_one_serving_l164_164697

theorem vegetables_in_one_serving
  (V : ℝ)
  (H1 : ∀ servings : ℝ, servings > 0 → servings * (V + 2.5) = 28)
  (H_pints_to_cups : 14 * 2 = 28) :
  V = 1 :=
by
  -- proof steps would go here
  sorry

end vegetables_in_one_serving_l164_164697


namespace solve_for_x_l164_164718

theorem solve_for_x :
  ∃ x : ℝ, x ≠ 0 ∧ (9 * x) ^ 18 = (27 * x) ^ 9 + 81 * x ∧ x = 1 / 3 :=
by
  sorry

end solve_for_x_l164_164718


namespace sets_tossed_per_show_l164_164884

-- Definitions
def sets_used_per_show : ℕ := 5
def number_of_shows : ℕ := 30
def total_sets_used : ℕ := 330

-- Statement to prove
theorem sets_tossed_per_show : 
  (total_sets_used - (sets_used_per_show * number_of_shows)) / number_of_shows = 6 := 
by
  sorry

end sets_tossed_per_show_l164_164884


namespace stratified_sample_correct_l164_164584

variable (popA popB popC : ℕ) (totalSample : ℕ)

def stratified_sample (popA popB popC totalSample : ℕ) : ℕ × ℕ × ℕ :=
  let totalChickens := popA + popB + popC
  let sampledA := (popA * totalSample) / totalChickens
  let sampledB := (popB * totalSample) / totalChickens
  let sampledC := (popC * totalSample) / totalChickens
  (sampledA, sampledB, sampledC)

theorem stratified_sample_correct
  (hA : popA = 12000) (hB : popB = 8000) (hC : popC = 4000) (hSample : totalSample = 120) :
  stratified_sample popA popB popC totalSample = (60, 40, 20) :=
by
  sorry

end stratified_sample_correct_l164_164584


namespace westbound_speed_is_275_l164_164226

-- Define the conditions for the problem at hand.
def east_speed : ℕ := 325
def separation_time : ℝ := 3.5
def total_distance : ℕ := 2100

-- Compute the known east-bound distance.
def east_distance : ℝ := east_speed * separation_time

-- Define the speed of the west-bound plane as an unknown variable.
variable (v : ℕ)

-- Compute the west-bound distance.
def west_distance := v * separation_time

-- The assertion that the sum of two distances equals the total distance.
def distance_equation := east_distance + (v * separation_time) = total_distance

-- Prove that the west-bound speed is 275 mph.
theorem westbound_speed_is_275 : v = 275 :=
by
  sorry

end westbound_speed_is_275_l164_164226


namespace moles_H2O_formed_l164_164541

-- Define the conditions
def moles_HCl : ℕ := 6
def moles_CaCO3 : ℕ := 3
def moles_CaCl2 : ℕ := 3
def moles_CO2 : ℕ := 3

-- Proposition that we need to prove
theorem moles_H2O_formed : moles_CaCl2 = 3 ∧ moles_CO2 = 3 ∧ moles_CaCO3 = 3 ∧ moles_HCl = 6 → moles_CaCO3 = 3 := by
  sorry

end moles_H2O_formed_l164_164541


namespace range_of_a_l164_164039

open Set

theorem range_of_a (a x : ℝ) (h : x^2 - 2 * x + 1 - a^2 < 0) (h2 : 0 < x) (h3 : x < 4) :
  a < -3 ∨ a > 3 :=
sorry

end range_of_a_l164_164039


namespace no_sum_of_three_squares_l164_164363

theorem no_sum_of_three_squares (n : ℤ) (h : n % 8 = 7) : 
  ¬ ∃ a b c : ℤ, a^2 + b^2 + c^2 = n :=
by 
sorry

end no_sum_of_three_squares_l164_164363


namespace arithmetic_sum_sequence_l164_164168

theorem arithmetic_sum_sequence (a : ℕ → ℝ) (d : ℝ)
  (h : ∀ n, a (n + 1) = a n + d) :
  ∃ d', 
    a 4 + a 5 + a 6 - (a 1 + a 2 + a 3) = d' ∧
    a 7 + a 8 + a 9 - (a 4 + a 5 + a 6) = d' :=
by
  sorry

end arithmetic_sum_sequence_l164_164168


namespace triangle_side_length_l164_164497

theorem triangle_side_length {x : ℝ} (h1 : 6 + x + x = 20) : x = 7 :=
by 
  sorry

end triangle_side_length_l164_164497


namespace students_no_A_in_any_subject_l164_164178

def total_students : ℕ := 50
def a_in_history : ℕ := 9
def a_in_math : ℕ := 15
def a_in_science : ℕ := 12
def a_in_math_and_history : ℕ := 5
def a_in_history_and_science : ℕ := 3
def a_in_science_and_math : ℕ := 4
def a_in_all_three : ℕ := 1

theorem students_no_A_in_any_subject : 
  (total_students - (a_in_history + a_in_math + a_in_science 
                      - a_in_math_and_history - a_in_history_and_science - a_in_science_and_math 
                      + a_in_all_three)) = 28 := by
  sorry

end students_no_A_in_any_subject_l164_164178


namespace temp_fri_l164_164438

-- Define the temperatures on Monday, Tuesday, Wednesday, Thursday, and Friday
variables (M T W Th F : ℝ)

-- Define the conditions as given in the problem
axiom avg_mon_thurs : (M + T + W + Th) / 4 = 48
axiom avg_tues_fri : (T + W + Th + F) / 4 = 46
axiom temp_mon : M = 39

-- The theorem to prove that the temperature on Friday is 31 degrees
theorem temp_fri : F = 31 :=
by
  -- placeholder for proof
  sorry

end temp_fri_l164_164438


namespace pounds_of_coffee_bought_l164_164514

theorem pounds_of_coffee_bought 
  (total_amount_gift_card : ℝ := 70) 
  (cost_per_pound : ℝ := 8.58) 
  (amount_left_on_card : ℝ := 35.68) :
  (total_amount_gift_card - amount_left_on_card) / cost_per_pound = 4 :=
sorry

end pounds_of_coffee_bought_l164_164514


namespace find_t_l164_164641

theorem find_t (t : ℤ) :
  ((t + 1) * (3 * t - 3)) = ((3 * t - 5) * (t + 2) + 2) → 
  t = 5 :=
by
  intros
  sorry

end find_t_l164_164641


namespace solution_to_first_equation_solution_to_second_equation_l164_164567

theorem solution_to_first_equation (x : ℝ) : 
  x^2 - 6 * x + 1 = 0 ↔ x = 3 + 2 * Real.sqrt 2 ∨ x = 3 - 2 * Real.sqrt 2 :=
by sorry

theorem solution_to_second_equation (x : ℝ) : 
  (2 * x - 3)^2 = 5 * (2 * x - 3) ↔ x = 3 / 2 ∨ x = 4 :=
by sorry

end solution_to_first_equation_solution_to_second_equation_l164_164567


namespace main_l164_164090

theorem main (x y : ℤ) (h1 : abs x = 5) (h2 : abs y = 3) (h3 : x * y > 0) : 
    x - y = 2 ∨ x - y = -2 := sorry

end main_l164_164090


namespace coins_player_1_received_l164_164147

def round_table := List Nat
def players := List Nat
def coins_received (table: round_table) (player_idx: Nat) : Nat :=
sorry -- the function to calculate coins received by player's index

-- Define the given conditions
def sectors : round_table := [1, 2, 3, 4, 5, 6, 7, 8, 9]
def num_players := 9
def num_rotations := 11
def player_4 := 4
def player_8 := 8
def player_1 := 1
def coins_player_4 := 90
def coins_player_8 := 35

theorem coins_player_1_received : coins_received sectors player_1 = 57 :=
by
  -- Setup the conditions
  have h1 : coins_received sectors player_4 = 90 := sorry
  have h2 : coins_received sectors player_8 = 35 := sorry
  -- Prove the target statement
  show coins_received sectors player_1 = 57
  sorry

end coins_player_1_received_l164_164147


namespace how_much_together_l164_164144

def madeline_money : ℕ := 48
def brother_money : ℕ := madeline_money / 2

theorem how_much_together : madeline_money + brother_money = 72 := by
  sorry

end how_much_together_l164_164144


namespace min_seats_occupied_l164_164895

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

end min_seats_occupied_l164_164895


namespace slope_of_line_inclination_angle_l164_164789

theorem slope_of_line_inclination_angle 
  (k : ℝ) (θ : ℝ)
  (hθ1 : 30 * (π / 180) < θ)
  (hθ2 : θ < 90 * (π / 180)) :
  k = Real.tan θ → k > Real.tan (30 * (π / 180)) :=
by
  intro h
  sorry

end slope_of_line_inclination_angle_l164_164789


namespace negation_exists_l164_164146

theorem negation_exists (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ ∀ x : ℝ, x^2 - a * x + 1 ≥ 0 :=
sorry

end negation_exists_l164_164146


namespace fraction_calculation_l164_164353

theorem fraction_calculation :
  (1 / 4) * (1 / 3) * (1 / 6) * 144 + (1 / 2) = (5 / 2) :=
by
  sorry

end fraction_calculation_l164_164353


namespace correct_option_is_C_l164_164259

-- Definitions of the expressions given in the conditions
def optionA (a : ℝ) : ℝ := 3 * a^5 - a^5
def optionB (a : ℝ) : ℝ := a^2 + a^5
def optionC (a : ℝ) : ℝ := a^5 + a^5
def optionD (x y : ℝ) : ℝ := x^2 * y + x * y^2

-- The problem is to prove that optionC is correct and the others are not
theorem correct_option_is_C (a x y : ℝ) :
  (optionC a = 2 * a^5) ∧ 
  (optionA a ≠ 3) ∧ 
  (optionB a ≠ a^7) ∧ 
  (optionD x y ≠ 2 * (x ^ 3) * (y ^ 3)) :=
by
  sorry

end correct_option_is_C_l164_164259


namespace solve_for_a_l164_164241

theorem solve_for_a (a : ℚ) (h : a + a / 4 = 10 / 4) : a = 2 :=
sorry

end solve_for_a_l164_164241


namespace inequality_problem_l164_164704

theorem inequality_problem
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) :=
sorry

end inequality_problem_l164_164704


namespace integral_f_eq_34_l164_164512

noncomputable def f (x : ℝ) := if x ∈ [0, 1] then (1 / Real.pi) * Real.sqrt (1 - x^2) else 2 - x

theorem integral_f_eq_34 :
  ∫ x in (0 : ℝ)..2, f x = 3 / 4 :=
by
  sorry

end integral_f_eq_34_l164_164512


namespace Hilltown_Volleyball_Club_Members_l164_164509

-- Definitions corresponding to the conditions
def knee_pad_cost : ℕ := 6
def uniform_cost : ℕ := 14
def total_expenditure : ℕ := 4000

-- Definition of total cost per member
def cost_per_member : ℕ := 2 * (knee_pad_cost + uniform_cost)

-- Proof statement
theorem Hilltown_Volleyball_Club_Members :
  total_expenditure % cost_per_member = 0 ∧ total_expenditure / cost_per_member = 100 := by
    sorry

end Hilltown_Volleyball_Club_Members_l164_164509


namespace cannot_achieve_90_cents_l164_164722

theorem cannot_achieve_90_cents :
  ∀ (p n d q : ℕ),        -- p: number of pennies, n: number of nickels, d: number of dimes, q: number of quarters
  (p + n + d + q = 6) →   -- exactly six coins chosen
  (p ≤ 4 ∧ n ≤ 4 ∧ d ≤ 4 ∧ q ≤ 4) →  -- no more than four of each kind of coin
  (p + 5 * n + 10 * d + 25 * q ≠ 90) -- total value should not equal 90 cents
:= by
  sorry

end cannot_achieve_90_cents_l164_164722


namespace ratio_of_part_to_whole_l164_164383

theorem ratio_of_part_to_whole (N : ℝ) (P : ℝ) (h1 : (1/4) * (2/5) * N = 17) (h2 : 0.40 * N = 204) :
  P = (2/5) * N → P / N = 2 / 5 :=
by
  intro h3
  sorry

end ratio_of_part_to_whole_l164_164383


namespace weight_of_new_person_l164_164402

theorem weight_of_new_person 
  (avg_increase : Real)
  (num_persons : Nat)
  (old_weight : Real)
  (new_avg_increase : avg_increase = 2.2)
  (number_of_persons : num_persons = 15)
  (weight_of_old_person : old_weight = 75)
  : (new_weight : Real) = old_weight + avg_increase * num_persons := 
  by sorry

end weight_of_new_person_l164_164402


namespace solve_for_a_l164_164134

theorem solve_for_a (a : ℤ) :
  (|2 * a + 1| = 3) ↔ (a = 1 ∨ a = -2) :=
by
  sorry

end solve_for_a_l164_164134


namespace exists_plane_intersecting_in_parallel_lines_l164_164454

variables {Point Line Plane : Type}
variables (a : Line) (S₁ S₂ : Plane)

-- Definitions and assumptions
def intersects_in (a : Line) (P : Plane) : Prop := sorry
def parallel_lines (l₁ l₂ : Line) : Prop := sorry

-- Proof problem statement
theorem exists_plane_intersecting_in_parallel_lines :
  ∃ P : Plane, intersects_in a P ∧
    (∀ l₁ l₂ : Line, (intersects_in l₁ S₁ ∧ intersects_in l₂ S₂ ∧ l₁ = l₂)
                     → parallel_lines l₁ l₂) :=
sorry

end exists_plane_intersecting_in_parallel_lines_l164_164454


namespace probability_of_specific_selection_l164_164326

/-- 
Given a drawer with 8 forks, 10 spoons, and 6 knives, 
the probability of randomly choosing one fork, one spoon, and one knife when three pieces of silverware are removed equals 120/506.
-/
theorem probability_of_specific_selection :
  let total_pieces := 24
  let total_ways := Nat.choose total_pieces 3
  let favorable_ways := 8 * 10 * 6
  (favorable_ways : ℚ) / total_ways = 120 / 506 := 
by
  sorry

end probability_of_specific_selection_l164_164326


namespace correct_exponentiation_l164_164192

variable (a : ℝ)

theorem correct_exponentiation : (a^2)^3 = a^6 := by
  sorry

end correct_exponentiation_l164_164192


namespace population_meets_capacity_l164_164264

-- Define the initial conditions and parameters
def initial_year : ℕ := 1998
def initial_population : ℕ := 100
def population_growth_rate : ℕ := 4  -- quadruples every 20 years
def years_per_growth_period : ℕ := 20
def land_area_hectares : ℕ := 15000
def hectares_per_person : ℕ := 2
def maximum_capacity : ℕ := land_area_hectares / hectares_per_person

-- Define the statement
theorem population_meets_capacity :
  ∃ (years_from_initial : ℕ), years_from_initial = 60 ∧
  initial_population * population_growth_rate ^ (years_from_initial / years_per_growth_period) ≥ maximum_capacity :=
by
  sorry

end population_meets_capacity_l164_164264


namespace find_C_D_l164_164805

theorem find_C_D (x C D : ℚ) 
  (h : 7 * x - 5 ≠ 0) -- Added condition to avoid zero denominator
  (hx : x^2 - 8 * x - 48 = (x - 12) * (x + 4))
  (h_eq : 7 * x - 5 = C * (x + 4) + D * (x - 12))
  (h_c : C = 79 / 16)
  (h_d : D = 33 / 16)
: 7 * x - 5 = 79 / 16 * (x + 4) + 33 / 16 * (x - 12) :=
by sorry

end find_C_D_l164_164805


namespace multiples_of_8_has_highest_avg_l164_164304

def average_of_multiples (m : ℕ) (a b : ℕ) : ℕ :=
(a + b) / 2

def multiples_of_7_avg := average_of_multiples 7 7 196 -- 101.5
def multiples_of_2_avg := average_of_multiples 2 2 200 -- 101
def multiples_of_8_avg := average_of_multiples 8 8 200 -- 104
def multiples_of_5_avg := average_of_multiples 5 5 200 -- 102.5
def multiples_of_9_avg := average_of_multiples 9 9 189 -- 99

theorem multiples_of_8_has_highest_avg :
  multiples_of_8_avg > multiples_of_7_avg ∧
  multiples_of_8_avg > multiples_of_2_avg ∧
  multiples_of_8_avg > multiples_of_5_avg ∧
  multiples_of_8_avg > multiples_of_9_avg :=
by
  sorry

end multiples_of_8_has_highest_avg_l164_164304


namespace parametric_graph_right_half_circle_l164_164088

theorem parametric_graph_right_half_circle (θ : ℝ) (x y : ℝ) (hx : x = 3 * Real.cos θ) (hy : y = 3 * Real.sin θ) (hθ : -Real.pi / 2 ≤ θ ∧ θ ≤ Real.pi / 2) :
  x^2 + y^2 = 9 ∧ x ≥ 0 :=
by
  sorry

end parametric_graph_right_half_circle_l164_164088


namespace find_a_l164_164173

namespace MathProof

theorem find_a (a : ℕ) (h_pos : a > 0) (h_eq : (a : ℚ) / (a + 18) = 47 / 50) : a = 282 :=
by
  sorry

end MathProof

end find_a_l164_164173


namespace least_value_of_x_for_divisibility_l164_164856

theorem least_value_of_x_for_divisibility (x : ℕ) (h : 1 + 8 + 9 + 4 = 22) :
  ∃ x : ℕ, (22 + x) % 3 = 0 ∧ x = 2 := by
sorry

end least_value_of_x_for_divisibility_l164_164856


namespace pencils_to_make_profit_l164_164615

theorem pencils_to_make_profit
  (total_pencils : ℕ)
  (cost_per_pencil : ℝ)
  (selling_price_per_pencil : ℝ)
  (desired_profit : ℝ)
  (pencils_to_be_sold : ℕ) :
  total_pencils = 2000 →
  cost_per_pencil = 0.08 →
  selling_price_per_pencil = 0.20 →
  desired_profit = 160 →
  pencils_to_be_sold = 1600 :=
sorry

end pencils_to_make_profit_l164_164615


namespace midpoint_in_polar_coordinates_l164_164611

theorem midpoint_in_polar_coordinates :
  let A := (9, Real.pi / 3)
  let B := (9, 2 * Real.pi / 3)
  let mid := (Real.sqrt (3) * 9 / 2, Real.pi / 2)
  (mid = (Real.sqrt (3) * 9 / 2, Real.pi / 2)) :=
by 
  sorry

end midpoint_in_polar_coordinates_l164_164611


namespace sum_f_values_l164_164281

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - 2 / x) + 1

theorem sum_f_values : 
  f (-7) + f (-5) + f (-3) + f (-1) + f (3) + f (5) + f (7) + f (9) = 8 := 
by
  sorry

end sum_f_values_l164_164281


namespace symmetric_point_origin_l164_164105

-- Define the original point
def original_point : ℝ × ℝ := (4, -1)

-- Define a function to find the symmetric point with respect to the origin
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- State the theorem
theorem symmetric_point_origin : symmetric_point original_point = (-4, 1) :=
sorry

end symmetric_point_origin_l164_164105


namespace mother_duck_multiple_of_first_two_groups_l164_164231

variables (num_ducklings : ℕ) (snails_first_batch : ℕ) (snails_second_batch : ℕ)
          (total_snails : ℕ) (mother_duck_snails : ℕ)

-- Given conditions
def conditions : Prop :=
  num_ducklings = 8 ∧ 
  snails_first_batch = 3 * 5 ∧ 
  snails_second_batch = 3 * 9 ∧ 
  total_snails = 294 ∧ 
  total_snails = snails_first_batch + snails_second_batch + 2 * mother_duck_snails ∧ 
  mother_duck_snails > 0

-- Our goal is to prove that the mother duck finds 3 times the snails the first two groups of ducklings find
theorem mother_duck_multiple_of_first_two_groups (h : conditions num_ducklings snails_first_batch snails_second_batch total_snails mother_duck_snails) : 
  mother_duck_snails / (snails_first_batch + snails_second_batch) = 3 :=
by 
  sorry

end mother_duck_multiple_of_first_two_groups_l164_164231


namespace equilateral_triangle_black_area_l164_164599

theorem equilateral_triangle_black_area :
  let initial_black_area := 1
  let change_fraction := 5/6 * 9/10
  let area_after_n_changes (n : Nat) : ℚ := initial_black_area * (change_fraction ^ n)
  area_after_n_changes 3 = 27/64 := 
by
  sorry

end equilateral_triangle_black_area_l164_164599


namespace domain_of_ln_function_l164_164543

theorem domain_of_ln_function (x : ℝ) : 3 - 4 * x > 0 ↔ x < 3 / 4 := 
by
  sorry

end domain_of_ln_function_l164_164543


namespace compute_expression_l164_164096

theorem compute_expression :
    (3 + 5)^2 + (3^2 + 5^2 + 3 * 5) = 113 := 
by sorry

end compute_expression_l164_164096


namespace geometric_sequence_sum_l164_164597

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 3 + a 5 = 20)
  (h2 : a 4 = 8)
  (h3 : ∀ n, a (n + 1) = a n * q) :
  a 2 + a 6 = 34 := 
sorry

end geometric_sequence_sum_l164_164597


namespace poker_cards_count_l164_164041

theorem poker_cards_count (total_cards kept_away : ℕ) 
  (h1 : total_cards = 52) 
  (h2 : kept_away = 7) : 
  total_cards - kept_away = 45 :=
by 
  sorry

end poker_cards_count_l164_164041


namespace value_of_a_g_odd_iff_m_eq_one_l164_164881

noncomputable def f (a x : ℝ) : ℝ := a ^ x

noncomputable def g (m x a : ℝ) : ℝ := m - 2 / (f a x + 1)

theorem value_of_a
  (a : ℝ)
  (h_pos : a > 0)
  (h_neq_one : a ≠ 1)
  (h_diff : ∀ x y : ℝ, x ∈ (Set.Icc 1 2) → y ∈ (Set.Icc 1 2) → abs (f a x - f a y) = 2) :
  a = 2 :=
sorry

theorem g_odd_iff_m_eq_one
  (a m : ℝ)
  (h_pos : a > 0)
  (h_neq_one : a ≠ 1)
  (h_a_eq : a = 2) :
  (∀ x : ℝ, g m x a = -g m (-x) a) ↔ m = 1 :=
sorry

end value_of_a_g_odd_iff_m_eq_one_l164_164881


namespace percentage_saved_l164_164735

theorem percentage_saved (saved spent : ℝ) (h_saved : saved = 3) (h_spent : spent = 27) : 
  (saved / (saved + spent)) * 100 = 10 := by
  sorry

end percentage_saved_l164_164735


namespace cos_value_l164_164110

theorem cos_value (α : ℝ) (h : Real.sin (π/4 + α) = 1/3) :
  Real.cos (π/2 - 2*α) = -7/9 :=
sorry

end cos_value_l164_164110


namespace train_distance_proof_l164_164035

-- Definitions
def speed_train1 : ℕ := 40
def speed_train2 : ℕ := 48
def time_hours : ℕ := 8
def initial_distance : ℕ := 892

-- Function to calculate distance after given time
def distance (speed time : ℕ) : ℕ := speed * time

-- Increased/Decreased distance after time
def distance_diff : ℕ := distance speed_train2 time_hours - distance speed_train1 time_hours

-- Final distances
def final_distance_same_direction : ℕ := initial_distance + distance_diff
def final_distance_opposite_direction : ℕ := initial_distance - distance_diff

-- Proof statement
theorem train_distance_proof :
  final_distance_same_direction = 956 ∧ final_distance_opposite_direction = 828 :=
by
  -- The proof is omitted here
  sorry

end train_distance_proof_l164_164035


namespace desired_value_l164_164123

noncomputable def find_sum (a b c : ℝ) (p q r : ℝ) : ℝ :=
  a / p + b / q + c / r

theorem desired_value (a b c : ℝ) (h1 : p = a / 2) (h2 : q = b / 2) (h3 : r = c / 2) :
  find_sum a b c p q r = 6 :=
by
  sorry

end desired_value_l164_164123


namespace length_BA_correct_area_ABCDE_correct_l164_164051

variables {BE CD CE CA : ℝ}
axiom BE_eq : BE = 13
axiom CD_eq : CD = 3
axiom CE_eq : CE = 10
axiom CA_eq : CA = 10

noncomputable def length_BA : ℝ := 3
noncomputable def area_ABCDE : ℝ := 4098 / 61

theorem length_BA_correct (h1 : BE = 13) (h2 : CD = 3) (h3 : CE = 10) (h4 : CA = 10) :
  length_BA = 3 := 
by { sorry }

theorem area_ABCDE_correct (h1 : BE = 13) (h2 : CD = 3) (h3 : CE = 10) (h4 : CA = 10) :
  area_ABCDE = 4098 / 61 := 
by { sorry }

end length_BA_correct_area_ABCDE_correct_l164_164051


namespace average_speed_of_horse_l164_164394

/-- Definitions of the conditions given in the problem. --/
def pony_speed : ℕ := 20
def pony_head_start_hours : ℕ := 3
def horse_chase_hours : ℕ := 4

-- Define a proof problem for the average speed of the horse.
theorem average_speed_of_horse : (pony_head_start_hours * pony_speed + horse_chase_hours * pony_speed) / horse_chase_hours = 35 := by
  -- Setting up the necessary distances
  let pony_head_start_distance := pony_head_start_hours * pony_speed
  let pony_additional_distance := horse_chase_hours * pony_speed
  let total_pony_distance := pony_head_start_distance + pony_additional_distance
  -- Asserting the average speed of the horse
  let horse_average_speed := total_pony_distance / horse_chase_hours
  show horse_average_speed = 35
  sorry

end average_speed_of_horse_l164_164394


namespace total_distance_walked_l164_164324

theorem total_distance_walked 
  (d1 : ℝ) (d2 : ℝ)
  (h1 : d1 = 0.75)
  (h2 : d2 = 0.25) :
  d1 + d2 = 1 :=
by
  sorry

end total_distance_walked_l164_164324


namespace vector_BC_calculation_l164_164253

/--
If \(\overrightarrow{AB} = (3, 6)\) and \(\overrightarrow{AC} = (1, 2)\),
then \(\overrightarrow{BC} = (-2, -4)\).
-/
theorem vector_BC_calculation (AB AC BC : ℤ × ℤ) 
  (hAB : AB = (3, 6))
  (hAC : AC = (1, 2)) : 
  BC = (-2, -4) := 
by
  sorry

end vector_BC_calculation_l164_164253


namespace charlie_coins_l164_164417

variables (a c : ℕ)

axiom condition1 : c + 2 = 5 * (a - 2)
axiom condition2 : c - 2 = 4 * (a + 2)

theorem charlie_coins : c = 98 :=
by {
    sorry
}

end charlie_coins_l164_164417


namespace consecutive_triples_with_product_divisible_by_1001_l164_164092

theorem consecutive_triples_with_product_divisible_by_1001 :
  ∃ (a b c : ℕ), 
    (a = 76 ∧ b = 77 ∧ c = 78) ∨ 
    (a = 77 ∧ b = 78 ∧ c = 79) ∧ 
    (a ≤ 100 ∧ b ≤ 100 ∧ c ≤ 100) ∧ 
    (b = a + 1 ∧ c = b + 1) ∧ 
    (1001 ∣ (a * b * c)) :=
by sorry

end consecutive_triples_with_product_divisible_by_1001_l164_164092


namespace max_positive_integers_on_circle_l164_164145

theorem max_positive_integers_on_circle (a : ℕ → ℕ) (h: ∀ k : ℕ, 2 < k → a k > a (k-1) + a (k-2)) :
  ∃ n : ℕ, (∀ i < 2018, a i > 0 -> n ≤ 1009) :=
  sorry

end max_positive_integers_on_circle_l164_164145


namespace three_point_three_seven_five_as_fraction_l164_164558

theorem three_point_three_seven_five_as_fraction :
  3.375 = (27 / 8 : ℚ) :=
by sorry

end three_point_three_seven_five_as_fraction_l164_164558


namespace fraction_equality_l164_164830

theorem fraction_equality (a b : ℝ) (h : a / 4 = b / 3) : b / (a - b) = 3 :=
sorry

end fraction_equality_l164_164830


namespace equality_of_integers_l164_164183

theorem equality_of_integers (a b : ℕ) (h1 : ∀ n : ℕ, ∃ m : ℕ, m > 0 ∧ (a^m + b^m) % (a^n + b^n) = 0) : a = b :=
sorry

end equality_of_integers_l164_164183


namespace average_weight_increase_l164_164534

theorem average_weight_increase 
  (A : ℝ) (X : ℝ)
  (h1 : 8 * (A + X) = 8 * A + 36) :
  X = 4.5 := 
sorry

end average_weight_increase_l164_164534


namespace regions_divided_by_7_tangents_l164_164318

-- Define the recursive function R for the number of regions divided by n tangents
def R : ℕ → ℕ
| 0       => 1
| (n + 1) => R n + (n + 1)

-- The theorem stating the specific case of the problem
theorem regions_divided_by_7_tangents : R 7 = 29 := by
  sorry

end regions_divided_by_7_tangents_l164_164318


namespace two_dollar_coin_is_toonie_l164_164142

/-- We define the $2 coin in Canada -/
def two_dollar_coin_name : String := "toonie"

/-- Antonella's wallet problem setup -/
def Antonella_has_ten_coins := 10
def loonies_value := 1
def toonies_value := 2
def coins_after_purchase := 11
def purchase_amount := 3
def initial_toonies := 4

/-- Proving that the $2 coin is called a "toonie" -/
theorem two_dollar_coin_is_toonie :
  two_dollar_coin_name = "toonie" :=
by
  -- Here, we place the logical steps to derive that two_dollar_coin_name = "toonie"
  sorry

end two_dollar_coin_is_toonie_l164_164142


namespace solution_set_inequality_l164_164634

theorem solution_set_inequality (x : ℝ) : (x^2-2*x-3)*(x^2+1) < 0 ↔ -1 < x ∧ x < 3 :=
by
  sorry

end solution_set_inequality_l164_164634


namespace sinC_calculation_maxArea_calculation_l164_164630

noncomputable def sinC_given_sides_and_angles (A B C a b c : ℝ) (h1 : 2 * Real.sin A = a * Real.cos B) (h2 : b = Real.sqrt 5) (h3 : c = 2) : ℝ :=
  Real.sin C

theorem sinC_calculation 
  (A B C a b c : ℝ) 
  (h1 : 2 * Real.sin A = a * Real.cos B)
  (h2 : b = Real.sqrt 5)
  (h3 : c = 2) 
  (h4 : Real.sin B = Real.sqrt 5 / 3) : 
  sinC_given_sides_and_angles A B C a b c h1 h2 h3 = 2 / 3 := by sorry

noncomputable def maxArea_given_sides_and_angles (A B C a b c : ℝ) (h1 : 2 * Real.sin A = a * Real.cos B) (h2 : b = Real.sqrt 5) (h3 : c = 2) : ℝ :=
  1 / 2 * a * c * Real.sin B

theorem maxArea_calculation 
  (A B C a b c : ℝ) 
  (h1 : 2 * Real.sin A = a * Real.cos B)
  (h2 : b = Real.sqrt 5)
  (h3 : c = 2)
  (h4 : Real.sin B = Real.sqrt 5 / 3) 
  (h5 : a * c ≤ 15 / 2) : 
  maxArea_given_sides_and_angles A B C a b c h1 h2 h3 = 5 * Real.sqrt 5 / 4 := by sorry

end sinC_calculation_maxArea_calculation_l164_164630


namespace smallest_n_for_congruence_l164_164212

theorem smallest_n_for_congruence :
  ∃ n : ℕ, n > 0 ∧ 7 ^ n % 4 = n ^ 7 % 4 ∧ ∀ m : ℕ, (m > 0 ∧ m < n → ¬ (7 ^ m % 4 = m ^ 7 % 4)) :=
by
  sorry

end smallest_n_for_congruence_l164_164212


namespace total_spider_legs_l164_164293

-- Define the number of legs per spider.
def legs_per_spider : ℕ := 8

-- Define half of the legs per spider.
def half_legs : ℕ := legs_per_spider / 2

-- Define the number of spiders in the group.
def num_spiders : ℕ := half_legs + 10

-- Prove the total number of spider legs in the group is 112.
theorem total_spider_legs : num_spiders * legs_per_spider = 112 := by
  -- Use 'sorry' to skip the detailed proof steps.
  sorry

end total_spider_legs_l164_164293


namespace monotonic_invertible_function_l164_164465

theorem monotonic_invertible_function (f : ℝ → ℝ) (c : ℝ) (h_mono : ∀ x y, x < y → f x < f y) (h_inv : ∀ x, f (f⁻¹ x) = x) :
  (∀ x, f x + f⁻¹ x = 2 * x) ↔ ∀ x, f x = x + c :=
sorry

end monotonic_invertible_function_l164_164465


namespace area_of_regular_octagon_l164_164483

theorem area_of_regular_octagon (BDEF_is_rectangle : true) (AB : ℝ) (BC : ℝ) 
    (capture_regular_octagon : true) (AB_eq_1 : AB = 1) (BC_eq_2 : BC = 2)
    (octagon_perimeter_touch : ∀ x, x = 1) : 
    ∃ A : ℝ, A = 11 :=
by
  sorry

end area_of_regular_octagon_l164_164483


namespace exists_k_ge_2_l164_164485

def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def weak (a b n : ℕ) : Prop :=
  ¬ ∃ x y : ℕ, a * x + b * y = n

theorem exists_k_ge_2 (a b n : ℕ) (h_coprime : coprime a b) (h_positive : 0 < n) (h_weak : weak a b n) (h_bound : n < a * b / 6) :
  ∃ k : ℕ, 2 ≤ k ∧ weak a b (k * n) :=
sorry

end exists_k_ge_2_l164_164485


namespace triplet_sum_not_equal_two_l164_164133

theorem triplet_sum_not_equal_two :
  ¬((1.2 + -2.2 + 2) = 2) ∧ ¬((- 4 / 3 + - 2 / 3 + 3) = 2) :=
by
  sorry

end triplet_sum_not_equal_two_l164_164133


namespace average_seq_13_to_52_l164_164525

-- Define the sequence of natural numbers from 13 to 52
def seq : List ℕ := List.range' 13 52

-- Define the average of a list of natural numbers
def average (xs : List ℕ) : ℚ := (xs.sum : ℚ) / xs.length

-- Define the specific set of numbers and their average
theorem average_seq_13_to_52 : average seq = 32.5 := 
by 
  sorry

end average_seq_13_to_52_l164_164525


namespace placemat_length_correct_l164_164221

noncomputable def placemat_length (r : ℝ) : ℝ :=
  2 * r * Real.sin (Real.pi / 8)

theorem placemat_length_correct (r : ℝ) (n : ℕ) (w : ℝ) (y : ℝ) (h_r : r = 5)
  (h_n : n = 8) (h_w : w = 1)
  (h_y : y = placemat_length r) :
  y = 5 * Real.sqrt (2 - Real.sqrt 2) :=
by
  sorry

end placemat_length_correct_l164_164221


namespace abs_neg_three_l164_164136

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l164_164136


namespace initial_number_18_l164_164885

theorem initial_number_18 (N : ℤ) (h : ∃ k : ℤ, N + 5 = 23 * k) : N = 18 := 
sorry

end initial_number_18_l164_164885


namespace tens_digit_3_pow_2016_eq_2_l164_164751

def tens_digit (n : Nat) : Nat := (n / 10) % 10

theorem tens_digit_3_pow_2016_eq_2 : tens_digit (3 ^ 2016) = 2 := by
  sorry

end tens_digit_3_pow_2016_eq_2_l164_164751


namespace vector_expression_eval_l164_164948

open Real

noncomputable def v1 : ℝ × ℝ := (3, -8)
noncomputable def v2 : ℝ × ℝ := (2, -4)
noncomputable def k : ℝ := 5

theorem vector_expression_eval : (v1.1 - k * v2.1, v1.2 - k * v2.2) = (-7, 12) :=
  by sorry

end vector_expression_eval_l164_164948


namespace max_value_of_f_l164_164500

def f (x : ℝ) : ℝ := 9 * x - 4 * x^2

theorem max_value_of_f :
  (∀ x : ℝ, f x ≤ 5.0625) ∧ (∃ x : ℝ, f x = 5.0625) :=
by
  sorry

end max_value_of_f_l164_164500


namespace min_distance_sum_l164_164796

theorem min_distance_sum
  (A B C D E P : ℝ)
  (h_collinear : B = A + 2 ∧ C = B + 2 ∧ D = C + 3 ∧ E = D + 4)
  (h_bisector : P = (A + E) / 2) :
  (A - P)^2 + (B - P)^2 + (C - P)^2 + (D - P)^2 + (E - P)^2 = 77.25 :=
by
  sorry

end min_distance_sum_l164_164796


namespace initial_amount_in_cookie_jar_l164_164166

theorem initial_amount_in_cookie_jar (M : ℝ) (h : 15 / 100 * (85 / 100 * (100 - 10) / 100 * (100 - 15) / 100 * M) = 15) : M = 24.51 :=
sorry

end initial_amount_in_cookie_jar_l164_164166


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l164_164691

def is_composite (n : Nat) : Prop := n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem exists_nine_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 8 ≤ 500) ∧ ∀ i ∈ (List.range 9), is_composite (a + i) :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ (a : Nat), (a ≥ 1 ∧ a + 10 ≤ 500) ∧ ∀ i ∈ (List.range 11), is_composite (a + i) :=
sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l164_164691


namespace simplify_expression_l164_164923

theorem simplify_expression (x y : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x * (x^2 * y - x * y^2) + x * y * (2 * x * y - x^2)) / (x^2 * y) = x := 
by
  sorry

end simplify_expression_l164_164923


namespace initial_number_of_men_l164_164409

theorem initial_number_of_men (W : ℝ) (M : ℝ) (h1 : (M * 15) = W / 2) (h2 : ((M - 2) * 25) = W / 2) : M = 5 :=
sorry

end initial_number_of_men_l164_164409


namespace find_circle_eqn_range_of_slope_l164_164480

noncomputable def circle_eqn_through_points (M N : ℝ × ℝ) (line : ℝ → ℝ → Prop) :=
  ∃ (C : ℝ × ℝ) (r : ℝ),
    C ∈ {P : ℝ × ℝ | line P.1 P.2} ∧
    dist C M = dist C N ∧
    (∀ (P : ℝ × ℝ), dist P C = r ↔ (P = M ∨ P = N))

noncomputable def circle_standard_eqn (C : ℝ × ℝ) (r : ℝ) :=
  ∀ (P : ℝ × ℝ), dist P C = r ↔ (P.1 - C.1)^2 + P.2^2 = r^2

theorem find_circle_eqn (M N : ℝ × ℝ) (line : ℝ → ℝ → Prop)
  (h : circle_eqn_through_points M N line) :
  ∃ r : ℝ, circle_standard_eqn (1, 0) r ∧ r = 5 := 
  sorry

theorem range_of_slope (k : ℝ) :
  0 < k → 8 * k^2 - 15 * k > 0 → k > (15 / 8) :=
  sorry

end find_circle_eqn_range_of_slope_l164_164480


namespace distinct_roots_iff_l164_164905

theorem distinct_roots_iff (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 + m + 3 = 0 ∧ x2^2 + m * x2 + m + 3 = 0) ↔ (m < -2 ∨ m > 6) := 
sorry

end distinct_roots_iff_l164_164905


namespace inverse_of_parallel_lines_l164_164448

theorem inverse_of_parallel_lines 
  (P Q : Prop) 
  (parallel_impl_alt_angles : P → Q) :
  (Q → P) := 
by
  sorry

end inverse_of_parallel_lines_l164_164448


namespace regular_polygon_sides_l164_164781

theorem regular_polygon_sides (perimeter side_length : ℝ) (h1 : perimeter = 180) (h2 : side_length = 15) :
  perimeter / side_length = 12 :=
by sorry

end regular_polygon_sides_l164_164781


namespace cuboid_volume_l164_164670

theorem cuboid_volume (a b c : ℝ) (h1 : a * b = 2) (h2 : b * c = 6) (h3 : a * c = 9) : a * b * c = 6 := by
  sorry

end cuboid_volume_l164_164670


namespace lamps_remain_lit_after_toggling_l164_164754

theorem lamps_remain_lit_after_toggling :
  let n := 1997
  let lcm_2_3_5 := Nat.lcm (Nat.lcm 2 3) 5
  let multiples_30 := n / 30
  let multiples_6 := n / (2 * 3)
  let multiples_15 := n / (3 * 5)
  let multiples_10 := n / (2 * 5)
  let multiples_2 := n / 2
  let multiples_3 := n / 3
  let multiples_5 := n / 5
  let pulled_three_times := multiples_30
  let pulled_twice := multiples_6 + multiples_15 + multiples_10 - 3 * pulled_three_times
  let pulled_once := multiples_2 + multiples_3 + multiples_5 - 2 * pulled_twice - 3 * pulled_three_times
  1997 - pulled_three_times - pulled_once = 999 := by
  let n := 1997
  let lcm_2_3_5 := Nat.lcm (Nat.lcm 2 3) 5
  let multiples_30 := n / 30
  let multiples_6 := n / (2 * 3)
  let multiples_15 := n / (3 * 5)
  let multiples_10 := n / (2 * 5)
  let multiples_2 := n / 2
  let multiples_3 := n / 3
  let multiples_5 := n / 5
  let pulled_three_times := multiples_30
  let pulled_twice := multiples_6 + multiples_15 + multiples_10 - 3 * pulled_three_times
  let pulled_once := multiples_2 + multiples_3 + multiples_5 - 2 * pulled_twice - 3 * pulled_three_times
  have h : 1997 - pulled_three_times - (pulled_once) = 999 := sorry
  exact h

end lamps_remain_lit_after_toggling_l164_164754


namespace work_done_by_gas_l164_164044

def gas_constant : ℝ := 8.31 -- J/(mol·K)
def temperature_change : ℝ := 100 -- K (since 100°C increase is equivalent to 100 K in Kelvin)
def moles_of_gas : ℝ := 1 -- one mole of gas

theorem work_done_by_gas :
  (1/2) * gas_constant * temperature_change = 415.5 :=
by sorry

end work_done_by_gas_l164_164044


namespace circle_passing_through_points_l164_164521

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l164_164521


namespace max_card_count_sum_l164_164983

theorem max_card_count_sum (W B R : ℕ) (total_cards : ℕ) 
  (white_cards black_cards red_cards : ℕ) : 
  total_cards = 300 ∧ white_cards = 100 ∧ black_cards = 100 ∧ red_cards = 100 ∧
  (∀ w, w < white_cards → ∃ b, b < black_cards) ∧ 
  (∀ b, b < black_cards → ∃ r, r < red_cards) ∧ 
  (∀ r, r < red_cards → ∃ w, w < white_cards) →
  ∃ max_sum, max_sum = 20000 :=
by
  sorry

end max_card_count_sum_l164_164983


namespace find_m_l164_164526

theorem find_m {m : ℕ} (h1 : Even (m^2 - 2 * m - 3)) (h2 : m^2 - 2 * m - 3 < 0) : m = 1 :=
sorry

end find_m_l164_164526


namespace time_to_run_home_l164_164976

-- Define the conditions
def blocks_run_per_time : ℚ := 2 -- Justin runs 2 blocks
def time_per_blocks : ℚ := 1.5 -- in 1.5 minutes
def blocks_to_home : ℚ := 8 -- Justin is 8 blocks from home

-- Define the theorem to prove the time taken for Justin to run home
theorem time_to_run_home : (blocks_to_home / blocks_run_per_time) * time_per_blocks = 6 :=
by
  sorry

end time_to_run_home_l164_164976


namespace part_I_part_II_l164_164721

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) - 1

theorem part_I {a : ℝ} (ha : a = 2) :
  { x : ℝ | f x a ≥ 4 - abs (x - 4)} = { x | x ≥ 11 / 2 ∨ x ≤ 1 / 2 } := 
by 
  sorry

theorem part_II {a : ℝ} (h : { x : ℝ | abs (f (2 * x + a) a - 2 * f x a) ≤ 1 } = 
      { x | 1 / 2 ≤ x ∧ x ≤ 1 }) : 
  a = 2 := 
by 
  sorry

end part_I_part_II_l164_164721


namespace masha_can_generate_all_integers_up_to_1093_l164_164025

theorem masha_can_generate_all_integers_up_to_1093 :
  ∃ (f : ℕ → ℤ), (∀ n, 1 ≤ n → n ≤ 1093 → f n ∈ {k | ∃ (a b c d e f g : ℤ), a * 1 + b * 3 + c * 9 + d * 27 + e * 81 + f * 243 + g * 729 = k}) :=
sorry

end masha_can_generate_all_integers_up_to_1093_l164_164025


namespace reciprocal_of_2022_l164_164987

theorem reciprocal_of_2022 : 1 / 2022 = (1 : ℝ) / 2022 :=
sorry

end reciprocal_of_2022_l164_164987


namespace minimum_berries_left_l164_164322

def geometric_sum (a r n : ℕ) : ℕ :=
  a * (r ^ n - 1) / (r - 1)

theorem minimum_berries_left {a r n S : ℕ} 
  (h_a : a = 1) 
  (h_r : r = 2) 
  (h_n : n = 100) 
  (h_S : S = geometric_sum a r n) 
  : S = 2^100 - 1 -> ∃ k, k = 100 :=
by
  sorry

end minimum_berries_left_l164_164322


namespace total_cost_of_apples_l164_164495

theorem total_cost_of_apples (cost_per_kg : ℝ) (packaging_fee : ℝ) (weight : ℝ) :
  cost_per_kg = 15.3 →
  packaging_fee = 0.25 →
  weight = 2.5 →
  (weight * (cost_per_kg + packaging_fee) = 38.875) :=
by
  intros h1 h2 h3
  sorry

end total_cost_of_apples_l164_164495


namespace quadratic_equation_value_l164_164699

theorem quadratic_equation_value (a : ℝ) (h₁ : a^2 - 2 = 2) (h₂ : a ≠ 2) : a = -2 :=
by
  sorry

end quadratic_equation_value_l164_164699


namespace johns_old_cards_l164_164769

def cards_per_page : ℕ := 3
def new_cards : ℕ := 8
def total_pages : ℕ := 8

def total_cards := total_pages * cards_per_page
def old_cards := total_cards - new_cards

theorem johns_old_cards :
  old_cards = 16 :=
by
  -- Note: No specific solution steps needed here, just stating the theorem
  sorry

end johns_old_cards_l164_164769


namespace first_part_amount_l164_164720

-- Given Definitions
def total_amount : ℝ := 3200
def interest_rate_part1 : ℝ := 0.03
def interest_rate_part2 : ℝ := 0.05
def total_interest : ℝ := 144

-- The problem to be proven
theorem first_part_amount : 
  ∃ (x : ℝ), 0.03 * x + 0.05 * (3200 - x) = 144 ∧ x = 800 :=
by
  sorry

end first_part_amount_l164_164720


namespace reciprocal_neg_one_thirteen_l164_164203

theorem reciprocal_neg_one_thirteen : -(1:ℝ) / 13⁻¹ = -13 := 
sorry

end reciprocal_neg_one_thirteen_l164_164203


namespace fraction_habitable_surface_l164_164930

noncomputable def fraction_land_not_covered_by_water : ℚ := 1 / 3
noncomputable def fraction_inhabitable_land : ℚ := 2 / 3

theorem fraction_habitable_surface :
  fraction_land_not_covered_by_water * fraction_inhabitable_land = 2 / 9 :=
by
  sorry

end fraction_habitable_surface_l164_164930


namespace gamma_lt_delta_l164_164965

open Real

variables (α β γ δ : ℝ)

-- Hypotheses as given in the problem
axiom h1 : 0 < α 
axiom h2 : α < β
axiom h3 : β < π / 2
axiom hg1 : 0 < γ
axiom hg2 : γ < π / 2
axiom htan_gamma_eq : tan γ = (tan α + tan β) / 2
axiom hd1 : 0 < δ
axiom hd2 : δ < π / 2
axiom hcos_delta_eq : (1 / cos δ) = (1 / 2) * (1 / cos α + 1 / cos β)

-- Goal to prove
theorem gamma_lt_delta : γ < δ := 
by 
sorry

end gamma_lt_delta_l164_164965


namespace transformations_map_onto_self_l164_164907

/-- Define the transformations -/
def T1 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for a 90 degree rotation around the center of a square
  sorry

def T2 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for a translation parallel to line ℓ
  sorry

def T3 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for reflection across line ℓ
  sorry

def T4 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for reflection across a line perpendicular to line ℓ
  sorry

/-- Define the pattern -/
def pattern (p : ℝ × ℝ) : Type :=
  -- Representation of alternating right triangles and squares along line ℓ
  sorry

/-- The main theorem:
    Prove that there are exactly 3 transformations (T1, T2, T3) that will map the pattern onto itself. -/
theorem transformations_map_onto_self : (∃ pattern : ℝ × ℝ → Type,
  (T1 pattern = pattern) ∧
  (T2 pattern = pattern) ∧
  (T3 pattern = pattern) ∧
  ¬ (T4 pattern = pattern)) → (3 = 3) :=
by
  sorry

end transformations_map_onto_self_l164_164907


namespace piggy_bank_donation_l164_164871

theorem piggy_bank_donation (total_earnings : ℕ) (cost_of_ingredients : ℕ) 
  (total_donation_homeless_shelter : ℕ) : 
  (total_earnings = 400) → (cost_of_ingredients = 100) → (total_donation_homeless_shelter = 160) → 
  (total_donation_homeless_shelter - (total_earnings - cost_of_ingredients) / 2 = 10) :=
by
  intros h1 h2 h3
  sorry

end piggy_bank_donation_l164_164871


namespace cricket_player_average_l164_164795

theorem cricket_player_average (A : ℝ) (h1 : 10 * A + 84 = 11 * (A + 4)) : A = 40 :=
by
  sorry

end cricket_player_average_l164_164795


namespace kamal_marks_in_mathematics_l164_164184

def kamal_marks_english : ℕ := 96
def kamal_marks_physics : ℕ := 82
def kamal_marks_chemistry : ℕ := 67
def kamal_marks_biology : ℕ := 85
def kamal_average_marks : ℕ := 79
def kamal_number_of_subjects : ℕ := 5

theorem kamal_marks_in_mathematics :
  let total_marks := kamal_average_marks * kamal_number_of_subjects
  let total_known_marks := kamal_marks_english + kamal_marks_physics + kamal_marks_chemistry + kamal_marks_biology
  total_marks - total_known_marks = 65 :=
by
  sorry

end kamal_marks_in_mathematics_l164_164184


namespace min_value_of_a_l164_164746

theorem min_value_of_a (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (x + y) * (1/x + a/y) ≥ 16) : a ≥ 9 :=
sorry

end min_value_of_a_l164_164746


namespace certain_number_is_32_l164_164714

theorem certain_number_is_32 (k t : ℚ) (certain_number : ℚ) 
  (h1 : t = 5/9 * (k - certain_number))
  (h2 : t = 75) (h3 : k = 167) :
  certain_number = 32 :=
sorry

end certain_number_is_32_l164_164714


namespace author_hardcover_percentage_l164_164643

variable {TotalPaperCopies : Nat}
variable {PricePerPaperCopy : ℝ}
variable {TotalHardcoverCopies : Nat}
variable {PricePerHardcoverCopy : ℝ}
variable {PaperPercentage : ℝ}
variable {TotalEarnings : ℝ}

theorem author_hardcover_percentage (TotalPaperCopies : Nat)
  (PricePerPaperCopy : ℝ) (TotalHardcoverCopies : Nat)
  (PricePerHardcoverCopy : ℝ) (PaperPercentage TotalEarnings : ℝ)
  (h1 : TotalPaperCopies = 32000) (h2 : PricePerPaperCopy = 0.20)
  (h3 : TotalHardcoverCopies = 15000) (h4 : PricePerHardcoverCopy = 0.40)
  (h5 : PaperPercentage = 0.06) (h6 : TotalEarnings = 1104) :
  (720 / (15000 * 0.40) * 100) = 12 := by
  sorry

end author_hardcover_percentage_l164_164643


namespace exists_v_satisfying_equation_l164_164825

noncomputable def custom_operation (v : ℝ) : ℝ :=
  v - (v / 3) + Real.sin v

theorem exists_v_satisfying_equation :
  ∃ v : ℝ, custom_operation (custom_operation v) = 24 := 
sorry

end exists_v_satisfying_equation_l164_164825


namespace discount_difference_l164_164376

def original_amount : ℚ := 20000
def single_discount_rate : ℚ := 0.30
def first_discount_rate : ℚ := 0.25
def second_discount_rate : ℚ := 0.05

theorem discount_difference :
  (original_amount * (1 - single_discount_rate)) - (original_amount * (1 - first_discount_rate) * (1 - second_discount_rate)) = 250 := by
  sorry

end discount_difference_l164_164376


namespace ab_value_l164_164045

theorem ab_value (a b c : ℤ) (h1 : a^2 = 16) (h2 : 2 * a * b = -40) : a * b = -20 := 
sorry

end ab_value_l164_164045


namespace arithmetic_sequence_l164_164219

noncomputable def M (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.sum (Finset.range n) (λ i => a (i + 1))) / n

theorem arithmetic_sequence (a : ℕ → ℝ) (C : ℝ)
  (h : ∀ {i j k : ℕ}, i ≠ j → j ≠ k → k ≠ i →
    (i - j) * M a k + (j - k) * M a i + (k - i) * M a j = C) :
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a 1 + n * d :=
sorry

end arithmetic_sequence_l164_164219


namespace number_of_boys_l164_164780

variable (x y : ℕ)

theorem number_of_boys (h1 : x + y = 900) (h2 : y = (x / 100) * 900) : x = 90 :=
by
  sorry

end number_of_boys_l164_164780
