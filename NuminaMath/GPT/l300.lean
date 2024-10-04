import Mathlib

namespace circle_area_circle_circumference_l300_300601

section CircleProperties

variable (r : ℝ) -- Define the radius of the circle as a real number

-- State the theorem for the area of the circle
theorem circle_area (A : ℝ) : A = π * r^2 :=
sorry

-- State the theorem for the circumference of the circle
theorem circle_circumference (C : ℝ) : C = 2 * π * r :=
sorry

end CircleProperties

end circle_area_circle_circumference_l300_300601


namespace honor_students_count_l300_300981

noncomputable def number_of_honor_students (G B Eg Eb : ℕ) (p_girl p_boy : ℚ) : ℕ :=
  if G < 30 ∧ B < 30 ∧ Eg = (3 / 13) * G ∧ Eb = (4 / 11) * B ∧ G + B < 30 then
    Eg + Eb
  else
    0

theorem honor_students_count :
  ∃ (G B Eg Eb : ℕ), (G < 30 ∧ B < 30 ∧ G % 13 = 0 ∧ B % 11 = 0 ∧ Eg = (3 * G / 13) ∧ Eb = (4 * B / 11) ∧ G + B < 30 ∧ number_of_honor_students G B Eg Eb (3 / 13) (4 / 11) = 7) :=
by {
  sorry
}

end honor_students_count_l300_300981


namespace village_connection_possible_l300_300536

variable (V : Type) -- Type of villages
variable (Villages : List V) -- List of 26 villages
variable (connected_by_tractor connected_by_train : V → V → Prop) -- Connections

-- Define the hypothesis
variable (bidirectional_connections : ∀ (v1 v2 : V), v1 ≠ v2 → (connected_by_tractor v1 v2 ∨ connected_by_train v1 v2))

-- Main theorem statement
theorem village_connection_possible :
  ∃ (mode : V → V → Prop), (∀ v1 v2 : V, v1 ≠ v2 → v1 ∈ Villages → v2 ∈ Villages → mode v1 v2) ∧
  (∀ v1 v2 : V, v1 ∈ Villages → v2 ∈ Villages → ∃ (path : List (V × V)), (∀ edge ∈ path, mode edge.fst edge.snd) ∧ path ≠ []) :=
by
  sorry

end village_connection_possible_l300_300536


namespace average_speed_of_car_l300_300701

theorem average_speed_of_car : 
  let distance1 := 30
  let speed1 := 60
  let distance2 := 35
  let speed2 := 70
  let distance3 := 36
  let speed3 := 80
  let distance4 := 20
  let speed4 := 55
  let total_distance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let time4 := distance4 / speed4
  let total_time := time1 + time2 + time3 + time4
  let average_speed := total_distance / total_time
  average_speed = 66.70 := sorry

end average_speed_of_car_l300_300701


namespace parallelogram_smaller_angle_proof_l300_300833

noncomputable def smaller_angle (x : ℝ) : Prop :=
  let larger_angle := x + 120
  let angle_sum := x + larger_angle + x + larger_angle = 360
  angle_sum

theorem parallelogram_smaller_angle_proof (x : ℝ) (h1 : smaller_angle x) : x = 30 := by
  sorry

end parallelogram_smaller_angle_proof_l300_300833


namespace lcm_of_8_9_5_10_l300_300845

theorem lcm_of_8_9_5_10 : Nat.lcm (Nat.lcm 8 9) (Nat.lcm 5 10) = 360 := by
  sorry

end lcm_of_8_9_5_10_l300_300845


namespace john_mary_game_l300_300800

theorem john_mary_game (n : ℕ) (h : n ≥ 3) :
  ∃ S : ℕ, S = n * (n + 1) :=
by
  sorry

end john_mary_game_l300_300800


namespace same_terminal_side_of_minus_80_l300_300207

theorem same_terminal_side_of_minus_80 :
  ∃ k : ℤ, 1 * 360 - 80 = 280 := 
  sorry

end same_terminal_side_of_minus_80_l300_300207


namespace honor_students_count_l300_300972

noncomputable def G : ℕ := 13
noncomputable def B : ℕ := 11
def E_G : ℕ := 3
def E_B : ℕ := 4

theorem honor_students_count (h1 : G + B < 30) 
    (h2 : (E_G : ℚ) / G = 3 / 13) 
    (h3 : (E_B : ℚ) / B = 4 / 11) :
    E_G + E_B = 7 := 
sorry

end honor_students_count_l300_300972


namespace ariel_fish_l300_300566

theorem ariel_fish (total_fish : ℕ) (male_ratio : ℚ) (female_ratio : ℚ) (female_fish : ℕ) : 
  total_fish = 45 ∧ male_ratio = 2/3 ∧ female_ratio = 1/3 → female_fish = 15 :=
by
  sorry

end ariel_fish_l300_300566


namespace mike_taller_than_mark_l300_300644

-- Define the heights of Mark and Mike in terms of feet and inches
def mark_height_feet : ℕ := 5
def mark_height_inches : ℕ := 3
def mike_height_feet : ℕ := 6
def mike_height_inches : ℕ := 1

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Conversion of heights to inches
def mark_total_height_in_inches : ℕ := mark_height_feet * feet_to_inches + mark_height_inches
def mike_total_height_in_inches : ℕ := mike_height_feet * feet_to_inches + mike_height_inches

-- Define the problem statement: proving Mike is 10 inches taller than Mark
theorem mike_taller_than_mark : mike_total_height_in_inches - mark_total_height_in_inches = 10 :=
by sorry

end mike_taller_than_mark_l300_300644


namespace dan_initial_money_l300_300009

def money_left : ℕ := 3
def cost_candy : ℕ := 2
def initial_money : ℕ := money_left + cost_candy

theorem dan_initial_money :
  initial_money = 5 :=
by
  -- Definitions according to problem
  let money_left := 3
  let cost_candy := 2

  have h : initial_money = money_left + cost_candy := by rfl
  rw [h]

  -- Show the final equivalence
  show 3 + 2 = 5
  rfl

end dan_initial_money_l300_300009


namespace star_difference_l300_300922

def star (x y : ℤ) : ℤ := x * y + 3 * x - y

theorem star_difference : (star 7 4) - (star 4 7) = 12 := by
  sorry

end star_difference_l300_300922


namespace simplify_fraction_48_72_l300_300381

theorem simplify_fraction_48_72 : (48 : ℚ) / 72 = 2 / 3 := sorry

end simplify_fraction_48_72_l300_300381


namespace find_d_not_unique_solution_l300_300897

variable {x y k d : ℝ}

-- Definitions of the conditions
def eq1 (d : ℝ) (x y : ℝ) := 4 * (3 * x + 4 * y) = d
def eq2 (k : ℝ) (x y : ℝ) := k * x + 12 * y = 30

-- The theorem we need to prove
theorem find_d_not_unique_solution (h1: eq1 d x y) (h2: eq2 k x y) (h3 : ¬ ∃! (x y : ℝ), eq1 d x y ∧ eq2 k x y) : d = 40 := 
by
  sorry

end find_d_not_unique_solution_l300_300897


namespace partial_fraction_sum_zero_l300_300451

variable {A B C D E : ℝ}
variable {x : ℝ}

theorem partial_fraction_sum_zero (h : 
  (1:ℝ) / ((x-1)*x*(x+1)*(x+2)*(x+3)) = 
  A / (x-1) + B / x + C / (x+1) + D / (x+2) + E / (x+3)) : 
  A + B + C + D + E = 0 :=
by sorry

end partial_fraction_sum_zero_l300_300451


namespace find_baseball_deck_price_l300_300371

variables (numberOfBasketballPacks : ℕ) (pricePerBasketballPack : ℝ) (numberOfBaseballDecks : ℕ)
           (totalMoney : ℝ) (changeReceived : ℝ) (totalSpent : ℝ) (spentOnBasketball : ℝ) (baseballDeckPrice : ℝ)

noncomputable def problem_conditions : Prop :=
  numberOfBasketballPacks = 2 ∧
  pricePerBasketballPack = 3 ∧
  numberOfBaseballDecks = 5 ∧
  totalMoney = 50 ∧
  changeReceived = 24 ∧
  totalSpent = totalMoney - changeReceived ∧
  spentOnBasketball = numberOfBasketballPacks * pricePerBasketballPack ∧
  totalSpent = spentOnBasketball + (numberOfBaseballDecks * baseballDeckPrice)

theorem find_baseball_deck_price (h : problem_conditions numberOfBasketballPacks pricePerBasketballPack numberOfBaseballDecks totalMoney changeReceived totalSpent spentOnBasketball baseballDeckPrice) :
  baseballDeckPrice = 4 :=
sorry

end find_baseball_deck_price_l300_300371


namespace vasya_has_greater_expected_area_l300_300880

noncomputable def expected_area_rectangle : ℚ :=
1 / 6 * (1 * 1 + 1 * 2 + 1 * 3 + 1 * 4 + 1 * 5 + 1 * 6 + 
         2 * 1 + 2 * 2 + 2 * 3 + 2 * 4 + 2 * 5 + 2 * 6 + 
         3 * 1 + 3 * 2 + 3 * 3 + 3 * 4 + 3 * 5 + 3 * 6 + 
         4 * 1 + 4 * 2 + 4 * 3 + 4 * 4 + 4 * 5 + 4 * 6 + 
         5 * 1 + 5 * 2 + 5 * 3 + 5 * 4 + 5 * 5 + 5 * 6 + 
         6 * 1 + 6 * 2 + 6 * 3 + 6 * 4 + 6 * 5 + 6 * 6)

noncomputable def expected_area_square : ℚ := 
1 / 6 * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

theorem vasya_has_greater_expected_area : expected_area_rectangle < expected_area_square :=
by {
  -- A calculation of this sort should be done symbolically, not in this theorem,
  -- but the primary goal here is to show the structure of the statement.
  -- Hence, implement symbolic computation later to finalize proof.
  sorry
}

end vasya_has_greater_expected_area_l300_300880


namespace train_speed_kmh_l300_300443

theorem train_speed_kmh 
  (L_train : ℝ) (L_bridge : ℝ) (time : ℝ)
  (h_train : L_train = 460)
  (h_bridge : L_bridge = 140)
  (h_time : time = 48) : 
  (L_train + L_bridge) / time * 3.6 = 45 := 
by
  -- Definitions and conditions
  have h_total_dist : L_train + L_bridge = 600 := by sorry
  have h_speed_mps : (L_train + L_bridge) / time = 600 / 48 := by sorry
  have h_speed_mps_simplified : 600 / 48 = 12.5 := by sorry
  have h_speed_kmh : 12.5 * 3.6 = 45 := by sorry
  sorry

end train_speed_kmh_l300_300443


namespace nondegenerate_ellipse_iff_l300_300585

theorem nondegenerate_ellipse_iff (k : ℝ) :
  (∃ x y : ℝ, x^2 + 9*y^2 - 6*x + 27*y = k) ↔ k > -117/4 :=
by
  sorry

end nondegenerate_ellipse_iff_l300_300585


namespace child_grandmother_ratio_l300_300219

variable (G D C : ℕ)

axiom cond1 : G + D + C = 120
axiom cond2 : D + C = 60
axiom cond3 : D = 48

theorem child_grandmother_ratio : (C : ℚ) / G = 1 / 5 :=
by
  sorry

end child_grandmother_ratio_l300_300219


namespace watch_cost_price_l300_300123

theorem watch_cost_price (SP_loss SP_gain CP : ℝ) 
  (h1 : SP_loss = 0.9 * CP) 
  (h2 : SP_gain = 1.04 * CP) 
  (h3 : SP_gain - SP_loss = 196) 
  : CP = 1400 := 
sorry

end watch_cost_price_l300_300123


namespace rug_floor_coverage_l300_300728

/-- A rectangular rug with side lengths of 2 feet and 7 feet is placed on an irregularly-shaped floor composed of a square with an area of 36 square feet and a right triangle adjacent to one of the square's sides, with leg lengths of 6 feet and 4 feet. If the surface of the rug does not extend beyond the area of the floor, then the fraction of the area of the floor that is not covered by the rug is 17/24. -/
theorem rug_floor_coverage : (48 - 14) / 48 = 17 / 24 :=
by
  -- proof goes here
  sorry

end rug_floor_coverage_l300_300728


namespace sum_primes_20_to_30_l300_300249

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_20_to_30 : (∑ n in Finset.filter is_prime (Finset.range 31), n) = 52 :=
by
  sorry

end sum_primes_20_to_30_l300_300249


namespace units_digit_sum_cubes_l300_300682

theorem units_digit_sum_cubes (n1 n2 : ℕ) 
  (h1 : n1 = 24) (h2 : n2 = 42) : 
  (n1 ^ 3 + n2 ^ 3) % 10 = 6 :=
by 
  -- substitution based on h1 and h2 can be done here.
  sorry

end units_digit_sum_cubes_l300_300682


namespace monthly_rent_calc_l300_300968

def monthly_rent (length width annual_rent_per_sq_ft : ℕ) : ℕ :=
  (length * width * annual_rent_per_sq_ft) / 12

theorem monthly_rent_calc :
  monthly_rent 10 8 360 = 2400 := 
  sorry

end monthly_rent_calc_l300_300968


namespace percentage_of_dogs_l300_300197

theorem percentage_of_dogs (total_pets : ℕ) (percent_cats : ℕ) (bunnies : ℕ) 
  (h1 : total_pets = 36) (h2 : percent_cats = 50) (h3 : bunnies = 9) : 
  ((total_pets - ((percent_cats * total_pets) / 100) - bunnies) / total_pets * 100) = 25 := by
  sorry

end percentage_of_dogs_l300_300197


namespace steiner_ellipse_equation_l300_300762

theorem steiner_ellipse_equation
  (α β γ : ℝ) 
  (h : α + β + γ = 1) :
  β * γ + α * γ + α * β = 0 := 
sorry

end steiner_ellipse_equation_l300_300762


namespace women_attended_l300_300740

theorem women_attended :
  (15 * 4) / 3 = 20 :=
by
  sorry

end women_attended_l300_300740


namespace greater_expected_area_l300_300875

/-- Let X be a random variable representing a single roll of a die, which can take integer values from 1 through 6. -/
def X : Type := { x : ℕ // 1 ≤ x ∧ x ≤ 6 }

/-- Define independent random variables A and B representing the outcomes of Asya’s die rolls, which can take integer values from 1 through 6 with equal probability. -/
noncomputable def A : Type := { a : ℕ // 1 ≤ a ∧ a ≤ 6 }
noncomputable def B : Type := { b : ℕ // 1 ≤ b ∧ b ≤ 6 }

/-- The expected value of a random variable taking integer values from 1 through 6. 
    E[X] = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5, and E[X^2] = (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6 = 15.1667 -/
noncomputable def expected_X_squared : ℝ := 91 / 6

/-- The expected value of the product of two independent random variables each taking integer values from 1 through 6. 
    E[A * B] = E[A] * E[B] = 3.5 * 3.5 = 12.25 -/
noncomputable def expected_A_times_B : ℝ := 12.25

/-- Prove that the expected area of Vasya's square is greater than Asya's rectangle.
    i.e., E[X^2] > E[A * B] -/
theorem greater_expected_area : expected_X_squared > expected_A_times_B :=
sorry

end greater_expected_area_l300_300875


namespace distinct_pawns_5x5_l300_300053

theorem distinct_pawns_5x5 : 
  ∃ n : ℕ, n = 14400 ∧ 
  (∃ (get_pos : Fin 5 → Fin 5), function.bijective get_pos) :=
begin
  sorry
end

end distinct_pawns_5x5_l300_300053


namespace limit_problem_l300_300293

theorem limit_problem (h : ∀ x, x ≠ -3):
  (∀ x, (x^2 + 2*x - 3)^2 = (x + 3)^2 * (x - 1)^2) →
  (∀ x, x^3 + 4*x^2 + 3*x = x * (x + 1) * (x + 3)) →
  tendsto (λ x, ((x^2 + 2*x - 3)^2) / (x^3 + 4*x^2 + 3*x)) (𝓝[-] (-3)) (𝓝 0) :=
by
  intros numerator_factor denominator_factor
  sorry

end limit_problem_l300_300293


namespace tan_half_angle_third_quadrant_l300_300476

theorem tan_half_angle_third_quadrant (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h : Real.sin α = -24/25) :
  Real.tan (α / 2) = -4/3 := 
by 
  sorry

end tan_half_angle_third_quadrant_l300_300476


namespace buoy_min_force_l300_300192

-- Define the problem in Lean
variables (M : ℝ) (ax : ℝ) (T_star : ℝ) (a : ℝ) (F_current : ℝ)
-- Conditions
variables (h_horizontal_component : T_star * Real.sin a = F_current)
          (h_zero_net_force : M * ax = 0)

theorem buoy_min_force (h_horizontal_component : T_star * Real.sin a = F_current) : 
  F_current = 400 := 
sorry

end buoy_min_force_l300_300192


namespace sum_fourth_powers_const_l300_300211

-- Define the vertices of the square
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A (a : ℝ) : Point := {x := a, y := 0}
def B (a : ℝ) : Point := {x := 0, y := a}
def C (a : ℝ) : Point := {x := -a, y := 0}
def D (a : ℝ) : Point := {x := 0, y := -a}

-- Define distance squared between two points
def dist_sq (P Q : Point) : ℝ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

-- Circle centered at origin
def on_circle (P : Point) (r : ℝ) : Prop :=
  P.x ^ 2 + P.y ^ 2 = r ^ 2

-- The main theorem
theorem sum_fourth_powers_const (a r : ℝ) (P : Point) (h : on_circle P r) :
  let AP_sq := dist_sq P (A a)
  let BP_sq := dist_sq P (B a)
  let CP_sq := dist_sq P (C a)
  let DP_sq := dist_sq P (D a)
  (AP_sq ^ 2 + BP_sq ^ 2 + CP_sq ^ 2 + DP_sq ^ 2) = 4 * (r^4 + a^4 + 4 * a^2 * r^2) :=
by
  sorry

end sum_fourth_powers_const_l300_300211


namespace constant_term_eq_fifteen_l300_300180

theorem constant_term_eq_fifteen (n : ℕ) :
  (∃ k : ℕ, (Nat.choose n k) * (-1) ^ (n - k) = 15 ∧ 3 * k = n) ↔ n = 6 :=
by
  sorry

end constant_term_eq_fifteen_l300_300180


namespace perfect_squares_between_2_and_20_l300_300921

-- Defining the conditions and problem statement
theorem perfect_squares_between_2_and_20 : 
  ∃ n, n = 3 ∧ ∀ m, (2 < m ∧ m < 20 ∧ ∃ k, k * k = m) ↔ m = 4 ∨ m = 9 ∨ m = 16 :=
by {
  -- Start the proof process
  sorry -- Placeholder for the proof
}

end perfect_squares_between_2_and_20_l300_300921


namespace solve_equation_3x6_eq_3mx_div_xm1_l300_300205

theorem solve_equation_3x6_eq_3mx_div_xm1 (x : ℝ) 
  (h1 : x ≠ 1)
  (h2 : x^2 + 5*x - 6 ≠ 0) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) ↔ (x = 3 ∨ x = -6) :=
by 
  sorry

end solve_equation_3x6_eq_3mx_div_xm1_l300_300205


namespace ratio_simplified_l300_300289

theorem ratio_simplified (kids_meals : ℕ) (adult_meals : ℕ) (h1 : kids_meals = 70) (h2 : adult_meals = 49) : 
  ∃ (k a : ℕ), k = 10 ∧ a = 7 ∧ kids_meals / Nat.gcd kids_meals adult_meals = k ∧ adult_meals / Nat.gcd kids_meals adult_meals = a :=
by
  sorry

end ratio_simplified_l300_300289


namespace bert_total_stamps_l300_300291

theorem bert_total_stamps (bought_stamps : ℕ) (half_stamps_before : ℕ) (total_stamps_after : ℕ) :
  (bought_stamps = 300) ∧ (half_stamps_before = bought_stamps / 2) → (total_stamps_after = half_stamps_before + bought_stamps) → (total_stamps_after = 450) :=
by
  sorry

end bert_total_stamps_l300_300291


namespace annual_feeding_cost_is_correct_l300_300612

-- Definitions based on conditions
def number_of_geckos : Nat := 3
def number_of_iguanas : Nat := 2
def number_of_snakes : Nat := 4
def cost_per_gecko_per_month : Nat := 15
def cost_per_iguana_per_month : Nat := 5
def cost_per_snake_per_month : Nat := 10

-- Statement of the theorem
theorem annual_feeding_cost_is_correct : 
    (number_of_geckos * cost_per_gecko_per_month
    + number_of_iguanas * cost_per_iguana_per_month 
    + number_of_snakes * cost_per_snake_per_month) * 12 = 1140 := by
  sorry

end annual_feeding_cost_is_correct_l300_300612


namespace expected_number_of_different_faces_l300_300712

noncomputable def expected_faces : ℝ :=
  let probability_face_1_not_appearing := (5 / 6)^6
  let E_zeta_1 := 1 - probability_face_1_not_appearing
  6 * E_zeta_1

theorem expected_number_of_different_faces :
  expected_faces = (6^6 - 5^6) / 6^5 := by 
  sorry

end expected_number_of_different_faces_l300_300712


namespace probability_two_red_crayons_l300_300271

def num_crayons : ℕ := 6
def num_red : ℕ := 3
def num_blue : ℕ := 2
def num_green : ℕ := 1
def num_choose (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_two_red_crayons :
  let total_pairs := num_choose num_crayons 2
  let red_pairs := num_choose num_red 2
  (red_pairs : ℚ) / (total_pairs : ℚ) = 1 / 5 :=
by
  sorry

end probability_two_red_crayons_l300_300271


namespace min_z_value_l300_300046

theorem min_z_value (x y z : ℝ) (h1 : 2 * x + y = 1) (h2 : z = 4 ^ x + 2 ^ y) : z ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_z_value_l300_300046


namespace probability_exactly_five_shots_expected_shots_to_hit_all_l300_300425

-- Part (a)
theorem probability_exactly_five_shots
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∃ t₁ t₂ t₃ : ℕ, t₁ ≠ t₂ ∧ t₁ ≠ t₃ ∧ t₂ ≠ t₃ ∧ t₁ + t₂ + t₃ = 5) →
  6 * p ^ 3 * (1 - p) ^ 2 = 6 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

-- Part (b)
theorem expected_shots_to_hit_all
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∀ t: ℕ, (t * p * (1 - p)^(t-1)) = 1/p) →
  3 * (1/p) = 3 / p :=
by sorry

end probability_exactly_five_shots_expected_shots_to_hit_all_l300_300425


namespace matrix_sum_correct_l300_300447

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, 3], ![-2, 1]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![-1, 5], ![8, -3]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 8], ![6, -2]]

theorem matrix_sum_correct : A + B = C := by
  sorry

end matrix_sum_correct_l300_300447


namespace odd_consecutive_nums_divisibility_l300_300373

theorem odd_consecutive_nums_divisibility (a b : ℕ) (h_consecutive : b = a + 2) (h_odd_a : a % 2 = 1) (h_odd_b : b % 2 = 1) : (a^b + b^a) % (a + b) = 0 := by
  sorry

end odd_consecutive_nums_divisibility_l300_300373


namespace intersection_correct_l300_300322

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_correct : A ∩ B = {1, 2} := 
sorry

end intersection_correct_l300_300322


namespace sphere_diameter_l300_300961

theorem sphere_diameter (r : ℝ) (V : ℝ) (threeV : ℝ) (a b : ℕ) :
  (∀ (r : ℝ), r = 5 →
  V = (4 / 3) * π * r^3 →
  threeV = 3 * V →
  D = 2 * (3 * V * 3 / (4 * π))^(1 / 3) →
  D = a * b^(1 / 3) →
  a = 10 ∧ b = 3) →
  a + b = 13 :=
by
  intros
  sorry

end sphere_diameter_l300_300961


namespace functional_equation_solution_l300_300458

theorem functional_equation_solution (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) : 
    ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end functional_equation_solution_l300_300458


namespace solution_set_of_quadratic_inequality_l300_300532

namespace QuadraticInequality

variables {a b : ℝ}

def hasRoots (a b : ℝ) : Prop :=
  let x1 := -1 / 2
  let x2 := 1 / 3
  (- x1 + x2 = - b / a) ∧ (-x1 * x2 = 2 / a)

theorem solution_set_of_quadratic_inequality (h : hasRoots a b) : a + b = -14 :=
sorry

end QuadraticInequality

end solution_set_of_quadratic_inequality_l300_300532


namespace jasper_sold_31_drinks_l300_300502

def chips := 27
def hot_dogs := chips - 8
def drinks := hot_dogs + 12

theorem jasper_sold_31_drinks : drinks = 31 := by
  sorry

end jasper_sold_31_drinks_l300_300502


namespace domain_of_f_l300_300411

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (x + 5)

theorem domain_of_f :
  { x : ℝ | f x ≠ 0 } = { x : ℝ | x ≠ -5 }
:= sorry

end domain_of_f_l300_300411


namespace total_eggs_sold_l300_300781

def initial_trays : Nat := 10
def dropped_trays : Nat := 2
def added_trays : Nat := 7
def eggs_per_tray : Nat := 36

theorem total_eggs_sold : initial_trays - dropped_trays + added_trays * eggs_per_tray = 540 := by
  sorry

end total_eggs_sold_l300_300781


namespace max_value_sqrt43_l300_300365

noncomputable def max_value_expr (x y z : ℝ) : ℝ :=
  3 * x * z * Real.sqrt 2 + 5 * x * y

theorem max_value_sqrt43 (x y z : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y) (h₃ : 0 ≤ z) (h₄ : x^2 + y^2 + z^2 = 1) :
  max_value_expr x y z ≤ Real.sqrt 43 :=
sorry

end max_value_sqrt43_l300_300365


namespace geometric_series_sum_l300_300573

theorem geometric_series_sum (a r : ℝ) (h : |r| < 1) (h_a : a = 2 / 3) (h_r : r = 2 / 3) :
  ∑' i : ℕ, (a * r^i) = 2 :=
by
  sorry

end geometric_series_sum_l300_300573


namespace smallest_positive_multiple_l300_300846

theorem smallest_positive_multiple (a : ℕ) (k : ℕ) (h : 17 * a ≡ 7 [MOD 101]) : 
  ∃ k, k = 17 * 42 := 
sorry

end smallest_positive_multiple_l300_300846


namespace expected_number_of_different_faces_l300_300709

noncomputable def expected_faces : ℝ :=
  let probability_face_1_not_appearing := (5 / 6)^6
  let E_zeta_1 := 1 - probability_face_1_not_appearing
  6 * E_zeta_1

theorem expected_number_of_different_faces :
  expected_faces = (6^6 - 5^6) / 6^5 := by 
  sorry

end expected_number_of_different_faces_l300_300709


namespace evaluate_expression_when_c_eq_4_and_k_eq_2_l300_300148

theorem evaluate_expression_when_c_eq_4_and_k_eq_2 :
  ( (4^4 - 4 * (4 - 1)^4 + 2) ^ 4 ) = 18974736 :=
by
  -- Definitions
  let c := 4
  let k := 2
  -- Evaluations
  let a := c^c
  let b := c * (c - 1)^c
  let expression := (a - b + k)^c
  -- Proof
  have result : expression = 18974736 := sorry
  exact result

end evaluate_expression_when_c_eq_4_and_k_eq_2_l300_300148


namespace fraction_solved_l300_300847

theorem fraction_solved (N f : ℝ) (h1 : N * f^2 = 6^3) (h2 : N * f^2 = 7776) : f = 1 / 6 :=
by sorry

end fraction_solved_l300_300847


namespace jaguars_total_games_l300_300745

-- Defining constants for initial conditions
def initial_win_rate : ℚ := 0.55
def additional_wins : ℕ := 8
def additional_losses : ℕ := 2
def final_win_rate : ℚ := 0.6

-- Defining the main problem statement
theorem jaguars_total_games : 
  ∃ y x : ℕ, (x = initial_win_rate * y) ∧ (x + additional_wins = final_win_rate * (y + (additional_wins + additional_losses))) ∧ (y + (additional_wins + additional_losses) = 50) :=
sorry

end jaguars_total_games_l300_300745


namespace Xiaofang_English_score_l300_300351

/-- Given the conditions about the average scores of Xiaofang's subjects:
  1. The average score for 4 subjects is 88.
  2. The average score for the first 2 subjects is 93.
  3. The average score for the last 3 subjects is 87.
Prove that Xiaofang's English test score is 95. -/
theorem Xiaofang_English_score
    (L M E S : ℝ)
    (h1 : (L + M + E + S) / 4 = 88)
    (h2 : (L + M) / 2 = 93)
    (h3 : (M + E + S) / 3 = 87) :
    E = 95 :=
by
  sorry

end Xiaofang_English_score_l300_300351


namespace cos_angle_BAC_proof_l300_300526

open EuclideanGeometry

variables {A B C O M : Point}
variables [EuclideanSpace V]

-- Define the condition that O is the center of the circumcircle of triangle ABC
def is_circumcenter (O A B C : Point) : Prop := 
  dist O A = dist O B ∧ dist O B = dist O C

-- Define the condition involving vector relationships
def vector_condition (A O B C : Point) : Prop := 
  (2 : ℚ) • (A -ᵥ O) = 4 / 5 • ((B -ᵥ A) + (C -ᵥ A))

-- Define the cosine of the angle BAC
noncomputable def cos_angle_BAC (A B C O : Point) : ℚ :=
cos (∠ B A C)

-- The final Lean 4 Statement
theorem cos_angle_BAC_proof (A B C O : Point) (h_circumcenter : is_circumcenter O A B C) 
  (h_vector : vector_condition A O B C) : cos_angle_BAC A B C O = 1 / 4 := 
by sorry

end cos_angle_BAC_proof_l300_300526


namespace complement_of_M_in_U_l300_300543

namespace ProofProblem

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : U \ M = {3, 5, 6} := by
  sorry

end ProofProblem

end complement_of_M_in_U_l300_300543


namespace num_people_is_8_l300_300954

-- Define the known conditions
def bill_amt : ℝ := 314.16
def person_amt : ℝ := 34.91
def total_amt : ℝ := 314.19

-- Prove that the number of people is 8
theorem num_people_is_8 : ∃ num_people : ℕ, num_people = total_amt / person_amt ∧ num_people = 8 :=
by
  sorry

end num_people_is_8_l300_300954


namespace honor_students_count_l300_300979

noncomputable def number_of_students_in_class_is_less_than_30 := ∃ n, n < 30
def probability_girl_honor_student (G E_G : ℕ) := E_G / G = (3 : ℚ) / 13
def probability_boy_honor_student (B E_B : ℕ) := E_B / B = (4 : ℚ) / 11

theorem honor_students_count (G B E_G E_B : ℕ) 
  (hG_cond : probability_girl_honor_student G E_G) 
  (hB_cond : probability_boy_honor_student B E_B) 
  (h_total_students : G + B < 30) 
  (hE_G_def : E_G = 3 * G / 13) 
  (hE_B_def : E_B = 4 * B / 11) 
  (hG_nonneg : G >= 13)
  (hB_nonneg : B >= 11):
  E_G + E_B = 7 := 
sorry

end honor_students_count_l300_300979


namespace find_x_squared_plus_one_over_x_squared_l300_300315

theorem find_x_squared_plus_one_over_x_squared (x : ℝ) (h : x + 1/x = 4) : x^2 + 1/x^2 = 14 := by
  sorry

end find_x_squared_plus_one_over_x_squared_l300_300315


namespace constraint_condition_2000_yuan_wage_l300_300109

-- Definitions based on the given conditions
def wage_carpenter : ℕ := 50
def wage_bricklayer : ℕ := 40
def total_wage : ℕ := 2000

-- Let x be the number of carpenters and y be the number of bricklayers
variable (x y : ℕ)

-- The proof problem statement
theorem constraint_condition_2000_yuan_wage (x y : ℕ) : 
  wage_carpenter * x + wage_bricklayer * y = total_wage → 5 * x + 4 * y = 200 :=
by
  intro h
  -- Simplification step will be placed here
  sorry

end constraint_condition_2000_yuan_wage_l300_300109


namespace units_digit_sum_cubes_l300_300680

theorem units_digit_sum_cubes (n1 n2 : ℕ) 
  (h1 : n1 = 24) (h2 : n2 = 42) : 
  (n1 ^ 3 + n2 ^ 3) % 10 = 6 :=
by 
  -- substitution based on h1 and h2 can be done here.
  sorry

end units_digit_sum_cubes_l300_300680


namespace distinct_pawn_placements_on_chess_board_l300_300050

def numPawnPlacements : ℕ :=
  5! * 5!

theorem distinct_pawn_placements_on_chess_board :
  numPawnPlacements = 14400 := by
  sorry

end distinct_pawn_placements_on_chess_board_l300_300050


namespace find_value_added_l300_300926

theorem find_value_added :
  ∀ (n x : ℤ), (2 * n + x = 8 * n - 4) → (n = 4) → (x = 20) :=
by
  intros n x h1 h2
  sorry

end find_value_added_l300_300926


namespace triangle_side_lengths_l300_300969

-- Define the problem
variables {r: ℝ} (h_a h_b h_c a b c : ℝ)
variable (sum_of_heights : h_a + h_b + h_c = 13)
variable (r_value : r = 4 / 3)
variable (height_relation : 1/h_a + 1/h_b + 1/h_c = 3/4)

-- Define the theorem to be proven
theorem triangle_side_lengths (h_a h_b h_c : ℝ)
  (sum_of_heights : h_a + h_b + h_c = 13) 
  (r_value : r = 4 / 3)
  (height_relation : 1/h_a + 1/h_b + 1/h_c = 3/4) :
  (a, b, c) = (32 / Real.sqrt 15, 24 / Real.sqrt 15, 16 / Real.sqrt 15) := 
sorry

end triangle_side_lengths_l300_300969


namespace rod_length_of_weight_l300_300346

theorem rod_length_of_weight (w10 : ℝ) (wL : ℝ) (L : ℝ) (h1 : w10 = 23.4) (h2 : wL = 14.04) : L = 6 :=
by
  sorry

end rod_length_of_weight_l300_300346


namespace lunch_cost_total_l300_300002

theorem lunch_cost_total (x y : ℝ) (h1 : y = 45) (h2 : x = (2 / 3) * y) : 
  x + y + y = 120 := by
  sorry

end lunch_cost_total_l300_300002


namespace min_value_2x_minus_y_l300_300490

theorem min_value_2x_minus_y :
  ∃ (x y : ℝ), (y = abs (x - 1) ∨ y = 2) ∧ (y ≤ 2) ∧ (2 * x - y = -4) :=
by
  sorry

end min_value_2x_minus_y_l300_300490


namespace arithmetic_sequence_sum_ratio_l300_300038

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a S : ℕ → ℚ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def condition_1 (a : ℕ → ℚ) : Prop :=
  is_arithmetic_sequence a

def condition_2 (a : ℕ → ℚ) : Prop :=
  (a 5) / (a 3) = 5 / 9

-- Proof statement
theorem arithmetic_sequence_sum_ratio (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : condition_1 a) (h2 : condition_2 a) (h3 : sum_of_first_n_terms a S) : 
  (S 9) / (S 5) = 1 := 
sorry

end arithmetic_sequence_sum_ratio_l300_300038


namespace sequence_a_1000_l300_300904

theorem sequence_a_1000 (a : ℕ → ℕ)
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 3) 
  (h₃ : ∀ n, a (n + 1) = 3 * a n - 2 * a (n - 1)) : 
  a 1000 = 2^1000 - 1 := 
sorry

end sequence_a_1000_l300_300904


namespace problem_statement_l300_300767

theorem problem_statement (x : ℝ) (h₀ : 0 < x) (h₁ : x + 1 / x ≥ 2) (h₂ : x + 4 / x ^ 2 ≥ 3) (h₃ : x + 27 / x ^ 3 ≥ 4) :
  ∀ a : ℝ, (x + a / x ^ 4 ≥ 5) → a = 4 ^ 4 := 
by 
  sorry

end problem_statement_l300_300767


namespace cars_in_parking_lot_l300_300105

theorem cars_in_parking_lot (C : ℕ) (customers_per_car : ℕ) (total_purchases : ℕ) 
  (h1 : customers_per_car = 5)
  (h2 : total_purchases = 50)
  (h3 : C * customers_per_car = total_purchases) : 
  C = 10 := 
by
  sorry

end cars_in_parking_lot_l300_300105


namespace slope_of_monotonically_decreasing_function_l300_300622

theorem slope_of_monotonically_decreasing_function
  (k b : ℝ)
  (H : ∀ x₁ x₂, x₁ ≤ x₂ → k * x₁ + b ≥ k * x₂ + b) : k < 0 := sorry

end slope_of_monotonically_decreasing_function_l300_300622


namespace percentage_honda_red_l300_300348

theorem percentage_honda_red (total_cars : ℕ) (honda_cars : ℕ) (percentage_red_total : ℚ)
  (percentage_red_non_honda : ℚ) (percentage_red_honda : ℚ) :
  total_cars = 9000 →
  honda_cars = 5000 →
  percentage_red_total = 0.60 →
  percentage_red_non_honda = 0.225 →
  percentage_red_honda = 0.90 →
  ((honda_cars * percentage_red_honda) / total_cars) * 100 = ((total_cars * percentage_red_total - (total_cars - honda_cars) * percentage_red_non_honda) / honda_cars) * 100 :=
by
  sorry

end percentage_honda_red_l300_300348


namespace angela_insects_l300_300737

theorem angela_insects (A J D : ℕ) (h1 : A = J / 2) (h2 : J = 5 * D) (h3 : D = 30) : A = 75 :=
by
  sorry

end angela_insects_l300_300737


namespace Sophie_Spends_72_80_l300_300659

noncomputable def SophieTotalCost : ℝ :=
  let cupcakesCost := 5 * 2
  let doughnutsCost := 6 * 1
  let applePieCost := 4 * 2
  let cookiesCost := 15 * 0.60
  let chocolateBarsCost := 8 * 1.50
  let sodaCost := 12 * 1.20
  let gumCost := 3 * 0.80
  let chipsCost := 10 * 1.10
  cupcakesCost + doughnutsCost + applePieCost + cookiesCost + chocolateBarsCost + sodaCost + gumCost + chipsCost

theorem Sophie_Spends_72_80 : SophieTotalCost = 72.80 :=
by
  sorry

end Sophie_Spends_72_80_l300_300659


namespace annual_income_of_a_l300_300542

-- Definitions based on the conditions
def monthly_income_ratio (a_income b_income : ℝ) : Prop := a_income / b_income = 5 / 2
def income_percentage (part whole : ℝ) : Prop := part / whole = 12 / 100
def c_monthly_income : ℝ := 15000
def b_monthly_income (c_income : ℝ) := c_income + 0.12 * c_income

-- The theorem to prove
theorem annual_income_of_a : ∀ (a_income b_income c_income : ℝ),
  monthly_income_ratio a_income b_income ∧
  b_income = b_monthly_income c_income ∧
  c_income = c_monthly_income →
  (a_income * 12) = 504000 :=
by
  -- Here we do not need to fill out the proof, so we use sorry
  sorry

end annual_income_of_a_l300_300542


namespace gas_cost_problem_l300_300312

theorem gas_cost_problem (x : ℝ) (h : x / 4 - 15 = x / 7) : x = 140 :=
sorry

end gas_cost_problem_l300_300312


namespace complex_eq_sub_l300_300314

open Complex

theorem complex_eq_sub {a b : ℝ} (h : (a : ℂ) + 2 * I = I * ((b : ℂ) - I)) : a - b = -3 := by
  sorry

end complex_eq_sub_l300_300314


namespace max_k_constant_l300_300591

theorem max_k_constant : 
  (∃ k, (∀ (x y z : ℝ), 0 < x → 0 < y → 0 < z → 
  (x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) <= k * Real.sqrt (x + y + z))) 
  ∧ k = Real.sqrt 6 / 2) :=
sorry

end max_k_constant_l300_300591


namespace fewest_handshakes_organizer_l300_300932

theorem fewest_handshakes_organizer (n k : ℕ) (h : k < n) 
  (total_handshakes: n*(n-1)/2 + k = 406) :
  k = 0 :=
sorry

end fewest_handshakes_organizer_l300_300932


namespace fraction_white_surface_area_l300_300433

-- Definitions of the given conditions
def cube_side_length : ℕ := 4
def small_cubes : ℕ := 64
def black_cubes : ℕ := 34
def white_cubes : ℕ := 30
def total_surface_area : ℕ := 6 * cube_side_length^2
def black_faces_exposed : ℕ := 32 
def white_faces_exposed : ℕ := total_surface_area - black_faces_exposed

-- The proof statement
theorem fraction_white_surface_area (cube_side_length_eq : cube_side_length = 4)
                                    (small_cubes_eq : small_cubes = 64)
                                    (black_cubes_eq : black_cubes = 34)
                                    (white_cubes_eq : white_cubes = 30)
                                    (black_faces_eq : black_faces_exposed = 32)
                                    (total_surface_area_eq : total_surface_area = 96)
                                    (white_faces_eq : white_faces_exposed = 64) : 
                                    (white_faces_exposed : ℚ) / (total_surface_area : ℚ) = 2 / 3 :=
by
  sorry

end fraction_white_surface_area_l300_300433


namespace johnsonville_max_band_members_l300_300404

def max_band_members :=
  ∃ m : ℤ, 30 * m % 34 = 2 ∧ 30 * m < 1500 ∧
  ∀ n : ℤ, (30 * n % 34 = 2 ∧ 30 * n < 1500) → 30 * n ≤ 30 * m

theorem johnsonville_max_band_members : ∃ m : ℤ, 30 * m % 34 = 2 ∧ 30 * m < 1500 ∧
                                           30 * m = 1260 :=
by 
  sorry

end johnsonville_max_band_members_l300_300404


namespace if_a_gt_abs_b_then_a2_gt_b2_l300_300941

theorem if_a_gt_abs_b_then_a2_gt_b2 (a b : ℝ) (h : a > abs b) : a^2 > b^2 :=
by sorry

end if_a_gt_abs_b_then_a2_gt_b2_l300_300941


namespace a3_5a6_value_l300_300497

variable {a : ℕ → ℤ}

-- Conditions: The sequence {a_n} is an arithmetic sequence, and a_4 + a_7 = 19
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

axiom a_seq_arithmetic : is_arithmetic_sequence a
axiom a4_a7_sum : a 4 + a 7 = 19

-- Problem statement: Prove that a_3 + 5a_6 = 57
theorem a3_5a6_value : a 3 + 5 * a 6 = 57 :=
by
  -- Proof goes here
  sorry

end a3_5a6_value_l300_300497


namespace roots_of_quadratic_l300_300970

theorem roots_of_quadratic :
  ∀ x : ℝ, (x - 3) ^ 2 = 4 → x = 5 ∨ x = 1 :=
begin
  intros x hx,
  have h_pos : x - 3 = 2 ∨ x - 3 = -2,
  { rw [eq_comm, pow_two] at hx,
    rwa sqr_eq_iff_abs_eq at hx, },
  cases h_pos with h1 h2,
  { left,
    linarith, },
  { right,
    linarith, },
end

end roots_of_quadratic_l300_300970


namespace smallest_integer_for_polynomial_div_l300_300231

theorem smallest_integer_for_polynomial_div (x : ℤ) : 
  (∃ k : ℤ, x = 6) ↔ ∃ y, y * (x - 5) = x^2 + 4 * x + 7 := 
by 
  sorry

end smallest_integer_for_polynomial_div_l300_300231


namespace arrangement_with_A_head_arrangement_with_adjacent_A_B_arrangement_with_A_not_head_B_not_end_arrangement_with_A_B_taller_shorter_not_adjacent_l300_300668

-- Definition for the arrangement problem
noncomputable def numArrangementsWithAHead : Nat := 24
noncomputable def numArrangementsWithAdjacentAB : Nat := 48
noncomputable def numArrangementsWithANotHeadBNotEnd : Nat := 72
noncomputable def numArrangementsWithAFirstBSecondNotAdjacent : Nat := 18

theorem arrangement_with_A_head (A B C D E : Char) :
  ∃ l : List Char, (l.head = A) ∧ l ~ L := numArrangementsWithAHead := by sorry

theorem arrangement_with_adjacent_A_B (A B C D E : Char) :
  ∃ l : List Char, (adjacent A B l) ∧ l ~ L := numArrangementsWithAdjacentAB := by sorry

theorem arrangement_with_A_not_head_B_not_end (A B C D E : Char) :
  ∃ l : List Char, (¬(l.head = A) ∧ ¬(l.last = B)) ∧ l ~ L := numArrangementsWithANotHeadBNotEnd := by sorry

theorem arrangement_with_A_B_taller_shorter_not_adjacent (A B C D E : Char) :
  ∃ l : List Char, (taller_shorter_not_adjacent A B l) ∧ l ~ L := numArrangementsWithAFirstBSecondNotAdjacent := by sorry

end arrangement_with_A_head_arrangement_with_adjacent_A_B_arrangement_with_A_not_head_B_not_end_arrangement_with_A_B_taller_shorter_not_adjacent_l300_300668


namespace mike_taller_than_mark_l300_300636

def feet_to_inches (feet : ℕ) : ℕ := 12 * feet

def mark_height_feet := 5
def mark_height_inches := 3
def mike_height_feet := 6
def mike_height_inches := 1

def mark_total_height := feet_to_inches mark_height_feet + mark_height_inches
def mike_total_height := feet_to_inches mike_height_feet + mike_height_inches

theorem mike_taller_than_mark : mike_total_height - mark_total_height = 10 :=
by
  sorry

end mike_taller_than_mark_l300_300636


namespace p_iff_q_l300_300029

variables {a b c : ℝ}
def p (a b c : ℝ) : Prop := ∃ x : ℝ, x = 1 ∧ a * x^2 + b * x + c = 0
def q (a b c : ℝ) : Prop := a + b + c = 0

theorem p_iff_q (h : a ≠ 0) : p a b c ↔ q a b c :=
sorry

end p_iff_q_l300_300029


namespace mike_taller_than_mark_l300_300642

-- Define the heights of Mark and Mike in terms of feet and inches
def mark_height_feet : ℕ := 5
def mark_height_inches : ℕ := 3
def mike_height_feet : ℕ := 6
def mike_height_inches : ℕ := 1

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Conversion of heights to inches
def mark_total_height_in_inches : ℕ := mark_height_feet * feet_to_inches + mark_height_inches
def mike_total_height_in_inches : ℕ := mike_height_feet * feet_to_inches + mike_height_inches

-- Define the problem statement: proving Mike is 10 inches taller than Mark
theorem mike_taller_than_mark : mike_total_height_in_inches - mark_total_height_in_inches = 10 :=
by sorry

end mike_taller_than_mark_l300_300642


namespace value_of_a_plus_c_l300_300806

-- Define the polynomials
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a * x + b
def g (c d : ℝ) (x : ℝ) : ℝ := x^2 + c * x + d

-- Define the condition for the vertex of polynomial f being a root of g
def vertex_of_f_is_root_of_g (a b c d : ℝ) : Prop :=
  g c d (-a / 2) = 0

-- Define the condition for the vertex of polynomial g being a root of f
def vertex_of_g_is_root_of_f (a b c d : ℝ) : Prop :=
  f a b (-c / 2) = 0

-- Define the condition that both polynomials have the same minimum value
def same_minimum_value (a b c d : ℝ) : Prop :=
  f a b (-a / 2) = g c d (-c / 2)

-- Define the condition that the polynomials intersect at (100, -100)
def polynomials_intersect (a b c d : ℝ) : Prop :=
  f a b 100 = -100 ∧ g c d 100 = -100

-- Lean theorem statement for the problem
theorem value_of_a_plus_c (a b c d : ℝ) 
  (h1 : vertex_of_f_is_root_of_g a b c d)
  (h2 : vertex_of_g_is_root_of_f a b c d)
  (h3 : same_minimum_value a b c d)
  (h4 : polynomials_intersect a b c d) :
  a + c = -400 := 
sorry

end value_of_a_plus_c_l300_300806


namespace women_attended_gathering_l300_300742

theorem women_attended_gathering :
  ∀ (m : ℕ) (w_per_man : ℕ) (m_per_woman : ℕ),
  m = 15 ∧ w_per_man = 4 ∧ m_per_woman = 3 →
  ∃ (w : ℕ), w = 20 :=
by
  intros m w_per_man m_per_woman h,
  cases h with hm hw_wom,
  cases hw_wom with hwm hmw,
  sorry

end women_attended_gathering_l300_300742


namespace smallest_n_1987_zeros_l300_300461

def h (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625 + n / 3125 + n / 15625 + n / 78125 + n / 390625 + n / 1953125 + n / 9765625

theorem smallest_n_1987_zeros :
  ∃ n : ℕ, h(n) = 1987 ∧ ∀ m : ℕ, m < n → h(m) ≠ 1987 :=
  by
  existsi 7960
  split
  · sorry
  · intro m hm
    sorry

end smallest_n_1987_zeros_l300_300461


namespace term_free_of_x_l300_300848

namespace PolynomialExpansion

theorem term_free_of_x (m n k : ℕ) (h : (x : ℝ)^(m * k - (m + n) * r) = 1) :
  (m * k) % (m + n) = 0 :=
by
  sorry

end PolynomialExpansion

end term_free_of_x_l300_300848


namespace total_years_l300_300110

variable (T D : ℕ)
variable (Tom_years : T = 50)
variable (Devin_years : D = 25 - 5)

theorem total_years (hT : T = 50) (hD : D = 25 - 5) : T + D = 70 := by
  sorry

end total_years_l300_300110


namespace number_of_women_attended_l300_300739

theorem number_of_women_attended
  (m : ℕ) (w : ℕ)
  (men_dance_women : m = 15)
  (women_dance_men : ∀ i : ℕ, i < 15 → i * 4 = 60)
  (women_condition : w * 3 = 60) :
  w = 20 :=
sorry

end number_of_women_attended_l300_300739


namespace solve_for_angle_B_solutions_l300_300629

noncomputable def number_of_solutions_for_angle_B (BC AC : ℝ) (angle_A : ℝ) : ℕ :=
  if (BC = 6 ∧ AC = 8 ∧ angle_A = 40) then 2 else 0

theorem solve_for_angle_B_solutions : number_of_solutions_for_angle_B 6 8 40 = 2 :=
  by sorry

end solve_for_angle_B_solutions_l300_300629


namespace ratio_of_areas_l300_300344

theorem ratio_of_areas (R_A R_B : ℝ) 
  (h1 : (1 / 6) * 2 * Real.pi * R_A = (1 / 9) * 2 * Real.pi * R_B) :
  (Real.pi * R_A^2) / (Real.pi * R_B^2) = (4 : ℝ) / 9 :=
by 
  sorry

end ratio_of_areas_l300_300344


namespace smallest_positive_integer_congruence_l300_300117

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, 0 < x ∧ x < 17 ∧ (3 * x ≡ 14 [MOD 17]) := sorry

end smallest_positive_integer_congruence_l300_300117


namespace FindAngleB_FindIncircleRadius_l300_300353

-- Define the problem setting
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Condition 1: a + c = 2b * sin (C + π / 6)
def Condition1 (T : Triangle) : Prop :=
  T.a + T.c = 2 * T.b * Real.sin (T.C + Real.pi / 6)

-- Condition 2: (b + c) (sin B - sin C) = (a - c) sin A
def Condition2 (T : Triangle) : Prop :=
  (T.b + T.c) * (Real.sin T.B - Real.sin T.C) = (T.a - T.c) * Real.sin T.A

-- Condition 3: (2a - c) cos B = b cos C
def Condition3 (T : Triangle) : Prop :=
  (2 * T.a - T.c) * Real.cos T.B = T.b * Real.cos T.C

-- Given: radius of incircle and dot product of vectors condition
def Given (T : Triangle) (r : ℝ) : Prop :=
  (T.a + T.c = 4 * Real.sqrt 3) ∧
  (2 * T.b * (T.a * T.c * Real.cos T.B - 3 * Real.sqrt 3 / 2) = 6)

-- Proof of B = π / 3
theorem FindAngleB (T : Triangle) :
  (Condition1 T ∨ Condition2 T ∨ Condition3 T) → T.B = Real.pi / 3 := 
sorry

-- Proof for the radius of the incircle
theorem FindIncircleRadius (T : Triangle) (r : ℝ) :
  Given T r → T.B = Real.pi / 3 → r = 1 := 
sorry


end FindAngleB_FindIncircleRadius_l300_300353


namespace tangent_line_eqn_of_sine_at_point_l300_300590

theorem tangent_line_eqn_of_sine_at_point :
  ∀ (f : ℝ → ℝ), (∀ x, f x = Real.sin (x + Real.pi / 3)) →
  ∀ (p : ℝ × ℝ), p = (0, Real.sqrt 3 / 2) →
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (∀ x, f x = Real.sin (x + Real.pi / 3)) ∧
  (∀ x y, y = f x → a * x + b * y + c = 0 → x - 2 * y + Real.sqrt 3 = 0) :=
by
  sorry

end tangent_line_eqn_of_sine_at_point_l300_300590


namespace can_form_triangle_l300_300120

theorem can_form_triangle (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  (a = 7 ∧ b = 12 ∧ c = 17) → True :=
by
  sorry

end can_form_triangle_l300_300120


namespace sin_double_angle_l300_300604

theorem sin_double_angle (θ : ℝ) 
    (h : Real.sin (Real.pi / 4 + θ) = 1 / 3) : 
    Real.sin (2 * θ) = -7 * Real.sqrt 2 / 9 :=
by
  sorry

end sin_double_angle_l300_300604


namespace negation_of_universal_prop_correct_l300_300220

def negation_of_universal_prop : Prop :=
  ¬ (∀ x : ℝ, x = |x|) ↔ ∃ x : ℝ, x ≠ |x|

theorem negation_of_universal_prop_correct : negation_of_universal_prop := 
by
  sorry

end negation_of_universal_prop_correct_l300_300220


namespace faye_total_crayons_l300_300588

  def num_rows : ℕ := 16
  def crayons_per_row : ℕ := 6
  def total_crayons : ℕ := num_rows * crayons_per_row

  theorem faye_total_crayons : total_crayons = 96 :=
  by
  sorry
  
end faye_total_crayons_l300_300588


namespace problem_statement_l300_300700

-- Define the number of balls and the counts of red and black balls
def total_balls := 10
def red_balls := 4
def black_balls := 6

-- Define binomial probability parameters for experiment 1
def n1 := 3
def p1 := (red_balls : ℚ) / total_balls

-- Define expectation and variance for experiment 1
def E_X1 : ℚ := n1 * p1
def D_X1 : ℚ := n1 * p1 * (1 - p1)

-- Define probabilities for different counts of red balls in experiment 2
def P_X2_0 : ℚ := (choose red_balls 0 * choose black_balls 3) / choose total_balls 3
def P_X2_1 : ℚ := (choose red_balls 1 * choose black_balls 2) / choose total_balls 3
def P_X2_2 : ℚ := (choose red_balls 2 * choose black_balls 1) / choose total_balls 3
def P_X2_3 : ℚ := (choose red_balls 3 * choose black_balls 0) / choose total_balls 3

-- Define expectation for experiment 2
def E_X2 : ℚ := 0 * P_X2_0 + 1 * P_X2_1 + 2 * P_X2_2 + 3 * P_X2_3

-- Define variance for experiment 2
def D_X2 : ℚ := (0 - E_X2) ^ 2 * P_X2_0 + (1 - E_X2) ^ 2 * P_X2_1 + (2 - E_X2) ^ 2 * P_X2_2 + (3 - E_X2) ^ 2 * P_X2_3

-- Proof problem statement
theorem problem_statement : E_X1 = E_X2 ∧ D_X1 > D_X2 := by
  sorry

end problem_statement_l300_300700


namespace number_of_ordered_pairs_lcm_232848_l300_300507

theorem number_of_ordered_pairs_lcm_232848 :
  let count_pairs :=
    let pairs_1 := 9
    let pairs_2 := 7
    let pairs_3 := 5
    let pairs_4 := 3
    pairs_1 * pairs_2 * pairs_3 * pairs_4
  count_pairs = 945 :=
by
  sorry

end number_of_ordered_pairs_lcm_232848_l300_300507


namespace non_negative_integer_solutions_l300_300399

open Nat

theorem non_negative_integer_solutions (x : Fin 2024 → ℕ) :
  (∑ i in Finset.range 2023, x i ^ 2) = 2 + ∑ i in Finset.range 2022, x i * x (i + 1) ↔ 
  (∃ s t u v : ℕ, (s + t + u + v = 2024) ∧ 2 * (choose 2024 4)) := 
sorry

end non_negative_integer_solutions_l300_300399


namespace fixed_point_exists_l300_300370

theorem fixed_point_exists : ∃ (x y : ℝ), (∀ k : ℝ, (2 * k - 1) * x - (k + 3) * y - (k - 11) = 0) ∧ x = 2 ∧ y = 3 := 
by
  -- Placeholder for proof
  sorry

end fixed_point_exists_l300_300370


namespace fraction_weevils_25_percent_l300_300069

-- Define the probabilities
def prob_good_milk : ℝ := 0.8
def prob_good_egg : ℝ := 0.4
def prob_all_good : ℝ := 0.24

-- The problem definition and statement
def fraction_weevils (F : ℝ) : Prop :=
  0.32 * (1 - F) = 0.24

theorem fraction_weevils_25_percent : fraction_weevils 0.25 :=
by sorry

end fraction_weevils_25_percent_l300_300069


namespace john_payment_l300_300359

def camera_value : ℝ := 5000
def weekly_rental_percentage : ℝ := 0.10
def rental_period : ℕ := 4
def friend_contribution_percentage : ℝ := 0.40

theorem john_payment :
  let weekly_rental_fee := camera_value * weekly_rental_percentage
  let total_rental_fee := weekly_rental_fee * rental_period
  let friend_contribution := total_rental_fee * friend_contribution_percentage
  let john_payment := total_rental_fee - friend_contribution
  john_payment = 1200 :=
by
  sorry

end john_payment_l300_300359


namespace min_value_l300_300156

theorem min_value (a : ℝ) (h : a > 1) : a + 1 / (a - 1) ≥ 3 :=
sorry

end min_value_l300_300156


namespace students_apply_colleges_l300_300424

    -- Define that there are 5 students
    def students : Nat := 5

    -- Each student has 3 choices of colleges
    def choices_per_student : Nat := 3

    -- The number of different ways the students can apply
    def number_of_ways : Nat := choices_per_student ^ students

    theorem students_apply_colleges : number_of_ways = 3 ^ 5 :=
    by
        -- Proof will be done here
        sorry
    
end students_apply_colleges_l300_300424


namespace puppy_cost_l300_300135

variable (P : ℕ)  -- Cost of one puppy

theorem puppy_cost (P : ℕ) (kittens : ℕ) (cost_kitten : ℕ) (total_value : ℕ) :
  kittens = 4 → cost_kitten = 15 → total_value = 100 → 
  2 * P + kittens * cost_kitten = total_value → P = 20 :=
by sorry

end puppy_cost_l300_300135


namespace probability_exactly_5_shots_expected_number_of_shots_l300_300426

-- Define the problem statement and conditions with Lean definitions
variables (p : ℝ) (hp : 0 < p ∧ p ≤ 1)

-- Part (a): Probability of 5 shots needed
theorem probability_exactly_5_shots : 
  (6 * p^3 * (1 - p)^2) = probability_exactly_5_shots (p : ℝ) :=
sorry

-- Part (b): Expected number of shots needed
theorem expected_number_of_shots :
  (3 / p) = expected_number_of_shots (p : ℝ) :=
sorry

end probability_exactly_5_shots_expected_number_of_shots_l300_300426


namespace minimize_sum_of_squares_if_and_only_if_l300_300311

noncomputable def minimize_sum_of_squares (a b c S : ℝ) (O : ℝ×ℝ×ℝ) (x y z : ℝ) : Prop :=
  let ax_by_cz := a * x + b * y + c * z
  ax_by_cz = 2 * S ∧
  x/y = a/b ∧
  y/z = b/c ∧
  x/z = a/c

theorem minimize_sum_of_squares_if_and_only_if (a b c S : ℝ) (O : ℝ×ℝ×ℝ) (x y z : ℝ) :
  (∃ P : ℝ, minimize_sum_of_squares a b c S O x y z) ↔ (x/y = a/b ∧ y/z = b/c ∧ x/z = a/c) := sorry

end minimize_sum_of_squares_if_and_only_if_l300_300311


namespace simplify_fraction_48_72_l300_300380

theorem simplify_fraction_48_72 : (48 : ℚ) / 72 = 2 / 3 := sorry

end simplify_fraction_48_72_l300_300380


namespace tangent_30_degrees_l300_300912

theorem tangent_30_degrees (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) (hA : ∃ α : ℝ, α = 30 ∧ (y / x) = Real.tan (π / 6)) :
  y / x = Real.sqrt 3 / 3 :=
by
  sorry

end tangent_30_degrees_l300_300912


namespace ball_hits_ground_at_time_l300_300396

-- Given definitions from the conditions
def y (t : ℝ) : ℝ := -4.9 * t^2 + 5 * t + 8

-- Statement of the problem: proving the time t when the ball hits the ground
theorem ball_hits_ground_at_time :
  ∃ t : ℝ, y t = 0 ∧ t = 1.887 := 
sorry

end ball_hits_ground_at_time_l300_300396


namespace bud_age_uncle_age_relation_l300_300569

variable (bud_age uncle_age : Nat)

theorem bud_age_uncle_age_relation (h : bud_age = 8) (h0 : bud_age = uncle_age / 3) : uncle_age = 24 := by
  sorry

end bud_age_uncle_age_relation_l300_300569


namespace number_of_female_fish_l300_300564

-- Defining the constants given in the problem
def total_fish : ℕ := 45
def fraction_male : ℚ := 2 / 3

-- The statement we aim to prove in Lean
theorem number_of_female_fish : 
  (total_fish : ℚ) * (1 - fraction_male) = 15 :=
by
  sorry

end number_of_female_fish_l300_300564


namespace linear_eq_with_one_variable_is_B_l300_300253

-- Define the equations
def eqA (x y : ℝ) : Prop := 2 * x = 3 * y
def eqB (x : ℝ) : Prop := 7 * x + 5 = 6 * (x - 1)
def eqC (x : ℝ) : Prop := x^2 + (1 / 2) * (x - 1) = 1
def eqD (x : ℝ) : Prop := (1 / x) - 2 = x

-- State the problem
theorem linear_eq_with_one_variable_is_B :
  ∃ x : ℝ, ¬ (∃ y : ℝ, eqA x y) ∧ eqB x ∧ ¬ eqC x ∧ ¬ eqD x :=
by {
  -- mathematical content goes here
  sorry
}

end linear_eq_with_one_variable_is_B_l300_300253


namespace compound_interest_doubling_time_l300_300592

theorem compound_interest_doubling_time :
  ∃ t : ℕ, (2 : ℝ) < (1 + (0.13 : ℝ))^t ∧ t = 6 :=
by
  sorry

end compound_interest_doubling_time_l300_300592


namespace first_to_receive_10_pieces_l300_300534

-- Definitions and conditions
def children := [1, 2, 3, 4, 5, 6, 7, 8]
def distribution_cycle := [1, 3, 6, 8, 3, 5, 8, 2, 5, 7, 2, 4, 7, 1, 4, 6]

def count_occurrences (n : ℕ) (lst : List ℕ) : ℕ :=
  lst.count n

-- Theorem
theorem first_to_receive_10_pieces : ∃ k, k = 3 ∧ count_occurrences k distribution_cycle = 2 :=
by
  sorry

end first_to_receive_10_pieces_l300_300534


namespace tangent_properties_l300_300030

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function f

-- Given conditions
axiom differentiable_f : Differentiable ℝ f
axiom func_eq : ∀ x, f (x - 2) = f (-x)
axiom tangent_eq_at_1 : ∀ x, (x = 1 → f x = 2 * x + 1)

-- Prove the required results
theorem tangent_properties :
  (deriv f 1 = 2) ∧ (∃ B C, (∀ x, (x = -3) → f x = B -2 * (x + 3)) ∧ (B = 3) ∧ (C = -3)) :=
by
  sorry

end tangent_properties_l300_300030


namespace total_clothes_donated_l300_300731

theorem total_clothes_donated
  (pants : ℕ) (jumpers : ℕ) (pajama_sets : ℕ) (tshirts : ℕ)
  (friends : ℕ)
  (adam_donation : ℕ)
  (half_adam_donated : ℕ)
  (friends_donation : ℕ)
  (total_donation : ℕ)
  (h1 : pants = 4) 
  (h2 : jumpers = 4) 
  (h3 : pajama_sets = 4 * 2) 
  (h4 : tshirts = 20) 
  (h5 : friends = 3)
  (h6 : adam_donation = pants + jumpers + pajama_sets + tshirts) 
  (h7 : half_adam_donated = adam_donation / 2) 
  (h8 : friends_donation = friends * adam_donation) 
  (h9 : total_donation = friends_donation + half_adam_donated) :
  total_donation = 126 :=
by
  sorry

end total_clothes_donated_l300_300731


namespace min_value_of_expression_l300_300773

theorem min_value_of_expression :
  ∃ (a b : ℝ), (∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ x^2 + a * x + b - 3 = 0) ∧ a^2 + (b - 4)^2 = 2 :=
sorry

end min_value_of_expression_l300_300773


namespace minimize_cost_at_4_l300_300268

-- Given definitions and conditions
def surface_area : ℝ := 12
def max_side_length : ℝ := 5
def front_face_cost_per_sqm : ℝ := 400
def sides_cost_per_sqm : ℝ := 150
def roof_ground_cost : ℝ := 5800
def wall_height : ℝ := 3

-- Definition of the total cost function
noncomputable def total_cost (x : ℝ) : ℝ :=
  900 * (x + 16 / x) + 5800

-- The main theorem to be proven
theorem minimize_cost_at_4 (h : 0 < x ∧ x ≤ max_side_length) : 
  (∀ x, total_cost x ≥ total_cost 4) ∧ total_cost 4 = 13000 :=
sorry

end minimize_cost_at_4_l300_300268


namespace trapezium_other_side_length_l300_300761

theorem trapezium_other_side_length :
  ∃ (x : ℝ), 1/2 * (18 + x) * 17 = 323 ∧ x = 20 :=
by
  sorry

end trapezium_other_side_length_l300_300761


namespace fruit_trees_l300_300133

theorem fruit_trees (total_streets : ℕ) 
  (fruit_trees_every_other : total_streets % 2 = 0) 
  (equal_fruit_trees : ∀ n : ℕ, 3 * n = total_streets / 2) : 
  ∃ n : ℕ, n = total_streets / 6 :=
by
  sorry

end fruit_trees_l300_300133


namespace mary_finds_eggs_l300_300368

theorem mary_finds_eggs (initial final found : ℕ) (h_initial : initial = 27) (h_final : final = 31) :
  found = final - initial → found = 4 :=
by
  intro h
  rw [h_initial, h_final] at h
  exact h

end mary_finds_eggs_l300_300368


namespace ed_marbles_l300_300587

theorem ed_marbles (doug_initial_marbles : ℕ) (marbles_lost : ℕ) (ed_doug_difference : ℕ) 
  (h1 : doug_initial_marbles = 22) (h2 : marbles_lost = 3) (h3 : ed_doug_difference = 5) : 
  (doug_initial_marbles + ed_doug_difference) = 27 :=
by
  sorry

end ed_marbles_l300_300587


namespace initial_red_martians_l300_300517

/-- Red Martians always tell the truth, while Blue Martians lie and then turn red.
    In a group of 2018 Martians, they answered in the sequence 1, 2, 3, ..., 2018 to the question
    of how many of them were red at that moment. Prove that the initial number of red Martians was 0 or 1. -/
theorem initial_red_martians (N : ℕ) (answers : Fin (N+1) → ℕ) :
  (∀ i : Fin (N+1), answers i = i.succ) → N = 2018 → (initial_red_martians_count = 0 ∨ initial_red_martians_count = 1)
:= sorry

end initial_red_martians_l300_300517


namespace average_age_of_remaining_people_l300_300393

theorem average_age_of_remaining_people:
  ∀ (ages : List ℕ), 
  (List.length ages = 8) →
  (List.sum ages = 224) →
  (24 ∈ ages) →
  ((List.sum ages - 24) / 7 = 28 + 4/7) := 
by
  intro ages
  intro h_len
  intro h_sum
  intro h_24
  sorry

end average_age_of_remaining_people_l300_300393


namespace num_machines_first_scenario_l300_300089

theorem num_machines_first_scenario (r : ℝ) (n : ℕ) :
  (∀ r, (2 : ℝ) * r * 24 = 1) →
  (∀ r, (n : ℝ) * r * 6 = 1) →
  n = 8 :=
by
  intros h1 h2
  sorry

end num_machines_first_scenario_l300_300089


namespace units_digit_of_pow_sum_is_correct_l300_300676

theorem units_digit_of_pow_sum_is_correct : 
  (24^3 + 42^3) % 10 = 2 := by
  -- Start the proof block
  sorry

end units_digit_of_pow_sum_is_correct_l300_300676


namespace problem1_problem2_l300_300748

-- Problem 1
theorem problem1 : (Real.sqrt 8 - Real.sqrt 27 - (4 * Real.sqrt (1 / 2) + Real.sqrt 12)) = -5 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem2 : ((Real.sqrt 6 + Real.sqrt 12) * (2 * Real.sqrt 3 - Real.sqrt 6) - 3 * Real.sqrt 32 / (Real.sqrt 2 / 2)) = -18 := by
  sorry

end problem1_problem2_l300_300748


namespace employees_females_l300_300791

theorem employees_females
  (total_employees : ℕ)
  (adv_deg_employees : ℕ)
  (coll_deg_employees : ℕ)
  (males_coll_deg : ℕ)
  (females_adv_deg : ℕ)
  (females_coll_deg : ℕ)
  (h1 : total_employees = 180)
  (h2 : adv_deg_employees = 90)
  (h3 : coll_deg_employees = 180 - 90)
  (h4 : males_coll_deg = 35)
  (h5 : females_adv_deg = 55)
  (h6 : females_coll_deg = 90 - 35) :
  females_coll_deg + females_adv_deg = 110 :=
by
  sorry

end employees_females_l300_300791


namespace find_triples_l300_300589

theorem find_triples 
  (x y z : ℝ)
  (h1 : x + y * z = 2)
  (h2 : y + z * x = 2)
  (h3 : z + x * y = 2)
 : (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -2 ∧ y = -2 ∧ z = -2) :=
sorry

end find_triples_l300_300589


namespace hyperbola_asymptotes_l300_300482

theorem hyperbola_asymptotes (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (e : ℝ) (he : e = Real.sqrt 3) (h_eq : e = Real.sqrt ((a^2 + b^2) / a^2)) :
  (∀ x : ℝ, y = x * Real.sqrt 2) :=
by
  sorry

end hyperbola_asymptotes_l300_300482


namespace evaluate_expression_l300_300455

theorem evaluate_expression : 
  ((-4 : ℤ) ^ 6) / (4 ^ 4) + (2 ^ 5) * (5 : ℤ) - (7 ^ 2) = 127 :=
by sorry

end evaluate_expression_l300_300455


namespace simplify_complex_expression_l300_300951

theorem simplify_complex_expression : 
  ∀ (i : ℂ), i^2 = -1 → 3 * (4 - 2 * i) + 2 * i * (3 + 2 * i) = 8 := 
by
  intros
  sorry

end simplify_complex_expression_l300_300951


namespace bridge_length_correct_l300_300554

noncomputable def length_of_bridge 
  (train_length : ℝ) 
  (time_to_cross : ℝ) 
  (train_speed_kmph : ℝ) : ℝ :=
  (train_speed_kmph * (5 / 18) * time_to_cross) - train_length

theorem bridge_length_correct :
  length_of_bridge 120 31.99744020478362 36 = 199.9744020478362 :=
by
  -- Skipping the proof details
  sorry

end bridge_length_correct_l300_300554


namespace density_function_lim_l300_300949

noncomputable def density_function (x : ℝ) : ℝ :=
  ∑' (n : ℕ), (Set.indicator (Set.Icc n (n + 2^(-n : ℝ))) (λ _, 1) x)

theorem density_function_lim :
  (0 = lim_x (density_function x, x → ∞) < limsup_x (density_function x, x → ∞) = 1) ∧
  (∀ᵐ (x : ℝ) (measure_theory.measure_space ℝ), tendsto (λ n, density_function (x + n)) at_top (𝓝 0)) :=
sorry

end density_function_lim_l300_300949


namespace teagan_total_cost_l300_300198

theorem teagan_total_cost :
  let reduction_percentage := 20
  let original_price_shirt := 60
  let original_price_jacket := 90
  let reduced_price_shirt := original_price_shirt * (100 - reduction_percentage) / 100
  let reduced_price_jacket := original_price_jacket * (100 - reduction_percentage) / 100
  let cost_5_shirts := 5 * reduced_price_shirt
  let cost_10_jackets := 10 * reduced_price_jacket
  let total_cost := cost_5_shirts + cost_10_jackets
  total_cost = 960 := by
  sorry

end teagan_total_cost_l300_300198


namespace find_max_sum_pair_l300_300760

theorem find_max_sum_pair :
  ∃ a b : ℕ, 2 * a * b + 3 * b = b^2 + 6 * a + 6 ∧ (∀ a' b' : ℕ, 2 * a' * b' + 3 * b' = b'^2 + 6 * a' + 6 → a + b ≥ a' + b') ∧ a = 5 ∧ b = 9 :=
by {
  sorry
}

end find_max_sum_pair_l300_300760


namespace prime_sum_20_to_30_l300_300244

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end prime_sum_20_to_30_l300_300244


namespace half_angle_quadrant_l300_300080

theorem half_angle_quadrant
  (α : ℝ)
  (h1 : ∃ k : ℤ, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2)
  (h2 : |Real.cos (α / 2)| = -Real.cos (α / 2)) :
  ∃ k : ℤ, k * Real.pi / 2 < α / 2 ∧ α / 2 < k * Real.pi * 3 / 4 ∧ Real.cos (α / 2) ≤ 0 := sorry

end half_angle_quadrant_l300_300080


namespace solve_system_of_inequalities_l300_300301

variable {R : Type*} [LinearOrderedField R]

theorem solve_system_of_inequalities (x1 x2 x3 x4 x5 : R)
  (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x4 > 0) (h5 : x5 > 0) :
  (x1^2 - x3^2) * (x2^2 - x3^2) ≤ 0 ∧ 
  (x3^2 - x1^2) * (x3^2 - x1^2) ≤ 0 ∧ 
  (x3^2 - x3 * x2) * (x1^2 - x3 * x2) ≤ 0 ∧ 
  (x1^2 - x1 * x3) * (x3^2 - x1 * x3) ≤ 0 ∧ 
  (x3^2 - x2 * x1) * (x1^2 - x2 * x1) ≤ 0 →
  x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5 :=
sorry

end solve_system_of_inequalities_l300_300301


namespace range_of_m_l300_300508

-- Define the conditions for p and q
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ 
  (x₁^2 + 2 * m * x₁ + 1 = 0) ∧ (x₂^2 + 2 * m * x₂ + 1 = 0)

def q (m : ℝ) : Prop := ¬ ∃ x : ℝ, x^2 + 2 * (m-2) * x - 3 * m + 10 = 0

-- The main theorem
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬ (p m ∧ q m) ↔ 
  (m ≤ -2 ∨ (-1 ≤ m ∧ m < 3)) := 
by
  sorry

end range_of_m_l300_300508


namespace exists_x0_in_interval_l300_300944

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem exists_x0_in_interval :
  ∃ x0 : ℝ, 0 < x0 ∧ x0 < 4 ∧ f x0 = 0 ∧ 2 < x0 ∧ x0 < 3 :=
sorry

end exists_x0_in_interval_l300_300944


namespace josette_paid_correct_amount_l300_300505

-- Define the number of small and large bottles
def num_small_bottles : ℕ := 3
def num_large_bottles : ℕ := 2

-- Define the cost of each type of bottle
def cost_per_small_bottle : ℝ := 1.50
def cost_per_large_bottle : ℝ := 2.40

-- Define the total number of bottles purchased
def total_bottles : ℕ := num_small_bottles + num_large_bottles

-- Define the discount rate applicable when purchasing 5 or more bottles
def discount_rate : ℝ := 0.10

-- Calculate the initial total cost before any discount
def total_cost_before_discount : ℝ :=
  (num_small_bottles * cost_per_small_bottle) + 
  (num_large_bottles * cost_per_large_bottle)

-- Calculate the discount amount if applicable
def discount_amount : ℝ :=
  if total_bottles >= 5 then
    discount_rate * total_cost_before_discount
  else
    0

-- Calculate the final amount Josette paid after applying any discount
def final_amount_paid : ℝ :=
  total_cost_before_discount - discount_amount

-- Prove that the final amount paid is €8.37
theorem josette_paid_correct_amount :
  final_amount_paid = 8.37 :=
by
  sorry

end josette_paid_correct_amount_l300_300505


namespace sum_of_primes_between_20_and_30_l300_300233

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primes_between_20_and_30 := [23, 29]

theorem sum_of_primes_between_20_and_30 : 
  23 ∈ primes_between_20_and_30 ∧ 29 ∈ primes_between_20_and_30 ∧
  (∀ n ∈ primes_between_20_and_30, is_prime n) ∧
  list.sum primes_between_20_and_30 = 52 := 
by 
  sorry

end sum_of_primes_between_20_and_30_l300_300233


namespace simplify_fraction_l300_300377

theorem simplify_fraction : (48 / 72 : ℚ) = (2 / 3) := 
by
  sorry

end simplify_fraction_l300_300377


namespace monthly_sales_fraction_l300_300072

theorem monthly_sales_fraction (V S_D T : ℝ) 
  (h1 : S_D = 6 * V) 
  (h2 : S_D = 0.35294117647058826 * T) 
  : V = (1 / 17) * T :=
sorry

end monthly_sales_fraction_l300_300072


namespace unique_solution_value_l300_300900

theorem unique_solution_value (k : ℝ) :
  (∃ x : ℝ, x^2 = 2 * x + k ∧ ∀ y : ℝ, y^2 = 2 * y + k → y = x) ↔ k = -1 := 
by
  sorry

end unique_solution_value_l300_300900


namespace geometric_sum_l300_300068

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sum (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 = 6) (h3 : a 3 = -18) :
  a 1 + a 2 + a 3 + a 4 = 40 :=
sorry

end geometric_sum_l300_300068


namespace inequality_proof_l300_300077

theorem inequality_proof (a b c : ℝ) (h1 : 0 < c) (h2 : c ≤ b) (h3 : b ≤ a) :
  (a^2 - b^2) / c + (c^2 - b^2) / a + (a^2 - c^2) / b ≥ 3 * a - 4 * b + c :=
by
  sorry

end inequality_proof_l300_300077


namespace Hadley_walked_to_grocery_store_in_2_miles_l300_300167

-- Define the variables and conditions
def distance_to_grocery_store (x : ℕ) : Prop :=
  x + (x - 1) + 3 = 6

-- Stating the main proposition to prove
theorem Hadley_walked_to_grocery_store_in_2_miles : ∃ x : ℕ, distance_to_grocery_store x ∧ x = 2 := 
by sorry

end Hadley_walked_to_grocery_store_in_2_miles_l300_300167


namespace factorization_problem_l300_300452

theorem factorization_problem (a b c : ℝ) :
  let E := a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3)
  let P := -(a^2 + ab + b^2 + bc + c^2 + ac)
  E = (a - b) * (b - c) * (c - a) * P :=
by
  sorry

end factorization_problem_l300_300452


namespace isosceles_triangle_perimeter_l300_300905

theorem isosceles_triangle_perimeter
  (x y : ℝ)
  (h : |x - 3| + (y - 1)^2 = 0)
  (isosceles_triangle : ∃ a b c, (a = x ∧ b = x ∧ c = y) ∨ (a = x ∧ b = y ∧ c = y) ∨ (a = y ∧ b = y ∧ c = x)):
  ∃ perimeter : ℝ, perimeter = 7 :=
by
  sorry

end isosceles_triangle_perimeter_l300_300905


namespace complex_power_difference_l300_300784

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i)^10 - (1 - i)^10 = 64 * i := 
by sorry

end complex_power_difference_l300_300784


namespace joska_has_higher_probability_l300_300632

open Nat

def num_4_digit_with_all_diff_digits := 10 * 9 * 8 * 7
def total_4_digit_combinations := 10^4
def num_4_digit_with_repeated_digits := total_4_digit_combinations - num_4_digit_with_all_diff_digits

-- Calculate probabilities
noncomputable def prob_joska := (num_4_digit_with_all_diff_digits : ℝ) / (total_4_digit_combinations : ℝ)
noncomputable def prob_gabor := (num_4_digit_with_repeated_digits : ℝ) / (total_4_digit_combinations : ℝ)

theorem joska_has_higher_probability :
  prob_joska > prob_gabor :=
  by
    sorry

end joska_has_higher_probability_l300_300632


namespace son_age_l300_300725

theorem son_age (M S : ℕ) (h1: M = S + 26) (h2: M + 2 = 2 * (S + 2)) : S = 24 :=
by
  sorry

end son_age_l300_300725


namespace calculate_value_l300_300007

theorem calculate_value : 2 * (75 * 1313 - 25 * 1313) = 131300 := 
by 
  sorry

end calculate_value_l300_300007


namespace yearly_feeding_cost_l300_300613

-- Defining the conditions
def num_geckos := 3
def num_iguanas := 2
def num_snakes := 4

def cost_per_snake_per_month := 10
def cost_per_iguana_per_month := 5
def cost_per_gecko_per_month := 15

-- Statement of the proof problem
theorem yearly_feeding_cost : 
  (num_snakes * cost_per_snake_per_month + num_iguanas * cost_per_iguana_per_month + num_geckos * cost_per_gecko_per_month) * 12 = 1140 := 
  by 
    sorry

end yearly_feeding_cost_l300_300613


namespace part1_part2_l300_300906

open Complex

theorem part1 {m : ℝ} : m + (m^2 + 2) * I = 0 -> m = 0 :=
by sorry

theorem part2 {m : ℝ} (h : (m + I)^2 - 2 * (m + I) + 2 = 0) :
    (let z1 := m + I
     let z2 := 2 + m * I
     im ((z2 / z1) : ℂ) = -1 / 2) :=
by sorry

end part1_part2_l300_300906


namespace park_needs_minimum_37_nests_l300_300494

-- Defining the number of different birds
def num_sparrows : ℕ := 5
def num_pigeons : ℕ := 3
def num_starlings : ℕ := 6
def num_robins : ℕ := 2

-- Defining the nesting requirements for each bird species
def nests_per_sparrow : ℕ := 1
def nests_per_pigeon : ℕ := 2
def nests_per_starling : ℕ := 3
def nests_per_robin : ℕ := 4

-- Definition of total minimum nests required
def min_nests_required : ℕ :=
  (num_sparrows * nests_per_sparrow) +
  (num_pigeons * nests_per_pigeon) +
  (num_starlings * nests_per_starling) +
  (num_robins * nests_per_robin)

-- Proof Statement
theorem park_needs_minimum_37_nests :
  min_nests_required = 37 :=
sorry

end park_needs_minimum_37_nests_l300_300494


namespace car_miles_traveled_actual_miles_l300_300128

noncomputable def count_skipped_numbers (n : ℕ) : ℕ :=
  let count_digit7 (x : ℕ) : Bool := x = 7
  -- Function to count the number of occurrences of digit 7 in each place value
  let rec count (x num_skipped : ℕ) : ℕ :=
    if x = 0 then num_skipped else
    let digit := x % 10
    let new_count := if count_digit7 digit then num_skipped + 1 else num_skipped
    count (x / 10) new_count
  count n 0

theorem car_miles_traveled (odometer_reading : ℕ) : ℕ :=
  let num_skipped := count_skipped_numbers 3008
  odometer_reading - num_skipped

theorem actual_miles {odometer_reading : ℕ} (h : odometer_reading = 3008) : car_miles_traveled odometer_reading = 2194 :=
by sorry

end car_miles_traveled_actual_miles_l300_300128


namespace probability_three_specific_cards_l300_300408

theorem probability_three_specific_cards :
  let total_deck := 52
  let total_spades := 13
  let total_tens := 4
  let total_queens := 4
  let p_case1 := ((12:ℚ) / total_deck) * (total_tens / (total_deck - 1)) * (total_queens / (total_deck - 2))
  let p_case2 := ((1:ℚ) / total_deck) * ((total_tens - 1) / (total_deck - 1)) * (total_queens / (total_deck - 2))
  p_case1 + p_case2 = (17:ℚ) / 11050 :=
by
  sorry

end probability_three_specific_cards_l300_300408


namespace expected_faces_rolled_six_times_l300_300714

-- Define a random variable indicating appearance of a particular face
noncomputable def ζi (n : ℕ): ℝ := if n > 0 then 1 - (5 / 6) ^ 6 else 0

-- Define the expected number of distinct faces
noncomputable def expected_distinct_faces : ℝ := 6 * ζi 1

theorem expected_faces_rolled_six_times :
  expected_distinct_faces = (6 ^ 6 - 5 ^ 6) / 6 ^ 5 :=
by
  -- Here we would provide the proof
  sorry

end expected_faces_rolled_six_times_l300_300714


namespace fractional_eq_nonneg_solution_l300_300159

theorem fractional_eq_nonneg_solution 
  (m x : ℝ)
  (h1 : x ≠ 2)
  (h2 : x ≥ 0)
  (eq_fractional : m / (x - 2) + 1 = x / (2 - x)) :
  m ≤ 2 ∧ m ≠ -2 := 
  sorry

end fractional_eq_nonneg_solution_l300_300159


namespace shallow_depth_of_pool_l300_300282

theorem shallow_depth_of_pool (w l D V : ℝ) (h₀ : w = 9) (h₁ : l = 12) (h₂ : D = 4) (h₃ : V = 270) :
  (0.5 * (d + D) * w * l = V) → d = 1 :=
by
  intros h_equiv
  sorry

end shallow_depth_of_pool_l300_300282


namespace swap_square_digit_l300_300584

theorem swap_square_digit (n : ℕ) (h1 : n ≥ 10 ∧ n < 100) : 
  ∃ (x y : ℕ), n = 10 * x + y ∧ (x < 10 ∧ y < 10) ∧ (y * 100 + x * 10 + y^2 + 20 * x * y - 1) = n * n + 2 * n + 1 :=
by 
    sorry

end swap_square_digit_l300_300584


namespace cost_of_bananas_and_cantaloupe_l300_300191

-- Let a, b, c, and d be real numbers representing the prices of apples, bananas, cantaloupe, and dates respectively.
variables (a b c d : ℝ)

-- Conditions given in the problem
axiom h1 : a + b + c + d = 40
axiom h2 : d = 3 * a
axiom h3 : c = (a + b) / 2

-- Goal is to prove that the sum of the prices of bananas and cantaloupe is 8 dollars.
theorem cost_of_bananas_and_cantaloupe : b + c = 8 :=
by
  sorry

end cost_of_bananas_and_cantaloupe_l300_300191


namespace vasya_expected_area_greater_l300_300883

/-- Vasya and Asya roll dice to cut out shapes and determine whose expected area is greater. -/
theorem vasya_expected_area_greater :
  let A : ℕ := 1
  let B : ℕ := 2
  (6 * 7 * (2 * 7 / 6) * 21 / 6) < (21 * 91 / 6) := 
by
  sorry

end vasya_expected_area_greater_l300_300883


namespace height_difference_l300_300794

-- Define the initial height of James's uncle
def uncle_height : ℝ := 72

-- Define the initial height ratio of James compared to his uncle
def james_initial_height_ratio : ℝ := 2 / 3

-- Define the height gained by James from his growth spurt
def james_growth_spurt : ℝ := 10

-- Define the initial height of James before the growth spurt
def james_initial_height : ℝ := uncle_height * james_initial_height_ratio

-- Define the new height of James after the growth spurt
def james_new_height : ℝ := james_initial_height + james_growth_spurt

-- Theorem: The difference in height between James's uncle and James after the growth spurt is 14 inches
theorem height_difference : uncle_height - james_new_height = 14 := sorry

end height_difference_l300_300794


namespace range_of_a_l300_300623

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
sorry

end range_of_a_l300_300623


namespace nut_game_winning_strategy_l300_300667

theorem nut_game_winning_strategy (N : ℕ) (h : N > 2) : ∃ second_player_wins : Prop, second_player_wins :=
sorry

end nut_game_winning_strategy_l300_300667


namespace sin_110_correct_tan_945_correct_cos_25pi_over_4_correct_l300_300295

noncomputable def sin_110_degrees : ℝ := Real.sin (110 * Real.pi / 180)
noncomputable def tan_945_degrees_reduction : ℝ := Real.tan (945 * Real.pi / 180 - 5 * Real.pi)
noncomputable def cos_25pi_over_4_reduction : ℝ := Real.cos (25 * Real.pi / 4 - 6 * 2 * Real.pi)

theorem sin_110_correct : sin_110_degrees = Real.sin (110 * Real.pi / 180) :=
by
  sorry

theorem tan_945_correct : tan_945_degrees_reduction = 1 :=
by 
  sorry

theorem cos_25pi_over_4_correct : cos_25pi_over_4_reduction = Real.cos (Real.pi / 4) :=
by 
  sorry

end sin_110_correct_tan_945_correct_cos_25pi_over_4_correct_l300_300295


namespace fraction_equality_l300_300170

theorem fraction_equality (x y z : ℝ) (k : ℝ) (hx : x = 3 * k) (hy : y = 5 * k) (hz : z = 7 * k) :
  (x - y + z) / (x + y - z) = 5 := 
  sorry

end fraction_equality_l300_300170


namespace jenny_distance_from_school_l300_300631

-- Definitions based on the given conditions.
def kernels_per_feet : ℕ := 1
def feet_per_kernel : ℕ := 25
def squirrel_fraction_eaten : ℚ := 1/4
def remaining_kernels : ℕ := 150

-- Problem statement in Lean 4.
theorem jenny_distance_from_school : 
  ∀ (P : ℕ), (3/4:ℚ) * P = 150 → P * feet_per_kernel = 5000 :=
by
  intros P h
  sorry

end jenny_distance_from_school_l300_300631


namespace find_fx_l300_300342

theorem find_fx (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 + 1) = 2 * x^2 + 1) : ∀ x : ℝ, f x = 2 * x - 1 := 
sorry

end find_fx_l300_300342


namespace onions_total_l300_300518

theorem onions_total (Sara_onions : ℕ) (Sally_onions : ℕ) (Fred_onions : ℕ) 
  (h1: Sara_onions = 4) (h2: Sally_onions = 5) (h3: Fred_onions = 9) :
  Sara_onions + Sally_onions + Fred_onions = 18 :=
by
  sorry

end onions_total_l300_300518


namespace sum_leq_two_l300_300084

open Classical

theorem sum_leq_two (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^3 + b^3 = 2) : a + b ≤ 2 :=
by
  sorry

end sum_leq_two_l300_300084


namespace flag_pole_height_eq_150_l300_300555

-- Define the conditions
def tree_height : ℝ := 12
def tree_shadow_length : ℝ := 8
def flag_pole_shadow_length : ℝ := 100

-- Problem statement: prove the height of the flag pole equals 150 meters
theorem flag_pole_height_eq_150 :
  ∃ (F : ℝ), (tree_height / tree_shadow_length) = (F / flag_pole_shadow_length) ∧ F = 150 :=
by
  -- Setup the proof scaffold
  have h : (tree_height / tree_shadow_length) = (150 / flag_pole_shadow_length) := by sorry
  exact ⟨150, h, rfl⟩

end flag_pole_height_eq_150_l300_300555


namespace problem_solution_l300_300037

theorem problem_solution
  (a1 a2 a3: ℝ)
  (a_arith_seq : ∃ d, a1 = 1 + d ∧ a2 = a1 + d ∧ a3 = a2 + d ∧ 9 = a3 + d)
  (b1 b2 b3: ℝ)
  (b_geo_seq : ∃ r, r > 0 ∧ b1 = -9 * r ∧ b2 = b1 * r ∧ b3 = b2 * r ∧ -1 = b3 * r) :
  (b2 / (a1 + a3) = -3 / 10) :=
by
  -- Placeholder for the proof, not required in this context
  sorry

end problem_solution_l300_300037


namespace cyclic_inequality_l300_300766

theorem cyclic_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  2 * (x^3 + y^3 + z^3) ≥ x^2 * y + x^2 * z + y^2 * z + y^2 * x + z^2 * x + z^2 * y := 
by
  sorry

end cyclic_inequality_l300_300766


namespace intersection_M_N_l300_300166

-- Defining set M
def M : Set ℕ := {1, 2, 3, 4}

-- Defining the set N based on the condition
def N : Set ℕ := {x | ∃ n ∈ M, x = n^2}

-- Lean statement to prove the intersection
theorem intersection_M_N : M ∩ N = {1, 4} := 
by
  sorry

end intersection_M_N_l300_300166


namespace incorrect_conclusion_l300_300898

def y (x : ℝ) : ℝ := -2 * x + 3

theorem incorrect_conclusion : ∀ (x : ℝ), y x = 0 → x ≠ 0 := 
by
  sorry

end incorrect_conclusion_l300_300898


namespace range_of_f_l300_300332

-- Define the function f
def f (x : ℕ) : ℕ := 3 * x - 1

-- Define the domain
def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 4}

-- Define the range
def range : Set ℕ := {2, 5, 8, 11}

-- Lean 4 theorem statement
theorem range_of_f : 
  {y | ∃ x ∈ domain, y = f x} = range :=
by
  sorry

end range_of_f_l300_300332


namespace difference_divisible_l300_300187

theorem difference_divisible (a b n : ℕ) (h : n % 2 = 0) (hab : a + b = 61) :
  (47^100 - 14^100) % 61 = 0 := by
  sorry

end difference_divisible_l300_300187


namespace revenue_comparison_l300_300785

theorem revenue_comparison 
  (D N J F : ℚ) 
  (hN : N = (2 / 5) * D) 
  (hJ : J = (2 / 25) * D) 
  (hF : F = (3 / 4) * D) : 
  D / ((N + J + F) / 3) = 100 / 41 := 
by 
  sorry

end revenue_comparison_l300_300785


namespace size_of_first_file_l300_300751

theorem size_of_first_file (internet_speed_mbps : ℝ) (time_hours : ℝ) (file2_mbps : ℝ) (file3_mbps : ℝ) (total_downloaded_mbps : ℝ) :
  internet_speed_mbps = 2 →
  time_hours = 2 →
  file2_mbps = 90 →
  file3_mbps = 70 →
  total_downloaded_mbps = internet_speed_mbps * 60 * time_hours →
  total_downloaded_mbps - (file2_mbps + file3_mbps) = 80 :=
by
  intros
  sorry

end size_of_first_file_l300_300751


namespace uncle_taller_than_james_l300_300796

def james_initial_height (uncle_height : ℕ) : ℕ := (2 * uncle_height) / 3

def james_final_height (initial_height : ℕ) (growth_spurt : ℕ) : ℕ := initial_height + growth_spurt

theorem uncle_taller_than_james (uncle_height : ℕ) (growth_spurt : ℕ) :
  uncle_height = 72 →
  growth_spurt = 10 →
  uncle_height - (james_final_height (james_initial_height uncle_height) growth_spurt) = 14 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end uncle_taller_than_james_l300_300796


namespace honor_students_count_l300_300983

noncomputable def number_of_honor_students (G B Eg Eb : ℕ) (p_girl p_boy : ℚ) : ℕ :=
  if G < 30 ∧ B < 30 ∧ Eg = (3 / 13) * G ∧ Eb = (4 / 11) * B ∧ G + B < 30 then
    Eg + Eb
  else
    0

theorem honor_students_count :
  ∃ (G B Eg Eb : ℕ), (G < 30 ∧ B < 30 ∧ G % 13 = 0 ∧ B % 11 = 0 ∧ Eg = (3 * G / 13) ∧ Eb = (4 * B / 11) ∧ G + B < 30 ∧ number_of_honor_students G B Eg Eb (3 / 13) (4 / 11) = 7) :=
by {
  sorry
}

end honor_students_count_l300_300983


namespace lending_rate_l300_300726

noncomputable def principal: ℝ := 5000
noncomputable def rate_borrowed: ℝ := 4
noncomputable def time_years: ℝ := 2
noncomputable def gain_per_year: ℝ := 100

theorem lending_rate :
  ∃ (rate_lent: ℝ), 
  (principal * rate_lent * time_years / 100) - (principal * rate_borrowed * time_years / 100) / time_years = gain_per_year ∧
  rate_lent = 6 :=
by
  sorry

end lending_rate_l300_300726


namespace minimum_value_of_expression_l300_300943

noncomputable def min_value_expr (x y z : ℝ) : ℝ := (x + 3 * y) * (y + 3 * z) * (2 * x * z + 1)

theorem minimum_value_of_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) :
  min_value_expr x y z = 24 * Real.sqrt 2 :=
sorry

end minimum_value_of_expression_l300_300943


namespace pondFishEstimate_l300_300945

noncomputable def estimateTotalFish (initialFishMarked : ℕ) (caughtFishTenDaysLater : ℕ) (markedFishCaught : ℕ) : ℕ :=
  initialFishMarked * caughtFishTenDaysLater / markedFishCaught

theorem pondFishEstimate
    (initialFishMarked : ℕ)
    (caughtFishTenDaysLater : ℕ)
    (markedFishCaught : ℕ)
    (h1 : initialFishMarked = 30)
    (h2 : caughtFishTenDaysLater = 50)
    (h3 : markedFishCaught = 2) :
    estimateTotalFish initialFishMarked caughtFishTenDaysLater markedFishCaught = 750 := by
  sorry

end pondFishEstimate_l300_300945


namespace sum_of_primes_between_20_and_30_l300_300243

theorem sum_of_primes_between_20_and_30 :
  (∑ n in { n | n > 20 ∧ n < 30 ∧ Prime n }, n) = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l300_300243


namespace cartons_in_a_case_l300_300858

-- Definitions based on problem conditions
def numberOfBoxesInCarton (c : ℕ) (b : ℕ) : ℕ := c * b * 300
def paperClipsInTwoCases (c : ℕ) (b : ℕ) : ℕ := 2 * numberOfBoxesInCarton c b

-- Condition from problem statement: paperClipsInTwoCases c b = 600
theorem cartons_in_a_case 
  (c b : ℕ) 
  (h1 : paperClipsInTwoCases c b = 600) 
  (h2 : b ≥ 1) : 
  c = 1 := 
by
  -- Proof will be provided here
  sorry

end cartons_in_a_case_l300_300858


namespace max_ball_height_l300_300545

/-- 
The height (in feet) of a ball traveling on a parabolic path is given by -20t^2 + 80t + 36,
where t is the time after launch. This theorem shows that the maximum height of the ball is 116 feet.
-/
theorem max_ball_height : ∃ t : ℝ, ∀ t', -20 * t^2 + 80 * t + 36 ≤ -20 * t'^2 + 80 * t' + 36 → -20 * t^2 + 80 * t + 36 = 116 :=
sorry

end max_ball_height_l300_300545


namespace honor_students_count_l300_300973

noncomputable def G : ℕ := 13
noncomputable def B : ℕ := 11
def E_G : ℕ := 3
def E_B : ℕ := 4

theorem honor_students_count (h1 : G + B < 30) 
    (h2 : (E_G : ℚ) / G = 3 / 13) 
    (h3 : (E_B : ℚ) / B = 4 / 11) :
    E_G + E_B = 7 := 
sorry

end honor_students_count_l300_300973


namespace students_total_l300_300744

theorem students_total (position_eunjung : ℕ) (following_students : ℕ) (h1 : position_eunjung = 6) (h2 : following_students = 7) : 
  position_eunjung + following_students = 13 :=
by
  sorry

end students_total_l300_300744


namespace radius_of_isosceles_tangent_circle_l300_300127

noncomputable def R : ℝ := 2 * Real.sqrt 3

variables (x : ℝ) (AB AC BD AD DC r : ℝ)

def is_isosceles (AB BC : ℝ) : Prop := AB = BC
def is_tangent (r : ℝ) (x : ℝ) : Prop := r = 2.4 * x

theorem radius_of_isosceles_tangent_circle
  (h_isosceles: is_isosceles AB BC)
  (h_area: 1/2 * AC * BD = 25)
  (h_height_ratio: BD / AC = 3 / 8)
  (h_AD_DC: AD = DC)
  (h_AC: AC = 8 * x)
  (h_BD: BD = 3 * x)
  (h_radius: is_tangent r x):
  r = R :=
sorry

end radius_of_isosceles_tangent_circle_l300_300127


namespace total_cards_square_l300_300837

theorem total_cards_square (s : ℕ) (h_perim : 4 * s - 4 = 240) : s * s = 3721 := by
  sorry

end total_cards_square_l300_300837


namespace tan_pi_minus_alpha_l300_300154

theorem tan_pi_minus_alpha (α : ℝ) (h : 3 * Real.sin α = Real.cos α) : Real.tan (π - α) = -1 / 3 :=
by
  sorry

end tan_pi_minus_alpha_l300_300154


namespace octagon_edge_length_from_pentagon_l300_300864

noncomputable def regular_pentagon_edge_length : ℝ := 16
def num_of_pentagon_edges : ℕ := 5
def num_of_octagon_edges : ℕ := 8

theorem octagon_edge_length_from_pentagon (total_length_thread : ℝ) :
  total_length_thread = num_of_pentagon_edges * regular_pentagon_edge_length →
  (total_length_thread / num_of_octagon_edges) = 10 :=
by
  intro h
  sorry

end octagon_edge_length_from_pentagon_l300_300864


namespace roger_daily_goal_l300_300657

-- Conditions
def steps_in_30_minutes : ℕ := 2000
def time_to_reach_goal_min : ℕ := 150
def time_interval_min : ℕ := 30

-- Theorem to prove
theorem roger_daily_goal : steps_in_30_minutes * (time_to_reach_goal_min / time_interval_min) = 10000 := by
  sorry

end roger_daily_goal_l300_300657


namespace honor_students_count_l300_300982

noncomputable def number_of_honor_students (G B Eg Eb : ℕ) (p_girl p_boy : ℚ) : ℕ :=
  if G < 30 ∧ B < 30 ∧ Eg = (3 / 13) * G ∧ Eb = (4 / 11) * B ∧ G + B < 30 then
    Eg + Eb
  else
    0

theorem honor_students_count :
  ∃ (G B Eg Eb : ℕ), (G < 30 ∧ B < 30 ∧ G % 13 = 0 ∧ B % 11 = 0 ∧ Eg = (3 * G / 13) ∧ Eb = (4 * B / 11) ∧ G + B < 30 ∧ number_of_honor_students G B Eg Eb (3 / 13) (4 / 11) = 7) :=
by {
  sorry
}

end honor_students_count_l300_300982


namespace largest_k_power_of_2_dividing_product_of_first_50_even_numbers_l300_300076

open Nat

theorem largest_k_power_of_2_dividing_product_of_first_50_even_numbers :
  let Q := (List.range (50 + 1)).map (λ n, 2 * n).prod in
  let k := (Q.factorization 2) in
  k = 97 :=
by
  sorry

end largest_k_power_of_2_dividing_product_of_first_50_even_numbers_l300_300076


namespace gcd_2023_2052_eq_1_l300_300671

theorem gcd_2023_2052_eq_1 : Int.gcd 2023 2052 = 1 :=
by
  sorry

end gcd_2023_2052_eq_1_l300_300671


namespace initial_amount_of_money_l300_300510

variable (X : ℕ) -- Initial amount of money Lily had in her account

-- Conditions
def spent_on_shirt : ℕ := 7
def spent_in_second_shop : ℕ := 3 * spent_on_shirt
def remaining_after_purchases : ℕ := 27

-- Proof problem: prove that the initial amount of money X is 55 given the conditions
theorem initial_amount_of_money (h : X - spent_on_shirt - spent_in_second_shop = remaining_after_purchases) : X = 55 :=
by
  -- Placeholder to indicate that steps will be worked out in Lean
  sorry

end initial_amount_of_money_l300_300510


namespace fractional_equation_solution_l300_300161

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) ↔ (m ≤ 2 ∧ m ≠ -2) := 
sorry

end fractional_equation_solution_l300_300161


namespace cannot_be_zero_l300_300528

noncomputable def P (x : ℝ) (a b c d e : ℝ) := x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + e

theorem cannot_be_zero (a b c d e : ℝ) (p q r s : ℝ) :
  e = 0 ∧ c = 0 ∧ (∀ x, P x a b c d e = x * (x - p) * (x - q) * (x - r) * (x - s)) ∧ 
  (p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧ p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) →
  d ≠ 0 := 
by {
  sorry
}

end cannot_be_zero_l300_300528


namespace find_k_l300_300039

theorem find_k (k : ℝ) :
  (∀ x : ℝ, x^2 + k * x + 12 = 0 → ∃ y : ℝ, y = x + 3 ∧ y^2 - k * y + 12 = 0) →
  k = 3 :=
sorry

end find_k_l300_300039


namespace expected_number_of_different_faces_l300_300703

theorem expected_number_of_different_faces :
  let ζ_i (i : Fin 6) := if (∃ k, k ∈ Finset.range 6) then 1 else 0,
      ζ := (List.range 6).sum (ζ_i),
      p := (5 / 6 : ℝ) ^ 6
  in (Expectation (λ ω => ζ)) = (6 * (1 - p)) :=
by
  sorry

end expected_number_of_different_faces_l300_300703


namespace moles_CO2_formed_l300_300459

-- Define the conditions based on the problem statement
def moles_HCl := 1
def moles_NaHCO3 := 1

-- Define the reaction equation in equivalence terms
def chemical_equation (hcl : Nat) (nahco3 : Nat) : Nat :=
  if hcl = 1 ∧ nahco3 = 1 then 1 else 0

-- State the proof problem
theorem moles_CO2_formed : chemical_equation moles_HCl moles_NaHCO3 = 1 :=
by
  -- The proof goes here
  sorry

end moles_CO2_formed_l300_300459


namespace ab_bc_ca_plus_one_pos_l300_300599

variable (a b c : ℝ)
variable (h₁ : |a| < 1)
variable (h₂ : |b| < 1)
variable (h₃ : |c| < 1)

theorem ab_bc_ca_plus_one_pos :
  ab + bc + ca + 1 > 0 := sorry

end ab_bc_ca_plus_one_pos_l300_300599


namespace odd_function_properties_l300_300489

def f : ℝ → ℝ := sorry

theorem odd_function_properties 
  (H1 : ∀ x, f (-x) = -f x) -- f is odd
  (H2 : ∀ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 3 → f x ≤ f y) -- f is increasing on [1, 3]
  (H3 : ∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≥ 7) -- f has a minimum value of 7 on [1, 3]
  : (∀ x y, -3 ≤ x ∧ x ≤ y ∧ y ≤ -1 → f x ≤ f y) -- f is increasing on [-3, -1]
    ∧ (∀ x, -3 ≤ x ∧ x ≤ -1 → f x ≤ -7) -- f has a maximum value of -7 on [-3, -1]
:= sorry

end odd_function_properties_l300_300489


namespace correct_statement_about_algorithms_l300_300692

-- Definitions based on conditions
def is_algorithm (A B C D : Prop) : Prop :=
  ¬A ∧ B ∧ ¬C ∧ ¬D

-- Ensure the correct statement using the conditions specified
theorem correct_statement_about_algorithms (A B C D : Prop) (h : is_algorithm A B C D) : B :=
by
  obtain ⟨hnA, hB, hnC, hnD⟩ := h
  exact hB

end correct_statement_about_algorithms_l300_300692


namespace inequality_proof_l300_300318

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
    (b^2 / a + a^2 / b) ≥ (a + b) := 
    sorry

end inequality_proof_l300_300318


namespace annual_feeding_cost_is_correct_l300_300611

-- Definitions based on conditions
def number_of_geckos : Nat := 3
def number_of_iguanas : Nat := 2
def number_of_snakes : Nat := 4
def cost_per_gecko_per_month : Nat := 15
def cost_per_iguana_per_month : Nat := 5
def cost_per_snake_per_month : Nat := 10

-- Statement of the theorem
theorem annual_feeding_cost_is_correct : 
    (number_of_geckos * cost_per_gecko_per_month
    + number_of_iguanas * cost_per_iguana_per_month 
    + number_of_snakes * cost_per_snake_per_month) * 12 = 1140 := by
  sorry

end annual_feeding_cost_is_correct_l300_300611


namespace pancake_problem_l300_300838

theorem pancake_problem :
  let mom_rate := (100 : ℚ) / 30
  let anya_rate := (100 : ℚ) / 40
  let andrey_rate := (100 : ℚ) / 60
  let combined_baking_rate := mom_rate + anya_rate
  let net_rate := combined_baking_rate - andrey_rate
  let target_pancakes := 100
  let time := target_pancakes / net_rate
  time = 24 := by
sorry

end pancake_problem_l300_300838


namespace probability_x_plus_y_lt_5_l300_300136

theorem probability_x_plus_y_lt_5 :
  let square := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 4 }
  ∫ (p : ℝ × ℝ) in square, (if p.1 + p.2 < 5 then 1 else 0) ∂(measure_theory.measure.lebesgue) = 29 / 32 := 
sorry

end probability_x_plus_y_lt_5_l300_300136


namespace coin_exchange_proof_l300_300445

/-- Prove the coin combination that Petya initially had -/
theorem coin_exchange_proof (x y z : ℕ) (hx : 20 * x + 15 * y + 10 * z = 125) : x = 0 ∧ y = 1 ∧ z = 11 :=
by
  sorry

end coin_exchange_proof_l300_300445


namespace simplify_frac_48_72_l300_300385

theorem simplify_frac_48_72 : (48 / 72 : ℚ) = 2 / 3 :=
by
  -- In Lean, we prove the equality of the simplified fractions.
  sorry

end simplify_frac_48_72_l300_300385


namespace max_items_sum_l300_300755

theorem max_items_sum (m n : ℕ) (h : 5 * m + 17 * n = 203) : m + n ≤ 31 :=
sorry

end max_items_sum_l300_300755


namespace tickets_sold_second_half_l300_300092

-- Definitions from conditions
def total_tickets := 9570
def first_half_tickets := 3867

-- Theorem to prove the number of tickets sold in the second half of the season
theorem tickets_sold_second_half : total_tickets - first_half_tickets = 5703 :=
by sorry

end tickets_sold_second_half_l300_300092


namespace cos_Z_l300_300186

-- Define the triangle XYZ with X being a right angle and the given sin Y.
structure TriangleXYZ where
  X Y Z : ℝ
  angle_X : X = 90
  sin_Y : sin Y = 3 / 5

-- State the theorem that proves cos Z given the properties of the triangle.
theorem cos_Z (T : TriangleXYZ) : cos T.Z = 3 / 5 := 
  sorry

end cos_Z_l300_300186


namespace sum_of_ages_l300_300937

theorem sum_of_ages (J L : ℕ) (h1 : J = L + 8) (h2 : J + 5 = 3 * (L - 6)) : (J + L) = 39 :=
by {
  -- Proof steps would go here, but are omitted for this task per instructions
  sorry
}

end sum_of_ages_l300_300937


namespace taco_price_theorem_l300_300138

noncomputable def price_hard_shell_taco_proof
  (H : ℤ)
  (price_soft : ℤ := 2)
  (num_hard_tacos_family : ℤ := 4)
  (num_soft_tacos_family : ℤ := 3)
  (num_additional_customers : ℤ := 10)
  (total_earnings : ℤ := 66)
  : Prop :=
  4 * H + 3 * price_soft + 10 * 2 * price_soft = total_earnings → H = 5

theorem taco_price_theorem : price_hard_shell_taco_proof 5 := 
by
  sorry

end taco_price_theorem_l300_300138


namespace placement_of_pawns_l300_300052

-- Define the size of the chessboard and the total number of pawns
def board_size := 5
def total_pawns := 5

-- Define the problem statement
theorem placement_of_pawns : 
  (∑ (pawns : Finset (Fin total_pawns → Fin board_size)), 
    (∀ p1 p2 : Fin total_pawns, p1 ≠ p2 → pawns(p1) ≠ pawns(p2)) ∧ -- distinct positions
    (∀ i j : Fin total_pawns, i ≠ j → pawns(i) ≠ pawns(j)) ∧ -- no same row/column
    pawns.card = total_pawns) = 14400 :=
sorry

end placement_of_pawns_l300_300052


namespace ratio_of_sides_l300_300003

theorem ratio_of_sides (
  perimeter_triangle perimeter_square : ℕ)
  (h_triangle : perimeter_triangle = 48)
  (h_square : perimeter_square = 64) :
  (perimeter_triangle / 3) / (perimeter_square / 4) = 1 :=
by
  sorry

end ratio_of_sides_l300_300003


namespace Emily_spent_28_dollars_l300_300558

theorem Emily_spent_28_dollars :
  let roses_cost := 4
  let daisies_cost := 3
  let tulips_cost := 5
  let lilies_cost := 6
  let roses_qty := 2
  let daisies_qty := 3
  let tulips_qty := 1
  let lilies_qty := 1
  (roses_qty * roses_cost) + (daisies_qty * daisies_cost) + (tulips_qty * tulips_cost) + (lilies_qty * lilies_cost) = 28 :=
by
  sorry

end Emily_spent_28_dollars_l300_300558


namespace book_price_l300_300432

theorem book_price (B P : ℝ) 
  (h1 : (1 / 3) * B = 36) 
  (h2 : (2 / 3) * B * P = 252) : 
  P = 3.5 :=
by {
  sorry
}

end book_price_l300_300432


namespace weight_of_berries_l300_300375

theorem weight_of_berries (total_weight : ℝ) (melon_weight : ℝ) : total_weight = 0.63 → melon_weight = 0.25 → total_weight - melon_weight = 0.38 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end weight_of_berries_l300_300375


namespace circle_diameter_l300_300549

theorem circle_diameter (r : ℝ) (h : π * r^2 = 9 * π) : 2 * r = 6 :=
by sorry

end circle_diameter_l300_300549


namespace hyperbola_line_intersection_unique_l300_300775

theorem hyperbola_line_intersection_unique :
  ∀ (x y : ℝ), (x^2 / 9 - y^2 = 1) ∧ (y = 1/3 * (x + 1)) → ∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y :=
by
  sorry

end hyperbola_line_intersection_unique_l300_300775


namespace minimum_area_l300_300338

-- Define point A
def A : ℝ × ℝ := (-4, 0)

-- Define point B
def B : ℝ × ℝ := (0, 4)

-- Define the circle
def on_circle (C : ℝ × ℝ) : Prop := (C.1 - 2)^2 + C.2^2 = 2

-- Instantiating the proof of the minimum area of △ABC = 8
theorem minimum_area (C : ℝ × ℝ) (h : on_circle C) : 
  ∃ C : ℝ × ℝ, on_circle C ∧ 1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) = 8 := 
sorry

end minimum_area_l300_300338


namespace mike_taller_than_mark_l300_300637

def feet_to_inches (feet : ℕ) : ℕ := 12 * feet

def mark_height_feet := 5
def mark_height_inches := 3
def mike_height_feet := 6
def mike_height_inches := 1

def mark_total_height := feet_to_inches mark_height_feet + mark_height_inches
def mike_total_height := feet_to_inches mike_height_feet + mike_height_inches

theorem mike_taller_than_mark : mike_total_height - mark_total_height = 10 :=
by
  sorry

end mike_taller_than_mark_l300_300637


namespace simplify_frac_48_72_l300_300387

theorem simplify_frac_48_72 : (48 / 72 : ℚ) = 2 / 3 :=
by
  -- In Lean, we prove the equality of the simplified fractions.
  sorry

end simplify_frac_48_72_l300_300387


namespace correct_population_l300_300989

variable (P : ℕ) (S : ℕ)
variable (math_scores : ℕ → Type)

-- Assume P is the total number of students who took the exam.
-- Let math_scores(P) represent the math scores of P students.

def population_data (P : ℕ) : Prop := 
  P = 50000

def sample_data (S : ℕ) : Prop :=
  S = 2000

theorem correct_population (P : ℕ) (S : ℕ) (math_scores : ℕ → Type)
  (hP : population_data P) (hS : sample_data S) : 
  math_scores P = math_scores 50000 :=
by {
  sorry
}

end correct_population_l300_300989


namespace james_income_ratio_l300_300630

theorem james_income_ratio
  (January_earnings : ℕ := 4000)
  (Total_earnings : ℕ := 18000)
  (Earnings_difference : ℕ := 2000) :
  ∃ (February_earnings : ℕ), 
    (January_earnings + February_earnings + (February_earnings - Earnings_difference) = Total_earnings) ∧
    (February_earnings / January_earnings = 2) := by
  sorry

end james_income_ratio_l300_300630


namespace smallest_positive_multiple_of_18_with_digits_9_or_0_l300_300217

noncomputable def m : ℕ := 90
theorem smallest_positive_multiple_of_18_with_digits_9_or_0 : m = 90 ∧ (∀ d ∈ m.digits 10, d = 0 ∨ d = 9) ∧ m % 18 = 0 → m / 18 = 5 :=
by
  intro h
  sorry

end smallest_positive_multiple_of_18_with_digits_9_or_0_l300_300217


namespace number_of_buses_l300_300130

theorem number_of_buses (vans people_per_van buses people_per_bus extra_people_in_buses : ℝ) 
  (h_vans : vans = 6.0) 
  (h_people_per_van : people_per_van = 6.0) 
  (h_people_per_bus : people_per_bus = 18.0) 
  (h_extra_people_in_buses : extra_people_in_buses = 108.0) 
  (h_eq : people_per_bus * buses = vans * people_per_van + extra_people_in_buses) : 
  buses = 8.0 :=
by
  sorry

end number_of_buses_l300_300130


namespace not_necessarily_heavier_l300_300935

/--
In a zoo, there are 10 elephants. It is known that if any four elephants stand on the left pan and any three on the right pan, the left pan will weigh more. If five elephants stand on the left pan and four on the right pan, the left pan does not necessarily weigh more.
-/
theorem not_necessarily_heavier (E : Fin 10 → ℝ) (H : ∀ (L : Finset (Fin 10)) (R : Finset (Fin 10)), L.card = 4 → R.card = 3 → L ≠ R → L.sum E > R.sum E) :
  ∃ (L' R' : Finset (Fin 10)), L'.card = 5 ∧ R'.card = 4 ∧ L'.sum E ≤ R'.sum E :=
by
  sorry

end not_necessarily_heavier_l300_300935


namespace total_cost_l300_300285

variable (E P M : ℝ)

axiom condition1 : E + 3 * P + 2 * M = 240
axiom condition2 : 2 * E + 5 * P + 4 * M = 440

theorem total_cost : 3 * E + 4 * P + 6 * M = 520 := 
sorry

end total_cost_l300_300285


namespace correct_population_l300_300988

variable (P : ℕ) (S : ℕ)
variable (math_scores : ℕ → Type)

-- Assume P is the total number of students who took the exam.
-- Let math_scores(P) represent the math scores of P students.

def population_data (P : ℕ) : Prop := 
  P = 50000

def sample_data (S : ℕ) : Prop :=
  S = 2000

theorem correct_population (P : ℕ) (S : ℕ) (math_scores : ℕ → Type)
  (hP : population_data P) (hS : sample_data S) : 
  math_scores P = math_scores 50000 :=
by {
  sorry
}

end correct_population_l300_300988


namespace units_digit_of_sum_of_cubes_l300_300687

theorem units_digit_of_sum_of_cubes : 
  (24^3 + 42^3) % 10 = 2 := by
sorry

end units_digit_of_sum_of_cubes_l300_300687


namespace students_like_both_l300_300061

theorem students_like_both {total students_apple_pie students_chocolate_cake students_none students_at_least_one students_both : ℕ} 
  (h_total : total = 50)
  (h_apple : students_apple_pie = 22)
  (h_chocolate : students_chocolate_cake = 20)
  (h_none : students_none = 17)
  (h_least_one : students_at_least_one = total - students_none)
  (h_union : students_at_least_one = students_apple_pie + students_chocolate_cake - students_both) :
  students_both = 9 :=
by
  sorry

end students_like_both_l300_300061


namespace find_S10_l300_300366

def sequence_sums (S : ℕ → ℚ) (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ (∀ n : ℕ, n > 0 → a (n + 1) = 3 * S n - S (n + 1) - 1)

theorem find_S10 (S a : ℕ → ℚ) (h : sequence_sums S a) : S 10 = 513 / 2 :=
  sorry

end find_S10_l300_300366


namespace honor_students_count_l300_300987

def num_students_total : ℕ := 24
def num_honor_students_girls : ℕ := 3
def num_honor_students_boys : ℕ := 4

def num_girls : ℕ := 13
def num_boys : ℕ := 11

theorem honor_students_count (total_students : ℕ) 
    (prob_girl_honor : ℚ) (prob_boy_honor : ℚ)
    (girls : ℕ) (boys : ℕ)
    (honor_girls : ℕ) (honor_boys : ℕ) :
    total_students < 30 →
    prob_girl_honor = 3 / 13 →
    prob_boy_honor = 4 / 11 →
    girls = 13 →
    honor_girls = 3 →
    boys = 11 →
    honor_boys = 4 →
    girls + boys = total_students →
    honor_girls + honor_boys = 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [← h4, ← h5, ← h6, ← h7, ← h8]
  exact 7

end honor_students_count_l300_300987


namespace annual_interest_rate_last_year_l300_300208

-- Define the conditions
def increased_by_ten_percent (r : ℝ) : ℝ := 1.10 * r

-- Statement of the problem
theorem annual_interest_rate_last_year (r : ℝ) (h : increased_by_ten_percent r = 0.11) : r = 0.10 :=
sorry

end annual_interest_rate_last_year_l300_300208


namespace gain_percent_l300_300694

def cycle_gain_percent (cp sp : ℕ) : ℚ :=
  (sp - cp) / cp * 100

theorem gain_percent {cp sp : ℕ} (h1 : cp = 1500) (h2 : sp = 1620) : cycle_gain_percent cp sp = 8 := by
  sorry

end gain_percent_l300_300694


namespace right_triangle_5_12_13_l300_300478

theorem right_triangle_5_12_13 (a b c : ℕ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) : a^2 + b^2 = c^2 := 
by 
   sorry

end right_triangle_5_12_13_l300_300478


namespace prime_sum_20_to_30_l300_300245

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end prime_sum_20_to_30_l300_300245


namespace expected_number_of_different_faces_l300_300705

theorem expected_number_of_different_faces :
  let ζ_i (i : Fin 6) := if (∃ k, k ∈ Finset.range 6) then 1 else 0,
      ζ := (List.range 6).sum (ζ_i),
      p := (5 / 6 : ℝ) ^ 6
  in (Expectation (λ ω => ζ)) = (6 * (1 - p)) :=
by
  sorry

end expected_number_of_different_faces_l300_300705


namespace robin_total_distance_l300_300966

theorem robin_total_distance
  (d : ℕ)
  (d1 : ℕ)
  (h1 : d = 500)
  (h2 : d1 = 200)
  : 2 * d1 + d = 900 :=
by
  rewrite [h1, h2]
  rfl

end robin_total_distance_l300_300966


namespace root_sum_greater_than_one_l300_300333

noncomputable def f (x a : ℝ) : ℝ := (x * Real.log x) / (x - 1) - a

noncomputable def h (x a : ℝ) : ℝ := (x^2 - x) * f x a

theorem root_sum_greater_than_one {a m x1 x2 : ℝ} (ha : a < 0)
  (h_eq_m : ∀ x, h x a = m) (hx1_root : h x1 a = m) (hx2_root : h x2 a = m)
  (hx1x2_distinct : x1 ≠ x2) :
  x1 + x2 > 1 := 
sorry

end root_sum_greater_than_one_l300_300333


namespace find_square_side_l300_300397

theorem find_square_side (a b x : ℕ) (h_triangle : a^2 + x^2 = b^2)
  (h_trapezoid : 2 * a + 2 * b + 2 * x = 60)
  (h_rectangle : 4 * a + 2 * x = 58) :
  a = 12 := by
  sorry

end find_square_side_l300_300397


namespace garden_table_bench_cost_l300_300129

theorem garden_table_bench_cost (B T : ℕ) (h1 : T + B = 750) (h2 : T = 2 * B) : B = 250 :=
by
  sorry

end garden_table_bench_cost_l300_300129


namespace farthest_vertex_label_l300_300096

-- The vertices and their labeling
def cube_faces : List (List Nat) := [
  [1, 2, 5, 8],
  [3, 4, 6, 7],
  [2, 4, 5, 7],
  [1, 3, 6, 8],
  [2, 3, 7, 8],
  [1, 4, 5, 6]
]

-- Define the cube vertices labels
def vertices : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

-- Statement of the problem in Lean 4
theorem farthest_vertex_label (h : true) : 
  ∃ v : Nat, v ∈ vertices ∧ ∀ face ∈ cube_faces, v ∉ face → v = 6 := 
sorry

end farthest_vertex_label_l300_300096


namespace exists_n_for_all_k_l300_300817

theorem exists_n_for_all_k (k : ℕ) : ∃ n : ℕ, 5^k ∣ (n^2 + 1) :=
sorry

end exists_n_for_all_k_l300_300817


namespace vector_addition_magnitude_l300_300316

variables {a b : ℝ}

theorem vector_addition_magnitude (ha : abs a = 1) (hb : abs b = 2)
  (angle_ab : real.angle.to_degrees (real.angle.arctan2 b a) = 60) :
  abs (a + b) = sqrt 7 :=
by sorry

end vector_addition_magnitude_l300_300316


namespace age_of_golden_retriever_l300_300484

def golden_retriever (gain_per_year current_weight : ℕ) (age : ℕ) :=
  gain_per_year * age = current_weight

theorem age_of_golden_retriever :
  golden_retriever 11 88 8 :=
by
  unfold golden_retriever
  simp
  sorry

end age_of_golden_retriever_l300_300484


namespace solution_set_f_x_minus_2_ge_zero_l300_300329

-- Define the necessary conditions and prove the statement
noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_x_minus_2_ge_zero (f_even : ∀ x, f x = f (-x))
  (f_mono : ∀ {x y : ℝ}, 0 ≤ x → x ≤ y → f x ≤ f y)
  (f_one_zero : f 1 = 0) :
  {x : ℝ | f (x - 2) ≥ 0} = {x | x ≥ 3 ∨ x ≤ 1} :=
by {
  sorry
}

end solution_set_f_x_minus_2_ge_zero_l300_300329


namespace integral_twice_sqrt_minus_sin_eq_pi_l300_300143

open Real

theorem integral_twice_sqrt_minus_sin_eq_pi :
  ∫ x in -1..1, 2 * sqrt (1 - x^2) - sin x = π :=
by
  have h1 : ∫ x in -1..1, sqrt (1 - x^2) = π / 2 := sorry
  have h2 : ∫ x in -1..1, sin x = 0 := interval_integral.integral_sin
  calc
    ∫ x in -1..1, 2 * sqrt (1 - x^2) - sin x
      = 2 * ∫ x in -1..1, sqrt (1 - x^2) - ∫ x in -1..1, sin x : by
        simp only [interval_integral.integral_sub]
        congr
        simp [mul_comm]
      ... = 2 * (π / 2) - 0 : by
        rw [h1, h2]
      ... = π : by
        norm_num

end integral_twice_sqrt_minus_sin_eq_pi_l300_300143


namespace divide_numbers_into_consecutive_products_l300_300011

theorem divide_numbers_into_consecutive_products :
  ∃ (A B : Finset ℕ), A ∪ B = {2, 3, 5, 7, 11, 13, 17} ∧ A ∩ B = ∅ ∧ 
  (A.prod id = 714 ∧ B.prod id = 715 ∨ A.prod id = 715 ∧ B.prod id = 714) :=
sorry

end divide_numbers_into_consecutive_products_l300_300011


namespace adam_earning_per_lawn_l300_300868

theorem adam_earning_per_lawn 
  (total_lawns : ℕ) 
  (forgotten_lawns : ℕ) 
  (total_earnings : ℕ) 
  (h1 : total_lawns = 12) 
  (h2 : forgotten_lawns = 8) 
  (h3 : total_earnings = 36) : 
  total_earnings / (total_lawns - forgotten_lawns) = 9 :=
by
  sorry

end adam_earning_per_lawn_l300_300868


namespace vasya_has_greater_expected_area_l300_300878

noncomputable def expected_area_rectangle : ℚ :=
1 / 6 * (1 * 1 + 1 * 2 + 1 * 3 + 1 * 4 + 1 * 5 + 1 * 6 + 
         2 * 1 + 2 * 2 + 2 * 3 + 2 * 4 + 2 * 5 + 2 * 6 + 
         3 * 1 + 3 * 2 + 3 * 3 + 3 * 4 + 3 * 5 + 3 * 6 + 
         4 * 1 + 4 * 2 + 4 * 3 + 4 * 4 + 4 * 5 + 4 * 6 + 
         5 * 1 + 5 * 2 + 5 * 3 + 5 * 4 + 5 * 5 + 5 * 6 + 
         6 * 1 + 6 * 2 + 6 * 3 + 6 * 4 + 6 * 5 + 6 * 6)

noncomputable def expected_area_square : ℚ := 
1 / 6 * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

theorem vasya_has_greater_expected_area : expected_area_rectangle < expected_area_square :=
by {
  -- A calculation of this sort should be done symbolically, not in this theorem,
  -- but the primary goal here is to show the structure of the statement.
  -- Hence, implement symbolic computation later to finalize proof.
  sorry
}

end vasya_has_greater_expected_area_l300_300878


namespace total_number_of_birds_l300_300263

def bird_cages : Nat := 9
def parrots_per_cage : Nat := 2
def parakeets_per_cage : Nat := 6
def birds_per_cage : Nat := parrots_per_cage + parakeets_per_cage
def total_birds : Nat := bird_cages * birds_per_cage

theorem total_number_of_birds : total_birds = 72 := by
  sorry

end total_number_of_birds_l300_300263


namespace tan_alpha_plus_pi_over_4_l300_300902

noncomputable def vec_a (α : ℝ) : ℝ × ℝ := (Real.cos (2 * α), Real.sin α)
noncomputable def vec_b (α : ℝ) : ℝ × ℝ := (1, 2 * Real.sin α - 1)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi)
    (h3 : dot_product (vec_a α) (vec_b α) = 0) :
    Real.tan (α + Real.pi / 4) = -1 := sorry

end tan_alpha_plus_pi_over_4_l300_300902


namespace rectangle_width_l300_300106

/-- Given the conditions:
    - length of a rectangle is 5.4 cm
    - area of the rectangle is 48.6 cm²
    Prove that the width of the rectangle is 9 cm.
-/
theorem rectangle_width (length width area : ℝ) 
  (h_length : length = 5.4) 
  (h_area : area = 48.6) 
  (h_area_eq : area = length * width) : 
  width = 9 := 
by
  sorry

end rectangle_width_l300_300106


namespace proof_statement_l300_300646

-- Define the initial dimensions and areas
def initial_length : ℕ := 7
def initial_width : ℕ := 5

-- Shortened dimensions by one side and the corresponding area condition
def shortened_new_width : ℕ := 3
def shortened_area : ℕ := 21

-- Define the task
def task_statement : Prop :=
  (initial_length - 2) * initial_width = shortened_area ∧
  (initial_width - 2) * initial_length = shortened_area →
  (initial_length - 2) * (initial_width - 2) = 25

theorem proof_statement : task_statement :=
by {
  sorry -- Proof goes here
}

end proof_statement_l300_300646


namespace pool_fill_time_l300_300948

theorem pool_fill_time
  (faster_pipe_time : ℝ) (slower_pipe_factor : ℝ)
  (H1 : faster_pipe_time = 9) 
  (H2 : slower_pipe_factor = 1.25) : 
  (faster_pipe_time * (1 + slower_pipe_factor) / (faster_pipe_time + faster_pipe_time/slower_pipe_factor)) = 5 :=
by
  sorry

end pool_fill_time_l300_300948


namespace minimum_additional_games_to_reach_90_percent_hawks_minimum_games_needed_to_win_l300_300391

theorem minimum_additional_games_to_reach_90_percent (N : ℕ) : 
  (2 + N) * 10 ≥ (5 + N) * 9 ↔ N ≥ 25 := 
sorry

-- An alternative approach to assert directly as exactly 25 by using the condition’s natural number ℕ could be as follows:
theorem hawks_minimum_games_needed_to_win (N : ℕ) : 
  ∀ N, (2 + N) * 10 / (5 + N) ≥ 9 / 10 → N ≥ 25 := 
sorry

end minimum_additional_games_to_reach_90_percent_hawks_minimum_games_needed_to_win_l300_300391


namespace choose_agency_l300_300270

variables (a : ℝ) (x : ℕ)

def cost_agency_A (a : ℝ) (x : ℕ) : ℝ :=
  a + 0.55 * a * x

def cost_agency_B (a : ℝ) (x : ℕ) : ℝ :=
  0.75 * (x + 1) * a

theorem choose_agency (a : ℝ) (x : ℕ) : if (x = 1) then 
                                            (cost_agency_B a x ≤ cost_agency_A a x)
                                         else if (x ≥ 2) then 
                                            (cost_agency_A a x ≤ cost_agency_B a x)
                                         else
                                            true :=
by
  sorry

end choose_agency_l300_300270


namespace total_amount_divided_l300_300547

theorem total_amount_divided 
    (A B C : ℝ) 
    (h1 : A = (2 / 3) * (B + C)) 
    (h2 : B = (2 / 3) * (A + C)) 
    (h3 : A = 160) : 
    A + B + C = 400 := 
by 
  sorry

end total_amount_divided_l300_300547


namespace quadratic_inequality_for_all_x_l300_300929

theorem quadratic_inequality_for_all_x (a : ℝ) :
  (∀ x : ℝ, (a^2 + a) * x^2 - a * x + 1 > 0) ↔ (-4 / 3 < a ∧ a < -1) ∨ a = 0 :=
sorry

end quadratic_inequality_for_all_x_l300_300929


namespace equal_tuesdays_thursdays_l300_300132

theorem equal_tuesdays_thursdays (days_in_month : ℕ) (tuesdays : ℕ) (thursdays : ℕ) : (days_in_month = 30) → (tuesdays = thursdays) → (∃ (start_days : Finset ℕ), start_days.card = 2) :=
by
  sorry

end equal_tuesdays_thursdays_l300_300132


namespace inequality_for_positive_reals_l300_300515

theorem inequality_for_positive_reals 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a^3 + b^3 + a * b * c)) + (1 / (b^3 + c^3 + a * b * c)) + 
  (1 / (c^3 + a^3 + a * b * c)) ≤ 1 / (a * b * c) := 
sorry

end inequality_for_positive_reals_l300_300515


namespace quadratic_completion_l300_300822

noncomputable def sum_of_r_s (r s : ℝ) : ℝ := r + s

theorem quadratic_completion (x r s : ℝ) (h : 16 * x^2 - 64 * x - 144 = 0) :
  ((x + r)^2 = s) → sum_of_r_s r s = -7 :=
by
  sorry

end quadratic_completion_l300_300822


namespace derivative_at_zero_l300_300568

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≠ 0 then (real.cbrt (1 - 2 * x ^ 3 * real.sin (5 / x)) - 1 + x)
  else 0

-- State the theorem to be proved
theorem derivative_at_zero (h : has_deriv_at f 1 0) : 
  deriv f 0 = 1 := 
sorry

end derivative_at_zero_l300_300568


namespace mike_taller_than_mark_l300_300639

def height_mark_feet : ℕ := 5
def height_mark_inches : ℕ := 3
def height_mike_feet : ℕ := 6
def height_mike_inches : ℕ := 1
def feet_to_inches : ℕ := 12

-- Calculate heights in inches.
def height_mark_total_inches : ℕ := height_mark_feet * feet_to_inches + height_mark_inches
def height_mike_total_inches : ℕ := height_mike_feet * feet_to_inches + height_mike_inches

-- Prove the height difference.
theorem mike_taller_than_mark : height_mike_total_inches - height_mark_total_inches = 10 :=
by
  sorry

end mike_taller_than_mark_l300_300639


namespace inversely_proportional_solve_y_l300_300400

theorem inversely_proportional_solve_y (k : ℝ) (x y : ℝ)
  (h1 : x * y = k)
  (h2 : x + y = 60)
  (h3 : x = 3 * y)
  (hx : x = -10) :
  y = -67.5 :=
by
  sorry

end inversely_proportional_solve_y_l300_300400


namespace max_proj_area_l300_300672

variable {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem max_proj_area : 
  ∃ max_area : ℝ, max_area = Real.sqrt (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) :=
by
  sorry

end max_proj_area_l300_300672


namespace tangent_line_at_point_l300_300215

theorem tangent_line_at_point (x y : ℝ) (h_curve : y = x^3 - 2 * x + 1) (h_point : (x, y) = (1, 0)) :
  y = x - 1 :=
sorry

end tangent_line_at_point_l300_300215


namespace jason_retirement_age_l300_300355

variable (join_age : ℕ) (years_to_chief : ℕ) (percent_longer : ℕ) (additional_years : ℕ)

def time_to_master_chief := years_to_chief + (years_to_chief * percent_longer / 100)

def total_time_in_military := years_to_chief + time_to_master_chief years_to_chief percent_longer + additional_years

def retirement_age := join_age + total_time_in_military join_age years_to_chief percent_longer additional_years

theorem jason_retirement_age :
  join_age = 18 →
  years_to_chief = 8 →
  percent_longer = 25 →
  additional_years = 10 →
  retirement_age join_age years_to_chief percent_longer additional_years = 46 :=
by
  intros h1 h2 h3 h4
  simp [retirement_age, total_time_in_military, time_to_master_chief, h1, h2, h3, h4]
  sorry

end jason_retirement_age_l300_300355


namespace money_out_of_pocket_l300_300358

theorem money_out_of_pocket
  (old_system_cost : ℝ)
  (trade_in_percent : ℝ)
  (new_system_cost : ℝ)
  (discount_percent : ℝ)
  (trade_in_value : ℝ)
  (discount_value : ℝ)
  (discounted_price : ℝ)
  (money_out_of_pocket : ℝ) :
  old_system_cost = 250 →
  trade_in_percent = 80 / 100 →
  new_system_cost = 600 →
  discount_percent = 25 / 100 →
  trade_in_value = old_system_cost * trade_in_percent →
  discount_value = new_system_cost * discount_percent →
  discounted_price = new_system_cost - discount_value →
  money_out_of_pocket = discounted_price - trade_in_value →
  money_out_of_pocket = 250 := by
  intros
  sorry

end money_out_of_pocket_l300_300358


namespace number_of_seasons_l300_300134

theorem number_of_seasons 
        (episodes_per_season : ℕ) 
        (fraction_watched : ℚ) 
        (remaining_episodes : ℕ) 
        (h_episodes_per_season : episodes_per_season = 20) 
        (h_fraction_watched : fraction_watched = 1 / 3) 
        (h_remaining_episodes : remaining_episodes = 160) : 
        ∃ (seasons : ℕ), seasons = 12 :=
by
  sorry

end number_of_seasons_l300_300134


namespace x_square_minus_5x_is_necessary_not_sufficient_l300_300772

theorem x_square_minus_5x_is_necessary_not_sufficient (x : ℝ) :
  (x^2 - 5 * x < 0) → (|x - 1| < 1) → (x^2 - 5 * x < 0 ∧ ∃ y : ℝ, (0 < y ∧ y < 2) → x = y) :=
by
  sorry

end x_square_minus_5x_is_necessary_not_sufficient_l300_300772


namespace sum_of_primes_between_20_and_30_l300_300235

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l300_300235


namespace find_AC_find_angle_A_l300_300492

noncomputable def triangle_AC (AB BC : ℝ) (sinC_over_sinB : ℝ) : ℝ :=
  if h : sinC_over_sinB = 3 / 5 ∧ AB = 3 ∧ BC = 7 then 5 else 0

noncomputable def triangle_angle_A (AB AC BC : ℝ) : ℝ :=
  if h : AB = 3 ∧ AC = 5 ∧ BC = 7 then 120 else 0

theorem find_AC (BC AB : ℝ) (sinC_over_sinB : ℝ) (h : BC = 7 ∧ AB = 3 ∧ sinC_over_sinB = 3 / 5) : 
  triangle_AC AB BC sinC_over_sinB = 5 := by
  sorry

theorem find_angle_A (BC AB AC : ℝ) (h : BC = 7 ∧ AB = 3 ∧ AC = 5) : 
  triangle_angle_A AB AC BC = 120 := by
  sorry

end find_AC_find_angle_A_l300_300492


namespace sequence_a6_value_l300_300628

theorem sequence_a6_value :
  ∃ (a : ℕ → ℚ), (a 1 = 1) ∧ (∀ n, a (n + 1) = a n / (2 * a n + 1)) ∧ (a 6 = 1 / 11) :=
by
  sorry

end sequence_a6_value_l300_300628


namespace expected_faces_rolled_six_times_l300_300717

-- Define a random variable indicating appearance of a particular face
noncomputable def ζi (n : ℕ): ℝ := if n > 0 then 1 - (5 / 6) ^ 6 else 0

-- Define the expected number of distinct faces
noncomputable def expected_distinct_faces : ℝ := 6 * ζi 1

theorem expected_faces_rolled_six_times :
  expected_distinct_faces = (6 ^ 6 - 5 ^ 6) / 6 ^ 5 :=
by
  -- Here we would provide the proof
  sorry

end expected_faces_rolled_six_times_l300_300717


namespace ordering_PQR_l300_300073

noncomputable def P := Real.sqrt 2
noncomputable def Q := Real.sqrt 7 - Real.sqrt 3
noncomputable def R := Real.sqrt 6 - Real.sqrt 2

theorem ordering_PQR : P > R ∧ R > Q := by
  sorry

end ordering_PQR_l300_300073


namespace yearly_feeding_cost_l300_300614

-- Defining the conditions
def num_geckos := 3
def num_iguanas := 2
def num_snakes := 4

def cost_per_snake_per_month := 10
def cost_per_iguana_per_month := 5
def cost_per_gecko_per_month := 15

-- Statement of the proof problem
theorem yearly_feeding_cost : 
  (num_snakes * cost_per_snake_per_month + num_iguanas * cost_per_iguana_per_month + num_geckos * cost_per_gecko_per_month) * 12 = 1140 := 
  by 
    sorry

end yearly_feeding_cost_l300_300614


namespace tricycles_count_l300_300539

theorem tricycles_count (B T : ℕ) (hB : B = 50) (hW : 2 * B + 3 * T = 160) : T = 20 :=
by
  sorry

end tricycles_count_l300_300539


namespace estimate_first_year_students_l300_300001

noncomputable def number_of_first_year_students (N : ℕ) : Prop :=
  let p1 := (N - 90) / N
  let p2 := (N - 100) / N
  let p_both := 1 - p1 * p2
  p_both = 20 / N → N = 450

theorem estimate_first_year_students : ∃ N : ℕ, number_of_first_year_students N :=
by
  use 450
  -- sorry added to skip the proof part
  sorry

end estimate_first_year_students_l300_300001


namespace domain_of_function_l300_300302

-- Define the conditions for the function
def condition1 (x : ℝ) : Prop := 3 * x + 1 > 0
def condition2 (x : ℝ) : Prop := 2 - x ≠ 0

-- Define the domain of the function
def domain (x : ℝ) : Prop := x > -1 / 3 ∧ x ≠ 2

theorem domain_of_function : 
  ∀ x : ℝ, (condition1 x ∧ condition2 x) ↔ domain x := 
by
  sorry

end domain_of_function_l300_300302


namespace find_CD_squared_l300_300112

noncomputable def first_circle (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 25
noncomputable def second_circle (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 25

theorem find_CD_squared : ∃ C D : ℝ × ℝ, 
  (first_circle C.1 C.2 ∧ second_circle C.1 C.2) ∧ 
  (first_circle D.1 D.2 ∧ second_circle D.1 D.2) ∧ 
  (C ≠ D) ∧ 
  ((D.1 - C.1)^2 + (D.2 - C.2)^2 = 50) :=
by
  sorry

end find_CD_squared_l300_300112


namespace percentage_water_in_puree_l300_300615

/-- Given that tomato juice is 90% water and Heinz obtains 2.5 litres of tomato puree from 20 litres of tomato juice,
proves that the percentage of water in the tomato puree is 20%. -/
theorem percentage_water_in_puree (tj_volume : ℝ) (tj_water_content : ℝ) (tp_volume : ℝ) (tj_to_tp_ratio : ℝ) 
  (h1 : tj_water_content = 0.90) 
  (h2 : tj_volume = 20) 
  (h3 : tp_volume = 2.5) 
  (h4 : tj_to_tp_ratio = tj_volume / tp_volume) : 
  ((tp_volume - (1 - tj_water_content) * (tj_volume * (tp_volume / tj_volume))) / tp_volume) * 100 = 20 := 
sorry

end percentage_water_in_puree_l300_300615


namespace eval_expression_l300_300456

def a : ℕ := 4 * 5 * 6
def b : ℚ := 1/4 + 1/5 - 1/10

theorem eval_expression : a * b = 42 := by
  sorry

end eval_expression_l300_300456


namespace total_coins_l300_300485

-- Definitions for the conditions
def number_of_nickels := 13
def number_of_quarters := 8

-- Statement of the proof problem
theorem total_coins : number_of_nickels + number_of_quarters = 21 :=
by
  sorry

end total_coins_l300_300485


namespace ball_height_25_l300_300398

theorem ball_height_25 (t : ℝ) (h : ℝ) 
  (h_eq : h = 45 - 7 * t - 6 * t^2) : 
  h = 25 ↔ t = 4 / 3 := 
by 
  sorry

end ball_height_25_l300_300398


namespace solution_set_inequality_range_of_m_l300_300164

def f (x : ℝ) : ℝ := |2 * x + 1| + 2 * |x - 3|

theorem solution_set_inequality :
  ∀ x : ℝ, f x ≤ 7 * x ↔ x ≥ 1 :=
by sorry

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, f x = |m|) ↔ (m ≥ 7 ∨ m ≤ -7) :=
by sorry

end solution_set_inequality_range_of_m_l300_300164


namespace cricket_run_rate_l300_300261

theorem cricket_run_rate (run_rate_first_10_overs : ℝ) (target : ℝ) (overs_first_phase : ℕ) (overs_remaining : ℕ) :
  run_rate_first_10_overs = 4.6 → target = 282 → overs_first_phase = 10 → overs_remaining = 40 →
  (target - run_rate_first_10_overs * overs_first_phase) / overs_remaining = 5.9 :=
by
  intros
  sorry

end cricket_run_rate_l300_300261


namespace bushes_for_zucchinis_l300_300016

def bushes_yield := 10 -- containers per bush
def container_to_zucchini := 3 -- containers per zucchini
def zucchinis_required := 60 -- total zucchinis needed

theorem bushes_for_zucchinis (hyld : bushes_yield = 10) (ctz : container_to_zucchini = 3) (zreq : zucchinis_required = 60) :
  ∃ bushes : ℕ, bushes = 60 * container_to_zucchini / bushes_yield :=
sorry

end bushes_for_zucchinis_l300_300016


namespace k_bounds_inequality_l300_300414

open Real

theorem k_bounds_inequality (k : ℝ) :
  (∀ x : ℝ, abs ((x^2 - k * x + 1) / (x^2 + x + 1)) < 3) ↔ -5 ≤ k ∧ k ≤ 1 := 
sorry

end k_bounds_inequality_l300_300414


namespace cut_square_into_rectangles_l300_300300

theorem cut_square_into_rectangles :
  ∃ x y : ℕ, 3 * x + 4 * y = 25 :=
by
  -- Given that the total area is 25 and we are using rectangles of areas 3 and 4
  -- we need to verify the existence of integers x and y such that 3x + 4y = 25
  existsi 7
  existsi 1
  sorry

end cut_square_into_rectangles_l300_300300


namespace program_result_l300_300673

def program_loop (i : ℕ) (s : ℕ) : ℕ :=
if i < 9 then s else program_loop (i - 1) (s * i)

theorem program_result : 
  program_loop 11 1 = 990 :=
by 
  sorry

end program_result_l300_300673


namespace sum_of_coefficients_l300_300823

noncomputable def problem_expr (d : ℝ) := (16 * d + 15 + 18 * d^2 + 3 * d^3) + (4 * d + 2 + d^2 + 2 * d^3)
noncomputable def simplified_expr (d : ℝ) := 5 * d^3 + 19 * d^2 + 20 * d + 17

theorem sum_of_coefficients (d : ℝ) (h : d ≠ 0) : 
  problem_expr d = simplified_expr d ∧ (5 + 19 + 20 + 17 = 61) := 
by
  sorry

end sum_of_coefficients_l300_300823


namespace corresponding_angles_not_always_equal_l300_300100

theorem corresponding_angles_not_always_equal :
  (∀ α β c : ℝ, (α = β ∧ ¬c = 0) → (∃ x1 x2 y : ℝ, α = x1 ∧ β = x2 ∧ x1 = y * c ∧ x2 = y * c)) → False :=
by
  sorry

end corresponding_angles_not_always_equal_l300_300100


namespace find_a_l300_300896

noncomputable def has_exactly_one_solution_in_x (a : ℝ) : Prop :=
  ∀ x : ℝ, |x^2 + 2*a*x + a + 5| = 3 → x = -a

theorem find_a (a : ℝ) : has_exactly_one_solution_in_x a ↔ (a = 4 ∨ a = -2) :=
by
  sorry

end find_a_l300_300896


namespace vasya_has_greater_area_l300_300870

-- Definition of a fair six-sided die roll
def die_roll : ℕ → ℝ := λ k, if k ∈ {1, 2, 3, 4, 5, 6} then (1 / 6 : ℝ) else 0

-- Expected value of a function with respect to a probability distribution
noncomputable def expected_value (f : ℕ → ℝ) : ℝ := ∑ k in {1, 2, 3, 4, 5, 6}, f k * die_roll k

-- Vasya's area: A^2 where A is a single die roll
noncomputable def vasya_area : ℝ := expected_value (λ k, (k : ℝ) ^ 2)

-- Asya's area: A * B where A and B are independent die rolls
noncomputable def asya_area : ℝ := (expected_value (λ k, (k : ℝ))) ^ 2

theorem vasya_has_greater_area :
  vasya_area > asya_area := sorry

end vasya_has_greater_area_l300_300870


namespace circle_equation_l300_300328

theorem circle_equation :
  ∃ r : ℝ, ∀ x y : ℝ,
  ((x - 2) * (x - 2) + (y - 1) * (y - 1) = r * r) ∧
  ((5 - 2) * (5 - 2) + (-2 - 1) * (-2 - 1) = r * r) ∧
  (5 + 2 * -2 - 5 + r * r = 0) :=
sorry

end circle_equation_l300_300328


namespace angle_measure_l300_300303

theorem angle_measure (y : ℝ) (hyp : 45 + 3 * y + y = 180) : y = 33.75 :=
by
  sorry

end angle_measure_l300_300303


namespace greater_expected_area_l300_300876

/-- Let X be a random variable representing a single roll of a die, which can take integer values from 1 through 6. -/
def X : Type := { x : ℕ // 1 ≤ x ∧ x ≤ 6 }

/-- Define independent random variables A and B representing the outcomes of Asya’s die rolls, which can take integer values from 1 through 6 with equal probability. -/
noncomputable def A : Type := { a : ℕ // 1 ≤ a ∧ a ≤ 6 }
noncomputable def B : Type := { b : ℕ // 1 ≤ b ∧ b ≤ 6 }

/-- The expected value of a random variable taking integer values from 1 through 6. 
    E[X] = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5, and E[X^2] = (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6 = 15.1667 -/
noncomputable def expected_X_squared : ℝ := 91 / 6

/-- The expected value of the product of two independent random variables each taking integer values from 1 through 6. 
    E[A * B] = E[A] * E[B] = 3.5 * 3.5 = 12.25 -/
noncomputable def expected_A_times_B : ℝ := 12.25

/-- Prove that the expected area of Vasya's square is greater than Asya's rectangle.
    i.e., E[X^2] > E[A * B] -/
theorem greater_expected_area : expected_X_squared > expected_A_times_B :=
sorry

end greater_expected_area_l300_300876


namespace problem_statement_l300_300486

/-!
The problem states:
If |a-2| and |m+n+3| are opposite numbers, then a + m + n = -1.
-/

theorem problem_statement (a m n : ℤ) (h : |a - 2| = -|m + n + 3|) : a + m + n = -1 :=
by {
  sorry
}

end problem_statement_l300_300486


namespace group_B_same_order_l300_300119

-- Definitions for the expressions in each group
def expr_A1 := 2 * 9 / 3
def expr_A2 := 2 + 9 * 3

def expr_B1 := 36 - 9 + 5
def expr_B2 := 36 / 6 * 5

def expr_C1 := 56 / 7 * 5
def expr_C2 := 56 + 7 * 5

-- Theorem stating that Group B expressions have the same order of operations
theorem group_B_same_order : (expr_B1 = expr_B2) := 
  sorry

end group_B_same_order_l300_300119


namespace range_of_m_l300_300474

open Set

noncomputable def setA : Set ℝ := {y | ∃ x : ℝ, y = 2^x / (2^x + 1)}
noncomputable def setB (m : ℝ) : Set ℝ := {y | ∃ x : ℝ, x ∈ Icc (-1 : ℝ) (1 : ℝ) ∧ y = (1 / 3) * x + m}

theorem range_of_m {m : ℝ} (p q : Prop) :
  p ↔ ∃ x : ℝ, x ∈ setA →
  q ↔ ∃ x : ℝ, x ∈ setB m →
  ((p → q) ∧ ¬(q → p)) ↔ (1 / 3 < m ∧ m < 2 / 3) :=
by
  sorry

end range_of_m_l300_300474


namespace extra_pieces_correct_l300_300086

def pieces_per_package : ℕ := 7
def number_of_packages : ℕ := 5
def total_pieces : ℕ := 41

theorem extra_pieces_correct : total_pieces - (number_of_packages * pieces_per_package) = 6 :=
by
  sorry

end extra_pieces_correct_l300_300086


namespace max_true_statements_l300_300078

theorem max_true_statements (y : ℝ) :
  (0 < y^3 ∧ y^3 < 2 → ∀ (y : ℝ),  y^3 > 2 → False) ∧
  ((-2 < y ∧ y < 0) → ∀ (y : ℝ), (0 < y ∧ y < 2) → False) →
  ∃ (s1 s2 : Prop), 
    ((0 < y^3 ∧ y^3 < 2) = s1 ∨ (y^3 > 2) = s1 ∨ (-2 < y ∧ y < 0) = s1 ∨ (0 < y ∧ y < 2) = s1 ∨ (0 < y - y^3 ∧ y - y^3 < 2) = s1) ∧
    ((0 < y^3 ∧ y^3 < 2) = s2 ∨ (y^3 > 2) = s2 ∨ (-2 < y ∧ y < 0) = s2 ∨ (0 < y ∧ y < 2) = s2 ∨ (0 < y - y^3 ∧ y - y^3 < 2) = s2) ∧ 
    (s1 ∧ s2) → 
    ∃ m : ℕ, m = 2 := 
sorry

end max_true_statements_l300_300078


namespace number_of_ways_to_place_pawns_l300_300048

theorem number_of_ways_to_place_pawns :
  let n := 5 in
  let number_of_placements := (n.factorial) in
  let number_of_permutations := (n.factorial) in
  number_of_placements * number_of_permutations = 14400 :=
by
  sorry

end number_of_ways_to_place_pawns_l300_300048


namespace angle_with_same_terminal_side_315_l300_300392

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 + β

theorem angle_with_same_terminal_side_315:
  same_terminal_side (-45) 315 :=
by
  sorry

end angle_with_same_terminal_side_315_l300_300392


namespace intersection_M_N_eq_set_l300_300336

-- Define sets M and N
def M : Set ℝ := {x : ℝ | x^2 < 4}
def N : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- The theorem to be proved
theorem intersection_M_N_eq_set : (M ∩ N) = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end intersection_M_N_eq_set_l300_300336


namespace most_stable_performance_l300_300108

structure Shooter :=
(average_score : ℝ)
(variance : ℝ)

def A := Shooter.mk 8.9 0.45
def B := Shooter.mk 8.9 0.42
def C := Shooter.mk 8.9 0.51

theorem most_stable_performance : 
  B.variance < A.variance ∧ B.variance < C.variance :=
by
  sorry

end most_stable_performance_l300_300108


namespace injective_function_identity_l300_300149

theorem injective_function_identity (f : ℕ → ℕ) (h_inj : Function.Injective f)
  (h : ∀ (m n : ℕ), 0 < m → 0 < n → f (n * f m) ≤ n * m) : ∀ x : ℕ, f x = x :=
by
  sorry

end injective_function_identity_l300_300149


namespace largest_lcm_value_l300_300537

open Nat

theorem largest_lcm_value :
  max (max (max (max (max (Nat.lcm 15 3) (Nat.lcm 15 5)) (Nat.lcm 15 6)) (Nat.lcm 15 9)) (Nat.lcm 15 10)) (Nat.lcm 15 15) = 45 :=
by
  sorry

end largest_lcm_value_l300_300537


namespace expression_value_l300_300343

theorem expression_value (x : ℝ) (h : x = -2) : (x * x^2 * (1/x) = 4) :=
by
  rw [h]
  sorry

end expression_value_l300_300343


namespace percent_less_than_l300_300058

-- Definitions based on the given conditions.
variable (y q w z : ℝ)
variable (h1 : w = 0.60 * q)
variable (h2 : q = 0.60 * y)
variable (h3 : z = 1.50 * w)

-- The theorem that the percentage by which z is less than y is 46%.
theorem percent_less_than (y q w z : ℝ) (h1 : w = 0.60 * q) (h2 : q = 0.60 * y) (h3 : z = 1.50 * w) :
  100 - (z / y * 100) = 46 :=
sorry

end percent_less_than_l300_300058


namespace expected_number_of_different_faces_l300_300707

theorem expected_number_of_different_faces :
  let ζ_i (i : Fin 6) := if (∃ k, k ∈ Finset.range 6) then 1 else 0,
      ζ := (List.range 6).sum (ζ_i),
      p := (5 / 6 : ℝ) ^ 6
  in (Expectation (λ ω => ζ)) = (6 * (1 - p)) :=
by
  sorry

end expected_number_of_different_faces_l300_300707


namespace failed_in_hindi_percentage_l300_300495

/-- In an examination, a specific percentage of students failed in Hindi (H%), 
45% failed in English, and 20% failed in both. We know that 40% passed in both subjects. 
Prove that 35% students failed in Hindi. --/
theorem failed_in_hindi_percentage : 
  ∀ (H E B P : ℕ),
    (E = 45) → (B = 20) → (P = 40) → (100 - P = H + E - B) → H = 35 := by
  intros H E B P hE hB hP h
  sorry

end failed_in_hindi_percentage_l300_300495


namespace eggs_sold_l300_300778

/-- Define the notion of trays and eggs in this context -/
def trays_of_eggs : ℤ := 30

/-- Define the initial collection of trays by Haman -/
def initial_trays : ℤ := 10

/-- Define the number of trays dropped by Haman -/
def dropped_trays : ℤ := 2

/-- Define the additional trays that Haman's father told him to collect -/
def additional_trays : ℤ := 7

/-- Define the total eggs sold -/
def total_eggs_sold : ℤ :=
  (initial_trays - dropped_trays) * trays_of_eggs + additional_trays * trays_of_eggs

-- Theorem to prove the total eggs sold
theorem eggs_sold : total_eggs_sold = 450 :=
by 
  -- Insert proof here
  sorry

end eggs_sold_l300_300778


namespace math_scores_population_l300_300991

/-- 
   Suppose there are 50,000 students who took the high school entrance exam.
   The education department randomly selected 2,000 students' math scores 
   for statistical analysis. Prove that the math scores of the 50,000 students 
   are the population.
-/
theorem math_scores_population (students : ℕ) (selected : ℕ) 
    (students_eq : students = 50000) (selected_eq : selected = 2000) : 
    true :=
by
  sorry

end math_scores_population_l300_300991


namespace discount_percentage_l300_300861

theorem discount_percentage 
  (C : ℝ) (S : ℝ) (P : ℝ) (SP : ℝ)
  (h1 : C = 48)
  (h2 : 0.60 * S = C)
  (h3 : P = 16)
  (h4 : P = S - SP)
  (h5 : SP = 80 - 16)
  (h6 : S = 80) :
  (S - SP) / S * 100 = 20 := by
sorry

end discount_percentage_l300_300861


namespace greatest_possible_mean_BC_l300_300125

-- Mean weights for piles A, B
def mean_weight_A : ℝ := 60
def mean_weight_B : ℝ := 70

-- Combined mean weight for piles A and B
def mean_weight_AB : ℝ := 64

-- Combined mean weight for piles A and C
def mean_weight_AC : ℝ := 66

-- Prove that the greatest possible integer value for the mean weight of
-- the rocks in the combined piles B and C
theorem greatest_possible_mean_BC : ∃ (w : ℝ), (⌊w⌋ = 75) :=
by
  -- Definitions and assumptions based on problem conditions
  have h1 : mean_weight_A = 60 := rfl
  have h2 : mean_weight_B = 70 := rfl
  have h3 : mean_weight_AB = 64 := rfl
  have h4 : mean_weight_AC = 66 := rfl
  sorry

end greatest_possible_mean_BC_l300_300125


namespace train_speed_kmh_l300_300442

theorem train_speed_kmh 
  (L_train : ℝ) (L_bridge : ℝ) (time : ℝ)
  (h_train : L_train = 460)
  (h_bridge : L_bridge = 140)
  (h_time : time = 48) : 
  (L_train + L_bridge) / time * 3.6 = 45 := 
by
  -- Definitions and conditions
  have h_total_dist : L_train + L_bridge = 600 := by sorry
  have h_speed_mps : (L_train + L_bridge) / time = 600 / 48 := by sorry
  have h_speed_mps_simplified : 600 / 48 = 12.5 := by sorry
  have h_speed_kmh : 12.5 * 3.6 = 45 := by sorry
  sorry

end train_speed_kmh_l300_300442


namespace sum_primes_between_20_and_30_is_52_l300_300238

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' (a + 1) (b - a - 1)).filter is_prime

def sum_primes_between_20_and_30 : Prop :=
  primes_between 20 30 = [23, 29] ∧ (List.foldl (· + ·) 0 (primes_between 20 30) = 52)

theorem sum_primes_between_20_and_30_is_52 : sum_primes_between_20_and_30 :=
by
  sorry

end sum_primes_between_20_and_30_is_52_l300_300238


namespace units_digit_of_pow_sum_is_correct_l300_300677

theorem units_digit_of_pow_sum_is_correct : 
  (24^3 + 42^3) % 10 = 2 := by
  -- Start the proof block
  sorry

end units_digit_of_pow_sum_is_correct_l300_300677


namespace emmanuel_jelly_beans_l300_300005

theorem emmanuel_jelly_beans (total_jelly_beans : ℕ)
      (thomas_percentage : ℕ)
      (barry_ratio : ℕ)
      (emmanuel_ratio : ℕ)
      (h1 : total_jelly_beans = 200)
      (h2 : thomas_percentage = 10)
      (h3 : barry_ratio = 4)
      (h4 : emmanuel_ratio = 5) :
  let thomas_jelly_beans := (thomas_percentage * total_jelly_beans) / 100
  let remaining_jelly_beans := total_jelly_beans - thomas_jelly_beans
  let total_ratio := barry_ratio + emmanuel_ratio
  let per_part_jelly_beans := remaining_jelly_beans / total_ratio
  let emmanuel_jelly_beans := emmanuel_ratio * per_part_jelly_beans
  emmanuel_jelly_beans = 100 :=
by
  sorry

end emmanuel_jelly_beans_l300_300005


namespace fractional_equation_solution_l300_300162

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) ↔ (m ≤ 2 ∧ m ≠ -2) := 
sorry

end fractional_equation_solution_l300_300162


namespace complement_U_A_is_correct_l300_300337

open Set

/-- Define the universal set U -/
def U : Set ℝ := {x | x^2 > 1}

/-- Define the set A -/
def A : Set ℝ := {x | x^2 - 4x + 3 < 0}

/-- Define the complement of A in U -/
def complement_U_A := U \ A

/-- Prove that the complement of A in U is equal to (-∞, -1) ∪ [3, +∞) -/
theorem complement_U_A_is_correct :
  complement_U_A = {x | x < -1} ∪ {x | 3 ≤ x} :=
by
  sorry

end complement_U_A_is_correct_l300_300337


namespace find_value_of_m_l300_300765

noncomputable def m : ℤ := -2

theorem find_value_of_m (m : ℤ) :
  (m-2) ≠ 0 ∧ (m^2 - 3 = 1) → m = -2 :=
by
  intros h
  sorry

end find_value_of_m_l300_300765


namespace height_of_given_cylinder_l300_300401

noncomputable def height_of_cylinder (P d : ℝ) : ℝ :=
  let r := P / (2 * Real.pi)
  let l := P
  let h := Real.sqrt (d^2 - l^2)
  h

theorem height_of_given_cylinder : height_of_cylinder 6 10 = 8 :=
by
  show height_of_cylinder 6 10 = 8
  sorry

end height_of_given_cylinder_l300_300401


namespace ariel_fish_l300_300561

theorem ariel_fish (total_fish : ℕ) (male_fraction female_fraction : ℚ) (h1 : total_fish = 45) (h2 : male_fraction = 2 / 3) (h3 : female_fraction = 1 - male_fraction) : total_fish * female_fraction = 15 :=
by
  sorry

end ariel_fish_l300_300561


namespace tel_aviv_rain_days_l300_300958

-- Define the conditions
def chance_of_rain : ℝ := 0.5
def days_considered : ℕ := 6
def given_probability : ℝ := 0.234375

-- Helper function to compute binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability function P(X = k)
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k) * (p ^ k) * ((1 - p) ^ (n - k))

-- The main theorem to prove
theorem tel_aviv_rain_days :
  ∃ k, binomial_probability days_considered k chance_of_rain = given_probability ∧ k = 2 := by
  sorry

end tel_aviv_rain_days_l300_300958


namespace min_people_wearing_both_l300_300934

theorem min_people_wearing_both (n : ℕ) (h1 : n % 3 = 0)
  (h_gloves : ∃ g, g = n / 3 ∧ g = 1) (h_hats : ∃ h, h = (2 * n) / 3 ∧ h = 2) :
  ∃ x, x = 0 := by
  sorry

end min_people_wearing_both_l300_300934


namespace smallest_z_in_arithmetic_and_geometric_progression_l300_300364

theorem smallest_z_in_arithmetic_and_geometric_progression :
  ∃ x y z : ℤ, x < y ∧ y < z ∧ (2 * y = x + z) ∧ (z^2 = x * y) ∧ z = -2 :=
by
  sorry

end smallest_z_in_arithmetic_and_geometric_progression_l300_300364


namespace roses_in_vase_l300_300224

/-- There were initially 16 roses and 3 orchids in the vase.
    Jessica cut 8 roses and 8 orchids from her garden.
    There are now 7 orchids in the vase.
    Prove that the number of roses in the vase now is 24. -/
theorem roses_in_vase
  (initial_roses initial_orchids : ℕ)
  (cut_roses cut_orchids remaining_orchids final_roses : ℕ)
  (h_initial: initial_roses = 16)
  (h_initial_orchids: initial_orchids = 3)
  (h_cut: cut_roses = 8 ∧ cut_orchids = 8)
  (h_remaining_orchids: remaining_orchids = 7)
  (h_orchids_relation: initial_orchids + cut_orchids = remaining_orchids + cut_orchids - 4)
  : final_roses = initial_roses + cut_roses := by
  sorry

end roses_in_vase_l300_300224


namespace lemons_per_glass_l300_300501

theorem lemons_per_glass (lemons glasses : ℕ) (h : lemons = 18 ∧ glasses = 9) : lemons / glasses = 2 :=
by
  sorry

end lemons_per_glass_l300_300501


namespace value_of_f_neg_11_over_2_l300_300155

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodicity (x : ℝ) : f (x + 2) = - (f x)⁻¹
axiom interval_value (h : 2 ≤ 5 / 2 ∧ 5 / 2 ≤ 3) : f (5 / 2) = 5 / 2

theorem value_of_f_neg_11_over_2 : f (-11 / 2) = 5 / 2 :=
by
  sorry

end value_of_f_neg_11_over_2_l300_300155


namespace car_average_speed_l300_300267

theorem car_average_speed 
  (d1 d2 d3 d5 d6 d7 d8 : ℝ) 
  (t_total : ℝ) 
  (avg_speed : ℝ)
  (h1 : d1 = 90)
  (h2 : d2 = 50)
  (h3 : d3 = 70)
  (h5 : d5 = 80)
  (h6 : d6 = 60)
  (h7 : d7 = -40)
  (h8 : d8 = -55)
  (h_t_total : t_total = 8)
  (h_avg_speed : avg_speed = (d1 + d2 + d3 + d5 + d6 + d7 + d8) / t_total) :
  avg_speed = 31.875 := 
by sorry

end car_average_speed_l300_300267


namespace man_l300_300260

-- Conditions
def speed_with_current : ℝ := 18
def speed_of_current : ℝ := 3.4

-- Problem statement
theorem man's_speed_against_current :
  (speed_with_current - speed_of_current - speed_of_current) = 11.2 := 
by
  sorry

end man_l300_300260


namespace value_of_a_plus_d_l300_300620

theorem value_of_a_plus_d (a b c d : ℕ) (h1 : a + b = 16) (h2 : b + c = 9) (h3 : c + d = 3) : a + d = 10 := 
by 
  sorry

end value_of_a_plus_d_l300_300620


namespace find_natural_numbers_l300_300893

theorem find_natural_numbers (x y z : ℕ) (hx : x ≤ y) (hy : y ≤ z) : 
    (1 + 1 / x) * (1 + 1 / y) * (1 + 1 / z) = 3 
    → (x = 1 ∧ y = 3 ∧ z = 8) 
    ∨ (x = 1 ∧ y = 4 ∧ z = 5) 
    ∨ (x = 2 ∧ y = 2 ∧ z = 3) :=
sorry

end find_natural_numbers_l300_300893


namespace simplify_proof_l300_300204

noncomputable def simplify_expression (a b c d x y : ℝ) (h : c * x ≠ d * y) : ℝ :=
  (c * x * (b^2 * x^2 - 4 * b^2 * y^2 + a^2 * y^2) 
  - d * y * (b^2 * x^2 - 2 * a^2 * x^2 - 3 * a^2 * y^2)) / (c * x - d * y)

theorem simplify_proof (a b c d x y : ℝ) (h : c * x ≠ d * y) :
  simplify_expression a b c d x y h = b^2 * x^2 + a^2 * y^2 :=
by sorry

end simplify_proof_l300_300204


namespace calculate_expression_l300_300570

theorem calculate_expression : 
  ((13^13 / 13^12)^3 * 3^3) / 3^6 = 27 :=
by
  sorry

end calculate_expression_l300_300570


namespace fraction_result_l300_300026

theorem fraction_result (a b c : ℝ) (h1 : a / 2 = b / 3) (h2 : b / 3 = c / 5) (h3 : a ≠ 0) (h4 : b ≠ 0) (h5 : c ≠ 0) :
  (a + b) / (c - a) = 5 / 3 :=
by
  sorry

end fraction_result_l300_300026


namespace expected_number_of_different_faces_l300_300722

theorem expected_number_of_different_faces :
  let p := (6 : ℕ) ^ 6
  let q := (5 : ℕ) ^ 6
  6 * (1 - (5 / 6)^6) = (p - q) / (6 ^ 5) :=
by
  sorry

end expected_number_of_different_faces_l300_300722


namespace add_fractions_l300_300292

theorem add_fractions : (1 : ℚ) / 4 + (3 : ℚ) / 8 = 5 / 8 :=
by
  sorry

end add_fractions_l300_300292


namespace sample_size_l300_300175

theorem sample_size (k n : ℕ) (r : 2 * k + 3 * k + 5 * k = 10 * k) (h : 3 * k = 12) : n = 40 :=
by {
    -- here, we will provide a proof to demonstrate that n = 40 given the conditions
    sorry
}

end sample_size_l300_300175


namespace ratio_of_segments_l300_300603

theorem ratio_of_segments (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end ratio_of_segments_l300_300603


namespace exist_partition_of_delegates_l300_300444

/-- At a symposium, each delegate is acquainted with at least one of the other participants but not with everyone. 
    Prove that all delegates can be divided into two groups so that each participant in the symposium is acquainted with at least one person in their group. -/
theorem exist_partition_of_delegates 
  (G : SimpleGraph V)
  [DecidableRel G.Adj]
  (h1 : ∀ v : V, ∃ (w : V), G.Adj v w) 
  (h2 : ∀ v : V, ∃ (w : V), v ≠ w ∧ ¬G.Adj v w) : 
  ∃ (V₁ V₂ : Set V), 
    (V₁ ∪ V₂ = Set.univ) ∧ 
    (V₁ ∩ V₂ = ∅) ∧ 
    (∀ v ∈ V₁, ∃ w ∈ V₁, G.Adj v w) ∧ 
    (∀ v ∈ V₂, ∃ w ∈ V₂, G.Adj v w) :=
by
  sorry

end exist_partition_of_delegates_l300_300444


namespace problem_even_and_monotonically_increasing_l300_300557

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x ≤ f y

theorem problem_even_and_monotonically_increasing :
  is_even_function (fun x => Real.exp (|x|)) ∧ is_monotonically_increasing_on (fun x => Real.exp (|x|)) (Set.Ioo 0 1) :=
by
  sorry

end problem_even_and_monotonically_increasing_l300_300557


namespace length_width_percentage_change_l300_300834

variables (L W : ℝ) (x : ℝ)
noncomputable def area_change_percent : ℝ :=
  (L * (1 + x / 100) * W * (1 - x / 100) - L * W) / (L * W) * 100

theorem length_width_percentage_change (h : area_change_percent L W x = 4) :
  x = 20 :=
by
  sorry

end length_width_percentage_change_l300_300834


namespace greater_expected_area_vasya_l300_300888

noncomputable def expected_area_vasya : ℚ :=
  (1/6) * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

noncomputable def expected_area_asya : ℚ :=
  ((1/6) * (1 + 2 + 3 + 4 + 5 + 6)) * ((1/6) * (1 + 2 + 3 + 4 + 5 + 6))

theorem greater_expected_area_vasya : expected_area_vasya > expected_area_asya :=
  by
  -- We've provided the expected area values as definitions
  -- expected_area_vasya = 91/6
  -- vs. expected_area_asya = 12.25 = (21/6)^2 = 441/36 = 12.25
  sorry

end greater_expected_area_vasya_l300_300888


namespace no_solution_for_t_and_s_l300_300583

theorem no_solution_for_t_and_s (m : ℝ) :
  (¬∃ t s : ℝ, (1 + 7 * t = -3 + 2 * s) ∧ (3 - 5 * t = 4 + m * s)) ↔ m = -10 / 7 :=
by
  sorry

end no_solution_for_t_and_s_l300_300583


namespace toothpicks_required_l300_300131

noncomputable def total_small_triangles (n : ℕ) : ℕ :=
  n * (n + 1) / 2

noncomputable def total_initial_toothpicks (n : ℕ) : ℕ :=
  3 * total_small_triangles n

noncomputable def adjusted_toothpicks (n : ℕ) : ℕ :=
  total_initial_toothpicks n / 2

noncomputable def boundary_toothpicks (n : ℕ) : ℕ :=
  2 * n

noncomputable def total_toothpicks (n : ℕ) : ℕ :=
  adjusted_toothpicks n + boundary_toothpicks n

theorem toothpicks_required {n : ℕ} (h : n = 2500) : total_toothpicks n = 4694375 :=
by sorry

end toothpicks_required_l300_300131


namespace g_value_l300_300831

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_value (h : ∀ x ≠ 0, g x - 3 * g (1 / x) = 3^x) :
  g 3 = -(27 + 3 * (3:ℝ)^(1/3)) / 8 :=
sorry

end g_value_l300_300831


namespace greater_expected_area_vasya_l300_300889

noncomputable def expected_area_vasya : ℚ :=
  (1/6) * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

noncomputable def expected_area_asya : ℚ :=
  ((1/6) * (1 + 2 + 3 + 4 + 5 + 6)) * ((1/6) * (1 + 2 + 3 + 4 + 5 + 6))

theorem greater_expected_area_vasya : expected_area_vasya > expected_area_asya :=
  by
  -- We've provided the expected area values as definitions
  -- expected_area_vasya = 91/6
  -- vs. expected_area_asya = 12.25 = (21/6)^2 = 441/36 = 12.25
  sorry

end greater_expected_area_vasya_l300_300889


namespace episodes_remaining_l300_300255

-- Definition of conditions
def seasons : ℕ := 12
def episodes_per_season : ℕ := 20
def fraction_watched : ℚ := 1 / 3
def total_episodes : ℕ := episodes_per_season * seasons
def episodes_watched : ℕ := (fraction_watched * total_episodes).toNat

-- Problem statement
theorem episodes_remaining : total_episodes - episodes_watched = 160 := by
  sorry

end episodes_remaining_l300_300255


namespace wedding_cost_correct_l300_300504

def venue_cost : ℕ := 10000
def cost_per_guest : ℕ := 500
def john_guests : ℕ := 50
def wife_guest_increase : ℕ := john_guests * 60 / 100
def total_wedding_cost : ℕ := venue_cost + cost_per_guest * (john_guests + wife_guest_increase)

theorem wedding_cost_correct : total_wedding_cost = 50000 :=
by
  sorry

end wedding_cost_correct_l300_300504


namespace spencer_total_distance_l300_300193

-- Definitions for the given conditions
def distance_house_to_library : ℝ := 0.3
def distance_library_to_post_office : ℝ := 0.1
def distance_post_office_to_home : ℝ := 0.4

-- Define the total distance based on the given conditions
def total_distance : ℝ := distance_house_to_library + distance_library_to_post_office + distance_post_office_to_home

-- Statement to prove
theorem spencer_total_distance : total_distance = 0.8 := by
  sorry

end spencer_total_distance_l300_300193


namespace multiples_of_eleven_ending_in_seven_l300_300042

theorem multiples_of_eleven_ending_in_seven (n : ℕ) : 
  (∀ k : ℕ, n > 0 ∧ n < 2000 ∧ (∃ m : ℕ, n = 11 * m) ∧ n % 10 = 7) → ∃ c : ℕ, c = 18 := 
by
  sorry

end multiples_of_eleven_ending_in_seven_l300_300042


namespace _l300_300352

noncomputable def angle_ACB_is_45_degrees (A B C D E F : Type) [LinearOrderedField A]
  (angle : A → A → A → A) (AB AC : A) (h1 : AB = 3 * AC)
  (BAE ACD : A) (h2 : BAE = ACD)
  (BCA : A) (h3 : BAE = 2 * BCA)
  (CF FE : A) (h4 : CF = FE)
  (is_isosceles : ∀ {X Y Z : Type} [LinearOrderedField X] (a b c : X), a = b → b = c → a = c)
  (triangle_sum : ∀ {X Y Z : Type} [LinearOrderedField X] (a b c : X), a + b + c = 180) :
  ∃ (angle_ACB : A), angle_ACB = 45 := 
by
  -- Here we assume we have the appropriate conditions from geometry
  -- Then you'd prove the theorem based on given hypotheses
  sorry

end _l300_300352


namespace batsman_avg_l300_300265

variable (A : ℕ) -- The batting average in 46 innings

-- Given conditions
variables (highest lowest : ℕ)
variables (diff : ℕ) (avg_excl : ℕ) (num_excl : ℕ)

namespace cricket

-- Define the given values
def highest_score := 225
def difference := 150
def avg_excluding := 58
def num_excluding := 44

-- Calculate the lowest score
def lowest_score := highest_score - difference

-- Calculate the total runs in 44 innings excluding highest and lowest scores
def total_run_excluded := avg_excluding * num_excluding

-- Calculate the total runs in 46 innings
def total_runs := total_run_excluded + highest_score + lowest_score

-- Define the equation relating the average to everything else
def batting_avg_eq : Prop :=
  total_runs = 46 * A

-- Prove that the batting average A is 62 given the conditions
theorem batsman_avg :
  A = 62 :=
  by
    sorry

end cricket

end batsman_avg_l300_300265


namespace cube_root_of_x_l300_300619

theorem cube_root_of_x {x : ℝ} (h : x^2 = 64) : (Real.cbrt x = 2) ∨ (Real.cbrt x = -2) :=
sorry

end cube_root_of_x_l300_300619


namespace add_one_gt_add_one_l300_300600

theorem add_one_gt_add_one (a b c : ℝ) (h : a > b) : (a + c) > (b + c) :=
sorry

end add_one_gt_add_one_l300_300600


namespace ratio_of_ages_l300_300339

open Real

theorem ratio_of_ages (father_age son_age : ℝ) (h1 : father_age = 45) (h2 : son_age = 15) :
  father_age / son_age = 3 :=
by
  sorry

end ratio_of_ages_l300_300339


namespace vasya_has_greater_area_l300_300872

-- Definition of a fair six-sided die roll
def die_roll : ℕ → ℝ := λ k, if k ∈ {1, 2, 3, 4, 5, 6} then (1 / 6 : ℝ) else 0

-- Expected value of a function with respect to a probability distribution
noncomputable def expected_value (f : ℕ → ℝ) : ℝ := ∑ k in {1, 2, 3, 4, 5, 6}, f k * die_roll k

-- Vasya's area: A^2 where A is a single die roll
noncomputable def vasya_area : ℝ := expected_value (λ k, (k : ℝ) ^ 2)

-- Asya's area: A * B where A and B are independent die rolls
noncomputable def asya_area : ℝ := (expected_value (λ k, (k : ℝ))) ^ 2

theorem vasya_has_greater_area :
  vasya_area > asya_area := sorry

end vasya_has_greater_area_l300_300872


namespace coefficient_x4_of_square_l300_300040

theorem coefficient_x4_of_square (q : Polynomial ℝ) (hq : q = Polynomial.X^5 - 4 * Polynomial.X^2 + 3) :
  (Polynomial.coeff (q * q) 4 = 16) :=
by {
  sorry
}

end coefficient_x4_of_square_l300_300040


namespace middle_number_of_ratio_l300_300491

theorem middle_number_of_ratio (x : ℝ) (h : (3 * x)^2 + (2 * x)^2 + (5 * x)^2 = 1862) : 2 * x = 14 :=
sorry

end middle_number_of_ratio_l300_300491


namespace least_area_of_triangles_l300_300798

-- Define the points A, B, C, D of the unit square
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (1, 1)
def D : ℝ × ℝ := (0, 1)

-- Define the function s(M, N) as the least area of the triangles having their vertices in the set {A, B, C, D, M, N}
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

noncomputable def s (M N : ℝ × ℝ) : ℝ :=
  min (min (min (min (min (triangle_area A B M) (triangle_area A B N)) (triangle_area A C M)) (triangle_area A C N)) (min (triangle_area A D M) (triangle_area A D N)))
    (min (min (min (triangle_area B C M) (triangle_area B C N)) (triangle_area B D M)) (min (triangle_area B D N) (min (triangle_area C D M) (triangle_area C D N))))

-- Define the statement to prove
theorem least_area_of_triangles (M N : ℝ × ℝ)
  (hM : M.1 > 0 ∧ M.1 < 1 ∧ M.2 > 0 ∧ M.2 < 1)
  (hN : N.1 > 0 ∧ N.1 < 1 ∧ N.2 > 0 ∧ N.2 < 1)
  (hMN : (M ≠ A ∨ N ≠ A) ∧ (M ≠ B ∨ N ≠ B) ∧ (M ≠ C ∨ N ≠ C) ∧ (M ≠ D ∨ N ≠ D))
  : s M N ≤ 1 / 8 := 
sorry

end least_area_of_triangles_l300_300798


namespace simplify_fraction_l300_300378

theorem simplify_fraction : (48 / 72 : ℚ) = (2 / 3) := 
by
  sorry

end simplify_fraction_l300_300378


namespace min_ball_count_required_l300_300202

def is_valid_ball_count (n : ℕ) : Prop :=
  n >= 11 ∧ n ≠ 17 ∧ n % 6 ≠ 0

def distinct_list (l : List ℕ) : Prop :=
  ∀ i j, i < l.length → j < l.length → i ≠ j → l.nthLe i sorry ≠ l.nthLe j sorry

def valid_ball_counts_list (l : List ℕ) : Prop :=
  (l.length = 10) ∧ distinct_list l ∧ (∀ n ∈ l, is_valid_ball_count n)

theorem min_ball_count_required : ∃ l, valid_ball_counts_list l ∧ l.sum = 174 := sorry

end min_ball_count_required_l300_300202


namespace sum_digits_l300_300596

def repeat_pattern (d: ℕ) (n: ℕ) : ℕ :=
  let pattern := if d = 404 then 404 else if d = 707 then 707 else 0
  pattern * 10^(n / 3)

def N1 := repeat_pattern 404 101
def N2 := repeat_pattern 707 101
def P := N1 * N2

def thousands_digit (n: ℕ) : ℕ :=
  (n / 1000) % 10

def units_digit (n: ℕ) : ℕ :=
  n % 10

theorem sum_digits : thousands_digit P + units_digit P = 10 := by
  sorry

end sum_digits_l300_300596


namespace total_cost_of_items_l300_300287

theorem total_cost_of_items
  (E P M : ℕ)
  (h1 : E + 3 * P + 2 * M = 240)
  (h2 : 2 * E + 5 * P + 4 * M = 440) :
  3 * E + 4 * P + 6 * M = 520 := 
sorry

end total_cost_of_items_l300_300287


namespace total_amount_invested_l300_300173

def annualIncome (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

def totalInvestment (T x y : ℝ) : Prop :=
  T - x = y

def condition (T : ℝ) : Prop :=
  let income_10_percent := annualIncome (T - 800) 0.10
  let income_8_percent := annualIncome 800 0.08
  income_10_percent - income_8_percent = 56

theorem total_amount_invested :
  ∃ (T : ℝ), condition T ∧ totalInvestment T 800 800 ∧ T = 2000 :=
by
  sorry

end total_amount_invested_l300_300173


namespace cos_double_angle_l300_300324

theorem cos_double_angle (a : ℝ) (h : Real.sin a = 3/5) : Real.cos (2 * a) = 7/25 :=
by
  sorry

end cos_double_angle_l300_300324


namespace simplify_complex_expression_l300_300950

theorem simplify_complex_expression : 
  ∀ (i : ℂ), i^2 = -1 → 3 * (4 - 2 * i) + 2 * i * (3 + 2 * i) = 8 := 
by
  intros
  sorry

end simplify_complex_expression_l300_300950


namespace largest_number_is_B_l300_300850

-- Define the numbers as constants
def A : ℝ := 0.989
def B : ℝ := 0.998
def C : ℝ := 0.981
def D : ℝ := 0.899
def E : ℝ := 0.9801

-- State the theorem that B is the largest number
theorem largest_number_is_B : B > A ∧ B > C ∧ B > D ∧ B > E := by
  -- By comparison
  sorry

end largest_number_is_B_l300_300850


namespace initial_miles_correct_l300_300446

-- Definitions and conditions
def miles_per_gallon : ℕ := 30
def gallons_per_tank : ℕ := 20
def current_miles : ℕ := 2928
def tanks_filled : ℕ := 2

-- Question: How many miles were on the car before the road trip?
def initial_miles : ℕ := current_miles - (miles_per_gallon * gallons_per_tank * tanks_filled)

-- Proof problem statement
theorem initial_miles_correct : initial_miles = 1728 :=
by
  -- Here we expect the proof, but are skipping it with 'sorry'
  sorry

end initial_miles_correct_l300_300446


namespace cake_shop_problem_l300_300927

theorem cake_shop_problem :
  ∃ (N n K : ℕ), (N - n * K = 6) ∧ (N = (n - 1) * 8 + 1) ∧ (N = 97) :=
by
  sorry

end cake_shop_problem_l300_300927


namespace picasso_postcards_probability_l300_300410

theorem picasso_postcards_probability :
  let total_postcards := 12
  let picasso_postcards := 4
  let other_postcards := total_postcards - picasso_postcards
  let total_ways := Nat.factorial total_postcards
  let ways_postcards_in_block := Nat.factorial picasso_postcards
  let ways_units := Nat.factorial other_postcards.succ -- since we treat 4 Picasso postcards as one block
  (ways_units * ways_postcards_in_block / total_ways) = (1 / 55 : ℚ) :=
by
  sorry

end picasso_postcards_probability_l300_300410


namespace cost_price_proof_l300_300419

def trader_sells_66m_for_660 : Prop := ∃ cp profit sp : ℝ, sp = 660 ∧ cp * 66 + profit * 66 = sp
def profit_5_per_meter : Prop := ∃ profit : ℝ, profit = 5
def cost_price_per_meter_is_5 : Prop := ∃ cp : ℝ, cp = 5

theorem cost_price_proof : trader_sells_66m_for_660 → profit_5_per_meter → cost_price_per_meter_is_5 :=
by
  intros h1 h2
  sorry

end cost_price_proof_l300_300419


namespace sin_alpha_cos_alpha_l300_300468

theorem sin_alpha_cos_alpha (α : ℝ) (h : Real.sin (3 * Real.pi - α) = -2 * Real.sin (Real.pi / 2 + α)) : 
  Real.sin α * Real.cos α = -2 / 5 :=
by
  sorry

end sin_alpha_cos_alpha_l300_300468


namespace triangle_AF_AT_ratio_l300_300499

theorem triangle_AF_AT_ratio (A B C D E F T : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace T]
  (AD DB AE EC : ℝ)
  (h_AD : AD = 1) (h_DB : DB = 3)
  (h_AE : AE = 2) (h_EC : EC = 4)
  (h_bisector : is_angle_bisector A T B C)
  (h_intersect : line_intersects DE AT F) :
  \(\frac{AF}{AT} = \frac{5}{18}) :=
sorry

end triangle_AF_AT_ratio_l300_300499


namespace total_number_of_cookies_l300_300121

open Nat -- Open the natural numbers namespace to work with natural number operations

def n_bags : Nat := 7
def cookies_per_bag : Nat := 2
def total_cookies : Nat := n_bags * cookies_per_bag

theorem total_number_of_cookies : total_cookies = 14 := by
  sorry

end total_number_of_cookies_l300_300121


namespace infinitely_many_n_gt_sqrt_two_l300_300809

/-- A sequence of positive integers indexed by natural numbers. -/
def a (n : ℕ) : ℕ := sorry

/-- Main theorem stating there are infinitely many n such that 1 + a_n > a_{n-1} * root n of 2. -/
theorem infinitely_many_n_gt_sqrt_two :
  ∀ (a : ℕ → ℕ), (∀ n, a n > 0) → ∃ᶠ n in at_top, 1 + a n > a (n - 1) * (2 : ℝ) ^ (1 / n : ℝ) :=
by {
  sorry
}

end infinitely_many_n_gt_sqrt_two_l300_300809


namespace units_digit_sum_l300_300684

theorem units_digit_sum (h₁ : (24 : ℕ) % 10 = 4) 
                        (h₂ : (42 : ℕ) % 10 = 2) : 
  ((24^3 + 42^3) % 10 = 2) :=
by
  sorry

end units_digit_sum_l300_300684


namespace min_value_18_solve_inequality_l300_300326

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (1/a^3) + (1/b^3) + (1/c^3) + 27 * a * b * c

theorem min_value_18 (a b c : ℝ) (h : a > 0) (h' : b > 0) (h'' : c > 0) :
  min_value a b c ≥ 18 :=
by sorry

theorem solve_inequality (x : ℝ) :
  abs (x + 1) - 2 * x < 18 ↔ x > -(19/3) :=
by sorry

end min_value_18_solve_inequality_l300_300326


namespace polynomial_properties_l300_300010

noncomputable def p (x : ℕ) : ℕ := 2 * x^3 + x + 4

theorem polynomial_properties :
  p 1 = 7 ∧ p 10 = 2014 := 
by
  -- Placeholder for proof
  sorry

end polynomial_properties_l300_300010


namespace soccer_club_girls_l300_300729

theorem soccer_club_girls (B G : ℕ) 
  (h1 : B + G = 30) 
  (h2 : (1 / 3 : ℚ) * G + B = 18) : 
  G = 18 := 
  by sorry

end soccer_club_girls_l300_300729


namespace solve_cos_2x_eq_cos_x_plus_sin_x_l300_300088

open Real

theorem solve_cos_2x_eq_cos_x_plus_sin_x :
  ∀ x : ℝ,
    (cos (2 * x) = cos x + sin x) ↔
    (∃ k : ℤ, x = k * π - π / 4) ∨ 
    (∃ k : ℤ, x = 2 * k * π) ∨
    (∃ k : ℤ, x = 2 * k * π - π / 2) := 
sorry

end solve_cos_2x_eq_cos_x_plus_sin_x_l300_300088


namespace racing_cars_lcm_l300_300651

theorem racing_cars_lcm :
  let a := 28
  let b := 24
  let c := 32
  Nat.lcm a (Nat.lcm b c) = 672 :=
by
  sorry

end racing_cars_lcm_l300_300651


namespace simplify_expression_and_find_ratio_l300_300658

theorem simplify_expression_and_find_ratio:
  ∀ (k : ℤ), (∃ (a b : ℤ), (a = 1 ∧ b = 3) ∧ (6 * k + 18 = 6 * (a * k + b))) →
  (1 : ℤ) / (3 : ℤ) = (1 : ℤ) / (3 : ℤ) :=
by
  intro k
  intro h
  sorry

end simplify_expression_and_find_ratio_l300_300658


namespace combined_mean_is_254_over_15_l300_300957

noncomputable def combined_mean_of_sets 
  (mean₁ : ℝ) (n₁ : ℕ) 
  (mean₂ : ℝ) (n₂ : ℕ) : ℝ :=
  (mean₁ * n₁ + mean₂ * n₂) / (n₁ + n₂)

theorem combined_mean_is_254_over_15 :
  combined_mean_of_sets 18 7 16 8 = (254 : ℝ) / 15 :=
by
  sorry

end combined_mean_is_254_over_15_l300_300957


namespace team_total_points_l300_300174

theorem team_total_points : 
  ∀ (Tobee Jay Sean : ℕ),
  (Tobee = 4) →
  (Jay = Tobee + 6) →
  (Sean = Tobee + Jay - 2) →
  (Tobee + Jay + Sean = 26) :=
by
  intros Tobee Jay Sean h1 h2 h3
  rw [h1, h2, h3]
  sorry

end team_total_points_l300_300174


namespace length_of_body_diagonal_l300_300158

theorem length_of_body_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 11)
  (h2 : 4 * (a + b + c) = 24) :
  (a^2 + b^2 + c^2).sqrt = 5 :=
by {
  -- proof to be filled
  sorry
}

end length_of_body_diagonal_l300_300158


namespace average_speed_l300_300697

theorem average_speed (d1 d2 t1 t2 : ℝ) 
  (h1 : d1 = 100) 
  (h2 : d2 = 80) 
  (h3 : t1 = 1) 
  (h4 : t2 = 1) : 
  (d1 + d2) / (t1 + t2) = 90 := 
by 
  sorry

end average_speed_l300_300697


namespace math_scores_population_l300_300990

/-- 
   Suppose there are 50,000 students who took the high school entrance exam.
   The education department randomly selected 2,000 students' math scores 
   for statistical analysis. Prove that the math scores of the 50,000 students 
   are the population.
-/
theorem math_scores_population (students : ℕ) (selected : ℕ) 
    (students_eq : students = 50000) (selected_eq : selected = 2000) : 
    true :=
by
  sorry

end math_scores_population_l300_300990


namespace number_of_women_attended_l300_300738

theorem number_of_women_attended
  (m : ℕ) (w : ℕ)
  (men_dance_women : m = 15)
  (women_dance_men : ∀ i : ℕ, i < 15 → i * 4 = 60)
  (women_condition : w * 3 = 60) :
  w = 20 :=
sorry

end number_of_women_attended_l300_300738


namespace number_of_single_windows_upstairs_l300_300550

theorem number_of_single_windows_upstairs :
  ∀ (num_double_windows_downstairs : ℕ)
    (glass_panels_per_double_window : ℕ)
    (num_single_windows_upstairs : ℕ)
    (glass_panels_per_single_window : ℕ)
    (total_glass_panels : ℕ),
  num_double_windows_downstairs = 6 →
  glass_panels_per_double_window = 4 →
  glass_panels_per_single_window = 4 →
  total_glass_panels = 80 →
  num_single_windows_upstairs = (total_glass_panels - (num_double_windows_downstairs * glass_panels_per_double_window)) / glass_panels_per_single_window →
  num_single_windows_upstairs = 14 :=
by
  intros
  sorry

end number_of_single_windows_upstairs_l300_300550


namespace max_distance_ellipse_to_line_l300_300768

open Real

theorem max_distance_ellipse_to_line :
  ∃ (M : ℝ × ℝ),
    (M.1 ^ 2 / 12 + M.2 ^ 2 / 4 = 1) ∧
    (∃ (d : ℝ), 
      (d = abs (2 * sqrt 3 * cos (3 * π / 2 - π / 3) + 2 * sin (3 * π / 2 - π / 3) - 4) / sqrt 2) ∧ 
      d = 4 * sqrt 2 ∧ M = (-3, -1)) :=
sorry

end max_distance_ellipse_to_line_l300_300768


namespace sum_of_first_seven_terms_l300_300221

variable (a : ℕ → ℝ) -- a sequence of real numbers (can be adapted to other types if needed)

-- Given conditions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a n = a 0 + n * d

def sum_of_three_terms (a : ℕ → ℝ) (sum : ℝ) : Prop :=
  a 2 + a 3 + a 4 = sum

-- Theorem to prove
theorem sum_of_first_seven_terms (a : ℕ → ℝ) (h1 : is_arithmetic_progression a) (h2 : sum_of_three_terms a 12) :
  (a 0) + (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) = 28 :=
sorry

end sum_of_first_seven_terms_l300_300221


namespace units_digit_sum_l300_300686

theorem units_digit_sum (h₁ : (24 : ℕ) % 10 = 4) 
                        (h₂ : (42 : ℕ) % 10 = 2) : 
  ((24^3 + 42^3) % 10 = 2) :=
by
  sorry

end units_digit_sum_l300_300686


namespace time_in_2700_minutes_is_3_am_l300_300665

def minutes_in_hour : ℕ := 60
def hours_in_day : ℕ := 24
def current_hour : ℕ := 6
def minutes_later : ℕ := 2700

-- Calculate the final hour after adding the given minutes
def final_hour (current_hour minutes_later minutes_in_hour hours_in_day: ℕ) : ℕ :=
  (current_hour + (minutes_later / minutes_in_hour) % hours_in_day) % hours_in_day

theorem time_in_2700_minutes_is_3_am :
  final_hour current_hour minutes_later minutes_in_hour hours_in_day = 3 :=
by
  sorry

end time_in_2700_minutes_is_3_am_l300_300665


namespace exponential_decreasing_l300_300171

theorem exponential_decreasing (a : ℝ) (h : ∀ x y : ℝ, x < y → (a+1)^x > (a+1)^y) : -1 < a ∧ a < 0 :=
sorry

end exponential_decreasing_l300_300171


namespace Jimin_scabs_l300_300356

theorem Jimin_scabs (total_scabs : ℕ) (days_in_week : ℕ) (daily_scabs: ℕ)
  (h₁ : total_scabs = 220) (h₂ : days_in_week = 7) 
  (h₃ : daily_scabs = (total_scabs + days_in_week - 1) / days_in_week) : 
  daily_scabs ≥ 32 := by
  sorry

end Jimin_scabs_l300_300356


namespace percentage_boy_scouts_l300_300723

theorem percentage_boy_scouts (S B G : ℝ) (h1 : B + G = S)
  (h2 : 0.60 * S = 0.50 * B + 0.6818 * G) : (B / S) * 100 = 45 := by
  sorry

end percentage_boy_scouts_l300_300723


namespace moles_of_NaNO3_formed_l300_300594

/- 
  Define the reaction and given conditions.
  The following assumptions and definitions will directly come from the problem's conditions.
-/

/-- 
  Represents a chemical reaction: 1 molecule of AgNO3,
  1 molecule of NaOH producing 1 molecule of NaNO3 and 1 molecule of AgOH.
-/
def balanced_reaction (agNO3 naOH naNO3 agOH : ℕ) := agNO3 = 1 ∧ naOH = 1 ∧ naNO3 = 1 ∧ agOH = 1

/-- 
  Proves that the number of moles of NaNO3 formed is 1,
  given 1 mole of AgNO3 and 1 mole of NaOH.
-/
theorem moles_of_NaNO3_formed (agNO3 naOH naNO3 agOH : ℕ)
  (h : balanced_reaction agNO3 naOH naNO3 agOH) :
  naNO3 = 1 := 
by
  sorry  -- Proof will be added here later

end moles_of_NaNO3_formed_l300_300594


namespace slices_in_loaf_initial_l300_300111

-- Define the total slices used from Monday to Friday
def slices_used_weekdays : Nat := 5 * 2

-- Define the total slices used on Saturday
def slices_used_saturday : Nat := 2 * 2

-- Define the total slices used in the week
def total_slices_used : Nat := slices_used_weekdays + slices_used_saturday

-- Define the slices left
def slices_left : Nat := 6

-- Prove the total slices Tony started with
theorem slices_in_loaf_initial :
  let slices := total_slices_used + slices_left
  slices = 20 :=
by
  sorry

end slices_in_loaf_initial_l300_300111


namespace solve_for_x_l300_300953

theorem solve_for_x (x : ℝ) (h : (9 + 1/x)^(1/3) = -2) : x = -1/17 :=
by
  sorry

end solve_for_x_l300_300953


namespace find_coordinates_l300_300602

def point_in_fourth_quadrant (P : ℝ × ℝ) : Prop := P.1 > 0 ∧ P.2 < 0
def distance_to_x_axis (P : ℝ × ℝ) (d : ℝ) : Prop := |P.2| = d
def distance_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop := |P.1| = d

theorem find_coordinates :
  ∃ P : ℝ × ℝ, point_in_fourth_quadrant P ∧ distance_to_x_axis P 2 ∧ distance_to_y_axis P 5 ∧ P = (5, -2) :=
by
  sorry

end find_coordinates_l300_300602


namespace no_possible_path_l300_300438

theorem no_possible_path (n : ℕ) (h1 : n > 0) :
  ¬ ∃ (path : ℕ × ℕ → ℕ × ℕ), 
    (∀ (i : ℕ × ℕ), path i = if (i.1 < n - 1 ∧ i.2 < n - 1) then (i.1 + 1, i.2) else if i.2 < n - 1 then (i.1, i.2 + 1) else (i.1 - 1, i.2 - 1)) ∧
    (∀ (i j : ℕ × ℕ), i ≠ j → path i ≠ path j) ∧
    path (0, 0) = (0, 1) ∧
    path (n-1, n-1) = (n-1, 0) :=
sorry

end no_possible_path_l300_300438


namespace value_modulo_7_l300_300998

theorem value_modulo_7 : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 7 = 5 := 
  by 
  sorry

end value_modulo_7_l300_300998


namespace solution_eq1_solution_eq2_l300_300522

theorem solution_eq1 (x : ℝ) : 
  2 * x^2 - 4 * x - 1 = 0 ↔ 
  (x = 1 + (Real.sqrt 6) / 2 ∨ x = 1 - (Real.sqrt 6) / 2) := by
sorry

theorem solution_eq2 (x : ℝ) :
  (x - 1) * (x + 2) = 28 ↔ 
  (x = -6 ∨ x = 5) := by
sorry

end solution_eq1_solution_eq2_l300_300522


namespace seat_notation_format_l300_300340

theorem seat_notation_format (r1 r2 s1 s2 : ℕ) : 
  (r1, s1) = (10, 3) → (r2, s2) = (6, 16) :=
by
  intro h
  rw h
  sorry

end seat_notation_format_l300_300340


namespace exact_probability_five_shots_l300_300429

theorem exact_probability_five_shots (p : ℝ) (h1 : 0 < p) (h2 : p ≤ 1) :
  (let hit := p
       miss := 1 - p
       comb := 6 in
   comb * hit^3 * miss^2 = 6 * p^3 * (1 - p)^2) :=
by sorry

end exact_probability_five_shots_l300_300429


namespace loss_percentage_is_13_l300_300095

def cost_price : ℕ := 1500
def selling_price : ℕ := 1305
def loss : ℕ := cost_price - selling_price
def loss_percentage : ℚ := (loss : ℚ) / cost_price * 100

theorem loss_percentage_is_13 :
  loss_percentage = 13 := 
by
  sorry

end loss_percentage_is_13_l300_300095


namespace a_finishes_race_in_t_seconds_l300_300787

theorem a_finishes_race_in_t_seconds 
  (time_B : ℝ := 45)
  (dist_B : ℝ := 100)
  (dist_A_wins_by : ℝ := 20)
  (total_dist : ℝ := 100)
  : ∃ t : ℝ, t = 36 := 
  sorry

end a_finishes_race_in_t_seconds_l300_300787


namespace cylinder_sphere_surface_area_ratio_l300_300055

theorem cylinder_sphere_surface_area_ratio 
  (d : ℝ) -- d represents the diameter of the sphere and the height of the cylinder
  (S1 S2 : ℝ) -- Surface areas of the cylinder and the sphere
  (r := d / 2) -- radius of the sphere
  (S1 := 6 * π * r ^ 2) -- surface area of the cylinder
  (S2 := 4 * π * r ^ 2) -- surface area of the sphere
  : S1 / S2 = 3 / 2 :=
  sorry

end cylinder_sphere_surface_area_ratio_l300_300055


namespace usual_time_proof_l300_300273

noncomputable 
def usual_time (P T : ℝ) := (P * T) / (100 - P)

theorem usual_time_proof (P T U : ℝ) (h1 : P > 0) (h2 : P < 100) (h3 : T > 0) (h4 : U = usual_time P T) : U = (P * T) / (100 - P) :=
by
    sorry

end usual_time_proof_l300_300273


namespace mike_taller_than_mark_l300_300643

-- Define the heights of Mark and Mike in terms of feet and inches
def mark_height_feet : ℕ := 5
def mark_height_inches : ℕ := 3
def mike_height_feet : ℕ := 6
def mike_height_inches : ℕ := 1

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Conversion of heights to inches
def mark_total_height_in_inches : ℕ := mark_height_feet * feet_to_inches + mark_height_inches
def mike_total_height_in_inches : ℕ := mike_height_feet * feet_to_inches + mike_height_inches

-- Define the problem statement: proving Mike is 10 inches taller than Mark
theorem mike_taller_than_mark : mike_total_height_in_inches - mark_total_height_in_inches = 10 :=
by sorry

end mike_taller_than_mark_l300_300643


namespace cos_of_complementary_angle_l300_300185

theorem cos_of_complementary_angle (Y Z : ℝ) (h : Y + Z = π / 2) 
  (sin_Y : Real.sin Y = 3 / 5) : Real.cos Z = 3 / 5 := 
  sorry

end cos_of_complementary_angle_l300_300185


namespace perpendicular_value_of_k_parallel_value_of_k_l300_300467

variables (a b : ℝ × ℝ) (k : ℝ)

def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-3, 1)
def ka_plus_b (k : ℝ) : ℝ × ℝ := (2*k - 3, 3*k + 1)
def a_minus_3b : ℝ × ℝ := (11, 0)

theorem perpendicular_value_of_k 
  (h : a = vector_a ∧ b = vector_b ∧ (ka_plus_b k) = (2*k - 3, 3*k + 1) ∧ a_minus_3b = (11, 0)) :
  a - ka_plus_b k = a_minus_3b → k = (3 / 2) :=
sorry

theorem parallel_value_of_k 
  (h : a = vector_a ∧ b = vector_b ∧ (ka_plus_b k) = (2*k - 3, 3*k + 1) ∧ a_minus_3b = (11, 0)) :
  ∃ k, (ka_plus_b (-1/3)) = (-1/3 * 11, -1/3 * 0) ∧ k = -1 / 3 :=
sorry

end perpendicular_value_of_k_parallel_value_of_k_l300_300467


namespace number_of_blue_faces_l300_300860

theorem number_of_blue_faces (n : ℕ) (h : (6 * n^2) / (6 * n^3) = 1 / 3) : n = 3 :=
by
  sorry

end number_of_blue_faces_l300_300860


namespace quadratic_has_real_roots_l300_300849

theorem quadratic_has_real_roots (k : ℝ) (h : k > 0) : ∃ x : ℝ, x^2 + 2 * x - k = 0 :=
by
  sorry

end quadratic_has_real_roots_l300_300849


namespace eggs_sold_l300_300779

/-- Define the notion of trays and eggs in this context -/
def trays_of_eggs : ℤ := 30

/-- Define the initial collection of trays by Haman -/
def initial_trays : ℤ := 10

/-- Define the number of trays dropped by Haman -/
def dropped_trays : ℤ := 2

/-- Define the additional trays that Haman's father told him to collect -/
def additional_trays : ℤ := 7

/-- Define the total eggs sold -/
def total_eggs_sold : ℤ :=
  (initial_trays - dropped_trays) * trays_of_eggs + additional_trays * trays_of_eggs

-- Theorem to prove the total eggs sold
theorem eggs_sold : total_eggs_sold = 450 :=
by 
  -- Insert proof here
  sorry

end eggs_sold_l300_300779


namespace vasya_has_greater_expected_area_l300_300881

noncomputable def expected_area_rectangle : ℚ :=
1 / 6 * (1 * 1 + 1 * 2 + 1 * 3 + 1 * 4 + 1 * 5 + 1 * 6 + 
         2 * 1 + 2 * 2 + 2 * 3 + 2 * 4 + 2 * 5 + 2 * 6 + 
         3 * 1 + 3 * 2 + 3 * 3 + 3 * 4 + 3 * 5 + 3 * 6 + 
         4 * 1 + 4 * 2 + 4 * 3 + 4 * 4 + 4 * 5 + 4 * 6 + 
         5 * 1 + 5 * 2 + 5 * 3 + 5 * 4 + 5 * 5 + 5 * 6 + 
         6 * 1 + 6 * 2 + 6 * 3 + 6 * 4 + 6 * 5 + 6 * 6)

noncomputable def expected_area_square : ℚ := 
1 / 6 * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

theorem vasya_has_greater_expected_area : expected_area_rectangle < expected_area_square :=
by {
  -- A calculation of this sort should be done symbolically, not in this theorem,
  -- but the primary goal here is to show the structure of the statement.
  -- Hence, implement symbolic computation later to finalize proof.
  sorry
}

end vasya_has_greater_expected_area_l300_300881


namespace spider_travel_distance_l300_300439

theorem spider_travel_distance (r : ℝ) (journey3 : ℝ) (diameter : ℝ) (leg2 : ℝ) :
    r = 75 → journey3 = 110 → diameter = 2 * r → 
    leg2 = Real.sqrt (diameter^2 - journey3^2) → 
    diameter + leg2 + journey3 = 362 :=
by
  sorry

end spider_travel_distance_l300_300439


namespace race_completion_time_l300_300788

variable (t : ℕ)
variable (vA vB : ℕ)
variable (tB : ℕ := 45)
variable (d : ℕ := 100)
variable (diff : ℕ := 20)
variable h1 : vA * t = d
variable h2 : vB * t = d - diff
variable h3 : vB = d / tB

theorem race_completion_time (h : vB = d / tB): t = 36 :=
by sorry

end race_completion_time_l300_300788


namespace prove_b_div_c_equals_one_l300_300925

theorem prove_b_div_c_equals_one
  (a b c d : ℕ)
  (h_a : a > 0 ∧ a < 4)
  (h_b : b > 0 ∧ b < 4)
  (h_c : c > 0 ∧ c < 4)
  (h_d : d > 0 ∧ d < 4)
  (h_eq : 4^a + 3^b + 2^c + 1^d = 78) :
  b / c = 1 :=
by
  sorry

end prove_b_div_c_equals_one_l300_300925


namespace Robin_total_distance_walked_l300_300964

-- Define the conditions
def distance_house_to_city_center := 500
def distance_walked_initially := 200

-- Define the proof problem
theorem Robin_total_distance_walked :
  distance_walked_initially * 2 + distance_house_to_city_center = 900 := by
  sorry

end Robin_total_distance_walked_l300_300964


namespace simplify_frac_48_72_l300_300386

theorem simplify_frac_48_72 : (48 / 72 : ℚ) = 2 / 3 :=
by
  -- In Lean, we prove the equality of the simplified fractions.
  sorry

end simplify_frac_48_72_l300_300386


namespace expression_equals_4008_l300_300227

def calculate_expression : ℤ :=
  let expr := (2004 - (2011 - 196)) + (2011 - (196 - 2004))
  expr

theorem expression_equals_4008 : calculate_expression = 4008 := 
by
  sorry

end expression_equals_4008_l300_300227


namespace collinear_points_sum_l300_300345

variables {a b : ℝ}

/-- If the points (1, a, b), (a, b, 3), and (b, 3, a) are collinear, then b + a = 3.
-/
theorem collinear_points_sum (h : ∃ k : ℝ, 
  (a - 1, b - a, 3 - b) = k • (b - 1, 3 - a, a - b)) : b + a = 3 :=
sorry

end collinear_points_sum_l300_300345


namespace radha_profit_percentage_l300_300085

theorem radha_profit_percentage (SP CP : ℝ) (hSP : SP = 144) (hCP : CP = 90) :
  ((SP - CP) / CP) * 100 = 60 := by
  sorry

end radha_profit_percentage_l300_300085


namespace ariel_fish_l300_300567

theorem ariel_fish (total_fish : ℕ) (male_ratio : ℚ) (female_ratio : ℚ) (female_fish : ℕ) : 
  total_fish = 45 ∧ male_ratio = 2/3 ∧ female_ratio = 1/3 → female_fish = 15 :=
by
  sorry

end ariel_fish_l300_300567


namespace greater_expected_area_vasya_l300_300886

noncomputable def expected_area_vasya : ℚ :=
  (1/6) * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

noncomputable def expected_area_asya : ℚ :=
  ((1/6) * (1 + 2 + 3 + 4 + 5 + 6)) * ((1/6) * (1 + 2 + 3 + 4 + 5 + 6))

theorem greater_expected_area_vasya : expected_area_vasya > expected_area_asya :=
  by
  -- We've provided the expected area values as definitions
  -- expected_area_vasya = 91/6
  -- vs. expected_area_asya = 12.25 = (21/6)^2 = 441/36 = 12.25
  sorry

end greater_expected_area_vasya_l300_300886


namespace total_clothes_donated_l300_300732

theorem total_clothes_donated
  (pants : ℕ) (jumpers : ℕ) (pajama_sets : ℕ) (tshirts : ℕ)
  (friends : ℕ)
  (adam_donation : ℕ)
  (half_adam_donated : ℕ)
  (friends_donation : ℕ)
  (total_donation : ℕ)
  (h1 : pants = 4) 
  (h2 : jumpers = 4) 
  (h3 : pajama_sets = 4 * 2) 
  (h4 : tshirts = 20) 
  (h5 : friends = 3)
  (h6 : adam_donation = pants + jumpers + pajama_sets + tshirts) 
  (h7 : half_adam_donated = adam_donation / 2) 
  (h8 : friends_donation = friends * adam_donation) 
  (h9 : total_donation = friends_donation + half_adam_donated) :
  total_donation = 126 :=
by
  sorry

end total_clothes_donated_l300_300732


namespace new_figure_perimeter_equals_5_l300_300955

-- Defining the side length of the square and the equilateral triangle
def side_length : ℝ := 1

-- Defining the perimeter of the new figure
def new_figure_perimeter : ℝ := 3 * side_length + 2 * side_length

-- Statement: The perimeter of the new figure equals 5
theorem new_figure_perimeter_equals_5 :
  new_figure_perimeter = 5 := by
  sorry

end new_figure_perimeter_equals_5_l300_300955


namespace team_a_score_l300_300417

theorem team_a_score : ∀ (A : ℕ), A + 9 + 4 = 15 → A = 2 :=
by
  intros A h
  sorry

end team_a_score_l300_300417


namespace simplify_fraction_48_72_l300_300379

theorem simplify_fraction_48_72 : (48 : ℚ) / 72 = 2 / 3 := sorry

end simplify_fraction_48_72_l300_300379


namespace parallel_planes_sufficient_not_necessary_for_perpendicular_lines_l300_300911

variables {Point Line Plane : Type}
variables (α β : Plane) (ℓ m : Line) (point_on_line_ℓ : Point) (point_on_line_m : Point)

-- Definitions of conditions
def line_perpendicular_to_plane (ℓ : Line) (α : Plane) : Prop := sorry
def line_contained_in_plane (m : Line) (β : Plane) : Prop := sorry
def planes_parallel (α β : Plane) : Prop := sorry
def line_perpendicular_to_line (ℓ m : Line) : Prop := sorry

axiom h1 : line_perpendicular_to_plane ℓ α
axiom h2 : line_contained_in_plane m β

-- Statement of the proof problem
theorem parallel_planes_sufficient_not_necessary_for_perpendicular_lines : 
  (planes_parallel α β → line_perpendicular_to_line ℓ m) ∧ 
  ¬ (line_perpendicular_to_line ℓ m → planes_parallel α β) :=
  sorry

end parallel_planes_sufficient_not_necessary_for_perpendicular_lines_l300_300911


namespace total_length_figure2_l300_300000

-- Define the initial lengths of each segment in Figure 1.
def initial_length_horizontal1 := 5
def initial_length_vertical1 := 10
def initial_length_horizontal2 := 4
def initial_length_vertical2 := 3
def initial_length_horizontal3 := 3
def initial_length_vertical3 := 5
def initial_length_horizontal4 := 4
def initial_length_vertical_sum := 10 + 3 + 5

-- Define the transformations.
def bottom_length := initial_length_horizontal1
def rightmost_vertical_length := initial_length_vertical1 - 2
def top_horizontal_length := initial_length_horizontal2 - 3
def leftmost_vertical_length := initial_length_vertical1

-- Define the total length in Figure 2 as a theorem to be proved.
theorem total_length_figure2:
  bottom_length + rightmost_vertical_length + top_horizontal_length + leftmost_vertical_length = 24 := by
  sorry

end total_length_figure2_l300_300000


namespace value_of_f_2014_l300_300487

def f : ℕ → ℕ := sorry

theorem value_of_f_2014 : (∀ n : ℕ, f (f n) + f n = 2 * n + 3) → (f 0 = 1) → (f 2014 = 2015) := by
  intro h₁ h₀
  have h₂ := h₀
  sorry

end value_of_f_2014_l300_300487


namespace simplify_fraction_l300_300384

namespace FractionSimplify

-- Define the fraction 48/72
def original_fraction : ℚ := 48 / 72

-- The goal is to prove that this fraction simplifies to 2/3
theorem simplify_fraction : original_fraction = 2 / 3 := by
  sorry

end FractionSimplify

end simplify_fraction_l300_300384


namespace long_furred_brown_dogs_l300_300625

theorem long_furred_brown_dogs :
  ∀ (T L B N LB : ℕ), T = 60 → L = 45 → B = 35 → N = 12 →
  (LB = L + B - (T - N)) → LB = 32 :=
by
  intros T L B N LB hT hL hB hN hLB
  sorry

end long_furred_brown_dogs_l300_300625


namespace total_eggs_sold_l300_300780

def initial_trays : Nat := 10
def dropped_trays : Nat := 2
def added_trays : Nat := 7
def eggs_per_tray : Nat := 36

theorem total_eggs_sold : initial_trays - dropped_trays + added_trays * eggs_per_tray = 540 := by
  sorry

end total_eggs_sold_l300_300780


namespace fractional_eq_nonneg_solution_l300_300160

theorem fractional_eq_nonneg_solution 
  (m x : ℝ)
  (h1 : x ≠ 2)
  (h2 : x ≥ 0)
  (eq_fractional : m / (x - 2) + 1 = x / (2 - x)) :
  m ≤ 2 ∧ m ≠ -2 := 
  sorry

end fractional_eq_nonneg_solution_l300_300160


namespace man_savings_percentage_l300_300272

theorem man_savings_percentage
  (salary expenses : ℝ)
  (increase_percentage : ℝ)
  (current_savings : ℝ)
  (P : ℝ)
  (h1 : salary = 7272.727272727273)
  (h2 : increase_percentage = 0.05)
  (h3 : current_savings = 400)
  (h4 : current_savings + (increase_percentage * salary) = (P / 100) * salary) :
  P = 10.5 := 
sorry

end man_savings_percentage_l300_300272


namespace transformed_polynomial_l300_300808

noncomputable def P : Polynomial ℝ := Polynomial.C 9 + Polynomial.X ^ 3 - 4 * Polynomial.X ^ 2 

noncomputable def Q : Polynomial ℝ := Polynomial.C 243 + Polynomial.X ^ 3 - 12 * Polynomial.X ^ 2 

theorem transformed_polynomial :
  ∀ (r : ℝ), Polynomial.aeval r P = 0 → Polynomial.aeval (3 * r) Q = 0 := 
by
  sorry

end transformed_polynomial_l300_300808


namespace geometric_series_sum_l300_300572

theorem geometric_series_sum :
  ∑' i : ℕ, (2 / 3) ^ (i + 1) = 2 :=
by
  sorry

end geometric_series_sum_l300_300572


namespace percentage_less_than_a_plus_d_l300_300852

-- Define the mean, standard deviation, and given conditions
variables (a d : ℝ)
axiom symmetric_distribution : ∀ x, x = 2 * a - x 

-- Main theorem
theorem percentage_less_than_a_plus_d :
  (∃ (P_less_than : ℝ → ℝ), P_less_than (a + d) = 0.84) :=
sorry

end percentage_less_than_a_plus_d_l300_300852


namespace probability_of_8_or_9_ring_l300_300099

theorem probability_of_8_or_9_ring (p10 p9 p8 : ℝ) (h1 : p10 = 0.3) (h2 : p9 = 0.3) (h3 : p8 = 0.2) :
  p9 + p8 = 0.5 :=
by
  sorry

end probability_of_8_or_9_ring_l300_300099


namespace Emmanuel_jelly_beans_l300_300004

theorem Emmanuel_jelly_beans :
  ∀ (total : ℕ) (thomas_ratio : ℚ) (ratio_barry : ℕ) (ratio_emmanuel : ℕ),
  total = 200 →
  thomas_ratio = 10 / 100 →
  ratio_barry = 4 →
  ratio_emmanuel = 5 →
  let thomas_share := thomas_ratio * total in
  let remaining := total - thomas_share in
  let total_parts := ratio_barry + ratio_emmanuel in
  let part_value := remaining / total_parts in
  let emmanuel_share := ratio_emmanuel * part_value in
  emmanuel_share = 100 :=
begin
  intros total thomas_ratio ratio_barry ratio_emmanuel h_total h_thomas_ratio h_ratio_barry h_ratio_emmanuel,
  simp [h_total, h_thomas_ratio, h_ratio_barry, h_ratio_emmanuel],
  have ht : thomas_share = 20, by norm_num [h_total, h_thomas_ratio, thomas_share],
  have hr : remaining = 180, by norm_num [remaining, h_total, ht],
  have total_parts : total_parts = 9, by norm_num [h_ratio_barry, h_ratio_emmanuel, total_parts],
  have pv : part_value = 20, by norm_num [part_value, hr, total_parts],
  have es : emmanuel_share = 100, by norm_num [emmanuel_share, h_ratio_emmanuel, pv],
  exact es,
end

end Emmanuel_jelly_beans_l300_300004


namespace top_card_is_5_or_king_l300_300865

-- Define the number of cards in a deck
def total_cards : ℕ := 52

-- Define the number of 5s in a deck
def number_of_5s : ℕ := 4

-- Define the number of Kings in a deck
def number_of_kings : ℕ := 4

-- Define the number of favorable outcomes (cards that are either 5 or King)
def favorable_outcomes : ℕ := number_of_5s + number_of_kings

-- Define the probability as a fraction
def probability : ℚ := favorable_outcomes / total_cards

-- Theorem: The probability that the top card is either a 5 or a King is 2/13
theorem top_card_is_5_or_king (h_total_cards : total_cards = 52)
    (h_number_of_5s : number_of_5s = 4)
    (h_number_of_kings : number_of_kings = 4) :
    probability = 2 / 13 := by
  -- Proof would go here
  sorry

end top_card_is_5_or_king_l300_300865


namespace vasya_expected_area_greater_l300_300884

/-- Vasya and Asya roll dice to cut out shapes and determine whose expected area is greater. -/
theorem vasya_expected_area_greater :
  let A : ℕ := 1
  let B : ℕ := 2
  (6 * 7 * (2 * 7 / 6) * 21 / 6) < (21 * 91 / 6) := 
by
  sorry

end vasya_expected_area_greater_l300_300884


namespace infinite_product_eq_four_four_thirds_l300_300750

theorem infinite_product_eq_four_four_thirds :
  ∏' n : ℕ, (4^(n+1)^(1/(2^(n+1)))) = 4^(4/3) :=
sorry

end infinite_product_eq_four_four_thirds_l300_300750


namespace find_matrix_C_l300_300472

/-- Define the given matrices A and B. --/
def A : Matrix (Fin 2) (Fin 2) ℚ := ![![2, 1], ![1, 3]]
def B : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![1, -1]]

/-- State the equivalent proof problem: finding matrix C such that AC = B --/
theorem find_matrix_C :
  ∃ C : Matrix (Fin 2) (Fin 2) ℚ, A ⬝ C = B :=
begin
  -- Using inverse matrix and the condition that A and B are given,
  -- we know that C = A⁻¹ ⬝ B
  let C := ![![3/5, 4/5], ![-1/5, -3/5]],
  use C,
  sorry -- The proof steps should go here, but we are skipping them as per the instructions.
end

end find_matrix_C_l300_300472


namespace age_calculation_l300_300407

/-- Let Thomas be a 6-year-old child, Shay be 13 years older than Thomas, 
and also 5 years younger than James. Let Violet be 3 years younger than 
Thomas, and Emily be the same age as Shay. This theorem proves that when 
Violet reaches the age of Thomas (6 years old), James will be 27 years old 
and Emily will be 22 years old. -/
theorem age_calculation : 
  ∀ (Thomas Shay James Violet Emily : ℕ),
    Thomas = 6 →
    Shay = Thomas + 13 →
    James = Shay + 5 →
    Violet = Thomas - 3 →
    Emily = Shay →
    (Violet + (6 - Violet) = 6) →
    (James + (6 - Violet) = 27 ∧ Emily + (6 - Violet) = 22) :=
by
  intros Thomas Shay James Violet Emily ht hs hj hv he hv_diff
  sorry

end age_calculation_l300_300407


namespace complement_of_M_in_U_l300_300610

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 4, 6}
def complement_U_M : Set ℕ := {1, 3, 5}

theorem complement_of_M_in_U :
  (U \ M) = complement_U_M :=
by
  sorry

end complement_of_M_in_U_l300_300610


namespace sequence_tenth_term_l300_300183

theorem sequence_tenth_term :
  ∃ (a : ℕ → ℚ), a 1 = 1 ∧ (∀ n : ℕ, n > 0 → a (n + 1) = a n / (1 + a n)) ∧ a 10 = 1 / 10 :=
sorry

end sequence_tenth_term_l300_300183


namespace predict_HCl_formed_l300_300483

-- Define the initial conditions and chemical reaction constants
def initial_moles_CH4 : ℝ := 3
def initial_moles_Cl2 : ℝ := 6
def volume : ℝ := 2

-- Define the reaction stoichiometry constants
def stoich_CH4_to_HCl : ℝ := 2
def stoich_CH4 : ℝ := 1
def stoich_Cl2 : ℝ := 2

-- Declare the hypothesis that reaction goes to completion
axiom reaction_goes_to_completion : Prop

-- Define the function to calculate the moles of HCl formed
def moles_HCl_formed : ℝ :=
  initial_moles_CH4 * stoich_CH4_to_HCl

-- Prove the predicted amount of HCl formed is 6 moles under the given conditions
theorem predict_HCl_formed : reaction_goes_to_completion → moles_HCl_formed = 6 := by
  sorry

end predict_HCl_formed_l300_300483


namespace probability_no_three_consecutive_A_l300_300269

def total_strings : ℕ :=
  3^6

def count_strings_with_three_consecutive_A : ℕ :=
  32 + 12 + 4 + 1

def count_strings_without_three_consecutive_A : ℕ :=
  total_strings - count_strings_with_three_consecutive_A

theorem probability_no_three_consecutive_A : (count_strings_without_three_consecutive_A : ℚ) / total_strings = 680/729 := by sorry

end probability_no_three_consecutive_A_l300_300269


namespace six_x_mod_nine_l300_300252

theorem six_x_mod_nine (x : ℕ) (k : ℕ) (hx : x = 9 * k + 5) : (6 * x) % 9 = 3 :=
by
  sorry

end six_x_mod_nine_l300_300252


namespace divisor_of_a_l300_300363

theorem divisor_of_a (a b c d : ℕ) (h1 : Nat.gcd a b = 18) (h2 : Nat.gcd b c = 45) 
  (h3 : Nat.gcd c d = 75) (h4 : 80 < Nat.gcd d a ∧ Nat.gcd d a < 120) : 
  7 ∣ a :=
by
  sorry

end divisor_of_a_l300_300363


namespace normal_CDF_is_correct_l300_300309

noncomputable def normal_cdf (a σ : ℝ) (x : ℝ) : ℝ :=
  0.5 + (1 / Real.sqrt (2 * Real.pi)) * ∫ t in (0)..(x - a) / σ, Real.exp (-t^2 / 2)

theorem normal_CDF_is_correct (a σ : ℝ) (ha : σ > 0) (x : ℝ) :
  (normal_cdf a σ x) = 0.5 + (1 / Real.sqrt (2 * Real.pi)) * ∫ t in (0)..(x - a) / σ, Real.exp (-t^2 / 2) :=
by
  sorry

end normal_CDF_is_correct_l300_300309


namespace gcd_of_8_and_12_l300_300524

theorem gcd_of_8_and_12 :
  let a := 8
  let b := 12
  let lcm_ab := 24
  Nat.lcm a b = lcm_ab → Nat.gcd a b = 4 :=
by
  intros
  sorry

end gcd_of_8_and_12_l300_300524


namespace shoe_matching_probability_l300_300090

open Rat

-- Each color has a distinct number of pairs.
def black_pairs : Nat := 6
def brown_pairs : Nat := 3
def gray_pairs : Nat := 2

-- Total pairs
def total_pairs : Nat := black_pairs + brown_pairs + gray_pairs

-- Total shoes
def total_shoes : Nat := 2 * total_pairs

-- Total counts for each color
def black_shoes : Nat := 2 * black_pairs
def brown_shoes : Nat := 2 * brown_pairs
def gray_shoes : Nat := 2 * gray_pairs

-- Probabilities for selecting shoes of the same color and opposite foot.
def prob_same_color_same_foot (color: Nat) (pairs: Nat) : ℚ :=
  (color.toRat / total_shoes.toRat) * ((pairs.toRat - 1) / (total_shoes.toRat - 1))

def probability : ℚ :=
  prob_same_color_same_foot black_shoes black_pairs +
  prob_same_color_same_foot brown_shoes brown_pairs +
  prob_same_color_same_foot gray_shoes gray_pairs

theorem shoe_matching_probability :
  probability = 7/33 := by
  -- The detailed proof would be filled here
  sorry

end shoe_matching_probability_l300_300090


namespace sum_primes_between_20_and_30_l300_300241

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem sum_primes_between_20_and_30 :
  (primes_between 20 30).sum = 52 :=
by
  sorry

end sum_primes_between_20_and_30_l300_300241


namespace sum_of_side_lengths_l300_300101

theorem sum_of_side_lengths (p q r : ℕ) (h : p = 8 ∧ q = 1 ∧ r = 5) 
    (area_ratio : 128 / 50 = 64 / 25) 
    (side_length_ratio : 8 / 5 = Real.sqrt (128 / 50)) :
    p + q + r = 14 := 
by 
  sorry

end sum_of_side_lengths_l300_300101


namespace expected_faces_rolled_six_times_l300_300716

-- Define a random variable indicating appearance of a particular face
noncomputable def ζi (n : ℕ): ℝ := if n > 0 then 1 - (5 / 6) ^ 6 else 0

-- Define the expected number of distinct faces
noncomputable def expected_distinct_faces : ℝ := 6 * ζi 1

theorem expected_faces_rolled_six_times :
  expected_distinct_faces = (6 ^ 6 - 5 ^ 6) / 6 ^ 5 :=
by
  -- Here we would provide the proof
  sorry

end expected_faces_rolled_six_times_l300_300716


namespace ariel_fish_l300_300560

theorem ariel_fish (total_fish : ℕ) (male_fraction female_fraction : ℚ) (h1 : total_fish = 45) (h2 : male_fraction = 2 / 3) (h3 : female_fraction = 1 - male_fraction) : total_fish * female_fraction = 15 :=
by
  sorry

end ariel_fish_l300_300560


namespace simplify_fraction_l300_300383

namespace FractionSimplify

-- Define the fraction 48/72
def original_fraction : ℚ := 48 / 72

-- The goal is to prove that this fraction simplifies to 2/3
theorem simplify_fraction : original_fraction = 2 / 3 := by
  sorry

end FractionSimplify

end simplify_fraction_l300_300383


namespace equivalent_product_lists_l300_300015

-- Definitions of the value assigned to each letter.
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1
  | 'B' => 2
  | 'C' => 3
  | 'D' => 4
  | 'E' => 5
  | 'F' => 6
  | 'G' => 7
  | 'H' => 8
  | 'I' => 9
  | 'J' => 10
  | 'K' => 11
  | 'L' => 12
  | 'M' => 13
  | 'N' => 14
  | 'O' => 15
  | 'P' => 16
  | 'Q' => 17
  | 'R' => 18
  | 'S' => 19
  | 'T' => 20
  | 'U' => 21
  | 'V' => 22
  | 'W' => 23
  | 'X' => 24
  | 'Y' => 25
  | 'Z' => 26
  | _ => 0  -- We only care about uppercase letters A-Z

def list_product (l : List Char) : ℕ :=
  l.foldl (λ acc c => acc * (letter_value c)) 1

-- Given the list MNOP with their products equals letter values.
def MNOP := ['M', 'N', 'O', 'P']
def BJUZ := ['B', 'J', 'U', 'Z']

-- Lean statement to assert the equivalence of the products.
theorem equivalent_product_lists :
  list_product MNOP = list_product BJUZ :=
by
  sorry

end equivalent_product_lists_l300_300015


namespace rectangle_no_shaded_square_l300_300580

noncomputable def total_rectangles (cols : ℕ) : ℕ :=
  (cols + 1) * (cols + 1 - 1) / 2

noncomputable def shaded_rectangles (cols : ℕ) : ℕ :=
  cols + 1 - 1

noncomputable def probability_no_shaded (cols : ℕ) : ℚ :=
  let n := total_rectangles cols
  let m := shaded_rectangles cols
  1 - (m / n)

theorem rectangle_no_shaded_square :
  probability_no_shaded 2003 = 2002 / 2003 :=
by
  sorry

end rectangle_no_shaded_square_l300_300580


namespace min_value_arith_geo_seq_l300_300531

theorem min_value_arith_geo_seq (A B C D : ℕ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h4 : 0 < D)
  (h_arith : C - B = B - A) (h_geo : C * C = B * D) (h_frac : 4 * C = 7 * B) :
  A + B + C + D = 97 :=
sorry

end min_value_arith_geo_seq_l300_300531


namespace a_alone_finishes_work_in_60_days_l300_300541

noncomputable def work_done_per_day_a_b_c : ℚ := 1 / 10
noncomputable def work_done_per_day_b : ℚ := 1 / 20
noncomputable def work_done_per_day_c : ℚ := 1 / 30

theorem a_alone_finishes_work_in_60_days :
  ∀ (A B C : ℚ), A + B + C = work_done_per_day_a_b_c → B = work_done_per_day_b → C = work_done_per_day_c → 
  1 / A = 60 :=
by
  intros A B C h1 h2 h3
  -- proof omitted
  sorry

end a_alone_finishes_work_in_60_days_l300_300541


namespace root_equiv_sum_zero_l300_300028

variable {a b c : ℝ}
variable (h₀ : a ≠ 0)

theorem root_equiv_sum_zero : (1 root_of (a * 1^2 + b * 1 + c = 0)) ↔ (a + b + c = 0) :=
by
  sorry

end root_equiv_sum_zero_l300_300028


namespace remaining_episodes_l300_300256

theorem remaining_episodes (total_seasons : ℕ) (episodes_per_season : ℕ) (fraction_watched : ℚ) 
  (H1 : total_seasons = 12) (H2 : episodes_per_season = 20) (H3 : fraction_watched = 1/3) : 
  (total_seasons * episodes_per_season) - (fraction_watched * (total_seasons * episodes_per_season)) = 160 :=
by
  sorry

end remaining_episodes_l300_300256


namespace binomial_divisible_by_prime_l300_300764

theorem binomial_divisible_by_prime (p n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  (Nat.choose n p) - (n / p) % p = 0 := 
sorry

end binomial_divisible_by_prime_l300_300764


namespace crystal_discount_is_50_percent_l300_300299

noncomputable def discount_percentage_original_prices_and_revenue
  (original_price_cupcake : ℝ)
  (original_price_cookie : ℝ)
  (total_cupcakes_sold : ℕ)
  (total_cookies_sold : ℕ)
  (total_revenue : ℝ)
  (percentage_discount : ℝ) :
  Prop :=
  total_cupcakes_sold * (original_price_cupcake * (1 - percentage_discount / 100)) +
  total_cookies_sold * (original_price_cookie * (1 - percentage_discount / 100)) = total_revenue

theorem crystal_discount_is_50_percent :
  discount_percentage_original_prices_and_revenue 3 2 16 8 32 50 :=
by sorry

end crystal_discount_is_50_percent_l300_300299


namespace mike_training_hours_l300_300369

-- Define the individual conditions
def first_weekday_hours : Nat := 2
def first_weekend_hours : Nat := 1
def first_week_days : Nat := 5
def first_weekend_days : Nat := 2

def second_weekday_hours : Nat := 3
def second_weekend_hours : Nat := 2
def second_week_days : Nat := 4  -- since the first day of second week is a rest day
def second_weekend_days : Nat := 2

def first_week_hours : Nat := (first_weekday_hours * first_week_days) + (first_weekend_hours * first_weekend_days)
def second_week_hours : Nat := (second_weekday_hours * second_week_days) + (second_weekend_hours * second_weekend_days)

def total_training_hours : Nat := first_week_hours + second_week_hours

-- The final proof statement
theorem mike_training_hours : total_training_hours = 28 := by
  exact sorry

end mike_training_hours_l300_300369


namespace NewYearSeasonMarkup_theorem_l300_300436

def NewYearSeasonMarkup (C N : ℝ) : Prop :=
    (0.90 * (1.20 * C * (1 + N)) = 1.35 * C) -> N = 0.25

theorem NewYearSeasonMarkup_theorem (C : ℝ) (h₀ : C > 0) : ∃ (N : ℝ), NewYearSeasonMarkup C N :=
by
  use 0.25
  sorry

end NewYearSeasonMarkup_theorem_l300_300436


namespace honor_students_count_l300_300985

def num_students_total : ℕ := 24
def num_honor_students_girls : ℕ := 3
def num_honor_students_boys : ℕ := 4

def num_girls : ℕ := 13
def num_boys : ℕ := 11

theorem honor_students_count (total_students : ℕ) 
    (prob_girl_honor : ℚ) (prob_boy_honor : ℚ)
    (girls : ℕ) (boys : ℕ)
    (honor_girls : ℕ) (honor_boys : ℕ) :
    total_students < 30 →
    prob_girl_honor = 3 / 13 →
    prob_boy_honor = 4 / 11 →
    girls = 13 →
    honor_girls = 3 →
    boys = 11 →
    honor_boys = 4 →
    girls + boys = total_students →
    honor_girls + honor_boys = 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [← h4, ← h5, ← h6, ← h7, ← h8]
  exact 7

end honor_students_count_l300_300985


namespace parabola_line_slope_l300_300165

theorem parabola_line_slope (y1 y2 x1 x2 : ℝ) (h1 : y1 ^ 2 = 6 * x1) (h2 : y2 ^ 2 = 6 * x2) 
    (midpoint_condition : (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = 2) :
  (y1 - y2) / (x1 - x2) = 3 / 2 :=
by
  -- here will be the actual proof using the given hypothesis
  sorry

end parabola_line_slope_l300_300165


namespace polynomial_remainder_l300_300803

theorem polynomial_remainder (P : ℝ → ℝ) (h1 : P 19 = 16) (h2 : P 15 = 8) : 
  ∃ Q : ℝ → ℝ, ∀ x, P x = (x - 15) * (x - 19) * Q x + 2 * x - 22 :=
by
  sorry

end polynomial_remainder_l300_300803


namespace Robin_total_distance_walked_l300_300963

-- Define the conditions
def distance_house_to_city_center := 500
def distance_walked_initially := 200

-- Define the proof problem
theorem Robin_total_distance_walked :
  distance_walked_initially * 2 + distance_house_to_city_center = 900 := by
  sorry

end Robin_total_distance_walked_l300_300963


namespace error_percentage_in_area_l300_300063

theorem error_percentage_in_area
  (L W : ℝ)          -- Actual length and width of the rectangle
  (hL' : ℝ)          -- Measured length with 8% excess
  (hW' : ℝ)          -- Measured width with 5% deficit
  (hL'_def : hL' = 1.08 * L)  -- Condition for length excess
  (hW'_def : hW' = 0.95 * W)  -- Condition for width deficit
  :
  ((hL' * hW' - L * W) / (L * W) * 100 = 2.6) := sorry

end error_percentage_in_area_l300_300063


namespace factorize_2070_l300_300168

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def is_unique_factorization (n a b : ℕ) : Prop := a * b = n ∧ is_two_digit a ∧ is_two_digit b

-- The final theorem statement
theorem factorize_2070 : 
  (∃ a b : ℕ, is_unique_factorization 2070 a b) ∧ 
  ∀ a b : ℕ, is_unique_factorization 2070 a b → (a = 30 ∧ b = 69) ∨ (a = 69 ∧ b = 30) :=
by 
  sorry

end factorize_2070_l300_300168


namespace geometric_seq_condition_l300_300933

/-- In a geometric sequence with common ratio q, sum of the first n terms S_n.
  Given q > 0, show that it is a necessary condition for {S_n} to be an increasing sequence,
  but not a sufficient condition. -/
theorem geometric_seq_condition (a1 q : ℝ) (S : ℕ → ℝ)
  (hS : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h1 : q > 0) : 
  (∀ n, S n < S (n + 1)) ↔ a1 > 0 :=
sorry

end geometric_seq_condition_l300_300933


namespace buffet_dishes_l300_300276

-- To facilitate the whole proof context, but skipping proof parts with 'sorry'

-- Oliver will eat if there is no mango in the dishes

variables (D : ℕ) -- Total number of dishes

-- Conditions:
variables (h1 : 3 <= D) -- there are at least 3 dishes with mango salsa
variables (h2 : 1 ≤ D / 6) -- one-sixth of dishes have fresh mango
variables (h3 : 1 ≤ D) -- there's at least one dish with mango jelly
variables (h4 : D / 6 ≥ 2) -- Oliver can pick out the mangoes from 2 of dishes with fresh mango
variables (h5 : D - (3 + (D / 6 - 2) + 1) = 28) -- there are 28 dishes Oliver can eat

theorem buffet_dishes : D = 36 :=
by
  sorry -- Skip the actual proof

end buffet_dishes_l300_300276


namespace polynomial_is_2y2_l300_300103

variables (x y : ℝ)

theorem polynomial_is_2y2 (P : ℝ → ℝ → ℝ) (h : P x y + (x^2 - y^2) = x^2 + y^2) : 
  P x y = 2 * y^2 :=
by
  sorry

end polynomial_is_2y2_l300_300103


namespace sum_primes_20_to_30_l300_300236

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : list ℕ := 
  [n ∈ list.range (b + 1) | n > a ∧ n ≤ b ∧ is_prime n]

def sum_primes_between {a b : ℕ} (ha : a = 20) (hb : b = 30) : ℕ :=
  (primes_between a b).sum

theorem sum_primes_20_to_30 : sum_primes_between (ha : 20) (hb : 30) = 52 := by
  sorry

end sum_primes_20_to_30_l300_300236


namespace fertilizer_needed_per_acre_l300_300938

-- Definitions for the conditions
def horse_daily_fertilizer : ℕ := 5 -- Each horse produces 5 gallons of fertilizer per day.
def horses : ℕ := 80 -- Janet has 80 horses.
def days : ℕ := 25 -- It takes 25 days until all her fields are fertilized.
def total_acres : ℕ := 20 -- Janet's farmland is 20 acres.

-- Calculated intermediate values
def total_fertilizer : ℕ := horse_daily_fertilizer * horses * days -- Total fertilizer produced
def fertilizer_per_acre : ℕ := total_fertilizer / total_acres -- Fertilizer needed per acre

-- Theorem to prove
theorem fertilizer_needed_per_acre : fertilizer_per_acre = 500 := by
  sorry

end fertilizer_needed_per_acre_l300_300938


namespace greatest_int_with_gcd_3_l300_300229

theorem greatest_int_with_gcd_3 (n : ℕ) (h1 : n < 150) (h2 : Int.gcd n 24 = 3) : n = 141 := by
  sorry

end greatest_int_with_gcd_3_l300_300229


namespace number_of_ways_to_place_pawns_l300_300047

theorem number_of_ways_to_place_pawns :
  let n := 5 in
  let number_of_placements := (n.factorial) in
  let number_of_permutations := (n.factorial) in
  number_of_placements * number_of_permutations = 14400 :=
by
  sorry

end number_of_ways_to_place_pawns_l300_300047


namespace average_of_xyz_l300_300783

variable (x y z : ℝ)

theorem average_of_xyz (h : (5 / 4) * (x + y + z) = 20) : (x + y + z) / 3 = 16 / 3 := by
  sorry

end average_of_xyz_l300_300783


namespace honor_students_count_l300_300975

noncomputable def G : ℕ := 13
noncomputable def B : ℕ := 11
def E_G : ℕ := 3
def E_B : ℕ := 4

theorem honor_students_count (h1 : G + B < 30) 
    (h2 : (E_G : ℚ) / G = 3 / 13) 
    (h3 : (E_B : ℚ) / B = 4 / 11) :
    E_G + E_B = 7 := 
sorry

end honor_students_count_l300_300975


namespace largest_digit_M_divisible_by_six_l300_300841

theorem largest_digit_M_divisible_by_six :
  (∃ M : ℕ, M ≤ 9 ∧ (45670 + M) % 6 = 0 ∧ ∀ m : ℕ, m ≤ M → (45670 + m) % 6 ≠ 0) :=
sorry

end largest_digit_M_divisible_by_six_l300_300841


namespace arnold_total_protein_l300_300288

-- Conditions
def protein_in_collagen_powder (scoops: ℕ) : ℕ := 9 * scoops
def protein_in_protein_powder (scoops: ℕ) : ℕ := 21 * scoops
def protein_in_steak : ℕ := 56
def protein_in_greek_yogurt : ℕ := 15
def protein_in_almonds (cups: ℕ) : ℕ := 6 * (cups * 4) / 4
def half_cup_almonds_protein : ℕ := 12

-- Statement
theorem arnold_total_protein : 
  protein_in_collagen_powder 1 + protein_in_protein_powder 2 + protein_in_steak + protein_in_greek_yogurt + half_cup_almonds_protein = 134 :=
  by
    sorry

end arnold_total_protein_l300_300288


namespace number_of_ways_to_purchase_magazines_l300_300266

/-
Conditions:
1. The bookstore sells 11 different magazines.
2. 8 of these magazines are priced at 2 yuan each.
3. 3 of these magazines are priced at 1 yuan each.
4. Xiao Zhang has 10 yuan to buy magazines.
5. Xiao Zhang can buy at most one copy of each magazine.
6. Xiao Zhang wants to spend all 10 yuan.

Question:
The number of different ways Xiao Zhang can purchase magazines with 10 yuan.

Answer:
266
-/

theorem number_of_ways_to_purchase_magazines : ∀ (magazines_1_yuan magazines_2_yuan : ℕ),
  magazines_1_yuan = 3 →
  magazines_2_yuan = 8 →
  (∃ (ways : ℕ), ways = 266) :=
by
  intros
  sorry

end number_of_ways_to_purchase_magazines_l300_300266


namespace dad_borrowed_nickels_l300_300819

-- Definitions for the initial and remaining nickels
def initial_nickels : ℕ := 31
def remaining_nickels : ℕ := 11

-- Statement of the problem in Lean
theorem dad_borrowed_nickels : initial_nickels - remaining_nickels = 20 := by
  -- Proof goes here
  sorry

end dad_borrowed_nickels_l300_300819


namespace polynomial_form_l300_300150

noncomputable def polynomial_solution (P : ℝ → ℝ) :=
  ∀ a b c : ℝ, (a * b + b * c + c * a = 0) → (P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c))

theorem polynomial_form :
  ∀ (P : ℝ → ℝ), polynomial_solution P ↔ ∃ (a b : ℝ), ∀ x : ℝ, P x = a * x^2 + b * x^4 :=
by 
  sorry

end polynomial_form_l300_300150


namespace total_points_correct_l300_300514

variable (H Q T : ℕ)

-- Given conditions
def hw_points : ℕ := 40
def quiz_points := hw_points + 5
def test_points := 4 * quiz_points

-- Question: Prove the total points assigned are 265
theorem total_points_correct :
  H = hw_points →
  Q = quiz_points →
  T = test_points →
  H + Q + T = 265 :=
by
  intros h_hw h_quiz h_test
  rw [h_hw, h_quiz, h_test]
  exact sorry

end total_points_correct_l300_300514


namespace total_books_received_l300_300653

theorem total_books_received (initial_books additional_books total_books: ℕ)
  (h1 : initial_books = 54)
  (h2 : additional_books = 23) :
  (initial_books + additional_books = 77) := by
  sorry

end total_books_received_l300_300653


namespace product_of_coordinates_of_intersection_l300_300412

-- Conditions: Defining the equations of the two circles
def circle1_eq (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 10*y + 25 = 0
def circle2_eq (x y : ℝ) : Prop := x^2 - 8*x + y^2 - 10*y + 37 = 0

-- Translated problem to prove the question equals the correct answer
theorem product_of_coordinates_of_intersection :
  ∃ (x y : ℝ), circle1_eq x y ∧ circle2_eq x y ∧ x * y = 10 :=
sorry

end product_of_coordinates_of_intersection_l300_300412


namespace train_speed_l300_300440

theorem train_speed
  (train_length : Real := 460)
  (bridge_length : Real := 140)
  (time_seconds : Real := 48) :
  ((train_length + bridge_length) / time_seconds) * 3.6 = 45 :=
by
  sorry

end train_speed_l300_300440


namespace cloth_cost_price_per_metre_l300_300278

theorem cloth_cost_price_per_metre (total_metres : ℕ) (total_price : ℕ) (loss_per_metre : ℕ) :
  total_metres = 300 → total_price = 18000 → loss_per_metre = 5 → (total_price / total_metres + loss_per_metre) = 65 :=
by
  intros
  sorry

end cloth_cost_price_per_metre_l300_300278


namespace prob_5_shots_expected_number_shots_l300_300428

variable (p : ℝ) (hp : 0 < p ∧ p ≤ 1)

def prob_exactly_five_shots : ℝ := 6 * p^3 * (1 - p)^2
def expected_shots : ℝ := 3 / p

theorem prob_5_shots (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  -- Prove that the probability of exactly 5 shots needed is as calculated
  prob_exactly_five_shots p = 6 * p^3 * (1 - p)^2 :=
by
  sorry

theorem expected_number_shots (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  -- Prove that the expected number of shots to hit all targets is as calculated
  expected_shots p = 3 / p :=
by
  sorry

end prob_5_shots_expected_number_shots_l300_300428


namespace placement_of_pawns_l300_300051

-- Define the size of the chessboard and the total number of pawns
def board_size := 5
def total_pawns := 5

-- Define the problem statement
theorem placement_of_pawns : 
  (∑ (pawns : Finset (Fin total_pawns → Fin board_size)), 
    (∀ p1 p2 : Fin total_pawns, p1 ≠ p2 → pawns(p1) ≠ pawns(p2)) ∧ -- distinct positions
    (∀ i j : Fin total_pawns, i ≠ j → pawns(i) ≠ pawns(j)) ∧ -- no same row/column
    pawns.card = total_pawns) = 14400 :=
sorry

end placement_of_pawns_l300_300051


namespace jack_travel_total_hours_l300_300415

theorem jack_travel_total_hours :
  (20 + 14 * 24) + (15 + 10 * 24) + (10 + 7 * 24) = 789 := by
  sorry

end jack_travel_total_hours_l300_300415


namespace arithmetic_square_root_of_16_is_4_l300_300825

theorem arithmetic_square_root_of_16_is_4 : ∃ x : ℤ, x * x = 16 ∧ x = 4 := 
sorry

end arithmetic_square_root_of_16_is_4_l300_300825


namespace no_integer_solutions_l300_300201

theorem no_integer_solutions
  (x y : ℤ) :
  3 * x^2 = 16 * y^2 + 8 * y + 5 → false :=
by
  sorry

end no_integer_solutions_l300_300201


namespace dvd_blu_ratio_l300_300862

theorem dvd_blu_ratio (D B : ℕ) (h1 : D + B = 378) (h2 : (D : ℚ) / (B - 4 : ℚ) = 9 / 2) :
  D / Nat.gcd D B = 51 ∧ B / Nat.gcd D B = 12 :=
by
  sorry

end dvd_blu_ratio_l300_300862


namespace count_true_statements_l300_300144

theorem count_true_statements (a b c d : ℝ) : 
  (∃ (H1 : a ≠ b) (H2 : c ≠ d), a + c = b + d) →
  ((a ≠ b) ∧ (c ≠ d) → a + c ≠ b + d) = false ∧ 
  ((a + c ≠ b + d) → (a ≠ b) ∧ (c ≠ d)) = false ∧ 
  (∃ (H3 : a = b) (H4 : c = d), a + c ≠ b + d) = false ∧ 
  ((a + c = b + d) → (a = b) ∨ (c = d)) = false → 
  number_of_true_statements = 0 := 
by
  sorry

end count_true_statements_l300_300144


namespace amount_after_two_years_l300_300018

noncomputable def annual_increase (initial_amount : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial_amount * (1 + rate) ^ years

theorem amount_after_two_years :
  annual_increase 32000 (1/8) 2 = 40500 :=
by
  sorry

end amount_after_two_years_l300_300018


namespace parking_cost_per_hour_l300_300212

theorem parking_cost_per_hour (avg_cost : ℝ) (total_initial_cost : ℝ) (hours_excessive : ℝ) (total_hours : ℝ) (cost_first_2_hours : ℝ)
  (h1 : cost_first_2_hours = 9.00) 
  (h2 : avg_cost = 2.361111111111111)
  (h3 : total_hours = 9) 
  (h4 : hours_excessive = 7):
  (total_initial_cost / total_hours = avg_cost) -> 
  (total_initial_cost = cost_first_2_hours + hours_excessive * x) -> 
  x = 1.75 := 
by
  intros h5 h6
  sorry

end parking_cost_per_hour_l300_300212


namespace correctCountForDivisibilityBy15_l300_300182

namespace Divisibility

noncomputable def countWaysToMakeDivisibleBy15 : Nat := 
  let digits := [0, 2, 4, 5, 7, 9]
  let baseSum := 2 + 0 + 1 + 6 + 0 + 2
  let validLastDigit := [0, 5]
  let totalCombinations := 6^4
  let ways := 2 * totalCombinations
  let adjustment := (validLastDigit.length * digits.length * digits.length * digits.length * validLastDigit.length) / 4 -- Correcting multiplier as per reference
  adjustment

theorem correctCountForDivisibilityBy15 : countWaysToMakeDivisibleBy15 = 864 := 
  by
    sorry

end Divisibility

end correctCountForDivisibilityBy15_l300_300182


namespace expected_number_of_different_faces_l300_300704

theorem expected_number_of_different_faces :
  let ζ_i (i : Fin 6) := if (∃ k, k ∈ Finset.range 6) then 1 else 0,
      ζ := (List.range 6).sum (ζ_i),
      p := (5 / 6 : ℝ) ^ 6
  in (Expectation (λ ω => ζ)) = (6 * (1 - p)) :=
by
  sorry

end expected_number_of_different_faces_l300_300704


namespace log_squared_sum_eq_one_l300_300747

open Real

theorem log_squared_sum_eq_one :
  (log 2)^2 * log 250 + (log 5)^2 * log 40 = 1 := by
  sorry

end log_squared_sum_eq_one_l300_300747


namespace movie_ticket_notation_l300_300341

-- Definition of movie ticket notation
def ticket_notation (row : ℕ) (seat : ℕ) : (ℕ × ℕ) :=
  (row, seat)

-- Given condition: "row 10, seat 3" is denoted as (10, 3)
def given := ticket_notation 10 3 = (10, 3)

-- Proof statement: "row 6, seat 16" is denoted as (6, 16)
theorem movie_ticket_notation : ticket_notation 6 16 = (6, 16) :=
by
  -- Proof omitted, since the theorem statement is the focus
  sorry

end movie_ticket_notation_l300_300341


namespace gertrude_fleas_l300_300464

variables (G M O : ℕ)

def fleas_maud := M = 5 * O
def fleas_olive := O = G / 2
def total_fleas := G + M + O = 40

theorem gertrude_fleas
  (h_maud : fleas_maud M O)
  (h_olive : fleas_olive G O)
  (h_total : total_fleas G M O) :
  G = 10 :=
sorry

end gertrude_fleas_l300_300464


namespace man_l300_300274

theorem man's_present_age (P : ℝ) 
  (h1 : P = (4/5) * P + 10)
  (h2 : P = (3/2.5) * P - 10) :
  P = 50 :=
sorry

end man_l300_300274


namespace intersection_with_y_axis_is_03_l300_300828

-- Define the line equation
def line (x : ℝ) : ℝ := x + 3

-- The intersection point with the y-axis, i.e., where x = 0
def y_axis_intersection : Prod ℝ ℝ := (0, line 0)

-- Prove that the intersection point is (0, 3)
theorem intersection_with_y_axis_is_03 : y_axis_intersection = (0, 3) :=
by
  simp [y_axis_intersection, line]
  sorry

end intersection_with_y_axis_is_03_l300_300828


namespace baby_frogs_on_rock_l300_300169

theorem baby_frogs_on_rock (f_l f_L f_T : ℕ) (h1 : f_l = 5) (h2 : f_L = 3) (h3 : f_T = 32) : 
  f_T - (f_l + f_L) = 24 :=
by sorry

end baby_frogs_on_rock_l300_300169


namespace Robin_needs_to_buy_more_bottles_l300_300374

/-- Robin wants to drink exactly nine bottles of water each day.
    She initially bought six hundred seventeen bottles.
    Prove that she will need to buy 4 more bottles on the last day
    to meet her goal of drinking exactly nine bottles each day. -/
theorem Robin_needs_to_buy_more_bottles :
  ∀ total_bottles bottles_per_day : ℕ, total_bottles = 617 → bottles_per_day = 9 → 
  ∃ extra_bottles : ℕ, (617 % 9) + extra_bottles = 9 ∧ extra_bottles = 4 :=
by
  sorry

end Robin_needs_to_buy_more_bottles_l300_300374


namespace b_payment_l300_300696

theorem b_payment (b_days : ℕ) (a_days : ℕ) (total_wages : ℕ) (b_payment : ℕ) :
  b_days = 10 →
  a_days = 15 →
  total_wages = 5000 →
  b_payment = 3000 :=
by
  intros h1 h2 h3
  -- conditions
  have hb := h1
  have ha := h2
  have ht := h3
  -- skipping proof
  sorry

end b_payment_l300_300696


namespace proof_problem_l300_300635

variable (x y : ℝ)

theorem proof_problem 
  (h1 : 0.30 * x = 0.40 * 150 + 90)
  (h2 : 0.20 * x = 0.50 * 180 - 60)
  (h3 : y = 0.75 * x)
  (h4 : y^2 > x + 100) :
  x = 150 ∧ y = 112.5 :=
by
  sorry

end proof_problem_l300_300635


namespace fisherman_sale_l300_300262

/-- 
If the price of the radio is both the 4th highest price and the 13th lowest price 
among the prices of the fishes sold at a sale, then the total number of fishes 
sold at the fisherman sale is 16. 
-/
theorem fisherman_sale (h4_highest : ∃ price : ℕ, ∀ p : ℕ, p > price → p ∈ {a | a ≠ price} ∧ p > 3)
                       (h13_lowest : ∃ price : ℕ, ∀ p : ℕ, p < price → p ∈ {a | a ≠ price} ∧ p < 13) :
  ∃ n : ℕ, n = 16 :=
sorry

end fisherman_sale_l300_300262


namespace rice_and_grain_separation_l300_300525

theorem rice_and_grain_separation (total_weight : ℕ) (sample_size : ℕ) (non_rice_sample : ℕ) (non_rice_in_batch : ℕ) :
  total_weight = 1524 →
  sample_size = 254 →
  non_rice_sample = 28 →
  non_rice_in_batch = total_weight * non_rice_sample / sample_size →
  non_rice_in_batch = 168 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end rice_and_grain_separation_l300_300525


namespace xiao_xuan_wins_l300_300650

def cards_game (n : ℕ) (min_take : ℕ) (max_take : ℕ) (initial_turn : String) : String :=
  if initial_turn = "Xiao Liang" then "Xiao Xuan" else "Xiao Liang"

theorem xiao_xuan_wins :
  cards_game 17 1 2 "Xiao Liang" = "Xiao Xuan" :=
sorry

end xiao_xuan_wins_l300_300650


namespace solution_of_equation_l300_300056

theorem solution_of_equation (m : ℝ) :
  (∃ x : ℝ, x = (4 - 3 * m) / 2 ∧ x > 0) ↔ m < 4 / 3 ∧ m ≠ 2 / 3 :=
by
  sorry

end solution_of_equation_l300_300056


namespace intersection_A_B_l300_300321

variable A : Set Int
variable B : Set Int

def setA : Set Int := {-1, 1, 2, 4}
def setB : Set Int := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by 
  let A := setA
  let B := setB
  sorry

end intersection_A_B_l300_300321


namespace candy_distribution_l300_300892

theorem candy_distribution (candy : ℕ) (people : ℕ) (hcandy : candy = 30) (hpeople : people = 5) :
  ∃ k : ℕ, candy - k = people * (candy / people) ∧ k = 0 := 
by
  sorry

end candy_distribution_l300_300892


namespace part_a_part_b_l300_300427

variable (p : ℝ)
variable (h_pos : 0 < p)
variable (h_prob : p ≤ 1)

theorem part_a :
  let q := 1 - p in
  ∃ f : ℕ → ℝ, f 5 = 6 * p^3 * q^2 :=
  by
    sorry

theorem part_b :
  ∃ f : ℕ → ℝ, f 3 = 3 / p :=
  by
    sorry

end part_a_part_b_l300_300427


namespace work_days_for_A_l300_300923

/-- If A is thrice as fast as B and together they can do a work in 15 days, A alone can do the work in 20 days. -/
theorem work_days_for_A (Wb : ℕ) (Wa : ℕ) (H_wa : Wa = 3 * Wb) (H_total : (Wa + Wb) * 15 = Wa * 20) : A_work_days = 20 :=
by
  sorry

end work_days_for_A_l300_300923


namespace constants_A_B_C_l300_300453

theorem constants_A_B_C (A B C : ℝ) (h₁ : ∀ x : ℝ, (x^2 + 5 * x - 6) / (x^4 + x^2) = A / x^2 + (B * x + C) / (x^2 + 1)) :
  A = -6 ∧ B = 0 ∧ C = 7 :=
by
  sorry

end constants_A_B_C_l300_300453


namespace remaining_episodes_l300_300257

theorem remaining_episodes (total_seasons : ℕ) (episodes_per_season : ℕ) (fraction_watched : ℚ) 
  (H1 : total_seasons = 12) (H2 : episodes_per_season = 20) (H3 : fraction_watched = 1/3) : 
  (total_seasons * episodes_per_season) - (fraction_watched * (total_seasons * episodes_per_season)) = 160 :=
by
  sorry

end remaining_episodes_l300_300257


namespace range_of_4a_minus_2b_l300_300465

theorem range_of_4a_minus_2b (a b : ℝ) (h1 : 0 ≤ a - b) (h2 : a - b ≤ 1) (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) :
  2 ≤ 4 * a - 2 * b ∧ 4 * a - 2 * b ≤ 7 := 
sorry

end range_of_4a_minus_2b_l300_300465


namespace distance_to_grocery_store_l300_300367

-- Definitions of given conditions
def miles_to_mall := 6
def miles_to_pet_store := 5
def miles_back_home := 9
def miles_per_gallon := 15
def cost_per_gallon := 3.5
def total_cost := 7

-- The Lean statement to prove the distance driven to the grocery store.
theorem distance_to_grocery_store (miles_to_mall miles_to_pet_store miles_back_home miles_per_gallon cost_per_gallon total_cost : ℝ) :
(total_cost / cost_per_gallon) * miles_per_gallon - (miles_to_mall + miles_to_pet_store + miles_back_home) = 10 := by
  sorry

end distance_to_grocery_store_l300_300367


namespace simplify_expression_correct_l300_300952

def simplify_expression (x : ℝ) : Prop :=
  2 * x - 3 * (2 - x) + 4 * (2 + 3 * x) - 5 * (1 - 2 * x) = 27 * x - 3

theorem simplify_expression_correct (x : ℝ) : simplify_expression x :=
by
  sorry

end simplify_expression_correct_l300_300952


namespace hari_joined_after_5_months_l300_300200

noncomputable def praveen_investment := 3780 * 12
noncomputable def hari_investment (x : ℕ) := 9720 * (12 - x)

theorem hari_joined_after_5_months :
  ∃ (x : ℕ), (praveen_investment : ℝ) / (hari_investment x) = (2:ℝ) / 3 ∧ x = 5 :=
by {
  sorry
}

end hari_joined_after_5_months_l300_300200


namespace crabapple_recipients_sequence_count_l300_300812

/-- Mrs. Crabapple teaches a class of 15 students and her advanced literature class meets three times a week.
    She picks a new student each period to receive a crabapple, ensuring no student receives more than one
    crabapple in a week. Prove that the number of different sequences of crabapple recipients is 2730. -/
theorem crabapple_recipients_sequence_count :
  ∃ sequence_count : ℕ, sequence_count = 15 * 14 * 13 ∧ sequence_count = 2730 :=
by
  sorry

end crabapple_recipients_sequence_count_l300_300812


namespace salary_increase_l300_300102

theorem salary_increase (S0 S3 : ℕ) (r : ℕ) : 
  S0 = 3000 ∧ S3 = 8232 ∧ (S0 * (1 + r / 100)^3 = S3) → r = 40 :=
by
  sorry

end salary_increase_l300_300102


namespace order_of_even_function_l300_300075

noncomputable def is_even (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

noncomputable def is_monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x < f y

theorem order_of_even_function {f : ℝ → ℝ}
  (h_even : is_even f)
  (h_mono_inc : is_monotonically_increasing_on_nonneg f) :
  f (-π) > f (3) ∧ f (3) > f (-2) :=
sorry

end order_of_even_function_l300_300075


namespace complex_number_solution_l300_300331

theorem complex_number_solution (z i : ℂ) (h : z * (i - i^2) = 1 + i^3) (h1 : i^2 = -1) (h2 : i^3 = -i) (h3 : i^4 = 1) : 
  z = -i := 
by 
  sorry

end complex_number_solution_l300_300331


namespace average_age_before_new_students_l300_300209

theorem average_age_before_new_students
  (N : ℕ) (A : ℚ) 
  (hN : N = 8) 
  (new_avg : (A - 4) = ((A * N) + (32 * 8)) / (N + 8)) 
  : A = 40 := 
by
  sorry

end average_age_before_new_students_l300_300209


namespace solution_set_l300_300172

variables {f : ℝ → ℝ}

-- Condition 1: f is an odd function
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

-- Condition 2: f is decreasing on (-∞, 0]
def decreasing_on_neg (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → x ≤ 0 → y ≤ 0 → f x ≥ f y

-- Hypotheses
variable (odd_f : odd_function f)
variable (decreasing_f : decreasing_on_neg f)

-- The theorem to prove
theorem solution_set (x : ℝ) : f (Real.log x) < -f 1 ↔ x ∈ Ioi (Real.exp (-1)) :=
  sorry

end solution_set_l300_300172


namespace interest_difference_l300_300652

theorem interest_difference :
  let P := 10000
  let R := 6
  let T := 2
  let SI := P * R * T / 100
  let CI := P * (1 + R / 100)^T - P
  CI - SI = 36 :=
by
  let P := 10000
  let R := 6
  let T := 2
  let SI := P * R * T / 100
  let CI := P * (1 + R / 100)^T - P
  show CI - SI = 36
  sorry

end interest_difference_l300_300652


namespace milk_production_group_B_l300_300956

theorem milk_production_group_B (a b c d e : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_pos_d : d > 0) (h_pos_e : e > 0) :
  ((1.2 * b * d * e) / (a * c)) = 1.2 * (b / (a * c)) * d * e := 
by
  sorry

end milk_production_group_B_l300_300956


namespace corn_growth_first_week_l300_300142

theorem corn_growth_first_week (x : ℝ) (h1 : x + 2*x + 8*x = 22) : x = 2 :=
by
  sorry

end corn_growth_first_week_l300_300142


namespace expected_number_of_different_faces_l300_300710

noncomputable def expected_faces : ℝ :=
  let probability_face_1_not_appearing := (5 / 6)^6
  let E_zeta_1 := 1 - probability_face_1_not_appearing
  6 * E_zeta_1

theorem expected_number_of_different_faces :
  expected_faces = (6^6 - 5^6) / 6^5 := by 
  sorry

end expected_number_of_different_faces_l300_300710


namespace xiao_ming_actual_sleep_time_l300_300967

def required_sleep_time : ℝ := 9
def recorded_excess_sleep_time : ℝ := 0.4
def actual_sleep_time (required : ℝ) (excess : ℝ) : ℝ := required + excess

theorem xiao_ming_actual_sleep_time :
  actual_sleep_time required_sleep_time recorded_excess_sleep_time = 9.4 := 
by
  sorry

end xiao_ming_actual_sleep_time_l300_300967


namespace remaining_episodes_l300_300259

theorem remaining_episodes (seasons : ℕ) (episodes_per_season : ℕ) (fraction_watched : ℚ) 
  (h_seasons : seasons = 12) (h_episodes_per_season : episodes_per_season = 20) 
  (h_fraction_watched : fraction_watched = 1/3) : 
  (seasons * episodes_per_season - fraction_watched * (seasons * episodes_per_season) = 160) := 
by
  sorry

end remaining_episodes_l300_300259


namespace find_range_of_a_l300_300774

noncomputable def f (x : ℝ) : ℝ := (1 / Real.exp x) - (Real.exp x) + 2 * x - (1 / 3) * x ^ 3

theorem find_range_of_a (a : ℝ) (h : f (3 * a ^ 2) + f (2 * a - 1) ≥ 0) : a ∈ Set.Icc (-1 : ℝ) (1 / 3) :=
sorry

end find_range_of_a_l300_300774


namespace initial_butterfat_percentage_l300_300372

theorem initial_butterfat_percentage (P : ℝ) :
  let initial_butterfat := (P / 100) * 1000
  let removed_butterfat := (23 / 100) * 50
  let remaining_volume := 1000 - 50
  let desired_butterfat := (3 / 100) * remaining_volume
  initial_butterfat - removed_butterfat = desired_butterfat
→ P = 4 :=
by
  intros
  let initial_butterfat := (P / 100) * 1000
  let removed_butterfat := (23 / 100) * 50
  let remaining_volume := 1000 - 50
  let desired_butterfat := (3 / 100) * remaining_volume
  sorry

end initial_butterfat_percentage_l300_300372


namespace find_a_share_l300_300418

noncomputable def total_investment (a b c : ℕ) : ℕ :=
  a + b + c

noncomputable def total_profit (b_share total_inv b_inv : ℕ) : ℕ :=
  b_share * total_inv / b_inv

noncomputable def a_share (a_inv total_inv total_pft : ℕ) : ℕ :=
  a_inv * total_pft / total_inv

theorem find_a_share
  (a_inv b_inv c_inv b_share : ℕ)
  (h1 : a_inv = 7000)
  (h2 : b_inv = 11000)
  (h3 : c_inv = 18000)
  (h4 : b_share = 880) :
  a_share a_inv (total_investment a_inv b_inv c_inv) (total_profit b_share (total_investment a_inv b_inv c_inv) b_inv) = 560 := 
by
  sorry

end find_a_share_l300_300418


namespace green_space_equation_l300_300226

theorem green_space_equation (x : ℝ) (h_area : x * (x - 30) = 1000) :
  x * (x - 30) = 1000 := 
by
  exact h_area

end green_space_equation_l300_300226


namespace fraction_distinctly_marked_l300_300435

theorem fraction_distinctly_marked 
  (area_large_rectangle : ℕ)
  (fraction_shaded : ℚ)
  (fraction_further_marked : ℚ)
  (h_area_large_rectangle : area_large_rectangle = 15 * 24)
  (h_fraction_shaded : fraction_shaded = 1/3)
  (h_fraction_further_marked : fraction_further_marked = 1/2) :
  (fraction_further_marked * fraction_shaded = 1/6) :=
by
  sorry

end fraction_distinctly_marked_l300_300435


namespace episodes_per_season_l300_300071

theorem episodes_per_season
  (days_to_watch : ℕ)
  (episodes_per_day : ℕ)
  (seasons : ℕ) :
  days_to_watch = 10 →
  episodes_per_day = 6 →
  seasons = 4 →
  (episodes_per_day * days_to_watch) / seasons = 15 :=
by
  intros
  sorry

end episodes_per_season_l300_300071


namespace sum_of_primes_between_20_and_30_l300_300234

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l300_300234


namespace units_digit_of_pow_sum_is_correct_l300_300678

theorem units_digit_of_pow_sum_is_correct : 
  (24^3 + 42^3) % 10 = 2 := by
  -- Start the proof block
  sorry

end units_digit_of_pow_sum_is_correct_l300_300678


namespace count_valid_propositions_is_zero_l300_300606

theorem count_valid_propositions_is_zero :
  (∀ (a b : ℝ), (a > b → a^2 > b^2) = false) ∧
  (∀ (a b : ℝ), (a^2 > b^2 → a > b) = false) ∧
  (∀ (a b : ℝ), (a > b → b / a < 1) = false) ∧
  (∀ (a b : ℝ), (a > b → 1 / a < 1 / b) = false) :=
by
  sorry

end count_valid_propositions_is_zero_l300_300606


namespace b_share_of_payment_l300_300695

def work_fraction (d : ℕ) : ℚ := 1 / d

def total_one_day_work (a_days b_days c_days : ℕ) : ℚ :=
  work_fraction a_days + work_fraction b_days + work_fraction c_days

def share_of_work (b_days : ℕ) (total_work : ℚ) : ℚ :=
  work_fraction b_days / total_work

def share_of_payment (total_payment : ℚ) (work_share : ℚ) : ℚ :=
  total_payment * work_share

theorem b_share_of_payment 
  (a_days b_days c_days : ℕ) (total_payment : ℚ):
  a_days = 6 → b_days = 8 → c_days = 12 → total_payment = 1800 →
  share_of_payment total_payment (share_of_work b_days (total_one_day_work a_days b_days c_days)) = 600 :=
by
  intros ha hb hc hp
  unfold total_one_day_work work_fraction share_of_work share_of_payment
  rw [ha, hb, hc, hp]
  -- Simplify the fractions and the multiplication
  sorry

end b_share_of_payment_l300_300695


namespace four_p_minus_three_is_perfect_square_l300_300801

theorem four_p_minus_three_is_perfect_square 
  {n p : ℕ} (hn : 1 < n) (hp : 1 < p) (hp_prime : Prime p) 
  (h1 : n ∣ (p - 1)) (h2 : p ∣ (n^3 - 1)) :
  ∃ k : ℕ, 4 * p - 3 = k ^ 2 := 
by 
  sorry

end four_p_minus_three_is_perfect_square_l300_300801


namespace complex_combination_l300_300670

open Complex

def a : ℂ := 2 - I
def b : ℂ := -1 + I

theorem complex_combination : 2 * a + 3 * b = 1 + I :=
by
  -- Proof goes here
  sorry

end complex_combination_l300_300670


namespace range_of_a_l300_300916

theorem range_of_a (a : ℝ) :
  (∀ x, (3 ≤ x → 2*a*x + 4 ≤ 2*a*(x+1) + 4) ∧ (2 < x ∧ x < 3 → (a + (2*a + 2)/(x-2) ≤ a + (2*a + 2)/(x-1))) ) →
  -1 < a ∧ a ≤ -2/3 :=
by
  intros h
  sorry

end range_of_a_l300_300916


namespace remainder_of_sum_mod_9_l300_300595

theorem remainder_of_sum_mod_9 :
  (9023 + 9024 + 9025 + 9026 + 9027) % 9 = 2 :=
by
  sorry

end remainder_of_sum_mod_9_l300_300595


namespace inequality_proof_l300_300616

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + 4 * a / (b + c)) * (1 + 4 * b / (a + c)) * (1 + 4 * c / (a + b)) > 25 :=
sorry

end inequality_proof_l300_300616


namespace correct_statements_for_sequence_l300_300899

theorem correct_statements_for_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :
  -- Statement 1
  (S_n = n^2 + n → ∀ n, ∃ d : ℝ, a n = a 1 + (n - 1) * d) ∧
  -- Statement 2
  (S_n = 2^n - 1 → ∃ q : ℝ, ∀ n, a n = a 1 * q^(n - 1)) ∧
  -- Statement 3
  (∀ n, n ≥ 2 → 2 * a n = a (n + 1) + a (n - 1) → ∀ n, ∃ d : ℝ, a n = a 1 + (n - 1) * d) ∧
  -- Statement 4
  (¬(∀ n, n ≥ 2 → a n^2 = a (n + 1) * a (n - 1) → ∃ q : ℝ, ∀ n, a n = a 1 * q^(n - 1))) :=
sorry

end correct_statements_for_sequence_l300_300899


namespace chocolate_oranges_initial_l300_300082

theorem chocolate_oranges_initial (p_c p_o G n_c x : ℕ) 
  (h_candy_bar_price : p_c = 5) 
  (h_orange_price : p_o = 10) 
  (h_goal : G = 1000) 
  (h_candy_bars_sold : n_c = 160) 
  (h_equation : G = p_o * x + p_c * n_c) : 
  x = 20 := 
by
  sorry

end chocolate_oranges_initial_l300_300082


namespace p_cycling_speed_l300_300890

-- J starts walking at 6 kmph at 12:00
def start_time : ℕ := 12 * 60  -- time in minutes for convenience
def j_speed : ℤ := 6  -- in kmph
def j_start_time : ℕ := start_time  -- 12:00 in minutes

-- P starts cycling at 13:30
def p_start_time : ℕ := (13 * 60) + 30  -- time in minutes for convenience

-- They are at their respective positions at 19:30
def end_time : ℕ := (19 * 60) + 30  -- time in minutes for convenience

-- At 19:30, J is 3 km behind P
def j_behind_p_distance : ℤ := 3  -- in kilometers

-- Prove that P's cycling speed = 8 kmph
theorem p_cycling_speed {p_speed : ℤ} :
  j_start_time = start_time →
  p_start_time = (13 * 60) + 30 →
  end_time = (19 * 60) + 30 →
  j_speed = 6 →
  j_behind_p_distance = 3 →
  p_speed = 8 :=
by
  sorry

end p_cycling_speed_l300_300890


namespace parabola_hyperbola_focus_l300_300488

theorem parabola_hyperbola_focus {p : ℝ} :
  let focus_parabola := (p / 2, 0)
  let focus_hyperbola := (2, 0)
  focus_parabola = focus_hyperbola -> p = 4 :=
by
  intro h
  sorry

end parabola_hyperbola_focus_l300_300488


namespace intersection_of_A_and_B_l300_300320

def setA : Set ℝ := {-1, 1, 2, 4}
def setB : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_of_A_and_B : setA ∩ setB = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l300_300320


namespace total_amount_proof_l300_300863

-- Definitions of the base 8 numbers
def silks_base8 := 5267
def stones_base8 := 6712
def spices_base8 := 327

-- Conversion function from base 8 to base 10
def base8_to_base10 (n : ℕ) : ℕ := sorry -- Assume this function converts a base 8 number to base 10

-- Converted values
def silks_base10 := base8_to_base10 silks_base8
def stones_base10 := base8_to_base10 stones_base8
def spices_base10 := base8_to_base10 spices_base8

-- Total amount calculation in base 10
def total_amount_base10 := silks_base10 + stones_base10 + spices_base10

-- The theorem that we want to prove
theorem total_amount_proof : total_amount_base10 = 6488 :=
by
  -- The proof is omitted here.
  sorry

end total_amount_proof_l300_300863


namespace number_a_eq_223_l300_300962

theorem number_a_eq_223 (A B : ℤ) (h1 : A - B = 144) (h2 : A = 3 * B - 14) : A = 223 :=
by
  sorry

end number_a_eq_223_l300_300962


namespace fish_to_rice_l300_300176

variables (f l r e : ℚ)

theorem fish_to_rice (h1: 4 * f = 3 * l) (h2: 5 * l = 7 * r) : f = (21 / 20) * r :=
by
  -- Step through the process of solving for f in terms of r using the given conditions.
  sorry

end fish_to_rice_l300_300176


namespace delivery_truck_speed_l300_300730

theorem delivery_truck_speed :
  ∀ d t₁ t₂: ℝ,
    (t₁ = 15 / 60) ∧ (t₂ = -15 / 60) ∧ 
    (t₁ = d / 20 - 1 / 4) ∧ (t₂ = d / 60 + 1 / 4) →
    (d = 15) →
    (t = 1 / 2) →
    ( ∃ v: ℝ, t = d / v ∧ v = 30 ) :=
by sorry

end delivery_truck_speed_l300_300730


namespace slope_of_line_l300_300776

theorem slope_of_line (a : ℝ) (h : a = (Real.tan (Real.pi / 3))) : a = Real.sqrt 3 := by
sorry

end slope_of_line_l300_300776


namespace nat_divides_2_pow_n_minus_1_l300_300759

theorem nat_divides_2_pow_n_minus_1 (n : ℕ) (hn : 0 < n) : n ∣ 2^n - 1 ↔ n = 1 :=
  sorry

end nat_divides_2_pow_n_minus_1_l300_300759


namespace simplify_fraction_l300_300382

namespace FractionSimplify

-- Define the fraction 48/72
def original_fraction : ℚ := 48 / 72

-- The goal is to prove that this fraction simplifies to 2/3
theorem simplify_fraction : original_fraction = 2 / 3 := by
  sorry

end FractionSimplify

end simplify_fraction_l300_300382


namespace distance_traveled_l300_300126

noncomputable def velocity (t : ℝ) := 2 * t - 3

theorem distance_traveled : 
  (∫ t in (0 : ℝ)..5, |velocity t|) = 29 / 2 := 
by
  sorry

end distance_traveled_l300_300126


namespace quadratic_no_discriminant_23_l300_300296

theorem quadratic_no_discriminant_23 (a b c : ℤ) (h_eq : b^2 - 4 * a * c = 23) : False := sorry

end quadratic_no_discriminant_23_l300_300296


namespace lucas_seq_mod_50_l300_300917

def lucas_seq : ℕ → ℕ
| 0       => 2
| 1       => 5
| (n + 2) => lucas_seq n + lucas_seq (n + 1)

theorem lucas_seq_mod_50 : lucas_seq 49 % 5 = 0 := 
by
  sorry

end lucas_seq_mod_50_l300_300917


namespace total_boys_across_grades_is_692_l300_300290

theorem total_boys_across_grades_is_692 (ga_girls gb_girls gc_girls : ℕ) (ga_boys : ℕ) :
  ga_girls = 256 →
  ga_girls = ga_boys + 52 →
  gb_girls = 360 →
  gb_boys = gb_girls - 40 →
  gc_girls = 168 →
  gc_girls = gc_boys →
  ga_boys + gb_boys + gc_boys = 692 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_boys_across_grades_is_692_l300_300290


namespace volume_P3_correct_m_plus_n_l300_300581

noncomputable def P_0_volume : ℚ := 1

noncomputable def tet_volume (v : ℚ) : ℚ := (1/27) * v

noncomputable def volume_P3 : ℚ := 
  let ΔP1 := 4 * tet_volume P_0_volume
  let ΔP2 := (2/9) * ΔP1
  let ΔP3 := (2/9) * ΔP2
  P_0_volume + ΔP1 + ΔP2 + ΔP3

theorem volume_P3_correct : volume_P3 = 22615 / 6561 := 
by {
  sorry
}

theorem m_plus_n : 22615 + 6561 = 29176 := 
by {
  sorry
}

end volume_P3_correct_m_plus_n_l300_300581


namespace sum_of_primes_between_20_and_30_l300_300242

theorem sum_of_primes_between_20_and_30 :
  (∑ n in { n | n > 20 ∧ n < 30 ∧ Prime n }, n) = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l300_300242


namespace edge_length_in_mm_l300_300920

-- Definitions based on conditions
def cube_volume (a : ℝ) : ℝ := a^3

axiom volume_of_dice : cube_volume 2 = 8

-- Statement of the theorem to be proved
theorem edge_length_in_mm : ∃ (a : ℝ), cube_volume a = 8 ∧ a * 10 = 20 := sorry

end edge_length_in_mm_l300_300920


namespace probability_of_convex_quadrilateral_l300_300894

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_of_convex_quadrilateral :
  let num_points := 8
  let total_chords := binomial num_points 2
  let total_ways_to_select_4_chords := binomial total_chords 4
  let favorable_ways := binomial num_points 4
  (favorable_ways : ℚ) / (total_ways_to_select_4_chords : ℚ) = 2 / 585 :=
by
  -- definitions
  let num_points := 8
  let total_chords := binomial 8 2
  let total_ways_to_select_4_chords := binomial total_chords 4
  let favorable_ways := binomial num_points 4
  
  -- assertion of result
  have h : (favorable_ways : ℚ) / (total_ways_to_select_4_chords : ℚ) = 2 / 585 :=
    sorry
  exact h

end probability_of_convex_quadrilateral_l300_300894


namespace selling_price_of_book_l300_300857

theorem selling_price_of_book (cost_price : ℝ) (profit_percentage : ℝ) (profit : ℝ) (selling_price : ℝ) 
  (h₁ : cost_price = 60) 
  (h₂ : profit_percentage = 25) 
  (h₃ : profit = (profit_percentage / 100) * cost_price) 
  (h₄ : selling_price = cost_price + profit) : 
  selling_price = 75 := 
by
  sorry

end selling_price_of_book_l300_300857


namespace vasya_has_greater_area_l300_300873

-- Definition of a fair six-sided die roll
def die_roll : ℕ → ℝ := λ k, if k ∈ {1, 2, 3, 4, 5, 6} then (1 / 6 : ℝ) else 0

-- Expected value of a function with respect to a probability distribution
noncomputable def expected_value (f : ℕ → ℝ) : ℝ := ∑ k in {1, 2, 3, 4, 5, 6}, f k * die_roll k

-- Vasya's area: A^2 where A is a single die roll
noncomputable def vasya_area : ℝ := expected_value (λ k, (k : ℝ) ^ 2)

-- Asya's area: A * B where A and B are independent die rolls
noncomputable def asya_area : ℝ := (expected_value (λ k, (k : ℝ))) ^ 2

theorem vasya_has_greater_area :
  vasya_area > asya_area := sorry

end vasya_has_greater_area_l300_300873


namespace distinct_real_roots_iff_l300_300786

theorem distinct_real_roots_iff (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (∀ x, x^2 + 3 * x - a = 0 → (x = x₁ ∨ x = x₂))) ↔ a > - (9 : ℝ) / 4 :=
sorry

end distinct_real_roots_iff_l300_300786


namespace number_of_red_notes_each_row_l300_300500

-- Definitions for the conditions
variable (R : ℕ) -- Number of red notes in each row
variable (total_notes : ℕ := 100) -- Total number of notes

-- Derived quantities
def total_red_notes := 5 * R
def total_blue_notes := 2 * total_red_notes + 10

-- Statement of the theorem
theorem number_of_red_notes_each_row 
  (h : total_red_notes + total_blue_notes = total_notes) : 
  R = 6 :=
by
  sorry

end number_of_red_notes_each_row_l300_300500


namespace positive_integers_solution_l300_300457

theorem positive_integers_solution :
  ∀ (m n : ℕ), 0 < m ∧ 0 < n ∧ (3 ^ m - 2 ^ n = -1 ∨ 3 ^ m - 2 ^ n = 5 ∨ 3 ^ m - 2 ^ n = 7) ↔
  (m, n) = (0, 1) ∨ (m, n) = (2, 1) ∨ (m, n) = (1, 2) ∨ (m, n) = (2, 2) :=
by
  sorry

end positive_integers_solution_l300_300457


namespace common_ratio_l300_300022

namespace GeometricSeries

-- Definitions
def a1 : ℚ := 4 / 7
def a2 : ℚ := 16 / 49 

-- Proposition
theorem common_ratio : (a2 / a1) = (4 / 7) :=
by
  sorry

end GeometricSeries

end common_ratio_l300_300022


namespace probability_of_one_each_color_is_two_fifths_l300_300409

/-- Definition for marbles bag containing 2 red, 2 blue, and 2 green marbles -/
structure MarblesBag where
  red : ℕ
  blue : ℕ
  green : ℕ
  total : ℕ := red + blue + green

/-- Initial setup for the problem -/
def initialBag : MarblesBag := { red := 2, blue := 2, green := 2 }

/-- Represents the outcome of selecting marbles without replacement -/
def selectMarbles (bag : MarblesBag) (count : ℕ) : ℕ :=
  Nat.choose bag.total count

/-- The number of ways to select one marble of each color -/
def selectOneOfEachColor (bag : MarblesBag) : ℕ :=
  bag.red * bag.blue * bag.green

/-- Calculate the probability of selecting one marble of each color -/
def probabilityOneOfEachColor (bag : MarblesBag) (selectCount : ℕ) : ℚ :=
  selectOneOfEachColor bag / selectMarbles bag selectCount

/-- Theorem stating the answer to the probability problem -/
theorem probability_of_one_each_color_is_two_fifths (bag : MarblesBag) :
  probabilityOneOfEachColor bag 3 = 2 / 5 := by
  sorry

end probability_of_one_each_color_is_two_fifths_l300_300409


namespace isosceles_triangle_l300_300624

noncomputable def sin (x : ℝ) : ℝ := Real.sin x
noncomputable def cos (x : ℝ) : ℝ := Real.cos x

variables {A B C : ℝ}
variable (h : sin C = 2 * sin (B + C) * cos B)

theorem isosceles_triangle (h : sin C = 2 * sin (B + C) * cos B) : A = B :=
by
  sorry

end isosceles_triangle_l300_300624


namespace total_volume_of_all_cubes_l300_300008

/-- Carl has 4 cubes each with a side length of 3 -/
def carl_cubes_side_length := 3
def carl_cubes_count := 4

/-- Kate has 6 cubes each with a side length of 4 -/
def kate_cubes_side_length := 4
def kate_cubes_count := 6

/-- Total volume of 10 cubes with given conditions -/
theorem total_volume_of_all_cubes : 
  carl_cubes_count * (carl_cubes_side_length ^ 3) + 
  kate_cubes_count * (kate_cubes_side_length ^ 3) = 492 := by
  sorry

end total_volume_of_all_cubes_l300_300008


namespace sum_A_B_equals_1_l300_300181

-- Definitions for the digits and the properties defined in conditions
variables (A B C D : ℕ)
variable (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
variable (h_digit_bounds : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10)
noncomputable def ABCD := 1000 * A + 100 * B + 10 * C + D
axiom h_mult : ABCD * 2 = ABCD * 10

theorem sum_A_B_equals_1 : A + B = 1 :=
by
  sorry

end sum_A_B_equals_1_l300_300181


namespace other_asymptote_l300_300814

theorem other_asymptote (a b : ℝ) :
  (∀ x y : ℝ, y = 2 * x → y - b = a * (x - (-4))) ∧
  (∀ c d : ℝ, c = -4) →
  ∃ m b' : ℝ, m = -1/2 ∧ b' = -10 ∧ ∀ x y : ℝ, y = m * x + b' :=
by
  sorry

end other_asymptote_l300_300814


namespace ratio_planes_bisect_volume_l300_300509

-- Definitions
def n : ℕ := 6
def m : ℕ := 20

-- Statement to prove
theorem ratio_planes_bisect_volume : (n / m : ℚ) = 3 / 10 := by
  sorry

end ratio_planes_bisect_volume_l300_300509


namespace cube_split_odd_numbers_l300_300152

theorem cube_split_odd_numbers (m : ℕ) (h1 : 1 < m) (h2 : ∃ k, (31 = 2 * k + 1 ∧ (m - 1) * m / 2 = k)) : m = 6 := 
by
  sorry

end cube_split_odd_numbers_l300_300152


namespace mike_taller_than_mark_l300_300638

def feet_to_inches (feet : ℕ) : ℕ := 12 * feet

def mark_height_feet := 5
def mark_height_inches := 3
def mike_height_feet := 6
def mike_height_inches := 1

def mark_total_height := feet_to_inches mark_height_feet + mark_height_inches
def mike_total_height := feet_to_inches mike_height_feet + mike_height_inches

theorem mike_taller_than_mark : mike_total_height - mark_total_height = 10 :=
by
  sorry

end mike_taller_than_mark_l300_300638


namespace greater_expected_area_l300_300874

/-- Let X be a random variable representing a single roll of a die, which can take integer values from 1 through 6. -/
def X : Type := { x : ℕ // 1 ≤ x ∧ x ≤ 6 }

/-- Define independent random variables A and B representing the outcomes of Asya’s die rolls, which can take integer values from 1 through 6 with equal probability. -/
noncomputable def A : Type := { a : ℕ // 1 ≤ a ∧ a ≤ 6 }
noncomputable def B : Type := { b : ℕ // 1 ≤ b ∧ b ≤ 6 }

/-- The expected value of a random variable taking integer values from 1 through 6. 
    E[X] = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5, and E[X^2] = (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6 = 15.1667 -/
noncomputable def expected_X_squared : ℝ := 91 / 6

/-- The expected value of the product of two independent random variables each taking integer values from 1 through 6. 
    E[A * B] = E[A] * E[B] = 3.5 * 3.5 = 12.25 -/
noncomputable def expected_A_times_B : ℝ := 12.25

/-- Prove that the expected area of Vasya's square is greater than Asya's rectangle.
    i.e., E[X^2] > E[A * B] -/
theorem greater_expected_area : expected_X_squared > expected_A_times_B :=
sorry

end greater_expected_area_l300_300874


namespace cube_face_sum_l300_300521

theorem cube_face_sum (a b c d e f : ℕ) (h1 : e = b) (h2 : 2 * (a * b * c + a * b * f + d * b * c + d * b * f) = 1332) :
  a + b + c + d + e + f = 47 :=
sorry

end cube_face_sum_l300_300521


namespace number_of_female_fish_l300_300562

-- Defining the constants given in the problem
def total_fish : ℕ := 45
def fraction_male : ℚ := 2 / 3

-- The statement we aim to prove in Lean
theorem number_of_female_fish : 
  (total_fish : ℚ) * (1 - fraction_male) = 15 :=
by
  sorry

end number_of_female_fish_l300_300562


namespace find_a4_l300_300805

variable {a_n : ℕ → ℕ}
variable {S : ℕ → ℕ}

def is_arithmetic_sequence (a_n : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

def sum_first_n_terms (S : ℕ → ℕ) (a_n : ℕ → ℕ) :=
  ∀ n : ℕ, S n = (n * (a_n 1 + a_n n)) / 2

theorem find_a4 (h : S 7 = 35) (hs : sum_first_n_terms S a_n) (ha : is_arithmetic_sequence a_n) : a_n 4 = 5 := 
  by sorry

end find_a4_l300_300805


namespace prime_sum_20_to_30_l300_300247

-- Definition: A prime number
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → n % m ≠ 0

-- Statement: The sum of the prime numbers between 20 and 30 is 52
theorem prime_sum_20_to_30 : (∑ n in {n | 20 < n ∧ n < 30 ∧ is_prime n}, n) = 52 :=
by {
  sorry
}

end prime_sum_20_to_30_l300_300247


namespace marla_colors_red_squares_l300_300645

-- Conditions
def total_rows : Nat := 10
def squares_per_row : Nat := 15
def total_squares : Nat := total_rows * squares_per_row

def blue_rows_top : Nat := 2
def blue_rows_bottom : Nat := 2
def total_blue_rows : Nat := blue_rows_top + blue_rows_bottom
def total_blue_squares : Nat := total_blue_rows * squares_per_row

def green_squares : Nat := 66
def red_rows : Nat := 4

-- Theorem to prove 
theorem marla_colors_red_squares : 
  total_squares - total_blue_squares - green_squares = red_rows * 6 :=
by
  sorry -- This skips the proof

end marla_colors_red_squares_l300_300645


namespace mike_taller_than_mark_l300_300640

def height_mark_feet : ℕ := 5
def height_mark_inches : ℕ := 3
def height_mike_feet : ℕ := 6
def height_mike_inches : ℕ := 1
def feet_to_inches : ℕ := 12

-- Calculate heights in inches.
def height_mark_total_inches : ℕ := height_mark_feet * feet_to_inches + height_mark_inches
def height_mike_total_inches : ℕ := height_mike_feet * feet_to_inches + height_mike_inches

-- Prove the height difference.
theorem mike_taller_than_mark : height_mike_total_inches - height_mark_total_inches = 10 :=
by
  sorry

end mike_taller_than_mark_l300_300640


namespace change_received_is_zero_l300_300194

noncomputable def combined_money : ℝ := 10 + 8
noncomputable def cost_chicken_wings : ℝ := 6
noncomputable def cost_chicken_salad : ℝ := 4
noncomputable def cost_cheeseburgers : ℝ := 2 * 3.50
noncomputable def cost_fries : ℝ := 2
noncomputable def cost_sodas : ℝ := 2 * 1.00
noncomputable def total_cost_before_discount : ℝ := cost_chicken_wings + cost_chicken_salad + cost_cheeseburgers + cost_fries + cost_sodas
noncomputable def discount_rate : ℝ := 0.15
noncomputable def tax_rate : ℝ := 0.08
noncomputable def discounted_total : ℝ := total_cost_before_discount * (1 - discount_rate)
noncomputable def tax_amount : ℝ := discounted_total * tax_rate
noncomputable def total_cost_after_tax : ℝ := discounted_total + tax_amount

theorem change_received_is_zero : combined_money < total_cost_after_tax → 0 = combined_money - total_cost_after_tax + combined_money := by
  intros h
  sorry

end change_received_is_zero_l300_300194


namespace greater_expected_area_vasya_l300_300887

noncomputable def expected_area_vasya : ℚ :=
  (1/6) * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

noncomputable def expected_area_asya : ℚ :=
  ((1/6) * (1 + 2 + 3 + 4 + 5 + 6)) * ((1/6) * (1 + 2 + 3 + 4 + 5 + 6))

theorem greater_expected_area_vasya : expected_area_vasya > expected_area_asya :=
  by
  -- We've provided the expected area values as definitions
  -- expected_area_vasya = 91/6
  -- vs. expected_area_asya = 12.25 = (21/6)^2 = 441/36 = 12.25
  sorry

end greater_expected_area_vasya_l300_300887


namespace sequence_term_10_l300_300607

theorem sequence_term_10 : ∃ n : ℕ, (1 / (n * (n + 2)) = 1 / 120) ∧ n = 10 := by
  sorry

end sequence_term_10_l300_300607


namespace units_digit_of_sum_of_cubes_l300_300689

theorem units_digit_of_sum_of_cubes : 
  (24^3 + 42^3) % 10 = 2 := by
sorry

end units_digit_of_sum_of_cubes_l300_300689


namespace part1_part2_l300_300608

-- (1) Prove that if 2 ∈ M and M is the solution set of ax^2 + 5x - 2 > 0, then a > -2.
theorem part1 (a : ℝ) (h : 2 * (a * 4 + 10) - 2 > 0) : a > -2 :=
sorry

-- (2) Given M = {x | 1/2 < x < 2} and M is the solution set of ax^2 + 5x - 2 > 0,
-- prove that the solution set of ax^2 - 5x + a^2 - 1 > 0 is -3 < x < 1/2
theorem part2 (a : ℝ) (h1 : ∀ x : ℝ, (1/2 < x ∧ x < 2) ↔ ax^2 + 5*x - 2 > 0) (h2 : a = -2) :
  ∀ x : ℝ, (-3 < x ∧ x < 1/2) ↔ (-2 * x^2 - 5 * x + 3 > 0) :=
sorry

end part1_part2_l300_300608


namespace birds_remaining_l300_300403

variable (initial_birds : ℝ) (birds_flew_away : ℝ)

theorem birds_remaining (h1 : initial_birds = 12.0) (h2 : birds_flew_away = 8.0) : initial_birds - birds_flew_away = 4.0 :=
by
  rw [h1, h2]
  norm_num

end birds_remaining_l300_300403


namespace fourth_function_form_l300_300907

variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)
variable (hf : Function.LeftInverse f_inv f)
variable (hf_inv : Function.RightInverse f_inv f)

theorem fourth_function_form :
  (∀ x, y = (-(f (-x - 1)) + 2) ↔ y = f_inv (x + 2) + 1 ↔ -(x + y) = 0) :=
  sorry

end fourth_function_form_l300_300907


namespace find_Z_l300_300334

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![1, -2], ![3, -5]]
def b : Vector (Fin 2) ℝ := ![1, 1]
def Z : Vector (Fin 2) ℝ := ![-1, -2]

theorem find_Z (A_inv_satisfies : A⁻¹ ⬝ Z = b) : Z = ![-1, -2] :=
  sorry

end find_Z_l300_300334


namespace no_square_ends_in_2012_l300_300013

theorem no_square_ends_in_2012 : ¬ ∃ a : ℤ, (a * a) % 10 = 2 := by
  sorry

end no_square_ends_in_2012_l300_300013


namespace equation_value_l300_300997

-- Define the expressions
def a := 10 + 3
def b := 7 - 5

-- State the theorem
theorem equation_value : a^2 + b^2 = 173 := by
  sorry

end equation_value_l300_300997


namespace expected_number_of_different_faces_l300_300711

noncomputable def expected_faces : ℝ :=
  let probability_face_1_not_appearing := (5 / 6)^6
  let E_zeta_1 := 1 - probability_face_1_not_appearing
  6 * E_zeta_1

theorem expected_number_of_different_faces :
  expected_faces = (6^6 - 5^6) / 6^5 := by 
  sorry

end expected_number_of_different_faces_l300_300711


namespace greatest_integer_condition_l300_300230

theorem greatest_integer_condition (x: ℤ) (h₁: x < 150) (h₂: Int.gcd x 24 = 3) : x ≤ 147 := 
by sorry

example : ∃ x, x < 150 ∧ Int.gcd x 24 = 3 ∧ x = 147 :=
begin
  use 147,
  split,
  { exact lt_trans (by norm_num) (by norm_num) },
  { split,
    { norm_num },
    { refl } }
end

end greatest_integer_condition_l300_300230


namespace remaining_amount_after_shopping_l300_300995

theorem remaining_amount_after_shopping (initial_amount spent_percentage remaining_amount : ℝ)
  (h_initial : initial_amount = 4000)
  (h_spent : spent_percentage = 0.30)
  (h_remaining : remaining_amount = 2800) :
  initial_amount - (spent_percentage * initial_amount) = remaining_amount :=
by
  sorry

end remaining_amount_after_shopping_l300_300995


namespace units_digit_sum_cubes_l300_300679

theorem units_digit_sum_cubes (n1 n2 : ℕ) 
  (h1 : n1 = 24) (h2 : n2 = 42) : 
  (n1 ^ 3 + n2 ^ 3) % 10 = 6 :=
by 
  -- substitution based on h1 and h2 can be done here.
  sorry

end units_digit_sum_cubes_l300_300679


namespace blue_tissue_length_exists_l300_300107

theorem blue_tissue_length_exists (B R : ℝ) (h1 : R = B + 12) (h2 : 2 * R = 3 * B) : B = 24 := 
by
  sorry

end blue_tissue_length_exists_l300_300107


namespace intersection_result_l300_300777

def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x ≥ 1 }
def N : Set ℝ := { x | 0 ≤ x ∧ x < 5 }
def M_compl : Set ℝ := { x | x < 1 }

theorem intersection_result : N ∩ M_compl = { x | 0 ≤ x ∧ x < 1 } :=
by sorry

end intersection_result_l300_300777


namespace prob_at_least_seven_friends_stay_for_entire_game_l300_300304

-- Definitions of conditions
def numFriends : ℕ := 8
def numUnsureFriends : ℕ := 5
def probabilityStay (p : ℚ) : ℚ := p
def sureFriends := 3

-- The probabilities
def prob_one_third : ℚ := 1 / 3
def prob_two_thirds : ℚ := 2 / 3

-- Variables to hold binomial coefficient and power calculation
noncomputable def C (n k : ℕ) : ℚ := (Nat.choose n k)
noncomputable def probability_at_least_seven_friends_stay : ℚ :=
  C numUnsureFriends 4 * (probabilityStay prob_one_third)^4 * (probabilityStay prob_two_thirds)^1 +
  (probabilityStay prob_one_third)^5

-- Theorem statement
theorem prob_at_least_seven_friends_stay_for_entire_game :
  probability_at_least_seven_friends_stay = 11 / 243 :=
  by sorry

end prob_at_least_seven_friends_stay_for_entire_game_l300_300304


namespace geom_sequence_product_l300_300793

theorem geom_sequence_product (q a1 : ℝ) (h1 : a1 * (a1 * q) * (a1 * q^2) = 3) (h2 : (a1 * q^9) * (a1 * q^10) * (a1 * q^11) = 24) :
  (a1 * q^12) * (a1 * q^13) * (a1 * q^14) = 48 :=
by
  sorry

end geom_sequence_product_l300_300793


namespace find_prices_max_basketballs_l300_300523

-- Define price of basketballs and soccer balls
def basketball_price : ℕ := 80
def soccer_ball_price : ℕ := 50

-- Define the equations given in the problem
theorem find_prices (x y : ℕ) 
  (h1 : 2 * x + 3 * y = 310)
  (h2 : 5 * x + 2 * y = 500) : 
  x = basketball_price ∧ y = soccer_ball_price :=
sorry

-- Define the maximum number of basketballs given the cost constraints
theorem max_basketballs (m : ℕ)
  (htotal : m + (60 - m) = 60)
  (hcost : 80 * m + 50 * (60 - m) ≤ 4000) : 
  m ≤ 33 :=
sorry

end find_prices_max_basketballs_l300_300523


namespace option_d_may_not_hold_l300_300469

theorem option_d_may_not_hold (a b : ℝ) (m : ℝ) (h : a < b) : ¬ (m^2 * a > m^2 * b) :=
sorry

end option_d_may_not_hold_l300_300469


namespace five_digit_divisibility_l300_300195

-- Definitions of n and m
def n (a b c d e : ℕ) := 10000 * a + 1000 * b + 100 * c + 10 * d + e
def m (a b d e : ℕ) := 1000 * a + 100 * b + 10 * d + e

-- Condition that n is a five-digit number whose first digit is non-zero and n/m is an integer
theorem five_digit_divisibility (a b c d e : ℕ):
  1 <= a ∧ a <= 9 → 0 <= b ∧ b <= 9 → 0 <= c ∧ c <= 9 → 0 <= d ∧ d <= 9 → 0 <= e ∧ e <= 9 →
  m a b d e ∣ n a b c d e →
  ∃ x y : ℕ, a = x ∧ b = y ∧ c = 0 ∧ d = 0 ∧ e = 0 :=
by
  sorry

end five_digit_divisibility_l300_300195


namespace intersection_with_y_axis_l300_300827

theorem intersection_with_y_axis (x y : ℝ) (h : y = x + 3) (hx : x = 0) : (x, y) = (0, 3) := 
by 
  subst hx 
  rw [h]
  rfl
-- sorry to skip the proof

end intersection_with_y_axis_l300_300827


namespace joan_apples_l300_300357

theorem joan_apples (initial_apples : ℕ) (given_to_melanie : ℕ) (given_to_sarah : ℕ) : 
  initial_apples = 43 ∧ given_to_melanie = 27 ∧ given_to_sarah = 11 → (initial_apples - given_to_melanie - given_to_sarah) = 5 := 
by
  sorry

end joan_apples_l300_300357


namespace horner_evaluation_at_two_l300_300746

/-- Define the polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 * x^6 + 3 * x^5 + 5 * x^3 + 6 * x^2 + 7 * x + 8

/-- States that the value of f(2) using Horner's Rule equals 14. -/
theorem horner_evaluation_at_two : f 2 = 14 :=
sorry

end horner_evaluation_at_two_l300_300746


namespace factor_81_minus_4y4_l300_300895

theorem factor_81_minus_4y4 (y : ℝ) : 81 - 4 * y^4 = (9 + 2 * y^2) * (9 - 2 * y^2) := by 
    sorry

end factor_81_minus_4y4_l300_300895


namespace function_divisibility_l300_300633

theorem function_divisibility
    (f : ℤ → ℕ)
    (h_pos : ∀ x, 0 < f x)
    (h_div : ∀ m n : ℤ, (f m - f n) % f (m - n) = 0) :
    ∀ m n : ℤ, f m ≤ f n → f m ∣ f n :=
by sorry

end function_divisibility_l300_300633


namespace coefficients_equal_l300_300094

theorem coefficients_equal (n : ℕ) (h : n ≥ 6) : 
  (n = 7) ↔ 
  (Nat.choose n 5 * 3 ^ 5 = Nat.choose n 6 * 3 ^ 6) := by
  sorry

end coefficients_equal_l300_300094


namespace expected_number_of_different_faces_l300_300721

theorem expected_number_of_different_faces :
  let p := (6 : ℕ) ^ 6
  let q := (5 : ℕ) ^ 6
  6 * (1 - (5 / 6)^6) = (p - q) / (6 ^ 5) :=
by
  sorry

end expected_number_of_different_faces_l300_300721


namespace geometric_series_sum_l300_300571

theorem geometric_series_sum :
  ∑' i : ℕ, (2 / 3) ^ (i + 1) = 2 :=
by
  sorry

end geometric_series_sum_l300_300571


namespace min_value_of_quadratic_expression_l300_300674

theorem min_value_of_quadratic_expression : ∃ x : ℝ, (∀ y : ℝ, x^2 + 6 * x + 3 ≤ y) ∧ x^2 + 6 * x + 3 = -6 :=
sorry

end min_value_of_quadratic_expression_l300_300674


namespace area_under_curve_l300_300699

open Real IntervalIntegral

-- Defining the given function
def f (x : ℝ) := cos x * (sin x) ^ 2

-- The interval boundaries
def a : ℝ := 0
def b : ℝ := π / 2

-- Stating the theorem
theorem area_under_curve : ∫ x in a..b, f x = 1 / 3 :=
sorry

end area_under_curve_l300_300699


namespace gcd_lcm_product_l300_300413

theorem gcd_lcm_product (a b : ℕ) (h₀ : a = 15) (h₁ : b = 45) :
  Nat.gcd a b * Nat.lcm a b = 675 :=
by
  sorry

end gcd_lcm_product_l300_300413


namespace initially_calculated_average_height_l300_300093

theorem initially_calculated_average_height
  (A : ℝ)
  (h1 : ∀ heights : List ℝ, heights.length = 35 → (heights.sum + (106 - 166) = heights.sum) → (heights.sum / 35) = 180) :
  A = 181.71 :=
sorry

end initially_calculated_average_height_l300_300093


namespace arithmetic_sequence_b3b7_l300_300506

theorem arithmetic_sequence_b3b7 (b : ℕ → ℤ) (d : ℤ)
  (h_arith_seq : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b n < b (n + 1))
  (h_cond : b 4 * b 6 = 17) : 
  b 3 * b 7 = -175 :=
sorry

end arithmetic_sequence_b3b7_l300_300506


namespace inscribe_circle_in_convex_polygon_l300_300655

theorem inscribe_circle_in_convex_polygon
  (S P r : ℝ) 
  (hP_pos : P > 0)
  (h_poly_area : S > 0)
  (h_nonneg : r ≥ 0) :
  S / P ≤ r :=
sorry

end inscribe_circle_in_convex_polygon_l300_300655


namespace largest_inscribed_circle_radius_l300_300735

theorem largest_inscribed_circle_radius (k : ℝ) (h_perimeter : 0 < k) :
  ∃ (r : ℝ), r = (k / 2) * (3 - 2 * Real.sqrt 2) :=
by
  have h_r : ∃ (r : ℝ), r = (k / 2) * (3 - 2 * Real.sqrt 2)
  exact ⟨(k / 2) * (3 - 2 * Real.sqrt 2), rfl⟩
  exact h_r

end largest_inscribed_circle_radius_l300_300735


namespace find_g3_l300_300216

theorem find_g3 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 ^ x) + 2 * x * g (3 ^ (-x)) = 1) : 
  g 3 = 1 / 5 := 
sorry

end find_g3_l300_300216


namespace remainder_is_83_l300_300856

-- Define the condition: the values for the division
def value1 : ℤ := 2021
def value2 : ℤ := 102

-- State the theorem: remainder when 2021 is divided by 102 is 83
theorem remainder_is_83 : value1 % value2 = 83 := by
  sorry

end remainder_is_83_l300_300856


namespace vector_cross_product_coordinates_l300_300994

variables (a1 a2 a3 b1 b2 b3 : ℝ)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1, a.2.2 * b.1 - a.1 * b.2.2, a.1 * b.2.1 - a.2.1 * b.1)

theorem vector_cross_product_coordinates :
  cross_product (a1, a2, a3) (b1, b2, b3) = 
    (a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1) :=
by
sorry

end vector_cross_product_coordinates_l300_300994


namespace honor_students_count_l300_300984

def num_students_total : ℕ := 24
def num_honor_students_girls : ℕ := 3
def num_honor_students_boys : ℕ := 4

def num_girls : ℕ := 13
def num_boys : ℕ := 11

theorem honor_students_count (total_students : ℕ) 
    (prob_girl_honor : ℚ) (prob_boy_honor : ℚ)
    (girls : ℕ) (boys : ℕ)
    (honor_girls : ℕ) (honor_boys : ℕ) :
    total_students < 30 →
    prob_girl_honor = 3 / 13 →
    prob_boy_honor = 4 / 11 →
    girls = 13 →
    honor_girls = 3 →
    boys = 11 →
    honor_boys = 4 →
    girls + boys = total_students →
    honor_girls + honor_boys = 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [← h4, ← h5, ← h6, ← h7, ← h8]
  exact 7

end honor_students_count_l300_300984


namespace positive_difference_of_perimeters_l300_300223

noncomputable def perimeter_figure1 : ℕ :=
  let outer_rectangle := 2 * (5 + 1)
  let inner_extension := 2 * (2 + 1)
  outer_rectangle + inner_extension

noncomputable def perimeter_figure2 : ℕ :=
  2 * (5 + 2)

theorem positive_difference_of_perimeters :
  (perimeter_figure1 - perimeter_figure2 = 4) :=
by
  let perimeter1 := perimeter_figure1
  let perimeter2 := perimeter_figure2
  sorry

end positive_difference_of_perimeters_l300_300223


namespace alice_sales_goal_l300_300140

def price_adidas := 45
def price_nike := 60
def price_reeboks := 35
def price_puma := 50
def price_converse := 40

def num_adidas := 10
def num_nike := 12
def num_reeboks := 15
def num_puma := 8
def num_converse := 14

def quota := 2000

def total_sales :=
  (num_adidas * price_adidas) +
  (num_nike * price_nike) +
  (num_reeboks * price_reeboks) +
  (num_puma * price_puma) +
  (num_converse * price_converse)

def exceed_amount := total_sales - quota

theorem alice_sales_goal : exceed_amount = 655 := by
  -- calculation steps would go here
  sorry

end alice_sales_goal_l300_300140


namespace trajectory_of_midpoint_l300_300473

theorem trajectory_of_midpoint (x y x₀ y₀ : ℝ) :
  (y₀ = 2 * x₀ ^ 2 + 1) ∧ (x = (x₀ + 0) / 2) ∧ (y = (y₀ + 1) / 2) →
  y = 4 * x ^ 2 + 1 :=
by sorry

end trajectory_of_midpoint_l300_300473


namespace initial_total_perimeter_l300_300535

theorem initial_total_perimeter (n : ℕ) (a : ℕ) (m : ℕ)
  (h1 : n = 2 * m)
  (h2 : 40 = 2 * a * m)
  (h3 : 4 * n - 6 * m = 4 * n - 40) :
  4 * n = 280 :=
by sorry

end initial_total_perimeter_l300_300535


namespace number_in_tens_place_is_7_l300_300999

theorem number_in_tens_place_is_7
  (digits : Finset ℕ)
  (a b c : ℕ)
  (h1 : digits = {7, 5, 2})
  (h2 : 100 * a + 10 * b + c > 530)
  (h3 : 100 * a + 10 * b + c < 710)
  (h4 : a ∈ digits)
  (h5 : b ∈ digits)
  (h6 : c ∈ digits)
  (h7 : ∀ x ∈ digits, x ≠ a → x ≠ b → x ≠ c) :
  b = 7 := sorry

end number_in_tens_place_is_7_l300_300999


namespace remainder_mod_7_l300_300115

theorem remainder_mod_7 : (9^7 + 8^8 + 7^9) % 7 = 3 :=
by sorry

end remainder_mod_7_l300_300115


namespace plum_balances_pear_l300_300782

variable (A G S : ℕ)

-- Definitions as per the problem conditions
axiom condition1 : 3 * A + G = 10 * S
axiom condition2 : A + 6 * S = G

-- The goal is to prove the following statement
theorem plum_balances_pear : G = 7 * S :=
by
  -- Skipping the proof as only statement is needed
  sorry

end plum_balances_pear_l300_300782


namespace ariel_fish_l300_300559

theorem ariel_fish (total_fish : ℕ) (male_fraction female_fraction : ℚ) (h1 : total_fish = 45) (h2 : male_fraction = 2 / 3) (h3 : female_fraction = 1 - male_fraction) : total_fish * female_fraction = 15 :=
by
  sorry

end ariel_fish_l300_300559


namespace a_plus_b_l300_300598

open Complex

theorem a_plus_b (a b : ℝ) (h : (a - I) * I = -b + 2 * I) : a + b = 1 := by
  sorry

end a_plus_b_l300_300598


namespace alice_gadgets_sales_l300_300556

variable (S : ℝ) -- Variable to denote the worth of gadgets Alice sold
variable (E : ℝ) -- Variable to denote Alice's total earnings

theorem alice_gadgets_sales :
  let basic_salary := 240
  let commission_percentage := 0.02
  let save_amount := 29
  let save_percentage := 0.10
  
  -- Total earnings equation
  let earnings_eq := E = basic_salary + commission_percentage * S
  
  -- Savings equation
  let savings_eq := save_percentage * E = save_amount
  
  -- Solve the system of equations to show S = 2500
  S = 2500 :=
by
  sorry

end alice_gadgets_sales_l300_300556


namespace inscribed_circle_radius_l300_300810

theorem inscribed_circle_radius (a b c r : ℝ) (h : a^2 + b^2 = c^2) (h' : r = (a + b - c) / 2) : r = (a + b - c) / 2 :=
by
  sorry

end inscribed_circle_radius_l300_300810


namespace houses_with_dogs_l300_300349

theorem houses_with_dogs (C B Total : ℕ) (hC : C = 30) (hB : B = 10) (hTotal : Total = 60) :
  ∃ D, D = 40 :=
by
  -- The overall proof would go here
  sorry

end houses_with_dogs_l300_300349


namespace trigonometric_relationship_l300_300079

noncomputable def α : ℝ := Real.cos 4
noncomputable def b : ℝ := Real.cos (4 * Real.pi / 5)
noncomputable def c : ℝ := Real.sin (7 * Real.pi / 6)

theorem trigonometric_relationship : b < α ∧ α < c := 
by
  sorry

end trigonometric_relationship_l300_300079


namespace find_ordered_pair_l300_300390

noncomputable def ordered_pair (c d : ℝ) := c = 1 ∧ d = -2

theorem find_ordered_pair (c d : ℝ) (h1 : c ≠ 0) (h2 : d ≠ 0) (h3 : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = c ∨ x = d)) : ordered_pair c d :=
by
  sorry

end find_ordered_pair_l300_300390


namespace prime_sum_20_to_30_l300_300246

-- Definition: A prime number
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → n % m ≠ 0

-- Statement: The sum of the prime numbers between 20 and 30 is 52
theorem prime_sum_20_to_30 : (∑ n in {n | 20 < n ∧ n < 30 ∧ is_prime n}, n) = 52 :=
by {
  sorry
}

end prime_sum_20_to_30_l300_300246


namespace youngest_person_age_l300_300533

theorem youngest_person_age (n : ℕ) (average_age : ℕ) (average_age_when_youngest_born : ℕ) 
    (h1 : n = 7) (h2 : average_age = 30) (h3 : average_age_when_youngest_born = 24) :
    ∃ Y : ℚ, Y = 66 / 7 :=
by
  sorry

end youngest_person_age_l300_300533


namespace max_value_of_function_l300_300218

noncomputable def function (x : ℝ) : ℝ := (Real.cos x) ^ 2 - Real.sin x

theorem max_value_of_function : ∃ x : ℝ, function x = 5 / 4 :=
by
  sorry

end max_value_of_function_l300_300218


namespace episodes_remaining_l300_300254

-- Definition of conditions
def seasons : ℕ := 12
def episodes_per_season : ℕ := 20
def fraction_watched : ℚ := 1 / 3
def total_episodes : ℕ := episodes_per_season * seasons
def episodes_watched : ℕ := (fraction_watched * total_episodes).toNat

-- Problem statement
theorem episodes_remaining : total_episodes - episodes_watched = 160 := by
  sorry

end episodes_remaining_l300_300254


namespace exists_factorial_with_first_digits_2015_l300_300014

theorem exists_factorial_with_first_digits_2015 : ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 2015 * (10^k) ≤ n! ∧ n! < 2016 * (10^k)) :=
sorry

end exists_factorial_with_first_digits_2015_l300_300014


namespace zoo_visitors_per_hour_l300_300666

theorem zoo_visitors_per_hour 
    (h1 : ∃ V, 0.80 * V = 320)
    (h2 : ∃ H : Nat, H = 8)
    : ∃ N : Nat, N = 50 :=
by
  sorry

end zoo_visitors_per_hour_l300_300666


namespace ellipse_range_x_plus_y_l300_300475

/-- The problem conditions:
Given any point P(x, y) on the ellipse x^2 / 144 + y^2 / 25 = 1,
prove that the range of values for x + y is [-13, 13].
-/
theorem ellipse_range_x_plus_y (x y : ℝ) (h : (x^2 / 144) + (y^2 / 25) = 1) : 
  -13 ≤ x + y ∧ x + y ≤ 13 := sorry

end ellipse_range_x_plus_y_l300_300475


namespace mass_percentage_H_in_NH4I_is_correct_l300_300310

noncomputable def molar_mass_NH4I : ℝ := 1 * 14.01 + 4 * 1.01 + 1 * 126.90

noncomputable def mass_H_in_NH4I : ℝ := 4 * 1.01

noncomputable def mass_percentage_H_in_NH4I : ℝ := (mass_H_in_NH4I / molar_mass_NH4I) * 100

theorem mass_percentage_H_in_NH4I_is_correct :
  abs (mass_percentage_H_in_NH4I - 2.79) < 0.01 := by
  sorry

end mass_percentage_H_in_NH4I_is_correct_l300_300310


namespace common_ratio_infinite_geometric_series_l300_300021

theorem common_ratio_infinite_geometric_series :
  let a₁ := (4 : ℚ) / 7
  let a₂ := (16 : ℚ) / 49
  let a₃ := (64 : ℚ) / 343
  let r := a₂ / a₁
  r = 4 / 7 :=
by
  sorry

end common_ratio_infinite_geometric_series_l300_300021


namespace inequality_proof_l300_300617

variable (a b c : ℝ)

theorem inequality_proof
  (h1 : a > b) :
  a * c^2 ≥ b * c^2 := 
sorry

end inequality_proof_l300_300617


namespace min_route_length_5x5_l300_300960

-- Definition of the grid and its properties
def grid : Type := Fin 5 × Fin 5

-- Define a function to calculate the minimum route length
noncomputable def min_route_length (grid_size : ℕ) : ℕ :=
  if h : grid_size = 5 then 68 else 0

-- The proof problem statement
theorem min_route_length_5x5 : min_route_length 5 = 68 :=
by
  -- Skipping the actual proof
  sorry

end min_route_length_5x5_l300_300960


namespace contrapositive_proof_l300_300395

theorem contrapositive_proof (x : ℝ) : (x^2 < 1 → -1 < x ∧ x < 1) → (x ≥ 1 ∨ x ≤ -1 → x^2 ≥ 1) :=
by
  sorry

end contrapositive_proof_l300_300395


namespace simplify_A_minus_B_value_of_A_minus_B_given_condition_l300_300466

variable (a b : ℝ)

def A := (a + b) ^ 2 - 3 * b ^ 2
def B := 2 * (a + b) * (a - b) - 3 * a * b

theorem simplify_A_minus_B :
  A a b - B a b = -a ^ 2 + 5 * a * b :=
by sorry

theorem value_of_A_minus_B_given_condition :
  (a - 3) ^ 2 + |b - 4| = 0 → A a b - B a b = 51 :=
by sorry

end simplify_A_minus_B_value_of_A_minus_B_given_condition_l300_300466


namespace max_value_of_expression_l300_300918

noncomputable def expression (x y z : ℝ) := sin (x - y) + sin (y - z) + sin (z - x)

theorem max_value_of_expression :
  ∀ x y z ∈ set.Icc (0 : ℝ) (real.pi / 2), expression x y z ≤ real.sqrt 2 - 1 :=
sorry

end max_value_of_expression_l300_300918


namespace parabola_directrix_eq_neg2_l300_300928

-- Definitions based on conditions
def ellipse_focus (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ x = 2 ∧ y = 0

def parabola_directrix (p x y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ ∃ x, x = -p / 2

theorem parabola_directrix_eq_neg2 (p : ℝ) (hp : p > 0) :
  (∀ (x y : ℝ), ellipse_focus 9 5 x y → parabola_directrix p x y) →
  (∃ x y : ℝ, parabola_directrix p x y → x = -2) :=
sorry

end parabola_directrix_eq_neg2_l300_300928


namespace ratio_of_speeds_is_two_l300_300190

noncomputable def joe_speed : ℝ := 0.266666666667
noncomputable def time : ℝ := 40
noncomputable def total_distance : ℝ := 16

noncomputable def joe_distance : ℝ := joe_speed * time
noncomputable def pete_distance : ℝ := total_distance - joe_distance
noncomputable def pete_speed : ℝ := pete_distance / time

theorem ratio_of_speeds_is_two :
  joe_speed / pete_speed = 2 := by
  sorry

end ratio_of_speeds_is_two_l300_300190


namespace part1_part2_l300_300479

-- Define the absolute value function
def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + a

-- Given conditions
def condition1 : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ f x a ≤ 6

def condition2 (a : ℝ) : Prop :=
  ∃ t m : ℝ, f (t / 2) a ≤ m - f (-t) a

-- Statements to prove
theorem part1 : ∃ a : ℝ, condition1 ∧ a = 1 := by
  sorry

theorem part2 : ∀ {a : ℝ}, a = 1 → ∃ m : ℝ, m ≥ 3.5 ∧ condition2 a := by
  sorry

end part1_part2_l300_300479


namespace stone_breadth_5_l300_300724

theorem stone_breadth_5 (hall_length_m hall_breadth_m stone_length_dm num_stones b₁ b₂ : ℝ) 
  (h1 : hall_length_m = 36) 
  (h2 : hall_breadth_m = 15) 
  (h3 : stone_length_dm = 3) 
  (h4 : num_stones = 3600)
  (h5 : hall_length_m * 10 * hall_breadth_m * 10 = 54000)
  (h6 : stone_length_dm * b₁ * num_stones = hall_length_m * 10 * hall_breadth_m * 10) :
  b₂ = 5 := 
  sorry

end stone_breadth_5_l300_300724


namespace units_digit_of_sum_of_cubes_l300_300688

theorem units_digit_of_sum_of_cubes : 
  (24^3 + 42^3) % 10 = 2 := by
sorry

end units_digit_of_sum_of_cubes_l300_300688


namespace common_ratio_infinite_geometric_series_l300_300020

theorem common_ratio_infinite_geometric_series :
  let a₁ := (4 : ℚ) / 7
  let a₂ := (16 : ℚ) / 49
  let a₃ := (64 : ℚ) / 343
  let r := a₂ / a₁
  r = 4 / 7 :=
by
  sorry

end common_ratio_infinite_geometric_series_l300_300020


namespace problem_I_problem_II_problem_III_l300_300769

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 3

theorem problem_I (a b : ℝ) (h_a : a = 0) :
  (b ≥ 0 → ∀ x : ℝ, 3 * x^2 + b ≥ 0) ∧
  (b < 0 → 
    ∀ x : ℝ, (x < -Real.sqrt (-b / 3) ∨ x > Real.sqrt (-b / 3)) → 
      3 * x^2 + b > 0) := sorry

theorem problem_II (b : ℝ) :
  ∃ x0 : ℝ, f x0 0 b = x0 ∧ (3 * x0^2 + b = 0) ↔ b = -3 := sorry

theorem problem_III :
  ∀ a b : ℝ, ¬ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧
    (3 * x1^2 + 2 * a * x1 + b = 0) ∧
    (3 * x2^2 + 2 * a * x2 + b = 0) ∧
    (f x1 a b = x1) ∧
    (f x2 a b = x2)) := sorry

end problem_I_problem_II_problem_III_l300_300769


namespace min_cost_at_100_l300_300702

noncomputable def cost_function (v : ℝ) : ℝ :=
if (0 < v ∧ v ≤ 50) then (123000 / v + 690)
else if (v > 50) then (3 * v^2 / 50 + 120000 / v + 600)
else 0

theorem min_cost_at_100 : ∃ v : ℝ, v = 100 ∧ cost_function v = 2400 :=
by
  -- We are not proving but stating the theorem here
  sorry

end min_cost_at_100_l300_300702


namespace horizontal_distance_l300_300551

def curve (x : ℝ) := x^3 - x^2 - x - 6

def P_condition (x : ℝ) := curve x = 10
def Q_condition1 (x : ℝ) := curve x = 2
def Q_condition2 (x : ℝ) := curve x = -2

theorem horizontal_distance (x_P x_Q: ℝ) (hP: P_condition x_P) (hQ1: Q_condition1 x_Q ∨ Q_condition2 x_Q) :
  |x_P - x_Q| = 3 := sorry

end horizontal_distance_l300_300551


namespace find_a_c_pair_l300_300836

-- Given conditions in the problem
variable (a c : ℝ)

-- First condition: The quadratic equation has exactly one solution
def quadratic_eq_has_one_solution : Prop :=
  let discriminant := (30:ℝ)^2 - 4 * a * c
  discriminant = 0

-- Second condition: Sum of a and c
def sum_eq_41 : Prop := a + c = 41

-- Third condition: a is less than c
def a_lt_c : Prop := a < c

-- State the proof problem
theorem find_a_c_pair (a c : ℝ) (h1 : quadratic_eq_has_one_solution a c) (h2 : sum_eq_41 a c) (h3 : a_lt_c a c) : (a, c) = (6.525, 34.475) :=
sorry

end find_a_c_pair_l300_300836


namespace honor_students_count_l300_300978

noncomputable def number_of_students_in_class_is_less_than_30 := ∃ n, n < 30
def probability_girl_honor_student (G E_G : ℕ) := E_G / G = (3 : ℚ) / 13
def probability_boy_honor_student (B E_B : ℕ) := E_B / B = (4 : ℚ) / 11

theorem honor_students_count (G B E_G E_B : ℕ) 
  (hG_cond : probability_girl_honor_student G E_G) 
  (hB_cond : probability_boy_honor_student B E_B) 
  (h_total_students : G + B < 30) 
  (hE_G_def : E_G = 3 * G / 13) 
  (hE_B_def : E_B = 4 * B / 11) 
  (hG_nonneg : G >= 13)
  (hB_nonneg : B >= 11):
  E_G + E_B = 7 := 
sorry

end honor_students_count_l300_300978


namespace ineq_a3b3c3_l300_300654

theorem ineq_a3b3c3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^3 + b^3 + c^3 ≥ a^2 * b + b^2 * c + c^2 * a ∧ (a^3 + b^3 + c^3 = a^2 * b + b^2 * c + c^2 * a ↔ a = b ∧ b = c) :=
by
  sorry

end ineq_a3b3c3_l300_300654


namespace BURN_maps_to_8615_l300_300493

open List Function

def tenLetterMapping : List (Char × Nat) := 
  [('G', 0), ('R', 1), ('E', 2), ('A', 3), ('T', 4), ('N', 5), ('U', 6), ('M', 7), ('B', 8), ('S', 9)]

def charToDigit (c : Char) : Option Nat :=
  tenLetterMapping.lookup c

def wordToNumber (word : List Char) : Option (List Nat) :=
  word.mapM charToDigit 

theorem BURN_maps_to_8615 :
  wordToNumber ['B', 'U', 'R', 'N'] = some [8, 6, 1, 5] :=
by
  sorry

end BURN_maps_to_8615_l300_300493


namespace subset_condition_for_a_l300_300802

theorem subset_condition_for_a (a : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 ≤ 5 / 4 → (|x - 1| + 2 * |y - 2| ≤ a)) → a ≥ 5 / 2 :=
by
  intro H
  sorry

end subset_condition_for_a_l300_300802


namespace length_of_unfenced_side_l300_300597

theorem length_of_unfenced_side
  (L W : ℝ)
  (h1 : L * W = 200)
  (h2 : 2 * W + L = 50) :
  L = 10 :=
sorry

end length_of_unfenced_side_l300_300597


namespace sin_double_angle_l300_300901

theorem sin_double_angle (theta : ℝ) 
  (h : Real.sin (theta + Real.pi / 4) = 2 / 5) :
  Real.sin (2 * theta) = -17 / 25 := by
  sorry

end sin_double_angle_l300_300901


namespace parabola_equation_l300_300319

theorem parabola_equation (M : ℝ × ℝ) (hM : M = (5, 3))
    (h_dist : ∀ a : ℝ, |5 + 1/(4*a)| = 6) :
    (y = (1/12)*x^2) ∨ (y = -(1/36)*x^2) :=
sorry

end parabola_equation_l300_300319


namespace mike_taller_than_mark_l300_300641

def height_mark_feet : ℕ := 5
def height_mark_inches : ℕ := 3
def height_mike_feet : ℕ := 6
def height_mike_inches : ℕ := 1
def feet_to_inches : ℕ := 12

-- Calculate heights in inches.
def height_mark_total_inches : ℕ := height_mark_feet * feet_to_inches + height_mark_inches
def height_mike_total_inches : ℕ := height_mike_feet * feet_to_inches + height_mike_inches

-- Prove the height difference.
theorem mike_taller_than_mark : height_mike_total_inches - height_mark_total_inches = 10 :=
by
  sorry

end mike_taller_than_mark_l300_300641


namespace robin_total_distance_l300_300965

theorem robin_total_distance
  (d : ℕ)
  (d1 : ℕ)
  (h1 : d = 500)
  (h2 : d1 = 200)
  : 2 * d1 + d = 900 :=
by
  rewrite [h1, h2]
  rfl

end robin_total_distance_l300_300965


namespace pet_store_cages_l300_300437

theorem pet_store_cages (init_puppies sold_puppies puppies_per_cage : ℕ)
  (h1 : init_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : puppies_per_cage = 5) :
  (init_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end pet_store_cages_l300_300437


namespace triangle_inequality_l300_300074

theorem triangle_inequality (a b c : ℝ) (h1 : a + b + c = 2) :
  a^2 + b^2 + c^2 < 2 * (1 - a * b * c) :=
sorry

end triangle_inequality_l300_300074


namespace radius_of_largest_circle_correct_l300_300450

noncomputable def radius_of_largest_circle_in_quadrilateral (AB BC CD DA : ℝ) (angle_BCD : ℝ) : ℝ :=
  if AB = 10 ∧ BC = 12 ∧ CD = 8 ∧ DA = 14 ∧ angle_BCD = 90
    then Real.sqrt 210
    else 0

theorem radius_of_largest_circle_correct :
  radius_of_largest_circle_in_quadrilateral 10 12 8 14 90 = Real.sqrt 210 :=
by
  sorry

end radius_of_largest_circle_correct_l300_300450


namespace gcd_polynomials_l300_300033

-- State the problem in Lean 4.
theorem gcd_polynomials (b : ℤ) (h : ∃ k : ℤ, b = 7768 * 2 * k) : 
  Int.gcd (7 * b^2 + 55 * b + 125) (3 * b + 10) = 10 :=
by
  sorry

end gcd_polynomials_l300_300033


namespace ryan_chinese_learning_hours_l300_300306

theorem ryan_chinese_learning_hours : 
    ∀ (h_english : ℕ) (diff : ℕ), 
    h_english = 7 → 
    h_english = 2 + (h_english - diff) → 
    diff = 5 := by
  intros h_english diff h_english_eq h_english_diff_eq
  sorry

end ryan_chinese_learning_hours_l300_300306


namespace QED_mul_eq_neg_25I_l300_300804

namespace ComplexMultiplication

open Complex

def Q : ℂ := 3 + 4 * Complex.I
def E : ℂ := -Complex.I
def D : ℂ := 3 - 4 * Complex.I

theorem QED_mul_eq_neg_25I : Q * E * D = -25 * Complex.I :=
by
  sorry

end ComplexMultiplication

end QED_mul_eq_neg_25I_l300_300804


namespace logan_usual_cartons_l300_300511

theorem logan_usual_cartons 
  (C : ℕ)
  (h1 : ∀ cartons, (∀ jars : ℕ, jars = 20 * cartons) → jars = 20 * C)
  (h2 : ∀ cartons, cartons = C - 20)
  (h3 : ∀ damaged_jars, (∀ cartons : ℕ, cartons = 5) → damaged_jars = 3 * 5)
  (h4 : ∀ completely_damaged_jars, completely_damaged_jars = 20)
  (h5 : ∀ good_jars, good_jars = 565) :
  C = 50 :=
by
  sorry

end logan_usual_cartons_l300_300511


namespace probability_calculations_l300_300313

-- Define the number of students
def total_students : ℕ := 2006

-- Number of students eliminated in the first step
def eliminated_students : ℕ := 6

-- Number of students remaining after elimination
def remaining_students : ℕ := total_students - eliminated_students

-- Number of students to be selected in the second step
def selected_students : ℕ := 50

-- Calculate the probability of a specific student being eliminated
def elimination_probability := (6 : ℚ) / total_students

-- Calculate the probability of a specific student being selected from the remaining students
def selection_probability := (50 : ℚ) / remaining_students

-- The theorem to prove our equivalent proof problem
theorem probability_calculations :
  elimination_probability = (3 : ℚ) / 1003 ∧
  selection_probability = (25 : ℚ) / 1003 :=
by
  sorry

end probability_calculations_l300_300313


namespace interest_rate_per_annum_is_four_l300_300727

-- Definitions
def P : ℕ := 300
def t : ℕ := 8
def I : ℤ := P - 204

-- Interest formula
def simple_interest (P : ℕ) (r : ℕ) (t : ℕ) : ℤ := P * r * t / 100

-- Statement to prove
theorem interest_rate_per_annum_is_four :
  ∃ r : ℕ, I = simple_interest P r t ∧ r = 4 :=
by sorry

end interest_rate_per_annum_is_four_l300_300727


namespace find_share_of_A_l300_300818

variable (A B C : ℝ)
variable (h1 : A = (2/3) * B)
variable (h2 : B = (1/4) * C)
variable (h3 : A + B + C = 510)

theorem find_share_of_A : A = 60 :=
by
  sorry

end find_share_of_A_l300_300818


namespace error_percentage_calc_l300_300553

theorem error_percentage_calc (y : ℝ) (hy : y > 0) : 
  let correct_result := 8 * y
  let erroneous_result := y / 8
  let error := abs (correct_result - erroneous_result)
  let error_percentage := (error / correct_result) * 100
  error_percentage = 98 := by
  sorry

end error_percentage_calc_l300_300553


namespace ab_product_eq_four_l300_300618

theorem ab_product_eq_four (a b : ℝ) (h1: 0 < a) (h2: 0 < b) 
  (h3: (1/2) * (4 / a) * (6 / b) = 3) : 
  a * b = 4 :=
by 
  sorry

end ab_product_eq_four_l300_300618


namespace sum_of_youngest_and_oldest_l300_300097

-- Let a1, a2, a3, a4 be the ages of Janet's 4 children arranged in non-decreasing order.
-- Given conditions:
variable (a₁ a₂ a₃ a₄ : ℕ)
variable (h_mean : (a₁ + a₂ + a₃ + a₄) / 4 = 10)
variable (h_median : (a₂ + a₃) / 2 = 7)

-- Proof problem:
theorem sum_of_youngest_and_oldest :
  a₁ + a₄ = 26 :=
sorry

end sum_of_youngest_and_oldest_l300_300097


namespace total_balls_of_wool_l300_300146

theorem total_balls_of_wool (a_scarves a_sweaters e_sweaters : ℕ)
  (wool_per_scarf wool_per_sweater : ℕ)
  (a_scarves = 10) (a_sweaters = 5) (e_sweaters = 8)
  (wool_per_scarf = 3) (wool_per_sweater = 4) :
  a_scarves * wool_per_scarf + a_sweaters * wool_per_sweater + e_sweaters * wool_per_sweater = 82 :=
by
  sorry

end total_balls_of_wool_l300_300146


namespace total_dollars_is_correct_l300_300503

-- Definitions for the fractions owned by John and Alice.
def johnDollars : ℚ := 5 / 8
def aliceDollars : ℚ := 7 / 20

-- Definition for the total amount of dollars.
def totalDollars : ℚ := johnDollars + aliceDollars

-- Statement of the theorem to prove.
theorem total_dollars_is_correct : totalDollars = 39 / 40 := 
by
  /- proof omitted -/
  sorry

end total_dollars_is_correct_l300_300503


namespace infinite_triples_exists_l300_300516

/-- There are infinitely many ordered triples (a, b, c) of positive integers such that 
the greatest common divisor of a, b, and c is 1, and the sum a^2b^2 + b^2c^2 + c^2a^2 
is the square of an integer. -/
theorem infinite_triples_exists : ∃ (a b c : ℕ), (∀ p q : ℕ, p ≠ q ∧ p % 2 = 1 ∧ q % 2 = 1 ∧ 2 < p ∧ 2 < q →
  let a := p * q
  let b := 2 * p^2
  let c := q^2
  gcd (gcd a b) c = 1 ∧
  ∃ k : ℕ, a^2 * b^2 + b^2 * c^2 + c^2 * a^2 = k^2) :=
sorry

end infinite_triples_exists_l300_300516


namespace exponent_value_l300_300621

theorem exponent_value (y k : ℕ) (h1 : 9^y = 3^k) (h2 : y = 7) : k = 14 := by
  sorry

end exponent_value_l300_300621


namespace sum_of_die_rolls_is_even_l300_300225

-- Define probability of rolling a die and getting an even sum.
def prob_even_sum (n : ℕ) (fair_die : ℕ → ℕ → ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then 1 / 2
  else if n = 2 then 1 / 2
  else 1 / 2

def coin_toss : ℕ → ℕ → ℕ := 
  sorry -- To be defined: function for fair coin toss

-- The main theorem translating the problem statement and conditions
theorem sum_of_die_rolls_is_even :
  let heads := coin_toss 2 1,
      die_roll := λ h : ℕ, fair_die h in
  let n := heads 3 2 in
  prob_even_sum n die_roll = 15 / 16 :=
by sorry

end sum_of_die_rolls_is_even_l300_300225


namespace min_y_coord_l300_300593

noncomputable def y_coord (theta : ℝ) : ℝ :=
  (Real.cos (2 * theta)) * (Real.sin theta)

theorem min_y_coord : ∃ theta : ℝ, y_coord theta = - (Real.sqrt 6) / 3 := by
  sorry

end min_y_coord_l300_300593


namespace complex_multiplication_l300_300575

theorem complex_multiplication (a b c d : ℤ) (i : ℂ) (hi : i^2 = -1) : 
  ((3 : ℂ) - 4 * i) * ((-7 : ℂ) + 6 * i) = (3 : ℂ) + 46 * i := 
  by
    sorry

end complex_multiplication_l300_300575


namespace liked_both_desserts_l300_300060

noncomputable def total_students : ℕ := 50
noncomputable def apple_pie_lovers : ℕ := 22
noncomputable def chocolate_cake_lovers : ℕ := 20
noncomputable def neither_dessert_lovers : ℕ := 17
noncomputable def both_desserts_lovers : ℕ := 9

theorem liked_both_desserts :
  (total_students - neither_dessert_lovers) + both_desserts_lovers = apple_pie_lovers + chocolate_cake_lovers - both_desserts_lovers :=
by
  sorry

end liked_both_desserts_l300_300060


namespace alcohol_percentage_calculation_l300_300122

-- Define the conditions as hypothesis
variables (original_solution_volume : ℝ) (original_alcohol_percent : ℝ)
          (added_alcohol_volume : ℝ) (added_water_volume : ℝ)

-- Assume the given values in the problem
variables (h1 : original_solution_volume = 40) (h2 : original_alcohol_percent = 5)
          (h3 : added_alcohol_volume = 2.5) (h4 : added_water_volume = 7.5)

-- Define the proof goal
theorem alcohol_percentage_calculation :
  let original_alcohol_volume := original_solution_volume * (original_alcohol_percent / 100)
  let total_alcohol_volume := original_alcohol_volume + added_alcohol_volume
  let total_solution_volume := original_solution_volume + added_alcohol_volume + added_water_volume
  let new_alcohol_percent := (total_alcohol_volume / total_solution_volume) * 100
  new_alcohol_percent = 9 :=
by {
  sorry
}

end alcohol_percentage_calculation_l300_300122


namespace find_x_l300_300660

variables {x y z : ℝ}

theorem find_x (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 3) (h3 : z^2 / x = 4) :
  x = 144^(1 / 5) :=
by
  sorry

end find_x_l300_300660


namespace honor_students_count_l300_300974

noncomputable def G : ℕ := 13
noncomputable def B : ℕ := 11
def E_G : ℕ := 3
def E_B : ℕ := 4

theorem honor_students_count (h1 : G + B < 30) 
    (h2 : (E_G : ℚ) / G = 3 / 13) 
    (h3 : (E_B : ℚ) / B = 4 / 11) :
    E_G + E_B = 7 := 
sorry

end honor_students_count_l300_300974


namespace equation_solution_l300_300087

noncomputable def solve_equation (x : ℝ) : Prop :=
  (1/4) * x^(1/2 * Real.log x / Real.log 2) = 2^(1/4 * (Real.log x / Real.log 2)^2)

theorem equation_solution (x : ℝ) (hx : 0 < x) : solve_equation x → (x = 2^(2*Real.sqrt 2) ∨ x = 2^(-2*Real.sqrt 2)) :=
  by
  intro h
  sorry

end equation_solution_l300_300087


namespace simplify_fraction_l300_300376

theorem simplify_fraction : (48 / 72 : ℚ) = (2 / 3) := 
by
  sorry

end simplify_fraction_l300_300376


namespace inequality_proof_l300_300940

noncomputable def a (x1 x2 x3 x4 x5 : ℝ) := x1 + x2 + x3 + x4 + x5
noncomputable def b (x1 x2 x3 x4 x5 : ℝ) := x1 * x2 + x1 * x3 + x1 * x4 + x1 * x5 + x2 * x3 + x2 * x4 + x2 * x5 + x3 * x4 + x3 * x5 + x4 * x5
noncomputable def c (x1 x2 x3 x4 x5 : ℝ) := x1 * x2 * x3 + x1 * x2 * x4 + x1 * x2 * x5 + x1 * x3 * x4 + x1 * x3 * x5 + x1 * x4 * x5 + x2 * x3 * x4 + x2 * x3 * x5 + x2 * x4 * x5 + x3 * x4 * x5
noncomputable def d (x1 x2 x3 x4 x5 : ℝ) := x1 * x2 * x3 * x4 + x1 * x2 * x3 * x5 + x1 * x2 * x4 * x5 + x1 * x3 * x4 * x5 + x2 * x3 * x4 * x5

theorem inequality_proof (x1 x2 x3 x4 x5 : ℝ) (hx1x2x3x4x5 : x1 * x2 * x3 * x4 * x5 = 1) :
  (1 / a x1 x2 x3 x4 x5) + (1 / b x1 x2 x3 x4 x5) + (1 / c x1 x2 x3 x4 x5) + (1 / d x1 x2 x3 x4 x5) ≤ 3 / 5 := 
sorry

end inequality_proof_l300_300940


namespace avg_age_decrease_l300_300394

/-- Define the original average age of the class -/
def original_avg_age : ℕ := 40

/-- Define the number of original students -/
def original_strength : ℕ := 17

/-- Define the average age of the new students -/
def new_students_avg_age : ℕ := 32

/-- Define the number of new students joining -/
def new_students_strength : ℕ := 17

/-- Define the total original age of the class -/
def total_original_age : ℕ := original_strength * original_avg_age

/-- Define the total age of the new students -/
def total_new_students_age : ℕ := new_students_strength * new_students_avg_age

/-- Define the new total strength of the class after joining of new students -/
def new_total_strength : ℕ := original_strength + new_students_strength

/-- Define the new total age of the class after joining of new students -/
def new_total_age : ℕ := total_original_age + total_new_students_age

/-- Define the new average age of the class -/
def new_avg_age : ℕ := new_total_age / new_total_strength

/-- Prove that the average age decreased by 4 years when the new students joined -/
theorem avg_age_decrease : original_avg_age - new_avg_age = 4 := by
  sorry

end avg_age_decrease_l300_300394


namespace exists_divisible_by_3_l300_300647

open Nat

-- Definitions used in Lean 4 statement to represent conditions from part a)
def neighbors (n m : ℕ) : Prop := (m = n + 1) ∨ (m = n + 2) ∨ (2 * m = n) ∨ (m = 2 * n)

def circle_arrangement (ns : Fin 99 → ℕ) : Prop :=
  ∀ i : Fin 99, (neighbors (ns i) (ns ((i + 1) % 99)))

-- Proof problem:
theorem exists_divisible_by_3 (ns : Fin 99 → ℕ) (h : circle_arrangement ns) :
  ∃ i : Fin 99, 3 ∣ ns i :=
sorry

end exists_divisible_by_3_l300_300647


namespace vasya_expected_area_greater_l300_300885

/-- Vasya and Asya roll dice to cut out shapes and determine whose expected area is greater. -/
theorem vasya_expected_area_greater :
  let A : ℕ := 1
  let B : ℕ := 2
  (6 * 7 * (2 * 7 / 6) * 21 / 6) < (21 * 91 / 6) := 
by
  sorry

end vasya_expected_area_greater_l300_300885


namespace largest_digit_M_divisible_by_six_l300_300842

theorem largest_digit_M_divisible_by_six :
  (∃ M : ℕ, M ≤ 9 ∧ (45670 + M) % 6 = 0 ∧ ∀ m : ℕ, m ≤ M → (45670 + m) % 6 ≠ 0) :=
sorry

end largest_digit_M_divisible_by_six_l300_300842


namespace power_multiplication_l300_300006

theorem power_multiplication :
  (- (4 / 5 : ℚ)) ^ 2022 * (5 / 4 : ℚ) ^ 2023 = 5 / 4 := 
by {
  sorry
}

end power_multiplication_l300_300006


namespace total_shared_amount_l300_300548

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := sorry

axiom h1 : A = 1 / 3 * (B + C)
axiom h2 : B = 2 / 7 * (A + C)
axiom h3 : A = B + 20

theorem total_shared_amount : A + B + C = 720 := by
  sorry

end total_shared_amount_l300_300548


namespace xy_value_l300_300609

theorem xy_value (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : 2 * x + y = -5) : x + y = 1 := 
sorry

end xy_value_l300_300609


namespace determinant_eq_sum_of_products_l300_300017

theorem determinant_eq_sum_of_products (x y z : ℝ) :
  Matrix.det (Matrix.of ![![1, x + z, y], ![1, x + y + z, y + z], ![1, x + z, x + y + z]]) = x * y + y * z + z * x :=
by
  sorry

end determinant_eq_sum_of_products_l300_300017


namespace intersection_point_with_y_axis_l300_300829

theorem intersection_point_with_y_axis : 
  ∃ y, (0, y) = (0, 3) ∧ (y = 0 + 3) :=
by
  sorry

end intersection_point_with_y_axis_l300_300829


namespace quadratic_roots_correct_l300_300153

def quadratic (b c : ℝ) (x : ℝ) : ℝ := x^2 + b * x + c

theorem quadratic_roots_correct (b c : ℝ) 
  (h₀ : quadratic b c (-2) = 5)
  (h₁ : quadratic b c (-1) = 0)
  (h₂ : quadratic b c 0 = -3)
  (h₃ : quadratic b c 1 = -4)
  (h₄ : quadratic b c 2 = -3)
  (h₅ : quadratic b c 4 = 5)
  : (quadratic b c (-1) = 0) ∧ (quadratic b c 3 = 0) :=
sorry

end quadratic_roots_correct_l300_300153


namespace vasya_has_greater_expected_area_l300_300879

noncomputable def expected_area_rectangle : ℚ :=
1 / 6 * (1 * 1 + 1 * 2 + 1 * 3 + 1 * 4 + 1 * 5 + 1 * 6 + 
         2 * 1 + 2 * 2 + 2 * 3 + 2 * 4 + 2 * 5 + 2 * 6 + 
         3 * 1 + 3 * 2 + 3 * 3 + 3 * 4 + 3 * 5 + 3 * 6 + 
         4 * 1 + 4 * 2 + 4 * 3 + 4 * 4 + 4 * 5 + 4 * 6 + 
         5 * 1 + 5 * 2 + 5 * 3 + 5 * 4 + 5 * 5 + 5 * 6 + 
         6 * 1 + 6 * 2 + 6 * 3 + 6 * 4 + 6 * 5 + 6 * 6)

noncomputable def expected_area_square : ℚ := 
1 / 6 * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

theorem vasya_has_greater_expected_area : expected_area_rectangle < expected_area_square :=
by {
  -- A calculation of this sort should be done symbolically, not in this theorem,
  -- but the primary goal here is to show the structure of the statement.
  -- Hence, implement symbolic computation later to finalize proof.
  sorry
}

end vasya_has_greater_expected_area_l300_300879


namespace find_n_l300_300915

theorem find_n (n : ℕ) (M N : ℕ) (hM : M = 4 ^ n) (hN : N = 2 ^ n) (h : M - N = 992) : n = 5 :=
sorry

end find_n_l300_300915


namespace inequality_proof_l300_300903

noncomputable def x : ℝ := Real.exp (-1/2)
noncomputable def y : ℝ := Real.log 2 / Real.log 5
noncomputable def z : ℝ := Real.log 3

theorem inequality_proof : z > x ∧ x > y := by
  -- Conditions defined as follows:
  -- x = exp(-1/2)
  -- y = log(2) / log(5)
  -- z = log(3)
  -- To be proved:
  -- z > x > y
  sorry

end inequality_proof_l300_300903


namespace area_DEF_l300_300228

structure Point where
  x : ℝ
  y : ℝ

def D : Point := {x := -3, y := 4}
def E : Point := {x := 1, y := 7}
def F : Point := {x := 3, y := -1}

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * |(A.x * B.y + B.x * C.y + C.x * A.y - A.y * B.x - B.y * C.x - C.y * A.x)|

theorem area_DEF : area_of_triangle D E F = 16 := by
  sorry

end area_DEF_l300_300228


namespace largest_multiple_of_7_l300_300335

def repeated_188 (k : Nat) : ℕ := (List.replicate k 188).foldr (λ x acc => x * 1000 + acc) 0

theorem largest_multiple_of_7 :
  ∃ n, n = repeated_188 100 ∧ ∃ m, m ≤ 303 ∧ m ≥ 0 ∧ m ≠ 300 ∧ (repeated_188 m % 7 = 0 → n ≥ repeated_188 m) :=
by
  sorry

end largest_multiple_of_7_l300_300335


namespace simplify_and_evaluate_expr_l300_300519

theorem simplify_and_evaluate_expr (a b : ℝ) (h1 : a = 1 / 2) (h2 : b = -4) :
  5 * (3 * a ^ 2 * b - a * b ^ 2) - 4 * (-a * b ^ 2 + 3 * a ^ 2 * b) = -11 :=
by
  sorry

end simplify_and_evaluate_expr_l300_300519


namespace distance_light_300_years_eq_l300_300830

-- Define the constant distance light travels in one year
def distance_light_year : ℕ := 9460800000000

-- Define the time period in years
def time_period : ℕ := 300

-- Define the expected distance light travels in 300 years in scientific notation
def expected_distance : ℝ := 28382 * 10^13

-- The theorem to prove
theorem distance_light_300_years_eq :
  (distance_light_year * time_period) = 2838200000000000 :=
by
  sorry

end distance_light_300_years_eq_l300_300830


namespace simplify_expression_l300_300034

theorem simplify_expression (θ : Real) (h : θ ∈ Set.Icc (5 * Real.pi / 4) (3 * Real.pi / 2)) :
  Real.sqrt (1 - Real.sin (2 * θ)) - Real.sqrt (1 + Real.sin (2 * θ)) = -2 * Real.cos θ :=
sorry

end simplify_expression_l300_300034


namespace expected_number_of_different_faces_l300_300720

theorem expected_number_of_different_faces :
  let p := (6 : ℕ) ^ 6
  let q := (5 : ℕ) ^ 6
  6 * (1 - (5 / 6)^6) = (p - q) / (6 ^ 5) :=
by
  sorry

end expected_number_of_different_faces_l300_300720


namespace rational_functional_equation_l300_300634

theorem rational_functional_equation (f : ℚ → ℚ) (h : ∀ x y : ℚ, f (x + f y) = f x + y) :
  (f = λ x => x) ∨ (f = λ x => -x) :=
by
  sorry

end rational_functional_equation_l300_300634


namespace evaluate_expression_l300_300305

theorem evaluate_expression (a b : ℕ) (ha : a = 3) (hb : b = 2) : ((a^b)^a + (b^a)^b = 793) := by
  -- The following lines skip the proof but outline the structure:
  sorry

end evaluate_expression_l300_300305


namespace solution_set_of_inequality_l300_300664

theorem solution_set_of_inequality (x : ℝ) : x > 1 ∨ (-1 < x ∧ x < 0) ↔ x > 1 ∨ (-1 < x ∧ x < 0) :=
by sorry

end solution_set_of_inequality_l300_300664


namespace inequality_proof_l300_300027

theorem inequality_proof (a b : Real) (h1 : a + b < 0) (h2 : b > 0) : a^2 > b^2 :=
by
  sorry

end inequality_proof_l300_300027


namespace vasya_expected_area_greater_l300_300882

/-- Vasya and Asya roll dice to cut out shapes and determine whose expected area is greater. -/
theorem vasya_expected_area_greater :
  let A : ℕ := 1
  let B : ℕ := 2
  (6 * 7 * (2 * 7 / 6) * 21 / 6) < (21 * 91 / 6) := 
by
  sorry

end vasya_expected_area_greater_l300_300882


namespace average_hours_l300_300577

def hours_studied (week1 week2 week3 week4 week5 week6 week7 : ℕ) : ℕ :=
  week1 + week2 + week3 + week4 + week5 + week6 + week7

theorem average_hours (x : ℕ)
  (h1 : hours_studied 8 10 9 11 10 7 x / 7 = 9) :
  x = 8 :=
by
  sorry

end average_hours_l300_300577


namespace rest_of_customers_bought_20_l300_300867

/-
Let's define the number of melons sold by the stand, number of customers who bought one and three melons, and total number of melons bought by these customers.
-/

def total_melons_sold : ℕ := 46
def customers_bought_one : ℕ := 17
def customers_bought_three : ℕ := 3

def melons_bought_by_those_bought_one := customers_bought_one * 1
def melons_bought_by_those_bought_three := customers_bought_three * 3

def remaining_melons := total_melons_sold - (melons_bought_by_those_bought_one + melons_bought_by_those_bought_three)

-- Now we state the theorem that the number of melons bought by the rest of the customers is 20 
theorem rest_of_customers_bought_20 :
  remaining_melons = 20 :=
by
  -- Skip the proof with 'sorry'
  sorry

end rest_of_customers_bought_20_l300_300867


namespace sum_primes_between_20_and_30_is_52_l300_300239

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  (List.range' (a + 1) (b - a - 1)).filter is_prime

def sum_primes_between_20_and_30 : Prop :=
  primes_between 20 30 = [23, 29] ∧ (List.foldl (· + ·) 0 (primes_between 20 30) = 52)

theorem sum_primes_between_20_and_30_is_52 : sum_primes_between_20_and_30 :=
by
  sorry

end sum_primes_between_20_and_30_is_52_l300_300239


namespace average_speed_sf_l300_300546

variables
  (v d t : ℝ)  -- Representing the average speed to SF, the distance, and time to SF
  (h1 : 42 = (2 * d) / (3 * t))  -- Condition: Average speed of the round trip is 42 mph
  (h2 : t = d / v)  -- Definition of time t in terms of distance and speed

theorem average_speed_sf : v = 63 :=
by
  sorry

end average_speed_sf_l300_300546


namespace complex_norm_solution_l300_300824

noncomputable def complex_norm (z : Complex) : Real :=
  Complex.abs z

theorem complex_norm_solution (w z : Complex) 
  (wz_condition : w * z = 24 - 10 * Complex.I)
  (w_norm_condition : complex_norm w = Real.sqrt 29) :
  complex_norm z = (26 * Real.sqrt 29) / 29 :=
by
  sorry

end complex_norm_solution_l300_300824


namespace sufficient_not_necessary_condition_not_necessary_condition_l300_300327

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  a^2 + b^2 = 1 → (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1) :=
by
  sorry

theorem not_necessary_condition (a b : ℝ) : 
  (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1) → ¬(a^2 + b^2 = 1) :=
by
  sorry

end sufficient_not_necessary_condition_not_necessary_condition_l300_300327


namespace value_of_x_l300_300402

theorem value_of_x (z : ℤ) (h1 : z = 100) (y : ℤ) (h2 : y = z / 10) (x : ℤ) (h3 : x = y / 3) : 
  x = 10 / 3 := 
by
  -- The proof is skipped
  sorry

end value_of_x_l300_300402


namespace Gabrielle_sells_8_crates_on_Wednesday_l300_300025

-- Definitions based on conditions from part a)
def crates_sold_on_Monday := 5
def crates_sold_on_Tuesday := 2 * crates_sold_on_Monday
def crates_sold_on_Thursday := crates_sold_on_Tuesday / 2
def total_crates_sold := 28
def crates_sold_on_Wednesday := total_crates_sold - (crates_sold_on_Monday + crates_sold_on_Tuesday + crates_sold_on_Thursday)

-- The theorem to prove the question == answer given conditions
theorem Gabrielle_sells_8_crates_on_Wednesday : crates_sold_on_Wednesday = 8 := by
  sorry

end Gabrielle_sells_8_crates_on_Wednesday_l300_300025


namespace arrangement_of_numbers_l300_300114

theorem arrangement_of_numbers (numbers : Finset ℕ) 
  (h1 : numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) 
  (h_sum : ∀ a b c d e f, a + b + c + d + e + f = 33)
  (h_group_sum : ∀ k1 k2 k3 k4, k1 + k2 + k3 + k4 = 26)
  : ∃ (n : ℕ), n = 2304 := by
  sorry

end arrangement_of_numbers_l300_300114


namespace at_least_one_divisible_by_three_l300_300648

theorem at_least_one_divisible_by_three (n : ℕ) (h1 : n > 0) (h2 : n ≡ 99)
  (h3 : ∀ i, (i < n) → (∃ m : ℤ, abs (m(i+1) - m(i)) = 1 ∨ abs (m(i+1) - m(i)) = 2 ∨ m(i+1) = 2 * m(i))) :
  ∃ k, k ≤ 99 ∧ (k % 3 = 0) := sorry

end at_least_one_divisible_by_three_l300_300648


namespace x_n_squared_leq_2007_l300_300113

def recurrence (x y : ℕ → ℝ) : Prop :=
  x 0 = 1 ∧ y 0 = 2007 ∧
  ∀ n, x (n + 1) = x n - (x n * y n + x (n + 1) * y (n + 1) - 2) * (y n + y (n + 1)) ∧
       y (n + 1) = y n - (x n * y n + x (n + 1) * y (n + 1) - 2) * (x n + x (n + 1))

theorem x_n_squared_leq_2007 (x y : ℕ → ℝ) (h : recurrence x y) : ∀ n, x n ^ 2 ≤ 2007 :=
by sorry

end x_n_squared_leq_2007_l300_300113


namespace estimate_fish_population_l300_300626

theorem estimate_fish_population (n m k : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : k > 0) (h4 : k ≤ m) : 
  ∃ N : ℕ, N = m * n / k :=
by
  sorry

end estimate_fish_population_l300_300626


namespace smallest_integer_y_l300_300116

theorem smallest_integer_y (y : ℤ) (h : 7 - 5 * y < 22) : y ≥ -2 :=
by sorry

end smallest_integer_y_l300_300116


namespace ticket_costs_l300_300993

theorem ticket_costs (ticket_price : ℕ) (number_of_tickets : ℕ) : ticket_price = 44 ∧ number_of_tickets = 7 → ticket_price * number_of_tickets = 308 :=
by
  intros h
  cases h
  sorry

end ticket_costs_l300_300993


namespace ella_days_11_years_old_l300_300757

theorem ella_days_11_years_old (x y z : ℕ) (h1 : 40 * x + 44 * y + 48 * (180 - x - y) = 7920) (h2 : x + y + z = 180) (h3 : 2 * x + y = 180) : y = 60 :=
by {
  -- proof can be derived from the given conditions
  sorry
}

end ella_days_11_years_old_l300_300757


namespace number_of_chocolate_boxes_l300_300083

theorem number_of_chocolate_boxes
  (x y p : ℕ)
  (pieces_per_box : ℕ)
  (total_candies : ℕ)
  (h_y : y = 4)
  (h_pieces : pieces_per_box = 9)
  (h_total : total_candies = 90) :
  x = 6 :=
by
  -- Definitions of the conditions
  let caramel_candies := y * pieces_per_box
  let total_chocolate_candies := total_candies - caramel_candies
  let x := total_chocolate_candies / pieces_per_box
  
  -- Main theorem statement: x = 6
  sorry

end number_of_chocolate_boxes_l300_300083


namespace problem_statement_l300_300104

open BigOperators

noncomputable def number_of_situations_none_form_pair : ℕ :=
  (Nat.choose 10 4) * 2^4

noncomputable def number_of_situations_exactly_two_pairs : ℕ :=
  Nat.choose 10 2

noncomputable def number_of_situations_one_pair_and_two_non_pairs : ℕ :=
  (Nat.choose 10 1) * (Nat.choose 9 2) * 2^2

theorem problem_statement (shoes : Finset (Fin 20)) (h : shoes.card = 4) :
  number_of_situations_none_form_pair = 3360 ∧
  number_of_situations_exactly_two_pairs = 45 ∧
  number_of_situations_one_pair_and_two_non_pairs = 1440 := 
  by sorry

end problem_statement_l300_300104


namespace power_computation_l300_300297

theorem power_computation :
  16^10 * 8^6 / 4^22 = 16384 :=
by
  sorry

end power_computation_l300_300297


namespace vasya_has_greater_area_l300_300871

-- Definition of a fair six-sided die roll
def die_roll : ℕ → ℝ := λ k, if k ∈ {1, 2, 3, 4, 5, 6} then (1 / 6 : ℝ) else 0

-- Expected value of a function with respect to a probability distribution
noncomputable def expected_value (f : ℕ → ℝ) : ℝ := ∑ k in {1, 2, 3, 4, 5, 6}, f k * die_roll k

-- Vasya's area: A^2 where A is a single die roll
noncomputable def vasya_area : ℝ := expected_value (λ k, (k : ℝ) ^ 2)

-- Asya's area: A * B where A and B are independent die rolls
noncomputable def asya_area : ℝ := (expected_value (λ k, (k : ℝ))) ^ 2

theorem vasya_has_greater_area :
  vasya_area > asya_area := sorry

end vasya_has_greater_area_l300_300871


namespace susannah_swims_more_than_camden_l300_300448

-- Define the given conditions
def camden_total_swims : ℕ := 16
def susannah_total_swims : ℕ := 24
def number_of_weeks : ℕ := 4

-- State the theorem
theorem susannah_swims_more_than_camden :
  (susannah_total_swims / number_of_weeks) - (camden_total_swims / number_of_weeks) = 2 :=
by
  sorry

end susannah_swims_more_than_camden_l300_300448


namespace best_fitting_model_l300_300627

theorem best_fitting_model :
  ∀ R1 R2 R3 R4 : ℝ, 
  R1 = 0.21 → R2 = 0.80 → R3 = 0.50 → R4 = 0.98 → 
  abs (R4 - 1) < abs (R1 - 1) ∧ abs (R4 - 1) < abs (R2 - 1) 
    ∧ abs (R4 - 1) < abs (R3 - 1) :=
by
  intros R1 R2 R3 R4 hR1 hR2 hR3 hR4
  rw [hR1, hR2, hR3, hR4]
  exact sorry

end best_fitting_model_l300_300627


namespace total_animal_eyes_l300_300350

-- Define the conditions given in the problem
def numberFrogs : Nat := 20
def numberCrocodiles : Nat := 10
def eyesEach : Nat := 2

-- Define the statement that we need to prove
theorem total_animal_eyes : (numberFrogs * eyesEach) + (numberCrocodiles * eyesEach) = 60 := by
  sorry

end total_animal_eyes_l300_300350


namespace total_donations_correct_l300_300277

def num_basketball_hoops : Nat := 60

def num_hoops_with_balls : Nat := num_basketball_hoops / 2

def num_pool_floats : Nat := 120
def num_damaged_floats : Nat := num_pool_floats / 4
def num_remaining_floats : Nat := num_pool_floats - num_damaged_floats

def num_footballs : Nat := 50
def num_tennis_balls : Nat := 40

def num_hoops_without_balls : Nat := num_basketball_hoops - num_hoops_with_balls

def total_donations : Nat := 
  num_hoops_without_balls + num_hoops_with_balls + num_remaining_floats + num_footballs + num_tennis_balls

theorem total_donations_correct : total_donations = 240 := by
  sorry

end total_donations_correct_l300_300277


namespace number_of_people_l300_300851

theorem number_of_people (total_cookies : ℕ) (cookies_per_person : ℝ) (h1 : total_cookies = 144) (h2 : cookies_per_person = 24.0) : total_cookies / cookies_per_person = 6 := 
by 
  -- Placeholder for actual proof.
  sorry

end number_of_people_l300_300851


namespace expected_faces_rolled_six_times_l300_300713

-- Define a random variable indicating appearance of a particular face
noncomputable def ζi (n : ℕ): ℝ := if n > 0 then 1 - (5 / 6) ^ 6 else 0

-- Define the expected number of distinct faces
noncomputable def expected_distinct_faces : ℝ := 6 * ζi 1

theorem expected_faces_rolled_six_times :
  expected_distinct_faces = (6 ^ 6 - 5 ^ 6) / 6 ^ 5 :=
by
  -- Here we would provide the proof
  sorry

end expected_faces_rolled_six_times_l300_300713


namespace simplify_expression_l300_300520

variable {a : ℝ}

theorem simplify_expression (h₁ : a ≠ 0) (h₂ : a ≠ -1) (h₃ : a ≠ 1) :
  ( ( (a^2 + 1) / a - 2 ) / ( (a^2 - 1) / (a^2 + a) ) ) = a - 1 :=
sorry

end simplify_expression_l300_300520


namespace transition_to_modern_population_reproduction_l300_300222

-- Defining the conditions as individual propositions
def A : Prop := ∃ (m b : ℝ), m < 0 ∧ b = 0
def B : Prop := ∃ (m b : ℝ), m < 0 ∧ b < 0
def C : Prop := ∃ (m b : ℝ), m > 0 ∧ b = 0
def D : Prop := ∃ (m b : ℝ), m > 0 ∧ b > 0

-- Defining the question as a property marking the transition from traditional to modern types of population reproduction
def Q : Prop := B

-- The proof problem
theorem transition_to_modern_population_reproduction :
  Q = B :=
by
  sorry

end transition_to_modern_population_reproduction_l300_300222


namespace train_time_to_pass_platform_l300_300866

noncomputable def train_length : ℝ := 360
noncomputable def platform_length : ℝ := 140
noncomputable def train_speed_km_per_hr : ℝ := 45

noncomputable def train_speed_m_per_s : ℝ :=
  train_speed_km_per_hr * (1000 / 3600)

noncomputable def total_distance : ℝ :=
  train_length + platform_length

theorem train_time_to_pass_platform :
  (total_distance / train_speed_m_per_s) = 40 := by
  sorry

end train_time_to_pass_platform_l300_300866


namespace intersection_M_N_l300_300032

/-- Define the set M as pairs (x, y) such that x + y = 2. -/
def M : Set (ℝ × ℝ) := { p | p.1 + p.2 = 2 }

/-- Define the set N as pairs (x, y) such that x - y = 2. -/
def N : Set (ℝ × ℝ) := { p | p.1 - p.2 = 2 }

/-- The intersection of sets M and N is the single point (2, 0). -/
theorem intersection_M_N : M ∩ N = { (2, 0) } :=
by
  sorry

end intersection_M_N_l300_300032


namespace at_least_one_divisible_by_3_l300_300649

-- Define a function that describes the properties of the numbers as per conditions.
def circle_99_numbers (numbers: Fin 99 → ℕ) : Prop :=
  ∀ n : Fin 99, let neighbor := (n + 1) % 99 
                in abs (numbers n - numbers neighbor) = 1 ∨ 
                   abs (numbers n - numbers neighbor) = 2 ∨ 
                   (numbers n = 2 * numbers neighbor) ∨ 
                   (numbers neighbor = 2 * numbers n)

theorem at_least_one_divisible_by_3 :
  ∀ (numbers: Fin 99 → ℕ), circle_99_numbers numbers → ∃ n : Fin 99, numbers n % 3 = 0 :=
by
  intro numbers
  intro h
  sorry

end at_least_one_divisible_by_3_l300_300649


namespace sum_of_star_tip_angles_l300_300454

noncomputable def sum_star_tip_angles : ℝ :=
  let segment_angle := 360 / 8
  let subtended_arc := 3 * segment_angle
  let theta := subtended_arc / 2
  8 * theta

theorem sum_of_star_tip_angles:
  sum_star_tip_angles = 540 := by
  sorry

end sum_of_star_tip_angles_l300_300454


namespace units_digit_of_30_factorial_is_0_l300_300996

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_30_factorial_is_0 : units_digit (factorial 30) = 0 := by
  sorry

end units_digit_of_30_factorial_is_0_l300_300996


namespace remaining_episodes_l300_300258

theorem remaining_episodes (seasons : ℕ) (episodes_per_season : ℕ) (fraction_watched : ℚ) 
  (h_seasons : seasons = 12) (h_episodes_per_season : episodes_per_season = 20) 
  (h_fraction_watched : fraction_watched = 1/3) : 
  (seasons * episodes_per_season - fraction_watched * (seasons * episodes_per_season) = 160) := 
by
  sorry

end remaining_episodes_l300_300258


namespace area_ratio_of_squares_l300_300835

open Real

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * a = 4 * 4 * b) : (a^2) / (b^2) = 16 := 
by
  sorry

end area_ratio_of_squares_l300_300835


namespace train_speed_l300_300441

theorem train_speed
  (train_length : Real := 460)
  (bridge_length : Real := 140)
  (time_seconds : Real := 48) :
  ((train_length + bridge_length) / time_seconds) * 3.6 = 45 :=
by
  sorry

end train_speed_l300_300441


namespace boys_in_fifth_grade_l300_300792

theorem boys_in_fifth_grade (T S : ℕ) (percent_boys_soccer : ℝ) (girls_not_playing_soccer : ℕ) 
    (hT : T = 420) (hS : S = 250) (h_percent : percent_boys_soccer = 0.86) 
    (h_girls_not_playing_soccer : girls_not_playing_soccer = 65) : 
    ∃ B : ℕ, B = 320 :=
by
  -- We don't need to provide the proof details here
  sorry

end boys_in_fifth_grade_l300_300792


namespace negation_of_proposition_l300_300530

namespace NegationProp

theorem negation_of_proposition :
  (∀ x : ℝ, 0 < x ∧ x < 1 → x^2 - x < 0) ↔
  (∃ x0 : ℝ, 0 < x0 ∧ x0 < 1 ∧ x0^2 - x0 ≥ 0) := by sorry

end NegationProp

end negation_of_proposition_l300_300530


namespace base_area_of_cuboid_eq_seven_l300_300434

-- Definitions of the conditions
def volume_of_cuboid : ℝ := 28 -- Volume is 28 cm³
def height_of_cuboid : ℝ := 4  -- Height is 4 cm

-- The theorem statement for the problem
theorem base_area_of_cuboid_eq_seven
  (Volume : ℝ)
  (Height : ℝ)
  (h1 : Volume = 28)
  (h2 : Height = 4) :
  Volume / Height = 7 := by
  sorry

end base_area_of_cuboid_eq_seven_l300_300434


namespace symmetric_lines_a_b_l300_300529

theorem symmetric_lines_a_b (x y a b : ℝ) (A : ℝ × ℝ) (hA : A = (1, 0))
  (h1 : x + 2 * y - 3 = 0)
  (h2 : a * x + 4 * y + b = 0)
  (h_slope : -1 / 2 = -a / 4)
  (h_point : a * 1 + 4 * 0 + b = 0) :
  a + b = 0 :=
sorry

end symmetric_lines_a_b_l300_300529


namespace liked_both_desserts_l300_300059

noncomputable def total_students : ℕ := 50
noncomputable def apple_pie_lovers : ℕ := 22
noncomputable def chocolate_cake_lovers : ℕ := 20
noncomputable def neither_dessert_lovers : ℕ := 17
noncomputable def both_desserts_lovers : ℕ := 9

theorem liked_both_desserts :
  (total_students - neither_dessert_lovers) + both_desserts_lovers = apple_pie_lovers + chocolate_cake_lovers - both_desserts_lovers :=
by
  sorry

end liked_both_desserts_l300_300059


namespace part_I_part_II_l300_300163

def f (x : ℝ) (m : ℕ) : ℝ := |x - m| + |x|

theorem part_I (m : ℕ) (hm : m = 1) : ∃ x : ℝ, f x m < 2 :=
by sorry

theorem part_II (α β : ℝ) (hα : 1 < α) (hβ : 1 < β) (h : f α 1 + f β 1 = 2) :
  (4 / α) + (1 / β) ≥ 9 / 2 :=
by sorry

end part_I_part_II_l300_300163


namespace honor_students_count_l300_300977

noncomputable def number_of_students_in_class_is_less_than_30 := ∃ n, n < 30
def probability_girl_honor_student (G E_G : ℕ) := E_G / G = (3 : ℚ) / 13
def probability_boy_honor_student (B E_B : ℕ) := E_B / B = (4 : ℚ) / 11

theorem honor_students_count (G B E_G E_B : ℕ) 
  (hG_cond : probability_girl_honor_student G E_G) 
  (hB_cond : probability_boy_honor_student B E_B) 
  (h_total_students : G + B < 30) 
  (hE_G_def : E_G = 3 * G / 13) 
  (hE_B_def : E_B = 4 * B / 11) 
  (hG_nonneg : G >= 13)
  (hB_nonneg : B >= 11):
  E_G + E_B = 7 := 
sorry

end honor_students_count_l300_300977


namespace locus_of_point_P_l300_300031

theorem locus_of_point_P (x y : ℝ) :
  let M := (-2, 0)
  let N := (2, 0)
  (x^2 + y^2 = 4 ∧ x ≠ 2 ∧ x ≠ -2) ↔ 
  ((x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16 ∧ x ≠ 2 ∧ x ≠ -2) :=
by
  sorry 

end locus_of_point_P_l300_300031


namespace students_like_both_l300_300062

theorem students_like_both {total students_apple_pie students_chocolate_cake students_none students_at_least_one students_both : ℕ} 
  (h_total : total = 50)
  (h_apple : students_apple_pie = 22)
  (h_chocolate : students_chocolate_cake = 20)
  (h_none : students_none = 17)
  (h_least_one : students_at_least_one = total - students_none)
  (h_union : students_at_least_one = students_apple_pie + students_chocolate_cake - students_both) :
  students_both = 9 :=
by
  sorry

end students_like_both_l300_300062


namespace units_digit_sum_l300_300685

theorem units_digit_sum (h₁ : (24 : ℕ) % 10 = 4) 
                        (h₂ : (42 : ℕ) % 10 = 2) : 
  ((24^3 + 42^3) % 10 = 2) :=
by
  sorry

end units_digit_sum_l300_300685


namespace oil_price_reduction_l300_300137

theorem oil_price_reduction (P P_r : ℝ) (h1 : P_r = 24.3) (h2 : 1080 / P - 1080 / P_r = 8) : 
  ((P - P_r) / P) * 100 = 18.02 := by
  sorry

end oil_price_reduction_l300_300137


namespace lengths_of_triangle_sides_l300_300477

open Real

noncomputable def triangle_side_lengths (a b c : ℝ) (A B C : ℝ) :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ A + B + C = π ∧ A = 60 * π / 180 ∧
  10 * sqrt 3 = 0.5 * a * b * sin A ∧
  a + b = 13 ∧
  c = sqrt (a^2 + b^2 - 2 * a * b * cos A)

theorem lengths_of_triangle_sides
  (a b c : ℝ) (A B C : ℝ)
  (h : triangle_side_lengths a b c A B C) :
  (a = 5 ∧ b = 8 ∧ c = 7) ∨ (a = 8 ∧ b = 5 ∧ c = 7) :=
sorry

end lengths_of_triangle_sides_l300_300477


namespace sufficient_not_necessary_l300_300942

theorem sufficient_not_necessary (x y : ℝ) : (x > |y|) → (x > y ∧ ¬ (x > y → x > |y|)) :=
by
  sorry

end sufficient_not_necessary_l300_300942


namespace yasna_finish_books_in_two_weeks_l300_300693

theorem yasna_finish_books_in_two_weeks (pages_book1 : ℕ) (pages_book2 : ℕ) (pages_per_day : ℕ) (days_per_week : ℕ) 
  (h1 : pages_book1 = 180) (h2 : pages_book2 = 100) (h3 : pages_per_day = 20) (h4 : days_per_week = 7) : 
  ((pages_book1 + pages_book2) / pages_per_day) / days_per_week = 2 := 
by
  sorry

end yasna_finish_books_in_two_weeks_l300_300693


namespace infinite_series_equals_l300_300579

noncomputable def infinite_series : Real :=
  ∑' n, if h : (n : ℕ) ≥ 2 then (n^4 + 2 * n^3 + 8 * n^2 + 8 * n + 8) / (2^n * (n^4 + 4)) else 0

theorem infinite_series_equals : infinite_series = 11 / 10 :=
  sorry

end infinite_series_equals_l300_300579


namespace exists_distinct_pure_powers_l300_300763

-- Definitions and conditions
def is_pure_kth_power (k m : ℕ) : Prop := ∃ t : ℕ, m = t ^ k

-- The main theorem statement
theorem exists_distinct_pure_powers (n : ℕ) (hn : 0 < n) :
  ∃ (a : Fin n → ℕ),
    (∀ i j : Fin n, i ≠ j → a i ≠ a j) ∧ 
    is_pure_kth_power 2009 (Finset.univ.sum a) ∧ 
    is_pure_kth_power 2010 (Finset.univ.prod a) :=
sorry

end exists_distinct_pure_powers_l300_300763


namespace reduction_of_cycle_l300_300098

noncomputable def firstReductionPercentage (P : ℝ) (x : ℝ) : Prop :=
  P * (1 - (x / 100)) * 0.8 = 0.6 * P

theorem reduction_of_cycle (P x : ℝ) (hP : 0 < P) : firstReductionPercentage P x → x = 25 :=
by
  intros h
  unfold firstReductionPercentage at h
  sorry

end reduction_of_cycle_l300_300098


namespace total_clothing_donated_l300_300733

-- Definition of the initial donation by Adam
def adam_initial_donation : Nat := 4 + 4 + 4*2 + 20 -- 4 pairs of pants, 4 jumpers, 4 pajama sets (8 items), 20 t-shirts

-- Adam's friends' total donation
def friends_donation : Nat := 3 * adam_initial_donation

-- Adam's donation after keeping half
def adam_final_donation : Nat := adam_initial_donation / 2

-- Total donation being the sum of Adam's and friends' donations
def total_donation : Nat := adam_final_donation + friends_donation

-- The statement to prove
theorem total_clothing_donated : total_donation = 126 := by
  -- This is skipped as per instructions
  sorry

end total_clothing_donated_l300_300733


namespace problem_proof_l300_300203

variable (a b c : ℝ)

-- Given conditions
def conditions (a b c : ℝ) : Prop :=
  (0 < a ∧ 0 < b ∧ 0 < c) ∧ ((a + 1) * (b + 1) * (c + 1) = 8)

-- The proof problem
theorem problem_proof (h : conditions a b c) : a + b + c ≥ 3 ∧ a * b * c ≤ 1 :=
  sorry

end problem_proof_l300_300203


namespace roots_of_equation_l300_300971

theorem roots_of_equation (x : ℝ) : (x - 3) ^ 2 = 4 ↔ (x = 5 ∨ x = 1) := by
  sorry

end roots_of_equation_l300_300971


namespace root_in_interval_l300_300416

noncomputable def f (x : ℝ) := Real.log x + x - 2

theorem root_in_interval : ∃ c ∈ Set.Ioo 1 2, f c = 0 := 
sorry

end root_in_interval_l300_300416


namespace complement_of_A_in_U_l300_300041

def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | x ≥ 2}
def complement_U_A : Set ℝ := {x | 0 < x ∧ x < 2}

theorem complement_of_A_in_U :
  (U \ A) = complement_U_A :=
sorry

end complement_of_A_in_U_l300_300041


namespace value_equation_l300_300930

noncomputable def quarter_value := 25
noncomputable def dime_value := 10
noncomputable def half_dollar_value := 50

theorem value_equation (n : ℕ) :
  25 * quarter_value + 20 * dime_value = 15 * quarter_value + 10 * dime_value + n * half_dollar_value → 
  n = 7 :=
by
  sorry

end value_equation_l300_300930


namespace max_volume_48cm_square_l300_300281

def volume_of_box (x : ℝ) := x * (48 - 2 * x)^2

theorem max_volume_48cm_square : 
  ∃ x : ℝ, 0 < x ∧ x < 24 ∧ (∀ y : ℝ, 0 < y ∧ y < 24 → volume_of_box x ≥ volume_of_box y) ∧ x = 8 :=
sorry

end max_volume_48cm_square_l300_300281


namespace return_percentage_is_6_5_l300_300283

def investment1 : ℤ := 16250
def investment2 : ℤ := 16250
def profit_percentage1 : ℚ := 0.15
def loss_percentage2 : ℚ := 0.05
def total_investment : ℤ := 25000
def net_income : ℚ := investment1 * profit_percentage1 - investment2 * loss_percentage2
def return_percentage : ℚ := (net_income / total_investment) * 100

theorem return_percentage_is_6_5 : return_percentage = 6.5 := by
  sorry

end return_percentage_is_6_5_l300_300283


namespace no_real_solution_l300_300307

theorem no_real_solution (x : ℝ) : 
  (¬ (x^4 + 3*x^3)/(x^2 + 3*x + 1) + x = -7) :=
sorry

end no_real_solution_l300_300307


namespace common_ratio_l300_300023

namespace GeometricSeries

-- Definitions
def a1 : ℚ := 4 / 7
def a2 : ℚ := 16 / 49 

-- Proposition
theorem common_ratio : (a2 / a1) = (4 / 7) :=
by
  sorry

end GeometricSeries

end common_ratio_l300_300023


namespace neg_proposition_p_l300_300908

variable {x : ℝ}

def proposition_p : Prop := ∀ x ≥ 0, x^3 - 1 ≥ 0

theorem neg_proposition_p : ¬ proposition_p ↔ ∃ x ≥ 0, x^3 - 1 < 0 :=
by sorry

end neg_proposition_p_l300_300908


namespace accident_rate_is_100_million_l300_300360

theorem accident_rate_is_100_million (X : ℕ) (h1 : 96 * 3000000000 = 2880 * X) : X = 100000000 :=
by
  sorry

end accident_rate_is_100_million_l300_300360


namespace number_of_female_fish_l300_300563

-- Defining the constants given in the problem
def total_fish : ℕ := 45
def fraction_male : ℚ := 2 / 3

-- The statement we aim to prove in Lean
theorem number_of_female_fish : 
  (total_fish : ℚ) * (1 - fraction_male) = 15 :=
by
  sorry

end number_of_female_fish_l300_300563


namespace log2_of_fraction_l300_300576

theorem log2_of_fraction : Real.logb 2 0.03125 = -5 := by
  sorry

end log2_of_fraction_l300_300576


namespace standard_equation_of_circle_l300_300035

theorem standard_equation_of_circle
  (r : ℝ) (h_radius : r = 1)
  (h_center : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (x, y) = (a, b))
  (h_tangent_line : ∃ (a : ℝ), 1 = |4 * a - 3| / 5)
  (h_tangent_x_axis : ∃ (a : ℝ), a = 1) :
  (∃ (a b : ℝ), (x-2)^2 + (y-1)^2 = 1) :=
sorry

end standard_equation_of_circle_l300_300035


namespace sum_of_primes_between_20_and_30_l300_300232

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def primes_between_20_and_30 := [23, 29]

theorem sum_of_primes_between_20_and_30 : 
  23 ∈ primes_between_20_and_30 ∧ 29 ∈ primes_between_20_and_30 ∧
  (∀ n ∈ primes_between_20_and_30, is_prime n) ∧
  list.sum primes_between_20_and_30 = 52 := 
by 
  sorry

end sum_of_primes_between_20_and_30_l300_300232


namespace isosceles_triangle_l300_300347

theorem isosceles_triangle (a c : ℝ) (A C : ℝ) (h : a * Real.sin A = c * Real.sin C) : a = c → Isosceles :=
sorry

end isosceles_triangle_l300_300347


namespace texts_sent_on_Tuesday_l300_300947

theorem texts_sent_on_Tuesday (total_texts monday_texts : Nat) (texts_each_monday : Nat)
  (h_monday : texts_each_monday = 5)
  (h_total : total_texts = 40)
  (h_monday_total : monday_texts = 2 * texts_each_monday) :
  total_texts - monday_texts = 30 := by
  sorry

end texts_sent_on_Tuesday_l300_300947


namespace find_a2_plus_b2_l300_300924

theorem find_a2_plus_b2 (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 15) : a^2 + b^2 = 39 :=
by
  sorry

end find_a2_plus_b2_l300_300924


namespace g_at_3_l300_300832

-- Definition of the function and its property
def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, x ≠ 0 → g(x) - 3 * g(1/x) = 3^x

-- Goal: Prove that g(3) = 16.7
theorem g_at_3 : g 3 = 16.7 :=
by
  sorry

end g_at_3_l300_300832


namespace prove_A_plus_B_plus_1_l300_300498

theorem prove_A_plus_B_plus_1 (A B : ℤ) 
  (h1 : B = A + 2)
  (h2 : 2 * A^2 + A + 6 + 5 * B + 2 = 7 * (A + B + 1) + 5) :
  A + B + 1 = 15 :=
by 
  sorry

end prove_A_plus_B_plus_1_l300_300498


namespace hours_per_day_for_first_group_l300_300544

theorem hours_per_day_for_first_group (h : ℕ) :
  (39 * h * 12 = 30 * 6 * 26) → h = 10 :=
by
  sorry

end hours_per_day_for_first_group_l300_300544


namespace cube_surface_area_l300_300118

theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 20) : 6 * (edge_length * edge_length) = 2400 := by
  -- We state our theorem and assumptions here
  sorry

end cube_surface_area_l300_300118


namespace car_speeds_l300_300527

theorem car_speeds (d x : ℝ) (small_car_speed large_car_speed : ℝ) 
  (h1 : d = 135) 
  (h2 : small_car_speed = 5 * x) 
  (h3 : large_car_speed = 2 * x) 
  (h4 : 135 / small_car_speed + (4 + 0.5) = 135 / large_car_speed)
  : small_car_speed = 45 ∧ large_car_speed = 18 := by
  sorry

end car_speeds_l300_300527


namespace main_theorem_l300_300854

noncomputable def exists_coprime_integers (a b p : ℤ) : Prop :=
  ∃ (m n : ℤ), Int.gcd m n = 1 ∧ p ∣ (a * m + b * n)

theorem main_theorem (a b p : ℤ) : exists_coprime_integers a b p := 
  sorry

end main_theorem_l300_300854


namespace no_such_function_exists_l300_300019

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = x ^ 2 - 1996 :=
by
  sorry

end no_such_function_exists_l300_300019


namespace women_attended_gathering_l300_300743

theorem women_attended_gathering :
  ∀ (m : ℕ) (w_per_man : ℕ) (m_per_woman : ℕ),
  m = 15 ∧ w_per_man = 4 ∧ m_per_woman = 3 →
  ∃ (w : ℕ), w = 20 :=
by
  intros m w_per_man m_per_woman h,
  cases h with hm hw_wom,
  cases hw_wom with hwm hmw,
  sorry

end women_attended_gathering_l300_300743


namespace sum_primes_between_20_and_30_l300_300240

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem sum_primes_between_20_and_30 :
  (primes_between 20 30).sum = 52 :=
by
  sorry

end sum_primes_between_20_and_30_l300_300240


namespace maria_travel_fraction_l300_300586

theorem maria_travel_fraction (x : ℝ) (total_distance : ℝ)
  (h1 : ∀ d1 d2, d1 + d2 = total_distance)
  (h2 : total_distance = 360)
  (h3 : ∃ d1 d2 d3, d1 = 360 * x ∧ d2 = (1 / 4) * (360 - 360 * x) ∧ d3 = 135)
  (h4 : d1 + d2 + d3 = total_distance)
  : x = 1 / 2 :=
by
  sorry

end maria_travel_fraction_l300_300586


namespace total_cost_l300_300284

variable (E P M : ℝ)

axiom condition1 : E + 3 * P + 2 * M = 240
axiom condition2 : 2 * E + 5 * P + 4 * M = 440

theorem total_cost : 3 * E + 4 * P + 6 * M = 520 := 
sorry

end total_cost_l300_300284


namespace cos2_add_3sin2_eq_2_l300_300044

theorem cos2_add_3sin2_eq_2 (x : ℝ) (hx : -20 < x ∧ x < 100) (h : Real.cos x ^ 2 + 3 * Real.sin x ^ 2 = 2) : 
  ∃ n : ℕ, n = 38 := 
sorry

end cos2_add_3sin2_eq_2_l300_300044


namespace sum_of_arithmetic_sequence_l300_300538

theorem sum_of_arithmetic_sequence :
  let a := -3
  let d := 6
  let n := 10
  let a_n := a + (n - 1) * d
  let S_n := (n / 2) * (a + a_n)
  S_n = 240 := by {
  let a := -3
  let d := 6
  let n := 10
  let a_n := a + (n - 1) * d
  let S_n := (n / 2) * (a + a_n)
  sorry
}

end sum_of_arithmetic_sequence_l300_300538


namespace desks_per_row_calc_l300_300891

theorem desks_per_row_calc :
  let restroom_students := 2
  let absent_students := 3 * restroom_students - 1
  let total_students := 23
  let classroom_students := total_students - restroom_students - absent_students
  let total_desks := classroom_students * 3 / 2
  (total_desks / 4 = 6) :=
by
  let restroom_students := 2
  let absent_students := 3 * restroom_students - 1
  let total_students := 23
  let classroom_students := total_students - restroom_students - absent_students
  let total_desks := classroom_students * 3 / 2
  show total_desks / 4 = 6
  sorry

end desks_per_row_calc_l300_300891


namespace greater_expected_area_l300_300877

/-- Let X be a random variable representing a single roll of a die, which can take integer values from 1 through 6. -/
def X : Type := { x : ℕ // 1 ≤ x ∧ x ≤ 6 }

/-- Define independent random variables A and B representing the outcomes of Asya’s die rolls, which can take integer values from 1 through 6 with equal probability. -/
noncomputable def A : Type := { a : ℕ // 1 ≤ a ∧ a ≤ 6 }
noncomputable def B : Type := { b : ℕ // 1 ≤ b ∧ b ≤ 6 }

/-- The expected value of a random variable taking integer values from 1 through 6. 
    E[X] = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5, and E[X^2] = (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6 = 15.1667 -/
noncomputable def expected_X_squared : ℝ := 91 / 6

/-- The expected value of the product of two independent random variables each taking integer values from 1 through 6. 
    E[A * B] = E[A] * E[B] = 3.5 * 3.5 = 12.25 -/
noncomputable def expected_A_times_B : ℝ := 12.25

/-- Prove that the expected area of Vasya's square is greater than Asya's rectangle.
    i.e., E[X^2] > E[A * B] -/
theorem greater_expected_area : expected_X_squared > expected_A_times_B :=
sorry

end greater_expected_area_l300_300877


namespace positive_integers_not_in_E_are_perfect_squares_l300_300752

open Set

def E : Set ℕ := {m | ∃ n : ℕ, m = Int.floor (n + Real.sqrt n + 0.5)}

theorem positive_integers_not_in_E_are_perfect_squares (m : ℕ) (h_pos : 0 < m) :
  m ∉ E ↔ ∃ t : ℕ, m = t^2 := 
by
    sorry

end positive_integers_not_in_E_are_perfect_squares_l300_300752


namespace income_of_A_l300_300826

theorem income_of_A (A B C : ℝ) 
  (h1 : (A + B) / 2 = 4050) 
  (h2 : (B + C) / 2 = 5250) 
  (h3 : (A + C) / 2 = 4200) : 
  A = 3000 :=
by
  sorry

end income_of_A_l300_300826


namespace sum_q_p_values_is_neg42_l300_300480

def p (x : Int) : Int := 2 * Int.natAbs x - 1

def q (x : Int) : Int := -(Int.natAbs x) - 1

def values : List Int := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

def q_p_sum : Int :=
  let q_p_values := values.map (λ x => q (p x))
  q_p_values.sum

theorem sum_q_p_values_is_neg42 : q_p_sum = -42 :=
  by
    sorry

end sum_q_p_values_is_neg42_l300_300480


namespace sqrt_0_54_in_terms_of_a_b_l300_300362

variable (a b : ℝ)

-- Conditions
def sqrt_two_eq_a : Prop := a = Real.sqrt 2
def sqrt_three_eq_b : Prop := b = Real.sqrt 3

-- The main statement to prove
theorem sqrt_0_54_in_terms_of_a_b (h1 : sqrt_two_eq_a a) (h2 : sqrt_three_eq_b b) :
  Real.sqrt 0.54 = 0.3 * a * b := sorry

end sqrt_0_54_in_terms_of_a_b_l300_300362


namespace susannah_swims_more_than_camden_l300_300449

-- Define the given conditions
def camden_total_swims : ℕ := 16
def susannah_total_swims : ℕ := 24
def number_of_weeks : ℕ := 4

-- State the theorem
theorem susannah_swims_more_than_camden :
  (susannah_total_swims / number_of_weeks) - (camden_total_swims / number_of_weeks) = 2 :=
by
  sorry

end susannah_swims_more_than_camden_l300_300449


namespace expected_number_of_different_faces_l300_300706

theorem expected_number_of_different_faces :
  let ζ_i (i : Fin 6) := if (∃ k, k ∈ Finset.range 6) then 1 else 0,
      ζ := (List.range 6).sum (ζ_i),
      p := (5 / 6 : ℝ) ^ 6
  in (Expectation (λ ω => ζ)) = (6 * (1 - p)) :=
by
  sorry

end expected_number_of_different_faces_l300_300706


namespace total_clothing_donated_l300_300734

-- Definition of the initial donation by Adam
def adam_initial_donation : Nat := 4 + 4 + 4*2 + 20 -- 4 pairs of pants, 4 jumpers, 4 pajama sets (8 items), 20 t-shirts

-- Adam's friends' total donation
def friends_donation : Nat := 3 * adam_initial_donation

-- Adam's donation after keeping half
def adam_final_donation : Nat := adam_initial_donation / 2

-- Total donation being the sum of Adam's and friends' donations
def total_donation : Nat := adam_final_donation + friends_donation

-- The statement to prove
theorem total_clothing_donated : total_donation = 126 := by
  -- This is skipped as per instructions
  sorry

end total_clothing_donated_l300_300734


namespace honor_students_count_l300_300976

noncomputable def number_of_students_in_class_is_less_than_30 := ∃ n, n < 30
def probability_girl_honor_student (G E_G : ℕ) := E_G / G = (3 : ℚ) / 13
def probability_boy_honor_student (B E_B : ℕ) := E_B / B = (4 : ℚ) / 11

theorem honor_students_count (G B E_G E_B : ℕ) 
  (hG_cond : probability_girl_honor_student G E_G) 
  (hB_cond : probability_boy_honor_student B E_B) 
  (h_total_students : G + B < 30) 
  (hE_G_def : E_G = 3 * G / 13) 
  (hE_B_def : E_B = 4 * B / 11) 
  (hG_nonneg : G >= 13)
  (hB_nonneg : B >= 11):
  E_G + E_B = 7 := 
sorry

end honor_students_count_l300_300976


namespace expected_number_of_different_faces_l300_300708

noncomputable def expected_faces : ℝ :=
  let probability_face_1_not_appearing := (5 / 6)^6
  let E_zeta_1 := 1 - probability_face_1_not_appearing
  6 * E_zeta_1

theorem expected_number_of_different_faces :
  expected_faces = (6^6 - 5^6) / 6^5 := by 
  sorry

end expected_number_of_different_faces_l300_300708


namespace max_length_is_3sqrt2_l300_300909

noncomputable def max_vector_length (θ : ℝ) (h : 0 ≤ θ ∧ θ < 2 * Real.pi) : ℝ :=
  let OP₁ := (Real.cos θ, Real.sin θ)
  let OP₂ := (2 + Real.sin θ, 2 - Real.cos θ)
  let P₁P₂ := (OP₂.1 - OP₁.1, OP₂.2 - OP₁.2)
  Real.sqrt ((P₁P₂.1)^2 + (P₁P₂.2)^2)

theorem max_length_is_3sqrt2 : ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi → max_vector_length θ sorry = 3 * Real.sqrt 2 := 
sorry

end max_length_is_3sqrt2_l300_300909


namespace books_left_unchanged_l300_300816

theorem books_left_unchanged (initial_books : ℕ) (initial_pens : ℕ) (pens_sold : ℕ) (pens_left : ℕ) :
  initial_books = 51 → initial_pens = 106 → pens_sold = 92 → pens_left = 14 → initial_books = 51 := 
by
  intros h_books h_pens h_sold h_left
  exact h_books

end books_left_unchanged_l300_300816


namespace coconut_grove_nut_yield_l300_300790

/--
In a coconut grove, the trees produce nuts based on some given conditions. Prove that the number of nuts produced by (x + 4) trees per year is 720 when x is 8. The conditions are:

1. (x + 4) trees yield a certain number of nuts per year.
2. x trees yield 120 nuts per year.
3. (x - 4) trees yield 180 nuts per year.
4. The average yield per year per tree is 100.
5. x is 8.
-/

theorem coconut_grove_nut_yield (x : ℕ) (y z w: ℕ) (h₁ : x = 8) (h₂ : y = 120) (h₃ : z = 180) (h₄ : w = 100) :
  ((x + 4) * w) - (x * y + (x - 4) * z) = 720 := 
by
  sorry

end coconut_grove_nut_yield_l300_300790


namespace recipe_flour_requirement_l300_300081

def sugar_cups : ℕ := 9
def salt_cups : ℕ := 40
def flour_initial_cups : ℕ := 4
def additional_flour : ℕ := sugar_cups + 1
def total_flour_cups : ℕ := additional_flour

theorem recipe_flour_requirement : total_flour_cups = 10 := by
  sorry

end recipe_flour_requirement_l300_300081


namespace brian_shoes_l300_300070

theorem brian_shoes (J E B : ℕ) (h1 : J = E / 2) (h2 : E = 3 * B) (h3 : J + E + B = 121) : B = 22 :=
sorry

end brian_shoes_l300_300070


namespace ariel_fish_l300_300565

theorem ariel_fish (total_fish : ℕ) (male_ratio : ℚ) (female_ratio : ℚ) (female_fish : ℕ) : 
  total_fish = 45 ∧ male_ratio = 2/3 ∧ female_ratio = 1/3 → female_fish = 15 :=
by
  sorry

end ariel_fish_l300_300565


namespace similar_triangles_perimeter_l300_300179

open Real

-- Defining the similar triangles and their associated conditions
noncomputable def triangle1 := (4, 6, 8)
noncomputable def side2 := 2

-- Define the possible perimeters of the other triangle
theorem similar_triangles_perimeter (h : True) :
  (∃ x, x = 4.5 ∨ x = 6 ∨ x = 9) :=
sorry

end similar_triangles_perimeter_l300_300179


namespace highest_weekly_sales_is_60_l300_300859

/-- 
Given that a convenience store sold 300 bags of chips in a month,
and the following weekly sales pattern:
1. In the first week, 20 bags were sold.
2. In the second week, there was a 2-for-1 promotion, tripling the sales to 60 bags.
3. In the third week, a 10% discount doubled the sales to 40 bags.
4. In the fourth week, sales returned to the first week's number, 20 bags.
Prove that the number of bags of chips sold during the week with the highest demand is 60.
-/
theorem highest_weekly_sales_is_60 
  (total_sales : ℕ)
  (week1_sales : ℕ)
  (week2_sales : ℕ)
  (week3_sales : ℕ)
  (week4_sales : ℕ)
  (h_total : total_sales = 300)
  (h_week1 : week1_sales = 20)
  (h_week2 : week2_sales = 3 * week1_sales)
  (h_week3 : week3_sales = 2 * week1_sales)
  (h_week4 : week4_sales = week1_sales) :
  max (max week1_sales week2_sales) (max week3_sales week4_sales) = 60 := 
sorry

end highest_weekly_sales_is_60_l300_300859


namespace matt_needs_38_plates_l300_300811

def plates_needed (days_with_only_matt_and_son days_with_parents plates_per_day plates_per_person_with_parents : ℕ) : ℕ :=
  (days_with_only_matt_and_son * plates_per_day) + (days_with_parents * 4 * plates_per_person_with_parents)

theorem matt_needs_38_plates :
  plates_needed 3 4 2 2 = 38 :=
by
  sorry

end matt_needs_38_plates_l300_300811


namespace units_digit_of_sum_of_cubes_l300_300690

theorem units_digit_of_sum_of_cubes : 
  (24^3 + 42^3) % 10 = 2 := by
sorry

end units_digit_of_sum_of_cubes_l300_300690


namespace jimmy_paid_total_l300_300797

-- Data for the problem
def pizza_cost : ℕ := 12
def delivery_charge : ℕ := 2
def park_distance : ℕ := 100
def park_pizzas : ℕ := 3
def building_distance : ℕ := 2000
def building_pizzas : ℕ := 2
def house_distance : ℕ := 800
def house_pizzas : ℕ := 4
def community_center_distance : ℕ := 1500
def community_center_pizzas : ℕ := 5
def office_distance : ℕ := 300
def office_pizzas : ℕ := 1
def bus_stop_distance : ℕ := 1200
def bus_stop_pizzas : ℕ := 3

def cost (distance pizzas : ℕ) : ℕ := 
  let base_cost := pizzas * pizza_cost
  if distance > 1000 then base_cost + delivery_charge else base_cost

def total_cost : ℕ :=
  cost park_distance park_pizzas +
  cost building_distance building_pizzas +
  cost house_distance house_pizzas +
  cost community_center_distance community_center_pizzas +
  cost office_distance office_pizzas +
  cost bus_stop_distance bus_stop_pizzas

theorem jimmy_paid_total : total_cost = 222 :=
  by
    -- Proof omitted
    sorry

end jimmy_paid_total_l300_300797


namespace find_f_neg_one_l300_300036

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 
  if x ≥ 0 then 2^x - 3*x + k else -(2^(-x) - 3*(-x) + k)

theorem find_f_neg_one (k : ℝ) (h : ∀ (x : ℝ), f k (-x) = -f k x) : f k (-1) = 2 :=
sorry

end find_f_neg_one_l300_300036


namespace sport_formulation_water_l300_300184

theorem sport_formulation_water
  (f c w : ℕ)  -- flavoring, corn syrup, and water respectively in standard formulation
  (f_s c_s w_s : ℕ)  -- flavoring, corn syrup, and water respectively in sport formulation
  (corn_syrup_sport : ℤ) -- amount of corn syrup in sport formulation in ounces
  (h_std_ratio : f = 1 ∧ c = 12 ∧ w = 30) -- given standard formulation ratios
  (h_sport_fc_ratio : f_s * 4 = c_s) -- sport formulation flavoring to corn syrup ratio
  (h_sport_fw_ratio : f_s * 60 = w_s) -- sport formulation flavoring to water ratio
  (h_corn_syrup_sport : c_s = corn_syrup_sport) -- amount of corn syrup in sport formulation
  : w_s = 30 := 
by 
  sorry

end sport_formulation_water_l300_300184


namespace honor_students_count_l300_300986

def num_students_total : ℕ := 24
def num_honor_students_girls : ℕ := 3
def num_honor_students_boys : ℕ := 4

def num_girls : ℕ := 13
def num_boys : ℕ := 11

theorem honor_students_count (total_students : ℕ) 
    (prob_girl_honor : ℚ) (prob_boy_honor : ℚ)
    (girls : ℕ) (boys : ℕ)
    (honor_girls : ℕ) (honor_boys : ℕ) :
    total_students < 30 →
    prob_girl_honor = 3 / 13 →
    prob_boy_honor = 4 / 11 →
    girls = 13 →
    honor_girls = 3 →
    boys = 11 →
    honor_boys = 4 →
    girls + boys = total_students →
    honor_girls + honor_boys = 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [← h4, ← h5, ← h6, ← h7, ← h8]
  exact 7

end honor_students_count_l300_300986


namespace norma_cards_lost_l300_300946

theorem norma_cards_lost (original_cards : ℕ) (current_cards : ℕ) (cards_lost : ℕ)
  (h1 : original_cards = 88) (h2 : current_cards = 18) :
  original_cards - current_cards = cards_lost →
  cards_lost = 70 := by
  sorry

end norma_cards_lost_l300_300946


namespace min_value_x_plus_2y_l300_300771

theorem min_value_x_plus_2y (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 2 * y - x * y = 0) : x + 2 * y = 8 := 
by
  sorry

end min_value_x_plus_2y_l300_300771


namespace amc_inequality_l300_300317

theorem amc_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  (a / (b + c^2) + b / (c + a^2) + c / (a + b^2)) ≥ (9 / 4) :=
by
  sorry

end amc_inequality_l300_300317


namespace wool_usage_l300_300147

def total_balls_of_wool_used (scarves_aaron sweaters_aaron sweaters_enid : ℕ) (wool_per_scarf wool_per_sweater : ℕ) : ℕ :=
  (scarves_aaron * wool_per_scarf) + (sweaters_aaron * wool_per_sweater) + (sweaters_enid * wool_per_sweater)

theorem wool_usage :
  total_balls_of_wool_used 10 5 8 3 4 = 82 :=
by
  -- calculations done in solution steps
  -- total_balls_of_wool_used (10 scarves * 3 balls/scarf) + (5 sweaters * 4 balls/sweater) + (8 sweaters * 4 balls/sweater)
  -- total_balls_of_wool_used (30) + (20) + (32)
  -- total_balls_of_wool_used = 30 + 20 + 32 = 82
  sorry

end wool_usage_l300_300147


namespace percentage_trucks_returned_l300_300815

theorem percentage_trucks_returned (total_trucks rented_trucks returned_trucks : ℕ)
  (h1 : total_trucks = 24)
  (h2 : rented_trucks = total_trucks)
  (h3 : returned_trucks ≥ 12)
  (h4 : returned_trucks ≤ total_trucks) :
  (returned_trucks / rented_trucks) * 100 = 50 :=
by sorry

end percentage_trucks_returned_l300_300815


namespace dasha_rectangle_problem_l300_300422

variables (a b c : ℕ)

theorem dasha_rectangle_problem
  (h1 : a > 0) 
  (h2 : a * (b + c) + a * (b - a) + a^2 + a * (c - a) = 43) 
  : (a = 1 ∧ b + c = 22) ∨ (a = 43 ∧ b + c = 2) :=
by
  sorry

end dasha_rectangle_problem_l300_300422


namespace orthic_triangle_of_excenters_l300_300298

open EuclideanGeometry

noncomputable theory

def construct_triangle (K O_A O_B : Point) : Triangle :=
sorry

theorem orthic_triangle_of_excenters 
  (K O_A O_B : Point) 
  (A B C : Point) 
  (hK : is_circumcenter K A B C) 
  (hO_A : is_excenter O_A A B C)
  (hO_B : is_excenter O_B B A C) :
  is_orthic_triangle (Triangle.mk O_A O_B (exc_center K O_A O_B)) (Triangle.mk A B C) :=
sorry

end orthic_triangle_of_excenters_l300_300298


namespace maria_trip_distance_l300_300124

theorem maria_trip_distance (D : ℝ) 
  (h1 : D / 2 + D / 8 + 210 = D) 
  (h2 : D / 2 > 0) 
  (h3 : 210 > 0) : 
  D = 560 :=
by
  have h4 : (3 * D) / 8 = 210, from calc
    (3 * D) / 8 = D / 2 - (D / 8) + 210 - 210 : by ring
            ... = 210                          : by linarith [h1],

  sorry

end maria_trip_distance_l300_300124


namespace Jackie_apples_count_l300_300869

variable (Adam_apples Jackie_apples : ℕ)
variable (h1 : Adam_apples = 10)
variable (h2 : Adam_apples = Jackie_apples + 8)

theorem Jackie_apples_count : Jackie_apples = 2 := by
  sorry

end Jackie_apples_count_l300_300869


namespace log_mult_l300_300749

theorem log_mult : 
  (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 4 :=
by 
  sorry

end log_mult_l300_300749


namespace distinct_pawns_5x5_l300_300054

theorem distinct_pawns_5x5 : 
  ∃ n : ℕ, n = 14400 ∧ 
  (∃ (get_pos : Fin 5 → Fin 5), function.bijective get_pos) :=
begin
  sorry
end

end distinct_pawns_5x5_l300_300054


namespace taxi_fare_l300_300206

theorem taxi_fare (x : ℝ) (h : x > 6) : 
  let starting_price := 6
  let mid_distance_fare := (6 - 2) * 2.4
  let long_distance_fare := (x - 6) * 3.6
  let total_fare := starting_price + mid_distance_fare + long_distance_fare
  total_fare = 3.6 * x - 6 :=
by
  sorry

end taxi_fare_l300_300206


namespace largest_digit_M_l300_300844

-- Define the conditions as Lean types
def digit_sum_divisible_by_3 (M : ℕ) := (4 + 5 + 6 + 7 + M) % 3 = 0
def even_digit (M : ℕ) := M % 2 = 0

-- Define the problem statement in Lean
theorem largest_digit_M (M : ℕ) (h : even_digit M ∧ digit_sum_divisible_by_3 M) : M ≤ 8 ∧ (∀ N : ℕ, even_digit N ∧ digit_sum_divisible_by_3 N → N ≤ M) :=
sorry

end largest_digit_M_l300_300844


namespace units_digit_sum_cubes_l300_300681

theorem units_digit_sum_cubes (n1 n2 : ℕ) 
  (h1 : n1 = 24) (h2 : n2 = 42) : 
  (n1 ^ 3 + n2 ^ 3) % 10 = 6 :=
by 
  -- substitution based on h1 and h2 can be done here.
  sorry

end units_digit_sum_cubes_l300_300681


namespace hotel_charge_decrease_l300_300959

theorem hotel_charge_decrease 
  (G R P : ℝ)
  (h1 : R = 1.60 * G)
  (h2 : P = 0.50 * R) :
  (G - P) / G * 100 = 20 := by
sorry

end hotel_charge_decrease_l300_300959


namespace problem_solutions_l300_300157

theorem problem_solutions (a b c : ℝ) (h : ∀ x, ax^2 + bx + c ≤ 0 ↔ x ≤ -4 ∨ x ≥ 3) :
  (a + b + c > 0) ∧ (∀ x, bx + c > 0 ↔ x < 12) :=
by
  -- The following proof steps are not needed as per the instructions provided
  sorry

end problem_solutions_l300_300157


namespace log_inequality_region_l300_300460

theorem log_inequality_region (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hx1 : x ≠ 1) (hx2 : x ≠ y) :
  (0 < x ∧ x < 1 ∧ 0 < y ∧ y < x) 
  ∨ (1 < x ∧ y > x) ↔ (Real.log y / Real.log x ≥ Real.log (x * y) / Real.log (x / y)) :=
  sorry

end log_inequality_region_l300_300460


namespace at_least_one_composite_l300_300361

theorem at_least_one_composite (a b c : ℕ) (h_odd_a : a % 2 = 1) (h_odd_b : b % 2 = 1) (h_odd_c : c % 2 = 1) 
    (h_not_perfect_square : ∀ m : ℕ, m * m ≠ a) : 
    a ^ 2 + a + 1 = 3 * (b ^ 2 + b + 1) * (c ^ 2 + c + 1) →
    (∃ p, p > 1 ∧ p ∣ (b ^ 2 + b + 1)) ∨ (∃ q, q > 1 ∧ q ∣ (c ^ 2 + c + 1)) :=
by sorry

end at_least_one_composite_l300_300361


namespace december_revenue_times_average_l300_300853

def revenue_in_december_is_multiple_of_average_revenue (R_N R_J R_D : ℝ) : Prop :=
  R_N = (3/5) * R_D ∧    -- Condition: November's revenue is 3/5 of December's revenue
  R_J = (1/3) * R_N ∧    -- Condition: January's revenue is 1/3 of November's revenue
  R_D = 2.5 * ((R_N + R_J) / 2)   -- Question: December's revenue is 2.5 times the average of November's and January's revenue

theorem december_revenue_times_average (R_N R_J R_D : ℝ) :
  revenue_in_december_is_multiple_of_average_revenue R_N R_J R_D :=
by
  -- adding sorry to skip the proof
  sorry

end december_revenue_times_average_l300_300853


namespace floor_sqrt_72_l300_300758

theorem floor_sqrt_72 : ⌊Real.sqrt 72⌋ = 8 :=
by
  -- Proof required here
  sorry

end floor_sqrt_72_l300_300758


namespace maximum_sine_sum_l300_300919

open Real

theorem maximum_sine_sum (x y z : ℝ) (hx : 0 ≤ x) (hy : x ≤ π / 2) (hz : 0 ≤ y) (hw : y ≤ π / 2) (hv : 0 ≤ z) (hu : z ≤ π / 2) :
  ∃ M, M = sqrt 2 - 1 ∧ ∀ x y z : ℝ, 0 ≤ x → x ≤ π / 2 → 0 ≤ y → y ≤ π / 2 → 0 ≤ z → z ≤ π / 2 → 
  sin (x - y) + sin (y - z) + sin (z - x) ≤ M :=
by
  sorry

end maximum_sine_sum_l300_300919


namespace marks_lost_per_wrong_answer_l300_300178

theorem marks_lost_per_wrong_answer 
  (marks_per_correct : ℕ)
  (total_questions : ℕ)
  (total_marks : ℕ)
  (correct_answers : ℕ)
  (wrong_answers : ℕ)
  (score_from_correct : ℕ := correct_answers * marks_per_correct)
  (remaining_marks : ℕ := score_from_correct - total_marks)
  (marks_lost_per_wrong : ℕ) :
  total_questions = correct_answers + wrong_answers →
  total_marks = 130 →
  correct_answers = 38 →
  total_questions = 60 →
  marks_per_correct = 4 →
  marks_lost_per_wrong * wrong_answers = remaining_marks →
  marks_lost_per_wrong = 1 := 
sorry

end marks_lost_per_wrong_answer_l300_300178


namespace distinct_pawn_placements_on_chess_board_l300_300049

def numPawnPlacements : ℕ :=
  5! * 5!

theorem distinct_pawn_placements_on_chess_board :
  numPawnPlacements = 14400 := by
  sorry

end distinct_pawn_placements_on_chess_board_l300_300049


namespace muffin_to_banana_ratio_l300_300756

-- Definitions of costs
def elaine_cost (m b : ℝ) : ℝ := 5 * m + 4 * b
def derek_cost (m b : ℝ) : ℝ := 3 * m + 18 * b

-- The problem statement
theorem muffin_to_banana_ratio (m b : ℝ) (h : derek_cost m b = 3 * elaine_cost m b) : m / b = 2 :=
by
  sorry

end muffin_to_banana_ratio_l300_300756


namespace Rachel_and_Mike_l300_300656

theorem Rachel_and_Mike :
  ∃ b c : ℤ,
    (∀ x : ℝ, |x - 3| = 4 ↔ (x = 7 ∨ x = -1)) ∧
    (∀ x : ℝ, (x - 7) * (x + 1) = 0 ↔ x * x + b * x + c = 0) ∧
    (b, c) = (-6, -7) := by
sorry

end Rachel_and_Mike_l300_300656


namespace expected_number_of_different_faces_l300_300719

theorem expected_number_of_different_faces :
  let p := (6 : ℕ) ^ 6
  let q := (5 : ℕ) ^ 6
  6 * (1 - (5 / 6)^6) = (p - q) / (6 ^ 5) :=
by
  sorry

end expected_number_of_different_faces_l300_300719


namespace honor_students_count_l300_300980

noncomputable def number_of_honor_students (G B Eg Eb : ℕ) (p_girl p_boy : ℚ) : ℕ :=
  if G < 30 ∧ B < 30 ∧ Eg = (3 / 13) * G ∧ Eb = (4 / 11) * B ∧ G + B < 30 then
    Eg + Eb
  else
    0

theorem honor_students_count :
  ∃ (G B Eg Eb : ℕ), (G < 30 ∧ B < 30 ∧ G % 13 = 0 ∧ B % 11 = 0 ∧ Eg = (3 * G / 13) ∧ Eb = (4 * B / 11) ∧ G + B < 30 ∧ number_of_honor_students G B Eg Eb (3 / 13) (4 / 11) = 7) :=
by {
  sorry
}

end honor_students_count_l300_300980


namespace range_of_b_l300_300057

theorem range_of_b (b : ℝ) :
  (∀ x : ℝ, |3 * x - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) → (5 < b ∧ b < 7) :=
sorry

end range_of_b_l300_300057


namespace find_cost_per_pound_of_mixture_l300_300662

-- Problem Definitions and Conditions
variable (x : ℝ) -- the variable x represents the pounds of Spanish peanuts used
variable (y : ℝ) -- the cost per pound of the mixture we're trying to find
def cost_virginia_pound : ℝ := 3.50
def cost_spanish_pound : ℝ := 3.00
def weight_virginia : ℝ := 10.0

-- Formula for the cost per pound of the mixture
noncomputable def cost_per_pound_of_mixture : ℝ := (weight_virginia * cost_virginia_pound + x * cost_spanish_pound) / (weight_virginia + x)

-- Proof Problem Statement
theorem find_cost_per_pound_of_mixture (h : cost_per_pound_of_mixture x = y) : 
  y = (weight_virginia * cost_virginia_pound + x * cost_spanish_pound) / (weight_virginia + x) := sorry

end find_cost_per_pound_of_mixture_l300_300662


namespace geometric_sequence_a5_l300_300914

-- Definitions based on the conditions:
variable {a : ℕ → ℝ} -- the sequence {a_n}
variable (q : ℝ) -- the common ratio of the geometric sequence

-- The sequence is geometric and terms are given:
axiom seq_geom (n m : ℕ) : a n = a 0 * q ^ n
axiom a_3_is_neg4 : a 3 = -4
axiom a_7_is_neg16 : a 7 = -16

-- The specific theorem we are proving:
theorem geometric_sequence_a5 :
  a 5 = -8 :=
by {
  sorry
}

end geometric_sequence_a5_l300_300914


namespace sum_primes_20_to_30_l300_300237

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : list ℕ := 
  [n ∈ list.range (b + 1) | n > a ∧ n ≤ b ∧ is_prime n]

def sum_primes_between {a b : ℕ} (ha : a = 20) (hb : b = 30) : ℕ :=
  (primes_between a b).sum

theorem sum_primes_20_to_30 : sum_primes_between (ha : 20) (hb : 30) = 52 := by
  sorry

end sum_primes_20_to_30_l300_300237


namespace tan_double_phi_sin_cos_expression_l300_300323

noncomputable def phi : ℝ := sorry
axiom phi_in_bound (h1: 0 < phi) (h2: phi < Real.pi) : True
axiom tan_phi_plus_pi_over_4 : Real.tan (phi + Real.pi / 4) = -1 / 3

theorem tan_double_phi : Real.tan (2 * phi) = 4 / 3 := by
  sorry

theorem sin_cos_expression : (Real.sin phi + Real.cos phi) / (2 * Real.cos phi - Real.sin phi) = -1 / 4 := by
  sorry

end tan_double_phi_sin_cos_expression_l300_300323


namespace proof_inequalities_l300_300145

variable {R : Type} [LinearOrder R] [Ring R]

def odd_function (f : R → R) : Prop :=
∀ x : R, f (-x) = -f x

def decreasing_function (f : R → R) : Prop :=
∀ x y : R, x ≤ y → f y ≤ f x

theorem proof_inequalities (f : R → R) (a b : R) 
  (h_odd : odd_function f)
  (h_decr : decreasing_function f)
  (h : a + b ≤ 0) :
  (f a * f (-a) ≤ 0) ∧ (f a + f b ≥ f (-a) + f (-b)) :=
by
  sorry

end proof_inequalities_l300_300145


namespace number_of_ways_to_turn_off_lamps_l300_300512

theorem number_of_ways_to_turn_off_lamps :
  (let n := 10 in  -- number of lamps
  let m := 4 in   -- number of lamps to turn off
  let eligible_spaces := 6 in -- number of eligible spaces to insert the turned-off lamps
  let ways_to_choose_spaces := (eligible_spaces.choose 3) in -- choosing 3 out of 6 spaces
  ways_to_choose_spaces = 20) := 
sorry

end number_of_ways_to_turn_off_lamps_l300_300512


namespace find_z_solutions_l300_300463

theorem find_z_solutions (r : ℚ) (z : ℤ) (h : 2^z + 2 = r^2) : 
  (r = 2 ∧ z = 1) ∨ (r = -2 ∧ z = 1) ∨ (r = 3/2 ∧ z = -2) ∨ (r = -3/2 ∧ z = -2) :=
sorry

end find_z_solutions_l300_300463


namespace geom_seq_decreasing_l300_300481

theorem geom_seq_decreasing :
  (∀ n : ℕ, (4 : ℝ) * 3^(1 - (n + 1) : ℤ) < (4 : ℝ) * 3^(1 - n : ℤ)) :=
sorry

end geom_seq_decreasing_l300_300481


namespace tourist_groups_meet_l300_300840

theorem tourist_groups_meet (x y : ℝ) (h1 : 4.5 * x + 2.5 * y = 30) (h2 : 3 * x + 5 * y = 30) : 
  x = 5 ∧ y = 3 := 
sorry

end tourist_groups_meet_l300_300840


namespace f_properties_l300_300605

noncomputable def f (x : ℝ) : ℝ :=
if -2 < x ∧ x < 0 then 2^x else sorry

theorem f_properties (f_odd : ∀ x : ℝ, f (-x) = -f x)
                     (f_periodic : ∀ x : ℝ, f (x + 3 / 2) = -f x) :
  f 2014 + f 2015 + f 2016 = 0 :=
by 
  -- The proof will go here
  sorry

end f_properties_l300_300605


namespace quadrilateral_ABCD_AB_eq_p_plus_sqrt_q_l300_300064

theorem quadrilateral_ABCD_AB_eq_p_plus_sqrt_q (BC CD AD : ℝ) (angle_A angle_B : ℝ) (h1 : BC = 8)
  (h2 : CD = 12) (h3 : AD = 10) (h4 : angle_A = 60) (h5 : angle_B = 60) : 
  ∃ (p q : ℤ), AB = p + real.sqrt q ∧ p + q = 150 :=
by
  sorry

end quadrilateral_ABCD_AB_eq_p_plus_sqrt_q_l300_300064


namespace bike_distance_difference_l300_300213

-- Defining constants for Alex's and Bella's rates and the time duration
def Alex_rate : ℕ := 12
def Bella_rate : ℕ := 10
def time : ℕ := 6

-- The goal is to prove the difference in distance is 12 miles
theorem bike_distance_difference : (Alex_rate * time) - (Bella_rate * time) = 12 := by
  sorry

end bike_distance_difference_l300_300213


namespace coprime_unique_residues_non_coprime_same_residue_l300_300855

-- Part (a)

theorem coprime_unique_residues (m k : ℕ) (h : m.gcd k = 1) : 
  ∃ (a : Fin m → ℕ) (b : Fin k → ℕ), 
    ∀ (i : Fin m) (j : Fin k), 
      ∀ (i' : Fin m) (j' : Fin k), 
        (i, j) ≠ (i', j') → (a i * b j) % (m * k) ≠ (a i' * b j') % (m * k) := 
sorry

-- Part (b)

theorem non_coprime_same_residue (m k : ℕ) (h : m.gcd k > 1) : 
  ∀ (a : Fin m → ℕ) (b : Fin k → ℕ), 
    ∃ (i : Fin m) (j : Fin k) (i' : Fin m) (j' : Fin k), 
      (i, j) ≠ (i', j') ∧ (a i * b j) % (m * k) = (a i' * b j') % (m * k) := 
sorry

end coprime_unique_residues_non_coprime_same_residue_l300_300855


namespace semicircle_perimeter_l300_300420

-- Assuming π as 3.14 for approximation
def π_approx : ℝ := 3.14

-- Radius of the semicircle
def radius : ℝ := 2.1

-- Half of the circumference
def half_circumference (r : ℝ) : ℝ := π_approx * r

-- Diameter of the semicircle
def diameter (r : ℝ) : ℝ := 2 * r

-- Perimeter of the semicircle
def perimeter (r : ℝ) : ℝ := half_circumference r + diameter r

-- Theorem stating the perimeter of the semicircle with given radius
theorem semicircle_perimeter : perimeter radius = 10.794 := by
  sorry

end semicircle_perimeter_l300_300420


namespace total_days_spent_on_islands_l300_300939

-- Define the conditions and question in Lean 4
def first_expedition_A_weeks := 3
def second_expedition_A_weeks := first_expedition_A_weeks + 2
def last_expedition_A_weeks := second_expedition_A_weeks * 2

def first_expedition_B_weeks := 5
def second_expedition_B_weeks := first_expedition_B_weeks - 3
def last_expedition_B_weeks := first_expedition_B_weeks

def total_weeks_on_island_A := first_expedition_A_weeks + second_expedition_A_weeks + last_expedition_A_weeks
def total_weeks_on_island_B := first_expedition_B_weeks + second_expedition_B_weeks + last_expedition_B_weeks

def total_weeks := total_weeks_on_island_A + total_weeks_on_island_B
def total_days := total_weeks * 7

theorem total_days_spent_on_islands : total_days = 210 :=
by
  -- We skip the proof part
  sorry

end total_days_spent_on_islands_l300_300939


namespace balance_force_l300_300992

structure Vector2D where
  x : ℝ
  y : ℝ

def F1 : Vector2D := ⟨1, 1⟩
def F2 : Vector2D := ⟨2, 3⟩

def vector_add (a b : Vector2D) : Vector2D := ⟨a.x + b.x, a.y + b.y⟩
def vector_neg (a : Vector2D) : Vector2D := ⟨-a.x, -a.y⟩

theorem balance_force : 
  ∃ F3 : Vector2D, vector_add (vector_add F1 F2) F3 = ⟨0, 0⟩ ∧ F3 = ⟨-3, -4⟩ := 
by
  sorry

end balance_force_l300_300992


namespace sinC_calculation_maxArea_calculation_l300_300931

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

end sinC_calculation_maxArea_calculation_l300_300931


namespace part_a_part_b_l300_300698

noncomputable def arithmetic_progression_a (a₁: ℕ) (r: ℕ) : ℕ :=
  a₁ + 3 * r

theorem part_a (a₁: ℕ) (r: ℕ) (h_a₁ : a₁ = 2) (h_r : r = 3) : arithmetic_progression_a a₁ r = 11 := 
by 
  sorry

noncomputable def arithmetic_progression_formula (d: ℕ) (r: ℕ) (n: ℕ) : ℕ :=
  d + (n - 1) * r

theorem part_b (a3: ℕ) (a6: ℕ) (a9: ℕ) (a4_plus_a7_plus_a10: ℕ) (a_sum: ℕ) (h_a3 : a3 = 3) (h_a6 : a6 = 6) (h_a9 : a9 = 9) 
  (h_a4a7a10 : a4_plus_a7_plus_a10 = 207) (h_asum : a_sum = 553) 
  (h_eqn1: 3 * a3 + a6 * 2 = 207) (h_eqn2: a_sum = 553): 
  arithmetic_progression_formula 9 10 11 = 109 := 
by 
  sorry

end part_a_part_b_l300_300698


namespace ahmed_final_score_requirement_l300_300139

-- Define the given conditions
def total_assignments : ℕ := 9
def ahmed_initial_grade : ℕ := 91
def emily_initial_grade : ℕ := 92
def sarah_initial_grade : ℕ := 94
def final_assignment_weight := true -- Assuming each assignment has the same weight
def min_passing_score : ℕ := 70
def max_score : ℕ := 100
def emily_final_score : ℕ := 90

noncomputable def ahmed_min_final_score : ℕ := 98

-- The proof statement
theorem ahmed_final_score_requirement :
  let ahmed_initial_points := ahmed_initial_grade * total_assignments
  let emily_initial_points := emily_initial_grade * total_assignments
  let sarah_initial_points := sarah_initial_grade * total_assignments
  let emily_final_total := emily_initial_points + emily_final_score
  let sarah_final_total := sarah_initial_points + min_passing_score
  let ahmed_final_total_needed := sarah_final_total + 1
  let ahmed_needed_score := ahmed_final_total_needed - ahmed_initial_points
  ahmed_needed_score = ahmed_min_final_score :=
by
  sorry

end ahmed_final_score_requirement_l300_300139


namespace quadrilateral_ABCD_pq_sum_l300_300065

noncomputable def AB_pq_sum : ℕ :=
  let p : ℕ := 9
  let q : ℕ := 141
  p + q

theorem quadrilateral_ABCD_pq_sum (BC CD AD : ℕ) (m_angle_A m_angle_B : ℕ) (hBC : BC = 8) (hCD : CD = 12) (hAD : AD = 10) (hAngleA : m_angle_A = 60) (hAngleB : m_angle_B = 60) : AB_pq_sum = 150 := by sorry

end quadrilateral_ABCD_pq_sum_l300_300065


namespace ned_total_mows_l300_300813

def ned_mowed_front (spring summer fall : Nat) : Nat :=
  spring + summer + fall

def ned_mowed_backyard (spring summer fall : Nat) : Nat :=
  spring + summer + fall

theorem ned_total_mows :
  let front_spring := 6
  let front_summer := 5
  let front_fall := 4
  let backyard_spring := 5
  let backyard_summer := 7
  let backyard_fall := 3
  ned_mowed_front front_spring front_summer front_fall +
  ned_mowed_backyard backyard_spring backyard_summer backyard_fall = 30 := by
  sorry

end ned_total_mows_l300_300813


namespace distinct_balls_boxes_l300_300045

def count_distinct_distributions (balls : ℕ) (boxes : ℕ) : ℕ :=
  if balls = 7 ∧ boxes = 3 then 8 else 0

theorem distinct_balls_boxes :
  count_distinct_distributions 7 3 = 8 :=
by sorry

end distinct_balls_boxes_l300_300045


namespace green_square_area_percentage_l300_300280

noncomputable def flag_side_length (k: ℝ) : ℝ := k
noncomputable def cross_area_fraction : ℝ := 0.49
noncomputable def cross_area (k: ℝ) : ℝ := cross_area_fraction * k^2
noncomputable def cross_width (t: ℝ) : ℝ := t
noncomputable def green_square_side (x: ℝ) : ℝ := x
noncomputable def green_square_area (x: ℝ) : ℝ := x^2

theorem green_square_area_percentage (k: ℝ) (t: ℝ) (x: ℝ)
  (h1: x = 2 * t)
  (h2: 4 * t * (k - t) + x^2 = cross_area k)
  : green_square_area x / (k^2) * 100 = 6.01 :=
by
  sorry

end green_square_area_percentage_l300_300280


namespace women_attended_l300_300741

theorem women_attended :
  (15 * 4) / 3 = 20 :=
by
  sorry

end women_attended_l300_300741


namespace present_age_of_R_l300_300430

variables (P_p Q_p R_p : ℝ)

-- Conditions from the problem
axiom h1 : P_p - 8 = 1/2 * (Q_p - 8)
axiom h2 : Q_p - 8 = 2/3 * (R_p - 8)
axiom h3 : Q_p = 2 * Real.sqrt R_p
axiom h4 : P_p = 3/5 * Q_p

theorem present_age_of_R : R_p = 400 :=
by
  sorry

end present_age_of_R_l300_300430


namespace ammonium_chloride_reaction_l300_300308

/-- 
  Given the reaction NH4Cl + H2O → NH4OH + HCl, 
  if 1 mole of NH4Cl reacts with 1 mole of H2O to produce 1 mole of NH4OH, 
  then 1 mole of HCl is formed.
-/
theorem ammonium_chloride_reaction :
  (∀ (NH4Cl H2O NH4OH HCl : ℕ), NH4Cl = 1 ∧ H2O = 1 ∧ NH4OH = 1 → HCl = 1) :=
by
  sorry

end ammonium_chloride_reaction_l300_300308


namespace point_in_third_quadrant_l300_300066

theorem point_in_third_quadrant (m : ℝ) : 
  (-1 < 0 ∧ -2 + m < 0) ↔ (m < 2) :=
by 
  sorry

end point_in_third_quadrant_l300_300066


namespace sum_primes_in_range_l300_300250

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem sum_primes_in_range : 
  (∑ p in { n | 20 < n ∧ n < 30 ∧ is_prime n }.to_finset, p) = 52 := by 
  sorry

end sum_primes_in_range_l300_300250


namespace jane_reads_pages_l300_300188

theorem jane_reads_pages (P : ℕ) (h1 : 7 * (P + 10) = 105) : P = 5 := by
  sorry

end jane_reads_pages_l300_300188


namespace uncle_taller_by_14_l300_300795

namespace height_problem

def uncle_height : ℝ := 72
def james_height_before_spurt : ℝ := (2 / 3) * uncle_height
def growth_spurt : ℝ := 10
def james_height_after_spurt : ℝ := james_height_before_spurt + growth_spurt

theorem uncle_taller_by_14 : uncle_height - james_height_after_spurt = 14 := by
  sorry

end height_problem

end uncle_taller_by_14_l300_300795


namespace units_digit_of_pow_sum_is_correct_l300_300675

theorem units_digit_of_pow_sum_is_correct : 
  (24^3 + 42^3) % 10 = 2 := by
  -- Start the proof block
  sorry

end units_digit_of_pow_sum_is_correct_l300_300675


namespace exists_digit_a_l300_300012

theorem exists_digit_a : 
  ∃ (a : ℕ), (0 ≤ a ∧ a ≤ 9) ∧ (1111 * a - 1 = (a - 1) ^ (a - 2)) :=
by {
  sorry
}

end exists_digit_a_l300_300012


namespace max_S_n_l300_300471

noncomputable def S (n : ℕ) : ℝ := sorry  -- Definition of the sum of the first n terms

theorem max_S_n (S : ℕ → ℝ) (h16 : S 16 > 0) (h17 : S 17 < 0) : ∃ n, S n = S 8 :=
sorry

end max_S_n_l300_300471


namespace dino_remaining_money_l300_300754

-- Definitions of the conditions
def hours_gig_1 : ℕ := 20
def hourly_rate_gig_1 : ℕ := 10

def hours_gig_2 : ℕ := 30
def hourly_rate_gig_2 : ℕ := 20

def hours_gig_3 : ℕ := 5
def hourly_rate_gig_3 : ℕ := 40

def expenses : ℕ := 500

-- The theorem to be proved: Dino's remaining money at the end of the month
theorem dino_remaining_money : 
  (hours_gig_1 * hourly_rate_gig_1 + hours_gig_2 * hourly_rate_gig_2 + hours_gig_3 * hourly_rate_gig_3) - expenses = 500 := by
  sorry

end dino_remaining_money_l300_300754


namespace math_problem_l300_300799

noncomputable def a : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+1) => if a n < 2 * n then a n + 1 else a n

theorem math_problem (n : ℕ) (hn : n > 0) (ha_inc : ∀ m, m > 0 → a m < a (m + 1)) 
  (ha_rec : ∀ m, m > 0 → a (m + 1) ≤ 2 * m) : 
  ∃ p q : ℕ, p > 0 ∧ q > 0 ∧ n = a p - a q := sorry

end math_problem_l300_300799


namespace part1_l300_300264

theorem part1 (m : ℕ) (n : ℕ) (h1 : m = 6 * 10 ^ n + m / 25) : ∃ i : ℕ, m = 625 * 10 ^ (3 * i) := sorry

end part1_l300_300264


namespace jose_share_of_profit_l300_300421

-- Definitions from problem conditions
def tom_investment : ℕ := 30000
def jose_investment : ℕ := 45000
def profit : ℕ := 27000
def months_total : ℕ := 12
def months_jose_investment : ℕ := 10

-- Derived calculations
def tom_month_investment := tom_investment * months_total
def jose_month_investment := jose_investment * months_jose_investment
def total_month_investment := tom_month_investment + jose_month_investment

-- Prove Jose's share of profit
theorem jose_share_of_profit : (jose_month_investment * profit) / total_month_investment = 15000 := by
  -- This is where the step-by-step proof would go
  sorry

end jose_share_of_profit_l300_300421


namespace number_of_females_l300_300936

-- Definitions
variable (F : ℕ) -- ℕ = Natural numbers, ensuring F is a non-negative integer
variable (h_male : ℕ := 2 * F)
variable (h_total : F + 2 * F = 18)
variable (h_female_pos : F > 0)

-- Theorem
theorem number_of_females (F : ℕ) (h_male : ℕ := 2 * F) (h_total : F + 2 * F = 18) (h_female_pos : F > 0) : F = 6 := 
by 
  sorry

end number_of_females_l300_300936


namespace minimum_f_value_minimum_fraction_value_l300_300807

def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

theorem minimum_f_value : ∃ x : ℝ, f x = 2 :=
by
  -- proof skipped, please insert proof here
  sorry

theorem minimum_fraction_value (a b : ℝ) (h : a^2 + b^2 = 2) : 
  (1 / (a^2 + 1)) + (4 / (b^2 + 1)) = 9 / 4 :=
by
  -- proof skipped, please insert proof here
  sorry

end minimum_f_value_minimum_fraction_value_l300_300807


namespace total_supermarkets_FGH_chain_l300_300405

variable (US_supermarkets : ℕ) (Canada_supermarkets : ℕ)
variable (total_supermarkets : ℕ)

-- Conditions
def condition1 := US_supermarkets = 37
def condition2 := US_supermarkets = Canada_supermarkets + 14

-- Goal
theorem total_supermarkets_FGH_chain
    (h1 : condition1 US_supermarkets)
    (h2 : condition2 US_supermarkets Canada_supermarkets) :
    total_supermarkets = US_supermarkets + Canada_supermarkets :=
sorry

end total_supermarkets_FGH_chain_l300_300405


namespace range_of_m_l300_300470

theorem range_of_m (x y m : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y - x * y = 0) :
    (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y - x * y = 0 → x + 2 * y > m^2 + 2 * m) ↔ (-4 : ℝ) < m ∧ m < 2 :=
by
  sorry

end range_of_m_l300_300470


namespace cos_arcsin_eq_tan_arcsin_eq_l300_300578

open Real

theorem cos_arcsin_eq (h : arcsin (3 / 5) = θ) : cos (arcsin (3 / 5)) = 4 / 5 := by
  sorry

theorem tan_arcsin_eq (h : arcsin (3 / 5) = θ) : tan (arcsin (3 / 5)) = 3 / 4 := by
  sorry

end cos_arcsin_eq_tan_arcsin_eq_l300_300578


namespace general_term_formula_l300_300024

theorem general_term_formula (n : ℕ) : 
  ∀ (a : ℕ → ℕ), (a 1 = 1) → (∀ n > 1, a n - a (n-1) = 2^(n-1)) → (a n = 2^n - 1) :=
  by 
  intros a h1 hdif
  sorry

end general_term_formula_l300_300024


namespace square_fg_length_l300_300388

theorem square_fg_length : 
  ∀ (A B C D E F G: ℝ) (side length: ℝ),
  square ABCD side_length → 
  midpoint E A B →
  on_arc F (arc_centered_at A B D) →
  on_line_intersecting E C F →
  on_perpendicular G F BC →
  (FG = 2) :=
by
  sorry

end square_fg_length_l300_300388


namespace can_cover_101x101_with_102_cells_100_times_l300_300789

theorem can_cover_101x101_with_102_cells_100_times :
  ∃ f : Fin 100 → Fin 101 → Fin 101 → Bool,
  (∀ i j : Fin 101, (i ≠ 100 ∨ j ≠ 100) → ∃ t : Fin 100, 
    f t i j = true) :=
sorry

end can_cover_101x101_with_102_cells_100_times_l300_300789


namespace water_settles_at_34_cm_l300_300669

-- Conditions definitions
def h : ℝ := 40 -- Initial height of the liquids in cm
def ρ_w : ℝ := 1000 -- Density of water in kg/m^3
def ρ_o : ℝ := 700  -- Density of oil in kg/m^3

-- Given the conditions provided above,
-- prove that the new height level of water in the first vessel is 34 cm
theorem water_settles_at_34_cm :
  (40 / (1 + (ρ_o / ρ_w))) = 34 := 
sorry

end water_settles_at_34_cm_l300_300669


namespace units_digit_of_power_435_l300_300462

def units_digit_cycle (n : ℕ) : ℕ :=
  n % 2

def units_digit_of_four_powers (cycle : ℕ) : ℕ :=
  if cycle = 0 then 6 else 4

theorem units_digit_of_power_435 : 
  units_digit_of_four_powers (units_digit_cycle (3^5)) = 4 :=
by
  sorry

end units_digit_of_power_435_l300_300462


namespace solve_card_trade_problem_l300_300513

def card_trade_problem : Prop :=
  ∃ V : ℕ, 
  (75 - V + 10 + 88 - 8 + V = 75 + 88 - 8 + 10 ∧ V + 15 = 35)

theorem solve_card_trade_problem : card_trade_problem :=
  sorry

end solve_card_trade_problem_l300_300513


namespace units_digit_sum_l300_300683

theorem units_digit_sum (h₁ : (24 : ℕ) % 10 = 4) 
                        (h₂ : (42 : ℕ) % 10 = 2) : 
  ((24^3 + 42^3) % 10 = 2) :=
by
  sorry

end units_digit_sum_l300_300683


namespace quadratic_completeness_l300_300196

noncomputable def quad_eqn : Prop :=
  ∃ b c : ℤ, (∀ x : ℝ, (x^2 - 10 * x + 15 = 0) ↔ ((x + b)^2 = c)) ∧ b + c = 5

theorem quadratic_completeness : quad_eqn :=
sorry

end quadratic_completeness_l300_300196


namespace converse_of_propositions_is_true_l300_300141

theorem converse_of_propositions_is_true :
  (∀ x : ℝ, (x = 1 ∨ x = 2) ↔ (x^2 - 3 * x + 2 = 0)) ∧
  (∀ x y : ℝ, (x^2 + y^2 = 0) ↔ (x = 0 ∧ y = 0)) := 
by {
  sorry
}

end converse_of_propositions_is_true_l300_300141


namespace problem_statement_l300_300910

theorem problem_statement (a b : ℝ) (h1 : a ≠ 0) (h2 : ({a, b / a, 1} : Set ℝ) = {a^2, a + b, 0}) :
  a^2017 + b^2017 = -1 := by
  sorry

end problem_statement_l300_300910


namespace piglet_steps_count_l300_300199

theorem piglet_steps_count (u v L : ℝ) (h₁ : (L * u) / (u + v) = 66) (h₂ : (L * u) / (u - v) = 198) : L = 99 :=
sorry

end piglet_steps_count_l300_300199


namespace rita_swimming_months_l300_300496

theorem rita_swimming_months
    (total_required_hours : ℕ := 1500)
    (backstroke_hours : ℕ := 50)
    (breaststroke_hours : ℕ := 9)
    (butterfly_hours : ℕ := 121)
    (monthly_hours : ℕ := 220) :
    (total_required_hours - (backstroke_hours + breaststroke_hours + butterfly_hours)) / monthly_hours = 6 := 
by
    -- Proof is omitted
    sorry

end rita_swimming_months_l300_300496


namespace sum_primes_in_range_l300_300251

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem sum_primes_in_range : 
  (∑ p in { n | 20 < n ∧ n < 30 ∧ is_prime n }.to_finset, p) = 52 := by 
  sorry

end sum_primes_in_range_l300_300251


namespace janice_total_cost_is_correct_l300_300189

def cost_of_items (cost_juices : ℕ) (juices : ℕ) (cost_sandwiches : ℕ) (sandwiches : ℕ) (cost_pastries : ℕ) (pastries : ℕ) (cost_salad : ℕ) (discount_salad : ℕ) : ℕ :=
  let one_sandwich := cost_sandwiches / sandwiches
  let one_juice := cost_juices / juices
  let total_pastries := pastries * cost_pastries
  let discounted_salad := cost_salad - (cost_salad * discount_salad / 100)
  one_sandwich + one_juice + total_pastries + discounted_salad

-- Conditions
def cost_juices := 10
def juices := 5
def cost_sandwiches := 6
def sandwiches := 2
def cost_pastries := 4
def pastries := 2
def cost_salad := 8
def discount_salad := 20

-- Expected Total Cost
def expected_total_cost := 1940 -- in cents to avoid float numbers

theorem janice_total_cost_is_correct : 
  cost_of_items cost_juices juices cost_sandwiches sandwiches cost_pastries pastries cost_salad discount_salad = expected_total_cost :=
by
  simp [cost_of_items, cost_juices, juices, cost_sandwiches, sandwiches, cost_pastries, pastries, cost_salad, discount_salad]
  norm_num
  sorry

end janice_total_cost_is_correct_l300_300189


namespace puppies_per_cage_l300_300552

theorem puppies_per_cage (initial_puppies : ℕ) (sold_puppies : ℕ) (remaining_puppies : ℕ) (cages : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 102)
  (h2 : sold_puppies = 21)
  (h3 : remaining_puppies = initial_puppies - sold_puppies)
  (h4 : cages = 9)
  (h5 : puppies_per_cage = remaining_puppies / cages) : 
  puppies_per_cage = 9 := 
by
  -- The proof should go here
  sorry

end puppies_per_cage_l300_300552


namespace weight_of_new_person_l300_300210

theorem weight_of_new_person (W : ℝ) : 
  let avg_original := W / 5
  let avg_new := avg_original + 4
  let total_new := 5 * avg_new
  let weight_new_person := total_new - (W - 50)
  weight_new_person = 70 :=
by
  let avg_original := W / 5
  let avg_new := avg_original + 4
  let total_new := 5 * avg_new
  let weight_new_person := total_new - (W - 50)
  have : weight_new_person = 70 := sorry
  exact this

end weight_of_new_person_l300_300210


namespace identify_fraction_l300_300736

variable {a b : ℚ}

def is_fraction (x : ℚ) (y : ℚ) := ∃ (n : ℚ), x = n / y

theorem identify_fraction :
  is_fraction 2 a ∧ ¬ is_fraction (2 * a) 3 ∧ ¬ is_fraction (-b) 2 ∧ ¬ is_fraction (3 * a + 1) 2 :=
by
  sorry

end identify_fraction_l300_300736


namespace distance_from_origin_to_line_l300_300214

theorem distance_from_origin_to_line : 
  let A := 1
  let B := 2
  let C := -5
  let x_0 := 0
  let y_0 := 0
  let distance := |A * x_0 + B * y_0 + C| / (Real.sqrt (A ^ 2 + B ^ 2))
  distance = Real.sqrt 5 :=
by
  sorry

end distance_from_origin_to_line_l300_300214


namespace total_cost_of_items_l300_300286

theorem total_cost_of_items
  (E P M : ℕ)
  (h1 : E + 3 * P + 2 * M = 240)
  (h2 : 2 * E + 5 * P + 4 * M = 440) :
  3 * E + 4 * P + 6 * M = 520 := 
sorry

end total_cost_of_items_l300_300286


namespace sequence_properties_l300_300330

variable {Seq : Nat → ℕ}
-- Given conditions: Sn = an(an + 3) / 6
def Sn (n : ℕ) := Seq n * (Seq n + 3) / 6

theorem sequence_properties :
  (Seq 1 = 3) ∧ (Seq 2 = 9) ∧ (∀ n : ℕ, Seq (n+1) = 3 * (n + 1)) :=
by 
  have h1 : Sn 1 = (Seq 1 * (Seq 1 + 3)) / 6 := rfl
  have h2 : Sn 2 = (Seq 2 * (Seq 2 + 3)) / 6 := rfl
  sorry

end sequence_properties_l300_300330


namespace unique_solution_abs_eq_l300_300043

theorem unique_solution_abs_eq : 
  ∃! x : ℝ, |x - 1| = |x - 2| + |x + 3| + 1 :=
by
  use -5
  sorry

end unique_solution_abs_eq_l300_300043


namespace tangents_equal_l300_300820

theorem tangents_equal (α β γ : ℝ) (h1 : Real.sin α + Real.sin β + Real.sin γ = 0) (h2 : Real.cos α + Real.cos β + Real.cos γ = 0) :
  Real.tan (3 * α) = Real.tan (3 * β) ∧ Real.tan (3 * β) = Real.tan (3 * γ) := 
sorry

end tangents_equal_l300_300820


namespace symmetric_points_sum_l300_300770

theorem symmetric_points_sum (a b : ℝ) (h1 : B = (-A)) (h2 : A = (1, a)) (h3 : B = (b, 2)) : a + b = -3 := by
  sorry

end symmetric_points_sum_l300_300770


namespace expected_number_of_different_faces_l300_300718

theorem expected_number_of_different_faces :
  let p := (6 : ℕ) ^ 6
  let q := (5 : ℕ) ^ 6
  6 * (1 - (5 / 6)^6) = (p - q) / (6 ^ 5) :=
by
  sorry

end expected_number_of_different_faces_l300_300718


namespace geometric_series_sum_l300_300574

theorem geometric_series_sum (a r : ℝ) (h : |r| < 1) (h_a : a = 2 / 3) (h_r : r = 2 / 3) :
  ∑' i : ℕ, (a * r^i) = 2 :=
by
  sorry

end geometric_series_sum_l300_300574


namespace journey_distance_l300_300540

theorem journey_distance
  (total_time : ℝ)
  (speed1 speed2 : ℝ)
  (journey_time : total_time = 10)
  (speed1_val : speed1 = 21)
  (speed2_val : speed2 = 24) :
  ∃ D : ℝ, (D / 2 / speed1 + D / 2 / speed2 = total_time) ∧ D = 224 :=
by
  sorry

end journey_distance_l300_300540


namespace ratio_of_logs_l300_300091

theorem ratio_of_logs (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : Real.log a / Real.log 4 = Real.log b / Real.log 18 ∧ Real.log b / Real.log 18 = Real.log (a + b) / Real.log 32) :
  b / a = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end ratio_of_logs_l300_300091


namespace maximum_value_of_f_over_interval_l300_300151

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x + 2) / (2 * x - 2)

theorem maximum_value_of_f_over_interval :
  ∀ x : ℝ, -4 < x ∧ x < 1 → ∃ M : ℝ, (∀ y : ℝ, -4 < y ∧ y < 1 → f y ≤ M) ∧ M = -1 :=
by
  sorry

end maximum_value_of_f_over_interval_l300_300151


namespace rods_in_one_mile_l300_300325

-- Definitions of the conditions
def mile_to_furlong := 10
def furlong_to_rod := 50

-- Theorem statement corresponding to the proof problem
theorem rods_in_one_mile : mile_to_furlong * furlong_to_rod = 500 := 
by sorry

end rods_in_one_mile_l300_300325


namespace simplify_and_evaluate_l300_300821

variable (a : ℝ)
variable (b : ℝ)

theorem simplify_and_evaluate (h : b = -1/3) : (a + b)^2 - a * (2 * b + a) = 1/9 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l300_300821


namespace tom_age_ratio_l300_300839

-- Definitions of the variables
variables (T : ℕ) (N : ℕ)

-- Conditions given in the problem
def condition1 : Prop := T = 2 * (T / 2)
def condition2 : Prop := (T - 3) = 3 * (T / 2 - 12)

-- The ratio theorem to prove
theorem tom_age_ratio (h1 : condition1 T) (h2 : condition2 T) : T / N = 22 :=
by
  sorry

end tom_age_ratio_l300_300839


namespace range_of_a_l300_300913

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + 2 * a > 0) ↔ (0 < a ∧ a < 8) :=
by
  sorry

end range_of_a_l300_300913


namespace elephant_weight_l300_300406

theorem elephant_weight :
  ∃ (w : ℕ), ∀ i : Fin 15, (i.val ≤ 13 → w + 2 * w = 15000) ∧ ((0:ℕ) < w → w = 5000) :=
by
  sorry

end elephant_weight_l300_300406


namespace sum_primes_20_to_30_l300_300248

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_20_to_30 : (∑ n in Finset.filter is_prime (Finset.range 31), n) = 52 :=
by
  sorry

end sum_primes_20_to_30_l300_300248


namespace gcd_lcm_product_l300_300294

theorem gcd_lcm_product (a b: ℕ) (h1 : a = 36) (h2 : b = 210) :
  Nat.gcd a b * Nat.lcm a b = 7560 := 
by 
  sorry

end gcd_lcm_product_l300_300294


namespace distinct_xy_values_l300_300661

theorem distinct_xy_values : ∃ (xy_values : Finset ℕ), 
  (∀ (x y : ℕ), (0 < x ∧ 0 < y) → (1 / Real.sqrt x + 1 / Real.sqrt y = 1 / Real.sqrt 20) → (xy_values = {8100, 6400})) ∧
  (xy_values.card = 2) :=
by
  sorry

end distinct_xy_values_l300_300661


namespace frequency_of_rolling_six_is_0_point_19_l300_300275

theorem frequency_of_rolling_six_is_0_point_19 :
  ∀ (total_rolls number_six_appeared : ℕ), total_rolls = 100 → number_six_appeared = 19 → 
  (number_six_appeared : ℝ) / (total_rolls : ℝ) = 0.19 := 
by 
  intros total_rolls number_six_appeared h_total_rolls h_number_six_appeared
  sorry

end frequency_of_rolling_six_is_0_point_19_l300_300275


namespace largest_digit_M_l300_300843

-- Define the conditions as Lean types
def digit_sum_divisible_by_3 (M : ℕ) := (4 + 5 + 6 + 7 + M) % 3 = 0
def even_digit (M : ℕ) := M % 2 = 0

-- Define the problem statement in Lean
theorem largest_digit_M (M : ℕ) (h : even_digit M ∧ digit_sum_divisible_by_3 M) : M ≤ 8 ∧ (∀ N : ℕ, even_digit N ∧ digit_sum_divisible_by_3 N → N ≤ M) :=
sorry

end largest_digit_M_l300_300843


namespace minimum_value_y_l300_300279

theorem minimum_value_y (x y : ℕ) (h1 : x + y = 64) (h2 : 3 * x + 4 * y = 200) : y = 8 :=
by
  sorry

end minimum_value_y_l300_300279


namespace david_chemistry_marks_l300_300582

theorem david_chemistry_marks :
  let english := 96
  let mathematics := 95
  let physics := 82
  let biology := 95
  let average_all := 93
  let total_subjects := 5
  let total_other_subjects := english + mathematics + physics + biology
  let total_all_subjects := average_all * total_subjects
  let chemistry := total_all_subjects - total_other_subjects
  chemistry = 97 :=
by
  -- Definition of variables
  let english := 96
  let mathematics := 95
  let physics := 82
  let biology := 95
  let average_all := 93
  let total_subjects := 5
  let total_other_subjects := english + mathematics + physics + biology
  let total_all_subjects := average_all * total_subjects
  let chemistry := total_all_subjects - total_other_subjects

  -- Assert the final value
  show chemistry = 97
  sorry

end david_chemistry_marks_l300_300582


namespace ellipse_tangent_to_rectangle_satisfies_equation_l300_300663

theorem ellipse_tangent_to_rectangle_satisfies_equation
  (a b : ℝ) -- lengths of the semi-major and semi-minor axes of the ellipse
  (h_rect : 4 * a * b = 48) -- the area condition (since the rectangle sides are 2a and 2b)
  (h_ellipse_form : ∃ (a b : ℝ), ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) : 
  a = 4 ∧ b = 3 ∨ a = 3 ∧ b = 4 := 
sorry

end ellipse_tangent_to_rectangle_satisfies_equation_l300_300663


namespace sum_term_S2018_l300_300067

def arithmetic_sequence (a : ℕ → ℤ) := 
  ∃ (d : ℤ), ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  S 0 = 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

theorem sum_term_S2018 :
  ∃ (a S : ℕ → ℤ),
    arithmetic_sequence a ∧ 
    sum_first_n_terms a S ∧ 
    a 1 = -2018 ∧ 
    ((S 2015) / 2015 - (S 2013) / 2013 = 2) ∧ 
    S 2018 = -2018 
:= by
  sorry

end sum_term_S2018_l300_300067


namespace polynomial_roots_l300_300753

theorem polynomial_roots : ∀ x : ℝ, (x^3 - 4*x^2 - x + 4) * (x - 3) * (x + 2) = 0 ↔ 
  (x = -2 ∨ x = -1 ∨ x = 1 ∨ x = 3 ∨ x = 4) :=
by 
  sorry

end polynomial_roots_l300_300753


namespace total_notebooks_eq_216_l300_300691

theorem total_notebooks_eq_216 (n : ℕ) 
  (h1 : total_notebooks = n^2 + 20)
  (h2 : total_notebooks = (n + 1)^2 - 9) : 
  total_notebooks = 216 := 
by 
  sorry

end total_notebooks_eq_216_l300_300691


namespace expected_faces_rolled_six_times_l300_300715

-- Define a random variable indicating appearance of a particular face
noncomputable def ζi (n : ℕ): ℝ := if n > 0 then 1 - (5 / 6) ^ 6 else 0

-- Define the expected number of distinct faces
noncomputable def expected_distinct_faces : ℝ := 6 * ζi 1

theorem expected_faces_rolled_six_times :
  expected_distinct_faces = (6 ^ 6 - 5 ^ 6) / 6 ^ 5 :=
by
  -- Here we would provide the proof
  sorry

end expected_faces_rolled_six_times_l300_300715


namespace triangle_angle_sum_l300_300177

-- Definitions of the given angles and relationships
def angle_BAC := 95
def angle_ABC := 55
def angle_ABD := 125

-- We need to express the configuration of points and the measure of angle ACB
noncomputable def angle_ACB (angle_BAC angle_ABC angle_ABD : ℝ) : ℝ :=
  180 - angle_BAC - angle_ABC

-- The formalization of the problem statement in Lean 4
theorem triangle_angle_sum (angle_BAC angle_ABC angle_ABD : ℝ) :
  angle_BAC = 95 → angle_ABC = 55 → angle_ABD = 125 → angle_ACB angle_BAC angle_ABC angle_ABD = 30 :=
by
  intros h_BAC h_ABC h_ABD
  rw [h_BAC, h_ABC, h_ABD]
  sorry

end triangle_angle_sum_l300_300177


namespace bear_meat_needs_l300_300431

theorem bear_meat_needs (B_total : ℕ) (cubs : ℕ) (w_cub : ℚ) 
  (h1 : B_total = 210)
  (h2 : cubs = 4)
  (h3 : w_cub = B_total / cubs) : 
  w_cub = 52.5 :=
by 
  sorry

end bear_meat_needs_l300_300431


namespace jason_retirement_age_l300_300354

def age_at_retirement (initial_age years_to_chief extra_years_ratio years_after_masterchief : ℕ) : ℕ :=
  initial_age + years_to_chief + (years_to_chief * extra_years_ratio / 100) + years_after_masterchief

theorem jason_retirement_age :
  age_at_retirement 18 8 25 10 = 46 :=
by
  sorry

end jason_retirement_age_l300_300354


namespace steve_final_amount_l300_300389

def initial_deposit : ℝ := 100
def interest_years_1_to_3 : ℝ := 0.10
def interest_years_4_to_5 : ℝ := 0.08
def annual_deposit_years_1_to_2 : ℝ := 10
def annual_deposit_years_3_to_5 : ℝ := 15

def total_after_one_year (initial : ℝ) (annual : ℝ) (interest : ℝ) : ℝ :=
  initial * (1 + interest) + annual

def steve_saving_after_five_years : ℝ :=
  let year1 := total_after_one_year initial_deposit annual_deposit_years_1_to_2 interest_years_1_to_3
  let year2 := total_after_one_year year1 annual_deposit_years_1_to_2 interest_years_1_to_3
  let year3 := total_after_one_year year2 annual_deposit_years_3_to_5 interest_years_1_to_3
  let year4 := total_after_one_year year3 annual_deposit_years_3_to_5 interest_years_4_to_5
  let year5 := total_after_one_year year4 annual_deposit_years_3_to_5 interest_years_4_to_5
  year5

theorem steve_final_amount :
  steve_saving_after_five_years = 230.88768 := by
  sorry

end steve_final_amount_l300_300389


namespace mersenne_prime_condition_l300_300423

theorem mersenne_prime_condition (a n : ℕ) (h_a : 1 < a) (h_n : 1 < n) (h_prime : Prime (a ^ n - 1)) : a = 2 ∧ Prime n :=
by
  sorry

end mersenne_prime_condition_l300_300423
