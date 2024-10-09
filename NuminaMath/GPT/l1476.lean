import Mathlib

namespace inequality_holds_for_all_l1476_147666

theorem inequality_holds_for_all (m : ℝ) 
  (h : ∀ x : ℝ, (x^2 - 8 * x + 20) / (m * x^2 - m * x - 1) < 0) : -4 < m ∧ m ≤ 0 := 
sorry

end inequality_holds_for_all_l1476_147666


namespace smallest_sum_of_digits_l1476_147635

noncomputable def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem smallest_sum_of_digits (n : ℕ) (h : sum_of_digits n = 2017) : sum_of_digits (n + 1) = 2 := 
sorry

end smallest_sum_of_digits_l1476_147635


namespace find_a_l1476_147611

theorem find_a (a : ℝ) (h : ∃ x : ℝ, x = 2 ∧ x^2 + a * x - 2 = 0) : a = -1 := 
by 
  sorry

end find_a_l1476_147611


namespace sum_of_n_and_k_l1476_147688

theorem sum_of_n_and_k (n k : ℕ) 
  (h1 : (n.choose k) * 3 = (n.choose (k + 1)))
  (h2 : (n.choose (k + 1)) * 2 = (n.choose (k + 2))) :
  n + k = 13 :=
by
  sorry

end sum_of_n_and_k_l1476_147688


namespace diana_owes_amount_l1476_147665

def principal : ℝ := 60
def rate : ℝ := 0.06
def time : ℝ := 1
def interest := principal * rate * time
def original_amount := principal
def total_amount := original_amount + interest

theorem diana_owes_amount :
  total_amount = 63.60 :=
by
  -- Placeholder for actual proof
  sorry

end diana_owes_amount_l1476_147665


namespace blue_beads_count_l1476_147627

-- Define variables and conditions
variables (r b : ℕ)

-- Define the conditions
def condition1 : Prop := r = 30
def condition2 : Prop := r / 3 = b / 2

-- State the theorem
theorem blue_beads_count (h1 : condition1 r) (h2 : condition2 r b) : b = 20 :=
sorry

end blue_beads_count_l1476_147627


namespace cube_root_of_8_l1476_147679

theorem cube_root_of_8 : (∃ x : ℝ, x * x * x = 8) ∧ (∃ y : ℝ, y * y * y = 8 → y = 2) :=
by
  sorry

end cube_root_of_8_l1476_147679


namespace line_passes_through_parabola_vertex_l1476_147653

theorem line_passes_through_parabola_vertex : 
  ∃ (c : ℝ), (∀ (x : ℝ), y = 2 * x + c → ∃ (x0 : ℝ), (x0 = 0 ∧ y = c^2)) ∧ 
  (∀ (c1 c2 : ℝ), (y = 2 * x + c1 ∧ y = 2 * x + c2 → c1 = c2)) → 
  ∃ c : ℝ, c = 0 ∨ c = 1 :=
by 
  -- Proof should be inserted here
  sorry

end line_passes_through_parabola_vertex_l1476_147653


namespace rectangle_perimeter_l1476_147678

theorem rectangle_perimeter (a b : ℕ) : 
  (2 * a + b = 6 ∨ a + 2 * b = 6 ∨ 2 * a + b = 9 ∨ a + 2 * b = 9) → 
  2 * a + 2 * b = 10 :=
by 
  sorry

end rectangle_perimeter_l1476_147678


namespace astronaut_revolutions_l1476_147695

theorem astronaut_revolutions (n : ℤ) (R : ℝ) (hn : n > 2) :
    ∃ k : ℤ, k = n - 1 := 
sorry

end astronaut_revolutions_l1476_147695


namespace probability_red_or_green_is_two_thirds_l1476_147631

-- Define the conditions
def total_balls := 2 + 3 + 4
def favorable_outcomes := 2 + 4

-- Define the probability calculation
def probability_red_or_green := (favorable_outcomes : ℚ) / total_balls

-- The theorem statement
theorem probability_red_or_green_is_two_thirds : probability_red_or_green = 2 / 3 := by
  -- This part will contain the proof using Lean, but we skip it with "sorry" for now.
  sorry

end probability_red_or_green_is_two_thirds_l1476_147631


namespace problem1_problem2_l1476_147637

-- Problem 1
theorem problem1 : ((2 / 3 - 1 / 12 - 1 / 15) * -60) = -31 := by
  sorry

-- Problem 2
theorem problem2 : ((-7 / 8) / ((7 / 4) - 7 / 8 - 7 / 12)) = -3 := by
  sorry

end problem1_problem2_l1476_147637


namespace completion_time_is_midnight_next_day_l1476_147639

-- Define the initial start time
def start_time : ℕ := 9 -- 9:00 AM in hours

-- Define the completion time for 1/4th of the mosaic
def partial_completion_time : ℕ := 3 * 60 + 45  -- 3 hours and 45 minutes in minutes

-- Calculate total_time needed to complete the whole mosaic
def total_time : ℕ := 4 * partial_completion_time -- total time in minutes

-- Define the time at which the artist should finish the entire mosaic
def end_time : ℕ := start_time * 60 + total_time -- end time in minutes

-- Assuming 24 hours in a day, calculate 12:00 AM next day in minutes from midnight
def midnight_next_day : ℕ := 24 * 60

-- Theorem proving the artist will finish at 12:00 AM next day
theorem completion_time_is_midnight_next_day :
  end_time = midnight_next_day := by
    sorry -- proof not required

end completion_time_is_midnight_next_day_l1476_147639


namespace lucas_initial_pet_beds_l1476_147676

-- Definitions from the problem conditions
def additional_beds := 8
def beds_per_pet := 2
def pets := 10

-- Statement to prove
theorem lucas_initial_pet_beds :
  (pets * beds_per_pet) - additional_beds = 12 := 
by
  sorry

end lucas_initial_pet_beds_l1476_147676


namespace sphere_radius_proportional_l1476_147684

theorem sphere_radius_proportional
  (k : ℝ)
  (r1 r2 : ℝ)
  (W1 W2 : ℝ)
  (h_weight_area : ∀ (r : ℝ), W1 = k * (4 * π * r^2))
  (h_given1: W2 = 32)
  (h_given2: r2 = 0.3)
  (h_given3: W1 = 8):
  r1 = 0.15 := 
by
  sorry

end sphere_radius_proportional_l1476_147684


namespace sqrt_meaningful_condition_l1476_147617

theorem sqrt_meaningful_condition (a : ℝ) : 2 - a ≥ 0 → a ≤ 2 := by
  sorry

end sqrt_meaningful_condition_l1476_147617


namespace goteborg_to_stockholm_distance_l1476_147682

/-- 
Given that the distance from Goteborg to Jonkoping on a map is 100 cm 
and the distance from Jonkoping to Stockholm is 150 cm, with a map scale of 1 cm: 20 km,
prove that the total distance from Goteborg to Stockholm passing through Jonkoping is 5000 km.
-/
theorem goteborg_to_stockholm_distance :
  let distance_G_to_J := 100 -- distance from Goteborg to Jonkoping in cm
  let distance_J_to_S := 150 -- distance from Jonkoping to Stockholm in cm
  let scale := 20 -- scale of the map, 1 cm : 20 km
  distance_G_to_J * scale + distance_J_to_S * scale = 5000 := 
by 
  let distance_G_to_J := 100 -- defining the distance from Goteborg to Jonkoping in cm
  let distance_J_to_S := 150 -- defining the distance from Jonkoping to Stockholm in cm
  let scale := 20 -- defining the scale of the map, 1 cm : 20 km
  sorry

end goteborg_to_stockholm_distance_l1476_147682


namespace arithmetic_mean_solution_l1476_147606

-- Define the Arithmetic Mean statement
theorem arithmetic_mean_solution (x : ℝ) (h : (x + 5 + 17 + 3 * x + 11 + 3 * x + 6) / 5 = 19) : 
  x = 8 :=
by
  sorry -- Proof is not required as per the instructions

end arithmetic_mean_solution_l1476_147606


namespace hyperbola_focal_length_l1476_147643

noncomputable def a : ℝ := Real.sqrt 10
noncomputable def b : ℝ := Real.sqrt 2
noncomputable def c : ℝ := Real.sqrt (a ^ 2 + b ^ 2)
noncomputable def focal_length : ℝ := 2 * c

theorem hyperbola_focal_length :
  focal_length = 4 * Real.sqrt 3 := by
  sorry

end hyperbola_focal_length_l1476_147643


namespace area_of_support_is_15_l1476_147625

-- Define the given conditions
def initial_mass : ℝ := 60
def reduced_mass : ℝ := initial_mass - 10
def area_reduction : ℝ := 5
def mass_per_area_increase : ℝ := 1

-- Define the area of the support and prove that it is 15 dm^2
theorem area_of_support_is_15 (x : ℝ) 
  (initial_mass_eq : initial_mass / x = initial_mass / x) 
  (new_mass_eq : reduced_mass / (x - area_reduction) = initial_mass / x + mass_per_area_increase) : 
  x = 15 :=
  sorry

end area_of_support_is_15_l1476_147625


namespace largest_domain_of_f_l1476_147685

theorem largest_domain_of_f (f : ℝ → ℝ) (dom : ℝ → Prop) :
  (∀ x : ℝ, dom x → dom (1 / x)) →
  (∀ x : ℝ, dom x → (f x + f (1 / x) = x)) →
  (∀ x : ℝ, dom x ↔ x = 1 ∨ x = -1) :=
by
  intro h1 h2
  sorry

end largest_domain_of_f_l1476_147685


namespace number_of_bottle_caps_put_inside_l1476_147624

-- Definitions according to the conditions
def initial_bottle_caps : ℕ := 7
def final_bottle_caps : ℕ := 14
def additional_bottle_caps (initial final : ℕ) := final - initial

-- The main theorem to prove
theorem number_of_bottle_caps_put_inside : additional_bottle_caps initial_bottle_caps final_bottle_caps = 7 :=
by
  sorry

end number_of_bottle_caps_put_inside_l1476_147624


namespace three_collinear_points_l1476_147603

theorem three_collinear_points (f : ℝ → Prop) (h_black_or_white : ∀ (x : ℝ), f x = true ∨ f x = false)
: ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (b = (a + c) / 2) ∧ ((f a = f b) ∧ (f b = f c)) :=
sorry

end three_collinear_points_l1476_147603


namespace locus_of_midpoint_l1476_147642

theorem locus_of_midpoint
  (x y : ℝ)
  (h : ∃ (A : ℝ × ℝ), A = (2*x, 2*y) ∧ (A.1)^2 + (A.2)^2 - 8*A.1 = 0) :
  x^2 + y^2 - 4*x = 0 :=
by
  sorry

end locus_of_midpoint_l1476_147642


namespace linear_equation_in_x_l1476_147634

theorem linear_equation_in_x (m : ℤ) (h : |m| = 1) (h₂ : m - 1 ≠ 0) : m = -1 :=
sorry

end linear_equation_in_x_l1476_147634


namespace binom_20_10_eq_184756_l1476_147630

theorem binom_20_10_eq_184756 (h1 : Nat.choose 18 8 = 43758)
                               (h2 : Nat.choose 18 9 = 48620)
                               (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 :=
by
  sorry

end binom_20_10_eq_184756_l1476_147630


namespace keys_per_lock_l1476_147622

-- Define the given conditions
def num_complexes := 2
def apartments_per_complex := 12
def total_keys := 72

-- Calculate the total number of apartments
def total_apartments := num_complexes * apartments_per_complex

-- The theorem statement to prove
theorem keys_per_lock : total_keys / total_apartments = 3 := 
by
  sorry

end keys_per_lock_l1476_147622


namespace prob_sum_seven_prob_two_fours_l1476_147687

-- Definitions and conditions
def total_outcomes : ℕ := 36
def outcomes_sum_seven : ℕ := 6
def outcomes_two_fours : ℕ := 1

-- Proof problem for question 1
theorem prob_sum_seven : outcomes_sum_seven / total_outcomes = 1 / 6 :=
by
  sorry

-- Proof problem for question 2
theorem prob_two_fours : outcomes_two_fours / total_outcomes = 1 / 36 :=
by
  sorry

end prob_sum_seven_prob_two_fours_l1476_147687


namespace percentage_decrease_of_larger_angle_l1476_147651

noncomputable def complementary_angles_decrease_percentage : Real :=
let total_degrees := 90
let ratio_sum := 3 + 7
let part := total_degrees / ratio_sum
let smaller_angle := 3 * part
let larger_angle := 7 * part
let increased_smaller_angle := smaller_angle * 1.2
let new_larger_angle := total_degrees - increased_smaller_angle
let decrease_amount := larger_angle - new_larger_angle
(decrease_amount / larger_angle) * 100

theorem percentage_decrease_of_larger_angle
  (smaller_increased_percentage : Real := 20)
  (ratio_three : Real := 3)
  (ratio_seven : Real := 7)
  (total_degrees : Real := 90)
  (expected_decrease : Real := 8.57):
  complementary_angles_decrease_percentage = expected_decrease := 
sorry

end percentage_decrease_of_larger_angle_l1476_147651


namespace spring_length_relationship_l1476_147675

def spring_length (x : ℝ) : ℝ := 6 + 0.3 * x

theorem spring_length_relationship (x : ℝ) : spring_length x = 0.3 * x + 6 :=
by sorry

end spring_length_relationship_l1476_147675


namespace min_M_inequality_l1476_147694

noncomputable def M_min : ℝ := 9 * Real.sqrt 2 / 32

theorem min_M_inequality :
  ∀ (a b c : ℝ),
    abs (a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2))
    ≤ M_min * (a^2 + b^2 + c^2)^2 :=
by
  sorry

end min_M_inequality_l1476_147694


namespace ice_cream_sandwiches_each_l1476_147649

theorem ice_cream_sandwiches_each (total_ice_cream_sandwiches : ℕ) (number_of_nieces : ℕ) 
  (h1 : total_ice_cream_sandwiches = 143) (h2 : number_of_nieces = 11) : 
  total_ice_cream_sandwiches / number_of_nieces = 13 :=
by
  sorry

end ice_cream_sandwiches_each_l1476_147649


namespace pizzas_served_during_lunch_l1476_147620

theorem pizzas_served_during_lunch {total_pizzas dinner_pizzas lunch_pizzas: ℕ} 
(h_total: total_pizzas = 15) (h_dinner: dinner_pizzas = 6) (h_eq: total_pizzas = dinner_pizzas + lunch_pizzas) : 
lunch_pizzas = 9 := by
  sorry

end pizzas_served_during_lunch_l1476_147620


namespace parabola_sum_coefficients_l1476_147628

theorem parabola_sum_coefficients :
  ∃ (a b c : ℤ), 
    (∀ x : ℝ, (x = 0 → a * (x^2) + b * x + c = 1)) ∧
    (∀ x : ℝ, (x = 2 → a * (x^2) + b * x + c = 9)) ∧
    (a * (1^2) + b * 1 + c = 4)
  → a + b + c = 4 :=
by sorry

end parabola_sum_coefficients_l1476_147628


namespace chess_match_duration_l1476_147648

def time_per_move_polly := 28
def time_per_move_peter := 40
def total_moves := 30
def moves_per_player := total_moves / 2

def Polly_time := moves_per_player * time_per_move_polly
def Peter_time := moves_per_player * time_per_move_peter
def total_time_seconds := Polly_time + Peter_time
def total_time_minutes := total_time_seconds / 60

theorem chess_match_duration : total_time_minutes = 17 := by
  sorry

end chess_match_duration_l1476_147648


namespace shopping_problem_l1476_147629

theorem shopping_problem
  (D S H N : ℝ)
  (h1 : (D - (D / 2 - 10)) + (S - 0.85 * S) + (H - (H - 30)) + (N - N) = 120)
  (T_sale : ℝ := (D / 2 - 10) + 0.85 * S + (H - 30) + N) :
  (120 + 0.10 * T_sale = 0.10 * 1200) →
  D + S + H + N = 1200 :=
by
  sorry

end shopping_problem_l1476_147629


namespace integer_solutions_of_quadratic_l1476_147600

theorem integer_solutions_of_quadratic (k : ℤ) :
  ∀ x : ℤ, (6 - k) * (9 - k) * x^2 - (117 - 15 * k) * x + 54 = 0 ↔
  k = 3 ∨ k = 7 ∨ k = 15 ∨ k = 6 ∨ k = 9 :=
by
  sorry

end integer_solutions_of_quadratic_l1476_147600


namespace minimum_ellipse_area_l1476_147610

theorem minimum_ellipse_area (a b : ℝ) (h₁ : 4 * (a : ℝ) ^ 2 * b ^ 2 = a ^ 2 + b ^ 4)
  (h₂ : (∀ x y : ℝ, ((x - 2) ^ 2 + y ^ 2 ≤ 4 → x ^ 2 / (4 * a ^ 2) + y ^ 2 / (4 * b ^ 2) ≤ 1)) 
       ∧ (∀ x y : ℝ, ((x + 2) ^ 2 + y ^ 2 ≤ 4 → x ^ 2 / (4 * a ^ 2) + y ^ 2 / (4 * b ^ 2) ≤ 1))) : 
  ∃ k : ℝ, (k = 16) ∧ (π * (4 * a * b) = k * π) :=
by sorry

end minimum_ellipse_area_l1476_147610


namespace initial_puppies_l1476_147668

-- Definitions based on the conditions in the problem
def sold : ℕ := 21
def puppies_per_cage : ℕ := 9
def number_of_cages : ℕ := 9

-- The statement to prove
theorem initial_puppies : sold + (puppies_per_cage * number_of_cages) = 102 := by
  sorry

end initial_puppies_l1476_147668


namespace onions_total_l1476_147698

theorem onions_total (Sara : ℕ) (Sally : ℕ) (Fred : ℕ)
  (hSara : Sara = 4) (hSally : Sally = 5) (hFred : Fred = 9) :
  Sara + Sally + Fred = 18 :=
by
  sorry

end onions_total_l1476_147698


namespace permutations_mississippi_l1476_147663

theorem permutations_mississippi : 
  let total_letters := 11
  let m_count := 1
  let i_count := 4
  let s_count := 4
  let p_count := 2
  (Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count)) = 34650 := 
by
  sorry

end permutations_mississippi_l1476_147663


namespace alfred_gain_percent_l1476_147664

theorem alfred_gain_percent :
  let purchase_price := 4700
  let repair_costs := 800
  let selling_price := 5800
  let total_cost := purchase_price + repair_costs
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost) * 100
  gain_percent = 5.45 := 
by
  sorry

end alfred_gain_percent_l1476_147664


namespace find_pq_l1476_147689

-- Define the constants function for the given equation and form
noncomputable def quadratic_eq (p q r : ℤ) : (ℤ × ℤ × ℤ) :=
(2*p*q, p^2 + 2*p*q + q^2 + r, q*q + r)

-- Define the theorem we want to prove
theorem find_pq (p q r: ℤ) (h : quadratic_eq 2 q r = (8, -24, -56)) : pq = -12 :=
by sorry

end find_pq_l1476_147689


namespace value_of_c_l1476_147626

theorem value_of_c (c : ℝ) : (∀ x : ℝ, x * (4 * x + 1) < c ↔ x > -5 / 2 ∧ x < 3) → c = 27 :=
by
  intros h
  sorry

end value_of_c_l1476_147626


namespace groups_needed_l1476_147696

theorem groups_needed (h_camper_count : 36 > 0) (h_group_limit : 12 > 0) : 
  ∃ x : ℕ, x = 36 / 12 ∧ x = 3 := by
  sorry

end groups_needed_l1476_147696


namespace focal_length_of_lens_l1476_147672

-- Define the conditions
def initial_screen_distance : ℝ := 80
def moved_screen_distance : ℝ := 40
def lens_formula (f v u : ℝ) : Prop := (1 / f) = (1 / v) + (1 / u)

-- Define the proof goal
theorem focal_length_of_lens :
  ∃ f : ℝ, (f = 100 ∨ f = 60) ∧
  lens_formula f f (1 / 0) ∧  -- parallel beam implies object at infinity u = 1/0
  initial_screen_distance = 80 ∧
  moved_screen_distance = 40 :=
sorry

end focal_length_of_lens_l1476_147672


namespace KodyAgeIs32_l1476_147609

-- Definition for Mohamed's current age
def mohamedCurrentAge : ℕ := 2 * 30

-- Definition for Mohamed's age four years ago
def mohamedAgeFourYrsAgo : ℕ := mohamedCurrentAge - 4

-- Definition for Kody's age four years ago
def kodyAgeFourYrsAgo : ℕ := mohamedAgeFourYrsAgo / 2

-- Definition to check Kody's current age
def kodyCurrentAge : ℕ := kodyAgeFourYrsAgo + 4

theorem KodyAgeIs32 : kodyCurrentAge = 32 := by
  sorry

end KodyAgeIs32_l1476_147609


namespace cupcakes_frosted_in_10_minutes_l1476_147618

-- Definitions representing the given conditions
def CagneyRate := 15 -- seconds per cupcake
def LaceyRate := 40 -- seconds per cupcake
def JessieRate := 30 -- seconds per cupcake
def initialDuration := 3 * 60 -- 3 minutes in seconds
def totalDuration := 10 * 60 -- 10 minutes in seconds
def afterJessieDuration := totalDuration - initialDuration -- 7 minutes in seconds

-- Proof statement
theorem cupcakes_frosted_in_10_minutes : 
  let combinedRateBefore := (CagneyRate * LaceyRate) / (CagneyRate + LaceyRate)
  let combinedRateAfter := (CagneyRate * LaceyRate * JessieRate) / (CagneyRate * LaceyRate + LaceyRate * JessieRate + JessieRate * CagneyRate)
  let cupcakesBefore := initialDuration / combinedRateBefore
  let cupcakesAfter := afterJessieDuration / combinedRateAfter
  cupcakesBefore + cupcakesAfter = 68 :=
by
  sorry

end cupcakes_frosted_in_10_minutes_l1476_147618


namespace find_f_7_over_2_l1476_147605

section
variable {f : ℝ → ℝ}

-- Conditions
axiom odd_fn : ∀ x : ℝ, f (-x) = -f (x)
axiom even_shift_fn : ∀ x : ℝ, f (x + 1) = f (1 - x)
axiom range_x : Π x : ℝ, -1 ≤ x ∧ x ≤ 0 → f (x) = 2 * x^2

-- Prove that f(7/2) = 1/2
theorem find_f_7_over_2 : f (7 / 2) = 1 / 2 :=
sorry
end

end find_f_7_over_2_l1476_147605


namespace marcia_oranges_l1476_147691

noncomputable def averageCost
  (appleCost bananaCost orangeCost : ℝ) 
  (numApples numBananas numOranges : ℝ) : ℝ :=
  (numApples * appleCost + numBananas * bananaCost + numOranges * orangeCost) /
  (numApples + numBananas + numOranges)

theorem marcia_oranges : 
  ∀ (appleCost bananaCost orangeCost avgCost : ℝ) 
  (numApples numBananas numOranges : ℝ),
  appleCost = 2 → 
  bananaCost = 1 → 
  orangeCost = 3 → 
  numApples = 12 → 
  numBananas = 4 → 
  avgCost = 2 → 
  averageCost appleCost bananaCost orangeCost numApples numBananas numOranges = avgCost → 
  numOranges = 4 :=
by 
  intros appleCost bananaCost orangeCost avgCost numApples numBananas numOranges
         h1 h2 h3 h4 h5 h6 h7
  sorry

end marcia_oranges_l1476_147691


namespace probability_heads_twice_in_three_flips_l1476_147614

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_heads_twice_in_three_flips :
  let p := 0.5
  let n := 3
  let k := 2
  (binomial_coefficient n k : ℝ) * p^k * (1 - p)^(n - k) = 0.375 :=
by
  sorry

end probability_heads_twice_in_three_flips_l1476_147614


namespace area_S3_l1476_147659

theorem area_S3 {s1 s2 s3 : ℝ} (h1 : s1^2 = 25)
  (h2 : s2 = s1 / Real.sqrt 2)
  (h3 : s3 = s2 / Real.sqrt 2)
  : s3^2 = 6.25 :=
by
  sorry

end area_S3_l1476_147659


namespace two_digit_numbers_div_quotient_remainder_l1476_147640

theorem two_digit_numbers_div_quotient_remainder (x y : ℕ) (N : ℕ) (h1 : N = 10 * x + y) (h2 : N = 7 * (x + y) + 6) (hx_range : 1 ≤ x ∧ x ≤ 9) (hy_range : 0 ≤ y ∧ y ≤ 9) :
  N = 62 ∨ N = 83 := sorry

end two_digit_numbers_div_quotient_remainder_l1476_147640


namespace daughters_and_granddaughters_without_daughters_l1476_147619

-- Given conditions
def melissa_daughters : ℕ := 10
def half_daughters_with_children : ℕ := melissa_daughters / 2
def grandchildren_per_daughter : ℕ := 4
def total_descendants : ℕ := 50

-- Calculations based on given conditions
def number_of_granddaughters : ℕ := total_descendants - melissa_daughters
def daughters_with_no_children : ℕ := melissa_daughters - half_daughters_with_children
def granddaughters_with_no_children : ℕ := number_of_granddaughters

-- The final result we need to prove
theorem daughters_and_granddaughters_without_daughters : 
  daughters_with_no_children + granddaughters_with_no_children = 45 := by
  sorry

end daughters_and_granddaughters_without_daughters_l1476_147619


namespace man_and_son_work_together_l1476_147641

-- Define the rates at which the man and his son can complete the work
def man_work_rate := 1 / 5
def son_work_rate := 1 / 20

-- Define the combined work rate when they work together
def combined_work_rate := man_work_rate + son_work_rate

-- Define the total time taken to complete the work together
def days_to_complete_together := 1 / combined_work_rate

-- The theorem stating that they will complete the work in 4 days
theorem man_and_son_work_together : days_to_complete_together = 4 := by
  sorry

end man_and_son_work_together_l1476_147641


namespace max_x_plus_y_l1476_147638

-- Define the conditions as hypotheses in a Lean statement
theorem max_x_plus_y (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^4 = (x - 1) * (y^3 - 23) - 1) :
  x + y ≤ 7 ∧ (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^4 = (x - 1) * (y^3 - 23) - 1 ∧ x + y = 7) :=
by
  sorry

end max_x_plus_y_l1476_147638


namespace sumata_family_miles_driven_per_day_l1476_147686

theorem sumata_family_miles_driven_per_day :
  let total_miles := 1837.5
  let number_of_days := 13.5
  let miles_per_day := total_miles / number_of_days
  (miles_per_day : Real) = 136.1111 :=
by
  sorry

end sumata_family_miles_driven_per_day_l1476_147686


namespace age_multiplier_l1476_147681

theorem age_multiplier (S F M X : ℕ) (h1 : S = 27) (h2 : F = 48) (h3 : S + F = 75)
  (h4 : 27 - X = F - S) (h5 : F = M * X) : M = 8 :=
by
  -- Proof will be filled in here
  sorry

end age_multiplier_l1476_147681


namespace inverse_of_f_l1476_147697

def f (x : ℝ) : ℝ := 7 - 3 * x

noncomputable def f_inv (x : ℝ) : ℝ := (7 - x) / 3

theorem inverse_of_f : ∀ x : ℝ, f (f_inv x) = x ∧ f_inv (f x) = x :=
by
  intros
  sorry

end inverse_of_f_l1476_147697


namespace A_star_B_eq_l1476_147657

def A : Set ℝ := {x | ∃ y, y = 2 * x - x^2}
def B : Set ℝ := {y | ∃ x, y = 2^x ∧ x > 0}
def A_star_B : Set ℝ := {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

theorem A_star_B_eq : A_star_B = {x | x ≤ 1} :=
by {
  sorry
}

end A_star_B_eq_l1476_147657


namespace battery_lasts_12_more_hours_l1476_147644

-- Define initial conditions
def standby_battery_life : ℕ := 36
def active_battery_life : ℕ := 4
def total_time_on : ℕ := 12
def active_usage_time : ℕ := 90  -- in minutes

-- Conversion and calculation functions
def active_usage_hours : ℚ := active_usage_time / 60
def standby_consumption_rate : ℚ := 1 / standby_battery_life
def active_consumption_rate : ℚ := 1 / active_battery_life
def battery_used_standby : ℚ := (total_time_on - active_usage_hours) * standby_consumption_rate
def battery_used_active : ℚ := active_usage_hours * active_consumption_rate
def total_battery_used : ℚ := battery_used_standby + battery_used_active
def remaining_battery : ℚ := 1 - total_battery_used
def additional_hours_standby : ℚ := remaining_battery / standby_consumption_rate

-- Proof statement
theorem battery_lasts_12_more_hours : additional_hours_standby = 12 := by
  sorry

end battery_lasts_12_more_hours_l1476_147644


namespace directrix_of_parabola_l1476_147654

-- Define the given condition
def parabola_eq (x y : ℝ) : Prop := y = -4 * x^2

-- The problem we need to prove
theorem directrix_of_parabola :
  ∃ y : ℝ, (∀ x : ℝ, parabola_eq x y) ↔ y = 1 / 16 :=
by
  sorry

end directrix_of_parabola_l1476_147654


namespace prime_exists_solution_l1476_147669

theorem prime_exists_solution (p : ℕ) [hp : Fact p.Prime] :
  ∃ n : ℕ, (6 * n^2 + 5 * n + 1) % p = 0 :=
by
  sorry

end prime_exists_solution_l1476_147669


namespace no_polynomials_exist_l1476_147616

open Polynomial

theorem no_polynomials_exist
  (a b : Polynomial ℂ) (c d : Polynomial ℂ) :
  ¬ (∀ x y : ℂ, 1 + x * y + x^2 * y^2 = a.eval x * c.eval y + b.eval x * d.eval y) :=
sorry

end no_polynomials_exist_l1476_147616


namespace cos_seven_pi_over_six_l1476_147621

open Real

theorem cos_seven_pi_over_six : cos (7 * π / 6) = - (sqrt 3 / 2) := 
by
  sorry

end cos_seven_pi_over_six_l1476_147621


namespace ramu_profit_percent_l1476_147615

noncomputable def profitPercent
  (purchase_price : ℝ)
  (repair_cost : ℝ)
  (selling_price : ℝ) : ℝ :=
  ((selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost)) * 100

theorem ramu_profit_percent :
  profitPercent 42000 13000 61900 = 12.55 :=
by
  sorry

end ramu_profit_percent_l1476_147615


namespace unique_solution_l1476_147680

def system_of_equations (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ) (x1 x2 x3 : ℝ) :=
  a11 * x1 + a12 * x2 + a13 * x3 = 0 ∧
  a21 * x1 + a22 * x2 + a23 * x3 = 0 ∧
  a31 * x1 + a32 * x2 + a33 * x3 = 0

theorem unique_solution
  (x1 x2 x3 : ℝ)
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ)
  (h_pos: 0 < a11 ∧ 0 < a22 ∧ 0 < a33)
  (h_neg: a12 < 0 ∧ a13 < 0 ∧ a21 < 0 ∧ a23 < 0 ∧ a31 < 0 ∧ a32 < 0)
  (h_sum_pos: 0 < a11 + a12 + a13 ∧ 0 < a21 + a22 + a23 ∧ 0 < a31 + a32 + a33)
  (h_system: system_of_equations a11 a12 a13 a21 a22 a23 a31 a32 a33 x1 x2 x3):
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 := sorry

end unique_solution_l1476_147680


namespace warriors_wins_count_l1476_147661

variable {wins : ℕ → ℕ}
variable (raptors hawks warriors spurs lakers : ℕ)

def conditions (wins : ℕ → ℕ) (raptors hawks warriors spurs lakers : ℕ) : Prop :=
  wins raptors > wins hawks ∧
  wins warriors > wins spurs ∧ wins warriors < wins lakers ∧
  wins spurs > 25

theorem warriors_wins_count
  (wins : ℕ → ℕ)
  (raptors hawks warriors spurs lakers : ℕ)
  (h : conditions wins raptors hawks warriors spurs lakers) :
  wins warriors = 37 := sorry

end warriors_wins_count_l1476_147661


namespace no_valid_formation_l1476_147683

-- Define the conditions related to the formation:
-- s : number of rows
-- t : number of musicians per row
-- Total musicians = s * t = 400
-- t is divisible by 4
-- 10 ≤ t ≤ 50
-- Additionally, the brass section needs to form a triangle in the first three rows
-- while maintaining equal distribution of musicians from each section in every row.

theorem no_valid_formation (s t : ℕ) (h_mul : s * t = 400) 
  (h_div : t % 4 = 0) 
  (h_range : 10 ≤ t ∧ t ≤ 50) 
  (h_triangle : ∀ (r1 r2 r3 : ℕ), r1 < r2 ∧ r2 < r3 → r1 + r2 + r3 = 100 → false) : 
  x = 0 := by
  sorry

end no_valid_formation_l1476_147683


namespace percent_decrease_l1476_147645

theorem percent_decrease (P S : ℝ) (h₀ : P = 100) (h₁ : S = 70) :
  ((P - S) / P) * 100 = 30 :=
by
  sorry

end percent_decrease_l1476_147645


namespace sum_leq_six_of_quadratic_roots_l1476_147662

theorem sum_leq_six_of_quadratic_roots (a b : ℤ) (h1 : a ≠ -1) (h2 : b ≠ -1) 
  (h3 : ∃ r1 r2 : ℤ, r1 ≠ r2 ∧ x^2 + ab * x + (a + b) = 0 ∧ 
         x = r1 ∧ x = r2) : a + b ≤ 6 :=
by
  sorry

end sum_leq_six_of_quadratic_roots_l1476_147662


namespace part_I_part_II_l1476_147636

noncomputable def f (x a : ℝ) : ℝ := x - 1 - a * Real.log x

theorem part_I (a : ℝ) (h1 : 0 < a) (h2 : ∀ x : ℝ, 0 < x → f x a ≥ 0) : a = 1 := 
sorry

theorem part_II (n : ℕ) (hn : 0 < n) : 
  let an := (1 + 1 / (n : ℝ)) ^ n
  let bn := (1 + 1 / (n : ℝ)) ^ (n + 1)
  an < Real.exp 1 ∧ Real.exp 1 < bn := 
sorry

end part_I_part_II_l1476_147636


namespace simplify_fraction_l1476_147660

variable {a b c : ℝ} -- assuming a, b, c are real numbers

theorem simplify_fraction (hc : a + b + c ≠ 0) :
  (a^2 + b^2 - c^2 + 2 * a * b) / (a^2 + c^2 - b^2 + 2 * a * c) = (a + b - c) / (a - b + c) :=
sorry

end simplify_fraction_l1476_147660


namespace simplify_frac_l1476_147646

variable (b c : ℕ)
variable (b_val : b = 2)
variable (c_val : c = 3)

theorem simplify_frac : (15 * b ^ 4 * c ^ 2) / (45 * b ^ 3 * c) = 2 :=
by
  rw [b_val, c_val]
  sorry

end simplify_frac_l1476_147646


namespace complement_intersection_l1476_147690

open Set

variable (R : Type) [LinearOrderedField R]

def A : Set R := {x | |x| < 1}
def B : Set R := {y | ∃ x, y = 2^x + 1}
def complement_A : Set R := {x | x ≤ -1 ∨ x ≥ 1}

theorem complement_intersection (x : R) : 
  x ∈ (complement_A R) ∩ B R ↔ x > 1 :=
by
  sorry

end complement_intersection_l1476_147690


namespace find_a2_geometric_sequence_l1476_147650

theorem find_a2_geometric_sequence (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = a n * r) 
  (h_a1 : a 1 = 1 / 4) (h_eq : a 3 * a 5 = 4 * (a 4 - 1)) : a 2 = 1 / 8 :=
by
  sorry

end find_a2_geometric_sequence_l1476_147650


namespace product_of_four_consecutive_odd_numbers_is_perfect_square_l1476_147671

theorem product_of_four_consecutive_odd_numbers_is_perfect_square (n : ℤ) :
    (n + 0) * (n + 2) * (n + 4) * (n + 6) = 9 :=
sorry

end product_of_four_consecutive_odd_numbers_is_perfect_square_l1476_147671


namespace cube_volume_l1476_147658

theorem cube_volume (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
sorry

end cube_volume_l1476_147658


namespace pyramid_pattern_l1476_147673

theorem pyramid_pattern
  (R : ℕ → ℕ)  -- a function representing the number of blocks in each row
  (R₁ : R 1 = 9)  -- the first row has 9 blocks
  (sum_eq : R 1 + R 2 + R 3 + R 4 + R 5 = 25)  -- the total number of blocks is 25
  (pattern : ∀ n, 1 ≤ n ∧ n < 5 → R (n + 1) = R n - 2) : ∃ d, d = 2 :=
by
  have pattern_valid : R 1 = 9 ∧ R 2 = 7 ∧ R 3 = 5 ∧ R 4 = 3 ∧ R 5 = 1 :=
    sorry  -- Proof omitted
  exact ⟨2, rfl⟩

end pyramid_pattern_l1476_147673


namespace smallest_N_satisfying_frequencies_l1476_147604

def percentageA := 1 / 5
def percentageB := 3 / 8
def percentageC := 1 / 4
def percentageD := 1 / 8
def percentageE := 1 / 20

def Divisible (n : ℕ) (d : ℕ) : Prop := ∃ (k : ℕ), n = k * d

theorem smallest_N_satisfying_frequencies :
  ∃ N : ℕ, 
    Divisible N 5 ∧ 
    Divisible N 8 ∧ 
    Divisible N 4 ∧ 
    Divisible N 20 ∧ 
    N = 40 := sorry

end smallest_N_satisfying_frequencies_l1476_147604


namespace locus_of_midpoint_of_tangents_l1476_147623

theorem locus_of_midpoint_of_tangents 
  (P Q Q1 Q2 : ℝ × ℝ)
  (L : P.2 = P.1 + 2)
  (C : ∀ p, p = Q1 ∨ p = Q2 → p.2 ^ 2 = 4 * p.1)
  (Q_is_midpoint : Q = ((Q1.1 + Q2.1) / 2, (Q1.2 + Q2.2) / 2)) :
  ∃ x y, (y - 1)^2 = 2 * (x - 3 / 2) := sorry

end locus_of_midpoint_of_tangents_l1476_147623


namespace no_real_solutions_l1476_147674

theorem no_real_solutions : ¬ ∃ (r s : ℝ),
  (r - 50) / 3 = (s - 2 * r) / 4 ∧
  r^2 + 3 * s = 50 :=
by {
  -- sorry, proof steps would go here
  sorry
}

end no_real_solutions_l1476_147674


namespace division_value_l1476_147601

theorem division_value (x y : ℝ) (h1 : (x - 5) / y = 7) (h2 : (x - 14) / 10 = 4) : y = 7 :=
sorry

end division_value_l1476_147601


namespace determine_m_with_opposite_roots_l1476_147655

theorem determine_m_with_opposite_roots (c d k : ℝ) (h : c + d ≠ 0):
  (∃ m : ℝ, ∀ x : ℝ, (x^2 - d * x) / (c * x - k) = (m - 2) / (m + 2) ∧ 
            (x = -y ∧ y = -x)) ↔ m = 2 * (c - d) / (c + d) :=
sorry

end determine_m_with_opposite_roots_l1476_147655


namespace last_year_ticket_cost_l1476_147613

theorem last_year_ticket_cost (this_year_cost : ℝ) (increase_percentage : ℝ) (last_year_cost : ℝ) :
  this_year_cost = last_year_cost * (1 + increase_percentage) ↔ last_year_cost = 85 :=
by
  let this_year_cost := 102
  let increase_percentage := 0.20
  sorry

end last_year_ticket_cost_l1476_147613


namespace range_of_k_l1476_147633

theorem range_of_k (k : ℝ) : (∀ (x : ℝ), k * x ^ 2 - k * x - 1 < 0) ↔ (-4 < k ∧ k ≤ 0) := 
by 
  sorry

end range_of_k_l1476_147633


namespace range_of_a_l1476_147647

theorem range_of_a (a : ℝ) :
  (a + 1)^2 > (3 - 2 * a)^2 ↔ (2 / 3) < a ∧ a < 4 :=
sorry

end range_of_a_l1476_147647


namespace closest_integers_to_2013_satisfy_trig_eq_l1476_147607

noncomputable def closestIntegersSatisfyingTrigEq (x : ℝ) : Prop := 
  (2^(Real.sin x)^2 + 2^(Real.cos x)^2 = 2 * Real.sqrt 2)

theorem closest_integers_to_2013_satisfy_trig_eq : closestIntegersSatisfyingTrigEq (1935 * (Real.pi / 180)) ∧ closestIntegersSatisfyingTrigEq (2025 * (Real.pi / 180)) :=
sorry

end closest_integers_to_2013_satisfy_trig_eq_l1476_147607


namespace skittles_distribution_l1476_147608

theorem skittles_distribution :
  let initial_skittles := 14
  let additional_skittles := 22
  let total_skittles := initial_skittles + additional_skittles
  let number_of_people := 7
  (total_skittles / number_of_people = 5) :=
by
  sorry

end skittles_distribution_l1476_147608


namespace value_of_m_l1476_147656

theorem value_of_m :
  ∀ m : ℝ, (x : ℝ) → (x^2 - 5 * x + m = (x - 3) * (x - 2)) → m = 6 :=
by
  sorry

end value_of_m_l1476_147656


namespace total_vessels_l1476_147652

open Nat

theorem total_vessels (x y z w : ℕ) (hx : x > 0) (hy : y > x) (hz : z > y) (hw : w > z) :
  ∃ total : ℕ, total = x * (2 * y + 1) + z * (1 + 1 / w) := sorry

end total_vessels_l1476_147652


namespace log_domain_eq_l1476_147667

noncomputable def quadratic_expr (x : ℝ) : ℝ := x^2 - 2 * x - 3

def log_domain (x : ℝ) : Prop := quadratic_expr x > 0

theorem log_domain_eq :
  {x : ℝ | log_domain x} = 
  {x : ℝ | x < -1} ∪ {x : ℝ | x > 3} :=
by {
  sorry
}

end log_domain_eq_l1476_147667


namespace arithmetic_sequence_max_sum_l1476_147699

-- Condition: first term is 23
def a1 : ℤ := 23

-- Condition: common difference is -2
def d : ℤ := -2

-- Sum of the first n terms of the arithmetic sequence
def Sn (n : ℕ) : ℤ := n * a1 + (n * (n - 1)) / 2 * d

-- Problem Statement: Prove the maximum value of Sn(n)
theorem arithmetic_sequence_max_sum : ∃ n : ℕ, Sn n = 144 :=
sorry

end arithmetic_sequence_max_sum_l1476_147699


namespace average_rate_of_trip_l1476_147677

theorem average_rate_of_trip (d : ℝ) (r1 : ℝ) (t1 : ℝ) (r_total : ℝ) :
  d = 640 →
  r1 = 80 →
  t1 = (320 / r1) →
  t2 = 3 * t1 →
  r_total = d / (t1 + t2) →
  r_total = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_rate_of_trip_l1476_147677


namespace jonah_profit_l1476_147602

def cost_per_pineapple (quantity : ℕ) : ℝ :=
  if quantity > 50 then 1.60 else if quantity > 40 then 1.80 else 2.00

def total_cost (quantity : ℕ) : ℝ :=
  cost_per_pineapple quantity * quantity

def bundle_revenue (bundles : ℕ) : ℝ :=
  bundles * 20

def single_ring_revenue (rings : ℕ) : ℝ :=
  rings * 4

def total_revenue (bundles : ℕ) (rings : ℕ) : ℝ :=
  bundle_revenue bundles + single_ring_revenue rings

noncomputable def profit (quantity bundles rings : ℕ) : ℝ :=
  total_revenue bundles rings - total_cost quantity

theorem jonah_profit : profit 60 35 150 = 1204 := by
  sorry

end jonah_profit_l1476_147602


namespace complex_number_in_first_quadrant_l1476_147612

noncomputable def z : ℂ := Complex.ofReal 1 + Complex.I

theorem complex_number_in_first_quadrant 
  (h : Complex.ofReal 1 + Complex.I = Complex.I / z) : 
  (0 < z.re ∧ 0 < z.im) :=
  sorry

end complex_number_in_first_quadrant_l1476_147612


namespace division_of_composite_products_l1476_147670

noncomputable def product_of_first_seven_composites : ℕ :=
  4 * 6 * 8 * 9 * 10 * 12 * 14

noncomputable def product_of_next_seven_composites : ℕ :=
  15 * 16 * 18 * 20 * 21 * 22 * 24

noncomputable def divided_product_composites : ℚ :=
  product_of_first_seven_composites / product_of_next_seven_composites

theorem division_of_composite_products : divided_product_composites = 1 / 176 := by
  sorry

end division_of_composite_products_l1476_147670


namespace find_x_squared_plus_inv_squared_l1476_147692

noncomputable def x : ℝ := sorry

theorem find_x_squared_plus_inv_squared (h : x^4 + 1 / x^4 = 240) : x^2 + 1 / x^2 = Real.sqrt 242 := by
  sorry

end find_x_squared_plus_inv_squared_l1476_147692


namespace oil_amount_correct_l1476_147632

-- Definitions based on the conditions in the problem
def initial_amount : ℝ := 0.16666666666666666
def additional_amount : ℝ := 0.6666666666666666
def final_amount : ℝ := 0.8333333333333333

-- Lean 4 statement to prove the given problem
theorem oil_amount_correct :
  initial_amount + additional_amount = final_amount :=
by
  sorry

end oil_amount_correct_l1476_147632


namespace multiplication_of_variables_l1476_147693

theorem multiplication_of_variables 
  (a b c d : ℚ)
  (h1 : 3 * a + 2 * b + 4 * c + 6 * d = 48)
  (h2 : 4 * (d + c) = b)
  (h3 : 4 * b + 2 * c = a)
  (h4 : 2 * c - 2 = d) :
  a * b * c * d = -58735360 / 81450625 := 
sorry

end multiplication_of_variables_l1476_147693
