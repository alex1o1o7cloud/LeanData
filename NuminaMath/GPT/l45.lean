import Mathlib

namespace toaster_popularity_l45_45946

theorem toaster_popularity
  (c₁ c₂ : ℤ) (p₁ p₂ k : ℤ)
  (h₀ : p₁ * c₁ = k)
  (h₁ : p₁ = 12)
  (h₂ : c₁ = 500)
  (h₃ : c₂ = 750)
  (h₄ : k = p₁ * c₁) :
  p₂ * c₂ = k → p₂ = 8 :=
by
  sorry

end toaster_popularity_l45_45946


namespace enlarged_poster_height_l45_45865

def original_poster_width : ℝ := 3
def original_poster_height : ℝ := 2
def new_poster_width : ℝ := 12

theorem enlarged_poster_height :
  new_poster_width / original_poster_width * original_poster_height = 8 := 
by
  sorry

end enlarged_poster_height_l45_45865


namespace total_amount_spent_l45_45730

def cost_of_haley_paper : ℝ := 3.75 + (3.75 * 0.5)
def cost_of_sister_paper : ℝ := (4.50 * 2) + (4.50 * 0.5)
def cost_of_haley_pens : ℝ := (1.45 * 5) - ((1.45 * 5) * 0.25)
def cost_of_sister_pens : ℝ := (1.65 * 7) - ((1.65 * 7) * 0.25)

def total_cost_of_supplies : ℝ := cost_of_haley_paper + cost_of_sister_paper + cost_of_haley_pens + cost_of_sister_pens

theorem total_amount_spent : total_cost_of_supplies = 30.975 :=
by
  sorry

end total_amount_spent_l45_45730


namespace functional_equation_solution_l45_45720

theorem functional_equation_solution (f : ℝ → ℝ) (t : ℝ) (h : t ≠ -1) :
  (∀ x y : ℝ, (t + 1) * f (1 + x * y) - f (x + y) = f (x + 1) * f (y + 1)) →
  (∀ x, f x = 0) ∨ (∀ x, f x = t) ∨ (∀ x, f x = (t + 1) * x - (t + 2)) :=
by
  sorry

end functional_equation_solution_l45_45720


namespace adam_more_apples_than_combined_l45_45450

def adam_apples : Nat := 10
def jackie_apples : Nat := 2
def michael_apples : Nat := 5

theorem adam_more_apples_than_combined : 
  adam_apples - (jackie_apples + michael_apples) = 3 :=
by
  sorry

end adam_more_apples_than_combined_l45_45450


namespace peggy_stamps_l45_45453

-- Defining the number of stamps Peggy, Ernie, and Bert have
variables (P : ℕ) (E : ℕ) (B : ℕ)

-- Given conditions
def bert_has_four_times_ernie (B : ℕ) (E : ℕ) : Prop := B = 4 * E
def ernie_has_three_times_peggy (E : ℕ) (P : ℕ) : Prop := E = 3 * P
def peggy_needs_stamps (P : ℕ) (B : ℕ) : Prop := B = P + 825

-- Question to Answer / Theorem Statement
theorem peggy_stamps (P : ℕ) (E : ℕ) (B : ℕ)
  (h1 : bert_has_four_times_ernie B E)
  (h2 : ernie_has_three_times_peggy E P)
  (h3 : peggy_needs_stamps P B) :
  P = 75 :=
sorry

end peggy_stamps_l45_45453


namespace length_first_train_l45_45655

/-- Let the speeds of two trains be 120 km/hr and 80 km/hr, respectively. 
These trains cross each other in 9 seconds, and the length of the second train is 250.04 meters. 
Prove that the length of the first train is 250 meters. -/
theorem length_first_train
  (FirstTrainSpeed : ℝ := 120)  -- speed of the first train in km/hr
  (SecondTrainSpeed : ℝ := 80)  -- speed of the second train in km/hr
  (TimeToCross : ℝ := 9)        -- time to cross each other in seconds
  (LengthSecondTrain : ℝ := 250.04) -- length of the second train in meters
  : FirstTrainSpeed / 0.36 + SecondTrainSpeed / 0.36 * TimeToCross - LengthSecondTrain = 250 :=
by
  -- omitted proof
  sorry

end length_first_train_l45_45655


namespace inequality_proof_l45_45364

variable (a b c d : ℝ)
variable (habcda : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ ab + bc + cd + da = 1)

theorem inequality_proof :
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧ (ab + bc + cd + da = 1) →
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3) :=
by sorry

end inequality_proof_l45_45364


namespace min_value_inequality_l45_45031

theorem min_value_inequality (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 3 * a + 2 * b = 1) : 
  ∃ (m : ℝ), m = 25 ∧ (∀ x y, (x > 0) → (y > 0) → (3 * x + 2 * y = 1) → (3 / x + 2 / y) ≥ m) :=
sorry

end min_value_inequality_l45_45031


namespace number_of_solutions_l45_45147

theorem number_of_solutions (f : ℕ → ℕ) (n : ℕ) : 
  (∀ n, f n = n^4 + 2 * n^3 - 20 * n^2 + 2 * n - 21) →
  (∀ n, 0 ≤ n ∧ n < 2013 → 2013 ∣ f n) → 
  ∃ k, k = 6 :=
by
  sorry

end number_of_solutions_l45_45147


namespace find_num_female_students_l45_45906

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

end find_num_female_students_l45_45906


namespace tangent_line_to_circle_l45_45718

theorem tangent_line_to_circle :
  ∀ (x y : ℝ), x^2 + y^2 = 5 → (x = 2 → y = -1 → 2 * x - y - 5 = 0) :=
by
  intros x y h_circle hx hy
  sorry

end tangent_line_to_circle_l45_45718


namespace smallest_nat_number_l45_45822

theorem smallest_nat_number (n : ℕ) (h1 : ∃ a, 0 ≤ a ∧ a < 20 ∧ n % 20 = a ∧ n % 21 = a + 1) (h2 : n % 22 = 2) : n = 838 := by 
  sorry

end smallest_nat_number_l45_45822


namespace merchant_loss_is_15_yuan_l45_45927

noncomputable def profit_cost_price : ℝ := (180 : ℝ) / 1.2
noncomputable def loss_cost_price : ℝ := (180 : ℝ) / 0.8

theorem merchant_loss_is_15_yuan :
  (180 + 180) - (profit_cost_price + loss_cost_price) = -15 := by
  sorry

end merchant_loss_is_15_yuan_l45_45927


namespace narrow_black_stripes_l45_45562

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end narrow_black_stripes_l45_45562


namespace largest_prime_factor_3136_l45_45100

theorem largest_prime_factor_3136 : ∃ p, nat.prime p ∧ p ∣ 3136 ∧ (∀ q, nat.prime q ∧ q ∣ 3136 → q ≤ p) :=
by {
  sorry
}

end largest_prime_factor_3136_l45_45100


namespace inheritance_split_l45_45206

theorem inheritance_split (total_money : ℝ) (num_people : ℕ) (amount_per_person : ℝ) 
  (h1 : total_money = 874532.13) (h2 : num_people = 7) 
  (h3 : amount_per_person = total_money / num_people) : 
  amount_per_person = 124933.16 := by 
  sorry

end inheritance_split_l45_45206


namespace sector_area_maximized_l45_45516

noncomputable def maximize_sector_area (r θ : ℝ) : Prop :=
  2 * r + θ * r = 20 ∧
  (r > 0 ∧ θ > 0) ∧
  ∀ (r' θ' : ℝ), (2 * r' + θ' * r' = 20 ∧ r' > 0 ∧ θ' > 0) → (1/2 * θ' * r'^2 ≤ 1/2 * θ * r^2)

theorem sector_area_maximized : maximize_sector_area 5 2 :=
by
  sorry

end sector_area_maximized_l45_45516


namespace connie_grandma_birth_year_l45_45457

theorem connie_grandma_birth_year :
  ∀ (B S G : ℕ),
  B = 1932 →
  S = 1936 →
  (S - B) * 2 = (S - G) →
  G = 1928 := 
by
  intros B S G hB hS hGap
  -- Proof goes here
  sorry

end connie_grandma_birth_year_l45_45457


namespace find_a_l45_45840

theorem find_a (f : ℝ → ℝ) (h1 : ∀ x, f (2^x) = x + 3) (h2 : f a = 5) : a = 4 := 
by
  sorry

end find_a_l45_45840


namespace binom_mult_eq_6720_l45_45694

theorem binom_mult_eq_6720 :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_mult_eq_6720_l45_45694


namespace inequality_proof_l45_45312

variable (a b c d e f : Real)

theorem inequality_proof (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end inequality_proof_l45_45312


namespace odd_square_not_sum_of_five_odd_squares_l45_45874

theorem odd_square_not_sum_of_five_odd_squares :
  ∀ (n : ℤ), (∃ k : ℤ, k^2 % 8 = n % 8 ∧ n % 8 = 1) →
             ¬(∃ a b c d e : ℤ, (a^2 % 8 = 1) ∧ (b^2 % 8 = 1) ∧ (c^2 % 8 = 1) ∧ (d^2 % 8 = 1) ∧ 
               (e^2 % 8 = 1) ∧ (n % 8 = (a^2 + b^2 + c^2 + d^2 + e^2) % 8)) :=
by
  sorry

end odd_square_not_sum_of_five_odd_squares_l45_45874


namespace number_of_attempted_problems_l45_45652

-- Lean statement to define the problem setup
def student_assignment_problem (x y : ℕ) : Prop :=
  8 * x - 5 * y = 13 ∧ x + y ≤ 20

-- The Lean statement asserting the solution to the problem
theorem number_of_attempted_problems : ∃ x y : ℕ, student_assignment_problem x y ∧ x + y = 13 := 
by
  sorry

end number_of_attempted_problems_l45_45652


namespace range_of_m_l45_45910

noncomputable def f (x m : ℝ) : ℝ := -x^2 + m * x

theorem range_of_m {m : ℝ} : (∀ x y : ℝ, x ≤ y → x ≤ 1 → y ≤ 1 → f x m ≤ f y m) ↔ 2 ≤ m := 
sorry

end range_of_m_l45_45910


namespace simplify_sqrt180_l45_45380

-- Conditions from the problem definition
def sqrt (x : ℕ) : ℝ := Real.sqrt x
def six := 6
def five := 5

-- The statement of the problem as a Lean theorem
theorem simplify_sqrt180 : sqrt 180 = six * sqrt five := by 
  sorry

end simplify_sqrt180_l45_45380


namespace jacque_suitcase_weight_l45_45357

noncomputable def suitcase_weight_return (original_weight : ℝ)
                                         (perfume_weight_oz : ℕ → ℝ)
                                         (chocolate_weight_lb : ℝ)
                                         (soap_weight_oz : ℕ → ℝ)
                                         (jam_weight_oz : ℕ → ℝ)
                                         (sculpture_weight_kg : ℝ)
                                         (shirt_weight_g : ℕ → ℝ)
                                         (oz_to_lb : ℝ)
                                         (kg_to_lb : ℝ)
                                         (g_to_kg : ℝ) : ℝ :=
  original_weight +
  (perfume_weight_oz 5 / oz_to_lb) +
  chocolate_weight_lb +
  (soap_weight_oz 2 / oz_to_lb) +
  (jam_weight_oz 2 / oz_to_lb) +
  (sculpture_weight_kg * kg_to_lb) +
  ((shirt_weight_g 3 / g_to_kg) * kg_to_lb)

theorem jacque_suitcase_weight :
  suitcase_weight_return 12 
                        (fun n => n * 1.2) 
                        4 
                        (fun n => n * 5) 
                        (fun n => n * 8)
                        3.5 
                        (fun n => n * 300) 
                        16 
                        2.20462 
                        1000 
  = 27.70 :=
sorry

end jacque_suitcase_weight_l45_45357


namespace find_number_of_rabbits_l45_45909

variable (R P : ℕ)

theorem find_number_of_rabbits (h1 : R + P = 60) (h2 : 4 * R + 2 * P = 192) : R = 36 := 
by
  sorry

end find_number_of_rabbits_l45_45909


namespace subtraction_divisible_l45_45109

theorem subtraction_divisible (n m d : ℕ) (h1 : n = 13603) (h2 : m = 31) (h3 : d = 13572) : 
  (n - m) % d = 0 := by
  sorry

end subtraction_divisible_l45_45109


namespace owen_turtles_l45_45067

theorem owen_turtles (o_initial : ℕ) (j_initial : ℕ) (o_after_month : ℕ) (j_remaining : ℕ) (o_final : ℕ) 
  (h1 : o_initial = 21)
  (h2 : j_initial = o_initial - 5)
  (h3 : o_after_month = 2 * o_initial)
  (h4 : j_remaining = j_initial / 2)
  (h5 : o_final = o_after_month + j_remaining) :
  o_final = 50 :=
sorry

end owen_turtles_l45_45067


namespace no_such_function_exists_l45_45535

theorem no_such_function_exists 
  (f : ℝ → ℝ) 
  (h_f_pos : ∀ x, 0 < x → 0 < f x) 
  (h_eq : ∀ x y, 0 < x → 0 < y → f (x + y) = f x + f y + (1 / 2012)) : 
  false :=
sorry

end no_such_function_exists_l45_45535


namespace jackson_souvenirs_total_l45_45187

def jacksons_collections := 
  let hermit_crabs := 120
  let spiral_shells_per_hermit_crab := 8
  let starfish_per_spiral_shell := 5
  let sand_dollars_per_starfish := 3
  let coral_structures_per_sand_dollars := 4
  let spiral_shells := hermit_crabs * spiral_shells_per_hermit_crab
  let starfish := spiral_shells * starfish_per_spiral_shell
  let sand_dollars := starfish * sand_dollars_per_starfish
  let coral_structures := sand_dollars / coral_structures_per_sand_dollars
  hermit_crabs + spiral_shells + starfish + sand_dollars + coral_structures

theorem jackson_souvenirs_total : jacksons_collections = 22880 := by sorry

end jackson_souvenirs_total_l45_45187


namespace average_speed_round_trip_36_l45_45928

variables (z : ℝ)

def eastward_speed_minutes_per_mile : ℝ := 3
def westward_speed_miles_per_minute : ℝ := 3

def total_distance (z : ℝ) : ℝ := 2 * z
def eastward_time (z : ℝ) (eastward_speed : ℝ) : ℝ := z * eastward_speed
def westward_time (z : ℝ) (westward_speed : ℝ) : ℝ := z / westward_speed
def total_time (z : ℝ) (eastward_speed : ℝ) (westward_speed : ℝ) : ℝ := eastward_time z eastward_speed + westward_time z westward_speed
def total_time_in_hours (total_time : ℝ) : ℝ := total_time / 60

def average_speed (total_distance : ℝ) (total_time_in_hours : ℝ) : ℝ := total_distance / total_time_in_hours

theorem average_speed_round_trip_36 (z : ℝ) :
  average_speed (total_distance z) (total_time_in_hours (total_time z eastward_speed_minutes_per_mile westward_speed_miles_per_minute)) = 36 := 
  sorry

end average_speed_round_trip_36_l45_45928


namespace strawberries_remaining_l45_45895

theorem strawberries_remaining (initial : ℝ) (eaten_yesterday : ℝ) (eaten_today : ℝ) :
  initial = 1.6 ∧ eaten_yesterday = 0.8 ∧ eaten_today = 0.3 → initial - eaten_yesterday - eaten_today = 0.5 :=
by
  sorry

end strawberries_remaining_l45_45895


namespace ellipse_eccentricity_l45_45728

noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

def ellipse_conditions (F1 B : ℝ × ℝ) (c b : ℝ) : Prop :=
  F1 = (-2, 0) ∧ B = (0, 1) ∧ c = 2 ∧ b = 1

theorem ellipse_eccentricity (F1 B : ℝ × ℝ) (c b a : ℝ)
  (h : ellipse_conditions F1 B c b) :
  eccentricity c a = 2 * Real.sqrt 5 / 5 := by
sorry

end ellipse_eccentricity_l45_45728


namespace smallestBeta_satisfies_l45_45533

noncomputable def validAlphaBeta (alpha beta : ℕ) : Prop :=
  16 / 37 < (alpha : ℚ) / beta ∧ (alpha : ℚ) / beta < 7 / 16

def smallestBeta : ℕ := 23

theorem smallestBeta_satisfies :
  (∀ (alpha beta : ℕ), validAlphaBeta alpha beta → beta ≥ 23) ∧
  (∃ (alpha : ℕ), validAlphaBeta alpha 23) :=
by sorry

end smallestBeta_satisfies_l45_45533


namespace math_problem_l45_45898

theorem math_problem (x y : ℤ) (h1 : x = 12) (h2 : y = 18) : (x - y) * ((x + y) ^ 2) = -5400 := by
  sorry

end math_problem_l45_45898


namespace five_fourths_of_twelve_fifths_eq_three_l45_45011

theorem five_fourths_of_twelve_fifths_eq_three : (5 : ℝ) / 4 * (12 / 5) = 3 := 
by 
  sorry

end five_fourths_of_twelve_fifths_eq_three_l45_45011


namespace calculate_expression_l45_45198

theorem calculate_expression
  (x y : ℚ)
  (D E : ℚ × ℚ)
  (hx : x = (D.1 + E.1) / 2)
  (hy : y = (D.2 + E.2) / 2)
  (hD : D = (15, -3))
  (hE : E = (-4, 12)) :
  3 * x - 5 * y = -6 :=
by
  subst hD
  subst hE
  subst hx
  subst hy
  sorry

end calculate_expression_l45_45198


namespace cole_drive_time_l45_45130

noncomputable def T_work (D : ℝ) : ℝ := D / 75
noncomputable def T_home (D : ℝ) : ℝ := D / 105

theorem cole_drive_time (v1 v2 T : ℝ) (D : ℝ) 
  (h_v1 : v1 = 75) (h_v2 : v2 = 105) (h_T : T = 4)
  (h_round_trip : T_work D + T_home D = T) : 
  T_work D = 140 / 60 :=
sorry

end cole_drive_time_l45_45130


namespace factor_x_squared_minus_64_l45_45292

-- Conditions
def a := x
def b := 8

-- Theorem statement
theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_64_l45_45292


namespace min_value_problem1_l45_45436

theorem min_value_problem1 (x : ℝ) (hx : x > -1) : 
  ∃ m, m = 2 * Real.sqrt 2 + 1 ∧ (∀ y, y = (x^2 + 3 * x + 4) / (x + 1) ∧ x > -1 → y ≥ m) :=
sorry

end min_value_problem1_l45_45436


namespace tennis_tournament_l45_45042

noncomputable def tennis_tournament_n (k : ℕ) : ℕ := 8 * k + 1

theorem tennis_tournament (n : ℕ) :
  (∃ k : ℕ, n = tennis_tournament_n k) ↔
  (∃ k : ℕ, n = 8 * k + 1) :=
by sorry

end tennis_tournament_l45_45042


namespace vacation_months_away_l45_45356

theorem vacation_months_away (total_savings : ℕ) (pay_per_check : ℕ) (checks_per_month : ℕ) :
  total_savings = 3000 → pay_per_check = 100 → checks_per_month = 2 → 
  total_savings / pay_per_check / checks_per_month = 15 :=
by 
  intros h1 h2 h3
  sorry

end vacation_months_away_l45_45356


namespace inequality_proof_l45_45316

theorem inequality_proof (a b c d e f : ℝ) (H : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end inequality_proof_l45_45316


namespace tom_driving_speed_l45_45998

theorem tom_driving_speed
  (v : ℝ)
  (hKarenSpeed : 60 = 60) -- Karen drives at an average speed of 60 mph
  (hKarenLateStart: 4 / 60 = 1 / 15) -- Karen starts 4 minutes late, which is 1/15 hours
  (hTomDistance : 24 = 24) -- Tom drives 24 miles before Karen wins the bet
  (hTimeEquation: 24 / v = 8 / 15): -- The equation derived from given conditions
  v = 45 := 
by
  sorry

end tom_driving_speed_l45_45998


namespace narrow_black_stripes_are_8_l45_45576

-- Define variables: w for wide black stripes, n for narrow black stripes, b for white stripes
variables (w n b : ℕ)

-- Given conditions
axiom cond1 : b = w + 7
axiom cond2 : w + n = b + 1

-- Theorem statement to prove that the number of narrow black stripes is 8
theorem narrow_black_stripes_are_8 : n = 8 :=
by sorry

end narrow_black_stripes_are_8_l45_45576


namespace number_of_narrow_black_stripes_l45_45579

-- Define the variables
variables {w n b : ℕ}

-- The conditions from the problem
def condition1 := w + n = b + 1
def condition2 := b = w + 7

-- The Lean statement to prove
theorem number_of_narrow_black_stripes (h1 : condition1) (h2 : condition2) : n = 8 :=
by {
  -- We import the conditions as hypotheses
  sorry
}

end number_of_narrow_black_stripes_l45_45579


namespace toms_speed_l45_45995

/--
Karen places a bet with Tom that she will beat Tom in a car race by 4 miles 
even if Karen starts 4 minutes late. Assuming that Karen drives at 
an average speed of 60 mph and that Tom will drive 24 miles before 
Karen wins the bet. Prove that Tom's average driving speed is \( \frac{300}{7} \) mph.
--/
theorem toms_speed (
  (karen_speed : ℕ) (karen_lateness : ℚ) (karen_beats_tom_by : ℕ) 
  (karen_distance_when_tom_drives_24_miles : ℕ) 
  (karen_speed = 60) 
  (karen_lateness = 4 / 60) 
  (karen_beats_tom_by = 4) 
  (karen_distance_when_tom_drives_24_miles = 24)) : 
  ∃ tom_speed : ℚ, tom_speed = 300 / 7 :=
begin
  sorry
end

end toms_speed_l45_45995


namespace length_of_floor_y_l45_45434

theorem length_of_floor_y
  (A B : ℝ)
  (hx : A = 10)
  (hy : B = 18)
  (width_y : ℝ)
  (length_y : ℝ)
  (width_y_eq : width_y = 9)
  (area_eq : A * B = width_y * length_y) :
  length_y = 20 := 
sorry

end length_of_floor_y_l45_45434


namespace number_of_integer_solutions_l45_45035

theorem number_of_integer_solutions : ∃ (n : ℕ), n = 120 ∧ ∀ (x y z : ℤ), x * y * z = 2008 → n = 120 :=
by
  sorry

end number_of_integer_solutions_l45_45035


namespace sum_of_squares_of_roots_l45_45281

theorem sum_of_squares_of_roots :
  let a := 1
  let b := 8
  let c := -12
  let r1_r2_sum := -(b:ℝ) / a
  let r1_r2_product := (c:ℝ) / a
  (r1_r2_sum) ^ 2 - 2 * r1_r2_product = 88 :=
by
  sorry

end sum_of_squares_of_roots_l45_45281


namespace simplify_sqrt180_l45_45379

-- Conditions from the problem definition
def sqrt (x : ℕ) : ℝ := Real.sqrt x
def six := 6
def five := 5

-- The statement of the problem as a Lean theorem
theorem simplify_sqrt180 : sqrt 180 = six * sqrt five := by 
  sorry

end simplify_sqrt180_l45_45379


namespace isosceles_triangle_perimeter_l45_45832

theorem isosceles_triangle_perimeter (a b c : ℝ) 
  (h1 : a = 4 ∨ b = 4 ∨ c = 4) 
  (h2 : a = 8 ∨ b = 8 ∨ c = 8) 
  (isosceles : a = b ∨ b = c ∨ a = c) : 
  a + b + c = 20 :=
by
  sorry

end isosceles_triangle_perimeter_l45_45832


namespace binom_coeff_mult_l45_45673

theorem binom_coeff_mult :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_coeff_mult_l45_45673


namespace correct_option_is_B_l45_45633

-- Define the Pythagorean theorem condition for right-angled triangles
def is_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Conditions given in the problem
def option_A : Prop := ¬is_right_angled_triangle 1 2 2
def option_B : Prop := is_right_angled_triangle 1 (Real.sqrt 3) 2
def option_C : Prop := ¬is_right_angled_triangle 4 5 6
def option_D : Prop := ¬is_right_angled_triangle 1 1 (Real.sqrt 3)

-- The formal proof problem statement
theorem correct_option_is_B : option_A ∧ option_B ∧ option_C ∧ option_D :=
by
  sorry

end correct_option_is_B_l45_45633


namespace Dawn_hourly_earnings_l45_45049

theorem Dawn_hourly_earnings :
  let t_per_painting := 2 
  let num_paintings := 12
  let total_earnings := 3600
  let total_time := t_per_painting * num_paintings
  let hourly_wage := total_earnings / total_time
  hourly_wage = 150 := by
  sorry

end Dawn_hourly_earnings_l45_45049


namespace find_a_l45_45339

-- Define the quadratic equation with the root condition
def quadratic_with_root_zero (a : ℝ) : Prop :=
  (a - 1) * 0^2 + 0 + a - 2 = 0

-- State the theorem to be proved
theorem find_a (a : ℝ) (h : quadratic_with_root_zero a) : a = 2 :=
by
  -- Statement placeholder, proof omitted
  sorry

end find_a_l45_45339


namespace roots_nonpositive_if_ac_le_zero_l45_45784

theorem roots_nonpositive_if_ac_le_zero (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : a * c ≤ 0) :
  ¬ (∀ x : ℝ, x^2 - (b/a)*x + (c/a) = 0 → x > 0) :=
sorry

end roots_nonpositive_if_ac_le_zero_l45_45784


namespace cos_double_angle_sum_l45_45481

theorem cos_double_angle_sum
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 := by
  sorry

end cos_double_angle_sum_l45_45481


namespace narrow_black_stripes_l45_45564

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end narrow_black_stripes_l45_45564


namespace factorize_expr_l45_45000

theorem factorize_expr (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) :=
  sorry

end factorize_expr_l45_45000


namespace spencer_total_distance_l45_45054

def distances : ℝ := 0.3 + 0.1 + 0.4

theorem spencer_total_distance :
  distances = 0.8 :=
sorry

end spencer_total_distance_l45_45054


namespace algebraic_expression_value_l45_45977

theorem algebraic_expression_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 - 7 = -6 := by
  sorry

end algebraic_expression_value_l45_45977


namespace largest_sum_l45_45275

theorem largest_sum :
  max (max (max (max (1/4 + 1/9) (1/4 + 1/10)) (1/4 + 1/11)) (1/4 + 1/12)) (1/4 + 1/13) = 13/36 := 
sorry

end largest_sum_l45_45275


namespace solve_by_completing_square_l45_45421

theorem solve_by_completing_square (x: ℝ) (h: x^2 + 4 * x - 3 = 0) : (x + 2)^2 = 7 := 
by 
  sorry

end solve_by_completing_square_l45_45421


namespace shaded_area_l45_45451

theorem shaded_area (whole_squares partial_squares : ℕ) (area_whole area_partial : ℝ)
  (h1 : whole_squares = 5)
  (h2 : partial_squares = 6)
  (h3 : area_whole = 1)
  (h4 : area_partial = 0.5) :
  (whole_squares * area_whole + partial_squares * area_partial) = 8 :=
by
  sorry

end shaded_area_l45_45451


namespace total_children_in_school_l45_45064

theorem total_children_in_school (B : ℕ) (C : ℕ) 
  (h1 : B = 2 * C)
  (h2 : B = 4 * (C - 350)) :
  C = 700 :=
by sorry

end total_children_in_school_l45_45064


namespace sum_due_is_42_l45_45394

-- Define the conditions
def BD : ℝ := 42
def TD : ℝ := 36

-- Statement to prove
theorem sum_due_is_42 (H1 : BD = 42) (H2 : TD = 36) : ∃ (FV : ℝ), FV = 42 := by
  -- Proof Placeholder
  sorry

end sum_due_is_42_l45_45394


namespace students_like_basketball_or_cricket_or_both_l45_45905

theorem students_like_basketball_or_cricket_or_both {A B C : ℕ} (hA : A = 12) (hB : B = 8) (hC : C = 3) :
    A + B - C = 17 :=
by
  sorry

end students_like_basketball_or_cricket_or_both_l45_45905


namespace find_angle_l45_45296

theorem find_angle (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by 
  sorry

end find_angle_l45_45296


namespace commercial_break_duration_l45_45466

theorem commercial_break_duration (n1 n2 t1 t2 : ℕ) (h1 : n1 = 3) (h2: t1 = 5) (h3 : n2 = 11) (h4 : t2 = 2) : 
  n1 * t1 + n2 * t2 = 37 := 
by 
  sorry

end commercial_break_duration_l45_45466


namespace factorize_x_l45_45004

theorem factorize_x^3_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
sorry

end factorize_x_l45_45004


namespace maxwell_distance_when_meeting_l45_45751

/-- Maxwell and Brad are moving towards each other from their respective homes.
Maxwell's walking speed is 3 km/h, Brad's running speed is 5 km/h,
and the distance between their homes is 40 kilometers.
Prove that the distance traveled by Maxwell when they meet is 15 kilometers. -/
theorem maxwell_distance_when_meeting
  (distance_between_homes : ℝ)
  (maxwell_speed : ℝ)
  (brad_speed : ℝ)
  (meeting_distance : ℝ) :
  distance_between_homes = 40 ∧ maxwell_speed = 3 ∧ brad_speed = 5 → meeting_distance = 15 :=
begin
  intros h,
  rcases h with ⟨d_eq_40, m_speed_eq_3, b_speed_eq_5⟩,
  sorry
end

end maxwell_distance_when_meeting_l45_45751


namespace opposite_of_negative_2023_l45_45401

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l45_45401


namespace B_pow_150_l45_45859

noncomputable def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem B_pow_150 : B ^ 150 = 1 :=
by
  sorry

end B_pow_150_l45_45859


namespace notebook_ratio_l45_45960

theorem notebook_ratio (C N : ℕ) (h1 : ∀ k, N = k / C)
  (h2 : ∃ k, N = k / (C / 2) ∧ 16 = k / (C / 2))
  (h3 : C * N = 512) : (N : ℚ) / C = 1 / 8 := 
by
  sorry

end notebook_ratio_l45_45960


namespace number_of_narrow_black_stripes_l45_45583

-- Define the variables
variables {w n b : ℕ}

-- The conditions from the problem
def condition1 := w + n = b + 1
def condition2 := b = w + 7

-- The Lean statement to prove
theorem number_of_narrow_black_stripes (h1 : condition1) (h2 : condition2) : n = 8 :=
by {
  -- We import the conditions as hypotheses
  sorry
}

end number_of_narrow_black_stripes_l45_45583


namespace sin_870_correct_l45_45808

noncomputable def sin_870_eq_half : Prop :=
  sin (870 : ℝ) = 1 / 2

theorem sin_870_correct : sin_870_eq_half :=
by
  sorry

end sin_870_correct_l45_45808


namespace trapezoid_median_l45_45810

theorem trapezoid_median 
  (h : ℝ)
  (triangle_base : ℝ := 24)
  (trapezoid_base1 : ℝ := 15)
  (trapezoid_base2 : ℝ := 33)
  (triangle_area_eq_trapezoid_area : (1 / 2) * triangle_base * h = ((trapezoid_base1 + trapezoid_base2) / 2) * h)
  : (trapezoid_base1 + trapezoid_base2) / 2 = 24 :=
by
  sorry

end trapezoid_median_l45_45810


namespace number_of_black_bears_l45_45961

-- Definitions of conditions
def brown_bears := 15
def white_bears := 24
def total_bears := 66

-- The proof statement
theorem number_of_black_bears : (total_bears - (brown_bears + white_bears) = 27) := by
  sorry

end number_of_black_bears_l45_45961


namespace single_elimination_games_l45_45177

theorem single_elimination_games (n : Nat) (h : n = 256) : n - 1 = 255 := by
  sorry

end single_elimination_games_l45_45177


namespace question_1_question_2_question_3_l45_45973

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 2 * b

-- Question 1
theorem question_1 (a b : ℝ) (h : a = b) (ha : a > 0) :
  ∀ x : ℝ, (f a b x < 0) ↔ (-2 < x ∧ x < 1) :=
sorry

-- Question 2
theorem question_2 (b : ℝ) :
  (∀ x : ℝ, x < 2 → (f 1 b x ≥ 1)) → (b ≤ 2 * Real.sqrt 3 - 4) :=
sorry

-- Question 3
theorem question_3 (a b : ℝ) (h1 : |f a b (-1)| ≤ 1) (h2 : |f a b 1| ≤ 3) :
  (5 / 3 ≤ |a| + |b + 2| ∧ |a| + |b + 2| ≤ 9) :=
sorry

end question_1_question_2_question_3_l45_45973


namespace smallest_repeating_block_digits_l45_45331

theorem smallest_repeating_block_digits (n : ℕ) (d : ℕ) (hd_pos : d > 0) (hd_coprime : Nat.gcd n d = 1)
  (h_fraction : (n : ℚ) / d = 8 / 11) : n = 2 :=
by
  -- proof will go here
  sorry

end smallest_repeating_block_digits_l45_45331


namespace largest_prime_factor_of_3136_l45_45101

theorem largest_prime_factor_of_3136 : ∃ p, p.prime ∧ p ∣ 3136 ∧ ∀ q, q.prime ∧ q ∣ 3136 → q ≤ p :=
sorry

end largest_prime_factor_of_3136_l45_45101


namespace narrow_black_stripes_are_8_l45_45578

-- Define variables: w for wide black stripes, n for narrow black stripes, b for white stripes
variables (w n b : ℕ)

-- Given conditions
axiom cond1 : b = w + 7
axiom cond2 : w + n = b + 1

-- Theorem statement to prove that the number of narrow black stripes is 8
theorem narrow_black_stripes_are_8 : n = 8 :=
by sorry

end narrow_black_stripes_are_8_l45_45578


namespace laborer_monthly_income_l45_45907

theorem laborer_monthly_income :
  (∃ (I D : ℤ),
    6 * I + D = 540 ∧
    4 * I - D = 270) →
  (∃ I : ℤ,
    I = 81) :=
by
  sorry

end laborer_monthly_income_l45_45907


namespace age_ratio_l45_45267

theorem age_ratio (S M : ℕ) (h₁ : M = S + 35) (h₂ : S = 33) : 
  (M + 2) / (S + 2) = 2 :=
by
  -- proof goes here
  sorry

end age_ratio_l45_45267


namespace domain_M_complement_domain_M_l45_45836

noncomputable def f (x : ℝ) : ℝ :=
  1 / Real.sqrt (1 - x)

noncomputable def g (x : ℝ) : ℝ :=
  Real.log (1 + x)

def M : Set ℝ :=
  {x | 1 - x > 0}

def N : Set ℝ :=
  {x | 1 + x > 0}

def complement_M : Set ℝ :=
  {x | 1 - x ≤ 0}

theorem domain_M :
  M = {x | x < 1} := by
  sorry

theorem complement_domain_M :
  complement_M = {x | x ≥ 1} := by
  sorry

end domain_M_complement_domain_M_l45_45836


namespace john_twice_james_l45_45992

def john_age : ℕ := 39
def years_ago : ℕ := 3
def years_future : ℕ := 6
def age_difference : ℕ := 4

theorem john_twice_james {J : ℕ} (h : 39 - years_ago = 2 * (J + years_future)) : 
  (J + age_difference = 16) :=
by
  sorry  -- Proof steps here

end john_twice_james_l45_45992


namespace cosine_function_range_l45_45461

theorem cosine_function_range : 
  (∀ x ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), -1/2 ≤ Real.cos x ∧ Real.cos x ≤ 1) ∧
  (∃ a ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), Real.cos a = 1) ∧
  (∃ b ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), Real.cos b = -1/2) :=
by
  sorry

end cosine_function_range_l45_45461


namespace kylie_total_beads_used_l45_45362

noncomputable def beads_monday_necklaces : ℕ := 10 * 20
noncomputable def beads_tuesday_necklaces : ℕ := 2 * 20
noncomputable def beads_wednesday_bracelets : ℕ := 5 * 10
noncomputable def beads_thursday_earrings : ℕ := 3 * 5
noncomputable def beads_friday_anklets : ℕ := 4 * 8
noncomputable def beads_friday_rings : ℕ := 6 * 7

noncomputable def total_beads_used : ℕ :=
  beads_monday_necklaces +
  beads_tuesday_necklaces +
  beads_wednesday_bracelets +
  beads_thursday_earrings +
  beads_friday_anklets +
  beads_friday_rings

theorem kylie_total_beads_used : total_beads_used = 379 := by
  sorry

end kylie_total_beads_used_l45_45362


namespace cubic_roots_sum_of_cubes_l45_45862

theorem cubic_roots_sum_of_cubes :
  ∀ (a b c : ℝ), 
  (∀ x : ℝ, 9 * x^3 + 14 * x^2 + 2047 * x + 3024 = 0 → (x = a ∨ x = b ∨ x = c)) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = -58198 / 729 :=
by
  intros a b c roota_eqn
  sorry

end cubic_roots_sum_of_cubes_l45_45862


namespace Question_D_condition_l45_45867

theorem Question_D_condition (P Q : Prop) (h : P → Q) : ¬ Q → ¬ P :=
by sorry

end Question_D_condition_l45_45867


namespace sum_series_eq_final_sum_l45_45133

-- Given N is greater than or equal to 2
variable (N : ℕ) (h : 2 ≤ N)

-- Define the series term
def seriesTerm (n : ℕ) : ℝ :=
  (6 * n^3 - 2 * n^2 - 2 * n + 2) / (n^6 - n^5 + n^4 - n^3 + n^2 - n)

-- Statement of the theorem
theorem sum_series_eq_final_sum :
  ∑ n in Finset.range (N - 1) + 1, seriesTerm n = FinalSum := 
begin
  -- Placeholder for the actual proof
  sorry
end

end sum_series_eq_final_sum_l45_45133


namespace ratio_of_people_on_buses_l45_45222

theorem ratio_of_people_on_buses (P_2 P_3 P_4 : ℕ) 
  (h1 : P_1 = 12) 
  (h2 : P_3 = P_2 - 6) 
  (h3 : P_4 = P_1 + 9) 
  (h4 : P_1 + P_2 + P_3 + P_4 = 75) : 
  P_2 / P_1 = 2 := 
by
  sorry

end ratio_of_people_on_buses_l45_45222


namespace geometric_sum_proof_l45_45721

theorem geometric_sum_proof (S : ℕ → ℝ) (a : ℕ → ℝ) (r : ℝ) (n : ℕ)
    (hS3 : S 3 = 8) (hS6 : S 6 = 7)
    (Sn_def : ∀ n, S n = a 0 * (1 - r ^ n) / (1 - r)) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = -7 / 8 :=
by
  sorry

end geometric_sum_proof_l45_45721


namespace find_angle_l45_45499

-- Definitions based on conditions
def is_complement (x : ℝ) : ℝ := 90 - x
def is_supplement (x : ℝ) : ℝ := 180 - x

-- Main statement
theorem find_angle (x : ℝ) (h : is_supplement x = 15 + 4 * is_complement x) : x = 65 :=
by
  sorry

end find_angle_l45_45499


namespace Owen_final_turtle_count_l45_45072

variable (Owen_turtles : ℕ) (Johanna_turtles : ℕ)

def final_turtles (Owen_turtles Johanna_turtles : ℕ) : ℕ :=
  let initial_Owen_turtles := Owen_turtles
  let initial_Johanna_turtles := Owen_turtles - 5
  let Owen_after_month := initial_Owen_turtles * 2
  let Johanna_after_losing_half := initial_Johanna_turtles / 2
  let Owen_after_donation := Owen_after_month + Johanna_after_losing_half
  Owen_after_donation

theorem Owen_final_turtle_count : final_turtles 21 (21 - 5) = 50 :=
by
  sorry

end Owen_final_turtle_count_l45_45072


namespace initial_distance_l45_45186

/-- Suppose Jack walks at a speed of 3 feet per second toward Christina,
    Christina walks at a speed of 3 feet per second toward Jack, and their dog Lindy
    runs at a speed of 10 feet per second back and forth between Jack and Christina.
    Given that Lindy travels a total of 400 feet when they meet, prove that the initial
    distance between Jack and Christina is 240 feet. -/
theorem initial_distance (initial_distance_jack_christina : ℝ)
  (jack_speed : ℝ := 3)
  (christina_speed : ℝ := 3)
  (lindy_speed : ℝ := 10)
  (lindy_total_distance : ℝ := 400):
  initial_distance_jack_christina = 240 :=
sorry

end initial_distance_l45_45186


namespace problem_statement_l45_45887

def op (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem problem_statement : ((op 7 4) - 12) * 5 = 105 := by
  sorry

end problem_statement_l45_45887


namespace area_difference_l45_45044

theorem area_difference (T_area : ℝ) (omega_area : ℝ) (H1 : T_area = (25 * Real.sqrt 3) / 4) 
  (H2 : omega_area = 4 * Real.pi) (H3 : 3 * (X - Y) = T_area - omega_area) :
  X - Y = (25 * Real.sqrt 3) / 12 - (4 * Real.pi) / 3 :=
by 
  sorry

end area_difference_l45_45044


namespace chocolate_bar_cost_l45_45139

def total_bars := 11
def bars_left := 7
def bars_sold := total_bars - bars_left
def total_money := 16
def cost := total_money / bars_sold

theorem chocolate_bar_cost : cost = 4 :=
by
  sorry

end chocolate_bar_cost_l45_45139


namespace bonus_percentage_is_correct_l45_45408

theorem bonus_percentage_is_correct (kills total_points enemies_points bonus_threshold bonus_percentage : ℕ) 
  (h1 : enemies_points = 10) 
  (h2 : kills = 150) 
  (h3 : total_points = 2250) 
  (h4 : bonus_threshold = 100) 
  (h5 : kills >= bonus_threshold) 
  (h6 : bonus_percentage = (total_points - kills * enemies_points) * 100 / (kills * enemies_points)) : 
  bonus_percentage = 50 := 
by
  sorry

end bonus_percentage_is_correct_l45_45408


namespace angle_BAC_measure_l45_45047

variable (A B C X Y : Type)
variables (angle_ABC angle_BAC : ℝ)
variables (len_AX len_XY len_YB len_BC : ℝ)

theorem angle_BAC_measure 
  (h1 : AX = XY) 
  (h2 : XY = YB) 
  (h3 : XY = 2 * AX) 
  (h4 : angle_ABC = 150) :
  angle_BAC = 26.25 :=
by
  -- The proof would be required here.
  -- Following the statement as per instructions.
  sorry

end angle_BAC_measure_l45_45047


namespace intersection_A_B_l45_45189

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | -2 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 1} :=
sorry

end intersection_A_B_l45_45189


namespace set_B_can_form_right_angled_triangle_l45_45628

-- Definition and condition from the problem
def isRightAngledTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- The actual proof problem statement
theorem set_B_can_form_right_angled_triangle : isRightAngledTriangle 1 (Real.sqrt 3) 2 :=
sorry

end set_B_can_form_right_angled_triangle_l45_45628


namespace min_value_exp_l45_45964

theorem min_value_exp (x y : ℝ) (h : x + 2 * y = 4) : ∃ z : ℝ, (2^x + 4^y = z) ∧ (∀ (a b : ℝ), a + 2 * b = 4 → 2^a + 4^b ≥ z) :=
sorry

end min_value_exp_l45_45964


namespace percentage_increase_l45_45805

theorem percentage_increase 
  (distance : ℝ) (time_q : ℝ) (time_y : ℝ) 
  (speed_q : ℝ) (speed_y : ℝ) 
  (percentage_increase : ℝ) 
  (h_distance : distance = 80)
  (h_time_q : time_q = 2)
  (h_time_y : time_y = 1.3333333333333333)
  (h_speed_q : speed_q = distance / time_q)
  (h_speed_y : speed_y = distance / time_y)
  (h_faster : speed_y > speed_q)
  : percentage_increase = ((speed_y - speed_q) / speed_q) * 100 :=
by
  sorry

end percentage_increase_l45_45805


namespace boxes_per_week_l45_45617

-- Define the given conditions
def cost_per_box : ℝ := 3.00
def weeks_in_year : ℝ := 52
def total_spent_per_year : ℝ := 312

-- The question we want to prove:
theorem boxes_per_week:
  (total_spent_per_year = cost_per_box * weeks_in_year * (total_spent_per_year / (weeks_in_year * cost_per_box))) → 
  (total_spent_per_year / (weeks_in_year * cost_per_box)) = 2 := sorry

end boxes_per_week_l45_45617


namespace cos_2alpha_2beta_l45_45483

variables (α β : ℝ)

open Real

theorem cos_2alpha_2beta (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) : cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_2alpha_2beta_l45_45483


namespace total_dog_legs_l45_45099

theorem total_dog_legs (total_animals cats dogs: ℕ) (h1: total_animals = 300) 
  (h2: cats = 2 / 3 * total_animals) 
  (h3: dogs = 1 / 3 * total_animals): (dogs * 4) = 400 :=
by
  sorry

end total_dog_legs_l45_45099


namespace total_tubes_in_consignment_l45_45919

theorem total_tubes_in_consignment (N : ℕ) 
  (h : (5 / (N : ℝ)) * (4 / (N - 1 : ℝ)) = 0.05263157894736842) : 
  N = 20 := 
sorry

end total_tubes_in_consignment_l45_45919


namespace multiples_of_6_or_8_but_not_both_l45_45511

/-- The number of positive integers less than 151 that are multiples of either 6 or 8 but not both is 31. -/
theorem multiples_of_6_or_8_but_not_both (n : ℕ) :
  (multiples_of_6 : Set ℕ) = {k | k < 151 ∧ k % 6 = 0}
  ∧ (multiples_of_8 : Set ℕ) = {k | k < 151 ∧ k % 8 = 0}
  ∧ (multiples_of_24 : Set ℕ) = {k | k < 151 ∧ k % 24 = 0}
  ∧ multiples_of_6_or_8 := {k | k ∈ multiples_of_6 ∨ k ∈ multiples_of_8}
  ∧ multiples_of_6_and_8 := {k | k ∈ multiples_of_6 ∧ k ∈ multiples_of_8}
  ∧ (card (multiples_of_6_or_8 \ multiples_of_6_and_8)) = 31 := sorry

end multiples_of_6_or_8_but_not_both_l45_45511


namespace largest_angle_in_triangle_l45_45184

theorem largest_angle_in_triangle (a b c : ℝ) (h1 : a + 3 * b + 3 * c = a ^ 2) (h2 : a + 3 * b - 3 * c = -4) 
  (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) (h6 : a + b > c) (h7 : a + c > b) (h8 : b + c > a) : 
  ∃ C : ℝ, C = 120 ∧ (by exact sorry) := sorry

end largest_angle_in_triangle_l45_45184


namespace consecutive_integers_divisible_by_12_l45_45170

theorem consecutive_integers_divisible_by_12 (a b c d : ℤ) 
  (h1 : b = a + 1) (h2 : c = b + 1) (h3 : d = c + 1) : 
  12 ∣ (a * b + a * c + a * d + b * c + b * d + c * d + 1) := 
sorry

end consecutive_integers_divisible_by_12_l45_45170


namespace binomial_product_l45_45688

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := by
  sorry

end binomial_product_l45_45688


namespace find_real_solutions_l45_45819

def equation (x : ℝ) : Prop := x^4 + (3 - x)^4 = 82

theorem find_real_solutions : 
  {x : ℝ // equation x} = {x | x ≈ 3.22 ∨ x ≈ -0.22} :=
by 
  sorry

end find_real_solutions_l45_45819


namespace exists_polyhedron_with_no_three_same_sided_faces_l45_45672

structure Face :=
  (sides : ℕ)

structure Polyhedron :=
  (faces : List Face)
  (closed : Bool)

-- Definition of specific faces
def triangular_face : Face := ⟨3⟩
def quadrilateral_face : Face := ⟨4⟩
def pentagonal_face : Face := ⟨5⟩

-- Definition of the polyhedron in terms of the conditions
def polyhedron_example : Polyhedron := 
  ⟨[triangular_face, triangular_face, quadrilateral_face, quadrilateral_face, pentagonal_face, pentagonal_face], true⟩

theorem exists_polyhedron_with_no_three_same_sided_faces : 
  ∃ (p : Polyhedron), p = polyhedron_example ∧ p.closed ∧ 
    (∀ n, (p.faces.filter (λ f, f.sides = n)).length < 3) :=
by
  sorry

end exists_polyhedron_with_no_three_same_sided_faces_l45_45672


namespace sum_of_squares_of_consecutive_integers_l45_45892

-- The sum of the squares of three consecutive positive integers equals 770.
-- We aim to prove that the largest integer among them is 17.
theorem sum_of_squares_of_consecutive_integers (n : ℕ) (h_pos : n > 0) 
    (h_sum : (n-1)^2 + n^2 + (n+1)^2 = 770) : n + 1 = 17 :=
sorry

end sum_of_squares_of_consecutive_integers_l45_45892


namespace nancy_total_savings_l45_45753

noncomputable def total_savings : ℝ :=
  let cost_this_month := 9 * 5
  let cost_last_month := 8 * 4
  let cost_next_month := 7 * 6
  let discount_this_month := 0.20 * cost_this_month
  let discount_last_month := 0.20 * cost_last_month
  let discount_next_month := 0.20 * cost_next_month
  discount_this_month + discount_last_month + discount_next_month

theorem nancy_total_savings : total_savings = 23.80 :=
by
  sorry

end nancy_total_savings_l45_45753


namespace find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5_find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5_also_924_l45_45473

theorem find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5 :
  ∃ (n : ℕ), (n % 21 = 0) ∧ (30 < real.sqrt n) ∧ (real.sqrt n < 30.5) :=
begin
  use 903,
  split,
  {
    -- proof that 21 divides 903
    rw nat.mod_eq_zero,
    exact dvd.refl _,
  },
  {
    split,
    {
      -- proof that 30 < sqrt(903)
      norm_num, 
      linarith,
    },
    {
      -- proof that sqrt(903) < 30.5
      norm_num,
      linarith,
    }
  }
end

theorem find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5_also_924 :
  ∃ (n : ℕ), (n % 21 = 0) ∧ (30 < real.sqrt n) ∧ (real.sqrt n < 30.5) :=
begin
  use 924,
  split,
  {
    -- proof that 21 divides 924
    rw nat.mod_eq_zero,
    exact dvd.refl _,
  },
  {
    split,
    {
      -- proof that 30 < sqrt(924)
      norm_num,
      linarith,
    },
    {
      -- proof that sqrt(924) < 30.5
      norm_num,
      linarith,
    }
  }
end

end find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5_find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5_also_924_l45_45473


namespace opposite_of_negative_2023_l45_45402

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l45_45402


namespace best_model_l45_45520

theorem best_model (R1 R2 R3 R4 : ℝ) :
  R1 = 0.78 → R2 = 0.85 → R3 = 0.61 → R4 = 0.31 →
  (R2 = max R1 (max R2 (max R3 R4))) :=
by
  intros hR1 hR2 hR3 hR4
  sorry

end best_model_l45_45520


namespace correct_option_B_l45_45621

variable {a b x y : ℤ}

def option_A (a : ℤ) : Prop := -a - a = 0
def option_B (x y : ℤ) : Prop := -(x + y) = -x - y
def option_C (b a : ℤ) : Prop := 3 * (b - 2 * a) = 3 * b - 2 * a
def option_D (a : ℤ) : Prop := 8 * a^4 - 6 * a^2 = 2 * a^2

theorem correct_option_B (x y : ℤ) : option_B x y := by
  -- The proof would go here
  sorry

end correct_option_B_l45_45621


namespace complement_M_l45_45324

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - 4 ≤ 0}
def C (s : Set ℝ) : Set ℝ := sᶜ -- complement of a set

theorem complement_M :
  C M = {x : ℝ | x < -2 ∨ x > 2} :=
by
  sorry

end complement_M_l45_45324


namespace sum_equals_target_l45_45490

open BigOperators

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_initial_condition : f 0 = 1

axiom f_functional_eq (x y : ℝ) : f (x * y + 1) = f x * f y - f y - x + 2

theorem sum_equals_target : (∑ i in finset.range 2023, 1 / (f i * f (i + 1))) = 2023 / 4050 :=
by
  sorry

end sum_equals_target_l45_45490


namespace length_of_train_l45_45251

theorem length_of_train (speed_kmh : ℕ) (time_seconds : ℕ) (h_speed : speed_kmh = 60) (h_time : time_seconds = 36) :
  let time_hours := (time_seconds : ℚ) / 3600
  let distance_km := (speed_kmh : ℚ) * time_hours
  let distance_m := distance_km * 1000
  distance_m = 600 :=
by
  sorry

end length_of_train_l45_45251


namespace simplify_complex_number_l45_45079

theorem simplify_complex_number (i : ℂ) (h : i^2 = -1) : i * (1 - i)^2 = 2 := by
  sorry

end simplify_complex_number_l45_45079


namespace science_books_have_9_copies_l45_45590

theorem science_books_have_9_copies :
  ∃ (A B C D : ℕ), A + B + C + D = 35 ∧ A + B = 17 ∧ B + C = 16 ∧ A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ B = 9 :=
by
  sorry

end science_books_have_9_copies_l45_45590


namespace units_digit_of_m_squared_plus_3_to_the_m_l45_45363

theorem units_digit_of_m_squared_plus_3_to_the_m (m : ℕ) (h : m = 2010^2 + 2^2010) : 
  (m^2 + 3^m) % 10 = 7 :=
by {
  sorry -- proof goes here
}

end units_digit_of_m_squared_plus_3_to_the_m_l45_45363


namespace binom_mult_l45_45687

open Nat

theorem binom_mult : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binom_mult_l45_45687


namespace rectangle_longer_side_l45_45441

theorem rectangle_longer_side
  (r : ℝ)
  (A_circle : ℝ)
  (A_rectangle : ℝ)
  (shorter_side : ℝ)
  (longer_side : ℝ) :
  r = 5 →
  A_circle = 25 * Real.pi →
  A_rectangle = 3 * A_circle →
  shorter_side = 2 * r →
  longer_side = A_rectangle / shorter_side →
  longer_side = 7.5 * Real.pi :=
by
  intros
  sorry

end rectangle_longer_side_l45_45441


namespace x_minus_y_options_l45_45234

theorem x_minus_y_options (x y : ℕ) (h : 3 * x^2 + x = 4 * y^2 + y) :
  (x - y ≠ 2013) ∧ (x - y ≠ 2014) ∧ (x - y ≠ 2015) ∧ (x - y ≠ 2016) := 
sorry

end x_minus_y_options_l45_45234


namespace certain_number_minus_two_l45_45913

theorem certain_number_minus_two (x : ℝ) (h : 6 - x = 2) : x - 2 = 2 := 
sorry

end certain_number_minus_two_l45_45913


namespace solve_inequality_system_l45_45391

theorem solve_inequality_system (x : ℝ) :
  (8 * x - 3 ≤ 13) ∧ ((x - 1) / 3 - 2 < x - 1) → -2 < x ∧ x ≤ 2 :=
by
  intros h
  sorry

end solve_inequality_system_l45_45391


namespace teacher_arrangements_l45_45662

theorem teacher_arrangements (T : Fin 30 → ℕ) (h1 : T 1 < T 2 ∧ T 2 < T 3 ∧ T 3 < T 4 ∧ T 4 < T 5)
  (h2 : ∀ i : Fin 4, T (i + 1) ≥ T i + 3)
  (h3 : 1 ≤ T 1)
  (h4 : T 5 ≤ 26) :
  ∃ n : ℕ, n = 26334 := by
  sorry

end teacher_arrangements_l45_45662


namespace portfolio_value_after_two_years_l45_45053

def initial_portfolio := 80

def first_year_growth_rate := 0.15
def add_after_6_months := 28
def withdraw_after_9_months := 10

def second_year_growth_first_6_months := 0.10
def second_year_decline_last_6_months := 0.04

def final_portfolio_value := 115.59

theorem portfolio_value_after_two_years 
  (initial_portfolio : ℝ)
  (first_year_growth_rate : ℝ)
  (add_after_6_months : ℕ)
  (withdraw_after_9_months : ℕ)
  (second_year_growth_first_6_months : ℝ)
  (second_year_decline_last_6_months : ℝ)
  (final_portfolio_value : ℝ) :
  (initial_portfolio = 80) →
  (first_year_growth_rate = 0.15) →
  (add_after_6_months = 28) →
  (withdraw_after_9_months = 10) →
  (second_year_growth_first_6_months = 0.10) →
  (second_year_decline_last_6_months = 0.04) →
  (final_portfolio_value = 115.59) :=
by
  sorry

end portfolio_value_after_two_years_l45_45053


namespace largest_integer_same_cost_l45_45237

def cost_base_10 (n : ℕ) : ℕ :=
  (n.digits 10).sum

def cost_base_2 (n : ℕ) : ℕ :=
  (n.digits 2).sum

theorem largest_integer_same_cost : ∃ n < 1000, 
  cost_base_10 n = cost_base_2 n ∧
  ∀ m < 1000, cost_base_10 m = cost_base_2 m → n ≥ m :=
sorry

end largest_integer_same_cost_l45_45237


namespace probability_of_binomial_distribution_l45_45838

open ProbabilityTheory

variables (ξ : ℕ)

def binomial_distribution (n : ℕ) (p : ℚ) : Distribution ℕ :=
  Distribution.binomial n p

theorem probability_of_binomial_distribution :
  (ξ ~ (binomial_distribution 3 (1 / 3))) → P(ξ = 1) = 4 / 9 :=
by
  sorry

end probability_of_binomial_distribution_l45_45838


namespace exists_four_integers_mod_5050_l45_45829

theorem exists_four_integers_mod_5050 (S : Finset ℕ) (hS_card : S.card = 101) (hS_bound : ∀ x ∈ S, x < 5050) : 
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (a + b - c - d) % 5050 = 0 :=
sorry

end exists_four_integers_mod_5050_l45_45829


namespace area_of_triangle_XPQ_l45_45854
open Real

/-- Given a triangle XYZ with area 15 square units and points P, Q, R on sides XY, YZ, and ZX respectively,
where XP = 3, PY = 6, and triangles XPQ and quadrilateral PYRQ have equal areas, 
prove that the area of triangle XPQ is 5/3 square units. -/
theorem area_of_triangle_XPQ 
  (Area_XYZ : ℝ) (h1 : Area_XYZ = 15)
  (XP PY : ℝ) (h2 : XP = 3) (h3 : PY = 6)
  (h4 : ∃ (Area_XPQ : ℝ) (Area_PYRQ : ℝ), Area_XPQ = Area_PYRQ) :
  ∃ (Area_XPQ : ℝ), Area_XPQ = 5/3 :=
sorry

end area_of_triangle_XPQ_l45_45854


namespace teams_dig_tunnel_in_10_days_l45_45282

theorem teams_dig_tunnel_in_10_days (hA : ℝ) (hB : ℝ) (work_A : hA = 15) (work_B : hB = 30) : 
  (1 / (1 / hA + 1 / hB)) = 10 := 
by
  sorry

end teams_dig_tunnel_in_10_days_l45_45282


namespace man_l45_45649

-- Constants and conditions
def V_down : ℝ := 18  -- downstream speed in km/hr
def V_c : ℝ := 3.4    -- speed of the current in km/hr

-- Main statement to prove
theorem man's_speed_against_the_current : (V_down - V_c - V_c) = 11.2 := by
  sorry

end man_l45_45649


namespace magnet_cost_times_sticker_l45_45271

theorem magnet_cost_times_sticker
  (M S A : ℝ)
  (hM : M = 3)
  (hA : A = 6)
  (hMagnetCost : M = (1/4) * 2 * A) :
  M = 4 * S :=
by
  -- Placeholder, the actual proof would go here
  sorry

end magnet_cost_times_sticker_l45_45271


namespace inequality_proof_l45_45968

theorem inequality_proof (a b c d : ℕ) (h₀: a + c ≤ 1982) (h₁: (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)) (h₂: (a:ℚ)/b + (c:ℚ)/d < 1) :
  1 - (a:ℚ)/b - (c:ℚ)/d > 1 / (1983 ^ 3) :=
sorry

end inequality_proof_l45_45968


namespace production_rate_problem_l45_45041

theorem production_rate_problem :
  ∀ (G T : ℕ), 
  (∀ w t, w * 3 * t = 450 * t / 150) ∧
  (∀ w t, w * 2 * t = 300 * t / 150) ∧
  (∀ w t, w * 2 * t = 360 * t / 90) ∧
  (∀ w t, w * (5/2) * t = 450 * t / 90) ∧
  (75 * 2 * 4 = 300) →
  (75 * 2 * 4 = 600) := sorry

end production_rate_problem_l45_45041


namespace exists_integer_a_l45_45536

theorem exists_integer_a (p : ℕ) (hp : p ≥ 5) [Fact (Nat.Prime p)] : 
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧ (¬ p^2 ∣ a^(p-1) - 1) ∧ (¬ p^2 ∣ (a+1)^(p-1) - 1) :=
by
  sorry

end exists_integer_a_l45_45536


namespace katie_total_marbles_l45_45360

theorem katie_total_marbles :
  ∀ (pink marbles orange marbles purple marbles total : ℕ),
    pink = 13 →
    orange = pink - 9 →
    purple = 4 * orange →
    total = pink + orange + purple →
    total = 33 :=
by
  intros pink marbles orange marbles purple marbles total
  assume h_pink h_orange h_purple h_total
  sorry

end katie_total_marbles_l45_45360


namespace sum_of_vertical_asymptotes_l45_45604

noncomputable def sum_of_roots (a b c : ℝ) (h_discriminant : b^2 - 4*a*c ≠ 0) : ℝ :=
-(b/a)

theorem sum_of_vertical_asymptotes :
  let f := (6 * (x^2) - 8) / (4 * (x^2) + 7*x + 3)
  ∃ c d, c ≠ d ∧ (4*c^2 + 7*c + 3 = 0) ∧ (4*d^2 + 7*d + 3 = 0)
  ∧ c + d = -7 / 4 :=
by
  sorry

end sum_of_vertical_asymptotes_l45_45604


namespace solve_tangent_problem_l45_45012

noncomputable def problem_statement : Prop :=
  ∃ (n : ℤ), (-90 < n ∧ n < 90) ∧ (Real.tan (n * Real.pi / 180) = Real.tan (255 * Real.pi / 180)) ∧ (n = 75)

-- This is the statement of the problem we are proving.
theorem solve_tangent_problem : problem_statement :=
by
  sorry

end solve_tangent_problem_l45_45012


namespace scientific_notation_of_935million_l45_45660

theorem scientific_notation_of_935million :
  935000000 = 9.35 * 10 ^ 8 :=
  sorry

end scientific_notation_of_935million_l45_45660


namespace dawns_earnings_per_hour_l45_45051

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

end dawns_earnings_per_hour_l45_45051


namespace smallest_side_is_10_l45_45884

noncomputable def smallest_side_of_triangle (x : ℝ) : ℝ :=
    let side1 := 10
    let side2 := 3 * x + 6
    let side3 := x + 5
    min side1 (min side2 side3)

theorem smallest_side_is_10 (x : ℝ) (h : 10 + (3 * x + 6) + (x + 5) = 60) : 
    smallest_side_of_triangle x = 10 :=
by
    sorry

end smallest_side_is_10_l45_45884


namespace ratio_of_canoes_to_kayaks_l45_45897

theorem ratio_of_canoes_to_kayaks 
    (canoe_cost kayak_cost total_revenue : ℕ) 
    (canoe_to_kayak_ratio extra_canoes : ℕ)
    (h1 : canoe_cost = 14)
    (h2 : kayak_cost = 15)
    (h3 : total_revenue = 288)
    (h4 : extra_canoes = 4)
    (h5 : canoe_to_kayak_ratio = 3) 
    (c k : ℕ)
    (h6 : c = k + extra_canoes)
    (h7 : c = canoe_to_kayak_ratio * k)
    (h8 : canoe_cost * c + kayak_cost * k = total_revenue) :
    c / k = 3 := 
sorry

end ratio_of_canoes_to_kayaks_l45_45897


namespace sum_of_sides_l45_45172

-- Definitions: Given conditions
def ratio (a b c : ℕ) : Prop := 
a * 5 = b * 3 ∧ b * 7 = c * 5

-- Given that the longest side is 21 cm and the ratio of the sides is 3:5:7
def similar_triangle (x y : ℕ) : Prop :=
ratio x y 21

-- Proof statement: The sum of the lengths of the other two sides is 24 cm
theorem sum_of_sides (x y : ℕ) (h : similar_triangle x y) : x + y = 24 :=
sorry

end sum_of_sides_l45_45172


namespace fathers_age_multiple_l45_45088

theorem fathers_age_multiple 
  (Johns_age : ℕ)
  (sum_of_ages : ℕ)
  (additional_years : ℕ)
  (m : ℕ)
  (h1 : Johns_age = 15)
  (h2 : sum_of_ages = 77)
  (h3 : additional_years = 32)
  (h4 : sum_of_ages = Johns_age + (Johns_age * m + additional_years)) :
  m = 2 := 
by 
  sorry

end fathers_age_multiple_l45_45088


namespace simplify_fraction_l45_45374

theorem simplify_fraction (x : ℝ) : (2 * x - 3) / 4 + (4 * x + 5) / 3 = (22 * x + 11) / 12 := by
  sorry

end simplify_fraction_l45_45374


namespace min_value_frac_sum_l45_45716

theorem min_value_frac_sum (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) :
  (∃ m, ∀ x y, m = 1 ∧ (
      (1 / (x + y)^2) + (1 / (x - y)^2) ≥ m)) :=
sorry

end min_value_frac_sum_l45_45716


namespace sin_585_eq_neg_sqrt2_div_2_l45_45698

theorem sin_585_eq_neg_sqrt2_div_2 : Real.sin (585 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_585_eq_neg_sqrt2_div_2_l45_45698


namespace parallel_lines_implies_m_no_perpendicular_lines_solution_l45_45842

noncomputable def parallel_slopes (m : ℝ) : Prop :=
  let y₁ := -m
  let y₂ := -2 / m
  y₁ = y₂

noncomputable def perpendicular_slopes (m : ℝ) : Prop :=
  let y₁ := -m
  let y₂ := -2 / m
  y₁ * y₂ = -1

theorem parallel_lines_implies_m (m : ℝ) : parallel_slopes m ↔ m = Real.sqrt 2 ∨ m = -Real.sqrt 2 :=
by
  sorry

theorem no_perpendicular_lines_solution (m : ℝ) : perpendicular_slopes m → false :=
by
  sorry

end parallel_lines_implies_m_no_perpendicular_lines_solution_l45_45842


namespace min_value_eval_l45_45058

noncomputable def min_value_expr (x y : ℝ) := 
  (x + 1/y) * (x + 1/y - 100) + (y + 1/x) * (y + 1/x - 100)

theorem min_value_eval (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x = y → min_value_expr x y = -2500 :=
by
  intros hxy
  -- Insert proof steps here
  sorry

end min_value_eval_l45_45058


namespace tom_average_speed_l45_45999

theorem tom_average_speed 
  (karen_speed : ℕ) (tom_distance : ℕ) (karen_advantage : ℕ) (delay : ℚ)
  (h1 : karen_speed = 60)
  (h2 : tom_distance = 24)
  (h3 : karen_advantage = 4)
  (h4 : delay = 4/60) :
  ∃ (v : ℚ), v = 45 := by
  sorry

end tom_average_speed_l45_45999


namespace geometric_sequence_fifth_term_l45_45789

theorem geometric_sequence_fifth_term (r : ℕ) (h₁ : 5 * r^3 = 405) : 5 * r^4 = 405 :=
sorry

end geometric_sequence_fifth_term_l45_45789


namespace sin_870_equals_half_l45_45809

theorem sin_870_equals_half :
  sin (870 * Real.pi / 180) = 1 / 2 := 
by
  -- Angle simplification
  have h₁ : 870 - 2 * 360 = 150 := by norm_num,
  -- Sine identity application
  have h₂ : sin (150 * Real.pi / 180) = sin (30 * Real.pi / 180) := by
    rw [mul_div_cancel_left 150 (ne_of_lt (by norm_num : 0 < (Real.pi : ℝ)) : (Real.pi : ℝ) ≠ 0)],
    congr,
    norm_num,

  -- Sine 30 degrees value
  have h₃ : sin (30 * Real.pi / 180) = 1 / 2 := by norm_num,

  -- Combine results
  rw [mul_div_cancel_left 870 (ne_of_lt (by norm_num : 0 < (Real.pi : ℝ)) : (Real.pi : ℝ) ≠ 0), h₁, h₂, h₃],
  sorry

end sin_870_equals_half_l45_45809


namespace right_angled_triangle_only_B_l45_45625

def forms_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_angled_triangle_only_B :
  forms_right_angled_triangle 1 (Real.sqrt 3) 2 ∧
  ¬forms_right_angled_triangle 1 2 2 ∧
  ¬forms_right_angled_triangle 4 5 6 ∧
  ¬forms_right_angled_triangle 1 1 (Real.sqrt 3) :=
by
  sorry

end right_angled_triangle_only_B_l45_45625


namespace sum_f_inv_l45_45491

noncomputable def f : ℝ → ℝ := sorry

axiom f_0 : f 0 = 1
axiom f_func_eq (x y : ℝ) : f (x * y + 1) = f x * f y - f y - x + 2

theorem sum_f_inv : ∑ i in Finset.range 2023 + 1, 1 / (f i * f (i + 1)) = 2023 / 4050 := sorry

end sum_f_inv_l45_45491


namespace cube_difference_divisibility_l45_45635

-- Given conditions
variables {m n : ℤ} (h1 : m % 2 = 1) (h2 : n % 2 = 1) (k : ℕ)

-- The equivalent statement to be proven
theorem cube_difference_divisibility (h1 : m % 2 = 1) (h2 : n % 2 = 1) : 
  (2^k ∣ m^3 - n^3) ↔ (2^k ∣ m - n) :=
sorry

end cube_difference_divisibility_l45_45635


namespace probability_two_english_teachers_l45_45908

open Nat

def num_english_teachers : ℕ := 3
def num_math_teachers : ℕ := 4
def num_social_studies_teachers : ℕ := 2
def total_teachers : ℕ := num_english_teachers + num_math_teachers + num_social_studies_teachers
def num_committees_of_size_2 : ℕ := choose total_teachers 2
def num_english_combinations_of_size_2 : ℕ := choose num_english_teachers 2

theorem probability_two_english_teachers :
  (num_english_combinations_of_size_2 : ℚ) / num_committees_of_size_2 = 1 / 12 :=
begin
  sorry
end

end probability_two_english_teachers_l45_45908


namespace binom_mult_eq_6720_l45_45693

theorem binom_mult_eq_6720 :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_mult_eq_6720_l45_45693


namespace train_passes_platform_in_43_2_seconds_l45_45124

open Real

noncomputable def length_of_train : ℝ := 360
noncomputable def length_of_platform : ℝ := 180
noncomputable def speed_of_train_kmph : ℝ := 45
noncomputable def speed_of_train_mps : ℝ := (45 * 1000) / 3600  -- Converting km/hr to m/s

noncomputable def total_distance : ℝ := length_of_train + length_of_platform
noncomputable def time_to_pass_platform : ℝ := total_distance / speed_of_train_mps

theorem train_passes_platform_in_43_2_seconds :
  time_to_pass_platform = 43.2 := by
  sorry

end train_passes_platform_in_43_2_seconds_l45_45124


namespace find_angle_find_area_l45_45831

noncomputable def degree_to_radian (d : ℝ) := d * π / 180

theorem find_angle {A B C : ℝ}
  (h : 2 * (Real.sin (B + C))^2 = Real.sqrt 3 * Real.sin (2 * A)) : A = degree_to_radian 60 :=
by {
  sorry
}

theorem find_area {A B C BC AC : ℝ}
  (h₁ : B + C = π/3)
  (h₂ : BC = 7)
  (h₃ : AC = 5)
  (h₄ : A = degree_to_radian 60) :
  let S := 0.5 * AC * BC * Real.sin A in
  S = 10 * Real.sqrt 3 :=
by {
  sorry
}

end find_angle_find_area_l45_45831


namespace narrow_black_stripes_l45_45568

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end narrow_black_stripes_l45_45568


namespace point_translation_proof_l45_45524

def Point := (ℝ × ℝ)

def translate_right (p : Point) (d : ℝ) : Point := (p.1 + d, p.2)

theorem point_translation_proof :
  let A : Point := (1, 2)
  let A' := translate_right A 2
  A' = (3, 2) :=
by
  let A : Point := (1, 2)
  let A' := translate_right A 2
  show A' = (3, 2)
  sorry

end point_translation_proof_l45_45524


namespace division_addition_problem_l45_45454

-- Define the terms used in the problem
def ten : ℕ := 10
def one_fifth : ℚ := 1 / 5
def six : ℕ := 6

-- Define the math problem
theorem division_addition_problem :
  (ten / one_fifth : ℚ) + six = 56 :=
by sorry

end division_addition_problem_l45_45454


namespace prove_union_sets_l45_45056

universe u

variable {α : Type u}
variable {M N : Set ℕ}
variable (a b : ℕ)

theorem prove_union_sets (h1 : M = {3, 4^a}) (h2 : N = {a, b}) (h3 : M ∩ N = {1}) : M ∪ N = {0, 1, 3} := sorry

end prove_union_sets_l45_45056


namespace volume_of_rectangular_prism_l45_45299

    theorem volume_of_rectangular_prism (height base_perimeter: ℝ) (h: height = 5) (b: base_perimeter = 16) :
      ∃ volume, volume = 80 := 
    by
      -- Mathematically equivalent proof goes here
      sorry
    
end volume_of_rectangular_prism_l45_45299


namespace factor_difference_of_squares_l45_45284

theorem factor_difference_of_squares (x : ℝ) : (x^2 - 64 = (x - 8) * (x + 8)) := by
  -- Conditions to state the problem with
  let a := x
  let b := 8
  have h1 : (x^2 - 64) = (a^2 - b^2), by
    rw [a, b]
  have h2 : (a^2 - b^2) = (a - b) * (a + b), from sorry
  -- Final equivalence
  exact (h1.trans h2)
  sorry
  -- Final proof is left as sorry.

end factor_difference_of_squares_l45_45284


namespace chord_length_l45_45646

theorem chord_length (r d AB : ℝ) (hr : r = 5) (hd : d = 4) : AB = 6 :=
by
  -- Given
  -- r = radius = 5
  -- d = distance from center to chord = 4

  -- prove AB = 6
  sorry

end chord_length_l45_45646


namespace owen_final_turtle_count_l45_45069

theorem owen_final_turtle_count (owen_initial johanna_initial : ℕ)
  (h1: owen_initial = 21)
  (h2: johanna_initial = owen_initial - 5) :
  let owen_after_1_month := 2 * owen_initial,
      johanna_after_1_month := johanna_initial / 2,
      owen_final := owen_after_1_month + johanna_after_1_month
  in
  owen_final = 50 :=
by
  -- Solution steps go here.
  sorry

end owen_final_turtle_count_l45_45069


namespace no_real_b_for_line_to_vertex_of_parabola_l45_45303

theorem no_real_b_for_line_to_vertex_of_parabola : 
  ¬ ∃ b : ℝ, ∃ x : ℝ, y = x + b ∧ y = x^2 + b^2 + 1 :=
by
  sorry

end no_real_b_for_line_to_vertex_of_parabola_l45_45303


namespace temperature_range_l45_45754

-- Define the problem conditions
def highest_temp := 26
def lowest_temp := 12

-- The theorem stating the range of temperature change
theorem temperature_range : ∀ t : ℝ, lowest_temp ≤ t ∧ t ≤ highest_temp :=
by sorry

end temperature_range_l45_45754


namespace factorize_x_l45_45002

theorem factorize_x^3_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
sorry

end factorize_x_l45_45002


namespace avg_bc_eq_70_l45_45763

-- Definitions of the given conditions
variables (a b c : ℝ)

def avg_ab (a b : ℝ) : Prop := (a + b) / 2 = 45
def diff_ca (a c : ℝ) : Prop := c - a = 50

-- The main theorem statement
theorem avg_bc_eq_70 (h1 : avg_ab a b) (h2 : diff_ca a c) : (b + c) / 2 = 70 :=
by
  sorry

end avg_bc_eq_70_l45_45763


namespace cost_of_one_bag_l45_45415

theorem cost_of_one_bag (x : ℝ) (h1 : ∀ p : ℝ, 60 * x = p -> 60 * p = 120 * x ) 
  (h2 : ∀ p1 p2: ℝ, 60 * x = p1 ∧ 15 * 1.6 * x = p2 ∧ 45 * 2.24 * x = 100.8 * x -> 124.8 * x - 120 * x = 1200) :
  x = 250 := 
sorry

end cost_of_one_bag_l45_45415


namespace sqrt_3_between_neg_1_and_2_l45_45800

theorem sqrt_3_between_neg_1_and_2 : -1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := by
  sorry

end sqrt_3_between_neg_1_and_2_l45_45800


namespace find_a_value_l45_45841

noncomputable def A (a : ℝ) : Set ℝ := {x | x = a}
noncomputable def B (a : ℝ) : Set ℝ := if a = 0 then ∅ else {x | a * x = 1}

theorem find_a_value (a : ℝ) :
  (A a ∩ B a = B a) → (a = 1 ∨ a = -1 ∨ a = 0) :=
by
  intro h
  sorry

end find_a_value_l45_45841


namespace length_of_goods_train_l45_45901

-- Define the given conditions
def speed_kmph := 72
def platform_length := 260
def crossing_time := 26

-- Convert speed to m/s
def speed_mps := (speed_kmph * 5) / 18

-- Calculate distance covered
def distance_covered := speed_mps * crossing_time

-- Define the length of the train
def train_length := distance_covered - platform_length

theorem length_of_goods_train : train_length = 260 := by
  sorry

end length_of_goods_train_l45_45901


namespace value_of_expression_l45_45168

theorem value_of_expression (m : ℝ) (h : 2 * m ^ 2 - 3 * m - 1 = 0) : 4 * m ^ 2 - 6 * m = 2 :=
sorry

end value_of_expression_l45_45168


namespace cross_product_correct_l45_45474

-- Define the vectors a and b
def a : (Fin 3 → ℤ) := ![3, 2, 4]
def b : (Fin 3 → ℤ) := ![6, -3, 8]

-- Define the cross product function for 3-dimensional vectors
def cross_product {R : Type*} [Ring R] (u v : Fin 3 → R) : Fin 3 → R :=
  ![
    u 1 * v 2 - u 2 * v 1,
    u 2 * v 0 - u 0 * v 2,
    u 0 * v 1 - u 1 * v 0
  ]

-- Define the expected result of the cross product
def result : (Fin 3 → ℤ) := ![28, 0, -21]

-- Theorem statement
theorem cross_product_correct : cross_product a b = result :=
by
  -- Here we leave the proof as a sorry placeholder
  sorry

end cross_product_correct_l45_45474


namespace proof_problem_l45_45881

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * Real.pi * x)

theorem proof_problem
  (a : ℝ)
  (h1 : ∀ x : ℝ, f (x - 1/2) = f (x + 1/2))
  (h2 : f (-1/4) = a) :
  f (9/4) = -a :=
by sorry

end proof_problem_l45_45881


namespace find_ellipse_equation_find_slope_l45_45308

noncomputable def eccentricity (a b : ℝ) : ℝ := 
  (real.sqrt (a ^ 2 - b ^ 2)) / a

def ellipse (a b : ℝ) (x y : ℝ) : Prop := 
  (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1

def isosceles_triangle (A B M : ℝ × ℝ) : Prop :=
  let AB := real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) in
  let AM := real.sqrt ((A.1 - M.1) ^ 2 + (A.2 - M.2) ^ 2) in
  let MB := real.sqrt ((M.1 - B.1) ^ 2 + (M.2 - B.2) ^ 2) in
  AM = MB

theorem find_ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity a b = real.sqrt 5 / 3) (x y : ℝ) 
  (h4 : ellipse a b (3 * real.sqrt 3 / 2) 1) : 
  ellipse 3 2 x y := sorry

theorem find_slope (a b : ℝ) (A B M : ℝ × ℝ) (h1 : A ≠ B) (h2 : M = (5/12, 0))
  (h3 : isosceles_triangle A B M) : 
  ((A.2 - B.2) / (A.1 - B.1)) = -2 / 3 := sorry

end find_ellipse_equation_find_slope_l45_45308


namespace probability_of_draw_l45_45896

noncomputable def P_A_winning : ℝ := 0.4
noncomputable def P_A_not_losing : ℝ := 0.9

theorem probability_of_draw : P_A_not_losing - P_A_winning = 0.5 :=
by 
  sorry

end probability_of_draw_l45_45896


namespace metal_waste_l45_45446

theorem metal_waste (a b : ℝ) (h : a < b) :
  let radius := a / 2
  let area_rectangle := a * b
  let area_circle := π * radius^2
  let side_square := a / Real.sqrt 2
  let area_square := side_square^2
  area_rectangle - area_square = a * b - ( a ^ 2 ) / 2 := by
  let radius := a / 2
  let area_rectangle := a * b
  let area_circle := π * (radius ^ 2)
  let side_square := a / Real.sqrt 2
  let area_square := side_square ^ 2
  sorry

end metal_waste_l45_45446


namespace polynomial_divisibility_l45_45074

theorem polynomial_divisibility (m : ℕ) (hm : 0 < m) :
  ∀ x : ℝ, x * (x + 1) * (2 * x + 1) ∣ (x + 1) ^ (2 * m) - x ^ (2 * m) - 2 * x - 1 :=
by
  intro x
  sorry

end polynomial_divisibility_l45_45074


namespace weight_of_b_l45_45393

theorem weight_of_b (A B C : ℕ) 
  (h1 : A + B + C = 129) 
  (h2 : A + B = 80) 
  (h3 : B + C = 86) : 
  B = 37 := 
by 
  sorry

end weight_of_b_l45_45393


namespace range_of_a_l45_45966

theorem range_of_a (a : ℝ) (h1 : a > 0)
  (h2 : ∃ x : ℝ, abs (Real.sin x) > a)
  (h3 : ∀ x : ℝ, x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) → (Real.sin x)^2 + a * Real.sin x - 1 ≥ 0) :
  a ∈ Set.Ico (Real.sqrt 2 / 2) 1 :=
sorry

end range_of_a_l45_45966


namespace trigonometric_comparison_l45_45318

open Real

theorem trigonometric_comparison :
  let a := 2 * sin (1 / 2)
  let b := 3 * sin (1 / 3)
  let c := 3 * cos (1 / 3)
  a < b ∧ b < c := 
by
  let a := 2 * sin (1 / 2)
  let b := 3 * sin (1 / 3)
  let c := 3 * cos (1 / 3)
  sorry

end trigonometric_comparison_l45_45318


namespace binom_mult_eq_6720_l45_45696

theorem binom_mult_eq_6720 :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_mult_eq_6720_l45_45696


namespace chemist_target_temperature_fahrenheit_l45_45917

noncomputable def kelvinToCelsius (K : ℝ) : ℝ := K - 273.15
noncomputable def celsiusToFahrenheit (C : ℝ) : ℝ := (C * 9 / 5) + 32

theorem chemist_target_temperature_fahrenheit :
  celsiusToFahrenheit (kelvinToCelsius (373.15 - 40)) = 140 :=
by
  sorry

end chemist_target_temperature_fahrenheit_l45_45917


namespace no_positive_integer_with_digits_greater_than_5_and_square_digits_less_than_5_l45_45283

theorem no_positive_integer_with_digits_greater_than_5_and_square_digits_less_than_5 :
  ¬ ∃ n : ℕ, (∀ d ∈ (Nat.digits 10 n), 5 < d) ∧ (∀ d ∈ (Nat.digits 10 (n^2)), d < 5) :=
by
  sorry

end no_positive_integer_with_digits_greater_than_5_and_square_digits_less_than_5_l45_45283


namespace trig_expression_value_l45_45833

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 :=
by
  sorry

end trig_expression_value_l45_45833


namespace b_joined_after_a_l45_45932

def months_b_joined (a_investment : ℕ) (b_investment : ℕ) (profit_ratio : ℕ × ℕ) (total_months : ℕ) : ℕ :=
  let a_months := total_months
  let b_months := total_months - (b_investment / (3500 * profit_ratio.snd / profit_ratio.fst / b_investment))
  total_months - b_months

theorem b_joined_after_a (a_investment b_investment total_months : ℕ) (profit_ratio : ℕ × ℕ) (h_a_investment : a_investment = 3500)
   (h_b_investment : b_investment = 21000) (h_profit_ratio : profit_ratio = (2, 3)) : months_b_joined a_investment b_investment profit_ratio total_months = 9 := by
  sorry

end b_joined_after_a_l45_45932


namespace linear_function_max_value_l45_45899

theorem linear_function_max_value (m x : ℝ) (h : -1 ≤ x ∧ x ≤ 3) (y : ℝ) 
  (hl : y = m * x - 2 * m) (hy : y = 6) : m = -2 ∨ m = 6 := 
by 
  sorry

end linear_function_max_value_l45_45899


namespace km_per_gallon_proof_l45_45128

-- Define the given conditions
def distance := 100
def gallons := 10

-- Define what we need to prove the correct answer
def kilometers_per_gallon := distance / gallons

-- Prove that the calculated kilometers per gallon is equal to 10
theorem km_per_gallon_proof : kilometers_per_gallon = 10 := by
  sorry

end km_per_gallon_proof_l45_45128


namespace no_solution_exists_l45_45827

theorem no_solution_exists (p : ℝ) : (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) = (x - p) / (x - 8) → false) ↔ p = 7 :=
by sorry

end no_solution_exists_l45_45827


namespace colorful_triangle_in_complete_graph_l45_45731

open SimpleGraph

theorem colorful_triangle_in_complete_graph (n : ℕ) (h : n ≥ 3) (colors : Fin n → Fin n → Fin (n - 1)) :
  ∃ (u v w : Fin n), u ≠ v ∧ v ≠ w ∧ w ≠ u ∧ colors u v ≠ colors v w ∧ colors v w ≠ colors w u ∧ colors w u ≠ colors u v :=
  sorry

end colorful_triangle_in_complete_graph_l45_45731


namespace inequality1_solution_inequality2_solution_l45_45014

variables (x a : ℝ)

theorem inequality1_solution : (∀ x : ℝ, (2 * x) / (x + 1) < 1 ↔ -1 < x ∧ x < 1) :=
by
  sorry

theorem inequality2_solution (a : ℝ) : 
  (∀ x : ℝ, x^2 + (2 - a) * x - 2 * a ≥ 0 ↔ 
    (a = -2 → true) ∧ 
    (a > -2 → (x ≤ -2 ∨ x ≥ a)) ∧ 
    (a < -2 → (x ≤ a ∨ x ≥ -2))) :=
by
  sorry

end inequality1_solution_inequality2_solution_l45_45014


namespace unique_four_digit_number_l45_45506

theorem unique_four_digit_number (N : ℕ) (a : ℕ) (x : ℕ) :
  (N = 1000 * a + x) ∧ (N = 7 * x) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (1 ≤ a ∧ a ≤ 9) →
  N = 3500 :=
by sorry

end unique_four_digit_number_l45_45506


namespace stephen_speed_second_third_l45_45392

theorem stephen_speed_second_third
  (first_third_speed : ℝ)
  (last_third_speed : ℝ)
  (total_distance : ℝ)
  (travel_time : ℝ)
  (time_in_hours : ℝ)
  (h1 : first_third_speed = 16)
  (h2 : last_third_speed = 20)
  (h3 : total_distance = 12)
  (h4 : travel_time = 15)
  (h5 : time_in_hours = travel_time / 60) :
  time_in_hours * (total_distance - (first_third_speed * time_in_hours + last_third_speed * time_in_hours)) = 12 := 
by 
  sorry

end stephen_speed_second_third_l45_45392


namespace smallest_integer_problem_l45_45246

theorem smallest_integer_problem (m : ℕ) (h1 : Nat.lcm 60 m / Nat.gcd 60 m = 28) : m = 105 := sorry

end smallest_integer_problem_l45_45246


namespace victory_points_value_l45_45518

theorem victory_points_value (V : ℕ) (H : ∀ (v d t : ℕ), 
    v + d + t = 20 ∧ v * V + d ≥ 40 ∧ v ≥ 6 ∧ (t = 20 - 5)) : 
    V = 3 := 
sorry

end victory_points_value_l45_45518


namespace mixed_price_calc_add_candy_a_to_mix_equal_weight_from_each_box_l45_45986

-- Problem 1
theorem mixed_price_calc (a b : ℕ) (m n : ℕ) (h_a : a = 30) (h_b : b = 25) 
                         (h_m : m = 30) (h_n : n = 20) :
  (a * m + b * n) / (m + n) = 28 := sorry

-- Problem 2
theorem add_candy_a_to_mix (a : ℕ) (x : ℝ) (h_a : a = 30) (price_mixed : ℝ) 
                           (weight_mixed : ℕ) (price_increase : ℝ) 
                           (h_price_mixed : price_mixed = 24)
                           (h_weight_mixed : weight_mixed = 100)
                           (h_price_increase : price_increase = 0.15) :
  let price_new := price_mixed * (1 + price_increase)
  (a * x + price_mixed * weight_mixed) / (x + weight_mixed) = price_new :=
  by
  let price_new := price_mixed * (1 + price_increase)
  have h_price_new : price_new = 24 * 1.15 := rfl
  exact sorry

-- Problem 3
theorem equal_weight_from_each_box (a b : ℕ) (m n : ℕ) (y : ℝ)
                                   (h_a : a = 30) (h_b : b = 25)
                                   (h_m : m = 40) (h_n : n = 60)
                                   (h_condition : (b * y + a * (40 - y)) / 40 = (a * y + b * (60 - y)) / 60) :
  y = 24 := sorry

end mixed_price_calc_add_candy_a_to_mix_equal_weight_from_each_box_l45_45986


namespace no_natural_number_exists_l45_45953

theorem no_natural_number_exists 
  (n : ℕ) : ¬ ∃ x y : ℕ, (2 * n * (n + 1) * (n + 2) * (n + 3) + 12) = x^2 + y^2 := 
by sorry

end no_natural_number_exists_l45_45953


namespace algebra_inequality_l45_45075

variable {x y z : ℝ}

theorem algebra_inequality
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^3 * (y^2 + z^2)^2 + y^3 * (z^2 + x^2)^2 + z^3 * (x^2 + y^2)^2
  ≥ x * y * z * (x * y * (x + y)^2 + y * z * (y + z)^2 + z * x * (z + x)^2) :=
sorry

end algebra_inequality_l45_45075


namespace average_temp_is_correct_l45_45873

-- Define the temperatures for each day
def sunday_temp : ℕ := 40
def monday_temp : ℕ := 50
def tuesday_temp : ℕ := 65
def wednesday_temp : ℕ := 36
def thursday_temp : ℕ := 82
def friday_temp : ℕ := 72
def saturday_temp : ℕ := 26

-- Define the total number of days in the week
def days_in_week : ℕ := 7

-- Define the total temperature for the week
def total_temperature : ℕ := sunday_temp + monday_temp + tuesday_temp + 
                             wednesday_temp + thursday_temp + friday_temp + 
                             saturday_temp

-- Define the average temperature calculation
def average_temperature : ℕ := total_temperature / days_in_week

-- The theorem to be proved
theorem average_temp_is_correct : average_temperature = 53 := by
  sorry

end average_temp_is_correct_l45_45873


namespace commercial_break_duration_l45_45467

theorem commercial_break_duration (n1 n2 m1 m2 : ℕ) (h1 : n1 = 3) (h2 : m1 = 5) (h3 : n2 = 11) (h4 : m2 = 2) :
  n1 * m1 + n2 * m2 = 37 :=
by
  -- Here, in a real proof, we would substitute and show the calculations.
  sorry

end commercial_break_duration_l45_45467


namespace water_supply_days_l45_45448

theorem water_supply_days (C V : ℕ) 
  (h1: C = 75 * (V + 10))
  (h2: C = 60 * (V + 20)) : 
  (C / V) = 100 := 
sorry

end water_supply_days_l45_45448


namespace at_least_one_casket_made_by_Cellini_son_l45_45848

-- Definitions for casket inscriptions
def golden_box := "The silver casket was made by Cellini"
def silver_box := "The golden casket was made by someone other than Cellini"

-- Predicate indicating whether a box was made by Cellini
def made_by_Cellini (box : String) : Prop :=
  box = "The golden casket was made by someone other than Cellini" ∨ box = "The silver casket was made by Cellini"

-- Our goal is to prove that at least one of the boxes was made by Cellini's son
theorem at_least_one_casket_made_by_Cellini_son :
  (¬ made_by_Cellini golden_box ∧ made_by_Cellini silver_box) ∨ (made_by_Cellini golden_box ∧ ¬ made_by_Cellini silver_box) → (¬ made_by_Cellini golden_box ∨ ¬ made_by_Cellini silver_box) :=
sorry

end at_least_one_casket_made_by_Cellini_son_l45_45848


namespace cost_of_one_bag_l45_45416

theorem cost_of_one_bag (x : ℝ) (h1 : ∀ p : ℝ, 60 * x = p -> 60 * p = 120 * x ) 
  (h2 : ∀ p1 p2: ℝ, 60 * x = p1 ∧ 15 * 1.6 * x = p2 ∧ 45 * 2.24 * x = 100.8 * x -> 124.8 * x - 120 * x = 1200) :
  x = 250 := 
sorry

end cost_of_one_bag_l45_45416


namespace narrow_black_stripes_l45_45546

variable (w n b : ℕ)

theorem narrow_black_stripes (w : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := by
  have h3 : w + n = (w + 7) + 1 := by rw [h1]
  have h4 : w + n = w + 8 := by linarith
  have h5 : n = 8 := by linarith
  exact h5

end narrow_black_stripes_l45_45546


namespace no_nat_transfer_initial_digit_end_increases_by_5_6_8_l45_45370

theorem no_nat_transfer_initial_digit_end_increases_by_5_6_8 :
  ∀ m : ℕ, m > 0 → (∃ a1 a2 … an : ℕ, (a1 ≠ 0) ∧ 
    (m = a1 * 10^(n-1) + T) ∧ 
    (m' = T * 10 + a1) ∧ 
    (∀ k ∈ {5, 6, 8}, m' ≠ k * m)) := 
begin
  intro m,
  intros m_pos,
  // Proof omitted
  sorry,
end

end no_nat_transfer_initial_digit_end_increases_by_5_6_8_l45_45370


namespace percentage_of_boys_from_schoolA_study_science_l45_45517

variable (T : ℝ) -- Total number of boys in the camp
variable (schoolA_boys : ℝ)
variable (science_boys : ℝ)

noncomputable def percentage_science_boys := (science_boys / schoolA_boys) * 100

theorem percentage_of_boys_from_schoolA_study_science 
  (h1 : schoolA_boys = 0.20 * T)
  (h2 : science_boys = schoolA_boys - 56)
  (h3 : T = 400) :
  percentage_science_boys science_boys schoolA_boys = 30 := 
by sorry

end percentage_of_boys_from_schoolA_study_science_l45_45517


namespace find_m_l45_45026

theorem find_m 
  (x1 x2 : ℝ) 
  (m : ℝ)
  (h1 : x1 + x2 = m)
  (h2 : x1 * x2 = 2 * m - 1)
  (h3 : x1^2 + x2^2 = 7) : 
  m = 5 :=
by
  sorry

end find_m_l45_45026


namespace remainder_div_82_l45_45734

theorem remainder_div_82 (x : ℤ) (h : ∃ k : ℤ, x + 17 = 41 * k + 22) : (x % 82 = 5) :=
by
  sorry

end remainder_div_82_l45_45734


namespace potato_cost_l45_45409

variables (x : ℝ)
variables (b a : ℝ)

def andrey_earnings (x : ℝ) : ℝ := 120 * x
def boris_earnings (x : ℝ) : ℝ := 124.8 * x

theorem potato_cost :
  (boris_earnings x) - (andrey_earnings x) = 1200 → x = 250 :=
  by
    unfold andrey_earnings
    unfold boris_earnings
    sorry

end potato_cost_l45_45409


namespace cost_of_one_bag_l45_45411

theorem cost_of_one_bag (x : ℝ) :
  let Andrey_earning := 60 * 2 * x
  let Boris_earning := 15 * 1.6 * x + 45 * (1.6 * 1.4) * x
  Boris_earning - Andrey_earning = 1200 →
  x = 250 := 
by
  intros
  let Andrey_earning := 60 * 2 * x
  let Boris_earning := 15 * 1.6 * x + 45 * (1.6 * 1.4) * x
  have h : Boris_earning - Andrey_earning = 1200 := by assumption
  let simplified_h := 
    calc
      Boris_earning - Andrey_earning
        = (15 * 1.6 * x + 45 * (1.6 * 1.4) * x) - (60 * 2 * x) : by simp [Andrey_earning, Boris_earning]
    ... = (15 * 1.6 * x + 45 * (1.6 * 1.4) * x) - 120 * x : by simp
    ... = (24 * x + 100.8 * x) - 120 * x : by simp
    ... = 124.8 * x - 120 * x : by simp
    ... = 4.8 * x : by simp
    ... = 1200 : by rw h
  exact (div_eq_iff (by norm_num : (4.8 : ℝ) ≠ 0)).1 simplified_h  -- solves for x

end cost_of_one_bag_l45_45411


namespace vector_dot_product_parallel_l45_45327

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b (m : ℝ) : ℝ × ℝ := (m, -1)
noncomputable def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = k • v

theorem vector_dot_product_parallel (m : ℝ) (h_parallel : is_parallel a (a.1 + m, a.2 + (-1))) :
  (a.1 * m + a.2 * (-1) = -5 / 2) :=
sorry

end vector_dot_product_parallel_l45_45327


namespace set_intersection_l45_45188

noncomputable def A : Set ℤ := {-1, 0, 1, 2}

noncomputable def B : Set ℝ := {x | ∃ y, y = Real.log 2 (4 - x^2) ∧ -2 < x ∧ x < 2}

theorem set_intersection : A ∩ B = {-1, 0, 1} :=
  sorry

end set_intersection_l45_45188


namespace inequality_proof_l45_45863

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) : ab < 1 ∧ 1 < (a^2 + b^2) / 2 :=
by
  sorry

end inequality_proof_l45_45863


namespace chip_exits_from_A2_l45_45346

noncomputable def chip_exit_cell (grid_size : ℕ) (initial_cell : ℕ × ℕ) (move_direction : ℕ × ℕ → ℕ × ℕ) : ℕ × ℕ :=
(1, 2) -- A2; we assume the implementation of function movement follows the solution as described

theorem chip_exits_from_A2 :
  chip_exit_cell 4 (3, 2) move_direction = (1, 2) :=
sorry  -- Proof omitted

end chip_exits_from_A2_l45_45346


namespace geometric_sequence_S6_div_S3_l45_45019

theorem geometric_sequence_S6_div_S3 (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h1 : a 1 + a 3 = 5 / 4)
  (h2 : a 2 + a 4 = 5 / 2)
  (hS : ∀ n, S n = a 1 * (1 - (2:ℝ) ^ n) / (1 - 2)) :
  S 6 / S 3 = 9 :=
by
  sorry

end geometric_sequence_S6_div_S3_l45_45019


namespace binomial_product_result_l45_45681

-- Defining the combination (binomial coefficient) formula
def combination (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- Lean theorem statement to prove the problem
theorem binomial_product_result : combination 10 3 * combination 8 3 = 6720 := by
  sorry

end binomial_product_result_l45_45681


namespace angle_relation_in_triangle_l45_45747

theorem angle_relation_in_triangle
    (a b c : ℝ)
    (A B C : ℝ)
    (h1 : b * (a + b) * (b + c) = a^3 + b * (a^2 + c^2) + c^3)
    (h2 : A + B + C = π) 
    (h3 : A > 0) 
    (h4 : B > 0) 
    (h5 : C > 0) :
    (1 / (Real.sqrt A + Real.sqrt B)) + (1 / (Real.sqrt B + Real.sqrt C)) = (2 / (Real.sqrt C + Real.sqrt A)) :=
sorry

end angle_relation_in_triangle_l45_45747


namespace count_multiples_6_or_8_not_both_l45_45509

-- Define the conditions
def is_multiple (n k : ℕ) : Prop := ∃ m : ℕ, n = k * m

-- Define the main proof statement
theorem count_multiples_6_or_8_not_both :
  (∑ k in Finset.filter (λ n, is_multiple n 6 ∨ is_multiple n 8 ∧ ¬(is_multiple n 6 ∧ is_multiple n 8)) (Finset.range 151), 1) = 31 := 
sorry

end count_multiples_6_or_8_not_both_l45_45509


namespace actual_distance_traveled_l45_45636

theorem actual_distance_traveled (D : ℝ) (h : D / 10 = (D + 20) / 20) : D = 20 :=
  sorry

end actual_distance_traveled_l45_45636


namespace scarves_per_box_l45_45248

theorem scarves_per_box (S : ℕ) 
  (boxes : ℕ := 8) 
  (mittens_per_box : ℕ := 6) 
  (total_clothing : ℕ := 80) 
  (total_mittens : ℕ := boxes * mittens_per_box) 
  (total_scarves : ℕ := total_clothing - total_mittens) 
  (scarves_per_box : ℕ := total_scarves / boxes) 
  : scarves_per_box = 4 := 
by 
  sorry

end scarves_per_box_l45_45248


namespace train_speed_l45_45249

-- Define the conditions in terms of distance and time
def train_length : ℕ := 160
def crossing_time : ℕ := 8

-- Define the expected speed
def expected_speed : ℕ := 20

-- The theorem stating the speed of the train given the conditions
theorem train_speed : (train_length / crossing_time) = expected_speed :=
by
  -- Note: The proof is omitted
  sorry

end train_speed_l45_45249


namespace square_area_l45_45404

theorem square_area (p : ℕ) (h : p = 48) : (p / 4) * (p / 4) = 144 := by
  sorry

end square_area_l45_45404


namespace required_fencing_l45_45268

-- Define constants given in the problem
def L : ℕ := 20
def A : ℕ := 720

-- Define the width W based on the area and the given length L
def W : ℕ := A / L

-- Define the total amount of fencing required
def F : ℕ := 2 * W + L

-- State the theorem that this amount of fencing is equal to 92
theorem required_fencing : F = 92 := by
  sorry

end required_fencing_l45_45268


namespace units_digit_expression_mod_10_l45_45015

theorem units_digit_expression_mod_10 : ((2 ^ 2023) * (5 ^ 2024) * (11 ^ 2025)) % 10 = 0 := 
by 
  -- Proof steps would go here
  sorry

end units_digit_expression_mod_10_l45_45015


namespace total_students_stratified_sampling_l45_45096

namespace HighSchool

theorem total_students_stratified_sampling 
  (sample_size : ℕ)
  (sample_grade10 : ℕ)
  (sample_grade11 : ℕ)
  (students_grade12 : ℕ) 
  (n : ℕ)
  (H1 : sample_size = 100)
  (H2 : sample_grade10 = 24)
  (H3 : sample_grade11 = 26)
  (H4 : students_grade12 = 600)
  (H5 : ∀ n, (students_grade12 / n * sample_size = sample_size - sample_grade10 - sample_grade11) → n = 1200) :
  n = 1200 :=
sorry

end HighSchool

end total_students_stratified_sampling_l45_45096


namespace sum_divisible_by_7_l45_45783

theorem sum_divisible_by_7 (n : ℕ) : (8^n + 6) % 7 = 0 := 
by
  sorry

end sum_divisible_by_7_l45_45783


namespace determine_a_range_l45_45985

open Real

theorem determine_a_range (a : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + a ≤ 0) → a ≤ 1 :=
sorry

end determine_a_range_l45_45985


namespace simplify_sqrt180_l45_45378

-- Conditions from the problem definition
def sqrt (x : ℕ) : ℝ := Real.sqrt x
def six := 6
def five := 5

-- The statement of the problem as a Lean theorem
theorem simplify_sqrt180 : sqrt 180 = six * sqrt five := by 
  sorry

end simplify_sqrt180_l45_45378


namespace number_of_triangles_in_polygon_with_200_sides_l45_45967

noncomputable def triangle_count (n : ℕ) (k : ℕ) : ℕ := (n.choose k)

theorem number_of_triangles_in_polygon_with_200_sides :
  triangle_count 200 3 = 1313400 :=
by
  sorry

end number_of_triangles_in_polygon_with_200_sides_l45_45967


namespace cos_7theta_l45_45166

theorem cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -45682/8192 :=
by
  sorry

end cos_7theta_l45_45166


namespace owen_turtles_l45_45068

theorem owen_turtles (o_initial : ℕ) (j_initial : ℕ) (o_after_month : ℕ) (j_remaining : ℕ) (o_final : ℕ) 
  (h1 : o_initial = 21)
  (h2 : j_initial = o_initial - 5)
  (h3 : o_after_month = 2 * o_initial)
  (h4 : j_remaining = j_initial / 2)
  (h5 : o_final = o_after_month + j_remaining) :
  o_final = 50 :=
sorry

end owen_turtles_l45_45068


namespace prove_seq_formula_l45_45155

noncomputable def seq (a : ℕ → ℝ) : ℕ → ℝ
| 0     => 1
| 1     => 5
| n + 2 => (2 * (seq a (n + 1))^2 - 3 * (seq a (n + 1)) - 9) / (2 * (seq a n))

theorem prove_seq_formula : ∀ (n : ℕ), seq a n = 2^(n + 2) - 3 :=
by
  sorry  -- Proof not needed for the mathematical translation

end prove_seq_formula_l45_45155


namespace no_real_solutions_l45_45598

theorem no_real_solutions :
  ¬ ∃ x : ℝ, (4 * x^3 + 3 * x^2 + x + 2) / (x - 2) = 4 * x^2 + 5 :=
by
  sorry

end no_real_solutions_l45_45598


namespace max_value_of_expression_l45_45190

theorem max_value_of_expression (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  8 * a + 3 * b + 5 * c ≤ 7 * Real.sqrt 2 :=
sorry

end max_value_of_expression_l45_45190


namespace factor_expression_l45_45471

theorem factor_expression (x y : ℝ) :
  75 * x^10 * y^3 - 150 * x^20 * y^6 = 75 * x^10 * y^3 * (1 - 2 * x^10 * y^3) :=
by
  sorry

end factor_expression_l45_45471


namespace holds_under_condition_l45_45976

theorem holds_under_condition (a b c : ℕ) (ha : a ≤ 10) (hb : b ≤ 10) (hc : c ≤ 10) (cond : b + 11 * c = 10 * a) :
  (10 * a + b) * (10 * a + c) = 100 * a * a + 100 * a + 11 * b * c :=
by
  sorry

end holds_under_condition_l45_45976


namespace secret_code_count_l45_45851

-- Conditions
def num_colors : ℕ := 8
def num_slots : ℕ := 5

-- The proof statement
theorem secret_code_count : (num_colors ^ num_slots) = 32768 := by
  sorry

end secret_code_count_l45_45851


namespace nathan_ate_total_gumballs_l45_45589

-- Define the constants and variables based on the conditions
def gumballs_small : Nat := 5
def gumballs_medium : Nat := 12
def gumballs_large : Nat := 20
def small_packages : Nat := 4
def medium_packages : Nat := 3
def large_packages : Nat := 2

-- The total number of gumballs Nathan ate
def total_gumballs : Nat := (small_packages * gumballs_small) + (medium_packages * gumballs_medium) + (large_packages * gumballs_large)

-- The theorem to prove
theorem nathan_ate_total_gumballs : total_gumballs = 96 :=
by
  unfold total_gumballs
  sorry

end nathan_ate_total_gumballs_l45_45589


namespace final_value_of_S_l45_45470

theorem final_value_of_S :
  ∀ (S n : ℕ), S = 1 → n = 1 →
  (∀ S n : ℕ, ¬ n > 3 → 
    (∃ S' n' : ℕ, S' = S + 2 * n ∧ n' = n + 1 ∧ 
      (∀ S n : ℕ, n > 3 → S' = 13))) :=
by 
  intros S n hS hn
  simp [hS, hn]
  sorry

end final_value_of_S_l45_45470


namespace narrow_black_stripes_are_eight_l45_45550

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end narrow_black_stripes_are_eight_l45_45550


namespace find_A_l45_45325

def U : Set ℕ := {1, 2, 3, 4, 5}

def compl_U (A : Set ℕ) : Set ℕ := U \ A

theorem find_A (A : Set ℕ) (hU : U = {1, 2, 3, 4, 5})
  (h_compl_U : compl_U A = {2, 3}) : A = {1, 4, 5} :=
by
  sorry

end find_A_l45_45325


namespace base_n_not_divisible_by_11_l45_45712

theorem base_n_not_divisible_by_11 :
  ∀ n, 2 ≤ n ∧ n ≤ 100 → (6 + 2*n + 5*n^2 + 4*n^3 + 2*n^4 + 4*n^5) % 11 ≠ 0 := by
  sorry

end base_n_not_divisible_by_11_l45_45712


namespace probability_of_square_root_less_than_seven_is_13_over_30_l45_45104

-- Definition of two-digit range and condition for square root check
def two_digit_numbers := Finset.range 100 \ Finset.range 10
def sqrt_condition (n : ℕ) : Prop := n < 49

-- The required probability calculation
def probability_square_root_less_than_seven : ℚ :=
  (↑(two_digit_numbers.filter sqrt_condition).card) / (↑two_digit_numbers.card)

-- The theorem stating the required probability
theorem probability_of_square_root_less_than_seven_is_13_over_30 :
  probability_square_root_less_than_seven = 13 / 30 := by
  sorry

end probability_of_square_root_less_than_seven_is_13_over_30_l45_45104


namespace probability_one_of_last_three_red_l45_45368

theorem probability_one_of_last_three_red :
  let total_balls := 10
  let red_balls := 3
  let total_children := 10
  let last_children := 3
  (3 / 10) * (7 / 10) * (7 / 10) * 3 = 21 / 100 :=
by
  sorry

end probability_one_of_last_three_red_l45_45368


namespace matrix_det_eq_seven_l45_45733

theorem matrix_det_eq_seven (p q r s : ℝ) (h : p * s - q * r = 7) : 
  (p - 2 * r) * s - (q - 2 * s) * r = 7 := 
sorry

end matrix_det_eq_seven_l45_45733


namespace jam_consumption_l45_45852

theorem jam_consumption (x y t : ℝ) :
  x + y = 100 →
  t = 45 * x / y →
  t = 20 * y / x →
  x = 40 ∧ y = 60 ∧ 
  (y / 45 = 4 / 3) ∧ 
  (x / 20 = 2) := by
  sorry

end jam_consumption_l45_45852


namespace probability_of_meeting_at_cafe_l45_45129

open Set

/-- Define the unit square where each side represents 1 hour (from 2:00 to 3:00 PM). -/
def unit_square : Set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 }

/-- Define the overlap condition for Cara and David meeting at the café. -/
def overlap_region : Set (ℝ × ℝ) :=
  { p | max (p.1 - 0.5) 0 ≤ p.2 ∧ p.2 ≤ min (p.1 + 0.5) 1 }

/-- The area of the overlap region within the unit square. -/
noncomputable def overlap_area : ℝ :=
  ∫ x in Icc 0 1, (min (x + 0.5) 1 - max (x - 0.5) 0)

theorem probability_of_meeting_at_cafe : overlap_area / 1 = 1 / 2 :=
by
  sorry

end probability_of_meeting_at_cafe_l45_45129


namespace smallest_positive_period_of_f_range_of_a_l45_45145

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem smallest_positive_period_of_f : (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (T = π) :=
by
  sorry

theorem range_of_a (a : ℝ) : (∀ x, f x ≤ a) → a ≥ Real.sqrt 2 :=
by
  sorry

end smallest_positive_period_of_f_range_of_a_l45_45145


namespace find_smallest_divisor_l45_45612

theorem find_smallest_divisor {n : ℕ} 
  (h : n = 44402) 
  (hdiv1 : (n + 2) % 30 = 0) 
  (hdiv2 : (n + 2) % 48 = 0) 
  (hdiv3 : (n + 2) % 74 = 0) 
  (hdiv4 : (n + 2) % 100 = 0) : 
  ∃ d, d = 37 ∧ d ∣ (n + 2) :=
sorry

end find_smallest_divisor_l45_45612


namespace coordinates_of_F_double_prime_l45_45618

-- Definitions of transformations
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Definition of initial point F
def F : ℝ × ℝ := (1, 1)

-- Definition of the transformations applied to point F
def F_prime : ℝ × ℝ := reflect_x F
def F_double_prime : ℝ × ℝ := reflect_y_eq_x F_prime

-- Theorem statement
theorem coordinates_of_F_double_prime : F_double_prime = (-1, 1) :=
by
  sorry

end coordinates_of_F_double_prime_l45_45618


namespace complete_square_solution_l45_45599

theorem complete_square_solution :
  ∀ x : ℝ, ∃ p q : ℝ, (5 * x^2 - 30 * x - 45 = 0) → ((x + p) ^ 2 = q) ∧ (p + q = 15) :=
by
  sorry

end complete_square_solution_l45_45599


namespace math_problem_l45_45804

theorem math_problem :
  (-1 : ℤ) ^ 49 + 2 ^ (4 ^ 3 + 3 ^ 2 - 7 ^ 2) = 16777215 := by
  sorry

end math_problem_l45_45804


namespace IntersectionOfAandB_l45_45729

def setA : Set ℝ := {x | x < 5}
def setB : Set ℝ := {x | -1 < x}

theorem IntersectionOfAandB : setA ∩ setB = {x | -1 < x ∧ x < 5} :=
sorry

end IntersectionOfAandB_l45_45729


namespace sum_of_nonnegative_reals_l45_45343

theorem sum_of_nonnegative_reals (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 27) :
  x + y + z = Real.sqrt 106 :=
sorry

end sum_of_nonnegative_reals_l45_45343


namespace range_of_a_l45_45156

theorem range_of_a (a : ℝ) :
  (a + 1 > 0 ∧ 3 - 2 * a > 0 ∧ a + 1 > 3 - 2 * a) ∨ (a + 1 < 0 ∧ 3 - 2 * a < 0 ∧ a + 1 > 3 - 2 * a) ∨ (a + 1 < 0 ∧ 3 - 2 * a > 0)
  → (2 / 3 < a ∧ a < 3 / 2) ∨ (a < -1) :=
by
  sorry

end range_of_a_l45_45156


namespace range_of_a_l45_45027

theorem range_of_a (a : ℝ) (h1 : a ≤ 1)
(h2 : ∃ n₁ n₂ n₃ : ℤ, a ≤ n₁ ∧ n₁ < n₂ ∧ n₂ < n₃ ∧ n₃ ≤ 2 - a
  ∧ (∀ x : ℤ, a ≤ x ∧ x ≤ 2 - a → x = n₁ ∨ x = n₂ ∨ x = n₃)) :
  -1 < a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l45_45027


namespace correct_calculation_l45_45623

theorem correct_calculation (a x y b : ℝ) :
  (-a - a = 0) = False ∧
  (- (x + y) = -x - y) = True ∧
  (3 * (b - 2 * a) = 3 * b - 2 * a) = False ∧
  (8 * a^4 - 6 * a^2 = 2 * a^2) = False :=
by
  sorry

end correct_calculation_l45_45623


namespace percentage_of_students_with_same_grades_l45_45918

noncomputable def same_grade_percentage (students_class : ℕ) (grades_A : ℕ) (grades_B : ℕ) (grades_C : ℕ) (grades_D : ℕ) (grades_E : ℕ) : ℚ :=
  ((grades_A + grades_B + grades_C + grades_D + grades_E : ℚ) / students_class) * 100

theorem percentage_of_students_with_same_grades :
  let students_class := 40
  let grades_A := 3
  let grades_B := 5
  let grades_C := 6
  let grades_D := 2
  let grades_E := 1
  same_grade_percentage students_class grades_A grades_B grades_C grades_D grades_E = 42.5 := by
  sorry

end percentage_of_students_with_same_grades_l45_45918


namespace division_example_l45_45900

theorem division_example : ∃ A B : ℕ, 23 = 6 * A + B ∧ A = 3 ∧ B < 6 := 
by sorry

end division_example_l45_45900


namespace division_quotient_l45_45427

theorem division_quotient (x : ℤ) (y : ℤ) (r : ℝ) (h1 : x > 0) (h2 : y = 96) (h3 : r = 11.52) :
  ∃ q : ℝ, q = (x - r) / y := 
sorry

end division_quotient_l45_45427


namespace smallest_six_digit_number_exists_l45_45654

def three_digit_number (n : ℕ) := n % 4 = 2 ∧ n % 5 = 2 ∧ n % 6 = 2 ∧ 100 ≤ n ∧ n < 1000

def valid_six_digit_number (m n : ℕ) := 
  (m * 1000 + n) % 4 = 0 ∧ (m * 1000 + n) % 5 = 0 ∧ (m * 1000 + n) % 6 = 0 ∧ 
  three_digit_number n ∧ 0 ≤ m ∧ m < 1000

theorem smallest_six_digit_number_exists : 
  ∃ m n, valid_six_digit_number m n ∧ (∀ m' n', valid_six_digit_number m' n' → m * 1000 + n ≤ m' * 1000 + n') :=
sorry

end smallest_six_digit_number_exists_l45_45654


namespace instantaneous_velocity_at_3_l45_45645

-- Define the displacement function s(t)
def displacement (t : ℝ) : ℝ := 2 * t^3

-- Define the time at which we want to calculate the instantaneous velocity
def time : ℝ := 3

-- Define the expected instantaneous velocity at t=3
def expected_velocity : ℝ := 54

-- Define the derivative of the displacement function as the velocity function
noncomputable def velocity (t : ℝ) : ℝ := deriv displacement t

-- Theorem: Prove that the instantaneous velocity at t=3 is 54
theorem instantaneous_velocity_at_3 : velocity time = expected_velocity := 
by {
  -- Here the detailed proof should go, but we skip it with sorry
  sorry
}

end instantaneous_velocity_at_3_l45_45645


namespace max_value_8a_3b_5c_l45_45192

theorem max_value_8a_3b_5c (a b c : ℝ) (h_condition : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  8 * a + 3 * b + 5 * c ≤ (Real.sqrt 373) / 6 :=
by
  sorry

end max_value_8a_3b_5c_l45_45192


namespace carnival_days_l45_45915

theorem carnival_days (d : ℕ) (h : 50 * d + 3 * (50 * d) - 30 * d - 75 = 895) : d = 5 :=
by
  sorry

end carnival_days_l45_45915


namespace perimeter_of_square_fence_l45_45238

theorem perimeter_of_square_fence :
  ∀ (n : ℕ) (post_gap post_width : ℝ), 
  4 * n - 4 = 24 →
  post_gap = 6 →
  post_width = 5 / 12 →
  4 * ((n - 1) * post_gap + n * post_width) = 156 :=
by
  intros n post_gap post_width h1 h2 h3
  sorry

end perimeter_of_square_fence_l45_45238


namespace intersect_value_l45_45989

noncomputable def coord_x_c : ℝ := 1
noncomputable def curve_C1 (x y : ℝ) : Prop := (x^2 / 4 + y^2 = 1)
noncomputable def line_l (x y t : ℝ) : Prop := (y = sqrt 3 + (sqrt 3 / 2) * t) ∧ (x = (1 / 2) * t)
noncomputable def point_P : (ℝ × ℝ) := (0, sqrt 3)
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem intersect_value : ∀ (A B : ℝ × ℝ), 
  (curve_C1 A.1 A.2) ∧ (curve_C1 B.1 B.2) ∧ (∃ t : ℝ, line_l A.1 A.2 t) ∧ (∃ t : ℝ, line_l B.1 B.2 t) →
  (1 / distance point_P A) + (1 / distance point_P B) = 3 / 2 :=
by
  sorry


end intersect_value_l45_45989


namespace exist_circle_subset_27_l45_45771

noncomputable def circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = radius^2

def circles : list (ℝ × ℝ) := 
  -- A list of 2015 distinct circle centers, which for simplicity we will just declare abstractly.
  sorry

def graph_of_circles (centers : list (ℝ × ℝ)) : Graph (ℝ × ℝ) :=
  ⟨centers, λ c1 c2, (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 ≤ (2 : ℝ)^2⟩

theorem exist_circle_subset_27 (centers : list (ℝ × ℝ)) (h : centers.length = 2015) :
  ∃ C : Finset (ℝ × ℝ), C.card = 27 ∧ (∀ x ∈ C, ∀ y ∈ C, circle x 1 → circle y 1 → ((x.1 - y.1)^2 + (x.2 - y.2)^2 ≤ (2 : ℝ)^2) ∨ ((x.1 - y.1)^2 + (x.2 - y.2)^2 > (2: ℝ)^2)) :=
begin
  sorry
end

end exist_circle_subset_27_l45_45771


namespace smallest_n_fact_expr_l45_45821

theorem smallest_n_fact_expr : ∃ n : ℕ, (∀ m : ℕ, m = 6 → n! = (n - 4) * (n - 3) * (n - 2) * (n - 1) * n * (n + 1)) ∧ n = 23 := by
  sorry

end smallest_n_fact_expr_l45_45821


namespace carrie_savings_l45_45669

-- Define the original prices and discount rates
def deltaPrice : ℝ := 850
def deltaDiscount : ℝ := 0.20
def unitedPrice : ℝ := 1100
def unitedDiscount : ℝ := 0.30

-- Calculate discounted prices
def deltaDiscountAmount : ℝ := deltaPrice * deltaDiscount
def unitedDiscountAmount : ℝ := unitedPrice * unitedDiscount

def deltaDiscountedPrice : ℝ := deltaPrice - deltaDiscountAmount
def unitedDiscountedPrice : ℝ := unitedPrice - unitedDiscountAmount

-- Define the savings by choosing the cheaper flight
def savingsByChoosingCheaperFlight : ℝ := unitedDiscountedPrice - deltaDiscountedPrice

-- The theorem stating the amount saved
theorem carrie_savings : savingsByChoosingCheaperFlight = 90 :=
by
  sorry

end carrie_savings_l45_45669


namespace select_two_people_l45_45759

theorem select_two_people {n : ℕ} (h1 : n ≠ 0) (h2 : n ≥ 2) (h3 : (n - 1) ^ 2 = 25) : n = 6 :=
by
  sorry

end select_two_people_l45_45759


namespace narrow_black_stripes_count_l45_45559

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end narrow_black_stripes_count_l45_45559


namespace problem_solution_l45_45169

theorem problem_solution
  (x : ℝ) (a b : ℕ) (hx_pos : 0 < x) (ha_pos : 0 < a) (hb_pos : 0 < b)
  (h_eq : x ^ 2 + 5 * x + 5 / x + 1 / x ^ 2 = 40)
  (h_form : x = a + Real.sqrt b) :
  a + b = 11 :=
sorry

end problem_solution_l45_45169


namespace functional_equation_solution_l45_45952

open Nat

theorem functional_equation_solution (f : ℕ+ → ℕ+) 
  (H : ∀ (m n : ℕ+), f (f (f m) * f (f m) + 2 * f (f n) * f (f n)) = m * m + 2 * n * n) : 
  ∀ n : ℕ+, f n = n := 
sorry

end functional_equation_solution_l45_45952


namespace max_marks_l45_45115

theorem max_marks (M : ℝ) (h1 : 0.25 * M = 185 + 25) : M = 840 :=
by
  sorry

end max_marks_l45_45115


namespace max_value_of_quadratic_l45_45323

theorem max_value_of_quadratic :
  ∃ y : ℝ, (∀ x : ℝ, y ≥ -x^2 + 5 * x - 4) ∧ y = 9 / 4 :=
sorry

end max_value_of_quadratic_l45_45323


namespace narrow_black_stripes_count_l45_45557

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end narrow_black_stripes_count_l45_45557


namespace complex_number_solution_l45_45157

theorem complex_number_solution (z : ℂ) (i : ℂ) (hi : i^2 = -1) (hz : i * (z - 1) = 1 - i) : z = -i :=
by sorry

end complex_number_solution_l45_45157


namespace modulo_calculation_l45_45274

theorem modulo_calculation : (68 * 97 * 113) % 25 = 23 := by
  sorry

end modulo_calculation_l45_45274


namespace range_of_m_l45_45154

theorem range_of_m (m : ℝ) : (∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → (x - 1) * (x - (m - 1)) > 0) → m > 1 :=
by
  intro h
  sorry

end range_of_m_l45_45154


namespace fiveLetterWordsWithAtLeastOneVowel_l45_45330

-- Definitions for the given conditions
def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'C', 'D', 'F']

-- Total number of 5-letter words with no restrictions
def totalWords := 6^5

-- Total number of 5-letter words containing no vowels
def noVowelWords := 4^5

-- Prove that the number of 5-letter words with at least one vowel is 6752
theorem fiveLetterWordsWithAtLeastOneVowel : (totalWords - noVowelWords) = 6752 := by
  sorry

end fiveLetterWordsWithAtLeastOneVowel_l45_45330


namespace Sally_cards_l45_45077

theorem Sally_cards (initial_cards : ℕ) (cards_from_dan : ℕ) (cards_bought : ℕ) :
  initial_cards = 27 →
  cards_from_dan = 41 →
  cards_bought = 20 →
  initial_cards + cards_from_dan + cards_bought = 88 :=
by {
  intros,
  sorry
}

end Sally_cards_l45_45077


namespace L_shaped_region_area_l45_45815

-- Define the conditions
def square_area (side_length : ℕ) : ℕ := side_length * side_length

def WXYZ_side_length : ℕ := 6
def XUVW_side_length : ℕ := 2
def TYXZ_side_length : ℕ := 3

-- Define the areas of the squares
def WXYZ_area : ℕ := square_area WXYZ_side_length
def XUVW_area : ℕ := square_area XUVW_side_length
def TYXZ_area : ℕ := square_area TYXZ_side_length

-- Lean statement to prove the area of the L-shaped region
theorem L_shaped_region_area : WXYZ_area - XUVW_area - TYXZ_area = 23 := by
  sorry

end L_shaped_region_area_l45_45815


namespace speed_of_man_is_approx_4_99_l45_45270

noncomputable def train_length : ℝ := 110  -- meters
noncomputable def train_speed : ℝ := 50  -- km/h
noncomputable def time_to_pass_man : ℝ := 7.2  -- seconds

def mps_to_kmph (speed : ℝ) : ℝ := speed * 3.6

noncomputable def relative_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

noncomputable def relative_speed_kmph : ℝ :=
  mps_to_kmph (relative_speed train_length time_to_pass_man)

noncomputable def speed_of_man (relative_speed_kmph : ℝ) (train_speed : ℝ) : ℝ :=
  relative_speed_kmph - train_speed

theorem speed_of_man_is_approx_4_99 :
  abs (speed_of_man relative_speed_kmph train_speed - 4.99) < 0.01 :=
by
  sorry

end speed_of_man_is_approx_4_99_l45_45270


namespace narrow_black_stripes_are_8_l45_45573

-- Define variables: w for wide black stripes, n for narrow black stripes, b for white stripes
variables (w n b : ℕ)

-- Given conditions
axiom cond1 : b = w + 7
axiom cond2 : w + n = b + 1

-- Theorem statement to prove that the number of narrow black stripes is 8
theorem narrow_black_stripes_are_8 : n = 8 :=
by sorry

end narrow_black_stripes_are_8_l45_45573


namespace graph1_higher_than_graph2_l45_45277

theorem graph1_higher_than_graph2 :
  ∀ (x : ℝ), (-x^2 + 2 * x + 3) ≥ (x^2 - 2 * x + 3) :=
by
  intros x
  sorry

end graph1_higher_than_graph2_l45_45277


namespace solve_diophantine_l45_45640

theorem solve_diophantine (x y : ℕ) (h1 : 1990 * x - 1989 * y = 1991) : x = 11936 ∧ y = 11941 := by
  have h_pos_x : 0 < x := by sorry
  have h_pos_y : 0 < y := by sorry
  have h_x : 1990 * 11936 = 1990 * x := by sorry
  have h_y : 1989 * 11941 = 1989 * y := by sorry
  sorry

end solve_diophantine_l45_45640


namespace part_I_part_II_l45_45500

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + a

theorem part_I (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≥ (1 : ℝ) / Real.exp 1 :=
sorry

theorem part_II (a x1 x2 x : ℝ) (hx1 : 0 < x1) (hx2 : x1 < x2) (hx : x1 < x ∧ x < x2) :
  (f x a - f x1 a) / (x - x1) < (f x a - f x2 a) / (x - x2) :=
sorry

end part_I_part_II_l45_45500


namespace compute_expression_in_terms_of_k_l45_45534

-- Define the main theorem to be proven, with all conditions directly translated to Lean statements.
theorem compute_expression_in_terms_of_k
  (x y : ℝ)
  (h : (x^2 + y^2) / (x^2 - y^2) + (x^2 - y^2) / (x^2 + y^2) = k) :
    (x^8 + y^8) / (x^8 - y^8) - (x^8 - y^8) / (x^8 + y^8) = ((k - 2)^2 * (k + 2)^2) / (4 * k * (k^2 + 4)) :=
by
  sorry

end compute_expression_in_terms_of_k_l45_45534


namespace perfect_square_sequence_l45_45769

theorem perfect_square_sequence (k : ℤ) (y : ℕ → ℤ) :
  (y 1 = 1) ∧ (y 2 = 1) ∧
  (∀ n : ℕ, y (n + 2) = (4 * k - 5) * y (n + 1) - y n + 4 - 2 * k) →
  (∀ n ≥ 1, ∃ m : ℤ, y n = m^2) ↔ (k = 1 ∨ k = 3) :=
sorry

end perfect_square_sequence_l45_45769


namespace cost_of_one_bag_of_potatoes_l45_45414

theorem cost_of_one_bag_of_potatoes :
  let x := 250 in
  ∀ (price : ℕ)
    (bags : ℕ)
    (andrey_initial_price : ℕ)
    (andrey_sold_price : ℕ)
    (boris_initial_price : ℕ)
    (boris_first_price : ℕ)
    (boris_second_price : ℕ)
    (earnings_andrey : ℕ)
    (earnings_boris_first : ℕ)
    (earnings_boris_second : ℕ)
    (total_earnings_boris : ℕ),
  bags = 60 →
  andrey_initial_price = price →
  andrey_sold_price = 2 * price →
  andrey_sold_price * bags = earnings_andrey →
  boris_initial_price = price →
  boris_first_price = 1.6 * price →
  boris_second_price = 2.24 * price →
  boris_first_price * 15 + boris_second_price * 45 = total_earnings_boris →
  total_earnings_boris = earnings_andrey + 1200 →
  price = x :=
by
  intros x price bags andrey_initial_price andrey_sold_price boris_initial_price boris_first_price boris_second_price earnings_andrey earnings_boris_first earnings_boris_second total_earnings_boris
  assume h_bags h_andrey_initial_price h_andrey_sold_price h_earnings_andrey h_boris_initial_price h_boris_first_price h_boris_second_price h_total_earnings_boris h_total_earnings_difference
  if h_necessary : x = 250 then
    sorry
  else
    sorry


end cost_of_one_bag_of_potatoes_l45_45414


namespace correct_operation_l45_45431

variables (a : ℝ)

-- defining the expressions to be compared
def lhs := 2 * a^2 * a^4
def rhs := 2 * a^6

theorem correct_operation : lhs a = rhs a := 
by sorry

end correct_operation_l45_45431


namespace inequality_problem_l45_45711

variable {a b c : ℝ}

theorem inequality_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
sorry

end inequality_problem_l45_45711


namespace correct_operation_l45_45430

variables (a : ℝ)

-- defining the expressions to be compared
def lhs := 2 * a^2 * a^4
def rhs := 2 * a^6

theorem correct_operation : lhs a = rhs a := 
by sorry

end correct_operation_l45_45430


namespace factorize_x_l45_45003

theorem factorize_x^3_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
sorry

end factorize_x_l45_45003


namespace sin_870_eq_half_l45_45807

theorem sin_870_eq_half : Real.sin (870 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_870_eq_half_l45_45807


namespace integer_solutions_l45_45477

theorem integer_solutions (n : ℕ) :
  n = 7 ↔ ∃ (x : ℤ), ∀ (x : ℤ), (3 * x^2 + 17 * x + 14 ≤ 20)  :=
by
  sorry

end integer_solutions_l45_45477


namespace binomial_product_l45_45690

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := by
  sorry

end binomial_product_l45_45690


namespace solve_for_y_l45_45150


theorem solve_for_y (b y : ℝ) (h : b ≠ 0) :
    Matrix.det ![
        ![y + b, y, y],
        ![y, y + b, y],
        ![y, y, y + b]] = 0 → y = -b := by
  sorry

end solve_for_y_l45_45150


namespace binom_coeff_mult_l45_45676

theorem binom_coeff_mult :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_coeff_mult_l45_45676


namespace price_reduction_l45_45269

variable (T : ℝ) -- The original price of the television
variable (first_discount : ℝ) -- First discount in percentage
variable (second_discount : ℝ) -- Second discount in percentage

theorem price_reduction (h1 : first_discount = 0.4) (h2 : second_discount = 0.4) : 
  (1 - (1 - first_discount) * (1 - second_discount)) = 0.64 :=
by
  sorry

end price_reduction_l45_45269


namespace TimTotalRunHoursPerWeek_l45_45094

def TimUsedToRunTimesPerWeek : ℕ := 3
def TimAddedExtraDaysPerWeek : ℕ := 2
def MorningRunHours : ℕ := 1
def EveningRunHours : ℕ := 1

theorem TimTotalRunHoursPerWeek :
  (TimUsedToRunTimesPerWeek + TimAddedExtraDaysPerWeek) * (MorningRunHours + EveningRunHours) = 10 :=
by
  sorry

end TimTotalRunHoursPerWeek_l45_45094


namespace correct_option_is_B_l45_45632

-- Define the Pythagorean theorem condition for right-angled triangles
def is_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Conditions given in the problem
def option_A : Prop := ¬is_right_angled_triangle 1 2 2
def option_B : Prop := is_right_angled_triangle 1 (Real.sqrt 3) 2
def option_C : Prop := ¬is_right_angled_triangle 4 5 6
def option_D : Prop := ¬is_right_angled_triangle 1 1 (Real.sqrt 3)

-- The formal proof problem statement
theorem correct_option_is_B : option_A ∧ option_B ∧ option_C ∧ option_D :=
by
  sorry

end correct_option_is_B_l45_45632


namespace binomial_product_result_l45_45682

-- Defining the combination (binomial coefficient) formula
def combination (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- Lean theorem statement to prove the problem
theorem binomial_product_result : combination 10 3 * combination 8 3 = 6720 := by
  sorry

end binomial_product_result_l45_45682


namespace binom_coeff_mult_l45_45675

theorem binom_coeff_mult :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_coeff_mult_l45_45675


namespace sqrt_180_eq_l45_45389

noncomputable def simplify_sqrt_180 : Real := 6 * Real.sqrt 5

theorem sqrt_180_eq : Real.sqrt 180 = simplify_sqrt_180 := 
by
  -- proof omitted
  sorry

end sqrt_180_eq_l45_45389


namespace narrow_black_stripes_l45_45567

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end narrow_black_stripes_l45_45567


namespace probability_contains_black_and_white_l45_45017

-- Define the conditions and proof statement
theorem probability_contains_black_and_white :
  let total_balls := 16 in
  let black_balls := 10 in
  let white_balls := 6 in
  let total_ways := Nat.choose total_balls 3 in
  let ways_all_black := Nat.choose black_balls 3 in
  let ways_all_white := Nat.choose white_balls 3 in
  let p_all_black_or_white := (ways_all_black + ways_all_white) / total_ways in
  (1 - p_all_black_or_white) = 3 / 4 :=
by
  let total_balls := 16
  let black_balls := 10
  let white_balls := 6
  let total_ways := Nat.choose total_balls 3
  let ways_all_black := Nat.choose black_balls 3
  let ways_all_white := Nat.choose white_balls 3
  let p_all_black_or_white := (ways_all_black + ways_all_white) / total_ways
  have p_ := (1 - p_all_black_or_white) = 3 / 4
  sorry

end probability_contains_black_and_white_l45_45017


namespace value_of_k_l45_45016

theorem value_of_k (k : ℝ) : (2 - k * 2 = -4 * (-1)) → k = -1 :=
by
  intro h
  sorry

end value_of_k_l45_45016


namespace cos_2alpha_2beta_l45_45484

variables (α β : ℝ)

open Real

theorem cos_2alpha_2beta (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) : cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_2alpha_2beta_l45_45484


namespace exists_nat_with_palindrome_decomp_l45_45463

def is_palindrome (n : ℕ) : Prop := 
  let s := n.toString
  s = s.reverse

theorem exists_nat_with_palindrome_decomp :
  ∃ n : ℕ, (∀ a b : ℕ, is_palindrome a → is_palindrome b → a * b = n → a ≠ b → (a, b) = (0, n) ∨ (b, a) = (0, n)) ∧ set.size { (a, b) | a * b = n ∧ is_palindrome a ∧ is_palindrome b } > 100 :=
begin
  use 2^101,
  sorry
end

end exists_nat_with_palindrome_decomp_l45_45463


namespace minimum_value_of_c_l45_45736

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  (Real.sqrt 3 / 12) * (a^2 + b^2 - c^2)

noncomputable def tan_formula (a b c B : ℝ) : Prop :=
  24 * (b * c - a) = b * Real.tan B

noncomputable def min_value_c (a b c : ℝ) : ℝ :=
  (2 * Real.sqrt 3) / 3

theorem minimum_value_of_c (a b c B : ℝ) (h₁ : 0 < B ∧ B < π / 2) (h₂ : 24 * (b * c - a) = b * Real.tan B)
  (h₃ : triangle_area a b c = (1/2) * a * b * Real.sin (π / 6)) :
  c ≥ min_value_c a b c :=
by
  sorry

end minimum_value_of_c_l45_45736


namespace roots_calc_l45_45098

theorem roots_calc {a b c d : ℝ} (h1: a ≠ 0) (h2 : 125 * a + 25 * b + 5 * c + d = 0) (h3 : -27 * a + 9 * b - 3 * c + d = 0) :
  (b + c) / a = -19 :=
by
  sorry

end roots_calc_l45_45098


namespace angle_of_inclination_l45_45137

theorem angle_of_inclination (α : ℝ) (h: 0 ≤ α ∧ α < 180) (slope_eq : Real.tan (Real.pi * α / 180) = Real.sqrt 3) :
  α = 60 :=
sorry

end angle_of_inclination_l45_45137


namespace average_percentage_taller_l45_45947

theorem average_percentage_taller 
  (h1 b1 h2 b2 h3 b3 : ℝ)
  (h1_eq : h1 = 228) (b1_eq : b1 = 200)
  (h2_eq : h2 = 120) (b2_eq : b2 = 100)
  (h3_eq : h3 = 147) (b3_eq : b3 = 140) :
  ((h1 - b1) / b1 * 100 + (h2 - b2) / b2 * 100 + (h3 - b3) / b3 * 100) / 3 = 13 := by
  rw [h1_eq, b1_eq, h2_eq, b2_eq, h3_eq, b3_eq]
  sorry

end average_percentage_taller_l45_45947


namespace pencils_in_boxes_l45_45512

theorem pencils_in_boxes (total_pencils : ℕ) (pencils_per_box : ℕ) (boxes_required : ℕ) 
    (h1 : total_pencils = 648) (h2 : pencils_per_box = 4) : boxes_required = 162 :=
sorry

end pencils_in_boxes_l45_45512


namespace transformation_matrix_l45_45103

open Real
open Matrix

def R_60 : Matrix (Fin 2) (Fin 2) ℝ :=
  !![(1/2 : ℝ), -(sqrt 3 / 2); (sqrt 3 / 2), (1/2)]

def S_2 : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, 0; 0, 2]

theorem transformation_matrix :
  (S_2 ⬝ R_60) = !![1, -sqrt 3; sqrt 3, 1] :=
by
  sorry

end transformation_matrix_l45_45103


namespace zongzi_unit_price_l45_45814

theorem zongzi_unit_price (uA uB : ℝ) (pA pB : ℝ) : 
  pA = 1200 → pB = 800 → uA = 2 * uB → pA / uA = pB / uB - 50 → uB = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end zongzi_unit_price_l45_45814


namespace find_c_l45_45397

theorem find_c (x c : ℤ) (h1 : 3 * x + 8 = 5) (h2 : c * x - 7 = 1) : c = -8 :=
sorry

end find_c_l45_45397


namespace infinitude_of_composite_z_l45_45076

theorem infinitude_of_composite_z (a : ℕ) (h : ∃ k : ℕ, k > 1 ∧ a = 4 * k^4) : 
  ∀ n : ℕ, ¬ Prime (n^4 + a) :=
by sorry

end infinitude_of_composite_z_l45_45076


namespace log_eq_solution_l45_45390

open Real

theorem log_eq_solution (x : ℝ) (h : x > 0) : log x + log (x + 1) = 2 ↔ x = (-1 + sqrt 401) / 2 :=
by
  sorry

end log_eq_solution_l45_45390


namespace number_of_cars_l45_45931

theorem number_of_cars (n s t C : ℕ) (h1 : n = 9) (h2 : s = 4) (h3 : t = 3) (h4 : n * s = t * C) : C = 12 :=
by
  sorry

end number_of_cars_l45_45931


namespace factorize_expression_l45_45705

theorem factorize_expression (x a : ℝ) : 4 * x - x * a^2 = x * (2 - a) * (2 + a) :=
by 
  sorry

end factorize_expression_l45_45705


namespace fraction_sum_l45_45114

variable {w x y : ℚ}  -- assuming w, x, and y are rational numbers

theorem fraction_sum (h1 : w / x = 1 / 3) (h2 : w / y = 2 / 3) : (x + y) / y = 3 :=
sorry

end fraction_sum_l45_45114


namespace narrow_black_stripes_are_eight_l45_45553

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end narrow_black_stripes_are_eight_l45_45553


namespace sqrt_180_simplify_l45_45381

theorem sqrt_180_simplify : real.sqrt 180 = 6 * real.sqrt 5 := by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  rw [h, real.sqrt_mul, real.sqrt_mul, real.sqrt_mul]
  all_goals { norm_num }
  sorry

end sqrt_180_simplify_l45_45381


namespace monitor_height_l45_45443

theorem monitor_height (width_in_inches : ℕ) (pixels_per_inch : ℕ) (total_pixels : ℕ) 
  (h1 : width_in_inches = 21) (h2 : pixels_per_inch = 100) (h3 : total_pixels = 2520000) : 
  total_pixels / (width_in_inches * pixels_per_inch) / pixels_per_inch = 12 :=
by
  sorry

end monitor_height_l45_45443


namespace neg_fraction_comparison_l45_45806

theorem neg_fraction_comparison : - (4 / 5 : ℝ) > - (5 / 6 : ℝ) :=
by {
  -- sorry to skip the proof
  sorry
}

end neg_fraction_comparison_l45_45806


namespace max_area_of_fencing_l45_45532

theorem max_area_of_fencing (P : ℕ) (hP : P = 150) 
  (x y : ℕ) (h1 : x + y = P / 2) : (x * y) ≤ 1406 :=
sorry

end max_area_of_fencing_l45_45532


namespace line_intersections_with_parabola_l45_45885

theorem line_intersections_with_parabola :
  ∃! (L : ℝ → ℝ) (l_count : ℕ),  
    l_count = 3 ∧
    (∀ x : ℝ, (L x) ∈ {x | (L 0 = 2) ∧ ∃ y, y * y = 8 * x ∧ L x = y}) := sorry

end line_intersections_with_parabola_l45_45885


namespace locus_of_right_angle_vertex_l45_45020

variables {x y : ℝ}

/-- Given points M(-2,0) and N(2,0), if P(x,y) is the right-angled vertex of
  a right-angled triangle with MN as its hypotenuse, then the locus equation
  of P is given by x^2 + y^2 = 4 with the condition x ≠ ±2. -/
theorem locus_of_right_angle_vertex (h : x ≠ 2 ∧ x ≠ -2) :
  x^2 + y^2 = 4 :=
sorry

end locus_of_right_angle_vertex_l45_45020


namespace commercial_break_duration_l45_45465

theorem commercial_break_duration (n1 n2 t1 t2 : ℕ) (h1 : n1 = 3) (h2: t1 = 5) (h3 : n2 = 11) (h4 : t2 = 2) : 
  n1 * t1 + n2 * t2 = 37 := 
by 
  sorry

end commercial_break_duration_l45_45465


namespace factorize_x_l45_45005

theorem factorize_x^3_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
sorry

end factorize_x_l45_45005


namespace narrow_black_stripes_l45_45565

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end narrow_black_stripes_l45_45565


namespace find_x_l45_45258

theorem find_x (x : ℝ) (h : 3550 - (x / 20.04) = 3500) : x = 1002 :=
by
  sorry

end find_x_l45_45258


namespace vance_family_stamp_cost_difference_l45_45762

theorem vance_family_stamp_cost_difference :
    let cost_rooster := 2 * 1.50
    let cost_daffodil := 5 * 0.75
    cost_daffodil - cost_rooster = 0.75 :=
by
    let cost_rooster := 2 * 1.50
    let cost_daffodil := 5 * 0.75
    show cost_daffodil - cost_rooster = 0.75
    sorry

end vance_family_stamp_cost_difference_l45_45762


namespace find_d_minus_c_l45_45929

variable (c d : ℝ)

def rotate180 (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (cx, cy) := center
  (2 * cx - x, 2 * cy - y)

def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (y, x)

def transformations (q : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_eq_x (rotate180 q (2, 3))

theorem find_d_minus_c :
  transformations (c, d) = (1, -4) → d - c = 7 :=
by
  intro h
  sorry

end find_d_minus_c_l45_45929


namespace root_interval_l45_45813

noncomputable def f (x : ℝ) : ℝ := 3^x + 3 * x - 8

theorem root_interval (h1 : f 1 < 0) (h2 : f 1.5 > 0) (h3 : f 1.25 < 0) :
  ∃ c, 1.25 < c ∧ c < 1.5 ∧ f c = 0 :=
by
  -- Proof by the Intermediate Value Theorem
  sorry

end root_interval_l45_45813


namespace domain_of_f_zeros_of_f_l45_45030

def log_a (a : ℝ) (x : ℝ) : ℝ := sorry -- Assume definition of logarithm base 'a'.

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log_a a (2 - x)

theorem domain_of_f (a : ℝ) : ∀ x : ℝ, 2 - x > 0 ↔ x < 2 :=
by
  sorry

theorem zeros_of_f (a : ℝ) : f a 1 = 0 :=
by
  sorry

end domain_of_f_zeros_of_f_l45_45030


namespace average_speed_l45_45226

theorem average_speed (d1 d2 : ℝ) (t1 t2 : ℝ) (h1 : d1 = 90) (h2 : d2 = 75) (ht1 : t1 = 1) (ht2 : t2 = 1) :
  (d1 + d2) / (t1 + t2) = 82.5 :=
by
  sorry

end average_speed_l45_45226


namespace sides_of_rectangle_EKMR_l45_45735

noncomputable def right_triangle_ACB (AC AB : ℕ) : Prop :=
AC = 3 ∧ AB = 4

noncomputable def rectangle_EKMR_area (area : ℚ) : Prop :=
area = 3/5

noncomputable def rectangle_EKMR_perimeter (x y : ℚ) : Prop :=
2 * (x + y) < 9

theorem sides_of_rectangle_EKMR (x y : ℚ) 
  (h_triangle : right_triangle_ACB 3 4)
  (h_area : rectangle_EKMR_area (3/5))
  (h_perimeter : rectangle_EKMR_perimeter x y) : 
  (x = 2 ∧ y = 3/10) ∨ (x = 3/10 ∧ y = 2) := 
sorry

end sides_of_rectangle_EKMR_l45_45735


namespace paint_cost_is_624_rs_l45_45217

-- Given conditions:
-- Length of floor is 21.633307652783934 meters.
-- Length is 200% more than the breadth (i.e., length = 3 * breadth).
-- Cost to paint the floor is Rs. 4 per square meter.

noncomputable def length : ℝ := 21.633307652783934
noncomputable def cost_per_sq_meter : ℝ := 4
noncomputable def breadth : ℝ := length / 3
noncomputable def area : ℝ := length * breadth
noncomputable def total_cost : ℝ := area * cost_per_sq_meter

theorem paint_cost_is_624_rs : total_cost = 624 := by
  sorry

end paint_cost_is_624_rs_l45_45217


namespace area_of_given_triangle_is_8_l45_45449

-- Define the vertices of the triangle
def x1 := 2
def y1 := -3
def x2 := -1
def y2 := 6
def x3 := 4
def y3 := -5

-- Define the determinant formula for the area of the triangle
def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℤ) : ℤ :=
  abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

theorem area_of_given_triangle_is_8 :
  area_of_triangle x1 y1 x2 y2 x3 y3 = 8 := by
  sorry

end area_of_given_triangle_is_8_l45_45449


namespace james_hours_per_year_l45_45529

def hours_per_day (trainings_per_day : Nat) (hours_per_training : Nat) : Nat :=
  trainings_per_day * hours_per_training

def days_per_week (total_days : Nat) (rest_days : Nat) : Nat :=
  total_days - rest_days

def hours_per_week (hours_day : Nat) (days_week : Nat) : Nat :=
  hours_day * days_week

def hours_per_year (hours_week : Nat) (weeks_year : Nat) : Nat :=
  hours_week * weeks_year

theorem james_hours_per_year :
  let trainings_per_day := 2
  let hours_per_training := 4
  let total_days_per_week := 7
  let rest_days_per_week := 2
  let weeks_per_year := 52
  hours_per_year 
    (hours_per_week 
      (hours_per_day trainings_per_day hours_per_training) 
      (days_per_week total_days_per_week rest_days_per_week)
    ) weeks_per_year
  = 2080 := by
  sorry

end james_hours_per_year_l45_45529


namespace instantaneous_velocity_at_3_l45_45220

-- Define the displacement function
def displacement (t : ℝ) : ℝ := t^2 - t

-- State the main theorem that we need to prove
theorem instantaneous_velocity_at_3 : (deriv displacement 3 = 5) := by
  sorry

end instantaneous_velocity_at_3_l45_45220


namespace simplify_sqrt_neg_five_squared_l45_45597

theorem simplify_sqrt_neg_five_squared : Real.sqrt ((-5 : ℝ)^2) = 5 := 
by
  sorry

end simplify_sqrt_neg_five_squared_l45_45597


namespace temperature_difference_l45_45766

theorem temperature_difference (T_high T_low : ℤ) (h_high : T_high = 11) (h_low : T_low = -11) :
  T_high - T_low = 22 := by
  sorry

end temperature_difference_l45_45766


namespace value_of_expression_l45_45406

theorem value_of_expression : (4 * 3) + 2 = 14 := by
  sorry

end value_of_expression_l45_45406


namespace product_divisible_by_four_l45_45240

noncomputable def probability_divisible_by_four : ℚ :=
  let p_odd := 1 / 2 in
  let p_two := 1 / 6 in
  let pr_not_div2 := ((p_odd) ^ 8) in
  let pr_div2_not_div4 := 8 * p_two * (p_odd ^ 7) in
  let pr_not_div4 := pr_not_div2 + pr_div2_not_div4 in
  1 - pr_not_div4

theorem product_divisible_by_four (ans : ℚ) : 
  ans = (757/768) ↔ probability_divisible_by_four = ans :=
by
  sorry

end product_divisible_by_four_l45_45240


namespace women_doubles_tournament_handshakes_l45_45941

theorem women_doubles_tournament_handshakes :
  ∀ (teams : List (List Prop)), List.length teams = 4 → (∀ t ∈ teams, List.length t = 2) →
  (∃ (handshakes : ℕ), handshakes = 24) :=
by
  intro teams h1 h2
  -- Assume teams are disjoint and participants shake hands meeting problem conditions
  -- The lean proof will follow the logical structure used for the mathematical solution
  -- We'll now formalize the conditions and the handshake calculation
  sorry

end women_doubles_tournament_handshakes_l45_45941


namespace trigonometric_identity_proof_l45_45745

open Real

theorem trigonometric_identity_proof (x y : ℝ) (hx : sin x / sin y = 4) (hy : cos x / cos y = 1 / 3) :
  (sin (2 * x) / sin (2 * y)) + (cos (2 * x) / cos (2 * y)) = 169 / 381 :=
by
  sorry

end trigonometric_identity_proof_l45_45745


namespace narrow_black_stripes_are_eight_l45_45549

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end narrow_black_stripes_are_eight_l45_45549


namespace no_two_distinct_integer_solutions_for_p_x_eq_2_l45_45537

open Polynomial

theorem no_two_distinct_integer_solutions_for_p_x_eq_2
  (p : ℤ[X])
  (h1 : ∃ a : ℤ, p.eval a = 1)
  (h3 : ∃ b : ℤ, p.eval b = 3) :
  ¬(∃ y1 y2 : ℤ, y1 ≠ y2 ∧ p.eval y1 = 2 ∧ p.eval y2 = 2) :=
by 
  sorry

end no_two_distinct_integer_solutions_for_p_x_eq_2_l45_45537


namespace find_p_l45_45982

theorem find_p (p q : ℚ) (h1 : 5 * p + 6 * q = 10) (h2 : 6 * p + 5 * q = 17) : p = 52 / 11 :=
by
  sorry

end find_p_l45_45982


namespace picture_distance_from_right_end_l45_45445

def distance_from_right_end_of_wall (wall_width picture_width position_from_left : ℕ) : ℕ := 
  wall_width - (position_from_left + picture_width)

theorem picture_distance_from_right_end :
  ∀ (wall_width picture_width position_from_left : ℕ), 
  wall_width = 24 -> 
  picture_width = 4 -> 
  position_from_left = 5 -> 
  distance_from_right_end_of_wall wall_width picture_width position_from_left = 15 :=
by
  intros wall_width picture_width position_from_left hw hp hp_left
  rw [hw, hp, hp_left]
  sorry

end picture_distance_from_right_end_l45_45445


namespace balanced_polygons_characterization_l45_45135

def convex_polygon (n : ℕ) (vertices : Fin n → Point) : Prop := 
  -- Definition of convex_polygon should go here
  sorry

def is_balanced (n : ℕ) (vertices : Fin n → Point) (M : Point) : Prop := 
  -- Definition of is_balanced should go here
  sorry

theorem balanced_polygons_characterization :
  ∀ (n : ℕ) (vertices : Fin n → Point) (M : Point),
  convex_polygon n vertices →
  is_balanced n vertices M →
  n = 3 ∨ n = 5 ∨ n = 7 :=
by sorry

end balanced_polygons_characterization_l45_45135


namespace range_of_x_for_valid_sqrt_l45_45341

theorem range_of_x_for_valid_sqrt (x : ℝ) (h : 2 * x - 4 ≥ 0) : x ≥ 2 :=
by
  sorry

end range_of_x_for_valid_sqrt_l45_45341


namespace parabola_vertex_b_l45_45403

theorem parabola_vertex_b (a b c p : ℝ) (h₁ : p ≠ 0)
  (h₂ : ∀ x, (x = p → -p = a * (p^2) + b * p + c) ∧ (x = 0 → p = c)) :
  b = - (4 / p) :=
sorry

end parabola_vertex_b_l45_45403


namespace final_match_l45_45355

-- Definitions of players and conditions
inductive Player
| Antony | Bart | Carl | Damian | Ed | Fred | Glen | Harry

open Player

-- Condition definitions
def beat (p1 p2 : Player) : Prop := sorry

-- Given conditions
axiom Bart_beats_Antony : beat Bart Antony
axiom Carl_beats_Damian : beat Carl Damian
axiom Glen_beats_Harry : beat Glen Harry
axiom Glen_beats_Carl : beat Glen Carl
axiom Carl_beats_Bart : beat Carl Bart
axiom Ed_beats_Fred : beat Ed Fred
axiom Glen_beats_Ed : beat Glen Ed

-- The proof statement
theorem final_match : beat Glen Carl :=
by
  sorry

end final_match_l45_45355


namespace nancy_indian_food_freq_l45_45202

-- Definitions based on the problem
def antacids_per_indian_day := 3
def antacids_per_mexican_day := 2
def antacids_per_other_day := 1
def mexican_per_week := 2
def total_antacids_per_month := 60
def weeks_per_month := 4
def days_per_week := 7

-- The proof statement
theorem nancy_indian_food_freq :
  ∃ (I : ℕ), (total_antacids_per_month = 
    weeks_per_month * (antacids_per_indian_day * I + 
    antacids_per_mexican_day * mexican_per_week + 
    antacids_per_other_day * (days_per_week - I - mexican_per_week))) ∧ I = 3 :=
by
  sorry

end nancy_indian_food_freq_l45_45202


namespace triangles_in_divided_square_l45_45521

theorem triangles_in_divided_square (V : ℕ) (marked_points : ℕ) (triangles : ℕ) 
  (h1 : V = 24) -- Vertices - 20 marked points and 4 vertices 
  (h2 : marked_points = 20) -- Marked points
  (h3 : triangles = F - 1) -- Each face (F) except the outer one is a triangle
  (h4 : V - E + F = 2) -- Euler's formula for planar graphs
  (h5 : E = (3*F + 1) / 2) -- Relationship between edges and faces
  (F : ℕ) -- Number of faces including the external face
  (E : ℕ) -- Number of edges
  : triangles = 42 := 
by 
  sorry

end triangles_in_divided_square_l45_45521


namespace trapezium_area_proof_l45_45637

def trapeziumArea (a b h : ℕ) : ℕ :=
  (1 / 2) * (a + b) * h

theorem trapezium_area_proof :
  let a := 20
  let b := 18
  let h := 14
  trapeziumArea a b h = 266 := by
  sorry

end trapezium_area_proof_l45_45637


namespace inequality_proof_l45_45310

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
sorry

end inequality_proof_l45_45310


namespace amount_after_two_years_l45_45902

-- Definition of initial amount and the rate of increase
def initial_value : ℝ := 32000
def rate_of_increase : ℝ := 0.125
def time_period : ℕ := 2

-- The compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- The proof problem: Prove that after 2 years the amount is 40500
theorem amount_after_two_years : compound_interest initial_value rate_of_increase time_period = 40500 :=
sorry

end amount_after_two_years_l45_45902


namespace sally_cards_l45_45078

theorem sally_cards (initial_cards dan_cards bought_cards : ℕ) (h1 : initial_cards = 27) (h2 : dan_cards = 41) (h3 : bought_cards = 20) :
  initial_cards + dan_cards + bought_cards = 88 := by
  sorry

end sally_cards_l45_45078


namespace problem_l45_45724

variable (x : ℝ)

theorem problem (h : x - (1 / x) = Real.sqrt 2) : x^1023 - (1 / x^1023) = 5 * Real.sqrt 2 :=
by
  sorry

end problem_l45_45724


namespace max_students_distribution_l45_45083

-- Define the four quantities
def pens : ℕ := 4261
def pencils : ℕ := 2677
def erasers : ℕ := 1759
def notebooks : ℕ := 1423

-- Prove that the greatest common divisor (GCD) of these four quantities is 1
theorem max_students_distribution : Nat.gcd (Nat.gcd (Nat.gcd pens pencils) erasers) notebooks = 1 :=
by
  sorry

end max_students_distribution_l45_45083


namespace minimum_value_l45_45059

open Real

theorem minimum_value {x y : ℝ} (hx : 0 < x) (hy : 0 < y) :
  ∃ u, (u = x + 1 / y + y + 1 / x) ∧ ((x + 1 / y) * (x + 1 / y - 100) + (y + 1 / x) * (y + 1 / x - 100) = 1 / 2 * (u - 100) ^ 2 - 2500) ∧ -2500 ≤ 1 / 2 * (u - 100) ^ 2 - 2500 :=
begin
  sorry
end

end minimum_value_l45_45059


namespace find_b_l45_45294

theorem find_b
  (a b c : ℚ)
  (h1 : (4 : ℚ) * a = 12)
  (h2 : (4 * (4 * b) = - (14:ℚ) + 3 * a)) :
  b = -(7:ℚ) / 2 :=
by sorry

end find_b_l45_45294


namespace shooting_average_l45_45650

noncomputable def total_points (a b c d : ℕ) : ℕ :=
  (a * 10) + (b * 9) + (c * 8) + (d * 7)

noncomputable def average_points (total : ℕ) (shots : ℕ) : ℚ :=
  total / shots

theorem shooting_average :
  let a := 1
  let b := 4
  let c := 3
  let d := 2
  let shots := 10
  total_points a b c d = 84 ∧
  average_points (total_points a b c d) shots = 8.4 :=
by {
  sorry
}

end shooting_average_l45_45650


namespace log_expression_value_l45_45893

theorem log_expression_value
  (h₁ : x + (Real.log 32 / Real.log 8) = 1.6666666666666667)
  (h₂ : Real.log 32 / Real.log 8 = 1.6666666666666667) :
  x = 0 :=
by
  sorry

end log_expression_value_l45_45893


namespace intersection_P_M_l45_45861

open Set Int

def P : Set ℤ := {x | 0 ≤ x ∧ x < 3}

def M : Set ℤ := {x | x^2 ≤ 9}

theorem intersection_P_M : P ∩ M = {0, 1, 2} := by
  sorry

end intersection_P_M_l45_45861


namespace abc_values_l45_45965

theorem abc_values (a b c : ℝ) 
  (ha : |a| > 1) 
  (hb : |b| > 1) 
  (hc : |c| > 1) 
  (hab : b = a^2 / (2 - a^2)) 
  (hbc : c = b^2 / (2 - b^2)) 
  (hca : a = c^2 / (2 - c^2)) : 
  a + b + c = 6 ∨ a + b + c = -4 ∨ a + b + c = -6 :=
sorry

end abc_values_l45_45965


namespace sin_cos_ratio_l45_45746

theorem sin_cos_ratio (x y : ℝ) (h1 : sin x / sin y = 4) (h2 : cos x / cos y = 1 / 3) :
  (sin (2 * x) / sin (2 * y)) + (cos (2 * x) / cos (2 * y)) = 395 / 381 :=
by
  sorry

end sin_cos_ratio_l45_45746


namespace tree_growth_per_two_weeks_l45_45866

-- Definitions based on conditions
def initial_height_meters : ℕ := 2
def initial_height_centimeters : ℕ := initial_height_meters * 100
def final_height_centimeters : ℕ := 600
def total_growth : ℕ := final_height_centimeters - initial_height_centimeters
def weeks_in_4_months : ℕ := 16
def number_of_two_week_periods : ℕ := weeks_in_4_months / 2

-- Objective: Prove that the growth every two weeks is 50 centimeters
theorem tree_growth_per_two_weeks :
  (total_growth / number_of_two_week_periods) = 50 :=
  by
  sorry

end tree_growth_per_two_weeks_l45_45866


namespace simplify_sqrt_180_l45_45386

theorem simplify_sqrt_180 : sqrt 180 = 6 * sqrt 5 :=
by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  calc
    sqrt 180
      = sqrt (2^2 * 3^2 * 5)     : by rw [h]
  ... = sqrt (2^2) * sqrt (3^2) * sqrt 5 : by rw [sqrt_mul, sqrt_mul]
  ... = 2 * 3 * sqrt 5          : by rw [sqrt_sq, sqrt_sq]
  ... = 6 * sqrt 5              : by norm_num

end simplify_sqrt_180_l45_45386


namespace problem1_problem2_l45_45151

theorem problem1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + y + 3) : x + y ≥ 6 :=
sorry

theorem problem2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + y + 3) : x * y ≥ 9 :=
sorry

end problem1_problem2_l45_45151


namespace arithmetic_sequence_S12_l45_45494

def arithmetic_sequence_sum (a d n : ℕ) : ℕ :=
  n * (2*a + (n-1)*d) / 2

def a_n (a d n : ℕ) : ℕ :=
  a + (n-1)*d

variable (a d : ℕ)

theorem arithmetic_sequence_S12 (h : a_n a d 4 + a_n a d 9 = 10) :
  arithmetic_sequence_sum a d 12 = 60 :=
by sorry

end arithmetic_sequence_S12_l45_45494


namespace beyonce_album_songs_l45_45273

theorem beyonce_album_songs
  (singles : ℕ)
  (album1_songs album2_songs album3_songs total_songs : ℕ)
  (h1 : singles = 5)
  (h2 : album1_songs = 15)
  (h3 : album2_songs = 15)
  (h4 : total_songs = 55) :
  album3_songs = 20 :=
by
  sorry

end beyonce_album_songs_l45_45273


namespace solve_quadratic_l45_45224

theorem solve_quadratic (x : ℝ) : x^2 = x ↔ (x = 0 ∨ x = 1) :=
by
  sorry

end solve_quadratic_l45_45224


namespace part_one_solution_part_two_solution_l45_45715

-- Definitions and conditions
def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part (1): When a = 1, solution set of the inequality f(x) > 1 is (1/2, +∞)
theorem part_one_solution (x : ℝ) :
  f x 1 > 1 ↔ x > 1 / 2 := sorry

-- Part (2): If the inequality f(x) > x holds for x ∈ (0,1), range of values for a is (0, 2]
theorem part_two_solution (a : ℝ) :
  (∀ x, 0 < x ∧ x < 1 → f x a > x) ↔ 0 < a ∧ a ≤ 2 := sorry

end part_one_solution_part_two_solution_l45_45715


namespace primes_x_y_eq_l45_45023

theorem primes_x_y_eq 
  {p q x y : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q)
  (hx : 0 < x) (hy : 0 < y)
  (hp_lt_x : x < p) (hq_lt_y : y < q)
  (h : (p : ℚ) / x + (q : ℚ) / y = (p * y + q * x) / (x * y)) :
  x = y :=
sorry

end primes_x_y_eq_l45_45023


namespace certain_event_abs_nonneg_l45_45937

theorem certain_event_abs_nonneg (x : ℝ) : |x| ≥ 0 :=
by
  sorry

end certain_event_abs_nonneg_l45_45937


namespace find_x_l45_45118

-- Define the condition from the problem statement
def condition1 (x : ℝ) : Prop := 70 = 0.60 * x + 22

-- Translate the question to the Lean statement form
theorem find_x (x : ℝ) (h : condition1 x) : x = 80 :=
by {
  sorry
}

end find_x_l45_45118


namespace sum_of_nonnegative_reals_l45_45342

theorem sum_of_nonnegative_reals (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 27) :
  x + y + z = Real.sqrt 106 :=
sorry

end sum_of_nonnegative_reals_l45_45342


namespace y_increases_as_x_increases_l45_45974

-- Define the linear function y = (m^2 + 2)x
def linear_function (m x : ℝ) : ℝ := (m^2 + 2) * x

-- Prove that y increases as x increases
theorem y_increases_as_x_increases (m x1 x2 : ℝ) (h : x1 < x2) : linear_function m x1 < linear_function m x2 :=
by
  -- because m^2 + 2 is always positive, the function is strictly increasing
  have hm : 0 < m^2 + 2 := by linarith [pow_two_nonneg m]
  have hx : (m^2 + 2) * x1 < (m^2 + 2) * x2 := by exact (mul_lt_mul_left hm).mpr h
  exact hx

end y_increases_as_x_increases_l45_45974


namespace new_unemployment_rate_is_66_percent_l45_45081

theorem new_unemployment_rate_is_66_percent
  (initial_unemployment_rate : ℝ)
  (initial_employment_rate : ℝ)
  (u_increases_by_10_percent : initial_unemployment_rate * 1.1 = new_unemployment_rate)
  (e_decreases_by_15_percent : initial_employment_rate * 0.85 = new_employment_rate)
  (sum_is_100_percent : initial_unemployment_rate + initial_employment_rate = 100) :
  new_unemployment_rate = 66 :=
by
  sorry

end new_unemployment_rate_is_66_percent_l45_45081


namespace cost_of_Roger_cookie_l45_45828

theorem cost_of_Roger_cookie
  (art_cookie_length : ℕ := 4)
  (art_cookie_width : ℕ := 3)
  (art_cookie_count : ℕ := 10)
  (roger_cookie_side : ℕ := 3)
  (art_cookie_price : ℕ := 50)
  (same_dough_used : ℕ := art_cookie_count * art_cookie_length * art_cookie_width)
  (roger_cookie_area : ℕ := roger_cookie_side * roger_cookie_side)
  (roger_cookie_count : ℕ := same_dough_used / roger_cookie_area) :
  (500 / roger_cookie_count) = 38 := by
  sorry

end cost_of_Roger_cookie_l45_45828


namespace B_work_rate_l45_45259

theorem B_work_rate (B : ℕ) (A_rate C_rate : ℚ) 
  (A_work : A_rate = 1 / 6)
  (C_work : C_rate = 1 / 8 * (1 / 6 + 1 / B))
  (combined_work : 1 / 6 + 1 / B + C_rate = 1 / 3) : 
  B = 28 :=
by 
  sorry

end B_work_rate_l45_45259


namespace average_speed_l45_45253

/--
On the first day of her vacation, Louisa traveled 100 miles.
On the second day, traveling at the same average speed, she traveled 175 miles.
If the 100-mile trip took 3 hours less than the 175-mile trip,
prove that her average speed (in miles per hour) was 25.
-/
theorem average_speed (v : ℝ) (h1 : 100 / v + 3 = 175 / v) : v = 25 :=
by 
  sorry

end average_speed_l45_45253


namespace no_natural_number_n_exists_l45_45955

theorem no_natural_number_n_exists (n : ℕ) :
  ¬ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 + y^2 = 2 * n * (n + 1) * (n + 2) * (n + 3) + 12 := 
sorry

end no_natural_number_n_exists_l45_45955


namespace vasya_can_guess_number_in_10_questions_l45_45588

noncomputable def log2 (n : ℕ) : ℝ := 
  Real.log n / Real.log 2

theorem vasya_can_guess_number_in_10_questions (n q : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 1000) (h3 : q = 10) :
  q ≥ log2 n := 
by
  sorry

end vasya_can_guess_number_in_10_questions_l45_45588


namespace correct_statement_c_l45_45432

theorem correct_statement_c (five_boys_two_girls : Nat := 7) (select_three : Nat := 3) :
  (∃ boys girls : Nat, boys + girls = five_boys_two_girls ∧ boys = 5 ∧ girls = 2) →
  (∃ selected_boys selected_girls : Nat, selected_boys + selected_girls = select_three ∧ selected_boys > 0) :=
by
  sorry

end correct_statement_c_l45_45432


namespace problem_lean_l45_45280

theorem problem_lean (k b : ℤ) : 
  ∃ n : ℤ, n = 25 ∧ n^2 = (k + 1)^4 - k^4 ∧ 3 * n + 100 = b^2 :=
sorry

end problem_lean_l45_45280


namespace Chris_age_proof_l45_45940

theorem Chris_age_proof (m c : ℕ) (h1 : c = 3 * m - 22) (h2 : c + m = 70) : c = 47 := by
  sorry

end Chris_age_proof_l45_45940


namespace ratio_of_sides_l45_45714
-- Import the complete math library

-- Define the conditions as hypotheses
variables (s x y : ℝ)
variable (h_outer_area : (3 * s)^2 = 9 * s^2)
variable (h_side_lengths : 3 * s = s + 2 * x)
variable (h_y_length : y + x = 3 * s)

-- State the theorem
theorem ratio_of_sides (h_outer_area : (3 * s)^2 = 9 * s^2)
  (h_side_lengths : 3 * s = s + 2 * x)
  (h_y_length : y + x = 3 * s) :
  y / x = 2 := by
  sorry

end ratio_of_sides_l45_45714


namespace probability_of_third_round_expected_value_of_X_variance_of_X_l45_45935

-- Define the probabilities for passing each round
def P_A : ℚ := 2 / 3
def P_B : ℚ := 3 / 4
def P_C : ℚ := 4 / 5

-- Prove the probability of reaching the third round
theorem probability_of_third_round :
  P_A * P_B = 1 / 2 := sorry

-- Define the probability distribution
def P_X (x : ℕ) : ℚ :=
  if x = 1 then 1 / 3 
  else if x = 2 then 1 / 6
  else if x = 3 then 1 / 2
  else 0

-- Expected value
def EX : ℚ := 1 * (1 / 3) + 2 * (1 / 6) + 3 * (1 / 2)

theorem expected_value_of_X :
  EX = 13 / 6 := sorry

-- E(X^2) computation
def EX2 : ℚ := 1^2 * (1 / 3) + 2^2 * (1 / 6) + 3^2 * (1 / 2)

-- Variance
def variance_X : ℚ := EX2 - EX^2

theorem variance_of_X :
  variance_X = 41 / 36 := sorry

end probability_of_third_round_expected_value_of_X_variance_of_X_l45_45935


namespace service_cost_is_correct_l45_45174

def service_cost_per_vehicle(cost_per_liter: ℝ)
                            (num_minivans: ℕ) 
                            (num_trucks: ℕ)
                            (total_cost: ℝ) 
                            (minivan_tank_liters: ℝ)
                            (truck_size_increase_pct: ℝ) 
                            (total_fuel: ℝ) 
                            (total_fuel_cost: ℝ) 
                            (total_service_cost: ℝ)
                            (num_vehicles: ℕ) 
                            (service_cost_per_vehicle: ℝ) : Prop :=
  cost_per_liter = 0.70 ∧
  num_minivans = 4 ∧
  num_trucks = 2 ∧
  total_cost = 395.4 ∧
  minivan_tank_liters = 65 ∧
  truck_size_increase_pct = 1.2 ∧
  total_fuel = (4 * minivan_tank_liters) + (2 * (minivan_tank_liters * (1 + truck_size_increase_pct))) ∧
  total_fuel_cost = total_fuel * cost_per_liter ∧
  total_service_cost = total_cost - total_fuel_cost ∧
  num_vehicles = num_minivans + num_trucks ∧
  service_cost_per_vehicle = total_service_cost / num_vehicles

-- Now, we state the theorem we want to prove.
theorem service_cost_is_correct :
  service_cost_per_vehicle 0.70 4 2 395.4 65 1.2 546 382.2 13.2 6 2.2 :=
by {
    sorry
}

end service_cost_is_correct_l45_45174


namespace value_added_to_number_l45_45092

theorem value_added_to_number (n v : ℤ) (h1 : n = 9)
  (h2 : 3 * (n + 2) = v + n) : v = 24 :=
by
  sorry

end value_added_to_number_l45_45092


namespace factorial_fraction_is_integer_l45_45022

open Nat

theorem factorial_fraction_is_integer (m n : ℕ) : 
  ↑((factorial (2 * m)) * (factorial (2 * n))) % (factorial m * factorial n * factorial (m + n)) = 0 := sorry

end factorial_fraction_is_integer_l45_45022


namespace factorize_expression_l45_45142

variable {R : Type} [CommRing R]

theorem factorize_expression (x y : R) : 
  4 * (x + y)^2 - (x^2 - y^2)^2 = (x + y)^2 * (2 + x - y) * (2 - x + y) := 
by 
  sorry

end factorize_expression_l45_45142


namespace binom_mult_l45_45685

open Nat

theorem binom_mult : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binom_mult_l45_45685


namespace pencil_count_l45_45215

theorem pencil_count (P N X : ℝ) 
  (h1 : 96 * P + 24 * N = 520) 
  (h2 : X * P + 4 * N = 60) 
  (h3 : P + N = 15.512820512820513) :
  X = 3 :=
by
  sorry

end pencil_count_l45_45215


namespace polynomial_identity_l45_45087

noncomputable def p (x : ℝ) : ℝ := x 

theorem polynomial_identity (p : ℝ → ℝ) (h : ∀ q : ℝ → ℝ, ∀ x : ℝ, p (q x) = q (p x)) : 
  (∀ x : ℝ, p x = x) :=
by
  sorry

end polynomial_identity_l45_45087


namespace probability_sqrt_two_digit_less_than_seven_l45_45106

noncomputable def prob_sqrt_less_than_seven : ℚ := 
  let favorable := 39
  let total := 90
  favorable / total

theorem probability_sqrt_two_digit_less_than_seven : 
  prob_sqrt_less_than_seven = 13 / 30 := by
  sorry

end probability_sqrt_two_digit_less_than_seven_l45_45106


namespace narrow_black_stripes_count_l45_45558

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end narrow_black_stripes_count_l45_45558


namespace factorization_of_x_squared_minus_64_l45_45289

theorem factorization_of_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factorization_of_x_squared_minus_64_l45_45289


namespace mike_total_money_l45_45366

theorem mike_total_money (num_bills : ℕ) (value_per_bill : ℕ) (h1 : num_bills = 9) (h2 : value_per_bill = 5) :
  (num_bills * value_per_bill) = 45 :=
by
  sorry

end mike_total_money_l45_45366


namespace z_in_second_quadrant_l45_45984

open Complex

-- Given the condition
def satisfies_eqn (z : ℂ) : Prop := z * (1 - I) = 4 * I

-- Define the second quadrant condition
def in_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

theorem z_in_second_quadrant (z : ℂ) (h : satisfies_eqn z) : in_second_quadrant z :=
  sorry

end z_in_second_quadrant_l45_45984


namespace correct_inequality_l45_45502

variables {a b c : ℝ}
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem correct_inequality (h_a_pos : a > 0) (h_discriminant_pos : b^2 - 4 * a * c > 0) (h_c_neg : c < 0) (h_b_neg : b < 0) :
  a * b * c > 0 :=
sorry

end correct_inequality_l45_45502


namespace vertical_angles_eq_l45_45247

theorem vertical_angles_eq (A B : Type) (are_vertical : A = B) :
  A = B := 
by
  exact are_vertical

end vertical_angles_eq_l45_45247


namespace total_ingredient_cups_l45_45086

def butter_flour_sugar_ratio_butter := 2
def butter_flour_sugar_ratio_flour := 5
def butter_flour_sugar_ratio_sugar := 3
def flour_used := 15

theorem total_ingredient_cups :
  butter_flour_sugar_ratio_butter + 
  butter_flour_sugar_ratio_flour + 
  butter_flour_sugar_ratio_sugar = 10 →
  flour_used / butter_flour_sugar_ratio_flour = 3 →
  6 + 15 + 9 = 30 := by
  intros
  sorry

end total_ingredient_cups_l45_45086


namespace transformed_center_coordinates_l45_45276

-- Define the original center of the circle
def center_initial : ℝ × ℝ := (3, -4)

-- Define the function for reflection across the x-axis
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the function for translation by a certain number of units up
def translate_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

-- Define the problem statement
theorem transformed_center_coordinates :
  translate_up (reflect_x_axis center_initial) 5 = (3, 9) :=
by
  sorry

end transformed_center_coordinates_l45_45276


namespace narrow_black_stripes_l45_45566

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end narrow_black_stripes_l45_45566


namespace count_multiples_of_6_or_8_but_not_both_l45_45507

theorem count_multiples_of_6_or_8_but_not_both: 
  let multiples_of_six := finset.filter (λ n, 6 ∣ n) (finset.range 151)
  let multiples_of_eight := finset.filter (λ n, 8 ∣ n) (finset.range 151)
  let multiples_of_twenty_four := finset.filter (λ n, 24 ∣ n) (finset.range 151)
  multiples_of_six.card + multiples_of_eight.card - 2 * multiples_of_twenty_four.card = 31 := 
by {
  -- Provided proof omitted
  sorry
}

end count_multiples_of_6_or_8_but_not_both_l45_45507


namespace num_children_in_family_l45_45648

def regular_ticket_cost := 15
def elderly_ticket_cost := 10
def adult_ticket_cost := 12
def child_ticket_cost := adult_ticket_cost - 5
def total_money_handled := 3 * 50
def change_received := 3
def num_adults := 4
def num_elderly := 2
def total_cost_for_adults := num_adults * adult_ticket_cost
def total_cost_for_elderly := num_elderly * elderly_ticket_cost
def total_cost_of_tickets := total_money_handled - change_received

theorem num_children_in_family : ∃ (num_children : ℕ), 
  total_cost_of_tickets = total_cost_for_adults + total_cost_for_elderly + num_children * child_ticket_cost ∧ 
  num_children = 11 := 
by
  sorry

end num_children_in_family_l45_45648


namespace adult_ticket_cost_l45_45200

theorem adult_ticket_cost (C : ℝ) (h1 : ∀ (a : ℝ), a = C + 8)
  (h2 : ∀ (s : ℝ), s = C + 4)
  (h3 : 5 * C + 2 * (C + 8) + 2 * (C + 4) = 150) :
  ∃ (a : ℝ), a = 22 :=
by {
  sorry
}

end adult_ticket_cost_l45_45200


namespace domain_of_function_l45_45880

theorem domain_of_function :
  ∀ x : ℝ, 3 * x - 2 > 0 ∧ 2 * x - 1 > 0 ↔ x > (2 / 3) := by
  intro x
  sorry

end domain_of_function_l45_45880


namespace weight_of_a_l45_45435

theorem weight_of_a (a b c d e : ℝ)
  (h1 : (a + b + c) / 3 = 84)
  (h2 : (a + b + c + d) / 4 = 80)
  (h3 : e = d + 8)
  (h4 : (b + c + d + e) / 4 = 79) :
  a = 80 :=
by
  sorry

end weight_of_a_l45_45435


namespace Tim_weekly_water_intake_l45_45093

variable (daily_bottle_intake : ℚ)
variable (additional_intake : ℚ)
variable (quart_to_ounces : ℚ)
variable (days_in_week : ℕ := 7)

theorem Tim_weekly_water_intake (H1 : daily_bottle_intake = 2 * 1.5)
                              (H2 : additional_intake = 20)
                              (H3 : quart_to_ounces = 32) :
  (daily_bottle_intake * quart_to_ounces + additional_intake) * days_in_week = 812 := by
  sorry

end Tim_weekly_water_intake_l45_45093


namespace permutations_no_solution_l45_45455

open Equiv

theorem permutations_no_solution :
  ¬(∃ (a b c d : Fin 50 → Fin 50), 
    a.perm (Fin.val 50) ∧ 
    b.perm (Fin.val 50) ∧ 
    c.perm (Fin.val 50) ∧ 
    d.perm (Fin.val 50) ∧ 
    (∑ i, a i * b i) = 2 * (∑ i, c i * d i)) :=
by
  sorry

end permutations_no_solution_l45_45455


namespace div_ad_bc_by_k_l45_45196

theorem div_ad_bc_by_k 
  (a b c d l k m n : ℤ)
  (h1 : a * l + b = k * m)
  (h2 : c * l + d = k * n) : 
  k ∣ (a * d - b * c) :=
sorry

end div_ad_bc_by_k_l45_45196


namespace irreducible_fraction_l45_45596

theorem irreducible_fraction (n : ℤ) : Int.gcd (2 * n + 1) (3 * n + 1) = 1 :=
sorry

end irreducible_fraction_l45_45596


namespace find_m_value_l45_45498

def f (x : ℝ) : ℝ := |x + 1| - |x - 1|

noncomputable def find_m (m : ℝ) : Prop :=
  f (f m) = f 2002 - 7 / 2

theorem find_m_value : find_m (-3 / 8) :=
by
  unfold find_m
  sorry

end find_m_value_l45_45498


namespace coordinates_of_point_P_l45_45835

theorem coordinates_of_point_P 
  (P : ℝ × ℝ)
  (h1 : P.1 < 0 ∧ P.2 < 0) 
  (h2 : abs P.2 = 3)
  (h3 : abs P.1 = 5) :
  P = (-5, -3) :=
sorry

end coordinates_of_point_P_l45_45835


namespace weight_of_replaced_oarsman_l45_45879

noncomputable def average_weight (W : ℝ) : ℝ := W / 20

theorem weight_of_replaced_oarsman (W : ℝ) (W_avg : ℝ) (H1 : average_weight W = W_avg) (H2 : average_weight (W + 40) = W_avg + 2) : W = 40 :=
by sorry

end weight_of_replaced_oarsman_l45_45879


namespace range_of_a_ineq_l45_45539

noncomputable def range_of_a (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 1 ∧ 1 < x₂ ∧ x₁ * x₁ + (a * a - 1) * x₁ + (a - 2) = 0 ∧
                x₂ * x₂ + (a * a - 1) * x₂ + (a - 2) = 0

theorem range_of_a_ineq (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ < 1 ∧ 1 < x₂ ∧
    x₁^2 + (a^2 - 1) * x₁ + (a - 2) = 0 ∧
    x₂^2 + (a^2 - 1) * x₂ + (a - 2) = 0) → -2 < a ∧ a < 1 :=
sorry

end range_of_a_ineq_l45_45539


namespace smallest_N_l45_45825

-- Conditions translated to Lean definitions
def f (n : ℕ) : ℕ := Nat.digits 5 n |>.sum
def g (n : ℕ) : ℕ := Nat.digits 9 (f n) |>.sum

-- The proof problem statement
theorem smallest_N'_mod_1000 : 
  ∃ N' : ℕ, (Nat.digits 18 (g N')).any (λ d, d = 10) ∧ N' % 1000 = 619 := by
  sorry

end smallest_N_l45_45825


namespace number_of_arrangements_l45_45350

theorem number_of_arrangements (n : ℕ) (h1 : 8 = n) (h2 : ¬ ∃ i : ℕ, i ≤ 7 ∧ i > 0 ∧ Alice = (people.nth i) ∧ Bob = (people.nth (i+1))) : 
  (fact 8 - fact 7 * 2) = 30240 :=
by
  sorry

end number_of_arrangements_l45_45350


namespace obtuse_angle_probability_l45_45204

noncomputable def probability_obtuse_angle : ℝ :=
  let F : ℝ × ℝ := (0, 3)
  let G : ℝ × ℝ := (5, 0)
  let H : ℝ × ℝ := (2 * Real.pi + 2, 0)
  let I : ℝ × ℝ := (2 * Real.pi + 2, 3)
  let rectangle_area : ℝ := (2 * Real.pi + 2) * 3
  let semicircle_radius : ℝ := Real.sqrt (2.5^2 + 1.5^2)
  let semicircle_area : ℝ := (1 / 2) * Real.pi * semicircle_radius^2
  semicircle_area / rectangle_area

theorem obtuse_angle_probability :
  probability_obtuse_angle = 17 / (24 + 4 * Real.pi) :=
by
  sorry

end obtuse_angle_probability_l45_45204


namespace line_through_point_intersecting_circle_eq_l45_45925

theorem line_through_point_intersecting_circle_eq :
  ∃ k l : ℝ, (x + 2*y + 9 = 0 ∨ 2*x - y + 3 = 0) ∧ 
    ∀ L : ℝ × ℝ,  
      (L = (-3, -3)) ∧ (x^2 + y^2 + 4*y - 21 = 0) → 
      (L = (-3,-3) → (x + 2*y + 9 = 0 ∨ 2*x - y + 3 = 0)) := 
sorry

end line_through_point_intersecting_circle_eq_l45_45925


namespace binomial_product_result_l45_45680

-- Defining the combination (binomial coefficient) formula
def combination (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- Lean theorem statement to prove the problem
theorem binomial_product_result : combination 10 3 * combination 8 3 = 6720 := by
  sorry

end binomial_product_result_l45_45680


namespace pages_in_book_l45_45165

theorem pages_in_book
  (x : ℝ)
  (h1 : x - (x / 6 + 10) = (5 * x) / 6 - 10)
  (h2 : (5 * x) / 6 - 10 - ((1 / 5) * ((5 * x) / 6 - 10) + 20) = (2 * x) / 3 - 28)
  (h3 : (2 * x) / 3 - 28 - ((1 / 4) * ((2 * x) / 3 - 28) + 25) = x / 2 - 46)
  (h4 : x / 2 - 46 = 72) :
  x = 236 := 
sorry

end pages_in_book_l45_45165


namespace jerry_can_escape_l45_45358

theorem jerry_can_escape (d : ℝ) (V_J V_T : ℝ) (h1 : (1 / 5) < d) (h2 : d < (1 / 4)) (h3 : V_T = 4 * V_J) :
  (4 * d) / V_J < 1 / (2 * V_J) :=
by
  sorry

end jerry_can_escape_l45_45358


namespace orthogonal_pairs_in_cube_is_36_l45_45959

-- Define a cube based on its properties, i.e., having vertices, edges, and faces.
structure Cube :=
(vertices : Fin 8 → Fin 3)
(edges : Fin 12 → (Fin 2 → Fin 8))
(faces : Fin 6 → (Fin 4 → Fin 8))

-- Define orthogonal pairs of a cube as an axiom.
axiom orthogonal_line_plane_pairs (c : Cube) : ℕ

-- The main theorem stating the problem's conclusion.
theorem orthogonal_pairs_in_cube_is_36 (c : Cube): orthogonal_line_plane_pairs c = 36 :=
by { sorry }

end orthogonal_pairs_in_cube_is_36_l45_45959


namespace max_abs_sum_eq_two_l45_45979

theorem max_abs_sum_eq_two (x y : ℝ) (h : x^2 + y^2 = 2) : |x| + |y| ≤ 2 :=
by
  sorry

end max_abs_sum_eq_two_l45_45979


namespace correctPairsAreSkating_l45_45480

def Friend := String
def Brother := String

structure SkatingPair where
  gentleman : Friend
  lady : Friend

-- Define the list of friends with their brothers
def friends : List Friend := ["Lyusya Egorova", "Olya Petrova", "Inna Krymova", "Anya Vorobyova"]
def brothers : List Brother := ["Andrey Egorov", "Serezha Petrov", "Dima Krymov", "Yura Vorobyov"]

-- Condition: The skating pairs such that gentlemen are taller than ladies and no one skates with their sibling
noncomputable def skatingPairs : List SkatingPair :=
  [ {gentleman := "Yura Vorobyov", lady := "Lyusya Egorova"},
    {gentleman := "Andrey Egorov", lady := "Olya Petrova"},
    {gentleman := "Serezha Petrov", lady := "Inna Krymova"},
    {gentleman := "Dima Krymov", lady := "Anya Vorobyova"} ]

-- Proving that the pairs are exactly as specified.
theorem correctPairsAreSkating :
  skatingPairs = 
    [ {gentleman := "Yura Vorobyov", lady := "Lyusya Egorova"},
      {gentleman := "Andrey Egorov", lady := "Olya Petrova"},
      {gentleman := "Serezha Petrov", lady := "Inna Krymova"},
      {gentleman := "Dima Krymov", lady := "Anya Vorobyova"} ] :=
by
  sorry

end correctPairsAreSkating_l45_45480


namespace balance_four_heartsuits_with_five_circles_l45_45119

variables (x y z : ℝ)

-- Given conditions
axiom condition1 : 4 * x + 3 * y = 12 * z
axiom condition2 : 2 * x = y + 3 * z

-- Statement to prove
theorem balance_four_heartsuits_with_five_circles : 4 * y = 5 * z :=
by sorry

end balance_four_heartsuits_with_five_circles_l45_45119


namespace find_m_l45_45793

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def point_on_x_axis_distance (x y : ℝ) : Prop :=
  y = 14

def point_distance_from_fixed_point (x y : ℝ) : Prop :=
  distance (x, y) (3, 8) = 8

def x_coordinate_condition (x : ℝ) : Prop :=
  x > 3

def m_distance (x y m : ℝ) : Prop :=
  distance (x, y) (0, 0) = m

theorem find_m (x y m : ℝ) 
  (h1 : point_on_x_axis_distance x y) 
  (h2 : point_distance_from_fixed_point x y) 
  (h3 : x_coordinate_condition x) :
  m_distance x y m → 
  m = Real.sqrt (233 + 12 * Real.sqrt 7) := by
  sorry

end find_m_l45_45793


namespace solve_system_l45_45600

noncomputable def solution1 (a b : ℝ) : ℝ × ℝ := 
  ((a + Real.sqrt (a^2 + 4 * b)) / 2, (-a + Real.sqrt (a^2 + 4 * b)) / 2)

noncomputable def solution2 (a b : ℝ) : ℝ × ℝ := 
  ((a - Real.sqrt (a^2 + 4 * b)) / 2, (-a - Real.sqrt (a^2 + 4 * b)) / 2)

theorem solve_system (a b x y : ℝ) : 
  (x - y = a ∧ x * y = b) ↔ ((x, y) = solution1 a b ∨ (x, y) = solution2 a b) := 
by sorry

end solve_system_l45_45600


namespace fraction_apple_juice_in_mixture_l45_45419

theorem fraction_apple_juice_in_mixture :
  let pitcher1_capacity := 800
  let pitcher2_capacity := 500
  let fraction_juice_pitcher1 := (1 : ℚ) / 4
  let fraction_juice_pitcher2 := (3 : ℚ) / 8
  let apple_juice_pitcher1 := pitcher1_capacity * fraction_juice_pitcher1
  let apple_juice_pitcher2 := pitcher2_capacity * fraction_juice_pitcher2
  let total_apple_juice := apple_juice_pitcher1 + apple_juice_pitcher2
  let total_capacity := pitcher1_capacity + pitcher2_capacity
  (total_apple_juice / total_capacity = 31 / 104) :=
by
  sorry

end fraction_apple_juice_in_mixture_l45_45419


namespace arithmetic_sequence_nth_term_l45_45228

theorem arithmetic_sequence_nth_term (a₁ : ℤ) (d : ℤ) (n : ℕ) :
  (a₁ = 11) →
  (d = -3) →
  (-49 = a₁ + (n - 1) * d) →
  (n = 21) :=
by 
  intros h₁ h₂ h₃
  sorry

end arithmetic_sequence_nth_term_l45_45228


namespace inequality_proof_l45_45313

variable (a b c d e f : Real)

theorem inequality_proof (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end inequality_proof_l45_45313


namespace movie_time_difference_l45_45994

theorem movie_time_difference
  (Nikki_movie : ℝ)
  (Michael_movie : ℝ)
  (Ryn_movie : ℝ)
  (Joyce_movie : ℝ)
  (total_hours : ℝ)
  (h1 : Nikki_movie = 30)
  (h2 : Michael_movie = Nikki_movie / 3)
  (h3 : Ryn_movie = (4 / 5) * Nikki_movie)
  (h4 : total_hours = 76)
  (h5 : total_hours = Michael_movie + Nikki_movie + Ryn_movie + Joyce_movie) :
  Joyce_movie - Michael_movie = 2 := 
by {
  sorry
}

end movie_time_difference_l45_45994


namespace toms_speed_l45_45996

/--
Karen places a bet with Tom that she will beat Tom in a car race by 4 miles 
even if Karen starts 4 minutes late. Assuming that Karen drives at 
an average speed of 60 mph and that Tom will drive 24 miles before 
Karen wins the bet. Prove that Tom's average driving speed is \( \frac{300}{7} \) mph.
--/
theorem toms_speed (
  (karen_speed : ℕ) (karen_lateness : ℚ) (karen_beats_tom_by : ℕ) 
  (karen_distance_when_tom_drives_24_miles : ℕ) 
  (karen_speed = 60) 
  (karen_lateness = 4 / 60) 
  (karen_beats_tom_by = 4) 
  (karen_distance_when_tom_drives_24_miles = 24)) : 
  ∃ tom_speed : ℚ, tom_speed = 300 / 7 :=
begin
  sorry
end

end toms_speed_l45_45996


namespace smallest_expression_value_l45_45297

theorem smallest_expression_value (a b c : ℝ) (h₁ : b > c) (h₂ : c > 0) (h₃ : a ≠ 0) :
  (2 * a + b) ^ 2 + (b - c) ^ 2 + (c - 2 * a) ^ 2 ≥ (4 / 3) * b ^ 2 :=
by
  sorry

end smallest_expression_value_l45_45297


namespace sugar_already_put_in_l45_45586

-- Define the conditions
def totalSugarRequired : Nat := 14
def sugarNeededToAdd : Nat := 12
def sugarAlreadyPutIn (total : Nat) (needed : Nat) : Nat := total - needed

--State the theorem
theorem sugar_already_put_in :
  sugarAlreadyPutIn totalSugarRequired sugarNeededToAdd = 2 := 
  by
    -- Providing 'sorry' as a placeholder for the actual proof
    sorry

end sugar_already_put_in_l45_45586


namespace first_term_of_arithmetic_series_l45_45300

theorem first_term_of_arithmetic_series 
  (a d : ℝ)
  (h1 : 20 * (2 * a + 39 * d) = 600)
  (h2 : 20 * (2 * a + 119 * d) = 1800) :
  a = 0.375 :=
by
  sorry

end first_term_of_arithmetic_series_l45_45300


namespace max_even_a_exists_max_even_a_l45_45702

theorem max_even_a (a : ℤ): (a^2 - 12 * a + 32 ≤ 0 ∧ ∃ k : ℤ, a = 2 * k) → a ≤ 8 := sorry

theorem exists_max_even_a : ∃ a : ℤ, (a^2 - 12 * a + 32 ≤ 0 ∧ ∃ k : ℤ, a = 2 * k ∧ a = 8) := sorry

end max_even_a_exists_max_even_a_l45_45702


namespace sum_remainder_div_9_l45_45780

theorem sum_remainder_div_9 : 
  let S := (20 / 2) * (1 + 20)
  S % 9 = 3 := 
by
  -- use let S to simplify the proof
  let S := (20 / 2) * (1 + 20)
  -- sum of first 20 natural numbers
  have H1 : S = 210 := by sorry
  -- division and remainder result
  have H2 : 210 % 9 = 3 := by sorry
  -- combine both results to conclude 
  exact H2

end sum_remainder_div_9_l45_45780


namespace lines_parallel_l45_45522

theorem lines_parallel 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (α : ℝ) (β : ℝ) (γ : ℝ)
  (h1 : Real.log (Real.sin α) + Real.log (Real.sin γ) = 2 * Real.log (Real.sin β)) :
  (∀ x y : ℝ, ∀ a b c : ℝ, 
    (x * (Real.sin α)^2 + y * Real.sin α = a) → 
    (x * (Real.sin β)^2 + y * Real.sin γ = c) →
    (-Real.sin α = -((Real.sin β)^2 / Real.sin γ))) :=
sorry

end lines_parallel_l45_45522


namespace set_B_can_form_right_angled_triangle_l45_45629

-- Definition and condition from the problem
def isRightAngledTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- The actual proof problem statement
theorem set_B_can_form_right_angled_triangle : isRightAngledTriangle 1 (Real.sqrt 3) 2 :=
sorry

end set_B_can_form_right_angled_triangle_l45_45629


namespace garden_area_l45_45605

theorem garden_area (w l A : ℕ) (h1 : w = 12) (h2 : l = 3 * w) (h3 : A = l * w) : A = 432 := by
  sorry

end garden_area_l45_45605


namespace largest_prime_factor_3136_l45_45102

theorem largest_prime_factor_3136 : ∀ (n : ℕ), n = 3136 → ∃ p : ℕ, Prime p ∧ (p ∣ n) ∧ ∀ q : ℕ, (Prime q ∧ q ∣ n) → p ≥ q :=
by {
  sorry
}

end largest_prime_factor_3136_l45_45102


namespace probability_exactly_one_instrument_l45_45175

-- Definitions of the conditions
def total_people : ℕ := 800
def frac_one_instrument : ℚ := 1 / 5
def people_two_or_more_instruments : ℕ := 64

-- Statement of the problem
theorem probability_exactly_one_instrument :
  let people_at_least_one_instrument := frac_one_instrument * total_people
  let people_exactly_one_instrument := people_at_least_one_instrument - people_two_or_more_instruments
  let probability := people_exactly_one_instrument / total_people
  probability = 3 / 25 :=
by
  -- Definitions
  let people_at_least_one_instrument : ℚ := frac_one_instrument * total_people
  let people_exactly_one_instrument : ℚ := people_at_least_one_instrument - people_two_or_more_instruments
  let probability : ℚ := people_exactly_one_instrument / total_people
  
  -- Sorry statement to skip the proof
  exact sorry

end probability_exactly_one_instrument_l45_45175


namespace probability_of_green_ball_l45_45812

theorem probability_of_green_ball :
  let P_X := 0.2
  let P_Y := 0.5
  let P_Z := 0.3
  let P_green_given_X := 5 / 10
  let P_green_given_Y := 3 / 10
  let P_green_given_Z := 8 / 10
  P_green_given_X * P_X + P_green_given_Y * P_Y + P_green_given_Z * P_Z = 0.49 :=
by {
  sorry
}

end probability_of_green_ball_l45_45812


namespace prove_b_minus_a_l45_45888

noncomputable def point := (ℝ × ℝ)

def rotate90 (p : point) (c : point) : point :=
  let (x, y) := p
  let (h, k) := c
  (h - (y - k), k + (x - h))

def reflect_y_eq_x (p : point) : point :=
  let (x, y) := p
  (y, x)

def transformed_point (a b : ℝ) : point :=
  reflect_y_eq_x (rotate90 (a, b) (2, 6))

theorem prove_b_minus_a (a b : ℝ) (h1 : transformed_point a b = (-7, 4)) : b - a = 15 :=
by
  sorry

end prove_b_minus_a_l45_45888


namespace find_k_l45_45528

variable (m n k : ℚ)

def line_eq (x y : ℚ) : Prop := x - (5/2 : ℚ) * y + 1 = 0

theorem find_k (h1 : line_eq m n) (h2 : line_eq (m + 1/2) (n + 1/k)) : k = 3/5 := by
  sorry

end find_k_l45_45528


namespace cost_of_one_bag_of_potatoes_l45_45413

theorem cost_of_one_bag_of_potatoes :
  let x := 250 in
  ∀ (price : ℕ)
    (bags : ℕ)
    (andrey_initial_price : ℕ)
    (andrey_sold_price : ℕ)
    (boris_initial_price : ℕ)
    (boris_first_price : ℕ)
    (boris_second_price : ℕ)
    (earnings_andrey : ℕ)
    (earnings_boris_first : ℕ)
    (earnings_boris_second : ℕ)
    (total_earnings_boris : ℕ),
  bags = 60 →
  andrey_initial_price = price →
  andrey_sold_price = 2 * price →
  andrey_sold_price * bags = earnings_andrey →
  boris_initial_price = price →
  boris_first_price = 1.6 * price →
  boris_second_price = 2.24 * price →
  boris_first_price * 15 + boris_second_price * 45 = total_earnings_boris →
  total_earnings_boris = earnings_andrey + 1200 →
  price = x :=
by
  intros x price bags andrey_initial_price andrey_sold_price boris_initial_price boris_first_price boris_second_price earnings_andrey earnings_boris_first earnings_boris_second total_earnings_boris
  assume h_bags h_andrey_initial_price h_andrey_sold_price h_earnings_andrey h_boris_initial_price h_boris_first_price h_boris_second_price h_total_earnings_boris h_total_earnings_difference
  if h_necessary : x = 250 then
    sorry
  else
    sorry


end cost_of_one_bag_of_potatoes_l45_45413


namespace child_tickets_sold_l45_45776

theorem child_tickets_sold (A C : ℕ) (h1 : A + C = 130) (h2 : 12 * A + 4 * C = 840) : C = 90 := by
  sorry

end child_tickets_sold_l45_45776


namespace radius_of_tangent_circle_l45_45458

theorem radius_of_tangent_circle (side_length : ℝ) (num_semicircles : ℕ)
  (r_s : ℝ) (r : ℝ)
  (h1 : side_length = 4)
  (h2 : num_semicircles = 16)
  (h3 : r_s = side_length / 4 / 2)
  (h4 : r = (9 : ℝ) / (2 * Real.sqrt 5)) :
  r = (9 * Real.sqrt 5) / 10 :=
by
  rw [h4]
  sorry

end radius_of_tangent_circle_l45_45458


namespace sin_vertex_angle_isosceles_triangle_l45_45089

theorem sin_vertex_angle_isosceles_triangle (α β : ℝ) (h_isosceles : β = 2 * α) (tan_base_angle : Real.tan α = 2 / 3) :
  Real.sin β = 12 / 13 := 
sorry

end sin_vertex_angle_isosceles_triangle_l45_45089


namespace average_eq_one_half_l45_45904

variable (w x y : ℝ)

-- Conditions
variables (h1 : 2 / w + 2 / x = 2 / y)
variables (h2 : w * x = y)

theorem average_eq_one_half : (w + x) / 2 = 1 / 2 :=
by
  sorry

end average_eq_one_half_l45_45904


namespace correct_remainder_l45_45126

-- Define the problem
def count_valid_tilings (n k : Nat) : Nat :=
  Nat.factorial (n + k) / (Nat.factorial n * Nat.factorial k) * (3 ^ (n + k) - 3 * 2 ^ (n + k) + 3)

noncomputable def tiles_mod_1000 : Nat :=
  let pairs := [(8, 0), (6, 1), (4, 2), (2, 3), (0, 4)]
  let M := pairs.foldl (λ acc (nk : Nat × Nat) => acc + count_valid_tilings nk.1 nk.2) 0
  M % 1000

theorem correct_remainder : tiles_mod_1000 = 328 :=
  by sorry

end correct_remainder_l45_45126


namespace tom_driving_speed_l45_45997

theorem tom_driving_speed
  (v : ℝ)
  (hKarenSpeed : 60 = 60) -- Karen drives at an average speed of 60 mph
  (hKarenLateStart: 4 / 60 = 1 / 15) -- Karen starts 4 minutes late, which is 1/15 hours
  (hTomDistance : 24 = 24) -- Tom drives 24 miles before Karen wins the bet
  (hTimeEquation: 24 / v = 8 / 15): -- The equation derived from given conditions
  v = 45 := 
by
  sorry

end tom_driving_speed_l45_45997


namespace solve_inequality_l45_45476

noncomputable def P (x : ℝ) : ℝ := x^2 - 3 * x + 2

theorem solve_inequality (x : ℝ) : (P x > 0) ↔ (x < 1 ∨ x > 2) := 
  sorry

end solve_inequality_l45_45476


namespace sqrt_180_eq_l45_45387

noncomputable def simplify_sqrt_180 : Real := 6 * Real.sqrt 5

theorem sqrt_180_eq : Real.sqrt 180 = simplify_sqrt_180 := 
by
  -- proof omitted
  sorry

end sqrt_180_eq_l45_45387


namespace geometric_progression_solution_l45_45295

theorem geometric_progression_solution 
  (b1 q : ℝ)
  (condition1 : (b1^2 / (1 + q + q^2) = 48 / 7))
  (condition2 : (b1^2 / (1 + q^2) = 144 / 17)) 
  : (b1 = 3 ∨ b1 = -3) ∧ q = 1 / 4 :=
by
  sorry

end geometric_progression_solution_l45_45295


namespace Tabitha_age_proof_l45_45951

variable (Tabitha_age current_hair_colors: ℕ)
variable (Adds_new_color_per_year: ℕ)
variable (initial_hair_colors: ℕ)
variable (years_passed: ℕ)

theorem Tabitha_age_proof (h1: Adds_new_color_per_year = 1)
                          (h2: initial_hair_colors = 2)
                          (h3: ∀ years_passed, Tabitha_age  = 15 + years_passed)
                          (h4: Adds_new_color_per_year  = 1 )
                          (h5: current_hair_colors =  8 - 3)
                          (h6: current_hair_colors  =  initial_hair_colors + 3)
                          : Tabitha_age = 18 := 
by {
  sorry  -- Proof omitted
}

end Tabitha_age_proof_l45_45951


namespace num_integers_with_factors_l45_45845

theorem num_integers_with_factors (a b lcm : ℕ) (lower upper : ℕ) (h_lcm : lcm = Nat.lcm a b) :
  (36 = Nat.lcm 12 9) → (a = 12) → (b = 9) → (lower = 200) → (upper = 500) →
  (finset.filter (λ x, x % lcm = 0) (finset.Icc lower upper)).card = 8 :=
by
  sorry

end num_integers_with_factors_l45_45845


namespace roots_can_be_integers_if_q_positive_roots_cannot_both_be_integers_if_q_negative_l45_45945

-- Part (a)
theorem roots_can_be_integers_if_q_positive (p q : ℤ) (hq : q > 0) :
  (∃ x y : ℤ, x * y = q ∧ x + y = p) ∧ (∃ x y : ℤ, x * y = q ∧ x + y = p + 1) :=
sorry

-- Part (b)
theorem roots_cannot_both_be_integers_if_q_negative (p q : ℤ) (hq : q < 0) :
  ¬(∃ x y z w : ℤ, x * y = q ∧ x + y = p ∧ z * w = q ∧ z + w = p + 1) :=
sorry

end roots_can_be_integers_if_q_positive_roots_cannot_both_be_integers_if_q_negative_l45_45945


namespace probability_sqrt_less_than_seven_l45_45107

-- Definitions and conditions from part a)
def is_two_digit_number (n : ℕ) : Prop := (10 ≤ n) ∧ (n ≤ 99)
def sqrt_less_than_seven (n : ℕ) : Prop := real.sqrt n < 7

-- Lean 4 statement for the actual proof problem
theorem probability_sqrt_less_than_seven : 
  (∃ n, is_two_digit_number n ∧ sqrt_less_than_seven n) → ∑ i in (finset.range 100).filter is_two_digit_number, if sqrt_less_than_seven i then 1 else 0 = 39 :=
sorry

end probability_sqrt_less_than_seven_l45_45107


namespace savings_on_cheapest_flight_l45_45668

theorem savings_on_cheapest_flight :
  let delta_price := 850
  let delta_discount := 0.20
  let united_price := 1100
  let united_discount := 0.30
  let delta_final_price := delta_price - delta_price * delta_discount
  let united_final_price := united_price - united_price * united_discount
  delta_final_price < united_final_price →
  united_final_price - delta_final_price = 90 :=
by
  sorry

end savings_on_cheapest_flight_l45_45668


namespace complex_exponentiation_l45_45744

open Complex

theorem complex_exponentiation :
  (Complex.div (1 + I) (1 - I))^2013 = I :=
by
  -- Here we take 'I' as complex number representing the imaginary unit 'i'.
  sorry

end complex_exponentiation_l45_45744


namespace seating_arrangements_l45_45349

open Nat

theorem seating_arrangements (n : ℕ) (h_n : n = 8) (alice : Fin n) (bob : Fin n) (h_alice : alice ≠ bob) :
  let total_arrangements := fact n,
      combined_arrangements := fact (n - 1) * 2,
      valid_arrangements := total_arrangements - combined_arrangements
  in valid_arrangements = 30240 := by
  sorry

end seating_arrangements_l45_45349


namespace ava_planted_9_trees_l45_45801

theorem ava_planted_9_trees
  (L : ℕ)
  (hAva : ∀ L, Ava = L + 3)
  (hTotal : L + (L + 3) = 15) : 
  Ava = 9 :=
by
  sorry

end ava_planted_9_trees_l45_45801


namespace compare_magnitudes_l45_45131

theorem compare_magnitudes : -0.5 > -0.75 :=
by
  have h1 : |(-0.5: ℝ)| = 0.5 := by norm_num
  have h2 : |(-0.75: ℝ)| = 0.75 := by norm_num
  have h3 : (0.5: ℝ) < 0.75 := by norm_num
  sorry

end compare_magnitudes_l45_45131


namespace theta_in_third_or_fourth_quadrant_l45_45497

-- Define the conditions as Lean definitions
def theta_condition (θ : ℝ) : Prop :=
  ∃ k : ℤ, θ = k * Real.pi + (-1 : ℝ)^(k + 1) * (Real.pi / 4)

-- Formulate the statement we need to prove
theorem theta_in_third_or_fourth_quadrant (θ : ℝ) (h : theta_condition θ) :
  ∃ q : ℤ, q = 3 ∨ q = 4 :=
sorry

end theta_in_third_or_fourth_quadrant_l45_45497


namespace narrow_black_stripes_count_l45_45556

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end narrow_black_stripes_count_l45_45556


namespace sum_of_youngest_and_oldest_friend_l45_45671

-- Given definitions
def mean_age_5 := 12
def median_age_5 := 11
def one_friend_age := 10

-- The total sum of ages is given by mean * number of friends
def total_sum_ages : ℕ := 5 * mean_age_5

-- Third friend's age as defined by median
def third_friend_age := 11

-- Proving the sum of the youngest and oldest friend's ages
theorem sum_of_youngest_and_oldest_friend:
  (∃ youngest oldest : ℕ, youngest + oldest = 38) :=
by
  sorry

end sum_of_youngest_and_oldest_friend_l45_45671


namespace hyperbola_k_range_l45_45028

theorem hyperbola_k_range (k : ℝ) : 
  (∀ x y : ℝ, (x^2 / (k + 2) - y^2 / (5 - k) = 1)) → (-2 < k ∧ k < 5) :=
by
  sorry

end hyperbola_k_range_l45_45028


namespace min_value_fraction_l45_45722

variable (x y : ℝ)

theorem min_value_fraction (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  ∃ (m : ℝ), (∀ z, (z = (1/x) + (9/y)) → z ≥ 16) ∧ ((1/x) + (9/y) = m) :=
sorry

end min_value_fraction_l45_45722


namespace sum_of_faces_l45_45229

theorem sum_of_faces (n_side_faces_per_prism : ℕ) (n_non_side_faces_per_prism : ℕ)
  (num_prisms : ℕ) (h1 : n_side_faces_per_prism = 3) (h2 : n_non_side_faces_per_prism = 2) 
  (h3 : num_prisms = 3) : 
  n_side_faces_per_prism * num_prisms + n_non_side_faces_per_prism * num_prisms = 15 :=
by
  sorry

end sum_of_faces_l45_45229


namespace Dawn_hourly_earnings_l45_45050

theorem Dawn_hourly_earnings :
  let t_per_painting := 2 
  let num_paintings := 12
  let total_earnings := 3600
  let total_time := t_per_painting * num_paintings
  let hourly_wage := total_earnings / total_time
  hourly_wage = 150 := by
  sorry

end Dawn_hourly_earnings_l45_45050


namespace sqrt_180_simplified_l45_45376

theorem sqrt_180_simplified : Real.sqrt 180 = 6 * Real.sqrt 5 :=
   sorry

end sqrt_180_simplified_l45_45376


namespace no_valid_n_values_l45_45302

theorem no_valid_n_values :
  ¬ ∃ n : ℕ, (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ (100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
by
  sorry

end no_valid_n_values_l45_45302


namespace full_price_ticket_revenue_l45_45442

theorem full_price_ticket_revenue (f d : ℕ) (p : ℝ) : 
  f + d = 200 → 
  f * p + d * (p / 3) = 3000 → 
  d = 200 - f → 
  (f * p) = 1500 := 
by
  intros h1 h2 h3
  sorry

end full_price_ticket_revenue_l45_45442


namespace annual_income_of_A_l45_45886

def monthly_income_ratios (A_income B_income : ℝ) : Prop := A_income / B_income = 5 / 2
def B_income_increase (B_income C_income : ℝ) : Prop := B_income = C_income + 0.12 * C_income

theorem annual_income_of_A (A_income B_income C_income : ℝ)
  (h1 : monthly_income_ratios A_income B_income)
  (h2 : B_income_increase B_income C_income)
  (h3 : C_income = 13000) :
  12 * A_income = 436800 :=
by 
  sorry

end annual_income_of_A_l45_45886


namespace right_angled_triangle_only_B_l45_45627

def forms_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_angled_triangle_only_B :
  forms_right_angled_triangle 1 (Real.sqrt 3) 2 ∧
  ¬forms_right_angled_triangle 1 2 2 ∧
  ¬forms_right_angled_triangle 4 5 6 ∧
  ¬forms_right_angled_triangle 1 1 (Real.sqrt 3) :=
by
  sorry

end right_angled_triangle_only_B_l45_45627


namespace cost_of_each_nose_spray_l45_45943

def total_nose_sprays : ℕ := 10
def total_cost : ℝ := 15
def buy_one_get_one_free : Bool := true

theorem cost_of_each_nose_spray :
  buy_one_get_one_free = true →
  total_nose_sprays = 10 →
  total_cost = 15 →
  (total_cost / (total_nose_sprays / 2)) = 3 :=
by
  intros h1 h2 h3
  sorry

end cost_of_each_nose_spray_l45_45943


namespace storks_initial_count_l45_45912

theorem storks_initial_count (S : ℕ) 
  (h1 : 6 = (S + 2) + 1) : S = 3 :=
sorry

end storks_initial_count_l45_45912


namespace square_of_binomial_l45_45426

theorem square_of_binomial (k : ℝ) : (∃ b : ℝ, (x^2 - 18 * x + k) = (x + b)^2) ↔ k = 81 :=
by
  sorry

end square_of_binomial_l45_45426


namespace tangent_line_at_b_l45_45606

theorem tangent_line_at_b (b : ℝ) : (∃ x : ℝ, (4*x^3 = 4) ∧ (4*x + b = x^4 - 1)) ↔ (b = -4) := 
by 
  sorry

end tangent_line_at_b_l45_45606


namespace number_of_narrow_black_stripes_l45_45580

-- Define the variables
variables {w n b : ℕ}

-- The conditions from the problem
def condition1 := w + n = b + 1
def condition2 := b = w + 7

-- The Lean statement to prove
theorem number_of_narrow_black_stripes (h1 : condition1) (h2 : condition2) : n = 8 :=
by {
  -- We import the conditions as hypotheses
  sorry
}

end number_of_narrow_black_stripes_l45_45580


namespace random_two_digit_sqrt_prob_lt_seven_l45_45105

theorem random_two_digit_sqrt_prob_lt_seven :
  let total_count := 90 in
  let count_lt_sqrt7 := 48 - 10 + 1 in
  (count_lt_sqrt7 : ℚ) / total_count = 13 / 30 :=
by
  let total_count := 90
  let count_lt_sqrt7 := 48 - 10 + 1
  have h1 : count_lt_sqrt7 = 39 := by linarith
  have h2 : (count_lt_sqrt7 : ℚ) / total_count = (39 : ℚ) / 90 := by rw h1
  have h3 : (39 : ℚ) / 90 = 13 / 30 := by norm_num
  rw [h2, h3]
  refl

end random_two_digit_sqrt_prob_lt_seven_l45_45105


namespace inequality_proof_l45_45314

variable (a b c d e f : Real)

theorem inequality_proof (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end inequality_proof_l45_45314


namespace find_k_l45_45183

theorem find_k (m n k : ℝ) (h1 : m = 2 * n + 5) (h2 : m + 3 = 2 * (n + k) + 5) : k = 3 / 2 := 
by 
  sorry

end find_k_l45_45183


namespace ways_to_make_change_l45_45335

theorem ways_to_make_change : ∃ ways : ℕ, ways = 60 ∧ (∀ (p n d q : ℕ), p + 5 * n + 10 * d + 25 * q = 55 → True) := 
by
  -- The proof will go here
  sorry

end ways_to_make_change_l45_45335


namespace expectation_variance_eta_l45_45307

noncomputable def expectation_of_eta : ℝ :=
  2

noncomputable def variance_of_eta : ℝ :=
  2.4

theorem expectation_variance_eta (η ξ : ℝ) (h1 : ξ ~ binomial 10 0.6) :
  (E η = expectation_of_eta) ∧ (Var η = variance_of_eta) :=
sorry

end expectation_variance_eta_l45_45307


namespace min_value_PF_PA_l45_45322

noncomputable def hyperbola_eq (x y : ℝ) := (x^2 / 4) - (y^2 / 12) = 1

noncomputable def focus_left : ℝ × ℝ := (-4, 0)
noncomputable def focus_right : ℝ × ℝ := (4, 0)
noncomputable def point_A : ℝ × ℝ := (1, 4)

theorem min_value_PF_PA (P : ℝ × ℝ)
  (hP : hyperbola_eq P.1 P.2)
  (hP_right_branch : P.1 > 0) :
  ∃ P : ℝ × ℝ, ∀ X : ℝ × ℝ, hyperbola_eq X.1 X.2 → X.1 > 0 → 
               (dist X focus_left + dist X point_A) ≥ 9 ∧
               (dist P focus_left + dist P point_A) = 9 := 
sorry

end min_value_PF_PA_l45_45322


namespace narrow_black_stripes_l45_45572

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end narrow_black_stripes_l45_45572


namespace number_of_zeros_of_f_is_3_l45_45767

def f (x : ℝ) : ℝ := x^3 - 64 * x

theorem number_of_zeros_of_f_is_3 : ∃ x1 x2 x3, (f x1 = 0) ∧ (f x2 = 0) ∧ (f x3 = 0) ∧ (x1 ≠ x2) ∧ (x2 ≠ x3) ∧ (x1 ≠ x3) :=
by
  sorry

end number_of_zeros_of_f_is_3_l45_45767


namespace narrow_black_stripes_l45_45570

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end narrow_black_stripes_l45_45570


namespace power_of_i_l45_45834

theorem power_of_i (i : ℂ) (h₀ : i^2 = -1) : i^(2016) = 1 :=
by {
  -- Proof will go here
  sorry
}

end power_of_i_l45_45834


namespace ratio_copper_zinc_l45_45218

theorem ratio_copper_zinc (total_mass zinc_mass : ℕ) (h1 : total_mass = 100) (h2 : zinc_mass = 35) : 
  ∃ (copper_mass : ℕ), 
    copper_mass = total_mass - zinc_mass ∧ (copper_mass / 5, zinc_mass / 5) = (13, 7) :=
by {
  sorry
}

end ratio_copper_zinc_l45_45218


namespace quadratic_roots_l45_45501

theorem quadratic_roots (r s : ℝ) (A : ℝ) (B : ℝ) (C : ℝ) (p q : ℝ) 
  (h1 : A = 3) (h2 : B = 4) (h3 : C = 5) 
  (h4 : r + s = -B / A) (h5 : rs = C / A) 
  (h6 : 4 * rs = q) :
  p = 56 / 9 :=
by 
  -- We assume the correct answer is given as we skip the proof details here.
  sorry

end quadratic_roots_l45_45501


namespace factor_difference_of_squares_l45_45285

theorem factor_difference_of_squares (x : ℝ) : (x^2 - 64 = (x - 8) * (x + 8)) := by
  -- Conditions to state the problem with
  let a := x
  let b := 8
  have h1 : (x^2 - 64) = (a^2 - b^2), by
    rw [a, b]
  have h2 : (a^2 - b^2) = (a - b) * (a + b), from sorry
  -- Final equivalence
  exact (h1.trans h2)
  sorry
  -- Final proof is left as sorry.

end factor_difference_of_squares_l45_45285


namespace pizzasServedDuringDinner_l45_45447

-- Definitions based on the conditions
def pizzasServedDuringLunch : ℕ := 9
def totalPizzasServedToday : ℕ := 15

-- Theorem statement
theorem pizzasServedDuringDinner : 
  totalPizzasServedToday - pizzasServedDuringLunch = 6 := 
  by 
    sorry

end pizzasServedDuringDinner_l45_45447


namespace manuscript_total_cost_l45_45663

theorem manuscript_total_cost
  (P R1 R2 R3 : ℕ)
  (RateFirst RateRevision : ℕ)
  (hP : P = 300)
  (hR1 : R1 = 55)
  (hR2 : R2 = 35)
  (hR3 : R3 = 25)
  (hRateFirst : RateFirst = 8)
  (hRateRevision : RateRevision = 6) :
  let RemainingPages := P - (R1 + R2 + R3)
  let CostNoRevisions := RemainingPages * RateFirst
  let CostOneRevision := R1 * (RateFirst + RateRevision)
  let CostTwoRevisions := R2 * (RateFirst + 2 * RateRevision)
  let CostThreeRevisions := R3 * (RateFirst + 3 * RateRevision)
  let TotalCost := CostNoRevisions + CostOneRevision + CostTwoRevisions + CostThreeRevisions
  TotalCost = 3600 :=
by
  sorry

end manuscript_total_cost_l45_45663


namespace no_positive_integer_solutions_l45_45708

theorem no_positive_integer_solutions (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : x^4 * y^4 - 14 * x^2 * y^2 + 49 ≠ 0 := 
by sorry

end no_positive_integer_solutions_l45_45708


namespace series_converges_to_l45_45132

noncomputable def series_sum := ∑' n : Nat, (4 * n + 3) / ((4 * n + 1) ^ 2 * (4 * n + 5) ^ 2)

theorem series_converges_to : series_sum = 1 / 200 := 
by 
  sorry

end series_converges_to_l45_45132


namespace W_3_7_eq_13_l45_45038

-- Define the operation W
def W (x y : ℤ) : ℤ := y + 5 * x - x^2

-- State the theorem
theorem W_3_7_eq_13 : W 3 7 = 13 := by
  sorry

end W_3_7_eq_13_l45_45038


namespace narrow_black_stripes_are_8_l45_45575

-- Define variables: w for wide black stripes, n for narrow black stripes, b for white stripes
variables (w n b : ℕ)

-- Given conditions
axiom cond1 : b = w + 7
axiom cond2 : w + n = b + 1

-- Theorem statement to prove that the number of narrow black stripes is 8
theorem narrow_black_stripes_are_8 : n = 8 :=
by sorry

end narrow_black_stripes_are_8_l45_45575


namespace right_angled_triangle_only_B_l45_45626

def forms_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_angled_triangle_only_B :
  forms_right_angled_triangle 1 (Real.sqrt 3) 2 ∧
  ¬forms_right_angled_triangle 1 2 2 ∧
  ¬forms_right_angled_triangle 4 5 6 ∧
  ¬forms_right_angled_triangle 1 1 (Real.sqrt 3) :=
by
  sorry

end right_angled_triangle_only_B_l45_45626


namespace trains_crossing_time_l45_45638

theorem trains_crossing_time
  (L : ℕ) (t1 t2 : ℕ)
  (h_length : L = 120)
  (h_t1 : t1 = 10)
  (h_t2 : t2 = 15) :
  let V1 := L / t1
  let V2 := L / t2
  let V_relative := V1 + V2
  let D := L + L
  (D / V_relative) = 12 :=
by
  sorry

end trains_crossing_time_l45_45638


namespace correct_option_is_B_l45_45631

-- Define the Pythagorean theorem condition for right-angled triangles
def is_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Conditions given in the problem
def option_A : Prop := ¬is_right_angled_triangle 1 2 2
def option_B : Prop := is_right_angled_triangle 1 (Real.sqrt 3) 2
def option_C : Prop := ¬is_right_angled_triangle 4 5 6
def option_D : Prop := ¬is_right_angled_triangle 1 1 (Real.sqrt 3)

-- The formal proof problem statement
theorem correct_option_is_B : option_A ∧ option_B ∧ option_C ∧ option_D :=
by
  sorry

end correct_option_is_B_l45_45631


namespace narrow_black_stripes_count_l45_45555

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end narrow_black_stripes_count_l45_45555


namespace simplify_expression_l45_45207

theorem simplify_expression (w x : ℝ) :
  3 * w + 6 * w + 9 * w + 12 * w + 15 * w - 2 * x - 4 * x - 6 * x - 8 * x - 10 * x + 24 = 
  45 * w - 30 * x + 24 :=
by sorry

end simplify_expression_l45_45207


namespace correct_statements_l45_45400

-- Define the statements
def statement_1 := true
def statement_2 := false
def statement_3 := true
def statement_4 := true

-- Define a function to count the number of true statements
def num_correct_statements (s1 s2 s3 s4 : Bool) : Nat :=
  [s1, s2, s3, s4].countP id

-- Define the theorem to prove that the number of correct statements is 3
theorem correct_statements :
  num_correct_statements statement_1 statement_2 statement_3 statement_4 = 3 :=
by
  -- You can use sorry to skip the proof
  sorry

end correct_statements_l45_45400


namespace series_sum_eq_l45_45699

noncomputable def series_sum : Real :=
  ∑' n : ℕ, (4 * (n + 1) + 1) / (((4 * (n + 1) - 1) ^ 3) * ((4 * (n + 1) + 3) ^ 3))

theorem series_sum_eq : series_sum = 1 / 5184 := sorry

end series_sum_eq_l45_45699


namespace arithmetic_progression_rth_term_l45_45146

open Nat

theorem arithmetic_progression_rth_term (n r : ℕ) (Sn : ℕ → ℕ) 
  (h : ∀ n, Sn n = 5 * n + 4 * n^2) : Sn r - Sn (r - 1) = 8 * r + 1 :=
by
  sorry

end arithmetic_progression_rth_term_l45_45146


namespace correct_product_l45_45523

theorem correct_product (a b : ℕ) (a' : ℕ) (h1 : a' = (a % 10) * 10 + (a / 10)) 
  (h2 : a' * b = 143) (h3 : 10 ≤ a ∧ a < 100):
  a * b = 341 :=
sorry

end correct_product_l45_45523


namespace wrestler_teams_possible_l45_45233

theorem wrestler_teams_possible :
  ∃ (team1 team2 team3 : Finset ℕ),
  (team1 ∪ team2 ∪ team3 = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (team1 ∩ team2 = ∅) ∧ (team1 ∩ team3 = ∅) ∧ (team2 ∩ team3 = ∅) ∧
  (team1.card = 3) ∧ (team2.card = 3) ∧ (team3.card = 3) ∧
  (team1.sum id = 15) ∧ (team2.sum id = 15) ∧ (team3.sum id = 15) ∧
  (∀ x ∈ team1, ∀ y ∈ team2, x > y) ∧
  (∀ x ∈ team2, ∀ y ∈ team3, x > y) ∧
  (∀ x ∈ team3, ∀ y ∈ team1, x > y) := sorry

end wrestler_teams_possible_l45_45233


namespace narrow_black_stripes_l45_45543

variable (w n b : ℕ)

theorem narrow_black_stripes (w : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := by
  have h3 : w + n = (w + 7) + 1 := by rw [h1]
  have h4 : w + n = w + 8 := by linarith
  have h5 : n = 8 := by linarith
  exact h5

end narrow_black_stripes_l45_45543


namespace convert_13_to_binary_l45_45439

def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (n : Nat) (acc : List Nat) : List Nat :=
      if n = 0 then acc
      else aux (n / 2) ((n % 2) :: acc)
    aux n []

theorem convert_13_to_binary : decimal_to_binary 13 = [1, 1, 0, 1] :=
  by
    sorry -- Proof to be provided

end convert_13_to_binary_l45_45439


namespace fifth_selected_ID_is_01_l45_45889

noncomputable def populationIDs : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

noncomputable def randomNumberTable : List (List ℕ) :=
  [[78, 16, 65, 72,  8, 2, 63, 14,  7, 2, 43, 69, 97, 28,  1, 98],
   [32,  4, 92, 34, 49, 35, 82,  0, 36, 23, 48, 69, 69, 38, 74, 81]]

noncomputable def selectedIDs (table : List (List ℕ)) : List ℕ :=
  [8, 2, 14, 7, 1]  -- Derived from the selection method

theorem fifth_selected_ID_is_01 : (selectedIDs randomNumberTable).get! 4 = 1 := by
  sorry

end fifth_selected_ID_is_01_l45_45889


namespace total_revenue_correct_l45_45661

def small_slices_price := 150
def large_slices_price := 250
def total_slices_sold := 5000
def small_slices_sold := 2000

def large_slices_sold := total_slices_sold - small_slices_sold

def revenue_from_small_slices := small_slices_sold * small_slices_price
def revenue_from_large_slices := large_slices_sold * large_slices_price
def total_revenue := revenue_from_small_slices + revenue_from_large_slices

theorem total_revenue_correct : total_revenue = 1050000 := by
  sorry

end total_revenue_correct_l45_45661


namespace mod_remainder_l45_45243

theorem mod_remainder (a b c d : ℕ) (h1 : a = 11) (h2 : b = 9) (h3 : c = 7) (h4 : d = 7) :
  (a^d + b^(d + 1) + c^(d + 2)) % d = 1 := 
by 
  sorry

end mod_remainder_l45_45243


namespace age_difference_l45_45923

theorem age_difference (A B C : ℕ) (h1 : B = 10) (h2 : B = 2 * C) (h3 : A + B + C = 27) : A - B = 2 :=
 by
  sorry

end age_difference_l45_45923


namespace cristobal_read_more_pages_l45_45701

theorem cristobal_read_more_pages (B : ℕ) (hB : B = 704) : 
  let C := 15 + 3 * B in
  C - B = 1423 :=
by
  let C := 15 + 3 * B
  have hC : C = 2127, by
    sorry
  have hDiff : C - B = 1423, by
    sorry
  exact hDiff

end cristobal_read_more_pages_l45_45701


namespace binom_mult_l45_45684

open Nat

theorem binom_mult : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binom_mult_l45_45684


namespace least_value_expression_l45_45778

open Real

theorem least_value_expression (x : ℝ) : 
  let expr := (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2023 + 3 * cos (2 * x)
  ∃ a : ℝ, expr = a ∧ ∀ b : ℝ, b < a → False :=
sorry

end least_value_expression_l45_45778


namespace greatest_possible_price_per_notebook_l45_45136

theorem greatest_possible_price_per_notebook (budget entrance_fee : ℝ) (notebooks : ℕ) (tax_rate : ℝ) (price_per_notebook : ℝ) :
  budget = 160 ∧ entrance_fee = 5 ∧ notebooks = 18 ∧ tax_rate = 0.05 ∧ price_per_notebook * notebooks * (1 + tax_rate) ≤ (budget - entrance_fee) →
  price_per_notebook = 8 :=
by
  sorry

end greatest_possible_price_per_notebook_l45_45136


namespace rice_containers_l45_45371

theorem rice_containers (total_weight_pounds : ℚ) (weight_per_container_ounces : ℚ) (pound_to_ounces : ℚ) : 
  total_weight_pounds = 29/4 → 
  weight_per_container_ounces = 29 → 
  pound_to_ounces = 16 → 
  (total_weight_pounds * pound_to_ounces) / weight_per_container_ounces = 4 := 
by
  intros h1 h2 h3 
  rw [h1, h2, h3]
  sorry

end rice_containers_l45_45371


namespace radius_of_circumscribed_sphere_l45_45611

noncomputable def circumscribed_sphere_radius (a : ℝ) : ℝ :=
  a / Real.sqrt 3

theorem radius_of_circumscribed_sphere 
  (a : ℝ) 
  (h_base_side : 0 < a)
  (h_distance : ∃ d : ℝ, d = a * Real.sqrt 2 / 8) : 
  circumscribed_sphere_radius a = a / Real.sqrt 3 :=
sorry

end radius_of_circumscribed_sphere_l45_45611


namespace intersection_of_P_with_complement_Q_l45_45326

-- Define the universal set U, and sets P and Q
def U : List ℕ := [1, 2, 3, 4]
def P : List ℕ := [1, 2]
def Q : List ℕ := [2, 3]

-- Define the complement of Q with respect to U
def complement (U Q : List ℕ) : List ℕ := U.filter (λ x => x ∉ Q)

-- Define the intersection of two sets
def intersection (A B : List ℕ) : List ℕ := A.filter (λ x => x ∈ B)

-- The proof statement we need to show
theorem intersection_of_P_with_complement_Q : intersection P (complement U Q) = [1] := by
  sorry

end intersection_of_P_with_complement_Q_l45_45326


namespace equation_represents_pair_of_lines_l45_45459

theorem equation_represents_pair_of_lines : ∀ x y : ℝ, 9 * x^2 - 25 * y^2 = 0 → 
                    (x = (5/3) * y ∨ x = -(5/3) * y) :=
by sorry

end equation_represents_pair_of_lines_l45_45459


namespace total_eggs_needed_l45_45329

-- Define the conditions
def eggsFromAndrew : ℕ := 155
def eggsToBuy : ℕ := 67

-- Define the total number of eggs
def totalEggs : ℕ := eggsFromAndrew + eggsToBuy

-- The theorem to be proven
theorem total_eggs_needed : totalEggs = 222 := by
  sorry

end total_eggs_needed_l45_45329


namespace tangent_line_at_M_l45_45703

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x - 6)

theorem tangent_line_at_M :
  let M : ℝ × ℝ := (2, 0)
  ∃ (m n : ℝ), n = f m ∧ m = 4 ∧ n = -2 * Real.exp 4 ∧
    ∀ (x y : ℝ), y = -Real.exp 4 * (x - 2) →
    M.2 = y :=
by
  sorry

end tangent_line_at_M_l45_45703


namespace cd_player_percentage_l45_45613

-- Define the percentage variables
def powerWindowsAndAntiLock : ℝ := 0.10
def antiLockAndCdPlayer : ℝ := 0.15
def powerWindowsAndCdPlayer : ℝ := 0.22
def cdPlayerAlone : ℝ := 0.38

-- Define the problem statement
theorem cd_player_percentage : 
  powerWindowsAndAntiLock = 0.10 → 
  antiLockAndCdPlayer = 0.15 → 
  powerWindowsAndCdPlayer = 0.22 → 
  cdPlayerAlone = 0.38 → 
  (antiLockAndCdPlayer + powerWindowsAndCdPlayer + cdPlayerAlone) = 0.75 :=
by
  intros
  sorry

end cd_player_percentage_l45_45613


namespace Mary_put_crayons_l45_45231

def initial_crayons : ℕ := 7
def final_crayons : ℕ := 10
def added_crayons (i f : ℕ) : ℕ := f - i

theorem Mary_put_crayons :
  added_crayons initial_crayons final_crayons = 3 := 
by
  sorry

end Mary_put_crayons_l45_45231


namespace rearrange_distinct_sums_mod_4028_l45_45538

theorem rearrange_distinct_sums_mod_4028 
  (x : Fin 2014 → ℤ) (y : Fin 2014 → ℤ) 
  (hx : ∀ i j : Fin 2014, i ≠ j → x i % 2014 ≠ x j % 2014)
  (hy : ∀ i j : Fin 2014, i ≠ j → y i % 2014 ≠ y j % 2014) :
  ∃ σ : Fin 2014 → Fin 2014, Function.Bijective σ ∧ 
  ∀ i j : Fin 2014, i ≠ j → ( x i + y (σ i) ) % 4028 ≠ ( x j + y (σ j) ) % 4028 
:= by
  sorry

end rearrange_distinct_sums_mod_4028_l45_45538


namespace A_plus_B_l45_45773

theorem A_plus_B {A B : ℚ} (h : ∀ x : ℚ, (Bx - 19) / (x^2 - 8*x + 15) = A / (x - 3) + 5 / (x - 5)) : 
  A + B = 33 / 5 := sorry

end A_plus_B_l45_45773


namespace students_count_l45_45111

theorem students_count :
  ∀ (sets marbles_per_set marbles_per_student total_students : ℕ),
    sets = 3 →
    marbles_per_set = 32 →
    marbles_per_student = 4 →
    total_students = (sets * marbles_per_set) / marbles_per_student →
    total_students = 24 :=
by
  intros sets marbles_per_set marbles_per_student total_students
  intros h_sets h_marbles_per_set h_marbles_per_student h_total_students
  rw [h_sets, h_marbles_per_set, h_marbles_per_student] at h_total_students
  exact h_total_students

end students_count_l45_45111


namespace EG_perpendicular_to_AC_l45_45601

noncomputable def rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 < B.1 ∧ A.2 = B.2 ∧ B.1 < C.1 ∧ B.2 < C.2 ∧ C.1 = D.1 ∧ C.2 > D.2 ∧ D.1 > A.1 ∧ D.2 = A.2

theorem EG_perpendicular_to_AC
  {A B C D E F G: ℝ × ℝ}
  (h1: rectangle A B C D)
  (h2: E = (B.1, C.2) ∨ E = (C.1, B.2)) -- Assuming E lies on BC or BA
  (h3: F = (B.1, A.2) ∨ F = (A.1, B.2)) -- Assuming F lies on BA or BC
  (h4: G = (C.1, D.2) ∨ G = (D.1, C.2)) -- Assuming G lies on CD
  (h5: (F.1, G.2) = (A.1, C.2)) -- Line through F parallel to AC meets CD at G
: ∃ (H : ℝ × ℝ → ℝ × ℝ → ℝ), H E G = 0 := sorry

end EG_perpendicular_to_AC_l45_45601


namespace narrow_black_stripes_l45_45548

variable (w n b : ℕ)

theorem narrow_black_stripes (w : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := by
  have h3 : w + n = (w + 7) + 1 := by rw [h1]
  have h4 : w + n = w + 8 := by linarith
  have h5 : n = 8 := by linarith
  exact h5

end narrow_black_stripes_l45_45548


namespace binom_mult_eq_6720_l45_45695

theorem binom_mult_eq_6720 :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_mult_eq_6720_l45_45695


namespace tangent_line_through_P_tangent_line_through_Q1_tangent_line_through_Q2_l45_45725

open Real

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 9

noncomputable def tangent_line_p (x y : ℝ) : Prop :=
  2 * x - sqrt 5 * y - 9 = 0

noncomputable def line_q1 (x y : ℝ) : Prop :=
  x = 3

noncomputable def line_q2 (x y : ℝ) : Prop :=
  8 * x - 15 * y + 51 = 0

theorem tangent_line_through_P :
  ∀ (x y : ℝ),
    circle_eq x y →
    (x, y) = (2, -sqrt 5) →
    tangent_line_p x y := 
sorry

theorem tangent_line_through_Q1 :
  ∀ (x y : ℝ),
    circle_eq x y →
    (x, y) = (3, 5) →
    line_q1 x y := 
sorry

theorem tangent_line_through_Q2 :
  ∀ (x y : ℝ),
    circle_eq x y →
    (x, y) = (3, 5) →
    line_q2 x y := 
sorry

end tangent_line_through_P_tangent_line_through_Q1_tangent_line_through_Q2_l45_45725


namespace three_solutions_exists_l45_45756

theorem three_solutions_exists (n : ℕ) (h_pos : 0 < n) (h_sol : ∃ x y : ℤ, x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ x1 y1 x2 y2 x3 y3 : ℤ, (x1^3 - 3 * x1 * y1^2 + y1^3 = n) ∧ (x2^3 - 3 * x2 * y2^2 + y2^3 = n) ∧ (x3^3 - 3 * x3 * y3^2 + y3^3 = n) ∧ (x1, y1) ≠ (x2, y2) ∧ (x2, y2) ≠ (x3, y3) ∧ (x1, y1) ≠ (x3, y3) :=
by
  sorry

end three_solutions_exists_l45_45756


namespace father_l45_45921

-- Let s be the circumference of the circular rink.
-- Let x be the son's speed.
-- Let k be the factor by which the father's speed is greater than the son's speed.

-- Define a theorem to state that k = 3/2.
theorem father's_speed_is_3_over_2_times_son's_speed
  (s x : ℝ) (k : ℝ) (h : s / (k * x - x) = (s / (k * x + x)) * 5) :
  k = 3 / 2 :=
by {
  sorry
}

end father_l45_45921


namespace semicircle_problem_l45_45526

open Real

theorem semicircle_problem (r : ℝ) (N : ℕ)
  (h1 : True) -- condition 1: There are N small semicircles each with radius r.
  (h2 : True) -- condition 2: The diameter of the large semicircle is 2Nr.
  (h3 : (N * (π * r^2) / 2) / ((π * (N^2 * r^2) / 2) - (N * (π * r^2) / 2)) = (1 : ℝ) / 12) -- given ratio A / B = 1 / 12 
  : N = 13 :=
sorry

end semicircle_problem_l45_45526


namespace find_A_l45_45760

-- Define the polynomial and the partial fraction decomposition equation
def polynomial (x : ℝ) : ℝ := x^3 - 3 * x^2 - 13 * x + 15

theorem find_A (A B C : ℝ) (h : ∀ x : ℝ, 1 / polynomial x = A / (x + 3) + B / (x - 1) + C / (x - 1)^2) : 
  A = 1 / 16 :=
sorry

end find_A_l45_45760


namespace quadratic_real_roots_iff_l45_45256

theorem quadratic_real_roots_iff (α : ℝ) : (∃ x : ℝ, x^2 - 2 * x + α = 0) ↔ α ≤ 1 :=
by
  sorry

end quadratic_real_roots_iff_l45_45256


namespace same_terminal_side_l45_45125

theorem same_terminal_side
  (k : ℤ)
  (angle1 := (π / 5))
  (angle2 := (21 * π / 5)) :
  ∃ k : ℤ, angle2 = 2 * k * π + angle1 := by
  sorry

end same_terminal_side_l45_45125


namespace interest_rate_per_annum_l45_45653

-- Definitions for the given conditions
def SI : ℝ := 4016.25
def P : ℝ := 44625
def T : ℝ := 9

-- The interest rate R must be 1 according to the conditions
theorem interest_rate_per_annum : (SI * 100) / (P * T) = 1 := by
  sorry

end interest_rate_per_annum_l45_45653


namespace cristobal_read_more_pages_l45_45700

-- Defining the given conditions
def pages_beatrix_read : ℕ := 704
def pages_cristobal_read (b : ℕ) : ℕ := 3 * b + 15

-- Stating the problem
theorem cristobal_read_more_pages (b : ℕ) (c : ℕ) (h : b = pages_beatrix_read) (h_c : c = pages_cristobal_read b) :
  (c - b) = 1423 :=
by
  sorry

end cristobal_read_more_pages_l45_45700


namespace parking_space_area_l45_45122

theorem parking_space_area
  (L : ℕ) (W : ℕ)
  (hL : L = 9)
  (hSum : 2 * W + L = 37) : L * W = 126 := 
by
  sorry

end parking_space_area_l45_45122


namespace frosting_cupcakes_in_10_minutes_l45_45803

def speed_Cagney := 1 / 20 -- Cagney frosts 1 cupcake every 20 seconds
def speed_Lacey := 1 / 30 -- Lacey frosts 1 cupcake every 30 seconds
def speed_Jamie := 1 / 15 -- Jamie frosts 1 cupcake every 15 seconds

def combined_speed := speed_Cagney + speed_Lacey + speed_Jamie -- Combined frosting rate (cupcakes per second)

def total_seconds := 10 * 60 -- 10 minutes converted to seconds

def number_of_cupcakes := combined_speed * total_seconds -- Total number of cupcakes frosted in 10 minutes

theorem frosting_cupcakes_in_10_minutes :
  number_of_cupcakes = 90 := by
  sorry

end frosting_cupcakes_in_10_minutes_l45_45803


namespace ratio_of_black_to_blue_l45_45872

universe u

-- Define the types of black and red pens
variables (B R : ℕ)

-- Define the conditions
def condition1 : Prop := 2 + B + R = 12
def condition2 : Prop := R = 2 * B - 2

-- Define the proof statement
theorem ratio_of_black_to_blue (h1 : condition1 B R) (h2 : condition2 B R) : B / 2 = 1 :=
by
  sorry

end ratio_of_black_to_blue_l45_45872


namespace sqrt_180_simplified_l45_45377

theorem sqrt_180_simplified : Real.sqrt 180 = 6 * Real.sqrt 5 :=
   sorry

end sqrt_180_simplified_l45_45377


namespace correct_option_B_l45_45622

variable {a b x y : ℤ}

def option_A (a : ℤ) : Prop := -a - a = 0
def option_B (x y : ℤ) : Prop := -(x + y) = -x - y
def option_C (b a : ℤ) : Prop := 3 * (b - 2 * a) = 3 * b - 2 * a
def option_D (a : ℤ) : Prop := 8 * a^4 - 6 * a^2 = 2 * a^2

theorem correct_option_B (x y : ℤ) : option_B x y := by
  -- The proof would go here
  sorry

end correct_option_B_l45_45622


namespace james_older_brother_age_l45_45990

def johnAge : ℕ := 39

def ageCondition (johnAge : ℕ) (jamesAgeIn6 : ℕ) : Prop :=
  johnAge - 3 = 2 * jamesAgeIn6

def jamesOlderBrother (james : ℕ) : ℕ :=
  james + 4

theorem james_older_brother_age (johnAge jamesOlderBrotherAge : ℕ) (james : ℕ) :
  johnAge = 39 →
  (johnAge - 3 = 2 * (james + 6)) →
  jamesOlderBrotherAge = jamesOlderBrother james →
  jamesOlderBrotherAge = 16 :=
by
  sorry

end james_older_brother_age_l45_45990


namespace pool_capacity_percentage_l45_45761

theorem pool_capacity_percentage
  (rate : ℕ := 60) -- cubic feet per minute
  (time : ℕ := 800) -- minutes
  (width : ℕ := 60) -- feet
  (length : ℕ := 100) -- feet
  (depth : ℕ := 10) -- feet
  : (rate * time * 100) / (width * length * depth) = 8 := by
{
  sorry
}

end pool_capacity_percentage_l45_45761


namespace fraction_of_x_l45_45981

theorem fraction_of_x (w x y f : ℝ) (h1 : 2 / w + f * x = 2 / y) (h2 : w * x = y) (h3 : (w + x) / 2 = 0.5) : f = 2 / x - 2 := 
sorry

end fraction_of_x_l45_45981


namespace ratio_of_tagged_fish_is_1_over_25_l45_45987

-- Define the conditions
def T70 : ℕ := 70  -- Number of tagged fish first caught and tagged
def T50 : ℕ := 50  -- Total number of fish caught in the second sample
def t2 : ℕ := 2    -- Number of tagged fish in the second sample

-- State the theorem/question
theorem ratio_of_tagged_fish_is_1_over_25 : (t2 / T50) = 1 / 25 :=
by
  sorry

end ratio_of_tagged_fish_is_1_over_25_l45_45987


namespace share_a_is_240_l45_45786

def total_profit : ℕ := 630

def initial_investment_a : ℕ := 3000
def initial_investment_b : ℕ := 4000

def months_a1 : ℕ := 8
def months_a2 : ℕ := 4
def investment_a1 : ℕ := initial_investment_a * months_a1
def investment_a2 : ℕ := (initial_investment_a - 1000) * months_a2
def total_investment_a : ℕ := investment_a1 + investment_a2

def months_b1 : ℕ := 8
def months_b2 : ℕ := 4
def investment_b1 : ℕ := initial_investment_b * months_b1
def investment_b2 : ℕ := (initial_investment_b + 1000) * months_b2
def total_investment_b : ℕ := investment_b1 + investment_b2

def ratio_a : ℕ := 8
def ratio_b : ℕ := 13
def total_ratio : ℕ := ratio_a + ratio_b

noncomputable def share_a (total_profit : ℕ) (ratio_a ratio_total : ℕ) : ℕ :=
  (ratio_a * total_profit) / ratio_total

theorem share_a_is_240 :
  share_a total_profit ratio_a total_ratio = 240 :=
by
  sorry

end share_a_is_240_l45_45786


namespace probability_is_correct_l45_45752

/-- Ms. Carr's reading list contains 12 books, each student chooses 6 books.
What is the probability that there are exactly 3 books that Harold 
and Betty both select? -/
def probability_exactly_3_shared_books : ℚ :=
  let total_ways := (Nat.choose 12 6) * (Nat.choose 12 6)
  let successful_ways := (Nat.choose 12 3) * (Nat.choose 9 3) * (Nat.choose 9 3)
  successful_ways / total_ways

/-- Proof that the probability is exactly 405/2223 when both select 6 books -/
theorem probability_is_correct :
  probability_exactly_3_shared_books = 405 / 2223 :=
by
  sorry

end probability_is_correct_l45_45752


namespace factor_difference_of_squares_l45_45286

theorem factor_difference_of_squares (x : ℝ) : (x^2 - 64 = (x - 8) * (x + 8)) := by
  -- Conditions to state the problem with
  let a := x
  let b := 8
  have h1 : (x^2 - 64) = (a^2 - b^2), by
    rw [a, b]
  have h2 : (a^2 - b^2) = (a - b) * (a + b), from sorry
  -- Final equivalence
  exact (h1.trans h2)
  sorry
  -- Final proof is left as sorry.

end factor_difference_of_squares_l45_45286


namespace area_of_rectangle_R_l45_45790

-- Define the side lengths of the squares and rectangles involved
def larger_square_side := 4
def smaller_square_side := 2
def rectangle_side1 := 1
def rectangle_side2 := 4

-- The areas of these shapes
def area_larger_square := larger_square_side * larger_square_side
def area_smaller_square := smaller_square_side * smaller_square_side
def area_first_rectangle := rectangle_side1 * rectangle_side2

-- Define the sum of all possible values for the area of rectangle R
def area_remaining := area_larger_square - (area_smaller_square + area_first_rectangle)

theorem area_of_rectangle_R : area_remaining = 8 := sorry

end area_of_rectangle_R_l45_45790


namespace common_tangent_line_range_a_l45_45515

open Real

theorem common_tangent_line_range_a (a : ℝ) (h_pos : 0 < a) :
  (∃ x₁ x₂ : ℝ, 2 * a * x₁ = exp x₂ ∧ (exp x₂ - a * x₁^2) / (x₂ - x₁) = 2 * a * x₁) →
  a ≥ exp 2 / 4 := 
sorry

end common_tangent_line_range_a_l45_45515


namespace fish_left_in_tank_l45_45513

-- Define the initial number of fish and the number of fish moved
def initialFish : Real := 212.0
def movedFish : Real := 68.0

-- Define the number of fish left in the tank
def fishLeft (initialFish : Real) (movedFish : Real) : Real := initialFish - movedFish

-- Theorem stating the problem
theorem fish_left_in_tank : fishLeft initialFish movedFish = 144.0 := by
  sorry

end fish_left_in_tank_l45_45513


namespace simplify_expr_l45_45080

theorem simplify_expr (x : ℝ) :
  2 * x^2 * (4 * x^3 - 3 * x + 5) - 4 * (x^3 - x^2 + 3 * x - 8) =
    8 * x^5 - 10 * x^3 + 14 * x^2 - 12 * x + 32 :=
by
  sorry

end simplify_expr_l45_45080


namespace time_to_write_all_rearrangements_l45_45877

-- Define the problem conditions
def sophie_name_length := 6
def rearrangements_per_minute := 18

-- Define the factorial function for calculating permutations
noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the total number of rearrangements of Sophie's name
noncomputable def total_rearrangements := factorial sophie_name_length

-- Define the time in minutes to write all rearrangements
noncomputable def time_in_minutes := total_rearrangements / rearrangements_per_minute

-- Convert the time to hours
noncomputable def minutes_to_hours (minutes : ℕ) : ℚ := minutes / 60

-- Prove the time in hours to write all the rearrangements
theorem time_to_write_all_rearrangements : minutes_to_hours time_in_minutes = (2 : ℚ) / 3 := 
  sorry

end time_to_write_all_rearrangements_l45_45877


namespace seating_arrangements_l45_45348

open Nat

theorem seating_arrangements (total_people : ℕ) (alice : ℕ) (bob : ℕ) (h_total : total_people = 8) (h_alice_bob : alice ≠ bob) :
  let total_arrangements := factorial total_people,
      alice_bob_together_arrangements := factorial 7 * factorial 2,
      arrangements_with_condition := total_arrangements - alice_bob_together_arrangements
  in arrangements_with_condition = 30240 :=
by 
  rw [h_total]
  sorry

end seating_arrangements_l45_45348


namespace total_cost_4kg_mangos_3kg_rice_5kg_flour_l45_45255

def cost_per_kg_mangos (M : ℝ) (R : ℝ) := (10 * M = 24 * R)
def cost_per_kg_flour_equals_rice (F : ℝ) (R : ℝ) := (6 * F = 2 * R)
def cost_of_flour (F : ℝ) := (F = 24)

theorem total_cost_4kg_mangos_3kg_rice_5kg_flour 
  (M R F : ℝ) 
  (h1 : cost_per_kg_mangos M R) 
  (h2 : cost_per_kg_flour_equals_rice F R) 
  (h3 : cost_of_flour F) : 
  4 * M + 3 * R + 5 * F = 1027.2 :=
by {
  sorry
}

end total_cost_4kg_mangos_3kg_rice_5kg_flour_l45_45255


namespace negation_of_not_both_are_not_even_l45_45084

variables {a b : ℕ}

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem negation_of_not_both_are_not_even :
  ¬ (¬ is_even a ∧ ¬ is_even b) ↔ (is_even a ∨ is_even b) :=
by
  sorry

end negation_of_not_both_are_not_even_l45_45084


namespace fewer_green_pens_than_pink_l45_45236

-- Define the variables
variables (G B : ℕ)

-- State the conditions
axiom condition1 : G < 12
axiom condition2 : B = G + 3
axiom condition3 : 12 + G + B = 21

-- Define the problem statement
theorem fewer_green_pens_than_pink : 12 - G = 9 :=
by
  -- Insert the proof steps here
  sorry

end fewer_green_pens_than_pink_l45_45236


namespace simplify_expression_correct_l45_45372

def simplify_expression (i : ℂ) (h : i ^ 2 = -1) : ℂ :=
  3 * (4 - 2 * i) + 2 * i * (3 - i)

theorem simplify_expression_correct (i : ℂ) (h : i ^ 2 = -1) : simplify_expression i h = 14 := 
by
  sorry

end simplify_expression_correct_l45_45372


namespace product_of_sums_of_squares_l45_45875

theorem product_of_sums_of_squares (a b : ℤ) 
  (h1 : ∃ x1 y1 : ℤ, a = x1^2 + y1^2)
  (h2 : ∃ x2 y2 : ℤ, b = x2^2 + y2^2) : 
  ∃ x y : ℤ, a * b = x^2 + y^2 :=
by
  sorry

end product_of_sums_of_squares_l45_45875


namespace fuel_tank_capacity_l45_45938

def ethanol_content_fuel_A (fuel_A : ℝ) : ℝ := 0.12 * fuel_A
def ethanol_content_fuel_B (fuel_B : ℝ) : ℝ := 0.16 * fuel_B

theorem fuel_tank_capacity (C : ℝ) :
  ethanol_content_fuel_A 122 + ethanol_content_fuel_B (C - 122) = 30 → C = 218 :=
by
  sorry

end fuel_tank_capacity_l45_45938


namespace percent_absent_math_dept_l45_45127

theorem percent_absent_math_dept (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (male_absent_fraction : ℚ) (female_absent_fraction : ℚ)
  (h1 : total_students = 160) 
  (h2 : male_students = 90) 
  (h3 : female_students = 70) 
  (h4 : male_absent_fraction = 1 / 5) 
  (h5 : female_absent_fraction = 2 / 7) :
  ((male_absent_fraction * male_students + female_absent_fraction * female_students) / total_students) * 100 = 23.75 :=
by
  sorry

end percent_absent_math_dept_l45_45127


namespace scientific_notation_of_935000000_l45_45657

theorem scientific_notation_of_935000000 :
  935000000 = 9.35 * 10^8 :=
by
  sorry

end scientific_notation_of_935000000_l45_45657


namespace press_t_denomination_l45_45777

def press_f_rate_per_minute := 1000
def press_t_rate_per_minute := 200
def time_in_seconds := 3
def f_denomination := 5
def additional_amount := 50

theorem press_t_denomination : 
  ∃ (x : ℝ), 
  (3 * (5 * (1000 / 60))) = (3 * (x * (200 / 60)) + 50) → 
  x = 20 := 
by 
  -- Proof logic here
  sorry

end press_t_denomination_l45_45777


namespace scientific_notation_of_935000000_l45_45658

theorem scientific_notation_of_935000000 :
  935000000 = 9.35 * 10^8 :=
by
  sorry

end scientific_notation_of_935000000_l45_45658


namespace num_perfect_square_factors_l45_45460

-- Define the exponents and their corresponding number of perfect square factors
def num_square_factors (exp : ℕ) : ℕ := exp / 2 + 1

-- Define the product of the prime factorization
def product : ℕ := 2^12 * 3^15 * 7^18

-- State the theorem
theorem num_perfect_square_factors :
  (num_square_factors 12) * (num_square_factors 15) * (num_square_factors 18) = 560 := by
  sorry

end num_perfect_square_factors_l45_45460


namespace find_number_l45_45709

variable (n : ℝ)

theorem find_number (h₁ : (0.47 * 1442 - 0.36 * n) + 63 = 3) : 
  n = 2049.28 := 
by 
  sorry

end find_number_l45_45709


namespace oprq_possible_figures_l45_45493

theorem oprq_possible_figures (x1 y1 x2 y2 : ℝ) (h : (x1, y1) ≠ (x2, y2)) : 
  -- Define the points P, Q, and R
  let P := (x1, y1)
  let Q := (x2, y2)
  let R := (x1 - x2, y1 - y2)
  -- Proving the geometric possibilities
  (∃ k : ℝ, x1 = k * x2 ∧ y1 = k * y2) ∨
  -- When the points are collinear
  ((x1 + x2, y1 + y2) = (x1, y1)) :=
sorry

end oprq_possible_figures_l45_45493


namespace B_alone_work_days_l45_45260

theorem B_alone_work_days (B : ℕ) (A_work : ℝ) (C_work : ℝ) (total_payment : ℝ) :
  (A_work = 1 / 6) →
  (total_payment = 3200) →
  (C_work = (400 / total_payment) * (1 / 3)) →
  (A_work + 1 / B + C_work = 1 / 3) →
  B = 8 :=
begin
  intros hA_work htotal_payment hC_work hcombined_work,
  sorry,
end

end B_alone_work_days_l45_45260


namespace binomial_product_l45_45689

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := by
  sorry

end binomial_product_l45_45689


namespace mass_percentage_ba_in_bao_l45_45013

-- Define the constants needed in the problem
def molarMassBa : ℝ := 137.33
def molarMassO : ℝ := 16.00

-- Calculate the molar mass of BaO
def molarMassBaO : ℝ := molarMassBa + molarMassO

-- Express the problem as a Lean theorem for proof
theorem mass_percentage_ba_in_bao : 
  (molarMassBa / molarMassBaO) * 100 = 89.55 := by
  sorry

end mass_percentage_ba_in_bao_l45_45013


namespace hungarian_math_olympiad_1927_l45_45452

-- Definitions
def is_coprime (a b : ℤ) : Prop :=
  Int.gcd a b = 1

-- The main statement
theorem hungarian_math_olympiad_1927
  (a b c d x y k m : ℤ) 
  (h_coprime : is_coprime a b)
  (h_m : m = a * d - b * c)
  (h_divides : m ∣ (a * x + b * y)) :
  m ∣ (c * x + d * y) :=
sorry

end hungarian_math_olympiad_1927_l45_45452


namespace place_synthetic_method_l45_45396

theorem place_synthetic_method :
  "Synthetic Method" = "Direct Proof" :=
sorry

end place_synthetic_method_l45_45396


namespace total_miles_run_correct_l45_45772

-- Define the number of people on the sprint team and the miles each person runs.
def number_of_people : Float := 150.0
def miles_per_person : Float := 5.0

-- Define the total miles run by the sprint team.
def total_miles_run : Float := number_of_people * miles_per_person

-- State the theorem to prove that the total miles run is equal to 750.0 miles.
theorem total_miles_run_correct : total_miles_run = 750.0 := sorry

end total_miles_run_correct_l45_45772


namespace winning_candidate_percentage_l45_45774

theorem winning_candidate_percentage (votes1 votes2 votes3 : ℕ) (h1 : votes1 = 1256) (h2 : votes2 = 7636) (h3 : votes3 = 11628) 
    : (votes3 : ℝ) / (votes1 + votes2 + votes3) * 100 = 56.67 := by
  sorry

end winning_candidate_percentage_l45_45774


namespace product_of_roots_l45_45864

variable {k m x1 x2 : ℝ}

theorem product_of_roots (h1 : 4 * x1 ^ 2 - k * x1 - m = 0) (h2 : 4 * x2 ^ 2 - k * x2 - m = 0) (h3 : x1 ≠ x2) :
  x1 * x2 = -m / 4 :=
sorry

end product_of_roots_l45_45864


namespace sum_of_arithmetic_sequence_l45_45770

theorem sum_of_arithmetic_sequence
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (hS : ∀ n : ℕ, S n = n * a n)
    (h_condition : a 1 - a 5 - a 10 - a 15 + a 19 = 2) :
    S 19 = -38 :=
sorry

end sum_of_arithmetic_sequence_l45_45770


namespace percentage_problem_l45_45241

variable (x : ℝ)
variable (y : ℝ)

theorem percentage_problem : 
  (x / 100 * 1442 - 36 / 100 * 1412) + 63 = 252 → x = 33.52 := by
  sorry

end percentage_problem_l45_45241


namespace boys_test_l45_45988

-- Define the conditions
def passing_time : ℝ := 14
def test_results : List ℝ := [0.6, -1.1, 0, -0.2, 2, 0.5]

-- Define the proof problem
theorem boys_test (number_did_not_pass : ℕ) (fastest_time : ℝ) (average_score : ℝ) :
  passing_time = 14 →
  test_results = [0.6, -1.1, 0, -0.2, 2, 0.5] →
  number_did_not_pass = 3 ∧
  fastest_time = 12.9 ∧
  average_score = 14.3 :=
by
  intros
  sorry

end boys_test_l45_45988


namespace polynomial_roots_l45_45820

noncomputable def f (x : ℝ) : ℝ := 8 * x^4 + 28 * x^3 - 74 * x^2 - 8 * x + 48

theorem polynomial_roots:
  ∃ (a b c d : ℝ), a = -3 ∧ b = -1 ∧ c = -1 ∧ d = 2 ∧ 
  (f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) :=
sorry

end polynomial_roots_l45_45820


namespace option_transformations_incorrect_l45_45037

variable {a b x : ℝ}

theorem option_transformations_incorrect (h : a < b) :
  ¬ (3 - a < 3 - b) := by
  -- Here, we would show the incorrectness of the transformation in Option B
  sorry

end option_transformations_incorrect_l45_45037


namespace each_episode_length_l45_45857

theorem each_episode_length (h_watch_time : ∀ d : ℕ, d = 5 → 2 * 60 * d = 600)
  (h_episodes : 20 > 0) : 600 / 20 = 30 := by
  -- Conditions used:
  -- 1. h_watch_time : John wants to finish a show in 5 days by watching 2 hours a day.
  -- 2. h_episodes : There are 20 episodes.
  -- Goal: Prove that each episode is 30 minutes long.
  sorry

end each_episode_length_l45_45857


namespace inequality_holds_for_all_real_l45_45205

theorem inequality_holds_for_all_real (a : ℝ) : a + a^3 - a^4 - a^6 < 1 :=
by
  sorry

end inequality_holds_for_all_real_l45_45205


namespace part1_part2_l45_45197

noncomputable def f (x : ℝ) : ℝ :=
  abs (2 * x - 3) + abs (x - 5)

theorem part1 : { x : ℝ | f x ≥ 4 } = { x : ℝ | x ≥ 2 ∨ x ≤ 4 / 3 } :=
by
  sorry

theorem part2 (a : ℝ) : (∃ x : ℝ, f x < a) ↔ a > 7 / 2 :=
by
  sorry

end part1_part2_l45_45197


namespace range_of_ab_min_value_of_ab_plus_inv_ab_l45_45717

theorem range_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 0 < a * b ∧ a * b ≤ 1 / 4 :=
sorry

theorem min_value_of_ab_plus_inv_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  (∃ ab, ab = a * b ∧ ab + 1 / ab = 17 / 4) :=
sorry

end range_of_ab_min_value_of_ab_plus_inv_ab_l45_45717


namespace remaining_average_l45_45212

-- Definitions
def original_average (n : ℕ) (avg : ℝ) := n = 50 ∧ avg = 38
def discarded_numbers (a b : ℝ) := a = 45 ∧ b = 55

-- Proof Statement
theorem remaining_average (n : ℕ) (avg : ℝ) (a b : ℝ) (s : ℝ) :
  original_average n avg →
  discarded_numbers a b →
  s = (n * avg - (a + b)) / (n - 2) →
  s = 37.5 :=
by
  intros h_avg h_discard h_s
  sorry

end remaining_average_l45_45212


namespace sum_of_coordinates_of_reflected_points_l45_45592

theorem sum_of_coordinates_of_reflected_points (C D : ℝ × ℝ) (hx : C.1 = 3) (hy : C.2 = 8) (hD : D = (-C.1, C.2)) :
  C.1 + C.2 + D.1 + D.2 = 16 := by
  sorry

end sum_of_coordinates_of_reflected_points_l45_45592


namespace correct_calculation_l45_45624

theorem correct_calculation (a x y b : ℝ) :
  (-a - a = 0) = False ∧
  (- (x + y) = -x - y) = True ∧
  (3 * (b - 2 * a) = 3 * b - 2 * a) = False ∧
  (8 * a^4 - 6 * a^2 = 2 * a^2) = False :=
by
  sorry

end correct_calculation_l45_45624


namespace smallest_a_condition_l45_45823

theorem smallest_a_condition:
  ∃ a: ℝ, (∀ x y z: ℝ, (0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 1) → a * (x^2 + y^2 + z^2) + x * y * z ≥ 10 / 27) ∧ a = 2 / 9 :=
sorry

end smallest_a_condition_l45_45823


namespace angle_A_is_70_l45_45853

-- Definitions of angles given as conditions in the problem
variables (BAD BAC ACB : ℝ)

def angle_BAD := 150
def angle_BAC := 80

-- The Lean 4 statement to prove the measure of angle ACB
theorem angle_A_is_70 (h1 : BAD = 150) (h2 : BAC = 80) : ACB = 70 :=
by {
  sorry
}

end angle_A_is_70_l45_45853


namespace solve_pairs_l45_45293

theorem solve_pairs (m n : ℕ) :
  m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ (m, n) = (6, 3) ∨ (m, n) = (9, 3) ∨ (m, n) = (9, 5) ∨ (m, n) = (54, 5) :=
by
  sorry

end solve_pairs_l45_45293


namespace map_scale_l45_45942

theorem map_scale (map_distance : ℝ) (time : ℝ) (speed : ℝ) (actual_distance : ℝ) (scale : ℝ) 
  (h1 : map_distance = 5) 
  (h2 : time = 1.5) 
  (h3 : speed = 60) 
  (h4 : actual_distance = speed * time) 
  (h5 : scale = map_distance / actual_distance) : 
  scale = 1 / 18 :=
by 
  sorry

end map_scale_l45_45942


namespace mary_machines_sold_l45_45365

open Nat

-- Definitions
def a₁ := 1
def d := 2

-- Sequence definition
def a (n : ℕ) := a₁ + (n - 1) * d

-- Sum of the arithmetic series
def S (n : ℕ) := (n * (a₁ + a n)) / 2

-- Problem statement
theorem mary_machines_sold : S 20 = 400 :=
by
  sorry

end mary_machines_sold_l45_45365


namespace general_formula_an_general_formula_bn_exists_arithmetic_sequence_bn_l45_45025

variable (a_n : ℕ → ℝ)
variable (b_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)
variable (d : ℝ)

-- Define the initial conditions
axiom a2_a3_condition : a_n 2 * a_n 3 = 15
axiom S4_condition : S_n 4 = 16
axiom b_recursion : ∀ (n : ℕ), b_n (n + 1) - b_n n = 1 / (a_n n * a_n (n + 1))

-- Define the proofs
theorem general_formula_an : ∀ (n : ℕ), a_n n = 2 * n - 1 :=
sorry

theorem general_formula_bn : ∀ (n : ℕ), b_n n = (3 * n - 2) / (2 * n - 1) :=
sorry

theorem exists_arithmetic_sequence_bn : ∃ (m n : ℕ), m ≠ n ∧ b_n 2 + b_n n = 2 * b_n m ∧ b_n 2 = 4 / 3 ∧ (n = 8 ∧ m = 3) :=
sorry

end general_formula_an_general_formula_bn_exists_arithmetic_sequence_bn_l45_45025


namespace binomial_product_result_l45_45678

-- Defining the combination (binomial coefficient) formula
def combination (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- Lean theorem statement to prove the problem
theorem binomial_product_result : combination 10 3 * combination 8 3 = 6720 := by
  sorry

end binomial_product_result_l45_45678


namespace car_rental_cost_per_mile_l45_45263

def daily_rental_rate := 29.0
def total_amount_paid := 46.12
def miles_driven := 214.0

theorem car_rental_cost_per_mile : 
  (total_amount_paid - daily_rental_rate) / miles_driven = 0.08 := 
by
  sorry

end car_rental_cost_per_mile_l45_45263


namespace sum_of_first_20_primes_l45_45944

theorem sum_of_first_20_primes :
  ( [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71].sum = 639 ) :=
by
  sorry

end sum_of_first_20_primes_l45_45944


namespace find_p_q_r_divisibility_l45_45980

theorem find_p_q_r_divisibility 
  (p q r : ℝ)
  (h_div : ∀ x, (x^4 + 4*x^3 + 6*p*x^2 + 4*q*x + r) % (x^3 + 3*x^2 + 9*x + 3) = 0)
  : (p + q) * r = 15 :=
by
  -- Proof steps would go here
  sorry

end find_p_q_r_divisibility_l45_45980


namespace find_f_neg_one_l45_45837

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem find_f_neg_one (f : ℝ → ℝ) (h_odd : is_odd f)
(h_pos : ∀ x, 0 < x → f x = x^2 + 1/x) : f (-1) = -2 := 
sorry

end find_f_neg_one_l45_45837


namespace no_natural_number_n_exists_l45_45956

theorem no_natural_number_n_exists (n : ℕ) :
  ¬ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 + y^2 = 2 * n * (n + 1) * (n + 2) * (n + 3) + 12 := 
sorry

end no_natural_number_n_exists_l45_45956


namespace max_value_8a_3b_5c_l45_45193

theorem max_value_8a_3b_5c (a b c : ℝ) (h_condition : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  8 * a + 3 * b + 5 * c ≤ (Real.sqrt 373) / 6 :=
by
  sorry

end max_value_8a_3b_5c_l45_45193


namespace chocolates_remaining_l45_45034

def chocolates := 24
def chocolates_first_day := 4
def chocolates_eaten_second_day := (2 * chocolates_first_day) - 3
def chocolates_eaten_third_day := chocolates_first_day - 2
def chocolates_eaten_fourth_day := chocolates_eaten_third_day - 1

theorem chocolates_remaining :
  chocolates - (chocolates_first_day + chocolates_eaten_second_day + chocolates_eaten_third_day + chocolates_eaten_fourth_day) = 12 := by
  sorry

end chocolates_remaining_l45_45034


namespace find_b_l45_45171

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

def derivative_at_one (a : ℝ) : ℝ := a + 1

def tangent_line (b : ℝ) (x : ℝ) : ℝ := 2 * x + b

theorem find_b (a b : ℝ) (h_deriv : derivative_at_one a = 2) (h_tangent : tangent_line b 1 = curve a 1) :
  b = -1 :=
by
  sorry

end find_b_l45_45171


namespace correct_operation_l45_45782

theorem correct_operation (a : ℕ) : a ^ 3 * a ^ 2 = a ^ 5 :=
by sorry

end correct_operation_l45_45782


namespace sqrt_sequence_convergence_l45_45336

theorem sqrt_sequence_convergence :
  ∃ x : ℝ, (x = Real.sqrt (1 + x) ∧ 1 < x ∧ x < 2) :=
sorry

end sqrt_sequence_convergence_l45_45336


namespace length_of_train_l45_45934

-- Define the conditions as variables
def speed : ℝ := 39.27272727272727
def time : ℝ := 55
def length_bridge : ℝ := 480

-- Calculate the total distance using the given conditions
def total_distance : ℝ := speed * time

-- Prove that the length of the train is 1680 meters
theorem length_of_train :
  (total_distance - length_bridge) = 1680 :=
by
  sorry

end length_of_train_l45_45934


namespace distance_between_intersections_l45_45438

theorem distance_between_intersections (a : ℝ) (a_pos : 0 < a) : 
  |(Real.log a / Real.log 2) - (Real.log (a / 3) / Real.log 2)| = Real.log 3 / Real.log 2 :=
by
  sorry

end distance_between_intersections_l45_45438


namespace shaded_total_area_l45_45045

theorem shaded_total_area:
  ∀ (r₁ r₂ r₃ : ℝ),
  π * r₁ ^ 2 = 100 * π →
  r₂ = r₁ / 2 →
  r₃ = r₂ / 2 →
  (1 / 2) * (π * r₁ ^ 2) + (1 / 2) * (π * r₂ ^ 2) + (1 / 2) * (π * r₃ ^ 2) = 65.625 * π :=
by
  intro r₁ r₂ r₃ h₁ h₂ h₃
  sorry

end shaded_total_area_l45_45045


namespace square_area_problem_l45_45239

theorem square_area_problem
    (x1 y1 x2 y2 : ℝ)
    (h1 : y1 = x1^2)
    (h2 : y2 = x2^2)
    (line_eq : ∃ a : ℝ, a = 2 ∧ ∃ b : ℝ, b = -22 ∧ ∀ x y : ℝ, y = 2 * x - 22 → (y = y1 ∨ y = y2)) :
    ∃ area : ℝ, area = 180 ∨ area = 980 :=
sorry

end square_area_problem_l45_45239


namespace company_starts_to_make_profit_in_third_year_first_option_more_cost_effective_l45_45117

-- Define the conditions about the fishing company's boat purchase and expenses
def initial_purchase_cost : ℕ := 980000
def first_year_expenses : ℕ := 120000
def expense_increment : ℕ := 40000
def annual_income : ℕ := 500000

-- Prove that the company starts to make a profit in the third year
theorem company_starts_to_make_profit_in_third_year : 
  ∃ (year : ℕ), year = 3 ∧ 
  annual_income * year > initial_purchase_cost + first_year_expenses + (expense_increment * (year - 1) * year / 2) :=
sorry

-- Prove that the first option is more cost-effective
theorem first_option_more_cost_effective : 
  (annual_income * 3 - (initial_purchase_cost + first_year_expenses + expense_increment * (3 - 1) * 3 / 2) + 260000) > 
  (annual_income * 5 - (initial_purchase_cost + first_year_expenses + expense_increment * (5 - 1) * 5 / 2) + 80000) :=
sorry

end company_starts_to_make_profit_in_third_year_first_option_more_cost_effective_l45_45117


namespace crumble_topping_correct_amount_l45_45201

noncomputable def crumble_topping_total_mass (flour butter sugar : ℕ) (factor : ℚ) : ℚ :=
  factor * (flour + butter + sugar) / 1000  -- convert grams to kilograms

theorem crumble_topping_correct_amount {flour butter sugar : ℕ} (factor : ℚ) (h_flour : flour = 100) (h_butter : butter = 50) (h_sugar : sugar = 50) (h_factor : factor = 2.5) :
  crumble_topping_total_mass flour butter sugar factor = 0.5 :=
by
  sorry

end crumble_topping_correct_amount_l45_45201


namespace average_age_of_boys_l45_45176

theorem average_age_of_boys
  (N : ℕ) (G : ℕ) (A_G : ℕ) (A_S : ℚ) (B : ℕ)
  (hN : N = 652)
  (hG : G = 163)
  (hA_G : A_G = 11)
  (hA_S : A_S = 11.75)
  (hB : B = N - G) :
  (163 * 11 + 489 * x = 11.75 * 652) → x = 12 := by
  sorry

end average_age_of_boys_l45_45176


namespace range_of_k_l45_45891

noncomputable def f (x k : ℝ) : ℝ := 2^x + 3*x - k

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, 1 ≤ x ∧ x < 2 ∧ f x k = 0) ↔ 5 ≤ k ∧ k < 10 :=
by sorry

end range_of_k_l45_45891


namespace find_special_integer_l45_45472

theorem find_special_integer :
  ∃ (n : ℕ), n > 0 ∧ (21 ∣ n) ∧ 30 ≤ Real.sqrt n ∧ Real.sqrt n ≤ 30.5 ∧ n = 903 := 
sorry

end find_special_integer_l45_45472


namespace max_profit_300_l45_45266

noncomputable def total_cost (x : ℝ) : ℝ := 20000 + 100 * x

noncomputable def total_revenue (x : ℝ) : ℝ :=
if x ≤ 400 then (400 * x - (1 / 2) * x^2)
else 80000

noncomputable def total_profit (x : ℝ) : ℝ :=
total_revenue x - total_cost x

theorem max_profit_300 :
    ∃ x : ℝ, (total_profit x = (total_revenue 300 - total_cost 300)) := sorry

end max_profit_300_l45_45266


namespace count_similar_divisors_l45_45796

def is_integrally_similar_divisible (a b c : ℕ) : Prop :=
  ∃ x y z : ℕ, a * c = b * z ∧
  x ≤ y ∧ y ≤ z ∧
  b = 2023 ∧ a * c = 2023^2

theorem count_similar_divisors (b : ℕ) (hb : b = 2023) :
  ∃ (n : ℕ), n = 7 ∧ 
    (∀ (a c : ℕ), a ≤ b ∧ b ≤ c → is_integrally_similar_divisible a b c) :=
by
  sorry

end count_similar_divisors_l45_45796


namespace relationship_among_f_l45_45018

theorem relationship_among_f (
  f : ℝ → ℝ
) (h_even : ∀ x, f x = f (-x))
  (h_periodic : ∀ x, f (x - 1) = f (x + 1))
  (h_increasing : ∀ a b, (0 ≤ a ∧ a < b ∧ b ≤ 1) → f a < f b) :
  f 2 < f (-5.5) ∧ f (-5.5) < f (-1) :=
by
  sorry

end relationship_among_f_l45_45018


namespace cost_per_spool_l45_45199

theorem cost_per_spool
  (p : ℕ) (f : ℕ) (y : ℕ) (t : ℕ) (n : ℕ)
  (hp : p = 15) (hf : f = 24) (hy : y = 5) (ht : t = 141) (hn : n = 2) :
  (t - (p + y * f)) / n = 3 :=
by sorry

end cost_per_spool_l45_45199


namespace binom_mult_eq_6720_l45_45697

theorem binom_mult_eq_6720 :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_mult_eq_6720_l45_45697


namespace speed_of_second_car_l45_45417

/-!
Two cars started from the same point, at 5 am, traveling in opposite directions. 
One car was traveling at 50 mph, and they were 450 miles apart at 10 am. 
Prove that the speed of the other car is 40 mph.
-/

variable (S : ℝ) -- Speed of the second car

theorem speed_of_second_car
    (h1 : ∀ t : ℝ, t = 5) -- The time of travel from 5 am to 10 am is 5 hours 
    (h2 : ∀ d₁ : ℝ, d₁ = 50 * 5) -- Distance traveled by the first car
    (h3 : ∀ d₂ : ℝ, d₂ = S * 5) -- Distance traveled by the second car
    (h4 : 450 = 50 * 5 + S * 5) -- Total distance between the two cars
    : S = 40 := sorry

end speed_of_second_car_l45_45417


namespace employed_females_percentage_l45_45252

theorem employed_females_percentage (total_population : ℝ) (total_employed_percentage : ℝ) (employed_males_percentage : ℝ) :
  total_employed_percentage = 0.7 →
  employed_males_percentage = 0.21 →
  total_population > 0 →
  (total_employed_percentage - employed_males_percentage) / total_employed_percentage * 100 = 70 :=
by
  intros h1 h2 h3
  -- Proof is omitted.
  sorry

end employed_females_percentage_l45_45252


namespace addition_example_l45_45221

theorem addition_example : 300 + 2020 + 10001 = 12321 := 
by 
  sorry

end addition_example_l45_45221


namespace Katie_marble_count_l45_45361

theorem Katie_marble_count :
  ∀ (pink_marbles orange_marbles purple_marbles total_marbles : ℕ),
  pink_marbles = 13 →
  orange_marbles = pink_marbles - 9 →
  purple_marbles = 4 * orange_marbles →
  total_marbles = pink_marbles + orange_marbles + purple_marbles →
  total_marbles = 33 :=
by
  intros pink_marbles orange_marbles purple_marbles total_marbles
  intros hpink horange hpurple htotal
  sorry

end Katie_marble_count_l45_45361


namespace factorize_cubic_l45_45009

theorem factorize_cubic : ∀ x : ℝ, x^3 - 4 * x = x * (x + 2) * (x - 2) :=
by
  sorry

end factorize_cubic_l45_45009


namespace coefficients_sum_correct_l45_45962

noncomputable def poly_expr (x : ℝ) : ℝ := (x + 2)^4

def coefficients_sum (a a_1 a_2 a_3 a_4 : ℝ) : ℝ :=
  a_1 + a_2 + a_3 + a_4

theorem coefficients_sum_correct (a a_1 a_2 a_3 a_4 : ℝ) :
  poly_expr 1 = a_4 * 1 ^ 4 + a_3 * 1 ^ 3 + a_2 * 1 ^ 2 + a_1 * 1 + a →
  a = 16 → coefficients_sum a a_1 a_2 a_3 a_4 = 65 :=
by
  intro h₁ h₂
  sorry

end coefficients_sum_correct_l45_45962


namespace total_wheels_in_storage_l45_45737

def wheels (n_bicycles n_tricycles n_unicycles n_quadbikes : ℕ) : ℕ :=
  (n_bicycles * 2) + (n_tricycles * 3) + (n_unicycles * 1) + (n_quadbikes * 4)

theorem total_wheels_in_storage :
  let n_bicycles := 24
  let n_tricycles := 14
  let n_unicycles := 10
  let n_quadbikes := 8
  wheels n_bicycles n_tricycles n_unicycles n_quadbikes = 132 :=
by
  let n_bicycles := 24
  let n_tricycles := 14
  let n_unicycles := 10
  let n_quadbikes := 8
  show wheels n_bicycles n_tricycles n_unicycles n_quadbikes = 132
  sorry

end total_wheels_in_storage_l45_45737


namespace binary_representation_of_28_l45_45765

-- Define a function to convert a number to binary representation.
def decimalToBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else 
    let rec aux (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else aux (n / 2) ((n % 2) :: acc)
    aux n []

theorem binary_representation_of_28 : decimalToBinary 28 = [1, 1, 1, 0, 0] := 
  sorry

end binary_representation_of_28_l45_45765


namespace painted_cells_possible_values_l45_45858

theorem painted_cells_possible_values (k l : ℕ) (hk : 2 * k + 1 > 0) (hl : 2 * l + 1 > 0) (h : k * l = 74) :
  (2 * k + 1) * (2 * l + 1) - 74 = 301 ∨ (2 * k + 1) * (2 * l + 1) - 74 = 373 := 
sorry

end painted_cells_possible_values_l45_45858


namespace larger_cuboid_length_is_16_l45_45843

def volume (l w h : ℝ) : ℝ := l * w * h

def cuboid_length_proof : Prop :=
  ∀ (length_large : ℝ), 
  (volume 5 4 3 * 32 = volume length_large 10 12) → 
  length_large = 16

theorem larger_cuboid_length_is_16 : cuboid_length_proof :=
by
  intros length_large eq_volume
  sorry

end larger_cuboid_length_is_16_l45_45843


namespace janelle_total_marbles_l45_45855

def initial_green_marbles := 26
def bags_of_blue_marbles := 12
def marbles_per_bag := 15
def gift_red_marbles := 7
def gift_green_marbles := 9
def gift_blue_marbles := 12
def gift_red_marbles_given := 3
def returned_blue_marbles := 8

theorem janelle_total_marbles :
  let total_green := initial_green_marbles - gift_green_marbles
  let total_blue := (bags_of_blue_marbles * marbles_per_bag) - gift_blue_marbles + returned_blue_marbles
  let total_red := gift_red_marbles - gift_red_marbles_given
  total_green + total_blue + total_red = 197 :=
by
  sorry

end janelle_total_marbles_l45_45855


namespace num_of_consecutive_sets_sum_18_eq_2_l45_45334

theorem num_of_consecutive_sets_sum_18_eq_2 : 
  ∃ (sets : Finset (Finset ℕ)), 
    (∀ s ∈ sets, (∃ n a, n ≥ 3 ∧ (s = Finset.range (a + n - 1) \ Finset.range (a - 1)) ∧ 
    s.sum id = 18)) ∧ 
    sets.card = 2 := 
sorry

end num_of_consecutive_sets_sum_18_eq_2_l45_45334


namespace equation_solution_l45_45223

theorem equation_solution (x : ℝ) : (3 : ℝ)^(x-1) = 1/9 ↔ x = -1 :=
by sorry

end equation_solution_l45_45223


namespace find_number_l45_45847

theorem find_number (n : ℝ) (h : (1 / 3) * n = 6) : n = 18 :=
sorry

end find_number_l45_45847


namespace unknown_sum_of_digits_l45_45179

theorem unknown_sum_of_digits 
  (A B C D : ℕ) 
  (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h2 : D = 1)
  (h3 : (A * 100 + B * 10 + C) * D = A * 1000 + B * 100 + C * 10 + D) : 
  A + B = 0 := 
sorry

end unknown_sum_of_digits_l45_45179


namespace smallest_nonpalindromic_power_of_7_l45_45957

noncomputable def isPalindrome (n : ℕ) : Bool :=
  let s := n.toString
  s == s.reverse

theorem smallest_nonpalindromic_power_of_7 :
  ∃ n : ℕ, ∃ m : ℕ, m = 7^n ∧ ¬ isPalindrome m ∧ ∀ k : ℕ, k < n → (isPalindrome (7^k) → False) → n = 4 ∧ m = 2401 :=
by sorry

end smallest_nonpalindromic_power_of_7_l45_45957


namespace selene_sandwiches_l45_45797

-- Define the context and conditions in Lean
variables (S : ℕ) (sandwich_cost hamburger_cost hotdog_cost juice_cost : ℕ)
  (selene_cost tanya_cost total_cost : ℕ)

-- Each item prices
axiom sandwich_price : sandwich_cost = 2
axiom hamburger_price : hamburger_cost = 2
axiom hotdog_price : hotdog_cost = 1
axiom juice_price : juice_cost = 2

-- Purchases
axiom selene_purchase : selene_cost = sandwich_cost * S + juice_cost
axiom tanya_purchase : tanya_cost = hamburger_cost * 2 + juice_cost * 2

-- Total spending
axiom total_spending : selene_cost + tanya_cost = 16

-- Goal: Prove that Selene bought 3 sandwiches
theorem selene_sandwiches : S = 3 :=
by {
  sorry
}

end selene_sandwiches_l45_45797


namespace valid_probability_distribution_of_X_probability_A_or_B_conditional_probability_B_given_A_eq_l45_45758

def number_of_students : ℕ := 6
def number_of_boys : ℕ := 4
def number_of_girls : ℕ := 2
def number_of_selected_students : ℕ := 3

noncomputable def probability_distribution_of_X : (Fin 3) → ℚ :=
  λ X, match X with
  | ⟨0, _⟩ => 1/5
  | ⟨1, _⟩ => 3/5
  | ⟨2, _⟩ => 1/5

axiom sum_of_probabilities : 
  probability_distribution_of_X ⟨0, _⟩ + 
  probability_distribution_of_X ⟨1, _⟩ + 
  probability_distribution_of_X ⟨2, _⟩ = 1

def event_A_selected : ℚ := 1/2  -- Probability that boy A is selected
def event_B_selected : ℚ := 1/2  -- Empirical value for girl B selected (not used further)
def event_AB_selected : ℚ := 1/5 -- Probability that both A and B are selected

noncomputable def probability_A_or_B_selected : ℚ := 4/5
noncomputable def conditional_probability_B_given_A : ℚ :=
  event_AB_selected / event_A_selected

theorem valid_probability_distribution_of_X : 
  ∀ X : Fin 3, probability_distribution_of_X X ∈ {1/5, 3/5} := by sorry

theorem probability_A_or_B : probability_A_or_B_selected = 4 / 5 := by sorry

theorem conditional_probability_B_given_A_eq : 
  conditional_probability_B_given_A = 2 / 5 := by sorry

end valid_probability_distribution_of_X_probability_A_or_B_conditional_probability_B_given_A_eq_l45_45758


namespace player_current_average_l45_45792

theorem player_current_average
  (A : ℕ) -- Assume A is a natural number (non-negative)
  (cond1 : 10 * A + 78 = 11 * (A + 4)) :
  A = 34 :=
by
  sorry

end player_current_average_l45_45792


namespace solution1_solution2_l45_45710

noncomputable def problem1 (x : ℝ) : Prop :=
  4 * x^2 - 25 = 0

theorem solution1 (x : ℝ) : problem1 x ↔ x = 5 / 2 ∨ x = -5 / 2 :=
by sorry

noncomputable def problem2 (x : ℝ) : Prop :=
  (x + 1)^3 = -27

theorem solution2 (x : ℝ) : problem2 x ↔ x = -4 :=
by sorry

end solution1_solution2_l45_45710


namespace num_cars_can_be_parked_l45_45740

theorem num_cars_can_be_parked (length width : ℝ) (useable_percentage : ℝ) (area_per_car : ℝ) 
  (h_length : length = 400) (h_width : width = 500) (h_useable_percentage : useable_percentage = 0.80) 
  (h_area_per_car : area_per_car = 10) : 
  length * width * useable_percentage / area_per_car = 16000 := 
by 
  sorry

end num_cars_can_be_parked_l45_45740


namespace factorize_cubic_l45_45008

theorem factorize_cubic : ∀ x : ℝ, x^3 - 4 * x = x * (x + 2) * (x - 2) :=
by
  sorry

end factorize_cubic_l45_45008


namespace max_abs_eq_one_vertices_l45_45817

theorem max_abs_eq_one_vertices (x y : ℝ) :
  (max (|x + y|) (|x - y|) = 1) ↔ (x = -1 ∧ y = 0) ∨ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = -1) ∨ (x = 0 ∧ y = 1) :=
sorry

end max_abs_eq_one_vertices_l45_45817


namespace paco_more_cookies_l45_45870

def paco_cookies_difference
  (initial_cookies : ℕ) 
  (cookies_eaten : ℕ) 
  (cookies_given : ℕ) : ℕ :=
  cookies_eaten - cookies_given

theorem paco_more_cookies 
  (initial_cookies : ℕ)
  (cookies_eaten : ℕ)
  (cookies_given : ℕ)
  (h1 : initial_cookies = 17)
  (h2 : cookies_eaten = 14)
  (h3 : cookies_given = 13) :
  paco_cookies_difference initial_cookies cookies_eaten cookies_given = 1 :=
by
  rw [h2, h3]
  exact rfl

end paco_more_cookies_l45_45870


namespace narrow_black_stripes_l45_45569

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end narrow_black_stripes_l45_45569


namespace count_perfect_squares_diff_l45_45333

theorem count_perfect_squares_diff (a b : ℕ) : 
  ∃ (count : ℕ), 
  count = 25 ∧ 
  (∀ (a : ℕ), (∃ (b : ℕ), a^2 = 2 * b + 1 ∧ a^2 < 2500) ↔ (∃ (k : ℕ), 1 ≤ k ∧ k ≤ 25 ∧ 2 * k - 1 = a)) :=
by
  sorry

end count_perfect_squares_diff_l45_45333


namespace customers_left_l45_45656

theorem customers_left (x : ℕ) 
  (h1 : 47 - x + 20 = 26) : 
  x = 41 :=
sorry

end customers_left_l45_45656


namespace inequality_proof_l45_45311

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
sorry

end inequality_proof_l45_45311


namespace correct_operation_B_l45_45428

variable (a : ℝ)

theorem correct_operation_B :
  2 * a^2 * a^4 = 2 * a^6 :=
by sorry

end correct_operation_B_l45_45428


namespace balls_in_boxes_l45_45164

open Nat

theorem balls_in_boxes : 
  let balls := 7
  let boxes := 4
  (∑ (x : Fin 11), match x.1 with
  | 0 => 1 
  | 1 => choose 7 6
  | 2 => choose 7 5 * choose 2 2
  | 3 => choose 7 5
  | 4 => choose 7 4 * choose 3 3
  | 5 => choose 7 4 * choose 3 2
  | 6 => choose 7 4
  | 7 => choose 7 3 * choose 4 3 / 2
  | 8 => choose 7 3 * choose 4 2 / 2
  | 9 => choose 7 3 * choose 4 2
  | 10 => choose 7 2 * choose 5 2 * choose 3 2 / 2
  end) = 890 :=
by 
  sorry

end balls_in_boxes_l45_45164


namespace ratio_of_speeds_l45_45073

theorem ratio_of_speeds (P R : ℝ) (total_time : ℝ) (time_rickey : ℝ)
  (h1 : total_time = 70)
  (h2 : time_rickey = 40)
  (h3 : total_time - time_rickey = 30) :
  P / R = 3 / 4 :=
by
  sorry

end ratio_of_speeds_l45_45073


namespace quadratic_roots_ratio_l45_45826

theorem quadratic_roots_ratio (p x1 x2 : ℝ) (h_eq : x1^2 + p * x1 - 16 = 0) (h_ratio : x1 / x2 = -4) :
  p = 6 ∨ p = -6 :=
by {
  sorry
}

end quadratic_roots_ratio_l45_45826


namespace carrie_savings_l45_45670

-- Define the original prices and discount rates
def deltaPrice : ℝ := 850
def deltaDiscount : ℝ := 0.20
def unitedPrice : ℝ := 1100
def unitedDiscount : ℝ := 0.30

-- Calculate discounted prices
def deltaDiscountAmount : ℝ := deltaPrice * deltaDiscount
def unitedDiscountAmount : ℝ := unitedPrice * unitedDiscount

def deltaDiscountedPrice : ℝ := deltaPrice - deltaDiscountAmount
def unitedDiscountedPrice : ℝ := unitedPrice - unitedDiscountAmount

-- Define the savings by choosing the cheaper flight
def savingsByChoosingCheaperFlight : ℝ := unitedDiscountedPrice - deltaDiscountedPrice

-- The theorem stating the amount saved
theorem carrie_savings : savingsByChoosingCheaperFlight = 90 :=
by
  sorry

end carrie_savings_l45_45670


namespace total_time_correct_l45_45061

-- Define the individual times
def driving_time_one_way : ℕ := 20
def attending_time : ℕ := 70

-- Define the total driving time as twice the one-way driving time
def total_driving_time : ℕ := driving_time_one_way * 2

-- Define the total time as the sum of total driving time and attending time
def total_time : ℕ := total_driving_time + attending_time

-- Prove that the total time is 110 minutes
theorem total_time_correct : total_time = 110 := by
  -- The proof is omitted, we're only interested in the statement format.
  sorry

end total_time_correct_l45_45061


namespace largest_remainder_a_correct_l45_45257

def largest_remainder_a (n : ℕ) (h : n < 150) : ℕ :=
  (269 % n)

theorem largest_remainder_a_correct : ∃ n < 150, largest_remainder_a n sorry = 133 :=
  sorry

end largest_remainder_a_correct_l45_45257


namespace gcd_three_numbers_l45_45948

def a : ℕ := 13680
def b : ℕ := 20400
def c : ℕ := 47600

theorem gcd_three_numbers (a b c : ℕ) : Nat.gcd (Nat.gcd a b) c = 80 :=
by
  sorry

end gcd_three_numbers_l45_45948


namespace arithmetic_sequence_solution_l45_45610

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
∀ n : ℕ, a n = a1 + n * d

noncomputable def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
n * a 0 + (n * (n - 1) / 2) * (a 1 - a 0)

theorem arithmetic_sequence_solution :
  ∃ d : ℤ,
  (∀ n : ℕ, n > 0 ∧ n < 10 → a n = 23 + n * d) ∧
  (23 + 5 * d > 0) ∧
  (23 + 6 * d < 0) ∧
  d = -4 ∧
  S a 6 = 78 ∧
  ∀ n : ℕ, S a n > 0 → n ≤ 12 :=
by
  sorry

end arithmetic_sequence_solution_l45_45610


namespace ylona_initial_bands_l45_45352

variable (B J Y : ℕ)  -- Represents the initial number of rubber bands for Bailey, Justine, and Ylona respectively

-- Define the conditions
axiom h1 : J = B + 10
axiom h2 : J = Y - 2
axiom h3 : B - 4 = 8

-- Formulate the statement
theorem ylona_initial_bands : Y = 24 :=
by
  sorry

end ylona_initial_bands_l45_45352


namespace problem_evaluation_l45_45141

theorem problem_evaluation : (726 * 726) - (725 * 727) = 1 := 
by 
  sorry

end problem_evaluation_l45_45141


namespace raft_travel_distance_l45_45444

theorem raft_travel_distance (v_b v_s t : ℝ) (h1 : t > 0) 
  (h2 : v_b + v_s = 90 / t) (h3 : v_b - v_s = 70 / t) : 
  v_s * t = 10 := by
  sorry

end raft_travel_distance_l45_45444


namespace quadratic_inequality_real_solution_l45_45163

theorem quadratic_inequality_real_solution (a : ℝ) :
  (∃ x : ℝ, 2*x^2 + (a-1)*x + 1/2 ≤ 0) ↔ (a ≤ -1 ∨ 3 ≤ a) := 
sorry

end quadratic_inequality_real_solution_l45_45163


namespace NY_Mets_fans_count_l45_45040

noncomputable def NY_Yankees_fans (M: ℝ) : ℝ := (3/2) * M
noncomputable def Boston_Red_Sox_fans (M: ℝ) : ℝ := (5/4) * M
noncomputable def LA_Dodgers_fans (R: ℝ) : ℝ := (2/7) * R

theorem NY_Mets_fans_count :
  ∃ M : ℕ, let Y := NY_Yankees_fans M
           let R := Boston_Red_Sox_fans M
           let D := LA_Dodgers_fans R
           Y + M + R + D = 780 ∧ M = 178 :=
by
  sorry

end NY_Mets_fans_count_l45_45040


namespace factor_x_squared_minus_64_l45_45290

-- Conditions
def a := x
def b := 8

-- Theorem statement
theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_64_l45_45290


namespace mean_of_set_with_median_l45_45608

theorem mean_of_set_with_median (m : ℝ) (h : m + 7 = 10) :
  (m + (m + 2) + (m + 7) + (m + 10) + (m + 12)) / 5 = 9.2 :=
by
  -- Placeholder for the proof.
  sorry

end mean_of_set_with_median_l45_45608


namespace area_of_rectangle_EFGH_l45_45775

theorem area_of_rectangle_EFGH :
  ∀ (a b c : ℕ), 
    a = 7 → 
    b = 3 * a → 
    c = 2 * a → 
    (area : ℕ) = b * c → 
    area = 294 := 
by
  sorry

end area_of_rectangle_EFGH_l45_45775


namespace total_amount_is_correct_l45_45933

variable (w x y z R : ℝ)
variable (hx : x = 0.345 * w)
variable (hy : y = 0.45625 * w)
variable (hz : z = 0.61875 * w)
variable (hy_value : y = 112.50)

theorem total_amount_is_correct :
  R = w + x + y + z → R = 596.8150684931507 := by
  sorry

end total_amount_is_correct_l45_45933


namespace fabric_area_l45_45143

theorem fabric_area (length width : ℝ) (h_length : length = 8) (h_width : width = 3) : 
  length * width = 24 := 
by
  rw [h_length, h_width]
  norm_num

end fabric_area_l45_45143


namespace distance_between_locations_A_and_B_l45_45369

theorem distance_between_locations_A_and_B 
  (speed_A speed_B speed_C : ℝ)
  (distance_CD : ℝ)
  (distance_initial_A : ℝ)
  (distance_A_to_B : ℝ)
  (h1 : speed_A = 3 * speed_C)
  (h2 : speed_A = 1.5 * speed_B)
  (h3 : distance_CD = 12)
  (h4 : distance_initial_A = 50)
  (h5 : distance_A_to_B = 130)
  : distance_A_to_B = 130 :=
by
  sorry

end distance_between_locations_A_and_B_l45_45369


namespace find_integer_in_range_divisible_by_18_l45_45010

theorem find_integer_in_range_divisible_by_18 
  (n : ℕ) (h1 : 900 ≤ n) (h2 : n ≤ 912) (h3 : n % 18 = 0) : n = 900 :=
sorry

end find_integer_in_range_divisible_by_18_l45_45010


namespace inequality_proof_l45_45309

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
sorry

end inequality_proof_l45_45309


namespace flight_time_is_10_hours_l45_45950

def time_watching_TV_episodes : ℕ := 3 * 25
def time_sleeping : ℕ := 4 * 60 + 30
def time_watching_movies : ℕ := 2 * (1 * 60 + 45)
def remaining_flight_time : ℕ := 45

def total_flight_time : ℕ := (time_watching_TV_episodes + time_sleeping + time_watching_movies + remaining_flight_time) / 60

theorem flight_time_is_10_hours : total_flight_time = 10 := by
  sorry

end flight_time_is_10_hours_l45_45950


namespace xy_product_l45_45319

variable {x y : ℝ}

theorem xy_product (h1 : x ≠ y) (h2 : x ≠ 0) (h3 : y ≠ 0) (h4 : x + 3/x = y + 3/y) : x * y = 3 :=
sorry

end xy_product_l45_45319


namespace factorization_of_x_squared_minus_64_l45_45287

theorem factorization_of_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factorization_of_x_squared_minus_64_l45_45287


namespace cos_double_angle_sum_l45_45482

theorem cos_double_angle_sum
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 := by
  sorry

end cos_double_angle_sum_l45_45482


namespace commercial_break_duration_l45_45468

theorem commercial_break_duration (n1 n2 m1 m2 : ℕ) (h1 : n1 = 3) (h2 : m1 = 5) (h3 : n2 = 11) (h4 : m2 = 2) :
  n1 * m1 + n2 * m2 = 37 :=
by
  -- Here, in a real proof, we would substitute and show the calculations.
  sorry

end commercial_break_duration_l45_45468


namespace warehouse_width_l45_45456

theorem warehouse_width (L : ℕ) (circles : ℕ) (total_distance : ℕ)
  (hL : L = 600)
  (hcircles : circles = 8)
  (htotal_distance : total_distance = 16000) : 
  ∃ W : ℕ, 2 * L + 2 * W = (total_distance / circles) ∧ W = 400 :=
by
  sorry

end warehouse_width_l45_45456


namespace arrests_per_day_in_each_city_l45_45420

-- Define the known conditions
def daysOfProtest := 30
def numberOfCities := 21
def daysInJailBeforeTrial := 4
def daysInJailAfterTrial := 7 / 2 * 7 -- half of a 2-week sentence in days, converted from weeks to days
def combinedJailTimeInWeeks := 9900
def combinedJailTimeInDays := combinedJailTimeInWeeks * 7

-- Define the proof statement
theorem arrests_per_day_in_each_city :
  (combinedJailTimeInDays / (daysInJailBeforeTrial + daysInJailAfterTrial)) / daysOfProtest / numberOfCities = 10 := 
by
  sorry

end arrests_per_day_in_each_city_l45_45420


namespace geometric_sequence_product_l45_45046

theorem geometric_sequence_product 
    (a : ℕ → ℝ)
    (h_geom : ∀ n m, a (n + m) = a n * a m)
    (h_roots : ∀ x, x^2 - 3*x + 2 = 0 → (x = a 7 ∨ x = a 13)) :
  a 2 * a 18 = 2 := 
sorry

end geometric_sequence_product_l45_45046


namespace FindDotsOnFaces_l45_45091

-- Define the structure of a die with specific dot distribution
structure Die where
  three_dots_face : ℕ
  two_dots_faces : ℕ
  one_dot_faces : ℕ

-- Define the problem scenario of 7 identical dice forming 'П' shape
noncomputable def SevenIdenticalDiceFormP (A B C : ℕ) : Prop :=
  ∃ (d : Die), 
    d.three_dots_face = 3 ∧
    d.two_dots_faces = 2 ∧
    d.one_dot_faces = 1 ∧
    (d.three_dots_face + d.two_dots_faces + d.one_dot_faces = 6) ∧
    (A = 2) ∧
    (B = 2) ∧
    (C = 3) 

-- State the theorem to prove A = 2, B = 2, C = 3 given the conditions
theorem FindDotsOnFaces (A B C : ℕ) (h : SevenIdenticalDiceFormP A B C) : A = 2 ∧ B = 2 ∧ C = 3 :=
  by sorry

end FindDotsOnFaces_l45_45091


namespace proof_problem_l45_45489

-- definitions of the given conditions
variable (a b c : ℝ)
variables (h₁ : 6 < a) (h₂ : a < 10) 
variable (h₃ : (a / 2) ≤ b ∧ b ≤ 2 * a) 
variable (h₄ : c = a + b)

-- statement to be proved
theorem proof_problem (h₁ : 6 < a) (h₂ : a < 10) (h₃ : (a / 2) ≤ b ∧ b ≤ 2 * a) (h₄ : c = a + b) : 9 < c ∧ c < 30 := 
sorry

end proof_problem_l45_45489


namespace economical_club_l45_45894

-- Definitions of cost functions for Club A and Club B
def f (x : ℕ) : ℕ := 5 * x

def g (x : ℕ) : ℕ := if x ≤ 30 then 90 else 2 * x + 30

-- Theorem to determine the more economical club
theorem economical_club (x : ℕ) (hx : 15 ≤ x ∧ x ≤ 40) :
  (15 ≤ x ∧ x < 18 → f x < g x) ∧
  (x = 18 → f x = g x) ∧
  (18 < x ∧ x ≤ 30 → f x > g x) ∧
  (30 < x ∧ x ≤ 40 → f x > g x) :=
sorry

end economical_club_l45_45894


namespace narrow_black_stripes_l45_45547

variable (w n b : ℕ)

theorem narrow_black_stripes (w : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := by
  have h3 : w + n = (w + 7) + 1 := by rw [h1]
  have h4 : w + n = w + 8 := by linarith
  have h5 : n = 8 := by linarith
  exact h5

end narrow_black_stripes_l45_45547


namespace solve_fraction_l45_45615

theorem solve_fraction (x : ℝ) (h1 : x + 2 = 0) (h2 : 2 * x - 4 ≠ 0) : x = -2 := 
by 
  sorry

end solve_fraction_l45_45615


namespace x_fifth_power_sum_l45_45978

theorem x_fifth_power_sum (x : ℝ) (h : x + 1 / x = -5) : x^5 + 1 / x^5 = -2525 := by
  sorry

end x_fifth_power_sum_l45_45978


namespace highlights_part_to_whole_relation_l45_45422

/-- A predicate representing different types of statistical graphs. -/
inductive StatGraphType where
  | BarGraph : StatGraphType
  | PieChart : StatGraphType
  | LineGraph : StatGraphType
  | FrequencyDistributionHistogram : StatGraphType

/-- A lemma specifying that the PieChart is the graph type that highlights the relationship between a part and the whole. -/
theorem highlights_part_to_whole_relation (t : StatGraphType) : t = StatGraphType.PieChart :=
  sorry

end highlights_part_to_whole_relation_l45_45422


namespace equation_nth_position_l45_45203

theorem equation_nth_position (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = 10 * n - 9 :=
by
  sorry

end equation_nth_position_l45_45203


namespace mixture_replacement_l45_45914

theorem mixture_replacement (A B T x : ℝ)
  (h1 : A / (A + B) = 7 / 12)
  (h2 : A = 21)
  (h3 : (A / (B + x)) = 7 / 9) :
  x = 12 :=
by
  sorry

end mixture_replacement_l45_45914


namespace exists_palindrome_product_more_than_100_ways_l45_45464

noncomputable def is_palindrome (n : ℕ) : Prop :=
  n = nat.reverse n

theorem exists_palindrome_product_more_than_100_ways :
  ∃ n : ℕ, (∃ palindromes : finset (ℕ × ℕ), 
    (∀ p ∈ palindromes, is_palindrome p.1 ∧ is_palindrome p.2 ∧ (p.1 * p.2 = n)) ∧ palindromes.card > 100) ∧ n = 2^101 :=
by sorry

end exists_palindrome_product_more_than_100_ways_l45_45464


namespace real_part_of_complex_l45_45749

theorem real_part_of_complex (z : ℂ) (h : i * (z + 1) = -3 + 2 * i) : z.re = 1 :=
sorry

end real_part_of_complex_l45_45749


namespace geom_series_eq_l45_45860

noncomputable def C (n : ℕ) := 256 * (1 - 1 / (4^n)) / (3 / 4)
noncomputable def D (n : ℕ) := 1024 * (1 - 1 / ((-2)^n)) / (3 / 2)

theorem geom_series_eq (n : ℕ) (h : n ≥ 1) : C n = D n ↔ n = 1 :=
by
  sorry

end geom_series_eq_l45_45860


namespace narrow_black_stripes_are_8_l45_45577

-- Define variables: w for wide black stripes, n for narrow black stripes, b for white stripes
variables (w n b : ℕ)

-- Given conditions
axiom cond1 : b = w + 7
axiom cond2 : w + n = b + 1

-- Theorem statement to prove that the number of narrow black stripes is 8
theorem narrow_black_stripes_are_8 : n = 8 :=
by sorry

end narrow_black_stripes_are_8_l45_45577


namespace technician_round_trip_completion_percentage_l45_45799

theorem technician_round_trip_completion_percentage :
  ∀ (d total_d : ℝ),
  d = 1 + (0.75 * 1) + (0.5 * 1) + (0.25 * 1) →
  total_d = 4 * 2 →
  (d / total_d) * 100 = 31.25 :=
by
  intros d total_d h1 h2
  sorry

end technician_round_trip_completion_percentage_l45_45799


namespace greatest_length_of_equal_pieces_l45_45542

theorem greatest_length_of_equal_pieces (a b c : ℕ) (h₁ : a = 42) (h₂ : b = 63) (h₃ : c = 84) :
  Nat.gcd (Nat.gcd a b) c = 21 :=
by
  rw [h₁, h₂, h₃]
  sorry

end greatest_length_of_equal_pieces_l45_45542


namespace quadratic_real_roots_iff_l45_45159

-- Define the statement of the problem in Lean
theorem quadratic_real_roots_iff (m : ℝ) :
  (∃ x : ℂ, m * x^2 + 2 * x - 1 = 0) ↔ (m ≥ -1 ∧ m ≠ 0) := 
by
  sorry

end quadratic_real_roots_iff_l45_45159


namespace arithmetic_geometric_means_l45_45594

theorem arithmetic_geometric_means (a b : ℝ) 
  (h1 : a + b = 40) 
  (h2 : a * b = 110) : 
  a^2 + b^2 = 1380 :=
sorry

end arithmetic_geometric_means_l45_45594


namespace total_races_needed_to_determine_champion_l45_45043

-- Defining the initial conditions
def num_sprinters : ℕ := 256
def lanes : ℕ := 8
def sprinters_per_race := lanes
def eliminated_per_race := sprinters_per_race - 1

-- The statement to be proved: The number of races required to determine the champion
theorem total_races_needed_to_determine_champion :
  ∃ (races : ℕ), races = 37 ∧
  ∀ s : ℕ, s = num_sprinters → 
  ∀ l : ℕ, l = lanes → 
  ∃ e : ℕ, e = eliminated_per_race →
  s - (races * e) = 1 :=
by sorry

end total_races_needed_to_determine_champion_l45_45043


namespace problem1_problem2_l45_45639

-- Problem 1
theorem problem1 : 40 + ((1 / 6) - (2 / 3) + (3 / 4)) * 12 = 43 :=
by
  sorry

-- Problem 2
theorem problem2 : (-1) ^ 2 * (-5) + ((-3) ^ 2 + 2 * (-5)) = 4 :=
by
  sorry

end problem1_problem2_l45_45639


namespace find_x_l45_45503

theorem find_x
  (a : ℝ × ℝ)
  (b : ℝ × ℝ)
  (h : a = (Real.sqrt 3, 0))
  (h1 : b = (x, -2))
  (h2 : a.1 * (a.1 - 2 * b.1) + a.2 * (a.2 - 2 * b.2) = 0) :
  x = Real.sqrt 3 / 2 :=
sorry

end find_x_l45_45503


namespace probability_two_white_balls_l45_45261

-- Definitions
def totalBalls : ℕ := 5
def whiteBalls : ℕ := 3
def blackBalls : ℕ := 2
def totalWaysToDrawTwoBalls : ℕ := Nat.choose totalBalls 2
def waysToDrawTwoWhiteBalls : ℕ := Nat.choose whiteBalls 2

-- Theorem statement
theorem probability_two_white_balls :
  (waysToDrawTwoWhiteBalls : ℚ) / totalWaysToDrawTwoBalls = 3 / 10 := by
  sorry

end probability_two_white_balls_l45_45261


namespace solution_set_for_inequality_l45_45298

theorem solution_set_for_inequality : {x : ℝ | x ≠ 0 ∧ (x-1)/x ≤ 0} = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end solution_set_for_inequality_l45_45298


namespace sum_of_odd_integers_less_than_50_l45_45423

def sumOddIntegersLessThan (n : Nat) : Nat :=
  List.sum (List.filter (λ x => x % 2 = 1) (List.range n))

theorem sum_of_odd_integers_less_than_50 : sumOddIntegersLessThan 50 = 625 :=
  by
    sorry

end sum_of_odd_integers_less_than_50_l45_45423


namespace proof_question_1_l45_45641

noncomputable def question_1 (x : ℝ) : ℝ :=
  (Real.sin (2 * x) + 2 * (Real.sin x)^2) / (1 - Real.tan x)

theorem proof_question_1 :
  ∀ x : ℝ, (Real.cos (π / 4 + x) = 3 / 5) →
  (17 * π / 12 < x ∧ x < 7 * π / 4) →
  question_1 x = -9 / 20 :=
by
  intros x h1 h2
  sorry

end proof_question_1_l45_45641


namespace student_l45_45519

theorem student's_incorrect_answer (D I : ℕ) (h1 : D / 36 = 58) (h2 : D / 87 = I) : I = 24 :=
sorry

end student_l45_45519


namespace exists_perfect_square_in_sequence_of_f_l45_45195

noncomputable def f (n : ℕ) : ℕ :=
  ⌊(n : ℝ) + Real.sqrt n⌋₊

theorem exists_perfect_square_in_sequence_of_f (m : ℕ) (h : m = 1111) :
  ∃ k, ∃ n, f^[n] m = k * k := 
sorry

end exists_perfect_square_in_sequence_of_f_l45_45195


namespace calculate_expr_l45_45665

theorem calculate_expr : (2023^0 + (-1/3) = 2/3) := by
  sorry

end calculate_expr_l45_45665


namespace find_a10_l45_45437

-- Conditions
variables (S : ℕ → ℕ) (a : ℕ → ℕ)
variables (hS9 : S 9 = 81) (ha2 : a 2 = 3)

-- Arithmetic sequence sum definition
def arithmetic_sequence_sum (n : ℕ) (a1 : ℕ) (d : ℕ) :=
  n * (2 * a1 + (n - 1) * d) / 2

-- a_n formula definition
def a_n (n a1 d : ℕ) := a1 + (n - 1) * d

-- Proof statement
theorem find_a10 (a1 d : ℕ) (hS9' : 9 * (2 * a1 + 8 * d) / 2 = 81) (ha2' : a1 + d = 3) :
  a 10 = a1 + 9 * d :=
sorry

end find_a10_l45_45437


namespace final_score_eq_l45_45066

variable (initial_score : ℝ)
def deduction_lost_answer : ℝ := 1
def deduction_error : ℝ := 0.5
def deduction_checks : ℝ := 0

def total_deduction : ℝ := deduction_lost_answer + deduction_error + deduction_checks

theorem final_score_eq : final_score = initial_score - total_deduction := by
  sorry

end final_score_eq_l45_45066


namespace multiples_of_6_or_8_but_not_both_l45_45508

theorem multiples_of_6_or_8_but_not_both (n : ℕ) : 
  n = 25 ∧ (n = 18) ∧ (n = 6) → (25 - 6) + (18 - 6) = 31 :=
by
  sorry

end multiples_of_6_or_8_but_not_both_l45_45508


namespace coefficient_expansion_l45_45029

noncomputable def coef_term (x y : ℕ) : ℕ :=
  (Nat.choose 5 2) * (2 * Nat.choose 3 1) * 45

theorem coefficient_expansion (x y : ℕ) :
  coef_term x y = 540 :=
by {
  sorry
}

end coefficient_expansion_l45_45029


namespace factorize_cubic_l45_45006

theorem factorize_cubic : ∀ x : ℝ, x^3 - 4 * x = x * (x + 2) * (x - 2) :=
by
  sorry

end factorize_cubic_l45_45006


namespace marla_errand_total_time_l45_45062

theorem marla_errand_total_time :
  let drive_time := 20
  let school_time := 70
  let total_time := 2 * drive_time + school_time
  total_time = 110 :=
by
  let drive_time := 20
  let school_time := 70
  let total_time := 2 * drive_time + school_time
  show total_time = 110
  sorry

end marla_errand_total_time_l45_45062


namespace cost_of_one_bag_l45_45412

theorem cost_of_one_bag (x : ℝ) :
  let Andrey_earning := 60 * 2 * x
  let Boris_earning := 15 * 1.6 * x + 45 * (1.6 * 1.4) * x
  Boris_earning - Andrey_earning = 1200 →
  x = 250 := 
by
  intros
  let Andrey_earning := 60 * 2 * x
  let Boris_earning := 15 * 1.6 * x + 45 * (1.6 * 1.4) * x
  have h : Boris_earning - Andrey_earning = 1200 := by assumption
  let simplified_h := 
    calc
      Boris_earning - Andrey_earning
        = (15 * 1.6 * x + 45 * (1.6 * 1.4) * x) - (60 * 2 * x) : by simp [Andrey_earning, Boris_earning]
    ... = (15 * 1.6 * x + 45 * (1.6 * 1.4) * x) - 120 * x : by simp
    ... = (24 * x + 100.8 * x) - 120 * x : by simp
    ... = 124.8 * x - 120 * x : by simp
    ... = 4.8 * x : by simp
    ... = 1200 : by rw h
  exact (div_eq_iff (by norm_num : (4.8 : ℝ) ≠ 0)).1 simplified_h  -- solves for x

end cost_of_one_bag_l45_45412


namespace greatest_k_l45_45794

noncomputable def n : ℕ := sorry
def k : ℕ := sorry

axiom d : ℕ → ℕ

axiom h1 : d n = 72
axiom h2 : d (5 * n) = 90

theorem greatest_k : ∃ k : ℕ, (∀ m : ℕ, m > k → ¬(5^m ∣ n)) ∧ 5^k ∣ n ∧ k = 3 :=
by
  sorry

end greatest_k_l45_45794


namespace intersection_A_B_l45_45742

def A : Set ℝ := { x | x > -1 }
def B : Set ℝ := { y | (y - 2) * (y + 3) < 0 }

theorem intersection_A_B : A ∩ B = Set.Ioo (-1) 2 :=
by
  sorry

end intersection_A_B_l45_45742


namespace narrow_black_stripes_are_eight_l45_45552

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end narrow_black_stripes_are_eight_l45_45552


namespace cubic_two_common_points_x_axis_l45_45162

theorem cubic_two_common_points_x_axis (c : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ^ 3 - 3 * x1 + c = 0 ∧ x2 ^ 3 - 3 * x2 + c = 0 ∧
    (∀ x ∈ Ioo (-1 : ℝ) 1, x^3 - 3 * x + c > 0) ∧ 
    ((∀ x ≤ -1, x^3 - 3*x + c ≠ 0) ∨ (∀ x ≥ 1, x^3 - 3*x + c ≠ 0)))
  ↔ c = -2 ∨ c = 2 :=
by
  sorry

end cubic_two_common_points_x_axis_l45_45162


namespace triangle_area_rational_l45_45235

theorem triangle_area_rational
  (x1 y1 x2 y2 x3 y3 : ℤ)
  (h : y1 = y2) :
  ∃ (k : ℚ), 
    k = abs ((x2 - x1) * y3) / 2 := sorry

end triangle_area_rational_l45_45235


namespace points_on_circle_l45_45304

theorem points_on_circle (t : ℝ) :
  let x := (t^3 - 1) / (t^3 + 1);
  let y := (2 * t^3) / (t^3 + 1);
  x^2 + y^2 = 1 :=
by
  let x := (t^3 - 1) / (t^3 + 1)
  let y := (2 * t^3) / (t^3 + 1)
  have h1 : x^2 + y^2 = ((t^3 - 1) / (t^3 + 1))^2 + ((2 * t^3) / (t^3 + 1))^2 := by rfl
  have h2 : (x^2 + y^2) = ( (t^3 - 1)^2 + (2 * t^3)^2 ) / (t^3 + 1)^2 := by sorry
  have h3 : (x^2 + y^2) = ( t^6 - 2 * t^3 + 1 + 4 * t^6 ) / (t^3 + 1)^2 := by sorry
  have h4 : (x^2 + y^2) = 1 := by sorry
  exact h4

end points_on_circle_l45_45304


namespace total_commencement_addresses_l45_45738

-- Define the given conditions
def sandoval_addresses := 12
def sandoval_rainy_addresses := 5
def sandoval_public_holidays := 2
def sandoval_non_rainy_addresses := sandoval_addresses - sandoval_rainy_addresses

def hawkins_addresses := sandoval_addresses / 2
def sloan_addresses := sandoval_addresses + 10
def sloan_non_rainy_addresses := sloan_addresses -- assuming no rainy day details are provided

def davenport_addresses := (sandoval_non_rainy_addresses + sloan_non_rainy_addresses) / 2 - 3
def davenport_addresses_rounded := 11 -- rounding down to nearest integer as per given solution

def adkins_addresses := hawkins_addresses + davenport_addresses_rounded + 2

-- Calculate the total number of addresses
def total_addresses := sandoval_addresses + hawkins_addresses + sloan_addresses + davenport_addresses_rounded + adkins_addresses

-- The proof goal statement
theorem total_commencement_addresses : total_addresses = 70 := by
  -- Proof to be provided here
  sorry

end total_commencement_addresses_l45_45738


namespace rational_powers_implies_rational_a_rational_powers_implies_rational_b_l45_45057

open Real

theorem rational_powers_implies_rational_a (x : ℝ) :
  (∃ r₁ r₂ : ℚ, x^7 = r₁ ∧ x^12 = r₂) → (∃ q : ℚ, x = q) :=
by
  sorry

theorem rational_powers_implies_rational_b (x : ℝ) :
  (∃ r₁ r₂ : ℚ, x^9 = r₁ ∧ x^12 = r₂) → (∃ q : ℚ, x = q) :=
by
  sorry

end rational_powers_implies_rational_a_rational_powers_implies_rational_b_l45_45057


namespace simplify_sqrt_180_l45_45384

theorem simplify_sqrt_180 : sqrt 180 = 6 * sqrt 5 :=
by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  calc
    sqrt 180
      = sqrt (2^2 * 3^2 * 5)     : by rw [h]
  ... = sqrt (2^2) * sqrt (3^2) * sqrt 5 : by rw [sqrt_mul, sqrt_mul]
  ... = 2 * 3 * sqrt 5          : by rw [sqrt_sq, sqrt_sq]
  ... = 6 * sqrt 5              : by norm_num

end simplify_sqrt_180_l45_45384


namespace cylinder_surface_area_proof_l45_45395

noncomputable def sphere_volume := (500 * Real.pi) / 3
noncomputable def cylinder_base_diameter := 8
noncomputable def cylinder_surface_area := 80 * Real.pi

theorem cylinder_surface_area_proof :
  ∀ (R : ℝ) (r h : ℝ), 
    (4 * Real.pi / 3) * R^3 = (500 * Real.pi) / 3 → -- sphere volume condition
    2 * r = cylinder_base_diameter →               -- base diameter condition
    r * r + (h / 2)^2 = R^2 →                      -- Pythagorean theorem (half height)
    2 * Real.pi * r * h + 2 * Real.pi * r^2 = cylinder_surface_area := -- surface area formula
by
  intros R r h sphere_vol_cond base_diameter_cond pythagorean_cond
  sorry

end cylinder_surface_area_proof_l45_45395


namespace factorize_expr_l45_45001

theorem factorize_expr (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) :=
  sorry

end factorize_expr_l45_45001


namespace range_of_m_l45_45160

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (m x : ℝ) : ℝ := m * x + 1
noncomputable def h (x : ℝ) : ℝ := (1 / x) - (2 * Real.log x / x)

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, (x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 2)) ∧ (g m x = 2 - 2 * f x)) ↔
  (-2 * Real.exp (-3/2) ≤ m ∧ m ≤ 3 * Real.exp 1) :=
sorry

end range_of_m_l45_45160


namespace solution_set_abs_inequality_l45_45405

theorem solution_set_abs_inequality : {x : ℝ | |x - 2| < 1} = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

end solution_set_abs_inequality_l45_45405


namespace rectangular_C₁_general_C₂_intersection_and_sum_l45_45182

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

end rectangular_C₁_general_C₂_intersection_and_sum_l45_45182


namespace frustum_volume_correct_l45_45924

-- Define the base edge of the original pyramid
def base_edge_pyramid := 16

-- Define the height (altitude) of the original pyramid
def height_pyramid := 10

-- Define the base edge of the smaller pyramid after the cut
def base_edge_smaller_pyramid := 8

-- Define the function to calculate the volume of a square pyramid
def volume_square_pyramid (base_edge : ℕ) (height : ℕ) : ℚ :=
  (1 / 3) * (base_edge ^ 2) * height

-- Calculate the volume of the original pyramid
def V := volume_square_pyramid base_edge_pyramid height_pyramid

-- Calculate the volume of the smaller pyramid
def V_small := volume_square_pyramid base_edge_smaller_pyramid (height_pyramid / 2)

-- Calculate the volume of the frustum
def V_frustum := V - V_small

-- Prove that the volume of the frustum is 213.33 cubic centimeters
theorem frustum_volume_correct : V_frustum = 213.33 := by
  sorry

end frustum_volume_correct_l45_45924


namespace red_to_blue_ratio_l45_45644

theorem red_to_blue_ratio
    (total_balls : ℕ)
    (num_white_balls : ℕ)
    (num_blue_balls : ℕ)
    (num_red_balls : ℕ) :
    total_balls = 100 →
    num_white_balls = 16 →
    num_blue_balls = num_white_balls + 12 →
    num_red_balls = total_balls - (num_white_balls + num_blue_balls) →
    (num_red_balls / num_blue_balls : ℚ) = 2 :=
by
  intro h1 h2 h3 h4
  -- Proof is omitted
  sorry

end red_to_blue_ratio_l45_45644


namespace probability_of_blue_candy_l45_45230

theorem probability_of_blue_candy (green blue red : ℕ) (h1 : green = 5) (h2 : blue = 3) (h3 : red = 4) :
  (blue : ℚ) / (green + blue + red : ℚ) = 1 / 4 :=
by
  rw [h1, h2, h3]
  norm_num


end probability_of_blue_candy_l45_45230


namespace find_fraction_l45_45540

theorem find_fraction (x y : ℝ) (hx : 0 < x) (hy : x < y) (h : x / y + y / x = 8) :
  (x + y) / (x - y) = Real.sqrt 15 / 3 :=
sorry

end find_fraction_l45_45540


namespace binom_coeff_mult_l45_45674

theorem binom_coeff_mult :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_coeff_mult_l45_45674


namespace solve_number_puzzle_l45_45787

def number_puzzle (N : ℕ) : Prop :=
  (1/4) * (1/3) * (2/5) * N = 14 → (40/100) * N = 168

theorem solve_number_puzzle : ∃ N, number_puzzle N := by
  sorry

end solve_number_puzzle_l45_45787


namespace sqrt_180_simplified_l45_45375

theorem sqrt_180_simplified : Real.sqrt 180 = 6 * Real.sqrt 5 :=
   sorry

end sqrt_180_simplified_l45_45375


namespace proportional_function_decreases_l45_45595

-- Define the function y = -2x
def proportional_function (x : ℝ) : ℝ := -2 * x

-- State the theorem to prove that y decreases as x increases
theorem proportional_function_decreases (x y : ℝ) (h : y = proportional_function x) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → proportional_function x₁ > proportional_function x₂ := 
sorry

end proportional_function_decreases_l45_45595


namespace min_value_l45_45194

theorem min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 3) : 
  ∃ c : ℝ, (c = 3 / 4) ∧ (∀ (a b c : ℝ), a = x ∧ b = y ∧ c = z → 
    (1/(a + 3*b) + 1/(b + 3*c) + 1/(c + 3*a)) ≥ c) :=
sorry

end min_value_l45_45194


namespace volume_of_sphere_l45_45024

theorem volume_of_sphere
    (area1 : ℝ) (area2 : ℝ) (distance : ℝ)
    (h1 : area1 = 9 * π)
    (h2 : area2 = 16 * π)
    (h3 : distance = 1) :
    ∃ R : ℝ, (4 / 3) * π * R ^ 3 = 500 * π / 3 :=
by
  sorry

end volume_of_sphere_l45_45024


namespace min_value_of_fraction_sum_l45_45487

theorem min_value_of_fraction_sum (a b : ℤ) (h1 : a = b + 1) : 
  (a > b) -> (∃ x, x > 0 ∧ ((a + b) / (a - b) + (a - b) / (a + b)) = 2) :=
by
  sorry

end min_value_of_fraction_sum_l45_45487


namespace acceptable_arrangements_correct_l45_45351

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n + 1) * factorial n

-- Define the total number of people
def total_people := 8

-- Calculate the total arrangements of 8 people
def total_arrangements := factorial total_people

-- Calculate the arrangements where Alice and Bob are together
def reduced_people := total_people - 1
def alice_bob_arrangements := factorial reduced_people * factorial 2

-- Calculate the acceptable arrangements where Alice and Bob are not together
def acceptable_arrangements := total_arrangements - alice_bob_arrangements

-- The theorem statement, asserting the correct answer
theorem acceptable_arrangements_correct : acceptable_arrangements = 30240 :=
by
  sorry

end acceptable_arrangements_correct_l45_45351


namespace max_digit_product_l45_45055

theorem max_digit_product (N : ℕ) (digits : List ℕ) (h1 : 0 < N) (h2 : digits.sum = 23) (h3 : digits.prod < 433) : 
  digits.prod ≤ 432 :=
sorry

end max_digit_product_l45_45055


namespace distance_between_points_l45_45706

theorem distance_between_points :
  let x1 := 1
  let y1 := 3
  let z1 := 2
  let x2 := 4
  let y2 := 1
  let z2 := 6
  let distance : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
  distance = Real.sqrt 29 := by
  sorry

end distance_between_points_l45_45706


namespace xiao_hua_spent_7_yuan_l45_45262

theorem xiao_hua_spent_7_yuan :
  ∃ (a b c d: ℕ), a + b + c + d = 30 ∧
                   ((a = 5 ∧ b = 5 ∧ c = 10 ∧ d = 10) ∨
                    (a = 5 ∧ b = 10 ∧ c = 5 ∧ d = 10) ∨
                    (a = 10 ∧ b = 5 ∧ c = 5 ∧ d = 10) ∨
                    (a = 10 ∧ b = 10 ∧ c = 5 ∧ d = 5) ∨
                    (a = 5 ∧ b = 10 ∧ c = 10 ∧ d = 5) ∨
                    (a = 10 ∧ b = 5 ∧ c = 10 ∧ d = 5)) ∧
                   10 * c + 15 * a + 25 * b + 40 * d = 700 :=
by {
  sorry
}

end xiao_hua_spent_7_yuan_l45_45262


namespace mary_candies_l45_45354

-- The conditions
def bob_candies : Nat := 10
def sue_candies : Nat := 20
def john_candies : Nat := 5
def sam_candies : Nat := 10
def total_candies : Nat := 50

-- The theorem to prove
theorem mary_candies :
  total_candies - (bob_candies + sue_candies + john_candies + sam_candies) = 5 := by
  -- Here is where the proof would go; currently using sorry to skip the proof
  sorry

end mary_candies_l45_45354


namespace bird_families_to_Asia_l45_45232

theorem bird_families_to_Asia (total_families initial_families left_families went_to_Africa went_to_Asia: ℕ) 
  (h1 : total_families = 85) 
  (h2 : went_to_Africa = 23) 
  (h3 : left_families = 25) 
  (h4 : went_to_Asia = total_families - left_families - went_to_Africa) 
  : went_to_Asia = 37 := 
by 
  rw [h1, h2, h3] at h4 
  simp at h4 
  exact h4

end bird_families_to_Asia_l45_45232


namespace no_hot_dogs_l45_45795

def hamburgers_initial := 9.0
def hamburgers_additional := 3.0
def hamburgers_total := 12.0

theorem no_hot_dogs (h1 : hamburgers_initial + hamburgers_additional = hamburgers_total) : 0 = 0 :=
by
  sorry

end no_hot_dogs_l45_45795


namespace lemonade_total_difference_is_1860_l45_45878

-- Define the conditions
def stanley_rate : Nat := 4
def stanley_price : Real := 1.50

def carl_rate : Nat := 7
def carl_price : Real := 1.30

def lucy_rate : Nat := 5
def lucy_price : Real := 1.80

def hours : Nat := 3

-- Compute the total amounts for each sibling
def stanley_total : Real := stanley_rate * hours * stanley_price
def carl_total : Real := carl_rate * hours * carl_price
def lucy_total : Real := lucy_rate * hours * lucy_price

-- Compute the individual differences
def diff_stanley_carl : Real := carl_total - stanley_total
def diff_stanley_lucy : Real := lucy_total - stanley_total
def diff_carl_lucy : Real := carl_total - lucy_total

-- Sum the differences
def total_difference : Real := diff_stanley_carl + diff_stanley_lucy + diff_carl_lucy

-- The proof statement
theorem lemonade_total_difference_is_1860 :
  total_difference = 18.60 :=
by
  sorry

end lemonade_total_difference_is_1860_l45_45878


namespace pipes_fill_tank_in_7_minutes_l45_45254

theorem pipes_fill_tank_in_7_minutes (T : ℕ) (R_A R_B R_combined : ℚ) 
  (h1 : R_A = 1 / 56) 
  (h2 : R_B = 7 * R_A)
  (h3 : R_combined = R_A + R_B)
  (h4 : T = 1 / R_combined) : 
  T = 7 := by 
  sorry

end pipes_fill_tank_in_7_minutes_l45_45254


namespace smallest_n_for_congruence_l45_45620

theorem smallest_n_for_congruence :
  ∃ n : ℕ, 827 * n % 36 = 1369 * n % 36 ∧ n > 0 ∧ (∀ m : ℕ, 827 * m % 36 = 1369 * m % 36 ∧ m > 0 → m ≥ 18) :=
by sorry

end smallest_n_for_congruence_l45_45620


namespace find_s_l45_45743

def is_monic_cubic (p : Polynomial ℝ) : Prop :=
  p.degree = 3 ∧ p.leadingCoeff = 1

def has_roots (p : Polynomial ℝ) (roots : Set ℝ) : Prop :=
  ∀ x ∈ roots, p.eval x = 0

def poly_condition (f g : Polynomial ℝ) (s : ℝ) : Prop :=
  ∀ x : ℝ, f.eval x - g.eval x = 2 * s

theorem find_s (s : ℝ)
  (f g : Polynomial ℝ)
  (hf_monic : is_monic_cubic f)
  (hg_monic : is_monic_cubic g)
  (hf_roots : has_roots f {s + 2, s + 6})
  (hg_roots : has_roots g {s + 4, s + 10})
  (h_condition : poly_condition f g s) :
  s = 10.67 :=
sorry

end find_s_l45_45743


namespace intersection_of_lines_l45_45707

-- Definitions for the lines given by their equations
def line1 (x y : ℝ) : Prop := 5 * x - 3 * y = 9
def line2 (x y : ℝ) : Prop := x^2 + 4 * x - y = 10

-- The statement to prove
theorem intersection_of_lines :
  (line1 2 (1 / 3) ∧ line2 2 (1 / 3)) ∨ (line1 (-3.5) (-8.83) ∧ line2 (-3.5) (-8.83)) :=
by
  sorry

end intersection_of_lines_l45_45707


namespace sum_squares_and_products_of_nonneg_reals_l45_45344

theorem sum_squares_and_products_of_nonneg_reals {x y z : ℝ} 
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) 
  (h2 : x*y + y*z + z*x = 27) : 
  x + y + z = Real.sqrt 106 := 
by 
  sorry

end sum_squares_and_products_of_nonneg_reals_l45_45344


namespace binomial_product_l45_45692

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := by
  sorry

end binomial_product_l45_45692


namespace company_ordered_weight_of_stone_l45_45120

theorem company_ordered_weight_of_stone :
  let weight_concrete := 0.16666666666666666
  let weight_bricks := 0.16666666666666666
  let total_material := 0.8333333333333334
  let weight_stone := total_material - (weight_concrete + weight_bricks)
  weight_stone = 0.5 :=
by
  sorry

end company_ordered_weight_of_stone_l45_45120


namespace nina_total_spending_l45_45868

-- Defining the quantities and prices of each category of items
def num_toys : Nat := 3
def price_per_toy : Nat := 10

def num_basketball_cards : Nat := 2
def price_per_card : Nat := 5

def num_shirts : Nat := 5
def price_per_shirt : Nat := 6

-- Calculating the total cost for each category
def cost_toys : Nat := num_toys * price_per_toy
def cost_cards : Nat := num_basketball_cards * price_per_card
def cost_shirts : Nat := num_shirts * price_per_shirt

-- Calculating the total amount spent
def total_cost : Nat := cost_toys + cost_cards + cost_shirts

-- The final theorem statement to verify the answer
theorem nina_total_spending : total_cost = 70 :=
by
  sorry

end nina_total_spending_l45_45868


namespace x_intercept_of_line_l45_45651

variables (x₁ y₁ x₂ y₂ : ℝ) (m : ℝ)

/-- The line passing through the points (-1, 1) and (3, 9) has an x-intercept of -3/2. -/
theorem x_intercept_of_line : 
  let x₁ := -1
  let y₁ := 1
  let x₂ := 3
  let y₂ := 9
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (0 : ℝ) = m * (x : ℝ) + b → x = (-3 / 2) := 
by 
  sorry

end x_intercept_of_line_l45_45651


namespace solve_quadratic_eqn_l45_45208

theorem solve_quadratic_eqn : ∀ (x : ℝ), x^2 - 4 * x - 3 = 0 ↔ (x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) :=
by
  sorry

end solve_quadratic_eqn_l45_45208


namespace range_of_a_l45_45219

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 2 * a * x + 3 ≤ 0) ↔ (a ∈ Set.Iic 0 ∪ Set.Ici 3) := 
sorry

end range_of_a_l45_45219


namespace binomial_product_result_l45_45679

-- Defining the combination (binomial coefficient) formula
def combination (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- Lean theorem statement to prove the problem
theorem binomial_product_result : combination 10 3 * combination 8 3 = 6720 := by
  sorry

end binomial_product_result_l45_45679


namespace inequality_proof_l45_45315

theorem inequality_proof (a b c d e f : ℝ) (H : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end inequality_proof_l45_45315


namespace potato_cost_l45_45410

variables (x : ℝ)
variables (b a : ℝ)

def andrey_earnings (x : ℝ) : ℝ := 120 * x
def boris_earnings (x : ℝ) : ℝ := 124.8 * x

theorem potato_cost :
  (boris_earnings x) - (andrey_earnings x) = 1200 → x = 250 :=
  by
    unfold andrey_earnings
    unfold boris_earnings
    sorry

end potato_cost_l45_45410


namespace triangle_angles_inequality_l45_45048

theorem triangle_angles_inequality (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h_sum : A + B + C = Real.pi) :
  1 / Real.sin (A / 2) + 1 / Real.sin (B / 2) + 1 / Real.sin (C / 2) ≥ 6 := 
sorry

end triangle_angles_inequality_l45_45048


namespace calculate_total_cups_l45_45890

variable (butter : ℕ) (flour : ℕ) (sugar : ℕ) (total_cups : ℕ)

def ratio_condition : Prop :=
  3 * butter = 2 * sugar ∧ 3 * flour = 5 * sugar

def sugar_condition : Prop :=
  sugar = 9

def total_cups_calculation : Prop :=
  total_cups = butter + flour + sugar

theorem calculate_total_cups (h1 : ratio_condition butter flour sugar) (h2 : sugar_condition sugar) :
  total_cups_calculation butter flour sugar total_cups -> total_cups = 30 := by
  sorry

end calculate_total_cups_l45_45890


namespace tan_alpha_l45_45723

theorem tan_alpha (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) 
  (hα3 : sin(α)^2 + cos(2 * α) = 3 / 4) : tan(α) = sqrt(3) / 3 :=
by
sorry

end tan_alpha_l45_45723


namespace count_less_than_threshold_is_zero_l45_45732

def numbers := [0.8, 0.5, 0.9]
def threshold := 0.4

theorem count_less_than_threshold_is_zero :
  (numbers.filter (λ x => x < threshold)).length = 0 :=
by
  sorry

end count_less_than_threshold_is_zero_l45_45732


namespace narrow_black_stripes_are_8_l45_45574

-- Define variables: w for wide black stripes, n for narrow black stripes, b for white stripes
variables (w n b : ℕ)

-- Given conditions
axiom cond1 : b = w + 7
axiom cond2 : w + n = b + 1

-- Theorem statement to prove that the number of narrow black stripes is 8
theorem narrow_black_stripes_are_8 : n = 8 :=
by sorry

end narrow_black_stripes_are_8_l45_45574


namespace chef_potatoes_l45_45916

theorem chef_potatoes (total_potatoes cooked_potatoes time_per_potato rest_time: ℕ)
  (h1 : total_potatoes = 15)
  (h2 : time_per_potato = 9)
  (h3 : rest_time = 63)
  (h4 : time_per_potato * (total_potatoes - cooked_potatoes) = rest_time) :
  cooked_potatoes = 8 :=
by sorry

end chef_potatoes_l45_45916


namespace mirella_read_more_pages_l45_45587

-- Define the number of books Mirella read
def num_purple_books := 8
def num_orange_books := 7
def num_blue_books := 5

-- Define the number of pages per book for each color
def pages_per_purple_book := 320
def pages_per_orange_book := 640
def pages_per_blue_book := 450

-- Calculate the total pages for each color
def total_purple_pages := num_purple_books * pages_per_purple_book
def total_orange_pages := num_orange_books * pages_per_orange_book
def total_blue_pages := num_blue_books * pages_per_blue_book

-- Calculate the combined total of orange and blue pages
def total_orange_blue_pages := total_orange_pages + total_blue_pages

-- Define the target value
def page_difference := 4170

-- State the theorem to prove
theorem mirella_read_more_pages :
  total_orange_blue_pages - total_purple_pages = page_difference := by
  sorry

end mirella_read_more_pages_l45_45587


namespace quadratic_rewriting_l45_45949

theorem quadratic_rewriting:
  ∃ (d e f : ℤ), (∀ x : ℝ, 4 * x^2 - 28 * x + 49 = (d * x + e)^2 + f) ∧ d * e = -14 :=
by {
  sorry
}

end quadratic_rewriting_l45_45949


namespace sufficient_but_not_necessary_not_necessary_l45_45788

theorem sufficient_but_not_necessary (x y : ℝ) (h : x < y ∧ y < 0) : x^2 > y^2 :=
by {
  -- a Lean 4 proof can be included here if desired
  sorry
}

theorem not_necessary (x y : ℝ) (h : x^2 > y^2) : ¬ (x < y ∧ y < 0) :=
by {
  -- a Lean 4 proof can be included here if desired
  sorry
}

end sufficient_but_not_necessary_not_necessary_l45_45788


namespace remainder_when_200_divided_by_k_l45_45478

theorem remainder_when_200_divided_by_k 
  (k : ℕ) (k_pos : 0 < k)
  (h : 120 % k^2 = 12) :
  200 % k = 2 :=
sorry

end remainder_when_200_divided_by_k_l45_45478


namespace different_language_classes_probability_l45_45531

theorem different_language_classes_probability :
  let total_students := 40
  let french_students := 28
  let spanish_students := 26
  let german_students := 15
  let french_and_spanish_students := 10
  let french_and_german_students := 6
  let spanish_and_german_students := 8
  let all_three_languages_students := 3
  let total_pairs := Nat.choose total_students 2
  let french_only := french_students - (french_and_spanish_students + french_and_german_students - all_three_languages_students) - all_three_languages_students
  let spanish_only := spanish_students - (french_and_spanish_students + spanish_and_german_students - all_three_languages_students) - all_three_languages_students
  let german_only := german_students - (french_and_german_students + spanish_and_german_students - all_three_languages_students) - all_three_languages_students
  let french_only_pairs := Nat.choose french_only 2
  let spanish_only_pairs := Nat.choose spanish_only 2
  let german_only_pairs := Nat.choose german_only 2
  let single_language_pairs := french_only_pairs + spanish_only_pairs + german_only_pairs
  let different_classes_probability := 1 - (single_language_pairs / total_pairs)
  different_classes_probability = (34 / 39) :=
by
  sorry

end different_language_classes_probability_l45_45531


namespace number_of_narrow_black_stripes_l45_45584

-- Define the variables
variables {w n b : ℕ}

-- The conditions from the problem
def condition1 := w + n = b + 1
def condition2 := b = w + 7

-- The Lean statement to prove
theorem number_of_narrow_black_stripes (h1 : condition1) (h2 : condition2) : n = 8 :=
by {
  -- We import the conditions as hypotheses
  sorry
}

end number_of_narrow_black_stripes_l45_45584


namespace smallest_repeating_block_of_fraction_l45_45332

theorem smallest_repeating_block_of_fraction (a b : ℕ) (h : a = 8 ∧ b = 11) :
  ∃ n : ℕ, n = 2 ∧ decimal_expansion_repeating_block_length (a / b) = n := by
  sorry

end smallest_repeating_block_of_fraction_l45_45332


namespace simplify_and_evaluate_l45_45876

theorem simplify_and_evaluate (m : ℝ) (h_root : m^2 + 3 * m - 2 = 0) :
  (m - 3) / (3 * m^2 - 6 * m) / (m + 2 - 5 / (m - 2)) = 1 / 6 :=
by
  sorry

end simplify_and_evaluate_l45_45876


namespace rate_of_interest_l45_45340

theorem rate_of_interest (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) (h : P > 0 ∧ T = 7 ∧ SI = P / 5 ∧ SI = (P * R * T) / 100) : 
  R = 20 / 7 := 
by
  sorry

end rate_of_interest_l45_45340


namespace savings_on_cheapest_flight_l45_45667

theorem savings_on_cheapest_flight :
  let delta_price := 850
  let delta_discount := 0.20
  let united_price := 1100
  let united_discount := 0.30
  let delta_final_price := delta_price - delta_price * delta_discount
  let united_final_price := united_price - united_price * united_discount
  delta_final_price < united_final_price →
  united_final_price - delta_final_price = 90 :=
by
  sorry

end savings_on_cheapest_flight_l45_45667


namespace line_equation_l45_45337

/-
Given points M(2, 3) and N(4, -5), and a line l passes through the 
point P(1, 2). Prove that the line l has equal distances from points 
M and N if and only if its equation is either 4x + y - 6 = 0 or 
3x + 2y - 7 = 0.
-/

theorem line_equation (M N P : ℝ × ℝ)
(hM : M = (2, 3))
(hN : N = (4, -5))
(hP : P = (1, 2))
(l : ℝ → ℝ → Prop)
(h_l : ∀ x y, l x y ↔ (4 * x + y - 6 = 0 ∨ 3 * x + 2 * y - 7 = 0))
: ∀ (dM dN : ℝ), 
(∀ x y , l x y → (x = 1) → (y = 2) ∧ (|M.1 - x| + |M.2 - y| = |N.1 - x| + |N.2 - y|)) :=
sorry

end line_equation_l45_45337


namespace arrangements_of_6_books_l45_45871

theorem arrangements_of_6_books : ∃ (n : ℕ), n = 720 ∧ n = Nat.factorial 6 :=
by
  use 720
  constructor
  · rfl
  · sorry

end arrangements_of_6_books_l45_45871


namespace evaluate_expression_l45_45140

theorem evaluate_expression : ((3^4)^3 + 5) - ((4^3)^4 + 5) = -16245775 := by
  sorry

end evaluate_expression_l45_45140


namespace ellipse_focal_length_l45_45603

theorem ellipse_focal_length :
  ∀ a b c : ℝ, (a^2 = 11) → (b^2 = 3) → (c^2 = a^2 - b^2) → (2 * c = 4 * Real.sqrt 2) :=
by
  sorry

end ellipse_focal_length_l45_45603


namespace reflected_parabola_equation_l45_45085

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := x^2

-- Define the line of reflection
def reflection_line (x : ℝ) : ℝ := x + 2

-- The reflected equation statement to be proved
theorem reflected_parabola_equation (x y : ℝ) :
  (parabola x = y) ∧ (reflection_line x = y) →
  (∃ y' x', x = y'^2 - 4 * y' + 2 ∧ y = x' + 2 ∧ x' = y - 2) :=
sorry

end reflected_parabola_equation_l45_45085


namespace circle_tangent_line_k_range_l45_45719

theorem circle_tangent_line_k_range
  (k : ℝ)
  (P Q : ℝ × ℝ)
  (c : ℝ × ℝ := (0, 1)) -- Circle center
  (r : ℝ := 1) -- Circle radius
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 - 2 * y = 0)
  (line_eq : ∀ (x y : ℝ), k * x + y + 3 = 0)
  (dist_pq : Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2) = Real.sqrt 3) :
  k ∈ Set.Iic (-Real.sqrt 3) ∪ Set.Ici (Real.sqrt 3) :=
by
  sorry

end circle_tangent_line_k_range_l45_45719


namespace necessary_but_not_sufficient_l45_45970

theorem necessary_but_not_sufficient (x : ℝ) :
  (x^2 < x) → ((x^2 < x) ↔ (0 < x ∧ x < 1)) ∧ ((1/x > 2) ↔ (0 < x ∧ x < 1/2)) := 
by 
  sorry

end necessary_but_not_sufficient_l45_45970


namespace non_adjacent_arrangements_l45_45347

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem non_adjacent_arrangements : 
  let total_arrangements := factorial 8
  let adjacent_arrangements := factorial 7 * factorial 2
  total_arrangements - adjacent_arrangements = 30240 := by
sorry

end non_adjacent_arrangements_l45_45347


namespace sum_of_three_quadratics_no_rot_l45_45739

def quad_poly_sum_no_root (p q : ℝ -> ℝ) : Prop :=
  ∀ x : ℝ, (p x + q x ≠ 0)

theorem sum_of_three_quadratics_no_rot (a b c d e f : ℝ)
    (h1 : quad_poly_sum_no_root (λ x => x^2 + a*x + b) (λ x => x^2 + c*x + d))
    (h2 : quad_poly_sum_no_root (λ x => x^2 + c*x + d) (λ x => x^2 + e*x + f))
    (h3 : quad_poly_sum_no_root (λ x => x^2 + e*x + f) (λ x => x^2 + a*x + b)) :
    quad_poly_sum_no_root (λ x => x^2 + a*x + b) 
                         (λ x => x^2 + c*x + d + x^2 + e*x + f) :=
sorry

end sum_of_three_quadratics_no_rot_l45_45739


namespace find_missing_score_l45_45148

theorem find_missing_score
  (scores : List ℕ)
  (h_scores : scores = [73, 83, 86, 73, x])
  (mean : ℚ)
  (h_mean : mean = 79.2)
  (h_length : scores.length = 5)
  : x = 81 := by
  sorry

end find_missing_score_l45_45148


namespace direction_vector_of_projection_l45_45607

-- Define the projection matrix
def projection_matrix : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![  3/17, -2/17, -1/3],
    ![-2/17,  1/17,  1/6],
    ![-1/3,   1/6,   5/6]]

-- Define the condition that a, b, c should be integers, a > 0, and gcd(|a|, |b|, |c|) = 1
structure DirectionVector (a b c : ℤ) : Prop :=
  (a_pos : a > 0)
  (gcd_cond : Int.gcd (Int.natAbs a) (Int.gcd (Int.natAbs b) (Int.natAbs c)) = 1)

-- The theorem stating the direction vector
theorem direction_vector_of_projection : ∃ (a b c : ℤ), 
  DirectionVector a b c ∧ 
  ![(a : ℚ), (b : ℚ), (c : ℚ)] ∝ 
  ![ 3, -2, -5] := by 
sorry

end direction_vector_of_projection_l45_45607


namespace sqrt_180_simplify_l45_45382

theorem sqrt_180_simplify : real.sqrt 180 = 6 * real.sqrt 5 := by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  rw [h, real.sqrt_mul, real.sqrt_mul, real.sqrt_mul]
  all_goals { norm_num }
  sorry

end sqrt_180_simplify_l45_45382


namespace units_digit_divisible_by_18_l45_45425

theorem units_digit_divisible_by_18 : ∃ n : ℕ, (3150 ≤ 315 * n) ∧ (315 * n < 3160) ∧ (n % 2 = 0) ∧ (315 * n % 18 = 0) ∧ (n = 0) :=
by
  use 0
  sorry

end units_digit_divisible_by_18_l45_45425


namespace expression_value_l45_45245

/--
Prove that for a = 51 and b = 15, the expression (a + b)^2 - (a^2 + b^2) equals 1530.
-/
theorem expression_value (a b : ℕ) (h1 : a = 51) (h2 : b = 15) : (a + b)^2 - (a^2 + b^2) = 1530 := by
  rw [h1, h2]
  sorry

end expression_value_l45_45245


namespace john_twice_james_l45_45993

def john_age : ℕ := 39
def years_ago : ℕ := 3
def years_future : ℕ := 6
def age_difference : ℕ := 4

theorem john_twice_james {J : ℕ} (h : 39 - years_ago = 2 * (J + years_future)) : 
  (J + age_difference = 16) :=
by
  sorry  -- Proof steps here

end john_twice_james_l45_45993


namespace narrow_black_stripes_l45_45563

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end narrow_black_stripes_l45_45563


namespace value_of_ab_plus_bc_plus_ca_l45_45485

theorem value_of_ab_plus_bc_plus_ca (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end value_of_ab_plus_bc_plus_ca_l45_45485


namespace avg_score_false_iff_unequal_ints_l45_45816

variable {a b m n : ℕ}

theorem avg_score_false_iff_unequal_ints 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (m_pos : 0 < m) 
  (n_pos : 0 < n) 
  (m_neq_n : m ≠ n) : 
  (∃ a b, (ma + nb) / (m + n) = (a + b)/2) ↔ a ≠ b := 
sorry

end avg_score_false_iff_unequal_ints_l45_45816


namespace num_true_propositions_l45_45846

theorem num_true_propositions : 
  (∀ (a b : ℝ), a = 0 → ab = 0) ∧
  (∀ (a b : ℝ), ab ≠ 0 → a ≠ 0) ∧
  ¬ (∀ (a b : ℝ), ab = 0 → a = 0) ∧
  ¬ (∀ (a b : ℝ), a ≠ 0 → ab ≠ 0) → 
  2 = 2 :=
by 
  sorry

end num_true_propositions_l45_45846


namespace dan_bought_one_candy_bar_l45_45279

-- Define the conditions
def initial_money : ℕ := 4
def cost_per_candy_bar : ℕ := 3
def money_left : ℕ := 1

-- Define the number of candy bars Dan bought
def number_of_candy_bars_bought : ℕ := (initial_money - money_left) / cost_per_candy_bar

-- Prove the number of candy bars bought is equal to 1
theorem dan_bought_one_candy_bar : number_of_candy_bars_bought = 1 := by
  sorry

end dan_bought_one_candy_bar_l45_45279


namespace maximize_value_l45_45060

noncomputable def maximum_value (x y : ℝ) : ℝ :=
  3 * x - 2 * y

theorem maximize_value (x y : ℝ) (h : x^2 + y^2 + x * y = 1) : maximum_value x y ≤ 5 :=
sorry

end maximize_value_l45_45060


namespace binomial_product_l45_45691

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := by
  sorry

end binomial_product_l45_45691


namespace narrow_black_stripes_count_l45_45560

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end narrow_black_stripes_count_l45_45560


namespace remainder_div_5_l45_45113

theorem remainder_div_5 (n : ℕ): (∃ k : ℤ, n = 10 * k + 7) → (∃ m : ℤ, n = 5 * m + 2) :=
by
  sorry

end remainder_div_5_l45_45113


namespace seashells_after_giving_away_l45_45856

-- Define the given conditions
def initial_seashells : ℕ := 79
def given_away_seashells : ℕ := 63

-- State the proof problem
theorem seashells_after_giving_away : (initial_seashells - given_away_seashells) = 16 :=
  by 
    sorry

end seashells_after_giving_away_l45_45856


namespace sqrt_180_eq_l45_45388

noncomputable def simplify_sqrt_180 : Real := 6 * Real.sqrt 5

theorem sqrt_180_eq : Real.sqrt 180 = simplify_sqrt_180 := 
by
  -- proof omitted
  sorry

end sqrt_180_eq_l45_45388


namespace narrow_black_stripes_l45_45545

variable (w n b : ℕ)

theorem narrow_black_stripes (w : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := by
  have h3 : w + n = (w + 7) + 1 := by rw [h1]
  have h4 : w + n = w + 8 := by linarith
  have h5 : n = 8 := by linarith
  exact h5

end narrow_black_stripes_l45_45545


namespace intersection_with_x_axis_intersection_with_y_axis_l45_45181

theorem intersection_with_x_axis (x y : ℝ) : y = -2 * x + 4 ∧ y = 0 ↔ x = 2 ∧ y = 0 := by
  sorry

theorem intersection_with_y_axis (x y : ℝ) : y = -2 * x + 4 ∧ x = 0 ↔ x = 0 ∧ y = 4 := by
  sorry

end intersection_with_x_axis_intersection_with_y_axis_l45_45181


namespace two_times_sum_of_squares_l45_45593

theorem two_times_sum_of_squares (P a b : ℤ) (h : P = a^2 + b^2) : 
  ∃ x y : ℤ, 2 * P = x^2 + y^2 := 
by 
  sorry

end two_times_sum_of_squares_l45_45593


namespace find_m_l45_45713

theorem find_m (x m : ℝ) (h_eq : (x + m) / (x - 2) + 1 / (2 - x) = 3) (h_root : x = 2) : m = -1 :=
by
  sorry

end find_m_l45_45713


namespace find_a_and_b_l45_45768

-- Define the two numbers a and b and the given conditions
variables (a b : ℕ)
variables (h1 : a - b = 831) (h2 : a = 21 * b + 11)

-- State the theorem to find the values of a and b
theorem find_a_and_b (a b : ℕ) (h1 : a - b = 831) (h2 : a = 21 * b + 11) : a = 872 ∧ b = 41 :=
by
  sorry

end find_a_and_b_l45_45768


namespace find_angle_B_find_area_of_ABC_l45_45185

noncomputable def angle_B (a b c : ℝ) (C : ℝ) (h1 : 2 * b * Real.cos C = 2 * a + c) : ℝ := 
  if b * Real.cos C = -a then Real.pi - 2 * Real.arctan (a / c)
  else 2 * Real.pi / 3

theorem find_angle_B (a b c : ℝ) (C : ℝ) (h1 : 2 * b * Real.cos C = 2 * a + c) :
  angle_B a b c C h1 = 2 * Real.pi / 3 := 
sorry

noncomputable def area_of_ABC (a b c : ℝ) (C B : ℝ) (d : ℝ) (position : ℕ) (h1 : 2 * b * Real.cos C = 2 * a + c) (h2 : b = 2 * Real.sqrt 3) (h3 : d = 1) : ℝ :=
  if position = 1 then /- calculation for BD bisector case -/ (a * c / 2) * Real.sin (2 * Real.pi / 3)
  else /- calculation for midpoint case -/ (a * c / 2) * Real.sin (2 * Real.pi / 3)

theorem find_area_of_ABC (a b c : ℝ) (C B : ℝ) (d : ℝ) (position : ℕ) (h1 : 2 * b * Real.cos C = 2 * a + c) (h2 : b = 2 * Real.sqrt 3) (h3 : d = 1) (hB : angle_B a b c C h1 = 2 * Real.pi / 3) :
  area_of_ABC a b c C (2 * Real.pi / 3) d position h1 h2 h3 = Real.sqrt 3 := 
sorry

end find_angle_B_find_area_of_ABC_l45_45185


namespace determine_compound_impossible_l45_45264

-- Define the conditions
def contains_Cl (compound : Type) : Prop := true -- Placeholder definition
def mass_percentage_Cl (compound : Type) : ℝ := 0 -- Placeholder definition

-- Define the main statement
theorem determine_compound_impossible (compound : Type) 
  (containsCl : contains_Cl compound) 
  (massPercentageCl : mass_percentage_Cl compound = 47.3) : 
  ∃ (distinct_element : Type), compound = distinct_element := 
sorry

end determine_compound_impossible_l45_45264


namespace average_sleep_per_day_l45_45328

-- Define a structure for time duration
structure TimeDuration where
  hours : ℕ
  minutes : ℕ

-- Define instances for each day
def mondayNight : TimeDuration := ⟨8, 15⟩
def mondayNap : TimeDuration := ⟨0, 30⟩
def tuesdayNight : TimeDuration := ⟨7, 45⟩
def tuesdayNap : TimeDuration := ⟨0, 45⟩
def wednesdayNight : TimeDuration := ⟨8, 10⟩
def wednesdayNap : TimeDuration := ⟨0, 50⟩
def thursdayNight : TimeDuration := ⟨10, 25⟩
def thursdayNap : TimeDuration := ⟨0, 20⟩
def fridayNight : TimeDuration := ⟨7, 50⟩
def fridayNap : TimeDuration := ⟨0, 40⟩

-- Function to convert TimeDuration to total minutes
def totalMinutes (td : TimeDuration) : ℕ :=
  td.hours * 60 + td.minutes

-- Define the total sleep time for each day
def mondayTotal := totalMinutes mondayNight + totalMinutes mondayNap
def tuesdayTotal := totalMinutes tuesdayNight + totalMinutes tuesdayNap
def wednesdayTotal := totalMinutes wednesdayNight + totalMinutes wednesdayNap
def thursdayTotal := totalMinutes thursdayNight + totalMinutes thursdayNap
def fridayTotal := totalMinutes fridayNight + totalMinutes fridayNap

-- Sum of all sleep times
def totalSleep := mondayTotal + tuesdayTotal + wednesdayTotal + thursdayTotal + fridayTotal
-- Average sleep in minutes per day
def averageSleep := totalSleep / 5
-- Convert average sleep in total minutes back to hours and minutes
def averageHours := averageSleep / 60
def averageMinutes := averageSleep % 60

theorem average_sleep_per_day :
  averageHours = 9 ∧ averageMinutes = 6 := by
  sorry

end average_sleep_per_day_l45_45328


namespace range_of_a_l45_45541

open Real

-- Definitions based on given conditions
def p (a : ℝ) : Prop := a > 2
def q (a : ℝ) : Prop := ∀ (x : ℝ), x > 0 → -3^x ≤ a

-- The main proposition combining the conditions
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) → -1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l45_45541


namespace freeRangingChickens_l45_45407

-- Define the number of chickens in the coop
def chickensInCoop : Nat := 14

-- Define the number of chickens in the run
def chickensInRun : Nat := 2 * chickensInCoop

-- Define the number of chickens free ranging
def chickensFreeRanging : Nat := 2 * chickensInRun - 4

-- State the theorem
theorem freeRangingChickens : chickensFreeRanging = 52 := by
  -- We cannot provide the proof, so we use sorry
  sorry

end freeRangingChickens_l45_45407


namespace Owen_final_turtle_count_l45_45071

variable (Owen_turtles : ℕ) (Johanna_turtles : ℕ)

def final_turtles (Owen_turtles Johanna_turtles : ℕ) : ℕ :=
  let initial_Owen_turtles := Owen_turtles
  let initial_Johanna_turtles := Owen_turtles - 5
  let Owen_after_month := initial_Owen_turtles * 2
  let Johanna_after_losing_half := initial_Johanna_turtles / 2
  let Owen_after_donation := Owen_after_month + Johanna_after_losing_half
  Owen_after_donation

theorem Owen_final_turtle_count : final_turtles 21 (21 - 5) = 50 :=
by
  sorry

end Owen_final_turtle_count_l45_45071


namespace additional_plates_correct_l45_45210

-- Define the conditions
def original_set_1 : Finset Char := {'B', 'F', 'J', 'N', 'T'}
def original_set_2 : Finset Char := {'E', 'U'}
def original_set_3 : Finset Char := {'G', 'K', 'R', 'Z'}

-- Define the sizes of the original sets
def size_set_1 := (original_set_1.card : Nat) -- 5
def size_set_2 := (original_set_2.card : Nat) -- 2
def size_set_3 := (original_set_3.card : Nat) -- 4

-- Sizes after adding new letters
def new_size_set_1 := size_set_1 + 1 -- 6
def new_size_set_2 := size_set_2 + 1 -- 3
def new_size_set_3 := size_set_3 + 1 -- 5

-- Calculate the original and new number of plates
def original_plates : Nat := size_set_1 * size_set_2 * size_set_3 -- 5 * 2 * 4 = 40
def new_plates : Nat := new_size_set_1 * new_size_set_2 * new_size_set_3 -- 6 * 3 * 5 = 90

-- Calculate the additional plates
def additional_plates : Nat := new_plates - original_plates -- 90 - 40 = 50

-- The proof statement
theorem additional_plates_correct : additional_plates = 50 :=
by
  -- Proof can be filled in here
  sorry

end additional_plates_correct_l45_45210


namespace ship_passengers_percentage_l45_45755

variables (P R : ℝ)

-- Conditions
def condition1 : Prop := (0.20 * P) = (0.60 * R)

-- Target
def target : Prop := R / P = 1 / 3

theorem ship_passengers_percentage
  (h1 : condition1 P R) :
  target P R :=
by
  sorry

end ship_passengers_percentage_l45_45755


namespace find_m_value_l45_45272

theorem find_m_value :
  let x_values := [8, 9.5, m, 10.5, 12]
  let y_values := [16, 10, 8, 6, 5]
  let regression_eq (x : ℝ) := -3.5 * x + 44
  let avg (l : List ℝ) := l.sum / l.length
  avg y_values = 9 →
  avg x_values = (40 + m) / 5 →
  9 = regression_eq (avg x_values) →
  m = 10 :=
by
  sorry

end find_m_value_l45_45272


namespace price_without_and_with_coupon_l45_45920

theorem price_without_and_with_coupon
  (commission_rate sale_tax_rate discount_rate : ℝ)
  (cost producer_price shipping_fee: ℝ)
  (S: ℝ)
  (h_commission: commission_rate = 0.20)
  (h_sale_tax: sale_tax_rate = 0.08)
  (h_discount: discount_rate = 0.10)
  (h_producer_price: producer_price = 20)
  (h_shipping_fee: shipping_fee = 5)
  (h_total_cost: cost = producer_price + shipping_fee)
  (h_profit: 0.20 * cost = 5)
  (h_total_earn: cost + sale_tax_rate * S + 5 = 0.80 * S)
  (h_S: S = 41.67):
  S = 41.67 ∧ 0.90 * S = 37.50 :=
by
  sorry

end price_without_and_with_coupon_l45_45920


namespace mystery_book_shelves_l45_45757

-- Define the conditions from the problem
def total_books : ℕ := 72
def picture_book_shelves : ℕ := 2
def books_per_shelf : ℕ := 9

-- Determine the number of mystery book shelves
theorem mystery_book_shelves : 
  let books_on_picture_shelves := picture_book_shelves * books_per_shelf
  let mystery_books := total_books - books_on_picture_shelves
  let mystery_shelves := mystery_books / books_per_shelf
  mystery_shelves = 6 :=
by {
  -- This space is intentionally left incomplete, as the proof itself is not required.
  sorry
}

end mystery_book_shelves_l45_45757


namespace greg_sarah_apples_l45_45033

-- Definitions and Conditions
variable {G : ℕ}
variable (H0 : 2 * G + 2 * G + (2 * G - 5) = 49)

-- Statement of the problem
theorem greg_sarah_apples : 
  2 * G = 18 :=
by
  sorry

end greg_sarah_apples_l45_45033


namespace shaded_square_ratio_l45_45152

theorem shaded_square_ratio (side_length : ℝ) (H : side_length = 5) :
  let large_square_area := side_length ^ 2
  let shaded_square_area := (side_length / 2) ^ 2
  shaded_square_area / large_square_area = 1 / 4 :=
by
  sorry

end shaded_square_ratio_l45_45152


namespace percentage_of_third_number_l45_45791

variable (T F S : ℝ)

-- Declare the conditions from step a)
def condition_one : Prop := S = 0.25 * T
def condition_two : Prop := F = 0.20 * S

-- Define the proof problem, proving that F is 5% of T given the conditions
theorem percentage_of_third_number
  (h1 : condition_one T S)
  (h2 : condition_two F S) :
  F = 0.05 * T := by
  sorry

end percentage_of_third_number_l45_45791


namespace number_of_correct_statements_l45_45399

def is_opposite (a b : ℤ) : Prop := a + b = 0

def statement1 : Prop := ∀ a b : ℤ, (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0) → is_opposite a b
def statement2 : Prop := ∀ n : ℤ, n = -n → n < 0
def statement3 : Prop := ∀ a b : ℤ, is_opposite a b → a + b = 0
def statement4 : Prop := ∀ a b : ℤ, is_opposite a b → (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0)

theorem number_of_correct_statements : (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ↔ (∃n : ℕ, n = 1) :=
by
  sorry

end number_of_correct_statements_l45_45399


namespace james_older_brother_age_l45_45991

def johnAge : ℕ := 39

def ageCondition (johnAge : ℕ) (jamesAgeIn6 : ℕ) : Prop :=
  johnAge - 3 = 2 * jamesAgeIn6

def jamesOlderBrother (james : ℕ) : ℕ :=
  james + 4

theorem james_older_brother_age (johnAge jamesOlderBrotherAge : ℕ) (james : ℕ) :
  johnAge = 39 →
  (johnAge - 3 = 2 * (james + 6)) →
  jamesOlderBrotherAge = jamesOlderBrother james →
  jamesOlderBrotherAge = 16 :=
by
  sorry

end james_older_brother_age_l45_45991


namespace fraction_of_power_l45_45748

theorem fraction_of_power (m : ℕ) (h : m = 16^1500) : m / 8 = 2^5997 := by
  sorry

end fraction_of_power_l45_45748


namespace four_digit_number_condition_solution_count_l45_45505

def valid_digits_count : ℕ := 5

theorem four_digit_number_condition (N x a : ℕ) (h1 : N = 1000 * a + x) (h2 : N = 7 * x) (h3 : 100 ≤ x ∧ x ≤ 999) :
  a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 5 :=
begin
  sorry
end

theorem solution_count : ∃ n:ℕ, n = valid_digits_count :=
begin
  use 5,
  sorry
end

end four_digit_number_condition_solution_count_l45_45505


namespace intersection_points_l45_45153

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 3
noncomputable def g (x : ℝ) : ℝ := -f x
noncomputable def h (x : ℝ) : ℝ := f (-x)

theorem intersection_points :
  let a := 2
  let b := 1
  10 * a + b = 21 :=
by
  sorry

end intersection_points_l45_45153


namespace elevation_above_sea_level_mauna_kea_correct_total_height_mauna_kea_correct_elevation_mount_everest_correct_l45_45911

-- Define the initial conditions
def sea_level_drop : ℝ := 397
def submerged_depth_initial : ℝ := 5000
def height_diff_mauna_kea_everest : ℝ := 358

-- Define intermediate calculations based on conditions
def submerged_depth_adjusted : ℝ := submerged_depth_initial - sea_level_drop
def total_height_mauna_kea : ℝ := 2 * submerged_depth_adjusted
def elevation_above_sea_level_mauna_kea : ℝ := total_height_mauna_kea - submerged_depth_initial
def elevation_mount_everest : ℝ := total_height_mauna_kea - height_diff_mauna_kea_everest

-- Define the proof statements
theorem elevation_above_sea_level_mauna_kea_correct :
  elevation_above_sea_level_mauna_kea = 4206 := by
  sorry

theorem total_height_mauna_kea_correct :
  total_height_mauna_kea = 9206 := by
  sorry

theorem elevation_mount_everest_correct :
  elevation_mount_everest = 8848 := by
  sorry

end elevation_above_sea_level_mauna_kea_correct_total_height_mauna_kea_correct_elevation_mount_everest_correct_l45_45911


namespace temperature_on_fifth_day_l45_45764

theorem temperature_on_fifth_day (T : ℕ → ℝ) (x : ℝ)
  (h1 : (T 1 + T 2 + T 3 + T 4) / 4 = 58)
  (h2 : (T 2 + T 3 + T 4 + T 5) / 4 = 59)
  (h3 : T 1 / T 5 = 7 / 8) :
  T 5 = 32 := 
sorry

end temperature_on_fifth_day_l45_45764


namespace john_burritos_left_l45_45741

def total_burritos (b1 b2 b3 b4 : ℕ) : ℕ :=
  b1 + b2 + b3 + b4

def burritos_left_after_giving_away (total : ℕ) (fraction : ℕ) : ℕ :=
  total - (total / fraction)

def burritos_left_after_eating (burritos_left : ℕ) (burritos_per_day : ℕ) (days : ℕ) : ℕ :=
  burritos_left - (burritos_per_day * days)

theorem john_burritos_left :
  let b1 := 15
  let b2 := 20
  let b3 := 25
  let b4 := 5
  let total := total_burritos b1 b2 b3 b4
  let burritos_after_give_away := burritos_left_after_giving_away total 3
  let burritos_after_eating := burritos_left_after_eating burritos_after_give_away 3 10
  burritos_after_eating = 14 :=
by
  sorry

end john_burritos_left_l45_45741


namespace solve_quadratic_l45_45225

theorem solve_quadratic (x : ℝ) : x^2 = x ↔ (x = 0 ∨ x = 1) :=
by
  sorry

end solve_quadratic_l45_45225


namespace smallest_positive_m_integral_solutions_l45_45244

theorem smallest_positive_m_integral_solutions (m : ℕ) :
  (∃ (x y : ℤ), 10 * x * x - m * x + 660 = 0 ∧ 10 * y * y - m * y + 660 = 0 ∧ x ≠ y)
  → m = 170 := sorry

end smallest_positive_m_integral_solutions_l45_45244


namespace part1_solution_set_a_eq_1_part2_range_of_values_a_l45_45321

def f (x a : ℝ) : ℝ := |(2 * x - a)| + |(x - 3 * a)|

theorem part1_solution_set_a_eq_1 :
  ∀ x : ℝ, f x 1 ≤ 4 ↔ 0 ≤ x ∧ x ≤ 2 :=
by sorry

theorem part2_range_of_values_a :
  ∀ a : ℝ, (∀ x : ℝ, f x a ≥ |(x - a / 2)| + a^2 + 1) ↔
    ((-2 : ℝ) ≤ a ∧ a ≤ -1 / 2) ∨ (1 / 2 ≤ a ∧ a ≤ 2) :=
by sorry

end part1_solution_set_a_eq_1_part2_range_of_values_a_l45_45321


namespace probability_nearest_odd_l45_45514

def is_odd_nearest (a b : ℝ) : Prop := ∃ k : ℤ, 2 * k + 1 = Int.floor ((a - b) / (a + b))

def is_valid (a b : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1

noncomputable def probability_odd_nearest : ℝ :=
  let interval_area := 1 -- the area of the unit square [0, 1] x [0, 1]
  let odd_area := 1 / 3 -- as derived from the geometric interpretation in the problem's solution
  odd_area / interval_area

theorem probability_nearest_odd (a b : ℝ) (h : is_valid a b) :
  probability_odd_nearest = 1 / 3 := by
  sorry

end probability_nearest_odd_l45_45514


namespace work_finished_days_earlier_l45_45936

theorem work_finished_days_earlier
  (D : ℕ) (M : ℕ) (A : ℕ) (Work : ℕ) (D_new : ℕ) (E : ℕ)
  (hD : D = 8)
  (hM : M = 30)
  (hA : A = 10)
  (hWork : Work = M * D)
  (hTotalWork : Work = 240)
  (hD_new : D_new = Work / (M + A))
  (hDnew_calculated : D_new = 6)
  (hE : E = D - D_new)
  (hE_calculated : E = 2) : 
  E = 2 :=
by
  sorry

end work_finished_days_earlier_l45_45936


namespace sequence_sum_l45_45664

theorem sequence_sum :
  (3 + 13 + 23 + 33 + 43 + 53) + (5 + 15 + 25 + 35 + 45 + 55) = 348 := by
  sorry

end sequence_sum_l45_45664


namespace factorize_cubic_l45_45007

theorem factorize_cubic : ∀ x : ℝ, x^3 - 4 * x = x * (x + 2) * (x - 2) :=
by
  sorry

end factorize_cubic_l45_45007


namespace find_a_l45_45496

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def f : ℝ → ℝ := sorry -- The definition of f is to be handled in the proof

theorem find_a (a : ℝ) (h1 : is_odd_function f)
  (h2 : ∀ x : ℝ, 0 < x → f x = 2^(x - a) - 2 / (x + 1))
  (h3 : f (-1) = 3 / 4) : a = 3 :=
sorry

end find_a_l45_45496


namespace no_natural_number_exists_l45_45954

theorem no_natural_number_exists 
  (n : ℕ) : ¬ ∃ x y : ℕ, (2 * n * (n + 1) * (n + 2) * (n + 3) + 12) = x^2 + y^2 := 
by sorry

end no_natural_number_exists_l45_45954


namespace positive_integers_no_common_factor_l45_45811

theorem positive_integers_no_common_factor (X Y Z : ℕ) 
    (X_pos : 0 < X) (Y_pos : 0 < Y) (Z_pos : 0 < Z)
    (coprime_XYZ : Nat.gcd (Nat.gcd X Y) Z = 1)
    (eqn : X * (Real.log 3 / Real.log 100) + Y * (Real.log 4 / Real.log 100) = Z^2) :
    X + Y + Z = 4 :=
sorry

end positive_integers_no_common_factor_l45_45811


namespace number_of_narrow_black_stripes_l45_45581

-- Define the variables
variables {w n b : ℕ}

-- The conditions from the problem
def condition1 := w + n = b + 1
def condition2 := b = w + 7

-- The Lean statement to prove
theorem number_of_narrow_black_stripes (h1 : condition1) (h2 : condition2) : n = 8 :=
by {
  -- We import the conditions as hypotheses
  sorry
}

end number_of_narrow_black_stripes_l45_45581


namespace narrow_black_stripes_l45_45561

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end narrow_black_stripes_l45_45561


namespace problem_statement_l45_45839

noncomputable def term_with_largest_binomial_coefficient
  (M N P : ℕ)
  (h_sum : M + N - P = 2016)
  (n : ℕ) : ℤ :=
-8064

noncomputable def term_with_largest_absolute_value_coefficient
  (M N P : ℕ)
  (h_sum : M + N - P = 2016)
  (n : ℕ) : ℤ × ℕ :=
(-15360, 8)

theorem problem_statement (M N P : ℕ) (h_sum : M + N - P = 2016) (n : ℕ) :
  ((term_with_largest_binomial_coefficient M N P h_sum n = -8064) ∧ 
   (term_with_largest_absolute_value_coefficient M N P h_sum n = (-15360, 8))) :=
by {
  -- proof goes here
  sorry
}

end problem_statement_l45_45839


namespace greatest_x_l45_45242

theorem greatest_x (x : ℕ) (h : x^2 < 32) : x ≤ 5 := 
sorry

end greatest_x_l45_45242


namespace factor_x_squared_minus_64_l45_45291

-- Conditions
def a := x
def b := 8

-- Theorem statement
theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_64_l45_45291


namespace find_c_l45_45161

noncomputable def y (x c : ℝ) : ℝ := x^3 - 3*x + c

theorem find_c (c : ℝ) (h : ∃ a b : ℝ, a ≠ b ∧ y a c = 0 ∧ y b c = 0) :
  c = -2 ∨ c = 2 :=
by sorry

end find_c_l45_45161


namespace find_g_l45_45963

-- Definitions for functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := sorry -- We will define this later in the statement

theorem find_g :
  (∀ x : ℝ, g (x + 2) = f x) →
  (∀ x : ℝ, g x = 2 * x - 1) :=
by
  intros h
  sorry

end find_g_l45_45963


namespace count_multiples_12_9_l45_45844

theorem count_multiples_12_9 :
  ∃ n : ℕ, n = 8 ∧ (∀ x : ℕ, x % 36 = 0 ∧ 200 ≤ x ∧ x ≤ 500 ↔ ∃ y : ℕ, (x = 36 * y ∧ 200 ≤ 36 * y ∧ 36 * y ≤ 500)) :=
by
  sorry

end count_multiples_12_9_l45_45844


namespace problem_solution_l45_45149

-- Lean 4 statement of the proof problem
theorem problem_solution (m : ℝ) (U : Set ℝ := Univ) (A : Set ℝ := {x | x^2 + 3*x + 2 = 0}) 
  (B : Set ℝ := {x | x^2 + (m + 1)*x + m = 0}) (h : ∀ x, x ∈ (U \ A) → x ∉ B) : 
  m = 1 ∨ m = 2 :=
by 
  -- This is where the proof would normally go
  sorry

end problem_solution_l45_45149


namespace twentieth_common_number_l45_45032

theorem twentieth_common_number : 
  (∃ (m n : ℤ), (4 * m - 1) = (3 * n + 2) ∧ 20 * 12 - 1 = 239) := 
by
  sorry

end twentieth_common_number_l45_45032


namespace problem_l45_45180

theorem problem : 
  let N := 63745.2981
  let place_value_7 := 1000 -- The place value of the digit 7 (thousands place)
  let place_value_2 := 0.1 -- The place value of the digit 2 (tenths place)
  place_value_7 / place_value_2 = 10000 :=
by
  sorry

end problem_l45_45180


namespace find_number_l45_45424

theorem find_number
  (a b c : ℕ)
  (h_a1 : a ≤ 3)
  (h_b1 : b ≤ 3)
  (h_c1 : c ≤ 3)
  (h_a2 : a ≠ 3)
  (h_b_condition1 : b ≠ 1 → 2 * a * b < 10)
  (h_b_condition2 : b ≠ 2 → 2 * a * b < 10)
  (h_c3 : c = 3)
  : a = 2 ∧ b = 3 ∧ c = 3 :=
by
  sorry

end find_number_l45_45424


namespace maximum_candies_purchase_l45_45530

theorem maximum_candies_purchase (c1 : ℕ) (c4 : ℕ) (c7 : ℕ) (n : ℕ)
    (H_single : c1 = 1)
    (H_pack4  : c4 = 4)
    (H_cost4  : c4 = 3) 
    (H_pack7  : c7 = 7) 
    (H_cost7  : c7 = 4) 
    (H_budget : n = 10) :
    ∃ k : ℕ, k = 16 :=
by
    -- We'll skip the proof since the task requires only the statement
    sorry

end maximum_candies_purchase_l45_45530


namespace max_value_of_expression_l45_45191

theorem max_value_of_expression (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  8 * a + 3 * b + 5 * c ≤ 7 * Real.sqrt 2 :=
sorry

end max_value_of_expression_l45_45191


namespace largest_divisor_of_m_l45_45983

theorem largest_divisor_of_m (m : ℕ) (hm : m > 0) (h : 54 ∣ m^2) : 18 ∣ m :=
sorry

end largest_divisor_of_m_l45_45983


namespace scaling_transformation_l45_45525

theorem scaling_transformation:
  ∀ (x y x' y': ℝ), 
  (x^2 + y^2 = 1) ∧ (x' = 5 * x) ∧ (y' = 3 * y) → 
  (x'^2 / 25 + y'^2 / 9 = 1) :=
by intros x y x' y'
   sorry

end scaling_transformation_l45_45525


namespace triangle_inequality_l45_45492

variables (a b c : ℝ)

theorem triangle_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) / (1 + a + b) > c / (1 + c) :=
sorry

end triangle_inequality_l45_45492


namespace angle_skew_lines_range_l45_45211

theorem angle_skew_lines_range (θ : ℝ) (h1 : 0 < θ) (h2 : θ ≤ 90) : 0 < θ ∧ θ ≤ 90 :=
by sorry

end angle_skew_lines_range_l45_45211


namespace Tim_running_hours_l45_45095

theorem Tim_running_hours
  (initial_days : ℕ)
  (additional_days : ℕ)
  (hours_per_session : ℕ)
  (sessions_per_day : ℕ)
  (total_days : ℕ)
  (total_hours_per_week : ℕ) :
  initial_days = 3 →
  additional_days = 2 →
  hours_per_session = 1 →
  sessions_per_day = 2 →
  total_days = initial_days + additional_days →
  total_hours_per_week = total_days * (hours_per_session * sessions_per_day) →
  total_hours_per_week = 10 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at h5
  rw [nat.add_comm (initial_days * (hours_per_session * sessions_per_day)), nat.add_assoc, nat.mul_comm hours_per_session sessions_per_day] at h5
  rw h5 at h6
  exact h6

end Tim_running_hours_l45_45095


namespace smallest_angle_l45_45353

noncomputable def smallest_angle_in_triangle (a b c : ℝ) : ℝ :=
  if h : 0 <= a ∧ 0 <= b ∧ 0 <= c ∧ a + b > c ∧ a + c > b ∧ b + c > a then
    Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  else
    0

theorem smallest_angle (a b c : ℝ) (h₁ : a = 4) (h₂ : b = 3) (h₃ : c = 2) :
  smallest_angle_in_triangle a b c = Real.arccos (7 / 8) :=
sorry

end smallest_angle_l45_45353


namespace container_capacity_l45_45250

theorem container_capacity (C : ℝ) 
  (h1 : 0.30 * C + 18 = 0.75 * C) : 
  C = 40 :=
by
  -- proof steps would go here
  sorry

end container_capacity_l45_45250


namespace correct_operation_B_l45_45429

variable (a : ℝ)

theorem correct_operation_B :
  2 * a^2 * a^4 = 2 * a^6 :=
by sorry

end correct_operation_B_l45_45429


namespace kitty_cleaning_time_l45_45065

def weekly_cleaning_time (pick_up: ℕ) (vacuum: ℕ) (clean_windows: ℕ) (dust: ℕ) : ℕ :=
  pick_up + vacuum + clean_windows + dust

def total_cleaning_time (weeks: ℕ) (pick_up: ℕ) (vacuum: ℕ) (clean_windows: ℕ) (dust: ℕ) : ℕ :=
  weeks * weekly_cleaning_time pick_up vacuum clean_windows dust

theorem kitty_cleaning_time :
  total_cleaning_time 4 5 20 15 10 = 200 := by
  sorry

end kitty_cleaning_time_l45_45065


namespace chloe_apples_l45_45504

theorem chloe_apples :
  ∃ x : ℕ, (∃ y : ℕ, x = y + 8 ∧ y = x / 3) ∧ x = 12 := 
by
  sorry

end chloe_apples_l45_45504


namespace find_value_l45_45972

-- Definitions of the curve and the line
def curve (a b : ℝ) (P : ℝ × ℝ) : Prop := (P.1*P.1) / a - (P.2*P.2) / b = 1
def line (P : ℝ × ℝ) : Prop := P.1 + P.2 - 1 = 0

-- Definition of the dot product condition
def dot_product_zero (P Q : ℝ × ℝ) : Prop :=
  P.1 * Q.1 + P.2 * Q.2 = 0

-- Theorem statement
theorem find_value (a b : ℝ) (P Q : ℝ × ℝ)
  (hc1 : curve a b P)
  (hc2 : curve a b Q)
  (hl1 : line P)
  (hl2 : line Q)
  (h_dot : dot_product_zero P Q) :
  1 / a - 1 / b = 2 :=
sorry

end find_value_l45_45972


namespace solve_system_l45_45209

-- Define the system of equations
def eq1 (x y : ℝ) : Prop := 2 * x - y = 8
def eq2 (x y : ℝ) : Prop := 3 * x + 2 * y = 5

-- State the theorem to be proved
theorem solve_system : ∃ (x y : ℝ), eq1 x y ∧ eq2 x y ∧ x = 3 ∧ y = -2 := 
by 
  exists 3
  exists -2
  -- Proof steps would go here, but we're using sorry to indicate it's incomplete
  sorry

end solve_system_l45_45209


namespace problem_sign_of_trig_product_l45_45167

open Real

theorem problem_sign_of_trig_product (θ : ℝ) (hθ : π / 2 < θ ∧ θ < π) :
  sin (cos θ) * cos (sin (2 * θ)) < 0 :=
sorry

end problem_sign_of_trig_product_l45_45167


namespace flat_odot_length_correct_l45_45134

noncomputable def sides : ℤ × ℤ × ℤ := (4, 5, 6)

noncomputable def semiperimeter (a b c : ℤ) : ℚ :=
  (a + b + c) / 2

noncomputable def length_flat_odot (a b c : ℤ) : ℚ :=
  (semiperimeter a b c) - b

theorem flat_odot_length_correct : length_flat_odot 4 5 6 = 2.5 := by
  sorry

end flat_odot_length_correct_l45_45134


namespace simplify_sqrt_180_l45_45385

theorem simplify_sqrt_180 : sqrt 180 = 6 * sqrt 5 :=
by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  calc
    sqrt 180
      = sqrt (2^2 * 3^2 * 5)     : by rw [h]
  ... = sqrt (2^2) * sqrt (3^2) * sqrt 5 : by rw [sqrt_mul, sqrt_mul]
  ... = 2 * 3 * sqrt 5          : by rw [sqrt_sq, sqrt_sq]
  ... = 6 * sqrt 5              : by norm_num

end simplify_sqrt_180_l45_45385


namespace ratio_traditionalists_progressives_l45_45265

variables (T P C : ℝ)

-- Conditions from the problem
-- There are 6 provinces and each province has the same number of traditionalists
-- The fraction of the country that is traditionalist is 0.6
def country_conditions (T P C : ℝ) :=
  (6 * T = 0.6 * C) ∧
  (C = P + 6 * T)

-- Theorem that needs to be proven
theorem ratio_traditionalists_progressives (T P C : ℝ) (h : country_conditions T P C) :
  T / P = 1 / 4 :=
by
  -- Setup conditions from the hypothesis h
  rcases h with ⟨h1, h2⟩
  -- Start the proof (Proof content is not required as per instructions)
  sorry

end ratio_traditionalists_progressives_l45_45265


namespace unique_suwy_product_l45_45704

def letter_value (c : Char) : Nat :=
  if 'A' ≤ c ∧ c ≤ 'Z' then Char.toNat c - Char.toNat 'A' + 1 else 0

def product_of_chars (l : List Char) : Nat :=
  l.foldr (λ c acc => letter_value c * acc) 1

theorem unique_suwy_product :
  ∀ (l : List Char), l.length = 4 → product_of_chars l = 19 * 21 * 23 * 25 → l = ['S', 'U', 'W', 'Y'] := 
by
  intro l hlen hproduct
  sorry

end unique_suwy_product_l45_45704


namespace domain_of_function_l45_45138

theorem domain_of_function : 
  {x : ℝ | x + 1 ≥ 0 ∧ x ≠ 1} = {x : ℝ | -1 ≤ x ∧ x < 1} ∪ {x : ℝ | 1 < x} :=
by 
  sorry

end domain_of_function_l45_45138


namespace rented_room_percentage_l45_45869

theorem rented_room_percentage (total_rooms : ℕ) (h1 : 3 * total_rooms / 4 = 3 * total_rooms / 4) 
                               (h2 : 3 * total_rooms / 5 = 3 * total_rooms / 5) 
                               (h3 : 2 * (3 * total_rooms / 5) / 3 = 2 * (3 * total_rooms / 5) / 3) :
  (1 * (3 * total_rooms / 5) / 5) / (1 * total_rooms / 4) * 100 = 80 := by
  sorry

end rented_room_percentage_l45_45869


namespace crickets_needed_to_reach_11_l45_45785

theorem crickets_needed_to_reach_11 (collected_crickets : ℕ) (wanted_crickets : ℕ) 
                                     (h : collected_crickets = 7) (h2 : wanted_crickets = 11) :
  wanted_crickets - collected_crickets = 4 :=
sorry

end crickets_needed_to_reach_11_l45_45785


namespace units_digit_fraction_l45_45781

theorem units_digit_fraction : (2^3 * 31 * 33 * 17 * 7) % 10 = 6 := by
  sorry

end units_digit_fraction_l45_45781


namespace factorization_of_x_squared_minus_64_l45_45288

theorem factorization_of_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factorization_of_x_squared_minus_64_l45_45288


namespace find_a_m_18_l45_45178

variable (a : ℕ → ℝ)
variable (r : ℝ)
variable (a1 : ℝ)
variable (m : ℕ)

noncomputable def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (r : ℝ) :=
  ∀ n : ℕ, a n = a1 * r^n

def problem_conditions (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ) (m : ℕ) :=
  (geometric_sequence a a1 r) ∧
  a m = 3 ∧
  a (m + 6) = 24

theorem find_a_m_18 (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ) (m : ℕ) :
  problem_conditions a r a1 m → a (m + 18) = 1536 :=
by
  sorry

end find_a_m_18_l45_45178


namespace coloring_ways_l45_45647

def num_colorings (total_circles blue_circles green_circles red_circles : ℕ) : ℕ :=
  if total_circles = blue_circles + green_circles + red_circles then
    (Nat.choose total_circles (green_circles + red_circles)) * (Nat.factorial (green_circles + red_circles) / (Nat.factorial green_circles * Nat.factorial red_circles))
  else
    0

theorem coloring_ways :
  num_colorings 6 4 1 1 = 30 :=
by sorry

end coloring_ways_l45_45647


namespace sequence_term_index_l45_45849

open Nat

noncomputable def arithmetic_sequence_term (a₁ d n : ℕ) : ℕ :=
a₁ + (n - 1) * d

noncomputable def term_index (a₁ d term : ℕ) : ℕ :=
1 + (term - a₁) / d

theorem sequence_term_index {a₅ a₄₅ term : ℕ}
  (h₁: a₅ = 33)
  (h₂: a₄₅ = 153)
  (h₃: ∀ n, arithmetic_sequence_term 21 3 n = if n = 5 then 33 else if n = 45 then 153 else (21 + (n - 1) * 3))
  : term_index 21 3 201 = 61 :=
sorry

end sequence_term_index_l45_45849


namespace count_multiples_6_or_8_but_not_both_l45_45510

theorem count_multiples_6_or_8_but_not_both : 
  (∑ i in Finset.range 150, ((if (i % 6 = 0 ∧ i % 24 ≠ 0) ∨ (i % 8 = 0 ∧ i % 24 ≠ 0) then 1 else 0) : ℕ)) = 31 := by
  sorry

end count_multiples_6_or_8_but_not_both_l45_45510


namespace mop_red_slip_identity_l45_45642

-- Define the problem statement in Lean
theorem mop_red_slip_identity (A B : ℕ) (hA : A ≤ 2010) (hB : B ≤ 2010) : 
  ∃ (f : Fin 2010 → Fin 2010) (g : Fin 2010 → Fin 2010),
    (∀ i : Fin 2010, f i = i) ∧ (∀ j : Fin 2010, g j = A * j % 2011) :=
sorry

end mop_red_slip_identity_l45_45642


namespace class_size_is_10_l45_45398

theorem class_size_is_10 
  (num_92 : ℕ) (num_80 : ℕ) (last_score : ℕ) (target_avg : ℕ) (total_score : ℕ) 
  (h_num_92 : num_92 = 5) (h_num_80 : num_80 = 4) (h_last_score : last_score = 70) 
  (h_target_avg : target_avg = 85) (h_total_score : total_score = 85 * (num_92 + num_80 + 1)) 
  : (num_92 * 92 + num_80 * 80 + last_score = total_score) → 
    (num_92 + num_80 + 1 = 10) :=
by {
  sorry
}

end class_size_is_10_l45_45398


namespace sheets_of_paper_l45_45367

theorem sheets_of_paper (x : ℕ) (sheets : ℕ) 
  (h1 : sheets = 3 * x + 31)
  (h2 : sheets = 4 * x + 8) : 
  sheets = 100 := by
  sorry

end sheets_of_paper_l45_45367


namespace grocer_initial_stock_l45_45121

noncomputable def initial_coffee_stock (x : ℝ) : Prop :=
  let initial_decaf := 0.20 * x
  let additional_coffee := 100
  let additional_decaf := 0.50 * additional_coffee
  let total_coffee := x + additional_coffee
  let total_decaf := initial_decaf + additional_decaf
  0.26 * total_coffee = total_decaf

theorem grocer_initial_stock :
  ∃ x : ℝ, initial_coffee_stock x ∧ x = 400 :=
by
  sorry

end grocer_initial_stock_l45_45121


namespace age_difference_l45_45614

theorem age_difference :
  ∃ a b : ℕ, (a < 10) ∧ (b < 10) ∧
    (∀ x y : ℕ, (x = 10 * a + b) ∧ (y = 10 * b + a) → 
    (x + 5 = 2 * (y + 5)) ∧ ((10 * a + b) - (10 * b + a) = 18)) :=
by
  sorry

end age_difference_l45_45614


namespace sticker_height_enlarged_l45_45802

theorem sticker_height_enlarged (orig_width orig_height new_width : ℝ)
    (h1 : orig_width = 3) (h2 : orig_height = 2) (h3 : new_width = 12) :
    new_width / orig_width * orig_height = 8 :=
by
  rw [h1, h2, h3]
  norm_num

end sticker_height_enlarged_l45_45802


namespace curve_symmetrical_about_theta_five_sixths_pi_l45_45527

noncomputable def curve_symmetry (ρ θ : ℝ) : ℝ := 4 * Real.sin(θ - Real.pi / 3)

theorem curve_symmetrical_about_theta_five_sixths_pi : 
  (∀ θ ρ, curve_symmetry ρ θ = 4 * Real.sin(θ - Real.pi / 3)) →
  (∀ θ, (curve_symmetry ρ θ = - curve_symmetry ρ (θ + Real.pi)) ∨
  curve_symmetry ρ θ = curve_symmetry ρ (θ + Real.pi)) :=
by
  sorry

end curve_symmetrical_about_theta_five_sixths_pi_l45_45527


namespace max_value_of_quadratic_l45_45824

-- Define the quadratic function
def f (x : ℝ) : ℝ := 12 * x - 4 * x^2 + 2

-- State the main theorem of finding the maximum value
theorem max_value_of_quadratic : ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∧ f x = 11 := sorry

end max_value_of_quadratic_l45_45824


namespace bella_grazing_area_l45_45922

open Real

theorem bella_grazing_area:
  let leash_length := 5
  let barn_width := 4
  let barn_height := 6
  let sector_fraction := 3 / 4
  let area_circle := π * leash_length^2
  let grazed_area := sector_fraction * area_circle
  grazed_area = 75 / 4 * π := 
by
  sorry

end bella_grazing_area_l45_45922


namespace cost_of_candy_l45_45278

theorem cost_of_candy (initial_amount remaining_amount : ℕ) (h_init : initial_amount = 4) (h_remaining : remaining_amount = 3) : initial_amount - remaining_amount = 1 :=
by
  sorry

end cost_of_candy_l45_45278


namespace program_selection_count_l45_45798

theorem program_selection_count :
  let courses := ["English", "Algebra", "Geometry", "History", "Science", "Art", "Latin"]
  let english := 1
  let math_courses := ["Algebra", "Geometry"]
  let science_courses := ["Science"]
  ∃ (programs : Finset (Finset String)) (count : ℕ),
    (count = 9) ∧
    (programs.card = count) ∧
    ∀ p ∈ programs,
      "English" ∈ p ∧
      (∃ m ∈ p, m ∈ math_courses) ∧
      (∃ s ∈ p, s ∈ science_courses) ∧
      p.card = 5 :=
sorry

end program_selection_count_l45_45798


namespace second_shirt_price_l45_45123

-- Define the conditions
def price_first_shirt := 82
def price_third_shirt := 90
def min_avg_price_remaining_shirts := 104
def total_shirts := 10
def desired_avg_price := 100

-- Prove the price of the second shirt
theorem second_shirt_price : 
  ∀ (P : ℝ), 
  (price_first_shirt + P + price_third_shirt + 7 * min_avg_price_remaining_shirts = total_shirts * desired_avg_price) → 
  P = 100 :=
by
  sorry

end second_shirt_price_l45_45123


namespace sample_size_is_100_l45_45097

-- Define the number of students selected for the sample.
def num_students_sampled : ℕ := 100

-- The statement that the sample size is equal to the number of students sampled.
theorem sample_size_is_100 : num_students_sampled = 100 := 
by {
  -- Proof goes here
  sorry
}

end sample_size_is_100_l45_45097


namespace question1_question2_l45_45975

noncomputable def f (x : ℝ) : ℝ :=
  if x < -4 then -x - 9
  else if x < 1 then 3 * x + 7
  else x + 9

theorem question1 (x : ℝ) (h : -10 ≤ x ∧ x ≤ -2) : f x ≤ 1 := sorry

theorem question2 (x a : ℝ) (hx : x > 1) (h : f x > -x^2 + a * x) : a < 7 := sorry

end question1_question2_l45_45975


namespace shelter_cats_l45_45173

theorem shelter_cats (initial_dogs initial_cats additional_cats : ℕ) 
  (h1 : initial_dogs = 75)
  (h2 : initial_dogs * 7 = initial_cats * 15)
  (h3 : initial_dogs * 11 = 15 * (initial_cats + additional_cats)) : 
  additional_cats = 20 :=
by
  sorry

end shelter_cats_l45_45173


namespace basketball_lineup_count_l45_45643

-- Define the number of players in the basketball team
def num_players : Nat := 12

-- Define the number of lineups
def num_lineups : Nat := 3960

-- Prove the number of lineups is 3960
theorem basketball_lineup_count (num_players = 12) : 
  ∃ num_lineups, num_lineups = 12 * Nat.choose 11 4 := by
  existsi 3960
  sorry

end basketball_lineup_count_l45_45643


namespace narrow_black_stripes_are_eight_l45_45554

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end narrow_black_stripes_are_eight_l45_45554


namespace smallest_power_of_7_not_palindrome_l45_45958

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem smallest_power_of_7_not_palindrome : ∃ n : ℕ, n > 0 ∧ 7^n = 2401 ∧ ¬is_palindrome (7^n) ∧ (∀ m : ℕ, m > 0 ∧ ¬is_palindrome (7^m) → 7^n ≤ 7^m) :=
by
  sorry

end smallest_power_of_7_not_palindrome_l45_45958


namespace sum_squares_and_products_of_nonneg_reals_l45_45345

theorem sum_squares_and_products_of_nonneg_reals {x y z : ℝ} 
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) 
  (h2 : x*y + y*z + z*x = 27) : 
  x + y + z = Real.sqrt 106 := 
by 
  sorry

end sum_squares_and_products_of_nonneg_reals_l45_45345


namespace ratio_lcm_gcf_240_360_l45_45779

theorem ratio_lcm_gcf_240_360 : Nat.lcm 240 360 / Nat.gcd 240 360 = 60 :=
by
  sorry

end ratio_lcm_gcf_240_360_l45_45779


namespace lindsey_saved_in_november_l45_45750

def savings_sept : ℕ := 50
def savings_oct : ℕ := 37
def additional_money : ℕ := 25
def spent_on_video_game : ℕ := 87
def money_left : ℕ := 36

def total_savings_before_november := savings_sept + savings_oct
def total_savings_after_november (N : ℕ) := total_savings_before_november + N + additional_money

theorem lindsey_saved_in_november : ∃ N : ℕ, total_savings_after_november N - spent_on_video_game = money_left ∧ N = 11 :=
by
  sorry

end lindsey_saved_in_november_l45_45750


namespace find_A_plus_B_l45_45616

theorem find_A_plus_B {A B : ℚ} (h : ∀ x : ℚ, 
                     (Bx - 17) / (x^2 - 9 * x + 20) = A / (x - 4) + 5 / (x - 5)) : 
                     A + B = 9 / 5 := sorry

end find_A_plus_B_l45_45616


namespace ratio_of_distances_l45_45634

-- Define the given conditions
variables (w x y : ℕ)
variables (h1 : w > 0) -- walking speed must be positive
variables (h2 : x > 0) -- distance from home must be positive
variables (h3 : y > 0) -- distance to stadium must be positive

-- Define the two times:
-- Time taken to walk directly to the stadium
def time_walk (w y : ℕ) := y / w

-- Time taken to walk home, then bike to the stadium
def time_walk_bike (w x y : ℕ) := x / w + (x + y) / (5 * w)

-- Given that both times are equal
def times_equal (w x y : ℕ) := time_walk w y = time_walk_bike w x y

-- We want to prove that the ratio of x to y is 2/3
theorem ratio_of_distances (w x y : ℕ) (h_time_eq : times_equal w x y) : x / y = 2 / 3 :=
by
  sorry

end ratio_of_distances_l45_45634


namespace suyeong_ran_distance_l45_45602

theorem suyeong_ran_distance 
  (circumference : ℝ) 
  (laps : ℕ) 
  (h_circumference : circumference = 242.7)
  (h_laps : laps = 5) : 
  (circumference * laps = 1213.5) := 
  by sorry

end suyeong_ran_distance_l45_45602


namespace dawns_earnings_per_hour_l45_45052

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

end dawns_earnings_per_hour_l45_45052


namespace bus_trip_children_difference_l45_45591

theorem bus_trip_children_difference :
  let initial := 41
  let final :=
    initial
    - 12 + 5   -- First bus stop
    - 7 + 10   -- Second bus stop
    - 14 + 3   -- Third bus stop
    - 9 + 6    -- Fourth bus stop
  initial - final = 18 :=
by sorry

end bus_trip_children_difference_l45_45591


namespace equilateral_triangle_area_percentage_l45_45939

noncomputable def percentage_area_of_triangle_in_pentagon (s : ℝ) : ℝ :=
  ((4 * Real.sqrt 3 - 3) / 13) * 100

theorem equilateral_triangle_area_percentage
  (s : ℝ) :
  let pentagon_area := s^2 * (1 + Real.sqrt 3 / 4)
  let triangle_area := (Real.sqrt 3 / 4) * s^2
  (triangle_area / pentagon_area) * 100 = percentage_area_of_triangle_in_pentagon s :=
by
  sorry

end equilateral_triangle_area_percentage_l45_45939


namespace number_of_hens_l45_45926

theorem number_of_hens
    (H C : ℕ) -- Hens and Cows
    (h1 : H + C = 44) -- Condition 1: The number of heads
    (h2 : 2 * H + 4 * C = 128) -- Condition 2: The number of feet
    : H = 24 :=
by
  sorry

end number_of_hens_l45_45926


namespace gcd_of_three_numbers_l45_45882

theorem gcd_of_three_numbers : Nat.gcd (Nat.gcd 324 243) 135 = 27 := 
by 
  sorry

end gcd_of_three_numbers_l45_45882


namespace solution_set_f_ge_0_l45_45320

variables {f : ℝ → ℝ}

-- Conditions
axiom h1 : ∀ x : ℝ, f (-x) = -f x  -- f is odd function
axiom h2 : ∀ x y : ℝ, 0 < x → x < y → f x < f y  -- f is monotonically increasing on (0, +∞)
axiom h3 : f 3 = 0  -- f(3) = 0

theorem solution_set_f_ge_0 : { x : ℝ | f x ≥ 0 } = { x : ℝ | -3 ≤ x ∧ x ≤ 0 } ∪ { x : ℝ | 3 ≤ x } :=
by
  sorry

end solution_set_f_ge_0_l45_45320


namespace owen_final_turtle_count_l45_45070

theorem owen_final_turtle_count (owen_initial johanna_initial : ℕ)
  (h1: owen_initial = 21)
  (h2: johanna_initial = owen_initial - 5) :
  let owen_after_1_month := 2 * owen_initial,
      johanna_after_1_month := johanna_initial / 2,
      owen_final := owen_after_1_month + johanna_after_1_month
  in
  owen_final = 50 :=
by
  -- Solution steps go here.
  sorry

end owen_final_turtle_count_l45_45070


namespace Phone_Bill_October_Phone_Bill_November_December_Extra_Cost_November_December_l45_45110

/-- Definitions for phone plans A and B and phone call durations -/
def fixed_cost_A : ℕ := 18
def free_minutes_A : ℕ := 1500
def price_per_minute_A : ℕ → ℚ := λ t => 0.1 * t

def fixed_cost_B : ℕ := 38
def free_minutes_B : ℕ := 4000
def price_per_minute_B : ℕ → ℚ := λ t => 0.07 * t

def call_duration_October : ℕ := 2600
def total_bill_November_December : ℚ := 176
def total_call_duration_November_December : ℕ := 5200

/-- Problem statements to be proven -/

theorem Phone_Bill_October : 
  fixed_cost_A + price_per_minute_A (call_duration_October - free_minutes_A) = 128 :=
  sorry

theorem Phone_Bill_November_December (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ total_call_duration_November_December) : 
  let bill_November := fixed_cost_A + price_per_minute_A (x - free_minutes_A)
  let bill_December := fixed_cost_B + price_per_minute_B (total_call_duration_November_December - x - free_minutes_B)
  bill_November + bill_December = total_bill_November_December :=
  sorry
  
theorem Extra_Cost_November_December :
  let actual_cost := 138 + 38
  let hypothetical_cost := fixed_cost_A + price_per_minute_A (total_call_duration_November_December - free_minutes_A)
  hypothetical_cost - actual_cost = 80 :=
  sorry

end Phone_Bill_October_Phone_Bill_November_December_Extra_Cost_November_December_l45_45110


namespace tan_beta_value_l45_45495

theorem tan_beta_value (α β : ℝ) (h1 : Real.tan α = 1 / 3) (h2 : Real.tan (α + β) = 1 / 2) : Real.tan β = 1 / 7 :=
by
  sorry

end tan_beta_value_l45_45495


namespace dogs_daily_food_total_l45_45469

theorem dogs_daily_food_total :
  let first_dog_food := 0.125
  let second_dog_food := 0.25
  let third_dog_food := 0.375
  let fourth_dog_food := 0.5
  first_dog_food + second_dog_food + third_dog_food + fourth_dog_food = 1.25 :=
by
  sorry

end dogs_daily_food_total_l45_45469


namespace min_sum_abc_l45_45609

theorem min_sum_abc (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 1716) :
  a + b + c = 31 :=
sorry

end min_sum_abc_l45_45609


namespace intersect_point_sum_l45_45082

theorem intersect_point_sum (a' b' : ℝ) (x y : ℝ) 
    (h1 : x = (1 / 3) * y + a')
    (h2 : y = (1 / 3) * x + b')
    (h3 : x = 2)
    (h4 : y = 4) : 
    a' + b' = 4 :=
by
  sorry

end intersect_point_sum_l45_45082


namespace bus_stops_per_hour_l45_45112

theorem bus_stops_per_hour 
  (bus_speed_without_stoppages : Float)
  (bus_speed_with_stoppages : Float)
  (bus_stops_per_hour_in_minutes : Float) :
  bus_speed_without_stoppages = 60 ∧ 
  bus_speed_with_stoppages = 45 → 
  bus_stops_per_hour_in_minutes = 15 := by
  sorry

end bus_stops_per_hour_l45_45112


namespace remainder_2_pow_224_plus_104_l45_45108

theorem remainder_2_pow_224_plus_104 (x : ℕ) (h1 : x = 2 ^ 56) : 
  (2 ^ 224 + 104) % (2 ^ 112 + 2 ^ 56 + 1) = 103 := 
by
  sorry

end remainder_2_pow_224_plus_104_l45_45108


namespace relationship_f_2011_2014_l45_45158

noncomputable def quadratic_func : Type := ℝ → ℝ

variable (f : quadratic_func)

-- The function is symmetric about x = 2013
axiom symmetry (x : ℝ) : f (2013 + x) = f (2013 - x)

-- The function opens upward (convexity)
axiom opens_upward (a b : ℝ) : f ((a + b) / 2) ≤ (f a + f b) / 2

theorem relationship_f_2011_2014 :
  f 2011 > f 2014 := 
sorry

end relationship_f_2011_2014_l45_45158


namespace apple_allocation_proof_l45_45433

theorem apple_allocation_proof : 
    ∃ (ann mary jane kate ned tom bill jack : ℕ), 
    ann = 1 ∧
    mary = 2 ∧
    jane = 3 ∧
    kate = 4 ∧
    ned = jane ∧
    tom = 2 * kate ∧
    bill = 3 * ann ∧
    jack = 4 * mary ∧
    ann + mary + jane + ned + kate + tom + bill + jack = 32 :=
by {
    sorry
}

end apple_allocation_proof_l45_45433


namespace decreasing_interval_of_even_function_l45_45338

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k*x^2 + (k - 1)*x + 2

theorem decreasing_interval_of_even_function (k : ℝ) (h : ∀ x : ℝ, f k x = f k (-x)) :
  ∃ k : ℝ, k = 1 ∧ ∀ x : ℝ, (x < 0 → f k x > f k (-x)) := 
sorry

end decreasing_interval_of_even_function_l45_45338


namespace more_stable_performance_l45_45418

theorem more_stable_performance (s_A_sq s_B_sq : ℝ) (hA : s_A_sq = 0.25) (hB : s_B_sq = 0.12) : s_A_sq > s_B_sq :=
by
  rw [hA, hB]
  sorry

end more_stable_performance_l45_45418


namespace find_k_exact_one_real_solution_l45_45479

theorem find_k_exact_one_real_solution (k : ℝ) :
  (∀ x : ℝ, (3*x + 6)*(x - 4) = -33 + k*x) ↔ (k = -6 + 6*Real.sqrt 3 ∨ k = -6 - 6*Real.sqrt 3) := 
by
  sorry

end find_k_exact_one_real_solution_l45_45479


namespace find_quotient_l45_45216

theorem find_quotient
  (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ)
  (h1 : dividend = 131) (h2 : divisor = 14) (h3 : remainder = 5)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 9 :=
by
  sorry

end find_quotient_l45_45216


namespace minimum_value_of_fm_plus_fp_l45_45726

def f (x a : ℝ) : ℝ := -x^3 + a * x^2 - 4

def f_prime (x a : ℝ) : ℝ := -3 * x^2 + 2 * a * x

theorem minimum_value_of_fm_plus_fp (a : ℝ) (h_extremum : f_prime 2 a = 0) (m n : ℝ) 
  (hm : -1 ≤ m ∧ m ≤ 1) (hn : -1 ≤ n ∧ n ≤ 1) : 
  f m a + f_prime n a = -13 := 
by
  -- steps of the proof would go here
  sorry

end minimum_value_of_fm_plus_fp_l45_45726


namespace smallest_n_satisfies_condition_l45_45475

theorem smallest_n_satisfies_condition : 
  ∃ (n : ℕ), n = 1806 ∧ ∀ (p : ℕ), Nat.Prime p → n % (p - 1) = 0 → n % p = 0 := 
sorry

end smallest_n_satisfies_condition_l45_45475


namespace ratio_of_playground_area_to_total_landscape_area_l45_45883

theorem ratio_of_playground_area_to_total_landscape_area {B L : ℝ} 
    (h1 : L = 8 * B)
    (h2 : L = 240)
    (h3 : 1200 = (240 * B * L) / (240 * B)) :
    1200 / (240 * B) = 1 / 6 :=
sorry

end ratio_of_playground_area_to_total_landscape_area_l45_45883


namespace tan_double_angle_l45_45969

theorem tan_double_angle (α : ℝ) (h : Real.tan α = 1 / 3) : Real.tan (2 * α) = 3 / 4 := 
by
  sorry

end tan_double_angle_l45_45969


namespace binom_coeff_mult_l45_45677

theorem binom_coeff_mult :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_coeff_mult_l45_45677


namespace tangent_parallel_to_line_l45_45227

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_to_line :
  ∃ a b : ℝ, (f a = b) ∧ (3 * a^2 + 1 = 4) ∧ (P = (1, 0) ∨ P = (-1, -4)) :=
by
  sorry

end tangent_parallel_to_line_l45_45227


namespace complex_multiplication_l45_45214

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (1 - 2 * i) = 2 + i :=
by
  sorry

end complex_multiplication_l45_45214


namespace sqrt_180_simplify_l45_45383

theorem sqrt_180_simplify : real.sqrt 180 = 6 * real.sqrt 5 := by
  have h : 180 = 2^2 * 3^2 * 5 := by norm_num
  rw [h, real.sqrt_mul, real.sqrt_mul, real.sqrt_mul]
  all_goals { norm_num }
  sorry

end sqrt_180_simplify_l45_45383


namespace smallest_value_of_x_l45_45462

theorem smallest_value_of_x :
  ∀ x : ℚ, ( ( (5 * x - 20) / (4 * x - 5) ) ^ 3
           + ( (5 * x - 20) / (4 * x - 5) ) ^ 2
           - ( (5 * x - 20) / (4 * x - 5) )
           - 15 = 0 ) → x = 10 / 3 :=
by
  sorry

end smallest_value_of_x_l45_45462


namespace log_addition_l45_45666

theorem log_addition (log_base_10 : ℝ → ℝ) (a b : ℝ) (h_base_10_log : log_base_10 10 = 1) :
  log_base_10 2 + log_base_10 5 = 1 :=
by
  sorry

end log_addition_l45_45666


namespace set_B_can_form_right_angled_triangle_l45_45630

-- Definition and condition from the problem
def isRightAngledTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- The actual proof problem statement
theorem set_B_can_form_right_angled_triangle : isRightAngledTriangle 1 (Real.sqrt 3) 2 :=
sorry

end set_B_can_form_right_angled_triangle_l45_45630


namespace binom_mult_l45_45683

open Nat

theorem binom_mult : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binom_mult_l45_45683


namespace decreasing_function_range_a_l45_45727

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 1 then (1 - 2 * a) ^ x else log a x + 1 / 3

theorem decreasing_function_range_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f x1 a - f x2 a) / (x1 - x2) < 0) ↔ (0 < a ∧ a ≤ 1 / 3) :=
sorry

end decreasing_function_range_a_l45_45727


namespace ratio_revenue_l45_45440

variable (N D J : ℝ)

theorem ratio_revenue (h1 : J = N / 3) (h2 : D = 2.5 * (N + J) / 2) : N / D = 3 / 5 := by
  sorry

end ratio_revenue_l45_45440


namespace no_real_solution_l45_45306

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

-- Lean statement: prove that the equation x^2 - 4x + 6 = 0 has no real solution
theorem no_real_solution : ¬ ∃ x : ℝ, f x = 0 :=
sorry

end no_real_solution_l45_45306


namespace pentagon_perimeter_l45_45850

-- Define the side length and number of sides for a regular pentagon
def side_length : ℝ := 5
def num_sides : ℕ := 5

-- Define the perimeter calculation as a constant
def perimeter (side_length : ℝ) (num_sides : ℕ) : ℝ := side_length * num_sides

theorem pentagon_perimeter : perimeter side_length num_sides = 25 := by
  sorry

end pentagon_perimeter_l45_45850


namespace maria_cartons_needed_l45_45585

theorem maria_cartons_needed : 
  ∀ (total_needed strawberries blueberries raspberries blackberries : ℕ), 
  total_needed = 36 →
  strawberries = 4 →
  blueberries = 8 →
  raspberries = 3 →
  blackberries = 5 →
  (total_needed - (strawberries + blueberries + raspberries + blackberries) = 16) :=
by
  intros total_needed strawberries blueberries raspberries blackberries ht hs hb hr hb
  -- ... the proof would go here
  sorry

end maria_cartons_needed_l45_45585


namespace real_solutions_x4_plus_3_minus_x4_eq_82_l45_45818

theorem real_solutions_x4_plus_3_minus_x4_eq_82 :
  ∀ x : ℝ, x = 2.6726 ∨ x = 0.3274 → x^4 + (3 - x)^4 = 82 := by
  sorry

end real_solutions_x4_plus_3_minus_x4_eq_82_l45_45818


namespace binom_mult_l45_45686

open Nat

theorem binom_mult : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binom_mult_l45_45686


namespace find_a_l45_45903

-- Define the given context (condition)
def condition (a : ℝ) : Prop := 0.5 / 100 * a = 75 / 100 -- since 1 paise = 1/100 rupee

-- Define the statement to prove
theorem find_a (a : ℝ) (h : condition a) : a = 150 := 
sorry

end find_a_l45_45903


namespace triangle_area_correct_l45_45039

noncomputable def triangle_area_given_conditions (a b c : ℝ) (A : ℝ) : ℝ :=
  if h : a = c + 4 ∧ b = c + 2 ∧ Real.cos A = -1/2 then
  1/2 * b * c * Real.sin A
  else 0

theorem triangle_area_correct :
  ∀ (a b c : ℝ), ∀ A : ℝ, a = c + 4 → b = c + 2 → Real.cos A = -1/2 → 
  triangle_area_given_conditions a b c A = 15 * Real.sqrt 3 / 4 :=
by
  intros a b c A ha hb hc
  simp [triangle_area_given_conditions, ha, hb, hc]
  sorry

end triangle_area_correct_l45_45039


namespace johns_change_l45_45359

/-- Define the cost of Slurpees and amount given -/
def cost_per_slurpee : ℕ := 2
def amount_given : ℕ := 20
def slurpees_bought : ℕ := 6

/-- Define the total cost of the Slurpees -/
def total_cost : ℕ := cost_per_slurpee * slurpees_bought

/-- Define the change John gets -/
def change (amount_given total_cost : ℕ) : ℕ := amount_given - total_cost

/-- The statement for Lean 4 that proves the change John gets is $8 given the conditions -/
theorem johns_change : change amount_given total_cost = 8 :=
by 
  -- Rest of the proof omitted
  sorry

end johns_change_l45_45359


namespace narrow_black_stripes_l45_45571

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end narrow_black_stripes_l45_45571


namespace length_of_bridge_l45_45116

noncomputable def speed_kmhr_to_ms (v : ℕ) : ℝ := (v : ℝ) * (1000 / 3600)

noncomputable def distance_traveled (v : ℝ) (t : ℕ) : ℝ := v * (t : ℝ)

theorem length_of_bridge 
  (length_train : ℕ) -- 90 meters
  (speed_train_kmhr : ℕ) -- 45 km/hr
  (time_cross_bridge : ℕ) -- 30 seconds
  (conversion_factor : ℝ := 1000 / 3600) 
  : ℝ := 
  let speed_train_ms := speed_kmhr_to_ms speed_train_kmhr
  let total_distance := distance_traveled speed_train_ms time_cross_bridge
  total_distance - (length_train : ℝ)

example : length_of_bridge 90 45 30 = 285 := by
  sorry

end length_of_bridge_l45_45116


namespace monthly_installment_amount_l45_45213

variable (cashPrice : ℕ) (deposit : ℕ) (monthlyInstallments : ℕ) (savingsIfCash : ℕ)

-- Defining the conditions
def conditions := 
  cashPrice = 8000 ∧ 
  deposit = 3000 ∧ 
  monthlyInstallments = 30 ∧ 
  savingsIfCash = 4000

-- Proving the amount of each monthly installment
theorem monthly_installment_amount (h : conditions cashPrice deposit monthlyInstallments savingsIfCash) : 
  (12000 - deposit) / monthlyInstallments = 300 :=
sorry

end monthly_installment_amount_l45_45213


namespace distinct_pairs_count_l45_45144

theorem distinct_pairs_count : 
  ∃ (S : Finset (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ S ↔ x = x^2 + y^2 ∧ y = 3 * x * y) ∧ 
    S.card = 4 :=
by
  sorry

end distinct_pairs_count_l45_45144


namespace cos_double_alpha_proof_l45_45036

theorem cos_double_alpha_proof (α : ℝ) (h1 : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.cos (π / 3 + 2 * α) = - 7 / 9 :=
by
  sorry

end cos_double_alpha_proof_l45_45036


namespace frequency_of_second_group_l45_45930

theorem frequency_of_second_group (total_capacity : ℕ) (freq_percentage : ℝ)
    (h_capacity : total_capacity = 80)
    (h_percentage : freq_percentage = 0.15) :
    total_capacity * freq_percentage = 12 :=
by
  sorry

end frequency_of_second_group_l45_45930


namespace find_cubic_expression_l45_45830

theorem find_cubic_expression (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 648 := by
  sorry

end find_cubic_expression_l45_45830


namespace narrow_black_stripes_l45_45544

variable (w n b : ℕ)

theorem narrow_black_stripes (w : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := by
  have h3 : w + n = (w + 7) + 1 := by rw [h1]
  have h4 : w + n = w + 8 := by linarith
  have h5 : n = 8 := by linarith
  exact h5

end narrow_black_stripes_l45_45544


namespace proof_not_sufficient_nor_necessary_l45_45488

noncomputable def not_sufficient_nor_necessary (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : Prop :=
  ¬ ((a > b) → (Real.log b / Real.log a < 1)) ∧ ¬ ((Real.log b / Real.log a < 1) → (a > b))

theorem proof_not_sufficient_nor_necessary (a b: ℝ) (h₁: 0 < a) (h₂: 0 < b) :
  not_sufficient_nor_necessary a b h₁ h₂ :=
  sorry

end proof_not_sufficient_nor_necessary_l45_45488


namespace narrow_black_stripes_are_eight_l45_45551

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end narrow_black_stripes_are_eight_l45_45551


namespace number_of_narrow_black_stripes_l45_45582

-- Define the variables
variables {w n b : ℕ}

-- The conditions from the problem
def condition1 := w + n = b + 1
def condition2 := b = w + 7

-- The Lean statement to prove
theorem number_of_narrow_black_stripes (h1 : condition1) (h2 : condition2) : n = 8 :=
by {
  -- We import the conditions as hypotheses
  sorry
}

end number_of_narrow_black_stripes_l45_45582


namespace inequality_proof_l45_45317

theorem inequality_proof (a b c d e f : ℝ) (H : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end inequality_proof_l45_45317


namespace slope_of_line_through_A_B_l45_45971

theorem slope_of_line_through_A_B :
  let A := (2, 1)
  let B := (-1, 3)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = -2/3 :=
by
  have A_x : Int := 2
  have A_y : Int := 1
  have B_x : Int := -1
  have B_y : Int := 3
  sorry

end slope_of_line_through_A_B_l45_45971


namespace c_is_perfect_square_l45_45021

theorem c_is_perfect_square (a b c : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h : c = a + b / a - 1 / b) : ∃ m : ℕ, c = m * m :=
by
  sorry

end c_is_perfect_square_l45_45021


namespace master_efficiency_comparison_l45_45063

theorem master_efficiency_comparison (z_parts : ℕ) (z_hours : ℕ) (l_parts : ℕ) (l_hours : ℕ)
    (hz : z_parts = 5) (hz_time : z_hours = 8)
    (hl : l_parts = 3) (hl_time : l_hours = 4) :
    (z_parts / z_hours : ℚ) < (l_parts / l_hours : ℚ) → false :=
by
  -- This is a placeholder for the proof, which is not needed as per the instructions.
  sorry

end master_efficiency_comparison_l45_45063


namespace simplify_expression_l45_45373

theorem simplify_expression (w : ℝ) : 3 * w + 6 * w - 9 * w + 12 * w - 15 * w + 21 = -3 * w + 21 :=
by
  sorry

end simplify_expression_l45_45373


namespace scientific_notation_of_935million_l45_45659

theorem scientific_notation_of_935million :
  935000000 = 9.35 * 10 ^ 8 :=
  sorry

end scientific_notation_of_935million_l45_45659


namespace xy_exists_5n_l45_45301

theorem xy_exists_5n (n : ℕ) (hpos : 0 < n) :
  ∃ x y : ℤ, x^2 + y^2 = 5^n ∧ Int.gcd x 5 = 1 ∧ Int.gcd y 5 = 1 :=
sorry

end xy_exists_5n_l45_45301


namespace minimum_candies_l45_45090

theorem minimum_candies (students : ℕ) (N : ℕ) (k : ℕ) : 
  students = 25 → 
  N = 25 * k → 
  (∀ n, 1 ≤ n → n ≤ students → ∃ m, n * k + m ≤ N) → 
  600 ≤ N := 
by
  intros hs hn hd
  sorry

end minimum_candies_l45_45090


namespace value_of_ab_plus_bc_plus_ca_l45_45486

theorem value_of_ab_plus_bc_plus_ca (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end value_of_ab_plus_bc_plus_ca_l45_45486


namespace quadratic_has_two_roots_l45_45305

theorem quadratic_has_two_roots (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : 5 * a + b + 2 * c = 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 := 
  sorry

end quadratic_has_two_roots_l45_45305


namespace chord_length_l45_45619

-- Define the key components.
structure Circle := 
(center : ℝ × ℝ)
(radius : ℝ)

-- Define the initial conditions.
def circle1 : Circle := { center := (0, 0), radius := 5 }
def circle2 : Circle := { center := (2, 0), radius := 3 }

-- Define the chord and tangency condition.
def touches_internally (C1 C2 : Circle) : Prop :=
  C1.radius > C2.radius ∧ dist C1.center C2.center = C1.radius - C2.radius

def chord_divided_ratio (AB_length : ℝ) (r1 r2 : ℝ) : Prop :=
  ∃ (x : ℝ), AB_length = 4 * x ∧ r1 = x ∧ r2 = 3 * x

-- The theorem to prove the length of the chord AB.
theorem chord_length (h1 : touches_internally circle1 circle2)
                     (h2 : chord_divided_ratio 8 2 (6)) : ∃ (AB_length : ℝ), AB_length = 8 :=
by
  sorry

end chord_length_l45_45619
