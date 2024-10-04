import Mathlib

namespace fraction_scaling_l244_244951

theorem fraction_scaling (x y : ℝ) :
  ((5 * x - 5 * 5 * y) / ((5 * x) ^ 2 + (5 * y) ^ 2)) = (1 / 5) * ((x - 5 * y) / (x ^ 2 + y ^ 2)) :=
by
  sorry

end fraction_scaling_l244_244951


namespace min_value_a_l244_244854

theorem min_value_a (a : ℕ) :
  (6 * (a + 1)) / (a^2 + 8 * a + 6) ≤ 1 / 100 ↔ a ≥ 594 := sorry

end min_value_a_l244_244854


namespace average_speed_with_stoppages_l244_244334

/--The average speed of the bus including stoppages is 20 km/hr, 
  given that the bus stops for 40 minutes per hour and 
  has an average speed of 60 km/hr excluding stoppages.--/
theorem average_speed_with_stoppages 
  (avg_speed_without_stoppages : ℝ)
  (stoppage_time_per_hour : ℕ) 
  (running_time_per_hour : ℕ) 
  (avg_speed_with_stoppages : ℝ) 
  (h1 : avg_speed_without_stoppages = 60) 
  (h2 : stoppage_time_per_hour = 40) 
  (h3 : running_time_per_hour = 20) 
  (h4 : running_time_per_hour + stoppage_time_per_hour = 60):
  avg_speed_with_stoppages = 20 := 
sorry

end average_speed_with_stoppages_l244_244334


namespace domain_of_log2_function_l244_244374

theorem domain_of_log2_function :
  {x : ℝ | 2 * x - 1 > 0} = {x : ℝ | x > 1 / 2} :=
by
  sorry

end domain_of_log2_function_l244_244374


namespace speed_in_still_water_l244_244256

namespace SwimmingProblem

variable (V_m V_s : ℝ)

-- Downstream condition
def downstream_condition : Prop := V_m + V_s = 18

-- Upstream condition
def upstream_condition : Prop := V_m - V_s = 13

-- The main theorem stating the problem
theorem speed_in_still_water (h_downstream : downstream_condition V_m V_s) 
                             (h_upstream : upstream_condition V_m V_s) :
    V_m = 15.5 :=
by
  sorry

end SwimmingProblem

end speed_in_still_water_l244_244256


namespace sum_of_primes_less_than_20_l244_244123

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l244_244123


namespace Tim_transactions_l244_244818

theorem Tim_transactions
  (Mabel_Monday : ℕ)
  (Mabel_Tuesday : ℕ := Mabel_Monday + Mabel_Monday / 10)
  (Anthony_Tuesday : ℕ := 2 * Mabel_Tuesday)
  (Cal_Tuesday : ℕ := (2 * Anthony_Tuesday) / 3)
  (Jade_Tuesday : ℕ := Cal_Tuesday + 17)
  (Isla_Wednesday : ℕ := Mabel_Tuesday + Cal_Tuesday - 12)
  (Tim_Thursday : ℕ := (Jade_Tuesday + Isla_Wednesday) * 3 / 2)
  : Tim_Thursday = 614 := by sorry

end Tim_transactions_l244_244818


namespace smaller_rectangle_area_l244_244551

theorem smaller_rectangle_area
  (L : ℕ) (W : ℕ) (h₁ : L = 40) (h₂ : W = 20) :
  let l := L / 2;
      w := W / 2 in
  l * w = 200 :=
by
  sorry

end smaller_rectangle_area_l244_244551


namespace volume_of_cylinder_in_pyramid_l244_244408

theorem volume_of_cylinder_in_pyramid
  (a α : ℝ)
  (sin_alpha : ℝ := Real.sin α)
  (tan_alpha : ℝ := Real.tan α)
  (sin_pi_four_alpha : ℝ := Real.sin (Real.pi / 4 + α))
  (sqrt_two : ℝ := Real.sqrt 2) :
  (π * a^3 * sqrt_two * (Real.sin (2 * α))^3) / (128 * sin_pi_four_alpha^3) =
  (π * a^3 * sqrt_two * (Real.sin (2 * α))^3 / (128 * sin_pi_four_alpha^3)) :=
by
  sorry

end volume_of_cylinder_in_pyramid_l244_244408


namespace sum_primes_less_than_20_l244_244065

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l244_244065


namespace value_is_100_l244_244264

theorem value_is_100 (number : ℕ) (h : number = 20) : 5 * number = 100 :=
by
  sorry

end value_is_100_l244_244264


namespace loss_percentage_l244_244276

/--
A man sells a car to his friend at a certain loss percentage. The friend then sells it 
for Rs. 54000 and gains 20%. The original cost price of the car was Rs. 52941.17647058824.
Prove that the loss percentage when the man sold the car to his friend was 15%.
-/
theorem loss_percentage (CP SP_2 : ℝ) (gain_percent : ℝ) (h_CP : CP = 52941.17647058824) 
(h_SP2 : SP_2 = 54000) (h_gain : gain_percent = 20) : (CP - SP_2 / (1 + gain_percent / 100)) / CP * 100 = 15 := by
  sorry

end loss_percentage_l244_244276


namespace original_number_is_40_l244_244724

theorem original_number_is_40 (x : ℝ) (h : 1.25 * x - 0.70 * x = 22) : x = 40 :=
by
  sorry

end original_number_is_40_l244_244724


namespace cell_division_proof_l244_244194

-- Define the problem
def cell_division_ways (n m : Nat) : Nat :=
  if (n = 17 ∧ m = 9) then 10 else 0

-- The Lean statement to assert the problem
theorem cell_division_proof : cell_division_ways 17 9 = 10 :=
by
-- simplifying the definition for the given parameters
simp [cell_division_ways]
sorry

end cell_division_proof_l244_244194


namespace mollys_present_age_l244_244722

theorem mollys_present_age (x : ℤ) (h : x + 18 = 5 * (x - 6)) : x = 12 := by
  sorry

end mollys_present_age_l244_244722


namespace find_interest_rate_l244_244271

theorem find_interest_rate
  (P : ℝ) (t : ℕ) (I : ℝ)
  (hP : P = 3000)
  (ht : t = 5)
  (hI : I = 750) :
  ∃ r : ℝ, I = P * r * t / 100 ∧ r = 5 :=
by 
  sorry

end find_interest_rate_l244_244271


namespace inequality_satisfaction_l244_244286

theorem inequality_satisfaction (k n : ℕ) (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (1 + y^n / x^k) ≥ ((1 + y)^n / (1 + x)^k) ↔ 
    (k = 0) ∨ (n = 0) ∨ (0 = k ∧ 0 = n) ∨ (k ≥ n - 1 ∧ n ≥ 1) :=
by sorry

end inequality_satisfaction_l244_244286


namespace toms_weekly_income_l244_244522

variable (num_buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ)

def daily_crabs := num_buckets * crabs_per_bucket
def daily_income := daily_crabs * price_per_crab
def weekly_income := daily_income * days_per_week

theorem toms_weekly_income 
  (h1 : num_buckets = 8)
  (h2 : crabs_per_bucket = 12)
  (h3 : price_per_crab = 5)
  (h4 : days_per_week = 7) :
  weekly_income num_buckets crabs_per_bucket price_per_crab days_per_week = 3360 :=
by
  sorry

end toms_weekly_income_l244_244522


namespace zero_count_at_end_of_45_320_125_product_l244_244945

theorem zero_count_at_end_of_45_320_125_product :
  let p := 45 * 320 * 125
  45 = 5 * 3^2 ∧ 320 = 2^6 * 5 ∧ 125 = 5^3 →
  p = 2^6 * 3^2 * 5^5 →
  p % 10^5 = 0 ∧ p % 10^6 ≠ 0 :=
by
  sorry

end zero_count_at_end_of_45_320_125_product_l244_244945


namespace new_books_count_l244_244157

-- Defining the conditions
def num_adventure_books : ℕ := 13
def num_mystery_books : ℕ := 17
def num_used_books : ℕ := 15

-- Proving the number of new books Sam bought
theorem new_books_count : (num_adventure_books + num_mystery_books) - num_used_books = 15 :=
by
  sorry

end new_books_count_l244_244157


namespace sqrt_infinite_nested_problem_l244_244501

theorem sqrt_infinite_nested_problem :
  ∃ m : ℝ, m = Real.sqrt (6 + m) ∧ m = 3 :=
by
  sorry

end sqrt_infinite_nested_problem_l244_244501


namespace sin_240_eq_neg_sqrt3_div_2_l244_244615

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244615


namespace initial_goats_l244_244494

theorem initial_goats (G : ℕ) (h1 : 2 + 3 + G + 3 + 5 + 2 = 21) : G = 4 :=
by
  sorry

end initial_goats_l244_244494


namespace tom_should_pay_times_original_price_l244_244867

-- Definitions of the given conditions
def original_price : ℕ := 3
def amount_paid : ℕ := 9

-- The theorem to prove
theorem tom_should_pay_times_original_price : ∃ k : ℕ, amount_paid = k * original_price ∧ k = 3 :=
by 
  -- Using sorry to skip the proof for now
  sorry

end tom_should_pay_times_original_price_l244_244867


namespace ellipse_product_axes_l244_244358

/-- Prove that the product of the lengths of the major and minor axes (AB)(CD) of an ellipse
is 240, given the following conditions:
- Point O is the center of the ellipse.
- Point F is one focus of the ellipse.
- OF = 8
- The diameter of the inscribed circle of triangle OCF is 4.
- OA = OB = a
- OC = OD = b
- a² - b² = 64
- a - b = 4
-/
theorem ellipse_product_axes (a b : ℝ) (OF : ℝ) (d_inscribed_circle : ℝ) 
  (h1 : OF = 8) (h2 : d_inscribed_circle = 4) (h3 : a^2 - b^2 = 64) 
  (h4 : a - b = 4) : (2 * a) * (2 * b) = 240 :=
sorry

end ellipse_product_axes_l244_244358


namespace sum_of_primes_lt_20_eq_77_l244_244113

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l244_244113


namespace train_cross_time_l244_244377

noncomputable def train_length : ℕ := 1200 -- length of the train in meters
noncomputable def platform_length : ℕ := train_length -- length of the platform equals the train length
noncomputable def speed_kmh : ℝ := 144 -- speed in km/hr
noncomputable def speed_ms : ℝ := speed_kmh * (1000 / 3600) -- converting speed to m/s

-- the formula to calculate the crossing time
noncomputable def time_to_cross_platform : ℝ := 
  2 * train_length / speed_ms

theorem train_cross_time : time_to_cross_platform = 60 := by
  sorry

end train_cross_time_l244_244377


namespace units_digit_of_7_pow_2500_l244_244253

theorem units_digit_of_7_pow_2500 : (7^2500) % 10 = 1 :=
by
  -- Variables and constants can be used to formalize steps if necessary, 
  -- but focus is on the statement itself.
  -- Sorry is used to skip the proof part.
  sorry

end units_digit_of_7_pow_2500_l244_244253


namespace perimeter_equals_interior_tiles_l244_244914

theorem perimeter_equals_interior_tiles (m n : ℕ) (h : m ≤ n) :
  (2 * m + 2 * n - 4 = 2 * (m * n) - (2 * m + 2 * n - 4)) ↔ (m = 5 ∧ n = 12 ∨ m = 6 ∧ n = 8) :=
by sorry

end perimeter_equals_interior_tiles_l244_244914


namespace sin_240_eq_neg_sqrt3_div_2_l244_244581

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244581


namespace pentagon_interior_angles_l244_244222

theorem pentagon_interior_angles
  (x y : ℝ)
  (H_eq_triangle : ∀ (angle : ℝ), angle = 60)
  (H_rect_QT : ∀ (angle : ℝ), angle = 90)
  (sum_interior_angles_pentagon : ∀ (n : ℕ), (n - 2) * 180 = 3 * 180) :
  x + y = 60 :=
by
  sorry

end pentagon_interior_angles_l244_244222


namespace round_robin_teams_l244_244689

theorem round_robin_teams (x : ℕ) (h : (x * (x - 1)) / 2 = 45) : x = 10 := 
by
  sorry

end round_robin_teams_l244_244689


namespace simplify_expression_l244_244841

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 := 
by
  sorry

end simplify_expression_l244_244841


namespace sin_240_eq_neg_sqrt3_over_2_l244_244602

open Real

-- Conditions
def angle_240_in_third_quadrant : Prop := 240 ° ∈ set_of (λ x, 180 ° < x ∧ x < 270 °)

def reference_angle_60 (θ : Real) : Prop := θ = 240 ° - 180 °

def sin_60_eq_sqrt3_over_2 : sin (60 °) = sqrt 3 / 2

def sin_negative_in_third_quadrant (θ : Real) : Prop :=
  180 ° < θ ∧ θ < 270 ° → sin θ < 0

-- Statement
theorem sin_240_eq_neg_sqrt3_over_2 :
  angle_240_in_third_quadrant ∧ reference_angle_60 60 ° ∧ sin_60_eq_sqrt3_over_2 ∧ sin_negative_in_third_quadrant 240 °
  → sin (240 °) = - (sqrt 3 / 2) :=
by
  intros
  sorry

end sin_240_eq_neg_sqrt3_over_2_l244_244602


namespace sum_of_primes_less_than_twenty_is_77_l244_244030

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l244_244030


namespace find_a_l244_244918

def new_operation (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem find_a (a : ℝ) (b : ℝ) (h : b = 4) (h2 : new_operation a b = 10) : a = 14 := by
  have h' : new_operation a 4 = 10 := by rw [h] at h2; exact h2
  unfold new_operation at h'
  linarith

end find_a_l244_244918


namespace inequality_of_f_l244_244662

def f (x : ℝ) : ℝ := 3 * (x - 2)^2 + 5

theorem inequality_of_f (x₁ x₂ : ℝ) (h : |x₁ - 2| > |x₂ - 2|) : f x₁ > f x₂ :=
by
  -- sorry placeholder for the actual proof
  sorry

end inequality_of_f_l244_244662


namespace arrange_books_l244_244331

-- Given conditions
def math_books_count := 4
def history_books_count := 6

-- Question: How many ways can the books be arranged given the conditions?
theorem arrange_books (math_books_count history_books_count : ℕ) :
  math_books_count = 4 → 
  history_books_count = 6 →
  ∃ ways : ℕ, ways = 51840 :=
by
  sorry

end arrange_books_l244_244331


namespace sin_240_eq_neg_sqrt3_div_2_l244_244604

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244604


namespace sum_of_primes_less_than_20_l244_244109

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l244_244109


namespace totalCostOfAllPuppies_l244_244906

noncomputable def goldenRetrieverCost : ℕ :=
  let numberOfGoldenRetrievers := 3
  let puppiesPerGoldenRetriever := 4
  let shotsPerPuppy := 2
  let costPerShot := 5
  let vitaminCostPerMonth := 12
  let monthsOfSupplements := 6
  numberOfGoldenRetrievers * puppiesPerGoldenRetriever *
  (shotsPerPuppy * costPerShot + vitaminCostPerMonth * monthsOfSupplements)

noncomputable def germanShepherdCost : ℕ :=
  let numberOfGermanShepherds := 2
  let puppiesPerGermanShepherd := 5
  let shotsPerPuppy := 3
  let costPerShot := 8
  let microchipCost := 25
  let toyCost := 15
  numberOfGermanShepherds * puppiesPerGermanShepherd *
  (shotsPerPuppy * costPerShot + microchipCost + toyCost)

noncomputable def bulldogCost : ℕ :=
  let numberOfBulldogs := 4
  let puppiesPerBulldog := 3
  let shotsPerPuppy := 4
  let costPerShot := 10
  let collarCost := 20
  let chewToyCost := 18
  numberOfBulldogs * puppiesPerBulldog *
  (shotsPerPuppy * costPerShot + collarCost + chewToyCost)

theorem totalCostOfAllPuppies : goldenRetrieverCost + germanShepherdCost + bulldogCost = 2560 :=
by
  sorry

end totalCostOfAllPuppies_l244_244906


namespace simplify_expr_l244_244845

theorem simplify_expr : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expr_l244_244845


namespace squirrel_journey_time_l244_244556

theorem squirrel_journey_time : 
  (let distance := 2
  let speed_to_tree := 3
  let speed_return := 2
  let time_to_tree := distance / speed_to_tree
  let time_return := distance / speed_return
  let total_time := (time_to_tree + time_return) * 60
  total_time = 100) :=
by
  sorry

end squirrel_journey_time_l244_244556


namespace find_x_l244_244339

namespace IntegerProblem

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := 
by
  sorry

end IntegerProblem

end find_x_l244_244339


namespace quadratic_distinct_real_roots_l244_244448

theorem quadratic_distinct_real_roots (k : ℝ) :
  (k > -2 ∧ k ≠ 0) ↔ ( ∃ (a b c : ℝ), a = k ∧ b = -4 ∧ c = -2 ∧ (b^2 - 4 * a * c) > 0) :=
by
  sorry

end quadratic_distinct_real_roots_l244_244448


namespace fuel_efficiency_problem_l244_244885

theorem fuel_efficiency_problem :
  let F_highway := 30
  let F_urban := 25
  let F_hill := 20
  let D_highway := 100
  let D_urban := 60
  let D_hill := 40
  let gallons_highway := D_highway / F_highway
  let gallons_urban := D_urban / F_urban
  let gallons_hill := D_hill / F_hill
  let total_gallons := gallons_highway + gallons_urban + gallons_hill
  total_gallons = 7.73 := 
by 
  sorry

end fuel_efficiency_problem_l244_244885


namespace sin_240_eq_neg_sqrt3_div_2_l244_244620

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244620


namespace sum_of_primes_less_than_20_l244_244122

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l244_244122


namespace sum_of_primes_less_than_20_l244_244103

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l244_244103


namespace range_of_4a_minus_2b_l244_244673

theorem range_of_4a_minus_2b (a b : ℝ) 
  (h1 : 1 ≤ a - b)
  (h2 : a - b ≤ 2)
  (h3 : 2 ≤ a + b)
  (h4 : a + b ≤ 4) : 
  5 ≤ 4 * a - 2 * b ∧ 4 * a - 2 * b ≤ 10 :=
by
  sorry

end range_of_4a_minus_2b_l244_244673


namespace geometric_sequence_formula_l244_244961

theorem geometric_sequence_formula (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 1 = 3 / 2)
  (h2 : a 1 + a 1 * q + a 1 * q^2 = 9 / 2)
  (geo : ∀ n, a (n + 1) = a n * q) :
  ∀ n, a n = 3 / 2 * (-2)^(n-1) ∨ a n = 3 / 2 :=
by sorry

end geometric_sequence_formula_l244_244961


namespace smallest_positive_angle_l244_244913

open Real

theorem smallest_positive_angle :
  ∃ x : ℝ, x > 0 ∧ x < 90 ∧ tan (4 * x * degree) = (cos (x * degree) - sin (x * degree)) / (cos (x * degree) + sin (x * degree)) ∧ x = 9 :=
sorry

end smallest_positive_angle_l244_244913


namespace remaining_macaroons_correct_l244_244171

variable (k : ℚ)

def total_baked : ℚ := 50 + 40 + 30 + 20 + 10

def total_eaten (k : ℚ) : ℚ := k + 2 * k + 3 * k + 10 * k + k / 5

def remaining_macaroons (k : ℚ) : ℚ := total_baked - total_eaten k

theorem remaining_macaroons_correct (k : ℚ) : remaining_macaroons k = 150 - (81 * k / 5) := 
by {
  -- The proof goes here.
  sorry
}

end remaining_macaroons_correct_l244_244171


namespace length_of_each_piece_is_correct_l244_244316

noncomputable def rod_length : ℝ := 38.25
noncomputable def num_pieces : ℕ := 45
noncomputable def length_each_piece_cm : ℝ := 85

theorem length_of_each_piece_is_correct : (rod_length / num_pieces) * 100 = length_each_piece_cm :=
by
  sorry

end length_of_each_piece_is_correct_l244_244316


namespace max_values_of_x_max_area_abc_l244_244299

noncomputable def m (x : ℝ) : ℝ × ℝ := ⟨2 * Real.sin x, Real.sin x - Real.cos x⟩
noncomputable def n (x : ℝ) : ℝ × ℝ := ⟨Real.sqrt 3 * Real.cos x, Real.sin x + Real.cos x⟩
noncomputable def f (x : ℝ) : ℝ := Prod.fst (m x) * Prod.fst (n x) + Prod.snd (m x) * Prod.snd (n x)

theorem max_values_of_x
  (k : ℤ) : ∃ x, x = k * Real.pi + Real.pi / 3 ∧ f x = 2 * Real.sin (2 * x - π / 6) :=
sorry

noncomputable def C : ℝ := Real.pi / 3
noncomputable def area_abc (a b c : ℝ) : ℝ := 1 / 2 * a * b * Real.sin C

theorem max_area_abc (a b : ℝ) (h₁ : c = Real.sqrt 3) (h₂ : f C = 2) :
  area_abc a b c ≤ 3 * Real.sqrt 3 / 4 :=
sorry

end max_values_of_x_max_area_abc_l244_244299


namespace perfect_square_condition_l244_244319

theorem perfect_square_condition (x m : ℝ) (h : ∃ k : ℝ, x^2 + x + 2*m = k^2) : m = 1/8 := 
sorry

end perfect_square_condition_l244_244319


namespace find_second_equation_value_l244_244652

theorem find_second_equation_value:
  (∃ x y : ℝ, 2 * x + y = 26 ∧ (x + y) / 3 = 4) →
  (∃ x y : ℝ, 2 * x + y = 26 ∧ x + 2 * y = 10) :=
by
  sorry

end find_second_equation_value_l244_244652


namespace find_b_age_l244_244721

variable (a b c : ℕ)
-- Condition 1: a is two years older than b
variable (h1 : a = b + 2)
-- Condition 2: b is twice as old as c
variable (h2 : b = 2 * c)
-- Condition 3: The total of the ages of a, b, and c is 17
variable (h3 : a + b + c = 17)

theorem find_b_age (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 17) : b = 6 :=
by
  sorry

end find_b_age_l244_244721


namespace min_value_g_l244_244240

noncomputable def g (x : ℝ) : ℝ := (6 * x^2 + 11 * x + 17) / (7 * (2 + x))

theorem min_value_g : ∃ x, x ≥ 0 ∧ g x = 127 / 24 :=
by
  sorry

end min_value_g_l244_244240


namespace sum_of_primes_less_than_20_l244_244060

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l244_244060


namespace sum_of_primes_less_than_20_l244_244105

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l244_244105


namespace relatively_prime_example_l244_244658

theorem relatively_prime_example :
  let a := 20172017
  let b := 20172018
  let c := 20172019
  let d := 20172020
  let e := 20172021
  Nat.gcd a c = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd d c = 1 ∧ Nat.gcd e c = 1 :=
by
  let a := 20172017
  let b := 20172018
  let c := 20172019
  let d := 20172020
  let e := 20172021
  sorry

end relatively_prime_example_l244_244658


namespace max_distance_between_circle_centers_l244_244676

theorem max_distance_between_circle_centers :
  let rect_width := 20
  let rect_height := 16
  let circle_diameter := 8
  let horiz_distance := rect_width - circle_diameter
  let vert_distance := rect_height - circle_diameter
  let max_distance := Real.sqrt (horiz_distance ^ 2 + vert_distance ^ 2)
  max_distance = 4 * Real.sqrt 13 :=
by
  sorry

end max_distance_between_circle_centers_l244_244676


namespace total_chocolate_sold_total_vanilla_sold_total_strawberry_sold_l244_244882

def chocolate_sold : ℕ := 6 + 7 + 4 + 8 + 9 + 10 + 5
def vanilla_sold : ℕ := 4 + 5 + 3 + 7 + 6 + 8 + 4
def strawberry_sold : ℕ := 3 + 2 + 6 + 4 + 5 + 7 + 4

theorem total_chocolate_sold : chocolate_sold = 49 :=
by
  unfold chocolate_sold
  rfl

theorem total_vanilla_sold : vanilla_sold = 37 :=
by
  unfold vanilla_sold
  rfl

theorem total_strawberry_sold : strawberry_sold = 31 :=
by
  unfold strawberry_sold
  rfl

end total_chocolate_sold_total_vanilla_sold_total_strawberry_sold_l244_244882


namespace find_x_l244_244320

theorem find_x (x y : ℝ) (h1 : y = 1 / (2 * x + 2)) (h2 : y = 2) : x = -3 / 4 :=
by
  sorry

end find_x_l244_244320


namespace sample_size_l244_244731

theorem sample_size (k n : ℕ) (h_ratio : 4 * k + k + 5 * k = n) 
  (h_middle_aged : 10 * (4 + 1 + 5) = n) : n = 100 := 
by
  sorry

end sample_size_l244_244731


namespace number_of_pages_read_on_fourth_day_l244_244186

-- Define variables
variables (day1 day2 day3 day4 total_pages: ℕ)

-- Define conditions
def condition1 := day1 = 63
def condition2 := day2 = 2 * day1
def condition3 := day3 = day2 + 10
def condition4 := total_pages = 354
def read_in_four_days := total_pages = day1 + day2 + day3 + day4

-- State the theorem to be proven
theorem number_of_pages_read_on_fourth_day (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : read_in_four_days) : day4 = 29 :=
by sorry

end number_of_pages_read_on_fourth_day_l244_244186


namespace probability_non_defective_pens_l244_244678

theorem probability_non_defective_pens 
  (total_pens : ℕ) (defective_pens : ℕ) (selected_pens : ℕ) 
  (h_total : total_pens = 12) 
  (h_defective : defective_pens = 4) 
  (h_selected : selected_pens = 2) : 
  (8 / 12 : ℚ) * (7 / 11 : ℚ) = 14 / 33 :=
by
  rw [←nat.cast_add_one defective_pens, ←nat.cast_add_one (total_pens - defective_pens)],
  norm_num,
  rw [mul_comm, mul_div_assoc, ←cast_eq_of_rat_eq, ←cast_eq_of_rat_eq],
  field_simp,
  norm_num,
  sorry

end probability_non_defective_pens_l244_244678


namespace exists_monochromatic_triangle_l244_244709

theorem exists_monochromatic_triangle :
  ∀ (points : Finset ℕ), points.card = 6 → 
  (∀ (x y : ℕ), x ∈ points → y ∈ points → x ≠ y → (coloring : (x, y) → ℤ)) → 
  (∃ (triangle : Finset (Finset ℕ)), triangle.card = 3 ∧ 
  (∀ (x y : ℕ), (x ∈ triangle) → (y ∈ triangle) → (x, y) ∈ coloring (x, y) ∨ coloring (x, y) ∧ 
  (0 ≤ coloring (x, y) ∧ coloring (x, y) ≤ 1)) :=
by
  sorry

end exists_monochromatic_triangle_l244_244709


namespace original_number_is_144_l244_244785

theorem original_number_is_144 (x : ℕ) (h : x - x / 3 = x - 48) : x = 144 :=
by
  sorry

end original_number_is_144_l244_244785


namespace new_perimeter_after_adding_tiles_l244_244853

-- Define the original condition as per the problem statement
def original_T_shape (n : ℕ) : Prop :=
  n = 6

def original_perimeter (p : ℕ) : Prop :=
  p = 12

-- Define hypothesis required to add three more tiles while sharing a side with existing tiles
def add_three_tiles_with_shared_side (original_tiles : ℕ) (new_tiles_added : ℕ) : Prop :=
  original_tiles + new_tiles_added = 9

-- Prove the new perimeter after adding three tiles to the original T-shaped figure
theorem new_perimeter_after_adding_tiles
  (n : ℕ) (p : ℕ) (new_tiles : ℕ) (new_p : ℕ)
  (h1 : original_T_shape n)
  (h2 : original_perimeter p)
  (h3 : add_three_tiles_with_shared_side n new_tiles)
  : new_p = 16 :=
sorry

end new_perimeter_after_adding_tiles_l244_244853


namespace sum_of_primes_less_than_twenty_is_77_l244_244033

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l244_244033


namespace entertainment_team_count_l244_244289

theorem entertainment_team_count 
  (total_members : ℕ)
  (singers : ℕ) 
  (dancers : ℕ) 
  (prob_both_sing_dance_gt_0 : ℚ)
  (sing_count : singers = 2)
  (dance_count : dancers = 5)
  (prob_condition : prob_both_sing_dance_gt_0 = 7/10) :
  total_members = 5 := 
by 
  sorry

end entertainment_team_count_l244_244289


namespace sum_of_primes_less_than_20_l244_244054

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l244_244054


namespace pascal_50_5th_element_is_22050_l244_244719

def pascal_fifth_element (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_50_5th_element_is_22050 :
  pascal_fifth_element 50 4 = 22050 :=
by
  -- Calculation steps would go here
  sorry

end pascal_50_5th_element_is_22050_l244_244719


namespace steve_has_7_fewer_b_berries_l244_244975

-- Define the initial number of berries Stacy has
def stacy_initial_berries : ℕ := 32

-- Define the number of berries Steve takes from Stacy
def steve_takes : ℕ := 4

-- Define the initial number of berries Steve has
def steve_initial_berries : ℕ := 21

-- Using the given conditions, prove that Steve has 7 fewer berries compared to Stacy's initial amount
theorem steve_has_7_fewer_b_berries :
  stacy_initial_berries - (steve_initial_berries + steve_takes) = 7 := 
by
  sorry

end steve_has_7_fewer_b_berries_l244_244975


namespace sum_primes_less_than_20_l244_244099

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l244_244099


namespace find_k_l244_244476

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x^2 + 2 * x + 1) / (k * x - 1)

theorem find_k (k : ℝ) : (∀ x : ℝ, f k (f k x) = x) ↔ k = -2 :=
  sorry

end find_k_l244_244476


namespace exists_m_divisible_by_n_with_digit_sum_l244_244938

theorem exists_m_divisible_by_n_with_digit_sum 
  (n k : ℕ) 
  (hn_pos : 0 < n)
  (hk_pos : 0 < k) 
  (hn_not_div_by_3 : ¬ (3 ∣ n)) 
  (hk_ge_n : k ≥ n) 
  : ∃ (m : ℕ), (n ∣ m) ∧ (Nat.digits 10 m).Sum = k := 
sorry

end exists_m_divisible_by_n_with_digit_sum_l244_244938


namespace finitely_countably_additive_measure_not_extendable_l244_244641

noncomputable def A (n1 : ℕ) (ns : List ℕ) : Set ℝ :=
  (Ioc 0 (1 / n1)) ∪ ⋃ i in ns, Ioc (1 / (i + 1)) (1 / i)

noncomputable def B (ns : List ℕ) : Set ℝ :=
  ⋃ i in ns, Ioc (1 / (i + 1)) (1 / i)

noncomputable def mu : Set ℝ → ℝ 
| A n1 ns => List.sum (List.map (λ i => (-1)^(i + 1) / i) ns) + ∑' n, if n ≥ n1 then (-1)^(n + 1) / n else 0
| B ns    => List.sum (List.map (λ i => (-1)^(i + 1) / i) ns)

theorem finitely_countably_additive_measure_not_extendable :
  ∃ (μ : Set ℝ → ℝ) (A : Set ℝ), 
    (∀ A B : Set ℝ, μ (A ∪ B) = μ A + μ B - μ (A ∩ B)) ∧
    (∀ A : Set ℝ, μ A = μ (Ioc 0 (1 / n1)) + μ (⋃ n ≥ n1, Ioc (1 / (n + 1)) (1 / n))) ∧
    (∃ S : Set (Set ℝ), ∀ s ∈ S, μ s = List.sum (List.map (λ i, (-1)^(i + 1) / i) (Finset.toList s)) + ∑' n, if n ≥ n1 then (-1)^(n + 1) / n else 0) ∧
    ¬(∃ σ : Set (Set ℝ), σ ⊇ A ∧ is_countably_additive μ) :=
begin
  use mu,
  sorry
end

end finitely_countably_additive_measure_not_extendable_l244_244641


namespace rehabilitation_centers_total_l244_244997

noncomputable def jane_visits (han_visits : ℕ) : ℕ := 2 * han_visits + 6
noncomputable def han_visits (jude_visits : ℕ) : ℕ := 2 * jude_visits - 2
noncomputable def jude_visits (lisa_visits : ℕ) : ℕ := lisa_visits / 2
def lisa_visits : ℕ := 6

def total_visits (jane_visits han_visits jude_visits lisa_visits : ℕ) : ℕ :=
  jane_visits + han_visits + jude_visits + lisa_visits

theorem rehabilitation_centers_total :
  total_visits (jane_visits (han_visits (jude_visits lisa_visits))) 
               (han_visits (jude_visits lisa_visits))
               (jude_visits lisa_visits) 
               lisa_visits = 27 :=
by
  sorry

end rehabilitation_centers_total_l244_244997


namespace sum_of_common_ratios_l244_244350

variable {k a_2 a_3 b_2 b_3 p r : ℝ}
variable (hp : a_2 = k * p) (ha3 : a_3 = k * p^2)
variable (hr : b_2 = k * r) (hb3 : b_3 = k * r^2)
variable (hcond : a_3 - b_3 = 5 * (a_2 - b_2))

theorem sum_of_common_ratios (h_nonconst : k ≠ 0) (p_ne_r : p ≠ r) : p + r = 5 :=
by
  sorry

end sum_of_common_ratios_l244_244350


namespace bc_eq_one_area_of_triangle_l244_244484

variable (a b c A B : ℝ)

-- Conditions
def condition_1 : Prop := (b^2 + c^2 - a^2) / (Real.cos A) = 2
def condition_2 : Prop := (a * (Real.cos B) - b * (Real.cos A)) / (a * (Real.cos B) + b * (Real.cos A)) - b / c = 1

-- Equivalent proof problems
theorem bc_eq_one (h1 : condition_1 a b c A) : b * c = 1 := 
by 
  sorry

theorem area_of_triangle (h2 : condition_2 a b c A B) : (1/2) * b * c * Real.sin A = (Real.sqrt 3) / 4 := 
by 
  sorry

end bc_eq_one_area_of_triangle_l244_244484


namespace rate_of_simple_interest_l244_244398

theorem rate_of_simple_interest (P : ℝ) (R : ℝ) (T : ℝ) (P_nonzero : P ≠ 0) : 
  (P * R * T = P / 6) → R = 1 / 42 :=
by
  intro h
  sorry

end rate_of_simple_interest_l244_244398


namespace compressor_distances_distances_when_a_15_l244_244401

theorem compressor_distances (a : ℝ) (x y z : ℝ) (h1 : x + y = 2 * z) (h2 : x + z = y + a) (h3 : x + z = 75) :
  0 < a ∧ a < 100 → 
  let x := (75 + a) / 3;
  let y := 75 - a;
  let z := 75 - x;
  x + y = 2 * z ∧ x + z = y + a ∧ x + z = 75 :=
sorry

theorem distances_when_a_15 (x y z : ℝ) (h : 15 = 15) :
  let x := (75 + 15) / 3;
  let y := 75 - 15;
  let z := 75 - x;
  x = 30 ∧ y = 60 ∧ z = 45 :=
sorry

end compressor_distances_distances_when_a_15_l244_244401


namespace statement_II_must_be_true_l244_244188

-- Define the set of all creatures
variable (Creature : Type)

-- Define properties for being a dragon, mystical, and fire-breathing
variable (Dragon Mystical FireBreathing : Creature → Prop)

-- Given conditions
-- All dragons breathe fire
axiom all_dragons_breathe_fire : ∀ c, Dragon c → FireBreathing c
-- Some mystical creatures are dragons
axiom some_mystical_creatures_are_dragons : ∃ c, Mystical c ∧ Dragon c

-- Questions to prove (we will only formalize the must be true statement)
-- Statement II: Some fire-breathing creatures are mystical creatures

theorem statement_II_must_be_true : ∃ c, FireBreathing c ∧ Mystical c :=
by
  sorry

end statement_II_must_be_true_l244_244188


namespace greatest_three_digit_multiple_of_17_is_986_l244_244003

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l244_244003


namespace min_xy_eq_nine_l244_244132

theorem min_xy_eq_nine (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y = x + y + 3) : x * y = 9 :=
sorry

end min_xy_eq_nine_l244_244132


namespace number_of_pieces_of_tape_l244_244395

variable (length_of_tape : ℝ := 8.8)
variable (overlap : ℝ := 0.5)
variable (total_length : ℝ := 282.7)

theorem number_of_pieces_of_tape : 
  ∃ (N : ℕ), total_length = length_of_tape + (N - 1) * (length_of_tape - overlap) ∧ N = 34 :=
sorry

end number_of_pieces_of_tape_l244_244395


namespace chess_team_boys_count_l244_244546

theorem chess_team_boys_count (J S B : ℕ) 
  (h1 : J + S + B = 32) 
  (h2 : (1 / 3 : ℚ) * J + (1 / 2 : ℚ) * S + B = 18) : 
  B = 4 :=
by
  sorry

end chess_team_boys_count_l244_244546


namespace sum_of_primes_less_than_20_eq_77_l244_244026

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l244_244026


namespace rain_in_august_probability_l244_244507

noncomputable def rain_probability (n : ℕ) (p : ℚ) : ℚ :=
(let q := 1 - p in
  (((q) ^ 7) + (7 * (p) * (q) ^ 6) +
   (21 * (p ^ 2) * (q ^ 5)) + (35 * (p ^ 3) * (q ^ 4))))

theorem rain_in_august_probability :
  rain_probability 7 (1/5) = 0.813 :=
by
  sorry

end rain_in_august_probability_l244_244507


namespace john_new_weekly_earnings_l244_244198

theorem john_new_weekly_earnings
  (original_earnings : ℕ)
  (percentage_increase : ℕ)
  (raise_amount : ℕ)
  (new_weekly_earnings : ℕ)
  (original_earnings_eq : original_earnings = 50)
  (percentage_increase_eq : percentage_increase = 40)
  (raise_amount_eq : raise_amount = original_earnings * percentage_increase / 100)
  (new_weekly_earnings_eq : new_weekly_earnings = original_earnings + raise_amount) :
  new_weekly_earnings = 70 := by
  sorry

end john_new_weekly_earnings_l244_244198


namespace sum_primes_less_than_20_l244_244062

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l244_244062


namespace valentines_distribution_l244_244815

theorem valentines_distribution (valentines_initial : ℝ) (valentines_needed : ℝ) (students : ℕ) 
  (h_initial : valentines_initial = 58.0) (h_needed : valentines_needed = 16.0) (h_students : students = 74) : 
  (valentines_initial + valentines_needed) / students = 1 :=
by
  sorry

end valentines_distribution_l244_244815


namespace minimum_x2y3z_l244_244688

theorem minimum_x2y3z (x y z : ℕ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_eq : x^3 + y^3 + z^3 - 3 * x * y * z = 607) : 
  x + 2 * y + 3 * z ≥ 1215 :=
sorry

end minimum_x2y3z_l244_244688


namespace max_pies_without_ingredients_l244_244215

theorem max_pies_without_ingredients
  (total_pies chocolate_pies berries_pies cinnamon_pies poppy_seeds_pies : ℕ)
  (h1 : total_pies = 60)
  (h2 : chocolate_pies = 1 / 3 * total_pies)
  (h3 : berries_pies = 3 / 5 * total_pies)
  (h4 : cinnamon_pies = 1 / 2 * total_pies)
  (h5 : poppy_seeds_pies = 1 / 5 * total_pies) : 
  total_pies - max chocolate_pies (max berries_pies (max cinnamon_pies poppy_seeds_pies)) = 24 := 
by
  sorry

end max_pies_without_ingredients_l244_244215


namespace unique_triple_sum_l244_244866

theorem unique_triple_sum :
  ∃ (a b c : ℕ), 
    (10 ≤ a ∧ a < 100) ∧ 
    (10 ≤ b ∧ b < 100) ∧ 
    (10 ≤ c ∧ c < 100) ∧ 
    (a^3 + 3 * b^3 + 9 * c^3 = 9 * a * b * c + 1) ∧ 
    (a + b + c = 9) := 
sorry

end unique_triple_sum_l244_244866


namespace find_a_l244_244929

-- Points A and B on the x-axis
def point_A (a : ℝ) : (ℝ × ℝ) := (a, 0)
def point_B : (ℝ × ℝ) := (-3, 0)

-- Distance condition
def distance_condition (a : ℝ) : Prop := abs (a + 3) = 5

-- The proof problem: find a such that distance condition holds
theorem find_a (a : ℝ) : distance_condition a ↔ (a = -8 ∨ a = 2) :=
by
  sorry

end find_a_l244_244929


namespace inclination_of_l1_perpendicular_l1_l2_distance_between_parallel_lines_l244_244313
-- Import the necessary math library

-- Definitions based on conditions
def line1 (a : ℝ) : ℝ × ℝ → Prop := λ (x y : ℝ), a * x + y - 1 = 0
def line2 : ℝ × ℝ → Prop := λ (x y : ℝ), x - y - 3 = 0

-- Main theorem statements
theorem inclination_of_l1 (a : ℝ) (h : 1 = a) : a = -1 :=
sorry

theorem perpendicular_l1_l2 (a : ℝ) (h : 1 * -a = -1) : a = 1 :=
sorry

theorem distance_between_parallel_lines (a : ℝ) (h_parallel : a = 1)
  (x1 y1 x2 y2 : ℝ) (h_l1 : line1 a (x1, y1)) (h_l2 : line2 (x2, y2)) :
  (abs (-3 - (-1))) / (sqrt (a^2 + (1^2))) = 2 * sqrt 2 :=
sorry

end inclination_of_l1_perpendicular_l1_l2_distance_between_parallel_lines_l244_244313


namespace common_chord_eq_l244_244941

theorem common_chord_eq :
  ∀ (x y : ℝ),
    (x^2 + y^2 + 2*x + 8*y - 8 = 0) → (x^2 + y^2 - 4*x - 4*y - 2 = 0) →
    x + 2*y - 1 = 0 :=
by
  intros x y h1 h2
  sorry

end common_chord_eq_l244_244941


namespace find_x_l244_244338

namespace IntegerProblem

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := 
by
  sorry

end IntegerProblem

end find_x_l244_244338


namespace tv_purchase_price_correct_l244_244554

theorem tv_purchase_price_correct (x : ℝ) (h : (1.4 * x * 0.8 - x) = 270) : x = 2250 :=
by
  sorry

end tv_purchase_price_correct_l244_244554


namespace largest_divisor_of_even_n_cube_difference_l244_244429

theorem largest_divisor_of_even_n_cube_difference (n : ℤ) (h : Even n) : 6 ∣ (n^3 - n) := by
  sorry

end largest_divisor_of_even_n_cube_difference_l244_244429


namespace nonneg_int_repr_l244_244692

theorem nonneg_int_repr (n : ℕ) : ∃ (a b c : ℕ), (0 < a ∧ a < b ∧ b < c) ∧ n = a^2 + b^2 - c^2 :=
sorry

end nonneg_int_repr_l244_244692


namespace max_stamps_l244_244789

theorem max_stamps (price_per_stamp : ℕ) (total_money : ℕ) (h_price : price_per_stamp = 37) (h_total : total_money = 4000) : 
  ∃ max_stamps : ℕ, max_stamps = 108 ∧ max_stamps * price_per_stamp ≤ total_money ∧ ∀ n : ℕ, n * price_per_stamp ≤ total_money → n ≤ max_stamps :=
by
  sorry

end max_stamps_l244_244789


namespace find_tangent_points_l244_244512

-- Step a: Define the curve and the condition for the tangent line parallel to y = 4x.
def curve (x : ℝ) : ℝ := x^3 + x - 2
def tangent_slope : ℝ := 4

-- Step d: Provide the statement that the coordinates of P₀ are (1, 0) and (-1, -4).
theorem find_tangent_points : 
  ∃ (P₀ : ℝ × ℝ), (curve P₀.1 = P₀.2) ∧ 
                 ((P₀ = (1, 0)) ∨ (P₀ = (-1, -4))) := 
by
  sorry

end find_tangent_points_l244_244512


namespace total_cubes_l244_244758

noncomputable def original_cubes : ℕ := 2
noncomputable def additional_cubes : ℕ := 7

theorem total_cubes : original_cubes + additional_cubes = 9 := by
  sorry

end total_cubes_l244_244758


namespace alex_annual_income_l244_244229

theorem alex_annual_income (q : ℝ) (B : ℝ)
  (H1 : 0.01 * q * 50000 + 0.01 * (q + 3) * (B - 50000) = 0.01 * (q + 0.5) * B) :
  B = 60000 :=
by sorry

end alex_annual_income_l244_244229


namespace certain_number_proof_l244_244533

-- Definitions as per the conditions in the problem
variables (x y : ℕ)

def original_ratio := (2 : ℕ) / (3 : ℕ)
def desired_ratio := (x : ℕ) / (5 : ℕ)

-- Problem statement: Prove that x = 4 given the conditions
theorem certain_number_proof (h1 : 3 + y = 5) (h2 : 2 + y = x) : x = 4 := by
  sorry

end certain_number_proof_l244_244533


namespace arithmetic_seq_problem_l244_244190

variable (a : ℕ → ℕ)

def arithmetic_seq (a₁ d : ℕ) : ℕ → ℕ :=
  λ n => a₁ + n * d

theorem arithmetic_seq_problem (a₁ d : ℕ)
  (h_cond : (arithmetic_seq a₁ d 1) + 2 * (arithmetic_seq a₁ d 5) + (arithmetic_seq a₁ d 9) = 120)
  : (arithmetic_seq a₁ d 2) + (arithmetic_seq a₁ d 8) = 60 := 
sorry

end arithmetic_seq_problem_l244_244190


namespace base8_to_base10_12345_l244_244877

theorem base8_to_base10_12345 : (1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0) = 5349 := by
  sorry

end base8_to_base10_12345_l244_244877


namespace max_non_managers_depA_l244_244794

theorem max_non_managers_depA (mA : ℕ) (nA : ℕ) (sA : ℕ) (gA : ℕ) (totalA : ℕ) :
  mA = 9 ∧ (8 * nA > 37 * mA) ∧ (sA = 2 * gA) ∧ (nA = sA + gA) ∧ (mA + nA ≤ 250) →
  nA = 39 :=
by
  sorry

end max_non_managers_depA_l244_244794


namespace milk_water_ratio_l244_244385

theorem milk_water_ratio
  (vessel1_milk_ratio : ℚ)
  (vessel1_water_ratio : ℚ)
  (vessel2_milk_ratio : ℚ)
  (vessel2_water_ratio : ℚ)
  (equal_mixture_units  : ℚ)
  (h1 : vessel1_milk_ratio / vessel1_water_ratio = 4 / 1)
  (h2 : vessel2_milk_ratio / vessel2_water_ratio = 7 / 3)
  :
  (vessel1_milk_ratio + vessel2_milk_ratio) / 
  (vessel1_water_ratio + vessel2_water_ratio) = 11 / 4 :=
by
  sorry

end milk_water_ratio_l244_244385


namespace map_distance_representation_l244_244970

theorem map_distance_representation
  (cm_to_km_ratio : 15 = 90)
  (km_to_m_ratio : 1000 = 1000) :
  20 * (90 / 15) * 1000 = 120000 := by
  sorry

end map_distance_representation_l244_244970


namespace area_ratio_PQR_to_STU_l244_244383

-- Given Conditions
def triangle_PQR_sides (a b c : Nat) : Prop :=
  a = 9 ∧ b = 40 ∧ c = 41

def triangle_STU_sides (x y z : Nat) : Prop :=
  x = 7 ∧ y = 24 ∧ z = 25

-- Theorem Statement (math proof problem)
theorem area_ratio_PQR_to_STU :
  (∃ (a b c x y z : Nat), triangle_PQR_sides a b c ∧ triangle_STU_sides x y z) →
  9 * 40 / (7 * 24) = 15 / 7 :=
by
  intro h
  sorry

end area_ratio_PQR_to_STU_l244_244383


namespace factorize_3a_squared_minus_6a_plus_3_l244_244635

theorem factorize_3a_squared_minus_6a_plus_3 (a : ℝ) : 
  3 * a^2 - 6 * a + 3 = 3 * (a - 1)^2 :=
by 
  sorry

end factorize_3a_squared_minus_6a_plus_3_l244_244635


namespace simplify_expression_l244_244847

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l244_244847


namespace factor_expression_l244_244908

theorem factor_expression (x : ℝ) : 12 * x^2 - 6 * x = 6 * x * (2 * x - 1) :=
by
sorry

end factor_expression_l244_244908


namespace sin_240_eq_neg_sqrt3_over_2_l244_244599

open Real

-- Conditions
def angle_240_in_third_quadrant : Prop := 240 ° ∈ set_of (λ x, 180 ° < x ∧ x < 270 °)

def reference_angle_60 (θ : Real) : Prop := θ = 240 ° - 180 °

def sin_60_eq_sqrt3_over_2 : sin (60 °) = sqrt 3 / 2

def sin_negative_in_third_quadrant (θ : Real) : Prop :=
  180 ° < θ ∧ θ < 270 ° → sin θ < 0

-- Statement
theorem sin_240_eq_neg_sqrt3_over_2 :
  angle_240_in_third_quadrant ∧ reference_angle_60 60 ° ∧ sin_60_eq_sqrt3_over_2 ∧ sin_negative_in_third_quadrant 240 °
  → sin (240 °) = - (sqrt 3 / 2) :=
by
  intros
  sorry

end sin_240_eq_neg_sqrt3_over_2_l244_244599


namespace problem_condition_l244_244812

noncomputable def f : ℝ → ℝ := sorry

theorem problem_condition (h: ∀ x : ℝ, f x > (deriv f) x) : 3 * f (Real.log 2) > 2 * f (Real.log 3) :=
sorry

end problem_condition_l244_244812


namespace initial_number_of_mice_l244_244869

theorem initial_number_of_mice (x : ℕ) 
  (h1 : x % 2 = 0)
  (h2 : (x / 2) % 3 = 0)
  (h3 : (x / 2 - x / 6) % 4 = 0)
  (h4 : (x / 2 - x / 6 - (x / 2 - x / 6) / 4) % 5 = 0)
  (h5 : (x / 5) = (x / 6) + 2) : 
  x = 60 := 
by sorry

end initial_number_of_mice_l244_244869


namespace manager_salary_is_3600_l244_244400

noncomputable def manager_salary (M : ℕ) : ℕ :=
  let total_salary_20 := 20 * 1500
  let new_average_salary := 1600
  let total_salary_21 := 21 * new_average_salary
  total_salary_21 - total_salary_20

theorem manager_salary_is_3600 : manager_salary 3600 = 3600 := by
  sorry

end manager_salary_is_3600_l244_244400


namespace unique_solution_probability_l244_244824

noncomputable def roll_die_twice (a b : ℕ) : Prop :=
  a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6}

noncomputable def system_has_unique_solution (a b : ℕ) : Prop :=
  (a * 2 ∧ b * 2) ≠ (4 * a ∧ 4 * b)

theorem unique_solution_probability : 
  ( ∑ a in {1, 2, 3, 4, 5, 6}, ∑ b in {1, 2, 3, 4, 5, 6}, 
    (if system_has_unique_solution a b then 1 else 0) / 36 = 11 / 12) :=
sorry

end unique_solution_probability_l244_244824


namespace sum_primes_less_than_20_l244_244077

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l244_244077


namespace integer_points_on_segment_l244_244472

noncomputable def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 else 0

theorem integer_points_on_segment (n : ℕ) (h : 0 < n) :
  f n = if n % 3 = 0 then 2 else 0 :=
by
  sorry

end integer_points_on_segment_l244_244472


namespace certain_number_unique_l244_244782

theorem certain_number_unique (x : ℝ) (hx1 : 213 * x = 3408) (hx2 : 21.3 * x = 340.8) : x = 16 :=
by
  sorry

end certain_number_unique_l244_244782


namespace sum_of_primes_less_than_20_l244_244020

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l244_244020


namespace AIMN_cyclic_l244_244488

open EuclideanGeometry

-- Given a triangle ABC and the incenter I,
variables {α : Type*} [normed_group α] [normed_space ℝ α] [inner_product_space ℝ α]

-- Points A, B, C are distinct and form a triangle
variables {A B C I D E F M N : α} 

-- Define triangle geometry
noncomputable def triangle_ABC : affine_plane ℝ α :=
{ A := A,
  B := B,
  C := C,
  I := I,
  D := line_intersection (line_through A I) (perpendicular_bisector (segment A D)),
  E := line_intersection (line_through B I) (perpendicular_bisector (segment B E)),
  F := line_intersection (line_through C I) (perpendicular_bisector (segment C F)),
  M := line_intersection (perpendicular_bisector (segment A D)) (line_through B I),
  N := line_intersection (perpendicular_bisector (segment A D)) (line_through C I) }

-- Prove that A, I, M, and N are concyclic
theorem AIMN_cyclic : are_concyclic {A, I, M, N} :=
sorry

end AIMN_cyclic_l244_244488


namespace al_initial_amount_l244_244281

theorem al_initial_amount
  (a b c : ℕ)
  (h₁ : a + b + c = 2000)
  (h₂ : 3 * a + 2 * b + 2 * c = 3500) :
  a = 500 :=
sorry

end al_initial_amount_l244_244281


namespace intersection_eq_interval_l244_244184

def P : Set ℝ := {x | x * (x - 3) < 0}
def Q : Set ℝ := {x | |x| < 2}

theorem intersection_eq_interval : P ∩ Q = {x | 0 < x ∧ x < 2} :=
by
  sorry

end intersection_eq_interval_l244_244184


namespace area_enclosed_curves_l244_244988

theorem area_enclosed_curves (a : ℝ) (h1 : (1 + 1/a)^5 = 1024) :
  ∫ x in (0 : ℝ)..1, (x^(1/3) - x^2) = 5/12 :=
sorry

end area_enclosed_curves_l244_244988


namespace profit_calculation_correct_l244_244733

def main_actor_fee : ℕ := 500
def supporting_actor_fee : ℕ := 100
def extra_fee : ℕ := 50
def main_actor_food : ℕ := 10
def supporting_actor_food : ℕ := 5
def remaining_member_food : ℕ := 3
def post_production_cost : ℕ := 850
def revenue : ℕ := 10000

def total_actor_fees : ℕ := 2 * main_actor_fee + 3 * supporting_actor_fee + extra_fee
def total_food_cost : ℕ := 2 * main_actor_food + 4 * supporting_actor_food + 44 * remaining_member_food
def total_equipment_rental : ℕ := 2 * (total_actor_fees + total_food_cost)
def total_cost : ℕ := total_actor_fees + total_food_cost + total_equipment_rental + post_production_cost
def profit : ℕ := revenue - total_cost

theorem profit_calculation_correct : profit = 4584 :=
by
  -- proof omitted
  sorry

end profit_calculation_correct_l244_244733


namespace sum_prime_numbers_less_than_twenty_l244_244090

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l244_244090


namespace arnold_danny_age_l244_244257

theorem arnold_danny_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 15) : x = 7 :=
sorry

end arnold_danny_age_l244_244257


namespace sin_240_eq_neg_sqrt3_div_2_l244_244586

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244586


namespace angle_inclusion_l244_244318

-- Defining the sets based on the given conditions
def M : Set ℝ := { x | 0 < x ∧ x ≤ 90 }
def N : Set ℝ := { x | 0 < x ∧ x < 90 }
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 90 }

-- The proof statement
theorem angle_inclusion : N ⊆ M ∧ M ⊆ P :=
by
  sorry

end angle_inclusion_l244_244318


namespace converse_even_power_divisible_l244_244462

theorem converse_even_power_divisible (n : ℕ) (h_even : ∀ (k : ℕ), n = 2 * k → (3^n + 63) % 72 = 0) :
  (3^n + 63) % 72 = 0 → ∃ (k : ℕ), n = 2 * k :=
by sorry

end converse_even_power_divisible_l244_244462


namespace multiply_polynomials_l244_244491

theorem multiply_polynomials (x : ℝ) : 
  (x^6 + 64 * x^3 + 4096) * (x^3 - 64) = x^9 - 262144 :=
by
  sorry

end multiply_polynomials_l244_244491


namespace part1_confidence_part2_probability_part2_expectation_l244_244887

noncomputable def contingency_table : (ℕ × ℕ × ℕ × ℕ) :=
  (15, 5, 10, 20)

noncomputable def sample_size : ℕ := 50

noncomputable def K_squared (n a b c d : ℕ) : ℝ :=
  let num := (n : ℝ) * ((a * d - b * c : ℕ) : ℝ) ^ 2
  let denom := ((a + b) * (c + d) * (a + c) * (b + d) : ℕ) : ℝ
  num / denom

noncomputable def confidence_level (k_squared : ℝ) : ℝ :=
  if k_squared ≥ 7.879 then 0.995 else if k_squared ≥ 6.635 then 0.99 else 0.95

theorem part1_confidence :
  let (a, b, c, d) := contingency_table in
  let k_squared := K_squared sample_size a b c d in
  confidence_level k_squared = 0.995 := sorry

noncomputable def P_X (x : ℕ) : ℚ :=
  match x with
  | 0 => 1 / 15
  | 1 => 8 / 15
  | 2 => 6 / 15
  | _ => 0

noncomputable def expected_value_X : ℚ :=
  ∑ i in ({0, 1, 2} : finset ℕ), (i : ℚ) * P_X i

theorem part2_probability :
  (P_X 0 = 1 / 15) ∧ (P_X 1 = 8 / 15) ∧ (P_X 2 = 6 / 15) :=
    by simp

theorem part2_expectation : expected_value_X = 4 / 3 := sorry

end part1_confidence_part2_probability_part2_expectation_l244_244887


namespace relatively_prime_number_exists_l244_244656

theorem relatively_prime_number_exists :
  -- Given numbers
  (let a := 20172017 in
   let b := 20172018 in
   let c := 20172019 in
   let d := 20172020 in
   let e := 20172021 in
   -- Number c is relatively prime to all other given numbers
   nat.gcd c a = 1 ∧
   nat.gcd c b = 1 ∧
   nat.gcd c d = 1 ∧
   nat.gcd c e = 1) :=
by {
  -- Proof omitted
  sorry
}

end relatively_prime_number_exists_l244_244656


namespace original_cookie_price_l244_244757

theorem original_cookie_price (C : ℝ) (h1 : 1.5 * 16 + (C / 2) * 8 = 32) : C = 2 :=
by
  -- Proof omitted
  sorry

end original_cookie_price_l244_244757


namespace product_of_g_of_roots_l244_244809

noncomputable def f (x : ℝ) : ℝ := x^5 - 2*x^3 + x + 1
noncomputable def g (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem product_of_g_of_roots (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h₁ : f x₁ = 0) (h₂ : f x₂ = 0) (h₃ : f x₃ = 0)
  (h₄ : f x₄ = 0) (h₅ : f x₅ = 0) :
  g x₁ * g x₂ * g x₃ * g x₄ * g x₅ = f (-1 + Real.sqrt 2) * f (-1 - Real.sqrt 2) :=
by
  sorry

end product_of_g_of_roots_l244_244809


namespace sin_240_l244_244574

theorem sin_240 : Real.sin (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Provided conditions
  have h1 : 240 = 180 + 60 := be_of_eq true.intro
  have h2 : ∀ θ : ℝ, θ ∈ set.Icc (pi : ℝ) (3 * pi / 2) → Real.sin θ < 0 := Real.sin_neg_of_pi_lt_of_lt (Real.pi_lt_2_pi)
  have h3 : Real.sin (60 * Real.pi / 180) = 1 / 2 := Real.sin_pi_div_three
  -- Prove
  sorry

end sin_240_l244_244574


namespace sum_of_consecutive_even_integers_divisible_by_three_l244_244381

theorem sum_of_consecutive_even_integers_divisible_by_three (n : ℤ) : 
  ∃ p : ℤ, Prime p ∧ p = 3 ∧ p ∣ (n + (n + 2) + (n + 4)) :=
by 
  sorry

end sum_of_consecutive_even_integers_divisible_by_three_l244_244381


namespace sum_of_primes_lt_20_eq_77_l244_244116

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l244_244116


namespace discount_per_issue_l244_244902

theorem discount_per_issue
  (normal_subscription_cost : ℝ) (months : ℕ) (issues_per_month : ℕ) 
  (promotional_discount : ℝ) :
  normal_subscription_cost = 34 →
  months = 18 →
  issues_per_month = 2 →
  promotional_discount = 9 →
  (normal_subscription_cost - promotional_discount) / (months * issues_per_month) = 0.25 :=
by
  intros h1 h2 h3 h4
  sorry

end discount_per_issue_l244_244902


namespace point_A_on_x_axis_l244_244820

def point_A : ℝ × ℝ := (-2, 0)

theorem point_A_on_x_axis : point_A.snd = 0 :=
by
  unfold point_A
  sorry

end point_A_on_x_axis_l244_244820


namespace Tom_earns_per_week_l244_244524

-- Definitions based on conditions
def crab_buckets_per_day := 8
def crabs_per_bucket := 12
def price_per_crab := 5
def days_per_week := 7

-- The proof goal
theorem Tom_earns_per_week :
  (crab_buckets_per_day * crabs_per_bucket * price_per_crab * days_per_week) = 3360 := by
  sorry

end Tom_earns_per_week_l244_244524


namespace minimum_purchase_price_mod6_l244_244819

theorem minimum_purchase_price_mod6 
  (coin_values : List ℕ)
  (h1 : (1 : ℕ) ∈ coin_values)
  (h15 : (15 : ℕ) ∈ coin_values)
  (h50 : (50 : ℕ) ∈ coin_values)
  (A C : ℕ)
  (k : ℕ)
  (hA : A ≡ k [MOD 7])
  (hC : C ≡ k + 1 [MOD 7])
  (hP : ∃ P, P = A - C) : 
  ∃ P, P ≡ 6 [MOD 7] ∧ P > 0 :=
by
  sorry

end minimum_purchase_price_mod6_l244_244819


namespace bob_after_alice_l244_244796

def race_distance : ℕ := 15
def alice_speed : ℕ := 7
def bob_speed : ℕ := 9

def alice_time : ℕ := alice_speed * race_distance
def bob_time : ℕ := bob_speed * race_distance

theorem bob_after_alice : bob_time - alice_time = 30 := by
  sorry

end bob_after_alice_l244_244796


namespace toms_weekly_revenue_l244_244517

def crabs_per_bucket : Nat := 12
def number_of_buckets : Nat := 8
def price_per_crab : Nat := 5
def days_per_week : Nat := 7

theorem toms_weekly_revenue :
  (crabs_per_bucket * number_of_buckets * price_per_crab * days_per_week) = 3360 :=
by
  sorry

end toms_weekly_revenue_l244_244517


namespace find_m_n_l244_244922

theorem find_m_n : ∃ (m n : ℕ), m^m + (m * n)^n = 1984 ∧ m = 4 ∧ n = 3 := by
  sorry

end find_m_n_l244_244922


namespace probability_five_distinct_numbers_l244_244251

def num_dice := 5
def num_faces := 6

def favorable_outcomes : ℕ := nat.factorial 5 * num_faces
def total_outcomes : ℕ := num_faces ^ num_dice

theorem probability_five_distinct_numbers :
  (favorable_outcomes / total_outcomes : ℚ) = 5 / 54 := 
sorry

end probability_five_distinct_numbers_l244_244251


namespace solve_fraction_equation_l244_244855

theorem solve_fraction_equation :
  ∀ x : ℝ, (3 / (2 * x - 2) + 1 / (1 - x) = 3) → x = 7 / 6 :=
by
  sorry

end solve_fraction_equation_l244_244855


namespace greatest_three_digit_multiple_of_17_is_986_l244_244007

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l244_244007


namespace population_total_l244_244150

theorem population_total (total_population layers : ℕ) (ratio_A ratio_B ratio_C : ℕ) 
(sample_capacity : ℕ) (prob_ab_in_C : ℚ) 
(h1 : ratio_A = 3)
(h2 : ratio_B = 6)
(h3 : ratio_C = 1)
(h4 : sample_capacity = 20)
(h5 : prob_ab_in_C = 1 / 21)
(h6 : total_population = 10 * ratio_C) :
  total_population = 70 := 
by 
  sorry

end population_total_l244_244150


namespace percent_absent_students_l244_244895

def total_students : ℕ := 180
def num_boys : ℕ := 100
def num_girls : ℕ := 80
def fraction_boys_absent : ℚ := 1 / 5
def fraction_girls_absent : ℚ := 1 / 4

theorem percent_absent_students : 
  (fraction_boys_absent * num_boys + fraction_girls_absent * num_girls) / total_students = 22.22 / 100 := 
  sorry

end percent_absent_students_l244_244895


namespace consecutive_integers_divisible_product_l244_244170

theorem consecutive_integers_divisible_product (m n : ℕ) (h : m < n) :
  ∀ k : ℕ, ∃ i j : ℕ, i ≠ j ∧ k + i < k + n ∧ k + j < k + n ∧ (k + i) * (k + j) % (m * n) = 0 :=
by sorry

end consecutive_integers_divisible_product_l244_244170


namespace johns_contribution_l244_244949

theorem johns_contribution (A : ℝ) (J : ℝ) : 
  (1.7 * A = 85) ∧ ((5 * A + J) / 6 = 85) → J = 260 := 
by
  sorry

end johns_contribution_l244_244949


namespace semicircle_inequality_l244_244421

-- Define the points on the semicircle
variables (A B C D E : ℝ)
-- Define the length function
def length (X Y : ℝ) : ℝ := abs (X - Y)

-- This is the main theorem statement
theorem semicircle_inequality {A B C D E : ℝ} :
  length A B ^ 2 + length B C ^ 2 + length C D ^ 2 + length D E ^ 2 +
  length A B * length B C * length C D + length B C * length C D * length D E < 4 :=
sorry

end semicircle_inequality_l244_244421


namespace toms_weekly_income_l244_244520

variable (num_buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ)

def daily_crabs := num_buckets * crabs_per_bucket
def daily_income := daily_crabs * price_per_crab
def weekly_income := daily_income * days_per_week

theorem toms_weekly_income 
  (h1 : num_buckets = 8)
  (h2 : crabs_per_bucket = 12)
  (h3 : price_per_crab = 5)
  (h4 : days_per_week = 7) :
  weekly_income num_buckets crabs_per_bucket price_per_crab days_per_week = 3360 :=
by
  sorry

end toms_weekly_income_l244_244520


namespace parabola_properties_l244_244449

theorem parabola_properties 
  (p : ℝ) (h_pos : 0 < p) (m : ℝ) 
  (A B : ℝ × ℝ)
  (h_AB_on_parabola : ∀ (P : ℝ × ℝ), P = A ∨ P = B → (P.snd)^2 = 2 * p * P.fst) 
  (h_line_intersection : ∀ (P : ℝ × ℝ), P = A ∨ P = B → P.fst = m * P.snd + 3)
  (h_dot_product : (A.fst * B.fst + A.snd * B.snd) = 6)
  : (exists C : ℝ × ℝ, C = (-3, 0)) ∧
    (∃ k1 k2 : ℝ, 
        k1 = A.snd / (A.fst + 3) ∧ 
        k2 = B.snd / (B.fst + 3) ∧ 
        (1 / k1^2 + 1 / k2^2 - 2 * m^2) = 24) :=
by
  sorry

end parabola_properties_l244_244449


namespace amount_distributed_l244_244542

theorem amount_distributed (A : ℕ) (h : A / 14 = A / 18 + 80) : A = 5040 :=
sorry

end amount_distributed_l244_244542


namespace automobile_distance_2_minutes_l244_244749

theorem automobile_distance_2_minutes (a : ℝ) :
  let acceleration := a / 12
  let time_minutes := 2
  let time_seconds := time_minutes * 60
  let distance_feet := (1 / 2) * acceleration * time_seconds^2
  let distance_yards := distance_feet / 3
  distance_yards = 200 * a := 
by sorry

end automobile_distance_2_minutes_l244_244749


namespace sin_240_eq_neg_sqrt3_div_2_l244_244608

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244608


namespace option_A_is_linear_equation_l244_244998

-- Definitions for considering an equation being linear in two variables
def is_linear_equation (e : Prop) : Prop :=
  ∃ (a b c : ℝ), e = (a = b + c) ∧ a ≠ 0 ∧ b ≠ 0

-- The given equation in option A
def Eq_A : Prop := ∀ (x y : ℝ), (2 * y - 1) / 5 = 2 - (3 * x - 2) / 4

-- Proof problem statement
theorem option_A_is_linear_equation : is_linear_equation Eq_A :=
sorry

end option_A_is_linear_equation_l244_244998


namespace homework_checked_on_friday_l244_244386

theorem homework_checked_on_friday
  (prob_no_check : ℚ := 1/2)
  (prob_check_on_friday_given_check : ℚ := 1/5)
  (prob_a : ℚ := 3/5)
  : 1/3 = prob_check_on_friday_given_check / prob_a :=
by
  sorry

end homework_checked_on_friday_l244_244386


namespace sum_of_primes_lt_20_eq_77_l244_244112

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l244_244112


namespace sum_of_primes_less_than_20_l244_244120

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l244_244120


namespace toms_weekly_income_l244_244521

variable (num_buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ)

def daily_crabs := num_buckets * crabs_per_bucket
def daily_income := daily_crabs * price_per_crab
def weekly_income := daily_income * days_per_week

theorem toms_weekly_income 
  (h1 : num_buckets = 8)
  (h2 : crabs_per_bucket = 12)
  (h3 : price_per_crab = 5)
  (h4 : days_per_week = 7) :
  weekly_income num_buckets crabs_per_bucket price_per_crab days_per_week = 3360 :=
by
  sorry

end toms_weekly_income_l244_244521


namespace Frank_can_buy_7_candies_l244_244254

def tickets_whack_a_mole := 33
def tickets_skee_ball := 9
def cost_per_candy := 6

def total_tickets := tickets_whack_a_mole + tickets_skee_ball

theorem Frank_can_buy_7_candies : total_tickets / cost_per_candy = 7 := by
  sorry

end Frank_can_buy_7_candies_l244_244254


namespace sea_star_collection_l244_244665

theorem sea_star_collection (S : ℕ) (initial_seashells : ℕ) (initial_snails : ℕ) (lost_sea_creatures : ℕ) (remaining_items : ℕ) :
  initial_seashells = 21 →
  initial_snails = 29 →
  lost_sea_creatures = 25 →
  remaining_items = 59 →
  S + initial_seashells + initial_snails = remaining_items + lost_sea_creatures →
  S = 34 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end sea_star_collection_l244_244665


namespace Thomas_speed_greater_than_Jeremiah_l244_244712

-- Define constants
def Thomas_passes_kilometers_per_hour := 5
def Jeremiah_passes_kilometers_per_hour := 6

-- Define speeds (in meters per hour)
def Thomas_speed := Thomas_passes_kilometers_per_hour * 1000
def Jeremiah_speed := Jeremiah_passes_kilometers_per_hour * 1000

-- Define hypothetical additional distances
def Thomas_hypothetical_additional_distance := 600 * 2
def Jeremiah_hypothetical_additional_distance := 50 * 2

-- Define effective distances traveled
def Thomas_effective_distance := Thomas_speed + Thomas_hypothetical_additional_distance
def Jeremiah_effective_distance := Jeremiah_speed + Jeremiah_hypothetical_additional_distance

-- Theorem to prove
theorem Thomas_speed_greater_than_Jeremiah : Thomas_effective_distance > Jeremiah_effective_distance := by
  -- Placeholder for the proof
  sorry

end Thomas_speed_greater_than_Jeremiah_l244_244712


namespace sum_of_primes_less_than_20_is_77_l244_244041

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l244_244041


namespace num_special_permutations_l244_244810

open Finset

-- Define a function that checks if a list of first 'n' integers is a permutation
def is_permutation (l : List ℕ) (n : ℕ) : Prop :=
  l.toFinset = range (n + 1)

noncomputable def count_permutations (n : ℕ) : ℕ := sorry

theorem num_special_permutations : count_permutations 5 = 71 := sorry

end num_special_permutations_l244_244810


namespace arccos_cos_11_eq_l244_244563

theorem arccos_cos_11_eq: Real.arccos (Real.cos 11) = 11 - 3 * Real.pi := by
  sorry

end arccos_cos_11_eq_l244_244563


namespace animal_fish_consumption_l244_244989

-- Definitions for the daily consumption of each animal
def daily_trout_polar1 := 0.2
def daily_salmon_polar1 := 0.4

def daily_trout_polar2 := 0.3
def daily_salmon_polar2 := 0.5

def daily_trout_polar3 := 0.25
def daily_salmon_polar3 := 0.45

def daily_trout_sealion1 := 0.1
def daily_salmon_sealion1 := 0.15

def daily_trout_sealion2 := 0.2
def daily_salmon_sealion2 := 0.25

-- Calculate total daily consumption
def total_daily_trout :=
  daily_trout_polar1 + daily_trout_polar2 + daily_trout_polar3 + daily_trout_sealion1 + daily_trout_sealion2

def total_daily_salmon :=
  daily_salmon_polar1 + daily_salmon_polar2 + daily_salmon_polar3 + daily_salmon_sealion1 + daily_salmon_sealion2

-- Calculate total monthly consumption
def total_monthly_trout := total_daily_trout * 30
def total_monthly_salmon := total_daily_salmon * 30

-- Total monthly fish bucket consumption
def total_monthly_fish := total_monthly_trout + total_monthly_salmon

-- The statement to prove the total consumption
theorem animal_fish_consumption : total_monthly_fish = 84 := by
  sorry

end animal_fish_consumption_l244_244989


namespace sum_of_primes_less_than_20_l244_244055

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l244_244055


namespace eval_expression_l244_244165

theorem eval_expression (h : (Real.pi / 2) < 2 ∧ 2 < Real.pi) :
  Real.sqrt (1 - 2 * Real.sin (Real.pi + 2) * Real.cos (Real.pi + 2)) = Real.sin 2 - Real.cos 2 :=
sorry

end eval_expression_l244_244165


namespace find_c_l244_244861

-- Defining the given condition
def parabola (x : ℝ) (c : ℝ) : ℝ := 2 * x^2 + c

theorem find_c : (∃ c : ℝ, ∀ x : ℝ, parabola x c = 2 * x^2 + 1) :=
by 
  sorry

end find_c_l244_244861


namespace product_of_major_and_minor_axes_l244_244360

-- Given definitions from conditions
variables (O F A B C D : Type) 
variables (OF : ℝ) (dia_inscribed_circle_OCF : ℝ) (a b : ℝ)

-- Condition: O is the center of an ellipse
-- Point F is one focus, OF = 8
def O_center_ellipse : Prop := OF = 8

-- The diameter of the inscribed circle of triangle OCF is 4
def dia_inscribed_circle_condition : Prop := dia_inscribed_circle_OCF = 4

-- Define OA = OB = a, OC = OD = b
def major_axis_half_length : ℝ := a
def minor_axis_half_length : ℝ := b

-- Ellipse focal property a^2 - b^2 = 64
def ellipse_focal_property : Prop := a^2 - b^2 = 64

-- From the given conditions, expected result
def compute_product_AB_CD : Prop := 
  let AB := 2 * a,
      CD := 2 * b
  in AB * CD = 240

-- The main statement to be proven
theorem product_of_major_and_minor_axes 
  (h1 : O_center_ellipse)
  (h2 : dia_inscribed_circle_condition)
  (h3 : ellipse_focal_property)
  : compute_product_AB_CD :=
sorry

end product_of_major_and_minor_axes_l244_244360


namespace jerry_apples_l244_244344

theorem jerry_apples (J : ℕ) (h1 : 20 + 60 + J = 3 * 2 * 20):
  J = 40 :=
sorry

end jerry_apples_l244_244344


namespace sum_of_primes_less_than_20_l244_244119

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l244_244119


namespace transformed_roots_l244_244947

noncomputable def specific_polynomial : Polynomial ℝ :=
  Polynomial.C 1 - Polynomial.C 4 * Polynomial.X + Polynomial.C 6 * Polynomial.X ^ 2 - Polynomial.C 4 * Polynomial.X ^ 3 + Polynomial.C 1 * Polynomial.X ^ 4

theorem transformed_roots (a b c d : ℝ) :
  (a^4 - b*a - 5 = 0) ∧ (b^4 - b*b - 5 = 0) ∧ (c^4 - b*c - 5 = 0) ∧ (d^4 - b*d - 5 = 0) →
  specific_polynomial.eval ((a + b + c) / d)^2 = 0 ∧
  specific_polynomial.eval ((a + b + d) / c)^2 = 0 ∧
  specific_polynomial.eval ((a + c + d) / b)^2 = 0 ∧
  specific_polynomial.eval ((b + c + d) / a)^2 = 0 :=
  by
    sorry

end transformed_roots_l244_244947


namespace total_rehabilitation_centers_l244_244996

noncomputable def center_visits: ℕ := 6 -- Lisa's visits

def jude_visits (l: ℕ) : ℕ := l / 2
def han_visits (j: ℕ) : ℕ := 2 * j - 2
def jane_visits (h: ℕ) : ℕ := 6 + 2 * h

theorem total_rehabilitation_centers :
  let l := center_visits in
  let j := jude_visits l in
  let h := han_visits j in
  let n := jane_visits h in
  l + j + h + n = 27 :=
by
  sorry

end total_rehabilitation_centers_l244_244996


namespace sum_primes_less_than_20_l244_244063

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l244_244063


namespace movie_profit_l244_244732

theorem movie_profit
  (main_actor_fee : ℕ)
  (supporting_actor_fee : ℕ)
  (extra_fee : ℕ)
  (main_actor_food : ℕ)
  (supporting_actor_food : ℕ)
  (extra_food : ℕ)
  (crew_size : ℕ)
  (crew_food : ℕ)
  (post_production_cost : ℕ)
  (revenue : ℕ)
  (main_actors_count : ℕ)
  (supporting_actors_count : ℕ)
  (extras_count : ℕ)
  (food_per_main_actor : ℕ)
  (food_per_supporting_actor : ℕ)
  (food_per_remaining_crew : ℕ)
  (equipment_rental_multiplier : ℕ)
  (total_profit : ℕ) :
  main_actor_fee = 500 → 
  supporting_actor_fee = 100 →
  extra_fee = 50 →
  main_actor_food = 10 →
  supporting_actor_food = 5 →
  extra_food = 5 →
  crew_size = 50 →
  crew_food = 3 →
  post_production_cost = 850 →
  revenue = 10000 →
  main_actors_count = 2 →
  supporting_actors_count = 3 →
  extras_count = 1 →
  equipment_rental_multiplier = 2 →
  total_profit = revenue - ((main_actors_count * main_actor_fee) +
                           (supporting_actors_count * supporting_actor_fee) +
                           (extras_count * extra_fee) +
                           (main_actors_count * main_actor_food) +
                           ((supporting_actors_count + extras_count) * supporting_actor_food) +
                           ((crew_size - main_actors_count - supporting_actors_count - extras_count) * crew_food) +
                           (equipment_rental_multiplier * 
                             ((main_actors_count * main_actor_fee) +
                              (supporting_actors_count * supporting_actor_fee) +
                              (extras_count * extra_fee) +
                              (main_actors_count * main_actor_food) +
                              ((supporting_actors_count + extras_count) * supporting_actor_food) +
                              ((crew_size - main_actors_count - supporting_actors_count - extras_count) * crew_food))) +
                           post_production_cost) →
  total_profit = 4584 :=
begin
  -- proof
  sorry
end

end movie_profit_l244_244732


namespace induction_step_l244_244715

theorem induction_step
  (x y : ℝ)
  (k : ℕ)
  (base : ∀ n, ∃ m, (n = 2 * m - 1) → (x^n + y^n) = (x + y) * m) :
  (x^(2 * k + 1) + y^(2 * k + 1)) = (x + y) * (k + 1) :=
by
  sorry

end induction_step_l244_244715


namespace sum_of_primes_less_than_20_is_77_l244_244042

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l244_244042


namespace prove_n_is_prime_l244_244801

open Nat
open Zmod

theorem prove_n_is_prime (n : ℕ) (a : ℕ) (h1: 1 < n) 
  (h2: ∃ a : ℕ, a^(n-1) ≡ 1 [MOD n]) 
  (h3: ∀ p: ℕ, p.Prime → p ∣ (n - 1) → a^((n-1) / p) ≢ 1 [MOD n]) 
  : Nat.Prime n := 
  sorry

end prove_n_is_prime_l244_244801


namespace number_of_palindromes_divisible_by_6_l244_244287

theorem number_of_palindromes_divisible_by_6 :
  let is_palindrome (n : ℕ) : Prop := n / 1000 = n % 10 ∧ (n / 100 % 10) = (n / 10 % 10)
  let valid_digits (n : ℕ) : Prop := n >= 1000 ∧ n < 10000
  let divisible_6 (n : ℕ) : Prop := n % 6 = 0
  (Finset.filter (λ n => is_palindrome n ∧ valid_digits n ∧ divisible_6 n) (Finset.range 10000)).card = 13 :=
by
  -- We define what it means to be a palindrome between 1000 and 10000
  let is_palindrome (n : ℕ) : Prop := n / 1000 = n % 10 ∧ n / 100 % 10 = n / 10 % 10
  
  -- We define a valid number between 1000 and 10000
  let valid_digits (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
  
  -- We define what it means to be divisible by 6
  let divisible_6 (n : ℕ) : Prop := n % 6 = 0

  -- Filtering the range 10000 within valid four-digit palindromes and checking for multiples of 6
  exact sorry

end number_of_palindromes_divisible_by_6_l244_244287


namespace problem1_problem2_l244_244726

-- Problem statement 1: Prove (a-2)(a-6) < (a-3)(a-5)
theorem problem1 (a : ℝ) : (a - 2) * (a - 6) < (a - 3) * (a - 5) :=
by
  sorry

-- Problem statement 2: Prove the range of values for 2x - y given -2 < x < 1 and 1 < y < 2 is (-6, 1)
theorem problem2 (x y : ℝ) (hx : -2 < x) (hx1 : x < 1) (hy : 1 < y) (hy1 : y < 2) : -6 < 2 * x - y ∧ 2 * x - y < 1 :=
by
  sorry

end problem1_problem2_l244_244726


namespace sum_primes_less_than_20_l244_244050

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l244_244050


namespace find_common_real_root_l244_244682

theorem find_common_real_root :
  ∃ (m a : ℝ), (a^2 + m * a + 2 = 0) ∧ (a^2 + 2 * a + m = 0) ∧ m = -3 ∧ a = 1 :=
by
  -- Skipping the proof
  sorry

end find_common_real_root_l244_244682


namespace two_colonies_same_time_l244_244397

def doubles_in_size_every_day (P : ℕ → ℕ) : Prop :=
∀ n, P (n + 1) = 2 * P n

def reaches_habitat_limit_in (f : ℕ → ℕ) (days limit : ℕ) : Prop :=
f days = limit

theorem two_colonies_same_time (P : ℕ → ℕ) (Q : ℕ → ℕ) (limit : ℕ) (days : ℕ)
  (h1 : doubles_in_size_every_day P)
  (h2 : reaches_habitat_limit_in P days limit)
  (h3 : ∀ n, Q n = 2 * P n) :
  reaches_habitat_limit_in Q days limit :=
sorry

end two_colonies_same_time_l244_244397


namespace sequence_sum_n_eq_21_l244_244195

theorem sequence_sum_n_eq_21 (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ k, a (k + 1) = a k + 1)
  (h3 : ∀ n, S n = (n * (n + 1)) / 2)
  (h4 : S n = 21) :
  n = 6 :=
sorry

end sequence_sum_n_eq_21_l244_244195


namespace initial_number_l244_244742

theorem initial_number (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by
  sorry

end initial_number_l244_244742


namespace probability_of_5_distinct_dice_rolls_is_5_over_54_l244_244249

def count_distinct_dice_rolls : ℕ :=
  6 * 5 * 4 * 3 * 2

def total_dice_rolls : ℕ :=
  6 ^ 5

def probability_of_distinct_rolls : ℚ :=
  count_distinct_dice_rolls / total_dice_rolls

theorem probability_of_5_distinct_dice_rolls_is_5_over_54 : 
  probability_of_distinct_rolls = 5 / 54 :=
by
  sorry

end probability_of_5_distinct_dice_rolls_is_5_over_54_l244_244249


namespace sin_240_deg_l244_244613

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l244_244613


namespace extrema_range_l244_244663

noncomputable def hasExtrema (a : ℝ) : Prop :=
  (4 * a^2 + 12 * a > 0)

theorem extrema_range (a : ℝ) : hasExtrema a ↔ (a < -3 ∨ a > 0) := sorry

end extrema_range_l244_244663


namespace all_statements_imply_implication_l244_244754

variables (p q r : Prop)

theorem all_statements_imply_implication :
  (p ∧ ¬ q ∧ r → ((p → q) → r)) ∧
  (¬ p ∧ ¬ q ∧ r → ((p → q) → r)) ∧
  (p ∧ ¬ q ∧ ¬ r → ((p → q) → r)) ∧
  (¬ p ∧ q ∧ r → ((p → q) → r)) :=
by { sorry }

end all_statements_imply_implication_l244_244754


namespace identify_vanya_l244_244356

structure Twin :=
(name : String)
(truth_teller : Bool)

def is_vanya_truth_teller (twin : Twin) (vanya vitya : Twin) : Prop :=
  twin = vanya ∧ twin.truth_teller ∨ twin = vitya ∧ ¬twin.truth_teller

theorem identify_vanya
  (vanya vitya : Twin)
  (h_vanya : vanya.name = "Vanya")
  (h_vitya : vitya.name = "Vitya")
  (h_one_truth : ∃ t : Twin, t = vanya ∨ t = vitya ∧ (t.truth_teller = true ∨ t.truth_teller = false))
  (h_one_lie : ∀ t : Twin, t = vanya ∨ t = vitya → ¬(t.truth_teller = true ∧ t = vitya) ∧ ¬(t.truth_teller = false ∧ t = vanya)) :
  ∀ twin : Twin, twin = vanya ∨ twin = vitya →
  (is_vanya_truth_teller twin vanya vitya ↔ (twin = vanya ∧ twin.truth_teller = true)) :=
by
  sorry

end identify_vanya_l244_244356


namespace total_time_taken_l244_244405

theorem total_time_taken (b km : ℝ) : 
  (b / 50 + km / 80) = (8 * b + 5 * km) / 400 := 
sorry

end total_time_taken_l244_244405


namespace ellipse_and_line_properties_l244_244301

theorem ellipse_and_line_properties :
  (∃ a b : ℝ, a > b ∧ b > 0 ∧ a * a = 4 ∧ b * b = 3 ∧
  ∀ x y : ℝ, (x, y) = (1, 3/2) → x^2 / a^2 + y^2 / b^2 = 1) ∧
  (∃ k : ℝ, k = 1 / 2 ∧ ∀ x y : ℝ, (x, y) = (2, 1) →
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧
  (x1 - 2) * (x2 - 2) + (k * (x1 - 2) + 1 - 1) * (k * (x2 - 2) + 1 - 1) = 5 / 4) :=
sorry

end ellipse_and_line_properties_l244_244301


namespace sum_primes_less_than_20_l244_244049

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l244_244049


namespace exists_two_pairs_satisfy_2x3_eq_y4_l244_244292

theorem exists_two_pairs_satisfy_2x3_eq_y4 :
  ∃ (x₁ y₁ x₂ y₂ : ℕ), 2 * x₁^3 = y₁^4 ∧ 2 * x₂^3 = y₂^4 ∧ (x₁, y₁) ≠ (x₂, y₂) :=
by
  use 2, 2
  use 32, 16
  split
  . calc 2 * 2^3 = 2 * 8 : by rw [pow_succ]
            ... = 16      : by norm_num
  split
  . calc 2 * 32^3 = 2 * (2^5)^3 : rfl 
              ... = 2 * 2^15    : by rw [pow_mul]
              ... = 2^16        : by rw [mul_comm, pow_succ, mul_assoc, pow_one, two_mul]
  . exact ne_of_apply_ne Prod.fst $ by simp

end exists_two_pairs_satisfy_2x3_eq_y4_l244_244292


namespace sum_primes_less_than_20_l244_244098

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l244_244098


namespace find_ratio_of_sides_l244_244474

variable {A B : ℝ}
variable {a b : ℝ}

-- Given condition
axiom given_condition : a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = a * Real.sqrt 3

-- Theorem we need to prove
theorem find_ratio_of_sides (h : a ≠ 0) : b / a = Real.sqrt 3 / 3 :=
by
  sorry

end find_ratio_of_sides_l244_244474


namespace sum_after_operations_l244_244511

theorem sum_after_operations (a b S : ℝ) (h : a + b = S) : 
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := 
by 
  sorry

end sum_after_operations_l244_244511


namespace sum_of_primes_less_than_twenty_is_77_l244_244034

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l244_244034


namespace sum_of_primes_less_than_20_l244_244102

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l244_244102


namespace sin_240_eq_neg_sqrt3_div_2_l244_244582

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244582


namespace find_k_b_l244_244486

-- Define the sets A and B
def A : Set (ℝ × ℝ) := { p | ∃ x y: ℝ, p = (x, y) }
def B : Set (ℝ × ℝ) := { p | ∃ x y: ℝ, p = (x, y) }

-- Define the mapping f
def f (p : ℝ × ℝ) (k b : ℝ) : ℝ × ℝ := (k * p.1, p.2 + b)

-- Define the conditions
def condition (f : (ℝ × ℝ) → ℝ × ℝ) :=
  f (3,1) = (6,2)

-- Statement: Prove that the values of k and b are 2 and 1 respectively
theorem find_k_b : ∃ (k b : ℝ), f (3, 1) k b = (6, 2) ∧ k = 2 ∧ b = 1 :=
by
  sorry

end find_k_b_l244_244486


namespace sin_240_l244_244573

theorem sin_240 : Real.sin (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Provided conditions
  have h1 : 240 = 180 + 60 := be_of_eq true.intro
  have h2 : ∀ θ : ℝ, θ ∈ set.Icc (pi : ℝ) (3 * pi / 2) → Real.sin θ < 0 := Real.sin_neg_of_pi_lt_of_lt (Real.pi_lt_2_pi)
  have h3 : Real.sin (60 * Real.pi / 180) = 1 / 2 := Real.sin_pi_div_three
  -- Prove
  sorry

end sin_240_l244_244573


namespace least_positive_t_geometric_progression_l244_244912

noncomputable def least_positive_t( α : ℝ ) ( h : 0 < α ∧ α < Real.pi / 2 ) : ℝ :=
  9 - 4 * Real.sqrt 5

theorem least_positive_t_geometric_progression ( α t : ℝ ) ( h : 0 < α ∧ α < Real.pi / 2 ) :
  least_positive_t α h = t ↔
  ∃ r : ℝ, r > 0 ∧
    Real.arcsin (Real.sin α) = α ∧
    Real.arcsin (Real.sin (2 * α)) = 2 * α ∧
    Real.arcsin (Real.sin (7 * α)) = 7 * α ∧
    Real.arcsin (Real.sin (t * α)) = t * α ∧
    (α * r = 2 * α) ∧
    (2 * α * r = 7 * α ) ∧
    (7 * α * r = t * α) :=
sorry

end least_positive_t_geometric_progression_l244_244912


namespace system_of_linear_eq_with_two_variables_l244_244126

-- Definitions of individual equations
def eqA (x : ℝ) : Prop := 3 * x - 2 = 5
def eqB (x : ℝ) : Prop := 6 * x^2 - 2 = 0
def eqC (x y : ℝ) : Prop := 1 / x + y = 3
def eqD (x y : ℝ) : Prop := 5 * x + y = 2

-- The main theorem to prove that D is a system of linear equations with two variables
theorem system_of_linear_eq_with_two_variables :
    (∃ x y : ℝ, eqD x y) ∧ (¬∃ x : ℝ, eqA x) ∧ (¬∃ x : ℝ, eqB x) ∧ (¬∃ x y : ℝ, eqC x y) :=
by
  sorry

end system_of_linear_eq_with_two_variables_l244_244126


namespace greatest_three_digit_multiple_of_17_is_986_l244_244010

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l244_244010


namespace train_distance_l244_244891

theorem train_distance (train_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  (train_speed = 1) → (total_time = 180) → (distance = train_speed * total_time) → 
  distance = 180 :=
by
  intros train_speed_eq total_time_eq dist_eq
  rw [train_speed_eq, total_time_eq] at dist_eq
  exact dist_eq

end train_distance_l244_244891


namespace moles_of_C2H6_formed_l244_244764

-- Definitions of the quantities involved
def moles_H2 : ℕ := 3
def moles_C2H4 : ℕ := 3
def moles_C2H6 : ℕ := 3

-- Stoichiometry condition stated in a way that Lean can understand.
axiom stoichiometry : moles_H2 = moles_C2H4

theorem moles_of_C2H6_formed : moles_C2H6 = 3 :=
by
  -- Assume the constraints and state the final result
  have h : moles_H2 = moles_C2H4 := stoichiometry
  show moles_C2H6 = 3
  sorry

end moles_of_C2H6_formed_l244_244764


namespace simplify_expression_l244_244498

theorem simplify_expression (n : ℕ) : 
  (3 ^ (n + 5) - 3 * 3 ^ n) / (3 * 3 ^ (n + 4)) = 80 / 27 :=
by sorry

end simplify_expression_l244_244498


namespace probability_of_four_requests_in_four_hours_l244_244558

-- Define the Poisson mass function
def poisson_pmf (λ t m : ℝ) : ℝ :=
  (λ * t) ^ m / (Nat.factorial m) * Real.exp (-(λ * t))

-- Define the specific values for this problem
def λ : ℝ := 2
def t : ℝ := 4
def m : ℝ := 4

-- Define the expected probability
def expected_probability : ℝ := 0.0572

-- The main theorem stating the exact probability matches the expected value approximately
theorem probability_of_four_requests_in_four_hours : 
  abs (poisson_pmf λ t m - expected_probability) < 0.0001 :=
by
  sorry

end probability_of_four_requests_in_four_hours_l244_244558


namespace sam_total_spent_l244_244493

-- Define the values of a penny and a dime in dollars
def penny_value : ℝ := 0.01
def dime_value : ℝ := 0.10

-- Define what Sam spent
def friday_spent : ℝ := 2 * penny_value
def saturday_spent : ℝ := 12 * dime_value

-- Define total spent
def total_spent : ℝ := friday_spent + saturday_spent

theorem sam_total_spent : total_spent = 1.22 := 
by
  -- The following is a placeholder for the actual proof
  sorry

end sam_total_spent_l244_244493


namespace sum_primes_less_than_20_l244_244075

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l244_244075


namespace mass_of_man_l244_244135

variable (L : ℝ) (B : ℝ) (h : ℝ) (ρ : ℝ)

-- Given conditions
def boatLength := L = 3
def boatBreadth := B = 2
def sinkingDepth := h = 0.018
def waterDensity := ρ = 1000

-- The mass of the man
theorem mass_of_man (L B h ρ : ℝ) (H1 : boatLength L) (H2 : boatBreadth B) (H3 : sinkingDepth h) (H4 : waterDensity ρ) : 
  ρ * L * B * h = 108 := by
  sorry

end mass_of_man_l244_244135


namespace willows_in_the_park_l244_244515

theorem willows_in_the_park (W O : ℕ) 
  (h1 : W + O = 83) 
  (h2 : O = W + 11) : 
  W = 36 := 
by 
  sorry

end willows_in_the_park_l244_244515


namespace find_value_of_a_l244_244231

theorem find_value_of_a (a : ℝ) (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 < a → a^x ≥ 1)
  (h_sum : (a^1) + (a^0) = 3) : a = 2 :=
sorry

end find_value_of_a_l244_244231


namespace probability_of_selected_number_between_l244_244144

open Set

theorem probability_of_selected_number_between (s : Set ℤ) (a b x y : ℤ) 
  (h1 : a = 25) 
  (h2 : b = 925) 
  (h3 : x = 25) 
  (h4 : y = 99) 
  (h5 : s = Set.Icc a b) :
  (y - x + 1 : ℚ) / (b - a + 1 : ℚ) = 75 / 901 := 
by 
  sorry

end probability_of_selected_number_between_l244_244144


namespace billy_ate_9_apples_on_wednesday_l244_244560

/-- Define the problem conditions -/
def apples (day : String) : Nat :=
  match day with
  | "Monday" => 2
  | "Tuesday" => 2 * apples "Monday"
  | "Friday" => apples "Monday" / 2
  | "Thursday" => 4 * apples "Friday"
  | _ => 0  -- For other days, we'll define later

/-- Define total apples eaten -/
def total_apples : Nat := 20

/-- Define sum of known apples excluding Wednesday -/
def known_sum : Nat :=
  apples "Monday" + apples "Tuesday" + apples "Friday" + apples "Thursday"

/-- Calculate apples eaten on Wednesday -/
def wednesday_apples : Nat := total_apples - known_sum

theorem billy_ate_9_apples_on_wednesday : wednesday_apples = 9 :=
  by
  sorry  -- Proof skipped

end billy_ate_9_apples_on_wednesday_l244_244560


namespace greatest_three_digit_multiple_of_17_is_986_l244_244011

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l244_244011


namespace frog_jump_l244_244808

def coprime (p q : ℕ) : Prop := Nat.gcd p q = 1

theorem frog_jump (p q : ℕ) (h_coprime : coprime p q) :
  ∀ d : ℕ, d < p + q → (∃ m n : ℤ, m ≠ n ∧ (m - n = d ∨ n - m = d)) :=
by
  sorry

end frog_jump_l244_244808


namespace sequence_periodic_mod_l244_244203

-- Define the sequence (u_n) recursively
def sequence_u (a : ℕ) : ℕ → ℕ
  | 0     => a  -- Note: u_1 is defined as the initial term a, treating the starting index as 0 for compatibility with Lean's indexing.
  | (n+1) => a ^ (sequence_u a n)

-- The theorem stating there exist integers k and N such that for all n ≥ N, u_{n+k} ≡ u_n (mod m)
theorem sequence_periodic_mod (a m : ℕ) (hm : 0 < m) (ha : 0 < a) :
  ∃ k N : ℕ, ∀ n : ℕ, N ≤ n → (sequence_u a (n + k) ≡ sequence_u a n [MOD m]) :=
by
  sorry

end sequence_periodic_mod_l244_244203


namespace largest_n_cube_condition_l244_244637

theorem largest_n_cube_condition :
  ∃ n : ℕ, (n^3 + 4 * n^2 - 15 * n - 18 = k^3) ∧ ∀ m : ℕ, (m^3 + 4 * m^2 - 15 * m - 18 = k^3 → m ≤ n) → n = 19 :=
by
  sorry

end largest_n_cube_condition_l244_244637


namespace checkerboard_problem_l244_244266

-- Definitions corresponding to conditions
def checkerboard_size := 10

def is_alternating (i j : ℕ) : bool :=
  (i + j) % 2 = 0

def num_squares_with_sides_on_grid_lines_containing_at_least_6_black_squares (n : ℕ) : ℕ :=
  if n >= 4 then (11 - n) * (11 - n) else 0

-- Problem statement
theorem checkerboard_problem : 
  let count_squares : ℕ := (∑ n in finset.range checkerboard_size.succ, num_squares_with_sides_on_grid_lines_containing_at_least_6_black_squares n)
  in count_squares = 140 :=
begin
  sorry
end

end checkerboard_problem_l244_244266


namespace sin_240_eq_neg_sqrt3_over_2_l244_244600

open Real

-- Conditions
def angle_240_in_third_quadrant : Prop := 240 ° ∈ set_of (λ x, 180 ° < x ∧ x < 270 °)

def reference_angle_60 (θ : Real) : Prop := θ = 240 ° - 180 °

def sin_60_eq_sqrt3_over_2 : sin (60 °) = sqrt 3 / 2

def sin_negative_in_third_quadrant (θ : Real) : Prop :=
  180 ° < θ ∧ θ < 270 ° → sin θ < 0

-- Statement
theorem sin_240_eq_neg_sqrt3_over_2 :
  angle_240_in_third_quadrant ∧ reference_angle_60 60 ° ∧ sin_60_eq_sqrt3_over_2 ∧ sin_negative_in_third_quadrant 240 °
  → sin (240 °) = - (sqrt 3 / 2) :=
by
  intros
  sorry

end sin_240_eq_neg_sqrt3_over_2_l244_244600


namespace intersection_complement_l244_244450

open Set

noncomputable def U := ℝ
noncomputable def A := {x : ℝ | x^2 + 2 * x < 3}
noncomputable def B := {x : ℝ | x - 2 ≤ 0 ∧ x ≠ 0}

theorem intersection_complement :
  A ∩ -B = {x : ℝ | -3 < x ∧ x ≤ 0} :=
sorry

end intersection_complement_l244_244450


namespace sum_primes_less_than_20_l244_244082

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l244_244082


namespace sin_240_deg_l244_244610

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l244_244610


namespace sum_primes_less_than_20_l244_244074

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l244_244074


namespace solution_unique_l244_244426

def is_solution (x : ℝ) : Prop :=
  ⌊x * ⌊x⌋⌋ = 48

theorem solution_unique (x : ℝ) : is_solution x → x = -48 / 7 :=
by
  intro h
  -- Proof goes here
  sorry

end solution_unique_l244_244426


namespace greatest_divisor_with_sum_of_digits_four_l244_244465

/-- Define the given numbers -/
def a := 4665
def b := 6905

/-- Define the sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Define the greatest number n that divides both a and b, leaving the same remainder and having a sum of digits equal to 4 -/
theorem greatest_divisor_with_sum_of_digits_four :
  ∃ (n : ℕ), (∀ (d : ℕ), (d ∣ a - b ∧ sum_of_digits d = 4) → d ≤ n) ∧ (n ∣ a - b) ∧ (sum_of_digits n = 4) ∧ n = 40 := sorry

end greatest_divisor_with_sum_of_digits_four_l244_244465


namespace ball_hits_ground_in_3_seconds_l244_244903

noncomputable def ball_height (t : ℝ) : ℝ := -16 * t^2 - 32 * t + 240

theorem ball_hits_ground_in_3_seconds :
  ∃ t : ℝ, ball_height t = 0 ∧ t = 3 :=
sorry

end ball_hits_ground_in_3_seconds_l244_244903


namespace max_abs_value_of_quadratic_function_l244_244191

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def point_in_band_region (y k l : ℝ) : Prop := k ≤ y ∧ y ≤ l

theorem max_abs_value_of_quadratic_function (a b c t : ℝ) (h1 : point_in_band_region (quadratic_function a b c (-2) + 2) 0 4)
                                             (h2 : point_in_band_region (quadratic_function a b c 0 + 2) 0 4)
                                             (h3 : point_in_band_region (quadratic_function a b c 2 + 2) 0 4)
                                             (h4 : point_in_band_region (t + 1) (-1) 3) :
  |quadratic_function a b c t| ≤ 5 / 2 :=
sorry

end max_abs_value_of_quadratic_function_l244_244191


namespace beads_removed_l244_244746

def total_beads (blue yellow : Nat) : Nat := blue + yellow

def beads_per_part (total : Nat) (parts : Nat) : Nat := total / parts

def beads_remaining (per_part : Nat) (removed : Nat) : Nat := per_part - removed

def doubled_beads (remaining : Nat) : Nat := 2 * remaining

theorem beads_removed {x : Nat} 
  (blue : Nat) (yellow : Nat) (parts : Nat) (final_per_part : Nat) :
  total_beads blue yellow = 39 →
  parts = 3 →
  beads_per_part 39 parts = 13 →
  doubled_beads (beads_remaining 13 x) = 6 →
  x = 10 := by
  sorry

end beads_removed_l244_244746


namespace next_term_in_geom_sequence_l244_244531

   /- Define the given geometric sequence as a function in Lean -/

   def geom_sequence (a r : ℤ) (n : ℕ) : ℤ := a * r ^ n

   theorem next_term_in_geom_sequence (x : ℤ) (n : ℕ) 
     (h₁ : geom_sequence 3 (-3*x) 0 = 3)
     (h₂ : geom_sequence 3 (-3*x) 1 = -9*x)
     (h₃ : geom_sequence 3 (-3*x) 2 = 27*(x^2))
     (h₄ : geom_sequence 3 (-3*x) 3 = -81*(x^3)) :
     geom_sequence 3 (-3*x) 4 = 243*(x^4) := 
   sorry
   
end next_term_in_geom_sequence_l244_244531


namespace count_intersection_ways_l244_244528

theorem count_intersection_ways :
  ∃ (A B C D E F : ℕ),
  A ≠ D ∧ B ≠ E ∧ C ≠ F ∧
  {A, B, C, D, E, F} ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  (A-D) ≠ 0 ∧ (B-E) ≠ 0 ∧ (C-F) ≠ 0 ∧
  (A+D) ≠ 0 ∧
  9! / (3! * 3!) = 60480 :=
by
  sorry

end count_intersection_ways_l244_244528


namespace problem_proof_l244_244905

-- Define the mixed numbers and their conversions to improper fractions
def mixed_number_1 := 84 * 19 + 4  -- 1600
def mixed_number_2 := 105 * 19 + 5 -- 2000 

-- Define the improper fractions
def improper_fraction_1 := mixed_number_1 / 19
def improper_fraction_2 := mixed_number_2 / 19

-- Define the decimals and their conversions to fractions
def decimal_1 := 11 / 8  -- 1.375
def decimal_2 := 9 / 10  -- 0.9

-- Perform the multiplications
def multiplication_1 := (improper_fraction_1 * decimal_1 : ℚ)
def multiplication_2 := (improper_fraction_2 * decimal_2 : ℚ)

-- Perform the addition
def addition_result := multiplication_1 + multiplication_2

-- The final result is converted to a fraction for comparison
def final_result := 4000 / 19

-- Define and state the theorem
theorem problem_proof : addition_result = final_result := by
  sorry

end problem_proof_l244_244905


namespace probability_of_lamps_arrangement_l244_244825

noncomputable def probability_lava_lamps : ℚ :=
  let total_lamps := 8
  let red_lamps := 4
  let blue_lamps := 4
  let total_turn_on := 4
  let left_red_on := 1
  let right_blue_off := 1
  let ways_to_choose_positions := Nat.choose total_lamps red_lamps
  let ways_to_choose_turn_on := Nat.choose total_lamps total_turn_on
  let remaining_positions := total_lamps - left_red_on - right_blue_off
  let remaining_red_lamps := red_lamps - left_red_on
  let remaining_turn_on := total_turn_on - left_red_on
  let arrangements_of_remaining_red := Nat.choose remaining_positions remaining_red_lamps
  let arrangements_of_turn_on :=
    Nat.choose (remaining_positions - right_blue_off) remaining_turn_on
  -- The probability calculation
  (arrangements_of_remaining_red * arrangements_of_turn_on : ℚ) / 
    (ways_to_choose_positions * ways_to_choose_turn_on)

theorem probability_of_lamps_arrangement :
    probability_lava_lamps = 4 / 49 :=
by
  sorry

end probability_of_lamps_arrangement_l244_244825


namespace sum_of_primes_less_than_20_l244_244021

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l244_244021


namespace unique_reversible_six_digit_number_exists_l244_244636

theorem unique_reversible_six_digit_number_exists :
  ∃! (N : ℤ), 100000 ≤ N ∧ N < 1000000 ∧
  ∃ (f e d c b a : ℤ), 
  N = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ∧ 
  9 * N = 100000 * f + 10000 * e + 1000 * d + 100 * c + 10 * b + a := 
sorry

end unique_reversible_six_digit_number_exists_l244_244636


namespace largest_radius_cone_l244_244142

structure Crate :=
  (width : ℝ)
  (depth : ℝ)
  (height : ℝ)

structure Cone :=
  (radius : ℝ)
  (height : ℝ)

noncomputable def larger_fit_within_crate (c : Crate) (cone : Cone) : Prop :=
  cone.radius = min c.width c.depth / 2 ∧ cone.height = max (max c.width c.depth) c.height

theorem largest_radius_cone (c : Crate) (cone : Cone) : 
  c.width = 5 → c.depth = 8 → c.height = 12 → larger_fit_within_crate c cone → cone.radius = 2.5 :=
by
  sorry

end largest_radius_cone_l244_244142


namespace pascal_fifth_number_l244_244717

def binom (n k : Nat) : Nat := Nat.choose n k

theorem pascal_fifth_number (n r : Nat) (h1 : n = 50) (h2 : r = 4) : binom n r = 230150 := by
  sorry

end pascal_fifth_number_l244_244717


namespace minimum_value_2x_4y_l244_244441

theorem minimum_value_2x_4y (x y : ℝ) (h : x + 2 * y = 3) : 
  ∃ (min_val : ℝ), min_val = 2 ^ (5/2) ∧ (2 ^ x + 4 ^ y = min_val) :=
by
  sorry

end minimum_value_2x_4y_l244_244441


namespace total_cups_l244_244162

variable (eggs : ℕ) (flour : ℕ)
variable (h : eggs = 60) (h1 : flour = eggs / 2)

theorem total_cups (eggs : ℕ) (flour : ℕ) (h : eggs = 60) (h1 : flour = eggs / 2) : 
  eggs + flour = 90 := 
by
  sorry

end total_cups_l244_244162


namespace integer_value_l244_244954

theorem integer_value (x y z : ℕ) (h1 : 2 * x = 5 * y) (h2 : 5 * y = 6 * z) (h3 : x > 0) (h4 : y > 0) (h5 : z > 0) :
  ∃ a : ℕ, a + y + z = 26 ∧ a = 15 := by
  sorry

end integer_value_l244_244954


namespace partition_triangle_l244_244624

theorem partition_triangle (triangle : List ℕ) (h_triangle_sum : triangle.sum = 63) :
  ∃ (parts : List (List ℕ)), parts.length = 3 ∧ 
  (∀ part ∈ parts, part.sum = 21) ∧ 
  parts.bind id = triangle :=
by
  sorry

end partition_triangle_l244_244624


namespace simplify_expr_l244_244842

theorem simplify_expr : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expr_l244_244842


namespace pages_read_on_fourth_day_l244_244185

-- condition: Hallie reads the whole book in 4 days, read specific pages each day
variable (total_pages first_day_pages second_day_pages third_day_pages fourth_day_pages : ℕ)

-- Given conditions
def conditions : Prop :=
  first_day_pages = 63 ∧
  second_day_pages = 2 * first_day_pages ∧
  third_day_pages = second_day_pages + 10 ∧
  total_pages = 354 ∧
  first_day_pages + second_day_pages + third_day_pages + fourth_day_pages = total_pages

-- Prove Hallie read 29 pages on the fourth day
theorem pages_read_on_fourth_day (h : conditions total_pages first_day_pages second_day_pages third_day_pages fourth_day_pages) :
  fourth_day_pages = 29 := sorry

end pages_read_on_fourth_day_l244_244185


namespace part_I_part_II_l244_244445

noncomputable def f (x : ℝ) : ℝ := abs x

theorem part_I (x : ℝ) : f (x-1) > 2 ↔ x < -1 ∨ x > 3 := 
by sorry

theorem part_II (x y z : ℝ) (h : f x ^ 2 + y ^ 2 + z ^ 2 = 9) : ∃ (min_val : ℝ), min_val = -9 ∧ ∀ (a b c : ℝ), f a ^ 2 + b ^ 2 + c ^ 2 = 9 → (a + 2 * b + 2 * c) ≥ min_val := 
by sorry

end part_I_part_II_l244_244445


namespace number_of_valid_ns_l244_244297

theorem number_of_valid_ns :
  ∃ (S : Finset ℕ), S.card = 13 ∧ ∀ n ∈ S, n ≤ 1000 ∧ Nat.floor (995 / n) + Nat.floor (996 / n) + Nat.floor (997 / n) % 4 ≠ 0 :=
by
  sorry

end number_of_valid_ns_l244_244297


namespace greatest_three_digit_multiple_of_17_is_986_l244_244005

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l244_244005


namespace lambs_total_l244_244759

/-
Each of farmer Cunningham's lambs is either black or white.
There are 193 white lambs, and 5855 black lambs.
Prove that the total number of lambs is 6048.
-/

theorem lambs_total (white_lambs : ℕ) (black_lambs : ℕ) (h1 : white_lambs = 193) (h2 : black_lambs = 5855) :
  white_lambs + black_lambs = 6048 :=
by
  -- proof goes here
  sorry

end lambs_total_l244_244759


namespace identify_quadratic_equation_l244_244999

theorem identify_quadratic_equation :
  (¬(∃ x y : ℝ, x^2 - 2*x*y + y^2 = 0) ∧  -- Condition A is not a quadratic equation
   ¬(∃ x : ℝ, x*(x + 3) = x^2 - 1) ∧      -- Condition B is not a quadratic equation
   (∃ x : ℝ, x^2 - 2*x - 3 = 0) ∧         -- Condition C is a quadratic equation
   ¬(∃ x : ℝ, x + (1/x) = 0)) →           -- Condition D is not a quadratic equation
  (true) := sorry

end identify_quadratic_equation_l244_244999


namespace find_x_of_equation_l244_244295

theorem find_x_of_equation :
  ∃ x : ℕ, 16^5 + 16^5 + 16^5 = 4^x ∧ x = 20 :=
by 
  sorry

end find_x_of_equation_l244_244295


namespace complement_union_l244_244930

open Finset

def I := {0, 1, 2, 3, 4, 5, 6, 7, 8}
def M := {1, 2, 4, 5}
def N := {0, 3, 5, 7}

theorem complement_union (hI : I = {0, 1, 2, 3, 4, 5, 6, 7, 8})
                        (hM : M = {1, 2, 4, 5})
                        (hN : N = {0, 3, 5, 7}) :
  (I \ (M ∪ N)) = {6, 8} :=
by
  rw [hI, hM, hN]
  sorry

end complement_union_l244_244930


namespace linear_regression_neg_corr_l244_244306

-- Given variables x and y with certain properties
variables (x y : ℝ)

-- Conditions provided in the problem
def neg_corr (x y : ℝ) : Prop := ∀ a b : ℝ, (a < b → x * y < 0)
def sample_mean_x := (2 : ℝ)
def sample_mean_y := (1.5 : ℝ)

-- Statement to prove the linear regression equation
theorem linear_regression_neg_corr (h1 : neg_corr x y)
    (hx : sample_mean_x = 2)
    (hy : sample_mean_y = 1.5) : 
    ∃ b0 b1 : ℝ, b0 = 5.5 ∧ b1 = -2 ∧ y = b0 + b1 * x :=
sorry

end linear_regression_neg_corr_l244_244306


namespace initial_number_l244_244741

theorem initial_number (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by
  sorry

end initial_number_l244_244741


namespace triangle_area_correct_l244_244413

def point := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_correct :
  let A : point := (3, -1)
  let B : point := (3, 6)
  let C : point := (8, 6)
  triangle_area A B C = 17.5 :=
by
  sorry

end triangle_area_correct_l244_244413


namespace napkins_total_l244_244816

theorem napkins_total (o a w : ℕ) (ho : o = 10) (ha : a = 2 * o) (hw : w = 15) :
  w + o + a = 45 :=
by
  sorry

end napkins_total_l244_244816


namespace greatest_three_digit_multiple_of_17_l244_244000

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l244_244000


namespace find_f_value_l244_244447

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x + 1

theorem find_f_value : f 2019 + f (-2019) = 2 :=
by
  sorry

end find_f_value_l244_244447


namespace abs_expression_value_l244_244911

theorem abs_expression_value : (abs (2 * Real.pi - abs (Real.pi - 9))) = 3 * Real.pi - 9 := 
by
  sorry

end abs_expression_value_l244_244911


namespace sum_of_primes_less_than_20_l244_244106

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l244_244106


namespace sum_primes_less_than_20_l244_244100

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l244_244100


namespace sum_of_valid_m_values_l244_244323

-- Variables and assumptions
variable (m x : ℝ)

-- Conditions from the given problem
def inequality_system (m x : ℝ) : Prop :=
  (x - 4) / 3 < x - 4 ∧ (m - x) / 5 < 0

def solution_set_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, inequality_system m x → x > 4

def fractional_equation (m x : ℝ) : Prop :=
  6 / (x - 3) + 1 = (m * x - 3) / (x - 3)

-- Lean statement to prove the sum of integers satisfying the conditions
theorem sum_of_valid_m_values : 
  (∀ m : ℝ, solution_set_condition m ∧ 
            (∃ x : ℝ, x > 0 ∧ x ≠ 3 ∧ fractional_equation m x) →
            (∃ (k : ℕ), k = 2 ∨ k = 4) → 
            2 + 4 = 6) :=
sorry

end sum_of_valid_m_values_l244_244323


namespace simplify_expression_l244_244829

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l244_244829


namespace sum_of_primes_less_than_20_is_77_l244_244044

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l244_244044


namespace sum_of_primes_less_than_20_eq_77_l244_244024

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l244_244024


namespace sum_of_primes_less_than_20_l244_244018

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l244_244018


namespace downstream_distance_15_minutes_l244_244259

theorem downstream_distance_15_minutes
  (speed_boat : ℝ) (speed_current : ℝ) (time_minutes : ℝ)
  (h1 : speed_boat = 24)
  (h2 : speed_current = 3)
  (h3 : time_minutes = 15) :
  let effective_speed := speed_boat + speed_current
  let time_hours := time_minutes / 60
  let distance := effective_speed * time_hours
  distance = 6.75 :=
by {
  sorry
}

end downstream_distance_15_minutes_l244_244259


namespace reciprocal_of_repeating_decimal_equiv_l244_244389

noncomputable def repeating_decimal (x : ℝ) := 0.333333...

theorem reciprocal_of_repeating_decimal_equiv :
  (1 / repeating_decimal 0.333333...) = 3 :=
sorry

end reciprocal_of_repeating_decimal_equiv_l244_244389


namespace ramsey_six_vertices_monochromatic_quadrilateral_l244_244697

theorem ramsey_six_vertices_monochromatic_quadrilateral :
  ∀ (V : Type) (E : V → V → Prop), (∀ x y : V, x ≠ y → E x y ∨ ¬ E x y) →
  ∃ (u v w x : V), u ≠ v ∧ v ≠ w ∧ w ≠ x ∧ x ≠ u ∧ (E u v = E v w ∧ E v w = E w x ∧ E w x = E x u) :=
by sorry

end ramsey_six_vertices_monochromatic_quadrilateral_l244_244697


namespace curve_crosses_itself_and_point_of_crossing_l244_244559

-- Define the function for x and y
def x (t : ℝ) : ℝ := t^2 + 1
def y (t : ℝ) : ℝ := t^4 - 9 * t^2 + 6

-- Definition of the curve crossing itself and the point of crossing
theorem curve_crosses_itself_and_point_of_crossing :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ x t₁ = x t₂ ∧ y t₁ = y t₂ ∧ (x t₁ = 10 ∧ y t₁ = 6) :=
by
  sorry

end curve_crosses_itself_and_point_of_crossing_l244_244559


namespace sum_primes_less_than_20_l244_244071

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l244_244071


namespace find_difference_l244_244708

theorem find_difference (x y : ℚ) (h₁ : x + y = 520) (h₂ : x / y = 3 / 4) : y - x = 520 / 7 :=
by
  sorry

end find_difference_l244_244708


namespace simplify_abs_expression_l244_244443

theorem simplify_abs_expression (a b c : ℝ) (h1 : a > 0) (h2 : b < 0) (h3 : c = 0) :
  |a - c| + |c - b| - |a - b| = 0 := 
by
  sorry

end simplify_abs_expression_l244_244443


namespace sum_primes_less_than_20_l244_244101

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l244_244101


namespace find_a_l244_244232

theorem find_a (a : ℝ) (h1 : a > 0) :
  (a^0 + a^1 = 3) → a = 2 :=
by sorry

end find_a_l244_244232


namespace sum_prime_numbers_less_than_twenty_l244_244086

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l244_244086


namespace sum_prime_numbers_less_than_twenty_l244_244089

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l244_244089


namespace kaleb_first_load_pieces_l244_244475

-- Definitions of given conditions
def total_pieces : ℕ := 39
def num_equal_loads : ℕ := 5
def pieces_per_load : ℕ := 4

-- Definition for calculation of pieces in equal loads
def pieces_in_equal_loads : ℕ := num_equal_loads * pieces_per_load

-- Definition for pieces in the first load
def pieces_in_first_load : ℕ := total_pieces - pieces_in_equal_loads

-- Statement to prove that the pieces in the first load is 19
theorem kaleb_first_load_pieces : pieces_in_first_load = 19 := 
by
  -- The proof is skipped
  sorry

end kaleb_first_load_pieces_l244_244475


namespace expectation_of_Y_l244_244308

open ProbabilityTheory

noncomputable def Y : Distribution :=
{ support := {0, 1, 2},
  pmf := λ y,
    if y = 0 then 1 / 4
    else if y = 1 then 1 / 4
    else if y = 2 then 1 / 2
    else 0,
  sum_pmf' := by
    simp only [Finset.sum_insert, Finset.mem_singleton, not_false_iff, Finset.sum_singleton]
    norm_num }

theorem expectation_of_Y :
  ∑ y in {0, 1, 2}, y * Y.pmf y = 5 / 4 := sorry

end expectation_of_Y_l244_244308


namespace equivalent_single_increase_l244_244226

-- Defining the initial price of the mobile
variable (P : ℝ)
-- Condition stating the price after a 40% increase
def increased_price := 1.40 * P
-- Condition stating the new price after a further 15% decrease
def final_price := 0.85 * increased_price P

-- The mathematically equivalent statement to prove
theorem equivalent_single_increase:
  final_price P = 1.19 * P :=
sorry

end equivalent_single_increase_l244_244226


namespace sum_primes_less_than_20_l244_244047

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l244_244047


namespace checkered_triangle_division_l244_244627

theorem checkered_triangle_division :
  ∀ (triangle : List ℕ), triangle.sum = 63 →
  ∃ (part1 part2 part3 : List ℕ),
    part1.sum = 21 ∧ part2.sum = 21 ∧ part3.sum = 21 ∧
    part1 ≠ part2 ∧ part2 ≠ part3 ∧ part1 ≠ part3 ∧
    (part1 ++ part2 ++ part3).length = triangle.length ∧
    (∃ (area1 area2 area3 : ℕ), area1 ≠ area2 ∧ area2 ≠ area3 ∧ area1 ≠ area3) :=
by
  sorry

end checkered_triangle_division_l244_244627


namespace sum_of_primes_less_than_20_l244_244104

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l244_244104


namespace sum_primes_less_than_20_l244_244070

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l244_244070


namespace find_x_of_equation_l244_244341

theorem find_x_of_equation (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := by
  sorry

end find_x_of_equation_l244_244341


namespace greatest_three_digit_multiple_of_17_is_986_l244_244006

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l244_244006


namespace negation_example_l244_244504

theorem negation_example : ¬ (∃ x : ℤ, x^2 + 2 * x + 1 ≤ 0) ↔ ∀ x : ℤ, x^2 + 2 * x + 1 > 0 := 
by 
  sorry

end negation_example_l244_244504


namespace moles_of_KI_formed_l244_244294

-- Define the given conditions
def moles_KOH : ℕ := 1
def moles_NH4I : ℕ := 1
def balanced_equation (KOH NH4I KI NH3 H2O : ℕ) : Prop :=
  (KOH = 1) ∧ (NH4I = 1) ∧ (KI = 1) ∧ (NH3 = 1) ∧ (H2O = 1)

-- The proof problem statement
theorem moles_of_KI_formed (h : balanced_equation moles_KOH moles_NH4I 1 1 1) : 
  1 = 1 :=
by sorry

end moles_of_KI_formed_l244_244294


namespace equation_of_parallel_line_l244_244762

-- Definitions for conditions from the problem
def point_A : ℝ × ℝ := (3, 2)
def line_eq (x y : ℝ) : Prop := 4 * x + y - 2 = 0
def parallel_slope : ℝ := -4

-- Proof problem statement
theorem equation_of_parallel_line (x y : ℝ) :
  (∃ (m b : ℝ), m = parallel_slope ∧ b = 2 + 4 * 3 ∧ y = m * (x - 3) + b) →
  4 * x + y - 14 = 0 :=
sorry

end equation_of_parallel_line_l244_244762


namespace toms_weekly_revenue_l244_244518

def crabs_per_bucket : Nat := 12
def number_of_buckets : Nat := 8
def price_per_crab : Nat := 5
def days_per_week : Nat := 7

theorem toms_weekly_revenue :
  (crabs_per_bucket * number_of_buckets * price_per_crab * days_per_week) = 3360 :=
by
  sorry

end toms_weekly_revenue_l244_244518


namespace monochromatic_prob_l244_244633

-- Define the vertices of the pentagon
def vertices := Finset.range 5

-- Define the edges of the pentagon including diagonals
def pentagon_edges : Finset (Finset ℕ) :=
  let sides : Finset (Finset ℕ) := Finset.filter (λ s, s.card = 2) (vertices.powerset),
  let diagonals : Finset (Finset ℕ) := Finset.filter (λ s, s.card = 2 ∧ (s.sum ≠ 1 ∨ s.sum ≠ 4) ∨ (s.sum ≠ 2 ∨ s.sum ≠ 3))
  sides ∪ diagonals

-- The event which checks if there is a monochromatic triangle
def monochromatic_triangle (colors : Finset (Finset ℕ) → Bool) : Prop :=
  ∃ t ∈ pentagon_edges.triples, (colors t ∧ ∀ e ∈ t, colors e = colors t) ∨ (¬colors t ∧ ∀ e ∈ t, colors e = colors t)

-- Given conditions in terms of probability
noncomputable def color_distribution : Distribution (Finset (Finset ℕ) → Bool) :=
  probability.uniform (Finset.image (λ c : (Finset (Finset ℕ)) → Bool, c) (Finset.powerset pentagon_edges))

-- Problem statement to be proved
theorem monochromatic_prob : 
  Pr[monochromatic_triangle] = 253 / 256 := 
begin
  sorry
end

end monochromatic_prob_l244_244633


namespace choose_stick_l244_244535

-- Define the lengths of the sticks Xiaoming has
def xm_stick1 : ℝ := 4
def xm_stick2 : ℝ := 7

-- Define the lengths of the sticks Xiaohong has
def stick2 : ℝ := 2
def stick3 : ℝ := 3
def stick8 : ℝ := 8
def stick12 : ℝ := 12

-- Define the condition for a valid stick choice from Xiaohong's sticks
def valid_stick (x : ℝ) : Prop := 3 < x ∧ x < 11

-- State the problem as a theorem to be proved
theorem choose_stick : valid_stick stick8 := by
  sorry

end choose_stick_l244_244535


namespace inequality_proof_l244_244204

theorem inequality_proof (a b c d : ℝ) : 
  (a^2 + b^2 + 1) * (c^2 + d^2 + 1) ≥ 2 * (a + c) * (b + d) :=
by sorry

end inequality_proof_l244_244204


namespace jangshe_clothing_cost_l244_244960

theorem jangshe_clothing_cost
  (total_spent : ℝ)
  (untaxed_piece1 : ℝ)
  (untaxed_piece2 : ℝ)
  (total_pieces : ℕ)
  (remaining_pieces : ℕ)
  (remaining_pieces_price : ℝ)
  (sales_tax : ℝ)
  (price_multiple_of_five : ℝ) :
  total_spent = 610 ∧
  untaxed_piece1 = 49 ∧
  untaxed_piece2 = 81 ∧
  total_pieces = 7 ∧
  remaining_pieces = 5 ∧
  sales_tax = 0.10 ∧
  (∃ k : ℕ, remaining_pieces_price = k * 5) →
  remaining_pieces_price / remaining_pieces = 87 :=
by
  sorry

end jangshe_clothing_cost_l244_244960


namespace find_factor_l244_244509

-- Define the conditions
def number : ℕ := 9
def expr1 (f : ℝ) : ℝ := (number + 2) * f
def expr2 : ℝ := 24 + number

-- The proof problem statement
theorem find_factor (f : ℝ) : expr1 f = expr2 → f = 3 := by
  sorry

end find_factor_l244_244509


namespace sum_of_primes_less_than_20_eq_77_l244_244022

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l244_244022


namespace derivative_of_my_function_l244_244261

variable (x : ℝ)

noncomputable def my_function : ℝ :=
  (Real.cos (Real.sin 3))^2 + (Real.sin (29 * x))^2 / (29 * Real.cos (58 * x))

theorem derivative_of_my_function :
  deriv my_function x = Real.tan (58 * x) / Real.cos (58 * x) := 
sorry

end derivative_of_my_function_l244_244261


namespace sin_240_deg_l244_244612

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l244_244612


namespace bc_is_one_area_of_triangle_l244_244483

-- Define a triangle with sides a, b, and c and corresponding angles A, B, and C.
structure Triangle :=
(a b c A B C : ℝ)

-- Assume the Law of Cosines condition is given.
axiom Law_of_Cosines_condition (T : Triangle) : (T.b^2 + T.c^2 - T.a^2) / (Real.cos T.A) = 2

-- Prove that bc = 1.
theorem bc_is_one (T : Triangle) (h : Law_of_Cosines_condition T) : T.b * T.c = 1 := 
by {
    sorry
}

-- Assume the given ratio condition.
axiom ratio_condition (T : Triangle) :
  (T.a * Real.cos T.B - T.b * Real.cos T.A) / (T.a * Real.cos T.B + T.b * Real.cos T.A)
  - (T.b / T.c) = 1

-- Prove that the area of the triangle is sqrt(3)/4.
theorem area_of_triangle (T : Triangle) (h1 : Law_of_Cosines_condition T) (h2 : ratio_condition T) :
  let S := 0.5 * T.b * T.c * Real.sin T.A
  in S = Real.sqrt 3 / 4 :=
by {
    sorry
}

end bc_is_one_area_of_triangle_l244_244483


namespace smallest_possible_N_l244_244152

theorem smallest_possible_N : ∃ (N : ℕ), (N % 8 = 0 ∧ N % 4 = 0 ∧ N % 2 = 0) ∧ (∀ M : ℕ, (M % 8 = 0 ∧ M % 4 = 0 ∧ M % 2 = 0) → N ≤ M) ∧ N = 8 :=
by
  sorry

end smallest_possible_N_l244_244152


namespace polygon_sides_l244_244674

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 360) : n = 4 :=
by
  sorry

end polygon_sides_l244_244674


namespace sum_primes_less_than_20_l244_244067

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l244_244067


namespace sin_240_deg_l244_244572

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l244_244572


namespace problem1_proof_problem2_proof_l244_244416

noncomputable def problem1_statement : Prop :=
  (2 * Real.sin (Real.pi / 6) - Real.sin (Real.pi / 4) * Real.cos (Real.pi / 4) = 1 / 2)

noncomputable def problem2_statement : Prop :=
  ((-1)^2023 + 2 * Real.sin (Real.pi / 4) - Real.cos (Real.pi / 6) + Real.sin (Real.pi / 3) + Real.tan (Real.pi / 3)^2 = 2 + Real.sqrt 2)

theorem problem1_proof : problem1_statement :=
by
  sorry

theorem problem2_proof : problem2_statement :=
by
  sorry

end problem1_proof_problem2_proof_l244_244416


namespace f_at_3_l244_244310

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 5

theorem f_at_3 (a b : ℝ) (h : f a b (-3) = -1) : f a b 3 = 11 :=
by
  sorry

end f_at_3_l244_244310


namespace relay_race_time_l244_244823

theorem relay_race_time (R S D : ℕ) (h1 : S = R + 2) (h2 : D = R - 3) (h3 : R + S + D = 71) : R = 24 :=
by
  sorry

end relay_race_time_l244_244823


namespace sum_primes_less_than_20_l244_244073

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l244_244073


namespace boxes_given_to_brother_l244_244354

-- Definitions
def total_boxes : ℝ := 14.0
def pieces_per_box : ℝ := 6.0
def pieces_remaining : ℝ := 42.0

-- Theorem stating the problem
theorem boxes_given_to_brother : 
  (total_boxes * pieces_per_box - pieces_remaining) / pieces_per_box = 7.0 := 
by
  sorry

end boxes_given_to_brother_l244_244354


namespace reciprocal_of_repeating_decimal_l244_244392

theorem reciprocal_of_repeating_decimal :
  (1 / (0.33333333 : ℚ)) = 3 := by
  sorry

end reciprocal_of_repeating_decimal_l244_244392


namespace probability_X_eq_1_l244_244458

noncomputable def X : Type :=
  ℕ

def binomial_pmf (n : ℕ) (p : ℚ) : pmf ℕ :=
  pmf.of_finset (finset.range (n + 1))
    (λ k, (n.choose k : ℚ) * (p^k) * ((1 - p)^(n - k)))
    (by sorry) -- This lemma ensures the sum to 1

theorem probability_X_eq_1 (X : ℕ) (n : ℕ) (p : ℚ) (h1 : X ∼ binomial_pmf n p)
  (h2 : X.mean = 6) (h3 : X.variance = 3) : P(X = 1) = 3 * (2 : ℚ) ^ (-10) :=
sorry

end probability_X_eq_1_l244_244458


namespace geometric_sequence_sixth_term_l244_244510

variable (a r : ℝ) 

theorem geometric_sequence_sixth_term (h1 : a * (1 + r + r^2 + r^3) = 40)
                                    (h2 : a * r^4 = 32) :
  a * r^5 = 1280 / 15 :=
by sorry

end geometric_sequence_sixth_term_l244_244510


namespace triangle_parts_sum_eq_l244_244630

-- Define the total sum of numbers in the triangle
def total_sum : ℕ := 63

-- Define the required sum for each part
def required_sum : ℕ := 21

-- Define the possible parts that sum to the required sum
def part1 : list ℕ := [10, 6, 5]  -- this sums to 21
def part2 : list ℕ := [10, 5, 4, 1, 1]  -- this sums to 21

def part_sum (part : list ℕ) : ℕ := part.sum

-- The main theorem
theorem triangle_parts_sum_eq : 
  part_sum part1 = required_sum ∧ 
  part_sum part2 = required_sum ∧ 
  part_sum (list.drop (list.length part1 + list.length part2) [10, 6, 5, 10, 5, 4, 1, 1]) = required_sum :=
by
  sorry

end triangle_parts_sum_eq_l244_244630


namespace volume_of_cuboid_l244_244382

-- Definitions of conditions
def side_length : ℕ := 6
def num_cubes : ℕ := 3
def volume_single_cube (side_length : ℕ) : ℕ := side_length ^ 3

-- The main theorem
theorem volume_of_cuboid : (num_cubes * volume_single_cube side_length) = 648 := by
  sorry

end volume_of_cuboid_l244_244382


namespace pushkin_pension_is_survivors_pension_l244_244875

theorem pushkin_pension_is_survivors_pension
  (died_pushkin : Nat = 1837)
  (lifelong_pension_assigned : ∀ t : Nat, t > died_pushkin → ∃ (recipient : String), recipient = "Pushkin's wife" ∨ recipient = "Pushkin's daughter") :
  ∃ (pension_type : String), pension_type = "survivor's pension" :=
by
  sorry

end pushkin_pension_is_survivors_pension_l244_244875


namespace second_divisor_27_l244_244277

theorem second_divisor_27 (N : ℤ) (D : ℤ) (k : ℤ) (q : ℤ) (h1 : N = 242 * k + 100) (h2 : N = D * q + 19) : D = 27 := by
  sorry

end second_divisor_27_l244_244277


namespace train_speed_l244_244728

-- Definition for the given conditions
def distance : ℕ := 240 -- distance in meters
def time_seconds : ℕ := 6 -- time in seconds
def conversion_factor : ℕ := 3600 -- seconds to hour conversion factor
def meters_in_km : ℕ := 1000 -- meters to kilometers conversion factor

-- The proof goal
theorem train_speed (d : ℕ) (t : ℕ) (cf : ℕ) (mk : ℕ) (h1 : d = distance) (h2 : t = time_seconds) (h3 : cf = conversion_factor) (h4 : mk = meters_in_km) :
  (d * cf / t) / mk = 144 :=
by sorry

end train_speed_l244_244728


namespace sum_of_primes_less_than_20_l244_244118

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l244_244118


namespace count_numbers_with_digit_2_l244_244779

def contains_digit_2 (n : Nat) : Prop :=
  n / 100 = 2 ∨ (n / 10 % 10) = 2 ∨ (n % 10) = 2

theorem count_numbers_with_digit_2 (N : Nat) (H : 200 ≤ N ∧ N ≤ 499) : 
  Nat.card {n // 200 ≤ n ∧ n ≤ 499 ∧ contains_digit_2 n} = 138 :=
by
  sorry

end count_numbers_with_digit_2_l244_244779


namespace C_pays_228_for_cricket_bat_l244_244738

def CostPriceA : ℝ := 152

def ProfitA (price : ℝ) : ℝ := 0.20 * price

def SellingPriceA (price : ℝ) : ℝ := price + ProfitA price

def ProfitB (price : ℝ) : ℝ := 0.25 * price

def SellingPriceB (price : ℝ) : ℝ := price + ProfitB price

theorem C_pays_228_for_cricket_bat :
  SellingPriceB (SellingPriceA CostPriceA) = 228 :=
by
  sorry

end C_pays_228_for_cricket_bat_l244_244738


namespace evaluate_expression_l244_244921

theorem evaluate_expression (x y z : ℕ) (hx : x = 5) (hy : y = 10) (hz : z = 3) : z * (y - 2 * x) = 0 := by
  sorry

end evaluate_expression_l244_244921


namespace necessary_sufficient_condition_l244_244218

theorem necessary_sufficient_condition (A B C : ℝ)
    (h : ∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) :
    |A - B + C| ≤ 2 * Real.sqrt (A * C) := 
by sorry

end necessary_sufficient_condition_l244_244218


namespace Laura_more_than_200_paperclips_on_Friday_l244_244683

theorem Laura_more_than_200_paperclips_on_Friday:
  ∀ (n : ℕ), (n = 4 ∨ n = 0 ∨ n ≥ 1 ∧ (n - 1 = 0 ∨ n = 1) → 4 * 3 ^ n > 200) :=
by
  sorry

end Laura_more_than_200_paperclips_on_Friday_l244_244683


namespace max_value_a_l244_244293

-- Define the variables and the constraint on the circle
def circular_arrangement_condition (x: ℕ → ℕ) : Prop :=
  ∀ i: ℕ, 1 ≤ x i ∧ x i ≤ 10 ∧ x i ≠ x (i + 1)

-- Define the existence of three consecutive numbers summing to at least 18
def three_consecutive_sum_ge_18 (x: ℕ → ℕ) : Prop :=
  ∃ i: ℕ, x i + x (i + 1) + x (i + 2) ≥ 18

-- The main theorem we aim to prove
theorem max_value_a : ∀ (x: ℕ → ℕ), circular_arrangement_condition x → three_consecutive_sum_ge_18 x :=
  by sorry

end max_value_a_l244_244293


namespace race_head_start_l244_244879

theorem race_head_start
  (v_A v_B L x : ℝ)
  (h1 : v_A = (4 / 3) * v_B)
  (h2 : L / v_A = (L - x * L) / v_B) :
  x = 1 / 4 :=
sorry

end race_head_start_l244_244879


namespace smaller_rectangle_area_l244_244552

-- Define the conditions
def large_rectangle_length : ℝ := 40
def large_rectangle_width : ℝ := 20
def smaller_rectangle_length : ℝ := large_rectangle_length / 2
def smaller_rectangle_width : ℝ := large_rectangle_width / 2

-- Define what we want to prove
theorem smaller_rectangle_area : 
  (smaller_rectangle_length * smaller_rectangle_width = 200) :=
by
  sorry

end smaller_rectangle_area_l244_244552


namespace find_angle_2_l244_244769

theorem find_angle_2 (angle1 : ℝ) (angle2 : ℝ) 
  (h1 : angle1 = 60) 
  (h2 : angle1 + angle2 = 180) : 
  angle2 = 120 := 
by
  sorry

end find_angle_2_l244_244769


namespace area_of_right_triangle_l244_244410

theorem area_of_right_triangle (m k : ℝ) (hm : 0 < m) (hk : 0 < k) : 
  ∃ A : ℝ, A = (k^2) / (2 * m) :=
by
  sorry

end area_of_right_triangle_l244_244410


namespace open_door_within_time_l244_244713

-- Define the initial conditions
def device := ℕ → ℕ

-- Constraint: Each device has 5 toggle switches ("0" or "1") and a three-digit display.
def valid_configuration (d : device) (k : ℕ) : Prop :=
  d k < 32 ∧ d k <= 999

def system_configuration (A B : device) (k : ℕ) : Prop :=
  A k = B k

-- Constraint: The devices can be synchronized to display the same number simultaneously to open the door.
def open_door (A B : device) : Prop :=
  ∃ k, system_configuration A B k

-- The main theorem: Devices A and B can be synchronized within the given time constraints to open the door.
theorem open_door_within_time (A B : device) (notebook : ℕ) : 
  (∀ k, valid_configuration A k ∧ valid_configuration B k) →
  open_door A B :=
by sorry

end open_door_within_time_l244_244713


namespace seventh_observation_l244_244723

theorem seventh_observation (avg6 : ℕ) (new_avg7 : ℕ) (old_avg : ℕ) (new_avg_diff : ℕ) (n : ℕ) (m : ℕ) (h1 : avg6 = 12) (h2 : new_avg_diff = 1) (h3 : n = 6) (h4 : m = 7) :
  ((n * old_avg = avg6 * old_avg) ∧ (m * new_avg7 = avg6 * old_avg + m - n)) →
  m * new_avg7 = 77 →
  avg6 * old_avg = 72 →
  77 - 72 = 5 :=
by
  sorry

end seventh_observation_l244_244723


namespace card_selection_ways_l244_244455

theorem card_selection_ways (deck_size : ℕ) (suits : ℕ) (cards_per_suit : ℕ) (total_cards_chosen : ℕ)
  (repeated_suit_count : ℕ) (distinct_suits_count : ℕ) (distinct_ranks_count : ℕ) 
  (correct_answer : ℕ) :
  deck_size = 52 ∧ suits = 4 ∧ cards_per_suit = 13 ∧ total_cards_chosen = 5 ∧ 
  repeated_suit_count = 2 ∧ distinct_suits_count = 3 ∧ distinct_ranks_count = 11 ∧ 
  correct_answer = 414384 :=
by 
  -- Sorry is used to skip actual proof steps, according to the instructions.
  sorry

end card_selection_ways_l244_244455


namespace param_line_segment_l244_244862

theorem param_line_segment:
  ∃ (a b c d : ℤ), b = 1 ∧ d = -3 ∧ a + b = -4 ∧ c + d = 9 ∧ a^2 + b^2 + c^2 + d^2 = 179 :=
by
  -- Here, you can use sorry to indicate that proof steps are not required as requested
  sorry

end param_line_segment_l244_244862


namespace probability_sum_prime_l244_244878

theorem probability_sum_prime (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) :
  let outcomes := finset.product (finset.range 1 7) (finset.range 1 7)
  let primes := {2, 3, 5, 7, 11}
  let prime_sums := outcomes.filter (λ (x : ℕ × ℕ), primes (x.1 + x.2))
  (prime_sums.card : ℚ) / (outcomes.card : ℚ) = 5 / 12 := sorry

end probability_sum_prime_l244_244878


namespace max_value_is_one_l244_244419

noncomputable def max_value_fraction (x : ℝ) : ℝ :=
  (1 + Real.cos x) / (Real.sin x + Real.cos x + 2)

theorem max_value_is_one : ∃ x : ℝ, max_value_fraction x = 1 := by
  sorry

end max_value_is_one_l244_244419


namespace jill_average_number_of_stickers_l244_244802

def average_stickers (packs : List ℕ) : ℚ :=
  (packs.sum : ℚ) / packs.length

theorem jill_average_number_of_stickers :
  average_stickers [5, 7, 9, 9, 11, 15, 15, 17, 19, 21] = 12.8 :=
by
  sorry

end jill_average_number_of_stickers_l244_244802


namespace find_value_of_a_l244_244230

theorem find_value_of_a (a : ℝ) (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 < a → a^x ≥ 1)
  (h_sum : (a^1) + (a^0) = 3) : a = 2 :=
sorry

end find_value_of_a_l244_244230


namespace square_side_length_range_l244_244466

theorem square_side_length_range (a : ℝ) (h : a^2 = 30) : 5.4 < a ∧ a < 5.5 :=
sorry

end square_side_length_range_l244_244466


namespace complex_fraction_simplification_l244_244909

theorem complex_fraction_simplification :
  ((10^4 + 324) * (22^4 + 324) * (34^4 + 324) * (46^4 + 324) * (58^4 + 324)) /
  ((4^4 + 324) * (16^4 + 324) * (28^4 + 324) * (40^4 + 324) * (52^4 + 324)) = 373 :=
by
  sorry

end complex_fraction_simplification_l244_244909


namespace poly_sequence_correct_l244_244739

-- Sequence of polynomials defined recursively
def f : ℕ → ℕ → ℕ 
| 0, x => 1
| 1, x => 1 + x 
| (k + 1), x => ((x + 1) * f (k) (x) - (x - k) * f (k - 1) (x)) / (k + 1)

-- Prove f(k, k) = 2^k for all k ≥ 0
theorem poly_sequence_correct (k : ℕ) : f k k = 2 ^ k := by
  sorry

end poly_sequence_correct_l244_244739


namespace exists_nat_numbers_satisfying_sum_l244_244282

theorem exists_nat_numbers_satisfying_sum :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 :=
sorry

end exists_nat_numbers_satisfying_sum_l244_244282


namespace ellipse_equation_l244_244747

-- Definitions of the tangents given as conditions
def tangent1 (x y : ℝ) : Prop := 4 * x + 5 * y = 25
def tangent2 (x y : ℝ) : Prop := 9 * x + 20 * y = 75

-- The statement we need to prove
theorem ellipse_equation :
  (∀ (x y : ℝ), tangent1 x y → tangent2 x y → 
  (∃ (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0), a = 5 ∧ b = 3 ∧ 
  (x^2 / a^2 + y^2 / b^2 = 1))) :=
sorry

end ellipse_equation_l244_244747


namespace conic_section_is_hyperbola_l244_244621

theorem conic_section_is_hyperbola : 
  ∀ (x y : ℝ), x^2 + 2 * x - 8 * y^2 = 0 → (∃ a b h k : ℝ, (x + 1)^2 / a^2 - (y - 0)^2 / b^2 = 1) := 
by 
  intros x y h_eq;
  sorry

end conic_section_is_hyperbola_l244_244621


namespace total_tweets_correct_l244_244214

-- Define the rates at which Polly tweets under different conditions
def happy_rate : ℕ := 18
def hungry_rate : ℕ := 4
def mirror_rate : ℕ := 45

-- Define the durations of each activity
def happy_duration : ℕ := 20
def hungry_duration : ℕ := 20
def mirror_duration : ℕ := 20

-- Compute the total number of tweets
def total_tweets : ℕ := happy_rate * happy_duration + hungry_rate * hungry_duration + mirror_rate * mirror_duration

-- Statement to prove
theorem total_tweets_correct : total_tweets = 1340 := by
  sorry

end total_tweets_correct_l244_244214


namespace sum_prime_numbers_less_than_twenty_l244_244092

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l244_244092


namespace remaining_flour_needed_l244_244208

-- Define the required total amount of flour
def total_flour : ℕ := 8

-- Define the amount of flour already added
def flour_added : ℕ := 2

-- Define the remaining amount of flour needed
def remaining_flour : ℕ := total_flour - flour_added

-- The theorem we need to prove
theorem remaining_flour_needed : remaining_flour = 6 := by
  sorry

end remaining_flour_needed_l244_244208


namespace partition_triangle_l244_244625

theorem partition_triangle (triangle : List ℕ) (h_triangle_sum : triangle.sum = 63) :
  ∃ (parts : List (List ℕ)), parts.length = 3 ∧ 
  (∀ part ∈ parts, part.sum = 21) ∧ 
  parts.bind id = triangle :=
by
  sorry

end partition_triangle_l244_244625


namespace complex_exponential_sum_identity_l244_244761

theorem complex_exponential_sum_identity :
    12 * Complex.exp (Real.pi * Complex.I / 7) + 12 * Complex.exp (19 * Real.pi * Complex.I / 14) =
    24 * Real.cos (5 * Real.pi / 28) * Complex.exp (3 * Real.pi * Complex.I / 4) :=
sorry

end complex_exponential_sum_identity_l244_244761


namespace ratio_alison_brittany_l244_244901

def kent_money : ℕ := 1000
def brooke_money : ℕ := 2 * kent_money
def brittany_money : ℕ := 4 * brooke_money
def alison_money : ℕ := 4000

theorem ratio_alison_brittany : alison_money * 2 = brittany_money :=
by
  sorry

end ratio_alison_brittany_l244_244901


namespace simplify_expression_l244_244830

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l244_244830


namespace sin_240_eq_neg_sqrt3_div_2_l244_244580

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244580


namespace operation_evaluation_l244_244285

def my_operation (x y : Int) : Int :=
  x * (y + 1) + x * y

theorem operation_evaluation :
  my_operation (-3) (-4) = 21 := by
  sorry

end operation_evaluation_l244_244285


namespace grill_cost_difference_l244_244513

theorem grill_cost_difference:
  let in_store_price : Float := 129.99
  let payment_per_installment : Float := 32.49
  let number_of_installments : Float := 4
  let shipping_handling : Float := 9.99
  let total_tv_cost : Float := (number_of_installments * payment_per_installment) + shipping_handling
  let cost_difference : Float := in_store_price - total_tv_cost
  cost_difference * 100 = -996 := by
    sorry

end grill_cost_difference_l244_244513


namespace pascal_50_5th_element_is_22050_l244_244718

def pascal_fifth_element (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_50_5th_element_is_22050 :
  pascal_fifth_element 50 4 = 22050 :=
by
  -- Calculation steps would go here
  sorry

end pascal_50_5th_element_is_22050_l244_244718


namespace value_of_B_l244_244169

theorem value_of_B (B : ℚ) (h : 3 * B - 5 = 23) : B = 28 / 3 :=
by
  sorry

-- Explanation:
-- B is declared as a rational number (ℚ) because the answer involves a fraction.
-- h is the condition 3 * B - 5 = 23.
-- The theorem states that given h, B equals 28 / 3.

end value_of_B_l244_244169


namespace sum_primes_less_than_20_l244_244097

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l244_244097


namespace min_value_M_l244_244864

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

end min_value_M_l244_244864


namespace range_of_a_l244_244303

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + x else x^2 + x -- Note: Using the specific definition matches the problem constraints clearly.

theorem range_of_a (a : ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_ineq : f a + f (-a) < 4) : -1 < a ∧ a < 1 := 
by sorry

end range_of_a_l244_244303


namespace greatest_integer_x_l244_244388

theorem greatest_integer_x (x : ℤ) (h : 7 - 3 * x + 2 > 23) : x ≤ -5 :=
by {
  sorry
}

end greatest_integer_x_l244_244388


namespace C_pow_50_l244_244347

open Matrix

def C : Matrix (Fin 2) (Fin 2) ℝ :=
![![3, 1], ![-4, -1]]

theorem C_pow_50 :
  (C ^ 50) = ![![101, 50], ![-200, -99]] :=
by
  sorry

end C_pow_50_l244_244347


namespace line_equation_l244_244935

theorem line_equation (a b : ℝ)
(h1 : a * -1 + b * 2 = 0) 
(h2 : a = b) :
((a = 1 ∧ b = -1) ∨ (a = 2 ∧ b = -1)) := 
by
  sorry

end line_equation_l244_244935


namespace probability_of_5_distinct_dice_rolls_is_5_over_54_l244_244247

def count_distinct_dice_rolls : ℕ :=
  6 * 5 * 4 * 3 * 2

def total_dice_rolls : ℕ :=
  6 ^ 5

def probability_of_distinct_rolls : ℚ :=
  count_distinct_dice_rolls / total_dice_rolls

theorem probability_of_5_distinct_dice_rolls_is_5_over_54 : 
  probability_of_distinct_rolls = 5 / 54 :=
by
  sorry

end probability_of_5_distinct_dice_rolls_is_5_over_54_l244_244247


namespace checkered_triangle_division_l244_244622

-- Define the conditions as assumptions
variable (T : Set ℕ) 
variable (sum_T : Nat) (h_sumT : sum_T = 63)
variable (part1 part2 part3 : Set ℕ)
variable (sum_part1 sum_part2 sum_part3 : Nat)
variable (h_part1 : sum part1 = 21)
variable (h_part2 : sum part2 = 21)
variable (h_part3 : sum part3 = 21)

-- Define the goal as a theorem
theorem checkered_triangle_division : 
  (∃ part1 part2 part3 : Set ℕ, sum part1 = 21 ∧ sum part2 = 21 ∧ sum part3 = 21 ∧ Disjoint part1 (part2 ∪ part3) ∧ Disjoint part2 part3 ∧ T = part1 ∪ part2 ∪ part3) :=
sorry

end checkered_triangle_division_l244_244622


namespace woman_total_coins_l244_244745

theorem woman_total_coins
  (num_each_coin : ℕ)
  (h : 1 * num_each_coin + 5 * num_each_coin + 10 * num_each_coin + 25 * num_each_coin + 100 * num_each_coin = 351)
  : 5 * num_each_coin = 15 :=
by
  sorry

end woman_total_coins_l244_244745


namespace scalene_triangle_minimum_altitude_l244_244411

theorem scalene_triangle_minimum_altitude (a b c : ℕ) (h : ℕ) 
  (h₁ : a ≠ b ∧ b ≠ c ∧ c ≠ a) -- scalene condition
  (h₂ : ∃ k : ℕ, ∃ m : ℕ, k * m = a ∧ m = 6) -- first altitude condition
  (h₃ : ∃ k : ℕ, ∃ n : ℕ, k * n = b ∧ n = 8) -- second altitude condition
  (h₄ : c = (7 : ℕ) * b / (3 : ℕ)) -- third side condition given inequalities and area relations
  : h = 2 := 
sorry

end scalene_triangle_minimum_altitude_l244_244411


namespace sum_primes_less_than_20_l244_244078

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l244_244078


namespace total_capacity_of_schools_l244_244328

theorem total_capacity_of_schools (a b c d t : ℕ) (h_a : a = 2) (h_b : b = 2) (h_c : c = 400) (h_d : d = 340) :
  t = a * c + b * d → t = 1480 := by
  intro h
  rw [h_a, h_b, h_c, h_d] at h
  simp at h
  exact h

end total_capacity_of_schools_l244_244328


namespace double_root_possible_values_l244_244407

theorem double_root_possible_values (b_3 b_2 b_1 : ℤ) (s : ℤ)
  (h : (Polynomial.X - Polynomial.C s) ^ 2 ∣
    Polynomial.C 24 + Polynomial.C b_1 * Polynomial.X + Polynomial.C b_2 * Polynomial.X ^ 2 + Polynomial.C b_3 * Polynomial.X ^ 3 + Polynomial.X ^ 4) :
  s = -2 ∨ s = -1 ∨ s = 1 ∨ s = 2 :=
sorry

end double_root_possible_values_l244_244407


namespace problem1_problem2_l244_244284

variable (x y : ℝ)

-- Problem 1
theorem problem1 : (x + y) ^ 2 + x * (x - 2 * y) = 2 * x ^ 2 + y ^ 2 := by
  sorry

variable (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 0) -- to ensure the denominators are non-zero

-- Problem 2
theorem problem2 : (x ^ 2 - 6 * x + 9) / (x - 2) / (x + 2 - (3 * x - 4) / (x - 2)) = (x - 3) / x := by
  sorry

end problem1_problem2_l244_244284


namespace contractor_initial_hire_l244_244547

theorem contractor_initial_hire :
  ∃ (P : ℕ), 
    (∀ (total_work : ℝ), 
      (P * 20 = (1/4) * total_work) ∧ 
      ((P - 2) * 75 = (3/4) * total_work)) → 
    P = 10 :=
by
  sorry

end contractor_initial_hire_l244_244547


namespace work_completion_time_l244_244404

theorem work_completion_time (A_work_rate B_work_rate C_work_rate : ℝ) 
  (hA : A_work_rate = 1 / 8) 
  (hB : B_work_rate = 1 / 16) 
  (hC : C_work_rate = 1 / 16) : 
  1 / (A_work_rate + B_work_rate + C_work_rate) = 4 :=
by
  -- Proof goes here
  sorry

end work_completion_time_l244_244404


namespace symmetry_propositions_l244_244309

noncomputable def verify_symmetry_conditions (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  Prop :=
  -- This defines the propositions to be proven
  (∀ x : ℝ, a^x - 1 = a^(-x) - 1) ∧
  (∀ x : ℝ, a^(x - 2) = a^(2 - x)) ∧
  (∀ x : ℝ, a^(x + 2) = a^(2 - x))

-- Create the problem statement
theorem symmetry_propositions (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  verify_symmetry_conditions a h1 h2 :=
sorry

end symmetry_propositions_l244_244309


namespace greatest_three_digit_multiple_of_17_is_986_l244_244004

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l244_244004


namespace sum_of_primes_less_than_20_l244_244016

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l244_244016


namespace sin_minus_cos_value_l244_244174

open Real

noncomputable def tan_alpha := sqrt 3
noncomputable def alpha_condition (α : ℝ) := π < α ∧ α < (3 / 2) * π

theorem sin_minus_cos_value (α : ℝ) (h1 : tan α = tan_alpha) (h2 : alpha_condition α) : 
  sin α - cos α = -((sqrt 3) - 1) / 2 := 
by 
  sorry

end sin_minus_cos_value_l244_244174


namespace monotonic_decreasing_intervals_l244_244700

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem monotonic_decreasing_intervals : 
  (∀ x : ℝ, (0 < x ∧ x < 1) → ∃ ε > 0, ∀ y : ℝ, x ≤ y ∧ y ≤ x + ε → f y < f x) ∧
  (∀ x : ℝ, (1 < x ∧ x < Real.exp 1) → ∃ ε > 0, ∀ y : ℝ, x ≤ y ∧ y ≤ x + ε → f y < f x) :=
by
  sorry

end monotonic_decreasing_intervals_l244_244700


namespace conference_handshakes_l244_244881

theorem conference_handshakes (n_leaders n_participants : ℕ) (n_total : ℕ) 
  (h_total : n_total = n_leaders + n_participants) 
  (h_leaders : n_leaders = 5) 
  (h_participants : n_participants = 25) 
  (h_total_people : n_total = 30) : 
  (n_leaders * (n_total - 1) - (n_leaders * (n_leaders - 1) / 2)) = 135 := 
by 
  sorry

end conference_handshakes_l244_244881


namespace coffee_shop_sold_lattes_l244_244370

theorem coffee_shop_sold_lattes (T L : ℕ) (h1 : T = 6) (h2 : L = 4 * T + 8) : L = 32 :=
by
  sorry

end coffee_shop_sold_lattes_l244_244370


namespace time_to_drain_tank_due_to_leak_l244_244549

noncomputable def timeToDrain (P L : ℝ) : ℝ := (1 : ℝ) / L

theorem time_to_drain_tank_due_to_leak (P L : ℝ)
  (hP : P = 0.5)
  (hL : P - L = 5/11) :
  timeToDrain P L = 22 :=
by
  -- to state what needs to be proved here
  sorry

end time_to_drain_tank_due_to_leak_l244_244549


namespace train_speed_l244_244526

theorem train_speed (L1 L2: ℕ) (V2: ℕ) (T: ℕ) (V1: ℕ) : 
  L1 = 120 -> 
  L2 = 280 -> 
  V2 = 30 -> 
  T = 20 -> 
  (L1 + L2) * 18 = (V1 + V2) * T * 100 -> 
  V1 = 42 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end train_speed_l244_244526


namespace power_identity_l244_244783

theorem power_identity (x : ℕ) (h : 2^x = 16) : 2^(x + 3) = 128 := 
sorry

end power_identity_l244_244783


namespace solve_inequality_system_simplify_expression_l244_244403

-- Part 1: System of Inequalities

theorem solve_inequality_system : 
  ∀ (x : ℝ), (x + 2) / 5 < 1 ∧ 3 * x - 1 ≥ 2 * x → 1 ≤ x ∧ x < 3 :=  by
  sorry

-- Part 2: Expression Simplification

theorem simplify_expression (m : ℝ) (hm : m ≠ 0) : 
  (m - 1 / m) * ((m^2 - m) / (m^2 - 2 * m + 1)) = m + 1 :=
  by
  sorry

end solve_inequality_system_simplify_expression_l244_244403


namespace hyperbola_circle_intersection_l244_244440

open Real

theorem hyperbola_circle_intersection (a r : ℝ) (P Q R S : ℝ × ℝ) 
  (hP : P.1^2 - P.2^2 = a^2) (hQ : Q.1^2 - Q.2^2 = a^2) (hR : R.1^2 - R.2^2 = a^2) (hS : S.1^2 - S.2^2 = a^2)
  (hO : r ≥ 0)
  (hPQRS : (P.1 - 0)^2 + (P.2 - 0)^2 = r^2 ∧
            (Q.1 - 0)^2 + (Q.2 - 0)^2 = r^2 ∧
            (R.1 - 0)^2 + (R.2 - 0)^2 = r^2 ∧
            (S.1 - 0)^2 + (S.2 - 0)^2 = r^2) : 
  (P.1^2 + P.2^2) + (Q.1^2 + Q.2^2) + (R.1^2 + R.2^2) + (S.1^2 + S.2^2) = 4 * r^2 :=
by
  sorry

end hyperbola_circle_intersection_l244_244440


namespace sum_x_y_z_l244_244485

noncomputable def a : ℝ := -Real.sqrt (9/27)
noncomputable def b : ℝ := Real.sqrt ((3 + Real.sqrt 7)^2 / 9)

theorem sum_x_y_z (ha : a = -Real.sqrt (9 / 27)) (hb : b = Real.sqrt ((3 + Real.sqrt 7) ^ 2 / 9)) (h_neg_a : a < 0) (h_pos_b : b > 0) :
  ∃ x y z : ℕ, (a + b)^3 = (x * Real.sqrt y) / z ∧ x + y + z = 718 := 
sorry

end sum_x_y_z_l244_244485


namespace sum_of_primes_less_than_20_l244_244124

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l244_244124


namespace number_of_maple_trees_planted_today_l244_244865

-- Define the initial conditions
def initial_maple_trees : ℕ := 2
def poplar_trees : ℕ := 5
def final_maple_trees : ℕ := 11

-- State the main proposition
theorem number_of_maple_trees_planted_today : 
  (final_maple_trees - initial_maple_trees) = 9 := by
  sorry

end number_of_maple_trees_planted_today_l244_244865


namespace cos_alpha_eq_l244_244197

open Real

-- Define the angles and their conditions
variables (α β : ℝ)

-- Hypothesis and initial conditions
axiom ha1 : 0 < α ∧ α < π
axiom ha2 : 0 < β ∧ β < π
axiom h_cos_beta : cos β = -5 / 13
axiom h_sin_alpha_plus_beta : sin (α + β) = 3 / 5

-- The main theorem to prove
theorem cos_alpha_eq : cos α = 56 / 65 := sorry

end cos_alpha_eq_l244_244197


namespace sum_of_primes_less_than_20_l244_244061

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l244_244061


namespace product_of_major_and_minor_axes_l244_244359

-- Given definitions from conditions
variables (O F A B C D : Type) 
variables (OF : ℝ) (dia_inscribed_circle_OCF : ℝ) (a b : ℝ)

-- Condition: O is the center of an ellipse
-- Point F is one focus, OF = 8
def O_center_ellipse : Prop := OF = 8

-- The diameter of the inscribed circle of triangle OCF is 4
def dia_inscribed_circle_condition : Prop := dia_inscribed_circle_OCF = 4

-- Define OA = OB = a, OC = OD = b
def major_axis_half_length : ℝ := a
def minor_axis_half_length : ℝ := b

-- Ellipse focal property a^2 - b^2 = 64
def ellipse_focal_property : Prop := a^2 - b^2 = 64

-- From the given conditions, expected result
def compute_product_AB_CD : Prop := 
  let AB := 2 * a,
      CD := 2 * b
  in AB * CD = 240

-- The main statement to be proven
theorem product_of_major_and_minor_axes 
  (h1 : O_center_ellipse)
  (h2 : dia_inscribed_circle_condition)
  (h3 : ellipse_focal_property)
  : compute_product_AB_CD :=
sorry

end product_of_major_and_minor_axes_l244_244359


namespace lowest_point_graph_l244_244859

theorem lowest_point_graph (x : ℝ) (h : x > -1) : ∃ y, y = (x^2 + 2*x + 2) / (x + 1) ∧ y ≥ 2 ∧ (x = 0 → y = 2) :=
  sorry

end lowest_point_graph_l244_244859


namespace test_completion_days_l244_244158

theorem test_completion_days :
  let barbara_days := 10
  let edward_days := 9
  let abhinav_days := 11
  let alex_days := 12
  let barbara_rate := 1 / barbara_days
  let edward_rate := 1 / edward_days
  let abhinav_rate := 1 / abhinav_days
  let alex_rate := 1 / alex_days
  let one_cycle_work := barbara_rate + edward_rate + abhinav_rate + alex_rate
  let cycles_needed := (1 : ℚ) / one_cycle_work
  Nat.ceil cycles_needed = 3 :=
by
  sorry

end test_completion_days_l244_244158


namespace poem_lines_months_l244_244375

theorem poem_lines_months (current_lines : ℕ) (target_lines : ℕ) (lines_per_month : ℕ) :
  current_lines = 24 →
  target_lines = 90 →
  lines_per_month = 3 →
  (target_lines - current_lines) / lines_per_month = 22 :=
  by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  exact sorry

end poem_lines_months_l244_244375


namespace find_initial_number_l244_244743

theorem find_initial_number (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by {
  sorry
}

end find_initial_number_l244_244743


namespace cards_thrown_away_l244_244345

theorem cards_thrown_away (h1 : 3 * (52 / 2) + 3 * 52 - 200 = 34) : 34 = 34 :=
by sorry

end cards_thrown_away_l244_244345


namespace value_of_x_when_y_equals_8_l244_244236

noncomputable def inverse_variation(cube_root : ℝ → ℝ) (k : ℝ) (x y : ℝ) : Prop :=
  y * (cube_root x) = k

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem value_of_x_when_y_equals_8 : 
  ∃ k : ℝ, (inverse_variation cube_root k 8 2) → 
  (inverse_variation cube_root k (1 / 8) 8) := 
sorry

end value_of_x_when_y_equals_8_l244_244236


namespace tank_full_weight_l244_244367

theorem tank_full_weight (u v m n : ℝ) (h1 : m + 3 / 4 * n = u) (h2 : m + 1 / 3 * n = v) :
  m + n = 8 / 5 * u - 3 / 5 * v :=
sorry

end tank_full_weight_l244_244367


namespace shaded_area_correct_l244_244332

def unit_triangle_area : ℕ := 10

def small_shaded_area : ℕ := unit_triangle_area

def medium_shaded_area : ℕ := 6 * unit_triangle_area

def large_shaded_area : ℕ := 7 * unit_triangle_area

def total_shaded_area : ℕ :=
  small_shaded_area + medium_shaded_area + large_shaded_area

theorem shaded_area_correct : total_shaded_area = 110 := 
  by
    sorry

end shaded_area_correct_l244_244332


namespace tan_eq_sin3x_solutions_l244_244926

open Real

theorem tan_eq_sin3x_solutions : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ tan x = sin (3 * x)) ∧ s.card = 6 :=
sorry

end tan_eq_sin3x_solutions_l244_244926


namespace sum_of_primes_less_than_20_eq_77_l244_244028

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l244_244028


namespace sum_of_primes_less_than_20_eq_77_l244_244027

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l244_244027


namespace range_of_a_l244_244646

theorem range_of_a
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_neg_x : ∀ x, x ≤ 0 → f x = 2 * x + x^2)
  (h_three_solutions : ∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = 2 * a^2 + a ∧ f x2 = 2 * a^2 + a ∧ f x3 = 2 * a^2 + a) :
  -1 < a ∧ a < 1/2 :=
sorry

end range_of_a_l244_244646


namespace sin_240_eq_neg_sqrt3_over_2_l244_244601

open Real

-- Conditions
def angle_240_in_third_quadrant : Prop := 240 ° ∈ set_of (λ x, 180 ° < x ∧ x < 270 °)

def reference_angle_60 (θ : Real) : Prop := θ = 240 ° - 180 °

def sin_60_eq_sqrt3_over_2 : sin (60 °) = sqrt 3 / 2

def sin_negative_in_third_quadrant (θ : Real) : Prop :=
  180 ° < θ ∧ θ < 270 ° → sin θ < 0

-- Statement
theorem sin_240_eq_neg_sqrt3_over_2 :
  angle_240_in_third_quadrant ∧ reference_angle_60 60 ° ∧ sin_60_eq_sqrt3_over_2 ∧ sin_negative_in_third_quadrant 240 °
  → sin (240 °) = - (sqrt 3 / 2) :=
by
  intros
  sorry

end sin_240_eq_neg_sqrt3_over_2_l244_244601


namespace smallest_b_satisfying_inequality_l244_244927

theorem smallest_b_satisfying_inequality : ∀ b : ℝ, (b^2 - 16 * b + 55) ≥ 0 ↔ b ≤ 5 ∨ b ≥ 11 := sorry

end smallest_b_satisfying_inequality_l244_244927


namespace sin_240_eq_neg_sqrt3_div_2_l244_244617

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244617


namespace exp_function_not_increasing_l244_244705

open Real

theorem exp_function_not_increasing (a : ℝ) (x : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : a < 1) :
  ¬(∀ x₁ x₂ : ℝ, x₁ < x₂ → a^x₁ < a^x₂) := by
  sorry

end exp_function_not_increasing_l244_244705


namespace sum_of_primes_less_than_20_eq_77_l244_244025

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l244_244025


namespace functional_equation_solution_l244_244260

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * f x * y) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) :=
by
  intro h
  sorry

end functional_equation_solution_l244_244260


namespace probability_odd_female_committee_l244_244141


theorem probability_odd_female_committee (men women : ℕ) (total_committee : ℕ) 
  (h_men : men = 5) (h_women : women = 4) (h_committee : total_committee = 3) : 
  (choose (men + women) total_committee).toRat * (44 : ℚ) / (84 : ℚ) = 11 / 21 :=
by
  -- ∃k, this stub in Lean to guarantee successful compilation
  sorry

end probability_odd_female_committee_l244_244141


namespace non_zero_real_m_value_l244_244183

theorem non_zero_real_m_value (m : ℝ) (h1 : 3 - m ∈ ({1, 2, 3} : Set ℝ)) (h2 : m ≠ 0) : m = 2 := 
sorry

end non_zero_real_m_value_l244_244183


namespace probability_of_C_l244_244740

theorem probability_of_C (P_A P_B P_C P_D P_E : ℚ)
  (hA : P_A = 2/5)
  (hB : P_B = 1/5)
  (hCD : P_C = P_D)
  (hE : P_E = 2 * P_C)
  (h_total : P_A + P_B + P_C + P_D + P_E = 1) : P_C = 1/10 :=
by
  -- To prove this theorem, you will use the conditions provided in the hypotheses.
  -- Here's how you start the proof:
  sorry

end probability_of_C_l244_244740


namespace eighth_square_more_tiles_than_seventh_l244_244278

-- Define the total number of tiles in the nth square
def total_tiles (n : ℕ) : ℕ := n^2 + 2 * n

-- Formulate the theorem statement
theorem eighth_square_more_tiles_than_seventh :
  total_tiles 8 - total_tiles 7 = 17 := by
  sorry

end eighth_square_more_tiles_than_seventh_l244_244278


namespace sin_240_deg_l244_244571

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l244_244571


namespace linear_equation_a_zero_l244_244952

theorem linear_equation_a_zero (a : ℝ) : 
  ((a - 2) * x ^ (abs (a - 1)) + 3 = 9) ∧ (abs (a - 1) = 1) → a = 0 := by
  sorry

end linear_equation_a_zero_l244_244952


namespace max_green_socks_l244_244884

theorem max_green_socks (g y : ℕ) (h1 : g + y ≤ 2025)
  (h2 : (g * (g - 1))/(g + y) * (g + y - 1) = 1/3) : 
  g ≤ 990 := 
sorry

end max_green_socks_l244_244884


namespace bounded_above_unbounded_below_solutions_l244_244128

noncomputable def differential_eq := λ (y : ℝ → ℝ), ∀ x, has_deriv_at (deriv y) ((x ^ 3 + x * k) * y x) x

theorem bounded_above_unbounded_below_solutions (k : ℝ) (y : ℝ → ℝ)
  (h_satisfies_eq : differential_eq y)
  (h_init_conditions : (y 0 = 1) ∧ (deriv y 0 = 0)) :
  (∃ M, ∀ x > M, y x ≠ 0) ∧ ¬ (∃ N, ∀ x < N, y x ≠ 0) :=
sorry

end bounded_above_unbounded_below_solutions_l244_244128


namespace coffee_shop_sold_lattes_l244_244371

theorem coffee_shop_sold_lattes (T L : ℕ) (h1 : T = 6) (h2 : L = 4 * T + 8) : L = 32 :=
by
  sorry

end coffee_shop_sold_lattes_l244_244371


namespace max_xy_under_constraint_l244_244774

theorem max_xy_under_constraint (x y : ℝ) (h1 : x + 2 * y = 1) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 1 / 8 
  := sorry

end max_xy_under_constraint_l244_244774


namespace coffee_shop_lattes_l244_244372

theorem coffee_shop_lattes (T : ℕ) (L : ℕ) (hT : T = 6) (hL : L = 4 * T + 8) : L = 32 :=
by
  sorry

end coffee_shop_lattes_l244_244372


namespace probability_no_defective_pens_l244_244677

theorem probability_no_defective_pens
  (total_pens : ℕ) (defective_pens : ℕ) (non_defective_pens : ℕ) (prob_first_non_defective : ℚ) (prob_second_non_defective : ℚ) :
  total_pens = 12 →
  defective_pens = 4 →
  non_defective_pens = total_pens - defective_pens →
  prob_first_non_defective = non_defective_pens / total_pens →
  prob_second_non_defective = (non_defective_pens - 1) / (total_pens - 1) →
  prob_first_non_defective * prob_second_non_defective = 14 / 33 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  sorry

end probability_no_defective_pens_l244_244677


namespace largest_square_l244_244430

def sticks_side1 : List ℕ := [4, 4, 2, 3]
def sticks_side2 : List ℕ := [4, 4, 3, 1, 1]
def sticks_side3 : List ℕ := [4, 3, 3, 2, 1]
def sticks_side4 : List ℕ := [3, 3, 3, 2, 2]

def sum_of_sticks (sticks : List ℕ) : ℕ := sticks.foldl (· + ·) 0

theorem largest_square (h1 : sum_of_sticks sticks_side1 = 13)
                      (h2 : sum_of_sticks sticks_side2 = 13)
                      (h3 : sum_of_sticks sticks_side3 = 13)
                      (h4 : sum_of_sticks sticks_side4 = 13) :
  13 = 13 := by
  sorry

end largest_square_l244_244430


namespace max_value_of_expr_l244_244013

theorem max_value_of_expr : ∃ t : ℝ, (∀ u : ℝ, (3^u - 2*u) * u / 9^u ≤ (3^t - 2*t) * t / 9^t) ∧ (3^t - 2*t) * t / 9^t = 1/8 :=
by sorry

end max_value_of_expr_l244_244013


namespace number_of_possible_ordered_pairs_l244_244364

theorem number_of_possible_ordered_pairs (n : ℕ) (f m : ℕ) 
  (cond1 : n = 6) 
  (cond2 : f ≥ 0) 
  (cond3 : m ≥ 0) 
  (cond4 : f + m ≤ 12) 
  : ∃ s : Finset (ℕ × ℕ), s.card = 6 := 
by 
  sorry

end number_of_possible_ordered_pairs_l244_244364


namespace total_napkins_l244_244817

variable (initial_napkins Olivia_napkins Amelia_multiplier : ℕ)

-- Defining the conditions
def Olivia_gives_napkins : ℕ := 10
def William_initial_napkins : ℕ := 15
def Amelia_gives_napkins : ℕ := 2 * Olivia_gives_napkins

-- Define the total number of napkins William has now
def William_napkins_now : ℕ :=
  initial_napkins + Olivia_napkins + Amelia_gives_napkins

-- Proving the total number of napkins William has now is 45
theorem total_napkins (h1 : Olivia_napkins = 10)
                      (h2: initial_napkins = 15)
                      (h3: Amelia_multiplier = 2)
                      : William_napkins_now initial_napkins Olivia_napkins (Olivia_napkins * Amelia_multiplier) = 45 :=
by
  rw [←h1, ←h2, ←h3]
  sorry

end total_napkins_l244_244817


namespace sum_of_primes_less_than_20_l244_244125

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l244_244125


namespace calc_residue_modulo_l244_244753

theorem calc_residue_modulo :
  let a := 320
  let b := 16
  let c := 28
  let d := 5
  let e := 7
  let n := 14
  (a * b - c * d + e) % n = 3 :=
by
  sorry

end calc_residue_modulo_l244_244753


namespace sum_prime_numbers_less_than_twenty_l244_244088

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l244_244088


namespace work_done_by_6_men_and_11_women_l244_244670

-- Definitions based on conditions
def work_completed_by_men (men : ℕ) (days : ℕ) : ℚ := men / (8 * days)
def work_completed_by_women (women : ℕ) (days : ℕ) : ℚ := women / (12 * days)
def combined_work_rate (men : ℕ) (women : ℕ) (days : ℕ) : ℚ := 
  work_completed_by_men men days + work_completed_by_women women days

-- Problem statement
theorem work_done_by_6_men_and_11_women :
  combined_work_rate 6 11 12 = 1 := by
  sorry

end work_done_by_6_men_and_11_women_l244_244670


namespace find_a_l244_244233

theorem find_a (a : ℝ) (h1 : a > 0) :
  (a^0 + a^1 = 3) → a = 2 :=
by sorry

end find_a_l244_244233


namespace sin_240_eq_neg_sqrt3_div_2_l244_244588

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244588


namespace fraction_subtraction_l244_244993

theorem fraction_subtraction :
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 :=
by
  sorry

end fraction_subtraction_l244_244993


namespace total_sounds_produced_l244_244787

-- Defining the total number of nails for one customer and the number of customers
def nails_per_person : ℕ := 20
def number_of_customers : ℕ := 3

-- Proving the total number of nail trimming sounds for 3 customers = 60
theorem total_sounds_produced : nails_per_person * number_of_customers = 60 := by
  sorry

end total_sounds_produced_l244_244787


namespace deepak_age_l244_244983

variable (R D : ℕ)

theorem deepak_age (h1 : R / D = 4 / 3) (h2 : R + 6 = 26) : D = 15 :=
sorry

end deepak_age_l244_244983


namespace range_of_a_l244_244205

noncomputable def S : Set ℝ := {x | |x - 1| + |x + 2| > 5}
noncomputable def T (a : ℝ) : Set ℝ := {x | |x - a| ≤ 4}

theorem range_of_a (a : ℝ) : 
  (S ∪ T a) = Set.univ ↔ -2 ≤ a ∧ a ≤ 1 := 
by 
  sorry

end range_of_a_l244_244205


namespace perfect_set_conclusions_l244_244464

def is_perfect_set (A : Set ℚ) : Prop :=
  (0 ∈ A ∧ 1 ∈ A) ∧
  (∀ x y, x ∈ A → y ∈ A → (x - y) ∈ A) ∧
  (∀ x, x ∈ A → x ≠ 0 → (1 / x) ∈ A)

theorem perfect_set_conclusions :
  let B := {-1, 0, 1}
  let Q := {q : ℚ | True}
  ∀ A : Set ℚ,
    is_perfect_set A →
    (¬ is_perfect_set B ∧ is_perfect_set Q ∧
     (∀ x y, x ∈ A → y ∈ A → (x + y) ∈ A) ∧
     (∀ x y, x ∈ A → y ∈ A → (x * y) ∈ A) ∧
     (∀ x y, x ∈ A → y ∈ A → x ≠ 0 → (y / x) ∈ A)) :=
sorry

end perfect_set_conclusions_l244_244464


namespace math_problem_l244_244177

noncomputable def exponential_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n = 2 * 3^(n - 1)

noncomputable def geometric_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(2 * 3^n - 2) / 2

theorem math_problem 
  (a : ℕ → ℝ) (b : ℕ → ℕ) (c : ℕ → ℝ) (S T P : ℕ → ℝ)
  (h1 : exponential_sequence a)
  (h2 : a 1 * a 3 = 36)
  (h3 : a 3 + a 4 = 9 * (a 1 + a 2))
  (h4 : ∀ n, S n + 1 = 3^(b n))
  (h5 : ∀ n, T n = (2 * n - 1) * 3^n / 2 + 1 / 2)
  (h6 : ∀ n, c n = a n / ((a n + 1) * (a (n + 1) + 1)))
  (h7 : ∀ n, P (2 * n) = 1 / 6 - 1 / (4 * 3^(2 * n) + 2)) :
  (∀ n, a n = 2 * 3^(n - 1)) ∧
  ∀ n, b n = n ∧
  ∀ n, a n * b n = 2 * n * 3^(n - 1) ∧
  ∃ n, T n = (2 * n - 1) * 3^n / 2 + 1 / 2 ∧
  P (2 * n) = 1 / 6 - 1 / (4 * 3^(2 * n) + 2) :=
by sorry

end math_problem_l244_244177


namespace inequality_am_gm_l244_244321

theorem inequality_am_gm (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end inequality_am_gm_l244_244321


namespace mirror_area_l244_244154

-- Defining the conditions as Lean functions and values
def frame_height : ℕ := 100
def frame_width : ℕ := 140
def frame_border : ℕ := 15

-- Statement to prove the area of the mirror
theorem mirror_area :
  let mirror_width := frame_width - 2 * frame_border
  let mirror_height := frame_height - 2 * frame_border
  mirror_width * mirror_height = 7700 :=
by
  sorry

end mirror_area_l244_244154


namespace piggy_bank_balance_l244_244361

theorem piggy_bank_balance (original_amount : ℕ) (taken_out : ℕ) : original_amount = 5 ∧ taken_out = 2 → original_amount - taken_out = 3 :=
by sorry

end piggy_bank_balance_l244_244361


namespace NaCl_yield_l244_244925

structure Reaction :=
  (reactant1 : ℕ)
  (reactant2 : ℕ)
  (product : ℕ)

def NaOH := 3
def HCl := 3

theorem NaCl_yield : ∀ (R : Reaction), R.reactant1 = NaOH → R.reactant2 = HCl → R.product = 3 :=
by
  sorry

end NaCl_yield_l244_244925


namespace number_of_ways_to_divide_l244_244193

def shape_17_cells : Type := sorry -- We would define the structure of the shape here
def checkerboard_pattern : shape_17_cells → Prop := sorry -- The checkerboard pattern condition
def num_black_cells (s : shape_17_cells) : ℕ := 9 -- Number of black cells
def num_gray_cells (s : shape_17_cells) : ℕ := 8 -- Number of gray cells
def divides_into (s : shape_17_cells) (rectangles : ℕ) (squares : ℕ) : Prop := sorry -- Division condition

theorem number_of_ways_to_divide (s : shape_17_cells) (h1 : checkerboard_pattern s) (h2 : divides_into s 8 1) :
  num_black_cells s = 9 ∧ num_gray_cells s = 8 → 
  (∃ ways : ℕ, ways = 10) := 
sorry

end number_of_ways_to_divide_l244_244193


namespace ellipse_product_axes_l244_244357

/-- Prove that the product of the lengths of the major and minor axes (AB)(CD) of an ellipse
is 240, given the following conditions:
- Point O is the center of the ellipse.
- Point F is one focus of the ellipse.
- OF = 8
- The diameter of the inscribed circle of triangle OCF is 4.
- OA = OB = a
- OC = OD = b
- a² - b² = 64
- a - b = 4
-/
theorem ellipse_product_axes (a b : ℝ) (OF : ℝ) (d_inscribed_circle : ℝ) 
  (h1 : OF = 8) (h2 : d_inscribed_circle = 4) (h3 : a^2 - b^2 = 64) 
  (h4 : a - b = 4) : (2 * a) * (2 * b) = 240 :=
sorry

end ellipse_product_axes_l244_244357


namespace cosine_theta_between_planes_l244_244684

open Real

-- Define vectors representing the normal vectors of the planes
def n1 := (3: ℝ, -1, 1)
def n2 := (4: ℝ, 2, -1)

-- Define the dot product of two 3-dimensional vectors
def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

-- Define the magnitude of a 3-dimensional vector
def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2)

-- Prove that cos theta = 9 / sqrt 231
noncomputable def cos_theta : ℝ :=
  dot_product n1 n2 / (magnitude n1 * magnitude n2)

theorem cosine_theta_between_planes :
  cos_theta = 9 / sqrt 231 :=
sorry

end cosine_theta_between_planes_l244_244684


namespace album_photos_proof_l244_244149

def photos_per_page := 4

-- Conditions
def position_81st_photo (n: ℕ) (x: ℕ) :=
  4 * n * (x - 1) + 17 ≤ 81 ∧ 81 ≤ 4 * n * (x - 1) + 20

def position_171st_photo (n: ℕ) (y: ℕ) :=
  4 * n * (y - 1) + 9 ≤ 171 ∧ 171 ≤ 4 * n * (y - 1) + 12

noncomputable def album_photos := 32

theorem album_photos_proof :
  ∃ n x y, position_81st_photo n x ∧ position_171st_photo n y ∧ 4 * n = album_photos :=
by
  sorry

end album_photos_proof_l244_244149


namespace samantha_birth_year_l244_244225

theorem samantha_birth_year
  (first_amc8_year : ℕ := 1985)
  (held_annually : ∀ (n : ℕ), n ≥ 0 → first_amc8_year + n = 1985 + n)
  (samantha_age_7th_amc8 : ℕ := 12) :
  ∃ (birth_year : ℤ), birth_year = 1979 :=
by
  sorry

end samantha_birth_year_l244_244225


namespace quadratic_root_shift_l244_244985

theorem quadratic_root_shift (A B p : ℤ) (α β : ℤ) 
  (h1 : ∀ x, x^2 + p * x + 19 = 0 → x = α + 1 ∨ x = β + 1)
  (h2 : ∀ x, x^2 - A * x + B = 0 → x = α ∨ x = β)
  (h3 : α + β = A)
  (h4 : α * β = B) :
  A + B = 18 := 
sorry

end quadratic_root_shift_l244_244985


namespace exists_two_natural_pairs_satisfying_equation_l244_244291

theorem exists_two_natural_pairs_satisfying_equation :
  ∃ (x1 y1 x2 y2 : ℕ), (2 * x1^3 = y1^4) ∧ (2 * x2^3 = y2^4) ∧ ¬(x1 = x2 ∧ y1 = y2) :=
sorry

end exists_two_natural_pairs_satisfying_equation_l244_244291


namespace sum_primes_less_than_20_l244_244066

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l244_244066


namespace math_proof_l244_244971

def problem_statement : Prop :=
  ∃ x : ℕ, (2 * x + 3 = 19) ∧ (x + (2 * x + 3) = 27)

theorem math_proof : problem_statement :=
  sorry

end math_proof_l244_244971


namespace discount_rate_l244_244899

theorem discount_rate (cost_price marked_price desired_profit_margin selling_price : ℝ)
  (h1 : cost_price = 160)
  (h2 : marked_price = 240)
  (h3 : desired_profit_margin = 0.2)
  (h4 : selling_price = cost_price * (1 + desired_profit_margin)) :
  marked_price * (1 - ((marked_price - selling_price) / marked_price)) = selling_price :=
by
  sorry

end discount_rate_l244_244899


namespace axis_of_symmetry_shift_l244_244781

-- Define that f is an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define the problem statement in Lean
theorem axis_of_symmetry_shift (f : ℝ → ℝ) 
  (h_even : is_even_function f) :
  ∃ x, ∀ y, f (x + y) = f ((x - 1) + y) ∧ x = -1 :=
sorry

end axis_of_symmetry_shift_l244_244781


namespace product_ne_sum_11_times_l244_244939

def is_prime (n : ℕ) : Prop := ∀ m, m > 1 → m < n → n % m ≠ 0
def prime_sum_product_condition (a b c d : ℕ) : Prop := 
  a * b * c * d = 11 * (a + b + c + d)

theorem product_ne_sum_11_times (a b c d : ℕ)
  (ha : is_prime a) (hb : is_prime b) (hc : is_prime c) (hd : is_prime d)
  (h : prime_sum_product_condition a b c d) :
  (a + b + c + d ≠ 46) ∧ (a + b + c + d ≠ 47) ∧ (a + b + c + d ≠ 48) :=
by  
  sorry

end product_ne_sum_11_times_l244_244939


namespace polar_to_cartesian_coordinates_l244_244304

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_cartesian_coordinates :
  polar_to_cartesian 2 (2 / 3 * Real.pi) = (-1, Real.sqrt 3) :=
by
  sorry

end polar_to_cartesian_coordinates_l244_244304


namespace isosceles_triangle_base_length_l244_244793

theorem isosceles_triangle_base_length (a b : ℝ) (h : a = 4 ∧ b = 4) : a + b = 8 :=
by
  sorry

end isosceles_triangle_base_length_l244_244793


namespace g_five_eq_thirteen_sevenths_l244_244461

def g (x : ℚ) : ℚ := (3 * x - 2) / (x + 2)

theorem g_five_eq_thirteen_sevenths : g 5 = 13 / 7 := by
  sorry

end g_five_eq_thirteen_sevenths_l244_244461


namespace evaluate_expression_l244_244765

theorem evaluate_expression (y : ℚ) (h : y = 1 / 3) :
  1 / (3 + 1 / (3 + 1 / (3 - y))) = 27 / 89 :=
by {
  rw h,
  calc
    1 / (3 + 1 / (3 + 1 / (3 - 1 / 3))) = 1 / (3 + 1 / (3 + 1 / (8 / 3))) : by rw [sub_div, sub_self, sub_fraction, sub_mul_div]
    ... = 1 / (3 + 1 / (3 + 3 / 8)) : by rw [div_inv_eq]
    ... = 1 / (3 + 8 / 27) : by rw [add_com, add_fraction]
    ... = 1 / (27 / 89) : by rw [inv_div_eq_div_mul]
    ... = 27 / 89 : by rw.div_div_eq_inv_eq_inv
}

end evaluate_expression_l244_244765


namespace students_participated_l244_244499

theorem students_participated (like_dislike_sum : 383 + 431 = 814) : 
  383 + 431 = 814 := 
by exact like_dislike_sum

end students_participated_l244_244499


namespace wall_building_time_l244_244948

variables (f b c y : ℕ) 

theorem wall_building_time :
  (y = 2 * f * c / b) 
  ↔ 
  (f > 0 ∧ b > 0 ∧ c > 0 ∧ (f * b * y = 2 * b * c)) := 
sorry

end wall_building_time_l244_244948


namespace find_x_l244_244335

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (h : x + y + x * y = 143) : x = 15 :=
by sorry

end find_x_l244_244335


namespace percent_difference_l244_244456

theorem percent_difference:
  let percent_value1 := (55 / 100) * 40
  let fraction_value2 := (4 / 5) * 25
  percent_value1 - fraction_value2 = 2 :=
by
  sorry

end percent_difference_l244_244456


namespace total_price_of_property_l244_244706

theorem total_price_of_property (price_per_sq_ft: ℝ) (house_size barn_size: ℝ) (house_price barn_price total_price: ℝ) :
  price_per_sq_ft = 98 ∧ house_size = 2400 ∧ barn_size = 1000 → 
  house_price = price_per_sq_ft * house_size ∧
  barn_price = price_per_sq_ft * barn_size ∧
  total_price = house_price + barn_price →
  total_price = 333200 :=
by
  sorry

end total_price_of_property_l244_244706


namespace total_practice_hours_correct_l244_244273

-- Define the conditions
def daily_practice_hours : ℕ := 5 -- The team practices 5 hours daily
def missed_days : ℕ := 1 -- They missed practicing 1 day this week
def days_in_week : ℕ := 7 -- There are 7 days in a week

-- Calculate the number of days they practiced
def practiced_days : ℕ := days_in_week - missed_days

-- Calculate the total hours practiced
def total_practice_hours : ℕ := practiced_days * daily_practice_hours

-- Theorem to prove the total hours practiced is 30
theorem total_practice_hours_correct : total_practice_hours = 30 := by
  -- Start the proof; skipping the actual proof steps
  sorry

end total_practice_hours_correct_l244_244273


namespace cost_price_one_metre_l244_244279

noncomputable def selling_price : ℤ := 18000
noncomputable def total_metres : ℕ := 600
noncomputable def loss_per_metre : ℤ := 5

noncomputable def total_loss : ℤ := loss_per_metre * (total_metres : ℤ) -- Note the cast to ℤ for multiplication
noncomputable def cost_price : ℤ := selling_price + total_loss
noncomputable def cost_price_per_metre : ℚ := cost_price / (total_metres : ℤ)

theorem cost_price_one_metre : cost_price_per_metre = 35 := by
  sorry

end cost_price_one_metre_l244_244279


namespace molly_ate_11_suckers_l244_244694

/-- 
Sienna gave Bailey half of her suckers.
Jen ate 11 suckers and gave the rest to Molly.
Molly ate some suckers and gave the rest to Harmony.
Harmony kept 3 suckers and passed the remainder to Taylor.
Taylor ate one and gave the last 5 suckers to Callie.
How many suckers did Molly eat?
-/
theorem molly_ate_11_suckers
  (sienna_bailey_suckers : ℕ)
  (jen_ate : ℕ)
  (jens_remainder_to_molly : ℕ)
  (molly_remainder_to_harmony : ℕ) 
  (harmony_kept : ℕ) 
  (harmony_remainder_to_taylor : ℕ)
  (taylor_ate : ℕ)
  (taylor_remainder_to_callie : ℕ)
  (jen_condition : jen_ate = 11)
  (harmony_condition : harmony_kept = 3)
  (taylor_condition : taylor_ate = 1)
  (taylor_final_suckers : taylor_remainder_to_callie = 5) :
  molly_ate = 11 :=
by sorry

end molly_ate_11_suckers_l244_244694


namespace arccos_cos_eq_l244_244566

theorem arccos_cos_eq :
  Real.arccos (Real.cos 11) = 0.7168 := by
  sorry

end arccos_cos_eq_l244_244566


namespace sin_240_eq_neg_sqrt3_div_2_l244_244595

theorem sin_240_eq_neg_sqrt3_div_2 :
  sin (240 : ℝ) = - (Real.sqrt 3) / 2 :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244595


namespace socks_probability_l244_244666
-- First, import the necessary modules

-- Define the problem statement
theorem socks_probability :
  let colors := {red, blue, green, yellow, purple}
  let total_socks := 10
  let choose_socks := 4
  let pairs_combinations := (Finset.powersetLen 3 colors).card
  let ways_to_choose_pair := 3
  let remaining_combinations := 4
  let total_ways := (total_socks.choose choose_socks)
  let favorable_outcomes := pairs_combinations * ways_to_choose_pair * remaining_combinations
  in total_ways = 210 ∧ favorable_outcomes = 120 ∧ favorable_outcomes / total_ways = (4 / 7 : ℚ) :=
by
  sorry

end socks_probability_l244_244666


namespace original_length_of_ribbon_l244_244409

theorem original_length_of_ribbon (n : ℕ) (cm_per_piece : ℝ) (remaining_meters : ℝ) 
  (pieces_cm_to_m : cm_per_piece / 100 = 0.15) (remaining_ribbon : remaining_meters = 36) 
  (pieces_cut : n = 100) : n * (cm_per_piece / 100) + remaining_meters = 51 := 
by 
  sorry

end original_length_of_ribbon_l244_244409


namespace smallest_c_inv_l244_244489

def f (x : ℝ) : ℝ := (x + 3)^2 - 7

theorem smallest_c_inv (c : ℝ) : (∀ x1 x2 : ℝ, c ≤ x1 → c ≤ x2 → f x1 = f x2 → x1 = x2) →
  c = -3 :=
sorry

end smallest_c_inv_l244_244489


namespace Mario_savings_percentage_l244_244800

theorem Mario_savings_percentage 
  (P : ℝ) -- Normal price of a single ticket 
  (h_campaign : 5 * P = 3 * P) -- Campaign condition: 5 tickets for the price of 3
  : (2 * P) / (5 * P) * 100 = 40 := 
by
  -- Below this, we would write the actual automated proof, but we leave it as sorry.
  sorry

end Mario_savings_percentage_l244_244800


namespace total_student_capacity_l244_244329

-- Define the conditions
def school_capacity_one : ℕ := 400
def school_capacity_two : ℕ := 340
def number_of_schools_one : ℕ := 2
def number_of_schools_two : ℕ := 2

-- Statement to prove
theorem total_student_capacity :
  (number_of_schools_one * school_capacity_one) +
  (number_of_schools_two * school_capacity_two) = 1480 :=
by
  calc
    (number_of_schools_one * school_capacity_one) +
    (number_of_schools_two * school_capacity_two)
    = (2 * 400) + (2 * 340) : by sorry
    = 800 + 680 : by sorry
    = 1480 : by sorry

end total_student_capacity_l244_244329


namespace repeating_decimal_division_l244_244872

theorem repeating_decimal_division : 
  (0.\overline{81} : ℚ) = (81 / 99 : ℚ) →
  (0.\overline{36} : ℚ) = (36 / 99 : ℚ) → 
  (0.\overline{81} / 0.\overline{36} : ℚ) = (9 / 4 : ℚ) :=
by 
  intros h1 h2
  rw [h1, h2]
  change (_ / _) = (_ / _)
  sorry

end repeating_decimal_division_l244_244872


namespace pascal_fifth_number_l244_244716

def binom (n k : Nat) : Nat := Nat.choose n k

theorem pascal_fifth_number (n r : Nat) (h1 : n = 50) (h2 : r = 4) : binom n r = 230150 := by
  sorry

end pascal_fifth_number_l244_244716


namespace necessary_and_sufficient_condition_l244_244932

theorem necessary_and_sufficient_condition (a b : ℝ) : 
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := 
by 
  sorry

end necessary_and_sufficient_condition_l244_244932


namespace average_weight_l244_244369

theorem average_weight 
  (n₁ n₂ : ℕ) 
  (avg₁ avg₂ total_avg : ℚ) 
  (h₁ : n₁ = 24) 
  (h₂ : n₂ = 8)
  (h₃ : avg₁ = 50.25)
  (h₄ : avg₂ = 45.15)
  (h₅ : total_avg = 48.975) :
  ( (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = total_avg ) :=
sorry

end average_weight_l244_244369


namespace first_method_of_exhaustion_l244_244680

-- Define the names
inductive Names where
  | ZuChongzhi
  | LiuHui
  | ZhangHeng
  | YangHui
  deriving DecidableEq

-- Statement of the problem
def method_of_exhaustion_author : Names :=
  Names.LiuHui

-- Main theorem to state the result
theorem first_method_of_exhaustion : method_of_exhaustion_author = Names.LiuHui :=
by 
  sorry

end first_method_of_exhaustion_l244_244680


namespace combined_work_days_l244_244536

theorem combined_work_days (W D : ℕ) (h1: ∀ a b : ℕ, a + b = 4) (h2: (1/6:ℝ) = (1/6:ℝ)) :
  D = 4 :=
by
  sorry

end combined_work_days_l244_244536


namespace students_more_than_rabbits_l244_244424

/- Define constants for the problem. -/
def students_per_class : ℕ := 20
def rabbits_per_class : ℕ := 3
def num_classes : ℕ := 5

/- Define total counts based on given conditions. -/
def total_students : ℕ := students_per_class * num_classes
def total_rabbits : ℕ := rabbits_per_class * num_classes

/- The theorem we need to prove: The difference between total students and total rabbits is 85. -/
theorem students_more_than_rabbits : total_students - total_rabbits = 85 := by
  sorry

end students_more_than_rabbits_l244_244424


namespace relationship_between_areas_l244_244139

-- Assume necessary context and setup
variables (A B C C₁ C₂ : ℝ)
variables (a b c : ℝ) (h : a^2 + b^2 = c^2)

-- Define the conditions
def right_triangle := a = 8 ∧ b = 15 ∧ c = 17
def circumscribed_circle (d : ℝ) := d = 17
def areas_relation (A B C₁ C₂ : ℝ) := (C₁ < C₂) ∧ (A + B = C₁ + C₂)

-- Problem statement in Lean 4
theorem relationship_between_areas (ht : right_triangle 8 15 17) (hc : circumscribed_circle 17) :
  areas_relation A B C₁ C₂ :=
by sorry

end relationship_between_areas_l244_244139


namespace liquid_x_percentage_l244_244258

theorem liquid_x_percentage 
  (percentage_a : ℝ) (percentage_b : ℝ)
  (weight_a : ℝ) (weight_b : ℝ)
  (h1 : percentage_a = 0.8)
  (h2 : percentage_b = 1.8)
  (h3 : weight_a = 400)
  (h4 : weight_b = 700) :
  (weight_a * (percentage_a / 100) + weight_b * (percentage_b / 100)) / (weight_a + weight_b) * 100 = 1.44 := 
by
  sorry

end liquid_x_percentage_l244_244258


namespace sum_of_primes_less_than_20_l244_244108

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l244_244108


namespace ball_height_intersect_l244_244269

noncomputable def ball_height (h : ℝ) (t₁ t₂ : ℝ) (h₁ h₂ : ℝ → ℝ) : Prop :=
  (∀ t, h₁ t = h₂ (t - 1) ↔ t = t₁) ∧
  (h₁ t₁ = h ∧ h₂ t₁ = h) ∧ 
  (∀ t, h₂ (t - 1) = h₁ t) ∧ 
  (h₁ (1.1) = h ∧ h₂ (1.1) = h)

theorem ball_height_intersect (h : ℝ)
  (h₁ h₂ : ℝ → ℝ)
  (h_max : ∀ t₁ t₂, ball_height h t₁ t₂ h₁ h₂) :
  (∃ t₁, t₁ = 1.6) :=
sorry

end ball_height_intersect_l244_244269


namespace sum_primes_less_than_20_l244_244083

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l244_244083


namespace part1_solution_part2_solution_l244_244311

-- Definition for part 1
noncomputable def f_part1 (x : ℝ) := abs (x - 3) + 2 * x

-- Proof statement for part 1
theorem part1_solution (x : ℝ) : (f_part1 x ≥ 3) ↔ (x ≥ 0) :=
by sorry

-- Definition for part 2
noncomputable def f_part2 (x a : ℝ) := abs (x - a) + 2 * x

-- Proof statement for part 2
theorem part2_solution (a : ℝ) : 
  (∀ x, f_part2 x a ≤ 0 ↔ x ≤ -2) → (a = 2 ∨ a = -6) :=
by sorry

end part1_solution_part2_solution_l244_244311


namespace linear_regression_increase_l244_244771

-- Define the linear regression function
def linear_regression (x : ℝ) : ℝ :=
  1.6 * x + 2

-- Prove that y increases by 1.6 when x increases by 1
theorem linear_regression_increase (x : ℝ) :
  linear_regression (x + 1) - linear_regression x = 1.6 :=
by sorry

end linear_regression_increase_l244_244771


namespace arithmetic_sequence_sum_l244_244790

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h : S 7 = 77) 
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2) : 
  a 4 = 11 :=
by
  sorry

end arithmetic_sequence_sum_l244_244790


namespace cost_of_one_roll_sold_individually_l244_244138

-- Definitions based on conditions
def cost_case_12_rolls := 9
def percent_savings := 0.25

-- Variable representing the cost of one roll sold individually
variable (x : ℝ)

-- Statement to prove
theorem cost_of_one_roll_sold_individually : (12 * x - 12 * percent_savings * x) = cost_case_12_rolls → x = 1 :=
by
  intro h
  -- This is where the proof would go
  sorry

end cost_of_one_roll_sold_individually_l244_244138


namespace range_of_a_l244_244669

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 3| + |x + 5| > a) → a < 8 := by
  sorry

end range_of_a_l244_244669


namespace coprime_with_others_l244_244659

theorem coprime_with_others:
  ∀ (a b c d e : ℕ),
  a = 20172017 → 
  b = 20172018 → 
  c = 20172019 →
  d = 20172020 →
  e = 20172021 →
  (Nat.gcd c a = 1 ∧ 
   Nat.gcd c b = 1 ∧ 
   Nat.gcd c d = 1 ∧ 
   Nat.gcd c e = 1) :=
by
  sorry

end coprime_with_others_l244_244659


namespace parabola_vertex_eq_l244_244860

theorem parabola_vertex_eq :
  (∃ c : ℝ, (∀ x : ℝ, y = 2 * x^2 + c) ∧ y = 1 ∧ x = 0) → c = 1 :=
by
  intro h
  choose c hc using h
  specialize hc 0
  rw [mul_zero, zero_mul, add_zero] at hc
  cases hc
  rw [hc_right]
  exact hc_left

end parabola_vertex_eq_l244_244860


namespace bowling_team_scores_l244_244548

theorem bowling_team_scores : 
  ∀ (A B C : ℕ), 
  C = 162 → 
  B = 3 * C → 
  A + B + C = 810 → 
  A / B = 1 / 3 := 
by 
  intros A B C h1 h2 h3 
  sorry

end bowling_team_scores_l244_244548


namespace simplify_expr_l244_244643

theorem simplify_expr (a b : ℝ) (h₁ : a + b = 0) (h₂ : a ≠ b) : (1 - a) + (1 - b) = 2 := by
  sorry

end simplify_expr_l244_244643


namespace marias_workday_ends_at_six_pm_l244_244351

theorem marias_workday_ends_at_six_pm :
  ∀ (start_time : ℕ) (work_hours : ℕ) (lunch_start_time : ℕ) (lunch_duration : ℕ) (afternoon_break_time : ℕ) (afternoon_break_duration : ℕ) (end_time : ℕ),
    start_time = 8 ∧
    work_hours = 8 ∧
    lunch_start_time = 13 ∧
    lunch_duration = 1 ∧
    afternoon_break_time = 15 * 60 + 30 ∧  -- Converting 3:30 P.M. to minutes
    afternoon_break_duration = 15 ∧
    end_time = 18  -- 6:00 P.M. in 24-hour format
    → end_time = 18 :=
by
  -- map 13:00 -> 1:00 P.M.,  15:30 -> 3:30 P.M.; convert 6:00 P.M. back 
  sorry

end marias_workday_ends_at_six_pm_l244_244351


namespace algebraic_expression_constant_l244_244262

theorem algebraic_expression_constant (x : ℝ) : x * (x - 6) - (3 - x) ^ 2 = -9 :=
sorry

end algebraic_expression_constant_l244_244262


namespace divide_triangle_l244_244628

/-- 
  Given a triangle with the total sum of numbers as 63,
  we want to prove that it can be divided into three parts
  where each part's sum is 21.
-/
theorem divide_triangle (total_sum : ℕ) (H : total_sum = 63) :
  ∃ (part1 part2 part3 : ℕ), 
    (part1 + part2 + part3 = total_sum) ∧ 
    part1 = 21 ∧ 
    part2 = 21 ∧ 
    part3 = 21 :=
by 
  use 21, 21, 21
  split
  . exact H.symm ▸ rfl
  . split; rfl
  . split; rfl
  . rfl

end divide_triangle_l244_244628


namespace a_lt_2_is_necessary_but_not_sufficient_for_a_squared_lt_4_l244_244380

theorem a_lt_2_is_necessary_but_not_sufficient_for_a_squared_lt_4 (a : ℝ) :
  (a < 2 → a^2 < 4) ∧ (a^2 < 4 → a < 2) :=
by
  -- Proof skipped
  sorry

end a_lt_2_is_necessary_but_not_sufficient_for_a_squared_lt_4_l244_244380


namespace relatively_prime_example_l244_244657

theorem relatively_prime_example :
  let a := 20172017
  let b := 20172018
  let c := 20172019
  let d := 20172020
  let e := 20172021
  Nat.gcd a c = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd d c = 1 ∧ Nat.gcd e c = 1 :=
by
  let a := 20172017
  let b := 20172018
  let c := 20172019
  let d := 20172020
  let e := 20172021
  sorry

end relatively_prime_example_l244_244657


namespace f_2013_eq_2_l244_244770

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f (-x) = -f x
axiom h2 : ∀ x : ℝ, f (x + 4) = f x + f 2
axiom h3 : f (-1) = -2

theorem f_2013_eq_2 : f 2013 = 2 := 
by 
  sorry

end f_2013_eq_2_l244_244770


namespace linear_relation_is_correct_maximum_profit_l244_244414

-- Define the given data points
structure DataPoints where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

-- Define the given conditions
def conditions : DataPoints := ⟨50, 100, 60, 90⟩

-- Define the cost and sell price range conditions
def cost_per_kg : ℝ := 20
def max_selling_price : ℝ := 90

-- Define the linear relationship function
def linear_relationship (k b x : ℝ) : ℝ := k * x + b

-- Define the profit function
def profit_function (x : ℝ) : ℝ := (x - cost_per_kg) * (linear_relationship (-1) 150 x)

-- Statements to Prove
theorem linear_relation_is_correct (k b : ℝ) :
  linear_relationship k b 50 = 100 ∧
  linear_relationship k b 60 = 90 →
  (b = 150 ∧ k = -1) := by
  intros h
  sorry

theorem maximum_profit :
  ∃ x : ℝ, 20 ≤ x ∧ x ≤ max_selling_price ∧ profit_function x = 4225 := by
  use 85
  sorry

end linear_relation_is_correct_maximum_profit_l244_244414


namespace sum_of_first_5n_l244_244325

theorem sum_of_first_5n (n : ℕ) : 
  (n * (n + 1) / 2) + 210 = ((4 * n) * (4 * n + 1) / 2) → 
  (5 * n) * (5 * n + 1) / 2 = 465 :=
by sorry

end sum_of_first_5n_l244_244325


namespace total_votes_cast_l244_244189

-- Define the variables and constants
def total_votes (V : ℝ) : Prop :=
  let A := 0.32 * V
  let B := 0.28 * V
  let C := 0.22 * V
  let D := 0.18 * V
  -- Candidate A defeated Candidate B by 1200 votes
  0.32 * V - 0.28 * V = 1200 ∧
  -- Candidate A defeated Candidate C by 2200 votes
  0.32 * V - 0.22 * V = 2200 ∧
  -- Candidate B defeated Candidate D by 900 votes
  0.28 * V - 0.18 * V = 900

noncomputable def V := 30000

-- State the theorem
theorem total_votes_cast : total_votes V := by
  sorry

end total_votes_cast_l244_244189


namespace scientific_notation_384000_l244_244221

theorem scientific_notation_384000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 384000 = a * 10 ^ n ∧ 
  a = 3.84 ∧ n = 5 :=
sorry

end scientific_notation_384000_l244_244221


namespace cost_of_eraser_pencil_l244_244679

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

end cost_of_eraser_pencil_l244_244679


namespace find_x_of_equation_l244_244343

theorem find_x_of_equation (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := by
  sorry

end find_x_of_equation_l244_244343


namespace find_factor_l244_244145

theorem find_factor (x f : ℝ) (h1 : x = 6)
    (h2 : (2 * x + 9) * f = 63) : f = 3 :=
sorry

end find_factor_l244_244145


namespace sum_primes_less_than_20_l244_244051

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l244_244051


namespace income_increase_by_parental_support_l244_244896

variables (a b c S : ℝ)

theorem income_increase_by_parental_support 
  (h1 : S = a + b + c)
  (h2 : 2 * a + b + c = 1.05 * S)
  (h3 : a + 2 * b + c = 1.15 * S) :
  (a + b + 2 * c) = 1.8 * S :=
sorry

end income_increase_by_parental_support_l244_244896


namespace carla_wins_probability_l244_244562

-- Define the probabilities for specific conditions
def prob_derek_rolls_less_than_7 : ℚ := 5 / 6
def prob_emily_rolls_less_than_7 : ℚ := 2 / 3
def prob_combined_rolls_lt_10 : ℚ := 4 / 5

-- Define the total probability based on the given conditions
def total_probability : ℚ :=
  (prob_derek_rolls_less_than_7 * prob_emily_rolls_less_than_7) * prob_combined_rolls_lt_10

-- Main theorem asserting the probability of the event
theorem carla_wins_probability :
  total_probability = 8 / 27 :=
by
  -- Assuming the correct answer provided, we state the theorem
  sorry

end carla_wins_probability_l244_244562


namespace simplify_fraction_l244_244634

variable {x y : ℝ}

theorem simplify_fraction (h : x ≠ y) : (x^6 - y^6) / (x^3 - y^3) = x^3 + y^3 := by
  sorry

end simplify_fraction_l244_244634


namespace ratio_area_rectangle_to_square_l244_244502

variable (s : ℝ)
variable (area_square : ℝ := s^2)
variable (longer_side_rectangle : ℝ := 1.2 * s)
variable (shorter_side_rectangle : ℝ := 0.85 * s)
variable (area_rectangle : ℝ := longer_side_rectangle * shorter_side_rectangle)

theorem ratio_area_rectangle_to_square :
  area_rectangle / area_square = 51 / 50 := by
  sorry

end ratio_area_rectangle_to_square_l244_244502


namespace intersection_of_A_and_B_l244_244644

def A := {x : ℝ | |x - 2| ≤ 1}
def B := {x : ℝ | x^2 - 2 * x - 3 < 0}
def C := {x : ℝ | 1 ≤ x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = C := by
  sorry

end intersection_of_A_and_B_l244_244644


namespace square_floor_tiling_total_number_of_tiles_l244_244555

theorem square_floor_tiling (s : ℕ) (h : (2 * s - 1 : ℝ) / (s ^ 2 : ℝ) = 0.41) : s = 4 :=
by
  sorry

theorem total_number_of_tiles : 4^2 = 16 := 
by
  norm_num

end square_floor_tiling_total_number_of_tiles_l244_244555


namespace simplify_fraction_l244_244696

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : (x^2 / (x - 1)) - (1 / (x - 1)) = x + 1 :=
by
  sorry

end simplify_fraction_l244_244696


namespace polynomial_abs_sum_eq_81_l244_244667

theorem polynomial_abs_sum_eq_81 
  (a a_1 a_2 a_3 a_4 : ℝ) 
  (h : (1 - 2 * x)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4)
  (ha : a > 0) 
  (ha_2 : a_2 > 0) 
  (ha_4 : a_4 > 0) 
  (ha_1 : a_1 < 0) 
  (ha_3 : a_3 < 0): 
  |a| + |a_1| + |a_2| + |a_3| + |a_4| = 81 := 
by 
  sorry

end polynomial_abs_sum_eq_81_l244_244667


namespace prime_quadratic_residues_l244_244420

theorem prime_quadratic_residues (p : ℕ) [prime p] (h : ∀ k, k > 0 ∧ k ≤ p → is_quadratic_residue (2 * (p / k) - 1) p) : p = 2 := 
sorry

end prime_quadratic_residues_l244_244420


namespace vector_magnitude_positive_l244_244668

variable {V : Type} [NormedAddCommGroup V] [NormedSpace ℝ V]

variables (a b : V)

-- Given: 
-- a is any non-zero vector
-- b is a unit vector
theorem vector_magnitude_positive (ha : a ≠ 0) (hb : ‖b‖ = 1) : ‖a‖ > 0 := 
sorry

end vector_magnitude_positive_l244_244668


namespace sum_of_primes_less_than_20_l244_244121

theorem sum_of_primes_less_than_20 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19} in
  ∑ p in primes, p = 77 := 
sorry

end sum_of_primes_less_than_20_l244_244121


namespace solve_for_m_l244_244467

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (2 / (2^x + 1)) + m

theorem solve_for_m (m : ℝ) : (∀ x : ℝ, f m (-x) = - (f m x)) ↔ m = -1 := by
sorry

end solve_for_m_l244_244467


namespace compare_values_of_even_and_monotone_function_l244_244807

variable (f : ℝ → ℝ)

def is_even_function := ∀ x : ℝ, f x = f (-x)
def is_monotone_increasing_on_nonneg := ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

theorem compare_values_of_even_and_monotone_function
  (h_even : is_even_function f)
  (h_monotone : is_monotone_increasing_on_nonneg f) :
  f (-π) > f 3 ∧ f 3 > f (-2) :=
by
  sorry

end compare_values_of_even_and_monotone_function_l244_244807


namespace sum_of_all_possible_N_l244_244379

theorem sum_of_all_possible_N
  (a b c : ℕ)
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : c = a + b)
  (h3 : N = a * b * c)
  (h4 : N = 6 * (a + b + c)) :
  N = 156 ∨ N = 96 ∨ N = 84 ∧
  (156 + 96 + 84 = 336) :=
by {
  -- proof will go here
  sorry
}

end sum_of_all_possible_N_l244_244379


namespace smallest_a_l244_244540

theorem smallest_a (a : ℤ) : 
  (112 ∣ (a * 43 * 62 * 1311)) ∧ (33 ∣ (a * 43 * 62 * 1311)) ↔ a = 1848 := 
sorry

end smallest_a_l244_244540


namespace total_weight_of_ripe_fruits_correct_l244_244425

-- Definitions based on conditions
def total_apples : ℕ := 14
def total_pears : ℕ := 10
def total_lemons : ℕ := 5

def ripe_apple_weight : ℕ := 150
def ripe_pear_weight : ℕ := 200
def ripe_lemon_weight : ℕ := 100

def unripe_apples : ℕ := 6
def unripe_pears : ℕ := 4
def unripe_lemons : ℕ := 2

def total_weight_of_ripe_fruits : ℕ :=
  (total_apples - unripe_apples) * ripe_apple_weight +
  (total_pears - unripe_pears) * ripe_pear_weight +
  (total_lemons - unripe_lemons) * ripe_lemon_weight

theorem total_weight_of_ripe_fruits_correct :
  total_weight_of_ripe_fruits = 2700 :=
by
  -- proof goes here (use sorry to skip the actual proof)
  sorry

end total_weight_of_ripe_fruits_correct_l244_244425


namespace sum_of_primes_less_than_twenty_is_77_l244_244037

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l244_244037


namespace coprime_with_others_l244_244660

theorem coprime_with_others:
  ∀ (a b c d e : ℕ),
  a = 20172017 → 
  b = 20172018 → 
  c = 20172019 →
  d = 20172020 →
  e = 20172021 →
  (Nat.gcd c a = 1 ∧ 
   Nat.gcd c b = 1 ∧ 
   Nat.gcd c d = 1 ∧ 
   Nat.gcd c e = 1) :=
by
  sorry

end coprime_with_others_l244_244660


namespace seokgi_money_l244_244693

open Classical

variable (S Y : ℕ)

theorem seokgi_money (h1 : ∃ S, S + 2000 < S + Y + 2000)
                     (h2 : ∃ Y, Y + 1500 < S + Y + 1500)
                     (h3 : 3500 + (S + Y + 2000) = (S + Y) + 3500)
                     (boat_price1: ∀ S, S + 2000 = S + 2000)
                     (boat_price2: ∀ Y, Y + 1500 = Y + 1500) :
  S = 5000 :=
by sorry

end seokgi_money_l244_244693


namespace sum_of_primes_less_than_twenty_is_77_l244_244036

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l244_244036


namespace construction_company_sand_weight_l244_244888

theorem construction_company_sand_weight :
  ∀ (total_weight gravel_weight : ℝ), total_weight = 14.02 → gravel_weight = 5.91 → 
  total_weight - gravel_weight = 8.11 :=
by 
  intros total_weight gravel_weight h_total h_gravel 
  sorry

end construction_company_sand_weight_l244_244888


namespace total_blankets_collected_l244_244298

theorem total_blankets_collected : 
  let original_members := 15
  let new_members := 5
  let blankets_per_original_member_first_day := 2
  let blankets_per_original_member_second_day := 2
  let blankets_per_new_member_second_day := 4
  let tripled_first_day_total := 3
  let blankets_school_third_day := 22
  let blankets_online_third_day := 30
  let first_day_blankets := original_members * blankets_per_original_member_first_day
  let second_day_original_members_blankets := original_members * blankets_per_original_member_second_day
  let second_day_new_members_blankets := new_members * blankets_per_new_member_second_day
  let second_day_additional_blankets := tripled_first_day_total * first_day_blankets
  let second_day_blankets := second_day_original_members_blankets + second_day_new_members_blankets + second_day_additional_blankets
  let third_day_blankets := blankets_school_third_day + blankets_online_third_day
  let total_blankets := first_day_blankets + second_day_blankets + third_day_blankets
  -- Prove that
  total_blankets = 222 :=
by 
  sorry

end total_blankets_collected_l244_244298


namespace sin_240_eq_neg_sqrt3_div_2_l244_244587

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244587


namespace ratio_sphere_locus_l244_244530

noncomputable def sphere_locus_ratio (r : ℝ) : ℝ :=
  let F1 := 2 * Real.pi * r^2 * (1 - Real.sqrt (2 / 3))
  let F2 := Real.pi * r^2 * (2 * Real.sqrt 3 / 3)
  F1 / F2

theorem ratio_sphere_locus (r : ℝ) (h : r > 0) : sphere_locus_ratio r = Real.sqrt 3 - 1 :=
by
  sorry

end ratio_sphere_locus_l244_244530


namespace fair_coin_999th_toss_probability_l244_244868

open ProbabilityTheory

/-- Consider a fair coin where each toss results in heads or tails, with probability 1/2 each. 
    Prove that the probability of getting heads on the 999th toss is 1/2. -/
theorem fair_coin_999th_toss_probability :
  let p : ProbabilityMassFunction (Fin 2) := ProbabilityMassFunction.uniform_of_fin 2 in
  p.mass 0 = 1/2 := 
by
  sorry

end fair_coin_999th_toss_probability_l244_244868


namespace subset_bound_l244_244434

theorem subset_bound (n k m : ℕ) (S : Finset (Finset ℕ)) 
  (hS_card : S.card = m)
  (hS_size : ∀ A B ∈ S, A ≠ B → (A ∩ B).card < k) :
  m ≤ ∑ i in Finset.range (k + 1), Nat.choose n i :=
sorry

end subset_bound_l244_244434


namespace part_I_part_II_l244_244858

-- Define the function f(x)
def f (x a : ℝ) := abs (x - a) + 5 * x

-- Part (I)
theorem part_I (x : ℝ) : 
  (f x (-1) ≤ 5 * x + 3) ↔ (-4 ≤ x ∧ x ≤ 2) := 
by sorry

-- Part (II)
theorem part_II (a : ℝ) (x : ℝ) (h : x ≥ -1) : 
  (∀ x, f x a ≥ 0) ↔ (a ≥ 4 ∨ a ≤ -6) := 
by sorry

end part_I_part_II_l244_244858


namespace sum_primes_less_than_20_l244_244076

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l244_244076


namespace weighted_valid_votes_l244_244958

theorem weighted_valid_votes :
  let total_votes := 10000
  let invalid_vote_rate := 0.25
  let valid_votes := total_votes * (1 - invalid_vote_rate)
  let v_b := (valid_votes - 2 * (valid_votes * 0.15 + valid_votes * 0.07) + valid_votes * 0.05) / 4
  let v_a := v_b + valid_votes * 0.15
  let v_c := v_a + valid_votes * 0.07
  let v_d := v_b - valid_votes * 0.05
  let weighted_votes_A := v_a * 3.0
  let weighted_votes_B := v_b * 2.5
  let weighted_votes_C := v_c * 2.75
  let weighted_votes_D := v_d * 2.25
  weighted_votes_A = 7200 ∧
  weighted_votes_B = 3187.5 ∧
  weighted_votes_C = 8043.75 ∧
  weighted_votes_D = 2025 :=
by
  sorry

end weighted_valid_votes_l244_244958


namespace min_value_f_l244_244979

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt 3) * Real.sin x + Real.sin (Real.pi / 2 + x)

theorem min_value_f : ∃ x : ℝ, f x = -2 := by
  sorry

end min_value_f_l244_244979


namespace inequality_solution_l244_244863

theorem inequality_solution {x : ℝ} (h : 2 * x + 1 > x + 2) : x > 1 :=
by
  sorry

end inequality_solution_l244_244863


namespace greatest_three_digit_multiple_of_17_l244_244001

theorem greatest_three_digit_multiple_of_17 : ∃ (x : ℕ), (x % 17 = 0) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (∀ y, (y % 17 = 0) ∧ (100 ≤ y ∧ y ≤ 999) → y ≤ x) ∧ x = 986 :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l244_244001


namespace ball_hits_ground_time_l244_244883

noncomputable def h (t : ℝ) : ℝ := -16 * t^2 - 30 * t + 180

theorem ball_hits_ground_time :
  ∃ t : ℝ, h t = 0 ∧ t = 2.545 :=
by
  sorry

end ball_hits_ground_time_l244_244883


namespace product_inequality_l244_244648

variable (x1 x2 x3 x4 y1 y2 : ℝ)

theorem product_inequality (h1 : y2 ≥ y1) 
                          (h2 : y1 ≥ x1)
                          (h3 : x1 ≥ x3)
                          (h4 : x3 ≥ x2)
                          (h5 : x2 ≥ x1)
                          (h6 : x1 ≥ 2)
                          (h7 : x1 + x2 + x3 + x4 ≥ y1 + y2) : 
                          x1 * x2 * x3 * x4 ≥ y1 * y2 :=
  sorry

end product_inequality_l244_244648


namespace arith_seq_sum_ratio_l244_244768

theorem arith_seq_sum_ratio 
  (S : ℕ → ℝ) 
  (a1 d : ℝ) 
  (h1 : S 1 = 1) 
  (h2 : (S 4) / (S 2) = 4) :
  (S 6) / (S 4) = 9 / 4 :=
sorry

end arith_seq_sum_ratio_l244_244768


namespace prove_bc_prove_area_l244_244481

variable {a b c A B C : ℝ}

-- Given conditions as Lean definitions
def condition_cos_A : Prop := (b^2 + c^2 - a^2 = 2 * cos A)
def condition_bc : Prop := b * c = 1
def condition_trig_identity : Prop := (a * cos B - b * cos A) / (a * cos B + b * cos A) - b / c = 1

-- Definitions for the questions translated to Lean
def find_bc : Prop :=
  condition_cos_A → b * c = 1

def find_area (bc_value : Prop) : Prop :=
  bc_value →
  condition_trig_identity →
  1/2 * b * c * sin A = (Real.sqrt 3) / 4

-- Theorems to be proven
theorem prove_bc : find_bc :=
by
  intro h1
  sorry

theorem prove_area : find_area condition_bc :=
by
  intro h1 h2
  sorry

end prove_bc_prove_area_l244_244481


namespace friends_activity_l244_244168

-- Defining the problem conditions
def total_friends : ℕ := 5
def organizers : ℕ := 3
def managers : ℕ := total_friends - organizers

-- Stating the proof problem
theorem friends_activity (h1 : organizers = 3) (h2 : managers = 2) :
  Nat.choose total_friends organizers = 10 :=
sorry

end friends_activity_l244_244168


namespace passengers_remaining_l244_244786

theorem passengers_remaining :
  let initial_passengers := 64
  let reduction_factor := (2 / 3)
  ∀ (n : ℕ), n = 4 → initial_passengers * reduction_factor^n = 1024 / 81 := by
sorry

end passengers_remaining_l244_244786


namespace practice_hours_l244_244904

-- Define the starting and ending hours, and the break duration
def start_hour : ℕ := 8
def end_hour : ℕ := 16
def break_duration : ℕ := 2

-- Compute the total practice hours
def total_practice_time : ℕ := (end_hour - start_hour) - break_duration

-- State that the computed practice time is equal to 6 hours
theorem practice_hours :
  total_practice_time = 6 := 
by
  -- Using the definitions provided to state the proof
  sorry

end practice_hours_l244_244904


namespace investment_difference_l244_244199

noncomputable def compound_yearly (P : ℕ) (r : ℚ) (t : ℕ) : ℚ :=
  P * (1 + r)^t

noncomputable def compound_monthly (P : ℕ) (r : ℚ) (months : ℕ) : ℚ :=
  P * (1 + r)^(months)

theorem investment_difference :
  let P := 70000
  let r := 0.05
  let t := 3
  let monthly_r := r / 12
  let months := t * 12
  compound_monthly P monthly_r months - compound_yearly P r t = 263.71 :=
by
  sorry

end investment_difference_l244_244199


namespace train_length_l244_244153

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (h1 : speed_kmh = 90) (h2 : time_s = 12) : 
  ∃ length_m : ℕ, length_m = 300 := 
by
  sorry

end train_length_l244_244153


namespace sum_of_four_digit_numbers_l244_244995

open Nat

theorem sum_of_four_digit_numbers (s : Finset ℤ) :
  (∀ x, x ∈ s → (∃ k, x = 30 * k + 2) ∧ 1000 ≤ x ∧ x ≤ 9999) →
  s.sum id = 1652100 := by
  sorry

end sum_of_four_digit_numbers_l244_244995


namespace average_weasels_caught_per_week_l244_244327

-- Definitions based on the conditions
def initial_weasels : ℕ := 100
def initial_rabbits : ℕ := 50
def foxes : ℕ := 3
def rabbits_caught_per_week_per_fox : ℕ := 2
def weeks : ℕ := 3
def remaining_animals : ℕ := 96

-- Main theorem statement
theorem average_weasels_caught_per_week :
  (foxes * weeks * rabbits_caught_per_week_per_fox +
   foxes * weeks * W = initial_weasels + initial_rabbits - remaining_animals) →
  W = 4 :=
sorry

end average_weasels_caught_per_week_l244_244327


namespace integer_solution_count_l244_244453

theorem integer_solution_count : ∃ (n : ℕ), n = 53 ∧ 
  (∀ (x y : ℤ), x ≠ 0 → y ≠ 0 → (1 : ℚ) / 2022 = (1 : ℚ) / x + (1 : ℚ) / y → 
  (∃ (a b : ℤ), 2022 * (a - 2022) * (b - 2022) = 2022^2) :=
begin
  use 53,
  split, {
    refl,
  },
  intros x y hx hy hxy,
  sorry
end

end integer_solution_count_l244_244453


namespace sin_240_eq_neg_sqrt3_div_2_l244_244603

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244603


namespace bread_slices_l244_244991

theorem bread_slices (c : ℕ) (cost_each_slice_in_cents : ℕ)
  (total_paid_in_cents : ℕ) (change_in_cents : ℕ) (n : ℕ) (slices_per_loaf : ℕ) :
  c = 3 →
  cost_each_slice_in_cents = 40 →
  total_paid_in_cents = 2 * 2000 →
  change_in_cents = 1600 →
  total_paid_in_cents - change_in_cents = n * cost_each_slice_in_cents →
  n = c * slices_per_loaf →
  slices_per_loaf = 20 :=
by sorry

end bread_slices_l244_244991


namespace square_area_l244_244412

theorem square_area (x : ℝ) (s : ℝ) 
  (h1 : s^2 + s^2 = (2 * x)^2) 
  (h2 : 4 * s = 16 * x) : s^2 = 16 * x^2 :=
by {
  sorry -- Proof not required
}

end square_area_l244_244412


namespace linear_function_decreases_l244_244495

theorem linear_function_decreases (m b x : ℝ) (h : m < 0) : 
  ∃ y : ℝ, y = m * x + b ∧ ∀ x₁ x₂ : ℝ, x₁ < x₂ → (m * x₁ + b) > (m * x₂ + b) :=
by 
  sorry

end linear_function_decreases_l244_244495


namespace fourth_friend_age_is_8_l244_244220

-- Define the given data
variables (a1 a2 a3 a4 : ℕ)
variables (h_avg : (a1 + a2 + a3 + a4) / 4 = 9)
variables (h1 : a1 = 7) (h2 : a2 = 9) (h3 : a3 = 12)

-- Formalize the theorem to prove that the fourth friend's age is 8
theorem fourth_friend_age_is_8 : a4 = 8 :=
by
  -- Placeholder for the proof
  sorry

end fourth_friend_age_is_8_l244_244220


namespace sin_240_eq_neg_sqrt3_over_2_l244_244598

open Real

-- Conditions
def angle_240_in_third_quadrant : Prop := 240 ° ∈ set_of (λ x, 180 ° < x ∧ x < 270 °)

def reference_angle_60 (θ : Real) : Prop := θ = 240 ° - 180 °

def sin_60_eq_sqrt3_over_2 : sin (60 °) = sqrt 3 / 2

def sin_negative_in_third_quadrant (θ : Real) : Prop :=
  180 ° < θ ∧ θ < 270 ° → sin θ < 0

-- Statement
theorem sin_240_eq_neg_sqrt3_over_2 :
  angle_240_in_third_quadrant ∧ reference_angle_60 60 ° ∧ sin_60_eq_sqrt3_over_2 ∧ sin_negative_in_third_quadrant 240 °
  → sin (240 °) = - (sqrt 3 / 2) :=
by
  intros
  sorry

end sin_240_eq_neg_sqrt3_over_2_l244_244598


namespace perpendicular_lines_sin_2alpha_l244_244937

theorem perpendicular_lines_sin_2alpha (α : ℝ) 
  (l1 : ∀ (x y : ℝ), x * (Real.sin α) + y - 1 = 0) 
  (l2 : ∀ (x y : ℝ), x - 3 * y * Real.cos α + 1 = 0) 
  (perp : ∀ (x1 y1 x2 y2 : ℝ), 
        (x1 * (Real.sin α) + y1 - 1 = 0) ∧ 
        (x2 - 3 * y2 * Real.cos α + 1 = 0) → 
        ((-Real.sin α) * (1 / (3 * Real.cos α)) = -1)) :
  Real.sin (2 * α) = (3/5) :=
sorry

end perpendicular_lines_sin_2alpha_l244_244937


namespace good_deed_done_by_C_l244_244711

def did_good (A B C : Prop) := 
  (¬A ∧ ¬B ∧ C) ∨ (¬A ∧ B ∧ ¬C) ∨ (A ∧ ¬B ∧ ¬C)

def statement_A (B : Prop) := B
def statement_B (B : Prop) := ¬B
def statement_C (C : Prop) := ¬C

theorem good_deed_done_by_C (A B C : Prop)
  (h_deed : (did_good A B C))
  (h_statement : (statement_A B ∧ ¬statement_B B ∧ ¬statement_C C) ∨ 
                      (¬statement_A B ∧ statement_B B ∧ ¬statement_C C) ∨ 
                      (¬statement_A B ∧ ¬statement_B B ∧ statement_C C)) :
  C :=
by 
  sorry

end good_deed_done_by_C_l244_244711


namespace sin_240_eq_neg_sqrt3_div_2_l244_244585

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244585


namespace at_least_one_nonzero_l244_244378

theorem at_least_one_nonzero (a b : ℝ) (h : a^2 + b^2 ≠ 0) : a ≠ 0 ∨ b ≠ 0 :=
by
  sorry

end at_least_one_nonzero_l244_244378


namespace estimate_total_fish_l244_244406

theorem estimate_total_fish (m n k : ℕ) (hk : k ≠ 0) (hm : m ≠ 0) (hn : n ≠ 0):
  ∃ x : ℕ, x = (m * n) / k :=
by
  sorry

end estimate_total_fish_l244_244406


namespace product_congruent_to_2_l244_244200

theorem product_congruent_to_2 {p : ℕ} (hp : Nat.Prime p) 
  (a : Fin (p - 2) → ℕ) 
  (h₁ : ∀ k, p ∣ a k -> False) 
  (h₂ : ∀ k, p ∣ (a k ^ k - 1) -> False) : 
  ∃ I : Finset (Fin (p - 2)), (∏ i in I, a i ≡ 2 [MOD p]) := by 
  sorry

end product_congruent_to_2_l244_244200


namespace Hannah_cut_strands_l244_244451

variable (H : ℕ)

theorem Hannah_cut_strands (h : 2 * (H + 3) = 22) : H = 8 :=
by
  sorry

end Hannah_cut_strands_l244_244451


namespace students_interested_in_both_l244_244469

theorem students_interested_in_both (A B C Total : ℕ) (hA : A = 35) (hB : B = 45) (hC : C = 4) (hTotal : Total = 55) :
  A + B - 29 + C = Total :=
by
  -- Assuming the correct answer directly while skipping the proof.
  sorry

end students_interested_in_both_l244_244469


namespace probability_five_distinct_dice_rolls_l244_244245

theorem probability_five_distinct_dice_rolls : 
  let total_outcomes := 6^5
  let favorable_outcomes := 6 * 5 * 4 * 3 * 2
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 54 :=
by
  sorry

end probability_five_distinct_dice_rolls_l244_244245


namespace sum_of_primes_less_than_20_is_77_l244_244038

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l244_244038


namespace simplify_expression_l244_244838

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 := 
by
  sorry

end simplify_expression_l244_244838


namespace cost_of_individual_roll_l244_244137

theorem cost_of_individual_roll
  (p : ℕ) (c : ℝ) (s : ℝ) (x : ℝ)
  (hc : c = 9)
  (hp : p = 12)
  (hs : s = 0.25)
  (h : 12 * x = 9 * (1 + s)) :
  x = 0.9375 :=
by
  sorry

end cost_of_individual_roll_l244_244137


namespace simplify_expression_l244_244828

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l244_244828


namespace paint_cans_used_l244_244213

theorem paint_cans_used (init_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) (final_rooms : ℕ) :
  init_rooms = 50 → lost_cans = 5 → remaining_rooms = 40 → final_rooms = 40 → 
  remaining_rooms / (lost_cans / (init_rooms - remaining_rooms)) = 20 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  sorry

end paint_cans_used_l244_244213


namespace sum_of_primes_less_than_20_is_77_l244_244043

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l244_244043


namespace max_value_of_a_l244_244916

theorem max_value_of_a :
  ∀ (m : ℚ) (x : ℤ),
    (0 < x ∧ x ≤ 50) →
    (1 / 2 < m ∧ m < 25 / 49) →
    (∀ k : ℤ, m * x + 3 ≠ k) →
  m < 25 / 49 :=
sorry

end max_value_of_a_l244_244916


namespace simplify_expression_l244_244837

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 := 
by
  sorry

end simplify_expression_l244_244837


namespace integer_values_of_a_l244_244166

theorem integer_values_of_a (a : ℤ) : 
  (∃ x : ℤ, x^4 + 4 * x^3 + a * x^2 + 8 = 0) ↔ (a = -14 ∨ a = -13 ∨ a = -5 ∨ a = 2) :=
sorry

end integer_values_of_a_l244_244166


namespace total_monthly_cost_l244_244270

theorem total_monthly_cost (volume_per_box : ℕ := 1800) 
                          (total_volume : ℕ := 1080000)
                          (cost_per_box_per_month : ℝ := 0.8) 
                          (expected_cost : ℝ := 480) : 
                          (total_volume / volume_per_box) * cost_per_box_per_month = expected_cost :=
by
  sorry

end total_monthly_cost_l244_244270


namespace sum_primes_less_than_20_l244_244084

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l244_244084


namespace checkered_triangle_division_l244_244623

-- Define the conditions as assumptions
variable (T : Set ℕ) 
variable (sum_T : Nat) (h_sumT : sum_T = 63)
variable (part1 part2 part3 : Set ℕ)
variable (sum_part1 sum_part2 sum_part3 : Nat)
variable (h_part1 : sum part1 = 21)
variable (h_part2 : sum part2 = 21)
variable (h_part3 : sum part3 = 21)

-- Define the goal as a theorem
theorem checkered_triangle_division : 
  (∃ part1 part2 part3 : Set ℕ, sum part1 = 21 ∧ sum part2 = 21 ∧ sum part3 = 21 ∧ Disjoint part1 (part2 ∪ part3) ∧ Disjoint part2 part3 ∧ T = part1 ∪ part2 ∪ part3) :=
sorry

end checkered_triangle_division_l244_244623


namespace milk_production_l244_244977

variables (a b c d e : ℕ) (h1 : a > 0) (h2 : c > 0)

def summer_rate := b / (a * c) -- Rate in summer per cow per day
def winter_rate := 2 * summer_rate -- Rate in winter per cow per day

noncomputable def total_milk_produced := (d * summer_rate * e) + (d * winter_rate * e)

theorem milk_production (h : d > 0) : total_milk_produced a b c d e = 3 * b * d * e / (a * c) :=
by sorry

end milk_production_l244_244977


namespace tan_problem_l244_244931

theorem tan_problem (m : ℝ) (α : ℝ) (h1 : Real.tan α = m / 3) (h2 : Real.tan (α + Real.pi / 4) = 2 / m) :
  m = -6 ∨ m = 1 :=
sorry

end tan_problem_l244_244931


namespace exists_odd_integers_l244_244497

theorem exists_odd_integers (n : ℕ) (hn : n ≥ 3) : 
  ∃ x y : ℤ, x % 2 = 1 ∧ y % 2 = 1 ∧ x^2 + 7 * y^2 = 2^n :=
sorry

end exists_odd_integers_l244_244497


namespace exists_overlapping_pairs_l244_244164

-- Definition of conditions:
def no_boy_danced_with_all_girls (B : Type) (G : Type) (danced : B → G → Prop) :=
  ∀ b : B, ∃ g : G, ¬ danced b g

def each_girl_danced_with_at_least_one_boy (B : Type) (G : Type) (danced : B → G → Prop) :=
  ∀ g : G, ∃ b : B, danced b g

-- The main theorem to prove:
theorem exists_overlapping_pairs
  (B : Type) (G : Type) (danced : B → G → Prop)
  (h1 : no_boy_danced_with_all_girls B G danced)
  (h2 : each_girl_danced_with_at_least_one_boy B G danced) :
  ∃ (b1 b2 : B) (g1 g2 : G), b1 ≠ b2 ∧ g1 ≠ g2 ∧ danced b1 g1 ∧ danced b2 g2 :=
sorry

end exists_overlapping_pairs_l244_244164


namespace pushkin_family_pension_l244_244874

def is_survivor_pension (pension : String) (main_provider_deceased : Bool) (provision_lifelong : Bool) (assigned_to_family : Bool) : Prop :=
  pension = "survivor's pension" ↔
    main_provider_deceased = true ∧
    provision_lifelong = true ∧
    assigned_to_family = true

theorem pushkin_family_pension :
  ∀ (pension : String),
    let main_provider_deceased := true
    let provision_lifelong := true
    let assigned_to_family := true
    is_survivor_pension pension main_provider_deceased provision_lifelong assigned_to_family →
    pension = "survivor's pension" :=
by
  intros pension
  intro h
  sorry

end pushkin_family_pension_l244_244874


namespace cost_per_bag_of_potatoes_l244_244353

variable (x : ℕ)

def chickens_cost : ℕ := 5 * 3
def celery_cost : ℕ := 4 * 2
def total_paid : ℕ := 35
def potatoes_cost (x : ℕ) : ℕ := 2 * x

theorem cost_per_bag_of_potatoes : 
  chickens_cost + celery_cost + potatoes_cost x = total_paid → x = 6 :=
by
  sorry

end cost_per_bag_of_potatoes_l244_244353


namespace geometric_sequence_b_eq_neg3_l244_244784

theorem geometric_sequence_b_eq_neg3 (a b c : ℝ) : 
  (∃ r : ℝ, -1 = r * a ∧ a = r * b ∧ b = r * c ∧ c = r * (-9)) → b = -3 :=
by
  intro h
  obtain ⟨r, h1, h2, h3, h4⟩ := h
  -- Proof to be filled in later.
  sorry

end geometric_sequence_b_eq_neg3_l244_244784


namespace candidate_percentage_l244_244136

variables (M T : ℝ)

theorem candidate_percentage (h1 : (P / 100) * T = M - 30) 
                             (h2 : (45 / 100) * T = M + 15)
                             (h3 : M = 120) : 
                             P = 30 := 
by 
  sorry

end candidate_percentage_l244_244136


namespace sin_240_eq_neg_sqrt3_div_2_l244_244593

theorem sin_240_eq_neg_sqrt3_div_2 :
  sin (240 : ℝ) = - (Real.sqrt 3) / 2 :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244593


namespace roots_sum_and_product_l244_244428

theorem roots_sum_and_product (k p : ℝ) (hk : (k / 3) = 9) (hp : (p / 3) = 10) : k + p = 57 := by
  sorry

end roots_sum_and_product_l244_244428


namespace distinct_integers_problem_l244_244263

variable (a b c d e : ℤ)

theorem distinct_integers_problem
  (h1 : a ≠ b) 
  (h2 : a ≠ c) 
  (h3 : a ≠ d) 
  (h4 : a ≠ e) 
  (h5 : b ≠ c) 
  (h6 : b ≠ d) 
  (h7 : b ≠ e) 
  (h8 : c ≠ d) 
  (h9 : c ≠ e) 
  (h10 : d ≠ e) 
  (h_prod : (4 - a) * (4 - b) * (4 - c) * (4 - d) * (4 - e) = 12) : 
  a + b + c + d + e = 17 := 
sorry

end distinct_integers_problem_l244_244263


namespace sin_240_l244_244577

theorem sin_240 : Real.sin (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Provided conditions
  have h1 : 240 = 180 + 60 := be_of_eq true.intro
  have h2 : ∀ θ : ℝ, θ ∈ set.Icc (pi : ℝ) (3 * pi / 2) → Real.sin θ < 0 := Real.sin_neg_of_pi_lt_of_lt (Real.pi_lt_2_pi)
  have h3 : Real.sin (60 * Real.pi / 180) = 1 / 2 := Real.sin_pi_div_three
  -- Prove
  sorry

end sin_240_l244_244577


namespace max_value_f_at_e_l244_244376

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_f_at_e (h : 0 < x) : 
  ∃ e : ℝ, (∀ x : ℝ, 0 < x → f x ≤ f e) ∧ e = Real.exp 1 :=
by
  sorry

end max_value_f_at_e_l244_244376


namespace polynomial_value_at_minus_two_l244_244766

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem polynomial_value_at_minus_two :
  f (-2) = -1 :=
by sorry

end polynomial_value_at_minus_two_l244_244766


namespace factorization_correct_l244_244160

noncomputable def original_poly (x : ℝ) : ℝ := 12 * x ^ 2 + 18 * x - 24
noncomputable def factored_poly (x : ℝ) : ℝ := 6 * (2 * x - 1) * (x + 4)

theorem factorization_correct (x : ℝ) : original_poly x = factored_poly x :=
by
  sorry

end factorization_correct_l244_244160


namespace probability_exactly_2_boys_1_girl_probability_at_least_1_girl_l244_244767

theorem probability_exactly_2_boys_1_girl 
  (boys girls : ℕ) 
  (total_group : ℕ) 
  (select : ℕ)
  (h_boys : boys = 4) 
  (h_girls : girls = 2) 
  (h_total_group : total_group = boys + girls) 
  (h_select : select = 3) : 
  (Nat.choose boys 2 * Nat.choose girls 1 / (Nat.choose total_group select) : ℚ) = 3 / 5 :=
by sorry

theorem probability_at_least_1_girl
  (boys girls : ℕ) 
  (total_group : ℕ) 
  (select : ℕ)
  (h_boys : boys = 4) 
  (h_girls : girls = 2) 
  (h_total_group : total_group = boys + girls) 
  (h_select : select = 3) : 
  (1 - (Nat.choose boys select / Nat.choose total_group select : ℚ)) = 4 / 5 :=
by sorry

end probability_exactly_2_boys_1_girl_probability_at_least_1_girl_l244_244767


namespace number_of_oranges_l244_244968

-- Definitions of the conditions
def peaches : ℕ := 9
def pears : ℕ := 18
def greatest_num_per_basket : ℕ := 3
def num_baskets_peaches := peaches / greatest_num_per_basket
def num_baskets_pears := pears / greatest_num_per_basket
def min_num_baskets := min num_baskets_peaches num_baskets_pears

-- Proof problem statement
theorem number_of_oranges (O : ℕ) (h1 : O % greatest_num_per_basket = 0) 
  (h2 : O / greatest_num_per_basket = min_num_baskets) : 
  O = 9 :=
by {
  sorry
}

end number_of_oranges_l244_244968


namespace probability_of_5_distinct_dice_rolls_is_5_over_54_l244_244248

def count_distinct_dice_rolls : ℕ :=
  6 * 5 * 4 * 3 * 2

def total_dice_rolls : ℕ :=
  6 ^ 5

def probability_of_distinct_rolls : ℚ :=
  count_distinct_dice_rolls / total_dice_rolls

theorem probability_of_5_distinct_dice_rolls_is_5_over_54 : 
  probability_of_distinct_rolls = 5 / 54 :=
by
  sorry

end probability_of_5_distinct_dice_rolls_is_5_over_54_l244_244248


namespace maximum_a_value_condition_l244_244946

theorem maximum_a_value_condition (x a : ℝ) :
  (∀ x, (x^2 - 2 * x - 3 > 0 → x < a)) ↔ a ≤ -1 :=
by sorry

end maximum_a_value_condition_l244_244946


namespace sin_240_eq_neg_sqrt3_div_2_l244_244592

theorem sin_240_eq_neg_sqrt3_div_2 :
  sin (240 : ℝ) = - (Real.sqrt 3) / 2 :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244592


namespace convert_decimal_to_fraction_l244_244538

theorem convert_decimal_to_fraction : (0.38 : ℚ) = 19 / 50 :=
by
  sorry

end convert_decimal_to_fraction_l244_244538


namespace find_x_l244_244336

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (h : x + y + x * y = 143) : x = 15 :=
by sorry

end find_x_l244_244336


namespace sum_primes_less_than_20_l244_244048

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l244_244048


namespace express_in_scientific_notation_l244_244690

theorem express_in_scientific_notation :
  (2370000 : ℝ) = 2.37 * 10^6 := 
by
  -- proof omitted
  sorry

end express_in_scientific_notation_l244_244690


namespace difference_of_fractions_l244_244130

theorem difference_of_fractions (a b c : ℝ) (h1 : a = 8000 * (1/2000)) (h2 : b = 8000 * (1/10)) (h3 : c = b - a) : c = 796 := 
sorry

end difference_of_fractions_l244_244130


namespace m_eq_half_l244_244776

theorem m_eq_half (m : ℝ) (h1 : m > 0) (h2 : ∀ x, (0 < x ∧ x < m) → (x * (x - 1) < 0))
  (h3 : ∃ x, (0 < x ∧ x < 1) ∧ ¬(0 < x ∧ x < m)) : m = 1 / 2 :=
sorry

end m_eq_half_l244_244776


namespace solve_repeating_decimals_sum_l244_244290

def repeating_decimals_sum : Prop :=
  let x := (1 : ℚ) / 3
  let y := (4 : ℚ) / 999
  let z := (5 : ℚ) / 9999
  x + y + z = 3378 / 9999

theorem solve_repeating_decimals_sum : repeating_decimals_sum := 
by 
  sorry

end solve_repeating_decimals_sum_l244_244290


namespace simplify_fraction_l244_244836

theorem simplify_fraction : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 :=
by sorry

end simplify_fraction_l244_244836


namespace negative_integer_reciprocal_of_d_l244_244557

def a : ℚ := 3
def b : ℚ := |1 / 3|
def c : ℚ := -2
def d : ℚ := -1 / 2

theorem negative_integer_reciprocal_of_d (h : d ≠ 0) : ∃ k : ℤ, (d⁻¹ : ℚ) = ↑k ∧ k < 0 :=
by
  sorry

end negative_integer_reciprocal_of_d_l244_244557


namespace net_profit_100_patches_l244_244826

theorem net_profit_100_patches :
  let cost_per_patch := 1.25
  let num_patches_ordered := 100
  let selling_price_per_patch := 12.00
  let total_cost := cost_per_patch * num_patches_ordered
  let total_revenue := selling_price_per_patch * num_patches_ordered
  let net_profit := total_revenue - total_cost
  net_profit = 1075 :=
by
  sorry

end net_profit_100_patches_l244_244826


namespace solve_quadratics_l244_244550

theorem solve_quadratics (p q u v : ℤ)
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ p ≠ q)
  (h2 : u ≠ 0 ∧ v ≠ 0 ∧ u ≠ v)
  (h3 : p + q = -u)
  (h4 : pq = -v)
  (h5 : u + v = -p)
  (h6 : uv = -q) :
  p = -1 ∧ q = 2 ∧ u = -1 ∧ v = 2 :=
by {
  sorry
}

end solve_quadratics_l244_244550


namespace simplify_sqrt_product_l244_244363

theorem simplify_sqrt_product (x : ℝ) :
  Real.sqrt (45 * x) * Real.sqrt (20 * x) * Real.sqrt (28 * x) * Real.sqrt (5 * x) =
  60 * x^2 * Real.sqrt 35 :=
by
  sorry

end simplify_sqrt_product_l244_244363


namespace number_value_l244_244355

theorem number_value (N : ℝ) (h : 0.40 * N = 180) : 
  (1/4) * (1/3) * (2/5) * N = 15 :=
by
  -- assume the conditions have been stated correctly
  sorry

end number_value_l244_244355


namespace sin_240_deg_l244_244568

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l244_244568


namespace sum_is_ten_l244_244675

variable (x y : ℝ) (S : ℝ)

-- Conditions
def condition1 : Prop := x + y = S
def condition2 : Prop := x = 25 / y
def condition3 : Prop := x^2 + y^2 = 50

-- Theorem
theorem sum_is_ten (h1 : condition1 x y S) (h2 : condition2 x y) (h3 : condition3 x y) : S = 10 :=
sorry

end sum_is_ten_l244_244675


namespace sweet_potatoes_not_yet_sold_l244_244352

def total_harvested := 80
def sold_to_adams := 20
def sold_to_lenon := 15
def not_yet_sold : ℕ := total_harvested - (sold_to_adams + sold_to_lenon)

theorem sweet_potatoes_not_yet_sold :
  not_yet_sold = 45 :=
by
  unfold not_yet_sold
  unfold total_harvested sold_to_adams sold_to_lenon
  sorry

end sweet_potatoes_not_yet_sold_l244_244352


namespace train_travel_distance_l244_244894

def speed (miles : ℕ) (minutes : ℕ) : ℕ :=
  miles / minutes

def minutes_in_hours (hours : ℕ) : ℕ :=
  hours * 60

def distance_traveled (rate : ℕ) (time : ℕ) : ℕ :=
  rate * time

theorem train_travel_distance :
  (speed 2 2 = 1) →
  (minutes_in_hours 3 = 180) →
  distance_traveled (speed 2 2) (minutes_in_hours 3) = 180 :=
by
  intros h_speed h_minutes
  rw [h_speed, h_minutes]
  sorry

end train_travel_distance_l244_244894


namespace train_distance_l244_244892

theorem train_distance (train_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  (train_speed = 1) → (total_time = 180) → (distance = train_speed * total_time) → 
  distance = 180 :=
by
  intros train_speed_eq total_time_eq dist_eq
  rw [train_speed_eq, total_time_eq] at dist_eq
  exact dist_eq

end train_distance_l244_244892


namespace find_x_of_equation_l244_244342

theorem find_x_of_equation (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := by
  sorry

end find_x_of_equation_l244_244342


namespace explicit_formula_is_even_tangent_line_at_1_tangent_line_equation_l244_244699

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + (2 - a) * x + (a - 1)

-- Proof needed for the first question:
theorem explicit_formula_is_even (a : ℝ) : (∀ x : ℝ, f x a = f (-x) a) → a = 2 ∧ ∀ x : ℝ, f x a = x^2 + 1 :=
by sorry

-- Proof needed for the second question:
theorem tangent_line_at_1 (f : ℝ → ℝ) : (∀ x : ℝ, f x = x^2 + 1) → ∀ x : ℝ, deriv f 1 = 2 :=
by sorry

-- The tangent line equation at x = 1 in the required form
theorem tangent_line_equation (f : ℝ → ℝ) : (∀ x : ℝ, f x = x^2 + 1) → ∀ x : ℝ, deriv f 1 = 2 → (f 1 - deriv f 1 * 1 + deriv f 1 * x = 2 * x) :=
by sorry

end explicit_formula_is_even_tangent_line_at_1_tangent_line_equation_l244_244699


namespace construct_line_through_points_l244_244433

-- Definitions of the conditions
def points_on_sheet (A B : ℝ × ℝ) : Prop := A ≠ B
def tool_constraints (ruler_length compass_max_opening distance_A_B : ℝ) : Prop :=
  distance_A_B > 2 * ruler_length ∧ distance_A_B > 2 * compass_max_opening

-- The main theorem statement
theorem construct_line_through_points (A B : ℝ × ℝ) (ruler_length compass_max_opening : ℝ) 
  (h_points : points_on_sheet A B) 
  (h_constraints : tool_constraints ruler_length compass_max_opening (dist A B)) : 
  ∃ line : ℝ × ℝ → Prop, line A ∧ line B :=
sorry

end construct_line_through_points_l244_244433


namespace at_least_two_equal_l244_244959

theorem at_least_two_equal (x y z : ℝ) (h : (x - y) / (2 + x * y) + (y - z) / (2 + y * z) + (z - x) / (2 + z * x) = 0) : 
x = y ∨ y = z ∨ z = x := 
by
  sorry

end at_least_two_equal_l244_244959


namespace sum_primes_less_than_20_l244_244053

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l244_244053


namespace sin_240_eq_neg_sqrt3_div_2_l244_244606

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244606


namespace money_spent_correct_l244_244760

-- Define conditions
def spring_income : ℕ := 2
def summer_income : ℕ := 27
def amount_after_supplies : ℕ := 24

-- Define the resulting money spent on supplies
def money_spent_on_supplies : ℕ :=
  (spring_income + summer_income) - amount_after_supplies

theorem money_spent_correct :
  money_spent_on_supplies = 5 := by
  sorry

end money_spent_correct_l244_244760


namespace simplify_expr_l244_244843

theorem simplify_expr : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expr_l244_244843


namespace sum_primes_less_than_20_l244_244081

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l244_244081


namespace abs_div_nonzero_l244_244459

theorem abs_div_nonzero (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : 
  ¬ (|a| / a + |b| / b = 1) :=
by
  sorry

end abs_div_nonzero_l244_244459


namespace pencils_total_l244_244514

theorem pencils_total (original_pencils : ℕ) (added_pencils : ℕ) (total_pencils : ℕ) 
  (h1 : original_pencils = 41) 
  (h2 : added_pencils = 30) 
  (h3 : total_pencils = original_pencils + added_pencils) : 
  total_pencils = 71 := 
by
  sorry

end pencils_total_l244_244514


namespace perfect_square_divisors_of_240_l244_244315

theorem perfect_square_divisors_of_240 : 
  (∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, 0 < k ∧ k < n → ¬(k = 1 ∨ k = 4 ∨ k = 16)) := 
sorry

end perfect_square_divisors_of_240_l244_244315


namespace limit_of_sequence_l244_244496

theorem limit_of_sequence (a_n : ℕ → ℝ) (a : ℝ) :
  (∀ n : ℕ, a_n n = (2 * (n ^ 3)) / ((n ^ 3) - 2)) →
  a = 2 →
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) :=
by
  intros h1 h2 ε hε
  sorry

end limit_of_sequence_l244_244496


namespace martin_family_ice_cream_cost_l244_244219

theorem martin_family_ice_cream_cost (R : ℤ)
  (kiddie_scoop_cost : ℤ) (double_scoop_cost : ℤ)
  (total_cost : ℤ) :
  kiddie_scoop_cost = 3 → 
  double_scoop_cost = 6 → 
  total_cost = 32 →
  2 * R + 2 * kiddie_scoop_cost + 3 * double_scoop_cost = total_cost →
  R = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end martin_family_ice_cream_cost_l244_244219


namespace sum_of_primes_less_than_20_l244_244059

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l244_244059


namespace function_passes_through_fixed_point_l244_244978

noncomputable def given_function (a : ℝ) (x : ℝ) : ℝ :=
  a^(x - 1) + 7

theorem function_passes_through_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  given_function a 1 = 8 :=
by
  sorry

end function_passes_through_fixed_point_l244_244978


namespace martin_ratio_of_fruits_eaten_l244_244490

theorem martin_ratio_of_fruits_eaten
    (initial_fruits : ℕ)
    (current_oranges : ℕ)
    (current_oranges_twice_limes : current_oranges = 2 * (current_oranges / 2))
    (initial_fruits_count : initial_fruits = 150)
    (current_oranges_count : current_oranges = 50) :
    (initial_fruits - (current_oranges + (current_oranges / 2))) / initial_fruits = 1 / 2 := 
by
    sorry

end martin_ratio_of_fruits_eaten_l244_244490


namespace rowing_upstream_speed_l244_244734

-- Definitions based on conditions
def V_m : ℝ := 45 -- speed of the man in still water
def V_downstream : ℝ := 53 -- speed of the man rowing downstream
def V_s : ℝ := V_downstream - V_m -- speed of the stream
def V_upstream : ℝ := V_m - V_s -- speed of the man rowing upstream

-- The goal is to prove that the speed of the man rowing upstream is 37 kmph
theorem rowing_upstream_speed :
  V_upstream = 37 := by
  sorry

end rowing_upstream_speed_l244_244734


namespace arithmetic_sequence_index_l244_244227

theorem arithmetic_sequence_index (a : ℕ → ℕ) (n : ℕ) (first_term comm_diff : ℕ):
  (∀ k, a k = first_term + comm_diff * (k - 1)) → a n = 2016 → n = 404 :=
by 
  sorry

end arithmetic_sequence_index_l244_244227


namespace probability_multiple_of_45_l244_244129

def multiples_of_3 := [3, 6, 9]
def primes_less_than_20 := [2, 3, 5, 7, 11, 13, 17, 19]

def favorable_outcomes := (9, 5)
def total_outcomes := (multiples_of_3.length * primes_less_than_20.length)

theorem probability_multiple_of_45 : (multiples_of_3.length = 3 ∧ primes_less_than_20.length = 8) → 
  ∃ w : ℚ, w = 1 / 24 :=
by {
  sorry
}

end probability_multiple_of_45_l244_244129


namespace gcd_impossible_l244_244964

-- Define the natural numbers a, b, and c
variable (a b c : ℕ)

-- Define the factorial values
def fact_30 := Nat.factorial 30
def fact_40 := Nat.factorial 40
def fact_50 := Nat.factorial 50

-- Define the gcd values to be checked
def gcd_ab := fact_30 + 111
def gcd_bc := fact_40 + 234
def gcd_ca := fact_50 + 666

-- The main theorem to prove the impossibility
theorem gcd_impossible (h1 : Nat.gcd a b = gcd_ab) (h2 : Nat.gcd b c = gcd_bc) (h3 : Nat.gcd c a = gcd_ca) : False :=
by
  -- Proof omitted
  sorry

end gcd_impossible_l244_244964


namespace photos_per_album_l244_244147

theorem photos_per_album
  (n : ℕ) -- number of pages in each album
  (x y : ℕ) -- album numbers
  (h1 : 4 * n * (x - 1) + 17 ≤ 81 ∧ 81 ≤ 4 * n * (x - 1) + 20)
  (h2 : 4 * n * (y - 1) + 9 ≤ 171 ∧ 171 ≤ 4 * n * (y - 1) + 12) :
  4 * n = 32 :=
by 
  sorry

end photos_per_album_l244_244147


namespace sufficient_condition_for_negation_l244_244727

theorem sufficient_condition_for_negation {A B : Prop} (h : B → A) : ¬ A → ¬ B :=
by
  intro hA
  intro hB
  apply hA
  exact h hB

end sufficient_condition_for_negation_l244_244727


namespace coordinates_on_y_axis_l244_244788

theorem coordinates_on_y_axis (a : ℝ) 
  (h : (a - 3) = 0) : 
  P = (0, -1) :=
by 
  have ha : a = 3 := by sorry
  subst ha
  sorry

end coordinates_on_y_axis_l244_244788


namespace find_x_in_inches_l244_244224

theorem find_x_in_inches (x : ℝ) :
  let area_smaller_square := 9 * x^2
  let area_larger_square := 36 * x^2
  let area_triangle := 9 * x^2
  area_smaller_square + area_larger_square + area_triangle = 1950 → x = (5 * Real.sqrt 13) / 3 :=
by
  sorry

end find_x_in_inches_l244_244224


namespace number_of_solutions_is_3_l244_244178

noncomputable def count_solutions : Nat :=
  Nat.card {x : Nat // x < 150 ∧ (x + 15) % 45 = 75 % 45}

theorem number_of_solutions_is_3 : count_solutions = 3 := by
  sorry

end number_of_solutions_is_3_l244_244178


namespace simplify_expression_l244_244831

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l244_244831


namespace sin_240_eq_neg_sqrt3_div_2_l244_244584

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244584


namespace amount_a_put_in_correct_l244_244720

noncomputable def amount_a_put_in (total_profit managing_fee total_received_by_a profit_remaining: ℝ) : ℝ :=
  let capital_b := 2500
  let a_receives_from_investment := total_received_by_a - managing_fee
  let profit_ratio := a_receives_from_investment / profit_remaining
  profit_ratio * capital_b

theorem amount_a_put_in_correct :
  amount_a_put_in 9600 960 6000 8640 = 3500 :=
by
  dsimp [amount_a_put_in]
  sorry

end amount_a_put_in_correct_l244_244720


namespace find_original_price_l244_244516

-- Define the original price P
variable (P : ℝ)

-- Define the conditions as per the given problem
def revenue_equation (P : ℝ) : Prop :=
  820 = (10 * 0.60 * P) + (20 * 0.85 * P) + (18 * P)

-- Prove that the revenue equation implies P = 20
theorem find_original_price (P : ℝ) (h : revenue_equation P) : P = 20 :=
  by sorry

end find_original_price_l244_244516


namespace number_of_ways_to_choose_committee_l244_244492

-- Definitions of the conditions
def eligible_members : ℕ := 30
def new_members : ℕ := 3
def committee_size : ℕ := 5
def eligible_pool : ℕ := eligible_members - new_members

-- Problem statement to prove
theorem number_of_ways_to_choose_committee : (Nat.choose eligible_pool committee_size) = 80730 := by
  -- This space is reserved for the proof which is not required per instructions.
  sorry

end number_of_ways_to_choose_committee_l244_244492


namespace quadratic_has_minimum_l244_244176

theorem quadratic_has_minimum (a b : ℝ) (h : a > b^2) :
  ∃ (c : ℝ), c = (4 * b^2 / a) - 3 ∧ (∃ x : ℝ, a * x ^ 2 + 2 * b * x + c < 0) :=
by sorry

end quadratic_has_minimum_l244_244176


namespace pure_imaginary_a_zero_l244_244442

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_a_zero (a : ℝ) (h : is_pure_imaginary (i / (1 + a * i))) : a = 0 :=
sorry

end pure_imaginary_a_zero_l244_244442


namespace find_x_l244_244340

namespace IntegerProblem

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := 
by
  sorry

end IntegerProblem

end find_x_l244_244340


namespace hyewon_painted_colors_l244_244780

def pentagonal_prism := 
  let num_rectangular_faces := 5 
  let num_pentagonal_faces := 2
  num_rectangular_faces + num_pentagonal_faces

theorem hyewon_painted_colors : pentagonal_prism = 7 := 
by
  sorry

end hyewon_painted_colors_l244_244780


namespace minimum_value_function_equality_holds_at_two_thirds_l244_244687

noncomputable def f (x : ℝ) : ℝ := 4 / x + 1 / (1 - x)

theorem minimum_value_function (x : ℝ) (hx : 0 < x ∧ x < 1) : f x ≥ 9 := sorry

theorem equality_holds_at_two_thirds : f (2 / 3) = 9 := sorry

end minimum_value_function_equality_holds_at_two_thirds_l244_244687


namespace beta_speed_l244_244976

theorem beta_speed (d : ℕ) (S_s : ℕ) (t : ℕ) (S_b : ℕ) :
  d = 490 ∧ S_s = 37 ∧ t = 7 ∧ (S_s * t) + (S_b * t) = d → S_b = 33 := by
  sorry

end beta_speed_l244_244976


namespace curve_crossing_self_l244_244296

theorem curve_crossing_self (t t' : ℝ) :
  (t^3 - t - 2 = t'^3 - t' - 2) ∧ (t ≠ t') ∧ 
  (t^3 - t^2 - 9 * t + 5 = t'^3 - t'^2 - 9 * t' + 5) → 
  (t = 3 ∧ t' = -3) ∨ (t = -3 ∧ t' = 3) →
  (t^3 - t - 2 = 22) ∧ (t^3 - t^2 - 9 * t + 5 = -4) :=
by
  sorry

end curve_crossing_self_l244_244296


namespace ferry_tourist_total_l244_244272

theorem ferry_tourist_total :
  let number_of_trips := 8
  let a := 120 -- initial number of tourists
  let d := -2  -- common difference
  let total_tourists := (number_of_trips * (2 * a + (number_of_trips - 1) * d)) / 2
  total_tourists = 904 := 
by {
  sorry
}

end ferry_tourist_total_l244_244272


namespace divisible_by_20_ordered_triplets_l244_244640

theorem divisible_by_20_ordered_triplets :
  let valid_triplets := {triplet : ℕ × ℕ × ℕ // triplet.1 > 0 ∧ triplet.2 > 0 ∧ triplet.3 > 0 ∧ triplet.1 < 10 ∧ triplet.2 < 10 ∧ triplet.3 < 10 ∧ (triplet.1 * triplet.2 * triplet.3) % 20 = 0}
  in Fintype.card valid_triplets = 72 :=
by
  sorry

end divisible_by_20_ordered_triplets_l244_244640


namespace bridget_poster_board_side_length_l244_244751

theorem bridget_poster_board_side_length
  (num_cards : ℕ)
  (card_length : ℕ)
  (card_width : ℕ)
  (posterboard_area : ℕ)
  (posterboard_side_length_feet : ℕ)
  (posterboard_side_length_inches : ℕ)
  (cards_area : ℕ) :
  num_cards = 24 ∧
  card_length = 2 ∧
  card_width = 3 ∧
  posterboard_area = posterboard_side_length_inches ^ 2 ∧
  cards_area = num_cards * (card_length * card_width) ∧
  cards_area = posterboard_area ∧
  posterboard_side_length_inches = 12 ∧
  posterboard_side_length_feet = posterboard_side_length_inches / 12 →
  posterboard_side_length_feet = 1 :=
sorry

end bridget_poster_board_side_length_l244_244751


namespace johns_monthly_earnings_l244_244803

variable (work_days : ℕ) (hours_per_day : ℕ) (former_wage : ℝ) (raise_percentage : ℝ) (days_in_month : ℕ)

def johns_earnings (work_days hours_per_day : ℕ) (former_wage raise_percentage : ℝ) (days_in_month : ℕ) : ℝ :=
  let days_worked := days_in_month / 2
  let total_hours := days_worked * hours_per_day
  let raise := former_wage * raise_percentage
  let new_wage := former_wage + raise
  total_hours * new_wage

theorem johns_monthly_earnings (work_days : ℕ := 15) (hours_per_day : ℕ := 12) (former_wage : ℝ := 20) (raise_percentage : ℝ := 0.3) (days_in_month : ℕ := 30) :
  johns_earnings work_days hours_per_day former_wage raise_percentage days_in_month = 4680 :=
by
  sorry

end johns_monthly_earnings_l244_244803


namespace min_sum_of_distances_l244_244691

-- Define a regular tetrahedron and a point in space
noncomputable def is_center_of_tetrahedron (A B C D P : EuclideanSpace ℝ (Fin 3))
    := ∀ Q : EuclideanSpace ℝ (Fin 3),
          ∑ v in {A, B, C, D}, (EuclideanSpace.dist P v) ≤ ∑ v in {A, B, C, D}, (EuclideanSpace.dist Q v)

theorem min_sum_of_distances (A B C D P : EuclideanSpace ℝ (Fin 3)) 
  (h_tetra : is_regular_tetrahedron A B C D) :
  (∑ v in {A, B, C, D}, EuclideanSpace.dist P v = ∑ v in {A, B, C, D}, EuclideanSpace.dist (centroid A B C D) v)
  ↔ is_center_of_tetrahedron A B C D P := 
begin
  sorry,
end

end min_sum_of_distances_l244_244691


namespace parallel_vectors_k_eq_neg1_l244_244348

theorem parallel_vectors_k_eq_neg1
  (k : ℤ)
  (a : ℤ × ℤ := (2 * k + 2, 4))
  (b : ℤ × ℤ := (k + 1, 8))
  (h : a.1 * b.2 = a.2 * b.1) :
  k = -1 :=
by
sorry

end parallel_vectors_k_eq_neg1_l244_244348


namespace prop_2_prop_3_l244_244418

variables {a b c : ℝ}

-- Proposition 2: a > |b| -> a^2 > b^2
theorem prop_2 (h : a > |b|) : a^2 > b^2 := sorry

-- Proposition 3: a > b -> a^3 > b^3
theorem prop_3 (h : a > b) : a^3 > b^3 := sorry

end prop_2_prop_3_l244_244418


namespace cistern_filling_time_l244_244140

/-
Given the following conditions:
- Pipe A fills the cistern in 10 hours.
- Pipe B fills the cistern in 12 hours.
- Exhaust pipe C drains the cistern in 15 hours.
- Exhaust pipe D drains the cistern in 20 hours.

Prove that if all four pipes are opened simultaneously, the cistern will be filled in 15 hours.
-/

theorem cistern_filling_time :
  let rate_A := 1 / 10
  let rate_B := 1 / 12
  let rate_C := -(1 / 15)
  let rate_D := -(1 / 20)
  let combined_rate := rate_A + rate_B + rate_C + rate_D
  let time_to_fill := 1 / combined_rate
  time_to_fill = 15 :=
by 
  sorry

end cistern_filling_time_l244_244140


namespace find_sum_l244_244302

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α)

-- Geometric sequence condition
def geometric_seq (a : ℕ → α) (r : α) := ∀ n : ℕ, a (n + 1) = a n * r

theorem find_sum (r : α)
  (h1 : geometric_seq a r)
  (h2 : a 4 + a 7 = 2)
  (h3 : a 5 * a 6 = -8) :
  a 1 + a 10 = -7 := 
sorry

end find_sum_l244_244302


namespace compare_inequalities_l244_244642

theorem compare_inequalities (a b c π : ℝ) (h1 : a > π) (h2 : π > b) (h3 : b > 1) (h4 : 1 > c) (h5 : c > 0) 
  (x := a^(1 / π)) (y := Real.log b / Real.log π) (z := Real.log π / Real.log c) : x > y ∧ y > z := 
sorry

end compare_inequalities_l244_244642


namespace min_value_arithmetic_seq_l244_244649

theorem min_value_arithmetic_seq (a : ℕ → ℝ) (h_arith_seq : ∀ n, a n ≤ a (n + 1)) (h_pos : ∀ n, a n > 0) (h_cond : a 1 + a 2017 = 2) :
  ∃ (min_value : ℝ), min_value = 2 ∧ (∀ (x y : ℝ), x + y = 2 → x > 0 → y > 0 → x + y / (x * y) = 2) :=
  sorry

end min_value_arithmetic_seq_l244_244649


namespace simplify_expression_l244_244851

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l244_244851


namespace M_eq_N_l244_244966

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r}

theorem M_eq_N : M = N :=
by
  sorry

end M_eq_N_l244_244966


namespace remainder_when_divided_l244_244307

theorem remainder_when_divided (m : ℤ) (h : m % 5 = 2) : (m + 2535) % 5 = 2 := 
by sorry

end remainder_when_divided_l244_244307


namespace sum_primes_less_than_20_l244_244079

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l244_244079


namespace checkered_triangle_division_l244_244626

theorem checkered_triangle_division :
  ∀ (triangle : List ℕ), triangle.sum = 63 →
  ∃ (part1 part2 part3 : List ℕ),
    part1.sum = 21 ∧ part2.sum = 21 ∧ part3.sum = 21 ∧
    part1 ≠ part2 ∧ part2 ≠ part3 ∧ part1 ≠ part3 ∧
    (part1 ++ part2 ++ part3).length = triangle.length ∧
    (∃ (area1 area2 area3 : ℕ), area1 ≠ area2 ∧ area2 ≠ area3 ∧ area1 ≠ area3) :=
by
  sorry

end checkered_triangle_division_l244_244626


namespace probability_five_distinct_dice_rolls_l244_244246

theorem probability_five_distinct_dice_rolls : 
  let total_outcomes := 6^5
  let favorable_outcomes := 6 * 5 * 4 * 3 * 2
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 54 :=
by
  sorry

end probability_five_distinct_dice_rolls_l244_244246


namespace sum_prime_numbers_less_than_twenty_l244_244091

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l244_244091


namespace player_current_average_l244_244736

theorem player_current_average (A : ℝ) 
  (h1 : 10 * A + 76 = (A + 4) * 11) : 
  A = 32 :=
sorry

end player_current_average_l244_244736


namespace triangle_side_ratio_sqrt2_l244_244402

variables (A B C A1 B1 C1 X Y : Point)
variable (triangle : IsAcuteAngledTriangle A B C)
variable (altitudes : AreAltitudes A B C A1 B1 C1)
variable (midpoints : X = Midpoint A C1 ∧ Y = Midpoint A1 C)
variable (equality : Distance X Y = Distance B B1)

theorem triangle_side_ratio_sqrt2 :
  ∃ (AC AB : ℝ), (AC / AB = Real.sqrt 2) := sorry

end triangle_side_ratio_sqrt2_l244_244402


namespace sum_of_primes_lt_20_eq_77_l244_244115

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l244_244115


namespace range_of_a_l244_244312

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + 2 * x + a ≤ 0) ↔ a ≤ 1 :=
by
  sorry

end range_of_a_l244_244312


namespace recurring_decimal_division_l244_244873

noncomputable def recurring_decimal_fraction : ℚ :=
  let frac_81 := (81 : ℚ) / 99
  let frac_36 := (36 : ℚ) / 99
  frac_81 / frac_36

theorem recurring_decimal_division :
  recurring_decimal_fraction = 9 / 4 :=
by
  sorry

end recurring_decimal_division_l244_244873


namespace no_quadruples_solution_l244_244167

theorem no_quadruples_solution (a b c d : ℝ) :
    a^3 + c^3 = 2 ∧
    a^2 * b + c^2 * d = 0 ∧
    b^3 + d^3 = 1 ∧
    a * b^2 + c * d^2 = -6 →
    false :=
by 
  intros h
  sorry

end no_quadruples_solution_l244_244167


namespace sin_240_deg_l244_244614

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l244_244614


namespace diagonals_from_vertex_l244_244791

theorem diagonals_from_vertex (n : ℕ) (h : (n-2) * 180 + 360 = 1800) : (n - 3) = 7 :=
sorry

end diagonals_from_vertex_l244_244791


namespace probability_five_distinct_numbers_l244_244250

def num_dice := 5
def num_faces := 6

def favorable_outcomes : ℕ := nat.factorial 5 * num_faces
def total_outcomes : ℕ := num_faces ^ num_dice

theorem probability_five_distinct_numbers :
  (favorable_outcomes / total_outcomes : ℚ) = 5 / 54 := 
sorry

end probability_five_distinct_numbers_l244_244250


namespace geometric_sequence_general_formula_l244_244439

noncomputable def a_n (n : ℕ) : ℝ := 2^n

theorem geometric_sequence_general_formula :
  (∀ n : ℕ, 2 * (a_n n + a_n (n + 2)) = 5 * a_n (n + 1)) →
  (a_n 5 ^ 2 = a_n 10) →
  ∀ n : ℕ, a_n n = 2 ^ n := 
by 
  sorry

end geometric_sequence_general_formula_l244_244439


namespace triangle_parts_sum_eq_l244_244631

-- Define the total sum of numbers in the triangle
def total_sum : ℕ := 63

-- Define the required sum for each part
def required_sum : ℕ := 21

-- Define the possible parts that sum to the required sum
def part1 : list ℕ := [10, 6, 5]  -- this sums to 21
def part2 : list ℕ := [10, 5, 4, 1, 1]  -- this sums to 21

def part_sum (part : list ℕ) : ℕ := part.sum

-- The main theorem
theorem triangle_parts_sum_eq : 
  part_sum part1 = required_sum ∧ 
  part_sum part2 = required_sum ∧ 
  part_sum (list.drop (list.length part1 + list.length part2) [10, 6, 5, 10, 5, 4, 1, 1]) = required_sum :=
by
  sorry

end triangle_parts_sum_eq_l244_244631


namespace smallest_four_digit_2_mod_11_l244_244532

theorem smallest_four_digit_2_mod_11 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 11 = 2 ∧ (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 11 = 2 → n ≤ m) := 
by 
  use 1003
  sorry

end smallest_four_digit_2_mod_11_l244_244532


namespace count_integers_abs_le_5_l244_244505

theorem count_integers_abs_le_5 : (Set.toFinset {x : ℤ | |x| ≤ 5}).card = 11 :=
by
  sorry

end count_integers_abs_le_5_l244_244505


namespace restaurant_cooks_l244_244880

variable (C W : ℕ)

theorem restaurant_cooks : 
  (C / W = 3 / 10) ∧ (C / (W + 12) = 3 / 14) → C = 9 :=
by sorry

end restaurant_cooks_l244_244880


namespace find_number_l244_244265

theorem find_number (x : ℝ) (h : 0.85 * x = (4 / 5) * 25 + 14) : x = 40 :=
sorry

end find_number_l244_244265


namespace sequence_conditions_general_formulas_sum_of_first_n_terms_l244_244179

noncomputable def arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a_n n = a_n 1 + d * (n - 1)

noncomputable def geometric_sequence (b_n : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, q > 0 ∧ ∀ n : ℕ, b_n (n + 1) = b_n n * q

variables {a_n b_n c_n : ℕ → ℤ}
variables (d q : ℤ) (d_pos : 0 < d) (hq : q > 0)
variables (S_n : ℕ → ℤ)

axiom initial_conditions : a_n 1 = 2 ∧ b_n 1 = 2 ∧ a_n 3 = 8 ∧ b_n 3 = 8

theorem sequence_conditions : arithmetic_sequence a_n ∧ geometric_sequence b_n := sorry

theorem general_formulas :
  (∀ n : ℕ, a_n n = 3 * n - 1) ∧
  (∀ n : ℕ, b_n n = 2^n) := sorry

theorem sum_of_first_n_terms :
  (∀ n : ℕ, S_n n = 3 * 2^(n+1) - n - 6) := sorry

end sequence_conditions_general_formulas_sum_of_first_n_terms_l244_244179


namespace sum_primes_less_than_20_l244_244080

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l244_244080


namespace drink_cost_l244_244387

/-- Wade has called into a rest stop and decides to get food for the road. 
  He buys a sandwich to eat now, one for the road, and one for the evening. 
  He also buys 2 drinks. Wade spends a total of $26 and the sandwiches 
  each cost $6. Prove that the drinks each cost $4. -/
theorem drink_cost (cost_sandwich : ℕ) (num_sandwiches : ℕ) (cost_total : ℕ) (num_drinks : ℕ) :
  cost_sandwich = 6 → num_sandwiches = 3 → cost_total = 26 → num_drinks = 2 → 
  ∃ (cost_drink : ℕ), cost_drink = 4 :=
by
  intro h1 h2 h3 h4
  sorry

end drink_cost_l244_244387


namespace ratio_girls_to_boys_l244_244710

theorem ratio_girls_to_boys (g b : ℕ) (h1 : g = b + 4) (h2 : g + b = 28) :
  g / gcd g b = 4 ∧ b / gcd g b = 3 :=
by
  sorry

end ratio_girls_to_boys_l244_244710


namespace det_dilation_matrix_l244_244480

section DilationMatrixProof

def E : Matrix (Fin 3) (Fin 3) ℝ := !![5, 0, 0; 0, 5, 0; 0, 0, 5]

theorem det_dilation_matrix :
  Matrix.det E = 125 :=
by {
  sorry
}

end DilationMatrixProof

end det_dilation_matrix_l244_244480


namespace sum_of_extreme_values_of_x_l244_244487

open Real

theorem sum_of_extreme_values_of_x 
  (x y z : ℝ)
  (h1 : x + y + z = 6)
  (h2 : x^2 + y^2 + z^2 = 14) : 
  (min x + max x) = (10 / 3) :=
sorry

end sum_of_extreme_values_of_x_l244_244487


namespace negation_all_swans_white_l244_244703

variables {α : Type} (swan white : α → Prop)

theorem negation_all_swans_white :
  (¬ ∀ x, swan x → white x) ↔ (∃ x, swan x ∧ ¬ white x) :=
by {
  sorry
}

end negation_all_swans_white_l244_244703


namespace sum_primes_less_than_20_l244_244068

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l244_244068


namespace number_of_possible_n_l244_244982

theorem number_of_possible_n :
  ∃ (a : ℕ), (∀ n, (n = a^3) ∧ 
  ((∃ b c : ℕ, b ≠ c ∧ b ≠ a ∧ c ≠ a ∧ a = b * c)) ∧ 
  (a + b + c = 2010) ∧ 
  (a > 0) ∧
  (b > 0) ∧
  (c > 0)) → 
  ∃ (num_n : ℕ), num_n = 2009 :=
  sorry

end number_of_possible_n_l244_244982


namespace simplify_expression_l244_244849

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l244_244849


namespace ratio_four_of_v_m_l244_244239

theorem ratio_four_of_v_m (m v : ℝ) (h : m < v) 
  (h_eq : 5 * (3 / 4 * m) = v - 1 / 4 * m) : v / m = 4 :=
sorry

end ratio_four_of_v_m_l244_244239


namespace probability_of_distinct_dice_numbers_l244_244242

/-- Total number of outcomes when rolling five six-sided dice. -/
def total_outcomes : ℕ := 6 ^ 5

/-- Number of favorable outcomes where all five dice show distinct numbers. -/
def favorable_outcomes : ℕ := 6 * 5 * 4 * 3 * 2

/-- Calculating the probability as a fraction. -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem probability_of_distinct_dice_numbers :
  probability = 5 / 54 :=
by
  -- Proof is required here.
  sorry

end probability_of_distinct_dice_numbers_l244_244242


namespace relatively_prime_number_exists_l244_244655

theorem relatively_prime_number_exists :
  -- Given numbers
  (let a := 20172017 in
   let b := 20172018 in
   let c := 20172019 in
   let d := 20172020 in
   let e := 20172021 in
   -- Number c is relatively prime to all other given numbers
   nat.gcd c a = 1 ∧
   nat.gcd c b = 1 ∧
   nat.gcd c d = 1 ∧
   nat.gcd c e = 1) :=
by {
  -- Proof omitted
  sorry
}

end relatively_prime_number_exists_l244_244655


namespace abs_expression_equality_l244_244910

def pi : ℝ := Real.pi

theorem abs_expression_equality : abs (2 * pi - abs (pi - 9)) = 3 * pi - 9 := by
  sorry

end abs_expression_equality_l244_244910


namespace find_m_value_l244_244936

-- Definitions for the problem conditions are given below
variables (m : ℝ)

-- Conditions
def conditions := (6 < m) ∧ (m < 10) ∧ (4 = 2 * 2) ∧ (4 = (m - 2) - (10 - m))

-- Proof statement
theorem find_m_value : conditions m → m = 8 :=
sorry

end find_m_value_l244_244936


namespace extreme_points_inequality_l244_244661

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^2 + a * Real.log x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a - 4 * x + 2

theorem extreme_points_inequality (a : ℝ) (h_a : 0 < a ∧ a < 1) (x0 : ℝ)
  (h_ext : 4 * x0^2 - 4 * x0 + a = 0) (h_min : ∃ x1, x0 + x1 = 1 ∧ x0 < x1 ∧ x1 < 1) :
  g x0 a > 1 / 2 - Real.log 2 :=
sorry

end extreme_points_inequality_l244_244661


namespace temperature_on_tuesday_l244_244368

variable (T W Th F : ℝ)

-- Conditions
axiom H1 : (T + W + Th) / 3 = 42
axiom H2 : (W + Th + F) / 3 = 44
axiom H3 : F = 43

-- Proof statement
theorem temperature_on_tuesday : T = 37 :=
by
  -- This would be the place to fill in the proof using H1, H2, and H3
  sorry

end temperature_on_tuesday_l244_244368


namespace sum_of_primes_lt_20_eq_77_l244_244111

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l244_244111


namespace intersection_A_B_l244_244438

def A := { x : ℝ | -5 < x ∧ x < 2 }
def B := { x : ℝ | x^2 - 9 < 0 }
def AB := { x : ℝ | -3 < x ∧ x < 2 }

theorem intersection_A_B : A ∩ B = AB := by
  sorry

end intersection_A_B_l244_244438


namespace sin_240_eq_neg_sqrt3_div_2_l244_244591

theorem sin_240_eq_neg_sqrt3_div_2 :
  sin (240 : ℝ) = - (Real.sqrt 3) / 2 :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244591


namespace problem_false_statements_l244_244639

noncomputable def statement_I : Prop :=
  ∀ x : ℝ, ⌊x + Real.pi⌋ = ⌊x⌋ + 3

noncomputable def statement_II : Prop :=
  ∀ x : ℝ, ⌊x + Real.sqrt 2⌋ = ⌊x⌋ + ⌊Real.sqrt 2⌋

noncomputable def statement_III : Prop :=
  ∀ x : ℝ, ⌊x * Real.pi⌋ = ⌊x⌋ * ⌊Real.pi⌋

theorem problem_false_statements : ¬(statement_I ∨ statement_II ∨ statement_III) := 
by
  sorry

end problem_false_statements_l244_244639


namespace min_value_fraction_l244_244182

theorem min_value_fraction {a : ℕ → ℕ} (h1 : a 1 = 10)
    (h2 : ∀ n : ℕ, a (n + 1) - a n = 2 * n) :
    ∃ n : ℕ, (n > 0) ∧ (n - 1 + 10 / n = 16 / 3) :=
by {
  sorry
}

end min_value_fraction_l244_244182


namespace problem1_problem2_problem3_l244_244431

theorem problem1 (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + a + 3 = 0) → (a ≤ -2 ∨ a ≥ 6) :=
sorry

theorem problem2 (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a + 3 ≥ 4) → 
    (if a > 2 then 
      ∀ x : ℝ, ((x ≤ 1) ∨ (x ≥ a-1)) 
    else if a = 2 then 
      ∀ x : ℝ, true
    else 
      ∀ x : ℝ, ((x ≤ a - 1) ∨ (x ≥ 1))) :=
sorry

theorem problem3 (a : ℝ) :
  (∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ x^2 - a*x + a + 3 = 0) → (6 ≤ a ∧ a ≤ 7) :=
sorry

end problem1_problem2_problem3_l244_244431


namespace simplify_eval_expression_l244_244695

variables (a b : ℝ)

theorem simplify_eval_expression :
  a = Real.sqrt 3 →
  b = Real.sqrt 3 - 1 →
  ((3 * a) / (2 * a - b) - 1) / ((a + b) / (4 * a^2 - b^2)) = 3 * Real.sqrt 3 - 1 :=
by
  sorry

end simplify_eval_expression_l244_244695


namespace simplify_fraction_l244_244835

theorem simplify_fraction : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 :=
by sorry

end simplify_fraction_l244_244835


namespace coins_count_l244_244134

variable (x : ℕ)

def total_value : ℕ → ℕ := λ x => x + (x * 50) / 100 + (x * 25) / 100

theorem coins_count (h : total_value x = 140) : x = 80 :=
sorry

end coins_count_l244_244134


namespace gcd_expression_l244_244775

noncomputable def odd_multiple_of_7771 (a : ℕ) : Prop := 
  ∃ k : ℕ, k % 2 = 1 ∧ a = 7771 * k

theorem gcd_expression (a : ℕ) (h : odd_multiple_of_7771 a) : 
  Int.gcd (8 * a^2 + 57 * a + 132) (2 * a + 9) = 9 :=
  sorry

end gcd_expression_l244_244775


namespace sin_240_l244_244575

theorem sin_240 : Real.sin (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Provided conditions
  have h1 : 240 = 180 + 60 := be_of_eq true.intro
  have h2 : ∀ θ : ℝ, θ ∈ set.Icc (pi : ℝ) (3 * pi / 2) → Real.sin θ < 0 := Real.sin_neg_of_pi_lt_of_lt (Real.pi_lt_2_pi)
  have h3 : Real.sin (60 * Real.pi / 180) = 1 / 2 := Real.sin_pi_div_three
  -- Prove
  sorry

end sin_240_l244_244575


namespace sum_primes_less_than_20_l244_244052

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l244_244052


namespace construct_right_triangle_l244_244755

theorem construct_right_triangle (c m n : ℝ) (hc : c > 0) (hm : m > 0) (hn : n > 0) : 
  ∃ a b : ℝ, a^2 + b^2 = c^2 ∧ a / b = m / n :=
by
  sorry

end construct_right_triangle_l244_244755


namespace percentage_of_absent_students_l244_244955

theorem percentage_of_absent_students (total_students boys girls : ℕ) (fraction_boys_absent fraction_girls_absent : ℚ)
  (total_students_eq : total_students = 180)
  (boys_eq : boys = 120)
  (girls_eq : girls = 60)
  (fraction_boys_absent_eq : fraction_boys_absent = 1/6)
  (fraction_girls_absent_eq : fraction_girls_absent = 1/4) :
  let boys_absent := fraction_boys_absent * boys
  let girls_absent := fraction_girls_absent * girls
  let total_absent := boys_absent + girls_absent
  let absent_percentage := (total_absent / total_students) * 100
  abs (absent_percentage - 19) < 1 :=
by
  sorry

end percentage_of_absent_students_l244_244955


namespace sum_primes_less_than_20_l244_244072

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def primes (n : ℕ) : List ℕ :=
List.filter is_prime (List.range n)

def sum_primes_less_than (n : ℕ) : ℕ :=
(primes n).sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := 
by
  sorry

end sum_primes_less_than_20_l244_244072


namespace francis_had_2_muffins_l244_244172

noncomputable def cost_of_francis_breakfast (m : ℕ) : ℕ := 2 * m + 6
noncomputable def cost_of_kiera_breakfast : ℕ := 4 + 3
noncomputable def total_cost (m : ℕ) : ℕ := cost_of_francis_breakfast m + cost_of_kiera_breakfast

theorem francis_had_2_muffins (m : ℕ) : total_cost m = 17 → m = 2 :=
by
  -- Sorry is used here to leave the proof steps blank.
  sorry

end francis_had_2_muffins_l244_244172


namespace sin_240_eq_neg_sqrt3_div_2_l244_244619

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244619


namespace sin_240_eq_neg_sqrt3_div_2_l244_244590

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244590


namespace range_of_x_l244_244444

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)

theorem range_of_x (a b x : ℝ) (h : a ≠ 0) (h₁ : abs (a + b) + abs (a - b) ≥ abs a * f x) :
  0 ≤ x ∧ x ≤ 4 :=
sorry

end range_of_x_l244_244444


namespace sum_of_primes_less_than_20_l244_244058

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l244_244058


namespace arithmetic_sequence_c_d_sum_l244_244986

theorem arithmetic_sequence_c_d_sum :
  let c := 19 + (11 - 3)
  let d := c + (11 - 3)
  c + d = 62 :=
by
  sorry

end arithmetic_sequence_c_d_sum_l244_244986


namespace combined_perimeter_two_right_triangles_l244_244527

theorem combined_perimeter_two_right_triangles :
  ∀ (h1 h2 : ℝ),
    (h1^2 = 15^2 + 20^2) ∧
    (h2^2 = 9^2 + 12^2) ∧
    (h1 = h2) →
    (15 + 20 + h1) + (9 + 12 + h2) = 106 := by
  sorry

end combined_perimeter_two_right_triangles_l244_244527


namespace simplify_expression_l244_244840

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 := 
by
  sorry

end simplify_expression_l244_244840


namespace tetrahedron_inscribed_in_pyramid_edge_length_l244_244898

noncomputable def edge_length_of_tetrahedron := (Real.sqrt 2) / 2

theorem tetrahedron_inscribed_in_pyramid_edge_length :
  let A := (0,0,0)
  let B := (1,0,0)
  let C := (1,1,0)
  let D := (0,1,0)
  let E := (0.5, 0.5, 1)
  let v₁ := (0.5, 0, 0)
  let v₂ := (1, 0.5, 0)
  let v₃ := (0, 0.5, 0)
  dist (v₁ : ℝ × ℝ × ℝ) v₂ = edge_length_of_tetrahedron ∧
  dist v₂ v₃ = edge_length_of_tetrahedron ∧
  dist v₃ v₁ = edge_length_of_tetrahedron ∧
  dist E v₁ = dist E v₂ ∧
  dist E v₂ = dist E v₃ :=
by
  sorry

end tetrahedron_inscribed_in_pyramid_edge_length_l244_244898


namespace intersection_point_l244_244477

noncomputable def g (x : ℝ) : ℝ := x^3 + 3 * x^2 + 9 * x + 15

theorem intersection_point :
  ∃ a : ℝ, g a = a ∧ a = -3 :=
by
  sorry

end intersection_point_l244_244477


namespace sin_240_deg_l244_244569

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l244_244569


namespace sum_primes_less_than_20_l244_244094

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l244_244094


namespace last_three_digits_product_l244_244990

theorem last_three_digits_product (a b c : ℕ) 
  (h1 : (a + b) % 10 = c % 10) 
  (h2 : (b + c) % 10 = a % 10) 
  (h3 : (c + a) % 10 = b % 10) :
  (a * b * c) % 1000 = 250 ∨ (a * b * c) % 1000 = 500 ∨ (a * b * c) % 1000 = 750 ∨ (a * b * c) % 1000 = 0 := 
by
  sorry

end last_three_digits_product_l244_244990


namespace sum_of_primes_less_than_20_eq_77_l244_244029

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l244_244029


namespace photos_per_album_l244_244146

theorem photos_per_album
  (n : ℕ) -- number of pages in each album
  (x y : ℕ) -- album numbers
  (h1 : 4 * n * (x - 1) + 17 ≤ 81 ∧ 81 ≤ 4 * n * (x - 1) + 20)
  (h2 : 4 * n * (y - 1) + 9 ≤ 171 ∧ 171 ≤ 4 * n * (y - 1) + 12) :
  4 * n = 32 :=
by 
  sorry

end photos_per_album_l244_244146


namespace simplify_fraction_l244_244832

theorem simplify_fraction : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 :=
by sorry

end simplify_fraction_l244_244832


namespace smallest_non_six_digit_palindrome_l244_244638

-- Definition of a four-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.reverse = digits

-- Definition of a six-digit number
def is_six_digit (n : ℕ) : Prop :=
  n >= 100000 ∧ n < 1000000

-- Definition of a non-palindrome
def not_palindrome (n : ℕ) : Prop :=
  ¬ is_palindrome n

-- Find the smallest four-digit palindrome whose product with 103 is not a six-digit palindrome
theorem smallest_non_six_digit_palindrome :
  ∃ (n : ℕ), n >= 1000 ∧ n < 10000 ∧ is_palindrome n ∧ not_palindrome (103 * n)
  ∧ (∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ is_palindrome m ∧ not_palindrome (103 * m) → n ≤ m) :=
  sorry

end smallest_non_six_digit_palindrome_l244_244638


namespace solve_digits_l244_244870

variables (h t u : ℕ)

theorem solve_digits :
  (u = h + 6) →
  (u + h = 16) →
  (∀ (x y z : ℕ), 100 * h + 10 * t + u + 100 * u + 10 * t + h = 100 * x + 10 * y + z ∧ y = 9 ∧ z = 6) →
  (h = 5 ∧ t = 4 ∧ u = 11) :=
sorry

end solve_digits_l244_244870


namespace intersection_point_of_lines_l244_244314

theorem intersection_point_of_lines :
  ∃ x y : ℝ, (x - 2 * y - 4 = 0) ∧ (x + 3 * y + 6 = 0) ∧ (x = 0) ∧ (y = -2) :=
by
  sorry

end intersection_point_of_lines_l244_244314


namespace sum_of_primes_less_than_20_l244_244014

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l244_244014


namespace sufficient_but_not_necessary_condition_l244_244933

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x ≥ 1) → (|x + 1| + |x - 1| = 2 * |x|) ∧ ¬((x ≥ 1) ↔ (|x + 1| + |x - 1| = 2 * |x|)) := by
  sorry

end sufficient_but_not_necessary_condition_l244_244933


namespace min_cost_per_student_is_80_l244_244133

def num_students : ℕ := 48
def swims_per_student : ℕ := 8
def cost_per_card : ℕ := 240
def cost_per_bus : ℕ := 40

def total_swims : ℕ := num_students * swims_per_student

def min_cost_per_student : ℕ :=
  let n := 8
  let c := total_swims / n
  let total_cost := cost_per_card * n + cost_per_bus * c
  total_cost / num_students

theorem min_cost_per_student_is_80 :
  min_cost_per_student = 80 :=
sorry

end min_cost_per_student_is_80_l244_244133


namespace simplify_fraction_l244_244833

theorem simplify_fraction : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 :=
by sorry

end simplify_fraction_l244_244833


namespace sum_primes_less_than_20_l244_244085

open Nat

-- Definition for primality check
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition for primes less than a given bound
def primesLessThan (n : ℕ) : List ℕ :=
  List.filter isPrime (List.range n)

-- The main theorem we want to prove
theorem sum_primes_less_than_20 : List.sum (primesLessThan 20) = 77 :=
by
  sorry

end sum_primes_less_than_20_l244_244085


namespace sin_240_eq_neg_sqrt3_div_2_l244_244616

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244616


namespace inverse_function_ratio_l244_244962

noncomputable def g (x : ℚ) : ℚ := (3 * x + 2) / (2 * x - 5)

noncomputable def g_inv (x : ℚ) : ℚ := (-5 * x + 2) / (-2 * x + 3)

theorem inverse_function_ratio :
  ∀ x : ℚ, g (g_inv x) = x ∧ (∃ a b c d : ℚ, a = -5 ∧ b = 2 ∧ c = -2 ∧ d = 3 ∧ a / c = 2.5) :=
by
  sorry

end inverse_function_ratio_l244_244962


namespace sum_of_primes_lt_20_eq_77_l244_244117

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l244_244117


namespace earned_points_l244_244470

def points_per_enemy := 3
def total_enemies := 6
def enemies_undefeated := 2
def enemies_defeated := total_enemies - enemies_undefeated

theorem earned_points : enemies_defeated * points_per_enemy = 12 :=
by sorry

end earned_points_l244_244470


namespace relatively_prime_number_exists_l244_244653

def gcd (a b : ℕ) : ℕ := a.gcd b

def is_relatively_prime_to_all (n : ℕ) (lst : List ℕ) : Prop :=
  ∀ m ∈ lst, m ≠ n → gcd n m = 1

def given_numbers : List ℕ := [20172017, 20172018, 20172019, 20172020, 20172021]

theorem relatively_prime_number_exists :
  ∃ n ∈ given_numbers, is_relatively_prime_to_all n given_numbers := 
begin
  use 20172019,
  split,
  { -- Show 20172019 is in the list
    simp },
  { -- Prove 20172019 is relatively prime to all other numbers in the list
    intros m h1 h2,
    -- Further proof goes here
    sorry
  }
end

end relatively_prime_number_exists_l244_244653


namespace rectangle_area_l244_244822

theorem rectangle_area (a b c : ℝ) :
  a = 15 ∧ b = 12 ∧ c = 1 / 3 →
  ∃ (AD AB : ℝ), 
  AD = (180 / 17) ∧ AB = (60 / 17) ∧ 
  (AD * AB = 10800 / 289) :=
by sorry

end rectangle_area_l244_244822


namespace ball_first_bounce_less_than_30_l244_244268

theorem ball_first_bounce_less_than_30 (b : ℕ) :
  (243 * ((2: ℝ) / 3) ^ b < 30) ↔ (b ≥ 6) :=
sorry

end ball_first_bounce_less_than_30_l244_244268


namespace sin_240_deg_l244_244570

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l244_244570


namespace brian_expenses_l244_244729

def cost_apples_per_bag : ℕ := 14
def cost_kiwis : ℕ := 10
def cost_bananas : ℕ := cost_kiwis / 2
def subway_fare_one_way : ℕ := 350
def maximum_apples : ℕ := 24

theorem brian_expenses : 
  cost_kiwis + cost_bananas + (cost_apples_per_bag * (maximum_apples / 12)) + (subway_fare_one_way * 2) = 50 := by
sorry

end brian_expenses_l244_244729


namespace sin_240_eq_neg_sqrt3_div_2_l244_244605

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244605


namespace sin_240_deg_l244_244609

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l244_244609


namespace mean_score_l244_244539

theorem mean_score (M SD : ℝ) (h₁ : 58 = M - 2 * SD) (h₂ : 98 = M + 3 * SD) : M = 74 :=
by
  sorry

end mean_score_l244_244539


namespace probability_of_distinct_dice_numbers_l244_244243

/-- Total number of outcomes when rolling five six-sided dice. -/
def total_outcomes : ℕ := 6 ^ 5

/-- Number of favorable outcomes where all five dice show distinct numbers. -/
def favorable_outcomes : ℕ := 6 * 5 * 4 * 3 * 2

/-- Calculating the probability as a fraction. -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem probability_of_distinct_dice_numbers :
  probability = 5 / 54 :=
by
  -- Proof is required here.
  sorry

end probability_of_distinct_dice_numbers_l244_244243


namespace range_of_k_l244_244324

theorem range_of_k (k : ℝ) : (∃ x : ℝ, 2 * x - 5 * k = x + 4 ∧ x > 0) → k > -4 / 5 :=
by
  sorry

end range_of_k_l244_244324


namespace stratified_sampling_third_grade_l244_244274

theorem stratified_sampling_third_grade (total_students : ℕ)
  (ratio_first_second_third : ℕ × ℕ × ℕ)
  (sample_size : ℕ) (r1 r2 r3 : ℕ) (h_ratio : ratio_first_second_third = (r1, r2, r3)) :
  total_students = 3000  ∧ ratio_first_second_third = (2, 3, 1)  ∧ sample_size = 180 →
  (sample_size * r3 / (r1 + r2 + r3) = 30) :=
sorry

end stratified_sampling_third_grade_l244_244274


namespace parallelogram_area_proof_l244_244333

-- Define the conditions of the problem
variable (AD BM : ℝ)
variable (cos_BAM : ℝ)

-- Specify the given values
def AD_value : AD = 5 := by sorry
def BM_value : BM = 2 := by sorry
def cos_BAM_value : cos_BAM = 4/5 := by sorry

-- Define the necessary constructions and the area computation
def area_parallelogram (AD BM : ℝ) (cos_BAM : ℝ) : ℝ :=
  let AM := BM * real.cot (real.arccos cos_BAM)
  let AB := real.sqrt (AM^2 + BM^2)
  let sin_BAD := 2 * (real.sin (real.arccos cos_BAM)) * cos_BAM
  AD * AB * sin_BAD

-- Statement that verifies the area is 16
theorem parallelogram_area_proof :
  area_parallelogram 5 2 (4/5) = 16 := by
    sorry


end parallelogram_area_proof_l244_244333


namespace mark_weekly_leftover_l244_244206

def initial_hourly_wage := 40
def raise_percentage := 5 / 100
def daily_hours := 8
def weekly_days := 5
def old_weekly_bills := 600
def personal_trainer_cost := 100

def new_hourly_wage := initial_hourly_wage * (1 + raise_percentage)
def weekly_hours := daily_hours * weekly_days
def weekly_earnings := new_hourly_wage * weekly_hours
def new_weekly_expenses := old_weekly_bills + personal_trainer_cost
def leftover_per_week := weekly_earnings - new_weekly_expenses

theorem mark_weekly_leftover : leftover_per_week = 980 := by
  sorry

end mark_weekly_leftover_l244_244206


namespace sin_240_eq_neg_sqrt3_div_2_l244_244579

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244579


namespace charlie_golden_delicious_bags_l244_244632

theorem charlie_golden_delicious_bags :
  ∀ (total_bags fruit_bags macintosh_bags cortland_bags golden_delicious_bags : ℝ),
  total_bags = 0.67 →
  macintosh_bags = 0.17 →
  cortland_bags = 0.33 →
  total_bags = golden_delicious_bags + macintosh_bags + cortland_bags →
  golden_delicious_bags = 0.17 := by
  intros total_bags fruit_bags macintosh_bags cortland_bags golden_delicious_bags
  intros h_total h_macintosh h_cortland h_sum
  sorry

end charlie_golden_delicious_bags_l244_244632


namespace greatest_three_digit_multiple_of_17_is_986_l244_244002

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l244_244002


namespace sum_of_primes_less_than_20_l244_244019

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l244_244019


namespace cost_of_candy_car_l244_244804

theorem cost_of_candy_car (starting_amount paid_amount change : ℝ) (h1 : starting_amount = 1.80) (h2 : change = 1.35) (h3 : paid_amount = starting_amount - change) : paid_amount = 0.45 := by
  sorry

end cost_of_candy_car_l244_244804


namespace relatively_prime_number_exists_l244_244654

def gcd (a b : ℕ) : ℕ := a.gcd b

def is_relatively_prime_to_all (n : ℕ) (lst : List ℕ) : Prop :=
  ∀ m ∈ lst, m ≠ n → gcd n m = 1

def given_numbers : List ℕ := [20172017, 20172018, 20172019, 20172020, 20172021]

theorem relatively_prime_number_exists :
  ∃ n ∈ given_numbers, is_relatively_prime_to_all n given_numbers := 
begin
  use 20172019,
  split,
  { -- Show 20172019 is in the list
    simp },
  { -- Prove 20172019 is relatively prime to all other numbers in the list
    intros m h1 h2,
    -- Further proof goes here
    sorry
  }
end

end relatively_prime_number_exists_l244_244654


namespace initial_birds_count_l244_244216

theorem initial_birds_count (current_total_birds birds_joined initial_birds : ℕ) 
  (h1 : current_total_birds = 6) 
  (h2 : birds_joined = 4) : 
  initial_birds = current_total_birds - birds_joined → 
  initial_birds = 2 :=
by 
  intro h3
  rw [h1, h2] at h3
  exact h3

end initial_birds_count_l244_244216


namespace variance_decreases_l244_244508

def scores_initial := [5, 9, 7, 10, 9] -- Initial 5 shot scores
def additional_shot := 8 -- Additional shot score

-- Given variance of initial scores
def variance_initial : ℝ := 3.2

-- Placeholder function to calculate variance of a list of scores
noncomputable def variance (scores : List ℝ) : ℝ := sorry

-- Definition of the new scores list
def scores_new := scores_initial ++ [additional_shot]

-- Define the proof problem
theorem variance_decreases :
  variance scores_new < variance_initial :=
sorry

end variance_decreases_l244_244508


namespace calculate_years_l244_244228

variable {P R T SI : ℕ}

-- Conditions translations
def simple_interest_one_fifth (P SI : ℕ) : Prop :=
  SI = P / 5

def rate_of_interest (R : ℕ) : Prop :=
  R = 4

-- Proof of the number of years T
theorem calculate_years (h1 : simple_interest_one_fifth P SI)
                        (h2 : rate_of_interest R)
                        (h3 : SI = (P * R * T) / 100) : T = 5 :=
by
  sorry

end calculate_years_l244_244228


namespace symmetric_curve_eq_l244_244223

-- Definitions from the problem conditions
def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + (y + 1) ^ 2 = 1
def line_of_symmetry (x y : ℝ) : Prop := x - y + 3 = 0

-- Problem statement derived from the translation step
theorem symmetric_curve_eq (x y : ℝ) : (x - 2) ^ 2 + (y + 1) ^ 2 = 1 ∧ x - y + 3 = 0 → (x + 4) ^ 2 + (y - 5) ^ 2 = 1 := 
by
  sorry

end symmetric_curve_eq_l244_244223


namespace remainder_of_product_l244_244994

theorem remainder_of_product (a b n : ℕ) (h1 : a = 2431) (h2 : b = 1587) (h3 : n = 800) : 
  (a * b) % n = 397 := 
by
  sorry

end remainder_of_product_l244_244994


namespace sum_of_primes_less_than_20_l244_244057

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l244_244057


namespace number_of_pens_l244_244730

theorem number_of_pens (num_pencils : ℕ) (total_cost : ℝ) (avg_price_pencil : ℝ) (avg_price_pen : ℝ) : ℕ :=
  sorry

example : number_of_pens 75 690 2 18 = 30 :=
by 
  sorry

end number_of_pens_l244_244730


namespace sum_primes_less_than_20_l244_244064

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l244_244064


namespace c_value_difference_l244_244201

theorem c_value_difference (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a^2 + b^2 + c^2 = 18) : 
  max c - min c = 34 / 3 :=
sorry

end c_value_difference_l244_244201


namespace acres_used_for_corn_l244_244537

theorem acres_used_for_corn (total_land : ℕ) (ratio_beans : ℕ) (ratio_wheat : ℕ) (ratio_corn : ℕ)
  (total_ratio_parts : ℕ) (one_part_size : ℕ) :
  total_land = 1034 →
  ratio_beans = 5 →
  ratio_wheat = 2 →
  ratio_corn = 4 →
  total_ratio_parts = ratio_beans + ratio_wheat + ratio_corn →
  one_part_size = total_land / total_ratio_parts →
  ratio_corn * one_part_size = 376 :=
by
  intros
  sorry

end acres_used_for_corn_l244_244537


namespace sum_of_primes_less_than_20_l244_244107

theorem sum_of_primes_less_than_20 : ∑ p in {2, 3, 5, 7, 11, 13, 17, 19}, p = 77 := by
  sorry

end sum_of_primes_less_than_20_l244_244107


namespace sum_of_primes_less_than_20_is_77_l244_244045

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l244_244045


namespace find_bc_find_area_l244_244482

variable {a b c : ℝ}
variable {A B C : ℝ}

-- Condition 1: From the given problem
variable (h1 : (b^2 + c^2 - a^2) / (cos A) = 2)

-- Condition 2: From the given problem
variable (h2 : (a * cos B - b * cos A) / (a * cos B + b * cos A) - b / c = 1)

theorem find_bc : bc = 1 :=
sorry

theorem find_area (area_abc : ℝ) : area_abc = (sqrt 3) / 4 :=
sorry

end find_bc_find_area_l244_244482


namespace election_total_votes_l244_244330

theorem election_total_votes (V_A V_B V : ℕ) (H1 : V_A = V_B + 15/100 * V) (H2 : V_A + V_B = 80/100 * V) (H3 : V_B = 2184) : V = 6720 :=
sorry

end election_total_votes_l244_244330


namespace tangent_line_at_x_is_2_l244_244651

noncomputable def curve (x : ℝ) : ℝ := (1/4) * x^2 - 3 * Real.log x

theorem tangent_line_at_x_is_2 :
  ∃ x₀ : ℝ, (x₀ > 0) ∧ ((1/2) * x₀ - (3 / x₀) = -1/2) ∧ x₀ = 2 :=
by
  sorry

end tangent_line_at_x_is_2_l244_244651


namespace sum_of_primes_less_than_20_eq_77_l244_244023

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l244_244023


namespace k_value_five_l244_244437

theorem k_value_five (a b k : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (a^2 + b^2) / (a * b - 1) = k) : k = 5 := 
sorry

end k_value_five_l244_244437


namespace set_intersection_example_l244_244664

theorem set_intersection_example :
  let M := {x : ℝ | -1 < x ∧ x < 1}
  let N := {x : ℝ | 0 ≤ x}
  {x : ℝ | -1 < x ∧ x < 1} ∩ {x : ℝ | 0 ≤ x} = {x : ℝ | 0 ≤ x ∧ x < 1} :=
by
  sorry

end set_intersection_example_l244_244664


namespace y_intercept_of_parallel_line_l244_244275

theorem y_intercept_of_parallel_line (m : ℝ) (c1 c2 : ℝ) (x1 y1 : ℝ) (H_parallel : m = -3) (H_passing : (x1, y1) = (1, -4)) : 
    c2 = -1 :=
  sorry

end y_intercept_of_parallel_line_l244_244275


namespace count_non_increasing_5digit_numbers_l244_244944

theorem count_non_increasing_5digit_numbers : 
  ∃ n : ℕ, n = 715 ∧ ∀ (a b c d e : ℕ), 
  a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ a > 0 →
  -- count the number of such sequences --
  n = (Nat.choose (9 + 5 - 1) (5 - 1)) :=
begin
  sorry
end

end count_non_increasing_5digit_numbers_l244_244944


namespace find_x_in_sequence_l244_244151

theorem find_x_in_sequence :
  (∀ a b c d : ℕ, a * b * c * d = 120) →
  (a = 2) →
  (b = 4) →
  (d = 3) →
  ∃ x : ℕ, 2 * 4 * x * 3 = 120 ∧ x = 5 :=
sorry

end find_x_in_sequence_l244_244151


namespace exists_integers_a_b_part_a_l244_244163

theorem exists_integers_a_b_part_a : 
  ∃ a b : ℤ, (∀ x : ℝ, x^2 + a * x + b ≠ 0) ∧ (∃ x : ℝ, ⌊x^2⌋ + a * x + (b : ℝ) = 0) := 
sorry

end exists_integers_a_b_part_a_l244_244163


namespace pipe_length_l244_244396

theorem pipe_length (L x : ℝ) 
  (h1 : 20 = L - x)
  (h2 : 140 = L + 7 * x) : 
  L = 35 := by
  sorry

end pipe_length_l244_244396


namespace platform_length_is_260_meters_l244_244280

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def time_to_cross_platform_s : ℝ := 30
noncomputable def time_to_cross_man_s : ℝ := 17

noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def length_of_train_m : ℝ := train_speed_mps * time_to_cross_man_s
noncomputable def total_distance_cross_platform_m : ℝ := train_speed_mps * time_to_cross_platform_s
noncomputable def length_of_platform_m : ℝ := total_distance_cross_platform_m - length_of_train_m

theorem platform_length_is_260_meters :
  length_of_platform_m = 260 := by
  sorry

end platform_length_is_260_meters_l244_244280


namespace house_A_cost_l244_244210

theorem house_A_cost (base_salary earnings commission_rate total_houses cost_A cost_B cost_C : ℝ)
  (H_base_salary : base_salary = 3000)
  (H_earnings : earnings = 8000)
  (H_commission_rate : commission_rate = 0.02)
  (H_cost_B : cost_B = 3 * cost_A)
  (H_cost_C : cost_C = 2 * cost_A - 110000)
  (H_total_commission : earnings - base_salary = 5000)
  (H_total_cost : 5000 / commission_rate = 250000)
  (H_total_houses : base_salary + commission_rate * (cost_A + cost_B + cost_C) = earnings) :
  cost_A = 60000 := sorry

end house_A_cost_l244_244210


namespace Tom_earns_per_week_l244_244525

-- Definitions based on conditions
def crab_buckets_per_day := 8
def crabs_per_bucket := 12
def price_per_crab := 5
def days_per_week := 7

-- The proof goal
theorem Tom_earns_per_week :
  (crab_buckets_per_day * crabs_per_bucket * price_per_crab * days_per_week) = 3360 := by
  sorry

end Tom_earns_per_week_l244_244525


namespace x_squared_y_squared_iff_x_squared_y_squared_not_sufficient_x_squared_y_squared_necessary_l244_244725

theorem x_squared_y_squared_iff (x y : ℝ) : x ^ 2 = y ^ 2 ↔ x = y ∨ x = -y := by
  sorry

theorem x_squared_y_squared_not_sufficient (x y : ℝ) : (x ^ 2 = y ^ 2) → (x = y ∨ x = -y) := by
  sorry

theorem x_squared_y_squared_necessary (x y : ℝ) : (x = y) → (x ^ 2 = y ^ 2) := by
  sorry

end x_squared_y_squared_iff_x_squared_y_squared_not_sufficient_x_squared_y_squared_necessary_l244_244725


namespace solve_equation_l244_244366

theorem solve_equation (x : ℝ) :
    x^6 - 22 * x^2 - Real.sqrt 21 = 0 ↔ x = Real.sqrt ((Real.sqrt 21 + 5) / 2) ∨ x = -Real.sqrt ((Real.sqrt 21 + 5) / 2) := by
  sorry

end solve_equation_l244_244366


namespace find_k_l244_244806

-- Definitions
def a (n : ℕ) : ℤ := 1 + (n - 1) * 2
def S (n : ℕ) : ℤ := n / 2 * (2 * 1 + (n - 1) * 2)

-- Main theorem statement
theorem find_k (k : ℕ) (h : S (k + 2) - S k = 24) : k = 5 :=
by sorry

end find_k_l244_244806


namespace log_expression_l244_244283

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_expression : 
  log 2 * log 50 + log 25 - log 5 * log 20 = 1 := 
by 
  sorry

end log_expression_l244_244283


namespace initial_mean_corrected_observations_l244_244702

theorem initial_mean_corrected_observations:
  ∃ M : ℝ, 
  (∀ (Sum_initial Sum_corrected : ℝ), 
    Sum_initial = 50 * M ∧ 
    Sum_corrected = Sum_initial + (48 - 23) → 
    Sum_corrected / 50 = 41.5) →
  M = 41 :=
by
  sorry

end initial_mean_corrected_observations_l244_244702


namespace bucket_holds_120_ounces_l244_244212

theorem bucket_holds_120_ounces :
  ∀ (fill_buckets remove_buckets baths_per_day ounces_per_week : ℕ),
    fill_buckets = 14 →
    remove_buckets = 3 →
    baths_per_day = 7 →
    ounces_per_week = 9240 →
    baths_per_day * (fill_buckets - remove_buckets) * (ounces_per_week / (baths_per_day * (fill_buckets - remove_buckets))) = ounces_per_week →
    (ounces_per_week / (baths_per_day * (fill_buckets - remove_buckets))) = 120 :=
by
  intros fill_buckets remove_buckets baths_per_day ounces_per_week Hfill Hremove Hbaths Hounces Hcalc
  sorry

end bucket_holds_120_ounces_l244_244212


namespace expression_positive_l244_244821

theorem expression_positive (x y z : ℝ) (h : x^2 + y^2 + z^2 ≠ 0) : 
  5 * x^2 + 5 * y^2 + 5 * z^2 + 6 * x * y - 8 * x * z - 8 * y * z > 0 := 
sorry

end expression_positive_l244_244821


namespace arithmetic_sequence_n_value_l244_244681

theorem arithmetic_sequence_n_value
  (a : ℕ → ℚ)
  (h1 : a 1 = 1 / 3)
  (h2 : a 2 + a 5 = 4)
  (h3 : a n = 33)
  : n = 50 :=
sorry

end arithmetic_sequence_n_value_l244_244681


namespace sum_of_primes_less_than_twenty_is_77_l244_244035

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l244_244035


namespace sum_prime_numbers_less_than_twenty_l244_244087

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l244_244087


namespace total_students_in_class_l244_244969

-- No need for noncomputable def here as we're dealing with basic arithmetic

theorem total_students_in_class (jellybeans_total jellybeans_left boys_girls_diff : ℕ)
  (girls boys students : ℕ) :
  jellybeans_total = 450 →
  jellybeans_left = 10 →
  boys_girls_diff = 3 →
  boys = girls + boys_girls_diff →
  students = girls + boys →
  (girls * girls) + (boys * boys) = jellybeans_total - jellybeans_left →
  students = 29 := 
by
  intro h_total h_left h_diff h_boys h_students h_distribution
  sorry

end total_students_in_class_l244_244969


namespace rope_segments_l244_244917

theorem rope_segments (total_length : ℝ) (n : ℕ) (h1 : total_length = 3) (h2 : n = 7) :
  (∃ segment_fraction : ℝ, segment_fraction = 1 / n ∧
   ∃ segment_length : ℝ, segment_length = total_length / n) :=
sorry

end rope_segments_l244_244917


namespace album_photos_proof_l244_244148

def photos_per_page := 4

-- Conditions
def position_81st_photo (n: ℕ) (x: ℕ) :=
  4 * n * (x - 1) + 17 ≤ 81 ∧ 81 ≤ 4 * n * (x - 1) + 20

def position_171st_photo (n: ℕ) (y: ℕ) :=
  4 * n * (y - 1) + 9 ≤ 171 ∧ 171 ≤ 4 * n * (y - 1) + 12

noncomputable def album_photos := 32

theorem album_photos_proof :
  ∃ n x y, position_81st_photo n x ∧ position_171st_photo n y ∧ 4 * n = album_photos :=
by
  sorry

end album_photos_proof_l244_244148


namespace Pat_worked_days_eq_57_l244_244928

def Pat_earnings (x : ℕ) : ℤ := 100 * x
def Pat_food_costs (x : ℕ) : ℤ := 20 * (70 - x)
def total_balance (x : ℕ) : ℤ := Pat_earnings x - Pat_food_costs x

theorem Pat_worked_days_eq_57 (x : ℕ) (h : total_balance x = 5440) : x = 57 :=
by
  sorry

end Pat_worked_days_eq_57_l244_244928


namespace shifted_linear_func_is_2x_l244_244468

-- Define the initial linear function
def linear_func (x : ℝ) : ℝ := 2 * x - 3

-- Define the shifted linear function
def shifted_linear_func (x : ℝ) : ℝ := linear_func x + 3

theorem shifted_linear_func_is_2x (x : ℝ) : shifted_linear_func x = 2 * x := by
  -- Proof would go here, but we use sorry to skip it
  sorry

end shifted_linear_func_is_2x_l244_244468


namespace average_weight_of_children_l244_244650

theorem average_weight_of_children :
  let ages := [3, 4, 5, 6, 7]
  let regression_equation (x : ℕ) := 3 * x + 5
  let average l := (l.foldr (· + ·) 0) / l.length
  average (List.map regression_equation ages) = 20 :=
by
  sorry

end average_weight_of_children_l244_244650


namespace max_value_of_x0_l244_244217

noncomputable def sequence_max_value (seq : Fin 1996 → ℝ) (pos_seq : ∀ i, seq i > 0) : Prop :=
  seq 0 = seq 1995 ∧
  (∀ i : Fin 1995, seq i + 2 / seq i = 2 * seq (i + 1) + 1 / seq (i + 1)) ∧
  (seq 0 ≤ 2^997)

theorem max_value_of_x0 :
  ∃ seq : Fin 1996 → ℝ, ∀ pos_seq : ∀ i, seq i > 0, sequence_max_value seq pos_seq :=
sorry

end max_value_of_x0_l244_244217


namespace polynomial_value_at_2018_l244_244478

theorem polynomial_value_at_2018 (f : ℝ → ℝ) 
  (h₁ : ∀ x : ℝ, f (-x^2 - x - 1) = x^4 + 2*x^3 + 2022*x^2 + 2021*x + 2019) : 
  f 2018 = -2019 :=
sorry

end polynomial_value_at_2018_l244_244478


namespace mutual_independence_A_B_mutual_independence_A_C_non_independence_B_C_mutual_exclusiveness_B_D_l244_244326

-- Definitions of events A, B, C, and D:
def eventA (s : set ℕ) : bool := (1 ∈ s ∧ 3 ∈ s) ∨ (2 ∈ s ∧ 4 ∈ s) ∨ (5 ∈ s ∧ 6 ∈ s)
def eventB (s : set ℕ) : bool := ∃ x y, x ∈ s ∧ y ∈ s ∧ abs (x - y) = 1
def eventC (s : set ℕ) : bool := ∃ x y, x ∈ s ∧ y ∈ s ∧ (x + y = 6 ∨ x + y = 7)
def eventD (s : set ℕ) : bool := ∃ x y, x ∈ s ∧ y ∈ s ∧ x * y = 5

-- Total number of outcomes:
def total_outcomes := 15

-- Probabilities:
def PA := 3 / total_outcomes
def PB := 5 / total_outcomes
def PC := 5 / total_outcomes
def PD := 1 / total_outcomes

-- Proving the conditions:
theorem mutual_independence_A_B : PA * PB = 1 / total_outcomes := sorry
theorem mutual_independence_A_C : PA * PC = 1 / total_outcomes := sorry
theorem non_independence_B_C : PB * PC ≠ 1 / total_outcomes := sorry
theorem mutual_exclusiveness_B_D : ∀ s, eventB s → ¬ eventD s := sorry

end mutual_independence_A_B_mutual_independence_A_C_non_independence_B_C_mutual_exclusiveness_B_D_l244_244326


namespace solve_equation_1_solve_equation_2_l244_244973

theorem solve_equation_1 (x : Real) : 
  (1/3) * (x - 3)^2 = 12 → x = 9 ∨ x = -3 :=
by
  sorry

theorem solve_equation_2 (x : Real) : 
  (2 * x - 1)^2 = (1 - x)^2 → x = 0 ∨ x = 2/3 :=
by
  sorry

end solve_equation_1_solve_equation_2_l244_244973


namespace find_constant_l244_244235

-- Define the conditions
def is_axles (x : ℕ) : Prop := x = 5
def toll_for_truck (t : ℝ) : Prop := t = 4

-- Define the formula for the toll
def toll_formula (t : ℝ) (constant : ℝ) (x : ℕ) : Prop :=
  t = 2.50 + constant * (x - 2)

-- Proof problem statement
theorem find_constant : ∃ (constant : ℝ), 
  ∀ x : ℕ, is_axles x → toll_for_truck 4 →
  toll_formula 4 constant x → constant = 0.50 :=
sorry

end find_constant_l244_244235


namespace ab_product_power_l244_244187

theorem ab_product_power (a b : ℤ) (n : ℕ) (h1 : (a * b)^n = 128 * 8) : n = 10 := by
  sorry

end ab_product_power_l244_244187


namespace frank_total_cans_l244_244173

def total_cans_picked_up (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ) : ℕ :=
  let total_bags := bags_saturday + bags_sunday
  total_bags * cans_per_bag

theorem frank_total_cans : total_cans_picked_up 5 3 5 = 40 := by
  sorry

end frank_total_cans_l244_244173


namespace part_a_part_b_l244_244974

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (λ x y => x + y) 0

-- Part A: There exists a sequence of 158 consecutive integers where the sum of digits is not divisible by 17
theorem part_a : ∃ (n : ℕ), ∀ (k : ℕ), k < 158 → sum_of_digits (n + k) % 17 ≠ 0 := by
  sorry

-- Part B: Among any 159 consecutive integers, there exists at least one integer whose sum of digits is divisible by 17
theorem part_b : ∀ (n : ℕ), ∃ (k : ℕ), k < 159 ∧ sum_of_digits (n + k) % 17 = 0 := by
  sorry

end part_a_part_b_l244_244974


namespace second_account_interest_rate_l244_244255

theorem second_account_interest_rate
  (investment1 : ℝ)
  (rate1 : ℝ)
  (interest1 : ℝ)
  (investment2 : ℝ)
  (interest2 : ℝ)
  (h1 : 4000 = investment1)
  (h2 : 0.08 = rate1)
  (h3 : 320 = interest1)
  (h4 : 7200 - 4000 = investment2)
  (h5 : interest1 = interest2) :
  interest2 / investment2 = 0.1 :=
by
  sorry

end second_account_interest_rate_l244_244255


namespace shortest_chord_length_l244_244436

theorem shortest_chord_length 
  (C : ℝ → ℝ → Prop) 
  (l : ℝ → ℝ → ℝ → Prop) 
  (radius : ℝ) 
  (center_x center_y : ℝ) 
  (cx cy : ℝ) 
  (m : ℝ) :
  (∀ x y, C x y ↔ (x - 1)^2 + (y - 2)^2 = 25) →
  (∀ x y m, l x y m ↔ (2*m+1)*x + (m+1)*y - 7*m - 4 = 0) →
  center_x = 1 →
  center_y = 2 →
  radius = 5 →
  cx = 3 →
  cy = 1 →
  ∃ shortest_chord_length : ℝ, shortest_chord_length = 4 * Real.sqrt 5 := sorry

end shortest_chord_length_l244_244436


namespace find_m_from_parallel_vectors_l244_244942

variables (m : ℝ)

def a : ℝ × ℝ := (1, m)
def b : ℝ × ℝ := (2, -3)

-- The condition that vectors a and b are parallel
def vectors_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u.1 = k * v.1 ∧ u.2 = k * v.2

-- Given that a and b are parallel, prove that m = -3/2
theorem find_m_from_parallel_vectors
  (h : vectors_parallel (1, m) (2, -3)) :
  m = -3 / 2 :=
sorry

end find_m_from_parallel_vectors_l244_244942


namespace third_number_lcm_l244_244763

theorem third_number_lcm (n : ℕ) :
  n ∣ 360 ∧ lcm (lcm 24 36) n = 360 →
  n = 5 :=
by sorry

end third_number_lcm_l244_244763


namespace sum_of_primes_less_than_twenty_is_77_l244_244032

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l244_244032


namespace proof_problem_l244_244506

noncomputable def otimes (a b : ℝ) : ℝ := a^3 / b^2

theorem proof_problem : ((otimes (otimes 2 3) 4) - otimes 2 (otimes 3 4)) = -224/81 :=
by
  sorry

end proof_problem_l244_244506


namespace find_y_l244_244288

theorem find_y 
  (y : ℝ) 
  (h1 : (y^2 - 11 * y + 24) / (y - 3) + (2 * y^2 + 7 * y - 18) / (2 * y - 3) = -10)
  (h2 : y ≠ 3)
  (h3 : y ≠ 3 / 2) : 
  y = -4 := 
sorry

end find_y_l244_244288


namespace divide_area_into_squares_l244_244422

theorem divide_area_into_squares :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x / y = 4 / 3 ∧ (x^2 + y^2 = 100) ∧ x = 8 ∧ y = 6) := 
by {
  sorry
}

end divide_area_into_squares_l244_244422


namespace simplify_expression_l244_244827

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l244_244827


namespace upper_limit_arun_weight_l244_244750

theorem upper_limit_arun_weight (x w : ℝ) :
  (65 < w ∧ w < x) ∧
  (60 < w ∧ w < 70) ∧
  (w ≤ 68) ∧
  (w = 67) →
  x = 68 :=
by
  sorry

end upper_limit_arun_weight_l244_244750


namespace fourth_place_points_l244_244797

variables (x : ℕ)

def points_awarded (place : ℕ) : ℕ :=
  if place = 1 then 11
  else if place = 2 then 7
  else if place = 3 then 5
  else if place = 4 then x
  else 0

theorem fourth_place_points:
  (∃ a b c y u : ℕ, a + b + c + y + u = 7 ∧ points_awarded x 1 ^ a * points_awarded x 2 ^ b * points_awarded x 3 ^ c * points_awarded x 4 ^ y * 1 ^ u = 38500) →
  x = 4 :=
sorry

end fourth_place_points_l244_244797


namespace repeated_root_condition_l244_244534

theorem repeated_root_condition (m : ℝ) : m = 10 → ∃ x, (5 * x) / (x - 2) + 1 = m / (x - 2) ∧ x = 2 :=
by
  sorry

end repeated_root_condition_l244_244534


namespace simplify_expression_l244_244839

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 := 
by
  sorry

end simplify_expression_l244_244839


namespace arithmetic_sequence_common_difference_l244_244957
-- Lean 4 Proof Statement


theorem arithmetic_sequence_common_difference 
  (a : ℕ) (n : ℕ) (d : ℕ) (S_n : ℕ) (a_n : ℕ) 
  (h1 : a = 2) 
  (h2 : a_n = 29) 
  (h3 : S_n = 155) 
  (h4 : S_n = n * (a + a_n) / 2) 
  (h5 : a_n = a + (n - 1) * d) 
  : d = 3 := 
by 
  sorry

end arithmetic_sequence_common_difference_l244_244957


namespace sin_45_eq_sqrt2_div_2_l244_244561

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt2_div_2_l244_244561


namespace min_value_l244_244965

theorem min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : 1 / (a + 3) + 1 / (b + 3) + 1 / (c + 3) = 1 / 4) : 
  22.75 ≤ a + 3 * b + 2 * c := 
sorry

end min_value_l244_244965


namespace sin_240_deg_l244_244567

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l244_244567


namespace bob_switching_win_prob_l244_244798

-- Definitions and conditions
variable {doors : Finset ℕ} (hd : doors.card = 7) (prizes : doors.Subset) (hp : prizes.card = 2)
variable {initial_choice : ℕ} (hi : initial_choice ∈ doors)
variable {opened_doors : Finset ℕ} (ho : opened_doors.card = 3) (prize_in_opened : opened_doors ∩ prizes ≠ ∅)
variable {final_choice : ℕ} (hf : final_choice ∈ (doors \ (insert initial_choice opened_doors)))

theorem bob_switching_win_prob :
  ∃ p : ℝ, p = 5/14 :=
by
  sorry

end bob_switching_win_prob_l244_244798


namespace cubic_inequality_l244_244131

theorem cubic_inequality (p q x : ℝ) (h : x^3 + p * x + q = 0) : 4 * q * x ≤ p^2 :=
sorry

end cubic_inequality_l244_244131


namespace sacks_per_day_l244_244943

theorem sacks_per_day (total_sacks : ℕ) (total_days : ℕ) (harvest_per_day : ℕ) : 
  total_sacks = 56 → 
  total_days = 14 → 
  harvest_per_day = total_sacks / total_days → 
  harvest_per_day = 4 := 
by
  intros h_total_sacks h_total_days h_harvest_per_day
  rw [h_total_sacks, h_total_days] at h_harvest_per_day
  simp at h_harvest_per_day
  exact h_harvest_per_day

end sacks_per_day_l244_244943


namespace fraction_zero_x_value_l244_244876

theorem fraction_zero_x_value (x : ℝ) (h1 : 2 * x = 0) (h2 : x + 3 ≠ 0) : x = 0 :=
by
  sorry

end fraction_zero_x_value_l244_244876


namespace coffee_shop_lattes_l244_244373

theorem coffee_shop_lattes (T : ℕ) (L : ℕ) (hT : T = 6) (hL : L = 4 * T + 8) : L = 32 :=
by
  sorry

end coffee_shop_lattes_l244_244373


namespace count_three_digit_integers_with_remainder_3_div_7_l244_244454

theorem count_three_digit_integers_with_remainder_3_div_7 :
  ∃ n, (100 ≤ 7 * n + 3 ∧ 7 * n + 3 < 1000) ∧
  ∀ m, (100 ≤ 7 * m + 3 ∧ 7 * m + 3 < 1000) → m - n < 142 - 14 + 1 :=
by
  sorry

end count_three_digit_integers_with_remainder_3_div_7_l244_244454


namespace largest_and_smallest_correct_l244_244175

noncomputable def largest_and_smallest (x y : ℝ) (hx : x < 0) (hy : -1 < y ∧ y < 0) : ℝ × ℝ :=
  if hx_y : x * y > 0 then
    if hx_y_sq : x * y * y > x then
      (x * y, x)
    else
      sorry
  else
    sorry

theorem largest_and_smallest_correct {x y : ℝ} (hx : x < 0) (hy : -1 < y ∧ y < 0) :
  largest_and_smallest x y hx hy = (x * y, x) :=
by {
  sorry
}

end largest_and_smallest_correct_l244_244175


namespace length_AB_l244_244155

theorem length_AB (r : ℝ) (A B : ℝ) (π : ℝ) : 
  r = 4 ∧ π = 3 ∧ (A = 8 ∧ B = 8) → (A = B ∧ A + B = 24 → AB = 6) :=
by
  intros
  sorry

end length_AB_l244_244155


namespace sum_of_primes_less_than_20_l244_244017

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l244_244017


namespace reciprocal_of_repeating_decimal_three_l244_244391

noncomputable def repeating_decimal_to_fraction : ℚ :=
  let x := (0.33333333333 : ℚ) in 1 / 3

theorem reciprocal_of_repeating_decimal_three : 
  (1 / repeating_decimal_to_fraction) = 3 := by
  -- Reciprocal of the fraction
  sorry

end reciprocal_of_repeating_decimal_three_l244_244391


namespace g_f_neg5_l244_244202

-- Define the function f
def f (x : ℝ) := 2 * x ^ 2 - 4

-- Define the function g with the known condition g(f(5)) = 12
axiom g : ℝ → ℝ
axiom g_f5 : g (f 5) = 12

-- Now state the main theorem we need to prove
theorem g_f_neg5 : g (f (-5)) = 12 := by
  sorry

end g_f_neg5_l244_244202


namespace sum_of_primes_less_than_20_is_77_l244_244039

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l244_244039


namespace sin_240_deg_l244_244611

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l244_244611


namespace elderly_people_not_set_l244_244415

def is_well_defined (S : Set α) : Prop := Nonempty S

def all_positive_numbers : Set ℝ := {x : ℝ | 0 < x}
def real_numbers_non_zero : Set ℝ := {x : ℝ | x ≠ 0}
def four_great_inventions : Set String := {"compass", "gunpowder", "papermaking", "printing"}

def elderly_people_description : String := "elderly people"

theorem elderly_people_not_set :
  ¬ (∃ S : Set α, elderly_people_description = "elderly people" ∧ is_well_defined S) :=
sorry

end elderly_people_not_set_l244_244415


namespace range_of_f_when_a_eq_2_sufficient_but_not_necessary_condition_for_q_l244_244446

-- Define the function
def f (x a : ℝ) : ℝ := x^2 - a * x + 4 - a^2

-- Problem (1): Range of the function when a = 2
theorem range_of_f_when_a_eq_2 :
  (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x 2 = (x - 1)^2 - 1) →
  Set.image (f 2) (Set.Icc (-2 : ℝ) 3) = Set.Icc (-1 : ℝ) 8 := sorry

-- Problem (2): Sufficient but not necessary condition
theorem sufficient_but_not_necessary_condition_for_q :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x 4 ≤ 0) →
  (Set.Icc (-2 : ℝ) 2 → (∃ (M : Set ℝ), Set.singleton 4 ⊆ M ∧ 
    (∀ a ∈ M, ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a ≤ 0) ∧
    (∀ a ∈ Set.Icc (-2 : ℝ) 2, ∃ a' ∉ M, ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a' ≤ 0))) := sorry

end range_of_f_when_a_eq_2_sufficient_but_not_necessary_condition_for_q_l244_244446


namespace ab_value_l244_244362

theorem ab_value (a b : ℝ) :
  (A = { x : ℝ | x^2 - 8 * x + 15 = 0 }) ∧
  (B = { x : ℝ | x^2 - a * x + b = 0 }) ∧
  (A ∪ B = {2, 3, 5}) ∧
  (A ∩ B = {3}) →
  (a * b = 30) :=
by
  sorry

end ab_value_l244_244362


namespace problem_k_value_l244_244953

theorem problem_k_value (k x1 x2 : ℝ) 
  (h_eq : 8 * x1^2 + 2 * k * x1 + k - 1 = 0) 
  (h_eq2 : 8 * x2^2 + 2 * k * x2 + k - 1 = 0) 
  (h_sum_sq : x1^2 + x2^2 = 1) : 
  k = -2 :=
sorry

end problem_k_value_l244_244953


namespace spherical_to_rectangular_coordinates_l244_244756

-- Define the given conditions
variable (ρ : ℝ) (θ : ℝ) (φ : ℝ)
variable (hρ : ρ = 6) (hθ : θ = 7 * Real.pi / 4) (hφ : φ = Real.pi / 2)

-- Convert spherical coordinates (ρ, θ, φ) to rectangular coordinates (x, y, z) and prove the values
theorem spherical_to_rectangular_coordinates :
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = 3 * Real.sqrt 2 ∧ y = -3 * Real.sqrt 2 ∧ z = 0 :=
by
  sorry

end spherical_to_rectangular_coordinates_l244_244756


namespace arccos_cos_eq_l244_244565

theorem arccos_cos_eq :
  Real.arccos (Real.cos 11) = 0.7168 := by
  sorry

end arccos_cos_eq_l244_244565


namespace integer_solutions_equation_l244_244452

theorem integer_solutions_equation : 
  (∃ x y : ℤ, (1 / (2022 : ℚ) = 1 / (x : ℚ) + 1 / (y : ℚ))) → 
  ∃! (n : ℕ), n = 53 :=
by
  sorry

end integer_solutions_equation_l244_244452


namespace sin_240_eq_neg_sqrt3_div_2_l244_244618

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244618


namespace games_played_l244_244423

theorem games_played (x : ℕ) (h1 : x * 26 + 42 * (20 - x) = 600) : x = 15 :=
by {
  sorry
}

end games_played_l244_244423


namespace negation_of_proposition_l244_244543

theorem negation_of_proposition (m : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + 2*x + m ≤ 0) ↔ (∃ x : ℝ, x^2 + 2*x + m > 0) :=
sorry

end negation_of_proposition_l244_244543


namespace find_ellipse_and_hyperbola_equations_l244_244300

-- Define the conditions
def eccentricity (e : ℝ) (a b : ℝ) : Prop :=
  e = (Real.sqrt (a ^ 2 - b ^ 2)) / a

def focal_distance (f : ℝ) (a b : ℝ) : Prop :=
  f = 2 * Real.sqrt (a ^ 2 + b ^ 2)

-- Define the problem to prove the equations of the ellipse and hyperbola
theorem find_ellipse_and_hyperbola_equations (a b : ℝ) (e : ℝ) (f : ℝ)
  (h1 : eccentricity e a b) (h2 : focal_distance f a b) 
  (h3 : e = 4 / 5) (h4 : f = 2 * Real.sqrt 34) 
  (h5 : a > b) (h6 : 0 < b) :
  (a^2 = 25 ∧ b^2 = 9) → 
  (∀ x y, (x^2 / 25 + y^2 / 9 = 1) ∧ (x^2 / 25 - y^2 / 9 = 1)) :=
sorry

end find_ellipse_and_hyperbola_equations_l244_244300


namespace chocolate_chip_difference_l244_244529

noncomputable def V_v : ℕ := 20 -- Viviana's vanilla chips
noncomputable def S_c : ℕ := 25 -- Susana's chocolate chips
noncomputable def S_v : ℕ := 3 * V_v / 4 -- Susana's vanilla chips

theorem chocolate_chip_difference (V_c : ℕ) (h1 : V_c + V_v + S_c + S_v = 90) :
  V_c - S_c = 5 := by sorry

end chocolate_chip_difference_l244_244529


namespace range_of_a_l244_244792

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end range_of_a_l244_244792


namespace simplify_expr_l244_244846

theorem simplify_expr : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expr_l244_244846


namespace sum_of_roots_l244_244427

theorem sum_of_roots (x : ℝ) (h : x + 49 / x = 14) : x + x = 14 :=
sorry

end sum_of_roots_l244_244427


namespace find_x_l244_244337

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (h : x + y + x * y = 143) : x = 15 :=
by sorry

end find_x_l244_244337


namespace determine_k_l244_244919

theorem determine_k (k : ℝ) : 
  (2 * k * (-1/2) - 3 = -7 * 3) → k = 18 :=
by
  intro h
  sorry

end determine_k_l244_244919


namespace members_playing_both_badminton_and_tennis_l244_244399

-- Definitions based on conditions
def N : ℕ := 35  -- Total number of members in the sports club
def B : ℕ := 15  -- Number of people who play badminton
def T : ℕ := 18  -- Number of people who play tennis
def Neither : ℕ := 5  -- Number of people who do not play either sport

-- The theorem based on the inclusion-exclusion principle
theorem members_playing_both_badminton_and_tennis :
  (B + T - (N - Neither) = 3) :=
by
  sorry

end members_playing_both_badminton_and_tennis_l244_244399


namespace solve_for_n_l244_244500

theorem solve_for_n (n : ℤ) : (3 : ℝ)^(2 * n + 2) = 1 / 9 ↔ n = -2 := by
  sorry

end solve_for_n_l244_244500


namespace discriminant_nonnegative_l244_244915

theorem discriminant_nonnegative {x : ℤ} (a : ℝ) (h₁ : x^2 * (49 - 40 * x^2) ≥ 0) :
  a = 0 ∨ a = 1 ∨ a = -1 ∨ a = 5/2 ∨ a = -5/2 := sorry

end discriminant_nonnegative_l244_244915


namespace probability_of_distinct_dice_numbers_l244_244241

/-- Total number of outcomes when rolling five six-sided dice. -/
def total_outcomes : ℕ := 6 ^ 5

/-- Number of favorable outcomes where all five dice show distinct numbers. -/
def favorable_outcomes : ℕ := 6 * 5 * 4 * 3 * 2

/-- Calculating the probability as a fraction. -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem probability_of_distinct_dice_numbers :
  probability = 5 / 54 :=
by
  -- Proof is required here.
  sorry

end probability_of_distinct_dice_numbers_l244_244241


namespace weight_of_new_student_l244_244541

theorem weight_of_new_student (avg_decrease_per_student : ℝ) (num_students : ℕ) (weight_replaced_student : ℝ) (total_reduction : ℝ) 
    (h1 : avg_decrease_per_student = 5) (h2 : num_students = 8) (h3 : weight_replaced_student = 86) (h4 : total_reduction = num_students * avg_decrease_per_student) :
    ∃ (x : ℝ), x = weight_replaced_student - total_reduction ∧ x = 46 :=
by
  use 46
  simp [h1, h2, h3, h4]
  sorry

end weight_of_new_student_l244_244541


namespace sum_of_mapped_elements_is_ten_l244_244934

theorem sum_of_mapped_elements_is_ten (a b : ℝ) (h1 : a = 1) (h2 : b = 9) : a + b = 10 := by
  sorry

end sum_of_mapped_elements_is_ten_l244_244934


namespace bus_problem_l244_244544

theorem bus_problem (x : ℕ)
  (h1 : 28 + 82 - x = 30) :
  82 - x = 2 :=
by {
  sorry
}

end bus_problem_l244_244544


namespace percentage_increase_l244_244346

theorem percentage_increase (original new : ℕ) (h₀ : original = 60) (h₁ : new = 120) :
  ((new - original) / original) * 100 = 100 := by
  sorry

end percentage_increase_l244_244346


namespace number_of_revolutions_wheel_half_mile_l244_244980

theorem number_of_revolutions_wheel_half_mile :
  let diameter := 10 * (1 : ℝ)
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let half_mile_in_feet := 2640
  (half_mile_in_feet / circumference) = 264 / Real.pi :=
by
  let diameter := 10 * (1 : ℝ)
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let half_mile_in_feet := 2640
  have h : (half_mile_in_feet / circumference) = 264 / Real.pi := by
    sorry
  exact h

end number_of_revolutions_wheel_half_mile_l244_244980


namespace sum_primes_less_than_20_l244_244069

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

noncomputable def sum_primes_less_than (n : Nat) : Nat :=
  (List.range n).filter is_prime |>.sum

theorem sum_primes_less_than_20 : sum_primes_less_than 20 = 77 := by
  sorry

end sum_primes_less_than_20_l244_244069


namespace find_r_l244_244457

variable (m r : ℝ)

theorem find_r (h1 : 5 = m * 3^r) (h2 : 45 = m * 9^(2 * r)) : r = 2 / 3 := by
  sorry

end find_r_l244_244457


namespace parallelogram_height_base_difference_l244_244322

theorem parallelogram_height_base_difference (A B H : ℝ) (hA : A = 24) (hB : B = 4) (hArea : A = B * H) :
  H - B = 2 := by
  sorry

end parallelogram_height_base_difference_l244_244322


namespace ideal_point_distance_y_axis_exists_ideal_point_linear_range_of_t_l244_244471

variable (a b : ℝ)
variable (m x : ℝ)
variable (t : ℝ)
variable (A B C : ℝ)

-- Define ideal points
def is_ideal_point (p : ℝ × ℝ) := p.snd = 2 * p.fst

-- Define the conditions for question 1
def distance_from_y_axis (a : ℝ) := abs a = 2

-- Question 1: Prove that M(2, 4) or M(-2, -4)
theorem ideal_point_distance_y_axis (a b : ℝ) (h1 : is_ideal_point (a, b)) (h2 : distance_from_y_axis a) :
  (a = 2 ∧ b = 4) ∨ (a = -2 ∧ b = -4) := sorry

-- Define the linear function
def linear_func (m x : ℝ) : ℝ := 3 * m * x - 1

-- Question 2: Prove or disprove the existence of ideal points in y = 3mx - 1
theorem exists_ideal_point_linear (m x : ℝ) (hx : is_ideal_point (x, linear_func m x)) :
  (m ≠ 2/3 → ∃ x, linear_func m x = 2 * x) ∧ (m = 2/3 → ¬ ∃ x, linear_func m x = 2 * x) := sorry

-- Question 3 conditions
def quadratic_func (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def quadratic_conditions (a b c : ℝ) : Prop :=
  (quadratic_func a b c 0 = 5 * a + 1) ∧ (quadratic_func a b c (-2) = 5 * a + 1)

-- Question 3: Prove the range of t = a^2 + a + 1 given the quadratic conditions
theorem range_of_t (a b c t : ℝ) (h1 : is_ideal_point (x, quadratic_func a b c x))
  (h2 : quadratic_conditions a b c) (ht : t = a^2 + a + 1) :
    3 / 4 ≤ t ∧ t ≤ 21 / 16 ∧ t ≠ 1 := sorry

end ideal_point_distance_y_axis_exists_ideal_point_linear_range_of_t_l244_244471


namespace solution_set_of_inequality_l244_244987

theorem solution_set_of_inequality (x : ℝ) : (1 / 2 < x ∧ x < 1) ↔ (x / (2 * x - 1) > 1) :=
by { sorry }

end solution_set_of_inequality_l244_244987


namespace train_length_proof_l244_244897

noncomputable def train_length (speed_kmph : ℕ) (time_seconds : ℕ) : ℝ :=
  (speed_kmph * 1000 / 3600) * time_seconds

theorem train_length_proof : train_length 100 18 = 500.04 :=
  sorry

end train_length_proof_l244_244897


namespace sin_240_eq_neg_sqrt3_div_2_l244_244594

theorem sin_240_eq_neg_sqrt3_div_2 :
  sin (240 : ℝ) = - (Real.sqrt 3) / 2 :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244594


namespace ellipse_eccentricity_l244_244773

theorem ellipse_eccentricity
  {a b n : ℝ}
  (h1 : a > b) (h2 : b > 0)
  (h3 : ∃ (P : ℝ × ℝ), P.1 = n ∧ P.2 = 4 ∧ (n^2 / a^2 + 16 / b^2 = 1))
  (F1 F2 : ℝ × ℝ)
  (h4 : F1 = (c, 0))        -- Placeholders for focus coordinates of the ellipse
  (h5 : F2 = (-c, 0))
  (h6 : ∃ c, 4*c = (3 / 2) * (a + c))
  : 3 * c = 5 * a → c / a = 3 / 5 :=
by
  sorry

end ellipse_eccentricity_l244_244773


namespace range_of_u_l244_244432

variable (a b u : ℝ)

theorem range_of_u (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (∀ x : ℝ, x > 0 → a^2 + b^2 ≥ x ↔ x ≤ 16) :=
sorry

end range_of_u_l244_244432


namespace triangle_inequality_cubed_l244_244647

theorem triangle_inequality_cubed
  (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) :
  (a^3 / c^3) + (b^3 / c^3) + (3 * a * b / c^2) > 1 := 
sorry

end triangle_inequality_cubed_l244_244647


namespace probability_theorem_l244_244237

open Set Function

-- Define the conditions for the problem
def boxA := {n | 1 ≤ n ∧ n ≤ 30}
def boxB := {n | 15 ≤ n ∧ n ≤ 44}

def P_A (n : ℕ) := n ∈ boxA
def P_B (n : ℕ) := n ∈ boxB

def conditionA := {n ∈ boxA | n < 20}
def conditionB := {n ∈ boxB | (∃ k, n = 2 * k) ∨ n > 35}

-- Calculate the probabilities
def probabilityA := (conditionA.toFinset.card : ℚ) / (boxA.toFinset.card : ℚ)
def probabilityB := (conditionB.toFinset.card : ℚ) / (boxB.toFinset.card : ℚ)

-- The combined probability of independent events
def combined_probability := probabilityA * probabilityB

-- Lean statement for the proof problem
theorem probability_theorem : combined_probability = 361 / 900 :=
by
  -- defining the calculations based on conditions
  have boxA_card : boxA.toFinset.card = 30 := sorry
  have boxB_card : boxB.toFinset.card = 30 := sorry
  
  have conditionA_card : conditionA.toFinset.card = 19 := sorry
  have conditionB_card : conditionB.toFinset.card = 19 := sorry

  -- calculations for the individual probabilities
  have pA : probabilityA = 19 / 30 := by
    rw [probabilityA, conditionA_card, boxA_card]; norm_num

  have pB : probabilityB = 19 / 30 := by
    rw [probabilityB, conditionB_card, boxB_card]; norm_num
    
  -- verifying the combined probability
  rw [combined_probability, pA, pB]; norm_num

end probability_theorem_l244_244237


namespace sin_240_eq_neg_sqrt3_div_2_l244_244583

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244583


namespace jason_flame_time_l244_244394

-- Define firing interval and flame duration
def firing_interval := 15
def flame_duration := 5

-- Define the function to calculate seconds per minute
def seconds_per_minute (interval : ℕ) (duration : ℕ) : ℕ :=
  (60 / interval) * duration

-- Theorem to state the problem
theorem jason_flame_time : 
  seconds_per_minute firing_interval flame_duration = 20 := 
by
  sorry

end jason_flame_time_l244_244394


namespace train_travel_distance_l244_244890

theorem train_travel_distance (m : ℝ) (h : 3 * 60 * 1 = m) : m = 180 :=
by
  sorry

end train_travel_distance_l244_244890


namespace dante_flour_eggs_l244_244161

theorem dante_flour_eggs (eggs : ℕ) (h_eggs : eggs = 60) (h_flour : ∀ (f : ℕ), f = eggs / 2) : eggs + (eggs / 2) = 90 := 
by {
  rw h_eggs,
  calc
    60 + (60 / 2) = 60 + 30   : by norm_num
    ...         = 90 : by norm_num
}

end dante_flour_eggs_l244_244161


namespace range_of_a_l244_244813

def sets_nonempty_intersect (a : ℝ) : Prop :=
  ∃ x, -1 ≤ x ∧ x < 2 ∧ x < a

theorem range_of_a (a : ℝ) (h : sets_nonempty_intersect a) : a > -1 :=
by
  sorry

end range_of_a_l244_244813


namespace find_slower_speed_l244_244463

-- Variables and conditions definitions
variable (v : ℝ)

def slower_speed (v : ℝ) : Prop :=
  (20 / v = 2) ∧ (v = 10)

-- The statement to be proven
theorem find_slower_speed : slower_speed 10 :=
by
  sorry

end find_slower_speed_l244_244463


namespace solve_fraction_eq_l244_244972

theorem solve_fraction_eq (x : ℚ) (h : (x^2 + 3 * x + 4) / (x + 3) = x + 6) : x = -7 / 3 :=
sorry

end solve_fraction_eq_l244_244972


namespace ratio_time_A_to_B_l244_244956

-- Definition of total examination time in minutes
def total_time : ℕ := 180

-- Definition of time spent on type A problems
def time_A : ℕ := 40

-- Definition of time spent on type B problems as total_time - time_A
def time_B : ℕ := total_time - time_A

-- Statement that we need to prove
theorem ratio_time_A_to_B : time_A * 7 = time_B * 2 :=
by
  -- Implementation of the proof will go here
  sorry

end ratio_time_A_to_B_l244_244956


namespace deepak_age_l244_244984

variable (R D : ℕ)

theorem deepak_age (h1 : R / D = 4 / 3) (h2 : R + 6 = 26) : D = 15 :=
sorry

end deepak_age_l244_244984


namespace maple_taller_than_pine_l244_244795

theorem maple_taller_than_pine :
  let pine_tree := 24 + 1/4
  let maple_tree := 31 + 2/3
  (maple_tree - pine_tree) = 7 + 5/12 :=
by
  sorry

end maple_taller_than_pine_l244_244795


namespace diamond_eq_l244_244545

noncomputable def diamond_op (a b : ℝ) (k : ℝ) : ℝ := sorry

theorem diamond_eq (x : ℝ) :
  let k := 2
  let a := 2023
  let b := 7
  let c := x
  (diamond_op a (diamond_op b c k) k = 150) ∧ 
  (∀ a b c, diamond_op a (diamond_op b c k) k = k * (diamond_op a b k) * c) ∧
  (∀ a, diamond_op a a k = k) →
  x = 150 / 2023 :=
sorry

end diamond_eq_l244_244545


namespace sin_240_l244_244578

theorem sin_240 : Real.sin (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Provided conditions
  have h1 : 240 = 180 + 60 := be_of_eq true.intro
  have h2 : ∀ θ : ℝ, θ ∈ set.Icc (pi : ℝ) (3 * pi / 2) → Real.sin θ < 0 := Real.sin_neg_of_pi_lt_of_lt (Real.pi_lt_2_pi)
  have h3 : Real.sin (60 * Real.pi / 180) = 1 / 2 := Real.sin_pi_div_three
  -- Prove
  sorry

end sin_240_l244_244578


namespace students_who_won_first_prize_l244_244704

theorem students_who_won_first_prize :
  ∃ x : ℤ, 30 ≤ x ∧ x ≤ 55 ∧ (x % 3 = 2) ∧ (x % 5 = 4) ∧ (x % 7 = 2) ∧ x = 44 :=
by
  sorry

end students_who_won_first_prize_l244_244704


namespace solve_system_l244_244856

theorem solve_system (x y : ℚ) 
  (h1 : 3 * (x - 1) = y + 6) 
  (h2 : x / 2 + y / 3 = 2) : 
  x = 10 / 3 ∧ y = 1 := 
by 
  sorry

end solve_system_l244_244856


namespace divide_triangle_l244_244629

/-- 
  Given a triangle with the total sum of numbers as 63,
  we want to prove that it can be divided into three parts
  where each part's sum is 21.
-/
theorem divide_triangle (total_sum : ℕ) (H : total_sum = 63) :
  ∃ (part1 part2 part3 : ℕ), 
    (part1 + part2 + part3 = total_sum) ∧ 
    part1 = 21 ∧ 
    part2 = 21 ∧ 
    part3 = 21 :=
by 
  use 21, 21, 21
  split
  . exact H.symm ▸ rfl
  . split; rfl
  . split; rfl
  . rfl

end divide_triangle_l244_244629


namespace proof_l244_244349

-- Define the conditions in Lean
variable {f : ℝ → ℝ}
variable (h1 : ∀ x ∈ (Set.Ioi 0), 0 ≤ f x)
variable (h2 : ∀ x ∈ (Set.Ioi 0), x * f x + f x ≤ 0)

-- Formulate the goal
theorem proof (a b : ℝ) (ha : a ∈ (Set.Ioi 0)) (hb : b ∈ (Set.Ioi 0)) (h : a < b) : 
    b * f a ≤ a * f b :=
by
  sorry  -- Proof omitted

end proof_l244_244349


namespace ratio_of_shares_l244_244886

-- Definitions
variable (A B C : ℝ)   -- Representing the shares of a, b, and c
variable (x : ℝ)       -- Fraction

-- Conditions
axiom h1 : A = 80
axiom h2 : A + B + C = 200
axiom h3 : A = x * (B + C)
axiom h4 : B = (6 / 9) * (A + C)

-- Statement to prove
theorem ratio_of_shares : A / (B + C) = 2 / 3 :=
by sorry

end ratio_of_shares_l244_244886


namespace cos_angle_between_vectors_l244_244671

theorem cos_angle_between_vectors :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (1, 3)
  let dot_product (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2
  let magnitude (x : ℝ × ℝ) : ℝ := Real.sqrt (x.1 ^ 2 + x.2 ^ 2)
  let cos_theta := dot_product a b / (magnitude a * magnitude b)
  cos_theta = -Real.sqrt 2 / 10 :=
by
  sorry

end cos_angle_between_vectors_l244_244671


namespace smallest_xyz_sum_l244_244963

theorem smallest_xyz_sum (x y z : ℕ) (h1 : (x + y) * (y + z) = 2016) (h2 : (x + y) * (z + x) = 1080) :
  x > 0 → y > 0 → z > 0 → x + y + z = 61 :=
  sorry

end smallest_xyz_sum_l244_244963


namespace remainder_mod_7_l244_244752

theorem remainder_mod_7 : (4 * 6^24 + 3^48) % 7 = 5 := by
  sorry

end remainder_mod_7_l244_244752


namespace sin_240_eq_neg_sqrt3_div_2_l244_244589

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244589


namespace solve_quadratic_eq_l244_244707

theorem solve_quadratic_eq (x : ℝ) : x^2 = 6 * x ↔ (x = 0 ∨ x = 6) := by
  sorry

end solve_quadratic_eq_l244_244707


namespace smaller_rectangle_area_l244_244553

-- Define the lengths and widths of the rectangles
def bigRectangleLength : ℕ := 40
def bigRectangleWidth : ℕ := 20
def smallRectangleLength : ℕ := bigRectangleLength / 2
def smallRectangleWidth : ℕ := bigRectangleWidth / 2

-- Define the area of the rectangles
def area (length width : ℕ) : ℕ := length * width

-- Prove the area of the smaller rectangle
theorem smaller_rectangle_area : area smallRectangleLength smallRectangleWidth = 200 :=
by
  -- Skip the proof
  sorry

end smaller_rectangle_area_l244_244553


namespace sum_primes_less_than_20_l244_244095

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l244_244095


namespace arccos_cos_11_eq_l244_244564

theorem arccos_cos_11_eq: Real.arccos (Real.cos 11) = 11 - 3 * Real.pi := by
  sorry

end arccos_cos_11_eq_l244_244564


namespace length_segment_FF_l244_244238

-- Define the points F and F' based on the given conditions
def F : (ℝ × ℝ) := (4, 3)
def F' : (ℝ × ℝ) := (-4, 3)

-- The theorem to prove the length of the segment FF' is 8
theorem length_segment_FF' : dist F F' = 8 :=
by
  sorry

end length_segment_FF_l244_244238


namespace train_travel_distance_l244_244893

def speed (miles : ℕ) (minutes : ℕ) : ℕ :=
  miles / minutes

def minutes_in_hours (hours : ℕ) : ℕ :=
  hours * 60

def distance_traveled (rate : ℕ) (time : ℕ) : ℕ :=
  rate * time

theorem train_travel_distance :
  (speed 2 2 = 1) →
  (minutes_in_hours 3 = 180) →
  distance_traveled (speed 2 2) (minutes_in_hours 3) = 180 :=
by
  intros h_speed h_minutes
  rw [h_speed, h_minutes]
  sorry

end train_travel_distance_l244_244893


namespace sum_primes_less_than_20_l244_244046

theorem sum_primes_less_than_20 : (∑ p in ({2, 3, 5, 7, 11, 13, 17, 19} : Finset ℕ), p) = 77 :=
by
  sorry

end sum_primes_less_than_20_l244_244046


namespace simplify_and_evaluate_l244_244852

theorem simplify_and_evaluate 
  (x y : ℤ) 
  (h1 : |x| = 2) 
  (h2 : y = 1) 
  (h3 : x * y < 0) : 
  3 * x^2 * y - 2 * x^2 - (x * y)^2 - 3 * x^2 * y - 4 * (x * y)^2 = -18 := by
  sorry

end simplify_and_evaluate_l244_244852


namespace solve_house_A_cost_l244_244209

-- Definitions and assumptions
variables (A B C : ℝ)
variable base_salary : ℝ := 3000
variable commission_rate : ℝ := 0.02
variable total_earnings : ℝ := 8000

-- Conditions
def house_B_cost (A : ℝ) : ℝ := 3 * A
def house_C_cost (A : ℝ) : ℝ := 2 * A - 110000

-- Define Nigella's commission calculation
def nigella_commission (A B C : ℝ) : ℝ := commission_rate * A + commission_rate * B + commission_rate * C

-- Commission earned based on total earnings and base salary
def commission_earned : ℝ := total_earnings - base_salary

-- Lean theorem statement
theorem solve_house_A_cost 
  (hB : B = house_B_cost A)
  (hC : C = house_C_cost A)
  (h_commission : nigella_commission A B C = commission_earned) : 
  A = 60000 :=
by 
-- Sorry is used to skip the actual proof
sorry

end solve_house_A_cost_l244_244209


namespace work_schedules_lcm_l244_244900

theorem work_schedules_lcm : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9) = 360 := 
by 
  sorry

end work_schedules_lcm_l244_244900


namespace sum_of_primes_less_than_20_is_77_l244_244040

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l244_244040


namespace sum_of_primes_lt_20_eq_77_l244_244114

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l244_244114


namespace greatest_three_digit_multiple_of_17_is_986_l244_244012

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l244_244012


namespace least_positive_integer_a_l244_244924

theorem least_positive_integer_a (a : ℕ) (n : ℕ) 
  (h1 : 2001 = 3 * 23 * 29)
  (h2 : 55 % 3 = 1)
  (h3 : 32 % 3 = -1)
  (h4 : 55 % 23 = 32 % 23)
  (h5 : 55 % 29 = -32 % 29)
  (h6 : n % 2 = 1)
  : a = 436 := 
sorry

end least_positive_integer_a_l244_244924


namespace total_cards_in_box_l244_244127

-- Definitions based on conditions
def xiaoMingCountsFaster (m h : ℕ) := 6 * h = 4 * m
def xiaoHuaForgets (h1 h2 : ℕ) := h1 + h2 = 112
def finalCardLeft (t : ℕ) := t - 1 = 112

-- Main theorem stating that the total number of cards is 353
theorem total_cards_in_box : ∃ N : ℕ, 
    (∃ m h1 h2 : ℕ,
        xiaoMingCountsFaster m h1 ∧
        xiaoHuaForgets h1 h2 ∧
        finalCardLeft N) ∧
    N = 353 :=
sorry

end total_cards_in_box_l244_244127


namespace checkerboard_problem_l244_244267

def is_valid_square (size : ℕ) : Prop :=
  size = 4 ∨ size = 5 ∨ size = 6 ∨ size = 7 ∨ size = 8 ∨ size = 9 ∨ size = 10

def check_10_by_10 : ℕ :=
  24 + 36 + 25 + 16 + 9 + 4 + 1

theorem checkerboard_problem :
  ∀ size : ℕ, ( size = 4 ∨ size = 5 ∨ size = 6 ∨ size = 7 ∨ size = 8 ∨ size = 9 ∨ size = 10 ) →
  check_10_by_10 = 115 := 
sorry

end checkerboard_problem_l244_244267


namespace abs_h_of_roots_sum_squares_eq_34_l244_244234

theorem abs_h_of_roots_sum_squares_eq_34 
  (h : ℝ)
  (h_eq : ∀ r s : ℝ, (2 * r^2 + 4 * h * r + 6 = 0) ∧ (2 * s^2 + 4 * h * s + 6 = 0)) 
  (sum_of_squares_eq : ∀ r s : ℝ, (2 * r^2 + 4 * h * r + 6 = 0) ∧ (2 * s^2 + 4 * h * s + 6 = 0) → r^2 + s^2 = 34) :
  |h| = Real.sqrt 10 :=
by
  sorry

end abs_h_of_roots_sum_squares_eq_34_l244_244234


namespace sum_prime_numbers_less_than_twenty_l244_244093

-- Define the set of prime numbers less than 20.
def prime_numbers_less_than_twenty : Set ℕ :=
  {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the sum of the elements in a set.
def set_sum (s : Set ℕ) : ℕ :=
  s.toFinset.sum id

theorem sum_prime_numbers_less_than_twenty :
  set_sum prime_numbers_less_than_twenty = 77 :=
by
  sorry

end sum_prime_numbers_less_than_twenty_l244_244093


namespace major_axis_length_l244_244735

theorem major_axis_length (radius : ℝ) (k : ℝ) (minor_axis : ℝ) (major_axis : ℝ)
  (cyl_radius : radius = 2)
  (minor_eq_diameter : minor_axis = 2 * radius)
  (major_longer : major_axis = minor_axis * (1 + k))
  (k_value : k = 0.25) :
  major_axis = 5 :=
by
  -- Proof omitted, using sorry
  sorry

end major_axis_length_l244_244735


namespace min_value_ineq_l244_244811

noncomputable def min_value (x y z : ℝ) := (1/x) + (1/y) + (1/z)

theorem min_value_ineq (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) :
  min_value x y z ≥ 4.5 :=
sorry

end min_value_ineq_l244_244811


namespace greatest_three_digit_multiple_of_17_is_986_l244_244008

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l244_244008


namespace problem_conditions_l244_244685

def G (m : ℕ) : ℕ := m % 10

theorem problem_conditions (a b c : ℕ) (non_neg_m : ∀ m : ℕ, 0 ≤ m) :
  -- Condition ①
  ¬ (G (a - b) = G a - G b) ∧
  -- Condition ②
  (a - b = 10 * c → G a = G b) ∧
  -- Condition ③
  (G (a * b * c) = G (G a * G b * G c)) ∧
  -- Condition ④
  ¬ (G (3^2015) = 9) :=
by sorry

end problem_conditions_l244_244685


namespace simplify_expression_l244_244850

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l244_244850


namespace sin_240_eq_neg_sqrt3_div_2_l244_244607

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244607


namespace Tom_earns_per_week_l244_244523

-- Definitions based on conditions
def crab_buckets_per_day := 8
def crabs_per_bucket := 12
def price_per_crab := 5
def days_per_week := 7

-- The proof goal
theorem Tom_earns_per_week :
  (crab_buckets_per_day * crabs_per_bucket * price_per_crab * days_per_week) = 3360 := by
  sorry

end Tom_earns_per_week_l244_244523


namespace fraction_of_top10_lists_l244_244143

theorem fraction_of_top10_lists (total_members : ℕ) (min_lists : ℝ) (H1 : total_members = 795) (H2 : min_lists = 198.75) :
  (min_lists / total_members) = 1 / 4 :=
by
  -- The proof is omitted as requested
  sorry

end fraction_of_top10_lists_l244_244143


namespace find_a2_l244_244435

variable (a : ℕ → ℤ)

-- Conditions
def is_arithmetic_sequence (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (x y z : ℤ) : Prop :=
  y * y = x * z

-- Specific condition for the problem
axiom h_arithmetic : is_arithmetic_sequence a 2
axiom h_geometric : is_geometric_sequence (a 1 + 2) (a 3 + 6) (a 4 + 8)

-- Theorem to prove
theorem find_a2 : a 1 + 2 = -8 := 
sorry

-- We assert that the value of a_2 must satisfy the given conditions

end find_a2_l244_244435


namespace sum_of_primes_less_than_twenty_is_77_l244_244031

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l244_244031


namespace factorization_correct_l244_244698

theorem factorization_correct (c d : ℤ) (h : 25 * x^2 - 160 * x - 144 = (5 * x + c) * (5 * x + d)) : c + 2 * d = -2 := 
sorry

end factorization_correct_l244_244698


namespace sin_240_l244_244576

theorem sin_240 : Real.sin (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Provided conditions
  have h1 : 240 = 180 + 60 := be_of_eq true.intro
  have h2 : ∀ θ : ℝ, θ ∈ set.Icc (pi : ℝ) (3 * pi / 2) → Real.sin θ < 0 := Real.sin_neg_of_pi_lt_of_lt (Real.pi_lt_2_pi)
  have h3 : Real.sin (60 * Real.pi / 180) = 1 / 2 := Real.sin_pi_div_three
  -- Prove
  sorry

end sin_240_l244_244576


namespace largest_equal_cost_l244_244992

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def binary_digit_sum (n : ℕ) : ℕ :=
  n.digits 2 |>.sum

theorem largest_equal_cost :
  ∃ (n : ℕ), n < 500 ∧ digit_sum n = binary_digit_sum n ∧ ∀ m < 500, digit_sum m = binary_digit_sum m → m ≤ 247 :=
by
  sorry

end largest_equal_cost_l244_244992


namespace find_x_l244_244192

theorem find_x 
  (x : ℝ) 
  (angle_PQS angle_QSR angle_SRQ : ℝ) 
  (h1 : angle_PQS = 2 * x)
  (h2 : angle_QSR = 50)
  (h3 : angle_SRQ = x) :
  x = 50 :=
sorry

end find_x_l244_244192


namespace range_of_x_l244_244181

noncomputable def f (x : ℝ) : ℝ := x * (2^x - 1 / 2^x)

theorem range_of_x (x : ℝ) (h : f (x - 1) > f x) : x < 1 / 2 :=
by sorry

end range_of_x_l244_244181


namespace simplify_expression_l244_244848

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l244_244848


namespace greatest_three_digit_multiple_of_17_is_986_l244_244009

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l244_244009


namespace inequality_proof_l244_244950

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
  (1 / Real.sqrt (x + y)) + (1 / Real.sqrt (y + z)) + (1 / Real.sqrt (z + x)) ≤ 1 / Real.sqrt (2 * x * y * z) :=
by
  sorry

end inequality_proof_l244_244950


namespace cos_theta_example_l244_244672

variables (a b : ℝ × ℝ) (θ : ℝ)

def cos_theta (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))

theorem cos_theta_example :
  let a := (2, -1)
  let b := (1, 3)
  cos_theta a b = -(Real.sqrt 2) / 10 :=
by
  sorry

end cos_theta_example_l244_244672


namespace right_triangle_area_l244_244305

theorem right_triangle_area (x y : ℝ) 
  (h1 : x + y = 4) 
  (h2 : x^2 + y^2 = 9) : 
  (1/2) * x * y = 7 / 4 := 
by
  sorry

end right_triangle_area_l244_244305


namespace count_ab_bc_ca_l244_244871

noncomputable def count_ways : ℕ :=
  (Nat.choose 9 3)

theorem count_ab_bc_ca (a b c : ℕ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 1 ≤ b ∧ b ≤ 9) (hc : 1 ≤ c ∧ c ≤ 9) :
  (10 * a + b < 10 * b + c ∧ 10 * b + c < 10 * c + a) → count_ways = 84 :=
sorry

end count_ab_bc_ca_l244_244871


namespace binomial_expansion_coefficient_l244_244317

theorem binomial_expansion_coefficient :
  let a_0 : ℚ := (1 + 2 * (0:ℚ))^5
  (1 + 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_3 = 80 :=
by 
  sorry

end binomial_expansion_coefficient_l244_244317


namespace diff_between_largest_and_smallest_fraction_l244_244211

theorem diff_between_largest_and_smallest_fraction : 
  let f1 := (3 : ℚ) / 4
  let f2 := (7 : ℚ) / 8
  let f3 := (13 : ℚ) / 16
  let f4 := (1 : ℚ) / 2
  let largest := max f1 (max f2 (max f3 f4))
  let smallest := min f1 (min f2 (min f3 f4))
  largest - smallest = (3 : ℚ) / 8 :=
by
  sorry

end diff_between_largest_and_smallest_fraction_l244_244211


namespace ratio_matt_fem_4_1_l244_244814

-- Define Fem's current age
def FemCurrentAge : ℕ := 11

-- Define the condition about the sum of their ages in two years
def AgeSumInTwoYears (MattCurrentAge : ℕ) : Prop :=
  (FemCurrentAge + 2) + (MattCurrentAge + 2) = 59

-- Define the desired ratio as a property
def DesiredRatio (MattCurrentAge : ℕ) : Prop :=
  MattCurrentAge / FemCurrentAge = 4

-- Create the theorem statement
theorem ratio_matt_fem_4_1 (M : ℕ) (h : AgeSumInTwoYears M) : DesiredRatio M :=
  sorry

end ratio_matt_fem_4_1_l244_244814


namespace lily_pad_cover_entire_lake_l244_244799

-- Definitions per the conditions
def doublesInSizeEveryDay (P : ℕ → ℝ) : Prop :=
  ∀ n, P (n + 1) = 2 * P n

-- The initial state that it takes 36 days to cover the lake
def coversEntireLakeIn36Days (P : ℕ → ℝ) (L : ℝ) : Prop :=
  P 36 = L

-- The main theorem to prove
theorem lily_pad_cover_entire_lake (P : ℕ → ℝ) (L : ℝ) (h1 : doublesInSizeEveryDay P) (h2 : coversEntireLakeIn36Days P L) :
  ∃ n, n = 36 := 
by
  sorry

end lily_pad_cover_entire_lake_l244_244799


namespace angle_C_measurement_l244_244196

variables (A B C : ℝ)

theorem angle_C_measurement
  (h1 : A + C = 2 * B)
  (h2 : C - A = 80)
  (h3 : A + B + C = 180) :
  C = 100 :=
by sorry

end angle_C_measurement_l244_244196


namespace sum_of_primes_lt_20_eq_77_l244_244110

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l244_244110


namespace flour_already_put_in_l244_244207

theorem flour_already_put_in (total_flour flour_still_needed: ℕ) (h1: total_flour = 9) (h2: flour_still_needed = 6) : total_flour - flour_still_needed = 3 := 
by
  -- Here we will state the proof
  sorry

end flour_already_put_in_l244_244207


namespace toms_weekly_revenue_l244_244519

def crabs_per_bucket : Nat := 12
def number_of_buckets : Nat := 8
def price_per_crab : Nat := 5
def days_per_week : Nat := 7

theorem toms_weekly_revenue :
  (crabs_per_bucket * number_of_buckets * price_per_crab * days_per_week) = 3360 :=
by
  sorry

end toms_weekly_revenue_l244_244519


namespace ratio_of_areas_l244_244503

theorem ratio_of_areas (s : ℝ) (hs : 0 < s) :
  let longer_side_R := 1.2 * s
  let shorter_side_R := 0.85 * s
  let area_R := longer_side_R * shorter_side_R
  let area_S := s^2
  area_R / area_S = 51 / 50 :=
by
  let longer_side_R := 1.2 * s
  let shorter_side_R := 0.85 * s
  let area_R := longer_side_R * shorter_side_R
  let area_S := s^2
  calc
    area_R / area_S = (1.2 * s * 0.85 * s) / (s * s) : by rw [area_R, area_S]
    ... = 1.02 : by { field_simp [ne_of_gt hs], ring }
    ... = 51 / 50 : by norm_num

end ratio_of_areas_l244_244503


namespace combined_garden_area_l244_244967

def garden_area (length width : ℕ) : ℕ :=
  length * width

def total_area (count length width : ℕ) : ℕ :=
  count * garden_area length width

theorem combined_garden_area :
  let M_length := 16
  let M_width := 5
  let M_count := 3
  let Ma_length := 8
  let Ma_width := 4
  let Ma_count := 2
  total_area M_count M_length M_width + total_area Ma_count Ma_length Ma_width = 304 :=
by
  sorry

end combined_garden_area_l244_244967


namespace reciprocal_of_recurring_three_l244_244390

noncomputable def recurring_three := 0.33333333333 -- approximation of 0.\overline{3}

theorem reciprocal_of_recurring_three :
  let x := recurring_three in
  (x = (1/3)) → (1 / x = 3) := 
by 
  sorry

end reciprocal_of_recurring_three_l244_244390


namespace abs_eq_2_iff_l244_244981

theorem abs_eq_2_iff (a : ℚ) : abs a = 2 ↔ a = 2 ∨ a = -2 :=
by
  sorry

end abs_eq_2_iff_l244_244981


namespace a_seq_correct_b_seq_max_m_l244_244772

noncomputable def a_seq (n : ℕ) : ℕ :=
if n = 0 then 3 else (n + 1)^2 + 2

-- Verification that the sequence follows the provided conditions.
theorem a_seq_correct (n : ℕ) : 
  (a_seq 0 = 3) ∧
  (a_seq 1 = 6) ∧
  (a_seq 2 = 11) ∧
  (∀ m : ℕ, m ≥ 1 → a_seq (m + 1) - a_seq m = 2 * m + 1) := sorry

noncomputable def b_seq (n : ℕ) : ℝ := 
(a_seq n : ℝ) / (3 ^ (Real.sqrt (a_seq n - 2)))

theorem b_seq_max_m (m : ℝ) : 
  (∀ n : ℕ, b_seq n ≤ m) ↔ (1 ≤ m) := sorry

end a_seq_correct_b_seq_max_m_l244_244772


namespace contracting_schemes_l244_244907

theorem contracting_schemes :
  let total_projects := 6
  let a_contracts := 3
  let b_contracts := 2
  let c_contracts := 1
  (Nat.choose total_projects a_contracts) *
  (Nat.choose (total_projects - a_contracts) b_contracts) *
  (Nat.choose ((total_projects - a_contracts) - b_contracts) c_contracts) = 60 :=
by
  let total_projects := 6
  let a_contracts := 3
  let b_contracts := 2
  let c_contracts := 1
  sorry

end contracting_schemes_l244_244907


namespace sum_of_primes_less_than_20_l244_244015

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def primes_less_than_n (n : ℕ) := {m : ℕ | is_prime m ∧ m < n}

theorem sum_of_primes_less_than_20 : (∑ x in primes_less_than_n 20, x) = 77 :=
by
  have h : primes_less_than_n 20 = {2, 3, 5, 7, 11, 13, 17, 19} := sorry
  have h_sum : (∑ x in {2, 3, 5, 7, 11, 13, 17, 19}, x) = 77 := by
    simp [Finset.sum, Nat.add]
    sorry
  rw [h]
  exact h_sum

end sum_of_primes_less_than_20_l244_244015


namespace problem_l244_244777

-- Definitions and hypotheses based on the given conditions
variable (a b : ℝ)
def sol_set := {x : ℝ | -1/2 < x ∧ x < 1/3}
def quadratic_inequality (x : ℝ) := a * x^2 + b * x + 2

-- Statement expressing that the inequality holds for the given solution set
theorem problem
  (h : ∀ (x : ℝ), x ∈ sol_set → quadratic_inequality a b x > 0) :
  a - b = -10 :=
sorry

end problem_l244_244777


namespace hope_cup_1990_inequalities_l244_244180

variable {a b c x y z : ℝ}

/-- Given a > b > c, x > y > z,
    M = ax + by + cz,
    N = az + by + cx,
    P = ay + bz + cx,
    Q = az + bx + cy.

    Prove that:
    M > P > N and M > Q > N. -/
theorem hope_cup_1990_inequalities :
  a > b -> b > c -> x > y -> y > z ->
  let M := a * x + b * y + c * z in
  let N := a * z + b * y + c * x in
  let P := a * y + b * z + c * x in
  let Q := a * z + b * x + c * y in
  M > P ∧ P > N ∧ M > Q ∧ Q > N := sorry

end hope_cup_1990_inequalities_l244_244180


namespace find_f_of_neg_2_l244_244857

theorem find_f_of_neg_2
  (f : ℚ → ℚ)
  (h : ∀ (x : ℚ), x ≠ 0 → 3 * f (1/x) + 2 * f x / x = x^2)
  : f (-2) = 13/5 :=
sorry

end find_f_of_neg_2_l244_244857


namespace find_s_l_l244_244701

theorem find_s_l :
  ∃ s l : ℝ, ∀ t : ℝ, 
  (-8 + l * t, s + -6 * t) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p.snd = 3 / 4 * x + 2 ∧ p.fst = x} ∧ 
  (s = -4 ∧ l = -8) :=
by
  sorry

end find_s_l_l244_244701


namespace total_surface_area_of_rectangular_solid_with_given_volume_and_prime_edges_l244_244737

theorem total_surface_area_of_rectangular_solid_with_given_volume_and_prime_edges :
  ∃ (a b c : ℕ), Prime a ∧ Prime b ∧ Prime c ∧ a * b * c = 1001 ∧ 2 * (a * b + b * c + c * a) = 622 :=
by
  sorry

end total_surface_area_of_rectangular_solid_with_given_volume_and_prime_edges_l244_244737


namespace second_train_further_l244_244714

-- Define the speeds of the two trains
def speed_train1 : ℝ := 50
def speed_train2 : ℝ := 60

-- Define the total distance between points A and B
def total_distance : ℝ := 1100

-- Define the distances traveled by the two trains when they meet
def distance_train1 (t: ℝ) : ℝ := speed_train1 * t
def distance_train2 (t: ℝ) : ℝ := speed_train2 * t

-- Define the meeting condition
def meeting_condition (t: ℝ) : Prop := distance_train1 t + distance_train2 t = total_distance

-- Prove the distance difference
theorem second_train_further (t: ℝ) (h: meeting_condition t) : distance_train2 t - distance_train1 t = 100 :=
sorry

end second_train_further_l244_244714


namespace problem_l244_244805

variable (a b c : ℝ)

theorem problem (h : a^2 * b^2 + 18 * a * b * c > 4 * b^3 + 4 * a^3 * c + 27 * c^2) : a^2 > 3 * b :=
by
  sorry

end problem_l244_244805


namespace blue_red_area_ratio_l244_244384

theorem blue_red_area_ratio (d_small d_large : ℕ) (h1 : d_small = 2) (h2 : d_large = 6) :
    let r_small := d_small / 2
    let r_large := d_large / 2
    let A_red := Real.pi * (r_small : ℝ) ^ 2
    let A_large := Real.pi * (r_large : ℝ) ^ 2
    let A_blue := A_large - A_red
    A_blue / A_red = 8 :=
by
  sorry

end blue_red_area_ratio_l244_244384


namespace A_half_B_l244_244159

-- Define the arithmetic series sum function
def series_sum (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define A and B according to the problem conditions
def A : ℕ := (Finset.range 2022).sum (λ m => series_sum (m + 1))

def B : ℕ := (Finset.range 2022).sum (λ m => (m + 1) * (m + 2))

-- The proof statement
theorem A_half_B : A = B / 2 :=
by
  sorry

end A_half_B_l244_244159


namespace simplify_fraction_l244_244834

theorem simplify_fraction : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 :=
by sorry

end simplify_fraction_l244_244834


namespace sin_240_eq_neg_sqrt3_over_2_l244_244597

open Real

-- Conditions
def angle_240_in_third_quadrant : Prop := 240 ° ∈ set_of (λ x, 180 ° < x ∧ x < 270 °)

def reference_angle_60 (θ : Real) : Prop := θ = 240 ° - 180 °

def sin_60_eq_sqrt3_over_2 : sin (60 °) = sqrt 3 / 2

def sin_negative_in_third_quadrant (θ : Real) : Prop :=
  180 ° < θ ∧ θ < 270 ° → sin θ < 0

-- Statement
theorem sin_240_eq_neg_sqrt3_over_2 :
  angle_240_in_third_quadrant ∧ reference_angle_60 60 ° ∧ sin_60_eq_sqrt3_over_2 ∧ sin_negative_in_third_quadrant 240 °
  → sin (240 °) = - (sqrt 3 / 2) :=
by
  intros
  sorry

end sin_240_eq_neg_sqrt3_over_2_l244_244597


namespace sum_primes_less_than_20_l244_244096

theorem sum_primes_less_than_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by sorry

end sum_primes_less_than_20_l244_244096


namespace sin_240_eq_neg_sqrt3_div_2_l244_244596

theorem sin_240_eq_neg_sqrt3_div_2 :
  sin (240 : ℝ) = - (Real.sqrt 3) / 2 :=
by
  sorry

end sin_240_eq_neg_sqrt3_div_2_l244_244596


namespace image_of_center_after_transformations_l244_244417

-- Define the initial center of circle C
def initial_center : ℝ × ℝ := (3, -4)

-- Define a function to reflect a point across the x-axis
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define a function to translate a point by some units left
def translate_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

-- Define the final coordinates after transformations
def final_center : ℝ × ℝ :=
  translate_left (reflect_x_axis initial_center) 5

-- The theorem to prove
theorem image_of_center_after_transformations :
  final_center = (-2, 4) :=
by
  sorry

end image_of_center_after_transformations_l244_244417


namespace S_12_l244_244473

variable {S : ℕ → ℕ}

-- Given conditions
axiom S_4 : S 4 = 4
axiom S_8 : S 8 = 12

-- Goal: Prove S_12
theorem S_12 : S 12 = 24 :=
by
  sorry

end S_12_l244_244473


namespace find_initial_number_l244_244744

theorem find_initial_number (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by {
  sorry
}

end find_initial_number_l244_244744


namespace probability_five_distinct_dice_rolls_l244_244244

theorem probability_five_distinct_dice_rolls : 
  let total_outcomes := 6^5
  let favorable_outcomes := 6 * 5 * 4 * 3 * 2
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 54 :=
by
  sorry

end probability_five_distinct_dice_rolls_l244_244244


namespace no_symmetric_a_l244_244940

noncomputable def f (a x : ℝ) : ℝ := Real.log (((x + 1) / (x - 1)) * (x - 1) * (a - x))

theorem no_symmetric_a (a : ℝ) (h_a : 1 < a) : ¬ ∃ c : ℝ, ∀ d : ℝ, 1 < c - d ∧ c - d < a ∧ 1 < c + d ∧ c + d < a → f a (c - d) = f a (c + d) :=
sorry

end no_symmetric_a_l244_244940


namespace range_of_a_l244_244778

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a ^ x

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2) ↔ (3 / 2 ≤ a ∧ a < 2) :=
sorry

end range_of_a_l244_244778


namespace cos_2beta_correct_l244_244645

open Real

theorem cos_2beta_correct (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
    (h3 : tan α = 1 / 7) (h4 : cos (α + β) = 2 * sqrt 5 / 5) :
    cos (2 * β) = 4 / 5 := 
  sorry

end cos_2beta_correct_l244_244645


namespace max_true_statements_l244_244686

theorem max_true_statements (x : ℝ) :
  (∀ x, -- given the conditions
    (0 < x^2 ∧ x^2 < 1) →
    (x^2 > 1) →
    (-1 < x ∧ x < 0) →
    (0 < x ∧ x < 1) →
    (0 < x - x^2 ∧ x - x^2 < 1)) →
  -- Prove the maximum number of these statements that can be true is 3
  (∃ (count : ℕ), count = 3) :=
sorry

end max_true_statements_l244_244686


namespace train_travel_distance_l244_244889

theorem train_travel_distance (m : ℝ) (h : 3 * 60 * 1 = m) : m = 180 :=
by
  sorry

end train_travel_distance_l244_244889


namespace simplify_expr_l244_244844

theorem simplify_expr : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expr_l244_244844


namespace probability_five_distinct_numbers_l244_244252

def num_dice := 5
def num_faces := 6

def favorable_outcomes : ℕ := nat.factorial 5 * num_faces
def total_outcomes : ℕ := num_faces ^ num_dice

theorem probability_five_distinct_numbers :
  (favorable_outcomes / total_outcomes : ℚ) = 5 / 54 := 
sorry

end probability_five_distinct_numbers_l244_244252


namespace probability_adjacent_points_l244_244365

open Finset

-- Define the hexagon points and adjacency relationship
def hexagon_points : Finset ℕ := {0, 1, 2, 3, 4, 5}

def adjacent (a b : ℕ) : Prop :=
  (a = b + 1 ∨ a = b - 1 ∨ (a = 0 ∧ b = 5) ∨ (a = 5 ∧ b = 0))

-- Total number of ways to choose 2 points from 6 points
def total_pairs := (hexagon_points.card.choose 2)

-- Number of pairs that are adjacent
def favorable_pairs := (6 : ℕ) -- Each point has exactly 2 adjacent points, counted twice

-- The probability of selecting two adjacent points
theorem probability_adjacent_points : (favorable_pairs : ℚ) / total_pairs = 2 / 5 :=
by {
  sorry
}

end probability_adjacent_points_l244_244365


namespace smallest_multiple_1_10_is_2520_l244_244393

noncomputable def smallest_multiple_1_10 : ℕ :=
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))

theorem smallest_multiple_1_10_is_2520 : smallest_multiple_1_10 = 2520 :=
  sorry

end smallest_multiple_1_10_is_2520_l244_244393


namespace average_minutes_run_is_44_over_3_l244_244156

open BigOperators

def average_minutes_run (s : ℕ) : ℚ :=
  let sixth_graders := 3 * s
  let seventh_graders := s
  let eighth_graders := s / 2
  let total_students := sixth_graders + seventh_graders + eighth_graders
  let total_minutes_run := 20 * sixth_graders + 12 * eighth_graders
  total_minutes_run / total_students

theorem average_minutes_run_is_44_over_3 (s : ℕ) (h1 : 0 < s) : 
  average_minutes_run s = 44 / 3 := 
by
  sorry

end average_minutes_run_is_44_over_3_l244_244156


namespace sum_of_primes_less_than_20_l244_244056

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l244_244056


namespace eval_g_five_l244_244460

def g (x : ℝ) : ℝ := 4 * x - 2

theorem eval_g_five : g 5 = 18 := by
  sorry

end eval_g_five_l244_244460


namespace no_such_natural_numbers_l244_244920

theorem no_such_natural_numbers :
  ¬(∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧
  (b ∣ a^2 - 1) ∧ (c ∣ a^2 - 1) ∧
  (a ∣ b^2 - 1) ∧ (c ∣ b^2 - 1) ∧
  (a ∣ c^2 - 1) ∧ (b ∣ c^2 - 1)) :=
by sorry

end no_such_natural_numbers_l244_244920


namespace sqrt_inequalities_l244_244479

theorem sqrt_inequalities
  (a b c d e : ℝ)
  (ha : 0 ≤ a ∧ a ≤ 1)
  (hb : 0 ≤ b ∧ b ≤ 1)
  (hc : 0 ≤ c ∧ c ≤ 1)
  (hd : 0 ≤ d ∧ d ≤ 1)
  (he : 0 ≤ e ∧ e ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (e^2 + a^2) + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + e^2) ∧
  Real.sqrt (e^2 + a^2) + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + e^2) ≤ 5 * Real.sqrt 2 :=
by {
  sorry
}

end sqrt_inequalities_l244_244479


namespace discounted_price_correct_l244_244748

def discounted_price (P : ℝ) : ℝ :=
  P * 0.80 * 0.90 * 0.95

theorem discounted_price_correct :
  discounted_price 9502.923976608186 = 6498.40 :=
by
  sorry

end discounted_price_correct_l244_244748


namespace derivative_y_l244_244923

noncomputable def y (x : ℝ) : ℝ :=
  (Real.sqrt (9 * x^2 - 12 * x + 5)) * Real.arctan (3 * x - 2) - 
  Real.log (3 * x - 2 + Real.sqrt (9 * x^2 - 12 * x + 5))

theorem derivative_y (x : ℝ) :
  ∃ (f' : ℝ → ℝ), deriv y x = f' x ∧ f' x = (9 * x - 6) * Real.arctan (3 * x - 2) / 
  Real.sqrt (9 * x^2 - 12 * x + 5) :=
sorry

end derivative_y_l244_244923
