import Mathlib

namespace NUMINAMATH_GPT_area_of_equilateral_triangle_inscribed_in_square_l1545_154530

variables {a : ℝ}

noncomputable def equilateral_triangle_area (a : ℝ) : ℝ :=
  a^2 * (2 * Real.sqrt 3 - 3)

theorem area_of_equilateral_triangle_inscribed_in_square (a : ℝ) :
  equilateral_triangle_area a = a^2 * (2 * Real.sqrt 3 - 3) :=
by sorry

end NUMINAMATH_GPT_area_of_equilateral_triangle_inscribed_in_square_l1545_154530


namespace NUMINAMATH_GPT_exists_positive_integer_m_l1545_154574

noncomputable def d (g1 : ℝ) (r : ℝ) : ℝ := g1 * (r - 1)
noncomputable def a_n (n : ℕ) (a1 d : ℝ) : ℝ := a1 + (n - 1) * d
noncomputable def g_n (n : ℕ) (g1 : ℝ) (r : ℝ) : ℝ := g1 * (r ^ (n - 1))

theorem exists_positive_integer_m (a1 g1 : ℝ) (r : ℝ) (h0 : g1 ≠ 0) (h1 : a1 = g1) (h2 : a2 = g2)
(h3 : a_n 10 a1 (d g1 r) = g_n 3 g1 r) :
  ∀ (p : ℕ), ∃ (m : ℕ), g_n p g1 r = a_n m a1 (d g1 r) := by
  sorry

end NUMINAMATH_GPT_exists_positive_integer_m_l1545_154574


namespace NUMINAMATH_GPT_problem1_eval_problem2_eval_l1545_154573

-- Problem 1 equivalent proof problem
theorem problem1_eval : |(-2 + 1/4)| - (-3/4) + 1 - |(1 - 1/2)| = 3 + 1/2 := 
by
  sorry

-- Problem 2 equivalent proof problem
theorem problem2_eval : -3^2 - (8 / (-2)^3 - 1) + 3 / 2 * (1 / 2) = -6 + 1/4 :=
by
  sorry

end NUMINAMATH_GPT_problem1_eval_problem2_eval_l1545_154573


namespace NUMINAMATH_GPT_bricks_needed_for_wall_l1545_154581

noncomputable def brick_volume (length : ℝ) (height : ℝ) (thickness : ℝ) : ℝ :=
  length * height * thickness

noncomputable def wall_volume (length : ℝ) (height : ℝ) (average_thickness : ℝ) : ℝ :=
  length * height * average_thickness

noncomputable def number_of_bricks (wall_vol : ℝ) (brick_vol : ℝ) : ℝ :=
  wall_vol / brick_vol

theorem bricks_needed_for_wall : 
  let length_wall := 800
  let height_wall := 660
  let avg_thickness_wall := (25 + 22.5) / 2 -- in cm
  let length_brick := 25
  let height_brick := 11.25
  let thickness_brick := 6
  let mortar_thickness := 1

  let adjusted_length_brick := length_brick + mortar_thickness
  let adjusted_height_brick := height_brick + mortar_thickness

  let volume_wall := wall_volume length_wall height_wall avg_thickness_wall
  let volume_brick_with_mortar := brick_volume adjusted_length_brick adjusted_height_brick thickness_brick

  number_of_bricks volume_wall volume_brick_with_mortar = 6565 :=
by
  sorry

end NUMINAMATH_GPT_bricks_needed_for_wall_l1545_154581


namespace NUMINAMATH_GPT_question1_question2_question3_l1545_154572

-- Define the scores and relevant statistics for seventh and eighth grades
def seventh_grade_scores : List ℕ := [96, 86, 96, 86, 99, 96, 90, 100, 89, 82]
def eighth_grade_C_scores : List ℕ := [94, 90, 92]
def total_eighth_grade_students : ℕ := 800

def a := 40
def b := 93
def c := 96

-- Define given statistics from the table
def seventh_grade_mean := 92
def seventh_grade_variance := 34.6
def eighth_grade_mean := 91
def eighth_grade_median := 93
def eighth_grade_mode := 100
def eighth_grade_variance := 50.4

-- Proof for question 1
theorem question1 : (a = 40) ∧ (b = 93) ∧ (c = 96) :=
by sorry

-- Proof for question 2 (stability comparison)
theorem question2 : seventh_grade_variance < eighth_grade_variance :=
by sorry

-- Proof for question 3 (estimating number of excellent students)
theorem question3 : (7 / 10 : ℝ) * total_eighth_grade_students = 560 :=
by sorry

end NUMINAMATH_GPT_question1_question2_question3_l1545_154572


namespace NUMINAMATH_GPT_sum_of_first_30_terms_l1545_154505

variable (a : Nat → ℤ)
variable (d : ℤ)
variable (S_30 : ℤ)

-- Conditions from part a)
def condition1 := a 1 + a 2 + a 3 = 3
def condition2 := a 28 + a 29 + a 30 = 165

-- Question translated to Lean 4 statement
theorem sum_of_first_30_terms 
  (h1 : condition1 a)
  (h2 : condition2 a) :
  S_30 = 840 := 
sorry

end NUMINAMATH_GPT_sum_of_first_30_terms_l1545_154505


namespace NUMINAMATH_GPT_upper_limit_opinion_l1545_154526

theorem upper_limit_opinion (w : ℝ) 
  (H1 : 61 < w ∧ w < 72) 
  (H2 : 60 < w ∧ w < 70) 
  (H3 : (61 + w) / 2 = 63) : w = 65 := 
by
  sorry

end NUMINAMATH_GPT_upper_limit_opinion_l1545_154526


namespace NUMINAMATH_GPT_increased_consumption_5_percent_l1545_154589

theorem increased_consumption_5_percent (T C : ℕ) (h1 : ¬ (T = 0)) (h2 : ¬ (C = 0)) :
  (0.80 * (1 + x/100) = 0.84) → (x = 5) :=
by
  sorry

end NUMINAMATH_GPT_increased_consumption_5_percent_l1545_154589


namespace NUMINAMATH_GPT_find_missing_digit_l1545_154551

theorem find_missing_digit (B : ℕ) : 
  (B = 2 ∨ B = 4 ∨ B = 7 ∨ B = 8 ∨ B = 9) → 
  (2 * 1000 + B * 100 + 4 * 10 + 0) % 15 = 0 → 
  B = 7 :=
by 
  intro h1 h2
  sorry

end NUMINAMATH_GPT_find_missing_digit_l1545_154551


namespace NUMINAMATH_GPT_scientific_notation_l1545_154501

theorem scientific_notation (n : ℝ) (h1 : n = 17600) : ∃ a b, (a = 1.76) ∧ (b = 4) ∧ n = a * 10^b :=
by {
  sorry
}

end NUMINAMATH_GPT_scientific_notation_l1545_154501


namespace NUMINAMATH_GPT_exists_a_bc_l1545_154588

-- Definitions & Conditions
def satisfies_conditions (a b c : ℤ) : Prop :=
  - (b + c) - 10 = a ∧ (b + 10) * (c + 10) = 1

-- Theorem Statement
theorem exists_a_bc : ∃ (a b c : ℤ), satisfies_conditions a b c := by
  -- Substitute the correct proof below
  sorry

end NUMINAMATH_GPT_exists_a_bc_l1545_154588


namespace NUMINAMATH_GPT_calculate_down_payment_l1545_154552

theorem calculate_down_payment : 
  let monthly_fee := 12
  let years := 3
  let total_paid := 482
  let num_months := years * 12
  let total_monthly_payments := num_months * monthly_fee
  let down_payment := total_paid - total_monthly_payments
  down_payment = 50 :=
by
  sorry

end NUMINAMATH_GPT_calculate_down_payment_l1545_154552


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l1545_154544

theorem arithmetic_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2)
  (h2 : d ≠ 0)
  (h3 : ∀ n, S n ≤ S 8) :
  d < 0 ∧ S 17 ≤ 0 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l1545_154544


namespace NUMINAMATH_GPT_diameterOuterBoundary_l1545_154524

-- Definitions based on the conditions in the problem
def widthWalkingPath : ℝ := 10
def widthGardenRing : ℝ := 12
def diameterPond : ℝ := 16

-- The main theorem that proves the diameter of the circle that forms the outer boundary of the walking path
theorem diameterOuterBoundary : 2 * ((diameterPond / 2) + widthGardenRing + widthWalkingPath) = 60 :=
by
  sorry

end NUMINAMATH_GPT_diameterOuterBoundary_l1545_154524


namespace NUMINAMATH_GPT_proof_third_length_gcd_l1545_154546

/-- Statement: The greatest possible length that can be used to measure the given lengths exactly is 1 cm, 
and the third length is an unspecified number of centimeters that is relatively prime to both 1234 cm and 898 cm. -/
def third_length_gcd (x : ℕ) : Prop := 
  Int.gcd 1234 898 = 1 ∧ Int.gcd (Int.gcd 1234 898) x = 1

noncomputable def greatest_possible_length : ℕ := 1

theorem proof_third_length_gcd (x : ℕ) (h : third_length_gcd x) : greatest_possible_length = 1 := by
  sorry

end NUMINAMATH_GPT_proof_third_length_gcd_l1545_154546


namespace NUMINAMATH_GPT_fraction_equality_l1545_154542

theorem fraction_equality (x y : ℝ) : (-x + y) / (-x - y) = (x - y) / (x + y) :=
by sorry

end NUMINAMATH_GPT_fraction_equality_l1545_154542


namespace NUMINAMATH_GPT_peanut_total_correct_l1545_154508

-- Definitions based on the problem conditions:

def jose_peanuts : ℕ := 85
def kenya_peanuts : ℕ := jose_peanuts + 48
def malachi_peanuts : ℕ := kenya_peanuts + 35
def total_peanuts : ℕ := jose_peanuts + kenya_peanuts + malachi_peanuts

-- Statement to be proven:
theorem peanut_total_correct : total_peanuts = 386 :=
by 
  -- The proof would be here, but we skip it according to the instruction
  sorry

end NUMINAMATH_GPT_peanut_total_correct_l1545_154508


namespace NUMINAMATH_GPT_calculate_total_driving_time_l1545_154543

/--
A rancher needs to transport 400 head of cattle to higher ground 60 miles away.
His truck holds 20 head of cattle and travels at 60 miles per hour.
Prove that the total driving time to transport all cattle is 40 hours.
-/
theorem calculate_total_driving_time
  (total_cattle : Nat)
  (cattle_per_trip : Nat)
  (distance_one_way : Nat)
  (speed : Nat)
  (round_trip_miles : Nat)
  (total_miles : Nat)
  (total_time_hours : Nat)
  (h1 : total_cattle = 400)
  (h2 : cattle_per_trip = 20)
  (h3 : distance_one_way = 60)
  (h4 : speed = 60)
  (h5 : round_trip_miles = 2 * distance_one_way)
  (h6 : total_miles = (total_cattle / cattle_per_trip) * round_trip_miles)
  (h7 : total_time_hours = total_miles / speed) :
  total_time_hours = 40 :=
by
  sorry

end NUMINAMATH_GPT_calculate_total_driving_time_l1545_154543


namespace NUMINAMATH_GPT_tom_already_has_4_pounds_of_noodles_l1545_154596

-- Define the conditions
def beef : ℕ := 10
def noodle_multiplier : ℕ := 2
def packages : ℕ := 8
def weight_per_package : ℕ := 2

-- Define the total noodles needed
def total_noodles_needed : ℕ := noodle_multiplier * beef

-- Define the total noodles bought
def total_noodles_bought : ℕ := packages * weight_per_package

-- Define the already owned noodles
def already_owned_noodles : ℕ := total_noodles_needed - total_noodles_bought

-- State the theorem to prove
theorem tom_already_has_4_pounds_of_noodles :
  already_owned_noodles = 4 :=
  sorry

end NUMINAMATH_GPT_tom_already_has_4_pounds_of_noodles_l1545_154596


namespace NUMINAMATH_GPT_total_cost_of_bill_l1545_154503

def original_price_curtis := 16.00
def original_price_rob := 18.00
def time_of_meal := 3

def is_early_bird_discount_applicable (time : ℕ) : Prop :=
  2 ≤ time ∧ time ≤ 4

theorem total_cost_of_bill :
  is_early_bird_discount_applicable time_of_meal →
  original_price_curtis / 2 + original_price_rob / 2 = 17.00 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_bill_l1545_154503


namespace NUMINAMATH_GPT_symmetric_point_x_axis_l1545_154536

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetricWithRespectToXAxis (p : Point3D) : Point3D :=
  {x := p.x, y := -p.y, z := -p.z}

theorem symmetric_point_x_axis :
  symmetricWithRespectToXAxis ⟨-1, -2, 3⟩ = ⟨-1, 2, -3⟩ :=
  by
    sorry

end NUMINAMATH_GPT_symmetric_point_x_axis_l1545_154536


namespace NUMINAMATH_GPT_greatest_integer_func_l1545_154549

noncomputable def pi_approx : ℝ := 3.14159

theorem greatest_integer_func : (⌊2 * pi_approx - 6⌋ : ℝ) = 0 := 
by
  sorry

end NUMINAMATH_GPT_greatest_integer_func_l1545_154549


namespace NUMINAMATH_GPT_apples_difference_l1545_154548

def jimin_apples : ℕ := 7
def grandpa_apples : ℕ := 13
def younger_brother_apples : ℕ := 8
def younger_sister_apples : ℕ := 5

theorem apples_difference :
  grandpa_apples - younger_sister_apples = 8 :=
by
  sorry

end NUMINAMATH_GPT_apples_difference_l1545_154548


namespace NUMINAMATH_GPT_joan_paid_amount_l1545_154556

theorem joan_paid_amount (J K : ℕ) (h1 : J + K = 400) (h2 : 2 * J = K + 74) : J = 158 :=
by
  sorry

end NUMINAMATH_GPT_joan_paid_amount_l1545_154556


namespace NUMINAMATH_GPT_determine_sunday_l1545_154533

def Brother := Prop -- A type to represent a brother

variable (A B : Brother)
variable (T D : Brother) -- T representing Tweedledum, D representing Tweedledee

-- Conditions translated into Lean
variable (H1 : (A = T) → (B = D))
variable (H2 : (B = D) → (A = T))

-- Define the day of the week as a proposition
def is_sunday := Prop

-- We want to state that given H1 and H2, it is Sunday
theorem determine_sunday (H1 : (A = T) → (B = D)) (H2 : (B = D) → (A = T)) : is_sunday := sorry

end NUMINAMATH_GPT_determine_sunday_l1545_154533


namespace NUMINAMATH_GPT_rearrangement_inequality_l1545_154541

theorem rearrangement_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + c) + b / (a + c) + c / (a + b)) ≥ (3 / 2) ∧ (a = b ∧ b = c ∧ c = a ↔ (a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2)) :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_rearrangement_inequality_l1545_154541


namespace NUMINAMATH_GPT_rectangle_ratio_l1545_154511

theorem rectangle_ratio (s : ℝ) (h : s > 0) :
    let large_square_side := 3 * s
    let rectangle_length := 3 * s
    let rectangle_width := 2 * s
    rectangle_length / rectangle_width = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l1545_154511


namespace NUMINAMATH_GPT_total_miles_traveled_l1545_154509

-- Define the conditions
def travel_time_per_mile (n : ℕ) : ℕ :=
  match n with
  | 0 => 10
  | _ => 10 + 6 * n

def daily_miles (n : ℕ) : ℕ :=
  60 / travel_time_per_mile n

-- Statement of the problem
theorem total_miles_traveled : (daily_miles 0 + daily_miles 1 + daily_miles 2 + daily_miles 3 + daily_miles 4) = 20 := by
  sorry

end NUMINAMATH_GPT_total_miles_traveled_l1545_154509


namespace NUMINAMATH_GPT_tangent_identity_l1545_154540

theorem tangent_identity :
  Real.tan (55 * Real.pi / 180) * 
  Real.tan (65 * Real.pi / 180) * 
  Real.tan (75 * Real.pi / 180) = 
  Real.tan (85 * Real.pi / 180) :=
sorry

end NUMINAMATH_GPT_tangent_identity_l1545_154540


namespace NUMINAMATH_GPT_prime_n_if_power_of_prime_l1545_154566

theorem prime_n_if_power_of_prime (n : ℕ) (h1 : n ≥ 2) (b : ℕ) (h2 : b > 0) (p : ℕ) (k : ℕ) 
  (hk : k > 0) (hb : (b^n - 1) / (b - 1) = p^k) : Nat.Prime n :=
sorry

end NUMINAMATH_GPT_prime_n_if_power_of_prime_l1545_154566


namespace NUMINAMATH_GPT_find_smaller_number_l1545_154531

noncomputable def smaller_number (x y : ℝ) := y

theorem find_smaller_number 
  (x y : ℝ) 
  (h1 : x - y = 9) 
  (h2 : x + y = 46) :
  smaller_number x y = 18.5 :=
sorry

end NUMINAMATH_GPT_find_smaller_number_l1545_154531


namespace NUMINAMATH_GPT_curve_transformation_l1545_154521

theorem curve_transformation :
  (∀ (x y : ℝ), 2 * (5 * x)^2 + 8 * (3 * y)^2 = 1) → (∀ (x y : ℝ), 50 * x^2 + 72 * y^2 = 1) :=
by
  intros h x y
  have h1 : 2 * (5 * x)^2 + 8 * (3 * y)^2 = 1 := h x y
  sorry

end NUMINAMATH_GPT_curve_transformation_l1545_154521


namespace NUMINAMATH_GPT_sculpture_cost_in_cny_l1545_154578

-- Define the equivalence rates
def usd_to_nad : ℝ := 8
def usd_to_cny : ℝ := 8

-- Define the cost of the sculpture in Namibian dollars
def sculpture_cost_nad : ℝ := 160

-- Theorem: Given the conversion rates, the sculpture cost in Chinese yuan is 160
theorem sculpture_cost_in_cny : (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 160 :=
by sorry

end NUMINAMATH_GPT_sculpture_cost_in_cny_l1545_154578


namespace NUMINAMATH_GPT_amount_of_bill_is_1575_l1545_154563

noncomputable def time_in_years := (9 : ℝ) / 12

noncomputable def true_discount := 189
noncomputable def rate := 16

noncomputable def face_value (TD : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (TD * 100) / (R * T)

theorem amount_of_bill_is_1575 :
  face_value true_discount rate time_in_years = 1575 := by
  sorry

end NUMINAMATH_GPT_amount_of_bill_is_1575_l1545_154563


namespace NUMINAMATH_GPT_bill_buys_125_bouquets_to_make_1000_l1545_154591

-- Defining the conditions
def cost_per_bouquet : ℕ := 20
def roses_per_bouquet_buy : ℕ := 7
def roses_per_bouquet_sell : ℕ := 5
def target_difference : ℕ := 1000

-- To be demonstrated: number of bouquets Bill needs to buy to get a profit difference of $1000
theorem bill_buys_125_bouquets_to_make_1000 : 
  let total_cost_to_buy (n : ℕ) := n * cost_per_bouquet
  let total_roses (n : ℕ) := n * roses_per_bouquet_buy
  let bouquets_sell_from_roses (roses : ℕ) := roses / roses_per_bouquet_sell
  let total_revenue (bouquets : ℕ) := bouquets * cost_per_bouquet
  let profit (n : ℕ) := total_revenue (bouquets_sell_from_roses (total_roses n)) - total_cost_to_buy n
  profit (125 * 5) = target_difference := 
sorry

end NUMINAMATH_GPT_bill_buys_125_bouquets_to_make_1000_l1545_154591


namespace NUMINAMATH_GPT_parallelogram_properties_l1545_154568

variable {b h : ℕ}

theorem parallelogram_properties
  (hb : b = 20)
  (hh : h = 4) :
  (b * h = 80) ∧ ((b^2 + h^2) = 416) :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_properties_l1545_154568


namespace NUMINAMATH_GPT_bacteria_elimination_l1545_154518

theorem bacteria_elimination (d N : ℕ) (hN : N = 50 - 6 * (d - 1)) (hCondition : N ≤ 0) : d = 10 :=
by
  -- We can straightforwardly combine the given conditions and derive the required theorem.
  sorry

end NUMINAMATH_GPT_bacteria_elimination_l1545_154518


namespace NUMINAMATH_GPT_circle_equation_tangent_to_line_l1545_154557

theorem circle_equation_tangent_to_line
  (h k : ℝ) (A B C : ℝ)
  (hxk : h = 2) (hyk : k = -1) 
  (hA : A = 3) (hB : B = -4) (hC : C = 5)
  (r_squared : ℝ := (|A * h + B * k + C| / Real.sqrt (A^2 + B^2))^2)
  (h_radius : r_squared = 9) :
  ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r_squared := 
by
  sorry

end NUMINAMATH_GPT_circle_equation_tangent_to_line_l1545_154557


namespace NUMINAMATH_GPT_geom_sequence_eq_l1545_154517

theorem geom_sequence_eq :
  ∀ {a : ℕ → ℝ} {q : ℝ}, (∀ n, a (n + 1) = q * a n) →
  a 1 + a 2 + a 3 + a 4 + a 5 = 3 →
  a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 = 15 →
  a 1 - a 2 + a 3 - a 4 + a 5 = 5 :=
by
  intro a q hgeom hsum hsum_sq
  sorry

end NUMINAMATH_GPT_geom_sequence_eq_l1545_154517


namespace NUMINAMATH_GPT_y_decreases_as_x_increases_l1545_154598

-- Define the function y = 7 - x
def my_function (x : ℝ) : ℝ := 7 - x

-- Prove that y decreases as x increases
theorem y_decreases_as_x_increases : ∀ x1 x2 : ℝ, x1 < x2 → my_function x1 > my_function x2 := by
  intro x1 x2 h
  unfold my_function
  sorry

end NUMINAMATH_GPT_y_decreases_as_x_increases_l1545_154598


namespace NUMINAMATH_GPT_silver_medals_count_l1545_154515

def total_medals := 67
def gold_medals := 19
def bronze_medals := 16
def silver_medals := total_medals - gold_medals - bronze_medals

theorem silver_medals_count : silver_medals = 32 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_silver_medals_count_l1545_154515


namespace NUMINAMATH_GPT_willy_days_worked_and_missed_l1545_154525

theorem willy_days_worked_and_missed:
  ∃ (x : ℚ), 8 * x = 10 * (30 - x) ∧ x = 50/3 ∧ (30 - x) = 40/3 :=
by
  sorry

end NUMINAMATH_GPT_willy_days_worked_and_missed_l1545_154525


namespace NUMINAMATH_GPT_range_of_m_l1545_154586

theorem range_of_m (f : ℝ → ℝ) 
  (Hmono : ∀ x y, -2 ≤ x → x ≤ 2 → -2 ≤ y → y ≤ 2 → x ≤ y → f x ≤ f y)
  (Hineq : ∀ m, f (Real.log m / Real.log 2) < f (Real.log (m + 2) / Real.log 4))
  : ∀ m, (1 / 4 : ℝ) ≤ m ∧ m < 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1545_154586


namespace NUMINAMATH_GPT_plane_distance_l1545_154527

theorem plane_distance :
  let distance_AB := 100
  let distance_BC := distance_AB + 50
  let distance_CD := 2 * distance_BC
  let total_distance_AD := distance_AB + distance_BC + distance_CD
  total_distance_AD = 550 :=
by
  intros
  let distance_AB := 100
  let distance_BC := distance_AB + 50
  let distance_CD := 2 * distance_BC
  let total_distance_AD := distance_AB + distance_BC + distance_CD
  sorry

end NUMINAMATH_GPT_plane_distance_l1545_154527


namespace NUMINAMATH_GPT_rain_probability_tel_aviv_l1545_154597

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
  by
    sorry

end NUMINAMATH_GPT_rain_probability_tel_aviv_l1545_154597


namespace NUMINAMATH_GPT_find_actual_balance_l1545_154587

-- Define the given conditions
def current_balance : ℝ := 90000
def rate : ℝ := 0.10

-- Define the target
def actual_balance_before_deduction (X : ℝ) : Prop :=
  (X * (1 - rate) = current_balance)

-- Statement of the theorem
theorem find_actual_balance : ∃ X : ℝ, actual_balance_before_deduction X :=
  sorry

end NUMINAMATH_GPT_find_actual_balance_l1545_154587


namespace NUMINAMATH_GPT_intersecting_line_at_one_point_l1545_154516

theorem intersecting_line_at_one_point (k : ℝ) :
  (∃ y : ℝ, k = -3 * y^2 - 4 * y + 7 ∧ 
           ∀ z : ℝ, k = -3 * z^2 - 4 * z + 7 → y = z) ↔ 
  k = 25 / 3 :=
by
  sorry

end NUMINAMATH_GPT_intersecting_line_at_one_point_l1545_154516


namespace NUMINAMATH_GPT_vector_condition_l1545_154593

open Real

def acute_angle (a b : ℝ × ℝ) : Prop := 
  (a.1 * b.1 + a.2 * b.2) > 0

def not_collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 ≠ 0

theorem vector_condition (x : ℝ) :
  acute_angle (2, x + 1) (x + 2, 6) ∧ not_collinear (2, x + 1) (x + 2, 6) ↔ x > -5/4 ∧ x ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_vector_condition_l1545_154593


namespace NUMINAMATH_GPT_fraction_multiplication_l1545_154523

theorem fraction_multiplication :
  (3 / 4) ^ 5 * (4 / 3) ^ 2 = 8 / 19 :=
by
  sorry

end NUMINAMATH_GPT_fraction_multiplication_l1545_154523


namespace NUMINAMATH_GPT_Jenny_wants_to_read_three_books_l1545_154558

noncomputable def books : Nat := 3

-- Definitions based on provided conditions
def reading_speed : Nat := 100 -- words per hour
def book1_words : Nat := 200 
def book2_words : Nat := 400
def book3_words : Nat := 300
def daily_reading_minutes : Nat := 54 
def days : Nat := 10

-- Derived definitions for the proof
def total_words : Nat := book1_words + book2_words + book3_words
def total_hours_needed : ℚ := total_words / reading_speed
def daily_reading_hours : ℚ := daily_reading_minutes / 60
def total_reading_hours : ℚ := daily_reading_hours * days

theorem Jenny_wants_to_read_three_books :
  total_reading_hours = total_hours_needed → books = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_Jenny_wants_to_read_three_books_l1545_154558


namespace NUMINAMATH_GPT_find_y_l1545_154585

theorem find_y (x y : ℝ) (h1 : x^2 - 4 * x = y + 5) (h2 : x = 7) : y = 16 := by
  sorry

end NUMINAMATH_GPT_find_y_l1545_154585


namespace NUMINAMATH_GPT_multiplication_integer_multiple_l1545_154567

theorem multiplication_integer_multiple (a b n : ℕ) (ha : 100 ≤ a ∧ a < 1000) (hb : 100 ≤ b ∧ b < 1000) 
(h_eq : 10000 * a + b = n * (a * b)) : n = 73 := 
sorry

end NUMINAMATH_GPT_multiplication_integer_multiple_l1545_154567


namespace NUMINAMATH_GPT_find_other_cat_weight_l1545_154561

variable (cat1 cat2 dog : ℕ)

def weight_of_other_cat (cat1 cat2 dog : ℕ) : Prop :=
  cat1 = 7 ∧
  dog = 34 ∧
  dog = 2 * (cat1 + cat2) ∧
  cat2 = 10

theorem find_other_cat_weight (cat1 : ℕ) (cat2 : ℕ) (dog : ℕ) :
  weight_of_other_cat cat1 cat2 dog := by
  sorry

end NUMINAMATH_GPT_find_other_cat_weight_l1545_154561


namespace NUMINAMATH_GPT_am_gm_inequality_l1545_154583

variable {x y z : ℝ}

theorem am_gm_inequality (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (x + y + z) / 3 ≥ Real.sqrt (Real.sqrt (x * y) * Real.sqrt z) :=
by
  sorry

end NUMINAMATH_GPT_am_gm_inequality_l1545_154583


namespace NUMINAMATH_GPT_sum_of_3x3_matrix_arithmetic_eq_45_l1545_154553

-- Statement: Prove that the sum of all nine elements of a 3x3 matrix, where each row and each column forms an arithmetic sequence and the middle element a_{22} = 5, is 45
theorem sum_of_3x3_matrix_arithmetic_eq_45 
  (matrix : ℤ → ℤ → ℤ)
  (arithmetic_row : ∀ i, matrix i 0 + matrix i 1 + matrix i 2 = 3 * matrix i 1)
  (arithmetic_col : ∀ j, matrix 0 j + matrix 1 j + matrix 2 j = 3 * matrix 1 j)
  (middle_elem : matrix 1 1 = 5) : 
  (matrix 0 0 + matrix 0 1 + matrix 0 2 + matrix 1 0 + matrix 1 1 + matrix 1 2 + matrix 2 0 + matrix 2 1 + matrix 2 2) = 45 :=
by
  sorry -- proof to be provided

end NUMINAMATH_GPT_sum_of_3x3_matrix_arithmetic_eq_45_l1545_154553


namespace NUMINAMATH_GPT_mother_age_when_harry_born_l1545_154576

variable (harry_age father_age mother_age : ℕ)

-- Conditions
def harry_is_50 (harry_age : ℕ) : Prop := harry_age = 50
def father_is_24_years_older (harry_age father_age : ℕ) : Prop := father_age = harry_age + 24
def mother_younger_by_1_25_of_harry_age (harry_age father_age mother_age : ℕ) : Prop := mother_age = father_age - harry_age / 25

-- Proof Problem
theorem mother_age_when_harry_born (harry_age father_age mother_age : ℕ) 
  (h₁ : harry_is_50 harry_age) 
  (h₂ : father_is_24_years_older harry_age father_age)
  (h₃ : mother_younger_by_1_25_of_harry_age harry_age father_age mother_age) :
  mother_age - harry_age = 22 :=
by
  sorry

end NUMINAMATH_GPT_mother_age_when_harry_born_l1545_154576


namespace NUMINAMATH_GPT_unsuccessful_attempts_124_l1545_154528

theorem unsuccessful_attempts_124 (num_digits: ℕ) (choices_per_digit: ℕ) (total_attempts: ℕ):
  num_digits = 3 → choices_per_digit = 5 → total_attempts = choices_per_digit ^ num_digits →
  total_attempts - 1 = 124 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact sorry

end NUMINAMATH_GPT_unsuccessful_attempts_124_l1545_154528


namespace NUMINAMATH_GPT_Jennifer_apples_l1545_154550

-- Define the conditions
def initial_apples : ℕ := 7
def found_apples : ℕ := 74

-- The theorem to prove
theorem Jennifer_apples : initial_apples + found_apples = 81 :=
by
  -- proof goes here, but we use sorry to skip the proof step
  sorry

end NUMINAMATH_GPT_Jennifer_apples_l1545_154550


namespace NUMINAMATH_GPT_JacobProof_l1545_154534

def JacobLadders : Prop :=
  let costPerRung : ℤ := 2
  let costPer50RungLadder : ℤ := 50 * costPerRung
  let num50RungLadders : ℤ := 10
  let totalPayment : ℤ := 3400
  let cost1 : ℤ := num50RungLadders * costPer50RungLadder
  let remainingAmount : ℤ := totalPayment - cost1
  let numRungs20Ladders : ℤ := remainingAmount / costPerRung
  numRungs20Ladders = 1200

theorem JacobProof : JacobLadders := by
  sorry

end NUMINAMATH_GPT_JacobProof_l1545_154534


namespace NUMINAMATH_GPT_chickens_and_rabbits_l1545_154565

-- Let x be the number of chickens and y be the number of rabbits
variables (x y : ℕ)

-- Conditions: There are 35 heads and 94 feet in total
def heads_eq : Prop := x + y = 35
def feet_eq : Prop := 2 * x + 4 * y = 94

-- Proof statement (no proof is required, so we use sorry)
theorem chickens_and_rabbits :
  (heads_eq x y) ∧ (feet_eq x y) ↔ (x + y = 35 ∧ 2 * x + 4 * y = 94) :=
by
  sorry

end NUMINAMATH_GPT_chickens_and_rabbits_l1545_154565


namespace NUMINAMATH_GPT_domain_of_f_l1545_154562

-- Define the conditions for the function
def condition1 (x : ℝ) : Prop := 1 - x > 0
def condition2 (x : ℝ) : Prop := 3 * x + 1 > 0

-- Define the domain interval
def domain (x : ℝ) : Prop := -1 / 3 < x ∧ x < 1

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 / (Real.sqrt (1 - x)) + Real.log (3 * x + 1)

-- The main theorem to prove
theorem domain_of_f : 
  (∀ x : ℝ, condition1 x ∧ condition2 x ↔ domain x) :=
by {
  sorry
}

end NUMINAMATH_GPT_domain_of_f_l1545_154562


namespace NUMINAMATH_GPT_jennifer_fifth_score_l1545_154500

theorem jennifer_fifth_score :
  ∀ (x : ℝ), (85 + 90 + 87 + 92 + x) / 5 = 89 → x = 91 :=
by
  sorry

end NUMINAMATH_GPT_jennifer_fifth_score_l1545_154500


namespace NUMINAMATH_GPT_range_of_a_for_inequality_solutions_to_equation_l1545_154559

noncomputable def f (x a : ℝ) := x^2 + 2 * a * x + 1
noncomputable def f_prime (x a : ℝ) := 2 * x + 2 * a

theorem range_of_a_for_inequality :
  (∀ x, -2 ≤ x ∧ x ≤ -1 → f x a ≤ f_prime x a) → a ≥ 3 / 2 :=
sorry

theorem solutions_to_equation (a : ℝ) (x : ℝ) :
  f x a = |f_prime x a| ↔ 
  (if a < -1 then x = -1 ∨ x = 1 - 2 * a 
  else if -1 ≤ a ∧ a ≤ 1 then x = 1 ∨ x = -1 ∨ x = 1 - 2 * a ∨ x = -(1 + 2 * a)
  else x = 1 ∨ x = -(1 + 2 * a)) :=
sorry

end NUMINAMATH_GPT_range_of_a_for_inequality_solutions_to_equation_l1545_154559


namespace NUMINAMATH_GPT_value_of_5_S_3_l1545_154545

def S (a b : ℕ) : ℕ := 4 * a + 6 * b + 1

theorem value_of_5_S_3 : S 5 3 = 39 := by
  sorry

end NUMINAMATH_GPT_value_of_5_S_3_l1545_154545


namespace NUMINAMATH_GPT_real_number_solution_pure_imaginary_solution_zero_solution_l1545_154512

noncomputable def real_number_condition (m : ℝ) : Prop :=
  m^2 - 3 * m + 2 = 0

noncomputable def pure_imaginary_condition (m : ℝ) : Prop :=
  (2 * m^2 - 3 * m - 2 = 0) ∧ ¬(m^2 - 3 * m + 2 = 0)

noncomputable def zero_condition (m : ℝ) : Prop :=
  (2 * m^2 - 3 * m - 2 = 0) ∧ (m^2 - 3 * m + 2 = 0)

theorem real_number_solution (m : ℝ) : real_number_condition m ↔ (m = 1 ∨ m = 2) := 
sorry

theorem pure_imaginary_solution (m : ℝ) : pure_imaginary_condition m ↔ (m = -1 / 2) :=
sorry

theorem zero_solution (m : ℝ) : zero_condition m ↔ (m = 2) :=
sorry

end NUMINAMATH_GPT_real_number_solution_pure_imaginary_solution_zero_solution_l1545_154512


namespace NUMINAMATH_GPT_find_a2_l1545_154537

theorem find_a2 
  (a1 a2 a3 : ℝ)
  (h1 : a1 * a2 * a3 = 15)
  (h2 : (3 / (a1 * 3 * a2)) + (15 / (3 * a2 * 5 * a3)) + (5 / (5 * a3 * a1)) = 3 / 5) :
  a2 = 3 :=
sorry

end NUMINAMATH_GPT_find_a2_l1545_154537


namespace NUMINAMATH_GPT_area_outside_small_squares_l1545_154594

theorem area_outside_small_squares (a b : ℕ) (ha : a = 10) (hb : b = 4) (n : ℕ) (hn: n = 2) :
  a^2 - n * b^2 = 68 :=
by
  rw [ha, hb, hn]
  sorry

end NUMINAMATH_GPT_area_outside_small_squares_l1545_154594


namespace NUMINAMATH_GPT_center_of_circle_l1545_154529

theorem center_of_circle : 
  ∀ x y : ℝ, 4 * x^2 + 8 * x + 4 * y^2 - 12 * y + 29 = 0 → (x = -1 ∧ y = 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_center_of_circle_l1545_154529


namespace NUMINAMATH_GPT_arithmetic_sequence_value_l1545_154539

variable (a : ℕ → ℝ)
variable (a₁ d a₇ a₅ : ℝ)
variable (h_seq : ∀ n, a n = a₁ + (n - 1) * d)
variable (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120)

theorem arithmetic_sequence_value :
  a 7 - 1/3 * a 5 = 16 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_value_l1545_154539


namespace NUMINAMATH_GPT_first_discount_percentage_l1545_154513

theorem first_discount_percentage (original_price final_price : ℝ) (additional_discount : ℝ) (x : ℝ) 
  (h1 : original_price = 400) 
  (h2 : additional_discount = 0.05) 
  (h3 : final_price = 342) 
  (hx : (original_price * (100 - x) / 100) * (1 - additional_discount) = final_price) :
  x = 10 := 
sorry

end NUMINAMATH_GPT_first_discount_percentage_l1545_154513


namespace NUMINAMATH_GPT_count_japanese_stamps_l1545_154592

theorem count_japanese_stamps (total_stamps : ℕ) (perc_chinese perc_us : ℕ) (h1 : total_stamps = 100) 
  (h2 : perc_chinese = 35) (h3 : perc_us = 20): 
  total_stamps - ((perc_chinese * total_stamps / 100) + (perc_us * total_stamps / 100)) = 45 :=
by
  sorry

end NUMINAMATH_GPT_count_japanese_stamps_l1545_154592


namespace NUMINAMATH_GPT_cos_relation_l1545_154580

theorem cos_relation 
  (a b c A B C : ℝ)
  (h1 : a = b * Real.cos C + c * Real.cos B)
  (h2 : b = c * Real.cos A + a * Real.cos C)
  (h3 : c = a * Real.cos B + b * Real.cos A)
  (h_abc_nonzero : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :
  Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 + 2 * Real.cos A * Real.cos B * Real.cos C = 1 :=
sorry

end NUMINAMATH_GPT_cos_relation_l1545_154580


namespace NUMINAMATH_GPT_rug_length_l1545_154504

theorem rug_length (d : ℕ) (x y : ℕ) (h1 : x * x + y * y = d * d) (h2 : y / x = 2) (h3 : (x = 25 ∧ y = 50)) : 
  x = 25 := 
sorry

end NUMINAMATH_GPT_rug_length_l1545_154504


namespace NUMINAMATH_GPT_min_disks_required_for_files_l1545_154595

theorem min_disks_required_for_files :
  ∀ (number_of_files : ℕ)
    (files_0_9MB : ℕ)
    (files_0_6MB : ℕ)
    (disk_capacity_MB : ℝ)
    (file_size_0_9MB : ℝ)
    (file_size_0_6MB : ℝ)
    (file_size_0_45MB : ℝ),
  number_of_files = 40 →
  files_0_9MB = 5 →
  files_0_6MB = 15 →
  disk_capacity_MB = 1.44 →
  file_size_0_9MB = 0.9 →
  file_size_0_6MB = 0.6 →
  file_size_0_45MB = 0.45 →
  ∃ (min_disks : ℕ), min_disks = 16 :=
by
  sorry

end NUMINAMATH_GPT_min_disks_required_for_files_l1545_154595


namespace NUMINAMATH_GPT_chips_removal_even_initial_40_chips_removal_minimum_moves_1000_l1545_154569

-- Part a: Prove that with 40 chips, exactly one chip cannot remain after both players have made two moves.
theorem chips_removal_even_initial_40 
  (initial_chips : Nat)
  (num_moves : Nat)
  (remaining_chips : Nat) :
  initial_chips = 40 → 
  num_moves = 4 → 
  remaining_chips = 1 → 
  False :=
by
  sorry

-- Part b: Prove that with 1000 chips, the minimum number of moves to reduce to one chip is 8.
theorem chips_removal_minimum_moves_1000
  (initial_chips : Nat)
  (min_moves : Nat)
  (remaining_chips : Nat) :
  initial_chips = 1000 → 
  remaining_chips = 1 → 
  min_moves = 8 :=
by
  sorry

end NUMINAMATH_GPT_chips_removal_even_initial_40_chips_removal_minimum_moves_1000_l1545_154569


namespace NUMINAMATH_GPT_football_field_area_l1545_154519

theorem football_field_area (total_fertilizer : ℝ) (partial_fertilizer : ℝ) (partial_area : ℝ) (fertilizer_rate : ℝ) (total_area : ℝ) 
  (h1 : total_fertilizer = 800)
  (h2: partial_fertilizer = 300)
  (h3: partial_area = 3600)
  (h4: fertilizer_rate = partial_fertilizer / partial_area)
  (h5: total_area = total_fertilizer / fertilizer_rate) 
  : total_area = 9600 := 
sorry

end NUMINAMATH_GPT_football_field_area_l1545_154519


namespace NUMINAMATH_GPT_stamp_problem_solution_l1545_154590

theorem stamp_problem_solution : ∃ n : ℕ, n > 1 ∧ (∀ m : ℕ, m ≥ 2 * n + 2 → ∃ a b : ℕ, m = n * a + (n + 2) * b) ∧ ∀ x : ℕ, 1 < x ∧ (∀ m : ℕ, m ≥ 2 * x + 2 → ∃ a b : ℕ, m = x * a + (x + 2) * b) → x ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_stamp_problem_solution_l1545_154590


namespace NUMINAMATH_GPT_max_dist_to_origin_from_curve_l1545_154506

noncomputable def curve (θ : ℝ) : ℝ × ℝ :=
  let x := 3 + Real.sin θ
  let y := Real.cos θ
  (x, y)

theorem max_dist_to_origin_from_curve :
  ∃ M : ℝ × ℝ, (∃ θ : ℝ, M = curve θ) ∧ Real.sqrt (M.fst^2 + M.snd^2) ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_max_dist_to_origin_from_curve_l1545_154506


namespace NUMINAMATH_GPT_arithmetic_sequence_n_value_l1545_154555

theorem arithmetic_sequence_n_value (a : ℕ → ℤ) (a1 : a 1 = 1) (d : ℤ) (d_def : d = 3) (an : ∃ n, a n = 22) :
  ∃ n, n = 8 :=
by
  -- Assume the general term formula for the arithmetic sequence
  have general_term : ∀ n, a n = a 1 + (n-1) * d := sorry
  -- Use the given conditions
  have a_n_22 : ∃ n, a n = 22 := an
  -- Calculations to derive n = 8, skipped here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_value_l1545_154555


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l1545_154507

theorem geometric_series_common_ratio (a : ℝ) (r : ℝ) (S : ℝ) (h1 : S = a / (1 - r))
  (h2 : S = 16 * (r^2 * S)) : |r| = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l1545_154507


namespace NUMINAMATH_GPT_circle_center_l1545_154564

theorem circle_center (x y : ℝ) (h : x^2 - 4 * x + y^2 - 6 * y - 12 = 0) : (x, y) = (2, 3) :=
sorry

end NUMINAMATH_GPT_circle_center_l1545_154564


namespace NUMINAMATH_GPT_range_of_m_for_inequality_l1545_154520

-- Define the condition
def condition (x : ℝ) := x ∈ Set.Iic (-1)

-- Define the inequality for proving the range of m
def inequality_holds (m x : ℝ) : Prop := (m - m^2) * 4^x + 2^x + 1 > 0

-- Prove the range of m for the given conditions such that the inequality holds
theorem range_of_m_for_inequality :
  (∀ (x : ℝ), condition x → inequality_holds m x) ↔ (-2 < m ∧ m < 3) :=
sorry

end NUMINAMATH_GPT_range_of_m_for_inequality_l1545_154520


namespace NUMINAMATH_GPT_total_number_of_animals_l1545_154538

-- Define the given conditions as hypotheses
def num_horses (T : ℕ) : Prop :=
  ∃ (H x z : ℕ), H + x + z = 75

def cows_vs_horses (T : ℕ) : Prop :=
  ∃ (w z : ℕ),  w = z + 10

-- Define the final conclusion we need to prove
def total_animals (T : ℕ) : Prop :=
  T = 170

-- The main theorem which states the conditions imply the conclusion
theorem total_number_of_animals (T : ℕ) (h1 : num_horses T) (h2 : cows_vs_horses T) : total_animals T :=
by
  -- Proof to be filled in later
  sorry

end NUMINAMATH_GPT_total_number_of_animals_l1545_154538


namespace NUMINAMATH_GPT_vertex_property_l1545_154570

theorem vertex_property (a b c m k : ℝ) (h : a ≠ 0)
  (vertex_eq : k = a * m^2 + b * m + c)
  (point_eq : m = a * k^2 + b * k + c) : a * (m - k) > 0 :=
sorry

end NUMINAMATH_GPT_vertex_property_l1545_154570


namespace NUMINAMATH_GPT_myrtle_hens_l1545_154554

/-- Myrtle has some hens that lay 3 eggs a day. She was gone for 7 days and told her neighbor 
    to take as many as they would like. The neighbor took 12 eggs. Once home, Myrtle collected 
    the remaining eggs, dropping 5 on the way into her house. Myrtle has 46 eggs. Prove 
    that Myrtle has 3 hens. -/
theorem myrtle_hens (eggs_per_hen_per_day hens days neighbor_took dropped remaining_hens_eggs : ℕ) 
    (h1 : eggs_per_hen_per_day = 3) 
    (h2 : days = 7) 
    (h3 : neighbor_took = 12) 
    (h4 : dropped = 5) 
    (h5 : remaining_hens_eggs = 46) : 
    hens = 3 := 
by 
  sorry

end NUMINAMATH_GPT_myrtle_hens_l1545_154554


namespace NUMINAMATH_GPT_length_of_second_train_l1545_154502

/-
  Given:
  - l₁ : Length of the first train in meters
  - v₁ : Speed of the first train in km/h
  - v₂ : Speed of the second train in km/h
  - t : Time to cross the second train in seconds

  Prove:
  - l₂ : Length of the second train in meters = 299.9560035197185 meters
-/

variable (l₁ : ℝ) (v₁ : ℝ) (v₂ : ℝ) (t : ℝ) (l₂ : ℝ)

theorem length_of_second_train
  (h₁ : l₁ = 250)
  (h₂ : v₁ = 72)
  (h₃ : v₂ = 36)
  (h₄ : t = 54.995600351971845)
  (h_result : l₂ = 299.9560035197185) :
  (v₁ * 1000 / 3600 - v₂ * 1000 / 3600) * t - l₁ = l₂ := by
  sorry

end NUMINAMATH_GPT_length_of_second_train_l1545_154502


namespace NUMINAMATH_GPT_units_digit_of_fraction_example_l1545_154535

def units_digit_of_fraction (n d : ℕ) : ℕ :=
  (n / d) % 10

theorem units_digit_of_fraction_example :
  units_digit_of_fraction (25 * 26 * 27 * 28 * 29 * 30) 1250 = 2 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_fraction_example_l1545_154535


namespace NUMINAMATH_GPT_sum_of_odd_integers_from_13_to_53_l1545_154577

-- Definition of the arithmetic series summing from 13 to 53 with common difference 2
def sum_of_arithmetic_series (a l d : ℕ) (n : ℕ) : ℕ :=
  (n * (a + l)) / 2

-- Main theorem
theorem sum_of_odd_integers_from_13_to_53 :
  sum_of_arithmetic_series 13 53 2 21 = 693 := 
sorry

end NUMINAMATH_GPT_sum_of_odd_integers_from_13_to_53_l1545_154577


namespace NUMINAMATH_GPT_tournament_matches_divisible_by_7_l1545_154547

-- Define the conditions of the chess tournament
def single_elimination_tournament_matches (players byes: ℕ) : ℕ :=
  players - 1

theorem tournament_matches_divisible_by_7 :
  single_elimination_tournament_matches 120 40 = 119 ∧ 119 % 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_tournament_matches_divisible_by_7_l1545_154547


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1545_154584

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_seq : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h1 : a 1 + a 3 + a 5 = 9) (h2 : a 2 + a 4 + a 6 = 15) : a 3 + a 4 = 8 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1545_154584


namespace NUMINAMATH_GPT_exponent_property_l1545_154582

variable {a : ℝ} {m n : ℕ}

theorem exponent_property (h1 : a^m = 2) (h2 : a^n = 3) : a^(2*m + n) = 12 :=
sorry

end NUMINAMATH_GPT_exponent_property_l1545_154582


namespace NUMINAMATH_GPT_num_orders_javier_constraint_l1545_154522

noncomputable def num_valid_orders : ℕ :=
  Nat.factorial 5 / 2

theorem num_orders_javier_constraint : num_valid_orders = 60 := 
by
  sorry

end NUMINAMATH_GPT_num_orders_javier_constraint_l1545_154522


namespace NUMINAMATH_GPT_edge_length_of_cube_l1545_154599

noncomputable def cost_per_quart : ℝ := 3.20
noncomputable def coverage_per_quart : ℕ := 120
noncomputable def total_cost : ℝ := 16
noncomputable def total_coverage : ℕ := 600 -- From 5 quarts * 120 square feet per quart
noncomputable def surface_area (edge_length : ℝ) : ℝ := 6 * (edge_length ^ 2)

theorem edge_length_of_cube :
  (∃ edge_length : ℝ, surface_area edge_length = total_coverage) → 
  ∃ edge_length : ℝ, edge_length = 10 :=
by
  sorry

end NUMINAMATH_GPT_edge_length_of_cube_l1545_154599


namespace NUMINAMATH_GPT_empty_with_three_pumps_in_12_minutes_l1545_154510

-- Define the conditions
def conditions (a b x : ℝ) : Prop :=
  x = a + b ∧ 2 * x = 3 * a + b

-- Define the main theorem to prove
theorem empty_with_three_pumps_in_12_minutes (a b x : ℝ) (h : conditions a b x) : 
  (3 * (1 / 5) * x = a + (1 / 5) * b) ∧ ((1 / 5) * 60 = 12) := 
by
  -- Use the given conditions in the proof.
  sorry

end NUMINAMATH_GPT_empty_with_three_pumps_in_12_minutes_l1545_154510


namespace NUMINAMATH_GPT_coefficient_x6_in_expansion_l1545_154579

theorem coefficient_x6_in_expansion :
  (∃ c : ℕ, c = 81648 ∧ (3 : ℝ) ^ 6 * c * 2 ^ 2  = c * (3 : ℝ) ^ 6 * 4) :=
sorry

end NUMINAMATH_GPT_coefficient_x6_in_expansion_l1545_154579


namespace NUMINAMATH_GPT_factor_difference_of_squares_l1545_154571
noncomputable def factored (t : ℝ) : ℝ := (t - 8) * (t + 8)

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = factored t := by
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l1545_154571


namespace NUMINAMATH_GPT_circle_radius_l1545_154560

theorem circle_radius {r : ℤ} (center: ℝ × ℝ) (inside_pt: ℝ × ℝ) (outside_pt: ℝ × ℝ)
  (h_center: center = (2, 1))
  (h_inside: dist center inside_pt < r)
  (h_outside: dist center outside_pt > r)
  (h_inside_pt: inside_pt = (-2, 1))
  (h_outside_pt: outside_pt = (2, -5))
  (h_integer: r > 0) :
  r = 5 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l1545_154560


namespace NUMINAMATH_GPT_m_plus_n_eq_123_l1545_154514

/- Define the smallest prime number -/
def m : ℕ := 2

/- Define the largest integer less than 150 with exactly three positive divisors -/
def n : ℕ := 121

/- Prove that the sum of m and n is 123 -/
theorem m_plus_n_eq_123 : m + n = 123 := by
  -- By definition, m is 2 and n is 121
  -- So, their sum is 123
  rfl

end NUMINAMATH_GPT_m_plus_n_eq_123_l1545_154514


namespace NUMINAMATH_GPT_product_of_ab_l1545_154575

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end NUMINAMATH_GPT_product_of_ab_l1545_154575


namespace NUMINAMATH_GPT_prism_surface_area_is_14_l1545_154532

-- Definition of the rectangular prism dimensions
def prism_length : ℕ := 3
def prism_width : ℕ := 1
def prism_height : ℕ := 1

-- Definition of the surface area of the rectangular prism
def surface_area (l w h : ℕ) : ℕ :=
  2 * (l * w + w * h + h * l)

-- Theorem statement: The surface area of the resulting prism is 14
theorem prism_surface_area_is_14 : surface_area prism_length prism_width prism_height = 14 :=
  sorry

end NUMINAMATH_GPT_prism_surface_area_is_14_l1545_154532
