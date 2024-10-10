import Mathlib

namespace extremum_and_solutions_l1483_148323

/-- A function with an extremum at x = 0 -/
noncomputable def f (a b x : ℝ) : ℝ := x^2 + x - Real.log (x + a) + 3*b

/-- The statement to be proved -/
theorem extremum_and_solutions (a b : ℝ) :
  (f a b 0 = 0 ∧ ∀ x, f a b x ≥ f a b 0) →
  (a = 1 ∧ b = 0) ∧
  ∀ m : ℝ, (∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, -1/2 ≤ x ∧ x ≤ 2 ∧ f 1 0 x = m) ↔
    (0 < m ∧ m ≤ -1/4 + Real.log 2) :=
by sorry


end extremum_and_solutions_l1483_148323


namespace quartic_arithmetic_sequence_roots_l1483_148336

/-- The coefficients of a quartic equation whose roots form an arithmetic sequence -/
theorem quartic_arithmetic_sequence_roots (C D : ℝ) :
  (∃ (a d : ℝ), {a - 3*d, a - d, a + d, a + 3*d} = 
    {x : ℝ | x^4 + 4*x^3 - 34*x^2 + C*x + D = 0}) →
  C = -76 ∧ D = 105 := by
  sorry


end quartic_arithmetic_sequence_roots_l1483_148336


namespace bruce_egg_count_after_loss_l1483_148313

/-- Given Bruce's initial egg count and the number of eggs he loses,
    calculate Bruce's final egg count. -/
def bruces_final_egg_count (initial_count : ℕ) (eggs_lost : ℕ) : ℕ :=
  initial_count - eggs_lost

/-- Theorem stating that given Bruce's initial egg count of 215 and a loss of 137 eggs,
    Bruce's final egg count is 78. -/
theorem bruce_egg_count_after_loss :
  bruces_final_egg_count 215 137 = 78 := by
  sorry

end bruce_egg_count_after_loss_l1483_148313


namespace ratio_w_to_y_l1483_148338

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 5 / 4)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 4) :
  w / y = 10 / 3 := by
  sorry

end ratio_w_to_y_l1483_148338


namespace a_equals_zero_l1483_148324

theorem a_equals_zero (a : ℝ) : 
  let A : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}
  1 ∈ A → a = 0 := by
sorry

end a_equals_zero_l1483_148324


namespace max_books_borrowed_l1483_148359

theorem max_books_borrowed (total_students : Nat) (zero_books : Nat) (one_book : Nat) 
  (two_books : Nat) (three_books : Nat) (avg_books : Nat) (max_books : Nat) :
  total_students = 50 →
  zero_books = 4 →
  one_book = 15 →
  two_books = 9 →
  three_books = 7 →
  avg_books = 3 →
  max_books = 10 →
  ∃ (max_single : Nat),
    max_single ≤ max_books ∧
    max_single = 40 ∧
    (total_students * avg_books - (one_book + 2 * two_books + 3 * three_books)) % 2 = 0 := by
  sorry

end max_books_borrowed_l1483_148359


namespace jack_evening_emails_l1483_148327

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 3

/-- The total number of emails Jack received in the morning and evening combined -/
def morning_evening_total : ℕ := 11

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := morning_evening_total - morning_emails

theorem jack_evening_emails : evening_emails = 8 := by
  sorry

end jack_evening_emails_l1483_148327


namespace ratio_equivalence_l1483_148341

theorem ratio_equivalence (x : ℚ) : (3 / x = 3 / 16) → x = 16 := by
  sorry

end ratio_equivalence_l1483_148341


namespace no_real_roots_l1483_148353

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem no_real_roots :
  ¬∃ x : ℝ, 1 - log10 (Real.sin x) = Real.cos x :=
sorry

end no_real_roots_l1483_148353


namespace expression_factorization_l1483_148395

theorem expression_factorization (x : ℝ) :
  (3 * x^3 + 70 * x^2 - 5) - (-4 * x^3 + 2 * x^2 - 5) = 7 * x^2 * (x + 68/7) := by
  sorry

end expression_factorization_l1483_148395


namespace logarithm_inconsistency_l1483_148386

-- Define a custom logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the given logarithmic values
def lg3 : ℝ := 0.47712
def lg1_5 : ℝ := 0.17609
def lg5 : ℝ := 0.69897
def lg2 : ℝ := 0.30103
def lg7_incorrect : ℝ := 0.84519

-- Theorem statement
theorem logarithm_inconsistency :
  lg 3 = lg3 ∧
  lg 1.5 = lg1_5 ∧
  lg 5 = lg5 ∧
  lg 2 = lg2 ∧
  lg 7 ≠ lg7_incorrect :=
by sorry

end logarithm_inconsistency_l1483_148386


namespace solution_set_equivalence_l1483_148366

/-- The set of real numbers x that satisfy (x+2)/(x-4) ≥ 3 is exactly the interval (4, 7]. -/
theorem solution_set_equivalence (x : ℝ) : (x + 2) / (x - 4) ≥ 3 ↔ x ∈ Set.Ioo 4 7 ∪ {7} := by
  sorry

end solution_set_equivalence_l1483_148366


namespace solution_set_inequality_l1483_148303

theorem solution_set_inequality (a : ℝ) (h : (4 : ℝ)^a = 2^(a + 2)) :
  {x : ℝ | a^(2*x + 1) > a^(x - 1)} = {x : ℝ | x > -2} := by
sorry

end solution_set_inequality_l1483_148303


namespace circle_with_diameter_OC_l1483_148306

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 6)^2 + (y - 8)^2 = 4

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the center of circle C
def center_C : ℝ × ℝ := (6, 8)

-- Define the equation of the circle with diameter OC
def circle_OC (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 25

-- Theorem statement
theorem circle_with_diameter_OC :
  ∀ x y : ℝ, circle_C x y → circle_OC x y :=
sorry

end circle_with_diameter_OC_l1483_148306


namespace probability_is_two_ninety_one_l1483_148384

/-- Represents the number of jellybeans of each color in the basket -/
structure JellyBeanBasket where
  red : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the probability of picking exactly 2 red and 2 blue jellybeans -/
def probability_two_red_two_blue (basket : JellyBeanBasket) : Rat :=
  let total := basket.red + basket.blue + basket.yellow
  let favorable := Nat.choose basket.red 2 * Nat.choose basket.blue 2
  let total_combinations := Nat.choose total 4
  favorable / total_combinations

/-- The main theorem stating the probability is 2/91 -/
theorem probability_is_two_ninety_one :
  probability_two_red_two_blue ⟨5, 3, 7⟩ = 2 / 91 := by
  sorry

end probability_is_two_ninety_one_l1483_148384


namespace water_level_rise_rate_l1483_148328

/-- The rate at which the water level rises in a cylinder when water drains from a cube -/
theorem water_level_rise_rate (cube_side : ℝ) (cylinder_radius : ℝ) (cube_fall_rate : ℝ) :
  cube_side = 100 →
  cylinder_radius = 100 →
  cube_fall_rate = 1 →
  (cylinder_radius ^ 2 * π) * (cube_side ^ 2 * cube_fall_rate) / (cylinder_radius ^ 2 * π) ^ 2 = 1 / π := by
  sorry

end water_level_rise_rate_l1483_148328


namespace four_digit_numbers_count_four_digit_numbers_exist_l1483_148351

def A (n m : ℕ) := n.factorial / (n - m).factorial

theorem four_digit_numbers_count : ℕ → Prop :=
  fun count => (count = A 5 4 - A 4 3) ∧ 
               (count = A 4 1 * A 4 3) ∧ 
               (count = A 4 4 + 3 * A 4 3) ∧ 
               (count ≠ A 5 4 - A 4 4)

theorem four_digit_numbers_exist : ∃ count : ℕ, four_digit_numbers_count count := by
  sorry

end four_digit_numbers_count_four_digit_numbers_exist_l1483_148351


namespace walking_jogging_time_difference_l1483_148333

/-- 
Given:
- Linda walks at 4 miles per hour
- Tom jogs at 9 miles per hour
- Linda starts walking 1 hour before Tom
- They walk in opposite directions

Prove that the difference in time (in minutes) for Tom to cover half and twice Linda's distance is 40 minutes.
-/
theorem walking_jogging_time_difference 
  (linda_speed : ℝ) 
  (tom_speed : ℝ) 
  (head_start : ℝ) :
  linda_speed = 4 →
  tom_speed = 9 →
  head_start = 1 →
  let linda_distance := linda_speed * head_start
  let half_distance := linda_distance / 2
  let double_distance := linda_distance * 2
  let time_half := (half_distance / tom_speed) * 60
  let time_double := (double_distance / tom_speed) * 60
  time_double - time_half = 40 := by
  sorry

end walking_jogging_time_difference_l1483_148333


namespace unchanged_total_plates_l1483_148349

/-- Represents the number of elements in each set of letters for license plates --/
structure LicensePlateSets :=
  (first : Nat)
  (second : Nat)
  (third : Nat)

/-- Calculates the total number of possible license plates --/
def totalPlates (sets : LicensePlateSets) : Nat :=
  sets.first * sets.second * sets.third

/-- The original configuration of letter sets --/
def originalSets : LicensePlateSets :=
  { first := 5, second := 3, third := 4 }

/-- The new configuration after moving one letter from the first to the third set --/
def newSets : LicensePlateSets :=
  { first := 4, second := 3, third := 5 }

/-- Theorem stating that the total number of license plates remains unchanged --/
theorem unchanged_total_plates :
  totalPlates originalSets = totalPlates newSets :=
by sorry

end unchanged_total_plates_l1483_148349


namespace tens_digit_of_8_pow_306_l1483_148381

/-- The function that returns the last two digits of 8^n -/
def lastTwoDigits (n : ℕ) : ℕ := (8^n) % 100

/-- The length of the cycle of last two digits of powers of 8 -/
def cycleLength : ℕ := 6

/-- The function that returns the tens digit of a number -/
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

theorem tens_digit_of_8_pow_306 : tensDigit (lastTwoDigits 306) = 6 := by
  sorry

end tens_digit_of_8_pow_306_l1483_148381


namespace divergent_series_convergent_combination_l1483_148318

/-- Two positive sequences with divergent series but convergent combined series -/
theorem divergent_series_convergent_combination :
  ∃ (a b : ℕ → ℝ),
    (∀ n, a n > 0) ∧
    (∀ n, b n > 0) ∧
    (¬ Summable a) ∧
    (¬ Summable b) ∧
    Summable (λ n ↦ (2 * a n * b n) / (a n + b n)) := by
  sorry

end divergent_series_convergent_combination_l1483_148318


namespace stock_percentage_sold_l1483_148390

/-- Proves that the percentage of stock sold is 0.25% given the specified conditions --/
theorem stock_percentage_sold (cash_realized : ℝ) (brokerage_rate : ℝ) (net_amount : ℝ)
  (h1 : cash_realized = 108.25)
  (h2 : brokerage_rate = 1 / 4 / 100)
  (h3 : net_amount = 108) :
  let brokerage_fee := cash_realized * brokerage_rate
  let percentage_sold := brokerage_fee / cash_realized * 100
  percentage_sold = 0.25 := by sorry

end stock_percentage_sold_l1483_148390


namespace trapezoid_perimeter_l1483_148358

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  AD : ℝ
  height : ℝ
  is_trapezoid : True
  AB_eq_CD : AB = CD
  BC_eq_10 : BC = 10
  AD_eq_22 : AD = 22
  height_eq_5 : height = 5

/-- The perimeter of the trapezoid ABCD is 2√61 + 32 -/
theorem trapezoid_perimeter (t : Trapezoid) : 
  t.AB + t.BC + t.CD + t.AD = 2 * Real.sqrt 61 + 32 := by
  sorry


end trapezoid_perimeter_l1483_148358


namespace gcd_lcm_sum_72_8712_l1483_148376

theorem gcd_lcm_sum_72_8712 : Nat.gcd 72 8712 + Nat.lcm 72 8712 = 26160 := by
  sorry

end gcd_lcm_sum_72_8712_l1483_148376


namespace negation_of_exp_positive_forall_l1483_148357

theorem negation_of_exp_positive_forall :
  (¬ ∀ x : ℝ, Real.exp x > 0) ↔ (∃ x : ℝ, Real.exp x ≤ 0) := by sorry

end negation_of_exp_positive_forall_l1483_148357


namespace boat_speed_in_still_water_l1483_148332

/-- Proves that the speed of a boat in still water is 25 km/hr, given the speed of the stream
    and the time and distance traveled downstream. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_time = 3)
  (h3 : downstream_distance = 90) :
  (downstream_distance / downstream_time) - stream_speed = 25 := by
  sorry

#check boat_speed_in_still_water

end boat_speed_in_still_water_l1483_148332


namespace factor_problem_l1483_148334

theorem factor_problem (initial_number : ℕ) (factor : ℚ) : 
  initial_number = 6 → 
  (2 * initial_number + 9) * factor = 63 → 
  factor = 3 := by
sorry

end factor_problem_l1483_148334


namespace triangles_in_decagon_count_l1483_148380

/-- The number of triangles that can be formed from the vertices of a regular decagon -/
def trianglesInDecagon : ℕ := 120

/-- Proof that the number of triangles in a regular decagon is correct -/
theorem triangles_in_decagon_count : 
  (Finset.univ.filter (λ s : Finset (Fin 10) => s.card = 3)).card = trianglesInDecagon := by
  sorry

end triangles_in_decagon_count_l1483_148380


namespace complex_magnitude_l1483_148304

theorem complex_magnitude (z : ℂ) (h : z * Complex.I = 2 - Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l1483_148304


namespace cosine_in_acute_triangle_l1483_148367

theorem cosine_in_acute_triangle (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  (1/2) * a * b * Real.sin C = 5 →
  a = 3 →
  b = 4 →
  Real.cos C = Real.sqrt 11 / 6 := by
sorry

end cosine_in_acute_triangle_l1483_148367


namespace cloth_sale_commission_calculation_l1483_148307

/-- Calculates the worth of cloth sold given the commission rate and commission amount. -/
def worth_of_cloth_sold (commission_rate : ℚ) (commission : ℚ) : ℚ :=
  commission * (100 / commission_rate)

/-- Theorem stating that for a 4% commission rate and Rs. 12.50 commission, 
    the worth of cloth sold is Rs. 312.50 -/
theorem cloth_sale_commission_calculation :
  worth_of_cloth_sold (4 : ℚ) (25/2 : ℚ) = (625/2 : ℚ) := by
  sorry

#eval worth_of_cloth_sold (4 : ℚ) (25/2 : ℚ)

end cloth_sale_commission_calculation_l1483_148307


namespace all_sections_clearance_l1483_148387

/-- Represents the percentage of candidates who cleared a specific number of sections -/
structure SectionClearance where
  zero : ℝ
  one : ℝ
  two : ℝ
  three : ℝ
  four : ℝ
  five : ℝ

/-- Theorem stating the percentage of candidates who cleared all 5 sections -/
theorem all_sections_clearance 
  (total_candidates : ℕ) 
  (three_section_candidates : ℕ) 
  (clearance : SectionClearance) :
  total_candidates = 1200 →
  three_section_candidates = 300 →
  clearance.zero = 5 →
  clearance.one = 25 →
  clearance.two = 24.5 →
  clearance.four = 20 →
  clearance.three = (three_section_candidates : ℝ) / (total_candidates : ℝ) * 100 →
  clearance.five = 0.5 :=
by sorry

end all_sections_clearance_l1483_148387


namespace employee_payment_percentage_l1483_148346

theorem employee_payment_percentage (total_payment y_payment : ℚ) 
  (h1 : total_payment = 616)
  (h2 : y_payment = 280) : 
  (total_payment - y_payment) / y_payment * 100 = 120 := by
  sorry

end employee_payment_percentage_l1483_148346


namespace sum_of_squares_of_roots_l1483_148378

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 17*x + 8 = 0 → ∃ s₁ s₂ : ℝ, s₁ + s₂ = 17 ∧ s₁ * s₂ = 8 ∧ s₁^2 + s₂^2 = 273 := by
  sorry

end sum_of_squares_of_roots_l1483_148378


namespace blue_balls_count_l1483_148311

theorem blue_balls_count (B : ℕ) : 
  (6 : ℚ) * 5 / ((8 + B) * (7 + B)) = 0.19230769230769232 → B = 5 := by
  sorry

end blue_balls_count_l1483_148311


namespace trig_expression_equality_l1483_148360

theorem trig_expression_equality : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end trig_expression_equality_l1483_148360


namespace prob_two_red_from_bag_l1483_148370

/-- The probability of picking two red balls from a bag -/
def probability_two_red_balls (red blue green : ℕ) : ℚ :=
  let total := red + blue + green
  (red : ℚ) / total * ((red - 1) : ℚ) / (total - 1)

/-- Theorem: The probability of picking two red balls from a bag with 3 red, 2 blue, and 4 green balls is 1/12 -/
theorem prob_two_red_from_bag : probability_two_red_balls 3 2 4 = 1 / 12 := by
  sorry

end prob_two_red_from_bag_l1483_148370


namespace sports_club_overlap_l1483_148310

theorem sports_club_overlap (total : ℕ) (badminton tennis neither : ℕ) 
  (h_total : total = 30)
  (h_badminton : badminton = 17)
  (h_tennis : tennis = 17)
  (h_neither : neither = 2)
  (h_sum : total = badminton + tennis - (total - neither)) :
  badminton + tennis - (total - neither) = 6 := by
  sorry

end sports_club_overlap_l1483_148310


namespace amanda_remaining_money_l1483_148362

/-- Calculates the remaining money after purchases -/
def remaining_money (initial_amount : ℕ) (item1_cost : ℕ) (item1_quantity : ℕ) (item2_cost : ℕ) : ℕ :=
  initial_amount - (item1_cost * item1_quantity + item2_cost)

/-- Proves that given the specific amounts in the problem, the remaining money is 7 -/
theorem amanda_remaining_money :
  remaining_money 50 9 2 25 = 7 := by
  sorry

end amanda_remaining_money_l1483_148362


namespace sin_210_plus_cos_60_equals_zero_l1483_148301

theorem sin_210_plus_cos_60_equals_zero :
  Real.sin (210 * π / 180) + Real.cos (60 * π / 180) = 0 := by
  sorry

end sin_210_plus_cos_60_equals_zero_l1483_148301


namespace systematic_sample_theorem_l1483_148322

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  first_element : ℕ
  h_population_size_pos : 0 < population_size
  h_sample_size_pos : 0 < sample_size
  h_sample_size_le_population : sample_size ≤ population_size
  h_first_element_in_range : first_element ≤ population_size

/-- Check if a number is in the systematic sample -/
def SystematicSample.contains (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.first_element + k * (s.population_size / s.sample_size) ∧ n ≤ s.population_size

/-- The main theorem to be proved -/
theorem systematic_sample_theorem (s : SystematicSample)
  (h_pop_size : s.population_size = 60)
  (h_sample_size : s.sample_size = 4)
  (h_contains_3 : s.contains 3)
  (h_contains_33 : s.contains 33)
  (h_contains_48 : s.contains 48) :
  s.contains 18 := by
  sorry


end systematic_sample_theorem_l1483_148322


namespace price_difference_is_80_cents_l1483_148317

/-- Represents the price calculation methods in Lintonville Fashion Store --/
def price_calculation (original_price discount_rate tax_rate coupon : ℝ) : ℝ × ℝ := 
  let bob_total := (original_price * (1 + tax_rate) * (1 - discount_rate)) - coupon
  let alice_total := (original_price * (1 - discount_rate) - coupon) * (1 + tax_rate)
  (bob_total, alice_total)

/-- The difference between Bob's and Alice's calculations is $0.80 --/
theorem price_difference_is_80_cents 
  (h_original_price : ℝ) 
  (h_discount_rate : ℝ) 
  (h_tax_rate : ℝ) 
  (h_coupon : ℝ) 
  (h_op : h_original_price = 120)
  (h_dr : h_discount_rate = 0.15)
  (h_tr : h_tax_rate = 0.08)
  (h_c : h_coupon = 10) : 
  let (bob_total, alice_total) := price_calculation h_original_price h_discount_rate h_tax_rate h_coupon
  bob_total - alice_total = 0.80 := by
  sorry

end price_difference_is_80_cents_l1483_148317


namespace quadratic_inequality_solution_set_l1483_148331

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 < x + 6} = {x : ℝ | -2 < x ∧ x < 3} := by sorry

end quadratic_inequality_solution_set_l1483_148331


namespace largest_three_digit_sum_l1483_148315

-- Define the sum function
def sum (A B : Nat) : Nat :=
  (100 * A + 10 * A + B) + (10 * B + A) + B

-- Theorem statement
theorem largest_three_digit_sum :
  ∃ (A B : Nat),
    A ≠ B ∧
    A < 10 ∧
    B < 10 ∧
    sum A B ≤ 999 ∧
    ∀ (X Y : Nat),
      X ≠ Y →
      X < 10 →
      Y < 10 →
      sum X Y ≤ 999 →
      sum X Y ≤ sum A B :=
by
  sorry

end largest_three_digit_sum_l1483_148315


namespace weight_loss_difference_equals_303_l1483_148371

/-- Calculates the total weight loss difference between Luca and Kim combined, and Barbi -/
def weight_loss_difference : ℝ :=
  let barbi_monthly_loss : ℝ := 1.5
  let barbi_months : ℕ := 2 * 12
  let luca_yearly_loss : ℝ := 9
  let luca_years : ℕ := 15
  let kim_first_year_monthly_loss : ℝ := 2
  let kim_remaining_monthly_loss : ℝ := 3
  let kim_remaining_months : ℕ := 5 * 12

  let barbi_total_loss := barbi_monthly_loss * barbi_months
  let luca_total_loss := luca_yearly_loss * luca_years
  let kim_first_year_loss := kim_first_year_monthly_loss * 12
  let kim_remaining_loss := kim_remaining_monthly_loss * kim_remaining_months
  let kim_total_loss := kim_first_year_loss + kim_remaining_loss

  (luca_total_loss + kim_total_loss) - barbi_total_loss

theorem weight_loss_difference_equals_303 : weight_loss_difference = 303 := by
  sorry

end weight_loss_difference_equals_303_l1483_148371


namespace opposite_sign_implications_l1483_148339

theorem opposite_sign_implications (a b : ℝ) 
  (h1 : |2*a + b| * Real.sqrt (3*b + 12) ≤ 0) 
  (h2 : |2*a + b| + Real.sqrt (3*b + 12) > 0) : 
  (Real.sqrt (2*a - 3*b) = 4 ∨ Real.sqrt (2*a - 3*b) = -4) ∧ 
  (∀ x : ℝ, a*x^2 + 4*b - 2 = 0 ↔ x = 3 ∨ x = -3) :=
by sorry

end opposite_sign_implications_l1483_148339


namespace rectangle_to_square_l1483_148369

theorem rectangle_to_square (l w : ℝ) : 
  (2 * (l + w) = 40) →  -- Perimeter of rectangle is 40cm
  (l - 8 = w + 2) →     -- Rectangle becomes square after changes
  (l - 8 = 7) :=        -- Side length of resulting square is 7cm
by
  sorry

end rectangle_to_square_l1483_148369


namespace same_color_probability_l1483_148375

def total_balls : ℕ := 13 + 7
def green_balls : ℕ := 13
def red_balls : ℕ := 7

theorem same_color_probability :
  (green_balls / total_balls) ^ 3 + (red_balls / total_balls) ^ 3 = 127 / 400 := by
  sorry

end same_color_probability_l1483_148375


namespace employee_discount_percentage_l1483_148379

theorem employee_discount_percentage
  (wholesale_cost : ℝ)
  (markup_percentage : ℝ)
  (employee_paid_price : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : markup_percentage = 20)
  (h3 : employee_paid_price = 180) :
  let retail_price := wholesale_cost * (1 + markup_percentage / 100)
  let discount_amount := retail_price - employee_paid_price
  let discount_percentage := (discount_amount / retail_price) * 100
  discount_percentage = 25 := by sorry

end employee_discount_percentage_l1483_148379


namespace merchant_profit_percentage_l1483_148321

-- Define the markup and discount percentages
def markup : ℝ := 0.40
def discount : ℝ := 0.15

-- Theorem statement
theorem merchant_profit_percentage :
  let marked_price := 1 + markup
  let selling_price := marked_price * (1 - discount)
  (selling_price - 1) * 100 = 19 := by
  sorry

end merchant_profit_percentage_l1483_148321


namespace lottery_comparison_l1483_148352

-- Define the number of red and white balls
def red_balls : ℕ := 4
def white_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := red_balls + white_balls

-- Define the probability of drawing two red balls
def prob_two_red : ℚ := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))

-- Define the probability of rolling at least one four with two dice
def prob_at_least_one_four : ℚ := 1 - (5/6) * (5/6)

-- Theorem to prove
theorem lottery_comparison : prob_two_red > prob_at_least_one_four := by
  sorry


end lottery_comparison_l1483_148352


namespace norm_scalar_multiple_l1483_148347

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem norm_scalar_multiple (v : E) (h : ‖v‖ = 7) : ‖(5 : ℝ) • v‖ = 35 := by
  sorry

end norm_scalar_multiple_l1483_148347


namespace cubic_three_distinct_roots_in_interval_l1483_148354

theorem cubic_three_distinct_roots_in_interval 
  (p q : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    (-2 < x₁ ∧ x₁ < 4) ∧ (-2 < x₂ ∧ x₂ < 4) ∧ (-2 < x₃ ∧ x₃ < 4) ∧
    x₁^3 + p*x₁ + q = 0 ∧ x₂^3 + p*x₂ + q = 0 ∧ x₃^3 + p*x₃ + q = 0) ↔ 
  (4*p^3 + 27*q^2 < 0 ∧ 2*p + 8 < q ∧ q < -4*p - 64) :=
sorry

end cubic_three_distinct_roots_in_interval_l1483_148354


namespace jerrys_pool_length_l1483_148372

/-- Represents the problem of calculating Jerry's pool length --/
theorem jerrys_pool_length :
  ∀ (total_water drinking_cooking_water shower_water num_showers pool_width pool_height : ℝ),
    total_water = 1000 →
    drinking_cooking_water = 100 →
    shower_water = 20 →
    num_showers = 15 →
    pool_width = 10 →
    pool_height = 6 →
    ∃ (pool_length : ℝ),
      pool_length = 10 ∧
      pool_length * pool_width * pool_height = 
        total_water - (drinking_cooking_water + shower_water * num_showers) :=
by sorry

end jerrys_pool_length_l1483_148372


namespace least_three_digit_12_heavy_is_105_12_heavy_is_105_three_digit_least_three_digit_12_heavy_is_105_l1483_148389

/-- A number is 12-heavy if its remainder when divided by 12 is greater than 8. -/
def is_12_heavy (n : ℕ) : Prop := n % 12 > 8

/-- The set of three-digit natural numbers. -/
def three_digit_numbers : Set ℕ := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

theorem least_three_digit_12_heavy : 
  ∀ n ∈ three_digit_numbers, is_12_heavy n → n ≥ 105 :=
by sorry

theorem is_105_12_heavy : is_12_heavy 105 :=
by sorry

theorem is_105_three_digit : 105 ∈ three_digit_numbers :=
by sorry

/-- 105 is the least three-digit 12-heavy whole number. -/
theorem least_three_digit_12_heavy_is_105 : 
  ∃ n ∈ three_digit_numbers, is_12_heavy n ∧ ∀ m ∈ three_digit_numbers, is_12_heavy m → n ≤ m :=
by sorry

end least_three_digit_12_heavy_is_105_12_heavy_is_105_three_digit_least_three_digit_12_heavy_is_105_l1483_148389


namespace everton_calculator_count_l1483_148392

/-- Represents the order of calculators by Everton college -/
structure CalculatorOrder where
  totalCost : ℕ
  scientificCost : ℕ
  graphingCost : ℕ
  scientificCount : ℕ

/-- Calculates the total number of calculators in an order -/
def totalCalculators (order : CalculatorOrder) : ℕ :=
  let graphingCount := (order.totalCost - order.scientificCount * order.scientificCost) / order.graphingCost
  order.scientificCount + graphingCount

/-- Theorem: The total number of calculators in Everton college's order is 45 -/
theorem everton_calculator_count :
  let order : CalculatorOrder := {
    totalCost := 1625,
    scientificCost := 10,
    graphingCost := 57,
    scientificCount := 20
  }
  totalCalculators order = 45 := by
  sorry

end everton_calculator_count_l1483_148392


namespace a_equals_one_sufficient_not_necessary_l1483_148340

-- Define the quadratic equation
def quadratic_equation (a : ℝ) (x : ℝ) : ℝ := x^2 - 3*x + a

-- Define the discriminant
def discriminant (a : ℝ) : ℝ := 9 - 4*a

-- Theorem statement
theorem a_equals_one_sufficient_not_necessary :
  (∃ x : ℝ, quadratic_equation 1 x = 0) ∧
  (∃ a : ℝ, a ≠ 1 ∧ ∃ x : ℝ, quadratic_equation a x = 0) :=
by sorry

end a_equals_one_sufficient_not_necessary_l1483_148340


namespace different_rhetorical_device_l1483_148382

-- Define the rhetorical devices
inductive RhetoricalDevice
| Metaphor
| Personification

-- Define a function to assign rhetorical devices to options
def assignRhetoricalDevice (option : Char) : RhetoricalDevice :=
  match option with
  | 'A' => RhetoricalDevice.Metaphor
  | _ => RhetoricalDevice.Personification

-- Theorem statement
theorem different_rhetorical_device :
  ∀ (x : Char), x ≠ 'A' →
  assignRhetoricalDevice 'A' ≠ assignRhetoricalDevice x :=
by
  sorry

#check different_rhetorical_device

end different_rhetorical_device_l1483_148382


namespace quadratic_inequality_range_l1483_148305

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0) ↔ -1 < m ∧ m ≤ 0 :=
by sorry

end quadratic_inequality_range_l1483_148305


namespace largest_less_than_point_seven_l1483_148355

theorem largest_less_than_point_seven : 
  let numbers : List ℝ := [0.8, 1/2, 0.9]
  let target : ℝ := 0.7
  (∀ x ∈ numbers, x ≤ target → x ≤ (1/2 : ℝ)) ∧ 
  ((1/2 : ℝ) ∈ numbers) ∧ 
  ((1/2 : ℝ) < target) := by
  sorry

end largest_less_than_point_seven_l1483_148355


namespace odd_function_extension_l1483_148348

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem odd_function_extension :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  (∀ x > 0, f x = x^2 - 2*x) →  -- f(x) = x^2 - 2x for x > 0
  (∀ x < 0, f x = -x^2 - 2*x) :=  -- f(x) = -x^2 - 2x for x < 0
by sorry

end odd_function_extension_l1483_148348


namespace expression_simplification_l1483_148363

theorem expression_simplification (x : ℝ) : 
  2*x - 3*(2+x) + 4*(2-x) - 5*(2+3*x) = -20*x - 8 := by
sorry

end expression_simplification_l1483_148363


namespace complementary_angles_difference_l1483_148356

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a = 3 * b →   -- ratio of angles is 3:1
  |a - b| = 45  -- positive difference is 45°
:= by sorry

end complementary_angles_difference_l1483_148356


namespace triangle_area_with_perimeter_12_l1483_148391

/-- A triangle with integral sides and perimeter 12 has an area of 6 -/
theorem triangle_area_with_perimeter_12 :
  ∀ a b c : ℕ,
  a + b + c = 12 →
  a + b > c →
  b + c > a →
  c + a > b →
  (a * b : ℝ) / 2 = 6 :=
by
  sorry

end triangle_area_with_perimeter_12_l1483_148391


namespace binary_decimal_base5_conversion_l1483_148388

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert decimal to base 5
def decimal_to_base5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

-- Theorem statement
theorem binary_decimal_base5_conversion :
  let binary : List Bool := [true, true, false, false, true, true]
  let decimal : Nat := 51
  let base5 : List Nat := [2, 0, 1]
  binary_to_decimal binary = decimal ∧ decimal_to_base5 decimal = base5 := by
  sorry


end binary_decimal_base5_conversion_l1483_148388


namespace solution_equation1_no_solution_equation2_l1483_148308

-- Define the equations
def equation1 (x : ℝ) : Prop := (1 / (x - 1) = 5 / (2 * x + 1))
def equation2 (x : ℝ) : Prop := ((x + 1) / (x - 1) - 4 / (x^2 - 1) = 1)

-- Theorem for equation 1
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 2 := by sorry

-- Theorem for equation 2
theorem no_solution_equation2 : ¬ ∃ x : ℝ, equation2 x := by sorry

end solution_equation1_no_solution_equation2_l1483_148308


namespace polynomial_division_remainder_l1483_148365

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 - 2*x^2 + 3 = (x^2 - 4*x + 7) * q + (28*x - 46) :=
by
  sorry

end polynomial_division_remainder_l1483_148365


namespace calculate_expression_l1483_148345

theorem calculate_expression : |-7| + Real.sqrt 16 - (-3)^2 = 2 := by
  sorry

end calculate_expression_l1483_148345


namespace square_area_from_diagonal_l1483_148300

theorem square_area_from_diagonal (d : ℝ) (h : d = 10) : 
  (d^2 / 2 : ℝ) = 50 := by sorry

end square_area_from_diagonal_l1483_148300


namespace roses_in_vase_l1483_148368

theorem roses_in_vase (initial_roses : ℕ) (initial_orchids : ℕ) (final_orchids : ℕ) (cut_orchids : ℕ) :
  initial_roses = 16 →
  initial_orchids = 3 →
  final_orchids = 7 →
  cut_orchids = 4 →
  ∃ (cut_roses : ℕ), cut_roses = cut_orchids →
  initial_roses + cut_roses = 24 :=
by sorry

end roses_in_vase_l1483_148368


namespace count_solution_pairs_l1483_148337

/-- The number of distinct ordered pairs of positive integers (x,y) satisfying x^4y^2 - 10x^2y + 9 = 0 -/
def solution_count : ℕ := 3

/-- A predicate that checks if a pair of positive integers satisfies the equation -/
def satisfies_equation (x y : ℕ+) : Prop :=
  (x.val ^ 4) * (y.val ^ 2) - 10 * (x.val ^ 2) * y.val + 9 = 0

theorem count_solution_pairs :
  (∃! (s : Finset (ℕ+ × ℕ+)), 
    (∀ p ∈ s, satisfies_equation p.1 p.2) ∧ 
    s.card = solution_count) := by sorry

end count_solution_pairs_l1483_148337


namespace minimum_employees_science_bureau_hiring_l1483_148393

theorem minimum_employees (water : ℕ) (air : ℕ) (both : ℕ) : ℕ :=
  let total := water + air - both
  total

theorem science_bureau_hiring : 
  minimum_employees 98 89 34 = 153 := by sorry

end minimum_employees_science_bureau_hiring_l1483_148393


namespace cassidy_poster_collection_l1483_148361

theorem cassidy_poster_collection (current_posters : ℕ) : current_posters = 22 :=
  by
  have two_years_ago : ℕ := 14
  have after_summer : ℕ := current_posters + 6
  have double_two_years_ago : after_summer = 2 * two_years_ago := by sorry
  sorry

end cassidy_poster_collection_l1483_148361


namespace drug_price_reduction_equation_l1483_148373

/-- Represents the price reduction scenario for a drug -/
def price_reduction (initial_price final_price : ℝ) (num_reductions : ℕ) (reduction_percentage : ℝ) : Prop :=
  initial_price * (1 - reduction_percentage) ^ num_reductions = final_price

/-- Theorem stating the equation for the drug price reduction scenario -/
theorem drug_price_reduction_equation :
  ∃ (x : ℝ), price_reduction 144 81 2 x :=
sorry

end drug_price_reduction_equation_l1483_148373


namespace find_b_l1483_148320

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - b else 2^x

theorem find_b : ∃ b : ℝ, f b (f b (5/6)) = 4 ∧ b = 11/8 := by
  sorry

end find_b_l1483_148320


namespace equal_solution_implies_k_value_l1483_148385

theorem equal_solution_implies_k_value :
  ∀ (k : ℚ), 
  (∃ (x : ℚ), 3 * x - 6 = 0 ∧ 2 * x - 5 * k = 11) →
  (∀ (x : ℚ), 3 * x - 6 = 0 ↔ 2 * x - 5 * k = 11) →
  k = -7/5 := by
sorry

end equal_solution_implies_k_value_l1483_148385


namespace area_of_region_l1483_148330

-- Define the curve and line
def curve (x : ℝ) : ℝ → Prop := λ y ↦ y^2 = 2*x
def line (x : ℝ) : ℝ → Prop := λ y ↦ y = x - 4

-- Define the region
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ curve x y ∧ line x y}

-- State the theorem
theorem area_of_region : MeasureTheory.volume region = 18 := by
  sorry

end area_of_region_l1483_148330


namespace expression_simplification_l1483_148325

theorem expression_simplification (x y : ℝ) (h : y ≠ 0) :
  ((x^2 + y^2) - (x - y)^2 + 2*y*(x - y)) / (4*y) = x - (1/2)*y :=
by sorry

end expression_simplification_l1483_148325


namespace equation_solutions_l1483_148329

theorem equation_solutions : 
  (∃ (x₁ x₂ : ℝ), x₁ = 5 + Real.sqrt 35 ∧ x₂ = 5 - Real.sqrt 35 ∧ 
    x₁^2 - 10*x₁ - 10 = 0 ∧ x₂^2 - 10*x₂ - 10 = 0) ∧
  (∃ (y₁ y₂ : ℝ), y₁ = 5 ∧ y₂ = 13/3 ∧ 
    3*(y₁ - 5)^2 = 2*(5 - y₁) ∧ 3*(y₂ - 5)^2 = 2*(5 - y₂)) :=
by sorry

end equation_solutions_l1483_148329


namespace divisible_by_64_l1483_148398

theorem divisible_by_64 (n : ℕ) (h : n > 0) : ∃ k : ℤ, 3^(2*n+2) - 8*n - 9 = 64*k := by
  sorry

end divisible_by_64_l1483_148398


namespace partnership_profit_calculation_l1483_148350

/-- Represents the partnership profit calculation problem --/
theorem partnership_profit_calculation
  (p q r : ℕ) -- Initial capitals
  (h_ratio : p / q = 3 / 2 ∧ q / r = 4 / 3) -- Initial capital ratio
  (h_p_withdraw : ℕ) -- Amount p withdraws after 2 months
  (h_q_share : ℕ) -- q's share of profit in rupees
  (h_duration : ℕ) -- Total duration of partnership in months
  (h_p_withdraw_time : ℕ) -- Time after which p withdraws half capital
  (h_p_withdraw_half : h_p_withdraw = p / 2) -- p withdraws half of initial capital
  (h_duration_val : h_duration = 12) -- Total duration is 12 months
  (h_p_withdraw_time_val : h_p_withdraw_time = 2) -- p withdraws after 2 months
  (h_q_share_val : h_q_share = 144) -- q's share is Rs 144
  : ∃ (total_profit : ℕ), total_profit = 486 := by
  sorry

end partnership_profit_calculation_l1483_148350


namespace periodic_last_digit_triangular_perfect_square_between_sums_l1483_148374

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def last_digit (n : ℕ) : ℕ := n % 10

def sum_triangular_numbers (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

theorem periodic_last_digit_triangular :
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, last_digit (triangular_number n) = last_digit (triangular_number (n + k)) :=
sorry

theorem perfect_square_between_sums (n : ℕ) (h : n ≥ 3) :
  ∃ k : ℕ, sum_triangular_numbers (n - 1) < k * k ∧ k * k < sum_triangular_numbers n :=
sorry

end periodic_last_digit_triangular_perfect_square_between_sums_l1483_148374


namespace probability_collinear_dots_l1483_148314

/-- Represents a rectangular array of dots -/
structure DotArray where
  rows : ℕ
  cols : ℕ
  total_dots : ℕ

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the number of collinear sets of 4 dots in a vertical line -/
def vertical_collinear_sets (arr : DotArray) : ℕ := 
  arr.cols * choose arr.rows 4

/-- Main theorem: Probability of choosing 4 collinear dots -/
theorem probability_collinear_dots (arr : DotArray) 
  (h1 : arr.rows = 5) 
  (h2 : arr.cols = 4) 
  (h3 : arr.total_dots = 20) : 
  (vertical_collinear_sets arr : ℚ) / (choose arr.total_dots 4) = 4 / 969 := by
  sorry

end probability_collinear_dots_l1483_148314


namespace square_remainder_mod_nine_l1483_148394

theorem square_remainder_mod_nine (n : ℤ) : 
  (n % 9 = 1 ∨ n % 9 = 8) → (n^2) % 9 = 1 := by
  sorry

end square_remainder_mod_nine_l1483_148394


namespace weight_system_l1483_148344

/-- Represents the weight of birds in jin -/
structure BirdWeight where
  sparrow : ℝ
  swallow : ℝ

/-- The conditions of the sparrow and swallow weight problem -/
def weightProblem (w : BirdWeight) : Prop :=
  (5 * w.sparrow + 6 * w.swallow = 1) ∧
  (w.sparrow > w.swallow) ∧
  (4 * w.sparrow + 7 * w.swallow = 5 * w.sparrow + 6 * w.swallow)

/-- The system of equations representing the sparrow and swallow weight problem -/
theorem weight_system (w : BirdWeight) (h : weightProblem w) :
  (5 * w.sparrow + 6 * w.swallow = 1) ∧
  (4 * w.sparrow + 7 * w.swallow = 5 * w.sparrow + 6 * w.swallow) :=
by sorry

end weight_system_l1483_148344


namespace special_number_satisfies_conditions_special_number_unique_l1483_148397

/-- A two-digit number that satisfies the given conditions -/
def special_number : ℕ := 50

/-- The property that defines our special number -/
def is_special_number (a : ℕ) : Prop :=
  (a ≥ 10 ∧ a ≤ 99) ∧  -- Two-digit number
  (∃ (q r : ℚ), 
    (101 * a - a^2) / (0.04 * a^2) = q + r ∧
    q = a / 2 ∧
    r = a / (0.04 * a^2))

theorem special_number_satisfies_conditions : 
  is_special_number special_number :=
sorry

theorem special_number_unique : 
  ∀ (n : ℕ), is_special_number n → n = special_number :=
sorry

end special_number_satisfies_conditions_special_number_unique_l1483_148397


namespace father_twice_son_age_l1483_148309

/-- Proves that the number of years after which a father aged 42 will be twice as old as his son aged 14 is 14 years. -/
theorem father_twice_son_age (father_age son_age : ℕ) (h1 : father_age = 42) (h2 : son_age = 14) :
  ∃ x : ℕ, father_age + x = 2 * (son_age + x) ∧ x = 14 :=
by sorry

end father_twice_son_age_l1483_148309


namespace wendy_unrecycled_bags_l1483_148396

/-- Proves that Wendy did not recycle 2 bags given the problem conditions --/
theorem wendy_unrecycled_bags :
  ∀ (total_bags : ℕ) (points_per_bag : ℕ) (total_possible_points : ℕ),
    total_bags = 11 →
    points_per_bag = 5 →
    total_possible_points = 45 →
    total_bags - (total_possible_points / points_per_bag) = 2 :=
by
  sorry

end wendy_unrecycled_bags_l1483_148396


namespace f_2019_is_zero_l1483_148343

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def satisfies_equation (f : ℝ → ℝ) : Prop := ∀ x, f (x + 6) - f x = 2 * f 3

theorem f_2019_is_zero (f : ℝ → ℝ) (h1 : is_even f) (h2 : satisfies_equation f) : f 2019 = 0 := by
  sorry

end f_2019_is_zero_l1483_148343


namespace selina_shirt_cost_l1483_148302

/-- Represents the price and quantity of an item of clothing --/
structure ClothingItem where
  price : ℕ
  quantity : ℕ

/-- Calculates the total money Selina got from selling her clothes --/
def totalSalesMoney (pants shorts shirts : ClothingItem) : ℕ :=
  pants.price * pants.quantity + shorts.price * shorts.quantity + shirts.price * shirts.quantity

/-- Represents the problem of finding the cost of each shirt Selina bought --/
theorem selina_shirt_cost (pants shorts shirts : ClothingItem)
  (bought_shirts : ℕ) (money_left : ℕ)
  (h_pants : pants = ⟨5, 3⟩)
  (h_shorts : shorts = ⟨3, 5⟩)
  (h_shirts : shirts = ⟨4, 5⟩)
  (h_bought_shirts : bought_shirts = 2)
  (h_money_left : money_left = 30) :
  (totalSalesMoney pants shorts shirts - money_left) / bought_shirts = 10 := by
  sorry

#check selina_shirt_cost

end selina_shirt_cost_l1483_148302


namespace factorization_equality_l1483_148316

theorem factorization_equality (a : ℝ) : 2 * a^2 - 4 * a + 2 = 2 * (a - 1)^2 := by
  sorry

end factorization_equality_l1483_148316


namespace function_inequality_l1483_148364

theorem function_inequality (f : ℝ → ℝ) (h1 : ∀ x, f x ≥ |x|) (h2 : ∀ x, f x ≥ 2^x) :
  ∀ a b : ℝ, f a ≤ 2^b → a ≤ b := by
  sorry

end function_inequality_l1483_148364


namespace garden_walkway_area_l1483_148383

theorem garden_walkway_area :
  let flower_bed_width : ℕ := 4
  let flower_bed_height : ℕ := 3
  let flower_bed_rows : ℕ := 4
  let flower_bed_columns : ℕ := 3
  let walkway_width : ℕ := 2
  let pond_width : ℕ := 3
  let pond_height : ℕ := 2

  let total_width : ℕ := flower_bed_width * flower_bed_columns + walkway_width * (flower_bed_columns + 1)
  let total_height : ℕ := flower_bed_height * flower_bed_rows + walkway_width * (flower_bed_rows + 1)
  let total_area : ℕ := total_width * total_height

  let pond_area : ℕ := pond_width * pond_height
  let adjusted_area : ℕ := total_area - pond_area

  let flower_bed_area : ℕ := flower_bed_width * flower_bed_height
  let total_flower_bed_area : ℕ := flower_bed_area * flower_bed_rows * flower_bed_columns

  let walkway_area : ℕ := adjusted_area - total_flower_bed_area

  walkway_area = 290 := by sorry

end garden_walkway_area_l1483_148383


namespace slope_angle_MN_l1483_148377

/-- Given points M(1, 2) and N(0, 1), the slope angle of line MN is π/4. -/
theorem slope_angle_MN : 
  let M : ℝ × ℝ := (1, 2)
  let N : ℝ × ℝ := (0, 1)
  let slope : ℝ := (M.2 - N.2) / (M.1 - N.1)
  let slope_angle : ℝ := Real.arctan slope
  slope_angle = π / 4 := by
  sorry

end slope_angle_MN_l1483_148377


namespace diet_soda_count_l1483_148326

/-- Represents the number of apples in the grocery store -/
def num_apples : ℕ := 36

/-- Represents the number of regular soda bottles in the grocery store -/
def num_regular_soda : ℕ := 80

/-- Represents the number of diet soda bottles in the grocery store -/
def num_diet_soda : ℕ := 54

/-- The total number of bottles is 98 more than the number of apples -/
axiom total_bottles_relation : num_regular_soda + num_diet_soda = num_apples + 98

theorem diet_soda_count : num_diet_soda = 54 := by sorry

end diet_soda_count_l1483_148326


namespace diamond_value_l1483_148312

theorem diamond_value (diamond : ℕ) (h1 : diamond < 10) 
  (h2 : diamond * 9 + 5 = diamond * 10 + 2) : diamond = 3 := by
  sorry

end diamond_value_l1483_148312


namespace pie_eating_difference_l1483_148335

theorem pie_eating_difference :
  let first_participant : ℚ := 5/6
  let second_participant : ℚ := 2/3
  first_participant - second_participant = 1/6 := by
sorry

end pie_eating_difference_l1483_148335


namespace max_pigs_buyable_l1483_148399

def budget : ℕ := 1300
def pig_cost : ℕ := 21
def duck_cost : ℕ := 23
def min_ducks : ℕ := 20

theorem max_pigs_buyable :
  ∀ p d : ℕ,
    p > 0 →
    d ≥ min_ducks →
    pig_cost * p + duck_cost * d ≤ budget →
    p ≤ 40 ∧
    ∃ p' d' : ℕ, p' = 40 ∧ d' ≥ min_ducks ∧ pig_cost * p' + duck_cost * d' = budget :=
by sorry

end max_pigs_buyable_l1483_148399


namespace gcd_from_lcm_and_ratio_l1483_148342

theorem gcd_from_lcm_and_ratio (C D : ℕ+) : 
  C.lcm D = 180 → C.val * 6 = D.val * 5 → C.gcd D = 6 := by
  sorry

end gcd_from_lcm_and_ratio_l1483_148342


namespace policy_effect_l1483_148319

-- Define the labor market for teachers
structure TeacherMarket where
  supply : ℝ → ℝ  -- Supply function
  demand : ℝ → ℝ  -- Demand function
  equilibrium_wage : ℝ  -- Equilibrium wage

-- Define the commercial education market
structure CommercialEducationMarket where
  supply : ℝ → ℝ  -- Supply function
  demand : ℝ → ℝ  -- Demand function
  equilibrium_price : ℝ  -- Equilibrium price

-- Define the government policy
def government_policy (min_years : ℕ) (locality : String) : Prop :=
  ∃ (requirement : Prop), requirement

-- Theorem statement
theorem policy_effect 
  (teacher_market : TeacherMarket)
  (commercial_market : CommercialEducationMarket)
  (min_years : ℕ)
  (locality : String) :
  government_policy min_years locality →
  ∃ (new_teacher_market : TeacherMarket)
    (new_commercial_market : CommercialEducationMarket),
    new_teacher_market.equilibrium_wage > teacher_market.equilibrium_wage ∧
    new_commercial_market.equilibrium_price < commercial_market.equilibrium_price :=
by
  sorry

end policy_effect_l1483_148319
