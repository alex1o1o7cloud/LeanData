import Mathlib

namespace sector_angle_measure_l2400_240098

/-- Given a sector with radius 2 cm and area 4 cm², 
    the radian measure of its central angle is 2. -/
theorem sector_angle_measure (r : ℝ) (S : ℝ) (θ : ℝ) : 
  r = 2 →  -- radius is 2 cm
  S = 4 →  -- area is 4 cm²
  S = 1/2 * r^2 * θ →  -- formula for sector area
  θ = 2 :=  -- central angle is 2 radians
by sorry

end sector_angle_measure_l2400_240098


namespace congruence_solution_l2400_240029

theorem congruence_solution (n : ℤ) : 
  -20 ≤ n ∧ n ≤ 20 ∧ n ≡ -127 [ZMOD 7] → n = -13 ∨ n = 1 ∨ n = 15 := by
  sorry

end congruence_solution_l2400_240029


namespace josh_siblings_count_josh_problem_l2400_240011

theorem josh_siblings_count (initial_candies : ℕ) 
  (candies_per_sibling : ℕ) (eat_himself : ℕ) (remaining_candies : ℕ) : ℕ :=
  let siblings_count := (initial_candies - 2 * (remaining_candies + eat_himself)) / (2 * candies_per_sibling)
  siblings_count

theorem josh_problem :
  josh_siblings_count 100 10 16 19 = 3 := by
  sorry

end josh_siblings_count_josh_problem_l2400_240011


namespace emerie_nickels_l2400_240005

/-- The number of coin types -/
def num_coin_types : ℕ := 3

/-- The number of coins Zain has -/
def zain_coins : ℕ := 48

/-- The number of quarters Emerie has -/
def emerie_quarters : ℕ := 6

/-- The number of dimes Emerie has -/
def emerie_dimes : ℕ := 7

/-- The number of extra coins Zain has for each type -/
def extra_coins_per_type : ℕ := 10

theorem emerie_nickels : 
  (zain_coins - num_coin_types * extra_coins_per_type) - (emerie_quarters + emerie_dimes) = 5 := by
  sorry

end emerie_nickels_l2400_240005


namespace sum_abc_equals_42_l2400_240004

theorem sum_abc_equals_42 
  (a b c : ℕ+) 
  (h1 : a * b + c = 41)
  (h2 : b * c + a = 41)
  (h3 : a * c + b = 41) : 
  a + b + c = 42 := by
  sorry

end sum_abc_equals_42_l2400_240004


namespace intersection_of_P_and_Q_l2400_240037

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {0, 3, 4, 5}

theorem intersection_of_P_and_Q : P ∩ Q = {3, 4} := by
  sorry

end intersection_of_P_and_Q_l2400_240037


namespace solution_set_iff_a_half_l2400_240009

theorem solution_set_iff_a_half (a : ℝ) :
  (∀ x : ℝ, (a * x) / (x - 1) < 1 ↔ x < 1 ∨ x > 2) ↔ a = 1/2 :=
by sorry

end solution_set_iff_a_half_l2400_240009


namespace hyperbola_properties_l2400_240064

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the length of the real axis
def real_axis_length : ℝ := 4

-- Define the distance from foci to asymptote
def foci_to_asymptote_distance : ℝ := 1

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola x y → 
    real_axis_length = 4 ∧ 
    foci_to_asymptote_distance = 1) := by
  sorry

end hyperbola_properties_l2400_240064


namespace probability_of_three_in_18_23_l2400_240028

/-- The decimal representation of a rational number -/
def decimalRepresentation (n d : ℕ) : List ℕ :=
  sorry

/-- Count the occurrences of a digit in a list of digits -/
def countOccurrences (digit : ℕ) (digits : List ℕ) : ℕ :=
  sorry

/-- The probability of selecting a specific digit from a decimal representation -/
def probabilityOfDigit (n d digit : ℕ) : ℚ :=
  let digits := decimalRepresentation n d
  (countOccurrences digit digits : ℚ) / (digits.length : ℚ)

theorem probability_of_three_in_18_23 :
  probabilityOfDigit 18 23 3 = 3 / 26 := by
  sorry

end probability_of_three_in_18_23_l2400_240028


namespace pokemon_cards_distribution_l2400_240083

theorem pokemon_cards_distribution (total_cards : ℕ) (num_friends : ℕ) 
  (h1 : total_cards = 56) (h2 : num_friends = 4) (h3 : num_friends > 0) :
  total_cards / num_friends = 14 := by
  sorry

end pokemon_cards_distribution_l2400_240083


namespace problem_solution_l2400_240039

theorem problem_solution (a b c d : ℚ) :
  (2*a + 2 = 3*b + 3) ∧
  (3*b + 3 = 4*c + 4) ∧
  (4*c + 4 = 5*d + 5) ∧
  (5*d + 5 = 2*a + 3*b + 4*c + 5*d + 6) →
  2*a + 3*b + 4*c + 5*d = -10/3 := by
sorry

end problem_solution_l2400_240039


namespace quadratic_positivity_set_l2400_240016

/-- Given a quadratic function with zeros at -2 and 3, prove its positivity set -/
theorem quadratic_positivity_set 
  (y : ℝ → ℝ) 
  (h1 : ∀ x, y x = x^2 + b*x + c) 
  (h2 : y (-2) = 0) 
  (h3 : y 3 = 0) :
  {x : ℝ | y x > 0} = {x | x < -2 ∨ x > 3} :=
sorry

end quadratic_positivity_set_l2400_240016


namespace distribution_plans_count_l2400_240090

/-- The number of ways to distribute 3 distinct items into 3 distinct boxes,
    where each box must contain at least one item -/
def distribution_count : ℕ := 12

/-- Theorem stating that the number of distribution plans is correct -/
theorem distribution_plans_count : distribution_count = 12 := by
  sorry

end distribution_plans_count_l2400_240090


namespace fruit_count_l2400_240061

theorem fruit_count (total fruits apples oranges bananas : ℕ) : 
  total = 12 → 
  apples = 3 → 
  oranges = 5 → 
  total = apples + oranges + bananas → 
  bananas = 4 :=
by sorry

end fruit_count_l2400_240061


namespace triangle_side_b_value_l2400_240027

noncomputable def triangle_side_b (a : ℝ) (A B : ℝ) : ℝ :=
  2 * Real.sqrt 6

theorem triangle_side_b_value (a : ℝ) (A B : ℝ) 
  (h1 : a = 3)
  (h2 : B = 2 * A)
  (h3 : Real.cos A = Real.sqrt 6 / 3) :
  triangle_side_b a A B = 2 * Real.sqrt 6 :=
by
  sorry

end triangle_side_b_value_l2400_240027


namespace deepak_age_l2400_240045

/-- Given the ratio of ages and future ages of Rahul and Sandeep, prove Deepak's present age --/
theorem deepak_age (r d s : ℕ) : 
  r / d = 4 / 3 →  -- ratio of Rahul to Deepak's age
  d / s = 1 / 2 →  -- ratio of Deepak to Sandeep's age
  r + 6 = 42 →     -- Rahul's age after 6 years
  s + 9 = 57 →     -- Sandeep's age after 9 years
  d = 27 :=        -- Deepak's present age
by sorry

end deepak_age_l2400_240045


namespace non_square_seq_2003_l2400_240073

/-- The sequence of positive integers with perfect squares removed -/
def non_square_seq : ℕ → ℕ := sorry

/-- The 2003rd term of the sequence of positive integers with perfect squares removed is 2048 -/
theorem non_square_seq_2003 : non_square_seq 2003 = 2048 := by sorry

end non_square_seq_2003_l2400_240073


namespace flood_damage_conversion_l2400_240006

/-- Converts Australian dollars to US dollars given an exchange rate -/
def aud_to_usd (aud_amount : ℝ) (exchange_rate : ℝ) : ℝ :=
  aud_amount * exchange_rate

/-- Theorem stating the conversion of flood damage from AUD to USD -/
theorem flood_damage_conversion (damage_aud : ℝ) (exchange_rate : ℝ) 
  (h1 : damage_aud = 45000000)
  (h2 : exchange_rate = 0.7) :
  aud_to_usd damage_aud exchange_rate = 31500000 :=
by sorry

end flood_damage_conversion_l2400_240006


namespace taxi_fare_fraction_l2400_240059

/-- Represents the taxi fare structure and proves the fraction of a mile for each part. -/
theorem taxi_fare_fraction (first_part_cost : ℚ) (additional_part_cost : ℚ) 
  (total_distance : ℚ) (total_cost : ℚ) : 
  first_part_cost = 21/10 →
  additional_part_cost = 2/5 →
  total_distance = 8 →
  total_cost = 177/10 →
  ∃ (part_fraction : ℚ), 
    part_fraction = 7/39 ∧
    first_part_cost + (total_distance - 1) * (additional_part_cost / part_fraction) = total_cost :=
by sorry

end taxi_fare_fraction_l2400_240059


namespace tank_height_problem_l2400_240077

/-- Given two right circular cylinders A and B, where A has a circumference of 6 meters,
    B has a height of 6 meters and a circumference of 10 meters, and A's capacity is 60% of B's capacity,
    prove that the height of A is 10 meters. -/
theorem tank_height_problem (h_A : ℝ) : 
  let r_A : ℝ := 3 / Real.pi
  let r_B : ℝ := 5 / Real.pi
  let volume_A : ℝ := Real.pi * r_A^2 * h_A
  let volume_B : ℝ := Real.pi * r_B^2 * 6
  volume_A = 0.6 * volume_B → h_A = 10 := by
  sorry

#check tank_height_problem

end tank_height_problem_l2400_240077


namespace number_division_problem_l2400_240055

theorem number_division_problem (x : ℝ) : (x / 2.5) / 3.1 + 3.1 = 8.9 → x = 44.95 := by
  sorry

end number_division_problem_l2400_240055


namespace ball_count_proof_l2400_240031

theorem ball_count_proof (a : ℕ) (h1 : a > 0) (h2 : 3 ≤ a) : 
  (3 : ℚ) / a = 1 / 4 → a = 12 := by
sorry

end ball_count_proof_l2400_240031


namespace max_sum_of_coefficients_l2400_240086

/-- Given a temperature function T(t) = a * sin(t) + b * cos(t) where t ∈ (0, +∞),
    a and b are positive real numbers, and the maximum temperature difference is 10°C,
    prove that the maximum value of a + b is 5√2. -/
theorem max_sum_of_coefficients (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ t, t > 0 → t < Real.pi → ∃ T, T = a * Real.sin t + b * Real.cos t) →
  (∃ t₁ t₂, t₁ > 0 ∧ t₂ > 0 ∧ t₁ < Real.pi ∧ t₂ < Real.pi ∧
    a * Real.sin t₁ + b * Real.cos t₁ - (a * Real.sin t₂ + b * Real.cos t₂) = 10) →
  a + b ≤ 5 * Real.sqrt 2 :=
sorry

end max_sum_of_coefficients_l2400_240086


namespace josh_film_purchase_l2400_240034

/-- The number of films Josh bought -/
def num_films : ℕ := 9

/-- The number of books Josh bought -/
def num_books : ℕ := 4

/-- The number of CDs Josh bought -/
def num_cds : ℕ := 6

/-- The cost of each film in dollars -/
def cost_per_film : ℕ := 5

/-- The cost of each book in dollars -/
def cost_per_book : ℕ := 4

/-- The cost of each CD in dollars -/
def cost_per_cd : ℕ := 3

/-- The total amount Josh spent in dollars -/
def total_spent : ℕ := 79

/-- Theorem stating that the number of films Josh bought is correct -/
theorem josh_film_purchase :
  num_films * cost_per_film + num_books * cost_per_book + num_cds * cost_per_cd = total_spent :=
by sorry

end josh_film_purchase_l2400_240034


namespace max_value_of_complex_expression_l2400_240080

theorem max_value_of_complex_expression (a b : ℝ) : 
  let z : ℂ := Complex.mk a b
  (Complex.abs z = 2) →
  (∃ (m : ℝ), m = 9 ∧ ∀ (x y : ℝ), 
    let w : ℂ := Complex.mk x y
    Complex.abs w = 2 → 
    Complex.abs ((w - 1) * (w + 1)^2) ≤ m) :=
by sorry

end max_value_of_complex_expression_l2400_240080


namespace worker_completion_time_l2400_240024

/-- Given workers A and B, where A can complete a job in 15 days,
    A works for 5 days, and B finishes the remaining work in 12 days,
    prove that B alone can complete the entire job in 18 days. -/
theorem worker_completion_time (a_total_days b_remaining_days : ℕ) 
    (h1 : a_total_days = 15)
    (h2 : b_remaining_days = 12) : 
  (18 : ℚ) = (b_remaining_days : ℚ) / ((a_total_days - 5 : ℚ) / a_total_days) := by
  sorry

end worker_completion_time_l2400_240024


namespace sum_of_seventh_row_l2400_240003

-- Define the sum function for the triangular array
def f : ℕ → ℕ
  | 0 => 0  -- Base case: f(0) = 0 (not used in the problem, but needed for recursion)
  | 1 => 2  -- Base case: f(1) = 2
  | n + 1 => 2 * f n + 4  -- Recursive case: f(n+1) = 2f(n) + 4

-- Theorem statement
theorem sum_of_seventh_row : f 7 = 284 := by
  sorry

end sum_of_seventh_row_l2400_240003


namespace prime_factors_of_2_pow_8_minus_1_l2400_240041

theorem prime_factors_of_2_pow_8_minus_1 :
  ∃ (p q r : ℕ), 
    Prime p ∧ Prime q ∧ Prime r ∧ 
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    p * q * r = 2^8 - 1 ∧
    p + q + r = 25 := by
  sorry

end prime_factors_of_2_pow_8_minus_1_l2400_240041


namespace pears_left_l2400_240032

theorem pears_left (keith_picked mike_picked keith_gave_away : ℕ) 
  (h1 : keith_picked = 47)
  (h2 : mike_picked = 12)
  (h3 : keith_gave_away = 46) :
  keith_picked - keith_gave_away + mike_picked = 13 := by
sorry

end pears_left_l2400_240032


namespace quadratic_function_properties_l2400_240066

/-- A quadratic function f(x) = ax^2 + bx satisfying certain conditions -/
def QuadraticFunction (a b : ℝ) (ha : a ≠ 0) : ℝ → ℝ := fun x ↦ a * x^2 + b * x

theorem quadratic_function_properties 
  (a b : ℝ) (ha : a ≠ 0) 
  (h1 : ∀ x, QuadraticFunction a b ha (x - 1) = QuadraticFunction a b ha (3 - x))
  (h2 : ∃! x, QuadraticFunction a b ha x = 2 * x) :
  (∀ x, QuadraticFunction a b ha x = -x^2 + 2*x) ∧ 
  (∀ t, (t : ℝ) > 0 → 
    (∀ x, x ∈ Set.Icc 0 t → QuadraticFunction a b ha x ≤ 
      (if t > 1 then 1 else -t^2 + 2*t))) := by
  sorry


end quadratic_function_properties_l2400_240066


namespace sports_league_games_l2400_240013

theorem sports_league_games (total_teams : Nat) (divisions : Nat) (teams_per_division : Nat)
  (intra_division_games : Nat) (inter_division_games : Nat) :
  total_teams = divisions * teams_per_division →
  divisions = 3 →
  teams_per_division = 4 →
  intra_division_games = 3 →
  inter_division_games = 1 →
  (total_teams * (((teams_per_division - 1) * intra_division_games) +
    ((total_teams - teams_per_division) * inter_division_games))) / 2 = 102 := by
  sorry

end sports_league_games_l2400_240013


namespace seven_valid_methods_l2400_240065

/-- The number of valid purchasing methods for software and tapes -/
def validPurchaseMethods : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    60 * p.1 + 70 * p.2 ≤ 500 ∧
    p.1 ≥ 3 ∧
    p.2 ≥ 2)
    (Finset.product (Finset.range 9) (Finset.range 8))).card

/-- Theorem stating that there are exactly 7 valid purchasing methods -/
theorem seven_valid_methods : validPurchaseMethods = 7 := by
  sorry

end seven_valid_methods_l2400_240065


namespace greatest_prime_factor_f_28_l2400_240082

def f (m : ℕ) : ℕ :=
  if m % 2 = 0 ∧ m > 0 then
    (List.range (m / 2)).foldl (λ acc i => acc * (2 * i + 2)) 1
  else
    0

theorem greatest_prime_factor_f_28 :
  (Nat.factors (f 28)).maximum? = some 13 :=
sorry

end greatest_prime_factor_f_28_l2400_240082


namespace printer_time_relationship_l2400_240020

/-- Represents a printer's capability to print leaflets -/
structure Printer :=
  (time : ℝ)  -- Time taken to print 800 leaflets

/-- Represents a system of two printers -/
structure PrinterSystem :=
  (printer1 : Printer)
  (printer2 : Printer)
  (combined_time : ℝ)  -- Time taken by both printers together to print 800 leaflets

/-- Theorem stating the relationship between individual printer times and combined time -/
theorem printer_time_relationship (system : PrinterSystem) 
    (h1 : system.printer1.time = 12)
    (h2 : system.combined_time = 3) :
    (1 / system.printer1.time) + (1 / system.printer2.time) = (1 / system.combined_time) :=
  sorry

end printer_time_relationship_l2400_240020


namespace polynomial_remainder_l2400_240062

def polynomial (x : ℝ) : ℝ := 5*x^8 - 3*x^7 + 2*x^6 - 8*x^4 + 6*x^3 - 9

def divisor (x : ℝ) : ℝ := 3*x - 9

theorem polynomial_remainder :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * (q x) + 26207 := by
  sorry

end polynomial_remainder_l2400_240062


namespace word_exists_l2400_240070

/-- Represents a word in the Russian language -/
structure RussianWord where
  word : String

/-- Represents a festive dance event -/
structure FestiveDanceEvent where
  name : String

/-- Represents a sport -/
inductive Sport
  | FigureSkating
  | RhythmicGymnastics

/-- Represents the Russian pension system -/
structure RussianPensionSystem where
  startYear : Nat
  calculationMethod : String

/-- The word we're looking for satisfies all conditions -/
def satisfiesAllConditions (w : RussianWord) (f : FestiveDanceEvent) (s : Sport) (p : RussianPensionSystem) : Prop :=
  (w.word.toLower = f.name.toLower) ∧ 
  (match s with
    | Sport.FigureSkating => true
    | Sport.RhythmicGymnastics => true) ∧
  (p.startYear = 2015 ∧ p.calculationMethod = w.word)

theorem word_exists : 
  ∃ (w : RussianWord) (f : FestiveDanceEvent) (s : Sport) (p : RussianPensionSystem), 
    satisfiesAllConditions w f s p :=
sorry

end word_exists_l2400_240070


namespace distance_ratio_l2400_240097

-- Define the total distance and traveled distance
def total_distance : ℝ := 234
def traveled_distance : ℝ := 156

-- Define the theorem
theorem distance_ratio :
  let remaining_distance := total_distance - traveled_distance
  (traveled_distance / remaining_distance) = 2 := by
sorry

end distance_ratio_l2400_240097


namespace unique_prime_with_remainder_l2400_240078

theorem unique_prime_with_remainder : ∃! m : ℕ,
  Prime m ∧ 30 < m ∧ m < 50 ∧ m % 12 = 7 :=
by
  -- The proof would go here
  sorry

end unique_prime_with_remainder_l2400_240078


namespace largest_integer_with_remainder_l2400_240014

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 74 ∧ n % 7 = 3 ∧ ∀ m : ℕ, m < 74 ∧ m % 7 = 3 → m ≤ n → n = 73 := by
  sorry

end largest_integer_with_remainder_l2400_240014


namespace quadratic_equation_roots_l2400_240019

theorem quadratic_equation_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + 1 = 0 → x < 0) → 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*a*x + 1 = 0 ∧ y^2 + 2*a*y + 1 = 0) → 
  a ≥ 1 :=
sorry

end quadratic_equation_roots_l2400_240019


namespace factorial_ratio_l2400_240052

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 10 = 132 := by
  sorry

end factorial_ratio_l2400_240052


namespace function_satisfies_conditions_l2400_240042

theorem function_satisfies_conditions (m n : ℕ) :
  let f : ℕ → ℕ → ℕ := λ m n => m * n
  (m ≥ 1 ∧ n ≥ 1 → 2 * f m n = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1)) ∧
  (∀ k : ℕ, f k 0 = 0 ∧ f 0 k = 0) :=
by sorry

end function_satisfies_conditions_l2400_240042


namespace complex_magnitude_l2400_240053

theorem complex_magnitude (a : ℝ) :
  (∃ (b : ℝ), (a + I) / (2 - I) = b * I) →
  Complex.abs (1/2 + (a + I) / (2 - I)) = Real.sqrt 2 / 2 := by
  sorry

end complex_magnitude_l2400_240053


namespace jezebel_roses_l2400_240068

/-- The number of red roses Jezebel needs to buy -/
def num_red_roses : ℕ := sorry

/-- The cost of one red rose in dollars -/
def cost_red_rose : ℚ := 3/2

/-- The number of sunflowers Jezebel needs to buy -/
def num_sunflowers : ℕ := 3

/-- The cost of one sunflower in dollars -/
def cost_sunflower : ℚ := 3

/-- The total cost of all flowers in dollars -/
def total_cost : ℚ := 45

theorem jezebel_roses :
  num_red_roses * cost_red_rose + num_sunflowers * cost_sunflower = total_cost ∧
  num_red_roses = 24 := by sorry

end jezebel_roses_l2400_240068


namespace negation_of_proposition_l2400_240060

theorem negation_of_proposition (P : Prop) :
  (¬ (∃ x : ℝ, 2 * x + 1 ≤ 0)) ↔ (∀ x : ℝ, 2 * x + 1 > 0) := by sorry

end negation_of_proposition_l2400_240060


namespace general_equation_l2400_240074

theorem general_equation (n : ℤ) : 
  (n / (n - 4)) + ((8 - n) / ((8 - n) - 4)) = 2 :=
by sorry

end general_equation_l2400_240074


namespace largest_divisible_number_l2400_240048

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem largest_divisible_number : 
  (∀ m : ℕ, 5 ≤ m ∧ m ≤ 10 → is_divisible 2520 m) ∧ 
  ¬(∀ m : ℕ, 5 ≤ m ∧ m ≤ 11 → is_divisible 2520 m) := by
  sorry

end largest_divisible_number_l2400_240048


namespace constant_b_proof_l2400_240071

theorem constant_b_proof (a b c : ℝ) : 
  (∀ x : ℝ, (3 * x^2 - 2 * x + 4) * (a * x^2 + b * x + c) = 
    6 * x^4 - 5 * x^3 + 11 * x^2 - 8 * x + 16) → 
  b = -1/3 := by
sorry

end constant_b_proof_l2400_240071


namespace right_triangle_inequality_l2400_240067

theorem right_triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a^2 + b^2 = c^2) : 
  (a^3 + b^3 + c^3) / (a * b * (a + b + c)) ≥ Real.sqrt 2 := by
  sorry

end right_triangle_inequality_l2400_240067


namespace floor_negative_fraction_l2400_240076

theorem floor_negative_fraction : ⌊(-19 : ℝ) / 3⌋ = -7 := by sorry

end floor_negative_fraction_l2400_240076


namespace student_A_consecutive_days_probability_l2400_240087

/-- The number of days for the volunteer activity -/
def total_days : ℕ := 5

/-- The total number of students participating -/
def total_students : ℕ := 4

/-- The number of days student A participates -/
def student_A_days : ℕ := 2

/-- The number of days each other student participates -/
def other_student_days : ℕ := 1

/-- The probability that student A participates for two consecutive days -/
def consecutive_days_probability : ℚ := 2 / 5

/-- Theorem stating that the probability of student A participating for two consecutive days is 2/5 -/
theorem student_A_consecutive_days_probability :
  consecutive_days_probability = 2 / 5 := by sorry

end student_A_consecutive_days_probability_l2400_240087


namespace max_sum_of_squares_l2400_240007

theorem max_sum_of_squares (a b c : ℝ) 
  (h1 : a + b = c - 1) 
  (h2 : a * b = c^2 - 7*c + 14) : 
  ∃ (m : ℝ), (∀ (x y z : ℝ), x + y = z - 1 → x * y = z^2 - 7*z + 14 → x^2 + y^2 ≤ m) ∧ a^2 + b^2 = m :=
sorry

end max_sum_of_squares_l2400_240007


namespace max_blocks_in_box_l2400_240051

/-- Represents the dimensions of a rectangular box. -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of a block. -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of blocks that can fit in a box. -/
def maxBlocksFit (box : BoxDimensions) (block : BlockDimensions) : ℕ :=
  let boxVolume := box.length * box.width * box.height
  let blockVolume := block.length * block.width * block.height
  boxVolume / blockVolume

/-- Theorem stating that the maximum number of 3×1×1 blocks that can fit in a 4×3×2 box is 8. -/
theorem max_blocks_in_box :
  let box := BoxDimensions.mk 4 3 2
  let block := BlockDimensions.mk 3 1 1
  maxBlocksFit box block = 8 := by
  sorry

#eval maxBlocksFit (BoxDimensions.mk 4 3 2) (BlockDimensions.mk 3 1 1)

end max_blocks_in_box_l2400_240051


namespace square_difference_sum_l2400_240081

theorem square_difference_sum : 
  27^2 - 25^2 + 23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 392 := by
  sorry

end square_difference_sum_l2400_240081


namespace fraction_difference_l2400_240025

theorem fraction_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = x / y) :
  1 / x - 1 / y = -(1 / y^2) := by
sorry

end fraction_difference_l2400_240025


namespace base_three_20121_equals_178_l2400_240058

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

theorem base_three_20121_equals_178 :
  base_three_to_decimal [1, 2, 1, 0, 2] = 178 := by
  sorry

end base_three_20121_equals_178_l2400_240058


namespace bamboo_sections_volume_l2400_240079

theorem bamboo_sections_volume (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →
  a 1 + a 2 + a 3 = 3.9 →
  a 6 + a 7 + a 8 + a 9 = 3 →
  a 4 + a 5 = 2.1 := by
  sorry

end bamboo_sections_volume_l2400_240079


namespace common_root_is_neg_half_l2400_240012

/-- Definition of the first polynomial -/
def p (a b c : ℝ) (x : ℝ) : ℝ := 50 * x^4 + a * x^3 + b * x^2 + c * x + 16

/-- Definition of the second polynomial -/
def q (d e f g : ℝ) (x : ℝ) : ℝ := 16 * x^5 + d * x^4 + e * x^3 + f * x^2 + g * x + 50

/-- Theorem stating that if p and q have a common negative rational root, it must be -1/2 -/
theorem common_root_is_neg_half (a b c d e f g : ℝ) :
  (∃ (k : ℚ), k < 0 ∧ p a b c k = 0 ∧ q d e f g k = 0) →
  (p a b c (-1/2 : ℚ) = 0 ∧ q d e f g (-1/2 : ℚ) = 0) :=
by sorry

end common_root_is_neg_half_l2400_240012


namespace cubic_function_property_l2400_240089

theorem cubic_function_property (p q r s : ℝ) :
  (∀ x : ℝ, p * x^3 + q * x^2 + r * x + s = x * (x - 1) * (x + 2) / 6) →
  5 * p - 3 * q + 2 * r - s = 5 :=
by sorry

end cubic_function_property_l2400_240089


namespace sum_of_roots_l2400_240043

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 := by
sorry

end sum_of_roots_l2400_240043


namespace lowest_divisible_by_primes_10_to_50_l2400_240093

def primes_10_to_50 : List Nat := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def is_divisible_by_all (n : Nat) (list : List Nat) : Prop :=
  ∀ p ∈ list, n % p = 0

theorem lowest_divisible_by_primes_10_to_50 :
  ∃ (n : Nat), n > 0 ∧
  is_divisible_by_all n primes_10_to_50 ∧
  ∀ (m : Nat), m > 0 ∧ is_divisible_by_all m primes_10_to_50 → n ≤ m :=
by sorry

end lowest_divisible_by_primes_10_to_50_l2400_240093


namespace divisibility_of_sum_of_squares_minus_2017_l2400_240008

theorem divisibility_of_sum_of_squares_minus_2017 :
  ∀ n : ℕ, ∃ x y : ℤ, (n : ℤ) ∣ (x^2 + y^2 - 2017) :=
by sorry

end divisibility_of_sum_of_squares_minus_2017_l2400_240008


namespace fahrenheit_diff_is_18_l2400_240046

-- Define the conversion function from Celsius to Fahrenheit
def celsius_to_fahrenheit (C : ℝ) : ℝ := 1.8 * C + 32

-- Define the temperature difference in Celsius
def celsius_diff : ℝ := 10

-- Theorem statement
theorem fahrenheit_diff_is_18 :
  celsius_to_fahrenheit (C + celsius_diff) - celsius_to_fahrenheit C = 18 :=
by sorry

end fahrenheit_diff_is_18_l2400_240046


namespace percentage_problem_l2400_240085

theorem percentage_problem (x y : ℝ) (P : ℝ) :
  (0.1 * x = P / 100 * y) →  -- 10% of x equals P% of y
  (x / y = 2) →              -- The ratio of x to y is 2
  P = 20 :=                  -- The percentage of y is 20%
by
  sorry

end percentage_problem_l2400_240085


namespace primitive_root_mod_p_squared_l2400_240033

theorem primitive_root_mod_p_squared (p : Nat) (x : Nat) 
  (h_p : Nat.Prime p) 
  (h_p_odd : Odd p) 
  (h_x_prim_root : IsPrimitiveRoot x p) : 
  IsPrimitiveRoot x (p^2) ∨ IsPrimitiveRoot (x + p) (p^2) := by
  sorry

end primitive_root_mod_p_squared_l2400_240033


namespace complement_of_63_degrees_l2400_240057

theorem complement_of_63_degrees :
  let angle : ℝ := 63
  let complement (x : ℝ) : ℝ := 90 - x
  complement angle = 27 := by
  sorry

end complement_of_63_degrees_l2400_240057


namespace rowing_time_ratio_l2400_240091

/-- Proves that the ratio of time taken to row against the stream to the time taken to row in favor of the stream is 2:1, given that the boat's speed in still water is 3 times the stream's speed. -/
theorem rowing_time_ratio (B S D : ℝ) (h_positive : B > 0 ∧ S > 0 ∧ D > 0) (h_speed_ratio : B = 3 * S) :
  (D / (B - S)) / (D / (B + S)) = 2 := by
  sorry

end rowing_time_ratio_l2400_240091


namespace solution_triples_l2400_240088

theorem solution_triples : ∀ (a b c : ℝ),
  (a^2 + a*b + c = 0 ∧ b^2 + b*c + a = 0 ∧ c^2 + c*a + b = 0) →
  ((a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = -1/2 ∧ b = -1/2 ∧ c = -1/2)) :=
by sorry

end solution_triples_l2400_240088


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2400_240021

/-- An isosceles triangle with two sides of length 6 and one side of length 3 has a perimeter of 15 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b : ℝ),
      a = 6 ∧
      b = 3 ∧
      (a = a ∧ b ≤ a + a) ∧  -- Triangle inequality
      perimeter = a + a + b ∧
      perimeter = 15

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 15 := by
  sorry

#check isosceles_triangle_perimeter
#check isosceles_triangle_perimeter_proof

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2400_240021


namespace cube_difference_of_sum_and_diff_l2400_240092

theorem cube_difference_of_sum_and_diff (x y : ℕ) 
  (sum_eq : x + y = 64) 
  (diff_eq : x - y = 16) 
  (x_pos : x > 0) 
  (y_pos : y > 0) : 
  x^3 - y^3 = 50176 := by
sorry

end cube_difference_of_sum_and_diff_l2400_240092


namespace correct_calculation_l2400_240030

theorem correct_calculation (m n : ℝ) : 4*m + 2*n - (n - m) = 5*m + n := by
  sorry

end correct_calculation_l2400_240030


namespace right_triangle_area_l2400_240036

/-- The area of a right triangle with legs 20 and 21 is 210 -/
theorem right_triangle_area : 
  let a : ℝ := 20
  let b : ℝ := 21
  let area : ℝ := (1/2) * a * b
  area = 210 := by
sorry

end right_triangle_area_l2400_240036


namespace ru_length_is_8_25_l2400_240017

/-- Triangle PQR with given side lengths and specific geometric constructions -/
structure SpecialTriangle where
  /-- Side length PQ -/
  pq : ℝ
  /-- Side length QR -/
  qr : ℝ
  /-- Side length RP -/
  rp : ℝ
  /-- Point S on QR where the angle bisector of ∠PQR intersects QR -/
  s : ℝ × ℝ
  /-- Point T on the circumcircle of PQR where the angle bisector of ∠PQR intersects (T ≠ P) -/
  t : ℝ × ℝ
  /-- Point U on PQ where the circumcircle of PST intersects (U ≠ P) -/
  u : ℝ × ℝ
  /-- PQ = 13 -/
  h_pq : pq = 13
  /-- QR = 30 -/
  h_qr : qr = 30
  /-- RP = 26 -/
  h_rp : rp = 26
  /-- S is on QR -/
  h_s_on_qr : s.1 + s.2 = qr
  /-- T is on the circumcircle of PQR -/
  h_t_on_circumcircle : True  -- placeholder
  /-- U is on PQ -/
  h_u_on_pq : u.1 + u.2 = pq
  /-- T ≠ P -/
  h_t_ne_p : t ≠ (0, 0)
  /-- U ≠ P -/
  h_u_ne_p : u ≠ (0, 0)

/-- The length of RU in the special triangle construction -/
def ruLength (tri : SpecialTriangle) : ℝ := sorry

/-- Theorem stating that RU = 8.25 in the special triangle construction -/
theorem ru_length_is_8_25 (tri : SpecialTriangle) : ruLength tri = 8.25 := by
  sorry

end ru_length_is_8_25_l2400_240017


namespace matrix_not_invertible_l2400_240047

theorem matrix_not_invertible : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![![2 * (29/13 : ℝ) - 1, 5], ![4 + (29/13 : ℝ), 9]]
  ¬(IsUnit (Matrix.det A)) := by sorry

end matrix_not_invertible_l2400_240047


namespace not_divisible_5n_minus_1_by_4n_minus_1_l2400_240099

theorem not_divisible_5n_minus_1_by_4n_minus_1 (n : ℕ) :
  ¬ (5^n - 1 ∣ 4^n - 1) := by sorry

end not_divisible_5n_minus_1_by_4n_minus_1_l2400_240099


namespace one_fourth_divided_by_one_eighth_l2400_240096

theorem one_fourth_divided_by_one_eighth : (1 / 4 : ℚ) / (1 / 8 : ℚ) = 2 := by
  sorry

end one_fourth_divided_by_one_eighth_l2400_240096


namespace folded_paper_sum_l2400_240056

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a folded piece of graph paper -/
structure FoldedPaper where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point
  h1 : p1 = ⟨3, 3⟩
  h2 : p2 = ⟨7, 1⟩
  h3 : p3 = ⟨9, 4⟩

/-- The theorem to be proven -/
theorem folded_paper_sum (paper : FoldedPaper) : paper.p4.x + paper.p4.y = 28/3 := by
  sorry


end folded_paper_sum_l2400_240056


namespace road_breadth_from_fallen_tree_l2400_240035

/-- The breadth of a road when a tree falls across it -/
theorem road_breadth_from_fallen_tree (tree_height : ℝ) (break_height : ℝ) (road_breadth : ℝ) : 
  tree_height = 36 →
  break_height = 16 →
  (tree_height - break_height) ^ 2 = road_breadth ^ 2 + break_height ^ 2 →
  road_breadth = 12 := by
sorry

end road_breadth_from_fallen_tree_l2400_240035


namespace ships_met_equals_journey_duration_atlantic_crossing_meets_seven_ships_l2400_240040

/-- Represents a steamship journey between two ports -/
structure Journey where
  departureDays : ℕ  -- Number of days since the first ship departed
  travelDays : ℕ     -- Number of days the journey takes

/-- The number of ships met during a journey -/
def shipsMetDuringJourney (j : Journey) : ℕ :=
  j.travelDays

/-- Theorem: The number of ships met during a journey is equal to the journey's duration -/
theorem ships_met_equals_journey_duration (j : Journey) :
  shipsMetDuringJourney j = j.travelDays :=
by sorry

/-- The specific journey described in the problem -/
def atlanticCrossing : Journey :=
  { departureDays := 1,  -- A ship departs every day
    travelDays := 7 }    -- The journey takes 7 days

/-- Theorem: A ship crossing the Atlantic meets 7 other ships -/
theorem atlantic_crossing_meets_seven_ships :
  shipsMetDuringJourney atlanticCrossing = 7 :=
by sorry

end ships_met_equals_journey_duration_atlantic_crossing_meets_seven_ships_l2400_240040


namespace nth_equation_specific_case_l2400_240084

theorem nth_equation (n : ℕ) (hn : n > 0) :
  Real.sqrt (1 - (2 * n - 1) / (n * n)) = (n - 1) / n :=
by sorry

theorem specific_case : Real.sqrt (1 - 199 / 10000) = 99 / 100 :=
by sorry

end nth_equation_specific_case_l2400_240084


namespace square_side_length_l2400_240075

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 144 → side * side = area → side = 12 := by
  sorry

end square_side_length_l2400_240075


namespace tiling_symmetry_l2400_240001

/-- A rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Tiling relation between rectangles -/
def CanTile (A B : Rectangle) : Prop :=
  ∃ (n m : ℕ), n * A.width = m * B.width ∧ n * A.height = m * B.height

/-- Similarity relation between rectangles -/
def IsSimilarTo (A B : Rectangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ A.width = k * B.width ∧ A.height = k * B.height

/-- Main theorem: If a rectangle similar to A can be tiled with B, 
    then a rectangle similar to B can be tiled with A -/
theorem tiling_symmetry (A B : Rectangle) :
  (∃ (C : Rectangle), IsSimilarTo C A ∧ CanTile C B) →
  (∃ (D : Rectangle), IsSimilarTo D B ∧ CanTile D A) :=
by
  sorry


end tiling_symmetry_l2400_240001


namespace circle_fraction_l2400_240002

theorem circle_fraction (n : ℕ) (m : ℕ) (h1 : n > 0) (h2 : m ≤ n) :
  (m : ℚ) / n = m * (1 / n) :=
by sorry

#check circle_fraction

end circle_fraction_l2400_240002


namespace f_even_iff_a_zero_f_min_value_when_x_geq_a_l2400_240050

/-- Definition of the function f(x) -/
def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

/-- Theorem about the evenness of f(x) -/
theorem f_even_iff_a_zero (a : ℝ) :
  (∀ x, f a x = f a (-x)) ↔ a = 0 := by sorry

/-- Theorem about the minimum value of f(x) when x ≥ a -/
theorem f_min_value_when_x_geq_a (a : ℝ) :
  (∀ x ≥ a, f a x ≥ (if a ≤ -1/2 then 3/4 - a else a^2 + 1)) ∧
  (∃ x ≥ a, f a x = (if a ≤ -1/2 then 3/4 - a else a^2 + 1)) := by sorry

end f_even_iff_a_zero_f_min_value_when_x_geq_a_l2400_240050


namespace total_miles_jogged_l2400_240038

/-- The number of miles a person jogs per day on weekdays -/
def miles_per_day : ℕ := 5

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weeks -/
def num_weeks : ℕ := 3

/-- Theorem: A person who jogs 5 miles per day on weekdays will run 75 miles over three weeks -/
theorem total_miles_jogged : 
  miles_per_day * weekdays_per_week * num_weeks = 75 := by sorry

end total_miles_jogged_l2400_240038


namespace angle_sum_is_pi_over_two_l2400_240044

theorem angle_sum_is_pi_over_two (α β : Real) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2) 
  (h_equation : (Real.sin α)^4 / (Real.cos β)^2 + (Real.cos α)^4 / (Real.sin β)^2 = 1) : 
  α + β = π / 2 := by
sorry

end angle_sum_is_pi_over_two_l2400_240044


namespace function_properties_l2400_240000

def f (x : ℝ) : ℝ := |2*x - 1| + |x - 2|

theorem function_properties :
  (∀ k : ℝ, (∀ x₀ : ℝ, f x₀ ≥ |k + 3| - |k - 2|) ↔ k ≤ 1/4) ∧
  (∀ m n : ℝ, (∀ x : ℝ, f x ≥ 1/m + 1/n) → m + n ≥ 8/3) ∧
  (∃ m n : ℝ, (∀ x : ℝ, f x ≥ 1/m + 1/n) ∧ m + n = 8/3) := by sorry

end function_properties_l2400_240000


namespace unique_solution_exponential_equation_l2400_240022

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ) ^ (6 * x + 3) * (4 : ℝ) ^ (3 * x + 6) = (8 : ℝ) ^ (-4 * x + 5) :=
by sorry

end unique_solution_exponential_equation_l2400_240022


namespace multiply_divide_sqrt_equation_l2400_240015

theorem multiply_divide_sqrt_equation (x : ℝ) (y : ℝ) (h1 : x > 0) (h2 : x = 1.3333333333333333) :
  (x * y) / 3 = x^2 ↔ y = 4 := by
  sorry

end multiply_divide_sqrt_equation_l2400_240015


namespace tree_height_after_two_years_l2400_240063

/-- The height of a tree after n years, given its initial height and growth factor -/
def tree_height (initial_height : ℝ) (growth_factor : ℝ) (n : ℕ) : ℝ :=
  initial_height * growth_factor ^ n

/-- Theorem: If a tree triples its height every year and reaches 81 feet after 4 years,
    then its height after 2 years is 9 feet -/
theorem tree_height_after_two_years
  (h : ∃ initial_height : ℝ, tree_height initial_height 3 4 = 81) :
  ∃ initial_height : ℝ, tree_height initial_height 3 2 = 9 := by
  sorry

end tree_height_after_two_years_l2400_240063


namespace complement_union_theorem_l2400_240069

def U : Set Nat := {0, 1, 3, 5, 6, 8}
def A : Set Nat := {1, 5, 8}
def B : Set Nat := {2}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 3, 6} := by sorry

end complement_union_theorem_l2400_240069


namespace circle_area_from_circumference_l2400_240054

theorem circle_area_from_circumference : 
  ∀ (r : ℝ), 
    (2 * π * r = 18 * π) → 
    (π * r^2 = 81 * π) := by
  sorry

end circle_area_from_circumference_l2400_240054


namespace gordons_lighter_bag_weight_l2400_240072

/-- 
Given:
- Trace has 5 shopping bags
- Gordon has 2 shopping bags
- Trace's 5 bags weigh the same as Gordon's 2 bags
- One of Gordon's bags weighs 7 pounds
- Each of Trace's bags weighs 2 pounds

Prove that Gordon's lighter bag weighs 3 pounds.
-/
theorem gordons_lighter_bag_weight :
  ∀ (trace_bags gordon_bags : ℕ) 
    (trace_bag_weight gordon_heavy_bag_weight : ℝ)
    (total_trace_weight total_gordon_weight : ℝ),
  trace_bags = 5 →
  gordon_bags = 2 →
  trace_bag_weight = 2 →
  gordon_heavy_bag_weight = 7 →
  total_trace_weight = trace_bags * trace_bag_weight →
  total_gordon_weight = gordon_heavy_bag_weight + (total_trace_weight - gordon_heavy_bag_weight) →
  total_trace_weight = total_gordon_weight →
  (total_trace_weight - gordon_heavy_bag_weight) = 3 :=
by sorry

end gordons_lighter_bag_weight_l2400_240072


namespace delay_and_wait_l2400_240049

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

def addMinutes (t : Time) (m : Nat) : Time := sorry

theorem delay_and_wait (start : Time) (delay : Nat) (wait : Nat) : 
  start.hours = 3 ∧ start.minutes = 0 → 
  delay = 30 → 
  wait = 2500 → 
  (addMinutes (addMinutes start delay) wait).hours = 21 ∧ 
  (addMinutes (addMinutes start delay) wait).minutes = 10 := by
  sorry

end delay_and_wait_l2400_240049


namespace product_sum_relation_l2400_240094

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 10 → b = 9 → b - a = 5 := by
  sorry

end product_sum_relation_l2400_240094


namespace kennel_arrangement_count_l2400_240095

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def num_chickens : ℕ := 6
def num_dogs : ℕ := 4
def num_cats : ℕ := 5

def total_arrangements : ℕ := 2 * factorial num_chickens * factorial num_dogs * factorial num_cats

theorem kennel_arrangement_count :
  total_arrangements = 4147200 :=
by sorry

end kennel_arrangement_count_l2400_240095


namespace arctan_tan_difference_l2400_240018

theorem arctan_tan_difference (θ₁ θ₂ : Real) (h₁ : θ₁ = 75 * π / 180) (h₂ : θ₂ = 35 * π / 180) :
  Real.arctan (Real.tan θ₁ - 2 * Real.tan θ₂) = 15 * π / 180 := by
  sorry

#check arctan_tan_difference

end arctan_tan_difference_l2400_240018


namespace stone_piles_total_l2400_240023

/-- Represents the number of stones in each pile -/
structure StonePiles where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ
  pile4 : ℕ
  pile5 : ℕ

/-- The conditions of the stone pile problem -/
def satisfiesConditions (piles : StonePiles) : Prop :=
  piles.pile5 = 6 * piles.pile3 ∧
  piles.pile2 = 2 * (piles.pile3 + piles.pile5) ∧
  piles.pile1 = piles.pile5 / 3 ∧
  piles.pile1 = piles.pile4 - 10 ∧
  piles.pile4 = piles.pile2 / 2

/-- The theorem stating that any StonePiles satisfying the conditions will have a total of 60 stones -/
theorem stone_piles_total (piles : StonePiles) :
  satisfiesConditions piles →
  piles.pile1 + piles.pile2 + piles.pile3 + piles.pile4 + piles.pile5 = 60 := by
  sorry

end stone_piles_total_l2400_240023


namespace second_sibling_age_difference_l2400_240010

theorem second_sibling_age_difference (Y x : ℕ) : 
  Y = 17 → 
  (Y + (Y + x) + (Y + 4) + (Y + 7)) / 4 = 21 → 
  x = 5 := by
sorry

end second_sibling_age_difference_l2400_240010


namespace john_arcade_spending_l2400_240026

/-- The fraction of John's allowance spent at the arcade -/
def arcade_fraction (allowance arcade_spent : ℚ) : ℚ :=
  arcade_spent / allowance

/-- The amount remaining after spending at the arcade and toy store -/
def remaining_after_toy_store (allowance arcade_spent : ℚ) : ℚ :=
  allowance - arcade_spent - (1/3) * (allowance - arcade_spent)

theorem john_arcade_spending :
  ∃ (arcade_spent : ℚ),
    arcade_fraction 3.30 arcade_spent = 3/5 ∧
    remaining_after_toy_store 3.30 arcade_spent = 0.88 := by
  sorry

end john_arcade_spending_l2400_240026
