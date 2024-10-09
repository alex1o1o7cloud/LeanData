import Mathlib

namespace total_payment_is_53_l1072_107220

-- Conditions
def bobBill : ℝ := 30
def kateBill : ℝ := 25
def bobDiscountRate : ℝ := 0.05
def kateDiscountRate : ℝ := 0.02

-- Calculations
def bobDiscount := bobBill * bobDiscountRate
def kateDiscount := kateBill * kateDiscountRate
def bobPayment := bobBill - bobDiscount
def katePayment := kateBill - kateDiscount

-- Goal
def totalPayment := bobPayment + katePayment

-- Theorem statement
theorem total_payment_is_53 : totalPayment = 53 := by
  sorry

end total_payment_is_53_l1072_107220


namespace count_divisible_by_45_l1072_107228

theorem count_divisible_by_45 : ∃ n : ℕ, n = 10 ∧ (∀ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ x % 100 = 45 → x % 45 = 0 → n = 10) :=
by {
  sorry
}

end count_divisible_by_45_l1072_107228


namespace find_PS_length_l1072_107224

theorem find_PS_length 
  (PT TR QS QP PQ : ℝ)
  (h1 : PT = 5)
  (h2 : TR = 10)
  (h3 : QS = 16)
  (h4 : QP = 13)
  (h5 : PQ = 7) : 
  PS = Real.sqrt 703 := 
sorry

end find_PS_length_l1072_107224


namespace purely_imaginary_z_value_l1072_107231

theorem purely_imaginary_z_value (a : ℝ) (h : (a^2 - a - 2) = 0 ∧ (a + 1) ≠ 0) : a = 2 :=
sorry

end purely_imaginary_z_value_l1072_107231


namespace normal_CDF_is_correct_l1072_107282

noncomputable def normal_cdf (a σ : ℝ) (x : ℝ) : ℝ :=
  0.5 + (1 / Real.sqrt (2 * Real.pi)) * ∫ t in (0)..(x - a) / σ, Real.exp (-t^2 / 2)

theorem normal_CDF_is_correct (a σ : ℝ) (ha : σ > 0) (x : ℝ) :
  (normal_cdf a σ x) = 0.5 + (1 / Real.sqrt (2 * Real.pi)) * ∫ t in (0)..(x - a) / σ, Real.exp (-t^2 / 2) :=
by
  sorry

end normal_CDF_is_correct_l1072_107282


namespace total_profit_is_35000_l1072_107272

open Real

-- Define the subscriptions of A, B, and C
def subscriptions (A B C : ℝ) : Prop :=
  A + B + C = 50000 ∧
  A = B + 4000 ∧
  B = C + 5000

-- Define the profit distribution and the condition for C's received profit
def profit (total_profit : ℝ) (A B C : ℝ) (C_profit : ℝ) : Prop :=
  C_profit / total_profit = C / (A + B + C) ∧
  C_profit = 8400

-- Lean 4 statement to prove total profit
theorem total_profit_is_35000 :
  ∃ A B C total_profit, subscriptions A B C ∧ profit total_profit A B C 8400 ∧ total_profit = 35000 :=
by
  sorry

end total_profit_is_35000_l1072_107272


namespace fountain_water_after_25_days_l1072_107246

def initial_volume : ℕ := 120
def evaporation_rate : ℕ := 8 / 10 -- Representing 0.8 gallons as 8/10
def rain_addition : ℕ := 5
def days : ℕ := 25
def rain_period : ℕ := 5

-- Calculate the amount of water after 25 days given the above conditions
theorem fountain_water_after_25_days :
  initial_volume + ((days / rain_period) * rain_addition) - (days * evaporation_rate) = 125 :=
by
  sorry

end fountain_water_after_25_days_l1072_107246


namespace find_m_l1072_107285

noncomputable def m_solution (m : ℝ) : ℂ := (m - 3 * Complex.I) / (2 + Complex.I)

theorem find_m :
  ∀ (m : ℝ), Complex.im (m_solution m) ≠ 0 → Complex.re (m_solution m) = 0 → m = 3 / 2 :=
by
  intro m h_im h_re
  sorry

end find_m_l1072_107285


namespace simplify_fraction_multiplication_l1072_107232

theorem simplify_fraction_multiplication:
  (101 / 5050) * 50 = 1 := by
  sorry

end simplify_fraction_multiplication_l1072_107232


namespace min_value_expression_l1072_107262

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x + y) * (1 / x + 1 / y) ≥ 6 := 
by
  sorry

end min_value_expression_l1072_107262


namespace age_ratio_l1072_107284

variables (A B : ℕ)
def present_age_of_A : ℕ := 15
def future_ratio (A B : ℕ) : Prop := (A + 6) / (B + 6) = 7 / 5

theorem age_ratio (A_eq : A = present_age_of_A) (future_ratio_cond : future_ratio A B) : A / B = 5 / 3 :=
sorry

end age_ratio_l1072_107284


namespace quadratic_perfect_square_form_l1072_107245

def quadratic_is_perfect_square (a b c : ℤ) : Prop :=
  ∀ x : ℤ, ∃ k : ℤ, a * x^2 + b * x + c = k^2

theorem quadratic_perfect_square_form (a b c : ℤ) (h : quadratic_is_perfect_square a b c) :
  ∃ d e : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (d * x + e)^2 :=
  sorry

end quadratic_perfect_square_form_l1072_107245


namespace that_three_digit_multiples_of_5_and_7_l1072_107266

/-- 
Define the count_three_digit_multiples function, 
which counts the number of three-digit integers that are multiples of both 5 and 7.
-/
def count_three_digit_multiples : ℕ :=
  let lcm := Nat.lcm 5 7
  let first := (100 + lcm - 1) / lcm * lcm
  let last := 999 / lcm * lcm
  (last - first) / lcm + 1

/-- 
Theorem that states the number of positive three-digit integers that are multiples of both 5 and 7 is 26. 
-/
theorem three_digit_multiples_of_5_and_7 : count_three_digit_multiples = 26 := by
  sorry

end that_three_digit_multiples_of_5_and_7_l1072_107266


namespace quadratic_transformation_l1072_107254

theorem quadratic_transformation (x d e : ℝ) (h : x^2 - 24*x + 45 = (x+d)^2 + e) : d + e = -111 :=
sorry

end quadratic_transformation_l1072_107254


namespace maximize_profit_l1072_107212

noncomputable section

def price (x : ℕ) : ℝ :=
  if 0 < x ∧ x ≤ 100 then 60
  else if 100 < x ∧ x ≤ 600 then 62 - 0.02 * x
  else 0

def profit (x : ℕ) : ℝ :=
  (price x - 40) * x

theorem maximize_profit :
  ∃ x : ℕ, (1 ≤ x ∧ x ≤ 600) ∧ (∀ y : ℕ, (1 ≤ y ∧ y ≤ 600 → profit y ≤ profit x)) ∧ profit x = 6050 :=
by sorry

end maximize_profit_l1072_107212


namespace base_conversion_and_operations_l1072_107251

-- Definitions to convert numbers from bases 7, 5, and 6 to base 10
def base7_to_nat (n : ℕ) : ℕ := 
  8 * 7^0 + 6 * 7^1 + 4 * 7^2 + 2 * 7^3

def base5_to_nat (n : ℕ) : ℕ := 
  1 * 5^0 + 2 * 5^1 + 1 * 5^2

def base6_to_nat (n : ℕ) : ℕ := 
  1 * 6^0 + 5 * 6^1 + 4 * 6^2 + 3 * 6^3

def base7_to_nat2 (n : ℕ) : ℕ := 
  1 * 7^0 + 9 * 7^1 + 8 * 7^2 + 7 * 7^3

-- Problem statement: Perform the arithmetical operations
theorem base_conversion_and_operations : 
  (base7_to_nat 2468 / base5_to_nat 121) - base6_to_nat 3451 + base7_to_nat2 7891 = 2059 := 
by
  sorry

end base_conversion_and_operations_l1072_107251


namespace inequality_proof_l1072_107269

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  sqrt (a^2 + b^2 - sqrt 2 * a * b) + sqrt (b^2 + c^2 - sqrt 2 * b * c)  ≥ sqrt (a^2 + c^2) :=
by sorry

end inequality_proof_l1072_107269


namespace average_age_of_girls_l1072_107277

variable (B G : ℝ)
variable (age_students age_boys age_girls : ℝ)
variable (ratio_boys_girls : ℝ)

theorem average_age_of_girls :
  age_students = 15.8 ∧ age_boys = 16.2 ∧ ratio_boys_girls = 1.0000000000000044 ∧ B / G = ratio_boys_girls →
  (B * age_boys + G * age_girls) / (B + G) = age_students →
  age_girls = 15.4 :=
by
  intros hconds haverage
  sorry

end average_age_of_girls_l1072_107277


namespace total_cost_correct_l1072_107276

-- Define the conditions
def uber_cost : ℤ := 22
def lyft_additional_cost : ℤ := 3
def taxi_additional_cost : ℤ := 4
def tip_percentage : ℚ := 0.20

-- Define the variables for cost of Lyft and Taxi based on the problem
def lyft_cost : ℤ := uber_cost - lyft_additional_cost
def taxi_cost : ℤ := lyft_cost - taxi_additional_cost

-- Calculate the tip
def tip : ℚ := taxi_cost * tip_percentage

-- Final total cost including the tip
def total_cost : ℚ := taxi_cost + tip

-- The theorem to prove
theorem total_cost_correct :
  total_cost = 18 := by
  sorry

end total_cost_correct_l1072_107276


namespace probability_one_white_one_black_two_touches_l1072_107252

def probability_white_ball : ℚ := 7 / 10
def probability_black_ball : ℚ := 3 / 10

theorem probability_one_white_one_black_two_touches :
  (probability_white_ball * probability_black_ball) + (probability_black_ball * probability_white_ball) = (7 / 10) * (3 / 10) + (3 / 10) * (7 / 10) :=
by
  -- The proof is omitted here.
  sorry

end probability_one_white_one_black_two_touches_l1072_107252


namespace children_count_l1072_107210

-- Define the total number of passengers on the airplane
def total_passengers : ℕ := 240

-- Define the ratio of men to women
def men_to_women_ratio : ℕ × ℕ := (3, 2)

-- Define the percentage of passengers who are either men or women
def percent_men_women : ℕ := 60

-- Define the number of children on the airplane
def number_of_children (total : ℕ) (percent : ℕ) : ℕ := 
  (total * (100 - percent)) / 100

theorem children_count :
  number_of_children total_passengers percent_men_women = 96 := by
  sorry

end children_count_l1072_107210


namespace Jake_peaches_l1072_107271

variables (Jake Steven Jill : ℕ)

def peaches_relation : Prop :=
  (Jake = Steven - 6) ∧
  (Steven = Jill + 18) ∧
  (Jill = 5)

theorem Jake_peaches : peaches_relation Jake Steven Jill → Jake = 17 := by
  sorry

end Jake_peaches_l1072_107271


namespace parallelogram_area_l1072_107234

theorem parallelogram_area (a b : ℝ) (theta : ℝ)
  (h1 : a = 10) (h2 : b = 20) (h3 : theta = 150) : a * b * Real.sin (theta * Real.pi / 180) = 100 * Real.sqrt 3 := by
  sorry

end parallelogram_area_l1072_107234


namespace star_three_five_l1072_107281

def star (x y : ℕ) := x^2 + 2 * x * y + y^2

theorem star_three_five : star 3 5 = 64 :=
by
  sorry

end star_three_five_l1072_107281


namespace x_squared_plus_y_squared_value_l1072_107255

theorem x_squared_plus_y_squared_value (x y : ℝ) (h : (x^2 + y^2 + 1) * (x^2 + y^2 + 2) = 6) : x^2 + y^2 = 1 :=
by
  sorry

end x_squared_plus_y_squared_value_l1072_107255


namespace nat_numbers_in_segment_l1072_107213

theorem nat_numbers_in_segment (a : ℕ → ℕ) (blue_index red_index : Set ℕ)
  (cond1 : ∀ i ∈ blue_index, i ≤ 200 → a (i - 1) = i)
  (cond2 : ∀ i ∈ red_index, i ≤ 200 → a (i - 1) = 201 - i) :
    ∀ i, 1 ≤ i ∧ i ≤ 100 → ∃ j, j < 100 ∧ a j = i := 
by
  sorry

end nat_numbers_in_segment_l1072_107213


namespace multiply_469111111_by_99999999_l1072_107293

theorem multiply_469111111_by_99999999 :
  469111111 * 99999999 = 46911111053088889 :=
sorry

end multiply_469111111_by_99999999_l1072_107293


namespace striped_nails_painted_l1072_107288

theorem striped_nails_painted (total_nails purple_nails blue_nails : ℕ) (h_total : total_nails = 20)
    (h_purple : purple_nails = 6) (h_blue : blue_nails = 8)
    (h_diff_percent : |(blue_nails:ℚ) / total_nails * 100 - 
    ((total_nails - purple_nails - blue_nails):ℚ) / total_nails * 100| = 10) :
    (total_nails - purple_nails - blue_nails) = 6 := 
by 
  sorry

end striped_nails_painted_l1072_107288


namespace area_of_WIN_sector_l1072_107242

theorem area_of_WIN_sector (r : ℝ) (p : ℝ) (A_circ : ℝ) (A_WIN : ℝ) 
    (h_r : r = 15) 
    (h_p : p = 1 / 3) 
    (h_A_circ : A_circ = π * r^2) 
    (h_A_WIN : A_WIN = p * A_circ) :
    A_WIN = 75 * π := 
sorry

end area_of_WIN_sector_l1072_107242


namespace reflect_and_shift_l1072_107250

def f : ℝ → ℝ := sorry  -- Assume f is some function from ℝ to ℝ

def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (6 - x)

theorem reflect_and_shift (f : ℝ → ℝ) (x : ℝ) : h f x = f (6 - x) :=
by
  -- provide the proof here
  sorry

end reflect_and_shift_l1072_107250


namespace maximum_area_of_right_angled_triangle_l1072_107294

noncomputable def max_area_right_angled_triangle (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : a + b + c = 48) : ℕ := 
  max (a * b / 2) 288

theorem maximum_area_of_right_angled_triangle (a b c : ℕ) 
  (h1 : a^2 + b^2 = c^2)    -- Pythagorean theorem
  (h2 : a + b + c = 48)     -- Perimeter condition
  (h3 : 0 < a)              -- Positive integer side length condition
  (h4 : 0 < b)              -- Positive integer side length condition
  (h5 : 0 < c)              -- Positive integer side length condition
  : max_area_right_angled_triangle a b c h1 h2 = 288 := 
sorry

end maximum_area_of_right_angled_triangle_l1072_107294


namespace min_max_value_of_expr_l1072_107289

theorem min_max_value_of_expr (p q r s : ℝ)
  (h1 : p + q + r + s = 10)
  (h2 : p^2 + q^2 + r^2 + s^2 = 20) :
  ∃ m M : ℝ, m = 2 ∧ M = 0 ∧ ∀ x, (x = 3 * (p^3 + q^3 + r^3 + s^3) - 2 * (p^4 + q^4 + r^4 + s^4)) → m ≤ x ∧ x ≤ M :=
sorry

end min_max_value_of_expr_l1072_107289


namespace fraction_of_remaining_prize_money_each_winner_receives_l1072_107290

-- Definitions based on conditions
def total_prize_money : ℕ := 2400
def first_winner_fraction : ℚ := 1 / 3
def each_following_winner_prize : ℕ := 160

-- Calculate the first winner's prize
def first_winner_prize : ℚ := first_winner_fraction * total_prize_money

-- Calculate the remaining prize money after the first winner
def remaining_prize_money : ℚ := total_prize_money - first_winner_prize

-- Calculate the fraction of the remaining prize money that each of the next ten winners will receive
def following_winner_fraction : ℚ := each_following_winner_prize / remaining_prize_money

-- Theorem statement
theorem fraction_of_remaining_prize_money_each_winner_receives :
  following_winner_fraction = 1 / 10 :=
sorry

end fraction_of_remaining_prize_money_each_winner_receives_l1072_107290


namespace merchant_marked_price_l1072_107223

theorem merchant_marked_price (L : ℝ) (x : ℝ) : 
  (L = 100) →
  (L - 0.3 * L = 70) →
  (0.75 * x - 70 = 0.225 * x) →
  x = 133.33 :=
by
  intro h1 h2 h3
  sorry

end merchant_marked_price_l1072_107223


namespace fraction_addition_l1072_107248

theorem fraction_addition (x : ℝ) (h : x + 1 ≠ 0) : (x / (x + 1) + 1 / (x + 1) = 1) :=
sorry

end fraction_addition_l1072_107248


namespace polynomial_at_3mnplus1_l1072_107261

noncomputable def polynomial_value (x : ℤ) : ℤ := x^2 + 4 * x + 6

theorem polynomial_at_3mnplus1 (m n : ℤ) (h₁ : 2 * m + n + 2 = m + 2 * n) (h₂ : m - n + 2 ≠ 0) :
  polynomial_value (3 * (m + n + 1)) = 3 := 
by 
  sorry

end polynomial_at_3mnplus1_l1072_107261


namespace proof_equivalent_problem_l1072_107227

noncomputable def polar_equation_curve : Prop :=
  ∀ (α : ℝ), 
    let x := 3 + 2 * Real.cos α;
    let y := 1 - 2 * Real.sin α;
    (x - 3) ^ 2 + (y - 1) ^ 2 - 4 = 0

noncomputable def polar_equation_line : Prop :=
  ∀ (θ ρ : ℝ), 
  (Real.sin θ - 2 * Real.cos θ = 1 / ρ) → (2 * (ρ * Real.cos θ) - (ρ * Real.sin θ) + 1 = 0)

noncomputable def distance_from_curve_to_line : Prop :=
  ∀ (α : ℝ), 
    let x := 3 + 2 * Real.cos α;
    let y := 1 - 2 * Real.sin α;
    ∃ d : ℝ, d = (|2 * x - y + 1| / Real.sqrt (2 ^ 2 + 1)) ∧
    d + 2 = (6 * Real.sqrt 5 / 5) + 2

theorem proof_equivalent_problem :
  polar_equation_curve ∧ polar_equation_line ∧ distance_from_curve_to_line :=
by
  constructor
  · exact sorry  -- polar_equation_curve proof
  · constructor
    · exact sorry  -- polar_equation_line proof
    · exact sorry  -- distance_from_curve_to_line proof

end proof_equivalent_problem_l1072_107227


namespace expression_value_l1072_107214

theorem expression_value : (5 - 2) / (2 + 1) = 1 := by
  sorry

end expression_value_l1072_107214


namespace no_solution_exists_l1072_107244

theorem no_solution_exists : ¬ ∃ (x : ℕ), (42 + x = 3 * (8 + x) ∧ 42 + x = 2 * (10 + x)) :=
by
  sorry

end no_solution_exists_l1072_107244


namespace michael_truck_meet_once_l1072_107249

-- Michael's walking speed.
def michael_speed := 4 -- feet per second

-- Distance between trash pails.
def pail_distance := 100 -- feet

-- Truck's speed.
def truck_speed := 8 -- feet per second

-- Time truck stops at each pail.
def truck_stop_time := 20 -- seconds

-- Prove how many times Michael and the truck will meet given the initial condition.
theorem michael_truck_meet_once :
  ∃ n : ℕ, michael_truck_meet_count == 1 :=
sorry

end michael_truck_meet_once_l1072_107249


namespace reduced_price_of_oil_l1072_107209

theorem reduced_price_of_oil (P R : ℝ) (h1: R = 0.75 * P) (h2: 600 / (0.75 * P) = 600 / P + 5) :
  R = 30 :=
by
  sorry

end reduced_price_of_oil_l1072_107209


namespace Carla_final_position_l1072_107291

-- Carla's initial position
def Carla_initial_position : ℤ × ℤ := (10, -10)

-- Function to calculate Carla's new position after each move
def Carla_move (pos : ℤ × ℤ) (direction : ℕ) (distance : ℤ) : ℤ × ℤ :=
  match direction % 4 with
  | 0 => (pos.1, pos.2 + distance)   -- North
  | 1 => (pos.1 + distance, pos.2)   -- East
  | 2 => (pos.1, pos.2 - distance)   -- South
  | 3 => (pos.1 - distance, pos.2)   -- West
  | _ => pos  -- This case will never happen due to the modulo operation

-- Recursive function to simulate Carla's journey
def Carla_journey : ℕ → ℤ × ℤ → ℤ × ℤ 
  | 0, pos => pos
  | n + 1, pos => 
    let next_pos := Carla_move pos n (2 + n / 2 * 2)
    Carla_journey n next_pos

-- Prove that after 100 moves, Carla's position is (-191, -10)
theorem Carla_final_position : Carla_journey 100 Carla_initial_position = (-191, -10) :=
sorry

end Carla_final_position_l1072_107291


namespace endangered_species_count_l1072_107206

section BirdsSanctuary

-- Define the given conditions
def pairs_per_species : ℕ := 7
def total_pairs : ℕ := 203

-- Define the result to be proved
theorem endangered_species_count : total_pairs / pairs_per_species = 29 := by
  sorry

end BirdsSanctuary

end endangered_species_count_l1072_107206


namespace sufficient_but_not_necessary_condition_l1072_107280

theorem sufficient_but_not_necessary_condition 
(a b : ℝ) : (b ≥ 0) → ((a + 1)^2 + b ≥ 0) ∧ (¬ (∀ a b, ((a + 1)^2 + b ≥ 0) → b ≥ 0)) :=
by sorry

end sufficient_but_not_necessary_condition_l1072_107280


namespace work_related_emails_count_l1072_107208

-- Definitions based on the identified conditions and the question
def total_emails : ℕ := 1200
def spam_percentage : ℕ := 27
def promotional_percentage : ℕ := 18
def social_percentage : ℕ := 15

-- The statement to prove, indicated the goal
theorem work_related_emails_count :
  (total_emails * (100 - spam_percentage - promotional_percentage - social_percentage)) / 100 = 480 :=
by
  sorry

end work_related_emails_count_l1072_107208


namespace problem1_problem2_problem3_problem4_l1072_107239

theorem problem1 : 6 + (-8) - (-5) = 3 := by
  sorry

theorem problem2 : (5 + 3/5) + (-(5 + 2/3)) + (4 + 2/5) + (-1/3) = 4 := by
  sorry

theorem problem3 : ((-1/2) + 1/6 - 1/4) * 12 = -7 := by
  sorry

theorem problem4 : -1^2022 + 27 * (-1/3)^2 - |(-5)| = -3 := by
  sorry

end problem1_problem2_problem3_problem4_l1072_107239


namespace total_marbles_count_l1072_107263

variable (r b g : ℝ)
variable (h1 : r = 1.4 * b) (h2 : g = 1.5 * r)

theorem total_marbles_count (r b g : ℝ) (h1 : r = 1.4 * b) (h2 : g = 1.5 * r) :
  r + b + g = 3.21 * r :=
by
  sorry

end total_marbles_count_l1072_107263


namespace number_of_integers_with_square_fraction_l1072_107247

theorem number_of_integers_with_square_fraction : 
  ∃! (S : Finset ℤ), (∀ (n : ℤ), n ∈ S ↔ ∃ (k : ℤ), (n = 15 * k^2) ∨ (15 - n = k^2)) ∧ S.card = 2 := 
sorry

end number_of_integers_with_square_fraction_l1072_107247


namespace loss_per_metre_eq_12_l1072_107221

-- Definitions based on the conditions
def totalMetres : ℕ := 200
def totalSellingPrice : ℕ := 12000
def costPricePerMetre : ℕ := 72

-- Theorem statement to prove the loss per metre of cloth
theorem loss_per_metre_eq_12 : (costPricePerMetre * totalMetres - totalSellingPrice) / totalMetres = 12 := 
by sorry

end loss_per_metre_eq_12_l1072_107221


namespace answer_is_correct_l1072_107202

-- We define the prime checking function
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

-- We define the set of candidates satisfying initial prime condition
def candidates : Set ℕ := {A | is_prime A ∧ A < 100 
                                   ∧ is_prime (A + 10) 
                                   ∧ is_prime (A - 20)
                                   ∧ is_prime (A + 30) 
                                   ∧ is_prime (A + 60) 
                                   ∧ is_prime (A + 70)}

-- The explicit set of valid answers
def valid_answers : Set ℕ := {37, 43, 79}

-- The statement that we need to prove
theorem answer_is_correct : candidates = valid_answers := 
sorry

end answer_is_correct_l1072_107202


namespace ratio_r_to_pq_l1072_107268

theorem ratio_r_to_pq (total : ℝ) (amount_r : ℝ) (amount_pq : ℝ) 
  (h1 : total = 9000) 
  (h2 : amount_r = 3600.0000000000005) 
  (h3 : amount_pq = total - amount_r) : 
  amount_r / amount_pq = 2 / 3 :=
by
  sorry

end ratio_r_to_pq_l1072_107268


namespace loss_recorded_as_negative_l1072_107265

-- Define the condition that a profit of 100 yuan is recorded as +100 yuan
def recorded_profit (p : ℤ) : Prop :=
  p = 100

-- Define the condition about how a profit is recorded
axiom profit_condition : recorded_profit 100

-- Define the function for recording profit or loss
def record (x : ℤ) : ℤ :=
  if x > 0 then x
  else -x

-- Theorem: If a profit of 100 yuan is recorded as +100 yuan, then a loss of 50 yuan is recorded as -50 yuan.
theorem loss_recorded_as_negative : ∀ x: ℤ, (x < 0) → record x = -x :=
by
  intros x h
  unfold record
  simp [h]
  -- sorry indicates the proof is not provided
  sorry

end loss_recorded_as_negative_l1072_107265


namespace find_shorter_piece_length_l1072_107233

noncomputable def shorter_piece_length (x : ℕ) : Prop :=
  x = 8

theorem find_shorter_piece_length : ∃ x : ℕ, (20 - x) > 0 ∧ 2 * x = (20 - x) + 4 ∧ shorter_piece_length x :=
by
  -- There exists an x that satisfies the conditions
  use 8
  -- Prove the conditions are met
  sorry

end find_shorter_piece_length_l1072_107233


namespace evaluate_expression_l1072_107260

theorem evaluate_expression : 10 * 0.2 * 5 * 0.1 + 5 = 6 :=
by
  -- transformed step-by-step mathematical proof goes here
  sorry

end evaluate_expression_l1072_107260


namespace coin_arrangements_l1072_107279

theorem coin_arrangements (n m : ℕ) (hp_pos : n = 5) (hq_pos : m = 5) :
  ∃ (num_arrangements : ℕ), num_arrangements = 8568 :=
by
  -- Note: 'sorry' is used to indicate here that the proof is omitted.
  sorry

end coin_arrangements_l1072_107279


namespace tangent_line_circle_l1072_107225

theorem tangent_line_circle (k : ℝ) (h1 : k = Real.sqrt 3) (h2 : ∃ (x y : ℝ), y = k * x + 2 ∧ x^2 + y^2 = 1) :
  (k = Real.sqrt 3 → (∃ (x y : ℝ), y = k * x + 2 ∧ x^2 + y^2 = 1)) ∧ (¬ (∀ (k : ℝ), (∃ (x y : ℝ), y = k * x + 2 ∧ x^2 + y^2 = 1) → k = Real.sqrt 3)) :=
  sorry

end tangent_line_circle_l1072_107225


namespace isosceles_triangle_area_of_triangle_l1072_107287

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions
axiom triangle_sides (a b c : ℝ) (A B C : ℝ) : c = 2
axiom cosine_condition (a b c : ℝ) (A B C : ℝ) : b^2 - 2 * b * c * Real.cos A = a^2 - 2 * a * c * Real.cos B

-- Questions
theorem isosceles_triangle (a b c : ℝ) (A B C : ℝ)
  (h1 : c = 2) 
  (h2 : b^2 - 2 * b * c * Real.cos A = a^2 - 2 * a * c * Real.cos B) :
  a = b :=
sorry

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h1 : c = 2) 
  (h2 : b^2 - 2 * b * c * Real.cos A = a^2 - 2 * a * c * Real.cos B)
  (h3 : 7 * Real.cos B = 2 * Real.cos C) 
  (h4 : a = b) :
  ∃ S : ℝ, S = Real.sqrt 15 :=
sorry

end isosceles_triangle_area_of_triangle_l1072_107287


namespace maximize_perimeter_OIH_l1072_107205

/-- In triangle ABC, given certain angles and side lengths, prove that
    angle ABC = 70° maximizes the perimeter of triangle OIH, where O, I,
    and H are the circumcenter, incenter, and orthocenter of triangle ABC. -/
theorem maximize_perimeter_OIH 
  (A : ℝ) (B : ℝ) (C : ℝ)
  (BC : ℝ) (AB : ℝ) (AC : ℝ)
  (BOC : ℝ) (BIC : ℝ) (BHC : ℝ) :
  A = 75 ∧ BC = 2 ∧ AB ≥ AC ∧
  BOC = 150 ∧ BIC = 127.5 ∧ BHC = 105 → 
  B = 70 :=
by
  sorry

end maximize_perimeter_OIH_l1072_107205


namespace values_of_d_divisible_by_13_l1072_107236

def base8to10 (d : ℕ) : ℕ := 3 * 8^3 + d * 8^2 + d * 8 + 7

theorem values_of_d_divisible_by_13 (d : ℕ) (h : d ≥ 0 ∧ d < 8) :
  (1543 + 72 * d) % 13 = 0 ↔ d = 1 ∨ d = 2 :=
by sorry

end values_of_d_divisible_by_13_l1072_107236


namespace hens_count_l1072_107237

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 140) : H = 26 := by
  sorry

end hens_count_l1072_107237


namespace maximum_interval_length_l1072_107211

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem maximum_interval_length 
  (m n : ℕ)
  (h1 : 0 < m)
  (h2 : m < n)
  (h3 : ∃ k : ℕ, ∀ i : ℕ, 0 ≤ i → i < k → ¬ is_multiple_of (m + i) 2000 ∧ (m + i) % 2021 = 0):
  n - m = 1999 :=
sorry

end maximum_interval_length_l1072_107211


namespace count_valid_arrays_l1072_107203

-- Define the integer array condition
def valid_array (x1 x2 x3 x4 : ℕ) : Prop :=
  0 < x1 ∧ x1 ≤ x2 ∧ x2 < x3 ∧ x3 ≤ x4 ∧ x4 < 7

-- State the theorem that proves the number of valid arrays is 70
theorem count_valid_arrays : ∃ (n : ℕ), n = 70 ∧ 
    ∀ (x1 x2 x3 x4 : ℕ), valid_array x1 x2 x3 x4 -> ∃ (n : ℕ), n = 70 :=
by
  -- The proof can be filled in later
  sorry

end count_valid_arrays_l1072_107203


namespace largest_smallest_divisible_by_99_l1072_107253

-- Definitions for distinct digits 3, 7, 9
def largest_number (x y z : Nat) : Nat := 100 * x + 10 * y + z
def smallest_number (x y z : Nat) : Nat := 100 * z + 10 * y + x

-- Proof problem statement
theorem largest_smallest_divisible_by_99 
  (a b c : Nat) (h : a > b ∧ b > c ∧ c > 0) : 
  ∃ (x y z : Nat), 
    (x = 9 ∧ y = 7 ∧ z = 3 ∧ largest_number x y z = 973 ∧ smallest_number x y z = 379) ∧
    99 ∣ (largest_number a b c - smallest_number a b c) :=
by
  sorry

end largest_smallest_divisible_by_99_l1072_107253


namespace calculate_flat_rate_shipping_l1072_107286

noncomputable def flat_rate_shipping : ℝ :=
  17.00

theorem calculate_flat_rate_shipping
  (price_per_shirt : ℝ)
  (num_shirts : ℤ)
  (price_pack_socks : ℝ)
  (num_packs_socks : ℤ)
  (price_per_short : ℝ)
  (num_shorts : ℤ)
  (price_swim_trunks : ℝ)
  (num_swim_trunks : ℤ)
  (total_bill : ℝ)
  (total_items_cost : ℝ)
  (shipping_cost : ℝ) :
  price_per_shirt * num_shirts + 
  price_pack_socks * num_packs_socks + 
  price_per_short * num_shorts +
  price_swim_trunks * num_swim_trunks = total_items_cost →
  total_bill - total_items_cost = shipping_cost →
  total_items_cost > 50 → 
  0.20 * total_items_cost ≠ shipping_cost →
  flat_rate_shipping = 17.00 := 
sorry

end calculate_flat_rate_shipping_l1072_107286


namespace minhyuk_needs_slices_l1072_107238

-- Definitions of Yeongchan and Minhyuk's apple division
def yeongchan_portion : ℚ := 1 / 3
def minhyuk_slices : ℚ := 1 / 12

-- Statement to prove
theorem minhyuk_needs_slices (x : ℕ) : yeongchan_portion = x * minhyuk_slices → x = 4 :=
by
  sorry

end minhyuk_needs_slices_l1072_107238


namespace find_k_l1072_107229

theorem find_k (k : ℝ) :
  ∃ k, ∀ x : ℝ, (3 * x^3 + k * x^2 - 8 * x + 52) % (3 * x + 4) = 7 :=
by
-- The proof would go here, we insert sorry to acknowledge the missing proof
sorry

end find_k_l1072_107229


namespace perpendicular_condition_parallel_condition_parallel_opposite_direction_l1072_107240

variables (a b : ℝ × ℝ) (k : ℝ)

-- Define the vectors
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (-3, 2)

-- Define the given expressions
def expression1 (k : ℝ) : ℝ × ℝ := (k * vec_a.1 + vec_b.1, k * vec_a.2 + vec_b.2)
def expression2 : ℝ × ℝ := (vec_a.1 - 3 * vec_b.1, vec_a.2 - 3 * vec_b.2)

-- Dot product function
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Perpendicular condition
theorem perpendicular_condition : (k : ℝ) → dot_product (expression1 k) expression2 = 0 → k = 19 :=
by sorry

-- Parallel and opposite condition
theorem parallel_condition : (k : ℝ) → (∃ m : ℝ, expression1 k = m • expression2) → k = -1 / 3 :=
by sorry

noncomputable def m (k : ℝ) : ℝ × ℝ := 
  let ex1 := expression1 k
  let ex2 := expression2
  (ex2.1 / ex1.1, ex2.2 / ex1.2)

theorem parallel_opposite_direction : (k : ℝ) → expression1 k = -1 / 3 • expression2 → k = -1 / 3 :=
by sorry

end perpendicular_condition_parallel_condition_parallel_opposite_direction_l1072_107240


namespace stratified_sampling_l1072_107257

-- We are defining the data given in the problem
def numStudents : ℕ := 50
def numFemales : ℕ := 20
def sampledFemales : ℕ := 4
def genderRatio := (numFemales : ℚ) / (numStudents : ℚ)

-- The theorem stating the given problem and its conclusion
theorem stratified_sampling : ∀ (n : ℕ), (sampledFemales : ℚ) / (n : ℚ) = genderRatio → n = 10 :=
by
  intro n
  intro h
  sorry

end stratified_sampling_l1072_107257


namespace a2_plus_b2_minus_abc_is_perfect_square_l1072_107275

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem a2_plus_b2_minus_abc_is_perfect_square {a b c : ℕ} (h : 0 < a^2 + b^2 - a * b * c ∧ a^2 + b^2 - a * b * c ≤ c) :
  is_perfect_square (a^2 + b^2 - a * b * c) :=
by
  sorry

end a2_plus_b2_minus_abc_is_perfect_square_l1072_107275


namespace total_area_is_82_l1072_107215

/-- Definition of the lengths of each segment as conditions -/
def length1 : ℤ := 7
def length2 : ℤ := 4
def length3 : ℤ := 5
def length4 : ℤ := 3
def length5 : ℤ := 2
def length6 : ℤ := 1

/-- Rectangle areas based on the given lengths -/
def area_A : ℤ := length1 * length2 -- 7 * 4
def area_B : ℤ := length3 * length2 -- 5 * 4
def area_C : ℤ := length1 * length4 -- 7 * 3
def area_D : ℤ := length3 * length5 -- 5 * 2
def area_E : ℤ := length4 * length6 -- 3 * 1

/-- The total area is the sum of all rectangle areas -/
def total_area : ℤ := area_A + area_B + area_C + area_D + area_E

/-- Theorem: The total area is 82 square units -/
theorem total_area_is_82 : total_area = 82 :=
by
  -- Proof left as an exercise
  sorry

end total_area_is_82_l1072_107215


namespace night_rides_total_l1072_107258

-- Definitions corresponding to the conditions in the problem
def total_ferris_wheel_rides : Nat := 13
def total_roller_coaster_rides : Nat := 9
def ferris_wheel_day_rides : Nat := 7
def roller_coaster_day_rides : Nat := 4

-- The total night rides proof problem
theorem night_rides_total :
  let ferris_wheel_night_rides := total_ferris_wheel_rides - ferris_wheel_day_rides
  let roller_coaster_night_rides := total_roller_coaster_rides - roller_coaster_day_rides
  ferris_wheel_night_rides + roller_coaster_night_rides = 11 :=
by
  -- Proof skipped
  sorry

end night_rides_total_l1072_107258


namespace find_h_in_standard_form_l1072_107270

-- The expression to be converted
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x - 24

-- The standard form with given h value
def standard_form (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

-- The theorem statement
theorem find_h_in_standard_form :
  ∃ k : ℝ, ∀ x : ℝ, quadratic_expr x = standard_form 3 (-1.5) k x :=
by
  let a := 3
  let h := -1.5
  existsi (-30.75)
  intro x
  sorry

end find_h_in_standard_form_l1072_107270


namespace unknown_number_is_five_l1072_107243

theorem unknown_number_is_five (x : ℕ) (h : 64 + x * 12 / (180 / 3) = 65) : x = 5 := 
by 
  sorry

end unknown_number_is_five_l1072_107243


namespace number_of_polynomials_is_seven_l1072_107273

-- Definitions of what constitutes a polynomial
def is_polynomial (expr : String) : Bool :=
  match expr with
  | "3/4*x^2" => true
  | "3ab" => true
  | "x+5" => true
  | "y/5x" => false
  | "-1" => true
  | "y/3" => true
  | "a^2-b^2" => true
  | "a" => true
  | _ => false

-- Given set of algebraic expressions
def expressions : List String := 
  ["3/4*x^2", "3ab", "x+5", "y/5x", "-1", "y/3", "a^2-b^2", "a"]

-- Count the number of polynomials in the given expressions
def count_polynomials (exprs : List String) : Nat :=
  exprs.foldr (fun expr count => if is_polynomial expr then count + 1 else count) 0

theorem number_of_polynomials_is_seven : count_polynomials expressions = 7 :=
  by
    sorry

end number_of_polynomials_is_seven_l1072_107273


namespace average_percentage_of_10_students_l1072_107278

theorem average_percentage_of_10_students 
  (avg_15_students : ℕ := 80)
  (n_15_students : ℕ := 15)
  (total_students : ℕ := 25)
  (overall_avg : ℕ := 84) : 
  ∃ (x : ℕ), ((n_15_students * avg_15_students + 10 * x) / total_students = overall_avg) → x = 90 := 
sorry

end average_percentage_of_10_students_l1072_107278


namespace parking_space_area_l1072_107298

theorem parking_space_area
  (L : ℕ) (W : ℕ)
  (hL : L = 9)
  (hSum : 2 * W + L = 37) : L * W = 126 := 
by
  sorry

end parking_space_area_l1072_107298


namespace find_c_plus_inv_b_l1072_107204

theorem find_c_plus_inv_b (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h1 : a * b * c = 1) (h2 : a + 1 / c = 7) (h3 : b + 1 / a = 35) :
  c + 1 / b = 11 / 61 :=
by
  sorry

end find_c_plus_inv_b_l1072_107204


namespace total_points_correct_l1072_107259

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

end total_points_correct_l1072_107259


namespace equivalence_of_sum_cubed_expression_l1072_107241

theorem equivalence_of_sum_cubed_expression (a b : ℝ) 
  (h₁ : a + b = 5) (h₂ : a * b = -14) : a^3 + a^2 * b + a * b^2 + b^3 = 265 :=
sorry

end equivalence_of_sum_cubed_expression_l1072_107241


namespace speed_maintained_l1072_107264

-- Given conditions:
def distance : ℕ := 324
def original_time : ℕ := 6
def new_time : ℕ := (3 * original_time) / 2

-- Correct answer:
def required_speed : ℕ := 36

-- Lean 4 statement to prove the equivalence:
theorem speed_maintained :
  (distance / new_time) = required_speed :=
sorry

end speed_maintained_l1072_107264


namespace fraction_dropped_l1072_107200

theorem fraction_dropped (f : ℝ) 
  (h1 : 0 ≤ f ∧ f ≤ 1) 
  (initial_passengers : ℝ) 
  (final_passenger_count : ℝ)
  (first_pickup : ℝ)
  (second_pickup : ℝ) 
  (first_drop_factor : ℝ)
  (second_drop_factor : ℕ):
  initial_passengers = 270 →
  final_passenger_count = 242 →
  first_pickup = 280 →
  second_pickup = 12 →
  first_drop_factor = f →
  second_drop_factor = 2 →
  ((initial_passengers - initial_passengers * first_drop_factor) + first_pickup) / second_drop_factor + second_pickup = final_passenger_count →
  f = 1 / 3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end fraction_dropped_l1072_107200


namespace sum_of_digits_divisible_by_7_l1072_107226

theorem sum_of_digits_divisible_by_7
  (a b : ℕ)
  (h_three_digit : 100 * a + 11 * b ≥ 100 ∧ 100 * a + 11 * b < 1000)
  (h_last_two_digits_equal : true)
  (h_divisible_by_7 : (100 * a + 11 * b) % 7 = 0) :
  (a + 2 * b) % 7 = 0 :=
sorry

end sum_of_digits_divisible_by_7_l1072_107226


namespace sin_135_eq_sqrt2_over_2_l1072_107230

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l1072_107230


namespace raisin_cookies_difference_l1072_107235

-- Definitions based on conditions:
def raisin_cookies_baked_yesterday : ℕ := 300
def raisin_cookies_baked_today : ℕ := 280

-- Proof statement:
theorem raisin_cookies_difference : raisin_cookies_baked_yesterday - raisin_cookies_baked_today = 20 := 
by
  sorry

end raisin_cookies_difference_l1072_107235


namespace tv_station_ads_l1072_107295

theorem tv_station_ads (n m : ℕ) :
  n > 1 → 
  ∃ (an : ℕ → ℕ), 
  (an 0 = m) ∧ 
  (∀ k, 1 ≤ k ∧ k < n → an k = an (k - 1) - (k + (1 / 8) * (an (k - 1) - k))) ∧
  an n = 0 →
  (n = 7 ∧ m = 49) :=
by
  intro h
  exists sorry
  sorry

-- The proof steps are omitted

end tv_station_ads_l1072_107295


namespace product_of_integers_l1072_107274

theorem product_of_integers (x y : ℕ) (h1 : x + y = 26) (h2 : x^2 - y^2 = 52) : x * y = 168 := by
  sorry

end product_of_integers_l1072_107274


namespace bc_eq_one_area_of_triangle_l1072_107216

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

end bc_eq_one_area_of_triangle_l1072_107216


namespace intersection_A_B_l1072_107218

noncomputable def A : Set ℝ := { x | abs (x - 1) < 2 }
noncomputable def B : Set ℝ := { x | x^2 + 3 * x - 4 < 0 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | -1 < x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l1072_107218


namespace trains_distance_apart_l1072_107201

-- Define the initial conditions
def cattle_train_speed : ℝ := 56
def diesel_train_speed : ℝ := cattle_train_speed - 33
def cattle_train_time : ℝ := 6 + 12
def diesel_train_time : ℝ := 12

-- Calculate distances
def cattle_train_distance : ℝ := cattle_train_speed * cattle_train_time
def diesel_train_distance : ℝ := diesel_train_speed * diesel_train_time

-- Define total distance apart
def distance_apart : ℝ := cattle_train_distance + diesel_train_distance

-- The theorem to prove
theorem trains_distance_apart :
  distance_apart = 1284 :=
by
  -- Skip the proof
  sorry

end trains_distance_apart_l1072_107201


namespace intersection_with_complement_l1072_107267

-- Definitions for the universal set and set A
def U : Set ℝ := Set.univ

def A : Set ℝ := { -1, 0, 1 }

-- Definition for set B using the given condition
def B : Set ℝ := { x : ℝ | (x - 2) / (x + 1) > 0 }

-- Definition for the complement of B
def B_complement : Set ℝ := { x : ℝ | -1 <= x ∧ x <= 0 }

-- Theorem stating the intersection of A and the complement of B equals {-1, 0, 1}
theorem intersection_with_complement : 
  A ∩ B_complement = { -1, 0, 1 } :=
by
  sorry

end intersection_with_complement_l1072_107267


namespace color_dots_l1072_107217

-- Define the vertices and the edges of the graph representing the figure
inductive Color : Type
| red : Color
| white : Color
| blue : Color

structure Dot :=
  (color : Color)

structure Edge :=
  (u : Dot)
  (v : Dot)

def valid_coloring (dots : List Dot) (edges : List Edge) : Prop :=
  ∀ e ∈ edges, e.u.color ≠ e.v.color

def count_colorings : Nat :=
  6 * 2

theorem color_dots (dots : List Dot) (edges : List Edge)
  (h1 : ∀ d ∈ dots, d.color = Color.red ∨ d.color = Color.white ∨ d.color = Color.blue)
  (h2 : valid_coloring dots edges) :
  count_colorings = 12 :=
by
  sorry

end color_dots_l1072_107217


namespace complete_square_identity_l1072_107283

theorem complete_square_identity (x : ℝ) : ∃ (d e : ℤ), (x^2 - 10 * x + 13 = 0 → (x + d)^2 = e ∧ d + e = 7) :=
sorry

end complete_square_identity_l1072_107283


namespace relationship_between_x_y_l1072_107296

theorem relationship_between_x_y (x y m : ℝ) (h₁ : x + m = 4) (h₂ : y - 5 = m) : x + y = 9 := 
sorry

end relationship_between_x_y_l1072_107296


namespace problem_statement_l1072_107219

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, ∃ d : ℤ, a (n + 1) = a n + d

noncomputable def given_conditions (a : ℕ → ℤ) : Prop :=
a 2 = 2 ∧ a 3 = 4

theorem problem_statement (a : ℕ → ℤ) (h1 : given_conditions a) (h2 : arithmetic_sequence a) :
  a 10 = 18 := by
  sorry

end problem_statement_l1072_107219


namespace c_share_l1072_107222

theorem c_share (A B C D : ℝ) 
    (h1 : A = 1/2 * B) 
    (h2 : B = 1/2 * C) 
    (h3 : D = 1/4 * 392) 
    (h4 : A + B + C + D = 392) : 
    C = 168 := 
by 
    sorry

end c_share_l1072_107222


namespace main_theorem_l1072_107292

-- The condition
def condition (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (2^x - 1) = 4^x - 1

-- The property we need to prove
def proves (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, -1 ≤ x → f x = x^2 + 2*x

-- The main theorem connecting the condition to the desired property
theorem main_theorem (f : ℝ → ℝ) (h : condition f) : proves f :=
sorry

end main_theorem_l1072_107292


namespace coin_flip_sequences_l1072_107299

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l1072_107299


namespace fraction_numerator_l1072_107256

theorem fraction_numerator (x : ℤ) (h₁ : 2 * x + 11 ≠ 0) (h₂ : (x : ℚ) / (2 * x + 11) = 3 / 4) : x = -33 / 2 :=
by
  sorry

end fraction_numerator_l1072_107256


namespace small_drinking_glasses_count_l1072_107207

theorem small_drinking_glasses_count :
  ∀ (large_jelly_beans_per_large_glass small_jelly_beans_per_small_glass total_jelly_beans : ℕ),
  (large_jelly_beans_per_large_glass = 50) →
  (small_jelly_beans_per_small_glass = large_jelly_beans_per_large_glass / 2) →
  (5 * large_jelly_beans_per_large_glass + n * small_jelly_beans_per_small_glass = total_jelly_beans) →
  (total_jelly_beans = 325) →
  n = 3 := by
  sorry

end small_drinking_glasses_count_l1072_107207


namespace radius_B_l1072_107297

noncomputable def radius_A := 2
noncomputable def radius_D := 4

theorem radius_B (r_B : ℝ) (x y : ℝ) 
  (h1 : (2 : ℝ) + y = x + (x^2 / 4)) 
  (h2 : y = 2 - (x^2 / 8)) 
  (h3 : x = (4: ℝ) / 3) 
  (h4 : y = x + (x^2 / 4)) : r_B = 20 / 9 :=
sorry

end radius_B_l1072_107297
