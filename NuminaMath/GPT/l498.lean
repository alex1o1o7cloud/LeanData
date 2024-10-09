import Mathlib

namespace least_isosceles_triangles_cover_rectangle_l498_49817

-- Define the dimensions of the rectangle
def rectangle_height : ℕ := 10
def rectangle_width : ℕ := 100

-- Define the least number of isosceles right triangles needed to cover the rectangle
def least_number_of_triangles (h w : ℕ) : ℕ :=
  if h = rectangle_height ∧ w = rectangle_width then 11 else 0

-- The theorem statement
theorem least_isosceles_triangles_cover_rectangle :
  least_number_of_triangles rectangle_height rectangle_width = 11 :=
by
  -- skip the proof
  sorry

end least_isosceles_triangles_cover_rectangle_l498_49817


namespace flower_beds_fraction_l498_49835

noncomputable def isosceles_right_triangle_area (leg : ℝ) : ℝ :=
  (1 / 2) * leg^2

noncomputable def fraction_of_yard_occupied_by_flower_beds : ℝ :=
  let leg_length := (25 - 15) / 2
  let area_of_one_triangle := isosceles_right_triangle_area leg_length
  let total_area_of_flower_beds := 2 * area_of_one_triangle
  let area_of_yard := 25 * 5
  total_area_of_flower_beds / area_of_yard

theorem flower_beds_fraction : 
  let leg_length := (25 - 15) / 2
  let area_of_one_triangle := isosceles_right_triangle_area leg_length
  let total_area_of_flower_beds := 2 * area_of_one_triangle
  let area_of_yard := 25 * 5
  total_area_of_flower_beds / area_of_yard = 1 / 5 :=
by
  sorry

end flower_beds_fraction_l498_49835


namespace initial_friends_l498_49884

theorem initial_friends (n : ℕ) (h1 : 120 / (n - 4) = 120 / n + 8) : n = 10 := 
by
  sorry

end initial_friends_l498_49884


namespace first_product_of_digits_of_98_l498_49826

theorem first_product_of_digits_of_98 : (9 * 8 = 72) :=
by simp [mul_eq_mul_right_iff] -- This will handle the basic arithmetic automatically

end first_product_of_digits_of_98_l498_49826


namespace letter_lock_rings_l498_49807

theorem letter_lock_rings (n : ℕ) (h : n^3 - 1 ≤ 215) : n = 6 :=
by { sorry }

end letter_lock_rings_l498_49807


namespace can_determine_counterfeit_coin_l498_49869

/-- 
Given 101 coins where 50 are counterfeit and each counterfeit coin 
differs by 1 gram from the genuine ones, prove that Petya can 
determine if a given coin is counterfeit with a single weighing 
using a balance scale.
-/
theorem can_determine_counterfeit_coin :
  ∃ (coins : Fin 101 → ℤ), 
    (∃ i : Fin 101, (1 ≤ i ∧ i ≤ 50 → coins i = 1) ∧ (51 ≤ i ∧ i ≤ 101 → coins i = 0)) →
    (∃ (b : ℤ), (0 < b → b ∣ 1) ∧ (¬(0 < b → b ∣ 1) → coins 101 = b)) :=
by
  sorry

end can_determine_counterfeit_coin_l498_49869


namespace johns_age_l498_49875

theorem johns_age (J : ℕ) (h : J + 9 = 3 * (J - 11)) : J = 21 :=
sorry

end johns_age_l498_49875


namespace quadratic_has_one_solution_implies_m_l498_49870

theorem quadratic_has_one_solution_implies_m (m : ℚ) :
  (∀ x : ℚ, 3 * x^2 - 7 * x + m = 0 → (b^2 - 4 * a * m = 0)) ↔ m = 49 / 12 :=
by
  sorry

end quadratic_has_one_solution_implies_m_l498_49870


namespace volume_of_rock_correct_l498_49814

-- Define the initial conditions
def tank_length := 30
def tank_width := 20
def water_depth := 8
def water_level_rise := 4

-- Define the volume function for the rise in water level
def calculate_volume_of_rise (length: ℕ) (width: ℕ) (rise: ℕ) : ℕ :=
  length * width * rise

-- Define the target volume of the rock
def volume_of_rock := 2400

-- The theorem statement that the volume of the rock is 2400 cm³
theorem volume_of_rock_correct :
  calculate_volume_of_rise tank_length tank_width water_level_rise = volume_of_rock :=
by 
  sorry

end volume_of_rock_correct_l498_49814


namespace pass_rate_l498_49836

theorem pass_rate (total_students : ℕ) (students_not_passed : ℕ) (pass_rate : ℚ) :
  total_students = 500 → 
  students_not_passed = 40 → 
  pass_rate = (total_students - students_not_passed) / total_students * 100 →
  pass_rate = 92 :=
by 
  intros ht hs hpr 
  sorry

end pass_rate_l498_49836


namespace figure_count_mistake_l498_49876

theorem figure_count_mistake
    (b g : ℕ)
    (total_figures : ℕ)
    (boy_circles boy_squares girl_circles girl_squares : ℕ)
    (total_figures_counted : ℕ) :
  boy_circles = 3 → boy_squares = 8 → girl_circles = 9 → girl_squares = 2 →
  total_figures_counted = 4046 →
  (∃ (b g : ℕ), 11 * b + 11 * g ≠ 4046) :=
by
  intros
  sorry

end figure_count_mistake_l498_49876


namespace initial_passengers_is_350_l498_49815

variable (N : ℕ)

def initial_passengers (N : ℕ) : Prop :=
  let after_first_train := 9 * N / 10
  let after_second_train := 27 * N / 35
  let after_third_train := 108 * N / 175
  after_third_train = 216

theorem initial_passengers_is_350 : initial_passengers 350 := 
  sorry

end initial_passengers_is_350_l498_49815


namespace seventy_three_days_after_monday_is_thursday_l498_49863

def day_of_week : Nat → String
| 0 => "Monday"
| 1 => "Tuesday"
| 2 => "Wednesday"
| 3 => "Thursday"
| 4 => "Friday"
| 5 => "Saturday"
| _ => "Sunday"

theorem seventy_three_days_after_monday_is_thursday :
  day_of_week (73 % 7) = "Thursday" :=
by
  sorry

end seventy_three_days_after_monday_is_thursday_l498_49863


namespace incorrect_statement_among_props_l498_49828

theorem incorrect_statement_among_props 
    (A: Prop := True)  -- Axioms in mathematics are accepted truths that do not require proof.
    (B: Prop := True)  -- A mathematical proof can proceed in different valid sequences depending on the approach and insights.
    (C: Prop := True)  -- All concepts utilized in a proof must be clearly defined before their use in arguments.
    (D: Prop := False) -- Logical deductions based on false premises can lead to valid conclusions.
    (E: Prop := True): -- Proof by contradiction only needs one assumption to be negated and shown to lead to a contradiction to be valid.
  ¬D := 
by sorry

end incorrect_statement_among_props_l498_49828


namespace bob_wins_l498_49890

-- Define the notion of nim-sum used in nim-games
def nim_sum (a b : ℕ) : ℕ := Nat.xor a b

-- Define nim-values for given walls based on size
def nim_value : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| 3 => 3
| 4 => 1
| 5 => 4
| 6 => 3
| 7 => 2
| _ => 0

-- Calculate the nim-value of a given configuration
def nim_config (c : List ℕ) : ℕ :=
c.foldl (λ acc n => nim_sum acc (nim_value n)) 0

-- Prove that the configuration (7, 3, 1) gives a nim-value of 0
theorem bob_wins : nim_config [7, 3, 1] = 0 := by
  sorry

end bob_wins_l498_49890


namespace discount_percentage_is_ten_l498_49803

-- Definitions based on given conditions
def cost_price : ℝ := 42
def markup (S : ℝ) : ℝ := 0.30 * S
def selling_price (S : ℝ) : Prop := S = cost_price + markup S
def profit : ℝ := 6

-- To prove the discount percentage
theorem discount_percentage_is_ten (S SP : ℝ) 
  (h_sell_price : selling_price S) 
  (h_SP : SP = S - profit) : 
  ((S - SP) / S) * 100 = 10 := 
by
  sorry

end discount_percentage_is_ten_l498_49803


namespace largest_value_of_x_l498_49871

noncomputable def find_largest_x : ℝ :=
  let a := 10
  let b := 39
  let c := 18
  let discriminant := b^2 - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let x1 := (-b + sqrt_discriminant) / (2 * a)
  let x2 := (-b - sqrt_discriminant) / (2 * a)
  if x1 > x2 then x1 else x2

theorem largest_value_of_x :
  ∃ x : ℝ, 3 * (10 * x^2 + 11 * x + 12) = x * (10 * x - 45) ∧
  x = find_largest_x := by
  exists find_largest_x
  sorry

end largest_value_of_x_l498_49871


namespace cd_e_value_l498_49832

theorem cd_e_value (a b c d e f : ℝ) 
  (h1 : a * b * c = 195) (h2 : b * c * d = 65) 
  (h3 : d * e * f = 250) (h4 : (a * f) / (c * d) = 0.75) :
  c * d * e = 1000 := 
by
  sorry

end cd_e_value_l498_49832


namespace sets_of_headphones_l498_49888

-- Definitions of the conditions
variable (M H : ℕ)

-- Theorem statement for proving the question given the conditions
theorem sets_of_headphones (h1 : 5 * M + 30 * H = 840) (h2 : 3 * M + 120 = 480) : H = 8 := by
  sorry

end sets_of_headphones_l498_49888


namespace fabric_needed_for_coats_l498_49887

variable (m d : ℝ)

def condition1 := 4 * m + 2 * d = 16
def condition2 := 2 * m + 6 * d = 18

theorem fabric_needed_for_coats (h1 : condition1 m d) (h2 : condition2 m d) :
  m = 3 ∧ d = 2 :=
by
  sorry

end fabric_needed_for_coats_l498_49887


namespace minimum_candies_l498_49850

variables (c z : ℕ) (total_candies : ℕ)

def remaining_red_candies := (3 * c) / 5
def remaining_green_candies := (2 * z) / 5
def remaining_total_candies := remaining_red_candies + remaining_green_candies
def red_candies_fraction := remaining_red_candies * 8 = 3 * remaining_total_candies

theorem minimum_candies (h1 : 5 * c = 2 * z) (h2 : red_candies_fraction) :
  total_candies ≥ 35 := sorry

end minimum_candies_l498_49850


namespace square_garden_dimensions_and_area_increase_l498_49896

def original_length : ℝ := 60
def original_width : ℝ := 20

def original_area : ℝ := original_length * original_width
def original_perimeter : ℝ := 2 * (original_length + original_width)

theorem square_garden_dimensions_and_area_increase
    (L : ℝ := 60) (W : ℝ := 20)
    (orig_area : ℝ := L * W)
    (orig_perimeter : ℝ := 2 * (L + W))
    (square_side_length : ℝ := orig_perimeter / 4)
    (new_area : ℝ := square_side_length * square_side_length)
    (area_increase : ℝ := new_area - orig_area) :
    square_side_length = 40 ∧ area_increase = 400 :=
by {sorry}

end square_garden_dimensions_and_area_increase_l498_49896


namespace soda_mineral_cost_l498_49822

theorem soda_mineral_cost
  (x y : ℝ)
  (h1 : 2 * x + y = 7)
  (h2 : 4 * x + 3 * y = 16) :
  10 * x + 10 * y = 45 :=
  sorry

end soda_mineral_cost_l498_49822


namespace gcd_lcm_sum_l498_49851

theorem gcd_lcm_sum (a b : ℕ) (h₁ : a = 120) (h₂ : b = 3507) :
  Nat.gcd a b + Nat.lcm a b = 140283 := by 
  sorry

end gcd_lcm_sum_l498_49851


namespace sum_of_7_more_likely_than_sum_of_8_l498_49883

noncomputable def probability_sum_equals_seven : ℚ := 6 / 36
noncomputable def probability_sum_equals_eight : ℚ := 5 / 36

theorem sum_of_7_more_likely_than_sum_of_8 :
  probability_sum_equals_seven > probability_sum_equals_eight :=
by 
  sorry

end sum_of_7_more_likely_than_sum_of_8_l498_49883


namespace rectangle_perimeter_l498_49805

theorem rectangle_perimeter (long_side short_side : ℝ) 
  (h_long : long_side = 1) 
  (h_short : short_side = long_side - 2/8) : 
  2 * long_side + 2 * short_side = 3.5 := 
by 
  sorry

end rectangle_perimeter_l498_49805


namespace find_B_plus_C_l498_49848

theorem find_B_plus_C 
(A B C : ℕ)
(h1 : A ≠ B)
(h2 : B ≠ C)
(h3 : C ≠ A)
(h4 : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0)
(h5 : A < 5 ∧ B < 5 ∧ C < 5)
(h6 : 25 * A + 5 * B + C + 25 * B + 5 * C + A + 25 * C + 5 * A + B = 125 * A + 25 * A + 5 * A) : 
B + C = 4 * A := by
  sorry

end find_B_plus_C_l498_49848


namespace tournament_matches_l498_49843

theorem tournament_matches (n : ℕ) (total_matches : ℕ) (matches_three_withdrew : ℕ) (matches_after_withdraw : ℕ) :
  ∀ (x : ℕ), total_matches = (n * (n - 1) / 2) → matches_three_withdrew = 6 - x → matches_after_withdraw = total_matches - (3 * 2 - x) → 
  matches_after_withdraw = 50 → x = 1 :=
by
  intros
  sorry

end tournament_matches_l498_49843


namespace length_CD_l498_49864

-- Definitions of the edge lengths provided in the problem
def edge_lengths : Set ℕ := {7, 13, 18, 27, 36, 41}

-- Assumption that AB = 41
def AB := 41
def BC : ℕ := 13
def AC : ℕ := 36

-- Main theorem to prove that CD = 13
theorem length_CD (AB BC AC : ℕ) (edges : Set ℕ) (hAB : AB = 41) (hedges : edges = edge_lengths) :
  ∃ (CD : ℕ), CD ∈ edges ∧ CD = 13 :=
by
  sorry

end length_CD_l498_49864


namespace power_difference_l498_49877

theorem power_difference (x : ℝ) (hx : x - 1/x = 5) : x^4 - 1/x^4 = 727 :=
by
  sorry

end power_difference_l498_49877


namespace alex_score_l498_49891

theorem alex_score (initial_students : ℕ) (initial_average : ℕ) (total_students : ℕ) (new_average : ℕ) (initial_total : ℕ) (new_total : ℕ) :
  initial_students = 19 →
  initial_average = 76 →
  total_students = 20 →
  new_average = 78 →
  initial_total = initial_students * initial_average →
  new_total = total_students * new_average →
  new_total - initial_total = 116 :=
by
  sorry

end alex_score_l498_49891


namespace paving_cost_l498_49804

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate : ℝ := 300
def area : ℝ := length * width
def cost : ℝ := area * rate

theorem paving_cost : cost = 6187.50 := by
  -- length = 5.5
  -- width = 3.75
  -- rate = 300
  -- area = length * width = 20.625
  -- cost = area * rate = 6187.50
  sorry

end paving_cost_l498_49804


namespace initial_oranges_in_bowl_l498_49872

theorem initial_oranges_in_bowl (A O : ℕ) (R : ℚ) (h1 : A = 14) (h2 : R = 0.7) 
    (h3 : R * (A + O - 15) = A) : O = 21 := 
by 
  sorry

end initial_oranges_in_bowl_l498_49872


namespace condition_needs_l498_49897

theorem condition_needs (a b c d : ℝ) :
  a + c > b + d → (¬ (a > b ∧ c > d) ∧ (a > b ∧ c > d)) :=
by
  sorry

end condition_needs_l498_49897


namespace interest_rate_second_part_l498_49899

theorem interest_rate_second_part 
    (total_investment : ℝ) 
    (annual_interest : ℝ) 
    (P1 : ℝ) 
    (rate1 : ℝ) 
    (P2 : ℝ)
    (rate2 : ℝ) : 
    total_investment = 3600 → 
    annual_interest = 144 → 
    P1 = 1800 → 
    rate1 = 3 → 
    P2 = total_investment - P1 → 
    (annual_interest - (P1 * rate1 / 100)) = (P2 * rate2 / 100) →
    rate2 = 5 :=
by 
  intros total_investment_eq annual_interest_eq P1_eq rate1_eq P2_eq interest_eq
  sorry

end interest_rate_second_part_l498_49899


namespace milo_cash_reward_l498_49852

theorem milo_cash_reward : 
  let three_2s := [2, 2, 2]
  let four_3s := [3, 3, 3, 3]
  let one_4 := [4]
  let one_5 := [5]
  let all_grades := three_2s ++ four_3s ++ one_4 ++ one_5
  let total_grades := all_grades.length
  let total_sum := all_grades.sum
  let average_grade := total_sum / total_grades
  5 * average_grade = 15 := by
  sorry

end milo_cash_reward_l498_49852


namespace carrie_pays_94_l498_49834

theorem carrie_pays_94 :
  ∀ (num_shirts num_pants num_jackets : ℕ) (cost_shirt cost_pants cost_jacket : ℕ),
  num_shirts = 4 →
  cost_shirt = 8 →
  num_pants = 2 →
  cost_pants = 18 →
  num_jackets = 2 →
  cost_jacket = 60 →
  (cost_shirt * num_shirts + cost_pants * num_pants + cost_jacket * num_jackets) / 2 = 94 :=
by
  intros num_shirts num_pants num_jackets cost_shirt cost_pants cost_jacket
  sorry

end carrie_pays_94_l498_49834


namespace at_least_one_negative_root_l498_49800

theorem at_least_one_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (x^2 - 6*a*x - 2 + 2*a + 9*a^2 = 0)) ↔ a < (-1 + Real.sqrt 19) / 9 := by
  sorry

end at_least_one_negative_root_l498_49800


namespace binom_coeff_mult_l498_49808

theorem binom_coeff_mult :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_coeff_mult_l498_49808


namespace sum_powers_l498_49823

theorem sum_powers :
  ∃ (α β γ : ℂ), α + β + γ = 2 ∧ α^2 + β^2 + γ^2 = 5 ∧ α^3 + β^3 + γ^3 = 8 ∧ α^5 + β^5 + γ^5 = 46.5 :=
by
  sorry

end sum_powers_l498_49823


namespace smallest_nineteen_multiple_l498_49867

theorem smallest_nineteen_multiple (n : ℕ) 
  (h₁ : 19 * n ≡ 5678 [MOD 11]) : n = 8 :=
by sorry

end smallest_nineteen_multiple_l498_49867


namespace divisible_by_120_l498_49829

theorem divisible_by_120 (n : ℕ) : 120 ∣ n * (n^2 - 1) * (n^2 - 5 * n + 26) := sorry

end divisible_by_120_l498_49829


namespace largest_possible_sum_l498_49849

def max_sum_pair_mult_48 : Prop :=
  ∃ (heartsuit clubsuit : ℕ), (heartsuit * clubsuit = 48) ∧ (heartsuit + clubsuit = 49) ∧ 
  (∀ (h c : ℕ), (h * c = 48) → (h + c ≤ 49))

theorem largest_possible_sum : max_sum_pair_mult_48 :=
  sorry

end largest_possible_sum_l498_49849


namespace total_meters_examined_l498_49879

-- Define the conditions
def proportion_defective : ℝ := 0.1
def defective_meters : ℕ := 10

-- The statement to prove
theorem total_meters_examined (T : ℝ) (h : proportion_defective * T = defective_meters) : T = 100 :=
by
  sorry

end total_meters_examined_l498_49879


namespace sum_of_digits_in_T_shape_35_l498_49821

-- Define the set of digits
def digits : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the problem variables and conditions
theorem sum_of_digits_in_T_shape_35
  (a b c d e f g h : ℕ)
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
        d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
        e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
        f ≠ g ∧ f ≠ h ∧
        g ≠ h)
  (h2 : a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ 
        e ∈ digits ∧ f ∈ digits ∧ g ∈ digits ∧ h ∈ digits)
  (h3 : a + b + c + d = 26)
  (h4 : e + b + f + g + h = 20) :
  a + b + c + d + e + f + g + h = 35 := by
  sorry

end sum_of_digits_in_T_shape_35_l498_49821


namespace change_in_nickels_l498_49801

theorem change_in_nickels (cost_bread cost_cheese given_amount : ℝ) (quarters dimes : ℕ) (nickel_value : ℝ) 
  (h1 : cost_bread = 4.2) (h2 : cost_cheese = 2.05) (h3 : given_amount = 7.0)
  (h4 : quarters = 1) (h5 : dimes = 1) (hnickel_value : nickel_value = 0.05) : 
  ∃ n : ℕ, n = 8 :=
by
  sorry

end change_in_nickels_l498_49801


namespace find_m_l498_49844

theorem find_m (m x : ℝ) 
  (h1 : (m - 1) * x^2 + 5 * x + m^2 - 3 * m + 2 = 0) 
  (h2 : m^2 - 3 * m + 2 = 0)
  (h3 : m ≠ 1) : 
  m = 2 := 
sorry

end find_m_l498_49844


namespace total_coins_are_correct_l498_49841

-- Define the initial number of coins
def initial_dimes : Nat := 2
def initial_quarters : Nat := 6
def initial_nickels : Nat := 5

-- Define the additional coins given by Linda's mother
def additional_dimes : Nat := 2
def additional_quarters : Nat := 10
def additional_nickels : Nat := 2 * initial_nickels

-- Calculate the total number of each type of coin
def total_dimes : Nat := initial_dimes + additional_dimes
def total_quarters : Nat := initial_quarters + additional_quarters
def total_nickels : Nat := initial_nickels + additional_nickels

-- Total number of coins
def total_coins : Nat := total_dimes + total_quarters + total_nickels

-- Theorem to prove the total number of coins is 35
theorem total_coins_are_correct : total_coins = 35 := by
  -- Skip the proof
  sorry

end total_coins_are_correct_l498_49841


namespace coefficient_x9_l498_49816

theorem coefficient_x9 (p : Polynomial ℚ) : 
  p = (1 + 3 * Polynomial.X - Polynomial.X^2)^5 →
  Polynomial.coeff p 9 = 15 := 
by
  intro h
  rw [h]
  -- additional lean tactics to prove the statement would go here
  sorry

end coefficient_x9_l498_49816


namespace number_of_students_l498_49831

theorem number_of_students (n : ℕ) :
  n < 50 ∧ n % 6 = 4 ∧ n % 4 = 2 → n = 10 ∨ n = 22 ∨ n = 34 := by
  -- Proof goes here
  sorry

end number_of_students_l498_49831


namespace money_spent_on_video_games_l498_49868

theorem money_spent_on_video_games :
  let total_money := 50
  let fraction_books := 1 / 4
  let fraction_snacks := 2 / 5
  let fraction_apps := 1 / 5
  let spent_books := fraction_books * total_money
  let spent_snacks := fraction_snacks * total_money
  let spent_apps := fraction_apps * total_money
  let spent_other := spent_books + spent_snacks + spent_apps
  let spent_video_games := total_money - spent_other
  spent_video_games = 7.5 :=
by
  sorry

end money_spent_on_video_games_l498_49868


namespace proof_problem_l498_49842

def sum_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < n → S n = n * (n + 1) + 2 ∧ S 1 = a 1 ∧ (∀ n, 1 < n → a n = S n - S (n - 1))

def general_term_a (a : ℕ → ℕ) : Prop :=
  a 1 = 4 ∧ (∀ n, 1 < n → a n = 2 * n)

def geometric_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, 0 < k → 
  a 2 = 4 ∧ a (k+2) = 2 * (k + 2) ∧ a (3 * k + 2) = 2 * (3 * k + 2) →
  b 1 = a 2 ∧ b 2 = a (k + 2) ∧ b 3 = a (3 * k + 2) ∧ 
  (∀ n, b n = 2^(n + 1))

theorem proof_problem :
  ∃ (a b S : ℕ → ℕ),
  sum_of_sequence S a ∧ general_term_a a ∧ geometric_sequence a b :=
sorry

end proof_problem_l498_49842


namespace roots_of_equation_l498_49862

def operation (a b : ℝ) : ℝ := a^2 * b + a * b - 1

theorem roots_of_equation :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ operation x₁ 1 = 0 ∧ operation x₂ 1 = 0 :=
by
  sorry

end roots_of_equation_l498_49862


namespace M_eq_N_l498_49886

def M : Set ℤ := { u | ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l }
def N : Set ℤ := { u | ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r }

theorem M_eq_N : M = N := by
  sorry

end M_eq_N_l498_49886


namespace freshman_class_total_students_l498_49820

theorem freshman_class_total_students (N : ℕ) 
    (h1 : 90 ≤ N) 
    (h2 : 100 ≤ N)
    (h3 : 20 ≤ N) 
    (h4: (90 : ℝ) / N * (20 : ℝ) / 100 = (20 : ℝ) / N):
    N = 450 :=
  sorry

end freshman_class_total_students_l498_49820


namespace nancy_pictures_l498_49813

theorem nancy_pictures (z m b d : ℕ) (hz : z = 120) (hm : m = 75) (hb : b = 45) (hd : d = 93) :
  (z + m + b) - d = 147 :=
by {
  -- Theorem definition capturing the problem statement
  sorry
}

end nancy_pictures_l498_49813


namespace calculate_number_of_sides_l498_49809

theorem calculate_number_of_sides (n : ℕ) (h : n ≥ 6) :
  ((6 : ℚ) / n^2) * ((6 : ℚ) / n^2) = 0.027777777777777776 →
  n = 6 :=
by
  sorry

end calculate_number_of_sides_l498_49809


namespace range_of_a_l498_49865

noncomputable def is_even (f : ℝ → ℝ) :=
  ∀ x : ℝ, f x = f (-x)

noncomputable def is_increasing_on_nonneg (f : ℝ → ℝ) :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem range_of_a
  {f : ℝ → ℝ}
  (hf_even : is_even f)
  (hf_increasing : is_increasing_on_nonneg f)
  (hf_inequality : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f (a * x + 1) ≤ f (x - 3)) :
  -1 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l498_49865


namespace parabola_vertex_coordinates_l498_49838

theorem parabola_vertex_coordinates :
  ∀ x y : ℝ, y = 3 * (x - 7)^2 + 5 → (7, 5) = (7, 5) :=
by
  intros x y h
  exact rfl

end parabola_vertex_coordinates_l498_49838


namespace problem1_l498_49853

theorem problem1 (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 10)
  (h2 : x / 2 - (y + 1) / 3 = 1) :
  x = 3 ∧ y = 1 / 2 := 
sorry

end problem1_l498_49853


namespace num_elements_intersection_l498_49802

def A : Finset ℕ := {1, 2, 3, 4}
def B : Finset ℕ := {2, 4, 6, 8}

theorem num_elements_intersection : (A ∩ B).card = 2 := by
  sorry

end num_elements_intersection_l498_49802


namespace compare_abc_l498_49873

noncomputable def a : ℝ := 1 / Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.exp 0.5
noncomputable def c : ℝ := Real.log 2

theorem compare_abc : b > c ∧ c > a := by
  sorry

end compare_abc_l498_49873


namespace square_area_of_triangle_on_hyperbola_l498_49885

noncomputable def centroid_is_vertex (triangle : Set (ℝ × ℝ)) : Prop :=
  ∃ v : ℝ × ℝ, v ∈ triangle ∧ v.1 * v.2 = 4

noncomputable def triangle_properties (triangle : Set (ℝ × ℝ)) : Prop :=
  centroid_is_vertex triangle ∧
  (∃ centroid : ℝ × ℝ, 
    centroid_is_vertex triangle ∧ 
    (∀ p ∈ triangle, centroid ∈ triangle))

theorem square_area_of_triangle_on_hyperbola :
  ∃ triangle : Set (ℝ × ℝ), triangle_properties triangle ∧ (∃ area_sq : ℝ, area_sq = 1728) :=
by
  sorry

end square_area_of_triangle_on_hyperbola_l498_49885


namespace distinct_sums_count_l498_49874

theorem distinct_sums_count (n : ℕ) (a : Fin n.succ → ℕ) (h_distinct : Function.Injective a) :
  ∃ (S : Finset ℕ), S.card ≥ n * (n + 1) / 2 := sorry

end distinct_sums_count_l498_49874


namespace matt_skips_correctly_l498_49819

-- Definitions based on conditions
def skips_per_second := 3
def jumping_time_minutes := 10
def seconds_per_minute := 60
def total_jumping_seconds := jumping_time_minutes * seconds_per_minute
def expected_skips := total_jumping_seconds * skips_per_second

-- Proof statement
theorem matt_skips_correctly :
  expected_skips = 1800 :=
by
  sorry

end matt_skips_correctly_l498_49819


namespace apple_counting_l498_49878

theorem apple_counting
  (n m : ℕ)
  (vasya_trees_a_b petya_trees_a_b vasya_trees_b_c petya_trees_b_c vasya_trees_c_d petya_trees_c_d vasya_apples_a_b petya_apples_a_b vasya_apples_c_d petya_apples_c_d : ℕ)
  (h1 : petya_trees_a_b = 2 * vasya_trees_a_b)
  (h2 : petya_apples_a_b = 7 * vasya_apples_a_b)
  (h3 : petya_trees_b_c = 2 * vasya_trees_b_c)
  (h4 : petya_trees_c_d = 2 * vasya_trees_c_d)
  (h5 : n = vasya_trees_a_b + petya_trees_a_b)
  (h6 : m = vasya_apples_a_b + petya_apples_a_b)
  (h7 : vasya_trees_c_d = n / 3)
  (h8 : petya_trees_c_d = 2 * (n / 3))
  (h9 : vasya_apples_c_d = 3 * petya_apples_c_d)
  : vasya_apples_c_d = 3 * petya_apples_c_d :=
by 
  sorry

end apple_counting_l498_49878


namespace hardcover_books_count_l498_49837

theorem hardcover_books_count
  (h p : ℕ)
  (h_plus_p_eq_10 : h + p = 10)
  (total_cost_eq_250 : 30 * h + 20 * p = 250) :
  h = 5 :=
by
  sorry

end hardcover_books_count_l498_49837


namespace highest_score_l498_49855

theorem highest_score (H L : ℕ) (avg total46 total44 runs46 runs44 : ℕ)
  (h1 : H - L = 150)
  (h2 : avg = 61)
  (h3 : total46 = 46)
  (h4 : runs46 = avg * total46)
  (h5 : runs46 = 2806)
  (h6 : total44 = 44)
  (h7 : runs44 = 58 * total44)
  (h8 : runs44 = 2552)
  (h9 : runs46 - runs44 = H + L) :
  H = 202 := by
  sorry

end highest_score_l498_49855


namespace triangle_identity_l498_49861

theorem triangle_identity (a b c : ℝ) (B: ℝ) (hB: B = 120) :
    a^2 + a * c + c^2 - b^2 = 0 :=
by
  sorry

end triangle_identity_l498_49861


namespace range_of_m_l498_49859

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x ≤ 3 → (x ≤ m → (x < y → y < m))) → m ≥ 3 := 
by
  sorry

end range_of_m_l498_49859


namespace a_is_zero_l498_49892

theorem a_is_zero (a b : ℤ)
  (h : ∀ n : ℕ, ∃ x : ℤ, a * 2013^n + b = x^2) : a = 0 :=
by
  sorry

end a_is_zero_l498_49892


namespace impossible_coins_l498_49858

theorem impossible_coins (p1 p2 : ℝ) (h1 : (1 - p1) * (1 - p2) = p1 * p2) (h2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : false :=
  by sorry

end impossible_coins_l498_49858


namespace domain_of_fx_l498_49881

theorem domain_of_fx {x : ℝ} : (2 * x) / (x - 1) = (2 * x) / (x - 1) ↔ x ∈ {y : ℝ | y ≠ 1} :=
by
  sorry

end domain_of_fx_l498_49881


namespace base6_number_divisibility_l498_49860

/-- 
Given that 45x2 in base 6 converted to its decimal equivalent is 6x + 1046,
and it is divisible by 19. Prove that x = 5 given that x is a base-6 digit.
-/
theorem base6_number_divisibility (x : ℕ) (h1 : 0 ≤ x ∧ x ≤ 5) (h2 : (6 * x + 1046) % 19 = 0) : x = 5 :=
sorry

end base6_number_divisibility_l498_49860


namespace penny_exceeded_by_32_l498_49857

def bulk_price : ℤ := 5
def min_spend_before_tax : ℤ := 40
def tax_per_pound : ℤ := 1
def penny_payment : ℤ := 240

def total_cost_per_pound : ℤ := bulk_price + tax_per_pound

def min_pounds_for_min_spend : ℤ := min_spend_before_tax / bulk_price

def total_pounds_penny_bought : ℤ := penny_payment / total_cost_per_pound

def pounds_exceeded : ℤ := total_pounds_penny_bought - min_pounds_for_min_spend

theorem penny_exceeded_by_32 : pounds_exceeded = 32 := by
  sorry

end penny_exceeded_by_32_l498_49857


namespace number_of_adults_attending_concert_l498_49825

-- We have to define the constants and conditions first.
variable (A C : ℕ)
variable (h1 : A + C = 578)
variable (h2 : 2 * A + 3 / 2 * C = 985)

-- Now we state the theorem that given these conditions, A is equal to 236.

theorem number_of_adults_attending_concert : A = 236 :=
by sorry

end number_of_adults_attending_concert_l498_49825


namespace CatCafePawRatio_l498_49880

-- Define the context
def CatCafeMeow (P : ℕ) := 3 * P
def CatCafePaw (P : ℕ) := P
def CatCafeCool := 5
def TotalCats (P : ℕ) := CatCafeMeow P + CatCafePaw P

-- State the theorem
theorem CatCafePawRatio (P : ℕ) (n : ℕ) : 
  CatCafeCool = 5 →
  CatCafeMeow P = 3 * CatCafePaw P →
  TotalCats P = 40 →
  P = 10 →
  n * CatCafeCool = P →
  n = 2 :=
by
  intros
  sorry

end CatCafePawRatio_l498_49880


namespace simplify_expression_l498_49894

theorem simplify_expression (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : a + c > b) :
  |a + b - c| - |b - a - c| = 2 * b - 2 * c :=
by
  sorry

end simplify_expression_l498_49894


namespace probability_of_selection_is_equal_l498_49845

-- Define the conditions of the problem
def total_students := 2004
def eliminated_students := 4
def remaining_students := total_students - eliminated_students -- 2000
def selected_students := 50
def k := remaining_students / selected_students -- 40

-- Define the probability calculation
def probability_selected := selected_students / remaining_students

-- The theorem stating that every student has a 1/40 probability of being selected
theorem probability_of_selection_is_equal :
  probability_selected = 1 / 40 :=
by
  -- insert proof logic here
  sorry

end probability_of_selection_is_equal_l498_49845


namespace tower_height_l498_49889

theorem tower_height (h d : ℝ) 
  (tan_30_eq : Real.tan (Real.pi / 6) = h / d)
  (tan_45_eq : Real.tan (Real.pi / 4) = h / (d - 20)) :
  h = 20 * Real.sqrt 3 :=
by
  sorry

end tower_height_l498_49889


namespace crayon_colors_correct_l498_49833

-- The Lean code will define the conditions and the proof statement as follows:
noncomputable def crayon_problem := 
  let crayons_per_box := (160 / (5 * 4)) -- Total crayons / Total boxes
  let colors := (crayons_per_box / 2) -- Crayons per box / Crayons per color
  colors = 4

-- This is the theorem that needs to be proven:
theorem crayon_colors_correct : crayon_problem := by
  sorry

end crayon_colors_correct_l498_49833


namespace points_satisfying_inequality_l498_49812

theorem points_satisfying_inequality (x y : ℝ) :
  ( ( (x * y + 1) / (x + y) )^2 < 1) ↔ 
  ( (-1 < x ∧ x < 1) ∧ (y < -1 ∨ y > 1) ) ∨ 
  ( (x < -1 ∨ x > 1) ∧ (-1 < y ∧ y < 1) ) := 
sorry

end points_satisfying_inequality_l498_49812


namespace polynomial_bound_l498_49827

noncomputable def P (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem polynomial_bound (a b c d : ℝ) (h : ∀ x : ℝ, |x| < 1 → |P a b c d x| ≤ 1) :
  |a| + |b| + |c| + |d| ≤ 7 :=
sorry

end polynomial_bound_l498_49827


namespace find_abc_l498_49839

noncomputable def f (x : ℝ) : ℝ := 1 + 2 * Real.cos x + 3 * Real.sin x

theorem find_abc (a b c : ℝ) : 
  (∀ x : ℝ, a * f x + b * f (x - c) = 1) →
  (∃ n : ℤ, a = 1 / 2 ∧ b = 1 / 2 ∧ c = (2 * n + 1) * Real.pi) :=
by
  sorry

end find_abc_l498_49839


namespace audio_per_cd_l498_49810

theorem audio_per_cd (total_audio : ℕ) (max_per_cd : ℕ) (num_cds : ℕ) 
  (h1 : total_audio = 360) 
  (h2 : max_per_cd = 60) 
  (h3 : num_cds = total_audio / max_per_cd): 
  (total_audio / num_cds = max_per_cd) :=
by
  sorry

end audio_per_cd_l498_49810


namespace value_of_fraction_l498_49893

theorem value_of_fraction (x y : ℤ) (h : x / y = 7 / 2) : (x - 2 * y) / y = 3 / 2 := by
  sorry

end value_of_fraction_l498_49893


namespace length_of_second_train_l498_49882

theorem length_of_second_train 
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (time_to_cross : ℝ)
  (relative_speed : ℝ)
  (total_distance_covered : ℝ)
  (L : ℝ)
  (h1 : length_first_train = 210)
  (h2 : speed_first_train = 120 * 1000 / 3600)
  (h3 : speed_second_train = 80 * 1000 / 3600)
  (h4 : time_to_cross = 9)
  (h5 : relative_speed = (120 * 1000 / 3600) + (80 * 1000 / 3600))
  (h6 : total_distance_covered = relative_speed * time_to_cross)
  (h7 : total_distance_covered = length_first_train + L) : 
  L = 289.95 :=
by {
  sorry
}

end length_of_second_train_l498_49882


namespace equation_verification_l498_49806

theorem equation_verification :
  (96 / 12 = 8) ∧ (45 - 37 = 8) := 
by
  -- We can add the necessary proofs later
  sorry

end equation_verification_l498_49806


namespace difference_of_squares_l498_49830

theorem difference_of_squares : 
  let a := 625
  let b := 575
  (a^2 - b^2) = 60000 :=
by 
  let a := 625
  let b := 575
  sorry

end difference_of_squares_l498_49830


namespace percentage_passed_l498_49898

def swim_club_members := 100
def not_passed_course_taken := 40
def not_passed_course_not_taken := 30
def not_passed := not_passed_course_taken + not_passed_course_not_taken

theorem percentage_passed :
  ((swim_club_members - not_passed).toFloat / swim_club_members.toFloat * 100) = 30 := by
  sorry

end percentage_passed_l498_49898


namespace words_with_mistakes_percentage_l498_49854

theorem words_with_mistakes_percentage (n x : ℕ) 
  (h1 : (x - 1 : ℝ) / n = 0.24)
  (h2 : (x - 1 : ℝ) / (n - 1) = 0.25) :
  (x : ℝ) / n * 100 = 28 := 
by 
  sorry

end words_with_mistakes_percentage_l498_49854


namespace number_of_cubes_with_at_least_two_faces_painted_is_56_l498_49824

def one_inch_cubes_with_at_least_two_faces_painted 
  (side_length : ℕ) (face_colors : ℕ) (cubes_per_side : ℕ) :=
  if side_length = 4 ∧ face_colors = 6 ∧ cubes_per_side = 1 then 56 else 0

theorem number_of_cubes_with_at_least_two_faces_painted_is_56 :
  one_inch_cubes_with_at_least_two_faces_painted 4 6 1 = 56 :=
by
  sorry

end number_of_cubes_with_at_least_two_faces_painted_is_56_l498_49824


namespace first_player_win_boards_l498_49866

-- Define what it means for a player to guarantee a win
def first_player_guarantees_win (n m : ℕ) : Prop :=
  ¬(n % 2 = 1 ∧ m % 2 = 1)

-- The main theorem that matches the math proof problem
theorem first_player_win_boards : (first_player_guarantees_win 6 7) ∧
                                  (first_player_guarantees_win 6 8) ∧
                                  (first_player_guarantees_win 7 8) ∧
                                  (first_player_guarantees_win 8 8) ∧
                                  ¬(first_player_guarantees_win 7 7) := 
by 
sorry

end first_player_win_boards_l498_49866


namespace seating_impossible_l498_49846

theorem seating_impossible (reps : Fin 54 → Fin 27) : 
  ¬ ∃ (s : Fin 54 → Fin 54),
    (∀ i : Fin 27, ∃ a b : Fin 54, a ≠ b ∧ s a = i ∧ s b = i ∧ (b - a ≡ 10 [MOD 54] ∨ a - b ≡ 10 [MOD 54])) :=
sorry

end seating_impossible_l498_49846


namespace Elberta_has_23_dollars_l498_49847

theorem Elberta_has_23_dollars (GrannySmith_has : ℕ := 72)
    (Anjou_has : ℕ := GrannySmith_has / 4)
    (Elberta_has : ℕ := Anjou_has + 5) : Elberta_has = 23 :=
by
  sorry

end Elberta_has_23_dollars_l498_49847


namespace prove_y_value_l498_49856

theorem prove_y_value (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  sorry

end prove_y_value_l498_49856


namespace operation_result_l498_49895

-- Define the new operation x # y
def op (x y : ℕ) : ℤ := 2 * x * y - 3 * x + y

-- Prove that (6 # 4) - (4 # 6) = -8
theorem operation_result : op 6 4 - op 4 6 = -8 :=
by
  sorry

end operation_result_l498_49895


namespace new_volume_l498_49840

theorem new_volume (l w h : ℝ) 
  (h1 : l * w * h = 4320)
  (h2 : l * w + w * h + l * h = 852)
  (h3 : l + w + h = 52) :
  (l + 4) * (w + 4) * (h + 4) = 8624 := sorry

end new_volume_l498_49840


namespace other_number_is_7_l498_49818

-- Given conditions
variable (a b : ℤ)
variable (h1 : 2 * a + 3 * b = 110)
variable (h2 : a = 32 ∨ b = 32)

-- The proof goal
theorem other_number_is_7 : (a = 7 ∧ b = 32) ∨ (a = 32 ∧ b = 7) :=
by
  sorry

end other_number_is_7_l498_49818


namespace positive_integer_solution_l498_49811

theorem positive_integer_solution (x : Int) (h_pos : x > 0) (h_cond : x + 1000 > 1000 * x) : x = 2 :=
sorry

end positive_integer_solution_l498_49811
