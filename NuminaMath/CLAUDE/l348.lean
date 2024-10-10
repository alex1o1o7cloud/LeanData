import Mathlib

namespace max_sum_expression_l348_34853

def is_valid_set (s : Finset ℕ) : Prop :=
  s.card = 4 ∧ s ⊆ {1, 3, 5, 7}

def sum_expression (a b c d : ℕ) : ℕ :=
  a * b + b * c + c * d + d * a + a^2 + b^2 + c^2 + d^2

theorem max_sum_expression (s : Finset ℕ) (hs : is_valid_set s) :
  ∃ (a b c d : ℕ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∀ (w x y z : ℕ), w ∈ s → x ∈ s → y ∈ s → z ∈ s →
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
  sum_expression a b c d ≥ sum_expression w x y z ∧
  sum_expression a b c d = 201 :=
sorry

end max_sum_expression_l348_34853


namespace a_pow_b_greater_than_three_pow_n_over_n_l348_34866

theorem a_pow_b_greater_than_three_pow_n_over_n 
  (a b n : ℕ) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : Odd b) 
  (h4 : n ≥ 1) 
  (h5 : (b^n : ℕ) ∣ (a^n - 1)) : 
  (a^b : ℚ) > (3^n : ℚ) / n := by
  sorry

end a_pow_b_greater_than_three_pow_n_over_n_l348_34866


namespace max_product_theorem_l348_34872

theorem max_product_theorem (x₁ x₂ x₃ : ℝ) 
  (h₁ : 0 ≤ x₁ ∧ x₁ ≤ 12)
  (h₂ : 0 ≤ x₂ ∧ x₂ ≤ 12)
  (h₃ : 0 ≤ x₃ ∧ x₃ ≤ 12)
  (h₄ : x₁ * x₂ * x₃ = ((12 - x₁) * (12 - x₂) * (12 - x₃))^2) :
  x₁ * x₂ * x₃ ≤ 729 :=
by sorry

end max_product_theorem_l348_34872


namespace negative_root_range_l348_34824

theorem negative_root_range (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (5 - a)) →
  -3 < a ∧ a < 1 := by
sorry

end negative_root_range_l348_34824


namespace average_speed_comparison_l348_34892

theorem average_speed_comparison (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) :
  3 / (1/u + 1/v + 1/w) ≤ (u + v + w) / 3 := by
  sorry

end average_speed_comparison_l348_34892


namespace titius_bode_ninth_planet_l348_34823

/-- The Titius-Bode law formula -/
def titius_bode (a b : ℝ) (n : ℕ) : ℝ := a + b * 2^(n - 1)

/-- Theorem stating the Titius-Bode law for the 9th planet -/
theorem titius_bode_ninth_planet :
  ∃ (a b : ℝ),
    (titius_bode a b 1 = 0.7) ∧
    (titius_bode a b 2 = 1) ∧
    (titius_bode a b 9 = 77.2) := by
  sorry

end titius_bode_ninth_planet_l348_34823


namespace H_range_l348_34898

def H (x : ℝ) : ℝ := |x + 2| - |x - 3| + 3 * x

theorem H_range : Set.range H = Set.Ici (-1) := by sorry

end H_range_l348_34898


namespace smallest_common_factor_l348_34877

theorem smallest_common_factor (n : ℕ) : 
  (∀ k < 165, k > 0 → Nat.gcd (11*k - 8) (5*k + 9) = 1) ∧ 
  Nat.gcd (11*165 - 8) (5*165 + 9) > 1 := by
  sorry

end smallest_common_factor_l348_34877


namespace mark_leftover_money_l348_34840

def original_wage : ℝ := 40
def raise_percentage : ℝ := 0.05
def hours_per_day : ℝ := 8
def days_per_week : ℝ := 5
def old_bills : ℝ := 600
def personal_trainer_cost : ℝ := 100

def new_wage : ℝ := original_wage * (1 + raise_percentage)
def daily_earnings : ℝ := new_wage * hours_per_day
def weekly_earnings : ℝ := daily_earnings * days_per_week
def weekly_expenses : ℝ := old_bills + personal_trainer_cost

theorem mark_leftover_money : weekly_earnings - weekly_expenses = 980 := by
  sorry

end mark_leftover_money_l348_34840


namespace single_burger_cost_l348_34876

theorem single_burger_cost 
  (total_spent : ℚ)
  (total_burgers : ℕ)
  (double_burger_cost : ℚ)
  (double_burgers : ℕ)
  (h1 : total_spent = 66.5)
  (h2 : total_burgers = 50)
  (h3 : double_burger_cost = 1.5)
  (h4 : double_burgers = 33) :
  (total_spent - double_burger_cost * double_burgers) / (total_burgers - double_burgers) = 1 := by
sorry

end single_burger_cost_l348_34876


namespace black_area_after_changes_l348_34851

/-- The fraction of black area remaining after one change -/
def remaining_fraction : ℚ := 8 / 9

/-- The number of changes -/
def num_changes : ℕ := 4

/-- The fraction of the original black area remaining after 'num_changes' changes -/
def final_fraction : ℚ := remaining_fraction ^ num_changes

theorem black_area_after_changes : final_fraction = 4096 / 6561 := by
  sorry

end black_area_after_changes_l348_34851


namespace unique_valid_assignment_l348_34865

/-- Represents the letters used in the triangle puzzle -/
inductive Letter
| A | B | C | D | E | F

/-- Represents the possible values for each letter -/
def LetterValue := Fin 6

/-- A function that assigns values to letters -/
def Assignment := Letter → LetterValue

/-- Checks if an assignment is valid according to the puzzle rules -/
def is_valid_assignment (f : Assignment) : Prop :=
  -- All values are distinct
  (∀ x y : Letter, x ≠ y → f x ≠ f y) ∧
  -- The sum of D, E, and B equals 14
  (f Letter.D).val + (f Letter.E).val + (f Letter.B).val = 14 ∧
  -- F is the sum of D and E minus the sum of A, B, and C
  (f Letter.F).val = (f Letter.D).val + (f Letter.E).val - ((f Letter.A).val + (f Letter.B).val + (f Letter.C).val)

/-- The unique valid assignment for the puzzle -/
def unique_assignment : Assignment :=
  fun l => match l with
  | Letter.A => ⟨0, by simp⟩  -- 1
  | Letter.B => ⟨2, by simp⟩  -- 3
  | Letter.C => ⟨1, by simp⟩  -- 2
  | Letter.D => ⟨4, by simp⟩  -- 5
  | Letter.E => ⟨5, by simp⟩  -- 6
  | Letter.F => ⟨3, by simp⟩  -- 4

/-- Theorem stating that the unique_assignment is the only valid assignment -/
theorem unique_valid_assignment :
  is_valid_assignment unique_assignment ∧
  ∀ f : Assignment, is_valid_assignment f → f = unique_assignment :=
sorry

end unique_valid_assignment_l348_34865


namespace room_width_calculation_l348_34819

theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (width : ℝ) : 
  length = 5.5 →
  cost_per_sqm = 400 →
  total_cost = 8250 →
  width * length * cost_per_sqm = total_cost →
  width = 3.75 := by
sorry

end room_width_calculation_l348_34819


namespace james_profit_l348_34848

def total_tickets : ℕ := 200
def ticket_prices : List ℚ := [1, 3, 4]
def ticket_percentages : List ℚ := [50, 30, 20]
def winning_odds : List ℚ := [30, 20, 10]
def winning_percentages : List ℚ := [80, 1, 19]
def winning_amounts : List ℚ := [5, 5000, 15]
def tax_rate : ℚ := 10

def calculate_profit (total_tickets : ℕ) (ticket_prices : List ℚ) (ticket_percentages : List ℚ)
  (winning_odds : List ℚ) (winning_percentages : List ℚ) (winning_amounts : List ℚ) (tax_rate : ℚ) : ℚ :=
  sorry

theorem james_profit :
  calculate_profit total_tickets ticket_prices ticket_percentages winning_odds winning_percentages winning_amounts tax_rate = 4109.50 := by
  sorry

end james_profit_l348_34848


namespace cows_bought_is_two_l348_34802

/-- The number of cows bought given the total cost, number of goats, and average prices. -/
def number_of_cows (total_cost goats goat_price cow_price : ℕ) : ℕ :=
  (total_cost - goats * goat_price) / cow_price

/-- Theorem stating that the number of cows bought is 2 under the given conditions. -/
theorem cows_bought_is_two :
  number_of_cows 1500 10 70 400 = 2 := by
  sorry

end cows_bought_is_two_l348_34802


namespace waiting_time_is_twenty_l348_34868

/-- Represents the time components of Mary's trip to the airport -/
structure TripTime where
  uber_to_house : ℕ
  uber_to_airport_multiplier : ℕ
  bag_check : ℕ
  security_multiplier : ℕ
  total_trip_time : ℕ

/-- Calculates the waiting time for the flight to start boarding -/
def waiting_time (t : TripTime) : ℕ :=
  let uber_to_airport := t.uber_to_house * t.uber_to_airport_multiplier
  let security := t.bag_check * t.security_multiplier
  let total_pre_wait := t.uber_to_house + uber_to_airport + t.bag_check + security
  let remaining_time := t.total_trip_time - total_pre_wait
  remaining_time / 3

/-- Theorem stating that the waiting time for the flight to start boarding is 20 minutes -/
theorem waiting_time_is_twenty (t : TripTime) 
  (h1 : t.uber_to_house = 10)
  (h2 : t.uber_to_airport_multiplier = 5)
  (h3 : t.bag_check = 15)
  (h4 : t.security_multiplier = 3)
  (h5 : t.total_trip_time = 180) : 
  waiting_time t = 20 := by
  sorry

end waiting_time_is_twenty_l348_34868


namespace sum_squared_equals_four_l348_34803

theorem sum_squared_equals_four (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 - 2*a + 4*b - 6*c + 14 = 0) : 
  (a + b + c)^2 = 4 := by
  sorry

end sum_squared_equals_four_l348_34803


namespace jonas_library_space_l348_34899

theorem jonas_library_space (total_space : ℝ) (shelf_space : ℝ) (num_shelves : ℕ) 
  (h1 : total_space = 400)
  (h2 : shelf_space = 80)
  (h3 : num_shelves = 3) :
  total_space - (↑num_shelves * shelf_space) = 160 := by
sorry

end jonas_library_space_l348_34899


namespace incorrect_analysis_is_B_l348_34878

/-- Represents the content of the novel -/
def novel_content : String := "..."

/-- Represents the four analysis options -/
inductive Analysis
| A : Analysis
| B : Analysis
| C : Analysis
| D : Analysis

/-- Checks if an analysis is correct based on the novel content -/
def is_correct_analysis (a : Analysis) (content : String) : Prop :=
  match a with
  | Analysis.A => True  -- Assumed correct for this problem
  | Analysis.B => False -- Known to be incorrect
  | Analysis.C => True  -- Assumed correct for this problem
  | Analysis.D => True  -- Assumed correct for this problem

/-- The main theorem stating that option B is the incorrect analysis -/
theorem incorrect_analysis_is_B (content : String) :
  ∃ (a : Analysis), ¬(is_correct_analysis a content) ∧ a = Analysis.B :=
by
  sorry


end incorrect_analysis_is_B_l348_34878


namespace quadratic_root_sum_l348_34828

/-- Given a quadratic equation x² + bx + c = 0 with one non-zero real root c,
    prove that b + c = -1 -/
theorem quadratic_root_sum (b c : ℝ) (h : c ≠ 0) 
  (h_root : c^2 + b*c + c = 0) : b + c = -1 := by
  sorry

end quadratic_root_sum_l348_34828


namespace xiaohuas_apples_l348_34863

theorem xiaohuas_apples :
  ∃ (x : ℕ), 
    x > 0 ∧ 
    (0 < 4 * x + 20 - 8 * (x - 1)) ∧ 
    (4 * x + 20 - 8 * (x - 1) < 8) ∧
    (4 * x + 20 = 44) := by
  sorry

end xiaohuas_apples_l348_34863


namespace grey_purchase_theorem_l348_34834

/-- Represents the number of chickens and ducks bought by a person -/
structure Purchase where
  chickens : Nat
  ducks : Nat

/-- The problem setup -/
def grey_purchase_problem : Prop :=
  ∃ (mary_purchase : Purchase),
    let ray_purchase : Purchase := ⟨10, 3⟩
    let john_purchase : Purchase := ⟨mary_purchase.chickens + 5, mary_purchase.ducks + 2⟩
    ray_purchase.chickens = mary_purchase.chickens - 6 ∧
    ray_purchase.ducks = mary_purchase.ducks - 1 ∧
    (john_purchase.chickens + john_purchase.ducks) - (ray_purchase.chickens + ray_purchase.ducks) = 14

theorem grey_purchase_theorem : grey_purchase_problem := by
  sorry

end grey_purchase_theorem_l348_34834


namespace max_value_of_p_l348_34806

theorem max_value_of_p (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y * z + x + z = y) :
  ∃ (p : ℝ), p = 2 / (x^2 + 1) - 2 / (y^2 + 1) + 3 / (z^2 + 1) ∧
  p ≤ 10 / 3 ∧
  ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    x' * y' * z' + x' + z' = y' ∧
    2 / (x'^2 + 1) - 2 / (y'^2 + 1) + 3 / (z'^2 + 1) = 10 / 3 := by
sorry

end max_value_of_p_l348_34806


namespace sum_of_three_nines_power_twenty_l348_34805

theorem sum_of_three_nines_power_twenty (n : ℕ) : 9^20 + 9^20 + 9^20 = 3^41 := by
  sorry

end sum_of_three_nines_power_twenty_l348_34805


namespace tate_total_years_l348_34850

/-- Calculate the total years Tate spent in education, travel, and work -/
def totalYears : ℕ :=
  let typicalHighSchool := 4
  let highSchool := typicalHighSchool - 1
  let gapYears := 2
  let bachelors := 2 * highSchool
  let certification := 1
  let workExperience := 1
  let masters := bachelors / 2
  let phd := 3 * (highSchool + bachelors + masters)
  highSchool + gapYears + bachelors + certification + workExperience + masters + phd

/-- Theorem stating that the total years Tate spent is 52 -/
theorem tate_total_years : totalYears = 52 := by
  sorry

end tate_total_years_l348_34850


namespace largest_odd_digit_multiple_of_5_proof_l348_34890

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

def largest_odd_digit_multiple_of_5 : ℕ :=
  9955

theorem largest_odd_digit_multiple_of_5_proof :
  (largest_odd_digit_multiple_of_5 < 10000) ∧
  (largest_odd_digit_multiple_of_5 % 5 = 0) ∧
  has_only_odd_digits largest_odd_digit_multiple_of_5 ∧
  ∀ n : ℕ, n < 10000 → n % 5 = 0 → has_only_odd_digits n →
    n ≤ largest_odd_digit_multiple_of_5 :=
by sorry

end largest_odd_digit_multiple_of_5_proof_l348_34890


namespace quadratic_inequality_condition_incorrect_l348_34818

theorem quadratic_inequality_condition_incorrect 
  (a b c : ℝ) (h1 : a < 0) (h2 : b^2 - 4*a*c ≤ 0) :
  ¬ (∀ x : ℝ, a*x^2 + b*x + c ≤ 0) :=
by sorry

end quadratic_inequality_condition_incorrect_l348_34818


namespace leila_time_allocation_l348_34894

/-- Leila's utility function --/
def utility (juggling_hours coding_hours : ℝ) : ℝ := juggling_hours * coding_hours

/-- Leila's time allocation problem --/
theorem leila_time_allocation (s : ℝ) : 
  utility s (12 - s) = utility (6 - s) (s + 4) → s = 12 / 5 := by
  sorry

end leila_time_allocation_l348_34894


namespace log_125_equals_3_minus_3log2_l348_34893

theorem log_125_equals_3_minus_3log2 : Real.log 125 = 3 - 3 * Real.log 2 := by
  sorry

end log_125_equals_3_minus_3log2_l348_34893


namespace tournament_teams_l348_34858

-- Define the number of teams
def n : ℕ := sorry

-- Define the total number of matches
def total_matches : ℕ := 28

-- Theorem: The number of teams in the tournament is 8
theorem tournament_teams : n = 8 := by
  -- Define the formula for the number of matches in a round-robin tournament
  have round_robin_formula : total_matches = n * (n - 1) / 2 := sorry
  
  -- Prove that n = 8 satisfies the formula
  sorry


end tournament_teams_l348_34858


namespace intersection_of_A_and_B_solution_set_theorem_l348_34801

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem for part (1)
theorem intersection_of_A_and_B : A ∩ B = A_intersect_B := by sorry

-- Define the solution set of x^2 + ax - b < 0
def solution_set (a b : ℝ) : Set ℝ := {x | x < -1 ∨ x > 2}

-- Theorem for part (2)
theorem solution_set_theorem (a b : ℝ) :
  ({x : ℝ | x^2 + a*x + b < 0} = A_intersect_B) →
  ({x : ℝ | x^2 + a*x - b < 0} = solution_set a b) := by sorry

end intersection_of_A_and_B_solution_set_theorem_l348_34801


namespace radical_conjugate_sum_product_l348_34836

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = 8 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 16 → 
  a + b = 4 := by
sorry

end radical_conjugate_sum_product_l348_34836


namespace scientific_notation_of_634000000_l348_34889

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_634000000 :
  toScientificNotation 634000000 = ScientificNotation.mk 6.34 8 (by norm_num) :=
sorry

end scientific_notation_of_634000000_l348_34889


namespace abs_a_eq_5_and_a_plus_b_eq_0_l348_34896

theorem abs_a_eq_5_and_a_plus_b_eq_0 (a b : ℝ) (h1 : |a| = 5) (h2 : a + b = 0) :
  a - b = 10 ∨ a - b = -10 := by sorry

end abs_a_eq_5_and_a_plus_b_eq_0_l348_34896


namespace gumball_probability_l348_34809

theorem gumball_probability (blue_prob : ℚ) (pink_prob : ℚ) : 
  blue_prob^2 = 9/49 → pink_prob = 4/7 := by
  sorry

end gumball_probability_l348_34809


namespace min_value_a_minus_2b_l348_34839

/-- A quadratic function f(x) = x^2 - ax + b with one root in [-1, 1] and another in [1, 2] -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  root_in_neg_one_to_one : ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ x^2 - a*x + b = 0
  root_in_one_to_two : ∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ x^2 - a*x + b = 0

/-- The minimum value of a - 2b for the given quadratic function is -1 -/
theorem min_value_a_minus_2b (f : QuadraticFunction) :
  ∃ m : ℝ, m = -1 ∧ ∀ x : ℝ, f.a - 2*f.b ≥ m :=
sorry

end min_value_a_minus_2b_l348_34839


namespace solve_system_of_equations_l348_34874

theorem solve_system_of_equations (x y : ℝ) 
  (eq1 : (x - y) * (x * y) = 30)
  (eq2 : (x + y) * (x * y) = 120) :
  x = 5 ∧ y = 3 := by
  sorry

end solve_system_of_equations_l348_34874


namespace yellow_daisy_percentage_l348_34841

/-- Represents the types of flowers in the collection -/
inductive FlowerType
| Tulip
| Daisy

/-- Represents the colors of flowers in the collection -/
inductive FlowerColor
| Red
| Yellow

/-- Represents the collection of flowers -/
structure FlowerCollection where
  total : ℕ
  tulips : ℕ
  redTulips : ℕ
  yellowDaisies : ℕ

/-- The theorem statement -/
theorem yellow_daisy_percentage
  (collection : FlowerCollection)
  (h_total : collection.total = 120)
  (h_tulips : collection.tulips = collection.total * 3 / 10)
  (h_redTulips : collection.redTulips = collection.tulips / 2)
  (h_yellowDaisies : collection.yellowDaisies = (collection.total - collection.tulips) * 3 / 5) :
  collection.yellowDaisies * 100 / collection.total = 40 := by
  sorry

end yellow_daisy_percentage_l348_34841


namespace lemon_heads_boxes_l348_34887

/-- The number of Lemon Heads in each package -/
def package_size : ℕ := 6

/-- The total number of Lemon Heads Louis ate -/
def total_eaten : ℕ := 54

/-- The number of whole boxes Louis ate -/
def whole_boxes : ℕ := total_eaten / package_size

theorem lemon_heads_boxes : whole_boxes = 9 := by
  sorry

end lemon_heads_boxes_l348_34887


namespace tim_stored_26_bales_l348_34847

/-- The number of bales Tim stored in the barn -/
def bales_stored (initial_bales final_bales : ℕ) : ℕ :=
  final_bales - initial_bales

/-- Proof that Tim stored 26 bales in the barn -/
theorem tim_stored_26_bales (initial_bales final_bales : ℕ)
  (h1 : initial_bales = 28)
  (h2 : final_bales = 54) :
  bales_stored initial_bales final_bales = 26 := by
  sorry

end tim_stored_26_bales_l348_34847


namespace angle_C_measure_l348_34891

/-- In a triangle ABC, given the area formula S = (a² + b² - c²) / 4, 
    prove that the measure of angle C is 45°. -/
theorem angle_C_measure (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (a^2 + b^2 - c^2) / 4
  ∃ A B C : ℝ, 
    A > 0 ∧ B > 0 ∧ C > 0 ∧
    A + B + C = Real.pi ∧
    S = (1/2) * a * b * Real.sin C ∧
    C = Real.pi / 4 :=
by sorry

end angle_C_measure_l348_34891


namespace parallel_line_through_point_l348_34815

/-- Given a line L1 with equation x - y + 2 = 0 and a point P (2, 1),
    the line L2 passing through P and parallel to L1 has equation x - y = 1 -/
theorem parallel_line_through_point (x y : ℝ) :
  (x - y + 2 = 0) →  -- L1: original line equation
  (∃ m b : ℝ, x - y = m * x + b) →  -- L2: general form of parallel line
  (2 - 1 = m * 2 + b) →  -- L2 passes through (2, 1)
  (x - y = 1) :=  -- L2: final equation
by sorry

end parallel_line_through_point_l348_34815


namespace square_difference_of_integers_l348_34830

theorem square_difference_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 72)
  (diff_eq : x - y = 18) :
  x^2 - y^2 = 1296 := by
sorry

end square_difference_of_integers_l348_34830


namespace least_number_of_trees_l348_34816

theorem least_number_of_trees (n : ℕ) : (n > 0 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0) → n ≥ 210 :=
by sorry

end least_number_of_trees_l348_34816


namespace xy_bounds_l348_34833

theorem xy_bounds (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) :
  -1 ≤ x * y ∧ x * y ≤ 1/3 := by
  sorry

end xy_bounds_l348_34833


namespace tp_rolls_count_l348_34829

/-- The time in seconds to clean up one egg -/
def egg_cleanup_time : ℕ := 15

/-- The time in minutes to clean up one roll of toilet paper -/
def tp_cleanup_time : ℕ := 30

/-- The total cleaning time in minutes -/
def total_cleanup_time : ℕ := 225

/-- The number of eggs to clean up -/
def num_eggs : ℕ := 60

/-- Theorem stating that the number of toilet paper rolls is 7 -/
theorem tp_rolls_count : ℕ := by
  sorry

end tp_rolls_count_l348_34829


namespace veranda_area_is_148_l348_34831

/-- Calculates the area of a veranda surrounding a rectangular room. -/
def verandaArea (roomLength roomWidth verandaWidth : ℝ) : ℝ :=
  let totalLength := roomLength + 2 * verandaWidth
  let totalWidth := roomWidth + 2 * verandaWidth
  let totalArea := totalLength * totalWidth
  let roomArea := roomLength * roomWidth
  totalArea - roomArea

/-- Proves that the area of the veranda is 148 m² given the specified dimensions. -/
theorem veranda_area_is_148 :
  verandaArea 21 12 2 = 148 := by
  sorry

end veranda_area_is_148_l348_34831


namespace relationship_between_variables_l348_34881

theorem relationship_between_variables (x y z w : ℝ) 
  (h : (x + y) / (y + z) = (2 * z + w) / (w + x)) : 
  x = 2 * z - w := by sorry

end relationship_between_variables_l348_34881


namespace brick_ratio_proof_l348_34825

theorem brick_ratio_proof (total_bricks : ℕ) (discount_price full_price total_spent : ℚ)
  (h1 : total_bricks = 1000)
  (h2 : discount_price = 0.25)
  (h3 : full_price = 0.50)
  (h4 : total_spent = 375)
  : ∃ (discounted_bricks : ℕ),
    discounted_bricks * discount_price + (total_bricks - discounted_bricks) * full_price = total_spent ∧
    discounted_bricks * 2 = total_bricks :=
by sorry

end brick_ratio_proof_l348_34825


namespace inequality_sufficient_conditions_l348_34883

theorem inequality_sufficient_conditions (a b x y : ℝ) :
  (a < 0 ∧ 0 < b → x < y) ∧
  (0 < b ∧ b < a → x < y) ∧
  ¬(b < a ∧ a < 0 → x < y) ∧
  ¬(b < 0 ∧ 0 < a → x < y) :=
by sorry

end inequality_sufficient_conditions_l348_34883


namespace weight_gain_calculation_l348_34880

/-- Calculates the new weight of a person after muscle and fat gain --/
def new_weight (initial_weight : ℝ) (muscle_gain_percent : ℝ) (fat_gain_ratio : ℝ) : ℝ :=
  let muscle_gain := initial_weight * muscle_gain_percent
  let fat_gain := muscle_gain * fat_gain_ratio
  initial_weight + muscle_gain + fat_gain

/-- Theorem stating that for the given conditions, the new weight is 150 kg --/
theorem weight_gain_calculation :
  new_weight 120 0.2 0.25 = 150 := by
  sorry

#eval new_weight 120 0.2 0.25

end weight_gain_calculation_l348_34880


namespace five_coin_probability_l348_34867

def num_coins : ℕ := 5

def total_outcomes : ℕ := 2^num_coins

def favorable_outcomes : ℕ := 2

def probability : ℚ := favorable_outcomes / total_outcomes

theorem five_coin_probability :
  probability = 1 / 16 := by sorry

end five_coin_probability_l348_34867


namespace equation_graph_is_two_lines_l348_34820

theorem equation_graph_is_two_lines :
  ∀ x y : ℝ, (x + y - 1)^2 = x^2 + y^2 - 1 ↔ x = 1 ∨ y = 1 := by
  sorry

end equation_graph_is_two_lines_l348_34820


namespace arithmetic_sequence_common_ratio_l348_34844

theorem arithmetic_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + q) →  -- arithmetic sequence with common ratio q
  (∀ n, S n = (n * (2 * a 1 + (n - 1) * q)) / 2) →  -- sum formula for arithmetic sequence
  S 2 = 3 * a 2 + 2 →
  S 4 = 3 * a 4 + 2 →
  q = -1 ∨ q = 3/2 := by
sorry

end arithmetic_sequence_common_ratio_l348_34844


namespace polynomial_simplification_l348_34864

theorem polynomial_simplification (s : ℝ) :
  (2 * s^2 + 5 * s - 3) - (s^2 + 9 * s - 4) = s^2 - 4 * s + 1 := by
  sorry

end polynomial_simplification_l348_34864


namespace chord_bisection_range_l348_34826

theorem chord_bisection_range :
  ∀ (x₀ : ℝ),
  (∃ (A B : ℝ × ℝ),
    (A.1^2 + A.2^2 = 1) ∧
    (B.1^2 + B.2^2 = 1) ∧
    (∃ (t : ℝ), A.1 + t * (B.1 - A.1) = x₀ / 2 ∧ A.2 + t * (B.2 - A.2) = 1 - x₀) ∧
    ((B.2 - A.2) * (x₀ / 2) = (A.1 - B.1) * (1 - x₀))) →
  0 < x₀ ∧ x₀ < 8/5 :=
by sorry

end chord_bisection_range_l348_34826


namespace supernatural_gathering_handshakes_l348_34895

/-- The number of gremlins at the Supernatural Gathering -/
def num_gremlins : ℕ := 25

/-- The number of imps at the Supernatural Gathering -/
def num_imps : ℕ := 20

/-- The number of imps shaking hands amongst themselves -/
def num_imps_shaking : ℕ := num_imps / 2

/-- Calculate the number of handshakes between two groups -/
def handshakes_between (group1 : ℕ) (group2 : ℕ) : ℕ := group1 * group2

/-- Calculate the number of handshakes within a group -/
def handshakes_within (group : ℕ) : ℕ := group * (group - 1) / 2

/-- The total number of handshakes at the Supernatural Gathering -/
def total_handshakes : ℕ :=
  handshakes_within num_gremlins +
  handshakes_within num_imps_shaking +
  handshakes_between num_gremlins num_imps

theorem supernatural_gathering_handshakes :
  total_handshakes = 845 := by sorry

end supernatural_gathering_handshakes_l348_34895


namespace solve_equation_solve_inequalities_l348_34822

-- Part 1: Equation
theorem solve_equation (x : ℝ) : 
  x + 3 ≠ 0 → ((2 * x + 1) / (x + 3) = 1 / (x + 3) + 1 ↔ x = 3) :=
sorry

-- Part 2: System of Inequalities
theorem solve_inequalities (x : ℝ) :
  (2 * x - 2 < x ∧ 3 * (x + 1) ≥ 6) ↔ (1 ≤ x ∧ x < 2) :=
sorry

end solve_equation_solve_inequalities_l348_34822


namespace price_reduction_achieves_target_profit_l348_34832

/-- Represents the price reduction in yuan -/
def price_reduction : ℝ := 20

/-- Initial average daily sale in pieces -/
def initial_sale : ℝ := 20

/-- Initial profit per piece in yuan -/
def initial_profit : ℝ := 40

/-- Additional pieces sold per yuan of price reduction -/
def sales_increase_rate : ℝ := 2

/-- Target average daily profit in yuan -/
def target_profit : ℝ := 1200

/-- Theorem stating that the given price reduction achieves the target profit -/
theorem price_reduction_achieves_target_profit :
  (initial_profit - price_reduction) * (initial_sale + sales_increase_rate * price_reduction) = target_profit :=
by sorry

end price_reduction_achieves_target_profit_l348_34832


namespace basket_weight_l348_34879

theorem basket_weight (pear_weight : ℝ) (num_pears : ℕ) (total_weight : ℝ) 
  (h1 : pear_weight = 0.36)
  (h2 : num_pears = 30)
  (h3 : total_weight = 11.26) :
  total_weight - pear_weight * (num_pears : ℝ) = 0.46 := by
  sorry

end basket_weight_l348_34879


namespace total_cost_is_1646_2_l348_34888

/-- Calculates the total cost of fruits given their quantities, prices, discounts, and taxes --/
def total_cost_of_fruits 
  (grapes_kg : ℝ) (grapes_price : ℝ) (grapes_discount : ℝ) (grapes_tax : ℝ)
  (mangoes_kg : ℝ) (mangoes_price : ℝ) (mangoes_tax : ℝ)
  (apples_kg : ℝ) (apples_price : ℝ) (apples_discount : ℝ)
  (oranges_kg : ℝ) (oranges_price : ℝ) (oranges_tax : ℝ)
  (oranges_free_kg : ℝ) : ℝ :=
  let grapes_cost := grapes_kg * grapes_price * (1 - grapes_discount) * (1 + grapes_tax)
  let mangoes_cost := mangoes_kg * mangoes_price * (1 + mangoes_tax)
  let apples_cost := apples_kg * apples_price * (1 - apples_discount)
  let oranges_cost := (oranges_kg - oranges_free_kg) * oranges_price * (1 + oranges_tax)
  grapes_cost + mangoes_cost + apples_cost + oranges_cost

/-- Theorem stating that the total cost of fruits is 1646.2 given the specified conditions --/
theorem total_cost_is_1646_2 :
  total_cost_of_fruits 
    8 70 0.1 0.05
    9 50 0.08
    5 100 0.15
    6 40 0.03 1 = 1646.2 := by sorry

end total_cost_is_1646_2_l348_34888


namespace distinct_values_count_l348_34817

def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

def distinct_fractions (S : Finset ℕ) : Finset ℚ :=
  (S.product S).filter (fun (a, b) => a ≠ b)
    |>.image (fun (a, b) => (a : ℚ) / b)

theorem distinct_values_count :
  (distinct_fractions S).card = 22 :=
sorry

end distinct_values_count_l348_34817


namespace geometric_series_product_l348_34875

theorem geometric_series_product (y : ℝ) : 
  (∑' n, (1/3)^n) * (∑' n, (-1/3)^n) = ∑' n, (1/y)^n → y = 9 := by
  sorry

end geometric_series_product_l348_34875


namespace integral_odd_function_is_zero_l348_34842

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The definite integral of an odd function from -1 to 1 is zero -/
theorem integral_odd_function_is_zero (f : ℝ → ℝ) (hf : IsOdd f) :
    ∫ x in (-1)..1, f x = 0 := by
  sorry

end integral_odd_function_is_zero_l348_34842


namespace unique_integer_satisfying_conditions_l348_34882

theorem unique_integer_satisfying_conditions (x : ℤ) :
  1 < x ∧ x < 9 ∧
  2 < x ∧ x < 15 ∧
  7 > x ∧ x > -1 ∧
  4 > x ∧ x > 0 ∧
  x + 1 < 5 →
  x = 3 :=
by sorry

end unique_integer_satisfying_conditions_l348_34882


namespace dana_marcus_pencil_difference_l348_34897

/-- Given that Dana has 15 more pencils than Jayden, Jayden has twice as many pencils as Marcus,
    and Jayden has 20 pencils, prove that Dana has 25 more pencils than Marcus. -/
theorem dana_marcus_pencil_difference :
  ∀ (dana jayden marcus : ℕ),
  dana = jayden + 15 →
  jayden = 2 * marcus →
  jayden = 20 →
  dana - marcus = 25 := by
sorry

end dana_marcus_pencil_difference_l348_34897


namespace rope_cutting_impossibility_l348_34800

theorem rope_cutting_impossibility : ¬ ∃ (n : ℕ), 5 + 4 * n = 2019 := by
  sorry

end rope_cutting_impossibility_l348_34800


namespace defective_smartphone_probability_l348_34871

/-- The probability of selecting two defective smartphones from a shipment -/
theorem defective_smartphone_probability
  (total : ℕ) (defective : ℕ) (h1 : total = 220) (h2 : defective = 84) :
  (defective : ℚ) / total * ((defective - 1) : ℚ) / (total - 1) =
  (84 : ℚ) / 220 * (83 : ℚ) / 219 :=
by sorry

end defective_smartphone_probability_l348_34871


namespace sequence_property_l348_34854

theorem sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → a n > 0) →
  (∀ n : ℕ, n > 0 → S n = (1 / 2) * (a n + 1 / a n)) →
  ∀ n : ℕ, n > 0 → S n = Real.sqrt n :=
by sorry

end sequence_property_l348_34854


namespace perfect_square_polynomial_l348_34843

theorem perfect_square_polynomial (x : ℤ) : 
  ∃ (y : ℤ), x^4 + x^3 + x^2 + x + 1 = y^2 ↔ x = -1 ∨ x = 0 ∨ x = 3 := by
sorry

end perfect_square_polynomial_l348_34843


namespace parallel_vectors_angle_l348_34835

theorem parallel_vectors_angle (α : Real) :
  let a : Fin 2 → Real := ![3/2, Real.sin α]
  let b : Fin 2 → Real := ![1, 1/3]
  (∃ (k : Real), a = k • b) →
  α = π/6 := by
  sorry

end parallel_vectors_angle_l348_34835


namespace tangent_line_d_value_l348_34827

-- Define the line equation
def line (x y d : ℝ) : Prop := y = 3 * x + d

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the tangency condition
def is_tangent (d : ℝ) : Prop :=
  ∃ x y : ℝ, line x y d ∧ parabola x y ∧
  ∀ x' y' : ℝ, line x' y' d → parabola x' y' → (x', y') = (x, y)

-- Theorem statement
theorem tangent_line_d_value :
  ∃ d : ℝ, is_tangent d ∧ d = 3 :=
sorry

end tangent_line_d_value_l348_34827


namespace special_triangle_angles_l348_34845

/-- A triangle with a special property -/
structure SpecialTriangle where
  /-- The angle at vertex A -/
  angle_a : Real
  /-- The angle at vertex B -/
  angle_b : Real
  /-- The angle at vertex C -/
  angle_c : Real
  /-- The sum of angles is 180° -/
  angle_sum : angle_a + angle_b + angle_c = Real.pi
  /-- The altitude, angle bisector, and median from vertex A divide the angle into four equal parts -/
  special_property : ∃ (α : Real), angle_a = 4 * α ∧ 0 < α ∧ α < Real.pi / 2

/-- The theorem stating the angles of the special triangle -/
theorem special_triangle_angles (t : SpecialTriangle) :
  t.angle_a = Real.pi / 2 ∧ 
  t.angle_b = Real.pi / 8 ∧ 
  t.angle_c = 3 * Real.pi / 8 := by
  sorry

#check special_triangle_angles

end special_triangle_angles_l348_34845


namespace product_with_9999_l348_34859

theorem product_with_9999 : ∃ x : ℕ, x * 9999 = 4691110842 ∧ x = 469211 := by
  sorry

end product_with_9999_l348_34859


namespace diagonal_exit_return_l348_34813

/-- A path on a 10x10 grid that visits each cell exactly once -/
def HamiltonianPath : Type := Fin 100 → Fin 100

/-- A function that checks if two cells are adjacent -/
def isAdjacent (a b : Fin 100) : Prop := sorry

/-- A function that checks if a cell is on the main diagonal -/
def isOnDiagonal (a : Fin 100) : Prop := sorry

/-- The theorem stating that for any Hamiltonian path on a 10x10 grid, 
    there must be a point where the path leaves and immediately returns to the diagonal -/
theorem diagonal_exit_return (path : HamiltonianPath) : 
  ∃ (i : Fin 99), isOnDiagonal (path i) ∧ 
                  ¬isOnDiagonal (path (i + 1)) ∧ 
                  isOnDiagonal (path (i + 2)) := by
  sorry

end diagonal_exit_return_l348_34813


namespace jack_total_miles_l348_34846

/-- Calculates the total miles driven given the number of years and miles driven per four months. -/
def total_miles_driven (years : ℕ) (miles_per_four_months : ℕ) : ℕ :=
  years * 3 * miles_per_four_months

/-- Proves that Jack has driven 999,000 miles given the conditions. -/
theorem jack_total_miles :
  total_miles_driven 9 37000 = 999000 := by
  sorry

end jack_total_miles_l348_34846


namespace student_presentation_time_l348_34812

theorem student_presentation_time 
  (num_students : ℕ) 
  (period_length : ℕ) 
  (num_periods : ℕ) 
  (h1 : num_students = 32) 
  (h2 : period_length = 40) 
  (h3 : num_periods = 4) : 
  (num_periods * period_length) / num_students = 5 := by
  sorry

end student_presentation_time_l348_34812


namespace system_solution_l348_34869

theorem system_solution :
  ∀ x y : ℝ, x > 0 ∧ y > 0 →
  (2 * x - Real.sqrt (x * y) - 4 * Real.sqrt (x / y) + 2 = 0) ∧
  (2 * x^2 + x^2 * y^4 = 18 * y^2) →
  ((x = 2 ∧ y = 2) ∨ (x = Real.sqrt (Real.sqrt 286) / 4 ∧ y = Real.sqrt (Real.sqrt 286))) :=
by sorry

end system_solution_l348_34869


namespace library_visits_ratio_l348_34885

theorem library_visits_ratio :
  ∀ (william_weekly_visits jason_monthly_visits : ℕ),
    william_weekly_visits = 2 →
    jason_monthly_visits = 32 →
    (jason_monthly_visits : ℚ) / (4 * william_weekly_visits) = 4 := by
  sorry

end library_visits_ratio_l348_34885


namespace positive_numbers_inequality_l348_34837

theorem positive_numbers_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 2*a*b) :
  (a + 2*b ≥ 4) ∧ (a^2 + 4*b^2 ≥ 8) := by
  sorry

end positive_numbers_inequality_l348_34837


namespace integer_solutions_count_l348_34862

theorem integer_solutions_count : ∃! (n : ℕ), ∃ (S : Finset ℤ),
  (∀ y ∈ S, (-4:ℤ) * y ≥ 2 * y + 10 ∧
            (-3:ℤ) * y ≤ 15 ∧
            (-5:ℤ) * y ≥ 3 * y + 24 ∧
            y ≤ -1) ∧
  (∀ y : ℤ, (-4:ℤ) * y ≥ 2 * y + 10 ∧
            (-3:ℤ) * y ≤ 15 ∧
            (-5:ℤ) * y ≥ 3 * y + 24 ∧
            y ≤ -1 → y ∈ S) ∧
  Finset.card S = n ∧ n = 3 :=
sorry

end integer_solutions_count_l348_34862


namespace inequality_system_solution_l348_34811

theorem inequality_system_solution :
  ∀ x : ℝ, (5 * (x - 1) ≤ x + 3 ∧ (x + 1) / 2 < 2 * x) ↔ (1/3 < x ∧ x ≤ 2) :=
by sorry

end inequality_system_solution_l348_34811


namespace john_reading_probability_l348_34838

/-- Probability of John reading a book on Monday -/
def prob_read_monday : ℝ := 0.8

/-- Probability of John playing soccer on Tuesday -/
def prob_soccer_tuesday : ℝ := 0.5

/-- Independence of activities -/
axiom activities_independent : True

/-- John reads every day when he decides to play soccer on the previous day -/
axiom reads_after_soccer : True

/-- Probability of John reading a book on both Monday and Tuesday -/
def prob_read_both_days : ℝ := prob_read_monday * prob_soccer_tuesday * prob_read_monday

theorem john_reading_probability :
  prob_read_both_days = 0.32 :=
sorry

end john_reading_probability_l348_34838


namespace similar_triangle_perimeter_l348_34852

theorem similar_triangle_perimeter :
  ∀ (a b c d e f : ℝ),
  a^2 + b^2 = c^2 →  -- right triangle condition
  d^2 + e^2 = f^2 →  -- right triangle condition
  a / d = b / e →    -- similarity condition
  a / d = c / f →    -- similarity condition
  (a, b, c) = (6, 8, 10) →  -- given smaller triangle
  f = 30 →           -- given longest side of larger triangle
  d + e + f = 72 :=  -- perimeter of larger triangle
by sorry

end similar_triangle_perimeter_l348_34852


namespace heaviest_weight_proof_l348_34814

/-- Represents a set of four weights in an increasing geometric progression -/
structure GeometricWeights (a q : ℝ) :=
  (a_pos : a > 0)
  (q_gt_one : q > 1)

/-- Proves that the heaviest weight is heavier than the sum of any two other weights -/
theorem heaviest_weight_proof {a q : ℝ} (gw : GeometricWeights a q) :
  a * q^3 > a + a * q ∧ 
  a * q^3 > a + a * q^2 ∧ 
  a * q^3 > a * q + a * q^2 :=
sorry

end heaviest_weight_proof_l348_34814


namespace remainder_sum_l348_34861

theorem remainder_sum (n : ℤ) : n % 20 = 14 → (n % 4 + n % 5 = 6) := by
  sorry

end remainder_sum_l348_34861


namespace book_distribution_count_l348_34856

/-- The number of ways to distribute books between the library and checked-out status -/
def distribution_count (total : ℕ) (min_in_library : ℕ) (max_in_library : ℕ) : ℕ :=
  (max_in_library - min_in_library + 1)

/-- Theorem stating the number of ways to distribute 8 books with given constraints -/
theorem book_distribution_count :
  distribution_count 8 2 6 = 5 := by sorry

end book_distribution_count_l348_34856


namespace count_integer_pairs_l348_34804

theorem count_integer_pairs : 
  ∃ (count : ℕ), 
    (2^2876 < 3^1250 ∧ 3^1250 < 2^2877) →
    count = (Finset.filter 
      (λ (pair : ℕ × ℕ) => 
        let (m, n) := pair
        1 ≤ m ∧ m ≤ 2875 ∧ 3^n < 2^m ∧ 2^m < 2^(m+3) ∧ 2^(m+3) < 3^(n+1))
      (Finset.product (Finset.range 2876) (Finset.range (1250 + 1)))).card ∧
    count = 3750 :=
by sorry

end count_integer_pairs_l348_34804


namespace parabola_f_value_l348_34884

/-- A parabola with equation y = dx² + ex + f -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.d * x^2 + p.e * x + p.f

/-- The vertex of a parabola -/
structure Vertex where
  x : ℝ
  y : ℝ

theorem parabola_f_value (p : Parabola) (v : Vertex) :
  p.y_at 3 = -5 →  -- vertex at (3, -5)
  p.y_at 5 = -1 →  -- passes through (5, -1)
  p.f = 4 := by
    sorry

end parabola_f_value_l348_34884


namespace bicycle_price_after_discounts_bicycle_price_proof_l348_34849

theorem bicycle_price_after_discounts (original_price : ℝ) 
  (first_discount_percent : ℝ) (second_discount_percent : ℝ) : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_percent / 100)
  let final_price := price_after_first_discount * (1 - second_discount_percent / 100)
  final_price

theorem bicycle_price_proof : 
  bicycle_price_after_discounts 200 20 25 = 120 := by
  sorry

end bicycle_price_after_discounts_bicycle_price_proof_l348_34849


namespace expression_evaluation_l348_34886

theorem expression_evaluation : 3 * (-3)^4 + 2 * (-3)^3 + (-3)^2 + 3^2 + 2 * 3^3 + 3 * 3^4 = 504 := by
  sorry

end expression_evaluation_l348_34886


namespace ordering_of_powers_l348_34857

theorem ordering_of_powers : 5^(1/5) > 0.5^(1/5) ∧ 0.5^(1/5) > 0.5^2 := by
  sorry

end ordering_of_powers_l348_34857


namespace store_comparison_l348_34808

/-- Represents the cost difference between Store B and Store A -/
def cost_difference (x : ℝ) : ℝ := 520 - 2.5 * x

theorem store_comparison (x : ℝ) (h : x > 40) :
  cost_difference x = 520 - 2.5 * x ∧
  cost_difference 80 > 0 :=
sorry

#check store_comparison

end store_comparison_l348_34808


namespace trig_simplification_l348_34873

theorem trig_simplification (α : ℝ) : 
  (2 * (Real.cos α)^2 - 1) / (2 * Real.tan (π/4 - α) * (Real.sin (π/4 + α))^2) = 1 := by
  sorry

end trig_simplification_l348_34873


namespace money_division_l348_34810

theorem money_division (r s t u : ℝ) (h1 : r / s = 2.5 / 3.5) (h2 : s / t = 3.5 / 7.5) 
  (h3 : t / u = 7.5 / 9.8) (h4 : t - s = 4500) : u - r = 8212.5 := by
  sorry

end money_division_l348_34810


namespace min_cone_cylinder_volume_ratio_l348_34870

/-- The minimum ratio of the volume of a cone circumscribed around a sphere
    to the volume of a cylinder circumscribed around the same sphere -/
theorem min_cone_cylinder_volume_ratio (r : ℝ) (hr : r > 0) :
  ∃ (V₁ V₂ : ℝ),
    (∀ (Vc Vn : ℝ),
      (Vc = 2 * π * r^3) →  -- Volume of circumscribed cylinder
      (∃ (R m : ℝ), Vn = (1/3) * π * R^2 * m ∧ 
        R^2 / m^2 = r^2 / (m * (m - 2*r))) →  -- Volume and geometry of circumscribed cone
      V₂ / V₁ ≤ Vn / Vc) ∧
    V₂ / V₁ = 4/3 :=
sorry

end min_cone_cylinder_volume_ratio_l348_34870


namespace symmetric_function_is_exponential_l348_34855

-- Define the logarithm function for base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Define the symmetry condition
def symmetric_to_log3 (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → (f x = y ↔ log3 y = x)

-- Theorem statement
theorem symmetric_function_is_exponential (f : ℝ → ℝ) :
  symmetric_to_log3 f → (∀ x : ℝ, f x = 3^x) :=
sorry

end symmetric_function_is_exponential_l348_34855


namespace greatest_three_digit_multiple_of_23_l348_34821

theorem greatest_three_digit_multiple_of_23 : 
  ∀ n : ℕ, n ≤ 999 → n % 23 = 0 → n ≤ 989 :=
by
  sorry

end greatest_three_digit_multiple_of_23_l348_34821


namespace maria_spent_60_dollars_l348_34860

/-- The amount Maria spent on flowers -/
def total_spent (price_per_flower : ℕ) (roses : ℕ) (daisies : ℕ) : ℕ :=
  price_per_flower * (roses + daisies)

/-- Proof that Maria spent 60 dollars on flowers -/
theorem maria_spent_60_dollars : total_spent 6 7 3 = 60 := by
  sorry

end maria_spent_60_dollars_l348_34860


namespace binary_10101_equals_21_l348_34807

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10101_equals_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end binary_10101_equals_21_l348_34807
