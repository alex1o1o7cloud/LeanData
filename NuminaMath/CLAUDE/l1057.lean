import Mathlib

namespace fayes_age_l1057_105746

/-- Given the ages of four people and their relationships, prove Faye's age -/
theorem fayes_age 
  (C D E F : ℕ) -- Chad's, Diana's, Eduardo's, and Faye's ages
  (h1 : D = E - 2) -- Diana is two years younger than Eduardo
  (h2 : E = C + 5) -- Eduardo is five years older than Chad
  (h3 : F = C + 4) -- Faye is four years older than Chad
  (h4 : D = 15) -- Diana is 15 years old
  : F = 16 := by
  sorry

end fayes_age_l1057_105746


namespace hot_air_balloon_theorem_l1057_105787

def hot_air_balloon_problem (initial_balloons : ℕ) : ℕ :=
  let after_first_30_min := initial_balloons - initial_balloons / 5
  let after_next_hour := after_first_30_min - (after_first_30_min * 3) / 10
  let durable_balloons := after_next_hour / 10
  let regular_balloons := after_next_hour - durable_balloons
  let blown_up_regular := min regular_balloons (2 * (initial_balloons - after_next_hour))
  durable_balloons

theorem hot_air_balloon_theorem :
  hot_air_balloon_problem 200 = 11 := by
  sorry

end hot_air_balloon_theorem_l1057_105787


namespace unique_integral_solution_l1057_105747

/-- Given a system of equations, prove that there is only one integral solution -/
theorem unique_integral_solution :
  ∃! (x y z : ℤ),
    (z : ℝ) ^ (x : ℝ) = (y : ℝ) ^ (2 * x : ℝ) ∧
    (2 : ℝ) ^ (z : ℝ) = 2 * (8 : ℝ) ^ (x : ℝ) ∧
    x + y + z = 18 ∧
    x = 8 ∧ y = 5 ∧ z = 25 := by
  sorry

end unique_integral_solution_l1057_105747


namespace complement_union_theorem_l1057_105785

def U : Finset Nat := {1,2,3,4,5,6,7}
def A : Finset Nat := {2,4,5,7}
def B : Finset Nat := {3,4,5}

theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {1,2,3,6,7} := by sorry

end complement_union_theorem_l1057_105785


namespace arithmetic_geometric_ratio_l1057_105793

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d ≠ 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d

/-- Condition for terms forming a geometric sequence -/
def isGeometric (s : ArithmeticSequence) : Prop :=
  (s.a 3)^2 = (s.a 1) * (s.a 9)

theorem arithmetic_geometric_ratio
  (s : ArithmeticSequence)
  (h : isGeometric s) :
  (s.a 1 + s.a 3 + s.a 9) / (s.a 2 + s.a 4 + s.a 10) = 13 / 16 := by
  sorry

end arithmetic_geometric_ratio_l1057_105793


namespace max_transition_BC_l1057_105726

def channel_A_transition : ℕ := 51
def channel_B_transition : ℕ := 63
def channel_C_transition : ℕ := 63

theorem max_transition_BC : 
  max channel_B_transition channel_C_transition = 63 := by
  sorry

end max_transition_BC_l1057_105726


namespace koala_weight_in_grams_l1057_105715

-- Define the conversion rate from kg to g
def kg_to_g : ℕ → ℕ := (· * 1000)

-- Define the weight of the baby koala
def koala_weight_kg : ℕ := 2
def koala_weight_extra_g : ℕ := 460

-- Theorem: The total weight of the baby koala in grams is 2460
theorem koala_weight_in_grams : 
  kg_to_g koala_weight_kg + koala_weight_extra_g = 2460 := by
  sorry

end koala_weight_in_grams_l1057_105715


namespace art_collection_cost_l1057_105700

/-- The total cost of John's art collection --/
def total_cost (first_three_cost : ℝ) (fourth_piece_cost : ℝ) : ℝ :=
  first_three_cost + fourth_piece_cost

/-- The cost of the fourth piece of art --/
def fourth_piece_cost (single_piece_cost : ℝ) : ℝ :=
  single_piece_cost * 1.5

theorem art_collection_cost :
  ∀ (single_piece_cost : ℝ),
    single_piece_cost > 0 →
    single_piece_cost * 3 = 45000 →
    total_cost (single_piece_cost * 3) (fourth_piece_cost single_piece_cost) = 67500 := by
  sorry

end art_collection_cost_l1057_105700


namespace jack_payback_l1057_105704

/-- The amount borrowed by Jack from Jill -/
def principal : ℝ := 1200

/-- The interest rate on the loan -/
def interest_rate : ℝ := 0.1

/-- The total amount Jack will pay back -/
def total_amount : ℝ := principal * (1 + interest_rate)

/-- Theorem stating that Jack will pay back $1320 -/
theorem jack_payback : total_amount = 1320 := by
  sorry

end jack_payback_l1057_105704


namespace xiaoxiang_age_problem_l1057_105731

theorem xiaoxiang_age_problem :
  let xiaoxiang_age : ℕ := 5
  let father_age : ℕ := 48
  let mother_age : ℕ := 42
  let years_passed : ℕ := 15
  (father_age + years_passed) + (mother_age + years_passed) = 6 * (xiaoxiang_age + years_passed) :=
by
  sorry

end xiaoxiang_age_problem_l1057_105731


namespace sugar_solution_percentage_l1057_105744

theorem sugar_solution_percentage (initial_sugar_percentage : ℝ) 
  (replaced_fraction : ℝ) (final_sugar_percentage : ℝ) :
  initial_sugar_percentage = 22 →
  replaced_fraction = 1/4 →
  final_sugar_percentage = 35 →
  let remaining_fraction := 1 - replaced_fraction
  let initial_sugar := initial_sugar_percentage * remaining_fraction
  let added_sugar := final_sugar_percentage - initial_sugar
  added_sugar / replaced_fraction = 74 := by
sorry

end sugar_solution_percentage_l1057_105744


namespace specific_trapezoid_area_l1057_105769

/-- A trapezoid with an inscribed circle -/
structure InscribedTrapezoid where
  -- The length of segment BL
  BL : ℝ
  -- The length of segment CL
  CL : ℝ
  -- The length of side AB
  AB : ℝ
  -- Assumption that BL is positive
  BL_pos : BL > 0
  -- Assumption that CL is positive
  CL_pos : CL > 0
  -- Assumption that AB is positive
  AB_pos : AB > 0

/-- The area of a trapezoid with an inscribed circle -/
def area (t : InscribedTrapezoid) : ℝ :=
  -- Define the area function here
  sorry

/-- Theorem: The area of the specific trapezoid is 6.75 -/
theorem specific_trapezoid_area :
  ∀ t : InscribedTrapezoid,
  t.BL = 4 → t.CL = 1/4 → t.AB = 6 →
  area t = 6.75 := by
  sorry

end specific_trapezoid_area_l1057_105769


namespace gcd_p4_minus_1_l1057_105765

theorem gcd_p4_minus_1 (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) : 
  ∃ k : ℕ, p^4 - 1 = 240 * k := by
sorry

end gcd_p4_minus_1_l1057_105765


namespace fourth_root_over_sixth_root_of_six_l1057_105714

theorem fourth_root_over_sixth_root_of_six (x : ℝ) (h : x = 6) :
  (x^(1/4)) / (x^(1/6)) = x^(1/12) :=
by sorry

end fourth_root_over_sixth_root_of_six_l1057_105714


namespace probability_sum_three_l1057_105797

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numSides * numSides

/-- The number of ways to roll a sum of 3 with two dice -/
def favorableOutcomes : ℕ := 2

/-- The probability of rolling a sum of 3 with two fair six-sided dice -/
theorem probability_sum_three (numSides : ℕ) (totalOutcomes : ℕ) (favorableOutcomes : ℕ) :
  numSides = 6 →
  totalOutcomes = numSides * numSides →
  favorableOutcomes = 2 →
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 18 := by
  sorry

end probability_sum_three_l1057_105797


namespace quadratic_equation_properties_l1057_105738

-- Define the set S with exactly two subsets
def S (a b : ℝ) := {x : ℝ | x^2 + a*x + b = 0}

-- Theorem statement
theorem quadratic_equation_properties
  (a b : ℝ)
  (h_a_pos : a > 0)
  (h_two_subsets : ∃ (x y : ℝ), x ≠ y ∧ S a b = {x, y}) :
  (a^2 - b^2 ≤ 4) ∧
  (a^2 + 1/b ≥ 4) ∧
  (∀ c x₁ x₂ : ℝ, (∀ x : ℝ, x^2 + a*x + b < c ↔ x₁ < x ∧ x < x₂) →
    |x₁ - x₂| = 4 → c = 4) :=
by sorry

end quadratic_equation_properties_l1057_105738


namespace quiz_score_average_l1057_105760

theorem quiz_score_average (n : ℕ) (initial_avg : ℚ) (dropped_score : ℚ) : 
  n = 16 → 
  initial_avg = 62.5 → 
  dropped_score = 55 → 
  let total_score := n * initial_avg
  let remaining_total := total_score - dropped_score
  let new_avg := remaining_total / (n - 1)
  new_avg = 63 := by sorry

end quiz_score_average_l1057_105760


namespace equation_solution_l1057_105741

theorem equation_solution : 
  {x : ℝ | 3 * (x + 2) = x * (x + 2)} = {-2, 3} := by sorry

end equation_solution_l1057_105741


namespace weeks_to_save_for_games_l1057_105772

/-- Calculates the minimum number of weeks required to save for a games console and a video game -/
theorem weeks_to_save_for_games (console_cost video_game_cost initial_savings weekly_allowance : ℚ)
  (tax_rate : ℚ) (h_console : console_cost = 282)
  (h_video_game : video_game_cost = 75) (h_tax : tax_rate = 0.1)
  (h_initial : initial_savings = 42) (h_allowance : weekly_allowance = 24) :
  ⌈(console_cost + video_game_cost * (1 + tax_rate) - initial_savings) / weekly_allowance⌉ = 14 := by
sorry

end weeks_to_save_for_games_l1057_105772


namespace decimal_product_sum_l1057_105725

-- Define the structure for our decimal representation
structure DecimalPair :=
  (whole : Nat)
  (decimal : Nat)

-- Define the multiplication operation for DecimalPair
def multiply_decimal_pairs (x y : DecimalPair) : Rat :=
  (x.whole + x.decimal / 10 : Rat) * (y.whole + y.decimal / 10 : Rat)

-- Define the addition operation for DecimalPair
def add_decimal_pairs (x y : DecimalPair) : Rat :=
  (x.whole + x.decimal / 10 : Rat) + (y.whole + y.decimal / 10 : Rat)

-- The main theorem
theorem decimal_product_sum (a b c d : Nat) :
  (a ≠ 0) → (b ≠ 0) → (c ≠ 0) → (d ≠ 0) →
  (a ≤ 9) → (b ≤ 9) → (c ≤ 9) → (d ≤ 9) →
  multiply_decimal_pairs ⟨a, b⟩ ⟨c, d⟩ = (56 : Rat) / 10 →
  add_decimal_pairs ⟨a, b⟩ ⟨c, d⟩ = (51 : Rat) / 10 := by
sorry

end decimal_product_sum_l1057_105725


namespace quadratic_roots_sum_product_l1057_105712

theorem quadratic_roots_sum_product (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ + 4 = 5*x₁ + 6) ∧ 
  (x₂^2 + 2*x₂ + 4 = 5*x₂ + 6) → 
  x₁*x₂ + x₁ + x₂ = 1 := by
sorry

end quadratic_roots_sum_product_l1057_105712


namespace equation_solution_is_origin_l1057_105780

theorem equation_solution_is_origin (x y : ℝ) : 
  (x + y)^2 = 2 * (x^2 + y^2) ↔ x = 0 ∧ y = 0 := by
  sorry

end equation_solution_is_origin_l1057_105780


namespace water_usage_difference_l1057_105706

theorem water_usage_difference (total_water plants_water : ℕ) : 
  total_water = 65 →
  plants_water < 14 →
  24 * 2 = 65 - 14 - plants_water →
  7 - plants_water = 4 := by
  sorry

end water_usage_difference_l1057_105706


namespace locus_and_fixed_point_l1057_105719

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the locus C
def C : Set (ℝ × ℝ) := {p | p.1^2/4 - p.2^2 = 1 ∧ p.1 ≠ 2 ∧ p.1 ≠ -2}

-- Define the line x = 1
def line_x_eq_1 : Set (ℝ × ℝ) := {p | p.1 = 1}

-- Define the property of point M
def is_valid_M (M : ℝ × ℝ) : Prop :=
  let slope_AM := (M.2 - A.2) / (M.1 - A.1)
  let slope_BM := (M.2 - B.2) / (M.1 - B.1)
  slope_AM * slope_BM = 1/4

-- Main theorem
theorem locus_and_fixed_point :
  (∀ M, is_valid_M M → M ∈ C) ∧
  (∀ T ∈ line_x_eq_1, 
    ∃ P Q, P ∈ C ∧ Q ∈ C ∧ 
    (P.2 - A.2) / (P.1 - A.1) = (T.2 - A.2) / (T.1 - A.1) ∧
    (Q.2 - B.2) / (Q.1 - B.1) = (T.2 - B.2) / (T.1 - B.1) ∧
    (Q.2 - P.2) / (Q.1 - P.1) = (0 - P.2) / (4 - P.1)) :=
sorry

end locus_and_fixed_point_l1057_105719


namespace fruit_problem_solution_l1057_105799

def fruit_problem (cost_A cost_B : ℝ) (weight_diff : ℝ) (total_weight : ℝ) 
  (selling_price_A selling_price_B : ℝ) : Prop :=
  ∃ (weight_A weight_B cost_per_kg_A cost_per_kg_B : ℝ),
    cost_A = weight_A * cost_per_kg_A ∧
    cost_B = weight_B * cost_per_kg_B ∧
    cost_per_kg_B = 1.5 * cost_per_kg_A ∧
    weight_A = weight_B + weight_diff ∧
    (∀ a b, a + b = total_weight ∧ a ≥ 3 * b →
      (13 - cost_per_kg_A) * a + (20 - cost_per_kg_B) * b ≤
      (13 - cost_per_kg_A) * 75 + (20 - cost_per_kg_B) * 25) ∧
    cost_per_kg_A = 10 ∧
    cost_per_kg_B = 15

theorem fruit_problem_solution :
  fruit_problem 300 300 10 100 13 20 :=
sorry

end fruit_problem_solution_l1057_105799


namespace work_completion_time_l1057_105767

theorem work_completion_time
  (A_work : ℝ)
  (B_work : ℝ)
  (C_work : ℝ)
  (h1 : A_work = 1 / 3)
  (h2 : B_work + C_work = 1 / 3)
  (h3 : B_work = 1 / 6)
  : 1 / (A_work + C_work) = 2 :=
by
  sorry

#check work_completion_time

end work_completion_time_l1057_105767


namespace prime_squares_sum_theorem_l1057_105742

theorem prime_squares_sum_theorem (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ (x y z : ℕ), p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) := by
  sorry

end prime_squares_sum_theorem_l1057_105742


namespace arithmetic_progression_coprime_terms_l1057_105705

theorem arithmetic_progression_coprime_terms :
  ∃ (a r : ℕ), 
    (∀ i j, 0 ≤ i ∧ i < j ∧ j < 100 → 
      (a + i * r).gcd (a + j * r) = 1) ∧
    (∀ i, 0 ≤ i ∧ i < 99 → a + i * r < a + (i + 1) * r) :=
by sorry

end arithmetic_progression_coprime_terms_l1057_105705


namespace school_club_profit_l1057_105776

/-- Calculates the profit for a school club selling cookies -/
def cookie_profit (num_cookies : ℕ) (buy_rate : ℚ) (sell_price : ℚ) (handling_fee : ℚ) : ℚ :=
  let cost := (num_cookies : ℚ) / buy_rate + handling_fee
  let revenue := (num_cookies : ℚ) * sell_price
  revenue - cost

/-- The profit for the school club selling cookies is $190 -/
theorem school_club_profit :
  cookie_profit 1200 3 (1/2) 10 = 190 := by
  sorry

end school_club_profit_l1057_105776


namespace complex_pure_imaginary_a_l1057_105745

def i : ℂ := Complex.I

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_pure_imaginary_a (a : ℝ) :
  is_pure_imaginary ((2 + a * i) / (2 - i)) → a = 4 := by
  sorry

end complex_pure_imaginary_a_l1057_105745


namespace angle_B_in_triangle_l1057_105707

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem angle_B_in_triangle (t : Triangle) :
  t.a = 4 →
  t.b = 2 * Real.sqrt 2 →
  t.A = π / 4 →
  t.B = π / 6 :=
by
  sorry

end angle_B_in_triangle_l1057_105707


namespace f_max_min_difference_l1057_105752

noncomputable def f (x : ℝ) : ℝ := |Real.sin x| + max (Real.sin (2 * x)) 0 + |Real.cos x|

theorem f_max_min_difference :
  (⨆ x, f x) - (⨅ x, f x) = Real.sqrt 2 := by sorry

end f_max_min_difference_l1057_105752


namespace homework_difference_l1057_105702

theorem homework_difference (reading_pages math_pages biology_pages : ℕ) 
  (h1 : reading_pages = 4)
  (h2 : math_pages = 7)
  (h3 : biology_pages = 19) :
  math_pages - reading_pages = 3 :=
by sorry

end homework_difference_l1057_105702


namespace steven_route_count_l1057_105703

def central_park_routes : ℕ := 
  let home_to_sw_corner := (Nat.choose 5 2)
  let ne_corner_to_office := (Nat.choose 6 3)
  let park_diagonals := 2
  home_to_sw_corner * park_diagonals * ne_corner_to_office

theorem steven_route_count : central_park_routes = 400 := by
  sorry

end steven_route_count_l1057_105703


namespace product_of_powers_equals_1260_l1057_105778

theorem product_of_powers_equals_1260 (w x y z : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z = 1260) : 
  3*w + 4*x + 2*y + 2*z = 18 := by
  sorry

end product_of_powers_equals_1260_l1057_105778


namespace intersection_A_complement_B_l1057_105748

def U : Set Int := Set.univ

def A : Set Int := {-1, 0, 1, 2}

def B : Set Int := {x | x^2 ≠ x}

theorem intersection_A_complement_B : A ∩ (U \ B) = {-1, 2} := by
  sorry

end intersection_A_complement_B_l1057_105748


namespace pipe_problem_l1057_105784

theorem pipe_problem (fill_rate_A fill_rate_B empty_rate_C : ℝ) 
  (h_A : fill_rate_A = 1 / 20)
  (h_B : fill_rate_B = 1 / 30)
  (h_C : empty_rate_C > 0)
  (h_fill : 2 * fill_rate_A + 2 * fill_rate_B - 2 * empty_rate_C = 1) :
  empty_rate_C = 1 / 3 :=
by sorry

end pipe_problem_l1057_105784


namespace quadratic_minimum_quadratic_minimum_achieved_l1057_105757

theorem quadratic_minimum (x : ℝ) : 7 * x^2 - 28 * x + 2015 ≥ 1987 := by sorry

theorem quadratic_minimum_achieved : ∃ x : ℝ, 7 * x^2 - 28 * x + 2015 = 1987 := by sorry

end quadratic_minimum_quadratic_minimum_achieved_l1057_105757


namespace fuel_cost_theorem_l1057_105773

theorem fuel_cost_theorem (x : ℝ) : 
  (x / 4 - x / 6 = 8) → x = 96 := by
  sorry

end fuel_cost_theorem_l1057_105773


namespace smallest_number_in_sequence_l1057_105751

theorem smallest_number_in_sequence (a b c d : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- Four positive integers
  (a + b + c + d) / 4 = 30 →  -- Arithmetic mean is 30
  b = 33 →  -- Second largest is 33
  d = b + 3 →  -- Largest is 3 more than second largest
  a < b ∧ b < c ∧ c < d →  -- Ascending order
  a = 17 :=  -- The smallest number is 17
by sorry

end smallest_number_in_sequence_l1057_105751


namespace perfect_square_between_prime_sums_l1057_105782

def S (n : ℕ) : ℕ := sorry

theorem perfect_square_between_prime_sums (n : ℕ) :
  ∃ k : ℕ, S n < k^2 ∧ k^2 < S (n + 1) :=
sorry

end perfect_square_between_prime_sums_l1057_105782


namespace collectors_edition_dolls_l1057_105796

/-- Prove that given the conditions, Ivy and Luna have 30 collectors edition dolls combined -/
theorem collectors_edition_dolls (dina ivy luna : ℕ) : 
  dina = 60 →
  dina = 2 * ivy →
  ivy = luna + 10 →
  dina + ivy + luna = 150 →
  (2 * ivy / 3 : ℚ) + (luna / 2 : ℚ) = 30 := by
  sorry

end collectors_edition_dolls_l1057_105796


namespace unique_values_theorem_l1057_105749

/-- Definition of the sequence P_n -/
def P (a b : ℝ) : ℕ → ℝ × ℝ
  | 0 => (1, 0)
  | n + 1 => let (x, y) := P a b n; (a * x - b * y, b * x + a * y)

/-- Condition (i): P_0 = P_6 -/
def condition_i (a b : ℝ) : Prop := P a b 0 = P a b 6

/-- Condition (ii): All P_0, P_1, P_2, P_3, P_4, P_5 are distinct -/
def condition_ii (a b : ℝ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < j ∧ j < 6 → P a b i ≠ P a b j

/-- The main theorem -/
theorem unique_values_theorem :
  {(a, b) : ℝ × ℝ | condition_i a b ∧ condition_ii a b} =
  {(1/2, Real.sqrt 3/2), (1/2, -Real.sqrt 3/2)} :=
sorry

end unique_values_theorem_l1057_105749


namespace stratified_sample_female_count_l1057_105750

theorem stratified_sample_female_count (male_count : ℕ) (female_count : ℕ) (sample_size : ℕ) :
  male_count = 48 →
  female_count = 36 →
  sample_size = 35 →
  (female_count : ℚ) / (male_count + female_count) * sample_size = 15 := by
  sorry

end stratified_sample_female_count_l1057_105750


namespace luther_pancakes_correct_l1057_105794

/-- The number of people in Luther's family -/
def family_size : ℕ := 8

/-- The number of additional pancakes needed for everyone to have a second pancake -/
def additional_pancakes : ℕ := 4

/-- The number of pancakes Luther made initially -/
def initial_pancakes : ℕ := 12

/-- Theorem stating that the number of pancakes Luther made initially is correct -/
theorem luther_pancakes_correct :
  initial_pancakes = family_size * 2 - additional_pancakes :=
by sorry

end luther_pancakes_correct_l1057_105794


namespace maple_leaf_picking_l1057_105753

theorem maple_leaf_picking (elder_points younger_points : ℕ) 
  (h1 : elder_points = 5)
  (h2 : younger_points = 3)
  (h3 : ∃ (x y : ℕ), elder_points * x + younger_points * y = 102 ∧ x = y + 6) :
  ∃ (x y : ℕ), x = 15 ∧ y = 9 ∧ 
    elder_points * x + younger_points * y = 102 ∧ x = y + 6 := by
  sorry

end maple_leaf_picking_l1057_105753


namespace cricket_bat_price_l1057_105779

/-- The final price of a cricket bat after two sales with given profits -/
def final_price (initial_cost : ℝ) (profit1 : ℝ) (profit2 : ℝ) : ℝ :=
  initial_cost * (1 + profit1) * (1 + profit2)

/-- Theorem stating the final price of the cricket bat -/
theorem cricket_bat_price :
  final_price 148 0.20 0.25 = 222 := by
  sorry

end cricket_bat_price_l1057_105779


namespace journey_fraction_is_one_fourth_l1057_105795

/-- Represents the journey from Petya's home to school -/
structure Journey where
  totalTime : ℕ
  timeBeforeBell : ℕ
  timeLateIfReturn : ℕ

/-- Calculates the fraction of the journey completed when Petya remembered the pen -/
def fractionCompleted (j : Journey) : ℚ :=
  let detourTime := j.timeBeforeBell + j.timeLateIfReturn
  let timeToRememberedPoint := detourTime / 2
  timeToRememberedPoint / j.totalTime

/-- Theorem stating that the fraction of the journey completed when Petya remembered the pen is 1/4 -/
theorem journey_fraction_is_one_fourth (j : Journey) 
  (h1 : j.totalTime = 20)
  (h2 : j.timeBeforeBell = 3)
  (h3 : j.timeLateIfReturn = 7) : 
  fractionCompleted j = 1/4 := by
  sorry

end journey_fraction_is_one_fourth_l1057_105795


namespace metal_rods_for_fence_l1057_105768

/-- Calculates the number of metal rods needed for a fence with given specifications. -/
theorem metal_rods_for_fence (
  sheets_per_panel : ℕ)
  (beams_per_panel : ℕ)
  (panels : ℕ)
  (rods_per_sheet : ℕ)
  (rods_per_beam : ℕ)
  (h1 : sheets_per_panel = 3)
  (h2 : beams_per_panel = 2)
  (h3 : panels = 10)
  (h4 : rods_per_sheet = 10)
  (h5 : rods_per_beam = 4)
  : sheets_per_panel * panels * rods_per_sheet + beams_per_panel * panels * rods_per_beam = 380 := by
  sorry

#check metal_rods_for_fence

end metal_rods_for_fence_l1057_105768


namespace negative_two_star_negative_three_l1057_105754

-- Define the new operation
def star (a b : ℤ) : ℤ := b^2 - a

-- State the theorem
theorem negative_two_star_negative_three : star (-2) (-3) = 11 := by
  sorry

end negative_two_star_negative_three_l1057_105754


namespace quadratic_equations_solutions_l1057_105718

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, x1 = (-1 + Real.sqrt 17) / 2 ∧ 
                x2 = (-1 - Real.sqrt 17) / 2 ∧ 
                x1^2 + x1 - 4 = 0 ∧ 
                x2^2 + x2 - 4 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = 1 ∧ 
                x2 = 2 ∧ 
                (2*x1 + 1)^2 + 15 = 8*(2*x1 + 1) ∧ 
                (2*x2 + 1)^2 + 15 = 8*(2*x2 + 1)) :=
by
  sorry


end quadratic_equations_solutions_l1057_105718


namespace sum_of_numbers_with_lcm_and_ratio_l1057_105770

/-- Given three positive integers a, b, and c in the ratio 2:3:5 with LCM 120, their sum is 40 -/
theorem sum_of_numbers_with_lcm_and_ratio (a b c : ℕ+) : 
  (a : ℕ) + b + c = 40 ∧ 
  Nat.lcm a (Nat.lcm b c) = 120 ∧ 
  3 * a = 2 * b ∧ 
  5 * a = 2 * c := by
sorry


end sum_of_numbers_with_lcm_and_ratio_l1057_105770


namespace figure_50_squares_initial_values_correct_l1057_105736

/-- Represents the number of nonoverlapping unit squares in the nth figure -/
def g (n : ℕ) : ℕ := 2 * n^2 + 5 * n + 2

/-- The theorem states that the 50th term of the sequence equals 5252 -/
theorem figure_50_squares : g 50 = 5252 := by
  sorry

/-- Verifies that the function g matches the given initial values -/
theorem initial_values_correct :
  g 0 = 2 ∧ g 1 = 9 ∧ g 2 = 20 ∧ g 3 = 35 := by
  sorry

end figure_50_squares_initial_values_correct_l1057_105736


namespace intersection_complement_equality_l1057_105710

def A : Set ℝ := {0, 1, 2, 3, 4}

def B : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}

theorem intersection_complement_equality :
  A ∩ (Set.univ \ B) = {0, 3, 4} := by sorry

end intersection_complement_equality_l1057_105710


namespace average_trees_is_36_l1057_105761

/-- The number of trees planted by class A -/
def trees_A : ℕ := 35

/-- The number of trees planted by class B -/
def trees_B : ℕ := trees_A + 6

/-- The number of trees planted by class C -/
def trees_C : ℕ := trees_A - 3

/-- The average number of trees planted by the three classes -/
def average_trees : ℚ := (trees_A + trees_B + trees_C) / 3

theorem average_trees_is_36 : average_trees = 36 := by
  sorry

end average_trees_is_36_l1057_105761


namespace intersection_M_N_l1057_105701

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 + x = 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := by
  sorry

end intersection_M_N_l1057_105701


namespace day_50_of_prev_year_is_tuesday_l1057_105756

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a year -/
structure Year where
  number : ℤ
  isLeapYear : Bool

/-- Returns the day of the week for a given day number in a year -/
def dayOfWeek (y : Year) (dayNumber : ℕ) : DayOfWeek := sorry

/-- Returns the next year -/
def nextYear (y : Year) : Year := sorry

/-- Returns the previous year -/
def prevYear (y : Year) : Year := sorry

theorem day_50_of_prev_year_is_tuesday 
  (N : Year)
  (h1 : dayOfWeek N 250 = DayOfWeek.Friday)
  (h2 : dayOfWeek (nextYear N) 150 = DayOfWeek.Friday)
  (h3 : (nextYear N).isLeapYear = false) :
  dayOfWeek (prevYear N) 50 = DayOfWeek.Tuesday := by sorry

end day_50_of_prev_year_is_tuesday_l1057_105756


namespace sum_a_b_equals_negative_one_l1057_105735

theorem sum_a_b_equals_negative_one (a b : ℝ) : 
  (|a + 3| + (b - 2)^2 = 0) → (a + b = -1) := by
  sorry

end sum_a_b_equals_negative_one_l1057_105735


namespace x_value_proof_l1057_105762

theorem x_value_proof : ∃ x : ℝ, 
  3.5 * ((3.6 * 0.48 * x) / (0.12 * 0.09 * 0.5)) = 2800.0000000000005 → x = 1 := by
  sorry

end x_value_proof_l1057_105762


namespace contest_probabilities_l1057_105798

/-- Represents the total number of questions -/
def total_questions : ℕ := 8

/-- Represents the number of listening questions -/
def listening_questions : ℕ := 3

/-- Represents the number of written response questions -/
def written_questions : ℕ := 5

/-- Calculates the probability of the first student drawing a listening question
    and the second student drawing a written response question -/
def prob_listening_written : ℚ :=
  (listening_questions * written_questions : ℚ) / (total_questions * (total_questions - 1))

/-- Calculates the probability of at least one student drawing a listening question -/
def prob_at_least_one_listening : ℚ :=
  1 - (written_questions * (written_questions - 1) : ℚ) / (total_questions * (total_questions - 1))

theorem contest_probabilities :
  prob_listening_written = 15 / 56 ∧ prob_at_least_one_listening = 9 / 14 := by
  sorry

end contest_probabilities_l1057_105798


namespace movie_revenue_growth_equation_l1057_105792

theorem movie_revenue_growth_equation 
  (initial_revenue : ℝ) 
  (revenue_after_three_weeks : ℝ) 
  (x : ℝ) 
  (h1 : initial_revenue = 2.5)
  (h2 : revenue_after_three_weeks = 3.6)
  (h3 : ∀ t : ℕ, t < 3 → 
    initial_revenue * (1 + x)^t < initial_revenue * (1 + x)^(t+1)) :
  initial_revenue * (1 + x)^2 = revenue_after_three_weeks :=
sorry

end movie_revenue_growth_equation_l1057_105792


namespace at_least_three_positive_and_negative_l1057_105743

theorem at_least_three_positive_and_negative 
  (a : Fin 12 → ℝ) 
  (h : ∀ i ∈ Finset.range 10, a (i + 2) * (a (i + 1) - a (i + 2) + a (i + 3)) < 0) :
  (∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i > 0 ∧ a j > 0 ∧ a k > 0) ∧
  (∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i < 0 ∧ a j < 0 ∧ a k < 0) :=
sorry

end at_least_three_positive_and_negative_l1057_105743


namespace property_value_calculation_l1057_105724

/-- Calculate the total value of a property with different types of buildings --/
theorem property_value_calculation (condo_price condo_area barn_price barn_area 
  detached_price detached_area garage_price garage_area : ℕ) : 
  condo_price = 98 → 
  condo_area = 2400 → 
  barn_price = 84 → 
  barn_area = 1200 → 
  detached_price = 102 → 
  detached_area = 3500 → 
  garage_price = 60 → 
  garage_area = 480 → 
  (condo_price * condo_area + barn_price * barn_area + 
   detached_price * detached_area + garage_price * garage_area) = 721800 := by
  sorry

end property_value_calculation_l1057_105724


namespace eunji_remaining_confetti_l1057_105763

def initial_green_confetti : ℕ := 9
def initial_red_confetti : ℕ := 1
def confetti_given_away : ℕ := 4

theorem eunji_remaining_confetti :
  initial_green_confetti + initial_red_confetti - confetti_given_away = 6 := by
  sorry

end eunji_remaining_confetti_l1057_105763


namespace four_digit_sum_plus_2001_l1057_105711

theorem four_digit_sum_plus_2001 :
  ∃! n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧
  n = (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10) + 2001 ∧
  n = 1977 := by
sorry

end four_digit_sum_plus_2001_l1057_105711


namespace measurement_error_probability_l1057_105737

/-- The standard deviation of the measurement errors -/
def σ : ℝ := 10

/-- The maximum allowed absolute error -/
def δ : ℝ := 15

/-- The cumulative distribution function of the standard normal distribution -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- The probability that the absolute error is less than δ -/
noncomputable def P (δ : ℝ) (σ : ℝ) : ℝ := 2 * Φ (δ / σ)

theorem measurement_error_probability :
  ∃ ε > 0, |P δ σ - 0.8664| < ε :=
sorry

end measurement_error_probability_l1057_105737


namespace scalene_to_right_triangle_l1057_105708

/-- 
For any scalene triangle with sides a < b < c, there exists a real number x 
such that the new triangle with sides (a+x), (b+x), and (c+x) is a right triangle.
-/
theorem scalene_to_right_triangle 
  (a b c : ℝ) 
  (h_scalene : a < b ∧ b < c) : 
  ∃ x : ℝ, (a + x)^2 + (b + x)^2 = (c + x)^2 := by
  sorry

end scalene_to_right_triangle_l1057_105708


namespace gcd_840_1764_l1057_105775

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l1057_105775


namespace jane_started_at_18_l1057_105717

/-- Represents Jane's babysitting career --/
structure BabysittingCareer where
  current_age : ℕ
  years_since_stopped : ℕ
  oldest_babysat_current_age : ℕ
  start_age : ℕ

/-- Checks if the babysitting career satisfies all conditions --/
def is_valid_career (career : BabysittingCareer) : Prop :=
  career.current_age = 34 ∧
  career.years_since_stopped = 12 ∧
  career.oldest_babysat_current_age = 25 ∧
  career.start_age ≤ career.current_age - career.years_since_stopped ∧
  ∀ (child_age : ℕ), child_age ≤ career.oldest_babysat_current_age →
    2 * (child_age - career.years_since_stopped) ≤ career.current_age - career.years_since_stopped

theorem jane_started_at_18 :
  ∃ (career : BabysittingCareer), is_valid_career career ∧ career.start_age = 18 := by
  sorry

end jane_started_at_18_l1057_105717


namespace bike_lock_rotation_l1057_105740

/-- Rotates a single digit by 180 degrees on a 10-digit wheel. -/
def rotate_digit (d : Nat) : Nat :=
  (d + 5) % 10

/-- The original code of the bike lock. -/
def original_code : List Nat := [6, 3, 4, 8]

/-- The correct code after rotation. -/
def correct_code : List Nat := [1, 8, 9, 3]

/-- Theorem stating that rotating each digit of the original code results in the correct code. -/
theorem bike_lock_rotation :
  original_code.map rotate_digit = correct_code := by
  sorry

#eval original_code.map rotate_digit

end bike_lock_rotation_l1057_105740


namespace triangle_problem_l1057_105755

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
  a = 2 * Real.sin B / Real.sqrt 3 →
  a = 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  A = π/3 ∧ b = 2 ∧ c = 2 := by
  sorry

end triangle_problem_l1057_105755


namespace vector_perpendicular_l1057_105766

/-- Given vectors a and b in ℝ², prove that a - b is perpendicular to b -/
theorem vector_perpendicular (a b : ℝ × ℝ) (h1 : a = (1, 0)) (h2 : b = (1/2, 1/2)) : 
  (a - b) • b = 0 := by
  sorry

end vector_perpendicular_l1057_105766


namespace systematic_sampling_first_number_l1057_105728

theorem systematic_sampling_first_number 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (last_sample : ℕ) 
  (h1 : population_size = 2000)
  (h2 : sample_size = 100)
  (h3 : last_sample = 1994)
  (h4 : last_sample < population_size) :
  let interval := population_size / sample_size
  let first_sample := last_sample - (sample_size - 1) * interval
  first_sample = 14 := by
sorry

end systematic_sampling_first_number_l1057_105728


namespace cylindrical_glass_volume_l1057_105788

/-- The volume of a cylindrical glass with specific straw conditions -/
theorem cylindrical_glass_volume : 
  ∀ (h r : ℝ),
  h > 0 → 
  r > 0 →
  h = 8 →
  r = 6 →
  h^2 + r^2 = 10^2 →
  (π : ℝ) = 3.14 →
  π * r^2 * h = 226.08 :=
by sorry

end cylindrical_glass_volume_l1057_105788


namespace polynomial_remainder_l1057_105791

theorem polynomial_remainder (x : ℝ) : 
  (x^3 - 3*x + 5) % (x - 1) = 3 := by sorry

end polynomial_remainder_l1057_105791


namespace M_is_range_of_f_l1057_105771

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = x^2}

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x^2

-- Theorem statement
theorem M_is_range_of_f : M = Set.range f := by sorry

end M_is_range_of_f_l1057_105771


namespace function_value_inequality_l1057_105713

theorem function_value_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = -f (-x))
  (h2 : ∀ x, 1 < x ∧ x < 2 → f x > 0) :
  f (-1.5) ≠ 1 := by
  sorry

end function_value_inequality_l1057_105713


namespace cube_volume_l1057_105758

theorem cube_volume (cube_side : ℝ) (h1 : cube_side > 0) (h2 : cube_side ^ 2 = 36) :
  cube_side ^ 3 = 216 :=
by sorry

end cube_volume_l1057_105758


namespace lattice_points_count_l1057_105709

/-- The number of lattice points on a line segment --/
def countLatticePoints (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem: The number of lattice points on the line segment from (4,19) to (39,239) is 6 --/
theorem lattice_points_count :
  countLatticePoints 4 19 39 239 = 6 := by sorry

end lattice_points_count_l1057_105709


namespace total_cement_is_15_point_1_l1057_105722

/-- The amount of cement (in tons) used for Lexi's street -/
def lexis_street_cement : ℝ := 10

/-- The amount of cement (in tons) used for Tess's street -/
def tess_street_cement : ℝ := 5.1

/-- The total amount of cement used by Roadster's Paving Company -/
def total_cement : ℝ := lexis_street_cement + tess_street_cement

/-- Theorem stating that the total cement used is 15.1 tons -/
theorem total_cement_is_15_point_1 : total_cement = 15.1 := by sorry

end total_cement_is_15_point_1_l1057_105722


namespace square_difference_divided_by_eleven_l1057_105781

theorem square_difference_divided_by_eleven : (131^2 - 120^2) / 11 = 251 := by
  sorry

end square_difference_divided_by_eleven_l1057_105781


namespace arithmetic_sequence_12th_term_l1057_105723

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_4th : a 4 = 1) :
  a 12 = 15 := by
sorry

end arithmetic_sequence_12th_term_l1057_105723


namespace hover_solution_l1057_105774

def hover_problem (central_day1 : ℝ) : Prop :=
  let mountain_day1 : ℝ := 3
  let eastern_day1 : ℝ := 2
  let extra_day2 : ℝ := 2
  let total_time : ℝ := 24
  let mountain_day2 : ℝ := mountain_day1 + extra_day2
  let central_day2 : ℝ := central_day1 + extra_day2
  let eastern_day2 : ℝ := eastern_day1 + extra_day2
  mountain_day1 + central_day1 + eastern_day1 + mountain_day2 + central_day2 + eastern_day2 = total_time

theorem hover_solution : hover_problem 4 := by
  sorry

end hover_solution_l1057_105774


namespace some_number_value_l1057_105720

theorem some_number_value (x : ℝ) (some_number : ℝ) 
  (h1 : (27 / 4) * x - some_number = 3 * x + 27) 
  (h2 : x = 12) : 
  some_number = 18 := by
  sorry

end some_number_value_l1057_105720


namespace certain_number_problem_l1057_105739

theorem certain_number_problem (x y : ℝ) 
  (h1 : 0.25 * x = 0.15 * y - 20) 
  (h2 : x = 820) : 
  y = 1500 := by
sorry

end certain_number_problem_l1057_105739


namespace carla_marbles_l1057_105730

/-- The number of marbles Carla has after buying more -/
def total_marbles (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem stating the total number of marbles Carla has -/
theorem carla_marbles :
  total_marbles 2289 489 = 2778 := by
  sorry

end carla_marbles_l1057_105730


namespace logarithm_sum_property_l1057_105759

theorem logarithm_sum_property (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by
  sorry

end logarithm_sum_property_l1057_105759


namespace division_remainder_theorem_l1057_105790

theorem division_remainder_theorem (a b : ℕ) :
  (∃ (q r : ℕ), a^2 + b^2 = (a + b) * q + r ∧ q^2 + r = 1977) →
  ((a = 37 ∧ b = 50) ∨ (a = 50 ∧ b = 37) ∨ (a = 7 ∧ b = 50) ∨ (a = 50 ∧ b = 7)) :=
by sorry

end division_remainder_theorem_l1057_105790


namespace oil_change_cost_l1057_105729

/-- Calculates the cost of each oil change given the specified conditions. -/
theorem oil_change_cost
  (miles_per_month : ℕ)
  (miles_per_oil_change : ℕ)
  (free_oil_changes_per_year : ℕ)
  (yearly_oil_change_cost : ℕ)
  (h1 : miles_per_month = 1000)
  (h2 : miles_per_oil_change = 3000)
  (h3 : free_oil_changes_per_year = 1)
  (h4 : yearly_oil_change_cost = 150) :
  yearly_oil_change_cost / (miles_per_month * 12 / miles_per_oil_change - free_oil_changes_per_year) = 50 := by
  sorry

end oil_change_cost_l1057_105729


namespace fourth_region_area_l1057_105733

/-- A regular hexagon divided into four regions by three line segments -/
structure DividedHexagon where
  /-- The total area of the hexagon -/
  total_area : ℝ
  /-- The areas of the four regions -/
  region_areas : Fin 4 → ℝ
  /-- The hexagon is regular and divided into four regions -/
  is_regular_divided : total_area = region_areas 0 + region_areas 1 + region_areas 2 + region_areas 3

/-- The theorem stating the area of the fourth region -/
theorem fourth_region_area (h : DividedHexagon) 
  (h1 : h.region_areas 0 = 2)
  (h2 : h.region_areas 1 = 3)
  (h3 : h.region_areas 2 = 4) :
  h.region_areas 3 = 11 := by
  sorry

end fourth_region_area_l1057_105733


namespace ivan_speed_ratio_l1057_105716

/-- Represents the speed of a person or group -/
structure Speed :=
  (value : ℝ)

/-- Represents time in hours -/
def Time : Type := ℝ

theorem ivan_speed_ratio (group_speed : Speed) (ivan_speed : Speed) : 
  -- Ivan left 15 minutes (0.25 hours) after the group started
  -- Ivan took 2.5 hours to catch up with the group after retrieving the flashlight
  -- Speeds of the group and Ivan (when not with the group) are constant
  (0.25 : ℝ) * group_speed.value + 2.5 * group_speed.value = 
    2.5 * ivan_speed.value + 2 * (0.25 * group_speed.value) →
  -- The ratio of Ivan's speed to the group's speed is 1.2
  ivan_speed.value / group_speed.value = 1.2 := by
  sorry

end ivan_speed_ratio_l1057_105716


namespace mall_garage_third_level_spaces_l1057_105734

/-- Represents the number of parking spaces on each level of a four-story parking garage -/
structure ParkingGarage :=
  (level1 : ℕ)
  (level2 : ℕ)
  (level3 : ℕ)
  (level4 : ℕ)

/-- Calculates the total number of parking spaces in the garage -/
def total_spaces (g : ParkingGarage) : ℕ := g.level1 + g.level2 + g.level3 + g.level4

/-- Represents the parking garage described in the problem -/
def mall_garage (x : ℕ) : ParkingGarage :=
  { level1 := 90
  , level2 := 90 + 8
  , level3 := 90 + 8 + x
  , level4 := 90 + 8 + x - 9 }

/-- The theorem to be proved -/
theorem mall_garage_third_level_spaces :
  ∃ x : ℕ, 
    total_spaces (mall_garage x) = 399 ∧ 
    x = 12 := by sorry

end mall_garage_third_level_spaces_l1057_105734


namespace sector_central_angle_l1057_105789

/-- A circular sector with perimeter 8 and area 4 has a central angle of 2 radians -/
theorem sector_central_angle (r : ℝ) (l : ℝ) (θ : ℝ) : 
  r > 0 → 
  2 * r + l = 8 →  -- perimeter equation
  1 / 2 * l * r = 4 →  -- area equation
  θ = l / r →  -- definition of central angle in radians
  θ = 2 := by
sorry

end sector_central_angle_l1057_105789


namespace twenty_one_less_than_sixty_thousand_l1057_105764

theorem twenty_one_less_than_sixty_thousand : 60000 - 21 = 59979 := by
  sorry

end twenty_one_less_than_sixty_thousand_l1057_105764


namespace johns_age_l1057_105732

theorem johns_age (john dad : ℕ) : 
  john = dad - 30 →
  john + dad = 80 →
  john = 25 := by sorry

end johns_age_l1057_105732


namespace joes_age_l1057_105721

theorem joes_age (B J E : ℕ) : 
  B = 3 * J →                  -- Billy's age is three times Joe's age
  E = (B + J) / 2 →            -- Emily's age is the average of Billy's and Joe's ages
  B + J + E = 90 →             -- The sum of their ages is 90
  J = 15 :=                    -- Joe's age is 15
by sorry

end joes_age_l1057_105721


namespace factorial_division_l1057_105777

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 5 = 30240 := by
  sorry

end factorial_division_l1057_105777


namespace complex_modulus_power_four_l1057_105786

theorem complex_modulus_power_four : Complex.abs ((2 + Complex.I) ^ 4) = 25 := by
  sorry

end complex_modulus_power_four_l1057_105786


namespace last_digits_of_powers_last_two_digits_of_nine_powers_last_six_digits_of_seven_powers_l1057_105783

theorem last_digits_of_powers (n m : ℕ) : 
  n^(n^n) ≡ n^(n^(n^n)) [MOD 10^m] :=
sorry

theorem last_two_digits_of_nine_powers : 
  9^(9^9) ≡ 9^(9^(9^9)) [MOD 100] ∧ 
  9^(9^9) ≡ 99 [MOD 100] :=
sorry

theorem last_six_digits_of_seven_powers : 
  7^(7^(7^7)) ≡ 7^(7^(7^(7^7))) [MOD 1000000] ∧ 
  7^(7^(7^7)) ≡ 999999 [MOD 1000000] :=
sorry

end last_digits_of_powers_last_two_digits_of_nine_powers_last_six_digits_of_seven_powers_l1057_105783


namespace cube_and_fifth_power_existence_l1057_105727

theorem cube_and_fifth_power_existence (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ n : ℕ, n ≥ 1 ∧ ∃ k l : ℕ, a * n = k^3 ∧ b * n = l^5 :=
sorry

end cube_and_fifth_power_existence_l1057_105727
