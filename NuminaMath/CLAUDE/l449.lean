import Mathlib

namespace eighteen_times_thirtysix_minus_twentyseven_times_eighteen_l449_44997

theorem eighteen_times_thirtysix_minus_twentyseven_times_eighteen : 
  18 * 36 - 27 * 18 = 162 := by
  sorry

end eighteen_times_thirtysix_minus_twentyseven_times_eighteen_l449_44997


namespace xiaoqiang_games_l449_44941

/-- Represents a participant in the chess tournament -/
inductive Participant
  | Jia
  | Yi
  | Bing
  | Ding
  | Xiaoqiang

/-- The number of games played by each participant -/
def games_played (p : Participant) : ℕ :=
  match p with
  | Participant.Jia => 4
  | Participant.Yi => 3
  | Participant.Bing => 2
  | Participant.Ding => 1
  | Participant.Xiaoqiang => 2  -- This is what we want to prove

/-- The total number of games played in the tournament -/
def total_games : ℕ := 10  -- (5 choose 2) = 10

theorem xiaoqiang_games :
  games_played Participant.Xiaoqiang = 2 :=
by sorry

end xiaoqiang_games_l449_44941


namespace competition_winners_l449_44945

theorem competition_winners (total_winners : Nat) (total_score : Nat) 
  (first_place_score : Nat) (second_place_score : Nat) (third_place_score : Nat) :
  total_winners = 5 →
  total_score = 94 →
  first_place_score = 20 →
  second_place_score = 19 →
  third_place_score = 18 →
  ∃ (first_place_winners second_place_winners third_place_winners : Nat),
    first_place_winners = 1 ∧
    second_place_winners = 2 ∧
    third_place_winners = 2 ∧
    first_place_winners + second_place_winners + third_place_winners = total_winners ∧
    first_place_winners * first_place_score + 
    second_place_winners * second_place_score + 
    third_place_winners * third_place_score = total_score :=
by sorry

end competition_winners_l449_44945


namespace student_count_l449_44990

theorem student_count : ∃ n : ℕ, n > 0 ∧ 
  (n / 2 : ℚ) + (n / 4 : ℚ) + (n / 8 : ℚ) + 3 = n ∧ n = 24 := by
  sorry

end student_count_l449_44990


namespace expression_evaluation_l449_44973

theorem expression_evaluation : (4 + 6 + 2) / 3 - 2 / 3 = 10 / 3 := by
  sorry

end expression_evaluation_l449_44973


namespace sum_of_squares_roots_l449_44928

theorem sum_of_squares_roots (x : ℝ) :
  Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4 →
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 :=
by
  sorry

end sum_of_squares_roots_l449_44928


namespace minimum_packages_to_breakeven_l449_44984

def bike_cost : ℕ := 1200
def earning_per_package : ℕ := 15
def maintenance_cost : ℕ := 5

theorem minimum_packages_to_breakeven :
  ∃ n : ℕ, n * (earning_per_package - maintenance_cost) ≥ bike_cost ∧
  ∀ m : ℕ, m * (earning_per_package - maintenance_cost) ≥ bike_cost → m ≥ n :=
by sorry

end minimum_packages_to_breakeven_l449_44984


namespace apples_per_pie_l449_44932

theorem apples_per_pie (initial_apples : ℕ) (handed_out : ℕ) (num_pies : ℕ) 
  (h1 : initial_apples = 86)
  (h2 : handed_out = 30)
  (h3 : num_pies = 7) :
  (initial_apples - handed_out) / num_pies = 8 :=
by
  sorry

end apples_per_pie_l449_44932


namespace craig_appliance_sales_l449_44907

/-- The number of appliances sold by Craig in a week -/
def num_appliances : ℕ := 6

/-- The total selling price of appliances in dollars -/
def total_selling_price : ℚ := 3620

/-- The total commission Craig earned in dollars -/
def total_commission : ℚ := 662

/-- The fixed commission per appliance in dollars -/
def fixed_commission : ℚ := 50

/-- The percentage of selling price Craig receives as commission -/
def commission_rate : ℚ := 1/10

theorem craig_appliance_sales :
  num_appliances = 6 ∧
  (num_appliances : ℚ) * fixed_commission + commission_rate * total_selling_price = total_commission :=
sorry

end craig_appliance_sales_l449_44907


namespace parabola_properties_l449_44948

/-- A parabola with equation y^2 = 2px and focus at (1,0) -/
structure Parabola where
  p : ℝ
  focus_x : ℝ
  focus_y : ℝ
  h_focus : (focus_x, focus_y) = (1, 0)

/-- The value of p for the parabola -/
def p_value (par : Parabola) : ℝ := par.p

/-- The equation of the directrix for the parabola -/
def directrix_equation (par : Parabola) : ℝ → Prop := fun x ↦ x = -1

theorem parabola_properties (par : Parabola) :
  p_value par = 2 ∧ directrix_equation par = fun x ↦ x = -1 := by
  sorry

end parabola_properties_l449_44948


namespace arccos_gt_arctan_iff_l449_44900

theorem arccos_gt_arctan_iff (x : ℝ) : Real.arccos x > Real.arctan x ↔ x ∈ Set.Ici (-1) ∩ Set.Iio 1 := by
  sorry

end arccos_gt_arctan_iff_l449_44900


namespace circle_y_is_eleven_l449_44944

/-- Represents the configuration of numbers in the circles. -/
structure CircleConfig where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  x : ℤ
  y : ℤ

/-- The conditions given in the problem. -/
def satisfiesConditions (config : CircleConfig) : Prop :=
  config.a + config.b + config.x = 30 ∧
  config.c + config.d + config.y = 30 ∧
  config.a + config.b + config.c + config.d = 40 ∧
  config.x + config.y + config.c + config.b = 40 ∧
  config.x = 9

/-- The theorem stating that if the conditions are satisfied, Y must be 11. -/
theorem circle_y_is_eleven (config : CircleConfig) 
  (h : satisfiesConditions config) : config.y = 11 := by
  sorry


end circle_y_is_eleven_l449_44944


namespace goals_scored_over_two_days_l449_44917

/-- The total number of goals scored by Gina and Tom over two days -/
def total_goals (gina_day1 gina_day2 tom_day1 tom_day2 : ℕ) : ℕ :=
  gina_day1 + gina_day2 + tom_day1 + tom_day2

/-- Theorem stating the total number of goals scored by Gina and Tom over two days -/
theorem goals_scored_over_two_days :
  ∃ (gina_day1 gina_day2 tom_day1 tom_day2 : ℕ),
    gina_day1 = 2 ∧
    tom_day1 = gina_day1 + 3 ∧
    tom_day2 = 6 ∧
    gina_day2 = tom_day2 - 2 ∧
    total_goals gina_day1 gina_day2 tom_day1 tom_day2 = 17 :=
by
  sorry

end goals_scored_over_two_days_l449_44917


namespace johns_sleep_theorem_l449_44931

theorem johns_sleep_theorem :
  let days_in_week : ℕ := 7
  let short_sleep_days : ℕ := 2
  let short_sleep_hours : ℝ := 3
  let recommended_sleep : ℝ := 8
  let sleep_percentage : ℝ := 0.6

  let normal_sleep_days : ℕ := days_in_week - short_sleep_days
  let normal_sleep_hours : ℝ := sleep_percentage * recommended_sleep
  
  let total_sleep : ℝ := 
    (short_sleep_days : ℝ) * short_sleep_hours + 
    (normal_sleep_days : ℝ) * normal_sleep_hours

  total_sleep = 30 := by sorry

end johns_sleep_theorem_l449_44931


namespace unique_element_implies_a_value_l449_44904

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

-- State the theorem
theorem unique_element_implies_a_value (a : ℝ) :
  (∃! x, x ∈ A a) → a = 0 ∨ a = 1 := by
  sorry

end unique_element_implies_a_value_l449_44904


namespace greatest_six_digit_divisible_l449_44902

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

theorem greatest_six_digit_divisible (n : ℕ) : 
  is_six_digit n ∧ 
  21 ∣ n ∧ 
  35 ∣ n ∧ 
  66 ∣ n ∧ 
  110 ∣ n ∧ 
  143 ∣ n → 
  n ≤ 990990 :=
sorry

end greatest_six_digit_divisible_l449_44902


namespace lesogoria_inhabitants_l449_44989

-- Define the types of inhabitants
inductive Inhabitant
| Elf
| Dwarf

-- Define the types of statements
inductive Statement
| AboutGold
| AboutDwarf
| Other

-- Define a function to determine if a statement is true based on the speaker and the type of statement
def isTruthful (speaker : Inhabitant) (statement : Statement) : Prop :=
  match speaker, statement with
  | Inhabitant.Dwarf, Statement.AboutGold => false
  | Inhabitant.Elf, Statement.AboutDwarf => false
  | _, _ => true

-- Define the statements made by A and B
def statementA : Statement := Statement.AboutGold
def statementB : Statement := Statement.Other

-- Define the theorem
theorem lesogoria_inhabitants :
  ∃ (a b : Inhabitant),
    (isTruthful a statementA = false) ∧
    (isTruthful b statementB = true) ∧
    (a = Inhabitant.Dwarf) ∧
    (b = Inhabitant.Dwarf) :=
  sorry


end lesogoria_inhabitants_l449_44989


namespace number_difference_l449_44966

theorem number_difference (N : ℝ) (h : 0.25 * N = 100) : N - (3/4 * N) = 100 := by
  sorry

end number_difference_l449_44966


namespace star_equation_solution_l449_44960

/-- Definition of the star operation -/
def star (a b : ℝ) : ℝ := a * b + b - a - 1

/-- Theorem: If 3 star x = 20, then x = 6 -/
theorem star_equation_solution :
  (∃ x : ℝ, star 3 x = 20) → (∃ x : ℝ, star 3 x = 20 ∧ x = 6) :=
by
  sorry

end star_equation_solution_l449_44960


namespace intersection_k_value_l449_44957

/-- The intersection point of two lines -3x + y = k and 2x + y = 20 when x = -10 -/
def intersection_point : ℝ × ℝ := (-10, 40)

/-- The first line equation: -3x + y = k -/
def line1 (k : ℝ) (p : ℝ × ℝ) : Prop :=
  -3 * p.1 + p.2 = k

/-- The second line equation: 2x + y = 20 -/
def line2 (p : ℝ × ℝ) : Prop :=
  2 * p.1 + p.2 = 20

/-- Theorem: The value of k is 70 given that the lines -3x + y = k and 2x + y = 20 intersect when x = -10 -/
theorem intersection_k_value :
  line2 intersection_point →
  (∃ k, line1 k intersection_point) →
  (∃! k, line1 k intersection_point ∧ k = 70) :=
by
  sorry

end intersection_k_value_l449_44957


namespace baker_usual_pastries_l449_44937

/-- The number of pastries the baker usually sells -/
def usual_pastries : ℕ := sorry

/-- The number of loaves the baker usually sells -/
def usual_loaves : ℕ := 10

/-- The number of pastries sold today -/
def today_pastries : ℕ := 14

/-- The number of loaves sold today -/
def today_loaves : ℕ := 25

/-- The price of a pastry in dollars -/
def pastry_price : ℚ := 2

/-- The price of a loaf in dollars -/
def loaf_price : ℚ := 4

/-- The difference between today's sales and average sales in dollars -/
def sales_difference : ℚ := 48

theorem baker_usual_pastries : 
  usual_pastries = 20 :=
by sorry

end baker_usual_pastries_l449_44937


namespace blueberries_count_l449_44996

/-- The number of strawberries in each red box -/
def strawberries_per_red_box : ℕ := 100

/-- The difference between strawberries in a red box and blueberries in a blue box -/
def berry_difference : ℕ := 30

/-- The number of blueberries in each blue box -/
def blueberries_per_blue_box : ℕ := strawberries_per_red_box - berry_difference

theorem blueberries_count : blueberries_per_blue_box = 70 := by
  sorry

end blueberries_count_l449_44996


namespace terminal_zeros_125_360_l449_44916

def number_of_terminal_zeros (a b : ℕ) : ℕ :=
  sorry

theorem terminal_zeros_125_360 : 
  let a := 125
  let b := 360
  number_of_terminal_zeros a b = 3 :=
by
  sorry

end terminal_zeros_125_360_l449_44916


namespace smallest_n_congruence_l449_44956

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ ∀ m : ℕ, 0 < m ∧ m < n → ¬(253 * m ≡ 989 * m [ZMOD 15])) →
  (253 * n ≡ 989 * n [ZMOD 15]) →
  n = 15 := by sorry

end smallest_n_congruence_l449_44956


namespace muffin_banana_price_ratio_l449_44955

theorem muffin_banana_price_ratio :
  ∀ (m b : ℝ),
  (3 * m + 5 * b > 0) →
  (4 * m + 10 * b = 3 * m + 5 * b + 12) →
  (m / b = 2) :=
by sorry

end muffin_banana_price_ratio_l449_44955


namespace quilt_block_shaded_half_l449_44979

/-- Represents a square quilt block divided into a 4x4 grid -/
structure QuiltBlock where
  grid_size : Nat
  full_shaded : Nat
  half_shaded : Nat

/-- Calculates the fraction of shaded area in a quilt block -/
def shaded_fraction (qb : QuiltBlock) : Rat :=
  (qb.full_shaded + qb.half_shaded / 2) / (qb.grid_size * qb.grid_size)

theorem quilt_block_shaded_half :
  ∀ qb : QuiltBlock,
    qb.grid_size = 4 →
    qb.full_shaded = 6 →
    qb.half_shaded = 4 →
    shaded_fraction qb = 1/2 := by
  sorry

end quilt_block_shaded_half_l449_44979


namespace tan_alpha_value_l449_44903

theorem tan_alpha_value (α : Real) (h : 3 * Real.sin α + 4 * Real.cos α = 5) : 
  Real.tan α = 3/4 := by
  sorry

end tan_alpha_value_l449_44903


namespace quaternary_2132_equals_septenary_314_l449_44954

/-- Converts a quaternary (base 4) number represented as a list of digits to decimal (base 10) -/
def quaternary_to_decimal (digits : List Nat) : Nat := sorry

/-- Converts a decimal (base 10) number to septenary (base 7) represented as a list of digits -/
def decimal_to_septenary (n : Nat) : List Nat := sorry

/-- Theorem stating that the quaternary number 2132 is equal to the septenary number 314 -/
theorem quaternary_2132_equals_septenary_314 :
  decimal_to_septenary (quaternary_to_decimal [2, 1, 3, 2]) = [3, 1, 4] := by sorry

end quaternary_2132_equals_septenary_314_l449_44954


namespace complex_fraction_magnitude_l449_44943

theorem complex_fraction_magnitude (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by sorry

end complex_fraction_magnitude_l449_44943


namespace green_ducks_percentage_in_larger_pond_l449_44999

/-- Represents the percentage of green ducks in the larger pond -/
def larger_pond_green_percentage : ℝ := 12

theorem green_ducks_percentage_in_larger_pond :
  let smaller_pond_ducks : ℕ := 30
  let larger_pond_ducks : ℕ := 50
  let smaller_pond_green_percentage : ℝ := 20
  let total_green_percentage : ℝ := 15
  (smaller_pond_green_percentage / 100 * smaller_pond_ducks +
   larger_pond_green_percentage / 100 * larger_pond_ducks) /
  (smaller_pond_ducks + larger_pond_ducks) * 100 = total_green_percentage :=
by sorry

end green_ducks_percentage_in_larger_pond_l449_44999


namespace fraction_simplification_l449_44983

theorem fraction_simplification : 
  ((3^2010)^2 - (3^2008)^2) / ((3^2009)^2 - (3^2007)^2) = 9 := by
  sorry

end fraction_simplification_l449_44983


namespace range_of_f_l449_44974

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((1 - x) / (1 + x)) + Real.arctan (2 * x)

theorem range_of_f :
  Set.range f = Set.Ioo (-Real.pi / 2) (Real.pi / 2) :=
sorry

end range_of_f_l449_44974


namespace new_person_weight_l449_44924

/-- Given a group of 15 people, proves that if replacing a person weighing 45 kg 
    with a new person increases the average weight by 8 kg, 
    then the new person weighs 165 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) 
  (replaced_weight : ℝ) (new_weight : ℝ) : 
  initial_count = 15 → 
  weight_increase = 8 → 
  replaced_weight = 45 → 
  new_weight = initial_count * weight_increase + replaced_weight → 
  new_weight = 165 := by
  sorry

end new_person_weight_l449_44924


namespace jose_profit_share_l449_44922

/-- Calculates the share of profit for an investor in a partnership --/
def calculate_profit_share (investment : ℕ) (months : ℕ) (total_profit : ℕ) (total_capital_months : ℕ) : ℕ :=
  (investment * months * total_profit) / total_capital_months

theorem jose_profit_share :
  let tom_investment : ℕ := 30000
  let tom_months : ℕ := 12
  let jose_investment : ℕ := 45000
  let jose_months : ℕ := 10
  let total_profit : ℕ := 36000
  let total_capital_months : ℕ := tom_investment * tom_months + jose_investment * jose_months
  
  calculate_profit_share jose_investment jose_months total_profit total_capital_months = 20000 := by
  sorry


end jose_profit_share_l449_44922


namespace sum_of_first_four_terms_l449_44977

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sum_of_first_four_terms 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a2 : a 2 = 9) 
  (h_a5 : a 5 = 243) : 
  (a 1 + a 2 + a 3 + a 4 = 120) := by
  sorry

end sum_of_first_four_terms_l449_44977


namespace continued_fraction_equation_solution_l449_44952

def continued_fraction (a : ℕ → ℕ) (n : ℕ) : ℚ :=
  if n = 0 then 0
  else (a n : ℚ)⁻¹ + continued_fraction a (n-1)

def left_side (n : ℕ) : ℚ :=
  1 - continued_fraction (fun i => i + 1) n

def right_side (x : ℕ → ℕ) (n : ℕ) : ℚ :=
  continued_fraction x n

theorem continued_fraction_equation_solution (n : ℕ) (h : n ≥ 2) :
  ∃! x : ℕ → ℕ, left_side n = right_side x n ∧
    x 1 = 1 ∧ x 2 = 1 ∧ ∀ i, 3 ≤ i → i ≤ n → x i = i :=
sorry

end continued_fraction_equation_solution_l449_44952


namespace additive_implies_odd_l449_44970

-- Define the property of the function
def is_additive (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem additive_implies_odd (f : ℝ → ℝ) (h : is_additive f) : is_odd f := by
  sorry

end additive_implies_odd_l449_44970


namespace division_sum_theorem_l449_44933

theorem division_sum_theorem (n d : ℕ) (h1 : n = 55) (h2 : d = 11) :
  n + d + (n / d) = 71 := by
  sorry

end division_sum_theorem_l449_44933


namespace spade_nested_calculation_l449_44920

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_nested_calculation : spade 5 (spade 3 (spade 8 12)) = 4 := by
  sorry

end spade_nested_calculation_l449_44920


namespace f_properties_l449_44905

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem f_properties :
  (∀ x ∈ Set.Ioo (-2 : ℝ) 0, ∀ y ∈ Set.Ioo (-2 : ℝ) 0, x < y → f x > f y) ∧
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = x₁ - 2012 ∧ f x₂ = x₂ - 2012 ∧
    ∀ x, x ≠ x₁ ∧ x ≠ x₂ → f x ≠ x - 2012) :=
by sorry

end f_properties_l449_44905


namespace distance_PQ_l449_44921

def P : ℝ × ℝ := (-1, 2)
def Q : ℝ × ℝ := (3, 0)

theorem distance_PQ : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end distance_PQ_l449_44921


namespace determinant_equality_l449_44942

theorem determinant_equality (x y z w : ℝ) : 
  x * w - y * z = 7 → (x + z) * w - (y + 2 * w) * z = 7 - w * z := by
  sorry

end determinant_equality_l449_44942


namespace graces_pool_filling_time_l449_44911

/-- The problem of filling Grace's pool --/
theorem graces_pool_filling_time 
  (pool_capacity : ℝ) 
  (first_hose_rate : ℝ) 
  (second_hose_rate : ℝ) 
  (additional_time : ℝ) 
  (h : pool_capacity = 390) 
  (r1 : first_hose_rate = 50) 
  (r2 : second_hose_rate = 70) 
  (t : additional_time = 2) :
  ∃ (wait_time : ℝ), 
    wait_time * first_hose_rate + 
    additional_time * (first_hose_rate + second_hose_rate) = 
    pool_capacity ∧ wait_time = 3 := by
  sorry

end graces_pool_filling_time_l449_44911


namespace power_of_two_equality_l449_44958

theorem power_of_two_equality (K : ℕ) : 32^2 * 4^5 = 2^K ↔ K = 20 := by
  sorry

end power_of_two_equality_l449_44958


namespace initial_water_percentage_l449_44901

/-- Given a container with capacity 100 liters, prove that the initial percentage
    of water is 30% if adding 45 liters makes it 3/4 full. -/
theorem initial_water_percentage
  (capacity : ℝ)
  (added_water : ℝ)
  (final_fraction : ℝ)
  (h1 : capacity = 100)
  (h2 : added_water = 45)
  (h3 : final_fraction = 3/4)
  (h4 : (initial_percentage / 100) * capacity + added_water = final_fraction * capacity) :
  initial_percentage = 30 :=
sorry

#check initial_water_percentage

end initial_water_percentage_l449_44901


namespace remaining_payment_l449_44969

def part_payment : ℝ := 875
def payment_percentage : ℝ := 0.25

theorem remaining_payment :
  let total_cost := part_payment / payment_percentage
  let remaining := total_cost - part_payment
  remaining = 2625 := by sorry

end remaining_payment_l449_44969


namespace mapping_A_to_B_l449_44919

def f (x : ℝ) : ℝ := 2 * x - 1

def B : Set ℝ := {-3, -1, 3}

theorem mapping_A_to_B :
  ∃ A : Set ℝ, (∀ x ∈ A, f x ∈ B) ∧ (∀ y ∈ B, ∃ x ∈ A, f x = y) ∧ A = {-1, 0, 2} := by
  sorry

end mapping_A_to_B_l449_44919


namespace groupD_correct_l449_44953

/-- Represents a group of Chinese words -/
structure WordGroup :=
  (words : List String)

/-- Checks if a word is correctly written -/
def isCorrectlyWritten (word : String) : Prop :=
  sorry -- Implementation details omitted

/-- Checks if all words in a group are correctly written -/
def allWordsCorrect (group : WordGroup) : Prop :=
  ∀ word ∈ group.words, isCorrectlyWritten word

/-- The four given groups of words -/
def groupA : WordGroup :=
  ⟨["萌孽", "青鸾", "契合", "苦思冥想", "情深意笃", "骇人听闻"]⟩

def groupB : WordGroup :=
  ⟨["斒斓", "彭觞", "木楔", "虚与委蛇", "肆无忌惮", "殒身不恤"]⟩

def groupC : WordGroup :=
  ⟨["青睐", "气概", "编辑", "呼天抢地", "轻歌慢舞", "长歌当哭"]⟩

def groupD : WordGroup :=
  ⟨["缧绁", "剌谬", "陷阱", "伶仃孤苦", "运筹帷幄", "作壁上观"]⟩

/-- Theorem stating that group D is the only group with all words correctly written -/
theorem groupD_correct :
  allWordsCorrect groupD ∧
  ¬allWordsCorrect groupA ∧
  ¬allWordsCorrect groupB ∧
  ¬allWordsCorrect groupC :=
sorry

end groupD_correct_l449_44953


namespace max_value_of_a_min_value_of_expression_l449_44959

-- Problem I
theorem max_value_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = |x - 5/2| + |x - a|)
  (h2 : ∀ x, f x ≥ a) :
  a ≤ 5/4 ∧ ∃ x, f x = 5/4 := by
sorry

-- Problem II
theorem min_value_of_expression (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + 2*y + 3*z = 1) :
  3/x + 2/y + 1/z ≥ 16 + 8*Real.sqrt 3 ∧ 
  ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    x + 2*y + 3*z = 1 ∧ 
    3/x + 2/y + 1/z = 16 + 8*Real.sqrt 3 := by
sorry

end max_value_of_a_min_value_of_expression_l449_44959


namespace sqrt_abs_sum_zero_implies_power_l449_44906

theorem sqrt_abs_sum_zero_implies_power (m n : ℝ) :
  Real.sqrt (m - 2) + |n + 3| = 0 → (m + n)^2023 = -1 := by
  sorry

end sqrt_abs_sum_zero_implies_power_l449_44906


namespace bridge_length_calculation_l449_44930

/-- Calculates the length of a bridge given train parameters -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  255 = (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length :=
by sorry

end bridge_length_calculation_l449_44930


namespace arithmetic_square_root_of_sqrt_16_l449_44968

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end arithmetic_square_root_of_sqrt_16_l449_44968


namespace sqrt_meaningful_iff_geq_neg_two_l449_44934

theorem sqrt_meaningful_iff_geq_neg_two (a : ℝ) : 
  (∃ x : ℝ, x^2 = a + 2) ↔ a ≥ -2 := by sorry

end sqrt_meaningful_iff_geq_neg_two_l449_44934


namespace equation_two_roots_l449_44963

-- Define the equation
def equation (x k : ℂ) : Prop :=
  x / (x + 3) + x / (x + 4) = k * x

-- Define the condition for having exactly two distinct roots
def has_two_distinct_roots (k : ℂ) : Prop :=
  ∃ x y : ℂ, x ≠ y ∧ equation x k ∧ equation y k ∧
  ∀ z : ℂ, equation z k → z = x ∨ z = y

-- State the theorem
theorem equation_two_roots :
  ∀ k : ℂ, has_two_distinct_roots k ↔ k = 7/12 ∨ k = 2*I ∨ k = -2*I :=
sorry

end equation_two_roots_l449_44963


namespace prize_plan_optimal_l449_44994

/-- Represents the prices and quantities of prizes A and B -/
structure PrizePlan where
  priceA : ℕ
  priceB : ℕ
  quantityA : ℕ
  quantityB : ℕ

/-- Conditions for the prize plan -/
def validPrizePlan (p : PrizePlan) : Prop :=
  3 * p.priceA + 2 * p.priceB = 130 ∧
  5 * p.priceA + 4 * p.priceB = 230 ∧
  p.quantityA + p.quantityB = 20 ∧
  p.quantityA ≥ 2 * p.quantityB

/-- Total cost of the prize plan -/
def totalCost (p : PrizePlan) : ℕ :=
  p.priceA * p.quantityA + p.priceB * p.quantityB

/-- The theorem to be proved -/
theorem prize_plan_optimal (p : PrizePlan) (h : validPrizePlan p) :
  p.priceA = 30 ∧ p.priceB = 20 ∧ p.quantityA = 14 ∧ p.quantityB = 6 ∧ totalCost p = 560 := by
  sorry

end prize_plan_optimal_l449_44994


namespace sqrt_inequality_square_sum_inequality_l449_44998

-- Theorem 1
theorem sqrt_inequality (a : ℝ) (h : a > 1) : 
  Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by
  sorry

-- Theorem 2
theorem square_sum_inequality (a b : ℝ) : 
  a^2 + b^2 ≥ a*b + a + b - 1 := by
  sorry

end sqrt_inequality_square_sum_inequality_l449_44998


namespace min_product_of_primes_l449_44912

theorem min_product_of_primes (x y z : Nat) : 
  Nat.Prime x ∧ Nat.Prime y ∧ Nat.Prime z ∧
  Odd x ∧ Odd y ∧ Odd z ∧
  (y^5 + 1) % x = 0 ∧
  (z^5 + 1) % y = 0 ∧
  (x^5 + 1) % z = 0 →
  ∀ a b c : Nat, 
    Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧
    Odd a ∧ Odd b ∧ Odd c ∧
    (b^5 + 1) % a = 0 ∧
    (c^5 + 1) % b = 0 ∧
    (a^5 + 1) % c = 0 →
    x * y * z ≤ a * b * c ∧
    x * y * z = 2013 := by
sorry

end min_product_of_primes_l449_44912


namespace inequality_solution_l449_44925

open Set

theorem inequality_solution (x : ℝ) : 
  3 * x - 2 < (x + 2)^2 ∧ (x + 2)^2 < 9 * x - 8 ↔ x ∈ Ioo 3 4 :=
by sorry

end inequality_solution_l449_44925


namespace similar_right_triangles_leg_length_l449_44964

/-- Given two similar right triangles, where one triangle has legs of length 12 and 9,
    and the other triangle has one leg of length 6, prove that the length of the other
    leg in the second triangle is 4.5. -/
theorem similar_right_triangles_leg_length
  (a b c d : ℝ)
  (h1 : a = 12)
  (h2 : b = 9)
  (h3 : c = 6)
  (h4 : a / b = c / d)
  : d = 4.5 := by
  sorry

end similar_right_triangles_leg_length_l449_44964


namespace storm_rainfall_l449_44936

theorem storm_rainfall (first_hour : ℝ) (second_hour : ℝ) : 
  second_hour = 2 * first_hour + 7 →
  first_hour + second_hour = 22 →
  first_hour = 5 := by
sorry

end storm_rainfall_l449_44936


namespace exactly_two_solutions_l449_44993

/-- The number of ordered pairs (a, b) of positive integers satisfying the equation -/
def solution_count : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    p.1 > 0 ∧ p.2 > 0 ∧
    2 * p.1 * p.2 + 108 = 15 * Nat.lcm p.1 p.2 + 18 * Nat.gcd p.1 p.2
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- The main theorem stating that there are exactly 2 solutions -/
theorem exactly_two_solutions : solution_count = 2 := by
  sorry

end exactly_two_solutions_l449_44993


namespace ratio_a_to_b_l449_44927

theorem ratio_a_to_b (a b : ℚ) (h : 2 * a = 3 * b) : 
  ∃ (k : ℚ), k > 0 ∧ a = (3 * k) ∧ b = (2 * k) := by
  sorry

end ratio_a_to_b_l449_44927


namespace age_difference_l449_44947

theorem age_difference (a b c : ℕ) : 
  b = 12 →
  b = 2 * c →
  a + b + c = 32 →
  a = b + 2 := by
sorry

end age_difference_l449_44947


namespace earliest_solution_l449_44985

/-- The temperature model as a function of time -/
def temperature (t : ℝ) : ℝ := -t^2 + 10*t + 60

/-- The theorem stating that 12.5 is the earliest non-negative solution -/
theorem earliest_solution :
  ∀ t : ℝ, t ≥ 0 → temperature t = 85 → t ≥ 12.5 := by sorry

end earliest_solution_l449_44985


namespace square_roots_problem_l449_44926

theorem square_roots_problem (x : ℝ) (n : ℝ) (h1 : n > 0) 
  (h2 : x + 1 = Real.sqrt n) (h3 : x - 5 = Real.sqrt n) : n = 9 := by
  sorry

end square_roots_problem_l449_44926


namespace janet_tulips_l449_44935

/-- The number of tulips Janet picked -/
def T : ℕ := sorry

/-- The total number of flowers Janet picked -/
def total_flowers : ℕ := T + 11

/-- The number of flowers Janet used -/
def used_flowers : ℕ := 11

/-- The number of extra flowers Janet had -/
def extra_flowers : ℕ := 4

theorem janet_tulips : T = 4 := by sorry

end janet_tulips_l449_44935


namespace integer_pairs_satisfying_equation_l449_44910

theorem integer_pairs_satisfying_equation :
  ∀ (x y : ℤ), x^5 + y^5 = (x + y)^3 ↔
    ((x = 0 ∧ y = 1) ∨
     (x = 1 ∧ y = 0) ∨
     (x = 0 ∧ y = -1) ∨
     (x = -1 ∧ y = 0) ∨
     (x = 2 ∧ y = 2) ∨
     (x = -2 ∧ y = -2) ∨
     (∃ (a : ℤ), x = a ∧ y = -a)) :=
by sorry

end integer_pairs_satisfying_equation_l449_44910


namespace extreme_values_imply_b_l449_44913

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x

/-- The derivative of f with respect to x -/
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_imply_b (a b : ℝ) :
  (f' a b 1 = 0) → (f' a b 2 = 0) → b = 4 := by
  sorry

end extreme_values_imply_b_l449_44913


namespace negation_equivalence_l449_44949

theorem negation_equivalence (m : ℝ) : 
  (¬ ∃ x < 0, x^2 + 2*x - m > 0) ↔ (∀ x < 0, x^2 + 2*x - m ≤ 0) := by
  sorry

end negation_equivalence_l449_44949


namespace ratio_of_a_to_d_l449_44965

theorem ratio_of_a_to_d (a b c d : ℚ) 
  (hab : a / b = 5 / 3)
  (hbc : b / c = 1 / 5)
  (hcd : c / d = 3 / 2) :
  a / d = 1 / 2 := by
  sorry

end ratio_of_a_to_d_l449_44965


namespace perfect_square_factors_of_3456_l449_44991

/-- Given that 3456 = 2^7 * 3^3, this function counts the number of its positive integer factors that are perfect squares. -/
def count_perfect_square_factors : ℕ :=
  let prime_factorization : List (ℕ × ℕ) := [(2, 7), (3, 3)]
  sorry

/-- The number of positive integer factors of 3456 that are perfect squares is 8. -/
theorem perfect_square_factors_of_3456 : count_perfect_square_factors = 8 := by
  sorry

end perfect_square_factors_of_3456_l449_44991


namespace room_width_calculation_l449_44914

/-- Given a rectangular room with length 12 feet and width w feet, 
    with a carpet placed leaving a 2-foot wide border all around, 
    if the area of the border is 72 square feet, 
    then the width of the room is 10 feet. -/
theorem room_width_calculation (w : ℝ) : 
  w > 0 →  -- width is positive
  12 * w - 8 * (w - 4) = 72 →  -- area of border is 72 sq ft
  w = 10 := by
sorry

end room_width_calculation_l449_44914


namespace unique_solution_exists_l449_44967

/-- Given a > 0 and a ≠ 1, there exists a unique x such that a^x = log_(1/4) x -/
theorem unique_solution_exists (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  ∃! x : ℝ, a^x = Real.log x / Real.log (1/4) := by
  sorry

end unique_solution_exists_l449_44967


namespace exterior_angle_theorem_l449_44950

-- Define the triangle
structure Triangle :=
  (a b c : ℝ)
  (sum_eq_180 : a + b + c = 180)

-- Define the problem
theorem exterior_angle_theorem (t : Triangle) 
  (h1 : t.a = 45)
  (h2 : t.b = 30) :
  180 - t.c = 75 := by
  sorry

end exterior_angle_theorem_l449_44950


namespace remainder_of_m_l449_44995

theorem remainder_of_m (m : ℕ) (h1 : m^3 % 7 = 6) (h2 : m^4 % 7 = 4) : m % 7 = 3 := by
  sorry

end remainder_of_m_l449_44995


namespace inscribed_circumscribed_ratio_l449_44978

/-- An equilateral triangle with inscribed and circumscribed circles -/
structure EquilateralTriangleWithCircles where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The radius of the circumscribed circle -/
  R : ℝ
  /-- The radius of the inscribed circle is positive -/
  r_pos : r > 0
  /-- The radius of the circumscribed circle is positive -/
  R_pos : R > 0

/-- The ratio of the inscribed circle radius to the circumscribed circle radius in an equilateral triangle is 1:2 -/
theorem inscribed_circumscribed_ratio (t : EquilateralTriangleWithCircles) : t.r / t.R = 1 / 2 :=
sorry

end inscribed_circumscribed_ratio_l449_44978


namespace smallest_solution_quadratic_l449_44992

theorem smallest_solution_quadratic : 
  let f : ℝ → ℝ := fun y ↦ 10 * y^2 - 47 * y + 49
  ∃ y : ℝ, f y = 0 ∧ (∀ z : ℝ, f z = 0 → y ≤ z) ∧ y = 1.4 := by
  sorry

end smallest_solution_quadratic_l449_44992


namespace leadership_combinations_count_l449_44980

def tribe_size : ℕ := 15
def num_supporting_chiefs : ℕ := 3
def num_inferior_officers_per_chief : ℕ := 2

def leadership_combinations : ℕ := 
  tribe_size * 
  (tribe_size - 1) * 
  (tribe_size - 2) * 
  (tribe_size - 3) * 
  Nat.choose (tribe_size - 4) num_inferior_officers_per_chief *
  Nat.choose (tribe_size - 6) num_inferior_officers_per_chief *
  Nat.choose (tribe_size - 8) num_inferior_officers_per_chief

theorem leadership_combinations_count : leadership_combinations = 19320300 := by
  sorry

end leadership_combinations_count_l449_44980


namespace rain_probability_both_locations_l449_44987

theorem rain_probability_both_locations (p_no_rain_A p_no_rain_B : ℝ) 
  (h1 : p_no_rain_A = 0.3)
  (h2 : p_no_rain_B = 0.4)
  (h3 : 0 ≤ p_no_rain_A ∧ p_no_rain_A ≤ 1)
  (h4 : 0 ≤ p_no_rain_B ∧ p_no_rain_B ≤ 1) :
  (1 - p_no_rain_A) * (1 - p_no_rain_B) = 0.42 := by
  sorry

end rain_probability_both_locations_l449_44987


namespace lottery_increment_proof_l449_44909

/-- Represents the increment in the price of each successive ticket -/
def increment : ℝ := 1

/-- The number of lottery tickets -/
def num_tickets : ℕ := 5

/-- The price of the first ticket -/
def first_ticket_price : ℝ := 1

/-- The profit Lily plans to keep -/
def profit : ℝ := 4

/-- The prize money for the lottery winner -/
def prize : ℝ := 11

/-- The total amount collected from selling all tickets -/
def total_collected (x : ℝ) : ℝ :=
  first_ticket_price + (first_ticket_price + x) + (first_ticket_price + 2*x) + 
  (first_ticket_price + 3*x) + (first_ticket_price + 4*x)

theorem lottery_increment_proof :
  total_collected increment = profit + prize :=
sorry

end lottery_increment_proof_l449_44909


namespace chi_square_greater_than_critical_expected_volleyball_recipients_correct_l449_44938

-- Define the total number of students
def total_students : ℕ := 200

-- Define the number of male and female students
def male_students : ℕ := 100
def female_students : ℕ := 100

-- Define the number of students in Group A (volleyball)
def group_a_total : ℕ := 96

-- Define the number of male students in Group A
def group_a_male : ℕ := 36

-- Define the critical value for α = 0.001
def critical_value : ℚ := 10828 / 1000

-- Define the chi-square statistic
def chi_square : ℚ := 11538 / 1000

-- Define the number of male students selected for stratified sampling
def stratified_sample : ℕ := 25

-- Define the number of students selected for gifts
def gift_recipients : ℕ := 3

-- Define the expected number of volleyball players among gift recipients
def expected_volleyball_recipients : ℚ := 621 / 575

-- Theorem 1: The chi-square value is greater than the critical value
theorem chi_square_greater_than_critical : chi_square > critical_value := by sorry

-- Theorem 2: The expected number of volleyball players among gift recipients is correct
theorem expected_volleyball_recipients_correct : 
  expected_volleyball_recipients = 621 / 575 := by sorry

end chi_square_greater_than_critical_expected_volleyball_recipients_correct_l449_44938


namespace expression_simplification_and_evaluation_l449_44939

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 3) :
  let original := (3 / (x - 1) - x - 1) / ((x^2 - 4*x + 4) / (x - 1))
  let simplified := -(x + 2) / (x - 2)
  original = simplified ∧ simplified = -5 := by sorry

end expression_simplification_and_evaluation_l449_44939


namespace sum_of_angles_with_given_tangents_l449_44975

theorem sum_of_angles_with_given_tangents (A B C : Real) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  Real.tan A = 1 →
  Real.tan B = 2 →
  Real.tan C = 3 →
  A + B + C = π := by
  sorry

end sum_of_angles_with_given_tangents_l449_44975


namespace F_is_odd_l449_44908

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define F in terms of f
def F (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f x - f (-x)

-- Theorem: F is an odd function
theorem F_is_odd (f : ℝ → ℝ) : ∀ x : ℝ, F f (-x) = -(F f x) :=
by
  sorry

end F_is_odd_l449_44908


namespace stating_currency_exchange_problem_l449_44929

/-- Represents the exchange rate from U.S. dollars to Canadian dollars -/
def exchange_rate : ℚ := 12 / 8

/-- Represents the amount spent in Canadian dollars -/
def amount_spent : ℕ := 72

/-- 
Theorem stating that if a person exchanges m U.S. dollars to Canadian dollars
at the given exchange rate, spends the specified amount, and is left with m
Canadian dollars, then m must equal 144.
-/
theorem currency_exchange_problem (m : ℕ) :
  (m : ℚ) * exchange_rate - amount_spent = m →
  m = 144 :=
by sorry

end stating_currency_exchange_problem_l449_44929


namespace students_taking_one_subject_l449_44951

/-- The number of students taking both Geometry and History -/
def both_geometry_history : ℕ := 15

/-- The total number of students taking Geometry -/
def total_geometry : ℕ := 30

/-- The number of students taking History only -/
def history_only : ℕ := 15

/-- The number of students taking both Geometry and Science -/
def both_geometry_science : ℕ := 8

/-- The number of students taking Science only -/
def science_only : ℕ := 10

/-- Theorem stating that the number of students taking only one subject is 32 -/
theorem students_taking_one_subject :
  (total_geometry - both_geometry_history - both_geometry_science) + history_only + science_only = 32 := by
  sorry

end students_taking_one_subject_l449_44951


namespace fish_given_by_sister_l449_44976

/-- Given that Mrs. Sheridan initially had 22 fish and now has 69 fish
    after receiving fish from her sister, prove that the number of fish
    her sister gave her is 47. -/
theorem fish_given_by_sister
  (initial_fish : ℕ)
  (final_fish : ℕ)
  (h1 : initial_fish = 22)
  (h2 : final_fish = 69) :
  final_fish - initial_fish = 47 := by
  sorry

end fish_given_by_sister_l449_44976


namespace probability_3_successes_in_7_trials_value_l449_44946

/-- The probability of getting exactly 3 successes in 7 independent trials,
    where the probability of success in each trial is 3/7. -/
def probability_3_successes_in_7_trials : ℚ :=
  (Nat.choose 7 3 : ℚ) * (3/7)^3 * (4/7)^4

/-- Theorem stating that the probability of getting exactly 3 successes
    in 7 independent trials, where the probability of success in each trial
    is 3/7, is equal to 242112/823543. -/
theorem probability_3_successes_in_7_trials_value :
  probability_3_successes_in_7_trials = 242112/823543 := by
  sorry

end probability_3_successes_in_7_trials_value_l449_44946


namespace square_sum_minus_one_le_zero_l449_44982

theorem square_sum_minus_one_le_zero (a b : ℝ) :
  a^2 + b^2 - 1 - a^2 * b^2 ≤ 0 ↔ (a^2 - 1) * (b^2 - 1) ≥ 0 := by
sorry

end square_sum_minus_one_le_zero_l449_44982


namespace sin_3x_periodic_l449_44971

/-- The function f(x) = sin(3x) is periodic with period 2π/3 -/
theorem sin_3x_periodic (x : ℝ) : Real.sin (3 * (x + 2 * Real.pi / 3)) = Real.sin (3 * x) := by
  sorry

end sin_3x_periodic_l449_44971


namespace smallest_taxicab_number_is_smallest_l449_44986

/-- The smallest positive integer that can be expressed as the sum of two cubes in two different ways -/
def smallest_taxicab_number : ℕ := 1729

/-- A function that checks if a number can be expressed as the sum of two cubes in two different ways -/
def is_taxicab (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a < c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a^3 + b^3 = n ∧ c^3 + d^3 = n ∧ (a, b) ≠ (c, d)

/-- Theorem stating that smallest_taxicab_number is indeed the smallest taxicab number -/
theorem smallest_taxicab_number_is_smallest :
  is_taxicab smallest_taxicab_number ∧
  ∀ m : ℕ, m < smallest_taxicab_number → ¬is_taxicab m :=
sorry

end smallest_taxicab_number_is_smallest_l449_44986


namespace inverse_36_mod_47_l449_44988

theorem inverse_36_mod_47 (h : (11⁻¹ : ZMod 47) = 43) : (36⁻¹ : ZMod 47) = 4 := by
  sorry

end inverse_36_mod_47_l449_44988


namespace negation_of_existence_negation_of_sin_equality_l449_44961

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_sin_equality : 
  (¬ ∃ x : ℝ, x = Real.sin x) ↔ (∀ x : ℝ, x ≠ Real.sin x) := by sorry

end negation_of_existence_negation_of_sin_equality_l449_44961


namespace max_value_inequality_max_value_equality_l449_44972

theorem max_value_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) :
  2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ (Real.sqrt 2 - 1) / 2 := by
sorry

theorem max_value_equality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) :
  (2 * Real.sqrt (a * b) - 4 * a^2 - b^2 = (Real.sqrt 2 - 1) / 2) ↔ (a = 1/4 ∧ b = 1/2) := by
sorry

end max_value_inequality_max_value_equality_l449_44972


namespace sucrose_concentration_in_mixture_l449_44940

/-- Concentration of sucrose in a mixture of two solutions --/
theorem sucrose_concentration_in_mixture 
  (conc_A : ℝ) (conc_B : ℝ) (vol_A : ℝ) (vol_B : ℝ) 
  (h1 : conc_A = 15.3) 
  (h2 : conc_B = 27.8) 
  (h3 : vol_A = 45) 
  (h4 : vol_B = 75) : 
  (conc_A * vol_A + conc_B * vol_B) / (vol_A + vol_B) = 
  (15.3 * 45 + 27.8 * 75) / (45 + 75) :=
by sorry

#eval (15.3 * 45 + 27.8 * 75) / (45 + 75)

end sucrose_concentration_in_mixture_l449_44940


namespace ninety_percent_of_600_equals_fifty_percent_of_x_l449_44923

theorem ninety_percent_of_600_equals_fifty_percent_of_x (x : ℝ) :
  (90 / 100) * 600 = (50 / 100) * x → x = 1080 := by
sorry

end ninety_percent_of_600_equals_fifty_percent_of_x_l449_44923


namespace hades_can_prevent_sisyphus_l449_44962

/-- Represents the state of the mountain with stones -/
structure MountainState where
  steps : Nat
  stones : Nat
  stone_positions : Finset Nat

/-- Defines the game rules and initial state -/
def initial_state : MountainState :=
  { steps := 1001
  , stones := 500
  , stone_positions := Finset.range 500 }

/-- Sisyphus's move: Lifts a stone to the nearest free step above -/
def sisyphus_move (state : MountainState) : MountainState :=
  sorry

/-- Hades's move: Lowers a stone to the nearest free step below -/
def hades_move (state : MountainState) : MountainState :=
  sorry

/-- Represents a full round of the game (Sisyphus's move followed by Hades's move) -/
def game_round (state : MountainState) : MountainState :=
  hades_move (sisyphus_move state)

/-- Theorem stating that Hades can prevent Sisyphus from reaching the top step -/
theorem hades_can_prevent_sisyphus (state : MountainState := initial_state) :
  ∀ n : Nat, (game_round^[n] state).stone_positions.max < state.steps :=
  sorry

end hades_can_prevent_sisyphus_l449_44962


namespace classroom_boys_count_l449_44918

theorem classroom_boys_count (initial_girls : ℕ) : 
  let initial_boys := initial_girls + 5
  let final_girls := initial_girls + 10
  let final_boys := initial_boys + 3
  final_girls = 22 →
  final_boys = 20 := by
sorry

end classroom_boys_count_l449_44918


namespace f_two_zeros_implies_a_range_l449_44981

/-- The function f(x) defined in terms of the parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * a * x + 3 * a - 5

/-- The derivative of f(x) with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * (x^2 - 1)

/-- Theorem stating that if f has at least two zeros, then 1 ≤ a ≤ 5 -/
theorem f_two_zeros_implies_a_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) →
  1 ≤ a ∧ a ≤ 5 :=
sorry

end f_two_zeros_implies_a_range_l449_44981


namespace modular_arithmetic_problem_l449_44915

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), a ≡ 61 [ZMOD 70] ∧ b ≡ 43 [ZMOD 70] ∧ (3 * a + 9 * b) ≡ 0 [ZMOD 70] := by
  sorry

end modular_arithmetic_problem_l449_44915
