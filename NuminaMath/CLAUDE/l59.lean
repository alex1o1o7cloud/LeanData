import Mathlib

namespace nickel_count_l59_5924

theorem nickel_count (total_value : ℚ) (nickel_value : ℚ) (quarter_value : ℚ) :
  total_value = 12 →
  nickel_value = 0.05 →
  quarter_value = 0.25 →
  ∃ n : ℕ, n * nickel_value + n * quarter_value = total_value ∧ n = 40 :=
by sorry

end nickel_count_l59_5924


namespace distance_focus_to_asymptote_l59_5964

-- Define the hyperbola
def hyperbola (m : ℝ) (x y : ℝ) : Prop := y^2 - m*x^2 = 3*m

-- Define the condition that m is positive
def m_positive (m : ℝ) : Prop := m > 0

-- Define a focus of the hyperbola
def is_focus (m : ℝ) (F : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a^2 = 3*m ∧ b^2 = 3 ∧ c^2 = a^2 + b^2 ∧ 
  (F.1 = 0 ∧ F.2 = c ∨ F.1 = 0 ∧ F.2 = -c)

-- Define an asymptote of the hyperbola
def is_asymptote (m : ℝ) (l : ℝ → ℝ) : Prop :=
  ∀ x, l x = Real.sqrt m * x ∨ l x = -Real.sqrt m * x

-- Theorem statement
theorem distance_focus_to_asymptote (m : ℝ) (F : ℝ × ℝ) (l : ℝ → ℝ) :
  m_positive m →
  hyperbola m F.1 F.2 →
  is_focus m F →
  is_asymptote m l →
  ∃ (d : ℝ), d = Real.sqrt 3 ∧ 
    d = |F.2 - l F.1| / Real.sqrt (1 + (Real.sqrt m)^2) :=
sorry

end distance_focus_to_asymptote_l59_5964


namespace percentage_problem_l59_5976

theorem percentage_problem (x : ℝ) (h : (30/100) * (15/100) * x = 18) :
  (15/100) * (30/100) * x = 18 := by sorry

end percentage_problem_l59_5976


namespace line_passes_through_fixed_point_l59_5990

/-- The line equation passes through a fixed point for all real m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (2 * m + 1) * 3 + (m + 1) * 1 - 7 * m - 4 = 0 := by
  sorry

#check line_passes_through_fixed_point

end line_passes_through_fixed_point_l59_5990


namespace sum_of_coefficients_l59_5988

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -1 :=
by
  sorry

end sum_of_coefficients_l59_5988


namespace positive_solution_x_l59_5977

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 10 - 3 * x - 2 * y)
  (eq2 : y * z = 8 - 3 * y - 2 * z)
  (eq3 : x * z = 40 - 5 * x - 3 * z)
  (x_pos : x > 0) :
  x = 3 := by
sorry

end positive_solution_x_l59_5977


namespace right_triangle_hypotenuse_l59_5939

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  -- Right-angled triangle condition (Pythagorean theorem)
  c^2 = a^2 + b^2 →
  -- Sum of squares of all sides is 2500
  a^2 + b^2 + c^2 = 2500 →
  -- Difference between hypotenuse and one side is 10
  c - a = 10 →
  -- Prove that the hypotenuse length is 25√2
  c = 25 * Real.sqrt 2 := by
sorry

end right_triangle_hypotenuse_l59_5939


namespace line_perp_from_plane_perp_l59_5907

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perpLine : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perpPlane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpLineLine : Line → Line → Prop)

-- Theorem statement
theorem line_perp_from_plane_perp 
  (a b : Line) (α β : Plane) 
  (h1 : perpLine a α) 
  (h2 : perpLine b β) 
  (h3 : perpPlane α β) : 
  perpLineLine a b :=
sorry

end line_perp_from_plane_perp_l59_5907


namespace expression_simplification_l59_5936

theorem expression_simplification (a : ℝ) (h : a = 2 + Real.sqrt 2) :
  (a / (a + 2) + 1 / (a^2 - 4)) / ((a - 1) / (a + 2)) + 1 / (a - 2) = Real.sqrt 2 + 1 := by
  sorry

end expression_simplification_l59_5936


namespace parallel_lines_l59_5999

/-- Two lines in the form ax + by + c = 0 are parallel if and only if they have the same a and b coefficients. -/
def are_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 = a2 ∧ b1 = b2 ∧ c1 ≠ c2

/-- The line x + 2y + 2 = 0 is parallel to the line x + 2y + 1 = 0. -/
theorem parallel_lines : are_parallel 1 2 2 1 2 1 := by
  sorry

end parallel_lines_l59_5999


namespace base_conversion_equality_l59_5968

theorem base_conversion_equality (b : ℕ) : b > 0 ∧ (4 * 6 + 2 = 1 * b^2 + 2 * b + 1) → b = 3 := by
  sorry

end base_conversion_equality_l59_5968


namespace shooting_competition_probability_l59_5934

theorem shooting_competition_probability 
  (p_single : ℝ) 
  (p_twice : ℝ) 
  (h1 : p_single = 4/5) 
  (h2 : p_twice = 1/2) : 
  p_twice / p_single = 5/8 := by
sorry

end shooting_competition_probability_l59_5934


namespace ratio_of_q_r_to_p_l59_5927

def p : ℝ := 47.99999999999999

theorem ratio_of_q_r_to_p : ∃ (f : ℝ), f = 1/6 ∧ 2 * f * p = p - 32 := by
  sorry

end ratio_of_q_r_to_p_l59_5927


namespace xiao_hong_age_expression_dad_age_when_xiao_hong_is_seven_l59_5910

-- Define Dad's age
def dad_age : ℕ → ℕ := λ a => a

-- Define Xiao Hong's age as a function of Dad's age
def xiao_hong_age : ℕ → ℚ := λ a => (a - 3) / 4

-- Theorem for Xiao Hong's age expression
theorem xiao_hong_age_expression (a : ℕ) :
  xiao_hong_age a = (a - 3) / 4 :=
sorry

-- Theorem for Dad's age when Xiao Hong is 7
theorem dad_age_when_xiao_hong_is_seven :
  ∃ a : ℕ, xiao_hong_age a = 7 ∧ dad_age a = 31 :=
sorry

end xiao_hong_age_expression_dad_age_when_xiao_hong_is_seven_l59_5910


namespace system_equation_ratio_l59_5952

theorem system_equation_ratio (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (eq1 : 8 * x - 6 * y = c) (eq2 : 12 * y - 18 * x = d) :
  c / d = -4 / 3 := by
  sorry

end system_equation_ratio_l59_5952


namespace income_data_mean_difference_l59_5965

/-- Represents the income data for a group of families -/
structure IncomeData where
  num_families : ℕ
  min_income : ℕ
  max_income : ℕ
  incorrect_max_income : ℕ

/-- Calculates the difference between the mean of incorrect data and actual data -/
def mean_difference (data : IncomeData) : ℚ :=
  (data.incorrect_max_income - data.max_income) / data.num_families

/-- Theorem stating the difference in means for the given scenario -/
theorem income_data_mean_difference :
  ∀ (data : IncomeData),
  data.num_families = 500 →
  data.min_income = 12000 →
  data.max_income = 150000 →
  data.incorrect_max_income = 1500000 →
  mean_difference data = 2700 := by
  sorry

end income_data_mean_difference_l59_5965


namespace vacuum_savings_theorem_l59_5991

/-- The number of weeks needed to save for a vacuum cleaner. -/
def weeks_to_save (initial_savings : ℕ) (weekly_savings : ℕ) (vacuum_cost : ℕ) : ℕ :=
  ((vacuum_cost - initial_savings) + weekly_savings - 1) / weekly_savings

/-- Theorem stating that it takes 10 weeks to save for the vacuum cleaner. -/
theorem vacuum_savings_theorem :
  weeks_to_save 20 10 120 = 10 := by
  sorry

end vacuum_savings_theorem_l59_5991


namespace unique_pairs_sum_product_l59_5945

theorem unique_pairs_sum_product (S P : ℝ) (h : S^2 ≥ 4*P) :
  ∃! (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ + y₁ = S ∧ x₁ * y₁ = P) ∧
    (x₂ + y₂ = S ∧ x₂ * y₂ = P) ∧
    x₁ = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧
    y₁ = S - x₁ ∧
    x₂ = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧
    y₂ = S - x₂ :=
by
  sorry

end unique_pairs_sum_product_l59_5945


namespace grid_cutting_ways_l59_5929

-- Define the shape of the grid
def GridShape : Type := Unit  -- Placeholder for the specific grid shape

-- Define the property of being cuttable into 1×2 rectangles
def IsCuttableInto1x2Rectangles (g : GridShape) : Prop := sorry

-- Define the function that counts the number of ways to cut the grid
def NumberOfWaysToCut (g : GridShape) : ℕ := sorry

-- The main theorem
theorem grid_cutting_ways (g : GridShape) : 
  IsCuttableInto1x2Rectangles g → NumberOfWaysToCut g = 27 := by sorry

end grid_cutting_ways_l59_5929


namespace cube_root_fourth_power_equals_81_l59_5905

theorem cube_root_fourth_power_equals_81 (y : ℝ) : (y^(1/3))^4 = 81 → y = 27 := by
  sorry

end cube_root_fourth_power_equals_81_l59_5905


namespace interesting_number_expected_value_l59_5992

/-- A type representing a 6-digit number with specific properties -/
structure InterestingNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  a_positive : a > 0
  b_positive : b > 0
  c_positive : c > 0
  d_positive : d > 0
  e_positive : e > 0
  f_positive : f > 0
  a_less_b : a < b
  b_less_c : b < c
  d_ge_e : d ≥ e
  e_ge_f : e ≥ f
  a_le_9 : a ≤ 9
  b_le_9 : b ≤ 9
  c_le_9 : c ≤ 9
  d_le_9 : d ≤ 9
  e_le_9 : e ≤ 9
  f_le_9 : f ≤ 9

/-- The expected value of an interesting number -/
def expectedValue (n : InterestingNumber) : ℝ :=
  100000 * n.a + 10000 * n.b + 1000 * n.c + 100 * n.d + 10 * n.e + n.f

/-- The theorem stating the expected value of all interesting numbers -/
theorem interesting_number_expected_value :
  ∃ (μ : ℝ), ∀ (n : InterestingNumber), μ = 308253 := by
  sorry

end interesting_number_expected_value_l59_5992


namespace parallel_intersection_l59_5981

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define the intersection operation for planes
variable (intersection : Plane → Plane → Line)

-- State the theorem
theorem parallel_intersection
  (l₁ l₂ l₃ : Line) (α β : Plane)
  (h1 : parallel l₁ l₂)
  (h2 : subset l₁ α)
  (h3 : subset l₂ β)
  (h4 : intersection α β = l₃) :
  parallel l₁ l₃ :=
sorry

end parallel_intersection_l59_5981


namespace jacket_price_calculation_l59_5921

/-- Calculates the final price of an item after applying three sequential discounts -/
def final_price (original_price : ℝ) (discount1 discount2 discount3 : ℝ) : ℝ :=
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)

/-- Theorem stating that the final price of a $250 jacket after three specific discounts is $94.5 -/
theorem jacket_price_calculation : 
  final_price 250 0.4 0.3 0.1 = 94.5 := by
  sorry

#eval final_price 250 0.4 0.3 0.1

end jacket_price_calculation_l59_5921


namespace total_balls_l59_5933

theorem total_balls (jungkook_red_balls : ℕ) (yoongi_blue_balls : ℕ) 
  (h1 : jungkook_red_balls = 3) (h2 : yoongi_blue_balls = 4) : 
  jungkook_red_balls + yoongi_blue_balls = 7 := by
  sorry

end total_balls_l59_5933


namespace point_M_satisfies_conditions_l59_5922

-- Define the function f(x) = 2x^2 + 1
def f (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x

theorem point_M_satisfies_conditions :
  let x₀ : ℝ := -2
  let y₀ : ℝ := 9
  f x₀ = y₀ ∧ f' x₀ = -8 := by
  sorry

end point_M_satisfies_conditions_l59_5922


namespace december_ear_muff_sales_l59_5949

/-- The number of type B ear muffs sold in December -/
def type_b_count : ℕ := 3258

/-- The price of each type B ear muff -/
def type_b_price : ℚ := 69/10

/-- The number of type C ear muffs sold in December -/
def type_c_count : ℕ := 3186

/-- The price of each type C ear muff -/
def type_c_price : ℚ := 74/10

/-- The total amount spent on ear muffs in December -/
def total_spent : ℚ := type_b_count * type_b_price + type_c_count * type_c_price

theorem december_ear_muff_sales :
  total_spent = 460566/10 := by sorry

end december_ear_muff_sales_l59_5949


namespace parallel_tangents_imply_a_range_l59_5978

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then -x^2 + (2*a - 2)*x
  else x^3 - (3*a + 3)*x^2 + a*x

-- Define the derivative of f(x)
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then -2*x + 2*a - 2
  else 3*x^2 - 6*(a + 1)*x + a

-- Theorem statement
theorem parallel_tangents_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f_prime a x₁ = f_prime a x₂ ∧ f_prime a x₂ = f_prime a x₃) →
  -1 < a ∧ a < 2 :=
by sorry

end parallel_tangents_imply_a_range_l59_5978


namespace school_play_tickets_l59_5900

theorem school_play_tickets (total_money : ℕ) (adult_price : ℕ) (child_price : ℕ) (total_tickets : ℕ) :
  total_money = 104 →
  adult_price = 6 →
  child_price = 4 →
  total_tickets = 21 →
  ∃ (adult_tickets : ℕ) (child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_tickets * adult_price + child_tickets * child_price = total_money ∧
    child_tickets = 11 :=
by
  sorry

end school_play_tickets_l59_5900


namespace triangle_side_sum_l59_5931

theorem triangle_side_sum (a b c : ℝ) (A B C : ℝ) :
  c = 2 →
  C = π / 3 →
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 →
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos C →
  a + b = 4 := by
sorry

end triangle_side_sum_l59_5931


namespace b_55_divisible_by_55_l59_5967

/-- Function that generates b_n as described in the problem -/
def b (n : ℕ) : ℕ := sorry

/-- Theorem stating that b(55) is divisible by 55 -/
theorem b_55_divisible_by_55 : 55 ∣ b 55 := by sorry

end b_55_divisible_by_55_l59_5967


namespace exists_non_identity_same_image_l59_5902

/-- Given two finite groups G and H, and two surjective but non-injective homomorphisms φ and ψ from G to H,
    there exists a non-identity element g in G such that φ(g) = ψ(g). -/
theorem exists_non_identity_same_image 
  {G H : Type*} [Group G] [Group H] [Fintype G] [Fintype H]
  (φ ψ : G →* H) 
  (hφ_surj : Function.Surjective φ) (hψ_surj : Function.Surjective ψ)
  (hφ_non_inj : ¬Function.Injective φ) (hψ_non_inj : ¬Function.Injective ψ) :
  ∃ g : G, g ≠ 1 ∧ φ g = ψ g := by
  sorry

end exists_non_identity_same_image_l59_5902


namespace hockey_league_games_l59_5903

structure Team where
  games_played : ℕ
  games_won : ℕ
  win_ratio : ℚ

def team_X : Team → Prop
| t => t.win_ratio = 3/4

def team_Y : Team → Prop
| t => t.win_ratio = 2/3

theorem hockey_league_games (X Y : Team) : 
  team_X X → team_Y Y → 
  Y.games_played = X.games_played + 12 →
  Y.games_won = X.games_won + 4 →
  X.games_played = 48 := by
  sorry

end hockey_league_games_l59_5903


namespace line_through_points_l59_5983

/-- Given a line y = ax + b passing through points (3,7) and (7,19), prove that a - b = 5 -/
theorem line_through_points (a b : ℝ) : 
  (3 * a + b = 7) → (7 * a + b = 19) → a - b = 5 := by sorry

end line_through_points_l59_5983


namespace trig_identity_l59_5944

theorem trig_identity (α : Real) 
  (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := by
  sorry

end trig_identity_l59_5944


namespace vincent_stickers_l59_5913

/-- The number of packs Vincent bought yesterday -/
def yesterday_packs : ℕ := 15

/-- The total number of packs Vincent has -/
def total_packs : ℕ := 40

/-- The number of additional packs Vincent bought today -/
def additional_packs : ℕ := total_packs - yesterday_packs

theorem vincent_stickers :
  additional_packs = 10 ∧ additional_packs > 0 := by
  sorry

end vincent_stickers_l59_5913


namespace gasoline_price_increase_l59_5901

theorem gasoline_price_increase 
  (spending_increase : Real) 
  (quantity_decrease : Real) 
  (price_increase : Real) : 
  spending_increase = 0.15 → 
  quantity_decrease = 0.08000000000000007 → 
  (1 + price_increase) * (1 - quantity_decrease) = 1 + spending_increase → 
  price_increase = 0.25 := by
sorry

end gasoline_price_increase_l59_5901


namespace count_solutions_quadratic_congruence_l59_5954

theorem count_solutions_quadratic_congruence (p : Nat) (a : Int) 
  (h_p : p.Prime ∧ p > 2) :
  let S := {(x, y) : Fin p × Fin p | (x.val^2 + y.val^2) % p = a % p}
  Fintype.card S = p + 1 := by
sorry

end count_solutions_quadratic_congruence_l59_5954


namespace stock_trade_profit_l59_5943

/-- Represents the stock trading scenario --/
structure StockTrade where
  initial_price : ℝ
  price_changes : List ℝ
  num_shares : ℕ
  buying_fee : ℝ
  selling_fee : ℝ
  transaction_tax : ℝ

/-- Calculates the final price of the stock --/
def final_price (trade : StockTrade) : ℝ :=
  trade.initial_price + trade.price_changes.sum

/-- Calculates the profit from the stock trade --/
def calculate_profit (trade : StockTrade) : ℝ :=
  let cost := trade.initial_price * trade.num_shares * (1 + trade.buying_fee)
  let revenue := (final_price trade) * trade.num_shares * (1 - trade.selling_fee - trade.transaction_tax)
  revenue - cost

/-- Theorem stating that the profit from the given stock trade is 889.5 yuan --/
theorem stock_trade_profit (trade : StockTrade) 
  (h1 : trade.initial_price = 27)
  (h2 : trade.price_changes = [4, 4.5, -1, -2.5, -6, 2])
  (h3 : trade.num_shares = 1000)
  (h4 : trade.buying_fee = 0.0015)
  (h5 : trade.selling_fee = 0.0015)
  (h6 : trade.transaction_tax = 0.001) :
  calculate_profit trade = 889.5 := by
  sorry

end stock_trade_profit_l59_5943


namespace bus_encounters_l59_5993

-- Define the schedule and travel time
def austin_departure_interval : ℕ := 2
def sanantonio_departure_interval : ℕ := 2
def sanantonio_departure_offset : ℕ := 1
def travel_time : ℕ := 7

-- Define the number of encounters
def encounters : ℕ := 4

-- Theorem statement
theorem bus_encounters :
  (austin_departure_interval = 2) →
  (sanantonio_departure_interval = 2) →
  (sanantonio_departure_offset = 1) →
  (travel_time = 7) →
  (encounters = 4) := by
  sorry

end bus_encounters_l59_5993


namespace constant_term_is_60_l59_5975

/-- The constant term in the binomial expansion of (2x^2 - 1/x)^6 -/
def constant_term : ℤ :=
  (Finset.range 7).sum (fun r => 
    (-1)^r * (Nat.choose 6 r) * 2^(6-r) * 
    if 12 - 3*r = 0 then 1 else 0)

/-- The constant term in the binomial expansion of (2x^2 - 1/x)^6 is 60 -/
theorem constant_term_is_60 : constant_term = 60 := by
  sorry

end constant_term_is_60_l59_5975


namespace m_range_proof_l59_5989

theorem m_range_proof (h : ∀ x, (|x - m| < 1) ↔ (1/3 < x ∧ x < 1/2)) :
  -1/2 ≤ m ∧ m ≤ 4/3 :=
by sorry

end m_range_proof_l59_5989


namespace wire_ratio_l59_5919

/-- Given a wire of total length 60 cm with a shorter piece of 20 cm,
    prove that the ratio of the shorter piece to the longer piece is 1/2. -/
theorem wire_ratio (total_length : ℝ) (shorter_piece : ℝ) 
  (h1 : total_length = 60)
  (h2 : shorter_piece = 20)
  (h3 : shorter_piece < total_length) :
  shorter_piece / (total_length - shorter_piece) = 1 / 2 := by
sorry

end wire_ratio_l59_5919


namespace shooter_stability_l59_5917

/-- A shooter's score set -/
structure ScoreSet where
  scores : Finset ℝ
  card_eq : scores.card = 10

/-- Standard deviation of a score set -/
def standardDeviation (s : ScoreSet) : ℝ := sorry

/-- Dispersion of a score set -/
def dispersion (s : ScoreSet) : ℝ := sorry

/-- Larger standard deviation implies greater dispersion -/
axiom std_dev_dispersion_relation (s₁ s₂ : ScoreSet) :
  standardDeviation s₁ > standardDeviation s₂ → dispersion s₁ > dispersion s₂

theorem shooter_stability (A B : ScoreSet) :
  standardDeviation A > standardDeviation B →
  dispersion A > dispersion B :=
by sorry

end shooter_stability_l59_5917


namespace max_min_x_plus_y_l59_5918

theorem max_min_x_plus_y (x y : ℝ) :
  (|x + 2| + |1 - x| = 9 - |y - 5| - |1 + y|) →
  (∃ (a b : ℝ), (∀ z w : ℝ, |z + 2| + |1 - z| = 9 - |w - 5| - |1 + w| → x + y ≤ a ∧ b ≤ z + w) ∧
                 a = 6 ∧ b = -3) :=
by sorry

end max_min_x_plus_y_l59_5918


namespace trig_identity_proof_l59_5925

theorem trig_identity_proof : 
  Real.sin (17 * π / 180) * Real.cos (43 * π / 180) + 
  Real.sin (73 * π / 180) * Real.sin (43 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end trig_identity_proof_l59_5925


namespace uncle_li_parking_duration_l59_5984

/-- Calculates the parking duration given the total amount paid and the fee structure -/
def parking_duration (total_paid : ℚ) (first_hour_fee : ℚ) (additional_half_hour_fee : ℚ) : ℚ :=
  (total_paid - first_hour_fee) / (additional_half_hour_fee / (1/2)) + 1

theorem uncle_li_parking_duration :
  let total_paid : ℚ := 25/2
  let first_hour_fee : ℚ := 5/2
  let additional_half_hour_fee : ℚ := 5/2
  parking_duration total_paid first_hour_fee additional_half_hour_fee = 3 := by
sorry

end uncle_li_parking_duration_l59_5984


namespace movie_ticket_change_l59_5956

/-- Calculates the change received by two sisters buying movie tickets -/
theorem movie_ticket_change (full_price : ℚ) (discount_percent : ℚ) (brought_money : ℚ) : 
  full_price = 8 →
  discount_percent = 25 / 100 →
  brought_money = 25 →
  let discounted_price := full_price * (1 - discount_percent)
  let total_cost := full_price + discounted_price
  brought_money - total_cost = 11 := by
sorry

end movie_ticket_change_l59_5956


namespace cube_surface_area_from_prism_l59_5958

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_from_prism (a b c : ℝ) (h1 : a = 5) (h2 : b = 4) (h3 : c = 40) :
  6 * (((a * b * c) ^ (1/3 : ℝ)) ^ 2) = 600 := by
  sorry

end cube_surface_area_from_prism_l59_5958


namespace sum_of_solutions_l59_5974

theorem sum_of_solutions (x : ℝ) : (x + 16 / x = 12) → (∃ y : ℝ, y + 16 / y = 12 ∧ y ≠ x) → x + y = 12 :=
by sorry

end sum_of_solutions_l59_5974


namespace expression_equality_l59_5926

theorem expression_equality (y a : ℝ) (h1 : y > 0) 
  (h2 : (a * y) / 20 + (3 * y) / 10 = 0.6 * y) : a = 6 := by
  sorry

end expression_equality_l59_5926


namespace nancy_savings_l59_5904

-- Define the number of quarters in a dozen
def dozen_quarters : ℕ := 12

-- Define the value of a quarter in cents
def quarter_value : ℕ := 25

-- Define the number of cents in a dollar
def cents_per_dollar : ℕ := 100

-- Theorem to prove
theorem nancy_savings : (dozen_quarters * quarter_value) / cents_per_dollar = 3 := by
  sorry

end nancy_savings_l59_5904


namespace solution_x_l59_5973

-- Define m and n as distinct non-zero real constants
variable (m n : ℝ) (h : m ≠ n) (hm : m ≠ 0) (hn : n ≠ 0)

-- Define the equation
def equation (x : ℝ) : Prop :=
  (x + m)^2 - 3*(x + n)^2 = m^2 - 3*n^2

-- Theorem statement
theorem solution_x (x : ℝ) : 
  equation m n x → (x = 0 ∨ x = m - 3*n) :=
by sorry

end solution_x_l59_5973


namespace abs_opposite_equal_l59_5916

theorem abs_opposite_equal (x : ℝ) : |x| = |-x| := by sorry

end abs_opposite_equal_l59_5916


namespace smallest_number_l59_5923

theorem smallest_number (π : ℝ) (h : π > 0) : min (-π) (min (-2) (min 0 (Real.sqrt 3))) = -π := by
  sorry

end smallest_number_l59_5923


namespace g_zero_at_three_l59_5942

/-- The polynomial function g(x) -/
def g (x s : ℝ) : ℝ := 3*x^5 - 2*x^4 + x^3 - 4*x^2 + 5*x + s

/-- Theorem stating that g(3) = 0 when s = -573 -/
theorem g_zero_at_three : g 3 (-573) = 0 := by sorry

end g_zero_at_three_l59_5942


namespace cosine_triple_angle_identity_l59_5909

theorem cosine_triple_angle_identity (x : ℝ) : 
  4 * Real.cos x * Real.cos (x + π/3) * Real.cos (x - π/3) = Real.cos (3*x) := by
  sorry

end cosine_triple_angle_identity_l59_5909


namespace circular_arrangement_multiple_of_four_l59_5986

/-- Represents a child in the circular arrangement -/
inductive Child
| Boy
| Girl

/-- Represents the circular arrangement of children -/
def CircularArrangement := List Child

/-- Counts the number of children whose right-hand neighbor is of the same gender -/
def countSameGenderNeighbors (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Counts the number of children whose right-hand neighbor is of a different gender -/
def countDifferentGenderNeighbors (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Checks if the arrangement satisfies the equal neighbor condition -/
def hasEqualNeighbors (arrangement : CircularArrangement) : Prop :=
  countSameGenderNeighbors arrangement = countDifferentGenderNeighbors arrangement

theorem circular_arrangement_multiple_of_four 
  (arrangement : CircularArrangement) 
  (h : hasEqualNeighbors arrangement) :
  ∃ k : Nat, arrangement.length = 4 * k :=
sorry

end circular_arrangement_multiple_of_four_l59_5986


namespace max_posters_purchasable_l59_5911

def initial_amount : ℕ := 20
def book1_price : ℕ := 8
def book2_price : ℕ := 4
def poster_price : ℕ := 4

theorem max_posters_purchasable :
  (initial_amount - book1_price - book2_price) / poster_price = 2 := by
  sorry

end max_posters_purchasable_l59_5911


namespace simplify_expression_l59_5972

theorem simplify_expression (x : ℝ) : x + 3 - 4*x - 5 + 6*x + 7 - 8*x - 9 = -5*x - 4 := by
  sorry

end simplify_expression_l59_5972


namespace point_not_on_line_l59_5912

theorem point_not_on_line (m b : ℝ) (h : m * b > 0) : 
  ¬(∃ y : ℝ, y = 3 * m * 4 + 4 * b ∧ y = 0) :=
sorry

end point_not_on_line_l59_5912


namespace rectangular_prism_diagonal_l59_5930

theorem rectangular_prism_diagonal (l w h : ℝ) (hl : l = 6) (hw : w = 8) (hh : h = 15) :
  Real.sqrt (l^2 + w^2 + h^2) = Real.sqrt 325 :=
by sorry

end rectangular_prism_diagonal_l59_5930


namespace library_visitors_library_visitors_proof_l59_5941

/-- Calculates the average number of visitors on non-Sunday days in a library -/
theorem library_visitors (sunday_avg : ℕ) (month_avg : ℕ) : ℕ :=
  let total_days : ℕ := 30
  let sunday_count : ℕ := 5
  let other_days : ℕ := total_days - sunday_count
  let total_visitors : ℕ := month_avg * total_days
  let sunday_visitors : ℕ := sunday_avg * sunday_count
  (total_visitors - sunday_visitors) / other_days

/-- Proves that the average number of visitors on non-Sunday days is 240 -/
theorem library_visitors_proof :
  library_visitors 660 310 = 240 := by
sorry

end library_visitors_library_visitors_proof_l59_5941


namespace no_real_roots_l59_5908

theorem no_real_roots : ∀ x : ℝ, 2 * Real.cos (x / 2) ≠ 10^x + 10^(-x) + 1 := by
  sorry

end no_real_roots_l59_5908


namespace garage_cars_count_l59_5928

/-- The number of cars in Connor's garage -/
def num_cars : ℕ := 10

/-- The number of bicycles in the garage -/
def num_bicycles : ℕ := 20

/-- The number of motorcycles in the garage -/
def num_motorcycles : ℕ := 5

/-- The total number of wheels in the garage -/
def total_wheels : ℕ := 90

/-- The number of wheels on a bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a car -/
def wheels_per_car : ℕ := 4

/-- The number of wheels on a motorcycle -/
def wheels_per_motorcycle : ℕ := 2

theorem garage_cars_count :
  num_bicycles * wheels_per_bicycle +
  num_cars * wheels_per_car +
  num_motorcycles * wheels_per_motorcycle = total_wheels :=
by sorry

end garage_cars_count_l59_5928


namespace expression_equals_588_times_10_to_1007_l59_5960

theorem expression_equals_588_times_10_to_1007 :
  (3^1006 + 7^1007)^2 - (3^1006 - 7^1007)^2 = 588 * 10^1007 := by
  sorry

end expression_equals_588_times_10_to_1007_l59_5960


namespace function_maximum_value_l59_5982

/-- Given a function f(x) = x / (x^2 + a) where a > 0, 
    if its maximum value on [1, +∞) is √3/3, then a = √3 - 1 -/
theorem function_maximum_value (a : ℝ) : 
  a > 0 → 
  (∀ x : ℝ, x ≥ 1 → x / (x^2 + a) ≤ Real.sqrt 3 / 3) →
  (∃ x : ℝ, x ≥ 1 ∧ x / (x^2 + a) = Real.sqrt 3 / 3) →
  a = Real.sqrt 3 - 1 := by
  sorry

end function_maximum_value_l59_5982


namespace equation_solution_l59_5946

theorem equation_solution (x : ℝ) : 144 / 0.144 = 14.4 / x → x = 0.0144 := by
  sorry

end equation_solution_l59_5946


namespace birthday_party_guests_solve_birthday_party_guests_l59_5947

theorem birthday_party_guests : ℕ → Prop :=
  fun total_guests =>
    -- Define the number of women, men, and children
    let women := total_guests / 2
    let men := 15
    let children := total_guests - women - men

    -- Define the number of people who left
    let men_left := men / 3
    let children_left := 5

    -- Define the number of people who stayed
    let people_stayed := total_guests - men_left - children_left

    -- State the conditions and the conclusion
    women = men ∧
    women + men + children = total_guests ∧
    people_stayed = 50 ∧
    total_guests = 60

-- The proof of the theorem
theorem solve_birthday_party_guests : birthday_party_guests 60 := by
  sorry

#check solve_birthday_party_guests

end birthday_party_guests_solve_birthday_party_guests_l59_5947


namespace sufficient_not_necessary_condition_l59_5995

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 2 → x - 1 > 0) ∧
  ¬(∀ x : ℝ, x - 1 > 0 → x > 2) :=
by sorry

end sufficient_not_necessary_condition_l59_5995


namespace n_accurate_to_hundred_thousandth_l59_5950

/-- The number we're considering -/
def n : ℝ := 5.374e8

/-- Definition of accuracy to the hundred thousandth place -/
def accurate_to_hundred_thousandth (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k : ℝ) * 1e5

/-- Theorem stating that our number is accurate to the hundred thousandth place -/
theorem n_accurate_to_hundred_thousandth : accurate_to_hundred_thousandth n := by
  sorry

end n_accurate_to_hundred_thousandth_l59_5950


namespace all_functions_have_clever_value_point_l59_5966

-- Define the concept of a "clever value point"
def has_clever_value_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = deriv f x₀

-- State the theorem
theorem all_functions_have_clever_value_point :
  (has_clever_value_point (λ x : ℝ => x^2)) ∧
  (has_clever_value_point (λ x : ℝ => Real.exp (-x))) ∧
  (has_clever_value_point (λ x : ℝ => Real.log x)) ∧
  (has_clever_value_point (λ x : ℝ => Real.tan x)) :=
sorry

end all_functions_have_clever_value_point_l59_5966


namespace representatives_count_l59_5957

/-- The number of ways to select 3 representatives from 4 boys and 4 girls,
    with at least two girls among them. -/
def select_representatives : ℕ :=
  Nat.choose 4 3 + Nat.choose 4 2 * Nat.choose 4 1

/-- Theorem stating that the number of ways to select the representatives is 28. -/
theorem representatives_count : select_representatives = 28 := by
  sorry

end representatives_count_l59_5957


namespace equal_probability_events_l59_5979

/-- Given a jar with 'a' white balls and 'b' black balls, where a ≠ b, this theorem proves that
    the probability of Event A (at some point, the number of drawn white balls equals the number
    of drawn black balls) is equal to the probability of Event B (at some point, the number of
    white balls remaining in the jar equals the number of black balls remaining in the jar),
    and that this probability is (2 * min(a, b)) / (a + b). -/
theorem equal_probability_events (a b : ℕ) (h : a ≠ b) :
  let total := a + b
  let prob_A := (2 * min a b) / total
  let prob_B := (2 * min a b) / total
  prob_A = prob_B ∧ prob_A = (2 * min a b) / total := by
  sorry

#check equal_probability_events

end equal_probability_events_l59_5979


namespace point_A_in_second_quadrant_l59_5940

/-- A point is in the second quadrant if its x-coordinate is negative and its y-coordinate is positive -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The coordinates of point A -/
def point_A : ℝ × ℝ := (-3, 4)

/-- Theorem: Point A is located in the second quadrant -/
theorem point_A_in_second_quadrant :
  is_in_second_quadrant point_A.1 point_A.2 := by
  sorry

end point_A_in_second_quadrant_l59_5940


namespace triangle_least_perimeter_l59_5920

theorem triangle_least_perimeter (a b x : ℕ) : 
  a = 15 → b = 24 → x > 0 → 
  a + x > b → b + x > a → a + b > x → 
  (∀ y : ℕ, y > 0 → y + a > b → b + y > a → a + b > y → a + b + y ≥ a + b + x) →
  a + b + x = 49 :=
sorry

end triangle_least_perimeter_l59_5920


namespace square_root_of_difference_l59_5969

theorem square_root_of_difference (n : ℕ+) :
  Real.sqrt ((10^(2*n.val) - 1)/9 - 2*(10^n.val - 1)/9) = (10^n.val - 1)/3 := by
  sorry

end square_root_of_difference_l59_5969


namespace laptop_tote_weight_difference_l59_5963

/-- Represents the weights of various items in pounds -/
structure Weights where
  karens_tote : ℝ
  kevins_empty_briefcase : ℝ
  kevins_full_briefcase : ℝ
  kevins_work_papers : ℝ
  kevins_laptop : ℝ

/-- Conditions of the problem -/
def problem_conditions (w : Weights) : Prop :=
  w.karens_tote = 8 ∧
  w.karens_tote = 2 * w.kevins_empty_briefcase ∧
  w.kevins_full_briefcase = 2 * w.karens_tote ∧
  w.kevins_work_papers = (w.kevins_full_briefcase - w.kevins_empty_briefcase) / 6 ∧
  w.kevins_laptop = w.kevins_full_briefcase - w.kevins_empty_briefcase - w.kevins_work_papers

/-- The theorem to be proved -/
theorem laptop_tote_weight_difference (w : Weights) 
  (h : problem_conditions w) : w.kevins_laptop - w.karens_tote = 2 := by
  sorry

end laptop_tote_weight_difference_l59_5963


namespace exists_always_white_cell_l59_5985

-- Define the grid plane
def GridPlane := ℤ × ℤ

-- Define the state of a cell (Black or White)
inductive CellState
| Black
| White

-- Define the initial state of the grid
def initial_grid : GridPlane → CellState :=
  sorry

-- Define the polygon M
def M : Set GridPlane :=
  sorry

-- Axiom: M covers more than one cell
axiom M_size : ∃ (c1 c2 : GridPlane), c1 ≠ c2 ∧ c1 ∈ M ∧ c2 ∈ M

-- Define a valid shift of M
def valid_shift (s : GridPlane) : Prop :=
  sorry

-- Define the state of the grid after a shift
def shift_grid (g : GridPlane → CellState) (s : GridPlane) : GridPlane → CellState :=
  sorry

-- Define the state of the grid after any number of shifts
def final_grid : GridPlane → CellState :=
  sorry

-- The theorem to prove
theorem exists_always_white_cell :
  ∃ (c : GridPlane), final_grid c = CellState.White :=
sorry

end exists_always_white_cell_l59_5985


namespace sqrt_equation_solution_l59_5938

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 * x + 7) = 10 → x = 31 := by sorry

end sqrt_equation_solution_l59_5938


namespace necessary_sufficient_condition_l59_5935

theorem necessary_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0) ↔ a ≥ 3 := by
  sorry

end necessary_sufficient_condition_l59_5935


namespace equation_solution_l59_5998

theorem equation_solution : ∃! y : ℚ, 
  (y ≠ 3 ∧ y ≠ 5/4) ∧ 
  (y^2 - 7*y + 12)/(y - 3) + (4*y^2 + 20*y - 25)/(4*y - 5) = 2 ∧
  y = 1/2 := by
  sorry

end equation_solution_l59_5998


namespace largest_four_digit_product_of_primes_l59_5994

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a four-digit positive integer -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem largest_four_digit_product_of_primes :
  ∃ (n x y : ℕ),
    isFourDigit n ∧
    isPrime x ∧
    isPrime y ∧
    x < 10 ∧
    y < 10 ∧
    isPrime (10 * y + x) ∧
    n = x * y * (10 * y + x) ∧
    (∀ (m a b : ℕ),
      isFourDigit m →
      isPrime a →
      isPrime b →
      a < 10 →
      b < 10 →
      isPrime (10 * b + a) →
      m = a * b * (10 * b + a) →
      m ≤ n) ∧
    n = 1533 :=
  sorry

end largest_four_digit_product_of_primes_l59_5994


namespace square_difference_times_three_l59_5996

theorem square_difference_times_three : (538^2 - 462^2) * 3 = 228000 := by
  sorry

end square_difference_times_three_l59_5996


namespace discount_profit_percentage_l59_5932

theorem discount_profit_percentage 
  (discount : Real) 
  (no_discount_profit : Real) 
  (h1 : discount = 0.05) 
  (h2 : no_discount_profit = 0.26) : 
  let marked_price := 1 + no_discount_profit
  let selling_price := marked_price * (1 - discount)
  let profit := selling_price - 1
  profit * 100 = 19.7 := by
sorry

end discount_profit_percentage_l59_5932


namespace inverse_proportion_ratio_l59_5980

/-- Given x is inversely proportional to y, prove that y₁/y₂ = 4/3 when x₁/x₂ = 3/4 -/
theorem inverse_proportion_ratio (x y : ℝ → ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h_inverse : ∀ t : ℝ, t ≠ 0 → x t * y t = x₁ * y₁)
  (h_x₁_nonzero : x₁ ≠ 0)
  (h_x₂_nonzero : x₂ ≠ 0)
  (h_y₁_nonzero : y₁ ≠ 0)
  (h_y₂_nonzero : y₂ ≠ 0)
  (h_x_ratio : x₁ / x₂ = 3 / 4) :
  y₁ / y₂ = 4 / 3 := by
  sorry


end inverse_proportion_ratio_l59_5980


namespace acquainted_pairs_bound_l59_5914

/-- Represents a company with n persons, where each person has no more than d acquaintances,
    and there exists a group of k persons (k ≥ d) who are not acquainted with each other. -/
structure Company where
  n : ℕ  -- Total number of persons
  d : ℕ  -- Maximum number of acquaintances per person
  k : ℕ  -- Size of the group of unacquainted persons
  h1 : k ≥ d  -- Condition that k is not less than d

/-- The number of acquainted pairs in the company -/
def acquaintedPairs (c : Company) : ℕ := sorry

/-- Theorem stating that the number of acquainted pairs is not greater than ⌊n²/4⌋ -/
theorem acquainted_pairs_bound (c : Company) : 
  acquaintedPairs c ≤ (c.n^2) / 4 := by sorry

end acquainted_pairs_bound_l59_5914


namespace min_positive_temperatures_l59_5948

theorem min_positive_temperatures (n : ℕ) (pos_products neg_products : ℕ) :
  n = 11 →
  pos_products = 62 →
  neg_products = 48 →
  ∃ (pos_temps : ℕ), pos_temps ≥ 3 ∧
    pos_temps * (pos_temps - 1) = pos_products ∧
    (n - pos_temps) * (n - 1 - pos_temps) = neg_products ∧
    ∀ (k : ℕ), k < pos_temps →
      k * (k - 1) ≠ pos_products ∨ (n - k) * (n - 1 - k) ≠ neg_products :=
by sorry

end min_positive_temperatures_l59_5948


namespace range_of_a_l59_5937

-- Define the set of real numbers x that satisfy 0 < x < 2
def P : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Define the set of real numbers x that satisfy a-1 < x ≤ a
def Q (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x ≤ a}

-- State the theorem
theorem range_of_a : 
  (∀ a : ℝ, (Q a ⊆ P) ∧ (Q a ≠ P)) → 
  {a : ℝ | 1 ≤ a ∧ a < 2} = {a : ℝ | ∃ x : ℝ, x ∈ Q a} :=
sorry

end range_of_a_l59_5937


namespace binary_11011011_to_base4_l59_5987

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_base4 (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List (Fin 4) :=
      if m = 0 then [] else (m % 4) :: aux (m / 4)
    aux n |>.reverse

theorem binary_11011011_to_base4 :
  decimal_to_base4 (binary_to_decimal [true, true, false, true, true, false, true, true]) =
  [3, 1, 2, 3] :=
by sorry

end binary_11011011_to_base4_l59_5987


namespace largest_class_has_61_students_l59_5997

/-- Represents a school with a given number of classes and students. -/
structure School where
  num_classes : ℕ
  total_students : ℕ
  class_diff : ℕ

/-- Calculates the number of students in the largest class of a school. -/
def largest_class_size (s : School) : ℕ :=
  (s.total_students + s.class_diff * (s.num_classes - 1) * s.num_classes / 2) / s.num_classes

/-- Theorem stating that for a school with 8 classes, 380 total students,
    and 4 students difference between classes, the largest class has 61 students. -/
theorem largest_class_has_61_students :
  let s : School := { num_classes := 8, total_students := 380, class_diff := 4 }
  largest_class_size s = 61 := by
  sorry

#eval largest_class_size { num_classes := 8, total_students := 380, class_diff := 4 }

end largest_class_has_61_students_l59_5997


namespace sqrt_14_bounds_l59_5971

theorem sqrt_14_bounds : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 := by
  sorry

end sqrt_14_bounds_l59_5971


namespace triangle_properties_l59_5915

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  Real.sqrt 3 * Real.tan t.A * Real.tan t.B - Real.tan t.A - Real.tan t.B = Real.sqrt 3 ∧
  t.c = 2 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.C = Real.pi / 3 ∧ 20 / 3 < t.a^2 + t.b^2 ∧ t.a^2 + t.b^2 ≤ 8 := by
  sorry

end triangle_properties_l59_5915


namespace complex_sum_theorem_l59_5906

theorem complex_sum_theorem (A O P S : ℂ) 
  (hA : A = 2 + I) 
  (hO : O = 3 - 2*I) 
  (hP : P = 1 + I) 
  (hS : S = 4 + 3*I) : 
  A - O + P + S = 4 + 7*I :=
by sorry

end complex_sum_theorem_l59_5906


namespace trapezoid_angle_bisector_inscribed_circle_l59_5951

noncomputable section

/-- Represents a point in 2D space -/
structure Point := (x : ℝ) (y : ℝ)

/-- Represents a trapezoid ABCD -/
structure Trapezoid :=
  (A B C D : Point)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- The length of a line segment between two points -/
def distance (p q : Point) : ℝ := sorry

/-- The angle between three points -/
def angle (p q r : Point) : ℝ := sorry

/-- The angle bisector of an angle -/
def angleBisector (p q r : Point) : Point := sorry

/-- Check if a point lies on a line segment -/
def onSegment (p q r : Point) : Prop := sorry

/-- Check if a circle is inscribed in a triangle -/
def isInscribed (c : Circle) (t : Triangle) : Prop := sorry

/-- Check if a point is a tangent point of a circle on a line segment -/
def isTangentPoint (p : Point) (c : Circle) (q r : Point) : Prop := sorry

theorem trapezoid_angle_bisector_inscribed_circle 
  (ABCD : Trapezoid) (E M H : Point) (c : Circle) :
  onSegment E ABCD.B ABCD.C →
  angleBisector ABCD.B ABCD.A ABCD.D = E →
  isInscribed c (Triangle.mk ABCD.A ABCD.B E) →
  isTangentPoint M c ABCD.A ABCD.B →
  isTangentPoint H c ABCD.B E →
  distance ABCD.A ABCD.B = 2 →
  distance M H = 1 →
  angle ABCD.B ABCD.A ABCD.D = 120 * π / 180 := by
  sorry

end trapezoid_angle_bisector_inscribed_circle_l59_5951


namespace milk_water_ratio_l59_5961

theorem milk_water_ratio (initial_volume : ℚ) (initial_milk_ratio : ℚ) (initial_water_ratio : ℚ) (added_water : ℚ) : 
  initial_volume = 45 ∧ 
  initial_milk_ratio = 4 ∧ 
  initial_water_ratio = 1 ∧ 
  added_water = 18 → 
  let initial_milk := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let initial_water := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let final_water := initial_water + added_water
  let final_milk_ratio := initial_milk / final_water
  let final_water_ratio := final_water / final_water
  final_milk_ratio / final_water_ratio = 4 / 3 := by
sorry

end milk_water_ratio_l59_5961


namespace not_square_difference_l59_5955

/-- The square difference formula -/
def square_difference (p q : ℝ) : ℝ := p^2 - q^2

/-- Expression that cannot be directly represented by the square difference formula -/
def problematic_expression (a : ℝ) : ℝ := (a - 1) * (-a + 1)

/-- Theorem stating that the problematic expression cannot be directly represented
    by the square difference formula for any real values of p and q -/
theorem not_square_difference :
  ∀ (a p q : ℝ), problematic_expression a ≠ square_difference p q :=
by sorry

end not_square_difference_l59_5955


namespace quadratic_sine_interpolation_l59_5959

theorem quadratic_sine_interpolation (f : ℝ → ℝ) (h : f = λ x => -4 / Real.pi ^ 2 * x ^ 2 + 4 / Real.pi * x) :
  f 0 = 0 ∧ f (Real.pi / 2) = 1 ∧ f Real.pi = 0 := by
  sorry

end quadratic_sine_interpolation_l59_5959


namespace triangle_count_l59_5953

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of points on the circle -/
def num_points : ℕ := 9

/-- The number of points needed to form a triangle -/
def points_per_triangle : ℕ := 3

/-- The number of different triangles that can be formed -/
def num_triangles : ℕ := binomial num_points points_per_triangle

theorem triangle_count : num_triangles = 84 := by sorry

end triangle_count_l59_5953


namespace floor_sqrt_48_squared_l59_5970

theorem floor_sqrt_48_squared : ⌊Real.sqrt 48⌋^2 = 36 := by sorry

end floor_sqrt_48_squared_l59_5970


namespace inequality_solution_l59_5962

theorem inequality_solution (a x : ℝ) :
  (a < 0 ∨ a > 1 → (((x - a) / (x - a^2) < 0) ↔ (a < x ∧ x < a^2))) ∧
  (0 < a ∧ a < 1 → (((x - a) / (x - a^2) < 0) ↔ (a^2 < x ∧ x < a))) ∧
  (a = 0 ∨ a = 1 → ¬∃x, (x - a) / (x - a^2) < 0) :=
by sorry

end inequality_solution_l59_5962
