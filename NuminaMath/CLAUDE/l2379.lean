import Mathlib

namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_div_2_l2379_237921

theorem sin_cos_sum_equals_sqrt3_div_2 :
  Real.sin (17 * π / 180) * Real.cos (43 * π / 180) + 
  Real.sin (73 * π / 180) * Real.sin (43 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_div_2_l2379_237921


namespace NUMINAMATH_CALUDE_value_of_x_l2379_237958

theorem value_of_x : ∃ x : ℝ, 3 * x + 15 = (1/3) * (7 * x + 45) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l2379_237958


namespace NUMINAMATH_CALUDE_points_on_circle_l2379_237950

-- Define the points
variable (A B C X Y A' : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (acute_triangle : IsAcute A B C)
(X_side : DifferentSide X C (Line.throughPoints A B))
(Y_side : DifferentSide Y B (Line.throughPoints A C))
(BX_eq_AC : dist B X = dist A C)
(CY_eq_AB : dist C Y = dist A B)
(AX_eq_AY : dist A X = dist A Y)
(A'_reflection : IsReflection A A' (Perp.bisector B C))
(XY_diff_sides : DifferentSide X Y (Line.throughPoints A A'))

-- State the theorem
theorem points_on_circle :
  ∃ (O : EuclideanSpace ℝ (Fin 2)) (r : ℝ), 
    dist O A = r ∧ dist O A' = r ∧ dist O X = r ∧ dist O Y = r :=
sorry

end NUMINAMATH_CALUDE_points_on_circle_l2379_237950


namespace NUMINAMATH_CALUDE_intersection_condition_l2379_237941

-- Define the line l
def line (k x y : ℝ) : Prop := y + k*x + 2 = 0

-- Define the curve C in polar coordinates
def curve (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define the curve C in Cartesian coordinates
def curve_cartesian (x y : ℝ) : Prop := x^2 + y^2 = 2*x

-- Theorem stating the condition for intersection
theorem intersection_condition (k : ℝ) :
  (∃ x y : ℝ, line k x y ∧ curve_cartesian x y) → k ≤ -3/4 :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l2379_237941


namespace NUMINAMATH_CALUDE_max_value_with_constraint_l2379_237912

theorem max_value_with_constraint (x y z : ℝ) (h : 4 * x^2 + y^2 + 16 * z^2 = 1) :
  7 * x + 2 * y + 8 * z ≤ 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_with_constraint_l2379_237912


namespace NUMINAMATH_CALUDE_quadratic_inequality_integer_solution_l2379_237930

theorem quadratic_inequality_integer_solution (z : ℕ) :
  z^2 - 50*z + 550 ≤ 10 ↔ 20 ≤ z ∧ z ≤ 30 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_integer_solution_l2379_237930


namespace NUMINAMATH_CALUDE_wafting_pie_egg_usage_l2379_237971

/-- The Wafting Pie Company's egg usage problem -/
theorem wafting_pie_egg_usage 
  (total_eggs : ℕ) 
  (morning_eggs : ℕ) 
  (h1 : total_eggs = 1339)
  (h2 : morning_eggs = 816) :
  total_eggs - morning_eggs = 523 := by
  sorry

end NUMINAMATH_CALUDE_wafting_pie_egg_usage_l2379_237971


namespace NUMINAMATH_CALUDE_bar_chart_ratio_difference_l2379_237936

theorem bar_chart_ratio_difference 
  (a b c d : ℝ) 
  (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) :
  a / (a + b) - c / (c + d) = (a * d - b * c) / ((a + b) * (c + d)) :=
by sorry

end NUMINAMATH_CALUDE_bar_chart_ratio_difference_l2379_237936


namespace NUMINAMATH_CALUDE_barbara_butcher_cost_l2379_237993

/-- The cost of Barbara's purchase at the butcher's --/
def butcher_cost (steak_weight : ℝ) (steak_price : ℝ) (chicken_weight : ℝ) (chicken_price : ℝ) : ℝ :=
  steak_weight * steak_price + chicken_weight * chicken_price

/-- Theorem: Barbara's total cost at the butcher's is $79.50 --/
theorem barbara_butcher_cost :
  butcher_cost 4.5 15 1.5 8 = 79.5 := by
  sorry

end NUMINAMATH_CALUDE_barbara_butcher_cost_l2379_237993


namespace NUMINAMATH_CALUDE_stratified_sampling_specific_case_l2379_237970

/-- The number of ways to select students using stratified sampling -/
def stratified_sampling_ways (n_female : ℕ) (n_male : ℕ) (k_female : ℕ) (k_male : ℕ) : ℕ :=
  Nat.choose n_female k_female * Nat.choose n_male k_male

/-- Theorem stating the number of ways to select 5 students from 6 female and 4 male students -/
theorem stratified_sampling_specific_case :
  stratified_sampling_ways 6 4 3 2 = Nat.choose 6 3 * Nat.choose 4 2 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_specific_case_l2379_237970


namespace NUMINAMATH_CALUDE_lg_sum_equals_one_power_product_equals_forty_thousand_l2379_237961

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem for the first part
theorem lg_sum_equals_one : lg 2 + lg 5 = 1 := by sorry

-- Theorem for the second part
theorem power_product_equals_forty_thousand : 4 * (-100)^4 = 40000 := by sorry

end NUMINAMATH_CALUDE_lg_sum_equals_one_power_product_equals_forty_thousand_l2379_237961


namespace NUMINAMATH_CALUDE_local_minimum_at_negative_one_l2379_237968

open Real

/-- The function f(x) = xe^x has a local minimum at x = -1 -/
theorem local_minimum_at_negative_one (f : ℝ → ℝ) (h : ∀ x, f x = x * exp x) :
  IsLocalMin f (-1) :=
sorry

end NUMINAMATH_CALUDE_local_minimum_at_negative_one_l2379_237968


namespace NUMINAMATH_CALUDE_geometric_series_convergence_l2379_237962

/-- The sum of the infinite geometric series with first term 5/3 and common ratio 1/3 -/
def geometric_series_sum : ℚ :=
  let a := 5 / 3
  let r := 1 / 3
  a / (1 - r)

/-- The infinite geometric series 5/3 - 5/9 + 5/81 - 5/729 + ... converges to 5/2 -/
theorem geometric_series_convergence : geometric_series_sum = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_convergence_l2379_237962


namespace NUMINAMATH_CALUDE_exp_convex_and_ln_concave_l2379_237995

-- Define the exponential function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the natural logarithm function
noncomputable def g (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem exp_convex_and_ln_concave :
  (∀ x y : ℝ, ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 → ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    g (t * x + (1 - t) * y) ≥ t * g x + (1 - t) * g y) :=
by sorry

end NUMINAMATH_CALUDE_exp_convex_and_ln_concave_l2379_237995


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l2379_237960

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x ≤ 1}
def N : Set ℝ := {x | x^2 - 4 < 0}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.compl M) ∩ N = {x | 1 < x ∧ x < 2} :=
sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l2379_237960


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2379_237975

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, x * (2 * x + 3) < 15 ↔ x ∈ Set.Ioo (-5/2) 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2379_237975


namespace NUMINAMATH_CALUDE_unique_line_through_point_with_equal_intercepts_l2379_237974

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The point (2, 1) -/
def point : ℝ × ℝ := (2, 1)

/-- A line passes through a given point -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

/-- A line has equal intercepts on both axes -/
def equal_intercepts (l : Line) : Prop :=
  ∃ a : ℝ, l.intercept = a ∧ (-l.intercept / l.slope) = a

/-- There exists a unique line passing through (2, 1) with equal intercepts -/
theorem unique_line_through_point_with_equal_intercepts :
  ∃! l : Line, passes_through l point ∧ equal_intercepts l := by
  sorry

end NUMINAMATH_CALUDE_unique_line_through_point_with_equal_intercepts_l2379_237974


namespace NUMINAMATH_CALUDE_no_divisible_by_five_l2379_237999

def g (x : ℤ) : ℤ := x^2 + 5*x + 3

def T : Set ℤ := {x | 0 ≤ x ∧ x ≤ 30}

theorem no_divisible_by_five : 
  ∀ t ∈ T, ¬(g t % 5 = 0) := by
sorry

end NUMINAMATH_CALUDE_no_divisible_by_five_l2379_237999


namespace NUMINAMATH_CALUDE_joseph_card_distribution_l2379_237988

theorem joseph_card_distribution (initial_cards : ℕ) (cards_per_student : ℕ) (remaining_cards : ℕ) :
  initial_cards = 357 →
  cards_per_student = 23 →
  remaining_cards = 12 →
  ∃ (num_students : ℕ), num_students = 15 ∧ initial_cards = cards_per_student * num_students + remaining_cards :=
by sorry

end NUMINAMATH_CALUDE_joseph_card_distribution_l2379_237988


namespace NUMINAMATH_CALUDE_eugene_model_house_l2379_237907

/-- The number of toothpicks required for each card -/
def toothpicks_per_card : ℕ := 75

/-- The total number of cards in a deck -/
def cards_in_deck : ℕ := 52

/-- The number of cards Eugene didn't use -/
def unused_cards : ℕ := 16

/-- The number of toothpicks in each box -/
def toothpicks_per_box : ℕ := 450

/-- Calculate the number of boxes of toothpicks Eugene used -/
def boxes_used : ℕ :=
  let cards_used := cards_in_deck - unused_cards
  let total_toothpicks := cards_used * toothpicks_per_card
  (total_toothpicks + toothpicks_per_box - 1) / toothpicks_per_box

theorem eugene_model_house :
  boxes_used = 6 := by
  sorry

end NUMINAMATH_CALUDE_eugene_model_house_l2379_237907


namespace NUMINAMATH_CALUDE_aquarium_water_volume_l2379_237914

/-- The initial volume of water in the aquarium -/
def initial_volume : ℝ := 36

/-- The volume of water after the cat spills half and Nancy triples the remainder -/
def final_volume : ℝ := 54

theorem aquarium_water_volume : 
  (3 * (initial_volume / 2)) = final_volume :=
by sorry

end NUMINAMATH_CALUDE_aquarium_water_volume_l2379_237914


namespace NUMINAMATH_CALUDE_samson_sandwich_difference_l2379_237928

/-- The number of sandwiches Samson ate for lunch on Monday -/
def monday_lunch : ℕ := 3

/-- The number of sandwiches Samson ate for dinner on Monday -/
def monday_dinner : ℕ := 2 * monday_lunch

/-- The number of sandwiches Samson ate for breakfast on Tuesday -/
def tuesday_breakfast : ℕ := 1

/-- The total number of sandwiches Samson ate on Monday -/
def monday_total : ℕ := monday_lunch + monday_dinner

/-- The difference between the number of sandwiches Samson ate on Monday and Tuesday -/
def sandwich_difference : ℕ := monday_total - tuesday_breakfast

theorem samson_sandwich_difference : sandwich_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_samson_sandwich_difference_l2379_237928


namespace NUMINAMATH_CALUDE_ship_passengers_l2379_237902

theorem ship_passengers : ∀ (P : ℕ),
  (P / 12 : ℚ) + (P / 4 : ℚ) + (P / 9 : ℚ) + (P / 6 : ℚ) + 42 = P →
  P = 108 := by
  sorry

end NUMINAMATH_CALUDE_ship_passengers_l2379_237902


namespace NUMINAMATH_CALUDE_big_n_conference_teams_l2379_237934

theorem big_n_conference_teams (n : ℕ) : n * (n - 1) / 2 = 28 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_big_n_conference_teams_l2379_237934


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l2379_237946

theorem closest_integer_to_cube_root : 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |m - (7^3 + 9^3)^(1/3)| ≥ |n - (7^3 + 9^3)^(1/3)| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l2379_237946


namespace NUMINAMATH_CALUDE_number_of_hiding_snakes_l2379_237948

/-- Given a cage with snakes, some of which are hiding, this theorem proves
    the number of hiding snakes. -/
theorem number_of_hiding_snakes
  (total_snakes : ℕ)
  (visible_snakes : ℕ)
  (h1 : total_snakes = 95)
  (h2 : visible_snakes = 31) :
  total_snakes - visible_snakes = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_of_hiding_snakes_l2379_237948


namespace NUMINAMATH_CALUDE_buttons_per_shirt_l2379_237967

/-- Given Jack's shirt-making scenario, prove the number of buttons per shirt. -/
theorem buttons_per_shirt (num_kids : ℕ) (shirts_per_kid : ℕ) (total_buttons : ℕ) : 
  num_kids = 3 →
  shirts_per_kid = 3 →
  total_buttons = 63 →
  ∃ (buttons_per_shirt : ℕ), 
    buttons_per_shirt * (num_kids * shirts_per_kid) = total_buttons ∧
    buttons_per_shirt = 7 :=
by sorry

end NUMINAMATH_CALUDE_buttons_per_shirt_l2379_237967


namespace NUMINAMATH_CALUDE_arcsin_sqrt3_div2_l2379_237955

theorem arcsin_sqrt3_div2 : Real.arcsin (Real.sqrt 3 / 2) = π / 3 := by sorry

end NUMINAMATH_CALUDE_arcsin_sqrt3_div2_l2379_237955


namespace NUMINAMATH_CALUDE_reciprocal_problem_l2379_237926

theorem reciprocal_problem (x : ℚ) : 8 * x = 16 → 200 * (1 / x) = 100 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l2379_237926


namespace NUMINAMATH_CALUDE_parabola_equation_hyperbola_equation_l2379_237919

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/64 + y^2/16 = 1

-- Define the parabola focus
def parabola_focus : ℝ × ℝ := (-8, 0)

-- Define the hyperbola asymptotes
def hyperbola_asymptote (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem for the parabola equation
theorem parabola_equation : 
  ∃ (x y : ℝ), (x, y) = parabola_focus → y^2 = -32*x := by sorry

-- Theorem for the hyperbola equation
theorem hyperbola_equation :
  (∀ (x y : ℝ), ellipse x y ↔ ellipse (-x) y) → 
  (∀ (x y : ℝ), hyperbola_asymptote x y) →
  ∃ (x y : ℝ), x^2/12 - y^2/36 = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_hyperbola_equation_l2379_237919


namespace NUMINAMATH_CALUDE_min_value_theorem_l2379_237965

/-- Given a function f(x) = x(x-a)(x-b) where f'(0) = 4, 
    the minimum value of a^2 + 2b^2 is 8√2 -/
theorem min_value_theorem (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x * (x - a) * (x - b)
  let f' : ℝ → ℝ := λ x => 3*x^2 - 2*(a+b)*x + a*b
  (f' 0 = 4) → (∀ a b : ℝ, a^2 + 2*b^2 ≥ 8 * Real.sqrt 2) ∧ 
  (∃ a b : ℝ, a^2 + 2*b^2 = 8 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_min_value_theorem_l2379_237965


namespace NUMINAMATH_CALUDE_carrie_weeks_to_buy_iphone_l2379_237966

def iphone_cost : ℕ := 1200
def trade_in_value : ℕ := 180
def weekly_earnings : ℕ := 50

def weeks_needed : ℕ :=
  (iphone_cost - trade_in_value + weekly_earnings - 1) / weekly_earnings

theorem carrie_weeks_to_buy_iphone :
  weeks_needed = 21 :=
sorry

end NUMINAMATH_CALUDE_carrie_weeks_to_buy_iphone_l2379_237966


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l2379_237929

theorem discount_percentage_calculation (marked_price : ℝ) (h1 : marked_price > 0) : 
  let cost_price := 0.64 * marked_price
  let gain := 0.375 * cost_price
  let selling_price := cost_price + gain
  let discount := marked_price - selling_price
  (discount / marked_price) * 100 = 12 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_calculation_l2379_237929


namespace NUMINAMATH_CALUDE_largest_band_size_l2379_237989

theorem largest_band_size :
  ∀ m r x : ℕ,
  m = r * x + 3 →
  m = (r - 1) * (x + 2) →
  m < 100 →
  ∃ m_max : ℕ,
  m_max = 69 ∧
  ∀ m' : ℕ,
  (∃ r' x' : ℕ, m' = r' * x' + 3 ∧ m' = (r' - 1) * (x' + 2) ∧ m' < 100) →
  m' ≤ m_max :=
sorry

end NUMINAMATH_CALUDE_largest_band_size_l2379_237989


namespace NUMINAMATH_CALUDE_area_of_sixth_rectangle_l2379_237982

/-- Given a rectangle divided into six smaller rectangles, prove that if five of these rectangles
    have areas 126, 63, 40, 20, and 161, then the area of the sixth rectangle is 101. -/
theorem area_of_sixth_rectangle (
  total_area : ℝ)
  (area1 area2 area3 area4 area5 : ℝ)
  (h1 : area1 = 126)
  (h2 : area2 = 63)
  (h3 : area3 = 40)
  (h4 : area4 = 20)
  (h5 : area5 = 161)
  (h_sum : total_area = area1 + area2 + area3 + area4 + area5 + (total_area - (area1 + area2 + area3 + area4 + area5))) :
  total_area - (area1 + area2 + area3 + area4 + area5) = 101 := by
  sorry


end NUMINAMATH_CALUDE_area_of_sixth_rectangle_l2379_237982


namespace NUMINAMATH_CALUDE_unique_function_solution_l2379_237938

theorem unique_function_solution (f : ℝ → ℝ) 
  (h1 : ∀ x, f (f x * f (1 - x)) = f x) 
  (h2 : ∀ x, f (f x) = 1 - f x) : 
  ∀ x, f x = (1 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_unique_function_solution_l2379_237938


namespace NUMINAMATH_CALUDE_sales_third_month_l2379_237947

def sales_problem (m1 m2 m4 m5 m6 avg : ℚ) : ℚ :=
  let total_sales := avg * 6
  let known_sales := m1 + m2 + m4 + m5 + m6
  total_sales - known_sales

theorem sales_third_month
  (m1 m2 m4 m5 m6 avg : ℚ)
  (h_avg : avg = 6600)
  (h_m1 : m1 = 6435)
  (h_m2 : m2 = 6927)
  (h_m4 : m4 = 7230)
  (h_m5 : m5 = 6562)
  (h_m6 : m6 = 5591) :
  sales_problem m1 m2 m4 m5 m6 avg = 14085 := by
  sorry

#eval sales_problem 6435 6927 7230 6562 5591 6600

end NUMINAMATH_CALUDE_sales_third_month_l2379_237947


namespace NUMINAMATH_CALUDE_min_product_of_prime_sum_l2379_237977

theorem min_product_of_prime_sum (m n p : ℕ) : 
  Prime m → Prime n → Prime p → 
  m ≠ n → m ≠ p → n ≠ p →
  m + n = p →
  (∀ m' n' p' : ℕ, Prime m' → Prime n' → Prime p' → 
    m' ≠ n' → m' ≠ p' → n' ≠ p' →
    m' + n' = p' → m' * n' * p' ≥ m * n * p) →
  m * n * p = 30 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_prime_sum_l2379_237977


namespace NUMINAMATH_CALUDE_deriv_sin_plus_cos_at_pi_fourth_l2379_237908

/-- The derivative of sin(x) + cos(x) at π/4 is 0 -/
theorem deriv_sin_plus_cos_at_pi_fourth (f : ℝ → ℝ) (h : f = λ x => Real.sin x + Real.cos x) :
  deriv f (π / 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_deriv_sin_plus_cos_at_pi_fourth_l2379_237908


namespace NUMINAMATH_CALUDE_sum_of_digits_45_times_40_l2379_237925

def product_45_40 : ℕ := 45 * 40

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_45_times_40 : sum_of_digits product_45_40 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_45_times_40_l2379_237925


namespace NUMINAMATH_CALUDE_four_days_left_l2379_237937

/-- The number of days needed to complete toy production -/
def days_left (total_toys : ℕ) (daily_production : ℕ) (days_worked : ℕ) : ℕ :=
  (total_toys - daily_production * days_worked) / daily_production

/-- Theorem: 4 days are left to complete 1000 toys at 100 toys per day after 6 days of work -/
theorem four_days_left : days_left 1000 100 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_days_left_l2379_237937


namespace NUMINAMATH_CALUDE_zoo_arrangement_count_l2379_237973

def num_lions : Nat := 3
def num_zebras : Nat := 4
def num_monkeys : Nat := 6
def total_animals : Nat := num_lions + num_zebras + num_monkeys

theorem zoo_arrangement_count :
  (Nat.factorial 3) * (Nat.factorial num_lions) * (Nat.factorial num_zebras) * (Nat.factorial num_monkeys) = 622080 :=
by sorry

end NUMINAMATH_CALUDE_zoo_arrangement_count_l2379_237973


namespace NUMINAMATH_CALUDE_mess_expense_increase_l2379_237935

theorem mess_expense_increase
  (initial_students : ℕ)
  (new_students : ℕ)
  (original_expenditure : ℕ)
  (average_decrease : ℕ)
  (h1 : initial_students = 35)
  (h2 : new_students = 7)
  (h3 : original_expenditure = 420)
  (h4 : average_decrease = 1)
  : (initial_students + new_students) * 
    (original_expenditure / initial_students - average_decrease) - 
    original_expenditure = 42 := by
  sorry

end NUMINAMATH_CALUDE_mess_expense_increase_l2379_237935


namespace NUMINAMATH_CALUDE_sqrt_of_four_is_plus_minus_two_l2379_237994

-- Define the square root function
def sqrt (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

-- Theorem statement
theorem sqrt_of_four_is_plus_minus_two : sqrt 4 = {2, -2} := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_four_is_plus_minus_two_l2379_237994


namespace NUMINAMATH_CALUDE_smallest_class_size_is_42_l2379_237987

/-- Represents the number of students in a physical education class. -/
def ClassSize (n : ℕ) : ℕ := 5 * n + 2

/-- The smallest class size satisfying the given conditions -/
def SmallestClassSize : ℕ := 42

theorem smallest_class_size_is_42 :
  (∀ m : ℕ, ClassSize m > 40 → m ≥ SmallestClassSize) ∧
  (ClassSize (SmallestClassSize - 1) ≤ 40) ∧
  (ClassSize SmallestClassSize > 40) :=
sorry

end NUMINAMATH_CALUDE_smallest_class_size_is_42_l2379_237987


namespace NUMINAMATH_CALUDE_binary_11111011111_equals_2015_l2379_237963

def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_11111011111_equals_2015 :
  binary_to_decimal [true, true, true, true, true, false, true, true, true, true, true] = 2015 := by
  sorry

end NUMINAMATH_CALUDE_binary_11111011111_equals_2015_l2379_237963


namespace NUMINAMATH_CALUDE_x_power_plus_inverse_l2379_237980

theorem x_power_plus_inverse (θ : ℝ) (x : ℂ) (n : ℤ) 
  (h1 : 0 < θ) (h2 : θ < π) (h3 : x + 1/x = 2 * Real.cos θ) :
  x^n + 1/x^n = 2 * Real.cos (n * θ) := by
  sorry

end NUMINAMATH_CALUDE_x_power_plus_inverse_l2379_237980


namespace NUMINAMATH_CALUDE_stratified_sampling_teachers_l2379_237957

theorem stratified_sampling_teachers (total : ℕ) (sample_size : ℕ) (students_in_sample : ℕ) 
  (h1 : total = 4000)
  (h2 : sample_size = 200)
  (h3 : students_in_sample = 190) :
  (sample_size : ℚ) / total * (sample_size - students_in_sample) = 200 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_teachers_l2379_237957


namespace NUMINAMATH_CALUDE_problem_solution_l2379_237954

theorem problem_solution (a : ℝ) : 3 ∈ ({a, a^2 - 2*a} : Set ℝ) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2379_237954


namespace NUMINAMATH_CALUDE_circle_max_area_l2379_237952

/-- Given a circle with equation x^2 + y^2 + kx + 2y + k^2 = 0, 
    prove that its area is maximized when its center is at (0, -1) -/
theorem circle_max_area (k : ℝ) : 
  let circle_eq (x y : ℝ) := x^2 + y^2 + k*x + 2*y + k^2 = 0
  let center := (0, -1)
  let radius_squared (k : ℝ) := 1 - (3/4) * k^2
  ∀ x y : ℝ, circle_eq x y → 
    radius_squared k ≤ radius_squared 0 ∧ 
    circle_eq (center.1) (center.2) := by
  sorry

end NUMINAMATH_CALUDE_circle_max_area_l2379_237952


namespace NUMINAMATH_CALUDE_multiples_of_15_between_17_and_158_l2379_237953

theorem multiples_of_15_between_17_and_158 : 
  (Finset.filter (λ x => x % 15 = 0) (Finset.range (158 - 17 + 1))).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_17_and_158_l2379_237953


namespace NUMINAMATH_CALUDE_sphere_surface_area_with_inscribed_cube_l2379_237949

theorem sphere_surface_area_with_inscribed_cube (edge_length : ℝ) (radius : ℝ) : 
  edge_length = 2 → 
  radius^2 = 3 →
  4 * π * radius^2 = 12 * π := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_with_inscribed_cube_l2379_237949


namespace NUMINAMATH_CALUDE_solve_for_y_l2379_237986

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2379_237986


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2s_l2379_237943

-- Define the displacement function
def s (t : ℝ) : ℝ := 3 * t^3 - 2 * t^2 + t + 1

-- Define the velocity function as the derivative of the displacement function
def v (t : ℝ) : ℝ := 9 * t^2 - 4 * t + 1

-- Theorem statement
theorem instantaneous_velocity_at_2s : v 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2s_l2379_237943


namespace NUMINAMATH_CALUDE_range_of_t_l2379_237939

theorem range_of_t (a b : ℝ) (h : a^2 + a*b + b^2 = 1) :
  let t := a*b - a^2 - b^2
  ∀ x, (∃ a b : ℝ, a^2 + a*b + b^2 = 1 ∧ t = a*b - a^2 - b^2) → -3 ≤ x ∧ x ≤ -1/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_l2379_237939


namespace NUMINAMATH_CALUDE_quadratic_max_min_ratio_bound_l2379_237969

/-- Given a quadratic function f(x) = ax^2 + bx + c with positive coefficients and real roots,
    the maximum value of min{(b+c)/a, (c+a)/b, (a+b)/c} is 5/4 -/
theorem quadratic_max_min_ratio_bound 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hroots : b^2 ≥ 4*a*c) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    min (min ((y+z)/x) ((z+x)/y)) ((x+y)/z) = 5/4 ∧
    ∀ (p q r : ℝ), p > 0 → q > 0 → r > 0 → 
      min (min ((q+r)/p) ((r+p)/q)) ((p+q)/r) ≤ 5/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_min_ratio_bound_l2379_237969


namespace NUMINAMATH_CALUDE_problem_solution_l2379_237998

theorem problem_solution :
  let x : ℝ := 88 + (4/3) * 88
  let y : ℝ := x + (3/5) * x
  let z : ℝ := (1/2) * (x + y)
  z = 266.9325 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2379_237998


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2379_237924

theorem arithmetic_calculation : 24 * 36 + 18 * 24 - 12 * (36 / 6) = 1224 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2379_237924


namespace NUMINAMATH_CALUDE_michael_record_score_l2379_237964

/-- Given a basketball team's total score and the average score of other players,
    calculate Michael's score that set the new school record. -/
theorem michael_record_score (total_score : ℕ) (other_players : ℕ) (avg_score : ℕ) 
    (h1 : total_score = 75)
    (h2 : other_players = 5)
    (h3 : avg_score = 6) :
    total_score - (other_players * avg_score) = 45 := by
  sorry

#check michael_record_score

end NUMINAMATH_CALUDE_michael_record_score_l2379_237964


namespace NUMINAMATH_CALUDE_transformed_curve_is_circle_l2379_237918

-- Define the initial polar equation
def initial_polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (3 * (Real.cos θ)^2 + 4 * (Real.sin θ)^2)

-- Define the scaling transformation
def scaling_transformation (x y x' y' : ℝ) : Prop :=
  x' = (1/2) * x ∧ y' = (Real.sqrt 3 / 3) * y

-- Theorem statement
theorem transformed_curve_is_circle :
  ∀ (x y x' y' : ℝ),
  (∃ (ρ θ : ℝ), initial_polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  scaling_transformation x y x' y' →
  ∃ (r : ℝ), x'^2 + y'^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_transformed_curve_is_circle_l2379_237918


namespace NUMINAMATH_CALUDE_shaded_area_correct_l2379_237923

/-- The area of a specific region formed by two squares with side lengths 6 and 8 --/
def shaded_area : ℝ := 50.24

/-- The value of π used in the calculation --/
def π : ℝ := 3.14

/-- The side length of the first square --/
def side1 : ℝ := 6

/-- The side length of the second square --/
def side2 : ℝ := 8

/-- Theorem stating that the shaded area is correct given the conditions --/
theorem shaded_area_correct : 
  ∃ (area : ℝ), area = shaded_area ∧ 
  π = 3.14 ∧ 
  side1 = 6 ∧ 
  side2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_correct_l2379_237923


namespace NUMINAMATH_CALUDE_car_travel_time_l2379_237981

theorem car_travel_time (distance : ℝ) (new_speed : ℝ) (time_ratio : ℝ) 
  (h1 : distance = 270)
  (h2 : new_speed = 30)
  (h3 : time_ratio = 3/2)
  : ∃ (initial_time : ℝ), 
    initial_time = 6 ∧ 
    distance = new_speed * (initial_time * time_ratio) :=
by sorry

end NUMINAMATH_CALUDE_car_travel_time_l2379_237981


namespace NUMINAMATH_CALUDE_john_needs_thirteen_l2379_237976

/-- The amount of additional money John needs to buy a pogo stick -/
def additional_money_needed (saturday_earnings sunday_earnings previous_weekend_earnings pogo_stick_cost : ℕ) : ℕ :=
  pogo_stick_cost - (saturday_earnings + sunday_earnings + previous_weekend_earnings)

/-- Theorem stating how much additional money John needs -/
theorem john_needs_thirteen : 
  ∀ (saturday_earnings sunday_earnings previous_weekend_earnings pogo_stick_cost : ℕ),
    saturday_earnings = 18 →
    sunday_earnings = saturday_earnings / 2 →
    previous_weekend_earnings = 20 →
    pogo_stick_cost = 60 →
    additional_money_needed saturday_earnings sunday_earnings previous_weekend_earnings pogo_stick_cost = 13 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_thirteen_l2379_237976


namespace NUMINAMATH_CALUDE_polar_coordinate_transformation_l2379_237956

theorem polar_coordinate_transformation (x y r θ : ℝ) :
  x = 8 ∧ y = 6 ∧ r = Real.sqrt (x^2 + y^2) ∧ 
  x = r * Real.cos θ ∧ y = r * Real.sin θ →
  ∃ (x' y' : ℝ), 
    x' = 2 * Real.sqrt 2 ∧ 
    y' = 14 * Real.sqrt 2 ∧
    x' = (2 * r) * Real.cos (θ + π/4) ∧ 
    y' = (2 * r) * Real.sin (θ + π/4) := by
  sorry

end NUMINAMATH_CALUDE_polar_coordinate_transformation_l2379_237956


namespace NUMINAMATH_CALUDE_rose_difference_l2379_237920

/-- Given the initial number of roses in a vase, the number of roses thrown away,
    and the final number of roses in the vase, calculate the difference between
    the number of roses thrown away and the number of roses cut from the garden. -/
theorem rose_difference (initial : ℕ) (thrown_away : ℕ) (final : ℕ) :
  initial = 21 → thrown_away = 34 → final = 15 →
  thrown_away - final = 19 := by sorry

end NUMINAMATH_CALUDE_rose_difference_l2379_237920


namespace NUMINAMATH_CALUDE_coin_collection_average_l2379_237909

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℕ → ℝ
  | 0 => a₁
  | k + 1 => arithmetic_sequence a₁ d n k + d

theorem coin_collection_average :
  let a₁ : ℝ := 5
  let d : ℝ := 6
  let n : ℕ := 7
  let seq := arithmetic_sequence a₁ d n
  (seq 0 + seq (n - 1)) / 2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_coin_collection_average_l2379_237909


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l2379_237979

theorem consecutive_even_numbers_sum (a b c : ℤ) : 
  (∃ k : ℤ, b = 2 * k) →  -- b is even
  (a = b - 2) →           -- a is the previous even number
  (c = b + 2) →           -- c is the next even number
  (a + b = 18) →          -- sum of first and second
  (a + c = 22) →          -- sum of first and third
  (b + c = 28) →          -- sum of second and third
  b = 11 :=               -- middle number is 11
by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l2379_237979


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l2379_237983

/-- The probability of having a boy or a girl -/
def p_boy_or_girl : ℚ := 1 / 2

/-- The number of children in the family -/
def num_children : ℕ := 4

/-- The probability of having at least one boy and one girl in a family of four children -/
theorem prob_at_least_one_boy_and_girl : 
  (1 : ℚ) - (p_boy_or_girl ^ num_children + p_boy_or_girl ^ num_children) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l2379_237983


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l2379_237904

theorem ratio_sum_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  x / y = 3 / 4 → x + y + 100 = 500 → y = 1600 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l2379_237904


namespace NUMINAMATH_CALUDE_figurine_cost_l2379_237911

def televisions : ℕ := 5
def television_cost : ℕ := 50
def figurines : ℕ := 10
def total_spent : ℕ := 260

theorem figurine_cost :
  (total_spent - televisions * television_cost) / figurines = 1 :=
sorry

end NUMINAMATH_CALUDE_figurine_cost_l2379_237911


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l2379_237945

/-- Theorem: Ratio of inscribed circle area to square area is π/4 -/
theorem inscribed_circle_area_ratio (a b : ℤ) (h : b ≠ 0) :
  let r : ℚ := a / b
  let circle_area := π * r^2
  let square_side := 2 * r
  let square_area := square_side^2
  circle_area / square_area = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l2379_237945


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_l2379_237910

/-- Given six rectangles with width 2 and lengths 1, 4, 9, 16, 25, and 36, 
    prove that the sum of their areas is 182. -/
theorem sum_of_rectangle_areas : 
  let width : ℕ := 2
  let lengths : List ℕ := [1, 4, 9, 16, 25, 36]
  let areas : List ℕ := lengths.map (λ l => l * width)
  areas.sum = 182 := by
sorry

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_l2379_237910


namespace NUMINAMATH_CALUDE_largest_three_digit_square_base7_l2379_237991

/-- The number of digits of a natural number in base 7 -/
def numDigitsBase7 (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log 7 n + 1

/-- Conversion from base 10 to base 7 -/
def toBase7 (n : ℕ) : ℕ :=
  sorry

theorem largest_three_digit_square_base7 :
  ∃ N : ℕ, N = 45 ∧
  (∀ m : ℕ, m > N → numDigitsBase7 (m^2) > 3) ∧
  numDigitsBase7 (N^2) = 3 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_square_base7_l2379_237991


namespace NUMINAMATH_CALUDE_sum_inequality_l2379_237984

theorem sum_inequality (a b c d : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d) :
  a * d + b * c < a * c + b * d ∧ a * c + b * d < a * b + c * d := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l2379_237984


namespace NUMINAMATH_CALUDE_tan_inequality_solution_set_l2379_237917

open Real

theorem tan_inequality_solution_set (x : ℝ) :
  (3 * tan x + Real.sqrt 3 > 0) ↔
  ∃ k : ℤ, x ∈ Set.Ioo ((-(π / 6) : ℝ) + k * π) ((π / 6 : ℝ) + k * π) :=
by sorry

end NUMINAMATH_CALUDE_tan_inequality_solution_set_l2379_237917


namespace NUMINAMATH_CALUDE_personal_planners_count_l2379_237901

/-- The cost of a spiral notebook in dollars -/
def spiral_notebook_cost : ℝ := 15

/-- The cost of a personal planner in dollars -/
def personal_planner_cost : ℝ := 10

/-- The number of spiral notebooks bought -/
def num_spiral_notebooks : ℕ := 4

/-- The discount rate applied to the total cost -/
def discount_rate : ℝ := 0.2

/-- The total discounted cost in dollars -/
def total_discounted_cost : ℝ := 112

/-- The number of personal planners bought -/
def num_personal_planners : ℕ := 8

theorem personal_planners_count :
  ∃ (x : ℕ),
    (1 - discount_rate) * (spiral_notebook_cost * num_spiral_notebooks + personal_planner_cost * x) = total_discounted_cost ∧
    x = num_personal_planners :=
by sorry

end NUMINAMATH_CALUDE_personal_planners_count_l2379_237901


namespace NUMINAMATH_CALUDE_ryan_commute_time_l2379_237997

/-- Ryan's weekly commute time calculation -/
theorem ryan_commute_time : 
  let bike_days : ℕ := 1
  let bus_days : ℕ := 3
  let friend_days : ℕ := 1
  let bike_time : ℕ := 30
  let bus_time : ℕ := bike_time + 10
  let friend_time : ℕ := bike_time / 3
  bike_days * bike_time + bus_days * bus_time + friend_days * friend_time = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_ryan_commute_time_l2379_237997


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l2379_237932

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  populationSize : ℕ
  sampleSize : ℕ
  firstElement : ℕ
  commonDifference : ℕ

/-- Checks if a given list is a valid systematic sample -/
def isValidSample (s : SystematicSample) (sample : List ℕ) : Prop :=
  sample.length = s.sampleSize ∧
  sample.head! = s.firstElement ∧
  ∀ i, 0 < i → i < s.sampleSize → 
    sample[i]! = sample[i-1]! + s.commonDifference ∧
    sample[i]! ≤ s.populationSize

/-- The main theorem to prove -/
theorem systematic_sample_theorem (s : SystematicSample) 
  (h1 : s.populationSize = 60)
  (h2 : s.sampleSize = 5)
  (h3 : 4 ∈ (List.range s.sampleSize).map (fun i => s.firstElement + i * s.commonDifference)) :
  isValidSample s [4, 16, 28, 40, 52] :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l2379_237932


namespace NUMINAMATH_CALUDE_rays_dog_walking_problem_l2379_237906

/-- Ray's dog walking problem -/
theorem rays_dog_walking_problem (x : ℕ) : 
  (∀ (total_blocks : ℕ), total_blocks = 3 * (x + 7 + 11) → total_blocks = 66) → 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_rays_dog_walking_problem_l2379_237906


namespace NUMINAMATH_CALUDE_pencil_box_theorems_l2379_237951

/-- Represents the number of pencils of each color in the box -/
structure PencilBox where
  blue : Nat
  red : Nat
  green : Nat
  yellow : Nat

/-- The initial state of the pencil box -/
def initialBox : PencilBox := {
  blue := 5,
  red := 9,
  green := 6,
  yellow := 4
}

/-- The minimum number of pencils to ensure at least one of each color -/
def minPencilsForAllColors (box : PencilBox) : Nat :=
  box.blue + box.red + box.green + box.yellow - 3

/-- The maximum number of pencils to ensure at least one of each color remains -/
def maxPencilsLeaveAllColors (box : PencilBox) : Nat :=
  min box.blue box.red |> min box.green |> min box.yellow |> (· - 1)

/-- The maximum number of pencils to ensure at least five red pencils remain -/
def maxPencilsLeaveFiveRed (box : PencilBox) : Nat :=
  max (box.red - 5) 0

theorem pencil_box_theorems (box : PencilBox := initialBox) :
  (minPencilsForAllColors box = 21) ∧
  (maxPencilsLeaveAllColors box = 3) ∧
  (maxPencilsLeaveFiveRed box = 4) := by
  sorry

end NUMINAMATH_CALUDE_pencil_box_theorems_l2379_237951


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l2379_237940

open Real

noncomputable def x (t : ℝ) : ℝ := exp (2 * t) * (-2 * cos t + sin t) + 2

noncomputable def y (t : ℝ) : ℝ := exp (2 * t) * (-cos t + 3 * sin t) + 3

theorem solution_satisfies_system :
  (∀ t, deriv x t = x t + y t - 3) ∧
  (∀ t, deriv y t = -2 * x t + 3 * y t + 1) ∧
  x 0 = 0 ∧
  y 0 = 0 :=
sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l2379_237940


namespace NUMINAMATH_CALUDE_rhombus_perimeter_and_area_l2379_237922

/-- A rhombus with given diagonal lengths -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- Calculate the perimeter of a rhombus -/
def perimeter (r : Rhombus) : ℝ := sorry

/-- Calculate the area of a rhombus -/
def area (r : Rhombus) : ℝ := sorry

/-- Theorem about a specific rhombus -/
theorem rhombus_perimeter_and_area :
  let r := Rhombus.mk 10 24
  perimeter r = 52 ∧ area r = 120 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_and_area_l2379_237922


namespace NUMINAMATH_CALUDE_problem_solution_l2379_237944

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (h1 : x/2 = y^2) (h2 : x/4 = 4*y) : x = 128 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2379_237944


namespace NUMINAMATH_CALUDE_keanu_fish_spending_l2379_237996

/-- The number of fish Keanu gave to his dog -/
def dog_fish : ℕ := 40

/-- The number of fish Keanu gave to his cat -/
def cat_fish : ℕ := dog_fish / 2

/-- The cost of each fish in dollars -/
def fish_cost : ℕ := 4

/-- The total number of fish Keanu bought -/
def total_fish : ℕ := dog_fish + cat_fish

/-- The total amount Keanu spent on fish in dollars -/
def total_spent : ℕ := total_fish * fish_cost

theorem keanu_fish_spending :
  total_spent = 240 :=
sorry

end NUMINAMATH_CALUDE_keanu_fish_spending_l2379_237996


namespace NUMINAMATH_CALUDE_smallest_integer_cube_root_l2379_237990

theorem smallest_integer_cube_root (m n : ℕ) (r : ℝ) : 
  (0 < n) →
  (0 < r) →
  (r < 1/100) →
  (m = ((n : ℝ) + r)^3) →
  (∀ k < m, ¬∃ (s : ℝ), 0 < s ∧ s < 1/100 ∧ (k : ℝ)^(1/3) = (k : ℝ) + s) →
  (n = 6) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_cube_root_l2379_237990


namespace NUMINAMATH_CALUDE_die_toss_results_l2379_237972

/-- The number of faces on a fair die -/
def numFaces : ℕ := 6

/-- The number of tosses when the process stops -/
def numTosses : ℕ := 5

/-- The number of different numbers recorded when the process stops -/
def numDifferent : ℕ := 3

/-- The total number of different recording results -/
def totalResults : ℕ := 840

/-- Theorem stating the total number of different recording results -/
theorem die_toss_results :
  (numFaces = 6) →
  (numTosses = 5) →
  (numDifferent = 3) →
  (totalResults = 840) := by
  sorry

end NUMINAMATH_CALUDE_die_toss_results_l2379_237972


namespace NUMINAMATH_CALUDE_calculate_expression_l2379_237992

theorem calculate_expression : -5 + 2 * (-3) + (-12) / (-2) = -5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2379_237992


namespace NUMINAMATH_CALUDE_cristine_lemons_l2379_237927

theorem cristine_lemons (initial_lemons : ℕ) (given_away_fraction : ℚ) (exchanged_fraction : ℚ) : 
  initial_lemons = 12 →
  given_away_fraction = 1/4 →
  exchanged_fraction = 1/3 →
  (initial_lemons - initial_lemons * given_away_fraction) * (1 - exchanged_fraction) = 6 := by
sorry

end NUMINAMATH_CALUDE_cristine_lemons_l2379_237927


namespace NUMINAMATH_CALUDE_gcd_18_30_l2379_237905

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l2379_237905


namespace NUMINAMATH_CALUDE_die_roll_events_l2379_237978

-- Define the sample space for a six-sided die roll
def Ω : Type := Fin 6

-- Define the events A_k
def A (k : Fin 6) : Set Ω := {ω : Ω | ω.val + 1 = k.val + 1}

-- Define event A: rolling an even number of points
def event_A : Set Ω := A 1 ∪ A 3 ∪ A 5

-- Define event B: rolling an odd number of points
def event_B : Set Ω := A 0 ∪ A 2 ∪ A 4

-- Define event C: rolling a multiple of three
def event_C : Set Ω := A 2 ∪ A 5

-- Define event D: rolling a number greater than three
def event_D : Set Ω := A 3 ∪ A 4 ∪ A 5

theorem die_roll_events :
  (event_A = A 1 ∪ A 3 ∪ A 5) ∧
  (event_B = A 0 ∪ A 2 ∪ A 4) ∧
  (event_C = A 2 ∪ A 5) ∧
  (event_D = A 3 ∪ A 4 ∪ A 5) := by sorry

end NUMINAMATH_CALUDE_die_roll_events_l2379_237978


namespace NUMINAMATH_CALUDE_petya_wins_against_sasha_l2379_237916

/-- Represents a player in the knockout tennis tournament -/
inductive Player : Type
| Petya : Player
| Sasha : Player
| Misha : Player

/-- The number of rounds played by each player -/
def rounds_played (p : Player) : ℕ :=
  match p with
  | Player.Petya => 12
  | Player.Sasha => 7
  | Player.Misha => 11

/-- The total number of games played in the tournament -/
def total_games : ℕ := (rounds_played Player.Petya + rounds_played Player.Sasha + rounds_played Player.Misha) / 2

/-- The number of games a player did not play -/
def games_not_played (p : Player) : ℕ := total_games - rounds_played p

/-- Theorem stating that Petya won 4 times against Sasha -/
theorem petya_wins_against_sasha : 
  games_not_played Player.Misha = 4 ∧ 
  (∀ p : Player, games_not_played p + rounds_played p = total_games) ∧
  (rounds_played Player.Sasha = 7 → games_not_played Player.Sasha = 8) :=
sorry

end NUMINAMATH_CALUDE_petya_wins_against_sasha_l2379_237916


namespace NUMINAMATH_CALUDE_total_fish_count_l2379_237942

/-- The number of fish per white duck -/
def fish_per_white_duck : ℕ := 5

/-- The number of fish per black duck -/
def fish_per_black_duck : ℕ := 10

/-- The number of fish per multicolor duck -/
def fish_per_multicolor_duck : ℕ := 12

/-- The number of white ducks -/
def white_ducks : ℕ := 3

/-- The number of black ducks -/
def black_ducks : ℕ := 7

/-- The number of multicolor ducks -/
def multicolor_ducks : ℕ := 6

/-- The total number of fish in the lake -/
def total_fish : ℕ := fish_per_white_duck * white_ducks + 
                      fish_per_black_duck * black_ducks + 
                      fish_per_multicolor_duck * multicolor_ducks

theorem total_fish_count : total_fish = 157 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l2379_237942


namespace NUMINAMATH_CALUDE_larger_number_problem_l2379_237931

theorem larger_number_problem (x y : ℝ) (h_sum : x + y = 30) (h_diff : x - y = 14) :
  max x y = 22 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2379_237931


namespace NUMINAMATH_CALUDE_kate_change_l2379_237903

def candy1_cost : ℚ := 54/100
def candy2_cost : ℚ := 35/100
def candy3_cost : ℚ := 68/100
def amount_paid : ℚ := 5

theorem kate_change : 
  amount_paid - (candy1_cost + candy2_cost + candy3_cost) = 343/100 := by
  sorry

end NUMINAMATH_CALUDE_kate_change_l2379_237903


namespace NUMINAMATH_CALUDE_reading_time_calculation_l2379_237985

theorem reading_time_calculation (pages_per_hour_1 pages_per_hour_2 pages_per_hour_3 : ℕ)
  (total_pages : ℕ) (h1 : pages_per_hour_1 = 21) (h2 : pages_per_hour_2 = 30)
  (h3 : pages_per_hour_3 = 45) (h4 : total_pages = 128) :
  let total_time := (3 * total_pages) / (pages_per_hour_1 + pages_per_hour_2 + pages_per_hour_3)
  total_time = 4 := by
sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l2379_237985


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_7_l2379_237900

theorem smallest_perfect_square_divisible_by_5_and_7 : ∃ n : ℕ,
  n > 0 ∧
  (∃ m : ℕ, n = m ^ 2) ∧
  n % 5 = 0 ∧
  n % 7 = 0 ∧
  n = 1225 ∧
  (∀ k : ℕ, k > 0 → (∃ l : ℕ, k = l ^ 2) → k % 5 = 0 → k % 7 = 0 → k ≥ 1225) :=
by
  sorry

#check smallest_perfect_square_divisible_by_5_and_7

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_7_l2379_237900


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l2379_237959

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The sum of the first n prime numbers -/
def sumFirstNPrimes (n : ℕ) : ℕ := sorry

theorem sum_first_six_primes_mod_seventh_prime :
  sumFirstNPrimes 6 % nthPrime 7 = 7 := by sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l2379_237959


namespace NUMINAMATH_CALUDE_only_A_and_B_excellent_l2379_237915

/-- Represents a student's scores in three components -/
structure StudentScores where
  written : ℝ
  practical : ℝ
  growth : ℝ

/-- Calculates the total evaluation score for a student -/
def totalScore (s : StudentScores) : ℝ :=
  0.5 * s.written + 0.2 * s.practical + 0.3 * s.growth

/-- Determines if a score is excellent (over 90) -/
def isExcellent (score : ℝ) : Prop :=
  score > 90

/-- The scores of student A -/
def studentA : StudentScores :=
  { written := 90, practical := 83, growth := 95 }

/-- The scores of student B -/
def studentB : StudentScores :=
  { written := 98, practical := 90, growth := 95 }

/-- The scores of student C -/
def studentC : StudentScores :=
  { written := 80, practical := 88, growth := 90 }

/-- Theorem stating that only students A and B have excellent scores -/
theorem only_A_and_B_excellent :
  isExcellent (totalScore studentA) ∧
  isExcellent (totalScore studentB) ∧
  ¬isExcellent (totalScore studentC) := by
  sorry


end NUMINAMATH_CALUDE_only_A_and_B_excellent_l2379_237915


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l2379_237913

/-- Theorem: If a fruit seller sells 40% of his apples and has 420 apples remaining, 
    then he originally had 700 apples. -/
theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples : ℚ) * (1 - 0.4) = 420 → initial_apples = 700 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l2379_237913


namespace NUMINAMATH_CALUDE_inverse_functions_l2379_237933

-- Define the type for our functions
def Function : Type := ℝ → ℝ

-- Define the property of having an inverse
def has_inverse (f : Function) : Prop := ∃ g : Function, ∀ x, g (f x) = x ∧ f (g x) = x

-- Define our functions based on their graphical properties
def F : Function := sorry
def G : Function := sorry
def H : Function := sorry
def I : Function := sorry

-- State the theorem
theorem inverse_functions :
  (has_inverse F) ∧ (has_inverse H) ∧ (has_inverse I) ∧ ¬(has_inverse G) := by sorry

end NUMINAMATH_CALUDE_inverse_functions_l2379_237933
