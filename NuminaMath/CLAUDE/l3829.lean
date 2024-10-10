import Mathlib

namespace race_time_patrick_l3829_382995

theorem race_time_patrick (patrick_time manu_time amy_time : ℕ) : 
  manu_time = patrick_time + 12 →
  amy_time * 2 = manu_time →
  amy_time = 36 →
  patrick_time = 60 := by
sorry

end race_time_patrick_l3829_382995


namespace negative_200_means_send_out_l3829_382914

/-- Represents a WeChat payment transaction -/
structure WeChatTransaction where
  amount : ℝ
  balance_before : ℝ
  balance_after : ℝ

/-- Axiom: Receiving money increases the balance -/
axiom receive_increases_balance {t : WeChatTransaction} (h : t.amount > 0) : 
  t.balance_after = t.balance_before + t.amount

/-- Axiom: Sending money decreases the balance -/
axiom send_decreases_balance {t : WeChatTransaction} (h : t.amount < 0) :
  t.balance_after = t.balance_before + t.amount

/-- The meaning of a -200 transaction in WeChat payments -/
theorem negative_200_means_send_out (t : WeChatTransaction) 
  (h1 : t.amount = -200)
  (h2 : t.balance_before = 867.35)
  (h3 : t.balance_after = 667.35) :
  "Sending out 200 yuan" = "The meaning of -200 in WeChat payments" := by
  sorry

end negative_200_means_send_out_l3829_382914


namespace black_cube_difference_l3829_382963

/-- Represents a 3x3x3 cube built with unit cubes -/
structure Cube :=
  (size : Nat)
  (total_cubes : Nat)
  (surface_area : Nat)

/-- Represents the distribution of colors on the cube's surface -/
structure SurfaceColor :=
  (black : Nat)
  (grey : Nat)
  (white : Nat)

/-- Defines a valid 3x3x3 cube with equal surface color distribution -/
def valid_cube (c : Cube) (sc : SurfaceColor) : Prop :=
  c.size = 3 ∧
  c.total_cubes = 27 ∧
  c.surface_area = 54 ∧
  sc.black = sc.grey ∧
  sc.grey = sc.white ∧
  sc.black + sc.grey + sc.white = c.surface_area

/-- The minimum number of black cubes that can be used -/
def min_black_cubes (c : Cube) (sc : SurfaceColor) : Nat :=
  sorry

/-- The maximum number of black cubes that can be used -/
def max_black_cubes (c : Cube) (sc : SurfaceColor) : Nat :=
  sorry

/-- Theorem stating the difference between max and min black cubes -/
theorem black_cube_difference (c : Cube) (sc : SurfaceColor) :
  valid_cube c sc → max_black_cubes c sc - min_black_cubes c sc = 7 :=
  sorry

end black_cube_difference_l3829_382963


namespace area_ratio_concentric_circles_l3829_382989

/-- Given two concentric circles where a 60-degree arc on the smaller circle
    has the same length as a 30-degree arc on the larger circle,
    the ratio of the area of the smaller circle to the area of the larger circle is 1/4. -/
theorem area_ratio_concentric_circles (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 * (2 * Real.pi * r₁) = 30 / 360 * (2 * Real.pi * r₂)) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 1 / 4 := by
  sorry

end area_ratio_concentric_circles_l3829_382989


namespace max_value_squared_l3829_382924

theorem max_value_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y)
  (h : x^3 + 2013*y = y^3 + 2013*x) :
  ∃ (M : ℝ), M = (Real.sqrt 3 + 1) * x + 2 * y ∧
    ∀ (N : ℝ), N = (Real.sqrt 3 + 1) * x + 2 * y → N^2 ≤ 16104 :=
by sorry

end max_value_squared_l3829_382924


namespace quadratic_inequality_range_l3829_382923

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * x + a ≥ 0) ↔ a ≥ 1 := by
  sorry

end quadratic_inequality_range_l3829_382923


namespace ceiling_floor_product_range_l3829_382988

theorem ceiling_floor_product_range (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 132 → y ∈ Set.Ioo (-12) (-11) := by
  sorry

end ceiling_floor_product_range_l3829_382988


namespace box_volume_problem_l3829_382913

theorem box_volume_problem :
  ∃! (x : ℕ+), (2 * x.val - 5 > 0) ∧
  ((x.val^2 + 5) * (2 * x.val - 5) * (x.val + 25) < 1200) := by
  sorry

end box_volume_problem_l3829_382913


namespace complex_product_theorem_l3829_382987

theorem complex_product_theorem (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2) 
  (h2 : Complex.abs z₂ = 3) 
  (h3 : 3 * z₁ - 2 * z₂ = (3/2 : ℂ) - Complex.I) : 
  z₁ * z₂ = -30/13 + 72/13 * Complex.I := by sorry

end complex_product_theorem_l3829_382987


namespace kylie_coins_left_l3829_382921

/-- The number of coins Kylie has after all transactions -/
def coins_left (piggy_bank : ℕ) (from_brother : ℕ) (from_father : ℕ) (given_to_friend : ℕ) : ℕ :=
  piggy_bank + from_brother + from_father - given_to_friend

/-- Theorem stating that Kylie is left with 15 coins -/
theorem kylie_coins_left : 
  coins_left 15 13 8 21 = 15 := by sorry

end kylie_coins_left_l3829_382921


namespace joe_spending_l3829_382955

def entrance_fee_under_18 : ℝ := 5
def entrance_fee_over_18 : ℝ := entrance_fee_under_18 * 1.2
def group_discount_rate : ℝ := 0.15
def ride_cost : ℝ := 0.5
def joe_age : ℕ := 30
def twin_age : ℕ := 6
def joe_rides : ℕ := 4
def twin_a_rides : ℕ := 3
def twin_b_rides : ℕ := 5

def group_size : ℕ := 3

def total_entrance_fee : ℝ := 
  entrance_fee_over_18 + 2 * entrance_fee_under_18

def discounted_entrance_fee : ℝ := 
  total_entrance_fee * (1 - group_discount_rate)

def total_ride_cost : ℝ := 
  ride_cost * (joe_rides + twin_a_rides + twin_b_rides)

theorem joe_spending (joe_spending : ℝ) : 
  joe_spending = discounted_entrance_fee + total_ride_cost ∧ 
  joe_spending = 19.60 := by sorry

end joe_spending_l3829_382955


namespace professor_seating_arrangements_l3829_382922

/-- Represents the number of chairs in a row -/
def total_chairs : ℕ := 14

/-- Represents the number of professors -/
def num_professors : ℕ := 4

/-- Represents the number of students -/
def num_students : ℕ := 10

/-- Represents the number of possible positions for professors (excluding first and last chair) -/
def professor_positions : ℕ := total_chairs - 2

/-- Theorem stating the number of ways professors can choose their chairs -/
theorem professor_seating_arrangements :
  (∃ (two_adjacent : ℕ) (three_adjacent : ℕ) (four_adjacent : ℕ),
    two_adjacent = (professor_positions - 1) * (Nat.choose (professor_positions - 2) 2) * (Nat.factorial num_professors / 2) ∧
    three_adjacent = (professor_positions - 2) * (professor_positions - 3) * (Nat.factorial num_professors) ∧
    four_adjacent = (professor_positions - 3) * (Nat.factorial num_professors) ∧
    two_adjacent + three_adjacent + four_adjacent = 5346) :=
by sorry

end professor_seating_arrangements_l3829_382922


namespace garage_sale_earnings_l3829_382959

/-- The total earnings from selling necklaces at a garage sale -/
def total_earnings (bead_count gemstone_count crystal_count wooden_count : ℕ)
                   (bead_price gemstone_price crystal_price wooden_price : ℕ) : ℕ :=
  bead_count * bead_price + 
  gemstone_count * gemstone_price + 
  crystal_count * crystal_price + 
  wooden_count * wooden_price

/-- Theorem stating that the total earnings from selling the specified necklaces is $53 -/
theorem garage_sale_earnings : 
  total_earnings 4 3 2 5 3 7 5 2 = 53 := by
  sorry

end garage_sale_earnings_l3829_382959


namespace even_function_implies_m_eq_two_l3829_382990

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The given function f(x) -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 2) * x + (m^2 - 7 * m + 12)

theorem even_function_implies_m_eq_two :
  ∀ m : ℝ, IsEven (f m) → m = 2 := by
  sorry

end even_function_implies_m_eq_two_l3829_382990


namespace equation_always_has_solution_l3829_382956

theorem equation_always_has_solution (a b : ℝ) (ha : a ≠ 0) 
  (h_at_most_one : ∃! x, a * x^2 - b * x - a + 3 = 0) :
  ∃ x, (b - 3) * x^2 + (a - 2 * b) * x + 3 * a + 3 = 0 := by
  sorry

end equation_always_has_solution_l3829_382956


namespace parabola_focus_coordinates_l3829_382906

/-- The focus of the parabola (y-1)^2 = 4(x-1) has coordinates (0, 1) -/
theorem parabola_focus_coordinates (x y : ℝ) : 
  ((y - 1)^2 = 4*(x - 1)) → (x = 0 ∧ y = 1) := by
  sorry

end parabola_focus_coordinates_l3829_382906


namespace orange_profit_calculation_l3829_382912

/-- Calculates the profit from an orange selling operation -/
def orange_profit (buy_quantity : ℕ) (buy_price : ℚ) (sell_quantity : ℕ) (sell_price : ℚ) 
                  (transport_cost : ℚ) (storage_fee : ℚ) : ℚ :=
  let total_cost := buy_price + 2 * transport_cost + storage_fee
  let revenue := sell_price
  revenue - total_cost

/-- The profit from the orange selling operation is -4r -/
theorem orange_profit_calculation : 
  orange_profit 11 10 10 11 2 1 = -4 := by
  sorry

end orange_profit_calculation_l3829_382912


namespace less_crowded_detector_time_is_ten_l3829_382958

/-- Represents the time Mark spends on courthouse activities in a week -/
structure CourthouseTime where
  workDays : ℕ
  parkingTime : ℕ
  walkingTime : ℕ
  crowdedDetectorDays : ℕ
  crowdedDetectorTime : ℕ
  totalWeeklyTime : ℕ

/-- Calculates the time it takes to get through the metal detector on less crowded days -/
def lessCrowdedDetectorTime (ct : CourthouseTime) : ℕ :=
  let weeklyParkingTime := ct.workDays * ct.parkingTime
  let weeklyWalkingTime := ct.workDays * ct.walkingTime
  let weeklyCrowdedDetectorTime := ct.crowdedDetectorDays * ct.crowdedDetectorTime
  let remainingTime := ct.totalWeeklyTime - weeklyParkingTime - weeklyWalkingTime - weeklyCrowdedDetectorTime
  remainingTime / (ct.workDays - ct.crowdedDetectorDays)

theorem less_crowded_detector_time_is_ten (ct : CourthouseTime)
  (h1 : ct.workDays = 5)
  (h2 : ct.parkingTime = 5)
  (h3 : ct.walkingTime = 3)
  (h4 : ct.crowdedDetectorDays = 2)
  (h5 : ct.crowdedDetectorTime = 30)
  (h6 : ct.totalWeeklyTime = 130) :
  lessCrowdedDetectorTime ct = 10 := by
  sorry

#eval lessCrowdedDetectorTime ⟨5, 5, 3, 2, 30, 130⟩

end less_crowded_detector_time_is_ten_l3829_382958


namespace potatoes_already_cooked_l3829_382936

theorem potatoes_already_cooked 
  (total_potatoes : ℕ) 
  (cooking_time_per_potato : ℕ) 
  (remaining_cooking_time : ℕ) 
  (h1 : total_potatoes = 16)
  (h2 : cooking_time_per_potato = 5)
  (h3 : remaining_cooking_time = 45) :
  total_potatoes - (remaining_cooking_time / cooking_time_per_potato) = 7 :=
by sorry

end potatoes_already_cooked_l3829_382936


namespace least_time_six_horses_at_start_l3829_382939

def horse_lap_time (k : ℕ) : ℕ := 2 * k - 1

def is_at_start (t : ℕ) (k : ℕ) : Prop :=
  t % (horse_lap_time k) = 0

def at_least_six_at_start (t : ℕ) : Prop :=
  ∃ (s : Finset ℕ), s.card ≥ 6 ∧ s ⊆ Finset.range 12 ∧ ∀ k ∈ s, is_at_start t (k + 1)

theorem least_time_six_horses_at_start :
  ∃! t : ℕ, t > 0 ∧ at_least_six_at_start t ∧ ∀ s, s > 0 ∧ s < t → ¬(at_least_six_at_start s) :=
by sorry

end least_time_six_horses_at_start_l3829_382939


namespace preimage_of_4_3_l3829_382905

/-- The mapping f from ℝ² to ℝ² defined by f(x, y) = (x + 2y, 2x - y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2 * p.2, 2 * p.1 - p.2)

/-- Theorem stating that (2, 1) is the pre-image of (4, 3) under the mapping f -/
theorem preimage_of_4_3 :
  f (2, 1) = (4, 3) ∧ ∀ p : ℝ × ℝ, f p = (4, 3) → p = (2, 1) :=
by sorry

end preimage_of_4_3_l3829_382905


namespace arithmetic_sequence_x_value_l3829_382900

/-- Given an arithmetic sequence with first three terms 2x-3, 3x, and 5x+1, prove that x = 2 -/
theorem arithmetic_sequence_x_value :
  ∀ x : ℝ,
  let a₁ := 2*x - 3
  let a₂ := 3*x
  let a₃ := 5*x + 1
  (a₂ - a₁ = a₃ - a₂) → x = 2 :=
by
  sorry

end arithmetic_sequence_x_value_l3829_382900


namespace expression_evaluation_l3829_382974

theorem expression_evaluation : 
  60 + (105 / 15) + (25 * 16) - 250 + (324 / 9)^2 = 1513 := by sorry

end expression_evaluation_l3829_382974


namespace range_of_even_quadratic_function_l3829_382972

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Define the property of being an even function
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem range_of_even_quadratic_function (a b : ℝ) :
  (∃ x, f a b x = 1 + a) →  -- Lower bound of the domain
  (∃ x, f a b x = 2) →      -- Upper bound of the domain
  is_even (f a b) →         -- f is an even function
  (∀ x, f a b x ∈ Set.Icc (-10) 2) ∧ 
  (∃ x, f a b x = -10) ∧ 
  (∃ x, f a b x = 2) :=
by sorry


end range_of_even_quadratic_function_l3829_382972


namespace min_circle_area_l3829_382973

theorem min_circle_area (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (3 / (2 + x)) + (3 / (2 + y)) = 1) :
  xy ≥ 16 ∧ (xy = 16 ↔ x = 4 ∧ y = 4) :=
by sorry

end min_circle_area_l3829_382973


namespace geometric_sequence_property_l3829_382971

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 + a 5 = Real.pi →
  a 4 * (a 2 + 2 * a 4 + a 6) = Real.pi^2 := by
  sorry

end geometric_sequence_property_l3829_382971


namespace equation_A_is_quadratic_l3829_382915

/-- Definition of a quadratic equation in terms of x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² = -1 -/
def equation_A (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: The equation x² = -1 is a quadratic equation -/
theorem equation_A_is_quadratic : is_quadratic_equation equation_A := by
  sorry


end equation_A_is_quadratic_l3829_382915


namespace pasta_preference_ratio_l3829_382975

theorem pasta_preference_ratio (total_students : ℕ) (ravioli_preference : ℕ) (tortellini_preference : ℕ)
  (h1 : total_students = 800)
  (h2 : ravioli_preference = 300)
  (h3 : tortellini_preference = 150) :
  (ravioli_preference : ℚ) / tortellini_preference = 2 := by
  sorry

end pasta_preference_ratio_l3829_382975


namespace smallest_multiples_sum_l3829_382908

/-- The smallest positive two-digit multiple of 5 -/
def c : ℕ := 10

/-- The smallest positive three-digit multiple of 6 -/
def d : ℕ := 102

theorem smallest_multiples_sum : c + d = 112 := by
  sorry

end smallest_multiples_sum_l3829_382908


namespace sqrt_054_in_terms_of_sqrt_2_and_sqrt_3_l3829_382970

theorem sqrt_054_in_terms_of_sqrt_2_and_sqrt_3 (a b : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = Real.sqrt 3) : 
  Real.sqrt 0.54 = 0.3 * a * b := by
  sorry

end sqrt_054_in_terms_of_sqrt_2_and_sqrt_3_l3829_382970


namespace inverse_proportion_problem_l3829_382966

theorem inverse_proportion_problem (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : a + b = 60) (h3 : a = 3 * b) :
  let b' := k / (-12)
  b' = -225/4 :=
by sorry

end inverse_proportion_problem_l3829_382966


namespace train_speed_l3829_382903

/-- Calculates the speed of a train given its length and time to cross an electric pole. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 800) (h2 : time = 20) :
  length / time = 40 := by
  sorry

end train_speed_l3829_382903


namespace midpoint_property_l3829_382941

/-- Given two points D and E, if F is their midpoint, then 3x - 5y = 9 --/
theorem midpoint_property (D E F : ℝ × ℝ) : 
  D = (30, 10) → 
  E = (6, 8) → 
  F.1 = (D.1 + E.1) / 2 → 
  F.2 = (D.2 + E.2) / 2 → 
  3 * F.1 - 5 * F.2 = 9 := by
sorry

end midpoint_property_l3829_382941


namespace encyclopedia_sorting_l3829_382997

/-- Represents the number of volumes in the encyclopedia --/
def n : ℕ := 30

/-- Represents an operation of swapping two adjacent volumes --/
def swap : ℕ → ℕ → List ℕ → List ℕ := sorry

/-- Checks if a list of volumes is in the correct order --/
def is_sorted : List ℕ → Prop := sorry

/-- The maximum number of disorders in any arrangement of n volumes --/
def max_disorders (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The minimum number of operations required to sort n volumes --/
def min_operations (n : ℕ) : ℕ := max_disorders n

theorem encyclopedia_sorting (arrangement : List ℕ) 
  (h : arrangement.length = n) :
  ∃ (sequence : List (ℕ × ℕ)), 
    sequence.length ≤ min_operations n ∧ 
    is_sorted (sequence.foldl (λ acc (i, j) => swap i j acc) arrangement) := by
  sorry

#eval min_operations n  -- Should evaluate to 435

end encyclopedia_sorting_l3829_382997


namespace volleyball_lineup_count_l3829_382934

def total_players : ℕ := 16
def lineup_size : ℕ := 7
def num_twins : ℕ := 2

theorem volleyball_lineup_count : 
  (Nat.choose total_players lineup_size) - 
  (Nat.choose (total_players - num_twins) lineup_size) = 8008 := by
  sorry

end volleyball_lineup_count_l3829_382934


namespace complex_to_exponential_l3829_382935

theorem complex_to_exponential : 
  let z : ℂ := 1 + Complex.I * Real.sqrt 3
  ∃ (r : ℝ) (θ : ℝ), z = r * Complex.exp (Complex.I * θ) ∧ r = 2 ∧ θ = π / 3 := by
  sorry

end complex_to_exponential_l3829_382935


namespace fuel_mixture_problem_l3829_382948

/-- Proves that 66 gallons of fuel A were added to a 204-gallon tank -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 204 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  ∃ (fuel_a : ℝ), 
    fuel_a = 66 ∧ 
    ethanol_a * fuel_a + ethanol_b * (tank_capacity - fuel_a) = total_ethanol :=
by sorry

end fuel_mixture_problem_l3829_382948


namespace unique_solution_for_system_l3829_382968

theorem unique_solution_for_system :
  ∀ (x y z : ℝ),
  (x^2 + y^2 + z^2 = 2) →
  (x - z = 2) →
  (x = 1 ∧ y = 0 ∧ z = -1) :=
by sorry

end unique_solution_for_system_l3829_382968


namespace lizzy_money_calculation_l3829_382954

/-- Calculates Lizzy's final amount after lending money and receiving it back with interest -/
def lizzys_final_amount (initial_amount loan_amount interest_rate : ℚ) : ℚ :=
  initial_amount - loan_amount + loan_amount * (1 + interest_rate)

/-- Theorem stating that Lizzy will have $33 after lending $15 from her initial $30 and receiving it back with 20% interest -/
theorem lizzy_money_calculation :
  lizzys_final_amount 30 15 (1/5) = 33 := by
  sorry

end lizzy_money_calculation_l3829_382954


namespace cookies_per_bag_l3829_382916

/-- Given 26 bags with an equal number of cookies and 52 cookies in total,
    prove that each bag contains 2 cookies. -/
theorem cookies_per_bag :
  ∀ (bags : ℕ) (total_cookies : ℕ) (cookies_per_bag : ℕ),
    bags = 26 →
    total_cookies = 52 →
    total_cookies = bags * cookies_per_bag →
    cookies_per_bag = 2 := by
  sorry

end cookies_per_bag_l3829_382916


namespace pencil_distribution_l3829_382952

def colored_pencils : ℕ := 14
def black_pencils : ℕ := 35
def siblings : ℕ := 3
def kept_pencils : ℕ := 10

theorem pencil_distribution :
  (colored_pencils + black_pencils - kept_pencils) / siblings = 13 :=
by sorry

end pencil_distribution_l3829_382952


namespace min_sum_of_roots_l3829_382902

theorem min_sum_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + a*x + 3*b = 0) 
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + a = 0) : 
  a + b ≥ 48/27 + 9/4 * (9216/6561)^(1/3) := by
  sorry

end min_sum_of_roots_l3829_382902


namespace smallest_multiple_thirty_six_satisfies_thirty_six_is_smallest_l3829_382909

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 450 * x % 648 = 0 → x ≥ 36 := by
  sorry

theorem thirty_six_satisfies : 450 * 36 % 648 = 0 := by
  sorry

theorem thirty_six_is_smallest : ∃ (x : ℕ), x > 0 ∧ 450 * x % 648 = 0 ∧ x = 36 := by
  sorry

end smallest_multiple_thirty_six_satisfies_thirty_six_is_smallest_l3829_382909


namespace inverse_inequality_l3829_382986

theorem inverse_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end inverse_inequality_l3829_382986


namespace barycentric_vector_relation_l3829_382925

/-- For a triangle ABC and a point X with barycentric coordinates (α:β:γ) where α + β + γ = 1,
    the vector →XA is equal to β→BA + γ→CA. -/
theorem barycentric_vector_relation (A B C X : EuclideanSpace ℝ (Fin 3))
  (α β γ : ℝ) (h_barycentric : α + β + γ = 1)
  (h_X : X = α • A + β • B + γ • C) :
  X - A = β • (B - A) + γ • (C - A) := by
  sorry

end barycentric_vector_relation_l3829_382925


namespace no_factors_l3829_382940

/-- The main polynomial -/
def f (x : ℝ) : ℝ := x^4 + 3*x^2 + 8

/-- Potential factors -/
def g₁ (x : ℝ) : ℝ := x^2 + 4
def g₂ (x : ℝ) : ℝ := x + 2
def g₃ (x : ℝ) : ℝ := x^2 - 4
def g₄ (x : ℝ) : ℝ := x^2 - x - 2

theorem no_factors : 
  (¬ ∃ (h : ℝ → ℝ), f = g₁ * h) ∧
  (¬ ∃ (h : ℝ → ℝ), f = g₂ * h) ∧
  (¬ ∃ (h : ℝ → ℝ), f = g₃ * h) ∧
  (¬ ∃ (h : ℝ → ℝ), f = g₄ * h) :=
by sorry

end no_factors_l3829_382940


namespace horse_distribution_l3829_382957

theorem horse_distribution (total_horses : ℕ) (son1_horses son2_horses son3_horses : ℕ) :
  total_horses = 17 ∧
  son1_horses = 9 ∧
  son2_horses = 6 ∧
  son3_horses = 2 →
  son1_horses / total_horses = 1/2 ∧
  son2_horses / total_horses = 1/3 ∧
  son3_horses / total_horses = 1/9 ∧
  son1_horses + son2_horses + son3_horses = total_horses :=
by
  sorry

#check horse_distribution

end horse_distribution_l3829_382957


namespace probability_qualified_bulb_factory_A_l3829_382942

/-- The probability of buying a qualified light bulb produced by Factory A from the market -/
theorem probability_qualified_bulb_factory_A 
  (factory_A_production_rate : ℝ) 
  (factory_A_pass_rate : ℝ) 
  (h1 : factory_A_production_rate = 0.7)
  (h2 : factory_A_pass_rate = 0.95) : 
  factory_A_production_rate * factory_A_pass_rate = 0.665 := by
sorry

end probability_qualified_bulb_factory_A_l3829_382942


namespace expected_heads_alice_given_more_than_bob_l3829_382907

/-- The number of coins each person flips -/
def n : ℕ := 20

/-- The expected number of heads Alice flipped given she flipped at least as many heads as Bob -/
noncomputable def expected_heads : ℝ :=
  n * (2^(2*n - 2) + Nat.choose (2*n - 1) (n - 1)) / (2^(2*n - 1) + Nat.choose (2*n - 1) (n - 1))

/-- Theorem stating the expected number of heads Alice flipped -/
theorem expected_heads_alice_given_more_than_bob :
  expected_heads = n * (2^(2*n - 2) + Nat.choose (2*n - 1) (n - 1)) / (2^(2*n - 1) + Nat.choose (2*n - 1) (n - 1)) :=
by sorry

end expected_heads_alice_given_more_than_bob_l3829_382907


namespace amelias_dinner_leftover_l3829_382946

/-- Calculates the amount of money Amelia has left after her dinner --/
def ameliasDinner (initialAmount : ℝ) (firstCourseCost : ℝ) (secondCourseExtra : ℝ) 
  (dessertPercent : ℝ) (drinkPercent : ℝ) (tipPercent : ℝ) : ℝ :=
  let secondCourseCost := firstCourseCost + secondCourseExtra
  let dessertCost := dessertPercent * secondCourseCost
  let firstThreeCoursesTotal := firstCourseCost + secondCourseCost + dessertCost
  let drinkCost := drinkPercent * firstThreeCoursesTotal
  let billBeforeTip := firstThreeCoursesTotal + drinkCost
  let tipAmount := tipPercent * billBeforeTip
  let totalBill := billBeforeTip + tipAmount
  initialAmount - totalBill

/-- Theorem stating that Amelia will have $4.80 left after her dinner --/
theorem amelias_dinner_leftover :
  ameliasDinner 60 15 5 0.25 0.20 0.15 = 4.80 := by
  sorry

#eval ameliasDinner 60 15 5 0.25 0.20 0.15

end amelias_dinner_leftover_l3829_382946


namespace largest_number_l3829_382920

-- Define a function to convert a number from base b to decimal
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

-- Define the numbers in their respective bases
def num_A : Nat := to_decimal [2, 1, 1] 6
def num_B : Nat := 41
def num_C : Nat := to_decimal [6, 4] 9
def num_D : Nat := to_decimal [11, 2] 16

-- State the theorem
theorem largest_number :
  num_A > num_B ∧ num_A > num_C ∧ num_A > num_D :=
sorry

end largest_number_l3829_382920


namespace unknown_number_proof_l3829_382978

theorem unknown_number_proof (x : ℝ) : 
  (14 + 32 + 53) / 3 = (21 + 47 + x) / 3 + 3 → x = 22 := by
  sorry

end unknown_number_proof_l3829_382978


namespace square_root_sum_l3829_382928

theorem square_root_sum (x : ℝ) : 
  (Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) → 
  (Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) + Real.sqrt (16 - x^2) = 9) := by
  sorry

end square_root_sum_l3829_382928


namespace factorization_x_squared_minus_one_l3829_382994

theorem factorization_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorization_x_squared_minus_one_l3829_382994


namespace least_positive_integer_with_remainders_l3829_382917

theorem least_positive_integer_with_remainders : ∃ (M : ℕ), 
  (M > 0) ∧ 
  (M % 6 = 5) ∧ 
  (M % 7 = 6) ∧ 
  (M % 9 = 8) ∧ 
  (M % 10 = 9) ∧ 
  (M % 11 = 10) ∧ 
  (∀ (N : ℕ), 
    (N > 0) ∧ 
    (N % 6 = 5) ∧ 
    (N % 7 = 6) ∧ 
    (N % 9 = 8) ∧ 
    (N % 10 = 9) ∧ 
    (N % 11 = 10) → 
    M ≤ N) ∧
  M = 6929 :=
by sorry

end least_positive_integer_with_remainders_l3829_382917


namespace like_terms_imply_abs_diff_l3829_382981

/-- 
If -5x^3y^(n-2) and 3x^(2m+5)y are like terms, then |n-5m| = 8.
-/
theorem like_terms_imply_abs_diff (n m : ℤ) : 
  (2 * m + 5 = 3 ∧ n - 2 = 1) → |n - 5 * m| = 8 := by
  sorry

end like_terms_imply_abs_diff_l3829_382981


namespace cone_volume_increase_l3829_382962

/-- Theorem: Volume increase of a cone with height increase of 160% and radius increase of k% -/
theorem cone_volume_increase (h r k : ℝ) (h_pos : h > 0) (r_pos : r > 0) (k_nonneg : k ≥ 0) :
  let new_height := 2.60 * h
  let new_radius := r * (1 + k / 100)
  let volume_ratio := (new_radius^2 * new_height) / (r^2 * h)
  let percentage_increase := (volume_ratio - 1) * 100
  percentage_increase = ((1 + k / 100)^2 * 2.60 - 1) * 100 :=
by sorry

end cone_volume_increase_l3829_382962


namespace tom_siblings_count_l3829_382943

/-- The number of siblings Tom invited -/
def num_siblings : ℕ :=
  let total_plates : ℕ := 144
  let days : ℕ := 4
  let meals_per_day : ℕ := 3
  let plates_per_meal : ℕ := 2
  let tom_and_parents : ℕ := 3
  let plates_per_person : ℕ := days * meals_per_day * plates_per_meal
  let total_people : ℕ := total_plates / plates_per_person
  total_people - tom_and_parents

theorem tom_siblings_count : num_siblings = 3 := by
  sorry

end tom_siblings_count_l3829_382943


namespace triple_equation_solution_l3829_382979

theorem triple_equation_solution :
  ∀ a b c : ℝ,
    a + b + c = 14 ∧
    a^2 + b^2 + c^2 = 84 ∧
    a^3 + b^3 + c^3 = 584 →
    ((a = 4 ∧ b = 2 ∧ c = 8) ∨
     (a = 2 ∧ b = 4 ∧ c = 8) ∨
     (a = 8 ∧ b = 2 ∧ c = 4)) :=
by
  sorry

end triple_equation_solution_l3829_382979


namespace quadratic_even_function_coeff_l3829_382985

/-- A quadratic function f(x) = ax^2 + (2a^2 - a)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2 * a^2 - a) * x + 1

/-- Definition of an even function -/
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem quadratic_even_function_coeff (a : ℝ) :
  is_even_function (f a) → a = 1/2 := by sorry

end quadratic_even_function_coeff_l3829_382985


namespace remainder_of_quotient_l3829_382961

theorem remainder_of_quotient (q₁ q₂ : ℝ → ℝ) (r₁ r₂ : ℝ) :
  (∃ k₁ : ℝ → ℝ, ∀ x, x^9 = (x - 1/3) * q₁ x + r₁) →
  (∃ k₂ : ℝ → ℝ, ∀ x, q₁ x = (x - 1/3) * q₂ x + r₂) →
  r₂ = 1/6561 := by
  sorry

end remainder_of_quotient_l3829_382961


namespace stating_simultaneous_ring_theorem_l3829_382938

/-- The time interval (in minutes) between bell rings for the post office -/
def post_office_interval : ℕ := 18

/-- The time interval (in minutes) between bell rings for the train station -/
def train_station_interval : ℕ := 24

/-- The time interval (in minutes) between bell rings for the town hall -/
def town_hall_interval : ℕ := 30

/-- The time (in minutes) after which all bells ring simultaneously again -/
def simultaneous_ring_time : ℕ := 360

/-- 
Theorem stating that the time after which all bells ring simultaneously
is the least common multiple of their individual intervals
-/
theorem simultaneous_ring_theorem :
  simultaneous_ring_time = Nat.lcm post_office_interval (Nat.lcm train_station_interval town_hall_interval) :=
by sorry

end stating_simultaneous_ring_theorem_l3829_382938


namespace puppy_food_consumption_l3829_382929

def feeding_schedule (days : ℕ) (portions_per_day : ℕ) (portion_size : ℚ) : ℚ :=
  (days : ℚ) * (portions_per_day : ℚ) * portion_size

theorem puppy_food_consumption : 
  let first_two_weeks := feeding_schedule 14 3 (1/4)
  let second_two_weeks := feeding_schedule 14 2 (1/2)
  let today := (1/2 : ℚ)
  first_two_weeks + second_two_weeks + today = 25
  := by sorry

end puppy_food_consumption_l3829_382929


namespace roots_opposite_signs_l3829_382996

theorem roots_opposite_signs (a b c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2 * b * x + c = 0 ∧ a * y^2 + 2 * b * y + c = 0) →
  (∀ z : ℝ, a^2 * z^2 + 2 * b^2 * z + c^2 ≠ 0) →
  a * c < 0 := by
sorry


end roots_opposite_signs_l3829_382996


namespace train_speed_problem_l3829_382949

/-- Given two trains A and B with lengths 225 m and 150 m respectively,
    if it takes 15 seconds for train A to completely cross train B,
    then the speed of train A is 90 km/hr. -/
theorem train_speed_problem (length_A length_B time_to_cross : ℝ) :
  length_A = 225 →
  length_B = 150 →
  time_to_cross = 15 →
  (length_A + length_B) / time_to_cross * 3.6 = 90 := by
  sorry

end train_speed_problem_l3829_382949


namespace coin_value_equality_l3829_382918

theorem coin_value_equality (n : ℕ) : 
  (15 * 25 + 20 * 10 = 5 * 25 + n * 10) → n = 45 := by
  sorry

end coin_value_equality_l3829_382918


namespace max_non_functional_segments_is_13_l3829_382992

/-- Represents a seven-segment display --/
structure SevenSegmentDisplay :=
  (segments : Fin 7 → Bool)

/-- Represents a four-digit clock display --/
structure ClockDisplay :=
  (digits : Fin 4 → SevenSegmentDisplay)

/-- The set of valid digits for each position --/
def validDigits : Fin 4 → Set ℕ
  | 0 => {0, 1, 2}
  | 1 => {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  | 2 => {0, 1, 2, 3, 4, 5}
  | 3 => {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- A function that determines if a time can be unambiguously read --/
def isUnambiguous (display : ClockDisplay) : Prop := sorry

/-- The maximum number of non-functional segments --/
def maxNonFunctionalSegments : ℕ := 13

/-- The main theorem --/
theorem max_non_functional_segments_is_13 :
  ∀ (display : ClockDisplay),
    (∀ (i : Fin 4), ∃ (d : ℕ), d ∈ validDigits i ∧ isUnambiguous display) →
    (∃ (n : ℕ), n = maxNonFunctionalSegments ∧
      ∀ (m : ℕ), m > n →
        ¬(∀ (i : Fin 4), ∃ (d : ℕ), d ∈ validDigits i ∧ isUnambiguous display)) :=
by sorry

end max_non_functional_segments_is_13_l3829_382992


namespace y_multiples_l3829_382932

theorem y_multiples : ∃ (a b c d : ℤ),
  let y := 112 + 160 + 272 + 432 + 1040 + 1264 + 4256
  y = 16 * a ∧ y = 8 * b ∧ y = 4 * c ∧ y = 2 * d :=
by sorry

end y_multiples_l3829_382932


namespace square_field_side_length_l3829_382910

theorem square_field_side_length (area : Real) (side_length : Real) :
  area = 196 ∧ area = side_length ^ 2 → side_length = 14 := by
  sorry

end square_field_side_length_l3829_382910


namespace find_a_find_m_range_l3829_382951

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

-- Part 1
theorem find_a : 
  (∀ x, f 1 x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧ 
  (∀ a, (∀ x, f a x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) → a = 1) :=
sorry

-- Part 2
theorem find_m_range : 
  ∀ m : ℝ, (∃ n : ℝ, f 1 n ≤ m - f 1 (-n)) ↔ m ≥ 4 :=
sorry

end find_a_find_m_range_l3829_382951


namespace power_function_through_point_is_sqrt_l3829_382904

/-- A power function that passes through the point (4, 2) is equal to the square root function. -/
theorem power_function_through_point_is_sqrt (f : ℝ → ℝ) :
  (∃ a : ℝ, ∀ x : ℝ, f x = x ^ a) →  -- f is a power function
  f 4 = 2 →                         -- f passes through (4, 2)
  ∀ x : ℝ, f x = Real.sqrt x :=     -- f is the square root function
by sorry

end power_function_through_point_is_sqrt_l3829_382904


namespace library_book_distribution_l3829_382911

/-- Represents the number of books in a library -/
def total_books : ℕ := 6

/-- Calculates the number of ways to distribute books between library and checked-out status -/
def distribution_ways (n : ℕ) : ℕ :=
  if n ≥ 2 then n - 1 else 0

/-- Theorem stating that the number of ways to distribute the books is 5 -/
theorem library_book_distribution :
  distribution_ways total_books = 5 := by sorry

end library_book_distribution_l3829_382911


namespace math_majors_consecutive_probability_l3829_382969

/-- The number of people sitting at the round table -/
def total_people : ℕ := 9

/-- The number of math majors -/
def math_majors : ℕ := 4

/-- The number of ways to choose seats for math majors -/
def total_arrangements : ℕ := Nat.choose total_people math_majors

/-- The number of ways for math majors to sit in consecutive seats -/
def consecutive_arrangements : ℕ := total_people

/-- The probability that all math majors sit in consecutive seats -/
def probability : ℚ := consecutive_arrangements / total_arrangements

theorem math_majors_consecutive_probability :
  probability = 1 / 14 := by sorry

end math_majors_consecutive_probability_l3829_382969


namespace lyn_donation_l3829_382930

theorem lyn_donation (X : ℝ) : 
  (1 / 3 : ℝ) * X + (1 / 2 : ℝ) * X + (1 / 4 : ℝ) * ((1 : ℝ) - (1 / 3 : ℝ) - (1 / 2 : ℝ)) * X + 30 = X 
  → X = 240 := by
sorry

end lyn_donation_l3829_382930


namespace z_has_max_min_iff_a_in_range_l3829_382984

/-- The set A defined by the given inequalities -/
def A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 - 2 * p.2 + 8 ≥ 0 ∧ p.1 - p.2 - 1 ≤ 0 ∧ 2 * p.1 + a * p.2 - 2 ≤ 0}

/-- The function z defined as y - x -/
def z (p : ℝ × ℝ) : ℝ := p.2 - p.1

/-- Theorem stating the equivalence between the existence of max and min values for z
    and the range of a -/
theorem z_has_max_min_iff_a_in_range (a : ℝ) :
  (∃ (max min : ℝ), ∀ p ∈ A a, min ≤ z p ∧ z p ≤ max) ↔ a ≥ 2 :=
sorry

end z_has_max_min_iff_a_in_range_l3829_382984


namespace square_sum_of_integers_l3829_382953

theorem square_sum_of_integers (x y : ℕ+) 
  (h1 : x * y + x + y = 117)
  (h2 : x^2 * y + x * y^2 = 1512) : 
  x^2 + y^2 = 549 := by
sorry

end square_sum_of_integers_l3829_382953


namespace largest_constant_inequality_l3829_382901

theorem largest_constant_inequality (x y z : ℝ) :
  ∃ (C : ℝ), C = (2 + 2 * Real.sqrt 7) / 3 ∧
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z - 1)) ∧
  (∀ (C' : ℝ), C' > C → ∃ (x y z : ℝ), x^2 + y^2 + z^2 + 2 < C' * (x + y + z - 1)) :=
by sorry

end largest_constant_inequality_l3829_382901


namespace average_of_five_numbers_l3829_382991

theorem average_of_five_numbers : 
  let numbers : List ℕ := [8, 9, 10, 11, 12]
  (numbers.sum / numbers.length : ℚ) = 10 := by
sorry

end average_of_five_numbers_l3829_382991


namespace polynomial_root_product_l3829_382982

theorem polynomial_root_product (d e : ℝ) : 
  (∀ x : ℝ, x^2 + d*x + e = 0 ↔ x = Real.cos (π/9) ∨ x = Real.cos (2*π/9)) →
  d * e = -5/64 := by
  sorry

end polynomial_root_product_l3829_382982


namespace cos_five_pi_sixth_minus_x_l3829_382927

theorem cos_five_pi_sixth_minus_x (x : ℝ) 
  (h : Real.sin (π / 3 - x) = 3 / 5) : 
  Real.cos (5 * π / 6 - x) = -(3 / 5) := by
sorry

end cos_five_pi_sixth_minus_x_l3829_382927


namespace intersection_equality_l3829_382993

theorem intersection_equality (a : ℝ) : 
  (∀ x, (1 < x ∧ x < 7) ∧ (a + 1 < x ∧ x < 2*a + 5) ↔ (3 < x ∧ x < 7)) → 
  a = 2 := by
sorry

end intersection_equality_l3829_382993


namespace total_instruments_is_21_instrument_group_equality_l3829_382980

-- Define the number of body parts
def num_fingers : Nat := 10
def num_hands : Nat := 2
def num_heads : Nat := 1

-- Define the number of each instrument based on the conditions
def num_trumpets : Nat := num_fingers - 3
def num_guitars : Nat := num_hands + 2
def num_trombones : Nat := num_heads + 2
def num_french_horns : Nat := num_guitars - 1
def num_violins : Nat := num_trumpets / 2
def num_saxophones : Nat := num_trombones / 3

-- State the theorem
theorem total_instruments_is_21 :
  num_trumpets + num_guitars + num_trombones + num_french_horns + num_violins + num_saxophones = 21 :=
by sorry

-- Additional condition: equality of instrument groups
theorem instrument_group_equality :
  num_trumpets + num_guitars = num_trombones + num_violins + num_saxophones :=
by sorry

end total_instruments_is_21_instrument_group_equality_l3829_382980


namespace inverse_composition_f_inv_of_f_inv_of_f_inv_4_l3829_382950

def f : ℕ → ℕ
| 1 => 4
| 2 => 6
| 3 => 2
| 4 => 5
| 5 => 3
| 6 => 1
| _ => 0  -- Default case for completeness

-- Assumption that f is invertible
axiom f_invertible : Function.Injective f

-- Define f_inv as the inverse of f
noncomputable def f_inv : ℕ → ℕ := Function.invFun f

theorem inverse_composition (n : ℕ) : f_inv (f n) = n :=
  sorry

theorem f_inv_of_f_inv_of_f_inv_4 : f_inv (f_inv (f_inv 4)) = 2 :=
  sorry

end inverse_composition_f_inv_of_f_inv_of_f_inv_4_l3829_382950


namespace impossible_to_blacken_board_l3829_382947

/-- Represents the state of a chessboard -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- A move is represented by its top-left corner and orientation -/
structure Move where
  row : Fin 8
  col : Fin 8
  horizontal : Bool

/-- Apply a move to a chessboard -/
def applyMove (board : Chessboard) (move : Move) : Chessboard :=
  sorry

/-- Count the number of black squares on the board -/
def countBlackSquares (board : Chessboard) : Nat :=
  sorry

/-- The initial all-white chessboard -/
def initialBoard : Chessboard :=
  fun _ _ => false

/-- The final all-black chessboard -/
def finalBoard : Chessboard :=
  fun _ _ => true

/-- Theorem: It's impossible to transform the initial board to the final board using only valid moves -/
theorem impossible_to_blacken_board :
  ¬∃ (moves : List Move), (moves.foldl applyMove initialBoard) = finalBoard :=
sorry

end impossible_to_blacken_board_l3829_382947


namespace identity_function_unique_l3829_382998

def PositiveInt := {n : ℤ // n > 0}

def DivisibilityCondition (f : PositiveInt → PositiveInt) : Prop :=
  ∀ a b : PositiveInt, (a.val - (f b).val) ∣ (a.val * (f a).val - b.val * (f b).val)

theorem identity_function_unique :
  ∀ f : PositiveInt → PositiveInt,
    DivisibilityCondition f →
    ∀ x : PositiveInt, f x = x :=
by
  sorry

end identity_function_unique_l3829_382998


namespace expression_value_l3829_382931

theorem expression_value (a b : ℤ) (ha : a = 4) (hb : b = -3) :
  -a - b^2 + a*b = -25 := by
  sorry

end expression_value_l3829_382931


namespace stock_change_is_negative_4_375_percent_l3829_382999

/-- The overall percent change in a stock value after three days of fluctuations -/
def stock_percent_change : ℝ := by
  -- Define the daily changes
  let day1_change : ℝ := 0.85  -- 15% decrease
  let day2_change : ℝ := 1.25  -- 25% increase
  let day3_change : ℝ := 0.90  -- 10% decrease

  -- Calculate the overall change
  let overall_change : ℝ := day1_change * day2_change * day3_change

  -- Calculate the percent change
  exact (overall_change - 1) * 100

/-- Theorem stating that the overall percent change in the stock is -4.375% -/
theorem stock_change_is_negative_4_375_percent : 
  stock_percent_change = -4.375 := by
  sorry

end stock_change_is_negative_4_375_percent_l3829_382999


namespace quadratic_sequence_formula_l3829_382937

theorem quadratic_sequence_formula (a : ℕ → ℚ) (α β : ℚ) :
  (∀ n : ℕ, a n * α^2 - a (n + 1) * α + 1 = 0) →
  (∀ n : ℕ, a n * β^2 - a (n + 1) * β + 1 = 0) →
  (6 * α - 2 * α * β + 6 * β = 3) →
  (a 1 = 7 / 6) →
  (∀ n : ℕ, a n = (1 / 2)^n + 2 / 3) :=
by sorry

end quadratic_sequence_formula_l3829_382937


namespace line_param_solution_l3829_382945

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = -x + 3

/-- The parameterization of the line -/
def parameterization (u v m : ℝ) (x y : ℝ) : Prop :=
  x = 2 + u * m ∧ y = v + u * 8

/-- Theorem stating that v = 1 and m = -8 satisfy the line equation and parameterization -/
theorem line_param_solution :
  ∃ (v m : ℝ), v = 1 ∧ m = -8 ∧
  (∀ (x y u : ℝ), parameterization u v m x y → line_equation x y) :=
sorry

end line_param_solution_l3829_382945


namespace smallest_integer_for_negative_quadratic_l3829_382926

theorem smallest_integer_for_negative_quadratic : 
  ∃ (x : ℤ), (∀ (y : ℤ), y^2 - 11*y + 24 < 0 → x ≤ y) ∧ (x^2 - 11*x + 24 < 0) ∧ x = 4 := by
  sorry

end smallest_integer_for_negative_quadratic_l3829_382926


namespace positive_integer_solutions_for_mn_equation_l3829_382976

theorem positive_integer_solutions_for_mn_equation :
  ∀ m n : ℕ+,
  m^(n : ℕ) = n^((m : ℕ) - (n : ℕ)) →
  ((m = 9 ∧ n = 3) ∨ (m = 8 ∧ n = 2)) :=
by sorry

end positive_integer_solutions_for_mn_equation_l3829_382976


namespace coin_arrangement_count_l3829_382964

/-- Represents a coin with its type and orientation -/
inductive Coin
| Gold : Bool → Coin
| Silver : Bool → Coin

/-- Checks if two adjacent coins are not face to face -/
def notFaceToFace (c1 c2 : Coin) : Prop := sorry

/-- Checks if three consecutive coins do not have the same orientation -/
def notSameOrientation (c1 c2 c3 : Coin) : Prop := sorry

/-- Represents a valid arrangement of coins -/
def ValidArrangement (arrangement : List Coin) : Prop :=
  arrangement.length = 10 ∧
  (arrangement.filter (λ c => match c with | Coin.Gold _ => true | _ => false)).length = 5 ∧
  (arrangement.filter (λ c => match c with | Coin.Silver _ => true | _ => false)).length = 5 ∧
  (∀ i, i < 9 → notFaceToFace (arrangement.get ⟨i, sorry⟩) (arrangement.get ⟨i+1, sorry⟩)) ∧
  (∀ i, i < 8 → notSameOrientation (arrangement.get ⟨i, sorry⟩) (arrangement.get ⟨i+1, sorry⟩) (arrangement.get ⟨i+2, sorry⟩))

/-- The number of valid arrangements -/
def numValidArrangements : ℕ := sorry

theorem coin_arrangement_count :
  numValidArrangements = 8568 := by sorry

end coin_arrangement_count_l3829_382964


namespace cos_shift_l3829_382960

theorem cos_shift (x : ℝ) : 
  Real.cos (1/2 * x + π/3) = Real.cos (1/2 * (x + 2*π/3)) := by
  sorry

end cos_shift_l3829_382960


namespace divisibility_property_l3829_382965

theorem divisibility_property (a m n : ℕ) (ha : a > 1) (hdiv : (a^m + 1) ∣ (a^n + 1)) : m ∣ n := by
  sorry

end divisibility_property_l3829_382965


namespace adjacent_knights_probability_l3829_382977

def total_knights : ℕ := 30
def chosen_knights : ℕ := 4

def prob_adjacent_knights : ℚ :=
  1 - (26 * 24 * 22 * 20 : ℚ) / (26 * 27 * 28 * 29 : ℚ)

theorem adjacent_knights_probability :
  prob_adjacent_knights = 553 / 1079 := by sorry

end adjacent_knights_probability_l3829_382977


namespace x_plus_y_equals_four_l3829_382933

-- Define the conditions
def conditions (x y : ℝ) : Prop :=
  x ≥ -2 ∧ 
  y ≥ -3 ∧ 
  x - 2 * Real.sqrt (x + 2) = 2 * Real.sqrt (y + 3) - y

-- Theorem statement
theorem x_plus_y_equals_four (x y : ℝ) (h : conditions x y) : x + y = 4 := by
  sorry

end x_plus_y_equals_four_l3829_382933


namespace opposite_of_2023_l3829_382983

-- Define the opposite of a real number
def opposite (x : ℝ) : ℝ := -x

-- State the theorem
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end opposite_of_2023_l3829_382983


namespace train_speed_l3829_382944

/-- Proves that a train with given length and time to cross a pole has a specific speed in km/h -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 300) (h2 : crossing_time = 15) :
  (train_length / crossing_time) * 3.6 = 72 := by
  sorry

end train_speed_l3829_382944


namespace smallest_money_for_pizza_l3829_382967

theorem smallest_money_for_pizza (x : ℕ) : x ≥ 6 ↔ ∃ (a b : ℕ), x - 1 = 5 * a + 7 * b := by
  sorry

end smallest_money_for_pizza_l3829_382967


namespace plant_branches_l3829_382919

theorem plant_branches (x : ℕ) 
  (h1 : x > 0)
  (h2 : 1 + x + x * x = 31) : x = 5 := by
  sorry

end plant_branches_l3829_382919
