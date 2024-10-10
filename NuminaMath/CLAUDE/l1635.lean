import Mathlib

namespace reciprocal_of_repeating_decimal_one_third_l1635_163588

theorem reciprocal_of_repeating_decimal_one_third (x : ℚ) : 
  (x = 1/3) → (1/x = 3) := by
  sorry

end reciprocal_of_repeating_decimal_one_third_l1635_163588


namespace train_passing_time_l1635_163592

/-- The time taken for a train to pass a man moving in the opposite direction -/
theorem train_passing_time (train_speed : ℝ) (train_length : ℝ) (man_speed : ℝ) :
  train_speed = 60 →
  train_length = 110 →
  man_speed = 6 →
  ∃ t : ℝ, t > 0 ∧ t < 7 ∧
  t = train_length / ((train_speed + man_speed) * (1000 / 3600)) :=
by sorry

end train_passing_time_l1635_163592


namespace line_intersects_parabola_vertex_once_l1635_163539

theorem line_intersects_parabola_vertex_once :
  ∃! b : ℝ, ∀ x y : ℝ,
    (y = 2 * x + b) ∧ (y = x^2 + 2 * b * x) →
    (x = -b ∧ y = -b^2) := by
  sorry

end line_intersects_parabola_vertex_once_l1635_163539


namespace travel_time_calculation_l1635_163528

/-- Given a person travels 2 miles in 8 minutes, prove they will travel 5 miles in 20 minutes at the same rate. -/
theorem travel_time_calculation (distance_1 : ℝ) (time_1 : ℝ) (distance_2 : ℝ) 
  (h1 : distance_1 = 2) 
  (h2 : time_1 = 8) 
  (h3 : distance_2 = 5) :
  (distance_2 / (distance_1 / time_1)) = 20 := by
  sorry

end travel_time_calculation_l1635_163528


namespace clock_cost_price_l1635_163566

/-- The cost price of each clock satisfies the given conditions -/
theorem clock_cost_price (total_clocks : ℕ) (sold_at_10_percent : ℕ) (sold_at_20_percent : ℕ) 
  (uniform_profit_percentage : ℚ) (price_difference : ℚ) :
  total_clocks = 90 →
  sold_at_10_percent = 40 →
  sold_at_20_percent = 50 →
  uniform_profit_percentage = 15 / 100 →
  price_difference = 40 →
  ∃ (cost_price : ℚ),
    cost_price = 80 ∧
    cost_price * (sold_at_10_percent * (1 + 10 / 100) + sold_at_20_percent * (1 + 20 / 100)) =
    cost_price * total_clocks * (1 + uniform_profit_percentage) + price_difference :=
by sorry

end clock_cost_price_l1635_163566


namespace mean_temperature_l1635_163598

def temperatures : List ℤ := [-3, 0, 2, -1, 4, 5, 3]

theorem mean_temperature : 
  (List.sum temperatures) / (List.length temperatures) = 10 / 7 := by
  sorry

end mean_temperature_l1635_163598


namespace height_in_meters_l1635_163569

-- Define Xiaochao's height in meters and centimeters
def height_m : ℝ := 1
def height_cm : ℝ := 36

-- Theorem to prove
theorem height_in_meters : height_m + height_cm / 100 = 1.36 := by
  sorry

end height_in_meters_l1635_163569


namespace sphere_surface_area_ratio_l1635_163511

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : r₁ / r₂ = 1 / 2) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 4 := by
  sorry

end sphere_surface_area_ratio_l1635_163511


namespace smallest_third_term_geometric_progression_l1635_163524

theorem smallest_third_term_geometric_progression (d : ℝ) :
  (5 : ℝ) * (33 + 2 * d) = (8 + d) ^ 2 →
  ∃ (x : ℝ), x = 5 + 2 * d + 28 ∧
    ∀ (y : ℝ), (5 : ℝ) * (33 + 2 * y) = (8 + y) ^ 2 →
      5 + 2 * y + 28 ≥ x ∧
      x ≥ -21 :=
by sorry

end smallest_third_term_geometric_progression_l1635_163524


namespace jason_read_all_books_l1635_163564

theorem jason_read_all_books 
  (jason_books : ℕ) 
  (mary_books : ℕ) 
  (total_books : ℕ) 
  (h1 : jason_books = 18) 
  (h2 : mary_books = 42) 
  (h3 : total_books = 60) 
  (h4 : jason_books + mary_books = total_books) : 
  jason_books = 18 := by
sorry

end jason_read_all_books_l1635_163564


namespace expression_simplification_l1635_163586

theorem expression_simplification (q : ℝ) : 
  ((7*q + 3) - 3*q*5)*4 + (5 - 2/4)*(8*q - 12) = 4*q - 42 := by
  sorry

end expression_simplification_l1635_163586


namespace negative_two_inequality_l1635_163517

theorem negative_two_inequality (a b : ℝ) (h : a < b) : -2*a > -2*b := by
  sorry

end negative_two_inequality_l1635_163517


namespace no_partition_exists_l1635_163591

theorem no_partition_exists : ¬∃ (A B C : Set ℕ), 
  (A ∪ B ∪ C = Set.univ) ∧ 
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧
  (A ≠ ∅) ∧ (B ≠ ∅) ∧ (C ≠ ∅) ∧
  (∀ a b, a ∈ A → b ∈ B → a + b + 2008 ∈ C) ∧
  (∀ b c, b ∈ B → c ∈ C → b + c + 2008 ∈ A) ∧
  (∀ c a, c ∈ C → a ∈ A → c + a + 2008 ∈ B) := by
sorry

end no_partition_exists_l1635_163591


namespace orange_ratio_problem_l1635_163533

theorem orange_ratio_problem (michaela_oranges : ℕ) (total_oranges : ℕ) (remaining_oranges : ℕ) :
  michaela_oranges = 20 →
  total_oranges = 90 →
  remaining_oranges = 30 →
  (total_oranges - remaining_oranges - michaela_oranges) / michaela_oranges = 2 :=
by sorry

end orange_ratio_problem_l1635_163533


namespace complex_expression_value_l1635_163514

theorem complex_expression_value (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 := by
  sorry

end complex_expression_value_l1635_163514


namespace smallest_k_l1635_163538

theorem smallest_k (a b c k : ℤ) : 
  (a + 2 = b - 2) → 
  (a + 2 = (c : ℚ) / 2) → 
  (a + b + c = 2001 * k) → 
  (∀ m : ℤ, m > 0 → m < k → ¬(∃ a' b' c' : ℤ, 
    (a' + 2 = b' - 2) ∧ 
    (a' + 2 = (c' : ℚ) / 2) ∧ 
    (a' + b' + c' = 2001 * m))) → 
  k = 4 :=
by sorry

end smallest_k_l1635_163538


namespace gcd_special_numbers_l1635_163522

theorem gcd_special_numbers : Nat.gcd 777777777 222222222222 = 999 := by
  sorry

end gcd_special_numbers_l1635_163522


namespace negation_of_existence_negation_of_quadratic_equation_l1635_163509

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≠ 0) := by sorry

end negation_of_existence_negation_of_quadratic_equation_l1635_163509


namespace square_of_binomial_l1635_163573

theorem square_of_binomial (d : ℝ) : 
  (∃ a b : ℝ, ∀ x, 8*x^2 + 24*x + d = (a*x + b)^2) → d = 18 := by
sorry

end square_of_binomial_l1635_163573


namespace log_18_15_l1635_163543

-- Define the logarithm base 10 (lg) function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the variables a and b
variable (a b : ℝ)

-- State the theorem
theorem log_18_15 (h1 : lg 2 = a) (h2 : lg 3 = b) :
  (Real.log 15) / (Real.log 18) = (b - a + 1) / (a + 2 * b) := by
  sorry

end log_18_15_l1635_163543


namespace gcd_digits_bound_l1635_163507

theorem gcd_digits_bound (a b : ℕ) (ha : 1000000 ≤ a ∧ a < 10000000) (hb : 1000000 ≤ b ∧ b < 10000000)
  (hlcm : Nat.lcm a b < 100000000000) : Nat.gcd a b < 1000 := by
  sorry

end gcd_digits_bound_l1635_163507


namespace tan_triple_inequality_l1635_163549

theorem tan_triple_inequality (x y : Real) 
  (hx : 0 < x ∧ x < Real.pi / 2)
  (hy : 0 < y ∧ y < Real.pi / 2)
  (h_tan : Real.tan x = 3 * Real.tan y) :
  x - y ≤ Real.pi / 6 ∧
  (x - y = Real.pi / 6 ↔ x = Real.pi / 3 ∧ y = Real.pi / 6) := by
sorry

end tan_triple_inequality_l1635_163549


namespace function_satisfying_inequality_is_constant_l1635_163512

/-- A function satisfying the given inequality is constant -/
theorem function_satisfying_inequality_is_constant
  (f : ℝ → ℝ)
  (h : ∀ (x y : ℝ), |f x - f y| ≤ (x - y)^2) :
  ∃ (c : ℝ), ∀ (x : ℝ), f x = c :=
sorry

end function_satisfying_inequality_is_constant_l1635_163512


namespace expression_evaluation_l1635_163548

theorem expression_evaluation : 150 * (150 - 4) - (150 * 150 - 6 + 2) = -596 := by
  sorry

end expression_evaluation_l1635_163548


namespace count_sets_satisfying_union_l1635_163561

theorem count_sets_satisfying_union (A B : Set ℕ) : 
  A = {1, 2} → 
  (A ∪ B = {1, 2, 3, 4, 5}) → 
  (∃! (count : ℕ), ∃ (S : Finset (Set ℕ)), 
    (Finset.card S = count) ∧ 
    (∀ C ∈ S, A ∪ C = {1, 2, 3, 4, 5}) ∧
    (∀ D, A ∪ D = {1, 2, 3, 4, 5} → D ∈ S) ∧
    count = 4) :=
by sorry

end count_sets_satisfying_union_l1635_163561


namespace election_votes_l1635_163506

theorem election_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (62 : ℚ) / 100 * total_votes - (38 : ℚ) / 100 * total_votes = 408) :
  (62 : ℚ) / 100 * total_votes = 1054 :=
by sorry

end election_votes_l1635_163506


namespace bus_ride_cost_l1635_163542

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℝ := 1.50

/-- The cost of a train ride from town P to town Q -/
def train_cost : ℝ := bus_cost + 6.85

/-- The theorem stating the cost of a bus ride from town P to town Q -/
theorem bus_ride_cost : bus_cost = 1.50 := by
  have h1 : train_cost = bus_cost + 6.85 := by rfl
  have h2 : bus_cost + train_cost = 9.85 := by sorry
  sorry

end bus_ride_cost_l1635_163542


namespace cost_price_calculation_l1635_163585

theorem cost_price_calculation (C : ℝ) : 0.18 * C - 0.09 * C = 72 → C = 800 := by
  sorry

end cost_price_calculation_l1635_163585


namespace equation_equivalence_l1635_163599

theorem equation_equivalence (x : ℝ) : 
  (2*x + 1) / 3 - (5*x - 3) / 2 = 1 ↔ 2*(2*x + 1) - 3*(5*x - 3) = 6 := by
  sorry

end equation_equivalence_l1635_163599


namespace tribe_leadership_count_l1635_163587

def tribe_size : ℕ := 15

def leadership_arrangements : ℕ :=
  tribe_size *
  (tribe_size - 1) *
  (tribe_size - 2) *
  (tribe_size - 3) *
  (tribe_size - 4) *
  (Nat.choose (tribe_size - 5) 2) *
  (Nat.choose (tribe_size - 7) 2)

theorem tribe_leadership_count :
  leadership_arrangements = 216216000 := by
  sorry

end tribe_leadership_count_l1635_163587


namespace balloons_given_correct_fred_balloons_l1635_163574

/-- The number of balloons Fred gave to Sandy -/
def balloons_given (initial current : ℕ) : ℕ := initial - current

theorem balloons_given_correct (initial current : ℕ) (h : initial ≥ current) :
  balloons_given initial current = initial - current :=
by sorry

/-- Fred's scenario -/
theorem fred_balloons :
  let initial : ℕ := 709
  let current : ℕ := 488
  balloons_given initial current = 221 :=
by sorry

end balloons_given_correct_fred_balloons_l1635_163574


namespace farm_area_theorem_l1635_163504

/-- Represents a rectangular farm with fencing on one long side, one short side, and the diagonal -/
structure RectangularFarm where
  short_side : ℝ
  long_side : ℝ
  diagonal : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ

/-- Calculates the area of a rectangular farm -/
def farm_area (farm : RectangularFarm) : ℝ :=
  farm.short_side * farm.long_side

/-- Theorem: If a rectangular farm has a short side of 30 meters, and the cost of fencing
    one long side, one short side, and the diagonal at Rs. 15 per meter totals Rs. 1800,
    then the area of the farm is 1200 square meters -/
theorem farm_area_theorem (farm : RectangularFarm)
    (h1 : farm.short_side = 30)
    (h2 : farm.fencing_cost_per_meter = 15)
    (h3 : farm.total_fencing_cost = 1800)
    (h4 : farm.long_side + farm.short_side + farm.diagonal = farm.total_fencing_cost / farm.fencing_cost_per_meter)
    (h5 : farm.diagonal^2 = farm.long_side^2 + farm.short_side^2) :
    farm_area farm = 1200 := by
  sorry


end farm_area_theorem_l1635_163504


namespace intersection_of_A_and_B_l1635_163535

def A : Set ℝ := {x | x - 1 < 5}
def B : Set ℝ := {x | -4*x + 8 < 0}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 6} := by sorry

end intersection_of_A_and_B_l1635_163535


namespace greatest_x_value_l1635_163589

theorem greatest_x_value : 
  (∃ (x : ℤ), ∀ (y : ℤ), (2.13 * (10 : ℝ)^(y : ℝ) < 2100) → y ≤ x) ∧ 
  (2.13 * (10 : ℝ)^(2 : ℝ) < 2100) ∧ 
  (∀ (z : ℤ), z > 2 → 2.13 * (10 : ℝ)^(z : ℝ) ≥ 2100) :=
by sorry

end greatest_x_value_l1635_163589


namespace complex_magnitude_calculation_l1635_163521

theorem complex_magnitude_calculation (ω : ℂ) (h : ω = 7 + 3*I) :
  Complex.abs (ω^2 + 5*ω + 50) = Real.sqrt 18874 := by
  sorry

end complex_magnitude_calculation_l1635_163521


namespace total_pieces_eq_59_l1635_163530

/-- The number of pieces of clothing in the first load -/
def first_load : ℕ := 32

/-- The number of equal loads for the remaining clothing -/
def num_equal_loads : ℕ := 9

/-- The number of pieces of clothing in each of the equal loads -/
def pieces_per_equal_load : ℕ := 3

/-- The total number of pieces of clothing Will had to wash -/
def total_pieces : ℕ := first_load + num_equal_loads * pieces_per_equal_load

theorem total_pieces_eq_59 : total_pieces = 59 := by sorry

end total_pieces_eq_59_l1635_163530


namespace id_number_2520_l1635_163558

/-- A type representing a 7-digit identification number -/
def IdNumber := Fin 7 → Fin 7

/-- The set of all valid identification numbers -/
def ValidIdNumbers : Set IdNumber :=
  {id | Function.Injective id ∧ (∀ i, id i < 7)}

/-- Lexicographical order on identification numbers -/
def IdLexOrder (id1 id2 : IdNumber) : Prop :=
  ∃ k, (∀ i < k, id1 i = id2 i) ∧ id1 k < id2 k

/-- The nth identification number in lexicographical order -/
noncomputable def nthIdNumber (n : ℕ) : IdNumber :=
  sorry

/-- The main theorem: the 2520th identification number is 4376521 -/
theorem id_number_2520 :
  nthIdNumber 2520 = λ i =>
    match i with
    | 0 => 3  -- 4 (0-indexed)
    | 1 => 5  -- 6
    | 2 => 2  -- 3
    | 3 => 5  -- 6
    | 4 => 4  -- 5
    | 5 => 1  -- 2
    | 6 => 0  -- 1
  := by sorry

end id_number_2520_l1635_163558


namespace pool_filling_time_l1635_163551

theorem pool_filling_time (pool_capacity : ℝ) (num_hoses : ℕ) (flow_rate : ℝ) : 
  pool_capacity = 24000 ∧ 
  num_hoses = 4 ∧ 
  flow_rate = 2.5 → 
  pool_capacity / (num_hoses * flow_rate * 60) = 40 := by
sorry

end pool_filling_time_l1635_163551


namespace craft_store_solution_l1635_163516

/-- Represents the craft store problem -/
structure CraftStore where
  markedPrice : ℝ
  costPrice : ℝ
  profitPerItem : ℝ
  discountedSales : ℕ
  discountPercentage : ℝ
  reducedPriceSales : ℕ
  priceReduction : ℝ
  dailySales : ℕ
  salesIncrease : ℕ
  priceDecreaseStep : ℝ

/-- The craft store problem statement -/
def craftStoreProblem (cs : CraftStore) : Prop :=
  -- Profit at marked price
  cs.profitPerItem = cs.markedPrice - cs.costPrice
  -- Equal profit for discounted and reduced price sales
  ∧ cs.discountedSales * (cs.markedPrice * cs.discountPercentage - cs.costPrice) =
    cs.reducedPriceSales * (cs.markedPrice - cs.priceReduction - cs.costPrice)
  -- Daily sales at marked price
  ∧ cs.dailySales * (cs.markedPrice - cs.costPrice) =
    (cs.dailySales + cs.salesIncrease) * (cs.markedPrice - cs.priceDecreaseStep - cs.costPrice)

/-- The theorem to be proved -/
theorem craft_store_solution (cs : CraftStore) 
  (h : craftStoreProblem cs) : 
  cs.costPrice = 155 
  ∧ cs.markedPrice = 200 
  ∧ (∃ optimalReduction maxProfit, 
      optimalReduction = 10 
      ∧ maxProfit = 4900 
      ∧ ∀ reduction, 
        cs.dailySales * (cs.markedPrice - reduction - cs.costPrice) 
        + (cs.salesIncrease * reduction / cs.priceDecreaseStep) 
          * (cs.markedPrice - reduction - cs.costPrice) 
        ≤ maxProfit) :=
sorry

end craft_store_solution_l1635_163516


namespace b_equals_484_l1635_163571

/-- Given two real numbers a and b satisfying certain conditions,
    prove that b equals 484. -/
theorem b_equals_484 (a b : ℝ) 
    (h1 : a + b = 1210)
    (h2 : (4/15) * a = (2/5) * b) : 
  b = 484 := by sorry

end b_equals_484_l1635_163571


namespace period_3_odd_function_inequality_l1635_163544

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem period_3_odd_function_inequality (f : ℝ → ℝ) (a : ℝ) 
    (h_periodic : is_periodic f 3)
    (h_odd : is_odd f)
    (h_f1 : f 1 < 1)
    (h_f2 : f 2 = (2*a - 1)/(a + 1)) :
    a < -1 ∨ a > 0 := by
  sorry

end period_3_odd_function_inequality_l1635_163544


namespace specific_grid_square_count_l1635_163557

/-- Represents a square grid with some incomplete squares at the edges -/
structure SquareGrid :=
  (width : ℕ)
  (height : ℕ)
  (hasIncompleteEdges : Bool)

/-- Counts the number of squares of a given size in the grid -/
def countSquares (grid : SquareGrid) (size : ℕ) : ℕ :=
  sorry

/-- Counts the total number of squares in the grid -/
def totalSquares (grid : SquareGrid) : ℕ :=
  (countSquares grid 1) + (countSquares grid 2) + (countSquares grid 3)

/-- The main theorem stating that the total number of squares in the specific grid is 38 -/
theorem specific_grid_square_count :
  ∃ (grid : SquareGrid), grid.width = 5 ∧ grid.height = 5 ∧ grid.hasIncompleteEdges = true ∧ totalSquares grid = 38 :=
  sorry

end specific_grid_square_count_l1635_163557


namespace specific_gathering_handshakes_l1635_163547

/-- Represents a gathering of married couples -/
structure Gathering where
  couples : ℕ
  people : ℕ
  circular : Bool
  shake_all : Bool
  no_spouse : Bool
  no_neighbors : Bool

/-- Calculates the number of handshakes in the gathering -/
def handshakes (g : Gathering) : ℕ :=
  (g.people * (g.people - 3)) / 2

/-- Theorem stating the number of handshakes for the specific gathering described in the problem -/
theorem specific_gathering_handshakes :
  let g : Gathering := {
    couples := 8,
    people := 16,
    circular := true,
    shake_all := true,
    no_spouse := true,
    no_neighbors := true
  }
  handshakes g = 96 := by
  sorry

end specific_gathering_handshakes_l1635_163547


namespace no_eighteen_consecutive_good_numbers_l1635_163508

/-- A natural number is good if it has exactly two prime divisors. -/
def IsGood (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ (∀ r : ℕ, Prime r → r ∣ n → r = p ∨ r = q)

/-- Theorem: It is impossible for 18 consecutive natural numbers to all be good. -/
theorem no_eighteen_consecutive_good_numbers :
  ¬∃ start : ℕ, ∀ i : ℕ, i < 18 → IsGood (start + i) := by
  sorry

end no_eighteen_consecutive_good_numbers_l1635_163508


namespace incorrect_factorization_l1635_163576

theorem incorrect_factorization (x : ℝ) : x^2 - 7*x + 12 ≠ x*(x - 7) + 12 := by
  sorry

end incorrect_factorization_l1635_163576


namespace sum_of_min_max_x_l1635_163596

theorem sum_of_min_max_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  ∃ (m M : ℝ), (∀ x' y' z' : ℝ, x' + y' + z' = 5 → x'^2 + y'^2 + z'^2 = 11 → m ≤ x' ∧ x' ≤ M) ∧
                m + M = 8/3 :=
sorry

end sum_of_min_max_x_l1635_163596


namespace pats_family_size_l1635_163541

theorem pats_family_size (total_desserts : ℕ) (desserts_per_person : ℕ) 
  (h1 : total_desserts = 126)
  (h2 : desserts_per_person = 18) :
  total_desserts / desserts_per_person = 7 := by
  sorry

end pats_family_size_l1635_163541


namespace total_turtles_l1635_163520

theorem total_turtles (kristen_turtles : ℕ) (kris_turtles : ℕ) (trey_turtles : ℕ) :
  kristen_turtles = 12 →
  kris_turtles = kristen_turtles / 4 →
  trey_turtles = 5 * kris_turtles →
  kristen_turtles + kris_turtles + trey_turtles = 30 := by
  sorry

end total_turtles_l1635_163520


namespace problem_proof_l1635_163518

theorem problem_proof : (-8: ℝ) ^ (1/3) + π^0 + Real.log 4 + Real.log 25 = 1 := by
  sorry

end problem_proof_l1635_163518


namespace absolute_value_equals_negative_l1635_163525

theorem absolute_value_equals_negative (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end absolute_value_equals_negative_l1635_163525


namespace points_on_line_l1635_163565

/-- Given points M(a, 1/b) and N(b, 1/c) on the line x + y = 1,
    prove that points P(c, 1/a) and Q(1/c, b) are also on the same line. -/
theorem points_on_line (a b c : ℝ) (ha : a + 1/b = 1) (hb : b + 1/c = 1) :
  c + 1/a = 1 ∧ 1/c + b = 1 := by
  sorry

end points_on_line_l1635_163565


namespace ice_cream_volume_l1635_163559

/-- The volume of ice cream on a cone -/
theorem ice_cream_volume (r h : ℝ) (hr : r = 3) (hh : h = 1) :
  (2/3) * π * r^3 + π * r^2 * h = 27 * π := by sorry

end ice_cream_volume_l1635_163559


namespace two_fifths_of_seven_point_five_l1635_163568

theorem two_fifths_of_seven_point_five : (2 / 5 : ℚ) * (15 / 2 : ℚ) = (3 : ℚ) := by
  sorry

end two_fifths_of_seven_point_five_l1635_163568


namespace repetend_of_five_seventeenths_l1635_163523

/-- The decimal representation of 5/17 has a repetend of 294117647058823529 -/
theorem repetend_of_five_seventeenths :
  ∃ (n : ℕ), (5 : ℚ) / 17 = (n : ℚ) / 999999999999999999 ∧
  n = 294117647058823529 := by
  sorry

end repetend_of_five_seventeenths_l1635_163523


namespace specific_cistern_wet_surface_area_l1635_163546

/-- Calculates the total wet surface area of a rectangular cistern. -/
def cistern_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem stating the total wet surface area of a specific cistern. -/
theorem specific_cistern_wet_surface_area :
  cistern_wet_surface_area 10 6 1.35 = 103.2 := by
  sorry

end specific_cistern_wet_surface_area_l1635_163546


namespace min_knights_in_tournament_l1635_163505

def knight_tournament (total_knights : ℕ) : Prop :=
  ∃ (lancelot_not_dueled : ℕ),
    lancelot_not_dueled = total_knights / 4 ∧
    ∃ (tristan_dueled : ℕ),
      tristan_dueled = (total_knights - lancelot_not_dueled - 1) / 7 ∧
      (total_knights - lancelot_not_dueled - 1) % 7 = 0

theorem min_knights_in_tournament :
  ∀ n : ℕ, knight_tournament n → n ≥ 20 :=
by sorry

end min_knights_in_tournament_l1635_163505


namespace max_right_angles_is_14_l1635_163581

/-- A triangular prism -/
structure TriangularPrism :=
  (faces : Nat)
  (angles : Nat)
  (h_faces : faces = 5)
  (h_angles : angles = 18)

/-- The maximum number of right angles in a triangular prism -/
def max_right_angles (prism : TriangularPrism) : Nat := 14

/-- Theorem: The maximum number of right angles in a triangular prism is 14 -/
theorem max_right_angles_is_14 (prism : TriangularPrism) :
  max_right_angles prism = 14 := by sorry

end max_right_angles_is_14_l1635_163581


namespace circle_triangle_area_l1635_163590

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def CircleTangentToLine (c : Circle) (l : Line) : Prop := sorry

def CirclesInternallyTangent (c1 c2 : Circle) : Prop := sorry

def CirclesExternallyTangent (c1 c2 : Circle) : Prop := sorry

def PointBetween (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

def AreaOfTriangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem circle_triangle_area 
  (A B C : Circle)
  (m : Line)
  (A' B' C' : ℝ × ℝ) :
  A.radius = 3 →
  B.radius = 4 →
  C.radius = 5 →
  CircleTangentToLine A m →
  CircleTangentToLine B m →
  CircleTangentToLine C m →
  PointBetween A' B' C' →
  CirclesInternallyTangent A B →
  CirclesExternallyTangent B C →
  AreaOfTriangle A.center B.center C.center = 7 := by
  sorry

end circle_triangle_area_l1635_163590


namespace benjamin_car_insurance_expenditure_l1635_163519

/-- The annual expenditure on car insurance, given the total expenditure over a decade -/
def annual_expenditure (total_expenditure : ℕ) (years : ℕ) : ℕ :=
  total_expenditure / years

/-- Theorem stating that the annual expenditure is 3000 dollars given the conditions -/
theorem benjamin_car_insurance_expenditure :
  annual_expenditure 30000 10 = 3000 := by
  sorry

end benjamin_car_insurance_expenditure_l1635_163519


namespace inequality_solution_l1635_163545

theorem inequality_solution (x : ℝ) : 
  (-1 < (x^2 - 10*x + 9) / (x^2 - 4*x + 8) ∧ (x^2 - 10*x + 9) / (x^2 - 4*x + 8) < 1) ↔ x > 1/6 := by
  sorry

end inequality_solution_l1635_163545


namespace equation_proof_l1635_163597

theorem equation_proof : (5568 / 87)^(1/3) + (72 * 2)^(1/2) = (256)^(1/2) := by
  sorry

end equation_proof_l1635_163597


namespace equation_solution_l1635_163552

theorem equation_solution : 
  ∃ x : ℝ, (5 * 0.85) / x - (8 * 2.25) = 5.5 ∧ x = 4.25 / 23.5 := by
  sorry

end equation_solution_l1635_163552


namespace divisibility_condition_l1635_163562

theorem divisibility_condition (a b : ℕ+) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔ 
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
by sorry

end divisibility_condition_l1635_163562


namespace inverse_iff_horizontal_line_test_l1635_163583

-- Define a type for our functions
def Function := ℝ → ℝ

-- Define what it means for a function to have an inverse
def has_inverse (f : Function) : Prop :=
  ∃ g : Function, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Define the horizontal line test
def passes_horizontal_line_test (f : Function) : Prop :=
  ∀ y : ℝ, ∀ x₁ x₂ : ℝ, f x₁ = y ∧ f x₂ = y → x₁ = x₂

-- State the theorem
theorem inverse_iff_horizontal_line_test (f : Function) :
  has_inverse f ↔ passes_horizontal_line_test f :=
sorry

end inverse_iff_horizontal_line_test_l1635_163583


namespace mean_median_difference_l1635_163567

/-- Represents the frequency histogram data for student absences -/
structure AbsenceData where
  zero_days : Nat
  one_day : Nat
  two_days : Nat
  three_days : Nat
  four_days : Nat
  total_students : Nat
  sum_condition : zero_days + one_day + two_days + three_days + four_days = total_students

/-- Calculates the mean number of days absent -/
def calculate_mean (data : AbsenceData) : Rat :=
  (0 * data.zero_days + 1 * data.one_day + 2 * data.two_days + 3 * data.three_days + 4 * data.four_days) / data.total_students

/-- Calculates the median number of days absent -/
def calculate_median (data : AbsenceData) : Nat :=
  if data.zero_days + data.one_day < data.total_students / 2 then 2 else 1

/-- Theorem stating the difference between mean and median -/
theorem mean_median_difference (data : AbsenceData) 
  (h : data.total_students = 20 ∧ 
       data.zero_days = 4 ∧ 
       data.one_day = 2 ∧ 
       data.two_days = 5 ∧ 
       data.three_days = 6 ∧ 
       data.four_days = 3) : 
  calculate_mean data - calculate_median data = 1 / 10 := by
  sorry

end mean_median_difference_l1635_163567


namespace calculation_proof_l1635_163537

theorem calculation_proof : 
  Real.sqrt 12 - abs (-1) + (1/2)⁻¹ + (2023 + Real.pi)^0 = 2 * Real.sqrt 3 + 2 := by
sorry

end calculation_proof_l1635_163537


namespace sum_of_ages_l1635_163555

-- Define the present ages of father and son
def father_age : ℚ := sorry
def son_age : ℚ := sorry

-- Define the conditions
def present_ratio : father_age / son_age = 7 / 4 := sorry
def future_ratio : (father_age + 10) / (son_age + 10) = 5 / 3 := sorry

-- Theorem to prove
theorem sum_of_ages : father_age + son_age = 220 := by sorry

end sum_of_ages_l1635_163555


namespace chord_circuit_l1635_163515

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle between two adjacent chords is 60°, then the minimum number of such chords
    needed to form a complete circuit is 3. -/
theorem chord_circuit (angle : ℝ) (n : ℕ) : angle = 60 → n * angle = 360 → n = 3 := by
  sorry

end chord_circuit_l1635_163515


namespace right_triangle_leg_sum_l1635_163584

theorem right_triangle_leg_sum : 
  ∀ (a b : ℕ), 
  (∃ k : ℕ, a = 2 * k ∧ b = 2 * k + 2) → -- legs are consecutive even whole numbers
  a^2 + b^2 = 50^2 → -- Pythagorean theorem with hypotenuse 50
  a + b = 80 := by
sorry

end right_triangle_leg_sum_l1635_163584


namespace susan_spending_l1635_163534

def carnival_spending (initial_amount food_cost : ℝ) : ℝ :=
  let ride_cost := 2 * food_cost
  let game_cost := 0.5 * food_cost
  let total_spent := food_cost + ride_cost + game_cost
  initial_amount - total_spent

theorem susan_spending :
  carnival_spending 80 15 = 27.5 := by
  sorry

end susan_spending_l1635_163534


namespace unique_a_value_l1635_163502

def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + 4 = 0}

theorem unique_a_value : ∃! a : ℝ, (B a).Nonempty ∧ B a ⊆ A ∧ a = 4 := by sorry

end unique_a_value_l1635_163502


namespace simplify_expression_l1635_163594

theorem simplify_expression (z : ℝ) : (5 - 4 * z^2) - (7 - 6 * z + 3 * z^2) = -2 - 7 * z^2 + 6 * z := by
  sorry

end simplify_expression_l1635_163594


namespace min_value_inequality_l1635_163529

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1/a + 1/b + 2 * Real.sqrt (a * b) ≥ 4 := by
  sorry

end min_value_inequality_l1635_163529


namespace total_tax_collection_l1635_163503

/-- Represents the farm tax collection in a village -/
structure FarmTaxCollection where
  totalTax : ℝ
  farmerTax : ℝ
  farmerLandRatio : ℝ

/-- Theorem: Given a farmer's tax payment and land ratio, prove the total tax collected -/
theorem total_tax_collection (ftc : FarmTaxCollection) 
  (h1 : ftc.farmerTax = 480)
  (h2 : ftc.farmerLandRatio = 0.3125)
  : ftc.totalTax = 1536 := by
  sorry

#check total_tax_collection

end total_tax_collection_l1635_163503


namespace age_ratio_problem_l1635_163577

/-- Proves that the ratio of b's age to c's age is 2:1 given the problem conditions -/
theorem age_ratio_problem (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  a + b + c = 52 →  -- total of ages is 52
  b = 20 →  -- b is 20 years old
  b = 2 * c  -- ratio of b's age to c's age is 2:1
:= by sorry

end age_ratio_problem_l1635_163577


namespace optimal_ships_l1635_163575

/-- The maximum annual shipbuilding capacity -/
def max_capacity : ℕ := 20

/-- The output value function -/
def R (x : ℕ) : ℚ := 3700 * x + 45 * x^2 - 10 * x^3

/-- The cost function -/
def C (x : ℕ) : ℚ := 460 * x + 5000

/-- The profit function -/
def p (x : ℕ) : ℚ := R x - C x

/-- The marginal function of a function f -/
def M (f : ℕ → ℚ) (x : ℕ) : ℚ := f (x + 1) - f x

/-- The theorem stating the optimal number of ships to build -/
theorem optimal_ships : 
  ∃ (x : ℕ), x ≤ max_capacity ∧ x > 0 ∧
  ∀ (y : ℕ), y ≤ max_capacity ∧ y > 0 → p x ≥ p y ∧
  x = 12 :=
sorry

end optimal_ships_l1635_163575


namespace trinomial_square_difference_l1635_163553

theorem trinomial_square_difference : (23 + 15 + 7)^2 - (23^2 + 15^2 + 7^2) = 1222 := by
  sorry

end trinomial_square_difference_l1635_163553


namespace rectangle_length_l1635_163563

theorem rectangle_length (width perimeter : ℝ) (h1 : width = 15) (h2 : perimeter = 70) :
  let length := (perimeter - 2 * width) / 2
  length = 20 :=
by
  sorry

end rectangle_length_l1635_163563


namespace triangle_proof_l1635_163540

theorem triangle_proof (A B C : Real) (a b c : Real) (R : Real) :
  let D := (A + C) / 2  -- D is midpoint of AC
  (1/2) * Real.sin (2*B) * Real.cos C + Real.cos B ^ 2 * Real.sin C - Real.sin (A/2) * Real.cos (A/2) = 0 →
  R = Real.sqrt 3 →
  B = π/3 ∧ 
  Real.sqrt ((a^2 + c^2) * 2 - 9) / 2 = 
    Real.sqrt ((Real.sin A * R)^2 + (Real.sin C * R)^2 - (Real.sin B * R)^2 / 4) :=
by sorry


end triangle_proof_l1635_163540


namespace equation_solutions_l1635_163570

theorem equation_solutions :
  ∃! (x y : ℝ), y = (x + 2)^2 ∧ x * y + 2 * y = 2 ∧
  ∃ (a b c d : ℂ), a ≠ x ∧ c ≠ x ∧
    (a, b) ≠ (c, d) ∧
    b = (a + 2)^2 ∧ a * b + 2 * b = 2 ∧
    d = (c + 2)^2 ∧ c * d + 2 * d = 2 :=
by sorry

end equation_solutions_l1635_163570


namespace wood_measurement_theorem_l1635_163536

/-- Represents the measurement of a piece of wood with a rope -/
structure WoodMeasurement where
  wood_length : ℝ
  rope_length : ℝ
  surplus : ℝ
  half_rope_shortage : ℝ

/-- The system of equations accurately represents the wood measurement situation -/
def accurate_representation (m : WoodMeasurement) : Prop :=
  (m.rope_length = m.wood_length + m.surplus) ∧
  (0.5 * m.rope_length = m.wood_length - m.half_rope_shortage)

/-- Theorem stating that the given conditions lead to the correct system of equations -/
theorem wood_measurement_theorem (m : WoodMeasurement) 
  (h1 : m.surplus = 4.5)
  (h2 : m.half_rope_shortage = 1) :
  accurate_representation m := by
  sorry

end wood_measurement_theorem_l1635_163536


namespace inverse_f_243_l1635_163578

def f (x : ℝ) : ℝ := sorry

theorem inverse_f_243 (h1 : f 5 = 3) (h2 : ∀ x, f (3 * x) = 3 * f x) : 
  f 405 = 243 := by sorry

end inverse_f_243_l1635_163578


namespace second_class_revenue_l1635_163593

/-- The amount collected from II class passengers given the passenger and fare ratios --/
theorem second_class_revenue (total_revenue : ℚ) 
  (h1 : total_revenue = 1325)
  (h2 : ∃ (x y : ℚ), x * y * 53 = total_revenue ∧ x > 0 ∧ y > 0) :
  ∃ (x y : ℚ), 50 * x * y = 1250 :=
sorry

end second_class_revenue_l1635_163593


namespace root_exists_in_interval_l1635_163501

theorem root_exists_in_interval : ∃ x : ℝ, 3/2 < x ∧ x < 2 ∧ 2^x = x^2 + 1/2 := by
  sorry

end root_exists_in_interval_l1635_163501


namespace maci_red_pens_l1635_163531

/-- The number of blue pens Maci needs -/
def blue_pens : ℕ := 10

/-- The cost of a blue pen in cents -/
def blue_pen_cost : ℕ := 10

/-- The cost of a red pen in cents -/
def red_pen_cost : ℕ := 2 * blue_pen_cost

/-- The total cost of all pens in cents -/
def total_cost : ℕ := 400

/-- The number of red pens Maci needs -/
def red_pens : ℕ := 15

theorem maci_red_pens :
  blue_pens * blue_pen_cost + red_pens * red_pen_cost = total_cost :=
sorry

end maci_red_pens_l1635_163531


namespace parents_john_age_ratio_l1635_163532

/-- Given information about Mark, John, and their parents' ages, prove the ratio of parents' age to John's age -/
theorem parents_john_age_ratio :
  ∀ (mark_age john_age parents_age : ℕ),
    mark_age = 18 →
    john_age = mark_age - 10 →
    parents_age = 22 + mark_age →
    parents_age / john_age = 5 := by
  sorry

end parents_john_age_ratio_l1635_163532


namespace road_trip_driving_time_l1635_163560

/-- Calculates the total driving time for a road trip given the number of days and daily driving hours for two people. -/
def total_driving_time (days : ℕ) (person1_hours : ℕ) (person2_hours : ℕ) : ℕ :=
  days * (person1_hours + person2_hours)

/-- Theorem stating that for a 3-day road trip with given driving hours, the total driving time is 42 hours. -/
theorem road_trip_driving_time :
  total_driving_time 3 8 6 = 42 := by
  sorry

end road_trip_driving_time_l1635_163560


namespace distinct_pairs_solution_l1635_163579

theorem distinct_pairs_solution (x y : ℝ) : 
  x ≠ y ∧ 
  x^100 - y^100 = 2^99 * (x - y) ∧ 
  x^200 - y^200 = 2^199 * (x - y) → 
  (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) := by
sorry

end distinct_pairs_solution_l1635_163579


namespace complex_polynomial_root_implies_abs_c_165_l1635_163526

def complex_polynomial (a b c : ℤ) : ℂ → ℂ := fun z ↦ a * z^4 + b * z^3 + c * z^2 + b * z + a

theorem complex_polynomial_root_implies_abs_c_165 (a b c : ℤ) :
  complex_polynomial a b c (3 + I) = 0 →
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 165 := by sorry

end complex_polynomial_root_implies_abs_c_165_l1635_163526


namespace frog_reach_edge_prob_l1635_163550

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Defines the 4x4 grid with wraparound edges -/
def Grid := Set Position

/-- Checks if a position is on the edge of the grid -/
def isEdge (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Represents a single hop direction -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Applies a hop in the given direction, considering wraparound -/
def hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨p.x, (p.y + 1) % 4⟩
  | Direction.Down => ⟨p.x, (p.y - 1 + 4) % 4⟩
  | Direction.Left => ⟨(p.x - 1 + 4) % 4, p.y⟩
  | Direction.Right => ⟨(p.x + 1) % 4, p.y⟩

/-- Defines the probability of reaching an edge within n hops -/
def probReachEdge (start : Position) (n : Nat) : ℝ :=
  sorry

theorem frog_reach_edge_prob :
  probReachEdge ⟨3, 3⟩ 5 = 1 := by sorry

end frog_reach_edge_prob_l1635_163550


namespace solution_set_when_m_neg_one_range_of_m_l1635_163595

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + |2*x - 1|

-- Part I
theorem solution_set_when_m_neg_one :
  {x : ℝ | f x (-1) ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Part II
def A (m : ℝ) : Set ℝ := {x : ℝ | f x m ≤ |2*x + 1|}

theorem range_of_m (h : Set.Icc (3/4 : ℝ) 2 ⊆ A m) :
  m ∈ Set.Icc (-11/4 : ℝ) 0 := by sorry

end solution_set_when_m_neg_one_range_of_m_l1635_163595


namespace right_triangle_area_l1635_163513

/-- The area of a right triangle with hypotenuse 5 and one leg 3 is 6 -/
theorem right_triangle_area : ∀ (a b c : ℝ), 
  a = 3 → 
  c = 5 → 
  a^2 + b^2 = c^2 → 
  (1/2) * a * b = 6 :=
by
  sorry

end right_triangle_area_l1635_163513


namespace lunch_costs_more_than_breakfast_l1635_163582

/-- Represents the cost of Anna's meals -/
structure MealCosts where
  bagel : ℝ
  orange_juice : ℝ
  sandwich : ℝ
  milk : ℝ

/-- Calculates the difference between lunch and breakfast costs -/
def lunch_breakfast_difference (costs : MealCosts) : ℝ :=
  (costs.sandwich + costs.milk) - (costs.bagel + costs.orange_juice)

/-- Theorem stating the difference between lunch and breakfast costs -/
theorem lunch_costs_more_than_breakfast (costs : MealCosts) 
  (h1 : costs.bagel = 0.95)
  (h2 : costs.orange_juice = 0.85)
  (h3 : costs.sandwich = 4.65)
  (h4 : costs.milk = 1.15) :
  lunch_breakfast_difference costs = 4.00 := by
  sorry

end lunch_costs_more_than_breakfast_l1635_163582


namespace solution_difference_l1635_163580

theorem solution_difference (r s : ℝ) : 
  (((6 * r - 18) / (r^2 + 3*r - 18) = r + 3) ∧
   ((6 * s - 18) / (s^2 + 3*s - 18) = s + 3) ∧
   (r ≠ s) ∧
   (r > s)) → 
  (r - s = 11) := by
sorry

end solution_difference_l1635_163580


namespace min_cans_needed_l1635_163510

/-- The capacity of each can in ounces -/
def can_capacity : ℕ := 15

/-- The minimum amount of soda needed in ounces -/
def min_soda_amount : ℕ := 192

/-- The minimum number of cans needed -/
def min_cans : ℕ := 13

theorem min_cans_needed : 
  (∀ n : ℕ, n * can_capacity ≥ min_soda_amount → n ≥ min_cans) ∧ 
  (min_cans * can_capacity ≥ min_soda_amount) := by
  sorry

end min_cans_needed_l1635_163510


namespace f_composition_value_l1635_163500

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sin (Real.pi * x)
  else Real.cos (Real.pi * x / 2 + Real.pi / 3)

theorem f_composition_value : f (f (15/2)) = Real.sqrt 3 / 2 := by
  sorry

end f_composition_value_l1635_163500


namespace rachel_picture_book_shelves_l1635_163572

/-- Calculates the number of picture book shelves given the total number of books,
    number of mystery book shelves, and books per shelf. -/
def picture_book_shelves (total_books : ℕ) (mystery_shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf

/-- Proves that Rachel has 2 shelves of picture books given the problem conditions. -/
theorem rachel_picture_book_shelves :
  picture_book_shelves 72 6 9 = 2 := by
  sorry

end rachel_picture_book_shelves_l1635_163572


namespace marble_count_l1635_163554

theorem marble_count (n : ℕ) (left_pos right_pos : ℕ) 
  (h1 : left_pos = 5)
  (h2 : right_pos = 3)
  (h3 : n = left_pos + right_pos - 1) :
  n = 7 := by
sorry

end marble_count_l1635_163554


namespace octagon_has_eight_sides_l1635_163556

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Theorem stating that an octagon has 8 sides -/
theorem octagon_has_eight_sides : octagon_sides = 8 := by
  sorry

end octagon_has_eight_sides_l1635_163556


namespace ship_passengers_l1635_163527

theorem ship_passengers :
  ∀ (P : ℕ),
  (P / 20 : ℚ) + (P / 15 : ℚ) + (P / 10 : ℚ) + (P / 12 : ℚ) + (P / 30 : ℚ) + 60 = P →
  P = 90 :=
by
  sorry

end ship_passengers_l1635_163527
