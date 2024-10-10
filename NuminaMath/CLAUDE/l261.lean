import Mathlib

namespace razorback_tshirt_sales_l261_26109

/-- The Razorback T-shirt Shop problem -/
theorem razorback_tshirt_sales (profit_per_shirt : ℕ) (total_profit : ℕ) 
    (h1 : profit_per_shirt = 9)
    (h2 : total_profit = 2205) :
  total_profit / profit_per_shirt = 245 := by
  sorry

#check razorback_tshirt_sales

end razorback_tshirt_sales_l261_26109


namespace rotate_D_180_about_origin_l261_26135

def rotate_180_about_origin (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem rotate_D_180_about_origin :
  let D : ℝ × ℝ := (-6, 2)
  rotate_180_about_origin D = (6, -2) := by
  sorry

end rotate_D_180_about_origin_l261_26135


namespace prob_three_odd_less_than_one_eighth_l261_26126

def n : ℕ := 2016

def odd_count : ℕ := n / 2

def prob_three_odd : ℚ :=
  (odd_count : ℚ) / n *
  ((odd_count - 1) : ℚ) / (n - 1) *
  ((odd_count - 2) : ℚ) / (n - 2)

theorem prob_three_odd_less_than_one_eighth :
  prob_three_odd < 1 / 8 := by
  sorry

end prob_three_odd_less_than_one_eighth_l261_26126


namespace power_equality_l261_26195

theorem power_equality (k : ℕ) : 9^4 = 3^k → k = 8 := by
  sorry

end power_equality_l261_26195


namespace min_colours_for_cube_l261_26141

/-- Represents a colouring of a cube's faces -/
def CubeColouring := Fin 6 → ℕ

/-- Checks if two face indices are adjacent on a cube -/
def are_adjacent (i j : Fin 6) : Prop :=
  (i.val + j.val) % 2 = 1 ∧ i ≠ j

/-- A valid colouring has different colours for adjacent faces -/
def is_valid_colouring (c : CubeColouring) : Prop :=
  ∀ i j : Fin 6, are_adjacent i j → c i ≠ c j

/-- The number of colours used in a colouring -/
def num_colours (c : CubeColouring) : ℕ :=
  Finset.card (Finset.image c Finset.univ)

/-- There exists a valid 3-colouring of a cube -/
axiom exists_valid_3_colouring : ∃ c : CubeColouring, is_valid_colouring c ∧ num_colours c = 3

/-- Any valid colouring of a cube uses at least 3 colours -/
axiom valid_colouring_needs_at_least_3 : ∀ c : CubeColouring, is_valid_colouring c → num_colours c ≥ 3

theorem min_colours_for_cube : ∃ n : ℕ, n = 3 ∧
  (∃ c : CubeColouring, is_valid_colouring c ∧ num_colours c = n) ∧
  (∀ c : CubeColouring, is_valid_colouring c → num_colours c ≥ n) :=
sorry

end min_colours_for_cube_l261_26141


namespace connie_grandmother_brother_birth_year_l261_26175

/-- The year Connie's grandmother's older brother was born -/
def older_brother_birth_year : ℕ := sorry

/-- The year Connie's grandmother's older sister was born -/
def older_sister_birth_year : ℕ := 1936

/-- The year Connie's grandmother was born -/
def grandmother_birth_year : ℕ := 1944

theorem connie_grandmother_brother_birth_year :
  (grandmother_birth_year - older_sister_birth_year = 2 * (older_sister_birth_year - older_brother_birth_year)) →
  older_brother_birth_year = 1932 := by
  sorry

end connie_grandmother_brother_birth_year_l261_26175


namespace problem_solution_l261_26193

theorem problem_solution : ∃ x : ℝ, 10 * x = 2 * x - 36 ∧ x = -4.5 := by
  sorry

end problem_solution_l261_26193


namespace tile_arrangements_l261_26184

/-- The number of distinguishable arrangements of tiles -/
def num_arrangements (brown purple red yellow : ℕ) : ℕ :=
  Nat.factorial (brown + purple + red + yellow) /
  (Nat.factorial brown * Nat.factorial purple * Nat.factorial red * Nat.factorial yellow)

/-- Theorem stating that the number of distinguishable arrangements
    of 1 brown, 2 purple, 2 red, and 3 yellow tiles is 1680 -/
theorem tile_arrangements :
  num_arrangements 1 2 2 3 = 1680 := by
  sorry

end tile_arrangements_l261_26184


namespace pizza_sales_distribution_l261_26180

/-- The total number of pizzas sold in a year -/
def total_pizzas : ℝ := 12.5

/-- The percentage of pizzas sold in summer -/
def summer_percent : ℝ := 0.4

/-- The number of pizzas sold in summer (in millions) -/
def summer_pizzas : ℝ := 5

/-- The percentage of pizzas sold in fall -/
def fall_percent : ℝ := 0.1

/-- The percentage of pizzas sold in winter -/
def winter_percent : ℝ := 0.2

/-- The number of pizzas sold in spring (in millions) -/
def spring_pizzas : ℝ := total_pizzas - (summer_pizzas + fall_percent * total_pizzas + winter_percent * total_pizzas)

theorem pizza_sales_distribution :
  spring_pizzas = 3.75 ∧
  summer_percent * total_pizzas = summer_pizzas ∧
  total_pizzas = summer_pizzas / summer_percent :=
by sorry

end pizza_sales_distribution_l261_26180


namespace max_profit_at_optimal_price_l261_26198

/-- Represents the e-commerce platform's T-shirt sales scenario -/
structure TShirtSales where
  cost : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ
  min_price : ℝ
  max_margin : ℝ

/-- Calculates the profit for a given selling price -/
def profit (s : TShirtSales) (price : ℝ) : ℝ :=
  (price - s.cost) * (s.initial_sales + s.price_sensitivity * (s.initial_price - price))

/-- Theorem stating the maximum profit and optimal price -/
theorem max_profit_at_optimal_price (s : TShirtSales) 
  (h_cost : s.cost = 40)
  (h_initial_price : s.initial_price = 60)
  (h_initial_sales : s.initial_sales = 500)
  (h_price_sensitivity : s.price_sensitivity = 50)
  (h_min_price : s.min_price = s.cost)
  (h_max_margin : s.max_margin = 0.3)
  (h_price_range : ∀ p, s.min_price ≤ p ∧ p ≤ s.cost * (1 + s.max_margin) → 
    profit s p ≤ profit s 52) :
  profit s 52 = 10800 ∧ 
  ∀ p, s.min_price ≤ p ∧ p ≤ s.cost * (1 + s.max_margin) → profit s p ≤ 10800 := by
  sorry


end max_profit_at_optimal_price_l261_26198


namespace geometric_sequence_sum_relation_l261_26145

def geometric_sequence (n : ℕ) : ℝ := 2^(n-1)

def sum_geometric_sequence (n : ℕ) : ℝ := 2^n - 1

theorem geometric_sequence_sum_relation (n : ℕ) :
  sum_geometric_sequence n = 2 * geometric_sequence n - 1 := by
  sorry

end geometric_sequence_sum_relation_l261_26145


namespace collinear_relation_vector_relation_l261_26112

-- Define points A, B, and C
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (3, -1)
def C : ℝ → ℝ → ℝ × ℝ := λ a b => (a, b)

-- Define vector from A to B
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define vector from A to C
def AC (a b : ℝ) : ℝ × ℝ := (a - A.1, b - A.2)

-- Define collinearity condition
def collinear (a b : ℝ) : Prop :=
  ∃ (t : ℝ), AC a b = (t * AB.1, t * AB.2)

-- Theorem 1: If A, B, and C are collinear, then a = 2-b
theorem collinear_relation (a b : ℝ) :
  collinear a b → a = 2 - b := by sorry

-- Theorem 2: If AC = 2AB, then C = (5, -3)
theorem vector_relation :
  ∃ (a b : ℝ), AC a b = (2 * AB.1, 2 * AB.2) ∧ C a b = (5, -3) := by sorry

end collinear_relation_vector_relation_l261_26112


namespace school_population_theorem_l261_26139

theorem school_population_theorem :
  ∀ (boys girls : ℕ),
  boys + girls = 150 →
  girls = (boys * 100) / 150 →
  boys = 90 := by
sorry

end school_population_theorem_l261_26139


namespace nonnegative_solutions_count_l261_26132

theorem nonnegative_solutions_count : 
  ∃! (n : ℕ), ∃ (x : ℝ), x ≥ 0 ∧ x^2 = -6*x ∧ n = 1 := by sorry

end nonnegative_solutions_count_l261_26132


namespace sphere_volume_after_drilling_l261_26123

/-- The remaining volume of a sphere after drilling two cylindrical holes -/
theorem sphere_volume_after_drilling (sphere_diameter : ℝ) (hole1_depth hole1_diameter hole2_depth hole2_diameter : ℝ) : 
  sphere_diameter = 12 ∧ 
  hole1_depth = 5 ∧ 
  hole1_diameter = 1 ∧ 
  hole2_depth = 5 ∧ 
  hole2_diameter = 1.5 → 
  (4 / 3 * π * (sphere_diameter / 2)^3) - (π * (hole1_diameter / 2)^2 * hole1_depth) - (π * (hole2_diameter / 2)^2 * hole2_depth) = 283.9375 * π := by
  sorry

#check sphere_volume_after_drilling

end sphere_volume_after_drilling_l261_26123


namespace largest_band_members_l261_26146

theorem largest_band_members :
  ∀ (m r x : ℕ),
    m < 100 →
    m = r * x + 4 →
    m = (r - 3) * (x + 2) →
    (∀ m' r' x' : ℕ,
      m' < 100 →
      m' = r' * x' + 4 →
      m' = (r' - 3) * (x' + 2) →
      m' ≤ m) →
    m = 88 :=
by sorry

end largest_band_members_l261_26146


namespace snail_distance_bound_l261_26177

/-- Represents the crawling of a snail over time -/
structure SnailCrawl where
  -- The distance function of the snail over time
  distance : ℝ → ℝ
  -- The distance function is non-decreasing (snail doesn't move backward)
  monotone : Monotone distance
  -- The total observation time
  total_time : ℝ
  -- The total time is 6 minutes
  total_time_is_six : total_time = 6

/-- Represents an observation of the snail -/
structure Observation where
  -- Start time of the observation
  start_time : ℝ
  -- Duration of the observation (1 minute)
  duration : ℝ
  duration_is_one : duration = 1
  -- The observation starts within the total time
  start_within_total : start_time ≥ 0 ∧ start_time + duration ≤ 6

/-- The theorem stating that the snail's total distance is at most 10 meters -/
theorem snail_distance_bound (crawl : SnailCrawl) 
  (observations : List Observation) 
  (observed_distance : ∀ obs ∈ observations, 
    crawl.distance (obs.start_time + obs.duration) - crawl.distance obs.start_time = 1) :
  crawl.distance crawl.total_time - crawl.distance 0 ≤ 10 := by
  sorry

end snail_distance_bound_l261_26177


namespace max_a_fourth_quadrant_l261_26140

theorem max_a_fourth_quadrant (a : ℤ) : 
  let z : ℂ := (2 + a * Complex.I) / (1 + 2 * Complex.I)
  (0 < z.re ∧ z.im < 0) → a ≤ 3 ∧ ∃ (b : ℤ), b ≤ 3 ∧ 
    let w : ℂ := (2 + b * Complex.I) / (1 + 2 * Complex.I)
    0 < w.re ∧ w.im < 0 := by
  sorry

end max_a_fourth_quadrant_l261_26140


namespace fraction_simplification_l261_26159

theorem fraction_simplification :
  (4 : ℝ) / (Real.sqrt 108 + 2 * Real.sqrt 12 + 2 * Real.sqrt 27) = Real.sqrt 3 / 12 := by
  sorry

end fraction_simplification_l261_26159


namespace distance_from_origin_l261_26158

theorem distance_from_origin (x y n : ℝ) : 
  x = 8 →
  y > 10 →
  (x - 3)^2 + (y - 10)^2 = 15^2 →
  n^2 = x^2 + y^2 →
  n = Real.sqrt (364 + 200 * Real.sqrt 2) :=
by sorry

end distance_from_origin_l261_26158


namespace buratino_spent_10_dollars_l261_26191

/-- Represents a transaction at the exchange point -/
inductive Transaction
  | type1  -- Give 2 euros, receive 3 dollars and a candy
  | type2  -- Give 5 dollars, receive 3 euros and a candy

/-- Represents Buratino's exchange activities -/
structure ExchangeActivity where
  transactions : List Transaction
  initialDollars : ℕ
  finalDollars : ℕ
  finalEuros : ℕ
  candiesReceived : ℕ

/-- Calculates the net dollar change for a given transaction -/
def netDollarChange (t : Transaction) : ℤ :=
  match t with
  | Transaction.type1 => 3
  | Transaction.type2 => -5

/-- Calculates the net euro change for a given transaction -/
def netEuroChange (t : Transaction) : ℤ :=
  match t with
  | Transaction.type1 => -2
  | Transaction.type2 => 3

/-- Theorem stating that Buratino spent 10 dollars -/
theorem buratino_spent_10_dollars (activity : ExchangeActivity) :
  activity.candiesReceived = 50 ∧
  activity.finalEuros = 0 ∧
  activity.finalDollars < activity.initialDollars →
  activity.initialDollars - activity.finalDollars = 10 := by
  sorry


end buratino_spent_10_dollars_l261_26191


namespace l_shape_area_is_52_l261_26121

/-- The area of an 'L' shaped figure formed from a rectangle with given dimensions,
    after subtracting a corner rectangle and an inner rectangle. -/
def l_shape_area (large_length large_width corner_length corner_width inner_length inner_width : ℕ) : ℕ :=
  large_length * large_width - (corner_length * corner_width + inner_length * inner_width)

/-- Theorem stating that the area of the specific 'L' shaped figure is 52 square units. -/
theorem l_shape_area_is_52 :
  l_shape_area 10 6 3 2 2 1 = 52 := by
  sorry

end l_shape_area_is_52_l261_26121


namespace min_value_expression_l261_26142

theorem min_value_expression (x : ℝ) : 
  (12 - x) * (10 - x) * (12 + x) * (10 + x) ≥ -484 ∧ 
  ∃ y : ℝ, (12 - y) * (10 - y) * (12 + y) * (10 + y) = -484 := by
  sorry

end min_value_expression_l261_26142


namespace prime_digits_imply_prime_count_l261_26110

theorem prime_digits_imply_prime_count (n : ℕ) (x : ℕ) : 
  (x = (10^n - 1) / 9) →  -- x is an integer with n digits, all equal to 1
  Nat.Prime x →           -- x is prime
  Nat.Prime n :=          -- n is prime
by sorry

end prime_digits_imply_prime_count_l261_26110


namespace division_remainder_problem_l261_26120

theorem division_remainder_problem (D : ℕ) : 
  D = 12 * 63 + (D % 12) →  -- Incorrect division equation
  D = 21 * 36 + (D % 21) →  -- Correct division equation
  D % 21 = 0 :=             -- Remainder of correct division is 0
by sorry

end division_remainder_problem_l261_26120


namespace choose_four_from_seven_l261_26190

theorem choose_four_from_seven : Nat.choose 7 4 = 35 := by sorry

end choose_four_from_seven_l261_26190


namespace means_inequality_l261_26106

theorem means_inequality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^2 + b^2) / 2 > (a + b) / 2 ∧ (a + b) / 2 > Real.sqrt (a * b) := by
  sorry

end means_inequality_l261_26106


namespace rachel_reading_homework_l261_26196

/-- The number of pages of reading homework Rachel had to complete -/
def reading_pages : ℕ := by sorry

/-- The number of pages of math homework Rachel had to complete -/
def math_pages : ℕ := 4

/-- The relationship between math and reading homework pages -/
axiom math_reading_relation : math_pages = reading_pages + 2

theorem rachel_reading_homework : reading_pages = 2 := by sorry

end rachel_reading_homework_l261_26196


namespace parallel_vectors_x_value_l261_26164

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (x, 1)
  are_parallel a b → x = -1/3 := by
  sorry

end parallel_vectors_x_value_l261_26164


namespace toms_fruit_purchase_l261_26151

/-- The problem of Tom's fruit purchase -/
theorem toms_fruit_purchase 
  (apple_kg : ℕ) 
  (apple_rate : ℕ) 
  (mango_rate : ℕ) 
  (total_paid : ℕ) 
  (h1 : apple_kg = 8)
  (h2 : apple_rate = 70)
  (h3 : mango_rate = 75)
  (h4 : total_paid = 1235)
  : ∃ (mango_kg : ℕ), 
    apple_kg * apple_rate + mango_kg * mango_rate = total_paid ∧ 
    mango_kg = 9 := by
  sorry

end toms_fruit_purchase_l261_26151


namespace triangle_angle_from_side_relation_l261_26197

theorem triangle_angle_from_side_relation (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  0 < a ∧ 0 < b ∧ 0 < c →
  Real.sqrt 2 * a = 2 * b * Real.sin A →
  B = π / 4 ∨ B = 3 * π / 4 := by
  sorry

end triangle_angle_from_side_relation_l261_26197


namespace same_color_prob_eq_half_l261_26103

/-- The probability of drawing two balls of the same color from an urn -/
def same_color_prob (n : ℕ) : ℚ :=
  (1 / (n + 5))^2 + (4 / (n + 5))^2 + (n / (n + 5))^2

/-- Theorem: The probability of drawing two balls of the same color is 1/2 iff n = 1 or n = 9 -/
theorem same_color_prob_eq_half (n : ℕ) :
  same_color_prob n = 1/2 ↔ n = 1 ∨ n = 9 := by
  sorry

#eval same_color_prob 1  -- Should output 1/2
#eval same_color_prob 9  -- Should output 1/2

end same_color_prob_eq_half_l261_26103


namespace quadratic_sum_l261_26128

/-- A quadratic function f(x) = ax^2 + bx + c with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := fun x ↦ (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

/-- The theorem stating that for a quadratic function with given properties, a + b - c = -7 -/
theorem quadratic_sum (a b c : ℤ) :
  let f := QuadraticFunction a b c
  (f 2 = 5) →  -- The graph passes through (2, 5)
  (∀ x, f x ≥ f 1) →  -- The vertex is at x = 1
  (f 1 = 3) →  -- The y-coordinate of the vertex is 3
  a + b - c = -7 :=
by
  sorry

end quadratic_sum_l261_26128


namespace china_gdp_scientific_notation_l261_26186

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem china_gdp_scientific_notation :
  toScientificNotation 86000 = ScientificNotation.mk 8.6 4 sorry := by
  sorry

end china_gdp_scientific_notation_l261_26186


namespace price_reduction_equation_l261_26147

/-- Theorem: For an item with an original price of 289 yuan and a final price of 256 yuan
    after two consecutive price reductions, where x represents the average percentage
    reduction each time, the equation 289(1-x)^2 = 256 holds true. -/
theorem price_reduction_equation (x : ℝ) 
  (h1 : 0 ≤ x) (h2 : x < 1) : 289 * (1 - x)^2 = 256 := by
  sorry

#check price_reduction_equation

end price_reduction_equation_l261_26147


namespace clock_strikes_in_day_l261_26122

def clock_strikes (hour : Nat) : Nat :=
  if hour ≤ 12 then hour else hour - 12

def total_strikes : Nat :=
  (List.range 24).map clock_strikes |> List.sum

theorem clock_strikes_in_day : total_strikes = 156 := by
  sorry

end clock_strikes_in_day_l261_26122


namespace sony_johnny_fish_ratio_l261_26148

def total_fishes : ℕ := 40
def johnny_fishes : ℕ := 8

theorem sony_johnny_fish_ratio :
  (total_fishes - johnny_fishes) / johnny_fishes = 4 := by
  sorry

end sony_johnny_fish_ratio_l261_26148


namespace sum_of_fractions_l261_26174

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end sum_of_fractions_l261_26174


namespace sum_of_roots_eq_six_l261_26156

theorem sum_of_roots_eq_six : 
  let f : ℝ → ℝ := λ x => (x - 3)^2 - 16
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 6 := by
  sorry

end sum_of_roots_eq_six_l261_26156


namespace min_value_of_parallel_lines_l261_26116

theorem min_value_of_parallel_lines (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_parallel : a * (b - 3) - 2 * b = 0) : 
  (∀ x y : ℝ, 2 * a + 3 * b ≥ 25) ∧ (∃ x y : ℝ, 2 * a + 3 * b = 25) :=
sorry

end min_value_of_parallel_lines_l261_26116


namespace can_space_before_compacting_l261_26143

theorem can_space_before_compacting :
  ∀ (n : ℕ) (total_space : ℝ) (compaction_ratio : ℝ),
    n = 60 →
    compaction_ratio = 0.2 →
    total_space = 360 →
    (n : ℝ) * compaction_ratio * (360 / (n * compaction_ratio)) = total_space →
    360 / (n * compaction_ratio) = 30 := by
  sorry

end can_space_before_compacting_l261_26143


namespace silas_payment_ratio_l261_26166

theorem silas_payment_ratio (total_bill : ℚ) (friend_payment : ℚ) 
  (h1 : total_bill = 150)
  (h2 : friend_payment = 18)
  (h3 : (5 : ℚ) * friend_payment + (total_bill / 10) = total_bill + (total_bill / 10) - (total_bill / 2)) :
  (total_bill / 2) / total_bill = 1 / 2 := by
  sorry

end silas_payment_ratio_l261_26166


namespace discounted_subscription_cost_l261_26183

/-- The discounted subscription cost problem -/
theorem discounted_subscription_cost
  (normal_cost : ℝ)
  (discount_percentage : ℝ)
  (h_normal_cost : normal_cost = 80)
  (h_discount : discount_percentage = 45) :
  normal_cost * (1 - discount_percentage / 100) = 44 :=
by sorry

end discounted_subscription_cost_l261_26183


namespace stating_hotel_booking_problem_l261_26187

/-- Represents the number of double rooms booked in a hotel. -/
def double_rooms : ℕ := 196

/-- Represents the number of single rooms booked in a hotel. -/
def single_rooms : ℕ := 260 - double_rooms

/-- The cost of a single room in dollars. -/
def single_room_cost : ℕ := 35

/-- The cost of a double room in dollars. -/
def double_room_cost : ℕ := 60

/-- The total revenue from all booked rooms in dollars. -/
def total_revenue : ℕ := 14000

/-- 
Theorem stating that given the conditions of the hotel booking problem,
the number of double rooms booked is 196.
-/
theorem hotel_booking_problem :
  (single_rooms + double_rooms = 260) ∧
  (single_room_cost * single_rooms + double_room_cost * double_rooms = total_revenue) →
  double_rooms = 196 :=
by sorry

end stating_hotel_booking_problem_l261_26187


namespace flagpole_break_height_approx_l261_26167

/-- The height of the flagpole in meters -/
def flagpole_height : ℝ := 5

/-- The distance from the base of the flagpole to where the broken part touches the ground, in meters -/
def ground_distance : ℝ := 1

/-- The approximate height where the flagpole breaks, in meters -/
def break_height : ℝ := 2.4

/-- Theorem stating that the break height is approximately correct -/
theorem flagpole_break_height_approx :
  let total_height := flagpole_height
  let distance := ground_distance
  let break_point := break_height
  abs (break_point - (total_height * distance / (2 * total_height))) < 0.1 := by
  sorry


end flagpole_break_height_approx_l261_26167


namespace job_completion_time_l261_26100

/-- The time it takes for Annie to complete the job alone -/
def annie_time : ℝ := 9

/-- The time the person works before stopping -/
def person_partial_time : ℝ := 4

/-- The time it takes Annie to complete the remaining work after the person stops -/
def annie_completion_time : ℝ := 6

/-- The time it takes for the person to complete the job alone -/
def person_total_time : ℝ := 12

theorem job_completion_time :
  (person_partial_time / person_total_time) + (annie_completion_time / annie_time) = 1 :=
sorry

end job_completion_time_l261_26100


namespace distance_center_to_point_l261_26165

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 6*x + 10*y + 9

-- Define the center of the circle
def circle_center : ℝ × ℝ :=
  let a := 3
  let b := 5
  (a, b)

-- Define the given point
def given_point : ℝ × ℝ := (-4, -2)

-- Theorem statement
theorem distance_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := given_point
  Real.sqrt ((cx - px)^2 + (cy - py)^2) = 7 * Real.sqrt 2 :=
by sorry

end distance_center_to_point_l261_26165


namespace train_speed_l261_26176

/-- Proves that a train with given specifications travels at 45 km/hr -/
theorem train_speed (train_length : Real) (crossing_time : Real) (total_length : Real) :
  train_length = 130 ∧ 
  crossing_time = 30 ∧ 
  total_length = 245 → 
  (total_length - train_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed

end train_speed_l261_26176


namespace sum_of_fourth_powers_less_than_150_l261_26171

theorem sum_of_fourth_powers_less_than_150 : 
  (Finset.filter (fun n : ℕ => n^4 < 150) (Finset.range 150)).sum id = 98 := by
  sorry

end sum_of_fourth_powers_less_than_150_l261_26171


namespace average_age_of_group_l261_26127

/-- The average age of a group of seventh-graders and their guardians -/
def average_age (num_students : ℕ) (student_avg_age : ℚ) (num_guardians : ℕ) (guardian_avg_age : ℚ) : ℚ :=
  ((num_students : ℚ) * student_avg_age + (num_guardians : ℚ) * guardian_avg_age) / ((num_students + num_guardians) : ℚ)

/-- Theorem stating that the average age of 40 seventh-graders (average age 13) and 60 guardians (average age 40) is 29.2 -/
theorem average_age_of_group : average_age 40 13 60 40 = 29.2 := by
  sorry

end average_age_of_group_l261_26127


namespace cube_with_tunnel_surface_area_l261_26170

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a tunnel drilled through it -/
structure CubeWithTunnel where
  /-- Length of the cube's edge -/
  edgeLength : ℝ
  /-- Point P, a corner of the cube -/
  p : Point3D
  /-- Point L on PQ -/
  l : Point3D
  /-- Point M on PR -/
  m : Point3D
  /-- Point N on PC -/
  n : Point3D

/-- The surface area of the cube with tunnel can be expressed as x + y√z -/
def surfaceAreaExpression (c : CubeWithTunnel) : ℕ × ℕ × ℕ :=
  sorry

theorem cube_with_tunnel_surface_area 
  (c : CubeWithTunnel)
  (h1 : c.edgeLength = 10)
  (h2 : c.p.x = 10 ∧ c.p.y = 10 ∧ c.p.z = 10)
  (h3 : c.l.x = 7.5 ∧ c.l.y = 10 ∧ c.l.z = 10)
  (h4 : c.m.x = 10 ∧ c.m.y = 7.5 ∧ c.m.z = 10)
  (h5 : c.n.x = 10 ∧ c.n.y = 10 ∧ c.n.z = 7.5) :
  let (x, y, z) := surfaceAreaExpression c
  x + y + z = 639 ∧ 
  (∀ p : ℕ, Prime p → ¬(p^2 ∣ z)) :=
sorry

end cube_with_tunnel_surface_area_l261_26170


namespace whirling_wonderland_capacity_l261_26181

/-- The 'Whirling Wonderland' ride problem -/
theorem whirling_wonderland_capacity :
  let people_per_carriage : ℕ := 12
  let number_of_carriages : ℕ := 15
  let total_capacity : ℕ := people_per_carriage * number_of_carriages
  total_capacity = 180 := by
  sorry

end whirling_wonderland_capacity_l261_26181


namespace subset_sum_indivisibility_implies_equality_l261_26149

theorem subset_sum_indivisibility_implies_equality (m : ℕ) (a : Fin m → ℕ) :
  (∀ i, a i ∈ Finset.range m) →
  (∀ s : Finset (Fin m), (s.sum a) % (m + 1) ≠ 0) →
  ∀ i j, a i = a j :=
sorry

end subset_sum_indivisibility_implies_equality_l261_26149


namespace combined_average_l261_26189

/-- Given two sets of results, one with 80 results averaging 32 and another with 50 results averaging 56,
    prove that the average of all results combined is (80 * 32 + 50 * 56) / (80 + 50) -/
theorem combined_average (set1_count : Nat) (set1_avg : ℚ) (set2_count : Nat) (set2_avg : ℚ)
    (h1 : set1_count = 80)
    (h2 : set1_avg = 32)
    (h3 : set2_count = 50)
    (h4 : set2_avg = 56) :
  (set1_count * set1_avg + set2_count * set2_avg) / (set1_count + set2_count) =
    (80 * 32 + 50 * 56) / (80 + 50) := by
  sorry

end combined_average_l261_26189


namespace complement_intersection_equality_l261_26172

def I : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {3, 4, 5}

theorem complement_intersection_equality :
  (I \ N) ∩ M = {1, 2} := by sorry

end complement_intersection_equality_l261_26172


namespace arithmetic_sequence_problem_l261_26105

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 70 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 7 + a 8 + a 9 + a 14 = 70

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : sum_condition a) : 
  a 8 = 14 := by
  sorry

end arithmetic_sequence_problem_l261_26105


namespace baker_cookies_total_l261_26161

theorem baker_cookies_total (chocolate_chip_batches oatmeal_batches : ℕ)
  (chocolate_chip_per_batch oatmeal_per_batch : ℕ)
  (sugar_cookies double_chocolate_cookies : ℕ) :
  chocolate_chip_batches = 5 →
  oatmeal_batches = 3 →
  chocolate_chip_per_batch = 8 →
  oatmeal_per_batch = 7 →
  sugar_cookies = 10 →
  double_chocolate_cookies = 6 →
  chocolate_chip_batches * chocolate_chip_per_batch +
  oatmeal_batches * oatmeal_per_batch +
  sugar_cookies + double_chocolate_cookies = 77 :=
by sorry

end baker_cookies_total_l261_26161


namespace hyperbola_equation_l261_26154

theorem hyperbola_equation (a b c : ℝ) (h1 : c = 4 * Real.sqrt 3) (h2 : a = 1) (h3 : b^2 = c^2 - a^2) :
  ∀ x y : ℝ, x^2 - y^2 / 47 = 1 ↔ x^2 - y^2 / b^2 = 1 :=
sorry

end hyperbola_equation_l261_26154


namespace expression_simplification_l261_26133

theorem expression_simplification : 
  ((3 + 4 + 5 + 6 + 7) / 3) + ((3 * 6 + 12) / 4) = 95 / 6 := by
sorry

end expression_simplification_l261_26133


namespace arithmetic_sequence_property_l261_26136

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 := by
  sorry

end arithmetic_sequence_property_l261_26136


namespace number_thought_of_l261_26194

theorem number_thought_of : ∃ x : ℝ, (x / 6) + 5 = 17 := by
  sorry

end number_thought_of_l261_26194


namespace list_number_relation_l261_26137

theorem list_number_relation (n : ℝ) (list : List ℝ) : 
  list.length = 21 ∧ 
  n ∈ list ∧
  n = (1 / 6 : ℝ) * list.sum →
  n = 4 * ((list.sum - n) / 20) := by
sorry

end list_number_relation_l261_26137


namespace dollar_four_neg_one_l261_26168

-- Define the $ operation
def dollar (x y : ℤ) : ℤ := x * (y + 2) + 2 * x * y

-- Theorem statement
theorem dollar_four_neg_one : dollar 4 (-1) = -4 := by
  sorry

end dollar_four_neg_one_l261_26168


namespace train_speed_problem_l261_26188

/-- Represents a train with its speed and travel time after meeting another train -/
structure Train where
  speed : ℝ
  time_after_meeting : ℝ

/-- Proves that given the conditions of the problem, the speed of train B is 225 km/h -/
theorem train_speed_problem (train_A train_B : Train) 
  (h1 : train_A.speed = 100)
  (h2 : train_A.time_after_meeting = 9)
  (h3 : train_B.time_after_meeting = 4)
  (h4 : train_A.speed * train_A.time_after_meeting = train_B.speed * train_B.time_after_meeting) :
  train_B.speed = 225 := by
  sorry

#check train_speed_problem

end train_speed_problem_l261_26188


namespace final_postcard_count_l261_26153

-- Define the exchange rates
def euro_to_usd : ℚ := 1.20
def gbp_to_usd : ℚ := 1.35
def usd_to_yen : ℚ := 110

-- Define the initial number of postcards and sales
def initial_postcards : ℕ := 18
def sold_euro : ℕ := 6
def sold_gbp : ℕ := 3
def sold_usd : ℕ := 2

-- Define the prices of sold postcards
def price_euro : ℚ := 10
def price_gbp : ℚ := 12
def price_usd : ℚ := 15

-- Define the price of new postcards in USD
def new_postcard_price_usd : ℚ := 8

-- Define the price of additional postcards in Yen
def additional_postcard_price_yen : ℚ := 800

-- Define the percentage of earnings used to buy new postcards
def percentage_for_new_postcards : ℚ := 0.70

-- Define the number of additional postcards bought
def additional_postcards : ℕ := 5

-- Theorem statement
theorem final_postcard_count :
  let total_earnings_usd := sold_euro * price_euro * euro_to_usd + 
                            sold_gbp * price_gbp * gbp_to_usd + 
                            sold_usd * price_usd
  let new_postcards := (total_earnings_usd * percentage_for_new_postcards / new_postcard_price_usd).floor
  let remaining_usd := total_earnings_usd - new_postcards * new_postcard_price_usd
  let additional_postcards_bought := (remaining_usd * usd_to_yen / additional_postcard_price_yen).floor
  initial_postcards - (sold_euro + sold_gbp + sold_usd) + new_postcards + additional_postcards_bought = 26 :=
by sorry

end final_postcard_count_l261_26153


namespace polynomial_equality_l261_26111

theorem polynomial_equality (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 4 = (x + 2)^2) → a = 4 := by
  sorry

end polynomial_equality_l261_26111


namespace michael_truck_meetings_l261_26118

/-- Represents the problem of Michael and the garbage truck --/
structure GarbageTruckProblem where
  michael_speed : ℝ
  michael_delay : ℝ
  pail_spacing : ℝ
  truck_speed : ℝ
  truck_stop_duration : ℝ
  initial_distance : ℝ

/-- Calculates the number of times Michael and the truck meet --/
def number_of_meetings (problem : GarbageTruckProblem) : ℕ :=
  sorry

/-- The specific problem instance --/
def our_problem : GarbageTruckProblem :=
  { michael_speed := 3
  , michael_delay := 20
  , pail_spacing := 300
  , truck_speed := 12
  , truck_stop_duration := 45
  , initial_distance := 300 }

/-- Theorem stating that Michael and the truck meet exactly 6 times --/
theorem michael_truck_meetings :
  number_of_meetings our_problem = 6 := by
  sorry

end michael_truck_meetings_l261_26118


namespace kevin_kangaroo_four_hops_l261_26173

def hop_distance (remaining : ℚ) : ℚ := (1 / 4) * remaining

def total_distance (n : ℕ) : ℚ :=
  let goal := 2
  let rec distance_after_hops (k : ℕ) (remaining : ℚ) (acc : ℚ) : ℚ :=
    if k = 0 then acc
    else
      let hop := hop_distance remaining
      distance_after_hops (k - 1) (remaining - hop) (acc + hop)
  distance_after_hops n goal 0

theorem kevin_kangaroo_four_hops :
  total_distance 4 = 175 / 128 := by sorry

end kevin_kangaroo_four_hops_l261_26173


namespace unique_twin_prime_sum_prime_power_l261_26114

-- Define twin primes
def is_twin_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime (p + 2)

-- Define prime power
def is_prime_power (n : ℕ) : Prop :=
  ∃ (q k : ℕ), Nat.Prime q ∧ k > 0 ∧ n = q^k

-- Theorem statement
theorem unique_twin_prime_sum_prime_power :
  ∃! (p : ℕ), is_twin_prime p ∧ is_prime_power (p + (p + 2)) :=
sorry

end unique_twin_prime_sum_prime_power_l261_26114


namespace unique_prime_with_remainder_l261_26169

theorem unique_prime_with_remainder : ∃! p : ℕ, 
  Prime p ∧ 
  20 < p ∧ p < 35 ∧ 
  p % 11 = 7 :=
by
  -- The proof goes here
  sorry

end unique_prime_with_remainder_l261_26169


namespace sum_of_digits_of_seven_to_eleven_l261_26129

/-- The sum of the tens digit and the ones digit of (3+4)^11 is 7 -/
theorem sum_of_digits_of_seven_to_eleven : 
  let n : ℕ := (3 + 4)^11
  let ones_digit : ℕ := n % 10
  let tens_digit : ℕ := (n / 10) % 10
  ones_digit + tens_digit = 7 := by sorry

end sum_of_digits_of_seven_to_eleven_l261_26129


namespace seven_non_drinkers_l261_26192

/-- Represents the number of businessmen who drank a specific beverage or combination of beverages -/
structure BeverageCounts where
  total : Nat
  coffee : Nat
  tea : Nat
  water : Nat
  coffeeAndTea : Nat
  teaAndWater : Nat
  coffeeAndWater : Nat
  allThree : Nat

/-- Calculates the number of businessmen who drank none of the beverages -/
def nonDrinkers (counts : BeverageCounts) : Nat :=
  counts.total - (counts.coffee + counts.tea + counts.water
                  - counts.coffeeAndTea - counts.teaAndWater - counts.coffeeAndWater
                  + counts.allThree)

/-- Theorem stating that given the conditions, 7 businessmen drank none of the beverages -/
theorem seven_non_drinkers (counts : BeverageCounts)
  (h1 : counts.total = 30)
  (h2 : counts.coffee = 15)
  (h3 : counts.tea = 13)
  (h4 : counts.water = 6)
  (h5 : counts.coffeeAndTea = 7)
  (h6 : counts.teaAndWater = 3)
  (h7 : counts.coffeeAndWater = 2)
  (h8 : counts.allThree = 1) :
  nonDrinkers counts = 7 := by
  sorry

#eval nonDrinkers { total := 30, coffee := 15, tea := 13, water := 6,
                    coffeeAndTea := 7, teaAndWater := 3, coffeeAndWater := 2, allThree := 1 }

end seven_non_drinkers_l261_26192


namespace thread_length_ratio_l261_26144

theorem thread_length_ratio : 
  let original_length : ℚ := 12
  let total_required : ℚ := 21
  let additional_length := total_required - original_length
  additional_length / original_length = 3 / 4 := by
  sorry

end thread_length_ratio_l261_26144


namespace negation_equivalence_l261_26125

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) := by
  sorry

end negation_equivalence_l261_26125


namespace ellipse_line_intersection_range_l261_26134

/-- Given that for all k ∈ ℝ, the line y - kx - 1 = 0 always intersects 
    with the ellipse x²/4 + y²/m = 1, prove that the range of m is [1, 4) ∪ (4, +∞) -/
theorem ellipse_line_intersection_range (m : ℝ) : 
  (∀ k : ℝ, ∃ x y : ℝ, y - k*x - 1 = 0 ∧ x^2/4 + y^2/m = 1) ↔ 
  (m ∈ Set.Icc 1 4 ∪ Set.Ioi 4) :=
sorry

end ellipse_line_intersection_range_l261_26134


namespace equal_pair_proof_l261_26130

theorem equal_pair_proof : (-4)^3 = -4^3 := by
  sorry

end equal_pair_proof_l261_26130


namespace smallest_number_drawn_l261_26157

/-- Represents a systematic sampling of classes -/
structure ClassSampling where
  total_classes : ℕ
  sample_size : ℕ
  sum_of_selected : ℕ

/-- Theorem: If we have 18 classes, sample 6 of them systematically, 
    and the sum of selected numbers is 57, then the smallest number drawn is 2 -/
theorem smallest_number_drawn (s : ClassSampling) 
  (h1 : s.total_classes = 18)
  (h2 : s.sample_size = 6)
  (h3 : s.sum_of_selected = 57) :
  ∃ x : ℕ, x = 2 ∧ 
    (x + (x + 3) + (x + 6) + (x + 9) + (x + 12) + (x + 15) = s.sum_of_selected) :=
sorry

end smallest_number_drawn_l261_26157


namespace kenya_peanuts_l261_26117

theorem kenya_peanuts (jose_peanuts : ℕ) (kenya_extra : ℕ) : 
  jose_peanuts = 85 → kenya_extra = 48 → jose_peanuts + kenya_extra = 133 := by
  sorry

end kenya_peanuts_l261_26117


namespace triangle_side_angle_ratio_l261_26179

theorem triangle_side_angle_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  b^2 = a * c →
  a^2 - c^2 = a * c - b * c →
  c / (b * Real.sin B) = 2 * Real.sqrt 3 / 3 := by
  sorry

end triangle_side_angle_ratio_l261_26179


namespace slower_time_to_top_l261_26104

/-- The time taken by the slower of two people to reach the top of a building --/
def time_to_top (stories : ℕ) (run_time : ℕ) (elevator_time : ℕ) (stop_time : ℕ) : ℕ :=
  max
    (stories * run_time)
    (stories * elevator_time + (stories - 1) * stop_time)

/-- Theorem stating that the slower person takes 217 seconds to reach the top floor --/
theorem slower_time_to_top :
  time_to_top 20 10 8 3 = 217 := by
  sorry

end slower_time_to_top_l261_26104


namespace volume_for_56_ounces_l261_26199

/-- A substance with volume directly proportional to weight -/
structure Substance where
  /-- Constant of proportionality between volume and weight -/
  k : ℚ
  /-- Assumption that k is positive -/
  k_pos : k > 0

/-- The volume of the substance given its weight -/
def volume (s : Substance) (weight : ℚ) : ℚ :=
  s.k * weight

theorem volume_for_56_ounces (s : Substance) 
  (h : volume s 112 = 48) : volume s 56 = 24 := by
  sorry

#check volume_for_56_ounces

end volume_for_56_ounces_l261_26199


namespace tan_ratio_from_sin_sum_diff_l261_26102

theorem tan_ratio_from_sin_sum_diff (a b : ℝ) 
  (h1 : Real.sin (a + b) = 5/8) 
  (h2 : Real.sin (a - b) = 1/4) : 
  Real.tan a / Real.tan b = 7/3 := by
  sorry

end tan_ratio_from_sin_sum_diff_l261_26102


namespace baseball_cards_distribution_l261_26107

theorem baseball_cards_distribution (total_cards : ℕ) (num_friends : ℕ) (cards_per_friend : ℕ) :
  total_cards = 24 →
  num_friends = 4 →
  total_cards = num_friends * cards_per_friend →
  cards_per_friend = 6 :=
by
  sorry

end baseball_cards_distribution_l261_26107


namespace binary_1011_equals_11_l261_26152

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1011_equals_11 :
  binary_to_decimal [true, true, false, true] = 11 := by
  sorry

end binary_1011_equals_11_l261_26152


namespace olivia_dvd_count_l261_26150

theorem olivia_dvd_count (dvds_per_season : ℕ) (seasons_bought : ℕ) : 
  dvds_per_season = 8 → seasons_bought = 5 → dvds_per_season * seasons_bought = 40 :=
by sorry

end olivia_dvd_count_l261_26150


namespace pilot_tuesday_miles_l261_26163

/-- 
Given a pilot's flight schedule where:
- The pilot flies x miles on Tuesday and 1475 miles on Thursday in 1 week
- This pattern is repeated for 3 weeks
- The total miles flown in 3 weeks is 7827 miles

Prove that the number of miles flown on Tuesday (x) is 1134.
-/
theorem pilot_tuesday_miles : 
  ∀ x : ℕ, 
  (3 * (x + 1475) = 7827) → 
  x = 1134 := by
sorry

end pilot_tuesday_miles_l261_26163


namespace intersection_area_l261_26162

theorem intersection_area (f g : ℝ → ℝ) (P Q : ℝ × ℝ) (B A : ℝ × ℝ) :
  (f = λ x => 2 * Real.cos (3 * x) + 1) →
  (g = λ x => - Real.cos (2 * x)) →
  (∃ x₁ x₂, 17 * π / 4 < x₁ ∧ x₁ < 21 * π / 4 ∧
            17 * π / 4 < x₂ ∧ x₂ < 21 * π / 4 ∧
            P = (x₁, f x₁) ∧ Q = (x₂, f x₂) ∧
            f x₁ = g x₁ ∧ f x₂ = g x₂) →
  (∃ m b, ∀ x, P.2 + m * (x - P.1) = b * x) →
  B.2 = 0 →
  A.1 = 0 →
  (area_triangle : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → ℝ) →
  area_triangle (0, 0) B A = 361 * π / 8 := by
sorry

end intersection_area_l261_26162


namespace original_mixture_volume_l261_26119

theorem original_mixture_volume 
  (original_alcohol_percentage : ℝ)
  (added_water : ℝ)
  (new_alcohol_percentage : ℝ)
  (h1 : original_alcohol_percentage = 0.25)
  (h2 : added_water = 3)
  (h3 : new_alcohol_percentage = 0.20833333333333336)
  : ∃ (original_volume : ℝ),
    original_volume * original_alcohol_percentage / (original_volume + added_water) = new_alcohol_percentage ∧
    original_volume = 15 :=
by sorry

end original_mixture_volume_l261_26119


namespace four_m₀_is_sum_of_three_or_four_primes_l261_26155

-- Define the existence of a prime between n and 2n for any positive integer n
axiom exists_prime_between (n : ℕ) (hn : 0 < n) : ∃ p : ℕ, Prime p ∧ n ≤ p ∧ p ≤ 2 * n

-- Define the smallest even number greater than 2 that can't be expressed as sum of two primes
axiom exists_smallest_non_goldbach : ∃ m₀ : ℕ, 1 < m₀ ∧ 
  (∀ k < m₀, ∃ p q : ℕ, Prime p ∧ Prime q ∧ 2 * k = p + q) ∧
  (¬∃ p q : ℕ, Prime p ∧ Prime q ∧ 2 * m₀ = p + q)

-- Theorem statement
theorem four_m₀_is_sum_of_three_or_four_primes :
  ∃ m₀ : ℕ, 1 < m₀ ∧ 
  (∀ k < m₀, ∃ p q : ℕ, Prime p ∧ Prime q ∧ 2 * k = p + q) ∧
  (¬∃ p q : ℕ, Prime p ∧ Prime q ∧ 2 * m₀ = p + q) →
  ∃ p₁ p₂ p₃ p₄ : ℕ, (Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ 4 * m₀ = p₁ + p₂ + p₃) ∨
                     (Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 4 * m₀ = p₁ + p₂ + p₃ + p₄) :=
by sorry

end four_m₀_is_sum_of_three_or_four_primes_l261_26155


namespace solution_set_quadratic_inequality_l261_26160

theorem solution_set_quadratic_inequality :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 4 * x - 7
  {x : ℝ | f x < 0} = {x : ℝ | -1 < x ∧ x < 7/3} := by
sorry

end solution_set_quadratic_inequality_l261_26160


namespace equation_solutions_l261_26178

theorem equation_solutions :
  (∀ x : ℝ, 9 * (x - 1)^2 = 25 ↔ x = 8/3 ∨ x = -2/3) ∧
  (∀ x : ℝ, (1/3) * (x + 2)^3 - 9 = 0 ↔ x = 1) :=
by sorry

end equation_solutions_l261_26178


namespace seating_arrangements_l261_26101

/-- The number of ways to arrange n people in k seats -/
def arrange (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose n items from k items -/
def choose (n k : ℕ) : ℕ := sorry

theorem seating_arrangements : 
  let total_seats : ℕ := 6
  let people : ℕ := 3
  let all_arrangements := arrange people total_seats
  let no_adjacent_empty := choose (total_seats - people + 1) people * arrange people people
  let all_empty_adjacent := choose (total_seats - people + 1) 1 * arrange people people
  all_arrangements - no_adjacent_empty - all_empty_adjacent = 72 := by sorry

end seating_arrangements_l261_26101


namespace hyperbola_asymptotes_l261_26138

/-- The focus of a parabola y² = 12x -/
def parabola_focus : ℝ × ℝ := (3, 0)

/-- The equation of a hyperbola -/
def is_hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 = 1

/-- The equation of asymptotes of a hyperbola -/
def is_asymptote (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x ∨ y = -k * x

/-- Main theorem -/
theorem hyperbola_asymptotes :
  ∃ (a : ℝ), (is_hyperbola a (parabola_focus.1) (parabola_focus.2)) →
  (∀ (x y : ℝ), is_asymptote (1/3) x y ↔ is_hyperbola a x y) :=
sorry

end hyperbola_asymptotes_l261_26138


namespace range_of_a_l261_26108

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x, x^2 - 2*x + a > 0

def q (a : ℝ) : Prop := ∀ x y, x < y → (a - 1)^x < (a - 1)^y

-- Define the theorem
theorem range_of_a :
  (∃ a, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∃ a, 1 < a ∧ a ≤ 2) ∧ (∀ a, (1 < a ∧ a ≤ 2) → (p a ∨ q a) ∧ ¬(p a ∧ q a)) :=
sorry

end range_of_a_l261_26108


namespace inverse_function_property_l261_26115

-- Define a function f and its inverse f_inv
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- Define the property of f and f_inv being inverse functions
def are_inverse (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- State the theorem
theorem inverse_function_property
  (h1 : are_inverse f f_inv)
  (h2 : f 2 = -1) :
  f_inv (-1) = 2 :=
sorry

end inverse_function_property_l261_26115


namespace no_natural_function_satisfies_equation_l261_26131

theorem no_natural_function_satisfies_equation :
  ¬ ∃ (f : ℕ → ℕ), ∀ (x : ℕ), f (f x) = x + 1 := by
  sorry

end no_natural_function_satisfies_equation_l261_26131


namespace job_completion_time_l261_26185

/-- Proves that if A and D together can complete a job in 5 hours, and D alone can complete
    the job in 10 hours, then A alone can complete the job in 10 hours. -/
theorem job_completion_time (A D : ℝ) (hAD : 1 / A + 1 / D = 1 / 5) (hD : D = 10) : A = 10 := by
  sorry

end job_completion_time_l261_26185


namespace horner_third_step_equals_12_l261_26113

def f (x : ℝ) : ℝ := 2*x^5 - 3*x^3 + 2*x^2 + x - 3

def horner_step (a : ℝ) (x : ℝ) (prev : ℝ) : ℝ := prev * x + a

def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (horner_step · x) 0

theorem horner_third_step_equals_12 :
  let coeffs := [2, 0, -3, 2, 1, -3]
  let x := 2
  let v3 := (horner_method (coeffs.take 4) x)
  v3 = 12 := by sorry

end horner_third_step_equals_12_l261_26113


namespace f_of_g_of_3_equals_29_l261_26124

def f (x : ℝ) : ℝ := 3 * x - 4

def g (x : ℝ) : ℝ := 2 * x + 3

theorem f_of_g_of_3_equals_29 : f (2 + g 3) = 29 := by
  sorry

end f_of_g_of_3_equals_29_l261_26124


namespace max_reach_is_nine_feet_l261_26182

/-- The maximum height Barry and Larry can reach when Barry stands on Larry's shoulders -/
def max_reach (barry_reach : ℝ) (larry_height : ℝ) (larry_shoulder_ratio : ℝ) : ℝ :=
  larry_height * larry_shoulder_ratio + barry_reach

/-- Theorem stating the maximum height Barry and Larry can reach -/
theorem max_reach_is_nine_feet :
  max_reach 5 5 0.8 = 9 := by
  sorry

#eval max_reach 5 5 0.8

end max_reach_is_nine_feet_l261_26182
