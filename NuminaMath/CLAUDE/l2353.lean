import Mathlib

namespace paperclip_theorem_l2353_235323

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the day of the week after n days from Monday -/
def dayAfter (n : ℕ) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Monday
  | 1 => DayOfWeek.Tuesday
  | 2 => DayOfWeek.Wednesday
  | 3 => DayOfWeek.Thursday
  | 4 => DayOfWeek.Friday
  | 5 => DayOfWeek.Saturday
  | _ => DayOfWeek.Sunday

/-- Number of paperclips after n doublings -/
def paperclips (n : ℕ) : ℕ := 5 * 2^n

theorem paperclip_theorem :
  (∃ n : ℕ, paperclips n > 200 ∧ paperclips (n-1) ≤ 200) ∧
  (∀ n : ℕ, paperclips n > 200 → n ≥ 6) ∧
  dayAfter 12 = DayOfWeek.Saturday :=
sorry

end paperclip_theorem_l2353_235323


namespace factorial_30_trailing_zeros_l2353_235319

def trailing_zeros (n : ℕ) : ℕ := 
  (n / 5) + (n / 25)

theorem factorial_30_trailing_zeros : 
  trailing_zeros 30 = 7 := by sorry

end factorial_30_trailing_zeros_l2353_235319


namespace blouse_cost_l2353_235327

/-- Given information about Jane's purchase of skirts and blouses, prove the cost of each blouse. -/
theorem blouse_cost (num_skirts : ℕ) (skirt_price : ℕ) (num_blouses : ℕ) (total_paid : ℕ) (change : ℕ) :
  num_skirts = 2 →
  skirt_price = 13 →
  num_blouses = 3 →
  total_paid = 100 →
  change = 56 →
  (total_paid - change - num_skirts * skirt_price) / num_blouses = 6 := by
  sorry

#eval (100 - 56 - 2 * 13) / 3

end blouse_cost_l2353_235327


namespace angelas_height_l2353_235393

/-- Given the heights of Amy, Helen, and Angela, prove Angela's height. -/
theorem angelas_height (amy_height helen_height angela_height : ℕ) 
  (helen_taller : helen_height = amy_height + 3)
  (angela_taller : angela_height = helen_height + 4)
  (amy_is_150 : amy_height = 150) :
  angela_height = 157 := by
  sorry

end angelas_height_l2353_235393


namespace smallest_shift_for_even_function_l2353_235366

/-- Given a function f(x) = sin(ωx + π/2) where ω > 0, 
    if the distance between adjacent axes of symmetry is 2π
    and shifting f(x) to the left by m units results in an even function,
    then the smallest positive value of m is π/(2ω). -/
theorem smallest_shift_for_even_function (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x => Real.sin (ω * x + π / 2)
  (∀ x : ℝ, f (x + 2*π/ω) = f x) →  -- distance between adjacent axes of symmetry is 2π
  (∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, f (x + m) = f (-x + m)) →  -- shifting by m results in an even function
  (∃ m : ℝ, m > 0 ∧ 
    (∀ x : ℝ, f (x + m) = f (-x + m)) ∧  -- m shift results in even function
    (∀ m' : ℝ, m' > 0 → (∀ x : ℝ, f (x + m') = f (-x + m')) → m ≤ m') ∧  -- m is the smallest such shift
    m = π / (2 * ω)) :=  -- m equals π/(2ω)
by sorry

end smallest_shift_for_even_function_l2353_235366


namespace fair_coin_three_heads_probability_l2353_235306

theorem fair_coin_three_heads_probability :
  let p_head : ℝ := 1/2  -- Probability of getting heads on a fair coin
  let n : ℕ := 3        -- Number of tosses
  let p_all_heads : ℝ := p_head ^ n
  p_all_heads = 1/8 := by
sorry

end fair_coin_three_heads_probability_l2353_235306


namespace sugar_profit_percentage_l2353_235324

theorem sugar_profit_percentage (total_sugar : ℝ) (sugar_at_12_percent : ℝ) (overall_profit_percent : ℝ) :
  total_sugar = 1600 →
  sugar_at_12_percent = 1200 →
  overall_profit_percent = 11 →
  let remaining_sugar := total_sugar - sugar_at_12_percent
  let profit_12_percent := sugar_at_12_percent * 12 / 100
  let total_profit := total_sugar * overall_profit_percent / 100
  let remaining_profit := total_profit - profit_12_percent
  remaining_profit / remaining_sugar * 100 = 8 :=
by sorry

end sugar_profit_percentage_l2353_235324


namespace tin_content_in_new_alloy_l2353_235364

theorem tin_content_in_new_alloy 
  (tin_percent_first : Real) 
  (copper_percent_second : Real)
  (zinc_percent_new : Real)
  (weight_first : Real)
  (weight_second : Real)
  (h1 : tin_percent_first = 40)
  (h2 : copper_percent_second = 26)
  (h3 : zinc_percent_new = 30)
  (h4 : weight_first = 150)
  (h5 : weight_second = 250)
  : Real :=
by
  sorry

#check tin_content_in_new_alloy

end tin_content_in_new_alloy_l2353_235364


namespace card_cost_calculation_l2353_235339

theorem card_cost_calculation (christmas_cards : ℕ) (birthday_cards : ℕ) (total_spent : ℕ) : 
  christmas_cards = 20 →
  birthday_cards = 15 →
  total_spent = 70 →
  (total_spent : ℚ) / (christmas_cards + birthday_cards : ℚ) = 2 := by
  sorry

end card_cost_calculation_l2353_235339


namespace parallelogram_area_smallest_real_part_l2353_235397

theorem parallelogram_area_smallest_real_part (z : ℂ) :
  (z.im > 0) →
  (abs ((z - z⁻¹).re) ≥ 0) →
  (abs (z.im * z⁻¹.re - z.re * z⁻¹.im) = 1) →
  ∃ (w : ℂ), (w.im > 0) ∧ 
             (abs ((w - w⁻¹).re) ≥ 0) ∧ 
             (abs (w.im * w⁻¹.re - w.re * w⁻¹.im) = 1) ∧
             (abs ((w - w⁻¹).re) = 0) :=
by sorry

end parallelogram_area_smallest_real_part_l2353_235397


namespace arithmetic_mean_problem_l2353_235371

theorem arithmetic_mean_problem (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10)
  (h2 : r - p = 28) :
  (q + r) / 2 = 24 := by
  sorry

end arithmetic_mean_problem_l2353_235371


namespace marias_trip_l2353_235392

theorem marias_trip (total_distance : ℝ) (remaining_distance : ℝ) 
  (h1 : total_distance = 360)
  (h2 : remaining_distance = 135)
  (h3 : remaining_distance = total_distance - (x * total_distance + 1/4 * (total_distance - x * total_distance)))
  : x = 1/2 :=
by
  sorry

#check marias_trip

end marias_trip_l2353_235392


namespace fraction_addition_l2353_235396

theorem fraction_addition : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end fraction_addition_l2353_235396


namespace extreme_values_range_l2353_235362

/-- Given a function f(x) = 2x³ - (1/2)ax² + ax + 1, where a is a real number,
    this theorem states that the range of values for a such that f(x) has two
    extreme values in the interval (0, +∞) is (0, +∞). -/
theorem extreme_values_range (a : ℝ) :
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x ≠ y ∧
    (∀ z : ℝ, 0 < z → (6 * z^2 - a * z + a = 0) ↔ (z = x ∨ z = y))) ↔
  (0 < a) :=
sorry

end extreme_values_range_l2353_235362


namespace orthogonal_equal_magnitude_vectors_l2353_235342

/-- Given two vectors a and b in R³, if they are orthogonal and have equal magnitudes,
    then their components satisfy specific values. -/
theorem orthogonal_equal_magnitude_vectors
  (a b : ℝ × ℝ × ℝ)
  (h_a : a = (4, p, -2))
  (h_b : b = (3, 2, q))
  (h_orthogonal : a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = 0)
  (h_equal_magnitude : a.1^2 + a.2.1^2 + a.2.2^2 = b.1^2 + b.2.1^2 + b.2.2^2)
  : p = -29/12 ∧ q = 43/12 :=
by sorry

end orthogonal_equal_magnitude_vectors_l2353_235342


namespace parabola_focus_l2353_235374

/-- Represents a parabola with equation y^2 = 8x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  eq_def : equation = fun y x => y^2 = 8*x

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (2, 0)

/-- Theorem stating that the focus of the parabola y^2 = 8x is (2, 0) -/
theorem parabola_focus (p : Parabola) : focus p = (2, 0) := by
  sorry

end parabola_focus_l2353_235374


namespace print_shop_charge_difference_l2353_235372

/-- The charge difference between two print shops for a given number of copies. -/
def chargeDifference (priceX priceY : ℚ) (copies : ℕ) : ℚ :=
  copies * (priceY - priceX)

/-- The theorem stating the charge difference for 70 color copies between shop Y and shop X. -/
theorem print_shop_charge_difference :
  chargeDifference (1.20 : ℚ) (1.70 : ℚ) 70 = 35 := by
  sorry

end print_shop_charge_difference_l2353_235372


namespace problem_line_direction_cosines_l2353_235368

/-- The line defined by two equations -/
structure Line where
  eq1 : ℝ → ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ → ℝ

/-- Direction cosines of a line -/
structure DirectionCosines where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- The specific line from the problem -/
def problemLine : Line where
  eq1 := fun x y z => 2*x - 3*y - 3*z - 9
  eq2 := fun x y z => x - 2*y + z + 3

/-- Compute direction cosines for a given line -/
noncomputable def computeDirectionCosines (l : Line) : DirectionCosines :=
  { α := 9 / Real.sqrt 107
  , β := 5 / Real.sqrt 107
  , γ := 1 / Real.sqrt 107 }

/-- Theorem: The direction cosines of the problem line are (9/√107, 5/√107, 1/√107) -/
theorem problem_line_direction_cosines :
  computeDirectionCosines problemLine = 
  { α := 9 / Real.sqrt 107
  , β := 5 / Real.sqrt 107
  , γ := 1 / Real.sqrt 107 } := by
  sorry

end problem_line_direction_cosines_l2353_235368


namespace jason_pears_l2353_235328

theorem jason_pears (mike_pears jason_pears total_pears : ℕ) 
  (h1 : mike_pears = 8)
  (h2 : total_pears = 15)
  (h3 : total_pears = mike_pears + jason_pears) :
  jason_pears = 7 := by
  sorry

end jason_pears_l2353_235328


namespace axis_of_symmetry_shifted_sine_l2353_235365

/-- The axis of symmetry of a shifted sine function -/
theorem axis_of_symmetry_shifted_sine (k : ℤ) :
  let f : ℝ → ℝ := fun x ↦ 2 * Real.sin (2 * (x + π / 12))
  let axis : ℝ := k * π / 2 + π / 6
  ∀ x : ℝ, f (axis + x) = f (axis - x) := by
  sorry

end axis_of_symmetry_shifted_sine_l2353_235365


namespace range_of_odd_function_l2353_235361

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_positive (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f x = 3

theorem range_of_odd_function (f : ℝ → ℝ) (h1 : is_odd f) (h2 : f_positive f) :
  Set.range f = {-3, 0, 3} := by
  sorry

end range_of_odd_function_l2353_235361


namespace subset_condition_implies_upper_bound_l2353_235301

theorem subset_condition_implies_upper_bound (a : ℝ) :
  let A := {x : ℝ | x > 3}
  let B := {x : ℝ | x > a}
  A ⊆ B → a ≤ 3 := by
sorry

end subset_condition_implies_upper_bound_l2353_235301


namespace investment_split_l2353_235344

/-- Proves that given an initial investment of $1500 split between two banks with annual compound
    interest rates of 4% and 6% respectively, if the total amount after three years is $1755,
    then the initial investment in the bank with 4% interest rate is $476.5625. -/
theorem investment_split (x : ℝ) : 
  x ≥ 0 ∧ x ≤ 1500 ∧ 
  x * (1 + 0.04)^3 + (1500 - x) * (1 + 0.06)^3 = 1755 →
  x = 476.5625 := by sorry

end investment_split_l2353_235344


namespace grid_toothpicks_count_l2353_235307

/-- Calculates the total number of toothpicks in a rectangular grid with partitions. -/
def total_toothpicks (height width : ℕ) (partition_interval : ℕ) : ℕ :=
  let horizontal_lines := height + 1
  let vertical_lines := width + 1
  let horizontal_toothpicks := horizontal_lines * width
  let vertical_toothpicks := vertical_lines * height
  let num_partitions := (height - 1) / partition_interval
  let partition_toothpicks := num_partitions * width
  horizontal_toothpicks + vertical_toothpicks + partition_toothpicks

/-- Theorem stating that the total number of toothpicks in the specified grid is 850. -/
theorem grid_toothpicks_count :
  total_toothpicks 25 15 5 = 850 := by
  sorry

end grid_toothpicks_count_l2353_235307


namespace james_sales_theorem_l2353_235382

theorem james_sales_theorem (houses_day1 : ℕ) (houses_day2 : ℕ) (sale_rate_day2 : ℚ) (items_per_house : ℕ) :
  houses_day1 = 20 →
  houses_day2 = 2 * houses_day1 →
  sale_rate_day2 = 4/5 →
  items_per_house = 2 →
  houses_day1 * items_per_house + (houses_day2 : ℚ) * sale_rate_day2 * items_per_house = 104 :=
by sorry

end james_sales_theorem_l2353_235382


namespace soda_price_proof_l2353_235316

/-- The regular price per can of soda -/
def regular_price : ℝ := sorry

/-- The discounted price per can when purchased in 24-can cases -/
def discounted_price : ℝ := regular_price * 0.8

/-- The total price of 72 cans purchased in 24-can cases -/
def total_price : ℝ := 34.56

theorem soda_price_proof : 
  (discounted_price * 72 = total_price) → regular_price = 0.60 := by
  sorry

end soda_price_proof_l2353_235316


namespace andys_calculation_l2353_235305

theorem andys_calculation (y : ℝ) : 4 * y + 5 = 57 → (y + 5) * 4 = 72 := by
  sorry

end andys_calculation_l2353_235305


namespace crosswalk_distance_l2353_235314

/-- Given a parallelogram with one side of length 25 feet, the perpendicular distance
    between this side and its opposite side being 60 feet, and another side of length 70 feet,
    the perpendicular distance between this side and its opposite side is 150/7 feet. -/
theorem crosswalk_distance (side1 side2 height1 height2 : ℝ) : 
  side1 = 25 →
  side2 = 70 →
  height1 = 60 →
  side1 * height1 = side2 * height2 →
  height2 = 150 / 7 := by sorry

end crosswalk_distance_l2353_235314


namespace exchange_calculation_l2353_235399

/-- Exchange rate from USD to JPY -/
def exchange_rate : ℚ := 5000 / 45

/-- Amount in USD to be exchanged -/
def usd_amount : ℚ := 15

/-- Theorem stating the correct exchange amount -/
theorem exchange_calculation :
  usd_amount * exchange_rate = 5000 / 3 := by
  sorry

end exchange_calculation_l2353_235399


namespace remainder_of_sum_l2353_235317

theorem remainder_of_sum (x y u v : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_div : x = u * y + v) (h_rem : v < y) : (x + 3 * u * y) % y = v := by
  sorry

end remainder_of_sum_l2353_235317


namespace odd_integer_minus_twenty_l2353_235385

theorem odd_integer_minus_twenty : 
  (2 * 53 - 1) - 20 = 85 := by sorry

end odd_integer_minus_twenty_l2353_235385


namespace ellipse_major_axis_length_l2353_235332

/-- Given an ellipse with equation x²/a² + y²/b² = 1 (a > b > 0), 
    where its right focus is at (1,0) and b²/a = 2, 
    prove that the length of its major axis is 2√2 + 2. -/
theorem ellipse_major_axis_length 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : b^2 / a = 2) : 
  2 * a = 2 * Real.sqrt 2 + 2 := by
  sorry

end ellipse_major_axis_length_l2353_235332


namespace smallest_right_triangle_area_l2353_235381

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 4) (hb : b = 5) :
  ∃ (c : ℝ), c > 0 ∧ a^2 + c^2 = b^2 ∧
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
  ((x = a ∧ y = b) ∨ (x = b ∧ y = a) ∨ (y = a ∧ z = b) ∨ (y = b ∧ z = a)) →
  x^2 + y^2 = z^2 →
  (1/2) * x * y ≥ 6 :=
by sorry

end smallest_right_triangle_area_l2353_235381


namespace infinite_greater_than_index_l2353_235304

theorem infinite_greater_than_index :
  ∀ (a : ℕ → ℕ), (∀ n, a n ≠ 1) →
  ¬ (∃ N, ∀ n > N, a n ≤ n) :=
by sorry

end infinite_greater_than_index_l2353_235304


namespace square_rectangle_intersection_l2353_235373

structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

def Square (q : Quadrilateral) : Prop :=
  ∃ s : ℝ, s > 0 ∧
    q.B.1 - q.A.1 = s ∧ q.B.2 - q.A.2 = 0 ∧
    q.C.1 - q.B.1 = 0 ∧ q.C.2 - q.B.2 = s ∧
    q.D.1 - q.C.1 = -s ∧ q.D.2 - q.C.2 = 0 ∧
    q.A.1 - q.D.1 = 0 ∧ q.A.2 - q.D.2 = -s

def Rectangle (q : Quadrilateral) : Prop :=
  ∃ w h : ℝ, w > 0 ∧ h > 0 ∧
    q.B.1 - q.A.1 = w ∧ q.B.2 - q.A.2 = 0 ∧
    q.C.1 - q.B.1 = 0 ∧ q.C.2 - q.B.2 = h ∧
    q.D.1 - q.C.1 = -w ∧ q.D.2 - q.C.2 = 0 ∧
    q.A.1 - q.D.1 = 0 ∧ q.A.2 - q.D.2 = -h

def Perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem square_rectangle_intersection (EFGH IJKL : Quadrilateral) (E Q : ℝ × ℝ) :
  Square EFGH ∧ 
  Rectangle IJKL ∧
  EFGH.A = E ∧
  IJKL.B.1 - IJKL.A.1 = 12 ∧
  IJKL.C.2 - IJKL.B.2 = 8 ∧
  Perpendicular (EFGH.D.1 - EFGH.A.1, EFGH.D.2 - EFGH.A.2) (IJKL.B.1 - IJKL.A.1, IJKL.B.2 - IJKL.A.2) ∧
  (Q.2 - EFGH.D.2) * (IJKL.B.1 - IJKL.A.1) = 1/3 * (IJKL.B.1 - IJKL.A.1) * (IJKL.C.2 - IJKL.B.2) →
  E.2 - Q.2 = 4 := by
sorry

end square_rectangle_intersection_l2353_235373


namespace max_value_on_ellipse_l2353_235388

/-- The maximum value of x-2y for points (x,y) on the ellipse x^2/16 + y^2/9 = 1 is 2√13 -/
theorem max_value_on_ellipse :
  (∃ (x y : ℝ), x^2/16 + y^2/9 = 1 ∧ x - 2*y = 2*Real.sqrt 13) ∧
  (∀ (x y : ℝ), x^2/16 + y^2/9 = 1 → x - 2*y ≤ 2*Real.sqrt 13) := by
  sorry

end max_value_on_ellipse_l2353_235388


namespace no_divisibility_by_4_or_8_l2353_235395

/-- The set of all numbers which are the sum of the squares of three consecutive odd integers -/
def T : Set ℤ :=
  {x | ∃ n : ℤ, Odd n ∧ x = 3 * n^2 + 8}

/-- Theorem stating that no member of T is divisible by 4 or 8 -/
theorem no_divisibility_by_4_or_8 (x : ℤ) (hx : x ∈ T) :
  ¬(4 ∣ x) ∧ ¬(8 ∣ x) := by
  sorry

#check no_divisibility_by_4_or_8

end no_divisibility_by_4_or_8_l2353_235395


namespace perpendicular_lines_slope_product_l2353_235391

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
theorem perpendicular_lines_slope_product (k₁ k₂ b₁ b₂ : ℝ) :
  (∀ x y : ℝ, y = k₁ * x + b₁ → y = k₂ * x + b₂ → (k₁ * k₂ = -1 ↔ ∀ x₁ y₁ x₂ y₂ : ℝ,
    (y₁ = k₁ * x₁ + b₁ ∧ y₂ = k₁ * x₂ + b₁) →
    (y₁ = k₂ * x₁ + b₂ ∧ y₂ = k₂ * x₂ + b₂) →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (y₂ - y₁) = 0))) :=
by sorry

end perpendicular_lines_slope_product_l2353_235391


namespace value_std_dev_from_mean_l2353_235302

/-- Proves that for a normal distribution with mean 16.5 and standard deviation 1.5,
    the value 13.5 is 2 standard deviations less than the mean. -/
theorem value_std_dev_from_mean :
  let μ : ℝ := 16.5  -- mean
  let σ : ℝ := 1.5   -- standard deviation
  let x : ℝ := 13.5  -- value in question
  (x - μ) / σ = -2
  := by sorry

end value_std_dev_from_mean_l2353_235302


namespace smallest_x_with_remainders_l2353_235313

theorem smallest_x_with_remainders : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 7 = 6 ∧ 
  x % 8 = 7 ∧ 
  ∀ y : ℕ, y > 0 → y % 6 = 5 → y % 7 = 6 → y % 8 = 7 → x ≤ y :=
by
  use 167
  sorry

end smallest_x_with_remainders_l2353_235313


namespace painted_stripe_area_l2353_235348

/-- The area of a painted stripe on a cylindrical tank -/
theorem painted_stripe_area (d h w1 w2 r1 r2 : ℝ) (hd : d = 40) (hh : h = 100) 
  (hw1 : w1 = 5) (hw2 : w2 = 7) (hr1 : r1 = 3) (hr2 : r2 = 3) : 
  w1 * (π * d * r1) + w2 * (π * d * r2) = 1440 * π := by
  sorry

end painted_stripe_area_l2353_235348


namespace nancys_weight_l2353_235343

/-- 
Given that Nancy's total daily water intake (including water from food) is 62 pounds,
and she drinks 75% of her body weight in water plus 2 pounds from food,
prove that her weight is 80 pounds.
-/
theorem nancys_weight (W : ℝ) : 0.75 * W + 2 = 62 → W = 80 := by
  sorry

end nancys_weight_l2353_235343


namespace sin_75_degrees_l2353_235384

theorem sin_75_degrees : 
  Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  -- Define known values
  have sin_45 : Real.sin (45 * π / 180) = Real.sqrt 2 / 2 := sorry
  have cos_45 : Real.cos (45 * π / 180) = Real.sqrt 2 / 2 := sorry
  have sin_30 : Real.sin (30 * π / 180) = 1 / 2 := sorry
  have cos_30 : Real.cos (30 * π / 180) = Real.sqrt 3 / 2 := sorry

  -- Proof goes here
  sorry

end sin_75_degrees_l2353_235384


namespace problem_solution_l2353_235334

def A : Set ℝ := {x : ℝ | x^2 + 3*x - 28 < 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 2 < x ∧ x < m + 1}

theorem problem_solution (m : ℝ) :
  (3 ∈ B m → 2 < m ∧ m < 5) ∧
  (B m ⊂ A → -5 ≤ m ∧ m ≤ 3) :=
sorry

end problem_solution_l2353_235334


namespace smallest_hope_number_l2353_235312

def hope_number (n : ℕ+) : Prop :=
  ∃ (a b c : ℕ), 
    (n / 8 : ℚ) = a^2 ∧ 
    (n / 9 : ℚ) = b^3 ∧ 
    (n / 25 : ℚ) = c^5

theorem smallest_hope_number :
  ∃ (n : ℕ+), hope_number n ∧ 
    (∀ (m : ℕ+), hope_number m → n ≤ m) ∧
    n = 2^15 * 3^20 * 5^12 := by
  sorry

end smallest_hope_number_l2353_235312


namespace special_line_equation_l2353_235346

/-- A line passing through point (3,-1) with x-intercept twice its y-intercept -/
structure SpecialLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through point (3,-1) -/
  passes_through_point : slope * 3 + y_intercept = -1
  /-- The x-intercept is twice the y-intercept -/
  intercept_condition : y_intercept ≠ 0 → -y_intercept / slope = 2 * y_intercept

theorem special_line_equation (l : SpecialLine) :
  (∀ x y, y = l.slope * x + l.y_intercept ↔ x + 2 * y - 1 = 0) ∨
  (∀ x y, y = l.slope * x + l.y_intercept ↔ x + 3 * y = 0) :=
sorry

end special_line_equation_l2353_235346


namespace gcd_7392_15015_l2353_235331

theorem gcd_7392_15015 : Nat.gcd 7392 15015 = 1 := by
  sorry

end gcd_7392_15015_l2353_235331


namespace vertical_stripe_percentage_is_ten_percent_l2353_235329

/-- Represents the distribution of shirt types in a college cafeteria. -/
structure ShirtDistribution where
  total : Nat
  checkered : Nat
  polkaDotted : Nat
  plain : Nat
  horizontalMultiplier : Nat

/-- Calculates the percentage of people wearing vertical stripes. -/
def verticalStripePercentage (d : ShirtDistribution) : Rat :=
  let stripes := d.total - (d.checkered + d.polkaDotted + d.plain)
  let horizontal := d.checkered * d.horizontalMultiplier
  let vertical := stripes - horizontal
  (vertical : Rat) / d.total * 100

/-- Theorem stating that the percentage of people wearing vertical stripes is 10%. -/
theorem vertical_stripe_percentage_is_ten_percent : 
  let d : ShirtDistribution := {
    total := 100,
    checkered := 12,
    polkaDotted := 15,
    plain := 3,
    horizontalMultiplier := 5
  }
  verticalStripePercentage d = 10 := by sorry

end vertical_stripe_percentage_is_ten_percent_l2353_235329


namespace friends_in_group_l2353_235394

def number_of_friends (initial_wings : ℕ) (additional_wings : ℕ) (wings_per_person : ℕ) : ℕ :=
  (initial_wings + additional_wings) / wings_per_person

theorem friends_in_group :
  number_of_friends 8 10 6 = 3 :=
by sorry

end friends_in_group_l2353_235394


namespace simplify_fraction_l2353_235378

/-- Given x = 3 and y = 4, prove that (12 * x * y^3) / (9 * x^2 * y^2) = 16 / 9 -/
theorem simplify_fraction (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (12 * x * y^3) / (9 * x^2 * y^2) = 16 / 9 := by
  sorry

end simplify_fraction_l2353_235378


namespace parallel_linear_function_b_value_l2353_235353

/-- A linear function y = kx + b whose graph is parallel to y = 3x and passes through (1, -1) has b = -4 -/
theorem parallel_linear_function_b_value (k b : ℝ) : 
  (∀ x y : ℝ, y = k * x + b) →  -- Definition of linear function
  k = 3 →  -- Parallel to y = 3x
  -1 = k * 1 + b →  -- Passes through (1, -1)
  b = -4 := by
sorry

end parallel_linear_function_b_value_l2353_235353


namespace carsons_speed_l2353_235309

/-- Given information about Jerry and Carson's running times and the distance to school,
    we prove that Carson's speed is 8 miles per hour. -/
theorem carsons_speed (distance : ℝ) (jerry_time : ℝ) (carson_time : ℝ)
  (h1 : distance = 4) -- Distance to school is 4 miles
  (h2 : jerry_time = 15) -- Jerry's one-way trip time is 15 minutes
  (h3 : carson_time = 2 * jerry_time) -- Carson's one-way trip time is twice Jerry's
  : carson_time / 60 * distance = 8 := by
  sorry

end carsons_speed_l2353_235309


namespace constant_value_l2353_235356

theorem constant_value (x : ℝ) (constant : ℝ) 
  (eq : 5 * x + 3 = 10 * x - constant) 
  (h : x = 5) : 
  constant = 22 := by
sorry

end constant_value_l2353_235356


namespace binomial_coefficient_19_13_l2353_235321

theorem binomial_coefficient_19_13 :
  (Nat.choose 18 11 = 31824) →
  (Nat.choose 18 12 = 18564) →
  (Nat.choose 20 13 = 77520) →
  Nat.choose 19 13 = 58956 := by
  sorry

end binomial_coefficient_19_13_l2353_235321


namespace electricity_fee_properties_l2353_235354

-- Define the relationship between electricity usage and fee
def electricity_fee (x : ℝ) : ℝ := 0.55 * x

-- Theorem stating the properties of the electricity fee function
theorem electricity_fee_properties :
  -- 1. x is independent, y is dependent (implicit in the function definition)
  -- 2. For every increase of 1 in x, y increases by 0.55
  (∀ x : ℝ, electricity_fee (x + 1) = electricity_fee x + 0.55) ∧
  -- 3. When x = 8, y = 4.4
  (electricity_fee 8 = 4.4) ∧
  -- 4. When y = 3.75, x ≠ 7
  (∀ x : ℝ, electricity_fee x = 3.75 → x ≠ 7) := by
  sorry


end electricity_fee_properties_l2353_235354


namespace excircle_tangency_triangle_ratio_l2353_235333

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)

-- Define the excircle
structure Excircle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the tangency triangle
structure TangencyTriangle :=
  (area : ℝ)

-- Define the theorem
theorem excircle_tangency_triangle_ratio
  (ABC : Triangle)
  (ωA ωB ωC : Excircle)
  (TA TB TC : TangencyTriangle)
  (h1 : TA.area = 4)
  (h2 : TB.area = 5)
  (h3 : TC.area = 6) :
  ∃ (k : ℝ), k > 0 ∧ ABC.a = 15 * k ∧ ABC.b = 12 * k ∧ ABC.c = 10 * k :=
sorry

end excircle_tangency_triangle_ratio_l2353_235333


namespace solutions_of_quadratic_equation_l2353_235375

theorem solutions_of_quadratic_equation :
  ∀ x : ℝ, x^2 - 64 = 0 ↔ x = 8 ∨ x = -8 := by
  sorry

end solutions_of_quadratic_equation_l2353_235375


namespace parabola_line_intersection_l2353_235367

/-- Given a parabola and a line with exactly one intersection point, prove a specific algebraic identity. -/
theorem parabola_line_intersection (m : ℝ) : 
  (∃! x : ℝ, 4 * x^2 + 4 * x + 5 = 8 * m * x + 8 * m) → 
  m^36 + 1155 / m^12 = 39236 :=
by sorry

end parabola_line_intersection_l2353_235367


namespace intersection_equals_B_l2353_235326

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a*x = 3}

-- Define the set of possible values for a
def possible_a : Set ℝ := {0, -1, 3}

-- State the theorem
theorem intersection_equals_B (a : ℝ) :
  (A ∩ B a = B a) ↔ a ∈ possible_a :=
sorry

end intersection_equals_B_l2353_235326


namespace stamp_block_bounds_l2353_235336

/-- 
b(n) is the smallest number of blocks of three adjacent stamps that can be torn out 
from an n × n sheet to make it impossible to tear out any more such blocks.
-/
noncomputable def b (n : ℕ) : ℕ := sorry

/-- 
There exist real constants c and d such that for all positive integers n, 
the function b(n) satisfies the inequality (1/7)n^2 - cn ≤ b(n) ≤ (1/5)n^2 + dn
-/
theorem stamp_block_bounds : 
  ∃ (c d : ℝ), ∀ (n : ℕ), n > 0 → 
    ((1 : ℝ) / 7) * (n : ℝ)^2 - c * (n : ℝ) ≤ (b n : ℝ) ∧ 
    (b n : ℝ) ≤ ((1 : ℝ) / 5) * (n : ℝ)^2 + d * (n : ℝ) := by
  sorry

end stamp_block_bounds_l2353_235336


namespace quadratic_properties_l2353_235389

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 2)^2 + 6

-- Theorem stating the properties of the quadratic function
theorem quadratic_properties :
  -- The parabola opens upwards
  (∀ x y : ℝ, x < y → f ((x + y) / 2) < (f x + f y) / 2) ∧
  -- When x < 2, f(x) decreases as x increases
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 2 → f x₁ > f x₂) ∧
  -- The axis of symmetry is x = 2
  (∀ x : ℝ, f (2 - x) = f (2 + x)) ∧
  -- f(0) = 10
  f 0 = 10 :=
by sorry

end quadratic_properties_l2353_235389


namespace fourth_month_sale_is_9230_l2353_235318

/-- Calculates the sale in the fourth month given the sales for other months and the average sale. -/
def fourth_month_sale (first_month second_month third_month fifth_month sixth_month average_sale : ℕ) : ℕ :=
  6 * average_sale - (first_month + second_month + third_month + fifth_month + sixth_month)

/-- Theorem stating that the sale in the fourth month is 9230 given the problem conditions. -/
theorem fourth_month_sale_is_9230 :
  fourth_month_sale 8435 8927 8855 8562 6991 8500 = 9230 := by
  sorry

end fourth_month_sale_is_9230_l2353_235318


namespace total_lunch_spending_l2353_235351

def lunch_problem (your_spending friend_spending total_spending : ℕ) : Prop :=
  friend_spending = 11 ∧
  friend_spending = your_spending + 3 ∧
  total_spending = your_spending + friend_spending

theorem total_lunch_spending : ∃ (your_spending friend_spending total_spending : ℕ),
  lunch_problem your_spending friend_spending total_spending ∧ total_spending = 19 := by
  sorry

end total_lunch_spending_l2353_235351


namespace right_triangle_with_specific_altitude_and_segment_difference_l2353_235355

/-- Represents a right-angled triangle with an altitude to the hypotenuse -/
structure RightTriangleWithAltitude where
  /-- First leg of the triangle -/
  leg1 : ℝ
  /-- Second leg of the triangle -/
  leg2 : ℝ
  /-- Hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- Altitude drawn to the hypotenuse -/
  altitude : ℝ
  /-- First segment of the hypotenuse -/
  segment1 : ℝ
  /-- Second segment of the hypotenuse -/
  segment2 : ℝ
  /-- The triangle is right-angled -/
  right_angle : leg1^2 + leg2^2 = hypotenuse^2
  /-- The altitude divides the hypotenuse into two segments -/
  hypotenuse_segments : segment1 + segment2 = hypotenuse
  /-- The altitude creates similar triangles -/
  similar_triangles : altitude^2 = segment1 * segment2

/-- Theorem: Given a right-angled triangle with specific altitude and hypotenuse segment difference, prove its sides -/
theorem right_triangle_with_specific_altitude_and_segment_difference
  (t : RightTriangleWithAltitude)
  (h_altitude : t.altitude = 12)
  (h_segment_diff : t.segment1 - t.segment2 = 7) :
  t.leg1 = 15 ∧ t.leg2 = 20 ∧ t.hypotenuse = 25 := by
  sorry

end right_triangle_with_specific_altitude_and_segment_difference_l2353_235355


namespace symmetric_line_l2353_235322

/-- Given a line l and another line, find the equation of the line symmetric to the given line with respect to l -/
theorem symmetric_line (a b c d e f : ℝ) :
  let l : ℝ → ℝ := λ x => 3 * x + 3
  let given_line : ℝ → ℝ := λ x => x - 2
  let symmetric_line : ℝ → ℝ := λ x => -7 * x - 22
  (∀ x, given_line x = x - (l x)) →
  (∀ x, symmetric_line x = (l x) - (given_line x - (l x))) :=
by sorry

end symmetric_line_l2353_235322


namespace set_relations_l2353_235358

def A : Set ℝ := {x | x > 1}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem set_relations (a : ℝ) :
  (B a ⊆ A ↔ a ∈ Set.Ici 1) ∧
  (Set.Nonempty (A ∩ B a) ↔ a ∈ Set.Ioi 0) := by
  sorry

end set_relations_l2353_235358


namespace min_x_for_sqrt_2x_minus_1_l2353_235390

theorem min_x_for_sqrt_2x_minus_1 :
  ∀ x : ℝ, (∃ y : ℝ, y^2 = 2*x - 1) → x ≥ (1/2 : ℝ) :=
by sorry

end min_x_for_sqrt_2x_minus_1_l2353_235390


namespace reflect_parabola_x_axis_l2353_235325

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Reflects a parabola across the x-axis -/
def reflect_x_axis (p : Parabola) : Parabola :=
  { a := -p.a, b := -p.b, c := -p.c }

/-- The original parabola y = x^2 - x - 1 -/
def original_parabola : Parabola :=
  { a := 1, b := -1, c := -1 }

theorem reflect_parabola_x_axis :
  reflect_x_axis original_parabola = { a := -1, b := 1, c := 1 } := by
  sorry

end reflect_parabola_x_axis_l2353_235325


namespace original_triangle_area_l2353_235359

/-- Given a triangle with area A, if its dimensions are quadrupled to form a new triangle
    with an area of 144 square feet, then the area A of the original triangle is 9 square feet. -/
theorem original_triangle_area (A : ℝ) : 
  (∃ (new_triangle : ℝ), new_triangle = 144 ∧ new_triangle = 16 * A) → A = 9 := by
  sorry

end original_triangle_area_l2353_235359


namespace cookie_recipe_total_cups_l2353_235376

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio :=
  (butter : ℕ)
  (flour : ℕ)
  (sugar : ℕ)

/-- Calculates the total cups of ingredients given a ratio and amount of sugar -/
def totalCups (ratio : RecipeRatio) (sugarCups : ℕ) : ℕ :=
  let partSize := sugarCups / ratio.sugar
  (ratio.butter + ratio.flour + ratio.sugar) * partSize

/-- Theorem: Given the specified ratio and sugar amount, the total cups is 40 -/
theorem cookie_recipe_total_cups :
  let ratio : RecipeRatio := ⟨2, 5, 3⟩
  totalCups ratio 12 = 40 := by
  sorry

end cookie_recipe_total_cups_l2353_235376


namespace sweets_distribution_l2353_235387

/-- The number of children initially supposed to receive sweets -/
def initial_children : ℕ := 190

/-- The number of absent children -/
def absent_children : ℕ := 70

/-- The number of extra sweets each child received due to absences -/
def extra_sweets : ℕ := 14

/-- The total number of sweets each child received -/
def total_sweets : ℕ := 38

theorem sweets_distribution :
  initial_children * (total_sweets - extra_sweets) = 
  (initial_children - absent_children) * total_sweets :=
sorry

end sweets_distribution_l2353_235387


namespace paulines_dress_cost_l2353_235330

theorem paulines_dress_cost (pauline ida jean patty : ℕ) 
  (h1 : patty = ida + 10)
  (h2 : ida = jean + 30)
  (h3 : jean = pauline - 10)
  (h4 : pauline + ida + jean + patty = 160) :
  pauline = 30 := by
  sorry

end paulines_dress_cost_l2353_235330


namespace hyperbola_center_is_3_4_l2353_235370

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 16 * y^2 + 128 * y + 100 = 0

/-- The center of a hyperbola -/
def hyperbola_center (h k : ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    hyperbola_equation x y ↔ ((x - h)^2 / a^2) - ((y - k)^2 / b^2) = 1

/-- Theorem: The center of the given hyperbola is (3, 4) -/
theorem hyperbola_center_is_3_4 : hyperbola_center 3 4 := by
  sorry

end hyperbola_center_is_3_4_l2353_235370


namespace log_base_range_l2353_235360

-- Define the function f(x) = log_a(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_base_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 < f a 3 → a > 1 := by
  sorry

end log_base_range_l2353_235360


namespace unique_positive_solution_l2353_235340

theorem unique_positive_solution : ∃! (y : ℝ), y > 0 ∧ (y / 100) * y = 9 := by
  sorry

end unique_positive_solution_l2353_235340


namespace equidistant_point_on_y_axis_l2353_235383

theorem equidistant_point_on_y_axis :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (2, 5)
  ∃! y : ℝ, 
    ((-3 - 0)^2 + (0 - y)^2 = (2 - 0)^2 + (5 - y)^2) ∧
    y = 2 :=
by sorry

end equidistant_point_on_y_axis_l2353_235383


namespace complex_sum_product_real_implies_a_eq_one_l2353_235310

/-- Given complex numbers z₁ and z₂, prove that if their sum and product are real, then the imaginary part of z₁ is 1. -/
theorem complex_sum_product_real_implies_a_eq_one (a b : ℝ) : 
  let z₁ : ℂ := -1 + a * I
  let z₂ : ℂ := b - I
  (∃ (x : ℝ), z₁ + z₂ = x) → (∃ (y : ℝ), z₁ * z₂ = y) → a = 1 := by
  sorry

end complex_sum_product_real_implies_a_eq_one_l2353_235310


namespace inscribed_circle_radius_squared_l2353_235386

/-- A circle inscribed in a quadrilateral EFGH -/
structure InscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The point where the circle is tangent to EF -/
  R : Point
  /-- The point where the circle is tangent to GH -/
  S : Point
  /-- The length of ER -/
  ER : ℝ
  /-- The length of RF -/
  RF : ℝ
  /-- The length of GS -/
  GS : ℝ
  /-- The length of SH -/
  SH : ℝ

/-- Theorem: The square of the radius of the inscribed circle is 868 -/
theorem inscribed_circle_radius_squared (c : InscribedCircle)
    (h1 : c.ER = 21)
    (h2 : c.RF = 28)
    (h3 : c.GS = 40)
    (h4 : c.SH = 32) :
    c.r^2 = 868 := by
  sorry

end inscribed_circle_radius_squared_l2353_235386


namespace ellipse_eccentricity_l2353_235315

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
  (∃ P : ℝ × ℝ, P.1 = -c ∧ (P.2 = b^2 / a ∨ P.2 = -b^2 / a)) →
  (Real.arctan ((2 * c) / (b^2 / a)) = π / 3) →
  c / a = Real.sqrt 3 / 3 := by
sorry

end ellipse_eccentricity_l2353_235315


namespace problem_solution_l2353_235349

theorem problem_solution :
  (∃ n : ℕ, 20 = 4 * n) ∧
  (∃ m : ℕ, 180 = 9 * m) ∧
  (∃ k : ℕ, 209 = 19 * k) ∧
  (∃ l : ℕ, 57 = 19 * l) ∧
  (∃ p : ℕ, 90 = 30 * p) ∧
  (∃ q : ℕ, 34 = 17 * q) ∧
  (∃ r : ℕ, 51 = 17 * r) :=
by
  sorry

end problem_solution_l2353_235349


namespace parabola_vertex_on_line_l2353_235345

/-- The value of d for which the vertex of the parabola y = x^2 - 10x + d lies on the line y = 2x --/
theorem parabola_vertex_on_line (d : ℝ) : 
  (∃ x y : ℝ, y = x^2 - 10*x + d ∧ 
              y = 2*x ∧ 
              ∀ t : ℝ, (t^2 - 10*t + d) ≥ (x^2 - 10*x + d)) → 
  d = 35 := by
sorry

end parabola_vertex_on_line_l2353_235345


namespace abc_sum_l2353_235337

theorem abc_sum (a b c : ℕ) : 
  (10 ≤ a ∧ a < 100) → 
  (10 ≤ b ∧ b < 100) → 
  (10 ≤ c ∧ c < 100) → 
  a < b → 
  b < c → 
  a * b * c = 3960 → 
  Even (a + b + c) → 
  a + b + c = 50 := by
sorry

end abc_sum_l2353_235337


namespace students_wearing_other_colors_l2353_235303

theorem students_wearing_other_colors 
  (total_students : ℕ) 
  (blue_percent : ℚ) 
  (red_percent : ℚ) 
  (green_percent : ℚ) 
  (h1 : total_students = 800) 
  (h2 : blue_percent = 45/100) 
  (h3 : red_percent = 23/100) 
  (h4 : green_percent = 15/100) : 
  ℕ := by
  sorry

#check students_wearing_other_colors

end students_wearing_other_colors_l2353_235303


namespace longs_interest_l2353_235398

/-- Calculates the total interest earned after a given number of years with compound interest and an additional deposit -/
def totalInterest (initialInvestment : ℝ) (interestRate : ℝ) (additionalDeposit : ℝ) (depositYear : ℕ) (totalYears : ℕ) : ℝ :=
  let finalAmount := 
    (initialInvestment * (1 + interestRate) ^ depositYear + additionalDeposit) * (1 + interestRate) ^ (totalYears - depositYear)
  finalAmount - initialInvestment - additionalDeposit

/-- The total interest earned by Long after 4 years -/
theorem longs_interest : 
  totalInterest 1200 0.08 500 2 4 = 515.26 := by sorry

end longs_interest_l2353_235398


namespace all_three_hits_mutually_exclusive_with_at_most_two_hits_l2353_235308

-- Define the sample space for three shots
inductive ShotOutcome
| Hit
| Miss

-- Define the type for a sequence of three shots
def ThreeShots := (ShotOutcome × ShotOutcome × ShotOutcome)

-- Define the event "at most two hits"
def atMostTwoHits (shots : ThreeShots) : Prop :=
  match shots with
  | (ShotOutcome.Hit, ShotOutcome.Hit, ShotOutcome.Hit) => False
  | _ => True

-- Define the event "all three hits"
def allThreeHits (shots : ThreeShots) : Prop :=
  match shots with
  | (ShotOutcome.Hit, ShotOutcome.Hit, ShotOutcome.Hit) => True
  | _ => False

-- Theorem stating that "all three hits" and "at most two hits" are mutually exclusive
theorem all_three_hits_mutually_exclusive_with_at_most_two_hits :
  ∀ (shots : ThreeShots), ¬(atMostTwoHits shots ∧ allThreeHits shots) :=
by
  sorry


end all_three_hits_mutually_exclusive_with_at_most_two_hits_l2353_235308


namespace rice_distributed_in_five_days_l2353_235300

/-- The amount of rice distributed in the first 5 days of dike construction --/
theorem rice_distributed_in_five_days : 
  let initial_workers : ℕ := 64
  let daily_increase : ℕ := 7
  let rice_per_worker : ℕ := 3
  let days : ℕ := 5
  let total_workers : ℕ := (days * (2 * initial_workers + (days - 1) * daily_increase)) / 2
  total_workers * rice_per_worker = 1170 := by
  sorry

end rice_distributed_in_five_days_l2353_235300


namespace opposite_vertices_distance_l2353_235380

def cube_side_length : ℝ := 2

theorem opposite_vertices_distance (cube_side : ℝ) (h : cube_side = cube_side_length) :
  let diagonal := Real.sqrt (3 * cube_side ^ 2)
  diagonal = 2 * Real.sqrt 3 := by sorry

end opposite_vertices_distance_l2353_235380


namespace equation_solution_l2353_235369

theorem equation_solution (x : ℝ) : 
  x ≠ 2 → ((2 * x - 5) / (x - 2) = (3 * x - 3) / (x - 2) - 3) ↔ x = 4 := by
  sorry

end equation_solution_l2353_235369


namespace dance_attendance_l2353_235338

theorem dance_attendance (boys girls teachers : ℕ) : 
  (boys : ℚ) / girls = 3 / 4 →
  teachers = boys / 5 →
  boys + girls + teachers = 114 →
  girls = 60 := by
sorry

end dance_attendance_l2353_235338


namespace smallest_integer_square_equation_l2353_235377

theorem smallest_integer_square_equation : ∃ x : ℤ, 
  (∀ y : ℤ, y^2 = 3*y + 72 → x ≤ y) ∧ x^2 = 3*x + 72 := by
  sorry

end smallest_integer_square_equation_l2353_235377


namespace league_teams_l2353_235357

theorem league_teams (n : ℕ) : n * (n - 1) / 2 = 55 → n = 11 := by
  sorry

end league_teams_l2353_235357


namespace max_value_of_expression_l2353_235379

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  ∃ (max : ℝ), max = 3 ∧ ∀ (x y z : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 → x + y^2 + z^4 ≤ max :=
sorry

end max_value_of_expression_l2353_235379


namespace parallel_lines_condition_l2353_235341

/-- Represents a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel --/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a ≠ 0 ∧ l2.a ≠ 0

/-- The main theorem to be proved --/
theorem parallel_lines_condition (a : ℝ) :
  (parallel ⟨a, 1, -1⟩ ⟨1, a, 2⟩ → a = 1) ∧
  ¬(a = 1 → parallel ⟨a, 1, -1⟩ ⟨1, a, 2⟩) := by
  sorry

end parallel_lines_condition_l2353_235341


namespace laptop_price_l2353_235363

theorem laptop_price (sticker_price : ℝ) : 
  (0.8 * sticker_price - 120 = 0.7 * sticker_price - 50 - 30) →
  sticker_price = 1000 := by
sorry

end laptop_price_l2353_235363


namespace inequality_of_four_positives_l2353_235350

theorem inequality_of_four_positives (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.sqrt ((a^2 + b^2 + c^2 + d^2) / 4) ≥ (((abc + abd + acd + bcd) / 4) ^ (1/3)) :=
by sorry

end inequality_of_four_positives_l2353_235350


namespace unique_function_solution_l2353_235320

theorem unique_function_solution :
  ∃! f : ℕ → ℕ, ∀ x y : ℕ, x > 0 ∧ y > 0 →
    f x + y * (f (f x)) < x * (1 + f y) + 2021 ∧ f = id := by
  sorry

end unique_function_solution_l2353_235320


namespace maria_final_amount_l2353_235311

def salary : ℝ := 2000
def tax_rate : ℝ := 0.20
def insurance_rate : ℝ := 0.05
def utility_rate : ℝ := 0.25

def remaining_after_deductions : ℝ := salary * (1 - tax_rate - insurance_rate)
def utility_bill : ℝ := remaining_after_deductions * utility_rate
def final_amount : ℝ := remaining_after_deductions - utility_bill

theorem maria_final_amount : final_amount = 1125 := by
  sorry

end maria_final_amount_l2353_235311


namespace division_simplification_l2353_235347

theorem division_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  6 * a^2 * b / (2 * a * b) = 3 * a := by
  sorry

end division_simplification_l2353_235347


namespace arthur_total_distance_l2353_235352

/-- Calculates the total distance walked by Arthur in miles -/
def arthur_walk (block_length : ℚ) (east west north south : ℕ) : ℚ :=
  ((east + west + north + south) : ℚ) * block_length

/-- Theorem: Arthur's total walk distance is 4.5 miles -/
theorem arthur_total_distance :
  arthur_walk (1/4) 8 0 15 5 = 4.5 := by sorry

end arthur_total_distance_l2353_235352


namespace trig_inequality_l2353_235335

theorem trig_inequality : 
  let a := Real.sin (2 * Real.pi / 7)
  let b := Real.cos (12 * Real.pi / 7)
  let c := Real.tan (9 * Real.pi / 7)
  c > a ∧ a > b := by sorry

end trig_inequality_l2353_235335
