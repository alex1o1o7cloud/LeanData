import Mathlib

namespace cubic_polynomial_roots_l3894_389470

theorem cubic_polynomial_roots (a b c : ℤ) (r₁ r₂ r₃ : ℤ) : 
  (∀ x : ℤ, x^3 + a*x^2 + b*x + c = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  (r₁ > 2 ∧ r₂ > 2 ∧ r₃ > 2) →
  a + b + c + 1 = -2009 →
  (r₁ - 1) * (r₂ - 1) * (r₃ - 1) = 2009 →
  a = -58 := by
sorry

end cubic_polynomial_roots_l3894_389470


namespace martha_clothes_count_l3894_389482

/-- Calculates the total number of clothes Martha takes home given the number of jackets and t-shirts bought -/
def total_clothes (jackets_bought : ℕ) (tshirts_bought : ℕ) : ℕ :=
  let free_jackets := jackets_bought / 2
  let free_tshirts := tshirts_bought / 3
  jackets_bought + free_jackets + tshirts_bought + free_tshirts

/-- Proves that Martha takes home 18 clothes when buying 4 jackets and 9 t-shirts -/
theorem martha_clothes_count : total_clothes 4 9 = 18 := by
  sorry

end martha_clothes_count_l3894_389482


namespace min_value_expression_equality_condition_l3894_389419

theorem min_value_expression (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0)
  (sum_eq_4 : a + b + c + d = 4) : 
  (a^8 / ((a^2+b)*(a^2+c)*(a^2+d))) + 
  (b^8 / ((b^2+c)*(b^2+d)*(b^2+a))) + 
  (c^8 / ((c^2+d)*(c^2+a)*(c^2+b))) + 
  (d^8 / ((d^2+a)*(d^2+b)*(d^2+c))) ≥ (1/2) := by
  sorry

theorem equality_condition (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0)
  (sum_eq_4 : a + b + c + d = 4) :
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ↔ 
  (a^8 / ((a^2+b)*(a^2+c)*(a^2+d))) + 
  (b^8 / ((b^2+c)*(b^2+d)*(b^2+a))) + 
  (c^8 / ((c^2+d)*(c^2+a)*(c^2+b))) + 
  (d^8 / ((d^2+a)*(d^2+b)*(d^2+c))) = (1/2) := by
  sorry

end min_value_expression_equality_condition_l3894_389419


namespace sqrt_2023_between_40_and_45_l3894_389437

theorem sqrt_2023_between_40_and_45 : 40 < Real.sqrt 2023 ∧ Real.sqrt 2023 < 45 := by
  sorry

end sqrt_2023_between_40_and_45_l3894_389437


namespace work_earnings_problem_l3894_389498

theorem work_earnings_problem (t : ℝ) : 
  (t + 2) * (3 * t - 2) = (3 * t - 4) * (t + 3) + 3 → t = 5 := by
  sorry

end work_earnings_problem_l3894_389498


namespace estimate_total_balls_l3894_389457

/-- Represents a box containing red and green balls -/
structure BallBox where
  redBalls : ℕ
  totalBalls : ℕ
  hRedBalls : redBalls > 0
  hTotalBalls : totalBalls ≥ redBalls

/-- The probability of drawing a red ball -/
def drawRedProbability (box : BallBox) : ℚ :=
  box.redBalls / box.totalBalls

theorem estimate_total_balls
  (box : BallBox)
  (hRedBalls : box.redBalls = 5)
  (hProbability : drawRedProbability box = 1/4) :
  box.totalBalls = 20 := by
sorry

end estimate_total_balls_l3894_389457


namespace hundred_days_after_wednesday_is_friday_l3894_389463

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => nextDay (dayAfter start n)

/-- Theorem stating that 100 days after Wednesday is Friday -/
theorem hundred_days_after_wednesday_is_friday :
  dayAfter DayOfWeek.Wednesday 100 = DayOfWeek.Friday := by
  sorry


end hundred_days_after_wednesday_is_friday_l3894_389463


namespace subset_relation_l3894_389488

theorem subset_relation (A B C : Set α) (h : A ∪ B = B ∩ C) : A ⊆ C := by
  sorry

end subset_relation_l3894_389488


namespace prime_square_plus_200_is_square_l3894_389433

theorem prime_square_plus_200_is_square (p : ℕ) : 
  Prime p ∧ ∃ (n : ℕ), p^2 + 200 = n^2 ↔ p = 5 ∨ p = 23 := by
  sorry

end prime_square_plus_200_is_square_l3894_389433


namespace dans_limes_l3894_389474

theorem dans_limes (limes_picked : ℕ) (limes_given : ℕ) : limes_picked = 9 → limes_given = 4 → limes_picked + limes_given = 13 := by
  sorry

end dans_limes_l3894_389474


namespace parabola_sum_l3894_389448

/-- A parabola with equation y = px^2 + qx + r -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.yCoord (para : Parabola) (x : ℝ) : ℝ :=
  para.p * x^2 + para.q * x + para.r

theorem parabola_sum (para : Parabola) :
  para.yCoord 3 = 2 →   -- Vertex at (3, 2)
  para.yCoord 1 = 6 →   -- Passes through (1, 6)
  para.p + para.q + para.r = 6 := by
sorry

end parabola_sum_l3894_389448


namespace polynomial_equality_l3894_389407

/-- Given that 4x^4 + 8x^3 + g(x) = 2x^4 - 5x^3 + 7x + 4,
    prove that g(x) = -2x^4 - 13x^3 + 7x + 4 -/
theorem polynomial_equality (x : ℝ) (g : ℝ → ℝ) 
    (h : ∀ x, 4 * x^4 + 8 * x^3 + g x = 2 * x^4 - 5 * x^3 + 7 * x + 4) :
  g x = -2 * x^4 - 13 * x^3 + 7 * x + 4 := by
  sorry

end polynomial_equality_l3894_389407


namespace rock_collection_difference_l3894_389484

theorem rock_collection_difference (joshua_rocks : ℕ) (jose_rocks : ℕ) (albert_rocks : ℕ)
  (joshua_80 : joshua_rocks = 80)
  (jose_fewer : jose_rocks < joshua_rocks)
  (albert_jose_diff : albert_rocks = jose_rocks + 20)
  (albert_joshua_diff : albert_rocks = joshua_rocks + 6) :
  joshua_rocks - jose_rocks = 14 := by
sorry

end rock_collection_difference_l3894_389484


namespace remainder_of_B_l3894_389454

theorem remainder_of_B (A B : ℕ) (h : B = 9 * A + 13) : B % 9 = 4 := by
  sorry

end remainder_of_B_l3894_389454


namespace cos_sum_max_min_points_l3894_389450

/-- Given a function f(x) = cos(2x) + sin(x), prove that the cosine of the sum of
    the abscissas of its maximum and minimum points equals 1/4. -/
theorem cos_sum_max_min_points (f : ℝ → ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, f x = Real.cos (2 * x) + Real.sin x) →
  (∀ x, f x ≤ f x₁) →
  (∀ x, f x ≥ f x₂) →
  Real.cos (x₁ + x₂) = 1/4 := by
  sorry

end cos_sum_max_min_points_l3894_389450


namespace roots_quadratic_sum_l3894_389418

theorem roots_quadratic_sum (a b : ℝ) : 
  (a^2 - 3*a + 1 = 0) → 
  (b^2 - 3*b + 1 = 0) → 
  (1 / (a^2 + 1) + 1 / (b^2 + 1) = 1) :=
by
  sorry

end roots_quadratic_sum_l3894_389418


namespace xiao_hua_seat_l3894_389462

structure Classroom where
  rows : Nat
  columns : Nat

structure Seat where
  row : Nat
  column : Nat

def is_valid_seat (c : Classroom) (s : Seat) : Prop :=
  s.row ≤ c.rows ∧ s.column ≤ c.columns

theorem xiao_hua_seat (c : Classroom) (s : Seat) :
  c.rows = 7 →
  c.columns = 8 →
  is_valid_seat c s →
  s.row = 5 →
  s.column = 2 →
  s = ⟨5, 2⟩ := by
  sorry

end xiao_hua_seat_l3894_389462


namespace students_playing_both_football_and_tennis_l3894_389473

/-- Given a class of students, calculates the number of students playing both football and long tennis. -/
def students_playing_both (total : ℕ) (football : ℕ) (long_tennis : ℕ) (neither : ℕ) : ℕ :=
  football + long_tennis - (total - neither)

/-- Theorem: In a class of 36 students, where 26 play football, 20 play long tennis, and 7 play neither,
    the number of students who play both football and long tennis is 17. -/
theorem students_playing_both_football_and_tennis :
  students_playing_both 36 26 20 7 = 17 := by
  sorry

end students_playing_both_football_and_tennis_l3894_389473


namespace unfair_coin_expected_value_l3894_389451

/-- The expected value of an unfair coin flip -/
theorem unfair_coin_expected_value :
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let win_amount : ℚ := 4
  let lose_amount : ℚ := 9
  let expected_value := p_heads * win_amount - p_tails * lose_amount
  expected_value = -1/3 := by
sorry

end unfair_coin_expected_value_l3894_389451


namespace bob_candies_count_l3894_389430

-- Define Bob's items
def bob_chewing_gums : ℕ := 15
def bob_chocolate_bars : ℕ := 20
def bob_assorted_candies : ℕ := 15

-- Theorem to prove
theorem bob_candies_count : bob_assorted_candies = 15 := by
  sorry

end bob_candies_count_l3894_389430


namespace second_number_value_l3894_389421

theorem second_number_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b = 1.2 * a) (h4 : a / b = 5 / 6) : b = 6 := by
  sorry

end second_number_value_l3894_389421


namespace food_price_consumption_reduction_l3894_389429

theorem food_price_consumption_reduction (initial_price : ℝ) (h : initial_price > 0) :
  let price_increase_factor := 1.5
  let consumption_reduction_factor := 2/3
  initial_price * price_increase_factor * consumption_reduction_factor = initial_price :=
by sorry

end food_price_consumption_reduction_l3894_389429


namespace candy_jar_problem_l3894_389404

theorem candy_jar_problem (banana_jar grape_jar peanut_butter_jar : ℕ) : 
  banana_jar = 43 →
  grape_jar = banana_jar + 5 →
  peanut_butter_jar = 4 * grape_jar →
  peanut_butter_jar = 192 := by
sorry

end candy_jar_problem_l3894_389404


namespace expression_value_at_three_l3894_389411

theorem expression_value_at_three :
  let f (x : ℝ) := (x^2 - 5*x + 4) / (x - 4)
  f 3 = 2 := by sorry

end expression_value_at_three_l3894_389411


namespace maria_car_trip_l3894_389435

theorem maria_car_trip (D : ℝ) : 
  (D / 2 + (D / 2) / 4 + 150 = D) → D = 400 := by sorry

end maria_car_trip_l3894_389435


namespace days_before_reinforcement_l3894_389464

/-- Proves that the number of days before reinforcement arrived is 12 --/
theorem days_before_reinforcement 
  (initial_garrison : ℕ) 
  (initial_provision_days : ℕ) 
  (reinforcement : ℕ) 
  (remaining_provision_days : ℕ) 
  (h1 : initial_garrison = 1850)
  (h2 : initial_provision_days = 28)
  (h3 : reinforcement = 1110)
  (h4 : remaining_provision_days = 10) :
  (initial_garrison * initial_provision_days - 
   (initial_garrison + reinforcement) * remaining_provision_days) / initial_garrison = 12 :=
by sorry

end days_before_reinforcement_l3894_389464


namespace locus_of_centers_l3894_389445

/-- Circle C1 with equation x^2 + y^2 = 1 -/
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C2 with equation (x - 2)^2 + y^2 = 25 -/
def C2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 25

/-- A circle is externally tangent to C1 if the distance between their centers equals the sum of their radii -/
def externally_tangent_C1 (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C2 if the distance between their centers equals the difference of their radii -/
def internally_tangent_C2 (a b r : ℝ) : Prop := (a - 2)^2 + b^2 = (5 - r)^2

/-- The main theorem: the locus of centers (a,b) of circles externally tangent to C1 and internally tangent to C2 -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C1 a b r ∧ internally_tangent_C2 a b r) → 
  3 * a^2 + b^2 + 44 * a + 121 = 0 :=
sorry

end locus_of_centers_l3894_389445


namespace right_rectangular_prism_diagonal_ratio_bound_right_rectangular_prism_diagonal_ratio_bound_tight_l3894_389427

theorem right_rectangular_prism_diagonal_ratio_bound 
  (a b h d : ℝ) (ha : a > 0) (hb : b > 0) (hh : h > 0) 
  (hd : d^2 = a^2 + b^2 + h^2) : 
  (a + b + h) / d ≤ Real.sqrt 3 := by
sorry

theorem right_rectangular_prism_diagonal_ratio_bound_tight : 
  ∃ (a b h d : ℝ), a > 0 ∧ b > 0 ∧ h > 0 ∧ d^2 = a^2 + b^2 + h^2 ∧ 
  (a + b + h) / d = Real.sqrt 3 := by
sorry

end right_rectangular_prism_diagonal_ratio_bound_right_rectangular_prism_diagonal_ratio_bound_tight_l3894_389427


namespace ramanujan_identity_l3894_389428

theorem ramanujan_identity : ∃ (p q r p₁ q₁ r₁ : ℕ), 
  p ≠ q ∧ p ≠ r ∧ p ≠ p₁ ∧ p ≠ q₁ ∧ p ≠ r₁ ∧
  q ≠ r ∧ q ≠ p₁ ∧ q ≠ q₁ ∧ q ≠ r₁ ∧
  r ≠ p₁ ∧ r ≠ q₁ ∧ r ≠ r₁ ∧
  p₁ ≠ q₁ ∧ p₁ ≠ r₁ ∧
  q₁ ≠ r₁ ∧
  p^2 + q^2 + r^2 = p₁^2 + q₁^2 + r₁^2 ∧
  p^4 + q^4 + r^4 = p₁^4 + q₁^4 + r₁^4 := by
  sorry

end ramanujan_identity_l3894_389428


namespace expand_expression_l3894_389493

theorem expand_expression (x : ℝ) : (5 * x^2 - 3) * 4 * x^3 = 20 * x^5 - 12 * x^3 := by
  sorry

end expand_expression_l3894_389493


namespace sin_150_degrees_l3894_389434

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end sin_150_degrees_l3894_389434


namespace marathon_practice_distance_l3894_389432

/-- Calculates the total distance run given the number of days and miles per day -/
def total_distance (days : ℕ) (miles_per_day : ℕ) : ℕ :=
  days * miles_per_day

/-- Proves that running 8 miles for 9 days results in a total of 72 miles -/
theorem marathon_practice_distance :
  total_distance 9 8 = 72 := by
  sorry

end marathon_practice_distance_l3894_389432


namespace min_value_2x_plus_y_l3894_389492

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - 2*x*y = 0) :
  ∀ z, z = 2*x + y → z ≥ 9/2 :=
sorry

end min_value_2x_plus_y_l3894_389492


namespace rectangle_x_value_l3894_389423

/-- A rectangle in a rectangular coordinate system with given properties -/
structure Rectangle where
  x : ℝ
  area : ℝ
  h1 : area = 90

/-- The x-coordinate of the first and last vertices of the rectangle is -9 -/
theorem rectangle_x_value (rect : Rectangle) : rect.x = -9 := by
  sorry

end rectangle_x_value_l3894_389423


namespace smallest_satisfying_number_l3894_389456

def is_smallest_satisfying_number (n : ℕ) : Prop :=
  (∀ m < n, ∃ p, Nat.Prime p ∧ m % (p - 1) = 0 ∧ m % p ≠ 0) ∧
  (∀ p, Nat.Prime p → n % (p - 1) = 0 → n % p = 0)

theorem smallest_satisfying_number :
  is_smallest_satisfying_number 1806 :=
sorry

end smallest_satisfying_number_l3894_389456


namespace farm_cows_l3894_389494

theorem farm_cows (milk_per_week : ℝ) (total_milk : ℝ) (num_weeks : ℕ) :
  milk_per_week = 108 →
  total_milk = 2160 →
  num_weeks = 5 →
  (total_milk / (milk_per_week / 6 * num_weeks) : ℝ) = 24 :=
by sorry

end farm_cows_l3894_389494


namespace consecutive_blue_red_probability_l3894_389415

def num_green : ℕ := 4
def num_blue : ℕ := 3
def num_red : ℕ := 5
def total_chips : ℕ := num_green + num_blue + num_red

def probability_consecutive_blue_red : ℚ :=
  (num_blue.factorial * num_red.factorial * (Nat.choose (num_green + 2) 2)) /
  total_chips.factorial

theorem consecutive_blue_red_probability :
  probability_consecutive_blue_red = 1 / 44352 := by
  sorry

end consecutive_blue_red_probability_l3894_389415


namespace smallest_class_size_l3894_389431

theorem smallest_class_size (n : ℕ) : 
  n > 9 ∧ 
  (∃ (a b c d e : ℕ), 
    a = n ∧ b = n ∧ c = n ∧ d = n + 2 ∧ e = n + 3 ∧
    a + b + c + d + e > 50) →
  (∀ m : ℕ, m > 9 ∧ 
    (∃ (a b c d e : ℕ), 
      a = m ∧ b = m ∧ c = m ∧ d = m + 2 ∧ e = m + 3 ∧
      a + b + c + d + e > 50) →
    5 * n + 5 ≤ 5 * m + 5) →
  5 * n + 5 = 55 := by
sorry

end smallest_class_size_l3894_389431


namespace quadratic_inequality_always_nonnegative_l3894_389458

theorem quadratic_inequality_always_nonnegative : ∀ x : ℝ, x^2 - x + 1 ≥ 0 := by
  sorry

end quadratic_inequality_always_nonnegative_l3894_389458


namespace value_in_scientific_notation_l3894_389496

/-- Represents 1 billion -/
def billion : ℝ := 10^9

/-- The value we want to express in scientific notation -/
def value : ℝ := 45 * billion

/-- The scientific notation representation of the value -/
def scientific_notation : ℝ := 4.5 * 10^9

theorem value_in_scientific_notation : value = scientific_notation := by
  sorry

end value_in_scientific_notation_l3894_389496


namespace house_representatives_difference_l3894_389476

theorem house_representatives_difference (total : Nat) (democrats : Nat) :
  total = 434 →
  democrats = 202 →
  democrats < total - democrats →
  total - 2 * democrats = 30 := by
sorry

end house_representatives_difference_l3894_389476


namespace profit_percentage_is_30_percent_l3894_389443

def cost_per_dog : ℝ := 1000
def selling_price_two_dogs : ℝ := 2600

theorem profit_percentage_is_30_percent :
  let cost_two_dogs := 2 * cost_per_dog
  let profit := selling_price_two_dogs - cost_two_dogs
  let profit_percentage := (profit / cost_two_dogs) * 100
  profit_percentage = 30 := by sorry

end profit_percentage_is_30_percent_l3894_389443


namespace crystal_lake_trail_length_l3894_389440

/-- Represents the Crystal Lake Trail hike --/
structure CrystalLakeTrail where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- Conditions of the Crystal Lake Trail hike --/
def hikingConditions (hike : CrystalLakeTrail) : Prop :=
  hike.day1 + hike.day2 = 28 ∧
  (hike.day2 + hike.day3) / 2 = 15 ∧
  hike.day3 + hike.day4 + hike.day5 = 42 ∧
  hike.day1 + hike.day4 = 30

/-- Theorem stating that the total length of the Crystal Lake Trail is 70 miles --/
theorem crystal_lake_trail_length 
  (hike : CrystalLakeTrail) 
  (h : hikingConditions hike) : 
  hike.day1 + hike.day2 + hike.day3 + hike.day4 + hike.day5 = 70 := by
  sorry

end crystal_lake_trail_length_l3894_389440


namespace division_remainder_l3894_389439

theorem division_remainder : ∃ q : ℕ, 1234567 = 137 * q + 102 ∧ 102 < 137 := by
  sorry

end division_remainder_l3894_389439


namespace binary_multiplication_theorem_l3894_389420

/-- Converts a list of binary digits to a natural number -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec to_bits (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_bits (m / 2)
  to_bits n

theorem binary_multiplication_theorem :
  let a := [true, true, false, true, true]  -- 11011₂
  let b := [true, true, true]               -- 111₂
  let result := [true, false, false, false, false, true, false, true]  -- 10000101₂
  binary_to_nat a * binary_to_nat b = binary_to_nat result := by
  sorry

#eval binary_to_nat [true, true, false, true, true]  -- Should output 27
#eval binary_to_nat [true, true, true]               -- Should output 7
#eval binary_to_nat [true, false, false, false, false, true, false, true]  -- Should output 133
#eval 27 * 7  -- Should output 189

end binary_multiplication_theorem_l3894_389420


namespace product_sum_fraction_equality_l3894_389475

theorem product_sum_fraction_equality : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end product_sum_fraction_equality_l3894_389475


namespace find_coefficient_a_l3894_389478

theorem find_coefficient_a (f' : ℝ → ℝ) (a : ℝ) :
  (∀ x, f' x = 2 * x^3 + a * x^2 + x) →
  f' 1 = 9 →
  a = 6 := by
sorry

end find_coefficient_a_l3894_389478


namespace at_least_one_geq_two_l3894_389449

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by sorry

end at_least_one_geq_two_l3894_389449


namespace greg_age_l3894_389447

/-- Given the ages and relationships of Cindy, Jan, Marcia, and Greg, prove Greg's age. -/
theorem greg_age (cindy_age : ℕ) (jan_age : ℕ) (marcia_age : ℕ) (greg_age : ℕ)
  (h1 : cindy_age = 5)
  (h2 : jan_age = cindy_age + 2)
  (h3 : marcia_age = 2 * jan_age)
  (h4 : greg_age = marcia_age + 2) :
  greg_age = 16 := by
  sorry

end greg_age_l3894_389447


namespace binary_to_decimal_101101_l3894_389455

theorem binary_to_decimal_101101 : 
  (1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 45 := by
  sorry

end binary_to_decimal_101101_l3894_389455


namespace spencer_walking_distance_l3894_389424

/-- The total distance walked by Spencer -/
def total_distance (initial_distance : ℝ) (first_segment : ℝ) (second_segment : ℝ) : ℝ :=
  first_segment + second_segment + initial_distance

/-- Theorem: Spencer's total walking distance is 1400 meters -/
theorem spencer_walking_distance :
  let initial_distance : ℝ := 1000
  let first_segment : ℝ := 200
  let second_segment : ℝ := 200
  total_distance initial_distance first_segment second_segment = 1400 := by
  sorry

#eval total_distance 1000 200 200

end spencer_walking_distance_l3894_389424


namespace smallest_3_4_cut_is_14_l3894_389459

/-- A positive integer n is m-cut if n-2 is divisible by m -/
def is_m_cut (n m : ℕ) : Prop :=
  n > 2 ∧ m > 2 ∧ (n - 2) % m = 0

/-- The smallest positive integer that is both 3-cut and 4-cut -/
def smallest_3_4_cut : ℕ := 14

/-- Theorem stating that 14 is the smallest positive integer that is both 3-cut and 4-cut -/
theorem smallest_3_4_cut_is_14 :
  (∀ n : ℕ, n < smallest_3_4_cut → ¬(is_m_cut n 3 ∧ is_m_cut n 4)) ∧
  (is_m_cut smallest_3_4_cut 3 ∧ is_m_cut smallest_3_4_cut 4) :=
by sorry

end smallest_3_4_cut_is_14_l3894_389459


namespace sum_of_integers_ending_in_3_is_11920_l3894_389471

/-- The sum of all integers between 100 and 500 which end in 3 -/
def sum_of_integers_ending_in_3 : ℕ :=
  let first_term := 103
  let last_term := 493
  let num_terms := (last_term - first_term) / 10 + 1
  num_terms * (first_term + last_term) / 2

theorem sum_of_integers_ending_in_3_is_11920 :
  sum_of_integers_ending_in_3 = 11920 := by
  sorry

end sum_of_integers_ending_in_3_is_11920_l3894_389471


namespace oldest_sibling_age_l3894_389479

theorem oldest_sibling_age (average_age : ℝ) (age1 age2 age3 : ℕ) :
  average_age = 9 ∧ age1 = 5 ∧ age2 = 8 ∧ age3 = 7 →
  ∃ (oldest_age : ℕ), (age1 + age2 + age3 + oldest_age) / 4 = average_age ∧ oldest_age = 16 :=
by sorry

end oldest_sibling_age_l3894_389479


namespace y_value_l3894_389402

theorem y_value : ∃ y : ℝ, 1.5 * y - 10 = 35 ∧ y = 30 := by
  sorry

end y_value_l3894_389402


namespace value_of_expression_l3894_389438

theorem value_of_expression (x y z : ℝ) 
  (eq1 : 2 * x + y - z = 7)
  (eq2 : x + 2 * y + z = 5)
  (eq3 : x - y + 2 * z = 3) :
  2 * x * y / 3 = 1.625 := by
  sorry

end value_of_expression_l3894_389438


namespace inequality_solution_set_l3894_389472

-- Define a monotonically increasing function on [0, +∞)
def monotone_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- Define the set of x that satisfies the inequality
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f (2 * x - 1) < f (1 / 3)}

-- Theorem statement
theorem inequality_solution_set 
  (f : ℝ → ℝ) 
  (h_monotone : monotone_increasing_on_nonneg f) :
  solution_set f = Set.Ici (1 / 2) ∩ Set.Iio (2 / 3) :=
sorry

end inequality_solution_set_l3894_389472


namespace parabola_equation_l3894_389436

/-- A parabola with vertex at the origin -/
structure Parabola where
  /-- The equation of the parabola in the form y^2 = kx -/
  k : ℝ
  /-- The focus of the parabola -/
  focus : ℝ × ℝ

/-- The condition that the focus is on the x-axis -/
def focus_on_x_axis (p : Parabola) : Prop :=
  p.focus.2 = 0

/-- The condition that a perpendicular line from the origin to a line passing through the focus has its foot at (2, 1) -/
def perpendicular_foot_condition (p : Parabola) : Prop :=
  ∃ (m : ℝ), m * p.focus.1 = p.focus.2 ∧ 2 * m = 1

/-- The theorem stating that if the two conditions are met, the parabola's equation is y^2 = 10x -/
theorem parabola_equation (p : Parabola) :
  focus_on_x_axis p → perpendicular_foot_condition p → p.k = 10 := by
  sorry

end parabola_equation_l3894_389436


namespace parabola_coefficient_l3894_389409

/-- A parabola passing through a specific point -/
def parabola_through_point (a : ℝ) (x y : ℝ) : Prop :=
  a ≠ 0 ∧ y = a * x^2

/-- Theorem: The parabola y = ax^2 passing through (2, -8) has a = -2 -/
theorem parabola_coefficient :
  ∀ a : ℝ, parabola_through_point a 2 (-8) → a = -2 := by
  sorry

end parabola_coefficient_l3894_389409


namespace expression_simplification_and_evaluation_l3894_389486

theorem expression_simplification_and_evaluation (a : ℤ) 
  (h1 : -2 < a) (h2 : a ≤ 2) (h3 : a ≠ 0) (h4 : a ≠ 1) :
  (a - (2 * a - 1) / a) / ((a - 1) / a) = a - 1 ∧
  (a = -1 ∨ a = 2) ∧
  (a = -1 → (a - (2 * a - 1) / a) / ((a - 1) / a) = -2) ∧
  (a = 2 → (a - (2 * a - 1) / a) / ((a - 1) / a) = 1) :=
by sorry

#check expression_simplification_and_evaluation

end expression_simplification_and_evaluation_l3894_389486


namespace common_term_value_l3894_389461

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a₁ : ℝ  -- First term
  a₂ : ℝ  -- Second term

/-- Represents a geometric progression -/
structure GeometricProgression where
  g₁ : ℝ  -- First term
  g₂ : ℝ  -- Second term

/-- Given arithmetic and geometric progressions, if there exists a common term, it is 37/3 -/
theorem common_term_value (x : ℝ) (ap : ArithmeticProgression) (gp : GeometricProgression) 
  (h_ap : ap.a₁ = 2*x - 3 ∧ ap.a₂ = 5*x - 11)
  (h_gp : gp.g₁ = x + 1 ∧ gp.g₂ = 2*x + 3)
  (h_common : ∃ t : ℝ, (∃ n : ℕ, t = ap.a₁ + (n - 1) * (ap.a₂ - ap.a₁)) ∧ 
                       (∃ m : ℕ, t = gp.g₁ * (gp.g₂ / gp.g₁) ^ (m - 1))) :
  ∃ t : ℝ, t = 37/3 ∧ (∃ n : ℕ, t = ap.a₁ + (n - 1) * (ap.a₂ - ap.a₁)) ∧ 
               (∃ m : ℕ, t = gp.g₁ * (gp.g₂ / gp.g₁) ^ (m - 1)) := by
  sorry

end common_term_value_l3894_389461


namespace custom_mul_four_three_l3894_389489

/-- Custom multiplication operation -/
def custom_mul (a b : ℤ) : ℤ := a^2 + a*b + a - b^2

/-- Theorem stating that 4 * 3 = 23 under the custom multiplication -/
theorem custom_mul_four_three : custom_mul 4 3 = 23 := by
  sorry

end custom_mul_four_three_l3894_389489


namespace impossible_closed_line_1989_sticks_l3894_389487

theorem impossible_closed_line_1989_sticks : ¬ ∃ (a b : ℕ), 2 * (a + b) = 1989 := by
  sorry

end impossible_closed_line_1989_sticks_l3894_389487


namespace sol_earnings_l3894_389422

/-- Calculates the earnings from selling candy bars over a week -/
def candy_bar_earnings (initial_sales : ℕ) (daily_increase : ℕ) (days : ℕ) (price_cents : ℕ) : ℚ :=
  let total_bars := (List.range days).map (λ i => initial_sales + i * daily_increase) |>.sum
  (total_bars * price_cents : ℚ) / 100

/-- Theorem stating that Sol's earnings from selling candy bars over a week is $12.00 -/
theorem sol_earnings : candy_bar_earnings 10 4 6 10 = 12 := by
  sorry

end sol_earnings_l3894_389422


namespace find_k_l3894_389466

theorem find_k (k : ℝ) (h : 24 / k = 4) : k = 6 := by
  sorry

end find_k_l3894_389466


namespace cost_to_feed_chickens_is_60_l3894_389426

/-- Calculates the cost to feed chickens given the total number of birds and the ratio of bird types -/
def cost_to_feed_chickens (total_birds : ℕ) (duck_ratio parrot_ratio chicken_ratio : ℕ) (chicken_feed_cost : ℚ) : ℚ :=
  let total_ratio := duck_ratio + parrot_ratio + chicken_ratio
  let birds_per_ratio := total_birds / total_ratio
  let num_chickens := birds_per_ratio * chicken_ratio
  (num_chickens : ℚ) * chicken_feed_cost

/-- Theorem stating that with given conditions, the cost to feed chickens is $60 -/
theorem cost_to_feed_chickens_is_60 :
  cost_to_feed_chickens 60 2 3 5 2 = 60 := by
  sorry

end cost_to_feed_chickens_is_60_l3894_389426


namespace rice_and_husk_division_l3894_389400

/-- Calculates the approximate amount of husks in a batch of grain --/
def calculate_husks (total_grain : ℕ) (sample_husks : ℕ) (sample_total : ℕ) : ℕ :=
  (total_grain * sample_husks) / sample_total

/-- The Rice and Husk Division problem from "The Nine Chapters on the Mathematical Art" --/
theorem rice_and_husk_division :
  let total_grain : ℕ := 1524
  let sample_husks : ℕ := 28
  let sample_total : ℕ := 254
  calculate_husks total_grain sample_husks sample_total = 168 := by
  sorry

#eval calculate_husks 1524 28 254

end rice_and_husk_division_l3894_389400


namespace least_positive_integer_to_multiple_of_five_l3894_389414

theorem least_positive_integer_to_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (525 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (525 + m) % 5 = 0 → n ≤ m :=
by sorry

end least_positive_integer_to_multiple_of_five_l3894_389414


namespace union_of_A_and_B_l3894_389468

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 5 }
def B : Set ℝ := { x | 3 < x ∧ x < 9 }

theorem union_of_A_and_B : A ∪ B = { x | -1 ≤ x ∧ x < 9 } := by
  sorry

end union_of_A_and_B_l3894_389468


namespace train_length_l3894_389481

/-- Calculates the length of a train given its speed and the time it takes to cross a platform of known length. -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * 1000 / 3600 →
  platform_length = 250 →
  crossing_time = 15 →
  (train_speed * crossing_time) - platform_length = 50 :=
by sorry

end train_length_l3894_389481


namespace octahedron_projection_area_l3894_389485

/-- A regular octahedron -/
structure RegularOctahedron where
  -- Add necessary fields here

/-- The area of a face of a regular octahedron -/
def face_area (o : RegularOctahedron) : ℝ :=
  sorry

/-- The area of the projection of one face onto the opposite face -/
def projection_area (o : RegularOctahedron) : ℝ :=
  sorry

/-- 
  In a regular octahedron, the perpendicular projection of one face 
  onto the plane of the opposite face covers 2/3 of the area of the opposite face
-/
theorem octahedron_projection_area (o : RegularOctahedron) :
  projection_area o = (2 / 3) * face_area o :=
sorry

end octahedron_projection_area_l3894_389485


namespace contractor_wage_l3894_389465

/-- Contractor's wage problem -/
theorem contractor_wage
  (total_days : ℕ)
  (absent_days : ℕ)
  (daily_fine : ℚ)
  (total_amount : ℚ)
  (h1 : total_days = 30)
  (h2 : absent_days = 10)
  (h3 : daily_fine = 7.5)
  (h4 : total_amount = 425)
  : ∃ (daily_wage : ℚ),
    daily_wage * (total_days - absent_days : ℚ) - daily_fine * absent_days = total_amount ∧
    daily_wage = 25 := by
  sorry

end contractor_wage_l3894_389465


namespace C₂_fixed_point_l3894_389410

/-- Parabola C₁ with vertex (√2-1, 1) and focus (√2-3/4, 1) -/
def C₁ : Set (ℝ × ℝ) :=
  {p | (p.2 - 1)^2 = 2 * (p.1 - (Real.sqrt 2 - 1))}

/-- Parabola C₂ with equation y² - ay + x + 2b = 0 -/
def C₂ (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2^2 - a * p.2 + p.1 + 2 * b = 0}

/-- The tangent line to C₁ at point p -/
def tangentC₁ (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q | 2 * p.2 * q.2 - q.1 - 2 * (p.2 + 1) = 0}

/-- The tangent line to C₂ at point p -/
def tangentC₂ (a : ℝ) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q | (2 * p.2 - a) * q.2 + q.1 - a * p.2 + p.1 + 4 * ((a - 2) * p.2 - p.1 - Real.sqrt 2) / 4 = 0}

/-- Perpendicularity condition for tangent lines -/
def perpendicularTangents (p : ℝ × ℝ) (a : ℝ) : Prop :=
  (p.2 - 1) * (2 * p.2 - a) = -1

theorem C₂_fixed_point (a b : ℝ) :
  (∃ p, p ∈ C₁ ∧ p ∈ C₂ a b ∧ perpendicularTangents p a) →
  (Real.sqrt 2 - 1/2, 1) ∈ C₂ a b := by
  sorry

end C₂_fixed_point_l3894_389410


namespace dog_treat_cost_theorem_l3894_389453

/-- Calculates the total cost of dog treats for a month -/
def total_treat_cost (treats_per_day : ℕ) (cost_per_treat : ℚ) (days_in_month : ℕ) : ℚ :=
  (treats_per_day * days_in_month : ℚ) * cost_per_treat

/-- Proves that the total cost of dog treats for a month with given parameters is $6 -/
theorem dog_treat_cost_theorem (treats_per_day : ℕ) (cost_per_treat : ℚ) (days_in_month : ℕ)
  (h1 : treats_per_day = 2)
  (h2 : cost_per_treat = 1/10)
  (h3 : days_in_month = 30) :
  total_treat_cost treats_per_day cost_per_treat days_in_month = 6 := by
  sorry

end dog_treat_cost_theorem_l3894_389453


namespace largest_odd_equal_cost_l3894_389412

/-- Calculates the sum of digits in decimal representation -/
def sumDigitsDecimal (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumDigitsDecimal (n / 10)

/-- Calculates the sum of digits in binary representation with two trailing zeros -/
def sumDigitsBinary (n : Nat) : Nat :=
  if n < 4 then 0 else (n % 2) + sumDigitsBinary (n / 2)

/-- Checks if a number is odd -/
def isOdd (n : Nat) : Prop := n % 2 = 1

/-- Theorem statement -/
theorem largest_odd_equal_cost :
  ∃ (n : Nat), n < 2000 ∧ isOdd n ∧
    sumDigitsDecimal n = sumDigitsBinary n ∧
    ∀ (m : Nat), m < 2000 ∧ isOdd m ∧ sumDigitsDecimal m = sumDigitsBinary m → m ≤ n :=
by
  -- Proof goes here
  sorry

end largest_odd_equal_cost_l3894_389412


namespace car_speed_before_servicing_l3894_389497

/-- The speed of a car before and after servicing -/
theorem car_speed_before_servicing (speed_serviced : ℝ) (time_serviced time_not_serviced : ℝ) 
  (h1 : speed_serviced = 90)
  (h2 : time_serviced = 3)
  (h3 : time_not_serviced = 6)
  (h4 : speed_serviced * time_serviced = speed_not_serviced * time_not_serviced) :
  speed_not_serviced = 45 := by
  sorry


end car_speed_before_servicing_l3894_389497


namespace range_of_positive_integers_in_list_l3894_389425

def consecutive_integers (start : Int) (n : Nat) : List Int :=
  List.range n |>.map (λ i => start + i)

def positive_integers (list : List Int) : List Int :=
  list.filter (λ x => x > 0)

def range_of_list (list : List Int) : Int :=
  list.maximum?.getD 0 - list.minimum?.getD 0

theorem range_of_positive_integers_in_list (k : List Int) :
  k = consecutive_integers (-4) 14 →
  range_of_list (positive_integers k) = 8 := by
  sorry

end range_of_positive_integers_in_list_l3894_389425


namespace solve_equation_l3894_389467

theorem solve_equation (x : ℝ) : (2 * x + 7) / 6 = 13 → x = 35.5 := by
  sorry

end solve_equation_l3894_389467


namespace r_div_p_equals_1100_l3894_389413

/-- The number of cards in the box -/
def total_cards : ℕ := 60

/-- The number of different numbers on the cards -/
def distinct_numbers : ℕ := 12

/-- The number of cards for each number -/
def cards_per_number : ℕ := 5

/-- The number of cards drawn -/
def drawn_cards : ℕ := 5

/-- The probability of drawing five cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / Nat.choose total_cards drawn_cards

/-- The probability of drawing three cards with one number and two with another -/
def r : ℚ := (13200 : ℚ) / Nat.choose total_cards drawn_cards

/-- Theorem stating the ratio of r to p -/
theorem r_div_p_equals_1100 : r / p = 1100 := by sorry

end r_div_p_equals_1100_l3894_389413


namespace angle_A_magnitude_max_area_l3894_389483

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a * Real.cos t.C = (2 * t.b - Real.sqrt 3 * t.c) * Real.cos t.A

-- Theorem for part I
theorem angle_A_magnitude (t : Triangle) (h : given_condition t) : t.A = π / 6 :=
sorry

-- Theorem for part II
theorem max_area (t : Triangle) (h1 : given_condition t) (h2 : t.a = 2) :
  ∃ (area : ℝ), area ≤ 2 + Real.sqrt 3 ∧
  ∀ (other_area : ℝ), (∃ (t' : Triangle), t'.a = 2 ∧ given_condition t' ∧ 
    other_area = (1 / 2) * t'.b * t'.c * Real.sin t'.A) → other_area ≤ area :=
sorry

end angle_A_magnitude_max_area_l3894_389483


namespace cricket_score_problem_l3894_389491

theorem cricket_score_problem :
  ∀ (a b c d e : ℕ),
    -- Average score is 36
    a + b + c + d + e = 36 * 5 →
    -- D scored 5 more than E
    d = e + 5 →
    -- E scored 8 fewer than A
    e = a - 8 →
    -- B scored as many as D and E combined
    b = d + e →
    -- E scored 20 runs
    e = 20 →
    -- Prove that B and C scored 107 runs between them
    b + c = 107 := by
  sorry

end cricket_score_problem_l3894_389491


namespace isosceles_right_triangle_area_l3894_389416

theorem isosceles_right_triangle_area (side_length : ℝ) : 
  side_length = 12 →
  ∃ (r s : ℝ), 
    r > 0 ∧ s > 0 ∧
    2 * (r ^ 2 + s ^ 2) = side_length ^ 2 ∧
    4 * (r ^ 2 / 2) = 72 :=
by sorry

end isosceles_right_triangle_area_l3894_389416


namespace marble_probability_l3894_389495

/-- Represents a box of marbles -/
structure Box where
  gold : Nat
  black : Nat

/-- The probability of selecting a gold marble from a box -/
def prob_gold (b : Box) : Rat :=
  b.gold / (b.gold + b.black)

/-- The probability of selecting a black marble from a box -/
def prob_black (b : Box) : Rat :=
  b.black / (b.gold + b.black)

/-- The initial state of the boxes -/
def initial_boxes : List Box :=
  [⟨1, 1⟩, ⟨1, 2⟩, ⟨1, 3⟩]

/-- The probability of the final outcome after the marble movements -/
def final_probability : Rat :=
  let box1 := initial_boxes[0]
  let box2 := initial_boxes[1]
  let box3 := initial_boxes[2]

  let prob_gold_to_box2 := prob_gold box1 * prob_gold (⟨box2.gold + 1, box2.black⟩) +
                           prob_black box1 * prob_gold box2
  
  let prob_black_to_box3 := 1 - prob_gold_to_box2

  prob_gold_to_box2 * prob_gold (⟨box3.gold + 1, box3.black⟩) +
  prob_black_to_box3 * prob_gold box3

theorem marble_probability :
  final_probability = 11 / 40 := by sorry

end marble_probability_l3894_389495


namespace tangent_line_equation_l3894_389452

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Define the point of tangency
def point : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem tangent_line_equation :
  let (x₀, y₀) := point
  let m := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x - y - 1 = 0) := by
  sorry

end tangent_line_equation_l3894_389452


namespace final_answer_calculation_l3894_389490

theorem final_answer_calculation (chosen_number : ℤ) (h : chosen_number = 848) : 
  (chosen_number / 8 : ℚ) - 100 = 6 := by
  sorry

end final_answer_calculation_l3894_389490


namespace savings_increase_percentage_l3894_389408

/-- Represents the financial situation of a man over two years --/
structure FinancialSituation where
  /-- Income in the first year --/
  income : ℝ
  /-- Savings rate in the first year (as a decimal) --/
  savingsRate : ℝ
  /-- Income increase rate in the second year (as a decimal) --/
  incomeIncreaseRate : ℝ

/-- Theorem stating the increase in savings percentage --/
theorem savings_increase_percentage (fs : FinancialSituation)
    (h1 : fs.savingsRate = 0.2)
    (h2 : fs.incomeIncreaseRate = 0.2)
    (h3 : fs.income > 0)
    (h4 : fs.income * (2 - fs.savingsRate) = 
          fs.income * (1 + fs.incomeIncreaseRate) * (1 - fs.savingsRate) + 
          fs.income * (1 - fs.savingsRate)) :
    (fs.income * (1 + fs.incomeIncreaseRate) * fs.savingsRate - 
     fs.income * fs.savingsRate) / 
    (fs.income * fs.savingsRate) = 1 := by
  sorry

#check savings_increase_percentage

end savings_increase_percentage_l3894_389408


namespace minimum_value_implies_m_l3894_389401

/-- If the function f(x) = x^2 - 2x + m has a minimum value of -2 on the interval [2, +∞),
    then m = -2. -/
theorem minimum_value_implies_m (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = x^2 - 2*x + m) →
  (∀ x ≥ 2, f x ≥ -2) →
  (∃ x ≥ 2, f x = -2) →
  m = -2 := by
sorry

end minimum_value_implies_m_l3894_389401


namespace reciprocal_sum_of_quadratic_roots_l3894_389441

theorem reciprocal_sum_of_quadratic_roots (α β : ℝ) : 
  (∃ r s : ℝ, 7 * r^2 - 8 * r + 6 = 0 ∧ 
               7 * s^2 - 8 * s + 6 = 0 ∧ 
               α = 1 / r ∧ 
               β = 1 / s) → 
  α + β = 4 / 3 := by
sorry

end reciprocal_sum_of_quadratic_roots_l3894_389441


namespace cos_2alpha_on_unit_circle_l3894_389480

theorem cos_2alpha_on_unit_circle (α : Real) :
  (Real.cos α = -Real.sqrt 5 / 5 ∧ Real.sin α = 2 * Real.sqrt 5 / 5) →
  Real.cos (2 * α) = -3 / 5 := by
  sorry

end cos_2alpha_on_unit_circle_l3894_389480


namespace min_removal_for_given_structure_l3894_389460

/-- Represents the structure of triangles made with toothpicks -/
structure TriangleStructure where
  totalToothpicks : ℕ
  baseTriangles : ℕ
  rows : ℕ

/-- Calculates the number of toothpicks needed to be removed to eliminate all triangles -/
def minRemovalCount (ts : TriangleStructure) : ℕ :=
  ts.rows

/-- Theorem stating that for the given structure, 5 toothpicks need to be removed -/
theorem min_removal_for_given_structure :
  let ts : TriangleStructure := {
    totalToothpicks := 50,
    baseTriangles := 5,
    rows := 5
  }
  minRemovalCount ts = 5 := by
  sorry

#check min_removal_for_given_structure

end min_removal_for_given_structure_l3894_389460


namespace nonreal_cube_root_sum_l3894_389442

/-- Given that ω is a nonreal root of x^3 = 1, prove that 
    (2 - 2ω + 2ω^2)^3 + (2 + 2ω - 2ω^2)^3 = 0 -/
theorem nonreal_cube_root_sum (ω : ℂ) 
  (h1 : ω^3 = 1) 
  (h2 : ω ≠ 1) : 
  (2 - 2*ω + 2*ω^2)^3 + (2 + 2*ω - 2*ω^2)^3 = 0 := by
  sorry

end nonreal_cube_root_sum_l3894_389442


namespace data_set_property_l3894_389444

theorem data_set_property (m n : ℝ) : 
  (m + n + 9 + 8 + 10) / 5 = 9 →
  ((m^2 + n^2 + 9^2 + 8^2 + 10^2) / 5) - 9^2 = 2 →
  |m - n| = 4 :=
by sorry

end data_set_property_l3894_389444


namespace smallest_n_candies_l3894_389469

theorem smallest_n_candies : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n + 6) % 7 = 0 ∧ 
  (n - 9) % 4 = 0 ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m + 6) % 7 = 0 ∧ (m - 9) % 4 = 0 → n ≤ m) ∧
  n = 113 := by
sorry

end smallest_n_candies_l3894_389469


namespace coin_collection_dime_difference_l3894_389499

theorem coin_collection_dime_difference :
  ∀ (nickels dimes quarters : ℕ),
  nickels + dimes + quarters = 120 →
  5 * nickels + 10 * dimes + 25 * quarters = 1265 →
  quarters ≥ 10 →
  ∃ (min_dimes max_dimes : ℕ),
    (∀ d : ℕ, 
      nickels + d + quarters = 120 ∧ 
      5 * nickels + 10 * d + 25 * quarters = 1265 →
      min_dimes ≤ d ∧ d ≤ max_dimes) ∧
    max_dimes - min_dimes = 92 :=
by sorry

end coin_collection_dime_difference_l3894_389499


namespace halloween_candy_count_l3894_389406

/-- Calculates the final candy count given initial count, eaten count, and received count. -/
def finalCandyCount (initial eaten received : ℕ) : ℕ :=
  initial - eaten + received

/-- Theorem stating that given the specific values from the problem, 
    the final candy count is 62. -/
theorem halloween_candy_count : 
  finalCandyCount 47 25 40 = 62 := by
  sorry

end halloween_candy_count_l3894_389406


namespace max_profit_theorem_l3894_389417

/-- Represents the online store's sales and profit model -/
structure OnlineStore where
  initialPrice : ℕ
  initialSales : ℕ
  cost : ℕ
  salesIncrease : ℕ
  priceReduction : ℕ

/-- Calculates the monthly sales volume based on the price -/
def monthlySales (store : OnlineStore) (price : ℕ) : ℤ :=
  store.initialSales + store.salesIncrease * (store.initialPrice - price)

/-- Calculates the monthly profit based on the price -/
def monthlyProfit (store : OnlineStore) (price : ℕ) : ℤ :=
  (price - store.cost) * (monthlySales store price)

/-- Theorem stating the maximum profit and optimal price reduction -/
theorem max_profit_theorem (store : OnlineStore) :
  store.initialPrice = 80 ∧
  store.initialSales = 100 ∧
  store.cost = 40 ∧
  store.salesIncrease = 5 →
  ∃ (optimalReduction : ℕ),
    optimalReduction = 10 ∧
    monthlyProfit store (store.initialPrice - optimalReduction) = 4500 ∧
    ∀ (price : ℕ), monthlyProfit store price ≤ 4500 :=
by sorry

#check max_profit_theorem

end max_profit_theorem_l3894_389417


namespace fixed_point_parabola_l3894_389477

theorem fixed_point_parabola (k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 9 * x^2 + 3 * k * x - 6 * k
  f 2 = 36 := by
sorry

end fixed_point_parabola_l3894_389477


namespace mary_next_birthday_l3894_389446

/-- Represents the ages of Mary, Sally, and Danielle -/
structure Ages where
  mary : ℝ
  sally : ℝ
  danielle : ℝ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.mary = 1.2 * ages.sally ∧
  ages.sally = 0.6 * ages.danielle ∧
  ages.mary + ages.sally + ages.danielle = 23.2

/-- The theorem to be proved -/
theorem mary_next_birthday (ages : Ages) :
  problem_conditions ages → ⌊ages.mary⌋ + 1 = 8 :=
sorry

end mary_next_birthday_l3894_389446


namespace forever_alive_characterization_l3894_389403

/-- Represents the state of a cell: alive or dead -/
inductive CellState
| Alive
| Dead

/-- Represents a grid of cells -/
def Grid (m n : ℕ) := Fin m → Fin n → CellState

/-- Counts the number of alive neighbors for a cell -/
def countAliveNeighbors (grid : Grid m n) (i j : Fin m) : ℕ := sorry

/-- Updates the state of a single cell based on its neighbors -/
def updateCell (grid : Grid m n) (i j : Fin m) : CellState := sorry

/-- Updates the entire grid for one time step -/
def updateGrid (grid : Grid m n) : Grid m n := sorry

/-- Checks if a grid has at least one alive cell -/
def hasAliveCell (grid : Grid m n) : Prop := sorry

/-- Represents the existence of an initial configuration that stays alive forever -/
def existsForeverAliveConfig (m n : ℕ) : Prop :=
  ∃ (initial : Grid m n), ∀ (t : ℕ), hasAliveCell (Nat.iterate updateGrid t initial)

/-- The main theorem: characterizes the pairs (m, n) for which an eternally alive configuration exists -/
theorem forever_alive_characterization (m n : ℕ) :
  existsForeverAliveConfig m n ↔ (m, n) ≠ (1, 1) ∧ (m, n) ≠ (1, 3) ∧ (m, n) ≠ (3, 1) :=
sorry

end forever_alive_characterization_l3894_389403


namespace quadratic_roots_ratio_l3894_389405

theorem quadratic_roots_ratio (m : ℚ) : 
  (∃ r s : ℚ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 ∧ 
   r^2 + 9*r + m = 0 ∧ s^2 + 9*s + m = 0) → 
  m = 243/16 :=
by sorry

end quadratic_roots_ratio_l3894_389405
