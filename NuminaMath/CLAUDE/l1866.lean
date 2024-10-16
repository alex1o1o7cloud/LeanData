import Mathlib

namespace NUMINAMATH_CALUDE_remainder_theorem_l1866_186670

theorem remainder_theorem : 7 * 10^20 + 1^20 ≡ 8 [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1866_186670


namespace NUMINAMATH_CALUDE_theater_seat_count_l1866_186675

/-- Represents the number of seats in a theater with a specific seating arrangement. -/
def theaterSeats (firstRowSeats : ℕ) (lastRowSeats : ℕ) : ℕ :=
  let additionalRows := (lastRowSeats - firstRowSeats) / 2
  let totalRows := additionalRows + 1
  let sumAdditionalSeats := additionalRows * (2 + (lastRowSeats - firstRowSeats)) / 2
  firstRowSeats * totalRows + sumAdditionalSeats

/-- Theorem stating that a theater with the given seating arrangement has 3434 seats. -/
theorem theater_seat_count :
  theaterSeats 12 128 = 3434 :=
by sorry

end NUMINAMATH_CALUDE_theater_seat_count_l1866_186675


namespace NUMINAMATH_CALUDE_hash_2_4_5_l1866_186648

/-- The # operation for real numbers -/
def hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem stating that hash(2, 4, 5) equals -24 -/
theorem hash_2_4_5 : hash 2 4 5 = -24 := by sorry

end NUMINAMATH_CALUDE_hash_2_4_5_l1866_186648


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l1866_186684

theorem largest_prime_factor_of_1729 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1729 → q ≤ p := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l1866_186684


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l1866_186662

theorem polynomial_product_expansion (x : ℝ) :
  (x^3 - 3*x + 3) * (x^2 + 3*x + 3) = x^5 + 3*x^4 - 6*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l1866_186662


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l1866_186654

/-- The cost of an adult ticket to the circus -/
def adult_ticket_cost : ℕ := 2

/-- The number of people in Mary's group -/
def total_people : ℕ := 4

/-- The number of children in Mary's group -/
def num_children : ℕ := 3

/-- The cost of a child's ticket -/
def child_ticket_cost : ℕ := 1

/-- The total amount Mary paid -/
def total_paid : ℕ := 5

theorem circus_ticket_cost :
  adult_ticket_cost = total_paid - (num_children * child_ticket_cost) :=
by sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l1866_186654


namespace NUMINAMATH_CALUDE_store_sales_total_l1866_186636

/-- The total money made from selling DVD players and a washing machine -/
def total_money (dvd_price : ℕ) (dvd_quantity : ℕ) (washing_machine_price : ℕ) : ℕ :=
  dvd_price * dvd_quantity + washing_machine_price

/-- Theorem: The total money made from selling 8 DVD players at 240 yuan each
    and one washing machine at 898 yuan is equal to 240 * 8 + 898 yuan -/
theorem store_sales_total :
  total_money 240 8 898 = 240 * 8 + 898 := by
  sorry

end NUMINAMATH_CALUDE_store_sales_total_l1866_186636


namespace NUMINAMATH_CALUDE_part1_part2_l1866_186664

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 + (1 - a) * x - 1

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ x - 3) ↔ (0 ≤ a ∧ a ≤ 8) :=
sorry

-- Part 2
theorem part2 (a : ℝ) (h : a < 0) :
  (∀ x : ℝ, f a x < 0 ↔ 
    (a = -1 ∧ x ≠ 1) ∨
    (-1 < a ∧ a < 0 ∧ (x < 1 ∨ x > -1/a)) ∨
    (a < -1 ∧ (x < -1/a ∨ x > 1))) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l1866_186664


namespace NUMINAMATH_CALUDE_shop_length_is_18_l1866_186649

/-- Calculates the length of a shop given its monthly rent, width, and annual rent per square foot. -/
def shop_length (monthly_rent : ℕ) (width : ℕ) (annual_rent_per_sqft : ℕ) : ℕ :=
  (monthly_rent * 12) / (width * annual_rent_per_sqft)

/-- Proves that the length of the shop is 18 feet given the specified conditions. -/
theorem shop_length_is_18 :
  shop_length 1440 20 48 = 18 := by
  sorry

end NUMINAMATH_CALUDE_shop_length_is_18_l1866_186649


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1866_186601

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := i / (1 - i)
  Complex.im z = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1866_186601


namespace NUMINAMATH_CALUDE_waiter_tables_l1866_186668

theorem waiter_tables (customers_per_table : ℕ) (total_customers : ℕ) (h1 : customers_per_table = 8) (h2 : total_customers = 48) :
  total_customers / customers_per_table = 6 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tables_l1866_186668


namespace NUMINAMATH_CALUDE_min_moves_correct_l1866_186660

/-- The minimum number of moves in Bethan's grid game -/
def min_moves (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n^2 / 2 + n
  else
    (n^2 + 1) / 2

/-- Theorem stating the minimum number of moves in Bethan's grid game -/
theorem min_moves_correct (n : ℕ) (h : n > 0) :
  min_moves n = if n % 2 = 0 then n^2 / 2 + n else (n^2 + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_moves_correct_l1866_186660


namespace NUMINAMATH_CALUDE_bisecting_line_min_value_bisecting_line_min_value_achievable_l1866_186653

/-- A line that bisects the circumference of a circle --/
structure BisectingLine where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  bisects : ∀ (x y : ℝ), a * x + 2 * b * y - 2 = 0 → 
    (x^2 + y^2 - 4*x - 2*y - 8 = 0 → 
      ∃ (d : ℝ), d > 0 ∧ (x - 2)^2 + (y - 1)^2 = d^2)

/-- The theorem stating the minimum value of 1/a + 2/b --/
theorem bisecting_line_min_value (l : BisectingLine) :
  (1 / l.a + 2 / l.b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

/-- The theorem stating that the minimum value is achievable --/
theorem bisecting_line_min_value_achievable :
  ∃ (l : BisectingLine), 1 / l.a + 2 / l.b = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_line_min_value_bisecting_line_min_value_achievable_l1866_186653


namespace NUMINAMATH_CALUDE_h_in_terms_of_c_l1866_186613

theorem h_in_terms_of_c (a b c d e f g h : ℚ) : 
  8 = (6 / 100) * a →
  6 = (8 / 100) * b →
  9 = (5 / 100) * d →
  7 = (3 / 100) * e →
  c = b / a →
  f = d / a →
  g = e / b →
  h = f + g →
  h = (803 / 20) * c := by
sorry

end NUMINAMATH_CALUDE_h_in_terms_of_c_l1866_186613


namespace NUMINAMATH_CALUDE_quadratic_points_relationship_l1866_186657

/-- A quadratic function f(x) = -x² + 2x + c --/
def f (c : ℝ) (x : ℝ) : ℝ := -x^2 + 2*x + c

/-- The y-coordinate of a point (x, f(x)) on the graph of f --/
def y (c : ℝ) (x : ℝ) : ℝ := f c x

theorem quadratic_points_relationship (c : ℝ) :
  let y₁ := y c (-1)
  let y₂ := y c 3
  let y₃ := y c 5
  y₁ = y₂ ∧ y₂ > y₃ := by sorry

end NUMINAMATH_CALUDE_quadratic_points_relationship_l1866_186657


namespace NUMINAMATH_CALUDE_correct_sum_l1866_186612

theorem correct_sum (a b : ℕ) (h1 : a % 10 = 1) (h2 : b / 10 % 10 = 8) 
  (h3 : (a - 1 + 7) + (b - 80 + 30) = 1946) : a + b = 1990 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_sum_l1866_186612


namespace NUMINAMATH_CALUDE_power_sum_theorem_l1866_186610

theorem power_sum_theorem (k : ℕ) :
  (∃ (n m : ℕ), m ≥ 2 ∧ 3^k + 5^k = n^m) → k = 1 :=
by sorry

end NUMINAMATH_CALUDE_power_sum_theorem_l1866_186610


namespace NUMINAMATH_CALUDE_slope_determines_m_l1866_186642

/-- Given two points A(-2, m) and B(m, 4), if the slope of line AB is -2, then m = -8 -/
theorem slope_determines_m (m : ℝ) : 
  let A : ℝ × ℝ := (-2, m)
  let B : ℝ × ℝ := (m, 4)
  (4 - m) / (m - (-2)) = -2 → m = -8 := by
sorry

end NUMINAMATH_CALUDE_slope_determines_m_l1866_186642


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l1866_186606

/-- Given a hyperbola with equation (y-4)^2/32 - (x+3)^2/18 = 1,
    the distance between its vertices is 8√2. -/
theorem hyperbola_vertices_distance :
  let k : ℝ := 4
  let h : ℝ := -3
  let a_squared : ℝ := 32
  let b_squared : ℝ := 18
  let hyperbola_eq := fun (x y : ℝ) => (y - k)^2 / a_squared - (x - h)^2 / b_squared = 1
  let vertices_distance := 2 * Real.sqrt a_squared
  hyperbola_eq x y → vertices_distance = 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l1866_186606


namespace NUMINAMATH_CALUDE_largest_common_divisor_408_340_l1866_186691

theorem largest_common_divisor_408_340 : ∃ (n : ℕ), n = Nat.gcd 408 340 ∧ n = 68 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_408_340_l1866_186691


namespace NUMINAMATH_CALUDE_alyssa_next_year_games_l1866_186661

/-- Represents the number of soccer games Alyssa attended or plans to attend --/
structure SoccerGames where
  this_year : ℕ
  missed_this_year : ℕ
  last_year : ℕ
  total_planned : ℕ

/-- Calculates the number of games Alyssa plans to attend next year --/
def games_next_year (g : SoccerGames) : ℕ :=
  g.total_planned - (g.this_year + g.last_year)

/-- Theorem stating that for Alyssa's specific case, the number of games she plans to attend next year is 15 --/
theorem alyssa_next_year_games :
  let g : SoccerGames := {
    this_year := 11,
    missed_this_year := 12,
    last_year := 13,
    total_planned := 39
  }
  games_next_year g = 15 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_next_year_games_l1866_186661


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1866_186678

theorem sum_of_coefficients (a b c d e : ℤ) : 
  (∀ x : ℚ, 1000 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 92 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1866_186678


namespace NUMINAMATH_CALUDE_tourist_group_size_is_five_l1866_186667

/-- Represents the number of people in a tourist group satisfying given rooming conditions. -/
def tourist_group_size : ℕ :=
  let large_room_capacity : ℕ := 3
  let small_room_capacity : ℕ := 2
  let small_rooms_rented : ℕ := 1
  let total_people : ℕ := 5

  total_people

theorem tourist_group_size_is_five :
  let large_room_capacity : ℕ := 3
  let small_room_capacity : ℕ := 2
  let small_rooms_rented : ℕ := 1
  tourist_group_size = 5 ∧
  tourist_group_size % large_room_capacity = small_room_capacity ∧
  tourist_group_size ≥ small_room_capacity * small_rooms_rented :=
by
  sorry

#eval tourist_group_size

end NUMINAMATH_CALUDE_tourist_group_size_is_five_l1866_186667


namespace NUMINAMATH_CALUDE_vector_collinearity_l1866_186681

theorem vector_collinearity (m : ℝ) : 
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-1, 2)
  (∃ (k : ℝ), k ≠ 0 ∧ (m * a.1 + 4 * b.1, m * a.2 + 4 * b.2) = k • (a.1 - 2 * b.1, a.2 - 2 * b.2)) →
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1866_186681


namespace NUMINAMATH_CALUDE_absolute_value_equation_l1866_186602

theorem absolute_value_equation (x : ℝ) : |x - 1| = 2*x → x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l1866_186602


namespace NUMINAMATH_CALUDE_meatballs_on_plate_l1866_186659

theorem meatballs_on_plate (num_sons : ℕ) (fraction_eaten : ℚ) (meatballs_left : ℕ) : 
  num_sons = 3 → 
  fraction_eaten = 2/3 → 
  meatballs_left = 3 → 
  ∃ (initial_meatballs : ℕ), 
    initial_meatballs = 3 ∧ 
    (num_sons : ℚ) * ((1 : ℚ) - fraction_eaten) * initial_meatballs = meatballs_left :=
by sorry

end NUMINAMATH_CALUDE_meatballs_on_plate_l1866_186659


namespace NUMINAMATH_CALUDE_solve_for_q_l1866_186690

theorem solve_for_q (n m q : ℚ) 
  (h1 : (7 : ℚ) / 9 = n / 81)
  (h2 : (7 : ℚ) / 9 = (m + n) / 99)
  (h3 : (7 : ℚ) / 9 = (q - m) / 135) : 
  q = 119 := by sorry

end NUMINAMATH_CALUDE_solve_for_q_l1866_186690


namespace NUMINAMATH_CALUDE_range_of_m_l1866_186658

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) ↔ m > 1 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1866_186658


namespace NUMINAMATH_CALUDE_existence_of_special_number_l1866_186621

theorem existence_of_special_number : 
  ∃ (n : ℕ) (N : ℕ), n > 2 ∧ 
  N = 2 * 10^(n+1) - 9 ∧ 
  N % 1991 = 0 := by
sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l1866_186621


namespace NUMINAMATH_CALUDE_play_role_assignment_l1866_186639

theorem play_role_assignment (men : ℕ) (women : ℕ) : men = 7 ∧ women = 5 →
  (men * women * (Nat.choose (men + women - 2) 4)) = 7350 :=
by sorry

end NUMINAMATH_CALUDE_play_role_assignment_l1866_186639


namespace NUMINAMATH_CALUDE_five_g_speed_ratio_l1866_186672

/-- The ratio of 5G to 4G peak download speeds -/
def speed_ratio : ℝ := 7

/-- The size of the folder in MB -/
def folder_size : ℝ := 1400

/-- The 4G peak download speed in MB/s -/
def speed_4g : ℝ := 50

/-- The time difference in seconds for downloading the folder between 4G and 5G -/
def time_difference : ℝ := 24

theorem five_g_speed_ratio :
  folder_size / speed_4g - folder_size / (speed_ratio * speed_4g) = time_difference :=
sorry

end NUMINAMATH_CALUDE_five_g_speed_ratio_l1866_186672


namespace NUMINAMATH_CALUDE_candles_per_small_box_l1866_186631

theorem candles_per_small_box 
  (small_boxes_per_big_box : Nat) 
  (num_big_boxes : Nat) 
  (total_candles : Nat) :
  small_boxes_per_big_box = 4 →
  num_big_boxes = 50 →
  total_candles = 8000 →
  (total_candles / (small_boxes_per_big_box * num_big_boxes) : Nat) = 40 := by
  sorry

end NUMINAMATH_CALUDE_candles_per_small_box_l1866_186631


namespace NUMINAMATH_CALUDE_water_mixture_percentage_l1866_186651

theorem water_mixture_percentage (initial_volume : ℝ) (initial_water_percentage : ℝ) (added_water : ℝ) :
  initial_volume = 125 ∧
  initial_water_percentage = 0.2 ∧
  added_water = 8.333333333333334 →
  let initial_water := initial_volume * initial_water_percentage
  let new_water := initial_water + added_water
  let new_volume := initial_volume + added_water
  new_water / new_volume = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_water_mixture_percentage_l1866_186651


namespace NUMINAMATH_CALUDE_gold_coins_percentage_l1866_186646

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  total : ℝ
  beads : ℝ
  coins : ℝ
  silver_coins : ℝ
  gold_coins : ℝ

/-- The conditions of the urn as given in the problem -/
def urn_conditions (u : UrnComposition) : Prop :=
  u.total > 0 ∧
  u.beads + u.coins = u.total ∧
  u.silver_coins + u.gold_coins = u.coins ∧
  u.beads = 0.3 * u.total ∧
  u.silver_coins = 0.3 * u.coins

/-- The theorem stating that 49% of the objects in the urn are gold coins -/
theorem gold_coins_percentage (u : UrnComposition) 
  (h : urn_conditions u) : u.gold_coins / u.total = 0.49 := by
  sorry


end NUMINAMATH_CALUDE_gold_coins_percentage_l1866_186646


namespace NUMINAMATH_CALUDE_book_arrangement_problem_l1866_186680

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (math_copies : ℕ) (novel_copies : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial math_copies * Nat.factorial novel_copies)

/-- Theorem stating the number of arrangements for the given problem -/
theorem book_arrangement_problem :
  arrange_books 7 3 2 = 420 := by sorry

end NUMINAMATH_CALUDE_book_arrangement_problem_l1866_186680


namespace NUMINAMATH_CALUDE_cannot_determine_package_size_l1866_186634

/-- Represents the number of candies in a package -/
def CandiesPerPackage := ℕ

/-- Represents the state of candies on the desk -/
structure CandyPile :=
  (initial : ℕ)
  (added : ℕ)
  (final : ℕ)

/-- Given a candy pile state, it's not possible to determine the number of candies per package -/
theorem cannot_determine_package_size (pile : CandyPile) : 
  pile.initial = 6 → pile.added = 4 → pile.final = 10 → 
  ¬∃ (package_size : CandiesPerPackage), ∀ (other_size : CandiesPerPackage), package_size = other_size :=
by sorry

end NUMINAMATH_CALUDE_cannot_determine_package_size_l1866_186634


namespace NUMINAMATH_CALUDE_identity_function_unique_l1866_186698

theorem identity_function_unique (f : ℤ → ℤ) 
  (h1 : ∀ x : ℤ, f (f x) = x)
  (h2 : ∀ x y : ℤ, Odd (x + y) → f x + f y ≥ x + y) :
  ∀ x : ℤ, f x = x := by sorry

end NUMINAMATH_CALUDE_identity_function_unique_l1866_186698


namespace NUMINAMATH_CALUDE_train_passing_time_l1866_186689

/-- The time it takes for a train to pass a man moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 110 →
  train_speed = 80 * 1000 / 3600 →
  man_speed = 8 * 1000 / 3600 →
  (train_length / (train_speed + man_speed)) = 4.5 := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l1866_186689


namespace NUMINAMATH_CALUDE_intersection_points_l1866_186633

/-- The intersection points of two curves -/
theorem intersection_points 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (x : ℝ) := -a * x^3 + b * x + c
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = 0 ∧ f x₁ = g x₁) ∧ 
    (x₂ = -1 ∧ f x₂ = g x₂) ∧
    (∀ x, f x = g x → x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_l1866_186633


namespace NUMINAMATH_CALUDE_max_value_and_monotonicity_l1866_186693

noncomputable def f (x : ℝ) : ℝ := (3 * Real.log (x + 2) - Real.log (x - 2)) / 2

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x - 1) - f x

theorem max_value_and_monotonicity (h : ∀ x, f x ≥ f 4) :
  (∀ x ∈ Set.Icc 3 7, f x ≤ f 7) ∧
  (∀ a ≥ 1, Monotone (F a) ∧ ∀ a < 1, ¬Monotone (F a)) := by sorry

end NUMINAMATH_CALUDE_max_value_and_monotonicity_l1866_186693


namespace NUMINAMATH_CALUDE_f_has_minimum_at_negative_four_l1866_186622

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 8*x + 2

-- Theorem stating that f has a minimum at x = -4
theorem f_has_minimum_at_negative_four :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x₀ ≤ f x ∧ x₀ = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_f_has_minimum_at_negative_four_l1866_186622


namespace NUMINAMATH_CALUDE_ladder_top_velocity_l1866_186671

/-- Given a ladder sliding down a wall, this theorem calculates the velocity of the top end of the ladder. -/
theorem ladder_top_velocity (a l τ : ℝ) (h_positive : a > 0 ∧ l > 0 ∧ τ > 0) :
  let x := a * τ^2 / 2
  let v₁ := a * τ
  let α := Real.arcsin (a * τ^2 / (2 * l))
  let v₂ := a^2 * τ^3 / Real.sqrt (4 * l^2 - a^2 * τ^4)
  (x = a * τ^2 / 2) →
  (v₁ = a * τ) →
  (Real.sin α = a * τ^2 / (2 * l)) →
  (v₁ * Real.sin α = v₂ * Real.cos α) →
  v₂ = a^2 * τ^3 / Real.sqrt (4 * l^2 - a^2 * τ^4) :=
by sorry

end NUMINAMATH_CALUDE_ladder_top_velocity_l1866_186671


namespace NUMINAMATH_CALUDE_inverse_function_decomposition_l1866_186611

noncomputable section

def PeriodOn (h : ℝ → ℝ) (d : ℝ) : Prop :=
  ∀ x, h (x + d) = h x

def IsPeriodic (h : ℝ → ℝ) : Prop :=
  ∃ d ≠ 0, PeriodOn h d

def MutuallyInverse (f g : ℝ → ℝ) : Prop :=
  (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

theorem inverse_function_decomposition
  (f g : ℝ → ℝ)
  (h : ℝ → ℝ)
  (k : ℝ)
  (h_inv : MutuallyInverse f g)
  (h_periodic : IsPeriodic h)
  (h_decomp : ∀ x, f x = k * x + h x) :
  ∃ p : ℝ → ℝ, (IsPeriodic p) ∧ (∀ y, g y = (1/k) * y + p y) :=
sorry

end NUMINAMATH_CALUDE_inverse_function_decomposition_l1866_186611


namespace NUMINAMATH_CALUDE_train_length_proof_l1866_186615

/-- The length of each train in meters -/
def train_length : ℝ := 62.5

/-- The speed of the faster train in km/hr -/
def fast_train_speed : ℝ := 46

/-- The speed of the slower train in km/hr -/
def slow_train_speed : ℝ := 36

/-- The time taken for the faster train to completely pass the slower train in seconds -/
def overtake_time : ℝ := 45

theorem train_length_proof :
  let relative_speed := (fast_train_speed - slow_train_speed) * 1000 / 3600
  let distance_covered := relative_speed * overtake_time
  2 * train_length = distance_covered := by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l1866_186615


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1866_186697

/-- A rectangle with whole number dimensions and perimeter 40 has a maximum area of 100 -/
theorem rectangle_max_area :
  ∀ l w : ℕ,
  l + w = 20 →
  ∀ l' w' : ℕ,
  l' + w' = 20 →
  l * w ≤ 100 ∧
  (∃ l'' w'' : ℕ, l'' + w'' = 20 ∧ l'' * w'' = 100) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1866_186697


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1866_186600

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (x : ℚ)⁻¹ + (y : ℚ)⁻¹ = 15⁻¹) :
  (x : ℕ) + y ≥ 64 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1866_186600


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l1866_186677

/-- A trinomial ax^2 + bx + c is a perfect square if and only if b^2 - 4ac = 0 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  b^2 = 4*a*c

theorem perfect_square_trinomial_m_values :
  ∀ m : ℝ, (is_perfect_square_trinomial 1 (-2*(m+3)) 9) → (m = 0 ∨ m = -6) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l1866_186677


namespace NUMINAMATH_CALUDE_sin_6phi_l1866_186630

theorem sin_6phi (φ : ℝ) (h : Complex.exp (Complex.I * φ) = (3 + Complex.I * Real.sqrt 8) / 5) :
  Real.sin (6 * φ) = -198 * Real.sqrt 2 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_sin_6phi_l1866_186630


namespace NUMINAMATH_CALUDE_point_7_8_numbered_72_l1866_186683

def first_quadrant_numbering (x y : ℕ) : ℕ :=
  sorry

theorem point_7_8_numbered_72 :
  first_quadrant_numbering 7 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_point_7_8_numbered_72_l1866_186683


namespace NUMINAMATH_CALUDE_equation_solution_l1866_186655

theorem equation_solution (x : ℝ) (h : x ≠ -2) :
  (x^2 - x - 2) / (x + 2) = x + 3 ↔ x = -4/3 :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l1866_186655


namespace NUMINAMATH_CALUDE_smallest_with_12_divisors_l1866_186652

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- n has exactly 12 positive integer divisors -/
def has_12_divisors (n : ℕ+) : Prop := num_divisors n = 12

theorem smallest_with_12_divisors :
  ∃ (n : ℕ+), has_12_divisors n ∧ ∀ (m : ℕ+), has_12_divisors m → n ≤ m := by
  use 288
  sorry

end NUMINAMATH_CALUDE_smallest_with_12_divisors_l1866_186652


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l1866_186699

/-- A right triangle with sides 5, 12, and 13 containing an inscribed square -/
structure InscribedSquare where
  /-- Side length of the inscribed square -/
  t : ℝ
  /-- The inscribed square has side length t -/
  is_square : t > 0
  /-- The triangle is a right triangle with sides 5, 12, and 13 -/
  is_right_triangle : 5^2 + 12^2 = 13^2
  /-- The square is inscribed in the triangle -/
  is_inscribed : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 13 ∧ t / x = 5 / 13 ∧ t / y = 12 / 13

/-- The side length of the inscribed square is 780/169 -/
theorem inscribed_square_side_length (s : InscribedSquare) : s.t = 780 / 169 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l1866_186699


namespace NUMINAMATH_CALUDE_deal_or_no_deal_elimination_l1866_186623

/-- The total number of boxes in the game -/
def total_boxes : ℕ := 26

/-- The number of boxes containing at least $200,000 -/
def high_value_boxes : ℕ := 9

/-- The probability threshold for holding a high-value box -/
def probability_threshold : ℚ := 1/2

/-- The minimum number of boxes that need to be eliminated -/
def boxes_to_eliminate : ℕ := 9

theorem deal_or_no_deal_elimination :
  boxes_to_eliminate = total_boxes - high_value_boxes - (total_boxes - high_value_boxes) / 2 :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_elimination_l1866_186623


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l1866_186628

def is_in_second_quadrant (α : Real) : Prop :=
  ∃ k : Int, 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi

def is_in_first_or_third_quadrant (α : Real) : Prop :=
  ∃ n : Int, (2 * n * Real.pi + Real.pi / 4 < α ∧ α < 2 * n * Real.pi + Real.pi / 2) ∨
             ((2 * n + 1) * Real.pi + Real.pi / 4 < α ∧ α < (2 * n + 1) * Real.pi + Real.pi / 2)

theorem half_angle_quadrant (α : Real) :
  is_in_second_quadrant α → is_in_first_or_third_quadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l1866_186628


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1866_186665

theorem p_necessary_not_sufficient_for_q :
  (∀ a b : ℝ, a^2 + b^2 = 0 → a + b = 0) ∧
  (∃ a b : ℝ, a + b = 0 ∧ a^2 + b^2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1866_186665


namespace NUMINAMATH_CALUDE_sum_squares_of_roots_l1866_186643

theorem sum_squares_of_roots (x₁ x₂ : ℝ) : 
  (3 * x₁^2 + 4 * x₁ - 9 = 0) →
  (3 * x₂^2 + 4 * x₂ - 9 = 0) →
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 70/9) := by
sorry

end NUMINAMATH_CALUDE_sum_squares_of_roots_l1866_186643


namespace NUMINAMATH_CALUDE_peters_glass_purchase_l1866_186632

/-- Peter's glass purchase problem -/
theorem peters_glass_purchase
  (small_price : ℕ)
  (large_price : ℕ)
  (total_money : ℕ)
  (change : ℕ)
  (large_count : ℕ)
  (h1 : small_price = 3)
  (h2 : large_price = 5)
  (h3 : total_money = 50)
  (h4 : change = 1)
  (h5 : large_count = 5)
  : (total_money - change - large_count * large_price) / small_price = 8 := by
  sorry

end NUMINAMATH_CALUDE_peters_glass_purchase_l1866_186632


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l1866_186604

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4 / 7
  let a₂ : ℚ := 16 / 21
  let r : ℚ := a₂ / a₁
  r = 4 / 3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l1866_186604


namespace NUMINAMATH_CALUDE_corey_candy_count_l1866_186694

theorem corey_candy_count (total : ℕ) (difference : ℕ) (corey : ℕ) : 
  total = 66 → difference = 8 → corey + (corey + difference) = total → corey = 29 := by
  sorry

end NUMINAMATH_CALUDE_corey_candy_count_l1866_186694


namespace NUMINAMATH_CALUDE_solution_exists_l1866_186685

theorem solution_exists (x : ℝ) : 3 ∈ ({x + 2, x^2 + 2*x} : Set ℝ) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l1866_186685


namespace NUMINAMATH_CALUDE_expression_evaluation_l1866_186682

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := 1
  2 * (x - 2*y)^2 - (2*y + x) * (-2*y + x) = 5 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1866_186682


namespace NUMINAMATH_CALUDE_odd_iff_a_eq_zero_l1866_186635

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  3 * Real.log (x + Real.sqrt (x^2 + 1)) + a * (7^x + 7^(-x))

def isOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_iff_a_eq_zero (a : ℝ) :
  isOdd (f a) ↔ a = 0 :=
sorry

end NUMINAMATH_CALUDE_odd_iff_a_eq_zero_l1866_186635


namespace NUMINAMATH_CALUDE_unique_divisible_digit_l1866_186676

def number (A : Nat) : Nat := 353809 * 10 + A

theorem unique_divisible_digit :
  ∃! (A : Nat), A < 10 ∧ (number A).mod 5 = 0 ∧ (number A).mod 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_digit_l1866_186676


namespace NUMINAMATH_CALUDE_max_value_when_m_1_solution_when_m_neg_2_l1866_186605

-- Define the function f(x, m)
def f (x m : ℝ) : ℝ := |m * x + 1| - |x - 1|

-- Theorem 1: Maximum value of f(x) when m = 1
theorem max_value_when_m_1 :
  ∃ (max : ℝ), max = 2 ∧ ∀ (x : ℝ), f x 1 ≤ max :=
sorry

-- Theorem 2: Solution to f(x) ≥ 1 when m = -2
theorem solution_when_m_neg_2 :
  ∀ (x : ℝ), f x (-2) ≥ 1 ↔ x ≤ -1 ∨ x ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_when_m_1_solution_when_m_neg_2_l1866_186605


namespace NUMINAMATH_CALUDE_integer_triple_solution_l1866_186619

theorem integer_triple_solution (a b c : ℤ) 
  (eq1 : a + b * c = 2017) 
  (eq2 : b + c * a = 8) : 
  c ∈ ({-6, 0, 2, 8} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_triple_solution_l1866_186619


namespace NUMINAMATH_CALUDE_triangle_area_problem_l1866_186669

theorem triangle_area_problem (x : ℝ) (h1 : x > 0) 
  (h2 : (1/2) * x * (3*x) = 96) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l1866_186669


namespace NUMINAMATH_CALUDE_max_non_managers_l1866_186637

/-- The maximum number of non-managers in a department with 9 managers,
    given that the ratio of managers to non-managers must be greater than 7:32 -/
theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 9 →
  (managers : ℚ) / non_managers > 7 / 32 →
  non_managers ≤ 41 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l1866_186637


namespace NUMINAMATH_CALUDE_g_comp_g_three_roots_l1866_186616

/-- The function g defined as g(x) = x^2 + 4x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- The composition of g with itself -/
def g_comp_g (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Theorem stating that g(g(x)) has exactly 3 distinct real roots if and only if d = 0 -/
theorem g_comp_g_three_roots :
  ∀ d : ℝ, (∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧
    g_comp_g d r1 = 0 ∧ g_comp_g d r2 = 0 ∧ g_comp_g d r3 = 0 ∧
    ∀ x : ℝ, g_comp_g d x = 0 → x = r1 ∨ x = r2 ∨ x = r3) ↔ d = 0 :=
sorry

end NUMINAMATH_CALUDE_g_comp_g_three_roots_l1866_186616


namespace NUMINAMATH_CALUDE_vector_dot_product_and_magnitude_l1866_186647

theorem vector_dot_product_and_magnitude :
  ∀ (t : ℝ),
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![2, t]
  (a 0 * b 0 + a 1 * b 1 = 0) →
  Real.sqrt ((b 0) ^ 2 + (b 1) ^ 2) = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_vector_dot_product_and_magnitude_l1866_186647


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l1866_186644

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate_plot2 overall_germination_rate : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate_plot2 = 35 / 100 →
  overall_germination_rate = 26 / 100 →
  ∃ (germination_rate_plot1 : ℚ),
    germination_rate_plot1 = 20 / 100 ∧
    germination_rate_plot1 * seeds_plot1 + germination_rate_plot2 * seeds_plot2 = 
    overall_germination_rate * (seeds_plot1 + seeds_plot2) := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l1866_186644


namespace NUMINAMATH_CALUDE_log_inequalities_l1866_186695

/-- Proves the inequalities for logarithms with different bases -/
theorem log_inequalities :
  (Real.log 4 / Real.log 8 > Real.log 4 / Real.log 9) ∧
  (Real.log 4 / Real.log 9 > Real.log 4 / Real.log 10) ∧
  (Real.log 4 / Real.log 0.3 < 0.3^2) ∧
  (0.3^2 < 2^0.4) := by
  sorry

end NUMINAMATH_CALUDE_log_inequalities_l1866_186695


namespace NUMINAMATH_CALUDE_parabolas_similar_l1866_186614

/-- Two parabolas are similar if there exists a homothety that transforms one into the other -/
theorem parabolas_similar (a : ℝ) : 
  (∃ (x y : ℝ), y = 2 * x^2 → (∃ (x' y' : ℝ), y' = x'^2 ∧ x' = 2*x ∧ y' = 2*y)) := by
  sorry

#check parabolas_similar

end NUMINAMATH_CALUDE_parabolas_similar_l1866_186614


namespace NUMINAMATH_CALUDE_town_street_lights_l1866_186674

/-- Calculates the total number of street lights in a town -/
def total_street_lights (num_neighborhoods : ℕ) (roads_per_neighborhood : ℕ) (lights_per_side : ℕ) : ℕ :=
  num_neighborhoods * roads_per_neighborhood * lights_per_side * 2

theorem town_street_lights :
  total_street_lights 10 4 250 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_town_street_lights_l1866_186674


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l1866_186620

theorem max_sum_given_constraints (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) :
  x + y ≤ 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l1866_186620


namespace NUMINAMATH_CALUDE_divisibility_problem_l1866_186687

theorem divisibility_problem (a b c d : ℤ) 
  (h : (a^4 + b^4 + c^4 + d^4) % 5 = 0) : 
  625 ∣ (a * b * c * d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1866_186687


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_14_mod_21_l1866_186656

theorem largest_four_digit_congruent_to_14_mod_21 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n ≡ 14 [MOD 21] → n ≤ 9979 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_14_mod_21_l1866_186656


namespace NUMINAMATH_CALUDE_power_of_seven_mod_six_l1866_186617

theorem power_of_seven_mod_six : 7^51 % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_six_l1866_186617


namespace NUMINAMATH_CALUDE_min_bushes_for_zucchinis_l1866_186686

/-- Represents the yield of blueberry containers per bush -/
def containers_per_bush : ℕ := 10

/-- Represents the number of containers needed to trade for one zucchini -/
def containers_per_zucchini : ℕ := 3

/-- Represents the target number of zucchinis -/
def target_zucchinis : ℕ := 72

/-- 
Calculates the minimum number of bushes needed to obtain at least the target number of zucchinis.
-/
def min_bushes_needed : ℕ :=
  ((target_zucchinis * containers_per_zucchini + containers_per_bush - 1) / containers_per_bush : ℕ)

theorem min_bushes_for_zucchinis :
  min_bushes_needed = 22 ∧
  min_bushes_needed * containers_per_bush ≥ target_zucchinis * containers_per_zucchini ∧
  (min_bushes_needed - 1) * containers_per_bush < target_zucchinis * containers_per_zucchini :=
by sorry

end NUMINAMATH_CALUDE_min_bushes_for_zucchinis_l1866_186686


namespace NUMINAMATH_CALUDE_intersection_M_N_l1866_186688

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1866_186688


namespace NUMINAMATH_CALUDE_collinear_points_theorem_l1866_186650

/-- Given three points A, B, and C in a 2D plane, this function checks if they are collinear -/
def are_collinear (A B C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem stating that if A(m, 0), B(0, 1), and C(3, -1) are collinear, then m = 3/2 -/
theorem collinear_points_theorem (m : ℝ) :
  are_collinear (m, 0) (0, 1) (3, -1) → m = 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_points_theorem_l1866_186650


namespace NUMINAMATH_CALUDE_chess_tournament_draw_fraction_l1866_186624

theorem chess_tournament_draw_fraction 
  (peter_wins : Rat) 
  (marc_wins : Rat) 
  (h1 : peter_wins = 2 / 5)
  (h2 : marc_wins = 1 / 4)
  : 1 - (peter_wins + marc_wins) = 7 / 20 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_draw_fraction_l1866_186624


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1866_186607

/-- Given a hyperbola with the equation y²/a² - x²/b² = l, where a > 0 and b > 0,
    if the point (1, 2) lies on the hyperbola, then its eccentricity e is greater than √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  4/a^2 - 1/b^2 = 1 → ∃ e : ℝ, e > Real.sqrt 5 / 2 ∧ e^2 = (a^2 + b^2)/a^2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1866_186607


namespace NUMINAMATH_CALUDE_condition_relationship_l1866_186696

theorem condition_relationship (x : ℝ) :
  (∀ x, x > 1 → 1/x < 1) ∧
  (∃ x, 1/x < 1 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l1866_186696


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l1866_186663

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) → m₁ = m₂

/-- The value of a for which the lines x + ay - 1 = 0 and (a-1)x + ay + 1 = 0 are parallel -/
theorem parallel_lines_a_value : 
  (∀ x y, x + a * y - 1 = 0 ↔ (a - 1) * x + a * y + 1 = 0) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l1866_186663


namespace NUMINAMATH_CALUDE_calculate_expression_l1866_186673

theorem calculate_expression : 14 - (-12) + (-25) - 17 = -16 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1866_186673


namespace NUMINAMATH_CALUDE_marias_blueberries_l1866_186679

/-- Proves that Maria has 8 cartons of blueberries given the problem conditions -/
theorem marias_blueberries 
  (total_needed : ℕ) 
  (strawberries : ℕ) 
  (to_buy : ℕ) 
  (h1 : total_needed = 21)
  (h2 : strawberries = 4)
  (h3 : to_buy = 9) :
  total_needed - (strawberries + to_buy) = 8 := by
  sorry

#eval 21 - (4 + 9)  -- Should output 8

end NUMINAMATH_CALUDE_marias_blueberries_l1866_186679


namespace NUMINAMATH_CALUDE_no_double_application_plus_one_l1866_186640

theorem no_double_application_plus_one : 
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1 := by
sorry

end NUMINAMATH_CALUDE_no_double_application_plus_one_l1866_186640


namespace NUMINAMATH_CALUDE_white_dogs_count_l1866_186608

theorem white_dogs_count (total brown black : ℕ) 
  (h_total : total = 45)
  (h_brown : brown = 20)
  (h_black : black = 15) :
  total - (brown + black) = 10 := by
  sorry

end NUMINAMATH_CALUDE_white_dogs_count_l1866_186608


namespace NUMINAMATH_CALUDE_no_real_solutions_l1866_186629

theorem no_real_solutions : ∀ x : ℝ, x^2 ≠ 4 → x ≠ 2 → x ≠ -2 → 
  (8*x)/(x^2 - 4) ≠ (3*x)/(x - 2) - 4/(x + 2) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1866_186629


namespace NUMINAMATH_CALUDE_coordinates_of_q_l1866_186626

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A triangle in 2D space -/
structure Triangle where
  p : Point2D
  q : Point2D
  r : Point2D

/-- Predicate for a right-angled triangle at Q -/
def isRightAngledAtQ (t : Triangle) : Prop :=
  -- Definition of right angle at Q (placeholder)
  True

/-- Predicate for a horizontal line segment -/
def isHorizontal (p1 p2 : Point2D) : Prop :=
  p1.y = p2.y

/-- Predicate for a vertical line segment -/
def isVertical (p1 p2 : Point2D) : Prop :=
  p1.x = p2.x

theorem coordinates_of_q (t : Triangle) :
  isRightAngledAtQ t →
  isHorizontal t.p t.q →
  isVertical t.q t.r →
  t.p = Point2D.mk 1 1 →
  t.r = Point2D.mk 5 3 →
  t.q = Point2D.mk 5 1 := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_q_l1866_186626


namespace NUMINAMATH_CALUDE_midpoint_complex_l1866_186638

theorem midpoint_complex (z₁ z₂ : ℂ) (h₁ : z₁ = 2 + I) (h₂ : z₂ = 4 - 3*I) :
  (z₁ + z₂) / 2 = 3 - I := by
  sorry

end NUMINAMATH_CALUDE_midpoint_complex_l1866_186638


namespace NUMINAMATH_CALUDE_f_property_f_at_two_l1866_186609

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_property (x : ℝ) : f x = (deriv f 1) * Real.exp (x - 1) - f 0 * x + (1/2) * x^2 := sorry

theorem f_at_two : f 2 = Real.exp 2 := by sorry

end NUMINAMATH_CALUDE_f_property_f_at_two_l1866_186609


namespace NUMINAMATH_CALUDE_remainder_difference_theorem_l1866_186645

theorem remainder_difference_theorem (k : ℕ) (r : ℕ) : 
  k > 1 ∧ 
  (∃ q₁ q₂ q₃ : ℕ, 1177 = k * q₁ + r ∧ 1573 = k * q₂ + r ∧ 2552 = k * q₃ + r) ∧ 
  (∀ m : ℕ, m > 1 → m ∣ (1573 - 1177) → m ∣ (2552 - 1573) → m ∣ (2552 - 1177) → m ≤ k) →
  k - r = 11 := by
sorry

end NUMINAMATH_CALUDE_remainder_difference_theorem_l1866_186645


namespace NUMINAMATH_CALUDE_regular_hexagon_perimeter_l1866_186692

/-- The perimeter of a regular hexagon with side length 5 cm is 30 cm. -/
theorem regular_hexagon_perimeter :
  ∀ (side_length : ℝ), side_length = 5 →
  (6 : ℝ) * side_length = 30 := by sorry

end NUMINAMATH_CALUDE_regular_hexagon_perimeter_l1866_186692


namespace NUMINAMATH_CALUDE_lexie_paintings_l1866_186641

theorem lexie_paintings (num_rooms : ℕ) (paintings_per_room : ℕ) 
  (h1 : num_rooms = 4) 
  (h2 : paintings_per_room = 8) : 
  num_rooms * paintings_per_room = 32 := by
sorry

end NUMINAMATH_CALUDE_lexie_paintings_l1866_186641


namespace NUMINAMATH_CALUDE_polynomial_existence_l1866_186666

theorem polynomial_existence : ∃ (P : ℤ → ℤ), 
  (∃ (a b c d e f g h i : ℤ), ∀ x, P x = a*x^8 + b*x^7 + c*x^6 + d*x^5 + e*x^4 + f*x^3 + g*x^2 + h*x + i) ∧ 
  (∀ x : ℤ, P x ≠ 0) ∧
  (∀ n : ℕ, n > 0 → ∃ x : ℤ, (n : ℤ) ∣ P x) := by
sorry

end NUMINAMATH_CALUDE_polynomial_existence_l1866_186666


namespace NUMINAMATH_CALUDE_equivalent_form_l1866_186625

theorem equivalent_form :
  (2 + 5) * (2^2 + 5^2) * (2^4 + 5^4) * (2^8 + 5^8) * 
  (2^16 + 5^16) * (2^32 + 5^32) * (2^64 + 5^64) = 5^128 - 2^128 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_form_l1866_186625


namespace NUMINAMATH_CALUDE_antonov_remaining_packs_l1866_186603

/-- Calculates the number of remaining candy packs given the initial number of candies,
    the number of candies per pack, and the number of packs given away. -/
def remaining_packs (initial_candies : ℕ) (candies_per_pack : ℕ) (packs_given : ℕ) : ℕ :=
  (initial_candies - packs_given * candies_per_pack) / candies_per_pack

/-- Proves that Antonov has 2 packs of candy remaining. -/
theorem antonov_remaining_packs :
  remaining_packs 60 20 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_antonov_remaining_packs_l1866_186603


namespace NUMINAMATH_CALUDE_vacation_days_l1866_186627

theorem vacation_days (rainy_days clear_mornings clear_afternoons : ℕ) 
  (h1 : rainy_days = 13)
  (h2 : clear_mornings = 11)
  (h3 : clear_afternoons = 12)
  (h4 : rainy_days = clear_mornings + clear_afternoons) :
  clear_mornings + clear_afternoons = 23 := by
  sorry

end NUMINAMATH_CALUDE_vacation_days_l1866_186627


namespace NUMINAMATH_CALUDE_product_19_reciprocal_squares_sum_l1866_186618

theorem product_19_reciprocal_squares_sum :
  ∀ a b : ℕ+, 
  (a * b : ℕ+) = 19 → 
  (1 : ℚ) / (a * a : ℚ) + (1 : ℚ) / (b * b : ℚ) = 362 / 361 := by
  sorry

end NUMINAMATH_CALUDE_product_19_reciprocal_squares_sum_l1866_186618
