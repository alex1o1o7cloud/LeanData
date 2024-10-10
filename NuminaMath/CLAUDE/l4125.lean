import Mathlib

namespace circle_plus_self_twice_l4125_412517

/-- Definition of the ⊕ operation -/
def circle_plus (x y : ℝ) : ℝ := x^3 + 2*x - y

/-- Theorem stating that k ⊕ (k ⊕ k) = k -/
theorem circle_plus_self_twice (k : ℝ) : circle_plus k (circle_plus k k) = k := by
  sorry

end circle_plus_self_twice_l4125_412517


namespace length_OP_greater_than_radius_l4125_412573

-- Define a circle with radius 5
def circle_radius : ℝ := 5

-- Define a point P outside the circle
def point_outside_circle (P : ℝ × ℝ) : Prop :=
  let O := (0, 0)  -- Assume the circle center is at the origin
  Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) > circle_radius

-- Theorem statement
theorem length_OP_greater_than_radius (P : ℝ × ℝ) 
  (h : point_outside_circle P) : 
  Real.sqrt ((P.1)^2 + (P.2)^2) > circle_radius :=
sorry

end length_OP_greater_than_radius_l4125_412573


namespace convenient_denominator_sum_or_diff_integer_l4125_412524

/-- A positive integer q is a convenient denominator for a real number α if 
    |α - p/q| < 1/(10q) for some integer p -/
def ConvenientDenominator (α : ℝ) (q : ℕ+) : Prop :=
  ∃ p : ℤ, |α - (p : ℝ) / q| < 1 / (10 * q)

theorem convenient_denominator_sum_or_diff_integer 
  (α β : ℝ) (hα : Irrational α) (hβ : Irrational β) :
  (∀ q : ℕ+, ConvenientDenominator α q ↔ ConvenientDenominator β q) →
  (∃ n : ℤ, α + β = n) ∨ (∃ n : ℤ, α - β = n) := by
  sorry

end convenient_denominator_sum_or_diff_integer_l4125_412524


namespace semicircle_chord_projection_l4125_412548

/-- Given a semicircle with diameter 2R and a chord intersecting the semicircle and its tangent,
    prove that the condition AC^2 + CD^2 + BD^2 = 4a^2 has a solution for the projection of C on AB
    if and only if a^2 ≥ R^2, and that this solution is unique. -/
theorem semicircle_chord_projection (R a : ℝ) (h : R > 0) :
  ∃! x, x > 0 ∧ x < 2*R ∧ 
    2*R*x + (4*R^2*(2*R - x)^2)/x^2 + (4*R^2*(2*R - x))/x = 4*a^2 ↔ 
  a^2 ≥ R^2 :=
by sorry

end semicircle_chord_projection_l4125_412548


namespace prob_purple_second_l4125_412585

-- Define the bags
def bag_A : Nat × Nat := (5, 5)  -- (red, green)
def bag_B : Nat × Nat := (8, 2)  -- (purple, orange)
def bag_C : Nat × Nat := (3, 7)  -- (purple, orange)

-- Define the probability of drawing a red marble from Bag A
def prob_red_A : Rat := bag_A.1 / (bag_A.1 + bag_A.2)

-- Define the probability of drawing a green marble from Bag A
def prob_green_A : Rat := bag_A.2 / (bag_A.1 + bag_A.2)

-- Define the probability of drawing a purple marble from Bag B
def prob_purple_B : Rat := bag_B.1 / (bag_B.1 + bag_B.2)

-- Define the probability of drawing a purple marble from Bag C
def prob_purple_C : Rat := bag_C.1 / (bag_C.1 + bag_C.2)

-- Theorem: The probability of drawing a purple marble as the second marble is 11/20
theorem prob_purple_second : 
  prob_red_A * prob_purple_B + prob_green_A * prob_purple_C = 11/20 := by
  sorry

end prob_purple_second_l4125_412585


namespace equation_solution_l4125_412569

theorem equation_solution (x : ℚ) : (40 / 60 : ℚ) = Real.sqrt (x / 60) → x = 80 / 3 := by
  sorry

end equation_solution_l4125_412569


namespace max_gold_coins_theorem_l4125_412591

/-- Represents a toy that can be created -/
structure Toy where
  planks : ℕ
  value : ℕ

/-- Calculates the maximum gold coins that can be earned given a number of planks and a list of toys -/
def maxGoldCoins (totalPlanks : ℕ) (toys : List Toy) : ℕ :=
  sorry

/-- The theorem stating the maximum gold coins that can be earned -/
theorem max_gold_coins_theorem :
  let windmill : Toy := ⟨5, 6⟩
  let steamboat : Toy := ⟨7, 8⟩
  let airplane : Toy := ⟨14, 19⟩
  let toys : List Toy := [windmill, steamboat, airplane]
  maxGoldCoins 130 toys = 172 := by
  sorry

end max_gold_coins_theorem_l4125_412591


namespace g_equality_l4125_412530

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := -2*x^5 + 3*x^4 - 11*x^3 + x^2 + 5*x - 5

-- State the theorem
theorem g_equality (x : ℝ) : 2*x^5 + 4*x^3 - 5*x + 3 + g x = 3*x^4 - 7*x^3 + x^2 - 2 := by
  sorry

end g_equality_l4125_412530


namespace total_pencils_l4125_412505

/-- Calculate the total number of pencils Asaf and Alexander have together -/
theorem total_pencils (asaf_age alexander_age asaf_pencils alexander_pencils : ℕ) :
  asaf_age + alexander_age = 140 →
  asaf_age = 50 →
  alexander_age - asaf_age = asaf_pencils / 2 →
  alexander_pencils = asaf_pencils + 60 →
  asaf_pencils + alexander_pencils = 220 := by
  sorry

end total_pencils_l4125_412505


namespace absolute_value_inequality_solution_set_l4125_412534

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 1| ≥ 3} = {x : ℝ | x ≤ -1 ∨ x ≥ 2} := by sorry

end absolute_value_inequality_solution_set_l4125_412534


namespace smaller_solution_form_l4125_412509

theorem smaller_solution_form : ∃ (p q : ℤ),
  ∃ (x : ℝ),
    x^(1/4) + (40 - x)^(1/4) = 2 ∧
    x = p - Real.sqrt q ∧
    ∀ (y : ℝ), y^(1/4) + (40 - y)^(1/4) = 2 → y ≥ x :=
by sorry

end smaller_solution_form_l4125_412509


namespace specific_cylinder_properties_l4125_412571

/-- Represents a cylinder with height and surface area as parameters. -/
structure Cylinder where
  height : ℝ
  surfaceArea : ℝ

/-- Calculates the radius of the base circle of a cylinder. -/
def baseRadius (c : Cylinder) : ℝ :=
  sorry

/-- Calculates the volume of a cylinder. -/
def volume (c : Cylinder) : ℝ :=
  sorry

/-- Theorem stating the properties of a specific cylinder. -/
theorem specific_cylinder_properties :
  let c := Cylinder.mk 8 (130 * Real.pi)
  baseRadius c = 5 ∧ volume c = 200 * Real.pi :=
sorry

end specific_cylinder_properties_l4125_412571


namespace no_solutions_exist_l4125_412560

theorem no_solutions_exist : ¬∃ (x y : ℕ+) (m : ℕ), 
  (x : ℝ)^2 + (y : ℝ)^2 = (x : ℝ)^5 ∧ x = m^6 + 1 := by
  sorry

end no_solutions_exist_l4125_412560


namespace vector_subtraction_scalar_multiplication_l4125_412538

theorem vector_subtraction_scalar_multiplication (a b : ℝ × ℝ) :
  a = (3, -8) → b = (2, -6) → a - 5 • b = (-7, 22) := by sorry

end vector_subtraction_scalar_multiplication_l4125_412538


namespace carrot_problem_l4125_412546

theorem carrot_problem (carol_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) :
  carol_carrots = 29 →
  mom_carrots = 16 →
  good_carrots = 38 →
  carol_carrots + mom_carrots - good_carrots = 7 :=
by sorry

end carrot_problem_l4125_412546


namespace car_replacement_problem_l4125_412545

theorem car_replacement_problem :
  let initial_fleet : ℕ := 20
  let new_cars_per_year : ℕ := 6
  let years : ℕ := 2
  ∃ (x : ℕ),
    x > 0 ∧
    initial_fleet - years * x < initial_fleet / 2 ∧
    ∀ (y : ℕ), y > 0 ∧ initial_fleet - years * y < initial_fleet / 2 → x ≤ y :=
by sorry

end car_replacement_problem_l4125_412545


namespace unique_valid_triple_l4125_412547

/-- Represents an ordered triple of integers (a, b, c) satisfying the given conditions -/
structure ValidTriple where
  a : ℕ
  b : ℕ
  c : ℕ
  a_ge_2 : a ≥ 2
  b_ge_1 : b ≥ 1
  log_cond : (Real.log b) / (Real.log a) = c^2
  sum_cond : a + b + c = 100

/-- There exists exactly one ordered triple of integers satisfying the given conditions -/
theorem unique_valid_triple : ∃! t : ValidTriple, True := by sorry

end unique_valid_triple_l4125_412547


namespace divisibility_floor_factorial_l4125_412518

theorem divisibility_floor_factorial (m n : ℤ) 
  (h1 : 1 < m) (h2 : m < n + 2) (h3 : n > 3) : 
  (m - 1) ∣ ⌊n! / m⌋ := by
  sorry

end divisibility_floor_factorial_l4125_412518


namespace matrix_not_invertible_l4125_412590

/-- A 2x2 matrix is not invertible if its determinant is zero. -/
def is_not_invertible (a b c d : ℚ) : Prop :=
  a * d - b * c = 0

/-- The matrix in question with x as a parameter. -/
def matrix (x : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![2 * x + 1, 9],
    ![4 - x, 10]]

/-- Theorem stating that the matrix is not invertible when x = 26/29. -/
theorem matrix_not_invertible :
  is_not_invertible (2 * (26/29) + 1) 9 (4 - (26/29)) 10 := by
  sorry

end matrix_not_invertible_l4125_412590


namespace isabel_song_count_l4125_412502

/-- The number of songs Isabel bought -/
def total_songs (country_albums pop_albums songs_per_album : ℕ) : ℕ :=
  (country_albums + pop_albums) * songs_per_album

/-- Theorem stating that Isabel bought 72 songs -/
theorem isabel_song_count :
  total_songs 4 5 8 = 72 := by
  sorry

end isabel_song_count_l4125_412502


namespace disk_at_nine_oclock_l4125_412580

/-- Represents a circular clock face with a smaller disk rolling on it. -/
structure ClockWithDisk where
  clock_radius : ℝ
  disk_radius : ℝ
  start_position : ℝ -- in radians, 0 represents 3 o'clock
  rotation_direction : Bool -- true for clockwise

/-- Calculates the position of the disk after one full rotation -/
def position_after_rotation (c : ClockWithDisk) : ℝ :=
  sorry

/-- Theorem stating that the disk will be at 9 o'clock after one full rotation -/
theorem disk_at_nine_oclock (c : ClockWithDisk) 
  (h1 : c.clock_radius = 30)
  (h2 : c.disk_radius = 15)
  (h3 : c.start_position = 0)
  (h4 : c.rotation_direction = true) :
  position_after_rotation c = π := by
  sorry

end disk_at_nine_oclock_l4125_412580


namespace second_sunday_on_13th_l4125_412549

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with specific properties -/
structure Month where
  /-- The day of the week on which the month starts -/
  startDay : DayOfWeek
  /-- The number of days in the month -/
  numDays : Nat
  /-- Predicate that is true if three Wednesdays fall on even dates -/
  threeWednesdaysOnEvenDates : Prop

/-- Given a month and a day number, returns the day of the week -/
def dayOfWeek (m : Month) (day : Nat) : DayOfWeek :=
  sorry

/-- Predicate that is true if the given day is a Sunday -/
def isSunday (dow : DayOfWeek) : Prop :=
  sorry

/-- Returns the date of the nth occurrence of a specific day in the month -/
def nthOccurrence (m : Month) (dow : DayOfWeek) (n : Nat) : Nat :=
  sorry

/-- Theorem stating that in a month where three Wednesdays fall on even dates, 
    the second Sunday of that month falls on the 13th -/
theorem second_sunday_on_13th (m : Month) :
  m.threeWednesdaysOnEvenDates → nthOccurrence m DayOfWeek.Sunday 2 = 13 :=
sorry

end second_sunday_on_13th_l4125_412549


namespace fraction_subtraction_l4125_412544

theorem fraction_subtraction : (5 : ℚ) / 12 - (3 : ℚ) / 18 = (1 : ℚ) / 4 := by sorry

end fraction_subtraction_l4125_412544


namespace tangent_slope_at_one_l4125_412507

noncomputable def f (x : ℝ) := x * Real.exp x

theorem tangent_slope_at_one :
  (deriv f) 1 = 2 * Real.exp 1 := by
sorry

end tangent_slope_at_one_l4125_412507


namespace smallest_k_for_binomial_divisibility_l4125_412594

theorem smallest_k_for_binomial_divisibility (k : ℕ) : 
  (k ≥ 25 ∧ 49 ∣ Nat.choose (2 * k) k) ∧ 
  (∀ m : ℕ, m < 25 → ¬(49 ∣ Nat.choose (2 * m) m)) :=
by sorry

end smallest_k_for_binomial_divisibility_l4125_412594


namespace intersection_x_coordinate_l4125_412578

/-- The x-coordinate of the intersection point of two lines -/
theorem intersection_x_coordinate (k b : ℝ) (h : k ≠ b) :
  ∃ x : ℝ, k * x + b = b * x + k ∧ x = 1 :=
sorry

end intersection_x_coordinate_l4125_412578


namespace vehicle_purchase_problem_l4125_412561

/-- Represents the purchase price and profit information for new energy vehicles -/
structure VehicleInfo where
  priceA : ℝ  -- Purchase price of type A vehicle in million yuan
  priceB : ℝ  -- Purchase price of type B vehicle in million yuan
  profitA : ℝ  -- Profit from selling one type A vehicle in million yuan
  profitB : ℝ  -- Profit from selling one type B vehicle in million yuan

/-- Represents a purchasing plan -/
structure PurchasePlan where
  countA : ℕ  -- Number of type A vehicles
  countB : ℕ  -- Number of type B vehicles

/-- Calculates the total cost of a purchase plan given vehicle info -/
def totalCost (plan : PurchasePlan) (info : VehicleInfo) : ℝ :=
  info.priceA * plan.countA + info.priceB * plan.countB

/-- Calculates the total profit of a purchase plan given vehicle info -/
def totalProfit (plan : PurchasePlan) (info : VehicleInfo) : ℝ :=
  info.profitA * plan.countA + info.profitB * plan.countB

/-- Theorem stating the properties of the vehicle purchase problem -/
theorem vehicle_purchase_problem (info : VehicleInfo) :
  (totalCost ⟨3, 2⟩ info = 95) →
  (totalCost ⟨4, 1⟩ info = 110) →
  (info.profitA = 0.012) →
  (info.profitB = 0.008) →
  (∃ (plans : List PurchasePlan),
    (∀ plan ∈ plans, totalCost plan info = 250) ∧
    (plans.length = 4) ∧
    (∃ maxProfit : ℝ, maxProfit = 18.4 ∧
      ∀ plan ∈ plans, totalProfit plan info ≤ maxProfit)) :=
sorry


end vehicle_purchase_problem_l4125_412561


namespace cross_product_scalar_m_l4125_412537

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the cross product operation
variable (cross : V → V → V)

-- Axioms for cross product
variable (cross_anticomm : ∀ a b : V, cross a b = - cross b a)
variable (cross_distributive : ∀ a b c : V, cross a (b + c) = cross a b + cross a c)
variable (cross_zero : ∀ a : V, cross a a = 0)

-- The main theorem
theorem cross_product_scalar_m (m : ℝ) : 
  (∀ u v w : V, u + v + w = 0 → 
    m • (cross v u) + cross v w + cross w u = cross v u) → 
  m = 3 := by
sorry

end cross_product_scalar_m_l4125_412537


namespace certain_number_problem_l4125_412558

theorem certain_number_problem (x : ℝ) (y : ℝ) : 
  x = y + 0.5 * y → x = 132 → y = 88 := by sorry

end certain_number_problem_l4125_412558


namespace equation_solution_l4125_412540

theorem equation_solution : 
  ∀ x : ℝ, (1 / (x + 1) + 1 / (x + 2) = 1 / x) ↔ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) := by
  sorry

end equation_solution_l4125_412540


namespace set_equality_l4125_412555

theorem set_equality : 
  {x : ℕ | x - 1 ≤ 2} = {0, 1, 2, 3} := by sorry

end set_equality_l4125_412555


namespace fixed_point_on_line_l4125_412516

/-- The line mx - y + 2m + 1 = 0 passes through the point (-2, 1) for any real m -/
theorem fixed_point_on_line (m : ℝ) : m * (-2) - 1 + 2 * m + 1 = 0 := by
  sorry

end fixed_point_on_line_l4125_412516


namespace smallest_b_value_l4125_412579

/-- Given real numbers a and b where 2 < a < b, and no triangle with positive area
    has side lengths 2, a, and b or 1/b, 1/a, and 1/2, the smallest possible value of b is 6. -/
theorem smallest_b_value (a b : ℝ) (h1 : 2 < a) (h2 : a < b)
  (h3 : ¬ (2 + a > b ∧ 2 + b > a ∧ a + b > 2))
  (h4 : ¬ (1/b + 1/a > 1/2 ∧ 1/b + 1/2 > 1/a ∧ 1/a + 1/2 > 1/b)) :
  6 ≤ b ∧ ∀ c, (2 < c → c < b → 
    ¬(2 + c > b ∧ 2 + b > c ∧ c + b > 2) → 
    ¬(1/b + 1/c > 1/2 ∧ 1/b + 1/2 > 1/c ∧ 1/c + 1/2 > 1/b) → 
    6 ≤ c) :=
sorry

end smallest_b_value_l4125_412579


namespace not_all_lines_perp_when_planes_perp_l4125_412563

-- Define the basic geometric objects
variable (α β : Plane) (l : Line)

-- Define perpendicularity between planes
def perp_planes (p q : Plane) : Prop := sorry

-- Define a line being within a plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

-- Define perpendicularity between a line and a plane
def perp_line_plane (l : Line) (p : Plane) : Prop := sorry

-- The statement to be proved
theorem not_all_lines_perp_when_planes_perp (α β : Plane) :
  perp_planes α β → ¬ (∀ l : Line, line_in_plane l α → perp_line_plane l β) := by
  sorry

end not_all_lines_perp_when_planes_perp_l4125_412563


namespace cricket_team_average_age_l4125_412593

/-- Represents a cricket team with its age-related properties -/
structure CricketTeam where
  totalMembers : ℕ
  averageAge : ℝ
  captainAgeDiff : ℝ
  remainingAverageAgeDiff : ℝ

/-- Theorem stating that the average age of the cricket team is 30 years -/
theorem cricket_team_average_age
  (team : CricketTeam)
  (h1 : team.totalMembers = 20)
  (h2 : team.averageAge = 30)
  (h3 : team.captainAgeDiff = 5)
  (h4 : team.remainingAverageAgeDiff = 3)
  : team.averageAge = 30 := by
  sorry

#check cricket_team_average_age

end cricket_team_average_age_l4125_412593


namespace sufficient_not_necessary_condition_existence_of_m_outside_interval_l4125_412584

theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, x > 1 → x^2 - m*x + 1 > 0) ↔ m < 2 :=
by sorry

theorem existence_of_m_outside_interval :
  ∃ m : ℝ, (m ≤ -2 ∨ m ≥ 2) ∧ (∀ x : ℝ, x > 1 → x^2 - m*x + 1 > 0) :=
by sorry

end sufficient_not_necessary_condition_existence_of_m_outside_interval_l4125_412584


namespace payment_plan_difference_l4125_412523

def purchase_price : ℕ := 1500
def down_payment : ℕ := 200
def num_monthly_payments : ℕ := 24
def monthly_payment : ℕ := 65

theorem payment_plan_difference :
  (down_payment + num_monthly_payments * monthly_payment) - purchase_price = 260 := by
  sorry

end payment_plan_difference_l4125_412523


namespace find_xy_l4125_412542

/-- Define the ⊕ operation for pairs of real numbers -/
def oplus (a b c d : ℝ) : ℝ × ℝ := (a + c, b * d)

/-- Theorem statement -/
theorem find_xy : ∃ (x y : ℝ), oplus x 1 2 y = (4, 2) ∧ (x, y) = (2, 2) := by
  sorry

end find_xy_l4125_412542


namespace rectangle_dimensions_l4125_412577

theorem rectangle_dimensions : ∃ (x y : ℝ), 
  x > 0 ∧ y > 0 ∧
  x = 2 * y ∧
  2 * (x + y) = 2 * (x * y) ∧
  x = 3 ∧ y = 1.5 :=
by sorry

end rectangle_dimensions_l4125_412577


namespace next_number_with_property_l4125_412543

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_property (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  is_perfect_square ((n / 100) * (n % 100))

theorem next_number_with_property :
  ∀ n : ℕ, n > 1818 →
  (∀ m : ℕ, 1818 < m ∧ m < n → ¬has_property m) →
  has_property n →
  n = 1832 :=
sorry

end next_number_with_property_l4125_412543


namespace second_stock_percentage_l4125_412504

/-- Prove that the percentage of the second stock is 15% given the investment conditions --/
theorem second_stock_percentage
  (total_investment : ℚ)
  (first_stock_percentage : ℚ)
  (first_stock_face_value : ℚ)
  (second_stock_face_value : ℚ)
  (total_dividend : ℚ)
  (first_stock_investment : ℚ)
  (h1 : total_investment = 12000)
  (h2 : first_stock_percentage = 12 / 100)
  (h3 : first_stock_face_value = 120)
  (h4 : second_stock_face_value = 125)
  (h5 : total_dividend = 1360)
  (h6 : first_stock_investment = 4000.000000000002)
  : (total_dividend - (first_stock_investment / first_stock_face_value * first_stock_percentage)) /
    ((total_investment - first_stock_investment) / second_stock_face_value) = 15 / 100 := by
  sorry

end second_stock_percentage_l4125_412504


namespace flag_designs_count_l4125_412521

/-- The number of colors available for the flag design -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The number of different flag designs possible -/
def num_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating that the number of different flag designs is 27 -/
theorem flag_designs_count : num_designs = 27 := by
  sorry

end flag_designs_count_l4125_412521


namespace train_platform_crossing_time_l4125_412529

/-- Given a train of length 1200 m that crosses a tree in 120 sec,
    prove that it takes 190 sec to pass a platform of length 700 m. -/
theorem train_platform_crossing_time :
  ∀ (train_length platform_length tree_crossing_time : ℝ),
    train_length = 1200 →
    platform_length = 700 →
    tree_crossing_time = 120 →
    (train_length + platform_length) / (train_length / tree_crossing_time) = 190 :=
by sorry

end train_platform_crossing_time_l4125_412529


namespace stream_current_rate_l4125_412520

/-- The rate of the stream's current in miles per hour -/
def w : ℝ := 3

/-- The man's rowing speed in still water in miles per hour -/
def r : ℝ := 6

/-- The distance traveled downstream and upstream in miles -/
def d : ℝ := 18

/-- Theorem stating that given the conditions, the stream's current is 3 mph -/
theorem stream_current_rate : 
  (d / (r + w) + 4 = d / (r - w)) ∧ 
  (d / (3 * r + w) + 2 = d / (3 * r - w)) → 
  w = 3 := by
  sorry

end stream_current_rate_l4125_412520


namespace range_of_m_l4125_412508

theorem range_of_m (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 9*a + b = a*b)
  (h : ∀ x : ℝ, a + b ≥ -x^2 + 2*x + 18 - m) :
  ∃ m₀ : ℝ, m₀ = 3 ∧ ∀ m : ℝ, (∀ x : ℝ, a + b ≥ -x^2 + 2*x + 18 - m) → m ≥ m₀ :=
sorry

end range_of_m_l4125_412508


namespace impossible_arrangement_l4125_412564

/-- Represents a cell in the table -/
structure Cell where
  row : Fin 2002
  col : Fin 2002

/-- Represents the table arrangement -/
def TableArrangement := Cell → Fin (2002^2)

/-- Checks if a triplet satisfies the product condition -/
def satisfiesProductCondition (a b c : Fin (2002^2)) : Prop :=
  (a.val + 1) * (b.val + 1) = c.val + 1 ∨
  (a.val + 1) * (c.val + 1) = b.val + 1 ∨
  (b.val + 1) * (c.val + 1) = a.val + 1

/-- Checks if a cell satisfies the condition in its row or column -/
def cellSatisfiesCondition (t : TableArrangement) (cell : Cell) : Prop :=
  ∃ (a b c : Cell),
    ((a.row = cell.row ∧ b.row = cell.row ∧ c.row = cell.row) ∨
     (a.col = cell.col ∧ b.col = cell.col ∧ c.col = cell.col)) ∧
    satisfiesProductCondition (t a) (t b) (t c)

/-- The main theorem stating the impossibility of the arrangement -/
theorem impossible_arrangement :
  ¬∃ (t : TableArrangement),
    (∀ (c₁ c₂ : Cell), c₁ ≠ c₂ → t c₁ ≠ t c₂) ∧
    (∀ (cell : Cell), cellSatisfiesCondition t cell) :=
  sorry

end impossible_arrangement_l4125_412564


namespace unique_value_of_a_l4125_412552

theorem unique_value_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27*x^3) 
  (h3 : a - b = 3*x) : 
  a = 3*x :=
sorry

end unique_value_of_a_l4125_412552


namespace sum_of_largest_and_smallest_l4125_412557

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 10000000 ∧ n ≤ 99999999) ∧
  (∃ (a b c d : ℕ), 
    a + b + c + d = 12 ∧
    List.count 4 (Nat.digits 10 n) = 2 ∧
    List.count 0 (Nat.digits 10 n) = 2 ∧
    List.count 2 (Nat.digits 10 n) = 2 ∧
    List.count 6 (Nat.digits 10 n) = 2)

def largest_valid_number : ℕ := 66442200
def smallest_valid_number : ℕ := 20024466

theorem sum_of_largest_and_smallest :
  is_valid_number largest_valid_number ∧
  is_valid_number smallest_valid_number ∧
  (∀ n : ℕ, is_valid_number n → n ≤ largest_valid_number) ∧
  (∀ n : ℕ, is_valid_number n → n ≥ smallest_valid_number) ∧
  largest_valid_number + smallest_valid_number = 86466666 := by
  sorry

end sum_of_largest_and_smallest_l4125_412557


namespace angle_problem_l4125_412511

theorem angle_problem (x y : ℝ) : 
  x + y + 120 = 360 →
  x = 2 * y →
  x = 160 ∧ y = 80 :=
by sorry

end angle_problem_l4125_412511


namespace farmland_width_l4125_412587

/-- Represents a rectangular plot of farmland -/
structure FarmPlot where
  length : ℝ
  width : ℝ
  area : ℝ

/-- Conversion factor from acres to square feet -/
def acreToSqFt : ℝ := 43560

/-- Theorem stating the width of the farmland plot -/
theorem farmland_width (plot : FarmPlot) 
  (h1 : plot.length = 360)
  (h2 : plot.area = 10 * acreToSqFt)
  (h3 : plot.area = plot.length * plot.width) :
  plot.width = 1210 := by
  sorry

end farmland_width_l4125_412587


namespace quadratic_point_relationship_l4125_412510

/-- A quadratic function f(x) = x^2 - 2x + m passing through three specific points -/
def QuadraticThroughPoints (m : ℝ) (y₁ y₂ y₃ : ℝ) : Prop :=
  let f := fun x => x^2 - 2*x + m
  f (-1) = y₁ ∧ f 2 = y₂ ∧ f 3 = y₃

/-- Theorem stating the relationship between y₁, y₂, and y₃ for the given quadratic function -/
theorem quadratic_point_relationship (m : ℝ) (y₁ y₂ y₃ : ℝ) 
    (h : QuadraticThroughPoints m y₁ y₂ y₃) : 
    y₂ < y₁ ∧ y₁ = y₃ := by
  sorry

end quadratic_point_relationship_l4125_412510


namespace quadratic_inequality_problem_l4125_412513

theorem quadratic_inequality_problem (m n : ℝ) (h1 : ∀ x : ℝ, x^2 - 3*x + m < 0 ↔ 1 < x ∧ x < n) :
  m = 2 ∧ n = 2 ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → m*a + 2*n*b = 3 → a*b ≤ 9/32) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ m*a + 2*n*b = 3 ∧ a*b = 9/32) :=
by sorry

end quadratic_inequality_problem_l4125_412513


namespace remainder_zero_prime_l4125_412531

theorem remainder_zero_prime (N : ℕ) (h_odd : Odd N) :
  (∀ i j, 2 ≤ i ∧ i < j ∧ j ≤ 1000 → N % i ≠ N % j) →
  (∃ k, 2 ≤ k ∧ k ≤ 1000 ∧ N % k = 0) →
  ∃ p, Prime p ∧ 500 < p ∧ p < 1000 ∧ N % p = 0 :=
sorry

end remainder_zero_prime_l4125_412531


namespace largest_mu_inequality_l4125_412515

theorem largest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  (∀ μ : ℝ, (a^2 + b^2 + c^2 + d^2 ≥ μ * a * b + b * c + 2 * c * d) → μ ≤ 13/2) ∧
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a^2 + b^2 + c^2 + d^2 ≥ 13/2 * a * b + b * c + 2 * c * d) :=
by sorry

end largest_mu_inequality_l4125_412515


namespace wheel_revolution_distance_l4125_412588

/-- Proves that given specific wheel sizes and revolution difference, the distance traveled is 315 feet -/
theorem wheel_revolution_distance 
  (back_wheel_perimeter : ℝ) 
  (front_wheel_perimeter : ℝ) 
  (revolution_difference : ℝ) 
  (h1 : back_wheel_perimeter = 9) 
  (h2 : front_wheel_perimeter = 7) 
  (h3 : revolution_difference = 10) :
  (front_wheel_perimeter⁻¹ - back_wheel_perimeter⁻¹)⁻¹ * revolution_difference = 315 :=
by sorry

end wheel_revolution_distance_l4125_412588


namespace polynomial_simplification_l4125_412586

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 =
  2*x^3 - x^2 + 23*x - 3 := by
  sorry

end polynomial_simplification_l4125_412586


namespace product_one_sum_at_least_two_l4125_412539

theorem product_one_sum_at_least_two (x : ℝ) (h1 : x > 0) (h2 : x * (1/x) = 1) : x + (1/x) ≥ 2 := by
  sorry

end product_one_sum_at_least_two_l4125_412539


namespace andy_max_cookies_l4125_412556

theorem andy_max_cookies (total_cookies : ℕ) (andy alexa alice : ℕ) : 
  total_cookies = 36 →
  alexa = 3 * andy →
  alice = 2 * andy →
  total_cookies = andy + alexa + alice →
  andy ≤ 6 ∧ ∃ (n : ℕ), n = 6 ∧ n = andy := by
  sorry

end andy_max_cookies_l4125_412556


namespace partner_q_active_months_l4125_412566

/-- Represents the investment and activity of a partner in the business -/
structure Partner where
  investment : ℝ
  monthlyReturn : ℝ
  activeMonths : ℕ

/-- Represents the business venture with three partners -/
structure Business where
  p : Partner
  q : Partner
  r : Partner
  totalProfit : ℝ

/-- The main theorem stating that partner Q was active for 6 months -/
theorem partner_q_active_months (b : Business) : b.q.activeMonths = 6 :=
  by
  have h1 : b.p.investment / b.q.investment = 7 / 5.00001 := sorry
  have h2 : b.q.investment / b.r.investment = 5.00001 / 3.99999 := sorry
  have h3 : b.p.monthlyReturn / b.q.monthlyReturn = 7.00001 / 10 := sorry
  have h4 : b.q.monthlyReturn / b.r.monthlyReturn = 10 / 6 := sorry
  have h5 : b.p.activeMonths = 5 := sorry
  have h6 : b.r.activeMonths = 8 := sorry
  have h7 : b.totalProfit = 200000 := sorry
  have h8 : b.p.investment * b.p.monthlyReturn * b.p.activeMonths = 50000 := sorry
  sorry

end partner_q_active_months_l4125_412566


namespace sequence_term_relation_l4125_412527

theorem sequence_term_relation (k : ℕ) (h_k : k > 1) :
  ∃ (a : ℕ → ℝ),
    (∀ n, a n ≥ a (n + 1)) ∧
    (∑' n, a n) = 1 ∧
    a 1 = 1 / (2 * k) ∧
    ∃ i₁ i₂ : Fin k, 
      ∀ j : Fin k, 
        a (i₁ : ℕ) ≥ a ((j : ℕ) + 1) ∧ 
        a ((i₂ : ℕ) + 1) > (1/2) * a (i₁ : ℕ) := by
  sorry

end sequence_term_relation_l4125_412527


namespace equation_solution_l4125_412525

theorem equation_solution (x : ℝ) : (40 / 80 = Real.sqrt (x / 80)) → x = 20 := by
  sorry

end equation_solution_l4125_412525


namespace coin_loss_recovery_l4125_412596

theorem coin_loss_recovery (x : ℚ) : 
  x > 0 → 
  let lost := x / 2
  let found := (4 / 5) * lost
  let remaining := x - lost + found
  x - remaining = x / 10 := by
sorry

end coin_loss_recovery_l4125_412596


namespace bruce_triple_age_in_six_years_l4125_412583

/-- The number of years it will take for Bruce to be three times as old as his son -/
def years_until_triple_age (bruce_age : ℕ) (son_age : ℕ) : ℕ :=
  let x : ℕ := (bruce_age - 3 * son_age) / 2
  x

/-- Theorem stating that it will take 6 years for Bruce to be three times as old as his son -/
theorem bruce_triple_age_in_six_years :
  years_until_triple_age 36 8 = 6 := by
  sorry

end bruce_triple_age_in_six_years_l4125_412583


namespace sock_selection_combinations_l4125_412567

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem sock_selection_combinations :
  choose 7 4 = 35 := by
  sorry

end sock_selection_combinations_l4125_412567


namespace sum_of_fifth_and_sixth_term_l4125_412533

theorem sum_of_fifth_and_sixth_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = n^3) : a 5 + a 6 = 152 := by
  sorry

end sum_of_fifth_and_sixth_term_l4125_412533


namespace function_bounded_by_identity_l4125_412522

/-- For a differentiable function f: ℝ → ℝ, if f(x) ≤ f'(x) for all x in ℝ, then f(x) ≤ x for all x in ℝ. -/
theorem function_bounded_by_identity (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x ≤ deriv f x) : ∀ x, f x ≤ x := by
  sorry

end function_bounded_by_identity_l4125_412522


namespace union_M_complement_N_equals_R_l4125_412506

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x < 2}
def N : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

-- State the theorem
theorem union_M_complement_N_equals_R : M ∪ Nᶜ = Set.univ :=
sorry

end union_M_complement_N_equals_R_l4125_412506


namespace amount_with_r_l4125_412535

/-- Given three people (p, q, r) with a total amount of 6000 among them,
    where r has two-thirds of the total amount that p and q have together,
    prove that the amount with r is 2400. -/
theorem amount_with_r (total : ℕ) (amount_r : ℕ) : 
  total = 6000 →
  amount_r = (2 / 3 : ℚ) * (total - amount_r) →
  amount_r = 2400 := by
sorry

end amount_with_r_l4125_412535


namespace cone_rolling_ratio_l4125_412501

/-- Represents a right circular cone -/
structure RightCircularCone where
  r : ℝ  -- base radius
  h : ℝ  -- height

/-- Represents the rolling properties of the cone -/
structure ConeRolling (cone : RightCircularCone) where
  rotations : ℕ
  no_slipping : Bool

theorem cone_rolling_ratio (cone : RightCircularCone) (rolling : ConeRolling cone) :
  rolling.rotations = 19 ∧ rolling.no_slipping = true →
  cone.h / cone.r = 6 * Real.sqrt 10 := by
  sorry

end cone_rolling_ratio_l4125_412501


namespace translation_problem_l4125_412574

-- Part 1
def part1 (A B A' B' : ℝ × ℝ) : Prop :=
  A = (-2, -1) ∧ B = (1, -3) ∧ A' = (2, 3) → B' = (5, 1)

-- Part 2
def part2 (A B A' B' : ℝ × ℝ) (m n : ℝ) : Prop :=
  A = (m, n) ∧ B = (2*n, m) ∧ A' = (3*m, n) ∧ B' = (6*n, m) → m = 2*n

-- Part 3
def part3 (A B A' B' : ℝ × ℝ) (m n : ℝ) : Prop :=
  A = (m, n+1) ∧ B = (n-1, n-2) ∧ A' = (2*n-5, 2*m+3) ∧ B' = (2*m+3, n+3) →
  A = (6, 10) ∧ B = (8, 7)

theorem translation_problem :
  ∀ (A B A' B' : ℝ × ℝ) (m n : ℝ),
    part1 A B A' B' ∧
    part2 A B A' B' m n ∧
    part3 A B A' B' m n :=
by sorry

end translation_problem_l4125_412574


namespace arithmetic_sequence_sum_l4125_412599

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) :
  (sum_arithmetic_sequence a₁ d 12 / 12 : ℚ) - (sum_arithmetic_sequence a₁ d 10 / 10 : ℚ) = 2 →
  sum_arithmetic_sequence a₁ d 2018 = -2018 :=
by
  sorry

end arithmetic_sequence_sum_l4125_412599


namespace goods_train_speed_l4125_412526

/-- The speed of a goods train passing another train in the opposite direction -/
theorem goods_train_speed (v_man : ℝ) (l_goods : ℝ) (t_pass : ℝ) :
  v_man = 40 →  -- Speed of man's train in km/h
  l_goods = 0.28 →  -- Length of goods train in km (280 m = 0.28 km)
  t_pass = 1 / 400 →  -- Time to pass in hours (9 seconds = 1/400 hours)
  ∃ v_goods : ℝ, v_goods = 72 ∧ (v_goods + v_man) * t_pass = l_goods :=
by sorry

end goods_train_speed_l4125_412526


namespace math_team_selection_l4125_412575

theorem math_team_selection (girls boys : ℕ) (h1 : girls = 4) (h2 : boys = 6) :
  (girls.choose 2) * (boys.choose 3) = 120 := by
  sorry

end math_team_selection_l4125_412575


namespace unsold_books_l4125_412514

theorem unsold_books (initial_stock : ℕ) (mon tue wed thu fri : ℕ) :
  initial_stock = 800 →
  mon = 60 →
  tue = 10 →
  wed = 20 →
  thu = 44 →
  fri = 66 →
  initial_stock - (mon + tue + wed + thu + fri) = 600 :=
by
  sorry

end unsold_books_l4125_412514


namespace mairead_exercise_ratio_l4125_412503

/-- Proves the ratio of miles walked to miles jogged for Mairead's exercise routine -/
theorem mairead_exercise_ratio :
  let miles_ran : ℝ := 40
  let miles_walked_fraction : ℝ := 3 / 5 * miles_ran
  let total_distance : ℝ := 184
  let miles_walked_multiple : ℝ := total_distance - miles_ran - miles_walked_fraction
  let total_miles_walked : ℝ := miles_walked_fraction + miles_walked_multiple
  total_miles_walked / miles_ran = 3.6 := by
  sorry

end mairead_exercise_ratio_l4125_412503


namespace negation_of_implication_intersection_l4125_412554

theorem negation_of_implication_intersection (A B : Set α) :
  ¬(∀ x, x ∈ A ∩ B → x ∈ A ∨ x ∈ B) ↔ ∃ x, x ∉ A ∩ B ∧ x ∉ A ∧ x ∉ B :=
sorry

end negation_of_implication_intersection_l4125_412554


namespace percentage_difference_l4125_412570

theorem percentage_difference (x y p : ℝ) (h : x = y * (1 + p / 100)) : 
  p = 100 * (x - y) / y :=
sorry

end percentage_difference_l4125_412570


namespace min_coach_handshakes_l4125_412595

/-- Represents the total number of handshakes -/
def total_handshakes : ℕ := 281

/-- Calculates the number of handshakes between gymnasts given the total number of gymnasts -/
def gymnast_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents the proposition that the coach's handshakes are minimized -/
def coach_handshakes_minimized (k : ℕ) : Prop :=
  ∃ (n : ℕ), 
    gymnast_handshakes n + k = total_handshakes ∧
    ∀ (m : ℕ), m > n → gymnast_handshakes m > total_handshakes

/-- The main theorem stating that the minimum number of coach's handshakes is 5 -/
theorem min_coach_handshakes : 
  ∃ (k : ℕ), k = 5 ∧ coach_handshakes_minimized k :=
sorry

end min_coach_handshakes_l4125_412595


namespace ones_digit_of_first_prime_in_sequence_l4125_412553

-- Define the property of being a prime number
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define the property of being an increasing arithmetic sequence
def isIncreasingArithmeticSequence (a b c d : ℕ) : Prop :=
  b - a = c - b ∧ c - b = d - c ∧ a < b ∧ b < c ∧ c < d

-- Define the ones digit of a natural number
def onesDigit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_first_prime_in_sequence (p q r s : ℕ) :
  isPrime p → isPrime q → isPrime r → isPrime s →
  isIncreasingArithmeticSequence p q r s →
  q - p = 4 →
  p > 5 →
  onesDigit p = 9 :=
sorry

end ones_digit_of_first_prime_in_sequence_l4125_412553


namespace tan_45_degrees_equals_one_l4125_412598

theorem tan_45_degrees_equals_one : Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_equals_one_l4125_412598


namespace min_correct_answers_to_pass_l4125_412559

/-- Represents the Fire Safety quiz selection -/
structure FireSafetyQuiz where
  total_questions : Nat
  correct_score : Int
  incorrect_score : Int
  passing_score : Int

/-- Calculates the total score based on the number of correct answers -/
def calculate_score (quiz : FireSafetyQuiz) (correct_answers : Nat) : Int :=
  (quiz.correct_score * correct_answers) + 
  (quiz.incorrect_score * (quiz.total_questions - correct_answers))

/-- Theorem: The minimum number of correct answers needed to pass the Fire Safety quiz is 12 -/
theorem min_correct_answers_to_pass (quiz : FireSafetyQuiz) 
  (h1 : quiz.total_questions = 20)
  (h2 : quiz.correct_score = 10)
  (h3 : quiz.incorrect_score = -5)
  (h4 : quiz.passing_score = 80) :
  ∀ n : Nat, calculate_score quiz n ≥ quiz.passing_score → n ≥ 12 :=
by sorry

end min_correct_answers_to_pass_l4125_412559


namespace optimal_bottle_volume_l4125_412500

theorem optimal_bottle_volume (vol1 vol2 vol3 : ℕ) 
  (h1 : vol1 = 4200) (h2 : vol2 = 3220) (h3 : vol3 = 2520) :
  Nat.gcd vol1 (Nat.gcd vol2 vol3) = 140 := by
  sorry

end optimal_bottle_volume_l4125_412500


namespace school_seats_cost_l4125_412551

/-- Calculate the total cost of seats with a group discount -/
def totalCostWithDiscount (rows : ℕ) (seatsPerRow : ℕ) (costPerSeat : ℕ) (discountPercent : ℕ) : ℕ :=
  let totalSeats := rows * seatsPerRow
  let fullGroupsOf10 := totalSeats / 10
  let costPer10Seats := 10 * costPerSeat
  let discountPer10Seats := costPer10Seats * discountPercent / 100
  let costPer10SeatsAfterDiscount := costPer10Seats - discountPer10Seats
  fullGroupsOf10 * costPer10SeatsAfterDiscount

theorem school_seats_cost :
  totalCostWithDiscount 5 8 30 10 = 1080 := by
  sorry

end school_seats_cost_l4125_412551


namespace white_pairs_count_l4125_412572

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCount where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of triangles when folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_blue : ℕ

/-- The main theorem stating the number of coinciding white pairs -/
theorem white_pairs_count (half_count : TriangleCount) (coinciding : CoincidingPairs) : 
  half_count.red = 4 ∧ 
  half_count.blue = 4 ∧ 
  half_count.white = 6 ∧
  coinciding.red_red = 3 ∧
  coinciding.blue_blue = 2 ∧
  coinciding.red_blue = 1 →
  (half_count.white : ℤ) = 6 := by
  sorry

end white_pairs_count_l4125_412572


namespace sine_of_angle_through_point_l4125_412532

theorem sine_of_angle_through_point (α : Real) :
  let P : Real × Real := (Real.cos (3 * Real.pi / 4), Real.sin (3 * Real.pi / 4))
  (∃ k : Real, k > 0 ∧ (k * Real.cos α = P.1) ∧ (k * Real.sin α = P.2)) →
  Real.sin α = Real.sqrt 2 / 2 ∨ Real.sin α = -Real.sqrt 2 / 2 := by
  sorry

end sine_of_angle_through_point_l4125_412532


namespace crazy_silly_school_books_l4125_412565

/-- The total number of books in the 'crazy silly school' series -/
def total_books (x y : ℕ) : ℕ := x^2 + y

/-- Theorem stating that the total number of books is 177 when x = 13 and y = 8 -/
theorem crazy_silly_school_books : total_books 13 8 = 177 := by
  sorry

end crazy_silly_school_books_l4125_412565


namespace square_difference_equals_324_l4125_412519

theorem square_difference_equals_324 : (422 + 404)^2 - (4 * 422 * 404) = 324 := by
  sorry

end square_difference_equals_324_l4125_412519


namespace f_prime_at_zero_l4125_412562

noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) * Real.exp x

theorem f_prime_at_zero : 
  (deriv f) 0 = 3 := by sorry

end f_prime_at_zero_l4125_412562


namespace two_thousand_two_in_sequence_l4125_412528

def next_in_sequence (b : ℕ) : ℕ :=
  b + (Nat.factors b).reverse.head!

def is_in_sequence (a : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, Nat.iterate next_in_sequence k a = n

theorem two_thousand_two_in_sequence (a : ℕ) :
  a > 1 → (is_in_sequence a 2002 ↔ a = 1859 ∨ a = 1991) :=
sorry

end two_thousand_two_in_sequence_l4125_412528


namespace calculate_expression_l4125_412581

theorem calculate_expression : (-1)^2022 + Real.sqrt 9 - 2 * Real.sin (30 * π / 180) = 3 := by
  sorry

end calculate_expression_l4125_412581


namespace quadratic_function_m_range_l4125_412541

/-- Given a quadratic function y = (m-2)x^2 + 2mx - (3-m), prove that the range of m
    satisfying all conditions is 2 < m < 3 --/
theorem quadratic_function_m_range (m : ℝ) : 
  let f (x : ℝ) := (m - 2) * x^2 + 2 * m * x - (3 - m)
  let vertex_x := -m / (m - 2)
  let vertex_y := (-5 * m + 6) / (m - 2)
  (∀ x, (m - 2) * x^2 + 2 * m * x - (3 - m) = f x) →
  (vertex_x < 0 ∧ vertex_y < 0) →
  (m - 2 > 0) →
  (-(3 - m) < 0) →
  (2 < m ∧ m < 3) := by
  sorry

end quadratic_function_m_range_l4125_412541


namespace complex_square_root_expression_l4125_412512

theorem complex_square_root_expression : 71 * Real.sqrt (3 + 2 * Real.sqrt 2) - Real.sqrt (3 - 2 * Real.sqrt 2) = 2 := by
  sorry

end complex_square_root_expression_l4125_412512


namespace gift_worth_l4125_412582

/-- Calculates the worth of each gift given the company's structure and budget --/
theorem gift_worth (num_blocks : ℕ) (workers_per_block : ℕ) (total_budget : ℕ) : 
  num_blocks = 15 → 
  workers_per_block = 200 → 
  total_budget = 6000 → 
  (total_budget : ℚ) / (num_blocks * workers_per_block : ℚ) = 2 := by
  sorry

#check gift_worth

end gift_worth_l4125_412582


namespace f_has_two_zeros_h_range_l4125_412576

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x - a) - x
def g (a : ℝ) (x : ℝ) : ℝ := Real.exp (x - a) - x * Real.log x + (1 - a) * x

-- Define the set of a values for g
def A : Set ℝ := Set.Ioo 1 (3 - Real.log 3)

-- Define h(a) as the local minimum of g(x) for a given a
noncomputable def h (a : ℝ) : ℝ := 
  let x₂ := Real.exp a
  2 * x₂ - x₂^2

-- Theorem statements
theorem f_has_two_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ a > 1 := by sorry

theorem h_range :
  Set.range h = Set.Icc (-3) 1 := by sorry

end

end f_has_two_zeros_h_range_l4125_412576


namespace x_plus_y_value_l4125_412536

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.cos y = 2010)
  (eq2 : x + 2010 * Real.sin y = 2009)
  (y_range : π / 2 ≤ y ∧ y ≤ π) :
  x + y = 2011 + π := by
  sorry

end x_plus_y_value_l4125_412536


namespace banana_division_existence_l4125_412589

theorem banana_division_existence :
  ∃ (n : ℕ) (b₁ b₂ b₃ b₄ : ℕ),
    n = b₁ + b₂ + b₃ + b₄ ∧
    (5 * (5 * b₁ + 4 * b₂ + 8 * b₃ + 6 * b₄) =
     3 * (b₁ + 10 * b₂ + 8 * b₃ + 6 * b₄)) ∧
    (5 * (b₁ + 4 * b₂ + 8 * b₃ + 6 * b₄) =
     2 * (b₁ + 4 * b₂ + 9 * b₃ + 6 * b₄)) ∧
    (5 * (b₁ + 4 * b₂ + 8 * b₃ + 6 * b₄) =
     (b₁ + 4 * b₂ + 8 * b₃ + 12 * b₄)) ∧
    (15 ∣ b₁) ∧ (15 ∣ b₂) ∧ (27 ∣ b₃) ∧ (36 ∣ b₄) :=
by sorry

#check banana_division_existence

end banana_division_existence_l4125_412589


namespace cabbage_sales_theorem_l4125_412550

/-- Calculates the total kilograms of cabbage sold given the price per kilogram and earnings from three days -/
def total_cabbage_sold (price_per_kg : ℚ) (day1_earnings day2_earnings day3_earnings : ℚ) : ℚ :=
  (day1_earnings + day2_earnings + day3_earnings) / price_per_kg

/-- Theorem stating that given the specific conditions, the total cabbage sold is 48 kg -/
theorem cabbage_sales_theorem :
  let price_per_kg : ℚ := 2
  let day1_earnings : ℚ := 30
  let day2_earnings : ℚ := 24
  let day3_earnings : ℚ := 42
  total_cabbage_sold price_per_kg day1_earnings day2_earnings day3_earnings = 48 := by
  sorry

end cabbage_sales_theorem_l4125_412550


namespace square_value_l4125_412568

theorem square_value (square : ℝ) : 
  (1.08 / 1.2) / 2.3 = 10.8 / square → square = 27.6 := by
  sorry

end square_value_l4125_412568


namespace square_recurrence_cube_recurrence_l4125_412597

-- Define the sequences
def a (n : ℕ) : ℕ := n^2
def b (n : ℕ) : ℕ := n^3

-- Theorem for the linear recurrence relation of a_n = n^2
theorem square_recurrence (n : ℕ) (h : n ≥ 3) :
  a n = 3 * a (n - 1) - 3 * a (n - 2) + a (n - 3) := by
  sorry

-- Theorem for the linear recurrence relation of a_n = n^3
theorem cube_recurrence (n : ℕ) (h : n ≥ 4) :
  b n = 4 * b (n - 1) - 6 * b (n - 2) + 4 * b (n - 3) - b (n - 4) := by
  sorry

end square_recurrence_cube_recurrence_l4125_412597


namespace smallest_common_factor_l4125_412592

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m < 43 → Nat.gcd (8*m - 3) (5*m + 2) = 1) ∧ 
  Nat.gcd (8*43 - 3) (5*43 + 2) > 1 :=
sorry

end smallest_common_factor_l4125_412592
