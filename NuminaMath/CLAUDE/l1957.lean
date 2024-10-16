import Mathlib

namespace NUMINAMATH_CALUDE_sequence_3078th_term_l1957_195712

/-- Calculates the sum of cubes of digits of a natural number -/
def sumOfCubesOfDigits (n : ℕ) : ℕ := sorry

/-- Generates the next term in the sequence -/
def nextTerm (n : ℕ) : ℕ := sumOfCubesOfDigits n

/-- Generates the nth term of the sequence starting with the given initial term -/
def nthTerm (initial : ℕ) (n : ℕ) : ℕ := sorry

/-- The main theorem to prove -/
theorem sequence_3078th_term (initial : ℕ) (h : initial = 3078) : 
  nthTerm initial 3078 = 153 := by sorry

end NUMINAMATH_CALUDE_sequence_3078th_term_l1957_195712


namespace NUMINAMATH_CALUDE_range_of_p_l1957_195727

-- Define the function p(x)
def p (x : ℝ) : ℝ := x^6 + 6*x^3 + 9

-- State the theorem
theorem range_of_p :
  {y : ℝ | ∃ x ≥ 0, p x = y} = {y : ℝ | y ≥ 9} :=
sorry

end NUMINAMATH_CALUDE_range_of_p_l1957_195727


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1957_195792

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 24 → volume = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1957_195792


namespace NUMINAMATH_CALUDE_parabola_locus_l1957_195759

-- Define the parabola and its properties
def is_prime (p : ℕ) : Prop := Nat.Prime p

-- Define the locus L
def locus (p : ℕ) (x y : ℝ) : Prop :=
  is_prime p ∧ p ≠ 2 ∧ y ≠ 0 ∧ 4 * y^2 = p * (x - p)

-- Theorem statement
theorem parabola_locus (p : ℕ) :
  is_prime p →
  p ≠ 2 →
  (∃ (x y : ℤ), locus p (x : ℝ) (y : ℝ)) ∧
  (∀ (x y : ℤ), locus p (x : ℝ) (y : ℝ) → ¬ ∃ (m : ℤ), (x : ℝ)^2 + (y : ℝ)^2 = (m : ℝ)^2) ∧
  (∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ locus p (x : ℝ) (y : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_parabola_locus_l1957_195759


namespace NUMINAMATH_CALUDE_domain_of_f_l1957_195732

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define our function f(x) = lg(x+1)
noncomputable def f (x : ℝ) := lg (x + 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | f x = f x} = {x : ℝ | x > -1} :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l1957_195732


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l1957_195723

/-- The quadratic function f(x) = -(x+1)^2 - 8 -/
def f (x : ℝ) : ℝ := -(x + 1)^2 - 8

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (-1, -8)

/-- Theorem: The vertex of the quadratic function f(x) = -(x+1)^2 - 8 is at the point (-1, -8) -/
theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≤ f (vertex.1)) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l1957_195723


namespace NUMINAMATH_CALUDE_marble_selection_probability_l1957_195729

/-- The probability of selecting 2 red, 1 blue, and 1 green marble when choosing 4 marbles
    without replacement from a bag containing 3 red, 3 blue, and 3 green marbles. -/
theorem marble_selection_probability :
  let total_marbles : ℕ := 9
  let red_marbles : ℕ := 3
  let blue_marbles : ℕ := 3
  let green_marbles : ℕ := 3
  let selected_marbles : ℕ := 4
  let favorable_outcomes : ℕ := Nat.choose red_marbles 2 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 1
  let total_outcomes : ℕ := Nat.choose total_marbles selected_marbles
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 14 :=
by sorry

end NUMINAMATH_CALUDE_marble_selection_probability_l1957_195729


namespace NUMINAMATH_CALUDE_min_packs_for_120_cans_l1957_195781

/-- Represents the available pack sizes for soda cans -/
inductive PackSize
  | small : PackSize  -- 9 cans
  | medium : PackSize -- 18 cans
  | large : PackSize  -- 30 cans

/-- Calculates the number of cans in a given pack -/
def cansInPack (p : PackSize) : ℕ :=
  match p with
  | .small => 9
  | .medium => 18
  | .large => 30

/-- Represents a combination of packs -/
structure PackCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of cans in a pack combination -/
def totalCans (c : PackCombination) : ℕ :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Checks if a combination qualifies for the promotion -/
def qualifiesForPromotion (c : PackCombination) : Bool :=
  c.large ≥ 2

/-- Represents the store's promotion rule -/
def applyPromotion (c : PackCombination) : PackCombination :=
  if qualifiesForPromotion c then
    { c with small := c.small + 1 }
  else
    c

/-- Calculates the total number of packs in a combination -/
def totalPacks (c : PackCombination) : ℕ :=
  c.small + c.medium + c.large

/-- The main theorem to prove -/
theorem min_packs_for_120_cans :
  ∃ (c : PackCombination),
    totalCans (applyPromotion c) = 120 ∧
    totalPacks c = 4 ∧
    (∀ (c' : PackCombination),
      totalCans (applyPromotion c') = 120 →
      totalPacks c' ≥ totalPacks c) :=
  sorry


end NUMINAMATH_CALUDE_min_packs_for_120_cans_l1957_195781


namespace NUMINAMATH_CALUDE_train_speed_l1957_195703

/-- Proves that a train of given length crossing a platform of given length in a given time has a specific speed in km/hr -/
theorem train_speed (train_length platform_length : ℝ) (crossing_time : ℝ) :
  train_length = 230 ∧ 
  platform_length = 290 ∧ 
  crossing_time = 26 →
  (train_length + platform_length) / crossing_time * 3.6 = 72 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l1957_195703


namespace NUMINAMATH_CALUDE_unique_repeated_solution_pair_l1957_195779

/-- A function that checks if a quadratic equation has exactly one repeated real solution -/
def has_one_repeated_solution (a b c : ℤ) : Prop :=
  b ^ 2 = 4 * a * c

/-- The theorem stating that there exists exactly one ordered pair (b, c) of positive integers
    such that both x^2 + bx + c = 0 and x^2 + cx + b = 0 have exactly one repeated real solution -/
theorem unique_repeated_solution_pair :
  ∃! p : ℤ × ℤ, 
    p.1 > 0 ∧ p.2 > 0 ∧ 
    has_one_repeated_solution 1 p.1 p.2 ∧
    has_one_repeated_solution 1 p.2 p.1 :=
  sorry

end NUMINAMATH_CALUDE_unique_repeated_solution_pair_l1957_195779


namespace NUMINAMATH_CALUDE_quadratic_roots_integer_P_l1957_195778

theorem quadratic_roots_integer_P (P : ℤ) 
  (h1 : 5 < P) (h2 : P < 20) 
  (h3 : ∃ x y : ℤ, x^2 - 2*(2*P - 3)*x + 4*P^2 - 14*P + 8 = 0 ∧ 
                   y^2 - 2*(2*P - 3)*y + 4*P^2 - 14*P + 8 = 0 ∧ 
                   x ≠ y) : 
  P = 12 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_integer_P_l1957_195778


namespace NUMINAMATH_CALUDE_solution_exists_in_interval_l1957_195754

-- Define the function f(x) = x^3 + x - 3
def f (x : ℝ) : ℝ := x^3 + x - 3

-- State the theorem
theorem solution_exists_in_interval :
  ∃! r : ℝ, r ∈ Set.Icc 1 2 ∧ f r = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_in_interval_l1957_195754


namespace NUMINAMATH_CALUDE_simplify_expression_l1957_195775

theorem simplify_expression (x y : ℝ) : (x - 3*y + 2) * (x + 3*y + 2) = x^2 + 4*x + 4 - 9*y^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1957_195775


namespace NUMINAMATH_CALUDE_fraction_problem_l1957_195747

theorem fraction_problem (f : ℚ) : f * 300 = (3/5 * 125) + 45 → f = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1957_195747


namespace NUMINAMATH_CALUDE_sequence1_correct_sequence2_correct_sequence3_correct_l1957_195710

-- Sequence 1
def sequence1 (n : ℕ) : ℤ := (-1)^n * (6*n - 5)

-- Sequence 2
def sequence2 (n : ℕ) : ℚ := 8/9 * (1 - 1/10^n)

-- Sequence 3
def sequence3 (n : ℕ) : ℚ := (-1)^n * (2^n - 3) / 2^n

theorem sequence1_correct (n : ℕ) : 
  sequence1 1 = -1 ∧ sequence1 2 = 7 ∧ sequence1 3 = -13 ∧ sequence1 4 = 19 := by sorry

theorem sequence2_correct (n : ℕ) : 
  sequence2 1 = 0.8 ∧ sequence2 2 = 0.88 ∧ sequence2 3 = 0.888 := by sorry

theorem sequence3_correct (n : ℕ) : 
  sequence3 1 = -1/2 ∧ sequence3 2 = 1/4 ∧ sequence3 3 = -5/8 ∧ 
  sequence3 4 = 13/16 ∧ sequence3 5 = -29/32 ∧ sequence3 6 = 61/64 := by sorry

end NUMINAMATH_CALUDE_sequence1_correct_sequence2_correct_sequence3_correct_l1957_195710


namespace NUMINAMATH_CALUDE_xyz_value_l1957_195719

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 15)
  (h3 : x + y + z = 5) :
  x * y * z = 10 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1957_195719


namespace NUMINAMATH_CALUDE_difference_of_squares_l1957_195786

theorem difference_of_squares (a : ℝ) : a^2 - 4 = (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1957_195786


namespace NUMINAMATH_CALUDE_work_completion_days_l1957_195715

/-- Calculates the initial planned days to complete a work given the original number of workers,
    the number of absent workers, and the days taken by the remaining workers. -/
def initialPlannedDays (originalWorkers : ℕ) (absentWorkers : ℕ) (daysWithFewerWorkers : ℕ) : ℚ :=
  (originalWorkers - absentWorkers) * daysWithFewerWorkers / originalWorkers

/-- Proves that given 15 original workers, 5 absent workers, and 60 days taken by the remaining workers,
    the initial planned days to complete the work is 40. -/
theorem work_completion_days : initialPlannedDays 15 5 60 = 40 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_days_l1957_195715


namespace NUMINAMATH_CALUDE_room_length_is_twelve_l1957_195745

/-- Represents the dimensions and carpet placement of a rectangular room. -/
structure RoomWithCarpet where
  length : ℝ
  width : ℝ
  borderWidth : ℝ

/-- Calculates the area of the border given room dimensions and border width. -/
def borderArea (room : RoomWithCarpet) : ℝ :=
  room.length * room.width - (room.length - 2 * room.borderWidth) * (room.width - 2 * room.borderWidth)

/-- Theorem: If a rectangular room has width 10 feet, a carpet is placed leaving a 2-foot 
    wide border all around, and the area of the border is 72 square feet, then the length 
    of the room is 12 feet. -/
theorem room_length_is_twelve (room : RoomWithCarpet) 
    (h1 : room.width = 10)
    (h2 : room.borderWidth = 2)
    (h3 : borderArea room = 72) : 
  room.length = 12 := by
  sorry


end NUMINAMATH_CALUDE_room_length_is_twelve_l1957_195745


namespace NUMINAMATH_CALUDE_bruce_bags_l1957_195756

/-- Calculates the number of bags Bruce can buy with the change after purchasing crayons, books, and calculators. -/
def bags_bought (crayons_packs : ℕ) (crayon_price : ℕ) (books : ℕ) (book_price : ℕ) 
                (calculators : ℕ) (calculator_price : ℕ) (initial_money : ℕ) (bag_price : ℕ) : ℕ :=
  let total_spent := crayons_packs * crayon_price + books * book_price + calculators * calculator_price
  let change := initial_money - total_spent
  change / bag_price

/-- Theorem stating that Bruce can buy 11 bags with the change. -/
theorem bruce_bags : 
  bags_bought 5 5 10 5 3 5 200 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_bruce_bags_l1957_195756


namespace NUMINAMATH_CALUDE_student_distribution_problem_l1957_195721

/-- The number of ways to distribute n distinguishable students among k distinguishable schools,
    with each school receiving at least one student. -/
def distribute_students (n k : ℕ) : ℕ :=
  if n < k then 0
  else (k.choose 2) * k.factorial

/-- The problem statement -/
theorem student_distribution_problem :
  distribute_students 4 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_student_distribution_problem_l1957_195721


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l1957_195711

theorem square_sum_geq_product_sum {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l1957_195711


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l1957_195785

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l1957_195785


namespace NUMINAMATH_CALUDE_lcm_28_72_l1957_195713

theorem lcm_28_72 : Nat.lcm 28 72 = 504 := by
  sorry

end NUMINAMATH_CALUDE_lcm_28_72_l1957_195713


namespace NUMINAMATH_CALUDE_profit_share_ratio_l1957_195762

/-- The ratio of profit shares for two investors is proportional to their investments -/
theorem profit_share_ratio (p_investment q_investment : ℕ) :
  p_investment = 30000 →
  q_investment = 45000 →
  (p_investment : ℚ) / (p_investment + q_investment) = 2 / 5 ∧
  (q_investment : ℚ) / (p_investment + q_investment) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_ratio_l1957_195762


namespace NUMINAMATH_CALUDE_modular_inverse_5_mod_19_l1957_195773

theorem modular_inverse_5_mod_19 : ∃ x : ℕ, x < 19 ∧ (5 * x) % 19 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_5_mod_19_l1957_195773


namespace NUMINAMATH_CALUDE_solve_seed_problem_l1957_195761

def seed_problem (total_seeds : ℕ) (left_seeds : ℕ) (right_multiplier : ℕ) (seeds_left : ℕ) : Prop :=
  let right_seeds := right_multiplier * left_seeds
  let initially_thrown := left_seeds + right_seeds
  let joined_later := total_seeds - initially_thrown - seeds_left
  joined_later = total_seeds - (left_seeds + right_multiplier * left_seeds) - seeds_left

theorem solve_seed_problem :
  seed_problem 120 20 2 30 := by
  sorry

end NUMINAMATH_CALUDE_solve_seed_problem_l1957_195761


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l1957_195753

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle :=
  (a b c : ℝ)

/-- Checks if two triangles are similar -/
def are_similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c

theorem similar_triangles_side_length 
  (FGH IJK : Triangle)
  (h_similar : are_similar FGH IJK)
  (h_GH : FGH.c = 30)
  (h_FG : FGH.a = 24)
  (h_IJ : IJK.a = 20) :
  IJK.c = 25 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l1957_195753


namespace NUMINAMATH_CALUDE_henrysFriendMoney_l1957_195700

/-- Calculates the amount of money Henry's friend has -/
def friendsMoney (henryInitial : ℕ) (henryEarned : ℕ) (totalCombined : ℕ) : ℕ :=
  totalCombined - (henryInitial + henryEarned)

/-- Theorem: Henry's friend has 13 dollars -/
theorem henrysFriendMoney : friendsMoney 5 2 20 = 13 := by
  sorry

end NUMINAMATH_CALUDE_henrysFriendMoney_l1957_195700


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1957_195799

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (10 + n) = 8 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1957_195799


namespace NUMINAMATH_CALUDE_least_integer_with_nine_factors_l1957_195722

/-- A function that returns the number of distinct positive factors of a positive integer -/
def number_of_factors (n : ℕ+) : ℕ :=
  sorry

/-- A function that checks if a number has exactly nine distinct positive factors -/
def has_nine_factors (n : ℕ+) : Prop :=
  number_of_factors n = 9

theorem least_integer_with_nine_factors :
  ∃ (n : ℕ+), has_nine_factors n ∧ ∀ (m : ℕ+), has_nine_factors m → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_nine_factors_l1957_195722


namespace NUMINAMATH_CALUDE_boosters_club_average_net_sales_l1957_195771

/-- Calculates the average net sales per month given monthly sales and expenses -/
def average_net_sales (monthly_sales : List ℕ) (monthly_expense : ℕ) : ℚ :=
  let total_sales := monthly_sales.sum
  let total_expenses := monthly_expense * monthly_sales.length
  let net_sales := total_sales - total_expenses
  (net_sales : ℚ) / monthly_sales.length

/-- The average net sales per month for the Boosters Club is $75 -/
theorem boosters_club_average_net_sales :
  average_net_sales [120, 80, 50, 130, 90, 160] 30 = 75 := by
  sorry

#eval average_net_sales [120, 80, 50, 130, 90, 160] 30

end NUMINAMATH_CALUDE_boosters_club_average_net_sales_l1957_195771


namespace NUMINAMATH_CALUDE_even_number_2018_in_group_27_l1957_195707

/-- The sum of the number of elements in the first n groups --/
def S (n : ℕ) : ℕ := (3 * n^2 - n) / 2

/-- The proposition that 2018 is in the 27th group --/
theorem even_number_2018_in_group_27 :
  S 26 < 1009 ∧ 1009 ≤ S 27 :=
sorry

end NUMINAMATH_CALUDE_even_number_2018_in_group_27_l1957_195707


namespace NUMINAMATH_CALUDE_peter_bought_five_large_glasses_l1957_195741

/-- Represents the purchase of glasses by Peter -/
structure GlassesPurchase where
  small_cost : ℕ             -- Cost of a small glass
  large_cost : ℕ             -- Cost of a large glass
  total_money : ℕ            -- Total money Peter has
  small_bought : ℕ           -- Number of small glasses bought
  change : ℕ                 -- Money left as change

/-- Calculates the number of large glasses Peter bought -/
def large_glasses_bought (purchase : GlassesPurchase) : ℕ :=
  (purchase.total_money - purchase.small_cost * purchase.small_bought - purchase.change) / purchase.large_cost

/-- Theorem stating that Peter bought 5 large glasses -/
theorem peter_bought_five_large_glasses :
  ∀ (purchase : GlassesPurchase),
    purchase.small_cost = 3 →
    purchase.large_cost = 5 →
    purchase.total_money = 50 →
    purchase.small_bought = 8 →
    purchase.change = 1 →
    large_glasses_bought purchase = 5 := by
  sorry


end NUMINAMATH_CALUDE_peter_bought_five_large_glasses_l1957_195741


namespace NUMINAMATH_CALUDE_exotic_courses_divisibility_l1957_195749

/-- Represents a country with airports -/
structure Country where
  airports : ℕ

/-- Represents the flight system between two countries -/
structure FlightSystem where
  countryA : Country
  countryB : Country
  flightsPerAirport : ℕ
  noInternalFlights : Bool

/-- Represents an exotic traveling course -/
structure ExoticTravelingCourse where
  flightSystem : FlightSystem
  courseLength : ℕ

/-- The number of all exotic traveling courses -/
def numberOfExoticCourses (f : FlightSystem) : ℕ :=
  sorry

theorem exotic_courses_divisibility (f : FlightSystem) 
  (h1 : f.countryA.airports = f.countryB.airports)
  (h2 : f.countryA.airports ≥ 2)
  (h3 : f.flightsPerAirport = 3)
  (h4 : f.noInternalFlights = true) :
  ∃ k : ℕ, numberOfExoticCourses f = 8 * f.countryA.airports * k ∧ Even k :=
sorry

end NUMINAMATH_CALUDE_exotic_courses_divisibility_l1957_195749


namespace NUMINAMATH_CALUDE_expression_simplification_l1957_195718

theorem expression_simplification (b : ℝ) : 
  ((3 * b + 10 - 5 * b^2) / 5) = -b^2 + (3 * b / 5) + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1957_195718


namespace NUMINAMATH_CALUDE_student_calculation_error_l1957_195763

theorem student_calculation_error (x : ℝ) : 
  (8/7) * x = (4/5) * x + 15.75 → x = 45.9375 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_error_l1957_195763


namespace NUMINAMATH_CALUDE_max_gcd_sum_780_l1957_195720

theorem max_gcd_sum_780 :
  ∃ (a b : ℕ+), a + b = 780 ∧ 
  ∀ (c d : ℕ+), c + d = 780 → Nat.gcd c d ≤ Nat.gcd a b ∧
  Nat.gcd a b = 390 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_sum_780_l1957_195720


namespace NUMINAMATH_CALUDE_locust_jump_symmetry_l1957_195797

/-- A locust on a line -/
structure Locust where
  position : ℝ

/-- A configuration of locusts on a line -/
def LocustConfiguration := List Locust

/-- A property that can be achieved by a locust configuration -/
def ConfigurationProperty := LocustConfiguration → Prop

/-- Jumping to the right -/
def jumpRight (config : LocustConfiguration) : LocustConfiguration := sorry

/-- Jumping to the left -/
def jumpLeft (config : LocustConfiguration) : LocustConfiguration := sorry

/-- Two locusts are 1 mm apart -/
def twoLocustsOneMillimeterApart (config : LocustConfiguration) : Prop := sorry

theorem locust_jump_symmetry 
  (initial_config : LocustConfiguration) 
  (h : ∃ (final_config : LocustConfiguration), 
       twoLocustsOneMillimeterApart final_config ∧ 
       ∃ (n : ℕ), final_config = (jumpRight^[n]) initial_config) :
  ∃ (left_final_config : LocustConfiguration), 
    twoLocustsOneMillimeterApart left_final_config ∧ 
    ∃ (m : ℕ), left_final_config = (jumpLeft^[m]) initial_config := 
by sorry

end NUMINAMATH_CALUDE_locust_jump_symmetry_l1957_195797


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1957_195772

theorem complex_equation_sum (a b : ℕ+) :
  Complex.abs ((a + Complex.I) * (2 + Complex.I)) =
  Complex.abs ((b - Complex.I) / (2 - Complex.I)) →
  a + b = 8 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1957_195772


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l1957_195731

theorem arctan_sum_equals_pi_over_four (n : ℕ+) :
  Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/5) + Real.arctan (1/n) = π/4 →
  n = 47 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l1957_195731


namespace NUMINAMATH_CALUDE_min_coins_for_distribution_l1957_195737

/-- The minimum number of additional coins needed for distribution -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed -/
theorem min_coins_for_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 95) :
  min_additional_coins num_friends initial_coins = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_coins_for_distribution_l1957_195737


namespace NUMINAMATH_CALUDE_log_sum_simplification_l1957_195791

theorem log_sum_simplification :
  1 / (Real.log 3 / Real.log 18 + 1) +
  1 / (Real.log 2 / Real.log 12 + 1) +
  1 / (Real.log 7 / Real.log 8 + 1) =
  13 / 12 := by sorry

end NUMINAMATH_CALUDE_log_sum_simplification_l1957_195791


namespace NUMINAMATH_CALUDE_trapezoid_reconstruction_l1957_195730

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Checks if two line segments are parallel -/
def parallel (p q r s : Point) : Prop :=
  (q.y - p.y) * (s.x - r.x) = (s.y - r.y) * (q.x - p.x)

/-- Checks if a point divides two line segments proportionally -/
def divides_proportionally (o p q r s : Point) : Prop :=
  (o.x - p.x) * (s.y - q.y) = (o.y - p.y) * (s.x - q.x)

/-- Theorem: Given three points A, B, C, and a point O, 
    there exists a point D such that ABCD forms a trapezoid 
    with O as the intersection of its diagonals -/
theorem trapezoid_reconstruction 
  (A B C O : Point) 
  (h1 : collinear A O C) 
  (h2 : ¬ collinear A B C) : 
  ∃ D : Point, 
    parallel A B C D ∧ 
    collinear B O D ∧
    divides_proportionally O A C B D :=
sorry

end NUMINAMATH_CALUDE_trapezoid_reconstruction_l1957_195730


namespace NUMINAMATH_CALUDE_canoe_rowing_probability_l1957_195768

theorem canoe_rowing_probability (p : ℝ) (h_p : p = 3/5) :
  let q := 1 - p
  p * p + p * q + q * p = 21/25 := by sorry

end NUMINAMATH_CALUDE_canoe_rowing_probability_l1957_195768


namespace NUMINAMATH_CALUDE_monomial_properties_l1957_195760

/-- Represents a monomial -3a²bc/5 -/
structure Monomial where
  coefficient : ℚ
  a_exponent : ℕ
  b_exponent : ℕ
  c_exponent : ℕ

/-- The specific monomial -3a²bc/5 -/
def our_monomial : Monomial :=
  { coefficient := -3/5
    a_exponent := 2
    b_exponent := 1
    c_exponent := 1 }

/-- The coefficient of a monomial is its numerical factor -/
def get_coefficient (m : Monomial) : ℚ := m.coefficient

/-- The degree of a monomial is the sum of its variable exponents -/
def get_degree (m : Monomial) : ℕ := m.a_exponent + m.b_exponent + m.c_exponent

theorem monomial_properties :
  (get_coefficient our_monomial = -3/5) ∧ (get_degree our_monomial = 4) := by
  sorry


end NUMINAMATH_CALUDE_monomial_properties_l1957_195760


namespace NUMINAMATH_CALUDE_baseball_ticket_cost_is_8_l1957_195790

/-- Calculates the cost of a baseball ticket given initial amount, cost of hot dog, and remaining amount -/
def baseball_ticket_cost (initial_amount : ℕ) (hot_dog_cost : ℕ) (remaining_amount : ℕ) : ℕ :=
  initial_amount - hot_dog_cost - remaining_amount

/-- Proves that the cost of the baseball ticket is 8 given the specified conditions -/
theorem baseball_ticket_cost_is_8 :
  baseball_ticket_cost 20 3 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_baseball_ticket_cost_is_8_l1957_195790


namespace NUMINAMATH_CALUDE_hypotenuse_length_l1957_195735

-- Define a right triangle with legs 3 and 5
def right_triangle (a b c : ℝ) : Prop :=
  a = 3 ∧ b = 5 ∧ c^2 = a^2 + b^2

-- Theorem statement
theorem hypotenuse_length :
  ∀ a b c : ℝ, right_triangle a b c → c = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l1957_195735


namespace NUMINAMATH_CALUDE_trig_expression_equals_four_l1957_195728

theorem trig_expression_equals_four :
  1 / Real.cos (10 * π / 180) - Real.sqrt 3 / Real.sin (10 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_four_l1957_195728


namespace NUMINAMATH_CALUDE_john_buys_three_boxes_l1957_195783

/-- The number of times John plays paintball per month. -/
def plays_per_month : ℕ := 3

/-- The cost of one box of paintballs in dollars. -/
def cost_per_box : ℕ := 25

/-- The total amount John spends on paintballs per month in dollars. -/
def monthly_spending : ℕ := 225

/-- The number of boxes of paintballs John buys each time he plays. -/
def boxes_per_play : ℚ := monthly_spending / (plays_per_month * cost_per_box)

theorem john_buys_three_boxes : boxes_per_play = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_buys_three_boxes_l1957_195783


namespace NUMINAMATH_CALUDE_binomial_coefficients_600_l1957_195755

theorem binomial_coefficients_600 (n : ℕ) (h : n = 600) : 
  Nat.choose n n = 1 ∧ Nat.choose n 0 = 1 ∧ Nat.choose n 1 = n := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficients_600_l1957_195755


namespace NUMINAMATH_CALUDE_coordinates_not_on_C_do_not_satisfy_F_l1957_195784

-- Define the curve C as a set of points in R²
def C : Set (ℝ × ℝ) := sorry

-- Define the function F
def F : ℝ → ℝ → ℝ := sorry

-- Theorem statement
theorem coordinates_not_on_C_do_not_satisfy_F :
  (∀ x y, F x y = 0 → (x, y) ∈ C) →
  ∀ x y, (x, y) ∉ C → F x y ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_coordinates_not_on_C_do_not_satisfy_F_l1957_195784


namespace NUMINAMATH_CALUDE_staircase_perimeter_l1957_195798

/-- Represents a staircase-shaped region -/
structure StaircaseRegion where
  -- Eight sides of length 1
  unit_sides : Fin 8 → ℝ
  unit_sides_length : ∀ i, unit_sides i = 1
  -- Area of the region
  area : ℝ
  area_value : area = 53
  -- Other properties of the staircase shape are implicit

/-- The perimeter of a staircase region -/
def perimeter (s : StaircaseRegion) : ℝ := sorry

/-- Theorem stating that the perimeter of the given staircase region is 32 -/
theorem staircase_perimeter (s : StaircaseRegion) : perimeter s = 32 := by
  sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l1957_195798


namespace NUMINAMATH_CALUDE_conditional_probability_wind_rain_l1957_195733

/-- Given probabilities of events A and B, and their intersection,
    prove that the conditional probability P(B|A) is 3/4 -/
theorem conditional_probability_wind_rain 
  (P_A P_B P_AB : ℝ) 
  (h_A : P_A = 0.4)
  (h_B : P_B = 0.5)
  (h_AB : P_AB = 0.3) :
  P_AB / P_A = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_wind_rain_l1957_195733


namespace NUMINAMATH_CALUDE_binomial_150_150_l1957_195750

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_150_150_l1957_195750


namespace NUMINAMATH_CALUDE_mrs_crocker_chicken_l1957_195738

def chicken_problem (lyndee_pieces : ℕ) (friend_pieces : ℕ) (num_friends : ℕ) : Prop :=
  lyndee_pieces = 1 ∧ friend_pieces = 2 ∧ num_friends = 5 →
  lyndee_pieces + friend_pieces * num_friends = 11

theorem mrs_crocker_chicken : chicken_problem 1 2 5 := by
  sorry

end NUMINAMATH_CALUDE_mrs_crocker_chicken_l1957_195738


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1957_195709

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)) →  -- Definition of sum for arithmetic sequence
  (∀ n, a (n + 1) - a n = a 2 - a 1) →                      -- Definition of arithmetic sequence
  S 3 = 6 →                                                 -- Given condition
  5 * a 1 + a 7 = 12 :=                                     -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1957_195709


namespace NUMINAMATH_CALUDE_max_min_product_of_three_l1957_195744

def S : Finset Int := {-1, -2, 3, 4}

theorem max_min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → 
    x * y * z ≤ 8 ∧ x * y * z ≥ -24) ∧ 
  (∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 8) ∧
  (∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = -24) :=
by sorry

end NUMINAMATH_CALUDE_max_min_product_of_three_l1957_195744


namespace NUMINAMATH_CALUDE_sqrt_meaningful_implies_a_geq_neg_one_l1957_195739

theorem sqrt_meaningful_implies_a_geq_neg_one (a : ℝ) : 
  (∃ (x : ℝ), x^2 = a + 1) → a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_implies_a_geq_neg_one_l1957_195739


namespace NUMINAMATH_CALUDE_shorter_leg_equals_median_length_l1957_195770

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  /-- The length of the median to the hypotenuse -/
  median_length : ℝ
  /-- The length of the shorter leg -/
  shorter_leg : ℝ
  /-- The length of the longer leg -/
  longer_leg : ℝ
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The median length is half the hypotenuse -/
  median_hypotenuse_relation : median_length = hypotenuse / 2
  /-- The sides are in the ratio 1 : √3 : 2 -/
  side_ratios : shorter_leg = hypotenuse / 2 ∧ 
                longer_leg = shorter_leg * Real.sqrt 3 ∧ 
                hypotenuse = shorter_leg * 2

/-- 
Theorem: In a 30-60-90 triangle, if the length of the median to the hypotenuse 
is 15 units, then the length of the shorter leg is also 15 units.
-/
theorem shorter_leg_equals_median_length (t : Triangle30_60_90) 
  (h : t.median_length = 15) : t.shorter_leg = 15 := by
  sorry

end NUMINAMATH_CALUDE_shorter_leg_equals_median_length_l1957_195770


namespace NUMINAMATH_CALUDE_inequality_proof_l1957_195787

theorem inequality_proof (a b c : ℝ) (h : a ≠ b) :
  Real.sqrt ((a - c)^2 + b^2) + Real.sqrt (a^2 + (b - c)^2) > Real.sqrt 2 * abs (a - b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1957_195787


namespace NUMINAMATH_CALUDE_intersection_condition_l1957_195769

/-- Curve C₁ in Cartesian coordinates -/
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

/-- Curve C₂ in Cartesian coordinates -/
def C₂ (x y : ℝ) : Prop := y = x

/-- C₂ translated downward by m units -/
def C₂_translated (x y m : ℝ) : Prop := y = x - m

/-- Two points in common between C₁ and translated C₂ -/
def two_intersections (m : ℝ) : Prop :=
  ∃! (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ 
    C₁ p₁.1 p₁.2 ∧ C₂_translated p₁.1 p₁.2 m ∧
    C₁ p₂.1 p₂.2 ∧ C₂_translated p₂.1 p₂.2 m

/-- Main theorem -/
theorem intersection_condition (m : ℝ) :
  (m > 0 ∧ two_intersections m) ↔ (4 ≤ m ∧ m < 2 + 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l1957_195769


namespace NUMINAMATH_CALUDE_sqrt_six_times_sqrt_two_equals_two_sqrt_three_l1957_195751

theorem sqrt_six_times_sqrt_two_equals_two_sqrt_three :
  Real.sqrt 6 * Real.sqrt 2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_times_sqrt_two_equals_two_sqrt_three_l1957_195751


namespace NUMINAMATH_CALUDE_base_conversion_536_7_to_6_l1957_195704

/-- Converts a number from base b1 to base b2 -/
def convert_base (n : ℕ) (b1 b2 : ℕ) : ℕ :=
  sorry

/-- Checks if a number n in base b has the given digits -/
def check_digits (n : ℕ) (b : ℕ) (digits : List ℕ) : Prop :=
  sorry

theorem base_conversion_536_7_to_6 :
  convert_base 536 7 6 = 1132 ∧ 
  check_digits 536 7 [6, 3, 5] ∧
  check_digits 1132 6 [2, 3, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_base_conversion_536_7_to_6_l1957_195704


namespace NUMINAMATH_CALUDE_nba_schedule_impossibility_l1957_195717

theorem nba_schedule_impossibility :
  ∀ (n k : ℕ) (x y z : ℕ),
    n = 30 →  -- Total number of teams
    k ≤ n →   -- Number of teams in one conference
    x + y + z = (n * 82) / 2 →  -- Total number of games
    82 * k = 2 * x + z →  -- Games played by teams in one conference
    82 * (n - k) = 2 * y + z →  -- Games played by teams in the other conference
    2 * z = x + y + z →  -- Inter-conference games are half of total games
    False :=
by sorry

end NUMINAMATH_CALUDE_nba_schedule_impossibility_l1957_195717


namespace NUMINAMATH_CALUDE_equation_satisfied_at_one_l1957_195796

/-- The function f(x) = 3x - 5 -/
def f (x : ℝ) : ℝ := 3 * x - 5

/-- Theorem stating that the equation 2 * [f(x)] - 16 = f(x - 6) is satisfied when x = 1 -/
theorem equation_satisfied_at_one :
  2 * (f 1) - 16 = f (1 - 6) := by sorry

end NUMINAMATH_CALUDE_equation_satisfied_at_one_l1957_195796


namespace NUMINAMATH_CALUDE_concert_ticket_revenue_l1957_195758

/-- Calculate the total revenue from concert ticket sales --/
theorem concert_ticket_revenue : 
  let original_price : ℚ := 20
  let first_group_size : ℕ := 10
  let second_group_size : ℕ := 20
  let third_group_size : ℕ := 15
  let first_discount : ℚ := 0.4
  let second_discount : ℚ := 0.15
  let third_premium : ℚ := 0.1
  let first_group_revenue := first_group_size * (original_price * (1 - first_discount))
  let second_group_revenue := second_group_size * (original_price * (1 - second_discount))
  let third_group_revenue := third_group_size * (original_price * (1 + third_premium))
  let total_revenue := first_group_revenue + second_group_revenue + third_group_revenue
  total_revenue = 790 := by
  sorry


end NUMINAMATH_CALUDE_concert_ticket_revenue_l1957_195758


namespace NUMINAMATH_CALUDE_trig_identity_l1957_195740

theorem trig_identity (α : Real) (h : Real.tan α = 3) :
  Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α - 3 * Real.cos α ^ 2 = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1957_195740


namespace NUMINAMATH_CALUDE_max_distance_with_tire_swap_l1957_195706

/-- Represents the maximum distance a tire can travel on the rear wheel before wearing out. -/
def rear_tire_limit : ℝ := 15000

/-- Represents the maximum distance a tire can travel on the front wheel before wearing out. -/
def front_tire_limit : ℝ := 25000

/-- Represents the maximum distance a truck can travel before all four tires are worn out,
    given that tires can be swapped between front and rear positions. -/
def max_truck_distance : ℝ := 18750

/-- Theorem stating that the maximum distance a truck can travel before all four tires
    are worn out is 18750 km, given the conditions on tire wear and the ability to swap tires. -/
theorem max_distance_with_tire_swap :
  max_truck_distance = 18750 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_with_tire_swap_l1957_195706


namespace NUMINAMATH_CALUDE_max_planes_with_six_points_l1957_195757

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if a point lies on a plane -/
def pointOnPlane (pt : Point3D) (pl : Plane3D) : Prop :=
  pl.a * pt.x + pl.b * pt.y + pl.c * pt.z + pl.d = 0

/-- Check if four points are collinear -/
def areCollinear (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (a b c d : ℝ), ∀ (x y z : ℝ),
    a*x + b*y + c*z + d = 0 ↔ (x = p1.x ∧ y = p1.y ∧ z = p1.z) ∨
                            (x = p2.x ∧ y = p2.y ∧ z = p2.z) ∨
                            (x = p3.x ∧ y = p3.y ∧ z = p3.z) ∨
                            (x = p4.x ∧ y = p4.y ∧ z = p4.z)

/-- Main theorem -/
theorem max_planes_with_six_points
  (points : Fin 6 → Point3D)
  (h_not_collinear : ∀ (i j k l : Fin 6), i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l →
                     ¬ areCollinear (points i) (points j) (points k) (points l)) :
  ∃ (planes : Fin 6 → Plane3D),
    (∀ (i : Fin 6), ∃ (p1 p2 p3 p4 : Fin 6),
      p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
      pointOnPlane (points p1) (planes i) ∧
      pointOnPlane (points p2) (planes i) ∧
      pointOnPlane (points p3) (planes i) ∧
      pointOnPlane (points p4) (planes i)) ∧
    (∀ (newPlane : Plane3D),
      (∃ (p1 p2 p3 p4 : Fin 6),
        p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
        pointOnPlane (points p1) newPlane ∧
        pointOnPlane (points p2) newPlane ∧
        pointOnPlane (points p3) newPlane ∧
        pointOnPlane (points p4) newPlane) →
      ∃ (i : Fin 6), newPlane = planes i) :=
by
  sorry


end NUMINAMATH_CALUDE_max_planes_with_six_points_l1957_195757


namespace NUMINAMATH_CALUDE_octagon_area_l1957_195782

theorem octagon_area (r : ℝ) (h : r = 3) : 
  let s := 2 * r * Real.sin (π / 8)
  let triangle_area := (1 / 2) * s^2 * Real.sin (π / 4)
  8 * triangle_area = 8 * (1 / 2) * (6 * Real.sin (π / 8))^2 * Real.sin (π / 4) := by
sorry

end NUMINAMATH_CALUDE_octagon_area_l1957_195782


namespace NUMINAMATH_CALUDE_rational_square_roots_existence_l1957_195726

theorem rational_square_roots_existence : ∃ (x : ℚ), 
  3 < x ∧ x < 4 ∧ 
  ∃ (a b : ℚ), a^2 = x - 3 ∧ b^2 = x + 1 ∧
  x = 481 / 144 := by
  sorry

end NUMINAMATH_CALUDE_rational_square_roots_existence_l1957_195726


namespace NUMINAMATH_CALUDE_sum_of_87th_and_95th_odd_integers_l1957_195725

theorem sum_of_87th_and_95th_odd_integers : 
  (2 * 87 - 1) + (2 * 95 - 1) = 362 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_87th_and_95th_odd_integers_l1957_195725


namespace NUMINAMATH_CALUDE_nabla_problem_l1957_195777

-- Define the ∇ operation
def nabla (a b : ℕ) : ℕ := 3 + a^b

-- Theorem to prove
theorem nabla_problem : nabla (nabla 2 1) 4 = 628 := by
  sorry

end NUMINAMATH_CALUDE_nabla_problem_l1957_195777


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l1957_195742

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_expression_evaluation :
  i^3 * (1 - i)^2 = -2 := by sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l1957_195742


namespace NUMINAMATH_CALUDE_train_length_problem_l1957_195702

theorem train_length_problem (speed1 speed2 time length2 : Real) 
  (h1 : speed1 = 120)
  (h2 : speed2 = 80)
  (h3 : time = 9)
  (h4 : length2 = 210.04)
  : ∃ length1 : Real, length1 = 290 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l1957_195702


namespace NUMINAMATH_CALUDE_family_income_and_tax_calculation_l1957_195734

/-- Family income and tax calculation -/
theorem family_income_and_tax_calculation 
  (father_monthly_income mother_monthly_income grandmother_monthly_pension mikhail_monthly_scholarship : ℕ)
  (property_cadastral_value property_area : ℕ)
  (lada_priora_hp lada_priora_months lada_xray_hp lada_xray_months : ℕ)
  (land_cadastral_value land_area : ℕ)
  (tour_cost_per_person : ℕ)
  (h1 : father_monthly_income = 50000)
  (h2 : mother_monthly_income = 28000)
  (h3 : grandmother_monthly_pension = 15000)
  (h4 : mikhail_monthly_scholarship = 3000)
  (h5 : property_cadastral_value = 6240000)
  (h6 : property_area = 78)
  (h7 : lada_priora_hp = 106)
  (h8 : lada_priora_months = 3)
  (h9 : lada_xray_hp = 122)
  (h10 : lada_xray_months = 8)
  (h11 : land_cadastral_value = 420300)
  (h12 : land_area = 10)
  (h13 : tour_cost_per_person = 17900) :
  ∃ (january_income annual_income property_tax transport_tax land_tax remaining_funds : ℕ),
    january_income = 86588 ∧
    annual_income = 137236 ∧
    property_tax = 4640 ∧
    transport_tax = 3775 ∧
    land_tax = 504 ∧
    remaining_funds = 38817 :=
by sorry


end NUMINAMATH_CALUDE_family_income_and_tax_calculation_l1957_195734


namespace NUMINAMATH_CALUDE_consecutive_powers_of_two_divisible_by_six_l1957_195746

theorem consecutive_powers_of_two_divisible_by_six (n : ℕ) :
  6 ∣ (2^n + 2^(n+1)) := by sorry

end NUMINAMATH_CALUDE_consecutive_powers_of_two_divisible_by_six_l1957_195746


namespace NUMINAMATH_CALUDE_system_solution_l1957_195701

theorem system_solution (x y z u : ℚ) (a b c d : ℚ) :
  (x * y) / (x + y) = 1 / a ∧
  (y * z) / (y + z) = 1 / b ∧
  (z * u) / (z + u) = 1 / c ∧
  (x * y * z * u) / (x + y + z + u) = 1 / d →
  ((a = 1 ∧ b = 2 ∧ c = -1 ∧ d = 1 →
    x = -4/3 ∧ y = 4/7 ∧ z = 4 ∧ u = -4/5) ∧
   (a = 1 ∧ b = 3 ∧ c = -2 ∧ d = 1 →
    (x = -1 ∧ y = 1/2 ∧ z = 1 ∧ u = -1/3) ∨
    (x = 1/9 ∧ y = -1/8 ∧ z = 1/11 ∧ u = -1/13))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1957_195701


namespace NUMINAMATH_CALUDE_exists_divisible_by_digit_sum_l1957_195743

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among any 18 consecutive positive integers less than or equal to 2005,
    there exists at least one integer that is divisible by the sum of its digits -/
theorem exists_divisible_by_digit_sum (start : ℕ) (h : start + 17 ≤ 2005) :
  ∃ k : ℕ, k ∈ Finset.range 18 ∧ (start + k) % sum_of_digits (start + k) = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_digit_sum_l1957_195743


namespace NUMINAMATH_CALUDE_probability_site_in_statistics_l1957_195794

def letters_statistics : List Char := ['S', 'T', 'A', 'T', 'I', 'S', 'T', 'I', 'C', 'S']
def letters_site : List Char := ['S', 'I', 'T', 'E']

def count_in_statistics (c : Char) : Nat :=
  (letters_statistics.filter (· = c)).length

def is_in_site (c : Char) : Bool :=
  letters_site.contains c

def favorable_outcomes : Nat :=
  (letters_statistics.filter is_in_site).length

def total_outcomes : Nat :=
  letters_statistics.length

theorem probability_site_in_statistics :
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_site_in_statistics_l1957_195794


namespace NUMINAMATH_CALUDE_distribution_plans_for_given_conditions_l1957_195780

/-- The number of ways to distribute employees between two departments --/
def distribution_plans (total_employees : ℕ) (translators : ℕ) (programmers : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of distribution plans for the given conditions --/
theorem distribution_plans_for_given_conditions :
  distribution_plans 8 2 3 = 36 :=
sorry

end NUMINAMATH_CALUDE_distribution_plans_for_given_conditions_l1957_195780


namespace NUMINAMATH_CALUDE_waynes_blocks_l1957_195788

/-- Wayne's block collection problem -/
theorem waynes_blocks (initial_blocks final_blocks father_blocks : ℕ) 
  (h1 : father_blocks = 6)
  (h2 : final_blocks = 15)
  (h3 : final_blocks = initial_blocks + father_blocks) : 
  initial_blocks = 9 := by
  sorry

end NUMINAMATH_CALUDE_waynes_blocks_l1957_195788


namespace NUMINAMATH_CALUDE_clock_ticks_theorem_l1957_195766

/-- Represents the number of ticks and time between first and last ticks for a clock -/
structure ClockTicks where
  num_ticks : ℕ
  time_between : ℕ

/-- Calculates the number of ticks given the time between first and last ticks -/
def calculate_ticks (reference : ClockTicks) (time : ℕ) : ℕ :=
  let interval := reference.time_between / (reference.num_ticks - 1)
  (time / interval) + 1

theorem clock_ticks_theorem (reference : ClockTicks) (time : ℕ) :
  reference.num_ticks = 8 ∧ reference.time_between = 42 ∧ time = 30 →
  calculate_ticks reference time = 6 :=
by
  sorry

#check clock_ticks_theorem

end NUMINAMATH_CALUDE_clock_ticks_theorem_l1957_195766


namespace NUMINAMATH_CALUDE_abs_inequality_solution_l1957_195774

theorem abs_inequality_solution (x : ℝ) : 
  abs (x + 3) + abs (2 * x - 1) < 7 ↔ -3 ≤ x ∧ x < 5/3 := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_l1957_195774


namespace NUMINAMATH_CALUDE_space_station_cost_share_l1957_195795

/-- Calculates the individual share of a project cost -/
def calculate_share (total_cost : ℕ) (total_population : ℕ) : ℚ :=
  (total_cost : ℚ) / ((total_population : ℚ) / 2)

theorem space_station_cost_share :
  let total_cost : ℕ := 50000000000 -- $50 billion in dollars
  let total_population : ℕ := 400000000 -- 400 million people
  calculate_share total_cost total_population = 250 := by sorry

end NUMINAMATH_CALUDE_space_station_cost_share_l1957_195795


namespace NUMINAMATH_CALUDE_cube_sum_2001_l1957_195776

theorem cube_sum_2001 :
  ∀ a b c : ℕ+,
  a^3 + b^3 + c^3 = 2001 ↔ (a = 10 ∧ b = 10 ∧ c = 1) ∨ (a = 10 ∧ b = 1 ∧ c = 10) ∨ (a = 1 ∧ b = 10 ∧ c = 10) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_2001_l1957_195776


namespace NUMINAMATH_CALUDE_ice_cream_theorem_l1957_195767

theorem ice_cream_theorem (n : ℕ) (h : n > 7) :
  ∃ x y : ℕ, 3 * x + 5 * y = n := by
sorry

end NUMINAMATH_CALUDE_ice_cream_theorem_l1957_195767


namespace NUMINAMATH_CALUDE_conversation_year_1941_l1957_195716

def is_valid_year (y : ℕ) : Prop := 1900 ≤ y ∧ y ≤ 1999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def swap_digits (n : ℕ) : ℕ :=
  ((n % 10) * 10) + (n / 10)

theorem conversation_year_1941 :
  ∃! (conv_year : ℕ) (elder_birth : ℕ) (younger_birth : ℕ),
    is_valid_year conv_year ∧
    is_valid_year elder_birth ∧
    is_valid_year younger_birth ∧
    elder_birth < younger_birth ∧
    conv_year - elder_birth = digit_sum younger_birth ∧
    conv_year - younger_birth = digit_sum elder_birth ∧
    swap_digits (conv_year - elder_birth) = conv_year - younger_birth ∧
    conv_year = 1941 :=
  sorry

end NUMINAMATH_CALUDE_conversation_year_1941_l1957_195716


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1957_195764

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 7 * a 12 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1957_195764


namespace NUMINAMATH_CALUDE_wedding_decoration_cost_l1957_195714

/-- Calculates the total cost of decorations for a wedding reception --/
def total_decoration_cost (num_tables : ℕ) 
                          (tablecloth_cost : ℕ) 
                          (place_setting_cost : ℕ) 
                          (place_settings_per_table : ℕ) 
                          (roses_per_centerpiece : ℕ) 
                          (rose_cost : ℕ) 
                          (lilies_per_centerpiece : ℕ) 
                          (lily_cost : ℕ) : ℕ :=
  num_tables * (tablecloth_cost + 
                place_settings_per_table * place_setting_cost + 
                roses_per_centerpiece * rose_cost + 
                lilies_per_centerpiece * lily_cost)

/-- Theorem stating that the total decoration cost for the given conditions is $3500 --/
theorem wedding_decoration_cost : 
  total_decoration_cost 20 25 10 4 10 5 15 4 = 3500 := by
  sorry

end NUMINAMATH_CALUDE_wedding_decoration_cost_l1957_195714


namespace NUMINAMATH_CALUDE_inequality_proof_l1957_195748

theorem inequality_proof (a b c : ℝ) (h1 : a > -b) (h2 : -b > 0) (h3 : c < 0) :
  a * (1 - c) > b * (c - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1957_195748


namespace NUMINAMATH_CALUDE_fraction_sum_zero_l1957_195789

theorem fraction_sum_zero (a b : ℚ) (h : b + 1 ≠ 0) : 
  a / (b + 1) + 2 * a / (b + 1) - 3 * a / (b + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_zero_l1957_195789


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1957_195708

/-- Given a regular polygon inscribed in a circle, if the central angle corresponding to one side is 72°, then the polygon has 5 sides. -/
theorem regular_polygon_sides (n : ℕ) (central_angle : ℝ) : 
  n ≥ 3 → 
  central_angle = 72 → 
  (360 : ℝ) / n = central_angle → 
  n = 5 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1957_195708


namespace NUMINAMATH_CALUDE_middle_number_proof_l1957_195752

theorem middle_number_proof (a b c : ℤ) : 
  a < b ∧ b < c ∧ 
  a + b = 18 ∧ 
  a + c = 23 ∧ 
  b + c = 27 → 
  b = 11 := by
sorry

end NUMINAMATH_CALUDE_middle_number_proof_l1957_195752


namespace NUMINAMATH_CALUDE_combination_sum_equals_4950_l1957_195793

theorem combination_sum_equals_4950 : Nat.choose 99 98 + Nat.choose 99 97 = 4950 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_equals_4950_l1957_195793


namespace NUMINAMATH_CALUDE_extrema_of_squared_sum_l1957_195724

theorem extrema_of_squared_sum (a b c : ℝ) 
  (h : |a + b| + |b + c| + |c + a| = 8) :
  (∃ (x y z : ℝ), x^2 + y^2 + z^2 = 16/3 ∧ 
    |x + y| + |y + z| + |z + x| = 8 ∧
    ∀ (p q r : ℝ), |p + q| + |q + r| + |r + p| = 8 → 
      p^2 + q^2 + r^2 ≥ 16/3) ∧
  (∃ (x y z : ℝ), x^2 + y^2 + z^2 = 32 ∧ 
    |x + y| + |y + z| + |z + x| = 8 ∧
    ∀ (p q r : ℝ), |p + q| + |q + r| + |r + p| = 8 → 
      p^2 + q^2 + r^2 ≤ 32) :=
by sorry

end NUMINAMATH_CALUDE_extrema_of_squared_sum_l1957_195724


namespace NUMINAMATH_CALUDE_no_valid_sequence_exists_l1957_195736

theorem no_valid_sequence_exists : ¬ ∃ (seq : Fin 100 → ℤ),
  (∀ i, Odd (seq i)) ∧ 
  (∀ i, i + 4 < 100 → ∃ k, (seq i + seq (i+1) + seq (i+2) + seq (i+3) + seq (i+4)) = k^2) ∧
  (∀ i, i + 8 < 100 → ∃ k, (seq i + seq (i+1) + seq (i+2) + seq (i+3) + seq (i+4) + 
                             seq (i+5) + seq (i+6) + seq (i+7) + seq (i+8)) = k^2) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_sequence_exists_l1957_195736


namespace NUMINAMATH_CALUDE_case_A_case_B_case_C_case_D_l1957_195765

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the number of solutions for a triangle
inductive TriangleSolutions
  | Unique
  | Two
  | None

-- Function to determine the number of solutions for a triangle
def triangleSolutions (t : Triangle) : TriangleSolutions := sorry

-- Theorem for case A
theorem case_A :
  let t : Triangle := { a := 5, b := 7, c := 8, A := 0, B := 0, C := 0 }
  triangleSolutions t = TriangleSolutions.Unique := by sorry

-- Theorem for case B
theorem case_B :
  let t : Triangle := { a := 0, b := 18, c := 20, A := 0, B := 60 * π / 180, C := 0 }
  triangleSolutions t = TriangleSolutions.None := by sorry

-- Theorem for case C
theorem case_C :
  let t : Triangle := { a := 8, b := 8 * Real.sqrt 2, c := 0, A := 0, B := 45 * π / 180, C := 0 }
  triangleSolutions t = TriangleSolutions.Two := by sorry

-- Theorem for case D
theorem case_D :
  let t : Triangle := { a := 30, b := 25, c := 0, A := 150 * π / 180, B := 0, C := 0 }
  triangleSolutions t = TriangleSolutions.Unique := by sorry

end NUMINAMATH_CALUDE_case_A_case_B_case_C_case_D_l1957_195765


namespace NUMINAMATH_CALUDE_min_value_expression_l1957_195705

theorem min_value_expression (x y : ℝ) : 5*x^2 + 4*y^2 - 8*x*y + 2*x + 4 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1957_195705
