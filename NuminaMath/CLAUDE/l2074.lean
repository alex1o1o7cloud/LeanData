import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2074_207436

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | x ≤ -3 ∨ x ≥ 4}

-- Theorem statement
theorem quadratic_inequality_theorem (a b c : ℝ) 
  (h : ∀ x, f a b c x ≥ 0 ↔ x ∈ solution_set a b c) : 
  (a > 0) ∧ 
  (∀ x, f c (-b) a x < 0 ↔ x < -1/4 ∨ x > 1/3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2074_207436


namespace NUMINAMATH_CALUDE_min_distance_between_points_l2074_207454

/-- The minimum distance between points A(x, √2-x) and B(√2/2, 0) is 1/2 -/
theorem min_distance_between_points :
  let A : ℝ → ℝ × ℝ := λ x ↦ (x, Real.sqrt 2 - x)
  let B : ℝ × ℝ := (Real.sqrt 2 / 2, 0)
  ∃ (min_dist : ℝ), min_dist = 1/2 ∧
    ∀ x, Real.sqrt ((A x).1 - B.1)^2 + ((A x).2 - B.2)^2 ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_points_l2074_207454


namespace NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_one_l2074_207419

/-- A system of equations parameterized by n -/
def system (n : ℝ) (x y z : ℝ) : Prop :=
  n * x + y = 2 ∧ n * y + z = 2 ∧ x + n^2 * z = 2

/-- The system has no solution if and only if n = -1 -/
theorem no_solution_iff_n_eq_neg_one :
  ∀ n : ℝ, (∀ x y z : ℝ, ¬system n x y z) ↔ n = -1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_one_l2074_207419


namespace NUMINAMATH_CALUDE_root_reciprocal_sum_l2074_207474

theorem root_reciprocal_sum (a b c : ℂ) : 
  (a^3 - 2*a^2 - a + 2 = 0) → 
  (b^3 - 2*b^2 - b + 2 = 0) → 
  (c^3 - 2*c^2 - c + 2 = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  1/(a-1) + 1/(b-1) + 1/(c-1) = -1 := by
sorry

end NUMINAMATH_CALUDE_root_reciprocal_sum_l2074_207474


namespace NUMINAMATH_CALUDE_luke_coin_count_l2074_207467

theorem luke_coin_count : 
  ∀ (quarter_piles dime_piles coins_per_pile : ℕ),
    quarter_piles = 5 →
    dime_piles = 5 →
    coins_per_pile = 3 →
    quarter_piles * coins_per_pile + dime_piles * coins_per_pile = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_luke_coin_count_l2074_207467


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2074_207471

theorem solve_linear_equation :
  ∃ x : ℝ, x + 1 = 3 ∧ x = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2074_207471


namespace NUMINAMATH_CALUDE_trig_identity_proof_l2074_207414

theorem trig_identity_proof (a : ℝ) : 
  Real.cos (a + π/6) * Real.sin (a - π/3) + Real.sin (a + π/6) * Real.cos (a - π/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l2074_207414


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2074_207479

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 12) = 10 → x = 88 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2074_207479


namespace NUMINAMATH_CALUDE_a_equals_two_l2074_207412

/-- The function f(x) = x^2 - 14x + 52 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 52

/-- The function g(x) = ax + b, where a and b are positive real numbers -/
def g (a b : ℝ) (x : ℝ) : ℝ := a*x + b

/-- Theorem stating that a = 2 given the conditions -/
theorem a_equals_two (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : f (g a b (-5)) = 3) (h2 : f (g a b 0) = 103) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_two_l2074_207412


namespace NUMINAMATH_CALUDE_circle_x_axis_intersection_sum_l2074_207431

/-- The sum of x-coordinates of intersection points between a circle and the x-axis -/
def sum_x_coordinates (h k r : ℝ) : ℝ :=
  2 * h

/-- Theorem: For a circle with center (3, -5) and radius 7, 
    the sum of x-coordinates of its intersection points with the x-axis is 6 -/
theorem circle_x_axis_intersection_sum :
  sum_x_coordinates 3 (-5) 7 = 6 := by
  sorry


end NUMINAMATH_CALUDE_circle_x_axis_intersection_sum_l2074_207431


namespace NUMINAMATH_CALUDE_number_of_boys_at_park_l2074_207413

theorem number_of_boys_at_park : 
  ∀ (girls parents groups group_size total_people boys : ℕ),
    girls = 14 →
    parents = 50 →
    groups = 3 →
    group_size = 25 →
    total_people = groups * group_size →
    boys = total_people - (girls + parents) →
    boys = 11 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_at_park_l2074_207413


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l2074_207409

theorem smaller_number_in_ratio (x y : ℝ) : 
  x > 0 → y > 0 → x / y = 2 / 5 → x + y = 21 → min x y = 6 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l2074_207409


namespace NUMINAMATH_CALUDE_bruce_triple_age_l2074_207451

/-- Bruce's current age -/
def bruce_age : ℕ := 36

/-- Bruce's son's current age -/
def son_age : ℕ := 8

/-- The number of years it will take for Bruce to be three times as old as his son -/
def years_until_triple : ℕ := 6

/-- Theorem stating that in 6 years, Bruce will be three times as old as his son -/
theorem bruce_triple_age :
  bruce_age + years_until_triple = 3 * (son_age + years_until_triple) :=
sorry

end NUMINAMATH_CALUDE_bruce_triple_age_l2074_207451


namespace NUMINAMATH_CALUDE_message_spread_time_l2074_207418

theorem message_spread_time (n : ℕ) : ∃ (m : ℕ), m ≥ 5 ∧ 2^(m+1) - 2 > 55 ∧ ∀ (k : ℕ), k < m → 2^(k+1) - 2 ≤ 55 := by
  sorry

end NUMINAMATH_CALUDE_message_spread_time_l2074_207418


namespace NUMINAMATH_CALUDE_train_speed_problem_l2074_207428

-- Define the speeds and times
def speed_A : ℝ := 90
def time_A : ℝ := 9
def time_B : ℝ := 4

-- Theorem statement
theorem train_speed_problem :
  ∃ (speed_B : ℝ),
    speed_B > 0 ∧
    speed_A * time_A = speed_B * time_B ∧
    speed_B = 202.5 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2074_207428


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2074_207408

/-- Given a geometric sequence with positive terms where a₁, ½a₃, 2a₂ form an arithmetic sequence,
    the ratio (a₁₃ + a₁₄) / (a₁₄ + a₁₅) equals √2 - 1. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
    (h_geom : ∃ q : ℝ, ∀ n, a (n + 1) = q * a n) 
    (h_arith : ∃ d : ℝ, a 1 + d = (1/2) * a 3 ∧ (1/2) * a 3 + d = 2 * a 2) :
  (a 13 + a 14) / (a 14 + a 15) = Real.sqrt 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2074_207408


namespace NUMINAMATH_CALUDE_variance_linear_transform_l2074_207426

-- Define the variance of a dataset
def variance (data : List ℝ) : ℝ := sorry

-- Define a linear transformation of a dataset
def linearTransform (a b : ℝ) (data : List ℝ) : List ℝ := 
  data.map (fun x => a * x + b)

theorem variance_linear_transform (data : List ℝ) :
  variance data = 2 → variance (linearTransform 3 (-2) data) = 18 := by
  sorry

end NUMINAMATH_CALUDE_variance_linear_transform_l2074_207426


namespace NUMINAMATH_CALUDE_six_digit_number_theorem_l2074_207423

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  ∃ (m k : ℕ),
    m < 10000 ∧ k < 100 ∧
    n = 100000 * (n / 100000) + 1000 * (n / 1000 % 100) + (n % 1000) ∧
    4 * n = k * 10000 + m ∧
    n = m * 100 + k

theorem six_digit_number_theorem :
  {n : ℕ | is_valid_number n} = {142857, 190476, 238095} :=
sorry

end NUMINAMATH_CALUDE_six_digit_number_theorem_l2074_207423


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2074_207455

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 986 ∧ 
  n % 17 = 0 ∧ 
  100 ≤ n ∧ n ≤ 999 ∧ 
  ∀ m : ℕ, m % 17 = 0 → 100 ≤ m → m ≤ 999 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2074_207455


namespace NUMINAMATH_CALUDE_rod_cutting_l2074_207402

theorem rod_cutting (rod_length : Real) (piece_length : Real) :
  rod_length = 42.5 →
  piece_length = 0.85 →
  Int.floor (rod_length / piece_length) = 50 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l2074_207402


namespace NUMINAMATH_CALUDE_most_colored_pencils_l2074_207498

theorem most_colored_pencils (total : ℕ) (red blue yellow : ℕ) : 
  total = 24 →
  red = total / 4 →
  blue = red + 6 →
  yellow = total - red - blue →
  blue > red ∧ blue > yellow :=
by sorry

end NUMINAMATH_CALUDE_most_colored_pencils_l2074_207498


namespace NUMINAMATH_CALUDE_product_expansion_l2074_207416

theorem product_expansion (x : ℝ) : (x + 3) * (x + 7) * (x - 2) = x^3 + 8*x^2 + x - 42 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2074_207416


namespace NUMINAMATH_CALUDE_reciprocal_equals_self_l2074_207466

theorem reciprocal_equals_self (x : ℝ) : x ≠ 0 → (x = 1/x ↔ x = 1 ∨ x = -1) := by sorry

end NUMINAMATH_CALUDE_reciprocal_equals_self_l2074_207466


namespace NUMINAMATH_CALUDE_average_weight_problem_l2074_207406

/-- The average weight problem -/
theorem average_weight_problem 
  (weight_A weight_B weight_C weight_D : ℝ)
  (h1 : (weight_A + weight_B + weight_C) / 3 = 60)
  (h2 : weight_A = 87)
  (h3 : (weight_B + weight_C + weight_D + (weight_D + 3)) / 4 = 64) :
  (weight_A + weight_B + weight_C + weight_D) / 4 = 65 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l2074_207406


namespace NUMINAMATH_CALUDE_completely_overlapping_implies_congruent_l2074_207450

/-- Two triangles are completely overlapping if all their corresponding vertices coincide. -/
def CompletelyOverlapping (T1 T2 : Set (ℝ × ℝ)) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ T1 ↔ p ∈ T2

/-- Two triangles are congruent if they have the same size and shape. -/
def Congruent (T1 T2 : Set (ℝ × ℝ)) : Prop :=
  ∃ f : ℝ × ℝ → ℝ × ℝ, Isometry f ∧ f '' T1 = T2

/-- If two triangles completely overlap, then they are congruent. -/
theorem completely_overlapping_implies_congruent
  (T1 T2 : Set (ℝ × ℝ)) (h : CompletelyOverlapping T1 T2) :
  Congruent T1 T2 := by
  sorry

end NUMINAMATH_CALUDE_completely_overlapping_implies_congruent_l2074_207450


namespace NUMINAMATH_CALUDE_football_yards_gained_l2074_207468

theorem football_yards_gained (initial_loss : ℤ) (final_progress : ℤ) (yards_gained : ℤ) : 
  initial_loss = -5 → final_progress = 2 → yards_gained = initial_loss + final_progress →
  yards_gained = 7 := by
sorry

end NUMINAMATH_CALUDE_football_yards_gained_l2074_207468


namespace NUMINAMATH_CALUDE_sally_bought_twenty_cards_l2074_207448

/-- The number of cards Sally bought -/
def cards_bought (initial : ℕ) (received : ℕ) (total : ℕ) : ℕ :=
  total - (initial + received)

/-- Theorem: Sally bought 20 cards -/
theorem sally_bought_twenty_cards :
  cards_bought 27 41 88 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sally_bought_twenty_cards_l2074_207448


namespace NUMINAMATH_CALUDE_product_of_specific_roots_l2074_207490

/-- Given distinct real numbers a, b, c, d satisfying specific equations, their product is 11 -/
theorem product_of_specific_roots (a b c d : ℝ) 
  (ha : a = Real.sqrt (4 + Real.sqrt (5 + a)))
  (hb : b = Real.sqrt (4 - Real.sqrt (5 + b)))
  (hc : c = Real.sqrt (4 + Real.sqrt (5 - c)))
  (hd : d = Real.sqrt (4 - Real.sqrt (5 - d)))
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  a * b * c * d = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_roots_l2074_207490


namespace NUMINAMATH_CALUDE_cheese_wedge_volume_l2074_207493

/-- The volume of a wedge of cheese that represents one-third of a cylindrical log --/
theorem cheese_wedge_volume (d h r : ℝ) : 
  d = 12 →  -- diameter is 12 cm
  h = d →   -- height is equal to diameter
  r = d / 2 →  -- radius is half the diameter
  (1 / 3) * (π * r^2 * h) = 144 * π := by
  sorry

end NUMINAMATH_CALUDE_cheese_wedge_volume_l2074_207493


namespace NUMINAMATH_CALUDE_floor_division_equality_l2074_207415

/-- Floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Theorem: For any positive real number a and any integer n,
    the floor of (floor of a) divided by n is equal to the floor of a divided by n -/
theorem floor_division_equality (a : ℝ) (n : ℤ) (h1 : 0 < a) (h2 : n ≠ 0) :
  floor ((floor a : ℝ) / n) = floor (a / n) := by
  sorry

end NUMINAMATH_CALUDE_floor_division_equality_l2074_207415


namespace NUMINAMATH_CALUDE_school_year_days_is_180_l2074_207487

/-- The number of days in a school year. -/
def school_year_days : ℕ := 180

/-- The maximum percentage of days that can be missed without taking exams. -/
def max_missed_percentage : ℚ := 5 / 100

/-- The number of days Hazel has already missed. -/
def days_already_missed : ℕ := 6

/-- The additional number of days Hazel can miss without taking exams. -/
def additional_days_can_miss : ℕ := 3

/-- Theorem stating that the number of days in the school year is 180. -/
theorem school_year_days_is_180 :
  (days_already_missed + additional_days_can_miss : ℚ) / school_year_days = max_missed_percentage :=
by sorry

end NUMINAMATH_CALUDE_school_year_days_is_180_l2074_207487


namespace NUMINAMATH_CALUDE_B_power_100_l2074_207457

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_100 : B ^ 100 = B := by sorry

end NUMINAMATH_CALUDE_B_power_100_l2074_207457


namespace NUMINAMATH_CALUDE_min_value_expression_l2074_207481

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (min : ℝ), min = 3 ∧ (∀ x y : ℝ, x > 0 → y > 0 → 
    (Real.sqrt ((x^2 + y^2) * (4*x^2 + y^2))) / (x*y) ≥ min) ∧
    ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
      (Real.sqrt ((x^2 + y^2) * (4*x^2 + y^2))) / (x*y) = min :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2074_207481


namespace NUMINAMATH_CALUDE_system_solution_independent_of_c_l2074_207442

theorem system_solution_independent_of_c :
  ∀ (c : ℝ),
    2 - 0 + 2*(-1) = 0 ∧
    -2*2 + 0 - 2*(-1) = -2 ∧
    2*2 + c*0 + 3*(-1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_independent_of_c_l2074_207442


namespace NUMINAMATH_CALUDE_sequence_problem_l2074_207485

/-- Given a sequence where each term is obtained by doubling the previous term and adding 4,
    if the third term is 52, then the first term is 10. -/
theorem sequence_problem (x : ℝ) : 
  let second_term := 2 * x + 4
  let third_term := 2 * second_term + 4
  third_term = 52 → x = 10 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l2074_207485


namespace NUMINAMATH_CALUDE_train_length_l2074_207424

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 15 → speed * time * (5 / 18) = 375 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2074_207424


namespace NUMINAMATH_CALUDE_number_plus_expression_l2074_207407

theorem number_plus_expression (x : ℝ) : x + 2 * (8 - 3) = 15 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_expression_l2074_207407


namespace NUMINAMATH_CALUDE_store_employees_l2074_207434

/-- The number of employees in Sergio's store -/
def num_employees : ℕ := 20

/-- The initial average number of items sold per employee -/
def initial_average : ℚ := 75

/-- The new average number of items sold per employee -/
def new_average : ℚ := 783/10

/-- The number of items sold by the top three performers on the next day -/
def top_three_sales : ℕ := 6 + 5 + 4

theorem store_employees :
  (initial_average * num_employees + top_three_sales + 3 * (num_employees - 3)) / num_employees = new_average :=
sorry

end NUMINAMATH_CALUDE_store_employees_l2074_207434


namespace NUMINAMATH_CALUDE_imaginary_power_difference_l2074_207462

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_difference : i^23 - i^210 = -i + 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_power_difference_l2074_207462


namespace NUMINAMATH_CALUDE_percentage_increase_l2074_207472

theorem percentage_increase (x : ℝ) (h1 : x = 90.4) (h2 : ∃ p, x = 80 * (1 + p / 100)) : 
  ∃ p, x = 80 * (1 + p / 100) ∧ p = 13 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l2074_207472


namespace NUMINAMATH_CALUDE_concession_stand_sales_l2074_207495

/-- Calculates the total number of items sold given the prices, total revenue, and number of hot dogs sold. -/
theorem concession_stand_sales
  (hot_dog_price : ℚ)
  (soda_price : ℚ)
  (total_revenue : ℚ)
  (hot_dogs_sold : ℕ)
  (h1 : hot_dog_price = 3/2)
  (h2 : soda_price = 1/2)
  (h3 : total_revenue = 157/2)
  (h4 : hot_dogs_sold = 35) :
  ∃ (sodas_sold : ℕ), hot_dogs_sold + sodas_sold = 87 :=
by sorry

end NUMINAMATH_CALUDE_concession_stand_sales_l2074_207495


namespace NUMINAMATH_CALUDE_combined_average_age_l2074_207453

theorem combined_average_age (people_a : ℕ) (people_b : ℕ) (avg_age_a : ℝ) (avg_age_b : ℝ)
  (h1 : people_a = 8)
  (h2 : people_b = 2)
  (h3 : avg_age_a = 38)
  (h4 : avg_age_b = 30) :
  (people_a * avg_age_a + people_b * avg_age_b) / (people_a + people_b) = 36.4 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_age_l2074_207453


namespace NUMINAMATH_CALUDE_kyler_wins_one_l2074_207445

structure ChessTournament where
  peter_wins : ℕ
  peter_losses : ℕ
  emma_wins : ℕ
  emma_losses : ℕ
  kyler_losses : ℕ

def kyler_wins (t : ChessTournament) : ℕ :=
  (t.peter_wins + t.emma_wins + t.kyler_losses) - (t.peter_losses + t.emma_losses)

theorem kyler_wins_one (t : ChessTournament) 
  (h1 : t.peter_wins = 4) 
  (h2 : t.peter_losses = 2) 
  (h3 : t.emma_wins = 3) 
  (h4 : t.emma_losses = 3) 
  (h5 : t.kyler_losses = 3) : 
  kyler_wins t = 1 := by
  sorry

end NUMINAMATH_CALUDE_kyler_wins_one_l2074_207445


namespace NUMINAMATH_CALUDE_circle_radius_problem_l2074_207420

-- Define the points
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (5, -2)

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem statement
theorem circle_radius_problem (M N : Circle) : 
  -- Conditions
  (M.center.1 = 0) →  -- Center of M is on y-axis
  (N.center.1 = 2) →  -- x-coordinate of N's center is 2
  (N.center.2 = 4 - M.center.2) →  -- y-coordinate of N's center
  (M.radius = N.radius) →  -- Equal radii
  (M.radius^2 = (B.1 - M.center.1)^2 + (B.2 - M.center.2)^2) →  -- M passes through B
  (N.radius^2 = (B.1 - N.center.1)^2 + (B.2 - N.center.2)^2) →  -- N passes through B
  (N.radius^2 = (C.1 - N.center.1)^2 + (C.2 - N.center.2)^2) →  -- N passes through C
  -- Conclusion
  M.radius = Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l2074_207420


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2074_207427

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 3*x

-- Define the point of tangency
def P : ℝ × ℝ := (1, 4)

-- State the theorem
theorem tangent_line_equation :
  let m := (2 * P.1 + 3) -- Slope of the tangent line
  (5 : ℝ) * x - y - 1 = 0 ↔ y - P.2 = m * (x - P.1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2074_207427


namespace NUMINAMATH_CALUDE_solve_equation_l2074_207494

theorem solve_equation (x : ℝ) : 3 * x + 15 = (1 / 3) * (6 * x + 45) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2074_207494


namespace NUMINAMATH_CALUDE_inequality_proof_l2074_207429

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 ∧
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2074_207429


namespace NUMINAMATH_CALUDE_greatest_number_of_fruit_baskets_l2074_207449

theorem greatest_number_of_fruit_baskets : Nat.gcd (Nat.gcd 18 27) 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_of_fruit_baskets_l2074_207449


namespace NUMINAMATH_CALUDE_blue_balls_count_l2074_207469

/-- Given a jar with white and blue balls in a 5:3 ratio, 
    prove that 15 white balls implies 9 blue balls -/
theorem blue_balls_count (white_balls blue_balls : ℕ) : 
  (white_balls : ℚ) / blue_balls = 5 / 3 → 
  white_balls = 15 → 
  blue_balls = 9 := by
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l2074_207469


namespace NUMINAMATH_CALUDE_greaterElementSumOfS_l2074_207440

def S : Finset ℕ := {8, 5, 1, 13, 34, 3, 21, 2}

def greaterElementSum (s : Finset ℕ) : ℕ :=
  s.sum (λ x => (s.filter (λ y => y < x)).card * x)

theorem greaterElementSumOfS : greaterElementSum S = 484 := by
  sorry

end NUMINAMATH_CALUDE_greaterElementSumOfS_l2074_207440


namespace NUMINAMATH_CALUDE_quadratic_function_b_value_l2074_207492

/-- Given a quadratic function f(x) = ax² + bx + c, if f(2) - f(-2) = 8, then b = 2 -/
theorem quadratic_function_b_value (a b c : ℝ) (y₁ y₂ : ℝ) : 
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = 8 →
  b = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_b_value_l2074_207492


namespace NUMINAMATH_CALUDE_xyz_value_l2074_207483

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9)
  (h3 : x + y + z = 3) : 
  x * y * z = 5 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l2074_207483


namespace NUMINAMATH_CALUDE_sandy_shorts_cost_l2074_207435

/-- Represents the amount Sandy spent on shorts -/
def S : ℝ := sorry

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℝ := 12.14

/-- The amount Sandy received for returning a jacket -/
def jacket_return : ℝ := 7.43

/-- The net amount Sandy spent on clothes -/
def net_spent : ℝ := 18.7

/-- Theorem stating that Sandy spent $13.99 on shorts -/
theorem sandy_shorts_cost : S = 13.99 :=
  by
    have h : S + shirt_cost - jacket_return = net_spent := by sorry
    sorry


end NUMINAMATH_CALUDE_sandy_shorts_cost_l2074_207435


namespace NUMINAMATH_CALUDE_triangle_side_sum_l2074_207430

theorem triangle_side_sum (a b c : ℝ) (h_angles : a = 60 ∧ b = 30 ∧ c = 90) 
  (h_side : 8 * Real.sqrt 3 = b) : 
  a + b + c = 24 * Real.sqrt 3 + 24 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l2074_207430


namespace NUMINAMATH_CALUDE_vending_machine_probability_l2074_207465

def num_toys : ℕ := 10
def min_cost : ℚ := 1/2
def max_cost : ℚ := 5
def cost_difference : ℚ := 1/2
def initial_half_dollars : ℕ := 10
def favorite_toy_cost : ℚ := 9/2

theorem vending_machine_probability :
  let total_permutations : ℕ := num_toys.factorial
  let favorable_outcomes : ℕ := (num_toys - 1).factorial + (num_toys - 2).factorial
  (1 : ℚ) - (favorable_outcomes : ℚ) / (total_permutations : ℚ) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_vending_machine_probability_l2074_207465


namespace NUMINAMATH_CALUDE_cone_volume_gravel_pile_l2074_207433

/-- The volume of a cone with base diameter 10 feet and height 80% of its diameter is 200π/3 cubic feet. -/
theorem cone_volume_gravel_pile :
  let base_diameter : ℝ := 10
  let height_ratio : ℝ := 0.8
  let height : ℝ := height_ratio * base_diameter
  let radius : ℝ := base_diameter / 2
  let volume : ℝ := (1 / 3) * π * radius^2 * height
  volume = 200 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_gravel_pile_l2074_207433


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2074_207403

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x < 8}

-- Define set B
def B : Set ℝ := {x | x^2 - 5*x - 14 ≥ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 7 ≤ x ∧ x < 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2074_207403


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2074_207477

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 3) = 7 → x = 46 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2074_207477


namespace NUMINAMATH_CALUDE_sqrt_64_equals_8_l2074_207456

theorem sqrt_64_equals_8 : Real.sqrt 64 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_64_equals_8_l2074_207456


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2074_207473

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) ↔
  (∀ x : ℝ, x ≥ 1 ∨ x ≤ -1 → x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2074_207473


namespace NUMINAMATH_CALUDE_probability_of_black_piece_l2074_207476

/-- Given a set of items with two types, this function calculates the probability of selecting an item of a specific type. -/
def probability_of_selection (total : ℕ) (type_a : ℕ) : ℚ :=
  type_a / total

/-- The probability of selecting a black piece from a set of Go pieces -/
theorem probability_of_black_piece : probability_of_selection 7 4 = 4 / 7 := by
  sorry

#eval probability_of_selection 7 4

end NUMINAMATH_CALUDE_probability_of_black_piece_l2074_207476


namespace NUMINAMATH_CALUDE_rectangle_count_l2074_207404

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of horizontal lines --/
def horizontal_lines : ℕ := 5

/-- The number of vertical lines --/
def vertical_lines : ℕ := 4

/-- The number of lines needed to form a rectangle --/
def lines_for_rectangle : ℕ := 4

/-- The number of horizontal lines needed for a rectangle --/
def horizontal_needed : ℕ := 2

/-- The number of vertical lines needed for a rectangle --/
def vertical_needed : ℕ := 2

theorem rectangle_count : 
  (choose horizontal_lines horizontal_needed) * (choose vertical_lines vertical_needed) = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_count_l2074_207404


namespace NUMINAMATH_CALUDE_estimate_fish_population_l2074_207417

/-- Estimates the number of fish in a lake using the mark-recapture method. -/
theorem estimate_fish_population (initial_marked : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) :
  initial_marked = 100 →
  second_catch = 200 →
  marked_in_second = 25 →
  (initial_marked * second_catch) / marked_in_second = 800 :=
by
  sorry

end NUMINAMATH_CALUDE_estimate_fish_population_l2074_207417


namespace NUMINAMATH_CALUDE_mean_median_difference_l2074_207432

def is_valid_set (x d : ℕ) : Prop :=
  x > 0 ∧ x + 2 > 0 ∧ x + 4 > 0 ∧ x + 7 > 0 ∧ x + d > 0

def median (x : ℕ) : ℕ := x + 4

def mean (x d : ℕ) : ℚ :=
  (x + (x + 2) + (x + 4) + (x + 7) + (x + d)) / 5

theorem mean_median_difference (x d : ℕ) :
  is_valid_set x d →
  mean x d = (median x : ℚ) + 5 →
  d = 32 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l2074_207432


namespace NUMINAMATH_CALUDE_coloring_books_problem_l2074_207441

theorem coloring_books_problem (total_colored : ℕ) (total_left : ℕ) (num_books : ℕ) :
  total_colored = 20 →
  total_left = 68 →
  num_books = 2 →
  (total_colored + total_left) % num_books = 0 →
  (total_colored + total_left) / num_books = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_coloring_books_problem_l2074_207441


namespace NUMINAMATH_CALUDE_cream_fraction_is_four_ninths_l2074_207411

/-- Represents the contents of a cup --/
structure CupContents where
  coffee : ℚ
  cream : ℚ

/-- Represents the state of both cups --/
structure CupState where
  cup1 : CupContents
  cup2 : CupContents

def initial_state : CupState :=
  { cup1 := { coffee := 5, cream := 0 },
    cup2 := { coffee := 0, cream := 5 } }

def pour_half_coffee (state : CupState) : CupState :=
  { cup1 := { coffee := state.cup1.coffee / 2, cream := state.cup1.cream },
    cup2 := { coffee := state.cup2.coffee + state.cup1.coffee / 2, cream := state.cup2.cream } }

def add_cream (state : CupState) : CupState :=
  { cup1 := state.cup1,
    cup2 := { coffee := state.cup2.coffee, cream := state.cup2.cream + 1 } }

def pour_half_back (state : CupState) : CupState :=
  let total_cup2 := state.cup2.coffee + state.cup2.cream
  let half_cup2 := total_cup2 / 2
  let coffee_ratio := state.cup2.coffee / total_cup2
  let cream_ratio := state.cup2.cream / total_cup2
  { cup1 := { coffee := state.cup1.coffee + half_cup2 * coffee_ratio,
              cream := state.cup1.cream + half_cup2 * cream_ratio },
    cup2 := { coffee := state.cup2.coffee - half_cup2 * coffee_ratio,
              cream := state.cup2.cream - half_cup2 * cream_ratio } }

theorem cream_fraction_is_four_ninths :
  let final_state := pour_half_back (add_cream (pour_half_coffee initial_state))
  let total_cup1 := final_state.cup1.coffee + final_state.cup1.cream
  final_state.cup1.cream / total_cup1 = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_cream_fraction_is_four_ninths_l2074_207411


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2074_207461

/-- The eccentricity of a hyperbola with the given conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (A B F : ℝ × ℝ) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  -- F is the right focus of C
  (F.1 = c ∧ F.2 = 0) →
  -- A and B are on the asymptotes of C
  (A.2 = (b / a) * A.1 ∧ B.2 = -(b / a) * B.1) →
  -- AF is perpendicular to the x-axis
  (A.1 = c ∧ A.2 = (b * c) / a) →
  -- AB is perpendicular to OB
  ((A.2 - B.2) / (A.1 - B.1) = a / b) →
  -- BF is parallel to OA
  ((F.2 - B.2) / (F.1 - B.1) = A.2 / A.1) →
  -- The eccentricity of the hyperbola
  c / a = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2074_207461


namespace NUMINAMATH_CALUDE_oil_price_reduction_l2074_207480

/-- Given a price reduction that allows buying 3 kg more for Rs. 700,
    and a reduced price of Rs. 70 per kg, prove that the percentage
    reduction in the price of oil is 30%. -/
theorem oil_price_reduction (original_price : ℝ) :
  (∃ (reduced_price : ℝ),
    reduced_price = 70 ∧
    700 / original_price + 3 = 700 / reduced_price) →
  (original_price - 70) / original_price * 100 = 30 :=
by sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l2074_207480


namespace NUMINAMATH_CALUDE_tulip_area_l2074_207459

/-- Given a flower bed with roses and tulips, calculate the area occupied by tulips -/
theorem tulip_area (total_area : Real) (rose_fraction : Real) (tulip_fraction : Real) 
  (h1 : total_area = 2.4)
  (h2 : rose_fraction = 1/3)
  (h3 : tulip_fraction = 1/4) :
  tulip_fraction * (total_area - rose_fraction * total_area) = 0.4 := by
  sorry

#check tulip_area

end NUMINAMATH_CALUDE_tulip_area_l2074_207459


namespace NUMINAMATH_CALUDE_range_of_a_l2074_207484

/-- The set A defined by the quadratic inequality -/
def A : Set ℝ := {x | x^2 + 2*x - 8 > 0}

/-- The set B defined by the distance from a point a -/
def B (a : ℝ) : Set ℝ := {x | |x - a| < 5}

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (h : A ∪ B a = Set.univ) : a ∈ Set.Icc (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2074_207484


namespace NUMINAMATH_CALUDE_race_problem_l2074_207421

/-- The race problem -/
theorem race_problem (john_speed steve_speed : ℝ) (duration : ℝ) (final_distance : ℝ) 
  (h1 : john_speed = 4.2)
  (h2 : steve_speed = 3.7)
  (h3 : duration = 28)
  (h4 : final_distance = 2) :
  john_speed * duration - steve_speed * duration - final_distance = 12 := by
  sorry

end NUMINAMATH_CALUDE_race_problem_l2074_207421


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2074_207497

theorem polynomial_division_theorem (x : ℝ) : 
  2*x^4 - 3*x^3 + x^2 + 5*x - 7 = (x + 1)*(2*x^3 - 5*x^2 + 6*x - 1) + (-6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2074_207497


namespace NUMINAMATH_CALUDE_expression_simplification_l2074_207463

theorem expression_simplification (x : ℝ) 
  (h1 : x ≠ 2) (h2 : x ≠ 4) (h3 : x ≠ 3) (h4 : x ≠ 5) :
  (x^2 - 2*x + 1) / (x^2 - 6*x + 8) / ((x^2 - 4*x + 3) / (x^2 - 8*x + 15)) = (x - 5) / (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2074_207463


namespace NUMINAMATH_CALUDE_average_headcount_rounded_l2074_207410

def fall_03_04_headcount : ℕ := 11500
def fall_04_05_headcount : ℕ := 11600
def fall_05_06_headcount : ℕ := 11300

def average_headcount : ℚ :=
  (fall_03_04_headcount + fall_04_05_headcount + fall_05_06_headcount) / 3

def round_to_nearest (x : ℚ) : ℤ :=
  if x - ⌊x⌋ < 1/2 then ⌊x⌋ else ⌈x⌉

theorem average_headcount_rounded : 
  round_to_nearest average_headcount = 11467 := by sorry

end NUMINAMATH_CALUDE_average_headcount_rounded_l2074_207410


namespace NUMINAMATH_CALUDE_quintuplet_babies_count_l2074_207482

/-- Represents the number of sets of a given multiple birth type -/
structure MultipleBirthSets where
  twins : ℕ
  triplets : ℕ
  quintuplets : ℕ

/-- Calculates the total number of babies from multiple birth sets -/
def totalBabies (s : MultipleBirthSets) : ℕ :=
  2 * s.twins + 3 * s.triplets + 5 * s.quintuplets

theorem quintuplet_babies_count (s : MultipleBirthSets) :
  s.triplets = 6 * s.quintuplets →
  s.twins = 2 * s.triplets →
  totalBabies s = 1500 →
  5 * s.quintuplets = 160 := by
sorry

end NUMINAMATH_CALUDE_quintuplet_babies_count_l2074_207482


namespace NUMINAMATH_CALUDE_point_coordinates_l2074_207425

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between a point and the x-axis -/
def distanceToXAxis (p : Point) : ℝ := |p.y|

/-- The distance between a point and the y-axis -/
def distanceToYAxis (p : Point) : ℝ := |p.x|

/-- Predicate to check if a point is in the fourth quadrant -/
def inFourthQuadrant (p : Point) : Prop := p.x > 0 ∧ p.y < 0

theorem point_coordinates (p : Point) 
  (h1 : inFourthQuadrant p) 
  (h2 : distanceToXAxis p = 2) 
  (h3 : distanceToYAxis p = 5) : 
  p = Point.mk 5 (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2074_207425


namespace NUMINAMATH_CALUDE_triangle_side_length_l2074_207405

theorem triangle_side_length (A B C : Real) (tanA : Real) (angleC : Real) (BC : Real) :
  tanA = 1 / 3 →
  angleC = 150 * π / 180 →
  BC = 1 →
  let sinA := Real.sqrt (1 - 1 / (1 + tanA^2))
  let AB := BC * Real.sin angleC / sinA
  AB = Real.sqrt 10 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2074_207405


namespace NUMINAMATH_CALUDE_min_monkeys_correct_l2074_207489

/-- Represents the problem of transporting weapons with monkeys --/
structure WeaponTransport where
  total_weight : ℕ
  max_weapon_weight : ℕ
  max_monkey_capacity : ℕ

/-- Calculates the minimum number of monkeys needed to transport all weapons --/
def min_monkeys_needed (wt : WeaponTransport) : ℕ :=
  23

/-- Theorem stating that the minimum number of monkeys needed is correct --/
theorem min_monkeys_correct (wt : WeaponTransport) 
  (h1 : wt.total_weight = 600)
  (h2 : wt.max_weapon_weight = 30)
  (h3 : wt.max_monkey_capacity = 50) :
  min_monkeys_needed wt = 23 ∧ 
  ∀ n : ℕ, n < 23 → ¬ (n * wt.max_monkey_capacity ≥ wt.total_weight) :=
sorry

end NUMINAMATH_CALUDE_min_monkeys_correct_l2074_207489


namespace NUMINAMATH_CALUDE_coin_value_equality_l2074_207491

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of quarters in the first group -/
def quarters_1 : ℕ := 15

/-- The number of dimes in the first group -/
def dimes_1 : ℕ := 10

/-- The number of quarters in the second group -/
def quarters_2 : ℕ := 25

theorem coin_value_equality (n : ℕ) : 
  quarters_1 * quarter_value + dimes_1 * dime_value = 
  quarters_2 * quarter_value + n * dime_value → n = 35 := by
sorry

end NUMINAMATH_CALUDE_coin_value_equality_l2074_207491


namespace NUMINAMATH_CALUDE_extremum_when_a_zero_range_of_m_l2074_207438

noncomputable section

def g (a x : ℝ) : ℝ := (2 - a) * Real.log x

def h (a x : ℝ) : ℝ := Real.log x + a * x^2

def f (a x : ℝ) : ℝ := g a x + (deriv (h a)) x

theorem extremum_when_a_zero :
  let f₀ := f 0
  ∃ (x_min : ℝ), x_min = 1/2 ∧ 
    (∀ x > 0, f₀ x ≥ f₀ x_min) ∧
    f₀ x_min = 2 - 2 * Real.log 2 ∧
    (∀ M : ℝ, ∃ x > 0, f₀ x > M) :=
sorry

theorem range_of_m (a : ℝ) (h : -8 < a ∧ a < -2) :
  let m_lower := 2 / (3 * Real.exp 2) - 4
  ∀ m > m_lower,
    ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 3 → x₂ ∈ Set.Icc 1 3 →
      |f a x₁ - f a x₂| > (m + Real.log 3) * a - 2 * Real.log 3 + 2/3 * Real.log (-a) :=
sorry

end NUMINAMATH_CALUDE_extremum_when_a_zero_range_of_m_l2074_207438


namespace NUMINAMATH_CALUDE_total_celestial_bodies_l2074_207475

-- Define the number of planets
def num_planets : ℕ := 20

-- Define the ratio of solar systems to planets
def solar_system_ratio : ℕ := 8

-- Theorem: The total number of solar systems and planets is 180
theorem total_celestial_bodies : 
  num_planets * (solar_system_ratio + 1) = 180 := by
  sorry

end NUMINAMATH_CALUDE_total_celestial_bodies_l2074_207475


namespace NUMINAMATH_CALUDE_f_less_than_g_implies_a_bound_l2074_207439

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + 1

noncomputable def g (x : ℝ) : ℝ := Real.log x - x / 4 + 3 / (4 * x)

theorem f_less_than_g_implies_a_bound (a : ℝ) :
  (∀ x₁ > 0, ∃ x₂ > 1, f a x₁ < g x₂) →
  a > (1 / 3) * Real.exp (1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_f_less_than_g_implies_a_bound_l2074_207439


namespace NUMINAMATH_CALUDE_line_parallel_to_intersection_l2074_207452

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Define the intersection operation between two planes
variable (intersect_planes : Plane → Plane → Line)

-- Define the parallel relation between two lines
variable (parallel_lines : Line → Line → Prop)

-- Theorem statement
theorem line_parallel_to_intersection
  (m n : Line) (α β : Plane)
  (h1 : parallel_line_plane m α)
  (h2 : subset_line_plane m β)
  (h3 : intersect_planes α β = n) :
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_intersection_l2074_207452


namespace NUMINAMATH_CALUDE_solution_set_of_even_increasing_function_l2074_207478

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := (x - 2) * (a * x + b)

-- State the theorem
theorem solution_set_of_even_increasing_function 
  (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x)) 
  (h_increasing : ∀ x y, 0 < x → x < y → f a b x < f a b y) :
  {x : ℝ | f a b (2 - x) > 0} = {x : ℝ | x > 4 ∨ x < 0} := by
sorry

end NUMINAMATH_CALUDE_solution_set_of_even_increasing_function_l2074_207478


namespace NUMINAMATH_CALUDE_simplify_expression_l2074_207446

theorem simplify_expression (b : ℝ) : ((3 * b + 6) - 6 * b) / 3 = -b + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2074_207446


namespace NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_l2074_207488

-- Define the types for lines and angles
variable (Line : Type) (Angle : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (corresponding_angles : Line → Line → Angle → Angle → Prop)
variable (equal_angles : Angle → Angle → Prop)

-- State the theorem
theorem parallel_lines_corresponding_angles 
  (l1 l2 : Line) (a1 a2 : Angle) : 
  (parallel l1 l2 → corresponding_angles l1 l2 a1 a2 → equal_angles a1 a2) ∧
  (corresponding_angles l1 l2 a1 a2 → equal_angles a1 a2 → parallel l1 l2) ∧
  (¬parallel l1 l2 → corresponding_angles l1 l2 a1 a2 → ¬equal_angles a1 a2) ∧
  (corresponding_angles l1 l2 a1 a2 → ¬equal_angles a1 a2 → ¬parallel l1 l2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_l2074_207488


namespace NUMINAMATH_CALUDE_special_numbers_l2074_207401

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def satisfies_condition (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → is_prime (d^2 - d + 1) ∧ is_prime (d^2 + d + 1)

theorem special_numbers :
  ∀ n : ℕ, satisfies_condition n ↔ n = 2 ∨ n = 3 ∨ n = 6 :=
sorry

end NUMINAMATH_CALUDE_special_numbers_l2074_207401


namespace NUMINAMATH_CALUDE_rectangle_area_l2074_207422

theorem rectangle_area (square_area : ℝ) (rectangle_width rectangle_length : ℝ) : 
  square_area = 16 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 48 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2074_207422


namespace NUMINAMATH_CALUDE_A_equals_B_l2074_207460

def A (a : ℕ) : Set ℕ :=
  {k : ℕ | ∃ x y : ℤ, x > Real.sqrt a ∧ k = (x^2 - a) / (x^2 - y^2)}

def B (a : ℕ) : Set ℕ :=
  {k : ℕ | ∃ x y : ℤ, 0 ≤ x ∧ x < Real.sqrt a ∧ k = (x^2 - a) / (x^2 - y^2)}

theorem A_equals_B (a : ℕ) (h : ¬ ∃ n : ℕ, n^2 = a) : A a = B a := by
  sorry

end NUMINAMATH_CALUDE_A_equals_B_l2074_207460


namespace NUMINAMATH_CALUDE_molecular_weight_proof_l2074_207447

/-- Given a compound where 9 moles weigh 8100 grams, prove that its molecular weight is 900 grams/mole. -/
theorem molecular_weight_proof (compound : Type) 
  (moles : ℕ) (total_weight : ℝ) (molecular_weight : ℝ) 
  (h1 : moles = 9) 
  (h2 : total_weight = 8100) 
  (h3 : total_weight = moles * molecular_weight) : 
  molecular_weight = 900 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_proof_l2074_207447


namespace NUMINAMATH_CALUDE_number_at_2002_2003_l2074_207464

/-- Represents the number at position (row, col) in the arrangement -/
def number_at_position (row : ℕ) (col : ℕ) : ℕ :=
  (col - 1)^2 + 1 + (row - 1)

/-- The theorem to be proved -/
theorem number_at_2002_2003 :
  number_at_position 2002 2003 = 2002 * 2003 := by
  sorry

#check number_at_2002_2003

end NUMINAMATH_CALUDE_number_at_2002_2003_l2074_207464


namespace NUMINAMATH_CALUDE_fruit_problem_equations_l2074_207458

/-- Represents the ancient Chinese fruit problem --/
structure FruitProblem where
  totalFruits : ℕ
  totalCost : ℕ
  bitterFruitCount : ℕ
  bitterFruitCost : ℕ
  sweetFruitCount : ℕ
  sweetFruitCost : ℕ

/-- The system of equations for the fruit problem --/
def fruitEquations (p : FruitProblem) (x y : ℚ) : Prop :=
  x + y = p.totalFruits ∧
  (4 / 7 : ℚ) * x + (11 / 9 : ℚ) * y = p.totalCost

/-- Theorem stating that the given system of equations correctly represents the fruit problem --/
theorem fruit_problem_equations (p : FruitProblem) 
  (h1 : p.totalFruits = 1000)
  (h2 : p.totalCost = 999)
  (h3 : p.bitterFruitCount = 7)
  (h4 : p.bitterFruitCost = 4)
  (h5 : p.sweetFruitCount = 9)
  (h6 : p.sweetFruitCost = 11) :
  ∃ x y : ℚ, fruitEquations p x y :=
sorry

end NUMINAMATH_CALUDE_fruit_problem_equations_l2074_207458


namespace NUMINAMATH_CALUDE_quadratic_minimum_quadratic_achieves_minimum_l2074_207499

theorem quadratic_minimum (x : ℝ) (h : x > 0) : x^2 - 2*x + 3 ≥ 2 := by
  sorry

theorem quadratic_achieves_minimum : ∃ (x : ℝ), x > 0 ∧ x^2 - 2*x + 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_quadratic_achieves_minimum_l2074_207499


namespace NUMINAMATH_CALUDE_square_sum_and_product_l2074_207486

theorem square_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 1) 
  (h2 : (x - y)^2 = 49) : 
  x^2 + y^2 = 25 ∧ x * y = -12 := by
sorry

end NUMINAMATH_CALUDE_square_sum_and_product_l2074_207486


namespace NUMINAMATH_CALUDE_total_towels_calculation_l2074_207496

/-- The number of loads of laundry washed -/
def loads : ℕ := 6

/-- The number of towels in each load -/
def towels_per_load : ℕ := 7

/-- The total number of towels washed -/
def total_towels : ℕ := loads * towels_per_load

theorem total_towels_calculation : total_towels = 42 := by
  sorry

end NUMINAMATH_CALUDE_total_towels_calculation_l2074_207496


namespace NUMINAMATH_CALUDE_rational_fraction_implication_l2074_207437

theorem rational_fraction_implication (x : ℝ) :
  (∃ a : ℚ, x / (x^2 + x + 1) = a) →
  (∃ b : ℚ, x^2 / (x^4 + x^2 + 1) = b) :=
by sorry

end NUMINAMATH_CALUDE_rational_fraction_implication_l2074_207437


namespace NUMINAMATH_CALUDE_distance_A_to_C_l2074_207444

/-- Proves the distance between city A and C given travel times, distance A to B, and speed ratio -/
theorem distance_A_to_C 
  (eddy_time : ℝ) 
  (freddy_time : ℝ) 
  (distance_AB : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : eddy_time = 3) 
  (h2 : freddy_time = 4) 
  (h3 : distance_AB = 540) 
  (h4 : speed_ratio = 2.4) : 
  distance_AB * freddy_time / (eddy_time * speed_ratio) = 300 := by
sorry

end NUMINAMATH_CALUDE_distance_A_to_C_l2074_207444


namespace NUMINAMATH_CALUDE_variance_of_X_l2074_207470

/-- The probability of Person A hitting the target -/
def prob_A : ℚ := 2/3

/-- The probability of Person B hitting the target -/
def prob_B : ℚ := 4/5

/-- The random variable X representing the number of people hitting the target -/
def X : ℕ → ℚ
| 0 => (1 - prob_A) * (1 - prob_B)
| 1 => prob_A * (1 - prob_B) + (1 - prob_A) * prob_B
| 2 => prob_A * prob_B
| _ => 0

/-- The expected value of X -/
def E_X : ℚ := 0 * X 0 + 1 * X 1 + 2 * X 2

/-- The variance of X -/
def Var_X : ℚ := (0 - E_X)^2 * X 0 + (1 - E_X)^2 * X 1 + (2 - E_X)^2 * X 2

theorem variance_of_X : Var_X = 86/225 := by sorry

end NUMINAMATH_CALUDE_variance_of_X_l2074_207470


namespace NUMINAMATH_CALUDE_sum_remainder_l2074_207400

theorem sum_remainder (a b c : ℕ) (ha : a % 36 = 15) (hb : b % 36 = 22) (hc : c % 36 = 9) :
  (a + b + c) % 36 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l2074_207400


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2074_207443

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = (1 : ℝ) / 4)
  (h_S : S = 80)
  (h_sum : S = a / (1 - r)) :
  a = 60 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2074_207443
