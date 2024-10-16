import Mathlib

namespace NUMINAMATH_CALUDE_oplus_example_1_oplus_example_2_l2068_206880

-- Define the ⊕ operation for rational numbers
def oplus (a b : ℚ) : ℚ := (a + 3 * b) / 2

-- Theorem for part (1)
theorem oplus_example_1 : 4 * (oplus 2 5) = 34 := by sorry

-- Define polynomials A and B
def A (x y : ℚ) : ℚ := x^2 + 2*x*y + y^2
def B (x y : ℚ) : ℚ := -2*x*y + y^2

-- Theorem for part (2)
theorem oplus_example_2 (x y : ℚ) : 
  (oplus (A x y) (B x y)) + (oplus (B x y) (A x y)) = 2*x^2 + 4*y^2 := by sorry

end NUMINAMATH_CALUDE_oplus_example_1_oplus_example_2_l2068_206880


namespace NUMINAMATH_CALUDE_birthday_party_guests_solve_birthday_party_guests_l2068_206895

theorem birthday_party_guests : ℕ → Prop :=
  fun total_guests =>
    -- Define the number of women, men, and children
    let women := total_guests / 2
    let men := 15
    let children := total_guests - women - men

    -- Define the number of people who left
    let men_left := men / 3
    let children_left := 5

    -- Define the number of people who stayed
    let people_stayed := total_guests - men_left - children_left

    -- State the conditions and the conclusion
    women = men ∧
    women + men + children = total_guests ∧
    people_stayed = 50 ∧
    total_guests = 60

-- The proof of the theorem
theorem solve_birthday_party_guests : birthday_party_guests 60 := by
  sorry

#check solve_birthday_party_guests

end NUMINAMATH_CALUDE_birthday_party_guests_solve_birthday_party_guests_l2068_206895


namespace NUMINAMATH_CALUDE_two_in_S_l2068_206873

def S : Set ℕ := {0, 1, 2}

theorem two_in_S : 2 ∈ S := by sorry

end NUMINAMATH_CALUDE_two_in_S_l2068_206873


namespace NUMINAMATH_CALUDE_dormitory_problem_l2068_206860

theorem dormitory_problem : ∃! x : ℕ+, ∃ n : ℕ+, 
  (x = 4 * n + 20) ∧ 
  (↑(n - 1) < (↑x : ℚ) / 8 ∧ (↑x : ℚ) / 8 < ↑n) := by
  sorry

end NUMINAMATH_CALUDE_dormitory_problem_l2068_206860


namespace NUMINAMATH_CALUDE_complex_number_location_l2068_206841

theorem complex_number_location (z : ℂ) (h : z * (-1 + 2 * Complex.I) = Complex.abs (1 + 3 * Complex.I)) :
  z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l2068_206841


namespace NUMINAMATH_CALUDE_total_smoothie_time_l2068_206885

/-- The time it takes to freeze ice cubes (in minutes) -/
def freezing_time : ℕ := 40

/-- The time it takes to make one smoothie (in minutes) -/
def smoothie_time : ℕ := 3

/-- The number of smoothies to be made -/
def num_smoothies : ℕ := 5

/-- Theorem stating the total time to make the smoothies -/
theorem total_smoothie_time : 
  freezing_time + num_smoothies * smoothie_time = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_smoothie_time_l2068_206885


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l2068_206811

theorem complex_magnitude_theorem (ω : ℂ) (h : ω = 8 + 3*I) : 
  Complex.abs (ω^2 + 6*ω + 73) = Real.sqrt 32740 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l2068_206811


namespace NUMINAMATH_CALUDE_greatest_integer_not_divisible_by_1111_l2068_206824

theorem greatest_integer_not_divisible_by_1111 :
  (∃ (N : ℕ), N > 0 ∧
    (∃ (x : Fin N → ℤ), ∀ (i j : Fin N), i ≠ j →
      ¬(1111 ∣ (x i)^2 - (x i) * (x j))) ∧
    (∀ (M : ℕ), M > N →
      ¬(∃ (y : Fin M → ℤ), ∀ (i j : Fin M), i ≠ j →
        ¬(1111 ∣ (y i)^2 - (y i) * (y j)))) ∧
  N = 1000) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_not_divisible_by_1111_l2068_206824


namespace NUMINAMATH_CALUDE_jerry_remaining_money_l2068_206832

/-- Given Jerry's initial amount and his expenses, calculate his remaining money. -/
theorem jerry_remaining_money (initial : ℕ) (video_games : ℕ) (snack : ℕ) :
  initial = 18 ∧ video_games = 6 ∧ snack = 3 →
  initial - (video_games + snack) = 9 := by
  sorry


end NUMINAMATH_CALUDE_jerry_remaining_money_l2068_206832


namespace NUMINAMATH_CALUDE_order_of_rationals_l2068_206819

theorem order_of_rationals (a b : ℚ) (h : b < 0) : a + b < a ∧ a < a - b := by
  sorry

end NUMINAMATH_CALUDE_order_of_rationals_l2068_206819


namespace NUMINAMATH_CALUDE_time_per_video_l2068_206816

-- Define the parameters
def setup_time : ℝ := 1
def cleanup_time : ℝ := 1
def painting_time_per_video : ℝ := 1
def editing_time_per_video : ℝ := 1.5
def num_videos : ℕ := 4

-- Define the theorem
theorem time_per_video : 
  (setup_time + cleanup_time + num_videos * painting_time_per_video + num_videos * editing_time_per_video) / num_videos = 3 := by
  sorry

end NUMINAMATH_CALUDE_time_per_video_l2068_206816


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2068_206845

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, (5 * x - 6 > x^2) → (|x + 1| > 2)) ∧
  (∃ x : ℝ, (|x + 1| > 2) ∧ ¬(5 * x - 6 > x^2)) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2068_206845


namespace NUMINAMATH_CALUDE_additions_per_hour_l2068_206852

/-- Represents the number of operations a computer can perform per second -/
def operations_per_second : ℕ := 15000

/-- Represents the number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem stating that the number of additions performed in an hour is 27 million -/
theorem additions_per_hour :
  (operations_per_second / 2) * seconds_per_hour = 27000000 := by
  sorry

end NUMINAMATH_CALUDE_additions_per_hour_l2068_206852


namespace NUMINAMATH_CALUDE_total_stones_in_five_piles_l2068_206889

/-- Given five piles of stones with the following properties:
    1. The number of stones in the fifth pile is six times the number of stones in the third pile
    2. The number of stones in the second pile is twice the total number of stones in the third and fifth piles combined
    3. The number of stones in the first pile is three times less than the number in the fifth pile and 10 less than the number in the fourth pile
    4. The number of stones in the fourth pile is half the number in the second pile
    Prove that the total number of stones in all five piles is 60. -/
theorem total_stones_in_five_piles (p1 p2 p3 p4 p5 : ℕ) 
  (h1 : p5 = 6 * p3)
  (h2 : p2 = 2 * (p3 + p5))
  (h3 : p1 = p5 / 3 ∧ p1 = p4 - 10)
  (h4 : p4 = p2 / 2) :
  p1 + p2 + p3 + p4 + p5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_stones_in_five_piles_l2068_206889


namespace NUMINAMATH_CALUDE_log_equality_implies_value_l2068_206877

theorem log_equality_implies_value (x : ℝ) (h : Real.log x = Real.log 4 + Real.log 3) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_value_l2068_206877


namespace NUMINAMATH_CALUDE_percentage_of_flowering_plants_l2068_206870

/-- Proves that the percentage of flowering plants is 40% given the conditions --/
theorem percentage_of_flowering_plants 
  (total_plants : ℕ)
  (porch_fraction : ℚ)
  (flowers_per_plant : ℕ)
  (total_porch_flowers : ℕ)
  (h1 : total_plants = 80)
  (h2 : porch_fraction = 1 / 4)
  (h3 : flowers_per_plant = 5)
  (h4 : total_porch_flowers = 40) :
  (total_porch_flowers : ℚ) / (porch_fraction * flowers_per_plant * total_plants) = 40 / 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_flowering_plants_l2068_206870


namespace NUMINAMATH_CALUDE_veena_payment_fraction_l2068_206863

/-- Represents the payment amounts of 6 friends at a restaurant -/
structure DinnerPayment where
  akshitha : ℚ
  veena : ℚ
  lasya : ℚ
  sandhya : ℚ
  ramesh : ℚ
  kavya : ℚ

/-- Theorem stating that Veena paid 1/8 of the total bill -/
theorem veena_payment_fraction (p : DinnerPayment) 
  (h1 : p.akshitha = 3/4 * p.veena)
  (h2 : p.veena = 1/2 * p.lasya)
  (h3 : p.lasya = 5/6 * p.sandhya)
  (h4 : p.sandhya = 4/8 * p.ramesh)
  (h5 : p.ramesh = 3/5 * p.kavya)
  : p.veena = 1/8 * (p.akshitha + p.veena + p.lasya + p.sandhya + p.ramesh + p.kavya) := by
  sorry


end NUMINAMATH_CALUDE_veena_payment_fraction_l2068_206863


namespace NUMINAMATH_CALUDE_maurice_age_l2068_206883

theorem maurice_age (ron_age : ℕ) (maurice_age : ℕ) : 
  ron_age = 43 → 
  ron_age + 5 = 4 * (maurice_age + 5) → 
  maurice_age = 7 := by
sorry

end NUMINAMATH_CALUDE_maurice_age_l2068_206883


namespace NUMINAMATH_CALUDE_no_real_roots_iff_k_gt_two_l2068_206856

theorem no_real_roots_iff_k_gt_two (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 2 * x + (1/2) ≠ 0) ↔ k > 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_iff_k_gt_two_l2068_206856


namespace NUMINAMATH_CALUDE_cubic_equation_root_sum_squares_l2068_206876

theorem cubic_equation_root_sum_squares (a b c : ℝ) : 
  a^3 - 6*a^2 - 7*a + 2 = 0 →
  b^3 - 6*b^2 - 7*b + 2 = 0 →
  c^3 - 6*c^2 - 7*c + 2 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 73/4 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_sum_squares_l2068_206876


namespace NUMINAMATH_CALUDE_range_of_sum_on_circle_l2068_206888

theorem range_of_sum_on_circle (x y : ℝ) (h : x^2 + y^2 - 4*x + 3 = 0) :
  ∃ (min max : ℝ), min = 2 - Real.sqrt 2 ∧ max = 2 + Real.sqrt 2 ∧
  min ≤ x + y ∧ x + y ≤ max :=
sorry

end NUMINAMATH_CALUDE_range_of_sum_on_circle_l2068_206888


namespace NUMINAMATH_CALUDE_hacker_can_achieve_goal_l2068_206830

/-- Represents a user in the social network -/
structure User where
  id : Nat
  followers : Finset Nat
  rating : Nat

/-- Represents the social network -/
structure SocialNetwork where
  users : Finset User
  m : Nat

/-- Represents a hacker's action: increasing a user's rating by 1 or doing nothing -/
inductive HackerAction
  | Increase (userId : Nat)
  | DoNothing

/-- Update ratings based on followers -/
def updateRatings (sn : SocialNetwork) : SocialNetwork :=
  sorry

/-- Apply hacker's action to the social network -/
def applyHackerAction (sn : SocialNetwork) (action : HackerAction) : SocialNetwork :=
  sorry

/-- Check if all ratings are divisible by m -/
def allRatingsDivisible (sn : SocialNetwork) : Prop :=
  sorry

/-- The main theorem -/
theorem hacker_can_achieve_goal (sn : SocialNetwork) :
  ∃ (actions : List HackerAction), allRatingsDivisible (actions.foldl applyHackerAction sn) :=
sorry

end NUMINAMATH_CALUDE_hacker_can_achieve_goal_l2068_206830


namespace NUMINAMATH_CALUDE_calculate_gross_profit_l2068_206813

/-- Calculate the gross profit given the sales price, gross profit margin, sales tax, and initial discount --/
theorem calculate_gross_profit (sales_price : ℝ) (gross_profit_margin : ℝ) (sales_tax : ℝ) (initial_discount : ℝ) :
  sales_price = 81 →
  gross_profit_margin = 1.7 →
  sales_tax = 0.07 →
  initial_discount = 0.15 →
  ∃ (gross_profit : ℝ), abs (gross_profit - 56.07) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_calculate_gross_profit_l2068_206813


namespace NUMINAMATH_CALUDE_max_overlapping_squares_theorem_l2068_206862

/-- Represents a square on the checkerboard -/
structure CheckerboardSquare where
  sideLength : Real
  (side_positive : sideLength > 0)

/-- Represents the square card -/
structure Card where
  sideLength : Real
  (side_positive : sideLength > 0)

/-- Calculates the maximum number of squares a card can overlap -/
def maxOverlappingSquares (square : CheckerboardSquare) (card : Card) (minOverlap : Real) : Nat :=
  sorry

theorem max_overlapping_squares_theorem (square : CheckerboardSquare) (card : Card) :
  square.sideLength = 0.75 →
  card.sideLength = 2 →
  maxOverlappingSquares square card 0.25 = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_overlapping_squares_theorem_l2068_206862


namespace NUMINAMATH_CALUDE_ellipse_properties_and_fixed_point_l2068_206828

/-- Ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Point on the x-axis -/
structure XAxisPoint where
  x : ℝ

/-- Intersection points of a line with the ellipse -/
structure IntersectionPoints where
  M : ℝ × ℝ
  N : ℝ × ℝ

/-- The theorem stating the properties of the ellipse and the existence of point D -/
theorem ellipse_properties_and_fixed_point 
  (C : Ellipse) 
  (h_eccentricity : C.a * C.a = C.b * C.b + 1)
  (h_max_area : C.a * C.b = 2 * Real.sqrt 3) :
  (∃ (D : XAxisPoint), 
    D.x = -11/8 ∧
    (∀ (l : IntersectionPoints), 
      let vec_DM := (l.M.1 - D.x, l.M.2)
      let vec_DN := (l.N.1 - D.x, l.N.2)
      (vec_DM.1 * vec_DN.1 + vec_DM.2 * vec_DN.2) = -135/64)) ∧
  (C.a = 2 ∧ C.b = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_and_fixed_point_l2068_206828


namespace NUMINAMATH_CALUDE_seashell_ratio_l2068_206897

def seashells_day1 : ℕ := 5
def seashells_day2 : ℕ := 7
def total_seashells : ℕ := 36

def seashells_first_two_days : ℕ := seashells_day1 + seashells_day2
def seashells_day3 : ℕ := total_seashells - seashells_first_two_days

theorem seashell_ratio :
  seashells_day3 / seashells_first_two_days = 2 := by sorry

end NUMINAMATH_CALUDE_seashell_ratio_l2068_206897


namespace NUMINAMATH_CALUDE_no_solution_to_equation_l2068_206803

theorem no_solution_to_equation :
  ¬∃ (x : ℝ), (x - 1) / (x + 1) - 4 / (x^2 - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_to_equation_l2068_206803


namespace NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l2068_206851

theorem greatest_four_digit_multiple_of_17 : ∃ n : ℕ, 
  n ≤ 9999 ∧ 
  n > 999 ∧
  n % 17 = 0 ∧
  ∀ m : ℕ, m ≤ 9999 ∧ m > 999 ∧ m % 17 = 0 → m ≤ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l2068_206851


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2068_206849

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (x_nonneg : x ≥ 0)
  (y_geq : y ≥ -3/2)
  (z_geq : z ≥ -1) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 ∧ 
    ∀ a b c : ℝ, a + b + c = 3 → a ≥ 0 → b ≥ -3/2 → c ≥ -1 →
      Real.sqrt (2 * a) + Real.sqrt (2 * b + 3) + Real.sqrt (2 * c + 2) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2068_206849


namespace NUMINAMATH_CALUDE_power_three_mod_five_l2068_206843

theorem power_three_mod_five : 3^19 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_five_l2068_206843


namespace NUMINAMATH_CALUDE_range_of_f_on_interval_existence_of_a_l2068_206887

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a

-- Part 1
theorem range_of_f_on_interval :
  let f1 := f 1
  ∃ (y : ℝ), y ∈ Set.range (fun x => f1 x) ∩ Set.Icc 0 4 ∧
  ∀ (z : ℝ), z ∈ Set.range (fun x => f1 x) ∩ Set.Icc 0 3 → z ∈ Set.Icc 0 4 :=
sorry

-- Part 2
theorem existence_of_a :
  ∃ (a : ℝ), 
    (∀ x, x ∈ Set.Icc (-1) 1 → f a x ∈ Set.Icc (-2) 2) ∧
    (∀ y, y ∈ Set.Icc (-2) 2 → ∃ x ∈ Set.Icc (-1) 1, f a x = y) ∧
    a = -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_on_interval_existence_of_a_l2068_206887


namespace NUMINAMATH_CALUDE_power_expression_equality_l2068_206833

theorem power_expression_equality (a b : ℝ) 
  (h1 : (40 : ℝ) ^ a = 2) 
  (h2 : (40 : ℝ) ^ b = 5) : 
  (20 : ℝ) ^ ((1 - a - b) / (2 * (1 - b))) = (20 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_power_expression_equality_l2068_206833


namespace NUMINAMATH_CALUDE_min_value_expression_l2068_206886

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (3 * x) / (3 * x + 2 * y) + y / (2 * x + y) ≥ 4 * Real.sqrt 3 - 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2068_206886


namespace NUMINAMATH_CALUDE_first_number_in_expression_l2068_206878

theorem first_number_in_expression : ∃ x : ℝ, (x * 12 * 20) / 3 + 125 = 2229 ∧ x = 26.3 := by
  sorry

end NUMINAMATH_CALUDE_first_number_in_expression_l2068_206878


namespace NUMINAMATH_CALUDE_leaves_broke_after_initial_loss_l2068_206861

/-- 
Given that Ryan initially collected 89 leaves, lost 24 leaves, and now has 22 leaves left,
this theorem proves that 43 leaves broke after the initial loss of 24 leaves.
-/
theorem leaves_broke_after_initial_loss 
  (initial_leaves : ℕ) 
  (initial_loss : ℕ) 
  (final_leaves : ℕ) 
  (h1 : initial_leaves = 89)
  (h2 : initial_loss = 24)
  (h3 : final_leaves = 22) :
  initial_leaves - initial_loss - final_leaves = 43 := by
  sorry

end NUMINAMATH_CALUDE_leaves_broke_after_initial_loss_l2068_206861


namespace NUMINAMATH_CALUDE_trajectory_of_m_l2068_206826

/-- The trajectory of point M given point P on the unit circle --/
theorem trajectory_of_m (x y : ℝ) (h1 : y ≠ 0) : 
  (∃ (t : ℝ), t^2 + (3*y/2)^2 = 1) → x^2 + 9*y^2/4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_m_l2068_206826


namespace NUMINAMATH_CALUDE_min_positive_temperatures_l2068_206896

theorem min_positive_temperatures (n : ℕ) (pos_products neg_products : ℕ) :
  n = 11 →
  pos_products = 62 →
  neg_products = 48 →
  ∃ (pos_temps : ℕ), pos_temps ≥ 3 ∧
    pos_temps * (pos_temps - 1) = pos_products ∧
    (n - pos_temps) * (n - 1 - pos_temps) = neg_products ∧
    ∀ (k : ℕ), k < pos_temps →
      k * (k - 1) ≠ pos_products ∨ (n - k) * (n - 1 - k) ≠ neg_products :=
by sorry

end NUMINAMATH_CALUDE_min_positive_temperatures_l2068_206896


namespace NUMINAMATH_CALUDE_intersection_M_N_l2068_206800

def M : Set ℤ := {-1, 1}
def N : Set ℤ := {x | -1 < x ∧ x < 4}

theorem intersection_M_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2068_206800


namespace NUMINAMATH_CALUDE_motorboat_travel_time_l2068_206865

/-- Represents the scenario of a motorboat and kayak traveling on a river --/
structure RiverTrip where
  r : ℝ  -- River current speed (also kayak speed)
  p : ℝ  -- Motorboat speed relative to the river
  t : ℝ  -- Time for motorboat to travel from X to Y

/-- The conditions of the river trip --/
def trip_conditions (trip : RiverTrip) : Prop :=
  trip.p > 0 ∧ 
  trip.r > 0 ∧ 
  trip.t > 0 ∧ 
  (trip.p + trip.r) * trip.t + (trip.p - trip.r) * (11 - trip.t) = 12 * trip.r

/-- The theorem stating that under the given conditions, 
    the motorboat's initial travel time from X to Y is 4 hours --/
theorem motorboat_travel_time (trip : RiverTrip) : 
  trip_conditions trip → trip.t = 4 := by
  sorry

end NUMINAMATH_CALUDE_motorboat_travel_time_l2068_206865


namespace NUMINAMATH_CALUDE_simplify_expression_one_simplify_expression_two_l2068_206825

-- Part 1
theorem simplify_expression_one : 2 * Real.sqrt 3 * 31.5 * 612 = 6 := by sorry

-- Part 2
theorem simplify_expression_two : 
  (Real.log 3 / Real.log 4 - Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 1/4 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_one_simplify_expression_two_l2068_206825


namespace NUMINAMATH_CALUDE_function_and_triangle_properties_l2068_206884

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) - Real.cos (ω * x) ^ 2 - 1/2

theorem function_and_triangle_properties 
  (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_distance : ∀ x₁ x₂, f ω x₁ = f ω x₂ → x₂ - x₁ = π / ω ∨ x₂ - x₁ = -π / ω) 
  (A B C : ℝ) 
  (h_c : Real.sqrt 7 = 2 * Real.sin (A/2) * Real.sin (B/2))
  (h_fC : f ω C = 0) 
  (h_sinB : Real.sin B = 3 * Real.sin A) :
  (∀ k : ℤ, ∀ x ∈ Set.Icc (-π/6 + k*π) (k*π + π/3), 
    ∀ y ∈ Set.Icc (-π/6 + k*π) (k*π + π/3), 
    x ≤ y → f ω x ≤ f ω y) ∧
  2 * Real.sin (A/2) = 1 ∧ 
  2 * Real.sin (B/2) = 3 := by
sorry

end NUMINAMATH_CALUDE_function_and_triangle_properties_l2068_206884


namespace NUMINAMATH_CALUDE_exp_inequality_l2068_206817

/-- The function f(x) = (x-3)³ + 2x - 6 -/
def f (x : ℝ) : ℝ := (x - 3)^3 + 2*x - 6

/-- Theorem stating that if f(2a-b) + f(6-b) > 0, then e^a > e^b -/
theorem exp_inequality (a b : ℝ) (h : f (2*a - b) + f (6 - b) > 0) : Real.exp a > Real.exp b := by
  sorry

end NUMINAMATH_CALUDE_exp_inequality_l2068_206817


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l2068_206808

def N : ℕ := 38 * 38 * 91 * 210

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors N) * 14 = sum_even_divisors N := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l2068_206808


namespace NUMINAMATH_CALUDE_binomial_26_6_l2068_206809

theorem binomial_26_6 (h1 : Nat.choose 24 5 = 42504) 
                      (h2 : Nat.choose 24 6 = 134596) 
                      (h3 : Nat.choose 24 7 = 346104) : 
  Nat.choose 26 6 = 657800 := by
  sorry

end NUMINAMATH_CALUDE_binomial_26_6_l2068_206809


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2068_206869

/-- Calculates the average speed of a trip given the following conditions:
  * The trip lasts for 12 hours
  * The car travels at 45 mph for the first 4 hours
  * The car travels at 75 mph for the remaining hours
-/
theorem average_speed_calculation (total_time : ℝ) (initial_speed : ℝ) (initial_duration : ℝ) (final_speed : ℝ) :
  total_time = 12 →
  initial_speed = 45 →
  initial_duration = 4 →
  final_speed = 75 →
  (initial_speed * initial_duration + final_speed * (total_time - initial_duration)) / total_time = 65 := by
  sorry

#check average_speed_calculation

end NUMINAMATH_CALUDE_average_speed_calculation_l2068_206869


namespace NUMINAMATH_CALUDE_function_derivative_value_l2068_206814

theorem function_derivative_value (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 * (deriv f (π/3)) + Real.sin x) :
  deriv f (π/3) = 3 / (6 - 4*π) := by sorry

end NUMINAMATH_CALUDE_function_derivative_value_l2068_206814


namespace NUMINAMATH_CALUDE_eggs_problem_l2068_206812

theorem eggs_problem (initial_eggs : ℕ) : 
  (initial_eggs / 2 : ℕ) - 15 = 21 → initial_eggs = 72 := by
  sorry

end NUMINAMATH_CALUDE_eggs_problem_l2068_206812


namespace NUMINAMATH_CALUDE_tan_negative_five_pi_fourths_l2068_206881

theorem tan_negative_five_pi_fourths : Real.tan (-5 * Real.pi / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_five_pi_fourths_l2068_206881


namespace NUMINAMATH_CALUDE_max_b_line_circle_intersection_l2068_206854

/-- The maximum value of b for a line intersecting a circle under specific conditions -/
theorem max_b_line_circle_intersection (b : ℝ) 
  (h1 : b > 0) 
  (h2 : ∃ P₁ P₂ : ℝ × ℝ, P₁ ≠ P₂ ∧ 
    (P₁.1^2 + P₁.2^2 = 4) ∧ 
    (P₂.1^2 + P₂.2^2 = 4) ∧ 
    (P₁.2 = P₁.1 + b) ∧ 
    (P₂.2 = P₂.1 + b))
  (h3 : ∀ P₁ P₂ : ℝ × ℝ, P₁ ≠ P₂ → 
    (P₁.1^2 + P₁.2^2 = 4) → 
    (P₂.1^2 + P₂.2^2 = 4) → 
    (P₁.2 = P₁.1 + b) → 
    (P₂.2 = P₂.1 + b) → 
    ((P₁.1 - P₂.1)^2 + (P₁.2 - P₂.2)^2 ≥ (P₁.1 + P₂.1)^2 + (P₁.2 + P₂.2)^2)) : 
  b ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_b_line_circle_intersection_l2068_206854


namespace NUMINAMATH_CALUDE_minimum_employees_l2068_206844

theorem minimum_employees (customer_service technical_support both : ℕ) 
  (h1 : customer_service = 95)
  (h2 : technical_support = 80)
  (h3 : both = 30)
  (h4 : both ≤ customer_service ∧ both ≤ technical_support) :
  customer_service + technical_support - both = 145 := by
  sorry

end NUMINAMATH_CALUDE_minimum_employees_l2068_206844


namespace NUMINAMATH_CALUDE_gray_trees_sum_l2068_206898

/-- Represents the number of trees in a photograph -/
structure PhotoTrees where
  total : ℕ
  white : ℕ
  gray : ℕ

/-- The problem statement -/
theorem gray_trees_sum (photo1 photo2 photo3 : PhotoTrees) :
  photo1.total = 100 →
  photo2.total = 90 →
  photo3.total = photo3.white →
  photo1.white = photo2.white →
  photo2.white = photo3.white →
  photo3.white = 82 →
  photo1.gray + photo2.gray = 26 :=
by sorry

end NUMINAMATH_CALUDE_gray_trees_sum_l2068_206898


namespace NUMINAMATH_CALUDE_sqrt_square_diff_l2068_206827

theorem sqrt_square_diff (a b : ℝ) :
  (a > b → Real.sqrt ((a - b)^2) = a - b) ∧
  (a < b → Real.sqrt ((a - b)^2) = b - a) := by
sorry

end NUMINAMATH_CALUDE_sqrt_square_diff_l2068_206827


namespace NUMINAMATH_CALUDE_perimeter_of_rearranged_rectangles_l2068_206806

/-- The perimeter of a shape formed by rearranging two equal rectangles cut from a square --/
theorem perimeter_of_rearranged_rectangles (square_side : ℝ) : square_side = 100 → 500 = 3 * square_side + 4 * (square_side / 2) := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_rearranged_rectangles_l2068_206806


namespace NUMINAMATH_CALUDE_f_equals_f_inv_at_two_point_five_l2068_206894

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 5 * x - 3

-- Define the inverse function f⁻¹
noncomputable def f_inv (x : ℝ) : ℝ := (5 + Real.sqrt (8 * x + 49)) / 4

-- Theorem statement
theorem f_equals_f_inv_at_two_point_five :
  f 2.5 = f_inv 2.5 := by sorry

end NUMINAMATH_CALUDE_f_equals_f_inv_at_two_point_five_l2068_206894


namespace NUMINAMATH_CALUDE_factorization_proof_l2068_206822

theorem factorization_proof (a b : ℝ) : a * b^2 - 5 * a * b = a * b * (b - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2068_206822


namespace NUMINAMATH_CALUDE_max_value_of_f_l2068_206805

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 a, f x ≤ f 0 ∧ f 0 = 3) ∨
  (a > 2 ∧ ∀ x ∈ Set.Icc 0 a, f x ≤ f a ∧ f a = a^2 - 2*a + 3) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2068_206805


namespace NUMINAMATH_CALUDE_circle_arc_angle_l2068_206872

theorem circle_arc_angle (E AB BC CD AD : ℝ) : 
  E = 40 →
  AB = BC →
  BC = CD →
  AB + BC + CD + AD = 360 →
  (AB - AD) / 2 = E →
  ∃ (ACD : ℝ), ACD = 15 := by
sorry

end NUMINAMATH_CALUDE_circle_arc_angle_l2068_206872


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_l2068_206839

/-- The line equation in polar coordinates -/
def line_equation (ρ θ : ℝ) : Prop :=
  ρ * (Real.sqrt 3 * Real.cos θ - Real.sin θ) = 2

/-- The circle equation in polar coordinates -/
def circle_equation (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.sin θ

/-- The theorem stating that the point (2, π/6) satisfies both equations -/
theorem intersection_point_satisfies_equations :
  line_equation 2 (Real.pi / 6) ∧ circle_equation 2 (Real.pi / 6) := by
  sorry


end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_l2068_206839


namespace NUMINAMATH_CALUDE_inequality_proof_l2068_206842

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ¬(1/a > 1/b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2068_206842


namespace NUMINAMATH_CALUDE_second_smallest_natural_with_remainder_l2068_206868

theorem second_smallest_natural_with_remainder : ∃ n : ℕ, 
  n > 500 ∧ 
  n % 7 = 3 ∧ 
  (∃! m : ℕ, m > 500 ∧ m % 7 = 3 ∧ m < n) ∧
  n = 514 :=
by sorry

end NUMINAMATH_CALUDE_second_smallest_natural_with_remainder_l2068_206868


namespace NUMINAMATH_CALUDE_box_volume_cubic_feet_l2068_206899

/-- Conversion factor from cubic inches to cubic feet -/
def cubic_inches_per_cubic_foot : ℕ := 1728

/-- Volume of the box in cubic inches -/
def box_volume_cubic_inches : ℕ := 1728

/-- Theorem stating that the volume of the box in cubic feet is 1 -/
theorem box_volume_cubic_feet : 
  (box_volume_cubic_inches : ℚ) / cubic_inches_per_cubic_foot = 1 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_cubic_feet_l2068_206899


namespace NUMINAMATH_CALUDE_mud_weight_after_evaporation_l2068_206802

/-- 
Given a train car with mud, prove that the final weight after water evaporation
is 4000 pounds, given the initial conditions and final water percentage.
-/
theorem mud_weight_after_evaporation 
  (initial_weight : ℝ) 
  (initial_water_percent : ℝ)
  (final_water_percent : ℝ)
  (hw : initial_weight = 6000)
  (hiw : initial_water_percent = 88)
  (hfw : final_water_percent = 82) :
  (initial_weight * (100 - initial_water_percent) / 100) / ((100 - final_water_percent) / 100) = 4000 :=
by sorry

end NUMINAMATH_CALUDE_mud_weight_after_evaporation_l2068_206802


namespace NUMINAMATH_CALUDE_correct_ways_select_four_correct_ways_select_five_l2068_206820

/-- Number of distinct red balls -/
def num_red_balls : ℕ := 4

/-- Number of distinct white balls -/
def num_white_balls : ℕ := 7

/-- Score for selecting a red ball -/
def red_score : ℕ := 2

/-- Score for selecting a white ball -/
def white_score : ℕ := 1

/-- The number of ways to select 4 balls such that the number of red balls
    is not less than the number of white balls -/
def ways_select_four : ℕ := 115

/-- The number of ways to select 5 balls such that the total score
    is at least 7 points -/
def ways_select_five : ℕ := 301

/-- Theorem stating the correct number of ways to select 4 balls -/
theorem correct_ways_select_four :
  ways_select_four = Nat.choose num_red_balls 4 +
    Nat.choose num_red_balls 3 * Nat.choose num_white_balls 1 +
    Nat.choose num_red_balls 2 * Nat.choose num_white_balls 2 := by sorry

/-- Theorem stating the correct number of ways to select 5 balls -/
theorem correct_ways_select_five :
  ways_select_five = Nat.choose num_red_balls 2 * Nat.choose num_white_balls 3 +
    Nat.choose num_red_balls 3 * Nat.choose num_white_balls 2 +
    Nat.choose num_red_balls 4 * Nat.choose num_white_balls 1 := by sorry

end NUMINAMATH_CALUDE_correct_ways_select_four_correct_ways_select_five_l2068_206820


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l2068_206853

theorem greatest_value_quadratic_inequality :
  ∃ (x_max : ℝ), x_max = 9 ∧
  (∀ x : ℝ, x^2 - 14*x + 45 ≤ 0 → x ≤ x_max) ∧
  (x_max^2 - 14*x_max + 45 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l2068_206853


namespace NUMINAMATH_CALUDE_factorial_difference_l2068_206857

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 / Nat.factorial 3 = 3568320 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l2068_206857


namespace NUMINAMATH_CALUDE_original_number_is_four_fifths_l2068_206874

theorem original_number_is_four_fifths (x : ℚ) :
  1 + 1 / x = 9 / 4 → x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_four_fifths_l2068_206874


namespace NUMINAMATH_CALUDE_division_multiplication_result_l2068_206815

theorem division_multiplication_result : (-1 : ℚ) / (-5 : ℚ) * (-1/5 : ℚ) = -1/25 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_result_l2068_206815


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptote_l2068_206891

-- Define the hyperbola
def hyperbola (m : ℝ) (x y : ℝ) : Prop := y^2 - m*x^2 = 3*m

-- Define the condition that m is positive
def m_positive (m : ℝ) : Prop := m > 0

-- Define a focus of the hyperbola
def is_focus (m : ℝ) (F : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a^2 = 3*m ∧ b^2 = 3 ∧ c^2 = a^2 + b^2 ∧ 
  (F.1 = 0 ∧ F.2 = c ∨ F.1 = 0 ∧ F.2 = -c)

-- Define an asymptote of the hyperbola
def is_asymptote (m : ℝ) (l : ℝ → ℝ) : Prop :=
  ∀ x, l x = Real.sqrt m * x ∨ l x = -Real.sqrt m * x

-- Theorem statement
theorem distance_focus_to_asymptote (m : ℝ) (F : ℝ × ℝ) (l : ℝ → ℝ) :
  m_positive m →
  hyperbola m F.1 F.2 →
  is_focus m F →
  is_asymptote m l →
  ∃ (d : ℝ), d = Real.sqrt 3 ∧ 
    d = |F.2 - l F.1| / Real.sqrt (1 + (Real.sqrt m)^2) :=
sorry

end NUMINAMATH_CALUDE_distance_focus_to_asymptote_l2068_206891


namespace NUMINAMATH_CALUDE_repair_labor_hours_l2068_206829

/-- Calculates the number of labor hours given the labor cost per hour, part cost, and total repair cost. -/
def labor_hours (labor_cost_per_hour : ℕ) (part_cost : ℕ) (total_cost : ℕ) : ℕ :=
  (total_cost - part_cost) / labor_cost_per_hour

/-- Proves that given the specified costs, the number of labor hours is 16. -/
theorem repair_labor_hours :
  labor_hours 75 1200 2400 = 16 := by
  sorry

end NUMINAMATH_CALUDE_repair_labor_hours_l2068_206829


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2068_206836

theorem cyclic_sum_inequality (x y z : ℝ) (α : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) (hα : α ≥ 0) :
  ((x^(α+3) + y^(α+3)) / (x^2 + x*y + y^2) +
   (y^(α+3) + z^(α+3)) / (y^2 + y*z + z^2) +
   (z^(α+3) + x^(α+3)) / (z^2 + z*x + x^2)) ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2068_206836


namespace NUMINAMATH_CALUDE_bill_with_late_charges_l2068_206818

/-- Calculates the final amount owed after applying three consecutive 2% increases to an original bill. -/
def final_amount (original_bill : ℝ) : ℝ :=
  original_bill * (1 + 0.02)^3

/-- Theorem stating that given an original bill of $500 and three consecutive 2% increases, 
    the final amount owed is $530.604 (rounded to 3 decimal places) -/
theorem bill_with_late_charges :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0005 ∧ |final_amount 500 - 530.604| < ε :=
sorry

end NUMINAMATH_CALUDE_bill_with_late_charges_l2068_206818


namespace NUMINAMATH_CALUDE_heartsuit_zero_heartsuit_self_heartsuit_positive_l2068_206810

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem 1: x ♡ 0 = x^2 for all real x
theorem heartsuit_zero (x : ℝ) : heartsuit x 0 = x^2 := by sorry

-- Theorem 2: x ♡ x = 0 for all real x
theorem heartsuit_self (x : ℝ) : heartsuit x x = 0 := by sorry

-- Theorem 3: If x > y, then x ♡ y > 0 for all real x and y
theorem heartsuit_positive {x y : ℝ} (h : x > y) : heartsuit x y > 0 := by sorry

end NUMINAMATH_CALUDE_heartsuit_zero_heartsuit_self_heartsuit_positive_l2068_206810


namespace NUMINAMATH_CALUDE_min_square_value_l2068_206882

theorem min_square_value (a b : ℕ+) 
  (h1 : ∃ x : ℕ+, (15 * a.val + 16 * b.val : ℕ) = x * x)
  (h2 : ∃ y : ℕ+, (16 * a.val - 15 * b.val : ℕ) = y * y) :
  min (15 * a.val + 16 * b.val) (16 * a.val - 15 * b.val) ≥ 231361 :=
by sorry

end NUMINAMATH_CALUDE_min_square_value_l2068_206882


namespace NUMINAMATH_CALUDE_red_balloons_count_l2068_206821

def total_balloons : ℕ := 17
def green_balloons : ℕ := 9

theorem red_balloons_count :
  total_balloons - green_balloons = 8 := by
  sorry

end NUMINAMATH_CALUDE_red_balloons_count_l2068_206821


namespace NUMINAMATH_CALUDE_art_museum_survey_l2068_206892

theorem art_museum_survey (V : ℕ) (E U : ℕ) : 
  E = U →                                     -- Number who enjoyed equals number who understood
  (3 : ℚ) / 4 * V = E →                       -- 3/4 of visitors both enjoyed and understood
  V = 520 →                                   -- Total number of visitors
  V - E = 130                                 -- Number who didn't enjoy and didn't understand
  := by sorry

end NUMINAMATH_CALUDE_art_museum_survey_l2068_206892


namespace NUMINAMATH_CALUDE_smallest_number_remainder_l2068_206838

theorem smallest_number_remainder (n : ℕ) : 
  (n = 210) → 
  (n % 13 = 3) → 
  (∀ m : ℕ, m < n → m % 13 ≠ 3 ∨ m % 17 ≠ n % 17) → 
  n % 17 = 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_remainder_l2068_206838


namespace NUMINAMATH_CALUDE_extremum_point_of_f_l2068_206835

def f (x : ℝ) := x^2 - 2*x

theorem extremum_point_of_f :
  ∃ (c : ℝ), c = 1 ∧ ∀ (x : ℝ), f x ≤ f c ∨ f x ≥ f c :=
sorry

end NUMINAMATH_CALUDE_extremum_point_of_f_l2068_206835


namespace NUMINAMATH_CALUDE_fourth_fifth_supplier_cars_l2068_206837

/-- The number of cars each of the fourth and fifth suppliers receive -/
def cars_per_last_supplier (total_cars : ℕ) (first_supplier : ℕ) (additional_second : ℕ) : ℕ :=
  let second_supplier := first_supplier + additional_second
  let third_supplier := first_supplier + second_supplier
  let remaining_cars := total_cars - (first_supplier + second_supplier + third_supplier)
  remaining_cars / 2

/-- Proof that given the conditions, the fourth and fifth suppliers each receive 325,000 cars -/
theorem fourth_fifth_supplier_cars :
  cars_per_last_supplier 5650000 1000000 500000 = 325000 := by
  sorry

end NUMINAMATH_CALUDE_fourth_fifth_supplier_cars_l2068_206837


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2068_206834

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2068_206834


namespace NUMINAMATH_CALUDE_scooter_price_l2068_206804

theorem scooter_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 240 → 
  upfront_percentage = 0.20 → 
  upfront_payment = upfront_percentage * total_price → 
  total_price = 1200 := by
sorry

end NUMINAMATH_CALUDE_scooter_price_l2068_206804


namespace NUMINAMATH_CALUDE_range_of_a_minus_abs_b_l2068_206840

theorem range_of_a_minus_abs_b (a b : ℝ) (ha : 1 < a ∧ a < 3) (hb : -4 < b ∧ b < 2) :
  -3 < a - |b| ∧ a - |b| < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_minus_abs_b_l2068_206840


namespace NUMINAMATH_CALUDE_prob_sum_div_3_correct_l2068_206848

/-- The probability of the sum of three fair six-sided dice being divisible by 3 -/
def prob_sum_div_3 : ℚ :=
  13 / 27

/-- The set of possible outcomes for a single die roll -/
def die_outcomes : Finset ℕ :=
  Finset.range 6

/-- The probability of rolling any specific number on a fair six-sided die -/
def single_roll_prob : ℚ :=
  1 / 6

/-- The set of all possible outcomes when rolling three dice -/
def all_outcomes : Finset (ℕ × ℕ × ℕ) :=
  die_outcomes.product (die_outcomes.product die_outcomes)

/-- The sum of the numbers shown on three dice -/
def sum_of_dice (roll : ℕ × ℕ × ℕ) : ℕ :=
  roll.1 + roll.2.1 + roll.2.2 + 3

/-- The set of favorable outcomes (sum divisible by 3) -/
def favorable_outcomes : Finset (ℕ × ℕ × ℕ) :=
  all_outcomes.filter (λ roll ↦ (sum_of_dice roll) % 3 = 0)

theorem prob_sum_div_3_correct :
  (favorable_outcomes.card : ℚ) / all_outcomes.card = prob_sum_div_3 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_div_3_correct_l2068_206848


namespace NUMINAMATH_CALUDE_hyperbola_equation_proof_l2068_206831

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  /-- The eccentricity of the hyperbola -/
  e : ℝ
  /-- The distance from the focus to the asymptote -/
  d : ℝ
  /-- The hyperbola is centered at the origin -/
  center_origin : True
  /-- The foci are on the x-axis -/
  foci_on_x_axis : True
  /-- The eccentricity is √6/2 -/
  e_value : e = Real.sqrt 6 / 2
  /-- The distance from the focus to the asymptote is 1 -/
  d_value : d = 1

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 = 1

/-- Theorem stating that the given hyperbola has the specified equation -/
theorem hyperbola_equation_proof (h : Hyperbola) :
  ∀ x y : ℝ, hyperbola_equation h x y ↔ x^2 / 2 - y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_proof_l2068_206831


namespace NUMINAMATH_CALUDE_remainder_of_prime_powers_l2068_206801

theorem remainder_of_prime_powers (p q : Nat) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) :
  (p^(q - 1) + q^(p - 1)) % (p * q) = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_prime_powers_l2068_206801


namespace NUMINAMATH_CALUDE_rational_absolute_value_inequality_l2068_206879

theorem rational_absolute_value_inequality (a : ℚ) (h : a - |a| = 2*a) : a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_absolute_value_inequality_l2068_206879


namespace NUMINAMATH_CALUDE_sum_mod_thirteen_l2068_206858

theorem sum_mod_thirteen : (1230 + 1231 + 1232 + 1233 + 1234) % 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_thirteen_l2068_206858


namespace NUMINAMATH_CALUDE_dads_contribution_undetermined_l2068_206866

/-- Represents the number of toy cars in Olaf's collection --/
structure ToyCarCollection where
  initial : ℕ
  fromUncle : ℕ
  fromGrandpa : ℕ
  fromAuntie : ℕ
  fromMum : ℕ
  fromDad : ℕ
  final : ℕ

/-- The conditions of Olaf's toy car collection --/
def olafCollection : ToyCarCollection where
  initial := 150
  fromUncle := 5
  fromGrandpa := 10
  fromAuntie := 6
  fromMum := 0  -- Unknown value
  fromDad := 0  -- Unknown value
  final := 196

/-- Theorem stating that Dad's contribution is undetermined --/
theorem dads_contribution_undetermined (c : ToyCarCollection) 
  (h1 : c.initial = 150)
  (h2 : c.fromGrandpa = 2 * c.fromUncle)
  (h3 : c.fromAuntie = c.fromUncle + 1)
  (h4 : c.final = 196)
  (h5 : c.final = c.initial + c.fromUncle + c.fromGrandpa + c.fromAuntie + c.fromMum + c.fromDad) :
  ∃ (x y : ℕ), x ≠ y ∧ 
    (c.fromMum = x ∧ c.fromDad = 25 - x) ∧
    (c.fromMum = y ∧ c.fromDad = 25 - y) :=
sorry

#check dads_contribution_undetermined

end NUMINAMATH_CALUDE_dads_contribution_undetermined_l2068_206866


namespace NUMINAMATH_CALUDE_f_minus_g_at_one_l2068_206864

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)
def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h x = -h (-x)

-- State the theorem
theorem f_minus_g_at_one
  (h1 : is_even f)
  (h2 : is_odd g)
  (h3 : ∀ x, f x + g x = x^3 + x^2 + 1) :
  f 1 - g 1 = 1 := by sorry

end NUMINAMATH_CALUDE_f_minus_g_at_one_l2068_206864


namespace NUMINAMATH_CALUDE_circle_center_in_second_quadrant_l2068_206859

/-- A line passing through the second, third, and fourth quadrants -/
structure Line where
  a : ℝ
  b : ℝ
  second_quadrant : a < 0 ∧ 0 < a * 0 - b
  third_quadrant : a * (-1) - b < 0
  fourth_quadrant : 0 < a * 1 - b

/-- The center of a circle (x-a)^2 + (y-b)^2 = 1 -/
def circle_center (l : Line) : ℝ × ℝ := (l.a, l.b)

/-- A point is in the second quadrant if its x-coordinate is negative and y-coordinate is positive -/
def in_second_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ 0 < p.2

theorem circle_center_in_second_quadrant (l : Line) :
  in_second_quadrant (circle_center l) := by sorry

end NUMINAMATH_CALUDE_circle_center_in_second_quadrant_l2068_206859


namespace NUMINAMATH_CALUDE_time_after_2023_hours_l2068_206807

def clock_add (current_time : ℕ) (hours_passed : ℕ) : ℕ :=
  (current_time + hours_passed) % 12

theorem time_after_2023_hours :
  clock_add 7 2023 = 2 := by
  sorry

end NUMINAMATH_CALUDE_time_after_2023_hours_l2068_206807


namespace NUMINAMATH_CALUDE_sphere_surface_area_with_holes_value_l2068_206846

/-- The surface area of a sphere with diameter 10 inches, after drilling three holes each with a radius of 0.5 inches -/
def sphere_surface_area_with_holes : ℝ := sorry

/-- The diameter of the bowling ball in inches -/
def ball_diameter : ℝ := 10

/-- The number of finger holes -/
def num_holes : ℕ := 3

/-- The radius of each finger hole in inches -/
def hole_radius : ℝ := 0.5

theorem sphere_surface_area_with_holes_value :
  sphere_surface_area_with_holes = (197 / 2) * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_with_holes_value_l2068_206846


namespace NUMINAMATH_CALUDE_exists_non_negative_sums_l2068_206850

/-- Represents a sign change operation on a matrix -/
inductive SignChange
| Row (i : Nat)
| Col (j : Nat)

/-- Apply a sequence of sign changes to a matrix -/
def applySignChanges (A : Matrix (Fin m) (Fin n) ℝ) (changes : List SignChange) : Matrix (Fin m) (Fin n) ℝ :=
  sorry

/-- Check if all row and column sums are non-negative -/
def allSumsNonNegative (A : Matrix (Fin m) (Fin n) ℝ) : Prop :=
  sorry

/-- Main theorem: For any real matrix, there exists a sequence of sign changes
    that results in all row and column sums being non-negative -/
theorem exists_non_negative_sums (A : Matrix (Fin m) (Fin n) ℝ) :
  ∃ (changes : List SignChange), allSumsNonNegative (applySignChanges A changes) :=
  sorry

end NUMINAMATH_CALUDE_exists_non_negative_sums_l2068_206850


namespace NUMINAMATH_CALUDE_line_with_45_degree_slope_l2068_206890

/-- Given a line passing through points (1, -2) and (a, 3) with a slope angle of 45°, 
    the value of a is 6. -/
theorem line_with_45_degree_slope (a : ℝ) : 
  (((3 - (-2)) / (a - 1) = Real.tan (π / 4)) → a = 6) :=
by sorry

end NUMINAMATH_CALUDE_line_with_45_degree_slope_l2068_206890


namespace NUMINAMATH_CALUDE_remaining_money_for_gas_and_maintenance_l2068_206847

def monthly_income : ℕ := 3200

def rent : ℕ := 1250
def utilities : ℕ := 150
def retirement_savings : ℕ := 400
def groceries : ℕ := 300
def insurance : ℕ := 200
def miscellaneous : ℕ := 200
def car_payment : ℕ := 350

def total_expenses : ℕ := rent + utilities + retirement_savings + groceries + insurance + miscellaneous + car_payment

theorem remaining_money_for_gas_and_maintenance :
  monthly_income - total_expenses = 350 := by sorry

end NUMINAMATH_CALUDE_remaining_money_for_gas_and_maintenance_l2068_206847


namespace NUMINAMATH_CALUDE_role_assignment_count_l2068_206855

/-- The number of ways to assign roles in a play. -/
def assign_roles (num_men : ℕ) (num_women : ℕ) (male_roles : ℕ) (female_roles : ℕ) (either_roles : ℕ) : ℕ :=
  (num_men.descFactorial male_roles) *
  (num_women.descFactorial female_roles) *
  ((num_men + num_women - male_roles - female_roles).descFactorial either_roles)

/-- Theorem stating the number of ways to assign roles in the given scenario. -/
theorem role_assignment_count :
  assign_roles 7 8 3 3 4 = 213955200 :=
by sorry

end NUMINAMATH_CALUDE_role_assignment_count_l2068_206855


namespace NUMINAMATH_CALUDE_gcd_7654321_6543210_l2068_206867

theorem gcd_7654321_6543210 : Nat.gcd 7654321 6543210 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7654321_6543210_l2068_206867


namespace NUMINAMATH_CALUDE_code_cracking_probabilities_l2068_206823

/-- The probability of person i cracking the code -/
def P (i : Fin 3) : ℚ :=
  match i with
  | 0 => 1/5
  | 1 => 1/4
  | 2 => 1/3

/-- The probability that exactly two people crack the code -/
def prob_two_crack : ℚ :=
  P 0 * P 1 * (1 - P 2) + P 0 * (1 - P 1) * P 2 + (1 - P 0) * P 1 * P 2

/-- The probability that no one cracks the code -/
def prob_none_crack : ℚ :=
  (1 - P 0) * (1 - P 1) * (1 - P 2)

theorem code_cracking_probabilities :
  prob_two_crack = 3/20 ∧ 
  (1 - prob_none_crack) > prob_none_crack := by
  sorry


end NUMINAMATH_CALUDE_code_cracking_probabilities_l2068_206823


namespace NUMINAMATH_CALUDE_cube_root_function_l2068_206893

theorem cube_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * (27 : ℝ)^(1/3) ∧ y = 3 * Real.sqrt 3) →
  k * (8 : ℝ)^(1/3) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_function_l2068_206893


namespace NUMINAMATH_CALUDE_second_month_sale_l2068_206875

theorem second_month_sale 
  (first_month : ℕ) 
  (third_month : ℕ) 
  (fourth_month : ℕ) 
  (fifth_month : ℕ) 
  (sixth_month : ℕ) 
  (average_sale : ℕ) 
  (h1 : first_month = 5435)
  (h2 : third_month = 5855)
  (h3 : fourth_month = 6230)
  (h4 : fifth_month = 5562)
  (h5 : sixth_month = 3991)
  (h6 : average_sale = 5500) :
  ∃ (second_month : ℕ), 
    (first_month + second_month + third_month + fourth_month + fifth_month + sixth_month) / 6 = average_sale ∧ 
    second_month = 5927 :=
by sorry

end NUMINAMATH_CALUDE_second_month_sale_l2068_206875


namespace NUMINAMATH_CALUDE_prob_boy_girl_twins_l2068_206871

/-- The probability of twins being born -/
def prob_twins : ℚ := 3 / 250

/-- The probability of twins being identical, given that they are twins -/
def prob_identical_given_twins : ℚ := 1 / 3

/-- The probability of twins being fraternal, given that they are twins -/
def prob_fraternal_given_twins : ℚ := 1 - prob_identical_given_twins

/-- The probability of fraternal twins being a boy and a girl -/
def prob_boy_girl_given_fraternal : ℚ := 1 / 2

/-- The theorem stating the probability of a pregnant woman giving birth to boy-girl twins -/
theorem prob_boy_girl_twins : 
  prob_twins * prob_fraternal_given_twins * prob_boy_girl_given_fraternal = 1 / 250 := by
  sorry

end NUMINAMATH_CALUDE_prob_boy_girl_twins_l2068_206871
