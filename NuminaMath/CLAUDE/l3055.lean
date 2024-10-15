import Mathlib

namespace NUMINAMATH_CALUDE_reciprocal_product_l3055_305521

theorem reciprocal_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 8 * x * y) :
  (1 / x) * (1 / y) = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_product_l3055_305521


namespace NUMINAMATH_CALUDE_anne_wandering_l3055_305537

/-- Anne's wandering problem -/
theorem anne_wandering (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 2.0 → time = 1.5 → distance = speed * time → distance = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_anne_wandering_l3055_305537


namespace NUMINAMATH_CALUDE_megan_initial_files_l3055_305549

-- Define the problem parameters
def added_files : ℝ := 21.0
def files_per_folder : ℝ := 8.0
def final_folders : ℝ := 14.25

-- Define the theorem
theorem megan_initial_files :
  ∃ (initial_files : ℝ),
    (initial_files + added_files) / files_per_folder = final_folders ∧
    initial_files = 93.0 := by
  sorry

end NUMINAMATH_CALUDE_megan_initial_files_l3055_305549


namespace NUMINAMATH_CALUDE_bike_truck_travel_time_indeterminate_equal_travel_time_l3055_305539

/-- Given a bike and a truck with the same speed covering the same distance,
    prove that their travel times are equal but indeterminate without knowing the speed. -/
theorem bike_truck_travel_time (distance : ℝ) (speed : ℝ) : 
  distance > 0 → speed > 0 → ∃ (time : ℝ), 
    time = distance / speed ∧ 
    (∀ (bike_time truck_time : ℝ), 
      bike_time = distance / speed → 
      truck_time = distance / speed → 
      bike_time = truck_time) :=
by sorry

/-- The specific distance covered by both vehicles -/
def covered_distance : ℝ := 72

/-- The speed difference between the bike and the truck -/
def speed_difference : ℝ := 0

/-- Theorem stating that the travel time for both vehicles is the same 
    but cannot be determined without knowing the speed -/
theorem indeterminate_equal_travel_time :
  ∃ (time : ℝ), 
    (∀ (bike_speed : ℝ), bike_speed > 0 →
      time = covered_distance / bike_speed) ∧
    (∀ (truck_speed : ℝ), truck_speed > 0 →
      time = covered_distance / truck_speed) ∧
    (∀ (bike_time truck_time : ℝ),
      bike_time = covered_distance / bike_speed →
      truck_time = covered_distance / truck_speed →
      bike_time = truck_time) :=
by sorry

end NUMINAMATH_CALUDE_bike_truck_travel_time_indeterminate_equal_travel_time_l3055_305539


namespace NUMINAMATH_CALUDE_discriminant_of_5x2_plus_3x_minus_8_l3055_305550

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 5x^2 + 3x - 8 is 169 -/
theorem discriminant_of_5x2_plus_3x_minus_8 :
  discriminant 5 3 (-8) = 169 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_5x2_plus_3x_minus_8_l3055_305550


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l3055_305517

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem smallest_two_digit_prime_with_composite_reverse :
  ∃ (n : ℕ), is_two_digit n ∧ 
             Nat.Prime n ∧ 
             tens_digit n = 2 ∧ 
             ¬(Nat.Prime (reverse_digits n)) ∧
             (∀ m : ℕ, is_two_digit m → 
                       Nat.Prime m → 
                       tens_digit m = 2 → 
                       ¬(Nat.Prime (reverse_digits m)) → 
                       n ≤ m) ∧
             n = 23 := by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l3055_305517


namespace NUMINAMATH_CALUDE_least_likely_score_l3055_305540

def class_size : ℕ := 50
def average_score : ℝ := 82
def score_variance : ℝ := 8.2

def score_options : List ℝ := [60, 70, 80, 100]

def distance_from_mean (score : ℝ) : ℝ :=
  |score - average_score|

theorem least_likely_score :
  ∃ (score : ℝ), score ∈ score_options ∧
    ∀ (other : ℝ), other ∈ score_options → other ≠ score →
      distance_from_mean score > distance_from_mean other :=
by sorry

end NUMINAMATH_CALUDE_least_likely_score_l3055_305540


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3055_305588

/-- Given a geometric sequence of positive numbers where the fifth term is 32 and the eleventh term is 2, the seventh term is 8. -/
theorem geometric_sequence_seventh_term (a : ℝ) (r : ℝ) (h1 : a * r^4 = 32) (h2 : a * r^10 = 2) :
  a * r^6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3055_305588


namespace NUMINAMATH_CALUDE_toys_donation_problem_l3055_305585

theorem toys_donation_problem (leila_bags : ℕ) (leila_toys_per_bag : ℕ) 
  (mohamed_bags : ℕ) (extra_toys : ℕ) :
  leila_bags = 2 →
  leila_toys_per_bag = 25 →
  mohamed_bags = 3 →
  extra_toys = 7 →
  (mohamed_bags * ((leila_bags * leila_toys_per_bag + extra_toys) / mohamed_bags) = 
   leila_bags * leila_toys_per_bag + extra_toys) ∧
  ((leila_bags * leila_toys_per_bag + extra_toys) / mohamed_bags = 19) :=
by sorry

end NUMINAMATH_CALUDE_toys_donation_problem_l3055_305585


namespace NUMINAMATH_CALUDE_fish_tagging_ratio_l3055_305599

theorem fish_tagging_ratio : 
  ∀ (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) (total_fish : ℕ),
  initial_tagged = 80 →
  second_catch = 80 →
  tagged_in_second = 2 →
  total_fish = 3200 →
  (tagged_in_second : ℚ) / second_catch = 1 / 40 := by
sorry

end NUMINAMATH_CALUDE_fish_tagging_ratio_l3055_305599


namespace NUMINAMATH_CALUDE_no_real_solutions_l3055_305560

theorem no_real_solutions : ¬∃ x : ℝ, |x| - 4 = (3 * |x|) / 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3055_305560


namespace NUMINAMATH_CALUDE_extreme_value_cubic_l3055_305579

/-- Given a cubic function f(x) = x^3 + ax^2 + bx + a^2 with an extreme value of 10 at x = 1,
    prove that f(2) = 18. -/
theorem extreme_value_cubic (a b : ℝ) :
  let f : ℝ → ℝ := λ x => x^3 + a*x^2 + b*x + a^2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1) ∧
  f 1 = 10 →
  f 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_cubic_l3055_305579


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l3055_305570

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (invalid_percentage : ℚ) 
  (candidate_valid_votes : ℕ) 
  (h1 : total_votes = 560000) 
  (h2 : invalid_percentage = 15 / 100) 
  (h3 : candidate_valid_votes = 333200) : 
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 70 / 100 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l3055_305570


namespace NUMINAMATH_CALUDE_right_triangle_30_60_90_l3055_305547

theorem right_triangle_30_60_90 (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_angle : a / c = 1 / 2) (h_hypotenuse : c = 10) : 
  a = 5 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_30_60_90_l3055_305547


namespace NUMINAMATH_CALUDE_jane_calculation_l3055_305551

theorem jane_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 17) 
  (h2 : x - y - z = 5) : 
  x - y = 11 := by
sorry

end NUMINAMATH_CALUDE_jane_calculation_l3055_305551


namespace NUMINAMATH_CALUDE_ball_count_and_probability_l3055_305518

/-- Represents the colors of the balls -/
inductive Color
  | Red
  | White
  | Blue

/-- Represents the bag of balls -/
structure Bag where
  total : Nat
  red : Nat
  white : Nat
  blue : Nat

/-- Represents the second bag with specific balls -/
structure SpecificBag where
  red1 : Bool
  white1 : Bool
  blue2 : Bool
  blue3 : Bool

def Bag.probability (b : Bag) (c : Color) : Rat :=
  match c with
  | Color.Red => b.red / b.total
  | Color.White => b.white / b.total
  | Color.Blue => b.blue / b.total

theorem ball_count_and_probability (b : Bag) :
  b.total = 24 ∧ b.blue = 3 ∧ b.probability Color.Red = 1/6 →
  b.red = 4 ∧
  (let sb : SpecificBag := ⟨true, true, true, true⟩
   (5 : Rat) / 12 = (Nat.choose 3 1 * Nat.choose 1 1) / (Nat.choose 4 2)) := by
  sorry


end NUMINAMATH_CALUDE_ball_count_and_probability_l3055_305518


namespace NUMINAMATH_CALUDE_divisibility_theorem_l3055_305553

theorem divisibility_theorem (a b c : ℕ) (h1 : a ∣ b * c) (h2 : Nat.gcd a b = 1) : a ∣ c := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l3055_305553


namespace NUMINAMATH_CALUDE_x_plus_q_equals_five_plus_two_q_l3055_305593

theorem x_plus_q_equals_five_plus_two_q (x q : ℝ) 
  (h1 : |x - 5| = q) 
  (h2 : x > 5) : 
  x + q = 5 + 2*q := by
sorry

end NUMINAMATH_CALUDE_x_plus_q_equals_five_plus_two_q_l3055_305593


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l3055_305581

theorem complex_fraction_equals_i : (1 + Complex.I) / (1 - Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l3055_305581


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l3055_305526

/-- Prove that given vectors a, b, u, and v with specific conditions, x = 1/2 --/
theorem vector_parallel_condition (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 1]
  let u : Fin 2 → ℝ := a + 2 • b
  let v : Fin 2 → ℝ := 2 • a - b
  (∃ (k : ℝ), u = k • v) → x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l3055_305526


namespace NUMINAMATH_CALUDE_roller_coaster_theorem_l3055_305500

/-- The number of different combinations for two rides with 7 people,
    where each ride accommodates 4 people and no person rides more than once. -/
def roller_coaster_combinations : ℕ := 525

/-- The total number of people in the group. -/
def total_people : ℕ := 7

/-- The number of people that can fit in a car for each ride. -/
def people_per_ride : ℕ := 4

/-- The number of rides. -/
def number_of_rides : ℕ := 2

theorem roller_coaster_theorem :
  roller_coaster_combinations =
    (Nat.choose total_people people_per_ride) *
    (Nat.choose (total_people - 1) people_per_ride) :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_theorem_l3055_305500


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3055_305559

/-- A complex number z is in the first quadrant if its real part is positive and its imaginary part is positive. -/
def in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

/-- Given a real number a, construct the complex number z = (3i - a) / i -/
def z (a : ℝ) : ℂ :=
  Complex.I * 3 - a

/-- The condition a > -1 is sufficient but not necessary for z(a) to be in the first quadrant -/
theorem sufficient_not_necessary (a : ℝ) :
  (∃ a₁ : ℝ, a₁ > -1 ∧ in_first_quadrant (z a₁)) ∧
  (∃ a₂ : ℝ, in_first_quadrant (z a₂) ∧ ¬(a₂ > -1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3055_305559


namespace NUMINAMATH_CALUDE_max_shot_radius_l3055_305594

/-- Given a sphere of radius 3 cm from which 27 shots can be made, 
    prove that the maximum radius of each shot is 1 cm. -/
theorem max_shot_radius (R : ℝ) (n : ℕ) (r : ℝ) : 
  R = 3 → n = 27 → (4 / 3 * Real.pi * R^3 = n * (4 / 3 * Real.pi * r^3)) → r ≤ 1 := by
  sorry

#check max_shot_radius

end NUMINAMATH_CALUDE_max_shot_radius_l3055_305594


namespace NUMINAMATH_CALUDE_N_div_15_is_square_N_div_10_is_cube_N_div_6_is_fifth_power_N_is_smallest_num_divisors_N_div_30_is_8400_l3055_305571

/-- The smallest positive integer N satisfying the given conditions -/
def N : ℕ := 2^16 * 3^21 * 5^25

/-- N/15 is a perfect square -/
theorem N_div_15_is_square : ∃ k : ℕ, N / 15 = k^2 := by sorry

/-- N/10 is a perfect cube -/
theorem N_div_10_is_cube : ∃ k : ℕ, N / 10 = k^3 := by sorry

/-- N/6 is a perfect fifth power -/
theorem N_div_6_is_fifth_power : ∃ k : ℕ, N / 6 = k^5 := by sorry

/-- N is the smallest positive integer satisfying the conditions -/
theorem N_is_smallest : ∀ m : ℕ, m < N → 
  (¬∃ k : ℕ, m / 15 = k^2) ∨ 
  (¬∃ k : ℕ, m / 10 = k^3) ∨ 
  (¬∃ k : ℕ, m / 6 = k^5) := by sorry

/-- The number of positive divisors of N/30 -/
def num_divisors_N_div_30 : ℕ := (15 + 1) * (20 + 1) * (24 + 1)

/-- Theorem: The number of positive divisors of N/30 is 8400 -/
theorem num_divisors_N_div_30_is_8400 : num_divisors_N_div_30 = 8400 := by sorry

end NUMINAMATH_CALUDE_N_div_15_is_square_N_div_10_is_cube_N_div_6_is_fifth_power_N_is_smallest_num_divisors_N_div_30_is_8400_l3055_305571


namespace NUMINAMATH_CALUDE_handshake_count_l3055_305534

theorem handshake_count (n : Nat) (h : n = 8) : 
  (n * (n - 1)) / 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l3055_305534


namespace NUMINAMATH_CALUDE_connie_marbles_l3055_305590

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℝ := 183.0

/-- The number of marbles Connie has left -/
def marbles_left : ℝ := 593.0

/-- The initial number of marbles Connie had -/
def initial_marbles : ℝ := marbles_given + marbles_left

theorem connie_marbles : initial_marbles = 776.0 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_l3055_305590


namespace NUMINAMATH_CALUDE_solve_bus_problem_l3055_305507

def bus_problem (first_stop : ℕ) (second_stop_off : ℕ) (second_stop_on : ℕ) (third_stop_off : ℕ) (final_count : ℕ) : Prop :=
  let after_first := first_stop
  let after_second := after_first - second_stop_off + second_stop_on
  let before_third_on := after_second - third_stop_off
  ∃ (third_stop_on : ℕ), before_third_on + third_stop_on = final_count ∧ third_stop_on = 4

theorem solve_bus_problem :
  bus_problem 7 3 5 2 11 :=
by
  sorry

#check solve_bus_problem

end NUMINAMATH_CALUDE_solve_bus_problem_l3055_305507


namespace NUMINAMATH_CALUDE_coin_toss_sequences_count_l3055_305575

/-- The number of ways to insert k items into n bins -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (n - 1)

/-- The number of different sequences of 17 coin tosses with specific subsequence counts -/
def coinTossSequences : ℕ := 
  let hh_insertions := starsAndBars 5 3  -- Insert 3 H into 5 existing H positions
  let tt_insertions := starsAndBars 4 6  -- Insert 6 T into 4 existing T positions
  hh_insertions * tt_insertions

/-- Theorem stating the number of coin toss sequences -/
theorem coin_toss_sequences_count :
  coinTossSequences = 2940 := by sorry

end NUMINAMATH_CALUDE_coin_toss_sequences_count_l3055_305575


namespace NUMINAMATH_CALUDE_concert_ticket_sales_l3055_305501

/-- Proves that given the conditions of the concert ticket sales, the number of back seat tickets sold is 14,500 --/
theorem concert_ticket_sales 
  (total_seats : ℕ) 
  (main_seat_price back_seat_price : ℚ) 
  (total_revenue : ℚ) 
  (h1 : total_seats = 20000)
  (h2 : main_seat_price = 55)
  (h3 : back_seat_price = 45)
  (h4 : total_revenue = 955000) :
  ∃ (main_seats back_seats : ℕ),
    main_seats + back_seats = total_seats ∧
    main_seat_price * main_seats + back_seat_price * back_seats = total_revenue ∧
    back_seats = 14500 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_sales_l3055_305501


namespace NUMINAMATH_CALUDE_blue_socks_cost_l3055_305576

/-- The cost of blue socks given the total cost, number of red and blue socks, and cost of red socks -/
def cost_of_blue_socks (total_cost : ℚ) (num_red : ℕ) (num_blue : ℕ) (cost_red : ℚ) : ℚ :=
  (total_cost - num_red * cost_red) / num_blue

/-- Theorem stating the cost of each pair of blue socks -/
theorem blue_socks_cost :
  cost_of_blue_socks 42 4 6 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_blue_socks_cost_l3055_305576


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l3055_305524

-- Define the function f
def f (x : ℝ) : ℝ := (x+3)*(x+2)*(x+1)*x*(x-1)*(x-2)*(x-3)

-- State the theorem
theorem f_derivative_at_one : 
  deriv f 1 = 48 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l3055_305524


namespace NUMINAMATH_CALUDE_volume_increase_l3055_305513

/-- Represents a rectangular solid -/
structure RectangularSolid where
  baseArea : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular solid -/
def volume (solid : RectangularSolid) : ℝ :=
  solid.baseArea * solid.height

/-- Theorem: Increase in volume of a rectangular solid -/
theorem volume_increase (solid : RectangularSolid) 
  (h1 : solid.baseArea = 12)
  (h2 : 5 > 0) :
  volume { baseArea := solid.baseArea, height := solid.height + 5 } - volume solid = 60 := by
  sorry

end NUMINAMATH_CALUDE_volume_increase_l3055_305513


namespace NUMINAMATH_CALUDE_beka_jackson_miles_difference_l3055_305525

/-- The difference in miles flown between Beka and Jackson -/
def miles_difference (beka_miles jackson_miles : ℕ) : ℕ :=
  beka_miles - jackson_miles

/-- Theorem stating the difference in miles flown between Beka and Jackson -/
theorem beka_jackson_miles_difference :
  miles_difference 873 563 = 310 :=
by sorry

end NUMINAMATH_CALUDE_beka_jackson_miles_difference_l3055_305525


namespace NUMINAMATH_CALUDE_fruit_basket_theorem_l3055_305597

/-- Calculates the number of possible fruit baskets given a number of apples and oranges. -/
def fruitBasketCount (apples : ℕ) (oranges : ℕ) : ℕ :=
  (apples + 1) * (oranges + 1) - 1

/-- Theorem stating that the number of fruit baskets with 4 apples and 8 oranges is 44. -/
theorem fruit_basket_theorem :
  fruitBasketCount 4 8 = 44 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_theorem_l3055_305597


namespace NUMINAMATH_CALUDE_power_function_uniqueness_l3055_305565

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

theorem power_function_uniqueness 
  (f : ℝ → ℝ) 
  (h1 : is_power_function f) 
  (h2 : f 27 = 3) : 
  ∀ x : ℝ, f x = x ^ (1/3) :=
sorry

end NUMINAMATH_CALUDE_power_function_uniqueness_l3055_305565


namespace NUMINAMATH_CALUDE_teacher_age_l3055_305502

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) :
  num_students = 23 →
  student_avg_age = 22 →
  new_avg_age = student_avg_age + 1 →
  (num_students : ℝ) * student_avg_age + (new_avg_age * (num_students + 1) - student_avg_age * num_students) = 46 * (num_students + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l3055_305502


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l3055_305586

/-- The perimeter of the shaded region formed by three identical touching circles --/
theorem shaded_region_perimeter (c : ℝ) (n : ℕ) (α : ℝ) : 
  c = 48 → n = 3 → α = 90 → c * (α / 360) * n = 36 := by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l3055_305586


namespace NUMINAMATH_CALUDE_cd_price_correct_l3055_305556

/-- The price of a CD in dollars -/
def price_cd : ℝ := 14

/-- The price of a cassette in dollars -/
def price_cassette : ℝ := 9

/-- The total amount Leanna has to spend in dollars -/
def total_money : ℝ := 37

/-- The amount left over when buying one CD and two cassettes in dollars -/
def money_left : ℝ := 5

theorem cd_price_correct : 
  (2 * price_cd + price_cassette = total_money) ∧ 
  (price_cd + 2 * price_cassette = total_money - money_left) :=
by sorry

end NUMINAMATH_CALUDE_cd_price_correct_l3055_305556


namespace NUMINAMATH_CALUDE_circle_radius_from_spherical_coordinates_l3055_305561

theorem circle_radius_from_spherical_coordinates : 
  let r : ℝ := Real.sqrt 3 / 2
  ∀ θ : ℝ, 
    let x : ℝ := Real.sin (π/3) * Real.cos θ
    let y : ℝ := Real.sin (π/3) * Real.sin θ
    Real.sqrt (x^2 + y^2) = r := by sorry

end NUMINAMATH_CALUDE_circle_radius_from_spherical_coordinates_l3055_305561


namespace NUMINAMATH_CALUDE_q_value_l3055_305523

theorem q_value (t R m q : ℝ) (h : R = t / ((2 + m) ^ q)) :
  q = Real.log (t / R) / Real.log (2 + m) := by
  sorry

end NUMINAMATH_CALUDE_q_value_l3055_305523


namespace NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l3055_305512

theorem sum_of_special_primes_is_prime (P Q : ℕ) : 
  P > 0 ∧ Q > 0 ∧ 
  Nat.Prime P ∧ Nat.Prime Q ∧ Nat.Prime (P - Q) ∧ Nat.Prime (P + Q) →
  Nat.Prime (P + Q + (P - Q) + P + Q) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l3055_305512


namespace NUMINAMATH_CALUDE_product_xyz_l3055_305548

theorem product_xyz (x y z : ℝ) 
  (sphere_eq : (x - 2)^2 + (y - 3)^2 + (z - 4)^2 = 9)
  (plane_eq : x + y + z = 12) :
  x * y * z = 42 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_l3055_305548


namespace NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l3055_305569

-- Define the custom operation ⊗
def otimes (a b : ℝ) : ℝ := a^2 - abs b

-- Theorem statement
theorem otimes_neg_two_neg_one : otimes (-2) (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l3055_305569


namespace NUMINAMATH_CALUDE_starting_lineup_count_l3055_305528

def team_size : ℕ := 15
def lineup_size : ℕ := 5
def special_players : ℕ := 3

theorem starting_lineup_count : 
  (lineup_size.choose (team_size - special_players)) + 
  (special_players * (lineup_size - 1).choose (team_size - special_players)) = 2277 :=
by sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l3055_305528


namespace NUMINAMATH_CALUDE_lasso_probability_l3055_305519

theorem lasso_probability (p : ℝ) (n : ℕ) (hp : p = 1 / 2) (hn : n = 4) :
  1 - (1 - p) ^ n = 15 / 16 :=
by sorry

end NUMINAMATH_CALUDE_lasso_probability_l3055_305519


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_l3055_305509

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_sum_of_powers (h : i^2 = -1) : (1 + i)^30 + (1 - i)^30 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_l3055_305509


namespace NUMINAMATH_CALUDE_hyperbola_foci_l3055_305504

theorem hyperbola_foci (x y : ℝ) :
  (x^2 / 3 - y^2 / 4 = 1) →
  (∃ f : ℝ, f = Real.sqrt 7 ∧ 
    ((x = f ∧ y = 0) ∨ (x = -f ∧ y = 0)) →
    (x^2 / 3 - y^2 / 4 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l3055_305504


namespace NUMINAMATH_CALUDE_positive_x_solution_l3055_305530

theorem positive_x_solution (x y z : ℝ) 
  (eq1 : x * y = 8 - 3 * x - 2 * y)
  (eq2 : y * z = 8 - 3 * y - 3 * z)
  (eq3 : x * z = 40 - 5 * x - 4 * z)
  (x_pos : x > 0) :
  x = Real.sqrt (14 * 17 * 60) / 17 - 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_x_solution_l3055_305530


namespace NUMINAMATH_CALUDE_set_properties_l3055_305584

-- Define set A
def A : Set Int := {x | ∃ m n : Int, x = m^2 - n^2}

-- Define set B
def B : Set Int := {x | ∃ k : Int, x = 2*k + 1}

-- Theorem statement
theorem set_properties :
  (8 ∈ A ∧ 9 ∈ A ∧ 10 ∉ A) ∧
  (∀ x, x ∈ B → x ∈ A) ∧
  (∃ x, x ∈ A ∧ x ∉ B) ∧
  (∀ x, x ∈ A ∧ Even x ↔ ∃ k : Int, x = 4*k) :=
by sorry

end NUMINAMATH_CALUDE_set_properties_l3055_305584


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l3055_305544

/-- Two triangles are similar -/
structure SimilarTriangles (Triangle1 Triangle2 : Type) :=
  (similar : Triangle1 → Triangle2 → Prop)

/-- Definition of a triangle with side lengths -/
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

/-- Theorem: If triangles DEF and ABC are similar, and DE = 6, EF = 12, BC = 18, then AB = 9 -/
theorem similar_triangles_side_length 
  (DEF ABC : Triangle)
  (h_similar : SimilarTriangles Triangle Triangle)
  (h_similar_triangles : h_similar.similar DEF ABC)
  (h_DE : DEF.side1 = 6)
  (h_EF : DEF.side2 = 12)
  (h_BC : ABC.side2 = 18) :
  ABC.side1 = 9 :=
sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l3055_305544


namespace NUMINAMATH_CALUDE_count_valid_license_plates_l3055_305578

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of possible digits -/
def digit_range : ℕ := 10

/-- The number of letter positions in a license plate -/
def letter_positions : ℕ := 3

/-- The number of digit positions in a license plate -/
def digit_positions : ℕ := 2

/-- Calculates the total number of valid license plates -/
def valid_license_plates : ℕ := alphabet_size ^ letter_positions * digit_range ^ digit_positions

theorem count_valid_license_plates : valid_license_plates = 1757600 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_license_plates_l3055_305578


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3055_305508

theorem smallest_integer_satisfying_inequality :
  ∃ x : ℤ, (∀ y : ℤ, 8 - 7 * y ≥ 4 * y - 3 → x ≤ y) ∧ (8 - 7 * x ≥ 4 * x - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3055_305508


namespace NUMINAMATH_CALUDE_first_patient_therapy_hours_l3055_305574

/-- Represents the cost structure and patient charges for a psychologist's therapy sessions. -/
structure TherapyCost where
  first_hour : ℕ           -- Cost of the first hour
  additional_hour : ℕ      -- Cost of each additional hour
  first_patient_total : ℕ  -- Total charge for the first patient
  two_hour_total : ℕ       -- Total charge for a patient receiving 2 hours

/-- Calculates the number of therapy hours for the first patient given the cost structure. -/
def calculate_therapy_hours (cost : TherapyCost) : ℕ :=
  -- The implementation is not provided as per the instructions
  sorry

/-- Theorem stating that given the specific cost structure, the first patient received 5 hours of therapy. -/
theorem first_patient_therapy_hours 
  (cost : TherapyCost)
  (h1 : cost.first_hour = cost.additional_hour + 35)
  (h2 : cost.two_hour_total = 161)
  (h3 : cost.first_patient_total = 350) :
  calculate_therapy_hours cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_patient_therapy_hours_l3055_305574


namespace NUMINAMATH_CALUDE_floor_equation_solution_l3055_305505

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊2 * x⌋ - 1/2⌋ = ⌊x + 2⌋ ↔ 5/2 ≤ x ∧ x < 7/2 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l3055_305505


namespace NUMINAMATH_CALUDE_fraction_sum_l3055_305591

theorem fraction_sum (w x y : ℝ) (h1 : (w + x) / 2 = 0.5) (h2 : w * x = y) : 
  5 / w + 5 / x = 20 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_l3055_305591


namespace NUMINAMATH_CALUDE_monic_quadratic_polynomial_l3055_305598

theorem monic_quadratic_polynomial (f : ℝ → ℝ) :
  (∃ a b : ℝ, ∀ x, f x = x^2 + a*x + b) →  -- monic quadratic polynomial
  f 1 = 3 →                               -- f(1) = 3
  f 2 = 12 →                              -- f(2) = 12
  ∀ x, f x = x^2 + 6*x - 4 :=              -- f(x) = x^2 + 6x - 4
by sorry

end NUMINAMATH_CALUDE_monic_quadratic_polynomial_l3055_305598


namespace NUMINAMATH_CALUDE_egg_groups_l3055_305531

theorem egg_groups (total_eggs : ℕ) (eggs_per_group : ℕ) (h1 : total_eggs = 35) (h2 : eggs_per_group = 7) :
  total_eggs / eggs_per_group = 5 := by
  sorry

end NUMINAMATH_CALUDE_egg_groups_l3055_305531


namespace NUMINAMATH_CALUDE_slips_theorem_l3055_305555

/-- The number of slips in the bag -/
def total_slips : ℕ := 15

/-- The expected value of a randomly drawn slip -/
def expected_value : ℚ := 46/10

/-- The value on some of the slips -/
def value1 : ℕ := 3

/-- The value on the rest of the slips -/
def value2 : ℕ := 8

/-- The number of slips with value1 -/
def slips_with_value1 : ℕ := 10

theorem slips_theorem : 
  ∃ (x : ℕ), x = slips_with_value1 ∧ 
  x ≤ total_slips ∧
  (x : ℚ) / total_slips * value1 + (total_slips - x : ℚ) / total_slips * value2 = expected_value :=
sorry

end NUMINAMATH_CALUDE_slips_theorem_l3055_305555


namespace NUMINAMATH_CALUDE_max_fraction_value_l3055_305542

theorem max_fraction_value (a b c d : ℕ) 
  (ha : 0 < a) (hab : a < b) (hbc : b < c) (hcd : c < d) (hd : d < 10) :
  (∀ w x y z : ℕ, 0 < w → w < x → x < y → y < z → z < 10 → 
    (a - b : ℚ) / (c - d : ℚ) ≥ (w - x : ℚ) / (y - z : ℚ)) →
  (a - b : ℚ) / (c - d : ℚ) = -6 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_value_l3055_305542


namespace NUMINAMATH_CALUDE_tangent_line_equality_l3055_305527

theorem tangent_line_equality (x₁ x₂ y₁ y₂ : ℝ) :
  (∃ m b : ℝ, (∀ x : ℝ, y₁ + (Real.exp x₁) * (x - x₁) = m * x + b) ∧
              (∀ x : ℝ, y₂ + (1 / x₂) * (x - x₂) = m * x + b)) →
  (x₁ + 1) * (x₂ - 1) = -2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equality_l3055_305527


namespace NUMINAMATH_CALUDE_train_length_l3055_305582

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 15 → ∃ length : ℝ, 
  (abs (length - 250.05) < 0.01) ∧ (length = speed * 1000 / 3600 * time) :=
sorry

end NUMINAMATH_CALUDE_train_length_l3055_305582


namespace NUMINAMATH_CALUDE_contract_copies_per_person_l3055_305515

theorem contract_copies_per_person 
  (contract_pages : ℕ) 
  (total_pages : ℕ) 
  (num_people : ℕ) 
  (h1 : contract_pages = 20) 
  (h2 : total_pages = 360) 
  (h3 : num_people = 9) :
  (total_pages / contract_pages) / num_people = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_contract_copies_per_person_l3055_305515


namespace NUMINAMATH_CALUDE_sum_of_consecutive_odd_numbers_l3055_305583

theorem sum_of_consecutive_odd_numbers : 
  let odd_numbers := [997, 999, 1001, 1003, 1005]
  (List.sum odd_numbers) = 5100 - 95 := by
sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_odd_numbers_l3055_305583


namespace NUMINAMATH_CALUDE_product_greater_than_sum_minus_one_l3055_305573

theorem product_greater_than_sum_minus_one {a₁ a₂ : ℝ} 
  (h₁ : 0 < a₁) (h₂ : a₁ < 1) (h₃ : 0 < a₂) (h₄ : a₂ < 1) : 
  a₁ * a₂ > a₁ + a₂ - 1 := by
  sorry

end NUMINAMATH_CALUDE_product_greater_than_sum_minus_one_l3055_305573


namespace NUMINAMATH_CALUDE_min_product_of_three_numbers_l3055_305564

theorem min_product_of_three_numbers (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1)
  (z_eq_3x : z = 3 * x)
  (ordered : x ≤ y ∧ y ≤ z)
  (max_triple : z ≤ 3 * x) : 
  ∃ (min_prod : ℝ), min_prod = 9 / 343 ∧ x * y * z ≥ min_prod :=
sorry

end NUMINAMATH_CALUDE_min_product_of_three_numbers_l3055_305564


namespace NUMINAMATH_CALUDE_problem1_problem2_l3055_305554

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the axioms
axiom parallel_trans_LP {l1 l2 : Line} {p : Plane} :
  parallel l1 l2 → parallelLP l2 p → (parallelLP l1 p ∨ subset l1 p)

axiom parallel_trans_PP {p1 p2 p3 : Plane} :
  parallelPP p1 p2 → parallelPP p2 p3 → parallelPP p1 p3

axiom perpendicular_parallel {l : Line} {p1 p2 : Plane} :
  perpendicular l p1 → parallelPP p1 p2 → perpendicular l p2

-- State the theorems
theorem problem1 {m n : Line} {α : Plane} :
  parallel m n → parallelLP n α → (parallelLP m α ∨ subset m α) :=
by sorry

theorem problem2 {m : Line} {α β γ : Plane} :
  parallelPP α β → parallelPP β γ → perpendicular m α → perpendicular m γ :=
by sorry

end NUMINAMATH_CALUDE_problem1_problem2_l3055_305554


namespace NUMINAMATH_CALUDE_danny_found_seven_caps_l3055_305543

/-- The number of bottle caps Danny found at the park -/
def bottle_caps_found (initial : ℕ) (final : ℕ) : ℕ := final - initial

/-- Theorem: Danny found 7 bottle caps at the park -/
theorem danny_found_seven_caps : bottle_caps_found 25 32 = 7 := by
  sorry

end NUMINAMATH_CALUDE_danny_found_seven_caps_l3055_305543


namespace NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l3055_305552

-- Define the equations
def equation1 (x : ℝ) : Prop := 4 / (x - 6) = 3 / (x + 1)
def equation2 (x : ℝ) : Prop := (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1

-- Theorem for equation1
theorem equation1_solution :
  ∃! x : ℝ, equation1 x ∧ x = -22 :=
sorry

-- Theorem for equation2
theorem equation2_no_solution :
  ¬∃ x : ℝ, equation2 x :=
sorry

end NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l3055_305552


namespace NUMINAMATH_CALUDE_min_abs_z_given_constraint_l3055_305503

open Complex

theorem min_abs_z_given_constraint (z : ℂ) (h : abs (z - 2*I) = 1) : 
  abs z ≥ 1 ∧ ∃ w : ℂ, abs (w - 2*I) = 1 ∧ abs w = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_given_constraint_l3055_305503


namespace NUMINAMATH_CALUDE_fraction_problem_l3055_305529

theorem fraction_problem (N : ℝ) (F : ℝ) 
  (h1 : (1/3) * F * N = 18) 
  (h2 : (3/10) * N = 64.8) : 
  F = 1/4 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l3055_305529


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3055_305592

-- Define the selling price
def selling_price : ℚ := 715

-- Define the profit percentage
def profit_percentage : ℚ := 10 / 100

-- Define the cost price
def cost_price : ℚ := 650

-- Theorem to prove
theorem cost_price_calculation :
  cost_price = selling_price / (1 + profit_percentage) :=
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3055_305592


namespace NUMINAMATH_CALUDE_buttons_multiple_l3055_305580

theorem buttons_multiple (sue_buttons kendra_buttons mari_buttons : ℕ) 
  (h1 : sue_buttons = 6)
  (h2 : kendra_buttons = 2 * sue_buttons)
  (h3 : ∃ m : ℕ, mari_buttons = m * kendra_buttons + 4)
  (h4 : mari_buttons = 64) : 
  ∃ m : ℕ, mari_buttons = m * kendra_buttons + 4 ∧ m = 5 :=
by sorry

end NUMINAMATH_CALUDE_buttons_multiple_l3055_305580


namespace NUMINAMATH_CALUDE_international_long_haul_all_services_probability_l3055_305535

/-- Represents a flight route -/
inductive FlightRoute
| Domestic
| InternationalShortHaul
| InternationalLongHaul

/-- Represents a service offered on a flight -/
inductive Service
| WirelessInternet
| FreeSnacks
| EntertainmentSystem
| ExtraLegroom

/-- Returns the probability of a service being offered on a given flight route -/
def serviceProbability (route : FlightRoute) (service : Service) : ℝ :=
  match route, service with
  | FlightRoute.InternationalLongHaul, Service.WirelessInternet => 0.65
  | FlightRoute.InternationalLongHaul, Service.FreeSnacks => 0.80
  | FlightRoute.InternationalLongHaul, Service.EntertainmentSystem => 0.75
  | FlightRoute.InternationalLongHaul, Service.ExtraLegroom => 0.70
  | _, _ => 0  -- Default case, not used in this problem

/-- The probability of experiencing all services on a given flight route -/
def allServicesProbability (route : FlightRoute) : ℝ :=
  (serviceProbability route Service.WirelessInternet) *
  (serviceProbability route Service.FreeSnacks) *
  (serviceProbability route Service.EntertainmentSystem) *
  (serviceProbability route Service.ExtraLegroom)

/-- Theorem: The probability of experiencing all services on an international long-haul flight is 0.273 -/
theorem international_long_haul_all_services_probability :
  allServicesProbability FlightRoute.InternationalLongHaul = 0.273 := by
  sorry

end NUMINAMATH_CALUDE_international_long_haul_all_services_probability_l3055_305535


namespace NUMINAMATH_CALUDE_dodecagon_areas_l3055_305545

/-- A regular dodecagon with side length 1 cm -/
structure RegularDodecagon where
  side_length : ℝ
  is_one_cm : side_length = 1

/-- An equilateral triangle within the dodecagon -/
structure EquilateralTriangle where
  area : ℝ
  is_one_cm_squared : area = 1

/-- A square within the dodecagon -/
structure Square where
  side_length : ℝ
  is_one_cm : side_length = 1

/-- A regular hexagon within the dodecagon -/
structure RegularHexagon where
  side_length : ℝ
  is_one_cm : side_length = 1

/-- The decomposition of the dodecagon -/
structure DodecagonDecomposition where
  triangles : Finset EquilateralTriangle
  squares : Finset Square
  hexagon : RegularHexagon
  triangle_count : triangles.card = 6
  square_count : squares.card = 6

theorem dodecagon_areas 
  (d : RegularDodecagon) 
  (decomp : DodecagonDecomposition) : 
  /- 1. The area of the hexagon is 6 cm² -/
  decomp.hexagon.side_length ^ 2 * Real.sqrt 3 / 4 * 6 = 6 ∧ 
  /- 2. The area of the figure formed by removing 12 equilateral triangles is 6 cm² -/
  (d.side_length ^ 2 * Real.sqrt 3 / 4 * 12 + 6 * d.side_length ^ 2) - 
    (d.side_length ^ 2 * Real.sqrt 3 / 4 * 12) = 6 ∧
  /- 3. The area of the figure formed by removing 2 regular hexagons is 6 cm² -/
  (d.side_length ^ 2 * Real.sqrt 3 / 4 * 12 + 6 * d.side_length ^ 2) - 
    (2 * (decomp.hexagon.side_length ^ 2 * Real.sqrt 3 / 4 * 6)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_dodecagon_areas_l3055_305545


namespace NUMINAMATH_CALUDE_car_distance_l3055_305532

/-- The distance traveled by a car in 30 minutes, given that it travels at 2/3 the speed of a train moving at 90 miles per hour -/
theorem car_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (time : ℝ) : 
  train_speed = 90 →
  car_speed_ratio = 2 / 3 →
  time = 1 / 2 →
  car_speed_ratio * train_speed * time = 30 := by
sorry

end NUMINAMATH_CALUDE_car_distance_l3055_305532


namespace NUMINAMATH_CALUDE_equation_solution_l3055_305538

theorem equation_solution (M : ℚ) : (5 + 7 + 9) / 3 = (4020 + 4021 + 4022) / M → M = 1723 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3055_305538


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l3055_305506

theorem consecutive_integers_problem (x y z : ℤ) : 
  (y = z + 1) →
  (x = z + 2) →
  (x > y) →
  (y > z) →
  (2 * x + 3 * y + 3 * z = 5 * y + 8) →
  (z = 2) →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l3055_305506


namespace NUMINAMATH_CALUDE_circle_equation_l3055_305520

-- Define the point P
def P : ℝ × ℝ := (3, 1)

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x + 2 * y + 3 = 0
def l₂ (x y : ℝ) : Prop := x + 2 * y - 7 = 0

-- Define the two possible circle equations
def circle₁ (x y : ℝ) : Prop := (x - 4/5)^2 + (y - 3/5)^2 = 5
def circle₂ (x y : ℝ) : Prop := (x - 4)^2 + (y + 1)^2 = 5

-- Theorem statement
theorem circle_equation (x y : ℝ) :
  (∃ (c : ℝ × ℝ → Prop), c P ∧
    (∀ (x y : ℝ), l₁ x y → (∃ (t : ℝ), c (x, y) ∧ (∀ (ε : ℝ), ε ≠ 0 → ¬ c (x + ε, y + 2 * ε)))) ∧
    (∀ (x y : ℝ), l₂ x y → (∃ (t : ℝ), c (x, y) ∧ (∀ (ε : ℝ), ε ≠ 0 → ¬ c (x + ε, y + 2 * ε))))) →
  circle₁ x y ∨ circle₂ x y :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3055_305520


namespace NUMINAMATH_CALUDE_souvenirs_for_45_colleagues_l3055_305563

def souvenir_pattern : Nat → Nat
| 0 => 1
| 1 => 3
| 2 => 5
| 3 => 7
| n + 4 => souvenir_pattern n

def total_souvenirs (n : Nat) : Nat :=
  (List.range n).map souvenir_pattern |>.sum

theorem souvenirs_for_45_colleagues :
  total_souvenirs 45 = 177 := by
  sorry

end NUMINAMATH_CALUDE_souvenirs_for_45_colleagues_l3055_305563


namespace NUMINAMATH_CALUDE_inequality_proof_l3055_305557

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (7 * a^2 + b^2 + c^2)) + 
  (b / Real.sqrt (a^2 + 7 * b^2 + c^2)) + 
  (c / Real.sqrt (a^2 + b^2 + 7 * c^2)) ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3055_305557


namespace NUMINAMATH_CALUDE_taxi_speed_l3055_305568

/-- Given a taxi and a bus with specific conditions, proves that the taxi's speed is 60 mph --/
theorem taxi_speed (taxi_speed bus_speed : ℝ) : 
  (taxi_speed > 0) →  -- Ensure positive speed
  (bus_speed > 0) →   -- Ensure positive speed
  (bus_speed = taxi_speed - 30) →  -- Bus is 30 mph slower
  (3 * taxi_speed = 6 * bus_speed) →  -- Taxi covers in 3 hours what bus covers in 6
  (taxi_speed = 60) :=
by
  sorry

#check taxi_speed

end NUMINAMATH_CALUDE_taxi_speed_l3055_305568


namespace NUMINAMATH_CALUDE_sector_central_angle_l3055_305587

theorem sector_central_angle (area : Real) (perimeter : Real) (r : Real) (θ : Real) :
  area = 1 ∧ perimeter = 4 ∧ area = (1/2) * r^2 * θ ∧ perimeter = 2*r + r*θ → θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3055_305587


namespace NUMINAMATH_CALUDE_ball_probabilities_l3055_305596

def red_balls : ℕ := 5
def black_balls : ℕ := 7
def additional_balls : ℕ := 6

def probability_red : ℚ := red_balls / (red_balls + black_balls)
def probability_black : ℚ := black_balls / (red_balls + black_balls)

def new_red_balls : ℕ := red_balls + 4
def new_black_balls : ℕ := black_balls + 2

theorem ball_probabilities :
  (probability_black > probability_red) ∧
  (new_red_balls / (new_red_balls + new_black_balls) = new_black_balls / (new_red_balls + new_black_balls)) :=
by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l3055_305596


namespace NUMINAMATH_CALUDE_dhoni_toys_l3055_305595

theorem dhoni_toys (x : ℕ) (avg_cost : ℚ) (new_toy_cost : ℚ) (new_avg_cost : ℚ) : 
  avg_cost = 10 →
  new_toy_cost = 16 →
  new_avg_cost = 11 →
  (x * avg_cost + new_toy_cost) / (x + 1) = new_avg_cost →
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_dhoni_toys_l3055_305595


namespace NUMINAMATH_CALUDE_min_distance_sliding_ruler_l3055_305558

/-- The minimum distance between a point and the endpoint of a sliding ruler -/
theorem min_distance_sliding_ruler (h s : ℝ) (h_pos : h > 0) (s_pos : s > 0) (h_gt_s : h > s) :
  let min_distance := Real.sqrt (h^2 - s^2)
  ∀ (distance : ℝ), distance ≥ min_distance :=
sorry

end NUMINAMATH_CALUDE_min_distance_sliding_ruler_l3055_305558


namespace NUMINAMATH_CALUDE_expression_simplification_l3055_305562

theorem expression_simplification : (((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4)) = 12.75 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3055_305562


namespace NUMINAMATH_CALUDE_f_fixed_points_l3055_305546

def f (x : ℝ) : ℝ := x^3 - 3*x^2

theorem f_fixed_points : 
  ∃ (x : ℝ), (f (f x) = f x) ∧ (x = 0 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_f_fixed_points_l3055_305546


namespace NUMINAMATH_CALUDE_point_outside_circle_l3055_305516

theorem point_outside_circle (a b : ℝ) 
  (h : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*x + b*y = 1) : 
  a^2 + b^2 > 1 :=
sorry

end NUMINAMATH_CALUDE_point_outside_circle_l3055_305516


namespace NUMINAMATH_CALUDE_negation_of_existence_logarithm_l3055_305541

theorem negation_of_existence_logarithm (x : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ = x₀ - 1) ↔
  (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_logarithm_l3055_305541


namespace NUMINAMATH_CALUDE_student_selection_probability_l3055_305577

theorem student_selection_probability 
  (total_students : ℕ) 
  (selected_students : ℕ) 
  (excluded_students : ℕ) 
  (h1 : total_students = 2008) 
  (h2 : selected_students = 50) 
  (h3 : excluded_students = 8) :
  (selected_students : ℚ) / total_students = 25 / 1004 :=
sorry

end NUMINAMATH_CALUDE_student_selection_probability_l3055_305577


namespace NUMINAMATH_CALUDE_set_closure_under_difference_l3055_305567

theorem set_closure_under_difference (A : Set ℝ) 
  (h1 : ∀ a ∈ A, (2 * a) ∈ A) 
  (h2 : ∀ a b, a ∈ A → b ∈ A → (a + b) ∈ A) : 
  ∀ x y, x ∈ A → y ∈ A → (x - y) ∈ A := by
  sorry

end NUMINAMATH_CALUDE_set_closure_under_difference_l3055_305567


namespace NUMINAMATH_CALUDE_cube_sum_difference_l3055_305522

theorem cube_sum_difference (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 210 → a + b + c = 7 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_difference_l3055_305522


namespace NUMINAMATH_CALUDE_rectangle_inscribed_circle_circumference_l3055_305536

theorem rectangle_inscribed_circle_circumference 
  (width : ℝ) (height : ℝ) (circumference : ℝ) :
  width = 7 ∧ height = 24 →
  circumference = Real.pi * Real.sqrt (width^2 + height^2) →
  circumference = 25 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_rectangle_inscribed_circle_circumference_l3055_305536


namespace NUMINAMATH_CALUDE_midpoint_distance_range_l3055_305589

/-- Given two parallel lines and a point constrained to lie between them, 
    prove the range of the squared distance from this point to the origin. -/
theorem midpoint_distance_range (x₀ y₀ : ℝ) : 
  (∃ x y u v : ℝ, 
    x - 2*y - 2 = 0 ∧ 
    u - 2*v - 6 = 0 ∧ 
    x₀ = (x + u) / 2 ∧ 
    y₀ = (y + v) / 2 ∧
    (x₀ - 2)^2 + (y₀ + 1)^2 ≤ 5) →
  16/5 ≤ x₀^2 + y₀^2 ∧ x₀^2 + y₀^2 ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_distance_range_l3055_305589


namespace NUMINAMATH_CALUDE_circle_radius_in_ellipse_l3055_305533

/-- Two circles of radius r are externally tangent to each other and internally tangent to the ellipse x² + 4y² = 5. -/
theorem circle_radius_in_ellipse (r : ℝ) : 
  (∃ (x y : ℝ), x^2 + 4*y^2 = 5 ∧ (x - r)^2 + y^2 = r^2) →
  r = Real.sqrt 15 / 4 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_in_ellipse_l3055_305533


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3055_305510

/-- Given a circle (x+a)^2 + y^2 = 4 and a line x - y - 4 = 0 intersecting the circle
    to form a chord of length 2√2, prove that a = -2 or a = -6 -/
theorem circle_line_intersection (a : ℝ) : 
  (∃ x y : ℝ, (x + a)^2 + y^2 = 4 ∧ x - y - 4 = 0) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ + a)^2 + y₁^2 = 4 ∧ x₁ - y₁ - 4 = 0 ∧
    (x₂ + a)^2 + y₂^2 = 4 ∧ x₂ - y₂ - 4 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) →
  a = -2 ∨ a = -6 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3055_305510


namespace NUMINAMATH_CALUDE_geometric_progressions_existence_l3055_305566

theorem geometric_progressions_existence :
  (∃ a r : ℚ, 
    (∀ k : ℕ, k < 4 → 200 ≤ a * r^k ∧ a * r^k ≤ 1200) ∧
    (∀ k : ℕ, k < 4 → ∃ n : ℕ, a * r^k = n)) ∧
  (∃ b s : ℚ, 
    (∀ k : ℕ, k < 6 → 200 ≤ b * s^k ∧ b * s^k ≤ 1200) ∧
    (∀ k : ℕ, k < 6 → ∃ n : ℕ, b * s^k = n)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progressions_existence_l3055_305566


namespace NUMINAMATH_CALUDE_jimmy_pizza_cost_per_slice_l3055_305511

/-- Calculates the cost per slice of a pizza given the following parameters:
    * base_cost: The cost of a large pizza
    * num_slices: The number of slices in a large pizza
    * first_topping_cost: The cost of the first topping
    * next_two_toppings_cost: The cost of each of the next two toppings
    * remaining_toppings_cost: The cost of each remaining topping
    * num_toppings: The total number of toppings ordered
-/
def cost_per_slice (base_cost : ℚ) (num_slices : ℕ) (first_topping_cost : ℚ) 
                   (next_two_toppings_cost : ℚ) (remaining_toppings_cost : ℚ) 
                   (num_toppings : ℕ) : ℚ :=
  let total_cost := base_cost + first_topping_cost +
                    2 * next_two_toppings_cost +
                    (num_toppings - 3) * remaining_toppings_cost
  total_cost / num_slices

theorem jimmy_pizza_cost_per_slice :
  cost_per_slice 10 8 2 1 (1/2) 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_pizza_cost_per_slice_l3055_305511


namespace NUMINAMATH_CALUDE_intersection_theorem_l3055_305514

def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

def N : Set ℝ := {x | ∃ y, y = Real.log (1 - x^2)}

theorem intersection_theorem : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l3055_305514


namespace NUMINAMATH_CALUDE_inequality_solution_l3055_305572

theorem inequality_solution (x : ℝ) :
  x ≠ 1 →
  (x^3 - 3*x^2 + 2*x + 1) / (x^2 - 2*x + 1) ≤ 2 ↔ 
  (2 - Real.sqrt 3 < x ∧ x < 1) ∨ (1 < x ∧ x < 2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3055_305572
