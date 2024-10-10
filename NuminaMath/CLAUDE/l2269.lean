import Mathlib

namespace polynomial_division_remainder_l2269_226907

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  (3 * X ^ 2 - 20 * X + 68 : Polynomial ℚ) = (X - 4) * q + 36 := by
  sorry

end polynomial_division_remainder_l2269_226907


namespace high_school_science_club_payment_l2269_226976

theorem high_school_science_club_payment (B C : Nat) : 
  B < 10 → C < 10 → 
  (100 * B + 40 + C) % 15 = 0 → 
  (100 * B + 40 + C) % 5 = 0 → 
  B = 5 := by
sorry

end high_school_science_club_payment_l2269_226976


namespace dormitory_arrangement_l2269_226954

/-- Given:
  - If each dormitory houses 4 students, there would be 20 students without accommodation.
  - If each dormitory houses 8 students, one dormitory would be neither full nor empty,
    with the rest being completely full.
  Prove that there are 44 new students needing accommodation and 6 dormitories provided. -/
theorem dormitory_arrangement (num_dorms : ℕ) (num_students : ℕ) : 
  (4 * num_dorms + 20 = num_students) →
  (∃ k : ℕ, 0 < k ∧ k < 8 ∧ 8 * (num_dorms - 1) + k = num_students) →
  num_students = 44 ∧ num_dorms = 6 := by
  sorry

end dormitory_arrangement_l2269_226954


namespace lawn_maintenance_time_l2269_226994

theorem lawn_maintenance_time (mow_time fertilize_time total_time : ℕ) : 
  mow_time = 40 →
  fertilize_time = 2 * mow_time →
  total_time = mow_time + fertilize_time →
  total_time = 120 := by
sorry

end lawn_maintenance_time_l2269_226994


namespace complete_square_plus_integer_l2269_226980

theorem complete_square_plus_integer :
  ∃ (k : ℤ) (b : ℝ), ∀ (x : ℝ), x^2 + 14*x + 60 = (x + b)^2 + k :=
by sorry

end complete_square_plus_integer_l2269_226980


namespace children_per_seat_l2269_226927

theorem children_per_seat (total_children : ℕ) (total_seats : ℕ) (h1 : total_children = 58) (h2 : total_seats = 29) :
  total_children / total_seats = 2 := by
  sorry

end children_per_seat_l2269_226927


namespace book_selection_theorem_l2269_226900

theorem book_selection_theorem (n m : ℕ) (h1 : n = 8) (h2 : m = 5) :
  (Nat.choose (n - 1) (m - 1)) = 35 := by
  sorry

end book_selection_theorem_l2269_226900


namespace tenth_fibonacci_is_89_l2269_226903

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => fibonacci n + fibonacci (n + 1)

theorem tenth_fibonacci_is_89 : fibonacci 9 = 89 := by
  sorry

end tenth_fibonacci_is_89_l2269_226903


namespace students_liking_sports_l2269_226982

theorem students_liking_sports (B C : Finset Nat) 
  (hB : B.card = 10)
  (hC : C.card = 8)
  (hBC : (B ∩ C).card = 4) :
  (B ∪ C).card = 14 := by
  sorry

end students_liking_sports_l2269_226982


namespace train_passing_time_l2269_226923

/-- Calculates the time for a train to pass a person moving in the opposite direction -/
theorem train_passing_time 
  (train_length : ℝ) 
  (train_speed : ℝ) 
  (person_speed : ℝ) 
  (h1 : train_length = 110) 
  (h2 : train_speed = 65) 
  (h3 : person_speed = 7) : 
  (train_length / ((train_speed + person_speed) * (5/18))) = 5.5 := by
  sorry

#check train_passing_time

end train_passing_time_l2269_226923


namespace eggs_for_cake_l2269_226939

/-- The number of eggs in a dozen -/
def dozen : ℕ := 12

/-- The number of eggs Megan bought -/
def bought : ℕ := dozen

/-- The number of eggs Megan's neighbor gave her -/
def given : ℕ := dozen

/-- The number of eggs Megan used for an omelet -/
def omelet : ℕ := 2

/-- The number of eggs Megan plans to use for her next meals -/
def meal_plan : ℕ := 3 * 3

theorem eggs_for_cake :
  ∃ (cake : ℕ),
    bought + given - omelet - (bought + given - omelet) / 2 - meal_plan = cake ∧
    cake = 2 := by
  sorry

end eggs_for_cake_l2269_226939


namespace constant_ratio_sum_l2269_226977

theorem constant_ratio_sum (x₁ x₂ x₃ x₄ : ℝ) (k : ℝ) 
  (h_not_all_equal : ¬(x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄))
  (h_ratio_12_34 : (x₁ + x₂) / (x₃ + x₄) = k)
  (h_ratio_13_24 : (x₁ + x₃) / (x₂ + x₄) = k)
  (h_ratio_14_23 : (x₁ + x₄) / (x₂ + x₃) = k)
  (h_ratio_34_12 : (x₃ + x₄) / (x₁ + x₂) = k)
  (h_ratio_24_13 : (x₂ + x₄) / (x₁ + x₃) = k)
  (h_ratio_23_14 : (x₂ + x₃) / (x₁ + x₄) = k) :
  k = -1 := by sorry

end constant_ratio_sum_l2269_226977


namespace kocourkov_coins_l2269_226968

theorem kocourkov_coins (a b : ℕ+) : 
  (∀ n > 53, ∃ x y : ℕ, n = a.val * x + b.val * y) ∧ 
  (¬ ∃ x y : ℕ, 53 = a.val * x + b.val * y) → 
  ((a.val = 2 ∧ b.val = 55) ∨ (a.val = 3 ∧ b.val = 28) ∨ 
   (a.val = 55 ∧ b.val = 2) ∨ (a.val = 28 ∧ b.val = 3)) :=
by sorry

end kocourkov_coins_l2269_226968


namespace simplify_radical_product_l2269_226969

theorem simplify_radical_product (x : ℝ) (hx : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x) = 120 * x * Real.sqrt (2 * x) := by
  sorry

end simplify_radical_product_l2269_226969


namespace units_digit_of_3_pow_2005_l2269_226990

/-- The units digit of 3^n for n ≥ 1 -/
def units_digit_of_3_pow (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 1

/-- The units digit of 3^2005 is 3 -/
theorem units_digit_of_3_pow_2005 : units_digit_of_3_pow 2005 = 3 := by
  sorry

end units_digit_of_3_pow_2005_l2269_226990


namespace remainder_3005_div_99_l2269_226942

theorem remainder_3005_div_99 : 3005 % 99 = 35 := by
  sorry

end remainder_3005_div_99_l2269_226942


namespace kolya_walking_speed_l2269_226958

/-- Represents the scenario of Kolya's journey to the store -/
structure JourneyScenario where
  total_distance : ℝ
  initial_speed : ℝ
  doubled_speed : ℝ
  store_closing_time : ℝ

/-- Calculates Kolya's walking speed given a JourneyScenario -/
def calculate_walking_speed (scenario : JourneyScenario) : ℝ :=
  -- The actual calculation would go here
  sorry

/-- Theorem stating that Kolya's walking speed is 20/3 km/h -/
theorem kolya_walking_speed (scenario : JourneyScenario) 
  (h1 : scenario.initial_speed = 10)
  (h2 : scenario.doubled_speed = 2 * scenario.initial_speed)
  (h3 : scenario.store_closing_time = scenario.total_distance / scenario.initial_speed)
  (h4 : scenario.total_distance > 0) :
  calculate_walking_speed scenario = 20 / 3 := by
  sorry

end kolya_walking_speed_l2269_226958


namespace sweet_shop_inventory_l2269_226935

/-- The Sweet Shop inventory problem -/
theorem sweet_shop_inventory (total_cases : ℕ) (chocolate_cases : ℕ) (lollipop_cases : ℕ) :
  total_cases = 80 →
  chocolate_cases = 25 →
  lollipop_cases = total_cases - chocolate_cases →
  lollipop_cases = 55 := by
  sorry

#check sweet_shop_inventory

end sweet_shop_inventory_l2269_226935


namespace sector_arc_length_l2269_226938

theorem sector_arc_length (r : ℝ) (θ_deg : ℝ) (l : ℝ) : 
  r = 3 → θ_deg = 150 → l = (5 * π) / 2 → 
  l = r * ((θ_deg * π) / 180) :=
sorry

end sector_arc_length_l2269_226938


namespace least_number_with_remainder_four_l2269_226948

def is_valid_number (n : ℕ) : Prop :=
  n % 5 = 4 ∧ n % 9 = 4 ∧ n % 12 = 4 ∧ n % 18 = 4

theorem least_number_with_remainder_four :
  is_valid_number 184 ∧ ∀ m : ℕ, m < 184 → ¬ is_valid_number m :=
sorry

end least_number_with_remainder_four_l2269_226948


namespace nelly_painting_payment_l2269_226950

/-- The amount Nelly paid for a painting at an auction, given Joe's bid and the condition of her payment. -/
theorem nelly_painting_payment (joe_bid : ℕ) (h : joe_bid = 160000) : 
  3 * joe_bid + 2000 = 482000 := by
  sorry

end nelly_painting_payment_l2269_226950


namespace clock_strike_problem_l2269_226993

/-- Represents a clock that strikes at regular intervals. -/
structure Clock where
  interval : ℕ

/-- Calculates the time of the last strike given two clocks and total strikes. -/
def lastStrikeTime (clock1 clock2 : Clock) (totalStrikes : ℕ) : ℕ :=
  sorry

/-- Calculates the time between first and last strikes. -/
def timeBetweenStrikes (clock1 clock2 : Clock) (totalStrikes : ℕ) : ℕ :=
  sorry

theorem clock_strike_problem :
  let clock1 : Clock := { interval := 2 }
  let clock2 : Clock := { interval := 3 }
  let totalStrikes : ℕ := 13
  timeBetweenStrikes clock1 clock2 totalStrikes = 18 :=
by sorry

end clock_strike_problem_l2269_226993


namespace seed_germination_experiment_l2269_226916

theorem seed_germination_experiment (seeds_plot1 seeds_plot2 : ℕ)
  (germination_rate_plot2 : ℚ) (total_germination_rate : ℚ)
  (h1 : seeds_plot1 = 300)
  (h2 : seeds_plot2 = 200)
  (h3 : germination_rate_plot2 = 35 / 100)
  (h4 : total_germination_rate = 32 / 100)
  (h5 : (seeds_plot1 + seeds_plot2) * total_germination_rate =
        seeds_plot1 * (germination_rate_plot1 : ℚ) + seeds_plot2 * germination_rate_plot2) :
  germination_rate_plot1 = 30 / 100 := by
  sorry

#check seed_germination_experiment

end seed_germination_experiment_l2269_226916


namespace reciprocal_comparison_l2269_226922

theorem reciprocal_comparison : 
  (let numbers := [-1/2, -3, 1/3, 3, 3/2]
   ∀ x ∈ numbers, x < 1/x ↔ (x = -3 ∨ x = 1/3)) := by
  sorry

end reciprocal_comparison_l2269_226922


namespace unique_number_five_times_less_than_digit_sum_l2269_226943

def sum_of_digits (x : ℝ) : ℕ :=
  sorry

theorem unique_number_five_times_less_than_digit_sum :
  ∃! x : ℝ, x ≠ 0 ∧ x = (sum_of_digits x : ℝ) / 5 ∧ x = 1.8 :=
sorry

end unique_number_five_times_less_than_digit_sum_l2269_226943


namespace x_zero_necessary_not_sufficient_l2269_226967

theorem x_zero_necessary_not_sufficient :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0) ∧
  ¬(∀ x y : ℝ, x = 0 → x^2 + y^2 = 0) :=
by sorry

end x_zero_necessary_not_sufficient_l2269_226967


namespace fitness_center_membership_ratio_l2269_226913

theorem fitness_center_membership_ratio 
  (f m : ℕ) -- f: number of female members, m: number of male members
  (hf : f > 0) -- ensure f is positive
  (hm : m > 0) -- ensure m is positive
  (h_avg : (45 * f + 20 * m) / (f + m) = 25) : -- condition for overall average age
  f / m = 1 / 4 := by
sorry

end fitness_center_membership_ratio_l2269_226913


namespace perpendicular_slope_l2269_226932

theorem perpendicular_slope (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let original_slope := -a / b
  let perpendicular_slope := -1 / original_slope
  perpendicular_slope = b / a :=
by sorry

end perpendicular_slope_l2269_226932


namespace third_arrangement_is_goat_monkey_donkey_l2269_226919

-- Define the animals
inductive Animal : Type
| Monkey : Animal
| Donkey : Animal
| Goat : Animal

-- Define a seating arrangement as a triple of animals
def Arrangement := (Animal × Animal × Animal)

-- Define the property of an animal being in a specific position
def isInPosition (a : Animal) (pos : Nat) (arr : Arrangement) : Prop :=
  match pos, arr with
  | 0, (x, _, _) => x = a
  | 1, (_, x, _) => x = a
  | 2, (_, _, x) => x = a
  | _, _ => False

-- Define the property that each animal has been in each position
def eachAnimalInEachPosition (arr1 arr2 arr3 : Arrangement) : Prop :=
  ∀ (a : Animal) (p : Nat), p < 3 → 
    isInPosition a p arr1 ∨ isInPosition a p arr2 ∨ isInPosition a p arr3

-- Main theorem
theorem third_arrangement_is_goat_monkey_donkey 
  (arr1 arr2 arr3 : Arrangement)
  (h1 : isInPosition Animal.Monkey 2 arr1)
  (h2 : isInPosition Animal.Donkey 1 arr2)
  (h3 : eachAnimalInEachPosition arr1 arr2 arr3) :
  arr3 = (Animal.Goat, Animal.Monkey, Animal.Donkey) :=
sorry

end third_arrangement_is_goat_monkey_donkey_l2269_226919


namespace tangent_points_parallel_to_line_y_coordinates_tangent_points_coordinates_l2269_226984

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_points_parallel_to_line (x : ℝ) :
  (f' x = 4) ↔ (x = 1 ∨ x = -1) :=
by sorry

-- Theorem to prove the y-coordinates
theorem y_coordinates (x : ℝ) :
  (x = 1 ∨ x = -1) → (f x = 0 ∨ f x = -4) :=
by sorry

-- Main theorem combining the above results
theorem tangent_points_coordinates :
  ∃ (x y : ℝ), (f' x = 4 ∧ f x = y) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) :=
by sorry

end tangent_points_parallel_to_line_y_coordinates_tangent_points_coordinates_l2269_226984


namespace square_a_minus_2b_l2269_226918

theorem square_a_minus_2b (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a - 2*b)^2 = 25 := by
  sorry

end square_a_minus_2b_l2269_226918


namespace range_of_a_l2269_226928

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + 2*x + a > 0) → a ≤ 1 := by
  sorry

end range_of_a_l2269_226928


namespace smaller_circle_radius_l2269_226989

theorem smaller_circle_radius (R : ℝ) (r : ℝ) : 
  R = 10 → -- The radius of the largest circle is 10 meters
  4 * r = 2 * R → -- The diameter of the larger circle equals 4 times the radius of smaller circles
  r = 5 := by sorry

end smaller_circle_radius_l2269_226989


namespace initial_sweets_count_proof_initial_sweets_count_l2269_226911

theorem initial_sweets_count : ℕ → Prop :=
  fun x => 
    (x / 2 + 4 + 7 = x) → 
    x = 22

-- The proof is omitted
theorem proof_initial_sweets_count : initial_sweets_count 22 := by
  sorry

end initial_sweets_count_proof_initial_sweets_count_l2269_226911


namespace smallest_abundant_not_multiple_of_5_l2269_226902

def is_abundant (n : ℕ) : Prop :=
  (Finset.sum (Finset.range n) (λ i => if n % (i + 1) = 0 then i + 1 else 0)) > n

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem smallest_abundant_not_multiple_of_5 : 
  (∀ m : ℕ, m < 12 → (¬is_abundant m ∨ is_multiple_of_5 m)) ∧
  is_abundant 12 ∧ 
  ¬is_multiple_of_5 12 :=
sorry

end smallest_abundant_not_multiple_of_5_l2269_226902


namespace money_distribution_l2269_226987

/-- Given three people A, B, and C with a total amount of money,
    prove that B and C together have a specific amount. -/
theorem money_distribution (total A_C C B_C : ℕ) : 
  total = 1000 →
  A_C = 700 →
  C = 300 →
  B_C = total - (A_C - C) →
  B_C = 600 := by
  sorry

#check money_distribution

end money_distribution_l2269_226987


namespace kids_went_home_l2269_226992

theorem kids_went_home (initial_kids : ℝ) (remaining_kids : ℕ) : 
  initial_kids = 22.0 → remaining_kids = 8 → initial_kids - remaining_kids = 14 := by
  sorry

end kids_went_home_l2269_226992


namespace triangle_side_length_l2269_226991

theorem triangle_side_length 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : c = Real.sqrt 2)
  (h2 : b = Real.sqrt 6)
  (h3 : B = 2 * π / 3) -- 120° in radians
  (h4 : A + B + C = π) -- sum of angles in a triangle
  (h5 : 0 < a ∧ 0 < b ∧ 0 < c) -- positive side lengths
  (h6 : a / (Real.sin A) = b / (Real.sin B)) -- sine rule
  (h7 : b / (Real.sin B) = c / (Real.sin C)) -- sine rule
  : a = Real.sqrt 2 := by
  sorry

end triangle_side_length_l2269_226991


namespace interest_rate_problem_l2269_226912

/-- Calculates the amount after simple interest is applied -/
def amountAfterSimpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem interest_rate_problem (originalRate : ℝ) :
  let principal : ℝ := 1000
  let time : ℝ := 5
  let increasedRate : ℝ := originalRate + 0.05
  amountAfterSimpleInterest principal increasedRate time = 1750 →
  amountAfterSimpleInterest principal originalRate time = 1500 :=
by
  sorry

end interest_rate_problem_l2269_226912


namespace count_permutations_2007_l2269_226970

/-- The number of permutations of integers 1 to n with exactly one descent -/
def permutations_with_one_descent (n : ℕ) : ℕ :=
  2^n - (n + 1)

/-- The theorem to be proved -/
theorem count_permutations_2007 :
  permutations_with_one_descent 2007 = 2^3 * (2^2004 - 251) := by
  sorry

end count_permutations_2007_l2269_226970


namespace max_value_M_l2269_226915

/-- The maximum value of M = 11xy + 3x + 2012yz, where x, y, z are non-negative integers and x + y + z = 1000 -/
theorem max_value_M : 
  ∃ (x y z : ℕ), 
    x + y + z = 1000 ∧ 
    ∀ (a b c : ℕ), 
      a + b + c = 1000 → 
      11 * x * y + 3 * x + 2012 * y * z ≥ 11 * a * b + 3 * a + 2012 * b * c ∧
      11 * x * y + 3 * x + 2012 * y * z = 503000000 :=
by sorry

end max_value_M_l2269_226915


namespace distance_from_pole_to_line_l2269_226930

/-- Given a line with polar equation ρ sin(θ + π/4) = 1, 
    the distance from the pole to this line is 1. -/
theorem distance_from_pole_to_line (ρ θ : ℝ) : 
  ρ * Real.sin (θ + π/4) = 1 → 
  (∃ d : ℝ, d = 1 ∧ d = abs (2) / Real.sqrt (2 + 2)) := by
  sorry

end distance_from_pole_to_line_l2269_226930


namespace converse_of_negative_square_positive_l2269_226979

theorem converse_of_negative_square_positive :
  (∀ x : ℝ, x < 0 → x^2 > 0) →
  (∀ x : ℝ, x^2 > 0 → x < 0) :=
sorry

end converse_of_negative_square_positive_l2269_226979


namespace complement_of_union_l2269_226917

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {4, 5, 6}

theorem complement_of_union : 
  U \ (A ∪ B) = {2} := by sorry

end complement_of_union_l2269_226917


namespace largest_logarithm_l2269_226946

theorem largest_logarithm (h : 0 < Real.log 2 ∧ Real.log 2 < 1) :
  2 * Real.log 2 > Real.log 2 ∧ 
  Real.log 2 > (Real.log 2)^2 ∧ 
  (Real.log 2)^2 > Real.log (Real.log 2) := by
  sorry

end largest_logarithm_l2269_226946


namespace wilson_number_l2269_226905

theorem wilson_number (N : ℚ) : N - (1/3) * N = 16/3 → N = 8 := by
  sorry

end wilson_number_l2269_226905


namespace quadratic_equation_property_l2269_226947

/-- A quadratic equation with two equal real roots -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  condition : a - b + c = 0
  equal_roots : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ (∀ y : ℝ, a * y^2 + b * y + c = 0 → y = x)

/-- Theorem stating that for a quadratic equation with two equal real roots and a - b + c = 0, we have 2a - b = 0 -/
theorem quadratic_equation_property (eq : QuadraticEquation) : 2 * eq.a - eq.b = 0 := by
  sorry

end quadratic_equation_property_l2269_226947


namespace problem_1_l2269_226986

theorem problem_1 (m n : ℝ) (h1 : m = 2) (h2 : n = 1) : 
  (2*m^2 - 3*m*n + 8) - (5*m*n - 4*m^2 + 8) = 8 := by sorry

end problem_1_l2269_226986


namespace height_percentage_difference_l2269_226944

theorem height_percentage_difference (height_A height_B : ℝ) :
  height_B = height_A * (1 + 0.42857142857142854) →
  (height_B - height_A) / height_B * 100 = 30 := by
  sorry

end height_percentage_difference_l2269_226944


namespace orchid_painting_time_l2269_226933

/-- The time it takes Ellen to paint various flowers and vines -/
structure PaintingTimes where
  lily : ℕ
  rose : ℕ
  vine : ℕ
  total : ℕ
  lilies : ℕ
  roses : ℕ
  orchids : ℕ
  vines : ℕ

/-- Theorem stating that the time to paint an orchid is 3 minutes -/
theorem orchid_painting_time (pt : PaintingTimes)
  (h1 : pt.lily = 5)
  (h2 : pt.rose = 7)
  (h3 : pt.vine = 2)
  (h4 : pt.total = 213)
  (h5 : pt.lilies = 17)
  (h6 : pt.roses = 10)
  (h7 : pt.orchids = 6)
  (h8 : pt.vines = 20) :
  (pt.total - (pt.lily * pt.lilies + pt.rose * pt.roses + pt.vine * pt.vines)) / pt.orchids = 3 :=
by sorry

end orchid_painting_time_l2269_226933


namespace sqrt_a_minus_2_real_l2269_226961

theorem sqrt_a_minus_2_real (a : ℝ) : (∃ x : ℝ, x^2 = a - 2) ↔ a ≥ 2 := by
  sorry

end sqrt_a_minus_2_real_l2269_226961


namespace subtraction_preserves_inequality_l2269_226934

theorem subtraction_preserves_inequality (a b c : ℝ) (h : a > b) : a - c > b - c := by
  sorry

end subtraction_preserves_inequality_l2269_226934


namespace factorization_equality_l2269_226906

theorem factorization_equality (x y : ℝ) :
  3 * y * (y^2 - 4) + 5 * x * (y^2 - 4) = (3*y + 5*x) * (y + 2) * (y - 2) := by
  sorry

end factorization_equality_l2269_226906


namespace geometric_sequence_fourth_term_l2269_226955

/-- Given a geometric sequence where the first term is x, the second term is 3x + 3, 
    and the third term is 5x + 5, the fourth term of this sequence is -5/4. -/
theorem geometric_sequence_fourth_term (x : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (3*x + 3) = x * r ∧ (5*x + 5) = (3*x + 3) * r) → 
  ∃ t : ℝ, t = -5/4 ∧ t = (5*x + 5) * (3*x + 3) / x := by
  sorry


end geometric_sequence_fourth_term_l2269_226955


namespace unique_solution_m_n_l2269_226952

theorem unique_solution_m_n : ∃! (m n : ℕ+), (m + n : ℕ)^(m : ℕ) = n^(m : ℕ) + 1413 :=
  sorry

end unique_solution_m_n_l2269_226952


namespace factorization1_factorization2_l2269_226997

-- Define the expressions
def expr1 (x y : ℝ) : ℝ := 4 - 12 * (x - y) + 9 * (x - y)^2

def expr2 (a x : ℝ) : ℝ := 2 * a * (x^2 + 1)^2 - 8 * a * x^2

-- State the theorems
theorem factorization1 (x y : ℝ) : expr1 x y = (2 - 3*x + 3*y)^2 := by sorry

theorem factorization2 (a x : ℝ) : expr2 a x = 2 * a * (x - 1)^2 * (x + 1)^2 := by sorry

end factorization1_factorization2_l2269_226997


namespace function_relation_implies_a_half_l2269_226996

/-- Given two functions f and g defined on ℝ satisfying certain conditions, prove that a = 1/2 -/
theorem function_relation_implies_a_half :
  ∀ (f g : ℝ → ℝ) (a : ℝ),
    (∀ x, f x = a^x * g x) →
    (a > 0) →
    (a ≠ 1) →
    (∀ x, g x ≠ 0 → f x * (deriv g x) > (deriv f x) * g x) →
    (f 1 / g 1 + f (-1) / g (-1) = 5/2) →
    a = 1/2 := by
  sorry

end function_relation_implies_a_half_l2269_226996


namespace impossible_ratio_l2269_226926

theorem impossible_ratio (n : ℕ) (boys girls : ℕ) : 
  30 < n → n < 40 → boys + girls = n → ¬(3 * girls = 7 * boys) := by
  sorry

end impossible_ratio_l2269_226926


namespace sqrt_equation_solution_l2269_226974

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 * x + 9) = 13 → x = 40 := by
  sorry

end sqrt_equation_solution_l2269_226974


namespace blue_highlighters_count_l2269_226904

def total_highlighters : ℕ := 15
def pink_highlighters : ℕ := 3
def yellow_highlighters : ℕ := 7

theorem blue_highlighters_count :
  total_highlighters - (pink_highlighters + yellow_highlighters) = 5 :=
by sorry

end blue_highlighters_count_l2269_226904


namespace original_number_proof_l2269_226956

theorem original_number_proof (x : ℚ) : (1 + 1 / x = 5 / 2) → x = 2 / 3 := by
  sorry

end original_number_proof_l2269_226956


namespace equation_proof_l2269_226971

theorem equation_proof : (15 : ℝ) ^ 3 * 7 ^ 4 / 5670 = 1428.75 := by
  sorry

end equation_proof_l2269_226971


namespace electron_transfer_for_N2_production_l2269_226962

-- Define the chemical elements and compounds
def Zn : Type := Unit
def H : Type := Unit
def N : Type := Unit
def O : Type := Unit
def HNO3 : Type := Unit
def NH4NO3 : Type := Unit
def H2O : Type := Unit
def ZnNO3_2 : Type := Unit

-- Define the reaction
def reaction : Type := Unit

-- Define Avogadro's constant
def Na : ℕ := sorry

-- Define the electron transfer function
def electron_transfer (r : reaction) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem electron_transfer_for_N2_production (r : reaction) :
  electron_transfer r 1 = 5 * Na := by sorry

end electron_transfer_for_N2_production_l2269_226962


namespace monotonic_decreasing_implies_a_leq_neg_one_l2269_226973

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a - 1)*x + 2

-- State the theorem
theorem monotonic_decreasing_implies_a_leq_neg_one :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → y ≤ 2 → f a x > f a y) → a ≤ -1 :=
by sorry

end monotonic_decreasing_implies_a_leq_neg_one_l2269_226973


namespace last_ball_is_green_l2269_226909

/-- Represents the color of a ball -/
inductive Color
  | Red
  | Blue
  | Green

/-- Represents the state of the box with balls -/
structure BoxState where
  red : Nat
  blue : Nat
  green : Nat

/-- Represents an exchange operation -/
inductive Exchange
  | RedBlueToGreen
  | RedGreenToBlue
  | BlueGreenToRed

/-- Applies an exchange operation to a box state -/
def applyExchange (state : BoxState) (ex : Exchange) : BoxState :=
  match ex with
  | Exchange.RedBlueToGreen => 
      { red := state.red - 1, blue := state.blue - 1, green := state.green + 1 }
  | Exchange.RedGreenToBlue => 
      { red := state.red - 1, blue := state.blue + 1, green := state.green - 1 }
  | Exchange.BlueGreenToRed => 
      { red := state.red + 1, blue := state.blue - 1, green := state.green - 1 }

/-- Checks if the box state has only one ball left -/
def isLastBall (state : BoxState) : Bool :=
  state.red + state.blue + state.green = 1

/-- Gets the color of the last ball -/
def getLastBallColor (state : BoxState) : Option Color :=
  if state.red = 1 then some Color.Red
  else if state.blue = 1 then some Color.Blue
  else if state.green = 1 then some Color.Green
  else none

/-- The main theorem to prove -/
theorem last_ball_is_green (exchanges : List Exchange) :
  let initialState : BoxState := { red := 10, blue := 11, green := 12 }
  let finalState := exchanges.foldl applyExchange initialState
  isLastBall finalState → getLastBallColor finalState = some Color.Green :=
by sorry

end last_ball_is_green_l2269_226909


namespace max_value_of_x_plus_y_l2269_226941

theorem max_value_of_x_plus_y : 
  ∃ (M : ℝ), M = 4 ∧ 
  ∀ (x y : ℝ), x^2 + y + 3*x - 3 = 0 → x + y ≤ M :=
sorry

end max_value_of_x_plus_y_l2269_226941


namespace smallest_denominator_between_fractions_l2269_226998

theorem smallest_denominator_between_fractions :
  ∃ (p q : ℕ), 
    q = 4027 ∧ 
    (1 : ℚ) / 2014 < (p : ℚ) / q ∧ 
    (p : ℚ) / q < (1 : ℚ) / 2013 ∧
    (∀ (p' q' : ℕ), 
      (1 : ℚ) / 2014 < (p' : ℚ) / q' ∧ 
      (p' : ℚ) / q' < (1 : ℚ) / 2013 → 
      q ≤ q') :=
by sorry

end smallest_denominator_between_fractions_l2269_226998


namespace range_of_m_l2269_226981

-- Define the inequality system
def inequality_system (x a : ℝ) : Prop :=
  (x - a) / 3 < 0 ∧ 2 * (x - 5) < 3 * x - 8

-- Define the solution set
def solution_set (a : ℝ) : Set ℤ :=
  {x : ℤ | inequality_system x a}

-- State the theorem
theorem range_of_m (a : ℝ) (m : ℝ) :
  (∀ x : ℤ, x ∈ solution_set a ↔ (x = -1 ∨ x = 0)) →
  (10 * a = 2 * m + 5) →
  -2.5 < m ∧ m ≤ 2.5 :=
sorry

end range_of_m_l2269_226981


namespace product_loss_percentage_l2269_226949

/-- Proves the percentage loss of a product given specific selling prices and gain percentages --/
theorem product_loss_percentage 
  (cp : ℝ) -- Cost price
  (sp_gain : ℝ) -- Selling price with gain
  (sp_loss : ℝ) -- Selling price with loss
  (gain_percent : ℝ) -- Gain percentage
  (h1 : sp_gain = cp * (1 + gain_percent / 100)) -- Condition for selling price with gain
  (h2 : sp_gain = 168) -- Given selling price with gain
  (h3 : gain_percent = 20) -- Given gain percentage
  (h4 : sp_loss = 119) -- Given selling price with loss
  : (cp - sp_loss) / cp * 100 = 15 := by
  sorry

end product_loss_percentage_l2269_226949


namespace lunks_for_dozen_apples_l2269_226988

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks (lunks : ℚ) : ℚ := (3 / 5) * lunks

/-- Exchange rate between kunks and apples -/
def kunks_to_apples (kunks : ℚ) : ℚ := 2 * kunks

/-- The number of lunks required to purchase a given number of apples -/
def lunks_for_apples (apples : ℚ) : ℚ :=
  (5 / 3) * (apples / 2)

theorem lunks_for_dozen_apples :
  lunks_for_apples 12 = 10 := by sorry

end lunks_for_dozen_apples_l2269_226988


namespace four_points_plane_count_l2269_226966

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a function to count the number of planes determined by four points
def countPlanesFromFourPoints (A B C D : Point3D) : Nat :=
  sorry

-- Theorem statement
theorem four_points_plane_count (A B C D : Point3D) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) : 
  countPlanesFromFourPoints A B C D = 1 ∨ countPlanesFromFourPoints A B C D = 4 :=
sorry

end four_points_plane_count_l2269_226966


namespace consecutive_eight_product_divisible_by_ten_l2269_226964

theorem consecutive_eight_product_divisible_by_ten (n : ℕ+) : 
  10 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) * (n + 7)) := by
  sorry

#check consecutive_eight_product_divisible_by_ten

end consecutive_eight_product_divisible_by_ten_l2269_226964


namespace log_expression_equals_negative_one_l2269_226951

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_negative_one :
  log10 (5/2) + 2 * log10 2 - (1/2)⁻¹ = -1 := by
  sorry

end log_expression_equals_negative_one_l2269_226951


namespace parabola_focus_value_hyperbola_standard_equation_l2269_226925

-- Problem 1
theorem parabola_focus_value (p : ℝ) (h1 : p > 0) :
  (∃ x y : ℝ, y^2 = 2*p*x ∧ 2*x - y - 4 = 0 ∧ x = p ∧ y = 0) →
  p = 2 := by sorry

-- Problem 2
theorem hyperbola_standard_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (b / a = 3 / 4) ∧ 
  (a^2 / (a^2 + b^2).sqrt = 16 / 5) →
  ∀ x y : ℝ, x^2 / 16 - y^2 / 9 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1 := by sorry

end parabola_focus_value_hyperbola_standard_equation_l2269_226925


namespace equation_describes_cone_l2269_226914

/-- Spherical coordinates -/
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Definition of a cone in spherical coordinates -/
def IsCone (c : ℝ) (f : SphericalCoordinates → Prop) : Prop :=
  ∀ p : SphericalCoordinates, f p ↔ p.ρ = c * Real.sin p.φ

/-- The main theorem: the equation ρ = c * sin φ describes a cone -/
theorem equation_describes_cone (c : ℝ) (hc : c > 0) :
  IsCone c (fun p => p.ρ = c * Real.sin p.φ) :=
sorry

end equation_describes_cone_l2269_226914


namespace volume_region_equivalence_l2269_226936

theorem volume_region_equivalence (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  |x + 2*y + z| + |x - y - z| ≤ 10 ↔ max (x + 2*y + z) (x - y - z) ≤ 5 := by
  sorry

#check volume_region_equivalence

end volume_region_equivalence_l2269_226936


namespace solve_exponential_equation_l2269_226999

theorem solve_exponential_equation :
  ∃ x : ℝ, 2^(2*x - 1) = (1/4 : ℝ) ∧ x = -1/2 := by
sorry

end solve_exponential_equation_l2269_226999


namespace two_thirds_in_M_l2269_226963

open Set

-- Define the sets A and B as open intervals
def A : Set ℝ := Ioo (-4) 1
def B : Set ℝ := Ioo (-2) 5

-- Define M as the intersection of A and B
def M : Set ℝ := A ∩ B

-- Theorem statement
theorem two_thirds_in_M : (2/3 : ℝ) ∈ M := by sorry

end two_thirds_in_M_l2269_226963


namespace h_of_three_equals_five_l2269_226995

-- Define the function h
def h (x : ℝ) : ℝ := 2*(x-2) + 3

-- State the theorem
theorem h_of_three_equals_five : h 3 = 5 := by
  sorry

end h_of_three_equals_five_l2269_226995


namespace factor_81_minus_4y4_l2269_226945

theorem factor_81_minus_4y4 (y : ℝ) : 81 - 4 * y^4 = (9 + 2 * y^2) * (9 - 2 * y^2) := by
  sorry

end factor_81_minus_4y4_l2269_226945


namespace find_numbers_l2269_226960

theorem find_numbers (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 12) : x = 26 ∧ y = 14 := by
  sorry

end find_numbers_l2269_226960


namespace min_value_expression_l2269_226937

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  a^2 + 6*a*b + 9*b^2 + 4*c^2 ≥ 180 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 27 ∧ 
    a₀^2 + 6*a₀*b₀ + 9*b₀^2 + 4*c₀^2 = 180 :=
by sorry

end min_value_expression_l2269_226937


namespace factorial_equality_l2269_226908

theorem factorial_equality (N : ℕ) (h : N > 0) :
  (7 : ℕ).factorial * (11 : ℕ).factorial = 18 * N.factorial → N = 11 := by
  sorry

end factorial_equality_l2269_226908


namespace cheese_needed_for_event_l2269_226924

def meat_for_10_sandwiches : ℝ := 4
def number_of_sandwiches_planned : ℕ := 30
def initial_sandwich_count : ℕ := 10

theorem cheese_needed_for_event :
  let meat_per_sandwich : ℝ := meat_for_10_sandwiches / initial_sandwich_count
  let cheese_per_sandwich : ℝ := meat_per_sandwich / 2
  cheese_per_sandwich * number_of_sandwiches_planned = 6 := by
sorry

end cheese_needed_for_event_l2269_226924


namespace no_common_root_l2269_226959

theorem no_common_root (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬∃ x : ℝ, (x^2 + b*x + c = 0) ∧ (x^2 + a*x + d = 0) :=
by sorry

end no_common_root_l2269_226959


namespace vector_problem_l2269_226975

/-- Given vectors in 2D space -/
def a : ℝ × ℝ := (5, 6)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def c (y : ℝ) : ℝ × ℝ := (2, y)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Perpendicular vectors have zero dot product -/
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

/-- Parallel vectors are scalar multiples of each other -/
def parallel (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), w = (k * v.1, k * v.2)

/-- Main theorem -/
theorem vector_problem :
  ∃ (x y : ℝ),
    perpendicular a (b x) ∧
    parallel a (c y) ∧
    x = -18/5 ∧
    y = 12/5 := by
  sorry

end vector_problem_l2269_226975


namespace sophies_money_correct_l2269_226929

/-- The amount of money Sophie's aunt gave her --/
def sophies_money : ℝ := 260

/-- The cost of one shirt --/
def shirt_cost : ℝ := 18.50

/-- The number of shirts Sophie bought --/
def num_shirts : ℕ := 2

/-- The cost of the trousers --/
def trouser_cost : ℝ := 63

/-- The cost of one additional article of clothing --/
def additional_item_cost : ℝ := 40

/-- The number of additional articles of clothing Sophie plans to buy --/
def num_additional_items : ℕ := 4

/-- Theorem stating that the amount of money Sophie's aunt gave her is correct --/
theorem sophies_money_correct : 
  sophies_money = 
    shirt_cost * num_shirts + 
    trouser_cost + 
    additional_item_cost * num_additional_items := by
  sorry

end sophies_money_correct_l2269_226929


namespace erased_number_proof_l2269_226920

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  n ≥ 3 →
  x ≥ 3 →
  x ≤ n →
  (n * (n + 1) / 2 - 3 - x) / (n - 2 : ℚ) = 151 / 3 →
  x = 3 := by
sorry

end erased_number_proof_l2269_226920


namespace seating_arrangements_l2269_226901

/-- The number of seats in the front row -/
def front_seats : ℕ := 11

/-- The number of seats in the back row -/
def back_seats : ℕ := 12

/-- The total number of seats -/
def total_seats : ℕ := front_seats + back_seats

/-- The number of restricted seats in the front row -/
def restricted_seats : ℕ := 3

/-- The number of people to be seated -/
def people : ℕ := 2

/-- The number of arrangements without restrictions -/
def arrangements_without_restrictions : ℕ := total_seats * (total_seats - 2)

/-- The number of arrangements with one person in restricted seats -/
def arrangements_with_one_restricted : ℕ := restricted_seats * (total_seats - 3)

/-- The number of arrangements with both people in restricted seats -/
def arrangements_both_restricted : ℕ := restricted_seats * (restricted_seats - 1)

theorem seating_arrangements :
  arrangements_without_restrictions - 2 * arrangements_with_one_restricted + arrangements_both_restricted = 346 := by
  sorry

end seating_arrangements_l2269_226901


namespace margarets_mean_score_l2269_226953

def scores : List ℝ := [86, 88, 91, 93, 95, 97, 99, 100]

theorem margarets_mean_score 
  (h1 : scores.length = 8)
  (h2 : ∃ (cyprian_scores margaret_scores : List ℝ), 
    cyprian_scores.length = 4 ∧ 
    margaret_scores.length = 4 ∧ 
    cyprian_scores ++ margaret_scores = scores)
  (h3 : ∃ (cyprian_scores : List ℝ), 
    cyprian_scores.length = 4 ∧ 
    cyprian_scores.sum / cyprian_scores.length = 92) :
  ∃ (margaret_scores : List ℝ), 
    margaret_scores.length = 4 ∧ 
    margaret_scores.sum / margaret_scores.length = 95.25 := by
  sorry

end margarets_mean_score_l2269_226953


namespace integer_roots_quadratic_l2269_226957

theorem integer_roots_quadratic (p q : ℕ) : 
  (∃ x y : ℤ, x^2 - p*q*x + p + q = 0 ∧ y^2 - p*q*y + p + q = 0 ∧ x ≠ y) ↔ 
  ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) ∨ (p = 2 ∧ q = 2) ∨ (p = 1 ∧ q = 5) ∨ (p = 5 ∧ q = 1)) :=
by sorry

end integer_roots_quadratic_l2269_226957


namespace jackie_sleep_time_l2269_226965

theorem jackie_sleep_time (total_hours work_hours exercise_hours free_hours : ℕ) 
  (h1 : total_hours = 24)
  (h2 : work_hours = 8)
  (h3 : exercise_hours = 3)
  (h4 : free_hours = 5) :
  total_hours - (work_hours + exercise_hours + free_hours) = 8 := by
  sorry

end jackie_sleep_time_l2269_226965


namespace sarah_investment_l2269_226910

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem sarah_investment :
  let principal : ℝ := 1500
  let rate : ℝ := 0.04
  let time : ℕ := 21
  let final_amount := compound_interest principal rate time
  ∃ ε > 0, |final_amount - 3046.28| < ε :=
sorry

end sarah_investment_l2269_226910


namespace digit_product_le_unique_solution_l2269_226972

-- Define p(n) as the product of digits of n
def digit_product (n : ℕ) : ℕ := sorry

-- Theorem 1: For any natural number n, p(n) ≤ n
theorem digit_product_le (n : ℕ) : digit_product n ≤ n := by sorry

-- Theorem 2: 45 is the only natural number satisfying 10p(n) = n^2 + 4n - 2005
theorem unique_solution :
  ∀ n : ℕ, 10 * (digit_product n) = n^2 + 4*n - 2005 ↔ n = 45 := by sorry

end digit_product_le_unique_solution_l2269_226972


namespace polynomial_evaluation_l2269_226983

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 9 = 0) :
  x^3 - 3*x^2 - 9*x + 27 = 27 := by
  sorry

end polynomial_evaluation_l2269_226983


namespace upstream_distance_l2269_226985

theorem upstream_distance
  (boat_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (upstream_time : ℝ)
  (h1 : boat_speed = 20)
  (h2 : downstream_distance = 96)
  (h3 : downstream_time = 3)
  (h4 : upstream_time = 11)
  : ∃ (upstream_distance : ℝ), upstream_distance = 88 :=
by
  sorry

#check upstream_distance

end upstream_distance_l2269_226985


namespace correct_result_l2269_226921

theorem correct_result (x : ℝ) : (-1.25 * x) - 0.25 = 1.25 * x → -1.25 * x = 0.125 := by
  sorry

end correct_result_l2269_226921


namespace percentage_less_than_l2269_226978

theorem percentage_less_than (w x y z P : ℝ) : 
  w = x * (1 - P / 100) →
  x = y * 0.6 →
  z = y * 0.54 →
  z = w * 1.5 →
  P = 40 :=
by sorry

end percentage_less_than_l2269_226978


namespace slope_y_intercept_ratio_l2269_226940

/-- A line in the coordinate plane with slope m, y-intercept b, and x-intercept 2 -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept
  x_intercept_eq_two : m * 2 + b = 0  -- condition for x-intercept = 2

/-- The slope is some fraction of the y-intercept -/
def slope_fraction (k : Line) (c : ℝ) : Prop :=
  k.m = c * k.b

theorem slope_y_intercept_ratio (k : Line) :
  ∃ c : ℝ, slope_fraction k c ∧ c = -1/2 := by
  sorry

end slope_y_intercept_ratio_l2269_226940


namespace recurrence_relation_solution_l2269_226931

def a (n : ℕ) : ℤ := 2 * 4^n - 2*n + 2
def b (n : ℕ) : ℤ := 2 * 4^n + 2*n - 2

theorem recurrence_relation_solution :
  (∀ n : ℕ, a (n + 1) = 3 * a n + b n - 4) ∧
  (∀ n : ℕ, b (n + 1) = 2 * a n + 2 * b n + 2) ∧
  a 0 = 4 ∧
  b 0 = 0 := by sorry

end recurrence_relation_solution_l2269_226931
