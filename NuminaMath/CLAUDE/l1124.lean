import Mathlib

namespace christian_initial_savings_l1124_112493

/-- The price of the perfume in dollars -/
def perfume_price : ℚ := 50

/-- Sue's initial savings in dollars -/
def sue_initial : ℚ := 7

/-- The number of yards Christian mowed -/
def yards_mowed : ℕ := 4

/-- The price Christian charged per yard in dollars -/
def price_per_yard : ℚ := 5

/-- The number of dogs Sue walked -/
def dogs_walked : ℕ := 6

/-- The price Sue charged per dog in dollars -/
def price_per_dog : ℚ := 2

/-- The additional amount needed in dollars -/
def additional_needed : ℚ := 6

/-- Christian's earnings from mowing yards -/
def christian_earnings : ℚ := yards_mowed * price_per_yard

/-- Sue's earnings from walking dogs -/
def sue_earnings : ℚ := dogs_walked * price_per_dog

/-- Total money they have after their work -/
def total_after_work : ℚ := christian_earnings + sue_earnings + sue_initial

/-- Christian's initial savings -/
def christian_initial : ℚ := perfume_price - total_after_work - additional_needed

theorem christian_initial_savings : christian_initial = 5 := by
  sorry

end christian_initial_savings_l1124_112493


namespace ancient_chinese_math_problem_l1124_112495

/-- Represents the problem from "The Compendious Book on Calculation by Completion and Balancing" --/
theorem ancient_chinese_math_problem (x y : ℕ) : 
  (∀ (room_capacity : ℕ), 
    (room_capacity = 7 → 7 * x + 7 = y) ∧ 
    (room_capacity = 9 → 9 * (x - 1) = y)) ↔ 
  (7 * x + 7 = y ∧ 9 * (x - 1) = y) :=
sorry

end ancient_chinese_math_problem_l1124_112495


namespace total_fruits_is_107_l1124_112430

/-- The number of fruits picked by George and Amelia -/
def total_fruits (george_oranges amelia_apples : ℕ) : ℕ :=
  let george_apples := amelia_apples + 5
  let amelia_oranges := george_oranges - 18
  (george_oranges + amelia_oranges) + (george_apples + amelia_apples)

/-- Theorem stating that the total number of fruits picked is 107 -/
theorem total_fruits_is_107 :
  total_fruits 45 15 = 107 := by sorry

end total_fruits_is_107_l1124_112430


namespace arithmetic_equations_correctness_l1124_112408

theorem arithmetic_equations_correctness : 
  (-2 + 8 ≠ 10) ∧ 
  (-1 - 3 = -4) ∧ 
  (-2 * 2 ≠ 4) ∧ 
  (-8 / -1 ≠ -1/8) :=
by sorry

end arithmetic_equations_correctness_l1124_112408


namespace noah_sales_revenue_l1124_112425

-- Define constants
def large_painting_price : ℝ := 60
def small_painting_price : ℝ := 30
def last_month_large_sales : ℕ := 8
def last_month_small_sales : ℕ := 4
def large_painting_discount : ℝ := 0.1
def small_painting_commission : ℝ := 0.05
def sales_tax_rate : ℝ := 0.07

-- Define the theorem
theorem noah_sales_revenue :
  let this_month_large_sales := 2 * last_month_large_sales
  let this_month_small_sales := 2 * last_month_small_sales
  let discounted_large_price := large_painting_price * (1 - large_painting_discount)
  let commissioned_small_price := small_painting_price * (1 - small_painting_commission)
  let total_sales_before_tax := 
    this_month_large_sales * discounted_large_price +
    this_month_small_sales * commissioned_small_price
  let sales_tax := total_sales_before_tax * sales_tax_rate
  let total_sales_revenue := total_sales_before_tax + sales_tax
  total_sales_revenue = 1168.44 := by
  sorry

end noah_sales_revenue_l1124_112425


namespace ice_palace_staircase_steps_l1124_112486

theorem ice_palace_staircase_steps 
  (time_for_20_steps : ℕ) 
  (steps_20 : ℕ) 
  (total_time : ℕ) 
  (h1 : time_for_20_steps = 120)
  (h2 : steps_20 = 20)
  (h3 : total_time = 180) :
  (total_time * steps_20) / time_for_20_steps = 30 :=
by sorry

end ice_palace_staircase_steps_l1124_112486


namespace parallel_iff_a_eq_neg_one_l1124_112417

/-- Line represented by a linear equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem parallel_iff_a_eq_neg_one :
  let l1 : Line := { a := 1, b := -1, c := -1 }
  let l2 : Line := { a := 1, b := a, c := -2 }
  parallel l1 l2 ↔ a = -1 :=
sorry

end parallel_iff_a_eq_neg_one_l1124_112417


namespace quadratic_equation_roots_l1124_112434

theorem quadratic_equation_roots (p q : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 → (x + 1)^2 - p^2*(x + 1) + p*q = 0) →
  ((p = 1 ∧ ∃ q : ℝ, True) ∨ (p = -2 ∧ q = -1)) :=
by sorry

end quadratic_equation_roots_l1124_112434


namespace band_size_correct_l1124_112498

/-- The number of flutes that tried out -/
def flutes : ℕ := 20

/-- The number of clarinets that tried out -/
def clarinets : ℕ := 30

/-- The number of trumpets that tried out -/
def trumpets : ℕ := 60

/-- The number of pianists that tried out -/
def pianists : ℕ := 20

/-- The fraction of flutes that got in -/
def flute_acceptance : ℚ := 4/5

/-- The fraction of clarinets that got in -/
def clarinet_acceptance : ℚ := 1/2

/-- The fraction of trumpets that got in -/
def trumpet_acceptance : ℚ := 1/3

/-- The fraction of pianists that got in -/
def pianist_acceptance : ℚ := 1/10

/-- The total number of people in the band -/
def band_total : ℕ := 53

theorem band_size_correct :
  (flutes : ℚ) * flute_acceptance +
  (clarinets : ℚ) * clarinet_acceptance +
  (trumpets : ℚ) * trumpet_acceptance +
  (pianists : ℚ) * pianist_acceptance = band_total := by
  sorry

end band_size_correct_l1124_112498


namespace cricket_theorem_l1124_112488

def cricket_problem (team_scores : List Nat) : Prop :=
  let n := team_scores.length
  let lost_matches := 6
  let won_matches := n - lost_matches
  let opponent_scores_lost := List.map (λ x => x + 2) (team_scores.take lost_matches)
  let opponent_scores_won := List.map (λ x => (x + 2) / 3) (team_scores.drop lost_matches)
  let total_opponent_score := opponent_scores_lost.sum + opponent_scores_won.sum
  
  n = 12 ∧
  team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] ∧
  total_opponent_score = 54

theorem cricket_theorem : 
  ∃ (team_scores : List Nat), cricket_problem team_scores :=
sorry

end cricket_theorem_l1124_112488


namespace ten_stairs_ways_l1124_112448

def stair_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | m + 4 => stair_ways m + stair_ways (m + 1) + stair_ways (m + 2) + stair_ways (m + 3)

theorem ten_stairs_ways : stair_ways 10 = 401 := by
  sorry

end ten_stairs_ways_l1124_112448


namespace train_speed_calculation_l1124_112404

/-- Given a train and tunnel with specified lengths and crossing time, calculate the train's speed in km/hr -/
theorem train_speed_calculation (train_length tunnel_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 415)
  (h2 : tunnel_length = 285)
  (h3 : crossing_time = 40) :
  (train_length + tunnel_length) / crossing_time * 3.6 = 63 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l1124_112404


namespace kyle_spent_one_third_l1124_112461

def dave_money : ℕ := 46
def kyle_initial_money : ℕ := 3 * dave_money - 12
def kyle_remaining_money : ℕ := 84

theorem kyle_spent_one_third : 
  (kyle_initial_money - kyle_remaining_money) / kyle_initial_money = 1/3 := by
  sorry

end kyle_spent_one_third_l1124_112461


namespace a_equals_3y_l1124_112423

theorem a_equals_3y (a b y : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27 * y^3) 
  (h3 : a - b = 3 * y) : 
  a = 3 * y := by
sorry

end a_equals_3y_l1124_112423


namespace c_investment_amount_l1124_112403

/-- A business partnership between C and D -/
structure Business where
  c_investment : ℕ
  d_investment : ℕ
  total_profit : ℕ
  d_profit_share : ℕ

/-- The business scenario as described in the problem -/
def scenario : Business where
  c_investment := 0  -- Unknown, to be proved
  d_investment := 1500
  total_profit := 500
  d_profit_share := 100

/-- Theorem stating C's investment amount -/
theorem c_investment_amount (b : Business) (h1 : b = scenario) :
  b.c_investment = 6000 := by
  sorry

end c_investment_amount_l1124_112403


namespace opposite_of_six_l1124_112470

-- Define the concept of opposite for real numbers
def opposite (x : ℝ) : ℝ := -x

-- Theorem stating that the opposite of 6 is -6
theorem opposite_of_six : opposite 6 = -6 := by
  sorry

end opposite_of_six_l1124_112470


namespace dalton_needs_four_dollars_l1124_112459

/-- The amount of additional money Dalton needs to buy all items -/
def additional_money_needed (jump_rope_cost board_game_cost ball_cost saved_allowance uncle_gift : ℕ) : ℕ :=
  let total_cost := jump_rope_cost + board_game_cost + ball_cost
  let available_money := saved_allowance + uncle_gift
  if total_cost > available_money then
    total_cost - available_money
  else
    0

/-- Theorem stating that Dalton needs $4 more to buy all items -/
theorem dalton_needs_four_dollars : 
  additional_money_needed 7 12 4 6 13 = 4 := by
  sorry

end dalton_needs_four_dollars_l1124_112459


namespace tangent_perpendicular_to_line_l1124_112445

open Real

theorem tangent_perpendicular_to_line (a : ℝ) : 
  let f (x : ℝ) := (2 - cos x) / sin x
  let x₀ : ℝ := π / 2
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  (y₀ = 2) → (m * (-1/a) = -1) → a = 1 := by
sorry

end tangent_perpendicular_to_line_l1124_112445


namespace sum_O_eq_321_l1124_112480

/-- O(n) represents the sum of odd digits in number n -/
def O (n : ℕ) : ℕ := sorry

/-- The sum of O(n) from 1 to 75 -/
def sum_O : ℕ := (Finset.range 75).sum (λ n => O (n + 1))

/-- Theorem: The sum of O(n) from 1 to 75 equals 321 -/
theorem sum_O_eq_321 : sum_O = 321 := by sorry

end sum_O_eq_321_l1124_112480


namespace domino_coverage_l1124_112427

theorem domino_coverage (n k : ℕ+) :
  (∃ (coverage : Fin n × Fin n → Fin k × Bool),
    (∀ (i j : Fin n), ∃ (x : Fin k) (b : Bool),
      coverage (i, j) = (x, b) ∧
      (b = true → coverage (i, j.succ) = (x, false)) ∧
      (b = false → coverage (i.succ, j) = (x, true))))
  ↔ k ∣ n := by sorry

end domino_coverage_l1124_112427


namespace lemonade_ratio_l1124_112484

/-- Given that 30 lemons make 25 gallons of lemonade, prove that 12 lemons make 10 gallons -/
theorem lemonade_ratio (lemons : ℕ) (gallons : ℕ) 
  (h : (30 : ℚ) / 25 = lemons / gallons) (h10 : gallons = 10) : lemons = 12 := by
  sorry

end lemonade_ratio_l1124_112484


namespace meaningful_fraction_l1124_112476

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 2)) ↔ x ≠ 2 := by sorry

end meaningful_fraction_l1124_112476


namespace c_percentage_less_than_d_l1124_112443

def full_marks : ℕ := 500
def d_marks : ℕ := (80 * full_marks) / 100
def a_marks : ℕ := 360

def b_marks : ℕ := a_marks * 100 / 90
def c_marks : ℕ := b_marks * 100 / 125

theorem c_percentage_less_than_d :
  (d_marks - c_marks) * 100 / d_marks = 20 := by sorry

end c_percentage_less_than_d_l1124_112443


namespace square_plus_minus_one_divisible_by_five_l1124_112407

theorem square_plus_minus_one_divisible_by_five (a : ℤ) : 
  ¬(5 ∣ a) → (5 ∣ (a^2 + 1)) ∨ (5 ∣ (a^2 - 1)) :=
by sorry

end square_plus_minus_one_divisible_by_five_l1124_112407


namespace distance_between_anastasia_and_bananastasia_l1124_112416

/-- The speed of sound in meters per second -/
def speed_of_sound : ℝ := 343

/-- The time difference in seconds between hearing Anastasia and Bananastasia when they yell simultaneously -/
def simultaneous_time_diff : ℝ := 5

/-- The time difference in seconds between hearing Bananastasia and Anastasia when Bananastasia yells first -/
def sequential_time_diff : ℝ := 5

/-- The distance between Anastasia and Bananastasia in meters -/
def distance : ℝ := 1715

theorem distance_between_anastasia_and_bananastasia :
  ∀ (d : ℝ),
  (d / speed_of_sound = simultaneous_time_diff) ∧
  (2 * d / speed_of_sound - d / speed_of_sound = sequential_time_diff) →
  d = distance := by
  sorry

end distance_between_anastasia_and_bananastasia_l1124_112416


namespace cylinder_radius_problem_l1124_112446

theorem cylinder_radius_problem (h : ℝ) (r : ℝ) :
  h = 2 →
  (π * (r + 5)^2 * h - π * r^2 * h = π * r^2 * (h + 4) - π * r^2 * h) →
  r = (5 + 5 * Real.sqrt 3) / 2 := by
  sorry

end cylinder_radius_problem_l1124_112446


namespace factory_production_excess_l1124_112420

theorem factory_production_excess (monthly_plan : ℝ) :
  let january_production := 1.05 * monthly_plan
  let february_production := 1.04 * january_production
  let two_month_plan := 2 * monthly_plan
  let total_production := january_production + february_production
  (total_production - two_month_plan) / two_month_plan = 0.071 := by
sorry

end factory_production_excess_l1124_112420


namespace cosine_equation_solutions_l1124_112405

open Real

theorem cosine_equation_solutions :
  let f := fun (x : ℝ) => 3 * (cos x)^4 - 6 * (cos x)^3 + 4 * (cos x)^2 - 1
  ∃! (s : Finset ℝ), s.card = 5 ∧ (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2*π ∧ f x = 0) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2*π ∧ f x = 0 → x ∈ s) :=
by sorry

end cosine_equation_solutions_l1124_112405


namespace expression_simplification_and_evaluation_l1124_112497

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 2) :
  (1 - 1 / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1)) = 2 :=
by sorry

end expression_simplification_and_evaluation_l1124_112497


namespace letians_estimate_l1124_112400

/-- Given x and y are positive real numbers with x > y, and z and w are small positive real numbers with z > w,
    prove that (x + z) - (y - w) > x - y. -/
theorem letians_estimate (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
    (hz : z > 0) (hw : w > 0) (hzw : z > w) : 
  (x + z) - (y - w) > x - y := by
  sorry

end letians_estimate_l1124_112400


namespace inequality_theorem_l1124_112472

theorem inequality_theorem (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, x * (deriv f x) > -f x) (a b : ℝ) (hab : a > b) : 
  a * f a > b * f b := by
  sorry

end inequality_theorem_l1124_112472


namespace range_of_a_l1124_112465

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) → 
  -3/5 < a ∧ a ≤ 1 :=
by sorry

end range_of_a_l1124_112465


namespace square_difference_equality_l1124_112436

theorem square_difference_equality : (15 + 12)^2 - (15 - 12)^2 = 720 := by
  sorry

end square_difference_equality_l1124_112436


namespace total_food_is_point_nine_l1124_112409

/-- The amount of cat food Jake needs to serve each day for one cat -/
def food_for_one_cat : ℝ := 0.5

/-- The extra amount of cat food needed for the second cat -/
def extra_food_for_second_cat : ℝ := 0.4

/-- The total amount of cat food Jake needs to serve each day for two cats -/
def total_food_for_two_cats : ℝ := food_for_one_cat + extra_food_for_second_cat

theorem total_food_is_point_nine :
  total_food_for_two_cats = 0.9 := by
  sorry

end total_food_is_point_nine_l1124_112409


namespace triangle_angle_calculation_l1124_112402

theorem triangle_angle_calculation (A B C : ℝ) :
  A + B + C = 180 →
  B = 4 * A →
  C - B = 27 →
  A = 17 ∧ B = 68 ∧ C = 95 := by
  sorry

end triangle_angle_calculation_l1124_112402


namespace product_of_roots_l1124_112454

theorem product_of_roots (p q r : ℂ) : 
  (3 * p^3 - 9 * p^2 + 5 * p - 15 = 0) →
  (3 * q^3 - 9 * q^2 + 5 * q - 15 = 0) →
  (3 * r^3 - 9 * r^2 + 5 * r - 15 = 0) →
  p * q * r = 5 := by sorry

end product_of_roots_l1124_112454


namespace multiples_of_seven_ending_in_five_l1124_112491

/-- The count of positive multiples of 7 less than 2000 that end with the digit 5 -/
theorem multiples_of_seven_ending_in_five (n : ℕ) : 
  (∃ k : ℕ, n = 7 * k ∧ n < 2000 ∧ n % 10 = 5) ↔ n ∈ Finset.range 29 :=
sorry

end multiples_of_seven_ending_in_five_l1124_112491


namespace pauls_crayons_l1124_112431

theorem pauls_crayons (birthday_crayons : Float) (school_year_crayons : Float) (neighbor_crayons : Float)
  (h1 : birthday_crayons = 479.0)
  (h2 : school_year_crayons = 134.0)
  (h3 : neighbor_crayons = 256.0) :
  birthday_crayons + school_year_crayons + neighbor_crayons = 869.0 := by
  sorry

end pauls_crayons_l1124_112431


namespace last_digit_of_2021_2021_l1124_112406

-- Define the table size
def n : Nat := 2021

-- Define the cell value function
def cellValue (x y : Nat) : Nat :=
  if x = 1 ∧ y = 1 then 0
  else 2^(x + y - 2) - 1

-- State the theorem
theorem last_digit_of_2021_2021 :
  (cellValue n n) % 10 = 5 := by sorry

end last_digit_of_2021_2021_l1124_112406


namespace problem_1_l1124_112422

theorem problem_1 (x y : ℝ) (h : |x + 2| + |y - 3| = 0) : x - y + 1 = -4 := by
  sorry

end problem_1_l1124_112422


namespace zoo_field_trip_count_l1124_112487

/-- Represents the number of individuals at the zoo during the field trip -/
def ZooFieldTrip : Type :=
  { n : ℕ // n ≤ 100 }

/-- The initial class size -/
def initial_class_size : ℕ := 10

/-- The number of parents who volunteered as chaperones -/
def parent_chaperones : ℕ := 5

/-- The number of teachers who joined -/
def teachers : ℕ := 2

/-- The number of students who left -/
def students_left : ℕ := 10

/-- The number of chaperones who left -/
def chaperones_left : ℕ := 2

/-- Function to calculate the final number of individuals at the zoo -/
def final_zoo_count (init_class : ℕ) (parents : ℕ) (teachers : ℕ) (students_gone : ℕ) (chaperones_gone : ℕ) : ZooFieldTrip :=
  ⟨2 * init_class + parents + teachers - students_gone - chaperones_gone, by sorry⟩

/-- Theorem stating that the final number of individuals at the zoo is 15 -/
theorem zoo_field_trip_count :
  (final_zoo_count initial_class_size parent_chaperones teachers students_left chaperones_left).val = 15 := by
  sorry

end zoo_field_trip_count_l1124_112487


namespace unique_lcm_gcd_relation_l1124_112469

theorem unique_lcm_gcd_relation : 
  ∃! (n : ℕ), n > 0 ∧ Nat.lcm n 100 = Nat.gcd n 100 + 450 :=
by
  -- The proof goes here
  sorry

end unique_lcm_gcd_relation_l1124_112469


namespace girls_in_algebra_class_l1124_112452

theorem girls_in_algebra_class (total : ℕ) (girls boys : ℕ) : 
  total = 84 →
  girls + boys = total →
  4 * boys = 3 * girls →
  girls = 48 := by
sorry

end girls_in_algebra_class_l1124_112452


namespace inequalities_proof_l1124_112482

theorem inequalities_proof (n : ℕ) (a : ℝ) (h1 : n ≥ 1) (h2 : a > 0) :
  2^(n-1) ≤ n! ∧ 
  n! ≤ n^n ∧ 
  (n+3)^2 ≤ 2^(n+3) ∧ 
  1 + n * a ≤ (1+a)^n := by
  sorry


end inequalities_proof_l1124_112482


namespace pole_height_pole_height_is_8_5_l1124_112481

/-- The height of a pole given specific cable and person measurements -/
theorem pole_height (cable_length : ℝ) (cable_ground_distance : ℝ) 
  (person_height : ℝ) (person_distance : ℝ) : ℝ :=
  cable_length * person_height / (cable_ground_distance - person_distance)

/-- Proof that a pole is 8.5 meters tall given specific measurements -/
theorem pole_height_is_8_5 :
  pole_height 5 5 1.7 4 = 8.5 := by
  sorry

end pole_height_pole_height_is_8_5_l1124_112481


namespace michaels_house_paint_area_l1124_112496

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total area to be painted in a house -/
def totalPaintArea (numRooms : ℕ) (dimensions : RoomDimensions) (windowDoorArea : ℝ) : ℝ :=
  let wallArea := 2 * (dimensions.length * dimensions.height + dimensions.width * dimensions.height)
  let paintableArea := wallArea - windowDoorArea
  numRooms * paintableArea

/-- Theorem: The total area to be painted in Michael's house is 1600 square feet -/
theorem michaels_house_paint_area :
  let dimensions : RoomDimensions := ⟨14, 11, 9⟩
  totalPaintArea 4 dimensions 50 = 1600 := by sorry

end michaels_house_paint_area_l1124_112496


namespace solution_set_quadratic_inequality_l1124_112415

theorem solution_set_quadratic_inequality (x : ℝ) :
  2 * x + 3 - x^2 > 0 ↔ -1 < x ∧ x < 3 := by
  sorry

end solution_set_quadratic_inequality_l1124_112415


namespace unique_quadratic_pair_l1124_112441

theorem unique_quadratic_pair : ∃! (b c : ℕ+), 
  (∃! x : ℝ, x^2 + b*x + c = 0) ∧ 
  (∃! x : ℝ, x^2 + c*x + b = 0) := by
sorry

end unique_quadratic_pair_l1124_112441


namespace translation_result_l1124_112413

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally -/
def translateHorizontal (p : Point) (dx : ℝ) : Point :=
  ⟨p.x + dx, p.y⟩

/-- Translates a point vertically -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  ⟨p.x, p.y + dy⟩

/-- The initial point M -/
def M : Point := ⟨2, 5⟩

/-- The resulting point after translations -/
def resultingPoint : Point :=
  translateVertical (translateHorizontal M (-2)) (-3)

theorem translation_result :
  resultingPoint = ⟨0, 2⟩ := by sorry

end translation_result_l1124_112413


namespace base_n_representation_of_b_l1124_112499

/-- Represents a number in base n -/
def BaseN (n : ℕ) (x : ℕ) : Prop :=
  ∃ (d₁ d₀ : ℕ), x = d₁ * n + d₀ ∧ d₀ < n

theorem base_n_representation_of_b
  (n : ℕ) (a b : ℕ) 
  (h_n : n > 9)
  (h_root : n^2 - a*n + b = 0)
  (h_a : BaseN n 19) :
  BaseN n 90 := by
  sorry

end base_n_representation_of_b_l1124_112499


namespace other_divisor_problem_l1124_112462

theorem other_divisor_problem (n : Nat) (d1 d2 : Nat) : 
  (n = 386) →
  (d1 = 35) →
  (n % d1 = 1) →
  (n % d2 = 1) →
  (∀ m : Nat, m < n → (m % d1 = 1 ∧ m % d2 = 1) → False) →
  (d2 = 11) := by
  sorry

end other_divisor_problem_l1124_112462


namespace problem_one_problem_two_l1124_112474

-- Problem 1
theorem problem_one (a b c : ℝ) (ha : |a| = 1) (hb : |b| = 2) (hc : |c| = 3) (horder : a > b ∧ b > c) :
  a + b - c = 2 ∨ a + b - c = 0 := by sorry

-- Problem 2
theorem problem_two (a b c d : ℚ) (hab : |a - b| ≤ 9) (hcd : |c - d| ≤ 16) (habcd : |a - b - c + d| = 25) :
  |b - a| - |d - c| = -7 := by sorry

end problem_one_problem_two_l1124_112474


namespace lauren_pencils_l1124_112414

/-- Proves that Lauren received 6 pencils given the conditions of the problem -/
theorem lauren_pencils (initial_pencils : ℕ) (remaining_pencils : ℕ) (matt_extra : ℕ) :
  initial_pencils = 24 →
  remaining_pencils = 9 →
  matt_extra = 3 →
  ∃ (lauren_pencils : ℕ),
    lauren_pencils + (lauren_pencils + matt_extra) = initial_pencils - remaining_pencils ∧
    lauren_pencils = 6 :=
by sorry

end lauren_pencils_l1124_112414


namespace bus_tour_tickets_l1124_112475

/-- Proves that the number of regular tickets sold is 41 -/
theorem bus_tour_tickets (total_tickets : ℕ) (senior_price regular_price : ℚ) (total_sales : ℚ) 
  (h1 : total_tickets = 65)
  (h2 : senior_price = 10)
  (h3 : regular_price = 15)
  (h4 : total_sales = 855) :
  ∃ (senior_tickets regular_tickets : ℕ),
    senior_tickets + regular_tickets = total_tickets ∧
    senior_price * senior_tickets + regular_price * regular_tickets = total_sales ∧
    regular_tickets = 41 := by
  sorry

end bus_tour_tickets_l1124_112475


namespace nancy_tuition_ratio_l1124_112449

/-- Calculates the ratio of student loan to scholarship for Nancy's university tuition --/
theorem nancy_tuition_ratio :
  let tuition : ℚ := 22000
  let parents_contribution : ℚ := tuition / 2
  let scholarship : ℚ := 3000
  let work_hours : ℚ := 200
  let hourly_rate : ℚ := 10
  let work_earnings : ℚ := work_hours * hourly_rate
  let total_available : ℚ := parents_contribution + scholarship + work_earnings
  let loan_amount : ℚ := tuition - total_available
  loan_amount / scholarship = 2 := by sorry

end nancy_tuition_ratio_l1124_112449


namespace mario_flower_count_l1124_112428

/-- The number of flowers on Mario's first hibiscus plant -/
def first_plant_flowers : ℕ := 2

/-- The number of flowers on Mario's second hibiscus plant -/
def second_plant_flowers : ℕ := 2 * first_plant_flowers

/-- The number of flowers on Mario's third hibiscus plant -/
def third_plant_flowers : ℕ := 4 * second_plant_flowers

/-- The total number of flowers on all of Mario's hibiscus plants -/
def total_flowers : ℕ := first_plant_flowers + second_plant_flowers + third_plant_flowers

theorem mario_flower_count : total_flowers = 22 := by
  sorry

end mario_flower_count_l1124_112428


namespace expression_evaluation_l1124_112463

theorem expression_evaluation :
  (3 : ℚ)^3010 * 2^3008 / 6^3009 = 3/2 := by
  sorry

end expression_evaluation_l1124_112463


namespace percentage_difference_l1124_112426

theorem percentage_difference (x y : ℝ) (h : x = 4 * y) :
  (x - y) / x * 100 = 75 := by
  sorry

end percentage_difference_l1124_112426


namespace min_m_value_l1124_112435

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(|x - a|)

theorem min_m_value (a : ℝ) :
  (∀ x, f a (1 + x) = f a (1 - x)) →
  (∃ m, ∀ x y, m ≤ x → x < y → f a x < f a y) →
  (∀ m', (∀ x y, m' ≤ x → x < y → f a x < f a y) → 1 ≤ m') :=
sorry

end min_m_value_l1124_112435


namespace license_plate_count_l1124_112473

/-- The number of letters in the Rotokas alphabet -/
def rotokas_alphabet_size : ℕ := 12

/-- The set of allowed first letters -/
def first_letters : Finset Char := {'G', 'K', 'P'}

/-- The required last letter -/
def last_letter : Char := 'T'

/-- The forbidden letter -/
def forbidden_letter : Char := 'R'

/-- The length of the license plate -/
def license_plate_length : ℕ := 5

/-- Calculates the number of valid license plates -/
def count_license_plates : ℕ :=
  first_letters.card * (rotokas_alphabet_size - 5) * (rotokas_alphabet_size - 6) * (rotokas_alphabet_size - 7)

theorem license_plate_count :
  count_license_plates = 630 :=
sorry

end license_plate_count_l1124_112473


namespace pablo_blocks_sum_l1124_112438

/-- The number of blocks in Pablo's toy block stacks -/
def pablo_blocks : ℕ → ℕ
| 0 => 5  -- First stack
| 1 => pablo_blocks 0 + 2  -- Second stack
| 2 => pablo_blocks 1 - 5  -- Third stack
| 3 => pablo_blocks 2 + 5  -- Fourth stack
| _ => 0  -- No more stacks

/-- The total number of blocks used by Pablo -/
def total_blocks : ℕ := pablo_blocks 0 + pablo_blocks 1 + pablo_blocks 2 + pablo_blocks 3

theorem pablo_blocks_sum : total_blocks = 21 := by
  sorry

end pablo_blocks_sum_l1124_112438


namespace smallest_possible_b_l1124_112479

theorem smallest_possible_b : 
  ∃ (b : ℝ), ∀ (a : ℝ), 
    (2 < a ∧ a < b) → 
    (2 + a ≤ b) → 
    (1 / a + 1 / b ≤ 1 / 2) → 
    (b = 3 + Real.sqrt 5) ∧
    (∀ (b' : ℝ), 
      (∃ (a' : ℝ), 
        (2 < a' ∧ a' < b') ∧ 
        (2 + a' ≤ b') ∧ 
        (1 / a' + 1 / b' ≤ 1 / 2)) → 
      b ≤ b') :=
sorry

end smallest_possible_b_l1124_112479


namespace B_subset_A_iff_A_disjoint_B_iff_l1124_112477

/-- Set A defined as {x | -3 < 2x-1 < 7} -/
def A : Set ℝ := {x | -3 < 2*x-1 ∧ 2*x-1 < 7}

/-- Set B defined as {x | 2a ≤ x ≤ a+3} -/
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a+3}

/-- Theorem stating the conditions for B to be a subset of A -/
theorem B_subset_A_iff (a : ℝ) : B a ⊆ A ↔ -1/2 < a ∧ a < 1 := by sorry

/-- Theorem stating the conditions for A and B to be disjoint -/
theorem A_disjoint_B_iff (a : ℝ) : A ∩ B a = ∅ ↔ a ≤ -4 ∨ a ≥ 2 := by sorry

end B_subset_A_iff_A_disjoint_B_iff_l1124_112477


namespace fraction_simplification_l1124_112485

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hxy : y - 1/x ≠ 0) : 
  (x - 1/y) / (y - 1/x) = x / y := by
  sorry

end fraction_simplification_l1124_112485


namespace find_m_value_l1124_112437

theorem find_m_value (n m : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + n) = x^2 + m*x - 21) → m = -4 :=
by sorry

end find_m_value_l1124_112437


namespace gcd_1821_2993_l1124_112464

theorem gcd_1821_2993 : Nat.gcd 1821 2993 = 1 := by
  sorry

end gcd_1821_2993_l1124_112464


namespace max_garden_area_l1124_112457

/-- The maximum area of a rectangular garden with 150 feet of fencing and natural number side lengths -/
theorem max_garden_area : 
  ∃ (l w : ℕ), 
    2 * (l + w) = 150 ∧ 
    l * w = 1406 ∧
    ∀ (a b : ℕ), 2 * (a + b) = 150 → a * b ≤ 1406 := by
  sorry

end max_garden_area_l1124_112457


namespace swamp_ecosystem_flies_eaten_l1124_112494

/-- Represents the number of flies eaten daily in a swamp ecosystem -/
def flies_eaten_daily (gharials : ℕ) (fish_per_gharial : ℕ) (frogs_per_fish : ℕ) (flies_per_frog : ℕ) : ℕ :=
  gharials * fish_per_gharial * frogs_per_fish * flies_per_frog

/-- Theorem stating the number of flies eaten daily in the given swamp ecosystem -/
theorem swamp_ecosystem_flies_eaten :
  flies_eaten_daily 9 15 8 30 = 32400 := by
  sorry

end swamp_ecosystem_flies_eaten_l1124_112494


namespace anton_number_is_729_l1124_112456

-- Define a three-digit number type
def ThreeDigitNumber := {n : ℕ // n ≥ 100 ∧ n ≤ 999}

-- Define a function to check if two numbers match in exactly one digit place
def matchesOneDigit (a b : ThreeDigitNumber) : Prop :=
  (a.val / 100 = b.val / 100 ∧ a.val % 100 ≠ b.val % 100) ∨
  (a.val % 100 / 10 = b.val % 100 / 10 ∧ a.val / 100 ≠ b.val / 100 ∧ a.val % 10 ≠ b.val % 10) ∨
  (a.val % 10 = b.val % 10 ∧ a.val / 10 ≠ b.val / 10)

theorem anton_number_is_729 (x : ThreeDigitNumber) :
  matchesOneDigit x ⟨109, by norm_num⟩ ∧
  matchesOneDigit x ⟨704, by norm_num⟩ ∧
  matchesOneDigit x ⟨124, by norm_num⟩ →
  x = ⟨729, by norm_num⟩ := by
  sorry

end anton_number_is_729_l1124_112456


namespace right_triangle_area_l1124_112490

theorem right_triangle_area (a b c m : ℝ) : 
  a = 10 →                -- One leg is 10
  m = 13 →                -- Shortest median is 13
  m^2 = (2*a^2 + 2*c^2 - b^2) / 4 →  -- Apollonius's theorem
  a^2 + b^2 = c^2 →       -- Pythagorean theorem
  a * b / 2 = 10 * Real.sqrt 69 :=   -- Area of the triangle
by sorry

end right_triangle_area_l1124_112490


namespace max_correct_is_38_l1124_112447

/-- Represents the scoring system and result of a multiple-choice test -/
structure TestScoring where
  total_questions : ℕ
  correct_points : ℤ
  blank_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correct answers possible given a TestScoring -/
def max_correct_answers (ts : TestScoring) : ℕ :=
  sorry

/-- Theorem stating that for the given test conditions, the maximum number of correct answers is 38 -/
theorem max_correct_is_38 : 
  let ts : TestScoring := {
    total_questions := 60,
    correct_points := 5,
    blank_points := 0,
    incorrect_points := -2,
    total_score := 150
  }
  max_correct_answers ts = 38 := by
  sorry

end max_correct_is_38_l1124_112447


namespace number_ratio_l1124_112451

/-- Given three numbers satisfying specific conditions, prove their ratio -/
theorem number_ratio (A B C : ℚ) : 
  A + B + C = 98 →
  B = 30 →
  8 * B = 5 * C →
  A * 3 = B * 2 := by
sorry

end number_ratio_l1124_112451


namespace cupcakes_leftover_l1124_112401

/-- Proves that given 40 cupcakes, after distributing to two classes and four individuals, 2 cupcakes remain. -/
theorem cupcakes_leftover (total : ℕ) (class1 : ℕ) (class2 : ℕ) (additional : ℕ) : 
  total = 40 → class1 = 18 → class2 = 16 → additional = 4 → 
  total - (class1 + class2 + additional) = 2 := by
sorry

end cupcakes_leftover_l1124_112401


namespace inscribed_cube_properties_l1124_112471

/-- A cube inscribed in a hemisphere -/
structure InscribedCube (R : ℝ) where
  -- The edge length of the cube
  a : ℝ
  -- The distance from the center of the hemisphere base to a vertex of the square face
  r : ℝ
  -- Four vertices of the cube are on the surface of the hemisphere
  vertices_on_surface : a ^ 2 + r ^ 2 = R ^ 2
  -- Four vertices of the cube are on the circular boundary of the hemisphere's base
  vertices_on_base : r = a * (Real.sqrt 2) / 2

/-- The edge length and distance properties of a cube inscribed in a hemisphere -/
theorem inscribed_cube_properties (R : ℝ) (h : R > 0) :
  ∃ (cube : InscribedCube R),
    cube.a = R * Real.sqrt (2/3) ∧
    cube.r = R / Real.sqrt 3 :=
by sorry

end inscribed_cube_properties_l1124_112471


namespace arc_length_unit_circle_30_degrees_l1124_112468

theorem arc_length_unit_circle_30_degrees :
  let r : ℝ := 1  -- radius of unit circle
  let θ : ℝ := 30 -- central angle in degrees
  let l : ℝ := θ * π * r / 180 -- arc length formula
  l = π / 6 := by
  sorry

end arc_length_unit_circle_30_degrees_l1124_112468


namespace union_complement_problem_l1124_112433

def U : Finset Nat := {1, 2, 3, 4, 5}
def A : Finset Nat := {2, 3, 4}
def B : Finset Nat := {2, 5}

theorem union_complement_problem : B ∪ (U \ A) = {1, 2, 5} := by sorry

end union_complement_problem_l1124_112433


namespace grandfather_gift_problem_l1124_112455

theorem grandfather_gift_problem (x y : ℕ) : 
  x + y = 30 → 
  5 * x * (x + 1) + 5 * y * (y + 1) = 2410 → 
  (x = 16 ∧ y = 14) ∨ (x = 14 ∧ y = 16) := by
sorry

end grandfather_gift_problem_l1124_112455


namespace line_tangent_to_circle_l1124_112460

theorem line_tangent_to_circle (k : ℝ) : 
  (∀ x y : ℝ, y = k * (x + Real.sqrt 3) → x^2 + (y - 1)^2 = 1 → 
    ∀ x' y' : ℝ, y' = k * (x' + Real.sqrt 3) → x'^2 + (y' - 1)^2 ≥ 1) →
  k = Real.sqrt 3 ∨ k = 0 := by
  sorry

end line_tangent_to_circle_l1124_112460


namespace compound_interest_problem_l1124_112418

/-- Calculate the compound interest given principal, rate, and time -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Calculate the total amount returned after compound interest -/
def total_amount (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

theorem compound_interest_problem (P : ℝ) :
  compound_interest P 0.05 2 = 492 →
  total_amount P 492 = 5292 := by
  sorry

end compound_interest_problem_l1124_112418


namespace set_membership_problem_l1124_112489

theorem set_membership_problem (n : ℕ) (x y z w : ℕ) 
  (hn : n ≥ 4)
  (hx : x ∈ Finset.range n)
  (hy : y ∈ Finset.range n)
  (hz : z ∈ Finset.range n)
  (hw : w ∈ Finset.range n)
  (hS : Set.Mem (x, y, z) S ∧ Set.Mem (z, w, x) S) :
  Set.Mem (y, z, w) S ∧ Set.Mem (x, y, w) S :=
by
  sorry
where
  X : Finset ℕ := Finset.range n
  S : Set (ℕ × ℕ × ℕ) := 
    {p | p.1 ∈ X ∧ p.2.1 ∈ X ∧ p.2.2 ∈ X ∧
      ((p.1 < p.2.1 ∧ p.2.1 < p.2.2) ∨
       (p.2.1 < p.2.2 ∧ p.2.2 < p.1) ∨
       (p.2.2 < p.1 ∧ p.1 < p.2.1)) ∧
      ¬((p.1 < p.2.1 ∧ p.2.1 < p.2.2) ∧
        (p.2.1 < p.2.2 ∧ p.2.2 < p.1) ∧
        (p.2.2 < p.1 ∧ p.1 < p.2.1))}

end set_membership_problem_l1124_112489


namespace A_is_uncountable_l1124_112429

-- Define the set A as the closed interval [0, 1]
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- Theorem stating that A is uncountable
theorem A_is_uncountable : ¬ (Countable A) := by
  sorry

end A_is_uncountable_l1124_112429


namespace factorization_problem1_factorization_problem2_l1124_112450

-- Problem 1
theorem factorization_problem1 (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) := by sorry

-- Problem 2
theorem factorization_problem2 (a b x y : ℝ) : 
  a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b) := by sorry

end factorization_problem1_factorization_problem2_l1124_112450


namespace division_result_l1124_112467

theorem division_result : (-1/20) / (-1/4 - 2/5 + 9/10 - 3/2) = 1/25 := by
  sorry

end division_result_l1124_112467


namespace jogger_distance_ahead_l1124_112483

/-- Calculates the distance a jogger is ahead of a train given their speeds, the train's length, and the time it takes for the train to pass the jogger. -/
theorem jogger_distance_ahead (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 210 →
  passing_time = 45 →
  (train_speed - jogger_speed) * passing_time - train_length = 240 :=
by sorry

end jogger_distance_ahead_l1124_112483


namespace parallel_vectors_y_value_l1124_112466

/-- Two vectors are parallel if and only if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (4, y)
  parallel a b → y = 6 := by
sorry

end parallel_vectors_y_value_l1124_112466


namespace f_fixed_point_l1124_112411

-- Define the function f
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- Define the repeated application of f
def repeat_f (p q : ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n+1 => f p q (repeat_f p q n x)

-- State the theorem
theorem f_fixed_point (p q : ℝ) :
  (∀ x ∈ Set.Icc 2 4, |f p q x| ≤ 1/2) →
  repeat_f p q 2017 ((5 - Real.sqrt 11) / 2) = (5 + Real.sqrt 11) / 2 :=
by sorry

end f_fixed_point_l1124_112411


namespace solution_set_inequality_l1124_112439

theorem solution_set_inequality (x : ℝ) :
  (x - 4) * (x + 1) > 0 ↔ x > 4 ∨ x < -1 := by sorry

end solution_set_inequality_l1124_112439


namespace trajectory_is_ellipse_l1124_112421

/-- The trajectory of point P(x,y) moving such that its distance from the line x=-4 
    is twice its distance from the fixed point F(-1,0) -/
def trajectory (x y : ℝ) : Prop :=
  let F := ((-1 : ℝ), (0 : ℝ))
  let d := |x + 4|
  let PF := Real.sqrt ((x + 1)^2 + y^2)
  d = 2 * PF ∧ x^2 / 4 + y^2 / 3 = 1

/-- The theorem stating that the trajectory satisfies the ellipse equation -/
theorem trajectory_is_ellipse (x y : ℝ) : 
  trajectory x y ↔ x^2 / 4 + y^2 / 3 = 1 := by
  sorry

end trajectory_is_ellipse_l1124_112421


namespace wages_problem_l1124_112453

/-- Given a sum of money that can pay b's wages for 28 days and both a's and b's wages for 12 days,
    prove that it can pay a's wages for 21 days. -/
theorem wages_problem (S : ℝ) (Wa Wb : ℝ) (S_pays_b_28_days : S = 28 * Wb) 
    (S_pays_both_12_days : S = 12 * (Wa + Wb)) : S = 21 * Wa := by
  sorry

end wages_problem_l1124_112453


namespace trip_cost_equals_bills_cost_l1124_112419

/-- Proves that the cost of Liam's trip to Paris is equal to the cost of his bills. -/
theorem trip_cost_equals_bills_cost (
  monthly_savings : ℕ)
  (savings_duration_years : ℕ)
  (bills_cost : ℕ)
  (money_left_after_bills : ℕ)
  (h1 : monthly_savings = 500)
  (h2 : savings_duration_years = 2)
  (h3 : bills_cost = 3500)
  (h4 : money_left_after_bills = 8500)
  : bills_cost = monthly_savings * savings_duration_years * 12 - money_left_after_bills :=
by sorry

end trip_cost_equals_bills_cost_l1124_112419


namespace conditional_probability_equal_marginal_l1124_112444

-- Define the sample space and events
variable (Ω : Type) [MeasurableSpace Ω]
variable (P : Measure Ω)
variable (A B : Set Ω)

-- Define the probabilities and independence
variable (hA : P A = 1/6)
variable (hB : P B = 1/2)
variable (hInd : P.Independent A B)

-- State the theorem
theorem conditional_probability_equal_marginal
  (h_prob_B_pos : P B > 0) :
  P.condProb A B = 1/6 := by
  sorry

end conditional_probability_equal_marginal_l1124_112444


namespace complement_union_problem_l1124_112442

universe u

def U : Set ℕ := {1, 2, 3, 4}

theorem complement_union_problem (A B : Set ℕ) 
  (h1 : (U \ A) ∩ B = {1})
  (h2 : A ∩ B = {3})
  (h3 : (U \ A) ∩ (U \ B) = {2}) :
  U \ (A ∪ B) = {2} := by
  sorry

end complement_union_problem_l1124_112442


namespace students_liking_sports_l1124_112410

theorem students_liking_sports (basketball : Finset ℕ) (cricket : Finset ℕ) 
  (h1 : basketball.card = 7)
  (h2 : cricket.card = 5)
  (h3 : (basketball ∩ cricket).card = 3) :
  (basketball ∪ cricket).card = 9 := by
  sorry

end students_liking_sports_l1124_112410


namespace diagonal_intersection_probability_l1124_112492

theorem diagonal_intersection_probability (n : ℕ) (h : n > 0) :
  let vertices := 2 * n + 1
  let total_diagonals := vertices * (vertices - 3) / 2
  let intersecting_diagonals := vertices.choose 4
  intersecting_diagonals / (total_diagonals.choose 2 : ℚ) = 
    n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) := by
  sorry

end diagonal_intersection_probability_l1124_112492


namespace convex_polyhedron_structure_l1124_112432

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  convex : Bool

/-- Represents a face of a polyhedron -/
structure Face where
  sides : Nat

/-- Represents a vertex of a polyhedron -/
structure Vertex where
  edges : Nat

/-- Definition of a convex polyhedron with its faces and vertices -/
def ConvexPolyhedronWithFacesAndVertices (p : ConvexPolyhedron) (faces : List Face) (vertices : List Vertex) : Prop :=
  p.convex ∧ faces.length > 0 ∧ vertices.length > 0

/-- Theorem stating that not all faces can have more than 3 sides 
    and not all vertices can have more than 3 edges simultaneously -/
theorem convex_polyhedron_structure 
  (p : ConvexPolyhedron) 
  (faces : List Face) 
  (vertices : List Vertex) 
  (h : ConvexPolyhedronWithFacesAndVertices p faces vertices) :
  ¬(∀ f ∈ faces, f.sides > 3 ∧ ∀ v ∈ vertices, v.edges > 3) :=
by sorry

end convex_polyhedron_structure_l1124_112432


namespace two_digit_number_problem_l1124_112412

theorem two_digit_number_problem (a b : ℕ) : 
  b = 2 * a → 
  (10 * a + b) - (10 * b + a) = 36 → 
  (a + b) - (b - a) = 8 := by
sorry

end two_digit_number_problem_l1124_112412


namespace dog_food_calculation_l1124_112440

theorem dog_food_calculation (num_dogs : ℕ) (total_food_kg : ℕ) (num_days : ℕ) 
  (h1 : num_dogs = 4)
  (h2 : total_food_kg = 14)
  (h3 : num_days = 14) :
  (total_food_kg * 1000) / (num_dogs * num_days) = 250 :=
by
  sorry

end dog_food_calculation_l1124_112440


namespace intersection_of_M_and_N_l1124_112478

-- Define set M
def M : Set ℝ := {y | ∃ x, y = x^2 + 2*x - 3}

-- Define set N
def N : Set ℝ := {x | |x - 2| ≤ 3}

-- Theorem statement
theorem intersection_of_M_and_N :
  M ∩ N = {y | -1 ≤ y ∧ y ≤ 5} := by sorry

end intersection_of_M_and_N_l1124_112478


namespace circle_segment_angle_l1124_112458

theorem circle_segment_angle (r₁ r₂ r₃ : ℝ) (shaded_ratio : ℝ) :
  r₁ = 4 →
  r₂ = 3 →
  r₃ = 2 →
  shaded_ratio = 3 / 5 →
  ∃ θ : ℝ,
    θ > 0 ∧
    θ < π / 2 ∧
    (θ * (r₁^2 + r₂^2 + r₃^2)) / ((π - θ) * (r₁^2 + r₂^2 + r₃^2)) = shaded_ratio ∧
    θ = 3 * π / 8 :=
by sorry

end circle_segment_angle_l1124_112458


namespace pentagon_count_l1124_112424

/-- The number of distinct points on the circumference of a circle -/
def n : ℕ := 15

/-- The number of vertices in each polygon -/
def k : ℕ := 5

/-- The number of distinct convex pentagons that can be formed -/
def num_pentagons : ℕ := Nat.choose n k

theorem pentagon_count :
  num_pentagons = 3003 :=
sorry

end pentagon_count_l1124_112424
