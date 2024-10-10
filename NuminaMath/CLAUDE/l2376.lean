import Mathlib

namespace factorial15_base16_zeros_l2376_237679

/-- The number of trailing zeros in n when expressed in base b -/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- 15 factorial -/
def factorial15 : ℕ :=
  sorry

theorem factorial15_base16_zeros :
  trailingZeros factorial15 16 = 2 := by
  sorry

end factorial15_base16_zeros_l2376_237679


namespace permutation_remainder_l2376_237638

/-- The number of characters in the string -/
def string_length : ℕ := 16

/-- The number of A's in the string -/
def count_A : ℕ := 4

/-- The number of B's in the string -/
def count_B : ℕ := 5

/-- The number of C's in the string -/
def count_C : ℕ := 5

/-- The number of D's in the string -/
def count_D : ℕ := 2

/-- The length of the first segment -/
def first_segment : ℕ := 5

/-- The length of the second segment -/
def second_segment : ℕ := 5

/-- The length of the third segment -/
def third_segment : ℕ := 6

/-- The function to calculate the number of valid permutations -/
def count_permutations : ℕ := sorry

theorem permutation_remainder :
  count_permutations % 1000 = 540 := by sorry

end permutation_remainder_l2376_237638


namespace fifth_root_of_unity_l2376_237604

theorem fifth_root_of_unity (p q r s t m : ℂ) :
  p ≠ 0 →
  p * m^4 + q * m^3 + r * m^2 + s * m + t = 0 →
  q * m^4 + r * m^3 + s * m^2 + t * m + p = 0 →
  m^5 = 1 :=
by sorry

end fifth_root_of_unity_l2376_237604


namespace total_wheels_is_132_l2376_237634

/-- The number of bicycles in the storage area -/
def num_bicycles : ℕ := 24

/-- The number of tricycles in the storage area -/
def num_tricycles : ℕ := 14

/-- The number of unicycles in the storage area -/
def num_unicycles : ℕ := 10

/-- The number of quadbikes in the storage area -/
def num_quadbikes : ℕ := 8

/-- The number of wheels on a bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a tricycle -/
def wheels_per_tricycle : ℕ := 3

/-- The number of wheels on a unicycle -/
def wheels_per_unicycle : ℕ := 1

/-- The number of wheels on a quadbike -/
def wheels_per_quadbike : ℕ := 4

/-- The total number of wheels in the storage area -/
def total_wheels : ℕ := 
  num_bicycles * wheels_per_bicycle +
  num_tricycles * wheels_per_tricycle +
  num_unicycles * wheels_per_unicycle +
  num_quadbikes * wheels_per_quadbike

theorem total_wheels_is_132 : total_wheels = 132 := by
  sorry

end total_wheels_is_132_l2376_237634


namespace contractor_work_problem_l2376_237639

theorem contractor_work_problem (M : ℕ) : 
  (M * 6 = (M - 5) * 10) → M = 13 :=
by
  sorry

end contractor_work_problem_l2376_237639


namespace rabbit_toy_cost_l2376_237663

theorem rabbit_toy_cost (total_cost pet_food_cost cage_cost found_money : ℚ)
  (h1 : total_cost = 24.81)
  (h2 : pet_food_cost = 5.79)
  (h3 : cage_cost = 12.51)
  (h4 : found_money = 1.00) :
  total_cost - (pet_food_cost + cage_cost) + found_money = 7.51 := by
  sorry

end rabbit_toy_cost_l2376_237663


namespace gcd_of_three_numbers_l2376_237630

theorem gcd_of_three_numbers : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end gcd_of_three_numbers_l2376_237630


namespace dvds_bought_online_l2376_237662

theorem dvds_bought_online (total : ℕ) (store : ℕ) (online : ℕ) : 
  total = 10 → store = 8 → online = total - store → online = 2 := by
  sorry

end dvds_bought_online_l2376_237662


namespace arithmetic_sequence_problem_l2376_237678

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a_2 + a_4 = 16, prove a_3 = 8 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_sum : a 2 + a 4 = 16) : 
  a 3 = 8 := by
sorry

end arithmetic_sequence_problem_l2376_237678


namespace paul_picked_72_cans_l2376_237615

/-- The total number of cans Paul picked up -/
def total_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) : ℕ :=
  (saturday_bags + sunday_bags) * cans_per_bag

/-- Theorem stating that Paul picked up 72 cans in total -/
theorem paul_picked_72_cans :
  total_cans 6 3 8 = 72 := by
  sorry

end paul_picked_72_cans_l2376_237615


namespace ratio_of_numbers_l2376_237675

theorem ratio_of_numbers (A B C : ℝ) (k : ℝ) 
  (h1 : A = k * B)
  (h2 : A = 3 * C)
  (h3 : (A + B + C) / 3 = 88)
  (h4 : A - C = 96) :
  A / B = 2 := by
  sorry

end ratio_of_numbers_l2376_237675


namespace right_triangle_trig_identity_l2376_237649

theorem right_triangle_trig_identity (A B C : ℝ) (h1 : 0 < A) (h2 : A < π / 2) :
  B = π / 2 →
  3 * Real.sin A = 4 * Real.cos A + Real.tan A →
  Real.sin A = 2 * Real.sqrt 2 / 3 := by
sorry

end right_triangle_trig_identity_l2376_237649


namespace house_cleaning_time_l2376_237603

/-- Proves that given John cleans the entire house in 6 hours and Nick takes 3 times as long as John to clean half the house, the time it takes for them to clean the entire house together is 3.6 hours. -/
theorem house_cleaning_time (john_time nick_time combined_time : ℝ) : 
  john_time = 6 → 
  nick_time = 3 * (john_time / 2) → 
  combined_time = 18 / 5 → 
  combined_time = 3.6 :=
by sorry

end house_cleaning_time_l2376_237603


namespace parallel_line_plane_condition_l2376_237620

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (subset_of : Line → Plane → Prop)

-- State the theorem
theorem parallel_line_plane_condition
  (m n : Line) (α : Plane)
  (h1 : subset_of n α)
  (h2 : ¬ subset_of m α) :
  (∀ m n, parallel_lines m n → parallel_line_plane m α) ∧
  ¬(∀ m α, parallel_line_plane m α → parallel_lines m n) :=
sorry

end parallel_line_plane_condition_l2376_237620


namespace parallel_vectors_k_value_l2376_237607

theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) :
  a = (2, 1) →
  b = (-1, k) →
  (∃ (t : ℝ), t ≠ 0 ∧ a.1 * t = b.1 ∧ a.2 * t = b.2) →
  k = -1/2 := by
sorry

end parallel_vectors_k_value_l2376_237607


namespace solve_linear_equation_l2376_237670

theorem solve_linear_equation (x : ℝ) (h : 4 * x + 12 = 48) : x = 9 := by
  sorry

end solve_linear_equation_l2376_237670


namespace eighth_roll_last_probability_l2376_237613

/-- The probability of the 8th roll being the last roll when rolling a standard six-sided die 
    until getting the same number on consecutive rolls -/
def prob_eighth_roll_last : ℚ := (5^6 : ℚ) / (6^7 : ℚ)

/-- The number of sides on a standard die -/
def standard_die_sides : ℕ := 6

/-- Theorem stating that the probability of the 8th roll being the last roll is correct -/
theorem eighth_roll_last_probability : 
  prob_eighth_roll_last = (5^6 : ℚ) / (6^7 : ℚ) :=
by sorry

end eighth_roll_last_probability_l2376_237613


namespace map_distance_to_actual_l2376_237671

/-- Given a map scale and a distance on the map, calculate the actual distance in kilometers. -/
theorem map_distance_to_actual (scale : ℚ) (map_distance : ℚ) :
  scale = 200000 →
  map_distance = 3.5 →
  (map_distance * scale) / 100000 = 7 := by
  sorry

end map_distance_to_actual_l2376_237671


namespace total_potatoes_l2376_237682

theorem total_potatoes (nancy_potatoes sandy_potatoes andy_potatoes : ℕ)
  (h1 : nancy_potatoes = 6)
  (h2 : sandy_potatoes = 7)
  (h3 : andy_potatoes = 9) :
  nancy_potatoes + sandy_potatoes + andy_potatoes = 22 :=
by sorry

end total_potatoes_l2376_237682


namespace hard_candy_coloring_is_30_l2376_237612

/-- The amount of food colouring used for each lollipop in milliliters -/
def lollipop_coloring : ℕ := 8

/-- The number of lollipops made in a day -/
def lollipops_made : ℕ := 150

/-- The number of hard candies made in a day -/
def hard_candies_made : ℕ := 20

/-- The total amount of food colouring used in a day in milliliters -/
def total_coloring : ℕ := 1800

/-- The amount of food colouring needed for each hard candy in milliliters -/
def hard_candy_coloring : ℕ := (total_coloring - lollipop_coloring * lollipops_made) / hard_candies_made

theorem hard_candy_coloring_is_30 : hard_candy_coloring = 30 := by
  sorry

end hard_candy_coloring_is_30_l2376_237612


namespace flagpole_shadow_length_l2376_237685

/-- Given a flagpole and a building under similar shadow-casting conditions,
    proves that the length of the shadow cast by the flagpole is 45 meters. -/
theorem flagpole_shadow_length
  (flagpole_height : ℝ)
  (building_height : ℝ)
  (building_shadow : ℝ)
  (h_flagpole : flagpole_height = 18)
  (h_building_height : building_height = 20)
  (h_building_shadow : building_shadow = 50)
  (h_similar_conditions : flagpole_height / building_height = building_shadow / building_shadow) :
  flagpole_height * building_shadow / building_height = 45 :=
sorry

end flagpole_shadow_length_l2376_237685


namespace rotation_90_clockwise_effect_l2376_237618

-- Define the shapes
inductive Shape
  | Pentagon
  | Ellipse
  | Rectangle

-- Define the positions on the circle
structure Position :=
  (angle : ℝ)

-- Define the configuration of shapes on the circle
structure Configuration :=
  (pentagon_pos : Position)
  (ellipse_pos : Position)
  (rectangle_pos : Position)

-- Define the rotation operation
def rotate_90_clockwise (config : Configuration) : Configuration :=
  { pentagon_pos := config.ellipse_pos,
    ellipse_pos := config.rectangle_pos,
    rectangle_pos := config.pentagon_pos }

-- Theorem statement
theorem rotation_90_clockwise_effect (initial_config : Configuration) :
  let final_config := rotate_90_clockwise initial_config
  (final_config.pentagon_pos = initial_config.ellipse_pos) ∧
  (final_config.ellipse_pos = initial_config.rectangle_pos) ∧
  (final_config.rectangle_pos = initial_config.pentagon_pos) :=
by
  sorry


end rotation_90_clockwise_effect_l2376_237618


namespace total_chinese_hours_l2376_237637

/-- The number of hours Ryan spends learning Chinese daily -/
def chinese_hours_per_day : ℕ := 4

/-- The number of days Ryan learns -/
def learning_days : ℕ := 6

/-- Theorem: The total hours Ryan spends learning Chinese over 6 days is 24 hours -/
theorem total_chinese_hours : chinese_hours_per_day * learning_days = 24 := by
  sorry

end total_chinese_hours_l2376_237637


namespace y_coordinate_is_three_l2376_237692

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the second quadrant of the Cartesian coordinate system -/
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- Theorem: If a point is in the second quadrant and its distance to the x-axis is 3, then its y-coordinate is 3 -/
theorem y_coordinate_is_three (P : Point) 
  (h1 : second_quadrant P) 
  (h2 : distance_to_x_axis P = 3) : 
  P.y = 3 := by
  sorry

end y_coordinate_is_three_l2376_237692


namespace b_share_calculation_l2376_237688

def total_cost : ℕ := 520
def hours_a : ℕ := 7
def hours_b : ℕ := 8
def hours_c : ℕ := 11

theorem b_share_calculation : 
  let total_hours := hours_a + hours_b + hours_c
  let cost_per_hour := total_cost / total_hours
  cost_per_hour * hours_b = 160 := by
  sorry

end b_share_calculation_l2376_237688


namespace election_majority_proof_l2376_237629

theorem election_majority_proof :
  ∀ (total_votes : ℕ) (winning_percentage : ℚ),
    total_votes = 470 →
    winning_percentage = 70 / 100 →
    ∃ (winning_votes losing_votes : ℕ),
      winning_votes = (winning_percentage * total_votes).floor ∧
      losing_votes = total_votes - winning_votes ∧
      winning_votes - losing_votes = 188 :=
by
  sorry

end election_majority_proof_l2376_237629


namespace find_k_l2376_237693

theorem find_k (a b c d k : ℝ) 
  (h1 : a * b * c * d = 2007)
  (h2 : a = Real.sqrt (55 + Real.sqrt (k + a)))
  (h3 : b = Real.sqrt (55 - Real.sqrt (k + b)))
  (h4 : c = Real.sqrt (55 + Real.sqrt (k - c)))
  (h5 : d = Real.sqrt (55 - Real.sqrt (k - d))) :
  k = 1018 := by sorry

end find_k_l2376_237693


namespace water_remaining_l2376_237631

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 4/3 → remaining = initial - used → remaining = 5/3 := by
  sorry

end water_remaining_l2376_237631


namespace men_entered_room_l2376_237623

/-- Proves that 2 men entered the room given the initial and final conditions --/
theorem men_entered_room : 
  ∀ (initial_men initial_women : ℕ),
  initial_men / initial_women = 4 / 5 →
  ∃ (men_entered : ℕ),
  2 * (initial_women - 3) = 24 ∧
  initial_men + men_entered = 14 →
  men_entered = 2 := by
sorry

end men_entered_room_l2376_237623


namespace point_transformation_quadrant_l2376_237666

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate. -/
def second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

/-- A point in the third quadrant has a negative x-coordinate and a negative y-coordinate. -/
def third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

/-- If point P (a, b) is in the second quadrant, then point P₁ (-b, a-1) is in the third quadrant. -/
theorem point_transformation_quadrant (a b : ℝ) :
  second_quadrant (a, b) → third_quadrant (-b, a - 1) := by
  sorry

end point_transformation_quadrant_l2376_237666


namespace journey_time_l2376_237625

/-- Given a journey where:
  * The distance is 320 miles
  * The speed is 50 miles per hour
  * There is a 30-minute stopover
Prove that the total trip time is 6.9 hours -/
theorem journey_time (distance : ℝ) (speed : ℝ) (stopover : ℝ) :
  distance = 320 →
  speed = 50 →
  stopover = 0.5 →
  distance / speed + stopover = 6.9 :=
by sorry

end journey_time_l2376_237625


namespace marble_leftover_l2376_237699

theorem marble_leftover (n m k : ℤ) : (7*n + 2 + 7*m + 5 + 7*k + 4) % 7 = 4 := by
  sorry

end marble_leftover_l2376_237699


namespace digit_789_of_7_29_l2376_237681

def decimal_representation_7_29 : List ℕ :=
  [2, 4, 1, 3, 7, 9, 3, 1, 0, 3, 4, 4, 8, 2, 7, 5, 8, 6, 2, 0, 6, 8, 9, 6, 5, 5, 1, 7]

def repeating_period : ℕ := 28

theorem digit_789_of_7_29 : 
  (decimal_representation_7_29[(789 % repeating_period) - 1]) = 6 := by sorry

end digit_789_of_7_29_l2376_237681


namespace positive_number_equation_solution_l2376_237695

theorem positive_number_equation_solution :
  ∃ n : ℝ, n > 0 ∧ 3 * n^2 + n = 219 ∧ abs (n - 8.38) < 0.01 := by
  sorry

end positive_number_equation_solution_l2376_237695


namespace stability_comparison_l2376_237617

-- Define the concept of a data set
def DataSet := List ℝ

-- Define the variance of a data set
def variance (s : DataSet) : ℝ := sorry

-- Define the concept of stability for a data set
def is_more_stable (s1 s2 : DataSet) : Prop := 
  variance s1 < variance s2

-- Theorem statement
theorem stability_comparison (A B : DataSet) 
  (h_mean : (A.sum / A.length) = (B.sum / B.length))
  (h_var_A : variance A = 0.3)
  (h_var_B : variance B = 0.02) :
  is_more_stable B A := by sorry

end stability_comparison_l2376_237617


namespace balls_sold_l2376_237680

/-- Proves that the number of balls sold is 17 given the conditions of the problem -/
theorem balls_sold (selling_price : ℕ) (cost_price : ℕ) (loss : ℕ) :
  selling_price = 720 →
  loss = 5 * cost_price →
  cost_price = 60 →
  selling_price + loss = 17 * cost_price :=
by
  sorry

#check balls_sold

end balls_sold_l2376_237680


namespace probability_three_heads_in_seven_tosses_l2376_237609

def coin_tosses : ℕ := 7
def heads_count : ℕ := 3

theorem probability_three_heads_in_seven_tosses :
  (Nat.choose coin_tosses heads_count) / (2 ^ coin_tosses) = 35 / 128 :=
by sorry

end probability_three_heads_in_seven_tosses_l2376_237609


namespace card_selection_two_suits_l2376_237689

theorem card_selection_two_suits (deck_size : ℕ) (suits : ℕ) (cards_per_suit : ℕ) 
  (selection_size : ℕ) (h1 : deck_size = suits * cards_per_suit) 
  (h2 : suits = 4) (h3 : cards_per_suit = 13) (h4 : selection_size = 3) : 
  (suits.choose 2) * (cards_per_suit.choose 2 * cards_per_suit.choose 1 + 
   cards_per_suit.choose 1 * cards_per_suit.choose 2) = 12168 :=
by sorry

end card_selection_two_suits_l2376_237689


namespace valid_outfits_count_l2376_237632

/-- The number of shirts, pants, and hats available -/
def num_items : ℕ := 5

/-- The number of colors available for each item -/
def num_colors : ℕ := 5

/-- The number of outfits where no two items are the same color -/
def num_valid_outfits : ℕ := num_items * (num_items - 1) * (num_items - 2)

theorem valid_outfits_count :
  num_valid_outfits = 60 :=
by sorry

end valid_outfits_count_l2376_237632


namespace boys_left_to_girl_l2376_237668

/-- Represents a group of children standing in a circle. -/
structure CircleGroup where
  boys : ℕ
  girls : ℕ
  boys_right_to_girl : ℕ

/-- The main theorem to be proved. -/
theorem boys_left_to_girl (group : CircleGroup) 
  (h1 : group.boys = 40)
  (h2 : group.girls = 28)
  (h3 : group.boys_right_to_girl = 18) :
  group.boys - (group.boys - group.boys_right_to_girl) = 18 := by
  sorry

#check boys_left_to_girl

end boys_left_to_girl_l2376_237668


namespace worker_efficiency_l2376_237644

/-- 
Proves that if worker A is thrice as efficient as worker B, 
and A takes 10 days less than B to complete a job, 
then B alone takes 15 days to complete the job.
-/
theorem worker_efficiency (days_b : ℕ) : 
  (days_b / 3 = days_b - 10) → days_b = 15 := by
  sorry

end worker_efficiency_l2376_237644


namespace crosswalk_height_l2376_237690

/-- Represents a parallelogram with given dimensions -/
structure Parallelogram where
  side1 : ℝ  -- Length of one side
  side2 : ℝ  -- Length of adjacent side
  base : ℝ   -- Length of base parallel to side1
  height1 : ℝ -- Height perpendicular to side1
  height2 : ℝ -- Height perpendicular to side2

/-- The area of a parallelogram can be calculated two ways -/
axiom area_equality (p : Parallelogram) : p.side1 * p.height1 = p.side2 * p.height2

/-- Theorem stating the height of the parallelogram perpendicular to the 80-foot side -/
theorem crosswalk_height (p : Parallelogram) 
    (h1 : p.side1 = 60)
    (h2 : p.side2 = 80)
    (h3 : p.base = 30)
    (h4 : p.height1 = 60) :
    p.height2 = 22.5 := by
  sorry

end crosswalk_height_l2376_237690


namespace complement_of_union_is_multiples_of_three_l2376_237621

-- Define the set of integers
variable (U : Set Int)

-- Define sets A and B
def A : Set Int := {x | ∃ k : Int, x = 3 * k + 1}
def B : Set Int := {x | ∃ k : Int, x = 3 * k + 2}

-- State the theorem
theorem complement_of_union_is_multiples_of_three (hU : U = Set.univ) :
  (U \ (A ∪ B)) = {x : Int | ∃ k : Int, x = 3 * k} :=
sorry

end complement_of_union_is_multiples_of_three_l2376_237621


namespace min_value_expression_l2376_237659

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  a^2 + 9*a*b + 9*b^2 + 3*c^2 ≥ 81 ∧
  (a^2 + 9*a*b + 9*b^2 + 3*c^2 = 81 ↔ a = 3 ∧ b = 1 ∧ c = 3) :=
by sorry

end min_value_expression_l2376_237659


namespace part_one_part_two_l2376_237661

-- Part 1
theorem part_one (x : ℝ) (h1 : x^2 - 4*x + 3 < 0) 
  (h2 : |x - 1| ≤ 2) (h3 : (x + 3) / (x - 2) ≥ 0) : 
  2 < x ∧ x ≤ 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h : a > 0) 
  (h_suff : ∀ x : ℝ, ¬(x^2 - 4*a*x + 3*a^2 < 0) → 
    ¬(|x - 1| ≤ 2 ∧ (x + 3) / (x - 2) ≥ 0))
  (h_not_nec : ∃ x : ℝ, ¬(x^2 - 4*a*x + 3*a^2 < 0) ∧ 
    (|x - 1| ≤ 2 ∧ (x + 3) / (x - 2) ≥ 0)) :
  a > 3/2 := by sorry

end part_one_part_two_l2376_237661


namespace log_identity_l2376_237686

theorem log_identity (x : ℝ) : 
  x = (Real.log 3 / Real.log 5) ^ (Real.log 5 / Real.log 3) →
  Real.log x / Real.log 7 = -(Real.log (Real.log 5 / Real.log 3) / Real.log 7) * (Real.log 5 / Real.log 3) :=
by sorry

end log_identity_l2376_237686


namespace largest_divisor_of_n_squared_div_72_l2376_237633

theorem largest_divisor_of_n_squared_div_72 (n : ℕ) (h : n > 0) (h_div : 72 ∣ n^2) :
  ∃ (t : ℕ), t > 0 ∧ t ∣ n ∧ ∀ (k : ℕ), k > 0 → k ∣ n → k ≤ t :=
by sorry

end largest_divisor_of_n_squared_div_72_l2376_237633


namespace cubic_derivative_value_l2376_237610

def f (x : ℝ) := x^3

theorem cubic_derivative_value (x₀ : ℝ) :
  (deriv f) x₀ = 3 → x₀ = 1 ∨ x₀ = -1 := by sorry

end cubic_derivative_value_l2376_237610


namespace parallelogram_area_l2376_237647

theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 18) (h2 : b = 10) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin (π - θ) = 90 :=
by sorry

end parallelogram_area_l2376_237647


namespace intersection_count_l2376_237641

/-- A point in the Cartesian plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Checks if a point is the intersection of two lines -/
def is_intersection (p : Point) (k : ℤ) : Prop :=
  p.y = p.x - 3 ∧ p.y = k * p.x - k

/-- The theorem statement -/
theorem intersection_count : 
  ∃ (s : Finset ℤ), s.card = 3 ∧ 
  (∀ k : ℤ, k ∈ s ↔ ∃ p : Point, is_intersection p k) :=
sorry

end intersection_count_l2376_237641


namespace equation_solution_l2376_237622

theorem equation_solution :
  ∃ x : ℝ, x ≠ 4 ∧ (x - 3) / (4 - x) - 1 = 1 / (x - 4) ∧ x = 3 := by
  sorry

end equation_solution_l2376_237622


namespace sum_of_absolute_ratios_l2376_237655

theorem sum_of_absolute_ratios (x y z : ℚ) 
  (sum_zero : x + y + z = 0) 
  (product_nonzero : x * y * z ≠ 0) : 
  (|x| / (y + z) + |y| / (x + z) + |z| / (x + y) = 1) ∨
  (|x| / (y + z) + |y| / (x + z) + |z| / (x + y) = -1) :=
sorry

end sum_of_absolute_ratios_l2376_237655


namespace sum_six_smallest_multiples_of_12_l2376_237684

theorem sum_six_smallest_multiples_of_12 : 
  (Finset.range 6).sum (fun i => 12 * (i + 1)) = 252 := by
  sorry

end sum_six_smallest_multiples_of_12_l2376_237684


namespace f_properties_l2376_237651

-- Define the function f
def f (x : ℝ) : ℝ := x * (2 * x^2 - 3 * x - 12) + 5

-- Define the interval
def interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

-- State the theorem
theorem f_properties :
  -- 1. Tangent line at x = 1
  (∃ (m c : ℝ), ∀ x y, y = f x → (x - 1) * (f 1 - y) = m * (x - 1)^2 + c * (x - 1) 
                     ∧ m = -12 ∧ c = 0) ∧
  -- 2. Maximum value
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 5) ∧
  -- 3. Minimum value
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y ∧ f x = -15) :=
sorry

end f_properties_l2376_237651


namespace prob_at_least_two_girls_is_two_sevenths_l2376_237697

def total_students : ℕ := 8
def boys : ℕ := 5
def girls : ℕ := 3
def selected : ℕ := 3

def prob_at_least_two_girls : ℚ :=
  (Nat.choose girls 2 * Nat.choose boys 1 + Nat.choose girls 3 * Nat.choose boys 0) /
  Nat.choose total_students selected

theorem prob_at_least_two_girls_is_two_sevenths :
  prob_at_least_two_girls = 2 / 7 := by
  sorry

end prob_at_least_two_girls_is_two_sevenths_l2376_237697


namespace area_of_triangle_BEF_l2376_237687

-- Define the rectangle ABCD
structure Rectangle :=
  (a : ℝ) (b : ℝ)
  (area_eq : a * b = 30)

-- Define points E and F
structure Points (rect : Rectangle) :=
  (E : ℝ × ℝ)
  (F : ℝ × ℝ)
  (E_on_AB : E.2 = 0 ∧ 0 ≤ E.1 ∧ E.1 ≤ rect.a)
  (F_on_BC : F.1 = rect.a ∧ 0 ≤ F.2 ∧ F.2 ≤ rect.b)

-- Define the theorem
theorem area_of_triangle_BEF
  (rect : Rectangle)
  (pts : Points rect)
  (area_CGF : ℝ)
  (area_EGF : ℝ)
  (h1 : area_CGF = 2)
  (h2 : area_EGF = 3) :
  (1/2) * pts.E.1 * pts.F.2 = 35/8 :=
sorry

end area_of_triangle_BEF_l2376_237687


namespace union_of_A_and_B_l2376_237645

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4, 6} := by
  sorry

end union_of_A_and_B_l2376_237645


namespace negation_equivalence_l2376_237628

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by sorry

end negation_equivalence_l2376_237628


namespace rational_coefficient_terms_count_l2376_237619

theorem rational_coefficient_terms_count :
  let expression := (x : ℝ) * (5 ^ (1/4 : ℝ)) + (y : ℝ) * (7 ^ (1/2 : ℝ))
  let power := 500
  let is_rational_coeff (k : ℕ) := (k % 4 = 0) ∧ ((power - k) % 2 = 0)
  (Finset.filter is_rational_coeff (Finset.range (power + 1))).card = 126 :=
by sorry

end rational_coefficient_terms_count_l2376_237619


namespace quadratic_solution_implication_l2376_237614

theorem quadratic_solution_implication (a b : ℝ) : 
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0 → 4 * a + 8 * b = -4 := by
  sorry

end quadratic_solution_implication_l2376_237614


namespace cot_thirty_degrees_l2376_237646

theorem cot_thirty_degrees : Real.cos (π / 6) / Real.sin (π / 6) = Real.sqrt 3 := by
  sorry

end cot_thirty_degrees_l2376_237646


namespace copy_pages_theorem_l2376_237657

/-- Given a cost per page in cents and a budget in dollars, 
    calculates the maximum number of full pages that can be copied. -/
def max_pages_copied (cost_per_page : ℕ) (budget_dollars : ℕ) : ℕ :=
  (budget_dollars * 100) / cost_per_page

/-- Theorem stating that with a cost of 3 cents per page and a budget of $25, 
    the maximum number of full pages that can be copied is 833. -/
theorem copy_pages_theorem : max_pages_copied 3 25 = 833 := by
  sorry

#eval max_pages_copied 3 25

end copy_pages_theorem_l2376_237657


namespace sum_of_distances_eq_three_halves_side_length_l2376_237640

/-- An equilateral triangle with a point inside it -/
structure EquilateralTriangleWithPoint where
  /-- The side length of the equilateral triangle -/
  a : ℝ
  /-- The point inside the triangle -/
  M : ℝ × ℝ
  /-- Assertion that the triangle is equilateral with side length a -/
  is_equilateral : a > 0
  /-- Assertion that M is inside the triangle -/
  M_inside : True  -- This is a simplification; in a real proof, we'd need to define this properly

/-- The sum of distances from a point to the sides of an equilateral triangle -/
def sum_of_distances (t : EquilateralTriangleWithPoint) : ℝ :=
  sorry  -- The actual calculation would go here

/-- Theorem: The sum of distances from any point inside an equilateral triangle
    to its sides is equal to 3/2 times the side length -/
theorem sum_of_distances_eq_three_halves_side_length (t : EquilateralTriangleWithPoint) :
  sum_of_distances t = 3/2 * t.a := by
  sorry

end sum_of_distances_eq_three_halves_side_length_l2376_237640


namespace daves_phone_files_l2376_237654

theorem daves_phone_files :
  ∀ (initial_apps initial_files current_apps : ℕ),
    initial_apps = 24 →
    initial_files = 9 →
    current_apps = 12 →
    current_apps = (current_apps - 7) + 7 →
    current_apps - 7 = 5 :=
by
  sorry

end daves_phone_files_l2376_237654


namespace concentric_circles_radii_difference_l2376_237665

theorem concentric_circles_radii_difference
  (r R : ℝ) -- r and R are real numbers representing radii
  (h_positive : r > 0) -- r is positive
  (h_ratio : π * R^2 = 4 * π * r^2) -- area ratio is 1:4
  : R - r = r :=
by sorry

end concentric_circles_radii_difference_l2376_237665


namespace valid_arrays_count_l2376_237648

/-- A 3x3 array with entries of 1 or -1 -/
def ValidArray : Type := Matrix (Fin 3) (Fin 3) Int

/-- Predicate to check if an entry is valid (1 or -1) -/
def isValidEntry (x : Int) : Prop := x = 1 ∨ x = -1

/-- Predicate to check if all entries in the array are valid -/
def hasValidEntries (arr : ValidArray) : Prop :=
  ∀ i j, isValidEntry (arr i j)

/-- Predicate to check if the sum of a row is zero -/
def rowSumZero (arr : ValidArray) (i : Fin 3) : Prop :=
  (arr i 0) + (arr i 1) + (arr i 2) = 0

/-- Predicate to check if the sum of a column is zero -/
def colSumZero (arr : ValidArray) (j : Fin 3) : Prop :=
  (arr 0 j) + (arr 1 j) + (arr 2 j) = 0

/-- Predicate to check if an array satisfies all conditions -/
def isValidArray (arr : ValidArray) : Prop :=
  hasValidEntries arr ∧
  (∀ i, rowSumZero arr i) ∧
  (∀ j, colSumZero arr j)

/-- The main theorem: there are exactly 6 valid arrays -/
theorem valid_arrays_count :
  ∃! (s : Finset ValidArray), (∀ arr ∈ s, isValidArray arr) ∧ s.card = 6 :=
sorry

end valid_arrays_count_l2376_237648


namespace container_volume_ratio_l2376_237643

theorem container_volume_ratio (A B C : ℚ) 
  (h1 : (3 : ℚ) / 4 * A = (2 : ℚ) / 3 * B) 
  (h2 : (2 : ℚ) / 3 * B = (1 : ℚ) / 2 * C) : 
  A / C = (2 : ℚ) / 3 := by
  sorry

end container_volume_ratio_l2376_237643


namespace line_tangent_to_circle_l2376_237691

/-- Given a line and a circle, prove that if the line is tangent to the circle, then the constant a in the circle equation equals 2 + √5. -/
theorem line_tangent_to_circle (t θ : ℝ) (a : ℝ) (h_a : a > 0) :
  let line : ℝ × ℝ → Prop := λ p => ∃ t, p.1 = 1 - t ∧ p.2 = 2 * t
  let circle : ℝ × ℝ → Prop := λ p => ∃ θ, p.1 = Real.cos θ ∧ p.2 = Real.sin θ + a
  (∀ p, line p → ¬ circle p) ∧ (∃ p, line p ∧ (∀ ε > 0, ∃ q, circle q ∧ dist p q < ε)) →
  a = 2 + Real.sqrt 5 :=
by sorry

end line_tangent_to_circle_l2376_237691


namespace strawberry_pies_l2376_237660

/-- The number of pies that can be made from strawberries picked by Christine and Rachel -/
def number_of_pies (christine_pounds : ℕ) (rachel_multiplier : ℕ) (pounds_per_pie : ℕ) : ℕ :=
  (christine_pounds + christine_pounds * rachel_multiplier) / pounds_per_pie

/-- Theorem: Christine and Rachel can make 10 pies given the conditions -/
theorem strawberry_pies : number_of_pies 10 2 3 = 10 := by
  sorry

end strawberry_pies_l2376_237660


namespace certain_number_is_fifteen_l2376_237664

/-- The number of Doberman puppies -/
def doberman_puppies : ℕ := 20

/-- The number of Schnauzers -/
def schnauzers : ℕ := 55

/-- The certain number calculated from the given expression -/
def certain_number : ℤ := 3 * doberman_puppies - 5 + (doberman_puppies - schnauzers)

/-- Theorem stating that the certain number equals 15 -/
theorem certain_number_is_fifteen : certain_number = 15 := by sorry

end certain_number_is_fifteen_l2376_237664


namespace student_age_problem_l2376_237616

theorem student_age_problem (total_students : ℕ) (total_avg_age : ℝ)
  (group1_size group2_size group3_size : ℕ) (group1_avg group2_avg group3_avg : ℝ) :
  total_students = 24 →
  total_avg_age = 18 →
  group1_size = 6 →
  group2_size = 10 →
  group3_size = 7 →
  group1_avg = 16 →
  group2_avg = 20 →
  group3_avg = 17 →
  ∃ (last_student_age : ℝ),
    (group1_size * group1_avg + group2_size * group2_avg + group3_size * group3_avg + last_student_age) / total_students = total_avg_age ∧
    last_student_age = 15 :=
by sorry

end student_age_problem_l2376_237616


namespace polynomial_degree_theorem_l2376_237626

theorem polynomial_degree_theorem : 
  let p : Polynomial ℝ := (X^3 + 1)^5 * (X^4 + 1)^2
  Polynomial.degree p = 23 := by sorry

end polynomial_degree_theorem_l2376_237626


namespace quadratic_function_property_l2376_237608

/-- A quadratic function y = (x + m - 3)(x - m) + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := (x + m - 3) * (x - m) + 3

theorem quadratic_function_property (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ < x₂ →
  f m x₁ = y₁ →
  f m x₂ = y₂ →
  x₁ + x₂ < 3 →
  y₁ > y₂ := by
  sorry

end quadratic_function_property_l2376_237608


namespace sum_of_roots_l2376_237602

/-- Given distinct real numbers p, q, r, s such that
    x^2 - 12px - 13q = 0 has roots r and s, and
    x^2 - 12rx - 13s = 0 has roots p and q,
    prove that p + q + r + s = 2028 -/
theorem sum_of_roots (p q r s : ℝ) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h_roots1 : ∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s)
  (h_roots2 : ∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) :
  p + q + r + s = 2028 := by
sorry

end sum_of_roots_l2376_237602


namespace prime_power_fraction_l2376_237653

theorem prime_power_fraction (u v : ℕ+) :
  (∃ (p : ℕ) (n : ℕ), Prime p ∧ (u.val * v.val^3 : ℚ) / (u.val^2 + v.val^2) = p^n) ↔
  (∃ (k : ℕ), k ≥ 1 ∧ u.val = 2^k ∧ v.val = 2^k) :=
by sorry

end prime_power_fraction_l2376_237653


namespace smallest_solution_abs_equation_l2376_237624

theorem smallest_solution_abs_equation :
  ∀ x : ℝ, |x - 3| = 8 → x ≥ -5 ∧ |-5 - 3| = 8 :=
by sorry

end smallest_solution_abs_equation_l2376_237624


namespace imaginary_part_of_complex_fraction_l2376_237658

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (2 + i) / (1 + i)
  Complex.im z = -1/2 := by
  sorry

end imaginary_part_of_complex_fraction_l2376_237658


namespace matrix_vector_computation_l2376_237605

variable {m n : ℕ}
variable (N : Matrix (Fin 2) (Fin n) ℝ)
variable (a b : Fin n → ℝ)

theorem matrix_vector_computation 
  (ha : N.mulVec a = ![2, -3])
  (hb : N.mulVec b = ![5, 4]) :
  N.mulVec (3 • a - 2 • b) = ![-4, -17] := by sorry

end matrix_vector_computation_l2376_237605


namespace divisible_by_30_implies_x_is_0_l2376_237652

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

def four_digit_number (x : ℕ) : ℕ := x * 1000 + 240 + x

theorem divisible_by_30_implies_x_is_0 (x : ℕ) (h : x < 10) :
  is_divisible_by (four_digit_number x) 30 → x = 0 := by
  sorry

end divisible_by_30_implies_x_is_0_l2376_237652


namespace problem_statement_l2376_237674

theorem problem_statement (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  (a - c > 2 * b) ∧ (a^2 > b^2) := by
  sorry

end problem_statement_l2376_237674


namespace investment_rate_calculation_l2376_237676

/-- Given an investment of 3000 units yielding an income of 210 units,
    prove that the investment rate is 7%. -/
theorem investment_rate_calculation (investment : ℝ) (income : ℝ) (rate : ℝ) :
  investment = 3000 →
  income = 210 →
  rate = income / investment * 100 →
  rate = 7 := by
  sorry

end investment_rate_calculation_l2376_237676


namespace complex_fraction_equals_i_l2376_237636

/- Define the imaginary unit -/
variable (i : ℂ)

/- Define real numbers m and n -/
variable (m n : ℝ)

/- State the theorem -/
theorem complex_fraction_equals_i
  (h1 : i * i = -1)
  (h2 : m * (1 + i) = 11 + n * i) :
  (m + n * i) / (m - n * i) = i :=
sorry

end complex_fraction_equals_i_l2376_237636


namespace star_sum_five_l2376_237677

def star (a b : ℕ) : ℕ := a^b - a*b

theorem star_sum_five :
  ∀ a b : ℕ,
  a ≥ 2 →
  b ≥ 2 →
  star a b = 2 →
  a + b = 5 :=
by sorry

end star_sum_five_l2376_237677


namespace rulers_in_drawer_l2376_237611

/-- Given an initial number of rulers and an additional number of rulers,
    calculate the total number of rulers in the drawer. -/
def total_rulers (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that with 46 rulers initially and 25 rulers added,
    the total number of rulers in the drawer is 71. -/
theorem rulers_in_drawer : total_rulers 46 25 = 71 := by
  sorry

end rulers_in_drawer_l2376_237611


namespace smallest_four_digit_multiple_of_18_l2376_237606

theorem smallest_four_digit_multiple_of_18 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n → 1008 ≤ n :=
by
  sorry

end smallest_four_digit_multiple_of_18_l2376_237606


namespace eight_bead_necklace_arrangements_l2376_237642

/-- The number of distinct arrangements of n distinct beads on a necklace, 
    considering rotations and reflections as identical -/
def necklace_arrangements (n : ℕ) : ℕ := Nat.factorial n / (n * 2)

/-- Theorem stating that for 8 distinct beads, the number of distinct necklace arrangements is 2520 -/
theorem eight_bead_necklace_arrangements : 
  necklace_arrangements 8 = 2520 := by
  sorry

end eight_bead_necklace_arrangements_l2376_237642


namespace day_temperature_difference_l2376_237650

def temperature_difference (lowest highest : ℤ) : ℤ :=
  highest - lowest

theorem day_temperature_difference :
  let lowest : ℤ := -15
  let highest : ℤ := 3
  temperature_difference lowest highest = 18 := by
  sorry

end day_temperature_difference_l2376_237650


namespace monkey_doll_price_difference_is_two_l2376_237672

def monkey_doll_price_difference (total_spending : ℕ) (large_doll_price : ℕ) (extra_small_dolls : ℕ) : ℕ :=
  let large_dolls := total_spending / large_doll_price
  let small_dolls := large_dolls + extra_small_dolls
  let small_doll_price := total_spending / small_dolls
  large_doll_price - small_doll_price

theorem monkey_doll_price_difference_is_two :
  monkey_doll_price_difference 300 6 25 = 2 := by sorry

end monkey_doll_price_difference_is_two_l2376_237672


namespace vasya_driving_distance_l2376_237698

theorem vasya_driving_distance 
  (total_distance : ℝ) 
  (anton_distance vasya_distance sasha_distance dima_distance : ℝ)
  (h1 : anton_distance = vasya_distance / 2)
  (h2 : sasha_distance = anton_distance + dima_distance)
  (h3 : dima_distance = total_distance / 10)
  (h4 : anton_distance + vasya_distance + sasha_distance + dima_distance = total_distance)
  : vasya_distance = (2 / 5) * total_distance := by
  sorry

end vasya_driving_distance_l2376_237698


namespace f_extrema_and_intersection_range_l2376_237667

-- Define the function f(x) = x³ - 3x - 1
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

-- State the theorem
theorem f_extrema_and_intersection_range :
  -- f(x) has a maximum at x = -1
  (∀ x : ℝ, f (-1) ≥ f x) ∧
  -- f(x) has a minimum at x = 1
  (∀ x : ℝ, f 1 ≤ f x) ∧
  -- The range of m for which y = m intersects y = f(x) at three distinct points is (-3, 1)
  (∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) ↔ -3 < m ∧ m < 1) :=
by sorry

end f_extrema_and_intersection_range_l2376_237667


namespace cap_production_l2376_237696

theorem cap_production (first_week second_week third_week total_target : ℕ) 
  (h1 : first_week = 320)
  (h2 : second_week = 400)
  (h3 : third_week = 300)
  (h4 : total_target = 1360) :
  total_target - (first_week + second_week + third_week) = 340 :=
by sorry

end cap_production_l2376_237696


namespace point_D_coordinates_l2376_237601

/-- Given points A and B, and the relation between vectors AD and AB,
    prove that the coordinates of point D are (-7, 9) -/
theorem point_D_coordinates (A B D : ℝ × ℝ) : 
  A = (2, 3) → 
  B = (-1, 5) → 
  D - A = 3 • (B - A) → 
  D = (-7, 9) := by
sorry

end point_D_coordinates_l2376_237601


namespace chemistry_books_count_l2376_237627

theorem chemistry_books_count (biology_books : ℕ) (total_ways : ℕ) : 
  biology_books = 14 →
  total_ways = 2548 →
  (∃ chemistry_books : ℕ, 
    total_ways = (biology_books.choose 2) * (chemistry_books.choose 2)) →
  ∃ chemistry_books : ℕ, chemistry_books = 8 :=
by sorry

end chemistry_books_count_l2376_237627


namespace circle_area_is_one_l2376_237635

theorem circle_area_is_one (r : ℝ) (h : r > 0) :
  (4 / (2 * Real.pi * r) = 2 * r) → Real.pi * r^2 = 1 := by
  sorry

end circle_area_is_one_l2376_237635


namespace rectangle_midpoint_distances_l2376_237669

theorem rectangle_midpoint_distances (a b : ℝ) (ha : a = 3) (hb : b = 5) :
  let vertex := (0 : ℝ × ℝ)
  let midpoints := [
    (a / 2, 0),
    (a, b / 2),
    (a / 2, b),
    (0, b / 2)
  ]
  (midpoints.map (λ m => Real.sqrt ((m.1 - vertex.1)^2 + (m.2 - vertex.2)^2))).sum = 13.1 := by
  sorry

end rectangle_midpoint_distances_l2376_237669


namespace final_value_is_correct_l2376_237694

/-- Calculates the final value of sold games in USD given initial conditions and exchange rates -/
def final_value_usd (initial_value : ℝ) (usd_to_eur : ℝ) (eur_to_jpy : ℝ) (jpy_to_usd : ℝ) : ℝ :=
  let tripled_value := initial_value * 3
  let eur_value := tripled_value * usd_to_eur
  let jpy_value := eur_value * eur_to_jpy
  let sold_portion := 0.4
  let sold_value_jpy := jpy_value * sold_portion
  sold_value_jpy * jpy_to_usd

/-- Theorem stating that the final value of sold games is $225.42 given the initial conditions -/
theorem final_value_is_correct :
  final_value_usd 200 0.85 130 0.0085 = 225.42 := by
  sorry

#eval final_value_usd 200 0.85 130 0.0085

end final_value_is_correct_l2376_237694


namespace power_sum_implications_l2376_237683

theorem power_sum_implications (a b c : ℝ) : 
  (¬ ((a^2013 + b^2013 + c^2013 = 0) → (a^2014 + b^2014 + c^2014 = 0))) ∧ 
  ((a^2014 + b^2014 + c^2014 = 0) → (a^2015 + b^2015 + c^2015 = 0)) ∧ 
  (¬ ((a^2013 + b^2013 + c^2013 = 0 ∧ a^2015 + b^2015 + c^2015 = 0) → (a^2014 + b^2014 + c^2014 = 0))) :=
by sorry

end power_sum_implications_l2376_237683


namespace rattle_ownership_l2376_237600

structure Brother :=
  (id : ℕ)
  (claims_ownership : Bool)

def Alice := Unit

def determine_owner (b1 b2 : Brother) (a : Alice) : Brother :=
  sorry

theorem rattle_ownership (b1 b2 : Brother) (a : Alice) :
  b1.id = 1 →
  b2.id = 2 →
  b1.claims_ownership = true →
  (determine_owner b1 b2 a).id = 1 :=
sorry

end rattle_ownership_l2376_237600


namespace composition_result_l2376_237673

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x
def g (x : ℝ) : ℝ := x^2

-- Define the inverse functions
noncomputable def f_inv (x : ℝ) : ℝ := x / 2
noncomputable def g_inv (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem composition_result :
  f (g_inv (f_inv (f_inv (g (f 8))))) = 16 := by
  sorry

end composition_result_l2376_237673


namespace janes_stick_length_l2376_237656

/-- Given information about Pat's stick and its relationship to Sarah's and Jane's sticks,
    prove that Jane's stick is 22 inches long. -/
theorem janes_stick_length :
  -- Pat's stick length
  ∀ (pat_stick : ℕ),
  -- Covered portion of Pat's stick
  ∀ (covered_portion : ℕ),
  -- Conversion factor from feet to inches
  ∀ (feet_to_inches : ℕ),
  -- Conditions
  pat_stick = 30 →
  covered_portion = 7 →
  feet_to_inches = 12 →
  -- Sarah's stick is twice as long as the uncovered portion of Pat's stick
  ∃ (sarah_stick : ℕ), sarah_stick = 2 * (pat_stick - covered_portion) →
  -- Jane's stick is two feet shorter than Sarah's stick
  ∃ (jane_stick : ℕ), jane_stick = sarah_stick - 2 * feet_to_inches →
  -- Conclusion: Jane's stick is 22 inches long
  jane_stick = 22 :=
by
  sorry

end janes_stick_length_l2376_237656
