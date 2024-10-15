import Mathlib

namespace NUMINAMATH_CALUDE_tan_product_special_angles_l2089_208973

theorem tan_product_special_angles :
  let A : Real := 30 * π / 180
  let B : Real := 60 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) = 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_special_angles_l2089_208973


namespace NUMINAMATH_CALUDE_mustard_at_third_table_l2089_208985

theorem mustard_at_third_table 
  (first_table : Real) 
  (second_table : Real) 
  (total_mustard : Real) 
  (h1 : first_table = 0.25)
  (h2 : second_table = 0.25)
  (h3 : total_mustard = 0.88) :
  total_mustard - (first_table + second_table) = 0.38 := by
sorry

end NUMINAMATH_CALUDE_mustard_at_third_table_l2089_208985


namespace NUMINAMATH_CALUDE_inverse_g_sum_l2089_208947

-- Define the function g
def g (x : ℝ) : ℝ := x * |x|^3

-- State the theorem
theorem inverse_g_sum : g⁻¹ 8 + g⁻¹ (-64) = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_sum_l2089_208947


namespace NUMINAMATH_CALUDE_football_team_right_handed_players_l2089_208917

/-- Given a football team with the following properties:
  * There are 120 players in total
  * 62 players are throwers
  * Of the non-throwers, three-fifths are left-handed
  * All throwers are right-handed
  Prove that the total number of right-handed players is 86 -/
theorem football_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (non_throwers : ℕ) 
  (left_handed_non_throwers : ℕ) 
  (right_handed_non_throwers : ℕ) : 
  total_players = 120 →
  throwers = 62 →
  non_throwers = total_players - throwers →
  left_handed_non_throwers = (3 * non_throwers) / 5 →
  right_handed_non_throwers = non_throwers - left_handed_non_throwers →
  throwers + right_handed_non_throwers = 86 := by
  sorry

#check football_team_right_handed_players

end NUMINAMATH_CALUDE_football_team_right_handed_players_l2089_208917


namespace NUMINAMATH_CALUDE_square_sum_equals_one_l2089_208969

theorem square_sum_equals_one (a b : ℝ) 
  (h1 : a * Real.sqrt (1 - b^2) + b * Real.sqrt (1 - a^2) = 1)
  (h2 : 0 ≤ a ∧ a ≤ 1)
  (h3 : 0 ≤ b ∧ b ≤ 1) : 
  a^2 + b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_one_l2089_208969


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_iff_m_eq_two_l2089_208972

/-- A system of linear equations in x and y with parameter m -/
structure LinearSystem (m : ℝ) where
  eq1 : ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ
  h1 : ∀ x y, eq1 x y = m * x + 4 * y - (m + 2)
  h2 : ∀ x y, eq2 x y = x + m * y - m

/-- The system has infinitely many solutions -/
def HasInfinitelySolutions (sys : LinearSystem m) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ sys.eq1 x₁ y₁ = 0 ∧ sys.eq2 x₁ y₁ = 0 ∧ sys.eq1 x₂ y₂ = 0 ∧ sys.eq2 x₂ y₂ = 0

/-- The main theorem: the system has infinitely many solutions iff m = 2 -/
theorem infinitely_many_solutions_iff_m_eq_two (m : ℝ) (sys : LinearSystem m) :
  HasInfinitelySolutions sys ↔ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_iff_m_eq_two_l2089_208972


namespace NUMINAMATH_CALUDE_wire_service_reporters_l2089_208959

theorem wire_service_reporters (total : ℝ) (h1 : total > 0) : 
  let local_politics := 0.35 * total
  let not_politics := 0.5 * total
  let politics := total - not_politics
  let not_local_politics := politics - local_politics
  (not_local_politics / politics) * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l2089_208959


namespace NUMINAMATH_CALUDE_p_and_not_q_is_true_l2089_208943

/-- Proposition p: There exists a real number x such that x - 2 > log_10(x) -/
def p : Prop := ∃ x : ℝ, x - 2 > Real.log x / Real.log 10

/-- Proposition q: For all real numbers x, x^2 > 0 -/
def q : Prop := ∀ x : ℝ, x^2 > 0

/-- Theorem: The conjunction of p and (not q) is true -/
theorem p_and_not_q_is_true : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_p_and_not_q_is_true_l2089_208943


namespace NUMINAMATH_CALUDE_seeds_sowed_l2089_208997

/-- Proves that the number of buckets of seeds sowed is 2.75 -/
theorem seeds_sowed (initial : ℝ) (final : ℝ) (h1 : initial = 8.75) (h2 : final = 6) :
  initial - final = 2.75 := by
  sorry

end NUMINAMATH_CALUDE_seeds_sowed_l2089_208997


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2089_208957

theorem polynomial_factorization (x : ℤ) : 
  x^4 + 3*x^3 - 15*x^2 - 19*x + 30 = (x+2)*(x+5)*(x-1)*(x-3) :=
by
  sorry

#check polynomial_factorization

end NUMINAMATH_CALUDE_polynomial_factorization_l2089_208957


namespace NUMINAMATH_CALUDE_total_fruits_is_107_l2089_208974

-- Define the number of oranges and apples picked by George and Amelia
def george_oranges : ℕ := 45
def amelia_apples : ℕ := 15
def george_amelia_apple_diff : ℕ := 5
def george_amelia_orange_diff : ℕ := 18

-- Define the number of apples picked by George
def george_apples : ℕ := amelia_apples + george_amelia_apple_diff

-- Define the number of oranges picked by Amelia
def amelia_oranges : ℕ := george_oranges - george_amelia_orange_diff

-- Define the total number of fruits picked
def total_fruits : ℕ := george_oranges + george_apples + amelia_oranges + amelia_apples

-- Theorem statement
theorem total_fruits_is_107 : total_fruits = 107 := by sorry

end NUMINAMATH_CALUDE_total_fruits_is_107_l2089_208974


namespace NUMINAMATH_CALUDE_sum_equation_implies_N_value_l2089_208924

theorem sum_equation_implies_N_value :
  481 + 483 + 485 + 487 + 489 + 491 = 3000 - N → N = 84 := by
  sorry

end NUMINAMATH_CALUDE_sum_equation_implies_N_value_l2089_208924


namespace NUMINAMATH_CALUDE_frog_return_probability_l2089_208900

/-- Represents the probability of the frog being at position (x, y) after n hops -/
def prob (n : ℕ) (x y : ℕ) : ℚ :=
  sorry

/-- The grid size is 2x2 -/
def grid_size : ℕ := 2

/-- The number of hops the frog makes -/
def num_hops : ℕ := 3

/-- The probability of each possible movement (right, down, or stay) -/
def move_prob : ℚ := 1 / 3

theorem frog_return_probability :
  prob num_hops 0 0 = 1 / 27 :=
sorry

end NUMINAMATH_CALUDE_frog_return_probability_l2089_208900


namespace NUMINAMATH_CALUDE_third_function_symmetry_l2089_208945

-- Define the type for our functions
def RealFunction := ℝ → ℝ

-- State the theorem
theorem third_function_symmetry 
  (f : RealFunction) 
  (f_inv : RealFunction) 
  (h : RealFunction) 
  (h1 : ∀ x, f_inv (f x) = x) -- f_inv is the inverse of f
  (h2 : ∀ x y, h y = -x ↔ f_inv (-y) = -x) -- h is symmetric to f_inv w.r.t. x + y = 0
  : ∀ x, h x = -f (-x) := by
  sorry


end NUMINAMATH_CALUDE_third_function_symmetry_l2089_208945


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l2089_208931

theorem quadratic_equation_from_means (a b : ℝ) : 
  (a + b) / 2 = 5 → 
  Real.sqrt (a * b) = 15 → 
  ∃ (x : ℝ), x^2 - 10*x + 225 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l2089_208931


namespace NUMINAMATH_CALUDE_candy_distribution_l2089_208928

theorem candy_distribution (total_candy : ℕ) (num_friends : ℕ) 
  (h1 : total_candy = 24) (h2 : num_friends = 5) :
  let pieces_to_remove := total_candy % num_friends
  let remaining_candy := total_candy - pieces_to_remove
  pieces_to_remove = 
    Nat.min pieces_to_remove (total_candy - (remaining_candy / num_friends) * num_friends) ∧
  (remaining_candy / num_friends) * num_friends = remaining_candy :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l2089_208928


namespace NUMINAMATH_CALUDE_unique_solution_x_power_x_power_x_eq_2_l2089_208967

theorem unique_solution_x_power_x_power_x_eq_2 :
  ∃! (x : ℝ), x > 0 ∧ x^(x^x) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_x_power_x_power_x_eq_2_l2089_208967


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l2089_208965

-- Define an increasing function on ℝ
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem increasing_function_inequality (f : ℝ → ℝ) (m : ℝ) 
  (h_increasing : IncreasingFunction f) 
  (h_inequality : f (m + 1) > f (2 * m - 1)) : 
  m < 2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l2089_208965


namespace NUMINAMATH_CALUDE_house_spacing_l2089_208907

/-- Given a city of length 11.5 km and 6 houses to be built at regular intervals
    including both ends, the distance between each house is 2.3 km. -/
theorem house_spacing (city_length : ℝ) (num_houses : ℕ) :
  city_length = 11.5 ∧ num_houses = 6 →
  (city_length / (num_houses - 1 : ℝ)) = 2.3 := by
  sorry

end NUMINAMATH_CALUDE_house_spacing_l2089_208907


namespace NUMINAMATH_CALUDE_base9_to_base10_3956_l2089_208951

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (a b c d : ℕ) : ℕ :=
  a * 9^3 + b * 9^2 + c * 9^1 + d * 9^0

/-- Theorem: The base-9 number 3956₉ is equal to 2967 in base-10 --/
theorem base9_to_base10_3956 : base9ToBase10 3 9 5 6 = 2967 := by
  sorry

end NUMINAMATH_CALUDE_base9_to_base10_3956_l2089_208951


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l2089_208984

theorem triangle_third_side_length
  (a b c : ℕ)
  (h1 : a = 2)
  (h2 : b = 5)
  (h3 : Odd c)
  (h4 : a + b > c)
  (h5 : b + c > a)
  (h6 : c + a > b) :
  c = 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l2089_208984


namespace NUMINAMATH_CALUDE_max_k_value_l2089_208914

open Real

noncomputable def f (x : ℝ) : ℝ := exp x - x - 2

theorem max_k_value (k : ℤ) :
  (∀ x : ℝ, x > 0 → (k - x) / (x + 1) * (exp x - 1) < 1) →
  k ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l2089_208914


namespace NUMINAMATH_CALUDE_exponent_calculation_l2089_208960

theorem exponent_calculation : (-4)^6 / 4^4 + 2^5 - 7^2 = -1 := by sorry

end NUMINAMATH_CALUDE_exponent_calculation_l2089_208960


namespace NUMINAMATH_CALUDE_combined_value_l2089_208938

def i : ℕ := 2  -- The only prime even integer from 2 to √500

def k : ℕ := 44  -- Sum of even integers from 8 to √200 (8 + 10 + 12 + 14)

def j : ℕ := 23  -- Sum of prime odd integers from 5 to √133 (5 + 7 + 11)

theorem combined_value : 2 * i - k + 3 * j = 29 := by
  sorry

end NUMINAMATH_CALUDE_combined_value_l2089_208938


namespace NUMINAMATH_CALUDE_probability_girl_grade4_l2089_208987

/-- The probability of selecting a girl from grade 4 in a school playground -/
theorem probability_girl_grade4 (g3 b3 g4 b4 g5 b5 : ℕ) : 
  g3 = 28 → b3 = 35 → g4 = 45 → b4 = 42 → g5 = 38 → b5 = 51 →
  (g4 : ℚ) / (g3 + b3 + g4 + b4 + g5 + b5) = 45 / 239 := by
  sorry

end NUMINAMATH_CALUDE_probability_girl_grade4_l2089_208987


namespace NUMINAMATH_CALUDE_insect_eggs_base_conversion_l2089_208908

theorem insect_eggs_base_conversion : 
  (2 * 7^2 + 3 * 7^1 + 5 * 7^0 : ℕ) = 124 := by sorry

end NUMINAMATH_CALUDE_insect_eggs_base_conversion_l2089_208908


namespace NUMINAMATH_CALUDE_jane_hector_meeting_l2089_208941

/-- Represents the points around the block area -/
inductive Point := | A | B | C | D | E

/-- The total distance around the block area -/
def total_distance : ℕ := 24

/-- Hector's walking speed -/
def hector_speed : ℝ := 1

/-- Jane's walking speed -/
def jane_speed : ℝ := 3 * hector_speed

/-- The distance walked by Hector when they meet -/
def hector_distance : ℝ := 6

/-- The distance walked by Jane when they meet -/
def jane_distance : ℝ := 18

/-- The point where Jane and Hector meet -/
def meeting_point : Point := Point.C

theorem jane_hector_meeting :
  (jane_speed = 3 * hector_speed) →
  (hector_distance + jane_distance = total_distance) →
  (jane_distance = 3 * hector_distance) →
  meeting_point = Point.C :=
by sorry

end NUMINAMATH_CALUDE_jane_hector_meeting_l2089_208941


namespace NUMINAMATH_CALUDE_production_difference_formula_l2089_208992

/-- Represents the widget production scenario for David --/
structure WidgetProduction where
  /-- Widgets produced per hour on Monday --/
  w : ℕ
  /-- Hours worked on Monday --/
  t : ℕ
  /-- Relationship between w and t --/
  w_eq_3t : w = 3 * t

/-- Calculates the difference in widget production between Monday and Tuesday --/
def productionDifference (p : WidgetProduction) : ℕ :=
  let monday_production := p.w * p.t
  let tuesday_production := (p.w + 6) * (p.t - 3)
  monday_production - tuesday_production

/-- Theorem stating the difference in widget production --/
theorem production_difference_formula (p : WidgetProduction) :
  productionDifference p = 3 * p.t + 18 := by
  sorry

#check production_difference_formula

end NUMINAMATH_CALUDE_production_difference_formula_l2089_208992


namespace NUMINAMATH_CALUDE_jasons_journey_l2089_208962

/-- The distance to Jason's home --/
def distance_to_home (total_time : ℝ) (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * (total_time - time1)

/-- Theorem statement for Jason's journey --/
theorem jasons_journey :
  let total_time : ℝ := 1.5
  let speed1 : ℝ := 60
  let time1 : ℝ := 0.5
  let speed2 : ℝ := 90
  distance_to_home total_time speed1 time1 speed2 = 120 := by
sorry

end NUMINAMATH_CALUDE_jasons_journey_l2089_208962


namespace NUMINAMATH_CALUDE_chess_game_draw_fraction_l2089_208930

theorem chess_game_draw_fraction 
  (ellen_wins : ℚ) 
  (john_wins : ℚ) 
  (h1 : ellen_wins = 4/9) 
  (h2 : john_wins = 2/9) : 
  1 - (ellen_wins + john_wins) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_chess_game_draw_fraction_l2089_208930


namespace NUMINAMATH_CALUDE_carrot_stick_calories_prove_carrot_stick_calories_l2089_208980

theorem carrot_stick_calories : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun total_calories burger_calories cookie_count cookie_calories carrot_stick_count carrot_stick_calories =>
    (total_calories = burger_calories + cookie_count * cookie_calories + carrot_stick_count * carrot_stick_calories) →
    (total_calories = 750) →
    (burger_calories = 400) →
    (cookie_count = 5) →
    (cookie_calories = 50) →
    (carrot_stick_count = 5) →
    (carrot_stick_calories = 20)

theorem prove_carrot_stick_calories : carrot_stick_calories 750 400 5 50 5 20 :=
by sorry

end NUMINAMATH_CALUDE_carrot_stick_calories_prove_carrot_stick_calories_l2089_208980


namespace NUMINAMATH_CALUDE_rope_contact_length_l2089_208963

/-- The length of rope in contact with a cylindrical tower, given specific conditions --/
theorem rope_contact_length 
  (rope_length : ℝ) 
  (tower_radius : ℝ) 
  (unicorn_height : ℝ) 
  (free_end_distance : ℝ) 
  (h1 : rope_length = 25) 
  (h2 : tower_radius = 10) 
  (h3 : unicorn_height = 3) 
  (h4 : free_end_distance = 5) : 
  ∃ (contact_length : ℝ), contact_length = rope_length - Real.sqrt 134 := by
  sorry

#check rope_contact_length

end NUMINAMATH_CALUDE_rope_contact_length_l2089_208963


namespace NUMINAMATH_CALUDE_square_binomial_plus_cube_problem_solution_l2089_208958

theorem square_binomial_plus_cube (a b : ℕ) : 
  a^2 + 2*a*b + b^2 + b^3 = (a + b)^2 + b^3 := by sorry

theorem problem_solution : 15^2 + 2*(15*5) + 5^2 + 5^3 = 525 := by
  have h1 : 15^2 + 2*(15*5) + 5^2 + 5^3 = (15 + 5)^2 + 5^3 := by
    exact square_binomial_plus_cube 15 5
  have h2 : (15 + 5)^2 = 400 := by norm_num
  have h3 : 5^3 = 125 := by norm_num
  calc
    15^2 + 2*(15*5) + 5^2 + 5^3 = (15 + 5)^2 + 5^3 := h1
    _ = 400 + 125 := by rw [h2, h3]
    _ = 525 := by norm_num

end NUMINAMATH_CALUDE_square_binomial_plus_cube_problem_solution_l2089_208958


namespace NUMINAMATH_CALUDE_smallest_cycle_length_cycle_length_21_smallest_b_is_21_l2089_208922

def g (x : ℤ) : ℤ :=
  if x % 5 = 0 ∧ x % 7 = 0 then x / 35
  else if x % 7 = 0 then 5 * x
  else if x % 5 = 0 then 7 * x
  else x + 5

def g_iterate (n : ℕ) (x : ℤ) : ℤ :=
  match n with
  | 0 => x
  | n + 1 => g (g_iterate n x)

theorem smallest_cycle_length :
  ∀ b : ℕ, b > 1 → g_iterate b 3 = g 3 → b ≥ 21 :=
by sorry

theorem cycle_length_21 : g_iterate 21 3 = g 3 :=
by sorry

theorem smallest_b_is_21 :
  ∃! b : ℕ, b > 1 ∧ g_iterate b 3 = g 3 ∧ ∀ k : ℕ, k > 1 → g_iterate k 3 = g 3 → k ≥ b :=
by sorry

end NUMINAMATH_CALUDE_smallest_cycle_length_cycle_length_21_smallest_b_is_21_l2089_208922


namespace NUMINAMATH_CALUDE_canoe_kayak_difference_l2089_208936

/-- Represents the daily rental cost of a canoe -/
def canoe_cost : ℕ := 15

/-- Represents the daily rental cost of a kayak -/
def kayak_cost : ℕ := 18

/-- Represents the total daily revenue -/
def total_revenue : ℕ := 405

/-- Theorem stating that the difference between canoes and kayaks rented is 5 -/
theorem canoe_kayak_difference :
  ∀ (c k : ℕ),
  (c : ℚ) / k = 3 / 2 →
  canoe_cost * c + kayak_cost * k = total_revenue →
  c - k = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_canoe_kayak_difference_l2089_208936


namespace NUMINAMATH_CALUDE_largest_factorial_as_consecutive_product_l2089_208916

theorem largest_factorial_as_consecutive_product : 
  ∀ n : ℕ, n > 3 → ¬(∃ k : ℕ, n! = (k + 1) * (k + 2) * (k + 3) * (k + 4)) :=
by sorry

end NUMINAMATH_CALUDE_largest_factorial_as_consecutive_product_l2089_208916


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l2089_208968

theorem smallest_k_no_real_roots : 
  ∀ k : ℤ, (∀ x : ℝ, 2*x*(k*x-4)-x^2+7 ≠ 0) → k ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l2089_208968


namespace NUMINAMATH_CALUDE_speed_limit_excess_l2089_208935

/-- Proves that a journey of 150 miles completed in 2 hours exceeds a 60 mph speed limit by 15 mph -/
theorem speed_limit_excess (distance : ℝ) (time : ℝ) (speed_limit : ℝ) : 
  distance = 150 ∧ time = 2 ∧ speed_limit = 60 →
  distance / time - speed_limit = 15 := by
sorry

end NUMINAMATH_CALUDE_speed_limit_excess_l2089_208935


namespace NUMINAMATH_CALUDE_semicircle_area_l2089_208979

theorem semicircle_area (d : ℝ) (h : d = 11) : 
  (1/2) * π * (d/2)^2 = (121/8) * π := by sorry

end NUMINAMATH_CALUDE_semicircle_area_l2089_208979


namespace NUMINAMATH_CALUDE_election_winner_votes_l2089_208937

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) : 
  winner_percentage = 62 / 100 →
  vote_difference = 324 →
  (winner_percentage * total_votes).num = total_votes * winner_percentage.num →
  (winner_percentage * total_votes).num - ((1 - winner_percentage) * total_votes).num = vote_difference →
  (winner_percentage * total_votes).num = 837 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l2089_208937


namespace NUMINAMATH_CALUDE_sine_central_angle_is_zero_l2089_208939

/-- Represents a circle with intersecting chords -/
structure IntersectingChords where
  radius : ℝ
  pq_length : ℝ
  rt_length : ℝ

/-- The sine of the central angle subtending arc PR in the given circle configuration -/
def sine_central_angle (c : IntersectingChords) : ℝ :=
  sorry

/-- Theorem stating that the sine of the central angle is 0 for the given configuration -/
theorem sine_central_angle_is_zero (c : IntersectingChords) 
  (h1 : c.radius = 7)
  (h2 : c.pq_length = 14)
  (h3 : c.rt_length = 5) : 
  sine_central_angle c = 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_central_angle_is_zero_l2089_208939


namespace NUMINAMATH_CALUDE_deepak_present_age_l2089_208994

-- Define the ages as natural numbers
variable (R D : ℕ)

-- Define the conditions
def ratio_condition : Prop := 4 * D = 3 * R
def future_age_condition : Prop := R + 6 = 26

-- Theorem statement
theorem deepak_present_age 
  (h1 : ratio_condition R D) 
  (h2 : future_age_condition R) : 
  D = 15 := by sorry

end NUMINAMATH_CALUDE_deepak_present_age_l2089_208994


namespace NUMINAMATH_CALUDE_initial_birds_count_l2089_208926

/-- The number of birds initially on the fence -/
def initial_birds : ℕ := sorry

/-- The number of birds that landed on the fence -/
def landed_birds : ℕ := 8

/-- The total number of birds after more birds landed -/
def total_birds : ℕ := 20

/-- Theorem stating that the initial number of birds is 12 -/
theorem initial_birds_count : initial_birds = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_count_l2089_208926


namespace NUMINAMATH_CALUDE_cereal_eating_time_l2089_208946

/-- The time it takes for three people to collectively eat a certain amount of cereal -/
def eating_time (fat_rate thin_rate medium_rate total_pounds : ℚ) : ℚ :=
  total_pounds / (1 / fat_rate + 1 / thin_rate + 1 / medium_rate)

/-- Theorem stating that given the eating rates of Mr. Fat, Mr. Thin, and Mr. Medium,
    the time required for them to collectively eat 5 pounds of cereal is 100/3 minutes -/
theorem cereal_eating_time :
  eating_time 20 30 15 5 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l2089_208946


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l2089_208988

theorem cosine_sine_identity : 
  Real.cos (35 * π / 180) * Real.cos (25 * π / 180) - 
  Real.sin (145 * π / 180) * Real.cos (65 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l2089_208988


namespace NUMINAMATH_CALUDE_other_number_is_198_l2089_208929

/-- Given two positive integers with specific HCF and LCM, prove one is 198 when the other is 24 -/
theorem other_number_is_198 (a b : ℕ+) : 
  Nat.gcd a b = 12 → 
  Nat.lcm a b = 396 → 
  a = 24 → 
  b = 198 := by
sorry

end NUMINAMATH_CALUDE_other_number_is_198_l2089_208929


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l2089_208909

/-- Given a triangle XYZ with specific point ratios, prove that the intersection of certain lines has coordinates (1/3, 2/3, 0) -/
theorem intersection_point_coordinates (X Y Z D E F P : ℝ × ℝ × ℝ) : 
  -- Triangle XYZ exists
  X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X →
  -- D is on YZ extended with ratio 4:1
  ∃ t : ℝ, t > 1 ∧ D = t • Z + (1 - t) • Y ∧ (t - 1) / (5 - t) = 4 →
  -- E is on XZ with ratio 3:2
  ∃ s : ℝ, 0 < s ∧ s < 1 ∧ E = s • X + (1 - s) • Z ∧ s / (1 - s) = 3 / 2 →
  -- F is on XY with ratio 2:1
  ∃ r : ℝ, 0 < r ∧ r < 1 ∧ F = r • X + (1 - r) • Y ∧ r / (1 - r) = 2 →
  -- P is the intersection of BF and YD
  ∃ u v : ℝ, P = u • F + (1 - u) • E ∧ P = v • D + (1 - v) • Y →
  -- Conclusion: P has coordinates (1/3, 2/3, 0) in terms of X, Y, Z
  P = (1/3) • X + (2/3) • Y + 0 • Z :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l2089_208909


namespace NUMINAMATH_CALUDE_sequence_range_l2089_208999

theorem sequence_range (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_recur : ∀ n, a (n + 1) ≥ 2 * a n + 1) 
  (h_bound : ∀ n, a n < 2^(n + 1)) : 
  0 < a 1 ∧ a 1 ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_range_l2089_208999


namespace NUMINAMATH_CALUDE_intersection_M_N_l2089_208977

open Set

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 4 > 0}
def N : Set ℝ := {x | x < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x < -2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2089_208977


namespace NUMINAMATH_CALUDE_inverse_matrices_values_l2089_208955

theorem inverse_matrices_values (a b : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -9; a, 14]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![14, b; 5, 4]
  (A * B = 1 ∧ B * A = 1) → (a = -5 ∧ b = 9) :=
by sorry

end NUMINAMATH_CALUDE_inverse_matrices_values_l2089_208955


namespace NUMINAMATH_CALUDE_car_mileage_l2089_208970

/-- Given a car that travels 200 kilometers using 5 gallons of gasoline, its mileage is 40 kilometers per gallon. -/
theorem car_mileage (distance : ℝ) (gasoline : ℝ) (h1 : distance = 200) (h2 : gasoline = 5) :
  distance / gasoline = 40 := by
  sorry

end NUMINAMATH_CALUDE_car_mileage_l2089_208970


namespace NUMINAMATH_CALUDE_min_value_of_sum_l2089_208933

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 1) :
  (3 / a + 2 / b) ≥ 25 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 2 * b₀ = 1 ∧ 3 / a₀ + 2 / b₀ = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l2089_208933


namespace NUMINAMATH_CALUDE_reflection_line_equation_l2089_208934

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  p : Point
  q : Point
  r : Point

/-- The reflection of a triangle -/
structure ReflectedTriangle where
  p' : Point
  q' : Point
  r' : Point

/-- The line of reflection -/
structure ReflectionLine where
  equation : ℝ → Prop

/-- Theorem: Given a triangle and its reflection, prove the equation of the reflection line -/
theorem reflection_line_equation 
  (t : Triangle) 
  (rt : ReflectedTriangle) 
  (h1 : t.p = ⟨2, 2⟩) 
  (h2 : t.q = ⟨6, 6⟩) 
  (h3 : t.r = ⟨-3, 5⟩)
  (h4 : rt.p' = ⟨2, -4⟩) 
  (h5 : rt.q' = ⟨6, -8⟩) 
  (h6 : rt.r' = ⟨-3, -7⟩) :
  ∃ (l : ReflectionLine), l.equation = λ y => y = -1 := by
  sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l2089_208934


namespace NUMINAMATH_CALUDE_supplement_congruence_l2089_208976

/-- Two angles are congruent if they have the same measure -/
def congruent_angles (α β : Real) : Prop := α = β

/-- The supplement of an angle is another angle that, when added to it, equals 180° -/
def supplement (α : Real) : Real := 180 - α

theorem supplement_congruence (α β : Real) :
  congruent_angles (supplement α) (supplement β) → congruent_angles α β := by
  sorry

end NUMINAMATH_CALUDE_supplement_congruence_l2089_208976


namespace NUMINAMATH_CALUDE_total_distance_calculation_l2089_208925

-- Define the distance walked per day
def distance_per_day : ℝ := 4.0

-- Define the number of days walked
def days_walked : ℝ := 3.0

-- Define the total distance walked
def total_distance : ℝ := distance_per_day * days_walked

-- Theorem statement
theorem total_distance_calculation :
  total_distance = 12.0 := by sorry

end NUMINAMATH_CALUDE_total_distance_calculation_l2089_208925


namespace NUMINAMATH_CALUDE_candy_bar_cost_l2089_208949

def candy_sales (n : Nat) : Nat :=
  10 + 4 * (n - 1)

def total_candy_sales (days : Nat) : Nat :=
  (List.range days).map candy_sales |>.sum

theorem candy_bar_cost (days : Nat) (total_earnings : Rat) :
  days = 6 ∧ total_earnings = 12 →
  total_earnings / total_candy_sales days = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l2089_208949


namespace NUMINAMATH_CALUDE_ABD_collinear_l2089_208971

/-- Given vectors in 2D space -/
def a : ℝ × ℝ := sorry
def b : ℝ × ℝ := sorry

/-- Define vectors AB, BC, and CD -/
def AB : ℝ × ℝ := a + 5 • b
def BC : ℝ × ℝ := -2 • a + 8 • b
def CD : ℝ × ℝ := 3 • (a - b)

/-- Define points A, B, C, and D -/
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := A + AB
def C : ℝ × ℝ := B + BC
def D : ℝ × ℝ := C + CD

/-- Theorem: Points A, B, and D are collinear -/
theorem ABD_collinear : ∃ (t : ℝ), D = A + t • (B - A) := by sorry

end NUMINAMATH_CALUDE_ABD_collinear_l2089_208971


namespace NUMINAMATH_CALUDE_acute_angle_is_40_isosceles_trapezoid_acute_angle_l2089_208944

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The length of the diagonal -/
  diagonal : ℝ
  /-- The angle between the diagonal and the leg -/
  angle_diagonal_leg : ℝ
  /-- The angle between the diagonal and the base -/
  angle_diagonal_base : ℝ
  /-- The area is √3 -/
  area_eq : area = Real.sqrt 3
  /-- The diagonal length is 2 -/
  diagonal_eq : diagonal = 2
  /-- The angle between the diagonal and the base is 20° greater than the angle between the diagonal and the leg -/
  angle_relation : angle_diagonal_base = angle_diagonal_leg + 20

/-- The acute angle of the trapezoid is 40° -/
theorem acute_angle_is_40 (t : IsoscelesTrapezoid) : ℝ :=
  40

/-- The main theorem: proving that the acute angle of the trapezoid is 40° -/
theorem isosceles_trapezoid_acute_angle 
  (t : IsoscelesTrapezoid) : acute_angle_is_40 t = 40 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_is_40_isosceles_trapezoid_acute_angle_l2089_208944


namespace NUMINAMATH_CALUDE_inequality_and_fraction_analysis_l2089_208920

theorem inequality_and_fraction_analysis (a : ℝ) (h : 2 < a ∧ a < 4) :
  (3 * a - 2 > 2 * a ∧ 4 * (a - 1) < 3 * a) ∧
  (a - (a + 4) / (a + 1) = (a^2 - 4) / (a + 1)) ∧
  ((a^2 - 4) / (a + 1) ≥ 0 ↔ a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_fraction_analysis_l2089_208920


namespace NUMINAMATH_CALUDE_f_properties_l2089_208927

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin x ^ 2 + (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + Real.cos x ^ 2

theorem f_properties :
  (∀ x : ℝ, f x ≤ 5/4) ∧
  (∀ x : ℝ, f x = 5/4 ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi / 6) ∧
  (∀ x : ℝ, f x = (1/2) * Real.sin (2 * x + Real.pi / 6) + 3/4) := by sorry

end NUMINAMATH_CALUDE_f_properties_l2089_208927


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2089_208903

/-- The eccentricity of a hyperbola with equation x^2 - y^2 = 1 is √2 -/
theorem hyperbola_eccentricity : 
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 - y^2 = 1}
  ∃ e : ℝ, e = Real.sqrt 2 ∧ 
    ∀ (a b c : ℝ), 
      (a = 1 ∧ b = 1 ∧ c^2 = a^2 + b^2) → 
      e = c / a :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2089_208903


namespace NUMINAMATH_CALUDE_subtracted_value_l2089_208982

theorem subtracted_value (x : ℤ) (h : 282 = x + 133) : x - 11 = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l2089_208982


namespace NUMINAMATH_CALUDE_roots_exist_when_q_positive_no_integer_roots_when_q_negative_l2089_208983

-- Define the quadratic equations
def equation1 (p q x : ℤ) : Prop := x^2 - p*x + q = 0
def equation2 (p q x : ℤ) : Prop := x^2 - (p+1)*x + q = 0

-- Theorem for q > 0
theorem roots_exist_when_q_positive (q : ℤ) (hq : q > 0) :
  ∃ (p x1 x2 x3 x4 : ℤ), 
    equation1 p q x1 ∧ equation1 p q x2 ∧
    equation2 p q x3 ∧ equation2 p q x4 :=
sorry

-- Theorem for q < 0
theorem no_integer_roots_when_q_negative (q : ℤ) (hq : q < 0) :
  ¬∃ (p x1 x2 x3 x4 : ℤ), 
    equation1 p q x1 ∧ equation1 p q x2 ∧
    equation2 p q x3 ∧ equation2 p q x4 :=
sorry

end NUMINAMATH_CALUDE_roots_exist_when_q_positive_no_integer_roots_when_q_negative_l2089_208983


namespace NUMINAMATH_CALUDE_nested_root_simplification_l2089_208996

theorem nested_root_simplification :
  (81 * Real.sqrt (27 * Real.sqrt 9)) ^ (1/4) = 3 * 9 ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_nested_root_simplification_l2089_208996


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l2089_208910

/-- A circle C tangent to the y-axis at point (0,2) and tangent to the line 4x-3y+9=0 -/
structure TangentCircle where
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- The radius of the circle -/
  radius : ℝ
  /-- The circle is tangent to the y-axis at (0,2) -/
  tangent_y_axis : center.1 = radius
  /-- The circle's center is on the line y=2 -/
  center_on_line : center.2 = 2
  /-- The circle is tangent to the line 4x-3y+9=0 -/
  tangent_line : |4 * center.1 - 3 * center.2 + 9| / Real.sqrt 25 = radius

/-- The standard equation of the circle is either (x-3)^2+(y-2)^2=9 or (x+1/3)^2+(y-2)^2=1/9 -/
theorem tangent_circle_equation (c : TangentCircle) :
  (c.center = (3, 2) ∧ c.radius = 3) ∨ (c.center = (-1/3, 2) ∧ c.radius = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l2089_208910


namespace NUMINAMATH_CALUDE_parabola_ellipse_coincident_foci_l2089_208981

/-- Given a parabola and an ellipse, proves that if their foci coincide, then the parameter of the parabola is 4. -/
theorem parabola_ellipse_coincident_foci (p : ℝ) : 
  (∀ x y, y^2 = 2*p*x → x^2/6 + y^2/2 = 1 → x = p/2 ∧ x = 2) → p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_ellipse_coincident_foci_l2089_208981


namespace NUMINAMATH_CALUDE_max_profit_l2089_208953

noncomputable section

-- Define the cost function G(x)
def G (x : ℝ) : ℝ := 2.8 + x

-- Define the revenue function R(x)
def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 5 then -0.4 * x^2 + 4.2 * x
  else 11

-- Define the profit function f(x)
def f (x : ℝ) : ℝ := R x - G x

-- Theorem stating the maximum profit and the corresponding production quantity
theorem max_profit :
  ∃ (x_max : ℝ), x_max = 4 ∧
  ∀ (x : ℝ), 0 ≤ x → f x ≤ f x_max ∧
  f x_max = 3.6 :=
sorry

end

end NUMINAMATH_CALUDE_max_profit_l2089_208953


namespace NUMINAMATH_CALUDE_circle_intersection_and_tangent_line_l2089_208904

-- Define the lines and circles
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def O (x y : ℝ) : Prop := x^2 + y^2 = 4
def l₂ (x y : ℝ) : Prop := y - 2 = 4/3 * (x + 1)
def M_center_line (x y : ℝ) : Prop := x - 2 * y = 0

-- Define the properties of circle M
def M (x y : ℝ) : Prop := (x - 8/3)^2 + (y - 4/3)^2 = 100/9

-- Theorem statement
theorem circle_intersection_and_tangent_line 
  (h₁ : ∀ x y, l₁ x y → l₂ x y → (x = -1 ∧ y = 2)) 
  (h₂ : ∃ x y, M x y ∧ M_center_line x y) 
  (h₃ : ∃ x y, M x y ∧ l₂ x y) 
  (h₄ : ∃ k, k > 0 ∧ ∀ x y, M x y → l₁ x y → 
    (∃ a₁ a₂, a₁ > 0 ∧ a₂ > 0 ∧ a₁ / a₂ = 2 ∧ a₁ + a₂ = 2 * π * k)) :
  (∃ x y, O x y ∧ l₁ x y ∧ 
    (∃ x' y', O x' y' ∧ l₁ x' y' ∧ (x - x')^2 + (y - y')^2 = 12)) ∧
  (∀ x y, M x y ↔ (x - 8/3)^2 + (y - 4/3)^2 = 100/9) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_and_tangent_line_l2089_208904


namespace NUMINAMATH_CALUDE_line_parametrization_l2089_208921

/-- The slope of the line -/
def m : ℚ := 3/4

/-- The y-intercept of the line -/
def b : ℚ := -5

/-- The x-coordinate of the point on the line -/
def x₀ : ℚ := -8

/-- The y-component of the direction vector -/
def v : ℚ := 7

/-- The equation of the line -/
def line_eq (x y : ℚ) : Prop := y = m * x + b

/-- The parametric form of the line -/
def parametric_eq (s l t x y : ℚ) : Prop :=
  x = x₀ + t * l ∧ y = s + t * v

theorem line_parametrization (s l : ℚ) :
  (∀ t x y, parametric_eq s l t x y → line_eq x y) →
  s = -11 ∧ l = 28/3 := by sorry

end NUMINAMATH_CALUDE_line_parametrization_l2089_208921


namespace NUMINAMATH_CALUDE_time_after_1550_minutes_l2089_208906

/-- Represents a time with day, hour, and minute components -/
structure DateTime where
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The starting DateTime (midnight on January 1, 2011) -/
def startTime : DateTime :=
  { day := 1, hour := 0, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : Nat := 1550

/-- The expected result DateTime -/
def expectedResult : DateTime :=
  { day := 2, hour := 1, minute := 50 }

/-- Theorem stating that adding 1550 minutes to midnight on January 1
    results in 1:50 AM on January 2 -/
theorem time_after_1550_minutes :
  addMinutes startTime minutesToAdd = expectedResult := by
  sorry

end NUMINAMATH_CALUDE_time_after_1550_minutes_l2089_208906


namespace NUMINAMATH_CALUDE_conference_center_occupancy_l2089_208950

def room_capacities : List ℕ := [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320]

def occupancy_rates : List ℚ := [3/4, 5/6, 2/3, 3/5, 4/9, 11/15, 7/10, 1/2, 5/8, 9/14, 8/15, 17/20]

theorem conference_center_occupancy :
  let occupied_rooms := List.zip room_capacities occupancy_rates
  let total_people := occupied_rooms.map (λ (cap, rate) => (cap : ℚ) * rate)
  ⌊total_people.sum⌋ = 1639 := by sorry

end NUMINAMATH_CALUDE_conference_center_occupancy_l2089_208950


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2089_208964

theorem polynomial_simplification (q : ℝ) : 
  (4 * q^3 - 7 * q^2 + 3 * q + 8) + (5 - 3 * q^3 + 9 * q^2 - 2 * q) = q^3 + 2 * q^2 + q + 13 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2089_208964


namespace NUMINAMATH_CALUDE_investment_decrease_l2089_208940

theorem investment_decrease (P : ℝ) (x : ℝ) 
  (h1 : P > 0)
  (h2 : 1.60 * P - (x / 100) * (1.60 * P) = 1.12 * P) :
  x = 30 := by
sorry

end NUMINAMATH_CALUDE_investment_decrease_l2089_208940


namespace NUMINAMATH_CALUDE_mobius_gauss_formula_pentagon_l2089_208966

/-- Given a pentagon ABCDE with triangle areas α, β, γ, θ, δ, 
    the total area S satisfies the Möbius-Gauss formula. -/
theorem mobius_gauss_formula_pentagon (α β γ θ δ S : ℝ) 
  (h_positive : α > 0 ∧ β > 0 ∧ γ > 0 ∧ θ > 0 ∧ δ > 0) 
  (h_area : S > 0) 
  (h_sum : S > (α + β + γ + θ + δ) / 2) :
  S^2 - (α + β + γ + θ + δ) * S + (α*β + β*γ + γ*θ + θ*δ + δ*α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_mobius_gauss_formula_pentagon_l2089_208966


namespace NUMINAMATH_CALUDE_rectangle_area_reduction_l2089_208954

theorem rectangle_area_reduction (original_area : ℝ) 
  (h1 : original_area = 432) 
  (length_reduction : ℝ) (width_reduction : ℝ)
  (h2 : length_reduction = 0.15)
  (h3 : width_reduction = 0.20) : 
  original_area * (1 - length_reduction) * (1 - width_reduction) = 293.76 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_reduction_l2089_208954


namespace NUMINAMATH_CALUDE_crab_fishing_income_l2089_208961

theorem crab_fishing_income 
  (num_buckets : ℕ) 
  (crabs_per_bucket : ℕ) 
  (price_per_crab : ℕ) 
  (days_per_week : ℕ) 
  (h1 : num_buckets = 8) 
  (h2 : crabs_per_bucket = 12) 
  (h3 : price_per_crab = 5) 
  (h4 : days_per_week = 7) : 
  num_buckets * crabs_per_bucket * price_per_crab * days_per_week = 3360 := by
sorry

end NUMINAMATH_CALUDE_crab_fishing_income_l2089_208961


namespace NUMINAMATH_CALUDE_binary_sum_equals_638_l2089_208991

def binary_to_decimal (b : ℕ) : ℕ := 2^b - 1

theorem binary_sum_equals_638 :
  (binary_to_decimal 9) + (binary_to_decimal 7) = 638 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_638_l2089_208991


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2089_208942

/-- An arithmetic sequence with a non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def GeometricSequence (x y z : ℚ) : Prop :=
  y^2 = x * z

theorem arithmetic_geometric_ratio
  (a : ℕ → ℚ) (d : ℚ)
  (h_arith : ArithmeticSequence a d)
  (h_geom : GeometricSequence (a 5) (a 9) (a 15)) :
  a 15 / a 9 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2089_208942


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l2089_208905

/-- The quadratic function f(x) = (x - 2)² - 3 -/
def f (x : ℝ) : ℝ := (x - 2)^2 - 3

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (2, -3)

theorem vertex_of_quadratic :
  ∀ x : ℝ, f x ≥ f (vertex.1) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l2089_208905


namespace NUMINAMATH_CALUDE_unique_solution_equation_l2089_208952

theorem unique_solution_equation (x : ℝ) (h1 : x ≠ 0) :
  (9 * x) ^ 18 = (27 * x) ^ 9 ↔ x = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l2089_208952


namespace NUMINAMATH_CALUDE_sum_of_roots_equal_one_l2089_208993

theorem sum_of_roots_equal_one : 
  ∃ (x₁ x₂ : ℝ), (x₁ + 2) * (x₁ - 3) = 16 ∧ 
                 (x₂ + 2) * (x₂ - 3) = 16 ∧ 
                 x₁ + x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equal_one_l2089_208993


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l2089_208919

/-- 
Given an equation x^2 + 8x + y^2 + 4y - k = 0 that represents a circle with radius 7,
prove that k = 29.
-/
theorem circle_equation_k_value (k : ℝ) :
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 4*y - k = 0 ↔ (x + 4)^2 + (y + 2)^2 = 7^2) →
  k = 29 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l2089_208919


namespace NUMINAMATH_CALUDE_area_enclosed_by_functions_l2089_208956

/-- The area enclosed by y = x and f(x) = 2 - x^2 -/
theorem area_enclosed_by_functions : ∃ (a : ℝ), a = (9 : ℝ) / 2 ∧ 
  a = ∫ x in (-2 : ℝ)..1, (2 - x^2 - x) := by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_functions_l2089_208956


namespace NUMINAMATH_CALUDE_x_fourth_plus_reciprocal_l2089_208923

theorem x_fourth_plus_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^4 + 1/x^4 = 527 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_reciprocal_l2089_208923


namespace NUMINAMATH_CALUDE_slipper_equation_l2089_208902

-- Define the original price of slippers
variable (x : ℝ)

-- Define the amount spent before and during the sale
def before_sale : ℝ := 120
def during_sale : ℝ := 100

-- Define the price reduction during the sale
def price_reduction : ℝ := 5

-- Define the additional pairs bought during the sale
def additional_pairs : ℕ := 2

-- Theorem stating the equation that represents the situation
theorem slipper_equation :
  before_sale / x = during_sale / (x - price_reduction) - additional_pairs :=
sorry

end NUMINAMATH_CALUDE_slipper_equation_l2089_208902


namespace NUMINAMATH_CALUDE_unique_solution_iff_b_less_than_two_l2089_208911

/-- The equation has exactly one real solution -/
def has_unique_real_solution (b : ℝ) : Prop :=
  ∃! x : ℝ, x^3 - b*x^2 - 3*b*x + b^2 - 4 = 0

/-- The main theorem -/
theorem unique_solution_iff_b_less_than_two :
  ∀ b : ℝ, has_unique_real_solution b ↔ b < 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_b_less_than_two_l2089_208911


namespace NUMINAMATH_CALUDE_base_r_palindrome_square_l2089_208901

theorem base_r_palindrome_square (r : ℕ) (h1 : r % 2 = 0) (h2 : r ≥ 18) : 
  let x := 5*r^3 + 5*r^2 + 5*r + 5
  let squared := x^2
  let a := squared / r^7 % r
  let b := squared / r^6 % r
  let c := squared / r^5 % r
  let d := squared / r^4 % r
  (squared = a*r^7 + b*r^6 + c*r^5 + d*r^4 + d*r^3 + c*r^2 + b*r + a) ∧ 
  (d - c = 2) →
  r = 24 := by sorry

end NUMINAMATH_CALUDE_base_r_palindrome_square_l2089_208901


namespace NUMINAMATH_CALUDE_rahul_twice_mary_age_l2089_208978

/-- Proves that Rahul will be twice as old as Mary after 20 years -/
theorem rahul_twice_mary_age : ∀ (x : ℕ),
  let mary_age : ℕ := 10
  let rahul_age : ℕ := mary_age + 30
  x = 20 ↔ rahul_age + x = 2 * (mary_age + x) :=
by sorry

end NUMINAMATH_CALUDE_rahul_twice_mary_age_l2089_208978


namespace NUMINAMATH_CALUDE_order_of_logarithms_and_root_l2089_208989

theorem order_of_logarithms_and_root (a b c : ℝ) : 
  a = 2 * Real.log 0.99 →
  b = Real.log 0.98 →
  c = Real.sqrt 0.96 - 1 →
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_order_of_logarithms_and_root_l2089_208989


namespace NUMINAMATH_CALUDE_min_distance_complex_l2089_208998

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 1 - 2*I) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_distance_complex_l2089_208998


namespace NUMINAMATH_CALUDE_calculation_proof_l2089_208912

theorem calculation_proof : (-49 : ℚ) * (4/7) - (4/7) / (-8/7) = -55/2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2089_208912


namespace NUMINAMATH_CALUDE_average_marks_math_chem_l2089_208915

theorem average_marks_math_chem (M P C B : ℕ) : 
  M + P = 80 →
  C + B = 120 →
  C = P + 20 →
  B = M - 15 →
  (M + C) / 2 = 50 := by
sorry

end NUMINAMATH_CALUDE_average_marks_math_chem_l2089_208915


namespace NUMINAMATH_CALUDE_ten_customers_miss_sunday_paper_l2089_208995

/-- Represents Kyle's newspaper delivery route -/
structure NewspaperRoute where
  totalHouses : ℕ
  dailyDeliveries : ℕ
  sundayOnlyDeliveries : ℕ
  weeklyTotalDeliveries : ℕ

/-- Calculates the number of customers who do not get the Sunday paper -/
def customersMissingSundayPaper (route : NewspaperRoute) : ℕ :=
  route.totalHouses - (route.totalHouses - (route.weeklyTotalDeliveries - 6 * route.totalHouses - route.sundayOnlyDeliveries))

/-- Theorem stating that 10 customers do not get the Sunday paper -/
theorem ten_customers_miss_sunday_paper (route : NewspaperRoute) 
  (h1 : route.totalHouses = 100)
  (h2 : route.dailyDeliveries = 100)
  (h3 : route.sundayOnlyDeliveries = 30)
  (h4 : route.weeklyTotalDeliveries = 720) :
  customersMissingSundayPaper route = 10 := by
  sorry

#eval customersMissingSundayPaper { totalHouses := 100, dailyDeliveries := 100, sundayOnlyDeliveries := 30, weeklyTotalDeliveries := 720 }

end NUMINAMATH_CALUDE_ten_customers_miss_sunday_paper_l2089_208995


namespace NUMINAMATH_CALUDE_blanket_price_problem_l2089_208948

/-- Proves that the unknown rate of two blankets is 228.75, given the conditions of the problem -/
theorem blanket_price_problem (price_3 : ℕ) (price_1 : ℕ) (discount : ℚ) (tax : ℚ) (avg_price : ℕ) :
  price_3 = 100 →
  price_1 = 150 →
  discount = 1/10 →
  tax = 3/20 →
  avg_price = 150 →
  let total_blankets : ℕ := 6
  let discounted_price_3 : ℚ := 3 * price_3 * (1 - discount)
  let taxed_price_1 : ℚ := price_1 * (1 + tax)
  let total_price : ℚ := total_blankets * avg_price
  ∃ x : ℚ, 
    x = (total_price - discounted_price_3 - taxed_price_1) / 2 ∧ 
    x = 457.5 / 2 := by
  sorry

#eval 457.5 / 2  -- Should output 228.75

end NUMINAMATH_CALUDE_blanket_price_problem_l2089_208948


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2089_208986

theorem geometric_series_sum : ∀ (a r : ℝ) (n : ℕ),
  a = 2 → r = 3 → n = 6 →
  a * (r^n - 1) / (r - 1) = 728 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2089_208986


namespace NUMINAMATH_CALUDE_tenth_largest_four_digit_odd_l2089_208918

/-- The set of odd digits -/
def OddDigits : Set Nat := {1, 3, 5, 7, 9}

/-- A four-digit number composed of only odd digits -/
def FourDigitOddNumber (a b c d : Nat) : Prop :=
  a ∈ OddDigits ∧ b ∈ OddDigits ∧ c ∈ OddDigits ∧ d ∈ OddDigits ∧
  1000 ≤ a * 1000 + b * 100 + c * 10 + d ∧ a * 1000 + b * 100 + c * 10 + d ≤ 9999

/-- The theorem stating that 9971 is the tenth largest four-digit number composed of only odd digits -/
theorem tenth_largest_four_digit_odd : 
  (∃ (n : Nat), n = 9 ∧ 
    (∃ (a b c d : Nat), FourDigitOddNumber a b c d ∧ 
      a * 1000 + b * 100 + c * 10 + d > 9971)) := by sorry

end NUMINAMATH_CALUDE_tenth_largest_four_digit_odd_l2089_208918


namespace NUMINAMATH_CALUDE_stream_speed_is_one_l2089_208990

/-- Represents the speed of a boat in still water and the speed of a stream. -/
structure BoatProblem where
  boat_speed : ℝ
  stream_speed : ℝ

/-- Given the conditions of the problem, proves that the stream speed is 1 km/h. -/
theorem stream_speed_is_one
  (bp : BoatProblem)
  (h1 : bp.boat_speed + bp.stream_speed = 100 / 10)
  (h2 : bp.boat_speed - bp.stream_speed = 200 / 25) :
  bp.stream_speed = 1 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_is_one_l2089_208990


namespace NUMINAMATH_CALUDE_upstream_downstream_time_difference_l2089_208932

/-- Proves that the difference in time between traveling upstream and downstream is 90 minutes -/
theorem upstream_downstream_time_difference 
  (distance : ℝ) 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : distance = 36) 
  (h2 : boat_speed = 10) 
  (h3 : stream_speed = 2) : 
  (distance / (boat_speed - stream_speed) - distance / (boat_speed + stream_speed)) * 60 = 90 := by
  sorry

#check upstream_downstream_time_difference

end NUMINAMATH_CALUDE_upstream_downstream_time_difference_l2089_208932


namespace NUMINAMATH_CALUDE_three_std_dev_below_mean_l2089_208975

/-- Given a distribution with mean 16.2 and standard deviation 2.3,
    the value 3 standard deviations below the mean is 9.3 -/
theorem three_std_dev_below_mean (μ : ℝ) (σ : ℝ) 
  (h_mean : μ = 16.2) (h_std_dev : σ = 2.3) :
  μ - 3 * σ = 9.3 := by
  sorry

end NUMINAMATH_CALUDE_three_std_dev_below_mean_l2089_208975


namespace NUMINAMATH_CALUDE_st_length_is_135_14_l2089_208913

/-- Triangle PQR with given side lengths and a parallel line ST containing the incenter -/
structure SpecialTriangle where
  -- Define the triangle PQR
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  -- Define points S and T
  S : ℝ × ℝ
  T : ℝ × ℝ
  -- Conditions
  pq_length : PQ = 13
  pr_length : PR = 14
  qr_length : QR = 15
  s_on_pq : S.1 ≥ 0 ∧ S.1 ≤ PQ
  t_on_pr : T.2 ≥ 0 ∧ T.2 ≤ PR
  st_parallel_qr : sorry -- ST is parallel to QR
  st_contains_incenter : sorry -- ST contains the incenter of PQR

/-- The length of ST in the special triangle -/
def ST_length (triangle : SpecialTriangle) : ℝ := sorry

/-- Theorem stating that ST length is 135/14 -/
theorem st_length_is_135_14 (triangle : SpecialTriangle) : 
  ST_length triangle = 135 / 14 := by sorry

end NUMINAMATH_CALUDE_st_length_is_135_14_l2089_208913
