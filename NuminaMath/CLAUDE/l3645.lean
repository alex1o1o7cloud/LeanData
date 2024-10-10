import Mathlib

namespace new_persons_weight_l3645_364559

theorem new_persons_weight (original_count : ℕ) (weight_increase : ℝ) 
  (replaced_weight1 : ℝ) (replaced_weight2 : ℝ) : 
  original_count = 20 →
  weight_increase = 5 →
  replaced_weight1 = 58 →
  replaced_weight2 = 64 →
  (original_count : ℝ) * weight_increase + replaced_weight1 + replaced_weight2 = 222 :=
by sorry

end new_persons_weight_l3645_364559


namespace material_mix_ratio_l3645_364521

theorem material_mix_ratio (x y : ℝ) 
  (h1 : 50 * x + 40 * y = 50 * (1 + 0.1) * x + 40 * (1 - 0.15) * y) : 
  x / y = 6 / 5 := by
  sorry

end material_mix_ratio_l3645_364521


namespace brick_width_calculation_l3645_364532

theorem brick_width_calculation (courtyard_length : ℝ) (courtyard_width : ℝ)
  (brick_length : ℝ) (total_bricks : ℕ) :
  courtyard_length = 18 →
  courtyard_width = 12 →
  brick_length = 0.12 →
  total_bricks = 30000 →
  ∃ (brick_width : ℝ),
    brick_width = 0.06 ∧
    courtyard_length * courtyard_width * 100 * 100 = total_bricks * brick_length * brick_width * 10000 :=
by sorry

end brick_width_calculation_l3645_364532


namespace wall_tiling_impossible_l3645_364540

/-- Represents the dimensions of a rectangular cuboid -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Checks if a smaller cuboid can tile a larger cuboid -/
def can_tile (wall : Dimensions) (brick : Dimensions) : Prop :=
  ∃ (a b c : ℕ), 
    (a * brick.length = wall.length ∧ 
     b * brick.width = wall.width ∧ 
     c * brick.height = wall.height) ∨
    (a * brick.length = wall.length ∧ 
     b * brick.width = wall.height ∧ 
     c * brick.height = wall.width) ∨
    (a * brick.length = wall.width ∧ 
     b * brick.width = wall.length ∧ 
     c * brick.height = wall.height) ∨
    (a * brick.length = wall.width ∧ 
     b * brick.width = wall.height ∧ 
     c * brick.height = wall.length) ∨
    (a * brick.length = wall.height ∧ 
     b * brick.width = wall.length ∧ 
     c * brick.height = wall.width) ∨
    (a * brick.length = wall.height ∧ 
     b * brick.width = wall.width ∧ 
     c * brick.height = wall.length)

theorem wall_tiling_impossible (wall : Dimensions) 
  (brick1 : Dimensions) (brick2 : Dimensions) : 
  wall.length = 27 ∧ wall.width = 16 ∧ wall.height = 15 →
  brick1.length = 3 ∧ brick1.width = 5 ∧ brick1.height = 7 →
  brick2.length = 2 ∧ brick2.width = 5 ∧ brick2.height = 6 →
  ¬(can_tile wall brick1 ∨ can_tile wall brick2) :=
sorry

end wall_tiling_impossible_l3645_364540


namespace square_sum_equals_36_l3645_364550

theorem square_sum_equals_36 (x y z w : ℝ) 
  (eq1 : x^2 / (2^2 - 1^2) + y^2 / (2^2 - 3^2) + z^2 / (2^2 - 5^2) + w^2 / (2^2 - 7^2) = 1)
  (eq2 : x^2 / (4^2 - 1^2) + y^2 / (4^2 - 3^2) + z^2 / (4^2 - 5^2) + w^2 / (4^2 - 7^2) = 1)
  (eq3 : x^2 / (6^2 - 1^2) + y^2 / (6^2 - 3^2) + z^2 / (6^2 - 5^2) + w^2 / (6^2 - 7^2) = 1)
  (eq4 : x^2 / (8^2 - 1^2) + y^2 / (8^2 - 3^2) + z^2 / (8^2 - 5^2) + w^2 / (8^2 - 7^2) = 1) :
  x^2 + y^2 + z^2 + w^2 = 36 := by
sorry


end square_sum_equals_36_l3645_364550


namespace larger_cuboid_height_l3645_364588

/-- Prove that the height of a larger cuboid is 2 meters given specific conditions -/
theorem larger_cuboid_height (small_length small_width small_height : ℝ)
  (large_length large_width : ℝ) (num_small_cuboids : ℝ) :
  small_length = 6 →
  small_width = 4 →
  small_height = 3 →
  large_length = 18 →
  large_width = 15 →
  num_small_cuboids = 7.5 →
  ∃ (large_height : ℝ),
    num_small_cuboids * (small_length * small_width * small_height) =
      large_length * large_width * large_height ∧
    large_height = 2 := by
  sorry

end larger_cuboid_height_l3645_364588


namespace jerry_total_miles_l3645_364598

/-- The total miles Jerry walked over three days -/
def total_miles (monday tuesday wednesday : ℕ) : ℕ :=
  monday + tuesday + wednesday

/-- Theorem stating that Jerry walked 45 miles in total -/
theorem jerry_total_miles :
  total_miles 15 18 12 = 45 := by
  sorry

end jerry_total_miles_l3645_364598


namespace ants_meet_after_11_laps_l3645_364558

/-- The number of laps on the small circle before the ants meet again -/
def num_laps_to_meet (large_radius small_radius : ℕ) : ℕ :=
  Nat.lcm large_radius small_radius / small_radius

theorem ants_meet_after_11_laps :
  num_laps_to_meet 33 9 = 11 := by sorry

end ants_meet_after_11_laps_l3645_364558


namespace special_quadrilateral_not_necessarily_square_l3645_364573

/-- A convex quadrilateral with special diagonal properties -/
structure SpecialQuadrilateral where
  /-- The quadrilateral is convex -/
  convex : Bool
  /-- Any diagonal divides the quadrilateral into two isosceles triangles -/
  diagonal_isosceles : Bool
  /-- Both diagonals divide the quadrilateral into four isosceles triangles -/
  both_diagonals_isosceles : Bool

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- The trapezoid has parallel bases -/
  parallel_bases : Bool
  /-- The non-parallel sides are equal -/
  equal_legs : Bool
  /-- The smaller base is equal to the legs -/
  base_equals_legs : Bool

/-- Theorem: There exists a quadrilateral satisfying the special properties that is not a square -/
theorem special_quadrilateral_not_necessarily_square :
  ∃ (q : SpecialQuadrilateral) (t : IsoscelesTrapezoid),
    q.convex ∧
    q.diagonal_isosceles ∧
    q.both_diagonals_isosceles ∧
    t.parallel_bases ∧
    t.equal_legs ∧
    t.base_equals_legs ∧
    (q ≠ square) := by
  sorry

end special_quadrilateral_not_necessarily_square_l3645_364573


namespace factors_of_N_l3645_364575

/-- The number of natural-number factors of N, where N = 2^5 * 3^4 * 5^3 * 7^2 * 11^1 -/
def number_of_factors (N : ℕ) : ℕ :=
  (5 + 1) * (4 + 1) * (3 + 1) * (2 + 1) * (1 + 1)

/-- Theorem stating that the number of natural-number factors of N is 720 -/
theorem factors_of_N :
  let N : ℕ := 2^5 * 3^4 * 5^3 * 7^2 * 11^1
  number_of_factors N = 720 := by
  sorry

end factors_of_N_l3645_364575


namespace square_plaza_area_l3645_364519

/-- The area of a square plaza with side length 5 × 10^2 m is 2.5 × 10^5 m^2. -/
theorem square_plaza_area :
  let side_length : ℝ := 5 * 10^2
  let area : ℝ := side_length^2
  area = 2.5 * 10^5 := by sorry

end square_plaza_area_l3645_364519


namespace complex_fraction_bounds_l3645_364597

theorem complex_fraction_bounds (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) :
  ∃ (min max : ℝ),
    (∀ z w : ℂ, z ≠ 0 → w ≠ 0 → min ≤ Complex.abs (z + w) / (Complex.abs z + Complex.abs w) ∧
                        Complex.abs (z + w) / (Complex.abs z + Complex.abs w) ≤ max) ∧
    min = 0 ∧ max = 1 ∧ max - min = 1 :=
by sorry

end complex_fraction_bounds_l3645_364597


namespace factor_implies_b_value_l3645_364564

theorem factor_implies_b_value (a b : ℤ) :
  (∃ c : ℤ, ∀ x : ℝ, (x^2 - 2*x - 1) * (c*x - 1) = a*x^3 + b*x^2 + 1) →
  b = -3 := by
  sorry

end factor_implies_b_value_l3645_364564


namespace expression_value_l3645_364527

theorem expression_value (x y : ℝ) (h : x - 2*y = 1) : 3 - 4*y + 2*x = 5 := by
  sorry

end expression_value_l3645_364527


namespace triangle_prime_angles_l3645_364514

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem triangle_prime_angles 
  (a b c : ℕ) 
  (sum_180 : a + b + c = 180) 
  (all_prime : is_prime a ∧ is_prime b ∧ is_prime c) 
  (all_less_120 : a < 120 ∧ b < 120 ∧ c < 120) : 
  ((a = 2 ∧ b = 71 ∧ c = 107) ∨ (a = 2 ∧ b = 89 ∧ c = 89)) ∨
  ((a = 71 ∧ b = 2 ∧ c = 107) ∨ (a = 89 ∧ b = 2 ∧ c = 89)) ∨
  ((a = 71 ∧ b = 107 ∧ c = 2) ∨ (a = 89 ∧ b = 89 ∧ c = 2)) :=
by sorry

end triangle_prime_angles_l3645_364514


namespace stamp_collection_value_l3645_364524

theorem stamp_collection_value
  (total_stamps : ℕ)
  (sample_stamps : ℕ)
  (sample_value : ℚ)
  (h1 : total_stamps = 18)
  (h2 : sample_stamps = 6)
  (h3 : sample_value = 15)
  : ℚ :=
by
  -- The total value of the stamp collection is 45 dollars
  sorry

end stamp_collection_value_l3645_364524


namespace checkout_speed_ratio_l3645_364554

/-- Represents the problem of determining the ratio of cashier checkout speed to the rate of increase in waiting people. -/
theorem checkout_speed_ratio
  (n : ℕ)  -- Initial number of people in line
  (y : ℝ)  -- Rate at which number of people waiting increases (people per minute)
  (x : ℝ)  -- Cashier's checkout speed (people per minute)
  (h1 : 20 * 2 * x = 20 * y + n)  -- Equation for 2 counters open for 20 minutes
  (h2 : 12 * 3 * x = 12 * y + n)  -- Equation for 3 counters open for 12 minutes
  : x = 2 * y :=
sorry

end checkout_speed_ratio_l3645_364554


namespace constant_term_of_expansion_l3645_364581

theorem constant_term_of_expansion (x : ℝ) (x_pos : x > 0) :
  ∃ (c : ℝ), (∀ (ε : ℝ), ε > 0 → 
    ∃ (δ : ℝ), δ > 0 ∧ 
    ∀ (y : ℝ), abs (y - x) < δ → 
    abs ((y.sqrt + 3 / y)^10 - c) < ε) ∧
  c = 59049 := by
sorry

end constant_term_of_expansion_l3645_364581


namespace video_game_lives_l3645_364511

theorem video_game_lives (initial_lives lost_lives gained_lives : ℕ) 
  (h1 : initial_lives = 43)
  (h2 : lost_lives = 14)
  (h3 : gained_lives = 27) :
  initial_lives - lost_lives + gained_lives = 56 :=
by sorry

end video_game_lives_l3645_364511


namespace northwest_molded_break_even_l3645_364547

/-- Break-even point calculation for Northwest Molded -/
theorem northwest_molded_break_even :
  let fixed_cost : ℝ := 7640
  let variable_cost : ℝ := 0.60
  let selling_price : ℝ := 4.60
  let break_even_point := fixed_cost / (selling_price - variable_cost)
  break_even_point = 1910 := by
  sorry

end northwest_molded_break_even_l3645_364547


namespace card_distribution_l3645_364556

theorem card_distribution (total_cards : Nat) (num_players : Nat) 
  (h1 : total_cards = 57) (h2 : num_players = 4) :
  ∃ (cards_per_player : Nat) (unassigned_cards : Nat),
    cards_per_player * num_players + unassigned_cards = total_cards ∧
    cards_per_player = 14 ∧
    unassigned_cards = 1 := by
  sorry

end card_distribution_l3645_364556


namespace max_word_ratio_bound_l3645_364553

/-- Represents a crossword on an n × n grid. -/
structure Crossword (n : ℕ) where
  cells : Set (Fin n × Fin n)
  nonempty : cells.Nonempty

/-- The number of words in a crossword. -/
def num_words (n : ℕ) (c : Crossword n) : ℕ := sorry

/-- The minimum number of words needed to cover a crossword. -/
def min_cover_words (n : ℕ) (c : Crossword n) : ℕ := sorry

/-- Theorem: The maximum ratio of words to minimum cover words is 1 + n/2 -/
theorem max_word_ratio_bound {n : ℕ} (hn : n ≥ 2) (c : Crossword n) :
  (num_words n c : ℚ) / (min_cover_words n c) ≤ 1 + n / 2 := by
  sorry

end max_word_ratio_bound_l3645_364553


namespace problem_solution_l3645_364577

theorem problem_solution (x y : ℝ) (h1 : 3*x + y = 5) (h2 : x + 3*y = 8) :
  5*x^2 + 11*x*y + 5*y^2 = 89 := by
  sorry

end problem_solution_l3645_364577


namespace greatest_integer_problem_l3645_364535

theorem greatest_integer_problem : 
  ⌊100 * (Real.cos (18.5 * π / 180) / Real.sin (17.5 * π / 180))⌋ = 273 := by
  sorry

end greatest_integer_problem_l3645_364535


namespace existence_of_h₁_h₂_l3645_364579

theorem existence_of_h₁_h₂ :
  ∃ (h₁ h₂ : ℝ → ℝ),
    ∀ (g₁ g₂ : ℝ → ℝ) (x : ℝ),
      (∀ s, 1 ≤ g₁ s) →
      (∀ s, 1 ≤ g₂ s) →
      (∃ M, ∀ s, g₁ s ≤ M) →
      (∃ N, ∀ s, g₂ s ≤ N) →
      (⨆ s, (g₁ s) ^ x * g₂ s) = ⨆ t, x * h₁ t + h₂ t :=
by
  sorry

end existence_of_h₁_h₂_l3645_364579


namespace sum_of_coefficients_l3645_364522

/-- A quadratic function y = px^2 + qx + r with specific properties -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  r : ℝ
  is_parabola : True
  vertex_x : p ≠ 0 → -q / (2 * p) = -3
  vertex_y : p ≠ 0 → p * (-3)^2 + q * (-3) + r = 4
  passes_through_origin : p * 0^2 + q * 0 + r = -2
  vertical_symmetry : True

/-- The sum of coefficients p, q, and r equals -20/3 -/
theorem sum_of_coefficients (f : QuadraticFunction) : f.p + f.q + f.r = -20/3 := by
  sorry


end sum_of_coefficients_l3645_364522


namespace remainder_theorem_l3645_364552

theorem remainder_theorem (N : ℤ) (h : ∃ k : ℤ, N = 39 * k + 18) : 
  ∃ m : ℤ, N = 13 * m + 5 := by
sorry

end remainder_theorem_l3645_364552


namespace inequality_statements_l3645_364574

theorem inequality_statements :
  (∀ a b c : ℝ, c ≠ 0 → (a * c^2 < b * c^2 → a < b)) ∧
  (∃ a x y : ℝ, x > y ∧ ¬(-a^2 * x < -a^2 * y)) ∧
  (∀ a b c : ℝ, c ≠ 0 → (a / c^2 < b / c^2 → a < b)) ∧
  (∀ a b : ℝ, a > b → 2 - a < 2 - b) :=
by sorry

end inequality_statements_l3645_364574


namespace fraction_value_l3645_364515

theorem fraction_value (a b : ℚ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end fraction_value_l3645_364515


namespace correct_years_passed_l3645_364525

def initial_ages : List Nat := [19, 34, 37, 42, 48]

def new_stem_leaf_plot : List (Nat × List Nat) := 
  [(1, []), (2, [5, 5]), (3, []), (4, [0, 3, 8]), (5, [4])]

def years_passed (initial : List Nat) (new_plot : List (Nat × List Nat)) : Nat :=
  sorry

theorem correct_years_passed :
  years_passed initial_ages new_stem_leaf_plot = 6 := by sorry

end correct_years_passed_l3645_364525


namespace second_car_speed_l3645_364576

/-- Given two cars starting from opposite ends of a 333-mile highway at the same time,
    with one car traveling at 54 mph and both cars meeting after 3 hours,
    prove that the speed of the second car is 57 mph. -/
theorem second_car_speed (highway_length : ℝ) (time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  highway_length = 333 →
  time = 3 →
  speed1 = 54 →
  speed1 * time + speed2 * time = highway_length →
  speed2 = 57 := by
  sorry

end second_car_speed_l3645_364576


namespace bills_milk_problem_l3645_364557

/-- Represents the problem of determining the amount of milk Bill got from his cow --/
theorem bills_milk_problem (M : ℝ) : 
  M > 0 ∧ 
  (M / 16) * 5 + (M / 8) * 6 + (M / 2) * 3 = 41 → 
  M = 16 :=
by sorry

end bills_milk_problem_l3645_364557


namespace mike_speaker_cost_l3645_364528

/-- The amount Mike spent on speakers -/
def speaker_cost (total_cost new_tire_cost : ℚ) : ℚ :=
  total_cost - new_tire_cost

/-- Theorem: Mike spent $118.54 on speakers -/
theorem mike_speaker_cost : 
  speaker_cost 224.87 106.33 = 118.54 := by sorry

end mike_speaker_cost_l3645_364528


namespace remainder_sum_powers_mod_seven_l3645_364509

theorem remainder_sum_powers_mod_seven :
  (9^6 + 8^7 + 7^8) % 7 = 2 := by
sorry

end remainder_sum_powers_mod_seven_l3645_364509


namespace sqrt_neg_three_squared_l3645_364534

theorem sqrt_neg_three_squared : Real.sqrt ((-3)^2) = 3 := by
  sorry

end sqrt_neg_three_squared_l3645_364534


namespace total_outfits_is_168_l3645_364504

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of ties available -/
def num_ties : ℕ := 7

/-- The number of hats available -/
def num_hats : ℕ := 2

/-- The number of hat options (including not wearing a hat) -/
def hat_options : ℕ := num_hats + 1

/-- The total number of possible outfits -/
def total_outfits : ℕ := num_shirts * num_ties * hat_options

/-- Theorem stating that the total number of outfits is 168 -/
theorem total_outfits_is_168 : total_outfits = 168 := by
  sorry

end total_outfits_is_168_l3645_364504


namespace jason_balloon_count_l3645_364560

/-- Calculates the final number of balloons Jason has after a series of changes. -/
def final_balloon_count (initial_violet : ℕ) (initial_red : ℕ) 
  (violet_given : ℕ) (red_given : ℕ) (violet_acquired : ℕ) : ℕ :=
  let remaining_violet := initial_violet - violet_given + violet_acquired
  let remaining_red := (initial_red - red_given) * 3
  remaining_violet + remaining_red

/-- Proves that Jason ends up with 35 balloons given the initial quantities and changes. -/
theorem jason_balloon_count : 
  final_balloon_count 15 12 3 5 2 = 35 := by
  sorry

end jason_balloon_count_l3645_364560


namespace ladybugs_without_spots_l3645_364518

theorem ladybugs_without_spots (total : Nat) (with_spots : Nat) (without_spots : Nat) : 
  total = 67082 → with_spots = 12170 → without_spots = total - with_spots → without_spots = 54912 := by
  sorry

end ladybugs_without_spots_l3645_364518


namespace intersection_at_midpoint_l3645_364539

/-- Given a line segment from (3,6) to (5,10) and a line x + y = b that
    intersects this segment at its midpoint, prove that b = 12. -/
theorem intersection_at_midpoint (b : ℝ) : 
  (∃ (x y : ℝ), x + y = b ∧ 
    x = (3 + 5) / 2 ∧ 
    y = (6 + 10) / 2) → 
  b = 12 := by
sorry

end intersection_at_midpoint_l3645_364539


namespace teacher_selection_arrangements_l3645_364571

theorem teacher_selection_arrangements (n_male : ℕ) (n_female : ℕ) (n_select : ℕ) : 
  n_male = 5 → n_female = 4 → n_select = 3 →
  (Nat.choose (n_male + n_female) n_select - Nat.choose n_male n_select - Nat.choose n_female n_select) = 70 := by
  sorry

end teacher_selection_arrangements_l3645_364571


namespace trig_identity_l3645_364583

theorem trig_identity (α : Real) (h : Real.sin α + Real.cos α = 1/5) :
  (Real.sin α - Real.cos α)^2 = 49/25 ∧ Real.sin α^3 + Real.cos α^3 = 37/125 := by
  sorry

end trig_identity_l3645_364583


namespace unique_function_exists_l3645_364500

/-- A function satisfying the given inequality for all real x, y, z and fixed positive integer k -/
def SatisfiesInequality (f : ℝ → ℝ) (k : ℕ+) : Prop :=
  ∀ x y z : ℝ, f (x * y) + f (x + z) + k * f x * f (y * z) ≥ k^2

/-- There exists only one function satisfying the inequality -/
theorem unique_function_exists (k : ℕ+) : ∃! f : ℝ → ℝ, SatisfiesInequality f k := by
  sorry

end unique_function_exists_l3645_364500


namespace trivia_team_score_l3645_364565

/-- Represents a trivia team with their scores -/
structure TriviaTeam where
  totalMembers : Nat
  absentMembers : Nat
  scores : List Nat

/-- Calculates the total score of a trivia team -/
def totalScore (team : TriviaTeam) : Nat :=
  team.scores.sum

/-- Theorem: The trivia team's total score is 26 points -/
theorem trivia_team_score : 
  ∀ (team : TriviaTeam), 
    team.totalMembers = 8 → 
    team.absentMembers = 3 → 
    team.scores = [4, 6, 8, 8] → 
    totalScore team = 26 := by
  sorry

end trivia_team_score_l3645_364565


namespace distance_between_points_l3645_364584

/-- The distance between points A and B -/
def distance : ℝ := sorry

/-- The speed of the first pedestrian -/
def speed1 : ℝ := sorry

/-- The speed of the second pedestrian -/
def speed2 : ℝ := sorry

theorem distance_between_points (h1 : distance / (2 * speed1) = 15 / speed2)
                                (h2 : 24 / speed1 = distance / (2 * speed2))
                                (h3 : distance / speed1 = distance / speed2) :
  distance = 40 := by sorry

end distance_between_points_l3645_364584


namespace remainder_of_x_divided_by_82_l3645_364523

theorem remainder_of_x_divided_by_82 (x : ℤ) (k m R : ℤ) 
  (h1 : x = 82 * k + R)
  (h2 : 0 ≤ R ∧ R < 82)
  (h3 : x + 7 = 41 * m + 12) :
  R = 5 := by
  sorry

end remainder_of_x_divided_by_82_l3645_364523


namespace typist_salary_problem_l3645_364513

/-- Proves that if a salary S is increased by 10% and then decreased by 5%,
    resulting in Rs. 6270, then the original salary S was Rs. 6000. -/
theorem typist_salary_problem (S : ℝ) : 
  (S * 1.1 * 0.95 = 6270) → S = 6000 := by
  sorry

end typist_salary_problem_l3645_364513


namespace units_digit_of_result_l3645_364569

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of the calculation -/
def result : ℕ := 7 * 18 * 1978 - 7^4

theorem units_digit_of_result : unitsDigit result = 7 := by
  sorry

end units_digit_of_result_l3645_364569


namespace inequality_preservation_l3645_364542

theorem inequality_preservation (a b : ℝ) (h : a < b) : -2 + 2*a < -2 + 2*b := by
  sorry

end inequality_preservation_l3645_364542


namespace division_problem_l3645_364595

theorem division_problem (x : ℝ) (h : 10 / x = 2) : 20 / x = 4 := by
  sorry

end division_problem_l3645_364595


namespace or_implies_at_least_one_true_l3645_364572

theorem or_implies_at_least_one_true (p q : Prop) : 
  (p ∨ q) → (p ∨ q) := by sorry

end or_implies_at_least_one_true_l3645_364572


namespace parabola_points_theorem_l3645_364549

/-- Parabola structure -/
structure Parabola where
  f : ℝ → ℝ
  eq : ∀ x, f x ^ 2 = 8 * x

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y ^ 2 = 8 * x

/-- Theorem about two points on a parabola -/
theorem parabola_points_theorem (p : Parabola) 
    (A B : PointOnParabola p) (F : ℝ × ℝ) :
  A.y + B.y = 8 →
  F = (2, 0) →
  (B.y - A.y) / (B.x - A.x) = 1 ∧
  ((A.x - F.1) ^ 2 + (A.y - F.2) ^ 2) ^ (1/2 : ℝ) +
  ((B.x - F.1) ^ 2 + (B.y - F.2) ^ 2) ^ (1/2 : ℝ) = 16 :=
by sorry

end parabola_points_theorem_l3645_364549


namespace min_value_complex_expression_l3645_364563

/-- Given a complex number z where |z - 3 + 2i| = 3, 
    the minimum value of |z + 1 - i|^2 + |z - 7 + 3i|^2 is 86. -/
theorem min_value_complex_expression (z : ℂ) 
  (h : Complex.abs (z - (3 - 2*Complex.I)) = 3) : 
  (Complex.abs (z + (1 - Complex.I)))^2 + (Complex.abs (z - (7 - 3*Complex.I)))^2 ≥ 86 ∧ 
  ∃ w : ℂ, Complex.abs (w - (3 - 2*Complex.I)) = 3 ∧ 
    (Complex.abs (w + (1 - Complex.I)))^2 + (Complex.abs (w - (7 - 3*Complex.I)))^2 = 86 :=
by sorry

end min_value_complex_expression_l3645_364563


namespace cycling_speeds_l3645_364561

/-- Represents the cycling speeds of four people -/
structure CyclingGroup where
  henry_speed : ℝ
  liz_speed : ℝ
  jack_speed : ℝ
  tara_speed : ℝ

/-- The cycling group satisfies the given conditions -/
def satisfies_conditions (g : CyclingGroup) : Prop :=
  g.henry_speed = 5 ∧
  g.liz_speed = 3/4 * g.henry_speed ∧
  g.jack_speed = 6/5 * g.liz_speed ∧
  g.tara_speed = 9/8 * g.jack_speed

/-- Theorem stating the cycling speeds of Jack and Tara -/
theorem cycling_speeds (g : CyclingGroup) 
  (h : satisfies_conditions g) : 
  g.jack_speed = 4.5 ∧ g.tara_speed = 5.0625 := by
  sorry

#check cycling_speeds

end cycling_speeds_l3645_364561


namespace square_between_endpoints_l3645_364592

theorem square_between_endpoints (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_cd : c * d = 1) :
  Real.sqrt ((a + c) * (b + d)) ≥ Real.sqrt (a * b) + 1 := by
  sorry

end square_between_endpoints_l3645_364592


namespace f_odd_f_2a_f_3a_f_monotone_decreasing_l3645_364578

/-- Function with specific properties -/
noncomputable def f (x : ℝ) : ℝ := sorry

/-- Positive constant a -/
noncomputable def a : ℝ := sorry

/-- Domain of f -/
def domain (x : ℝ) : Prop := ∀ k : ℤ, x ≠ k * Real.pi

axiom f_domain : ∀ x : ℝ, domain x → f x ≠ 0

axiom f_equation : ∀ x y : ℝ, domain x → domain y → 
  f (x - y) = (f x * f y + 1) / (f y - f x)

axiom f_a : f a = 1

axiom a_pos : a > 0

axiom f_pos_interval : ∀ x : ℝ, 0 < x → x < 2 * a → f x > 0

/-- f is an odd function -/
theorem f_odd : ∀ x : ℝ, domain x → f (-x) = -f x := by sorry

/-- f(2a) = 0 -/
theorem f_2a : f (2 * a) = 0 := by sorry

/-- f(3a) = -1 -/
theorem f_3a : f (3 * a) = -1 := by sorry

/-- f is monotonically decreasing on [2a, 3a] -/
theorem f_monotone_decreasing : 
  ∀ x y : ℝ, 2 * a ≤ x → x < y → y ≤ 3 * a → f x > f y := by sorry

end f_odd_f_2a_f_3a_f_monotone_decreasing_l3645_364578


namespace equation_solution_l3645_364516

theorem equation_solution :
  ∃ (x : ℝ), x ≠ -3 ∧ (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 5) ∧ x = -9 :=
by sorry

end equation_solution_l3645_364516


namespace log_23_between_consecutive_integers_l3645_364505

theorem log_23_between_consecutive_integers :
  ∃ (a b : ℤ), (a + 1 = b) ∧ (a < Real.log 23 / Real.log 10) ∧ (Real.log 23 / Real.log 10 < b) ∧ (a + b = 3) := by
  sorry

end log_23_between_consecutive_integers_l3645_364505


namespace complex_equation_solution_l3645_364512

theorem complex_equation_solution (x : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (1 - 2*i) * (x + i) = 4 - 3*i) : 
  x = 2 := by sorry

end complex_equation_solution_l3645_364512


namespace least_three_digit_8_heavy_l3645_364580

def is_8_heavy (n : ℕ) : Prop := n % 8 > 6

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_8_heavy : 
  (∀ n : ℕ, is_three_digit n → is_8_heavy n → 103 ≤ n) ∧ 
  is_three_digit 103 ∧ 
  is_8_heavy 103 :=
sorry

end least_three_digit_8_heavy_l3645_364580


namespace robotics_club_theorem_l3645_364508

theorem robotics_club_theorem (total : ℕ) (cs : ℕ) (elec : ℕ) (both : ℕ)
  (h_total : total = 75)
  (h_cs : cs = 44)
  (h_elec : elec = 40)
  (h_both : both = 25) :
  total - (cs + elec - both) = 16 := by
  sorry

end robotics_club_theorem_l3645_364508


namespace final_coin_count_l3645_364503

/-- Represents the number of coins in the jar at each hour -/
def coin_count : Fin 11 → ℕ
| 0 => 0  -- Initial state
| 1 => 20
| 2 => coin_count 1 + 30
| 3 => coin_count 2 + 30
| 4 => coin_count 3 + 40
| 5 => coin_count 4 - (coin_count 4 * 20 / 100)
| 6 => coin_count 5 + 50
| 7 => coin_count 6 + 60
| 8 => coin_count 7 - (coin_count 7 / 5)
| 9 => coin_count 8 + 70
| 10 => coin_count 9 - (coin_count 9 * 15 / 100)

theorem final_coin_count : coin_count 10 = 200 := by
  sorry

end final_coin_count_l3645_364503


namespace symmetry_about_origin_l3645_364502

/-- Given a point (x, y) in R^2, its symmetrical point about the origin is (-x, -y) -/
def symmetrical_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

/-- The original point -/
def original_point : ℝ × ℝ := (2, -3)

/-- The proposed symmetrical point -/
def proposed_symmetrical_point : ℝ × ℝ := (-2, 3)

theorem symmetry_about_origin :
  symmetrical_point original_point = proposed_symmetrical_point :=
by sorry

end symmetry_about_origin_l3645_364502


namespace solution_set_of_equation_l3645_364551

def is_solution (x : ℝ) : Prop :=
  -2*x > 0 ∧ 3 - x^2 > 0 ∧ -2*x = 3 - x^2

theorem solution_set_of_equation : 
  {x : ℝ | is_solution x} = {-1} := by sorry

end solution_set_of_equation_l3645_364551


namespace quadratic_inequality_range_l3645_364562

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + a ≥ 0) → a ∈ Set.Icc 0 4 := by
  sorry

end quadratic_inequality_range_l3645_364562


namespace sum_of_three_numbers_l3645_364520

theorem sum_of_three_numbers (x y z : ℝ) 
  (sum_xy : x + y = 29)
  (sum_yz : y + z = 46)
  (sum_zx : z + x = 53) :
  x + y + z = 64 := by
sorry

end sum_of_three_numbers_l3645_364520


namespace polynomial_alternating_sum_l3645_364517

theorem polynomial_alternating_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ - a₁ + a₂ - a₃ + a₄ = 1 := by
  sorry

end polynomial_alternating_sum_l3645_364517


namespace high_school_math_club_payment_l3645_364587

theorem high_school_math_club_payment (B : ℕ) : 
  B < 10 → (∃ k : ℤ, 200 + 10 * B + 5 = 13 * k) → B = 1 :=
by sorry

end high_school_math_club_payment_l3645_364587


namespace larger_number_proof_l3645_364599

theorem larger_number_proof (x y : ℝ) (sum_eq : x + y = 30) (diff_eq : x - y = 4) : 
  max x y = 17 := by
  sorry

end larger_number_proof_l3645_364599


namespace room_length_calculation_l3645_364536

/-- Given a room with specified width, total paving cost, and paving rate per square meter,
    calculate the length of the room. -/
theorem room_length_calculation (width : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  width = 3.75 →
  total_cost = 6187.5 →
  rate_per_sqm = 300 →
  (total_cost / rate_per_sqm) / width = 5.5 :=
by
  sorry

end room_length_calculation_l3645_364536


namespace minimum_blue_beads_l3645_364567

/-- Represents the color of a bead -/
inductive BeadColor
  | Red
  | Blue
  | Green

/-- Represents a necklace as a cyclic list of bead colors -/
def Necklace := List BeadColor

/-- Returns true if the necklace satisfies the condition that each red bead has neighbors of different colors -/
def redBeadCondition (n : Necklace) : Prop := sorry

/-- Returns true if the necklace satisfies the condition that any segment between two green beads contains at least one blue bead -/
def greenSegmentCondition (n : Necklace) : Prop := sorry

/-- Counts the number of blue beads in the necklace -/
def countBlueBeads (n : Necklace) : Nat := sorry

theorem minimum_blue_beads (n : Necklace) :
  n.length = 175 →
  redBeadCondition n →
  greenSegmentCondition n →
  countBlueBeads n ≥ 30 ∧ ∃ (m : Necklace), m.length = 175 ∧ redBeadCondition m ∧ greenSegmentCondition m ∧ countBlueBeads m = 30 :=
sorry

end minimum_blue_beads_l3645_364567


namespace interest_rate_calculation_l3645_364530

theorem interest_rate_calculation (total amount : ℕ) (first_part : ℕ) (first_rate : ℚ) (yearly_income : ℕ) : 
  total = 2500 →
  first_part = 2000 →
  first_rate = 5/100 →
  yearly_income = 130 →
  ∃ second_rate : ℚ,
    (first_part * first_rate + (total - first_part) * second_rate = yearly_income) ∧
    second_rate = 6/100 := by
  sorry

end interest_rate_calculation_l3645_364530


namespace multiple_properties_l3645_364546

-- Define x and y as integers
variable (x y : ℤ)

-- Define the conditions
variable (h1 : ∃ k : ℤ, x = 8 * k)
variable (h2 : ∃ m : ℤ, y = 12 * m)

-- Theorem to prove
theorem multiple_properties :
  (∃ n : ℤ, y = 4 * n) ∧ (∃ p : ℤ, x - y = 4 * p) :=
by sorry

end multiple_properties_l3645_364546


namespace ages_solution_l3645_364589

def mother_daughter_ages (daughter_age : ℕ) (mother_age : ℕ) : Prop :=
  (mother_age = daughter_age + 45) ∧
  (mother_age - 5 = 6 * (daughter_age - 5))

theorem ages_solution : ∃ (daughter_age : ℕ) (mother_age : ℕ),
  mother_daughter_ages daughter_age mother_age ∧
  daughter_age = 14 ∧ mother_age = 59 := by
  sorry

end ages_solution_l3645_364589


namespace tangent_line_y_intercept_l3645_364596

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Predicate to check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  ∃ (p : ℝ × ℝ), (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
                 p.2 = l.slope * p.1 + l.yIntercept

theorem tangent_line_y_intercept : 
  ∀ (l : Line) (c1 c2 : Circle),
    c1.center = (3, 0) →
    c1.radius = 3 →
    c2.center = (8, 0) →
    c2.radius = 2 →
    isTangent l c1 →
    isTangent l c2 →
    (∃ (p1 p2 : ℝ × ℝ), 
      isTangent l c1 ∧ 
      isTangent l c2 ∧ 
      p1.2 > 0 ∧ 
      p2.2 > 0) →
    l.yIntercept = Real.sqrt 5 :=
sorry

end tangent_line_y_intercept_l3645_364596


namespace no_integer_solutions_l3645_364570

theorem no_integer_solutions : ¬∃ (x y z : ℤ),
  (x^2 - 4*x*y + 3*y^2 - z^2 = 24) ∧
  (-x^2 + 3*y*z + 5*z^2 = 60) ∧
  (x^2 + 2*x*y + 5*z^2 = 85) :=
by sorry

end no_integer_solutions_l3645_364570


namespace arithmetic_mean_fractions_l3645_364585

theorem arithmetic_mean_fractions (x a : ℝ) (hx : x ≠ 0) :
  (1 / 2) * ((x + 2 * a) / x + (x - 3 * a) / x) = 1 - a / (2 * x) := by
  sorry

end arithmetic_mean_fractions_l3645_364585


namespace smallest_norm_u_l3645_364526

theorem smallest_norm_u (u : ℝ × ℝ) (h : ‖u + (5, 2)‖ = 10) :
  ∃ (v : ℝ × ℝ), ‖v‖ = 10 - Real.sqrt 29 ∧ ∀ w : ℝ × ℝ, ‖w + (5, 2)‖ = 10 → ‖v‖ ≤ ‖w‖ := by
  sorry

end smallest_norm_u_l3645_364526


namespace first_earthquake_collapse_l3645_364507

/-- Represents the number of buildings collapsed in the first earthquake -/
def first_collapse : ℕ := sorry

/-- Represents the total number of collapsed buildings after four earthquakes -/
def total_collapse : ℕ := 60

/-- Theorem stating that the number of buildings collapsed in the first earthquake is 4 -/
theorem first_earthquake_collapse : 
  (first_collapse + 2 * first_collapse + 4 * first_collapse + 8 * first_collapse = total_collapse) → 
  first_collapse = 4 := by
  sorry

end first_earthquake_collapse_l3645_364507


namespace alternating_sequence_sum_l3645_364533

def alternating_sequence (first last step : ℕ) : List ℤ :=
  let n := (first - last) / step + 1
  List.range n |> List.map (λ i => first - i * step) |> List.map (λ x => if x % (2 * step) = 0 then x else -x)

theorem alternating_sequence_sum (first last step : ℕ) :
  first > last ∧ step > 0 ∧ (first - last) % step = 0 →
  List.sum (alternating_sequence first last step) = 520 :=
by
  sorry

#eval List.sum (alternating_sequence 1050 20 20)

end alternating_sequence_sum_l3645_364533


namespace problem_4_l3645_364501

theorem problem_4 (x y : ℝ) (hx : x = 1) (hy : y = 2^100) :
  (x + 2*y)^2 + (x + 2*y)*(x - 2*y) - 4*x*y = 2 := by
sorry

end problem_4_l3645_364501


namespace min_players_team_l3645_364545

theorem min_players_team (n : ℕ) : 
  (n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) → n ≥ 7920 :=
by sorry

end min_players_team_l3645_364545


namespace envelope_stuffing_l3645_364548

/-- The total number of envelopes Rachel needs to stuff -/
def total_envelopes : ℕ := 1500

/-- The total time Rachel has to complete the task -/
def total_time : ℕ := 8

/-- The number of envelopes Rachel stuffs in the first hour -/
def first_hour : ℕ := 135

/-- The number of envelopes Rachel stuffs in the second hour -/
def second_hour : ℕ := 141

/-- The number of envelopes Rachel needs to stuff per hour to finish the job -/
def required_rate : ℕ := 204

theorem envelope_stuffing :
  total_envelopes = first_hour + second_hour + required_rate * (total_time - 2) := by
  sorry

end envelope_stuffing_l3645_364548


namespace cube_root_fraction_equivalence_l3645_364529

theorem cube_root_fraction_equivalence :
  let x : ℝ := 12.75
  let y : ℚ := 51 / 4
  x = y →
  (6 / x) ^ (1/3 : ℝ) = 2 / (17 ^ (1/3 : ℝ)) :=
by
  sorry

end cube_root_fraction_equivalence_l3645_364529


namespace valid_string_count_l3645_364543

/-- A run is a set of consecutive identical letters in a string. -/
def Run := ℕ

/-- A valid string is a 10-letter string composed of A's and B's where no more than 3 consecutive letters are the same. -/
def ValidString := Fin 10 → Bool

/-- The number of runs in a valid string is between 4 and 10, inclusive. -/
def ValidRunCount (n : ℕ) : Prop := 4 ≤ n ∧ n ≤ 10

/-- The generating function for a single run is x + x^2 + x^3. -/
def SingleRunGeneratingFunction (x : ℝ) : ℝ := x + x^2 + x^3

/-- The coefficient of x^(10-n) in the expansion of ((1-x^3)^n) / ((1-x)^n). -/
def Coefficient (n : ℕ) : ℕ := sorry

/-- The total number of valid strings. -/
def TotalValidStrings : ℕ := 2 * (Coefficient 4 + Coefficient 5 + Coefficient 6 + Coefficient 7 + Coefficient 8 + Coefficient 9 + Coefficient 10)

theorem valid_string_count : TotalValidStrings = 548 := by sorry

end valid_string_count_l3645_364543


namespace closest_ratio_l3645_364537

/-- The annual interest rate -/
def interest_rate : ℝ := 0.05

/-- The number of years -/
def years : ℕ := 10

/-- The ratio of final amount to initial amount after compound interest -/
def ratio : ℝ := (1 + interest_rate) ^ years

/-- The given options for the ratio -/
def options : List ℝ := [1.5, 1.6, 1.7, 1.8]

/-- Theorem stating that 1.6 is the closest option to the actual ratio -/
theorem closest_ratio : 
  ∃ (x : ℝ), x ∈ options ∧ ∀ (y : ℝ), y ∈ options → |ratio - x| ≤ |ratio - y| ∧ x = 1.6 :=
sorry

end closest_ratio_l3645_364537


namespace solution_set_a_range_l3645_364544

-- Define the functions f and g
def f (x a : ℝ) : ℝ := |x - 1| + |x + a|
def g (a : ℝ) : ℝ := a^2 - a - 2

-- Part 1
theorem solution_set (x : ℝ) :
  (f x 3 > g 3 + 2) ↔ (x < -4 ∨ x > 2) :=
sorry

-- Part 2
theorem a_range (a : ℝ) :
  (∀ x ∈ Set.Icc (-a) 1, f x a ≤ g a) → a ≥ 3 :=
sorry

end solution_set_a_range_l3645_364544


namespace geometric_sequence_n_l3645_364568

theorem geometric_sequence_n (a₁ q aₙ : ℚ) (n : ℕ) : 
  a₁ = 1/2 → q = 1/2 → aₙ = 1/32 → aₙ = a₁ * q^(n-1) → n = 5 := by
  sorry

end geometric_sequence_n_l3645_364568


namespace right_triangle_area_l3645_364582

/-- The area of a right triangle with legs of 60 feet and 80 feet is 345600 square inches -/
theorem right_triangle_area : 
  let leg1_feet : ℝ := 60
  let leg2_feet : ℝ := 80
  let inches_per_foot : ℝ := 12
  let leg1_inches : ℝ := leg1_feet * inches_per_foot
  let leg2_inches : ℝ := leg2_feet * inches_per_foot
  let area : ℝ := (1/2) * leg1_inches * leg2_inches
  area = 345600 := by sorry

end right_triangle_area_l3645_364582


namespace number_of_diagonals_sum_of_interior_angles_l3645_364591

-- Define the number of sides
def n : ℕ := 150

-- Theorem for the number of diagonals
theorem number_of_diagonals : 
  n * (n - 3) / 2 = 11025 :=
sorry

-- Theorem for the sum of interior angles
theorem sum_of_interior_angles : 
  180 * (n - 2) = 26640 :=
sorry

end number_of_diagonals_sum_of_interior_angles_l3645_364591


namespace inequality_equivalence_l3645_364590

theorem inequality_equivalence :
  (∀ x : ℝ, |x + 1| + |x - 1| ≥ a) ↔ (a ≤ 2) :=
by sorry

end inequality_equivalence_l3645_364590


namespace okeydokey_earthworms_calculation_l3645_364566

/-- The number of apples Okeydokey invested -/
def okeydokey_apples : ℕ := 5

/-- The number of apples Artichokey invested -/
def artichokey_apples : ℕ := 7

/-- The total number of earthworms in the box -/
def total_earthworms : ℕ := 60

/-- The number of earthworms Okeydokey should receive -/
def okeydokey_earthworms : ℕ := 25

/-- Theorem stating that Okeydokey should receive 25 earthworms -/
theorem okeydokey_earthworms_calculation :
  (okeydokey_apples : ℚ) / (okeydokey_apples + artichokey_apples : ℚ) * total_earthworms = okeydokey_earthworms := by
  sorry

end okeydokey_earthworms_calculation_l3645_364566


namespace no_positive_integer_solution_l3645_364506

theorem no_positive_integer_solution : 
  ¬∃ (n m : ℕ+), n^4 - m^4 = 42 := by
  sorry

end no_positive_integer_solution_l3645_364506


namespace no_same_color_in_large_rectangle_l3645_364531

/-- A coloring of the plane is a function from pairs of integers to colors. -/
def Coloring (Color : Type) := ℤ × ℤ → Color

/-- A rectangle in the plane is defined by its top-left and bottom-right corners. -/
structure Rectangle :=
  (top_left : ℤ × ℤ)
  (bottom_right : ℤ × ℤ)

/-- The perimeter of a rectangle. -/
def Rectangle.perimeter (r : Rectangle) : ℤ :=
  2 * (r.bottom_right.1 - r.top_left.1 + r.top_left.2 - r.bottom_right.2)

/-- A predicate that checks if a coloring satisfies the condition that
    no rectangle with perimeter 100 contains two squares of the same color. -/
def valid_coloring (c : Coloring (Fin 1201)) : Prop :=
  ∀ r : Rectangle, r.perimeter = 100 →
    ∀ x y : ℤ × ℤ, x ≠ y →
      x.1 ∈ Set.Icc r.top_left.1 r.bottom_right.1 →
      x.2 ∈ Set.Icc r.bottom_right.2 r.top_left.2 →
      y.1 ∈ Set.Icc r.top_left.1 r.bottom_right.1 →
      y.2 ∈ Set.Icc r.bottom_right.2 r.top_left.2 →
      c x ≠ c y

/-- The main theorem: if a coloring is valid, then no 1×1201 or 1201×1 rectangle
    contains two squares of the same color. -/
theorem no_same_color_in_large_rectangle
  (c : Coloring (Fin 1201)) (h : valid_coloring c) :
  (∀ r : Rectangle,
    (r.bottom_right.1 - r.top_left.1 = 1200 ∧ r.top_left.2 - r.bottom_right.2 = 0) ∨
    (r.bottom_right.1 - r.top_left.1 = 0 ∧ r.top_left.2 - r.bottom_right.2 = 1200) →
    ∀ x y : ℤ × ℤ, x ≠ y →
      x.1 ∈ Set.Icc r.top_left.1 r.bottom_right.1 →
      x.2 ∈ Set.Icc r.bottom_right.2 r.top_left.2 →
      y.1 ∈ Set.Icc r.top_left.1 r.bottom_right.1 →
      y.2 ∈ Set.Icc r.bottom_right.2 r.top_left.2 →
      c x ≠ c y) :=
by sorry

end no_same_color_in_large_rectangle_l3645_364531


namespace perfect_squares_theorem_l3645_364594

theorem perfect_squares_theorem (x y z : ℕ+) 
  (h_coprime : ∀ d : ℕ, d > 1 → ¬(d ∣ x ∧ d ∣ y ∧ d ∣ z))
  (h_eq : (x : ℚ)⁻¹ + (y : ℚ)⁻¹ = (z : ℚ)⁻¹) :
  ∃ (a b : ℕ), 
    (x : ℤ) - (z : ℤ) = a^2 ∧ 
    (y : ℤ) - (z : ℤ) = b^2 ∧ 
    (x : ℤ) + (y : ℤ) = (a + b)^2 := by
  sorry

end perfect_squares_theorem_l3645_364594


namespace f_properties_l3645_364593

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x - x

theorem f_properties :
  let f := f
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ > f x₂) ∧
  (∀ x, x > 0 → f x ≥ -1) ∧
  f 1 = -1 :=
by sorry

end f_properties_l3645_364593


namespace four_solutions_l3645_364555

-- Define the piecewise function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else k * x + 2

-- Theorem statement
theorem four_solutions (k : ℝ) (h : k > 0) :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x ∈ s, |f k x| = 1 :=
sorry

end four_solutions_l3645_364555


namespace prime_square_minus_one_divisible_by_thirty_l3645_364510

theorem prime_square_minus_one_divisible_by_thirty {p : ℕ} (hp : Prime p) (hp_ge_7 : p ≥ 7) :
  30 ∣ p^2 - 1 := by
  sorry

end prime_square_minus_one_divisible_by_thirty_l3645_364510


namespace circle_and_trajectory_l3645_364538

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 4)^2 + p.2^2 = 5}

-- Define points M and N
def M : ℝ × ℝ := (5, 2)
def N : ℝ × ℝ := (3, 2)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem circle_and_trajectory :
  (M ∈ circle_C) ∧ 
  (N ∈ circle_C) ∧ 
  (∀ p ∈ circle_C, p.1 = 4 → p.2 = 0) →
  (∀ A ∈ circle_C, 
    ∃ P : ℝ × ℝ, 
      (P.1 - O.1 = 2 * (A.1 - O.1)) ∧ 
      (P.2 - O.2 = 2 * (A.2 - O.2)) ∧
      (P.1 - 8)^2 + P.2^2 = 20) := by
  sorry

end circle_and_trajectory_l3645_364538


namespace total_trees_is_86_l3645_364541

/-- Calculates the number of trees that can be planted on a street --/
def treesOnStreet (length : ℕ) (spacing : ℕ) : ℕ :=
  (length / spacing) + 1

/-- The total number of trees that can be planted on all five streets --/
def totalTrees : ℕ :=
  treesOnStreet 151 14 + treesOnStreet 210 18 + treesOnStreet 275 12 +
  treesOnStreet 345 20 + treesOnStreet 475 22

theorem total_trees_is_86 : totalTrees = 86 := by
  sorry

end total_trees_is_86_l3645_364541


namespace largest_positive_integer_solution_l3645_364586

theorem largest_positive_integer_solution :
  ∀ x : ℕ+, 2 * (x + 1) ≥ 5 * x - 3 ↔ x ≤ 1 :=
by sorry

end largest_positive_integer_solution_l3645_364586
