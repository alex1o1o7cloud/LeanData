import Mathlib

namespace NUMINAMATH_CALUDE_negative_one_to_zero_power_l666_66672

theorem negative_one_to_zero_power : (-1 : ℝ) ^ (0 : ℕ) = 1 := by sorry

end NUMINAMATH_CALUDE_negative_one_to_zero_power_l666_66672


namespace NUMINAMATH_CALUDE_least_sum_equation_l666_66645

theorem least_sum_equation (x y z : ℕ+) 
  (h1 : 6 * z.val = 2 * x.val) 
  (h2 : x.val + y.val + z.val = 26) : 
  6 * z.val = 36 := by
sorry

end NUMINAMATH_CALUDE_least_sum_equation_l666_66645


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l666_66600

theorem coloring_book_shelves (initial_stock : ℝ) (acquired : ℝ) (books_per_shelf : ℝ) 
  (h1 : initial_stock = 40.0)
  (h2 : acquired = 20.0)
  (h3 : books_per_shelf = 4.0) :
  (initial_stock + acquired) / books_per_shelf = 15 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l666_66600


namespace NUMINAMATH_CALUDE_soccer_lineup_theorem_l666_66660

/-- The number of ways to choose a soccer lineup -/
def soccer_lineup_count : ℕ := 18 * (Nat.choose 17 4) * (Nat.choose 13 3) * (Nat.choose 10 3)

/-- Theorem stating the number of possible soccer lineups -/
theorem soccer_lineup_theorem : soccer_lineup_count = 147497760 := by
  sorry

end NUMINAMATH_CALUDE_soccer_lineup_theorem_l666_66660


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l666_66663

/-- Given that angle α satisfies the conditions sin(2α) < 0 and sin(α) - cos(α) < 0,
    prove that α is in the fourth quadrant. -/
theorem angle_in_fourth_quadrant (α : Real) 
    (h1 : Real.sin (2 * α) < 0) 
    (h2 : Real.sin α - Real.cos α < 0) : 
  π < α ∧ α < (3 * π) / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l666_66663


namespace NUMINAMATH_CALUDE_shortest_chord_through_focus_of_ellipse_l666_66630

/-- 
Given an ellipse with equation x²/16 + y²/9 = 1, 
prove that the length of the shortest chord passing through a focus is 9/2.
-/
theorem shortest_chord_through_focus_of_ellipse :
  let ellipse := fun (x y : ℝ) => x^2/16 + y^2/9 = 1
  ∃ (f : ℝ × ℝ), 
    (ellipse f.1 f.2) ∧ 
    (∀ (p q : ℝ × ℝ), ellipse p.1 p.2 ∧ ellipse q.1 q.2 ∧ 
      (∃ (t : ℝ), (1 - t) • f + t • p = q) →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ 9/2) ∧
    (∃ (p q : ℝ × ℝ), ellipse p.1 p.2 ∧ ellipse q.1 q.2 ∧
      (∃ (t : ℝ), (1 - t) • f + t • p = q) ∧
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_shortest_chord_through_focus_of_ellipse_l666_66630


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_angle_range_l666_66647

/-- The range of inclination angles for which a line intersects an ellipse at two distinct points -/
theorem line_ellipse_intersection_angle_range 
  (A : ℝ × ℝ) 
  (l : ℝ → ℝ × ℝ) 
  (α : ℝ) 
  (ellipse : ℝ × ℝ → Prop) : 
  A = (-2, 0) →
  (∀ t, l t = (-2 + t * Real.cos α, t * Real.sin α)) →
  (ellipse (x, y) ↔ x^2 / 2 + y^2 = 1) →
  (∃ B C, B ≠ C ∧ ellipse B ∧ ellipse C ∧ ∃ t₁ t₂, l t₁ = B ∧ l t₂ = C) ↔
  (0 ≤ α ∧ α < Real.arcsin (Real.sqrt 3 / 3)) ∨ 
  (Real.pi - Real.arcsin (Real.sqrt 3 / 3) < α ∧ α < Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_angle_range_l666_66647


namespace NUMINAMATH_CALUDE_arithmetic_mean_increase_l666_66613

theorem arithmetic_mean_increase (a b c d e : ℝ) :
  let original_set := [a, b, c, d, e]
  let new_set := original_set.map (λ x => x + 15)
  let original_mean := (a + b + c + d + e) / 5
  let new_mean := (new_set.sum) / 5
  new_mean = original_mean + 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_increase_l666_66613


namespace NUMINAMATH_CALUDE_toy_truck_cost_l666_66674

/-- The amount spent on a toy truck given initial amount, pencil case cost, and remaining amount -/
theorem toy_truck_cost (initial : ℝ) (pencil_case : ℝ) (remaining : ℝ) :
  initial = 10 → pencil_case = 2 → remaining = 5 →
  initial - pencil_case - remaining = 3 := by
  sorry

end NUMINAMATH_CALUDE_toy_truck_cost_l666_66674


namespace NUMINAMATH_CALUDE_prob_no_standing_pairs_10_l666_66681

/-- Represents the number of valid arrangements for n people where no two adjacent people form a standing pair -/
def b : ℕ → ℕ
| 0 => 1
| 1 => 2
| n+2 => 3 * b (n+1) - b n

/-- The probability of no standing pairs for n people -/
def prob_no_standing_pairs (n : ℕ) : ℚ :=
  (b n : ℚ) / (2^n : ℚ)

theorem prob_no_standing_pairs_10 :
  prob_no_standing_pairs 10 = 31 / 128 := by sorry

end NUMINAMATH_CALUDE_prob_no_standing_pairs_10_l666_66681


namespace NUMINAMATH_CALUDE_distance_between_cities_l666_66628

/-- The distance between two cities given the speeds of two travelers and their meeting point --/
theorem distance_between_cities (john_speed lewis_speed : ℝ) (meeting_point : ℝ) : 
  john_speed = 40 →
  lewis_speed = 60 →
  meeting_point = 160 →
  ∃ (distance : ℝ), 
    distance > 0 ∧ 
    (distance + meeting_point) / lewis_speed = (distance - meeting_point) / john_speed ∧
    distance = 800 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l666_66628


namespace NUMINAMATH_CALUDE_baseball_cards_count_l666_66669

theorem baseball_cards_count (initial_cards additional_cards : ℕ) 
  (h1 : initial_cards = 87)
  (h2 : additional_cards = 13) :
  initial_cards + additional_cards = 100 :=
by sorry

end NUMINAMATH_CALUDE_baseball_cards_count_l666_66669


namespace NUMINAMATH_CALUDE_average_first_five_subjects_l666_66688

/-- Given the average marks for 6 subjects and the marks for the 6th subject,
    calculate the average marks for the first 5 subjects. -/
theorem average_first_five_subjects
  (total_subjects : ℕ)
  (average_six_subjects : ℚ)
  (marks_sixth_subject : ℕ)
  (h1 : total_subjects = 6)
  (h2 : average_six_subjects = 78)
  (h3 : marks_sixth_subject = 98) :
  (average_six_subjects * total_subjects - marks_sixth_subject) / (total_subjects - 1) = 74 :=
by sorry

end NUMINAMATH_CALUDE_average_first_five_subjects_l666_66688


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l666_66614

theorem sum_of_roots_quadratic (m n : ℝ) : 
  (m^2 + 2*m - 1 = 0) → (n^2 + 2*n - 1 = 0) → (m + n = -2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l666_66614


namespace NUMINAMATH_CALUDE_weight_of_Na2Ca_CO3_2_l666_66653

-- Define molar masses of elements
def Na_mass : ℝ := 22.99
def Ca_mass : ℝ := 40.08
def C_mass : ℝ := 12.01
def O_mass : ℝ := 16.00

-- Define the number of atoms in Na2Ca(CO3)2
def Na_count : ℕ := 2
def Ca_count : ℕ := 1
def C_count : ℕ := 2
def O_count : ℕ := 6

-- Define the number of moles of Na2Ca(CO3)2
def moles : ℝ := 3.75

-- Define the molar mass of Na2Ca(CO3)2
def Na2Ca_CO3_2_mass : ℝ :=
  Na_count * Na_mass + Ca_count * Ca_mass + C_count * C_mass + O_count * O_mass

-- Theorem: The weight of 3.75 moles of Na2Ca(CO3)2 is 772.8 grams
theorem weight_of_Na2Ca_CO3_2 : moles * Na2Ca_CO3_2_mass = 772.8 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_Na2Ca_CO3_2_l666_66653


namespace NUMINAMATH_CALUDE_sum_equals_80790_l666_66610

theorem sum_equals_80790 : 30 + 80000 + 700 + 60 = 80790 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_80790_l666_66610


namespace NUMINAMATH_CALUDE_badge_exchange_l666_66699

theorem badge_exchange (x : ℕ) : 
  (x + 5 - (24 * (x + 5)) / 100 + (20 * x) / 100 = x - (20 * x) / 100 + (24 * (x + 5)) / 100 - 1) → 
  (x = 45 ∧ x + 5 = 50) :=
by sorry

end NUMINAMATH_CALUDE_badge_exchange_l666_66699


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l666_66665

theorem fruit_seller_apples : ∀ (original : ℕ), 
  (original : ℝ) * (1 - 0.3) = 420 → original = 600 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l666_66665


namespace NUMINAMATH_CALUDE_log_difference_equals_eight_l666_66690

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_difference_equals_eight :
  log 3 243 - log 3 (1/27) = 8 := by sorry

end NUMINAMATH_CALUDE_log_difference_equals_eight_l666_66690


namespace NUMINAMATH_CALUDE_water_level_rise_l666_66627

/-- Given a cube and a rectangular vessel, calculate the rise in water level when the cube is fully immersed. -/
theorem water_level_rise (cube_edge : ℝ) (vessel_length vessel_width : ℝ) (h_cube : cube_edge = 12) 
    (h_length : vessel_length = 20) (h_width : vessel_width = 15) : 
    (cube_edge ^ 3) / (vessel_length * vessel_width) = 5.76 := by
  sorry

end NUMINAMATH_CALUDE_water_level_rise_l666_66627


namespace NUMINAMATH_CALUDE_omelet_time_is_100_l666_66655

/-- Time to prepare and cook omelets -/
def total_omelet_time (
  pepper_chop_time : ℕ)
  (onion_chop_time : ℕ)
  (mushroom_slice_time : ℕ)
  (tomato_dice_time : ℕ)
  (cheese_grate_time : ℕ)
  (vegetable_saute_time : ℕ)
  (egg_cheese_cook_time : ℕ)
  (num_peppers : ℕ)
  (num_onions : ℕ)
  (num_mushrooms : ℕ)
  (num_tomatoes : ℕ)
  (num_omelets : ℕ) : ℕ :=
  let prep_time := 
    pepper_chop_time * num_peppers +
    onion_chop_time * num_onions +
    mushroom_slice_time * num_mushrooms +
    tomato_dice_time * num_tomatoes +
    cheese_grate_time * num_omelets
  let cook_time := vegetable_saute_time + egg_cheese_cook_time
  let omelets_during_prep := prep_time / cook_time
  let remaining_omelets := num_omelets - omelets_during_prep
  prep_time + remaining_omelets * cook_time

/-- Theorem: The total time to prepare and cook 10 omelets is 100 minutes -/
theorem omelet_time_is_100 :
  total_omelet_time 3 4 2 3 1 4 6 8 4 6 6 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_omelet_time_is_100_l666_66655


namespace NUMINAMATH_CALUDE_garden_perimeter_is_800_l666_66609

/-- The perimeter of a rectangular garden with given length and breadth -/
def garden_perimeter (length breadth : ℝ) : ℝ :=
  2 * (length + breadth)

/-- Theorem: The perimeter of a rectangular garden with length 300 m and breadth 100 m is 800 m -/
theorem garden_perimeter_is_800 :
  garden_perimeter 300 100 = 800 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_is_800_l666_66609


namespace NUMINAMATH_CALUDE_algebraic_expressions_l666_66604

theorem algebraic_expressions (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : (x - 2) * (y - 2) = -3) : 
  x * y = 3 ∧ x^2 + 4*x*y + y^2 = 31 ∧ x^2 + x*y + 5*y = 25 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expressions_l666_66604


namespace NUMINAMATH_CALUDE_exactly_100_valid_rules_l666_66684

/-- A type representing a set of 100 cards drawn from an infinite deck of real numbers. -/
def CardSet := Fin 100 → ℝ

/-- A rule for determining the winner between two sets of cards. -/
def WinningRule := CardSet → CardSet → Bool

/-- The condition that the winner only depends on the relative order of the 200 cards. -/
def RelativeOrderCondition (rule : WinningRule) : Prop :=
  ∀ (A B : CardSet) (f : ℝ → ℝ), StrictMono f →
    rule A B = rule (f ∘ A) (f ∘ B)

/-- The condition that if a_i > b_i for all i, then A beats B. -/
def DominanceCondition (rule : WinningRule) : Prop :=
  ∀ (A B : CardSet), (∀ i, A i > B i) → rule A B

/-- The transitivity condition: if A beats B and B beats C, then A beats C. -/
def TransitivityCondition (rule : WinningRule) : Prop :=
  ∀ (A B C : CardSet), rule A B → rule B C → rule A C

/-- A valid rule satisfies all three conditions. -/
def ValidRule (rule : WinningRule) : Prop :=
  RelativeOrderCondition rule ∧ DominanceCondition rule ∧ TransitivityCondition rule

/-- The main theorem: there are exactly 100 valid rules. -/
theorem exactly_100_valid_rules :
  ∃! (rules : Finset WinningRule), rules.card = 100 ∧ ∀ rule ∈ rules, ValidRule rule :=
sorry

end NUMINAMATH_CALUDE_exactly_100_valid_rules_l666_66684


namespace NUMINAMATH_CALUDE_cos_42_cos_18_minus_cos_48_sin_18_l666_66631

theorem cos_42_cos_18_minus_cos_48_sin_18 :
  Real.cos (42 * π / 180) * Real.cos (18 * π / 180) - 
  Real.cos (48 * π / 180) * Real.sin (18 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_42_cos_18_minus_cos_48_sin_18_l666_66631


namespace NUMINAMATH_CALUDE_mothers_age_l666_66668

theorem mothers_age (D M : ℕ) 
  (h1 : 2 * D + M = 70)
  (h2 : D + 2 * M = 95) : 
  M = 40 := by
sorry

end NUMINAMATH_CALUDE_mothers_age_l666_66668


namespace NUMINAMATH_CALUDE_rayden_has_more_birds_l666_66633

/-- The number of ducks Lily bought -/
def lily_ducks : ℕ := 20

/-- The number of geese Lily bought -/
def lily_geese : ℕ := 10

/-- The number of ducks Rayden bought -/
def rayden_ducks : ℕ := 3 * lily_ducks

/-- The number of geese Rayden bought -/
def rayden_geese : ℕ := 4 * lily_geese

/-- The difference in the total number of birds between Rayden and Lily -/
def bird_difference : ℕ := (rayden_ducks - lily_ducks) + (rayden_geese - lily_geese)

theorem rayden_has_more_birds :
  bird_difference = 70 := by sorry

end NUMINAMATH_CALUDE_rayden_has_more_birds_l666_66633


namespace NUMINAMATH_CALUDE_jade_transactions_l666_66615

theorem jade_transactions (mabel_transactions : ℕ) 
  (anthony_transactions : ℕ) (cal_transactions : ℕ) (jade_transactions : ℕ) : 
  mabel_transactions = 90 →
  anthony_transactions = mabel_transactions + (mabel_transactions * 10 / 100) →
  cal_transactions = anthony_transactions * 2 / 3 →
  jade_transactions = cal_transactions + 16 →
  jade_transactions = 82 := by
sorry

end NUMINAMATH_CALUDE_jade_transactions_l666_66615


namespace NUMINAMATH_CALUDE_complex_number_trigonometric_form_l666_66658

/-- Prove that the complex number z = sin 36° + i cos 54° is equal to √2 sin 36° (cos 45° + i sin 45°) -/
theorem complex_number_trigonometric_form 
  (z : ℂ) 
  (h1 : z = Complex.ofReal (Real.sin (36 * π / 180)) + Complex.I * Complex.ofReal (Real.cos (54 * π / 180)))
  (h2 : Real.cos (54 * π / 180) = Real.sin (36 * π / 180)) :
  z = Complex.ofReal (Real.sqrt 2 * Real.sin (36 * π / 180)) * 
      (Complex.ofReal (Real.cos (45 * π / 180)) + Complex.I * Complex.ofReal (Real.sin (45 * π / 180))) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_trigonometric_form_l666_66658


namespace NUMINAMATH_CALUDE_smallest_z_l666_66659

theorem smallest_z (x y z : ℤ) : 
  x < y → y < z → 
  2 * y = x + z →  -- arithmetic progression
  z * z = x * y →  -- geometric progression
  z ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_z_l666_66659


namespace NUMINAMATH_CALUDE_triangle_inequality_l666_66643

/-- Given a non-isosceles triangle with sides a, b, c and area S,
    prove the inequality relating the sides and the area. -/
theorem triangle_inequality (a b c S : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- sides are positive
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- non-isosceles condition
  S > 0 →  -- area is positive
  S = Real.sqrt (((a + b + c) / 2) * (((a + b + c) / 2) - a) * 
    (((a + b + c) / 2) - b) * (((a + b + c) / 2) - c)) →  -- Heron's formula
  (a^3 / ((a-b)*(a-c))) + (b^3 / ((b-c)*(b-a))) + 
    (c^3 / ((c-a)*(c-b))) > 2 * 3^(3/4) * S^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l666_66643


namespace NUMINAMATH_CALUDE_problem_solution_l666_66689

theorem problem_solution (x y z : ℝ) 
  (h1 : y / (x - y) = x / (y + z))
  (h2 : z^2 = x*(y + z) - y*(x - y)) :
  (y^2 + z^2 - x^2) / (2*y*z) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l666_66689


namespace NUMINAMATH_CALUDE_equation_solutions_l666_66679

theorem equation_solutions :
  (∃ x : ℚ, 3 * x - 5 = 10) ∧
  (∃ x : ℚ, 2 * x + 4 * (2 * x - 3) = 6 - 2 * (x + 1)) :=
by
  constructor
  · use 5
    norm_num
  · use 4/3
    norm_num
    
#check equation_solutions

end NUMINAMATH_CALUDE_equation_solutions_l666_66679


namespace NUMINAMATH_CALUDE_s_range_l666_66601

-- Define the piecewise function
noncomputable def s (t : ℝ) : ℝ :=
  if t ≥ 1 then 3 * t else 4 * t - t^2

-- State the theorem
theorem s_range :
  Set.range s = Set.Icc (-5 : ℝ) 9 := by sorry

end NUMINAMATH_CALUDE_s_range_l666_66601


namespace NUMINAMATH_CALUDE_complex_number_properties_l666_66605

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 8*m + 15) (m^2 - 5*m)

theorem complex_number_properties :
  (∃ m : ℝ, z m = Complex.I * (z m).im) ∧
  (∃ m : ℝ, z m = 3 + 6*Complex.I) ∧
  (∃ m : ℝ, 0 < m ∧ m < 3 ∧ (z m).re > 0 ∧ (z m).im < 0) :=
by sorry


end NUMINAMATH_CALUDE_complex_number_properties_l666_66605


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l666_66651

/-- Given a geometric sequence {a_n} with common ratio q = 2 and S_n being the sum of the first n terms, 
    prove that S_4 / a_2 = -15/2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- Common ratio q = 2
  (∀ n, S n = (a 1) * (1 - 2^n) / (1 - 2)) →  -- Sum formula for geometric sequence
  S 4 / a 2 = -15/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l666_66651


namespace NUMINAMATH_CALUDE_cyclists_meeting_time_l666_66678

/-- Two cyclists on a circular track problem -/
theorem cyclists_meeting_time 
  (track_circumference : ℝ) 
  (speed1 speed2 : ℝ) 
  (h1 : track_circumference = 600) 
  (h2 : speed1 = 7) 
  (h3 : speed2 = 8) : 
  track_circumference / (speed1 + speed2) = 40 := by
  sorry

#check cyclists_meeting_time

end NUMINAMATH_CALUDE_cyclists_meeting_time_l666_66678


namespace NUMINAMATH_CALUDE_power_of_i_third_quadrant_l666_66656

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Statement 1: i^2023 = -i
theorem power_of_i : i^2023 = -i := by sorry

-- Statement 2: -2-i is in the third quadrant
theorem third_quadrant : 
  let z : ℂ := -2 - i
  z.re < 0 ∧ z.im < 0 := by sorry

end NUMINAMATH_CALUDE_power_of_i_third_quadrant_l666_66656


namespace NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l666_66683

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 5 ∨ d = 9

def contains_5_and_9 (n : ℕ) : Prop :=
  5 ∈ n.digits 10 ∧ 9 ∈ n.digits 10

theorem smallest_valid_number_last_four_digits :
  ∃ n : ℕ,
    (n > 0) ∧
    (n % 5 = 0) ∧
    (n % 9 = 0) ∧
    is_valid_number n ∧
    contains_5_and_9 n ∧
    (∀ m : ℕ, m > 0 ∧ m % 5 = 0 ∧ m % 9 = 0 ∧ is_valid_number m ∧ contains_5_and_9 m → n ≤ m) ∧
    (n % 10000 = 9995) :=
  sorry

end NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l666_66683


namespace NUMINAMATH_CALUDE_lcm_of_16_24_45_l666_66611

theorem lcm_of_16_24_45 : Nat.lcm (Nat.lcm 16 24) 45 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_16_24_45_l666_66611


namespace NUMINAMATH_CALUDE_prob_at_least_one_qualified_prob_merchant_rejects_l666_66662

-- Define the probability of a product being qualified
def p_qualified : ℝ := 0.8

-- Define the number of products inspected by the company
def n_company_inspect : ℕ := 4

-- Define the total number of products sent to the merchant
def n_total : ℕ := 20

-- Define the number of unqualified products
def n_unqualified : ℕ := 3

-- Define the number of products inspected by the merchant
def n_merchant_inspect : ℕ := 2

-- Theorem for part I
theorem prob_at_least_one_qualified :
  1 - (1 - p_qualified) ^ n_company_inspect = 0.9984 := by sorry

-- Theorem for part II
theorem prob_merchant_rejects :
  (Nat.choose (n_total - n_unqualified) 1 * Nat.choose n_unqualified 1 +
   Nat.choose n_unqualified 2) / Nat.choose n_total 2 = 27 / 95 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_qualified_prob_merchant_rejects_l666_66662


namespace NUMINAMATH_CALUDE_pyramid_equal_volume_division_l666_66616

theorem pyramid_equal_volume_division (m : ℝ) (hm : m > 0) :
  ∃ (x y z : ℝ),
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    x + y + z = m ∧
    x^3 = (1/3) * m^3 ∧
    (x + y)^3 = (2/3) * m^3 ∧
    x = m / Real.rpow 3 (1/3) ∧
    y = (m / Real.rpow 3 (1/3)) * (Real.rpow 2 (1/3) - 1) ∧
    z = m * (1 - Real.rpow (2/3) (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_equal_volume_division_l666_66616


namespace NUMINAMATH_CALUDE_divide_fraction_by_integer_l666_66625

theorem divide_fraction_by_integer : (3 : ℚ) / 7 / 4 = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_divide_fraction_by_integer_l666_66625


namespace NUMINAMATH_CALUDE_function_behavior_l666_66694

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 10

-- Define the interval
def interval : Set ℝ := Set.Ioo 2 4

-- Theorem statement
theorem function_behavior (x y : ℝ) (hx : x ∈ interval) (hy : y ∈ interval) :
  (x < 3 ∧ y < 3 → f x > f y) ∧
  (x > 3 ∧ y > 3 → f x < f y) ∧
  (x < 3 ∧ y > 3 → f x > f y) :=
sorry

end NUMINAMATH_CALUDE_function_behavior_l666_66694


namespace NUMINAMATH_CALUDE_eighth_roll_last_probability_l666_66666

/-- A standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The probability of rolling a different number than the previous roll -/
def probDifferent : ℚ := 5/6

/-- The probability of rolling the same number as the previous roll -/
def probSame : ℚ := 1/6

/-- The number of rolls we're interested in -/
def numRolls : ℕ := 8

/-- The probability that the 8th roll is the last roll -/
def probEighthRollLast : ℚ := probDifferent^(numRolls - 2) * probSame

theorem eighth_roll_last_probability :
  probEighthRollLast = 15625/279936 := by sorry

end NUMINAMATH_CALUDE_eighth_roll_last_probability_l666_66666


namespace NUMINAMATH_CALUDE_smallest_divisible_by_72_l666_66622

theorem smallest_divisible_by_72 (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(72 ∣ m * 40)) ∧ 
  (72 ∣ n * 40) ∧ 
  (n ≥ 5) ∧
  (∃ k : ℕ, n * 40 = 72 * k) →
  n = 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_72_l666_66622


namespace NUMINAMATH_CALUDE_boat_stream_speed_l666_66673

/-- Proves that given a boat with a speed of 36 kmph in still water,
    if it can cover 80 km downstream or 40 km upstream in the same time,
    then the speed of the stream is 12 kmph. -/
theorem boat_stream_speed 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (h1 : boat_speed = 36)
  (h2 : downstream_distance = 80)
  (h3 : upstream_distance = 40)
  (h4 : downstream_distance / (boat_speed + stream_speed) = upstream_distance / (boat_speed - stream_speed)) :
  stream_speed = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_boat_stream_speed_l666_66673


namespace NUMINAMATH_CALUDE_oak_trees_after_planting_l666_66620

/-- The number of oak trees in the park after planting -/
def trees_after_planting (initial_trees new_trees : ℕ) : ℕ :=
  initial_trees + new_trees

/-- Theorem: There will be 11 oak trees after planting -/
theorem oak_trees_after_planting :
  trees_after_planting 9 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_after_planting_l666_66620


namespace NUMINAMATH_CALUDE_cape_may_shark_count_l666_66691

/-- The number of sharks in Daytona Beach -/
def daytona_sharks : ℕ := 12

/-- The number of sharks in Cape May -/
def cape_may_sharks : ℕ := 2 * daytona_sharks + 8

theorem cape_may_shark_count : cape_may_sharks = 32 := by
  sorry

end NUMINAMATH_CALUDE_cape_may_shark_count_l666_66691


namespace NUMINAMATH_CALUDE_exists_valid_assignment_with_difference_one_l666_66664

/-- Represents a position on an infinite checkerboard -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- Represents the color of a square on the checkerboard -/
inductive Color
  | White
  | Black

/-- Determines the color of a square based on its position -/
def color (p : Position) : Color :=
  if (p.x + p.y) % 2 = 0 then Color.White else Color.Black

/-- Represents an assignment of non-zero integers to white squares -/
def Assignment := Position → ℤ

/-- Checks if an assignment is valid (all non-zero integers on white squares) -/
def is_valid_assignment (f : Assignment) : Prop :=
  ∀ p, color p = Color.White → f p ≠ 0

/-- Calculates the product difference for a black square -/
def product_difference (f : Assignment) (p : Position) : ℤ :=
  f {x := p.x - 1, y := p.y} * f {x := p.x + 1, y := p.y} -
  f {x := p.x, y := p.y - 1} * f {x := p.x, y := p.y + 1}

/-- The main theorem: there exists a valid assignment satisfying the condition -/
theorem exists_valid_assignment_with_difference_one :
  ∃ f : Assignment, is_valid_assignment f ∧
    ∀ p, color p = Color.Black → product_difference f p = 1 :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_assignment_with_difference_one_l666_66664


namespace NUMINAMATH_CALUDE_local_max_implies_c_eq_6_l666_66646

/-- The function f(x) = x(x-c)^2 has a local maximum at x=2 -/
def has_local_max_at_2 (c : ℝ) : Prop :=
  let f := fun x => x * (x - c)^2
  (∃ δ > 0, ∀ x, |x - 2| < δ → f x ≤ f 2) ∧
  (∀ ε > 0, ∃ x, |x - 2| < ε ∧ f x < f 2)

/-- If f(x) = x(x-c)^2 has a local maximum at x=2, then c = 6 -/
theorem local_max_implies_c_eq_6 :
  ∀ c : ℝ, has_local_max_at_2 c → c = 6 := by
  sorry

end NUMINAMATH_CALUDE_local_max_implies_c_eq_6_l666_66646


namespace NUMINAMATH_CALUDE_soccer_team_goals_l666_66617

theorem soccer_team_goals (total_players : ℕ) (games_played : ℕ) (goals_other_players : ℕ) : 
  total_players = 24 →
  games_played = 15 →
  goals_other_players = 30 →
  (total_players / 3 * games_played + goals_other_players : ℕ) = 150 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_goals_l666_66617


namespace NUMINAMATH_CALUDE_triangle_inequality_l666_66676

/-- For any triangle with sides a, b, c, angle A opposite side a, and semiperimeter p,
    the inequality (bc cos A) / (b + c) + a < p < (bc + a^2) / a holds. -/
theorem triangle_inequality (a b c : ℝ) (A : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_angle : 0 < A ∧ A < π) :
  let p := (a + b + c) / 2
  (b * c * Real.cos A) / (b + c) + a < p ∧ p < (b * c + a^2) / a :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l666_66676


namespace NUMINAMATH_CALUDE_pennies_count_l666_66657

def pennies_in_jar (nickels dimes quarters : ℕ) (ice_cream_cost leftover : ℕ) : ℕ :=
  let nickel_value := 5
  let dime_value := 10
  let quarter_value := 25
  let total_without_pennies := nickels * nickel_value + dimes * dime_value + quarters * quarter_value
  let total_in_jar := ice_cream_cost + leftover
  total_in_jar - total_without_pennies

theorem pennies_count (nickels dimes quarters : ℕ) (ice_cream_cost leftover : ℕ) :
  nickels = 85 → dimes = 35 → quarters = 26 → ice_cream_cost = 1500 → leftover = 48 →
  pennies_in_jar nickels dimes quarters ice_cream_cost leftover = 123 := by
  sorry

#eval pennies_in_jar 85 35 26 1500 48

end NUMINAMATH_CALUDE_pennies_count_l666_66657


namespace NUMINAMATH_CALUDE_calculation_proof_l666_66641

theorem calculation_proof : (1955 - 1875)^2 / 64 = 100 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l666_66641


namespace NUMINAMATH_CALUDE_smallest_two_digit_number_divisible_by_170_l666_66680

def sum_of_digits (n : ℕ) : ℕ := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem smallest_two_digit_number_divisible_by_170 :
  ∃ (N : ℕ), is_two_digit N ∧
  (sum_of_digits (10^N - N) % 170 = 0) ∧
  (∀ (M : ℕ), is_two_digit M → sum_of_digits (10^M - M) % 170 = 0 → N ≤ M) ∧
  N = 20 := by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_number_divisible_by_170_l666_66680


namespace NUMINAMATH_CALUDE_legs_multiple_of_heads_l666_66697

/-- Represents the number of legs for each animal type -/
def legs_per_animal : Fin 3 → ℕ
| 0 => 2  -- Ducks
| 1 => 4  -- Cows
| 2 => 4  -- Buffaloes

/-- Represents the number of animals of each type -/
structure AnimalCounts where
  ducks : ℕ
  cows : ℕ
  buffaloes : ℕ
  buffalo_count_eq : buffaloes = 6

/-- Calculates the total number of legs -/
def total_legs (counts : AnimalCounts) : ℕ :=
  counts.ducks * legs_per_animal 0 +
  counts.cows * legs_per_animal 1 +
  counts.buffaloes * legs_per_animal 2

/-- Calculates the total number of heads -/
def total_heads (counts : AnimalCounts) : ℕ :=
  counts.ducks + counts.cows + counts.buffaloes

/-- The theorem to be proved -/
theorem legs_multiple_of_heads (counts : AnimalCounts) :
  ∃ m : ℕ, m ≥ 2 ∧ total_legs counts = m * total_heads counts + 12 ∧
  ∀ k : ℕ, k < m → ¬(total_legs counts = k * total_heads counts + 12) :=
sorry

end NUMINAMATH_CALUDE_legs_multiple_of_heads_l666_66697


namespace NUMINAMATH_CALUDE_assembly_line_arrangements_l666_66637

def num_tasks : ℕ := 5

theorem assembly_line_arrangements :
  (Finset.range num_tasks).card.factorial = 120 := by
  sorry

end NUMINAMATH_CALUDE_assembly_line_arrangements_l666_66637


namespace NUMINAMATH_CALUDE_tent_production_equation_l666_66603

theorem tent_production_equation (x : ℝ) (h : x > 0) : 
  (7000 / x) - (7000 / (1.4 * x)) = 4 ↔ 
  ∃ (original_days actual_days : ℝ),
    original_days > 0 ∧ 
    actual_days > 0 ∧
    original_days = 7000 / x ∧ 
    actual_days = 7000 / (1.4 * x) ∧
    original_days - actual_days = 4 :=
by sorry

end NUMINAMATH_CALUDE_tent_production_equation_l666_66603


namespace NUMINAMATH_CALUDE_product_of_differences_l666_66649

theorem product_of_differences (m n : ℝ) 
  (hm : m = 1 / (Real.sqrt 3 + Real.sqrt 2)) 
  (hn : n = 1 / (Real.sqrt 3 - Real.sqrt 2)) : 
  (m - 1) * (n - 1) = -2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_differences_l666_66649


namespace NUMINAMATH_CALUDE_marble_remainder_l666_66629

theorem marble_remainder (r p : ℤ) 
  (hr : r % 5 = 2) 
  (hp : p % 5 = 4) : 
  (r + p) % 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_marble_remainder_l666_66629


namespace NUMINAMATH_CALUDE_exists_min_value_subject_to_constraint_l666_66693

/-- The constraint function for a, b, c, d -/
def constraint (a b c d : ℝ) : Prop :=
  a^4 + b^4 + c^4 + d^4 = 16

/-- The function to be minimized -/
def objective (a b c d : ℝ) : ℝ :=
  a^3 + b^3 + c^3 + d^3

/-- Theorem stating the existence of a minimum value for the objective function
    subject to the given constraint -/
theorem exists_min_value_subject_to_constraint :
  ∃ (min : ℝ), ∀ (a b c d : ℝ), constraint a b c d →
    objective a b c d ≥ min ∧
    (∃ (a' b' c' d' : ℝ), constraint a' b' c' d' ∧ objective a' b' c' d' = min) :=
by sorry

end NUMINAMATH_CALUDE_exists_min_value_subject_to_constraint_l666_66693


namespace NUMINAMATH_CALUDE_shaded_percentage_is_59_l666_66661

def large_square_side_length : ℕ := 5
def small_square_side_length : ℕ := 1
def border_squares_count : ℕ := 16
def shaded_border_squares_count : ℕ := 8
def central_region_shaded_fraction : ℚ := 3 / 4

theorem shaded_percentage_is_59 :
  let total_area : ℚ := (large_square_side_length ^ 2 : ℚ)
  let border_area : ℚ := (border_squares_count * small_square_side_length ^ 2 : ℚ)
  let central_area : ℚ := total_area - border_area
  let shaded_border_area : ℚ := (shaded_border_squares_count * small_square_side_length ^ 2 : ℚ)
  let shaded_central_area : ℚ := central_region_shaded_fraction * central_area
  let total_shaded_area : ℚ := shaded_border_area + shaded_central_area
  (total_shaded_area / total_area) * 100 = 59 :=
by sorry

end NUMINAMATH_CALUDE_shaded_percentage_is_59_l666_66661


namespace NUMINAMATH_CALUDE_mixed_number_properties_l666_66623

theorem mixed_number_properties :
  let x : ℚ := -1 - 2/7
  (1 / x = -7/9) ∧
  (-x = 1 + 2/7) ∧
  (|x| = 1 + 2/7) :=
by sorry

end NUMINAMATH_CALUDE_mixed_number_properties_l666_66623


namespace NUMINAMATH_CALUDE_meaningful_expression_l666_66636

/-- The expression (x+3)/(x-1) + (x-2)^0 is meaningful if and only if x ≠ 1 -/
theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (x + 3) / (x - 1) + (x - 2)^0) ↔ x ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_meaningful_expression_l666_66636


namespace NUMINAMATH_CALUDE_problem_solution_l666_66642

-- Define x as the solution to the equation x = 1 + √3 / x
noncomputable def x : ℝ := Real.sqrt 3 + 1

-- State the theorem
theorem problem_solution :
  1 / ((x + 1) * (x - 3)) = (-Real.sqrt 3 - 4) / 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l666_66642


namespace NUMINAMATH_CALUDE_estevan_blankets_l666_66692

theorem estevan_blankets (initial_blankets : ℕ) : 
  (initial_blankets / 3 : ℚ) + 2 = 10 → initial_blankets = 24 := by
  sorry

end NUMINAMATH_CALUDE_estevan_blankets_l666_66692


namespace NUMINAMATH_CALUDE_rain_probability_l666_66624

/-- Given probabilities of rain events in counties, prove the probability of rain on both days -/
theorem rain_probability (p_monday p_tuesday p_no_rain : ℝ) 
  (h1 : p_monday = 0.6)
  (h2 : p_tuesday = 0.55)
  (h3 : p_no_rain = 0.25) :
  p_monday + p_tuesday - (1 - p_no_rain) = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_rain_probability_l666_66624


namespace NUMINAMATH_CALUDE_card_game_combinations_l666_66607

/-- The number of cards in the deck -/
def deck_size : ℕ := 60

/-- The number of cards in a hand -/
def hand_size : ℕ := 12

/-- The number of distinct unordered hands -/
def num_hands : ℕ := 75287520

theorem card_game_combinations :
  Nat.choose deck_size hand_size = num_hands := by
  sorry

end NUMINAMATH_CALUDE_card_game_combinations_l666_66607


namespace NUMINAMATH_CALUDE_butterfly_cocoon_time_l666_66634

theorem butterfly_cocoon_time :
  ∀ (cocoon_time larva_time : ℕ),
    cocoon_time + larva_time = 120 →
    larva_time = 3 * cocoon_time →
    cocoon_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_cocoon_time_l666_66634


namespace NUMINAMATH_CALUDE_simplify_fraction_l666_66626

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l666_66626


namespace NUMINAMATH_CALUDE_elvin_internet_charge_l666_66687

/-- Represents Elvin's monthly telephone bill structure -/
structure MonthlyBill where
  callCharge : ℝ
  internetCharge : ℝ

/-- Calculates the total bill amount -/
def totalBill (bill : MonthlyBill) : ℝ :=
  bill.callCharge + bill.internetCharge

theorem elvin_internet_charge :
  ∀ (jan_bill feb_bill : MonthlyBill),
    totalBill jan_bill = 46 →
    totalBill feb_bill = 76 →
    feb_bill.callCharge = 2 * jan_bill.callCharge →
    jan_bill.internetCharge = feb_bill.internetCharge →
    jan_bill.internetCharge = 16 := by
  sorry

end NUMINAMATH_CALUDE_elvin_internet_charge_l666_66687


namespace NUMINAMATH_CALUDE_triangle_side_length_l666_66667

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 5 → b = 3 → C = 2 * π / 3 → c = 7 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l666_66667


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l666_66682

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  prop1 : a 5 * a 7 = 2
  prop2 : a 2 + a 10 = 3

/-- The main theorem -/
theorem geometric_sequence_ratio (seq : GeometricSequence) :
  seq.a 12 / seq.a 4 = 2 ∨ seq.a 12 / seq.a 4 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l666_66682


namespace NUMINAMATH_CALUDE_choir_members_count_l666_66686

theorem choir_members_count : ∃ n₁ n₂ : ℕ, 
  150 < n₁ ∧ n₁ < 250 ∧
  150 < n₂ ∧ n₂ < 250 ∧
  n₁ % 3 = 1 ∧
  n₁ % 6 = 2 ∧
  n₁ % 8 = 3 ∧
  n₂ % 3 = 1 ∧
  n₂ % 6 = 2 ∧
  n₂ % 8 = 3 ∧
  n₁ = 195 ∧
  n₂ = 219 ∧
  ∀ n : ℕ, (150 < n ∧ n < 250 ∧ n % 3 = 1 ∧ n % 6 = 2 ∧ n % 8 = 3) → (n = 195 ∨ n = 219) :=
by sorry

end NUMINAMATH_CALUDE_choir_members_count_l666_66686


namespace NUMINAMATH_CALUDE_class_size_is_20_l666_66644

/-- Represents the number of students in a class with specific age distributions. -/
def num_students : ℕ := by sorry

/-- The average age of all students in the class. -/
def average_age : ℝ := 20

/-- The average age of a group of 9 students. -/
def average_age_group1 : ℝ := 11

/-- The average age of a group of 10 students. -/
def average_age_group2 : ℝ := 24

/-- The age of the 20th student. -/
def age_20th_student : ℝ := 61

/-- Theorem stating that the number of students in the class is 20. -/
theorem class_size_is_20 : num_students = 20 := by sorry

end NUMINAMATH_CALUDE_class_size_is_20_l666_66644


namespace NUMINAMATH_CALUDE_max_value_x_1_minus_2x_l666_66606

theorem max_value_x_1_minus_2x : 
  ∃ (max : ℝ), max = 1/8 ∧ 
  (∀ x : ℝ, 0 < x → x < 1/2 → x * (1 - 2*x) ≤ max) ∧
  (∃ x : ℝ, 0 < x ∧ x < 1/2 ∧ x * (1 - 2*x) = max) := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_1_minus_2x_l666_66606


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l666_66619

/-- Definition of the repeating decimal 0.4444... -/
def repeating_4 : ℚ := 4 / 9

/-- Definition of the repeating decimal 0.3535... -/
def repeating_35 : ℚ := 35 / 99

/-- The sum of the repeating decimals 0.4444... and 0.3535... is equal to 79/99 -/
theorem sum_of_repeating_decimals : repeating_4 + repeating_35 = 79 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l666_66619


namespace NUMINAMATH_CALUDE_simplify_fraction_l666_66696

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (10 * x * y^2) / (5 * x * y) = 2 * y :=
sorry

end NUMINAMATH_CALUDE_simplify_fraction_l666_66696


namespace NUMINAMATH_CALUDE_j_walking_speed_l666_66675

/-- Represents the walking speed of J in kmph -/
def j_speed : ℝ := 5.945

/-- Represents the cycling speed of P in kmph -/
def p_speed : ℝ := 8

/-- Represents the time (in hours) between J's start and P's start -/
def time_difference : ℝ := 1.5

/-- Represents the total time (in hours) from J's start to when P catches up -/
def total_time : ℝ := 7.3

/-- Represents the time (in hours) P cycles before catching up to J -/
def p_cycle_time : ℝ := 5.8

/-- Represents the distance (in km) J is behind P when P catches up -/
def distance_behind : ℝ := 3

theorem j_walking_speed :
  p_speed * p_cycle_time = j_speed * total_time + distance_behind :=
sorry

end NUMINAMATH_CALUDE_j_walking_speed_l666_66675


namespace NUMINAMATH_CALUDE_median_of_special_list_l666_66654

/-- The sum of integers from 1 to n -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total count of numbers in the list -/
def total_count : ℕ := triangular_number 250

/-- The position of the median in the list -/
def median_position : ℕ := total_count / 2 + 1

/-- The number that appears at the median position -/
def median_number : ℕ := 177

theorem median_of_special_list :
  median_number = 177 ∧
  triangular_number (median_number - 1) < median_position ∧
  median_position ≤ triangular_number median_number :=
sorry

end NUMINAMATH_CALUDE_median_of_special_list_l666_66654


namespace NUMINAMATH_CALUDE_triangle_property_l666_66602

open Real

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a * sin B - Real.sqrt 3 * b * cos B * cos C = Real.sqrt 3 * c * (cos B)^2 →
  (B = π / 3 ∧
   (0 < C ∧ C < π / 2 → 1 < a^2 + b^2 ∧ a^2 + b^2 < 7)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l666_66602


namespace NUMINAMATH_CALUDE_intersection_A_B_l666_66632

-- Define the set of positive integers
def PositiveInt : Set ℕ := {n : ℕ | n > 0}

-- Define set A
def A : Set ℕ := {x ∈ PositiveInt | x ≤ Real.exp 1}

-- Define set B
def B : Set ℕ := {0, 1, 2, 3}

-- Theorem to prove
theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l666_66632


namespace NUMINAMATH_CALUDE_right_triangle_area_l666_66640

/-- Given a right triangle with one leg of length a and the ratio of its circumradius
    to inradius being 5:2, its area is either 2a²/3 or 3a²/8 -/
theorem right_triangle_area (a : ℝ) (h : a > 0) :
  ∃ (R r : ℝ), R > 0 ∧ r > 0 ∧ R / r = 5 / 2 ∧
  (∃ (b c : ℝ), b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧
   (1/2 * a * b = 2*a^2/3 ∨ 1/2 * a * b = 3*a^2/8)) :=
by sorry


end NUMINAMATH_CALUDE_right_triangle_area_l666_66640


namespace NUMINAMATH_CALUDE_blocks_needed_for_wall_l666_66618

/-- Represents the dimensions of a wall -/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  length : Set ℚ
  height : ℕ

/-- Calculates the number of blocks needed for a wall with given conditions -/
def calculateBlocksNeeded (wall : WallDimensions) (block : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating that 540 blocks are needed for the given wall -/
theorem blocks_needed_for_wall :
  let wall := WallDimensions.mk 120 9
  let block := BlockDimensions.mk {2, 1.5, 1} 1
  calculateBlocksNeeded wall block = 540 := by
    sorry

end NUMINAMATH_CALUDE_blocks_needed_for_wall_l666_66618


namespace NUMINAMATH_CALUDE_container_volume_ratio_l666_66612

theorem container_volume_ratio : 
  ∀ (C D : ℚ), C > 0 → D > 0 → 
  (3 / 4 : ℚ) * C = (5 / 8 : ℚ) * D → 
  C / D = (5 / 6 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l666_66612


namespace NUMINAMATH_CALUDE_runner_speed_l666_66635

/-- Proves that a runner covering 11.4 km in 2 minutes has a speed of 95 m/s -/
theorem runner_speed : ∀ (distance : ℝ) (time : ℝ),
  distance = 11.4 ∧ time = 2 →
  (distance * 1000) / (time * 60) = 95 := by
  sorry

end NUMINAMATH_CALUDE_runner_speed_l666_66635


namespace NUMINAMATH_CALUDE_expression_simplification_l666_66695

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (1 / (x + 1) + 1 / (x^2 - 1)) / (x / (x - 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l666_66695


namespace NUMINAMATH_CALUDE_multiple_properties_l666_66698

-- Define x and y as integers
variable (x y : ℤ)

-- Define the conditions
def x_multiple_of_4 : Prop := ∃ k : ℤ, x = 4 * k
def y_multiple_of_9 : Prop := ∃ m : ℤ, y = 9 * m

-- Theorem to prove
theorem multiple_properties
  (hx : x_multiple_of_4 x)
  (hy : y_multiple_of_9 y) :
  (∃ n : ℤ, y = 3 * n) ∧ (∃ p : ℤ, x - y = 4 * p) :=
by sorry

end NUMINAMATH_CALUDE_multiple_properties_l666_66698


namespace NUMINAMATH_CALUDE_direct_inverse_variation_l666_66677

theorem direct_inverse_variation (k : ℝ) (P₀ Q₀ R₀ P₁ R₁ : ℝ) :
  P₀ = k * Q₀ / Real.sqrt R₀ →
  P₀ = 9/4 →
  R₀ = 16/25 →
  Q₀ = 5/8 →
  P₁ = 27 →
  R₁ = 1/36 →
  k * (5/8) / Real.sqrt (16/25) = 9/4 →
  ∃ Q₁ : ℝ, P₁ = k * Q₁ / Real.sqrt R₁ ∧ Q₁ = 1.56 := by
sorry

end NUMINAMATH_CALUDE_direct_inverse_variation_l666_66677


namespace NUMINAMATH_CALUDE_count_special_divisors_l666_66638

/-- The number of positive integer divisors of 998^49999 that are not divisors of 998^49998 -/
def special_divisors : ℕ := 99999

/-- 998 as a product of its prime factors -/
def factor_998 : ℕ × ℕ := (2, 499)

theorem count_special_divisors :
  (factor_998.1 * factor_998.2)^49999 = 998^49999 →
  (∃ (d : ℕ → ℕ × ℕ),
    (∀ (n : ℕ), n < special_divisors →
      (factor_998.1^(d n).1 * factor_998.2^(d n).2 ∣ 998^49999) ∧
      ¬(factor_998.1^(d n).1 * factor_998.2^(d n).2 ∣ 998^49998)) ∧
    (∀ (n m : ℕ), n < special_divisors → m < special_divisors → n ≠ m →
      factor_998.1^(d n).1 * factor_998.2^(d n).2 ≠ factor_998.1^(d m).1 * factor_998.2^(d m).2) ∧
    (∀ (k : ℕ), (k ∣ 998^49999) ∧ ¬(k ∣ 998^49998) →
      ∃ (n : ℕ), n < special_divisors ∧ k = factor_998.1^(d n).1 * factor_998.2^(d n).2)) :=
by sorry

end NUMINAMATH_CALUDE_count_special_divisors_l666_66638


namespace NUMINAMATH_CALUDE_one_third_of_recipe_l666_66652

theorem one_third_of_recipe (full_recipe : ℚ) (one_third_recipe : ℚ) : 
  full_recipe = 17 / 3 ∧ one_third_recipe = full_recipe / 3 → one_third_recipe = 17 / 9 := by
  sorry

#check one_third_of_recipe

end NUMINAMATH_CALUDE_one_third_of_recipe_l666_66652


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l666_66650

theorem not_necessarily_right_triangle (a b c : ℝ) : 
  a^2 = 5 → b^2 = 12 → c^2 = 13 → 
  ¬ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l666_66650


namespace NUMINAMATH_CALUDE_min_value_implies_a_eq_6_l666_66648

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1/3 then 3 - Real.sin (a * x) else a * x + Real.log x / Real.log 3

-- State the theorem
theorem min_value_implies_a_eq_6 (a : ℝ) (h1 : a > 0) :
  (∀ x, f a x ≥ 1) ∧ (∃ x, f a x = 1) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_eq_6_l666_66648


namespace NUMINAMATH_CALUDE_probability_not_exceeding_60W_l666_66608

def total_bulbs : ℕ := 250
def bulbs_100W : ℕ := 100
def bulbs_60W : ℕ := 50
def bulbs_25W : ℕ := 50
def bulbs_15W : ℕ := 50

theorem probability_not_exceeding_60W :
  let p := (bulbs_60W + bulbs_25W + bulbs_15W) / total_bulbs
  p = 3/5 := by sorry

end NUMINAMATH_CALUDE_probability_not_exceeding_60W_l666_66608


namespace NUMINAMATH_CALUDE_smallest_checkered_rectangle_l666_66685

/-- A rectangle that can be divided into both 1 × 13 rectangles and three-cell corners -/
structure CheckeredRectangle where
  width : ℕ
  height : ℕ
  dividable_13 : width * height % 13 = 0
  dividable_3 : width ≥ 2 ∧ height ≥ 2

/-- The area of a CheckeredRectangle -/
def area (r : CheckeredRectangle) : ℕ := r.width * r.height

/-- The perimeter of a CheckeredRectangle -/
def perimeter (r : CheckeredRectangle) : ℕ := 2 * (r.width + r.height)

/-- The set of all valid CheckeredRectangles -/
def valid_rectangles : Set CheckeredRectangle :=
  {r : CheckeredRectangle | true}

theorem smallest_checkered_rectangle :
  ∃ (r : CheckeredRectangle),
    r ∈ valid_rectangles ∧
    area r = 78 ∧
    (∀ (s : CheckeredRectangle), s ∈ valid_rectangles → area s ≥ area r) ∧
    (∃ (p : List ℕ), p = [38, 58, 82] ∧ (perimeter r) ∈ p) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_checkered_rectangle_l666_66685


namespace NUMINAMATH_CALUDE_jade_rate_ratio_l666_66639

/-- The "jade rate" for a shape is the constant k in the volume formula V = kD³,
    where D is the characteristic length of the shape. -/
def jade_rate (volume : Real → Real) : Real :=
  volume 1

theorem jade_rate_ratio :
  let sphere_volume (a : Real) := (4 / 3) * Real.pi * (a / 2)^3
  let cylinder_volume (a : Real) := Real.pi * (a / 2)^2 * a
  let cube_volume (a : Real) := a^3
  let k₁ := jade_rate sphere_volume
  let k₂ := jade_rate cylinder_volume
  let k₃ := jade_rate cube_volume
  k₁ / k₂ = (Real.pi / 6) / (Real.pi / 4) ∧ k₂ / k₃ = Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_jade_rate_ratio_l666_66639


namespace NUMINAMATH_CALUDE_farm_tax_problem_l666_66671

/-- Represents the farm tax problem -/
theorem farm_tax_problem (total_tax : ℝ) (william_tax : ℝ) (taxable_percentage : ℝ) 
  (h1 : total_tax = 5000)
  (h2 : william_tax = 480)
  (h3 : taxable_percentage = 0.60) :
  william_tax / total_tax * 100 = 5.76 := by
  sorry

end NUMINAMATH_CALUDE_farm_tax_problem_l666_66671


namespace NUMINAMATH_CALUDE_det_B_squared_minus_3B_l666_66670

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : Matrix.det ((B ^ 2) - 3 • B) = 88 := by
  sorry

end NUMINAMATH_CALUDE_det_B_squared_minus_3B_l666_66670


namespace NUMINAMATH_CALUDE_xiaolong_exam_score_l666_66621

theorem xiaolong_exam_score (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℤ) 
  (xiaolong_score : ℕ) (max_answered : ℕ) :
  total_questions = 50 →
  correct_points = 3 →
  incorrect_points = -1 →
  xiaolong_score = 120 →
  max_answered = 48 →
  ∃ (correct incorrect : ℕ),
    correct + incorrect ≤ max_answered ∧
    correct * correct_points + incorrect * incorrect_points = xiaolong_score ∧
    correct ≤ 42 ∧
    ∀ (c i : ℕ), 
      c + i ≤ max_answered →
      c * correct_points + i * incorrect_points = xiaolong_score →
      c ≤ 42 :=
by sorry

end NUMINAMATH_CALUDE_xiaolong_exam_score_l666_66621
