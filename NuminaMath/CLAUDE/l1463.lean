import Mathlib

namespace NUMINAMATH_CALUDE_pizza_combinations_l1463_146385

def number_of_toppings : ℕ := 8

theorem pizza_combinations : 
  (number_of_toppings) +                    -- one-topping pizzas
  (number_of_toppings.choose 2) +           -- two-topping pizzas
  (number_of_toppings.choose 3) = 92 :=     -- three-topping pizzas
by sorry

end NUMINAMATH_CALUDE_pizza_combinations_l1463_146385


namespace NUMINAMATH_CALUDE_club_selection_theorem_l1463_146371

/-- The number of ways to choose a president, vice-president, and secretary from a club -/
def club_selection_ways (total_members boys girls : ℕ) : ℕ :=
  let president_vp_ways := boys * girls + girls * boys
  let secretary_ways := boys * (boys - 1) + girls * (girls - 1)
  president_vp_ways * secretary_ways

/-- Theorem stating the number of ways to choose club positions under specific conditions -/
theorem club_selection_theorem :
  club_selection_ways 25 15 10 = 90000 :=
by sorry

end NUMINAMATH_CALUDE_club_selection_theorem_l1463_146371


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1463_146328

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Define the inequality
def inequality (x : ℝ) : Prop := log_half (2*x + 1) ≥ log_half 3

-- Define the solution set
def solution_set : Set ℝ := Set.Ioc (-1/2) 1

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1463_146328


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1463_146389

theorem quadratic_equal_roots (b c : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + c = 0 ∧ (∀ y : ℝ, y^2 + b*y + c = 0 → y = x)) → 
  b^2 - 2*(1+2*c) = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1463_146389


namespace NUMINAMATH_CALUDE_track_length_l1463_146344

/-- The length of a circular track given total distance and number of laps --/
theorem track_length (total_distance : ℝ) (completed_laps : ℕ) (remaining_laps : ℕ) :
  total_distance = 2400 ∧ completed_laps = 6 ∧ remaining_laps = 4 →
  total_distance / (completed_laps + remaining_laps : ℝ) = 240 := by
  sorry

end NUMINAMATH_CALUDE_track_length_l1463_146344


namespace NUMINAMATH_CALUDE_fq_length_l1463_146315

-- Define the triangle DEF
structure RightTriangle where
  DE : ℝ
  DF : ℝ
  rightAngleAtE : True

-- Define the circle
structure TangentCircle where
  centerOnDE : True
  tangentToDF : True
  tangentToEF : True

-- Define the theorem
theorem fq_length
  (triangle : RightTriangle)
  (circle : TangentCircle)
  (h1 : triangle.DF = Real.sqrt 85)
  (h2 : triangle.DE = 7)
  : ∃ Q : ℝ × ℝ, ∃ F : ℝ × ℝ, ‖F - Q‖ = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_fq_length_l1463_146315


namespace NUMINAMATH_CALUDE_sqrt_equation_root_l1463_146353

theorem sqrt_equation_root : 
  ∃ x : ℝ, x = 35.0625 ∧ Real.sqrt (x - 2) + Real.sqrt (x + 4) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_root_l1463_146353


namespace NUMINAMATH_CALUDE_length_BC_is_44_div_3_l1463_146356

/-- Two externally tangent circles with a common external tangent line -/
structure TangentCircles where
  /-- Center of the first circle -/
  A : ℝ × ℝ
  /-- Center of the second circle -/
  B : ℝ × ℝ
  /-- Radius of the first circle -/
  r₁ : ℝ
  /-- Radius of the second circle -/
  r₂ : ℝ
  /-- Point where the external tangent line intersects ray AB -/
  C : ℝ × ℝ
  /-- The circles are externally tangent -/
  externally_tangent : dist A B = r₁ + r₂
  /-- The line through C is externally tangent to both circles -/
  is_external_tangent : ∃ (D E : ℝ × ℝ), 
    dist A D = r₁ ∧ dist B E = r₂ ∧ 
    (C.1 - D.1) * (A.1 - D.1) + (C.2 - D.2) * (A.2 - D.2) = 0 ∧
    (C.1 - E.1) * (B.1 - E.1) + (C.2 - E.2) * (B.2 - E.2) = 0
  /-- C lies on ray AB -/
  C_on_ray_AB : ∃ (t : ℝ), t ≥ 0 ∧ C = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

/-- The length of BC in the TangentCircles configuration -/
def length_BC (tc : TangentCircles) : ℝ :=
  dist tc.B tc.C

/-- The main theorem: length of BC is 44/3 -/
theorem length_BC_is_44_div_3 (tc : TangentCircles) (h₁ : tc.r₁ = 7) (h₂ : tc.r₂ = 4) : 
  length_BC tc = 44 / 3 := by
  sorry


end NUMINAMATH_CALUDE_length_BC_is_44_div_3_l1463_146356


namespace NUMINAMATH_CALUDE_shekars_social_studies_score_l1463_146317

theorem shekars_social_studies_score 
  (math_score science_score english_score biology_score : ℕ)
  (average_score : ℚ)
  (total_subjects : ℕ)
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : english_score = 67)
  (h4 : biology_score = 85)
  (h5 : average_score = 75)
  (h6 : total_subjects = 5) :
  ∃ (social_studies_score : ℕ),
    social_studies_score = 82 ∧
    (math_score + science_score + english_score + biology_score + social_studies_score : ℚ) / total_subjects = average_score :=
by
  sorry

end NUMINAMATH_CALUDE_shekars_social_studies_score_l1463_146317


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1463_146394

/-- The eccentricity of the hyperbola x²/3 - y²/6 = 1 is √3 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = Real.sqrt 3 ∧
  ∀ x y : ℝ, x^2 / 3 - y^2 / 6 = 1 → 
  e = Real.sqrt ((x^2 / 3 + y^2 / 6) / (x^2 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1463_146394


namespace NUMINAMATH_CALUDE_amaya_total_marks_l1463_146318

/-- Represents the marks scored in different subjects -/
structure Marks where
  music : ℕ
  social_studies : ℕ
  arts : ℕ
  maths : ℕ

/-- Calculates the total marks across all subjects -/
def total_marks (m : Marks) : ℕ :=
  m.music + m.social_studies + m.arts + m.maths

/-- Theorem stating the total marks Amaya scored -/
theorem amaya_total_marks :
  ∀ (m : Marks),
  m.music = 70 →
  m.social_studies = m.music + 10 →
  m.arts - m.maths = 20 →
  m.maths = (9 : ℕ) * m.arts / 10 →
  total_marks m = 530 := by
  sorry


end NUMINAMATH_CALUDE_amaya_total_marks_l1463_146318


namespace NUMINAMATH_CALUDE_exists_special_number_l1463_146339

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that returns the number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if all digits of a natural number are non-zero -/
def all_digits_nonzero (n : ℕ) : Prop := sorry

/-- Theorem: There exists a 1000-digit natural number with all non-zero digits that is divisible by the sum of its digits -/
theorem exists_special_number : 
  ∃ n : ℕ, 
    num_digits n = 1000 ∧ 
    all_digits_nonzero n ∧ 
    n % sum_of_digits n = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_special_number_l1463_146339


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l1463_146370

-- Define the number of Republicans and Democrats in the Senate committee
def totalRepublicans : ℕ := 10
def totalDemocrats : ℕ := 7

-- Define the number of Republicans and Democrats needed for the subcommittee
def subcommitteeRepublicans : ℕ := 4
def subcommitteeDemocrats : ℕ := 3

-- Theorem statement
theorem subcommittee_formation_count :
  (Nat.choose totalRepublicans subcommitteeRepublicans) *
  (Nat.choose totalDemocrats subcommitteeDemocrats) = 7350 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l1463_146370


namespace NUMINAMATH_CALUDE_outfit_combinations_l1463_146365

def num_shirts : ℕ := 8
def num_ties : ℕ := 5
def num_pants : ℕ := 4

theorem outfit_combinations : num_shirts * num_ties * num_pants = 160 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l1463_146365


namespace NUMINAMATH_CALUDE_bread_recipe_scaling_l1463_146345

/-- Given a recipe that requires 60 mL of water and 80 mL of milk for every 400 mL of flour,
    this theorem proves the amount of water and milk needed for 1200 mL of flour. -/
theorem bread_recipe_scaling (flour : ℝ) (water : ℝ) (milk : ℝ) 
  (h1 : flour = 1200)
  (h2 : water = 60 * (flour / 400))
  (h3 : milk = 80 * (flour / 400)) :
  water = 180 ∧ milk = 240 := by
  sorry

end NUMINAMATH_CALUDE_bread_recipe_scaling_l1463_146345


namespace NUMINAMATH_CALUDE_digit_difference_l1463_146313

theorem digit_difference (e : ℕ) (X Y : ℕ) : 
  e > 8 →
  X < e →
  Y < e →
  (e * X + Y) + (e * X + X) = 2 * e^2 + 4 * e + 3 →
  X - Y = (2 * e^2 + 4 * e - 726) / 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_l1463_146313


namespace NUMINAMATH_CALUDE_intersection_M_N_l1463_146301

def M : Set ℝ := {-1, 0, 1}

def N : Set ℝ := {x : ℝ | (x + 2) * (x - 1) < 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1463_146301


namespace NUMINAMATH_CALUDE_tan_15_plus_3sin_15_l1463_146326

theorem tan_15_plus_3sin_15 : 
  Real.tan (15 * π / 180) + 3 * Real.sin (15 * π / 180) = 
    (Real.sqrt 6 - Real.sqrt 2 + 3) / (Real.sqrt 6 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_tan_15_plus_3sin_15_l1463_146326


namespace NUMINAMATH_CALUDE_gold_coins_percentage_l1463_146397

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  beads : ℝ
  sculptures : ℝ
  coins : ℝ
  silverCoins : ℝ
  goldCoins : ℝ

/-- Theorem stating the percentage of gold coins in the urn -/
theorem gold_coins_percentage (u : UrnComposition) 
  (beads_percent : u.beads = 0.3)
  (sculptures_percent : u.sculptures = 0.1)
  (total_percent : u.beads + u.sculptures + u.coins = 1)
  (silver_coins_percent : u.silverCoins = 0.3 * u.coins)
  (coins_composition : u.silverCoins + u.goldCoins = u.coins) : 
  u.goldCoins = 0.42 := by
  sorry


end NUMINAMATH_CALUDE_gold_coins_percentage_l1463_146397


namespace NUMINAMATH_CALUDE_rectangle_segment_product_l1463_146390

theorem rectangle_segment_product (AB BC CD DE x : ℝ) : 
  AB = 5 →
  BC = 11 →
  CD = 3 →
  DE = 9 →
  0 < x →
  x < DE →
  AB * (AB + BC + CD + x) = x * (DE - x) →
  x = 11.95 := by
sorry

end NUMINAMATH_CALUDE_rectangle_segment_product_l1463_146390


namespace NUMINAMATH_CALUDE_emma_numbers_l1463_146357

theorem emma_numbers : ∃ (a b : ℤ), 
  ((a = 17 ∧ b = 31) ∨ (a = 31 ∧ b = 17)) ∧ 3 * a + 4 * b = 161 := by
  sorry

end NUMINAMATH_CALUDE_emma_numbers_l1463_146357


namespace NUMINAMATH_CALUDE_dalton_savings_l1463_146376

def jump_rope_cost : ℕ := 7
def board_game_cost : ℕ := 12
def playground_ball_cost : ℕ := 4
def uncle_money : ℕ := 13
def additional_money_needed : ℕ := 4

def total_cost : ℕ := jump_rope_cost + board_game_cost + playground_ball_cost

theorem dalton_savings (savings : ℕ) : 
  savings = total_cost - (uncle_money + additional_money_needed) := by
  sorry

end NUMINAMATH_CALUDE_dalton_savings_l1463_146376


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1463_146335

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 + x - 6 > 0 ↔ (x < -3 ∨ x > 2)) ↔
  (∀ x : ℝ, (x ≥ -3 ∧ x ≤ 2) → x^2 + x - 6 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1463_146335


namespace NUMINAMATH_CALUDE_product_of_fractions_l1463_146325

theorem product_of_fractions (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1463_146325


namespace NUMINAMATH_CALUDE_number_of_proper_subsets_l1463_146327

def U : Finset Nat := {0, 1, 2, 3}

def A : Finset Nat := {0, 1, 3}

def complement_A : Finset Nat := {2}

theorem number_of_proper_subsets :
  (U = {0, 1, 2, 3}) →
  (complement_A = {2}) →
  (A = U \ complement_A) →
  (Finset.powerset A).card - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_proper_subsets_l1463_146327


namespace NUMINAMATH_CALUDE_watermelon_seeds_l1463_146300

/-- Represents a watermelon slice with black and white seeds -/
structure WatermelonSlice where
  blackSeeds : ℕ
  whiteSeeds : ℕ

/-- Calculates the total number of seeds in a watermelon -/
def totalSeeds (slices : ℕ) (slice : WatermelonSlice) : ℕ :=
  slices * (slice.blackSeeds + slice.whiteSeeds)

/-- Theorem: The total number of seeds in the watermelon is 1600 -/
theorem watermelon_seeds :
  ∀ (slice : WatermelonSlice),
    slice.blackSeeds = 20 →
    slice.whiteSeeds = 20 →
    totalSeeds 40 slice = 1600 :=
by
  sorry

end NUMINAMATH_CALUDE_watermelon_seeds_l1463_146300


namespace NUMINAMATH_CALUDE_expression_undefined_l1463_146302

theorem expression_undefined (θ : ℝ) (h1 : θ > 0) (h2 : θ + 90 = 180) : 
  ¬∃x : ℝ, x = (Real.sin θ + Real.sin (2*θ) + Real.sin (3*θ) + Real.sin (4*θ)) / 
            (Real.cos (θ/2) * Real.cos θ * Real.cos (2*θ)) := by
  sorry

end NUMINAMATH_CALUDE_expression_undefined_l1463_146302


namespace NUMINAMATH_CALUDE_chris_age_l1463_146369

def problem (a b c : ℚ) : Prop :=
  -- The average of Amy's, Ben's, and Chris's ages is 10
  (a + b + c) / 3 = 10 ∧
  -- Five years ago, Chris was twice the age that Amy is now
  c - 5 = 2 * a ∧
  -- In 4 years, Ben's age will be 3/4 of Amy's age at that time
  b + 4 = 3 / 4 * (a + 4)

theorem chris_age (a b c : ℚ) (h : problem a b c) : c = 263 / 11 := by
  sorry

end NUMINAMATH_CALUDE_chris_age_l1463_146369


namespace NUMINAMATH_CALUDE_time_to_reach_B_after_second_meeting_l1463_146349

-- Define the variables
variable (S : ℝ) -- Total distance between A and B
variable (v_A v_B : ℝ) -- Speeds of A and B
variable (t : ℝ) -- Time taken by B to catch up with A

-- Define the theorem
theorem time_to_reach_B_after_second_meeting : 
  -- A starts 48 minutes (4/5 hours) before B
  v_A * (t + 4/5) = 2/3 * S →
  -- B catches up with A when A has traveled 2/3 of the distance
  v_B * t = 2/3 * S →
  -- They meet again 6 minutes (1/10 hour) after B leaves B
  v_A * (t + 4/5 + 1/2 * t + 1/10) + 1/10 * v_B = S →
  -- The time it takes for A to reach B after meeting B again is 12 minutes (1/5 hour)
  1/5 = S / v_A - (t + 4/5 + 1/2 * t + 1/10) := by
  sorry

end NUMINAMATH_CALUDE_time_to_reach_B_after_second_meeting_l1463_146349


namespace NUMINAMATH_CALUDE_probability_from_odds_l1463_146358

/-- Given odds in favor of an event as a ratio of two natural numbers -/
def OddsInFavor : Type := ℕ × ℕ

/-- Calculate the probability of an event given its odds in favor -/
def probability (odds : OddsInFavor) : ℚ :=
  let (favorable, unfavorable) := odds
  favorable / (favorable + unfavorable)

theorem probability_from_odds :
  let odds : OddsInFavor := (3, 5)
  probability odds = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_from_odds_l1463_146358


namespace NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l1463_146399

theorem inequality_holds_iff_a_in_range (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π/2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≥ 7/2 ∨ a ≤ Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l1463_146399


namespace NUMINAMATH_CALUDE_factorization_equality_l1463_146306

theorem factorization_equality (a : ℝ) : 
  (a^2 + a)^2 + 4*(a^2 + a) - 12 = (a - 1)*(a + 2)*(a^2 + a + 6) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1463_146306


namespace NUMINAMATH_CALUDE_dani_initial_pants_l1463_146342

/-- Represents the number of pants Dani receives each year as a reward -/
def yearly_reward : ℕ := 4 * 2

/-- Represents the number of years -/
def years : ℕ := 5

/-- Represents the total number of pants Dani will have after 5 years -/
def total_pants : ℕ := 90

/-- Calculates the number of pants Dani initially had -/
def initial_pants : ℕ := total_pants - (yearly_reward * years)

theorem dani_initial_pants :
  initial_pants = 50 := by sorry

end NUMINAMATH_CALUDE_dani_initial_pants_l1463_146342


namespace NUMINAMATH_CALUDE_mrs_hilt_pies_l1463_146364

/-- The total number of pies Mrs. Hilt needs to bake for the bigger event -/
def total_pies (pecan_initial : Float) (apple_initial : Float) (cherry_initial : Float)
                (pecan_multiplier : Float) (apple_multiplier : Float) (cherry_multiplier : Float) : Float :=
  pecan_initial * pecan_multiplier + apple_initial * apple_multiplier + cherry_initial * cherry_multiplier

/-- Theorem stating that Mrs. Hilt needs to bake 193.5 pies for the bigger event -/
theorem mrs_hilt_pies : 
  total_pies 16.5 14.25 12.75 4.3 3.5 5.7 = 193.5 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pies_l1463_146364


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l1463_146340

/-- A line with equal x and y intercepts passing through a given point. -/
structure EqualInterceptLine where
  -- The x-coordinate of the point the line passes through
  x : ℝ
  -- The y-coordinate of the point the line passes through
  y : ℝ
  -- The common intercept value
  a : ℝ
  -- The line passes through the point (x, y)
  point_on_line : x / a + y / a = 1

/-- The equation of a line with equal x and y intercepts passing through (2,1) is x + y - 3 = 0 -/
theorem equal_intercept_line_equation :
  ∀ (l : EqualInterceptLine), l.x = 2 ∧ l.y = 1 → (λ x y => x + y - 3 = 0) = (λ x y => x / l.a + y / l.a = 1) :=
by sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l1463_146340


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1463_146324

theorem functional_equation_solution (f : ℚ → ℚ) :
  (∀ x y : ℚ, f (x + f y) = f x * f y) →
  (∀ x : ℚ, f x = 0) ∨ (∀ x : ℚ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1463_146324


namespace NUMINAMATH_CALUDE_tan_difference_special_angle_l1463_146362

theorem tan_difference_special_angle (α : Real) :
  2 * Real.tan α = 3 * Real.tan (π / 8) →
  Real.tan (α - π / 8) = (5 * Real.sqrt 2 + 1) / 49 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_special_angle_l1463_146362


namespace NUMINAMATH_CALUDE_power_zero_l1463_146384

theorem power_zero (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_l1463_146384


namespace NUMINAMATH_CALUDE_expression_value_l1463_146329

theorem expression_value (x y : ℝ) (h : 2 * y - x = 5) :
  5 * (x - 2 * y)^2 + 3 * (x - 2 * y) + 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1463_146329


namespace NUMINAMATH_CALUDE_quadratic_root_constant_l1463_146372

/-- 
Given a quadratic equation 5x^2 + 6x + k = 0 with roots (-3 ± √69) / 10,
prove that k = -1.65
-/
theorem quadratic_root_constant (k : ℝ) : 
  (∀ x : ℝ, 5 * x^2 + 6 * x + k = 0 ↔ x = (-3 - Real.sqrt 69) / 10 ∨ x = (-3 + Real.sqrt 69) / 10) →
  k = -1.65 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_constant_l1463_146372


namespace NUMINAMATH_CALUDE_probability_two_green_bottles_l1463_146360

/-- The probability of selecting 2 green bottles out of 4 green bottles and 38 black bottles -/
theorem probability_two_green_bottles (green_bottles : ℕ) (black_bottles : ℕ) : 
  green_bottles = 4 → black_bottles = 38 → 
  (Nat.choose green_bottles 2 : ℚ) / (Nat.choose (green_bottles + black_bottles) 2) = 1 / 143.5 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_green_bottles_l1463_146360


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1463_146361

theorem coefficient_x_cubed_in_expansion : ∃ (c : ℤ), 
  (2 - X) * (1 - X)^4 = -14 * X^3 + c * X^4 + X^2 * (2 - X) * (1 - X)^2 + (2 - X) * (1 - X)^4 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1463_146361


namespace NUMINAMATH_CALUDE_cos_135_and_point_on_unit_circle_l1463_146354

theorem cos_135_and_point_on_unit_circle :
  let angle : Real := 135 * π / 180
  let Q : ℝ × ℝ := (Real.cos angle, Real.sin angle)
  (Real.cos angle = -Real.sqrt 2 / 2) ∧
  (Q = (-Real.sqrt 2 / 2, Real.sqrt 2 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_cos_135_and_point_on_unit_circle_l1463_146354


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1463_146305

theorem arithmetic_calculation : 2 + 8 * 3 - 4 + 7 * 2 / 2 * 3 = 43 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1463_146305


namespace NUMINAMATH_CALUDE_number_problem_l1463_146312

theorem number_problem (n p q : ℝ) 
  (h1 : n / p = 6)
  (h2 : n / q = 15)
  (h3 : p - q = 0.3) :
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l1463_146312


namespace NUMINAMATH_CALUDE_sangho_video_votes_l1463_146366

theorem sangho_video_votes (score : ℤ) (like_percentage : ℚ) (total_votes : ℕ) :
  score = 120 ∧
  like_percentage = 3/4 ∧
  (like_percentage * total_votes : ℚ) - ((1 - like_percentage) * total_votes : ℚ) = score ∧
  total_votes > 0
  → total_votes = 240 := by
  sorry

end NUMINAMATH_CALUDE_sangho_video_votes_l1463_146366


namespace NUMINAMATH_CALUDE_movie_date_candy_cost_l1463_146333

theorem movie_date_candy_cost
  (ticket_cost : ℝ)
  (combo_cost : ℝ)
  (total_spend : ℝ)
  (num_candy : ℕ)
  (h1 : ticket_cost = 20)
  (h2 : combo_cost = 11)
  (h3 : total_spend = 36)
  (h4 : num_candy = 2) :
  (total_spend - ticket_cost - combo_cost) / num_candy = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_movie_date_candy_cost_l1463_146333


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l1463_146348

open Set

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 ≤ x ∧ x < 3}

theorem set_intersection_theorem : A ∩ B = Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l1463_146348


namespace NUMINAMATH_CALUDE_factorization_equality_l1463_146367

theorem factorization_equality (a b : ℝ) : a * b^2 - a = a * (b + 1) * (b - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1463_146367


namespace NUMINAMATH_CALUDE_even_product_probability_l1463_146334

def set_A_odd : ℕ := 7
def set_A_even : ℕ := 9
def set_B_odd : ℕ := 5
def set_B_even : ℕ := 4

def total_A : ℕ := set_A_odd + set_A_even
def total_B : ℕ := set_B_odd + set_B_even

def prob_even_product : ℚ := 109 / 144

theorem even_product_probability :
  (set_A_even : ℚ) / total_A * (set_B_even : ℚ) / total_B +
  (set_A_odd : ℚ) / total_A * (set_B_even : ℚ) / total_B +
  (set_A_even : ℚ) / total_A * (set_B_odd : ℚ) / total_B = prob_even_product :=
by sorry

end NUMINAMATH_CALUDE_even_product_probability_l1463_146334


namespace NUMINAMATH_CALUDE_problem_solution_l1463_146363

theorem problem_solution (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 16) : y = 64 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1463_146363


namespace NUMINAMATH_CALUDE_volume_of_sphere_wedge_l1463_146382

/-- The volume of a wedge when a sphere with circumference 18π is cut into 6 congruent parts -/
theorem volume_of_sphere_wedge : 
  ∀ (r : ℝ) (V : ℝ),
  (2 * Real.pi * r = 18 * Real.pi) →  -- Circumference condition
  (V = (4/3) * Real.pi * r^3) →       -- Volume of sphere formula
  (V / 6 = 162 * Real.pi) :=          -- Volume of one wedge
by sorry

end NUMINAMATH_CALUDE_volume_of_sphere_wedge_l1463_146382


namespace NUMINAMATH_CALUDE_triangle_c_coordinates_l1463_146368

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the area function
def area (t : Triangle) : ℝ := sorry

-- Define the line equation
def onLine (p : ℝ × ℝ) : Prop :=
  3 * p.1 - p.2 + 3 = 0

-- Theorem statement
theorem triangle_c_coordinates :
  ∀ (t : Triangle),
    t.A = (3, 2) →
    t.B = (-1, 5) →
    onLine t.C →
    area t = 10 →
    (t.C = (-1, 0) ∨ t.C = (5/3, 8)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_c_coordinates_l1463_146368


namespace NUMINAMATH_CALUDE_a_neg_two_sufficient_not_necessary_l1463_146341

/-- The line l₁ with equation x + ay - 2 = 0 -/
def l₁ (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + a * p.2 - 2 = 0}

/-- The line l₂ with equation (a+1)x - ay + 1 = 0 -/
def l₂ (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (a + 1) * p.1 - a * p.2 + 1 = 0}

/-- Two lines are parallel if they have the same slope -/
def parallel (l₁ l₂ : Set (ℝ × ℝ)) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ l₁ ↔ (x, k * y) ∈ l₂

/-- Theorem stating that a = -2 is a sufficient but not necessary condition for l₁ ∥ l₂ -/
theorem a_neg_two_sufficient_not_necessary :
  (∃ (a : ℝ), a ≠ -2 ∧ parallel (l₁ a) (l₂ a)) ∧
  (parallel (l₁ (-2)) (l₂ (-2))) := by
  sorry

end NUMINAMATH_CALUDE_a_neg_two_sufficient_not_necessary_l1463_146341


namespace NUMINAMATH_CALUDE_trajectory_is_circle_l1463_146309

/-- The ellipse with equation x²/7 + y²/3 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 7 + p.2^2 / 3 = 1}

/-- The left focus of the ellipse -/
def F₁ : ℝ × ℝ := (-2, 0)

/-- The right focus of the ellipse -/
def F₂ : ℝ × ℝ := (2, 0)

/-- The set of all points Q obtained by extending F₁P to Q such that |PQ| = |PF₂| for all P on the ellipse -/
def TrajectoryQ (P : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {Q : ℝ × ℝ | P ∈ Ellipse ∧ ∃ t : ℝ, t > 1 ∧ Q = (t • (P - F₁) + F₁) ∧ 
    ‖Q - P‖ = ‖P - F₂‖}

/-- The theorem stating that the trajectory of Q is a circle -/
theorem trajectory_is_circle : 
  ∀ Q : ℝ × ℝ, (∃ P : ℝ × ℝ, Q ∈ TrajectoryQ P) ↔ (Q.1 + 2)^2 + Q.2^2 = 28 :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_circle_l1463_146309


namespace NUMINAMATH_CALUDE_simplify_expression_l1463_146337

theorem simplify_expression : 20 + (-14) - (-18) + 13 = 37 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1463_146337


namespace NUMINAMATH_CALUDE_cylinder_volume_from_unit_square_l1463_146347

/-- The volume of a cylinder formed by rolling a unit square -/
theorem cylinder_volume_from_unit_square : 
  ∃ (V : ℝ), V = (1 : ℝ) / (4 * Real.pi) ∧ 
  (∃ (r h : ℝ), r = (1 : ℝ) / (2 * Real.pi) ∧ h = 1 ∧ V = Real.pi * r^2 * h) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_unit_square_l1463_146347


namespace NUMINAMATH_CALUDE_collinear_probability_l1463_146380

/-- Represents a rectangular grid of dots -/
structure DotGrid :=
  (rows : ℕ)
  (columns : ℕ)

/-- Calculates the total number of dots in the grid -/
def DotGrid.total_dots (g : DotGrid) : ℕ := g.rows * g.columns

/-- Calculates the number of ways to choose 4 dots from n dots -/
def choose_four (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3) / 24

/-- Represents the number of collinear sets in the grid -/
def collinear_sets (g : DotGrid) : ℕ := 
  g.rows * 2 + g.columns + 4

/-- The main theorem stating the probability of four randomly chosen dots being collinear -/
theorem collinear_probability (g : DotGrid) (h1 : g.rows = 4) (h2 : g.columns = 5) : 
  (collinear_sets g : ℚ) / (choose_four (g.total_dots) : ℚ) = 17 / 4845 := by
  sorry

#eval collinear_sets (DotGrid.mk 4 5)
#eval choose_four 20

end NUMINAMATH_CALUDE_collinear_probability_l1463_146380


namespace NUMINAMATH_CALUDE_band_gigs_played_l1463_146304

/-- Represents the earnings of each band member per gig -/
structure BandEarnings :=
  (leadSinger : ℕ)
  (guitarist : ℕ)
  (bassist : ℕ)
  (drummer : ℕ)
  (keyboardist : ℕ)
  (backupSinger1 : ℕ)
  (backupSinger2 : ℕ)
  (backupSinger3 : ℕ)

/-- Calculates the total earnings per gig for the band -/
def totalEarningsPerGig (earnings : BandEarnings) : ℕ :=
  earnings.leadSinger + earnings.guitarist + earnings.bassist + earnings.drummer +
  earnings.keyboardist + earnings.backupSinger1 + earnings.backupSinger2 + earnings.backupSinger3

/-- Theorem: The band has played 21 gigs -/
theorem band_gigs_played (earnings : BandEarnings) 
  (h1 : earnings.leadSinger = 30)
  (h2 : earnings.guitarist = 25)
  (h3 : earnings.bassist = 20)
  (h4 : earnings.drummer = 25)
  (h5 : earnings.keyboardist = 20)
  (h6 : earnings.backupSinger1 = 15)
  (h7 : earnings.backupSinger2 = 18)
  (h8 : earnings.backupSinger3 = 12)
  (h9 : totalEarningsPerGig earnings * 21 = 3465) :
  21 = 3465 / (totalEarningsPerGig earnings) :=
by sorry

end NUMINAMATH_CALUDE_band_gigs_played_l1463_146304


namespace NUMINAMATH_CALUDE_weight_of_pecans_l1463_146396

/-- Given the total weight of nuts and the weight of almonds, calculate the weight of pecans. -/
theorem weight_of_pecans (total_weight : ℝ) (almond_weight : ℝ) 
  (h1 : total_weight = 0.52) 
  (h2 : almond_weight = 0.14) : 
  total_weight - almond_weight = 0.38 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_pecans_l1463_146396


namespace NUMINAMATH_CALUDE_or_false_sufficient_not_necessary_for_and_false_l1463_146387

theorem or_false_sufficient_not_necessary_for_and_false (p q : Prop) :
  (¬(p ∨ q) → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → ¬(p ∨ q)) :=
sorry

end NUMINAMATH_CALUDE_or_false_sufficient_not_necessary_for_and_false_l1463_146387


namespace NUMINAMATH_CALUDE_choose_three_from_fifteen_l1463_146398

theorem choose_three_from_fifteen : Nat.choose 15 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_fifteen_l1463_146398


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_3_dividing_27_factorial_l1463_146320

/-- The largest power of 3 that divides 27! -/
def largest_power_of_3 : ℕ := 13

/-- The ones digit of 3^n -/
def ones_digit_of_3_power (n : ℕ) : ℕ :=
  (3^n) % 10

theorem ones_digit_of_largest_power_of_3_dividing_27_factorial :
  ones_digit_of_3_power largest_power_of_3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_3_dividing_27_factorial_l1463_146320


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1463_146393

/-- Two vectors are parallel if the ratio of their corresponding components is equal -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 / b.1 = a.2 / b.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  parallel a b → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1463_146393


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1463_146378

/-- The number of games played in a chess tournament -/
def numGames (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 15 participants, where each participant
    plays exactly one game with each of the remaining participants,
    the total number of games played is 105. -/
theorem chess_tournament_games :
  numGames 15 = 105 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1463_146378


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l1463_146377

theorem absolute_value_equation_solution_difference : ∃ (x y : ℝ), 
  (x ≠ y ∧ 
   (|x^2 + 3*x + 3| = 15 ∧ |y^2 + 3*y + 3| = 15) ∧
   ∀ z : ℝ, |z^2 + 3*z + 3| = 15 → (z = x ∨ z = y)) →
  |x - y| = 7 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l1463_146377


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l1463_146388

theorem sum_of_two_numbers (A B : ℝ) (h1 : A + B = 147) (h2 : A = 0.375 * B + 4) : A + B = 147 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l1463_146388


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1463_146316

theorem inequality_equivalence (x : ℝ) :
  (x - 3) / ((x - 4)^2 + 8) ≥ 0 ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1463_146316


namespace NUMINAMATH_CALUDE_cross_tangential_cubic_cross_tangential_sine_cross_tangential_tangent_l1463_146374

-- Define the concept of cross-tangential intersection
def cross_tangential_intersection (l : ℝ → ℝ) (c : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  let (x₀, y₀) := p
  -- Condition (i): The line l is tangent to the curve C at the point P(x₀, y₀)
  (deriv c x₀ = deriv l x₀) ∧
  -- Condition (ii): The curve C lies on both sides of the line l near point P
  (∃ δ > 0, ∀ x ∈ Set.Ioo (x₀ - δ) (x₀ + δ),
    (x < x₀ → c x < l x) ∧ (x > x₀ → c x > l x) ∨
    (x < x₀ → c x > l x) ∧ (x > x₀ → c x < l x))

-- Statement 1
theorem cross_tangential_cubic :
  cross_tangential_intersection (λ _ => 0) (λ x => x^3) (0, 0) :=
sorry

-- Statement 3
theorem cross_tangential_sine :
  cross_tangential_intersection (λ x => x) Real.sin (0, 0) :=
sorry

-- Statement 4
theorem cross_tangential_tangent :
  cross_tangential_intersection (λ x => x) Real.tan (0, 0) :=
sorry

end NUMINAMATH_CALUDE_cross_tangential_cubic_cross_tangential_sine_cross_tangential_tangent_l1463_146374


namespace NUMINAMATH_CALUDE_magazine_cost_l1463_146350

theorem magazine_cost (b m : ℝ) 
  (h1 : 2 * b + 2 * m = 26) 
  (h2 : b + 3 * m = 27) : 
  m = 7 := by
sorry

end NUMINAMATH_CALUDE_magazine_cost_l1463_146350


namespace NUMINAMATH_CALUDE_classroom_ratio_problem_l1463_146310

theorem classroom_ratio_problem (total_students : ℕ) (girl_ratio boy_ratio : ℕ) 
  (h1 : total_students = 30)
  (h2 : girl_ratio = 1)
  (h3 : boy_ratio = 2) : 
  (total_students * boy_ratio) / (girl_ratio + boy_ratio) = 20 := by
  sorry

end NUMINAMATH_CALUDE_classroom_ratio_problem_l1463_146310


namespace NUMINAMATH_CALUDE_circle_passes_through_intersections_and_tangent_to_line_l1463_146383

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0
def l (x y : ℝ) : Prop := x + 2*y = 0

-- Define the desired circle
def desiredCircle (x y : ℝ) : Prop := (x - 1/2)^2 + (y - 1)^2 = 5/4

-- Theorem statement
theorem circle_passes_through_intersections_and_tangent_to_line :
  ∀ x y : ℝ,
  (C₁ x y ∧ C₂ x y → desiredCircle x y) ∧
  (∃ t : ℝ, l (1/2 + t) (1 - t/2) ∧
    ∀ s : ℝ, s ≠ t → ¬(desiredCircle (1/2 + s) (1 - s/2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_intersections_and_tangent_to_line_l1463_146383


namespace NUMINAMATH_CALUDE_ellipse_theorem_l1463_146303

/-- Definition of the ellipse C -/
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the major axis length -/
def majorAxisLength (a : ℝ) : Prop :=
  2 * a = 2 * Real.sqrt 2

/-- Definition of the range for point N's x-coordinate -/
def NxRange (x : ℝ) : Prop :=
  -1/4 < x ∧ x < 0

/-- Main theorem -/
theorem ellipse_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : majorAxisLength a) :
  (∀ x y : ℝ, ellipse x y a b ↔ x^2 / 2 + y^2 = 1) ∧
  (∀ N A B : ℝ × ℝ,
    NxRange N.1 →
    (∃ k : ℝ, 
      ellipse A.1 A.2 (Real.sqrt 2) 1 ∧
      ellipse B.1 B.2 (Real.sqrt 2) 1 ∧
      A.2 = k * (A.1 + 1) ∧
      B.2 = k * (B.1 + 1)) →
    3 * Real.sqrt 2 / 2 < Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) < 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l1463_146303


namespace NUMINAMATH_CALUDE_dish_temperature_l1463_146319

/-- Calculates the final temperature of a dish in an oven -/
def final_temperature (start_temp : ℝ) (heating_rate : ℝ) (cooking_time : ℝ) : ℝ :=
  start_temp + heating_rate * cooking_time

/-- Proves that the dish reaches 100 degrees given the specified conditions -/
theorem dish_temperature : final_temperature 20 5 16 = 100 := by
  sorry

end NUMINAMATH_CALUDE_dish_temperature_l1463_146319


namespace NUMINAMATH_CALUDE_number_divisibility_l1463_146330

theorem number_divisibility (x : ℝ) : x / 14.5 = 171 → x = 2479.5 := by
  sorry

end NUMINAMATH_CALUDE_number_divisibility_l1463_146330


namespace NUMINAMATH_CALUDE_lost_pages_problem_l1463_146338

/-- Calculates the number of lost pages of stickers -/
def lost_pages (stickers_per_page : ℕ) (initial_pages : ℕ) (remaining_stickers : ℕ) : ℕ :=
  (stickers_per_page * initial_pages - remaining_stickers) / stickers_per_page

theorem lost_pages_problem :
  let stickers_per_page : ℕ := 20
  let initial_pages : ℕ := 12
  let remaining_stickers : ℕ := 220
  lost_pages stickers_per_page initial_pages remaining_stickers = 1 := by
  sorry

end NUMINAMATH_CALUDE_lost_pages_problem_l1463_146338


namespace NUMINAMATH_CALUDE_inequality_conditions_l1463_146375

theorem inequality_conditions (a b c : ℝ) 
  (h : ∀ (x y z : ℝ), a * (x - y) * (x - z) + b * (y - x) * (y - z) + c * (z - x) * (z - y) ≥ 0) : 
  (-a + 2*b + 2*c ≥ 0) ∧ (2*a - b + 2*c ≥ 0) ∧ (2*a + 2*b - c ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_conditions_l1463_146375


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1463_146308

theorem fraction_multiplication : (1 / 3 : ℚ) * (3 / 4 : ℚ) * (4 / 5 : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1463_146308


namespace NUMINAMATH_CALUDE_find_B_l1463_146351

theorem find_B (A B : ℕ) (h1 : A < 10) (h2 : B < 10) (h3 : 6 * 100 * A + 5 + 100 * B + 3 = 748) : B = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_B_l1463_146351


namespace NUMINAMATH_CALUDE_intersection_perpendicular_tangents_l1463_146323

open Real

theorem intersection_perpendicular_tangents (a : ℝ) : 
  ∃ (x : ℝ), 0 < x ∧ x < π / 2 ∧ 
  2 * sin x = a * cos x ∧
  (2 * cos x) * (-a * sin x) = -1 →
  a = 2 * sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_tangents_l1463_146323


namespace NUMINAMATH_CALUDE_squash_league_max_salary_l1463_146311

/-- Represents the maximum salary a player can earn in a professional squash league --/
def max_salary (team_size : ℕ) (min_salary : ℕ) (total_payroll : ℕ) : ℕ :=
  total_payroll - (team_size - 1) * min_salary

/-- Theorem stating the maximum salary in the given conditions --/
theorem squash_league_max_salary :
  max_salary 22 16000 880000 = 544000 := by
  sorry

end NUMINAMATH_CALUDE_squash_league_max_salary_l1463_146311


namespace NUMINAMATH_CALUDE_min_value_cube_root_plus_inverse_square_l1463_146355

theorem min_value_cube_root_plus_inverse_square (x : ℝ) (h : x > 0) :
  3 * x^(1/3) + 1 / x^2 ≥ 4 ∧
  (3 * x^(1/3) + 1 / x^2 = 4 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_cube_root_plus_inverse_square_l1463_146355


namespace NUMINAMATH_CALUDE_school_population_l1463_146331

theorem school_population (girls boys teachers : ℕ) 
  (h1 : girls = 315) 
  (h2 : boys = 309) 
  (h3 : teachers = 772) : 
  girls + boys + teachers = 1396 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l1463_146331


namespace NUMINAMATH_CALUDE_john_daily_earnings_l1463_146381

/-- Calculate daily earnings from website visits -/
def daily_earnings (visits_per_month : ℕ) (days_per_month : ℕ) (earnings_per_visit : ℚ) : ℚ :=
  (visits_per_month : ℚ) * earnings_per_visit / (days_per_month : ℚ)

/-- Prove that John's daily earnings are $10 -/
theorem john_daily_earnings :
  daily_earnings 30000 30 (1 / 100) = 10 := by
  sorry

end NUMINAMATH_CALUDE_john_daily_earnings_l1463_146381


namespace NUMINAMATH_CALUDE_integer_roots_parity_l1463_146343

theorem integer_roots_parity (n : ℤ) (x₁ x₂ : ℤ) : 
  x₁^2 + (4*n + 1)*x₁ + 2*n = 0 ∧ 
  x₂^2 + (4*n + 1)*x₂ + 2*n = 0 →
  (Odd x₁ ∧ Even x₂) ∨ (Even x₁ ∧ Odd x₂) :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_parity_l1463_146343


namespace NUMINAMATH_CALUDE_percentage_problem_l1463_146314

theorem percentage_problem (N : ℝ) (P : ℝ) 
  (h1 : P / 100 * N = 160)
  (h2 : 60 / 100 * N = 240) : 
  P = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l1463_146314


namespace NUMINAMATH_CALUDE_goldfish_cost_price_l1463_146352

theorem goldfish_cost_price (selling_price : ℝ) (goldfish_sold : ℕ) (tank_cost : ℝ) (profit_percentage : ℝ) :
  selling_price = 0.75 →
  goldfish_sold = 110 →
  tank_cost = 100 →
  profit_percentage = 0.55 →
  ∃ (cost_price : ℝ),
    cost_price = 0.25 ∧
    (goldfish_sold : ℝ) * (selling_price - cost_price) = profit_percentage * tank_cost :=
by sorry

end NUMINAMATH_CALUDE_goldfish_cost_price_l1463_146352


namespace NUMINAMATH_CALUDE_square_root_difference_l1463_146391

theorem square_root_difference (a b : ℝ) (ha : a = 7 + 4 * Real.sqrt 3) (hb : b = 7 - 4 * Real.sqrt 3) :
  Real.sqrt a - Real.sqrt b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_difference_l1463_146391


namespace NUMINAMATH_CALUDE_prob_zeros_not_adjacent_is_point_six_l1463_146322

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 3

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of elements to be arranged -/
def total_elements : ℕ := num_ones + num_zeros

/-- The probability that two zeros are not adjacent when arranging num_ones ones and num_zeros zeros in a row -/
def prob_zeros_not_adjacent : ℚ :=
  1 - (2 * (Nat.factorial (total_elements - 1))) / (Nat.factorial total_elements)

theorem prob_zeros_not_adjacent_is_point_six :
  prob_zeros_not_adjacent = 3/5 :=
sorry

end NUMINAMATH_CALUDE_prob_zeros_not_adjacent_is_point_six_l1463_146322


namespace NUMINAMATH_CALUDE_anna_swept_ten_rooms_l1463_146332

/-- Represents the time in minutes for various chores -/
structure ChoreTime where
  sweepingPerRoom : ℕ
  washingPerDish : ℕ
  laundryPerLoad : ℕ

/-- Represents the chores assigned to Billy -/
structure BillyChores where
  laundryLoads : ℕ
  dishesToWash : ℕ

/-- Calculates the total time Billy spends on chores -/
def billyTotalTime (ct : ChoreTime) (bc : BillyChores) : ℕ :=
  bc.laundryLoads * ct.laundryPerLoad + bc.dishesToWash * ct.washingPerDish

/-- Theorem stating that Anna swept 10 rooms -/
theorem anna_swept_ten_rooms (ct : ChoreTime) (bc : BillyChores) 
    (h1 : ct.sweepingPerRoom = 3)
    (h2 : ct.washingPerDish = 2)
    (h3 : ct.laundryPerLoad = 9)
    (h4 : bc.laundryLoads = 2)
    (h5 : bc.dishesToWash = 6) :
    ∃ (rooms : ℕ), rooms * ct.sweepingPerRoom = billyTotalTime ct bc ∧ rooms = 10 := by
  sorry

end NUMINAMATH_CALUDE_anna_swept_ten_rooms_l1463_146332


namespace NUMINAMATH_CALUDE_airport_walk_probability_l1463_146336

/-- Represents an airport with a given number of gates and distance between adjacent gates -/
structure Airport where
  num_gates : ℕ
  distance_between_gates : ℕ

/-- Calculates the number of gate pairs within a given distance -/
def count_pairs_within_distance (a : Airport) (max_distance : ℕ) : ℕ :=
  sorry

/-- The probability of walking at most a given distance between two random gates -/
def probability_within_distance (a : Airport) (max_distance : ℕ) : ℚ :=
  sorry

theorem airport_walk_probability :
  let a : Airport := ⟨15, 90⟩
  probability_within_distance a 360 = 59 / 105 := by
  sorry

end NUMINAMATH_CALUDE_airport_walk_probability_l1463_146336


namespace NUMINAMATH_CALUDE_total_village_tax_l1463_146373

/-- Represents the farm tax collected from a village -/
structure FarmTax where
  total_tax : ℝ
  willam_tax : ℝ
  willam_land_percentage : ℝ

/-- Theorem stating the total tax collected from the village -/
theorem total_village_tax (ft : FarmTax) 
  (h1 : ft.willam_tax = 480)
  (h2 : ft.willam_land_percentage = 25) :
  ft.total_tax = 1920 := by
  sorry

end NUMINAMATH_CALUDE_total_village_tax_l1463_146373


namespace NUMINAMATH_CALUDE_f_continuous_at_5_l1463_146395

def f (x : ℝ) : ℝ := 2 * x^2 + 8

theorem f_continuous_at_5 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 5| < δ → |f x - f 5| < ε :=
by sorry

end NUMINAMATH_CALUDE_f_continuous_at_5_l1463_146395


namespace NUMINAMATH_CALUDE_sum_in_range_l1463_146392

theorem sum_in_range : ∃ (s : ℚ), 
  s = (1 + 3/8) + (4 + 1/3) + (6 + 2/21) ∧ 11 < s ∧ s < 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_range_l1463_146392


namespace NUMINAMATH_CALUDE_square_sum_equals_eight_l1463_146386

theorem square_sum_equals_eight (m : ℝ) 
  (h : (2018 + m) * (2020 + m) = 2) : 
  (2018 + m)^2 + (2020 + m)^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_eight_l1463_146386


namespace NUMINAMATH_CALUDE_lucy_fish_count_l1463_146379

theorem lucy_fish_count (initial_fish : ℕ) (fish_to_buy : ℕ) (total_fish : ℕ) : 
  initial_fish = 212 → fish_to_buy = 68 → total_fish = initial_fish + fish_to_buy → total_fish = 280 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_count_l1463_146379


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1463_146321

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |2*x - 5| = 3*x - 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1463_146321


namespace NUMINAMATH_CALUDE_abigail_report_time_l1463_146359

/-- Calculates the time needed to finish a report given the total words required,
    words already written, and typing speed. -/
def timeToFinishReport (totalWords : ℕ) (writtenWords : ℕ) (wordsPerHalfHour : ℕ) : ℕ :=
  let remainingWords := totalWords - writtenWords
  let wordsPerMinute := wordsPerHalfHour / 30
  remainingWords / wordsPerMinute

/-- Proves that given the conditions in the problem, 
    it will take 80 minutes to finish the report. -/
theorem abigail_report_time : 
  timeToFinishReport 1000 200 300 = 80 := by
  sorry

end NUMINAMATH_CALUDE_abigail_report_time_l1463_146359


namespace NUMINAMATH_CALUDE_derivative_even_implies_a_equals_three_l1463_146307

/-- Given a function f(x) = x³ + (a-3)x² + αx, prove that if its derivative f'(x) is an even function, then a = 3 -/
theorem derivative_even_implies_a_equals_three (a α : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 + (a - 3) * x^2 + α * x
  let f' : ℝ → ℝ := λ x ↦ deriv f x
  (∀ x, f' (-x) = f' x) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_even_implies_a_equals_three_l1463_146307


namespace NUMINAMATH_CALUDE_binary_sum_theorem_l1463_146346

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Adds two binary numbers (represented as lists of bits) and returns the result as a list of bits. -/
def add_binary (a b : List Bool) : List Bool :=
  sorry -- Implementation details omitted

/-- Theorem: The sum of 1101₂, 100₂, 111₂, and 11010₂ is equal to 111001₂ -/
theorem binary_sum_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [false, false, true]       -- 100₂
  let c := [true, true, true]         -- 111₂
  let d := [false, true, false, true, true]  -- 11010₂
  let result := [true, false, false, true, true, true]  -- 111001₂
  add_binary (add_binary (add_binary a b) c) d = result := by
  sorry

#eval binary_to_nat [true, false, false, true, true, true]  -- Should output 57

end NUMINAMATH_CALUDE_binary_sum_theorem_l1463_146346
