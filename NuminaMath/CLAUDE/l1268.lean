import Mathlib

namespace flower_count_l1268_126838

theorem flower_count (roses tulips lilies : ℕ) : 
  roses = 58 ∧ 
  roses = tulips + 15 ∧ 
  roses = lilies - 25 → 
  roses + tulips + lilies = 184 := by
sorry

end flower_count_l1268_126838


namespace net_weekly_increase_is_five_l1268_126898

/-- Calculates the net weekly increase in earnings given a raise, work hours, and housing benefit reduction -/
def netWeeklyIncrease (raise : ℚ) (workHours : ℕ) (housingBenefitReduction : ℚ) : ℚ :=
  let weeklyRaise := raise * workHours
  let weeklyHousingBenefitReduction := housingBenefitReduction / 4
  weeklyRaise - weeklyHousingBenefitReduction

/-- Theorem stating that given the specified conditions, the net weekly increase is $5 -/
theorem net_weekly_increase_is_five :
  netWeeklyIncrease (1/2) 40 60 = 5 := by
  sorry

end net_weekly_increase_is_five_l1268_126898


namespace triangle_radius_inequality_l1268_126877

structure Triangle where
  R : ℝ  -- circumradius
  r : ℝ  -- inradius
  P : ℝ  -- perimeter
  is_acute_or_obtuse : Bool
  is_right_angled : Bool
  positive_R : R > 0
  positive_r : r > 0
  positive_P : P > 0

theorem triangle_radius_inequality (t : Triangle) : 
  (t.is_acute_or_obtuse ∧ t.R > (Real.sqrt 3 / 3) * Real.sqrt (t.P * t.r)) ∨
  (t.is_right_angled ∧ t.R ≥ (Real.sqrt 2 / 2) * Real.sqrt (t.P * t.r)) :=
sorry

end triangle_radius_inequality_l1268_126877


namespace alcohol_solution_proof_l1268_126859

theorem alcohol_solution_proof (initial_volume : ℝ) (initial_percentage : ℝ) 
  (target_percentage : ℝ) (added_alcohol : ℝ) : 
  initial_volume = 100 ∧ 
  initial_percentage = 0.20 ∧ 
  target_percentage = 0.30 ∧
  added_alcohol = 14.2857 →
  (initial_volume * initial_percentage + added_alcohol) / (initial_volume + added_alcohol) = target_percentage :=
by
  sorry

end alcohol_solution_proof_l1268_126859


namespace negation_equivalence_l1268_126840

theorem negation_equivalence (m : ℤ) : 
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) := by
  sorry

end negation_equivalence_l1268_126840


namespace ceiling_of_negative_three_point_seven_l1268_126835

theorem ceiling_of_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end ceiling_of_negative_three_point_seven_l1268_126835


namespace max_min_f_on_interval_l1268_126850

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) := x^3 - 3*x

-- Define the interval [-2, 0]
def interval := Set.Icc (-2 : ℝ) 0

-- Theorem statement
theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max = 2 ∧ min = -2 := by
  sorry

end max_min_f_on_interval_l1268_126850


namespace no_integer_roots_l1268_126857

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- Evaluates a polynomial at a given integer -/
def eval (p : IntPolynomial) (x : ℤ) : ℤ := p x

/-- Predicate for odd integers -/
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem no_integer_roots (p : IntPolynomial) 
  (h0 : is_odd (eval p 0)) 
  (h1 : is_odd (eval p 1)) : 
  ∀ k : ℤ, eval p k ≠ 0 := by
sorry

end no_integer_roots_l1268_126857


namespace complement_of_M_l1268_126878

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x : ℝ | |x| > 2}

-- State the theorem
theorem complement_of_M :
  Mᶜ = {x : ℝ | -2 ≤ x ∧ x ≤ 2} :=
by sorry

end complement_of_M_l1268_126878


namespace toms_age_ratio_l1268_126864

theorem toms_age_ratio (T M : ℝ) : T > 0 → M > 0 → T - M = 3 * (T - 4 * M) → T / M = 11 / 2 := by
  sorry

end toms_age_ratio_l1268_126864


namespace david_money_left_l1268_126818

/-- Represents the money situation of a person on a trip -/
def MoneyOnTrip (initial_amount spent_amount remaining_amount : ℕ) : Prop :=
  (initial_amount = spent_amount + remaining_amount) ∧
  (remaining_amount = spent_amount - 800)

theorem david_money_left :
  ∃ (spent_amount : ℕ), MoneyOnTrip 1800 spent_amount 500 :=
sorry

end david_money_left_l1268_126818


namespace arithmetic_calculation_l1268_126803

theorem arithmetic_calculation : (28 * 9 + 18 * 19 + 8 * 29) / 14 = 59 := by
  sorry

end arithmetic_calculation_l1268_126803


namespace power_seven_mod_nine_l1268_126820

theorem power_seven_mod_nine : 7^145 % 9 = 7 := by
  sorry

end power_seven_mod_nine_l1268_126820


namespace find_a_l1268_126861

def U (a : ℝ) : Set ℝ := {2, 3, a^2 - 2*a - 3}
def A (a : ℝ) : Set ℝ := {2, |a - 7|}

theorem find_a : ∀ a : ℝ, (U a) \ (A a) = {5} → a = 4 ∨ a = -2 := by sorry

end find_a_l1268_126861


namespace marys_max_earnings_l1268_126821

/-- Mary's work schedule and pay structure --/
structure WorkSchedule where
  maxHours : Nat
  regularHours : Nat
  regularRate : ℚ
  overtimeRateIncrease : ℚ

/-- Calculate Mary's maximum weekly earnings --/
def calculateMaxEarnings (schedule : WorkSchedule) : ℚ :=
  let regularEarnings := schedule.regularRate * schedule.regularHours
  let overtimeHours := schedule.maxHours - schedule.regularHours
  let overtimeRate := schedule.regularRate * (1 + schedule.overtimeRateIncrease)
  let overtimeEarnings := overtimeRate * overtimeHours
  regularEarnings + overtimeEarnings

/-- Mary's specific work schedule --/
def marysSchedule : WorkSchedule :=
  { maxHours := 40
  , regularHours := 20
  , regularRate := 8
  , overtimeRateIncrease := 1/4 }

/-- Theorem: Mary's maximum weekly earnings are $360 --/
theorem marys_max_earnings :
  calculateMaxEarnings marysSchedule = 360 := by
  sorry

end marys_max_earnings_l1268_126821


namespace max_area_of_specific_prism_l1268_126896

/-- A prism with vertical edges parallel to the z-axis and a square cross-section -/
structure Prism where
  side_length : ℝ
  cutting_plane : ℝ → ℝ → ℝ → Prop

/-- The maximum area of the cross-section of the prism cut by a plane -/
def max_cross_section_area (p : Prism) : ℝ := sorry

/-- The theorem stating the maximum area of the cross-section for the given prism -/
theorem max_area_of_specific_prism :
  let p : Prism := {
    side_length := 12,
    cutting_plane := fun x y z ↦ 3 * x - 5 * y + 5 * z = 30
  }
  max_cross_section_area p = 360 := by sorry

end max_area_of_specific_prism_l1268_126896


namespace chocolate_candy_difference_l1268_126832

/-- The difference in cost between chocolate and candy bar -/
def cost_difference (chocolate_cost candy_cost : ℕ) : ℕ :=
  chocolate_cost - candy_cost

/-- Theorem stating the difference in cost between chocolate and candy bar -/
theorem chocolate_candy_difference :
  cost_difference 7 2 = 5 := by sorry

end chocolate_candy_difference_l1268_126832


namespace cyclist_speed_ratio_l1268_126882

theorem cyclist_speed_ratio (D : ℝ) (v_r v_w : ℝ) (t_r t_w : ℝ) : 
  D > 0 → v_r > 0 → v_w > 0 → t_r > 0 → t_w > 0 →
  (2 / 3 : ℝ) * D = v_r * t_r →
  (1 / 3 : ℝ) * D = v_w * t_w →
  t_w = 2 * t_r →
  v_r = 4 * v_w := by
  sorry

#check cyclist_speed_ratio

end cyclist_speed_ratio_l1268_126882


namespace third_number_is_seven_l1268_126893

def hcf (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem third_number_is_seven (x : ℕ) 
  (hcf_condition : hcf 136 144 x = 8)
  (lcm_condition : lcm 136 144 x = 2^4 * 3^2 * 17 * 7) :
  x = 7 := by
  sorry

end third_number_is_seven_l1268_126893


namespace joe_initial_money_l1268_126876

/-- The amount of money Joe spends on video games each month -/
def monthly_spend : ℕ := 50

/-- The amount of money Joe earns from selling games each month -/
def monthly_earn : ℕ := 30

/-- The number of months Joe can continue buying and selling games -/
def months : ℕ := 12

/-- The initial amount of money Joe has -/
def initial_money : ℕ := (monthly_spend - monthly_earn) * months

theorem joe_initial_money :
  initial_money = 240 :=
sorry

end joe_initial_money_l1268_126876


namespace not_all_zero_equiv_one_nonzero_l1268_126848

theorem not_all_zero_equiv_one_nonzero (a b c : ℝ) :
  ¬(a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) := by
  sorry

end not_all_zero_equiv_one_nonzero_l1268_126848


namespace distance_between_points_l1268_126811

/-- Two cars traveling towards each other -/
structure CarProblem where
  /-- Speed of Car A in km/h -/
  speed_a : ℝ
  /-- Speed of Car B in km/h -/
  speed_b : ℝ
  /-- Time in hours until cars meet -/
  time_to_meet : ℝ
  /-- Additional time for Car A to reach point B after meeting -/
  additional_time : ℝ

/-- The theorem stating the distance between points A and B -/
theorem distance_between_points (p : CarProblem)
  (h1 : p.speed_a = p.speed_b + 20)
  (h2 : p.time_to_meet = 4)
  (h3 : p.additional_time = 3) :
  p.speed_a * p.time_to_meet + p.speed_b * p.time_to_meet = 240 := by
  sorry

#check distance_between_points

end distance_between_points_l1268_126811


namespace m_range_l1268_126833

theorem m_range (x m : ℝ) :
  (∀ x, (1/3 < x ∧ x < 1/2) ↔ |x - m| < 1) →
  -1/2 ≤ m ∧ m ≤ 4/3 :=
by sorry

end m_range_l1268_126833


namespace moles_of_ki_formed_l1268_126804

/-- Represents the chemical reaction NH4I + KOH → NH3 + KI + H2O -/
structure ChemicalReaction where
  nh4i : ℝ  -- moles of NH4I
  koh : ℝ   -- moles of KOH
  nh3 : ℝ   -- moles of NH3
  ki : ℝ    -- moles of KI
  h2o : ℝ   -- moles of H2O

/-- The molar mass of NH4I in g/mol -/
def molar_mass_nh4i : ℝ := 144.95

/-- The total mass of NH4I in grams -/
def total_mass_nh4i : ℝ := 435

/-- Theorem stating that the number of moles of KI formed is 3 -/
theorem moles_of_ki_formed
  (reaction : ChemicalReaction)
  (h1 : reaction.koh = 3)
  (h2 : reaction.nh3 = 3)
  (h3 : reaction.h2o = 3)
  (h4 : reaction.nh4i = total_mass_nh4i / molar_mass_nh4i)
  (h5 : reaction.nh4i = reaction.koh) :
  reaction.ki = 3 := by
  sorry

end moles_of_ki_formed_l1268_126804


namespace quadratic_discriminant_l1268_126827

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of the quadratic equation 5x^2 - 2x - 7 is 144 -/
theorem quadratic_discriminant :
  discriminant 5 (-2) (-7) = 144 := by
  sorry

end quadratic_discriminant_l1268_126827


namespace min_teachers_is_16_l1268_126816

/-- Represents the number of teachers in each subject --/
structure TeacherCounts where
  maths : Nat
  physics : Nat
  chemistry : Nat

/-- Calculates the minimum number of teachers required --/
def minTeachersRequired (counts : TeacherCounts) : Nat :=
  counts.maths + counts.physics + counts.chemistry

/-- Theorem stating the minimum number of teachers required --/
theorem min_teachers_is_16 (counts : TeacherCounts) 
  (h_maths : counts.maths = 6)
  (h_physics : counts.physics = 5)
  (h_chemistry : counts.chemistry = 5) :
  minTeachersRequired counts = 16 := by
  sorry

end min_teachers_is_16_l1268_126816


namespace mall_spending_l1268_126895

def total_spent : ℚ := 347
def movie_cost : ℚ := 24
def num_movies : ℕ := 3
def bean_cost : ℚ := 1.25
def num_bean_bags : ℕ := 20

theorem mall_spending (mall_spent : ℚ) : 
  mall_spent = total_spent - (↑num_movies * movie_cost + ↑num_bean_bags * bean_cost) → 
  mall_spent = 250 := by
  sorry

end mall_spending_l1268_126895


namespace twentieth_term_of_sequence_l1268_126867

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem twentieth_term_of_sequence : arithmetic_sequence 3 5 20 = 98 := by
  sorry

end twentieth_term_of_sequence_l1268_126867


namespace chess_game_draw_probability_l1268_126891

/-- Given a chess game between A and B:
    * The game can end in A winning, B winning, or a draw.
    * The probability of A not losing is 0.6.
    * The probability of B not losing is 0.7.
    This theorem proves that the probability of the game ending in a draw is 0.3. -/
theorem chess_game_draw_probability :
  ∀ (p_a_win p_b_win p_draw : ℝ),
    p_a_win + p_b_win + p_draw = 1 →
    p_a_win + p_draw = 0.6 →
    p_b_win + p_draw = 0.7 →
    p_draw = 0.3 :=
by sorry

end chess_game_draw_probability_l1268_126891


namespace reflect_M_across_y_axis_l1268_126806

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The theorem stating that reflecting M(3,2) across the y-axis results in (-3,2) -/
theorem reflect_M_across_y_axis :
  let M : Point := { x := 3, y := 2 }
  reflectAcrossYAxis M = { x := -3, y := 2 } := by
  sorry

end reflect_M_across_y_axis_l1268_126806


namespace solutions_satisfy_system_l1268_126843

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  26 * x^2 + 42 * x * y + 17 * y^2 = 10 ∧
  10 * x^2 + 18 * x * y + 8 * y^2 = 6

/-- The solutions to the system of equations -/
def solutions : List (ℝ × ℝ) :=
  [(-1, 2), (-11, 14), (11, -14), (1, -2)]

/-- Theorem stating that the given points are solutions to the system -/
theorem solutions_satisfy_system :
  ∀ (p : ℝ × ℝ), p ∈ solutions → system p.1 p.2 := by
  sorry

end solutions_satisfy_system_l1268_126843


namespace max_crates_third_trip_l1268_126887

/-- Given a trailer with a maximum weight capacity and a minimum weight per crate,
    prove the maximum number of crates for the third trip. -/
theorem max_crates_third_trip
  (max_weight : ℕ)
  (min_crate_weight : ℕ)
  (trip1_crates : ℕ)
  (trip2_crates : ℕ)
  (h_max_weight : max_weight = 750)
  (h_min_crate_weight : min_crate_weight = 150)
  (h_trip1 : trip1_crates = 3)
  (h_trip2 : trip2_crates = 4)
  (h_weight_constraint : ∀ n : ℕ, n * min_crate_weight ≤ max_weight → n ≤ trip1_crates ∨ n ≤ trip2_crates ∨ n ≤ max_weight / min_crate_weight) :
  (max_weight / min_crate_weight : ℕ) = 5 :=
sorry

end max_crates_third_trip_l1268_126887


namespace intersection_of_A_and_B_l1268_126854

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by
  sorry

end intersection_of_A_and_B_l1268_126854


namespace sqrt_two_irrational_others_rational_l1268_126824

theorem sqrt_two_irrational_others_rational : 
  (∃ (q : ℚ), Real.sqrt 2 = (q : ℝ)) ∧ 
  (∃ (q : ℚ), (1 : ℝ) = (q : ℝ)) ∧ 
  (∃ (q : ℚ), (0 : ℝ) = (q : ℝ)) ∧ 
  (∃ (q : ℚ), (-1 : ℝ) = (q : ℝ)) →
  False :=
by sorry

end sqrt_two_irrational_others_rational_l1268_126824


namespace merchant_profit_theorem_l1268_126802

/-- Calculates the profit percentage given the ratio of articles sold to articles bought --/
def profit_percentage (articles_sold : ℕ) (articles_bought : ℕ) : ℚ :=
  ((articles_bought : ℚ) / (articles_sold : ℚ) - 1) * 100

/-- Proves that when 25 articles' cost price equals 18 articles' selling price, the profit is (7/18) * 100 percent --/
theorem merchant_profit_theorem (cost_price selling_price : ℚ) 
  (h : 25 * cost_price = 18 * selling_price) : 
  profit_percentage 18 25 = (7 / 18) * 100 := by
  sorry

#eval profit_percentage 18 25

end merchant_profit_theorem_l1268_126802


namespace area_under_curve_l1268_126874

-- Define the curve
def f (x : ℝ) := x^2

-- Define the boundaries
def lower_bound : ℝ := 0
def upper_bound : ℝ := 1

-- State the theorem
theorem area_under_curve :
  (∫ x in lower_bound..upper_bound, f x) = (1 : ℝ) / 3 := by
  sorry

end area_under_curve_l1268_126874


namespace complex_fourth_quadrant_l1268_126897

theorem complex_fourth_quadrant (m : ℝ) :
  (∃ z : ℂ, z = (m + Complex.I) / (1 + Complex.I) ∧ 
   z.re > 0 ∧ z.im < 0) ↔ m > 1 := by
  sorry

end complex_fourth_quadrant_l1268_126897


namespace book_price_increase_l1268_126837

theorem book_price_increase (final_price : ℝ) (increase_percentage : ℝ) 
  (h1 : final_price = 360)
  (h2 : increase_percentage = 20) :
  let original_price := final_price / (1 + increase_percentage / 100)
  original_price = 300 := by
sorry

end book_price_increase_l1268_126837


namespace problem_solution_l1268_126884

/-- The function f(x) = x^2 - 2ax + 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

theorem problem_solution (a : ℝ) (h : a > 1) :
  /- Part 1 -/
  (∀ x, x ∈ Set.Icc 1 a ↔ f a x ∈ Set.Icc 1 a) →
  a = 2
  ∧
  /- Part 2 -/
  ((∀ x ≤ 2, ∀ y ≤ 2, x < y → f a x > f a y) ∧
   (∀ x ∈ Set.Icc 1 2, f a x ≤ 0)) →
  a ≥ 3 :=
by sorry

end problem_solution_l1268_126884


namespace hexagon_side_length_l1268_126890

/-- A hexagon is a polygon with 6 sides -/
def Hexagon : ℕ := 6

/-- Given a hexagon with perimeter 42 inches, prove that each side length is 7 inches -/
theorem hexagon_side_length (perimeter : ℝ) (h1 : perimeter = 42) :
  perimeter / Hexagon = 7 := by
  sorry

end hexagon_side_length_l1268_126890


namespace optimal_price_and_profit_l1268_126849

/-- Represents the daily sales volume as a function of price -/
def sales_volume (x : ℝ) : ℝ := -20 * x + 1600

/-- Represents the daily profit as a function of price -/
def daily_profit (x : ℝ) : ℝ := (x - 40) * (sales_volume x)

/-- The selling price must not be less than 45 yuan -/
def min_price : ℝ := 45

/-- Theorem stating the optimal price and maximum profit -/
theorem optimal_price_and_profit :
  ∃ (x : ℝ), x ≥ min_price ∧
  (∀ y : ℝ, y ≥ min_price → daily_profit y ≤ daily_profit x) ∧
  x = 60 ∧ daily_profit x = 8000 := by
  sorry

#check optimal_price_and_profit

end optimal_price_and_profit_l1268_126849


namespace inscribed_triangle_area_l1268_126865

theorem inscribed_triangle_area (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  (∃ (r : ℝ), r = 4 ∧ ∃ (A B C : ℝ), 
    a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧ 
    c / Real.sin C = 2 * r) →
  a * b * c = 16 * Real.sqrt 2 →
  (1 / 2) * a * b * Real.sin (Real.arcsin (c / 8)) = Real.sqrt 2 := by
sorry

end inscribed_triangle_area_l1268_126865


namespace nicholas_crackers_l1268_126815

theorem nicholas_crackers (marcus_crackers mona_crackers nicholas_crackers : ℕ) : 
  marcus_crackers = 3 * mona_crackers →
  nicholas_crackers = mona_crackers + 6 →
  marcus_crackers = 27 →
  nicholas_crackers = 15 := by sorry

end nicholas_crackers_l1268_126815


namespace well_depth_calculation_l1268_126875

/-- The depth of the well in feet -/
def well_depth : ℝ := 1255.64

/-- The time it takes for the stone to hit the bottom and the sound to reach the top -/
def total_time : ℝ := 10

/-- The gravitational constant for the stone's fall -/
def gravity_constant : ℝ := 16

/-- The velocity of sound in feet per second -/
def sound_velocity : ℝ := 1100

/-- The function describing the stone's fall distance after t seconds -/
def stone_fall (t : ℝ) : ℝ := gravity_constant * t^2

/-- Theorem stating that the calculated well depth is correct given the conditions -/
theorem well_depth_calculation :
  ∃ (t_fall : ℝ), 
    t_fall > 0 ∧ 
    t_fall + (well_depth / sound_velocity) = total_time ∧ 
    stone_fall t_fall = well_depth := by
  sorry

end well_depth_calculation_l1268_126875


namespace sticker_distribution_l1268_126855

/-- The number of ways to distribute n identical objects into k distinct groups,
    where each group must have at least one object -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- The number of stickers -/
def num_stickers : ℕ := 10

/-- The number of sheets of paper -/
def num_sheets : ℕ := 5

theorem sticker_distribution :
  distribute num_stickers num_sheets = 126 := by
  sorry

end sticker_distribution_l1268_126855


namespace scout_camp_chocolate_cost_l1268_126829

/-- The cost of chocolate bars for a scout camp out --/
def chocolate_cost (bar_cost : ℚ) (sections_per_bar : ℕ) (num_scouts : ℕ) (smores_per_scout : ℕ) : ℚ :=
  let total_smores := num_scouts * smores_per_scout
  let bars_needed := (total_smores + sections_per_bar - 1) / sections_per_bar
  bars_needed * bar_cost

/-- Theorem: The cost of chocolate bars for the given scout camp out is $15.00 --/
theorem scout_camp_chocolate_cost :
  chocolate_cost (3/2) 3 15 2 = 15 := by
  sorry

end scout_camp_chocolate_cost_l1268_126829


namespace same_terminal_side_angle_l1268_126889

theorem same_terminal_side_angle : ∃ θ : ℝ, 
  0 ≤ θ ∧ θ < 2*π ∧ 
  ∃ k : ℤ, θ = 2*k*π + (-4*π/3) ∧
  θ = 2*π/3 := by
  sorry

end same_terminal_side_angle_l1268_126889


namespace total_questions_is_60_l1268_126839

-- Define the scoring system
def correct_score : ℕ := 4
def incorrect_score : ℕ := 1

-- Define the given information
def total_score : ℕ := 140
def correct_answers : ℕ := 40

-- Define the total number of questions attempted
def total_questions : ℕ := correct_answers + (correct_score * correct_answers - total_score)

-- Theorem to prove
theorem total_questions_is_60 : total_questions = 60 := by
  sorry

end total_questions_is_60_l1268_126839


namespace girth_bound_l1268_126847

/-- The minimum degree of a graph G -/
def min_degree (G : Type*) : ℕ := sorry

/-- The girth of a graph G -/
def girth (G : Type*) : ℕ := sorry

/-- The number of vertices in a graph G -/
def num_vertices (G : Type*) : ℕ := sorry

/-- Theorem: For any graph G with minimum degree ≥ 3, the girth is less than 2 log |G| -/
theorem girth_bound (G : Type*) (h : min_degree G ≥ 3) : 
  girth G < 2 * Real.log (num_vertices G) := by
  sorry

end girth_bound_l1268_126847


namespace longest_segment_in_cylinder_cylinder_surface_area_l1268_126883

-- Define the cylinder
def cylinder_radius : ℝ := 5
def cylinder_height : ℝ := 10

-- Theorem for the longest segment
theorem longest_segment_in_cylinder :
  Real.sqrt ((2 * cylinder_radius) ^ 2 + cylinder_height ^ 2) = 10 * Real.sqrt 2 := by sorry

-- Theorem for the total surface area
theorem cylinder_surface_area :
  2 * Real.pi * cylinder_radius * (cylinder_height + cylinder_radius) = 150 * Real.pi := by sorry

end longest_segment_in_cylinder_cylinder_surface_area_l1268_126883


namespace perry_phil_difference_l1268_126870

/-- The number of games won by each player -/
structure GolfWins where
  perry : ℕ
  dana : ℕ
  charlie : ℕ
  phil : ℕ

/-- The conditions of the golf game -/
def golf_game (g : GolfWins) : Prop :=
  g.perry = g.dana + 5 ∧
  g.charlie = g.dana - 2 ∧
  g.phil = g.charlie + 3 ∧
  g.phil = 12

theorem perry_phil_difference (g : GolfWins) (h : golf_game g) : 
  g.perry - g.phil = 4 := by
  sorry

#check perry_phil_difference

end perry_phil_difference_l1268_126870


namespace reciprocal_roots_quadratic_l1268_126853

theorem reciprocal_roots_quadratic (k : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ r₁ * r₂ = 1 ∧ 
    (∀ x : ℝ, 5.2 * x * x + 14.3 * x + k = 0 ↔ (x = r₁ ∨ x = r₂))) → 
  k = 5.2 := by
sorry


end reciprocal_roots_quadratic_l1268_126853


namespace factor_of_polynomial_l1268_126866

theorem factor_of_polynomial (x : ℝ) : 
  (x - 1/2) ∣ (8*x^3 + 17*x^2 + 2*x - 3) := by
  sorry

end factor_of_polynomial_l1268_126866


namespace quadratic_radicals_same_type_l1268_126851

-- Define the two quadratic expressions
def f (a : ℝ) : ℝ := 3 * a - 8
def g (a : ℝ) : ℝ := 17 - 2 * a

-- Theorem statement
theorem quadratic_radicals_same_type :
  ∃ (a : ℝ), a = 5 ∧ f a = g a :=
sorry

end quadratic_radicals_same_type_l1268_126851


namespace prob_A_wins_3_1_l1268_126856

/-- The probability of Team A winning a single game -/
def prob_A_win : ℚ := 1/2

/-- The probability of Team B winning a single game -/
def prob_B_win : ℚ := 1/2

/-- The number of games in a best-of-five series where one team wins 3-1 -/
def games_played : ℕ := 4

/-- The number of ways to arrange 3 wins in 4 games -/
def winning_scenarios : ℕ := 3

/-- The probability of Team A winning with a score of 3:1 in a best-of-five series -/
theorem prob_A_wins_3_1 : 
  (prob_A_win ^ 3 * prob_B_win) * winning_scenarios = 3/16 := by
  sorry

end prob_A_wins_3_1_l1268_126856


namespace intersection_of_A_and_B_l1268_126830

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | (x + 1) * (x - 4) > 0}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by sorry

end intersection_of_A_and_B_l1268_126830


namespace parallel_vector_m_values_l1268_126886

def vector_a (m : ℝ) : Fin 2 → ℝ := λ i => if i = 0 then 2 else m

theorem parallel_vector_m_values (m : ℝ) :
  (∃ b : Fin 2 → ℝ, b ≠ 0 ∧ ∃ k : ℝ, k ≠ 0 ∧ vector_a m = λ i => k * b i) →
  m = 2 ∨ m = -2 := by
  sorry

end parallel_vector_m_values_l1268_126886


namespace simplify_trig_expression_l1268_126880

theorem simplify_trig_expression (x : ℝ) :
  Real.sqrt 2 * Real.cos x - Real.sqrt 6 * Real.sin x = 2 * Real.sqrt 2 * Real.cos (π / 3 + x) :=
by sorry

end simplify_trig_expression_l1268_126880


namespace not_p_and_q_l1268_126812

-- Define proposition p
def p : Prop := ∀ (a b c : ℝ), a < b → a * c^2 < b * c^2

-- Define proposition q
def q : Prop := ∃ (x₀ : ℝ), x₀ > 0 ∧ x₀ - 1 - Real.log x₀ = 0

-- Theorem statement
theorem not_p_and_q : (¬p) ∧ q := by sorry

end not_p_and_q_l1268_126812


namespace range_of_m_plus_n_l1268_126871

noncomputable def f (m n x : ℝ) : ℝ := 2^x * m + x^2 + n * x

theorem range_of_m_plus_n (m n : ℝ) :
  (∃ x, f m n x = 0) ∧ 
  (∀ x, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
by sorry

end range_of_m_plus_n_l1268_126871


namespace triangle_area_implies_ab_value_l1268_126869

theorem triangle_area_implies_ab_value (a b : ℝ) : 
  a > 0 → b > 0 → 
  (1/2 * (12/a) * (12/b) = 9) → 
  a * b = 8 := by
sorry

end triangle_area_implies_ab_value_l1268_126869


namespace trig_range_equivalence_l1268_126808

theorem trig_range_equivalence (α : Real) :
  (0 < α ∧ α < 2 * Real.pi) →
  (Real.sin α < Real.sqrt 3 / 2 ∧ Real.cos α > 1 / 2) ↔
  ((0 < α ∧ α < Real.pi / 3) ∨ (5 * Real.pi / 3 < α ∧ α < 2 * Real.pi)) :=
by sorry

end trig_range_equivalence_l1268_126808


namespace cherry_bag_cost_l1268_126892

/-- The cost of a four-pound bag of cherries -/
def cherry_cost : ℝ := 13.5

/-- The cost of the pie crust ingredients -/
def crust_cost : ℝ := 4.5

/-- The total cost of the cheapest pie -/
def cheapest_pie_cost : ℝ := 18

/-- The cost of the blueberry pie -/
def blueberry_pie_cost : ℝ := 18

theorem cherry_bag_cost : 
  cherry_cost = cheapest_pie_cost - crust_cost ∧ 
  blueberry_pie_cost = cheapest_pie_cost :=
by sorry

end cherry_bag_cost_l1268_126892


namespace decimal_to_fraction_sum_l1268_126814

theorem decimal_to_fraction_sum (a b : ℕ+) :
  (a : ℚ) / (b : ℚ) = 324375 / 1000000 ∧
  (∀ (c d : ℕ+), (c : ℚ) / (d : ℚ) = 324375 / 1000000 → a ≤ c) →
  (a : ℕ) + (b : ℕ) = 2119 := by
sorry

end decimal_to_fraction_sum_l1268_126814


namespace min_value_and_fraction_sum_l1268_126801

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |2*x - 1|

-- Theorem statement
theorem min_value_and_fraction_sum :
  (∃ a : ℝ, (∀ x : ℝ, f x ≥ a) ∧ (∃ x₀ : ℝ, f x₀ = a) ∧ a = 3/2) ∧
  (∀ m n : ℝ, m > 0 → n > 0 → m + n = 3/2 → 1/m + 4/n ≥ 6) ∧
  (∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ m₀ + n₀ = 3/2 ∧ 1/m₀ + 4/n₀ = 6) :=
by sorry

end min_value_and_fraction_sum_l1268_126801


namespace largest_three_digit_congruence_l1268_126813

theorem largest_three_digit_congruence :
  ∃ m : ℕ,
    100 ≤ m ∧ m ≤ 999 ∧
    40 * m ≡ 120 [MOD 200] ∧
    ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 40 * n ≡ 120 [MOD 200] → n ≤ m ∧
    m = 998 :=
by sorry

end largest_three_digit_congruence_l1268_126813


namespace triangle_area_approx_l1268_126860

/-- The area of a triangle with sides 35 cm, 23 cm, and 41 cm is approximately 402.65 cm² --/
theorem triangle_area_approx (a b c : ℝ) (ha : a = 35) (hb : b = 23) (hc : c = 41) :
  ∃ (area : ℝ), abs (area - ((a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2)) / 2) - 402.65) < 0.01 := by
  sorry

end triangle_area_approx_l1268_126860


namespace qt_length_l1268_126800

/-- Square with side length 4 and special points T and U -/
structure SpecialSquare where
  -- Square PQRS with side length 4
  side : ℝ
  side_eq : side = 4

  -- Point T on side PQ
  t : ℝ × ℝ
  t_on_pq : t.1 ≥ 0 ∧ t.1 ≤ side ∧ t.2 = 0

  -- Point U on side PS
  u : ℝ × ℝ
  u_on_ps : u.1 = 0 ∧ u.2 ≥ 0 ∧ u.2 ≤ side

  -- Lines QT and SU divide the square into four equal areas
  equal_areas : (side * t.1) / 2 = (side * side) / 4

/-- The length of QT in a SpecialSquare is 2√3 -/
theorem qt_length (sq : SpecialSquare) : 
  Real.sqrt ((sq.side - sq.t.1)^2 + sq.t.1^2) = 2 * Real.sqrt 3 := by
  sorry

end qt_length_l1268_126800


namespace combustion_reaction_result_l1268_126810

-- Define the thermochemical equations
def nitrobenzene_combustion (x : ℝ) : ℝ := 3094.88 * x
def aniline_combustion (y : ℝ) : ℝ := 3392.15 * y
def ethanol_combustion (z : ℝ) : ℝ := 1370 * z

-- Define the relationship between x and y based on nitrogen production
def nitrogen_production (x y : ℝ) : Prop := 0.5 * x + 0.5 * y = 0.15

-- Define the total heat released
def total_heat_released (x y z : ℝ) : Prop :=
  nitrobenzene_combustion x + aniline_combustion y + ethanol_combustion z = 1467.4

-- Define the mass of the solution
def solution_mass (x : ℝ) : ℝ := 470 * x

-- Define the theorem
theorem combustion_reaction_result :
  ∃ (x y z : ℝ),
    nitrogen_production x y ∧
    total_heat_released x y z ∧
    x = 0.1 ∧
    solution_mass x = 47 := by
  sorry

end combustion_reaction_result_l1268_126810


namespace largest_three_digit_with_seven_hundreds_l1268_126828

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_seven_in_hundreds_place (n : ℕ) : Prop := (n / 100) % 10 = 7

theorem largest_three_digit_with_seven_hundreds : 
  ∀ n : ℕ, is_three_digit n → has_seven_in_hundreds_place n → n ≤ 799 :=
by sorry

end largest_three_digit_with_seven_hundreds_l1268_126828


namespace lemming_average_distance_l1268_126823

/-- The average distance from a point to the sides of a square --/
theorem lemming_average_distance (side_length : ℝ) (diagonal_distance : ℝ) (turn_angle : ℝ) (final_distance : ℝ) : 
  side_length = 12 →
  diagonal_distance = 7.8 →
  turn_angle = 60 * π / 180 →
  final_distance = 3 →
  let d := (diagonal_distance / (side_length * Real.sqrt 2))
  let x := d * side_length + final_distance * Real.cos (π/2 - turn_angle)
  let y := d * side_length + final_distance * Real.sin (π/2 - turn_angle)
  (x + y + (side_length - x) + (side_length - y)) / 4 = 6 := by
sorry

end lemming_average_distance_l1268_126823


namespace cubic_polynomials_inequality_l1268_126822

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of a cubic polynomial -/
def roots (p : CubicPolynomial) : Finset ℝ := sorry

/-- Check if all roots of a polynomial are positive -/
def all_roots_positive (p : CubicPolynomial) : Prop :=
  ∀ r ∈ roots p, r > 0

/-- Given two cubic polynomials, check if the roots of one are reciprocals of the other -/
def roots_are_reciprocals (p q : CubicPolynomial) : Prop :=
  ∀ r ∈ roots p, (1 / r) ∈ roots q

theorem cubic_polynomials_inequality (p q : CubicPolynomial) 
  (h_positive : all_roots_positive p)
  (h_reciprocal : roots_are_reciprocals p q) :
  p.a * q.a > 9 ∧ p.b * q.b > 9 := by
  sorry

end cubic_polynomials_inequality_l1268_126822


namespace lorelai_jellybeans_l1268_126868

def jellybeans_problem (gigi rory luke lane lorelai : ℕ) : Prop :=
  gigi = 15 ∧
  rory = gigi + 30 ∧
  luke = 2 * rory ∧
  lane = gigi + 10 ∧
  lorelai = 3 * (gigi + luke + lane)

theorem lorelai_jellybeans :
  ∀ gigi rory luke lane lorelai : ℕ,
  jellybeans_problem gigi rory luke lane lorelai →
  lorelai = 390 :=
by
  sorry

end lorelai_jellybeans_l1268_126868


namespace payment_combinations_eq_six_l1268_126846

/-- Represents the number of ways to make a payment of 230 yuan using given bills -/
def payment_combinations : ℕ :=
  (Finset.filter (fun (x, y, z) => 
    50 * x + 20 * y + 10 * z = 230 ∧ 
    x ≤ 5 ∧ y ≤ 6 ∧ z ≤ 7)
    (Finset.product (Finset.range 6) (Finset.product (Finset.range 7) (Finset.range 8)))).card

/-- The theorem stating that there are exactly 6 ways to make the payment -/
theorem payment_combinations_eq_six : payment_combinations = 6 := by
  sorry

end payment_combinations_eq_six_l1268_126846


namespace folded_polyhedron_volume_l1268_126805

/-- Represents a polyhedron formed by folding four squares and two equilateral triangles -/
structure FoldedPolyhedron where
  square_side_length : ℝ
  triangle_side_length : ℝ
  h_square_side : square_side_length = 2
  h_triangle_side : triangle_side_length = Real.sqrt 8

/-- Calculates the volume of the folded polyhedron -/
noncomputable def volume (p : FoldedPolyhedron) : ℝ :=
  (16 * Real.sqrt 2 / 3) * Real.sqrt (2 - Real.sqrt 2)

/-- Theorem stating that the volume of the folded polyhedron is correct -/
theorem folded_polyhedron_volume (p : FoldedPolyhedron) :
  volume p = (16 * Real.sqrt 2 / 3) * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

end folded_polyhedron_volume_l1268_126805


namespace polynomial_division_remainder_l1268_126872

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  2 * X^4 + 9 * X^3 - 38 * X^2 - 50 * X + 35 = 
  (X^2 + 5 * X - 6) * q + (61 * X - 91) := by sorry

end polynomial_division_remainder_l1268_126872


namespace j20_most_suitable_for_census_l1268_126885

/-- Represents a survey option -/
inductive SurveyOption
  | HuaweiPhoneBattery
  | J20Components
  | SpringFestivalMovie
  | HomeworkTime

/-- Determines if a survey option is suitable for a comprehensive survey (census) -/
def isSuitableForCensus (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.HuaweiPhoneBattery => False
  | SurveyOption.J20Components => True
  | SurveyOption.SpringFestivalMovie => False
  | SurveyOption.HomeworkTime => False

/-- Theorem stating that the J20Components survey is the most suitable for a comprehensive survey -/
theorem j20_most_suitable_for_census :
  ∀ (option : SurveyOption),
    option ≠ SurveyOption.J20Components →
    ¬(isSuitableForCensus option) ∧ isSuitableForCensus SurveyOption.J20Components :=
by sorry

end j20_most_suitable_for_census_l1268_126885


namespace simplify_fraction_l1268_126836

theorem simplify_fraction (b : ℝ) (h : b = 5) : 15 * b^4 / (75 * b^3) = 1 := by
  sorry

end simplify_fraction_l1268_126836


namespace cos_symmetry_center_l1268_126888

theorem cos_symmetry_center (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.cos (2 * x + π / 3)
  let center : ℝ × ℝ := (π / 12, 0)
  ∀ t : ℝ, f (center.1 + t) = f (center.1 - t) :=
by sorry

end cos_symmetry_center_l1268_126888


namespace intersection_point_solution_l1268_126899

-- Define the lines
def line1 (x y : ℝ) : Prop := y = -x + 4
def line2 (x y m : ℝ) : Prop := y = 2*x + m

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x + y - 4 = 0
def equation2 (x y m : ℝ) : Prop := 2*x - y + m = 0

-- Theorem statement
theorem intersection_point_solution (m n : ℝ) :
  (line1 3 n ∧ line2 3 n m) →
  (equation1 3 1 ∧ equation2 3 1 m) :=
by sorry

end intersection_point_solution_l1268_126899


namespace five_hour_charge_l1268_126819

/-- Represents the pricing structure and total charge calculation for a psychologist's therapy sessions. -/
structure TherapyPricing where
  /-- The charge for the first hour of therapy -/
  first_hour : ℕ
  /-- The charge for each additional hour of therapy -/
  additional_hour : ℕ
  /-- The first hour costs $30 more than each additional hour -/
  first_hour_premium : first_hour = additional_hour + 30
  /-- The total charge for 3 hours of therapy is $252 -/
  three_hour_charge : first_hour + 2 * additional_hour = 252

/-- Theorem stating that given the pricing structure, the total charge for 5 hours of therapy is $400 -/
theorem five_hour_charge (p : TherapyPricing) : p.first_hour + 4 * p.additional_hour = 400 := by
  sorry

end five_hour_charge_l1268_126819


namespace triangle_inequality_theorem_l1268_126825

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (triangle_inequality : a < b + c ∧ b < c + a ∧ c < a + b)

-- Define the angle bisector points
structure AngleBisectorPoints (t : Triangle) :=
  (A₁ B₁ C₁ : ℝ × ℝ)

-- Define the property of points being concyclic
def are_concyclic (p₁ p₂ p₃ p₄ : ℝ × ℝ) : Prop := sorry

-- Define the theorem
theorem triangle_inequality_theorem (t : Triangle) (abp : AngleBisectorPoints t) :
  are_concyclic abp.A₁ abp.B₁ abp.C₁ (0, t.b) →
  (t.a / (t.b + t.c)) + (t.b / (t.c + t.a)) + (t.c / (t.a + t.b)) ≥ (Real.sqrt 17 - 1) / 2 :=
by sorry

end triangle_inequality_theorem_l1268_126825


namespace water_channel_length_l1268_126831

theorem water_channel_length : ∀ L : ℝ,
  L > 0 →
  (3/4 * L - 5/28 * L) = 4/7 * L →
  (4/7 * L - 2/7 * L) = 2/7 * L →
  2/7 * L = 100 →
  L = 350 := by
sorry

end water_channel_length_l1268_126831


namespace sunday_only_papers_l1268_126881

/-- The number of papers Kyle delivers in a week -/
def total_papers : ℕ := 720

/-- The number of houses Kyle delivers to from Monday to Saturday -/
def regular_houses : ℕ := 100

/-- The number of regular customers who don't receive the Sunday paper -/
def sunday_opt_out : ℕ := 10

/-- The number of days Kyle delivers from Monday to Saturday -/
def weekdays : ℕ := 6

theorem sunday_only_papers : 
  total_papers - (regular_houses * weekdays) - (regular_houses - sunday_opt_out) = 30 := by
  sorry

end sunday_only_papers_l1268_126881


namespace g_neg_two_l1268_126834

def g (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

theorem g_neg_two : g (-2) = -16 := by
  sorry

end g_neg_two_l1268_126834


namespace no_increasing_function_with_properties_l1268_126809

theorem no_increasing_function_with_properties :
  ¬ ∃ (f : ℕ → ℕ),
    (∀ (a b : ℕ), a < b → f a < f b) ∧
    (f 2 = 2) ∧
    (∀ (n m : ℕ), f (n * m) = f n + f m) := by
  sorry

end no_increasing_function_with_properties_l1268_126809


namespace inequality_proof_l1268_126894

theorem inequality_proof (a b t : ℝ) (ha : a > 1) (hb : b > 1) (ht : t > 0) :
  (a^2 / (b^t - 1)) + (b^(2*t) / (a^t - 1)) ≥ 8 := by
  sorry

end inequality_proof_l1268_126894


namespace max_sector_area_l1268_126842

/-- Sector represents a circular sector with radius and central angle -/
structure Sector where
  radius : ℝ
  angle : ℝ

/-- The perimeter of a sector -/
def sectorPerimeter (s : Sector) : ℝ := s.radius * s.angle + 2 * s.radius

/-- The area of a sector -/
def sectorArea (s : Sector) : ℝ := 0.5 * s.radius^2 * s.angle

/-- Theorem: Maximum area of a sector with perimeter 40 -/
theorem max_sector_area (s : Sector) (h : sectorPerimeter s = 40) :
  sectorArea s ≤ 100 ∧ (sectorArea s = 100 ↔ s.angle = 2) := by sorry

end max_sector_area_l1268_126842


namespace max_sum_of_cubes_l1268_126826

/-- Given real numbers a, b, c, d satisfying the condition,
    the sum of their cubes is bounded above by 4√10 -/
theorem max_sum_of_cubes (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d = 10) : 
  a^3 + b^3 + c^3 + d^3 ≤ 4 * Real.sqrt 10 := by
  sorry

end max_sum_of_cubes_l1268_126826


namespace rectangle_ratio_is_2_sqrt_6_l1268_126879

/-- Represents a hexagon with side length s -/
structure Hexagon where
  s : ℝ
  h_positive : s > 0

/-- Represents a rectangle with sides x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ
  h_positive_x : x > 0
  h_positive_y : y > 0

/-- The arrangement of rectangles around hexagons -/
structure HexagonArrangement where
  inner : Hexagon
  outer : Hexagon
  rectangle : Rectangle
  h_area_ratio : outer.s^2 = 6 * inner.s^2
  h_outer_perimeter : 6 * rectangle.x = 6 * outer.s
  h_inner_side : rectangle.y = inner.s / 2

theorem rectangle_ratio_is_2_sqrt_6 (arr : HexagonArrangement) :
  arr.rectangle.x / arr.rectangle.y = 2 * Real.sqrt 6 := by
  sorry

end rectangle_ratio_is_2_sqrt_6_l1268_126879


namespace total_books_read_is_36sc_l1268_126863

/-- The number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  let books_per_month : ℕ := 3
  let months_per_year : ℕ := 12
  let books_per_student_per_year : ℕ := books_per_month * months_per_year
  let total_students : ℕ := c * s
  books_per_student_per_year * total_students

/-- Theorem stating that the total number of books read is 36 * s * c -/
theorem total_books_read_is_36sc (c s : ℕ) :
  total_books_read c s = 36 * s * c := by
  sorry

end total_books_read_is_36sc_l1268_126863


namespace brad_probability_l1268_126873

/-- Represents the outcome of answering a math problem -/
inductive Answer
| correct
| incorrect

/-- Represents a sequence of answers to math problems -/
def AnswerSequence := List Answer

/-- Calculates the probability of a specific answer sequence -/
def probability (seq : AnswerSequence) : Real :=
  sorry

/-- Counts the number of correct answers in a sequence -/
def countCorrect (seq : AnswerSequence) : Nat :=
  sorry

/-- Generates all possible answer sequences for the remaining 8 problems -/
def generateSequences : List AnswerSequence :=
  sorry

theorem brad_probability :
  let allSequences := generateSequences
  let validSequences := allSequences.filter (λ seq => countCorrect (Answer.correct :: Answer.incorrect :: seq) = 5)
  (validSequences.map probability).sum = 1 / 9 := by
  sorry

end brad_probability_l1268_126873


namespace jesse_mall_trip_l1268_126845

def mall_trip (initial_amount novel_cost : ℕ) : ℕ :=
  let lunch_cost := 2 * novel_cost
  let total_spent := novel_cost + lunch_cost
  initial_amount - total_spent

theorem jesse_mall_trip :
  mall_trip 50 7 = 29 := by
  sorry

end jesse_mall_trip_l1268_126845


namespace f_symmetry_l1268_126817

/-- Given a function f(x) = x^2005 + ax^3 - b/x - 8, where a and b are real constants,
    if f(-2) = 10, then f(2) = -26 -/
theorem f_symmetry (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2005 + a*x^3 - b/x - 8
  f (-2) = 10 → f 2 = -26 := by
sorry

end f_symmetry_l1268_126817


namespace ana_guarantee_l1268_126852

/-- The hat game setup -/
structure HatGame where
  n : ℕ
  h_n_gt_1 : n > 1

/-- The minimum number of correct guesses Ana can guarantee -/
def min_correct_guesses (game : HatGame) : ℕ :=
  (game.n - 1) / 2

/-- The theorem stating Ana's guarantee -/
theorem ana_guarantee (game : HatGame) :
  ∃ (strategy : Type),
    ∀ (bob_distribution : Type),
      ∃ (correct_guesses : ℕ),
        correct_guesses ≥ min_correct_guesses game :=
sorry


end ana_guarantee_l1268_126852


namespace max_imaginary_part_of_roots_l1268_126858

def polynomial (z : ℂ) : ℂ := z^9 + z^7 - z^5 + z^3 - z

def is_root (z : ℂ) : Prop := polynomial z = 0

def imaginary_part (z : ℂ) : ℝ := z.im

theorem max_imaginary_part_of_roots :
  ∃ (θ : ℝ), 
    -π/2 ≤ θ ∧ θ ≤ π/2 ∧
    (∀ (z : ℂ), is_root z → imaginary_part z ≤ Real.sin θ) ∧
    θ = π/2 :=
sorry

end max_imaginary_part_of_roots_l1268_126858


namespace range_of_expression_l1268_126807

theorem range_of_expression (x y : ℝ) (h1 : x + 2*y - 6 = 0) (h2 : 0 < x) (h3 : x < 3) :
  1 < (x + 2) / (y - 1) ∧ (x + 2) / (y - 1) < 10 := by
  sorry

end range_of_expression_l1268_126807


namespace green_peaches_count_l1268_126844

/-- The number of green peaches in a basket, given the number of red, yellow, and total green and yellow peaches. -/
def num_green_peaches (red : ℕ) (yellow : ℕ) (green_and_yellow : ℕ) : ℕ :=
  green_and_yellow - yellow

/-- Theorem stating that there are 6 green peaches in the basket. -/
theorem green_peaches_count :
  let red : ℕ := 5
  let yellow : ℕ := 14
  let green_and_yellow : ℕ := 20
  num_green_peaches red yellow green_and_yellow = 6 := by
  sorry

end green_peaches_count_l1268_126844


namespace quadratic_transformation_l1268_126841

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := x^2

-- Define the transformation
def transform (f : ℝ → ℝ) (horizontal_shift : ℝ) (vertical_shift : ℝ) : ℝ → ℝ :=
  λ x => f (x - horizontal_shift) + vertical_shift

-- Define the new function after transformation
def new_function : ℝ → ℝ := transform original_function 3 3

-- Theorem stating the equivalence
theorem quadratic_transformation :
  ∀ x : ℝ, new_function x = (x - 3)^2 + 3 :=
by
  sorry

end quadratic_transformation_l1268_126841


namespace sphere_surface_area_l1268_126862

/-- The surface area of a sphere, given specific conditions for a hemisphere --/
theorem sphere_surface_area (r : ℝ) (h1 : π * r^2 = 3) (h2 : 3 * π * r^2 = 9) :
  ∃ S : ℝ → ℝ, ∀ x : ℝ, S x = 4 * π * x^2 := by sorry

end sphere_surface_area_l1268_126862
