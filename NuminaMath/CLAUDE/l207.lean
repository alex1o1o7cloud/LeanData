import Mathlib

namespace NUMINAMATH_CALUDE_comic_books_liked_by_males_l207_20760

theorem comic_books_liked_by_males 
  (total : ℕ) 
  (female_like_percent : ℚ) 
  (dislike_percent : ℚ) 
  (h_total : total = 300)
  (h_female_like : female_like_percent = 30 / 100)
  (h_dislike : dislike_percent = 30 / 100) :
  (total : ℚ) * (1 - female_like_percent - dislike_percent) = 120 := by
  sorry

end NUMINAMATH_CALUDE_comic_books_liked_by_males_l207_20760


namespace NUMINAMATH_CALUDE_household_expense_sharing_l207_20737

theorem household_expense_sharing (X Y : ℝ) (h : X > Y) :
  (X - Y) / 2 = (X + Y) / 2 - Y := by
  sorry

end NUMINAMATH_CALUDE_household_expense_sharing_l207_20737


namespace NUMINAMATH_CALUDE_largest_c_for_five_in_range_l207_20794

/-- The quadratic function f(x) = 2x^2 - 4x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 4 * x + c

/-- Theorem: The largest value of c such that 5 is in the range of f(x) = 2x^2 - 4x + c is 7 -/
theorem largest_c_for_five_in_range : 
  (∃ (x : ℝ), f 7 x = 5) ∧ 
  (∀ (c : ℝ), c > 7 → ¬∃ (x : ℝ), f c x = 5) := by
  sorry

end NUMINAMATH_CALUDE_largest_c_for_five_in_range_l207_20794


namespace NUMINAMATH_CALUDE_fold_paper_l207_20765

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Function to check if a line is perpendicular bisector of two points -/
def isPerpBisector (l : Line) (p1 p2 : Point) : Prop :=
  let midpoint : Point := ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩
  (midpoint.y = l.slope * midpoint.x + l.intercept) ∧
  (l.slope * (p2.x - p1.x) = -(p2.y - p1.y))

/-- Function to check if two points are symmetric about a line -/
def areSymmetric (l : Line) (p1 p2 : Point) : Prop :=
  isPerpBisector l p1 p2

/-- Main theorem -/
theorem fold_paper (l : Line) (p1 p2 p3 : Point) (p q : ℝ) :
  areSymmetric l ⟨1, 3⟩ ⟨5, 1⟩ →
  areSymmetric l ⟨8, 4⟩ ⟨p, q⟩ →
  p + q = 8 := by
  sorry

end NUMINAMATH_CALUDE_fold_paper_l207_20765


namespace NUMINAMATH_CALUDE_factorial_of_factorial_div_factorial_l207_20793

theorem factorial_of_factorial_div_factorial :
  (Nat.factorial (Nat.factorial 3)) / (Nat.factorial 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_of_factorial_div_factorial_l207_20793


namespace NUMINAMATH_CALUDE_parents_years_in_america_before_aziz_birth_l207_20739

def current_year : ℕ := 2021
def aziz_age : ℕ := 36
def parents_moved_to_america : ℕ := 1982
def parents_return_home : ℕ := 1995
def parents_return_america : ℕ := 1997

def aziz_birth_year : ℕ := current_year - aziz_age

def years_in_america : ℕ := aziz_birth_year - parents_moved_to_america

theorem parents_years_in_america_before_aziz_birth :
  years_in_america = 3 := by sorry

end NUMINAMATH_CALUDE_parents_years_in_america_before_aziz_birth_l207_20739


namespace NUMINAMATH_CALUDE_profit_percentage_l207_20724

theorem profit_percentage (selling_price cost_price profit : ℝ) : 
  cost_price = 0.75 * selling_price →
  profit = selling_price - cost_price →
  (profit / cost_price) * 100 = 100/3 :=
by sorry

end NUMINAMATH_CALUDE_profit_percentage_l207_20724


namespace NUMINAMATH_CALUDE_arithmetic_sequences_problem_l207_20767

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequences_problem 
  (a b : ℕ → ℝ) (d₁ d₂ : ℝ) 
  (ha : arithmetic_sequence a d₁)
  (hb : arithmetic_sequence b d₂)
  (A : ℕ → ℝ)
  (B : ℕ → ℝ)
  (hA : ∀ n, A n = a n + b n)
  (hB : ∀ n, B n = a n * b n)
  (hA₁ : A 1 = 1)
  (hA₂ : A 2 = 3)
  (hB_arith : arithmetic_sequence B (B 2 - B 1)) :
  (∀ n, A n = 2 * n - 1) ∧ d₁ * d₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_problem_l207_20767


namespace NUMINAMATH_CALUDE_mike_investment_l207_20782

/-- Prove that Mike's investment is $350 given the partnership conditions --/
theorem mike_investment (mary_investment : ℝ) (total_profit : ℝ) (profit_difference : ℝ) :
  mary_investment = 650 →
  total_profit = 2999.9999999999995 →
  profit_difference = 600 →
  ∃ (mike_investment : ℝ),
    mike_investment = 350 ∧
    (1/3 * total_profit / 2 + 2/3 * total_profit * mary_investment / (mary_investment + mike_investment) =
     1/3 * total_profit / 2 + 2/3 * total_profit * mike_investment / (mary_investment + mike_investment) + profit_difference) :=
by sorry

end NUMINAMATH_CALUDE_mike_investment_l207_20782


namespace NUMINAMATH_CALUDE_susna_class_f_fraction_l207_20766

/-- Represents the fractions of students getting each grade in Mrs. Susna's class -/
structure GradeDistribution where
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ
  f : ℚ

/-- The conditions of the problem -/
def susna_class : GradeDistribution where
  a := 1/4
  b := 1/2
  c := 1/8
  d := 1/12
  f := 0 -- We'll prove this is actually 1/24

theorem susna_class_f_fraction :
  let g := susna_class
  (g.a + g.b + g.c = 7/8) →
  (g.a + g.b + g.c = 0.875) →
  (g.a + g.b + g.c + g.d + g.f = 1) →
  g.f = 1/24 := by sorry

end NUMINAMATH_CALUDE_susna_class_f_fraction_l207_20766


namespace NUMINAMATH_CALUDE_weight_gain_ratio_l207_20754

/-- The weight gain problem at the family reunion --/
theorem weight_gain_ratio (orlando jose fernando : ℕ) : 
  orlando = 5 →
  jose = 2 * orlando + 2 →
  orlando + jose + fernando = 20 →
  fernando * 4 = jose := by
  sorry

end NUMINAMATH_CALUDE_weight_gain_ratio_l207_20754


namespace NUMINAMATH_CALUDE_expand_and_simplify_l207_20751

theorem expand_and_simplify (x y : ℝ) : (2*x + 3*y)^2 - (2*x - 3*y)^2 = 24*x*y := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l207_20751


namespace NUMINAMATH_CALUDE_fraction_of_number_l207_20731

theorem fraction_of_number : (3 / 4 : ℚ) * (1 / 2 : ℚ) * (2 / 5 : ℚ) * 5020 = 753 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_number_l207_20731


namespace NUMINAMATH_CALUDE_new_member_amount_l207_20742

theorem new_member_amount (group : Finset ℕ) (group_sum : ℕ) (new_member : ℕ) : 
  Finset.card group = 7 →
  group_sum / 7 = 20 →
  (group_sum + new_member) / 8 = 14 →
  new_member = 756 := by
sorry

end NUMINAMATH_CALUDE_new_member_amount_l207_20742


namespace NUMINAMATH_CALUDE_certain_number_proof_l207_20761

theorem certain_number_proof (n : ℝ) : n / 1.25 = 5700 → n = 7125 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l207_20761


namespace NUMINAMATH_CALUDE_brad_balloons_l207_20706

/-- Given that Brad has 8 red balloons and 9 green balloons, prove that he has 17 balloons in total. -/
theorem brad_balloons (red_balloons green_balloons : ℕ) 
  (h1 : red_balloons = 8) 
  (h2 : green_balloons = 9) : 
  red_balloons + green_balloons = 17 := by
  sorry

end NUMINAMATH_CALUDE_brad_balloons_l207_20706


namespace NUMINAMATH_CALUDE_decreasing_function_property_l207_20779

/-- A function f is decreasing on ℝ if for any x₁ < x₂, we have f(x₁) > f(x₂) -/
def DecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

theorem decreasing_function_property (f : ℝ → ℝ) (a : ℝ) 
  (h : DecreasingOn f) : f (a^2 + 1) < f a := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_property_l207_20779


namespace NUMINAMATH_CALUDE_bbq_ice_cost_l207_20725

/-- The cost of ice for Chad's BBQ --/
theorem bbq_ice_cost (people : ℕ) (ice_per_person : ℕ) (bags_per_pack : ℕ) (price_per_pack : ℚ) : 
  people = 15 →
  ice_per_person = 2 →
  bags_per_pack = 10 →
  price_per_pack = 3 →
  (people * ice_per_person : ℚ) / bags_per_pack * price_per_pack = 9 :=
by
  sorry

#check bbq_ice_cost

end NUMINAMATH_CALUDE_bbq_ice_cost_l207_20725


namespace NUMINAMATH_CALUDE_incandescent_bulbs_on_l207_20738

/-- Prove that the number of switched-on incandescent bulbs is 420 -/
theorem incandescent_bulbs_on (total_bulbs : ℕ) 
  (incandescent_percent fluorescent_percent led_percent halogen_percent : ℚ)
  (total_on_percent : ℚ)
  (incandescent_on_percent fluorescent_on_percent led_on_percent halogen_on_percent : ℚ) :
  total_bulbs = 3000 →
  incandescent_percent = 40 / 100 →
  fluorescent_percent = 30 / 100 →
  led_percent = 20 / 100 →
  halogen_percent = 10 / 100 →
  total_on_percent = 55 / 100 →
  incandescent_on_percent = 35 / 100 →
  fluorescent_on_percent = 50 / 100 →
  led_on_percent = 80 / 100 →
  halogen_on_percent = 30 / 100 →
  (incandescent_percent * total_bulbs : ℚ) * incandescent_on_percent = 420 :=
by
  sorry


end NUMINAMATH_CALUDE_incandescent_bulbs_on_l207_20738


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l207_20776

theorem danny_bottle_caps (thrown_away : ℕ) (found : ℕ) (final : ℕ) :
  thrown_away = 60 →
  found = 58 →
  final = 67 →
  final = (thrown_away - found + final) →
  thrown_away - found + final = 69 :=
by sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l207_20776


namespace NUMINAMATH_CALUDE_min_employees_for_tech_company_l207_20735

/-- Calculates the minimum number of employees needed given the number of employees
    for hardware, software, and those working on both. -/
def min_employees (hardware : ℕ) (software : ℕ) (both : ℕ) : ℕ :=
  hardware + software - both

/-- Theorem stating that given 150 employees for hardware, 130 for software,
    and 50 for both, the minimum number of employees needed is 230. -/
theorem min_employees_for_tech_company :
  min_employees 150 130 50 = 230 := by
  sorry

#eval min_employees 150 130 50

end NUMINAMATH_CALUDE_min_employees_for_tech_company_l207_20735


namespace NUMINAMATH_CALUDE_dans_age_l207_20717

theorem dans_age (dans_present_age : ℕ) : dans_present_age = 6 :=
  by
  have h : dans_present_age + 18 = 8 * (dans_present_age - 3) :=
    by sorry
  
  sorry

end NUMINAMATH_CALUDE_dans_age_l207_20717


namespace NUMINAMATH_CALUDE_soccer_league_teams_l207_20732

/-- The number of teams in a soccer league where each team plays every other team once 
    and the total number of games is 105. -/
def num_teams : ℕ := 15

/-- The total number of games played in the league. -/
def total_games : ℕ := 105

/-- Formula for the number of games in a round-robin tournament. -/
def games_formula (n : ℕ) : ℕ := n * (n - 1) / 2

theorem soccer_league_teams : 
  games_formula num_teams = total_games ∧ num_teams > 0 :=
sorry

end NUMINAMATH_CALUDE_soccer_league_teams_l207_20732


namespace NUMINAMATH_CALUDE_sum_three_squares_to_four_fractions_l207_20734

theorem sum_three_squares_to_four_fractions (A B C : ℤ) :
  ∃ (x y z : ℝ), 
    (A : ℝ)^2 + (B : ℝ)^2 + (C : ℝ)^2 = 
      ((A * (x^2 + y^2 - z^2) + B * (2*x*z) + C * (2*y*z)) / (x^2 + y^2 + z^2))^2 +
      ((A * (2*x*z) - B * (x^2 + y^2 - z^2)) / (x^2 + y^2 + z^2))^2 +
      ((B * (2*y*z) - C * (2*x*z)) / (x^2 + y^2 + z^2))^2 +
      ((C * (x^2 + y^2 - z^2) - A * (2*y*z)) / (x^2 + y^2 + z^2))^2 :=
by sorry

end NUMINAMATH_CALUDE_sum_three_squares_to_four_fractions_l207_20734


namespace NUMINAMATH_CALUDE_class_size_proof_l207_20715

theorem class_size_proof (avg_age : ℝ) (avg_age_5 : ℝ) (avg_age_9 : ℝ) (age_15th : ℕ) : 
  avg_age = 15 → 
  avg_age_5 = 14 → 
  avg_age_9 = 16 → 
  age_15th = 11 → 
  ∃ (N : ℕ), N = 15 ∧ N * avg_age = 5 * avg_age_5 + 9 * avg_age_9 + age_15th :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l207_20715


namespace NUMINAMATH_CALUDE_always_integer_solution_l207_20764

theorem always_integer_solution (a : ℕ+) : ∃ x y : ℤ, x^2 - y^2 = (a : ℤ)^3 := by
  sorry

end NUMINAMATH_CALUDE_always_integer_solution_l207_20764


namespace NUMINAMATH_CALUDE_three_turns_sufficient_l207_20710

/-- Represents a five-digit number with distinct digits -/
structure FiveDigitNumber where
  digits : Fin 5 → Fin 10
  distinct : ∀ i j, i ≠ j → digits i ≠ digits j

/-- Represents a turn where positions are selected and digits are revealed -/
structure Turn where
  positions : Set (Fin 5)
  revealed_digits : Set (Fin 10)

/-- Represents the process of guessing the number -/
def guess_number (n : FiveDigitNumber) (turns : List Turn) : Prop :=
  ∀ m : FiveDigitNumber, 
    (∀ t ∈ turns, {n.digits i | i ∈ t.positions} = t.revealed_digits) →
    (∀ t ∈ turns, {m.digits i | i ∈ t.positions} = t.revealed_digits) →
    n = m

/-- The main theorem stating that 3 turns are sufficient -/
theorem three_turns_sufficient :
  ∃ strategy : List Turn, 
    strategy.length ≤ 3 ∧ 
    ∀ n : FiveDigitNumber, guess_number n strategy :=
sorry

end NUMINAMATH_CALUDE_three_turns_sufficient_l207_20710


namespace NUMINAMATH_CALUDE_lines_dont_intersect_if_points_not_coplanar_l207_20757

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- Check if four points are coplanar -/
def are_coplanar (a b c d : Point3D) : Prop := sorry

/-- Check if two lines intersect -/
def lines_intersect (l1 l2 : Line3D) : Prop := sorry

theorem lines_dont_intersect_if_points_not_coplanar 
  (a b c d : Point3D) 
  (h : ¬ are_coplanar a b c d) : 
  ¬ lines_intersect (Line3D.mk a b) (Line3D.mk c d) := by
  sorry

end NUMINAMATH_CALUDE_lines_dont_intersect_if_points_not_coplanar_l207_20757


namespace NUMINAMATH_CALUDE_perfect_square_factorization_l207_20769

theorem perfect_square_factorization (a : ℝ) : a^2 - 2*a + 1 = (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_factorization_l207_20769


namespace NUMINAMATH_CALUDE_value_of_p_l207_20785

theorem value_of_p (n : ℝ) (p : ℝ) : 
  n = 9/4 → p = 4*n*(1/2^2009)^(Real.log 1) → p = 9 := by sorry

end NUMINAMATH_CALUDE_value_of_p_l207_20785


namespace NUMINAMATH_CALUDE_train_length_l207_20741

/-- The length of a train given specific passing times -/
theorem train_length : ∃ (L : ℝ), 
  (∀ (V : ℝ), V = L / 24 → V = (L + 650) / 89) → L = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l207_20741


namespace NUMINAMATH_CALUDE_amount_ratio_l207_20796

theorem amount_ratio (total : ℚ) (b_amt : ℚ) (a_fraction : ℚ) :
  total = 1440 →
  b_amt = 270 →
  a_fraction = 1/3 →
  ∃ (c_amt : ℚ),
    total = a_fraction * b_amt + b_amt + c_amt ∧
    b_amt / c_amt = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_amount_ratio_l207_20796


namespace NUMINAMATH_CALUDE_puppy_food_consumption_l207_20733

/-- Given the cost of a puppy, the duration of food supply, the amount and cost of food per bag,
    and the total cost, this theorem proves the daily food consumption of the puppy. -/
theorem puppy_food_consumption
  (puppy_cost : ℚ)
  (food_duration_weeks : ℕ)
  (food_per_bag : ℚ)
  (cost_per_bag : ℚ)
  (total_cost : ℚ)
  (h1 : puppy_cost = 10)
  (h2 : food_duration_weeks = 3)
  (h3 : food_per_bag = 7/2)
  (h4 : cost_per_bag = 2)
  (h5 : total_cost = 14) :
  (total_cost - puppy_cost) / cost_per_bag * food_per_bag / (food_duration_weeks * 7 : ℚ) = 1/3 :=
sorry

end NUMINAMATH_CALUDE_puppy_food_consumption_l207_20733


namespace NUMINAMATH_CALUDE_cost_of_48_doughnuts_l207_20705

/-- The cost of buying a specified number of doughnuts -/
def doughnutCost (n : ℕ) : ℚ :=
  1 + 6 * ((n - 1) / 12)

/-- Theorem stating the cost of 48 doughnuts -/
theorem cost_of_48_doughnuts : doughnutCost 48 = 25 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_48_doughnuts_l207_20705


namespace NUMINAMATH_CALUDE_mean_of_first_set_l207_20787

def first_set (x : ℝ) : List ℝ := [28, x, 70, 88, 104]
def second_set (x : ℝ) : List ℝ := [50, 62, 97, 124, x]

theorem mean_of_first_set :
  ∀ x : ℝ,
  (List.sum (second_set x)) / 5 = 75.6 →
  (List.sum (first_set x)) / 5 = 67 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_of_first_set_l207_20787


namespace NUMINAMATH_CALUDE_opposite_rational_division_l207_20720

theorem opposite_rational_division (a : ℚ) : 
  (a ≠ 0 → a / (-a) = -1) ∧ (a = 0 → a / (-a) = 0/0) :=
sorry

end NUMINAMATH_CALUDE_opposite_rational_division_l207_20720


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l207_20790

theorem bobby_candy_problem (total_candy : ℕ) (chocolate_eaten : ℕ) (gummy_eaten : ℕ)
  (h1 : total_candy = 36)
  (h2 : chocolate_eaten = 12)
  (h3 : gummy_eaten = 9)
  (h4 : chocolate_eaten = 2 * (chocolate_eaten + (total_candy - chocolate_eaten - gummy_eaten)) / 3)
  (h5 : gummy_eaten = 3 * (gummy_eaten + (total_candy - chocolate_eaten - gummy_eaten)) / 4) :
  total_candy - chocolate_eaten - gummy_eaten = 9 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l207_20790


namespace NUMINAMATH_CALUDE_harriet_speed_back_l207_20763

/-- Harriet's round trip between A-ville and B-town -/
def harriet_trip (speed_to_b : ℝ) (total_time : ℝ) (time_to_b_minutes : ℝ) : Prop :=
  let time_to_b : ℝ := time_to_b_minutes / 60
  let distance : ℝ := speed_to_b * time_to_b
  let time_from_b : ℝ := total_time - time_to_b
  let speed_from_b : ℝ := distance / time_from_b
  speed_from_b = 140

theorem harriet_speed_back :
  harriet_trip 110 5 168 := by sorry

end NUMINAMATH_CALUDE_harriet_speed_back_l207_20763


namespace NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l207_20711

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem third_term_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q → a 1 = 3 → q = -2 → a 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l207_20711


namespace NUMINAMATH_CALUDE_adjacent_supplementary_angles_l207_20730

theorem adjacent_supplementary_angles (angle1 angle2 : ℝ) : 
  (angle1 + angle2 = 180) → (angle1 = 80) → (angle2 = 100) := by
  sorry

end NUMINAMATH_CALUDE_adjacent_supplementary_angles_l207_20730


namespace NUMINAMATH_CALUDE_matchstick_rearrangement_l207_20756

theorem matchstick_rearrangement : |(22 : ℝ) / 7 - Real.pi| < (1 : ℝ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_rearrangement_l207_20756


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l207_20727

theorem inequality_system_integer_solutions :
  ∀ x : ℤ, (3 * x + 6 > x + 8 ∧ x / 4 ≥ (x - 1) / 3) ↔ x ∈ ({2, 3, 4} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l207_20727


namespace NUMINAMATH_CALUDE_brick_width_calculation_l207_20752

theorem brick_width_calculation (wall_length wall_width wall_height : ℝ)
                                (brick_length brick_height : ℝ)
                                (num_bricks : ℕ) :
  wall_length = 800 →
  wall_width = 600 →
  wall_height = 22.5 →
  brick_length = 80 →
  brick_height = 6 →
  num_bricks = 2000 →
  ∃ brick_width : ℝ,
    num_bricks * (brick_length * brick_width * brick_height) = wall_length * wall_width * wall_height ∧
    brick_width = 5.625 := by
  sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l207_20752


namespace NUMINAMATH_CALUDE_complex_power_difference_l207_20784

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^24 - (1 - i)^24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l207_20784


namespace NUMINAMATH_CALUDE_pencil_box_sequence_l207_20780

theorem pencil_box_sequence (a : ℕ → ℕ) (h1 : a 0 = 78) (h2 : a 1 = 87) (h3 : a 2 = 96) (h4 : a 3 = 105)
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) : a 4 = 114 := by
  sorry

end NUMINAMATH_CALUDE_pencil_box_sequence_l207_20780


namespace NUMINAMATH_CALUDE_monotone_function_k_range_l207_20774

/-- Given a function f(x) = e^x + kx - ln x that is monotonically increasing on (1, +∞),
    prove that k ∈ [1-e, +∞) -/
theorem monotone_function_k_range (k : ℝ) :
  (∀ x > 1, Monotone (fun x => Real.exp x + k * x - Real.log x)) →
  k ∈ Set.Ici (1 - Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_monotone_function_k_range_l207_20774


namespace NUMINAMATH_CALUDE_paint_mixture_fraction_l207_20758

theorem paint_mixture_fraction (original_intensity replacement_intensity new_intensity : ℝ) 
  (h1 : original_intensity = 0.5)
  (h2 : replacement_intensity = 0.25)
  (h3 : new_intensity = 0.45) :
  ∃ (x : ℝ), 
    x ≥ 0 ∧ x ≤ 1 ∧
    original_intensity * (1 - x) + replacement_intensity * x = new_intensity ∧
    x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_paint_mixture_fraction_l207_20758


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l207_20791

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (a > 2 → a ∈ Set.Ici 2) ∧ (∃ x, x ∈ Set.Ici 2 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l207_20791


namespace NUMINAMATH_CALUDE_option_B_more_cost_effective_l207_20773

/-- The cost function for Option A -/
def cost_A (x : ℝ) : ℝ := 60 + 18 * x

/-- The cost function for Option B -/
def cost_B (x : ℝ) : ℝ := 150 + 15 * x

/-- Theorem: Option B is more cost-effective for 40 kilograms of blueberries -/
theorem option_B_more_cost_effective :
  cost_B 40 < cost_A 40 := by
  sorry

end NUMINAMATH_CALUDE_option_B_more_cost_effective_l207_20773


namespace NUMINAMATH_CALUDE_greatest_multiple_of_three_cubed_less_than_1000_l207_20753

theorem greatest_multiple_of_three_cubed_less_than_1000 :
  ∃ (x : ℕ), 
    x > 0 ∧ 
    ∃ (k : ℕ), x = 3 * k ∧ 
    x^3 < 1000 ∧
    ∀ (y : ℕ), y > 0 → (∃ (m : ℕ), y = 3 * m) → y^3 < 1000 → y ≤ x ∧
    x = 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_three_cubed_less_than_1000_l207_20753


namespace NUMINAMATH_CALUDE_expand_expression_l207_20778

theorem expand_expression (x : ℝ) : (7*x^2 + 5*x + 8) * 3*x = 21*x^3 + 15*x^2 + 24*x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l207_20778


namespace NUMINAMATH_CALUDE_hank_reads_seven_days_a_week_l207_20703

/-- Represents Hank's reading habits and total reading time in a week -/
structure ReadingHabits where
  weekdayReadingTime : ℕ  -- Daily reading time on weekdays in minutes
  weekendReadingTime : ℕ  -- Daily reading time on weekends in minutes
  totalWeeklyTime : ℕ     -- Total reading time in a week in minutes

/-- Calculates the number of days Hank reads in a week based on his reading habits -/
def daysReadingPerWeek (habits : ReadingHabits) : ℕ :=
  if (5 * habits.weekdayReadingTime + 2 * habits.weekendReadingTime) = habits.totalWeeklyTime
  then 7
  else 0

/-- Theorem stating that Hank reads 7 days a week given his reading habits -/
theorem hank_reads_seven_days_a_week :
  let habits : ReadingHabits := {
    weekdayReadingTime := 90,
    weekendReadingTime := 180,
    totalWeeklyTime := 810
  }
  daysReadingPerWeek habits = 7 := by sorry

end NUMINAMATH_CALUDE_hank_reads_seven_days_a_week_l207_20703


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l207_20743

/-- An isosceles triangle with sides 4 and 9 has a perimeter of 22 -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 4 ∨ a = 9) ∧  -- One side is either 4 or 9
    (b = a) ∧          -- The triangle is isosceles
    (c = if a = 4 then 9 else 4) ∧  -- The third side is whichever of 4 or 9 that a is not
    (a + b + c = 22)   -- The perimeter is 22

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof :
  ∃ a b c, isosceles_triangle_perimeter a b c :=
by
  sorry  -- The proof is omitted as per instructions

#check isosceles_triangle_perimeter
#check isosceles_triangle_perimeter_proof

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l207_20743


namespace NUMINAMATH_CALUDE_jerry_log_count_l207_20709

/-- The number of logs produced by a pine tree -/
def logsPerPine : ℕ := 80

/-- The number of logs produced by a maple tree -/
def logsPerMaple : ℕ := 60

/-- The number of logs produced by a walnut tree -/
def logsPerWalnut : ℕ := 100

/-- The number of pine trees Jerry cuts -/
def pineTreesCut : ℕ := 8

/-- The number of maple trees Jerry cuts -/
def mapleTreesCut : ℕ := 3

/-- The number of walnut trees Jerry cuts -/
def walnutTreesCut : ℕ := 4

/-- The total number of logs Jerry gets -/
def totalLogs : ℕ := logsPerPine * pineTreesCut + logsPerMaple * mapleTreesCut + logsPerWalnut * walnutTreesCut

theorem jerry_log_count : totalLogs = 1220 := by
  sorry

end NUMINAMATH_CALUDE_jerry_log_count_l207_20709


namespace NUMINAMATH_CALUDE_contestant_A_score_l207_20783

def speech_contest_score (content_score : ℕ) (skills_score : ℕ) (effects_score : ℕ) : ℚ :=
  (4 * content_score + 2 * skills_score + 4 * effects_score) / 10

theorem contestant_A_score :
  speech_contest_score 90 80 90 = 88 := by
  sorry

end NUMINAMATH_CALUDE_contestant_A_score_l207_20783


namespace NUMINAMATH_CALUDE_hexagon_area_fraction_l207_20795

/-- Represents a tiling pattern of the plane -/
structure TilingPattern where
  /-- The number of smaller units in one side of a large square -/
  units_per_side : ℕ
  /-- The number of units occupied by hexagons in a large square -/
  hexagon_units : ℕ

/-- The fraction of the plane enclosed by hexagons -/
def hexagon_fraction (pattern : TilingPattern) : ℚ :=
  pattern.hexagon_units / (pattern.units_per_side ^ 2 : ℚ)

/-- The specific tiling pattern described in the problem -/
def problem_pattern : TilingPattern :=
  { units_per_side := 4
  , hexagon_units := 8 }

theorem hexagon_area_fraction :
  hexagon_fraction problem_pattern = 1/2 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_fraction_l207_20795


namespace NUMINAMATH_CALUDE_arith_geom_seq_iff_not_squarefree_l207_20789

/-- A sequence in ℤ/mℤ is both arithmetic and geometric progression -/
def is_arith_geom_seq (m : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∃ (a d r : ℕ), ∀ n : ℕ,
    (seq n) % m = (a + n * d) % m ∧
    (seq n) % m = (a * r^n) % m

/-- A sequence is nonconstant -/
def is_nonconstant (m : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∃ i j : ℕ, (seq i) % m ≠ (seq j) % m

/-- m is not squarefree -/
def not_squarefree (m : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ p^2 ∣ m

/-- Main theorem -/
theorem arith_geom_seq_iff_not_squarefree (m : ℕ) :
  (∃ seq : ℕ → ℕ, is_arith_geom_seq m seq ∧ is_nonconstant m seq) ↔ not_squarefree m :=
sorry

end NUMINAMATH_CALUDE_arith_geom_seq_iff_not_squarefree_l207_20789


namespace NUMINAMATH_CALUDE_tom_final_balance_l207_20750

def calculate_final_balance (initial_allowance : ℚ) (extra_earning : ℚ) (final_spending : ℚ) : ℚ :=
  let week1_balance := initial_allowance - initial_allowance / 3
  let week2_balance := week1_balance - week1_balance / 4
  let week3_balance_before_spending := week2_balance + extra_earning
  let week3_balance_after_spending := week3_balance_before_spending / 2
  week3_balance_after_spending - final_spending

theorem tom_final_balance :
  calculate_final_balance 12 5 3 = (5/2 : ℚ) := by sorry

end NUMINAMATH_CALUDE_tom_final_balance_l207_20750


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l207_20759

theorem quadratic_equation_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x > 0 ∧ 
    x^2 - a*x + a^2 - 4 = 0 ∧ 
    y^2 - a*y + a^2 - 4 = 0) ↔ 
  -2 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l207_20759


namespace NUMINAMATH_CALUDE_diagonals_bisect_in_special_quadrilaterals_l207_20781

-- Define a type for quadrilaterals
inductive Quadrilateral
  | Parallelogram
  | Rectangle
  | Rhombus
  | Square

-- Define a function to check if diagonals bisect each other
def diagonalsBisectEachOther (q : Quadrilateral) : Prop :=
  match q with
  | Quadrilateral.Parallelogram => true
  | Quadrilateral.Rectangle => true
  | Quadrilateral.Rhombus => true
  | Quadrilateral.Square => true

-- Theorem statement
theorem diagonals_bisect_in_special_quadrilaterals (q : Quadrilateral) :
  diagonalsBisectEachOther q := by
  sorry

end NUMINAMATH_CALUDE_diagonals_bisect_in_special_quadrilaterals_l207_20781


namespace NUMINAMATH_CALUDE_reimbursement_is_correct_l207_20721

/-- Represents the type of client --/
inductive ClientType
| Industrial
| Commercial

/-- Represents the day of the week --/
inductive DayType
| Weekday
| Weekend

/-- Calculates the reimbursement rate based on client type and day type --/
def reimbursementRate (client : ClientType) (day : DayType) : ℚ :=
  match client, day with
  | ClientType.Industrial, DayType.Weekday => 36/100
  | ClientType.Commercial, DayType.Weekday => 42/100
  | _, DayType.Weekend => 45/100

/-- Represents a day's travel --/
structure DayTravel where
  miles : ℕ
  client : ClientType
  day : DayType

/-- Calculates the reimbursement for a single day --/
def dailyReimbursement (travel : DayTravel) : ℚ :=
  (travel.miles : ℚ) * reimbursementRate travel.client travel.day

/-- The week's travel schedule --/
def weekSchedule : List DayTravel := [
  ⟨18, ClientType.Industrial, DayType.Weekday⟩,
  ⟨26, ClientType.Commercial, DayType.Weekday⟩,
  ⟨20, ClientType.Industrial, DayType.Weekday⟩,
  ⟨20, ClientType.Commercial, DayType.Weekday⟩,
  ⟨16, ClientType.Industrial, DayType.Weekday⟩,
  ⟨12, ClientType.Commercial, DayType.Weekend⟩
]

/-- Calculates the total reimbursement for the week --/
def totalReimbursement (schedule : List DayTravel) : ℚ :=
  schedule.map dailyReimbursement |>.sum

/-- Theorem: The total reimbursement for the given week schedule is $44.16 --/
theorem reimbursement_is_correct : totalReimbursement weekSchedule = 4416/100 := by
  sorry

end NUMINAMATH_CALUDE_reimbursement_is_correct_l207_20721


namespace NUMINAMATH_CALUDE_cube_surface_coverage_l207_20712

/-- Represents a cube -/
structure Cube where
  vertices : ℕ
  angle_sum_at_vertex : ℕ

/-- Represents a triangle -/
structure Triangle where
  angle_sum : ℕ

/-- The problem statement -/
theorem cube_surface_coverage (c : Cube) (t : Triangle) : 
  c.vertices = 8 → 
  c.angle_sum_at_vertex = 270 → 
  t.angle_sum = 180 → 
  ¬ (3 * t.angle_sum ≥ c.vertices * 90) :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_coverage_l207_20712


namespace NUMINAMATH_CALUDE_ordered_pairs_count_l207_20799

theorem ordered_pairs_count : ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
  p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 36) (Finset.range 37 ×ˢ Finset.range 37)).card ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_ordered_pairs_count_l207_20799


namespace NUMINAMATH_CALUDE_platform_length_l207_20700

/-- Given a train of length 300 meters that crosses a platform in 39 seconds
    and a signal pole in 26 seconds, the length of the platform is 150 meters. -/
theorem platform_length (train_length : ℝ) (platform_cross_time : ℝ) (pole_cross_time : ℝ) :
  train_length = 300 →
  platform_cross_time = 39 →
  pole_cross_time = 26 →
  ∃ platform_length : ℝ,
    platform_length = 150 ∧
    (train_length / pole_cross_time) * platform_cross_time = train_length + platform_length :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l207_20700


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l207_20748

/-- Given an arithmetic sequence with first term -1 and third term 5, prove that the fifth term is 11. -/
theorem arithmetic_sequence_fifth_term :
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
    a 1 = -1 →                                        -- first term
    a 3 = 5 →                                         -- third term
    a 5 = 11 :=                                       -- fifth term (to prove)
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l207_20748


namespace NUMINAMATH_CALUDE_ratio_evaluation_l207_20786

theorem ratio_evaluation : (2^2003 * 3^2002) / 6^2002 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_evaluation_l207_20786


namespace NUMINAMATH_CALUDE_ivanov_family_net_worth_l207_20713

/-- The net worth of the Ivanov family -/
def ivanov_net_worth : ℕ := by sorry

/-- The value of the Ivanov family's apartment in rubles -/
def apartment_value : ℕ := 3000000

/-- The value of the Ivanov family's car in rubles -/
def car_value : ℕ := 900000

/-- The amount in the Ivanov family's bank deposit in rubles -/
def bank_deposit : ℕ := 300000

/-- The value of the Ivanov family's securities in rubles -/
def securities_value : ℕ := 200000

/-- The amount of liquid cash the Ivanov family has in rubles -/
def liquid_cash : ℕ := 100000

/-- The remaining mortgage balance of the Ivanov family in rubles -/
def mortgage_balance : ℕ := 1500000

/-- The remaining car loan balance of the Ivanov family in rubles -/
def car_loan_balance : ℕ := 500000

/-- The debt the Ivanov family owes to relatives in rubles -/
def debt_to_relatives : ℕ := 200000

/-- The total assets of the Ivanov family -/
def total_assets : ℕ := apartment_value + car_value + bank_deposit + securities_value + liquid_cash

/-- The total liabilities of the Ivanov family -/
def total_liabilities : ℕ := mortgage_balance + car_loan_balance + debt_to_relatives

theorem ivanov_family_net_worth :
  ivanov_net_worth = total_assets - total_liabilities := by sorry

end NUMINAMATH_CALUDE_ivanov_family_net_worth_l207_20713


namespace NUMINAMATH_CALUDE_smallest_candy_count_l207_20701

theorem smallest_candy_count : ∃ n : ℕ, 
  (n ≥ 100) ∧ (n < 1000) ∧ 
  ((n + 7) % 6 = 0) ∧ 
  ((n - 5) % 9 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < n ∧ m < 1000 → 
    ((m + 7) % 6 ≠ 0) ∨ ((m - 5) % 9 ≠ 0)) ∧
  n = 113 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l207_20701


namespace NUMINAMATH_CALUDE_number_above_345_l207_20798

/-- Represents the triangular array structure -/
structure TriangularArray where
  /-- Returns the number of elements in the k-th row -/
  elementsInRow : ℕ → ℕ
  /-- Returns the sum of elements up to and including the k-th row -/
  sumUpToRow : ℕ → ℕ
  /-- First row has one element -/
  first_row_one : elementsInRow 1 = 1
  /-- Each row has three more elements than the previous -/
  row_increment : ∀ k, elementsInRow (k + 1) = elementsInRow k + 3
  /-- Sum formula for elements up to k-th row -/
  sum_formula : ∀ k, sumUpToRow k = k * (3 * k - 1) / 2

theorem number_above_345 (arr : TriangularArray) :
  ∃ (row : ℕ) (pos : ℕ),
    arr.sumUpToRow (row - 1) < 345 ∧
    345 ≤ arr.sumUpToRow row ∧
    pos = 345 - arr.sumUpToRow (row - 1) ∧
    arr.sumUpToRow (row - 2) + pos = 308 :=
  sorry

end NUMINAMATH_CALUDE_number_above_345_l207_20798


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l207_20771

theorem modulus_of_complex_fraction (z : ℂ) : z = (1 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l207_20771


namespace NUMINAMATH_CALUDE_probability_four_primes_in_six_rolls_l207_20718

/-- The probability of getting exactly 4 prime numbers in 6 rolls of a fair 8-sided die -/
theorem probability_four_primes_in_six_rolls (die : Finset ℕ) 
  (h_die : die = {1, 2, 3, 4, 5, 6, 7, 8}) 
  (h_prime : {n ∈ die | Nat.Prime n} = {2, 3, 5, 7}) : 
  (Nat.choose 6 4 * (4 / 8)^4 * (4 / 8)^2) = 15 / 64 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_primes_in_six_rolls_l207_20718


namespace NUMINAMATH_CALUDE_average_score_is_71_l207_20772

def mathematics_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 47
def biology_score : ℕ := 85

def total_score : ℕ := mathematics_score + science_score + social_studies_score + english_score + biology_score
def number_of_subjects : ℕ := 5

theorem average_score_is_71 : (total_score : ℚ) / number_of_subjects = 71 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_71_l207_20772


namespace NUMINAMATH_CALUDE_crew_members_count_l207_20762

/-- Represents the number of passengers at each stage of the flight --/
structure PassengerCount where
  initial : ℕ
  after_texas : ℕ
  after_north_carolina : ℕ

/-- Calculates the final number of passengers --/
def final_passengers (p : PassengerCount) : ℕ :=
  p.initial - 58 + 24 - 47 + 14

/-- Represents the flight data --/
structure FlightData where
  passenger_count : PassengerCount
  total_landed : ℕ

/-- Calculates the number of crew members --/
def crew_members (f : FlightData) : ℕ :=
  f.total_landed - final_passengers f.passenger_count

/-- Theorem stating the number of crew members --/
theorem crew_members_count (f : FlightData) 
  (h1 : f.passenger_count.initial = 124)
  (h2 : f.total_landed = 67) : 
  crew_members f = 10 := by
  sorry

#check crew_members_count

end NUMINAMATH_CALUDE_crew_members_count_l207_20762


namespace NUMINAMATH_CALUDE_students_per_class_l207_20736

/-- Prove the number of students per class in a school's reading program -/
theorem students_per_class (c : ℕ) (h1 : c > 0) : 
  let books_per_student_per_year := 5 * 12
  let total_books_read := 60
  let s := total_books_read / (c * books_per_student_per_year)
  s = 1 / c := by
  sorry

end NUMINAMATH_CALUDE_students_per_class_l207_20736


namespace NUMINAMATH_CALUDE_sum_of_fractions_l207_20716

theorem sum_of_fractions : (1 : ℚ) / 3 + (1 : ℚ) / 4 = 7 / 12 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l207_20716


namespace NUMINAMATH_CALUDE_fred_money_last_week_l207_20755

theorem fred_money_last_week 
  (fred_now : ℕ)
  (jason_now : ℕ)
  (jason_earned : ℕ)
  (jason_last_week : ℕ)
  (h1 : fred_now = 112)
  (h2 : jason_now = 63)
  (h3 : jason_earned = 60)
  (h4 : jason_last_week = 3)
  : fred_now - (jason_earned + jason_last_week) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fred_money_last_week_l207_20755


namespace NUMINAMATH_CALUDE_scaled_recipe_correct_l207_20768

/-- Represents a cookie recipe -/
structure CookieRecipe where
  cookies : ℕ
  flour : ℕ
  eggs : ℕ

/-- Scales a cookie recipe by a given factor -/
def scaleRecipe (recipe : CookieRecipe) (factor : ℕ) : CookieRecipe :=
  { cookies := recipe.cookies * factor
  , flour := recipe.flour * factor
  , eggs := recipe.eggs * factor }

theorem scaled_recipe_correct (original : CookieRecipe) (scaled : CookieRecipe) :
  original.cookies = 40 ∧
  original.flour = 3 ∧
  original.eggs = 2 ∧
  scaled = scaleRecipe original 3 →
  scaled.cookies = 120 ∧
  scaled.flour = 9 ∧
  scaled.eggs = 6 := by
  sorry

#check scaled_recipe_correct

end NUMINAMATH_CALUDE_scaled_recipe_correct_l207_20768


namespace NUMINAMATH_CALUDE_f_definition_f_of_five_l207_20719

noncomputable def f : ℝ → ℝ := λ u => (u^3 + 6*u^2 + 21*u + 40) / 27

theorem f_definition (x : ℝ) : f (3*x - 1) = x^3 + x^2 + x + 1 := by sorry

theorem f_of_five : f 5 = 140 / 9 := by sorry

end NUMINAMATH_CALUDE_f_definition_f_of_five_l207_20719


namespace NUMINAMATH_CALUDE_problem_solution_l207_20788

theorem problem_solution :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 ∧ x ≠ 3 →
    (3*x - 8) / (x - 1) - (x + 1) / x / ((x^2 - 1) / (x^2 - 3*x)) = (2*x - 5) / (x - 1)) ∧
  ((Real.sqrt 12 - (-1/2)⁻¹ - |Real.sqrt 3 + 3| + (2023 - Real.pi)^0) = Real.sqrt 3) ∧
  ((2*2 - 5) / (2 - 1) = -1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l207_20788


namespace NUMINAMATH_CALUDE_some_number_equation_l207_20744

theorem some_number_equation (x : ℤ) : |x - 8*(3 - 12)| - |5 - 11| = 70 ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_some_number_equation_l207_20744


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l207_20770

/-- A polynomial is a perfect square trinomial if it can be expressed as (ax + b)^2 -/
def is_perfect_square_trinomial (a b m : ℝ) : Prop :=
  ∀ x, x^2 + m*x + 4 = (a*x + b)^2

/-- If x^2 + mx + 4 is a perfect square trinomial, then m = 4 or m = -4 -/
theorem perfect_square_trinomial_m_value (m : ℝ) :
  (∃ a b : ℝ, is_perfect_square_trinomial a b m) → m = 4 ∨ m = -4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l207_20770


namespace NUMINAMATH_CALUDE_infinitely_many_terms_same_prime_factors_l207_20792

/-- An arithmetic progression of natural numbers -/
def arithmeticProgression (a d : ℕ) : ℕ → ℕ := fun n => a + n * d

/-- The set of prime factors of a natural number -/
def primeFactors (n : ℕ) : Set ℕ := {p : ℕ | Nat.Prime p ∧ p ∣ n}

/-- There are infinitely many terms in an arithmetic progression with the same prime factors -/
theorem infinitely_many_terms_same_prime_factors (a d : ℕ) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, primeFactors (arithmeticProgression a d n) = primeFactors a :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_terms_same_prime_factors_l207_20792


namespace NUMINAMATH_CALUDE_sum_of_sequences_l207_20775

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def sum_list (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

theorem sum_of_sequences : 
  let seq1 := arithmetic_sequence 3 10 6
  let seq2 := arithmetic_sequence 5 10 6
  sum_list seq1 + sum_list seq2 = 348 := by
sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l207_20775


namespace NUMINAMATH_CALUDE_sum_of_series_equals_two_l207_20749

/-- The sum of the infinite series ∑(n=1 to ∞) (4n-1)/3^n is equal to 2 -/
theorem sum_of_series_equals_two :
  let series := fun n : ℕ => (4 * n - 1) / (3 ^ n : ℝ)
  (∑' n, series n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_series_equals_two_l207_20749


namespace NUMINAMATH_CALUDE_ml_to_litre_fraction_l207_20728

theorem ml_to_litre_fraction (ml_per_litre : ℝ) (volume_ml : ℝ) :
  ml_per_litre = 1000 →
  volume_ml = 30 →
  volume_ml / ml_per_litre = 0.03 := by
sorry

end NUMINAMATH_CALUDE_ml_to_litre_fraction_l207_20728


namespace NUMINAMATH_CALUDE_abs_x_minus_one_necessary_not_sufficient_l207_20723

theorem abs_x_minus_one_necessary_not_sufficient :
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) ∧
  (∃ x : ℝ, |x - 1| < 2 ∧ ¬(x * (x - 3) < 0)) := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_necessary_not_sufficient_l207_20723


namespace NUMINAMATH_CALUDE_boat_speed_l207_20740

/-- The average speed of a boat in still water, given travel times with and against a current. -/
theorem boat_speed (time_with_current time_against_current current_speed : ℝ)
  (h1 : time_with_current = 2)
  (h2 : time_against_current = 2.5)
  (h3 : current_speed = 3)
  (h4 : time_with_current * (x + current_speed) = time_against_current * (x - current_speed)) :
  x = 27 :=
by sorry


end NUMINAMATH_CALUDE_boat_speed_l207_20740


namespace NUMINAMATH_CALUDE_initial_weavers_count_l207_20797

/-- The number of mat-weavers initially weaving -/
def initial_weavers : ℕ := sorry

/-- The number of mats woven by the initial weavers -/
def initial_mats : ℕ := 4

/-- The number of days taken by the initial weavers -/
def initial_days : ℕ := 4

/-- The number of mat-weavers in the second scenario -/
def second_weavers : ℕ := 8

/-- The number of mats woven in the second scenario -/
def second_mats : ℕ := 16

/-- The number of days taken in the second scenario -/
def second_days : ℕ := 8

/-- The rate of weaving is consistent across both scenarios -/
axiom consistent_rate : 
  (initial_mats : ℚ) / (initial_weavers * initial_days) = 
  (second_mats : ℚ) / (second_weavers * second_days)

theorem initial_weavers_count : initial_weavers = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_weavers_count_l207_20797


namespace NUMINAMATH_CALUDE_prove_some_number_l207_20729

theorem prove_some_number (a : ℕ) (some_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = 21 * 35 * some_number * 35) :
  some_number = 21 := by
sorry

end NUMINAMATH_CALUDE_prove_some_number_l207_20729


namespace NUMINAMATH_CALUDE_younger_son_future_age_l207_20704

def age_difference : ℕ := 10
def elder_son_current_age : ℕ := 40
def years_in_future : ℕ := 30

theorem younger_son_future_age :
  let younger_son_current_age := elder_son_current_age - age_difference
  younger_son_current_age + years_in_future = 60 := by sorry

end NUMINAMATH_CALUDE_younger_son_future_age_l207_20704


namespace NUMINAMATH_CALUDE_games_next_month_l207_20707

/-- Calculates the number of games Jason plans to attend next month -/
theorem games_next_month 
  (games_this_month : ℕ) 
  (games_last_month : ℕ) 
  (total_games : ℕ) 
  (h1 : games_this_month = 11)
  (h2 : games_last_month = 17)
  (h3 : total_games = 44) :
  total_games - (games_this_month + games_last_month) = 16 := by
sorry

end NUMINAMATH_CALUDE_games_next_month_l207_20707


namespace NUMINAMATH_CALUDE_right_triangle_stable_l207_20714

/-- A shape is considered stable if it maintains its form without deformation under normal conditions. -/
def Stable (shape : Type) : Prop := sorry

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
def Parallelogram : Type := sorry

/-- A square is a quadrilateral with four equal sides and four right angles. -/
def Square : Type := sorry

/-- A rectangle is a quadrilateral with four right angles. -/
def Rectangle : Type := sorry

/-- A right triangle is a triangle with one right angle. -/
def RightTriangle : Type := sorry

/-- Theorem stating that among the given shapes, only the right triangle is inherently stable. -/
theorem right_triangle_stable :
  ¬Stable Parallelogram ∧
  ¬Stable Square ∧
  ¬Stable Rectangle ∧
  Stable RightTriangle :=
sorry

end NUMINAMATH_CALUDE_right_triangle_stable_l207_20714


namespace NUMINAMATH_CALUDE_largest_n_multiple_of_7_l207_20708

def is_multiple_of_7 (n : ℕ) : Prop :=
  (5 * (n - 3)^6 - 2 * n^3 + 20 * n - 35) % 7 = 0

theorem largest_n_multiple_of_7 :
  ∀ n : ℕ, n < 100000 →
    (is_multiple_of_7 n → n ≤ 99998) ∧
    (n > 99998 → ¬is_multiple_of_7 n) ∧
    is_multiple_of_7 99998 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_multiple_of_7_l207_20708


namespace NUMINAMATH_CALUDE_geometric_sequence_and_sum_l207_20745

-- Define the geometric sequence a_n
def a (n : ℕ) : ℝ := 2 * 3^(n - 1)

-- Define the arithmetic sequence c_n
def c (n : ℕ) : ℝ := 2 * n + 2

-- Define the sequence b_n
def b (n : ℕ) : ℝ := c n - a n

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℝ := n^2 + 3*n - 3^n + 1

theorem geometric_sequence_and_sum :
  (∀ n, a (n + 1) / a n > 1) ∧  -- Common ratio > 1
  a 2 = 6 ∧
  a 1 + a 2 + a 3 = 26 ∧
  (∀ n, c n = a n + b n) ∧
  (∀ n, c (n + 1) - c n = c 2 - c 1) ∧  -- c_n is arithmetic
  b 1 = a 1 ∧
  b 3 = -10 →
  (∀ n, a n = 2 * 3^(n - 1)) ∧
  (∀ n, S n = n^2 + 3*n - 3^n + 1) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_and_sum_l207_20745


namespace NUMINAMATH_CALUDE_watch_cost_price_l207_20746

-- Define the cost price of the watch
def cost_price : ℝ := 1166.67

-- Define the selling price at 10% loss
def selling_price_loss : ℝ := 0.90 * cost_price

-- Define the selling price at 2% gain
def selling_price_gain : ℝ := 1.02 * cost_price

-- Theorem statement
theorem watch_cost_price :
  (selling_price_loss = 0.90 * cost_price) ∧
  (selling_price_gain = 1.02 * cost_price) ∧
  (selling_price_gain = selling_price_loss + 140) →
  cost_price = 1166.67 := by
sorry

end NUMINAMATH_CALUDE_watch_cost_price_l207_20746


namespace NUMINAMATH_CALUDE_economics_and_law_tournament_l207_20747

theorem economics_and_law_tournament (n : ℕ) (m : ℕ) : 
  220 < n → n < 254 →
  m < n →
  (n - 2*m)^2 = n →
  ∀ k : ℕ, (220 < k ∧ k < 254 ∧ k < n ∧ (k - 2*(n-k))^2 = k) → n - m ≤ k - (n - k) →
  n - m = 105 :=
sorry

end NUMINAMATH_CALUDE_economics_and_law_tournament_l207_20747


namespace NUMINAMATH_CALUDE_book_selection_theorem_l207_20702

theorem book_selection_theorem :
  let mystery_count : ℕ := 4
  let fantasy_count : ℕ := 3
  let biography_count : ℕ := 3
  let different_genre_pairs : ℕ := 
    mystery_count * fantasy_count + 
    mystery_count * biography_count + 
    fantasy_count * biography_count
  different_genre_pairs = 33 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l207_20702


namespace NUMINAMATH_CALUDE_orange_distribution_l207_20722

/-- Given a number of oranges, pieces per orange, and number of friends,
    calculate the number of pieces each friend receives. -/
def pieces_per_friend (oranges : ℕ) (pieces_per_orange : ℕ) (friends : ℕ) : ℚ :=
  (oranges * pieces_per_orange : ℚ) / friends

/-- Theorem stating that given 80 oranges, each divided into 10 pieces,
    and 200 friends, each friend will receive 4 pieces. -/
theorem orange_distribution :
  pieces_per_friend 80 10 200 = 4 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_l207_20722


namespace NUMINAMATH_CALUDE_number_puzzle_l207_20726

theorem number_puzzle (x : ℝ) : (1/2 : ℝ) * x - 300 = 350 → (x + 200) * 2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l207_20726


namespace NUMINAMATH_CALUDE_yellow_beans_percentage_approx_32_percent_l207_20777

def bag1_total : ℕ := 24
def bag2_total : ℕ := 32
def bag3_total : ℕ := 34

def bag1_yellow_percent : ℚ := 40 / 100
def bag2_yellow_percent : ℚ := 30 / 100
def bag3_yellow_percent : ℚ := 25 / 100

def total_beans : ℕ := bag1_total + bag2_total + bag3_total

def yellow_beans : ℚ := 
  bag1_total * bag1_yellow_percent + 
  bag2_total * bag2_yellow_percent + 
  bag3_total * bag3_yellow_percent

def mixed_yellow_percent : ℚ := yellow_beans / total_beans

theorem yellow_beans_percentage_approx_32_percent :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |mixed_yellow_percent - 32/100| < ε :=
sorry

end NUMINAMATH_CALUDE_yellow_beans_percentage_approx_32_percent_l207_20777
