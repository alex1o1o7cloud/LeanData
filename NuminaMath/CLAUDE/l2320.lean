import Mathlib

namespace sector_area_l2320_232029

theorem sector_area (diameter : ℝ) (central_angle : ℝ) :
  diameter = 6 →
  central_angle = 120 →
  (π * (diameter / 2)^2 * central_angle / 360) = 3 * π :=
by sorry

end sector_area_l2320_232029


namespace no_solution_iff_n_eq_neg_half_l2320_232090

/-- The system of equations has no solution if and only if n = -1/2 -/
theorem no_solution_iff_n_eq_neg_half (n : ℝ) : 
  (∀ x y z : ℝ, ¬(2*n*x + y = 2 ∧ n*y + 2*z = 2 ∧ x + 2*n*z = 2)) ↔ n = -1/2 := by
  sorry

end no_solution_iff_n_eq_neg_half_l2320_232090


namespace negation_of_universal_proposition_l2320_232014

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by
  sorry

end negation_of_universal_proposition_l2320_232014


namespace min_difference_sine_extrema_l2320_232084

open Real

theorem min_difference_sine_extrema (f : ℝ → ℝ) (h : ∀ x, f x = 2 * sin x) :
  (∃ x₁ x₂, ∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂) →
  (∃ x₁ x₂, ∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂ ∧ |x₁ - x₂| = π) ∧
  (∀ x₁ x₂, (∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂) → |x₁ - x₂| ≥ π) :=
sorry

end min_difference_sine_extrema_l2320_232084


namespace circle_points_equidistant_l2320_232083

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point is on the circle if its distance from the center equals the radius -/
def IsOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem circle_points_equidistant (c : Circle) (p : ℝ × ℝ) :
  IsOnCircle c p → (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 := by
  sorry

end circle_points_equidistant_l2320_232083


namespace negation_equivalence_l2320_232057

-- Define a triangle
def Triangle : Type := Unit

-- Define an angle in a triangle
def Angle (t : Triangle) : Type := Unit

-- Define the property of being obtuse for an angle
def IsObtuse (t : Triangle) (a : Angle t) : Prop := sorry

-- Define the statement "at most one angle is obtuse"
def AtMostOneObtuse (t : Triangle) : Prop :=
  ∃ (a : Angle t), IsObtuse t a ∧ ∀ (b : Angle t), IsObtuse t b → b = a

-- Define the statement "at least two angles are obtuse"
def AtLeastTwoObtuse (t : Triangle) : Prop :=
  ∃ (a b : Angle t), a ≠ b ∧ IsObtuse t a ∧ IsObtuse t b

-- The theorem stating the negation equivalence
theorem negation_equivalence (t : Triangle) :
  ¬(AtMostOneObtuse t) ↔ AtLeastTwoObtuse t :=
sorry

end negation_equivalence_l2320_232057


namespace complex_product_real_l2320_232086

theorem complex_product_real (b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (∃ (r : ℝ), (2 + Complex.I) * (b + Complex.I) = r) →
  b = -2 := by
sorry

end complex_product_real_l2320_232086


namespace sum_of_roots_l2320_232095

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 12*a^2 + 47*a - 60 = 0)
  (hb : 8*b^3 - 48*b^2 + 18*b + 162 = 0) : 
  a + b = 3 := by
sorry

end sum_of_roots_l2320_232095


namespace product_one_sum_greater_than_reciprocals_l2320_232044

theorem product_one_sum_greater_than_reciprocals 
  (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_prod : a * b * c = 1) 
  (h_sum : a + b + c > 1/a + 1/b + 1/c) : 
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ 
  (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ 
  (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
sorry

end product_one_sum_greater_than_reciprocals_l2320_232044


namespace lindsey_october_savings_l2320_232096

/-- Represents the amount of money Lindsey saved in October -/
def october_savings : ℕ := 37

/-- Represents Lindsey's savings in September -/
def september_savings : ℕ := 50

/-- Represents Lindsey's savings in November -/
def november_savings : ℕ := 11

/-- Represents the amount Lindsey's mom gave her -/
def mom_gift : ℕ := 25

/-- Represents the cost of the video game -/
def video_game_cost : ℕ := 87

/-- Represents the amount Lindsey had left after buying the video game -/
def remaining_money : ℕ := 36

/-- Represents the condition that Lindsey saved more than $75 -/
def saved_more_than_75 : Prop :=
  september_savings + october_savings + november_savings > 75

theorem lindsey_october_savings : 
  september_savings + october_savings + november_savings + mom_gift - video_game_cost = remaining_money ∧
  saved_more_than_75 :=
sorry

end lindsey_october_savings_l2320_232096


namespace integer_solutions_l2320_232005

def is_solution (x y z : ℤ) : Prop :=
  x + y + z = 3 ∧ x^3 + y^3 + z^3 = 3

theorem integer_solutions :
  ∀ x y z : ℤ, is_solution x y z ↔ 
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨
     (x = 4 ∧ y = 4 ∧ z = -5) ∨
     (x = 4 ∧ y = -5 ∧ z = 4) ∨
     (x = -5 ∧ y = 4 ∧ z = 4)) :=
by sorry

end integer_solutions_l2320_232005


namespace tent_capacity_l2320_232065

/-- The number of seating sections in the circus tent -/
def num_sections : ℕ := 4

/-- The number of people each section can accommodate -/
def people_per_section : ℕ := 246

/-- The total number of people the tent can accommodate -/
def total_capacity : ℕ := num_sections * people_per_section

theorem tent_capacity : total_capacity = 984 := by
  sorry

end tent_capacity_l2320_232065


namespace expression_simplification_l2320_232099

theorem expression_simplification (b : ℝ) (h1 : b ≠ 1/2) (h2 : b ≠ 1) :
  1/2 - 1/(1 + b/(1 - 2*b)) = (3*b - 1)/(2*(1 - b)) := by
  sorry

end expression_simplification_l2320_232099


namespace whiteboard_numbers_l2320_232025

theorem whiteboard_numbers (n k : ℕ) : 
  n > 0 ∧ k > 0 ∧ k ≤ n ∧ 
  Odd n ∧
  (((n * (n + 1)) / 2 - k) : ℚ) / (n - 1) = 22 →
  n = 43 ∧ k = 22 := by
  sorry

end whiteboard_numbers_l2320_232025


namespace alcohol_solution_percentage_l2320_232006

theorem alcohol_solution_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_alcohol : ℝ) : 
  initial_volume = 6 →
  initial_percentage = 0.2 →
  added_alcohol = 3.6 →
  let final_volume := initial_volume + added_alcohol
  let initial_alcohol := initial_volume * initial_percentage
  let final_alcohol := initial_alcohol + added_alcohol
  let final_percentage := final_alcohol / final_volume
  final_percentage = 0.5 := by sorry

end alcohol_solution_percentage_l2320_232006


namespace quadratic_complete_square_l2320_232026

theorem quadratic_complete_square (x : ℝ) : ∃ (a k : ℝ), 
  3 * x^2 + 8 * x + 15 = a * (x - (-4/3))^2 + k :=
by sorry

end quadratic_complete_square_l2320_232026


namespace length_difference_l2320_232017

/-- Represents a rectangular plot. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ

/-- The cost of fencing per meter. -/
def fencingCostPerMeter : ℝ := 26.50

/-- The total cost of fencing the plot. -/
def totalFencingCost : ℝ := 5300

/-- Calculates the perimeter of a rectangular plot. -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth)

theorem length_difference (plot : RectangularPlot) :
  plot.length = 57 →
  perimeter plot = totalFencingCost / fencingCostPerMeter →
  plot.length - plot.breadth = 14 := by
  sorry

end length_difference_l2320_232017


namespace y₁_less_than_y₂_l2320_232041

/-- Linear function f(x) = 2x + 1 -/
def f (x : ℝ) : ℝ := 2 * x + 1

/-- y₁ is the value of f at x = -5 -/
def y₁ : ℝ := f (-5)

/-- y₂ is the value of f at x = 3 -/
def y₂ : ℝ := f 3

theorem y₁_less_than_y₂ : y₁ < y₂ := by
  sorry

end y₁_less_than_y₂_l2320_232041


namespace quadratic_roots_relation_l2320_232097

theorem quadratic_roots_relation (b c : ℚ) : 
  (∃ r s : ℚ, 4 * r^2 - 7 * r - 10 = 0 ∧ 4 * s^2 - 7 * s - 10 = 0 ∧
   ∀ x : ℚ, x^2 + b * x + c = 0 ↔ (x = r + 3 ∨ x = s + 3)) →
  c = 47 / 4 := by
sorry

end quadratic_roots_relation_l2320_232097


namespace courtyard_length_courtyard_length_is_25_l2320_232050

/-- Proves that the length of a rectangular courtyard is 25 meters -/
theorem courtyard_length : ℝ → ℝ → ℝ → ℝ → Prop :=
  λ (width : ℝ) (num_bricks : ℝ) (brick_length : ℝ) (brick_width : ℝ) =>
    width = 16 ∧
    num_bricks = 20000 ∧
    brick_length = 0.2 ∧
    brick_width = 0.1 →
    (num_bricks * brick_length * brick_width) / width = 25

/-- The length of the courtyard is 25 meters -/
theorem courtyard_length_is_25 :
  courtyard_length 16 20000 0.2 0.1 := by
  sorry

end courtyard_length_courtyard_length_is_25_l2320_232050


namespace rationalize_and_simplify_l2320_232021

theorem rationalize_and_simplify :
  ∃ (A B C D : ℕ), 
    (A * Real.sqrt B + C) / D = Real.sqrt 50 / (Real.sqrt 25 - Real.sqrt 5) ∧
    A * Real.sqrt B + C = 5 * Real.sqrt 2 + Real.sqrt 10 ∧
    D = 4 ∧
    A + B + C + D = 12 ∧
    ∀ (A' B' C' D' : ℕ), 
      (A' * Real.sqrt B' + C') / D' = Real.sqrt 50 / (Real.sqrt 25 - Real.sqrt 5) →
      A' + B' + C' + D' ≥ 12 := by
  sorry

end rationalize_and_simplify_l2320_232021


namespace square_difference_inapplicable_l2320_232036

/-- The square difference formula cannot be directly applied to (x-y)(-x+y) -/
theorem square_difference_inapplicable (x y : ℝ) :
  ¬ ∃ (a b : ℝ) (c₁ c₂ c₃ c₄ : ℝ), 
    (a = c₁ * x + c₂ * y ∧ b = c₃ * x + c₄ * y) ∧
    ((x - y) * (-x + y) = (a + b) * (a - b) ∨ (x - y) * (-x + y) = (a - b) * (a + b)) :=
by sorry

end square_difference_inapplicable_l2320_232036


namespace candy_soda_price_before_increase_l2320_232077

theorem candy_soda_price_before_increase 
  (candy_price_after : ℝ) 
  (soda_price_after : ℝ) 
  (candy_increase_rate : ℝ) 
  (soda_increase_rate : ℝ) 
  (h1 : candy_price_after = 15) 
  (h2 : soda_price_after = 6) 
  (h3 : candy_increase_rate = 0.25) 
  (h4 : soda_increase_rate = 0.5) : 
  candy_price_after / (1 + candy_increase_rate) + 
  soda_price_after / (1 + soda_increase_rate) = 21 := by
  sorry

#check candy_soda_price_before_increase

end candy_soda_price_before_increase_l2320_232077


namespace no_positive_integer_solutions_l2320_232004

theorem no_positive_integer_solutions :
  ∀ A : ℕ, 1 ≤ A → A ≤ 9 →
  ¬∃ x : ℕ, x > 0 ∧ x^2 - (10 * A + 1) * x + (10 * A + A) = 0 :=
by sorry

end no_positive_integer_solutions_l2320_232004


namespace team_a_construction_team_b_construction_l2320_232016

-- Define the parameters
def total_length_1 : ℝ := 600
def initial_days : ℝ := 5
def additional_days : ℝ := 2
def daily_increase : ℝ := 20
def total_length_2 : ℝ := 1800
def team_b_initial : ℝ := 360
def team_b_increase : ℝ := 0.2

-- Define Team A's daily construction after increase
def team_a_daily (x : ℝ) : ℝ := x

-- Define Team B's daily construction after increase
def team_b_daily (m : ℝ) : ℝ := m * (1 + team_b_increase)

-- Theorem for Team A's daily construction
theorem team_a_construction :
  ∃ x : ℝ, initial_days * (team_a_daily x - daily_increase) + additional_days * team_a_daily x = total_length_1 ∧
  team_a_daily x = 100 := by sorry

-- Theorem for Team B's original daily construction
theorem team_b_construction :
  ∃ m : ℝ, team_b_initial / m + (total_length_2 / 2 - team_b_initial) / (team_b_daily m) = total_length_2 / 2 / 100 ∧
  m = 90 := by sorry

end team_a_construction_team_b_construction_l2320_232016


namespace no_simultaneous_squares_l2320_232003

theorem no_simultaneous_squares (n : ℕ+) : ¬∃ (x y : ℕ+), (n + 1 : ℕ) = x^2 ∧ (4*n + 1 : ℕ) = y^2 := by
  sorry

end no_simultaneous_squares_l2320_232003


namespace base8_to_base10_157_l2320_232081

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The base-8 representation of the number --/
def base8Number : List Nat := [7, 5, 1]

theorem base8_to_base10_157 :
  base8ToBase10 base8Number = 111 := by
  sorry

end base8_to_base10_157_l2320_232081


namespace sequence_equation_l2320_232007

theorem sequence_equation (n : ℕ+) : 9 * (n - 1) + n = (n - 1) * 10 + 1 := by
  sorry

end sequence_equation_l2320_232007


namespace cube_tetrahedron_surface_area_ratio_l2320_232043

/-- The ratio of the surface area of a cube to the surface area of a regular tetrahedron
    formed by four vertices of the cube, given that the cube has side length 2. -/
theorem cube_tetrahedron_surface_area_ratio :
  let cube_side_length : ℝ := 2
  let tetrahedron_side_length : ℝ := 2 * Real.sqrt 2
  let cube_surface_area : ℝ := 6 * cube_side_length ^ 2
  let tetrahedron_surface_area : ℝ := Real.sqrt 3 * tetrahedron_side_length ^ 2
  cube_surface_area / tetrahedron_surface_area = Real.sqrt 3 := by
  sorry


end cube_tetrahedron_surface_area_ratio_l2320_232043


namespace negation_of_proposition_l2320_232027

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ x y : ℝ, x^2 + y^2 - 1 > 0)) ↔ (∃ x y : ℝ, x^2 + y^2 - 1 ≤ 0) :=
by sorry

end negation_of_proposition_l2320_232027


namespace one_book_selection_ways_l2320_232031

/-- The number of ways to take one book from a shelf with Chinese, math, and English books. -/
def ways_to_take_one_book (chinese_books math_books english_books : ℕ) : ℕ :=
  chinese_books + math_books + english_books

/-- Theorem: There are 37 ways to take one book from a shelf with 12 Chinese books, 14 math books, and 11 English books. -/
theorem one_book_selection_ways :
  ways_to_take_one_book 12 14 11 = 37 := by
  sorry

end one_book_selection_ways_l2320_232031


namespace jacksons_vacation_months_l2320_232080

/-- Proves that Jackson's vacation is 15 months away given his saving plan -/
theorem jacksons_vacation_months (total_savings : ℝ) (paychecks_per_month : ℕ) (savings_per_paycheck : ℝ)
  (h1 : total_savings = 3000)
  (h2 : paychecks_per_month = 2)
  (h3 : savings_per_paycheck = 100) :
  (total_savings / savings_per_paycheck) / paychecks_per_month = 15 := by
  sorry

end jacksons_vacation_months_l2320_232080


namespace quadratic_sum_equals_five_l2320_232018

theorem quadratic_sum_equals_five (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) : x^2 + y^2 = 5 := by
  sorry

end quadratic_sum_equals_five_l2320_232018


namespace inequality_proof_l2320_232028

theorem inequality_proof (u v w : ℝ) 
  (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h_sum : u + v + w + Real.sqrt (u * v * w) = 4) :
  Real.sqrt (u * v / w) + Real.sqrt (v * w / u) + Real.sqrt (w * u / v) ≥ u + v + w :=
by sorry

end inequality_proof_l2320_232028


namespace nominations_distribution_l2320_232024

/-- The number of ways to distribute nominations among schools -/
def distribute_nominations (total_nominations : ℕ) (num_schools : ℕ) : ℕ :=
  Nat.choose (total_nominations - num_schools + num_schools - 1) (num_schools - 1)

/-- Theorem stating that there are 84 ways to distribute 10 nominations among 7 schools -/
theorem nominations_distribution :
  distribute_nominations 10 7 = 84 := by
  sorry

end nominations_distribution_l2320_232024


namespace path_length_is_pi_l2320_232053

/-- Represents a rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the path length of a dot on the center of the top face when the prism is rolled -/
def pathLength (prism : RectangularPrism) : ℝ :=
  sorry

/-- Theorem stating that the path length for a 2x2x4 cm prism is π cm -/
theorem path_length_is_pi :
  let prism := RectangularPrism.mk 4 2 2
  pathLength prism = π :=
sorry

end path_length_is_pi_l2320_232053


namespace trivia_team_scoring_l2320_232012

/-- Trivia team scoring problem -/
theorem trivia_team_scoring
  (total_members : ℕ)
  (absent_members : ℕ)
  (total_points : ℕ)
  (h1 : total_members = 5)
  (h2 : absent_members = 2)
  (h3 : total_points = 18)
  : (total_points / (total_members - absent_members) = 6) :=
by
  sorry

#check trivia_team_scoring

end trivia_team_scoring_l2320_232012


namespace area_ratio_l2320_232010

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the lengths of the sides
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (30, 50, 54)

-- Define points D and E
def point_D (t : Triangle) : ℝ × ℝ := sorry
def point_E (t : Triangle) : ℝ × ℝ := sorry

-- Define the distances AD and AE
def dist_AD (t : Triangle) : ℝ := 21
def dist_AE (t : Triangle) : ℝ := 18

-- Define the areas of triangle ADE and quadrilateral BCED
def area_ADE (t : Triangle) : ℝ := sorry
def area_BCED (t : Triangle) : ℝ := sorry

-- State the theorem
theorem area_ratio (t : Triangle) :
  area_ADE t / area_BCED t = 49 / 51 := by sorry

end area_ratio_l2320_232010


namespace a_power_of_two_l2320_232069

def a : ℕ → ℕ
  | 0 => 0
  | n + 1 => 2 * a n + 2^n

theorem a_power_of_two (k : ℕ) : ∃ m : ℕ, a (2^k) = 2^m := by
  sorry

end a_power_of_two_l2320_232069


namespace P_divisibility_l2320_232079

/-- The polynomial P(x) defined in terms of a and b -/
def P (a b x : ℚ) : ℚ := (a + b) * x^5 + a * b * x^2 + 1

/-- The theorem stating the conditions for P(x) to be divisible by x^2 - 3x + 2 -/
theorem P_divisibility (a b : ℚ) : 
  (∀ x, (x^2 - 3*x + 2) ∣ P a b x) ↔ 
  ((a = -1 ∧ b = 31/28) ∨ (a = 31/28 ∧ b = -1)) :=
sorry

end P_divisibility_l2320_232079


namespace six_balls_three_boxes_l2320_232092

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distributeBalls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 122 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distributeBalls 6 3 = 122 := by sorry

end six_balls_three_boxes_l2320_232092


namespace transportation_cost_independent_of_order_l2320_232000

/-- Represents a destination with its distance from the city and the weight of goods to be delivered -/
structure Destination where
  distance : ℝ
  weight : ℝ
  weight_eq_distance : weight = distance

/-- Calculates the cost of transportation for a single trip -/
def transportCost (d : Destination) (extraDistance : ℝ) : ℝ :=
  d.weight * (d.distance + extraDistance)

/-- Theorem stating that the total transportation cost is independent of the order of visits -/
theorem transportation_cost_independent_of_order (m n : Destination) :
  transportCost m 0 + transportCost n m.distance =
  transportCost n 0 + transportCost m n.distance := by
  sorry

#check transportation_cost_independent_of_order

end transportation_cost_independent_of_order_l2320_232000


namespace max_value_of_a_l2320_232011

theorem max_value_of_a (a : ℝ) :
  (∀ x : ℝ, x < a → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ x ≥ a) →
  a ≤ -1 :=
sorry

end max_value_of_a_l2320_232011


namespace farmer_brown_additional_cost_farmer_brown_specific_additional_cost_l2320_232058

/-- The additional cost for Farmer Brown's new hay requirements -/
theorem farmer_brown_additional_cost 
  (original_bales : ℕ) 
  (multiplier : ℕ) 
  (original_cost_per_bale : ℕ) 
  (premium_cost_per_bale : ℕ) : ℕ :=
  let new_bales := original_bales * multiplier
  let original_total_cost := original_bales * original_cost_per_bale
  let new_total_cost := new_bales * premium_cost_per_bale
  new_total_cost - original_total_cost

/-- The additional cost for Farmer Brown's specific hay requirements is $3500 -/
theorem farmer_brown_specific_additional_cost :
  farmer_brown_additional_cost 20 5 25 40 = 3500 := by
  sorry

end farmer_brown_additional_cost_farmer_brown_specific_additional_cost_l2320_232058


namespace age_relation_l2320_232038

/-- Proves that A was twice as old as B 10 years ago given the conditions -/
theorem age_relation (b_age : ℕ) (a_age : ℕ) (x : ℕ) : 
  b_age = 42 →
  a_age = b_age + 12 →
  a_age + 10 = 2 * (b_age - x) →
  x = 10 := by
  sorry

end age_relation_l2320_232038


namespace arithmetic_sequence_100th_term_l2320_232072

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = a n + 4

theorem arithmetic_sequence_100th_term (a : ℕ → ℕ) 
  (h : arithmetic_sequence a) : a 100 = 397 := by
  sorry

end arithmetic_sequence_100th_term_l2320_232072


namespace rice_containers_l2320_232082

theorem rice_containers (total_weight : ℚ) (container_weight : ℕ) (pound_to_ounce : ℕ) : 
  total_weight = 29/4 →
  container_weight = 29 →
  pound_to_ounce = 16 →
  (total_weight * pound_to_ounce / container_weight : ℚ) = 4 := by
  sorry

end rice_containers_l2320_232082


namespace james_tshirts_l2320_232042

/-- Calculates the number of t-shirts bought given the discount rate, original price, and total amount paid -/
def tshirts_bought (discount_rate : ℚ) (original_price : ℚ) (total_paid : ℚ) : ℚ :=
  total_paid / (original_price * (1 - discount_rate))

/-- Proves that James bought 6 t-shirts -/
theorem james_tshirts :
  let discount_rate : ℚ := 1/2
  let original_price : ℚ := 20
  let total_paid : ℚ := 60
  tshirts_bought discount_rate original_price total_paid = 6 := by
sorry

end james_tshirts_l2320_232042


namespace quadratic_roots_contradiction_l2320_232013

theorem quadratic_roots_contradiction (a : ℝ) : 
  a ≥ 1 → ¬(∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + a = 0 ∧ y^2 - 2*y + a = 0) :=
by
  sorry


end quadratic_roots_contradiction_l2320_232013


namespace intersection_of_M_and_N_l2320_232085

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| < 2}
def N : Set ℝ := {x | x * (x - 3) < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 0 < x ∧ x < 3} := by sorry

end intersection_of_M_and_N_l2320_232085


namespace special_isosceles_triangle_angles_l2320_232046

/-- An isosceles triangle with a special angle bisector property -/
structure SpecialIsoscelesTriangle where
  -- The base angles of the isosceles triangle
  base_angle : ℝ
  -- The angle between the angle bisector from the vertex and the angle bisector to the lateral side
  bisector_angle : ℝ
  -- The condition that the bisector angle equals the vertex angle
  h_bisector_eq_vertex : bisector_angle = 180 - 2 * base_angle

/-- The possible angles of a special isosceles triangle -/
def special_triangle_angles (t : SpecialIsoscelesTriangle) : Prop :=
  (t.base_angle = 36 ∧ 180 - 2 * t.base_angle = 108) ∨
  (t.base_angle = 60 ∧ 180 - 2 * t.base_angle = 60)

/-- Theorem: The angles of a special isosceles triangle are either (36°, 36°, 108°) or (60°, 60°, 60°) -/
theorem special_isosceles_triangle_angles (t : SpecialIsoscelesTriangle) :
  special_triangle_angles t := by
  sorry

end special_isosceles_triangle_angles_l2320_232046


namespace complex_arithmetic_result_l2320_232002

theorem complex_arithmetic_result :
  let B : ℂ := 3 + 2*I
  let Q : ℂ := -3
  let T : ℂ := 2*I
  let U : ℂ := 1 + 5*I
  2*B - Q + 3*T + U = 10 + 15*I :=
by sorry

end complex_arithmetic_result_l2320_232002


namespace sqrt_200_equals_10_l2320_232059

theorem sqrt_200_equals_10 : Real.sqrt 200 = 10 := by
  sorry

end sqrt_200_equals_10_l2320_232059


namespace max_value_of_sum_of_roots_l2320_232040

theorem max_value_of_sum_of_roots (x : ℝ) (h : 3 < x ∧ x < 6) :
  ∃ (k : ℝ), k = Real.sqrt 6 ∧ ∀ y : ℝ, (Real.sqrt (x - 3) + Real.sqrt (6 - x) ≤ y) → y ≥ k :=
sorry

end max_value_of_sum_of_roots_l2320_232040


namespace average_physics_math_l2320_232054

/-- Given the scores of three subjects, prove the average of two specific subjects -/
theorem average_physics_math (total_average : ℝ) (physics_chem_average : ℝ) (physics_score : ℝ) : 
  total_average = 60 →
  physics_chem_average = 70 →
  physics_score = 140 →
  (physics_score + (3 * total_average - physics_score - 
    (2 * physics_chem_average - physics_score))) / 2 = 90 := by
  sorry


end average_physics_math_l2320_232054


namespace vector_linear_combination_l2320_232087

/-- Given vectors a, b, and c in ℝ², prove that c = 2a - b -/
theorem vector_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 2)) 
  (hb : b = (-2, 3)) 
  (hc : c = (4, 1)) : 
  c = 2 • a - b := by sorry

end vector_linear_combination_l2320_232087


namespace cone_sphere_ratio_l2320_232091

theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) : 
  (1 / 3 * π * r^2 * h = 1 / 3 * (4 / 3 * π * r^3)) → h / r = 4 / 3 := by
  sorry

end cone_sphere_ratio_l2320_232091


namespace quadratic_real_roots_range_l2320_232001

theorem quadratic_real_roots_range (k : ℝ) : 
  (∃ x : ℝ, (k + 2) * x^2 - 2 * x - 1 = 0) ↔ (k ≥ -3 ∧ k ≠ -2) :=
sorry

end quadratic_real_roots_range_l2320_232001


namespace no_solution_condition_l2320_232022

theorem no_solution_condition (k : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (k * x) / (x - 1) - (2 * k - 1) / (1 - x) ≠ 2) ↔ 
  (k = 1/3 ∨ k = 2) :=
sorry

end no_solution_condition_l2320_232022


namespace square_area_ratio_l2320_232035

theorem square_area_ratio (r : ℝ) (h : r > 0) : 
  (4 * r^2) / (2 * r^2) = 2 := by sorry

end square_area_ratio_l2320_232035


namespace characterization_of_divisibility_implication_l2320_232061

theorem characterization_of_divisibility_implication (n : ℕ) (hn : n ≥ 1) :
  (∀ a b : ℕ, (11 ∣ a^n + b^n) → (11 ∣ a ∧ 11 ∣ b)) ↔ Even n :=
by sorry

end characterization_of_divisibility_implication_l2320_232061


namespace q_at_zero_l2320_232076

-- Define polynomials p, q, and r
variable (p q r : ℝ[X])

-- Define the relationship between p, q, and r
axiom poly_product : r = p * q

-- Define the constant terms of p and r
axiom p_constant : p.coeff 0 = 5
axiom r_constant : r.coeff 0 = -10

-- Theorem to prove
theorem q_at_zero : q.eval 0 = -2 := by
  sorry

end q_at_zero_l2320_232076


namespace domain_all_reals_l2320_232015

theorem domain_all_reals (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (3 * k * x^2 - 4 * x + 7) / (-7 * x^2 - 4 * x + k)) ↔ 
  k < -4/7 :=
sorry

end domain_all_reals_l2320_232015


namespace square_of_fraction_l2320_232078

theorem square_of_fraction (a b c : ℝ) (hc : c ≠ 0) :
  ((-2 * a^2 * b) / (3 * c))^2 = (4 * a^4 * b^2) / (9 * c^2) := by
  sorry

end square_of_fraction_l2320_232078


namespace jacksons_running_distance_l2320_232019

/-- Calculates the final daily running distance after a given number of weeks,
    starting from an initial distance and increasing by a fixed amount each week. -/
def finalRunningDistance (initialDistance : ℕ) (weeklyIncrease : ℕ) (totalWeeks : ℕ) : ℕ :=
  initialDistance + weeklyIncrease * (totalWeeks - 1)

/-- Proves that Jackson's final daily running distance is 7 miles
    after 5 weeks of training. -/
theorem jacksons_running_distance :
  finalRunningDistance 3 1 5 = 7 := by
  sorry

end jacksons_running_distance_l2320_232019


namespace sin_transformation_equivalence_l2320_232060

theorem sin_transformation_equivalence (x : ℝ) :
  let f (x : ℝ) := Real.sin x
  let g (x : ℝ) := Real.sin (2*x - π/5)
  let transform1 (x : ℝ) := Real.sin (2*(x - π/5))
  let transform2 (x : ℝ) := Real.sin (2*(x - π/10))
  (∀ x, g x = transform1 x) ∧ (∀ x, g x = transform2 x) :=
by sorry

end sin_transformation_equivalence_l2320_232060


namespace expression_equality_l2320_232023

theorem expression_equality : (50 - (5020 - 520)) + (5020 - (520 - 50)) = 100 := by
  sorry

end expression_equality_l2320_232023


namespace power_multiplication_l2320_232070

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by sorry

end power_multiplication_l2320_232070


namespace polygon_sides_when_angles_equal_l2320_232051

theorem polygon_sides_when_angles_equal : ∀ n : ℕ,
  n > 2 →
  (n - 2) * 180 = 360 →
  n = 4 :=
by
  sorry

end polygon_sides_when_angles_equal_l2320_232051


namespace lcm_problem_l2320_232030

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 60 := by
  sorry

end lcm_problem_l2320_232030


namespace min_value_3a_3b_l2320_232009

theorem min_value_3a_3b (a b : ℝ) (h : a * b = 2) : 3 * a + 3 * b ≥ 6 := by
  sorry

end min_value_3a_3b_l2320_232009


namespace inequality_system_solution_l2320_232073

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (4 + x) / 3 > (x + 2) / 2 ∧ (x + a) / 2 < 0 ↔ x < 2) →
  a ≤ -2 :=
by sorry

end inequality_system_solution_l2320_232073


namespace book_price_increase_l2320_232034

/-- Given a book with an original price and a percentage increase, 
    calculate the new price after the increase. -/
theorem book_price_increase (original_price : ℝ) (percent_increase : ℝ) 
  (h1 : original_price = 300)
  (h2 : percent_increase = 10) : 
  original_price * (1 + percent_increase / 100) = 330 := by
  sorry

end book_price_increase_l2320_232034


namespace max_sum_of_product_107_l2320_232071

theorem max_sum_of_product_107 (a b : ℤ) (h : a * b = 107) :
  ∃ (c d : ℤ), c * d = 107 ∧ c + d ≥ a + b ∧ c + d = 108 :=
by sorry

end max_sum_of_product_107_l2320_232071


namespace at_least_one_greater_than_one_l2320_232094

theorem at_least_one_greater_than_one (x y : ℝ) (h : x + y > 2) :
  x > 1 ∨ y > 1 := by
  sorry

end at_least_one_greater_than_one_l2320_232094


namespace flower_beds_count_l2320_232020

theorem flower_beds_count (total_seeds : ℕ) (seeds_per_bed : ℕ) (h1 : total_seeds = 270) (h2 : seeds_per_bed = 9) :
  total_seeds / seeds_per_bed = 30 := by
  sorry

end flower_beds_count_l2320_232020


namespace min_value_of_f_inequality_theorem_l2320_232088

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Theorem for the minimum value of f
theorem min_value_of_f : ∃ (p : ℝ), ∀ (x : ℝ), f x ≥ p ∧ ∃ (x₀ : ℝ), f x₀ = p :=
  sorry

-- Theorem for the inequality
theorem inequality_theorem (a b c : ℝ) (h : a^2 + 2*b^2 + 3*c^2 = 6) :
  |a + 2*b + 3*c| ≤ 6 :=
  sorry

end min_value_of_f_inequality_theorem_l2320_232088


namespace fraction_equality_l2320_232045

theorem fraction_equality (x y : ℚ) (hx : x = 4/6) (hy : y = 8/10) :
  (6 * x^2 + 10 * y) / (60 * x * y) = 11/36 := by
  sorry

end fraction_equality_l2320_232045


namespace average_weight_increase_l2320_232032

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 4 →
  old_weight = 95 →
  new_weight = 129 →
  (new_weight - old_weight) / initial_count = 8.5 :=
by
  sorry

end average_weight_increase_l2320_232032


namespace square_side_length_l2320_232064

theorem square_side_length (w h r s : ℕ) : 
  w = 4000 →
  h = 2300 →
  2 * r + s = h →
  2 * r + 3 * s = w →
  s = 850 :=
by sorry

end square_side_length_l2320_232064


namespace seeking_cause_is_necessary_condition_l2320_232063

/-- "Seeking the cause from the effect" in analytical proof -/
def seeking_cause_from_effect : Prop := sorry

/-- Necessary condition in a proposition -/
def necessary_condition : Prop := sorry

/-- Theorem stating that "seeking the cause from the effect" refers to seeking the necessary condition -/
theorem seeking_cause_is_necessary_condition : 
  seeking_cause_from_effect ↔ necessary_condition := by sorry

end seeking_cause_is_necessary_condition_l2320_232063


namespace line_slope_intercept_product_l2320_232075

/-- Given a line passing through the points (-2, -3) and (3, 4),
    the product of its slope and y-intercept is equal to -7/25. -/
theorem line_slope_intercept_product : 
  let p₁ : ℝ × ℝ := (-2, -3)
  let p₂ : ℝ × ℝ := (3, 4)
  let m : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)  -- slope
  let b : ℝ := p₁.2 - m * p₁.1  -- y-intercept
  m * b = -7/25 :=
by sorry

end line_slope_intercept_product_l2320_232075


namespace statement_a_statement_b_statements_a_and_b_correct_l2320_232068

-- Statement A
theorem statement_a (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a + c > b + c := by
  sorry

-- Statement B
theorem statement_b (a b : ℝ) (h1 : a > b) (h2 : b > 0) : (a + b) / 2 > Real.sqrt (a * b) := by
  sorry

-- Combined theorem for A and B
theorem statements_a_and_b_correct :
  (∀ (a b c : ℝ), a > b → c < 0 → a + c > b + c) ∧
  (∀ (a b : ℝ), a > b → b > 0 → (a + b) / 2 > Real.sqrt (a * b)) := by
  sorry

end statement_a_statement_b_statements_a_and_b_correct_l2320_232068


namespace rhombus_symmetry_proposition_l2320_232048

-- Define the set of all rhombuses
variable (Rhombus : Type)

-- Define the property of having central symmetry
variable (has_central_symmetry : Rhombus → Prop)

-- Define the universal quantifier proposition
def universal_proposition : Prop := ∀ r : Rhombus, has_central_symmetry r

-- Define the negation of the proposition
def negation_proposition : Prop := ∃ r : Rhombus, ¬has_central_symmetry r

-- Theorem stating that the original proposition is a universal quantifier
-- and its negation is an existential quantifier with negated property
theorem rhombus_symmetry_proposition :
  (universal_proposition Rhombus has_central_symmetry) ∧
  (negation_proposition Rhombus has_central_symmetry) :=
sorry

end rhombus_symmetry_proposition_l2320_232048


namespace initial_speed_is_4_l2320_232093

/-- Represents the scenario of a person walking to a bus stand -/
structure BusScenario where
  distance : ℝ  -- Distance to the bus stand in km
  faster_speed : ℝ  -- Speed at which the person arrives early (km/h)
  early_time : ℝ  -- Time arrived early when walking at faster_speed (minutes)
  late_time : ℝ  -- Time arrived late when walking at initial speed (minutes)

/-- Calculates the initial walking speed given a BusScenario -/
def initial_speed (scenario : BusScenario) : ℝ :=
  sorry

/-- Theorem stating that the initial walking speed is 4 km/h for the given scenario -/
theorem initial_speed_is_4 (scenario : BusScenario) 
  (h1 : scenario.distance = 5)
  (h2 : scenario.faster_speed = 5)
  (h3 : scenario.early_time = 5)
  (h4 : scenario.late_time = 10) :
  initial_speed scenario = 4 :=
sorry

end initial_speed_is_4_l2320_232093


namespace range_of_x_when_f_positive_l2320_232033

/-- A linear function obtained by translating y = x upwards by 2 units -/
def f (x : ℝ) : ℝ := x + 2

/-- The range of x when f(x) > 0 -/
theorem range_of_x_when_f_positive : 
  {x : ℝ | f x > 0} = {x : ℝ | x > -2} := by
  sorry

end range_of_x_when_f_positive_l2320_232033


namespace dartboard_angle_l2320_232098

theorem dartboard_angle (probability : ℝ) (angle : ℝ) : 
  probability = 1/4 → angle = 90 := by
  sorry

end dartboard_angle_l2320_232098


namespace inequality_range_l2320_232052

theorem inequality_range (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π/2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) 
  ↔ 
  (a ≤ Real.sqrt 6 ∨ a ≥ 7/2) :=
sorry

end inequality_range_l2320_232052


namespace A_must_be_four_l2320_232067

/-- Represents a six-digit number in the form 32BA33 -/
def SixDigitNumber (A : Nat) : Nat :=
  320000 + A * 100 + 33

/-- Rounds a number to the nearest hundred -/
def roundToNearestHundred (n : Nat) : Nat :=
  ((n + 50) / 100) * 100

/-- Theorem stating that if 32BA33 rounds to 323400, then A must be 4 -/
theorem A_must_be_four :
  ∀ A : Nat, A < 10 →
  roundToNearestHundred (SixDigitNumber A) = 323400 →
  A = 4 := by
sorry

end A_must_be_four_l2320_232067


namespace milly_science_homework_time_l2320_232037

/-- The time Milly spends studying various subjects -/
structure StudyTime where
  math : ℕ
  geography : ℕ
  science : ℕ
  total : ℕ

/-- Milly's study time satisfies the given conditions -/
def millysStudyTime : StudyTime where
  math := 60
  geography := 30
  science := 45
  total := 135

theorem milly_science_homework_time :
  ∀ (st : StudyTime),
    st.math = 60 →
    st.geography = st.math / 2 →
    st.total = 135 →
    st.science = st.total - st.math - st.geography →
    st.science = 45 := by
  sorry

end milly_science_homework_time_l2320_232037


namespace two_digit_seven_times_sum_of_digits_l2320_232047

theorem two_digit_seven_times_sum_of_digits : 
  (∃! (s : Finset Nat), 
    (∀ n ∈ s, 10 ≤ n ∧ n < 100 ∧ n = 7 * (n / 10 + n % 10)) ∧ 
    Finset.card s = 4) := by
  sorry

end two_digit_seven_times_sum_of_digits_l2320_232047


namespace fraction_of_seniors_studying_japanese_l2320_232074

theorem fraction_of_seniors_studying_japanese 
  (num_juniors : ℝ) 
  (num_seniors : ℝ) 
  (fraction_juniors_studying : ℝ) 
  (fraction_total_studying : ℝ) :
  num_seniors = 3 * num_juniors →
  fraction_juniors_studying = 3 / 4 →
  fraction_total_studying = 0.4375 →
  (fraction_total_studying * (num_juniors + num_seniors) - fraction_juniors_studying * num_juniors) / num_seniors = 1 / 3 :=
by sorry

end fraction_of_seniors_studying_japanese_l2320_232074


namespace kylie_jewelry_beads_l2320_232039

/-- The number of beads Kylie uses in total to make her jewelry over the week -/
def total_beads : ℕ :=
  let necklace_beads := 20
  let bracelet_beads := 10
  let earring_beads := 5
  let anklet_beads := 8
  let ring_beads := 7
  let monday_necklaces := 10
  let tuesday_necklaces := 2
  let wednesday_bracelets := 5
  let thursday_earrings := 3
  let friday_anklets := 4
  let friday_rings := 6
  (necklace_beads * (monday_necklaces + tuesday_necklaces)) +
  (bracelet_beads * wednesday_bracelets) +
  (earring_beads * thursday_earrings) +
  (anklet_beads * friday_anklets) +
  (ring_beads * friday_rings)

theorem kylie_jewelry_beads : total_beads = 379 := by
  sorry

end kylie_jewelry_beads_l2320_232039


namespace empty_set_problem_l2320_232062

-- Define the sets
def set_A : Set ℝ := {x | x^2 - 4 = 0}
def set_B : Set ℝ := {x | x > 9 ∨ x < 3}
def set_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 0}
def set_D : Set ℝ := {x | x > 9 ∧ x < 3}

-- Theorem statement
theorem empty_set_problem :
  (set_A ≠ ∅) ∧ (set_B ≠ ∅) ∧ (set_C ≠ ∅) ∧ (set_D = ∅) :=
sorry

end empty_set_problem_l2320_232062


namespace prime_pair_fraction_integer_l2320_232066

theorem prime_pair_fraction_integer :
  ∀ p q : ℕ,
    Prime p → Prime q → p > q →
    (∃ n : ℤ, (((p + q : ℕ)^(p + q) * (p - q : ℕ)^(p - q) - 1) : ℤ) = 
              n * (((p + q : ℕ)^(p - q) * (p - q : ℕ)^(p + q) - 1) : ℤ)) →
    p = 3 ∧ q = 2 := by
sorry

end prime_pair_fraction_integer_l2320_232066


namespace x_value_l2320_232049

def M (x : ℝ) : Set ℝ := {2, 0, x}
def N : Set ℝ := {0, 1}

theorem x_value : ∀ x : ℝ, N ⊆ M x → x = 1 := by
  sorry

end x_value_l2320_232049


namespace a_collinear_b_l2320_232056

/-- Two 2D vectors are collinear if and only if their cross product is zero -/
def collinear (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 = 0

/-- The vector a -/
def a : ℝ × ℝ := (1, 2)

/-- The vector b -/
def b : ℝ × ℝ := (-1, -2)

/-- Proof that vectors a and b are collinear -/
theorem a_collinear_b : collinear a b := by
  sorry

end a_collinear_b_l2320_232056


namespace fourth_square_area_l2320_232055

-- Define the triangles and their properties
structure Triangle (X Y Z : ℝ × ℝ) where
  is_right : (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0

-- Define the theorem
theorem fourth_square_area 
  (XYZ : Triangle X Y Z) 
  (XZW : Triangle X Z W) 
  (square1_area : ℝ) 
  (square2_area : ℝ) 
  (square3_area : ℝ) 
  (h1 : square1_area = 25) 
  (h2 : square2_area = 4) 
  (h3 : square3_area = 49) : 
  ∃ (fourth_square_area : ℝ), fourth_square_area = 78 := by
  sorry

end fourth_square_area_l2320_232055


namespace greatest_partition_size_l2320_232008

theorem greatest_partition_size (m n p : ℕ) (h_m : m > 0) (h_n : n > 0) (h_p : Nat.Prime p) :
  ∃ (s : ℕ), s > 0 ∧ s ≤ m ∧
  ∀ (t : ℕ), t > s →
    ¬∃ (partition : Fin (t * n * p) → Fin t),
      ∀ (i : Fin t),
        ∃ (r : ℕ),
          ∀ (j k : Fin (t * n * p)),
            partition j = i → partition k = i →
              (j.val + k.val) % p = r :=
by sorry

end greatest_partition_size_l2320_232008


namespace jason_debt_l2320_232089

def mowing_value (hour : ℕ) : ℕ :=
  match hour % 3 with
  | 1 => 3
  | 2 => 5
  | 0 => 7
  | _ => 0

def total_earnings (hours : ℕ) : ℕ :=
  (List.range hours).map mowing_value |>.sum

theorem jason_debt : total_earnings 25 = 123 := by
  sorry

end jason_debt_l2320_232089
