import Mathlib

namespace distance_between_trees_l1536_153627

/-- Given a yard of length 375 meters with 26 trees planted at equal distances,
    with one tree at each end, the distance between two consecutive trees is 15 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) 
    (h1 : yard_length = 375)
    (h2 : num_trees = 26) :
    yard_length / (num_trees - 1) = 15 := by
  sorry

end distance_between_trees_l1536_153627


namespace clown_balloons_l1536_153646

/-- The number of additional balloons blown up by the clown -/
def additional_balloons (initial final : ℕ) : ℕ := final - initial

/-- Theorem: Given the initial and final number of balloons, prove that the clown blew up 13 more balloons -/
theorem clown_balloons : additional_balloons 47 60 = 13 := by
  sorry

end clown_balloons_l1536_153646


namespace sweater_price_calculation_l1536_153612

/-- Given the price of shirts and the price difference between shirts and sweaters,
    calculate the total price of sweaters. -/
theorem sweater_price_calculation (shirt_total : ℕ) (shirt_count : ℕ) (sweater_count : ℕ)
    (price_difference : ℕ) (h1 : shirt_total = 360) (h2 : shirt_count = 20)
    (h3 : sweater_count = 45) (h4 : price_difference = 2) :
    let shirt_avg : ℚ := shirt_total / shirt_count
    let sweater_avg : ℚ := shirt_avg + price_difference
    sweater_avg * sweater_count = 900 := by
  sorry

end sweater_price_calculation_l1536_153612


namespace students_not_enrolled_l1536_153660

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ)
  (h1 : total = 94)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 40 := by
  sorry

end students_not_enrolled_l1536_153660


namespace compound_interest_rate_l1536_153636

/-- Compound interest calculation --/
theorem compound_interest_rate (P : ℝ) (t : ℝ) (interest : ℝ) (h1 : P > 0) (h2 : t > 0) (h3 : interest > 0) :
  let A := P + interest
  let n := 1
  let r := (((A / P) ^ (1 / (n * t))) - 1) * n
  r = 0.1 :=
by sorry

end compound_interest_rate_l1536_153636


namespace equal_numbers_l1536_153634

theorem equal_numbers (x : Fin 100 → ℝ) 
  (h : ∀ i : Fin 100, (x i)^3 + x (i + 1) = (x (i + 1))^3 + x (i + 2)) : 
  ∀ i j : Fin 100, x i = x j := by
  sorry

end equal_numbers_l1536_153634


namespace arithmetic_sequence_value_l1536_153686

/-- 
Given that -7, a, and 1 form an arithmetic sequence, prove that a = -3.
-/
theorem arithmetic_sequence_value (a : ℝ) : 
  (∃ d : ℝ, a - (-7) = d ∧ 1 - a = d) → a = -3 := by
  sorry

end arithmetic_sequence_value_l1536_153686


namespace linda_car_rental_cost_l1536_153664

/-- Calculates the total cost of renting a car given the daily rate, mileage rate, number of days, and miles driven. -/
def total_rental_cost (daily_rate : ℚ) (mileage_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mileage_rate * miles

/-- Proves that the total cost for Linda's car rental is $165. -/
theorem linda_car_rental_cost :
  total_rental_cost 30 0.25 3 300 = 165 := by
  sorry

end linda_car_rental_cost_l1536_153664


namespace probability_win_first_two_given_earn_3_l1536_153606

def win_probability : ℝ := 0.6

def points_for_win (sets_won : ℕ) : ℕ :=
  if sets_won = 3 then 3 else if sets_won = 2 then 2 else 0

def points_for_loss (sets_lost : ℕ) : ℕ :=
  if sets_lost = 2 then 1 else 0

def prob_win_3_0 : ℝ := win_probability ^ 3

def prob_win_3_1 : ℝ := 3 * (win_probability ^ 2) * (1 - win_probability) * win_probability

def prob_earn_3_points : ℝ := prob_win_3_0 + prob_win_3_1

def prob_win_first_two_and_earn_3 : ℝ := 
  prob_win_3_0 + (win_probability ^ 2) * (1 - win_probability) * win_probability

theorem probability_win_first_two_given_earn_3 :
  prob_win_first_two_and_earn_3 / prob_earn_3_points = 7 / 11 := by
  sorry

end probability_win_first_two_given_earn_3_l1536_153606


namespace kalebs_clothing_l1536_153682

def total_clothing (first_load : ℕ) (num_equal_loads : ℕ) (pieces_per_equal_load : ℕ) : ℕ :=
  first_load + num_equal_loads * pieces_per_equal_load

theorem kalebs_clothing :
  total_clothing 19 5 4 = 39 := by
  sorry

end kalebs_clothing_l1536_153682


namespace total_steel_parts_l1536_153647

/-- Represents the number of machines of type A -/
def a : ℕ := sorry

/-- Represents the number of machines of type B -/
def b : ℕ := sorry

/-- The total number of machines -/
def total_machines : ℕ := 21

/-- The total number of chrome parts -/
def total_chrome_parts : ℕ := 66

/-- Steel parts in a type A machine -/
def steel_parts_A : ℕ := 3

/-- Chrome parts in a type A machine -/
def chrome_parts_A : ℕ := 2

/-- Steel parts in a type B machine -/
def steel_parts_B : ℕ := 2

/-- Chrome parts in a type B machine -/
def chrome_parts_B : ℕ := 4

theorem total_steel_parts :
  a + b = total_machines ∧
  chrome_parts_A * a + chrome_parts_B * b = total_chrome_parts →
  steel_parts_A * a + steel_parts_B * b = 51 := by
  sorry

end total_steel_parts_l1536_153647


namespace equality_comparison_l1536_153641

theorem equality_comparison : 
  (2^3 ≠ 6) ∧ 
  (-1^2 ≠ (-1)^2) ∧ 
  (-2^3 = (-2)^3) ∧ 
  (4^2 / 9 ≠ (4/9)^2) :=
by sorry

end equality_comparison_l1536_153641


namespace pets_after_one_month_l1536_153669

/-- Calculates the number of pets in an animal shelter after one month --/
theorem pets_after_one_month
  (initial_dogs : ℕ)
  (initial_cats : ℕ)
  (initial_lizards : ℕ)
  (dog_adoption_rate : ℚ)
  (cat_adoption_rate : ℚ)
  (lizard_adoption_rate : ℚ)
  (new_pets : ℕ)
  (h_dogs : initial_dogs = 30)
  (h_cats : initial_cats = 28)
  (h_lizards : initial_lizards = 20)
  (h_dog_rate : dog_adoption_rate = 1/2)
  (h_cat_rate : cat_adoption_rate = 1/4)
  (h_lizard_rate : lizard_adoption_rate = 1/5)
  (h_new_pets : new_pets = 13) :
  ↑initial_dogs + ↑initial_cats + ↑initial_lizards -
  (↑initial_dogs * dog_adoption_rate + ↑initial_cats * cat_adoption_rate + ↑initial_lizards * lizard_adoption_rate) +
  ↑new_pets = 65 := by
  sorry


end pets_after_one_month_l1536_153669


namespace converse_square_right_angles_false_l1536_153699

-- Define a quadrilateral
structure Quadrilateral :=
  (is_right_angled : Bool)
  (is_square : Bool)

-- Define the property that all angles are right angles
def all_angles_right (q : Quadrilateral) : Prop :=
  q.is_right_angled = true

-- Define the property of being a square
def is_square (q : Quadrilateral) : Prop :=
  q.is_square = true

-- Theorem: The converse of "All four angles of a square are right angles" is false
theorem converse_square_right_angles_false :
  ¬ (∀ q : Quadrilateral, all_angles_right q → is_square q) :=
by sorry

end converse_square_right_angles_false_l1536_153699


namespace square_39_relation_l1536_153692

theorem square_39_relation : (39 : ℕ)^2 = (40 : ℕ)^2 - 79 := by
  sorry

end square_39_relation_l1536_153692


namespace circle_area_sum_l1536_153600

theorem circle_area_sum : 
  let radius : ℕ → ℝ := λ n => 2 * (1/3)^(n-1)
  let area : ℕ → ℝ := λ n => π * (radius n)^2
  let series_sum : ℝ := ∑' n, area n
  series_sum = 9*π/2 := by
sorry

end circle_area_sum_l1536_153600


namespace shaded_area_circular_pattern_l1536_153672

/-- The area of the shaded region in a circular arc pattern -/
theorem shaded_area_circular_pattern (r : ℝ) (l : ℝ) : 
  r = 3 → l = 24 → (2 * l / (2 * r)) * (π * r^2 / 2) = 18 * π :=
by
  sorry

end shaded_area_circular_pattern_l1536_153672


namespace line_tangent_to_parabola_l1536_153676

/-- The line 4x + 3y + k = 0 is tangent to the parabola y² = 16x if and only if k = 9 -/
theorem line_tangent_to_parabola (k : ℝ) : 
  (∀ x y : ℝ, 4*x + 3*y + k = 0 → y^2 = 16*x) ↔ k = 9 := by
  sorry

end line_tangent_to_parabola_l1536_153676


namespace price_increase_achieves_target_profit_l1536_153651

/-- Represents the supermarket's pomelo sales scenario -/
structure PomeloSales where
  initial_profit_per_kg : ℝ
  initial_daily_sales : ℝ
  price_increase : ℝ
  sales_decrease_per_yuan : ℝ
  target_daily_profit : ℝ

/-- Calculates the daily profit based on the price increase -/
def daily_profit (s : PomeloSales) : ℝ :=
  (s.initial_profit_per_kg + s.price_increase) *
  (s.initial_daily_sales - s.sales_decrease_per_yuan * s.price_increase)

/-- Theorem stating that a 5 yuan price increase achieves the target profit -/
theorem price_increase_achieves_target_profit (s : PomeloSales)
  (h1 : s.initial_profit_per_kg = 10)
  (h2 : s.initial_daily_sales = 500)
  (h3 : s.sales_decrease_per_yuan = 20)
  (h4 : s.target_daily_profit = 6000)
  (h5 : s.price_increase = 5) :
  daily_profit s = s.target_daily_profit :=
by sorry


end price_increase_achieves_target_profit_l1536_153651


namespace all_terms_irrational_l1536_153648

theorem all_terms_irrational (a : ℕ → ℝ) 
  (h_pos : ∀ k, a k > 0)
  (h_rel : ∀ k, (a (k + 1) + k) * a k = 1) :
  ∀ k, Irrational (a k) := by
sorry

end all_terms_irrational_l1536_153648


namespace gcd_217_155_l1536_153684

theorem gcd_217_155 : Nat.gcd 217 155 = 1 := by
  sorry

end gcd_217_155_l1536_153684


namespace tangent_line_at_origin_l1536_153670

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^2 - x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 4 * x - 1

-- Theorem statement
theorem tangent_line_at_origin : 
  ∀ x y : ℝ, (x + y = 0) ↔ (∃ t : ℝ, y = f t ∧ y - f 0 = f' 0 * (x - 0)) := by
  sorry

end tangent_line_at_origin_l1536_153670


namespace geometric_sequence_iff_square_middle_l1536_153608

/-- A sequence (a, b, c) is geometric if there exists a common ratio r such that b = ar and c = br. -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- The main theorem stating that b² = ac is necessary and sufficient for (a, b, c) to form a geometric sequence. -/
theorem geometric_sequence_iff_square_middle (a b c : ℝ) :
  IsGeometricSequence a b c ↔ b^2 = a * c :=
sorry

end geometric_sequence_iff_square_middle_l1536_153608


namespace midpoint_locus_of_constant_area_segment_l1536_153605

/-- Given a line segment PQ with endpoints on the parabola y = x^2 such that the area bounded by PQ
    and the parabola is always 4/3, the locus of the midpoint M of PQ is described by the equation
    y = x^2 + 1 -/
theorem midpoint_locus_of_constant_area_segment (P Q : ℝ × ℝ) :
  (∃ α β : ℝ, α < β ∧ 
    P = (α, α^2) ∧ Q = (β, β^2) ∧ 
    (∫ (x : ℝ) in α..β, ((β - α)⁻¹ * ((β * α^2 - α * β^2) + (β - α) * x) - x^2)) = 4/3) →
  let M := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  M.2 = M.1^2 + 1 := by
sorry


end midpoint_locus_of_constant_area_segment_l1536_153605


namespace infinite_functions_satisfying_condition_l1536_153657

theorem infinite_functions_satisfying_condition :
  ∃ (S : Set (ℝ → ℝ)), (Set.Infinite S) ∧ 
  (∀ f ∈ S, 2 * f 3 - 10 = f 1) := by
sorry

end infinite_functions_satisfying_condition_l1536_153657


namespace least_subtraction_for_divisibility_problem_solution_l1536_153622

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  let n := 5474827
  let d := 12
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 ∧ k = 7 :=
by
  sorry

end least_subtraction_for_divisibility_problem_solution_l1536_153622


namespace geometric_sequence_a6_l1536_153689

def is_increasing_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 1 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  is_increasing_geometric_sequence a →
  a 1 + a 4 = 9 →
  a 2 * a 3 = 8 →
  a 6 = 32 := by
sorry

end geometric_sequence_a6_l1536_153689


namespace letter_distribution_l1536_153639

theorem letter_distribution (n : ℕ) (k : ℕ) : 
  n = 4 ∧ k = 3 → k^n = 81 := by
  sorry

end letter_distribution_l1536_153639


namespace fundraising_goal_l1536_153631

/-- Fundraising goal calculation for a school's community outreach program -/
theorem fundraising_goal (families_20 families_10 families_5 : ℕ) 
  (donation_20 donation_10 donation_5 : ℕ) (additional_needed : ℕ) : 
  families_20 = 2 → 
  families_10 = 8 → 
  families_5 = 10 → 
  donation_20 = 20 → 
  donation_10 = 10 → 
  donation_5 = 5 → 
  additional_needed = 30 → 
  families_20 * donation_20 + families_10 * donation_10 + families_5 * donation_5 + additional_needed = 200 := by
sorry

#eval 2 * 20 + 8 * 10 + 10 * 5 + 30

end fundraising_goal_l1536_153631


namespace committee_formations_count_l1536_153601

/-- Represents a department in the division of mathematical sciences -/
inductive Department
| Mathematics
| Statistics
| ComputerScience

/-- Represents the gender of a professor -/
inductive Gender
| Male
| Female

/-- Represents a professor with their department and gender -/
structure Professor :=
  (department : Department)
  (gender : Gender)

/-- The total number of departments -/
def num_departments : Nat := 3

/-- The number of male professors in each department -/
def male_professors_per_dept : Nat := 3

/-- The number of female professors in each department -/
def female_professors_per_dept : Nat := 3

/-- The total number of professors in the committee -/
def committee_size : Nat := 8

/-- The number of male professors required in the committee -/
def male_professors_in_committee : Nat := 4

/-- The number of female professors required in the committee -/
def female_professors_in_committee : Nat := 4

/-- Calculates the number of ways to form a committee satisfying all conditions -/
def count_committee_formations : Nat :=
  sorry

/-- Theorem stating that the number of possible committee formations is 59049 -/
theorem committee_formations_count :
  count_committee_formations = 59049 := by sorry

end committee_formations_count_l1536_153601


namespace difference_of_squares_l1536_153661

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 8) :
  x^2 - y^2 = 160 := by sorry

end difference_of_squares_l1536_153661


namespace area_of_ring_area_of_specific_ring_l1536_153618

/-- The area of a ring formed by two concentric circles -/
theorem area_of_ring (r₁ r₂ : ℝ) (h : r₁ > r₂) : 
  (π * r₁^2 - π * r₂^2) = π * (r₁^2 - r₂^2) := by sorry

/-- The area of the ring between two concentric circles with radii 15 and 9 -/
theorem area_of_specific_ring : 
  (π * 15^2 - π * 9^2) = 144 * π := by sorry

end area_of_ring_area_of_specific_ring_l1536_153618


namespace grandfather_grandson_age_ratio_not_six_l1536_153690

theorem grandfather_grandson_age_ratio_not_six : 
  let grandson_age_now : ℕ := 12
  let grandfather_age_now : ℕ := 72
  let grandson_age_three_years_ago : ℕ := grandson_age_now - 3
  let grandfather_age_three_years_ago : ℕ := grandfather_age_now - 3
  ¬ (grandfather_age_three_years_ago = 6 * grandson_age_three_years_ago) :=
by sorry

end grandfather_grandson_age_ratio_not_six_l1536_153690


namespace largest_fraction_l1536_153677

theorem largest_fraction : 
  let fractions := [5/12, 7/15, 23/45, 89/178, 199/400]
  ∀ x ∈ fractions, (23/45 : ℚ) ≥ x := by
  sorry

end largest_fraction_l1536_153677


namespace garland_arrangement_l1536_153675

theorem garland_arrangement (blue : Nat) (red : Nat) (white : Nat) :
  blue = 8 →
  red = 7 →
  white = 12 →
  (Nat.choose (blue + red) blue) * (Nat.choose (blue + red + 1) white) = 11711700 :=
by sorry

end garland_arrangement_l1536_153675


namespace garden_width_l1536_153653

/-- A rectangular garden with specific dimensions. -/
structure RectangularGarden where
  width : ℝ
  length : ℝ
  perimeter_eq : width + length = 30
  length_eq : length = width + 8

/-- The width of the garden is 11 feet. -/
theorem garden_width (g : RectangularGarden) : g.width = 11 := by
  sorry

#check garden_width

end garden_width_l1536_153653


namespace pool_width_proof_l1536_153650

theorem pool_width_proof (drain_rate : ℝ) (drain_time : ℝ) (length : ℝ) (depth : ℝ) 
  (h1 : drain_rate = 60)
  (h2 : drain_time = 2000)
  (h3 : length = 150)
  (h4 : depth = 10) :
  drain_rate * drain_time / (length * depth) = 80 :=
by sorry

end pool_width_proof_l1536_153650


namespace nh4i_equilibrium_constant_l1536_153614

/-- Equilibrium constant for a chemical reaction --/
def equilibrium_constant (c_nh3 c_hi : ℝ) : ℝ := c_nh3 * c_hi

/-- Concentration of HI produced from NH₄I decomposition --/
def c_hi_from_nh4i (c_hi c_h2 : ℝ) : ℝ := c_hi + 2 * c_h2

theorem nh4i_equilibrium_constant (c_h2 c_hi : ℝ) 
  (h1 : c_h2 = 1) 
  (h2 : c_hi = 4) :
  equilibrium_constant (c_hi_from_nh4i c_hi c_h2) c_hi = 24 := by
  sorry

end nh4i_equilibrium_constant_l1536_153614


namespace flowers_in_pot_l1536_153687

theorem flowers_in_pot (chrysanthemums : ℕ) (roses : ℕ) : 
  chrysanthemums = 5 → roses = 2 → chrysanthemums + roses = 7 := by
  sorry

end flowers_in_pot_l1536_153687


namespace expansion_coefficients_l1536_153667

-- Define the coefficient of x in the expansion
def S (n : ℕ) : ℚ := (n + 1 : ℚ) / (2 * (Nat.factorial (n - 1)))

-- Define the ratio of coefficients T_n / S_n
def T_S_ratio (n : ℕ) : ℚ := (1 / 4 : ℚ) * n^2 - (1 / 12 : ℚ) * n - (1 / 6 : ℚ)

-- Theorem statement
theorem expansion_coefficients (n : ℕ) (h : n ≥ 2) : 
  S n = (n + 1 : ℚ) / (2 * (Nat.factorial (n - 1))) ∧ 
  T_S_ratio n = (1 / 4 : ℚ) * n^2 - (1 / 12 : ℚ) * n - (1 / 6 : ℚ) := by
  sorry

end expansion_coefficients_l1536_153667


namespace product_zero_implies_factor_zero_l1536_153609

theorem product_zero_implies_factor_zero (a b c : ℝ) : a * b * c = 0 → (a = 0 ∨ b = 0 ∨ c = 0) := by
  sorry

end product_zero_implies_factor_zero_l1536_153609


namespace furniture_store_problem_l1536_153652

/-- Furniture store problem -/
theorem furniture_store_problem 
  (a : ℝ) 
  (table_price : ℝ → ℝ) 
  (chair_price : ℝ → ℝ) 
  (table_retail : ℝ) 
  (chair_retail : ℝ) 
  (set_price : ℝ) 
  (h1 : table_price a = a) 
  (h2 : chair_price a = a - 140) 
  (h3 : table_retail = 380) 
  (h4 : chair_retail = 160) 
  (h5 : set_price = 940) 
  (h6 : 600 / (a - 140) = 1300 / a) 
  (x : ℝ) 
  (h7 : x + 5 * x + 20 ≤ 200) 
  (profit : ℝ → ℝ) 
  (h8 : profit x = (set_price - table_price a - 4 * chair_price a) * (1/2 * x) + 
                   (table_retail - table_price a) * (1/2 * x) + 
                   (chair_retail - chair_price a) * (5 * x + 20 - 4 * (1/2 * x))) :
  a = 260 ∧ 
  (∃ (max_x : ℝ), max_x = 30 ∧ 
    (∀ y, y + 5 * y + 20 ≤ 200 → profit y ≤ profit max_x) ∧ 
    profit max_x = 9200) := by
  sorry

end furniture_store_problem_l1536_153652


namespace tan_problem_l1536_153683

theorem tan_problem (α β : Real) 
  (h1 : Real.tan (π/4 + α) = 2) 
  (h2 : Real.tan β = 1/2) : 
  Real.tan α = 1/3 ∧ 
  (Real.sin (α + β) - 2 * Real.sin α * Real.cos β) / 
  (2 * Real.sin α * Real.sin β + Real.cos (α + β)) = 1/7 := by
  sorry

end tan_problem_l1536_153683


namespace equal_cost_sharing_l1536_153668

theorem equal_cost_sharing (X Y Z : ℝ) (h : Y > X) :
  let total_cost := X + Y + Z
  let equal_share := total_cost / 2
  let nina_payment := equal_share - Y
  nina_payment = (X + Z - Y) / 2 := by
  sorry

end equal_cost_sharing_l1536_153668


namespace parking_lot_cars_l1536_153607

theorem parking_lot_cars : ∃ (total : ℕ), 
  (total / 3 : ℚ) + (total / 2 : ℚ) + 86 = total ∧ total = 516 := by
  sorry

end parking_lot_cars_l1536_153607


namespace linear_inequality_condition_l1536_153617

theorem linear_inequality_condition (m : ℝ) : 
  (|m - 3| = 1 ∧ m - 4 ≠ 0) ↔ m = 2 := by sorry

end linear_inequality_condition_l1536_153617


namespace factorization1_factorization2_factorization3_l1536_153674

-- Given formulas
axiom formula1 (x a b : ℝ) : (x + a) * (x + b) = x^2 + (a + b) * x + a * b
axiom formula2 (x y : ℝ) : (x + y)^2 + 2 * (x + y) + 1 = (x + y + 1)^2

-- Theorems to prove
theorem factorization1 (x : ℝ) : x^2 + 4 * x + 3 = (x + 3) * (x + 1) := by sorry

theorem factorization2 (x y : ℝ) : (x - y)^2 - 10 * (x - y) + 25 = (x - y - 5)^2 := by sorry

theorem factorization3 (m : ℝ) : (m^2 - 2 * m) * (m^2 - 2 * m + 4) + 3 = (m^2 - 2 * m + 3) * (m - 1)^2 := by sorry

end factorization1_factorization2_factorization3_l1536_153674


namespace equal_cost_guests_l1536_153642

def caesars_cost (guests : ℕ) : ℚ := 800 + 30 * guests
def venus_cost (guests : ℕ) : ℚ := 500 + 35 * guests

theorem equal_cost_guests : ∃ (x : ℕ), caesars_cost x = venus_cost x ∧ x = 60 := by
  sorry

end equal_cost_guests_l1536_153642


namespace jimin_has_greater_sum_l1536_153688

theorem jimin_has_greater_sum : 
  let jungkook_num1 : ℕ := 4
  let jungkook_num2 : ℕ := 4
  let jimin_num1 : ℕ := 3
  let jimin_num2 : ℕ := 6
  jimin_num1 + jimin_num2 > jungkook_num1 + jungkook_num2 :=
by
  sorry

end jimin_has_greater_sum_l1536_153688


namespace bachuan_jiaoqing_extrema_l1536_153691

/-- Definition of a "Bachuan Jiaoqing password number" -/
def is_bachuan_jiaoqing (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  1000 ≤ n ∧ n < 10000 ∧ b ≥ c ∧ a = b + c ∧ d = b - c

/-- Additional divisibility condition -/
def satisfies_divisibility (n : ℕ) : Prop :=
  let a := n / 1000
  let bcd := n % 1000
  (bcd - 7 * a) % 13 = 0

/-- Theorem stating the largest and smallest "Bachuan Jiaoqing password numbers" -/
theorem bachuan_jiaoqing_extrema :
  (∀ n, is_bachuan_jiaoqing n → n ≤ 9909) ∧
  (∃ n, is_bachuan_jiaoqing n ∧ satisfies_divisibility n ∧
    ∀ m, is_bachuan_jiaoqing m ∧ satisfies_divisibility m → n ≤ m) ∧
  (is_bachuan_jiaoqing 9909) ∧
  (is_bachuan_jiaoqing 5321 ∧ satisfies_divisibility 5321) := by
  sorry

end bachuan_jiaoqing_extrema_l1536_153691


namespace zero_exponent_l1536_153603

theorem zero_exponent (x : ℝ) (hx : x ≠ 0) : x^0 = 1 := by
  sorry

end zero_exponent_l1536_153603


namespace max_value_of_sum_product_l1536_153693

theorem max_value_of_sum_product (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 200 → 
  a * b + a * c + a * d ≤ 10000 :=
by sorry

end max_value_of_sum_product_l1536_153693


namespace simplify_fraction_l1536_153680

theorem simplify_fraction :
  (5 : ℝ) / (Real.sqrt 50 + Real.sqrt 32 + 3 * Real.sqrt 18) = (5 * Real.sqrt 2) / 36 := by
  sorry

end simplify_fraction_l1536_153680


namespace probability_sum_nine_l1536_153604

/-- A standard die with six faces -/
def Die : Type := Fin 6

/-- The sample space of rolling a die twice -/
def SampleSpace : Type := Die × Die

/-- The event of getting a sum of 9 -/
def SumNine (outcome : SampleSpace) : Prop :=
  (outcome.1.val + 1) + (outcome.2.val + 1) = 9

/-- The number of favorable outcomes (sum of 9) -/
def FavorableOutcomes : ℕ := 4

/-- The total number of possible outcomes -/
def TotalOutcomes : ℕ := 36

/-- The probability of getting a sum of 9 -/
def ProbabilitySumNine : ℚ := FavorableOutcomes / TotalOutcomes

theorem probability_sum_nine :
  ProbabilitySumNine = 1 / 9 := by
  sorry

end probability_sum_nine_l1536_153604


namespace tan_alpha_equals_two_l1536_153613

theorem tan_alpha_equals_two (α : ℝ) 
  (h : Real.sin (2 * α + Real.pi / 4) - 7 * Real.sin (2 * α + 3 * Real.pi / 4) = 5 * Real.sqrt 2) : 
  Real.tan α = 2 := by
  sorry

end tan_alpha_equals_two_l1536_153613


namespace stratified_sample_size_is_72_l1536_153696

/-- Represents the number of teachers in each category -/
structure TeacherCounts where
  fullProf : Nat
  assocProf : Nat
  lecturers : Nat
  teachingAssistants : Nat

/-- Calculates the total number of teachers -/
def totalTeachers (counts : TeacherCounts) : Nat :=
  counts.fullProf + counts.assocProf + counts.lecturers + counts.teachingAssistants

/-- Calculates the sample size for stratified sampling -/
def stratifiedSampleSize (counts : TeacherCounts) (lecturersDrawn : Nat) : Nat :=
  let samplingRate := lecturersDrawn / counts.lecturers
  (totalTeachers counts) * samplingRate

/-- Theorem: Given the specific teacher counts and 16 lecturers drawn, 
    the stratified sample size is 72 -/
theorem stratified_sample_size_is_72 
  (counts : TeacherCounts) 
  (h1 : counts.fullProf = 120) 
  (h2 : counts.assocProf = 100) 
  (h3 : counts.lecturers = 80) 
  (h4 : counts.teachingAssistants = 60) 
  (h5 : stratifiedSampleSize counts 16 = 72) : 
  stratifiedSampleSize counts 16 = 72 := by
  sorry

#eval stratifiedSampleSize 
  { fullProf := 120, assocProf := 100, lecturers := 80, teachingAssistants := 60 } 16

end stratified_sample_size_is_72_l1536_153696


namespace min_sum_with_reciprocal_constraint_l1536_153681

theorem min_sum_with_reciprocal_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 2/b = 2) : 
  a + b ≥ 3/2 + Real.sqrt 2 := by
sorry

end min_sum_with_reciprocal_constraint_l1536_153681


namespace right_triangle_set_l1536_153685

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The theorem stating that only one set of numbers forms a right triangle --/
theorem right_triangle_set :
  ¬(is_right_triangle 0.1 0.2 0.3) ∧
  ¬(is_right_triangle 1 1 2) ∧
  is_right_triangle 10 24 26 ∧
  ¬(is_right_triangle 9 16 25) :=
sorry

end right_triangle_set_l1536_153685


namespace exist_expression_24_set1_exist_expression_24_set2_l1536_153671

-- Define a type for arithmetic operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a type for arithmetic expressions
inductive Expr
  | Num (n : ℕ)
  | BinOp (op : Operation) (e1 e2 : Expr)

-- Define a function to evaluate expressions
def eval : Expr → ℚ
  | Expr.Num n => n
  | Expr.BinOp Operation.Add e1 e2 => eval e1 + eval e2
  | Expr.BinOp Operation.Sub e1 e2 => eval e1 - eval e2
  | Expr.BinOp Operation.Mul e1 e2 => eval e1 * eval e2
  | Expr.BinOp Operation.Div e1 e2 => eval e1 / eval e2

-- Define a function to check if an expression uses all given numbers exactly once
def usesAllNumbers (e : Expr) (nums : List ℕ) : Prop := sorry

-- Theorem for the first set of numbers
theorem exist_expression_24_set1 :
  ∃ (e : Expr), usesAllNumbers e [7, 12, 9, 12] ∧ eval e = 24 := by sorry

-- Theorem for the second set of numbers
theorem exist_expression_24_set2 :
  ∃ (e : Expr), usesAllNumbers e [3, 9, 5, 9] ∧ eval e = 24 := by sorry

end exist_expression_24_set1_exist_expression_24_set2_l1536_153671


namespace linear_diophantine_equation_solutions_l1536_153635

theorem linear_diophantine_equation_solutions
  (a b c : ℤ) (x₀ y₀ : ℤ) (h : a * x₀ + b * y₀ = c) :
  ∀ x y : ℤ, a * x + b * y = c ↔ ∃ k : ℤ, x = x₀ + k * b ∧ y = y₀ - k * a :=
by sorry

end linear_diophantine_equation_solutions_l1536_153635


namespace inequality_solution_l1536_153698

theorem inequality_solution (x : ℝ) : 2 ≤ x / (2 * x - 4) ∧ x / (2 * x - 4) < 7 ↔ x ∈ Set.Ici 2 ∩ Set.Iio (28 / 13) :=
sorry

end inequality_solution_l1536_153698


namespace hyperbola_mn_value_l1536_153638

/-- Given a hyperbola with equation x²/m - y²/n = 1, eccentricity 2, and one focus at (1,0), prove that mn = 3/16 -/
theorem hyperbola_mn_value (m n : ℝ) (h1 : m * n ≠ 0) :
  (∀ x y : ℝ, x^2 / m - y^2 / n = 1) →  -- Hyperbola equation
  (∃ a b : ℝ, (x - a)^2 / m - (y - b)^2 / n = 1 ∧ ((a + 1)^2 + b^2)^(1/2) = 2) →  -- Eccentricity is 2
  (∃ x y : ℝ, x^2 / m - y^2 / n = 1 ∧ x = 1 ∧ y = 0) →  -- One focus at (1,0)
  m * n = 3/16 :=
by sorry

end hyperbola_mn_value_l1536_153638


namespace triangle_area_l1536_153697

/-- A triangle with sides 8, 15, and 17 has an area of 60 -/
theorem triangle_area : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c area =>
    a = 8 ∧ b = 15 ∧ c = 17 →
    area = 60

/-- The proof of the theorem -/
lemma prove_triangle_area : triangle_area 8 15 17 60 := by
  sorry

end triangle_area_l1536_153697


namespace green_flash_count_l1536_153633

def total_time : ℕ := 671
def green_interval : ℕ := 3
def red_interval : ℕ := 5
def blue_interval : ℕ := 7

def green_flashes : ℕ := total_time / green_interval
def green_red_flashes : ℕ := total_time / (Nat.lcm green_interval red_interval)
def green_blue_flashes : ℕ := total_time / (Nat.lcm green_interval blue_interval)
def all_color_flashes : ℕ := total_time / (Nat.lcm green_interval (Nat.lcm red_interval blue_interval))

theorem green_flash_count : 
  green_flashes - green_red_flashes - green_blue_flashes + all_color_flashes = 154 := by
  sorry

end green_flash_count_l1536_153633


namespace triangular_pyramid_projections_not_equal_l1536_153621

/-- Represents a three-dimensional solid object -/
structure Solid :=
  (shape : Type)

/-- Represents an orthogonal projection (view) of a solid -/
structure Projection :=
  (shape : Type)
  (size : ℝ)

/-- Returns the front projection of a solid -/
def front_view (s : Solid) : Projection :=
  sorry

/-- Returns the top projection of a solid -/
def top_view (s : Solid) : Projection :=
  sorry

/-- Returns the side projection of a solid -/
def side_view (s : Solid) : Projection :=
  sorry

/-- Defines a triangular pyramid -/
def triangular_pyramid : Solid :=
  sorry

/-- Theorem stating that a triangular pyramid cannot have all three
    orthogonal projections of the same shape and size -/
theorem triangular_pyramid_projections_not_equal :
  ∃ (p1 p2 : Projection), 
    (p1 = front_view triangular_pyramid ∧
     p2 = top_view triangular_pyramid ∧
     p1 ≠ p2) ∨
    (p1 = front_view triangular_pyramid ∧
     p2 = side_view triangular_pyramid ∧
     p1 ≠ p2) ∨
    (p1 = top_view triangular_pyramid ∧
     p2 = side_view triangular_pyramid ∧
     p1 ≠ p2) :=
  sorry

end triangular_pyramid_projections_not_equal_l1536_153621


namespace friend_team_assignment_count_l1536_153632

-- Define the number of friends and teams
def num_friends : ℕ := 6
def num_teams : ℕ := 4

-- Theorem statement
theorem friend_team_assignment_count :
  (num_teams ^ num_friends : ℕ) = 4096 := by
  sorry

end friend_team_assignment_count_l1536_153632


namespace candle_ratio_l1536_153620

/-- Proves that the ratio of candles in Kalani's bedroom to candles in the living room is 2:1 -/
theorem candle_ratio :
  ∀ (bedroom_candles living_room_candles donovan_candles total_candles : ℕ),
    bedroom_candles = 20 →
    donovan_candles = 20 →
    total_candles = 50 →
    bedroom_candles + living_room_candles + donovan_candles = total_candles →
    (bedroom_candles : ℚ) / living_room_candles = 2 := by
  sorry

end candle_ratio_l1536_153620


namespace det_dilation_matrix_5_l1536_153659

/-- A 2x2 matrix representing a dilation with scale factor k centered at the origin -/
def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

/-- Theorem: The determinant of a 2x2 dilation matrix with scale factor 5 is 25 -/
theorem det_dilation_matrix_5 :
  let E := dilation_matrix 5
  Matrix.det E = 25 := by sorry

end det_dilation_matrix_5_l1536_153659


namespace base7_312_equals_base4_2310_l1536_153679

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a number from base 10 to base 4 -/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base7_312_equals_base4_2310 :
  base10ToBase4 (base7ToBase10 [2, 1, 3]) = [2, 3, 1, 0] := by
  sorry

end base7_312_equals_base4_2310_l1536_153679


namespace min_value_sum_l1536_153666

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + y ≥ 16 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 1 ∧ x + y = 16 :=
by sorry

end min_value_sum_l1536_153666


namespace brothers_sisters_ratio_l1536_153663

theorem brothers_sisters_ratio :
  ∀ (num_brothers : ℕ),
    (num_brothers + 2) * 2 = 12 →
    num_brothers / 2 = 1 :=
by
  sorry

end brothers_sisters_ratio_l1536_153663


namespace sweater_markup_percentage_l1536_153626

/-- Given a sweater with wholesale cost and retail price, proves the markup percentage. -/
theorem sweater_markup_percentage 
  (W : ℝ) -- Wholesale cost
  (R : ℝ) -- Normal retail price
  (h1 : W > 0) -- Wholesale cost is positive
  (h2 : R > 0) -- Retail price is positive
  (h3 : 0.4 * R = 1.2 * W) -- Condition for 60% discount and 20% profit
  : (R - W) / W * 100 = 200 :=
by sorry

end sweater_markup_percentage_l1536_153626


namespace min_teachers_for_given_counts_l1536_153665

/-- Represents the number of teachers for each subject -/
structure TeacherCounts where
  english : Nat
  history : Nat
  geography : Nat

/-- Calculates the minimum number of teachers required to cover all subjects -/
def minTeachersRequired (counts : TeacherCounts) : Nat :=
  sorry

/-- Theorem stating the minimum number of teachers required for the given conditions -/
theorem min_teachers_for_given_counts :
  let counts : TeacherCounts := { english := 9, history := 7, geography := 6 }
  minTeachersRequired counts = 10 := by
  sorry

end min_teachers_for_given_counts_l1536_153665


namespace cyclist_speed_proof_l1536_153662

/-- The speed of the east-bound cyclist in mph -/
def east_speed : ℝ := 18

/-- The speed of the west-bound cyclist in mph -/
def west_speed : ℝ := east_speed + 4

/-- The time traveled in hours -/
def time : ℝ := 5

/-- The total distance between the cyclists after the given time -/
def total_distance : ℝ := 200

theorem cyclist_speed_proof :
  east_speed * time + west_speed * time = total_distance :=
by sorry

end cyclist_speed_proof_l1536_153662


namespace angle_x_is_180_l1536_153615

-- Define the geometric configuration
structure GeometricConfiguration where
  -- Triangle ABC
  angle_ABC : Real
  angle_ACB : Real
  -- Straight angles
  angle_ADC_straight : Bool
  angle_AEB_straight : Bool

-- Theorem statement
theorem angle_x_is_180 (config : GeometricConfiguration) 
  (h1 : config.angle_ABC = 50)
  (h2 : config.angle_ACB = 70)
  (h3 : config.angle_ADC_straight = true)
  (h4 : config.angle_AEB_straight = true) :
  ∃ x : Real, x = 180 := by
  sorry

end angle_x_is_180_l1536_153615


namespace sphere_radius_from_depression_l1536_153643

theorem sphere_radius_from_depression (r : ℝ) 
  (depression_depth : ℝ) (depression_diameter : ℝ) : 
  depression_depth = 8 ∧ 
  depression_diameter = 24 ∧ 
  r^2 = (r - depression_depth)^2 + (depression_diameter / 2)^2 → 
  r = 13 := by
  sorry

end sphere_radius_from_depression_l1536_153643


namespace range_of_a_l1536_153616

-- Define the sets A, B, and C
def A : Set ℝ := {x | 0 < 2*x + 4 ∧ 2*x + 4 < 10}
def B : Set ℝ := {x | x < -4 ∨ x > 2}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0}

-- Define the union of A and B
def AUB : Set ℝ := {x | x ∈ A ∨ x ∈ B}

-- Define the complement of A ∪ B
def comp_AUB : Set ℝ := {x | x ∉ AUB}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ comp_AUB → x ∈ C a) → -2 < a ∧ a < -4/3 := by sorry

end range_of_a_l1536_153616


namespace investment_profit_distribution_l1536_153623

/-- Represents the business investment scenario -/
structure BusinessInvestment where
  total_contribution : ℝ
  a_duration : ℝ
  b_duration : ℝ
  a_final : ℝ
  b_final : ℝ
  a_contribution : ℝ
  b_contribution : ℝ

/-- Theorem stating that the given contributions satisfy the profit distribution -/
theorem investment_profit_distribution (investment : BusinessInvestment)
  (h1 : investment.total_contribution = 1500)
  (h2 : investment.a_duration = 3)
  (h3 : investment.b_duration = 4)
  (h4 : investment.a_final = 690)
  (h5 : investment.b_final = 1080)
  (h6 : investment.a_contribution = 600)
  (h7 : investment.b_contribution = 900)
  (h8 : investment.a_contribution + investment.b_contribution = investment.total_contribution) :
  (investment.a_final - investment.a_contribution) / (investment.b_final - investment.b_contribution) =
  (investment.a_duration * investment.a_contribution) / (investment.b_duration * investment.b_contribution) :=
by sorry

end investment_profit_distribution_l1536_153623


namespace range_of_p_l1536_153695

/-- The set A of real numbers x satisfying the quadratic equation x^2 + (p+2)x + 1 = 0 -/
def A (p : ℝ) : Set ℝ := {x | x^2 + (p+2)*x + 1 = 0}

/-- The theorem stating the range of p given the conditions -/
theorem range_of_p (p : ℝ) (h : A p ∩ Set.Ici (0 : ℝ) = ∅) : p > -4 :=
sorry

end range_of_p_l1536_153695


namespace set_c_is_well_defined_l1536_153624

-- Define the universe of discourse
def Student : Type := sorry

-- Define the properties
def SeniorHighSchool (s : Student) : Prop := sorry
def EnrolledAtDudeSchool (s : Student) : Prop := sorry
def EnrolledInJanuary2013 (s : Student) : Prop := sorry

-- Define the set C
def SetC : Set Student :=
  {s : Student | SeniorHighSchool s ∧ EnrolledAtDudeSchool s ∧ EnrolledInJanuary2013 s}

-- Define the property of being well-defined
def WellDefined (S : Set Student) : Prop :=
  ∀ s : Student, s ∈ S → (∃ (criterion : Student → Prop), criterion s)

-- Theorem statement
theorem set_c_is_well_defined :
  WellDefined SetC ∧ 
  (∀ S : Set Student, S ≠ SetC → ¬WellDefined S) :=
sorry

end set_c_is_well_defined_l1536_153624


namespace hua_luogeng_birthday_factorization_l1536_153678

theorem hua_luogeng_birthday_factorization (h : 19101112 = 1163 * 16424) :
  Nat.Prime 1163 ∧ ¬ Nat.Prime 16424 := by
  sorry

end hua_luogeng_birthday_factorization_l1536_153678


namespace douglas_vote_percentage_l1536_153611

theorem douglas_vote_percentage (total_percentage : ℝ) (county_y_percentage : ℝ) :
  total_percentage = 64 →
  county_y_percentage = 40.00000000000002 →
  let county_x_votes : ℝ := 2
  let county_y_votes : ℝ := 1
  let total_votes : ℝ := county_x_votes + county_y_votes
  let county_x_percentage : ℝ := 
    (total_percentage * total_votes - county_y_percentage * county_y_votes) / county_x_votes
  county_x_percentage = 76 := by
sorry

end douglas_vote_percentage_l1536_153611


namespace stock_price_increase_l1536_153640

theorem stock_price_increase (P : ℝ) (X : ℝ) : 
  P * (1 + X / 100) * 0.75 * 1.35 = P * 1.215 → X = 20 := by
  sorry

end stock_price_increase_l1536_153640


namespace unique_integer_triangle_with_unit_incircle_l1536_153629

/-- A triangle with integer side lengths and an inscribed circle of radius 1 -/
structure IntegerTriangleWithUnitIncircle where
  a : ℕ
  b : ℕ
  c : ℕ
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b
  h_incircle : (a + b + c) * 2 = (a + b - c) * (b + c - a) * (c + a - b)

/-- The only triangle with integer side lengths and an inscribed circle of radius 1 has sides 5, 4, and 3 -/
theorem unique_integer_triangle_with_unit_incircle :
  ∀ t : IntegerTriangleWithUnitIncircle, t.a = 5 ∧ t.b = 4 ∧ t.c = 3 :=
by sorry

end unique_integer_triangle_with_unit_incircle_l1536_153629


namespace age_ratio_after_two_years_l1536_153625

/-- Given two people a and b, where their initial age ratio is 5:3 and b's age is 6,
    prove that their age ratio after 2 years is 3:2 -/
theorem age_ratio_after_two_years 
  (a b : ℕ) 
  (h1 : a = 5 * b / 3)  -- Initial ratio condition
  (h2 : b = 6)          -- b's initial age
  : (a + 2) / (b + 2) = 3 / 2 := by
  sorry


end age_ratio_after_two_years_l1536_153625


namespace largest_angle_60_degrees_l1536_153630

/-- 
Given a triangle ABC with side lengths a, b, and c satisfying the equation
a^2 + b^2 = c^2 - ab, the largest interior angle of the triangle is 60°.
-/
theorem largest_angle_60_degrees 
  (a b c : ℝ) 
  (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (eq : a^2 + b^2 = c^2 - a*b) : 
  ∃ θ : ℝ, θ ≤ 60 * π / 180 ∧ 
    θ = Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) ∧
    θ = Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)) ∧
    θ = Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)) :=
sorry

end largest_angle_60_degrees_l1536_153630


namespace binomial_coefficient_200_l1536_153654

theorem binomial_coefficient_200 :
  (Nat.choose 200 200 = 1) ∧ (Nat.choose 200 0 = 1) := by
  sorry

end binomial_coefficient_200_l1536_153654


namespace indistinguishable_ball_sequences_l1536_153655

/-- The number of different sequences when drawing indistinguishable balls -/
def number_of_sequences (total : ℕ) (white : ℕ) (black : ℕ) : ℕ :=
  Nat.choose total white

theorem indistinguishable_ball_sequences :
  number_of_sequences 13 8 5 = 1287 := by
  sorry

end indistinguishable_ball_sequences_l1536_153655


namespace seven_balls_three_boxes_l1536_153637

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes,
    with each box containing at least one ball. -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute 7 indistinguishable balls into 3 indistinguishable boxes,
    with each box containing at least one ball, is equal to 4. -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 4 := by
  sorry

end seven_balls_three_boxes_l1536_153637


namespace equation_proof_l1536_153610

theorem equation_proof : Real.sqrt (3^2 + 4^2) / Real.sqrt (25 - 1) = 5 * Real.sqrt 6 / 12 := by
  sorry

end equation_proof_l1536_153610


namespace function_zeros_inequality_l1536_153673

open Real

theorem function_zeros_inequality (a b c : ℝ) (x₁ x₂ : ℝ) :
  0 < a → a < 1 → b > 0 →
  let f := fun x => a * exp x - b * x - c
  f x₁ = 0 → f x₂ = 0 → x₁ > x₂ →
  exp x₁ / a + exp x₂ / (1 - a) > 4 * b / a :=
by sorry

end function_zeros_inequality_l1536_153673


namespace sum_of_squared_coefficients_is_2395_l1536_153656

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ := 3 * (x^2 - 3*x + 3) - 8 * (x^3 - 2*x^2 + 4*x - 1)

/-- The sum of squares of coefficients of the simplified expression -/
def sum_of_squared_coefficients : ℝ := 2395

theorem sum_of_squared_coefficients_is_2395 :
  sum_of_squared_coefficients = 2395 := by sorry

end sum_of_squared_coefficients_is_2395_l1536_153656


namespace investment_amount_l1536_153694

/-- Represents the investment scenario with changing interest rates and inflation --/
structure Investment where
  principal : ℝ
  baseRate : ℝ
  years : ℕ
  rateChangeYear2 : ℝ
  rateChangeYear4 : ℝ
  inflationRate : ℝ
  interestDifference : ℝ

/-- Calculates the total interest earned with rate changes --/
def totalInterestWithChanges (inv : Investment) : ℝ :=
  inv.principal * (5 * inv.baseRate + inv.rateChangeYear2 + inv.rateChangeYear4)

/-- Calculates the total interest earned without rate changes --/
def totalInterestWithoutChanges (inv : Investment) : ℝ :=
  inv.principal * 5 * inv.baseRate

/-- Theorem stating that the original investment amount is $30,000 --/
theorem investment_amount (inv : Investment) 
  (h1 : inv.years = 5)
  (h2 : inv.rateChangeYear2 = 0.005)
  (h3 : inv.rateChangeYear4 = 0.01)
  (h4 : inv.inflationRate = 0.01)
  (h5 : totalInterestWithChanges inv - totalInterestWithoutChanges inv = inv.interestDifference)
  (h6 : inv.interestDifference = 450) :
  inv.principal = 30000 := by
  sorry

#check investment_amount

end investment_amount_l1536_153694


namespace arithmetic_sum_example_l1536_153628

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a : ℤ) (d : ℤ) (l : ℤ) : ℤ :=
  let n : ℤ := (l - a) / d + 1
  n * (a + l) / 2

/-- Theorem: The sum of the arithmetic sequence with first term -41,
    common difference 3, and last term 7 is -289 -/
theorem arithmetic_sum_example : arithmetic_sum (-41) 3 7 = -289 := by
  sorry

end arithmetic_sum_example_l1536_153628


namespace number_of_gyms_l1536_153619

def number_of_bikes_per_gym : ℕ := 10
def number_of_treadmills_per_gym : ℕ := 5
def number_of_ellipticals_per_gym : ℕ := 5

def cost_of_bike : ℕ := 700
def cost_of_treadmill : ℕ := cost_of_bike + cost_of_bike / 2
def cost_of_elliptical : ℕ := 2 * cost_of_treadmill

def total_replacement_cost : ℕ := 455000

def cost_per_gym : ℕ := 
  number_of_bikes_per_gym * cost_of_bike +
  number_of_treadmills_per_gym * cost_of_treadmill +
  number_of_ellipticals_per_gym * cost_of_elliptical

theorem number_of_gyms : 
  total_replacement_cost / cost_per_gym = 20 := by sorry

end number_of_gyms_l1536_153619


namespace sum_of_squares_l1536_153658

theorem sum_of_squares (n : ℕ) (h1 : n > 2) 
  (h2 : ∃ m : ℕ, n^2 = (m + 1)^3 - m^3) : 
  ∃ a b : ℕ, n = a^2 + b^2 := by
sorry

end sum_of_squares_l1536_153658


namespace mary_seashells_count_l1536_153645

/-- The number of seashells Sam found -/
def sam_seashells : ℕ := 18

/-- The total number of seashells Sam and Mary found together -/
def total_seashells : ℕ := 65

/-- The number of seashells Mary found -/
def mary_seashells : ℕ := total_seashells - sam_seashells

theorem mary_seashells_count : mary_seashells = 47 := by sorry

end mary_seashells_count_l1536_153645


namespace stability_comparison_l1536_153649

/-- Represents an athlete's shooting performance -/
structure Athlete where
  average_score : ℝ
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines stability of performance based on variance -/
def more_stable (a b : Athlete) : Prop :=
  a.variance < b.variance

theorem stability_comparison 
  (a b : Athlete) 
  (h_same_avg : a.average_score = b.average_score) 
  (h_var_a : a.variance = 0.4) 
  (h_var_b : b.variance = 2) : 
  more_stable a b :=
sorry

end stability_comparison_l1536_153649


namespace larger_number_is_nine_l1536_153644

theorem larger_number_is_nine (a b : ℕ+) (h1 : a - b = 3) (h2 : a^2 + b^2 = 117) : a = 9 := by
  sorry

end larger_number_is_nine_l1536_153644


namespace factorization_proof_l1536_153602

theorem factorization_proof (b : ℝ) : 56 * b^3 + 168 * b^2 = 56 * b^2 * (b + 3) := by
  sorry

end factorization_proof_l1536_153602
