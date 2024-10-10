import Mathlib

namespace isosceles_right_triangle_from_equation_l1397_139744

/-- Given a triangle ABC with side lengths a, b, and c satisfying the equation
    √(c² - a² - b²) + |a - b| = 0, prove that ABC is an isosceles right triangle. -/
theorem isosceles_right_triangle_from_equation 
  (a b c : ℝ) 
  (triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h : Real.sqrt (c^2 - a^2 - b^2) + |a - b| = 0) : 
  a = b ∧ c^2 = a^2 + b^2 := by
  sorry

end isosceles_right_triangle_from_equation_l1397_139744


namespace equation_solutions_l1397_139755

theorem equation_solutions :
  (∃ x : ℚ, (5 / (x + 1) = 1 / (x - 3)) ∧ x = 4) ∧
  (∃ x : ℚ, ((2 - x) / (x - 3) + 2 = 1 / (3 - x)) ∧ x = 7/3) := by
  sorry

end equation_solutions_l1397_139755


namespace fraction_equality_implies_value_l1397_139789

theorem fraction_equality_implies_value (b : ℝ) :
  b / (b + 30) = 0.92 → b = 345 := by sorry

end fraction_equality_implies_value_l1397_139789


namespace fraction_evaluation_l1397_139777

theorem fraction_evaluation (a b : ℝ) (h1 : a = 5) (h2 : b = 3) :
  3 / (a + b) = 3 / 8 := by
  sorry

end fraction_evaluation_l1397_139777


namespace complement_A_intersect_B_a_greater_than_seven_l1397_139752

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}

-- Theorem for part (1)
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10)} := by sorry

-- Theorem for part (2)
theorem a_greater_than_seven (h : A ⊆ C a) : a > 7 := by sorry

end complement_A_intersect_B_a_greater_than_seven_l1397_139752


namespace parabola_vertex_l1397_139771

/-- The vertex of the parabola y = 2(x-3)^2 + 1 is at the point (3, 1). -/
theorem parabola_vertex (x y : ℝ) : 
  y = 2 * (x - 3)^2 + 1 → (3, 1) = (x, y) := by sorry

end parabola_vertex_l1397_139771


namespace debby_jogged_nine_km_on_wednesday_l1397_139708

/-- The distance Debby jogged on Monday in kilometers -/
def monday_distance : ℕ := 2

/-- The distance Debby jogged on Tuesday in kilometers -/
def tuesday_distance : ℕ := 5

/-- The total distance Debby jogged over three days in kilometers -/
def total_distance : ℕ := 16

/-- The distance Debby jogged on Wednesday in kilometers -/
def wednesday_distance : ℕ := total_distance - (monday_distance + tuesday_distance)

theorem debby_jogged_nine_km_on_wednesday :
  wednesday_distance = 9 := by sorry

end debby_jogged_nine_km_on_wednesday_l1397_139708


namespace sqrt_expression_equivalence_l1397_139726

theorem sqrt_expression_equivalence (x : ℝ) (h : x < -1) :
  Real.sqrt (x / (1 - x * (1 - 1 / (x + 1)))) = abs x :=
by sorry

end sqrt_expression_equivalence_l1397_139726


namespace jane_change_l1397_139751

def skirt_price : ℕ := 13
def skirt_quantity : ℕ := 2
def blouse_price : ℕ := 6
def blouse_quantity : ℕ := 3
def amount_paid : ℕ := 100

def total_cost : ℕ := skirt_price * skirt_quantity + blouse_price * blouse_quantity

theorem jane_change : amount_paid - total_cost = 56 := by
  sorry

end jane_change_l1397_139751


namespace new_average_production_l1397_139737

theorem new_average_production (n : ℕ) (past_average : ℝ) (today_production : ℝ) :
  n = 1 ∧ past_average = 50 ∧ today_production = 60 →
  (n * past_average + today_production) / (n + 1) = 55 := by
sorry

end new_average_production_l1397_139737


namespace functional_equation_solution_l1397_139728

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y + f (x + y) = x * y

/-- The theorem stating that the only functions satisfying the equation are x - 1 or -x - 1 -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f → (∀ x : ℝ, f x = x - 1 ∨ f x = -x - 1) :=
by sorry

end functional_equation_solution_l1397_139728


namespace two_integers_sum_l1397_139719

theorem two_integers_sum (a b : ℕ+) : a - b = 4 → a * b = 96 → a + b = 20 := by
  sorry

end two_integers_sum_l1397_139719


namespace apollonius_circle_exists_l1397_139730

-- Define a circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a tangency relation between two circles
def is_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2 ∨
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius - c2.radius)^2

-- Theorem statement
theorem apollonius_circle_exists (S1 S2 S3 : Circle) :
  ∃ S : Circle, is_tangent S S1 ∧ is_tangent S S2 ∧ is_tangent S S3 :=
sorry

end apollonius_circle_exists_l1397_139730


namespace retailer_profit_percent_l1397_139786

/-- Calculates the profit percent for a retailer given the purchase price, overhead expenses, and selling price. -/
def profit_percent (purchase_price overhead_expenses selling_price : ℚ) : ℚ :=
  let cost_price := purchase_price + overhead_expenses
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percent for the given scenario is approximately 21.46% -/
theorem retailer_profit_percent :
  let ε := 0.01
  let result := profit_percent 232 15 300
  (result > 21.46 - ε) ∧ (result < 21.46 + ε) :=
by
  sorry

end retailer_profit_percent_l1397_139786


namespace distance_driven_margies_car_distance_l1397_139799

/-- Proves that given a car's fuel efficiency and gas price, 
    we can calculate the distance driven with a certain amount of money. -/
theorem distance_driven (efficiency : ℝ) (gas_price : ℝ) (money : ℝ) :
  efficiency > 0 → gas_price > 0 → money > 0 →
  (efficiency * (money / gas_price) = 200) ↔ 
  (efficiency = 40 ∧ gas_price = 5 ∧ money = 25) :=
sorry

/-- Specific instance of the theorem for Margie's car -/
theorem margies_car_distance : 
  ∃ (efficiency gas_price money : ℝ),
    efficiency > 0 ∧ gas_price > 0 ∧ money > 0 ∧
    efficiency = 40 ∧ gas_price = 5 ∧ money = 25 ∧
    efficiency * (money / gas_price) = 200 :=
sorry

end distance_driven_margies_car_distance_l1397_139799


namespace race_outcomes_count_l1397_139791

/-- Represents the number of participants in the race -/
def num_participants : ℕ := 6

/-- Represents the number of top positions we're considering -/
def num_top_positions : ℕ := 4

/-- Calculates the number of permutations of k items chosen from n items -/
def permutations (n k : ℕ) : ℕ := 
  if k > n then 0
  else (List.range n).foldr (λ i acc => (i + 1) * acc) 1

/-- Theorem stating the number of possible race outcomes -/
theorem race_outcomes_count : 
  (permutations (num_participants - 1) (num_top_positions - 1)) * num_participants - 
  (permutations (num_participants - 1) (num_top_positions - 1)) = 300 := by
  sorry


end race_outcomes_count_l1397_139791


namespace four_digit_count_l1397_139731

/-- The count of four-digit numbers -/
def count_four_digit_numbers : ℕ := 9999 - 1000 + 1

/-- The smallest four-digit number -/
def min_four_digit : ℕ := 1000

/-- The largest four-digit number -/
def max_four_digit : ℕ := 9999

/-- Theorem: The count of integers from 1000 to 9999 (inclusive) is equal to 9000 -/
theorem four_digit_count :
  count_four_digit_numbers = 9000 := by sorry

end four_digit_count_l1397_139731


namespace complex_equation_solutions_l1397_139793

theorem complex_equation_solutions :
  ∃! (s : Finset ℂ), (∀ z ∈ s, Complex.abs z < 20 ∧ Complex.exp z = (z - 1) / (z + 1)) ∧ Finset.card s = 8 := by
  sorry

end complex_equation_solutions_l1397_139793


namespace cosine_graph_transformation_l1397_139792

theorem cosine_graph_transformation (x : ℝ) :
  let f (x : ℝ) := 2 * Real.cos (x + π / 3)
  let g (x : ℝ) := 2 * Real.cos (2 * x + π / 6)
  let h (x : ℝ) := f (2 * x)
  h (x - π / 12) = g x :=
by sorry

end cosine_graph_transformation_l1397_139792


namespace parallel_lines_k_values_l1397_139743

/-- Definition of Line l₁ -/
def l₁ (k : ℝ) (x y : ℝ) : Prop :=
  (k - 3) * x + (4 - k) * y + 1 = 0

/-- Definition of Line l₂ -/
def l₂ (k : ℝ) (x y : ℝ) : Prop :=
  2 * (k - 3) * x - 2 * y + 3 = 0

/-- Definition of parallel lines -/
def parallel (l₁ l₂ : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ (m : ℝ), ∀ (k x y : ℝ), l₁ k x y ↔ l₂ k (m * x) y

/-- Theorem stating that for l₁ and l₂ to be parallel, k must be 2, 3, or 6 -/
theorem parallel_lines_k_values :
  parallel l₁ l₂ ↔ (∃ k : ℝ, k = 2 ∨ k = 3 ∨ k = 6) :=
sorry

end parallel_lines_k_values_l1397_139743


namespace least_four_digit_divisible_by_2_3_5_7_l1397_139748

theorem least_four_digit_divisible_by_2_3_5_7 :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 1050 → ¬(2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n)) ∧
  (1050 ≥ 1000) ∧
  (2 ∣ 1050) ∧ (3 ∣ 1050) ∧ (5 ∣ 1050) ∧ (7 ∣ 1050) :=
by sorry

end least_four_digit_divisible_by_2_3_5_7_l1397_139748


namespace quadratic_inequality_solution_set_l1397_139703

/-- Given that the solution set of ax^2 - 5x + b > 0 is {x | -3 < x < 2},
    prove that the solution set of bx^2 - 5x + a > 0 is {x | x < -1/3 or x > 1/2} -/
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : ∀ x : ℝ, ax^2 - 5*x + b > 0 ↔ -3 < x ∧ x < 2) :
  ∀ x : ℝ, b*x^2 - 5*x + a > 0 ↔ x < -1/3 ∨ x > 1/2 := by
  sorry

end quadratic_inequality_solution_set_l1397_139703


namespace dividend_divisor_change_l1397_139760

theorem dividend_divisor_change (a b : ℝ) (h : b ≠ 0) :
  (11 * a) / (10 * b) ≠ a / b :=
sorry

end dividend_divisor_change_l1397_139760


namespace geometric_series_sum_l1397_139787

/-- The sum of the infinite geometric series 4/3 - 5/12 + 25/144 - 125/1728 + ... -/
theorem geometric_series_sum : 
  let a : ℚ := 4/3
  let r : ℚ := -5/16
  let series_sum : ℚ := a / (1 - r)
  series_sum = 64/63 := by sorry

end geometric_series_sum_l1397_139787


namespace tournament_ordered_victories_l1397_139766

/-- A round-robin tournament with 2^n players -/
def Tournament (n : ℕ) := Fin (2^n)

/-- The result of a match between two players -/
def Defeats (t : Tournament n) : Tournament n → Tournament n → Prop := sorry

/-- The property that player i defeats player j if and only if i < j -/
def OrderedVictories (t : Tournament n) (s : Fin (n+1) → Tournament n) : Prop :=
  ∀ i j, i < j → Defeats t (s i) (s j)

/-- The main theorem: In any tournament of 2^n players, there exists an ordered sequence of n+1 players -/
theorem tournament_ordered_victories (n : ℕ) :
  ∀ t : Tournament n, ∃ s : Fin (n+1) → Tournament n, OrderedVictories t s := by
  sorry

end tournament_ordered_victories_l1397_139766


namespace mistaken_division_correction_l1397_139769

theorem mistaken_division_correction (N : ℕ) : 
  N % 23 = 17 ∧ N / 23 = 3 → (N / 32) + (N % 32) = 24 := by
sorry

end mistaken_division_correction_l1397_139769


namespace toy_shipment_calculation_l1397_139785

theorem toy_shipment_calculation (displayed_percentage : ℚ) (stored_toys : ℕ) : 
  displayed_percentage = 30 / 100 →
  stored_toys = 140 →
  (1 - displayed_percentage) * 200 = stored_toys := by
  sorry

end toy_shipment_calculation_l1397_139785


namespace right_prism_cut_count_l1397_139783

theorem right_prism_cut_count : 
  let b : ℕ := 2023
  let count := (Finset.filter 
    (fun p : ℕ × ℕ => 
      let (a, c) := p
      a ≤ b ∧ b ≤ c ∧ a * c = b * b ∧ a < c)
    (Finset.product (Finset.range (b + 1)) (Finset.range (b * b + 1)))).card
  count = 13 := by
sorry

end right_prism_cut_count_l1397_139783


namespace cards_per_set_is_13_l1397_139724

/-- The number of trading cards in one set -/
def cards_per_set (initial_cards : ℕ) (sets_to_brother : ℕ) (sets_to_sister : ℕ) (sets_to_friend : ℕ) (total_cards_given : ℕ) : ℕ :=
  total_cards_given / (sets_to_brother + sets_to_sister + sets_to_friend)

/-- Proof that the number of trading cards in one set is 13 -/
theorem cards_per_set_is_13 :
  cards_per_set 365 8 5 2 195 = 13 := by
  sorry

end cards_per_set_is_13_l1397_139724


namespace min_reciprocal_sum_l1397_139773

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 12) :
  (1 / x + 1 / y) ≥ (1 / 3) := by
sorry

end min_reciprocal_sum_l1397_139773


namespace triangle_angle_proof_l1397_139711

theorem triangle_angle_proof (A B C a b c : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- angles are positive
  A + B + C = π ∧ -- sum of angles in a triangle
  0 < a ∧ 0 < b ∧ 0 < c ∧ -- sides are positive
  a * Real.cos C + c * Real.cos A = 2 * b * Real.cos B → -- given condition
  B = π / 3 := by
sorry

end triangle_angle_proof_l1397_139711


namespace algebraic_expression_value_l1397_139722

theorem algebraic_expression_value : 
  let a : ℝ := Real.sqrt 5 + 1
  (a^2 - 2*a + 7) = 11 := by sorry

end algebraic_expression_value_l1397_139722


namespace expression_evaluation_l1397_139759

theorem expression_evaluation : -(-2) + 2 * Real.cos (60 * π / 180) + (-1/8)⁻¹ + (Real.pi - 3.14)^0 = -4 := by
  sorry

end expression_evaluation_l1397_139759


namespace Z_three_seven_l1397_139797

def Z (a b : ℝ) : ℝ := b + 15 * a - a^2

theorem Z_three_seven : Z 3 7 = 43 := by sorry

end Z_three_seven_l1397_139797


namespace sqrt_sum_reciprocal_l1397_139772

theorem sqrt_sum_reciprocal (x : ℝ) (hx_pos : x > 0) (hx_sum : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end sqrt_sum_reciprocal_l1397_139772


namespace sin_75_cos_75_eq_half_l1397_139753

theorem sin_75_cos_75_eq_half : 2 * Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1 / 2 := by
  sorry

end sin_75_cos_75_eq_half_l1397_139753


namespace point_coordinates_l1397_139798

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the third quadrant -/
def in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (p : Point) 
  (h1 : in_third_quadrant p)
  (h2 : distance_to_x_axis p = 8)
  (h3 : distance_to_y_axis p = 5) :
  p = Point.mk (-5) (-8) := by
  sorry

end point_coordinates_l1397_139798


namespace smallest_multiple_of_eleven_l1397_139788

theorem smallest_multiple_of_eleven (x y : ℤ) 
  (h1 : ∃ k : ℤ, x + 2 = 11 * k) 
  (h2 : ∃ m : ℤ, y - 1 = 11 * m) : 
  (∃ n : ℕ+, ∃ p : ℤ, x^2 + x*y + y^2 + n = 11 * p) ∧ 
  (∀ n : ℕ+, n < 8 → ¬∃ p : ℤ, x^2 + x*y + y^2 + n = 11 * p) :=
by sorry

end smallest_multiple_of_eleven_l1397_139788


namespace club_membership_increase_l1397_139762

theorem club_membership_increase (current_members : ℕ) (h : current_members = 10) : 
  (2 * current_members + 5) - current_members = 15 := by
  sorry

end club_membership_increase_l1397_139762


namespace disjunction_true_given_p_l1397_139716

theorem disjunction_true_given_p (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by sorry

end disjunction_true_given_p_l1397_139716


namespace inequality_solution_l1397_139701

theorem inequality_solution (x : ℝ) :
  x ≠ 2 → x ≠ 0 →
  ((x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4 ↔ 
   (0 < x ∧ x ≤ 1/2) ∨ (2 < x ∧ x ≤ 11/2)) :=
by sorry

end inequality_solution_l1397_139701


namespace g_is_max_g_symmetric_points_l1397_139700

noncomputable def f (a x : ℝ) : ℝ := a * Real.sqrt (1 - x^2) + Real.sqrt (1 + x) + Real.sqrt (1 - x)

noncomputable def g (a : ℝ) : ℝ := 
  if a > -1/2 then a + 2
  else if a > -Real.sqrt 2 / 2 then -a - 1/(2*a)
  else Real.sqrt 2

theorem g_is_max (a : ℝ) : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f a x ≤ g a := by sorry

theorem g_symmetric_points (a : ℝ) : 
  ((-Real.sqrt 2 ≤ a ∧ a ≤ -Real.sqrt 2 / 2) ∨ a = 1) ↔ g a = g (1/a) := by sorry

end g_is_max_g_symmetric_points_l1397_139700


namespace min_value_of_exp_minus_x_l1397_139768

theorem min_value_of_exp_minus_x :
  ∃ (x : ℝ), ∀ (y : ℝ), Real.exp y - y ≥ Real.exp x - x ∧ Real.exp x - x = 1 := by
  sorry

end min_value_of_exp_minus_x_l1397_139768


namespace pink_balls_count_l1397_139746

/-- The number of pink balls initially in the bag -/
def initial_pink_balls : ℕ := 23

/-- The number of green balls initially in the bag -/
def initial_green_balls : ℕ := 9

/-- The number of green balls added -/
def added_green_balls : ℕ := 14

theorem pink_balls_count :
  (initial_green_balls + added_green_balls = initial_pink_balls) ∧
  (initial_green_balls + added_green_balls : ℚ) / initial_pink_balls = 1 := by
  sorry

end pink_balls_count_l1397_139746


namespace circle_radius_implies_c_l1397_139713

/-- Given a circle with equation x^2 + 6x + y^2 - 4y + c = 0 and radius 6, prove that c = -23 -/
theorem circle_radius_implies_c (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 6*x + y^2 - 4*y + c = 0 → (x+3)^2 + (y-2)^2 = 36) → 
  c = -23 := by
  sorry

end circle_radius_implies_c_l1397_139713


namespace rectangle_split_area_l1397_139756

theorem rectangle_split_area (c : ℝ) : 
  let total_area : ℝ := 8
  let smaller_area : ℝ := total_area / 3
  let larger_area : ℝ := 2 * smaller_area
  let triangle_area : ℝ := 2 * (4 - c)
  (4 + total_area - triangle_area = larger_area) → c = 8/9 := by
  sorry

end rectangle_split_area_l1397_139756


namespace twoDigitNumberRepresentation_l1397_139721

/-- Represents a two-digit number with x in the tens place and 5 in the ones place -/
def twoDigitNumber (x : ℕ) : ℕ := 10 * x + 5

/-- Proves that a two-digit number with x in the tens place and 5 in the ones place
    can be represented as 10x + 5 -/
theorem twoDigitNumberRepresentation (x : ℕ) (h : x < 10) :
  twoDigitNumber x = 10 * x + 5 := by
  sorry

end twoDigitNumberRepresentation_l1397_139721


namespace inequality_proof_l1397_139780

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h1 : a^2 < 16*b*c) (h2 : b^2 < 16*c*a) (h3 : c^2 < 16*a*b) :
  a^2 + b^2 + c^2 < 2*(a*b + b*c + c*a) := by
  sorry

end inequality_proof_l1397_139780


namespace second_frog_hops_l1397_139761

theorem second_frog_hops :
  ∀ (h1 h2 h3 : ℕ),
  h1 = 4 * h2 →
  h2 = 2 * h3 →
  h1 + h2 + h3 = 99 →
  h2 = 18 := by
sorry

end second_frog_hops_l1397_139761


namespace joe_first_lift_weight_l1397_139750

theorem joe_first_lift_weight (first_lift second_lift : ℝ) 
  (total_weight : first_lift + second_lift = 800)
  (lift_relation : 3 * first_lift = 2 * second_lift + 450) :
  first_lift = 410 := by
sorry

end joe_first_lift_weight_l1397_139750


namespace four_digit_multiples_of_five_l1397_139739

theorem four_digit_multiples_of_five : 
  (Finset.filter (fun n => n % 5 = 0) (Finset.range 9000)).card = 1800 := by
  sorry

end four_digit_multiples_of_five_l1397_139739


namespace kim_total_points_l1397_139763

/-- Calculates the total points in a math contest with three rounds -/
def totalPoints (easyPoints averagePoints hardPoints : ℕ) 
                (easyCorrect averageCorrect hardCorrect : ℕ) : ℕ :=
  easyPoints * easyCorrect + averagePoints * averageCorrect + hardPoints * hardCorrect

/-- Theorem: Kim's total points in the contest -/
theorem kim_total_points :
  totalPoints 2 3 5 6 2 4 = 38 := by
  sorry

end kim_total_points_l1397_139763


namespace common_point_for_gp_lines_l1397_139745

/-- A line in the form ax + by = c where a, b, c form a geometric progression -/
structure GPLine where
  a : ℝ
  r : ℝ
  h_r_nonzero : r ≠ 0

/-- The equation of a GPLine -/
def GPLine.equation (l : GPLine) (x y : ℝ) : Prop :=
  l.a * x + (l.a * l.r) * y = l.a * l.r^2

theorem common_point_for_gp_lines :
  ∀ (l : GPLine), l.equation 1 0 :=
sorry

end common_point_for_gp_lines_l1397_139745


namespace inequality_proof_l1397_139767

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := by
  sorry

end inequality_proof_l1397_139767


namespace pet_store_birds_l1397_139732

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 7

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 4

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 3

/-- The number of cockatiels in each cage -/
def cockatiels_per_cage : ℕ := 2

/-- The number of canaries in each cage -/
def canaries_per_cage : ℕ := 1

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage + cockatiels_per_cage + canaries_per_cage)

theorem pet_store_birds : total_birds = 70 := by
  sorry

end pet_store_birds_l1397_139732


namespace lost_ship_depth_l1397_139749

/-- The depth of a lost ship given the descent rate and time taken to reach it. -/
def depth_of_lost_ship (descent_rate : ℝ) (time_taken : ℝ) : ℝ :=
  descent_rate * time_taken

/-- Theorem: The depth of the lost ship is 3600 feet. -/
theorem lost_ship_depth :
  let descent_rate : ℝ := 60
  let time_taken : ℝ := 60
  depth_of_lost_ship descent_rate time_taken = 3600 := by
sorry

end lost_ship_depth_l1397_139749


namespace lcm_gcd_product_l1397_139790

theorem lcm_gcd_product (a b : ℕ) (h1 : a = 24) (h2 : b = 54) :
  (Nat.lcm a b) * (Nat.gcd a b) = 1296 := by
  sorry

end lcm_gcd_product_l1397_139790


namespace percentage_of_720_is_356_4_l1397_139742

theorem percentage_of_720_is_356_4 : 
  let whole : ℝ := 720
  let part : ℝ := 356.4
  let percentage : ℝ := (part / whole) * 100
  percentage = 49.5 := by sorry

end percentage_of_720_is_356_4_l1397_139742


namespace range_of_a_l1397_139794

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * Real.sin x + Real.cos x < 2) → 
  -Real.sqrt 3 < a ∧ a < Real.sqrt 3 :=
by sorry

end range_of_a_l1397_139794


namespace xyz_equals_seven_l1397_139734

theorem xyz_equals_seven (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) :
  x * y * z = 7 := by
sorry

end xyz_equals_seven_l1397_139734


namespace certain_number_is_even_l1397_139776

theorem certain_number_is_even (z : ℕ) (h1 : z > 0) (h2 : 4 ∣ z) :
  ∀ x : ℤ, (z * (2 + x + z) + 3) % 2 = 1 ↔ Even x :=
by sorry

end certain_number_is_even_l1397_139776


namespace polyhedron_volume_from_parallelepiped_l1397_139741

/-- Given a parallelepiped with volume V, the volume of the polyhedron formed by
    connecting the centers of its faces is 1/6 * V -/
theorem polyhedron_volume_from_parallelepiped (V : ℝ) (V_pos : V > 0) :
  ∃ (polyhedron_volume : ℝ),
    polyhedron_volume = (1 / 6 : ℝ) * V ∧
    polyhedron_volume > 0 := by
  sorry

end polyhedron_volume_from_parallelepiped_l1397_139741


namespace annes_height_l1397_139727

/-- Proves that Anne's height is 80 cm given the relationships between heights of Anne, her sister, and Bella -/
theorem annes_height (sister_height : ℝ) (anne_height : ℝ) (bella_height : ℝ) : 
  anne_height = 2 * sister_height →
  bella_height = 3 * anne_height →
  bella_height - sister_height = 200 →
  anne_height = 80 := by
sorry

end annes_height_l1397_139727


namespace circle_alignment_exists_l1397_139784

-- Define the circle type
structure Circle where
  circumference : ℝ
  marked_points : ℕ
  arc_length : ℝ

-- Define the theorem
theorem circle_alignment_exists (c1 c2 : Circle)
  (h1 : c1.circumference = 100)
  (h2 : c2.circumference = 100)
  (h3 : c1.marked_points = 100)
  (h4 : c2.arc_length < 1) :
  ∃ (alignment : ℝ), ∀ (point : ℕ) (arc : ℝ),
    point < c1.marked_points →
    arc < c2.arc_length →
    (point : ℝ) * c1.circumference / c1.marked_points + alignment ≠ arc :=
sorry

end circle_alignment_exists_l1397_139784


namespace largest_solution_of_equation_l1397_139770

theorem largest_solution_of_equation (x : ℝ) :
  (x^2 - x - 72) / (x - 9) = 5 / (x + 4) →
  x ≤ -3 :=
by sorry

end largest_solution_of_equation_l1397_139770


namespace expression_evaluation_l1397_139707

theorem expression_evaluation (a b : ℚ) (ha : a = 7) (hb : b = 5) :
  3 * (a^3 + b^3) / (a^2 - a*b + b^2) = 36 := by
  sorry

end expression_evaluation_l1397_139707


namespace divided_hexagon_areas_l1397_139717

/-- Represents a regular hexagon divided by four diagonals -/
structure DividedHexagon where
  /-- The area of the central quadrilateral -/
  quadrilateralArea : ℝ
  /-- The areas of the six triangles -/
  triangleAreas : Fin 6 → ℝ

/-- Theorem about the areas of triangles in a divided regular hexagon -/
theorem divided_hexagon_areas (h : DividedHexagon) 
  (hq : h.quadrilateralArea = 1.8) : 
  (h.triangleAreas 0 = 1.2 ∧ 
   h.triangleAreas 1 = 1.2 ∧ 
   h.triangleAreas 2 = 0.6 ∧ 
   h.triangleAreas 3 = 0.6 ∧ 
   h.triangleAreas 4 = 1.2 ∧ 
   h.triangleAreas 5 = 0.6) := by
  sorry

end divided_hexagon_areas_l1397_139717


namespace prop_p_false_prop_q_true_l1397_139723

-- Define the curve C
def curve_C (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (25 - k) + p.2^2 / (k - 9) = 1}

-- Define what it means for a curve to be an ellipse
def is_ellipse (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ S = {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}

-- Define what it means for a curve to be a hyperbola with foci on the x-axis
def is_hyperbola_x_axis (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ S = {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}

-- Theorem 1: Proposition p is false
theorem prop_p_false : ¬(∀ k : ℝ, 9 < k ∧ k < 25 → is_ellipse (curve_C k)) :=
  sorry

-- Theorem 2: Proposition q is true
theorem prop_q_true : ∀ k : ℝ, is_hyperbola_x_axis (curve_C k) → k < 9 :=
  sorry

end prop_p_false_prop_q_true_l1397_139723


namespace fraction_equality_l1397_139733

theorem fraction_equality : (1/5 - 1/6) / (1/3 - 1/4) = 2/5 := by
  sorry

end fraction_equality_l1397_139733


namespace parabola_minimum_distance_l1397_139764

/-- Parabola defined by y² = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- Line with slope k passing through (2, 0) -/
def line (k x y : ℝ) : Prop := y = k*(x - 2)

/-- Distance between two x-coordinates on the parabola -/
def distance_on_parabola (x1 x2 : ℝ) : ℝ := |x1 - x2|

theorem parabola_minimum_distance (k1 k2 : ℝ) :
  k1 * k2 = -2 →
  ∃ (xA xC xB xD : ℝ),
    parabola xA (k1*(xA - 2)) ∧
    parabola xC (k1*(xC - 2)) ∧
    parabola xB (k2*(xB - 2)) ∧
    parabola xD (k2*(xD - 2)) ∧
    (∀ x1A x1C x2B x2D : ℝ,
      parabola x1A (k1*(x1A - 2)) →
      parabola x1C (k1*(x1C - 2)) →
      parabola x2B (k2*(x2B - 2)) →
      parabola x2D (k2*(x2D - 2)) →
      distance_on_parabola xA xC + distance_on_parabola xB xD ≤
      distance_on_parabola x1A x1C + distance_on_parabola x2B x2D) ∧
    distance_on_parabola xA xC + distance_on_parabola xB xD = 24 :=
by sorry


end parabola_minimum_distance_l1397_139764


namespace parallel_EX_AP_l1397_139738

noncomputable section

-- Define the points on a complex plane
variable (a b c p h e q r x : ℂ)

-- Define the triangle ABC on the unit circle
def on_unit_circle (z : ℂ) : Prop := Complex.abs z = 1

-- Define the orthocenter condition
def is_orthocenter (a b c h : ℂ) : Prop := a + b + c = h

-- Define the circumcircle condition
def on_circumcircle (a b c p : ℂ) : Prop := on_unit_circle p

-- Define the foot of altitude condition
def is_foot_of_altitude (a b c e : ℂ) : Prop :=
  e = (1 / 2) * (a + b + c - (a * c) / b)

-- Define parallelogram conditions
def is_parallelogram_PAQB (a b p q : ℂ) : Prop := q = a + b - p
def is_parallelogram_PARC (a c p r : ℂ) : Prop := r = a + c - p

-- Define the intersection point condition
def is_intersection (a q h r x : ℂ) : Prop :=
  ∃ t₁ t₂ : ℝ, x = a + t₁ * (q - a) ∧ x = h + t₂ * (r - h)

-- Main theorem
theorem parallel_EX_AP (a b c p h e q r x : ℂ) 
  (h_circle : on_unit_circle a ∧ on_unit_circle b ∧ on_unit_circle c)
  (h_orthocenter : is_orthocenter a b c h)
  (h_circumcircle : on_circumcircle a b c p)
  (h_foot : is_foot_of_altitude a b c e)
  (h_para1 : is_parallelogram_PAQB a b p q)
  (h_para2 : is_parallelogram_PARC a c p r)
  (h_intersect : is_intersection a q h r x) :
  ∃ k : ℂ, e - x = k * (a - p) :=
sorry

end parallel_EX_AP_l1397_139738


namespace rationalize_denominator_l1397_139704

theorem rationalize_denominator : 7 / Real.sqrt 175 = Real.sqrt 7 / 5 := by
  sorry

end rationalize_denominator_l1397_139704


namespace expected_informed_after_pairing_l1397_139736

/-- Represents the scenario of scientists sharing news during a conference break -/
def ScientistNewsSharing (total : ℕ) (initial_informed : ℕ) : Prop :=
  total = 18 ∧ initial_informed = 10

/-- Calculates the expected number of scientists who know the news after pairing -/
noncomputable def expected_informed (total : ℕ) (initial_informed : ℕ) : ℝ :=
  initial_informed + (total - initial_informed) * (initial_informed / (total - 1))

/-- Theorem stating the expected number of informed scientists after pairing -/
theorem expected_informed_after_pairing {total initial_informed : ℕ} 
  (h : ScientistNewsSharing total initial_informed) :
  expected_informed total initial_informed = 14.7 := by
  sorry

end expected_informed_after_pairing_l1397_139736


namespace tomato_count_l1397_139778

/-- Represents a rectangular garden with tomatoes -/
structure TomatoGarden where
  rows : ℕ
  columns : ℕ
  tomato_position : ℕ × ℕ

/-- Calculates the total number of tomatoes in the garden -/
def total_tomatoes (garden : TomatoGarden) : ℕ :=
  garden.rows * garden.columns

/-- Theorem stating the total number of tomatoes in the garden -/
theorem tomato_count (garden : TomatoGarden) 
  (h1 : garden.tomato_position.1 = 8)  -- 8th row from front
  (h2 : garden.rows - garden.tomato_position.1 + 1 = 14)  -- 14th row from back
  (h3 : garden.tomato_position.2 = 7)  -- 7th row from left
  (h4 : garden.columns - garden.tomato_position.2 + 1 = 13)  -- 13th row from right
  : total_tomatoes garden = 399 := by
  sorry

#eval total_tomatoes { rows := 21, columns := 19, tomato_position := (8, 7) }

end tomato_count_l1397_139778


namespace students_with_b_in_dawsons_class_l1397_139729

theorem students_with_b_in_dawsons_class 
  (charles_total : ℕ) 
  (charles_b : ℕ) 
  (dawson_total : ℕ) 
  (h1 : charles_total = 20)
  (h2 : charles_b = 12)
  (h3 : dawson_total = 30)
  (h4 : charles_b * dawson_total = charles_total * dawson_b) :
  dawson_b = 18 := by
    sorry

#check students_with_b_in_dawsons_class

end students_with_b_in_dawsons_class_l1397_139729


namespace positive_integer_from_operations_l1397_139715

def integers : Set ℚ := {0, -3, 5, -100, 2008, -1}
def fractions : Set ℚ := {1/2, -1/3, 1/5, -3/2, -1/100}

theorem positive_integer_from_operations : ∃ (a b : ℚ) (c d : ℚ) (op1 op2 : ℚ → ℚ → ℚ),
  a ∈ integers ∧ b ∈ integers ∧ c ∈ fractions ∧ d ∈ fractions ∧
  (op1 = (· + ·) ∨ op1 = (· - ·) ∨ op1 = (· * ·) ∨ op1 = (· / ·)) ∧
  (op2 = (· + ·) ∨ op2 = (· - ·) ∨ op2 = (· * ·) ∨ op2 = (· / ·)) ∧
  ∃ (n : ℕ), (op2 (op1 a b) (op1 c d) : ℚ) = n := by
  sorry

end positive_integer_from_operations_l1397_139715


namespace square_root_problem_l1397_139795

theorem square_root_problem (y z x : ℝ) (hy : y > 0) (hx : x > 0) :
  y^z = (Real.sqrt 16)^3 → x^2 = y^z → x = 8 := by
  sorry

end square_root_problem_l1397_139795


namespace det_submatrix_l1397_139754

theorem det_submatrix (a b c d : ℝ) :
  Matrix.det !![1, a, b; 2, c, d; 3, 0, 0] = 6 →
  Matrix.det !![a, b; c, d] = 2 := by
sorry

end det_submatrix_l1397_139754


namespace reflection_composition_l1397_139796

theorem reflection_composition :
  let x_reflection : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, -1]
  let y_reflection : Matrix (Fin 2) (Fin 2) ℝ := !![-1, 0; 0, 1]
  x_reflection * y_reflection = !![-1, 0; 0, -1] := by
sorry

end reflection_composition_l1397_139796


namespace quadratic_inequality_no_solution_l1397_139718

theorem quadratic_inequality_no_solution : ∀ x : ℝ, x^2 - 2*x + 3 ≥ 0 := by
  sorry

end quadratic_inequality_no_solution_l1397_139718


namespace sufficient_not_necessary_implies_necessary_not_sufficient_l1397_139765

theorem sufficient_not_necessary_implies_necessary_not_sufficient 
  (A B : Prop) (h : (A → B) ∧ ¬(B → A)) : 
  ((¬B → ¬A) ∧ ¬(¬A → ¬B)) := by
  sorry

end sufficient_not_necessary_implies_necessary_not_sufficient_l1397_139765


namespace alison_lollipops_l1397_139779

theorem alison_lollipops :
  ∀ (alison henry diane : ℕ),
  henry = alison + 30 →
  alison = diane / 2 →
  alison + henry + diane = 45 * 6 →
  alison = 60 := by
sorry

end alison_lollipops_l1397_139779


namespace matrix_addition_problem_l1397_139712

theorem matrix_addition_problem : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 0, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 2; 7, -10]
  A + B = !![-2, -1; 7, -5] := by
sorry

end matrix_addition_problem_l1397_139712


namespace solution_set_inequality_l1397_139747

theorem solution_set_inequality (x : ℝ) :
  (Set.Icc 1 2 : Set ℝ) = {x | -x^2 + 3*x - 2 ≥ 0} :=
by sorry

end solution_set_inequality_l1397_139747


namespace no_rational_solution_l1397_139705

theorem no_rational_solution : ¬∃ (x y z : ℚ), 
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x^5 + 2*y^5 + 5*z^5 = 11 := by
  sorry

end no_rational_solution_l1397_139705


namespace exists_x_squared_minus_two_x_plus_one_nonpositive_l1397_139781

theorem exists_x_squared_minus_two_x_plus_one_nonpositive :
  ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0 := by
  sorry

end exists_x_squared_minus_two_x_plus_one_nonpositive_l1397_139781


namespace smallest_reciprocal_l1397_139774

theorem smallest_reciprocal (a b c : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_order : a > b ∧ b > c) :
  (1 : ℚ) / a < (1 : ℚ) / b ∧ (1 : ℚ) / b < (1 : ℚ) / c :=
by sorry

end smallest_reciprocal_l1397_139774


namespace complex_division_result_l1397_139702

theorem complex_division_result : 
  let i : ℂ := Complex.I
  (1 + i) / (-2 * i) = -1/2 + 1/2 * i := by sorry

end complex_division_result_l1397_139702


namespace lily_pad_growth_rate_l1397_139706

/-- Represents the coverage of the lake by lily pads -/
def LakeCoverage := ℝ

/-- The time it takes for the lily pads to cover the entire lake -/
def fullCoverageTime : ℕ := 50

/-- The time it takes for the lily pads to cover half the lake -/
def halfCoverageTime : ℕ := 49

/-- The growth rate of the lily pad patch -/
def growthRate : ℝ → Prop := λ r => 
  ∀ t : ℝ, (1 : ℝ) = (1/2 : ℝ) * (1 + r) ^ (t + 1) → t = (fullCoverageTime - halfCoverageTime : ℝ)

theorem lily_pad_growth_rate : 
  growthRate 1 := by sorry

end lily_pad_growth_rate_l1397_139706


namespace min_reciprocal_sum_l1397_139782

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : (a - b)^2 = 4 * (a * b)^3) : 
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ (x - y)^2 = 4 * (x * y)^3 → 1/a + 1/b ≤ 1/x + 1/y :=
by sorry

end min_reciprocal_sum_l1397_139782


namespace equation_proof_l1397_139709

theorem equation_proof : 441 + 2 * 21 * 7 + 49 = 784 := by
  sorry

end equation_proof_l1397_139709


namespace average_first_5_subjects_l1397_139757

-- Define the given conditions
def total_subjects : ℕ := 6
def average_6_subjects : ℚ := 77
def marks_6th_subject : ℕ := 92

-- Define the theorem to prove
theorem average_first_5_subjects :
  let total_marks := average_6_subjects * total_subjects
  let marks_5_subjects := total_marks - marks_6th_subject
  (marks_5_subjects / (total_subjects - 1) : ℚ) = 74 := by
  sorry

end average_first_5_subjects_l1397_139757


namespace megan_problem_solving_rate_l1397_139735

theorem megan_problem_solving_rate 
  (math_problems : ℕ) 
  (spelling_problems : ℕ) 
  (total_hours : ℕ) 
  (h1 : math_problems = 36)
  (h2 : spelling_problems = 28)
  (h3 : total_hours = 8) :
  (math_problems + spelling_problems) / total_hours = 8 :=
by
  sorry

end megan_problem_solving_rate_l1397_139735


namespace repeating_decimal_incorrect_expression_l1397_139740

/-- Represents a repeating decimal -/
structure RepeatingDecimal where
  P : ℕ  -- non-repeating part
  Q : ℕ  -- repeating part
  r : ℕ  -- number of digits in P
  s : ℕ  -- number of digits in Q

/-- The theorem stating that the given expression is not always true for repeating decimals -/
theorem repeating_decimal_incorrect_expression (D : RepeatingDecimal) :
  ¬ (∀ (D : RepeatingDecimal), 10^D.r * (10^D.s - 1) * (D.P / 10^D.r + D.Q / (10^D.r * (10^D.s - 1))) = D.Q * (D.P - 1)) :=
by sorry

end repeating_decimal_incorrect_expression_l1397_139740


namespace next_four_valid_numbers_l1397_139725

/-- Represents a bag of milk with a unique number -/
structure BagOfMilk where
  number : Nat
  h_number : number ≤ 850

/-- Checks if a number is valid for bag selection -/
def isValidNumber (n : Nat) : Bool :=
  n ≥ 1 ∧ n ≤ 850

/-- Selects the next valid numbers from a given sequence -/
def selectNextValidNumbers (sequence : List Nat) (count : Nat) : List Nat :=
  sequence.filter isValidNumber |>.take count

theorem next_four_valid_numbers 
  (sequence : List Nat)
  (h_sequence : sequence = [614, 593, 379, 242, 203, 722, 104, 887, 088]) :
  selectNextValidNumbers (sequence.drop 4) 4 = [203, 722, 104, 088] := by
  sorry

#eval selectNextValidNumbers [614, 593, 379, 242, 203, 722, 104, 887, 088] 4

end next_four_valid_numbers_l1397_139725


namespace equal_prod_of_divisors_implies_equal_numbers_l1397_139714

/-- The sum of positive divisors of a natural number -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The product of positive divisors of a natural number -/
def prod_of_divisors (n : ℕ) : ℕ := n ^ ((sum_of_divisors n).div 2)

/-- The number of positive divisors of a natural number -/
def num_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the product of all positive divisors of two natural numbers are equal, 
    then the two numbers are equal -/
theorem equal_prod_of_divisors_implies_equal_numbers (n m : ℕ) : 
  prod_of_divisors n = prod_of_divisors m → n = m := by sorry

end equal_prod_of_divisors_implies_equal_numbers_l1397_139714


namespace indefinite_integral_arctg_sqrt_2x_minus_1_l1397_139720

theorem indefinite_integral_arctg_sqrt_2x_minus_1 (x : ℝ) :
  HasDerivAt (fun x => x * Real.arctan (Real.sqrt (2 * x - 1)) - (1/2) * Real.sqrt (2 * x - 1))
             (Real.arctan (Real.sqrt (2 * x - 1)))
             x :=
by sorry

end indefinite_integral_arctg_sqrt_2x_minus_1_l1397_139720


namespace calculate_expression_l1397_139710

theorem calculate_expression : (-1 : ℤ) ^ 53 + 2 ^ (4^3 + 5^2 - 7^2) = 1099511627775 := by
  sorry

end calculate_expression_l1397_139710


namespace complex_fraction_equals_i_l1397_139758

theorem complex_fraction_equals_i : (3 + 2*I) / (2 - 3*I) = I := by sorry

end complex_fraction_equals_i_l1397_139758


namespace inequality_system_solution_l1397_139775

theorem inequality_system_solution (x : ℝ) :
  (3 * (x + 2) - x > 4 ∧ (1 + 2*x) / 3 ≥ x - 1) ↔ (-1 < x ∧ x ≤ 4) := by sorry

end inequality_system_solution_l1397_139775
