import Mathlib

namespace NUMINAMATH_CALUDE_remainder_17_63_mod_7_l1547_154705

theorem remainder_17_63_mod_7 : 17^63 % 7 = 6 := by sorry

end NUMINAMATH_CALUDE_remainder_17_63_mod_7_l1547_154705


namespace NUMINAMATH_CALUDE_algebra_test_male_students_l1547_154736

theorem algebra_test_male_students 
  (total_average : ℝ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (female_count : ℕ) :
  total_average = 90 →
  male_average = 85 →
  female_average = 92 →
  female_count = 20 →
  ∃ (male_count : ℕ), 
    (male_count : ℝ) * male_average + (female_count : ℝ) * female_average = 
      ((male_count + female_count) : ℝ) * total_average ∧
    male_count = 8 :=
by sorry

end NUMINAMATH_CALUDE_algebra_test_male_students_l1547_154736


namespace NUMINAMATH_CALUDE_balls_distribution_l1547_154791

-- Define the number of balls and boxes
def num_balls : ℕ := 6
def num_boxes : ℕ := 3

-- Define the function to calculate the number of ways to distribute balls
def distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  Nat.choose (balls + boxes - 1) (boxes - 1)

-- Theorem statement
theorem balls_distribution :
  distribute_balls num_balls num_boxes = 28 := by
  sorry

end NUMINAMATH_CALUDE_balls_distribution_l1547_154791


namespace NUMINAMATH_CALUDE_fencing_cost_is_5300_l1547_154706

/-- A rectangular plot with given dimensions and fencing cost -/
structure Plot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ
  length_breadth_difference : length = breadth + 60
  length_value : length = 80

/-- Calculate the total cost of fencing for a given plot -/
def total_fencing_cost (p : Plot) : ℝ :=
  2 * (p.length + p.breadth) * p.fencing_cost_per_meter

/-- Theorem: The total fencing cost for the given plot is 5300 currency units -/
theorem fencing_cost_is_5300 (p : Plot) (h : p.fencing_cost_per_meter = 26.50) : 
  total_fencing_cost p = 5300 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_is_5300_l1547_154706


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1547_154792

theorem rectangle_perimeter (width : ℝ) (length : ℝ) :
  width > 0 →
  length > 0 →
  length = 2 * width →
  width * length = 576 →
  2 * (width + length) = 72 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1547_154792


namespace NUMINAMATH_CALUDE_quadratic_always_negative_l1547_154732

theorem quadratic_always_negative (m : ℝ) :
  (∀ x : ℝ, m * x^2 + (m - 1) * x + (m - 1) < 0) ↔ m < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_negative_l1547_154732


namespace NUMINAMATH_CALUDE_percent_of_y_l1547_154760

theorem percent_of_y (y : ℝ) (h : y > 0) : ((4 * y) / 20 + (3 * y) / 10) / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l1547_154760


namespace NUMINAMATH_CALUDE_horner_method_f_3_l1547_154789

def f (x : ℝ) : ℝ := x^5 - 2*x^3 + 3*x^2 - x + 1

def horner_v3 (x : ℝ) : ℝ := ((((x + 0)*x - 2)*x + 3)*x - 1)*x + 1

theorem horner_method_f_3 : horner_v3 3 = 24 := by sorry

end NUMINAMATH_CALUDE_horner_method_f_3_l1547_154789


namespace NUMINAMATH_CALUDE_parallel_vectors_solution_l1547_154772

open Real

theorem parallel_vectors_solution (x : ℝ) :
  let a : ℝ × ℝ := (sin x, 3/4)
  let b : ℝ × ℝ := (1/3, (1/2) * cos x)
  let c : ℝ × ℝ := (1/6, cos x)
  0 < x ∧ x < (5 * π) / 12 ∧ 
  (∃ (k : ℝ), k * a.1 = (b.1 + c.1) ∧ k * a.2 = (b.2 + c.2)) →
  x = π / 12 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_solution_l1547_154772


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1547_154734

theorem possible_values_of_a (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^3 + y^3 = a) 
  (h3 : x^5 + y^5 = a) : 
  a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1547_154734


namespace NUMINAMATH_CALUDE_roxy_garden_plants_l1547_154784

/-- Calculates the total number of plants in Roxy's garden after buying and giving away plants -/
def total_plants_remaining (initial_flowering : ℕ) (bought_flowering : ℕ) (bought_fruiting : ℕ) 
  (given_away_flowering : ℕ) (given_away_fruiting : ℕ) : ℕ :=
  let initial_fruiting := 2 * initial_flowering
  let final_flowering := initial_flowering + bought_flowering - given_away_flowering
  let final_fruiting := initial_fruiting + bought_fruiting - given_away_fruiting
  final_flowering + final_fruiting

/-- Theorem stating that the total number of plants remaining in Roxy's garden is 21 -/
theorem roxy_garden_plants : 
  total_plants_remaining 7 3 2 1 4 = 21 := by
  sorry

end NUMINAMATH_CALUDE_roxy_garden_plants_l1547_154784


namespace NUMINAMATH_CALUDE_josh_bought_six_cds_l1547_154730

/-- Represents the shopping problem where Josh buys films, books, and CDs. -/
def ShoppingProblem (num_films num_books total_spent : ℕ) (film_cost book_cost cd_cost : ℚ) : Prop :=
  ∃ (num_cds : ℕ),
    (num_films : ℚ) * film_cost + (num_books : ℚ) * book_cost + (num_cds : ℚ) * cd_cost = total_spent

/-- Proves that Josh bought 6 CDs given the problem conditions. -/
theorem josh_bought_six_cds :
  ShoppingProblem 9 4 79 5 4 3 → (∃ (num_cds : ℕ), num_cds = 6) :=
by
  sorry

#check josh_bought_six_cds

end NUMINAMATH_CALUDE_josh_bought_six_cds_l1547_154730


namespace NUMINAMATH_CALUDE_gcd_98_63_l1547_154778

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_63_l1547_154778


namespace NUMINAMATH_CALUDE_permutation_count_modulo_l1547_154756

/-- The number of characters in the string -/
def string_length : ℕ := 15

/-- The number of A's in the string -/
def num_A : ℕ := 4

/-- The number of B's in the string -/
def num_B : ℕ := 5

/-- The number of C's in the string -/
def num_C : ℕ := 5

/-- The number of D's in the string -/
def num_D : ℕ := 2

/-- The length of the first segment where A's are not allowed -/
def first_segment : ℕ := 4

/-- The length of the second segment where B's are not allowed -/
def second_segment : ℕ := 5

/-- The length of the third segment where C's and D's are not allowed -/
def third_segment : ℕ := 6

/-- The function to calculate the number of valid permutations -/
def num_permutations : ℕ := sorry

theorem permutation_count_modulo :
  num_permutations ≡ 715 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_permutation_count_modulo_l1547_154756


namespace NUMINAMATH_CALUDE_inequality_solution_l1547_154731

theorem inequality_solution : ∀ x : ℕ+, 
  (2 * x.val + 9 ≥ 3 * (x.val + 2)) ↔ (x.val = 1 ∨ x.val = 2 ∨ x.val = 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1547_154731


namespace NUMINAMATH_CALUDE_sum_local_values_2345_l1547_154752

def local_value (digit : Nat) (place : Nat) : Nat := digit * (10 ^ place)

theorem sum_local_values_2345 :
  let thousands := local_value 2 3
  let hundreds := local_value 3 2
  let tens := local_value 4 1
  let ones := local_value 5 0
  thousands + hundreds + tens + ones = 2345 := by
sorry

end NUMINAMATH_CALUDE_sum_local_values_2345_l1547_154752


namespace NUMINAMATH_CALUDE_unique_solution_of_equation_l1547_154714

theorem unique_solution_of_equation :
  ∃! x : ℝ, (x^16 + 1) * (x^12 + x^8 + x^4 + 1) = 18 * x^8 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_of_equation_l1547_154714


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1547_154758

theorem imaginary_part_of_z (z : ℂ) (h : Complex.abs (z + 2 * Complex.I) = Complex.abs z) :
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1547_154758


namespace NUMINAMATH_CALUDE_all_cards_same_number_l1547_154727

theorem all_cards_same_number (m : ℕ) (cards : Fin m → ℕ) : 
  (∀ i : Fin m, 1 ≤ cards i ∧ cards i ≤ m) →
  (∀ s : Finset (Fin m), (s.sum cards) % (m + 1) ≠ 0) →
  ∀ i j : Fin m, cards i = cards j :=
sorry

end NUMINAMATH_CALUDE_all_cards_same_number_l1547_154727


namespace NUMINAMATH_CALUDE_stability_comparison_A_more_stable_than_B_l1547_154735

-- Define a structure for a student's test scores
structure StudentScores where
  average : ℝ
  variance : ℝ

-- Define the stability comparison function
def more_stable (a b : StudentScores) : Prop :=
  a.average = b.average ∧ a.variance < b.variance

-- Theorem statement
theorem stability_comparison (a b : StudentScores) 
  (h_avg : a.average = b.average) 
  (h_var : a.variance < b.variance) : 
  more_stable a b := by
  sorry

-- Define students A and B
def student_A : StudentScores := { average := 88, variance := 0.61 }
def student_B : StudentScores := { average := 88, variance := 0.72 }

-- Theorem application to students A and B
theorem A_more_stable_than_B : more_stable student_A student_B := by
  sorry

end NUMINAMATH_CALUDE_stability_comparison_A_more_stable_than_B_l1547_154735


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1547_154799

/-- Given a quadratic function f(x) = ax^2 + (1-a)x + a - 2,
    if f(x) ≥ -2 for all real x, then a ≥ 1/3 -/
theorem quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, a * x^2 + (1 - a) * x + a - 2 ≥ -2) → a ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1547_154799


namespace NUMINAMATH_CALUDE_work_completion_time_l1547_154771

theorem work_completion_time (ajay_time vijay_time combined_time : ℝ) : 
  ajay_time = 8 →
  combined_time = 6 →
  1 / ajay_time + 1 / vijay_time = 1 / combined_time →
  vijay_time = 24 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1547_154771


namespace NUMINAMATH_CALUDE_original_girls_count_l1547_154775

/-- Represents the number of boys and girls in a school club. -/
structure ClubMembers where
  boys : ℕ
  girls : ℕ

/-- Defines the conditions of the club membership problem. -/
def ClubProblem (initial : ClubMembers) : Prop :=
  -- Initially, there was one boy for every girl
  initial.boys = initial.girls ∧
  -- After 25 girls leave, there are three boys for each remaining girl
  3 * (initial.girls - 25) = initial.boys ∧
  -- After that, 60 boys leave, and then there are six girls for each remaining boy
  6 * (initial.boys - 60) = initial.girls - 25

/-- Theorem stating that given the conditions, the original number of girls is 67. -/
theorem original_girls_count (initial : ClubMembers) :
  ClubProblem initial → initial.girls = 67 := by
  sorry


end NUMINAMATH_CALUDE_original_girls_count_l1547_154775


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_min_dot_product_l1547_154729

-- Define the fixed point F
def F : ℝ × ℝ := (0, 1)

-- Define the line l₁
def l₁ (x : ℝ) : ℝ := -1

-- Define the trajectory of point C
def trajectory (x y : ℝ) : Prop := x^2 = 4*y

-- Define a line passing through F
def l₂ (k : ℝ) (x : ℝ) : ℝ := k*x + 1

-- Define the dot product of two 2D vectors
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  (v₁.1 * v₂.1) + (v₁.2 * v₂.2)

-- Theorem 1: The trajectory of point C is x² = 4y
theorem trajectory_is_parabola :
  ∀ (x y : ℝ), trajectory x y ↔ x^2 = 4*y :=
sorry

-- Theorem 2: The minimum value of RP · RQ is 16
theorem min_dot_product :
  ∃ (k : ℝ), 
    ∀ (x₁ y₁ x₂ y₂ : ℝ),
      trajectory x₁ y₁ →
      trajectory x₂ y₂ →
      y₁ = l₂ k x₁ →
      y₂ = l₂ k x₂ →
      let R : ℝ × ℝ := (-2/k, l₁ (-2/k));
      let P : ℝ × ℝ := (x₁, y₁);
      let Q : ℝ × ℝ := (x₂, y₂);
      dot_product (P.1 - R.1, P.2 - R.2) (Q.1 - R.1, Q.2 - R.2) ≥ 16 ∧
      (∃ (x₁' y₁' x₂' y₂' : ℝ),
        trajectory x₁' y₁' ∧
        trajectory x₂' y₂' ∧
        y₁' = l₂ k x₁' ∧
        y₂' = l₂ k x₂' ∧
        let R' : ℝ × ℝ := (-2/k, l₁ (-2/k));
        let P' : ℝ × ℝ := (x₁', y₁');
        let Q' : ℝ × ℝ := (x₂', y₂');
        dot_product (P'.1 - R'.1, P'.2 - R'.2) (Q'.1 - R'.1, Q'.2 - R'.2) = 16) :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_min_dot_product_l1547_154729


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l1547_154770

theorem smallest_k_no_real_roots : 
  ∃ (k : ℤ), (∀ (x : ℝ), 3 * x * (k * x - 5) - 2 * x^2 + 8 ≠ 0) ∧ 
  (∀ (j : ℤ), j < k → ∃ (x : ℝ), 3 * x * (j * x - 5) - 2 * x^2 + 8 = 0) := by
  sorry

#check smallest_k_no_real_roots

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l1547_154770


namespace NUMINAMATH_CALUDE_solve_equation_l1547_154739

theorem solve_equation (x : ℝ) (h : 5 - 5/x = 4 + 4/x) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1547_154739


namespace NUMINAMATH_CALUDE_point_on_number_line_l1547_154737

theorem point_on_number_line (x : ℝ) : 
  (|x| = 2021) ↔ (x = 2021 ∨ x = -2021) := by sorry

end NUMINAMATH_CALUDE_point_on_number_line_l1547_154737


namespace NUMINAMATH_CALUDE_jerry_makes_two_trips_l1547_154761

def jerry_trips (carry_capacity : ℕ) (total_trays : ℕ) : ℕ :=
  (total_trays + carry_capacity - 1) / carry_capacity

theorem jerry_makes_two_trips (carry_capacity : ℕ) (total_trays : ℕ) :
  carry_capacity = 8 → total_trays = 16 → jerry_trips carry_capacity total_trays = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerry_makes_two_trips_l1547_154761


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l1547_154709

theorem solve_cubic_equation (m : ℝ) : (m - 4)^3 = (1/8)⁻¹ ↔ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l1547_154709


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1547_154796

theorem ratio_of_sum_to_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (h : x + y = 7 * (x - y)) : x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1547_154796


namespace NUMINAMATH_CALUDE_journey_speed_problem_l1547_154798

theorem journey_speed_problem (total_distance : ℝ) (total_time : ℝ) 
  (speed1 : ℝ) (speed2 : ℝ) (segment_time : ℝ) :
  total_distance = 150 →
  total_time = 2 →
  speed1 = 50 →
  speed2 = 70 →
  segment_time = 2/3 →
  ∃ (speed3 : ℝ),
    speed3 = 105 ∧
    total_distance = speed1 * segment_time + speed2 * segment_time + speed3 * segment_time :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_problem_l1547_154798


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l1547_154764

theorem circle_tangent_to_line (r : ℝ) (h : r = Real.sqrt 5) :
  ∃ (c1 c2 : ℝ × ℝ),
    c1.2 = 0 ∧ c2.2 = 0 ∧
    (∀ (x y : ℝ), (x - c1.1)^2 + (y - c1.2)^2 = r^2 ↔ (x + 2*y)^2 = 5) ∧
    (∀ (x y : ℝ), (x - c2.1)^2 + (y - c2.2)^2 = r^2 ↔ (x + 2*y)^2 = 5) ∧
    c1 = (5, 0) ∧ c2 = (-5, 0) := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l1547_154764


namespace NUMINAMATH_CALUDE_line_circle_intersection_l1547_154720

/-- Given a line mx + y + √3 = 0 intersecting a circle (x+1)² + y² = 2 with a chord length of 2,
    prove that m = √3/3 -/
theorem line_circle_intersection (m : ℝ) : 
  (∃ x y : ℝ, mx + y + Real.sqrt 3 = 0 ∧ (x + 1)^2 + y^2 = 2) → 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    mx₁ + y₁ + Real.sqrt 3 = 0 ∧ (x₁ + 1)^2 + y₁^2 = 2 ∧
    mx₂ + y₂ + Real.sqrt 3 = 0 ∧ (x₂ + 1)^2 + y₂^2 = 2 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4) →
  m = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l1547_154720


namespace NUMINAMATH_CALUDE_orange_cost_24_pounds_l1547_154726

/-- The cost of oranges given a rate and a quantity -/
def orange_cost (rate_price : ℚ) (rate_weight : ℚ) (weight : ℚ) : ℚ :=
  (rate_price / rate_weight) * weight

/-- Theorem: The cost of 24 pounds of oranges at a rate of $6 per 8 pounds is $18 -/
theorem orange_cost_24_pounds : orange_cost 6 8 24 = 18 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_24_pounds_l1547_154726


namespace NUMINAMATH_CALUDE_jinho_money_problem_l1547_154713

theorem jinho_money_problem (initial_money : ℕ) : 
  (initial_money / 2 + 300 + 
   ((initial_money - (initial_money / 2 + 300)) / 2 + 400) = initial_money) → 
  initial_money = 2200 :=
by sorry

end NUMINAMATH_CALUDE_jinho_money_problem_l1547_154713


namespace NUMINAMATH_CALUDE_league_members_l1547_154723

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 7

/-- The cost of shorts in dollars -/
def shorts_cost : ℕ := tshirt_cost

/-- The total cost for one member's equipment in dollars -/
def member_cost : ℕ := 2 * (sock_cost + tshirt_cost + shorts_cost)

/-- The total cost for the league's equipment in dollars -/
def total_cost : ℕ := 4719

/-- The number of members in the league -/
def num_members : ℕ := 74

theorem league_members : 
  sock_cost = 6 ∧ 
  tshirt_cost = sock_cost + 7 ∧ 
  shorts_cost = tshirt_cost ∧
  member_cost = 2 * (sock_cost + tshirt_cost + shorts_cost) ∧
  total_cost = 4719 → 
  num_members * member_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_league_members_l1547_154723


namespace NUMINAMATH_CALUDE_dean_taller_than_ron_l1547_154700

theorem dean_taller_than_ron (water_depth : ℝ) (ron_height : ℝ) (dean_height : ℝ) :
  water_depth = 255 ∧ ron_height = 13 ∧ water_depth = 15 * dean_height →
  dean_height - ron_height = 4 := by
  sorry

end NUMINAMATH_CALUDE_dean_taller_than_ron_l1547_154700


namespace NUMINAMATH_CALUDE_joe_paint_usage_l1547_154787

/-- The amount of paint Joe used in total -/
def total_paint_used (initial_paint : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) : ℚ :=
  let first_week_usage := first_week_fraction * initial_paint
  let remaining_paint := initial_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage

/-- Theorem stating that Joe used 264 gallons of paint -/
theorem joe_paint_usage :
  total_paint_used 360 (2/3) (1/5) = 264 := by
  sorry

end NUMINAMATH_CALUDE_joe_paint_usage_l1547_154787


namespace NUMINAMATH_CALUDE_fresh_fruits_ratio_l1547_154711

/-- Represents the quantity of fruits in the store -/
structure FruitQuantity where
  pineapples : ℕ
  apples : ℕ
  oranges : ℕ

/-- Represents the spoilage rates of fruits -/
structure SpoilageRate where
  pineapples : ℚ
  apples : ℚ
  oranges : ℚ

def initialQuantity : FruitQuantity :=
  { pineapples := 200, apples := 300, oranges := 100 }

def soldQuantity : FruitQuantity :=
  { pineapples := 56, apples := 128, oranges := 22 }

def spoilageRate : SpoilageRate :=
  { pineapples := 1/10, apples := 15/100, oranges := 1/20 }

def remainingFruits (initial : FruitQuantity) (sold : FruitQuantity) : FruitQuantity :=
  { pineapples := initial.pineapples - sold.pineapples,
    apples := initial.apples - sold.apples,
    oranges := initial.oranges - sold.oranges }

def spoiledFruits (remaining : FruitQuantity) (rate : SpoilageRate) : FruitQuantity :=
  { pineapples := (remaining.pineapples : ℚ) * rate.pineapples |> round |> Int.toNat,
    apples := (remaining.apples : ℚ) * rate.apples |> round |> Int.toNat,
    oranges := (remaining.oranges : ℚ) * rate.oranges |> round |> Int.toNat }

def freshFruits (remaining : FruitQuantity) (spoiled : FruitQuantity) : FruitQuantity :=
  { pineapples := remaining.pineapples - spoiled.pineapples,
    apples := remaining.apples - spoiled.apples,
    oranges := remaining.oranges - spoiled.oranges }

theorem fresh_fruits_ratio :
  let remaining := remainingFruits initialQuantity soldQuantity
  let spoiled := spoiledFruits remaining spoilageRate
  let fresh := freshFruits remaining spoiled
  fresh.pineapples = 130 ∧ fresh.apples = 146 ∧ fresh.oranges = 74 := by sorry

end NUMINAMATH_CALUDE_fresh_fruits_ratio_l1547_154711


namespace NUMINAMATH_CALUDE_one_fifths_in_one_tenth_l1547_154788

theorem one_fifths_in_one_tenth : (1 / 10 : ℚ) / (1 / 5 : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_fifths_in_one_tenth_l1547_154788


namespace NUMINAMATH_CALUDE_grasshopper_final_position_l1547_154749

/-- The number of positions in the circular arrangement -/
def num_positions : ℕ := 6

/-- The number of jumps the grasshopper makes -/
def num_jumps : ℕ := 100

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The final position of the grasshopper after num_jumps -/
def final_position : ℕ := (sum_first_n num_jumps) % num_positions + 1

theorem grasshopper_final_position :
  final_position = 5 := by sorry

end NUMINAMATH_CALUDE_grasshopper_final_position_l1547_154749


namespace NUMINAMATH_CALUDE_integer_part_sqrt_39_minus_3_l1547_154751

theorem integer_part_sqrt_39_minus_3 : 
  ⌊Real.sqrt 39 - 3⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_integer_part_sqrt_39_minus_3_l1547_154751


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1547_154785

theorem diophantine_equation_solutions :
  (∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    4 * p.1 + 7 * p.2 = 1003 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 1004) (Finset.range 1004))).card ∧ n = 36) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1547_154785


namespace NUMINAMATH_CALUDE_tan_205_in_terms_of_cos_155_l1547_154744

theorem tan_205_in_terms_of_cos_155 (a : ℝ) (h : Real.cos (155 * π / 180) = a) :
  Real.tan (205 * π / 180) = -Real.sqrt (1 - a^2) / a := by
  sorry

end NUMINAMATH_CALUDE_tan_205_in_terms_of_cos_155_l1547_154744


namespace NUMINAMATH_CALUDE_smallest_valid_n_l1547_154769

def is_valid (n : ℕ) : Prop :=
  n > 1 ∧ ∀ p, Nat.Prime p → p ∣ n → n > 1200 * p

theorem smallest_valid_n :
  is_valid 3888 ∧ ∀ m, m < 3888 → ¬is_valid m :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l1547_154769


namespace NUMINAMATH_CALUDE_all_contradictions_valid_l1547_154783

/-- A type representing the different kinds of contradictions in a proof by contradiction -/
inductive ContradictionType
  | KnownFact
  | Assumption
  | DefinitionTheoremAxiomLaw
  | Fact

/-- Definition of a valid contradiction in a proof by contradiction -/
def is_valid_contradiction (c : ContradictionType) : Prop :=
  match c with
  | ContradictionType.KnownFact => True
  | ContradictionType.Assumption => True
  | ContradictionType.DefinitionTheoremAxiomLaw => True
  | ContradictionType.Fact => True

/-- Theorem stating that all types of contradictions are valid in a proof by contradiction -/
theorem all_contradictions_valid :
  ∀ (c : ContradictionType), is_valid_contradiction c :=
by sorry

end NUMINAMATH_CALUDE_all_contradictions_valid_l1547_154783


namespace NUMINAMATH_CALUDE_symmetric_points_count_l1547_154718

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x^2 + 4 * x + 1 else 2 / Real.exp x

theorem symmetric_points_count :
  ∃! (p : ℕ), p = 2 ∧
  ∃ (S : Finset (ℝ × ℝ)),
    S.card = p ∧
    (∀ (x y : ℝ), (x, y) ∈ S → y = f x) ∧
    (∀ (x y : ℝ), (x, y) ∈ S → (-x, -y) ∈ S) ∧
    (∀ (x y : ℝ), (x, y) ∈ S → (x ≠ 0 ∨ y ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_count_l1547_154718


namespace NUMINAMATH_CALUDE_remaining_sales_l1547_154738

-- Define the weekly goal
def weekly_goal : ℕ := 90

-- Define Monday's sales
def monday_sales : ℕ := 45

-- Define Tuesday's sales
def tuesday_sales : ℕ := monday_sales - 16

-- Define the total sales so far
def total_sales : ℕ := monday_sales + tuesday_sales

-- Theorem to prove
theorem remaining_sales : weekly_goal - total_sales = 16 := by
  sorry

end NUMINAMATH_CALUDE_remaining_sales_l1547_154738


namespace NUMINAMATH_CALUDE_football_season_games_james_football_season_l1547_154779

/-- Calculates the number of games in a football season based on a player's performance -/
theorem football_season_games (touchdowns_per_game : ℕ) (points_per_touchdown : ℕ) 
  (two_point_conversions : ℕ) (old_record : ℕ) (points_above_record : ℕ) : ℕ :=
  let total_points := old_record + points_above_record
  let points_from_conversions := two_point_conversions * 2
  let points_from_touchdowns := total_points - points_from_conversions
  let points_per_game := touchdowns_per_game * points_per_touchdown
  points_from_touchdowns / points_per_game

/-- The number of games in James' football season -/
theorem james_football_season : 
  football_season_games 4 6 6 300 72 = 15 := by
  sorry

end NUMINAMATH_CALUDE_football_season_games_james_football_season_l1547_154779


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1547_154754

-- Define sets A and B
def A : Set (ℝ × ℝ) := {p | 3 * p.1 + p.2 = 0}
def B : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 = 3}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {(3/5, -9/5)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1547_154754


namespace NUMINAMATH_CALUDE_subcommittee_count_l1547_154722

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid subcommittees -/
def validSubcommittees (totalMembers teachers subcommitteeSize : ℕ) : ℕ :=
  choose totalMembers subcommitteeSize - choose (totalMembers - teachers) subcommitteeSize

theorem subcommittee_count :
  validSubcommittees 12 5 5 = 771 := by sorry

end NUMINAMATH_CALUDE_subcommittee_count_l1547_154722


namespace NUMINAMATH_CALUDE_five_variable_inequality_l1547_154755

theorem five_variable_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) : 
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 > 4 * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅ + x₅*x₁) := by
  sorry

end NUMINAMATH_CALUDE_five_variable_inequality_l1547_154755


namespace NUMINAMATH_CALUDE_correct_selection_ways_l1547_154746

def total_students : ℕ := 50
def class_leaders : ℕ := 2
def students_to_select : ℕ := 5

def selection_ways : ℕ := sorry

theorem correct_selection_ways :
  selection_ways = Nat.choose class_leaders 1 * Nat.choose (total_students - class_leaders) 4 +
                   Nat.choose class_leaders 2 * Nat.choose (total_students - class_leaders) 3 ∧
  selection_ways = Nat.choose total_students students_to_select - 
                   Nat.choose (total_students - class_leaders) students_to_select ∧
  selection_ways = Nat.choose class_leaders 1 * Nat.choose (total_students - 1) 4 - 
                   Nat.choose (total_students - class_leaders) 3 ∧
  selection_ways ≠ Nat.choose class_leaders 1 * Nat.choose (total_students - 1) 4 :=
by sorry

end NUMINAMATH_CALUDE_correct_selection_ways_l1547_154746


namespace NUMINAMATH_CALUDE_factors_of_34848_l1547_154708

/-- The number of positive factors of 34848 -/
def num_factors : ℕ := 54

/-- 34848 as a natural number -/
def n : ℕ := 34848

theorem factors_of_34848 : Nat.card (Nat.divisors n) = num_factors := by
  sorry

end NUMINAMATH_CALUDE_factors_of_34848_l1547_154708


namespace NUMINAMATH_CALUDE_water_difference_proof_l1547_154741

/-- The difference in initial water amounts between Ji-hoon and Hyo-joo, given the conditions of the problem -/
def water_difference (j h : ℕ) : Prop :=
  (j - 152 = h + 152 + 346) → (j - h = 650)

/-- Theorem stating the water difference problem -/
theorem water_difference_proof :
  ∀ j h : ℕ, water_difference j h :=
by
  sorry

end NUMINAMATH_CALUDE_water_difference_proof_l1547_154741


namespace NUMINAMATH_CALUDE_additional_interest_proof_l1547_154710

/-- Calculate the simple interest given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time

theorem additional_interest_proof :
  let principal : ℚ := 2500
  let time : ℚ := 2
  let higherRate : ℚ := 18 / 100
  let lowerRate : ℚ := 12 / 100
  simpleInterest principal higherRate time - simpleInterest principal lowerRate time = 300 := by
sorry

end NUMINAMATH_CALUDE_additional_interest_proof_l1547_154710


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l1547_154745

theorem cube_root_equation_solution :
  ∃! x : ℚ, Real.rpow (5 + x) (1/3 : ℝ) = 4/3 :=
by
  use -71/27
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l1547_154745


namespace NUMINAMATH_CALUDE_nut_mixture_weight_l1547_154774

/-- Represents a mixture of nuts -/
structure NutMixture where
  almond_ratio : ℚ
  walnut_ratio : ℚ
  almond_weight : ℚ

/-- Calculates the total weight of a nut mixture -/
def total_weight (mix : NutMixture) : ℚ :=
  (mix.almond_weight / mix.almond_ratio) * (mix.almond_ratio + mix.walnut_ratio)

/-- Theorem: The total weight of the given nut mixture is 210 pounds -/
theorem nut_mixture_weight :
  let mix : NutMixture := {
    almond_ratio := 5,
    walnut_ratio := 2,
    almond_weight := 150
  }
  total_weight mix = 210 := by
  sorry

end NUMINAMATH_CALUDE_nut_mixture_weight_l1547_154774


namespace NUMINAMATH_CALUDE_parallel_tangents_imply_m_value_monotonicity_intervals_l1547_154733

noncomputable section

variable (m : ℝ)

def f (x : ℝ) : ℝ := (1/2) * m * x^2 + 1

def g (x : ℝ) : ℝ := 2 * Real.log x - (2*m + 1) * x - 1

def h (x : ℝ) : ℝ := f m x + g m x

def h_derivative (x : ℝ) : ℝ := m * x - (2*m + 1) + 2 / x

theorem parallel_tangents_imply_m_value :
  (h_derivative m 1 = h_derivative m 3) → m = 2/3 := by sorry

theorem monotonicity_intervals (x : ℝ) (hx : x > 0) :
  (m ≤ 0 → 
    (x < 2 → h_derivative m x > 0) ∧ 
    (x > 2 → h_derivative m x < 0)) ∧
  (0 < m ∧ m < 1/2 → 
    ((x < 2 ∨ x > 1/m) → h_derivative m x > 0) ∧ 
    (2 < x ∧ x < 1/m → h_derivative m x < 0)) ∧
  (m = 1/2 → h_derivative m x > 0) ∧
  (m > 1/2 → 
    ((x < 1/m ∨ x > 2) → h_derivative m x > 0) ∧ 
    (1/m < x ∧ x < 2 → h_derivative m x < 0)) := by sorry

end

end NUMINAMATH_CALUDE_parallel_tangents_imply_m_value_monotonicity_intervals_l1547_154733


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l1547_154766

theorem orthogonal_vectors (y : ℝ) : 
  (2 * -3 + -4 * y + 5 * 2 = 0) ↔ (y = 1) :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_l1547_154766


namespace NUMINAMATH_CALUDE_rectangle_ratio_is_two_l1547_154797

/-- Represents the configuration of squares and rectangles -/
structure SquareRectConfig where
  inner_side : ℝ
  outer_side : ℝ
  rect_short : ℝ
  rect_long : ℝ
  area_ratio : ℝ
  h_area_ratio : area_ratio = 9
  h_outer_side : outer_side = inner_side + 2 * rect_short
  h_rect_long : rect_long + rect_short = outer_side

/-- The ratio of the longer side to the shorter side of the rectangle is 2 -/
theorem rectangle_ratio_is_two (config : SquareRectConfig) :
  config.rect_long / config.rect_short = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_is_two_l1547_154797


namespace NUMINAMATH_CALUDE_integer_solution_system_l1547_154757

theorem integer_solution_system (a b c : ℤ) : 
  a^2 = b*c + 1 ∧ b^2 = a*c + 1 ↔ 
  (a = 1 ∧ b = 0 ∧ c = -1) ∨
  (a = 1 ∧ b = -1 ∧ c = 0) ∨
  (a = -1 ∧ b = 0 ∧ c = 1) ∨
  (a = -1 ∧ b = 1 ∧ c = 0) := by
sorry

end NUMINAMATH_CALUDE_integer_solution_system_l1547_154757


namespace NUMINAMATH_CALUDE_rest_stop_distance_l1547_154725

/-- Proves that the distance between rest stops is 10 miles for a man walking 50 miles in 320 minutes with given conditions. -/
theorem rest_stop_distance (walking_speed : ℝ) (rest_duration : ℝ) (total_distance : ℝ) (total_time : ℝ) 
  (h1 : walking_speed = 10) -- walking speed in mph
  (h2 : rest_duration = 5 / 60) -- rest duration in hours
  (h3 : total_distance = 50) -- total distance in miles
  (h4 : total_time = 320 / 60) -- total time in hours
  : ∃ (x : ℝ), x = 10 ∧ 
    (total_distance / walking_speed + rest_duration * (total_distance / x - 1) = total_time) := by
  sorry


end NUMINAMATH_CALUDE_rest_stop_distance_l1547_154725


namespace NUMINAMATH_CALUDE_ice_cream_permutations_l1547_154782

theorem ice_cream_permutations :
  Finset.card (Finset.univ.image (fun σ : Equiv.Perm (Fin 5) => σ)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_permutations_l1547_154782


namespace NUMINAMATH_CALUDE_inequalities_hold_l1547_154777

theorem inequalities_hold (m n : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) : 
  (m - Real.sqrt m ≥ -1/4) ∧ 
  (1/2 * (m + n)^2 + 1/4 * (m + n) ≥ m * Real.sqrt n + n * Real.sqrt m) := by
  sorry

#check inequalities_hold

end NUMINAMATH_CALUDE_inequalities_hold_l1547_154777


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1547_154786

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 5*x = 4) ↔ (∃ x : ℝ, x^2 + 5*x ≠ 4) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1547_154786


namespace NUMINAMATH_CALUDE_inequality_theorem_l1547_154748

-- Define the functions p and q
variable (p q : ℝ → ℝ)

-- Define the theorem
theorem inequality_theorem 
  (h1 : Differentiable ℝ p) 
  (h2 : Differentiable ℝ q)
  (h3 : p 0 = q 0)
  (h4 : p 0 > 0)
  (h5 : ∀ x ∈ Set.Icc 0 1, deriv p x * Real.sqrt (deriv q x) = Real.sqrt 2) :
  ∀ x ∈ Set.Icc 0 1, p x + 2 * q x > 3 * x := by
sorry


end NUMINAMATH_CALUDE_inequality_theorem_l1547_154748


namespace NUMINAMATH_CALUDE_min_sum_distances_l1547_154717

/-- The minimum sum of distances from a point on the x-axis to two fixed points -/
theorem min_sum_distances (P : ℝ × ℝ) (A B : ℝ × ℝ) : 
  A = (1, 1) → B = (3, 4) → P.2 = 0 → 
  ∀ Q : ℝ × ℝ, Q.2 = 0 → Real.sqrt 29 ≤ dist P A + dist P B :=
sorry

end NUMINAMATH_CALUDE_min_sum_distances_l1547_154717


namespace NUMINAMATH_CALUDE_yanni_paintings_area_l1547_154762

/-- The total area of Yanni's paintings -/
def total_area (num_small : ℕ) (small_side : ℕ) (large_length : ℕ) (large_width : ℕ) (final_height : ℕ) (final_width : ℕ) : ℕ :=
  num_small * small_side * small_side + large_length * large_width + final_height * final_width

/-- Theorem stating that the total area of Yanni's paintings is 200 square feet -/
theorem yanni_paintings_area :
  total_area 3 5 10 8 5 9 = 200 := by
  sorry

end NUMINAMATH_CALUDE_yanni_paintings_area_l1547_154762


namespace NUMINAMATH_CALUDE_circle_center_sum_l1547_154747

/-- Given a circle with equation x^2 + y^2 = 6x + 8y - 48, 
    the sum of the coordinates of its center is 7 -/
theorem circle_center_sum : 
  ∀ (h k : ℝ), 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 8*y - 48 ↔ (x - h)^2 + (y - k)^2 = 2) →
  h + k = 7 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1547_154747


namespace NUMINAMATH_CALUDE_ice_cream_flavors_count_l1547_154704

/-- The number of ways to distribute n indistinguishable objects into k distinguishable categories -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of unique ice cream flavors that can be created -/
def ice_cream_flavors : ℕ := stars_and_bars 4 4

theorem ice_cream_flavors_count : ice_cream_flavors = 35 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_count_l1547_154704


namespace NUMINAMATH_CALUDE_product_of_four_numbers_l1547_154712

theorem product_of_four_numbers (a b c d : ℚ) : 
  a + b + c + d = 36 →
  a = 3 * (b + c + d) →
  b = 5 * c →
  d = (1/2) * c →
  a * b * c * d = 178.5 := by
sorry

end NUMINAMATH_CALUDE_product_of_four_numbers_l1547_154712


namespace NUMINAMATH_CALUDE_distance_between_points_l1547_154795

/-- The distance between two points (-3, 5) and (4, -9) is √245 -/
theorem distance_between_points : Real.sqrt 245 = Real.sqrt ((4 - (-3))^2 + (-9 - 5)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1547_154795


namespace NUMINAMATH_CALUDE_cut_cube_edge_count_l1547_154701

/-- Represents a cube with smaller cubes removed from its corners -/
structure CutCube where
  side_length : ℝ
  cut_length : ℝ

/-- Calculates the number of edges in a CutCube -/
def edge_count (c : CutCube) : ℕ :=
  12 + 8 * 9 / 3

/-- Theorem stating that a cube of side length 4 with corners of side length 1.5 removed has 36 edges -/
theorem cut_cube_edge_count :
  let c : CutCube := { side_length := 4, cut_length := 1.5 }
  edge_count c = 36 := by
  sorry

end NUMINAMATH_CALUDE_cut_cube_edge_count_l1547_154701


namespace NUMINAMATH_CALUDE_number_and_square_sum_l1547_154743

theorem number_and_square_sum (x : ℝ) : x + x^2 = 342 → x = 18 ∨ x = -19 := by
  sorry

end NUMINAMATH_CALUDE_number_and_square_sum_l1547_154743


namespace NUMINAMATH_CALUDE_soda_price_increase_l1547_154780

theorem soda_price_increase (initial_total : ℝ) (new_candy_price new_soda_price : ℝ) 
  (candy_increase : ℝ) :
  initial_total = 16 →
  new_candy_price = 10 →
  new_soda_price = 12 →
  candy_increase = 0.25 →
  (new_soda_price / (initial_total - new_candy_price / (1 + candy_increase)) - 1) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_increase_l1547_154780


namespace NUMINAMATH_CALUDE_probability_not_snowing_l1547_154768

theorem probability_not_snowing (p_snowing : ℚ) (h : p_snowing = 5/8) : 
  1 - p_snowing = 3/8 := by
sorry

end NUMINAMATH_CALUDE_probability_not_snowing_l1547_154768


namespace NUMINAMATH_CALUDE_arithmetic_progression_cubic_coeff_conditions_l1547_154765

/-- A cubic polynomial with coefficients a, b, c whose roots form an arithmetic progression -/
structure ArithmeticProgressionCubic where
  a : ℝ
  b : ℝ
  c : ℝ
  roots_in_ap : ∃ (r₁ r₂ r₃ : ℝ), r₁ < r₂ ∧ r₂ < r₃ ∧
    r₂ - r₁ = r₃ - r₂ ∧
    r₁ + r₂ + r₃ = -a ∧
    r₁ * r₂ + r₁ * r₃ + r₂ * r₃ = b ∧
    r₁ * r₂ * r₃ = -c

/-- The coefficients of a cubic polynomial with roots in arithmetic progression satisfy specific conditions -/
theorem arithmetic_progression_cubic_coeff_conditions (p : ArithmeticProgressionCubic) :
  27 * p.c = 3 * p.a * p.b - 2 * p.a^3 ∧ 3 * p.b ≤ p.a^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_cubic_coeff_conditions_l1547_154765


namespace NUMINAMATH_CALUDE_only_solutions_are_24_and_42_l1547_154793

/-- Reverses the digits of a natural number -/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- Computes the product of digits of a natural number -/
def product_of_digits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number has no zeros in its decimal representation -/
def no_zeros (n : ℕ) : Prop := sorry

/-- The main theorem stating that 24 and 42 are the only solutions -/
theorem only_solutions_are_24_and_42 :
  {X : ℕ | no_zeros X ∧ X * (reverse_digits X) = 1000 + product_of_digits X} = {24, 42} :=
sorry

end NUMINAMATH_CALUDE_only_solutions_are_24_and_42_l1547_154793


namespace NUMINAMATH_CALUDE_owen_sleep_hours_l1547_154721

theorem owen_sleep_hours (total_hours work_hours chore_hours sleep_hours : ℕ) :
  total_hours = 24 ∧ work_hours = 6 ∧ chore_hours = 7 ∧ 
  sleep_hours = total_hours - (work_hours + chore_hours) →
  sleep_hours = 11 := by
  sorry

end NUMINAMATH_CALUDE_owen_sleep_hours_l1547_154721


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1547_154724

theorem complex_modulus_problem (x y : ℝ) :
  (Complex.I * (x + 2 * Complex.I) = y - Complex.I) →
  Complex.abs (x - y * Complex.I) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1547_154724


namespace NUMINAMATH_CALUDE_remainder_problem_l1547_154703

theorem remainder_problem (k : ℕ) (h1 : k > 0) (h2 : k < 84) 
  (h3 : k % 5 = 2) (h4 : k % 6 = 5) (h5 : k % 8 = 7) : k % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1547_154703


namespace NUMINAMATH_CALUDE_product_equation_minimum_sum_l1547_154715

theorem product_equation_minimum_sum (x y z a : ℤ) : 
  (x - 10) * (y - a) * (z - 2) = 1000 →
  x + y + z ≥ 7 →
  (∀ x' y' z' : ℤ, (x' - 10) * (y' - a) * (z' - 2) = 1000 → x' + y' + z' ≥ x + y + z) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equation_minimum_sum_l1547_154715


namespace NUMINAMATH_CALUDE_second_night_difference_l1547_154763

/-- The sleep pattern over four nights -/
structure SleepPattern where
  first_night : ℝ
  second_night : ℝ
  third_night : ℝ
  fourth_night : ℝ

/-- The conditions of Billy's sleep pattern -/
def billySleepPattern (sp : SleepPattern) : Prop :=
  sp.first_night = 6 ∧
  sp.third_night = sp.second_night / 2 ∧
  sp.fourth_night = 3 * sp.third_night ∧
  sp.first_night + sp.second_night + sp.third_night + sp.fourth_night = 30

/-- The theorem stating the difference in sleep between the second and first night -/
theorem second_night_difference (sp : SleepPattern) 
  (h : billySleepPattern sp) : sp.second_night - sp.first_night = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_night_difference_l1547_154763


namespace NUMINAMATH_CALUDE_remaining_sugar_is_one_l1547_154767

/-- Represents the recipe and Mary's baking process -/
structure Recipe where
  sugar_total : ℕ
  salt_total : ℕ
  flour_added : ℕ
  sugar_salt_diff : ℕ

/-- Calculates the remaining sugar to be added based on the recipe and current state -/
def remaining_sugar (r : Recipe) : ℕ :=
  r.sugar_total - (r.salt_total + r.sugar_salt_diff)

/-- Theorem stating that the remaining sugar to be added is 1 cup -/
theorem remaining_sugar_is_one (r : Recipe) 
  (h1 : r.sugar_total = 8)
  (h2 : r.salt_total = 7)
  (h3 : r.flour_added = 5)
  (h4 : r.sugar_salt_diff = 1) : 
  remaining_sugar r = 1 := by
  sorry

#check remaining_sugar_is_one

end NUMINAMATH_CALUDE_remaining_sugar_is_one_l1547_154767


namespace NUMINAMATH_CALUDE_opposite_of_negative_twelve_l1547_154728

-- Define the concept of opposite for integers
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_negative_twelve : opposite (-12) = 12 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_twelve_l1547_154728


namespace NUMINAMATH_CALUDE_find_M_l1547_154753

theorem find_M (p q r s M : ℚ) 
  (sum_eq : p + q + r + s = 100)
  (p_eq : p + 10 = M)
  (q_eq : q - 5 = M)
  (r_eq : 10 * r = M)
  (s_eq : s / 2 = M) :
  M = 1050 / 41 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l1547_154753


namespace NUMINAMATH_CALUDE_final_numbers_correct_l1547_154759

/-- The number of elements in the initial sequence -/
def n : ℕ := 2022

/-- The number of operations performed -/
def operations : ℕ := (n - 2) / 2

/-- The arithmetic mean operation on squares -/
def arithmetic_mean_operation (x : ℕ) : ℕ := x^2 + 1

/-- The final two numbers after applying the arithmetic mean operation -/
def final_numbers : Fin 2 → ℕ
| 0 => arithmetic_mean_operation 1011 + operations
| 1 => arithmetic_mean_operation 1012 + operations

theorem final_numbers_correct :
  final_numbers 0 = 1023131 ∧ final_numbers 1 = 1025154 := by sorry

end NUMINAMATH_CALUDE_final_numbers_correct_l1547_154759


namespace NUMINAMATH_CALUDE_gcd_12m_18n_lower_bound_l1547_154702

theorem gcd_12m_18n_lower_bound (m n : ℕ+) (h : Nat.gcd m n = 18) :
  Nat.gcd (12 * m) (18 * n) ≥ 108 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12m_18n_lower_bound_l1547_154702


namespace NUMINAMATH_CALUDE_pool_fill_time_l1547_154776

/-- Proves that a pool with the given specifications takes 25 hours to fill -/
theorem pool_fill_time (pool_volume : ℝ) (hose1_rate : ℝ) (hose2_rate : ℝ) 
  (h_volume : pool_volume = 15000)
  (h_hose1 : hose1_rate = 2)
  (h_hose2 : hose2_rate = 3) : 
  pool_volume / (2 * hose1_rate + 2 * hose2_rate) / 60 = 25 := by
  sorry

end NUMINAMATH_CALUDE_pool_fill_time_l1547_154776


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l1547_154781

theorem cubic_inequality_solution (x : ℝ) : 
  -2 * x^3 + 5 * x^2 + 7 * x - 10 < 0 ↔ x < -1.35 ∨ (1.85 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l1547_154781


namespace NUMINAMATH_CALUDE_function_transformation_l1547_154719

theorem function_transformation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = x^2) :
  ∀ x : ℝ, f x = (x + 1)^2 := by
sorry

end NUMINAMATH_CALUDE_function_transformation_l1547_154719


namespace NUMINAMATH_CALUDE_representation_of_231_l1547_154716

theorem representation_of_231 : 
  ∃ (list : List ℕ), (list.sum = 231) ∧ (list.prod = 231) := by
  sorry

end NUMINAMATH_CALUDE_representation_of_231_l1547_154716


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l1547_154773

theorem quadratic_form_equivalence :
  ∀ x : ℝ, 2 * x^2 + 3 * x - 1 = 2 * (x + 3/4)^2 - 17/8 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l1547_154773


namespace NUMINAMATH_CALUDE_pascal_triangle_row20_element5_l1547_154750

theorem pascal_triangle_row20_element5 : Nat.choose 20 4 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row20_element5_l1547_154750


namespace NUMINAMATH_CALUDE_probability_at_least_one_head_l1547_154707

theorem probability_at_least_one_head (p : ℝ) (n : ℕ) : 
  p = 1/2 → n = 3 → 1 - (1 - p)^n = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_head_l1547_154707


namespace NUMINAMATH_CALUDE_monkey_doll_difference_l1547_154740

theorem monkey_doll_difference (total_budget : ℕ) (large_doll_cost : ℕ) (cost_difference : ℕ) : 
  total_budget = 300 → 
  large_doll_cost = 6 → 
  cost_difference = 2 → 
  (total_budget / (large_doll_cost - cost_difference) : ℕ) - (total_budget / large_doll_cost : ℕ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_monkey_doll_difference_l1547_154740


namespace NUMINAMATH_CALUDE_essay_competition_probability_l1547_154790

theorem essay_competition_probability (n : ℕ) (h : n = 6) :
  let total_outcomes := n * n
  let favorable_outcomes := n * (n - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 6 :=
by sorry

end NUMINAMATH_CALUDE_essay_competition_probability_l1547_154790


namespace NUMINAMATH_CALUDE_flagpole_height_l1547_154742

/-- The height of a flagpole given shadow lengths -/
theorem flagpole_height (flagpole_shadow : ℝ) (tree_height : ℝ) (tree_shadow : ℝ)
  (h1 : flagpole_shadow = 90)
  (h2 : tree_height = 15)
  (h3 : tree_shadow = 30)
  : ∃ (flagpole_height : ℝ), flagpole_height = 45 :=
by sorry

end NUMINAMATH_CALUDE_flagpole_height_l1547_154742


namespace NUMINAMATH_CALUDE_tile_border_ratio_l1547_154794

theorem tile_border_ratio (s d : ℝ) (h1 : s > 0) (h2 : d > 0) : 
  (25 * s)^2 / ((25 * s + 2 * d)^2) = 0.81 → d / s = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l1547_154794
