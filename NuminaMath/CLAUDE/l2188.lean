import Mathlib

namespace NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l2188_218848

theorem tan_45_degrees_equals_one : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l2188_218848


namespace NUMINAMATH_CALUDE_function_equality_l2188_218876

theorem function_equality (x : ℝ) : x = Real.log (Real.exp x) := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l2188_218876


namespace NUMINAMATH_CALUDE_orthocenter_symmetry_and_equal_circles_l2188_218821

/-- A circle in a plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- A point in a plane -/
def Point := ℝ × ℝ

/-- The orthocenter of a triangle -/
def orthocenter (A B C : Point) : Point := sorry

/-- Checks if four points are on the same circle -/
def on_same_circle (A B C D : Point) (S : Circle) : Prop := sorry

/-- Checks if two quadrilaterals are symmetric with respect to a point -/
def symmetric_quadrilaterals (A B C D A' B' C' D' H : Point) : Prop := sorry

/-- Checks if four points are on a circle with the same radius as another circle -/
def on_equal_circle (A B C D : Point) (S : Circle) : Prop := sorry

theorem orthocenter_symmetry_and_equal_circles 
  (A₁ A₂ A₃ A₄ : Point) (S : Circle)
  (h_same_circle : on_same_circle A₁ A₂ A₃ A₄ S)
  (H₁ := orthocenter A₂ A₃ A₄)
  (H₂ := orthocenter A₁ A₃ A₄)
  (H₃ := orthocenter A₁ A₂ A₄)
  (H₄ := orthocenter A₁ A₂ A₃) :
  ∃ (H : Point),
    (symmetric_quadrilaterals A₁ A₂ A₃ A₄ H₁ H₂ H₃ H₄ H) ∧
    (on_equal_circle A₁ A₂ H₃ H₄ S) ∧
    (on_equal_circle A₁ A₃ H₂ H₄ S) ∧
    (on_equal_circle A₁ A₄ H₂ H₃ S) ∧
    (on_equal_circle A₂ A₃ H₁ H₄ S) ∧
    (on_equal_circle A₂ A₄ H₁ H₃ S) ∧
    (on_equal_circle A₃ A₄ H₁ H₂ S) ∧
    (on_equal_circle H₁ H₂ H₃ H₄ S) :=
  sorry

end NUMINAMATH_CALUDE_orthocenter_symmetry_and_equal_circles_l2188_218821


namespace NUMINAMATH_CALUDE_gcd_8157_2567_l2188_218801

theorem gcd_8157_2567 : Nat.gcd 8157 2567 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8157_2567_l2188_218801


namespace NUMINAMATH_CALUDE_meaningful_fraction_l2188_218819

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = (x - 3) / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l2188_218819


namespace NUMINAMATH_CALUDE_geometric_sum_example_l2188_218814

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Proof that the sum of the first 8 terms of a geometric sequence
    with first term 1/4 and common ratio 2 is equal to 255/4 -/
theorem geometric_sum_example :
  geometric_sum (1/4) 2 8 = 255/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_example_l2188_218814


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2188_218856

theorem solve_exponential_equation : ∃ x : ℝ, (1000 : ℝ)^5 = 40^x ∧ x = 15 := by sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2188_218856


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l2188_218883

theorem complex_product_pure_imaginary (a : ℝ) : 
  (Complex.I * 2 + a) * (Complex.I * 3 + 1) = Complex.I * b ∧ b ≠ 0 → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l2188_218883


namespace NUMINAMATH_CALUDE_min_value_theorem_l2188_218870

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a + 8) * x + a^2 + a - 12

theorem min_value_theorem (a : ℝ) (h1 : a < 0) 
  (h2 : f a (a^2 - 4) = f a (2*a - 8)) :
  ∀ n : ℕ+, (f a n - 4*a) / (n + 1) ≥ 37/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2188_218870


namespace NUMINAMATH_CALUDE_freds_allowance_l2188_218857

theorem freds_allowance (allowance : ℝ) : 
  (allowance / 2 + 6 = 14) → allowance = 16 := by
  sorry

end NUMINAMATH_CALUDE_freds_allowance_l2188_218857


namespace NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l2188_218823

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_16_to_30 :
  sum_of_squares 30 - sum_of_squares 15 = 8195 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l2188_218823


namespace NUMINAMATH_CALUDE_right_triangle_area_l2188_218807

theorem right_triangle_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_angle : a = b) (h_side : a = 5) : (1/2) * a * b = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2188_218807


namespace NUMINAMATH_CALUDE_train_speed_l2188_218849

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 100)
  (h2 : bridge_length = 300)
  (h3 : crossing_time = 12) :
  (train_length + bridge_length) / crossing_time = 400 / 12 := by
sorry

end NUMINAMATH_CALUDE_train_speed_l2188_218849


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2188_218896

/-- Given a hyperbola C and a circle F with specific properties, prove that the eccentricity of C is 2 -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let F := {(x, y) : ℝ × ℝ | (x - c)^2 + y^2 = c^2}
  let l := {(x, y) : ℝ × ℝ | y = -(a / b) * (x - 2 * a / 3)}
  ∃ (chord_length : ℝ), 
    (∀ (p q : ℝ × ℝ), p ∈ F ∧ q ∈ F ∧ p ∈ l ∧ q ∈ l → ‖p - q‖ = chord_length) ∧
    chord_length = 4 * Real.sqrt 2 * c / 3 →
  c / a = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2188_218896


namespace NUMINAMATH_CALUDE_successful_hatch_percentage_l2188_218873

/-- The number of eggs laid by each turtle -/
def eggs_per_turtle : ℕ := 20

/-- The number of turtles -/
def num_turtles : ℕ := 6

/-- The number of hatchlings produced -/
def num_hatchlings : ℕ := 48

/-- The percentage of eggs that successfully hatch -/
def hatch_percentage : ℚ := 40

theorem successful_hatch_percentage :
  (eggs_per_turtle * num_turtles : ℚ) * (hatch_percentage / 100) = num_hatchlings :=
sorry

end NUMINAMATH_CALUDE_successful_hatch_percentage_l2188_218873


namespace NUMINAMATH_CALUDE_average_age_when_youngest_born_l2188_218881

/-- Proves that given a group of 7 people with an average age of 50 years,
    where the youngest is 5 years old, the average age of the remaining 6 people
    5 years ago was 57.5 years. -/
theorem average_age_when_youngest_born
  (total_people : ℕ)
  (average_age : ℝ)
  (youngest_age : ℝ)
  (total_age : ℝ)
  (h1 : total_people = 7)
  (h2 : average_age = 50)
  (h3 : youngest_age = 5)
  (h4 : total_age = average_age * total_people)
  : (total_age - youngest_age) / (total_people - 1) = 57.5 := by
  sorry

#check average_age_when_youngest_born

end NUMINAMATH_CALUDE_average_age_when_youngest_born_l2188_218881


namespace NUMINAMATH_CALUDE_disease_mortality_percentage_l2188_218809

theorem disease_mortality_percentage (population : ℝ) (affected_percentage : ℝ) (mortality_rate : ℝ) 
  (h1 : affected_percentage = 15)
  (h2 : mortality_rate = 8) :
  (affected_percentage / 100) * (mortality_rate / 100) * 100 = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_disease_mortality_percentage_l2188_218809


namespace NUMINAMATH_CALUDE_g_composition_of_three_l2188_218869

def g (x : ℝ) : ℝ := 7 * x - 3

theorem g_composition_of_three : g (g (g 3)) = 858 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l2188_218869


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a1_l2188_218831

/-- An arithmetic sequence with specific conditions -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  (∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 3 = -6 ∧
  a 7 = a 5 + 4

/-- Theorem stating that under given conditions, a_1 = -10 -/
theorem arithmetic_sequence_a1 (a : ℕ → ℤ) :
  arithmetic_sequence a → a 1 = -10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a1_l2188_218831


namespace NUMINAMATH_CALUDE_prob_two_red_shoes_l2188_218845

/-- The probability of drawing two red shoes from a set of 4 red shoes and 4 green shoes -/
theorem prob_two_red_shoes : 
  let total_shoes : ℕ := 4 + 4
  let red_shoes : ℕ := 4
  let draw_count : ℕ := 2
  let total_ways := Nat.choose total_shoes draw_count
  let red_ways := Nat.choose red_shoes draw_count
  (red_ways : ℚ) / total_ways = 3 / 14 := by sorry

end NUMINAMATH_CALUDE_prob_two_red_shoes_l2188_218845


namespace NUMINAMATH_CALUDE_sum_of_bottom_circles_l2188_218871

-- Define the type for circle positions
inductive Position
| Top | UpperLeft | UpperMiddle | UpperRight | Middle | LowerLeft | LowerMiddle | LowerRight

-- Define the function type for number placement
def Placement := Position → Nat

-- Define the conditions of the problem
def validPlacement (p : Placement) : Prop :=
  (∀ i : Position, p i ∈ Finset.range 9 \ {0}) ∧ 
  (∀ i j : Position, i ≠ j → p i ≠ p j) ∧
  p Position.Top * p Position.UpperLeft * p Position.UpperMiddle = 30 ∧
  p Position.Top * p Position.UpperMiddle * p Position.UpperRight = 40 ∧
  p Position.UpperLeft * p Position.Middle * p Position.LowerLeft = 28 ∧
  p Position.UpperRight * p Position.Middle * p Position.LowerRight = 35 ∧
  p Position.LowerLeft * p Position.LowerMiddle * p Position.LowerRight = 20

-- State the theorem
theorem sum_of_bottom_circles (p : Placement) (h : validPlacement p) : 
  p Position.LowerLeft + p Position.LowerMiddle + p Position.LowerRight = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_bottom_circles_l2188_218871


namespace NUMINAMATH_CALUDE_angle_problem_l2188_218826

theorem angle_problem (x y : ℝ) : 
  x + y + 120 = 360 →
  x = 2 * y →
  x = 160 ∧ y = 80 :=
by sorry

end NUMINAMATH_CALUDE_angle_problem_l2188_218826


namespace NUMINAMATH_CALUDE_line_through_two_points_l2188_218804

/-- Given a line with equation x = 4y + 5 passing through points (m, n) and (m + 2, n + p),
    prove that p = 1/2 -/
theorem line_through_two_points (m n p : ℝ) : 
  (m = 4 * n + 5) ∧ (m + 2 = 4 * (n + p) + 5) → p = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_two_points_l2188_218804


namespace NUMINAMATH_CALUDE_tennis_players_count_l2188_218842

theorem tennis_players_count (total : ℕ) (baseball : ℕ) (both : ℕ) (no_sport : ℕ) :
  total = 310 →
  baseball = 255 →
  both = 94 →
  no_sport = 11 →
  ∃ tennis : ℕ, tennis = 138 ∧ total = tennis + baseball - both + no_sport :=
by sorry

end NUMINAMATH_CALUDE_tennis_players_count_l2188_218842


namespace NUMINAMATH_CALUDE_percentage_female_officers_on_duty_l2188_218882

/-- Given a police force with female officers, calculate the percentage on duty. -/
theorem percentage_female_officers_on_duty
  (total_on_duty : ℕ)
  (half_on_duty_female : ℕ)
  (total_female_officers : ℕ)
  (h1 : total_on_duty = 300)
  (h2 : half_on_duty_female = total_on_duty / 2)
  (h3 : total_female_officers = 1000) :
  (half_on_duty_female : ℚ) / total_female_officers * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_percentage_female_officers_on_duty_l2188_218882


namespace NUMINAMATH_CALUDE_restaurant_tip_percentage_l2188_218891

/-- Calculates the tip percentage given the total bill and tip amount -/
def tip_percentage (total_bill : ℚ) (tip_amount : ℚ) : ℚ :=
  (tip_amount / total_bill) * 100

/-- Proves that for a $40 bill and $4 tip, the tip percentage is 10% -/
theorem restaurant_tip_percentage : tip_percentage 40 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_tip_percentage_l2188_218891


namespace NUMINAMATH_CALUDE_salt_dilution_l2188_218833

theorem salt_dilution (initial_seawater : ℝ) (initial_salt_percentage : ℝ) 
  (final_salt_percentage : ℝ) (added_freshwater : ℝ) :
  initial_seawater = 40 →
  initial_salt_percentage = 0.05 →
  final_salt_percentage = 0.02 →
  added_freshwater = 60 →
  (initial_seawater * initial_salt_percentage) / (initial_seawater + added_freshwater) = final_salt_percentage :=
by
  sorry

#check salt_dilution

end NUMINAMATH_CALUDE_salt_dilution_l2188_218833


namespace NUMINAMATH_CALUDE_milk_carton_delivery_l2188_218827

theorem milk_carton_delivery (total_cartons : ℕ) (damaged_per_customer : ℕ) (total_accepted : ℕ) :
  total_cartons = 400 →
  damaged_per_customer = 60 →
  total_accepted = 160 →
  ∃ (num_customers : ℕ),
    num_customers > 0 ∧
    num_customers * (total_cartons / num_customers - damaged_per_customer) = total_accepted ∧
    num_customers = 4 :=
by sorry

end NUMINAMATH_CALUDE_milk_carton_delivery_l2188_218827


namespace NUMINAMATH_CALUDE_sum_of_xyz_l2188_218850

theorem sum_of_xyz (x y z : ℤ) 
  (hz : z = 4)
  (hxy : x + y = 7)
  (hxz : x + z = 8) : 
  x + y + z = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l2188_218850


namespace NUMINAMATH_CALUDE_sin_225_degrees_l2188_218820

theorem sin_225_degrees : Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_225_degrees_l2188_218820


namespace NUMINAMATH_CALUDE_son_father_height_relationship_l2188_218890

/-- Represents the possible types of relationships between variables -/
inductive RelationshipType
  | Deterministic
  | Correlation
  | Functional
  | None

/-- Represents the relationship between a son's height and his father's height -/
structure HeightRelationship where
  type : RelationshipType
  isUncertain : Bool

/-- Theorem: The relationship between a son's height and his father's height is a correlation relationship -/
theorem son_father_height_relationship :
  ∀ (r : HeightRelationship), r.isUncertain → r.type = RelationshipType.Correlation :=
by sorry

end NUMINAMATH_CALUDE_son_father_height_relationship_l2188_218890


namespace NUMINAMATH_CALUDE_travel_ratio_l2188_218893

theorem travel_ratio (george joseph patrick zack : ℕ) : 
  george = 6 →
  joseph = george / 2 →
  patrick = joseph * 3 →
  zack = 18 →
  zack / patrick = 2 :=
by sorry

end NUMINAMATH_CALUDE_travel_ratio_l2188_218893


namespace NUMINAMATH_CALUDE_compound_interest_proof_l2188_218858

/-- Given a principal amount for which the simple interest over 2 years at 10% rate is $600,
    prove that the compound interest over 2 years at 10% rate is $630 --/
theorem compound_interest_proof (P : ℝ) : 
  P * 0.1 * 2 = 600 → P * (1 + 0.1)^2 - P = 630 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_proof_l2188_218858


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2188_218805

theorem arithmetic_geometric_mean_inequality (a b : ℝ) : 
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2188_218805


namespace NUMINAMATH_CALUDE_original_list_size_l2188_218898

/-- The number of integers in the original list -/
def n : ℕ := sorry

/-- The mean of the original list -/
def m : ℚ := sorry

/-- The sum of the integers in the original list -/
def original_sum : ℚ := n * m

/-- The equation representing the first condition -/
axiom first_condition : (m + 2) * (n + 1) = original_sum + 15

/-- The equation representing the second condition -/
axiom second_condition : (m + 1) * (n + 2) = original_sum + 16

theorem original_list_size : n = 4 := by sorry

end NUMINAMATH_CALUDE_original_list_size_l2188_218898


namespace NUMINAMATH_CALUDE_nail_sizes_l2188_218866

theorem nail_sizes (fraction_2d : ℝ) (fraction_2d_or_4d : ℝ) (fraction_4d : ℝ) :
  fraction_2d = 0.25 →
  fraction_2d_or_4d = 0.75 →
  fraction_4d = fraction_2d_or_4d - fraction_2d →
  fraction_4d = 0.50 := by
sorry

end NUMINAMATH_CALUDE_nail_sizes_l2188_218866


namespace NUMINAMATH_CALUDE_jacoby_hourly_wage_l2188_218894

/-- Proves that Jacoby's hourly wage is $19 given the conditions of his savings and expenses --/
theorem jacoby_hourly_wage :
  let total_needed : ℕ := 5000
  let hours_worked : ℕ := 10
  let cookies_sold : ℕ := 24
  let cookie_price : ℕ := 4
  let lottery_win : ℕ := 500
  let sister_gift : ℕ := 500
  let remaining_needed : ℕ := 3214
  let hourly_wage : ℕ := (total_needed - remaining_needed - (cookies_sold * cookie_price) - lottery_win - 2 * sister_gift + 10) / hours_worked
  hourly_wage = 19
  := by sorry

end NUMINAMATH_CALUDE_jacoby_hourly_wage_l2188_218894


namespace NUMINAMATH_CALUDE_garys_gold_cost_per_gram_l2188_218840

/-- Proves that Gary's gold costs $15 per gram given the conditions of the problem -/
theorem garys_gold_cost_per_gram (gary_grams : ℝ) (anna_grams : ℝ) (anna_cost_per_gram : ℝ) (total_cost : ℝ)
  (h1 : gary_grams = 30)
  (h2 : anna_grams = 50)
  (h3 : anna_cost_per_gram = 20)
  (h4 : total_cost = 1450)
  (h5 : gary_grams * x + anna_grams * anna_cost_per_gram = total_cost) :
  x = 15 := by
  sorry

#check garys_gold_cost_per_gram

end NUMINAMATH_CALUDE_garys_gold_cost_per_gram_l2188_218840


namespace NUMINAMATH_CALUDE_dagger_example_l2188_218843

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (q / n)

-- Theorem statement
theorem dagger_example : dagger (3/7) (11/4) = 132/7 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l2188_218843


namespace NUMINAMATH_CALUDE_second_day_visitors_count_l2188_218837

/-- Represents the food bank scenario --/
structure FoodBank where
  initial_stock : ℕ
  first_day_visitors : ℕ
  first_day_cans_per_person : ℕ
  first_restock : ℕ
  second_day_cans_per_person : ℕ
  second_restock : ℕ
  second_day_cans_given : ℕ

/-- Calculates the number of people who showed up on the second day --/
def second_day_visitors (fb : FoodBank) : ℕ :=
  fb.second_day_cans_given / fb.second_day_cans_per_person

/-- Theorem stating that given the conditions, 1250 people showed up on the second day --/
theorem second_day_visitors_count (fb : FoodBank) 
  (h1 : fb.initial_stock = 2000)
  (h2 : fb.first_day_visitors = 500)
  (h3 : fb.first_day_cans_per_person = 1)
  (h4 : fb.first_restock = 1500)
  (h5 : fb.second_day_cans_per_person = 2)
  (h6 : fb.second_restock = 3000)
  (h7 : fb.second_day_cans_given = 2500) :
  second_day_visitors fb = 1250 := by
  sorry

#eval second_day_visitors {
  initial_stock := 2000,
  first_day_visitors := 500,
  first_day_cans_per_person := 1,
  first_restock := 1500,
  second_day_cans_per_person := 2,
  second_restock := 3000,
  second_day_cans_given := 2500
}

end NUMINAMATH_CALUDE_second_day_visitors_count_l2188_218837


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l2188_218800

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l2188_218800


namespace NUMINAMATH_CALUDE_best_play_win_probability_correct_l2188_218885

/-- The probability of the best play winning in a contest where 2m jurors are randomly selected from 2n moms. -/
def best_play_win_probability (n m : ℕ) : ℚ :=
  let C := fun (n k : ℕ) => Nat.choose n k
  1 / (C (2*n) n * C (2*n) (2*m)) *
  (Finset.sum (Finset.range (2*m + 1)) (fun q =>
    C n q * C n (2*m - q) *
    (Finset.sum (Finset.range (min q (m-1) + 1)) (fun t =>
      C q t * C (2*n - q) (n - t)))))

/-- Theorem stating the probability of the best play winning. -/
theorem best_play_win_probability_correct (n m : ℕ) (h : 2*m ≤ n) :
  best_play_win_probability n m = 
  (1 / (Nat.choose (2*n) n * Nat.choose (2*n) (2*m))) *
  (Finset.sum (Finset.range (2*m + 1)) (fun q =>
    Nat.choose n q * Nat.choose n (2*m - q) *
    (Finset.sum (Finset.range (min q (m-1) + 1)) (fun t =>
      Nat.choose q t * Nat.choose (2*n - q) (n - t))))) :=
by sorry

end NUMINAMATH_CALUDE_best_play_win_probability_correct_l2188_218885


namespace NUMINAMATH_CALUDE_joannas_family_money_ratio_l2188_218817

/-- Prove that given the conditions of Joanna's family's money, the ratio of her brother's money to Joanna's money is 3:1 -/
theorem joannas_family_money_ratio :
  ∀ (brother_multiple : ℚ),
  (8 : ℚ) + 8 * brother_multiple + 4 = 36 →
  brother_multiple = 3 :=
by sorry

end NUMINAMATH_CALUDE_joannas_family_money_ratio_l2188_218817


namespace NUMINAMATH_CALUDE_profit_increase_approx_l2188_218892

/-- Represents the monthly profit changes as factors -/
def march_to_april : ℝ := 1.35
def april_to_may : ℝ := 0.80
def may_to_june : ℝ := 1.50
def june_to_july : ℝ := 0.75
def july_to_august : ℝ := 1.45

/-- The overall factor of profit change from March to August -/
def overall_factor : ℝ :=
  march_to_april * april_to_may * may_to_june * june_to_july * july_to_august

/-- The overall percentage increase from March to August -/
def overall_percentage_increase : ℝ := (overall_factor - 1) * 100

/-- Theorem stating the overall percentage increase is approximately 21.95% -/
theorem profit_increase_approx :
  ∃ ε > 0, abs (overall_percentage_increase - 21.95) < ε :=
sorry

end NUMINAMATH_CALUDE_profit_increase_approx_l2188_218892


namespace NUMINAMATH_CALUDE_rotation_result_l2188_218803

-- Define the shapes
inductive Shape
  | Rectangle
  | SmallCircle
  | Pentagon

-- Define the positions
inductive Position
  | Top
  | LeftBottom
  | RightBottom

-- Define the circular plane
structure CircularPlane where
  shapes : List Shape
  positions : List Position
  arrangement : Shape → Position

-- Define the rotation
def rotate150 (plane : CircularPlane) : CircularPlane := sorry

-- Theorem statement
theorem rotation_result (plane : CircularPlane) :
  plane.arrangement Shape.Rectangle = Position.Top →
  plane.arrangement Shape.SmallCircle = Position.LeftBottom →
  plane.arrangement Shape.Pentagon = Position.RightBottom →
  (rotate150 plane).arrangement Shape.SmallCircle = Position.Top := by
  sorry

end NUMINAMATH_CALUDE_rotation_result_l2188_218803


namespace NUMINAMATH_CALUDE_expand_and_factor_l2188_218867

theorem expand_and_factor (a b c : ℝ) : (a + b - c) * (a - b + c) = (a + (b - c)) * (a - (b - c)) := by
  sorry

end NUMINAMATH_CALUDE_expand_and_factor_l2188_218867


namespace NUMINAMATH_CALUDE_quadratic_vertex_l2188_218852

/-- Represents a quadratic function of the form y = ax^2 + bx - 3 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- Checks if a point (x, y) lies on the quadratic function -/
def QuadraticFunction.contains (f : QuadraticFunction) (x y : ℝ) : Prop :=
  y = f.a * x^2 + f.b * x - 3

theorem quadratic_vertex (f : QuadraticFunction) :
  f.contains (-2) 5 →
  f.contains (-1) 0 →
  f.contains 0 (-3) →
  f.contains 1 (-4) →
  f.contains 2 (-3) →
  ∃ (a b : ℝ), f = ⟨a, b⟩ ∧ (1, -4) = (- b / (2 * a), - (b^2 - 4*a*3) / (4 * a)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l2188_218852


namespace NUMINAMATH_CALUDE_sally_picked_peaches_l2188_218828

/-- Calculates the number of peaches Sally picked at the orchard. -/
def peaches_picked (initial_peaches final_peaches : ℕ) : ℕ :=
  final_peaches - initial_peaches

/-- Theorem stating that Sally picked 55 peaches at the orchard. -/
theorem sally_picked_peaches : peaches_picked 13 68 = 55 := by
  sorry

end NUMINAMATH_CALUDE_sally_picked_peaches_l2188_218828


namespace NUMINAMATH_CALUDE_final_expression_l2188_218855

theorem final_expression (b : ℚ) : ((3 * b + 6) - 5 * b) / 3 = -2/3 * b + 2 := by
  sorry

end NUMINAMATH_CALUDE_final_expression_l2188_218855


namespace NUMINAMATH_CALUDE_divisible_by_seven_l2188_218899

theorem divisible_by_seven (k : ℕ) : 
  7 ∣ (2^(6*k+1) + 3^(6*k+1) + 5^(6*k+1)) := by
sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l2188_218899


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2188_218834

theorem quadratic_equation_roots : ∃ x y : ℝ, x ≠ y ∧ 
  (x^2 + 2*x - 3 = 0) ∧ (y^2 + 2*y - 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2188_218834


namespace NUMINAMATH_CALUDE_points_always_odd_l2188_218802

/-- The number of points after k operations of adding a point between every two neighboring points. -/
def num_points (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then n
  else 2 * (num_points n (k - 1)) - 1

/-- Theorem: The number of points is always odd after each operation. -/
theorem points_always_odd (n : ℕ) (k : ℕ) (h : n ≥ 2) :
  Odd (num_points n k) :=
sorry

end NUMINAMATH_CALUDE_points_always_odd_l2188_218802


namespace NUMINAMATH_CALUDE_modular_inverse_12_mod_997_l2188_218854

theorem modular_inverse_12_mod_997 : ∃ x : ℤ, 12 * x ≡ 1 [ZMOD 997] :=
by
  use 914
  sorry

end NUMINAMATH_CALUDE_modular_inverse_12_mod_997_l2188_218854


namespace NUMINAMATH_CALUDE_remainder_2_power_2015_mod_20_l2188_218836

theorem remainder_2_power_2015_mod_20 : ∃ (seq : Fin 4 → Nat),
  (∀ (n : Nat), (2^n : Nat) % 20 = seq (n % 4)) ∧
  (seq 0 = 4 ∧ seq 1 = 8 ∧ seq 2 = 16 ∧ seq 3 = 12) →
  (2^2015 : Nat) % 20 = 8 := by
sorry

end NUMINAMATH_CALUDE_remainder_2_power_2015_mod_20_l2188_218836


namespace NUMINAMATH_CALUDE_point_not_on_transformed_plane_l2188_218851

/-- The original plane equation -/
def plane_a (x y z : ℝ) : Prop := 2*x + 3*y + z - 1 = 0

/-- The similarity transformation with scale factor k -/
def similarity_transform (k : ℝ) (x y z : ℝ) : Prop := 2*x + 3*y + z - k = 0

/-- The point A -/
def point_A : ℝ × ℝ × ℝ := (1, 2, -1)

/-- The scale factor -/
def k : ℝ := 2

/-- Theorem stating that point A does not lie on the transformed plane -/
theorem point_not_on_transformed_plane : 
  ¬ similarity_transform k point_A.1 point_A.2.1 point_A.2.2 :=
sorry

end NUMINAMATH_CALUDE_point_not_on_transformed_plane_l2188_218851


namespace NUMINAMATH_CALUDE_sum_of_solution_l2188_218889

theorem sum_of_solution (a b : ℝ) : 
  3 * a + 7 * b = 1977 → 
  5 * a + b = 2007 → 
  a + b = 498 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solution_l2188_218889


namespace NUMINAMATH_CALUDE_sphere_carved_cube_surface_area_l2188_218877

theorem sphere_carved_cube_surface_area :
  let sphere_diameter : ℝ := Real.sqrt 3
  let cube_side_length : ℝ := 1
  let cube_diagonal : ℝ := cube_side_length * Real.sqrt 3
  cube_diagonal = sphere_diameter →
  (6 : ℝ) * cube_side_length ^ 2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_sphere_carved_cube_surface_area_l2188_218877


namespace NUMINAMATH_CALUDE_square_fraction_count_l2188_218812

theorem square_fraction_count : 
  ∃! n : ℤ, (∃ k : ℤ, 30 - 2*n ≠ 0 ∧ n/(30 - 2*n) = k^2 ∧ n/(30 - 2*n) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_square_fraction_count_l2188_218812


namespace NUMINAMATH_CALUDE_unique_solution_lcm_gcd_l2188_218878

theorem unique_solution_lcm_gcd : 
  ∃! n : ℕ+, n.lcm 120 = n.gcd 120 + 300 ∧ n = 180 := by sorry

end NUMINAMATH_CALUDE_unique_solution_lcm_gcd_l2188_218878


namespace NUMINAMATH_CALUDE_capri_sun_cost_per_pouch_l2188_218880

theorem capri_sun_cost_per_pouch :
  let boxes : ℕ := 10
  let pouches_per_box : ℕ := 6
  let total_cost_dollars : ℕ := 12
  let total_pouches : ℕ := boxes * pouches_per_box
  let total_cost_cents : ℕ := total_cost_dollars * 100
  total_cost_cents / total_pouches = 20 := by sorry

end NUMINAMATH_CALUDE_capri_sun_cost_per_pouch_l2188_218880


namespace NUMINAMATH_CALUDE_john_total_spent_l2188_218875

/-- The total amount John spends on t-shirts and pants -/
def total_spent (num_tshirts : ℕ) (price_per_tshirt : ℕ) (pants_cost : ℕ) : ℕ :=
  num_tshirts * price_per_tshirt + pants_cost

/-- Theorem: John spends $110 in total -/
theorem john_total_spent :
  total_spent 3 20 50 = 110 := by
  sorry

end NUMINAMATH_CALUDE_john_total_spent_l2188_218875


namespace NUMINAMATH_CALUDE_factorization_equality_l2188_218813

theorem factorization_equality (a : ℝ) : 2 * a^2 - 4 * a + 2 = 2 * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2188_218813


namespace NUMINAMATH_CALUDE_misha_initial_dollars_l2188_218832

/-- The amount of dollars Misha needs to earn -/
def dollars_to_earn : ℕ := 13

/-- The total amount of dollars Misha will have after earning -/
def total_dollars : ℕ := 47

/-- Misha's initial amount of dollars -/
def initial_dollars : ℕ := total_dollars - dollars_to_earn

theorem misha_initial_dollars : initial_dollars = 34 := by
  sorry

end NUMINAMATH_CALUDE_misha_initial_dollars_l2188_218832


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l2188_218844

theorem fraction_zero_implies_x_equals_one (x : ℝ) : 
  (x^2 - 1) / (x + 1) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l2188_218844


namespace NUMINAMATH_CALUDE_solution_difference_l2188_218884

theorem solution_difference (p q : ℝ) : 
  (p - 3) * (p + 3) = 21 * p - 63 →
  (q - 3) * (q + 3) = 21 * q - 63 →
  p ≠ q →
  p > q →
  p - q = 15 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l2188_218884


namespace NUMINAMATH_CALUDE_evaluate_g_l2188_218895

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 8

theorem evaluate_g : 3 * g 3 + 2 * g (-3) = 160 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l2188_218895


namespace NUMINAMATH_CALUDE_cube_sum_geq_product_of_squares_and_sum_l2188_218830

theorem cube_sum_geq_product_of_squares_and_sum {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 + b^3 + c^3 ≥ (1/3) * (a^2 + b^2 + c^2) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_geq_product_of_squares_and_sum_l2188_218830


namespace NUMINAMATH_CALUDE_perpendicular_lines_l2188_218811

theorem perpendicular_lines (a : ℝ) : 
  (∃ (x y : ℝ), y = a * x - 2) ∧ 
  (∃ (x y : ℝ), y = (a + 2) * x + 1) ∧ 
  (a * (a + 2) = -1) → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l2188_218811


namespace NUMINAMATH_CALUDE_paper_I_maximum_marks_l2188_218816

theorem paper_I_maximum_marks :
  ∀ (max_marks passing_mark secured_marks deficit : ℕ),
    passing_mark = (max_marks * 40) / 100 →
    secured_marks = 40 →
    deficit = 20 →
    passing_mark = secured_marks + deficit →
    max_marks = 150 := by
  sorry

end NUMINAMATH_CALUDE_paper_I_maximum_marks_l2188_218816


namespace NUMINAMATH_CALUDE_area_of_rotated_rectangle_curve_l2188_218815

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle defined by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Rotation of a point around a center point -/
def rotate90Clockwise (p : Point) (center : Point) : Point :=
  { x := center.x + (p.y - center.y),
    y := center.y - (p.x - center.x) }

/-- The area enclosed by the curve traced by a point on a rectangle under rotations -/
def areaEnclosedByCurve (rect : Rectangle) (initialPoint : Point) (rotationCenters : List Point) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem area_of_rotated_rectangle_curve (rect : Rectangle) (initialPoint : Point) 
  (rotationCenters : List Point) : 
  rect.width = 2 ∧ rect.height = 3 ∧
  initialPoint = { x := 1, y := 1 } ∧
  rotationCenters = [{ x := 2, y := 0 }, { x := 5, y := 0 }, { x := 7, y := 0 }, { x := 10, y := 0 }] →
  areaEnclosedByCurve rect initialPoint rotationCenters = 6 + 7 * Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_rotated_rectangle_curve_l2188_218815


namespace NUMINAMATH_CALUDE_product_expansion_sum_l2188_218860

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x : ℝ, (4 * x^2 - 6 * x + 5) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  9 * a + 3 * b + 2 * c + d = -44 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l2188_218860


namespace NUMINAMATH_CALUDE_max_value_f2019_l2188_218835

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the recursive function f_n
def f_n : ℕ → (ℝ → ℝ)
  | 0 => f
  | n + 1 => λ x => f (f_n n x)

-- State the theorem
theorem max_value_f2019 :
  ∀ x ∈ Set.Icc 1 2,
  f_n 2019 x ≤ 3^(2^2019) - 1 ∧
  ∃ y ∈ Set.Icc 1 2, f_n 2019 y = 3^(2^2019) - 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_f2019_l2188_218835


namespace NUMINAMATH_CALUDE_rectangular_box_diagonals_l2188_218822

theorem rectangular_box_diagonals 
  (x y z : ℝ) 
  (surface_area : 2 * (x*y + y*z + z*x) = 106) 
  (edge_sum : 4 * (x + y + z) = 52) :
  4 * Real.sqrt (x^2 + y^2 + z^2) = 12 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_diagonals_l2188_218822


namespace NUMINAMATH_CALUDE_fg_difference_of_squares_l2188_218879

def f (x : ℝ) : ℝ := x - 2

def g (x : ℝ) : ℝ := 2 * x + 4

theorem fg_difference_of_squares : (f (g 3))^2 - (g (f 3))^2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_fg_difference_of_squares_l2188_218879


namespace NUMINAMATH_CALUDE_aleesia_weight_loss_l2188_218887

/-- Aleesia's weekly weight loss problem -/
theorem aleesia_weight_loss 
  (aleesia_weeks : ℕ) 
  (alexei_weeks : ℕ) 
  (alexei_weekly_loss : ℝ) 
  (total_loss : ℝ) 
  (h1 : aleesia_weeks = 10) 
  (h2 : alexei_weeks = 8) 
  (h3 : alexei_weekly_loss = 2.5) 
  (h4 : total_loss = 35) :
  ∃ (aleesia_weekly_loss : ℝ), 
    aleesia_weekly_loss * aleesia_weeks + alexei_weekly_loss * alexei_weeks = total_loss ∧ 
    aleesia_weekly_loss = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_aleesia_weight_loss_l2188_218887


namespace NUMINAMATH_CALUDE_f_properties_l2188_218853

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -a * x + x + a

-- Define the open interval (0,1]
def openUnitInterval : Set ℝ := { x | 0 < x ∧ x ≤ 1 }

-- Theorem statement
theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x ∈ openUnitInterval, ∀ y ∈ openUnitInterval, x < y → f a x < f a y) ↔ (0 < a ∧ a ≤ 1) ∧
  (∃ M : ℝ, M = 1 ∧ ∀ x ∈ openUnitInterval, f a x ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2188_218853


namespace NUMINAMATH_CALUDE_unique_solution_linear_equation_l2188_218846

theorem unique_solution_linear_equation (a b : ℝ) :
  (a * 1 + b * 2 = 2) ∧ (a * 2 + b * 5 = 2) → a = 6 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_linear_equation_l2188_218846


namespace NUMINAMATH_CALUDE_unique_real_root_l2188_218861

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - 10

-- Theorem statement
theorem unique_real_root : ∃! x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_real_root_l2188_218861


namespace NUMINAMATH_CALUDE_exists_n_congruence_l2188_218824

/-- ν(n) denotes the exponent of 2 in the prime factorization of n! -/
def ν (n : ℕ) : ℕ := sorry

/-- For any positive integers a and m, there exists an integer n > 1 such that ν(n) ≡ a (mod m) -/
theorem exists_n_congruence (a m : ℕ+) : ∃ n : ℕ, n > 1 ∧ ν n % m = a % m := by
  sorry

end NUMINAMATH_CALUDE_exists_n_congruence_l2188_218824


namespace NUMINAMATH_CALUDE_dice_sum_symmetry_l2188_218897

def num_dice : ℕ := 8
def min_face : ℕ := 1
def max_face : ℕ := 6

def sum_symmetric (s : ℕ) : ℕ :=
  2 * ((num_dice * min_face + num_dice * max_face) / 2) - s

theorem dice_sum_symmetry :
  sum_symmetric 12 = 44 :=
by sorry

end NUMINAMATH_CALUDE_dice_sum_symmetry_l2188_218897


namespace NUMINAMATH_CALUDE_pond_to_field_ratio_l2188_218838

/-- Given a rectangular field with length double its width and length 28 m, 
    containing a square pond with side length 7 m, 
    the ratio of the pond's area to the field's area is 1:8. -/
theorem pond_to_field_ratio : 
  let field_length : ℝ := 28
  let field_width : ℝ := field_length / 2
  let pond_side : ℝ := 7
  let field_area : ℝ := field_length * field_width
  let pond_area : ℝ := pond_side ^ 2
  pond_area / field_area = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_pond_to_field_ratio_l2188_218838


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt14_l2188_218810

theorem sqrt_sum_equals_2sqrt14 : 
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt14_l2188_218810


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l2188_218859

/-- Number of games required to determine a champion in a single-elimination tournament -/
def games_required (num_players : ℕ) : ℕ := num_players - 1

/-- Theorem: In a single-elimination tournament with 512 players, 511 games are required to determine the champion -/
theorem single_elimination_tournament_games (num_players : ℕ) (h : num_players = 512) :
  games_required num_players = 511 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l2188_218859


namespace NUMINAMATH_CALUDE_square_recurrence_cube_recurrence_l2188_218847

-- Define the sequences
def a (n : ℕ) : ℕ := n^2
def b (n : ℕ) : ℕ := n^3

-- Theorem for the linear recurrence relation of a_n = n^2
theorem square_recurrence (n : ℕ) (h : n ≥ 3) :
  a n = 3 * a (n - 1) - 3 * a (n - 2) + a (n - 3) := by
  sorry

-- Theorem for the linear recurrence relation of a_n = n^3
theorem cube_recurrence (n : ℕ) (h : n ≥ 4) :
  b n = 4 * b (n - 1) - 6 * b (n - 2) + 4 * b (n - 3) - b (n - 4) := by
  sorry

end NUMINAMATH_CALUDE_square_recurrence_cube_recurrence_l2188_218847


namespace NUMINAMATH_CALUDE_base4_division_theorem_l2188_218864

/-- Converts a number from base 4 to base 10 -/
def base4ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a number from base 10 to base 4 -/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem base4_division_theorem :
  let a := [2, 3, 3, 1]  -- 1332 in base 4 (least significant digit first)
  let b := [3, 1]        -- 13 in base 4 (least significant digit first)
  let result := [2, 0, 1] -- 102 in base 4 (least significant digit first)
  (base4ToBase10 a) / (base4ToBase10 b) = base4ToBase10 result :=
by sorry

end NUMINAMATH_CALUDE_base4_division_theorem_l2188_218864


namespace NUMINAMATH_CALUDE_gcd_3570_4840_l2188_218888

theorem gcd_3570_4840 : Nat.gcd 3570 4840 = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcd_3570_4840_l2188_218888


namespace NUMINAMATH_CALUDE_regular_polygon_45_symmetry_l2188_218865

/-- A regular polygon that coincides with its original shape for the first time
    after rotating 45° around its center. -/
structure RegularPolygon45 where
  /-- The number of sides in the polygon -/
  sides : ℕ
  /-- The polygon is regular -/
  is_regular : True
  /-- The polygon coincides with its original shape after rotating 45° -/
  rotation_45 : sides * 45 = 360

/-- Axial symmetry property -/
def is_axially_symmetric (p : RegularPolygon45) : Prop := True

/-- Central symmetry property -/
def is_centrally_symmetric (p : RegularPolygon45) : Prop := True

/-- Theorem stating that a RegularPolygon45 is both axially and centrally symmetric -/
theorem regular_polygon_45_symmetry (p : RegularPolygon45) :
  is_axially_symmetric p ∧ is_centrally_symmetric p :=
sorry

end NUMINAMATH_CALUDE_regular_polygon_45_symmetry_l2188_218865


namespace NUMINAMATH_CALUDE_solve_distance_problem_l2188_218841

def distance_problem (initial_speed : ℝ) (initial_time : ℝ) (speed_increase : ℝ) (additional_time : ℝ) : Prop :=
  let initial_distance := initial_speed * initial_time
  let new_speed := initial_speed * (1 + speed_increase)
  let additional_distance := new_speed * additional_time
  let total_distance := initial_distance + additional_distance
  total_distance = 13

theorem solve_distance_problem :
  distance_problem 2 2 0.5 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_distance_problem_l2188_218841


namespace NUMINAMATH_CALUDE_expression_evaluation_l2188_218868

theorem expression_evaluation : 
  let x : ℝ := 2
  (2 * x + 3) * (2 * x - 3) + (x - 2)^2 - 3 * x * (x - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2188_218868


namespace NUMINAMATH_CALUDE_fruit_arrangement_count_l2188_218806

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem fruit_arrangement_count : 
  let total_fruits : ℕ := 8
  let apples : ℕ := 3
  let oranges : ℕ := 2
  let bananas : ℕ := 3
  factorial total_fruits / (factorial apples * factorial oranges * factorial bananas) = 560 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangement_count_l2188_218806


namespace NUMINAMATH_CALUDE_prob_four_odd_in_five_rolls_l2188_218829

/-- The probability of rolling an odd number on a fair 6-sided die -/
def prob_odd : ℚ := 1 / 2

/-- The number of rolls -/
def num_rolls : ℕ := 5

/-- The number of rolls we want to be odd -/
def target_odd : ℕ := 4

/-- The probability of getting exactly 4 odd numbers in 5 rolls of a fair 6-sided die -/
theorem prob_four_odd_in_five_rolls :
  (Nat.choose num_rolls target_odd : ℚ) * prob_odd ^ target_odd * (1 - prob_odd) ^ (num_rolls - target_odd) = 5 / 32 := by
  sorry

end NUMINAMATH_CALUDE_prob_four_odd_in_five_rolls_l2188_218829


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2188_218863

theorem complex_modulus_problem (z : ℂ) (i : ℂ) (h1 : i^2 = -1) (h2 : (1 - i) * z = 1) : 
  Complex.abs (4 * z - 3) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2188_218863


namespace NUMINAMATH_CALUDE_fenced_area_with_cutouts_l2188_218872

theorem fenced_area_with_cutouts (yard_length yard_width cutout1_side cutout2_side : ℝ) 
  (h1 : yard_length = 20)
  (h2 : yard_width = 15)
  (h3 : cutout1_side = 4)
  (h4 : cutout2_side = 2) :
  yard_length * yard_width - (cutout1_side * cutout1_side + cutout2_side * cutout2_side) = 280 := by
  sorry

end NUMINAMATH_CALUDE_fenced_area_with_cutouts_l2188_218872


namespace NUMINAMATH_CALUDE_chocolate_distribution_l2188_218886

theorem chocolate_distribution (x y : ℕ) (h1 : y = x + 1) (h2 : ∃ z : ℕ, y = (x - 1) * z + z) : 
  ∃ z : ℕ, y = (x - 1) * z + z ∧ z = 2 := by
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l2188_218886


namespace NUMINAMATH_CALUDE_inequality_solution_l2188_218818

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (2 - x)) ∧
  (∀ x₁ x₂, 1 ≤ x₁ → x₁ < x₂ → (x₁ - x₂) / (f x₁ - f x₂) > 0)

/-- The solution set of the inequality -/
def solution_set (x : ℝ) : Prop :=
  x ≤ 0 ∨ x ≥ 4/3

/-- Theorem stating the solution of the inequality -/
theorem inequality_solution (f : ℝ → ℝ) (h : special_function f) :
  ∀ x, f (2*x - 1) - f (3 - x) ≥ 0 ↔ solution_set x :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2188_218818


namespace NUMINAMATH_CALUDE_count_parallel_edges_l2188_218862

structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  distinct : length ≠ width ∧ width ≠ height ∧ length ≠ height

def parallel_edge_pairs (prism : RectangularPrism) : ℕ := 6

theorem count_parallel_edges (prism : RectangularPrism) :
  parallel_edge_pairs prism = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_parallel_edges_l2188_218862


namespace NUMINAMATH_CALUDE_intercept_plane_equation_point_on_intercept_plane_l2188_218808

/-- A plane in 3D space with intercepts a, b, c on x, y, z axes respectively --/
structure InterceptPlane where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

/-- The equation of a plane given its intercepts --/
def plane_equation (p : InterceptPlane) (x y z : ℝ) : Prop :=
  x / p.a + y / p.b + z / p.c = 1

/-- Theorem stating that the given equation represents the plane with given intercepts --/
theorem intercept_plane_equation (p : InterceptPlane) :
  ∀ x y z : ℝ, (x = p.a ∧ y = 0 ∧ z = 0) ∨ (x = 0 ∧ y = p.b ∧ z = 0) ∨ (x = 0 ∧ y = 0 ∧ z = p.c) →
  plane_equation p x y z := by
  sorry

/-- Theorem stating that any point satisfying the equation lies on the plane --/
theorem point_on_intercept_plane (p : InterceptPlane) :
  ∀ x y z : ℝ, plane_equation p x y z →
  ∃ t u v : ℝ, t + u + v = 1 ∧ x = t * p.a ∧ y = u * p.b ∧ z = v * p.c := by
  sorry

end NUMINAMATH_CALUDE_intercept_plane_equation_point_on_intercept_plane_l2188_218808


namespace NUMINAMATH_CALUDE_parabola_coef_sum_zero_l2188_218874

/-- A parabola with equation y = ax^2 + bx + c, vertex (3, 4), and passing through (1, 0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 3
  vertex_y : ℝ := 4
  point_x : ℝ := 1
  point_y : ℝ := 0
  eq_at_vertex : vertex_y = a * vertex_x^2 + b * vertex_x + c
  eq_at_point : point_y = a * point_x^2 + b * point_x + c

/-- The sum of coefficients a, b, and c for the specified parabola is 0 -/
theorem parabola_coef_sum_zero (p : Parabola) : p.a + p.b + p.c = 0 := by
  sorry


end NUMINAMATH_CALUDE_parabola_coef_sum_zero_l2188_218874


namespace NUMINAMATH_CALUDE_quadratic_point_relationship_l2188_218825

/-- A quadratic function f(x) = x^2 - 2x + m passing through three specific points -/
def QuadraticThroughPoints (m : ℝ) (y₁ y₂ y₃ : ℝ) : Prop :=
  let f := fun x => x^2 - 2*x + m
  f (-1) = y₁ ∧ f 2 = y₂ ∧ f 3 = y₃

/-- Theorem stating the relationship between y₁, y₂, and y₃ for the given quadratic function -/
theorem quadratic_point_relationship (m : ℝ) (y₁ y₂ y₃ : ℝ) 
    (h : QuadraticThroughPoints m y₁ y₂ y₃) : 
    y₂ < y₁ ∧ y₁ = y₃ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_relationship_l2188_218825


namespace NUMINAMATH_CALUDE_sine_of_supplementary_angles_l2188_218839

theorem sine_of_supplementary_angles (α β : Real) :
  α + β = Real.pi → Real.sin α = Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_sine_of_supplementary_angles_l2188_218839
