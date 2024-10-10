import Mathlib

namespace unique_quadratic_function_max_min_values_l4046_404601

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

-- State the theorem
theorem unique_quadratic_function 
  (a b : ℝ) 
  (h1 : a ≠ 0)
  (h2 : f a b 2 = 0)
  (h3 : ∃! x, f a b x = x) :
  ∀ x, f a b x = -1/2 * x^2 + x := by
sorry

-- For the second part of the question
theorem max_min_values
  (h : ∀ x, f 1 (-2) x = x^2 - 2*x) :
  (∀ x ∈ [-1, 2], f 1 (-2) x ≤ 3) ∧
  (∀ x ∈ [-1, 2], f 1 (-2) x ≥ -1) ∧
  (∃ x ∈ [-1, 2], f 1 (-2) x = 3) ∧
  (∃ x ∈ [-1, 2], f 1 (-2) x = -1) := by
sorry

end unique_quadratic_function_max_min_values_l4046_404601


namespace two_digit_number_between_30_and_40_with_units_digit_2_l4046_404619

theorem two_digit_number_between_30_and_40_with_units_digit_2 (n : ℕ) :
  (n ≥ 30 ∧ n < 40) →  -- two-digit number between 30 and 40
  (n % 10 = 2) →       -- units digit is 2
  n = 32 :=
by sorry

end two_digit_number_between_30_and_40_with_units_digit_2_l4046_404619


namespace nathan_tokens_used_l4046_404656

/-- The number of times Nathan played air hockey -/
def air_hockey_games : ℕ := 2

/-- The number of times Nathan played basketball -/
def basketball_games : ℕ := 4

/-- The cost in tokens for each game -/
def tokens_per_game : ℕ := 3

/-- The total number of tokens Nathan used -/
def total_tokens : ℕ := air_hockey_games * tokens_per_game + basketball_games * tokens_per_game

theorem nathan_tokens_used : total_tokens = 18 := by
  sorry

end nathan_tokens_used_l4046_404656


namespace negation_equivalence_l4046_404621

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 < 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) :=
by sorry

end negation_equivalence_l4046_404621


namespace expression_equality_l4046_404650

theorem expression_equality : (50 + 20 / 90) * 90 = 4520 := by
  sorry

end expression_equality_l4046_404650


namespace sum_of_solutions_eq_eight_l4046_404624

theorem sum_of_solutions_eq_eight : 
  ∃ (N₁ N₂ : ℝ), N₁ * (N₁ - 8) = -7 ∧ N₂ * (N₂ - 8) = -7 ∧ N₁ + N₂ = 8 := by
  sorry

end sum_of_solutions_eq_eight_l4046_404624


namespace problem_solution_l4046_404617

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 10}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - a^2 ≥ 0}

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

theorem problem_solution (a : ℝ) (h : a > 0) :
  ((A ∩ B a = ∅) → a ≥ 9) ∧
  ((∀ x, (¬p x → q a x) ∧ (∃ y, q a y ∧ p y)) → (a ≤ 3)) := by
  sorry

end problem_solution_l4046_404617


namespace absolute_value_inequality_l4046_404672

theorem absolute_value_inequality (x : ℝ) : 
  |((x + 1) / x)| > ((x + 1) / x) ↔ -1 < x ∧ x < 0 :=
by sorry

end absolute_value_inequality_l4046_404672


namespace intersection_area_of_specific_circles_l4046_404611

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The area of intersection of two circles -/
def intersectionArea (c1 c2 : Circle) : ℝ := sorry

/-- The first circle centered at (3,0) with radius 3 -/
def circle1 : Circle := { center := (3, 0), radius := 3 }

/-- The second circle centered at (0,3) with radius 3 -/
def circle2 : Circle := { center := (0, 3), radius := 3 }

/-- Theorem stating the area of intersection of the two given circles -/
theorem intersection_area_of_specific_circles :
  intersectionArea circle1 circle2 = (9 * Real.pi - 18) / 2 := by sorry

end intersection_area_of_specific_circles_l4046_404611


namespace price_reduction_for_1750_profit_max_profit_1800_at_20_l4046_404609

-- Define the initial conditions
def initial_sales : ℕ := 40
def initial_profit_per_shirt : ℕ := 40
def sales_increase_rate : ℚ := 2  -- 1 shirt per 0.5 yuan decrease

-- Define the profit function
def profit_function (price_reduction : ℚ) : ℚ :=
  (initial_profit_per_shirt - price_reduction) * (initial_sales + sales_increase_rate * price_reduction)

-- Theorem 1: The price reduction for 1750 yuan profit is 15 yuan
theorem price_reduction_for_1750_profit :
  ∃ (x : ℚ), profit_function x = 1750 ∧ x = 15 := by sorry

-- Theorem 2: The maximum profit is 1800 yuan at 20 yuan price reduction
theorem max_profit_1800_at_20 :
  ∃ (max_profit : ℚ) (optimal_reduction : ℚ),
    max_profit = 1800 ∧
    optimal_reduction = 20 ∧
    (∀ x, profit_function x ≤ max_profit) ∧
    profit_function optimal_reduction = max_profit := by sorry

end price_reduction_for_1750_profit_max_profit_1800_at_20_l4046_404609


namespace lunchroom_students_l4046_404633

theorem lunchroom_students (tables : ℕ) (seated_per_table : ℕ) (standing : ℕ) : 
  tables = 34 → seated_per_table = 6 → standing = 15 →
  tables * seated_per_table + standing = 219 := by
  sorry

end lunchroom_students_l4046_404633


namespace imaginary_part_of_i_minus_one_l4046_404630

theorem imaginary_part_of_i_minus_one (i : ℂ) (h : i * i = -1) :
  Complex.im (i - 1) = 1 := by
  sorry

end imaginary_part_of_i_minus_one_l4046_404630


namespace circle_radius_increase_l4046_404653

theorem circle_radius_increase (r₁ r₂ : ℝ) : 
  2 * Real.pi * r₁ = 30 → 2 * Real.pi * r₂ = 40 → r₂ - r₁ = 5 / Real.pi := by
  sorry

end circle_radius_increase_l4046_404653


namespace train_crossing_time_l4046_404660

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_pass_man : ℝ) 
  (h1 : train_length = 186)
  (h2 : platform_length = 279)
  (h3 : time_pass_man = 8) :
  (train_length + platform_length) / (train_length / time_pass_man) = 20 := by
  sorry

end train_crossing_time_l4046_404660


namespace julie_monthly_salary_l4046_404628

/-- Calculates the monthly salary for a worker given their hourly rate, hours per day,
    days per week, and number of missed days in a month. -/
def monthly_salary (hourly_rate : ℚ) (hours_per_day : ℕ) (days_per_week : ℕ) (missed_days : ℕ) : ℚ :=
  let daily_earnings := hourly_rate * hours_per_day
  let weekly_earnings := daily_earnings * days_per_week
  let monthly_earnings := weekly_earnings * 4
  monthly_earnings - (daily_earnings * missed_days)

/-- Proves that Julie's monthly salary after missing a day of work is $920. -/
theorem julie_monthly_salary :
  monthly_salary 5 8 6 1 = 920 := by
  sorry

end julie_monthly_salary_l4046_404628


namespace variation_relationship_l4046_404696

theorem variation_relationship (x y z : ℝ) (k j : ℝ) (h1 : x = k * y^2) (h2 : y = j * z^(1/3)) :
  ∃ m : ℝ, x = m * z^(2/3) :=
by
  sorry

end variation_relationship_l4046_404696


namespace min_value_x_plus_4y_l4046_404684

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : x + y = 2 * x * y) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + y' = 2 * x' * y' → x' + 4 * y' ≥ 9/2) ∧
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 2 * x₀ * y₀ ∧ x₀ + 4 * y₀ = 9/2) :=
by sorry

end min_value_x_plus_4y_l4046_404684


namespace equation_solution_l4046_404626

theorem equation_solution (x : ℝ) : 
  (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
   1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 4) ↔ 
  (x = 5 ∨ x = 1) :=
by sorry

end equation_solution_l4046_404626


namespace sphere_radius_from_intersection_l4046_404638

theorem sphere_radius_from_intersection (width depth : ℝ) (h_width : width = 30) (h_depth : depth = 10) :
  let r := Real.sqrt ((width / 2) ^ 2 + (width / 4 + depth) ^ 2)
  ∃ ε > 0, abs (r - 22.1129) < ε :=
sorry

end sphere_radius_from_intersection_l4046_404638


namespace jogging_distance_l4046_404692

theorem jogging_distance (x t : ℝ) 
  (h1 : (x + 3/4) * (3*t/4) = x * t)
  (h2 : (x - 3/4) * (t + 3) = x * t) :
  x * t = 13.5 := by
  sorry

end jogging_distance_l4046_404692


namespace million_to_scientific_notation_l4046_404678

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem million_to_scientific_notation :
  toScientificNotation (42.39 * 1000000) = ScientificNotation.mk 4.239 7 (by norm_num) :=
sorry

end million_to_scientific_notation_l4046_404678


namespace hamburgers_served_equals_three_l4046_404695

/-- The number of hamburgers made by the restaurant -/
def total_hamburgers : ℕ := 9

/-- The number of hamburgers left over -/
def leftover_hamburgers : ℕ := 6

/-- The number of hamburgers served -/
def served_hamburgers : ℕ := total_hamburgers - leftover_hamburgers

theorem hamburgers_served_equals_three : served_hamburgers = 3 := by
  sorry

end hamburgers_served_equals_three_l4046_404695


namespace teacher_arrangement_count_teacher_arrangement_proof_l4046_404654

theorem teacher_arrangement_count : Nat → Nat → Nat
  | n, k => Nat.choose n k

theorem teacher_arrangement_proof :
  teacher_arrangement_count 22 5 = 26334 := by
  sorry

end teacher_arrangement_count_teacher_arrangement_proof_l4046_404654


namespace negation_equivalence_l4046_404675

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x < 0 ∧ Real.log (x^2 - 2*x - 1) ≥ 0) ↔
  (∀ x : ℝ, x < 0 → Real.log (x^2 - 2*x - 1) < 0) :=
by sorry

end negation_equivalence_l4046_404675


namespace reciprocal_problem_l4046_404644

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Define the smallest composite number
def smallest_composite : ℕ := 4

theorem reciprocal_problem :
  (reciprocal 0.8 = 5/4) ∧
  (reciprocal (1/4) = smallest_composite) := by
  sorry

end reciprocal_problem_l4046_404644


namespace three_x_intercepts_l4046_404657

/-- The function representing the curve x = y^3 - 4y^2 + 3y + 2 -/
def f (y : ℝ) : ℝ := y^3 - 4*y^2 + 3*y + 2

/-- Theorem stating that the equation f(y) = 0 has exactly 3 real solutions -/
theorem three_x_intercepts : ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ y : ℝ, y ∈ s ↔ f y = 0 := by
  sorry

end three_x_intercepts_l4046_404657


namespace rachel_furniture_time_l4046_404637

def chairs : ℕ := 7
def tables : ℕ := 3
def time_per_piece : ℕ := 4

theorem rachel_furniture_time :
  chairs * time_per_piece + tables * time_per_piece = 40 := by
  sorry

end rachel_furniture_time_l4046_404637


namespace odd_function_value_l4046_404669

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + 9

-- State the theorem
theorem odd_function_value (hf_odd : ∀ x, f (-x) = -f x) (hg : g (-2) = 3) : f 2 = 6 := by
  sorry

end odd_function_value_l4046_404669


namespace length_width_relation_l4046_404602

/-- A rectangle enclosed by a wire -/
structure WireRectangle where
  wireLength : ℝ
  width : ℝ
  length : ℝ
  wireLength_positive : 0 < wireLength
  width_positive : 0 < width
  length_positive : 0 < length
  perimeter_eq_wireLength : 2 * (width + length) = wireLength

/-- The relationship between length and width for a 20-meter wire rectangle -/
theorem length_width_relation (rect : WireRectangle) 
    (h : rect.wireLength = 20) : 
    rect.length = -rect.width + 10 := by
  sorry

end length_width_relation_l4046_404602


namespace triangle_area_l4046_404681

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ A < π ∧ B > 0 ∧ B < π ∧ C > 0 ∧ C < π →
  A + B + C = π →
  b^2 + c^2 = a^2 - Real.sqrt 3 * b * c →
  b * c * Real.cos A = -4 →
  (1/2) * b * c * Real.sin A = (2 * Real.sqrt 3) / 3 := by
sorry

end triangle_area_l4046_404681


namespace prob_sum_greater_than_four_l4046_404636

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of outcomes where the sum is 4 or less -/
def outcomes_sum_4_or_less : ℕ := 6

/-- The probability of an event occurring is the number of favorable outcomes
    divided by the total number of possible outcomes -/
theorem prob_sum_greater_than_four :
  (1 - (outcomes_sum_4_or_less : ℚ) / total_outcomes) = 5 / 6 := by
  sorry

end prob_sum_greater_than_four_l4046_404636


namespace binomial_18_10_l4046_404632

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 43758 := by
  sorry

end binomial_18_10_l4046_404632


namespace digit_equation_proof_l4046_404648

theorem digit_equation_proof :
  ∀ (A B D : ℕ),
    A ≤ 9 → B ≤ 9 → D ≤ 9 →
    A ≥ B →
    (100 * A + 10 * B + D) * (A + B + D) = 1323 →
    D = 1 := by
  sorry

end digit_equation_proof_l4046_404648


namespace burger_expenditure_l4046_404634

theorem burger_expenditure (total : ℚ) (movies music ice_cream : ℚ) 
  (h1 : total = 30)
  (h2 : movies = 1/3 * total)
  (h3 : music = 3/10 * total)
  (h4 : ice_cream = 1/5 * total) :
  total - (movies + music + ice_cream) = 5 := by
  sorry

end burger_expenditure_l4046_404634


namespace color_copies_proof_l4046_404600

/-- The price per color copy at print shop X -/
def price_x : ℚ := 1.25

/-- The price per color copy at print shop Y -/
def price_y : ℚ := 2.75

/-- The difference in total charge between print shop Y and X -/
def charge_difference : ℚ := 120

/-- The number of color copies -/
def num_copies : ℚ := 80

theorem color_copies_proof :
  price_y * num_copies = price_x * num_copies + charge_difference :=
by sorry

end color_copies_proof_l4046_404600


namespace complex_magnitude_product_l4046_404629

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 2 - 3 * Complex.I) * 
               (Real.sqrt 5 + 5 * Complex.I) * 
               (2 - 2 * Complex.I)) = 18 * Real.sqrt 10 := by
  sorry

end complex_magnitude_product_l4046_404629


namespace poly_expansion_nonzero_terms_l4046_404690

/-- The polynomial expression -/
def poly (x : ℝ) : ℝ := (2*x+5)*(3*x^2 - x + 4) - 4*(2*x^3 - 3*x^2 + x - 1)

/-- The expanded form of the polynomial -/
def expanded_poly (x : ℝ) : ℝ := 14*x^3 + x^2 + 7*x + 16

/-- The number of nonzero terms in the expanded polynomial -/
def num_nonzero_terms : ℕ := 4

theorem poly_expansion_nonzero_terms :
  (∀ x : ℝ, poly x = expanded_poly x) →
  num_nonzero_terms = 4 :=
by sorry

end poly_expansion_nonzero_terms_l4046_404690


namespace cone_volume_from_lateral_surface_l4046_404603

/-- Given a cone whose lateral surface area is equal to the area of a semicircle with area 2π,
    prove that the volume of the cone is (√3/3)π. -/
theorem cone_volume_from_lateral_surface (cone : Real → Real → Real) 
  (lateral_surface_area : Real) (semicircle_area : Real) :
  lateral_surface_area = semicircle_area →
  semicircle_area = 2 * Real.pi →
  (∃ (r h : Real), cone r h = (1/3) * Real.pi * r^2 * h ∧ 
                   cone r h = (Real.sqrt 3 / 3) * Real.pi) :=
by sorry

end cone_volume_from_lateral_surface_l4046_404603


namespace existence_of_counterexample_l4046_404618

theorem existence_of_counterexample (x y : ℝ) (h : x > y) :
  ∃ (x y : ℝ), x > y ∧ x^2 - 3 ≤ y^2 - 3 := by
  sorry

end existence_of_counterexample_l4046_404618


namespace emily_necklaces_l4046_404643

def beads_per_necklace : ℕ := 28
def total_beads : ℕ := 308

theorem emily_necklaces :
  (total_beads / beads_per_necklace : ℕ) = 11 :=
by sorry

end emily_necklaces_l4046_404643


namespace right_triangle_cos_b_l4046_404673

theorem right_triangle_cos_b (A B C : ℝ) (h1 : A = 90) (h2 : Real.sin B = 3/5) :
  Real.cos B = 4/5 := by
  sorry

end right_triangle_cos_b_l4046_404673


namespace faye_initial_apps_l4046_404641

/-- Represents the number of apps Faye had initially -/
def initial_apps : ℕ := sorry

/-- Represents the number of apps Faye deleted -/
def deleted_apps : ℕ := 8

/-- Represents the number of apps Faye had left after deleting -/
def remaining_apps : ℕ := 4

/-- Theorem stating that the initial number of apps was 12 -/
theorem faye_initial_apps : initial_apps = 12 := by
  sorry

end faye_initial_apps_l4046_404641


namespace not_divide_power_plus_one_l4046_404614

theorem not_divide_power_plus_one (p q m : ℕ) : 
  Nat.Prime p → Nat.Prime q → Odd p → Odd q → p > q → m > 0 → ¬(p * q ∣ m^(p - q) + 1) := by
  sorry

end not_divide_power_plus_one_l4046_404614


namespace final_value_properties_l4046_404691

/-- Represents the transformation process on the blackboard -/
def blackboard_transform (n : ℕ) : Set ℕ → Set ℕ := sorry

/-- The final state of the blackboard after transformations -/
def final_state (n : ℕ) : Set ℕ := sorry

/-- Theorem stating the properties of the final value k -/
theorem final_value_properties (n : ℕ) (h : n ≥ 3) :
  ∃ (k t : ℕ), final_state n = {k} ∧ k = 2^t ∧ k ≥ n :=
sorry

end final_value_properties_l4046_404691


namespace function_properties_l4046_404605

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.sin x + Real.cos x

theorem function_properties (m : ℝ) :
  f m (π / 2) = 1 →
  (∃ T : ℝ, ∀ x : ℝ, f m x = f m (x + T) ∧ T > 0 ∧ ∀ S : ℝ, (∀ x : ℝ, f m x = f m (x + S) ∧ S > 0) → T ≤ S) →
  (∃ M : ℝ, ∀ x : ℝ, f m x ≤ M ∧ ∃ y : ℝ, f m y = M) →
  m = 1 ∧
  (∀ x : ℝ, f m x = Real.sqrt 2 * Real.sin (x + π / 4)) ∧
  (∃ T : ℝ, T = 2 * π ∧ ∀ x : ℝ, f m x = f m (x + T) ∧ T > 0 ∧ ∀ S : ℝ, (∀ x : ℝ, f m x = f m (x + S) ∧ S > 0) → T ≤ S) ∧
  (∃ M : ℝ, M = Real.sqrt 2 ∧ ∀ x : ℝ, f m x ≤ M ∧ ∃ y : ℝ, f m y = M) :=
by sorry

end function_properties_l4046_404605


namespace floor_tiles_1517_902_l4046_404646

/-- The least number of square tiles required to pave a rectangular floor -/
def leastSquareTiles (length width : ℕ) : ℕ :=
  let tileSize := Nat.gcd length width
  ((length + tileSize - 1) / tileSize) * ((width + tileSize - 1) / tileSize)

/-- Proof that 814 square tiles are required for a 1517 cm x 902 cm floor -/
theorem floor_tiles_1517_902 :
  leastSquareTiles 1517 902 = 814 := by
  sorry

end floor_tiles_1517_902_l4046_404646


namespace exponent_multiplication_l4046_404658

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l4046_404658


namespace triangle_side_length_l4046_404677

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  (2 * b = a + c) →  -- Arithmetic sequence
  (B = π / 6) →      -- Angle B = 30°
  (1 / 2 * a * c * Real.sin B = 3 / 2) →  -- Area of triangle
  -- Conclusion
  b = 1 + Real.sqrt 3 := by
  sorry

end triangle_side_length_l4046_404677


namespace incorrect_statement_B_l4046_404666

/-- Definition of a "2 times root equation" -/
def is_two_times_root_equation (a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ (a*x^2 + b*x + c = 0) ∧ (a*y^2 + b*y + c = 0) ∧ (x = 2*y ∨ y = 2*x)

/-- The statement to be proven false -/
theorem incorrect_statement_B :
  ¬(∀ (m n : ℝ), is_two_times_root_equation 1 (m-2) (-2*m) → m + n = 0) :=
sorry

end incorrect_statement_B_l4046_404666


namespace greatest_t_value_l4046_404671

theorem greatest_t_value : ∃ (t : ℝ), 
  (∀ (s : ℝ), (s^2 - s - 40) / (s - 8) = 5 / (s + 5) → s ≤ t) ∧
  (t^2 - t - 40) / (t - 8) = 5 / (t + 5) ∧
  t = -2 := by
  sorry

end greatest_t_value_l4046_404671


namespace spring_excursion_participants_l4046_404698

theorem spring_excursion_participants :
  let water_students : ℕ := 80
  let fruit_students : ℕ := 70
  let neither_students : ℕ := 6
  let both_students : ℕ := (water_students + fruit_students - neither_students) / 2
  let total_participants : ℕ := both_students * 2
  total_participants = 104 :=
by sorry

end spring_excursion_participants_l4046_404698


namespace correct_outfit_count_l4046_404612

/-- The number of shirts -/
def num_shirts : ℕ := 5

/-- The number of pants -/
def num_pants : ℕ := 6

/-- The number of formal pants -/
def num_formal_pants : ℕ := 3

/-- The number of casual pants -/
def num_casual_pants : ℕ := num_pants - num_formal_pants

/-- The number of shirts that can be paired with formal pants -/
def num_shirts_for_formal : ℕ := 3

/-- Calculate the number of different outfits -/
def num_outfits : ℕ :=
  (num_casual_pants * num_shirts) + (num_formal_pants * num_shirts_for_formal)

theorem correct_outfit_count : num_outfits = 24 := by
  sorry

end correct_outfit_count_l4046_404612


namespace jurassic_zoo_bill_l4046_404665

/-- The Jurassic Zoo billing problem -/
theorem jurassic_zoo_bill :
  let adult_price : ℕ := 8
  let child_price : ℕ := 4
  let total_people : ℕ := 201
  let total_children : ℕ := 161
  let total_adults : ℕ := total_people - total_children
  let adults_bill : ℕ := total_adults * adult_price
  let children_bill : ℕ := total_children * child_price
  let total_bill : ℕ := adults_bill + children_bill
  total_bill = 964 := by
  sorry

end jurassic_zoo_bill_l4046_404665


namespace least_addition_for_divisibility_l4046_404687

theorem least_addition_for_divisibility : 
  ∃ (n : ℕ), n = 256 ∧ 
  (∀ m : ℕ, (1019 + m) % 25 = 0 ∧ (1019 + m) % 17 = 0 → m ≥ n) ∧
  (1019 + n) % 25 = 0 ∧ (1019 + n) % 17 = 0 :=
by sorry

end least_addition_for_divisibility_l4046_404687


namespace mindy_message_count_l4046_404674

/-- The number of emails and phone messages Mindy has in total -/
def total_messages (phone_messages : ℕ) (emails : ℕ) : ℕ :=
  phone_messages + emails

/-- The relationship between emails and phone messages -/
def email_phone_relation (phone_messages : ℕ) : ℕ :=
  9 * phone_messages - 7

theorem mindy_message_count :
  ∃ (phone_messages : ℕ),
    email_phone_relation phone_messages = 83 ∧
    total_messages phone_messages 83 = 93 := by
  sorry

end mindy_message_count_l4046_404674


namespace chocolate_bar_count_l4046_404688

/-- The number of chocolate bars in a box, given the cost per bar and the sales amount when all but 3 bars are sold. -/
def number_of_bars (cost_per_bar : ℕ) (sales_amount : ℕ) : ℕ :=
  (sales_amount + 3 * cost_per_bar) / cost_per_bar

theorem chocolate_bar_count : number_of_bars 3 18 = 9 := by
  sorry

end chocolate_bar_count_l4046_404688


namespace total_rose_bushes_l4046_404610

theorem total_rose_bushes (rose_cost : ℕ) (aloe_cost : ℕ) (friend_roses : ℕ) (total_spent : ℕ) (aloe_count : ℕ) : 
  rose_cost = 75 → 
  friend_roses = 2 → 
  aloe_cost = 100 → 
  aloe_count = 2 → 
  total_spent = 500 → 
  (total_spent - aloe_count * aloe_cost) / rose_cost + friend_roses = 6 := by
sorry

end total_rose_bushes_l4046_404610


namespace at_least_one_angle_leq_60_l4046_404645

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : a + b + c = 180

-- Theorem statement
theorem at_least_one_angle_leq_60 (t : Triangle) : 
  t.a ≤ 60 ∨ t.b ≤ 60 ∨ t.c ≤ 60 := by
  sorry

end at_least_one_angle_leq_60_l4046_404645


namespace complex_fraction_inequality_l4046_404651

theorem complex_fraction_inequality (a b c : ℂ) 
  (h1 : a * b + a * c - b * c ≠ 0) 
  (h2 : b * a + b * c - a * c ≠ 0) 
  (h3 : c * a + c * b - a * b ≠ 0) : 
  Complex.abs (a^2 / (a * b + a * c - b * c)) + 
  Complex.abs (b^2 / (b * a + b * c - a * c)) + 
  Complex.abs (c^2 / (c * a + c * b - a * b)) ≥ 3/2 := by
  sorry

end complex_fraction_inequality_l4046_404651


namespace product_unit_digit_l4046_404639

-- Define a function to get the unit digit of a number
def unitDigit (n : ℕ) : ℕ := n % 10

-- Define the numbers given in the problem
def a : ℕ := 7858
def b : ℕ := 1086
def c : ℕ := 4582
def d : ℕ := 9783

-- State the theorem
theorem product_unit_digit :
  unitDigit (a * b * c * d) = 4 := by
  sorry

end product_unit_digit_l4046_404639


namespace fire_truck_ladder_height_l4046_404615

theorem fire_truck_ladder_height (distance_to_building : ℝ) (ladder_length : ℝ) :
  distance_to_building = 5 →
  ladder_length = 13 →
  ∃ (height : ℝ), height^2 + distance_to_building^2 = ladder_length^2 ∧ height = 12 :=
by sorry

end fire_truck_ladder_height_l4046_404615


namespace prime_power_divisibility_l4046_404606

theorem prime_power_divisibility (p n : ℕ) (hp : Prime p) (h : p ∣ n^2020) : p^2020 ∣ n^2020 := by
  sorry

end prime_power_divisibility_l4046_404606


namespace polynomial_division_l4046_404694

def P (x : ℝ) : ℝ := x^6 - 6*x^4 - 4*x^3 + 9*x^2 + 12*x + 4

def Q (x : ℝ) : ℝ := x^4 + x^3 - 3*x^2 - 5*x - 2

def R (x : ℝ) : ℝ := x^2 - x - 2

theorem polynomial_division :
  ∀ x : ℝ, P x = Q x * R x :=
by sorry

end polynomial_division_l4046_404694


namespace no_prime_divisible_by_55_l4046_404613

theorem no_prime_divisible_by_55 : ¬ ∃ p : ℕ, Nat.Prime p ∧ 55 ∣ p := by
  sorry

end no_prime_divisible_by_55_l4046_404613


namespace horner_operations_for_f_l4046_404652

/-- Represents a polynomial of degree 4 -/
structure Polynomial4 where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- Counts the number of operations in Horner's method for a polynomial of degree 4 -/
def hornerOperations (p : Polynomial4) : ℕ × ℕ :=
  sorry

/-- The specific polynomial f(x) = 2x^4 + 3x^3 - 2x^2 + 4x - 6 -/
def f : Polynomial4 := {
  a := 2
  b := 3
  c := -2
  d := 4
  e := -6
}

theorem horner_operations_for_f :
  hornerOperations f = (4, 4) := by sorry

end horner_operations_for_f_l4046_404652


namespace shaded_fraction_of_rectangle_l4046_404608

theorem shaded_fraction_of_rectangle (length width : ℕ) (shaded_area : ℚ) :
  length = 15 →
  width = 20 →
  shaded_area = (1 / 2 : ℚ) * (1 / 4 : ℚ) * (length * width : ℚ) →
  shaded_area / (length * width : ℚ) = 1 / 8 := by
  sorry

end shaded_fraction_of_rectangle_l4046_404608


namespace painting_price_change_l4046_404647

theorem painting_price_change (P : ℝ) (x : ℝ) 
  (h1 : P > 0) 
  (h2 : (1.1 * P) * (1 - x / 100) = 0.935 * P) : 
  x = 15 := by
  sorry

end painting_price_change_l4046_404647


namespace farthest_corner_distance_l4046_404642

/-- Represents a rectangular pool with given dimensions -/
structure Pool :=
  (length : ℝ)
  (width : ℝ)

/-- Calculates the perimeter of a rectangular pool -/
def perimeter (p : Pool) : ℝ := 2 * (p.length + p.width)

/-- Theorem: In a 10m × 25m pool, if three children walk 50m total,
    the distance to the farthest corner is 20m -/
theorem farthest_corner_distance (p : Pool) 
  (h1 : p.length = 25)
  (h2 : p.width = 10)
  (h3 : ∃ (x : ℝ), x ≥ 0 ∧ x ≤ perimeter p ∧ perimeter p - x = 50) :
  ∃ (y : ℝ), y = 20 ∧ y = perimeter p - 50 := by
  sorry

end farthest_corner_distance_l4046_404642


namespace fish_population_estimate_l4046_404668

/-- Estimate the number of fish in a pond using mark-recapture method -/
theorem fish_population_estimate 
  (initially_marked : ℕ) 
  (recaptured : ℕ) 
  (marked_in_recapture : ℕ) 
  (h1 : initially_marked = 2000)
  (h2 : recaptured = 500)
  (h3 : marked_in_recapture = 40) :
  (initially_marked * recaptured) / marked_in_recapture = 25000 := by
  sorry

#eval (2000 * 500) / 40

end fish_population_estimate_l4046_404668


namespace monotonic_interval_implies_a_bound_l4046_404627

open Real

theorem monotonic_interval_implies_a_bound (a : ℝ) :
  (∃ x ∈ Set.Ioo 1 2, (fun x => 1/x + 2*a*x - 2) > 0) →
  a > -1/2 := by
sorry

end monotonic_interval_implies_a_bound_l4046_404627


namespace min_quotient_is_53_5_l4046_404622

/-- A three-digit number with distinct non-zero digits -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0
  a_lt_ten : a < 10
  b_lt_ten : b < 10
  c_lt_ten : c < 10
  distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of digits of a three-digit number -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.a + n.b + n.c

/-- The quotient of a three-digit number divided by the sum of its digits -/
def quotient (n : ThreeDigitNumber) : Rat :=
  (value n : Rat) / (digitSum n : Rat)

theorem min_quotient_is_53_5 :
  ∃ (min : Rat), ∀ (n : ThreeDigitNumber), quotient n ≥ min ∧ (∃ (m : ThreeDigitNumber), quotient m = min) ∧ min = 53.5 := by
  sorry

end min_quotient_is_53_5_l4046_404622


namespace g_monotonic_intervals_exactly_two_tangent_points_l4046_404683

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else -x^2 + 2*x - 1/2

-- Define g(x) = x * f(x)
noncomputable def g (x : ℝ) : ℝ := x * f x

-- Theorem for monotonic intervals of g(x)
theorem g_monotonic_intervals :
  (∀ x y, x < y ∧ y < -1 → g y < g x) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y ≤ 0 → g x < g y) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < (4 - Real.sqrt 10) / 6 → g y < g x) ∧
  (∀ x y, (4 - Real.sqrt 10) / 6 < x ∧ x < y ∧ y < (4 + Real.sqrt 10) / 6 → g x < g y) ∧
  (∀ x y, (4 + Real.sqrt 10) / 6 < x ∧ x < y → g y < g x) :=
sorry

-- Theorem for existence of exactly two tangent points
theorem exactly_two_tangent_points :
  ∃! (x₁ x₂ : ℝ), x₁ < x₂ ∧
    ∃ (m b : ℝ), 
      (∀ x, f x ≤ m * x + b) ∧
      f x₁ = m * x₁ + b ∧
      f x₂ = m * x₂ + b :=
sorry

end g_monotonic_intervals_exactly_two_tangent_points_l4046_404683


namespace geometric_sequence_sum_l4046_404623

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q)
    (h_sum1 : a 3 + a 6 = 6) (h_sum2 : a 5 + a 8 = 9) :
  a 7 + a 10 = 27 / 2 := by
  sorry

end geometric_sequence_sum_l4046_404623


namespace no_real_solutions_l4046_404655

theorem no_real_solutions : ¬ ∃ (x : ℝ), (x^(1/4) : ℝ) = 20 / (9 - 2 * (x^(1/4) : ℝ)) := by
  sorry

end no_real_solutions_l4046_404655


namespace unused_ribbon_length_l4046_404685

/-- Given a ribbon of length 30 meters cut into 6 equal parts, 
    if 4 parts are used, then 10 meters of ribbon are not used. -/
theorem unused_ribbon_length 
  (total_length : ℝ) 
  (num_parts : ℕ) 
  (used_parts : ℕ) 
  (h1 : total_length = 30) 
  (h2 : num_parts = 6) 
  (h3 : used_parts = 4) : 
  total_length - (total_length / num_parts) * used_parts = 10 := by
  sorry


end unused_ribbon_length_l4046_404685


namespace tangent_circle_center_l4046_404682

/-- A circle tangent to two parallel lines with its center on a third line --/
structure TangentCircle where
  /-- The x-coordinate of the circle's center --/
  x : ℝ
  /-- The y-coordinate of the circle's center --/
  y : ℝ
  /-- The circle is tangent to the line 4x - 3y = 30 --/
  tangent_line1 : 4 * x - 3 * y = 30
  /-- The circle is tangent to the line 4x - 3y = -10 --/
  tangent_line2 : 4 * x - 3 * y = -10
  /-- The center of the circle lies on the line 2x + y = 0 --/
  center_line : 2 * x + y = 0

/-- The center of the circle satisfies all conditions and has coordinates (1, -2) --/
theorem tangent_circle_center : 
  ∃ (c : TangentCircle), c.x = 1 ∧ c.y = -2 :=
sorry

end tangent_circle_center_l4046_404682


namespace product_of_differences_l4046_404697

theorem product_of_differences (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) :
  (a - 1) * (b - 1) = -1 := by
  sorry

end product_of_differences_l4046_404697


namespace bus_stop_time_l4046_404616

/-- The time a bus stops per hour given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 54)
  (h2 : speed_with_stops = 45) :
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 10 := by
  sorry

end bus_stop_time_l4046_404616


namespace negation_of_existence_negation_of_quadratic_equation_l4046_404631

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) :=
by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x > 0, x^2 - 5*x + 6 = 0) ↔ (∀ x > 0, x^2 - 5*x + 6 ≠ 0) :=
by sorry

end negation_of_existence_negation_of_quadratic_equation_l4046_404631


namespace functional_equation_solution_l4046_404689

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y z t : ℝ, (f x + f z) * (f y + f t) = f (x * y - z * t) + f (x * t + y * z)) →
  (∀ x : ℝ, f x = 0 ∨ f x = (1 : ℝ) / 2 ∨ f x = x ^ 2) :=
by sorry

end functional_equation_solution_l4046_404689


namespace uma_income_l4046_404635

theorem uma_income (uma_income bala_income uma_expenditure bala_expenditure : ℚ)
  (income_ratio : uma_income / bala_income = 4 / 3)
  (expenditure_ratio : uma_expenditure / bala_expenditure = 3 / 2)
  (uma_savings : uma_income - uma_expenditure = 5000)
  (bala_savings : bala_income - bala_expenditure = 5000) :
  uma_income = 20000 := by
  sorry

end uma_income_l4046_404635


namespace all_offers_count_l4046_404670

def stadium_capacity : ℕ := 4500

def hot_dog_interval : ℕ := 90
def soda_interval : ℕ := 45
def popcorn_interval : ℕ := 60
def ice_cream_interval : ℕ := 45

def fans_with_all_offers : ℕ := stadium_capacity / (Nat.lcm hot_dog_interval (Nat.lcm soda_interval popcorn_interval))

theorem all_offers_count :
  fans_with_all_offers = 25 :=
sorry

end all_offers_count_l4046_404670


namespace xiaohuas_stamp_buying_ways_l4046_404686

/-- Represents the number of ways to buy stamps given the total money and stamp prices -/
def waysToByStamps (totalMoney : ℕ) (stamp1Price : ℕ) (stamp2Price : ℕ) : ℕ := 
  let maxStamp1 := totalMoney / stamp1Price
  let maxStamp2 := totalMoney / stamp2Price
  (maxStamp1 + 1) * (maxStamp2 + 1) - 1

/-- The problem statement -/
theorem xiaohuas_stamp_buying_ways :
  waysToByStamps 7 2 3 = 7 := by
  sorry

end xiaohuas_stamp_buying_ways_l4046_404686


namespace rectangle_dimensions_l4046_404659

theorem rectangle_dimensions (x : ℝ) : 
  (x + 3) * (3 * x - 2) = 9 * x + 1 → 
  x > 0 → 
  3 * x - 2 > 0 → 
  x = (11 + Real.sqrt 205) / 6 := by
sorry

end rectangle_dimensions_l4046_404659


namespace quadratic_equation_solution_l4046_404676

/-- A quadratic function f(x) = x² + bx - 5 with axis of symmetry at x = 2 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 5

/-- The axis of symmetry of f is at x = 2 -/
def axis_of_symmetry (b : ℝ) : Prop := -b/(2*1) = 2

/-- The equation f(x) = 2x - 13 -/
def equation (b : ℝ) (x : ℝ) : Prop := f b x = 2*x - 13

theorem quadratic_equation_solution (b : ℝ) :
  axis_of_symmetry b →
  (∃ x y : ℝ, x = 2 ∧ y = 4 ∧ equation b x ∧ equation b y ∧
    ∀ z : ℝ, equation b z → (z = x ∨ z = y)) :=
by sorry

end quadratic_equation_solution_l4046_404676


namespace problem_statement_l4046_404607

theorem problem_statement : 
  Real.sqrt 12 + |1 - Real.sqrt 3| + (π - 2023)^0 = 3 * Real.sqrt 3 := by sorry

end problem_statement_l4046_404607


namespace square_difference_48_3_l4046_404661

theorem square_difference_48_3 : 48^2 - 2*(48*3) + 3^2 = 2025 := by
  sorry

end square_difference_48_3_l4046_404661


namespace remainder_sum_l4046_404680

theorem remainder_sum (n : ℤ) : n % 15 = 7 → (n % 3 + n % 5 = 3) := by
  sorry

end remainder_sum_l4046_404680


namespace test_score_problem_l4046_404679

theorem test_score_problem (total_questions : ℕ) (score : ℤ) 
  (correct_answers : ℕ) (incorrect_answers : ℕ) : 
  total_questions = 100 →
  score = correct_answers - 2 * incorrect_answers →
  correct_answers + incorrect_answers = total_questions →
  score = 73 →
  correct_answers = 91 := by
sorry

end test_score_problem_l4046_404679


namespace arithmetic_geometric_mean_inequality_for_two_l4046_404640

theorem arithmetic_geometric_mean_inequality_for_two (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt x + Real.sqrt y) / 2 ≤ Real.sqrt ((x + y) / 2) := by
  sorry

end arithmetic_geometric_mean_inequality_for_two_l4046_404640


namespace novels_difference_l4046_404620

def jordan_novels : ℕ := 120

def alexandre_novels : ℕ := jordan_novels / 10

theorem novels_difference : jordan_novels - alexandre_novels = 108 := by
  sorry

end novels_difference_l4046_404620


namespace g_expression_l4046_404625

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define the relationship between f and g
def g_relation (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 2) = f x

-- Theorem statement
theorem g_expression (g : ℝ → ℝ) (h : g_relation g) :
  ∀ x, g x = 2 * x - 1 := by
  sorry

end g_expression_l4046_404625


namespace sin_2alpha_plus_5pi_18_l4046_404649

theorem sin_2alpha_plus_5pi_18 (α : ℝ) (h : Real.sin (π / 9 - α) = 1 / 3) : 
  Real.sin (2 * α + 5 * π / 18) = 7 / 9 := by
  sorry

end sin_2alpha_plus_5pi_18_l4046_404649


namespace sum_equals_210_l4046_404667

theorem sum_equals_210 : 145 + 35 + 25 + 5 = 210 := by
  sorry

end sum_equals_210_l4046_404667


namespace percentage_difference_in_earnings_l4046_404663

/-- Given Mike's and Phil's hourly earnings, calculate the percentage difference -/
theorem percentage_difference_in_earnings (mike_earnings phil_earnings : ℝ) 
  (h1 : mike_earnings = 14)
  (h2 : phil_earnings = 7) :
  (mike_earnings - phil_earnings) / mike_earnings * 100 = 50 := by
sorry

end percentage_difference_in_earnings_l4046_404663


namespace term_properties_l4046_404604

-- Define a structure for a monomial term
structure Monomial where
  coefficient : ℚ
  x_power : ℕ
  y_power : ℕ

-- Define the monomial -1/3 * x * y^2
def term : Monomial := {
  coefficient := -1/3,
  x_power := 1,
  y_power := 2
}

-- Define the coefficient of a monomial
def coefficient (m : Monomial) : ℚ := m.coefficient

-- Define the degree of a monomial
def degree (m : Monomial) : ℕ := m.x_power + m.y_power

-- Theorem stating the coefficient and degree of the term
theorem term_properties :
  coefficient term = -1/3 ∧ degree term = 3 := by
  sorry


end term_properties_l4046_404604


namespace solution_value_l4046_404662

theorem solution_value (k : ℝ) : (2 * 3 - k + 1 = 0) → k = 7 := by
  sorry

end solution_value_l4046_404662


namespace distance_ratio_theorem_l4046_404699

/-- Represents a square pyramid -/
structure SquarePyramid where
  -- Base side length
  a : ℝ
  -- Height
  h : ℝ
  -- Assume positive dimensions
  a_pos : 0 < a
  h_pos : 0 < h

/-- A point inside the base square -/
structure BasePoint where
  x : ℝ
  y : ℝ
  -- Assume the point is inside the base square
  x_bound : 0 < x ∧ x < 1
  y_bound : 0 < y ∧ y < 1

/-- Sum of distances from a point to the triangular faces -/
noncomputable def sumDistancesToFaces (p : SquarePyramid) (e : BasePoint) : ℝ := sorry

/-- Sum of distances from a point to the base edges -/
noncomputable def sumDistancesToEdges (p : SquarePyramid) (e : BasePoint) : ℝ := sorry

/-- The main theorem -/
theorem distance_ratio_theorem (p : SquarePyramid) (e : BasePoint) :
  sumDistancesToFaces p e / sumDistancesToEdges p e = p.h / p.a := by sorry

end distance_ratio_theorem_l4046_404699


namespace dividend_calculation_l4046_404693

theorem dividend_calculation (quotient divisor remainder : ℕ) 
  (hq : quotient = 120) 
  (hd : divisor = 456) 
  (hr : remainder = 333) : 
  divisor * quotient + remainder = 55053 := by
sorry

end dividend_calculation_l4046_404693


namespace digestion_period_correct_l4046_404664

/-- The period (in days) for a python to completely digest an alligator -/
def digestion_period : ℕ := 7

/-- The number of days observed -/
def observation_days : ℕ := 616

/-- The maximum number of alligators eaten in the observation period -/
def max_alligators_eaten : ℕ := 88

/-- Theorem stating that the digestion period is correct given the observed data -/
theorem digestion_period_correct : 
  digestion_period * max_alligators_eaten = observation_days :=
by sorry

end digestion_period_correct_l4046_404664
