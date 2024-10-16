import Mathlib

namespace NUMINAMATH_CALUDE_square_minus_three_product_plus_square_l2861_286170

theorem square_minus_three_product_plus_square (a b : ℝ) 
  (sum_eq : a + b = 8) 
  (product_eq : a * b = 9) : 
  a^2 - 3*a*b + b^2 = 19 := by
sorry

end NUMINAMATH_CALUDE_square_minus_three_product_plus_square_l2861_286170


namespace NUMINAMATH_CALUDE_carrie_punch_ice_amount_l2861_286152

/-- Represents the ingredients and result of Carrie's punch recipe --/
structure PunchRecipe where
  mountain_dew_cans : Nat
  mountain_dew_oz_per_can : Nat
  fruit_juice_oz : Nat
  servings : Nat
  oz_per_serving : Nat

/-- Calculates the amount of ice added to the punch --/
def ice_added (recipe : PunchRecipe) : Nat :=
  recipe.servings * recipe.oz_per_serving - 
  (recipe.mountain_dew_cans * recipe.mountain_dew_oz_per_can + recipe.fruit_juice_oz)

/-- Theorem stating that Carrie added 28 oz of ice to her punch --/
theorem carrie_punch_ice_amount : 
  ice_added { mountain_dew_cans := 6
            , mountain_dew_oz_per_can := 12
            , fruit_juice_oz := 40
            , servings := 14
            , oz_per_serving := 10 } = 28 := by
  sorry

end NUMINAMATH_CALUDE_carrie_punch_ice_amount_l2861_286152


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l2861_286101

theorem salt_solution_mixture : ∀ (x y : ℝ),
  x > 0 ∧ y > 0 ∧ x + y = 90 ∧
  0.05 * x + 0.20 * y = 0.07 * 90 →
  x = 78 ∧ y = 12 :=
by sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l2861_286101


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l2861_286103

/-- Given a geometric sequence where the third term is 27 and the fourth term is 36,
    prove that the first term of the sequence is 243/16. -/
theorem geometric_sequence_first_term (a : ℚ) (r : ℚ) :
  a * r^2 = 27 ∧ a * r^3 = 36 → a = 243/16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l2861_286103


namespace NUMINAMATH_CALUDE_expression_value_l2861_286172

theorem expression_value (x y : ℤ) (hx : x = -2) (hy : y = 4) :
  5 * x - 2 * y + 7 = -11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2861_286172


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l2861_286194

theorem quadratic_form_minimum (x y z : ℝ) :
  x^2 + 2*x*y + 3*y^2 + 2*x*z + 3*z^2 ≥ 0 ∧
  (x^2 + 2*x*y + 3*y^2 + 2*x*z + 3*z^2 = 0 ↔ x = 0 ∧ y = 0 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l2861_286194


namespace NUMINAMATH_CALUDE_new_girl_weight_l2861_286155

theorem new_girl_weight (n : ℕ) (initial_weight replaced_weight : ℝ) 
  (h1 : n = 25)
  (h2 : replaced_weight = 55)
  (h3 : (initial_weight - replaced_weight + new_weight) / n = initial_weight / n + 1) :
  new_weight = 80 :=
sorry

end NUMINAMATH_CALUDE_new_girl_weight_l2861_286155


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_l2861_286130

theorem min_value_sum_fractions (a b c k : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k > 0) :
  (a / (k * b) + b / (k * c) + c / (k * a)) ≥ 3 / k ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    (a₀ / (k * b₀) + b₀ / (k * c₀) + c₀ / (k * a₀)) = 3 / k :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_l2861_286130


namespace NUMINAMATH_CALUDE_whisky_alcohol_percentage_l2861_286181

/-- The initial percentage of alcohol in a jar of whisky -/
def initial_alcohol_percentage : ℝ := 40

/-- The percentage of alcohol in the replacement whisky -/
def replacement_alcohol_percentage : ℝ := 19

/-- The percentage of alcohol after replacement -/
def final_alcohol_percentage : ℝ := 24

/-- The quantity of whisky replaced -/
def replaced_quantity : ℝ := 0.7619047619047619

/-- The total volume of whisky in the jar -/
def total_volume : ℝ := 1

theorem whisky_alcohol_percentage :
  initial_alcohol_percentage / 100 * (total_volume - replaced_quantity) +
  replacement_alcohol_percentage / 100 * replaced_quantity =
  final_alcohol_percentage / 100 * total_volume := by
  sorry

end NUMINAMATH_CALUDE_whisky_alcohol_percentage_l2861_286181


namespace NUMINAMATH_CALUDE_all_inequalities_true_l2861_286119

theorem all_inequalities_true (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hz : z > 0) :
  (x + z > y + z) ∧
  (x - 2*z > y - 2*z) ∧
  (x*z^2 > y*z^2) ∧
  (x/z > y/z) ∧
  (x - z^2 > y - z^2) := by
  sorry

end NUMINAMATH_CALUDE_all_inequalities_true_l2861_286119


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_when_a_is_2_A_intersect_B_equals_B_iff_a_less_than_0_l2861_286186

-- Define sets A and B
def A : Set ℝ := {x | x < -3 ∨ x ≥ 2}
def B (a : ℝ) : Set ℝ := {x | x ≤ a - 3}

-- Theorem 1
theorem complement_A_intersect_B_when_a_is_2 :
  (Set.univ \ A) ∩ B 2 = {x | -3 ≤ x ∧ x ≤ -1} := by sorry

-- Theorem 2
theorem A_intersect_B_equals_B_iff_a_less_than_0 (a : ℝ) :
  A ∩ B a = B a ↔ a < 0 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_when_a_is_2_A_intersect_B_equals_B_iff_a_less_than_0_l2861_286186


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2861_286124

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a1 : a 1 = 2) 
  (h_sum : a 3 + a 5 = 10) : 
  a 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2861_286124


namespace NUMINAMATH_CALUDE_initial_balls_count_l2861_286118

theorem initial_balls_count (initial : ℕ) (current : ℕ) (removed : ℕ) : 
  current = 6 → removed = 2 → initial = current + removed → initial = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_balls_count_l2861_286118


namespace NUMINAMATH_CALUDE_equation_describes_hyperbola_l2861_286148

/-- The equation (x-y)^2 = x^2 + y^2 - 2 describes a hyperbola -/
theorem equation_describes_hyperbola :
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (∀ (x y : ℝ), (x - y)^2 = x^2 + y^2 - 2 ↔ (x * y = 1)) :=
sorry

end NUMINAMATH_CALUDE_equation_describes_hyperbola_l2861_286148


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l2861_286100

theorem max_value_of_fraction (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) :
  ∃ (z : ℝ), z = (y + x) / x ∧ z ≤ 1 + Real.sqrt 3 ∧
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ (y₀ + x₀) / x₀ = 1 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l2861_286100


namespace NUMINAMATH_CALUDE_solve_pretzel_problem_l2861_286147

def pretzel_problem (barry_pretzels : ℕ) (angie_ratio : ℕ) (shelly_ratio : ℚ) (dave_percentage : ℚ) : Prop :=
  let shelly_pretzels := barry_pretzels * shelly_ratio
  let angie_pretzels := shelly_pretzels * angie_ratio
  let dave_pretzels := (angie_pretzels + shelly_pretzels) * dave_percentage
  angie_pretzels = 18 ∧ dave_pretzels = 6

theorem solve_pretzel_problem :
  pretzel_problem 12 3 (1/2) (1/4) := by
  sorry

end NUMINAMATH_CALUDE_solve_pretzel_problem_l2861_286147


namespace NUMINAMATH_CALUDE_square_and_cube_roots_l2861_286162

theorem square_and_cube_roots (a b : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ (a + 3)^2 = x ∧ (2*a - 15)^2 = x) → 
  ((-2)^3 = b) → 
  (a = 18 ∧ b = -8) ∧ 
  Real.sqrt (2*a - b) = 4 := by
sorry

end NUMINAMATH_CALUDE_square_and_cube_roots_l2861_286162


namespace NUMINAMATH_CALUDE_max_red_socks_l2861_286110

theorem max_red_socks (r b : ℕ) : 
  let t := r + b
  (t ≤ 2023) →
  (r * (r - 1) + b * (b - 1)) / (t * (t - 1)) = 2 / 5 →
  r ≤ 990 ∧ ∃ (r' : ℕ), r' = 990 ∧ 
    ∃ (b' : ℕ), (r' + b' ≤ 2023) ∧ 
    (r' * (r' - 1) + b' * (b' - 1)) / ((r' + b') * (r' + b' - 1)) = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_max_red_socks_l2861_286110


namespace NUMINAMATH_CALUDE_book_categorization_l2861_286178

/-- Proves that given 800 books initially divided into 4 equal categories, 
    then each category divided into 5 groups, the number of final categories 
    when each group is further divided into categories of 20 books each is 40. -/
theorem book_categorization (total_books : Nat) (initial_categories : Nat) 
    (groups_per_category : Nat) (books_per_final_category : Nat) 
    (h1 : total_books = 800)
    (h2 : initial_categories = 4)
    (h3 : groups_per_category = 5)
    (h4 : books_per_final_category = 20) : 
    (total_books / initial_categories / groups_per_category / books_per_final_category) * 
    (initial_categories * groups_per_category) = 40 := by
  sorry

#check book_categorization

end NUMINAMATH_CALUDE_book_categorization_l2861_286178


namespace NUMINAMATH_CALUDE_inequality_proof_l2861_286106

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) ≥ Real.sqrt (3 / 2) * Real.sqrt (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2861_286106


namespace NUMINAMATH_CALUDE_cubic_difference_l2861_286169

theorem cubic_difference (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 20) :
  x^3 - y^3 = -448 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_l2861_286169


namespace NUMINAMATH_CALUDE_inequality_proof_l2861_286113

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b / (a^5 + b^5 + a * b)) + (b * c / (b^5 + c^5 + b * c)) + (c * a / (c^5 + a^5 + c * a)) ≤ 1 ∧
  ((a * b / (a^5 + b^5 + a * b)) + (b * c / (b^5 + c^5 + b * c)) + (c * a / (c^5 + a^5 + c * a)) = 1 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2861_286113


namespace NUMINAMATH_CALUDE_charlies_coins_l2861_286145

theorem charlies_coins (total_coins : ℕ) (pennies nickels : ℕ) : 
  total_coins = 17 →
  pennies + nickels = total_coins →
  pennies = nickels + 2 →
  pennies * 1 + nickels * 5 = 44 :=
by sorry

end NUMINAMATH_CALUDE_charlies_coins_l2861_286145


namespace NUMINAMATH_CALUDE_cubic_root_reciprocal_sum_l2861_286175

theorem cubic_root_reciprocal_sum (a b c : ℝ) : 
  a^3 - 15*a^2 + 26*a - 8 = 0 → 
  b^3 - 15*b^2 + 26*b - 8 = 0 → 
  c^3 - 15*c^2 + 26*c - 8 = 0 → 
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 109/16 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_reciprocal_sum_l2861_286175


namespace NUMINAMATH_CALUDE_simplify_T_l2861_286129

theorem simplify_T (x : ℝ) : 
  (x + 2)^6 + 6*(x + 2)^5 + 15*(x + 2)^4 + 20*(x + 2)^3 + 15*(x + 2)^2 + 6*(x + 2) + 1 = (x + 3)^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_T_l2861_286129


namespace NUMINAMATH_CALUDE_intersection_in_first_quadrant_implies_a_greater_than_two_l2861_286149

/-- Two lines intersect in the first quadrant implies a > 2 -/
theorem intersection_in_first_quadrant_implies_a_greater_than_two 
  (a : ℝ) 
  (l₁ : ℝ → ℝ → Prop) 
  (l₂ : ℝ → ℝ → Prop) 
  (h₁ : ∀ x y, l₁ x y ↔ a * x - y + 2 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ x + y - a = 0)
  (h_intersection : ∃ x y, l₁ x y ∧ l₂ x y ∧ x > 0 ∧ y > 0) : 
  a > 2 := by
sorry


end NUMINAMATH_CALUDE_intersection_in_first_quadrant_implies_a_greater_than_two_l2861_286149


namespace NUMINAMATH_CALUDE_food_lasts_five_more_days_l2861_286191

/-- Calculates the number of additional days food lasts after more men join -/
def additional_days_food_lasts (initial_men : ℕ) (initial_days : ℕ) (days_before_joining : ℕ) (additional_men : ℕ) : ℕ :=
  let total_food := initial_men * initial_days
  let remaining_food := total_food - (initial_men * days_before_joining)
  let total_men := initial_men + additional_men
  remaining_food / total_men

/-- Proves that given the initial conditions, the food lasts for 5 additional days -/
theorem food_lasts_five_more_days :
  additional_days_food_lasts 760 22 2 2280 = 5 := by
  sorry

#eval additional_days_food_lasts 760 22 2 2280

end NUMINAMATH_CALUDE_food_lasts_five_more_days_l2861_286191


namespace NUMINAMATH_CALUDE_root_equation_problem_l2861_286121

theorem root_equation_problem (c d : ℝ) : 
  (∃! x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    ((x + c) * (x + d) * (x + 10)) / ((x + 5)^2) = 0 ∧
    ((y + c) * (y + d) * (y + 10)) / ((y + 5)^2) = 0 ∧
    ((z + c) * (z + d) * (z + 10)) / ((z + 5)^2) = 0) ∧
  (∃! w : ℝ, ((w + 3*c) * (w + 2) * (w + 4)) / ((w + d) * (w + 10)) = 0) →
  50 * c + 10 * d = 310 / 3 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l2861_286121


namespace NUMINAMATH_CALUDE_probability_of_selecting_girl_l2861_286199

theorem probability_of_selecting_girl (num_boys num_girls : ℕ) 
  (h_boys : num_boys = 3) 
  (h_girls : num_girls = 2) : 
  (num_girls : ℚ) / ((num_boys + num_girls) : ℚ) = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_selecting_girl_l2861_286199


namespace NUMINAMATH_CALUDE_odd_number_difference_difference_is_98_l2861_286168

theorem odd_number_difference : ℕ → Prop :=
  fun n => ∃ (a b : ℕ), 
    (a ≤ 100 ∧ b ≤ 100) ∧  -- Numbers are in the range 1 to 100
    (Odd a ∧ Odd b) ∧      -- Both numbers are odd
    (∀ k, k ≤ 100 → Odd k → a ≤ k ∧ k ≤ b) ∧  -- a is smallest, b is largest odd number
    b - a = n              -- Their difference is n

theorem difference_is_98 : odd_number_difference 98 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_difference_difference_is_98_l2861_286168


namespace NUMINAMATH_CALUDE_family_weight_ratio_l2861_286196

/-- Given the weights of a family, prove the ratio of child's weight to grandmother's weight -/
theorem family_weight_ratio 
  (total_weight : ℝ) 
  (daughter_child_weight : ℝ) 
  (daughter_weight : ℝ) 
  (h1 : total_weight = 150) 
  (h2 : daughter_child_weight = 60) 
  (h3 : daughter_weight = 42) : 
  ∃ (child_weight grandmother_weight : ℝ), 
    total_weight = grandmother_weight + daughter_weight + child_weight ∧ 
    daughter_child_weight = daughter_weight + child_weight ∧
    child_weight / grandmother_weight = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_family_weight_ratio_l2861_286196


namespace NUMINAMATH_CALUDE_negative_three_less_than_negative_two_l2861_286112

theorem negative_three_less_than_negative_two : -3 < -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_less_than_negative_two_l2861_286112


namespace NUMINAMATH_CALUDE_range_of_expression_l2861_286151

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem range_of_expression (a b : ℕ) 
  (ha : isPrime a ∧ 49 < a ∧ a < 61) 
  (hb : isPrime b ∧ 59 < b ∧ b < 71) : 
  -297954 ≤ (a^2 : ℤ) - b^3 ∧ (a^2 : ℤ) - b^3 ≤ -223500 :=
sorry

end NUMINAMATH_CALUDE_range_of_expression_l2861_286151


namespace NUMINAMATH_CALUDE_cars_meeting_time_l2861_286111

/-- Two cars meeting on a highway -/
theorem cars_meeting_time 
  (highway_length : ℝ) 
  (car1_speed : ℝ) 
  (car2_speed : ℝ) 
  (h1 : highway_length = 45) 
  (h2 : car1_speed = 14) 
  (h3 : car2_speed = 16) : 
  (highway_length / (car1_speed + car2_speed)) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l2861_286111


namespace NUMINAMATH_CALUDE_remainder_nineteen_power_nineteen_plus_nineteen_mod_twenty_l2861_286161

theorem remainder_nineteen_power_nineteen_plus_nineteen_mod_twenty :
  (19^19 + 19) % 20 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_nineteen_power_nineteen_plus_nineteen_mod_twenty_l2861_286161


namespace NUMINAMATH_CALUDE_inequality_range_l2861_286117

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| - |x + 2| ≤ a) ↔ a ∈ Set.Ici 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l2861_286117


namespace NUMINAMATH_CALUDE_cylinder_radius_problem_l2861_286128

theorem cylinder_radius_problem (r : ℝ) :
  (r > 0) →
  (5 * π * (r + 4)^2 = 15 * π * r^2) →
  (r = 2 + 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_radius_problem_l2861_286128


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l2861_286108

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (invalid_percentage : ℚ) 
  (candidate_valid_votes : ℕ) 
  (h1 : total_votes = 560000) 
  (h2 : invalid_percentage = 15/100) 
  (h3 : candidate_valid_votes = 404600) : 
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 85/100 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l2861_286108


namespace NUMINAMATH_CALUDE_correct_sums_l2861_286135

theorem correct_sums (total : ℕ) (h1 : total = 75) : ∃ (right : ℕ), right * 3 = total ∧ right = 25 := by
  sorry

end NUMINAMATH_CALUDE_correct_sums_l2861_286135


namespace NUMINAMATH_CALUDE_find_a_find_m_range_l2861_286123

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

-- Part 1
theorem find_a : 
  (∀ x, f 1 x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧ 
  (∀ a, (∀ x, f a x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) → a = 1) :=
sorry

-- Part 2
theorem find_m_range : 
  ∀ m : ℝ, (∃ n : ℝ, f 1 n ≤ m - f 1 (-n)) ↔ m ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_find_a_find_m_range_l2861_286123


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l2861_286197

theorem coefficient_x_squared_in_binomial_expansion :
  (Finset.range 5).sum (fun k => (Nat.choose 4 k) * (1^(4-k)) * (1^k)) = 16 ∧
  (Nat.choose 4 2) = 6 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l2861_286197


namespace NUMINAMATH_CALUDE_four_digit_divisor_characterization_l2861_286183

/-- Represents a four-digit number in decimal notation -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_nonzero : a ≠ 0
  b_digit : b < 10
  c_digit : c < 10
  d_digit : d < 10

/-- Converts a FourDigitNumber to its decimal value -/
def to_decimal (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- Checks if one FourDigitNumber divides another -/
def divides (m n : FourDigitNumber) : Prop :=
  ∃ k : Nat, k * (to_decimal m) = to_decimal n

/-- Main theorem: Characterization of four-digit numbers that divide their rotations -/
theorem four_digit_divisor_characterization (n : FourDigitNumber) :
  (divides n {a := n.b, b := n.c, c := n.d, d := n.a, 
              a_nonzero := sorry, b_digit := n.c_digit, c_digit := n.d_digit, d_digit := sorry}) ∨
  (divides n {a := n.c, b := n.d, c := n.a, d := n.b, 
              a_nonzero := sorry, b_digit := n.d_digit, c_digit := sorry, d_digit := n.b_digit}) ∨
  (divides n {a := n.d, b := n.a, c := n.b, d := n.c, 
              a_nonzero := sorry, b_digit := sorry, c_digit := n.b_digit, d_digit := n.c_digit})
  ↔
  n.a = n.c ∧ n.b = n.d ∧ n.b ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_four_digit_divisor_characterization_l2861_286183


namespace NUMINAMATH_CALUDE_student_count_problem_l2861_286157

theorem student_count_problem (A B : ℕ) : 
  A = (5 : ℕ) * B / (7 : ℕ) →
  A + 3 = (4 : ℕ) * (B - 3) / (5 : ℕ) →
  A = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_student_count_problem_l2861_286157


namespace NUMINAMATH_CALUDE_parentheses_removal_l2861_286193

theorem parentheses_removal (a b : ℝ) : a + (5 * a - 3 * b) = 6 * a - 3 * b := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_l2861_286193


namespace NUMINAMATH_CALUDE_zero_product_implies_zero_factor_l2861_286138

theorem zero_product_implies_zero_factor (x y : ℝ) : 
  x * y = 0 → x = 0 ∨ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_product_implies_zero_factor_l2861_286138


namespace NUMINAMATH_CALUDE_square_field_dimensions_l2861_286136

/-- Proves that a square field with the given fence properties has a side length of 16000 meters -/
theorem square_field_dimensions (x : ℝ) : 
  x > 0 ∧ 
  (1.6 * x = x^2 / 10000) → 
  x = 16000 := by
sorry

end NUMINAMATH_CALUDE_square_field_dimensions_l2861_286136


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l2861_286140

theorem square_rectangle_area_relation : 
  ∃ (x₁ x₂ : ℝ), 
    (∀ x : ℝ, 3 * (x - 4)^2 = (x - 5) * (x + 6) ↔ x = x₁ ∨ x = x₂) ∧ 
    x₁ + x₂ = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l2861_286140


namespace NUMINAMATH_CALUDE_ternary_121_equals_16_l2861_286187

/-- Converts a ternary (base 3) number to decimal (base 10) --/
def ternary_to_decimal (a b c : ℕ) : ℕ :=
  a * 3^2 + b * 3^1 + c * 3^0

/-- The ternary number 121₃ is equal to 16 in decimal (base 10) --/
theorem ternary_121_equals_16 : ternary_to_decimal 1 2 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ternary_121_equals_16_l2861_286187


namespace NUMINAMATH_CALUDE_range_of_a_l2861_286125

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a^2 + 1}
def B (a : ℝ) : Set ℝ := {x | x^2 - 3*(a+1)*x + 2*(3*a+1) ≤ 0}

-- Define the condition that p is sufficient for q
def p_sufficient_for_q (a : ℝ) : Prop := A a ⊆ B a

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, p_sufficient_for_q a ↔ (1 ≤ a ∧ a ≤ 3) ∨ a = -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2861_286125


namespace NUMINAMATH_CALUDE_inverse_composition_f_inv_of_f_inv_of_f_inv_4_l2861_286122

def f : ℕ → ℕ
| 1 => 4
| 2 => 6
| 3 => 2
| 4 => 5
| 5 => 3
| 6 => 1
| _ => 0  -- Default case for completeness

-- Assumption that f is invertible
axiom f_invertible : Function.Injective f

-- Define f_inv as the inverse of f
noncomputable def f_inv : ℕ → ℕ := Function.invFun f

theorem inverse_composition (n : ℕ) : f_inv (f n) = n :=
  sorry

theorem f_inv_of_f_inv_of_f_inv_4 : f_inv (f_inv (f_inv 4)) = 2 :=
  sorry

end NUMINAMATH_CALUDE_inverse_composition_f_inv_of_f_inv_of_f_inv_4_l2861_286122


namespace NUMINAMATH_CALUDE_diploma_monthly_pay_l2861_286107

/-- The annual salary of a person with a degree -/
def annual_salary_degree : ℕ := 144000

/-- The ratio of salary between a person with a degree and a diploma holder -/
def salary_ratio : ℕ := 3

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- The monthly pay for a person holding a diploma certificate -/
def monthly_pay_diploma : ℚ := annual_salary_degree / (salary_ratio * months_per_year)

theorem diploma_monthly_pay :
  monthly_pay_diploma = 4000 := by sorry

end NUMINAMATH_CALUDE_diploma_monthly_pay_l2861_286107


namespace NUMINAMATH_CALUDE_fence_area_inequality_l2861_286198

theorem fence_area_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_fence_area_inequality_l2861_286198


namespace NUMINAMATH_CALUDE_students_between_minyoung_and_hoseok_l2861_286144

/-- Given 13 students in a line, with Minyoung at the 8th position from the left
    and Hoseok at the 9th position from the right, prove that the number of
    students between Minyoung and Hoseok is 2. -/
theorem students_between_minyoung_and_hoseok :
  let total_students : ℕ := 13
  let minyoung_position : ℕ := 8
  let hoseok_position_from_right : ℕ := 9
  let hoseok_position : ℕ := total_students - hoseok_position_from_right + 1
  (minyoung_position - hoseok_position - 1 : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_students_between_minyoung_and_hoseok_l2861_286144


namespace NUMINAMATH_CALUDE_advanced_purchase_tickets_l2861_286177

/-- Proves that the number of advanced-purchase tickets sold is 40 --/
theorem advanced_purchase_tickets (total_tickets : ℕ) (total_amount : ℕ) 
  (advanced_price : ℕ) (door_price : ℕ) (h1 : total_tickets = 140) 
  (h2 : total_amount = 1720) (h3 : advanced_price = 8) (h4 : door_price = 14) :
  ∃ (advanced_tickets : ℕ) (door_tickets : ℕ),
    advanced_tickets + door_tickets = total_tickets ∧
    advanced_price * advanced_tickets + door_price * door_tickets = total_amount ∧
    advanced_tickets = 40 := by
  sorry

end NUMINAMATH_CALUDE_advanced_purchase_tickets_l2861_286177


namespace NUMINAMATH_CALUDE_data_median_and_variance_l2861_286139

def data : List ℝ := [2, 3, 3, 3, 6, 6, 4, 5]

def median (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem data_median_and_variance :
  median data = 3.5 ∧ variance data = 2 := by sorry

end NUMINAMATH_CALUDE_data_median_and_variance_l2861_286139


namespace NUMINAMATH_CALUDE_some_number_added_l2861_286105

theorem some_number_added (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁ + a)^2 / (3 * x₁ + 65) = 2 ∧ 
                (x₂ + a)^2 / (3 * x₂ + 65) = 2 ∧ 
                |x₁ - x₂| = 22) → 
  a = 3 := by sorry

end NUMINAMATH_CALUDE_some_number_added_l2861_286105


namespace NUMINAMATH_CALUDE_quadratic_roots_distinct_l2861_286150

theorem quadratic_roots_distinct (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 - (k+3)*x₁ + k = 0) ∧ 
  (x₂^2 - (k+3)*x₂ + k = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_distinct_l2861_286150


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2861_286104

theorem purely_imaginary_complex_number (m : ℝ) : 
  (m^2 - 3*m = 0) ∧ (m^2 - 5*m + 6 ≠ 0) → m = 0 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2861_286104


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l2861_286153

/-- The quadratic function -/
def f (x : ℝ) : ℝ := -x^2 + 6*x + 3

/-- The x-coordinate of the vertex -/
def h : ℝ := 3

/-- The y-coordinate of the vertex -/
def k : ℝ := 12

/-- Theorem: The vertex of the quadratic function f(x) = -x^2 + 6x + 3 is at (3, 12) -/
theorem vertex_of_quadratic :
  (∀ x : ℝ, f x = -(x - h)^2 + k) ∧ f h = k :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l2861_286153


namespace NUMINAMATH_CALUDE_longest_diagonal_path_in_5x8_rectangle_l2861_286114

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a diagonal path within a rectangle -/
structure DiagonalPath where
  rectangle : Rectangle
  num_diagonals : ℕ

/-- Checks if a path is valid according to the problem constraints -/
def is_valid_path (path : DiagonalPath) : Prop :=
  path.num_diagonals > 0 ∧
  path.num_diagonals ≤ path.rectangle.width * path.rectangle.height ∧
  path.num_diagonals % 2 = 0  -- Closed path must have even number of diagonals

/-- Theorem stating the maximum number of diagonals in the longest path -/
theorem longest_diagonal_path_in_5x8_rectangle :
  ∃ (path : DiagonalPath),
    path.rectangle.width = 5 ∧
    path.rectangle.height = 8 ∧
    is_valid_path path ∧
    path.num_diagonals = 24 ∧
    ∀ (other_path : DiagonalPath),
      other_path.rectangle.width = 5 ∧
      other_path.rectangle.height = 8 ∧
      is_valid_path other_path →
      other_path.num_diagonals ≤ path.num_diagonals :=
by
  sorry

end NUMINAMATH_CALUDE_longest_diagonal_path_in_5x8_rectangle_l2861_286114


namespace NUMINAMATH_CALUDE_exam_average_score_l2861_286160

/-- Given an exam with a maximum score and the percentages scored by three students,
    calculate the average mark scored by all three students. -/
theorem exam_average_score (max_score : ℕ) (amar_percent bhavan_percent chetan_percent : ℕ) :
  max_score = 900 ∧ amar_percent = 64 ∧ bhavan_percent = 36 ∧ chetan_percent = 44 →
  (amar_percent * max_score / 100 + bhavan_percent * max_score / 100 + chetan_percent * max_score / 100) / 3 = 432 :=
by sorry

end NUMINAMATH_CALUDE_exam_average_score_l2861_286160


namespace NUMINAMATH_CALUDE_walters_money_percentage_l2861_286173

/-- The value of a penny in cents -/
def penny : ℕ := 1

/-- The value of a nickel in cents -/
def nickel : ℕ := 5

/-- The value of a dime in cents -/
def dime : ℕ := 10

/-- The value of a quarter in cents -/
def quarter : ℕ := 25

/-- The total number of cents in Walter's pocket -/
def walters_money : ℕ := penny + 2 * nickel + dime + 2 * quarter

/-- Theorem: Walter's money is 71% of a dollar -/
theorem walters_money_percentage :
  (walters_money : ℚ) / 100 = 71 / 100 := by sorry

end NUMINAMATH_CALUDE_walters_money_percentage_l2861_286173


namespace NUMINAMATH_CALUDE_max_daily_revenue_l2861_286134

def P (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 70
  else 0

def Q (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

def dailyRevenue (t : ℕ) : ℝ := P t * Q t

theorem max_daily_revenue :
  (∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ dailyRevenue t = 1125) ∧
  (∀ t : ℕ, 0 < t ∧ t ≤ 30 → dailyRevenue t ≤ 1125) ∧
  (∀ t : ℕ, 0 < t ∧ t ≤ 30 ∧ dailyRevenue t = 1125 → t = 25) :=
by sorry

end NUMINAMATH_CALUDE_max_daily_revenue_l2861_286134


namespace NUMINAMATH_CALUDE_min_value_a_l2861_286171

theorem min_value_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (a - 1) * x^2 - 2 * Real.sqrt 2 * x * y + a * y^2 ≥ 0) →
  a ≥ 2 ∧ ∀ b : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → (b - 1) * x^2 - 2 * Real.sqrt 2 * x * y + b * y^2 ≥ 0) → b ≥ a :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l2861_286171


namespace NUMINAMATH_CALUDE_T_formula_l2861_286156

def T : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 6
  | (n + 3) => (n + 7) * T (n + 2) - 4 * (n + 3) * T (n + 1) + (4 * (n + 3) - 8) * T n

theorem T_formula (n : ℕ) : T n = n.factorial + 2^n := by
  sorry

end NUMINAMATH_CALUDE_T_formula_l2861_286156


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l2861_286158

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, 3^x + x < 0) ↔ (∀ x : ℝ, 3^x + x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l2861_286158


namespace NUMINAMATH_CALUDE_ab_value_l2861_286131

theorem ab_value (a b : ℝ) (h1 : (a + b)^2 = 4) (h2 : (a - b)^2 = 3) : a * b = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2861_286131


namespace NUMINAMATH_CALUDE_function_defined_on_reals_l2861_286132

/-- The function f(x) = (x^2 - 2)/(x^2 + 1) is defined for all real numbers x. -/
theorem function_defined_on_reals : ∀ x : ℝ, ∃ y : ℝ, y = (x^2 - 2)/(x^2 + 1) :=
sorry

end NUMINAMATH_CALUDE_function_defined_on_reals_l2861_286132


namespace NUMINAMATH_CALUDE_parabola_point_focus_distance_l2861_286189

/-- Theorem: Distance between a point on a parabola and its focus
Given a parabola y^2 = 16x with focus F at (4, 0), and a point P on the parabola
that is 12 units away from the x-axis, the distance between P and F is 13 units. -/
theorem parabola_point_focus_distance
  (P : ℝ × ℝ) -- Point P on the parabola
  (h_on_parabola : (P.2)^2 = 16 * P.1) -- P satisfies the parabola equation
  (h_distance_from_x_axis : abs P.2 = 12) -- P is 12 units from x-axis
  : Real.sqrt ((P.1 - 4)^2 + P.2^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_focus_distance_l2861_286189


namespace NUMINAMATH_CALUDE_cantaloupes_sum_l2861_286154

/-- The number of cantaloupes grown by Fred -/
def fred_cantaloupes : ℕ := 38

/-- The number of cantaloupes grown by Tim -/
def tim_cantaloupes : ℕ := 44

/-- The total number of cantaloupes grown by Fred and Tim -/
def total_cantaloupes : ℕ := fred_cantaloupes + tim_cantaloupes

theorem cantaloupes_sum : total_cantaloupes = 82 := by
  sorry

end NUMINAMATH_CALUDE_cantaloupes_sum_l2861_286154


namespace NUMINAMATH_CALUDE_jackson_points_l2861_286159

theorem jackson_points (total_points : ℕ) (num_players : ℕ) (other_players : ℕ) (avg_points : ℕ) :
  total_points = 75 →
  num_players = 8 →
  other_players = 7 →
  avg_points = 6 →
  total_points - (other_players * avg_points) = 33 :=
by sorry

end NUMINAMATH_CALUDE_jackson_points_l2861_286159


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2861_286142

def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2861_286142


namespace NUMINAMATH_CALUDE_melissa_score_l2861_286180

/-- Calculates the total score for a player given points per game and number of games played -/
def totalScore (pointsPerGame : ℕ) (numGames : ℕ) : ℕ :=
  pointsPerGame * numGames

/-- Proves that a player scoring 7 points per game for 3 games has a total score of 21 points -/
theorem melissa_score : totalScore 7 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_melissa_score_l2861_286180


namespace NUMINAMATH_CALUDE_fraction_sum_equals_three_tenths_l2861_286133

theorem fraction_sum_equals_three_tenths : 
  (1 : ℚ) / 10 + (2 : ℚ) / 20 + (3 : ℚ) / 30 = (3 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_three_tenths_l2861_286133


namespace NUMINAMATH_CALUDE_blue_pens_count_l2861_286143

/-- Given a total number of pens and a number of black pens, 
    calculate the number of blue pens. -/
def blue_pens (total : ℕ) (black : ℕ) : ℕ :=
  total - black

/-- Theorem: When the total number of pens is 8 and the number of black pens is 4,
    the number of blue pens is 4. -/
theorem blue_pens_count : blue_pens 8 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_blue_pens_count_l2861_286143


namespace NUMINAMATH_CALUDE_piggy_bank_theorem_l2861_286188

/-- The value of a piggy bank containing dimes and quarters -/
def piggy_bank_value (num_dimes num_quarters : ℕ) (dime_value quarter_value : ℚ) : ℚ :=
  (num_dimes : ℚ) * dime_value + (num_quarters : ℚ) * quarter_value

/-- Theorem: The value of a piggy bank with 35 dimes and 65 quarters is $19.75 -/
theorem piggy_bank_theorem :
  piggy_bank_value 35 65 (10 / 100) (25 / 100) = 1975 / 100 := by
  sorry

#eval piggy_bank_value 35 65 (10 / 100) (25 / 100)

end NUMINAMATH_CALUDE_piggy_bank_theorem_l2861_286188


namespace NUMINAMATH_CALUDE_apples_used_correct_l2861_286182

/-- The number of apples used to make lunch in the school cafeteria -/
def apples_used : ℕ := 20

/-- The initial number of apples in the cafeteria -/
def initial_apples : ℕ := 23

/-- The number of apples bought after making lunch -/
def apples_bought : ℕ := 6

/-- The final number of apples in the cafeteria -/
def final_apples : ℕ := 9

/-- Theorem stating that the number of apples used for lunch is correct -/
theorem apples_used_correct : 
  initial_apples - apples_used + apples_bought = final_apples :=
by sorry

end NUMINAMATH_CALUDE_apples_used_correct_l2861_286182


namespace NUMINAMATH_CALUDE_park_short_trees_after_planting_l2861_286185

/-- The number of short trees in the park after planting -/
def total_short_trees (initial_short_trees newly_planted_short_trees : ℕ) : ℕ :=
  initial_short_trees + newly_planted_short_trees

/-- Theorem stating that the total number of short trees after planting is 98 -/
theorem park_short_trees_after_planting :
  total_short_trees 41 57 = 98 := by
  sorry


end NUMINAMATH_CALUDE_park_short_trees_after_planting_l2861_286185


namespace NUMINAMATH_CALUDE_profit_share_difference_is_1000_l2861_286127

/-- Represents the profit share calculation for a business partnership --/
structure BusinessPartnership where
  investment_a : ℕ
  investment_b : ℕ
  investment_c : ℕ
  profit_share_b : ℕ

/-- Calculates the difference between profit shares of partners C and A --/
def profit_share_difference (bp : BusinessPartnership) : ℕ :=
  let total_investment := bp.investment_a + bp.investment_b + bp.investment_c
  let total_profit := bp.profit_share_b * total_investment / bp.investment_b
  let share_a := total_profit * bp.investment_a / total_investment
  let share_c := total_profit * bp.investment_c / total_investment
  share_c - share_a

/-- Theorem stating that for the given investments and B's profit share, 
    the difference between C's and A's profit shares is 1000 --/
theorem profit_share_difference_is_1000 :
  profit_share_difference ⟨8000, 10000, 12000, 2500⟩ = 1000 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_difference_is_1000_l2861_286127


namespace NUMINAMATH_CALUDE_max_visible_cubes_is_274_l2861_286176

/-- The dimension of the cube --/
def n : ℕ := 10

/-- The total number of unit cubes in the cube --/
def total_cubes : ℕ := n^3

/-- The number of unit cubes on one face of the cube --/
def face_cubes : ℕ := n^2

/-- The number of visible faces from a corner --/
def visible_faces : ℕ := 3

/-- The number of shared edges between visible faces --/
def shared_edges : ℕ := 3

/-- The length of each edge --/
def edge_length : ℕ := n

/-- The number of unit cubes along a shared edge, excluding the corner --/
def edge_cubes : ℕ := edge_length - 1

/-- The maximum number of visible unit cubes from a single point --/
def max_visible_cubes : ℕ := visible_faces * face_cubes - shared_edges * edge_cubes + 1

theorem max_visible_cubes_is_274 : max_visible_cubes = 274 := by
  sorry

end NUMINAMATH_CALUDE_max_visible_cubes_is_274_l2861_286176


namespace NUMINAMATH_CALUDE_sum_of_powers_l2861_286141

theorem sum_of_powers (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^2013 + (y / (x + y))^2013 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l2861_286141


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l2861_286174

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo 0 2, f' x < 0 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l2861_286174


namespace NUMINAMATH_CALUDE_opposite_of_neg_abs_two_thirds_l2861_286116

theorem opposite_of_neg_abs_two_thirds (m : ℚ) : 
  m = -(-(|-(2/3)|)) → m = 2/3 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_neg_abs_two_thirds_l2861_286116


namespace NUMINAMATH_CALUDE_golden_ratio_between_consecutive_integers_l2861_286190

theorem golden_ratio_between_consecutive_integers :
  ∃ (a b : ℤ), (a + 1 = b) ∧ (a < (Real.sqrt 5 + 1) / 2) ∧ ((Real.sqrt 5 + 1) / 2 < b) → a + b = 3 :=
by sorry

end NUMINAMATH_CALUDE_golden_ratio_between_consecutive_integers_l2861_286190


namespace NUMINAMATH_CALUDE_ellen_stuffing_time_l2861_286115

/-- Earl's envelope stuffing rate in envelopes per minute -/
def earl_rate : ℝ := 36

/-- Time taken by Earl and Ellen together to stuff 180 envelopes in minutes -/
def combined_time : ℝ := 3

/-- Number of envelopes stuffed by Earl and Ellen together -/
def combined_envelopes : ℝ := 180

/-- Ellen's time to stuff the same number of envelopes as Earl in minutes -/
def ellen_time : ℝ := 1.5

theorem ellen_stuffing_time :
  earl_rate * ellen_time + earl_rate = combined_envelopes / combined_time :=
by sorry

end NUMINAMATH_CALUDE_ellen_stuffing_time_l2861_286115


namespace NUMINAMATH_CALUDE_remaining_requests_after_two_weeks_l2861_286137

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the number of weekdays in a week -/
def weekdaysInWeek : ℕ := 5

/-- Represents the number of weekend days in a week -/
def weekendDaysInWeek : ℕ := daysInWeek - weekdaysInWeek

/-- Represents the number of requests Maia gets on a weekday -/
def weekdayRequests : ℕ := 8

/-- Represents the number of requests Maia gets on a weekend day -/
def weekendRequests : ℕ := 5

/-- Represents the number of requests Maia works on each day (except Sunday) -/
def requestsWorkedPerDay : ℕ := 4

/-- Represents the number of weeks we're considering -/
def numberOfWeeks : ℕ := 2

/-- Represents the number of days Maia works in a week -/
def workDaysPerWeek : ℕ := daysInWeek - 1

theorem remaining_requests_after_two_weeks : 
  (weekdayRequests * weekdaysInWeek + weekendRequests * weekendDaysInWeek) * numberOfWeeks - 
  (requestsWorkedPerDay * workDaysPerWeek) * numberOfWeeks = 52 := by
  sorry

end NUMINAMATH_CALUDE_remaining_requests_after_two_weeks_l2861_286137


namespace NUMINAMATH_CALUDE_two_machines_half_hour_copies_l2861_286109

/-- Represents a copy machine with a constant copying rate. -/
structure CopyMachine where
  copies_per_minute : ℕ

/-- Calculates the total number of copies made by two machines in a given time. -/
def total_copies (machine1 machine2 : CopyMachine) (minutes : ℕ) : ℕ :=
  (machine1.copies_per_minute + machine2.copies_per_minute) * minutes

/-- Theorem stating that two specific copy machines working together for 30 minutes will produce 2850 copies. -/
theorem two_machines_half_hour_copies :
  let machine1 : CopyMachine := ⟨40⟩
  let machine2 : CopyMachine := ⟨55⟩
  total_copies machine1 machine2 30 = 2850 := by
  sorry


end NUMINAMATH_CALUDE_two_machines_half_hour_copies_l2861_286109


namespace NUMINAMATH_CALUDE_football_player_goals_l2861_286120

theorem football_player_goals (average_increase : ℝ) (fifth_match_goals : ℕ) : 
  average_increase = 0.3 →
  fifth_match_goals = 2 →
  ∃ (initial_average : ℝ),
    initial_average * 4 + fifth_match_goals = (initial_average + average_increase) * 5 ∧
    (initial_average + average_increase) * 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_football_player_goals_l2861_286120


namespace NUMINAMATH_CALUDE_decimal_25_to_binary_l2861_286164

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

theorem decimal_25_to_binary :
  decimal_to_binary 25 = [true, false, false, true, true] := by
sorry

end NUMINAMATH_CALUDE_decimal_25_to_binary_l2861_286164


namespace NUMINAMATH_CALUDE_alex_jamie_pairing_probability_l2861_286146

/-- The number of students participating in the event -/
def total_students : ℕ := 32

/-- The probability of Alex being paired with Jamie in a random pairing -/
def probability_alex_jamie : ℚ := 1 / 31

/-- Theorem stating that the probability of Alex being paired with Jamie
    in a random pairing of 32 students is 1/31 -/
theorem alex_jamie_pairing_probability :
  probability_alex_jamie = 1 / (total_students - 1) :=
sorry

end NUMINAMATH_CALUDE_alex_jamie_pairing_probability_l2861_286146


namespace NUMINAMATH_CALUDE_tram_speed_l2861_286163

/-- The speed of trams given pedestrian speed and relative speed ratios -/
theorem tram_speed (pedestrian_speed : ℝ) (approaching_ratio : ℝ) (overtaking_ratio : ℝ) :
  pedestrian_speed = 5 →
  approaching_ratio = 600 →
  overtaking_ratio = 225 →
  ∃ V : ℝ, V > pedestrian_speed ∧
    (V + pedestrian_speed) / (V - pedestrian_speed) = approaching_ratio / overtaking_ratio ∧
    V = 11 :=
by sorry

end NUMINAMATH_CALUDE_tram_speed_l2861_286163


namespace NUMINAMATH_CALUDE_convex_polyhedron_same_edge_count_l2861_286165

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  faces : ℕ
  max_edges : ℕ
  faces_ge_max_edges : faces ≥ max_edges
  min_edges_per_face : max_edges ≥ 3

/-- Theorem: A convex polyhedron always has two faces with the same number of edges -/
theorem convex_polyhedron_same_edge_count (P : ConvexPolyhedron) :
  ∃ (e : ℕ) (f₁ f₂ : ℕ), f₁ ≠ f₂ ∧ f₁ ≤ P.faces ∧ f₂ ≤ P.faces ∧
  (∃ (edges_of_face : ℕ → ℕ), 
    (∀ f, f ≤ P.faces → 3 ≤ edges_of_face f ∧ edges_of_face f ≤ P.max_edges) ∧
    edges_of_face f₁ = e ∧ edges_of_face f₂ = e) :=
sorry

end NUMINAMATH_CALUDE_convex_polyhedron_same_edge_count_l2861_286165


namespace NUMINAMATH_CALUDE_highest_power_of_seven_in_100_factorial_l2861_286184

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem highest_power_of_seven_in_100_factorial :
  ∃ (k : ℕ), factorial 100 % (7^16) = 0 ∧ factorial 100 % (7^17) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_highest_power_of_seven_in_100_factorial_l2861_286184


namespace NUMINAMATH_CALUDE_percentage_equation_l2861_286179

theorem percentage_equation (x : ℝ) : (0.3 / 100) * x = 0.15 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_l2861_286179


namespace NUMINAMATH_CALUDE_largest_crate_dimension_l2861_286192

def crate_width : ℝ := 5
def crate_length : ℝ := 8
def pillar_radius : ℝ := 5

theorem largest_crate_dimension (height : ℝ) :
  height ≥ 2 * pillar_radius →
  crate_width ≥ 2 * pillar_radius →
  crate_length ≥ 2 * pillar_radius →
  (∃ (max_dim : ℝ), max_dim = max height (max crate_width crate_length) ∧ max_dim = 2 * pillar_radius) :=
by sorry

end NUMINAMATH_CALUDE_largest_crate_dimension_l2861_286192


namespace NUMINAMATH_CALUDE_largest_positive_integer_satisfying_condition_l2861_286167

def binary_op (n : ℤ) : ℤ := n - (n * 5)

theorem largest_positive_integer_satisfying_condition :
  ∀ n : ℤ, n > 0 → binary_op n < -15 → n ≤ 4 ∧
  binary_op 4 < -15 ∧
  ∀ m : ℤ, m > 4 → binary_op m ≥ -15 := by
sorry

end NUMINAMATH_CALUDE_largest_positive_integer_satisfying_condition_l2861_286167


namespace NUMINAMATH_CALUDE_unusual_coin_probability_l2861_286102

theorem unusual_coin_probability (p q : ℝ) : 
  0 ≤ p ∧ 0 ≤ q ∧ q ≤ p ∧ p + q + 1/6 = 1 ∧ 
  p^2 + q^2 + (1/6)^2 = 1/2 → 
  p = 2/3 := by sorry

end NUMINAMATH_CALUDE_unusual_coin_probability_l2861_286102


namespace NUMINAMATH_CALUDE_marks_reading_time_marks_reading_proof_l2861_286195

theorem marks_reading_time (increase : ℕ) (target : ℕ) (days_in_week : ℕ) : ℕ :=
  let initial_daily_hours : ℕ := (target - increase) / days_in_week
  initial_daily_hours

theorem marks_reading_proof :
  marks_reading_time 4 18 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_marks_reading_time_marks_reading_proof_l2861_286195


namespace NUMINAMATH_CALUDE_oak_grove_library_books_l2861_286126

/-- The number of books in Oak Grove's public library -/
def public_library_books : ℕ := 1986

/-- The total number of books in all Oak Grove libraries -/
def total_books : ℕ := 7092

/-- The number of books in Oak Grove's school libraries -/
def school_library_books : ℕ := total_books - public_library_books

theorem oak_grove_library_books : school_library_books = 5106 := by
  sorry

end NUMINAMATH_CALUDE_oak_grove_library_books_l2861_286126


namespace NUMINAMATH_CALUDE_complex_imaginary_part_l2861_286166

theorem complex_imaginary_part (a : ℝ) :
  let z : ℂ := (1 - a * Complex.I) / (1 + Complex.I)
  (z.re = -1) → (z.im = -2) := by
  sorry

end NUMINAMATH_CALUDE_complex_imaginary_part_l2861_286166
