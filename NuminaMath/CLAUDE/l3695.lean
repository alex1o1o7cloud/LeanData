import Mathlib

namespace quadratic_condition_l3695_369523

theorem quadratic_condition (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ (m^2 - 1) * x^2 + x + m = a * x^2 + b * x + c) ↔ 
  (m ≠ 1 ∧ m ≠ -1) :=
by sorry

end quadratic_condition_l3695_369523


namespace fraction_division_l3695_369511

theorem fraction_division (x : ℚ) : 
  (37 + 1/2 : ℚ) = 450 * x → x = 1/12 ∧ (37 + 1/2 : ℚ) / x = 450 := by
  sorry

end fraction_division_l3695_369511


namespace man_ownership_fraction_l3695_369509

/-- Proves that the fraction of the business the man owns is 2/3, given the conditions -/
theorem man_ownership_fraction (sold_fraction : ℚ) (sold_value : ℕ) (total_value : ℕ) 
  (h1 : sold_fraction = 3 / 4)
  (h2 : sold_value = 45000)
  (h3 : total_value = 90000) :
  ∃ (x : ℚ), x * sold_fraction * total_value = sold_value ∧ x = 2 / 3 := by
  sorry

#check man_ownership_fraction

end man_ownership_fraction_l3695_369509


namespace total_ages_is_56_l3695_369521

/-- Given Craig's age and the age difference with his mother, calculate the total of their ages -/
def total_ages (craig_age : ℕ) (age_difference : ℕ) : ℕ :=
  craig_age + (craig_age + age_difference)

/-- Theorem: The total of Craig and his mother's ages is 56 years -/
theorem total_ages_is_56 : total_ages 16 24 = 56 := by
  sorry

end total_ages_is_56_l3695_369521


namespace sqrt_equation_l3695_369560

theorem sqrt_equation (n : ℝ) : Real.sqrt (8 + n) = 9 → n = 73 := by
  sorry

end sqrt_equation_l3695_369560


namespace geometric_sequence_problem_l3695_369559

-- Define a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geometric : is_geometric a) 
  (h_product : a 2 * a 10 = 4)
  (h_sum_positive : a 2 + a 10 > 0) :
  a 6 = 2 := by
  sorry

end geometric_sequence_problem_l3695_369559


namespace simplest_proper_fraction_with_7_numerator_simplest_improper_fraction_with_7_denominator_l3695_369569

-- Define a function to check if a fraction is in its simplest form
def isSimplestForm (n d : ℕ) : Prop :=
  n.gcd d = 1

-- Define a function to check if a fraction is proper
def isProper (n d : ℕ) : Prop :=
  n < d

-- Define a function to check if a fraction is improper
def isImproper (n d : ℕ) : Prop :=
  n ≥ d

-- Theorem for the simplest proper fraction with 7 as numerator
theorem simplest_proper_fraction_with_7_numerator :
  isSimplestForm 7 8 ∧ isProper 7 8 ∧
  ∀ d : ℕ, d > 7 → isSimplestForm 7 d → d ≥ 8 :=
sorry

-- Theorem for the simplest improper fraction with 7 as denominator
theorem simplest_improper_fraction_with_7_denominator :
  isSimplestForm 8 7 ∧ isImproper 8 7 ∧
  ∀ n : ℕ, n > 7 → isSimplestForm n 7 → n ≥ 8 :=
sorry

end simplest_proper_fraction_with_7_numerator_simplest_improper_fraction_with_7_denominator_l3695_369569


namespace books_loaned_out_l3695_369514

/-- Proves the number of books loaned out given initial and final book counts and return rate -/
theorem books_loaned_out 
  (initial_books : ℕ) 
  (final_books : ℕ) 
  (return_rate : ℚ) 
  (h1 : initial_books = 150)
  (h2 : final_books = 122)
  (h3 : return_rate = 65 / 100) :
  ∃ (loaned_books : ℕ), 
    (initial_books : ℚ) - (loaned_books : ℚ) * (1 - return_rate) = final_books ∧ 
    loaned_books = 80 := by
  sorry

end books_loaned_out_l3695_369514


namespace range_of_a_l3695_369580

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 > 0) → -1 < a ∧ a < 3 := by
  sorry

end range_of_a_l3695_369580


namespace fair_coin_probability_l3695_369563

def n : ℕ := 5
def k : ℕ := 2
def p : ℚ := 1/2

theorem fair_coin_probability : 
  (n.choose k) * p^k * (1 - p)^(n - k) = 10/32 := by sorry

end fair_coin_probability_l3695_369563


namespace multiply_three_negative_two_l3695_369582

theorem multiply_three_negative_two : 3 * (-2) = -6 := by
  sorry

end multiply_three_negative_two_l3695_369582


namespace initial_walking_speed_l3695_369578

/-- Proves that given a specific distance and time difference between two speeds,
    the initial speed is 11.25 kmph. -/
theorem initial_walking_speed 
  (distance : ℝ) 
  (time_diff : ℝ) 
  (faster_speed : ℝ) :
  distance = 9.999999999999998 →
  time_diff = 1/3 →
  faster_speed = 15 →
  ∃ (initial_speed : ℝ),
    distance / initial_speed - distance / faster_speed = time_diff ∧
    initial_speed = 11.25 := by
  sorry

#check initial_walking_speed

end initial_walking_speed_l3695_369578


namespace farm_animals_l3695_369588

theorem farm_animals (sheep ducks : ℕ) : 
  sheep + ducks = 15 → 
  4 * sheep + 2 * ducks = 22 + 2 * (sheep + ducks) → 
  sheep = 11 := by
sorry

end farm_animals_l3695_369588


namespace probability_third_key_opens_door_l3695_369534

/-- The probability of opening a door with the third key, given 5 keys with only one correct key --/
theorem probability_third_key_opens_door : 
  ∀ (n : ℕ) (p : ℝ),
    n = 5 →  -- There are 5 keys
    p = 1 / n →  -- The probability of selecting the correct key is 1/n
    p = 1 / 5  -- The probability of opening the door on the third attempt is 1/5
    := by sorry

end probability_third_key_opens_door_l3695_369534


namespace rationalize_denominator_l3695_369579

theorem rationalize_denominator :
  18 / (Real.sqrt 36 + Real.sqrt 2) = 54 / 17 - 9 * Real.sqrt 2 / 17 := by
sorry

end rationalize_denominator_l3695_369579


namespace x_plus_y_equals_negative_27_l3695_369572

theorem x_plus_y_equals_negative_27 (x y : ℤ) 
  (h1 : x + 1 = y - 8) 
  (h2 : x = 2 * y) : 
  x + y = -27 := by
  sorry

end x_plus_y_equals_negative_27_l3695_369572


namespace course_size_l3695_369529

theorem course_size (num_d : ℕ) (h_d : num_d = 25) :
  ∃ (total : ℕ),
    total > 0 ∧
    (total : ℚ) = num_d + (1/5 : ℚ) * total + (1/4 : ℚ) * total + (1/2 : ℚ) * total ∧
    total = 500 := by
  sorry

end course_size_l3695_369529


namespace kelsey_sister_age_difference_l3695_369538

/-- Represents the age difference between Kelsey and her older sister -/
def age_difference (kelsey_birth_year : ℕ) (sister_birth_year : ℕ) : ℕ :=
  kelsey_birth_year - sister_birth_year

theorem kelsey_sister_age_difference :
  ∀ (kelsey_birth_year : ℕ) (sister_birth_year : ℕ),
  kelsey_birth_year + 25 = 1999 →
  sister_birth_year + 50 = 2021 →
  age_difference kelsey_birth_year sister_birth_year = 3 := by
  sorry

end kelsey_sister_age_difference_l3695_369538


namespace remaining_card_theorem_l3695_369565

/-- Definition of the operation sequence on a stack of cards -/
def operationSequence (n : ℕ) : List ℕ :=
  sorry

/-- L(n) is the number on the remaining card after performing the operation sequence -/
def L (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating the form of k for which L(3k) = k -/
theorem remaining_card_theorem (k : ℕ) :
  (L (3 * k) = k) ↔ 
  (∃ j : ℕ, (k = (2 * 3^(6*j) - 2) / 7) ∨ (k = (3^(6*j + 2) - 2) / 7)) :=
by sorry

end remaining_card_theorem_l3695_369565


namespace area_not_covered_by_circles_l3695_369592

/-- The area of a square not covered by four inscribed circles -/
theorem area_not_covered_by_circles (square_side : ℝ) (circle_radius : ℝ) 
  (h1 : square_side = 10)
  (h2 : circle_radius = 5)
  (h3 : circle_radius * 2 = square_side) :
  square_side ^ 2 - 4 * Real.pi * circle_radius ^ 2 + 4 * Real.pi * circle_radius ^ 2 / 2 = 100 - 50 * Real.pi := by
  sorry

#check area_not_covered_by_circles

end area_not_covered_by_circles_l3695_369592


namespace green_pepper_weight_l3695_369537

def hannah_peppers (total_weight red_weight green_weight : Real) : Prop :=
  total_weight = 0.66 ∧ 
  red_weight = 0.33 ∧ 
  green_weight = total_weight - red_weight

theorem green_pepper_weight : 
  ∀ (total_weight red_weight green_weight : Real),
  hannah_peppers total_weight red_weight green_weight →
  green_weight = 0.33 :=
by
  sorry

end green_pepper_weight_l3695_369537


namespace vegan_soy_free_fraction_l3695_369517

theorem vegan_soy_free_fraction (total_dishes : ℕ) (vegan_dishes : ℕ) (soy_dishes : ℕ)
  (h1 : vegan_dishes = total_dishes / 4)
  (h2 : soy_dishes = 4 * vegan_dishes / 5)
  (h3 : vegan_dishes > 0)
  (h4 : total_dishes > 0) :
  (vegan_dishes - soy_dishes) / total_dishes = 1 / 20 :=
by sorry

end vegan_soy_free_fraction_l3695_369517


namespace elective_course_selection_l3695_369522

def category_A : ℕ := 3
def category_B : ℕ := 4
def total_courses : ℕ := 3

theorem elective_course_selection :
  (Nat.choose category_A 1 * Nat.choose category_B 2) +
  (Nat.choose category_A 2 * Nat.choose category_B 1) = 30 := by
  sorry

end elective_course_selection_l3695_369522


namespace polynomial_has_real_root_l3695_369598

/-- The polynomial in question -/
def polynomial (b x : ℝ) : ℝ := x^4 + b*x^3 - 2*x^2 + b*x + 2

/-- Theorem stating the condition for the polynomial to have at least one real root -/
theorem polynomial_has_real_root (b : ℝ) :
  (∃ x : ℝ, polynomial b x = 0) ↔ b ≤ 0 := by sorry

end polynomial_has_real_root_l3695_369598


namespace min_sum_products_l3695_369532

theorem min_sum_products (m n : ℕ) : 
  (m * (m - 1)) / ((m + n) * (m + n - 1)) = 1 / 2 →
  m ≥ 1 →
  n ≥ 1 →
  m + n ≥ 4 := by
  sorry

end min_sum_products_l3695_369532


namespace inequality_proof_l3695_369575

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) :
  2 * a * b * Real.log (b / a) < b^2 - a^2 := by
  sorry

end inequality_proof_l3695_369575


namespace midpoint_property_l3695_369557

/-- Given two points D and E in the plane, and F as their midpoint, 
    prove that 2x - 4y = 14 where F = (x, y) -/
theorem midpoint_property (D E F : ℝ × ℝ) : 
  D = (30, 10) →
  E = (6, 1) →
  F = ((D.1 + E.1) / 2, (D.2 + E.2) / 2) →
  2 * F.1 - 4 * F.2 = 14 := by
sorry

end midpoint_property_l3695_369557


namespace mary_flour_amount_l3695_369547

/-- The amount of flour Mary uses in her cake recipe -/
def flour_recipe : ℝ := 7.0

/-- The extra amount of flour Mary adds -/
def flour_extra : ℝ := 2.0

/-- The total amount of flour Mary uses -/
def flour_total : ℝ := flour_recipe + flour_extra

theorem mary_flour_amount : flour_total = 9.0 := by
  sorry

end mary_flour_amount_l3695_369547


namespace twins_age_problem_l3695_369554

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 13 → age = 6 := by sorry

end twins_age_problem_l3695_369554


namespace trajectory_equation_l3695_369520

/-- Given a fixed point A(1,2) and a moving point P(x,y), if the projection of vector OP on vector OA is -√5,
    then the equation x + 2y + 5 = 0 represents the trajectory of point P. -/
theorem trajectory_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (1, 2)
  let P : ℝ × ℝ := (x, y)
  let OA : ℝ × ℝ := A
  let OP : ℝ × ℝ := P
  (OP.1 * OA.1 + OP.2 * OA.2) / Real.sqrt (OA.1^2 + OA.2^2) = -Real.sqrt 5 →
  x + 2*y + 5 = 0 :=
by sorry

end trajectory_equation_l3695_369520


namespace bankers_discount_calculation_l3695_369597

/-- Calculates the banker's discount for a given period and rate -/
def bankers_discount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Theorem: The banker's discount for the given conditions is 18900 -/
theorem bankers_discount_calculation (principal : ℝ) 
  (rate1 rate2 rate3 : ℝ) (time1 time2 time3 : ℝ) :
  principal = 180000 ∧ 
  rate1 = 0.12 ∧ rate2 = 0.14 ∧ rate3 = 0.16 ∧
  time1 = 0.25 ∧ time2 = 0.25 ∧ time3 = 0.25 →
  bankers_discount principal rate1 time1 + 
  bankers_discount principal rate2 time2 + 
  bankers_discount principal rate3 time3 = 18900 := by
  sorry

#eval bankers_discount 180000 0.12 0.25 + 
      bankers_discount 180000 0.14 0.25 + 
      bankers_discount 180000 0.16 0.25

end bankers_discount_calculation_l3695_369597


namespace fruit_arrangement_count_l3695_369502

def number_of_arrangements (a o b g : ℕ) : ℕ :=
  Nat.factorial 14 / (Nat.factorial a * Nat.factorial o * Nat.factorial b * Nat.factorial g)

theorem fruit_arrangement_count :
  number_of_arrangements 4 3 3 4 = 4204200 :=
by
  sorry

#eval number_of_arrangements 4 3 3 4

end fruit_arrangement_count_l3695_369502


namespace distance_home_to_school_l3695_369550

theorem distance_home_to_school :
  ∀ (d : ℝ) (t : ℝ),
    d = 6 * (t + 7/60) ∧
    d = 12 * (t - 8/60) →
    d = 3 := by
  sorry

end distance_home_to_school_l3695_369550


namespace quadratic_one_positive_root_l3695_369568

theorem quadratic_one_positive_root (a : ℝ) : 
  (∃! x : ℝ, x > 0 ∧ x^2 - a*x + a - 2 = 0) → a ≤ 2 := by
  sorry

end quadratic_one_positive_root_l3695_369568


namespace doubled_added_tripled_l3695_369546

theorem doubled_added_tripled (y : ℝ) : 3 * (2 * 7 + y) = 69 → y = 9 := by
  sorry

end doubled_added_tripled_l3695_369546


namespace round_trip_with_car_percentage_l3695_369589

/-- The percentage of passengers with round-trip tickets who did not take their cars -/
def no_car_percentage : ℝ := 60

/-- The percentage of all passengers who held round-trip tickets -/
def round_trip_percentage : ℝ := 62.5

/-- The theorem to prove -/
theorem round_trip_with_car_percentage :
  (100 - no_car_percentage) * round_trip_percentage / 100 = 25 := by
  sorry

end round_trip_with_car_percentage_l3695_369589


namespace min_distance_log_circle_l3695_369551

theorem min_distance_log_circle (e : ℝ) (h : e > 0) :
  let f := fun x : ℝ => Real.log x
  let circle := fun (x y : ℝ) => (x - (e + 1/e))^2 + y^2 = 1/4
  ∃ (min_dist : ℝ),
    (∀ (x₁ y₁ x₂ y₂ : ℝ), f x₁ = y₁ → circle x₂ y₂ →
      min_dist ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)) ∧
    min_dist = (2 * Real.sqrt (e^2 + 1) - e) / (2 * e) :=
by sorry

end min_distance_log_circle_l3695_369551


namespace rect_to_cylindrical_conversion_l3695_369539

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical_conversion (x y z : ℝ) (r θ : ℝ) 
  (h1 : x = 6)
  (h2 : y = 6)
  (h3 : z = -10)
  (h4 : r > 0)
  (h5 : 0 ≤ θ ∧ θ < 2 * π)
  (h6 : x = r * Real.cos θ)
  (h7 : y = r * Real.sin θ) :
  r = 6 * Real.sqrt 2 ∧ θ = π / 4 ∧ z = -10 := by
  sorry

end rect_to_cylindrical_conversion_l3695_369539


namespace weight_gain_difference_l3695_369513

def weight_gain_problem (orlando_gain jose_gain fernando_gain : ℕ) : Prop :=
  orlando_gain = 5 ∧
  jose_gain = 2 * orlando_gain + 2 ∧
  fernando_gain < jose_gain / 2 ∧
  orlando_gain + jose_gain + fernando_gain = 20

theorem weight_gain_difference (orlando_gain jose_gain fernando_gain : ℕ) 
  (h : weight_gain_problem orlando_gain jose_gain fernando_gain) :
  jose_gain / 2 - fernando_gain = 3 := by
  sorry

end weight_gain_difference_l3695_369513


namespace town_friendship_theorem_l3695_369527

structure Town where
  inhabitants : Set Nat
  friendship : inhabitants → inhabitants → Prop
  enemy : inhabitants → inhabitants → Prop

def Town.canBecomeFriends (t : Town) : Prop :=
  ∃ (steps : ℕ), ∀ (a b : t.inhabitants), t.friendship a b

theorem town_friendship_theorem (t : Town) 
  (h1 : ∀ (a b : t.inhabitants), t.friendship a b ∨ t.enemy a b)
  (h2 : ∀ (a b c : t.inhabitants), t.friendship a b → t.friendship b c → t.friendship a c)
  (h3 : ∀ (a b c : t.inhabitants), t.friendship a b ∨ t.friendship a c ∨ t.friendship b c)
  (h4 : ∀ (day : ℕ), ∃ (a : t.inhabitants), 
    ∀ (b : t.inhabitants), 
      (t.friendship a b → t.enemy a b) ∧ 
      (t.enemy a b → t.friendship a b)) :
  t.canBecomeFriends :=
sorry

end town_friendship_theorem_l3695_369527


namespace line_arrangement_count_l3695_369530

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of people in the line. -/
def totalPeople : ℕ := 7

/-- The number of people in the family that must stay together. -/
def familySize : ℕ := 3

/-- The number of individual entities to arrange (family counts as one entity). -/
def entities : ℕ := totalPeople - familySize + 1

/-- The number of ways to arrange the line of people with the family staying together. -/
def arrangements : ℕ := factorial entities * factorial familySize

theorem line_arrangement_count : arrangements = 720 := by sorry

end line_arrangement_count_l3695_369530


namespace one_fourth_of_8_8_l3695_369574

theorem one_fourth_of_8_8 : 
  (8.8 : ℚ) / 4 = 11 / 5 := by sorry

end one_fourth_of_8_8_l3695_369574


namespace phone_number_probability_correct_probability_l3695_369542

theorem phone_number_probability : ℝ → Prop :=
  fun p => (∀ n : ℕ, n ≤ 3 → n > 0 → (1 - (9/10)^n) ≤ p) ∧ p ≤ 3/10

theorem correct_probability : phone_number_probability (3/10) := by
  sorry

end phone_number_probability_correct_probability_l3695_369542


namespace logarithm_expression_equality_l3695_369533

theorem logarithm_expression_equality : 
  (Real.log 8 / Real.log 5 * Real.log 2 / Real.log 5 + 25 ^ (Real.log 3 / Real.log 5)) / 
  (Real.log 4 + Real.log 25) + 5 * Real.log 2 / Real.log 3 - Real.log (32/9) / Real.log 3 = 8 := by
  sorry

end logarithm_expression_equality_l3695_369533


namespace smartphone_price_l3695_369515

theorem smartphone_price :
  ∀ (S : ℝ),
  (∃ (PC Tablet : ℝ),
    PC = S + 500 ∧
    Tablet = S + (S + 500) ∧
    S + PC + Tablet = 2200) →
  S = 300 := by
sorry

end smartphone_price_l3695_369515


namespace total_tables_is_40_l3695_369577

/-- Represents the number of tables and seating capacity in a restaurant --/
structure Restaurant where
  new_tables : ℕ
  original_tables : ℕ
  new_table_capacity : ℕ
  original_table_capacity : ℕ
  total_seating_capacity : ℕ

/-- The conditions of the restaurant problem --/
def restaurant_conditions (r : Restaurant) : Prop :=
  r.new_table_capacity = 6 ∧
  r.original_table_capacity = 4 ∧
  r.total_seating_capacity = 212 ∧
  r.new_tables = r.original_tables + 12 ∧
  r.new_tables * r.new_table_capacity + r.original_tables * r.original_table_capacity = r.total_seating_capacity

/-- The theorem stating that the total number of tables is 40 --/
theorem total_tables_is_40 (r : Restaurant) (h : restaurant_conditions r) : 
  r.new_tables + r.original_tables = 40 := by
  sorry

end total_tables_is_40_l3695_369577


namespace correct_quadratic_equation_l3695_369518

/-- The correct quadratic equation given the conditions of the problem -/
theorem correct_quadratic_equation :
  ∀ (b c : ℝ),
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 5 ∨ x = 3 ∨ x = -6 ∨ x = -4) →
  (5 + 3 = -(b)) →
  ((-6) * (-4) = c) →
  (∀ x : ℝ, x^2 - 8*x + 24 = 0 ↔ x^2 + b*x + c = 0) :=
by sorry

end correct_quadratic_equation_l3695_369518


namespace complex_equality_l3695_369501

theorem complex_equality (z : ℂ) : z = Complex.I ↔ 
  Complex.abs (z - 2) = Complex.abs (z + 1 - Complex.I) ∧ 
  Complex.abs (z - 2) = Complex.abs (z - (1 + 2*Complex.I)) := by
  sorry

end complex_equality_l3695_369501


namespace two_lines_forming_angle_with_skew_lines_l3695_369599

/-- Represents a line in 3D space -/
structure Line3D where
  -- We'll use a simplified representation of a line
  -- More details could be added if needed

/-- Represents a point in 3D space -/
structure Point3D where
  -- We'll use a simplified representation of a point
  -- More details could be added if needed

/-- The angle between two lines -/
def angle_between_lines (l1 l2 : Line3D) : ℝ :=
  sorry

/-- Whether two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry

/-- Whether a line passes through a point -/
def line_passes_through (l : Line3D) (p : Point3D) : Prop :=
  sorry

/-- The main theorem -/
theorem two_lines_forming_angle_with_skew_lines 
  (a b : Line3D) (P : Point3D) 
  (h_skew : are_skew a b) 
  (h_angle : angle_between_lines a b = 50) : 
  ∃! (s : Finset Line3D), 
    s.card = 2 ∧ 
    ∀ l ∈ s, line_passes_through l P ∧ 
              angle_between_lines l a = 30 ∧ 
              angle_between_lines l b = 30 :=
sorry

end two_lines_forming_angle_with_skew_lines_l3695_369599


namespace tape_recorder_cost_l3695_369507

theorem tape_recorder_cost :
  ∃ (x : ℕ) (p : ℝ),
    x > 2 ∧
    170 < p ∧ p < 195 ∧
    p / (x - 2) - p / x = 1 ∧
    p = 180 := by
  sorry

end tape_recorder_cost_l3695_369507


namespace perpendicular_lines_l3695_369508

-- Define the slopes of the lines
def m1 : ℚ := 3/4
def m2 : ℚ := -3/4
def m3 : ℚ := -3/4
def m4 : ℚ := -4/3

-- Define a function to check if two lines are perpendicular
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines :
  (are_perpendicular m1 m4) ∧
  (¬ are_perpendicular m1 m2) ∧
  (¬ are_perpendicular m1 m3) ∧
  (¬ are_perpendicular m2 m3) ∧
  (¬ are_perpendicular m2 m4) ∧
  (¬ are_perpendicular m3 m4) :=
sorry

end perpendicular_lines_l3695_369508


namespace unique_solution_l3695_369590

def is_valid_digit (d : ℕ) : Prop := d > 0 ∧ d ≤ 9

def are_distinct (a b c d e f g : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g

def to_six_digit_number (a b c d e f : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f

theorem unique_solution :
  ∀ A B : ℕ,
    is_valid_digit A →
    is_valid_digit B →
    are_distinct 1 2 3 4 5 A B →
    (to_six_digit_number A 1 2 3 4 5) % B = 0 →
    (to_six_digit_number 1 2 3 4 5 A) % B = 0 →
    A = 9 ∧ B = 7 :=
by sorry

end unique_solution_l3695_369590


namespace expected_value_Y_l3695_369524

/-- A random variable following a binomial distribution -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  X : ℝ → ℝ

/-- The expected value of a binomial random variable -/
def expectedValue (X : BinomialRV n p) : ℝ := n * p

/-- Two random variables X and Y satisfying X + Y = 8, where X follows B(10, 0.6) -/
structure RandomVariables where
  X : BinomialRV 10 0.6
  Y : ℝ → ℝ
  sum_constraint : ∀ ω, X.X ω + Y ω = 8

/-- The theorem stating that E(Y) = 2 -/
theorem expected_value_Y (rv : RandomVariables) : 
  ∃ (E_Y : ℝ → ℝ), (∀ ω, E_Y ω = rv.Y ω) ∧ (∀ ω, E_Y ω = 2) :=
sorry

end expected_value_Y_l3695_369524


namespace min_perimeter_triangle_l3695_369510

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  a : ℕ  -- side EF
  b : ℕ  -- side DE and DF
  h : a = 2 * b  -- EF is twice the length of DE and DF

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an excircle of a triangle -/
structure Excircle where
  center : ℝ × ℝ
  radius : ℝ

/-- The incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The incircle of a triangle -/
def incircle (t : Triangle) : Circle := sorry

/-- The excircle of a triangle opposite to side EF -/
def excircle_EF (t : Triangle) : Excircle := sorry

/-- The excircles of a triangle opposite to sides DE and DF -/
def excircles_DE_DF (t : Triangle) : Excircle × Excircle := sorry

/-- Checks if two circles are internally tangent -/
def internally_tangent (c1 c2 : Circle) : Prop := sorry

/-- Checks if two circles are externally tangent -/
def externally_tangent (c1 c2 : Circle) : Prop := sorry

/-- The main theorem -/
theorem min_perimeter_triangle (t : Triangle) :
  let χ : Circle := incircle t
  let exc_EF : Excircle := excircle_EF t
  let (exc_DE, exc_DF) := excircles_DE_DF t
  (internally_tangent ⟨exc_EF.center, exc_EF.radius⟩ χ) ∧
  (externally_tangent ⟨exc_DE.center, exc_DE.radius⟩ χ) ∧
  (externally_tangent ⟨exc_DF.center, exc_DF.radius⟩ χ) →
  t.a + 2 * t.b ≥ 40 :=
sorry

end min_perimeter_triangle_l3695_369510


namespace equation_solution_l3695_369543

theorem equation_solution (y : ℝ) : 
  (y / 5) / 3 = 15 / (y / 3) → y = 15 * Real.sqrt 3 ∨ y = -15 * Real.sqrt 3 := by
  sorry

end equation_solution_l3695_369543


namespace arithmetic_sequence_length_l3695_369571

/-- 
Prove that an arithmetic sequence with the given properties has 15 terms.
-/
theorem arithmetic_sequence_length :
  ∀ (a l d : ℤ) (n : ℕ),
  a = -5 →  -- First term
  l = 65 →  -- Last term
  d = 5 →   -- Common difference
  l = a + (n - 1) * d →  -- Arithmetic sequence formula
  n = 15 :=  -- Number of terms
by sorry

end arithmetic_sequence_length_l3695_369571


namespace tax_revenue_consumption_relation_l3695_369504

/-- Proves that a 40% tax reduction and 25% revenue decrease results in a 25% consumption increase -/
theorem tax_revenue_consumption_relation 
  (T : ℝ) -- Original tax rate
  (C : ℝ) -- Original consumption
  (h1 : T > 0) -- Assumption: Original tax rate is positive
  (h2 : C > 0) -- Assumption: Original consumption is positive
  : 
  let new_tax := 0.6 * T -- New tax rate after 40% reduction
  let new_revenue := 0.75 * T * C -- New revenue after 25% decrease
  let new_consumption := new_revenue / new_tax -- New consumption
  new_consumption = 1.25 * C -- Proves 25% increase in consumption
  := by sorry

end tax_revenue_consumption_relation_l3695_369504


namespace circle_parabola_intersection_l3695_369503

theorem circle_parabola_intersection (b : ℝ) : 
  (∃ (a : ℝ), -- center of the circle (a, b)
    (∃ (r : ℝ), r > 0 ∧ -- radius of the circle
      (∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = r^2 → -- equation of the circle
        ((y = 3/4 * x^2) ∨ (x = 0 ∧ y = 0) ∨ (y = 3/4 * x + b)) -- intersections
      ) ∧
      (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ -- two distinct intersection points
        (3/4 * x1^2 = 3/4 * x1 + b) ∧ 
        (3/4 * x2^2 = 3/4 * x2 + b)
      )
    )
  ) → b = 25/12 :=
by sorry

end circle_parabola_intersection_l3695_369503


namespace lingling_tourist_growth_l3695_369500

/-- The average annual growth rate of tourists visiting Lingling Ancient City from 2018 to 2020 -/
def average_growth_rate : ℝ := 0.125

/-- The number of tourists (in millions) visiting Lingling Ancient City in 2018 -/
def tourists_2018 : ℝ := 6.4

/-- The number of tourists (in millions) visiting Lingling Ancient City in 2020 -/
def tourists_2020 : ℝ := 8.1

/-- The time period in years -/
def years : ℕ := 2

theorem lingling_tourist_growth :
  tourists_2018 * (1 + average_growth_rate) ^ years = tourists_2020 := by
  sorry

end lingling_tourist_growth_l3695_369500


namespace peanuts_in_box_l3695_369525

/-- The number of peanuts in a box after adding more -/
def total_peanuts (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The total number of peanuts is 10 when starting with 4 and adding 6 -/
theorem peanuts_in_box : total_peanuts 4 6 = 10 := by
  sorry

end peanuts_in_box_l3695_369525


namespace range_of_a_l3695_369594

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

-- Define the theorem
theorem range_of_a : 
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → -2 < a ∧ a < 1 :=
sorry

end range_of_a_l3695_369594


namespace negations_universal_and_true_l3695_369558

-- Define the propositions
def prop_A (x : ℝ) := x^2 - x + 1/4 < 0
def prop_C (x : ℝ) := x^2 + 2*x + 2 ≤ 0
def prop_D (x : ℝ) := x^3 + 1 = 0

-- Define the negations
def neg_A := ∀ x : ℝ, ¬(prop_A x)
def neg_C := ∀ x : ℝ, ¬(prop_C x)
def neg_D := ∀ x : ℝ, ¬(prop_D x)

-- Theorem statement
theorem negations_universal_and_true :
  (neg_A ∧ neg_C) ∧ 
  (∃ x : ℝ, prop_D x) :=
sorry

end negations_universal_and_true_l3695_369558


namespace max_xy_on_line_AB_l3695_369519

/-- Given points A(3,0) and B(0,4), prove that the maximum value of xy for any point P(x,y) on the line AB is 3. -/
theorem max_xy_on_line_AB :
  let A : ℝ × ℝ := (3, 0)
  let B : ℝ × ℝ := (0, 4)
  let line_AB (x : ℝ) := -4/3 * x + 4
  ∀ x y : ℝ, y = line_AB x → x * y ≤ 3 :=
by sorry

end max_xy_on_line_AB_l3695_369519


namespace positive_real_solution_l3695_369526

theorem positive_real_solution (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^b = b^a) (h4 : b = 4*a) : a = Real.rpow 4 (1/3) := by
  sorry

end positive_real_solution_l3695_369526


namespace hyperbola_eccentricity_l3695_369586

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 3 / 2) : 
  let e := Real.sqrt (1 + (b / a) ^ 2)
  e = Real.sqrt 13 / 2 := by sorry

end hyperbola_eccentricity_l3695_369586


namespace return_speed_calculation_l3695_369573

/-- Proves that given a round trip of 4 miles (2 miles each way), where the first half
    takes 1 hour and the average speed for the entire trip is 3 miles/hour,
    the speed for the second half of the trip is 6 miles/hour. -/
theorem return_speed_calculation (total_distance : ℝ) (outbound_distance : ℝ) 
    (outbound_time : ℝ) (average_speed : ℝ) :
  total_distance = 4 →
  outbound_distance = 2 →
  outbound_time = 1 →
  average_speed = 3 →
  ∃ (return_speed : ℝ), 
    return_speed = 6 ∧ 
    average_speed = total_distance / (outbound_time + outbound_distance / return_speed) := by
  sorry


end return_speed_calculation_l3695_369573


namespace builder_boards_count_l3695_369567

/-- The number of boards in each package -/
def boards_per_package : ℕ := 3

/-- The number of packages the builder needs to buy -/
def packages_needed : ℕ := 52

/-- The total number of boards needed -/
def total_boards : ℕ := boards_per_package * packages_needed

theorem builder_boards_count : total_boards = 156 := by
  sorry

end builder_boards_count_l3695_369567


namespace sixth_power_sum_l3695_369545

theorem sixth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 23)
  (h4 : a * x^4 + b * y^4 = 50)
  (h5 : a * x^5 + b * y^5 = 106) :
  a * x^6 + b * y^6 = 238 := by
  sorry

end sixth_power_sum_l3695_369545


namespace largest_n_for_equation_l3695_369506

theorem largest_n_for_equation : 
  (∃ n : ℕ, 
    (∀ m : ℕ, m > n → 
      ¬∃ x y z : ℕ+, m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10) ∧
    (∃ x y z : ℕ+, n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10)) ∧
  (∀ n : ℕ, 
    (∀ m : ℕ, m > n → 
      ¬∃ x y z : ℕ+, m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10) ∧
    (∃ x y z : ℕ+, n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10) →
    n = 4) :=
by sorry

end largest_n_for_equation_l3695_369506


namespace min_value_expression_l3695_369555

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt ((2 * x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) ≥ 2 * Real.rpow 3 (1/4) :=
sorry

end min_value_expression_l3695_369555


namespace cookies_in_bag_l3695_369595

/-- The number of cookies that can fit in one paper bag given a total number of cookies and bags -/
def cookies_per_bag (total_cookies : ℕ) (total_bags : ℕ) : ℕ :=
  (total_cookies / total_bags : ℕ)

/-- Theorem stating that given 292 cookies and 19 paper bags, one bag can hold at most 15 cookies -/
theorem cookies_in_bag : cookies_per_bag 292 19 = 15 := by
  sorry

end cookies_in_bag_l3695_369595


namespace symmetry_point_l3695_369541

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is the midpoint of two other points -/
def isMidpoint (mid : Point) (p1 : Point) (p2 : Point) : Prop :=
  mid.x = (p1.x + p2.x) / 2 ∧ mid.y = (p1.y + p2.y) / 2

theorem symmetry_point (m n : ℝ) :
  let M : Point := ⟨4, m⟩
  let N : Point := ⟨n, -3⟩
  let P : Point := ⟨6, -9⟩
  isMidpoint N M P → m = 3 ∧ n = 5 := by
  sorry

end symmetry_point_l3695_369541


namespace two_correct_conclusions_l3695_369566

-- Define the type for analogical conclusions
inductive AnalogyConclusion
| ComplexRational
| VectorParallel
| PlanePlanar

-- Function to check if a conclusion is correct
def isCorrectConclusion (c : AnalogyConclusion) : Prop :=
  match c with
  | .ComplexRational => True
  | .VectorParallel => False
  | .PlanePlanar => True

-- Theorem statement
theorem two_correct_conclusions :
  (∃ (c1 c2 : AnalogyConclusion), c1 ≠ c2 ∧ 
    isCorrectConclusion c1 ∧ isCorrectConclusion c2 ∧
    (∀ (c3 : AnalogyConclusion), c3 ≠ c1 ∧ c3 ≠ c2 → ¬isCorrectConclusion c3)) :=
by sorry

end two_correct_conclusions_l3695_369566


namespace distinct_reals_with_integer_differences_are_integers_l3695_369581

theorem distinct_reals_with_integer_differences_are_integers 
  (a b : ℝ) 
  (distinct : a ≠ b) 
  (int_diff : ∀ k : ℕ, ∃ n : ℤ, a^k - b^k = n) : 
  ∃ m n : ℤ, (a : ℝ) = m ∧ (b : ℝ) = n := by
  sorry

end distinct_reals_with_integer_differences_are_integers_l3695_369581


namespace vector_at_minus_2_l3695_369585

/-- A line in a plane parametrized by s -/
def line (s : ℝ) : ℝ × ℝ := sorry

/-- The vector on the line at s = 1 is (2, 5) -/
axiom vector_at_1 : line 1 = (2, 5)

/-- The vector on the line at s = 4 is (8, -7) -/
axiom vector_at_4 : line 4 = (8, -7)

/-- The vector on the line at s = -2 is (-4, 17) -/
theorem vector_at_minus_2 : line (-2) = (-4, 17) := by sorry

end vector_at_minus_2_l3695_369585


namespace arithmetic_simplification_l3695_369549

theorem arithmetic_simplification :
  (4 + 6 + 4) / 3 - 4 / 3 = 10 / 3 := by
sorry

end arithmetic_simplification_l3695_369549


namespace path_area_and_cost_l3695_369583

/-- Calculates the area of a path surrounding a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per square meter -/
def construction_cost (path_area cost_per_sqm : ℝ) : ℝ :=
  path_area * cost_per_sqm

theorem path_area_and_cost (field_length field_width path_width cost_per_sqm : ℝ)
  (h1 : field_length = 65)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5)
  (h4 : cost_per_sqm = 2) :
  path_area field_length field_width path_width = 625 ∧
  construction_cost (path_area field_length field_width path_width) cost_per_sqm = 1250 := by
  sorry

end path_area_and_cost_l3695_369583


namespace gwen_race_time_l3695_369540

/-- Represents the time Gwen spent jogging and walking during a race. -/
structure RaceTime where
  jogging : ℕ
  walking : ℕ

/-- Calculates if the given race time satisfies the required ratio and walking time. -/
def is_valid_race_time (rt : RaceTime) : Prop :=
  rt.jogging * 3 = rt.walking * 5 ∧ rt.walking = 9

/-- Theorem stating that the race time with 15 minutes of jogging and 9 minutes of walking
    satisfies the required conditions. -/
theorem gwen_race_time : ∃ (rt : RaceTime), is_valid_race_time rt ∧ rt.jogging = 15 := by
  sorry

end gwen_race_time_l3695_369540


namespace function_bound_l3695_369556

noncomputable def f (a x : ℝ) : ℝ := a * Real.sin x - 1/2 * Real.cos (2*x) + a - 3/a + 1/2

theorem function_bound (a : ℝ) (ha : a ≠ 0) :
  (∀ x, f a x ≤ 0) → 0 < a ∧ a ≤ 3 := by sorry

end function_bound_l3695_369556


namespace max_distance_ellipse_point_l3695_369587

/-- 
Given an ellipse x²/a² + y²/b² = 1 with a > b > 0, and A(0, b),
the maximum value of |PA| for any point P on the ellipse is max(a²/√(a² - b²), 2b).
-/
theorem max_distance_ellipse_point (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let ellipse := {P : ℝ × ℝ | (P.1^2 / a^2) + (P.2^2 / b^2) = 1}
  let A := (0, b)
  let dist_PA (P : ℝ × ℝ) := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  (∀ P ∈ ellipse, dist_PA P ≤ max (a^2 / Real.sqrt (a^2 - b^2)) (2*b)) ∧
  (∃ P ∈ ellipse, dist_PA P = max (a^2 / Real.sqrt (a^2 - b^2)) (2*b))
:= by sorry

end max_distance_ellipse_point_l3695_369587


namespace percentage_of_prize_money_kept_l3695_369536

-- Define the original repair cost
def original_repair_cost : ℝ := 20000

-- Define the discount percentage
def discount_percentage : ℝ := 0.20

-- Define the prize money
def prize_money : ℝ := 70000

-- Define John's profit
def profit : ℝ := 47000

-- Theorem to prove
theorem percentage_of_prize_money_kept (ε : ℝ) (h : ε > 0) :
  ∃ (percentage : ℝ), 
    abs (percentage - (profit / prize_money * 100)) < ε ∧ 
    abs (percentage - 67.14) < ε :=
sorry

end percentage_of_prize_money_kept_l3695_369536


namespace deficiency_and_excess_l3695_369552

theorem deficiency_and_excess (people : ℕ) (price : ℕ) : 
  (5 * people + 45 = price) →
  (7 * people + 3 = price) →
  (people = 21 ∧ price = 150) := by
  sorry

end deficiency_and_excess_l3695_369552


namespace max_intersections_math_city_l3695_369584

/-- Represents the number of streets in Math City -/
def total_streets : ℕ := 10

/-- Represents the number of parallel streets -/
def parallel_streets : ℕ := 2

/-- Represents the number of non-parallel streets -/
def non_parallel_streets : ℕ := total_streets - parallel_streets

/-- 
  Theorem: Maximum number of intersections in Math City
  Given:
  - There are 10 streets in total
  - Exactly 2 streets are parallel to each other
  - No other pair of streets is parallel
  - No three streets meet at a single point
  Prove: The maximum number of intersections is 44
-/
theorem max_intersections_math_city : 
  (non_parallel_streets.choose 2) + (parallel_streets * non_parallel_streets) = 44 := by
  sorry

end max_intersections_math_city_l3695_369584


namespace simplify_and_evaluate_l3695_369562

-- Define the expression as a function of x
def f (x : ℝ) : ℝ := (x + 1) * (x - 1) + x * (2 - x) + (x - 1)^2

-- Theorem stating the simplification and evaluation
theorem simplify_and_evaluate :
  (∀ x : ℝ, f x = x^2) ∧ (f 100 = 10000) := by sorry

end simplify_and_evaluate_l3695_369562


namespace quadratic_composition_theorem_l3695_369535

/-- A quadratic polynomial is a polynomial of degree 2 -/
def QuadraticPolynomial (R : Type*) [CommRing R] := {p : Polynomial R // p.degree = 2}

theorem quadratic_composition_theorem {R : Type*} [CommRing R] :
  ∀ (f : QuadraticPolynomial R),
  ∃ (g h : QuadraticPolynomial R),
  (f.val * (f.val.comp (Polynomial.X + 1))) = g.val.comp h.val :=
sorry

end quadratic_composition_theorem_l3695_369535


namespace x_over_y_value_l3695_369505

theorem x_over_y_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5*x + y)^2019 + x^2019 + 30*x + 5*y = 0) : 
  x / y = -1 / 6 := by
sorry

end x_over_y_value_l3695_369505


namespace product_sign_l3695_369591

theorem product_sign (a b c d e : ℝ) : ab^2*c^3*d^4*e^5 < 0 → ab^2*c*d^4*e < 0 := by
  sorry

end product_sign_l3695_369591


namespace coprime_35_58_in_base_l3695_369553

/-- Two natural numbers are coprime if their greatest common divisor is 1. -/
def Coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- A numeral system base is valid if it's greater than 1. -/
def ValidBase (base : ℕ) : Prop := base > 1

theorem coprime_35_58_in_base (base : ℕ) (h : ValidBase base) (h_base : base > 8) :
  Coprime 35 58 := by
  sorry

#check coprime_35_58_in_base

end coprime_35_58_in_base_l3695_369553


namespace square_difference_formula_expression_equivalence_l3695_369576

/-- The square difference formula -/
theorem square_difference_formula (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by sorry

/-- Proof that (x+y)(-x+y) is equivalent to y^2 - x^2 -/
theorem expression_equivalence (x y : ℝ) : (x + y) * (-x + y) = y^2 - x^2 := by sorry

end square_difference_formula_expression_equivalence_l3695_369576


namespace complex_equation_sum_l3695_369528

theorem complex_equation_sum (x y : ℝ) :
  (x / (1 - Complex.I)) + (y / (1 - 2 * Complex.I)) = 5 / (1 - 3 * Complex.I) →
  x + y = 4 := by
  sorry

end complex_equation_sum_l3695_369528


namespace solution_composition_l3695_369570

/-- Represents the initial percentage of liquid X in the solution -/
def initial_percentage : ℝ := 30

/-- The initial weight of the solution in kg -/
def initial_weight : ℝ := 10

/-- The weight of water that evaporates in kg -/
def evaporated_water : ℝ := 2

/-- The weight of the original solution added back in kg -/
def added_solution : ℝ := 2

/-- The final percentage of liquid X in the new solution -/
def final_percentage : ℝ := 36

theorem solution_composition :
  let remaining_weight := initial_weight - evaporated_water
  let new_total_weight := remaining_weight + added_solution
  let initial_liquid_x := initial_percentage / 100 * initial_weight
  let added_liquid_x := initial_percentage / 100 * added_solution
  let total_liquid_x := initial_liquid_x + added_liquid_x
  total_liquid_x / new_total_weight * 100 = final_percentage :=
by sorry

end solution_composition_l3695_369570


namespace cow_calf_total_cost_l3695_369596

theorem cow_calf_total_cost (cow_cost calf_cost : ℕ) 
  (h1 : cow_cost = 880)
  (h2 : calf_cost = 110)
  (h3 : cow_cost = 8 * calf_cost) : 
  cow_cost + calf_cost = 990 := by
  sorry

end cow_calf_total_cost_l3695_369596


namespace dog_toy_discount_l3695_369564

/-- Proves that the discount on the second toy in each pair is $6.00 given the conditions --/
theorem dog_toy_discount (toy_price : ℝ) (num_toys : ℕ) (total_spent : ℝ) 
  (h1 : toy_price = 12)
  (h2 : num_toys = 4)
  (h3 : total_spent = 36) :
  (toy_price * num_toys - total_spent) / 2 = 6 := by
sorry

end dog_toy_discount_l3695_369564


namespace tree_distance_l3695_369561

/-- Given a yard of length 180 meters with 11 trees planted at equal distances,
    with one tree at each end, the distance between two consecutive trees is 18 meters. -/
theorem tree_distance (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 180 →
  num_trees = 11 →
  let num_spaces := num_trees - 1
  yard_length / num_spaces = 18 :=
by sorry

end tree_distance_l3695_369561


namespace bee_multiple_l3695_369512

theorem bee_multiple (bees_day1 bees_day2 : ℕ) (h1 : bees_day1 = 144) (h2 : bees_day2 = 432) :
  bees_day2 / bees_day1 = 3 := by
  sorry

end bee_multiple_l3695_369512


namespace easter_egg_hunt_l3695_369531

/-- The number of eggs found by Cheryl exceeds the combined total of eggs found by Kevin, Bonnie, and George by 29. -/
theorem easter_egg_hunt (kevin bonnie george cheryl : ℕ) 
  (h1 : kevin = 5) 
  (h2 : bonnie = 13) 
  (h3 : george = 9) 
  (h4 : cheryl = 56) : 
  cheryl - (kevin + bonnie + george) = 29 := by
  sorry

end easter_egg_hunt_l3695_369531


namespace group_average_l3695_369548

theorem group_average (x : ℝ) : 
  (5 + 5 + x + 6 + 8) / 5 = 6 → x = 6 := by
sorry

end group_average_l3695_369548


namespace demand_proportion_for_constant_income_l3695_369516

theorem demand_proportion_for_constant_income
  (original_price original_demand : ℝ)
  (price_increase_factor : ℝ := 1.20)
  (demand_increase_factor : ℝ := 1.12)
  (h_price_positive : original_price > 0)
  (h_demand_positive : original_demand > 0) :
  let new_price := price_increase_factor * original_price
  let new_demand := (14 / 15) * original_demand
  new_price * new_demand = original_price * original_demand :=
by sorry

end demand_proportion_for_constant_income_l3695_369516


namespace additional_donation_amount_l3695_369593

/-- A proof that the additional donation was $20.00 given the conditions of the raffle ticket sale --/
theorem additional_donation_amount (num_tickets : ℕ) (ticket_price : ℚ) (num_fixed_donations : ℕ) (fixed_donation_amount : ℚ) (total_raised : ℚ) : 
  num_tickets = 25 →
  ticket_price = 2 →
  num_fixed_donations = 2 →
  fixed_donation_amount = 15 →
  total_raised = 100 →
  total_raised - (↑num_tickets * ticket_price + ↑num_fixed_donations * fixed_donation_amount) = 20 :=
by
  sorry

#check additional_donation_amount

end additional_donation_amount_l3695_369593


namespace tangent_line_b_value_l3695_369544

/-- A line tangent to a cubic curve -/
structure TangentLine where
  k : ℝ
  a : ℝ
  b : ℝ

/-- The tangent line y = kx + 1 is tangent to the curve y = x^3 + ax + b at the point (1, 3) -/
def is_tangent (t : TangentLine) : Prop :=
  3 = t.k * 1 + 1 ∧
  3 = 1^3 + t.a * 1 + t.b ∧
  t.k = 3 * 1^2 + t.a

theorem tangent_line_b_value (t : TangentLine) (h : is_tangent t) : t.b = 3 := by
  sorry

#check tangent_line_b_value

end tangent_line_b_value_l3695_369544
