import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_function_range_l3647_364769

/-- Given a quadratic function f(x) = ax^2 + bx, where 1 ≤ f(-1) ≤ 2 and 2 ≤ f(1) ≤ 4,
    the range of f(-2) is [6, 10]. -/
theorem quadratic_function_range (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x
  (1 ≤ f (-1) ∧ f (-1) ≤ 2) ∧ (2 ≤ f 1 ∧ f 1 ≤ 4) →
  6 ≤ f (-2) ∧ f (-2) ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3647_364769


namespace NUMINAMATH_CALUDE_inverse_proposition_l3647_364791

theorem inverse_proposition : 
  (∀ a b : ℝ, (a + b = 2 → ¬(a < 1 ∧ b < 1))) ↔ 
  (∀ a b : ℝ, (a < 1 ∧ b < 1 → a + b ≠ 2)) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proposition_l3647_364791


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3647_364737

theorem simplify_sqrt_expression (x : ℝ) (h : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 1) / (2*x^3))^2) = (x^3 / 2) + (1 / (2*x^3)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3647_364737


namespace NUMINAMATH_CALUDE_largest_area_error_l3647_364706

theorem largest_area_error (actual_side : ℝ) (max_error_percent : ℝ) :
  actual_side = 30 →
  max_error_percent = 20 →
  let max_measured_side := actual_side * (1 + max_error_percent / 100)
  let actual_area := actual_side ^ 2
  let max_measured_area := max_measured_side ^ 2
  let max_percent_error := (max_measured_area - actual_area) / actual_area * 100
  max_percent_error = 44 := by
sorry

end NUMINAMATH_CALUDE_largest_area_error_l3647_364706


namespace NUMINAMATH_CALUDE_regular_nonagon_side_length_l3647_364722

/-- A regular nonagon with perimeter 171 centimeters has sides of length 19 centimeters -/
theorem regular_nonagon_side_length : 
  ∀ (perimeter side_length : ℝ),
    perimeter = 171 →
    side_length * 9 = perimeter →
    side_length = 19 :=
by sorry

end NUMINAMATH_CALUDE_regular_nonagon_side_length_l3647_364722


namespace NUMINAMATH_CALUDE_intersection_M_N_l3647_364760

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 2 < 0}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Define the open interval (0, 1]
def open_unit_interval : Set ℝ := {x | 0 < x ∧ x ≤ 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = open_unit_interval := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3647_364760


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_l3647_364776

theorem complex_arithmetic_expression : 
  (2*(3*(2*(3*(2*(3 * (2+1) * 2)+2)*2)+2)*2)+2) = 5498 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_l3647_364776


namespace NUMINAMATH_CALUDE_heart_ratio_l3647_364746

-- Define the ♡ operation
def heart (n m : ℕ) : ℚ := 3 * (n^3 : ℚ) * (m^2 : ℚ)

-- State the theorem
theorem heart_ratio : (heart 3 5) / (heart 5 3) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_heart_ratio_l3647_364746


namespace NUMINAMATH_CALUDE_squarefree_juicy_integers_l3647_364718

def is_juicy (n : ℕ) : Prop :=
  n > 1 ∧ ∀ (d₁ d₂ : ℕ), d₁ ∣ n → d₂ ∣ n → d₁ < d₂ → (d₂ - d₁) ∣ n

def is_squarefree (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p * p ∣ n → p = 1)

theorem squarefree_juicy_integers :
  {n : ℕ | is_squarefree n ∧ is_juicy n} = {2, 6, 42, 1806} :=
sorry

end NUMINAMATH_CALUDE_squarefree_juicy_integers_l3647_364718


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3647_364719

theorem functional_equation_solution (c : ℝ) :
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, (f x + 1) * (f y + 1) = f (x + y) + f (x * y + c)) →
  c = 1 ∨ c = -1 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3647_364719


namespace NUMINAMATH_CALUDE_total_price_after_tax_l3647_364751

def original_price : ℝ := 200
def tax_rate : ℝ := 0.15

theorem total_price_after_tax :
  original_price * (1 + tax_rate) = 230 := by sorry

end NUMINAMATH_CALUDE_total_price_after_tax_l3647_364751


namespace NUMINAMATH_CALUDE_square_of_good_is_good_l3647_364792

def is_averaging_sequence (a : ℕ → ℤ) : Prop :=
  ∀ k, 2 * a (k + 1) = a k + a (k + 1)

def is_good_sequence (x : ℕ → ℤ) : Prop :=
  ∀ n, is_averaging_sequence (λ k => x (n + k))

theorem square_of_good_is_good (x : ℕ → ℤ) :
  is_good_sequence x → is_good_sequence (λ k => x k ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_square_of_good_is_good_l3647_364792


namespace NUMINAMATH_CALUDE_rescue_center_dog_count_l3647_364772

/-- Calculates the final number of dogs in an animal rescue center after a series of events. -/
def final_dog_count (initial : ℕ) (moved_in : ℕ) (first_adoption : ℕ) (second_adoption : ℕ) : ℕ :=
  initial + moved_in - first_adoption - second_adoption

/-- Theorem stating that given specific values for initial count, dogs moved in, and adoptions,
    the final count of dogs is 200. -/
theorem rescue_center_dog_count :
  final_dog_count 200 100 40 60 = 200 := by
  sorry

#eval final_dog_count 200 100 40 60

end NUMINAMATH_CALUDE_rescue_center_dog_count_l3647_364772


namespace NUMINAMATH_CALUDE_jacob_age_problem_l3647_364763

theorem jacob_age_problem (x : ℕ) : 
  (40 + x : ℕ) = 3 * (10 + x) ∧ 
  (40 - x : ℕ) = 7 * (10 - x) → 
  x = 5 := by sorry

end NUMINAMATH_CALUDE_jacob_age_problem_l3647_364763


namespace NUMINAMATH_CALUDE_negation_of_existence_l3647_364734

theorem negation_of_existence (x : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_l3647_364734


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3647_364740

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a^2 - b^2 = Real.sqrt 3 * b * c →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  a = 2 * Real.sin A →
  b = 2 * Real.sin B →
  c = 2 * Real.sin C →
  A + B + C = π →
  A = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3647_364740


namespace NUMINAMATH_CALUDE_compute_expression_l3647_364749

theorem compute_expression : 8 * (2/3)^4 + 2 = 290/81 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3647_364749


namespace NUMINAMATH_CALUDE_students_absent_eq_three_l3647_364703

/-- The number of cupcakes in a dozen -/
def dozen : ℕ := 12

/-- The number of cupcakes Dani brings -/
def cupcakes_brought : ℕ := (2 * dozen) + (dozen / 2)

/-- The total number of people in the class (including Dani) -/
def total_people : ℕ := 29

/-- The number of cupcakes left after distribution -/
def cupcakes_left : ℕ := 4

/-- The number of students who called in sick -/
def students_absent : ℕ := total_people - (cupcakes_brought - cupcakes_left)

theorem students_absent_eq_three : students_absent = 3 := by
  sorry

end NUMINAMATH_CALUDE_students_absent_eq_three_l3647_364703


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l3647_364700

/-- The area of the shaded region inside a square with circles at its vertices -/
theorem shaded_area_square_with_circles (side_length : ℝ) (circle_radius : ℝ) 
  (h_side : side_length = 8)
  (h_radius : circle_radius = 3 * Real.sqrt 2) : 
  let square_area := side_length ^ 2
  let triangle_area := (side_length / 2) ^ 2 / 2
  let circle_sector_area := π * circle_radius ^ 2 / 4
  let total_excluded_area := 4 * (triangle_area + circle_sector_area)
  square_area - total_excluded_area = 46 - 18 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l3647_364700


namespace NUMINAMATH_CALUDE_amelia_win_probability_l3647_364741

/-- Probability of Amelia's coin landing on heads -/
def p_amelia : ℚ := 3/7

/-- Probability of Blaine's coin landing on heads -/
def p_blaine : ℚ := 1/3

/-- Probability of at least one head in a simultaneous toss -/
def p_start : ℚ := 1 - (1 - p_amelia) * (1 - p_blaine)

/-- Probability of Amelia winning on her turn -/
def p_amelia_win : ℚ := p_amelia * p_amelia

/-- Probability of Blaine winning on his turn -/
def p_blaine_win : ℚ := p_blaine * p_blaine

/-- Probability of delay (neither wins) -/
def p_delay : ℚ := 1 - p_amelia_win - p_blaine_win

/-- The probability that Amelia wins the game -/
theorem amelia_win_probability : 
  (p_amelia_win / (1 - p_delay^2 : ℚ)) = 21609/64328 := by
  sorry

end NUMINAMATH_CALUDE_amelia_win_probability_l3647_364741


namespace NUMINAMATH_CALUDE_abc_inequality_l3647_364781

theorem abc_inequality (a b c : ℝ) (ha : |a| < 1) (hb : |b| < 1) (hc : |c| < 1) :
  a * b + b * c + c * a > -1 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3647_364781


namespace NUMINAMATH_CALUDE_elliptical_path_derivative_l3647_364797

/-- The derivative of a vector function representing an elliptical path. -/
theorem elliptical_path_derivative (a b t : ℝ) :
  let r : ℝ → ℝ × ℝ := fun t => (a * Real.cos t, b * Real.sin t)
  let dr : ℝ → ℝ × ℝ := fun t => (-a * Real.sin t, b * Real.cos t)
  (deriv r) t = dr t := by
  sorry

end NUMINAMATH_CALUDE_elliptical_path_derivative_l3647_364797


namespace NUMINAMATH_CALUDE_solution_uniqueness_l3647_364789

theorem solution_uniqueness (x y z : ℝ) :
  x^2 * y + y^2 * z + z^2 = 0 ∧
  z^3 + z^2 * y + z * y^3 + x^2 * y = 1/4 * (x^4 + y^4) →
  x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_uniqueness_l3647_364789


namespace NUMINAMATH_CALUDE_stock_value_decrease_l3647_364712

theorem stock_value_decrease (n : ℕ) (n_pos : 0 < n) : (0.99 : ℝ) ^ n < 1 := by
  sorry

#check stock_value_decrease

end NUMINAMATH_CALUDE_stock_value_decrease_l3647_364712


namespace NUMINAMATH_CALUDE_a_representation_theorem_l3647_364796

theorem a_representation_theorem (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  ∃ k : ℕ, ((n + Real.sqrt (n^2 - 4)) / 2) ^ m = (k + Real.sqrt (k^2 - 4)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_a_representation_theorem_l3647_364796


namespace NUMINAMATH_CALUDE_volume_of_specific_prism_l3647_364730

/-- A regular triangular prism inscribed in a sphere -/
structure InscribedPrism where
  /-- Radius of the sphere -/
  R : ℝ
  /-- Length of AD, where D is on the diameter CD -/
  AD : ℝ

/-- The volume of the inscribed prism -/
def prism_volume (p : InscribedPrism) : ℝ := sorry

/-- Theorem: The volume of the specific inscribed prism is 48√15 -/
theorem volume_of_specific_prism :
  let p : InscribedPrism := { R := 6, AD := 4 * Real.sqrt 6 }
  prism_volume p = 48 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_prism_l3647_364730


namespace NUMINAMATH_CALUDE_homework_problems_per_page_l3647_364799

theorem homework_problems_per_page 
  (total_problems : ℕ) 
  (solved_percentage : ℚ) 
  (remaining_pages : ℕ) 
  (h1 : total_problems = 550) 
  (h2 : solved_percentage = 65 / 100) 
  (h3 : remaining_pages = 3) : 
  (total_problems - Int.floor (solved_percentage * total_problems)) / remaining_pages = 64 := by
  sorry

end NUMINAMATH_CALUDE_homework_problems_per_page_l3647_364799


namespace NUMINAMATH_CALUDE_projectile_meeting_time_l3647_364736

/-- Time for two projectiles to meet --/
theorem projectile_meeting_time (initial_distance : ℝ) (speed1 speed2 : ℝ) :
  initial_distance = 1998 →
  speed1 = 444 →
  speed2 = 555 →
  (initial_distance / (speed1 + speed2)) * 60 = 120 := by
  sorry

end NUMINAMATH_CALUDE_projectile_meeting_time_l3647_364736


namespace NUMINAMATH_CALUDE_remainder_of_111222333_div_37_l3647_364790

theorem remainder_of_111222333_div_37 : 111222333 % 37 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_111222333_div_37_l3647_364790


namespace NUMINAMATH_CALUDE_average_equation_solution_l3647_364726

theorem average_equation_solution (x : ℝ) : 
  ((2*x + 12) + (3*x + 3) + (5*x - 8)) / 3 = 3*x + 2 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l3647_364726


namespace NUMINAMATH_CALUDE_divisibility_property_l3647_364788

theorem divisibility_property (a b c d m x y : ℤ) 
  (h1 : m = a * d - b * c)
  (h2 : Nat.gcd a.natAbs m.natAbs = 1)
  (h3 : Nat.gcd b.natAbs m.natAbs = 1)
  (h4 : Nat.gcd c.natAbs m.natAbs = 1)
  (h5 : Nat.gcd d.natAbs m.natAbs = 1)
  (h6 : m ∣ (a * x + b * y)) :
  m ∣ (c * x + d * y) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3647_364788


namespace NUMINAMATH_CALUDE_eggs_in_club_house_l3647_364710

theorem eggs_in_club_house (total eggs_in_park eggs_in_town_hall eggs_in_club_house : ℕ) :
  total = eggs_in_club_house + eggs_in_park + eggs_in_town_hall →
  eggs_in_park = 25 →
  eggs_in_town_hall = 15 →
  total = 80 →
  eggs_in_club_house = 40 := by
sorry

end NUMINAMATH_CALUDE_eggs_in_club_house_l3647_364710


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3647_364731

theorem sqrt_equation_solution : ∃! z : ℚ, Real.sqrt (10 + 3 * z) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3647_364731


namespace NUMINAMATH_CALUDE_chemistry_physics_difference_l3647_364757

/-- Represents the scores of a student in three subjects -/
structure Scores where
  math : ℕ
  physics : ℕ
  chemistry : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (s : Scores) : Prop :=
  s.math + s.physics = 70 ∧
  s.chemistry > s.physics ∧
  (s.math + s.chemistry) / 2 = 45

/-- The theorem to be proved -/
theorem chemistry_physics_difference (s : Scores) 
  (h : satisfiesConditions s) : s.chemistry - s.physics = 20 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_physics_difference_l3647_364757


namespace NUMINAMATH_CALUDE_compound_interest_period_l3647_364787

theorem compound_interest_period (P s k n : ℝ) (h_pos : k > -1) :
  P = s / (1 + k)^n →
  n = Real.log (s/P) / Real.log (1 + k) :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_period_l3647_364787


namespace NUMINAMATH_CALUDE_correct_student_activities_and_championships_l3647_364725

/-- The number of ways for students to sign up for activities and the number of possible championship outcomes -/
def student_activities_and_championships 
  (num_students : ℕ) 
  (num_activities : ℕ) 
  (num_championships : ℕ) : ℕ × ℕ :=
  (num_activities ^ num_students, num_students ^ num_championships)

/-- Theorem stating the correct number of ways for 4 students to sign up for 3 activities and compete in 3 championships -/
theorem correct_student_activities_and_championships :
  student_activities_and_championships 4 3 3 = (3^4, 4^3) := by
  sorry

end NUMINAMATH_CALUDE_correct_student_activities_and_championships_l3647_364725


namespace NUMINAMATH_CALUDE_min_value_a_min_value_a_achieved_l3647_364754

theorem min_value_a (a b : ℕ) (h1 : a > 0) (h2 : a = b - 2005) 
  (h3 : ∃ x : ℕ, x > 0 ∧ x^2 - a*x + b = 0) : a ≥ 95 := by
  sorry

theorem min_value_a_achieved (a b : ℕ) (h1 : a > 0) (h2 : a = b - 2005) : 
  (∃ x : ℕ, x > 0 ∧ x^2 - 95*x + (95 + 2005) = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_min_value_a_achieved_l3647_364754


namespace NUMINAMATH_CALUDE_altered_coin_probability_l3647_364723

theorem altered_coin_probability :
  ∃! p : ℝ, 0 < p ∧ p < 1/2 ∧ (20 : ℝ) * p^3 * (1-p)^3 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_altered_coin_probability_l3647_364723


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l3647_364759

theorem quadratic_equation_m_value : 
  ∀ m : ℝ, 
  (∀ x : ℝ, (m - 1) * x^(|m| + 1) + 2 * m * x + 3 = 0 → 
    (|m| + 1 = 2 ∧ m - 1 ≠ 0)) → 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l3647_364759


namespace NUMINAMATH_CALUDE_boat_speed_l3647_364774

/-- Proves that the speed of a boat in still water is 30 kmph given specific conditions -/
theorem boat_speed (x : ℝ) (h1 : x > 0) : 
  (∃ t : ℝ, t > 0 ∧ 80 = (x + 10) * t ∧ 40 = (x - 10) * t) → x = 30 := by
  sorry

#check boat_speed

end NUMINAMATH_CALUDE_boat_speed_l3647_364774


namespace NUMINAMATH_CALUDE_jia_age_is_24_l3647_364782

/-- Represents the ages of four individuals -/
structure Ages where
  jia : ℕ
  yi : ℕ
  bing : ℕ
  ding : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- Jia is 4 years older than Yi
  ages.jia = ages.yi + 4 ∧
  -- Ding is 17 years old
  ages.ding = 17 ∧
  -- The average age of Jia, Yi, and Bing is 1 year more than the average age of all four people
  (ages.jia + ages.yi + ages.bing) / 3 = (ages.jia + ages.yi + ages.bing + ages.ding) / 4 + 1 ∧
  -- The average age of Jia and Yi is 1 year more than the average age of Jia, Yi, and Bing
  (ages.jia + ages.yi) / 2 = (ages.jia + ages.yi + ages.bing) / 3 + 1

/-- The theorem stating that if the conditions are satisfied, Jia's age is 24 -/
theorem jia_age_is_24 (ages : Ages) (h : satisfies_conditions ages) : ages.jia = 24 := by
  sorry

end NUMINAMATH_CALUDE_jia_age_is_24_l3647_364782


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_three_l3647_364780

theorem gcd_of_powers_of_three : Nat.gcd (3^1200 - 1) (3^1210 - 1) = 3^10 - 1 := by sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_three_l3647_364780


namespace NUMINAMATH_CALUDE_max_candies_karlson_candy_theorem_l3647_364768

/-- Represents the process of combining numbers and counting products -/
def combine_numbers (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

/-- The maximum number of candies Karlson can eat -/
theorem max_candies : combine_numbers 26 = 325 := by
  sorry

/-- Proves that the maximum number of candies is achieved -/
theorem karlson_candy_theorem (initial_count : ℕ) (operation_count : ℕ) 
  (h1 : initial_count = 26) (h2 : operation_count = 25) : 
  combine_numbers initial_count = 325 := by
  sorry

end NUMINAMATH_CALUDE_max_candies_karlson_candy_theorem_l3647_364768


namespace NUMINAMATH_CALUDE_rental_cost_equality_l3647_364752

/-- The daily rate for Sunshine Car Rentals in dollars -/
def sunshine_base : ℝ := 17.99

/-- The per-mile rate for Sunshine Car Rentals in dollars -/
def sunshine_per_mile : ℝ := 0.18

/-- The daily rate for City Rentals in dollars -/
def city_base : ℝ := 18.95

/-- The per-mile rate for City Rentals in dollars -/
def city_per_mile : ℝ := 0.16

/-- The mileage at which the cost is the same for both rental companies -/
def equal_cost_mileage : ℝ := 48

theorem rental_cost_equality :
  sunshine_base + sunshine_per_mile * equal_cost_mileage =
  city_base + city_per_mile * equal_cost_mileage :=
by sorry

end NUMINAMATH_CALUDE_rental_cost_equality_l3647_364752


namespace NUMINAMATH_CALUDE_total_distance_to_school_l3647_364727

-- Define the distances
def bus_distance_km : ℝ := 2
def walking_distance_m : ℝ := 560

-- Define the conversion factor
def km_to_m : ℝ := 1000

-- Theorem to prove
theorem total_distance_to_school :
  bus_distance_km * km_to_m + walking_distance_m = 2560 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_to_school_l3647_364727


namespace NUMINAMATH_CALUDE_max_product_sum_2006_l3647_364701

theorem max_product_sum_2006 :
  ∃ (a b : ℤ), a + b = 2006 ∧
    ∀ (x y : ℤ), x + y = 2006 → x * y ≤ a * b ∧
    a * b = 1006009 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2006_l3647_364701


namespace NUMINAMATH_CALUDE_fraction_existence_and_nonexistence_l3647_364708

theorem fraction_existence_and_nonexistence :
  (∀ n : ℕ+, ∃ a b : ℤ, 0 < b ∧ b ≤ Real.sqrt n + 1 ∧ Real.sqrt n ≤ a / b ∧ a / b ≤ Real.sqrt (n + 1)) ∧
  (∃ f : ℕ+ → ℕ+, Function.Injective f ∧ ∀ n : ℕ+, ¬∃ a b : ℤ, 0 < b ∧ b ≤ Real.sqrt (f n) ∧ Real.sqrt (f n) ≤ a / b ∧ a / b ≤ Real.sqrt (f n + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_existence_and_nonexistence_l3647_364708


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3647_364721

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n, S n = (n : ℝ) * (a 0 + a (n-1)) / 2

/-- Theorem: If S_n / S_2n = (n+1) / (4n+2) for an arithmetic sequence,
    then a_3 / a_5 = 3/5 -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : ∀ n, seq.S n / seq.S (2*n) = (n + 1 : ℝ) / (4*n + 2)) : 
  seq.a 3 / seq.a 5 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3647_364721


namespace NUMINAMATH_CALUDE_inscribed_trapezoid_leg_length_l3647_364717

/-- A trapezoid inscribed in a circle -/
structure InscribedTrapezoid where
  radius : ℝ
  base1 : ℝ
  base2 : ℝ
  centerInside : Bool

/-- The average length of the legs of the trapezoid squared -/
def averageLegLengthSquared (t : InscribedTrapezoid) : ℝ :=
  sorry

/-- Theorem: For a trapezoid JANE inscribed in a circle of radius 25 with the center inside,
    if the bases are 14 and 30, then the average leg length squared is 2000 -/
theorem inscribed_trapezoid_leg_length
    (t : InscribedTrapezoid)
    (h1 : t.radius = 25)
    (h2 : t.base1 = 14)
    (h3 : t.base2 = 30)
    (h4 : t.centerInside = true) :
  averageLegLengthSquared t = 2000 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_trapezoid_leg_length_l3647_364717


namespace NUMINAMATH_CALUDE_sin_negative_nineteen_pi_sixths_l3647_364794

theorem sin_negative_nineteen_pi_sixths (π : Real) : 
  let sine_is_odd : ∀ x, Real.sin (-x) = -Real.sin x := by sorry
  let sine_period : ∀ x, Real.sin (x + 2 * π) = Real.sin x := by sorry
  let sine_cofunction : ∀ θ, Real.sin (π + θ) = -Real.sin θ := by sorry
  let sin_pi_sixth : Real.sin (π / 6) = 1 / 2 := by sorry
  Real.sin (-19 * π / 6) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_negative_nineteen_pi_sixths_l3647_364794


namespace NUMINAMATH_CALUDE_total_toy_cost_l3647_364729

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59

theorem total_toy_cost : football_cost + marbles_cost = 12.30 := by
  sorry

end NUMINAMATH_CALUDE_total_toy_cost_l3647_364729


namespace NUMINAMATH_CALUDE_sin_neg_seven_pi_thirds_l3647_364716

theorem sin_neg_seven_pi_thirds : Real.sin (-7 * Real.pi / 3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_seven_pi_thirds_l3647_364716


namespace NUMINAMATH_CALUDE_direction_vector_b_l3647_364707

/-- Prove that for a line passing through points (-6, 0) and (-3, 3), its direction vector (3, b) has b = 3. -/
theorem direction_vector_b (b : ℝ) : 
  let p1 : ℝ × ℝ := (-6, 0)
  let p2 : ℝ × ℝ := (-3, 3)
  let direction_vector : ℝ × ℝ := (3, b)
  (p2.1 - p1.1 = direction_vector.1 ∧ p2.2 - p1.2 = direction_vector.2) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_direction_vector_b_l3647_364707


namespace NUMINAMATH_CALUDE_proportional_segments_l3647_364720

/-- A set of four line segments (a, b, c, d) is proportional if a * d = b * c -/
def isProportional (a b c d : ℝ) : Prop := a * d = b * c

/-- The set of line segments (2, 4, 8, 16) is proportional -/
theorem proportional_segments : isProportional 2 4 8 16 := by
  sorry

end NUMINAMATH_CALUDE_proportional_segments_l3647_364720


namespace NUMINAMATH_CALUDE_inverse_157_mod_263_l3647_364735

/-- The multiplicative inverse of 157 modulo 263 is 197 -/
theorem inverse_157_mod_263 : ∃ x : ℕ, x < 263 ∧ (157 * x) % 263 = 1 :=
by
  use 197
  sorry

end NUMINAMATH_CALUDE_inverse_157_mod_263_l3647_364735


namespace NUMINAMATH_CALUDE_horner_v3_value_l3647_364742

def f (x : ℝ) : ℝ := 3*x^6 + 5*x^5 + 6*x^4 + 79*x^3 - 8*x^2 + 35*x + 12

def horner_v3 (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  v2 * x + 79

theorem horner_v3_value :
  horner_v3 f (-4) = -57 := by sorry

end NUMINAMATH_CALUDE_horner_v3_value_l3647_364742


namespace NUMINAMATH_CALUDE_additional_amount_needed_l3647_364786

def pencil_cost : ℚ := 6
def notebook_cost : ℚ := 7/2
def pen_cost : ℚ := 9/4
def initial_amount : ℚ := 5
def borrowed_amount : ℚ := 53/100

def total_cost : ℚ := pencil_cost + notebook_cost + pen_cost
def total_available : ℚ := initial_amount + borrowed_amount

theorem additional_amount_needed : total_cost - total_available = 311/50 := by
  sorry

end NUMINAMATH_CALUDE_additional_amount_needed_l3647_364786


namespace NUMINAMATH_CALUDE_jelly_bean_distribution_l3647_364745

/-- Represents the number of jelly beans each person has -/
structure JellyBeans :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Performs the first distribution: A gives to B and C -/
def firstDistribution (jb : JellyBeans) : JellyBeans :=
  ⟨jb.a - jb.b - jb.c, jb.b + jb.b, jb.c + jb.c⟩

/-- Performs the second distribution: B gives to A and C -/
def secondDistribution (jb : JellyBeans) : JellyBeans :=
  ⟨jb.a + jb.a, jb.b - jb.a - jb.c, jb.c + jb.c⟩

/-- Performs the third distribution: C gives to A and B -/
def thirdDistribution (jb : JellyBeans) : JellyBeans :=
  ⟨jb.a + jb.a, jb.b + jb.b, jb.c - jb.a - jb.b⟩

theorem jelly_bean_distribution :
  let initial := JellyBeans.mk 104 56 32
  let final := thirdDistribution (secondDistribution (firstDistribution initial))
  final.a = 64 ∧ final.b = 64 ∧ final.c = 64 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_distribution_l3647_364745


namespace NUMINAMATH_CALUDE_total_monthly_payment_l3647_364778

def basic_service : ℕ := 15
def movie_channels : ℕ := 12
def sports_channels : ℕ := movie_channels - 3

theorem total_monthly_payment :
  basic_service + movie_channels + sports_channels = 36 :=
by sorry

end NUMINAMATH_CALUDE_total_monthly_payment_l3647_364778


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l3647_364779

theorem cube_sum_inequality (a b c : ℝ) 
  (h1 : a ≥ -1) (h2 : b ≥ -1) (h3 : c ≥ -1) 
  (h4 : a^3 + b^3 + c^3 = 1) : 
  a + b + c + a^2 + b^2 + c^2 ≤ 4 ∧ 
  (a + b + c + a^2 + b^2 + c^2 = 4 ↔ 
    (a = -1 ∧ b = 1 ∧ c = 1) ∨ 
    (a = -1 ∧ b = 1 ∧ c = 1) ∨ 
    (a = 1 ∧ b = -1 ∧ c = 1) ∨ 
    (a = 1 ∧ b = 1 ∧ c = -1) ∨ 
    (a = -1 ∧ b = 1 ∧ c = 1) ∨ 
    (a = 1 ∧ b = -1 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l3647_364779


namespace NUMINAMATH_CALUDE_floor_sqrt_45_squared_plus_twice_floor_sqrt_45_plus_1_l3647_364733

theorem floor_sqrt_45_squared_plus_twice_floor_sqrt_45_plus_1 : 
  Int.floor (Real.sqrt 45)^2 + 2 * Int.floor (Real.sqrt 45) + 1 = 49 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_45_squared_plus_twice_floor_sqrt_45_plus_1_l3647_364733


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3647_364709

/-- The number of games played in a round-robin tournament -/
def gamesPlayed (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess group with 15 players, where each player plays every other player exactly once,
    and each game is played by two players, the total number of games played is 105. -/
theorem chess_tournament_games :
  gamesPlayed 15 = 105 := by
  sorry

#eval gamesPlayed 15  -- This will evaluate to 105

end NUMINAMATH_CALUDE_chess_tournament_games_l3647_364709


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_two_sufficient_condition_implies_m_range_l3647_364777

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 ≤ 0}

-- Theorem 1
theorem intersection_implies_m_equals_two :
  ∀ m : ℝ, A ∩ B m = Set.Icc 0 3 → m = 2 := by sorry

-- Theorem 2
theorem sufficient_condition_implies_m_range :
  ∀ m : ℝ, A ⊆ Set.univ \ B m → m > 5 ∨ m < -3 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_two_sufficient_condition_implies_m_range_l3647_364777


namespace NUMINAMATH_CALUDE_carrots_and_cauliflower_cost_l3647_364704

/-- The cost of a bunch of carrots and a cauliflower given specific pricing conditions -/
theorem carrots_and_cauliflower_cost :
  ∀ (p c f o : ℝ),
    p + c + f + o = 30 →  -- Total cost
    o = 3 * p →           -- Oranges cost thrice potatoes
    f = p + c →           -- Cauliflower costs sum of potatoes and carrots
    c + f = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_carrots_and_cauliflower_cost_l3647_364704


namespace NUMINAMATH_CALUDE_impossibleOneLight_l3647_364744

/- Define the grid size -/
def gridSize : Nat := 8

/- Define the state of the grid as a function from coordinates to bool -/
def GridState := Fin gridSize → Fin gridSize → Bool

/- Define the initial state where all bulbs are on -/
def initialState : GridState := fun _ _ => true

/- Define the toggle operation for a row -/
def toggleRow (state : GridState) (row : Fin gridSize) : GridState :=
  fun i j => if i = row then !state i j else state i j

/- Define the toggle operation for a column -/
def toggleColumn (state : GridState) (col : Fin gridSize) : GridState :=
  fun i j => if j = col then !state i j else state i j

/- Define a property that checks if exactly one bulb is on -/
def exactlyOneBulbOn (state : GridState) : Prop :=
  ∃! i j, state i j = true

/- The main theorem -/
theorem impossibleOneLight : 
  ¬∃ (toggleSequence : List (Bool × Fin gridSize)), 
    let finalState := toggleSequence.foldl 
      (fun acc (toggle) => 
        match toggle with
        | (true, n) => toggleRow acc n
        | (false, n) => toggleColumn acc n) 
      initialState
    exactlyOneBulbOn finalState :=
by
  sorry

end NUMINAMATH_CALUDE_impossibleOneLight_l3647_364744


namespace NUMINAMATH_CALUDE_solution_set_implies_a_values_l3647_364739

theorem solution_set_implies_a_values (a : ℕ) 
  (h : ∀ x : ℝ, (a - 2 : ℝ) * x > (a - 2 : ℝ) ↔ x < 1) : 
  a = 0 ∨ a = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_values_l3647_364739


namespace NUMINAMATH_CALUDE_complex_number_problem_l3647_364747

theorem complex_number_problem (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (Complex.I * (α - 3 * β)).re > 0)
  (h3 : β = 4 + 3 * Complex.I) :
  α = 12 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3647_364747


namespace NUMINAMATH_CALUDE_tobias_allowance_is_five_l3647_364728

/-- Represents Tobias's financial situation --/
structure TobiasFinances where
  shoe_cost : ℕ
  saving_months : ℕ
  change : ℕ
  lawns_mowed : ℕ
  lawn_price : ℕ
  driveways_shoveled : ℕ
  driveway_price : ℕ

/-- Calculates Tobias's monthly allowance --/
def monthly_allowance (tf : TobiasFinances) : ℕ :=
  let total_earned := tf.lawns_mowed * tf.lawn_price + tf.driveways_shoveled * tf.driveway_price
  let total_had := tf.shoe_cost + tf.change
  let allowance_total := total_had - total_earned
  allowance_total / tf.saving_months

/-- Theorem stating that Tobias's monthly allowance is $5 --/
theorem tobias_allowance_is_five (tf : TobiasFinances) 
  (h1 : tf.shoe_cost = 95)
  (h2 : tf.saving_months = 3)
  (h3 : tf.change = 15)
  (h4 : tf.lawns_mowed = 4)
  (h5 : tf.lawn_price = 15)
  (h6 : tf.driveways_shoveled = 5)
  (h7 : tf.driveway_price = 7) :
  monthly_allowance tf = 5 := by
  sorry

end NUMINAMATH_CALUDE_tobias_allowance_is_five_l3647_364728


namespace NUMINAMATH_CALUDE_vector_problem_l3647_364784

def a : ℝ × ℝ := (-3, 1)
def b : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (1, 1)

theorem vector_problem :
  (∃ θ : ℝ, θ = Real.arccos ((-3 * 1 + 1 * (-2)) / (Real.sqrt 10 * Real.sqrt 5)) ∧ θ = 3 * π / 4) ∧
  (∃ k : ℝ, (∃ t : ℝ, t ≠ 0 ∧ c = t • (a + k • b)) ∧ k = 4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l3647_364784


namespace NUMINAMATH_CALUDE_carson_gardening_time_l3647_364761

/-- Represents the gardening tasks Carson needs to complete -/
structure GardeningTasks where
  mow_lines : ℕ
  mow_time_per_line : ℕ
  flower_rows : ℕ
  flowers_per_row : ℕ
  planting_time_per_flower : ℚ
  garden_sections : ℕ
  watering_time_per_section : ℕ
  hedges : ℕ
  trimming_time_per_hedge : ℕ

/-- Calculates the total gardening time in minutes -/
def total_gardening_time (tasks : GardeningTasks) : ℚ :=
  tasks.mow_lines * tasks.mow_time_per_line +
  tasks.flower_rows * tasks.flowers_per_row * tasks.planting_time_per_flower +
  tasks.garden_sections * tasks.watering_time_per_section +
  tasks.hedges * tasks.trimming_time_per_hedge

/-- Theorem stating that Carson's total gardening time is 162 minutes -/
theorem carson_gardening_time :
  let tasks : GardeningTasks := {
    mow_lines := 40,
    mow_time_per_line := 2,
    flower_rows := 10,
    flowers_per_row := 8,
    planting_time_per_flower := 1/2,
    garden_sections := 4,
    watering_time_per_section := 3,
    hedges := 5,
    trimming_time_per_hedge := 6
  }
  total_gardening_time tasks = 162 := by
  sorry


end NUMINAMATH_CALUDE_carson_gardening_time_l3647_364761


namespace NUMINAMATH_CALUDE_anthony_pencil_count_l3647_364766

/-- Given Anthony's initial pencil count and the number of pencils Kathryn gives him,
    prove that the total number of pencils Anthony has is equal to the sum of these two quantities. -/
theorem anthony_pencil_count (initial : ℕ) (given : ℕ) : initial + given = initial + given :=
by sorry

end NUMINAMATH_CALUDE_anthony_pencil_count_l3647_364766


namespace NUMINAMATH_CALUDE_piglet_banana_count_l3647_364756

/-- Represents the number of bananas eaten by each character -/
structure BananaCount where
  winnie : ℕ
  owl : ℕ
  rabbit : ℕ
  piglet : ℕ

/-- The conditions of the banana distribution problem -/
def BananaDistribution (bc : BananaCount) : Prop :=
  bc.winnie + bc.owl + bc.rabbit + bc.piglet = 70 ∧
  bc.owl + bc.rabbit = 45 ∧
  bc.winnie > bc.owl ∧
  bc.winnie > bc.rabbit ∧
  bc.winnie > bc.piglet ∧
  bc.winnie ≥ 1 ∧
  bc.owl ≥ 1 ∧
  bc.rabbit ≥ 1 ∧
  bc.piglet ≥ 1

theorem piglet_banana_count (bc : BananaCount) :
  BananaDistribution bc → bc.piglet = 1 := by
  sorry

end NUMINAMATH_CALUDE_piglet_banana_count_l3647_364756


namespace NUMINAMATH_CALUDE_video_recorder_markup_percentage_l3647_364771

/-- Proves that the percentage markup over wholesale cost is 20% for a video recorder --/
theorem video_recorder_markup_percentage
  (wholesale_cost : ℝ)
  (employee_discount : ℝ)
  (employee_paid : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : employee_discount = 0.05)
  (h3 : employee_paid = 228)
  : ∃ (markup_percentage : ℝ),
    markup_percentage = 20 ∧
    employee_paid = (1 - employee_discount) * (wholesale_cost * (1 + markup_percentage / 100)) :=
by sorry

end NUMINAMATH_CALUDE_video_recorder_markup_percentage_l3647_364771


namespace NUMINAMATH_CALUDE_opposite_of_one_l3647_364713

/-- Two real numbers are opposites if their sum is zero -/
def IsOpposite (x y : ℝ) : Prop := x + y = 0

/-- If a is the opposite of 1, then a = -1 -/
theorem opposite_of_one (a : ℝ) (h : IsOpposite a 1) : a = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_l3647_364713


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3647_364758

theorem quadratic_root_property (a : ℝ) : 
  a^2 - a - 100 = 0 → a^4 - 201*a = 10100 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3647_364758


namespace NUMINAMATH_CALUDE_new_boarders_correct_l3647_364785

/-- The number of new boarders that joined the school -/
def new_boarders : ℕ := 15

/-- The initial number of boarders -/
def initial_boarders : ℕ := 60

/-- The initial ratio of boarders to day students -/
def initial_ratio : ℚ := 2 / 5

/-- The final ratio of boarders to day students -/
def final_ratio : ℚ := 1 / 2

/-- The theorem stating that the number of new boarders is correct -/
theorem new_boarders_correct :
  let initial_day_students := (initial_boarders : ℚ) / initial_ratio
  (initial_boarders + new_boarders : ℚ) / initial_day_students = final_ratio :=
by sorry

end NUMINAMATH_CALUDE_new_boarders_correct_l3647_364785


namespace NUMINAMATH_CALUDE_fraction_equality_l3647_364750

theorem fraction_equality (a b : ℝ) : (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3647_364750


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_smallest_number_of_eggs_proof_l3647_364705

theorem smallest_number_of_eggs : ℕ → Prop :=
  fun n =>
    (n > 100) ∧
    (∃ c : ℕ, n = 15 * c - 3) ∧
    (∀ m : ℕ, m > 100 ∧ (∃ d : ℕ, m = 15 * d - 3) → m ≥ n) →
    n = 102

-- The proof goes here
theorem smallest_number_of_eggs_proof : smallest_number_of_eggs 102 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_smallest_number_of_eggs_proof_l3647_364705


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3647_364793

theorem sin_cos_identity : 
  Real.sin (10 * π / 180) * Real.cos (70 * π / 180) - 
  Real.cos (10 * π / 180) * Real.cos (20 * π / 180) = 
  -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3647_364793


namespace NUMINAMATH_CALUDE_fold_lines_cover_outside_l3647_364743

/-- A circle with center O and radius R -/
structure Circle where
  O : ℝ × ℝ
  R : ℝ

/-- A point A inside the circle -/
structure InnerPoint (c : Circle) where
  A : ℝ × ℝ
  dist_OA : Real.sqrt ((A.1 - c.O.1)^2 + (A.2 - c.O.2)^2) < c.R

/-- A point on the circumference of the circle -/
def CircumferencePoint (c : Circle) : Type :=
  { p : ℝ × ℝ // Real.sqrt ((p.1 - c.O.1)^2 + (p.2 - c.O.2)^2) = c.R }

/-- The set of all points on a fold line -/
def FoldLine (c : Circle) (A : InnerPoint c) (A' : CircumferencePoint c) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • A.A + t • A'.val }

/-- The set of all points on all possible fold lines -/
def AllFoldLines (c : Circle) (A : InnerPoint c) : Set (ℝ × ℝ) :=
  ⋃ (A' : CircumferencePoint c), FoldLine c A A'

/-- The set of points outside and on the circle -/
def OutsideAndOnCircle (c : Circle) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | Real.sqrt ((p.1 - c.O.1)^2 + (p.2 - c.O.2)^2) ≥ c.R }

/-- The main theorem -/
theorem fold_lines_cover_outside (c : Circle) (A : InnerPoint c) :
  AllFoldLines c A = OutsideAndOnCircle c := by sorry

end NUMINAMATH_CALUDE_fold_lines_cover_outside_l3647_364743


namespace NUMINAMATH_CALUDE_frankies_pets_l3647_364715

/-- The number of pets Frankie has -/
def total_pets (cats : ℕ) : ℕ :=
  let snakes := 2 * cats
  let parrots := cats - 1
  let tortoises := parrots + 1
  let dogs := 2
  let hamsters := 3
  let fish := 5
  cats + snakes + parrots + tortoises + dogs + hamsters + fish

/-- Theorem stating the total number of Frankie's pets -/
theorem frankies_pets :
  ∃ (cats : ℕ),
    2 * cats + cats + 2 = 14 ∧
    total_pets cats = 39 := by
  sorry

end NUMINAMATH_CALUDE_frankies_pets_l3647_364715


namespace NUMINAMATH_CALUDE_wheel_revolutions_for_one_mile_l3647_364755

-- Define the wheel diameter in feet
def wheel_diameter : ℝ := 8

-- Define one mile in feet
def mile_in_feet : ℝ := 5280

-- Theorem statement
theorem wheel_revolutions_for_one_mile :
  (mile_in_feet / (π * wheel_diameter)) = 660 / π := by
  sorry

end NUMINAMATH_CALUDE_wheel_revolutions_for_one_mile_l3647_364755


namespace NUMINAMATH_CALUDE_max_y_over_x_l3647_364738

theorem max_y_over_x (x y : ℝ) (h : x^2 + y^2 - 6*x - 6*y + 12 = 0) :
  ∃ (k : ℝ), k = 3 + 2 * Real.sqrt 2 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 - 6*x' - 6*y' + 12 = 0 → y' / x' ≤ k := by
  sorry

end NUMINAMATH_CALUDE_max_y_over_x_l3647_364738


namespace NUMINAMATH_CALUDE_semicircle_path_equality_l3647_364748

theorem semicircle_path_equality :
  let large_diameter : ℝ := 20
  let small_diameter : ℝ := 10
  let large_arc_length := π * large_diameter / 2
  let small_arc_length := π * small_diameter / 2
  large_arc_length = 2 * small_arc_length :=
by sorry

end NUMINAMATH_CALUDE_semicircle_path_equality_l3647_364748


namespace NUMINAMATH_CALUDE_train_distance_theorem_l3647_364770

/-- Calculates the total distance traveled by a train given its speed, initial distance, and time -/
def total_distance (speed : ℚ) (initial_distance : ℚ) (time : ℚ) : ℚ :=
  speed * time + initial_distance

/-- Proves that a train traveling at 1 mile every 2 minutes, starting with an initial distance of 5 miles, 
    will cover a total distance of 50 miles in 1 hour and 30 minutes -/
theorem train_distance_theorem : 
  let speed : ℚ := 1 / 2  -- 1 mile per 2 minutes
  let initial_distance : ℚ := 5
  let time : ℚ := 90  -- 1 hour and 30 minutes in minutes
  total_distance speed initial_distance time = 50 := by
sorry

#eval total_distance (1/2) 5 90

end NUMINAMATH_CALUDE_train_distance_theorem_l3647_364770


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l3647_364795

theorem quadratic_roots_to_coefficients :
  ∀ (b c : ℝ),
  (∀ x : ℝ, x^2 - b*x + c = 0 ↔ x = 1 ∨ x = -2) →
  b = -1 ∧ c = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l3647_364795


namespace NUMINAMATH_CALUDE_baseball_card_packs_l3647_364724

/-- The number of packs of baseball cards for a group of people -/
def total_packs (num_people : ℕ) (cards_per_person : ℕ) (cards_per_pack : ℕ) : ℕ :=
  num_people * (cards_per_person / cards_per_pack)

/-- Theorem: Four people buying 540 cards each, with 20 cards per pack, have 108 packs in total -/
theorem baseball_card_packs :
  total_packs 4 540 20 = 108 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_packs_l3647_364724


namespace NUMINAMATH_CALUDE_prime_equation_solutions_l3647_364783

/-- A prime number (not necessarily positive) -/
def IsPrime (n : ℤ) : Prop := n ≠ 0 ∧ n ≠ 1 ∧ n ≠ -1 ∧ ∀ m : ℤ, m ∣ n → (m = 1 ∨ m = -1 ∨ m = n ∨ m = -n)

/-- The set of solutions -/
def SolutionSet : Set (ℤ × ℤ × ℤ) :=
  {(5, 2, 2), (-5, -2, -2), (-5, 3, -2), (-5, -2, 3), (5, 2, -3), (5, -3, 2)}

theorem prime_equation_solutions :
  ∀ p q r : ℤ,
    IsPrime p ∧ IsPrime q ∧ IsPrime r →
    (1 / (p - q - r : ℚ) = 1 / (q : ℚ) + 1 / (r : ℚ)) ↔ (p, q, r) ∈ SolutionSet :=
by sorry

end NUMINAMATH_CALUDE_prime_equation_solutions_l3647_364783


namespace NUMINAMATH_CALUDE_polyhedron_problem_l3647_364711

/-- Represents a convex polyhedron with hexagonal and quadrilateral faces. -/
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  hexagons : ℕ
  quadrilaterals : ℕ
  H : ℕ
  Q : ℕ

/-- Euler's formula for convex polyhedra -/
def euler_formula (p : Polyhedron) : Prop :=
  p.vertices - p.edges + p.faces = 2

/-- The number of edges in terms of hexagons and quadrilaterals -/
def edge_count (p : Polyhedron) : Prop :=
  p.edges = 2 * p.quadrilaterals + 3 * p.hexagons

/-- Theorem about the specific polyhedron described in the problem -/
theorem polyhedron_problem :
  ∀ p : Polyhedron,
    p.faces = 44 →
    p.hexagons = 12 →
    p.quadrilaterals = 32 →
    p.H = 2 →
    p.Q = 2 →
    euler_formula p →
    edge_count p →
    100 * p.H + 10 * p.Q + p.vertices = 278 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_problem_l3647_364711


namespace NUMINAMATH_CALUDE_cookie_price_calculation_l3647_364775

def trip_cost : ℝ := 5000
def hourly_wage : ℝ := 20
def hours_worked : ℝ := 10
def cookies_sold : ℝ := 24
def lottery_ticket_cost : ℝ := 10
def lottery_winnings : ℝ := 500
def gift_per_sister : ℝ := 500
def num_sisters : ℝ := 2
def additional_money_needed : ℝ := 3214

theorem cookie_price_calculation (trip_cost hourly_wage hours_worked 
  cookies_sold lottery_ticket_cost lottery_winnings gift_per_sister 
  num_sisters additional_money_needed : ℝ) :
  let total_earnings := hourly_wage * hours_worked + 
    lottery_winnings + gift_per_sister * num_sisters - lottery_ticket_cost
  let cookie_revenue := trip_cost - total_earnings
  cookie_revenue / cookies_sold = 204.33 := by
  sorry

end NUMINAMATH_CALUDE_cookie_price_calculation_l3647_364775


namespace NUMINAMATH_CALUDE_fran_travel_time_l3647_364714

/-- Proves that given Joann's speed and time, and Fran's speed, Fran will take 3 hours to travel the same distance as Joann. -/
theorem fran_travel_time (joann_speed fran_speed : ℝ) (joann_time : ℝ) 
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_speed = 20) :
  (joann_speed * joann_time) / fran_speed = 3 := by
  sorry

#check fran_travel_time

end NUMINAMATH_CALUDE_fran_travel_time_l3647_364714


namespace NUMINAMATH_CALUDE_circle_parabola_tangency_height_difference_l3647_364798

/-- Given a parabola y = 4x^2 and a circle tangent to it at two points,
    the height difference between the circle's center and the points of tangency is 1/8 -/
theorem circle_parabola_tangency_height_difference :
  ∀ (a b r : ℝ),
  (∀ x y : ℝ, y = 4 * x^2 → x^2 + (y - b)^2 = r^2) →  -- Circle equation
  (a^2 + (4 * a^2 - b)^2 = r^2) →                     -- Tangency condition at (a, 4a^2)
  ((-a)^2 + (4 * (-a)^2 - b)^2 = r^2) →               -- Tangency condition at (-a, 4a^2)
  b - 4 * a^2 = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_circle_parabola_tangency_height_difference_l3647_364798


namespace NUMINAMATH_CALUDE_last_digit_of_power_difference_l3647_364753

theorem last_digit_of_power_difference : (7^95 - 3^58) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_power_difference_l3647_364753


namespace NUMINAMATH_CALUDE_classroom_students_l3647_364767

theorem classroom_students (n : ℕ) : 
  n < 50 → n % 8 = 5 → n % 6 = 3 → (n = 21 ∨ n = 45) := by
  sorry

end NUMINAMATH_CALUDE_classroom_students_l3647_364767


namespace NUMINAMATH_CALUDE_book_price_percentage_l3647_364773

theorem book_price_percentage (suggested_retail_price : ℝ) 
  (h1 : suggested_retail_price > 0) : 
  let marked_price := 0.6 * suggested_retail_price
  let alice_paid := 0.6 * marked_price
  alice_paid / suggested_retail_price = 0.36 := by
sorry

end NUMINAMATH_CALUDE_book_price_percentage_l3647_364773


namespace NUMINAMATH_CALUDE_turning_process_terminates_l3647_364765

/-- Represents the direction a soldier is facing -/
inductive Direction
  | East
  | West

/-- Represents the state of the line of soldiers -/
def SoldierLine := List Direction

/-- Performs one step of the turning process -/
def turn_step (line : SoldierLine) : SoldierLine :=
  sorry

/-- Checks if the line is stable (no more turns needed) -/
def is_stable (line : SoldierLine) : Prop :=
  sorry

/-- The main theorem: the turning process will eventually stop -/
theorem turning_process_terminates (initial_line : SoldierLine) :
  ∃ (n : ℕ) (final_line : SoldierLine), 
    (n.iterate turn_step initial_line = final_line) ∧ is_stable final_line :=
  sorry

end NUMINAMATH_CALUDE_turning_process_terminates_l3647_364765


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3647_364764

theorem quadratic_inequality_solution (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (x - a) * (x - 1/a) > 0} = {x : ℝ | x < a ∨ x > 1/a} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3647_364764


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3647_364702

theorem quadratic_inequality_solution (x : ℝ) :
  -3 * x^2 + 5 * x + 4 < 0 ↔ -4/3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3647_364702


namespace NUMINAMATH_CALUDE_exists_n_in_sequence_l3647_364732

theorem exists_n_in_sequence (a : ℕ → ℕ) : (∀ n, a n = n^2 + n) → ∃ n, a n = 30 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_in_sequence_l3647_364732


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l3647_364762

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 2)^2 + (y - 3)^2 + (z - 6)^2 = 0 → x + y + z = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l3647_364762
