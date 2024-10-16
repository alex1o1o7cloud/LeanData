import Mathlib

namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1905_190598

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem necessary_not_sufficient_condition (a b : ℝ) :
  let z : ℂ := ⟨a, b⟩
  (is_pure_imaginary z → a = 0) ∧
  ¬(a = 0 → is_pure_imaginary z) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1905_190598


namespace NUMINAMATH_CALUDE_unique_k_with_infinite_k_numbers_l1905_190540

/-- Definition of a k-number -/
def is_k_number (k n : ℕ) : Prop :=
  ∃ (r m : ℕ), r > 0 ∧ m > 0 ∧ n = r * (r + k) ∧ n = m^2 - k

/-- There are infinitely many k-numbers -/
def infinitely_many_k_numbers (k : ℕ) : Prop :=
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ is_k_number k n

/-- Theorem: k = 4 is the only positive integer with infinitely many k-numbers -/
theorem unique_k_with_infinite_k_numbers :
  ∀ k : ℕ, k > 0 → (infinitely_many_k_numbers k ↔ k = 4) :=
sorry

end NUMINAMATH_CALUDE_unique_k_with_infinite_k_numbers_l1905_190540


namespace NUMINAMATH_CALUDE_circle_and_tangent_lines_l1905_190566

-- Define the points
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (1, -2)
def N : ℝ × ℝ := (5, 3)

-- Define the line l: 2x + y - 4 = 0
def l (x y : ℝ) : Prop := 2 * x + y - 4 = 0

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 3)^2 + (y + 2)^2 = 4

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = 5
def tangent_line_2 (x y : ℝ) : Prop := 21 * x - 20 * y - 45 = 0

theorem circle_and_tangent_lines :
  ∃ (center_x center_y : ℝ),
    -- The center of M lies on line l
    l center_x center_y ∧
    -- M passes through A and B
    circle_M A.1 A.2 ∧ circle_M B.1 B.2 ∧
    -- The tangent lines pass through N and are tangent to M
    (tangent_line_1 N.1 ∨ tangent_line_2 N.1 N.2) ∧
    (∀ x y, tangent_line_1 x ∨ tangent_line_2 x y → 
      ((x - center_x)^2 + (y - center_y)^2 = 4 → x = N.1 ∧ y = N.2)) :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangent_lines_l1905_190566


namespace NUMINAMATH_CALUDE_ellipse_equation_l1905_190551

/-- An ellipse with major axis three times the minor axis and focal distance 8 -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Half focal distance
  h_major_minor : a = 3 * b
  h_focal : c = 4
  h_positive : a > 0 ∧ b > 0
  h_ellipse : a^2 = b^2 + c^2

/-- The standard equation of the ellipse is either x²/18 + y²/2 = 1 or y²/18 + x²/2 = 1 -/
theorem ellipse_equation (e : Ellipse) :
  (∀ x y : ℝ, x^2 / 18 + y^2 / 2 = 1) ∨ (∀ x y : ℝ, y^2 / 18 + x^2 / 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1905_190551


namespace NUMINAMATH_CALUDE_expression_equals_four_l1905_190537

theorem expression_equals_four :
  2 * Real.cos (30 * π / 180) + (-1/2)⁻¹ + |Real.sqrt 3 - 2| + (2 * Real.sqrt (9/4))^0 + Real.sqrt 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_four_l1905_190537


namespace NUMINAMATH_CALUDE_hotel_meal_expenditure_l1905_190561

theorem hotel_meal_expenditure (num_persons : ℕ) (regular_cost : ℕ) (extra_cost : ℕ) (total_cost : ℕ) : 
  num_persons = 9 →
  regular_cost = 12 →
  extra_cost = 8 →
  total_cost = 117 →
  ∃ (x : ℕ), (num_persons - 1) * regular_cost + (x + extra_cost) = total_cost ∧ x = 13 := by
sorry

end NUMINAMATH_CALUDE_hotel_meal_expenditure_l1905_190561


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1905_190560

def f (x : ℝ) : ℝ := (x - 2)^2 + 1

theorem quadratic_inequality : f 2 < f 3 ∧ f 3 < f 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1905_190560


namespace NUMINAMATH_CALUDE_last_three_average_l1905_190511

theorem last_three_average (list : List ℝ) (h1 : list.length = 7) 
  (h2 : list.sum / 7 = 62) (h3 : (list.take 4).sum / 4 = 58) : 
  (list.drop 4).sum / 3 = 202 / 3 := by
  sorry

end NUMINAMATH_CALUDE_last_three_average_l1905_190511


namespace NUMINAMATH_CALUDE_average_of_4_8_N_l1905_190510

theorem average_of_4_8_N (N : ℝ) (h : 7 < N ∧ N < 15) : 
  let avg := (4 + 8 + N) / 3
  avg = 7 ∨ avg = 9 := by
sorry

end NUMINAMATH_CALUDE_average_of_4_8_N_l1905_190510


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1905_190528

theorem fixed_point_of_exponential_function :
  let f : ℝ → ℝ := λ x => 2^(x + 2) + 1
  f (-2) = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1905_190528


namespace NUMINAMATH_CALUDE_hyperbola_specific_equation_l1905_190536

/-- Represents a hyperbola with center at the origin -/
structure Hyperbola where
  /-- The distance from the center to a focus -/
  c : ℝ
  /-- The slope of the asymptotes -/
  m : ℝ

/-- The equation of the hyperbola given its parameters -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / (h.c^2 / (1 + h.m^2)) - y^2 / (h.c^2 * h.m^2 / (1 + h.m^2)) = 1

theorem hyperbola_specific_equation :
  let h : Hyperbola := ⟨5, 3/4⟩
  ∀ x y : ℝ, hyperbola_equation h x y ↔ x^2/16 - y^2/9 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_specific_equation_l1905_190536


namespace NUMINAMATH_CALUDE_sports_club_overlap_l1905_190555

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) 
  (h1 : total = 42)
  (h2 : badminton = 20)
  (h3 : tennis = 23)
  (h4 : neither = 6) :
  badminton + tennis - (total - neither) = 7 :=
by sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l1905_190555


namespace NUMINAMATH_CALUDE_budget_allocation_l1905_190512

theorem budget_allocation (microphotonics : ℝ) (home_electronics : ℝ) (food_additives : ℝ) 
  (genetically_modified : ℝ) (basic_astrophysics_degrees : ℝ) 
  (h1 : microphotonics = 13)
  (h2 : home_electronics = 24)
  (h3 : food_additives = 15)
  (h4 : genetically_modified = 29)
  (h5 : basic_astrophysics_degrees = 39.6) :
  100 - (microphotonics + home_electronics + food_additives + genetically_modified + 
    (basic_astrophysics_degrees / 360 * 100)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_budget_allocation_l1905_190512


namespace NUMINAMATH_CALUDE_max_nickels_in_jar_l1905_190519

theorem max_nickels_in_jar (total_nickels : ℕ) (jar_score : ℕ) (ground_score : ℕ) (final_score : ℕ) :
  total_nickels = 40 →
  jar_score = 5 →
  ground_score = 2 →
  final_score = 88 →
  ∃ (jar_nickels ground_nickels : ℕ),
    jar_nickels + ground_nickels = total_nickels ∧
    jar_score * jar_nickels - ground_score * ground_nickels = final_score ∧
    jar_nickels ≤ 24 ∧
    (∀ (x : ℕ), x > 24 →
      ¬(∃ (y : ℕ), x + y = total_nickels ∧
        jar_score * x - ground_score * y = final_score)) :=
by sorry

end NUMINAMATH_CALUDE_max_nickels_in_jar_l1905_190519


namespace NUMINAMATH_CALUDE_peacock_count_l1905_190544

theorem peacock_count (total_legs total_heads : ℕ) 
  (peacock_legs peacock_heads rabbit_legs rabbit_heads : ℕ) :
  total_legs = 32 →
  total_heads = 12 →
  peacock_legs = 2 →
  peacock_heads = 1 →
  rabbit_legs = 4 →
  rabbit_heads = 1 →
  ∃ (num_peacocks num_rabbits : ℕ),
    num_peacocks * peacock_legs + num_rabbits * rabbit_legs = total_legs ∧
    num_peacocks * peacock_heads + num_rabbits * rabbit_heads = total_heads ∧
    num_peacocks = 8 :=
by sorry

end NUMINAMATH_CALUDE_peacock_count_l1905_190544


namespace NUMINAMATH_CALUDE_equation_solution_l1905_190590

theorem equation_solution : ∃! x : ℝ, x ≠ 1 ∧ (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1905_190590


namespace NUMINAMATH_CALUDE_gcd_problem_l1905_190532

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 570 * k) :
  Int.gcd (4 * b^3 + 2 * b^2 + 5 * b + 171) b = 171 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1905_190532


namespace NUMINAMATH_CALUDE_total_snowman_drawings_l1905_190526

/-- The number of cards Melody made -/
def num_cards : ℕ := 13

/-- The number of snowman drawings on each card -/
def drawings_per_card : ℕ := 4

/-- The total number of snowman drawings printed -/
def total_drawings : ℕ := num_cards * drawings_per_card

theorem total_snowman_drawings : total_drawings = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_snowman_drawings_l1905_190526


namespace NUMINAMATH_CALUDE_area_of_region_l1905_190501

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 32 ∧ 
   A = Real.pi * (Real.sqrt ((x + 4)^2 + (y + 5)^2))^2 ∧
   x^2 + y^2 + 8*x + 10*y = -9) := by
sorry

end NUMINAMATH_CALUDE_area_of_region_l1905_190501


namespace NUMINAMATH_CALUDE_ice_cream_unsold_l1905_190500

theorem ice_cream_unsold (chocolate mango vanilla strawberry : ℕ)
  (h_chocolate : chocolate = 50)
  (h_mango : mango = 54)
  (h_vanilla : vanilla = 80)
  (h_strawberry : strawberry = 40)
  (sold_chocolate : ℚ)
  (sold_mango : ℚ)
  (sold_vanilla : ℚ)
  (sold_strawberry : ℚ)
  (h_sold_chocolate : sold_chocolate = 3 / 5)
  (h_sold_mango : sold_mango = 2 / 3)
  (h_sold_vanilla : sold_vanilla = 3 / 4)
  (h_sold_strawberry : sold_strawberry = 5 / 8) :
  chocolate - Int.floor (sold_chocolate * chocolate) +
  mango - Int.floor (sold_mango * mango) +
  vanilla - Int.floor (sold_vanilla * vanilla) +
  strawberry - Int.floor (sold_strawberry * strawberry) = 73 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_unsold_l1905_190500


namespace NUMINAMATH_CALUDE_seven_lines_29_regions_l1905_190557

/-- The number of regions created by n lines in a plane, where no two lines are parallel and no three are concurrent -/
def num_regions (n : ℕ) : ℕ := 1 + n + (n * (n - 1) / 2)

/-- Seven straight lines in a plane with no two parallel and no three concurrent divide the plane into 29 regions -/
theorem seven_lines_29_regions : num_regions 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_seven_lines_29_regions_l1905_190557


namespace NUMINAMATH_CALUDE_expression_evaluation_l1905_190584

theorem expression_evaluation (m : ℤ) : 
  m = -1 → (6 * m^2 - m + 3) + (-5 * m^2 + 2 * m + 1) = 4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1905_190584


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l1905_190502

theorem quadratic_equation_result (x : ℝ) (h : x^2 - 3*x = 4) : 3*x^2 - 9*x + 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l1905_190502


namespace NUMINAMATH_CALUDE_log_inequality_l1905_190521

theorem log_inequality (x : ℝ) (h : 2 * Real.log x / Real.log 2 - 1 < 0) : 0 < x ∧ x < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1905_190521


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1905_190547

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 - 5 * p - 7 = 0) → 
  (3 * q^2 - 5 * q - 7 = 0) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1905_190547


namespace NUMINAMATH_CALUDE_power_multiplication_l1905_190564

theorem power_multiplication (m : ℝ) : (m^2)^3 * m^4 = m^10 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1905_190564


namespace NUMINAMATH_CALUDE_hyperbola_sufficient_condition_l1905_190507

-- Define the equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / (m - 1) + y^2 / (4 - m) = 1

-- Define the condition for a hyperbola with foci on the x-axis
def is_hyperbola_x_axis (m : ℝ) : Prop :=
  m - 1 > 0 ∧ 4 - m < 0

-- The theorem to prove
theorem hyperbola_sufficient_condition :
  ∃ (m : ℝ), m > 5 → is_hyperbola_x_axis m ∧
  ∃ (m' : ℝ), is_hyperbola_x_axis m' ∧ m' ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_sufficient_condition_l1905_190507


namespace NUMINAMATH_CALUDE_author_paperback_percentage_is_six_percent_l1905_190558

/-- Represents the book sales problem --/
structure BookSales where
  paperback_copies : ℕ
  paperback_price : ℚ
  hardcover_copies : ℕ
  hardcover_price : ℚ
  hardcover_percentage : ℚ
  total_earnings : ℚ

/-- Calculates the author's percentage from paperback sales --/
def paperback_percentage (sales : BookSales) : ℚ :=
  let paperback_sales := sales.paperback_copies * sales.paperback_price
  let hardcover_sales := sales.hardcover_copies * sales.hardcover_price
  let hardcover_earnings := sales.hardcover_percentage * hardcover_sales
  let paperback_earnings := sales.total_earnings - hardcover_earnings
  paperback_earnings / paperback_sales

/-- Theorem stating that the author's percentage from paperback sales is 6% --/
theorem author_paperback_percentage_is_six_percent (sales : BookSales) 
  (h1 : sales.paperback_copies = 32000)
  (h2 : sales.paperback_price = 1/5)
  (h3 : sales.hardcover_copies = 15000)
  (h4 : sales.hardcover_price = 2/5)
  (h5 : sales.hardcover_percentage = 12/100)
  (h6 : sales.total_earnings = 1104) :
  paperback_percentage sales = 6/100 := by
  sorry


end NUMINAMATH_CALUDE_author_paperback_percentage_is_six_percent_l1905_190558


namespace NUMINAMATH_CALUDE_jason_joining_age_l1905_190577

/-- Calculates Jason's age when he joined the military given his career progression --/
theorem jason_joining_age (years_to_chief : ℕ) (years_chief_to_master : ℕ) (years_after_master : ℕ) (retirement_age : ℕ) : 
  years_to_chief = 8 →
  years_chief_to_master = years_to_chief + years_to_chief / 4 →
  years_after_master = 10 →
  retirement_age = 46 →
  retirement_age - (years_to_chief + years_chief_to_master + years_after_master) = 18 := by
sorry

end NUMINAMATH_CALUDE_jason_joining_age_l1905_190577


namespace NUMINAMATH_CALUDE_divisors_of_power_minus_one_l1905_190516

/-- The number of distinct positive divisors of a positive integer -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- Main theorem -/
theorem divisors_of_power_minus_one (a n : ℕ) (ha : a > 1) (hn : n > 0) 
  (h_prime : Nat.Prime (a^n + 1)) : num_divisors (a^n - 1) ≥ n := by
  sorry


end NUMINAMATH_CALUDE_divisors_of_power_minus_one_l1905_190516


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1905_190508

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) (m : ℕ) :
  arithmetic_sequence a d →
  d ≠ 0 →
  a 3 + a 6 + a 10 + a 13 = 32 →
  a m = 8 →
  m = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1905_190508


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l1905_190585

theorem no_solution_for_equation : 
  ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (1 / a + 1 / b = 1 / (2 * a + 3 * b)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l1905_190585


namespace NUMINAMATH_CALUDE_sunflowers_per_packet_l1905_190515

theorem sunflowers_per_packet (eggplants_per_packet : ℕ) (eggplant_packets : ℕ) (sunflower_packets : ℕ) (total_plants : ℕ) :
  eggplants_per_packet = 14 →
  eggplant_packets = 4 →
  sunflower_packets = 6 →
  total_plants = 116 →
  total_plants = eggplants_per_packet * eggplant_packets + sunflower_packets * (total_plants - eggplants_per_packet * eggplant_packets) / sunflower_packets →
  (total_plants - eggplants_per_packet * eggplant_packets) / sunflower_packets = 10 :=
by sorry

end NUMINAMATH_CALUDE_sunflowers_per_packet_l1905_190515


namespace NUMINAMATH_CALUDE_game_tie_fraction_l1905_190574

theorem game_tie_fraction (jack_wins emily_wins : ℚ) 
  (h1 : jack_wins = 5 / 12)
  (h2 : emily_wins = 1 / 4) : 
  1 - (jack_wins + emily_wins) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_game_tie_fraction_l1905_190574


namespace NUMINAMATH_CALUDE_vector_operations_and_parallelism_l1905_190594

/-- Given three vectors in R², prove the results of vector operations and parallelism condition. -/
theorem vector_operations_and_parallelism 
  (a b c : ℝ × ℝ) 
  (ha : a = (3, 2)) 
  (hb : b = (-1, 2)) 
  (hc : c = (4, 1)) : 
  (3 • a + b - 2 • c = (0, 6)) ∧ 
  (∃ k : ℝ, k = -16/13 ∧ ∃ t : ℝ, t • (a + k • c) = 2 • b - a) := by
sorry


end NUMINAMATH_CALUDE_vector_operations_and_parallelism_l1905_190594


namespace NUMINAMATH_CALUDE_smallest_integer_above_sqrt5_plus_sqrt3_to_6th_l1905_190591

theorem smallest_integer_above_sqrt5_plus_sqrt3_to_6th (x : ℝ) :
  x = (Real.sqrt 5 + Real.sqrt 3)^6 → ⌈x⌉ = 3323 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_above_sqrt5_plus_sqrt3_to_6th_l1905_190591


namespace NUMINAMATH_CALUDE_bills_found_l1905_190513

def initial_amount : ℕ := 75
def final_amount : ℕ := 135
def bill_value : ℕ := 20

theorem bills_found : (final_amount - initial_amount) / bill_value = 3 := by
  sorry

end NUMINAMATH_CALUDE_bills_found_l1905_190513


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l1905_190582

/-- A hyperbola with equation mx^2 + y^2 = 1 -/
structure Hyperbola (m : ℝ) where
  equation : ∀ x y : ℝ, m * x^2 + y^2 = 1

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola m) : ℝ := sorry

/-- The slope of an asymptote of a hyperbola -/
def asymptote_slope (h : Hyperbola m) : ℝ := sorry

theorem hyperbola_m_value (m : ℝ) (h : Hyperbola m) :
  (∃ k : ℝ, k > 0 ∧ eccentricity h = 2 * k ∧ asymptote_slope h = k) →
  m = -3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l1905_190582


namespace NUMINAMATH_CALUDE_probability_inside_circle_l1905_190531

def is_inside_circle (x y : ℕ) : Prop := x^2 + y^2 < 9

def favorable_outcomes : ℕ := 4

def total_outcomes : ℕ := 36

theorem probability_inside_circle :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 9 :=
sorry

end NUMINAMATH_CALUDE_probability_inside_circle_l1905_190531


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1905_190588

theorem ratio_of_sum_and_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1905_190588


namespace NUMINAMATH_CALUDE_range_of_quadratic_expression_l1905_190539

theorem range_of_quadratic_expression (x : ℝ) :
  ((x - 1) * (x - 2) < 2) →
  ∃ y, y = (x + 1) * (x - 3) ∧ -4 ≤ y ∧ y < 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_quadratic_expression_l1905_190539


namespace NUMINAMATH_CALUDE_prime_power_sum_l1905_190535

theorem prime_power_sum (a b c d e : ℕ) : 
  2^a * 3^b * 5^c * 7^d * 11^e = 6930 → 2*a + 3*b + 5*c + 7*d + 11*e = 31 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_l1905_190535


namespace NUMINAMATH_CALUDE_volunteer_team_statistics_l1905_190527

def frequencies : List ℕ := [10, 10, 10, 8, 8, 8, 8, 7, 7, 4]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

def mean (l : List ℕ) : ℚ := sorry

theorem volunteer_team_statistics :
  mode frequencies = 8 ∧
  median frequencies = 8 ∧
  mean frequencies = 8 := by sorry

end NUMINAMATH_CALUDE_volunteer_team_statistics_l1905_190527


namespace NUMINAMATH_CALUDE_percentage_materialB_in_final_mixture_l1905_190506

/-- Represents a mixture of oil and material B -/
structure Mixture where
  total : ℝ
  oil : ℝ
  materialB : ℝ

/-- The initial mixture A -/
def initialMixtureA : Mixture :=
  { total := 8
    oil := 8 * 0.2
    materialB := 8 * 0.8 }

/-- The mixture after adding 2 kg of oil -/
def mixtureAfterOil : Mixture :=
  { total := initialMixtureA.total + 2
    oil := initialMixtureA.oil + 2
    materialB := initialMixtureA.materialB }

/-- The additional 6 kg of mixture A -/
def additionalMixtureA : Mixture :=
  { total := 6
    oil := 6 * 0.2
    materialB := 6 * 0.8 }

/-- The final mixture -/
def finalMixture : Mixture :=
  { total := mixtureAfterOil.total + additionalMixtureA.total
    oil := mixtureAfterOil.oil + additionalMixtureA.oil
    materialB := mixtureAfterOil.materialB + additionalMixtureA.materialB }

theorem percentage_materialB_in_final_mixture :
  finalMixture.materialB / finalMixture.total = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_percentage_materialB_in_final_mixture_l1905_190506


namespace NUMINAMATH_CALUDE_at_least_seven_stay_probability_l1905_190529

def total_friends : ℕ := 8
def sure_friends : ℕ := 3
def unsure_friends : ℕ := 5
def stay_probability : ℚ := 1/3

def probability_at_least_seven_stay : ℚ :=
  (Nat.choose unsure_friends 4 * stay_probability^4 * (1 - stay_probability)^1) +
  (stay_probability^5)

theorem at_least_seven_stay_probability :
  probability_at_least_seven_stay = 11/243 :=
sorry

end NUMINAMATH_CALUDE_at_least_seven_stay_probability_l1905_190529


namespace NUMINAMATH_CALUDE_red_cells_count_l1905_190593

/-- Represents the dimensions of the grid -/
structure GridDim where
  rows : Nat
  cols : Nat

/-- Represents the painter's movement -/
structure Movement where
  left : Nat
  down : Nat

/-- Calculates the number of distinct cells visited before returning to the start -/
def distinctCellsVisited (dim : GridDim) (move : Movement) : Nat :=
  Nat.lcm dim.rows dim.cols

/-- The main theorem stating the number of red cells on the grid -/
theorem red_cells_count (dim : GridDim) (move : Movement) 
  (h1 : dim.rows = 2000) 
  (h2 : dim.cols = 70) 
  (h3 : move.left = 1) 
  (h4 : move.down = 1) : 
  distinctCellsVisited dim move = 14000 := by
  sorry

#eval distinctCellsVisited ⟨2000, 70⟩ ⟨1, 1⟩

end NUMINAMATH_CALUDE_red_cells_count_l1905_190593


namespace NUMINAMATH_CALUDE_negative_sqrt_six_squared_equals_six_l1905_190565

theorem negative_sqrt_six_squared_equals_six : (-Real.sqrt 6)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_six_squared_equals_six_l1905_190565


namespace NUMINAMATH_CALUDE_coles_return_speed_coles_return_speed_is_105_l1905_190581

/-- Calculates the average speed of the return trip given the conditions of Cole's journey -/
theorem coles_return_speed (speed_to_work : ℝ) (total_time : ℝ) (time_to_work : ℝ) : ℝ :=
  let distance_to_work := speed_to_work * (time_to_work / 60)
  let time_back_home := total_time - (time_to_work / 60)
  distance_to_work / time_back_home

/-- Proves that Cole's average speed driving back home is 105 km/h -/
theorem coles_return_speed_is_105 :
  coles_return_speed 75 6 210 = 105 := by
  sorry

end NUMINAMATH_CALUDE_coles_return_speed_coles_return_speed_is_105_l1905_190581


namespace NUMINAMATH_CALUDE_quadratic_equation_root_difference_l1905_190579

theorem quadratic_equation_root_difference (k : ℝ) : 
  (∃ a b : ℝ, 3 * a^2 + 2 * a + k = 0 ∧ 
              3 * b^2 + 2 * b + k = 0 ∧ 
              |a - b| = (a^2 + b^2).sqrt) ↔ 
  (k = 0 ∨ k = -4/15) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_difference_l1905_190579


namespace NUMINAMATH_CALUDE_equation_solution_l1905_190545

theorem equation_solution :
  ∃ y : ℚ, (4 / 7) * (1 / 5) * y - 2 = 14 ∧ y = 140 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1905_190545


namespace NUMINAMATH_CALUDE_bob_painting_fraction_l1905_190514

-- Define the time it takes Bob to paint a whole house
def full_painting_time : ℕ := 60

-- Define the time we want to calculate the fraction for
def partial_painting_time : ℕ := 15

-- Theorem statement
theorem bob_painting_fraction :
  (partial_painting_time : ℚ) / full_painting_time = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_bob_painting_fraction_l1905_190514


namespace NUMINAMATH_CALUDE_car_speed_problem_l1905_190553

theorem car_speed_problem (d : ℝ) (v : ℝ) (h1 : d > 0) (h2 : v > 0) :
  let t := d / v
  let return_time := 2 * t
  let total_distance := 2 * d
  let total_time := t + return_time
  (total_distance / total_time = 30) → v = 45 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1905_190553


namespace NUMINAMATH_CALUDE_max_sum_of_square_roots_l1905_190522

theorem max_sum_of_square_roots (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 2013) : 
  Real.sqrt (3 * a + 12) + Real.sqrt (3 * b + 12) + Real.sqrt (3 * c + 12) ≤ 135 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_square_roots_l1905_190522


namespace NUMINAMATH_CALUDE_intersection_circle_line_l1905_190534

theorem intersection_circle_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | y = x + 1}
  let intersection := circle ∩ line
  intersection = {(-1, 0), (0, 1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_circle_line_l1905_190534


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1905_190530

theorem quadratic_no_real_roots : 
  ∀ (x : ℝ), x^2 - 2*x + 3 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1905_190530


namespace NUMINAMATH_CALUDE_movement_increases_dimension_l1905_190568

/-- Dimension of geometric objects -/
inductive GeometricDimension
  | point
  | line
  | surface
  deriving Repr

/-- Function that returns the dimension of the object formed by moving an object of a given dimension -/
def dimensionAfterMovement (d : GeometricDimension) : GeometricDimension :=
  match d with
  | GeometricDimension.point => GeometricDimension.line
  | GeometricDimension.line => GeometricDimension.surface
  | GeometricDimension.surface => GeometricDimension.surface

/-- Theorem stating that moving a point forms a line and moving a line forms a surface -/
theorem movement_increases_dimension :
  (dimensionAfterMovement GeometricDimension.point = GeometricDimension.line) ∧
  (dimensionAfterMovement GeometricDimension.line = GeometricDimension.surface) :=
by sorry

end NUMINAMATH_CALUDE_movement_increases_dimension_l1905_190568


namespace NUMINAMATH_CALUDE_distinct_projections_exist_l1905_190542

/-- Represents a student's marks as a point in 12-dimensional space -/
def Student := Fin 12 → ℝ

/-- The set of 7 students -/
def Students := Fin 7 → Student

theorem distinct_projections_exist (students : Students) 
  (h : ∀ i j, i ≠ j → students i ≠ students j) :
  ∃ (subjects : Fin 6 → Fin 12), 
    ∀ i j, i ≠ j → 
      ∃ k, (students i (subjects k)) ≠ (students j (subjects k)) := by
  sorry

end NUMINAMATH_CALUDE_distinct_projections_exist_l1905_190542


namespace NUMINAMATH_CALUDE_square_ratio_side_length_sum_l1905_190575

theorem square_ratio_side_length_sum (area_ratio : ℚ) :
  area_ratio = 27 / 50 →
  ∃ (a b c : ℕ), 
    (a * (b.sqrt : ℝ) / c : ℝ) = (area_ratio : ℝ).sqrt ∧
    a = 3 ∧ b = 6 ∧ c = 10 ∧
    a + b + c = 19 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_sum_l1905_190575


namespace NUMINAMATH_CALUDE_max_M_value_l1905_190533

/-- J is a function that takes a natural number m and returns 10^5 + m -/
def J (m : ℕ) : ℕ := 10^5 + m

/-- M is a function that takes a natural number a and returns the number of factors of 2
    in the prime factorization of J(2^a) -/
def M (a : ℕ) : ℕ := (J (2^a)).factors.count 2

/-- The maximum value of M(a) for a ≥ 0 is 5 -/
theorem max_M_value : ∃ (k : ℕ), k = 5 ∧ ∀ (a : ℕ), M a ≤ k :=
sorry

end NUMINAMATH_CALUDE_max_M_value_l1905_190533


namespace NUMINAMATH_CALUDE_prob_red_card_standard_deck_l1905_190505

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (red_suits : ℕ)
  (cards_per_suit : ℕ)

/-- A standard deck of 52 cards -/
def standard_deck : Deck :=
  { total_cards := 52,
    ranks := 13,
    suits := 4,
    red_suits := 2,
    cards_per_suit := 13 }

/-- The probability of drawing a red suit card from the top of a randomly shuffled deck -/
def prob_red_card (d : Deck) : ℚ :=
  (d.red_suits * d.cards_per_suit) / d.total_cards

/-- Theorem stating that the probability of drawing a red suit card from a standard deck is 1/2 -/
theorem prob_red_card_standard_deck :
  prob_red_card standard_deck = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_card_standard_deck_l1905_190505


namespace NUMINAMATH_CALUDE_expression_evaluation_l1905_190518

theorem expression_evaluation (a x : ℝ) (h1 : a = x^2) (h2 : a = Real.sqrt 2) :
  4 * a^3 / (x^4 + a^4) + 1 / (a + x) + 2 * a / (x^2 + a^2) + 1 / (a - x) = 16 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1905_190518


namespace NUMINAMATH_CALUDE_runners_in_picture_probability_l1905_190596

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ  -- Time to complete one lap in seconds
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Calculates the probability of both runners being in the picture -/
def probability_both_in_picture (runner1 runner2 : Runner) (pictureTime : ℝ) : ℚ :=
  sorry

/-- Theorem stating the probability of both runners being in the picture -/
theorem runners_in_picture_probability 
  (runner1 : Runner) 
  (runner2 : Runner) 
  (pictureTime : ℝ) 
  (h1 : runner1.lapTime = 100)
  (h2 : runner2.lapTime = 75)
  (h3 : runner1.direction = true)
  (h4 : runner2.direction = false)
  (h5 : 720 ≤ pictureTime ∧ pictureTime ≤ 780) :
  probability_both_in_picture runner1 runner2 pictureTime = 111 / 200 :=
by sorry

end NUMINAMATH_CALUDE_runners_in_picture_probability_l1905_190596


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l1905_190504

theorem geometric_progression_fourth_term (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 2^(1/4)) (h₂ : a₂ = 2^(1/6)) (h₃ : a₃ = 2^(1/12)) :
  let r := a₂ / a₁
  a₃ * r = 1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l1905_190504


namespace NUMINAMATH_CALUDE_distance_equals_scientific_notation_l1905_190580

/-- Represents the distance in kilometers -/
def distance : ℝ := 30000000

/-- Represents the scientific notation of the distance -/
def scientific_notation : ℝ := 3 * (10 ^ 7)

/-- Theorem stating that the distance is equal to its scientific notation representation -/
theorem distance_equals_scientific_notation : distance = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_distance_equals_scientific_notation_l1905_190580


namespace NUMINAMATH_CALUDE_trevor_remaining_eggs_l1905_190562

def chicken_eggs : List Nat := [4, 3, 2, 2, 5, 1, 3]

def total_eggs : Nat := chicken_eggs.sum

theorem trevor_remaining_eggs :
  total_eggs - 2 - 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_trevor_remaining_eggs_l1905_190562


namespace NUMINAMATH_CALUDE_dividend_divisor_quotient_l1905_190589

theorem dividend_divisor_quotient (x y z : ℚ) :
  x = y * z + 15 ∧ y = 25 ∧ 3 * x - 4 * y + 2 * z = 0 →
  x = 230 / 7 ∧ z = 5 / 7 := by
sorry

end NUMINAMATH_CALUDE_dividend_divisor_quotient_l1905_190589


namespace NUMINAMATH_CALUDE_range_of_a_l1905_190559

def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (a - 1) * x < (a - 1) * y

def q (a : ℝ) : Prop := ∀ x : ℝ, -x^2 + 2*x - 2 ≤ a

theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Icc (-1 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1905_190559


namespace NUMINAMATH_CALUDE_limit_f_limit_φ_l1905_190563

-- Function f(x)
def f (x : ℝ) : ℝ := x^3 - 5*x^2 + 2*x + 4

-- Function φ(t)
noncomputable def φ (t : ℝ) : ℝ := t * Real.sqrt (t^2 - 20) - Real.log (t + Real.sqrt (t^2 - 20)) / Real.log 10

-- Theorem for the limit of f(x) as x → -3
theorem limit_f : 
  Filter.Tendsto f (nhds (-3)) (nhds (-74)) :=
sorry

-- Theorem for the limit of φ(t) as t → 6
theorem limit_φ :
  Filter.Tendsto φ (nhds 6) (nhds 23) :=
sorry

end NUMINAMATH_CALUDE_limit_f_limit_φ_l1905_190563


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1905_190595

/-- Given an arithmetic sequence {a_n} where a_4 = 4, prove that S_7 = 28 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = n / 2 * (a 1 + a n)) →  -- Definition of S_n
  (∀ k m, a (k + m) - a k = m * (a 2 - a 1)) →  -- Definition of arithmetic sequence
  a 4 = 4 →  -- Given condition
  S 7 = 28 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1905_190595


namespace NUMINAMATH_CALUDE_monogram_count_l1905_190573

theorem monogram_count : ∀ n : ℕ, n = 12 → (n.choose 2) = 66 := by
  sorry

end NUMINAMATH_CALUDE_monogram_count_l1905_190573


namespace NUMINAMATH_CALUDE_tenth_grade_enrollment_l1905_190538

/-- Represents the number of students enrolled only in science class -/
def students_only_science (total_students science_students art_students : ℕ) : ℕ :=
  science_students - (science_students + art_students - total_students)

/-- Theorem stating that given the conditions, 65 students are enrolled only in science class -/
theorem tenth_grade_enrollment (total_students science_students art_students : ℕ) 
  (h1 : total_students = 140)
  (h2 : science_students = 100)
  (h3 : art_students = 75) :
  students_only_science total_students science_students art_students = 65 := by
  sorry

#eval students_only_science 140 100 75

end NUMINAMATH_CALUDE_tenth_grade_enrollment_l1905_190538


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_1_range_of_a_for_f_always_greater_than_1_l1905_190567

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x + 4| + |x - a|

-- Theorem for part I
theorem solution_set_for_a_eq_1 :
  {x : ℝ | f 1 x ≤ 5} = {x : ℝ | -8/3 ≤ x ∧ x ≤ 0} := by sorry

-- Theorem for part II
theorem range_of_a_for_f_always_greater_than_1 :
  ∀ a : ℝ, (∀ x : ℝ, f a x > 1) ↔ (a < -3 ∨ a > -1) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_1_range_of_a_for_f_always_greater_than_1_l1905_190567


namespace NUMINAMATH_CALUDE_trihedral_angle_existence_l1905_190509

/-- A trihedral angle -/
structure TrihedralAngle where
  α : Real
  β : Real
  γ : Real

/-- Given three dihedral angles, there exists a trihedral angle with these angles -/
theorem trihedral_angle_existence (α β γ : Real) : 
  ∃ (T : TrihedralAngle), T.α = α ∧ T.β = β ∧ T.γ = γ := by
  sorry

end NUMINAMATH_CALUDE_trihedral_angle_existence_l1905_190509


namespace NUMINAMATH_CALUDE_ratio_from_log_difference_l1905_190572

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem ratio_from_log_difference (a b : ℝ) (h : a > 0 ∧ b > 0) :
  lg (3 * a^3) - lg (3 * b^3) = 9 → a / b = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_ratio_from_log_difference_l1905_190572


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1905_190525

theorem quadratic_equation_roots (a : ℝ) (m : ℝ) :
  let x₁ : ℝ := Real.sqrt (a + 2) - Real.sqrt (8 - a) + Real.sqrt (-a^2)
  (∃ x₂ : ℝ, (1/2) * m * x₁^2 + Real.sqrt 2 * x₁ + m^2 = 0 ∧
             (1/2) * m * x₂^2 + Real.sqrt 2 * x₂ + m^2 = 0) →
  (m = 1 ∧ x₁ = -Real.sqrt 2 ∧ x₂ = -Real.sqrt 2) ∨
  (m = -2 ∧ x₁ = -Real.sqrt 2 ∧ x₂ = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1905_190525


namespace NUMINAMATH_CALUDE_right_triangle_angles_l1905_190597

theorem right_triangle_angles (a b c R r : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_R : R = c / 2) (h_r : r = (a + b - c) / 2) (h_ratio : R / r = Real.sqrt 3 + 1) :
  ∃ (α β : ℝ), α + β = Real.pi / 2 ∧ 
  (α = Real.pi / 6 ∧ β = Real.pi / 3) ∨ (α = Real.pi / 3 ∧ β = Real.pi / 6) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_angles_l1905_190597


namespace NUMINAMATH_CALUDE_james_has_winning_strategy_l1905_190503

/-- Represents a player in the coin-choosing game -/
inductive Player : Type
| John : Player
| James : Player

/-- The state of the game at any point -/
structure GameState :=
  (coins_left : List ℕ)
  (john_kopeks : ℕ)
  (james_kopeks : ℕ)
  (current_chooser : Player)

/-- A strategy is a function that takes the current game state and returns the chosen coin -/
def Strategy := GameState → ℕ

/-- The result of the game -/
inductive GameResult
| JohnWins : GameResult
| JamesWins : GameResult
| Draw : GameResult

/-- Play the game given strategies for both players -/
def play_game (john_strategy : Strategy) (james_strategy : Strategy) : GameResult :=
  sorry

/-- A winning strategy for a player ensures they always win or draw -/
def is_winning_strategy (player : Player) (strategy : Strategy) : Prop :=
  match player with
  | Player.John => ∀ james_strategy, play_game strategy james_strategy ≠ GameResult.JamesWins
  | Player.James => ∀ john_strategy, play_game john_strategy strategy ≠ GameResult.JohnWins

/-- The main theorem: James has a winning strategy -/
theorem james_has_winning_strategy :
  ∃ (strategy : Strategy), is_winning_strategy Player.James strategy :=
sorry

end NUMINAMATH_CALUDE_james_has_winning_strategy_l1905_190503


namespace NUMINAMATH_CALUDE_just_passed_count_l1905_190541

def total_students : ℕ := 1000

def first_division_percent : ℚ := 25 / 100
def second_division_percent : ℚ := 35 / 100
def third_division_percent : ℚ := 20 / 100
def fourth_division_percent : ℚ := 10 / 100
def failed_percent : ℚ := 4 / 100

theorem just_passed_count : 
  ∃ (just_passed : ℕ), 
    just_passed = total_students - 
      (first_division_percent * total_students).num - 
      (second_division_percent * total_students).num - 
      (third_division_percent * total_students).num - 
      (fourth_division_percent * total_students).num - 
      (failed_percent * total_students).num ∧ 
    just_passed = 60 := by
  sorry

end NUMINAMATH_CALUDE_just_passed_count_l1905_190541


namespace NUMINAMATH_CALUDE_impossibleTiling_l1905_190583

/-- Represents the types of pieces that can be used for tiling -/
inductive PieceType
  | A  -- 2x2 piece with one corner square of a different color
  | B  -- L-shaped piece covering 3 unit squares
  | C  -- 2x2 piece covering one square of each of four different colors

/-- Represents a board that can be tiled -/
structure Board where
  rows : Nat
  cols : Nat

/-- Represents a tiling of a board with a specific piece type -/
structure Tiling where
  board : Board
  pieceType : PieceType
  pieceCount : Nat

/-- Checks if a tiling is valid for a given board and piece type -/
def isValidTiling (t : Tiling) : Prop :=
  t.board.rows = 10 ∧ t.board.cols = 10 ∧ t.pieceCount = 25

/-- The main theorem stating that it's impossible to tile a 10x10 board with 25 pieces of any type -/
theorem impossibleTiling (t : Tiling) : isValidTiling t → False := by
  sorry


end NUMINAMATH_CALUDE_impossibleTiling_l1905_190583


namespace NUMINAMATH_CALUDE_function_symmetry_l1905_190523

/-- The function f(x) defined as √3 sin(2x) + 2 cos²x is symmetric about the line x = π/6 -/
theorem function_symmetry (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2
  f (π / 6 + x) = f (π / 6 - x) :=
by sorry

end NUMINAMATH_CALUDE_function_symmetry_l1905_190523


namespace NUMINAMATH_CALUDE_equation_positive_root_implies_m_equals_one_l1905_190543

-- Define the equation
def equation (x m : ℝ) : Prop :=
  (x - 4) / (x - 3) - m - 4 = m / (3 - x)

-- Define the theorem
theorem equation_positive_root_implies_m_equals_one :
  (∃ x : ℝ, x > 0 ∧ equation x m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_root_implies_m_equals_one_l1905_190543


namespace NUMINAMATH_CALUDE_minimum_cost_theorem_l1905_190524

/-- Represents the number and cost of diesel generators --/
structure DieselGenerators where
  totalCount : Nat
  typeACount : Nat
  typeBCount : Nat
  typeCCount : Nat
  typeACost : Nat
  typeBCost : Nat
  typeCCost : Nat

/-- Represents the irrigation capacity of the generators --/
def irrigationCapacity (g : DieselGenerators) : Nat :=
  4 * g.typeACount + 3 * g.typeBCount + 2 * g.typeCCount

/-- Represents the total cost of operating the generators --/
def operatingCost (g : DieselGenerators) : Nat :=
  g.typeACost * g.typeACount + g.typeBCost * g.typeBCount + g.typeCCost * g.typeCCount

/-- Theorem stating the minimum cost of operation --/
theorem minimum_cost_theorem (g : DieselGenerators) :
  g.totalCount = 10 ∧
  g.typeACount > 0 ∧ g.typeBCount > 0 ∧ g.typeCCount > 0 ∧
  g.typeACount + g.typeBCount + g.typeCCount = g.totalCount ∧
  irrigationCapacity g = 32 ∧
  g.typeACost = 130 ∧ g.typeBCost = 120 ∧ g.typeCCost = 100 →
  ∃ (minCost : Nat), minCost = 1190 ∧
    ∀ (h : DieselGenerators), 
      h.totalCount = 10 ∧
      h.typeACount > 0 ∧ h.typeBCount > 0 ∧ h.typeCCount > 0 ∧
      h.typeACount + h.typeBCount + h.typeCCount = h.totalCount ∧
      irrigationCapacity h = 32 ∧
      h.typeACost = 130 ∧ h.typeBCost = 120 ∧ h.typeCCost = 100 →
      operatingCost h ≥ minCost := by
  sorry

end NUMINAMATH_CALUDE_minimum_cost_theorem_l1905_190524


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l1905_190570

theorem largest_integer_in_interval : 
  ∃ (y : ℤ), (1/4 : ℚ) < (y : ℚ)/6 ∧ (y : ℚ)/6 < 7/12 ∧ 
  ∀ (z : ℤ), (1/4 : ℚ) < (z : ℚ)/6 → (z : ℚ)/6 < 7/12 → z ≤ y :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l1905_190570


namespace NUMINAMATH_CALUDE_range_of_x_range_of_p_l1905_190548

-- Define the inequality function
def inequality (x p : ℝ) : Prop := x^2 + p*x + 1 > 2*x + p

-- Theorem 1
theorem range_of_x (p : ℝ) (h : |p| ≤ 2) :
  ∀ x, inequality x p → x < -1 ∨ x > 3 :=
sorry

-- Theorem 2
theorem range_of_p (x : ℝ) (h : 2 ≤ x ∧ x ≤ 4) :
  ∀ p, inequality x p → p > -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_p_l1905_190548


namespace NUMINAMATH_CALUDE_chess_game_probability_l1905_190552

theorem chess_game_probability (draw_prob : ℚ) (b_win_prob : ℚ) (a_win_prob : ℚ) : 
  draw_prob = 1/2 → b_win_prob = 1/3 → a_win_prob = 1 - draw_prob - b_win_prob → a_win_prob = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l1905_190552


namespace NUMINAMATH_CALUDE_num_selection_methods_l1905_190520

/-- The number of fleets --/
def num_fleets : ℕ := 7

/-- The total number of vehicles to be selected --/
def total_vehicles : ℕ := 10

/-- The minimum number of vehicles in each fleet --/
def min_vehicles_per_fleet : ℕ := 5

/-- Function to calculate the number of ways to select vehicles --/
def select_vehicles (n f t m : ℕ) : ℕ :=
  Nat.choose n 1 + n * (n - 1) + Nat.choose n 3

/-- Theorem stating the number of ways to select vehicles --/
theorem num_selection_methods :
  select_vehicles num_fleets num_fleets total_vehicles min_vehicles_per_fleet = 84 := by
  sorry


end NUMINAMATH_CALUDE_num_selection_methods_l1905_190520


namespace NUMINAMATH_CALUDE_solve_equation_l1905_190549

theorem solve_equation (C : ℝ) (h : 5 * C - 6 = 34) : C = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1905_190549


namespace NUMINAMATH_CALUDE_expression_factorization_l1905_190550

theorem expression_factorization (x : ℝ) : 
  (8 * x^4 + 34 * x^3 - 120 * x + 150) - (-2 * x^4 + 12 * x^3 - 5 * x + 10) = 
  5 * x * (2 * x^3 + (22/5) * x^2 - 23 * x + 28) := by
sorry

end NUMINAMATH_CALUDE_expression_factorization_l1905_190550


namespace NUMINAMATH_CALUDE_parabola_segment_length_l1905_190576

/-- Represents a parabola of the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Length of segment AB for a parabola with given conditions --/
theorem parabola_segment_length 
  (p : Parabola) 
  (A B : Point) 
  (h1 : A.x = -2 ∧ A.y = 0)
  (h2 : B.y = 0)
  (h3 : 2 = (A.x + B.x) / 2) -- axis of symmetry
  : abs (B.x - A.x) = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_segment_length_l1905_190576


namespace NUMINAMATH_CALUDE_episode_filming_time_increase_l1905_190517

/-- The percentage increase in filming time compared to episode duration -/
theorem episode_filming_time_increase (episode_duration : ℕ) (episodes_per_week : ℕ) (filming_time : ℕ) : 
  episode_duration = 20 →
  episodes_per_week = 5 →
  filming_time = 600 →
  (((filming_time / (episodes_per_week * 4)) - episode_duration) / episode_duration) * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_episode_filming_time_increase_l1905_190517


namespace NUMINAMATH_CALUDE_farm_area_ratio_l1905_190546

theorem farm_area_ratio :
  ∀ (s : ℝ),
  s > 0 →
  3 * s + 4 * s = 12 →
  (6 - s^2) / 6 = 145 / 147 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_area_ratio_l1905_190546


namespace NUMINAMATH_CALUDE_jill_red_packs_l1905_190587

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 18

/-- The number of packs of yellow bouncy balls Jill bought -/
def yellow_packs : ℕ := 4

/-- The additional number of red bouncy balls compared to yellow bouncy balls -/
def additional_red_balls : ℕ := 18

/-- The number of packs of red bouncy balls Jill bought -/
def red_packs : ℕ := 5

theorem jill_red_packs : 
  red_packs * balls_per_pack = yellow_packs * balls_per_pack + additional_red_balls :=
by sorry

end NUMINAMATH_CALUDE_jill_red_packs_l1905_190587


namespace NUMINAMATH_CALUDE_average_candies_sikyung_l1905_190599

def sikyung_group : Finset ℕ := {16, 22, 30, 26, 18, 20}

theorem average_candies_sikyung : 
  (sikyung_group.sum id) / sikyung_group.card = 22 := by
  sorry

end NUMINAMATH_CALUDE_average_candies_sikyung_l1905_190599


namespace NUMINAMATH_CALUDE_largest_four_digit_square_in_base7_l1905_190578

/-- The largest integer whose square has exactly 4 digits when written in base 7 -/
def M : ℕ := 48

/-- Converts a natural number to its representation in base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- Counts the number of digits in a number when represented in base 7 -/
def digitsInBase7 (n : ℕ) : ℕ := (toBase7 n).length

theorem largest_four_digit_square_in_base7 :
  M = 48 ∧ 
  digitsInBase7 (M^2) = 4 ∧
  ∀ n : ℕ, n > M → digitsInBase7 (n^2) > 4 ∧
  toBase7 M = [6, 6] := by sorry

end NUMINAMATH_CALUDE_largest_four_digit_square_in_base7_l1905_190578


namespace NUMINAMATH_CALUDE_sally_buttons_l1905_190556

/-- The number of buttons Sally needs for all shirts -/
def total_buttons (monday tuesday wednesday buttons_per_shirt : ℕ) : ℕ :=
  (monday + tuesday + wednesday) * buttons_per_shirt

/-- Theorem: Sally needs 45 buttons for all shirts -/
theorem sally_buttons : total_buttons 4 3 2 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_sally_buttons_l1905_190556


namespace NUMINAMATH_CALUDE_terminating_decimal_count_l1905_190571

theorem terminating_decimal_count : 
  (Finset.filter (fun n : ℕ => n % 13 = 0) (Finset.range 543)).card = 41 := by
  sorry

end NUMINAMATH_CALUDE_terminating_decimal_count_l1905_190571


namespace NUMINAMATH_CALUDE_range_of_a_l1905_190554

theorem range_of_a (x a : ℝ) : 
  (∀ x, x - a ≥ 1 → x ≥ 1) ∧ 
  (1 - a ≥ 1) ∧ 
  ¬(-1 - a ≥ 1) → 
  -2 < a ∧ a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1905_190554


namespace NUMINAMATH_CALUDE_elizabeth_salon_cost_l1905_190586

/-- Represents a salon visit with hair cut length and treatment cost -/
structure SalonVisit where
  cutLength : Float
  treatmentCost : Float
  discountPercentage : Float

/-- Calculate the discounted cost for a salon visit -/
def discountedCost (visit : SalonVisit) : Float :=
  visit.treatmentCost * (1 - visit.discountPercentage)

/-- Calculate the total cost of salon visits after discounts -/
def totalCost (visits : List SalonVisit) : Float :=
  visits.map discountedCost |>.sum

/-- Theorem: The total cost of Elizabeth's salon visits is $88.25 -/
theorem elizabeth_salon_cost :
  let visits : List SalonVisit := [
    { cutLength := 0.375, treatmentCost := 25, discountPercentage := 0.1 },
    { cutLength := 0.5, treatmentCost := 35, discountPercentage := 0.15 },
    { cutLength := 0.75, treatmentCost := 45, discountPercentage := 0.2 }
  ]
  totalCost visits = 88.25 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_salon_cost_l1905_190586


namespace NUMINAMATH_CALUDE_x_value_l1905_190569

theorem x_value : ∃ x : ℝ, (x = 90 * (1 + 11/100)) ∧ (x = 99.9) := by sorry

end NUMINAMATH_CALUDE_x_value_l1905_190569


namespace NUMINAMATH_CALUDE_subset_implies_m_geq_one_l1905_190592

theorem subset_implies_m_geq_one (m : ℝ) : 
  ({x : ℝ | 0 < x ∧ x < 1} ⊆ {x : ℝ | 0 < x ∧ x < m}) → m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_geq_one_l1905_190592
