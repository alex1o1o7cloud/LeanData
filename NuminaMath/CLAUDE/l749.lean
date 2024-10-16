import Mathlib

namespace NUMINAMATH_CALUDE_shopping_tax_calculation_l749_74987

theorem shopping_tax_calculation (total : ℝ) (total_positive : 0 < total) : 
  let clothing_percent : ℝ := 0.5
  let food_percent : ℝ := 0.2
  let other_percent : ℝ := 0.3
  let clothing_tax_rate : ℝ := 0.04
  let food_tax_rate : ℝ := 0
  let other_tax_rate : ℝ := 0.08
  let clothing_amount := clothing_percent * total
  let food_amount := food_percent * total
  let other_amount := other_percent * total
  let clothing_tax := clothing_tax_rate * clothing_amount
  let food_tax := food_tax_rate * food_amount
  let other_tax := other_tax_rate * other_amount
  let total_tax := clothing_tax + food_tax + other_tax
  (total_tax / total) = 0.044 := by sorry

end NUMINAMATH_CALUDE_shopping_tax_calculation_l749_74987


namespace NUMINAMATH_CALUDE_bobby_bought_two_packets_l749_74988

/-- The number of packets of candy Bobby bought -/
def bobby_candy_packets : ℕ :=
  let candies_per_packet : ℕ := 18
  let weekdays : ℕ := 5
  let weekend_days : ℕ := 2
  let weeks : ℕ := 3
  let candies_per_weekday : ℕ := 2
  let candies_per_weekend_day : ℕ := 1
  let candies_per_week : ℕ := weekdays * candies_per_weekday + weekend_days * candies_per_weekend_day
  let total_candies : ℕ := candies_per_week * weeks
  total_candies / candies_per_packet

theorem bobby_bought_two_packets : bobby_candy_packets = 2 := by
  sorry

end NUMINAMATH_CALUDE_bobby_bought_two_packets_l749_74988


namespace NUMINAMATH_CALUDE_linear_equation_power_l749_74978

theorem linear_equation_power (n m : ℕ) :
  (∃ a b c : ℝ, ∀ x y : ℝ, a * x + b * y = c ↔ 2 * x^(n - 3) - (1/3) * y^(2*m + 1) = 0) →
  n^m = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_power_l749_74978


namespace NUMINAMATH_CALUDE_mike_picked_seven_apples_l749_74943

/-- The number of apples Mike picked -/
def mike_apples : ℝ := 7.0

/-- The number of apples Nancy ate -/
def nancy_ate : ℝ := 3.0

/-- The number of apples Keith picked -/
def keith_picked : ℝ := 6.0

/-- The number of apples left -/
def apples_left : ℝ := 10.0

/-- Theorem stating that Mike picked 7.0 apples -/
theorem mike_picked_seven_apples : 
  mike_apples = mike_apples - nancy_ate + keith_picked - apples_left + apples_left :=
by sorry

end NUMINAMATH_CALUDE_mike_picked_seven_apples_l749_74943


namespace NUMINAMATH_CALUDE_seating_arrangement_probability_l749_74972

/-- The number of delegates --/
def num_delegates : ℕ := 12

/-- The number of countries --/
def num_countries : ℕ := 3

/-- The number of delegates per country --/
def delegates_per_country : ℕ := 4

/-- The probability that each delegate sits next to at least one delegate from another country --/
def seating_probability : ℚ := 21 / 22

/-- Theorem stating the probability of the seating arrangement --/
theorem seating_arrangement_probability :
  let total_arrangements := (num_delegates.factorial) / (delegates_per_country.factorial ^ num_countries)
  let unwanted_arrangements := num_countries * num_delegates * (num_delegates - delegates_per_country).factorial / 
                               (delegates_per_country.factorial ^ (num_countries - 1)) -
                               (num_countries.choose 2) * num_delegates * delegates_per_country +
                               num_delegates * (num_countries - 1)
  (total_arrangements - unwanted_arrangements) / total_arrangements = seating_probability :=
sorry

end NUMINAMATH_CALUDE_seating_arrangement_probability_l749_74972


namespace NUMINAMATH_CALUDE_team_B_is_better_l749_74948

/-- Represents the expected cost of drug development for Team A -/
def expected_cost_A (p : ℝ) (m : ℝ) : ℝ :=
  -2 * m * p^2 + 6 * m

/-- Represents the expected cost of drug development for Team B -/
def expected_cost_B (q : ℝ) (n : ℝ) : ℝ :=
  6 * n * q^3 - 9 * n * q^2 + 6 * n

/-- Theorem stating that Team B's expected cost is less than Team A's when n = 2/3m and p = q -/
theorem team_B_is_better (p q m n : ℝ) 
  (h1 : 0 < p ∧ p < 1) 
  (h2 : m > 0) 
  (h3 : n = 2/3 * m) 
  (h4 : p = q) : 
  expected_cost_B q n < expected_cost_A p m :=
sorry

end NUMINAMATH_CALUDE_team_B_is_better_l749_74948


namespace NUMINAMATH_CALUDE_geometric_arithmetic_geometric_sequences_l749_74936

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression (a b c : ℚ) : Prop :=
  b^2 = a * c

/-- Checks if three numbers form an arithmetic progression -/
def is_arithmetic_progression (a b c : ℚ) : Prop :=
  2 * b = a + c

/-- Represents a triple of rational numbers -/
structure Triple where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if a triple satisfies all the conditions -/
def satisfies_conditions (t : Triple) : Prop :=
  is_geometric_progression t.a t.b t.c ∧
  is_arithmetic_progression t.a t.b (t.c - 4) ∧
  is_geometric_progression t.a (t.b - 1) (t.c - 5)

theorem geometric_arithmetic_geometric_sequences :
  ∃ t₁ t₂ : Triple,
    satisfies_conditions t₁ ∧
    satisfies_conditions t₂ ∧
    t₁ = ⟨1/9, 7/9, 49/9⟩ ∧
    t₂ = ⟨1, 3, 9⟩ :=
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_geometric_sequences_l749_74936


namespace NUMINAMATH_CALUDE_shrink_ray_reduction_l749_74998

/-- The shrink ray problem -/
theorem shrink_ray_reduction (initial_cups : ℕ) (initial_coffee_per_cup : ℝ) (final_total_coffee : ℝ) :
  initial_cups = 5 →
  initial_coffee_per_cup = 8 →
  final_total_coffee = 20 →
  (1 - final_total_coffee / (initial_cups * initial_coffee_per_cup)) * 100 = 50 := by
  sorry

#check shrink_ray_reduction

end NUMINAMATH_CALUDE_shrink_ray_reduction_l749_74998


namespace NUMINAMATH_CALUDE_cubic_roots_arithmetic_progression_l749_74973

-- Define the polynomial
def cubic_polynomial (a b : ℝ) (x : ℂ) : ℂ := x^3 - 9*x^2 + b*x + a

-- Define the arithmetic progression property for complex roots
def arithmetic_progression (r₁ r₂ r₃ : ℂ) : Prop :=
  ∃ (d : ℝ), r₂ - r₁ = d ∧ r₃ - r₂ = d

-- State the theorem
theorem cubic_roots_arithmetic_progression (a b : ℝ) :
  (∃ (r₁ r₂ r₃ : ℂ), 
    (∀ x : ℂ, cubic_polynomial a b x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧
    arithmetic_progression r₁ r₂ r₃ ∧
    (∃ i : ℝ, r₂ = i * Complex.I)) →
  (a = 27 + 3 * (Real.sqrt ((a - 27) / 3))^2 ∧ b = -27) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_arithmetic_progression_l749_74973


namespace NUMINAMATH_CALUDE_ratio_problem_l749_74902

theorem ratio_problem (x y : ℝ) (h : (2*x - 3*y) / (x + 2*y) = 5/4) :
  x / y = 22/3 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l749_74902


namespace NUMINAMATH_CALUDE_units_digit_17_pow_2024_l749_74935

theorem units_digit_17_pow_2024 : (17^2024) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_pow_2024_l749_74935


namespace NUMINAMATH_CALUDE_square_land_side_length_l749_74953

theorem square_land_side_length 
  (area : ℝ) 
  (h : area = Real.sqrt 100) : 
  ∃ (side : ℝ), side * side = area ∧ side = 10 :=
sorry

end NUMINAMATH_CALUDE_square_land_side_length_l749_74953


namespace NUMINAMATH_CALUDE_lenas_collage_friends_l749_74947

/-- Given the conditions of Lena's collage project, prove the number of friends' pictures glued. -/
theorem lenas_collage_friends (clippings_per_friend : ℕ) (glue_per_clipping : ℕ) (total_glue : ℕ) 
  (h1 : clippings_per_friend = 3)
  (h2 : glue_per_clipping = 6)
  (h3 : total_glue = 126) :
  total_glue / (clippings_per_friend * glue_per_clipping) = 7 := by
  sorry

end NUMINAMATH_CALUDE_lenas_collage_friends_l749_74947


namespace NUMINAMATH_CALUDE_total_money_available_l749_74951

/-- Represents the cost of a single gumdrop in cents -/
def cost_per_gumdrop : ℕ := 4

/-- Represents the number of gumdrops that can be purchased -/
def num_gumdrops : ℕ := 20

/-- Theorem stating that the total amount of money available is 80 cents -/
theorem total_money_available : cost_per_gumdrop * num_gumdrops = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_money_available_l749_74951


namespace NUMINAMATH_CALUDE_cube_monotone_l749_74990

theorem cube_monotone (a b : ℝ) : a > b → a^3 > b^3 := by sorry

end NUMINAMATH_CALUDE_cube_monotone_l749_74990


namespace NUMINAMATH_CALUDE_abs_value_inequality_l749_74916

theorem abs_value_inequality (x : ℝ) : |x + 1| < 5 ↔ -6 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_inequality_l749_74916


namespace NUMINAMATH_CALUDE_rd_cost_productivity_relation_l749_74929

/-- The R&D costs required to increase the average labor productivity by 1 million rubles per person -/
def rd_cost_per_unit_productivity : ℝ := 4576

/-- The current R&D costs in million rubles -/
def current_rd_cost : ℝ := 3157.61

/-- The change in average labor productivity in million rubles per person -/
def delta_productivity : ℝ := 0.69

/-- Theorem stating that the R&D costs required to increase the average labor productivity
    by 1 million rubles per person is equal to the ratio of current R&D costs to the change
    in average labor productivity -/
theorem rd_cost_productivity_relation :
  rd_cost_per_unit_productivity = current_rd_cost / delta_productivity := by
  sorry

end NUMINAMATH_CALUDE_rd_cost_productivity_relation_l749_74929


namespace NUMINAMATH_CALUDE_hyperbola_midpoint_exists_l749_74992

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

-- Define the midpoint
def is_midpoint (x1 y1 x2 y2 x0 y0 : ℝ) : Prop :=
  x0 = (x1 + x2) / 2 ∧ y0 = (y1 + y2) / 2

theorem hyperbola_midpoint_exists :
  ∃ (x1 y1 x2 y2 : ℝ),
    is_on_hyperbola x1 y1 ∧
    is_on_hyperbola x2 y2 ∧
    is_midpoint x1 y1 x2 y2 (-1) (-4) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_midpoint_exists_l749_74992


namespace NUMINAMATH_CALUDE_jane_crayons_jane_crayons_proof_l749_74969

/-- Proves that Jane ends up with 80 crayons after starting with 87 and losing 7 to a hippopotamus. -/
theorem jane_crayons : ℕ → ℕ → ℕ → Prop :=
  fun initial_crayons eaten_crayons final_crayons =>
    initial_crayons = 87 ∧ 
    eaten_crayons = 7 ∧ 
    final_crayons = initial_crayons - eaten_crayons →
    final_crayons = 80

/-- The proof of the theorem. -/
theorem jane_crayons_proof : jane_crayons 87 7 80 := by
  sorry

end NUMINAMATH_CALUDE_jane_crayons_jane_crayons_proof_l749_74969


namespace NUMINAMATH_CALUDE_response_rate_percentage_l749_74909

theorem response_rate_percentage (responses_needed : ℕ) (questionnaires_mailed : ℕ) : 
  responses_needed = 900 → questionnaires_mailed = 1500 → 
  (responses_needed : ℝ) / questionnaires_mailed * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_response_rate_percentage_l749_74909


namespace NUMINAMATH_CALUDE_complex_sum_real_part_l749_74959

theorem complex_sum_real_part (a b : ℝ) : 
  (1 + Complex.I) / Complex.I + (1 + Complex.I * Real.sqrt 3) ^ 2 = Complex.mk a b →
  a + b = 2 * Real.sqrt 3 - 2 :=
sorry

end NUMINAMATH_CALUDE_complex_sum_real_part_l749_74959


namespace NUMINAMATH_CALUDE_recurring_decimal_sum_diff_main_theorem_l749_74950

/-- Represents a recurring decimal of the form 0.nnn... where n is a single digit -/
def recurring_decimal (n : ℕ) : ℚ := n / 9

theorem recurring_decimal_sum_diff (a b c : ℕ) (ha : a < 10) (hb : b < 10) (hc : c < 10) :
  recurring_decimal a + recurring_decimal b - recurring_decimal c = (a + b - c : ℚ) / 9 := by sorry

theorem main_theorem : 
  recurring_decimal 6 + recurring_decimal 2 - recurring_decimal 4 = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_sum_diff_main_theorem_l749_74950


namespace NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l749_74917

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 9) (h2 : x * y = -1) : x^2 + y^2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l749_74917


namespace NUMINAMATH_CALUDE_remainder_987654_div_6_l749_74984

theorem remainder_987654_div_6 : 987654 % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_987654_div_6_l749_74984


namespace NUMINAMATH_CALUDE_ellipse_and_circle_tangent_lines_l749_74958

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a circle with center (x, y) and radius r -/
structure Circle where
  x : ℝ
  y : ℝ
  r : ℝ
  h_pos : 0 < r

theorem ellipse_and_circle_tangent_lines 
  (C : Ellipse) 
  (E : Circle)
  (h_minor : C.b^2 = 3)
  (h_focus : C.a^2 - C.b^2 = 3)
  (h_radius : E.r^2 = 2)
  (h_center : E.x^2 / C.a^2 + E.y^2 / C.b^2 = 1)
  (k₁ k₂ : ℝ)
  (h_tangent₁ : (E.x - x)^2 + (k₁ * x - E.y)^2 = E.r^2)
  (h_tangent₂ : (E.x - x)^2 + (k₂ * x - E.y)^2 = E.r^2) :
  (C.a^2 = 6 ∧ C.b^2 = 3) ∧ k₁ * k₂ = -1/2 := by sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_tangent_lines_l749_74958


namespace NUMINAMATH_CALUDE_calvins_collection_size_l749_74904

/-- Calculates the total number of insects in Calvin's collection. -/
def calvinsTotalInsects (roaches scorpions : ℕ) : ℕ :=
  let crickets := roaches / 2
  let caterpillars := scorpions * 2
  roaches + scorpions + crickets + caterpillars

/-- Proves that Calvin has 27 insects in his collection. -/
theorem calvins_collection_size :
  calvinsTotalInsects 12 3 = 27 := by
  sorry

#eval calvinsTotalInsects 12 3

end NUMINAMATH_CALUDE_calvins_collection_size_l749_74904


namespace NUMINAMATH_CALUDE_mean_height_of_players_l749_74911

def player_heights : List ℝ := [47, 48, 50, 50, 54, 55, 57, 59, 63, 63, 64, 65]

theorem mean_height_of_players : 
  (player_heights.sum / player_heights.length : ℝ) = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_mean_height_of_players_l749_74911


namespace NUMINAMATH_CALUDE_right_triangle_three_four_five_l749_74907

theorem right_triangle_three_four_five :
  ∀ (a b c : ℝ),
    a = 3 ∧ b = 4 ∧ c = 5 →
    a^2 + b^2 = c^2 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_three_four_five_l749_74907


namespace NUMINAMATH_CALUDE_concession_stand_soda_cost_l749_74983

/-- Proves that the cost of each soda is $0.50 given the conditions of the concession stand problem -/
theorem concession_stand_soda_cost 
  (total_revenue : ℝ)
  (total_items : ℕ)
  (hot_dogs_sold : ℕ)
  (hot_dog_cost : ℝ)
  (h1 : total_revenue = 78.50)
  (h2 : total_items = 87)
  (h3 : hot_dogs_sold = 35)
  (h4 : hot_dog_cost = 1.50) :
  let soda_cost := (total_revenue - hot_dogs_sold * hot_dog_cost) / (total_items - hot_dogs_sold)
  soda_cost = 0.50 := by
    sorry

#check concession_stand_soda_cost

end NUMINAMATH_CALUDE_concession_stand_soda_cost_l749_74983


namespace NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l749_74928

theorem cube_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l749_74928


namespace NUMINAMATH_CALUDE_ratio_composition_l749_74944

theorem ratio_composition (a b c : ℚ) 
  (hab : a / b = 4 / 3) 
  (hbc : b / c = 1 / 5) : 
  a / c = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_ratio_composition_l749_74944


namespace NUMINAMATH_CALUDE_lcm_18_30_l749_74952

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_30_l749_74952


namespace NUMINAMATH_CALUDE_data_transmission_time_data_transmission_problem_l749_74957

theorem data_transmission_time : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (blocks : ℝ) (chunks_per_block : ℝ) (chunks_per_second : ℝ) (time_in_minutes : ℝ) =>
    blocks * chunks_per_block / chunks_per_second / 60 = time_in_minutes

theorem data_transmission_problem :
  data_transmission_time 100 600 150 7 := by
  sorry

end NUMINAMATH_CALUDE_data_transmission_time_data_transmission_problem_l749_74957


namespace NUMINAMATH_CALUDE_divisible_by_64_l749_74933

theorem divisible_by_64 (n : ℕ) (h : n > 0) : ∃ k : ℤ, 3^(2*n+2) - 8*n - 9 = 64*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_64_l749_74933


namespace NUMINAMATH_CALUDE_probability_red_black_white_l749_74967

def total_balls : ℕ := 12
def red_balls : ℕ := 5
def black_balls : ℕ := 4
def white_balls : ℕ := 2
def green_balls : ℕ := 1

theorem probability_red_black_white :
  (red_balls + black_balls + white_balls : ℚ) / total_balls = 11 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_black_white_l749_74967


namespace NUMINAMATH_CALUDE_power_of_three_l749_74937

theorem power_of_three (m n : ℕ) (h1 : 3^m = 4) (h2 : 3^n = 5) : 3^(2*m + n) = 80 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_l749_74937


namespace NUMINAMATH_CALUDE_quartic_sum_l749_74905

/-- A quartic polynomial with specific properties -/
structure QuarticPolynomial (m : ℝ) where
  Q : ℝ → ℝ
  is_quartic : ∃ (a b c d : ℝ), ∀ x, Q x = a * x^4 + b * x^3 + c * x^2 + d * x + m
  at_zero : Q 0 = m
  at_one : Q 1 = 3 * m
  at_neg_one : Q (-1) = 4 * m
  at_two : Q 2 = 5 * m

/-- The sum of the polynomial evaluated at 3 and -3 equals 407m -/
theorem quartic_sum (m : ℝ) (P : QuarticPolynomial m) : P.Q 3 + P.Q (-3) = 407 * m := by
  sorry

end NUMINAMATH_CALUDE_quartic_sum_l749_74905


namespace NUMINAMATH_CALUDE_no_solution_system_l749_74974

theorem no_solution_system :
  ¬∃ (x y : ℝ), (3 * x - 4 * y = 12) ∧ (9 * x - 12 * y = 15) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_system_l749_74974


namespace NUMINAMATH_CALUDE_simultaneous_pipe_filling_time_l749_74919

theorem simultaneous_pipe_filling_time 
  (fill_time_A : ℝ) 
  (fill_time_B : ℝ) 
  (h1 : fill_time_A = 50) 
  (h2 : fill_time_B = 75) : 
  (1 / (1 / fill_time_A + 1 / fill_time_B)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_pipe_filling_time_l749_74919


namespace NUMINAMATH_CALUDE_trig_identities_l749_74900

theorem trig_identities :
  (Real.sin (15 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4) ∧
  (Real.cos (15 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4) ∧
  (Real.sin (18 * π / 180) = (-1 + Real.sqrt 5) / 4) ∧
  (Real.cos (18 * π / 180) = Real.sqrt (10 + 2 * Real.sqrt 5) / 4) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l749_74900


namespace NUMINAMATH_CALUDE_M_intersect_N_l749_74921

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | ∃ y, y = Real.log (6 - x)}

theorem M_intersect_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l749_74921


namespace NUMINAMATH_CALUDE_inequality_proof_l749_74989

theorem inequality_proof (x : ℝ) : 4 ≤ (x + 1) / (3 * x - 7) ∧ (x + 1) / (3 * x - 7) < 9 ↔ x ∈ Set.Ioo (32 / 13) (29 / 11) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l749_74989


namespace NUMINAMATH_CALUDE_pam_has_1200_apples_l749_74971

/-- The number of apples Pam has in total -/
def pams_total_apples (pams_bags : ℕ) (geralds_apples_per_bag : ℕ) : ℕ :=
  pams_bags * (3 * geralds_apples_per_bag)

/-- Theorem stating that Pam has 1200 apples given the conditions -/
theorem pam_has_1200_apples :
  pams_total_apples 10 40 = 1200 := by
  sorry

#eval pams_total_apples 10 40

end NUMINAMATH_CALUDE_pam_has_1200_apples_l749_74971


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l749_74997

/-- 
Given a quadratic equation x^2 + 8x + k = 0 with nonzero roots in the ratio 3:1,
prove that k = 12
-/
theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 8*x + k = 0 ∧ y^2 + 8*y + k = 0) → k = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l749_74997


namespace NUMINAMATH_CALUDE_expressions_equality_l749_74985

/-- 
Theorem: The expressions 2a+3bc and (a+b)(2a+c) are equal if and only if a+b+c = 2.
-/
theorem expressions_equality (a b c : ℝ) : 2*a + 3*b*c = (a+b)*(2*a+c) ↔ a + b + c = 2 := by
  sorry

end NUMINAMATH_CALUDE_expressions_equality_l749_74985


namespace NUMINAMATH_CALUDE_max_sum_squared_distances_l749_74976

/-- The incircle of a triangle -/
def Incircle (A B O : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | ∃ θ : ℝ, P = (1 + Real.cos θ, 4/3 + Real.sin θ)}

/-- The squared distance between two points -/
def squaredDistance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem max_sum_squared_distances :
  let A : ℝ × ℝ := (3, 0)
  let B : ℝ × ℝ := (0, 4)
  let O : ℝ × ℝ := (0, 0)
  ∀ P ∈ Incircle A B O,
    squaredDistance P A + squaredDistance P B + squaredDistance P O ≤ 22 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squared_distances_l749_74976


namespace NUMINAMATH_CALUDE_roots_polynomial_sum_l749_74934

theorem roots_polynomial_sum (α β : ℝ) : 
  (α^2 - 3*α + 1 = 0) → 
  (β^2 - 3*β + 1 = 0) → 
  3*α^3 + 7*β^4 = 448 := by
sorry

end NUMINAMATH_CALUDE_roots_polynomial_sum_l749_74934


namespace NUMINAMATH_CALUDE_max_profit_theorem_l749_74901

/-- Represents the store's lamp purchasing problem -/
structure LampProblem where
  cost_diff : ℕ  -- Cost difference between type A and B lamps
  budget_A : ℕ   -- Budget for type A lamps
  budget_B : ℕ   -- Budget for type B lamps
  total_lamps : ℕ -- Total number of lamps to purchase
  max_budget : ℕ  -- Maximum total budget
  price_A : ℕ    -- Selling price of type A lamp
  price_B : ℕ    -- Selling price of type B lamp

/-- Calculates the maximum profit for the given LampProblem -/
def max_profit (p : LampProblem) : ℕ :=
  let cost_A := p.budget_A * 2 / 5  -- Cost of type A lamp
  let cost_B := cost_A - p.cost_diff -- Cost of type B lamp
  let max_A := (p.max_budget - cost_B * p.total_lamps) / (cost_A - cost_B)
  let profit := (p.price_A - cost_A) * max_A + (p.price_B - cost_B) * (p.total_lamps - max_A)
  profit

/-- Theorem stating the maximum profit for the given problem -/
theorem max_profit_theorem (p : LampProblem) : 
  p.cost_diff = 40 ∧ 
  p.budget_A = 2000 ∧ 
  p.budget_B = 1600 ∧ 
  p.total_lamps = 80 ∧ 
  p.max_budget = 14550 ∧ 
  p.price_A = 300 ∧ 
  p.price_B = 200 →
  max_profit p = 5780 ∧ 
  (p.max_budget - 160 * p.total_lamps) / 40 = 43 :=
by sorry


end NUMINAMATH_CALUDE_max_profit_theorem_l749_74901


namespace NUMINAMATH_CALUDE_salary_change_percentage_l749_74923

theorem salary_change_percentage (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 0.51 → x = 70 := by sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l749_74923


namespace NUMINAMATH_CALUDE_bakers_cakes_l749_74925

/-- Baker's cake problem -/
theorem bakers_cakes (total_cakes : ℕ) (cakes_left : ℕ) (cakes_sold : ℕ) : 
  total_cakes = 217 → cakes_left = 72 → cakes_sold = total_cakes - cakes_left → cakes_sold = 145 := by
  sorry

end NUMINAMATH_CALUDE_bakers_cakes_l749_74925


namespace NUMINAMATH_CALUDE_min_value_of_linear_function_l749_74941

/-- Given a system of linear inequalities, prove that the minimum value of a linear function is -6. -/
theorem min_value_of_linear_function (x y : ℝ) :
  x - 2*y + 2 ≥ 0 →
  2*x - y - 2 ≤ 0 →
  y ≥ 0 →
  ∀ z : ℝ, z = 3*x + y → z ≥ -6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_linear_function_l749_74941


namespace NUMINAMATH_CALUDE_third_card_different_suit_probability_l749_74940

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards per suit in a standard deck -/
def CardsPerSuit : ℕ := StandardDeck / NumberOfSuits

/-- The probability of picking a third card of a different suit than the first two,
    given that the first two cards are of different suits -/
def thirdCardDifferentSuitProbability : ℚ :=
  (StandardDeck - 2 - 2 * CardsPerSuit) / (StandardDeck - 2)

/-- Theorem stating that the probability of the third card being of a different suit
    than the first two is 12/25, given the conditions of the problem -/
theorem third_card_different_suit_probability :
  thirdCardDifferentSuitProbability = 12 / 25 := by
  sorry

end NUMINAMATH_CALUDE_third_card_different_suit_probability_l749_74940


namespace NUMINAMATH_CALUDE_unique_star_solution_l749_74924

/-- Definition of the ★ operation -/
def star (x y : ℝ) : ℝ := 5*x - 4*y + 2*x*y

/-- Theorem stating that there exists a unique real number y such that 4 ★ y = 10 -/
theorem unique_star_solution : ∃! y : ℝ, star 4 y = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_star_solution_l749_74924


namespace NUMINAMATH_CALUDE_base5_addition_puzzle_l749_74906

/-- Converts a base 10 number to base 5 --/
def toBase5 (n : ℕ) : ℕ := sorry

/-- Represents a digit in base 5 --/
structure Digit5 where
  value : ℕ
  property : value < 5

theorem base5_addition_puzzle :
  ∀ (S H E : Digit5),
    S.value ≠ 0 ∧ H.value ≠ 0 ∧ E.value ≠ 0 →
    S.value ≠ H.value ∧ S.value ≠ E.value ∧ H.value ≠ E.value →
    (S.value * 25 + H.value * 5 + E.value) + (H.value * 5 + E.value) = 
    (S.value * 25 + E.value * 5 + S.value) →
    S.value = 4 ∧ H.value = 1 ∧ E.value = 2 ∧ 
    toBase5 (S.value + H.value + E.value) = 12 :=
by sorry

end NUMINAMATH_CALUDE_base5_addition_puzzle_l749_74906


namespace NUMINAMATH_CALUDE_circular_garden_ratio_l749_74970

theorem circular_garden_ratio (r : ℝ) (h : r = 6) : 
  (2 * Real.pi * r) / (Real.pi * r^2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_ratio_l749_74970


namespace NUMINAMATH_CALUDE_checkerboard_achievable_l749_74977

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 4x4 grid -/
def Grid := Fin 4 → Fin 4 → Color

/-- Initial grid configuration -/
def initial_grid : Grid :=
  λ i j => if j.val < 2 then Color.Black else Color.White

/-- Checkerboard pattern grid -/
def checkerboard : Grid :=
  λ i j => if (i.val + j.val) % 2 = 0 then Color.White else Color.Black

/-- Represents a rectangular subgrid -/
structure Rectangle where
  top_left : Fin 4 × Fin 4
  bottom_right : Fin 4 × Fin 4

/-- Toggles the color of cells within a rectangle -/
def toggle_rectangle (g : Grid) (r : Rectangle) : Grid :=
  λ i j => if i.val ≥ r.top_left.1.val && i.val ≤ r.bottom_right.1.val &&
             j.val ≥ r.top_left.2.val && j.val ≤ r.bottom_right.2.val
           then
             match g i j with
             | Color.Black => Color.White
             | Color.White => Color.Black
           else g i j

/-- Theorem stating that the checkerboard pattern is achievable in three operations -/
theorem checkerboard_achievable :
  ∃ (r1 r2 r3 : Rectangle),
    toggle_rectangle (toggle_rectangle (toggle_rectangle initial_grid r1) r2) r3 = checkerboard :=
  sorry

end NUMINAMATH_CALUDE_checkerboard_achievable_l749_74977


namespace NUMINAMATH_CALUDE_sandwich_cost_l749_74955

theorem sandwich_cost (N B J : ℕ) (h1 : N > 1) (h2 : B > 0) (h3 : J > 0)
  (h4 : N * (3 * B + 6 * J) = 306) : 6 * N * J = 288 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_l749_74955


namespace NUMINAMATH_CALUDE_tomato_price_equality_l749_74931

/-- Prove that the original price per pound of tomatoes equals the selling price of remaining tomatoes --/
theorem tomato_price_equality (original_price : ℝ) 
  (ruined_percentage : ℝ) (profit_percentage : ℝ) (selling_price : ℝ) : 
  ruined_percentage = 0.2 →
  profit_percentage = 0.08 →
  selling_price = 1.08 →
  (1 - ruined_percentage) * selling_price = (1 + profit_percentage) * original_price :=
by sorry

end NUMINAMATH_CALUDE_tomato_price_equality_l749_74931


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_statement_l749_74927

theorem negation_of_absolute_value_statement (x : ℝ) :
  ¬(abs x ≤ 3 ∨ abs x > 5) ↔ (abs x > 3 ∧ abs x ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_statement_l749_74927


namespace NUMINAMATH_CALUDE_running_time_proof_l749_74995

/-- Proves that the time taken for Joe and Pete to be 16 km apart is 80 minutes -/
theorem running_time_proof (joe_speed : ℝ) (pete_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  joe_speed = 0.133333333333 →
  pete_speed = joe_speed / 2 →
  distance = 16 →
  time * (joe_speed + pete_speed) = distance →
  time = 80 := by
sorry

end NUMINAMATH_CALUDE_running_time_proof_l749_74995


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_m_l749_74968

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- State the theorem
theorem infinitely_many_divisible_by_m (m : ℤ) :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ m ∣ fib n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_m_l749_74968


namespace NUMINAMATH_CALUDE_line_intercept_sum_l749_74949

/-- Given a line mx + 3y - 12 = 0 where m is a real number,
    if the sum of its intercepts on the x and y axes is 7,
    then m = 4. -/
theorem line_intercept_sum (m : ℝ) : 
  (∃ x y : ℝ, m * x + 3 * y - 12 = 0 ∧ 
   (x = 0 ∨ y = 0) ∧
   (∃ x₀ y₀ : ℝ, m * x₀ + 3 * y₀ - 12 = 0 ∧ 
    x₀ = 0 ∧ y₀ = 0 ∧ x + y₀ = 7)) → 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l749_74949


namespace NUMINAMATH_CALUDE_second_chapter_pages_l749_74945

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  chapter1_pages : ℕ
  chapter2_pages : ℕ
  two_chapters : chapter1_pages + chapter2_pages = total_pages

/-- The specific book in the problem -/
def problem_book : Book where
  total_pages := 93
  chapter1_pages := 60
  chapter2_pages := 33
  two_chapters := by sorry

theorem second_chapter_pages (b : Book) 
  (h1 : b.total_pages = 93) 
  (h2 : b.chapter1_pages = 60) : 
  b.chapter2_pages = 33 := by
  sorry

end NUMINAMATH_CALUDE_second_chapter_pages_l749_74945


namespace NUMINAMATH_CALUDE_product_equality_l749_74982

theorem product_equality (h : 213 * 16 = 3408) : 16 * 21.3 = 340.8 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l749_74982


namespace NUMINAMATH_CALUDE_hyperbola_equation_l749_74981

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), y = 2*x + 10 ∧ x^2 + y^2 = (a^2 + b^2)) → 
  (b / a = 2) → 
  (a^2 = 5 ∧ b^2 = 20) := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l749_74981


namespace NUMINAMATH_CALUDE_new_total_cucumber_weight_l749_74922

/-- Calculates the new weight of cucumbers after evaporation -/
def new_cucumber_weight (initial_weight : ℝ) (water_percentage : ℝ) (evaporation_rate : ℝ) : ℝ :=
  let water_weight := initial_weight * water_percentage
  let dry_weight := initial_weight * (1 - water_percentage)
  let evaporated_water := water_weight * evaporation_rate
  (water_weight - evaporated_water) + dry_weight

/-- Theorem stating the new total weight of cucumbers after evaporation -/
theorem new_total_cucumber_weight :
  let batch1 := new_cucumber_weight 50 0.99 0.01
  let batch2 := new_cucumber_weight 30 0.98 0.02
  let batch3 := new_cucumber_weight 20 0.97 0.03
  batch1 + batch2 + batch3 = 98.335 := by
  sorry

#eval new_cucumber_weight 50 0.99 0.01 +
      new_cucumber_weight 30 0.98 0.02 +
      new_cucumber_weight 20 0.97 0.03

end NUMINAMATH_CALUDE_new_total_cucumber_weight_l749_74922


namespace NUMINAMATH_CALUDE_innocent_knight_convincing_l749_74903

-- Define the types of people
inductive PersonType
| Normal
| Knight
| Liar

-- Define the properties of a person
structure Person where
  type : PersonType
  guilty : Bool

-- Define the criminal
def criminal : Person := { type := PersonType.Liar, guilty := true }

-- Define the statement made by the person
def statement (p : Person) : Prop := p.type = PersonType.Knight ∧ ¬p.guilty

-- Theorem to prove
theorem innocent_knight_convincing (p : Person) 
  (h1 : p.type ≠ PersonType.Normal) 
  (h2 : ¬p.guilty) 
  (h3 : p.type ≠ PersonType.Liar) :
  statement p → (¬p.guilty ∧ p.type ≠ PersonType.Liar) :=
by sorry

end NUMINAMATH_CALUDE_innocent_knight_convincing_l749_74903


namespace NUMINAMATH_CALUDE_two_digit_swap_difference_divisible_by_nine_l749_74932

theorem two_digit_swap_difference_divisible_by_nine 
  (a b : ℕ) 
  (h1 : a ≤ 9) 
  (h2 : b ≤ 9) 
  (h3 : a ≠ b) : 
  ∃ k : ℤ, (|(10 * a + b) - (10 * b + a)| : ℤ) = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_two_digit_swap_difference_divisible_by_nine_l749_74932


namespace NUMINAMATH_CALUDE_abc_sign_determination_l749_74914

theorem abc_sign_determination (a b c : ℝ) 
  (h1 : (a > 0 ∧ b < 0 ∧ c = 0) ∨ (a > 0 ∧ b = 0 ∧ c < 0) ∨ (a < 0 ∧ b > 0 ∧ c = 0) ∨ 
        (a < 0 ∧ b = 0 ∧ c > 0) ∨ (a = 0 ∧ b > 0 ∧ c < 0) ∨ (a = 0 ∧ b < 0 ∧ c > 0))
  (h2 : a * b^2 * (a + c) * (b + c) < 0) :
  a > 0 ∧ b < 0 ∧ c = 0 :=
by sorry

end NUMINAMATH_CALUDE_abc_sign_determination_l749_74914


namespace NUMINAMATH_CALUDE_girls_in_school_l749_74908

theorem girls_in_school (total_students sample_size girls_boys_diff : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : girls_boys_diff = 20) : 
  ∃ (girls : ℕ), girls = 860 ∧ 
  girls + (total_students - girls) = total_students ∧
  (girls : ℚ) / total_students * sample_size = 
    (total_students - girls : ℚ) / total_students * sample_size - girls_boys_diff := by
  sorry

end NUMINAMATH_CALUDE_girls_in_school_l749_74908


namespace NUMINAMATH_CALUDE_perimeter_bisector_min_value_l749_74966

/-- A line that always bisects the perimeter of a circle -/
structure PerimeterBisector where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : ∀ (x y : ℝ), a * x + b * y + 1 = 0 → x^2 + y^2 + 8*x + 2*y + 1 = 0 →
       ∃ (c : ℝ), c > 0 ∧ (x + 4)^2 + (y + 1)^2 = c^2 ∧ a * (-4) + b * (-1) + 1 = 0

/-- The minimum value of 1/a + 4/b for a perimeter bisector is 16 -/
theorem perimeter_bisector_min_value (pb : PerimeterBisector) :
  (1 / pb.a + 4 / pb.b) ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_bisector_min_value_l749_74966


namespace NUMINAMATH_CALUDE_tangent_line_distance_l749_74954

/-- A line with slope 1 is tangent to y = e^x and y^2 = 4x at two different points. -/
theorem tangent_line_distance : ∃ (x₁ y₁ x₂ y₂ : ℝ),
  -- The line is tangent to y = e^x at (x₁, y₁)
  (Real.exp x₁ = y₁) ∧ 
  (Real.exp x₁ = 1) ∧
  -- The line is tangent to y^2 = 4x at (x₂, y₂)
  (y₂^2 = 4 * x₂) ∧
  (y₂ = 2 * Real.sqrt x₂) ∧
  -- Both points lie on a line with slope 1
  (y₂ - y₁ = x₂ - x₁) ∧
  -- The distance between the two points is √2
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_tangent_line_distance_l749_74954


namespace NUMINAMATH_CALUDE_percentage_of_number_seventy_six_point_five_percent_of_1287_l749_74996

theorem percentage_of_number (x : ℝ) (y : ℝ) (z : ℝ) (h : z = x * (y / 100)) :
  z = x * (y / 100) := by
  sorry

theorem seventy_six_point_five_percent_of_1287 :
  (76.5 / 100) * 1287 = 984.495 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_number_seventy_six_point_five_percent_of_1287_l749_74996


namespace NUMINAMATH_CALUDE_hyperbola_min_value_l749_74965

/-- The minimum value of (b² + 1) / a for a hyperbola with eccentricity 2 -/
theorem hyperbola_min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2  -- eccentricity
  let c := e * a  -- focal distance
  (c^2 = a^2 + b^2) →  -- hyperbola property
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (b^2 + 1) / a ≥ 2 * Real.sqrt 3) ∧
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ (b^2 + 1) / a = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_min_value_l749_74965


namespace NUMINAMATH_CALUDE_remainder_problem_l749_74961

theorem remainder_problem (n : ℕ) (h1 : n = 349) (h2 : n % 17 = 9) : n % 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l749_74961


namespace NUMINAMATH_CALUDE_penny_frog_count_l749_74964

/-- The number of tree frogs Penny counted -/
def tree_frogs : ℕ := 55

/-- The number of poison frogs Penny counted -/
def poison_frogs : ℕ := 10

/-- The number of wood frogs Penny counted -/
def wood_frogs : ℕ := 13

/-- The total number of frogs Penny counted -/
def total_frogs : ℕ := tree_frogs + poison_frogs + wood_frogs

theorem penny_frog_count : total_frogs = 78 := by
  sorry

end NUMINAMATH_CALUDE_penny_frog_count_l749_74964


namespace NUMINAMATH_CALUDE_max_value_of_expression_l749_74942

theorem max_value_of_expression (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l749_74942


namespace NUMINAMATH_CALUDE_polyhedron_property_l749_74938

/-- A convex polyhedron with specific face and vertex properties -/
structure ConvexPolyhedron where
  faces : ℕ
  triangles : ℕ
  pentagons : ℕ
  hexagons : ℕ
  vertices : ℕ
  P : ℕ  -- number of pentagons meeting at each vertex
  H : ℕ  -- number of hexagons meeting at each vertex
  T : ℕ  -- number of triangles meeting at each vertex

/-- The properties of the specific polyhedron in the problem -/
def problem_polyhedron : ConvexPolyhedron where
  faces := 38
  triangles := 20
  pentagons := 10
  hexagons := 8
  vertices := 115
  P := 4
  H := 2
  T := 2

/-- The theorem to be proved -/
theorem polyhedron_property (poly : ConvexPolyhedron) 
  (h1 : poly.faces = 38)
  (h2 : poly.triangles = 2 * poly.pentagons)
  (h3 : poly.hexagons = 8)
  (h4 : poly.P = 2 * poly.H)
  (h5 : poly.faces = poly.triangles + poly.pentagons + poly.hexagons)
  (h6 : poly = problem_polyhedron) :
  100 * poly.P + 10 * poly.T + poly.vertices = 535 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_property_l749_74938


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l749_74991

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 42 →
  boys + girls = total_students →
  3 * girls = 4 * boys →
  girls - boys = 6 := by
sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l749_74991


namespace NUMINAMATH_CALUDE_cubic_root_sum_l749_74918

theorem cubic_root_sum (a b : ℝ) : 
  (∃ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
   x^3 - 8*x^2 + a*x - b = 0 ∧
   y^3 - 8*y^2 + a*y - b = 0 ∧
   z^3 - 8*z^2 + a*z - b = 0) →
  a + b = 27 ∨ a + b = 31 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l749_74918


namespace NUMINAMATH_CALUDE_digit_1257_of_7_19th_l749_74975

/-- The decimal representation of 7/19 repeats every 18 digits -/
def period : ℕ := 18

/-- The repeating sequence of digits in the decimal representation of 7/19 -/
def repeating_sequence : List ℕ := [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7]

/-- The position we're interested in -/
def target_position : ℕ := 1257

/-- Theorem stating that the 1257th digit after the decimal point in 7/19 is 7 -/
theorem digit_1257_of_7_19th : 
  (repeating_sequence.get? ((target_position - 1) % period)) = some 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_1257_of_7_19th_l749_74975


namespace NUMINAMATH_CALUDE_sum_of_zeros_is_14_l749_74956

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := (x - 3)^2 + 4

-- Define the final parabola after transformations
def final_parabola (x : ℝ) : ℝ := -(x - 7)^2 + 1

-- Define the zeros of the final parabola
def p : ℝ := 8
def q : ℝ := 6

-- Theorem statement
theorem sum_of_zeros_is_14 : p + q = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_zeros_is_14_l749_74956


namespace NUMINAMATH_CALUDE_race_distance_proof_l749_74910

/-- The total distance Jesse and Mia each need to run in a week-long race. -/
def total_distance : ℝ := 48

theorem race_distance_proof (jesse_first_three : ℝ) (jesse_day_four : ℝ) (mia_first_four : ℝ) (final_three_avg : ℝ) :
  jesse_first_three = 3 * (2/3) →
  jesse_day_four = 10 →
  mia_first_four = 4 * 3 →
  final_three_avg = 6 →
  total_distance = jesse_first_three + jesse_day_four + (3 * 2 * final_three_avg) / 2 :=
by sorry

end NUMINAMATH_CALUDE_race_distance_proof_l749_74910


namespace NUMINAMATH_CALUDE_expression_value_l749_74926

theorem expression_value : (3^2 - 2 * 3) - (5^2 - 2 * 5) + (7^2 - 2 * 7) = 23 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l749_74926


namespace NUMINAMATH_CALUDE_two_circles_distance_formula_l749_74913

/-- Two circles with radii R and r, whose centers are at distance d apart,
    and whose common internal tangents define four points of tangency
    that form a quadrilateral circumscribed around a circle. -/
structure TwoCirclesConfig where
  R : ℝ
  r : ℝ
  d : ℝ

/-- The theorem stating the relationship between the radii and the distance between centers -/
theorem two_circles_distance_formula (config : TwoCirclesConfig) :
  config.d ^ 2 = (config.R + config.r) ^ 2 + 4 * config.R * config.r :=
sorry

end NUMINAMATH_CALUDE_two_circles_distance_formula_l749_74913


namespace NUMINAMATH_CALUDE_log_xy_value_l749_74986

theorem log_xy_value (x y : ℝ) (hxy3 : Real.log (x * y^3) = 1) (hx2y : Real.log (x^2 * y) = 1) :
  Real.log (x * y) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_log_xy_value_l749_74986


namespace NUMINAMATH_CALUDE_equation_a_solution_l749_74963

theorem equation_a_solution (x : ℝ) : 
  1/(x-1) + 3/(x-3) - 9/(x-5) + 5/(x-7) = 0 ↔ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_a_solution_l749_74963


namespace NUMINAMATH_CALUDE_weight_of_one_bag_is_five_l749_74999

-- Define the given values
def total_harvest : ℕ := 405
def juice_amount : ℕ := 90
def restaurant_amount : ℕ := 60
def total_revenue : ℕ := 408
def price_per_bag : ℕ := 8

-- Define the weight of one bag as a function of the given values
def weight_of_one_bag : ℚ :=
  (total_harvest - juice_amount - restaurant_amount) / (total_revenue / price_per_bag)

-- Theorem to prove
theorem weight_of_one_bag_is_five :
  weight_of_one_bag = 5 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_one_bag_is_five_l749_74999


namespace NUMINAMATH_CALUDE_success_permutations_l749_74930

/-- The number of unique arrangements of letters in "SUCCESS" -/
def success_arrangements : ℕ := 420

/-- The total number of letters in "SUCCESS" -/
def total_letters : ℕ := 7

/-- The number of S's in "SUCCESS" -/
def num_s : ℕ := 3

/-- The number of C's in "SUCCESS" -/
def num_c : ℕ := 2

/-- The number of U's in "SUCCESS" -/
def num_u : ℕ := 1

/-- The number of E's in "SUCCESS" -/
def num_e : ℕ := 1

theorem success_permutations :
  success_arrangements = (Nat.factorial total_letters) / ((Nat.factorial num_s) * (Nat.factorial num_c)) :=
by sorry

end NUMINAMATH_CALUDE_success_permutations_l749_74930


namespace NUMINAMATH_CALUDE_mirasol_account_balance_l749_74979

def remaining_amount (initial : ℕ) (expense1 : ℕ) (expense2 : ℕ) : ℕ :=
  initial - (expense1 + expense2)

theorem mirasol_account_balance : remaining_amount 50 10 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mirasol_account_balance_l749_74979


namespace NUMINAMATH_CALUDE_system_solution_l749_74915

theorem system_solution :
  ∃ x y : ℚ, (4 * x - 3 * y = -7) ∧ (5 * x + 6 * y = 4) ∧ (x = -10/13) ∧ (y = 17/13) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l749_74915


namespace NUMINAMATH_CALUDE_cube_surface_area_l749_74993

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p q : Point3D) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2

/-- The vertices of the cube -/
def P : Point3D := ⟨6, 11, 11⟩
def Q : Point3D := ⟨7, 7, 2⟩
def R : Point3D := ⟨10, 2, 10⟩

theorem cube_surface_area : 
  squaredDistance P Q = squaredDistance P R ∧ 
  squaredDistance P R = squaredDistance Q R ∧
  squaredDistance Q R = 98 →
  (6 * ((squaredDistance P Q).sqrt / Real.sqrt 2)^2 : ℝ) = 294 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l749_74993


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l749_74962

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_proof (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h1 : ∀ n : ℕ, 2 * S n = n * a n)
    (h2 : a 2 = 1) :
  (∀ n : ℕ, n ≥ 1 → a n = n - 1) ∧ is_arithmetic_sequence a :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l749_74962


namespace NUMINAMATH_CALUDE_michelle_gas_usage_l749_74939

theorem michelle_gas_usage (start_gas end_gas : ℝ) (h1 : start_gas = 0.5) (h2 : end_gas = 0.17) :
  start_gas - end_gas = 0.33 := by
sorry

end NUMINAMATH_CALUDE_michelle_gas_usage_l749_74939


namespace NUMINAMATH_CALUDE_function_periodicity_l749_74980

open Real

-- Define the function f and the constant a
variable (f : ℝ → ℝ) (a : ℝ)

-- State the theorem
theorem function_periodicity 
  (h : ∀ x, f (x + a) = (1 + f x) / (1 - f x)) 
  (ha : a ≠ 0) : 
  ∀ x, f (x + 4 * a) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l749_74980


namespace NUMINAMATH_CALUDE_exam_attendance_l749_74912

theorem exam_attendance (passed_percentage : ℝ) (failed_count : ℕ) : 
  passed_percentage = 35 →
  failed_count = 481 →
  (100 - passed_percentage) / 100 * 740 = failed_count :=
by
  sorry

end NUMINAMATH_CALUDE_exam_attendance_l749_74912


namespace NUMINAMATH_CALUDE_probability_for_given_dice_l749_74920

/-- Represents a 20-sided die with color distributions -/
structure TwentySidedDie :=
  (maroon : Nat)
  (teal : Nat)
  (cyan : Nat)
  (sparkly : Nat)
  (total : Nat)
  (sum_eq_total : maroon + teal + cyan + sparkly = total)

/-- Calculate the probability of two 20-sided dice showing the same color
    and a 6-sided die showing a number greater than 4 -/
def probability_same_color_and_high_roll 
  (die1 : TwentySidedDie) 
  (die2 : TwentySidedDie) : ℚ :=
  let same_color_prob := 
    (die1.maroon * die2.maroon + 
     die1.teal * die2.teal + 
     die1.cyan * die2.cyan + 
     die1.sparkly * die2.sparkly : ℚ) / 
    (die1.total * die2.total : ℚ)
  let high_roll_prob : ℚ := 1 / 3
  same_color_prob * high_roll_prob

/-- The main theorem stating the probability for the given dice configuration -/
theorem probability_for_given_dice : 
  let die1 : TwentySidedDie := ⟨3, 9, 7, 1, 20, by norm_num⟩
  let die2 : TwentySidedDie := ⟨5, 6, 8, 1, 20, by norm_num⟩
  probability_same_color_and_high_roll die1 die2 = 21 / 200 := by
  sorry

end NUMINAMATH_CALUDE_probability_for_given_dice_l749_74920


namespace NUMINAMATH_CALUDE_digit_property_characterization_l749_74946

def has_property (z : Nat) : Prop :=
  z < 10 ∧ 
  ∀ k : Nat, k ≥ 1 → 
    ∃ n : Nat, n ≥ 1 ∧ 
      ∃ m : Nat, n^9 = m * 10^k + z * ((10^k - 1) / 9)

theorem digit_property_characterization :
  ∀ z : Nat, has_property z ↔ z ∈ ({0, 1, 3, 7, 9} : Set Nat) :=
sorry

end NUMINAMATH_CALUDE_digit_property_characterization_l749_74946


namespace NUMINAMATH_CALUDE_v_sum_zero_l749_74960

noncomputable def v (x : ℝ) : ℝ := -x + (3/2) * Real.sin (x * Real.pi / 2)

theorem v_sum_zero : v (-3.14) + v (-1) + v 1 + v 3.14 = 0 := by sorry

end NUMINAMATH_CALUDE_v_sum_zero_l749_74960


namespace NUMINAMATH_CALUDE_original_dish_price_l749_74994

theorem original_dish_price (price : ℝ) : 
  (price * 0.9 + price * 0.15 = price * 0.9 + price * 0.9 * 0.15 + 1.26) → 
  price = 84 :=
by sorry

end NUMINAMATH_CALUDE_original_dish_price_l749_74994
