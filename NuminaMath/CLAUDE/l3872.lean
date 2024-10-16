import Mathlib

namespace NUMINAMATH_CALUDE_unique_perfect_square_divisor_l3872_387237

theorem unique_perfect_square_divisor : ∃! (n : ℕ), n > 0 ∧ ∃ (k : ℕ), (n^3 - 1989) / n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_perfect_square_divisor_l3872_387237


namespace NUMINAMATH_CALUDE_cube_difference_l3872_387299

/-- Calculates the number of cubes needed for a hollow block -/
def hollow_block_cubes (length width depth : ℕ) : ℕ :=
  2 * length * width + 4 * (length + width) * (depth - 2) - 8 * (depth - 2)

/-- Calculates the number of cubes in a solid block -/
def solid_block_cubes (length width depth : ℕ) : ℕ :=
  length * width * depth

theorem cube_difference (length width depth : ℕ) 
  (h1 : length = 7)
  (h2 : width = 7)
  (h3 : depth = 6) : 
  solid_block_cubes length width depth - hollow_block_cubes length width depth = 100 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l3872_387299


namespace NUMINAMATH_CALUDE_incorrect_exponent_operation_l3872_387225

theorem incorrect_exponent_operation (a : ℝ) : (-a^2)^3 ≠ -a^5 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_exponent_operation_l3872_387225


namespace NUMINAMATH_CALUDE_josiah_hans_age_ratio_l3872_387236

theorem josiah_hans_age_ratio :
  ∀ (hans_age josiah_age : ℕ),
    hans_age = 15 →
    josiah_age + 3 + hans_age + 3 = 66 →
    josiah_age / hans_age = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_josiah_hans_age_ratio_l3872_387236


namespace NUMINAMATH_CALUDE_shop_profit_calculation_l3872_387296

/-- The amount the shop makes off each jersey -/
def jersey_profit : ℝ := 34

/-- The amount the shop makes off each t-shirt -/
def tshirt_profit : ℝ := 192

/-- The difference in cost between a t-shirt and a jersey -/
def cost_difference : ℝ := 158

theorem shop_profit_calculation :
  jersey_profit = tshirt_profit - cost_difference :=
by sorry

end NUMINAMATH_CALUDE_shop_profit_calculation_l3872_387296


namespace NUMINAMATH_CALUDE_speedster_roadster_convertibles_l3872_387278

/-- Represents the inventory of an automobile company -/
structure Inventory where
  total : ℕ
  speedsters : ℕ
  roadsters : ℕ
  cruisers : ℕ
  speedster_convertibles : ℕ
  roadster_convertibles : ℕ
  cruiser_convertibles : ℕ

/-- Theorem stating the number of Speedster and Roadster convertibles -/
theorem speedster_roadster_convertibles (inv : Inventory) : 
  inv.speedster_convertibles + inv.roadster_convertibles = 52 :=
  by
  have h1 : inv.total = 100 := by sorry
  have h2 : inv.speedsters = inv.total * 2 / 5 := by sorry
  have h3 : inv.roadsters = inv.total * 3 / 10 := by sorry
  have h4 : inv.cruisers = inv.total - inv.speedsters - inv.roadsters := by sorry
  have h5 : inv.speedster_convertibles = inv.speedsters * 4 / 5 := by sorry
  have h6 : inv.roadster_convertibles = inv.roadsters * 2 / 3 := by sorry
  have h7 : inv.cruiser_convertibles = inv.cruisers * 1 / 4 := by sorry
  have h8 : inv.total - inv.speedsters = 60 := by sorry
  sorry

end NUMINAMATH_CALUDE_speedster_roadster_convertibles_l3872_387278


namespace NUMINAMATH_CALUDE_max_CP_value_l3872_387264

-- Define the equilateral triangle ABC
def EquilateralTriangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

-- Define the point P
def P : ℝ × ℝ := sorry

-- Define the distances
def AP : ℝ := 2
def BP : ℝ := 3

-- Theorem statement
theorem max_CP_value 
  (A B C : ℝ × ℝ) 
  (h_equilateral : EquilateralTriangle A B C) 
  (h_AP : dist A P = AP) 
  (h_BP : dist B P = BP) :
  ∀ P', dist C P' ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_CP_value_l3872_387264


namespace NUMINAMATH_CALUDE_isabellas_haircut_l3872_387252

/-- Isabella's haircut problem -/
theorem isabellas_haircut (original_length cut_length : ℕ) (h1 : original_length = 18) (h2 : cut_length = 9) :
  original_length - cut_length = 9 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_haircut_l3872_387252


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l3872_387298

/-- Given two vectors a and b in ℝ³, where a = (2, 4, 5) and b = (3, x, y),
    if a is parallel to b, then x + y = 27/2 -/
theorem parallel_vectors_sum (x y : ℝ) :
  let a : Fin 3 → ℝ := ![2, 4, 5]
  let b : Fin 3 → ℝ := ![3, x, y]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  x + y = 27/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l3872_387298


namespace NUMINAMATH_CALUDE_zero_is_monomial_l3872_387269

/-- Definition of a monomial -/
def is_monomial (expr : ℕ → ℚ) : Prop :=
  ∃ (c : ℚ) (n : ℕ), ∀ (k : ℕ), expr k = if k = n then c else 0

/-- Theorem: 0 is a monomial -/
theorem zero_is_monomial : is_monomial (λ _ => 0) := by
  sorry

end NUMINAMATH_CALUDE_zero_is_monomial_l3872_387269


namespace NUMINAMATH_CALUDE_x_value_l3872_387229

theorem x_value : ∃ x : ℝ, 3 * x - 48.2 = 0.25 * (4 * x + 56.8) ∧ x = 31.2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3872_387229


namespace NUMINAMATH_CALUDE_income_comparison_l3872_387293

theorem income_comparison (A B : ℝ) (h : B = A * (1 + 1/3)) : 
  A = B * (1 - 1/4) := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l3872_387293


namespace NUMINAMATH_CALUDE_car_profit_percentage_l3872_387222

/-- Calculates the profit percentage on the original price of a car
    given the discount percentage on purchase and markup percentage on sale. -/
theorem car_profit_percentage
  (P : ℝ)                    -- Original price of the car
  (discount : ℝ)             -- Discount percentage on purchase
  (markup : ℝ)               -- Markup percentage on sale
  (h_discount : discount = 20)
  (h_markup : markup = 45)
  : (((1 - discount / 100) * (1 + markup / 100) - 1) * 100 = 16) := by
  sorry

end NUMINAMATH_CALUDE_car_profit_percentage_l3872_387222


namespace NUMINAMATH_CALUDE_arrangement_count_is_2880_l3872_387214

/-- The number of ways to arrange 4 boys and 3 girls in a row with constraints -/
def arrangementCount : ℕ :=
  let numBoys : ℕ := 4
  let numGirls : ℕ := 3
  let waysToChooseTwoGirls : ℕ := Nat.choose numGirls 2
  let waysToArrangeBoys : ℕ := Nat.factorial numBoys
  let spacesForGirlUnits : ℕ := numBoys + 1
  let waysToInsertGirlUnits : ℕ := Nat.descFactorial spacesForGirlUnits 2
  waysToChooseTwoGirls * waysToArrangeBoys * waysToInsertGirlUnits

theorem arrangement_count_is_2880 : arrangementCount = 2880 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_2880_l3872_387214


namespace NUMINAMATH_CALUDE_candy_distribution_l3872_387223

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) : 
  total_candy = 16 → 
  num_bags = 2 → 
  total_candy = num_bags * candy_per_bag →
  candy_per_bag = 8 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3872_387223


namespace NUMINAMATH_CALUDE_ant_distance_theorem_l3872_387255

theorem ant_distance_theorem (n : ℕ) (points : Fin n → ℝ × ℝ) :
  n = 1390 →
  (∀ i, abs (points i).2 < 1) →
  (∀ i j, i ≠ j → dist (points i) (points j) > 2) →
  ∃ i j, dist (points i) (points j) ≥ 1000 :=
by sorry

#check ant_distance_theorem

end NUMINAMATH_CALUDE_ant_distance_theorem_l3872_387255


namespace NUMINAMATH_CALUDE_power_multiplication_division_equality_l3872_387274

theorem power_multiplication_division_equality : (15^2 * 8^3) / 256 = 450 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_division_equality_l3872_387274


namespace NUMINAMATH_CALUDE_tangent_sum_difference_l3872_387266

theorem tangent_sum_difference (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) : 
  Real.tan (α + π/4) = 3/22 := by
sorry

end NUMINAMATH_CALUDE_tangent_sum_difference_l3872_387266


namespace NUMINAMATH_CALUDE_expression_simplification_l3872_387224

theorem expression_simplification (a b : ℤ) (h1 : a = 1) (h2 : b = -2) :
  2 * (a^2 - 3*a*b + 1) - (2*a^2 - b^2) + 5*a*b = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3872_387224


namespace NUMINAMATH_CALUDE_larger_complementary_angle_measure_l3872_387203

def complementary_angles (a b : ℝ) : Prop := a + b = 90

theorem larger_complementary_angle_measure :
  ∀ (x y : ℝ),
    complementary_angles x y →
    x / y = 4 / 3 →
    x > y →
    x = 51 + 3 / 7 :=
by sorry

end NUMINAMATH_CALUDE_larger_complementary_angle_measure_l3872_387203


namespace NUMINAMATH_CALUDE_cake_cost_calculation_l3872_387295

/-- The cost of a cake given initial money and remaining money after purchase -/
def cake_cost (initial_money remaining_money : ℚ) : ℚ :=
  initial_money - remaining_money

theorem cake_cost_calculation (initial_money remaining_money : ℚ) 
  (h1 : initial_money = 59.5)
  (h2 : remaining_money = 42) : 
  cake_cost initial_money remaining_money = 17.5 := by
  sorry

#eval cake_cost 59.5 42

end NUMINAMATH_CALUDE_cake_cost_calculation_l3872_387295


namespace NUMINAMATH_CALUDE_holiday_approval_count_l3872_387240

theorem holiday_approval_count (total : ℕ) (oppose_percent : ℚ) (indifferent_percent : ℚ) 
  (h_total : total = 600)
  (h_oppose : oppose_percent = 6 / 100)
  (h_indifferent : indifferent_percent = 14 / 100) :
  ↑total * (1 - oppose_percent - indifferent_percent) = 480 :=
by sorry

end NUMINAMATH_CALUDE_holiday_approval_count_l3872_387240


namespace NUMINAMATH_CALUDE_yolanda_departure_time_yolanda_left_30_minutes_before_l3872_387253

/-- Prove that Yolanda left 30 minutes before her husband caught up to her. -/
theorem yolanda_departure_time 
  (yolanda_speed : ℝ) 
  (husband_speed : ℝ) 
  (husband_delay : ℝ) 
  (catch_up_time : ℝ) 
  (h1 : yolanda_speed = 20)
  (h2 : husband_speed = 40)
  (h3 : husband_delay = 15 / 60)  -- Convert 15 minutes to hours
  (h4 : catch_up_time = 15 / 60)  -- Convert 15 minutes to hours
  : yolanda_speed * (husband_delay + catch_up_time) = husband_speed * catch_up_time :=
by sorry

/-- Yolanda's departure time before being caught -/
def yolanda_departure_before_catch (yolanda_speed : ℝ) (husband_speed : ℝ) (husband_delay : ℝ) (catch_up_time : ℝ) : ℝ :=
  husband_delay + catch_up_time

/-- Prove that Yolanda left 30 minutes (0.5 hours) before her husband caught up to her -/
theorem yolanda_left_30_minutes_before
  (yolanda_speed : ℝ) 
  (husband_speed : ℝ) 
  (husband_delay : ℝ) 
  (catch_up_time : ℝ) 
  (h1 : yolanda_speed = 20)
  (h2 : husband_speed = 40)
  (h3 : husband_delay = 15 / 60)
  (h4 : catch_up_time = 15 / 60)
  : yolanda_departure_before_catch yolanda_speed husband_speed husband_delay catch_up_time = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_yolanda_departure_time_yolanda_left_30_minutes_before_l3872_387253


namespace NUMINAMATH_CALUDE_gym_spending_l3872_387260

theorem gym_spending (total_spent adidas_cost nike_cost skechers_cost clothes_cost : ℝ) : 
  total_spent = 8000 →
  nike_cost = 3 * adidas_cost →
  adidas_cost = (1 / 5) * skechers_cost →
  adidas_cost = 600 →
  total_spent = adidas_cost + nike_cost + skechers_cost + clothes_cost →
  clothes_cost = 2600 := by
sorry

end NUMINAMATH_CALUDE_gym_spending_l3872_387260


namespace NUMINAMATH_CALUDE_multiply_algebraic_expressions_l3872_387217

theorem multiply_algebraic_expressions (x y : ℝ) :
  6 * x * y^3 * (-1/2 * x^3 * y^2) = -3 * x^4 * y^5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_algebraic_expressions_l3872_387217


namespace NUMINAMATH_CALUDE_roots_sum_reciprocals_l3872_387290

theorem roots_sum_reciprocals (α β : ℝ) : 
  3 * α^2 + α - 1 = 0 →
  3 * β^2 + β - 1 = 0 →
  α > β →
  (α / β) + (β / α) = -7/3 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocals_l3872_387290


namespace NUMINAMATH_CALUDE_arrangements_count_l3872_387262

/-- The number of arrangements for 7 students with specific conditions -/
def num_arrangements : ℕ :=
  let total_students : ℕ := 7
  let middle_student : ℕ := 1
  let together_students : ℕ := 2
  let remaining_students : ℕ := total_students - middle_student - together_students
  let ways_to_place_together : ℕ := 2  -- left or right of middle
  let arrangements_within_together : ℕ := 2  -- B-C or C-B
  let permutations_of_remaining : ℕ := Nat.factorial remaining_students
  ways_to_place_together * arrangements_within_together * permutations_of_remaining

theorem arrangements_count : num_arrangements = 192 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l3872_387262


namespace NUMINAMATH_CALUDE_deborahs_mailing_cost_l3872_387258

/-- The total cost of mailing letters with given conditions -/
def total_mailing_cost (num_letters : ℕ) (num_international : ℕ) (standard_postage : ℚ) (international_surcharge : ℚ) : ℚ :=
  (num_letters - num_international) * standard_postage + 
  num_international * (standard_postage + international_surcharge)

/-- Proof that the total mailing cost for Deborah's letters is $4.60 -/
theorem deborahs_mailing_cost :
  total_mailing_cost 4 2 (108/100) (14/100) = 460/100 := by
  sorry

end NUMINAMATH_CALUDE_deborahs_mailing_cost_l3872_387258


namespace NUMINAMATH_CALUDE_complex_quadrant_l3872_387210

theorem complex_quadrant (a b : ℝ) (z : ℂ) :
  z = a + b * Complex.I →
  (a / (1 - Complex.I) + b / (1 - 2 * Complex.I) = 5 / (3 + Complex.I)) →
  0 < a ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l3872_387210


namespace NUMINAMATH_CALUDE_sum_of_integers_l3872_387257

theorem sum_of_integers (a b c d : ℤ) 
  (eq1 : a - b + c = 7)
  (eq2 : b - c + d = 4)
  (eq3 : c - d + a = 2)
  (eq4 : d - a + b = 1) :
  a + b + c + d = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3872_387257


namespace NUMINAMATH_CALUDE_sum_interior_angles_specific_polyhedron_l3872_387276

/-- A convex polyhedron with given number of vertices and edges -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ

/-- The sum of interior angles of all faces of a convex polyhedron -/
def sum_interior_angles (p : ConvexPolyhedron) : ℝ :=
  sorry

/-- Theorem stating the sum of interior angles for a specific convex polyhedron -/
theorem sum_interior_angles_specific_polyhedron :
  let p : ConvexPolyhedron := ⟨20, 30⟩
  sum_interior_angles p = 6480 :=
by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_specific_polyhedron_l3872_387276


namespace NUMINAMATH_CALUDE_exists_solution_with_y_twelve_l3872_387272

theorem exists_solution_with_y_twelve :
  ∃ (x z t : ℕ+), x + 12 + z + t = 15 := by
sorry

end NUMINAMATH_CALUDE_exists_solution_with_y_twelve_l3872_387272


namespace NUMINAMATH_CALUDE_zebra_stripes_l3872_387227

theorem zebra_stripes (w n b : ℕ) : 
  w + n = b + 1 →  -- Total black stripes = white stripes + 1
  b = w + 7 →      -- White stripes = wide black stripes + 7
  n = 8 :=         -- Number of narrow black stripes is 8
by sorry

end NUMINAMATH_CALUDE_zebra_stripes_l3872_387227


namespace NUMINAMATH_CALUDE_smallest_integer_square_triple_l3872_387256

theorem smallest_integer_square_triple (x : ℤ) : x^2 = 3*x + 75 → x ≥ -5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_square_triple_l3872_387256


namespace NUMINAMATH_CALUDE_power_product_simplification_l3872_387254

theorem power_product_simplification (a : ℝ) : (3 * a)^2 * a^5 = 9 * a^7 := by
  sorry

end NUMINAMATH_CALUDE_power_product_simplification_l3872_387254


namespace NUMINAMATH_CALUDE_gina_credits_l3872_387211

/-- Proves that given the conditions of Gina's college expenses, she is taking 14 credits -/
theorem gina_credits : 
  let credit_cost : ℕ := 450
  let textbook_count : ℕ := 5
  let textbook_cost : ℕ := 120
  let facilities_fee : ℕ := 200
  let total_cost : ℕ := 7100
  ∃ (credits : ℕ), 
    credits * credit_cost + 
    textbook_count * textbook_cost + 
    facilities_fee = total_cost ∧ 
    credits = 14 :=
by sorry

end NUMINAMATH_CALUDE_gina_credits_l3872_387211


namespace NUMINAMATH_CALUDE_fence_painting_problem_l3872_387273

theorem fence_painting_problem (initial_people : ℕ) (initial_time : ℝ) (new_time : ℝ) :
  initial_people = 8 →
  initial_time = 3 →
  new_time = 2 →
  ∃ (new_people : ℕ), 
    (initial_people : ℝ) * initial_time = (new_people : ℝ) * new_time ∧ 
    new_people = 12 :=
by sorry

end NUMINAMATH_CALUDE_fence_painting_problem_l3872_387273


namespace NUMINAMATH_CALUDE_sqrt_calculation_l3872_387209

theorem sqrt_calculation : (Real.sqrt 12 - Real.sqrt (1/3)) * Real.sqrt 6 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l3872_387209


namespace NUMINAMATH_CALUDE_alyssa_cookies_l3872_387291

-- Define the number of cookies Aiyanna has
def aiyanna_cookies : ℕ := 140

-- Define the difference between Alyssa's and Aiyanna's cookies
def cookie_difference : ℕ := 11

-- Theorem stating Alyssa's number of cookies
theorem alyssa_cookies : 
  ∃ (a : ℕ), a = aiyanna_cookies + cookie_difference :=
by
  sorry

end NUMINAMATH_CALUDE_alyssa_cookies_l3872_387291


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3872_387202

theorem absolute_value_inequality (x : ℝ) :
  (4 ≤ |x + 2| ∧ |x + 2| ≤ 8) ↔ ((-10 ≤ x ∧ x ≤ -6) ∨ (2 ≤ x ∧ x ≤ 6)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3872_387202


namespace NUMINAMATH_CALUDE_b_5_times_b_9_equals_16_l3872_387221

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem b_5_times_b_9_equals_16 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h1 : arithmetic_sequence a)
  (h2 : 2 * a 2 - (a 7)^2 + 2 * a 12 = 0)
  (h3 : geometric_sequence b)
  (h4 : b 7 = a 7) :
  b 5 * b 9 = 16 := by
sorry

end NUMINAMATH_CALUDE_b_5_times_b_9_equals_16_l3872_387221


namespace NUMINAMATH_CALUDE_g_sum_property_l3872_387289

def g (x : ℝ) : ℝ := 3 * x^6 + 5 * x^4 - 6 * x^2 + 7

theorem g_sum_property : g 2 + g (-2) = 8 :=
by
  have h1 : g 2 = 4 := by sorry
  sorry

end NUMINAMATH_CALUDE_g_sum_property_l3872_387289


namespace NUMINAMATH_CALUDE_a_minus_b_value_l3872_387247

theorem a_minus_b_value (a b : ℚ) 
  (ha : |a| = 5) 
  (hb : |b| = 2) 
  (hab : |a + b| = a + b) : 
  a - b = 3 ∨ a - b = 7 :=
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l3872_387247


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l3872_387241

/-- A regular nonagon is a 9-sided regular polygon -/
def RegularNonagon : Type := Unit

/-- The number of vertices in a regular nonagon -/
def num_vertices : ℕ := 9

/-- The number of diagonals in a regular nonagon -/
def num_diagonals (n : RegularNonagon) : ℕ := (num_vertices.choose 2) - num_vertices

/-- The number of ways to choose 2 diagonals from all diagonals -/
def num_diagonal_pairs (n : RegularNonagon) : ℕ := (num_diagonals n).choose 2

/-- The number of ways to choose 4 vertices that form a convex quadrilateral -/
def num_intersecting_pairs (n : RegularNonagon) : ℕ := num_vertices.choose 4

/-- The probability that two randomly chosen diagonals intersect -/
def intersection_probability (n : RegularNonagon) : ℚ :=
  (num_intersecting_pairs n : ℚ) / (num_diagonal_pairs n : ℚ)

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  intersection_probability n = 14 / 39 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l3872_387241


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3872_387235

theorem stratified_sampling_theorem (total_sample : ℕ) 
  (school_A : ℕ) (school_B : ℕ) (school_C : ℕ) : 
  total_sample = 60 → 
  school_A = 180 → 
  school_B = 270 → 
  school_C = 90 → 
  (school_C * total_sample) / (school_A + school_B + school_C) = 10 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3872_387235


namespace NUMINAMATH_CALUDE_function_satisfies_condition_l3872_387233

-- Define the function y
def y (x : ℝ) : ℝ := x - 2

-- State the theorem
theorem function_satisfies_condition :
  y 1 = -1 := by sorry

end NUMINAMATH_CALUDE_function_satisfies_condition_l3872_387233


namespace NUMINAMATH_CALUDE_angle_rotation_l3872_387213

theorem angle_rotation (initial_angle rotation : ℝ) (h1 : initial_angle = 60) (h2 : rotation = 630) :
  (initial_angle + rotation) % 360 = 330 ∧ 360 - (initial_angle + rotation) % 360 = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_rotation_l3872_387213


namespace NUMINAMATH_CALUDE_ellipse_properties_l3872_387231

-- Define the ellipse C
def ellipse_C (x y : ℝ) (a : ℝ) : Prop :=
  x^2 / a^2 + y^2 / (7 - a^2) = 1 ∧ a > 0

-- Define the focal distance
def focal_distance (a : ℝ) : ℝ := 2

-- Define the standard form of the ellipse
def standard_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define a line passing through (4,0)
def line_through_R (x y : ℝ) (k : ℝ) : Prop :=
  y = k * (x - 4)

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the right focus
def right_focus : (ℝ × ℝ) := (1, 0)

-- Theorem statement
theorem ellipse_properties (a : ℝ) (k : ℝ) (x1 y1 x2 y2 : ℝ) :
  (∀ x y, ellipse_C x y a ↔ standard_ellipse x y) ∧
  (∀ x y, line_through_R x y k ∧ point_on_ellipse x y →
    ∃ xn yn xq yq,
      point_on_ellipse xn yn ∧
      point_on_ellipse xq yq ∧
      xn = x1 ∧ yn = -y1 ∧
      (yn - y2) * (xq - right_focus.1) = (yq - right_focus.2) * (xn - right_focus.1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3872_387231


namespace NUMINAMATH_CALUDE_find_n_l3872_387285

theorem find_n (n : ℕ) (h1 : Nat.lcm n 12 = 42) (h2 : Nat.gcd n 12 = 6) : n = 21 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l3872_387285


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l3872_387243

/-- For a hyperbola x²/a² - y²/b² = 1 with a > b, if the angle between asymptotes is 45°, then a/b = 1 -/
theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  (angle_between_asymptotes = Real.pi / 4) → 
  a / b = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l3872_387243


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3872_387249

theorem polynomial_coefficient_sum (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) : 
  (∀ x : ℝ, x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 3*x^2 - 2*x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃)) →
  b₁*c₁ + b₂*c₂ + b₃*c₃ = -2 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3872_387249


namespace NUMINAMATH_CALUDE_function_existence_iff_divisibility_l3872_387288

theorem function_existence_iff_divisibility (k a : ℕ) :
  (∃ f : ℕ → ℕ, ∀ n : ℕ, (f^[k] n = n + a)) ↔ (a ≥ 0 ∧ k ∣ a) :=
sorry

end NUMINAMATH_CALUDE_function_existence_iff_divisibility_l3872_387288


namespace NUMINAMATH_CALUDE_binomial_seven_four_l3872_387265

theorem binomial_seven_four : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_seven_four_l3872_387265


namespace NUMINAMATH_CALUDE_pyramid_sphere_radii_relation_main_theorem_l3872_387220

/-- A regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  R : ℝ  -- radius of circumscribed sphere
  r : ℝ  -- radius of inscribed sphere
  h : ℝ  -- height of the pyramid
  a : ℝ  -- half of the base edge length

/-- The theorem stating the relationship between R and r for a regular quadrilateral pyramid -/
theorem pyramid_sphere_radii_relation (p : RegularQuadrilateralPyramid) :
  p.R ≥ (Real.sqrt 2 + 1) * p.r := by
  sorry

/-- Main theorem to be proved -/
theorem main_theorem :
  ∀ p : RegularQuadrilateralPyramid, p.R ≥ (Real.sqrt 2 + 1) * p.r := by
  intro p
  exact pyramid_sphere_radii_relation p

end NUMINAMATH_CALUDE_pyramid_sphere_radii_relation_main_theorem_l3872_387220


namespace NUMINAMATH_CALUDE_solution_set_when_m_is_5_range_of_m_for_inequality_l3872_387283

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| + |x + 6|

-- Part I
theorem solution_set_when_m_is_5 :
  {x : ℝ | f 5 x ≤ 12} = {x : ℝ | -13/2 ≤ x ∧ x ≤ 11/2} := by sorry

-- Part II
theorem range_of_m_for_inequality :
  {m : ℝ | ∀ x : ℝ, f m x ≥ 7} = {m : ℝ | m ≤ -13 ∨ m ≥ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_is_5_range_of_m_for_inequality_l3872_387283


namespace NUMINAMATH_CALUDE_volleyball_tournament_equation_l3872_387279

/-- Represents a volleyball tournament. -/
structure VolleyballTournament where
  /-- The number of teams in the tournament. -/
  num_teams : ℕ
  /-- The total number of matches played. -/
  total_matches : ℕ
  /-- Each pair of teams plays against each other once. -/
  each_pair_plays_once : True

/-- Theorem stating the correct equation for the volleyball tournament. -/
theorem volleyball_tournament_equation (t : VolleyballTournament) 
  (h : t.total_matches = 28) : 
  (t.num_teams * (t.num_teams - 1)) / 2 = t.total_matches := by
  sorry

end NUMINAMATH_CALUDE_volleyball_tournament_equation_l3872_387279


namespace NUMINAMATH_CALUDE_f_of_two_equals_negative_twenty_six_l3872_387268

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_of_two_equals_negative_twenty_six 
  (a b : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8)
  (h2 : f (-2) = 10) :
  f 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_equals_negative_twenty_six_l3872_387268


namespace NUMINAMATH_CALUDE_savings_percentage_l3872_387284

/-- Represents the financial situation of a man over two years --/
structure FinancialSituation where
  income_year1 : ℝ
  savings_year1 : ℝ
  income_year2 : ℝ
  savings_year2 : ℝ

/-- The financial situation satisfies the given conditions --/
def satisfies_conditions (fs : FinancialSituation) : Prop :=
  fs.income_year1 > 0 ∧
  fs.savings_year1 > 0 ∧
  fs.income_year2 = 1.5 * fs.income_year1 ∧
  fs.savings_year2 = 2 * fs.savings_year1 ∧
  (fs.income_year1 - fs.savings_year1) + (fs.income_year2 - fs.savings_year2) = 2 * (fs.income_year1 - fs.savings_year1)

/-- The theorem stating that the man saved 50% of his income in the first year --/
theorem savings_percentage (fs : FinancialSituation) (h : satisfies_conditions fs) :
  fs.savings_year1 / fs.income_year1 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_savings_percentage_l3872_387284


namespace NUMINAMATH_CALUDE_river_width_proof_l3872_387232

/-- The width of a river where two men start from opposite banks, meet 340 meters from one bank
    on their forward journey, and 170 meters from the other bank on their backward journey. -/
def river_width : ℝ := 340

theorem river_width_proof (forward_meeting : ℝ) (backward_meeting : ℝ) 
  (h1 : forward_meeting = 340)
  (h2 : backward_meeting = 170)
  (h3 : forward_meeting + (river_width - forward_meeting) = river_width)
  (h4 : backward_meeting + (river_width - backward_meeting) = river_width) :
  river_width = 340 := by
  sorry

end NUMINAMATH_CALUDE_river_width_proof_l3872_387232


namespace NUMINAMATH_CALUDE_parabola_vertex_l3872_387238

/-- Given a parabola y = -x^2 + ax + b ≤ 0 with roots at x = -4 and x = 6,
    prove that its vertex is at (1, 25). -/
theorem parabola_vertex (a b : ℝ) :
  (∀ x, -x^2 + a*x + b ≤ 0 ↔ x ∈ Set.Ici 6 ∪ Set.Iic (-4)) →
  ∃ k, -1^2 + a*1 + b = k ∧ ∀ x, -x^2 + a*x + b ≤ k :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3872_387238


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l3872_387230

-- Problem 1
theorem problem_1 : -|-2/3 - 3/2| - |-1/5 + (-2/5)| = -83/30 := by sorry

-- Problem 2
theorem problem_2 : (-7.33) * 42.07 + (-2.07) * (-7.33) = -293.2 := by sorry

-- Problem 3
theorem problem_3 : -4 - 28 - (-19) + (-24) = -37 := by sorry

-- Problem 4
theorem problem_4 : -|-2023| - (-2023) + 2023 = 2023 := by sorry

-- Problem 5
theorem problem_5 : 19 * (31/32) * (-4) = -79 - 7/8 := by sorry

-- Problem 6
theorem problem_6 : (1/2 + 5/6 - 7/12) * (-36) = -27 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l3872_387230


namespace NUMINAMATH_CALUDE_quadratic_root_implies_b_l3872_387215

theorem quadratic_root_implies_b (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x - 6 = 0) ∧ (2^2 + b*2 - 6 = 0) → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_b_l3872_387215


namespace NUMINAMATH_CALUDE_tangent_trapezoid_EQ_length_l3872_387277

/-- Represents a trapezoid with a circle tangent to two sides --/
structure TangentTrapezoid where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ
  EQ : ℝ
  QF : ℝ
  EQ_QF_ratio : EQ / QF = 5 / 3

/-- The theorem stating the length of EQ in the given trapezoid --/
theorem tangent_trapezoid_EQ_length (t : TangentTrapezoid) 
  (h1 : t.EF = 150)
  (h2 : t.FG = 65)
  (h3 : t.GH = 35)
  (h4 : t.HE = 90)
  (h5 : t.EF = t.EQ + t.QF) :
  t.EQ = 375 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_trapezoid_EQ_length_l3872_387277


namespace NUMINAMATH_CALUDE_min_box_value_l3872_387204

theorem min_box_value (a b box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 45 * x^2 + box * x + 45) →
  a ≠ b ∧ b ≠ box ∧ a ≠ box →
  (∃ box_min : ℤ, box_min = 106 ∧ box ≥ box_min ∧
    (∀ a' b' box' : ℤ, 
      (∀ x, (a' * x + b') * (b' * x + a') = 45 * x^2 + box' * x + 45) →
      a' ≠ b' ∧ b' ≠ box' ∧ a' ≠ box' →
      box' ≥ box_min)) :=
sorry

end NUMINAMATH_CALUDE_min_box_value_l3872_387204


namespace NUMINAMATH_CALUDE_triangle_inradius_l3872_387280

theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h1 : p = 40) 
  (h2 : A = 50) 
  (h3 : A = r * p / 2) : r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l3872_387280


namespace NUMINAMATH_CALUDE_imaginary_number_product_l3872_387208

theorem imaginary_number_product (z : ℂ) (a : ℝ) : 
  (z.im ≠ 0 ∧ z.re = 0) → (Complex.I * z.im = z) → ((3 - Complex.I) * z = a + Complex.I) → a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_number_product_l3872_387208


namespace NUMINAMATH_CALUDE_unique_solution_for_prime_equation_l3872_387228

theorem unique_solution_for_prime_equation (p : ℕ) (hp : Prime p) :
  ∀ x y : ℕ+, p * (x - y) = x * y → x = p^2 - p ∧ y = p + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_prime_equation_l3872_387228


namespace NUMINAMATH_CALUDE_quarter_circles_sum_limit_l3872_387275

theorem quarter_circles_sum_limit (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |n * (π * D / (4 * n)) - (π * D / 4)| < ε :=
sorry

end NUMINAMATH_CALUDE_quarter_circles_sum_limit_l3872_387275


namespace NUMINAMATH_CALUDE_limit_cubic_difference_quotient_l3872_387292

theorem limit_cubic_difference_quotient :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → |(x^3 - 1) / (x - 1) - 3| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_cubic_difference_quotient_l3872_387292


namespace NUMINAMATH_CALUDE_situps_total_l3872_387286

/-- The number of sit-ups Barney can do in one minute -/
def barney_situps : ℕ := 45

/-- The number of sit-ups Carrie can do in one minute -/
def carrie_situps : ℕ := 2 * barney_situps

/-- The number of sit-ups Jerrie can do in one minute -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- The number of minutes Barney performs sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie performs sit-ups -/
def carrie_minutes : ℕ := 2

/-- The number of minutes Jerrie performs sit-ups -/
def jerrie_minutes : ℕ := 3

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ := barney_situps * barney_minutes + 
                        carrie_situps * carrie_minutes + 
                        jerrie_situps * jerrie_minutes

theorem situps_total : total_situps = 510 := by
  sorry

end NUMINAMATH_CALUDE_situps_total_l3872_387286


namespace NUMINAMATH_CALUDE_twelfth_term_of_arithmetic_progression_l3872_387201

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

/-- Theorem: The 12th term of an arithmetic progression with first term 2 and common difference 8 is 90 -/
theorem twelfth_term_of_arithmetic_progression :
  arithmeticProgressionTerm 2 8 12 = 90 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_arithmetic_progression_l3872_387201


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3872_387271

theorem arithmetic_calculations :
  (2 / 19 * 8 / 25 + 17 / 25 / (19 / 2) = 2 / 19) ∧
  (1 / 4 * 125 * 1 / 25 * 8 = 10) ∧
  ((1 / 3 + 1 / 4) / (1 / 2 - 1 / 3) = 7 / 2) ∧
  ((1 / 6 + 1 / 8) * 24 * 1 / 9 = 7 / 9) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3872_387271


namespace NUMINAMATH_CALUDE_existence_of_ones_divisible_by_2019_l3872_387282

theorem existence_of_ones_divisible_by_2019 :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, k > 0 ∧ (10^n - 1) / 9 = k * 2019) :=
sorry

end NUMINAMATH_CALUDE_existence_of_ones_divisible_by_2019_l3872_387282


namespace NUMINAMATH_CALUDE_smallest_sum_proof_l3872_387212

def is_valid_sum (a b c d e f : ℕ) : Prop :=
  a ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  b ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  c ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  d ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  e ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  f ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  100 ≤ a ∧ a ≤ 999 ∧
  100 ≤ d ∧ d ≤ 999 ∧
  (100 * a + 10 * b + c) + (100 * d + 10 * e + f) ≤ 1500

theorem smallest_sum_proof :
  ∀ a b c d e f : ℕ,
    is_valid_sum a b c d e f →
    (100 * a + 10 * b + c) + (100 * d + 10 * e + f) ≥ 417 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_proof_l3872_387212


namespace NUMINAMATH_CALUDE_max_fraction_sum_l3872_387297

theorem max_fraction_sum (A B C D : ℕ) : 
  A ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) → 
  B ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) → 
  C ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) → 
  D ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) → 
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  B ≠ 0 → D ≠ 0 →
  (A : ℚ) / B + (C : ℚ) / D ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_sum_l3872_387297


namespace NUMINAMATH_CALUDE_consecutive_sum_2016_l3872_387200

def is_valid_n (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a : ℕ, n * (2 * a + n - 1) = 4032

theorem consecutive_sum_2016 :
  {n : ℕ | is_valid_n n} = {3, 7, 9, 21, 63} :=
by sorry

end NUMINAMATH_CALUDE_consecutive_sum_2016_l3872_387200


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l3872_387251

theorem at_least_one_greater_than_one (a b c : ℝ) : 
  (a - 1) * (b - 1) * (c - 1) > 0 → (a > 1 ∨ b > 1 ∨ c > 1) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l3872_387251


namespace NUMINAMATH_CALUDE_max_pyramid_volume_is_four_l3872_387207

/-- A triangular prism with given ratios on its lateral edges -/
structure TriangularPrism where
  volume : ℝ
  AM_ratio : ℝ
  BN_ratio : ℝ
  CK_ratio : ℝ

/-- The maximum volume of a pyramid formed inside the prism -/
def max_pyramid_volume (prism : TriangularPrism) : ℝ := sorry

/-- Theorem stating the maximum volume of the pyramid MNKP -/
theorem max_pyramid_volume_is_four (prism : TriangularPrism) 
  (h1 : prism.volume = 16)
  (h2 : prism.AM_ratio = 1/2)
  (h3 : prism.BN_ratio = 1/3)
  (h4 : prism.CK_ratio = 1/4) :
  max_pyramid_volume prism = 4 := by sorry

end NUMINAMATH_CALUDE_max_pyramid_volume_is_four_l3872_387207


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3872_387218

/-- The quadratic function f(x) = x^2 - 14x + 45 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 45

theorem quadratic_minimum :
  (∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x) ∧
  (∃ (x : ℝ), f x = -4) ∧
  (∀ (y : ℝ), f y ≥ -4) ∧
  f 7 = -4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3872_387218


namespace NUMINAMATH_CALUDE_bernie_postcards_final_count_l3872_387246

/-- Calculates the number of postcards Bernie has after his transactions -/
def postcards_after_transactions (initial_postcards : ℕ) (sell_price : ℕ) (buy_price : ℕ) : ℕ :=
  let sold_postcards := initial_postcards / 2
  let remaining_postcards := initial_postcards - sold_postcards
  let money_earned := sold_postcards * sell_price
  let new_postcards := money_earned / buy_price
  remaining_postcards + new_postcards

/-- Theorem stating that Bernie will have 36 postcards after his transactions -/
theorem bernie_postcards_final_count :
  postcards_after_transactions 18 15 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_bernie_postcards_final_count_l3872_387246


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3872_387287

def A : Set ℕ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℕ := {1, 2, 3}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3872_387287


namespace NUMINAMATH_CALUDE_roots_sum_product_equal_l3872_387250

theorem roots_sum_product_equal (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 - (m+2)*x + m^2 = 0 ∧ 
    y^2 - (m+2)*y + m^2 = 0 ∧ 
    x + y = x * y) → 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_product_equal_l3872_387250


namespace NUMINAMATH_CALUDE_rachel_coloring_books_l3872_387234

/-- The number of pictures Rachel still has to color -/
def remaining_pictures (book1_pictures book2_pictures colored_pictures : ℕ) : ℕ :=
  book1_pictures + book2_pictures - colored_pictures

/-- Theorem: Rachel has 11 pictures left to color -/
theorem rachel_coloring_books :
  remaining_pictures 23 32 44 = 11 := by
  sorry

end NUMINAMATH_CALUDE_rachel_coloring_books_l3872_387234


namespace NUMINAMATH_CALUDE_amy_jeremy_age_ratio_l3872_387206

/-- Proves that the ratio of Amy's age to Jeremy's age is 1:3 given the specified conditions -/
theorem amy_jeremy_age_ratio :
  ∀ (amy_age jeremy_age chris_age : ℕ),
    jeremy_age = 66 →
    amy_age + jeremy_age + chris_age = 132 →
    chris_age = 2 * amy_age →
    (amy_age : ℚ) / jeremy_age = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_amy_jeremy_age_ratio_l3872_387206


namespace NUMINAMATH_CALUDE_islet_cell_transplant_indicators_l3872_387261

/-- Represents the type of transplantation performed -/
inductive TransplantationType
| IsletCell

/-- Represents the possible indicators for determining cure and medication needed -/
inductive Indicator
| UrineSugar
| Insulin
| Antiallergics
| BloodSugar
| Immunosuppressants

/-- Represents a pair of indicators -/
structure IndicatorPair :=
  (first second : Indicator)

/-- Function to determine the correct indicators based on transplantation type -/
def correctIndicators (transplantType : TransplantationType) : IndicatorPair :=
  match transplantType with
  | TransplantationType.IsletCell => ⟨Indicator.BloodSugar, Indicator.Immunosuppressants⟩

/-- Theorem stating that for islet cell transplantation, the correct indicators are blood sugar and immunosuppressants -/
theorem islet_cell_transplant_indicators :
  correctIndicators TransplantationType.IsletCell = ⟨Indicator.BloodSugar, Indicator.Immunosuppressants⟩ :=
by sorry

end NUMINAMATH_CALUDE_islet_cell_transplant_indicators_l3872_387261


namespace NUMINAMATH_CALUDE_sum_of_squares_l3872_387226

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_cubes_eq_sum_fifth : a^3 + b^3 + c^3 = a^5 + b^5 + c^5) :
  a^2 + b^2 + c^2 = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3872_387226


namespace NUMINAMATH_CALUDE_gum_sharing_l3872_387242

/-- The number of pieces of gum each person will receive when shared equally --/
def gum_per_person (john cole aubrey maria : ℕ) : ℕ :=
  (john + cole + aubrey + maria) / 6

/-- Theorem stating that given the initial gum distribution, each person will receive 34 pieces --/
theorem gum_sharing (john cole aubrey maria : ℕ) 
  (h_john : john = 54)
  (h_cole : cole = 45)
  (h_aubrey : aubrey = 37)
  (h_maria : maria = 70) :
  gum_per_person john cole aubrey maria = 34 := by
  sorry

end NUMINAMATH_CALUDE_gum_sharing_l3872_387242


namespace NUMINAMATH_CALUDE_lcm_144_132_l3872_387216

theorem lcm_144_132 : lcm 144 132 = 1584 := by
  sorry

end NUMINAMATH_CALUDE_lcm_144_132_l3872_387216


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3872_387219

theorem trigonometric_inequality (a b : ℝ) : 
  (∀ x : ℝ, |a * Real.sin x + b * Real.sin (2 * x)| ≤ 1) →
  ((a = 4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = 4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3))) :=
by sorry


end NUMINAMATH_CALUDE_trigonometric_inequality_l3872_387219


namespace NUMINAMATH_CALUDE_square_sum_theorem_l3872_387244

theorem square_sum_theorem (r s : ℝ) (h1 : r * s = 16) (h2 : r + s = 10) : r^2 + s^2 = 68 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l3872_387244


namespace NUMINAMATH_CALUDE_positive_roots_l3872_387239

theorem positive_roots (x y z : ℝ) 
  (sum_pos : x + y + z > 0) 
  (sum_prod_pos : x*y + y*z + z*x > 0) 
  (prod_pos : x*y*z > 0) : 
  x > 0 ∧ y > 0 ∧ z > 0 := by
sorry

end NUMINAMATH_CALUDE_positive_roots_l3872_387239


namespace NUMINAMATH_CALUDE_track_length_is_600_l3872_387263

/-- Represents a circular running track with two runners -/
structure CircularTrack where
  length : ℝ
  brenda_speed : ℝ
  sally_speed : ℝ

/-- The conditions of the problem -/
def problem_conditions (track : CircularTrack) : Prop :=
  ∃ (first_meeting_time second_meeting_time : ℝ),
    -- Brenda runs 120 meters before first meeting
    track.brenda_speed * first_meeting_time = 120 ∧
    -- Sally runs (length/2 - 120) meters before first meeting
    track.sally_speed * first_meeting_time = track.length / 2 - 120 ∧
    -- Sally runs an additional 180 meters between meetings
    track.sally_speed * (second_meeting_time - first_meeting_time) = 180 ∧
    -- Brenda's position at second meeting
    track.brenda_speed * second_meeting_time =
      track.length - (track.length / 2 - 120 + 180)

/-- The theorem to be proven -/
theorem track_length_is_600 (track : CircularTrack) :
  problem_conditions track → track.length = 600 := by
  sorry


end NUMINAMATH_CALUDE_track_length_is_600_l3872_387263


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3872_387245

/-- The parabola is defined by the equation y^2 = 8x -/
def is_parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (2, 0)

/-- The equation of the directrix -/
def directrix_x : ℝ := -2

/-- The distance from a point to a vertical line -/
def distance_to_line (p : ℝ × ℝ) (line_x : ℝ) : ℝ :=
  |p.1 - line_x|

theorem parabola_focus_directrix_distance :
  distance_to_line focus directrix_x = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3872_387245


namespace NUMINAMATH_CALUDE_concert_audience_fraction_l3872_387259

/-- The fraction of the audience for the second band at a concert -/
def fraction_second_band (total_audience : ℕ) (under_30_percent : ℚ) 
  (women_percent : ℚ) (men_under_30 : ℕ) : ℚ :=
  2 / 3

theorem concert_audience_fraction 
  (total_audience : ℕ) 
  (under_30_percent : ℚ) 
  (women_percent : ℚ) 
  (men_under_30 : ℕ) : 
  fraction_second_band total_audience under_30_percent women_percent men_under_30 = 2 / 3 :=
by
  sorry

#check concert_audience_fraction 150 (1/2) (3/5) 20

end NUMINAMATH_CALUDE_concert_audience_fraction_l3872_387259


namespace NUMINAMATH_CALUDE_mean_of_three_numbers_l3872_387270

theorem mean_of_three_numbers (x y z : ℝ) 
  (h1 : (x + y) / 2 = 5)
  (h2 : (y + z) / 2 = 9)
  (h3 : (z + x) / 2 = 10) :
  (x + y + z) / 3 = 8 := by sorry

end NUMINAMATH_CALUDE_mean_of_three_numbers_l3872_387270


namespace NUMINAMATH_CALUDE_work_completion_time_l3872_387281

-- Define the work rates for each person
def amit_rate : ℚ := 1 / 15
def ananthu_rate : ℚ := 1 / 90
def chandra_rate : ℚ := 1 / 45

-- Define the number of days each person worked alone
def amit_solo_days : ℕ := 3
def ananthu_solo_days : ℕ := 6

-- Define the combined work rate of all three people
def combined_rate : ℚ := amit_rate + ananthu_rate + chandra_rate

-- Theorem statement
theorem work_completion_time : 
  let work_done_solo := amit_rate * amit_solo_days + ananthu_rate * ananthu_solo_days
  let remaining_work := 1 - work_done_solo
  let days_together := (remaining_work / combined_rate).ceil
  amit_solo_days + ananthu_solo_days + days_together = 17 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3872_387281


namespace NUMINAMATH_CALUDE_tower_block_count_l3872_387267

/-- The total number of blocks in a tower after adding more blocks -/
def total_blocks (initial : Float) (added : Float) : Float :=
  initial + added

/-- Theorem: The total number of blocks is the sum of initial and added blocks -/
theorem tower_block_count (initial : Float) (added : Float) :
  total_blocks initial added = initial + added := by
  sorry

end NUMINAMATH_CALUDE_tower_block_count_l3872_387267


namespace NUMINAMATH_CALUDE_sports_club_members_l3872_387294

theorem sports_club_members (B T Both Neither : ℕ) 
  (hB : B = 17)
  (hT : T = 19)
  (hBoth : Both = 11)
  (hNeither : Neither = 2) :
  B + T - Both + Neither = 27 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l3872_387294


namespace NUMINAMATH_CALUDE_nested_square_root_fourth_power_l3872_387205

theorem nested_square_root_fourth_power :
  (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1))))^4 = 2 + 2 * Real.sqrt 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_fourth_power_l3872_387205


namespace NUMINAMATH_CALUDE_symmetrical_line_passes_through_point_l3872_387248

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The line y = x -/
def lineYEqX : Line := ⟨1, -1, 0⟩

/-- Get the symmetrical line with respect to y = x -/
def symmetricalLine (l : Line) : Line :=
  ⟨l.b, l.a, l.c⟩

theorem symmetrical_line_passes_through_point :
  let l₁ : Line := ⟨2, 1, -1⟩  -- y = -2x + 1 rewritten as 2x + y - 1 = 0
  let l₂ := symmetricalLine l₁
  let p : Point := ⟨3, -1⟩
  pointOnLine l₂ p := by sorry

end NUMINAMATH_CALUDE_symmetrical_line_passes_through_point_l3872_387248
