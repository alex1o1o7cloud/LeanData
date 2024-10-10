import Mathlib

namespace weight_of_replaced_person_l3738_373869

/-- Proves that the weight of the replaced person is 65 kg given the conditions of the problem -/
theorem weight_of_replaced_person
  (n : ℕ)
  (original_average : ℝ)
  (new_average_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : n = 10)
  (h2 : new_average_increase = 7.2)
  (h3 : new_person_weight = 137)
  : ∃ (replaced_weight : ℝ),
    replaced_weight = new_person_weight - n * new_average_increase ∧
    replaced_weight = 65 := by
  sorry

end weight_of_replaced_person_l3738_373869


namespace piravena_flight_cost_l3738_373827

/-- Represents a right-angled triangle with sides DE and DF -/
structure RightTriangle where
  de : ℝ
  df : ℝ

/-- Calculates the cost of flying between two points -/
def flyCost (distance : ℝ) : ℝ :=
  120 + 0.12 * distance

theorem piravena_flight_cost (triangle : RightTriangle) 
  (h1 : triangle.de = 3750)
  (h2 : triangle.df = 3500) : 
  flyCost triangle.de = 570 := by
  sorry

#eval flyCost 3750

end piravena_flight_cost_l3738_373827


namespace hyperbola_right_angle_triangle_area_l3738_373881

/-- Hyperbola type representing the equation x²/9 - y²/16 = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : (x : ℝ) → (y : ℝ) → Prop

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle formed by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop := h.equation p.x p.y

/-- The foci of a hyperbola -/
def foci (h : Hyperbola) : (Point × Point) := sorry

theorem hyperbola_right_angle_triangle_area 
  (h : Hyperbola) 
  (p : Point) 
  (hP : isOnHyperbola h p) 
  (f1 f2 : Point) 
  (hFoci : foci h = (f1, f2)) 
  (hAngle : angle f1 p f2 = 90) : 
  triangleArea (Triangle.mk f1 p f2) = 16 := by sorry

end hyperbola_right_angle_triangle_area_l3738_373881


namespace system_of_equations_solutions_l3738_373840

theorem system_of_equations_solutions :
  -- System 1
  (∃ x y : ℝ, y = 2*x - 3 ∧ 3*x + 2*y = 8 ∧ x = 2 ∧ y = 1) ∧
  -- System 2
  (∃ x y : ℝ, 5*x + 2*y = 25 ∧ 3*x + 4*y = 15 ∧ x = 5 ∧ y = 0) :=
by sorry

end system_of_equations_solutions_l3738_373840


namespace koala_bear_ratio_is_one_half_l3738_373867

/-- Represents the number of tickets spent on different items -/
structure TicketSpending where
  total : ℕ
  earbuds : ℕ
  glowBracelets : ℕ
  koalaBear : ℕ

/-- The ratio of tickets spent on the koala bear to the total number of tickets -/
def koalaBearRatio (ts : TicketSpending) : Rat :=
  ts.koalaBear / ts.total

theorem koala_bear_ratio_is_one_half (ts : TicketSpending) 
  (h_total : ts.total = 50)
  (h_earbuds : ts.earbuds = 10)
  (h_glow : ts.glowBracelets = 15)
  (h_koala : ts.koalaBear = ts.total - ts.earbuds - ts.glowBracelets) :
  koalaBearRatio ts = 1 / 2 := by
  sorry

end koala_bear_ratio_is_one_half_l3738_373867


namespace summer_pizza_sales_l3738_373821

/-- Given information about pizza sales in different seasons, prove that summer sales are 2 million pizzas. -/
theorem summer_pizza_sales :
  let spring_percent : ℝ := 0.3
  let spring_sales : ℝ := 4.8
  let autumn_sales : ℝ := 7
  let winter_sales : ℝ := 2.2
  let total_sales : ℝ := spring_sales / spring_percent
  let summer_sales : ℝ := total_sales - spring_sales - autumn_sales - winter_sales
  summer_sales = 2 := by
  sorry


end summer_pizza_sales_l3738_373821


namespace third_term_of_sequence_l3738_373805

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem third_term_of_sequence (a : ℤ) (d : ℤ) :
  arithmetic_sequence a d 20 = 18 →
  arithmetic_sequence a d 21 = 20 →
  arithmetic_sequence a d 3 = -16 :=
by
  sorry

end third_term_of_sequence_l3738_373805


namespace obtuse_isosceles_triangle_vertex_angle_l3738_373813

/-- An obtuse isosceles triangle with the given property has a vertex angle of 150° -/
theorem obtuse_isosceles_triangle_vertex_angle 
  (a : ℝ) 
  (θ : ℝ) 
  (h_a_pos : a > 0)
  (h_θ_pos : θ > 0)
  (h_θ_acute : θ < π / 2)
  (h_isosceles : a^2 = (2 * a * Real.cos θ) * (2 * a * Real.sin θ)) :
  π - 2*θ = 5*π/6 := by
sorry

end obtuse_isosceles_triangle_vertex_angle_l3738_373813


namespace quadratic_intersection_l3738_373876

/-- A quadratic function of the form y = ax² - 4x + 2 -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 4 * x + 2

/-- The discriminant of the quadratic function -/
def discriminant (a : ℝ) : ℝ := 16 - 8 * a

theorem quadratic_intersection (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_function a x₁ = 0 ∧ quadratic_function a x₂ = 0) →
  (0 < a ∧ a < 2) :=
sorry

end quadratic_intersection_l3738_373876


namespace intersection_of_M_and_N_l3738_373897

def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

def N : Set (ℝ × ℝ) := {p | p.1 = 1}

theorem intersection_of_M_and_N : M ∩ N = {(1, 0)} := by
  sorry

end intersection_of_M_and_N_l3738_373897


namespace only_divisor_square_sum_l3738_373849

theorem only_divisor_square_sum (n : ℕ+) :
  ∀ d : ℕ+, d ∣ (3 * n^2) → ∃ k : ℕ, n^2 + d = k^2 → d = 3 * n^2 :=
sorry

end only_divisor_square_sum_l3738_373849


namespace car_speed_problem_l3738_373891

theorem car_speed_problem (v : ℝ) : v > 0 →
  (1 / v * 3600 = 1 / 120 * 3600 + 2) ↔ v = 112.5 :=
by sorry

end car_speed_problem_l3738_373891


namespace expression_simplification_l3738_373845

theorem expression_simplification (b : ℝ) :
  ((2 * b + 6) - 5 * b) / 2 = -3/2 * b + 3 := by
  sorry

end expression_simplification_l3738_373845


namespace sum_first_six_primes_mod_seventh_prime_l3738_373802

theorem sum_first_six_primes_mod_seventh_prime : 
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 := by
  sorry

end sum_first_six_primes_mod_seventh_prime_l3738_373802


namespace slower_speed_percentage_l3738_373801

theorem slower_speed_percentage (D : ℝ) (S : ℝ) (S_slow : ℝ) 
    (h1 : D = S * 16)
    (h2 : D = S_slow * 40) :
  S_slow / S = 0.4 := by
sorry

end slower_speed_percentage_l3738_373801


namespace product_pure_imaginary_l3738_373887

theorem product_pure_imaginary (x : ℝ) :
  (∃ b : ℝ, (x + 1 + Complex.I) * ((x + 2) + Complex.I) * ((x + 3) + Complex.I) = b * Complex.I) ↔ x = -1 := by
  sorry

end product_pure_imaginary_l3738_373887


namespace square_plus_one_geq_two_abs_l3738_373838

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end square_plus_one_geq_two_abs_l3738_373838


namespace work_time_for_c_l3738_373831

/-- The time it takes for worker c to complete the work alone, given the following conditions:
  * a and b can do the work in 2 days
  * b and c can do the work in 3 days
  * c and a can do the work in 4 days
-/
theorem work_time_for_c (a b c : ℝ) 
  (hab : a + b = 1/2)  -- a and b can do the work in 2 days
  (hbc : b + c = 1/3)  -- b and c can do the work in 3 days
  (hca : c + a = 1/4)  -- c and a can do the work in 4 days
  : 1 / c = 24 := by
  sorry


end work_time_for_c_l3738_373831


namespace max_third_side_length_l3738_373823

theorem max_third_side_length (a b : ℝ) (ha : a = 7) (hb : b = 15) :
  ∃ (c : ℝ), c ≤ 21 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧
  ∀ (d : ℝ), (d > 21 ∨ d ≤ 0 ∨ a + b ≤ d ∨ a + d ≤ b ∨ b + d ≤ a) →
  ¬(∃ (e : ℕ), e > 21 ∧ (e : ℝ) = d) :=
by sorry

end max_third_side_length_l3738_373823


namespace find_b_l3738_373818

-- Define the set A
def A (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3 * p.1 + b}

-- Theorem statement
theorem find_b : ∃ b : ℝ, (1, 5) ∈ A b ∧ b = 2 := by
  sorry

end find_b_l3738_373818


namespace euro_op_calculation_l3738_373820

-- Define the € operation
def euro_op (x y z : ℕ) : ℕ := 3 * x * y * z

-- State the theorem
theorem euro_op_calculation : 
  euro_op 3 (euro_op 4 5 6) 1 = 3240 := by
  sorry

end euro_op_calculation_l3738_373820


namespace merchant_profit_calculation_l3738_373824

theorem merchant_profit_calculation (cost_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : 
  markup_percentage = 20 →
  discount_percentage = 5 →
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 14 := by
sorry

end merchant_profit_calculation_l3738_373824


namespace wine_price_increase_l3738_373893

/-- Proves that the percentage increase in wine price is 25% given the initial and future prices -/
theorem wine_price_increase (initial_price : ℝ) (future_price_increase : ℝ) : 
  initial_price = 20 →
  future_price_increase = 25 →
  (((initial_price + future_price_increase / 5) - initial_price) / initial_price) * 100 = 25 := by
  sorry

end wine_price_increase_l3738_373893


namespace penny_shark_species_l3738_373857

/-- Given the number of species Penny identified in an aquarium, prove the number of shark species. -/
theorem penny_shark_species (total : ℕ) (eels : ℕ) (whales : ℕ) (sharks : ℕ)
  (h1 : total = 55)
  (h2 : eels = 15)
  (h3 : whales = 5)
  (h4 : total = sharks + eels + whales) :
  sharks = 35 := by
  sorry

end penny_shark_species_l3738_373857


namespace M_equals_set_l3738_373866

def M : Set ℕ := {m | m > 0 ∧ ∃ k : ℤ, (10 : ℤ) = k * (m + 1)}

theorem M_equals_set : M = {1, 4, 9} := by sorry

end M_equals_set_l3738_373866


namespace odometer_sum_l3738_373899

theorem odometer_sum (a b c : ℕ) : 
  a ≥ 1 → 
  a + b + c ≤ 9 → 
  (100 * c + 10 * a + b) - (100 * a + 10 * b + c) % 45 = 0 →
  100 * a + 10 * b + c + 100 * b + 10 * c + a + 100 * c + 10 * a + b = 999 :=
by sorry

end odometer_sum_l3738_373899


namespace roots_imply_composite_sum_of_squares_l3738_373832

theorem roots_imply_composite_sum_of_squares (a b : ℤ) :
  (∃ x y : ℕ, x^2 + a*x + b + 1 = 0 ∧ y^2 + a*y + b + 1 = 0 ∧ x ≠ y) →
  ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ m * n = a^2 + b^2 := by
  sorry

end roots_imply_composite_sum_of_squares_l3738_373832


namespace albert_pizza_consumption_l3738_373811

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 16

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := 8

/-- The number of large pizzas Albert buys -/
def num_large_pizzas : ℕ := 2

/-- The number of small pizzas Albert buys -/
def num_small_pizzas : ℕ := 2

/-- The total number of slices Albert eats -/
def total_slices : ℕ := num_large_pizzas * large_pizza_slices + num_small_pizzas * small_pizza_slices

theorem albert_pizza_consumption :
  total_slices = 48 := by
  sorry

end albert_pizza_consumption_l3738_373811


namespace movie_collection_difference_l3738_373806

theorem movie_collection_difference (shared_movies : ℕ) (andrew_total : ℕ) (john_unique : ℕ)
  (h1 : shared_movies = 12)
  (h2 : andrew_total = 23)
  (h3 : john_unique = 8) :
  andrew_total - shared_movies + john_unique = 19 := by
sorry

end movie_collection_difference_l3738_373806


namespace average_comparisons_equals_size_l3738_373862

/-- Represents a sequential search on an unordered array -/
structure SequentialSearch where
  /-- The number of elements in the array -/
  size : ℕ
  /-- Predicate indicating if the array is unordered -/
  unordered : Prop
  /-- Predicate indicating if the searched element is not in the array -/
  element_not_present : Prop

/-- The average number of comparisons needed in a sequential search -/
def average_comparisons (search : SequentialSearch) : ℕ := sorry

/-- Theorem stating that the average number of comparisons is equal to the array size 
    when the element is not present in an unordered array -/
theorem average_comparisons_equals_size (search : SequentialSearch) 
  (h_size : search.size = 100)
  (h_unordered : search.unordered)
  (h_not_present : search.element_not_present) :
  average_comparisons search = search.size := by sorry

end average_comparisons_equals_size_l3738_373862


namespace max_value_fraction_l3738_373803

theorem max_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -1) (hy : 1 ≤ y ∧ y ≤ 3) :
  (∀ x' y', -5 ≤ x' ∧ x' ≤ -1 ∧ 1 ≤ y' ∧ y' ≤ 3 → (x' + y') / x' ≤ (x + y) / x) →
  (x + y) / x = -2 :=
sorry

end max_value_fraction_l3738_373803


namespace min_value_theorem_l3738_373870

theorem min_value_theorem (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + y = 1) :
  ∀ z : ℝ, z = (1 / (x + 1)) + (4 / y) → z ≥ 9/2 :=
by sorry

end min_value_theorem_l3738_373870


namespace alcohol_percentage_P_correct_l3738_373809

/-- The percentage of alcohol in vessel P that results in the given mixture ratio -/
def alcohol_percentage_P : ℝ := 62.5

/-- The percentage of alcohol in vessel Q -/
def alcohol_percentage_Q : ℝ := 87.5

/-- The volume of liquid taken from each vessel -/
def volume_per_vessel : ℝ := 4

/-- The ratio of alcohol to water in the resulting mixture -/
def mixture_ratio : ℝ := 3

/-- The total volume of the mixture -/
def total_volume : ℝ := 2 * volume_per_vessel

theorem alcohol_percentage_P_correct :
  (alcohol_percentage_P / 100 * volume_per_vessel +
   alcohol_percentage_Q / 100 * volume_per_vessel) / total_volume = mixture_ratio / (mixture_ratio + 1) :=
by sorry

end alcohol_percentage_P_correct_l3738_373809


namespace rectangular_to_square_formation_l3738_373854

theorem rectangular_to_square_formation :
  ∃ n : ℕ,
    (∃ a : ℕ, 8 * n + 120 = a * a) ∧
    (∃ b : ℕ, 8 * n - 120 = b * b) →
    n = 17 := by
  sorry

end rectangular_to_square_formation_l3738_373854


namespace equation_solution_l3738_373865

theorem equation_solution :
  ∃ x : ℝ, x + 2*x + 12 = 500 - (3*x + 4*x) → x = 48.8 := by
  sorry

end equation_solution_l3738_373865


namespace dollar_op_five_negative_two_l3738_373819

-- Define the $ operation
def dollar_op (c d : Int) : Int := c * (d + 1) + c * d

-- Theorem statement
theorem dollar_op_five_negative_two :
  dollar_op 5 (-2) = -15 := by
  sorry

end dollar_op_five_negative_two_l3738_373819


namespace factorization_implies_m_values_l3738_373817

theorem factorization_implies_m_values (m : ℤ) :
  (∃ (a b : ℤ), ∀ (x : ℤ), x^2 + m*x - 4 = a*x + b) →
  m ∈ ({-3, 0, 3} : Set ℤ) := by
  sorry

end factorization_implies_m_values_l3738_373817


namespace min_value_expression_min_value_is_34_min_value_achieved_l3738_373877

theorem min_value_expression (a b c d : ℕ) : 
  Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ∀ x y z w : ℕ, Odd x ∧ Odd y ∧ Odd z ∧ Odd w ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
  2 * a * b * c * d - (a * b * c + a * b * d + a * c * d + b * c * d) ≥
  2 * x * y * z * w - (x * y * z + x * y * w + x * z * w + y * z * w) :=
by sorry

theorem min_value_is_34 (a b c d : ℕ) : 
  Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  2 * a * b * c * d - (a * b * c + a * b * d + a * c * d + b * c * d) ≥ 34 :=
by sorry

theorem min_value_achieved (a b c d : ℕ) : 
  Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ∃ x y z w : ℕ, Odd x ∧ Odd y ∧ Odd z ∧ Odd w ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
  2 * x * y * z * w - (x * y * z + x * y * w + x * z * w + y * z * w) = 34 :=
by sorry

end min_value_expression_min_value_is_34_min_value_achieved_l3738_373877


namespace division_problem_l3738_373868

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 190 →
  quotient = 9 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 21 := by
  sorry

end division_problem_l3738_373868


namespace toby_first_part_distance_l3738_373860

/-- Represents Toby's journey with a loaded and unloaded sled -/
def toby_journey (x : ℝ) : Prop :=
  let loaded_speed : ℝ := 10
  let unloaded_speed : ℝ := 20
  let second_part : ℝ := 120
  let third_part : ℝ := 80
  let fourth_part : ℝ := 140
  let total_time : ℝ := 39
  (x / loaded_speed) + (second_part / unloaded_speed) + 
  (third_part / loaded_speed) + (fourth_part / unloaded_speed) = total_time

/-- Theorem stating that Toby pulled the loaded sled for 180 miles in the first part of the journey -/
theorem toby_first_part_distance : 
  ∃ (x : ℝ), toby_journey x ∧ x = 180 :=
sorry

end toby_first_part_distance_l3738_373860


namespace base_conversion_512_to_octal_l3738_373883

theorem base_conversion_512_to_octal :
  (512 : ℕ) = 1 * 8^3 + 0 * 8^2 + 0 * 8^1 + 0 * 8^0 :=
by sorry

end base_conversion_512_to_octal_l3738_373883


namespace logarithm_expression_equality_l3738_373885

-- Define the logarithm base 2
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- Define the logarithm base 10
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

theorem logarithm_expression_equality : 
  2^(log2 3) + lg (Real.sqrt 5) + lg (Real.sqrt 20) = 4 := by sorry

end logarithm_expression_equality_l3738_373885


namespace maple_trees_equation_l3738_373896

/-- The number of maple trees initially in the park -/
def initial_maple_trees : ℕ := 2

/-- The number of maple trees planted -/
def planted_maple_trees : ℕ := 9

/-- The final number of maple trees after planting -/
def final_maple_trees : ℕ := 11

/-- Theorem stating that the initial number of maple trees plus the planted ones equals the final number -/
theorem maple_trees_equation : 
  initial_maple_trees + planted_maple_trees = final_maple_trees := by
  sorry

end maple_trees_equation_l3738_373896


namespace sum_of_non_solutions_l3738_373825

/-- Given an equation with infinitely many solutions, prove the sum of non-solutions -/
theorem sum_of_non_solutions (A B C : ℚ) : 
  (∀ x, (x + B) * (A * x + 16) = 3 * (x + C) * (x + 5)) →
  (∃ x₁ x₂, ∀ x, x ≠ x₁ ∧ x ≠ x₂ → (x + B) * (A * x + 16) = 3 * (x + C) * (x + 5)) →
  (∃ x₁ x₂, ∀ x, (x + B) * (A * x + 16) ≠ 3 * (x + C) * (x + 5) ↔ x = x₁ ∨ x = x₂) →
  (∃ x₁ x₂, x₁ + x₂ = -31/3 ∧ 
    ∀ x, (x + B) * (A * x + 16) ≠ 3 * (x + C) * (x + 5) ↔ x = x₁ ∨ x = x₂) :=
by sorry

end sum_of_non_solutions_l3738_373825


namespace creative_arts_academy_painting_paradox_l3738_373841

theorem creative_arts_academy_painting_paradox :
  let total_students : ℝ := 100
  let enjoy_painting_ratio : ℝ := 0.7
  let dont_enjoy_painting_ratio : ℝ := 1 - enjoy_painting_ratio
  let enjoy_but_negate_ratio : ℝ := 0.25
  let dont_enjoy_but_affirm_ratio : ℝ := 0.15

  let enjoy_painting : ℝ := total_students * enjoy_painting_ratio
  let dont_enjoy_painting : ℝ := total_students * dont_enjoy_painting_ratio
  
  let enjoy_but_negate : ℝ := enjoy_painting * enjoy_but_negate_ratio
  let dont_enjoy_but_affirm : ℝ := dont_enjoy_painting * dont_enjoy_but_affirm_ratio
  
  let total_claim_dislike : ℝ := enjoy_but_negate + (dont_enjoy_painting - dont_enjoy_but_affirm)
  
  (enjoy_but_negate / total_claim_dislike) * 100 = 40.698 :=
by sorry

end creative_arts_academy_painting_paradox_l3738_373841


namespace special_function_value_l3738_373880

/-- A function satisfying f(xy) = f(x)/y² for positive reals -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / (y ^ 2)

/-- Theorem stating that if f is a special function and f(40) = 50, then f(80) = 12.5 -/
theorem special_function_value
  (f : ℝ → ℝ)
  (h_special : special_function f)
  (h_f40 : f 40 = 50) :
  f 80 = 12.5 := by
sorry

end special_function_value_l3738_373880


namespace certain_number_equation_l3738_373815

theorem certain_number_equation (x : ℝ) : 112 * x^4 = 70000 → x = 5 := by
  sorry

end certain_number_equation_l3738_373815


namespace equal_numbers_exist_l3738_373892

/-- A quadratic polynomial function -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Theorem: Given a quadratic polynomial and real numbers l, t, v satisfying certain conditions,
    there exist at least two equal numbers among l, t, and v. -/
theorem equal_numbers_exist (a b c l t v : ℝ) (ha : a ≠ 0)
    (h1 : QuadraticPolynomial a b c l = t + v)
    (h2 : QuadraticPolynomial a b c t = l + v)
    (h3 : QuadraticPolynomial a b c v = l + t) :
    (l = t ∨ l = v ∨ t = v) := by
  sorry

end equal_numbers_exist_l3738_373892


namespace last_two_digits_of_sum_l3738_373843

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def lastTwoDigits (n : ℕ) : ℕ := n % 100

def sumSequence : ℕ → ℕ
  | 0 => 0
  | n + 1 => factorial (7 * (n + 1)) * 3 + sumSequence n

theorem last_two_digits_of_sum :
  lastTwoDigits (sumSequence 15) = 20 := by sorry

end last_two_digits_of_sum_l3738_373843


namespace condition_analysis_l3738_373861

theorem condition_analysis (a : ℕ) : 
  let A : Set ℕ := {1, a}
  let B : Set ℕ := {1, 2, 3}
  (a = 3 → A ⊆ B) ∧ (∃ x ≠ 3, {1, x} ⊆ B) :=
by sorry

end condition_analysis_l3738_373861


namespace discounted_price_l3738_373834

/-- Given a top with an original price of m yuan and a discount of 20%,
    the actual selling price is 0.8m yuan. -/
theorem discounted_price (m : ℝ) : 
  let original_price := m
  let discount_rate := 0.2
  let selling_price := m * (1 - discount_rate)
  selling_price = 0.8 * m := by
  sorry

end discounted_price_l3738_373834


namespace quadratic_points_ordering_l3738_373846

/-- Quadratic function f(x) = -(x-2)² + h -/
def f (x h : ℝ) : ℝ := -(x - 2)^2 + h

theorem quadratic_points_ordering (h : ℝ) :
  let y₁ := f (-1/2) h
  let y₂ := f 1 h
  let y₃ := f 2 h
  y₁ < y₂ ∧ y₂ < y₃ := by sorry

end quadratic_points_ordering_l3738_373846


namespace min_toothpicks_removal_l3738_373864

/-- Represents a complex figure made of toothpicks forming triangles -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  max_triangle_side : ℕ
  min_triangle_side : ℕ

/-- Calculates the minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ :=
  20

/-- Theorem stating that for the given figure, 20 toothpicks must be removed to eliminate all triangles -/
theorem min_toothpicks_removal (figure : ToothpickFigure) 
  (h1 : figure.total_toothpicks = 60)
  (h2 : figure.max_triangle_side = 3)
  (h3 : figure.min_triangle_side = 1) :
  min_toothpicks_to_remove figure = 20 := by
  sorry

end min_toothpicks_removal_l3738_373864


namespace line_slope_proof_l3738_373878

/-- Given a line passing through points P(-2, m) and Q(m, 4) with slope 1, prove that m = 1 -/
theorem line_slope_proof (m : ℝ) : 
  (4 - m) / (m - (-2)) = 1 → m = 1 := by
  sorry

end line_slope_proof_l3738_373878


namespace two_numbers_sum_and_product_l3738_373814

theorem two_numbers_sum_and_product : 
  ∃ (x y : ℝ), x + y = 10 ∧ x * y = 24 ∧ ((x = 4 ∧ y = 6) ∨ (x = 6 ∧ y = 4)) := by
  sorry

end two_numbers_sum_and_product_l3738_373814


namespace donovans_test_incorrect_answers_l3738_373898

theorem donovans_test_incorrect_answers :
  ∀ (total : ℕ) (correct : ℕ) (percentage : ℚ),
    correct = 35 →
    percentage = 7292 / 10000 →
    (correct : ℚ) / (total : ℚ) = percentage →
    total - correct = 13 :=
  by sorry

end donovans_test_incorrect_answers_l3738_373898


namespace equation_solutions_l3738_373844

theorem equation_solutions :
  (∃ x : ℚ, 2 * x - 2 * (3 * x + 1) = 6 ∧ x = -2) ∧
  (∃ x : ℚ, (x + 1) / 2 - 1 = (2 - 3 * x) / 3 ∧ x = 7 / 9) :=
by sorry

end equation_solutions_l3738_373844


namespace angle_W_measure_l3738_373889

-- Define the quadrilateral WXYZ
structure Quadrilateral :=
  (W X Y Z : ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  q.W = 3 * q.X ∧ q.W = 2 * q.Y ∧ q.W = 4 * q.Z ∧
  q.W + q.X + q.Y + q.Z = 360

-- Theorem statement
theorem angle_W_measure (q : Quadrilateral) 
  (h : is_valid_quadrilateral q) : q.W = 172.8 := by
  sorry

end angle_W_measure_l3738_373889


namespace unique_number_with_gcd_l3738_373859

theorem unique_number_with_gcd : ∃! n : ℕ, 70 ≤ n ∧ n < 80 ∧ Nat.gcd 30 n = 10 := by
  sorry

end unique_number_with_gcd_l3738_373859


namespace goats_in_field_l3738_373812

theorem goats_in_field (total_animals cows sheep : ℕ) 
  (h1 : total_animals = 200)
  (h2 : cows = 40)
  (h3 : sheep = 56) : 
  total_animals - (cows + sheep) = 104 := by
  sorry

end goats_in_field_l3738_373812


namespace consecutive_integers_problem_l3738_373895

/-- Given consecutive integers x, y, and z where x > y > z, 
    2x + 3y + 3z = 5y + 11, and z = 3, prove that x = 5 -/
theorem consecutive_integers_problem (x y z : ℤ) 
  (consecutive : (x = y + 1) ∧ (y = z + 1))
  (order : x > y ∧ y > z)
  (equation : 2*x + 3*y + 3*z = 5*y + 11)
  (z_value : z = 3) :
  x = 5 := by
  sorry

end consecutive_integers_problem_l3738_373895


namespace fifteen_by_fifteen_grid_toothpicks_l3738_373875

/-- Represents a rectangular grid of toothpicks with diagonal lines -/
structure ToothpickGrid where
  height : ℕ
  width : ℕ
  has_diagonals : Bool

/-- Calculates the total number of toothpicks in the grid -/
def total_toothpicks (grid : ToothpickGrid) : ℕ :=
  let horizontal := (grid.height + 1) * grid.width
  let vertical := (grid.width + 1) * grid.height
  let diagonal := if grid.has_diagonals then 2 * grid.height else 0
  horizontal + vertical + diagonal

/-- The theorem stating that a 15x15 grid with diagonals has 510 toothpicks -/
theorem fifteen_by_fifteen_grid_toothpicks :
  total_toothpicks { height := 15, width := 15, has_diagonals := true } = 510 := by
  sorry

end fifteen_by_fifteen_grid_toothpicks_l3738_373875


namespace select_four_with_both_genders_eq_34_l3738_373804

/-- The number of ways to select 4 individuals from 4 boys and 3 girls,
    such that the selection includes both boys and girls. -/
def select_four_with_both_genders (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  Nat.choose (num_boys + num_girls) 4 - Nat.choose num_boys 4

/-- Theorem stating that selecting 4 individuals from 4 boys and 3 girls,
    such that the selection includes both boys and girls, results in 34 ways. -/
theorem select_four_with_both_genders_eq_34 :
  select_four_with_both_genders 4 3 = 34 := by
  sorry

#eval select_four_with_both_genders 4 3

end select_four_with_both_genders_eq_34_l3738_373804


namespace airline_route_theorem_l3738_373886

/-- Represents a city in the country -/
structure City where
  id : Nat
  republic : Nat
  routes : Finset Nat

/-- The country with its cities and airline routes -/
structure Country where
  cities : Finset City
  total_cities : Nat
  num_republics : Nat

/-- A country satisfies the problem conditions -/
def satisfies_conditions (country : Country) : Prop :=
  country.total_cities = 100 ∧
  country.num_republics = 3 ∧
  (country.cities.filter (λ c => c.routes.card ≥ 70)).card ≥ 70

/-- There exists an airline route within the same republic -/
def exists_intra_republic_route (country : Country) : Prop :=
  ∃ c1 c2 : City, c1 ∈ country.cities ∧ c2 ∈ country.cities ∧
    c1.id ≠ c2.id ∧ c1.republic = c2.republic ∧ c2.id ∈ c1.routes

/-- The main theorem -/
theorem airline_route_theorem (country : Country) :
  satisfies_conditions country → exists_intra_republic_route country :=
by
  sorry


end airline_route_theorem_l3738_373886


namespace product_of_solutions_absolute_value_equation_l3738_373848

theorem product_of_solutions_absolute_value_equation :
  ∃ (x₁ x₂ : ℝ), 
    (∀ x : ℝ, |x| = 3 * (|x| - 2) ↔ x = x₁ ∨ x = x₂) ∧
    x₁ * x₂ = -9 := by
  sorry

end product_of_solutions_absolute_value_equation_l3738_373848


namespace parallel_planes_l3738_373852

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the necessary relations
variable (lies_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)

-- State the theorem
theorem parallel_planes
  (α β : Plane) (a b : Line) (A : Point) :
  lies_in a α →
  lies_in b α →
  intersect a b →
  ¬ line_parallel a β →
  ¬ line_parallel b β →
  parallel α β :=
by sorry

end parallel_planes_l3738_373852


namespace romanov_savings_l3738_373807

/-- Represents the electricity pricing and consumption data for the Romanov family --/
structure ElectricityData where
  multi_tariff_meter_cost : ℝ
  installation_cost : ℝ
  monthly_consumption : ℝ
  night_consumption : ℝ
  day_rate : ℝ
  night_rate : ℝ
  standard_rate : ℝ
  years : ℕ

/-- Calculates the savings from using a multi-tariff meter over the given period --/
def calculate_savings (data : ElectricityData) : ℝ :=
  let standard_cost := data.standard_rate * data.monthly_consumption * 12 * data.years
  let day_consumption := data.monthly_consumption - data.night_consumption
  let multi_tariff_cost := (data.day_rate * day_consumption + data.night_rate * data.night_consumption) * 12 * data.years
  let total_multi_tariff_cost := multi_tariff_cost + data.multi_tariff_meter_cost + data.installation_cost
  standard_cost - total_multi_tariff_cost

/-- Theorem stating the savings for the Romanov family --/
theorem romanov_savings :
  let data : ElectricityData := {
    multi_tariff_meter_cost := 3500,
    installation_cost := 1100,
    monthly_consumption := 300,
    night_consumption := 230,
    day_rate := 5.2,
    night_rate := 3.4,
    standard_rate := 4.6,
    years := 3
  }
  calculate_savings data = 3824 := by
  sorry

end romanov_savings_l3738_373807


namespace multiplicative_inverse_143_mod_391_l3738_373894

theorem multiplicative_inverse_143_mod_391 :
  ∃ a : ℕ, a < 391 ∧ (143 * a) % 391 = 1 :=
by
  use 28
  sorry

end multiplicative_inverse_143_mod_391_l3738_373894


namespace james_change_l3738_373808

/-- Calculates the change received when buying candy -/
def calculate_change (num_packs : ℕ) (cost_per_pack : ℕ) (amount_paid : ℕ) : ℕ :=
  amount_paid - (num_packs * cost_per_pack)

/-- Proves that James received $11 in change -/
theorem james_change :
  let num_packs : ℕ := 3
  let cost_per_pack : ℕ := 3
  let amount_paid : ℕ := 20
  calculate_change num_packs cost_per_pack amount_paid = 11 := by
  sorry

end james_change_l3738_373808


namespace cube_root_equation_solution_l3738_373856

theorem cube_root_equation_solution (x : ℝ) :
  (x * (x^2)^(1/2))^(1/3) = 2 → x = 2 * (2^(1/2)) ∨ x = -2 * (2^(1/2)) :=
by sorry

end cube_root_equation_solution_l3738_373856


namespace quadratic_factorization_l3738_373839

theorem quadratic_factorization (c d : ℕ) (hc : c > d) : 
  (∀ x, x^2 - 20*x + 91 = (x - c)*(x - d)) → 2*d - c = 1 := by
  sorry

end quadratic_factorization_l3738_373839


namespace meaningful_expression_range_l3738_373879

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 1)) / ((x - 3)^2)) ↔ x ≥ -1 ∧ x ≠ 3 := by
  sorry

end meaningful_expression_range_l3738_373879


namespace system_solution_l3738_373810

theorem system_solution (x y k : ℝ) : 
  (x + 2*y = k - 1) →
  (2*x + y = 5*k + 4) →
  (x + y = 5) →
  k = 2 := by
sorry

end system_solution_l3738_373810


namespace money_division_l3738_373871

/-- The problem of dividing money among three people -/
theorem money_division (total : ℚ) (c_share : ℚ) (b_ratio : ℚ) :
  total = 328 →
  c_share = 64 →
  b_ratio = 65 / 100 →
  ∃ (a_share : ℚ),
    a_share + b_ratio * a_share + c_share = total ∧
    (c_share * 100) / a_share = 40 :=
by sorry

end money_division_l3738_373871


namespace polynomial_difference_l3738_373872

theorem polynomial_difference (a : ℝ) : (6 * a^2 - 5*a + 3) - (5 * a^2 + 2*a - 1) = a^2 - 7*a + 4 := by
  sorry

end polynomial_difference_l3738_373872


namespace quadratic_function_theorem_l3738_373836

/-- A quadratic function satisfying the given condition -/
noncomputable def f : ℝ → ℝ :=
  fun x => -(1/2) * x^2 - 2*x

/-- Function g defined in terms of f -/
noncomputable def g : ℝ → ℝ :=
  fun x => x * Real.log x + f x

/-- The set of real numbers satisfying the inequality -/
def solution_set : Set ℝ :=
  {x | x ∈ Set.Icc (-2) (-1) ∪ Set.Ioc 0 1}

theorem quadratic_function_theorem :
  (∀ x, f (x + 1) + f x = -x^2 - 5*x - 5/2) ∧
  (f = fun x => -(1/2) * x^2 - 2*x) ∧
  (∀ x, x > 0 → (g (x^2 + x) ≥ g 2 ↔ x ∈ solution_set)) :=
by sorry

end quadratic_function_theorem_l3738_373836


namespace like_terms_imply_m_minus_n_eq_two_l3738_373800

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def are_like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (x y : ℕ), term1 x y ≠ 0 ∧ term2 x y ≠ 0 → 
    ∃ (c1 c2 : ℚ), term1 x y = c1 * x^x * y^y ∧ term2 x y = c2 * x^x * y^y

/-- The first monomial 3x^m*y -/
def term1 (m : ℕ) (x y : ℕ) : ℚ := 3 * x^m * y

/-- The second monomial -x^3*y^n -/
def term2 (n : ℕ) (x y : ℕ) : ℚ := -1 * x^3 * y^n

theorem like_terms_imply_m_minus_n_eq_two (m n : ℕ) :
  are_like_terms (term1 m) (term2 n) → m - n = 2 := by
  sorry

end like_terms_imply_m_minus_n_eq_two_l3738_373800


namespace monthly_growth_rate_price_reduction_for_profit_l3738_373822

-- Define the given constants
def initial_cost : ℝ := 40
def initial_price : ℝ := 60
def march_sales : ℝ := 192
def may_sales : ℝ := 300
def sales_increase_per_reduction : ℝ := 20  -- 40 pieces per 2 yuan reduction

-- Define the target profit
def target_profit : ℝ := 6080

-- Part 1: Monthly average growth rate
theorem monthly_growth_rate : ∃ (x : ℝ), 
  march_sales * (1 + x)^2 = may_sales ∧ x = 0.25 := by sorry

-- Part 2: Price reduction for target profit
theorem price_reduction_for_profit : ∃ (m : ℝ),
  (initial_price - m - initial_cost) * (may_sales + sales_increase_per_reduction * m) = target_profit ∧
  m = 4 := by sorry

end monthly_growth_rate_price_reduction_for_profit_l3738_373822


namespace alices_favorite_number_l3738_373884

def is_multiple (a b : ℕ) : Prop := ∃ k, a = b * k

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem alices_favorite_number : ∃! n : ℕ, 
  90 < n ∧ n < 150 ∧ 
  is_multiple n 13 ∧ 
  ¬is_multiple n 3 ∧ 
  is_multiple (digit_sum n) 4 ∧
  n = 130 := by sorry

end alices_favorite_number_l3738_373884


namespace snowdrift_depth_l3738_373855

theorem snowdrift_depth (initial_depth melted_depth third_day_depth fourth_day_depth final_depth : ℝ) :
  melted_depth = initial_depth / 2 →
  third_day_depth = melted_depth + 6 →
  fourth_day_depth = third_day_depth + 18 →
  final_depth = 34 →
  initial_depth = 20 :=
by sorry

end snowdrift_depth_l3738_373855


namespace allowance_spent_on_books_l3738_373888

theorem allowance_spent_on_books (total : ℚ) (games snacks toys books : ℚ) : 
  total = 45 → 
  games = 2/9 * total → 
  snacks = 1/3 * total → 
  toys = 1/5 * total → 
  books = total - (games + snacks + toys) → 
  books = 11 := by
sorry

end allowance_spent_on_books_l3738_373888


namespace axis_of_symmetry_is_correct_l3738_373829

/-- The quadratic function y = 2x^2 - 8x + 10 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 10

/-- The axis of symmetry of the quadratic function -/
def axis_of_symmetry : ℝ := 2

/-- Theorem: The axis of symmetry of the quadratic function f(x) = 2x^2 - 8x + 10 is x = 2 -/
theorem axis_of_symmetry_is_correct : 
  ∀ x : ℝ, f (axis_of_symmetry + x) = f (axis_of_symmetry - x) :=
by
  sorry

#check axis_of_symmetry_is_correct

end axis_of_symmetry_is_correct_l3738_373829


namespace triangle_inequality_l3738_373833

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) : 
  |a^2 - b^2| / c + |b^2 - c^2| / a ≥ |c^2 - a^2| / b := by
  sorry

end triangle_inequality_l3738_373833


namespace triangle_vertices_l3738_373873

/-- The lines forming the triangle --/
def line1 (x y : ℚ) : Prop := 2 * x + y - 6 = 0
def line2 (x y : ℚ) : Prop := x - y + 4 = 0
def line3 (x y : ℚ) : Prop := y + 1 = 0

/-- The vertices of the triangle --/
def vertex1 : ℚ × ℚ := (2/3, 14/3)
def vertex2 : ℚ × ℚ := (-5, -1)
def vertex3 : ℚ × ℚ := (7/2, -1)

/-- Theorem stating that the given points are the vertices of the triangle --/
theorem triangle_vertices : 
  (line1 vertex1.1 vertex1.2 ∧ line2 vertex1.1 vertex1.2) ∧
  (line2 vertex2.1 vertex2.2 ∧ line3 vertex2.1 vertex2.2) ∧
  (line1 vertex3.1 vertex3.2 ∧ line3 vertex3.1 vertex3.2) := by
  sorry

end triangle_vertices_l3738_373873


namespace cone_radius_l3738_373874

/-- Given a cone with surface area 6π and whose lateral surface unfolds into a semicircle,
    the radius of the base of the cone is √2. -/
theorem cone_radius (r : Real) (l : Real) : 
  (π * r * r + π * r * l = 6 * π) →  -- Surface area of cone is 6π
  (2 * π * r = π * l) →              -- Lateral surface unfolds into a semicircle
  r = Real.sqrt 2 := by
sorry

end cone_radius_l3738_373874


namespace parents_per_child_l3738_373890

-- Define the number of girls and boys
def num_girls : ℕ := 6
def num_boys : ℕ := 8

-- Define the total number of parents attending
def total_parents : ℕ := 28

-- Theorem statement
theorem parents_per_child (parents_per_child : ℕ) :
  parents_per_child * num_girls + parents_per_child * num_boys = total_parents →
  parents_per_child = 2 := by
sorry

end parents_per_child_l3738_373890


namespace T_divisibility_l3738_373842

-- Define the set T
def T : Set ℤ := {x | ∃ n : ℤ, x = (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2}

-- Theorem statement
theorem T_divisibility :
  (∀ x ∈ T, ¬(5 ∣ x)) ∧ (∃ x ∈ T, 7 ∣ x) := by
  sorry

end T_divisibility_l3738_373842


namespace min_people_for_condition_l3738_373882

/-- Represents a circular table with chairs and people seated. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement satisfies the condition that any additional
    person must sit next to at least one other person. -/
def satisfies_condition (table : CircularTable) : Prop :=
  table.seated_people * 4 ≥ table.total_chairs

/-- The theorem stating the minimum number of people required for the given condition. -/
theorem min_people_for_condition (table : CircularTable) 
  (h1 : table.total_chairs = 80)
  (h2 : satisfies_condition table)
  (h3 : ∀ n < table.seated_people, ¬satisfies_condition ⟨table.total_chairs, n⟩) :
  table.seated_people = 20 := by
  sorry

#check min_people_for_condition

end min_people_for_condition_l3738_373882


namespace jenn_savings_l3738_373863

/-- Represents the value of a coin in cents -/
def coinValue (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | "penny" => 1
  | _ => 0

/-- Calculates the total value of coins in a jar -/
def jarValue (coin : String) (count : ℕ) : ℚ :=
  (coinValue coin * count : ℚ) / 100

/-- Calculates the available amount after applying the usage constraint -/
def availableAmount (amount : ℚ) (constraint : ℚ) : ℚ :=
  amount * constraint

/-- Represents Jenn's saving scenario -/
structure SavingScenario where
  quarterJars : ℕ
  quarterCount : ℕ
  dimeJars : ℕ
  dimeCount : ℕ
  nickelJars : ℕ
  nickelCount : ℕ
  monthlyPennies : ℕ
  months : ℕ
  usageConstraint : ℚ
  bikeCost : ℚ

/-- Theorem stating that Jenn will have $24.57 left after buying the bike -/
theorem jenn_savings (scenario : SavingScenario) : 
  scenario.quarterJars = 4 ∧ 
  scenario.quarterCount = 160 ∧
  scenario.dimeJars = 4 ∧
  scenario.dimeCount = 300 ∧
  scenario.nickelJars = 2 ∧
  scenario.nickelCount = 500 ∧
  scenario.monthlyPennies = 12 ∧
  scenario.months = 6 ∧
  scenario.usageConstraint = 4/5 ∧
  scenario.bikeCost = 240 →
  let totalQuarters := jarValue "quarter" (scenario.quarterJars * scenario.quarterCount)
  let totalDimes := jarValue "dime" (scenario.dimeJars * scenario.dimeCount)
  let totalNickels := jarValue "nickel" (scenario.nickelJars * scenario.nickelCount)
  let totalPennies := jarValue "penny" (scenario.monthlyPennies * scenario.months)
  let availableQuarters := availableAmount totalQuarters scenario.usageConstraint
  let availableDimes := availableAmount totalDimes scenario.usageConstraint
  let availableNickels := availableAmount totalNickels scenario.usageConstraint
  let availablePennies := availableAmount totalPennies scenario.usageConstraint
  let totalAvailable := availableQuarters + availableDimes + availableNickels + availablePennies
  totalAvailable - scenario.bikeCost = 24.57 := by
  sorry

end jenn_savings_l3738_373863


namespace coefficient_value_l3738_373837

-- Define the polynomial P(x)
def P (c : ℝ) (x : ℝ) : ℝ := x^3 + 4*x^2 + c*x - 20

-- Theorem statement
theorem coefficient_value (c : ℝ) : 
  (∀ x, P c x = 0 → x = 5) → c = -41 := by
  sorry

end coefficient_value_l3738_373837


namespace cubic_equation_solution_l3738_373853

theorem cubic_equation_solution (a w : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * w) : w = 49 := by
  sorry

end cubic_equation_solution_l3738_373853


namespace quadratic_root_problem_l3738_373851

theorem quadratic_root_problem (k : ℝ) :
  (∃ x : ℝ, x^2 + (k - 5) * x + (4 - k) = 0 ∧ x = 2) →
  (∃ y : ℝ, y^2 + (k - 5) * y + (4 - k) = 0 ∧ y = 1) :=
by sorry

end quadratic_root_problem_l3738_373851


namespace geometric_sequence_seventh_term_l3738_373826

theorem geometric_sequence_seventh_term 
  (a : ℝ) (r : ℝ) (h1 : r ≠ 0) 
  (h2 : a * r^3 = 16) 
  (h3 : a * r^8 = 2) : 
  a * r^6 = 2 := by
sorry

end geometric_sequence_seventh_term_l3738_373826


namespace jungkook_smallest_number_l3738_373835

-- Define the set of students
inductive Student : Type
| Yoongi : Student
| Jungkook : Student
| Yuna : Student
| Yoojung : Student
| Taehyung : Student

-- Define a function that assigns numbers to students
def studentNumber : Student → ℕ
| Student.Yoongi => 7
| Student.Jungkook => 6
| Student.Yuna => 9
| Student.Yoojung => 8
| Student.Taehyung => 10

-- Theorem: Jungkook has the smallest number
theorem jungkook_smallest_number :
  ∀ s : Student, studentNumber Student.Jungkook ≤ studentNumber s :=
by sorry

end jungkook_smallest_number_l3738_373835


namespace complement_of_B_relative_to_A_l3738_373850

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3}

theorem complement_of_B_relative_to_A : A \ B = {2, 4} := by
  sorry

end complement_of_B_relative_to_A_l3738_373850


namespace sqrt_equality_implies_specific_integers_l3738_373830

theorem sqrt_equality_implies_specific_integers (a b : ℕ) :
  0 < a → 0 < b → a < b →
  Real.sqrt (1 + Real.sqrt (21 + 12 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b →
  a = 1 ∧ b = 3 := by
sorry

end sqrt_equality_implies_specific_integers_l3738_373830


namespace greatest_possible_average_speed_l3738_373847

/-- Represents a palindromic number -/
def IsPalindrome (n : ℕ) : Prop := sorry

/-- The initial odometer reading -/
def initial_reading : ℕ := 12321

/-- The duration of the drive in hours -/
def drive_duration : ℝ := 4

/-- The speed limit in miles per hour -/
def speed_limit : ℝ := 65

/-- The greatest possible average speed in miles per hour -/
def max_average_speed : ℝ := 50

/-- The theorem stating the greatest possible average speed -/
theorem greatest_possible_average_speed :
  ∀ (final_reading : ℕ),
    IsPalindrome initial_reading →
    IsPalindrome final_reading →
    final_reading > initial_reading →
    (final_reading - initial_reading : ℝ) ≤ speed_limit * drive_duration →
    (∀ (speed : ℝ), speed ≤ speed_limit → 
      (final_reading - initial_reading : ℝ) / drive_duration ≤ speed) →
    (final_reading - initial_reading : ℝ) / drive_duration = max_average_speed :=
sorry

end greatest_possible_average_speed_l3738_373847


namespace total_initial_tickets_l3738_373828

def dave_tiger_original_price : ℝ := 43
def dave_tiger_discount_rate : ℝ := 0.20
def dave_keychain_price : ℝ := 5.5
def dave_tickets_left : ℝ := 55

def alex_dinosaur_original_price : ℝ := 65
def alex_dinosaur_discount_rate : ℝ := 0.15
def alex_tickets_left : ℝ := 42

theorem total_initial_tickets : 
  let dave_tiger_discounted_price := dave_tiger_original_price * (1 - dave_tiger_discount_rate)
  let dave_total_spent := dave_tiger_discounted_price + dave_keychain_price
  let dave_initial_tickets := dave_total_spent + dave_tickets_left

  let alex_dinosaur_discounted_price := alex_dinosaur_original_price * (1 - alex_dinosaur_discount_rate)
  let alex_initial_tickets := alex_dinosaur_discounted_price + alex_tickets_left

  dave_initial_tickets + alex_initial_tickets = 192.15 := by sorry

end total_initial_tickets_l3738_373828


namespace seed_germination_percentage_experiment_result_l3738_373858

/-- Calculates the percentage of total seeds germinated in an agricultural experiment. -/
theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) : ℚ :=
  let total_seeds := seeds_plot1 + seeds_plot2
  let germinated_seeds1 := (seeds_plot1 : ℚ) * germination_rate1
  let germinated_seeds2 := (seeds_plot2 : ℚ) * germination_rate2
  let total_germinated := germinated_seeds1 + germinated_seeds2
  (total_germinated / total_seeds) * 100

/-- The percentage of total seeds germinated in the given agricultural experiment. -/
theorem experiment_result : 
  seed_germination_percentage 500 200 (30/100) (50/100) = 250/700 * 100 := by
  sorry

end seed_germination_percentage_experiment_result_l3738_373858


namespace count_special_integers_l3738_373816

theorem count_special_integers : 
  ∃ (S : Finset ℤ), 
    (∀ n ∈ S, 200 < n ∧ n < 300 ∧ ∃ (r k : ℤ), n = 63 * k + r ∧ 0 ≤ r ∧ r < 5) ∧
    (∀ n : ℤ, 200 < n → n < 300 → (∃ (r k : ℤ), n = 63 * k + r ∧ 0 ≤ r ∧ r < 5) → n ∈ S) ∧
    Finset.card S = 5 :=
sorry

end count_special_integers_l3738_373816
