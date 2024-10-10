import Mathlib

namespace digit_sum_last_digit_match_l1138_113815

theorem digit_sum_last_digit_match 
  (digits : Finset ℕ) 
  (h_digits_size : digits.card = 7) 
  (h_digits_distinct : ∀ (a b : ℕ), a ∈ digits → b ∈ digits → a ≠ b → a ≠ b) 
  (h_digits_range : ∀ d ∈ digits, d < 10) :
  ∀ n : ℕ, ∃ (a b : ℕ), a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ (a + b) % 10 = n % 10 := by
  sorry


end digit_sum_last_digit_match_l1138_113815


namespace miss_evans_class_size_l1138_113829

theorem miss_evans_class_size :
  let total_contribution : ℕ := 90
  let class_funds : ℕ := 14
  let student_contribution : ℕ := 4
  let remaining_contribution := total_contribution - class_funds
  let num_students := remaining_contribution / student_contribution
  num_students = 19 := by sorry

end miss_evans_class_size_l1138_113829


namespace sum_of_terms_l1138_113848

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_terms (a : ℕ → ℕ) : 
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 10 →
  a 3 = 17 →
  a 6 = 31 →
  a 4 + a 5 + a 7 = 93 :=
by
  sorry

end sum_of_terms_l1138_113848


namespace inequality_range_l1138_113876

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → x^2 - 2*x + 1 - a^2 < 0) → 
  (a > 3 ∨ a < -3) :=
by sorry

end inequality_range_l1138_113876


namespace factorial_inequality_l1138_113827

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem factorial_inequality :
  factorial (factorial 100) < (factorial 99)^(factorial 100) * (factorial 100)^(factorial 99) :=
by sorry

end factorial_inequality_l1138_113827


namespace square_sum_given_product_and_sum_l1138_113831

theorem square_sum_given_product_and_sum (r s : ℝ) 
  (h1 : r * s = 16) 
  (h2 : r + s = 8) : 
  r^2 + s^2 = 32 := by
sorry

end square_sum_given_product_and_sum_l1138_113831


namespace correct_regression_l1138_113820

/-- Represents a linear regression equation -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Checks if two variables are positively correlated -/
def positively_correlated (x y : ℝ → ℝ) : Prop := sorry

/-- Calculates the sample mean of a variable -/
def sample_mean (x : ℝ → ℝ) : ℝ := sorry

/-- Checks if a linear regression equation is valid for given data -/
def is_valid_regression (reg : LinearRegression) (x y : ℝ → ℝ) : Prop :=
  positively_correlated x y ∧
  sample_mean x = 3 ∧
  sample_mean y = 3.5 ∧
  reg.slope > 0 ∧
  reg.slope * (sample_mean x) + reg.intercept = sample_mean y

theorem correct_regression :
  is_valid_regression ⟨0.4, 2.3⟩ (λ _ => sorry) (λ _ => sorry) := by sorry

end correct_regression_l1138_113820


namespace complex_fraction_problem_l1138_113807

theorem complex_fraction_problem (x y : ℂ) 
  (h : (x + y) / (x - y) - (x - y) / (x + y) = 4) :
  (x^5 + y^5) / (x^5 - y^5) + (x^5 - y^5) / (x^5 + y^5) = 130 / 17 := by
  sorry

end complex_fraction_problem_l1138_113807


namespace prob_second_red_three_two_l1138_113825

/-- Represents a bag of colored balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- Calculates the probability of drawing a red ball on the second draw,
    given that the first ball drawn is red -/
def prob_second_red_given_first_red (b : Bag) : ℚ :=
  if b.red > 0 then
    (b.red - 1) / (b.red + b.white - 1)
  else
    0

/-- Theorem stating that for a bag with 3 red and 2 white balls,
    the probability of drawing a red ball on the second draw,
    given that the first ball drawn is red, is 1/2 -/
theorem prob_second_red_three_two : 
  prob_second_red_given_first_red ⟨3, 2⟩ = 1/2 := by
  sorry

end prob_second_red_three_two_l1138_113825


namespace ninth_triangle_shaded_fraction_l1138_113857

/- Define the sequence of shaded triangles -/
def shaded_triangles (n : ℕ) : ℕ := 2 * n - 1

/- Define the sequence of total triangles -/
def total_triangles (n : ℕ) : ℕ := 4^(n - 1)

/- Theorem statement -/
theorem ninth_triangle_shaded_fraction :
  shaded_triangles 9 / total_triangles 9 = 17 / 65536 := by
  sorry

end ninth_triangle_shaded_fraction_l1138_113857


namespace infinite_series_sum_l1138_113841

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) is equal to 55/12 -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * n - 2) / (n * (n + 1) * (n + 3))) = 55 / 12 :=
by sorry

end infinite_series_sum_l1138_113841


namespace bicycle_speed_problem_l1138_113896

/-- Proves that given the conditions of the bicycle problem, student B's speed is 12 km/h -/
theorem bicycle_speed_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) 
  (h1 : distance = 12)
  (h2 : speed_ratio = 1.2)
  (h3 : time_difference = 1/6) : 
  ∃ (speed_B : ℝ), speed_B = 12 ∧ 
    distance / speed_B - distance / (speed_ratio * speed_B) = time_difference := by
  sorry

end bicycle_speed_problem_l1138_113896


namespace max_piles_is_30_l1138_113877

/-- Represents a configuration of stone piles -/
structure StonePiles where
  piles : List Nat
  sum_stones : (piles.sum = 660)
  valid_ratio : ∀ i j, i < piles.length → j < piles.length → 2 * piles[i]! > piles[j]!

/-- The maximum number of piles that can be formed -/
def max_piles : Nat := 30

/-- Theorem stating that 30 is the maximum number of piles -/
theorem max_piles_is_30 :
  ∀ sp : StonePiles, sp.piles.length ≤ max_piles :=
by sorry

end max_piles_is_30_l1138_113877


namespace nonzero_digits_after_decimal_l1138_113837

theorem nonzero_digits_after_decimal (n : ℕ) (d : ℕ) (h : d > 0) :
  let frac := (72 : ℚ) / ((2^4 * 3^6) : ℚ)
  ∃ (a b c : ℕ) (r : ℚ),
    frac = (a : ℚ) / 10 + (b : ℚ) / 100 + (c : ℚ) / 1000 + r ∧
    0 < a ∧ a < 10 ∧
    0 < b ∧ b < 10 ∧
    0 < c ∧ c < 10 ∧
    0 ≤ r ∧ r < 1/1000 :=
by sorry

end nonzero_digits_after_decimal_l1138_113837


namespace even_function_order_l1138_113851

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def is_monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem even_function_order (f : ℝ → ℝ) (h1 : is_even f) 
  (h2 : ∀ x, f (2 + x) = f (2 - x)) 
  (h3 : is_monotone_decreasing f (-2) 0) :
  f 5 < f (Real.sqrt 2) ∧ f (Real.sqrt 2) < f (-1.5) := by
  sorry

end even_function_order_l1138_113851


namespace slope_angle_of_line_l1138_113897

theorem slope_angle_of_line (x y : ℝ) : 
  x - y + 3 = 0 → Real.arctan 1 = π / 4 := by
  sorry

end slope_angle_of_line_l1138_113897


namespace quadrilateral_trapezoid_or_parallelogram_l1138_113882

/-- A quadrilateral with areas of triangles formed by diagonals -/
structure Quadrilateral where
  s₁ : ℝ
  s₂ : ℝ
  s₃ : ℝ
  s₄ : ℝ
  area_positive : s₁ > 0 ∧ s₂ > 0 ∧ s₃ > 0 ∧ s₄ > 0

/-- Definition of a trapezoid or parallelogram based on triangle areas -/
def is_trapezoid_or_parallelogram (q : Quadrilateral) : Prop :=
  q.s₁ = q.s₃ ∨ q.s₂ = q.s₄

/-- The main theorem -/
theorem quadrilateral_trapezoid_or_parallelogram (q : Quadrilateral) :
  (q.s₁ + q.s₂) * (q.s₃ + q.s₄) = (q.s₁ + q.s₄) * (q.s₂ + q.s₃) →
  is_trapezoid_or_parallelogram q :=
by sorry


end quadrilateral_trapezoid_or_parallelogram_l1138_113882


namespace geometric_sequence_ratio_l1138_113859

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_decr : ∀ n, a (n + 1) < a n)
  (h_geom : geometric_sequence a)
  (h_prod : a 2 * a 8 = 6)
  (h_sum : a 4 + a 6 = 5) :
  a 5 / a 7 = 3/2 := by
sorry

end geometric_sequence_ratio_l1138_113859


namespace mushroom_collection_problem_l1138_113864

/-- Represents the mushroom distribution pattern for a given girl --/
def mushroom_distribution (total : ℕ) (girl_number : ℕ) : ℚ :=
  (girl_number + 19) + 0.04 * (total - (girl_number + 19))

/-- Theorem stating the solution to the mushroom collection problem --/
theorem mushroom_collection_problem :
  ∃ (n : ℕ) (total : ℕ), 
    (∀ i j, i ≤ n → j ≤ n → mushroom_distribution total i = mushroom_distribution total j) ∧
    n = 5 ∧
    total = 120 := by
  sorry

end mushroom_collection_problem_l1138_113864


namespace sculpture_area_is_62_l1138_113833

/-- Represents a cube with a given edge length -/
structure Cube where
  edge : ℝ

/-- Represents a layer of the sculpture -/
structure Layer where
  cubes : ℕ
  exposed_top : ℕ
  exposed_sides : ℕ

/-- Represents the entire sculpture -/
structure Sculpture where
  bottom : Layer
  middle : Layer
  top : Layer

/-- Calculates the exposed surface area of a layer -/
def layer_area (c : Cube) (l : Layer) : ℝ :=
  (l.exposed_top + l.exposed_sides) * c.edge ^ 2

/-- Calculates the total exposed surface area of the sculpture -/
def total_area (c : Cube) (s : Sculpture) : ℝ :=
  layer_area c s.bottom + layer_area c s.middle + layer_area c s.top

/-- The sculpture described in the problem -/
def problem_sculpture : Sculpture :=
  { bottom := { cubes := 12, exposed_top := 12, exposed_sides := 24 },
    middle := { cubes := 6, exposed_top := 6, exposed_sides := 10 },
    top := { cubes := 2, exposed_top := 2, exposed_sides := 8 } }

/-- The cube used in the sculpture -/
def unit_cube : Cube :=
  { edge := 1 }

theorem sculpture_area_is_62 :
  total_area unit_cube problem_sculpture = 62 := by
  sorry

end sculpture_area_is_62_l1138_113833


namespace no_solution_implies_m_equals_two_l1138_113846

theorem no_solution_implies_m_equals_two (m : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (2 * x) / (x - 1) - 1 ≠ m / (x - 1)) → m = 2 :=
by sorry

end no_solution_implies_m_equals_two_l1138_113846


namespace isosceles_triangle_perimeter_l1138_113842

-- Define an isosceles triangle with sides a, b, and c
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b ∧ c ≠ a) ∨ (b = c ∧ a ≠ b) ∨ (a = c ∧ b ≠ a)
  validTriangle : a + b > c ∧ b + c > a ∧ a + c > b

-- Define the perimeter of a triangle
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle,
    (t.a = 4 ∧ t.b = 7) ∨ (t.a = 7 ∧ t.b = 4) →
    perimeter t = 15 ∨ perimeter t = 18 :=
by sorry

end isosceles_triangle_perimeter_l1138_113842


namespace box_width_is_48_l1138_113893

/-- Represents the dimensions of a box and the number of cubes that fill it -/
structure BoxWithCubes where
  length : ℕ
  width : ℕ
  depth : ℕ
  num_cubes : ℕ

/-- The box is completely filled by the cubes -/
def is_filled (box : BoxWithCubes) : Prop :=
  ∃ (cube_side : ℕ), 
    cube_side > 0 ∧
    box.length % cube_side = 0 ∧
    box.width % cube_side = 0 ∧
    box.depth % cube_side = 0 ∧
    box.length * box.width * box.depth = box.num_cubes * (cube_side ^ 3)

/-- The main theorem: if a box with given dimensions is filled with 80 cubes, its width is 48 inches -/
theorem box_width_is_48 (box : BoxWithCubes) : 
  box.length = 30 → box.depth = 12 → box.num_cubes = 80 → is_filled box → box.width = 48 := by
  sorry

end box_width_is_48_l1138_113893


namespace novel_writing_speed_l1138_113858

/-- Calculates the average writing speed given the total number of words and hours spent writing. -/
def average_writing_speed (total_words : ℕ) (total_hours : ℕ) : ℚ :=
  total_words / total_hours

/-- Theorem stating that for a novel with 60,000 words completed in 120 hours, 
    the average writing speed is 500 words per hour. -/
theorem novel_writing_speed :
  average_writing_speed 60000 120 = 500 := by
  sorry

end novel_writing_speed_l1138_113858


namespace money_division_l1138_113871

/-- Given a sum of money divided among three people a, b, and c, with the following conditions:
  1. a gets one-third of what b and c together get
  2. b gets two-sevenths of what a and c together get
  3. a receives $20 more than b
  Prove that the total amount shared is $720 -/
theorem money_division (a b c : ℚ) : 
  a = (1/3) * (b + c) →
  b = (2/7) * (a + c) →
  a = b + 20 →
  a + b + c = 720 := by
  sorry


end money_division_l1138_113871


namespace sisters_age_difference_l1138_113821

/-- The age difference between two sisters, Denise and Diane -/
def ageDifference (deniseFutureAge deniseFutureYears dianeFutureAge dianeFutureYears : ℕ) : ℕ :=
  (deniseFutureAge - deniseFutureYears) - (dianeFutureAge - dianeFutureYears)

/-- Theorem stating that the age difference between Denise and Diane is 4 years -/
theorem sisters_age_difference :
  ageDifference 25 2 25 6 = 4 := by
  sorry

end sisters_age_difference_l1138_113821


namespace magnitude_of_c_l1138_113832

/-- Given vectors a and b, if there exists a vector c satisfying certain conditions, then the magnitude of c is 2√5. -/
theorem magnitude_of_c (a b c : ℝ × ℝ) : 
  a = (-1, 2) →
  b = (3, -6) →
  (c.1 * b.1 + c.2 * b.2) / (Real.sqrt (c.1^2 + c.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -1/2 →
  c.1 * (-1) + c.2 * 8 = 5 →
  Real.sqrt (c.1^2 + c.2^2) = 2 * Real.sqrt 5 :=
by sorry

end magnitude_of_c_l1138_113832


namespace solve_system_l1138_113862

theorem solve_system (x y : ℤ) 
  (h1 : x + y = 270) 
  (h2 : x - y = 200) : 
  y = 35 := by
sorry

end solve_system_l1138_113862


namespace modulo_17_residue_l1138_113878

theorem modulo_17_residue : (342 + 6 * 47 + 8 * 157 + 3^3 * 21) % 17 = 10 := by
  sorry

end modulo_17_residue_l1138_113878


namespace thirteenth_number_l1138_113883

theorem thirteenth_number (results : Vector ℝ 25) 
  (h1 : results.toList.sum / 25 = 18)
  (h2 : (results.take 12).toList.sum / 12 = 14)
  (h3 : (results.drop 13).toList.sum / 12 = 17) :
  results[12] = 78 := by
  sorry

end thirteenth_number_l1138_113883


namespace total_earnings_proof_l1138_113861

structure LaundryShop where
  regular_rate : ℝ
  delicate_rate : ℝ
  bulky_rate : ℝ

def day_earnings (shop : LaundryShop) (regular_kilos delicate_kilos : ℝ) (bulky_items : ℕ) (delicate_discount : ℝ := 0) : ℝ :=
  shop.regular_rate * regular_kilos +
  shop.delicate_rate * delicate_kilos * (1 - delicate_discount) +
  shop.bulky_rate * (bulky_items : ℝ)

theorem total_earnings_proof (shop : LaundryShop)
  (h1 : shop.regular_rate = 3)
  (h2 : shop.delicate_rate = 4)
  (h3 : shop.bulky_rate = 5)
  (h4 : day_earnings shop 7 4 2 = 47)
  (h5 : day_earnings shop 10 6 3 = 69)
  (h6 : day_earnings shop 20 4 0 0.2 = 72.8) :
  day_earnings shop 7 4 2 + day_earnings shop 10 6 3 + day_earnings shop 20 4 0 0.2 = 188.8 := by
  sorry

#eval day_earnings ⟨3, 4, 5⟩ 7 4 2 + day_earnings ⟨3, 4, 5⟩ 10 6 3 + day_earnings ⟨3, 4, 5⟩ 20 4 0 0.2

end total_earnings_proof_l1138_113861


namespace population_reaches_limit_l1138_113811

/-- The number of years it takes for the population to reach or exceed the sustainable limit -/
def years_to_sustainable_limit : ℕ :=
  -- We'll define this later in the proof
  210

/-- The amount of acres required per person for sustainable living -/
def acres_per_person : ℕ := 2

/-- The total available acres for human habitation -/
def total_acres : ℕ := 35000

/-- The initial population in 2005 -/
def initial_population : ℕ := 150

/-- The number of years it takes for the population to double -/
def years_to_double : ℕ := 30

/-- The maximum sustainable population -/
def max_sustainable_population : ℕ := total_acres / acres_per_person

/-- The population after a given number of years -/
def population_after_years (years : ℕ) : ℕ :=
  initial_population * (2 ^ (years / years_to_double))

/-- Theorem stating that the population reaches or exceeds the sustainable limit in the specified number of years -/
theorem population_reaches_limit :
  population_after_years years_to_sustainable_limit ≥ max_sustainable_population ∧
  population_after_years (years_to_sustainable_limit - years_to_double) < max_sustainable_population :=
by sorry

end population_reaches_limit_l1138_113811


namespace dinner_set_cost_calculation_l1138_113881

/-- The cost calculation for John's dinner set purchase --/
theorem dinner_set_cost_calculation :
  let fork_cost : ℚ := 25
  let knife_cost : ℚ := 30
  let spoon_cost : ℚ := 20
  let silverware_cost : ℚ := fork_cost + knife_cost + spoon_cost
  let plate_cost : ℚ := silverware_cost * (1/2)
  let total_cost : ℚ := silverware_cost + plate_cost
  let discount_rate : ℚ := 1/10
  let final_cost : ℚ := total_cost * (1 - discount_rate)
  final_cost = 101.25 := by
  sorry

end dinner_set_cost_calculation_l1138_113881


namespace no_right_triangle_with_perimeter_5_times_inradius_l1138_113838

theorem no_right_triangle_with_perimeter_5_times_inradius :
  ¬∃ (a b c : ℕ+), 
    (a.val^2 + b.val^2 = c.val^2) ∧  -- right triangle condition
    ((a.val + b.val + c.val : ℚ) = 5 * (a.val * b.val : ℚ) / (a.val + b.val + c.val : ℚ)) 
    -- perimeter = 5 * in-radius condition
  := by sorry

end no_right_triangle_with_perimeter_5_times_inradius_l1138_113838


namespace min_dials_for_lighting_l1138_113835

/-- Represents a stack of 12-sided dials -/
def DialStack := ℕ → Fin 12 → Fin 12

/-- The sum of numbers in a column of the dial stack -/
def columnSum (stack : DialStack) (column : Fin 12) : ℕ :=
  sorry

/-- Predicate that checks if all column sums have the same remainder mod 12 -/
def allColumnSumsEqualMod12 (stack : DialStack) : Prop :=
  ∀ i j : Fin 12, columnSum stack i % 12 = columnSum stack j % 12

/-- The minimum number of dials required for the Christmas tree to light up -/
theorem min_dials_for_lighting : 
  ∃ (n : ℕ), n = 12 ∧ 
  (∃ (stack : DialStack), (∀ i : ℕ, i < n → ∃ (dial : Fin 12 → Fin 12), stack i = dial) ∧ 
   allColumnSumsEqualMod12 stack) ∧
  (∀ (m : ℕ), m < n → 
   ∀ (stack : DialStack), (∀ i : ℕ, i < m → ∃ (dial : Fin 12 → Fin 12), stack i = dial) → 
   ¬allColumnSumsEqualMod12 stack) :=
sorry

end min_dials_for_lighting_l1138_113835


namespace bedroom_painting_area_l1138_113870

/-- The total area of walls to be painted in multiple bedrooms -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - unpaintable_area
  num_bedrooms * paintable_area

/-- Theorem: The total area of walls to be painted in 4 bedrooms is 1520 square feet -/
theorem bedroom_painting_area : 
  total_paintable_area 4 14 11 9 70 = 1520 := by
  sorry


end bedroom_painting_area_l1138_113870


namespace last_two_digits_sum_sum_of_last_two_digits_main_result_l1138_113824

theorem last_two_digits_sum (n : ℕ) : ∃ (k : ℕ), 11^2004 - 5 = k * 100 + 36 :=
by sorry

theorem sum_of_last_two_digits : (11^2004 - 5) % 100 = 36 :=
by sorry

theorem main_result : (((11^2004 - 5) / 10) % 10) + ((11^2004 - 5) % 10) = 9 :=
by sorry

end last_two_digits_sum_sum_of_last_two_digits_main_result_l1138_113824


namespace unique_root_range_l1138_113844

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then a * Real.exp x else -Real.log x

theorem unique_root_range (a : ℝ) :
  (∃! x, f a (f a x) = 0) → a ∈ Set.Ioi 0 ∪ Set.Iio 1 := by sorry

end unique_root_range_l1138_113844


namespace madeline_homework_hours_l1138_113899

/-- Calculates the number of hours Madeline spends on homework per day -/
theorem madeline_homework_hours (class_hours_per_week : ℕ) 
                                 (sleep_hours_per_day : ℕ) 
                                 (work_hours_per_week : ℕ) 
                                 (leftover_hours : ℕ) 
                                 (days_per_week : ℕ) 
                                 (hours_per_day : ℕ) :
  class_hours_per_week = 18 →
  sleep_hours_per_day = 8 →
  work_hours_per_week = 20 →
  leftover_hours = 46 →
  days_per_week = 7 →
  hours_per_day = 24 →
  (hours_per_day * days_per_week - 
   (class_hours_per_week + sleep_hours_per_day * days_per_week + 
    work_hours_per_week + leftover_hours)) / days_per_week = 4 := by
  sorry

end madeline_homework_hours_l1138_113899


namespace olivia_wallet_problem_l1138_113872

theorem olivia_wallet_problem (initial_amount spent_amount : ℕ) 
  (h1 : initial_amount = 128)
  (h2 : spent_amount = 38) :
  initial_amount - spent_amount = 90 := by sorry

end olivia_wallet_problem_l1138_113872


namespace binomial_expansion_102_l1138_113845

theorem binomial_expansion_102 : 
  102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 104040401 := by
  sorry

end binomial_expansion_102_l1138_113845


namespace smallest_distance_between_complex_numbers_l1138_113853

theorem smallest_distance_between_complex_numbers (z w : ℂ) 
  (hz : Complex.abs (z + 2 + 4*I) = 2)
  (hw : Complex.abs (w - 6 - 7*I) = 4) :
  ∃ (min_dist : ℝ), 
    (∀ (z' w' : ℂ), Complex.abs (z' + 2 + 4*I) = 2 → Complex.abs (w' - 6 - 7*I) = 4 
      → Complex.abs (z' - w') ≥ min_dist) ∧ 
    (∃ (z₀ w₀ : ℂ), Complex.abs (z₀ + 2 + 4*I) = 2 ∧ Complex.abs (w₀ - 6 - 7*I) = 4 
      ∧ Complex.abs (z₀ - w₀) = min_dist) ∧
    min_dist = Real.sqrt 185 - 6 :=
by sorry

end smallest_distance_between_complex_numbers_l1138_113853


namespace line_point_k_value_l1138_113892

/-- Given a line containing points (7, 10), (1, k), and (-5, 3), prove that k = 6.5 -/
theorem line_point_k_value : ∀ k : ℝ,
  (∃ (line : Set (ℝ × ℝ)),
    (7, 10) ∈ line ∧ (1, k) ∈ line ∧ (-5, 3) ∈ line ∧
    (∀ p q r : ℝ × ℝ, p ∈ line → q ∈ line → r ∈ line →
      (p.2 - q.2) * (q.1 - r.1) = (q.2 - r.2) * (p.1 - q.1))) →
  k = 13/2 :=
by sorry

end line_point_k_value_l1138_113892


namespace sqrt_six_minus_one_over_two_lt_one_l1138_113889

theorem sqrt_six_minus_one_over_two_lt_one : (Real.sqrt 6 - 1) / 2 < 1 := by
  sorry

end sqrt_six_minus_one_over_two_lt_one_l1138_113889


namespace zero_is_monomial_l1138_113849

/-- A monomial is a polynomial with a single term. -/
def IsMonomial (p : Polynomial ℝ) : Prop :=
  ∃ c a, p = c * Polynomial.X ^ a

/-- Zero is a monomial. -/
theorem zero_is_monomial : IsMonomial (0 : Polynomial ℝ) := by
  sorry

end zero_is_monomial_l1138_113849


namespace power_sixteen_seven_fourths_l1138_113895

theorem power_sixteen_seven_fourths : (16 : ℝ) ^ (7/4) = 128 := by
  sorry

end power_sixteen_seven_fourths_l1138_113895


namespace cube_root_equality_l1138_113808

theorem cube_root_equality (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (m^4 * n^4)^(1/3) = (m * n)^(4/3) := by sorry

end cube_root_equality_l1138_113808


namespace fraction_sum_equality_l1138_113828

theorem fraction_sum_equality : (20 : ℚ) / 50 - 3 / 8 + 1 / 4 = 11 / 40 := by
  sorry

end fraction_sum_equality_l1138_113828


namespace semicircle_perimeter_l1138_113884

/-- The perimeter of a semicircle with radius r is πr + 2r -/
theorem semicircle_perimeter (r : ℝ) (h : r > 0) : 
  let P := r * Real.pi + 2 * r
  P = r * Real.pi + 2 * r :=
by sorry

#check semicircle_perimeter

end semicircle_perimeter_l1138_113884


namespace zoe_total_earnings_l1138_113809

/-- Represents Zoe's babysitting and pool cleaning earnings -/
structure ZoeEarnings where
  zachary_sessions : ℕ
  julie_sessions : ℕ
  chloe_sessions : ℕ
  zachary_earnings : ℕ
  pool_cleaning_earnings : ℕ

/-- Calculates Zoe's total earnings -/
def total_earnings (e : ZoeEarnings) : ℕ :=
  e.zachary_earnings + e.pool_cleaning_earnings

/-- Theorem stating that Zoe's total earnings are $3200 -/
theorem zoe_total_earnings (e : ZoeEarnings) 
  (h1 : e.julie_sessions = 3 * e.zachary_sessions)
  (h2 : e.zachary_sessions = e.chloe_sessions / 5)
  (h3 : e.zachary_earnings = 600)
  (h4 : e.pool_cleaning_earnings = 2600) : 
  total_earnings e = 3200 := by
  sorry


end zoe_total_earnings_l1138_113809


namespace quadratic_prime_values_l1138_113856

theorem quadratic_prime_values (p : ℕ) (hp : p > 1) :
  ∀ x : ℕ, 0 ≤ x ∧ x < p →
    (Nat.Prime (x^2 - x + p) ↔ (x = 0 ∨ x = 1)) :=
by sorry

end quadratic_prime_values_l1138_113856


namespace binary_101011_equals_43_l1138_113802

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101011_equals_43 :
  binary_to_decimal [true, true, false, true, false, true] = 43 := by
  sorry

end binary_101011_equals_43_l1138_113802


namespace team_selection_with_girl_l1138_113805

theorem team_selection_with_girl (n m k : ℕ) (hn : n = 5) (hm : m = 5) (hk : k = 3) :
  Nat.choose (n + m) k - Nat.choose n k = 110 := by
  sorry

end team_selection_with_girl_l1138_113805


namespace intersection_of_A_and_B_l1138_113873

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define set A
def A : Set ℂ := {i, i^2, i^3, i^4}

-- Define set B
def B : Set ℂ := {1, -1}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {1, -1} := by
  sorry

end intersection_of_A_and_B_l1138_113873


namespace isosceles_triangle_area_l1138_113868

/-- 
Given an isosceles triangle with height H that is twice as long as 
its projection on the lateral side, prove that its area is H^2 * √3.
-/
theorem isosceles_triangle_area (H : ℝ) (h : H > 0) : 
  let projection := H / 2
  let base := 2 * H * Real.sqrt 3
  let area := (1 / 2) * base * H
  area = H^2 * Real.sqrt 3 := by
sorry

end isosceles_triangle_area_l1138_113868


namespace cosine_of_angle_through_point_l1138_113869

/-- If the terminal side of angle α passes through point P (-1, -√2), then cos α = -√3/3 -/
theorem cosine_of_angle_through_point :
  ∀ α : Real,
  let P : Real × Real := (-1, -Real.sqrt 2)
  (∃ t : Real, t > 0 ∧ P = (t * Real.cos α, t * Real.sin α)) →
  Real.cos α = -Real.sqrt 3 / 3 := by
sorry

end cosine_of_angle_through_point_l1138_113869


namespace turtle_contradiction_l1138_113826

/-- Represents the position of a turtle in the line -/
inductive Position
  | Front
  | Middle
  | Back

/-- Represents a turtle with its position and statements about other turtles -/
structure Turtle where
  position : Position
  turtles_behind : Nat
  turtles_in_front : Nat

/-- The scenario of three turtles in a line -/
def turtle_scenario : List Turtle :=
  [ { position := Position.Front
    , turtles_behind := 2
    , turtles_in_front := 0 }
  , { position := Position.Middle
    , turtles_behind := 1
    , turtles_in_front := 1 }
  , { position := Position.Back
    , turtles_behind := 1
    , turtles_in_front := 1 } ]

/-- Theorem stating that the turtle scenario leads to a contradiction -/
theorem turtle_contradiction : 
  ∀ (t : Turtle), t ∈ turtle_scenario → 
    (t.position = Position.Front → t.turtles_behind = 2) ∧
    (t.position = Position.Middle → t.turtles_behind = 1 ∧ t.turtles_in_front = 1) ∧
    (t.position = Position.Back → t.turtles_behind = 1 ∧ t.turtles_in_front = 1) →
    False := by
  sorry

end turtle_contradiction_l1138_113826


namespace hannah_adblock_efficiency_l1138_113812

/-- The percentage of ads not blocked by Hannah's AdBlock -/
def ads_not_blocked : ℝ := sorry

/-- The percentage of not blocked ads that are interesting -/
def interesting_not_blocked_ratio : ℝ := 0.20

/-- The percentage of all ads that are not interesting and not blocked -/
def not_interesting_not_blocked_ratio : ℝ := 0.16

theorem hannah_adblock_efficiency :
  ads_not_blocked = 0.20 :=
sorry

end hannah_adblock_efficiency_l1138_113812


namespace schoolClubProfit_l1138_113847

/-- Represents the candy bar sale scenario for a school club -/
structure CandyBarSale where
  totalBars : ℕ
  purchaseRate : ℚ
  regularSellRate : ℚ
  bulkSellRate : ℚ

/-- Calculates the profit for the candy bar sale -/
def calculateProfit (sale : CandyBarSale) : ℚ :=
  let costPerBar := sale.purchaseRate / 4
  let totalCost := costPerBar * sale.totalBars
  let revenuePerBar := sale.regularSellRate / 3
  let totalRevenue := revenuePerBar * sale.totalBars
  totalRevenue - totalCost

/-- The given candy bar sale scenario -/
def schoolClubSale : CandyBarSale :=
  { totalBars := 1200
  , purchaseRate := 3
  , regularSellRate := 2
  , bulkSellRate := 3/5 }

/-- Theorem stating that the profit for the school club is -100 dollars -/
theorem schoolClubProfit :
  calculateProfit schoolClubSale = -100 := by
  sorry


end schoolClubProfit_l1138_113847


namespace teaspoons_per_tablespoon_l1138_113885

/-- Given the following definitions:
  * One cup contains 480 grains of rice
  * Half a cup is 8 tablespoons
  * One teaspoon contains 10 grains of rice
  Prove that there are 3 teaspoons in one tablespoon -/
theorem teaspoons_per_tablespoon 
  (grains_per_cup : ℕ) 
  (tablespoons_per_half_cup : ℕ) 
  (grains_per_teaspoon : ℕ) 
  (h1 : grains_per_cup = 480)
  (h2 : tablespoons_per_half_cup = 8)
  (h3 : grains_per_teaspoon = 10) : 
  (grains_per_cup / 2) / tablespoons_per_half_cup / grains_per_teaspoon = 3 :=
by sorry

end teaspoons_per_tablespoon_l1138_113885


namespace value_of_x_l1138_113874

theorem value_of_x (x y z : ℚ) : x = y / 3 → y = z / 4 → z = 48 → x = 4 := by
  sorry

end value_of_x_l1138_113874


namespace roots_of_g_are_cubes_of_roots_of_f_l1138_113804

/-- The original polynomial f(x) -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 1

/-- The polynomial g(x) whose roots are the cubes of the roots of f(x) -/
def g (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

/-- Theorem stating that the roots of g are the cubes of the roots of f -/
theorem roots_of_g_are_cubes_of_roots_of_f :
  ∀ r : ℝ, f r = 0 → g (r^3) = 0 := by sorry

end roots_of_g_are_cubes_of_roots_of_f_l1138_113804


namespace remainder_divisibility_l1138_113860

theorem remainder_divisibility (N : ℤ) : 
  ∃ k : ℤ, N = 142 * k + 110 → ∃ m : ℤ, N = 14 * m + 8 := by
  sorry

end remainder_divisibility_l1138_113860


namespace solution_is_twelve_l1138_113817

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 4 * a - 2 * b

/-- Theorem stating that 12 is the solution to the equation -/
theorem solution_is_twelve :
  ∃ (x : ℝ), custom_mul 3 (custom_mul 6 x) = 12 ∧ x = 12 := by
  sorry

end solution_is_twelve_l1138_113817


namespace one_fourth_of_8_4_l1138_113867

theorem one_fourth_of_8_4 : (8.4 : ℚ) / 4 = 21 / 10 := by
  sorry

end one_fourth_of_8_4_l1138_113867


namespace least_cans_for_given_volumes_l1138_113863

/-- The least number of cans required to pack drinks -/
def leastCans (maaza pepsi sprite : ℕ) : ℕ :=
  let gcd := Nat.gcd (Nat.gcd maaza pepsi) sprite
  (maaza / gcd) + (pepsi / gcd) + (sprite / gcd)

/-- Theorem stating the least number of cans required for given volumes -/
theorem least_cans_for_given_volumes :
  leastCans 40 144 368 = 69 := by
  sorry

end least_cans_for_given_volumes_l1138_113863


namespace square_sum_from_system_l1138_113823

theorem square_sum_from_system (x y : ℝ) 
  (h1 : x * y = 6)
  (h2 : x^2 * y + x * y^2 + x + y = 63) : 
  x^2 + y^2 = 69 := by
sorry

end square_sum_from_system_l1138_113823


namespace smallest_k_for_zero_diff_l1138_113834

def u (n : ℕ) : ℕ := n^3 + 2*n^2 + n

def diff (f : ℕ → ℕ) (n : ℕ) : ℕ := f (n + 1) - f n

def diff_k (f : ℕ → ℕ) : ℕ → (ℕ → ℕ)
  | 0 => f
  | k + 1 => diff (diff_k f k)

theorem smallest_k_for_zero_diff (n : ℕ) : 
  (∀ n, diff_k u 4 n = 0) ∧ 
  (∀ k < 4, ∃ n, diff_k u k n ≠ 0) :=
sorry

end smallest_k_for_zero_diff_l1138_113834


namespace quadratic_expression_sum_l1138_113843

theorem quadratic_expression_sum (d : ℝ) (h : d ≠ 0) :
  ∃ (a b c : ℤ), (15 * d + 16 + 17 * d^2) + (3 * d + 2) = a * d^2 + b * d + c ∧ a + b + c = 53 := by
  sorry

end quadratic_expression_sum_l1138_113843


namespace sum_of_squares_zero_implies_sum_l1138_113818

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 6)^2 + (y - 7)^2 + (z - 8)^2 = 0 → x + y + z = 21 := by
  sorry

end sum_of_squares_zero_implies_sum_l1138_113818


namespace smallest_slope_tangent_line_l1138_113810

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

/-- Theorem: The equation of the tangent line with the smallest slope -/
theorem smallest_slope_tangent_line :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, y = f x → (∀ m : ℝ, ∃ x₀ y₀ : ℝ, y₀ = f x₀ ∧ m = f' x₀ → m ≥ f' (-1)) ∧ 
    a*x + b*y + c = 0 ∧ 
    -14 = f (-1) ∧ 
    3 = f' (-1) ∧
    a = 3 ∧ b = -1 ∧ c = -11) :=
sorry

end smallest_slope_tangent_line_l1138_113810


namespace isosceles_right_triangle_probability_l1138_113840

/-- Represents an isosceles right-angled triangle -/
structure IsoscelesRightTriangle where
  leg_length : ℝ

/-- Represents a point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The probability of choosing a point within distance 1 from the right angle -/
def probability_within_distance (t : IsoscelesRightTriangle) : ℝ :=
  sorry

theorem isosceles_right_triangle_probability 
  (t : IsoscelesRightTriangle) 
  (h : t.leg_length = 2) : 
  probability_within_distance t = π / 8 := by
  sorry

end isosceles_right_triangle_probability_l1138_113840


namespace unique_value_at_half_l1138_113850

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x * y + f x) = 2 * x * f y + f x

theorem unique_value_at_half (f : ℝ → ℝ) (hf : special_function f) :
  ∃! v : ℝ, f (1/2) = v ∧ v = -1 :=
sorry

end unique_value_at_half_l1138_113850


namespace max_digits_product_5_4_l1138_113854

theorem max_digits_product_5_4 : 
  ∃ (a b : ℕ), 
    (10000 ≤ a ∧ a < 100000) ∧ 
    (1000 ≤ b ∧ b < 10000) ∧ 
    (∀ x y : ℕ, (10000 ≤ x ∧ x < 100000) → (1000 ≤ y ∧ y < 10000) → x * y ≤ a * b) ∧
    (Nat.digits 10 (a * b)).length = 10 :=
by sorry

end max_digits_product_5_4_l1138_113854


namespace x_squared_plus_3x_plus_4_range_l1138_113836

theorem x_squared_plus_3x_plus_4_range :
  ∀ x : ℝ, x^2 - 8*x + 15 < 0 → 22 < x^2 + 3*x + 4 ∧ x^2 + 3*x + 4 < 44 := by
  sorry

end x_squared_plus_3x_plus_4_range_l1138_113836


namespace no_solutions_exist_l1138_113886

theorem no_solutions_exist : ¬∃ (a b : ℕ), 2019 * a^2018 = 2017 + b^2016 := by
  sorry

end no_solutions_exist_l1138_113886


namespace fish_pond_population_l1138_113855

/-- The number of fish initially tagged and returned to the pond -/
def initial_tagged : ℕ := 80

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 80

/-- The number of tagged fish found in the second catch -/
def tagged_in_second : ℕ := 2

/-- The approximate total number of fish in the pond -/
def total_fish : ℕ := 3200

/-- Theorem stating that the given conditions lead to the approximate number of fish in the pond -/
theorem fish_pond_population :
  (initial_tagged : ℚ) / total_fish = (tagged_in_second : ℚ) / second_catch :=
by sorry

end fish_pond_population_l1138_113855


namespace hungry_bear_purchase_cost_l1138_113880

/-- Represents the cost of items at Hungry Bear Diner -/
structure DinerCost where
  sandwich_price : ℕ
  soda_price : ℕ
  cookie_price : ℕ

/-- Calculates the total cost of a purchase at Hungry Bear Diner -/
def total_cost (prices : DinerCost) (num_sandwiches num_sodas num_cookies : ℕ) : ℕ :=
  prices.sandwich_price * num_sandwiches +
  prices.soda_price * num_sodas +
  prices.cookie_price * num_cookies

/-- Theorem stating that the total cost of 3 sandwiches, 5 sodas, and 4 cookies is $35 -/
theorem hungry_bear_purchase_cost :
  ∃ (prices : DinerCost),
    prices.sandwich_price = 4 ∧
    prices.soda_price = 3 ∧
    prices.cookie_price = 2 ∧
    total_cost prices 3 5 4 = 35 := by
  sorry

end hungry_bear_purchase_cost_l1138_113880


namespace like_terms_mn_value_l1138_113800

/-- 
Given two algebraic terms are like terms, prove that m^n = 8.
-/
theorem like_terms_mn_value (n m : ℕ) : 
  (∃ (k : ℚ), k * X^n * Y^2 = X^3 * Y^m) → m^n = 8 := by
  sorry

end like_terms_mn_value_l1138_113800


namespace sequence_on_line_geometric_l1138_113890

/-- Given a sequence {a_n} where (n, a_n) is on the line 2x - y + 1 = 0,
    if a_1, a_4, and a_m form a geometric sequence, then m = 13. -/
theorem sequence_on_line_geometric (a : ℕ → ℝ) :
  (∀ n, 2 * n - a n + 1 = 0) →
  (∃ r, a 4 = a 1 * r ∧ a m = a 4 * r) →
  m = 13 :=
by sorry

end sequence_on_line_geometric_l1138_113890


namespace share_difference_l1138_113801

/-- Given a distribution of money among three people with a specific ratio and one known share,
    calculate the difference between the largest and smallest shares. -/
theorem share_difference (total_parts ratio_faruk ratio_vasim ratio_ranjith vasim_share : ℕ) 
    (h1 : total_parts = ratio_faruk + ratio_vasim + ratio_ranjith)
    (h2 : ratio_faruk = 3)
    (h3 : ratio_vasim = 5)
    (h4 : ratio_ranjith = 11)
    (h5 : vasim_share = 1500) :
    ratio_ranjith * (vasim_share / ratio_vasim) - ratio_faruk * (vasim_share / ratio_vasim) = 2400 := by
  sorry

end share_difference_l1138_113801


namespace no_integer_solution_l1138_113814

theorem no_integer_solution :
  ¬ ∃ (A B C : ℤ),
    (A - B = 1620) ∧
    ((75 : ℚ) / 1000 * A = (125 : ℚ) / 1000 * B) ∧
    (A + B = (1 : ℚ) / 2 * C^4) ∧
    (A^2 + B^2 = C^2) := by
  sorry

end no_integer_solution_l1138_113814


namespace sum_of_squares_zero_implies_sum_l1138_113866

theorem sum_of_squares_zero_implies_sum (a b c : ℝ) :
  (a - 5)^2 + (b - 6)^2 + (c - 7)^2 = 0 → a + b + c = 18 := by
  sorry

end sum_of_squares_zero_implies_sum_l1138_113866


namespace root_difference_indeterminate_l1138_113887

/-- A function with the property f(1 + x) = f(1 - x) for all real x -/
def symmetric_around_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 + x) = f (1 - x)

/-- A function has exactly two distinct real roots -/
def has_two_distinct_roots (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧
  ∀ z : ℝ, f z = 0 → z = x ∨ z = y

theorem root_difference_indeterminate (f : ℝ → ℝ) 
  (h1 : symmetric_around_one f) 
  (h2 : has_two_distinct_roots f) : 
  ¬ ∃ d : ℝ, ∀ x y : ℝ, f x = 0 → f y = 0 → x ≠ y → |x - y| = d :=
sorry

end root_difference_indeterminate_l1138_113887


namespace mortdecai_donation_l1138_113898

/-- Represents the number of eggs in a dozen --/
def eggsPerDozen : ℕ := 12

/-- Represents the number of days Mortdecai collects eggs in a week --/
def collectionDays : ℕ := 2

/-- Represents the number of dozens of eggs Mortdecai collects each collection day --/
def collectedDozens : ℕ := 8

/-- Represents the number of dozens of eggs Mortdecai delivers to the market --/
def marketDeliveryDozens : ℕ := 3

/-- Represents the number of dozens of eggs Mortdecai delivers to the mall --/
def mallDeliveryDozens : ℕ := 5

/-- Represents the number of dozens of eggs Mortdecai uses for pie --/
def pieDozens : ℕ := 4

/-- Calculates the number of eggs Mortdecai donates to charity --/
def donatedEggs : ℕ :=
  (collectedDozens * collectionDays - marketDeliveryDozens - mallDeliveryDozens - pieDozens) * eggsPerDozen

theorem mortdecai_donation :
  donatedEggs = 48 := by
  sorry

end mortdecai_donation_l1138_113898


namespace trapezoid_DG_length_l1138_113839

-- Define the trapezoids and their properties
structure Trapezoid where
  BC : ℝ
  AD : ℝ
  CT : ℝ
  TD : ℝ
  DG : ℝ

-- Define the theorem
theorem trapezoid_DG_length (ABCD AEFG : Trapezoid) : 
  ABCD.BC = 4 →
  ABCD.AD = 7 →
  ABCD.CT = 1 →
  ABCD.TD = 2 →
  -- ABCD and AEFG are right trapezoids with BC ∥ EF and CD ∥ FG (assumed)
  -- ABCD and AEFG have the same area (assumed)
  AEFG.DG = 9/4 := by
  sorry


end trapezoid_DG_length_l1138_113839


namespace equation_solutions_l1138_113816

variable (a : ℝ)

theorem equation_solutions :
  {x : ℝ | x * (x + a)^3 * (5 - x) = 0} = {0, -a, 5} := by
sorry

end equation_solutions_l1138_113816


namespace length_units_ordering_l1138_113891

-- Define an enumeration for length units
inductive LengthUnit
  | Kilometer
  | Meter
  | Centimeter
  | Millimeter

-- Define a function to compare two length units
def isLargerThan (a b : LengthUnit) : Prop :=
  match a, b with
  | LengthUnit.Kilometer, _ => a ≠ b
  | LengthUnit.Meter, LengthUnit.Centimeter => True
  | LengthUnit.Meter, LengthUnit.Millimeter => True
  | LengthUnit.Centimeter, LengthUnit.Millimeter => True
  | _, _ => False

-- Theorem to prove the correct ordering of length units
theorem length_units_ordering :
  isLargerThan LengthUnit.Kilometer LengthUnit.Meter ∧
  isLargerThan LengthUnit.Meter LengthUnit.Centimeter ∧
  isLargerThan LengthUnit.Centimeter LengthUnit.Millimeter :=
by sorry


end length_units_ordering_l1138_113891


namespace negative_quarter_and_negative_four_power_l1138_113888

theorem negative_quarter_and_negative_four_power :
  (-0.25)^11 * (-4)^12 = -4 := by
  sorry

end negative_quarter_and_negative_four_power_l1138_113888


namespace perimeter_ratio_from_area_ratio_l1138_113822

theorem perimeter_ratio_from_area_ratio (s1 s2 : ℝ) (h : s1 ^ 2 / s2 ^ 2 = 49 / 64) :
  (4 * s1) / (4 * s2) = 7 / 8 := by
  sorry

end perimeter_ratio_from_area_ratio_l1138_113822


namespace walters_age_calculation_l1138_113803

/-- Walter's age at the end of 2000 -/
def walters_age_2000 : ℝ := 37.5

/-- Walter's grandmother's age at the end of 2000 -/
def grandmothers_age_2000 : ℝ := 3 * walters_age_2000

/-- The sum of Walter's and his grandmother's birth years -/
def birth_years_sum : ℕ := 3850

/-- Walter's age at the end of 2010 -/
def walters_age_2010 : ℝ := walters_age_2000 + 10

theorem walters_age_calculation :
  (2000 - walters_age_2000) + (2000 - grandmothers_age_2000) = birth_years_sum ∧
  walters_age_2010 = 47.5 := by
  sorry

#eval walters_age_2010

end walters_age_calculation_l1138_113803


namespace line_equations_correct_l1138_113813

/-- Triangle ABC with vertices A(0,4), B(-2,6), and C(-8,0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Given triangle ABC, compute the equation of line AB -/
def lineAB (t : Triangle) : LineEquation :=
  { a := 1, b := 1, c := -4 }

/-- Given triangle ABC, compute the midpoint D of side AC -/
def midpointD (t : Triangle) : ℝ × ℝ :=
  ((-4 : ℝ), (2 : ℝ))

/-- Given triangle ABC, compute the equation of line BD where D is the midpoint of AC -/
def lineBD (t : Triangle) : LineEquation :=
  { a := 2, b := -1, c := 10 }

/-- Theorem stating that for the given triangle, the computed line equations are correct -/
theorem line_equations_correct (t : Triangle) 
    (h : t.A = (0, 4) ∧ t.B = (-2, 6) ∧ t.C = (-8, 0)) : 
  (lineAB t = { a := 1, b := 1, c := -4 }) ∧ 
  (lineBD t = { a := 2, b := -1, c := 10 }) := by
  sorry

end line_equations_correct_l1138_113813


namespace sin_negative_sixty_degrees_l1138_113879

theorem sin_negative_sixty_degrees : Real.sin (-(60 * π / 180)) = -Real.sqrt 3 / 2 := by
  sorry

end sin_negative_sixty_degrees_l1138_113879


namespace quadratic_equation_m_value_l1138_113894

theorem quadratic_equation_m_value (x₁ x₂ m : ℝ) : 
  (∀ x, x^2 - 4*x + m = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ + 3*x₂ = 5 →
  m = 7/4 := by
sorry

end quadratic_equation_m_value_l1138_113894


namespace rhombus_with_60_degree_angles_l1138_113819

/-- A configuration of four points in the plane -/
structure QuadConfig where
  A₁ : ℝ × ℝ
  A₂ : ℝ × ℝ
  A₃ : ℝ × ℝ
  A₄ : ℝ × ℝ

/-- The sum of the smallest angles in the four triangles formed by the points -/
def sumSmallestAngles (q : QuadConfig) : ℝ := sorry

/-- Predicate to check if four points form a rhombus -/
def isRhombus (q : QuadConfig) : Prop := sorry

/-- Predicate to check if all angles in a quadrilateral are at least 60° -/
def allAnglesAtLeast60 (q : QuadConfig) : Prop := sorry

/-- The main theorem -/
theorem rhombus_with_60_degree_angles 
  (q : QuadConfig) 
  (h : sumSmallestAngles q = π) : 
  isRhombus q ∧ allAnglesAtLeast60 q := by
  sorry

end rhombus_with_60_degree_angles_l1138_113819


namespace simplify_and_evaluate_l1138_113865

theorem simplify_and_evaluate (x y : ℚ) 
  (hx : x = 1/2) (hy : y = 1/3) : 
  (3*x - y)^2 - (3*x + 2*y)*(3*x - 2*y) = -4/9 := by
  sorry

end simplify_and_evaluate_l1138_113865


namespace hyperbola_eccentricity_l1138_113852

/-- Given a hyperbola with equation x²/a² - y²/3 = 1 (a > 0) that passes through (-2, 0),
    its eccentricity is √7/2 -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) :
  (4 / a^2 - 0 / 3 = 1) →
  let b := Real.sqrt 3
  let c := Real.sqrt (a^2 + b^2)
  c / a = Real.sqrt 7 / 2 := by
  sorry

end hyperbola_eccentricity_l1138_113852


namespace staircase_shape_perimeter_l1138_113806

/-- A shape formed by cutting out a staircase from a rectangle --/
structure StaircaseShape where
  width : ℝ
  height : ℝ
  step_size : ℝ
  num_steps : ℕ
  total_area : ℝ

/-- Calculate the perimeter of a StaircaseShape --/
def perimeter (shape : StaircaseShape) : ℝ :=
  shape.width + shape.height + shape.step_size * (2 * shape.num_steps)

/-- The main theorem --/
theorem staircase_shape_perimeter : 
  ∀ (shape : StaircaseShape), 
    shape.width = 11 ∧ 
    shape.step_size = 2 ∧ 
    shape.num_steps = 10 ∧ 
    shape.total_area = 130 →
    perimeter shape = 54.45 := by
  sorry


end staircase_shape_perimeter_l1138_113806


namespace perpendicular_line_x_intercept_l1138_113875

/-- The slope of the original line -/
def original_slope : ℚ := 3 / 2

/-- The slope of the perpendicular line -/
def perpendicular_slope : ℚ := -2 / 3

/-- The y-intercept of the perpendicular line -/
def y_intercept : ℚ := 5

/-- The x-intercept of the perpendicular line -/
def x_intercept : ℚ := 15 / 2

theorem perpendicular_line_x_intercept :
  let line := fun (x : ℚ) => perpendicular_slope * x + y_intercept
  (∀ x, line x = 0 ↔ x = x_intercept) ∧
  perpendicular_slope * original_slope = -1 :=
by sorry

end perpendicular_line_x_intercept_l1138_113875


namespace reflection_line_equation_l1138_113830

/-- The line of reflection for a triangle --/
structure ReflectionLine where
  equation : ℝ → Prop

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points --/
structure Triangle where
  p : Point
  q : Point
  r : Point

/-- Reflects a point about a horizontal line --/
def reflect (p : Point) (y : ℝ) : Point :=
  { x := p.x, y := 2 * y - p.y }

/-- The theorem stating the equation of the reflection line --/
theorem reflection_line_equation 
  (t : Triangle)
  (t' : Triangle)
  (h1 : t.p = Point.mk 1 4)
  (h2 : t.q = Point.mk 8 9)
  (h3 : t.r = Point.mk (-3) 7)
  (h4 : t'.p = Point.mk 1 (-6))
  (h5 : t'.q = Point.mk 8 (-11))
  (h6 : t'.r = Point.mk (-3) (-9))
  (h7 : ∃ (y : ℝ), t'.p = reflect t.p y ∧ 
                   t'.q = reflect t.q y ∧ 
                   t'.r = reflect t.r y) :
  ∃ (m : ReflectionLine), m.equation = λ y => y = -1 :=
sorry

end reflection_line_equation_l1138_113830
