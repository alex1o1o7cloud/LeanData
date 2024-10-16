import Mathlib

namespace NUMINAMATH_CALUDE_triangle_area_proof_l3439_343960

theorem triangle_area_proof (A B C : ℝ) (a b c : ℝ) : 
  C = π / 3 →
  c = Real.sqrt 7 →
  b = 3 * a →
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l3439_343960


namespace NUMINAMATH_CALUDE_dorothy_score_l3439_343939

theorem dorothy_score (tatuya ivanna dorothy : ℚ) 
  (h1 : tatuya = 2 * ivanna)
  (h2 : ivanna = (3/5) * dorothy)
  (h3 : (tatuya + ivanna + dorothy) / 3 = 84) :
  dorothy = 90 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_score_l3439_343939


namespace NUMINAMATH_CALUDE_inequality_proof_l3439_343952

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  2 * x + 1 / (x^2 - 2*x*y + y^2) ≥ 2 * y + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3439_343952


namespace NUMINAMATH_CALUDE_correct_operation_l3439_343972

theorem correct_operation (x y : ℝ) : 5 * x * y - 4 * x * y = x * y := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3439_343972


namespace NUMINAMATH_CALUDE_problem_statement_l3439_343973

theorem problem_statement (a : ℝ) (h : a = 2) : (4 * a^2 - 11 * a + 5) * (3 * a - 4) = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3439_343973


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3439_343933

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3439_343933


namespace NUMINAMATH_CALUDE_weight_of_b_l3439_343954

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 45) :
  b = 35 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l3439_343954


namespace NUMINAMATH_CALUDE_flash_catch_up_distance_l3439_343967

theorem flash_catch_up_distance 
  (v : ℝ) -- Ace's speed
  (z : ℝ) -- Flash's speed multiplier
  (k : ℝ) -- Ace's head start distance
  (t₀ : ℝ) -- Time Ace runs before Flash starts
  (h₁ : v > 0) -- Ace's speed is positive
  (h₂ : z > 1) -- Flash is faster than Ace
  (h₃ : k ≥ 0) -- Head start is non-negative
  (h₄ : t₀ ≥ 0) -- Time before Flash starts is non-negative
  : 
  ∃ (t : ℝ), t > 0 ∧ z * v * t = v * (t + t₀) + k ∧
  z * v * t = z * (t₀ * v + k) / (z - 1) :=
sorry

end NUMINAMATH_CALUDE_flash_catch_up_distance_l3439_343967


namespace NUMINAMATH_CALUDE_divisibility_by_1001_l3439_343919

theorem divisibility_by_1001 (n : ℤ) : n ≡ 300^3000 [ZMOD 1001] → n ≡ 1 [ZMOD 1001] := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_1001_l3439_343919


namespace NUMINAMATH_CALUDE_max_non_managers_l3439_343900

/-- Represents the number of managers in a department -/
def managers : ℕ := 11

/-- Represents the ratio of managers to non-managers -/
def ratio : ℚ := 7 / 37

/-- Theorem stating the maximum number of non-managers in a department -/
theorem max_non_managers :
  ∀ n : ℕ, (managers : ℚ) / n > ratio → n ≤ 58 :=
sorry

end NUMINAMATH_CALUDE_max_non_managers_l3439_343900


namespace NUMINAMATH_CALUDE_third_derivative_y_l3439_343996

noncomputable def y (x : ℝ) : ℝ := (Real.log (3 + x)) / (3 + x)

theorem third_derivative_y (x : ℝ) (h : x ≠ -3) : 
  (deriv^[3] y) x = (11 - 6 * Real.log (3 + x)) / (3 + x)^4 :=
by sorry

end NUMINAMATH_CALUDE_third_derivative_y_l3439_343996


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l3439_343905

def population_size : ℕ := 30
def sample_size : ℕ := 6

def systematic_sampling_interval (pop_size sample_size : ℕ) : ℕ :=
  pop_size / sample_size

def generate_sample (start interval : ℕ) (size : ℕ) : List ℕ :=
  List.range size |>.map (λ i => start + i * interval)

theorem correct_systematic_sample :
  let interval := systematic_sampling_interval population_size sample_size
  let sample := generate_sample 2 interval sample_size
  (interval = 5) ∧ (sample = [2, 7, 12, 17, 22, 27]) := by
  sorry

#eval systematic_sampling_interval population_size sample_size
#eval generate_sample 2 (systematic_sampling_interval population_size sample_size) sample_size

end NUMINAMATH_CALUDE_correct_systematic_sample_l3439_343905


namespace NUMINAMATH_CALUDE_max_students_distribution_l3439_343941

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 2010) (h2 : pencils = 1050) : 
  (∃ (notebooks : ℕ), notebooks ≥ 30 ∧ 
    (∃ (distribution : ℕ → ℕ × ℕ × ℕ), 
      (∀ i j, i ≠ j → (distribution i).2.2 ≠ (distribution j).2.2) ∧
      (∀ i, i < 30 → (distribution i).1 = pens / 30 ∧ (distribution i).2.1 = pencils / 30))) ∧
  (∀ n : ℕ, n > 30 → 
    ¬(∃ (notebooks : ℕ), notebooks ≥ n ∧ 
      (∃ (distribution : ℕ → ℕ × ℕ × ℕ), 
        (∀ i j, i ≠ j → (distribution i).2.2 ≠ (distribution j).2.2) ∧
        (∀ i, i < n → (distribution i).1 = pens / n ∧ (distribution i).2.1 = pencils / n)))) :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l3439_343941


namespace NUMINAMATH_CALUDE_max_area_rectangle_l3439_343989

/-- A rectangle with non-negative length and width. -/
structure Rectangle where
  length : ℝ
  width : ℝ
  length_nonneg : 0 ≤ length
  width_nonneg : 0 ≤ width

/-- The perimeter of a rectangle. -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- The area of a rectangle. -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- A rectangle with perimeter at least 80. -/
def RectangleWithLargePerimeter := {r : Rectangle // r.perimeter ≥ 80}

theorem max_area_rectangle (r : RectangleWithLargePerimeter) :
  r.val.area ≤ 400 ∧ 
  (r.val.area = 400 ↔ r.val.length = 20 ∧ r.val.width = 20) := by
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l3439_343989


namespace NUMINAMATH_CALUDE_second_rate_is_five_percent_l3439_343984

def total_sum : ℚ := 2678
def second_part : ℚ := 1648
def first_part : ℚ := total_sum - second_part
def first_rate : ℚ := 3 / 100
def first_duration : ℚ := 8
def second_duration : ℚ := 3

def first_interest : ℚ := first_part * first_rate * first_duration

theorem second_rate_is_five_percent : 
  ∃ (second_rate : ℚ), 
    second_rate * 100 = 5 ∧ 
    first_interest = second_part * second_rate * second_duration :=
sorry

end NUMINAMATH_CALUDE_second_rate_is_five_percent_l3439_343984


namespace NUMINAMATH_CALUDE_rent_spending_percentage_l3439_343987

theorem rent_spending_percentage (x : ℝ) : 
  x > 0 ∧ x < 100 ∧ 
  x + (x - 0.2 * x) + 28 = 100 → 
  x = 40 := by
sorry

end NUMINAMATH_CALUDE_rent_spending_percentage_l3439_343987


namespace NUMINAMATH_CALUDE_soccer_team_average_goals_l3439_343918

/-- Calculates the average number of goals per game for a soccer team -/
def average_goals_per_game (slices_per_pizza : ℕ) (pizzas_bought : ℕ) (games_played : ℕ) : ℚ :=
  (slices_per_pizza * pizzas_bought : ℚ) / games_played

/-- Theorem: Given the conditions, the average number of goals per game is 9 -/
theorem soccer_team_average_goals :
  let slices_per_pizza : ℕ := 12
  let pizzas_bought : ℕ := 6
  let games_played : ℕ := 8
  average_goals_per_game slices_per_pizza pizzas_bought games_played = 9 := by
sorry

#eval average_goals_per_game 12 6 8

end NUMINAMATH_CALUDE_soccer_team_average_goals_l3439_343918


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l3439_343964

def f (x : ℝ) : ℝ := x^2 - 6*x + 5

theorem monotonic_decreasing_interval :
  ∀ x y : ℝ, x < y → y ≤ 3 → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l3439_343964


namespace NUMINAMATH_CALUDE_platform_length_l3439_343988

/-- The length of a platform given train passing times and speed -/
theorem platform_length (train_speed : ℝ) (platform_pass_time : ℝ) (man_pass_time : ℝ) :
  train_speed = 54 →
  platform_pass_time = 32 →
  man_pass_time = 20 →
  (train_speed * (5/18) * platform_pass_time) - (train_speed * (5/18) * man_pass_time) = 180 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l3439_343988


namespace NUMINAMATH_CALUDE_a_gt_b_sufficient_not_necessary_for_a_sq_gt_b_sq_l3439_343948

theorem a_gt_b_sufficient_not_necessary_for_a_sq_gt_b_sq :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), x > 0 ∧ y > 0 → (x > y → x^2 > y^2)) ∧
  (a^2 > b^2 ∧ a ≤ b) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_b_sufficient_not_necessary_for_a_sq_gt_b_sq_l3439_343948


namespace NUMINAMATH_CALUDE_mechanic_average_earning_l3439_343953

/-- The average earning of a mechanic for a week, given specific conditions --/
theorem mechanic_average_earning (first_four_avg : ℚ) (last_four_avg : ℚ) (fourth_day : ℚ) :
  first_four_avg = 18 →
  last_four_avg = 22 →
  fourth_day = 13 →
  (4 * first_four_avg + 4 * last_four_avg - fourth_day) / 7 = 160 / 7 := by
  sorry

#eval (160 : ℚ) / 7

end NUMINAMATH_CALUDE_mechanic_average_earning_l3439_343953


namespace NUMINAMATH_CALUDE_proposition_is_false_l3439_343955

theorem proposition_is_false : ¬(∀ x : ℤ, x ∈ ({1, -1, 0} : Set ℤ) → 2*x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_proposition_is_false_l3439_343955


namespace NUMINAMATH_CALUDE_chocolate_division_l3439_343936

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) :
  total_chocolate = 48/5 →
  num_piles = 4 →
  total_chocolate / num_piles = 12/5 := by
sorry

end NUMINAMATH_CALUDE_chocolate_division_l3439_343936


namespace NUMINAMATH_CALUDE_jar_servings_calculation_l3439_343908

/-- Represents the contents and serving sizes of peanut butter and jelly in a jar -/
structure JarContents where
  pb_amount : ℚ  -- Amount of peanut butter in tablespoons
  jelly_amount : ℚ  -- Amount of jelly in tablespoons
  pb_serving : ℚ  -- Size of one peanut butter serving in tablespoons
  jelly_serving : ℚ  -- Size of one jelly serving in tablespoons

/-- Calculates the number of servings for peanut butter and jelly -/
def calculate_servings (jar : JarContents) : ℚ × ℚ :=
  (jar.pb_amount / jar.pb_serving, jar.jelly_amount / jar.jelly_serving)

/-- Theorem stating the correct number of servings for the given jar contents -/
theorem jar_servings_calculation (jar : JarContents)
  (h1 : jar.pb_amount = 35 + 2/3)
  (h2 : jar.jelly_amount = 18 + 1/2)
  (h3 : jar.pb_serving = 2 + 1/6)
  (h4 : jar.jelly_serving = 1) :
  calculate_servings jar = (16 + 18/39, 18 + 1/2) := by
  sorry

#eval calculate_servings {
  pb_amount := 35 + 2/3,
  jelly_amount := 18 + 1/2,
  pb_serving := 2 + 1/6,
  jelly_serving := 1
}

end NUMINAMATH_CALUDE_jar_servings_calculation_l3439_343908


namespace NUMINAMATH_CALUDE_min_value_3x_plus_9y_l3439_343997

theorem min_value_3x_plus_9y (x y : ℝ) (h : x + 2 * y = 2) :
  3 * x + 9 * y ≥ 6 ∧ ∃ x₀ y₀ : ℝ, x₀ + 2 * y₀ = 2 ∧ 3 * x₀ + 9 * y₀ = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_3x_plus_9y_l3439_343997


namespace NUMINAMATH_CALUDE_max_brownies_l3439_343962

theorem max_brownies (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) : 
  (m - 2) * (n - 2) = 2 * m + 2 * n - 4 → m * n ≤ 60 := by
sorry

end NUMINAMATH_CALUDE_max_brownies_l3439_343962


namespace NUMINAMATH_CALUDE_cookie_problem_l3439_343912

theorem cookie_problem (frank mike millie : ℕ) : 
  frank = (mike / 2) - 3 →
  mike = 3 * millie →
  frank = 3 →
  millie = 4 := by
sorry

end NUMINAMATH_CALUDE_cookie_problem_l3439_343912


namespace NUMINAMATH_CALUDE_line_points_l3439_343956

-- Define the points
def p1 : ℝ × ℝ := (8, 10)
def p2 : ℝ × ℝ := (2, -2)

-- Define the function to check if a point is on the line
def is_on_line (p : ℝ × ℝ) : Prop :=
  let m := (p1.2 - p2.2) / (p1.1 - p2.1)
  let b := p1.2 - m * p1.1
  p.2 = m * p.1 + b

-- Theorem statement
theorem line_points :
  is_on_line (5, 4) ∧
  is_on_line (1, -4) ∧
  ¬is_on_line (4, 1) ∧
  ¬is_on_line (3, -1) ∧
  ¬is_on_line (6, 7) :=
by sorry

end NUMINAMATH_CALUDE_line_points_l3439_343956


namespace NUMINAMATH_CALUDE_cubic_expansion_coefficient_l3439_343983

theorem cubic_expansion_coefficient (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = (a * x + Real.sqrt x)^3) ∧ 
   (∃ c : ℝ, ∀ x, f x = c * x^3 + x^(5/2) * Real.sqrt x + x^2 + Real.sqrt x * x + 1 ∧ c = 20)) →
  a = Real.rpow 20 (1/3) :=
sorry

end NUMINAMATH_CALUDE_cubic_expansion_coefficient_l3439_343983


namespace NUMINAMATH_CALUDE_georgie_guacamole_servings_l3439_343934

/-- The number of servings of guacamole Georgie can make -/
def guacamole_servings (avocados_per_serving : ℕ) (initial_avocados : ℕ) (bought_avocados : ℕ) : ℕ :=
  (initial_avocados + bought_avocados) / avocados_per_serving

/-- Theorem: Georgie can make 3 servings of guacamole -/
theorem georgie_guacamole_servings :
  guacamole_servings 3 5 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_georgie_guacamole_servings_l3439_343934


namespace NUMINAMATH_CALUDE_base_number_proof_l3439_343977

theorem base_number_proof (x : ℝ) (h : Real.sqrt (x^12) = 64) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l3439_343977


namespace NUMINAMATH_CALUDE_x_equation_solution_l3439_343937

theorem x_equation_solution (x : ℝ) (h : x + 1/x = 3) :
  x^12 - 7*x^8 + 2*x^4 = 44387*x - 15088 := by
  sorry

end NUMINAMATH_CALUDE_x_equation_solution_l3439_343937


namespace NUMINAMATH_CALUDE_base_8_to_7_conversion_l3439_343911

def base_8_to_10 (n : ℕ) : ℕ := 
  5 * 8^2 + 3 * 8^1 + 6 * 8^0

def base_10_to_7 (n : ℕ) : ℕ := 
  1 * 7^3 + 0 * 7^2 + 1 * 7^1 + 0 * 7^0

theorem base_8_to_7_conversion : 
  base_10_to_7 (base_8_to_10 536) = 1010 := by
  sorry

end NUMINAMATH_CALUDE_base_8_to_7_conversion_l3439_343911


namespace NUMINAMATH_CALUDE_line_point_k_l3439_343906

/-- A line is defined by three points it passes through -/
structure Line where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

/-- Check if a point lies on a given line -/
def lies_on_line (l : Line) (p : ℝ × ℝ) : Prop :=
  let (x1, y1) := l.p1
  let (x2, y2) := l.p2
  let (x3, y3) := l.p3
  let (x, y) := p
  (y - y1) * (x2 - x1) = (y2 - y1) * (x - x1) ∧
  (y - y2) * (x3 - x2) = (y3 - y2) * (x - x2)

/-- The main theorem -/
theorem line_point_k (l : Line) (k : ℝ) :
  l.p1 = (-1, 1) →
  l.p2 = (2, 5) →
  l.p3 = (5, 9) →
  lies_on_line l (50, k) →
  k = 69 := by
  sorry

end NUMINAMATH_CALUDE_line_point_k_l3439_343906


namespace NUMINAMATH_CALUDE_sarah_and_matt_age_sum_l3439_343980

/-- Given the age relationship between Sarah and Matt, prove that the sum of their current ages is 41 years. -/
theorem sarah_and_matt_age_sum :
  ∀ (sarah_age matt_age : ℝ),
  sarah_age = matt_age + 8 →
  sarah_age + 10 = 3 * (matt_age - 5) →
  sarah_age + matt_age = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_sarah_and_matt_age_sum_l3439_343980


namespace NUMINAMATH_CALUDE_division_remainder_l3439_343909

theorem division_remainder : ∃ q : ℕ, 1234567 = 256 * q + 229 ∧ 229 < 256 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l3439_343909


namespace NUMINAMATH_CALUDE_product_sum_relation_l3439_343907

theorem product_sum_relation (a b N : ℤ) : 
  b = 7 → 
  b - a = 4 → 
  a * b = 2 * (a + b) + N → 
  N = 1 := by
sorry

end NUMINAMATH_CALUDE_product_sum_relation_l3439_343907


namespace NUMINAMATH_CALUDE_art_dealer_etchings_sold_l3439_343901

theorem art_dealer_etchings_sold (total_earnings : ℕ) (price_low : ℕ) (price_high : ℕ) (num_low : ℕ) :
  total_earnings = 630 →
  price_low = 35 →
  price_high = 45 →
  num_low = 9 →
  ∃ (num_high : ℕ), num_low * price_low + num_high * price_high = total_earnings ∧ num_low + num_high = 16 :=
by sorry

end NUMINAMATH_CALUDE_art_dealer_etchings_sold_l3439_343901


namespace NUMINAMATH_CALUDE_tamias_dinner_problem_l3439_343914

/-- The number of smaller pieces each large slice is cut into, given the total number of bell peppers,
    the number of large slices per bell pepper, and the total number of slices and pieces. -/
def smaller_pieces_per_slice (total_peppers : ℕ) (large_slices_per_pepper : ℕ) (total_slices_and_pieces : ℕ) : ℕ :=
  let total_large_slices := total_peppers * large_slices_per_pepper
  let large_slices_to_cut := total_large_slices / 2
  let smaller_pieces_needed := total_slices_and_pieces - total_large_slices
  smaller_pieces_needed / large_slices_to_cut

theorem tamias_dinner_problem :
  smaller_pieces_per_slice 5 20 200 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tamias_dinner_problem_l3439_343914


namespace NUMINAMATH_CALUDE_range_of_m_l3439_343921

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, f m x < -m + 4) → m < 5/7 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3439_343921


namespace NUMINAMATH_CALUDE_sqrt_sum_reciprocals_l3439_343978

theorem sqrt_sum_reciprocals : Real.sqrt (1 / 4 + 1 / 25) = Real.sqrt 29 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_reciprocals_l3439_343978


namespace NUMINAMATH_CALUDE_frog_grasshopper_jump_difference_l3439_343999

/-- The difference between the frog's jump distance and the grasshopper's jump distance is 15 inches -/
theorem frog_grasshopper_jump_difference :
  let frog_jump : ℕ := 40
  let grasshopper_jump : ℕ := 25
  frog_jump - grasshopper_jump = 15 := by
sorry

end NUMINAMATH_CALUDE_frog_grasshopper_jump_difference_l3439_343999


namespace NUMINAMATH_CALUDE_short_bingo_first_column_count_l3439_343981

def short_bingo_first_column_possibilities : ℕ := 360360

theorem short_bingo_first_column_count :
  (Finset.range 15).card.choose 5 = short_bingo_first_column_possibilities :=
by sorry

end NUMINAMATH_CALUDE_short_bingo_first_column_count_l3439_343981


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l3439_343949

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 4 : ℝ) * Real.sqrt (4 * t.leg^2 - t.base^2 : ℝ)

/-- Theorem stating the minimum common perimeter of two specific isosceles triangles -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1.base * 8 = t2.base * 9 ∧
    t1 ≠ t2 ∧
    area t1 = area t2 ∧
    perimeter t1 = perimeter t2 ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1.base * 8 = s2.base * 9 →
      s1 ≠ s2 →
      area s1 = area s2 →
      perimeter s1 = perimeter s2 →
      perimeter t1 ≤ perimeter s1 ∧
    perimeter t1 = 960 :=
  sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l3439_343949


namespace NUMINAMATH_CALUDE_complement_of_P_in_U_l3439_343927

def U : Finset Int := {-1, 0, 1, 2}

def P : Set Int := {x | -Real.sqrt 2 < x ∧ x < Real.sqrt 2}

theorem complement_of_P_in_U : 
  (U.toSet \ P) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_P_in_U_l3439_343927


namespace NUMINAMATH_CALUDE_parallelogram_base_l3439_343990

/-- Given a parallelogram with area 864 square cm and height 24 cm, its base is 36 cm. -/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 864 ∧ height = 24 ∧ area = base * height → base = 36 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l3439_343990


namespace NUMINAMATH_CALUDE_exam_pass_probability_l3439_343931

/-- The probability of answering a single question correctly -/
def p : ℝ := 0.4

/-- The number of questions in the exam -/
def n : ℕ := 4

/-- The minimum number of correct answers required to pass -/
def k : ℕ := 3

/-- The probability of passing the exam -/
def pass_probability : ℝ := 
  (Nat.choose n k * p^k * (1-p)^(n-k)) + (p^n)

theorem exam_pass_probability : pass_probability = 112/625 := by
  sorry

end NUMINAMATH_CALUDE_exam_pass_probability_l3439_343931


namespace NUMINAMATH_CALUDE_distribution_centers_count_l3439_343985

/-- The number of unique representations using either a single color or a pair of different colors -/
def uniqueRepresentations (n : ℕ) : ℕ := n + n.choose 2

/-- Theorem stating that with 5 colors, there are 15 unique representations -/
theorem distribution_centers_count : uniqueRepresentations 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_distribution_centers_count_l3439_343985


namespace NUMINAMATH_CALUDE_min_side_difference_l3439_343991

theorem min_side_difference (a b c : ℕ) : 
  a + b + c = 3010 →
  a < b →
  b ≤ c →
  (∀ x y z : ℕ, x + y + z = 3010 → x < y → y ≤ z → b - a ≤ y - x) →
  b - a = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_side_difference_l3439_343991


namespace NUMINAMATH_CALUDE_sum_of_squares_geq_twice_product_l3439_343951

theorem sum_of_squares_geq_twice_product (a b : ℝ) : a^2 + b^2 ≥ 2*a*b := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_geq_twice_product_l3439_343951


namespace NUMINAMATH_CALUDE_number_divisibility_l3439_343942

theorem number_divisibility (N : ℕ) (h1 : N % 68 = 0) (h2 : N % 67 = 1) : N = 68 := by
  sorry

end NUMINAMATH_CALUDE_number_divisibility_l3439_343942


namespace NUMINAMATH_CALUDE_shadow_length_change_l3439_343986

/-- Represents the length of a shadow -/
inductive ShadowLength
  | Long
  | Short

/-- Represents a time of day -/
inductive TimeOfDay
  | Morning
  | Noon
  | Afternoon

/-- Represents the direction of a shadow -/
inductive ShadowDirection
  | West
  | North
  | East

/-- Function to determine shadow length based on time of day -/
def shadowLengthAtTime (time : TimeOfDay) : ShadowLength :=
  match time with
  | TimeOfDay.Morning => ShadowLength.Long
  | TimeOfDay.Noon => ShadowLength.Short
  | TimeOfDay.Afternoon => ShadowLength.Long

/-- Function to determine shadow direction based on time of day -/
def shadowDirectionAtTime (time : TimeOfDay) : ShadowDirection :=
  match time with
  | TimeOfDay.Morning => ShadowDirection.West
  | TimeOfDay.Noon => ShadowDirection.North
  | TimeOfDay.Afternoon => ShadowDirection.East

/-- Theorem stating the change in shadow length throughout the day -/
theorem shadow_length_change :
  ∀ (t1 t2 t3 : TimeOfDay),
    t1 = TimeOfDay.Morning →
    t2 = TimeOfDay.Noon →
    t3 = TimeOfDay.Afternoon →
    (shadowLengthAtTime t1 = ShadowLength.Long ∧
     shadowLengthAtTime t2 = ShadowLength.Short ∧
     shadowLengthAtTime t3 = ShadowLength.Long) :=
by
  sorry

#check shadow_length_change

end NUMINAMATH_CALUDE_shadow_length_change_l3439_343986


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3439_343961

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def N : Set ℝ := {x | 2*x + 1 < 5}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3439_343961


namespace NUMINAMATH_CALUDE_marks_difference_l3439_343944

/-- Given that the average mark in chemistry and mathematics is 55,
    prove that the difference between the total marks in all three subjects
    and the marks in physics is 110. -/
theorem marks_difference (P C M : ℝ) 
    (h1 : (C + M) / 2 = 55) : 
    (P + C + M) - P = 110 := by
  sorry

end NUMINAMATH_CALUDE_marks_difference_l3439_343944


namespace NUMINAMATH_CALUDE_power_3_2023_mod_5_l3439_343982

theorem power_3_2023_mod_5 : 3^2023 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_3_2023_mod_5_l3439_343982


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_12_l3439_343943

theorem largest_four_digit_divisible_by_12 : ∃ n : ℕ, n = 9996 ∧ 
  n % 12 = 0 ∧ 
  n ≤ 9999 ∧ 
  n ≥ 1000 ∧
  ∀ m : ℕ, m % 12 = 0 ∧ m ≤ 9999 ∧ m ≥ 1000 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_12_l3439_343943


namespace NUMINAMATH_CALUDE_betty_lipstick_count_l3439_343928

/-- Represents an order with different items -/
structure Order where
  total_items : ℕ
  slipper_count : ℕ
  slipper_price : ℚ
  lipstick_price : ℚ
  hair_color_count : ℕ
  hair_color_price : ℚ
  total_paid : ℚ

/-- Calculates the number of lipstick pieces in an order -/
def lipstick_count (o : Order) : ℕ :=
  let slipper_cost := o.slipper_count * o.slipper_price
  let hair_color_cost := o.hair_color_count * o.hair_color_price
  let lipstick_cost := o.total_paid - slipper_cost - hair_color_cost
  (lipstick_cost / o.lipstick_price).num.toNat

/-- Betty's order satisfies the given conditions -/
def betty_order : Order :=
  { total_items := 18
  , slipper_count := 6
  , slipper_price := 5/2
  , lipstick_price := 5/4
  , hair_color_count := 8
  , hair_color_price := 3
  , total_paid := 44 }

theorem betty_lipstick_count : lipstick_count betty_order = 4 := by
  sorry

end NUMINAMATH_CALUDE_betty_lipstick_count_l3439_343928


namespace NUMINAMATH_CALUDE_percentage_difference_l3439_343940

theorem percentage_difference (X : ℝ) (h : X > 0) : 
  let first_number := 0.70 * X
  let second_number := 0.63 * X
  (first_number - second_number) / first_number * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l3439_343940


namespace NUMINAMATH_CALUDE_cosine_sum_product_simplification_l3439_343916

theorem cosine_sum_product_simplification (α β : ℝ) :
  Real.cos (α + β) * Real.cos β + Real.sin (α + β) * Real.sin β = Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_product_simplification_l3439_343916


namespace NUMINAMATH_CALUDE_largest_valid_number_l3439_343903

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧  -- four-digit number
  (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10) ∧  -- all digits are different
  (∀ i j, i < j → (n / 10^i) % 10 ≤ (n / 10^j) % 10)  -- digits in ascending order

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 7089 :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3439_343903


namespace NUMINAMATH_CALUDE_power_of_two_equality_l3439_343995

theorem power_of_two_equality (x : ℕ) : 32^10 + 32^10 + 32^10 + 32^10 + 32^10 = 2^x ↔ x = 52 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l3439_343995


namespace NUMINAMATH_CALUDE_yah_to_bah_conversion_l3439_343915

/-- Define conversion rates between bahs, rahs, and yahs -/
def bah_to_rah_rate : ℚ := 27 / 18
def rah_to_yah_rate : ℚ := 20 / 12

/-- Theorem stating the equivalence between 800 yahs and 320 bahs -/
theorem yah_to_bah_conversion : 
  ∀ (bahs rahs yahs : ℚ),
  (18 : ℚ) * bahs = (27 : ℚ) * rahs →
  (12 : ℚ) * rahs = (20 : ℚ) * yahs →
  (800 : ℚ) * yahs = (320 : ℚ) * bahs := by
  sorry

end NUMINAMATH_CALUDE_yah_to_bah_conversion_l3439_343915


namespace NUMINAMATH_CALUDE_stamp_ratio_problem_l3439_343923

theorem stamp_ratio_problem (k a : ℕ) : 
  k > 0 ∧ a > 0 →  -- Initial numbers of stamps are positive
  (k - 12) / (a + 12) = 8 / 6 →  -- Ratio after exchange
  k - 12 = a + 12 + 32 →  -- Kaye has 32 more stamps after exchange
  k / a = 5 / 3 :=  -- Initial ratio
by sorry

end NUMINAMATH_CALUDE_stamp_ratio_problem_l3439_343923


namespace NUMINAMATH_CALUDE_jacket_price_reduction_l3439_343917

/-- Calculates the final price of a jacket after two successive price reductions -/
theorem jacket_price_reduction (initial_price : ℝ) (first_reduction : ℝ) (second_reduction : ℝ) :
  initial_price = 20 ∧ 
  first_reduction = 0.2 ∧ 
  second_reduction = 0.25 →
  initial_price * (1 - first_reduction) * (1 - second_reduction) = 12 :=
by sorry

end NUMINAMATH_CALUDE_jacket_price_reduction_l3439_343917


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3439_343959

theorem units_digit_of_product : (5^3 * 7^52) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3439_343959


namespace NUMINAMATH_CALUDE_work_completion_time_l3439_343926

/-- Represents the work rate of one person for one hour -/
structure WorkRate where
  man : ℝ
  woman : ℝ

/-- Represents a work scenario -/
structure WorkScenario where
  men : ℕ
  women : ℕ
  hours : ℝ
  days : ℝ

/-- Calculates the total work done in a scenario -/
def totalWork (rate : WorkRate) (scenario : WorkScenario) : ℝ :=
  (scenario.men * rate.man + scenario.women * rate.woman) * scenario.hours * scenario.days

/-- The theorem to be proved -/
theorem work_completion_time 
  (rate : WorkRate)
  (scenario1 scenario2 scenario3 : WorkScenario) :
  scenario1.men = 1 ∧ 
  scenario1.women = 3 ∧ 
  scenario1.hours = 7 ∧
  scenario2.men = 4 ∧ 
  scenario2.women = 4 ∧ 
  scenario2.hours = 3 ∧ 
  scenario2.days = 7 ∧
  scenario3.men = 7 ∧ 
  scenario3.women = 0 ∧ 
  scenario3.hours = 4 ∧ 
  scenario3.days = 5.000000000000001 ∧
  totalWork rate scenario1 = totalWork rate scenario2 ∧
  totalWork rate scenario2 = totalWork rate scenario3
  →
  scenario1.days = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3439_343926


namespace NUMINAMATH_CALUDE_sara_savings_l3439_343929

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of quarters Sara has -/
def sara_quarters : ℕ := 11

/-- Theorem: Sara's total savings in cents -/
theorem sara_savings : quarter_value * sara_quarters = 275 := by
  sorry

end NUMINAMATH_CALUDE_sara_savings_l3439_343929


namespace NUMINAMATH_CALUDE_apple_count_in_second_group_l3439_343970

/-- The cost of an apple in dollars -/
def apple_cost : ℚ := 21/100

/-- The cost of an orange in dollars -/
def orange_cost : ℚ := 17/100

/-- The number of apples in the second group -/
def x : ℕ := 2

theorem apple_count_in_second_group :
  (6 * apple_cost + 3 * orange_cost = 177/100) →
  (↑x * apple_cost + 5 * orange_cost = 127/100) →
  (apple_cost = 21/100) →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_apple_count_in_second_group_l3439_343970


namespace NUMINAMATH_CALUDE_hash_2_neg1_4_l3439_343946

def hash (a b c : ℝ) : ℝ := a * b^2 - 3 * a - 5 * c

theorem hash_2_neg1_4 : hash 2 (-1) 4 = -24 := by
  sorry

end NUMINAMATH_CALUDE_hash_2_neg1_4_l3439_343946


namespace NUMINAMATH_CALUDE_urn_probability_l3439_343976

/-- Represents the contents of the urn -/
structure UrnContents where
  red : ℕ
  blue : ℕ

/-- Represents a single operation of drawing and adding balls -/
inductive Operation
  | DrawRed
  | DrawBlue

/-- The initial state of the urn -/
def initial_urn : UrnContents := ⟨2, 1⟩

/-- Perform a single operation on the urn -/
def perform_operation (urn : UrnContents) (op : Operation) : UrnContents :=
  match op with
  | Operation.DrawRed => ⟨urn.red + 2, urn.blue⟩
  | Operation.DrawBlue => ⟨urn.red, urn.blue + 2⟩

/-- Perform a sequence of operations on the urn -/
def perform_operations (urn : UrnContents) (ops : List Operation) : UrnContents :=
  ops.foldl perform_operation urn

/-- Calculate the probability of a specific sequence of operations -/
def sequence_probability (ops : List Operation) : ℚ :=
  sorry

/-- Calculate the total probability of all valid sequences -/
def total_probability (valid_sequences : List (List Operation)) : ℚ :=
  sorry

theorem urn_probability : 
  ∃ (valid_sequences : List (List Operation)),
    (∀ seq ∈ valid_sequences, seq.length = 5) ∧
    (∀ seq ∈ valid_sequences, 
      let final_urn := perform_operations initial_urn seq
      final_urn.red + final_urn.blue = 12 ∧
      final_urn.red = 7 ∧ final_urn.blue = 5) ∧
    total_probability valid_sequences = 25 / 224 :=
  sorry

end NUMINAMATH_CALUDE_urn_probability_l3439_343976


namespace NUMINAMATH_CALUDE_shaded_area_of_overlapping_sectors_l3439_343935

/-- The area of the shaded region formed by two overlapping sectors of a circle -/
theorem shaded_area_of_overlapping_sectors (r : ℝ) (θ : ℝ) (h1 : r = 15) (h2 : θ = 45) :
  let sector_area := θ / 360 * π * r^2
  let triangle_area := Real.sqrt 3 / 4 * r^2
  2 * (sector_area - triangle_area) = 56.25 * π - 112.5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_overlapping_sectors_l3439_343935


namespace NUMINAMATH_CALUDE_apartment_333_on_third_floor_l3439_343993

/-- Represents a building with apartments -/
structure Building where
  floors : ℕ
  entrances : ℕ
  apartments_per_floor : ℕ

/-- Calculates the total number of apartments in the building -/
def total_apartments (b : Building) : ℕ :=
  b.floors * b.entrances * b.apartments_per_floor

/-- Calculates the floor number for a given apartment number -/
def apartment_floor (b : Building) (apartment_number : ℕ) : ℕ :=
  ((apartment_number - 1) / b.apartments_per_floor) % b.floors + 1

/-- The specific building described in the problem -/
def problem_building : Building :=
  { floors := 9
  , entrances := 10
  , apartments_per_floor := 4 }

theorem apartment_333_on_third_floor :
  apartment_floor problem_building 333 = 3 := by
  sorry

#eval apartment_floor problem_building 333

end NUMINAMATH_CALUDE_apartment_333_on_third_floor_l3439_343993


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_ge_five_l3439_343957

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*(a-1)*x + 2

/-- Theorem stating the condition for f(x) to be increasing on (-∞, 4) -/
theorem f_increasing_iff_a_ge_five (a : ℝ) :
  (∀ x y, x < y ∧ y < 4 → f a x < f a y) ↔ a ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_ge_five_l3439_343957


namespace NUMINAMATH_CALUDE_probability_red_ball_3_2_l3439_343924

/-- Represents the probability of drawing a red ball from a box containing red and yellow balls. -/
def probability_red_ball (red_balls yellow_balls : ℕ) : ℚ :=
  red_balls / (red_balls + yellow_balls)

/-- Theorem stating that the probability of drawing a red ball from a box with 3 red balls and 2 yellow balls is 3/5. -/
theorem probability_red_ball_3_2 :
  probability_red_ball 3 2 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_ball_3_2_l3439_343924


namespace NUMINAMATH_CALUDE_tall_students_not_well_defined_other_options_well_defined_l3439_343913

-- Define a type for potential sets
inductive PotentialSet
  | NaturalNumbers1to20
  | AllRectangles
  | NaturalNumbersLessThan10
  | TallStudents

-- Define a predicate for well-defined sets
def isWellDefinedSet (s : PotentialSet) : Prop :=
  match s with
  | PotentialSet.NaturalNumbers1to20 => true
  | PotentialSet.AllRectangles => true
  | PotentialSet.NaturalNumbersLessThan10 => true
  | PotentialSet.TallStudents => false

-- Theorem stating that "Tall students" is not a well-defined set
theorem tall_students_not_well_defined :
  ¬(isWellDefinedSet PotentialSet.TallStudents) :=
by sorry

-- Theorem stating that other options are well-defined sets
theorem other_options_well_defined :
  (isWellDefinedSet PotentialSet.NaturalNumbers1to20) ∧
  (isWellDefinedSet PotentialSet.AllRectangles) ∧
  (isWellDefinedSet PotentialSet.NaturalNumbersLessThan10) :=
by sorry

end NUMINAMATH_CALUDE_tall_students_not_well_defined_other_options_well_defined_l3439_343913


namespace NUMINAMATH_CALUDE_scientific_notation_239000000_l3439_343930

theorem scientific_notation_239000000 :
  239000000 = 2.39 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_239000000_l3439_343930


namespace NUMINAMATH_CALUDE_sum_of_decimals_l3439_343968

/-- The sum of 0.2, 0.03, 0.004, 0.0005, and 0.00006 is equal to 5864/25000 -/
theorem sum_of_decimals : 
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 = 5864 / 25000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l3439_343968


namespace NUMINAMATH_CALUDE_bookstore_problem_l3439_343925

theorem bookstore_problem (x y : ℕ) 
  (h1 : x + y = 5000)
  (h2 : (x - 400) / 2 - (y + 400) = 400) :
  x - y = 3000 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_problem_l3439_343925


namespace NUMINAMATH_CALUDE_perfect_square_difference_l3439_343963

theorem perfect_square_difference (m n : ℕ+) 
  (h : 2001 * m^2 + m = 2002 * n^2 + n) :
  ∃ k : ℕ, m - n = k^2 := by sorry

end NUMINAMATH_CALUDE_perfect_square_difference_l3439_343963


namespace NUMINAMATH_CALUDE_salary_proof_l3439_343958

/-- The weekly salary of employee N -/
def N_salary : ℝ := 275

/-- The weekly salary of employee M -/
def M_salary (N_salary : ℝ) : ℝ := 1.2 * N_salary

/-- The total weekly salary for both employees -/
def total_salary : ℝ := 605

theorem salary_proof :
  N_salary + M_salary N_salary = total_salary :=
sorry

end NUMINAMATH_CALUDE_salary_proof_l3439_343958


namespace NUMINAMATH_CALUDE_exponent_equation_solution_l3439_343910

theorem exponent_equation_solution : ∃ n : ℤ, 5^3 - 7 = 6^2 + n ∧ n = 82 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_solution_l3439_343910


namespace NUMINAMATH_CALUDE_four_plus_six_equals_ten_l3439_343966

theorem four_plus_six_equals_ten : 4 + 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_four_plus_six_equals_ten_l3439_343966


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3439_343979

theorem arithmetic_expression_equality : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 - (-9) = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3439_343979


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_iff_abs_a_gt_one_l3439_343945

-- Define the equation
def equation (a x y : ℤ) : Prop := x^2 + a*x*y + y^2 = 1

-- Define the property of having infinitely many integer solutions
def has_infinitely_many_solutions (a : ℤ) : Prop :=
  ∀ n : ℕ, ∃ (x y : ℤ), equation a x y ∧ x.natAbs + y.natAbs > n

-- Theorem statement
theorem infinitely_many_solutions_iff_abs_a_gt_one (a : ℤ) :
  has_infinitely_many_solutions a ↔ a.natAbs > 1 := by sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_iff_abs_a_gt_one_l3439_343945


namespace NUMINAMATH_CALUDE_decision_block_two_exits_l3439_343938

-- Define the types of program blocks
inductive ProgramBlock
  | Termination
  | InputOutput
  | Processing
  | Decision

-- Define a function to determine if a block has two exit directions
def hasTwoExitDirections (block : ProgramBlock) : Prop :=
  match block with
  | ProgramBlock.Decision => true
  | _ => false

-- Theorem statement
theorem decision_block_two_exits :
  ∀ (block : ProgramBlock),
    hasTwoExitDirections block ↔ block = ProgramBlock.Decision :=
by sorry

end NUMINAMATH_CALUDE_decision_block_two_exits_l3439_343938


namespace NUMINAMATH_CALUDE_angle_measure_problem_l3439_343974

theorem angle_measure_problem (α : ℝ) : 
  (180 - α) = 3 * (90 - α) + 10 → α = 50 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_problem_l3439_343974


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3439_343992

theorem inequality_solution_range (k : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x^2 + k*x - 1 > 0) → k > -3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3439_343992


namespace NUMINAMATH_CALUDE_odds_against_C_winning_l3439_343994

-- Define the type for horses
inductive Horse : Type
| A
| B
| C

-- Define the function for odds against winning
def oddsAgainst (h : Horse) : ℚ :=
  match h with
  | Horse.A => 4 / 1
  | Horse.B => 3 / 4
  | Horse.C => 27 / 8

-- State the theorem
theorem odds_against_C_winning :
  (∀ h : Horse, oddsAgainst h > 0) →  -- Ensure all odds are positive
  (∀ h1 h2 : Horse, h1 ≠ h2 → oddsAgainst h1 ≠ oddsAgainst h2) →  -- No ties
  oddsAgainst Horse.A = 4 / 1 →
  oddsAgainst Horse.B = 3 / 4 →
  oddsAgainst Horse.C = 27 / 8 := by
  sorry


end NUMINAMATH_CALUDE_odds_against_C_winning_l3439_343994


namespace NUMINAMATH_CALUDE_cubic_function_uniqueness_l3439_343975

/-- Given a cubic function f(x) = ax³ + bx² passing through (-1, 2) with slope -3 at x = -1,
    prove that f(x) = x³ + 3x² -/
theorem cubic_function_uniqueness (a b : ℝ) : 
  let f := fun (x : ℝ) ↦ a * x^3 + b * x^2
  let f' := fun (x : ℝ) ↦ 3 * a * x^2 + 2 * b * x
  (f (-1) = 2) → (f' (-1) = -3) → (a = 1 ∧ b = 3) := by sorry

end NUMINAMATH_CALUDE_cubic_function_uniqueness_l3439_343975


namespace NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l3439_343920

theorem students_taking_neither_music_nor_art 
  (total_students : ℕ) 
  (music_students : ℕ) 
  (art_students : ℕ) 
  (both_students : ℕ) 
  (h1 : total_students = 500) 
  (h2 : music_students = 30) 
  (h3 : art_students = 10) 
  (h4 : both_students = 10) : 
  total_students - (music_students + art_students - both_students) = 460 := by
sorry

end NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l3439_343920


namespace NUMINAMATH_CALUDE_smallest_780_divisible_by_1125_l3439_343902

def is_composed_of_780 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 7 ∨ d = 8 ∨ d = 0

theorem smallest_780_divisible_by_1125 :
  ∀ n : ℕ, n > 0 → is_composed_of_780 n → n % 1125 = 0 → n ≥ 77778000 :=
sorry

end NUMINAMATH_CALUDE_smallest_780_divisible_by_1125_l3439_343902


namespace NUMINAMATH_CALUDE_cups_per_serving_l3439_343969

/-- Given a recipe that requires 18.0 servings of cereal and 36 cups in total,
    prove that each serving consists of 2 cups. -/
theorem cups_per_serving (servings : Real) (total_cups : Nat) 
    (h1 : servings = 18.0) (h2 : total_cups = 36) : 
    (total_cups : Real) / servings = 2 := by
  sorry

end NUMINAMATH_CALUDE_cups_per_serving_l3439_343969


namespace NUMINAMATH_CALUDE_octal_calculation_l3439_343965

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Converts a natural number to its octal representation --/
def to_octal (n : ℕ) : OctalNumber :=
  sorry

/-- Performs subtraction in base 8 --/
def octal_sub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Theorem stating the result of the given octal calculation --/
theorem octal_calculation : 
  octal_sub (octal_sub (to_octal 123) (to_octal 51)) (to_octal 15) = to_octal 25 :=
sorry

end NUMINAMATH_CALUDE_octal_calculation_l3439_343965


namespace NUMINAMATH_CALUDE_toms_work_schedule_l3439_343971

theorem toms_work_schedule (summer_hours_per_week : ℝ) (summer_weeks : ℕ) 
  (summer_total_earnings : ℝ) (semester_weeks : ℕ) (semester_target_earnings : ℝ) :
  summer_hours_per_week = 40 →
  summer_weeks = 8 →
  summer_total_earnings = 3200 →
  semester_weeks = 24 →
  semester_target_earnings = 2400 →
  let hourly_wage := summer_total_earnings / (summer_hours_per_week * summer_weeks)
  let semester_hours_per_week := semester_target_earnings / (hourly_wage * semester_weeks)
  semester_hours_per_week = 10 := by
  sorry

end NUMINAMATH_CALUDE_toms_work_schedule_l3439_343971


namespace NUMINAMATH_CALUDE_grandma_crane_folding_l3439_343932

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem grandma_crane_folding :
  let foldTime : Nat := 3  -- Time to fold one crane
  let restTime : Nat := 1  -- Rest time after folding each crane
  let startTime : Time := { hours := 14, minutes := 30 }  -- 2:30 PM
  let numCranes : Nat := 5
  
  let totalFoldTime := foldTime * numCranes
  let totalRestTime := restTime * (numCranes - 1)
  let totalTime := totalFoldTime + totalRestTime
  
  addMinutes startTime totalTime = { hours := 14, minutes := 49 }  -- 2:49 PM
  := by sorry

end NUMINAMATH_CALUDE_grandma_crane_folding_l3439_343932


namespace NUMINAMATH_CALUDE_min_handshakes_30_people_l3439_343904

/-- The minimum number of handshakes in a gathering -/
def min_handshakes (n : ℕ) (k : ℕ) : ℕ := n * k / 2

/-- Theorem: In a gathering of 30 people, where each person shakes hands
    with at least three other people, the minimum possible number of handshakes is 45 -/
theorem min_handshakes_30_people :
  let n : ℕ := 30
  let k : ℕ := 3
  min_handshakes n k = 45 := by
  sorry


end NUMINAMATH_CALUDE_min_handshakes_30_people_l3439_343904


namespace NUMINAMATH_CALUDE_apple_pairing_l3439_343922

theorem apple_pairing (weights : Fin 300 → ℝ) 
  (h_positive : ∀ i, weights i > 0)
  (h_ratio : ∀ i j, weights i ≤ 3 * weights j) :
  ∃ (pairs : Fin 150 → Fin 300 × Fin 300),
    (∀ i, (pairs i).1 ≠ (pairs i).2) ∧
    (∀ i, i ≠ j → (pairs i).1 ≠ (pairs j).1 ∧ (pairs i).1 ≠ (pairs j).2 ∧
                  (pairs i).2 ≠ (pairs j).1 ∧ (pairs i).2 ≠ (pairs j).2) ∧
    (∀ i j, weights (pairs i).1 + weights (pairs i).2 ≤ 
            2 * (weights (pairs j).1 + weights (pairs j).2)) :=
sorry

end NUMINAMATH_CALUDE_apple_pairing_l3439_343922


namespace NUMINAMATH_CALUDE_vector_properties_l3439_343998

def a : ℝ × ℝ := (3, 0)
def b : ℝ × ℝ := (-5, 5)
def c (k : ℝ) : ℝ × ℝ := (2, k)

theorem vector_properties :
  (∃ θ : ℝ, θ = Real.pi * 3 / 4 ∧ 
    Real.cos θ = (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) ∧
  (∃ k : ℝ, b.1 / (c k).1 = b.2 / (c k).2 → k = -2) ∧
  (∃ k : ℝ, b.1 * (a.1 + (c k).1) + b.2 * (a.2 + (c k).2) = 0 → k = 5) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l3439_343998


namespace NUMINAMATH_CALUDE_petting_zoo_count_l3439_343950

/-- The number of animals Mary counted -/
def mary_count : ℕ := 130

/-- The number of animals Mary double-counted -/
def double_counted : ℕ := 19

/-- The number of animals Mary missed -/
def missed : ℕ := 39

/-- The actual number of animals in the petting zoo -/
def actual_count : ℕ := 150

theorem petting_zoo_count : 
  mary_count - double_counted + missed = actual_count := by sorry

end NUMINAMATH_CALUDE_petting_zoo_count_l3439_343950


namespace NUMINAMATH_CALUDE_units_digit_G_500_l3439_343947

/-- The function G_n is defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem: The units digit of G(500) is 0 -/
theorem units_digit_G_500 : unitsDigit (G 500) = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_500_l3439_343947
