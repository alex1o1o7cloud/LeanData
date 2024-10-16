import Mathlib

namespace NUMINAMATH_CALUDE_hannah_dessert_cost_l308_30874

def county_fair_problem (initial_amount : ℝ) (amount_left : ℝ) : Prop :=
  let total_spent := initial_amount - amount_left
  let rides_cost := initial_amount / 2
  let dessert_cost := total_spent - rides_cost
  dessert_cost = 5

theorem hannah_dessert_cost :
  county_fair_problem 30 10 := by
  sorry

end NUMINAMATH_CALUDE_hannah_dessert_cost_l308_30874


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l308_30801

theorem consecutive_integers_product (a b c d e : ℤ) : 
  (a + b + c + d + e) / 5 = 17 ∧ 
  d = 12 ∧ 
  e = 22 ∧ 
  (∃ n : ℤ, a = n ∧ b = n + 1 ∧ c = n + 2) →
  a * b * c = 4896 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l308_30801


namespace NUMINAMATH_CALUDE_cards_per_player_l308_30830

/-- Proves that evenly distributing 54 cards among 3 players results in 18 cards per player -/
theorem cards_per_player (initial_cards : ℕ) (added_cards : ℕ) (num_players : ℕ) :
  initial_cards = 52 →
  added_cards = 2 →
  num_players = 3 →
  (initial_cards + added_cards) / num_players = 18 := by
sorry

end NUMINAMATH_CALUDE_cards_per_player_l308_30830


namespace NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l308_30868

theorem chocolate_bars_in_large_box :
  let small_boxes : ℕ := 16
  let bars_per_small_box : ℕ := 25
  let total_bars : ℕ := small_boxes * bars_per_small_box
  total_bars = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l308_30868


namespace NUMINAMATH_CALUDE_longest_side_of_special_triangle_l308_30887

-- Define a triangle with sides in arithmetic progression
structure ArithmeticTriangle where
  a : ℝ
  d : ℝ
  angle : ℝ

-- Theorem statement
theorem longest_side_of_special_triangle (t : ArithmeticTriangle) 
  (h1 : t.d = 2)
  (h2 : t.angle = 2 * π / 3) -- 120° in radians
  (h3 : (t.a + t.d)^2 = (t.a - t.d)^2 + t.a^2 - 2*(t.a - t.d)*t.a*(- 1/2)) -- Law of Cosines for 120°
  : t.a + t.d = 7 := by
  sorry

end NUMINAMATH_CALUDE_longest_side_of_special_triangle_l308_30887


namespace NUMINAMATH_CALUDE_correct_dial_probability_l308_30862

/-- The probability of correctly dialing a phone number with a missing last digit -/
def dial_probability : ℚ := 3 / 10

/-- The number of possible digits for a phone number -/
def num_digits : ℕ := 10

/-- The maximum number of attempts allowed -/
def max_attempts : ℕ := 3

theorem correct_dial_probability :
  (∀ n : ℕ, n ≤ max_attempts → (1 : ℚ) / num_digits = 1 / 10) →
  (∀ n : ℕ, n < max_attempts → (num_digits - n : ℚ) / num_digits * (1 : ℚ) / (num_digits - n) = 1 / 10) →
  dial_probability = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_dial_probability_l308_30862


namespace NUMINAMATH_CALUDE_popcorn_cost_l308_30814

/-- The cost of each box of popcorn for three friends splitting movie expenses -/
theorem popcorn_cost (ticket_price movie_tickets popcorn_boxes milktea_price milktea_cups individual_contribution : ℚ) :
  (ticket_price = 7) →
  (movie_tickets = 3) →
  (popcorn_boxes = 2) →
  (milktea_price = 3) →
  (milktea_cups = 3) →
  (individual_contribution = 11) →
  (((ticket_price * movie_tickets) + (milktea_price * milktea_cups) + 
    (popcorn_boxes * ((individual_contribution * 3) - 
    (ticket_price * movie_tickets) - (milktea_price * milktea_cups)) / popcorn_boxes)) / 3 = individual_contribution) →
  ((individual_contribution * 3) - (ticket_price * movie_tickets) - (milktea_price * milktea_cups)) / popcorn_boxes = (3/2 : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_popcorn_cost_l308_30814


namespace NUMINAMATH_CALUDE_quadratic_function_from_roots_and_point_l308_30836

theorem quadratic_function_from_roots_and_point (f : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →  -- f is quadratic
  f 0 = 2 →                                       -- f(0) = 2
  (∃ x, f x = 0 ∧ x = -2) →                       -- -2 is a root
  (∃ x, f x = 0 ∧ x = 1) →                        -- 1 is a root
  ∀ x, f x = -x^2 - x + 2 :=                      -- Conclusion
by sorry

end NUMINAMATH_CALUDE_quadratic_function_from_roots_and_point_l308_30836


namespace NUMINAMATH_CALUDE_heather_initial_blocks_l308_30873

/-- The number of blocks Heather shared with Jose -/
def shared_blocks : ℕ := 41

/-- The number of blocks Heather ended up with -/
def remaining_blocks : ℕ := 45

/-- The initial number of blocks Heather had -/
def initial_blocks : ℕ := shared_blocks + remaining_blocks

theorem heather_initial_blocks : initial_blocks = 86 := by
  sorry

end NUMINAMATH_CALUDE_heather_initial_blocks_l308_30873


namespace NUMINAMATH_CALUDE_probability_one_white_ball_l308_30880

/-- The probability of drawing exactly one white ball when randomly selecting two balls from a bag containing 2 white and 3 black balls -/
theorem probability_one_white_ball (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) :
  total_balls = white_balls + black_balls →
  total_balls = 5 →
  white_balls = 2 →
  black_balls = 3 →
  (Nat.choose white_balls 1 * Nat.choose black_balls 1 : ℚ) / Nat.choose total_balls 2 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_white_ball_l308_30880


namespace NUMINAMATH_CALUDE_largest_inscribed_square_l308_30802

theorem largest_inscribed_square (outer_square_side : ℝ) 
  (h_outer_square : outer_square_side = 12) : ℝ :=
  let triangle_side := 4 * Real.sqrt 6
  let inscribed_square_side := 6 - 2 * Real.sqrt 3
  inscribed_square_side

#check largest_inscribed_square

end NUMINAMATH_CALUDE_largest_inscribed_square_l308_30802


namespace NUMINAMATH_CALUDE_committee_vote_change_l308_30858

theorem committee_vote_change (total : ℕ) (a b a' b' : ℕ) : 
  total = 300 →
  a + b = total →
  b > a →
  a' + b' = total →
  a' - b' = 3 * (b - a) →
  a' = (7 * b) / 6 →
  a' - a = 55 :=
by sorry

end NUMINAMATH_CALUDE_committee_vote_change_l308_30858


namespace NUMINAMATH_CALUDE_imaginary_part_i_2015_l308_30863

theorem imaginary_part_i_2015 : Complex.im (Complex.I ^ 2015) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_i_2015_l308_30863


namespace NUMINAMATH_CALUDE_base_difference_equals_59_l308_30846

def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

def base_6_number : List Nat := [5, 2, 3]
def base_5_number : List Nat := [1, 3, 2]

theorem base_difference_equals_59 :
  to_base_10 base_6_number 6 - to_base_10 base_5_number 5 = 59 := by
  sorry

end NUMINAMATH_CALUDE_base_difference_equals_59_l308_30846


namespace NUMINAMATH_CALUDE_system_unique_solution_l308_30895

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  Real.sqrt ((x - 6)^2 + (y - 13)^2) + Real.sqrt ((x - 18)^2 + (y - 4)^2) = 15 ∧
  (x - 2*a)^2 + (y - 4*a)^2 = 1/4

-- Define the set of a values for which the system has a unique solution
def unique_solution_set : Set ℝ :=
  {a | a = 145/44 ∨ a = 135/44 ∨ (63/20 < a ∧ a < 13/4)}

-- Theorem statement
theorem system_unique_solution (a : ℝ) :
  (∃! p : ℝ × ℝ, system p.1 p.2 a) ↔ a ∈ unique_solution_set :=
sorry

end NUMINAMATH_CALUDE_system_unique_solution_l308_30895


namespace NUMINAMATH_CALUDE_stream_speed_l308_30829

/-- Given a man's downstream and upstream speeds, calculate the speed of the stream --/
theorem stream_speed (downstream_speed upstream_speed : ℝ) 
  (h1 : downstream_speed = 10)
  (h2 : upstream_speed = 8) :
  (downstream_speed - upstream_speed) / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l308_30829


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l308_30828

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop := x^2 + y^2 + 4*x = 0

/-- The center of a circle -/
def Center : ℝ × ℝ := (-2, 0)

/-- The radius of a circle -/
def Radius : ℝ := 2

/-- Theorem: The circle described by x^2 + y^2 + 4x = 0 has center (-2, 0) and radius 2 -/
theorem circle_center_and_radius :
  (∀ x y : ℝ, CircleEquation x y ↔ (x + 2)^2 + y^2 = 4) ∧
  Center = (-2, 0) ∧
  Radius = 2 := by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l308_30828


namespace NUMINAMATH_CALUDE_women_fair_hair_percentage_l308_30809

-- Define the percentage of fair-haired employees who are women
def fair_haired_women_percentage : ℝ := 0.40

-- Define the percentage of employees who have fair hair
def fair_haired_percentage : ℝ := 0.70

-- Theorem statement
theorem women_fair_hair_percentage :
  fair_haired_women_percentage * fair_haired_percentage = 0.28 := by
  sorry

end NUMINAMATH_CALUDE_women_fair_hair_percentage_l308_30809


namespace NUMINAMATH_CALUDE_average_stickers_per_pack_l308_30882

def sticker_counts : List ℕ := [5, 8, 0, 12, 15, 20, 22, 25, 30, 35]

def num_packs : ℕ := 10

theorem average_stickers_per_pack :
  (sticker_counts.sum : ℚ) / num_packs = 17.2 := by
  sorry

end NUMINAMATH_CALUDE_average_stickers_per_pack_l308_30882


namespace NUMINAMATH_CALUDE_consecutive_product_plus_one_is_square_l308_30857

theorem consecutive_product_plus_one_is_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_plus_one_is_square_l308_30857


namespace NUMINAMATH_CALUDE_salon_extra_cans_l308_30884

/-- Represents the daily operations of a hair salon --/
structure Salon where
  customers : ℕ
  cans_bought : ℕ
  cans_per_customer : ℕ

/-- Calculates the number of extra cans of hairspray bought by the salon each day --/
def extra_cans (s : Salon) : ℕ :=
  s.cans_bought - (s.customers * s.cans_per_customer)

/-- Theorem stating that the salon buys 5 extra cans of hairspray each day --/
theorem salon_extra_cans :
  ∀ (s : Salon), s.customers = 14 ∧ s.cans_bought = 33 ∧ s.cans_per_customer = 2 →
  extra_cans s = 5 := by
  sorry

end NUMINAMATH_CALUDE_salon_extra_cans_l308_30884


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l308_30818

def vector_a : Fin 2 → ℝ := ![4, 2]
def vector_b (y : ℝ) : Fin 2 → ℝ := ![6, y]

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (∀ i, u i = k * v i)

theorem parallel_vectors_y_value :
  parallel vector_a (vector_b y) → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l308_30818


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l308_30878

/-- The circle C with center (1, 0) and radius 5 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 25}

/-- The point M -/
def M : ℝ × ℝ := (-3, 3)

/-- The proposed tangent line -/
def tangentLine (x y : ℝ) : Prop :=
  4 * x - 3 * y + 21 = 0

/-- Theorem stating that the proposed line is tangent to C at M -/
theorem tangent_line_to_circle :
  (M ∈ C) ∧
  (∃ (p : ℝ × ℝ), p ∈ C ∧ p ≠ M ∧ tangentLine p.1 p.2) ∧
  (∀ (q : ℝ × ℝ), q ∈ C → q ≠ M → ¬tangentLine q.1 q.2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l308_30878


namespace NUMINAMATH_CALUDE_hurricane_damage_calculation_l308_30800

/-- Calculates the total hurricane damage in Canadian dollars, including a recovery tax -/
theorem hurricane_damage_calculation (damage_usd : ℝ) (assets_cad : ℝ) (exchange_rate : ℝ) (tax_rate : ℝ) :
  damage_usd = 45000000 →
  assets_cad = 15000000 →
  exchange_rate = 1.25 →
  tax_rate = 0.1 →
  let damage_cad := damage_usd * exchange_rate + assets_cad
  let total_with_tax := damage_cad * (1 + tax_rate)
  total_with_tax = 78375000 := by
sorry

end NUMINAMATH_CALUDE_hurricane_damage_calculation_l308_30800


namespace NUMINAMATH_CALUDE_percentage_of_difference_l308_30812

theorem percentage_of_difference (x y : ℝ) (P : ℝ) :
  P / 100 * (x - y) = 15 / 100 * (x + y) →
  y = 14.285714285714285 / 100 * x →
  P = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_difference_l308_30812


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l308_30879

/-- The number of enchanted herbs available to the wizard. -/
def num_herbs : ℕ := 4

/-- The number of mystical crystals available to the wizard. -/
def num_crystals : ℕ := 6

/-- The number of incompatible crystals. -/
def num_incompatible_crystals : ℕ := 2

/-- The number of herbs incompatible with the incompatible crystals. -/
def num_incompatible_herbs : ℕ := 3

/-- The number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_crystals - num_incompatible_crystals * num_incompatible_herbs

theorem wizard_elixir_combinations :
  valid_combinations = 18 := by
  sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l308_30879


namespace NUMINAMATH_CALUDE_fourth_power_of_nested_square_roots_l308_30865

theorem fourth_power_of_nested_square_roots : 
  (Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))^4 = 6 + 4 * Real.sqrt (2 + Real.sqrt 2) + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_of_nested_square_roots_l308_30865


namespace NUMINAMATH_CALUDE_fraction_integer_values_fraction_values_l308_30824

theorem fraction_integer_values (n : ℕ) : 
  (∃ k : ℤ, (8 * n + 157 : ℤ) / (4 * n + 7) = k) ↔ (n = 1 ∨ n = 34) :=
by sorry

theorem fraction_values (n : ℕ) :
  n = 1 → (8 * n + 157 : ℤ) / (4 * n + 7) = 15 ∧
  n = 34 → (8 * n + 157 : ℤ) / (4 * n + 7) = 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_integer_values_fraction_values_l308_30824


namespace NUMINAMATH_CALUDE_distribute_five_students_three_classes_l308_30854

/-- The number of ways to distribute n students among k classes with a maximum of m students per class -/
def distributeStudents (n k m : ℕ) : ℕ := sorry

/-- Theorem: Distributing 5 students among 3 classes with at most 2 students per class yields 90 possibilities -/
theorem distribute_five_students_three_classes : distributeStudents 5 3 2 = 90 := by sorry

end NUMINAMATH_CALUDE_distribute_five_students_three_classes_l308_30854


namespace NUMINAMATH_CALUDE_three_digit_number_difference_l308_30823

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundreds_range : hundreds ≥ 1 ∧ hundreds ≤ 9
  tens_range : tens ≥ 0 ∧ tens ≤ 9
  ones_range : ones ≥ 0 ∧ ones ≤ 9

/-- Calculates the value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Calculates the product of digits of a three-digit number -/
def ThreeDigitNumber.digitProduct (n : ThreeDigitNumber) : Nat :=
  n.hundreds * n.tens * n.ones

theorem three_digit_number_difference (a b c : ThreeDigitNumber) :
  a.digitProduct = 64 →
  b.digitProduct = 35 →
  c.digitProduct = 81 →
  a.hundreds + b.hundreds + c.hundreds = 24 →
  a.tens + b.tens + c.tens = 12 →
  a.ones + b.ones + c.ones = 6 →
  max (max a.value b.value) c.value - min (min a.value b.value) c.value = 182 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_difference_l308_30823


namespace NUMINAMATH_CALUDE_teachers_not_adjacent_l308_30881

/-- The number of ways to arrange 2 teachers and 3 students in a row, 
    such that the teachers are not adjacent -/
def arrangement_count : ℕ := 72

/-- The number of teachers -/
def teacher_count : ℕ := 2

/-- The number of students -/
def student_count : ℕ := 3

theorem teachers_not_adjacent : 
  arrangement_count = 
    (Nat.factorial student_count) * (Nat.factorial (student_count + 1)) / 
    (Nat.factorial (student_count + 1 - teacher_count)) := by
  sorry

end NUMINAMATH_CALUDE_teachers_not_adjacent_l308_30881


namespace NUMINAMATH_CALUDE_similar_triangles_ratio_equality_l308_30861

/-- Two triangles are similar if there exists a complex number k that maps one triangle to the other -/
def similar_triangles (a b c a' b' c' : ℂ) : Prop :=
  ∃ k : ℂ, k ≠ 0 ∧ b - a = k * (b' - a') ∧ c - a = k * (c' - a')

/-- Theorem: For similar triangles abc and a'b'c' on the complex plane, 
    the ratio (b-a)/(c-a) equals (b'-a')/(c'-a') -/
theorem similar_triangles_ratio_equality 
  (a b c a' b' c' : ℂ) 
  (h : similar_triangles a b c a' b' c') 
  (h1 : c ≠ a) 
  (h2 : c' ≠ a') : 
  (b - a) / (c - a) = (b' - a') / (c' - a') := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_ratio_equality_l308_30861


namespace NUMINAMATH_CALUDE_hypotenuse_squared_of_complex_zeros_l308_30849

-- Define the polynomial P(z)
def P (z : ℂ) : ℂ := z^3 - 2*z^2 + 2*z + 4

-- State the theorem
theorem hypotenuse_squared_of_complex_zeros (a b c : ℂ) :
  P a = 0 → P b = 0 → P c = 0 →
  Complex.abs a ^ 2 + Complex.abs b ^ 2 + Complex.abs c ^ 2 = 300 →
  (a - b).re * (c - b).re + (a - b).im * (c - b).im = 0 →
  (Complex.abs (b - c)) ^ 2 = 450 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_squared_of_complex_zeros_l308_30849


namespace NUMINAMATH_CALUDE_median_sufficiency_for_top_half_l308_30870

theorem median_sufficiency_for_top_half (scores : Finset ℝ) (xiaofen_score : ℝ) :
  Finset.card scores = 12 →
  Finset.card (Finset.filter (λ x => x = xiaofen_score) scores) ≤ 1 →
  (∃ median : ℝ, Finset.card (Finset.filter (λ x => x ≤ median) scores) = 6 ∧
                 Finset.card (Finset.filter (λ x => x ≥ median) scores) = 6) →
  (xiaofen_score > median ↔ Finset.card (Finset.filter (λ x => x > xiaofen_score) scores) < 6) :=
by sorry

end NUMINAMATH_CALUDE_median_sufficiency_for_top_half_l308_30870


namespace NUMINAMATH_CALUDE_solution_system_1_solution_system_2_l308_30898

-- System (1)
theorem solution_system_1 (x y : ℝ) : 
  (4*x + 8*y = 12 ∧ 3*x - 2*y = 5) → (x = 2 ∧ y = 1/2) := by sorry

-- System (2)
theorem solution_system_2 (x y : ℝ) : 
  ((1/2)*x - (y+1)/3 = 1 ∧ 6*x + 2*y = 10) → (x = 2 ∧ y = -1) := by sorry

end NUMINAMATH_CALUDE_solution_system_1_solution_system_2_l308_30898


namespace NUMINAMATH_CALUDE_min_distinct_values_with_unique_mode_l308_30804

theorem min_distinct_values_with_unique_mode (list_size : ℕ) (mode_frequency : ℕ) 
  (h1 : list_size = 3000)
  (h2 : mode_frequency = 15) :
  (∃ (distinct_values : ℕ), 
    distinct_values ≥ 215 ∧ 
    distinct_values * (mode_frequency - 1) + mode_frequency ≥ list_size ∧
    ∀ (n : ℕ), n < 215 → n * (mode_frequency - 1) + mode_frequency < list_size) :=
by sorry

end NUMINAMATH_CALUDE_min_distinct_values_with_unique_mode_l308_30804


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_product_l308_30806

/-- Given an arithmetic sequence {a_n} where a_4 = 2, the maximum value of a_2 * a_6 is 4. -/
theorem arithmetic_sequence_max_product (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
  a 4 = 2 →                                        -- given condition
  ∃ (x : ℝ), x = a 2 * a 6 ∧ x ≤ 4 ∧ 
  ∀ (y : ℝ), y = a 2 * a 6 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_product_l308_30806


namespace NUMINAMATH_CALUDE_profit_distribution_l308_30851

/-- Represents the profit distribution problem -/
theorem profit_distribution 
  (john_investment : ℕ) (john_months : ℕ)
  (rose_investment : ℕ) (rose_months : ℕ)
  (tom_investment : ℕ) (tom_months : ℕ)
  (profit_share_diff : ℕ) :
  john_investment = 18000 →
  john_months = 12 →
  rose_investment = 12000 →
  rose_months = 9 →
  tom_investment = 9000 →
  tom_months = 8 →
  profit_share_diff = 370 →
  ∃ (total_profit : ℕ),
    total_profit = 4070 ∧
    (rose_investment * rose_months * total_profit) / 
      (john_investment * john_months + rose_investment * rose_months + tom_investment * tom_months) -
    (tom_investment * tom_months * total_profit) / 
      (john_investment * john_months + rose_investment * rose_months + tom_investment * tom_months) = 
    profit_share_diff :=
by sorry

end NUMINAMATH_CALUDE_profit_distribution_l308_30851


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l308_30864

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  1 / x + 1 / y = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l308_30864


namespace NUMINAMATH_CALUDE_product_pqr_l308_30822

theorem product_pqr (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 30)
  (h3 : 1 / p + 1 / q + 1 / r + 240 / (p * q * r) = 1) :
  p * q * r = 1080 := by
sorry

end NUMINAMATH_CALUDE_product_pqr_l308_30822


namespace NUMINAMATH_CALUDE_pipe_b_rate_is_30_l308_30853

/-- Represents the capacity of the tank in liters -/
def tank_capacity : ℕ := 900

/-- Represents the rate at which pipe A fills the tank in liters per minute -/
def pipe_a_rate : ℕ := 40

/-- Represents the rate at which pipe C drains the tank in liters per minute -/
def pipe_c_rate : ℕ := 20

/-- Represents the time taken to fill the tank in minutes -/
def fill_time : ℕ := 54

/-- Represents the duration of each pipe's operation in a cycle in minutes -/
def cycle_duration : ℕ := 3

/-- Theorem: Given the tank capacity, fill rates of pipes A and C, fill time, and cycle duration,
    the fill rate of pipe B is 30 liters per minute -/
theorem pipe_b_rate_is_30 :
  ∃ (pipe_b_rate : ℕ),
    pipe_b_rate = 30 ∧
    (fill_time / cycle_duration) * (pipe_a_rate + pipe_b_rate - pipe_c_rate) = tank_capacity :=
  sorry

end NUMINAMATH_CALUDE_pipe_b_rate_is_30_l308_30853


namespace NUMINAMATH_CALUDE_sum_in_base6_l308_30896

/-- Converts a number from base 6 to base 10 -/
def base6To10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a number from base 10 to base 6 -/
def base10To6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The sum of the given numbers in base 6 equals 1214 in base 6 -/
theorem sum_in_base6 :
  let n1 := [5, 5, 5]
  let n2 := [5, 5]
  let n3 := [5]
  let n4 := [1, 1, 1]
  let sum := base6To10 n1 + base6To10 n2 + base6To10 n3 + base6To10 n4
  base10To6 sum = [1, 2, 1, 4] :=
by sorry

end NUMINAMATH_CALUDE_sum_in_base6_l308_30896


namespace NUMINAMATH_CALUDE_hexagonal_prism_intersection_area_l308_30856

-- Define the hexagonal prism
structure HexagonalPrism :=
  (height : ℝ)
  (side_length : ℝ)

-- Define the plane
structure Plane :=
  (normal : ℝ × ℝ × ℝ)
  (point : ℝ × ℝ × ℝ)

-- Define the area of intersection
def area_of_intersection (prism : HexagonalPrism) (plane : Plane) : ℝ := sorry

-- Theorem statement
theorem hexagonal_prism_intersection_area 
  (prism : HexagonalPrism) 
  (plane : Plane) 
  (h1 : prism.height = 5) 
  (h2 : prism.side_length = 6) 
  (h3 : plane.point = (6, 0, 0)) 
  (h4 : (∃ (t : ℝ), plane.point = (-3, 3 * Real.sqrt 3, 5))) 
  (h5 : (∃ (t : ℝ), plane.point = (-3, -3 * Real.sqrt 3, 0))) : 
  area_of_intersection prism plane = 6 * Real.sqrt 399 := by sorry

end NUMINAMATH_CALUDE_hexagonal_prism_intersection_area_l308_30856


namespace NUMINAMATH_CALUDE_symmetric_points_mn_l308_30805

/-- Given two points P and Q that are symmetric about the origin, prove that mn = -2 --/
theorem symmetric_points_mn (m n : ℝ) : 
  (m - n = -3 ∧ 1 = -(m + n)) → m * n = -2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_mn_l308_30805


namespace NUMINAMATH_CALUDE_only_C_is_random_event_l308_30813

-- Define the structure for an event
structure Event where
  description : String
  is_possible : Bool
  is_certain : Bool

-- Define the events
def event_A : Event := ⟨"Scoring 105 points in a percentile-based exam", false, false⟩
def event_B : Event := ⟨"Area of a rectangle with sides a and b is ab", true, true⟩
def event_C : Event := ⟨"Taking out 2 parts from 100 parts (2 defective, 98 non-defective), both are defective", true, false⟩
def event_D : Event := ⟨"Tossing a coin, it lands with either heads or tails up", true, true⟩

-- Define what a random event is
def is_random_event (e : Event) : Prop := e.is_possible ∧ ¬e.is_certain

-- Theorem stating that only event C is a random event
theorem only_C_is_random_event : 
  ¬is_random_event event_A ∧ 
  ¬is_random_event event_B ∧ 
  is_random_event event_C ∧ 
  ¬is_random_event event_D := by sorry

end NUMINAMATH_CALUDE_only_C_is_random_event_l308_30813


namespace NUMINAMATH_CALUDE_paper_reams_for_haley_l308_30827

theorem paper_reams_for_haley (total_reams sister_reams : ℕ) 
  (h1 : total_reams = 5)
  (h2 : sister_reams = 3) :
  total_reams - sister_reams = 2 := by
  sorry

end NUMINAMATH_CALUDE_paper_reams_for_haley_l308_30827


namespace NUMINAMATH_CALUDE_article_original_price_l308_30897

/-- Given an article with a discounted price after a 24% decrease, 
    prove that its original price was Rs. 1400. -/
theorem article_original_price (discounted_price : ℝ) : 
  discounted_price = 1064 → 
  ∃ (original_price : ℝ), 
    original_price * (1 - 0.24) = discounted_price ∧ 
    original_price = 1400 := by
  sorry

end NUMINAMATH_CALUDE_article_original_price_l308_30897


namespace NUMINAMATH_CALUDE_harkamal_payment_l308_30875

/-- The total amount paid by Harkamal to the shopkeeper for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Harkamal paid 1100 to the shopkeeper -/
theorem harkamal_payment :
  total_amount 8 70 9 60 = 1100 := by
  sorry

#eval total_amount 8 70 9 60

end NUMINAMATH_CALUDE_harkamal_payment_l308_30875


namespace NUMINAMATH_CALUDE_min_value_2a6_plus_a5_l308_30831

/-- A positive geometric sequence -/
def PositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ a 1 > 0 ∧ ∀ n, a (n + 1) = q * a n

/-- The theorem stating the minimum value of 2a_6 + a_5 for a specific geometric sequence -/
theorem min_value_2a6_plus_a5 (a : ℕ → ℝ) :
  PositiveGeometricSequence a →
  (2 * a 4 + a 3 = 2 * a 2 + a 1 + 8) →
  (∀ x, 2 * a 6 + a 5 ≥ x) →
  x = 32 := by
  sorry

end NUMINAMATH_CALUDE_min_value_2a6_plus_a5_l308_30831


namespace NUMINAMATH_CALUDE_chocolate_box_count_l308_30839

theorem chocolate_box_count : ∀ (total caramels nougats truffles peanut_clusters : ℕ),
  caramels = 3 →
  nougats = 2 * caramels →
  truffles = caramels + 6 →
  peanut_clusters = total - (caramels + nougats + truffles) →
  (peanut_clusters : ℚ) / total = 64 / 100 →
  total = 50 := by sorry

end NUMINAMATH_CALUDE_chocolate_box_count_l308_30839


namespace NUMINAMATH_CALUDE_log_equality_ratio_l308_30889

theorem log_equality_ratio (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : Real.log a / Real.log 8 = Real.log b / Real.log 18 ∧ 
       Real.log a / Real.log 8 = Real.log (a + b) / Real.log 32) : 
  b / a = (3 + 2 * (Real.log 3 / Real.log 2)) / (1 + 2 * (Real.log 3 / Real.log 2) + 5) := by
sorry

end NUMINAMATH_CALUDE_log_equality_ratio_l308_30889


namespace NUMINAMATH_CALUDE_disney_banquet_attendees_l308_30826

/-- The number of people who attended a Disney banquet -/
theorem disney_banquet_attendees :
  ∀ (resident_price non_resident_price total_revenue : ℚ) 
    (num_residents : ℕ) (total_attendees : ℕ),
  resident_price = 1295/100 →
  non_resident_price = 1795/100 →
  total_revenue = 942370/100 →
  num_residents = 219 →
  total_revenue = (num_residents : ℚ) * resident_price + 
    ((total_attendees - num_residents) : ℚ) * non_resident_price →
  total_attendees = 586 := by
sorry

end NUMINAMATH_CALUDE_disney_banquet_attendees_l308_30826


namespace NUMINAMATH_CALUDE_circle_tangent_to_directrix_l308_30841

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2

-- Define the point A
def point_A : ℝ × ℝ := (0, 1)

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property that the circle passes through point A
def passes_through_A (c : Circle) : Prop :=
  let (x, y) := c.center
  (x - point_A.1)^2 + (y - point_A.2)^2 = c.radius^2

-- Define the property that the circle's center lies on the parabola
def center_on_parabola (c : Circle) : Prop :=
  let (x, y) := c.center
  parabola x y

-- Define the property that the circle is tangent to line l
def tangent_to_l (c : Circle) (l : ℝ → ℝ) : Prop :=
  let (x, y) := c.center
  (y - l x)^2 = c.radius^2

-- State the theorem
theorem circle_tangent_to_directrix :
  ∀ c : Circle,
  passes_through_A c →
  center_on_parabola c →
  ∃ l : ℝ → ℝ, (∀ x, l x = -1) ∧ tangent_to_l c l :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_directrix_l308_30841


namespace NUMINAMATH_CALUDE_sum_of_150_consecutive_integers_l308_30833

def sum_of_consecutive_integers (n : ℕ) (count : ℕ) : ℕ :=
  count * (2 * n + count - 1) / 2

theorem sum_of_150_consecutive_integers :
  ∃ (n : ℕ), sum_of_consecutive_integers n 150 = 1725225 ∧
  (∀ (m : ℕ), sum_of_consecutive_integers m 150 ≠ 3410775) ∧
  (∀ (m : ℕ), sum_of_consecutive_integers m 150 ≠ 2245600) ∧
  (∀ (m : ℕ), sum_of_consecutive_integers m 150 ≠ 1257925) ∧
  (∀ (m : ℕ), sum_of_consecutive_integers m 150 ≠ 4146950) :=
by
  sorry

#check sum_of_150_consecutive_integers

end NUMINAMATH_CALUDE_sum_of_150_consecutive_integers_l308_30833


namespace NUMINAMATH_CALUDE_perpendicular_distance_extrema_l308_30892

/-- Given two points on a line, prove that the sum of j values for (6, j) 
    that maximize and minimize squared perpendicular distances to the line is 13 -/
theorem perpendicular_distance_extrema (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : x₁ = 2 ∧ y₁ = 9) (h₂ : x₂ = 14 ∧ y₂ = 20) : 
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  let y_line := m * 6 + b
  let j_max := ⌈y_line⌉ 
  let j_min := ⌊y_line⌋
  j_max + j_min = 13 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_distance_extrema_l308_30892


namespace NUMINAMATH_CALUDE_min_candies_pile_l308_30821

theorem min_candies_pile : ∃ N : ℕ, N > 0 ∧ 
  (∃ k₁ : ℕ, N - 5 = 2 * k₁) ∧ 
  (∃ k₂ : ℕ, N - 2 = 3 * k₂) ∧ 
  (∃ k₃ : ℕ, N - 3 = 5 * k₃) ∧ 
  (∀ M : ℕ, M > 0 → 
    ((∃ m₁ : ℕ, M - 5 = 2 * m₁) ∧ 
     (∃ m₂ : ℕ, M - 2 = 3 * m₂) ∧ 
     (∃ m₃ : ℕ, M - 3 = 5 * m₃)) → M ≥ N) ∧
  N = 53 := by
sorry

end NUMINAMATH_CALUDE_min_candies_pile_l308_30821


namespace NUMINAMATH_CALUDE_final_state_values_l308_30810

/-- Represents the state of variables a, b, and c -/
structure State :=
  (a : Int) (b : Int) (c : Int)

/-- Applies the sequence of operations to the initial state -/
def applyOperations (initial : State) : State :=
  let step1 := State.mk initial.b initial.b initial.c
  let step2 := State.mk step1.a step1.c step1.b
  State.mk step2.a step2.b step2.a

/-- The theorem stating the final values after operations -/
theorem final_state_values (initial : State := State.mk 3 (-5) 8) :
  let final := applyOperations initial
  final.a = -5 ∧ final.b = 8 ∧ final.c = -5 := by
  sorry

end NUMINAMATH_CALUDE_final_state_values_l308_30810


namespace NUMINAMATH_CALUDE_accidental_vs_correct_calculation_l308_30837

theorem accidental_vs_correct_calculation (x : ℝ) : 
  7 * ((x + 24) / 5) = 70 → (5 * x + 24) / 7 = 22 := by
sorry

end NUMINAMATH_CALUDE_accidental_vs_correct_calculation_l308_30837


namespace NUMINAMATH_CALUDE_curves_with_property_P_l308_30808

-- Define the line equation
def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  k * x - y + 1 = 0

-- Define property P
def property_P (curve : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, ∃ A B : ℝ × ℝ, 
    curve A.1 A.2 ∧ curve B.1 B.2 ∧
    line_equation k A.1 A.2 ∧ line_equation k B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = k^2

-- Define the three curves
def curve1 (x y : ℝ) : Prop := y = -abs x

def curve2 (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

def curve3 (x y : ℝ) : Prop := y = (x + 1)^2

-- Theorem statement
theorem curves_with_property_P :
  ¬(property_P curve1) ∧ 
  property_P curve2 ∧ 
  property_P curve3 :=
sorry

end NUMINAMATH_CALUDE_curves_with_property_P_l308_30808


namespace NUMINAMATH_CALUDE_investment_rate_problem_l308_30848

/-- Proves that given the conditions of the investment problem, the lower interest rate is 12% --/
theorem investment_rate_problem (sum : ℝ) (time : ℝ) (high_rate : ℝ) (interest_diff : ℝ) :
  sum = 14000 →
  time = 2 →
  high_rate = 15 →
  interest_diff = 840 →
  sum * high_rate * time / 100 - sum * time * (sum * high_rate * time / 100 - interest_diff) / (sum * time) = 12 := by
  sorry


end NUMINAMATH_CALUDE_investment_rate_problem_l308_30848


namespace NUMINAMATH_CALUDE_monotonicity_and_extrema_l308_30876

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + (2*a - 1) * x - Real.log x

theorem monotonicity_and_extrema (a : ℝ) :
  (∀ x, x > 0 → f a x = a * x^2 + (2*a - 1) * x - Real.log x) →
  (a = 1/2 →
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₁ > f a x₂) ∧
    (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
    (∀ x, x > 0 → f a x ≥ 1/2) ∧
    f a 1 = 1/2) ∧
  (a ≤ 0 →
    ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂) ∧
  (a > 0 →
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/(2*a) → f a x₁ > f a x₂) ∧
    (∀ x₁ x₂, 1/(2*a) < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_and_extrema_l308_30876


namespace NUMINAMATH_CALUDE_polynomial_equality_l308_30820

theorem polynomial_equality (x : ℝ) :
  let k : ℝ → ℝ := λ x => -5*x^5 + 7*x^4 - 7*x^3 - x + 2
  5*x^5 + 3*x^3 + x + k x = 7*x^4 - 4*x^3 + 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l308_30820


namespace NUMINAMATH_CALUDE_sector_area_l308_30803

theorem sector_area (α : Real) (r : Real) (h1 : α = 150 * π / 180) (h2 : r = Real.sqrt 3) :
  (α * r^2) / 2 = 5 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l308_30803


namespace NUMINAMATH_CALUDE_intersection_values_are_four_and_fourteen_l308_30838

/-- The set of possible k values for which the graphs of |z - 4| = 3|z + 4| and |z| = k
    intersect at exactly one point in the complex plane. -/
def intersectionValues : Set ℝ :=
  {k : ℝ | ∃! z : ℂ, Complex.abs (z - 4) = 3 * Complex.abs (z + 4) ∧ Complex.abs z = k}

/-- Theorem stating that the intersection values are precisely 4 and 14. -/
theorem intersection_values_are_four_and_fourteen :
  intersectionValues = {4, 14} := by
  sorry

end NUMINAMATH_CALUDE_intersection_values_are_four_and_fourteen_l308_30838


namespace NUMINAMATH_CALUDE_inverse_of_A_l308_30860

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 7; -1, -1]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![-1/3, -7/3; 1/3, 4/3]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l308_30860


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l308_30855

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 58) 
  (h2 : throwers = 37) 
  (h3 : throwers ≤ total_players) 
  (h4 : (total_players - throwers) % 3 = 0) -- Ensures one-third of non-throwers can be left-handed
  : (throwers + ((total_players - throwers) - (total_players - throwers) / 3)) = 51 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l308_30855


namespace NUMINAMATH_CALUDE_cube_difference_l308_30811

theorem cube_difference (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 40) : 
  a^3 - b^3 = 208 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l308_30811


namespace NUMINAMATH_CALUDE_linear_and_quadratic_sequences_properties_l308_30885

def is_second_order_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, (a (n + 2) - a (n + 1)) - (a (n + 1) - a n) = d

def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

def is_local_geometric (a : ℕ → ℝ) : Prop :=
  ¬is_geometric a ∧ ∃ i j k : ℕ, i < j ∧ j < k ∧ (a j)^2 = a i * a k

theorem linear_and_quadratic_sequences_properties :
  (is_second_order_arithmetic (fun n => n : ℕ → ℝ) ∧
   is_local_geometric (fun n => n : ℕ → ℝ)) ∧
  (is_second_order_arithmetic (fun n => n^2 : ℕ → ℝ) ∧
   is_local_geometric (fun n => n^2 : ℕ → ℝ)) := by sorry

end NUMINAMATH_CALUDE_linear_and_quadratic_sequences_properties_l308_30885


namespace NUMINAMATH_CALUDE_rectangle_area_l308_30815

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l308_30815


namespace NUMINAMATH_CALUDE_square_base_exponent_l308_30872

theorem square_base_exponent (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) :
  (a^2)^(2*b) = a^b * y^b → y = a^3 := by sorry

end NUMINAMATH_CALUDE_square_base_exponent_l308_30872


namespace NUMINAMATH_CALUDE_floor_abs_negative_34_1_l308_30899

theorem floor_abs_negative_34_1 : ⌊|(-34.1 : ℝ)|⌋ = 34 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_34_1_l308_30899


namespace NUMINAMATH_CALUDE_hall_length_l308_30816

/-- The length of a hall given its breadth, number of stones, and stone dimensions -/
theorem hall_length (breadth : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ) :
  breadth = 15 ∧ 
  num_stones = 5400 ∧
  stone_length = 0.2 ∧
  stone_width = 0.5 →
  (num_stones * stone_length * stone_width) / breadth = 36 := by
sorry


end NUMINAMATH_CALUDE_hall_length_l308_30816


namespace NUMINAMATH_CALUDE_binary_1010101_equals_85_l308_30894

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1010 101₍₂₎ -/
def binary_number : List Bool := [true, false, true, false, true, false, true]

/-- Theorem stating that 1010 101₍₂₎ is equal to 85 in decimal -/
theorem binary_1010101_equals_85 : binary_to_decimal binary_number = 85 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010101_equals_85_l308_30894


namespace NUMINAMATH_CALUDE_product_eleven_one_seventeenth_thirtyfour_l308_30840

theorem product_eleven_one_seventeenth_thirtyfour : 11 * (1 / 17) * 34 = 22 := by
  sorry

end NUMINAMATH_CALUDE_product_eleven_one_seventeenth_thirtyfour_l308_30840


namespace NUMINAMATH_CALUDE_bargain_bin_books_l308_30835

theorem bargain_bin_books (initial_books sold_books added_books remaining_books : ℕ) :
  initial_books - sold_books + added_books = remaining_books →
  sold_books = 33 →
  added_books = 2 →
  remaining_books = 10 →
  initial_books = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_bargain_bin_books_l308_30835


namespace NUMINAMATH_CALUDE_angle_measure_proof_l308_30859

theorem angle_measure_proof (x : ℝ) : x + (4 * x + 5) = 90 → x = 17 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l308_30859


namespace NUMINAMATH_CALUDE_problem_solution_l308_30844

-- Define proposition p
def p : Prop := ∀ a b c : ℝ, a < b → a * c^2 < b * c^2

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 + Real.log x₀ = 0

-- Theorem to prove
theorem problem_solution : (¬p) ∧ q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l308_30844


namespace NUMINAMATH_CALUDE_inverse_of_3_mod_179_l308_30834

theorem inverse_of_3_mod_179 : ∃ x : ℕ, x < 179 ∧ (3 * x) % 179 = 1 :=
by
  use 60
  sorry

#eval (3 * 60) % 179  -- Should output 1

end NUMINAMATH_CALUDE_inverse_of_3_mod_179_l308_30834


namespace NUMINAMATH_CALUDE_hunter_has_ten_rats_l308_30850

/-- The number of rats Hunter has -/
def hunter_rats : ℕ := sorry

/-- The number of rats Elodie has -/
def elodie_rats : ℕ := hunter_rats + 30

/-- The number of rats Kenia has -/
def kenia_rats : ℕ := 3 * (hunter_rats + elodie_rats)

/-- The total number of pets -/
def total_pets : ℕ := 200

theorem hunter_has_ten_rats :
  hunter_rats + elodie_rats + kenia_rats = total_pets →
  hunter_rats = 10 := by sorry

end NUMINAMATH_CALUDE_hunter_has_ten_rats_l308_30850


namespace NUMINAMATH_CALUDE_student_average_age_l308_30877

theorem student_average_age 
  (num_students : ℕ) 
  (teacher_age : ℕ) 
  (average_with_teacher : ℚ) :
  num_students = 20 →
  teacher_age = 36 →
  average_with_teacher = 16 →
  (num_students * (average_with_teacher : ℚ) + teacher_age) / (num_students + 1 : ℚ) = average_with_teacher →
  (num_students * (average_with_teacher : ℚ) + teacher_age - teacher_age) / num_students = 15 := by
  sorry

end NUMINAMATH_CALUDE_student_average_age_l308_30877


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l308_30886

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 5th term of an arithmetic sequence equals 8, given a₃ + a₇ = 16 -/
theorem fifth_term_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h : arithmetic_sequence a) (sum_eq : a 3 + a 7 = 16) : 
  a 5 = 8 := by sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l308_30886


namespace NUMINAMATH_CALUDE_parabola_intersection_l308_30893

/-- Two parabolas with different vertices have equations y = px^2 and y = q(x-a)^2 + b, 
    where (0,0) is the vertex of the first parabola and (a,b) is the vertex of the second parabola. 
    Each vertex lies on the other parabola. -/
theorem parabola_intersection (p q a b : ℝ) (h1 : a ≠ 0) (h2 : b = p * a^2) (h3 : 0 = q * a^2 + b) : 
  p + q = 0 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l308_30893


namespace NUMINAMATH_CALUDE_anns_number_l308_30871

theorem anns_number (y : ℚ) : 5 * (3 * y + 15) = 200 → y = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_anns_number_l308_30871


namespace NUMINAMATH_CALUDE_polynomial_properties_l308_30888

def f (x : ℝ) : ℝ := x^3 - 2*x

theorem polynomial_properties :
  (∀ x y : ℚ, f x = f y → x = y) ∧
  (∃ a b : ℝ, a ≠ b ∧ f a = f b) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_properties_l308_30888


namespace NUMINAMATH_CALUDE_staircase_extension_l308_30825

/-- Calculates the number of additional toothpicks needed to extend a staircase -/
def additional_toothpicks (initial_steps : ℕ) (final_steps : ℕ) (initial_toothpicks : ℕ) (increase_rate : ℕ) : ℕ :=
  sorry

/-- Theorem: Given a 4-step staircase with 28 toothpicks and an increase rate of 3,
    33 additional toothpicks are needed to build a 6-step staircase -/
theorem staircase_extension :
  additional_toothpicks 4 6 28 3 = 33 :=
sorry

end NUMINAMATH_CALUDE_staircase_extension_l308_30825


namespace NUMINAMATH_CALUDE_altitude_to_base_l308_30845

/-- Given a triangle ABC with known sides and area, prove the altitude to base AB -/
theorem altitude_to_base (a b c area h : ℝ) : 
  a = 30 → b = 17 → c = 25 → area = 120 → 
  area = (1/2) * a * h → h = 8 := by sorry

end NUMINAMATH_CALUDE_altitude_to_base_l308_30845


namespace NUMINAMATH_CALUDE_sum_is_zero_l308_30890

theorem sum_is_zero (x y z : ℝ) 
  (h1 : x^2 = y + 2) 
  (h2 : y^2 = z + 2) 
  (h3 : z^2 = x + 2) : 
  x + y + z = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_is_zero_l308_30890


namespace NUMINAMATH_CALUDE_rebecca_egg_groups_l308_30819

/-- Given a total number of eggs and the number of eggs per group, 
    calculate the number of groups that can be created. -/
def calculate_groups (total_eggs : ℕ) (eggs_per_group : ℕ) : ℕ :=
  total_eggs / eggs_per_group

/-- Theorem stating that with 15 eggs and 5 eggs per group, 
    the number of groups is 3. -/
theorem rebecca_egg_groups : 
  calculate_groups 15 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_egg_groups_l308_30819


namespace NUMINAMATH_CALUDE_polygon_sides_when_interior_triple_exterior_l308_30869

theorem polygon_sides_when_interior_triple_exterior : ∃ n : ℕ,
  (n ≥ 3) ∧
  ((n - 2) * 180 = 3 * 360) ∧
  (∀ m : ℕ, m ≥ 3 → (m - 2) * 180 = 3 * 360 → m = n) :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_when_interior_triple_exterior_l308_30869


namespace NUMINAMATH_CALUDE_x_value_proof_l308_30832

theorem x_value_proof (x : ℚ) 
  (h1 : 8 * x^2 + 9 * x - 2 = 0) 
  (h2 : 16 * x^2 + 35 * x - 4 = 0) : 
  x = 1/8 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l308_30832


namespace NUMINAMATH_CALUDE_variance_transformation_l308_30842

-- Define a type for our dataset
def Dataset := Fin 10 → ℝ

-- Define the variance of a dataset
noncomputable def variance (data : Dataset) : ℝ := sorry

-- State the theorem
theorem variance_transformation (data : Dataset) :
  variance data = 3 →
  variance (fun i => 2 * (data i) + 3) = 12 := by sorry

end NUMINAMATH_CALUDE_variance_transformation_l308_30842


namespace NUMINAMATH_CALUDE_natural_number_pairs_l308_30817

theorem natural_number_pairs : ∀ (a b : ℕ+), 
  (∃ (k l : ℕ+), (a + 1 : ℕ) = k * b ∧ (b + 1 : ℕ) = l * a) →
  ((a = 1 ∧ b = 1) ∨ 
   (a = 1 ∧ b = 2) ∨ 
   (a = 2 ∧ b = 3) ∨ 
   (a = 2 ∧ b = 1) ∨ 
   (a = 3 ∧ b = 2)) :=
by sorry


end NUMINAMATH_CALUDE_natural_number_pairs_l308_30817


namespace NUMINAMATH_CALUDE_milk_container_problem_l308_30852

-- Define the capacity of container A
def A : ℝ := 1232

-- Define the quantity of milk in container B after initial pouring
def B : ℝ := 0.375 * A

-- Define the quantity of milk in container C after initial pouring
def C : ℝ := 0.625 * A

-- Define the amount transferred from C to B
def transfer : ℝ := 154

-- Theorem statement
theorem milk_container_problem :
  -- All milk from A was poured into B and C
  (B + C = A) ∧
  -- B had 62.5% less milk than A's capacity
  (B = 0.375 * A) ∧
  -- After transfer, B and C have equal quantities
  (B + transfer = C - transfer) →
  -- The initial quantity of milk in A was 1232 liters
  A = 1232 := by
  sorry


end NUMINAMATH_CALUDE_milk_container_problem_l308_30852


namespace NUMINAMATH_CALUDE_circle_radius_existence_l308_30891

/-- Representation of a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Representation of a point -/
def Point := ℝ × ℝ

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a point is on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop := sorry

/-- Check if two circles intersect at two points -/
def circlesIntersect (c1 c2 : Circle) : Prop := sorry

/-- Check if a point is outside a circle -/
def isOutside (p : Point) (c : Circle) : Prop := sorry

/-- Check if a circle is the circumcircle of a triangle -/
def isCircumcircle (c : Circle) (p1 p2 p3 : Point) : Prop := sorry

theorem circle_radius_existence :
  ∃! r : ℝ, r > 0 ∧
  ∃ (C1 C2 : Circle) (O X Y Z : Point),
    C1.radius = r ∧
    C1.center = O ∧
    isOnCircle O C2 ∧
    circlesIntersect C1 C2 ∧
    isOnCircle X C1 ∧ isOnCircle X C2 ∧
    isOnCircle Y C1 ∧ isOnCircle Y C2 ∧
    isOnCircle Z C2 ∧
    isOutside Z C1 ∧
    distance X Z = 15 ∧
    distance O Z = 13 ∧
    distance Y Z = 9 ∧
    isCircumcircle C2 X O Z ∧
    isCircumcircle C2 O Y Z :=
sorry

end NUMINAMATH_CALUDE_circle_radius_existence_l308_30891


namespace NUMINAMATH_CALUDE_ninas_age_l308_30883

theorem ninas_age (lisa mike nina : ℝ) 
  (h1 : (lisa + mike + nina) / 3 = 12)
  (h2 : nina - 5 = 2 * lisa)
  (h3 : mike + 2 = (lisa + 2) / 2) :
  nina = 34.6 := by
  sorry

end NUMINAMATH_CALUDE_ninas_age_l308_30883


namespace NUMINAMATH_CALUDE_range_of_a_l308_30866

/-- The line y = x + 2 intersects the x-axis at point M and the y-axis at point N. -/
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (0, 2)

/-- Point P moves on the circle (x-a)^2 + y^2 = 2, where a > 0 -/
def circle_equation (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + y^2 = 2

/-- Angle MPN is always acute -/
def angle_MPN_acute (P : ℝ × ℝ) : Prop := sorry

theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ P : ℝ × ℝ, circle_equation a P.1 P.2 → angle_MPN_acute P) →
  a > Real.sqrt 7 - 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l308_30866


namespace NUMINAMATH_CALUDE_isha_pencil_length_l308_30847

/-- The length of a pencil after sharpening, given its original length and the length sharpened off. -/
def pencil_length_after_sharpening (original_length sharpened_off : ℕ) : ℕ :=
  original_length - sharpened_off

/-- Theorem stating that a 31-inch pencil sharpened by 17 inches results in a 14-inch pencil. -/
theorem isha_pencil_length :
  pencil_length_after_sharpening 31 17 = 14 := by
  sorry

end NUMINAMATH_CALUDE_isha_pencil_length_l308_30847


namespace NUMINAMATH_CALUDE_parabola_equation_l308_30867

/-- A parabola with vertex at the origin and directrix x = 4 has the standard equation y^2 = -16x -/
theorem parabola_equation (y x : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ y^2 = -2*p*x) → -- Standard form of parabola equation
  (4 = p/2) →                        -- Condition for directrix at x = 4
  y^2 = -16*x :=                     -- Resulting equation
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l308_30867


namespace NUMINAMATH_CALUDE_tan_product_equals_fifteen_l308_30807

theorem tan_product_equals_fifteen : 
  15 * Real.tan (44 * π / 180) * Real.tan (45 * π / 180) * Real.tan (46 * π / 180) = 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_equals_fifteen_l308_30807


namespace NUMINAMATH_CALUDE_integer_sum_problem_l308_30843

theorem integer_sum_problem (x y : ℤ) : 
  x > 0 → y > 0 → x - y = 16 → x * y = 162 → x + y = 30 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l308_30843
