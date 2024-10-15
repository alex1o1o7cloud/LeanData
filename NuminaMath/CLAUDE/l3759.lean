import Mathlib

namespace NUMINAMATH_CALUDE_stamp_difference_l3759_375911

theorem stamp_difference (k a : ℕ) (h1 : k * 3 = a * 5) 
  (h2 : (k - 12) * 6 = (a + 12) * 8) : k - 12 - (a + 12) = 32 := by
  sorry

end NUMINAMATH_CALUDE_stamp_difference_l3759_375911


namespace NUMINAMATH_CALUDE_angle_C_is_30_degrees_l3759_375927

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angle measure function
def angle_measure (T : Triangle) (vertex : ℕ) : ℝ := sorry

-- Define the side length function
def side_length (T : Triangle) (side : ℕ) : ℝ := sorry

theorem angle_C_is_30_degrees (T : Triangle) :
  angle_measure T 1 = π / 4 →  -- ∠A = 45°
  side_length T 1 = Real.sqrt 2 →  -- AB = √2
  side_length T 2 = 2 →  -- BC = 2
  angle_measure T 3 = π / 6  -- ∠C = 30°
  := by sorry

end NUMINAMATH_CALUDE_angle_C_is_30_degrees_l3759_375927


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l3759_375909

theorem systematic_sampling_theorem (population : ℕ) (sample_size : ℕ) 
  (h1 : population = 1650) (h2 : sample_size = 35) :
  ∃ (exclude : ℕ) (segment_size : ℕ),
    exclude = population % sample_size ∧
    segment_size = (population - exclude) / sample_size ∧
    exclude = 5 ∧
    segment_size = 47 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l3759_375909


namespace NUMINAMATH_CALUDE_A_3_2_equals_13_l3759_375992

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2_equals_13 : A 3 2 = 13 := by sorry

end NUMINAMATH_CALUDE_A_3_2_equals_13_l3759_375992


namespace NUMINAMATH_CALUDE_leftover_floss_amount_l3759_375900

/-- Calculates the amount of leftover floss when distributing to students -/
def leftover_floss (num_students : ℕ) (floss_per_student : ℚ) (floss_per_packet : ℕ) : ℚ :=
  let total_needed : ℚ := num_students * floss_per_student
  let packets_needed : ℕ := (total_needed / floss_per_packet).ceil.toNat
  packets_needed * floss_per_packet - total_needed

/-- Theorem stating the leftover floss amount for the given problem -/
theorem leftover_floss_amount :
  leftover_floss 20 (3/2) 35 = 5 := by
sorry

end NUMINAMATH_CALUDE_leftover_floss_amount_l3759_375900


namespace NUMINAMATH_CALUDE_ninety_nine_squared_l3759_375934

theorem ninety_nine_squared : 99 * 99 = 9801 := by
  sorry

end NUMINAMATH_CALUDE_ninety_nine_squared_l3759_375934


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3759_375918

theorem arithmetic_calculation : 5 * 12 + 6 * 11 - 2 * 15 + 7 * 9 = 159 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3759_375918


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3759_375974

theorem arithmetic_expression_equality : 
  (0.15 : ℝ)^3 - (0.06 : ℝ)^3 / (0.15 : ℝ)^2 + 0.009 + (0.06 : ℝ)^2 = 0.006375 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3759_375974


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l3759_375959

theorem pizza_toppings_combinations (n : ℕ) (h : n = 8) : 
  n + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l3759_375959


namespace NUMINAMATH_CALUDE_sum_of_squares_values_l3759_375925

theorem sum_of_squares_values (x y z : ℤ) 
  (sum_eq : x + y + z = 3) 
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_values_l3759_375925


namespace NUMINAMATH_CALUDE_system_solutions_l3759_375924

def system_solution (x y : ℝ) : Prop :=
  (x > 0 ∧ y > 0) ∧
  (x^(Real.log x) * y^(Real.log y) = 243) ∧
  ((3 / Real.log x) * x * y^(Real.log y) = 1)

theorem system_solutions :
  {(x, y) : ℝ × ℝ | system_solution x y} =
  {(9, 3), (3, 9), (1/9, 1/3), (1/3, 1/9)} := by
sorry

end NUMINAMATH_CALUDE_system_solutions_l3759_375924


namespace NUMINAMATH_CALUDE_photographers_selection_l3759_375987

theorem photographers_selection (n m : ℕ) (h1 : n = 10) (h2 : m = 3) :
  Nat.choose n m = 120 := by
  sorry

end NUMINAMATH_CALUDE_photographers_selection_l3759_375987


namespace NUMINAMATH_CALUDE_expand_triple_product_l3759_375905

theorem expand_triple_product (x y z : ℝ) :
  (x + 8) * (3 * y + 12) * (2 * z + 4) =
  6 * x * y * z + 12 * x * z + 24 * y * z + 12 * x * y + 48 * x + 96 * y + 96 * z + 384 := by
  sorry

end NUMINAMATH_CALUDE_expand_triple_product_l3759_375905


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3759_375984

theorem isosceles_triangle_vertex_angle (α β γ : ℝ) : 
  -- Triangle conditions
  α + β + γ = 180 ∧
  -- Isosceles triangle condition (two angles are equal)
  (α = β ∨ β = γ ∨ α = γ) ∧
  -- One angle is 80°
  (α = 80 ∨ β = 80 ∨ γ = 80) →
  -- The vertex angle (the one that's not equal to the other two) is either 20° or 80°
  (α ≠ β → γ = 20 ∨ γ = 80) ∧
  (β ≠ γ → α = 20 ∨ α = 80) ∧
  (α ≠ γ → β = 20 ∨ β = 80) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3759_375984


namespace NUMINAMATH_CALUDE_fraction_order_l3759_375978

theorem fraction_order : (24 : ℚ) / 19 < 23 / 17 ∧ 23 / 17 < 11 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l3759_375978


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3759_375910

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 6*a*x + 9

-- Define the theorem
theorem quadratic_function_properties (a : ℝ) :
  -- Part (1)
  (f a 2 = 7 → 
    a = 1/2 ∧ 
    ∃ (x y : ℝ), x = 3/2 ∧ y = 27/4 ∧ ∀ (t : ℝ), f a t ≥ f a x) ∧
  -- Part (2)
  (f a 2 = 7 → 
    ∀ (x : ℝ), -1 ≤ x ∧ x < 3 → 27/4 ≤ f a x ∧ f a x ≤ 13) ∧
  -- Part (3)
  (∀ (x : ℝ), x ≥ 3 → ∀ (y : ℝ), y > x → f a y > f a x) →
  (∀ (x₁ x₂ : ℝ), 3*a - 2 ≤ x₁ ∧ x₁ ≤ 5 ∧ 3*a - 2 ≤ x₂ ∧ x₂ ≤ 5 → 
    f a x₁ - f a x₂ ≤ 9*a^2 + 20) →
  1/6 ≤ a ∧ a ≤ 1 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l3759_375910


namespace NUMINAMATH_CALUDE_fourth_root_cube_problem_l3759_375912

theorem fourth_root_cube_problem : 
  (((2 * Real.sqrt 2) ^ 3) ^ (1/4)) ^ 3 = 16 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_fourth_root_cube_problem_l3759_375912


namespace NUMINAMATH_CALUDE_delores_money_theorem_l3759_375960

/-- The amount of money Delores had at first, given her purchases and remaining money. -/
def delores_initial_money (computer_price printer_price headphones_price discount_rate remaining_money : ℚ) : ℚ :=
  let discounted_computer_price := computer_price * (1 - discount_rate)
  let total_spent := discounted_computer_price + printer_price + headphones_price
  total_spent + remaining_money

/-- Theorem stating that Delores had $470 at first. -/
theorem delores_money_theorem :
  delores_initial_money 400 40 60 (1/10) 10 = 470 := by
  sorry

end NUMINAMATH_CALUDE_delores_money_theorem_l3759_375960


namespace NUMINAMATH_CALUDE_exam_question_count_l3759_375994

theorem exam_question_count :
  ∀ (num_type_a num_type_b : ℕ) (time_per_a time_per_b : ℚ),
    num_type_a = 100 →
    time_per_a = 2 * time_per_b →
    num_type_a * time_per_a = 120 →
    num_type_a * time_per_a + num_type_b * time_per_b = 180 →
    num_type_a + num_type_b = 200 := by
  sorry

end NUMINAMATH_CALUDE_exam_question_count_l3759_375994


namespace NUMINAMATH_CALUDE_drug_storage_temperature_range_l3759_375995

def central_temp : ℝ := 20
def variation : ℝ := 2

def lower_limit : ℝ := central_temp - variation
def upper_limit : ℝ := central_temp + variation

theorem drug_storage_temperature_range : 
  (lower_limit = 18 ∧ upper_limit = 22) := by sorry

end NUMINAMATH_CALUDE_drug_storage_temperature_range_l3759_375995


namespace NUMINAMATH_CALUDE_smallest_staircase_steps_l3759_375996

theorem smallest_staircase_steps : ∃ (n : ℕ), 
  n > 20 ∧ 
  n % 6 = 5 ∧ 
  n % 7 = 1 ∧ 
  (∀ m : ℕ, m > 20 ∧ m % 6 = 5 ∧ m % 7 = 1 → m ≥ n) ∧
  n = 29 := by
sorry

end NUMINAMATH_CALUDE_smallest_staircase_steps_l3759_375996


namespace NUMINAMATH_CALUDE_fraction_problem_l3759_375937

theorem fraction_problem (f : ℚ) : 
  0.60 * 412.5 = f * 412.5 + 110 → f = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3759_375937


namespace NUMINAMATH_CALUDE_certain_event_red_ball_l3759_375976

/-- A bag containing colored balls -/
structure Bag where
  yellow : ℕ
  red : ℕ

/-- The probability of drawing at least one red ball when drawing two balls from the bag -/
def prob_at_least_one_red (b : Bag) : ℚ :=
  1 - (b.yellow / (b.yellow + b.red)) * ((b.yellow - 1) / (b.yellow + b.red - 1))

/-- Theorem stating that drawing at least one red ball is a certain event 
    when drawing two balls from a bag with one yellow and three red balls -/
theorem certain_event_red_ball : 
  let b : Bag := { yellow := 1, red := 3 }
  prob_at_least_one_red b = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_event_red_ball_l3759_375976


namespace NUMINAMATH_CALUDE_negative_four_squared_equals_sixteen_l3759_375942

theorem negative_four_squared_equals_sixteen :
  (-4 : ℤ) ^ 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_negative_four_squared_equals_sixteen_l3759_375942


namespace NUMINAMATH_CALUDE_prob_at_least_two_diff_fruits_l3759_375903

/-- Represents the types of fruit Joe can choose from -/
inductive Fruit
  | apple
  | orange
  | banana
  | grape

/-- The probability of choosing a specific fruit -/
def fruit_prob (f : Fruit) : ℝ :=
  match f with
  | Fruit.apple => 0.4
  | Fruit.orange => 0.3
  | Fruit.banana => 0.2
  | Fruit.grape => 0.1

/-- The probability of choosing the same fruit for all three meals -/
def same_fruit_prob : ℝ :=
  (fruit_prob Fruit.apple) ^ 3 +
  (fruit_prob Fruit.orange) ^ 3 +
  (fruit_prob Fruit.banana) ^ 3 +
  (fruit_prob Fruit.grape) ^ 3

/-- Theorem: The probability of eating at least two different kinds of fruit in a day is 0.9 -/
theorem prob_at_least_two_diff_fruits :
  1 - same_fruit_prob = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_diff_fruits_l3759_375903


namespace NUMINAMATH_CALUDE_apple_cost_price_l3759_375946

/-- The cost price of an apple given its selling price and loss ratio. -/
def cost_price (selling_price : ℚ) (loss_ratio : ℚ) : ℚ :=
  selling_price / (1 - loss_ratio)

/-- Theorem stating the cost price of an apple given specific conditions. -/
theorem apple_cost_price :
  let selling_price : ℚ := 17
  let loss_ratio : ℚ := 1/6
  cost_price selling_price loss_ratio = 20.4 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_price_l3759_375946


namespace NUMINAMATH_CALUDE_john_bought_three_sodas_l3759_375955

/-- Given a payment, cost per soda, and change received, calculate the number of sodas bought --/
def sodas_bought (payment : ℕ) (cost_per_soda : ℕ) (change : ℕ) : ℕ :=
  (payment - change) / cost_per_soda

/-- Theorem: John bought 3 sodas --/
theorem john_bought_three_sodas :
  sodas_bought 20 2 14 = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_bought_three_sodas_l3759_375955


namespace NUMINAMATH_CALUDE_difference_of_squares_l3759_375982

theorem difference_of_squares : 49^2 - 25^2 = 1776 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3759_375982


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l3759_375907

theorem cylinder_height_relationship (r1 h1 r2 h2 : ℝ) :
  r1 > 0 ∧ h1 > 0 ∧ r2 > 0 ∧ h2 > 0 →
  r2 = 1.2 * r1 →
  π * r1^2 * h1 = π * r2^2 * h2 →
  h1 = 1.44 * h2 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l3759_375907


namespace NUMINAMATH_CALUDE_share_difference_l3759_375933

theorem share_difference (total : ℕ) (a b c : ℕ) : 
  total = 120 →
  a = b + 20 →
  a < c →
  b = 20 →
  c - a = 20 := by
sorry

end NUMINAMATH_CALUDE_share_difference_l3759_375933


namespace NUMINAMATH_CALUDE_inequality_relation_l3759_375922

theorem inequality_relation (n : ℕ) (hn : n > 1) :
  (1 : ℝ) / n > Real.log ((n + 1 : ℝ) / n) ∧
  Real.log ((n + 1 : ℝ) / n) > (1 : ℝ) / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_relation_l3759_375922


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3759_375914

/-- The sum of the coordinates of the midpoint of a segment with endpoints (8, -2) and (2, 10) is 9. -/
theorem midpoint_coordinate_sum : 
  let x1 : ℝ := 8
  let y1 : ℝ := -2
  let x2 : ℝ := 2
  let y2 : ℝ := 10
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 9 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3759_375914


namespace NUMINAMATH_CALUDE_square_equality_implies_m_equals_four_l3759_375993

theorem square_equality_implies_m_equals_four (n m : ℝ) :
  (∀ x : ℝ, (x + n)^2 = x^2 + 4*x + m) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_implies_m_equals_four_l3759_375993


namespace NUMINAMATH_CALUDE_jim_victory_percentage_l3759_375901

def total_votes : ℕ := 6000
def geoff_percent : ℚ := 1/200

theorem jim_victory_percentage (laura_votes geoff_votes jim_votes : ℕ) :
  geoff_votes = (geoff_percent * total_votes).num ∧
  laura_votes = 2 * geoff_votes ∧
  jim_votes = total_votes - (laura_votes + geoff_votes) ∧
  geoff_votes + 3000 > laura_votes ∧
  geoff_votes + 3000 > jim_votes →
  (jim_votes : ℚ) / total_votes ≥ 5052 / 10000 :=
by sorry

end NUMINAMATH_CALUDE_jim_victory_percentage_l3759_375901


namespace NUMINAMATH_CALUDE_beyonce_song_count_l3759_375956

/-- The number of singles released by Beyonce -/
def singles : Nat := 5

/-- The number of albums with 15 songs -/
def albums_15 : Nat := 2

/-- The number of songs in each of the albums_15 -/
def songs_per_album_15 : Nat := 15

/-- The number of albums with 20 songs -/
def albums_20 : Nat := 1

/-- The number of songs in each of the albums_20 -/
def songs_per_album_20 : Nat := 20

/-- The total number of songs released by Beyonce -/
def total_songs : Nat := singles + albums_15 * songs_per_album_15 + albums_20 * songs_per_album_20

theorem beyonce_song_count : total_songs = 55 := by
  sorry

end NUMINAMATH_CALUDE_beyonce_song_count_l3759_375956


namespace NUMINAMATH_CALUDE_money_ratio_to_jenna_l3759_375908

/-- Represents the financial transaction scenario with John, his uncle, and Jenna --/
def john_transaction (money_from_uncle money_to_jenna groceries_cost money_remaining : ℚ) : Prop :=
  money_from_uncle - money_to_jenna - groceries_cost = money_remaining

/-- Theorem stating the ratio of money given to Jenna to money received from uncle --/
theorem money_ratio_to_jenna :
  ∃ (money_to_jenna : ℚ),
    john_transaction 100 money_to_jenna 40 35 ∧
    money_to_jenna / 100 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_money_ratio_to_jenna_l3759_375908


namespace NUMINAMATH_CALUDE_sad_girls_count_l3759_375904

theorem sad_girls_count (total_children happy_children sad_children neutral_children
                         boys girls happy_boys neutral_boys : ℕ) : 
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  boys = 19 →
  girls = 41 →
  happy_boys = 6 →
  neutral_boys = 7 →
  total_children = happy_children + sad_children + neutral_children →
  total_children = boys + girls →
  ∃ (sad_girls : ℕ), sad_girls = 4 ∧ sad_children = sad_girls + (boys - happy_boys - neutral_boys) :=
by sorry

end NUMINAMATH_CALUDE_sad_girls_count_l3759_375904


namespace NUMINAMATH_CALUDE_inequality_solution_l3759_375986

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) > 1 / 5) ↔ 
  (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3759_375986


namespace NUMINAMATH_CALUDE_problem_solution_l3759_375921

/-- The function f(x) as defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + 8*x

/-- The function g(x) as defined in the problem -/
def g (a : ℝ) (x : ℝ) : ℝ := f a x - 7*x - a^2 + 3

theorem problem_solution :
  (∀ x > -2, ∀ a > 0,
    (a = 1 → {x | f a x ≥ 2*x + 1} = {x | x ≥ 0}) ∧
    ({a | ∀ x > -2, g a x ≥ 0} = Set.Ioo 0 2)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3759_375921


namespace NUMINAMATH_CALUDE_max_value_expression_l3759_375999

theorem max_value_expression (y : ℝ) (h : y > 0) :
  (y^2 + 3 - Real.sqrt (y^4 + 9)) / y ≤ 4 * Real.sqrt 6 - 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3759_375999


namespace NUMINAMATH_CALUDE_common_tangent_intersection_l3759_375932

/-- Ellipse C₁ -/
def C₁ (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Ellipse C₂ -/
def C₂ (x y : ℝ) : Prop := (x-2)^2 + 4*y^2 = 1

/-- Common tangent to C₁ and C₂ -/
def common_tangent (x y : ℝ) : Prop :=
  ∃ (k b : ℝ), y = k*x + b ∧
    (∀ x' y', C₁ x' y' → (y' - (k*x' + b))^2 ≥ (k*(x - x'))^2) ∧
    (∀ x' y', C₂ x' y' → (y' - (k*x' + b))^2 ≥ (k*(x - x'))^2)

theorem common_tangent_intersection :
  ∃ (x y : ℝ), common_tangent x y ∧ y = 0 ∧ x = 4 :=
sorry

end NUMINAMATH_CALUDE_common_tangent_intersection_l3759_375932


namespace NUMINAMATH_CALUDE_range_of_a_l3759_375964

theorem range_of_a (a : ℝ) : 
  (∀ x, |x - 1| < 1 → x ≥ a) ∧ 
  (∃ x, x ≥ a ∧ ¬(|x - 1| < 1)) →
  a ≤ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3759_375964


namespace NUMINAMATH_CALUDE_quadratic_root_l3759_375902

/-- Given a quadratic equation 2x^2 + 3x - k = 0 where k = 44, 
    prove that 4 is one of its roots. -/
theorem quadratic_root : ∃ x : ℝ, 2 * x^2 + 3 * x - 44 = 0 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_l3759_375902


namespace NUMINAMATH_CALUDE_f_inequality_l3759_375966

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Theorem statement
theorem f_inequality (h1 : ∀ x, HasDerivAt f (f' x) x) (h2 : ∀ x, f' x < f x) : 
  f 3 < Real.exp 3 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l3759_375966


namespace NUMINAMATH_CALUDE_probability_not_snowing_l3759_375953

theorem probability_not_snowing (p_snow : ℚ) (h : p_snow = 2 / 5) : 
  1 - p_snow = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_snowing_l3759_375953


namespace NUMINAMATH_CALUDE_exam_average_l3759_375929

theorem exam_average (n1 n2 : ℕ) (avg1 avg2 : ℚ) (h1 : n1 = 15) (h2 : n2 = 10) 
  (h3 : avg1 = 80/100) (h4 : avg2 = 90/100) : 
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 84/100 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_l3759_375929


namespace NUMINAMATH_CALUDE_exists_quadrilateral_perpendicular_diagonals_not_all_natural_cubed_greater_than_squared_l3759_375979

-- Define a structure for a quadrilateral
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

-- Define a function to check if diagonals are perpendicular
def diagonalsPerpendicular (q : Quadrilateral) : Prop :=
  sorry

-- Statement 1
theorem exists_quadrilateral_perpendicular_diagonals :
  ∃ q : Quadrilateral, diagonalsPerpendicular q :=
sorry

-- Statement 2
theorem not_all_natural_cubed_greater_than_squared :
  ¬ ∀ x : ℕ, x^3 > x^2 :=
sorry

end NUMINAMATH_CALUDE_exists_quadrilateral_perpendicular_diagonals_not_all_natural_cubed_greater_than_squared_l3759_375979


namespace NUMINAMATH_CALUDE_arc_length_formula_l3759_375917

theorem arc_length_formula (r : ℝ) (θ : ℝ) (h : r = 8) (h' : θ = 5 * π / 3) :
  r * θ = 40 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_formula_l3759_375917


namespace NUMINAMATH_CALUDE_equation_solution_exists_l3759_375967

theorem equation_solution_exists : ∃ (x y z t : ℕ+), x + y + z + t = 10 ∧ z = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l3759_375967


namespace NUMINAMATH_CALUDE_balloon_difference_l3759_375958

def james_balloons : ℕ := 1222
def amy_balloons : ℕ := 513
def felix_balloons : ℕ := 687

theorem balloon_difference : james_balloons - (amy_balloons + felix_balloons) = 22 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l3759_375958


namespace NUMINAMATH_CALUDE_x_cube_plus_reciprocal_l3759_375949

theorem x_cube_plus_reciprocal (x : ℝ) (h : 11 = x^6 + 1/x^6) : x^3 + 1/x^3 = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_x_cube_plus_reciprocal_l3759_375949


namespace NUMINAMATH_CALUDE_extremum_of_f_and_range_of_a_l3759_375952

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 1 - a * Real.log x

noncomputable def g (x : ℝ) : ℝ := Real.exp x / x

theorem extremum_of_f_and_range_of_a :
  (∃ (x : ℝ), x > 0 ∧ f (1 / Real.exp 1) x = 1 / Real.exp 1 ∧
    ∀ (y : ℝ), y > 0 → f (1 / Real.exp 1) y ≥ f (1 / Real.exp 1) x) ∧
  (∀ (a : ℝ), a < 0 →
    (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 4 5 → x₂ ∈ Set.Icc 4 5 → x₁ ≠ x₂ →
      |f a x₁ - f a x₂| < |g x₁ - g x₂|) →
    4 - 3 / 4 * Real.exp 4 ≤ a) := by
  sorry

end NUMINAMATH_CALUDE_extremum_of_f_and_range_of_a_l3759_375952


namespace NUMINAMATH_CALUDE_tree_subgraph_existence_l3759_375940

-- Define a tree
def is_tree (T : SimpleGraph V) : Prop := sorry

-- Define the order of a graph
def graph_order (G : SimpleGraph V) : ℕ := sorry

-- Define the minimum degree of a graph
def min_degree (G : SimpleGraph V) : ℕ := sorry

-- Define graph isomorphism
def is_isomorphic_subgraph (T G : SimpleGraph V) : Prop := sorry

theorem tree_subgraph_existence 
  {V : Type*} (T G : SimpleGraph V) :
  is_tree T →
  min_degree G ≥ graph_order T - 1 →
  is_isomorphic_subgraph T G :=
by sorry

end NUMINAMATH_CALUDE_tree_subgraph_existence_l3759_375940


namespace NUMINAMATH_CALUDE_five_n_plus_three_composite_l3759_375920

theorem five_n_plus_three_composite (n : ℕ+) 
  (h1 : ∃ k : ℕ+, 2 * n + 1 = k^2) 
  (h2 : ∃ m : ℕ+, 3 * n + 1 = m^2) : 
  ¬(Nat.Prime (5 * n + 3)) :=
by sorry

end NUMINAMATH_CALUDE_five_n_plus_three_composite_l3759_375920


namespace NUMINAMATH_CALUDE_set_equality_l3759_375989

theorem set_equality : {x : ℕ | x > 1 ∧ x ≤ 3} = {x : ℕ | x = 2 ∨ x = 3} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l3759_375989


namespace NUMINAMATH_CALUDE_unique_integer_congruence_l3759_375983

theorem unique_integer_congruence : ∃! n : ℤ, 4 ≤ n ∧ n ≤ 8 ∧ n ≡ 7882 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_congruence_l3759_375983


namespace NUMINAMATH_CALUDE_no_solution_a_no_solution_b_no_solution_c_no_solution_d_no_solution_e_no_solution_f_no_solution_g_no_solution_h_l3759_375971

-- a) √(x+2) = -2
theorem no_solution_a : ¬∃ x : ℝ, Real.sqrt (x + 2) = -2 := by sorry

-- b) √(2x+3) + √(x+3) = 0
theorem no_solution_b : ¬∃ x : ℝ, Real.sqrt (2*x + 3) + Real.sqrt (x + 3) = 0 := by sorry

-- c) √(4-x) - √(x-6) = 2
theorem no_solution_c : ¬∃ x : ℝ, Real.sqrt (4 - x) - Real.sqrt (x - 6) = 2 := by sorry

-- d) √(-1-x) = ∛(x-5)
theorem no_solution_d : ¬∃ x : ℝ, Real.sqrt (-1 - x) = (x - 5) ^ (1/3 : ℝ) := by sorry

-- e) 5√x - 3√(-x) + 17/x = 4
theorem no_solution_e : ¬∃ x : ℝ, 5 * Real.sqrt x - 3 * Real.sqrt (-x) + 17 / x = 4 := by sorry

-- f) √(x-3) - √(x+9) = √(x-2)
theorem no_solution_f : ¬∃ x : ℝ, Real.sqrt (x - 3) - Real.sqrt (x + 9) = Real.sqrt (x - 2) := by sorry

-- g) √x + √(x+9) = 2
theorem no_solution_g : ¬∃ x : ℝ, Real.sqrt x + Real.sqrt (x + 9) = 2 := by sorry

-- h) ∛(x + 1/x) = √(-x) - 1
theorem no_solution_h : ¬∃ x : ℝ, (x + 1/x) ^ (1/3 : ℝ) = Real.sqrt (-x) - 1 := by sorry

end NUMINAMATH_CALUDE_no_solution_a_no_solution_b_no_solution_c_no_solution_d_no_solution_e_no_solution_f_no_solution_g_no_solution_h_l3759_375971


namespace NUMINAMATH_CALUDE_cos_600_degrees_l3759_375916

theorem cos_600_degrees : Real.cos (600 * π / 180) = - (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_600_degrees_l3759_375916


namespace NUMINAMATH_CALUDE_largest_square_tile_size_l3759_375913

/-- The length of the courtyard in centimeters -/
def courtyard_length : ℕ := 378

/-- The width of the courtyard in centimeters -/
def courtyard_width : ℕ := 525

/-- The size of the largest square tile in centimeters -/
def largest_tile_size : ℕ := 21

theorem largest_square_tile_size :
  (largest_tile_size ∣ courtyard_length) ∧
  (largest_tile_size ∣ courtyard_width) ∧
  ∀ n : ℕ, n > largest_tile_size →
    ¬(n ∣ courtyard_length) ∨ ¬(n ∣ courtyard_width) :=
by sorry

end NUMINAMATH_CALUDE_largest_square_tile_size_l3759_375913


namespace NUMINAMATH_CALUDE_fraction_power_product_l3759_375998

theorem fraction_power_product :
  (8 / 9 : ℚ)^3 * (5 / 3 : ℚ)^3 = 64000 / 19683 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l3759_375998


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_m_leq_neg_one_l3759_375962

/-- Set A defined by the equation x^2 + mx - y + 2 = 0 -/
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + m * p.1 - p.2 + 2 = 0}

/-- Set B defined by the equation x - y + 1 = 0 with 0 ≤ x ≤ 2 -/
def B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 1 = 0 ∧ 0 ≤ p.1 ∧ p.1 ≤ 2}

/-- The main theorem stating that A ∩ B is nonempty if and only if m ≤ -1 -/
theorem intersection_nonempty_iff_m_leq_neg_one (m : ℝ) :
  (A m ∩ B).Nonempty ↔ m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_m_leq_neg_one_l3759_375962


namespace NUMINAMATH_CALUDE_stating_equal_cost_guests_proof_l3759_375985

/-- The number of guests for which the costs of renting Caesar's and Venus Hall are equal -/
def equal_cost_guests : ℕ := 60

/-- The room rental cost for Caesar's -/
def caesars_rental : ℕ := 800

/-- The per-meal cost for Caesar's -/
def caesars_per_meal : ℕ := 30

/-- The room rental cost for Venus Hall -/
def venus_rental : ℕ := 500

/-- The per-meal cost for Venus Hall -/
def venus_per_meal : ℕ := 35

/-- 
Theorem stating that the number of guests for which the costs of renting 
Caesar's and Venus Hall are equal is 60, given the rental and per-meal costs for each venue.
-/
theorem equal_cost_guests_proof :
  caesars_rental + caesars_per_meal * equal_cost_guests = 
  venus_rental + venus_per_meal * equal_cost_guests :=
by sorry

end NUMINAMATH_CALUDE_stating_equal_cost_guests_proof_l3759_375985


namespace NUMINAMATH_CALUDE_total_sales_equals_250_l3759_375973

/-- Represents the commission rate as a percentage -/
def commission_rate : ℚ := 5 / 100

/-- Represents the commission earned in Rupees -/
def commission_earned : ℚ := 25 / 2

/-- Calculates the total sales given the commission rate and commission earned -/
def total_sales (rate : ℚ) (earned : ℚ) : ℚ := earned / rate

/-- Theorem stating that the total sales equal 250 Rupees -/
theorem total_sales_equals_250 : 
  total_sales commission_rate commission_earned = 250 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_equals_250_l3759_375973


namespace NUMINAMATH_CALUDE_radio_contest_winner_l3759_375939

theorem radio_contest_winner (n : ℕ) : 
  n > 1 ∧ 
  n < 35 ∧ 
  35 % n = 0 ∧ 
  35 % 7 = 0 ∧ 
  n ≠ 7 → 
  n = 5 := by sorry

end NUMINAMATH_CALUDE_radio_contest_winner_l3759_375939


namespace NUMINAMATH_CALUDE_triangle_abc_is_right_l3759_375972

/-- Given three points in a 2D plane, determines if they form a right triangle --/
def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
  let ab_squared := (B.1 - A.1)^2 + (B.2 - A.2)^2
  let bc_squared := (C.1 - B.1)^2 + (C.2 - B.2)^2
  let ca_squared := (A.1 - C.1)^2 + (A.2 - C.2)^2
  (ab_squared = bc_squared + ca_squared) ∨
  (bc_squared = ab_squared + ca_squared) ∨
  (ca_squared = ab_squared + bc_squared)

/-- The triangle formed by points A(5, -2), B(1, 5), and C(-1, 2) is a right triangle --/
theorem triangle_abc_is_right :
  is_right_triangle (5, -2) (1, 5) (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_is_right_l3759_375972


namespace NUMINAMATH_CALUDE_light_bulb_investigation_l3759_375945

/-- Represents the method of investigation -/
inductive InvestigationMethod
  | SamplingSurvey
  | Census

/-- Represents the characteristics of the investigation -/
structure InvestigationCharacteristics where
  largeQuantity : Bool
  destructiveTesting : Bool

/-- Determines the appropriate investigation method based on the characteristics -/
def appropriateMethod (chars : InvestigationCharacteristics) : InvestigationMethod :=
  if chars.largeQuantity && chars.destructiveTesting then
    InvestigationMethod.SamplingSurvey
  else
    InvestigationMethod.Census

/-- Theorem stating that for light bulb service life investigation with given characteristics, 
    sampling survey is the appropriate method -/
theorem light_bulb_investigation 
  (chars : InvestigationCharacteristics) 
  (h1 : chars.largeQuantity = true) 
  (h2 : chars.destructiveTesting = true) : 
  appropriateMethod chars = InvestigationMethod.SamplingSurvey := by
  sorry

end NUMINAMATH_CALUDE_light_bulb_investigation_l3759_375945


namespace NUMINAMATH_CALUDE_solution_set_and_range_l3759_375906

def f (x : ℝ) : ℝ := |2*x + 1| + 2*|x - 3|

theorem solution_set_and_range :
  (∃ (S : Set ℝ), S = {x : ℝ | f x ≤ 7*x} ∧ S = {x : ℝ | x ≥ 1}) ∧
  (∃ (M : Set ℝ), M = {m : ℝ | ∃ x : ℝ, f x = |m|} ∧ M = {m : ℝ | m ≥ 7 ∨ m ≤ -7}) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_and_range_l3759_375906


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l3759_375969

-- Define the basic structures
structure Point := (x y : ℝ)

structure Line := (a b c : ℝ)

-- Define the quadrilateral ABCD
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry
def D : Point := sorry

-- Define points E and F on CD
def E : Point := sorry
def F : Point := sorry

-- Define the circumcenters G and H
def G : Point := sorry
def H : Point := sorry

-- Define the lines AB, CD, and GH
def AB : Line := sorry
def CD : Line := sorry
def GH : Line := sorry

-- Define the property of being cyclic
def is_cyclic (p q r s : Point) : Prop := sorry

-- Define the property of lines being concurrent or parallel
def lines_concurrent_or_parallel (l m n : Line) : Prop := sorry

-- Define the property of a point lying on a line
def point_on_line (p : Point) (l : Line) : Prop := sorry

-- Define the property of being a circumcenter
def is_circumcenter (p : Point) (a b c : Point) : Prop := sorry

-- Main theorem
theorem cyclic_quadrilateral_theorem :
  (∀ (X : Point), is_cyclic A B C D) →  -- ABCD is cyclic
  (¬ (AB.a * CD.b = AB.b * CD.a)) →  -- AD is not parallel to BC
  point_on_line E CD →  -- E lies on CD
  point_on_line F CD →  -- F lies on CD
  is_circumcenter G B C E →  -- G is circumcenter of BCE
  is_circumcenter H A D F →  -- H is circumcenter of ADF
  (lines_concurrent_or_parallel AB CD GH ↔ is_cyclic A B E F) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l3759_375969


namespace NUMINAMATH_CALUDE_profit_1200_optimal_price_reduction_l3759_375977

/-- Represents the shirt sales scenario --/
structure ShirtSales where
  baseSales : ℕ := 20
  baseProfit : ℕ := 40
  salesIncrease : ℕ := 2
  priceReduction : ℚ

/-- Calculates the daily profit for a given price reduction --/
def dailyProfit (s : ShirtSales) : ℚ :=
  (s.baseProfit - s.priceReduction) * (s.baseSales + s.salesIncrease * s.priceReduction)

/-- Theorem for the price reductions that result in a daily profit of 1200 yuan --/
theorem profit_1200 (s : ShirtSales) :
  dailyProfit s = 1200 ↔ s.priceReduction = 10 ∨ s.priceReduction = 20 := by sorry

/-- Theorem for the optimal price reduction and maximum profit --/
theorem optimal_price_reduction (s : ShirtSales) :
  (∀ x, dailyProfit { s with priceReduction := x } ≤ dailyProfit { s with priceReduction := 15 }) ∧
  dailyProfit { s with priceReduction := 15 } = 1250 := by sorry

end NUMINAMATH_CALUDE_profit_1200_optimal_price_reduction_l3759_375977


namespace NUMINAMATH_CALUDE_inverse_function_intersection_implies_root_l3759_375931

theorem inverse_function_intersection_implies_root (f : ℝ → ℝ) (f_inv : ℝ → ℝ) :
  (∀ x, f_inv (f x) = x) →  -- f_inv is the inverse of f
  (∀ x, -f_inv x = f_inv (-x)) →  -- given condition about -f^(-1)(x)
  f_inv 0 = 2 →  -- intersection point (0, 2)
  f 2 = 0 :=  -- conclusion: 2 is a root of f(x) = 0
by
  sorry


end NUMINAMATH_CALUDE_inverse_function_intersection_implies_root_l3759_375931


namespace NUMINAMATH_CALUDE_election_defeat_margin_l3759_375930

theorem election_defeat_margin 
  (total_votes : ℕ) 
  (invalid_votes : ℕ) 
  (defeated_percentage : ℚ) :
  total_votes = 90830 →
  invalid_votes = 83 →
  defeated_percentage = 45/100 →
  ∃ (valid_votes winning_votes losing_votes : ℕ),
    valid_votes = total_votes - invalid_votes ∧
    losing_votes = ⌊(defeated_percentage : ℝ) * valid_votes⌋ ∧
    winning_votes = valid_votes - losing_votes ∧
    winning_votes - losing_votes = 9074 :=
by sorry

end NUMINAMATH_CALUDE_election_defeat_margin_l3759_375930


namespace NUMINAMATH_CALUDE_parabola_vertex_l3759_375943

/-- The parabola defined by the equation y = 3(x-1)^2 + 2 has vertex at (1, 2) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 3*(x-1)^2 + 2 → (1, 2) = (x, y) := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3759_375943


namespace NUMINAMATH_CALUDE_dinner_bill_problem_l3759_375928

theorem dinner_bill_problem (P : ℝ) : 
  P > 0 →  -- Assuming the price is positive
  (0.9 * P + 0.15 * P) = (0.9 * P + 0.15 * 0.9 * P + 0.51) →
  P = 34 := by
  sorry

#check dinner_bill_problem

end NUMINAMATH_CALUDE_dinner_bill_problem_l3759_375928


namespace NUMINAMATH_CALUDE_parallelepiped_net_theorem_l3759_375981

/-- Represents a parallelepiped with dimensions length, width, and height -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a net of unfolded parallelepiped -/
structure Net where
  squares : ℕ

/-- Calculates the surface area of a parallelepiped -/
def surfaceArea (p : Parallelepiped) : ℕ :=
  2 * (p.length * p.width + p.length * p.height + p.width * p.height)

/-- Unfolds a parallelepiped into a net -/
def unfold (p : Parallelepiped) : Net :=
  { squares := surfaceArea p }

/-- Removes one square from a net -/
def removeSquare (n : Net) : Net :=
  { squares := n.squares - 1 }

theorem parallelepiped_net_theorem (p : Parallelepiped) 
  (h1 : p.length = 2) (h2 : p.width = 1) (h3 : p.height = 1) :
  ∃ (n : Net), 
    (unfold p).squares = 10 ∧ 
    (removeSquare (unfold p)).squares = 9 ∧
    ∃ (valid : Bool), valid = true :=
  sorry

end NUMINAMATH_CALUDE_parallelepiped_net_theorem_l3759_375981


namespace NUMINAMATH_CALUDE_glass_volume_l3759_375948

/-- The volume of a glass given pessimist and optimist perspectives --/
theorem glass_volume (V : ℝ) (h1 : V > 0) : 
  let pessimist_empty_percent : ℝ := 0.6
  let optimist_full_percent : ℝ := 0.6
  let water_difference : ℝ := 46
  (optimist_full_percent * V) - ((1 - pessimist_empty_percent) * V) = water_difference →
  V = 230 := by
  sorry

end NUMINAMATH_CALUDE_glass_volume_l3759_375948


namespace NUMINAMATH_CALUDE_negative_square_root_operations_l3759_375991

theorem negative_square_root_operations :
  (-Real.sqrt (2^2) < 0) ∧
  ((Real.sqrt 2)^2 ≥ 0) ∧
  (Real.sqrt (2^2) ≥ 0) ∧
  (Real.sqrt ((-2)^2) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negative_square_root_operations_l3759_375991


namespace NUMINAMATH_CALUDE_circle_chord_and_area_l3759_375975

theorem circle_chord_and_area (r : ℝ) (d : ℝ) (h1 : r = 5) (h2 : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  let area := π * r^2
  chord_length = 6 ∧ area = 25 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_chord_and_area_l3759_375975


namespace NUMINAMATH_CALUDE_two_numbers_problem_l3759_375957

theorem two_numbers_problem (x y : ℝ) 
  (sum_condition : x + y = 15)
  (relation_condition : 3 * x = 5 * y - 11)
  (smaller_number : x = 7) :
  y = 8 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l3759_375957


namespace NUMINAMATH_CALUDE_paint_leftover_l3759_375990

/-- Given the following conditions:
    1. The total number of paint containers is 16
    2. There are 4 equally-sized walls
    3. One wall is not painted
    4. One container is used for the ceiling
    Prove that the number of leftover paint containers is 3. -/
theorem paint_leftover (total_containers : ℕ) (num_walls : ℕ) (unpainted_walls : ℕ) (ceiling_containers : ℕ) :
  total_containers = 16 →
  num_walls = 4 →
  unpainted_walls = 1 →
  ceiling_containers = 1 →
  total_containers - (num_walls - unpainted_walls) * (total_containers / num_walls) - ceiling_containers = 3 :=
by sorry

end NUMINAMATH_CALUDE_paint_leftover_l3759_375990


namespace NUMINAMATH_CALUDE_rabbit_jumps_l3759_375988

def N (a : ℤ) : ℕ :=
  sorry

theorem rabbit_jumps (a : ℤ) : Odd (N a) ↔ a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_jumps_l3759_375988


namespace NUMINAMATH_CALUDE_expression_evaluation_l3759_375954

theorem expression_evaluation : 2 * (5 * 9) + 3 * (4 * 11) + (2^3 * 7) + 6 * (3 * 5) = 368 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3759_375954


namespace NUMINAMATH_CALUDE_custom_operation_equality_l3759_375926

/-- Custom binary operation ⊗ -/
def otimes (a b : ℚ) : ℚ := a^2 / b

/-- Theorem statement -/
theorem custom_operation_equality : 
  (otimes (otimes 3 4) 6) - (otimes 3 (otimes 4 6)) - 1 = -113/32 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_equality_l3759_375926


namespace NUMINAMATH_CALUDE_books_on_shelves_l3759_375963

theorem books_on_shelves (total : ℕ) (bottom middle top : ℕ) : 
  bottom = (total - bottom) / 2 →
  middle = (total - middle) / 3 →
  top = 30 →
  total = bottom + middle + top →
  total = 72 := by
sorry

end NUMINAMATH_CALUDE_books_on_shelves_l3759_375963


namespace NUMINAMATH_CALUDE_smallest_number_is_negative_sqrt_5_l3759_375950

theorem smallest_number_is_negative_sqrt_5 :
  let a := (-5 : ℝ)^0
  let b := -Real.sqrt 5
  let c := -(1 / 5 : ℝ)
  let d := |(-5 : ℝ)|
  b < a ∧ b < c ∧ b < d := by sorry

end NUMINAMATH_CALUDE_smallest_number_is_negative_sqrt_5_l3759_375950


namespace NUMINAMATH_CALUDE_largest_x_value_l3759_375947

theorem largest_x_value (x : ℝ) : 
  (x / 4 + 1 / (6 * x) = 2 / 3) → 
  x ≤ (4 + Real.sqrt 10) / 3 ∧ 
  ∃ y, y / 4 + 1 / (6 * y) = 2 / 3 ∧ y = (4 + Real.sqrt 10) / 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l3759_375947


namespace NUMINAMATH_CALUDE_system_solution_equation_solution_l3759_375938

-- Problem 1: System of equations
theorem system_solution (x y : ℝ) : x + 2*y = 3 ∧ 2*x - y = 1 → x = 1 ∧ y = 1 := by
  sorry

-- Problem 2: Algebraic equation
theorem equation_solution (x : ℝ) : x ≠ 1 → (1 / (x - 1) + 2 = 5 / (1 - x)) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_equation_solution_l3759_375938


namespace NUMINAMATH_CALUDE_sequence_properties_l3759_375970

def sequence_sum (n : ℕ) : ℝ := n^2

def sequence_term (n : ℕ+) : ℝ := 2 * n.val - 1

def is_geometric_triple (a b c : ℝ) : Prop :=
  a * c = b^2

theorem sequence_properties :
  (∀ n : ℕ+, n > 1 →
    1 / Real.sqrt (sequence_sum (n.val - 1)) -
    1 / Real.sqrt (sequence_sum n.val) -
    1 / Real.sqrt (sequence_sum n.val * sequence_sum (n.val - 1)) = 0) →
  sequence_term 1 = 1 →
  (∀ n : ℕ+, sequence_term n = 2 * n.val - 1) ∧
  (∀ m t : ℕ+, 1 < m → m < t → t ≤ 100 →
    is_geometric_triple (1 / sequence_term 2) (1 / sequence_term m) (1 / sequence_term t) ↔
    (m = 5 ∧ t = 14) ∨ (m = 8 ∧ t = 38) ∨ (m = 11 ∧ t = 74)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3759_375970


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3759_375919

theorem system_solution_ratio (x y c d : ℝ) : 
  (4 * x - 3 * y = c) →
  (6 * y - 8 * x = d) →
  (d ≠ 0) →
  (∃ x y, (4 * x - 3 * y = c) ∧ (6 * y - 8 * x = d)) →
  c / d = -1 / 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3759_375919


namespace NUMINAMATH_CALUDE_other_solution_quadratic_l3759_375935

theorem other_solution_quadratic (h : 48 * (3/4)^2 + 31 = 74 * (3/4) - 16) :
  48 * (11/12)^2 + 31 = 74 * (11/12) - 16 := by
  sorry

end NUMINAMATH_CALUDE_other_solution_quadratic_l3759_375935


namespace NUMINAMATH_CALUDE_cos_sin_eq_sin_cos_third_l3759_375965

theorem cos_sin_eq_sin_cos_third (x : ℝ) :
  -π ≤ x ∧ x ≤ π ∧ Real.cos (Real.sin x) = Real.sin (Real.cos (x / 3)) → x = 0 :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_eq_sin_cos_third_l3759_375965


namespace NUMINAMATH_CALUDE_integral_exp_plus_2x_equals_e_l3759_375923

theorem integral_exp_plus_2x_equals_e : ∫ x in (0 : ℝ)..1, (Real.exp x + 2 * x) = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_exp_plus_2x_equals_e_l3759_375923


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3759_375951

theorem cubic_root_sum (a b m n p : ℝ) 
  (hm : m^3 + a*m + b = 0)
  (hn : n^3 + a*n + b = 0)
  (hp : p^3 + a*p + b = 0)
  (hmn : m ≠ n)
  (hnp : n ≠ p)
  (hmp : m ≠ p) :
  m + n + p = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3759_375951


namespace NUMINAMATH_CALUDE_equation_solution_l3759_375915

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3759_375915


namespace NUMINAMATH_CALUDE_pyramid_blocks_l3759_375997

/-- Calculates the number of blocks in a pyramid layer given the number in the layer above -/
def blocks_in_layer (blocks_above : ℕ) : ℕ := 3 * blocks_above

/-- Calculates the total number of blocks in a pyramid with the given number of layers -/
def total_blocks (layers : ℕ) : ℕ :=
  match layers with
  | 0 => 0
  | n + 1 => (blocks_in_layer^[n] 1) + total_blocks n

theorem pyramid_blocks :
  total_blocks 4 = 40 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_blocks_l3759_375997


namespace NUMINAMATH_CALUDE_condition_one_condition_two_l3759_375944

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- Theorem for condition (1)
theorem condition_one :
  ∃! a : ℝ, A a ∩ B = A a ∪ B := by sorry

-- Theorem for condition (2)
theorem condition_two :
  ∃! a : ℝ, (A a ∩ B ≠ ∅) ∧ (A a ∩ C = ∅) := by sorry

end NUMINAMATH_CALUDE_condition_one_condition_two_l3759_375944


namespace NUMINAMATH_CALUDE_second_hole_depth_l3759_375936

/-- Calculates the depth of a second hole given the conditions of two digging scenarios -/
theorem second_hole_depth
  (workers₁ : ℕ) (hours₁ : ℕ) (depth₁ : ℝ)
  (workers₂ : ℕ) (hours₂ : ℕ) :
  workers₁ = 45 →
  hours₁ = 8 →
  depth₁ = 30 →
  workers₂ = workers₁ + 45 →
  hours₂ = 6 →
  (workers₂ * hours₂ : ℝ) * (depth₁ / (workers₁ * hours₁ : ℝ)) = 45 :=
by sorry

end NUMINAMATH_CALUDE_second_hole_depth_l3759_375936


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l3759_375980

theorem no_solutions_for_equation : ¬∃ (n : ℕ+), (1 + 1 / n.val : ℝ) ^ (n.val + 1) = (1 + 1 / 1998 : ℝ) ^ 1998 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l3759_375980


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3759_375961

theorem solve_linear_equation (x : ℝ) (h : x + 1 = 4) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3759_375961


namespace NUMINAMATH_CALUDE_expression_simplification_l3759_375968

theorem expression_simplification (a b : ℝ) :
  (3 * a^5 * b^3 + a^4 * b^2) / ((-a^2 * b)^2) - (2 + a) * (2 - a) - a * (a - 5 * b) = 8 * a * b - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3759_375968


namespace NUMINAMATH_CALUDE_train_stop_time_l3759_375941

/-- Calculates the time a train stops per hour given its speeds with and without stoppages -/
theorem train_stop_time (speed_without_stoppages speed_with_stoppages : ℝ) : 
  speed_without_stoppages = 45 →
  speed_with_stoppages = 42 →
  (speed_without_stoppages - speed_with_stoppages) / speed_without_stoppages * 60 = 4 := by
  sorry

#check train_stop_time

end NUMINAMATH_CALUDE_train_stop_time_l3759_375941
