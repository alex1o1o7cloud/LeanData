import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_smaller_angle_measure_l2845_284549

/-- 
A parallelogram with one angle exceeding the other by 50 degrees has a smaller angle of 65 degrees.
-/
theorem parallelogram_smaller_angle_measure : 
  ∀ (a b : ℝ), a > 0 → b > 0 → 
  (a = b + 50) → (a + b = 180) →
  b = 65 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_smaller_angle_measure_l2845_284549


namespace NUMINAMATH_CALUDE_equation_solution_l2845_284535

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (9 / x^2) - (6 / x) + 1 = 0 → 2 / x = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2845_284535


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_300_l2845_284534

theorem least_integer_greater_than_sqrt_300 : ∃ n : ℕ, n > ⌊Real.sqrt 300⌋ ∧ ∀ m : ℕ, m > ⌊Real.sqrt 300⌋ → m ≥ n :=
by
  sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_300_l2845_284534


namespace NUMINAMATH_CALUDE_binomial_square_simplification_l2845_284513

theorem binomial_square_simplification (m n p : ℝ) :
  ¬(∃ a b, (-m - n) * (m + n) = a^2 - b^2) ∧
  (∃ a b, (-m - n) * (-m + n) = a^2 - b^2) ∧
  (∃ a b, (m * n + p) * (m * n - p) = a^2 - b^2) ∧
  (∃ a b, (0.3 * m - n) * (-n - 0.3 * m) = a^2 - b^2) :=
by sorry

end NUMINAMATH_CALUDE_binomial_square_simplification_l2845_284513


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2845_284569

theorem complex_fraction_equality : Complex.I ^ 2 = -1 → (3 + Complex.I) / (1 - Complex.I) = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2845_284569


namespace NUMINAMATH_CALUDE_frustum_slant_height_l2845_284596

theorem frustum_slant_height (r₁ r₂ : ℝ) (h : r₁ = 2 ∧ r₂ = 5) :
  let l := (π * (r₁^2 + r₂^2)) / (π * (r₁ + r₂))
  l = 29 / 7 := by
  sorry

end NUMINAMATH_CALUDE_frustum_slant_height_l2845_284596


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2845_284572

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem fifth_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_third : a 3 = -4) 
  (h_seventh : a 7 = -16) : 
  a 5 = -8 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2845_284572


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2845_284531

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 - I) / (1 - 3*I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2845_284531


namespace NUMINAMATH_CALUDE_samosa_price_is_two_l2845_284592

/-- Represents the cost of a meal at Delicious Delhi restaurant --/
structure MealCost where
  samosa_price : ℝ
  samosa_quantity : ℕ
  pakora_price : ℝ
  pakora_quantity : ℕ
  lassi_price : ℝ
  tip_percentage : ℝ
  total_with_tax : ℝ

/-- Theorem stating that the samosa price is $2 given the conditions of Hilary's meal --/
theorem samosa_price_is_two (meal : MealCost) : meal.samosa_price = 2 :=
  by
  have h1 : meal.samosa_quantity = 3 := by sorry
  have h2 : meal.pakora_price = 3 := by sorry
  have h3 : meal.pakora_quantity = 4 := by sorry
  have h4 : meal.lassi_price = 2 := by sorry
  have h5 : meal.tip_percentage = 0.25 := by sorry
  have h6 : meal.total_with_tax = 25 := by sorry
  
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_samosa_price_is_two_l2845_284592


namespace NUMINAMATH_CALUDE_angle_expression_proof_l2845_284516

theorem angle_expression_proof (α : Real) (h : Real.tan α = 2) :
  (Real.cos (α - π) - 2 * Real.cos (π / 2 + α)) / (Real.sin (α - 3 * π / 2) - Real.sin α) = -3 :=
by sorry

end NUMINAMATH_CALUDE_angle_expression_proof_l2845_284516


namespace NUMINAMATH_CALUDE_line_moved_down_three_units_l2845_284529

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Moves a linear function vertically by a given amount -/
def moveVertically (f : LinearFunction) (amount : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept - amount }

theorem line_moved_down_three_units :
  let original := LinearFunction.mk 2 5
  let moved := moveVertically original 3
  moved = LinearFunction.mk 2 2 := by
  sorry

end NUMINAMATH_CALUDE_line_moved_down_three_units_l2845_284529


namespace NUMINAMATH_CALUDE_school_election_votes_l2845_284533

/-- Represents the total number of votes in a school election --/
def total_votes : ℕ := 180

/-- Represents Brenda's share of the total votes --/
def brenda_fraction : ℚ := 4 / 15

/-- Represents the number of votes Brenda received --/
def brenda_votes : ℕ := 48

/-- Represents the number of votes Colby received --/
def colby_votes : ℕ := 35

/-- Theorem stating that given the conditions, the total number of votes is 180 --/
theorem school_election_votes : 
  (brenda_fraction * total_votes = brenda_votes) ∧ 
  (colby_votes < total_votes) ∧ 
  (brenda_votes + colby_votes < total_votes) :=
sorry


end NUMINAMATH_CALUDE_school_election_votes_l2845_284533


namespace NUMINAMATH_CALUDE_max_value_x_plus_2y_l2845_284586

theorem max_value_x_plus_2y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : 
  ∃ (max : ℝ), max = 3 ∧ ∀ (a b : ℝ), 3 * (a^2 + b^2) = a + b → a + 2*b ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_x_plus_2y_l2845_284586


namespace NUMINAMATH_CALUDE_table_tennis_probabilities_l2845_284505

-- Define the probability of scoring on a serve
def p_score : ℝ := 0.6

-- Define events
def A_i (i : Nat) : ℝ := 
  if i = 0 then (1 - p_score)^2
  else if i = 1 then 2 * p_score * (1 - p_score)
  else p_score^2

def B_i (i : Nat) : ℝ := 
  if i = 0 then p_score^2
  else if i = 1 then 2 * (1 - p_score) * p_score
  else (1 - p_score)^2

def A : ℝ := 1 - p_score

-- Define the probabilities we want to prove
def p_B : ℝ := A_i 0 * A + A_i 1 * (1 - A)
def p_C : ℝ := A_i 1 * B_i 2 + A_i 2 * B_i 1 + A_i 2 * B_i 2

theorem table_tennis_probabilities : 
  p_B = 0.352 ∧ p_C = 0.3072 := by sorry

end NUMINAMATH_CALUDE_table_tennis_probabilities_l2845_284505


namespace NUMINAMATH_CALUDE_john_running_speed_equation_l2845_284558

theorem john_running_speed_equation :
  ∃ (x : ℝ), x > 0 ∧ 6.6 * x^2 - 31.6 * x - 16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_john_running_speed_equation_l2845_284558


namespace NUMINAMATH_CALUDE_square_root_division_problem_l2845_284580

theorem square_root_division_problem : ∃ x : ℝ, (Real.sqrt 5184) / x = 4 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_root_division_problem_l2845_284580


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l2845_284524

theorem cube_sum_inequality (x y z : Real) 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) : 
  2 * (x^3 + y^3 + z^3) - (x^2 * y + y^2 * z + z^2 * x) ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l2845_284524


namespace NUMINAMATH_CALUDE_parabola_min_value_l2845_284555

theorem parabola_min_value (x y : ℝ) : 
  y^2 = 4*x → (∀ x' y' : ℝ, y'^2 = 4*x' → 1/2 * y'^2 + x'^2 + 3 ≥ 1/2 * y^2 + x^2 + 3) → 
  1/2 * y^2 + x^2 + 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_min_value_l2845_284555


namespace NUMINAMATH_CALUDE_rope_pieces_needed_l2845_284568

/-- The number of stories Tom needs to lower the rope --/
def stories : ℕ := 6

/-- The height of one story in feet --/
def story_height : ℕ := 10

/-- The length of one piece of rope in feet --/
def rope_length : ℕ := 20

/-- The percentage of rope lost when lashing pieces together --/
def rope_loss_percentage : ℚ := 1/4

/-- The number of pieces of rope Tom needs to buy --/
def pieces_needed : ℕ := 4

theorem rope_pieces_needed :
  (stories * story_height : ℚ) ≤ pieces_needed * (rope_length * (1 - rope_loss_percentage)) ∧
  (stories * story_height : ℚ) > (pieces_needed - 1) * (rope_length * (1 - rope_loss_percentage)) :=
sorry

end NUMINAMATH_CALUDE_rope_pieces_needed_l2845_284568


namespace NUMINAMATH_CALUDE_xyz_sum_root_l2845_284528

theorem xyz_sum_root (x y z : ℝ) 
  (eq1 : y + z = 14)
  (eq2 : z + x = 15)
  (eq3 : x + y = 16) :
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 134.24375 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_root_l2845_284528


namespace NUMINAMATH_CALUDE_log_equation_solution_l2845_284590

theorem log_equation_solution (b x : ℝ) (hb_pos : b > 0) (hb_neq_one : b ≠ 1) (hx_neq_one : x ≠ 1) :
  (Real.log x) / (Real.log (b^3)) + (Real.log b) / (Real.log (x^3)) = 3 →
  x = Real.exp ((9 + Real.sqrt 77) * Real.log b / 2) ∨
  x = Real.exp ((9 - Real.sqrt 77) * Real.log b / 2) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2845_284590


namespace NUMINAMATH_CALUDE_probability_red_or_green_l2845_284577

/-- The probability of drawing a red or green marble from a bag with specified marble counts. -/
theorem probability_red_or_green (red green blue yellow : ℕ) : 
  let total := red + green + blue + yellow
  (red + green : ℚ) / total = 9 / 14 :=
by
  sorry

#check probability_red_or_green 5 4 2 3

end NUMINAMATH_CALUDE_probability_red_or_green_l2845_284577


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_360_factorization_360_l2845_284591

/-- The number of perfect square factors of 360 -/
def perfect_square_factors_360 : ℕ :=
  4

theorem count_perfect_square_factors_360 :
  perfect_square_factors_360 = 4 := by
  sorry

/-- Prime factorization of 360 -/
theorem factorization_360 : 360 = 2^3 * 3^2 * 5 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_360_factorization_360_l2845_284591


namespace NUMINAMATH_CALUDE_mark_speeding_ticket_cost_l2845_284576

/-- Calculate the total cost of Mark's speeding ticket --/
def speeding_ticket_cost (base_fine : ℕ) (fine_increase_per_mph : ℕ) 
  (mark_speed : ℕ) (speed_limit : ℕ) (court_costs : ℕ) 
  (lawyer_fee_per_hour : ℕ) (lawyer_hours : ℕ) : ℕ := 
  let speed_difference := mark_speed - speed_limit
  let speed_fine := base_fine + fine_increase_per_mph * speed_difference
  let doubled_fine := 2 * speed_fine
  let total_without_lawyer := doubled_fine + court_costs
  let lawyer_cost := lawyer_fee_per_hour * lawyer_hours
  total_without_lawyer + lawyer_cost

theorem mark_speeding_ticket_cost : 
  speeding_ticket_cost 50 2 75 30 300 80 3 = 820 := by
  sorry

end NUMINAMATH_CALUDE_mark_speeding_ticket_cost_l2845_284576


namespace NUMINAMATH_CALUDE_julia_tag_game_l2845_284557

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 16

/-- The difference in the number of kids Julia played with on Monday compared to Tuesday -/
def difference : ℕ := 12

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := monday_kids - difference

theorem julia_tag_game :
  tuesday_kids = 4 :=
sorry

end NUMINAMATH_CALUDE_julia_tag_game_l2845_284557


namespace NUMINAMATH_CALUDE_equation_solutions_l2845_284544

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define our equation
def equation (x : ℝ) : Prop := (floor x : ℝ) * (x^2 + 1) = x^3

-- Theorem statement
theorem equation_solutions :
  (∀ k : ℕ, ∃! x : ℝ, k ≤ x ∧ x < k + 1 ∧ equation x) ∧
  (∀ x : ℝ, x > 0 → equation x → ¬ (∃ q : ℚ, (q : ℝ) = x)) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2845_284544


namespace NUMINAMATH_CALUDE_prob_exactly_one_correct_l2845_284597

variable (p₁ p₂ : ℝ)

-- A and B independently solve the same problem
axiom prob_A : 0 ≤ p₁ ∧ p₁ ≤ 1
axiom prob_B : 0 ≤ p₂ ∧ p₂ ≤ 1

-- The probability that exactly one person solves the problem
def prob_exactly_one : ℝ := p₁ * (1 - p₂) + p₂ * (1 - p₁)

-- Theorem stating that the probability of exactly one person solving is correct
theorem prob_exactly_one_correct :
  prob_exactly_one p₁ p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end NUMINAMATH_CALUDE_prob_exactly_one_correct_l2845_284597


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2845_284514

/-- Given a parabola and a line passing through its focus, 
    prove the value of p when the triangle area is 4 -/
theorem parabola_line_intersection (p : ℝ) : 
  let parabola := fun (x y : ℝ) => x^2 = 2*p*y
  let focus := (0, p/2)
  let line := fun (x y : ℝ) => y = Real.sqrt 3 * x + p/2
  let origin := (0, 0)
  let triangle_area (A B : ℝ × ℝ) := 
    abs ((A.1 - origin.1) * (B.2 - origin.2) - (B.1 - origin.1) * (A.2 - origin.2)) / 2
  ∃ (A B : ℝ × ℝ), 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    triangle_area A B = 4 →
    p = 2 * Real.sqrt 2 ∨ p = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2845_284514


namespace NUMINAMATH_CALUDE_winnie_lollipops_l2845_284509

theorem winnie_lollipops (total_lollipops : ℕ) (num_friends : ℕ) (h1 : total_lollipops = 400) (h2 : num_friends = 13) :
  total_lollipops - (num_friends * (total_lollipops / num_friends)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_winnie_lollipops_l2845_284509


namespace NUMINAMATH_CALUDE_min_value_and_inequality_solution_l2845_284517

theorem min_value_and_inequality_solution :
  ∃ m : ℝ,
    (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 →
      (1 / a^3 + 1 / b^3 + 1 / c^3 + 27 * a * b * c) ≥ m) ∧
    (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
      (1 / a^3 + 1 / b^3 + 1 / c^3 + 27 * a * b * c) = m) ∧
    m = 18 ∧
    (∀ x : ℝ, |x + 1| - 2 * x < m ↔ x > -19/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_solution_l2845_284517


namespace NUMINAMATH_CALUDE_shopkeeper_cloth_cost_price_l2845_284521

/-- Given a shopkeeper who sells cloth at a loss, calculate the cost price per metre. -/
theorem shopkeeper_cloth_cost_price
  (total_metres : ℕ)
  (selling_price : ℕ)
  (loss_per_metre : ℕ)
  (h1 : total_metres = 500)
  (h2 : selling_price = 18000)
  (h3 : loss_per_metre = 5) :
  (selling_price + total_metres * loss_per_metre) / total_metres = 41 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_cloth_cost_price_l2845_284521


namespace NUMINAMATH_CALUDE_sine_cosine_difference_equals_half_l2845_284520

theorem sine_cosine_difference_equals_half : 
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) - 
  Real.cos (43 * π / 180) * Real.sin (13 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_difference_equals_half_l2845_284520


namespace NUMINAMATH_CALUDE_reflection_line_sum_l2845_284501

/-- Given a line y = mx + b, if the point (-2, 0) is reflected to (6, 4) across this line, then m + b = 4 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), x = 6 ∧ y = 4 ∧ 
    (x - (-2))^2 + (y - 0)^2 = ((x + 2)/2 - (m * ((x + (-2))/2) + b))^2 + 
    ((y + 0)/2 - ((x + (-2))/(2*m) + b))^2) → 
  m + b = 4 :=
by sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l2845_284501


namespace NUMINAMATH_CALUDE_fraction_calculation_l2845_284537

theorem fraction_calculation : (1/4 + 3/8 - 7/12) / (1/24) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2845_284537


namespace NUMINAMATH_CALUDE_zoo_total_animals_l2845_284527

def zoo_animals (num_penguins : ℕ) (num_polar_bears : ℕ) : ℕ :=
  num_penguins + num_polar_bears

theorem zoo_total_animals :
  let num_penguins : ℕ := 21
  let num_polar_bears : ℕ := 2 * num_penguins
  zoo_animals num_penguins num_polar_bears = 63 := by
  sorry

end NUMINAMATH_CALUDE_zoo_total_animals_l2845_284527


namespace NUMINAMATH_CALUDE_square_park_fencing_cost_l2845_284508

/-- The total cost of fencing a square-shaped park -/
theorem square_park_fencing_cost (cost_per_side : ℕ) (h : cost_per_side = 72) : 
  cost_per_side * 4 = 288 := by
  sorry

#check square_park_fencing_cost

end NUMINAMATH_CALUDE_square_park_fencing_cost_l2845_284508


namespace NUMINAMATH_CALUDE_symmetric_linear_factor_implies_quadratic_factor_l2845_284561

-- Define a polynomial in two variables
variable (P : ℝ → ℝ → ℝ)

-- Define the property of being symmetric
def IsSymmetric (P : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, P x y = P y x

-- Define the property of having (x - y) as a factor
def HasLinearFactor (P : ℝ → ℝ → ℝ) : Prop :=
  ∃ Q : ℝ → ℝ → ℝ, ∀ x y, P x y = (x - y) * Q x y

-- Define the property of having (x - y)² as a factor
def HasQuadraticFactor (P : ℝ → ℝ → ℝ) : Prop :=
  ∃ R : ℝ → ℝ → ℝ, ∀ x y, P x y = (x - y)^2 * R x y

-- State the theorem
theorem symmetric_linear_factor_implies_quadratic_factor
  (hSymmetric : IsSymmetric P) (hLinearFactor : HasLinearFactor P) :
  HasQuadraticFactor P := by
  sorry

end NUMINAMATH_CALUDE_symmetric_linear_factor_implies_quadratic_factor_l2845_284561


namespace NUMINAMATH_CALUDE_twenty_paise_coins_l2845_284565

theorem twenty_paise_coins (total_coins : ℕ) (total_value : ℚ) : 
  total_coins = 500 ∧ 
  total_value = 105 ∧ 
  ∃ (x y : ℕ), x + y = total_coins ∧ 
                (20 : ℚ)/100 * x + (25 : ℚ)/100 * y = total_value →
  x = 400 :=
by sorry

end NUMINAMATH_CALUDE_twenty_paise_coins_l2845_284565


namespace NUMINAMATH_CALUDE_bumper_car_line_theorem_l2845_284564

/-- The number of people initially in line for bumper cars -/
def initial_people : ℕ := 9

/-- The number of people who left the line -/
def people_left : ℕ := 6

/-- The number of people who joined the line -/
def people_joined : ℕ := 3

/-- The final number of people in line -/
def final_people : ℕ := 6

/-- Theorem stating that the initial number of people satisfies the given conditions -/
theorem bumper_car_line_theorem :
  initial_people - people_left + people_joined = final_people :=
by sorry

end NUMINAMATH_CALUDE_bumper_car_line_theorem_l2845_284564


namespace NUMINAMATH_CALUDE_triangulated_rectangle_has_36_triangles_l2845_284598

/-- Represents a rectangle divided into triangles -/
structure TriangulatedRectangle where
  smallest_triangles : ℕ
  has_isosceles_triangles : Bool
  has_large_right_triangles : Bool

/-- Counts the total number of triangles in a triangulated rectangle -/
def count_triangles (rect : TriangulatedRectangle) : ℕ :=
  sorry

/-- Theorem: A rectangle divided into 16 smallest right triangles contains 36 total triangles -/
theorem triangulated_rectangle_has_36_triangles :
  ∀ (rect : TriangulatedRectangle),
    rect.smallest_triangles = 16 →
    rect.has_isosceles_triangles = true →
    rect.has_large_right_triangles = true →
    count_triangles rect = 36 :=
  sorry

end NUMINAMATH_CALUDE_triangulated_rectangle_has_36_triangles_l2845_284598


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_all_solutions_are_general_l2845_284588

/-- The differential equation -/
def diff_eq (x y : ℝ) : Prop :=
  ∃ (dx dy : ℝ), (y^3 - 2*x*y) * dx + (3*x*y^2 - x^2) * dy = 0

/-- The general solution -/
def general_solution (x y C : ℝ) : Prop :=
  y^3 * x - x^2 * y = C

/-- Theorem stating that the general solution satisfies the differential equation -/
theorem solution_satisfies_equation :
  ∀ (x y C : ℝ), general_solution x y C → diff_eq x y :=
by sorry

/-- Theorem stating that any solution to the differential equation is of the form of the general solution -/
theorem all_solutions_are_general :
  ∀ (x y : ℝ), diff_eq x y → ∃ (C : ℝ), general_solution x y C :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_all_solutions_are_general_l2845_284588


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2845_284551

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 8 → b = 15 → c^2 = a^2 + b^2 → c = 17 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2845_284551


namespace NUMINAMATH_CALUDE_prime_condition_l2845_284541

theorem prime_condition (p : ℕ) : 
  Nat.Prime p ∧ Nat.Prime (p^4 - 3*p^2 + 9) → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_condition_l2845_284541


namespace NUMINAMATH_CALUDE_inequality_solution_l2845_284548

theorem inequality_solution (x : ℝ) : 
  (x ≠ 1 ∧ x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5) →
  (1 / (x - 1) - 5 / (x - 2) + 5 / (x - 3) - 1 / (x - 5) < 1 / 24) ↔ 
  (x < -2 ∨ (1 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ 5 < x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2845_284548


namespace NUMINAMATH_CALUDE_sean_net_profit_l2845_284593

/-- Represents the pricing tiers for patches --/
inductive PricingTier
  | small
  | medium
  | large
  | xlarge

/-- Calculates the price per patch based on the pricing tier --/
def price_per_patch (tier : PricingTier) : ℚ :=
  match tier with
  | .small => 12
  | .medium => 11.5
  | .large => 11
  | .xlarge => 10.5

/-- Represents a sale of patches --/
structure Sale :=
  (quantity : ℕ)
  (customers : ℕ)
  (tier : PricingTier)

/-- Calculates the total cost for ordering patches --/
def total_cost (patches : ℕ) : ℚ :=
  let units := (patches + 99) / 100  -- Round up to nearest 100
  1.25 * patches + 20 * units

/-- Calculates the revenue from a sale --/
def sale_revenue (sale : Sale) : ℚ :=
  sale.quantity * sale.customers * price_per_patch sale.tier

/-- Calculates the total revenue from all sales --/
def total_revenue (sales : List Sale) : ℚ :=
  sales.map sale_revenue |> List.sum

/-- The main theorem stating Sean's net profit --/
theorem sean_net_profit (sales : List Sale) 
  (h_sales : sales = [
    {quantity := 15, customers := 5, tier := .small},
    {quantity := 50, customers := 2, tier := .medium},
    {quantity := 25, customers := 1, tier := .large}
  ]) : 
  total_revenue sales - total_cost (sales.map (λ s => s.quantity * s.customers) |> List.sum) = 2035 := by
  sorry


end NUMINAMATH_CALUDE_sean_net_profit_l2845_284593


namespace NUMINAMATH_CALUDE_timothy_chickens_l2845_284578

def land_cost : ℕ := 30 * 20
def house_cost : ℕ := 120000
def cows_cost : ℕ := 20 * 1000
def solar_panel_cost : ℕ := 6 * 100 + 6000
def chicken_price : ℕ := 5
def total_cost : ℕ := 147700

theorem timothy_chickens :
  ∃ (num_chickens : ℕ),
    land_cost + house_cost + cows_cost + solar_panel_cost + num_chickens * chicken_price = total_cost ∧
    num_chickens = 100 :=
by sorry

end NUMINAMATH_CALUDE_timothy_chickens_l2845_284578


namespace NUMINAMATH_CALUDE_marbles_left_l2845_284530

def marbles_in_box : ℕ := 50
def white_marbles : ℕ := 20

def red_blue_marbles : ℕ := marbles_in_box - white_marbles
def blue_marbles : ℕ := red_blue_marbles / 2
def red_marbles : ℕ := red_blue_marbles / 2

def marbles_removed : ℕ := 2 * (white_marbles - blue_marbles)

theorem marbles_left : marbles_in_box - marbles_removed = 40 := by
  sorry

end NUMINAMATH_CALUDE_marbles_left_l2845_284530


namespace NUMINAMATH_CALUDE_triangle_is_right_angle_l2845_284553

theorem triangle_is_right_angle (u : ℝ) 
  (h1 : 0 < 3*u - 2) 
  (h2 : 0 < 3*u + 2) 
  (h3 : 0 < 6*u) : 
  (3*u - 2) + (3*u + 2) = 6*u := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_right_angle_l2845_284553


namespace NUMINAMATH_CALUDE_fibonacci_periodicity_l2845_284511

-- Define p-arithmetic system
class PArithmetic (p : ℕ) where
  sqrt5_extractable : ∃ x, x^2 = 5
  fermat_little : ∀ a : ℤ, a ≠ 0 → a^(p-1) ≡ 1 [ZMOD p]

-- Define Fibonacci sequence
def fibonacci (v₀ v₁ : ℤ) : ℕ → ℤ
| 0 => v₀
| 1 => v₁
| (n+2) => fibonacci v₀ v₁ n + fibonacci v₀ v₁ (n+1)

-- Theorem statement
theorem fibonacci_periodicity {p : ℕ} [PArithmetic p] (v₀ v₁ : ℤ) :
  ∀ k : ℕ, fibonacci v₀ v₁ (k + p - 1) = fibonacci v₀ v₁ k :=
sorry

end NUMINAMATH_CALUDE_fibonacci_periodicity_l2845_284511


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_range_l2845_284525

open Real

theorem tangent_slope_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, (a / x - 2 * (x - 1)) > 1) →
  a ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_range_l2845_284525


namespace NUMINAMATH_CALUDE_impossible_all_defective_l2845_284503

theorem impossible_all_defective (total : ℕ) (defective : ℕ) (selected : ℕ)
  (h1 : total = 25)
  (h2 : defective = 2)
  (h3 : selected = 3)
  (h4 : defective < total)
  (h5 : selected ≤ total) :
  Nat.choose defective selected / Nat.choose total selected = 0 :=
by sorry

end NUMINAMATH_CALUDE_impossible_all_defective_l2845_284503


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l2845_284599

def original_price : ℝ := 77.95
def sale_price : ℝ := 59.95

theorem price_decrease_percentage :
  let decrease := original_price - sale_price
  let percentage_decrease := (decrease / original_price) * 100
  ∃ ε > 0, abs (percentage_decrease - 23.08) < ε := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l2845_284599


namespace NUMINAMATH_CALUDE_ratio_of_segments_l2845_284510

/-- Given five consecutive points on a straight line, prove the ratio of two segments --/
theorem ratio_of_segments (a b c d e : ℝ) : 
  (b < c) ∧ (c < d) ∧  -- Consecutive points
  (e - d = 8) ∧        -- de = 8
  (b - a = 5) ∧        -- ab = 5
  (c - a = 11) ∧       -- ac = 11
  (e - a = 21)         -- ae = 21
  → (c - b) / (d - c) = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l2845_284510


namespace NUMINAMATH_CALUDE_min_value_of_fraction_sum_l2845_284539

theorem min_value_of_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (2 / x + 1 / y) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_sum_l2845_284539


namespace NUMINAMATH_CALUDE_vector_combination_vectors_parallel_l2845_284547

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![-1, 2]
def c : Fin 2 → ℝ := ![4, 1]

/-- The theorem states that a = (5/9)b + (8/9)c -/
theorem vector_combination : a = (5/9 • b) + (8/9 • c) := by sorry

/-- Helper function to check if two vectors are parallel -/
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (t : ℝ), v = t • w ∨ w = t • v

/-- The theorem states that (a + kc) is parallel to (2b - a) when k = -16/13 -/
theorem vectors_parallel : are_parallel (a + (-16/13 • c)) (2 • b - a) := by sorry

end NUMINAMATH_CALUDE_vector_combination_vectors_parallel_l2845_284547


namespace NUMINAMATH_CALUDE_largest_prime_with_2023_digits_p_is_prime_p_has_2023_digits_smallest_k_divisible_by_30_l2845_284554

/-- The largest prime with 2023 digits -/
def p : ℕ := sorry

theorem largest_prime_with_2023_digits (q : ℕ) (h : q > p) : ¬ Prime q := sorry

theorem p_is_prime : Prime p := sorry

theorem p_has_2023_digits : (Nat.digits 10 p).length = 2023 := sorry

theorem smallest_k_divisible_by_30 : 
  ∃ k : ℕ, k > 0 ∧ 30 ∣ (p^3 - k) ∧ ∀ m : ℕ, 0 < m ∧ m < k → ¬(30 ∣ (p^3 - m)) :=
sorry

end NUMINAMATH_CALUDE_largest_prime_with_2023_digits_p_is_prime_p_has_2023_digits_smallest_k_divisible_by_30_l2845_284554


namespace NUMINAMATH_CALUDE_phone_repair_cost_l2845_284587

theorem phone_repair_cost (laptop_cost computer_cost : ℕ) 
  (phone_repairs laptop_repairs computer_repairs : ℕ) (total_earnings : ℕ) :
  laptop_cost = 15 →
  computer_cost = 18 →
  phone_repairs = 5 →
  laptop_repairs = 2 →
  computer_repairs = 2 →
  total_earnings = 121 →
  ∃ (phone_cost : ℕ), 
    phone_cost * phone_repairs + 
    laptop_cost * laptop_repairs + 
    computer_cost * computer_repairs = total_earnings ∧
    phone_cost = 11 :=
by sorry

end NUMINAMATH_CALUDE_phone_repair_cost_l2845_284587


namespace NUMINAMATH_CALUDE_marbles_distribution_l2845_284512

theorem marbles_distribution (total_marbles : ℕ) (kept_marbles : ℕ) (best_friends : ℕ) (marbles_per_best_friend : ℕ) (neighborhood_friends : ℕ) :
  total_marbles = 1125 →
  kept_marbles = 100 →
  best_friends = 2 →
  marbles_per_best_friend = 50 →
  neighborhood_friends = 7 →
  (total_marbles - kept_marbles - best_friends * marbles_per_best_friend) / neighborhood_friends = 132 :=
by sorry

end NUMINAMATH_CALUDE_marbles_distribution_l2845_284512


namespace NUMINAMATH_CALUDE_second_denomination_value_l2845_284536

theorem second_denomination_value (total_amount : ℕ) (total_notes : ℕ) : 
  total_amount = 400 →
  total_notes = 75 →
  ∃ (x : ℕ), 
    x > 1 ∧ 
    x < 10 ∧
    (total_notes / 3) * (1 + x + 10) = total_amount →
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_second_denomination_value_l2845_284536


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l2845_284559

/-- Given a man who is 24 years older than his son, and the son's present age is 22,
    prove that it will take 2 years for the man's age to be twice the age of his son. -/
theorem mans_age_twice_sons (man_age son_age future_years : ℕ) : 
  man_age = son_age + 24 →
  son_age = 22 →
  future_years = 2 →
  (man_age + future_years) = 2 * (son_age + future_years) :=
by sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l2845_284559


namespace NUMINAMATH_CALUDE_polynomial_degree_l2845_284502

/-- The degree of the polynomial 3 + 7x^2 + (1/2)x^5 - 10x + 11 is 5 -/
theorem polynomial_degree : 
  let p : Polynomial ℚ := 3 + 7 * X^2 + (1/2) * X^5 - 10 * X + 11
  Polynomial.degree p = 5 := by sorry

end NUMINAMATH_CALUDE_polynomial_degree_l2845_284502


namespace NUMINAMATH_CALUDE_gayle_bicycle_ride_l2845_284571

/-- Gayle's bicycle ride problem -/
theorem gayle_bicycle_ride 
  (sunny_speed : ℝ) 
  (rainy_speed : ℝ) 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (h1 : sunny_speed = 40)
  (h2 : rainy_speed = 25)
  (h3 : total_distance = 20)
  (h4 : total_time = 50/60) -- Convert 50 minutes to hours
  : ∃ (rainy_time : ℝ), 
    rainy_time = 32/60 ∧ -- Convert 32 minutes to hours
    rainy_time * rainy_speed + (total_time - rainy_time) * sunny_speed = total_distance :=
by
  sorry


end NUMINAMATH_CALUDE_gayle_bicycle_ride_l2845_284571


namespace NUMINAMATH_CALUDE_mayoral_election_votes_l2845_284518

theorem mayoral_election_votes (z : ℕ) (hz : z = 25000) : ∃ x y : ℕ,
  y = z - (2 * z / 5) ∧
  x = y + (y / 2) ∧
  x = 22500 :=
by
  sorry

end NUMINAMATH_CALUDE_mayoral_election_votes_l2845_284518


namespace NUMINAMATH_CALUDE_max_non_overlapping_ge_min_covering_l2845_284519

/-- A polygon in a 2D plane -/
structure Polygon where
  -- Add necessary fields for a polygon

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a circle's center is inside a polygon -/
def Circle.centerInside (c : Circle) (p : Polygon) : Prop :=
  sorry

/-- Checks if two circles are non-overlapping -/
def Circle.nonOverlapping (c1 c2 : Circle) : Prop :=
  sorry

/-- Checks if a set of circles covers a polygon -/
def covers (circles : Set Circle) (p : Polygon) : Prop :=
  sorry

/-- The maximum number of non-overlapping circles of diameter 1 with centers inside the polygon -/
def maxNonOverlappingCircles (p : Polygon) : ℕ :=
  sorry

/-- The minimum number of circles of radius 1 that can cover the polygon -/
def minCoveringCircles (p : Polygon) : ℕ :=
  sorry

/-- Theorem: The maximum number of non-overlapping circles of diameter 1 with centers inside a polygon
    is greater than or equal to the minimum number of circles of radius 1 needed to cover the polygon -/
theorem max_non_overlapping_ge_min_covering (p : Polygon) :
  maxNonOverlappingCircles p ≥ minCoveringCircles p :=
sorry

end NUMINAMATH_CALUDE_max_non_overlapping_ge_min_covering_l2845_284519


namespace NUMINAMATH_CALUDE_three_from_fifteen_combination_l2845_284573

theorem three_from_fifteen_combination : (Nat.choose 15 3) = 455 := by sorry

end NUMINAMATH_CALUDE_three_from_fifteen_combination_l2845_284573


namespace NUMINAMATH_CALUDE_company_stores_l2845_284506

theorem company_stores (total_uniforms : ℕ) (uniforms_per_store : ℕ) (h1 : total_uniforms = 121) (h2 : uniforms_per_store = 4) :
  (total_uniforms / uniforms_per_store : ℕ) = 30 :=
by sorry

end NUMINAMATH_CALUDE_company_stores_l2845_284506


namespace NUMINAMATH_CALUDE_square_number_ratio_l2845_284574

theorem square_number_ratio (k : ℕ) (h : k ≥ 2) :
  ∀ a b : ℕ, a ≠ 0 → b ≠ 0 →
  (a^2 + b^2) / (a * b + 1) = k^2 ↔ a = k ∧ b = k^3 := by
sorry

end NUMINAMATH_CALUDE_square_number_ratio_l2845_284574


namespace NUMINAMATH_CALUDE_circle_max_area_l2845_284570

/-- Given a circle equation with parameter m, prove that when the area is maximum, 
    the standard equation of the circle is (x-1)^2 + (y+3)^2 = 1 -/
theorem circle_max_area (x y m : ℝ) : 
  (∃ r, x^2 + y^2 - 2*x + 2*m*y + 2*m^2 - 6*m + 9 = 0 ↔ (x-1)^2 + (y+m)^2 = r^2) →
  (∀ m', ∃ r', x^2 + y^2 - 2*x + 2*m'*y + 2*m'^2 - 6*m' + 9 = 0 → 
    (x-1)^2 + (y+m')^2 = r'^2 ∧ r'^2 ≤ 1) →
  (x-1)^2 + (y+3)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_circle_max_area_l2845_284570


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2845_284515

/-- Given a complex number z satisfying zi = 2 + i, prove that the real part of z is positive
    and the imaginary part of z is negative. -/
theorem z_in_fourth_quadrant (z : ℂ) (h : z * Complex.I = 2 + Complex.I) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2845_284515


namespace NUMINAMATH_CALUDE_tiles_needed_for_room_l2845_284522

/-- Proves that the number of 3-inch by 5-inch tiles needed to cover a 10-foot by 15-foot room is 1440 -/
theorem tiles_needed_for_room : 
  let room_length : ℚ := 10
  let room_width : ℚ := 15
  let tile_length : ℚ := 3 / 12  -- 3 inches in feet
  let tile_width : ℚ := 5 / 12   -- 5 inches in feet
  let room_area := room_length * room_width
  let tile_area := tile_length * tile_width
  let tiles_needed := room_area / tile_area
  ⌈tiles_needed⌉ = 1440 := by sorry

end NUMINAMATH_CALUDE_tiles_needed_for_room_l2845_284522


namespace NUMINAMATH_CALUDE_fixed_point_coordinates_l2845_284567

/-- Given that for any real number k, the line (3+k)x + (1-2k)y + 1 + 5k = 0
    passes through a fixed point A, prove that the coordinates of A are (-1, 2). -/
theorem fixed_point_coordinates (A : ℝ × ℝ) :
  (∀ k : ℝ, (3 + k) * A.1 + (1 - 2*k) * A.2 + 1 + 5*k = 0) →
  A = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_coordinates_l2845_284567


namespace NUMINAMATH_CALUDE_rectangle_area_is_72_l2845_284595

-- Define the radius of the circles
def circle_radius : ℝ := 3

-- Define the rectangle
structure Rectangle where
  length : ℝ
  width : ℝ

-- Define the property that the circles touch each other and the rectangle sides
def circles_touch_rectangle_and_each_other (r : Rectangle) : Prop :=
  r.length = 4 * circle_radius ∧ r.width = 2 * circle_radius

-- Theorem statement
theorem rectangle_area_is_72 (r : Rectangle) 
  (h : circles_touch_rectangle_and_each_other r) : r.length * r.width = 72 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_72_l2845_284595


namespace NUMINAMATH_CALUDE_divisor_and_equation_l2845_284526

theorem divisor_and_equation (k : ℕ) : 
  (∃ n : ℕ, n * (18^k) = 1) → (6^k - k^6 = 1 ↔ k = 0) := by
sorry

end NUMINAMATH_CALUDE_divisor_and_equation_l2845_284526


namespace NUMINAMATH_CALUDE_geometric_solid_surface_area_l2845_284584

/-- Given a geometric solid that is a quarter of a cylinder with height 2,
    base area π, and radius 2, prove that its surface area is 8 + 4π. -/
theorem geometric_solid_surface_area
  (h : ℝ) (base_area : ℝ) (radius : ℝ) :
  h = 2 →
  base_area = π →
  radius = 2 →
  (2 * base_area + 2 * radius * h + (1/4) * 2 * π * radius * h) = 8 + 4 * π :=
by sorry

end NUMINAMATH_CALUDE_geometric_solid_surface_area_l2845_284584


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2845_284563

/-- The original expression as a function of x and square -/
def original_expr (x : ℝ) (square : ℝ) : ℝ :=
  (3 - 2*x^2 - 5*x) - (square*x^2 + 3*x - 4)

/-- The simplified expression as a function of x and square -/
def simplified_expr (x : ℝ) (square : ℝ) : ℝ :=
  (-2 - square)*x^2 - 8*x + 7

theorem expression_simplification_and_evaluation :
  ∀ (x : ℝ) (square : ℝ),
  /- 1. The simplified form is correct -/
  original_expr x square = simplified_expr x square ∧
  /- 2. When x=-2 and square=-2, the expression evaluates to -17 -/
  original_expr (-2) (-2) = -17 ∧
  /- 3. The value of square that eliminates the quadratic term is -2 -/
  ∃ (square : ℝ), (-2 - square) = 0 ∧ square = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2845_284563


namespace NUMINAMATH_CALUDE_S_is_open_line_segment_l2845_284523

-- Define the set of points satisfying the conditions
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = 1 ∧ p.1^2 + p.2^2 < 25}

-- Theorem statement
theorem S_is_open_line_segment :
  ∃ (a b : ℝ × ℝ), a ≠ b ∧
    S = {p : ℝ × ℝ | ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ p = (1 - t) • a + t • b} :=
sorry

end NUMINAMATH_CALUDE_S_is_open_line_segment_l2845_284523


namespace NUMINAMATH_CALUDE_yanni_found_money_l2845_284583

/-- The amount of money Yanni found at the mall -/
def money_found (initial_money mother_gave toy_cost money_left : ℚ) : ℚ :=
  (toy_cost + money_left) - (initial_money + mother_gave)

/-- Theorem stating how much money Yanni found at the mall -/
theorem yanni_found_money : 
  money_found 0.85 0.40 1.60 0.15 = 0.50 := by sorry

end NUMINAMATH_CALUDE_yanni_found_money_l2845_284583


namespace NUMINAMATH_CALUDE_white_line_length_l2845_284550

theorem white_line_length 
  (blue_length : ℝ) 
  (length_difference : ℝ) 
  (h1 : blue_length = 3.33) 
  (h2 : length_difference = 4.33) : 
  blue_length + length_difference = 7.66 := by
sorry

end NUMINAMATH_CALUDE_white_line_length_l2845_284550


namespace NUMINAMATH_CALUDE_arithmetic_equality_l2845_284538

theorem arithmetic_equality : 3889 + 12.808 - 47.80600000000004 = 3854.002 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l2845_284538


namespace NUMINAMATH_CALUDE_units_digit_of_p_l2845_284589

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_of_p (p : ℤ) : 
  p > 0 → 
  units_digit p > 0 →
  units_digit (p^3) - units_digit (p^2) = 0 →
  units_digit (p + 4) = 0 →
  units_digit p = 6 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_p_l2845_284589


namespace NUMINAMATH_CALUDE_chad_odd_jobs_income_l2845_284575

theorem chad_odd_jobs_income 
  (savings_rate : Real)
  (mowing_income : Real)
  (birthday_income : Real)
  (videogame_income : Real)
  (total_savings : Real)
  (h1 : savings_rate = 0.4)
  (h2 : mowing_income = 600)
  (h3 : birthday_income = 250)
  (h4 : videogame_income = 150)
  (h5 : total_savings = 460) :
  ∃ (odd_jobs_income : Real),
    odd_jobs_income = 150 ∧
    total_savings = savings_rate * (mowing_income + birthday_income + videogame_income + odd_jobs_income) :=
by
  sorry


end NUMINAMATH_CALUDE_chad_odd_jobs_income_l2845_284575


namespace NUMINAMATH_CALUDE_chocolate_theorem_l2845_284579

def chocolate_problem (total_boxes : ℕ) (pieces_per_box : ℕ) (remaining_pieces : ℕ) : ℕ :=
  (total_boxes * pieces_per_box - remaining_pieces) / pieces_per_box

theorem chocolate_theorem : chocolate_problem 12 6 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_theorem_l2845_284579


namespace NUMINAMATH_CALUDE_divisibility_32xy76_l2845_284560

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_32xy76 (x y : ℕ) : ℕ := 320000 + 10000 * x + 1000 * y + 76

theorem divisibility_32xy76 (x y : ℕ) (hx : is_digit x) (hy : is_digit y) :
  ∃ k : ℕ, number_32xy76 x y = 4 * k :=
sorry

end NUMINAMATH_CALUDE_divisibility_32xy76_l2845_284560


namespace NUMINAMATH_CALUDE_average_weight_problem_l2845_284556

/-- Given the average weight of three people and two subsets of them, prove the average weight of the remaining subset. -/
theorem average_weight_problem (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 60)  -- Average weight of a, b, and c is 60 kg
  (h2 : (b + c) / 2 = 50)      -- Average weight of b and c is 50 kg
  (h3 : b = 60)                -- Weight of b is 60 kg
  : (a + b) / 2 = 70 :=        -- Average weight of a and b is 70 kg
by sorry

end NUMINAMATH_CALUDE_average_weight_problem_l2845_284556


namespace NUMINAMATH_CALUDE_seating_arrangements_l2845_284566

/-- The number of seats in the row -/
def total_seats : ℕ := 8

/-- The number of people to be seated -/
def people : ℕ := 3

/-- The number of seats that must be left empty at the ends -/
def end_seats : ℕ := 2

/-- The number of seats available for seating after accounting for end seats -/
def available_seats : ℕ := total_seats - end_seats

/-- The number of gaps between seated people (including before first and after last) -/
def gaps : ℕ := people + 1

/-- Theorem stating the number of seating arrangements -/
theorem seating_arrangements :
  (Nat.choose available_seats gaps) * (Nat.factorial people) = 24 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l2845_284566


namespace NUMINAMATH_CALUDE_unique_solution_for_squared_geometric_sum_l2845_284500

theorem unique_solution_for_squared_geometric_sum : 
  ∃! (n m : ℕ), n > 0 ∧ m > 0 ∧ n^2 = (m^5 - 1) / (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_squared_geometric_sum_l2845_284500


namespace NUMINAMATH_CALUDE_proposition_equivalence_l2845_284540

-- Define the concept of angles
def Angle : Type := ℝ

-- Define what it means for two angles to be equal
def equal_angles (a b : Angle) : Prop := a = b

-- Define what it means for two angles to be vertical angles
def vertical_angles (a b : Angle) : Prop := sorry

-- State the original proposition
def original_proposition : Prop :=
  ∀ a b : Angle, equal_angles a b → vertical_angles a b

-- State the conditional form
def conditional_form : Prop :=
  ∀ a b : Angle, vertical_angles a b → equal_angles a b

-- Theorem stating the equivalence of the two forms
theorem proposition_equivalence : original_proposition ↔ conditional_form :=
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l2845_284540


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_l2845_284545

theorem smallest_integer_with_remainder (n : ℕ) : n = 169 →
  n > 16 ∧
  n % 6 = 1 ∧
  n % 7 = 1 ∧
  n % 8 = 1 ∧
  ∀ m : ℕ, m > 16 ∧ m % 6 = 1 ∧ m % 7 = 1 ∧ m % 8 = 1 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_l2845_284545


namespace NUMINAMATH_CALUDE_ab_value_l2845_284546

theorem ab_value (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 50) : a * b = 7 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2845_284546


namespace NUMINAMATH_CALUDE_probability_of_two_pairs_l2845_284552

def number_of_dice : ℕ := 7
def sides_per_die : ℕ := 6

def total_outcomes : ℕ := sides_per_die ^ number_of_dice

def ways_to_choose_pair_numbers : ℕ := Nat.choose 6 2
def ways_to_choose_dice_for_pairs : ℕ := Nat.choose number_of_dice 4
def ways_to_arrange_pairs : ℕ := 6  -- 4! / (2! * 2!)
def ways_to_choose_remaining_numbers : ℕ := 4 * 3 * 2

def successful_outcomes : ℕ := 
  ways_to_choose_pair_numbers * ways_to_choose_dice_for_pairs * 
  ways_to_arrange_pairs * ways_to_choose_remaining_numbers

theorem probability_of_two_pairs (h : successful_outcomes = 151200 ∧ total_outcomes = 279936) :
  (successful_outcomes : ℚ) / total_outcomes = 175 / 324 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_two_pairs_l2845_284552


namespace NUMINAMATH_CALUDE_intersection_when_m_3_range_of_m_when_B_subset_A_l2845_284585

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem intersection_when_m_3 : A ∩ B 3 = {x | 4 ≤ x ∧ x ≤ 5} := by sorry

theorem range_of_m_when_B_subset_A : 
  ∀ m : ℝ, B m ⊆ A → m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_3_range_of_m_when_B_subset_A_l2845_284585


namespace NUMINAMATH_CALUDE_no_solution_exists_l2845_284582

theorem no_solution_exists (x y : ℕ+) : 3 * y^2 ≠ x^4 + x := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2845_284582


namespace NUMINAMATH_CALUDE_parabola_equation_from_ellipse_focus_l2845_284542

/-- The standard equation of a parabola with its focus at the right focus of the ellipse x^2/3 + y^2 = 1 -/
theorem parabola_equation_from_ellipse_focus : 
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ (x y : ℝ), (x^2 / 3 + y^2 = 1 ∧ x > 0) → 
    (∀ (u v : ℝ), v^2 = 4 * Real.sqrt 2 * u ↔ 
      (u - x)^2 + v^2 = (u - a)^2 + v^2)) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_from_ellipse_focus_l2845_284542


namespace NUMINAMATH_CALUDE_evaluate_expression_l2845_284504

theorem evaluate_expression (x y z : ℤ) (hx : x = -1) (hy : y = 4) (hz : z = 2) :
  z * (2 * y - 3 * x) = 22 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2845_284504


namespace NUMINAMATH_CALUDE_counterclockwise_notation_l2845_284562

/-- Represents the direction of rotation -/
inductive RotationDirection
  | Clockwise
  | Counterclockwise

/-- Represents a rotation with its direction and angle -/
structure Rotation :=
  (direction : RotationDirection)
  (angle : ℝ)

/-- Converts a rotation to its signed angle representation -/
def Rotation.toSignedAngle (r : Rotation) : ℝ :=
  match r.direction with
  | RotationDirection.Clockwise => r.angle
  | RotationDirection.Counterclockwise => -r.angle

theorem counterclockwise_notation (angle : ℝ) :
  (Rotation.toSignedAngle { direction := RotationDirection.Counterclockwise, angle := angle }) = -angle :=
by sorry

end NUMINAMATH_CALUDE_counterclockwise_notation_l2845_284562


namespace NUMINAMATH_CALUDE_club_members_count_l2845_284532

theorem club_members_count : ∃ (M : ℕ), 
  M > 0 ∧ 
  (2 : ℚ) / 5 * M = (M : ℚ) - (3 : ℚ) / 5 * M ∧ 
  (1 : ℚ) / 3 * ((3 : ℚ) / 5 * M) = (1 : ℚ) / 5 * M ∧ 
  (2 : ℚ) / 5 * M = 6 ∧ 
  M = 15 :=
by sorry

end NUMINAMATH_CALUDE_club_members_count_l2845_284532


namespace NUMINAMATH_CALUDE_intersection_digit_l2845_284507

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def third_digit (n : ℕ) : ℕ := (n / 100) % 10

def four_digit_power_of_2 (m : ℕ) : Prop :=
  ∃ k, is_four_digit (2^k) ∧ m = third_digit (2^k)

def four_digit_power_of_5 (n : ℕ) : Prop :=
  ∃ k, is_four_digit (5^k) ∧ n = third_digit (5^k)

theorem intersection_digit :
  ∃! d, four_digit_power_of_2 d ∧ four_digit_power_of_5 d :=
sorry

end NUMINAMATH_CALUDE_intersection_digit_l2845_284507


namespace NUMINAMATH_CALUDE_andrews_piggy_bank_donation_l2845_284543

/-- Calculates the amount Andrew donated from his piggy bank to the homeless shelter --/
theorem andrews_piggy_bank_donation
  (total_earnings : ℕ)
  (ingredient_cost : ℕ)
  (total_shelter_donation : ℕ)
  (h1 : total_earnings = 400)
  (h2 : ingredient_cost = 100)
  (h3 : total_shelter_donation = 160) :
  total_shelter_donation - ((total_earnings - ingredient_cost) / 2) = 10 := by
  sorry

#check andrews_piggy_bank_donation

end NUMINAMATH_CALUDE_andrews_piggy_bank_donation_l2845_284543


namespace NUMINAMATH_CALUDE_parabola_translation_l2845_284594

-- Define the original and transformed parabolas
def original_parabola (x : ℝ) : ℝ := x^2
def transformed_parabola (x : ℝ) : ℝ := x^2 - 5

-- Define the translation
def translation (y : ℝ) : ℝ := y - 5

-- Theorem statement
theorem parabola_translation :
  ∀ x : ℝ, transformed_parabola x = translation (original_parabola x) := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l2845_284594


namespace NUMINAMATH_CALUDE_two_tangents_from_three_zero_l2845_284581

/-- The curve y = x^2 - 2x -/
def curve (x : ℝ) : ℝ := x^2 - 2*x

/-- Condition for two tangents to exist from a point (a, b) to the curve -/
def two_tangents_condition (a b : ℝ) : Prop :=
  a^2 - 2*a - b > 0

/-- Theorem stating that (3, 0) satisfies the two tangents condition -/
theorem two_tangents_from_three_zero :
  two_tangents_condition 3 0 := by
  sorry

end NUMINAMATH_CALUDE_two_tangents_from_three_zero_l2845_284581
