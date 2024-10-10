import Mathlib

namespace market_purchase_cost_l3812_381201

/-- The total cost of buying tomatoes and cabbage -/
def total_cost (a b : ℝ) : ℝ :=
  30 * a + 50 * b

/-- Theorem: The total cost of buying 30 kg of tomatoes at 'a' yuan per kg
    and 50 kg of cabbage at 'b' yuan per kg is 30a + 50b yuan -/
theorem market_purchase_cost (a b : ℝ) :
  total_cost a b = 30 * a + 50 * b := by
  sorry

end market_purchase_cost_l3812_381201


namespace ladder_rope_difference_l3812_381293

/-- Proves that the ladder is 10 feet longer than the rope given the climbing scenario -/
theorem ladder_rope_difference (
  num_flights : ℕ) 
  (flight_height : ℝ) 
  (total_height : ℝ) 
  (h1 : num_flights = 3)
  (h2 : flight_height = 10)
  (h3 : total_height = 70) : 
  let stairs_height := num_flights * flight_height
  let rope_height := stairs_height / 2
  let ladder_height := total_height - (stairs_height + rope_height)
  ladder_height - rope_height = 10 := by
sorry

end ladder_rope_difference_l3812_381293


namespace total_candy_is_98_l3812_381223

/-- The number of boxes of chocolate candy Adam bought -/
def chocolate_boxes : ℕ := 3

/-- The number of pieces in each box of chocolate candy -/
def chocolate_pieces_per_box : ℕ := 6

/-- The number of boxes of caramel candy Adam bought -/
def caramel_boxes : ℕ := 5

/-- The number of pieces in each box of caramel candy -/
def caramel_pieces_per_box : ℕ := 8

/-- The number of boxes of gummy candy Adam bought -/
def gummy_boxes : ℕ := 4

/-- The number of pieces in each box of gummy candy -/
def gummy_pieces_per_box : ℕ := 10

/-- The total number of candy pieces Adam bought -/
def total_candy : ℕ := chocolate_boxes * chocolate_pieces_per_box + 
                        caramel_boxes * caramel_pieces_per_box + 
                        gummy_boxes * gummy_pieces_per_box

theorem total_candy_is_98 : total_candy = 98 := by
  sorry

end total_candy_is_98_l3812_381223


namespace greatest_n_product_consecutive_odds_l3812_381247

theorem greatest_n_product_consecutive_odds : 
  ∃ (n : ℕ), n = 899 ∧ 
  n < 1000 ∧ 
  (∃ (m : ℤ), 4 * n^3 - 3 * n = (2 * m - 1) * (2 * m + 1)) ∧
  (∀ (k : ℕ), k < 1000 → k > n → 
    ¬∃ (m : ℤ), 4 * k^3 - 3 * k = (2 * m - 1) * (2 * m + 1)) :=
by sorry

end greatest_n_product_consecutive_odds_l3812_381247


namespace largest_gcd_of_sum_780_l3812_381220

theorem largest_gcd_of_sum_780 :
  ∃ (x y : ℕ+), x + y = 780 ∧ 
  ∀ (a b : ℕ+), a + b = 780 → Nat.gcd x y ≥ Nat.gcd a b ∧
  Nat.gcd x y = 390 :=
sorry

end largest_gcd_of_sum_780_l3812_381220


namespace salary_adjustment_l3812_381212

theorem salary_adjustment (original_salary : ℝ) (original_salary_positive : original_salary > 0) :
  let reduced_salary := original_salary * 0.9
  (reduced_salary * (1 + 100/9 * 0.01) : ℝ) = original_salary := by
  sorry

end salary_adjustment_l3812_381212


namespace complex_equation_solution_l3812_381281

theorem complex_equation_solution (z : ℂ) :
  z * (2 - Complex.I) = 10 + 5 * Complex.I → z = 3 + 4 * Complex.I := by
  sorry

end complex_equation_solution_l3812_381281


namespace sale_price_calculation_l3812_381215

theorem sale_price_calculation (original_price : ℝ) :
  let increased_price := original_price * 1.3
  let sale_price := increased_price * 0.9
  sale_price = original_price * 1.17 :=
by sorry

end sale_price_calculation_l3812_381215


namespace sum_difference_of_squares_l3812_381264

theorem sum_difference_of_squares (n : ℤ) : ∃ a b c d : ℤ, n = a^2 + b^2 - c^2 - d^2 := by
  sorry

end sum_difference_of_squares_l3812_381264


namespace mary_has_five_candies_l3812_381284

/-- The number of candies Mary has on Halloween -/
def marys_candies (bob_candies sue_candies john_candies sam_candies total_candies : ℕ) : ℕ :=
  total_candies - (bob_candies + sue_candies + john_candies + sam_candies)

/-- Theorem: Mary has 5 candies given the Halloween candy distribution -/
theorem mary_has_five_candies :
  marys_candies 10 20 5 10 50 = 5 := by
  sorry

end mary_has_five_candies_l3812_381284


namespace school_boys_count_l3812_381226

/-- The number of girls in the school -/
def num_girls : ℕ := 34

/-- The difference between the number of boys and girls -/
def difference : ℕ := 807

/-- The number of boys in the school -/
def num_boys : ℕ := num_girls + difference

theorem school_boys_count : num_boys = 841 := by
  sorry

end school_boys_count_l3812_381226


namespace polynomial_remainder_l3812_381257

theorem polynomial_remainder (x : ℝ) : 
  (8 * x^3 - 20 * x^2 + 28 * x - 26) % (4 * x - 8) = 14 := by
sorry

end polynomial_remainder_l3812_381257


namespace roots_of_polynomial_l3812_381258

def p (x : ℝ) : ℝ := 8 * x^4 + 26 * x^3 - 65 * x^2 + 24 * x

theorem roots_of_polynomial :
  (p 0 = 0) ∧ (p (1/2) = 0) ∧ (p (3/2) = 0) ∧ (p (-4) = 0) :=
sorry

end roots_of_polynomial_l3812_381258


namespace unique_solution_condition_l3812_381230

theorem unique_solution_condition (p q : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + p = q * x - 2) ↔ q ≠ 4 := by
  sorry

end unique_solution_condition_l3812_381230


namespace correct_ordering_l3812_381253

theorem correct_ordering (m n p q : ℝ) 
  (h1 : m < n) 
  (h2 : p < q) 
  (h3 : (p - m) * (p - n) < 0) 
  (h4 : (q - m) * (q - n) < 0) : 
  m < p ∧ p < q ∧ q < n := by
  sorry

end correct_ordering_l3812_381253


namespace gilda_remaining_marbles_l3812_381267

/-- The percentage of marbles Gilda has left after giving away to her friends and family -/
def gildasRemainingMarbles : ℝ :=
  let initialMarbles := 100
  let afterPedro := initialMarbles * (1 - 0.30)
  let afterEbony := afterPedro * (1 - 0.05)
  let afterJimmy := afterEbony * (1 - 0.30)
  let afterTina := afterJimmy * (1 - 0.10)
  afterTina

/-- Theorem stating that Gilda has 41.895% of her original marbles left -/
theorem gilda_remaining_marbles :
  ∀ ε > 0, |gildasRemainingMarbles - 41.895| < ε :=
sorry

end gilda_remaining_marbles_l3812_381267


namespace average_children_in_families_with_children_l3812_381268

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 12)
  (h2 : total_average = 3)
  (h3 : childless_families = 3) :
  (total_families : ℚ) * total_average / (total_families - childless_families : ℚ) = 4 :=
by sorry

end average_children_in_families_with_children_l3812_381268


namespace job_completion_time_l3812_381299

/-- The time taken for machines to complete a job given specific conditions -/
theorem job_completion_time : 
  -- Machine R completion time
  let r_time : ℝ := 36
  -- Machine S completion time
  let s_time : ℝ := 2
  -- Number of each type of machine used
  let n : ℝ := 0.9473684210526315
  -- Total rate of job completion
  let total_rate : ℝ := n * (1 / r_time) + n * (1 / s_time)
  -- Time taken to complete the job
  let completion_time : ℝ := 1 / total_rate
  -- Proof that the completion time is 2 hours
  completion_time = 2 := by
  sorry

end job_completion_time_l3812_381299


namespace no_solution_iff_m_eq_neg_four_l3812_381280

theorem no_solution_iff_m_eq_neg_four :
  ∀ m : ℝ, (∀ x : ℝ, (x ≠ 2 ∧ x ≠ -2) → 
    ((x - 2) / (x + 2) - m * x / (x^2 - 4) ≠ 1)) ↔ m = -4 :=
by sorry

end no_solution_iff_m_eq_neg_four_l3812_381280


namespace complex_equation_solution_l3812_381249

theorem complex_equation_solution (z : ℂ) : (1 + 2*I)*z = 5 → z = 1 - 2*I := by
  sorry

end complex_equation_solution_l3812_381249


namespace largest_divisor_of_m_l3812_381259

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 54 ∣ m^2) : 
  ∃ (d : ℕ), d ∣ m ∧ d = 9 ∧ ∀ (k : ℕ), k ∣ m → k ≤ d :=
sorry

end largest_divisor_of_m_l3812_381259


namespace quadratic_roots_property_l3812_381296

theorem quadratic_roots_property (α β : ℝ) : 
  (α^2 + α - 1 = 0) → 
  (β^2 + β - 1 = 0) → 
  (α ≠ β) →
  α^4 - 3*β = 5 := by
sorry

end quadratic_roots_property_l3812_381296


namespace b_100_mod_50_l3812_381209

/-- Define the sequence b_n = 7^n + 9^n -/
def b (n : ℕ) : ℕ := 7^n + 9^n

/-- Theorem: b_100 ≡ 2 (mod 50) -/
theorem b_100_mod_50 : b 100 ≡ 2 [MOD 50] := by
  sorry

end b_100_mod_50_l3812_381209


namespace fruit_basket_problem_l3812_381252

/-- Represents the number of times fruits are taken out -/
def x : ℕ := sorry

/-- The original number of apples -/
def initial_apples : ℕ := 3 * x + 1

/-- The original number of oranges -/
def initial_oranges : ℕ := 4 * x + 12

/-- The condition that the number of oranges is twice that of apples -/
axiom orange_apple_ratio : initial_oranges = 2 * initial_apples

theorem fruit_basket_problem : x = 5 := by sorry

end fruit_basket_problem_l3812_381252


namespace smallest_positive_value_36k_minus_5m_l3812_381232

theorem smallest_positive_value_36k_minus_5m (k m : ℕ+) :
  (∀ n : ℕ+, 36^(k : ℕ) - 5^(m : ℕ) ≠ n) →
  (36^(k : ℕ) - 5^(m : ℕ) = 11 ∨ 36^(k : ℕ) - 5^(m : ℕ) > 11) :=
by sorry

end smallest_positive_value_36k_minus_5m_l3812_381232


namespace runners_meet_count_l3812_381233

-- Constants
def total_time : ℝ := 45
def odell_speed : ℝ := 260
def odell_radius : ℝ := 70
def kershaw_speed : ℝ := 320
def kershaw_radius : ℝ := 80
def kershaw_delay : ℝ := 5

-- Theorem statement
theorem runners_meet_count :
  let odell_angular_speed := odell_speed / odell_radius
  let kershaw_angular_speed := kershaw_speed / kershaw_radius
  let relative_angular_speed := odell_angular_speed + kershaw_angular_speed
  let effective_time := total_time - kershaw_delay
  let meet_count := ⌊(effective_time * relative_angular_speed) / (2 * Real.pi)⌋
  meet_count = 49 := by
  sorry

end runners_meet_count_l3812_381233


namespace division_remainder_proof_l3812_381208

theorem division_remainder_proof (D R r : ℕ) : 
  D = 12 * 42 + R →
  D = 21 * 24 + r →
  0 ≤ r →
  r < 21 →
  r = 0 := by
sorry

end division_remainder_proof_l3812_381208


namespace arithmetic_sequence_second_term_l3812_381250

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The second term of a sequence. -/
def second_term (a : ℕ → ℝ) : ℝ := a 1

theorem arithmetic_sequence_second_term
  (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a) (h_sum : a 0 + a 2 = 8) :
  second_term a = 4 := by
  sorry

end arithmetic_sequence_second_term_l3812_381250


namespace lcm_prime_sum_l3812_381214

theorem lcm_prime_sum (x y : ℕ) : 
  Nat.Prime x → Nat.Prime y → x > y → Nat.lcm x y = 10 → 2 * x + y = 12 := by
  sorry

end lcm_prime_sum_l3812_381214


namespace x_squared_coefficient_in_expansion_l3812_381246

/-- The coefficient of x² in the expansion of (2+x)(1-2x)^5 is 70 -/
theorem x_squared_coefficient_in_expansion : Int := by
  sorry

end x_squared_coefficient_in_expansion_l3812_381246


namespace sum_is_zero_l3812_381251

/-- Given a finite subset M of real numbers with more than 2 elements,
    if for each element the absolute value is at least as large as
    the absolute value of the sum of the other elements,
    then the sum of all elements in M is zero. -/
theorem sum_is_zero (M : Finset ℝ) (h_size : 2 < M.card) :
  (∀ a ∈ M, |a| ≥ |M.sum id - a|) → M.sum id = 0 := by
  sorry

end sum_is_zero_l3812_381251


namespace max_value_a_l3812_381282

theorem max_value_a (a : ℝ) : 
  (∀ x > 0, x * Real.exp x - a * (x + 1) ≥ Real.log x) → a ≤ 1 :=
by sorry

end max_value_a_l3812_381282


namespace line_intersects_y_axis_l3812_381211

/-- A line passing through two given points intersects the y-axis at (0, 0) -/
theorem line_intersects_y_axis (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : x₁ = 3 ∧ y₁ = 9)
  (h₂ : x₂ = -7 ∧ y₂ = -21) :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) ∧
    0 = m * 0 + b :=
by sorry

end line_intersects_y_axis_l3812_381211


namespace circle_area_polar_l3812_381202

/-- The area of the circle described by the polar equation r = 4 cos θ - 3 sin θ is 25π/4 -/
theorem circle_area_polar (θ : ℝ) (r : ℝ → ℝ) : 
  (r θ = 4 * Real.cos θ - 3 * Real.sin θ) → 
  (∃ c : ℝ × ℝ, ∃ radius : ℝ, 
    (∀ x y : ℝ, (x - c.1)^2 + (y - c.2)^2 = radius^2 ↔ 
      ∃ θ : ℝ, x = r θ * Real.cos θ ∧ y = r θ * Real.sin θ) ∧
    π * radius^2 = 25 * π / 4) :=
sorry

end circle_area_polar_l3812_381202


namespace imaginary_part_of_complex_fraction_l3812_381218

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.im (i^2 / (2*i - 1)) = 2/5 := by
  sorry

end imaginary_part_of_complex_fraction_l3812_381218


namespace square_measurement_error_l3812_381260

theorem square_measurement_error (area_error : Real) (side_error : Real) : 
  area_error = 8.16 → side_error = 4 := by
  sorry

end square_measurement_error_l3812_381260


namespace product_of_large_numbers_l3812_381236

theorem product_of_large_numbers : (300000 : ℕ) * 300000 * 3 = 270000000000 := by
  sorry

end product_of_large_numbers_l3812_381236


namespace equal_expressions_l3812_381234

theorem equal_expressions : 10006 - 8008 = 10000 - 8002 := by sorry

end equal_expressions_l3812_381234


namespace smallest_triangle_side_l3812_381297

-- Define the triangle sides
def a : ℕ := 7
def b : ℕ := 11

-- Define the triangle inequality
def is_triangle (x y z : ℕ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- Define the property we want to prove
def smallest_side (s : ℕ) : Prop :=
  is_triangle a b s ∧ ∀ t : ℕ, t < s → ¬(is_triangle a b t)

-- The theorem to prove
theorem smallest_triangle_side : smallest_side 5 := by
  sorry

end smallest_triangle_side_l3812_381297


namespace first_car_speed_l3812_381243

theorem first_car_speed (v : ℝ) (h1 : v > 0) : 
  v * 2.25 * 4 = 720 → v * 1.25 = 100 := by
  sorry

end first_car_speed_l3812_381243


namespace seven_is_target_digit_l3812_381277

/-- The numeral we're examining -/
def numeral : ℕ := 657903

/-- The difference between local value and face value -/
def difference : ℕ := 6993

/-- Function to get the local value of a digit in a specific place -/
def localValue (digit : ℕ) (place : ℕ) : ℕ := digit * (10 ^ place)

/-- Function to get the face value of a digit -/
def faceValue (digit : ℕ) : ℕ := digit

/-- Theorem stating that 7 is the only digit in the numeral with the given difference -/
theorem seven_is_target_digit :
  ∃! d : ℕ, d < 10 ∧ 
    (∃ p : ℕ, p < 6 ∧ 
      (numeral / (10 ^ p)) % 10 = d ∧
      localValue d p - faceValue d = difference) :=
sorry

end seven_is_target_digit_l3812_381277


namespace steven_has_16_apples_l3812_381204

/-- Represents the number of fruits a person has -/
structure FruitCount where
  peaches : ℕ
  apples : ℕ

/-- Given information about Steven and Jake's fruit counts -/
def steven_jake_fruits : Prop :=
  ∃ (steven jake : FruitCount),
    steven.peaches = 17 ∧
    steven.peaches = steven.apples + 1 ∧
    jake.peaches + 6 = steven.peaches ∧
    jake.apples = steven.apples + 8

/-- Theorem stating that Steven has 16 apples -/
theorem steven_has_16_apples :
  steven_jake_fruits → ∃ (steven : FruitCount), steven.apples = 16 := by
  sorry

end steven_has_16_apples_l3812_381204


namespace min_value_and_existence_l3812_381294

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + t*x + 1) / Real.log 2

theorem min_value_and_existence (t : ℝ) (h : t > -2) :
  (∀ x ∈ Set.Icc 0 2, f t x ≥ (if -2 < t ∧ t < 0 then Real.log (1 - t^2/4) / Real.log 2 else 0)) ∧
  (∃ a b : ℝ, a ≠ b ∧ a ∈ Set.Ioo 0 2 ∧ b ∈ Set.Ioo 0 2 ∧ 
   f t a = Real.log a / Real.log 2 ∧ f t b = Real.log b / Real.log 2 ↔ 
   t > -3/2 ∧ t < -1) :=
by sorry

end min_value_and_existence_l3812_381294


namespace divisibility_by_eleven_l3812_381241

/-- Given a positive integer, returns the number obtained by reversing its digits -/
def reverse_digits (n : ℕ) : ℕ := sorry

theorem divisibility_by_eleven (A : ℕ) (h : A > 0) :
  let B := reverse_digits A
  (11 ∣ (A + B)) ∨ (11 ∣ (A - B)) := by sorry

end divisibility_by_eleven_l3812_381241


namespace divisibility_problem_specific_divisibility_problem_l3812_381222

theorem divisibility_problem (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x y : ℕ),
    (x = (d - n % d) % d) ∧
    (y = n % d) ∧
    ((n + x) % d = 0) ∧
    ((n - y) % d = 0) ∧
    (∀ x' : ℕ, x' < x → (n + x') % d ≠ 0) ∧
    (∀ y' : ℕ, y' < y → (n - y') % d ≠ 0) :=
by sorry

-- Specific instance for the given problem
theorem specific_divisibility_problem :
  ∃ (x y : ℕ),
    (x = 10) ∧
    (y = 27) ∧
    ((1100 + x) % 37 = 0) ∧
    ((1100 - y) % 37 = 0) ∧
    (∀ x' : ℕ, x' < x → (1100 + x') % 37 ≠ 0) ∧
    (∀ y' : ℕ, y' < y → (1100 - y') % 37 ≠ 0) :=
by sorry

end divisibility_problem_specific_divisibility_problem_l3812_381222


namespace flower_bed_fraction_l3812_381254

-- Define the yard and flower beds
def yard_length : ℝ := 25
def yard_width : ℝ := 5
def flower_bed_area : ℝ := 50

-- Define the theorem
theorem flower_bed_fraction :
  let total_yard_area := yard_length * yard_width
  let total_flower_bed_area := 2 * flower_bed_area
  (total_flower_bed_area / total_yard_area) = 4/5 := by
  sorry

end flower_bed_fraction_l3812_381254


namespace cliffs_rock_collection_l3812_381248

theorem cliffs_rock_collection (sedimentary : ℕ) (igneous : ℕ) : 
  igneous = sedimentary / 2 →
  (2 : ℕ) * (igneous / 3) = 40 →
  sedimentary + igneous = 180 := by
sorry

end cliffs_rock_collection_l3812_381248


namespace studentG_score_l3812_381298

-- Define the answer types
inductive Answer
| Correct
| Incorrect
| Unanswered

-- Define the scoring function
def score (a : Answer) : Nat :=
  match a with
  | Answer.Correct => 2
  | Answer.Incorrect => 0
  | Answer.Unanswered => 1

-- Define Student G's answer pattern
def studentG_answers : List Answer :=
  [Answer.Correct, Answer.Incorrect, Answer.Correct, Answer.Correct, Answer.Incorrect, Answer.Correct]

-- Theorem: Student G's total score is 8 points
theorem studentG_score :
  (studentG_answers.map score).sum = 8 := by
  sorry

end studentG_score_l3812_381298


namespace particle_speed_at_2_l3812_381269

/-- The position of a particle at time t -/
def particle_position (t : ℝ) : ℝ × ℝ :=
  (t^2 + 2*t + 7, 3*t^2 + 4*t - 13)

/-- The speed of the particle at time t -/
noncomputable def particle_speed (t : ℝ) : ℝ :=
  let pos_t := particle_position t
  let pos_next := particle_position (t + 1)
  let dx := pos_next.1 - pos_t.1
  let dy := pos_next.2 - pos_t.2
  Real.sqrt (dx^2 + dy^2)

/-- Theorem: The speed of the particle at t = 2 is √410 -/
theorem particle_speed_at_2 :
  particle_speed 2 = Real.sqrt 410 := by
  sorry

end particle_speed_at_2_l3812_381269


namespace value_of_expression_l3812_381295

theorem value_of_expression (x y : ℝ) 
  (h1 : x^2 - x*y = 12) 
  (h2 : y^2 - x*y = 15) : 
  2*(x-y)^2 - 3 = 51 := by
sorry

end value_of_expression_l3812_381295


namespace vector_sum_simplification_l3812_381242

variable {V : Type*} [AddCommGroup V]
variable (A B C D : V)

theorem vector_sum_simplification :
  (B - A) + (A - C) + (D - B) = D - C :=
by sorry

end vector_sum_simplification_l3812_381242


namespace complement_of_A_l3812_381283

def A : Set ℝ := {x : ℝ | x ≥ 1}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | x < 1} := by sorry

end complement_of_A_l3812_381283


namespace ellipse_intersection_properties_l3812_381262

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define the upper vertex A
def A : ℝ × ℝ := (0, 1)

-- Define a line not passing through A
def line (k m : ℝ) (x : ℝ) : ℝ := k * x + m

-- Define the condition that the line intersects the ellipse at P and Q
def intersects (k m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    y₁ = line k m x₁ ∧ y₂ = line k m x₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - A.1) * (x₂ - A.1) + (y₁ - A.2) * (y₂ - A.2) = 0

-- Main theorem
theorem ellipse_intersection_properties :
  ∀ (k m : ℝ),
    intersects k m →
    (∀ (x₁ y₁ x₂ y₂ : ℝ),
      ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
      y₁ = line k m x₁ ∧ y₂ = line k m x₂ →
      perpendicular x₁ y₁ x₂ y₂) →
    (m = -1/2) ∧
    (∃ (S : Set ℝ), S = {s | s ≥ 9/4} ∧
      ∀ (x₁ y₁ x₂ y₂ : ℝ),
        ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
        y₁ = line k m x₁ ∧ y₂ = line k m x₂ →
        ∃ (area : ℝ), area ∈ S ∧
          (∃ (Bx By : ℝ), area = 1/2 * |Bx - x₁| * |By - y₁|)) :=
by sorry

end ellipse_intersection_properties_l3812_381262


namespace statue_weight_calculation_l3812_381216

/-- The weight of a marble statue after three weeks of carving --/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  let week1_remainder := initial_weight * (1 - 0.28)
  let week2_remainder := week1_remainder * (1 - 0.18)
  let week3_remainder := week2_remainder * (1 - 0.20)
  week3_remainder

/-- Theorem stating the final weight of the statue --/
theorem statue_weight_calculation :
  final_statue_weight 180 = 85.0176 := by
  sorry

end statue_weight_calculation_l3812_381216


namespace no_valid_prime_pairs_l3812_381261

theorem no_valid_prime_pairs : 
  ∀ a b : ℕ, 
    Prime a → 
    Prime b → 
    b > a → 
    (a - 8) * (b - 8) = 64 → 
    False :=
by
  sorry

end no_valid_prime_pairs_l3812_381261


namespace problem_statement_l3812_381238

theorem problem_statement (a b c d : ℕ+) (r : ℚ) 
  (h1 : r = 1 - (a : ℚ) / b - (c : ℚ) / d)
  (h2 : a + c ≤ 1993)
  (h3 : r > 0) :
  r > 1 / (1993 : ℚ)^3 := by
sorry

end problem_statement_l3812_381238


namespace divisibility_circle_l3812_381279

/-- Given seven natural numbers in a circle where each adjacent pair has a divisibility relation,
    there exists a non-adjacent pair with the same property. -/
theorem divisibility_circle (a : Fin 7 → ℕ) 
  (h : ∀ i : Fin 7, (a i ∣ a (i + 1)) ∨ (a (i + 1) ∣ a i)) :
  ∃ i j : Fin 7, i ≠ j ∧ (j ≠ i + 1) ∧ (j ≠ i - 1) ∧ ((a i ∣ a j) ∨ (a j ∣ a i)) :=
sorry

end divisibility_circle_l3812_381279


namespace b_contribution_is_90000_l3812_381231

/-- Represents the business partnership between A and B --/
structure Partnership where
  a_investment : ℕ  -- A's initial investment
  b_join_time : ℕ  -- Time when B joins (in months)
  total_time : ℕ   -- Total investment period (in months)
  profit_ratio_a : ℕ  -- A's part in profit ratio
  profit_ratio_b : ℕ  -- B's part in profit ratio

/-- Calculates B's contribution given the partnership details --/
def calculate_b_contribution (p : Partnership) : ℕ :=
  -- Placeholder for the actual calculation
  0

/-- Theorem stating that B's contribution is 90000 given the specific partnership details --/
theorem b_contribution_is_90000 :
  let p : Partnership := {
    a_investment := 35000,
    b_join_time := 5,
    total_time := 12,
    profit_ratio_a := 2,
    profit_ratio_b := 3
  }
  calculate_b_contribution p = 90000 := by
  sorry


end b_contribution_is_90000_l3812_381231


namespace quadratic_coefficient_l3812_381207

theorem quadratic_coefficient (b : ℝ) : 
  ((-14 : ℝ)^2 + b * (-14) + 49 = 0) → b = 17.5 := by
  sorry

end quadratic_coefficient_l3812_381207


namespace interest_rate_problem_l3812_381266

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Problem statement -/
theorem interest_rate_problem (principal interest time : ℚ) 
  (h1 : principal = 2000)
  (h2 : interest = 500)
  (h3 : time = 2)
  (h4 : simple_interest principal (12.5 : ℚ) time = interest) :
  12.5 = (interest * 100) / (principal * time) := by
  sorry

end interest_rate_problem_l3812_381266


namespace max_log_function_l3812_381270

theorem max_log_function (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + 2*y = 1/2) :
  ∃ (max_u : ℝ), max_u = 0 ∧ 
  ∀ (u : ℝ), u = Real.log (8*x*y + 4*y^2 + 1) / Real.log (1/2) → u ≤ max_u :=
sorry

end max_log_function_l3812_381270


namespace expression_evaluation_l3812_381278

theorem expression_evaluation :
  let x : ℕ := 3
  let y : ℕ := 2
  5 * x^y + 2 * y^x + x^2 * y^2 = 97 := by
sorry

end expression_evaluation_l3812_381278


namespace yue_bao_scientific_notation_l3812_381217

theorem yue_bao_scientific_notation : 5853 = 5.853 * (10 ^ 3) := by
  sorry

end yue_bao_scientific_notation_l3812_381217


namespace vertices_form_parabola_l3812_381245

/-- The set of vertices of a family of parabolas forms a parabola -/
theorem vertices_form_parabola (a c d : ℝ) (ha : a > 0) (hc : c > 0) (hd : d > 0) :
  ∃ (f : ℝ → ℝ × ℝ), ∀ (b : ℝ),
    let (x, y) := f b
    (∀ t, y = a * t^2 + b * t + c * t + d → (x - t) * (2 * a * t + b + c) = 0) ∧
    y = -a * x^2 + d :=
  sorry

end vertices_form_parabola_l3812_381245


namespace complement_union_problem_l3812_381224

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem complement_union_problem : (U \ A) ∪ B = {2, 3, 4} := by sorry

end complement_union_problem_l3812_381224


namespace equilateral_triangle_third_vertex_y_coordinate_l3812_381228

/-- Given an equilateral triangle with two vertices at (1, 10) and (9, 10),
    and the third vertex in the first quadrant, 
    prove that the y-coordinate of the third vertex is 10 + 4√3 -/
theorem equilateral_triangle_third_vertex_y_coordinate 
  (A B C : ℝ × ℝ) : 
  A = (1, 10) → 
  B = (9, 10) → 
  C.1 ≥ 0 → 
  C.2 ≥ 0 →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 → 
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 → 
  C.2 = 10 + 4 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_third_vertex_y_coordinate_l3812_381228


namespace sum_of_roots_quadratic_l3812_381285

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ - 4 = 0) → 
  (x₂^2 + 2*x₂ - 4 = 0) → 
  (x₁ + x₂ = -2) := by
sorry

end sum_of_roots_quadratic_l3812_381285


namespace quadrilateral_property_l3812_381274

-- Define the quadrilateral ABCD
variable (A B C D : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- Define the distance between two points
def distance (P Q : Point) : ℝ := sorry

-- State the theorem
theorem quadrilateral_property (h1 : angle A D C = 135)
  (h2 : angle A D B - angle A B D = 2 * angle D A B)
  (h3 : angle A D B - angle A B D = 4 * angle C B D)
  (h4 : distance B C = Real.sqrt 2 * distance C D) :
  distance A B = distance B C + distance A D := by
  sorry

end quadrilateral_property_l3812_381274


namespace max_value_constraint_l3812_381237

theorem max_value_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2*x^2 - 3*x*y + 4*y^2 = 15) : 
  3*x^2 + 2*x*y + y^2 ≤ 50*Real.sqrt 3 + 65 := by
  sorry

end max_value_constraint_l3812_381237


namespace f_odd_f_max_on_interval_l3812_381288

/-- A function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The function f satisfies the additive property -/
axiom f_additive (x y : ℝ) : f (x + y) = f x + f y

/-- The function f is negative for positive inputs -/
axiom f_neg_for_pos (x : ℝ) (h : x > 0) : f x < 0

/-- The value of f at 1 is -2 -/
axiom f_one : f 1 = -2

/-- f is an odd function -/
theorem f_odd : ∀ x, f (-x) = -f x := by sorry

/-- The maximum value of f on [-3, 3] is 6 -/
theorem f_max_on_interval : ∃ x ∈ Set.Icc (-3) 3, ∀ y ∈ Set.Icc (-3) 3, f y ≤ f x ∧ f x = 6 := by sorry

end f_odd_f_max_on_interval_l3812_381288


namespace second_company_base_rate_l3812_381290

/-- The base rate of United Telephone in dollars -/
def united_base_rate : ℝ := 7

/-- The per-minute rate of United Telephone in dollars -/
def united_per_minute : ℝ := 0.25

/-- The per-minute rate of the second telephone company in dollars -/
def second_per_minute : ℝ := 0.20

/-- The number of minutes for which the bills are equal -/
def equal_minutes : ℝ := 100

/-- The base rate of the second telephone company in dollars -/
def second_base_rate : ℝ := 12

theorem second_company_base_rate :
  united_base_rate + united_per_minute * equal_minutes =
  second_base_rate + second_per_minute * equal_minutes :=
by sorry

end second_company_base_rate_l3812_381290


namespace sin_cos_identity_l3812_381291

theorem sin_cos_identity (x y : ℝ) :
  Real.sin (x - y) * Real.cos y + Real.cos (x - y) * Real.sin y = Real.sin x := by
  sorry

end sin_cos_identity_l3812_381291


namespace smallest_a2_l3812_381206

def sequence_property (a : ℕ → ℝ) : Prop :=
  a 0 = 0 ∧ a 1 = 1 ∧ a 2 > 0 ∧
  ∀ n ∈ Finset.range 7, a (n + 2) * a n * a (n - 1) = a (n + 2) + a n + a (n - 1)

def no_extension (a : ℕ → ℝ) : Prop :=
  ∀ x : ℝ, x * a 8 * a 7 ≠ x + a 8 + a 7

theorem smallest_a2 (a : ℕ → ℝ) (h1 : sequence_property a) (h2 : no_extension a) :
  a 2 = Real.sqrt 2 - 1 :=
sorry

end smallest_a2_l3812_381206


namespace average_difference_l3812_381225

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 7 → x = 19 := by
  sorry

end average_difference_l3812_381225


namespace least_six_digit_congruent_to_3_mod_17_l3812_381271

theorem least_six_digit_congruent_to_3_mod_17 : ∃ (n : ℕ), 
  (n ≥ 100000 ∧ n < 1000000) ∧ 
  n % 17 = 3 ∧
  ∀ (m : ℕ), (m ≥ 100000 ∧ m < 1000000 ∧ m % 17 = 3) → n ≤ m :=
by sorry

end least_six_digit_congruent_to_3_mod_17_l3812_381271


namespace difference_2010th_2008th_triangular_l3812_381240

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_2010th_2008th_triangular : 
  triangular_number 2010 - triangular_number 2008 = 4019 := by
  sorry

end difference_2010th_2008th_triangular_l3812_381240


namespace complex_equation_solution_l3812_381205

theorem complex_equation_solution (z : ℂ) : (z - 2*Complex.I) * (2 - Complex.I) = 5 → z = 2 + 3*Complex.I := by
  sorry

end complex_equation_solution_l3812_381205


namespace smallest_n_for_m_independent_same_color_lines_l3812_381210

/-- A two-coloring of a complete graph -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Fin 2

/-- Predicate for m lines of the same color with no common endpoints -/
def HasMIndependentSameColorLines (c : TwoColoring n) (m : ℕ) : Prop :=
  ∃ (edges : Fin m → Fin n × Fin n),
    (∀ i j, i ≠ j → (edges i).1 ≠ (edges j).1 ∧ (edges i).1 ≠ (edges j).2 ∧
                    (edges i).2 ≠ (edges j).1 ∧ (edges i).2 ≠ (edges j).2) ∧
    (∀ i j, c (edges i).1 (edges i).2 = c (edges j).1 (edges j).2)

/-- The main theorem -/
theorem smallest_n_for_m_independent_same_color_lines (m : ℕ) :
  (∀ n, n ≥ 3 * m - 1 → ∀ c : TwoColoring n, HasMIndependentSameColorLines c m) ∧
  (∀ n, n < 3 * m - 1 → ∃ c : TwoColoring n, ¬HasMIndependentSameColorLines c m) :=
sorry

end smallest_n_for_m_independent_same_color_lines_l3812_381210


namespace mikas_height_mikas_height_is_70_l3812_381229

/-- Proves that Mika's current height is 70 inches given the problem conditions -/
theorem mikas_height (original_height : ℝ) (sheas_growth_rate : ℝ) (mikas_growth_ratio : ℝ) 
  (sheas_current_height : ℝ) : ℝ :=
  let sheas_growth := sheas_current_height - original_height
  let mikas_growth := mikas_growth_ratio * sheas_growth
  original_height + mikas_growth
where
  -- Shea and Mika were originally the same height
  original_height_positive : 0 < original_height := by sorry
  -- Shea has grown by 25%
  sheas_growth_rate_def : sheas_growth_rate = 0.25 := by sorry
  -- Mika has grown two-thirds as many inches as Shea
  mikas_growth_ratio_def : mikas_growth_ratio = 2/3 := by sorry
  -- Shea is now 75 inches tall
  sheas_current_height_def : sheas_current_height = 75 := by sorry
  -- Shea's current height is 25% more than the original height
  sheas_growth_equation : sheas_current_height = original_height * (1 + sheas_growth_rate) := by sorry

theorem mikas_height_is_70 : mikas_height 60 0.25 (2/3) 75 = 70 := by sorry

end mikas_height_mikas_height_is_70_l3812_381229


namespace inequality_proof_l3812_381213

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) :
  a * b < 1 ∧ 1 < (a^2 + b^2) / 2 := by
  sorry

end inequality_proof_l3812_381213


namespace abs_eq_sqrt_square_l3812_381219

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end abs_eq_sqrt_square_l3812_381219


namespace a_in_interval_l3812_381256

/-- The function f(x) = x^2 + ax + b -/
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The set A = {x ∈ ℝ | f(x) ≤ 0} -/
def set_A (a b : ℝ) : Set ℝ := {x | f a b x ≤ 0}

/-- The set B = {x ∈ ℝ | f(f(x) + 1) ≤ 0} -/
def set_B (a b : ℝ) : Set ℝ := {x | f a b (f a b x + 1) ≤ 0}

/-- Theorem: If A = B ≠ ∅, then a ∈ [-2, 2] -/
theorem a_in_interval (a b : ℝ) :
  set_A a b = set_B a b ∧ (set_A a b).Nonempty → a ∈ Set.Icc (-2) 2 := by
  sorry

end a_in_interval_l3812_381256


namespace square_difference_l3812_381235

theorem square_difference (n : ℕ) (h : (n + 1)^2 = n^2 + 2*n + 1) :
  n^2 - (n - 1)^2 = 2*n - 1 := by
  sorry

#check square_difference 50

end square_difference_l3812_381235


namespace g_is_odd_l3812_381263

noncomputable def g (x : ℝ) : ℝ := Real.log (x^3 + Real.sqrt (1 + x^6))

theorem g_is_odd : ∀ x, g (-x) = -g x := by sorry

end g_is_odd_l3812_381263


namespace system_solution_l3812_381265

theorem system_solution (x y : ℝ) 
  (h1 : x * y = 6)
  (h2 : x^2 * y + x * y^2 + x + y = 63) :
  x^2 + y^2 = 69 := by
  sorry

end system_solution_l3812_381265


namespace intersection_chord_length_l3812_381289

-- Define the polar equations
def line_polar (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - 2 * Real.pi / 3) = -Real.sqrt 3

def circle_polar (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ + 2 * Real.sin θ

-- Define the Cartesian equations
def line_cartesian (x y : ℝ) : Prop := Real.sqrt 3 * x + y = 2 * Real.sqrt 3

def circle_cartesian (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

-- Theorem statement
theorem intersection_chord_length :
  ∀ A B : ℝ × ℝ,
  (∃ θ_A ρ_A, line_polar ρ_A θ_A ∧ circle_polar ρ_A θ_A ∧ A = (ρ_A * Real.cos θ_A, ρ_A * Real.sin θ_A)) →
  (∃ θ_B ρ_B, line_polar ρ_B θ_B ∧ circle_polar ρ_B θ_B ∧ B = (ρ_B * Real.cos θ_B, ρ_B * Real.sin θ_B)) →
  A ≠ B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 19 :=
by sorry

end intersection_chord_length_l3812_381289


namespace sum_six_odd_squares_not_2020_l3812_381221

theorem sum_six_odd_squares_not_2020 : ¬ ∃ (a b c d e f : ℤ),
  (2 * a + 1)^2 + (2 * b + 1)^2 + (2 * c + 1)^2 + 
  (2 * d + 1)^2 + (2 * e + 1)^2 + (2 * f + 1)^2 = 2020 :=
by sorry

end sum_six_odd_squares_not_2020_l3812_381221


namespace quadratic_equal_roots_l3812_381255

theorem quadratic_equal_roots (m n : ℝ) : 
  (∃ x : ℝ, x^(m-1) + 4*x - n = 0 ∧ 
   ∀ y : ℝ, y^(m-1) + 4*y - n = 0 → y = x) →
  m + n = -1 := by
  sorry

end quadratic_equal_roots_l3812_381255


namespace sum_of_squares_divisible_by_three_l3812_381239

theorem sum_of_squares_divisible_by_three (a b c : ℤ) 
  (ha : ¬ 3 ∣ a) (hb : ¬ 3 ∣ b) (hc : ¬ 3 ∣ c) : 
  3 ∣ (a^2 + b^2 + c^2) := by
  sorry

end sum_of_squares_divisible_by_three_l3812_381239


namespace max_m_value_l3812_381244

theorem max_m_value (m : ℝ) (h1 : m > 1) 
  (h2 : ∃ x : ℝ, x ∈ Set.Icc (-2) 0 ∧ x^2 + 2*m*x + m^2 - m ≤ 0) : 
  (∀ n : ℝ, (n > 1 ∧ ∃ y : ℝ, y ∈ Set.Icc (-2) 0 ∧ y^2 + 2*n*y + n^2 - n ≤ 0) → n ≤ m) →
  m = 4 :=
sorry

end max_m_value_l3812_381244


namespace greatest_root_of_g_l3812_381227

def g (x : ℝ) : ℝ := 12 * x^5 - 24 * x^3 + 9 * x

theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt (3/2) ∧
  g r = 0 ∧
  ∀ x, g x = 0 → x ≤ r :=
sorry

end greatest_root_of_g_l3812_381227


namespace pet_shelter_adoption_time_l3812_381292

/-- Given a pet shelter scenario, calculate the number of days needed to adopt all puppies -/
theorem pet_shelter_adoption_time (initial_puppies : ℕ) (additional_puppies : ℕ) (adoption_rate : ℕ) : 
  initial_puppies = 9 → additional_puppies = 12 → adoption_rate = 3 →
  (initial_puppies + additional_puppies) / adoption_rate = 7 := by
sorry

end pet_shelter_adoption_time_l3812_381292


namespace smallest_sum_of_reciprocals_l3812_381273

theorem smallest_sum_of_reciprocals (x y : ℕ) : 
  x ≠ y → 
  x > 0 → 
  y > 0 → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 24 → 
  ∀ a b : ℕ, a ≠ b → a > 0 → b > 0 → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 24 → 
  x + y ≤ a + b →
  x + y = 98 := by
sorry

end smallest_sum_of_reciprocals_l3812_381273


namespace circumcircle_theorem_tangent_circles_theorem_l3812_381286

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (0, 3)
def C : ℝ × ℝ := (0, 0)

-- Define the circumcircle equation
def circumcircle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 3*y = 0

-- Define the circles with center on y-axis and radius 5
def circle_eq_1 (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 25

def circle_eq_2 (x y : ℝ) : Prop :=
  x^2 + (y - 11)^2 = 25

-- Theorem for the circumcircle
theorem circumcircle_theorem :
  circumcircle_eq A.1 A.2 ∧
  circumcircle_eq B.1 B.2 ∧
  circumcircle_eq C.1 C.2 :=
sorry

-- Theorem for the circles tangent to y = 6
theorem tangent_circles_theorem :
  (∃ x y : ℝ, circle_eq_1 x y ∧ y = 6) ∧
  (∃ x y : ℝ, circle_eq_2 x y ∧ y = 6) :=
sorry

end circumcircle_theorem_tangent_circles_theorem_l3812_381286


namespace m_minus_n_equals_three_l3812_381272

-- Define the sets M and N
def M (m : ℕ) : Set ℕ := {1, 2, 3, m}
def N (n : ℕ) : Set ℕ := {4, 7, n^4, n^2 + 3*n}

-- Define the function f
def f (x : ℕ) : ℕ := 3*x + 1

-- State the theorem
theorem m_minus_n_equals_three (m n : ℕ) : 
  (∃ y ∈ M m, ∃ z ∈ N n, f y = z) → m - n = 3 := by
  sorry

end m_minus_n_equals_three_l3812_381272


namespace intersection_points_on_ellipse_l3812_381203

/-- The points of intersection of two parametric lines lie on an ellipse -/
theorem intersection_points_on_ellipse (s : ℝ) : 
  ∃ (a b : ℝ) (h : a > 0 ∧ b > 0), 
    ∀ (x y : ℝ), 
      (s * x - 3 * y - 4 * s = 0 ∧ x - 3 * s * y + 4 = 0) → 
      (x^2 / a^2 + y^2 / b^2 = 1) := by
sorry

end intersection_points_on_ellipse_l3812_381203


namespace starting_lineup_combinations_l3812_381276

def total_players : ℕ := 15
def lineup_size : ℕ := 5
def preselected_players : ℕ := 3

theorem starting_lineup_combinations : 
  Nat.choose (total_players - preselected_players) (lineup_size - preselected_players) = 66 := by
  sorry

end starting_lineup_combinations_l3812_381276


namespace sqrt_28_div_sqrt_7_l3812_381275

theorem sqrt_28_div_sqrt_7 : Real.sqrt 28 / Real.sqrt 7 = 2 := by
  sorry

end sqrt_28_div_sqrt_7_l3812_381275


namespace quadratic_roots_condition_l3812_381200

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ 
   x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) → 
  m > 2 := by
sorry

end quadratic_roots_condition_l3812_381200


namespace min_value_abs_2a_minus_b_l3812_381287

theorem min_value_abs_2a_minus_b (a b : ℝ) (h : 2 * a^2 - b^2 = 1) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (x y : ℝ), 2 * x^2 - y^2 = 1 → |2 * x - y| ≥ m :=
sorry

end min_value_abs_2a_minus_b_l3812_381287
