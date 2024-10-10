import Mathlib

namespace not_general_term_formula_l1673_167306

def alternating_sequence (n : ℕ) : ℤ := (-1)^(n + 1)

theorem not_general_term_formula :
  ∃ n : ℕ, ((-1 : ℤ)^n ≠ alternating_sequence n) ∧
  ((-1 : ℤ)^(n + 1) = alternating_sequence n) ∧
  ((-1 : ℤ)^(n - 1) = alternating_sequence n) ∧
  (if n % 2 = 0 then -1 else 1 : ℤ) = alternating_sequence n :=
sorry

end not_general_term_formula_l1673_167306


namespace quadratic_max_value_l1673_167319

theorem quadratic_max_value :
  let f : ℝ → ℝ := fun x ↦ -3 * x^2 + 6 * x + 4
  ∃ (max : ℝ), ∀ (x : ℝ), f x ≤ max ∧ ∃ (x_max : ℝ), f x_max = max ∧ max = 7 := by
  sorry

end quadratic_max_value_l1673_167319


namespace modulo_eleven_residue_l1673_167357

theorem modulo_eleven_residue : (255 + 6 * 41 + 8 * 154 + 5 * 18) % 11 = 8 := by
  sorry

end modulo_eleven_residue_l1673_167357


namespace sum_first_50_digits_1001_l1673_167364

/-- The decimal expansion of 1/1001 -/
def decimalExpansion1001 : ℕ → ℕ
| n => match n % 6 with
  | 0 => 0
  | 1 => 0
  | 2 => 0
  | 3 => 9
  | 4 => 9
  | 5 => 9
  | _ => 0  -- This case should never occur

/-- The sum of the first n digits in the decimal expansion of 1/1001 -/
def sumFirstNDigits (n : ℕ) : ℕ :=
  (List.range n).map decimalExpansion1001 |> List.sum

/-- Theorem: The sum of the first 50 digits after the decimal point
    in the decimal expansion of 1/1001 is 216 -/
theorem sum_first_50_digits_1001 : sumFirstNDigits 50 = 216 := by
  sorry

end sum_first_50_digits_1001_l1673_167364


namespace x_minus_y_equals_eight_l1673_167310

theorem x_minus_y_equals_eight (x y : ℝ) 
  (hx : x + (-3) = 0) 
  (hy : |y| = 5) 
  (hxy : x * y < 0) : 
  x - y = 8 := by
sorry

end x_minus_y_equals_eight_l1673_167310


namespace first_student_guess_l1673_167398

/-- Represents the number of jellybeans guessed by each student -/
structure JellybeanGuesses where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The conditions of the jellybean guessing problem -/
def jellybean_problem (g : JellybeanGuesses) : Prop :=
  g.second = 8 * g.first ∧
  g.third = g.second - 200 ∧
  g.fourth = (g.first + g.second + g.third) / 3 + 25 ∧
  g.fourth = 525

/-- Theorem stating that the first student's guess is 100 jellybeans -/
theorem first_student_guess :
  ∀ g : JellybeanGuesses, jellybean_problem g → g.first = 100 :=
by
  sorry


end first_student_guess_l1673_167398


namespace total_pebbles_after_fifteen_days_l1673_167327

/-- The number of pebbles collected on the first day -/
def initial_pebbles : ℕ := 3

/-- The daily increase in the number of pebbles collected -/
def daily_increase : ℕ := 2

/-- The number of days Murtha collects pebbles -/
def collection_days : ℕ := 15

/-- The arithmetic sequence of daily pebble collections -/
def pebble_sequence (n : ℕ) : ℕ := initial_pebbles + (n - 1) * daily_increase

/-- The total number of pebbles collected after a given number of days -/
def total_pebbles (n : ℕ) : ℕ := n * (initial_pebbles + pebble_sequence n) / 2

theorem total_pebbles_after_fifteen_days :
  total_pebbles collection_days = 255 := by sorry

end total_pebbles_after_fifteen_days_l1673_167327


namespace mustard_at_first_table_l1673_167328

-- Define the amount of mustard at each table
def mustard_table1 : ℝ := sorry
def mustard_table2 : ℝ := 0.25
def mustard_table3 : ℝ := 0.38

-- Define the total amount of mustard
def total_mustard : ℝ := 0.88

-- Theorem statement
theorem mustard_at_first_table :
  mustard_table1 + mustard_table2 + mustard_table3 = total_mustard →
  mustard_table1 = 0.25 := by
  sorry

end mustard_at_first_table_l1673_167328


namespace triangle_theorem_l1673_167313

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h3 : A + B + C = π)

-- State the theorem
theorem triangle_theorem (t : Triangle) 
  (h : Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)) :
  2 * t.a^2 = t.b^2 + t.c^2 ∧ 
  (t.a = 5 ∧ Real.cos t.A = 25/31 → t.a + t.b + t.c = 14) :=
by sorry

end triangle_theorem_l1673_167313


namespace prob_red_ball_specific_l1673_167338

/-- Represents a bag of colored balls -/
structure ColoredBalls where
  total : ℕ
  red : ℕ
  yellow : ℕ
  green : ℕ
  sum_colors : total = red + yellow + green

/-- The probability of drawing a red ball from a bag of colored balls -/
def prob_red_ball (bag : ColoredBalls) : ℚ :=
  bag.red / bag.total

/-- Theorem: The probability of drawing a red ball from a bag with 15 balls, 
    of which 8 are red, is 8/15 -/
theorem prob_red_ball_specific : 
  ∃ (bag : ColoredBalls), bag.total = 15 ∧ bag.red = 8 ∧ prob_red_ball bag = 8/15 := by
  sorry


end prob_red_ball_specific_l1673_167338


namespace bolzano_weierstrass_unit_interval_l1673_167317

/-- Bolzano-Weierstrass theorem for sequences in [0, 1) -/
theorem bolzano_weierstrass_unit_interval (s : ℕ → ℝ) (h : ∀ n, 0 ≤ s n ∧ s n < 1) :
  (∃ (a : Set ℕ), Set.Infinite a ∧ (∀ n ∈ a, s n < 1/2)) ∨
  (∃ (b : Set ℕ), Set.Infinite b ∧ (∀ n ∈ b, 1/2 ≤ s n)) ∧
  ∀ ε > 0, ε < 1/2 → ∃ α : ℝ, 0 ≤ α ∧ α ≤ 1 ∧
    ∃ (c : Set ℕ), Set.Infinite c ∧ ∀ n ∈ c, |s n - α| < ε :=
by sorry

end bolzano_weierstrass_unit_interval_l1673_167317


namespace tangent_line_of_even_cubic_l1673_167370

/-- Given a real number a, if f(x) = x^3 + ax^2 + (a-3)x is an even function,
    then the equation of the tangent line to y = f(x) at (2, f(2)) is 9x - y - 16 = 0 -/
theorem tangent_line_of_even_cubic (a : ℝ) : 
  (∀ x, (x^3 + a*x^2 + (a-3)*x) = ((- x)^3 + a*(- x)^2 + (a-3)*(- x))) →
  ∃ m b, (m * 2 + b = 2^3 + a*2^2 + (a-3)*2) ∧ 
         (∀ x y, y = x^3 + a*x^2 + (a-3)*x → m*x - y - b = 0) ∧
         (m = 9 ∧ b = 16) := by
  sorry

end tangent_line_of_even_cubic_l1673_167370


namespace equation_solution_l1673_167334

theorem equation_solution :
  ∃! (x : ℝ), x ≠ 0 ∧ (7 * x)^5 = (14 * x)^4 ∧ x = 16/7 := by sorry

end equation_solution_l1673_167334


namespace power_tower_at_three_l1673_167397

theorem power_tower_at_three : (3^3)^(3^(3^3)) = 27^(3^27) := by sorry

end power_tower_at_three_l1673_167397


namespace fourth_side_length_l1673_167361

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The lengths of the four sides of the quadrilateral -/
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- The theorem statement -/
theorem fourth_side_length
  (q : InscribedQuadrilateral)
  (h1 : q.radius = 150)
  (h2 : q.side1 = 200)
  (h3 : q.side2 = 200)
  (h4 : q.side3 = 100) :
  q.side4 = 300 := by
  sorry


end fourth_side_length_l1673_167361


namespace perpendicular_segments_equal_length_l1673_167337

/-- Two lines in a plane are parallel if they do not intersect. -/
def parallel (l₁ l₂ : Set (ℝ × ℝ)) : Prop := ∀ p, p ∈ l₁ → p ∉ l₂

/-- A line segment is perpendicular to a line if it forms a right angle with the line. -/
def perpendicular (seg : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : Prop := sorry

/-- The length of a line segment. -/
def length (seg : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: All perpendicular line segments between two parallel lines are equal in length. -/
theorem perpendicular_segments_equal_length 
  (l₁ l₂ : Set (ℝ × ℝ)) 
  (h_parallel : parallel l₁ l₂) 
  (seg₁ seg₂ : Set (ℝ × ℝ)) 
  (h_perp₁ : perpendicular seg₁ l₁ ∧ perpendicular seg₁ l₂)
  (h_perp₂ : perpendicular seg₂ l₁ ∧ perpendicular seg₂ l₂) :
  length seg₁ = length seg₂ :=
sorry

end perpendicular_segments_equal_length_l1673_167337


namespace clothes_transport_equals_savings_l1673_167326

/-- Represents Mr. Yadav's monthly financial breakdown --/
structure MonthlyFinances where
  salary : ℝ
  consumable_rate : ℝ
  clothes_transport_rate : ℝ
  savings_rate : ℝ

/-- Calculates the yearly savings based on monthly finances --/
def yearly_savings (m : MonthlyFinances) : ℝ :=
  12 * m.savings_rate * m.salary

/-- Theorem stating that the monthly amount spent on clothes and transport
    is equal to the monthly savings --/
theorem clothes_transport_equals_savings
  (m : MonthlyFinances)
  (h1 : m.consumable_rate = 0.6)
  (h2 : m.clothes_transport_rate = 0.5 * (1 - m.consumable_rate))
  (h3 : m.savings_rate = 1 - m.consumable_rate - m.clothes_transport_rate)
  (h4 : yearly_savings m = 48456) :
  m.clothes_transport_rate * m.salary = m.savings_rate * m.salary :=
by sorry

end clothes_transport_equals_savings_l1673_167326


namespace zoo_field_trip_l1673_167304

theorem zoo_field_trip (students_class1 students_class2 parent_chaperones : ℕ)
  (students_left chaperones_left remaining : ℕ) :
  students_class1 = 10 →
  students_class2 = 10 →
  parent_chaperones = 5 →
  students_left = 10 →
  chaperones_left = 2 →
  remaining = 15 →
  ∃ (teachers : ℕ),
    teachers = 2 ∧
    (students_class1 + students_class2 + parent_chaperones + teachers) -
    (students_left + chaperones_left) = remaining :=
by sorry

end zoo_field_trip_l1673_167304


namespace cantaloupes_sum_l1673_167395

/-- The number of cantaloupes grown by Fred -/
def fred_cantaloupes : ℕ := 38

/-- The number of cantaloupes grown by Tim -/
def tim_cantaloupes : ℕ := 44

/-- The total number of cantaloupes grown by Fred and Tim -/
def total_cantaloupes : ℕ := fred_cantaloupes + tim_cantaloupes

theorem cantaloupes_sum : total_cantaloupes = 82 := by
  sorry

end cantaloupes_sum_l1673_167395


namespace mall_a_better_deal_l1673_167316

/-- Calculates the discount for a given spent amount and promotion rule -/
def calculate_discount (spent : ℕ) (promotion_threshold : ℕ) (promotion_discount : ℕ) : ℕ :=
  (spent / promotion_threshold) * promotion_discount

/-- Calculates the final cost after applying the discount -/
def calculate_final_cost (total : ℕ) (discount : ℕ) : ℕ :=
  total - discount

theorem mall_a_better_deal (shoes_price : ℕ) (sweater_price : ℕ)
    (h_shoes : shoes_price = 699)
    (h_sweater : sweater_price = 910)
    (mall_a_threshold : ℕ) (mall_a_discount : ℕ)
    (mall_b_threshold : ℕ) (mall_b_discount : ℕ)
    (h_mall_a : mall_a_threshold = 200 ∧ mall_a_discount = 101)
    (h_mall_b : mall_b_threshold = 101 ∧ mall_b_discount = 50) :
    let total := shoes_price + sweater_price
    let discount_a := calculate_discount total mall_a_threshold mall_a_discount
    let discount_b := calculate_discount total mall_b_threshold mall_b_discount
    let final_cost_a := calculate_final_cost total discount_a
    let final_cost_b := calculate_final_cost total discount_b
    final_cost_a < final_cost_b ∧ final_cost_a = 801 := by
  sorry

end mall_a_better_deal_l1673_167316


namespace largest_packet_size_l1673_167323

theorem largest_packet_size (jonathan_sets elena_sets : ℕ) 
  (h1 : jonathan_sets = 36) 
  (h2 : elena_sets = 60) : 
  Nat.gcd jonathan_sets elena_sets = 12 := by
  sorry

end largest_packet_size_l1673_167323


namespace ternary_121_equals_16_l1673_167393

/-- Converts a ternary (base 3) number to decimal (base 10) --/
def ternary_to_decimal (a b c : ℕ) : ℕ :=
  a * 3^2 + b * 3^1 + c * 3^0

/-- The ternary number 121₃ is equal to 16 in decimal (base 10) --/
theorem ternary_121_equals_16 : ternary_to_decimal 1 2 1 = 16 := by
  sorry

end ternary_121_equals_16_l1673_167393


namespace sum_of_coefficients_l1673_167384

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 10 := by
sorry

end sum_of_coefficients_l1673_167384


namespace special_square_pt_length_l1673_167356

/-- Square with side length √2 and special folding property -/
structure SpecialSquare where
  -- Square side length
  side : ℝ
  side_eq : side = Real.sqrt 2
  -- Points T and U on sides PQ and RQ
  t : ℝ
  u : ℝ
  t_range : 0 < t ∧ t < side
  u_range : 0 < u ∧ u < side
  -- PT = QU
  pt_eq_qu : t = u
  -- Folding property: PS and RS coincide on diagonal QS
  folding : t * Real.sqrt 2 = side

/-- The length of PT in a SpecialSquare can be expressed as √8 - 2 -/
theorem special_square_pt_length (s : SpecialSquare) : s.t = Real.sqrt 8 - 2 := by
  sorry

#check special_square_pt_length

end special_square_pt_length_l1673_167356


namespace cookie_batches_l1673_167388

/-- The number of batches of cookies made from one bag of chocolate chips -/
def num_batches (chips_per_cookie : ℕ) (chips_per_bag : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  chips_per_bag / (chips_per_cookie * cookies_per_batch)

/-- Theorem: The number of batches of cookies made from one bag of chocolate chips is 3 -/
theorem cookie_batches :
  num_batches 9 81 3 = 3 :=
by
  sorry

end cookie_batches_l1673_167388


namespace quadratic_rational_root_implies_even_coeff_l1673_167396

theorem quadratic_rational_root_implies_even_coeff 
  (a b c : ℤ) (h_a_nonzero : a ≠ 0) 
  (h_rational_root : ∃ (p q : ℤ) (h_q_nonzero : q ≠ 0), a * (p * p) + b * (p * q) + c * (q * q) = 0) :
  Even a ∨ Even b ∨ Even c := by
sorry

end quadratic_rational_root_implies_even_coeff_l1673_167396


namespace apples_used_correct_l1673_167344

/-- The number of apples used to make lunch in the school cafeteria -/
def apples_used : ℕ := 20

/-- The initial number of apples in the cafeteria -/
def initial_apples : ℕ := 23

/-- The number of apples bought after making lunch -/
def apples_bought : ℕ := 6

/-- The final number of apples in the cafeteria -/
def final_apples : ℕ := 9

/-- Theorem stating that the number of apples used for lunch is correct -/
theorem apples_used_correct : 
  initial_apples - apples_used + apples_bought = final_apples :=
by sorry

end apples_used_correct_l1673_167344


namespace interest_less_than_principal_l1673_167348

/-- Calculates the simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Calculates the difference between principal and interest -/
def interest_difference (principal : ℝ) (interest : ℝ) : ℝ :=
  principal - interest

theorem interest_less_than_principal : 
  let principal : ℝ := 400.00000000000006
  let rate : ℝ := 0.04
  let time : ℝ := 8
  let interest := simple_interest principal rate time
  interest_difference principal interest = 272 := by
  sorry

end interest_less_than_principal_l1673_167348


namespace log_sum_approximation_l1673_167386

theorem log_sum_approximation : 
  let x := Real.log 3 / Real.log 10 + 3 * Real.log 4 / Real.log 10 + 
           2 * Real.log 5 / Real.log 10 + 4 * Real.log 2 / Real.log 10 + 
           Real.log 9 / Real.log 10
  ∃ ε > 0, |x - 5.8399| < ε :=
by sorry

end log_sum_approximation_l1673_167386


namespace find_divisor_l1673_167376

theorem find_divisor (x : ℕ) (h_x : x = 75) :
  ∃ D : ℕ,
    (∃ Q R : ℕ, x = D * Q + R ∧ R < D ∧ Q = (x % 34) + 8) ∧
    (∀ D' : ℕ, D' < D → ¬(∃ Q' R' : ℕ, x = D' * Q' + R' ∧ R' < D' ∧ Q' = (x % 34) + 8)) ∧
    D = 5 := by
  sorry

end find_divisor_l1673_167376


namespace sqrt_expression_equality_l1673_167346

theorem sqrt_expression_equality : 
  (Real.sqrt 24 - Real.sqrt 6) / Real.sqrt 3 + Real.sqrt (1/2) = (3 * Real.sqrt 2) / 2 := by
  sorry

end sqrt_expression_equality_l1673_167346


namespace f_properties_l1673_167303

def f_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)

theorem f_properties (f : ℝ → ℝ) (h : f_property f) :
  (∀ x : ℝ, f (f x) ≤ 0) ∧
  (f 0 ≥ 0 → ∀ x : ℝ, f x = 0) :=
by sorry

end f_properties_l1673_167303


namespace floor_sqrt_80_l1673_167362

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by sorry

end floor_sqrt_80_l1673_167362


namespace two_digit_number_reverse_sum_l1673_167312

theorem two_digit_number_reverse_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 7 * (a + b) →
  (10 * a + b) + (10 * b + a) = 99 := by
sorry

end two_digit_number_reverse_sum_l1673_167312


namespace euler_line_equation_l1673_167399

/-- Triangle ABC with vertices A(3,1), B(4,2), and C(2,3) -/
structure Triangle where
  A : ℝ × ℝ := (3, 1)
  B : ℝ × ℝ := (4, 2)
  C : ℝ × ℝ := (2, 3)

/-- The Euler line of a triangle -/
def EulerLine (t : Triangle) : ℝ → ℝ → Prop :=
  fun x y => x + y - 5 = 0

/-- Theorem: The equation of the Euler line for the given triangle ABC is x + y - 5 = 0 -/
theorem euler_line_equation (t : Triangle) : EulerLine t = fun x y => x + y - 5 = 0 := by
  sorry

end euler_line_equation_l1673_167399


namespace albert_pizza_count_l1673_167350

def pizza_problem (large_pizzas small_pizzas : ℕ) 
  (slices_per_large slices_per_small total_slices : ℕ) : Prop :=
  large_pizzas = 2 ∧ 
  slices_per_large = 16 ∧ 
  slices_per_small = 8 ∧ 
  total_slices = 48 ∧
  small_pizzas * slices_per_small = total_slices - (large_pizzas * slices_per_large)

theorem albert_pizza_count : 
  ∃ (large_pizzas small_pizzas slices_per_large slices_per_small total_slices : ℕ),
    pizza_problem large_pizzas small_pizzas slices_per_large slices_per_small total_slices ∧ 
    small_pizzas = 2 := by
  sorry

end albert_pizza_count_l1673_167350


namespace largest_x_abs_value_equation_l1673_167372

theorem largest_x_abs_value_equation : 
  (∃ (x : ℝ), |x - 8| = 15 ∧ ∀ (y : ℝ), |y - 8| = 15 → y ≤ x) → 
  (∃ (x : ℝ), |x - 8| = 15 ∧ ∀ (y : ℝ), |y - 8| = 15 → y ≤ 23) :=
by sorry

end largest_x_abs_value_equation_l1673_167372


namespace stratified_sample_imported_count_l1673_167332

/-- Represents the number of marker lights in a population -/
structure MarkerLightPopulation where
  total : ℕ
  coDeveloped : ℕ
  domestic : ℕ
  h_sum : total = imported + coDeveloped + domestic

/-- Represents a stratified sample of marker lights -/
structure StratifiedSample where
  populationSize : ℕ
  sampleSize : ℕ

/-- Theorem stating that the number of imported marker lights in a stratified sample
    is proportional to their representation in the population -/
theorem stratified_sample_imported_count 
  (population : MarkerLightPopulation)
  (sample : StratifiedSample)
  (h_pop_size : sample.populationSize = population.total)
  (h_imported : sample.importedInPopulation = population.imported)
  (h_sample_size : sample.sampleSize = 20)
  (h_stratified : sample.importedInSample * population.total = 
                  population.imported * sample.sampleSize) :
  sample.importedInSample = 2 := by
  sorry

end stratified_sample_imported_count_l1673_167332


namespace initial_amount_correct_l1673_167315

/-- The amount of money John initially gave when buying barbells -/
def initial_amount : ℕ := 850

/-- The number of barbells John bought -/
def num_barbells : ℕ := 3

/-- The cost of each barbell in dollars -/
def barbell_cost : ℕ := 270

/-- The amount of change John received in dollars -/
def change_received : ℕ := 40

/-- Theorem stating that the initial amount John gave is correct -/
theorem initial_amount_correct : 
  initial_amount = num_barbells * barbell_cost + change_received :=
by sorry

end initial_amount_correct_l1673_167315


namespace fraction_simplification_l1673_167354

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) : (x - y) / (y - x) = -1 := by
  sorry

end fraction_simplification_l1673_167354


namespace taxi_ride_cost_l1673_167387

theorem taxi_ride_cost (uber_cost lyft_cost taxi_cost tip_percentage : ℝ) : 
  uber_cost = lyft_cost + 3 →
  lyft_cost = taxi_cost + 4 →
  uber_cost = 22 →
  tip_percentage = 0.2 →
  taxi_cost + (tip_percentage * taxi_cost) = 18 := by
sorry

end taxi_ride_cost_l1673_167387


namespace complement_intersection_equals_set_l1673_167330

def U : Finset ℕ := {1, 2, 3, 4, 6}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4}

theorem complement_intersection_equals_set : (U \ (A ∩ B)) = {1, 4, 6} := by sorry

end complement_intersection_equals_set_l1673_167330


namespace min_p_plus_q_l1673_167390

theorem min_p_plus_q (p q : ℕ+) (h : 162 * p = q^3) : 
  ∀ (p' q' : ℕ+), 162 * p' = q'^3 → p + q ≤ p' + q' :=
sorry

end min_p_plus_q_l1673_167390


namespace students_not_in_biology_l1673_167351

theorem students_not_in_biology (total_students : ℕ) (enrolled_percentage : ℚ) 
  (h1 : total_students = 880)
  (h2 : enrolled_percentage = 40 / 100) :
  (total_students : ℚ) * (1 - enrolled_percentage) = 528 := by
  sorry

end students_not_in_biology_l1673_167351


namespace largest_stamps_per_page_l1673_167300

theorem largest_stamps_per_page (book1_stamps book2_stamps : ℕ) 
  (h1 : book1_stamps = 924) 
  (h2 : book2_stamps = 1386) : 
  Nat.gcd book1_stamps book2_stamps = 462 := by
  sorry

end largest_stamps_per_page_l1673_167300


namespace valid_numbers_characterization_l1673_167374

def is_valid (n : ℕ) : Prop :=
  n ≥ 10 ∧ (100 * (n / 10) + n % 10) % n = 0

def S : Set ℕ := {10, 20, 30, 40, 50, 60, 70, 80, 90, 15, 18, 45}

theorem valid_numbers_characterization :
  ∀ n : ℕ, is_valid n ↔ n ∈ S :=
sorry

end valid_numbers_characterization_l1673_167374


namespace quadratic_discriminant_l1673_167373

/-- The discriminant of a quadratic equation ax² + bx + c is equal to b² - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x² + (5 + 1/5)x + 1/5 -/
def a : ℚ := 5
def b : ℚ := 5 + 1/5
def c : ℚ := 1/5

theorem quadratic_discriminant : discriminant a b c = 576/25 := by
  sorry

end quadratic_discriminant_l1673_167373


namespace scenario_one_registration_methods_scenario_two_registration_methods_scenario_three_registration_methods_l1673_167383

/- Define the number of students and events -/
def num_students : ℕ := 6
def num_events : ℕ := 3

/- Theorem for scenario 1 -/
theorem scenario_one_registration_methods :
  (num_events ^ num_students : ℕ) = 729 := by sorry

/- Theorem for scenario 2 -/
theorem scenario_two_registration_methods :
  (num_students * (num_students - 1) * (num_students - 2) : ℕ) = 120 := by sorry

/- Theorem for scenario 3 -/
theorem scenario_three_registration_methods :
  (num_students ^ num_events : ℕ) = 216 := by sorry

end scenario_one_registration_methods_scenario_two_registration_methods_scenario_three_registration_methods_l1673_167383


namespace arithmetic_mean_of_reciprocals_first_five_primes_l1673_167321

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

def reciprocals (lst : List Nat) : List Rat :=
  lst.map (λ x => 1 / x)

def arithmetic_mean (lst : List Rat) : Rat :=
  lst.sum / lst.length

theorem arithmetic_mean_of_reciprocals_first_five_primes :
  arithmetic_mean (reciprocals first_five_primes) = 2927 / 11550 := by
  sorry

end arithmetic_mean_of_reciprocals_first_five_primes_l1673_167321


namespace cube_volume_from_face_perimeter_l1673_167305

/-- Given a cube with face perimeter 20 cm, its volume is 125 cubic centimeters. -/
theorem cube_volume_from_face_perimeter :
  ∀ (cube : ℝ → ℝ), 
  (∃ (side : ℝ), side > 0 ∧ 4 * side = 20) →
  cube (20 / 4) = 125 :=
by
  sorry

end cube_volume_from_face_perimeter_l1673_167305


namespace trig_expression_equality_l1673_167307

theorem trig_expression_equality : 
  (Real.sin (30 * π / 180) * Real.cos (24 * π / 180) + 
   Real.cos (150 * π / 180) * Real.cos (84 * π / 180)) / 
  (Real.sin (34 * π / 180) * Real.cos (16 * π / 180) + 
   Real.cos (146 * π / 180) * Real.cos (76 * π / 180)) = 
  Real.sin (51 * π / 180) / Real.sin (55 * π / 180) := by
sorry

end trig_expression_equality_l1673_167307


namespace cubic_function_properties_l1673_167368

/-- A cubic function with specific properties -/
def f (a c d : ℝ) (x : ℝ) : ℝ := a * x^3 + c * x + d

theorem cubic_function_properties (a c d : ℝ) (h_a : a ≠ 0) :
  (∀ x, f a c d x = -f a c d (-x)) →  -- f is odd
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a c d 1 ≤ f a c d x) →  -- f(1) is an extreme value
  (f a c d 1 = -2) →  -- f(1) = -2
  (∀ x, f a c d x = x^3 - 3*x) ∧  -- f(x) = x^3 - 3x
  (∀ x, f a c d x ≤ 2)  -- maximum value is 2
  := by sorry

end cubic_function_properties_l1673_167368


namespace percentage_less_l1673_167366

theorem percentage_less (w e y z : ℝ) 
  (hw : w = 0.60 * e) 
  (hz : z = 0.54 * y) 
  (hzw : z = 1.5000000000000002 * w) : 
  e = 0.60 * y := by
sorry

end percentage_less_l1673_167366


namespace seating_arrangement_theorem_l1673_167320

/-- Represents the seating position of the k-th person -/
def seat_position (n k : ℕ) : ℕ := (k * (k - 1) / 2) % n

/-- Checks if all seating positions are distinct -/
def all_distinct_positions (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → seat_position n i ≠ seat_position n j

/-- Checks if a number is a power of 2 -/
def is_power_of_two (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 2^m

theorem seating_arrangement_theorem (n : ℕ) :
  n > 0 → (all_distinct_positions n ↔ is_power_of_two n) :=
by sorry

end seating_arrangement_theorem_l1673_167320


namespace sum_of_extrema_l1673_167308

theorem sum_of_extrema (x y : ℝ) (h : 1 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 4) :
  ∃ (min max : ℝ),
    (∀ z w : ℝ, 1 ≤ z^2 + w^2 ∧ z^2 + w^2 ≤ 4 → min ≤ z^2 - z*w + w^2 ∧ z^2 - z*w + w^2 ≤ max) ∧
    min + max = 13/2 :=
by sorry

end sum_of_extrema_l1673_167308


namespace factorization_equality_l1673_167345

theorem factorization_equality (x y : ℝ) : 
  x^2 - y^2 + 3*x - y + 2 = (x + y + 2)*(x - y + 1) := by
  sorry

end factorization_equality_l1673_167345


namespace complement_A_intersect_B_when_a_is_2_A_intersect_B_equals_B_iff_a_less_than_0_l1673_167360

-- Define sets A and B
def A : Set ℝ := {x | x < -3 ∨ x ≥ 2}
def B (a : ℝ) : Set ℝ := {x | x ≤ a - 3}

-- Theorem 1
theorem complement_A_intersect_B_when_a_is_2 :
  (Set.univ \ A) ∩ B 2 = {x | -3 ≤ x ∧ x ≤ -1} := by sorry

-- Theorem 2
theorem A_intersect_B_equals_B_iff_a_less_than_0 (a : ℝ) :
  A ∩ B a = B a ↔ a < 0 := by sorry

end complement_A_intersect_B_when_a_is_2_A_intersect_B_equals_B_iff_a_less_than_0_l1673_167360


namespace quadratic_congruence_solutions_l1673_167341

theorem quadratic_congruence_solutions (x : ℕ) : 
  (x^2 + x - 6) % 143 = 0 ↔ x ∈ ({2, 41, 101, 140} : Set ℕ) := by
  sorry

end quadratic_congruence_solutions_l1673_167341


namespace extra_flowers_l1673_167340

theorem extra_flowers (tulips roses used : ℕ) : 
  tulips = 36 → roses = 37 → used = 70 → tulips + roses - used = 3 := by
  sorry

end extra_flowers_l1673_167340


namespace maggi_ate_five_cupcakes_l1673_167311

/-- Calculates the number of cupcakes Maggi ate -/
def cupcakes_eaten (initial_packages : ℕ) (cupcakes_per_package : ℕ) (cupcakes_left : ℕ) : ℕ :=
  initial_packages * cupcakes_per_package - cupcakes_left

/-- Proves that Maggi ate 5 cupcakes -/
theorem maggi_ate_five_cupcakes :
  cupcakes_eaten 3 4 7 = 5 := by
  sorry

end maggi_ate_five_cupcakes_l1673_167311


namespace weight_of_pecans_l1673_167381

/-- Given the total weight of nuts and the weight of almonds, calculate the weight of pecans. -/
theorem weight_of_pecans (total_weight : ℝ) (almond_weight : ℝ) 
  (h1 : total_weight = 0.52) 
  (h2 : almond_weight = 0.14) : 
  total_weight - almond_weight = 0.38 := by
  sorry

end weight_of_pecans_l1673_167381


namespace isabel_bouquets_l1673_167355

def flowers_to_bouquets (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) : ℕ :=
  (initial_flowers - wilted_flowers) / flowers_per_bouquet

theorem isabel_bouquets :
  flowers_to_bouquets 66 8 10 = 7 := by
  sorry

end isabel_bouquets_l1673_167355


namespace smallest_sum_reciprocals_l1673_167325

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ) :=
by sorry

end smallest_sum_reciprocals_l1673_167325


namespace sum_of_four_numbers_l1673_167324

theorem sum_of_four_numbers : 4321 + 3214 + 2143 + 1432 = 11110 := by
  sorry

end sum_of_four_numbers_l1673_167324


namespace monica_reading_plan_l1673_167329

/-- The number of books Monica read last year -/
def books_last_year : ℕ := 16

/-- The number of books Monica read this year -/
def books_this_year : ℕ := 2 * books_last_year

/-- The number of books Monica will read next year -/
def books_next_year : ℕ := 2 * books_this_year + 5

theorem monica_reading_plan : books_next_year = 69 := by
  sorry

end monica_reading_plan_l1673_167329


namespace maria_coffee_order_l1673_167359

-- Define the variables
def visits_per_day : ℕ := 2
def total_cups_per_day : ℕ := 6

-- Define the function to calculate cups per visit
def cups_per_visit : ℕ := total_cups_per_day / visits_per_day

-- Theorem statement
theorem maria_coffee_order :
  cups_per_visit = 3 :=
by
  sorry

end maria_coffee_order_l1673_167359


namespace hyperbola_eccentricity_is_sqrt3_l1673_167353

noncomputable def hyperbola_eccentricity (a : ℝ) : ℝ := 
  let b : ℝ := Real.sqrt 8
  let c : ℝ := Real.sqrt (a^2 + b^2)
  c / a

theorem hyperbola_eccentricity_is_sqrt3 :
  ∃ (a : ℝ), a > 0 ∧ 
  (∃ (x y : ℝ), x = y ∧ x^2/a^2 - y^2/8 = 1) ∧
  (∃ (x1 x2 : ℝ), x1 * x2 = -8 ∧ 
    x1^2/a^2 - x1^2/8 = 1 ∧ 
    x2^2/a^2 - x2^2/8 = 1) ∧
  hyperbola_eccentricity a = Real.sqrt 3 :=
by sorry

end hyperbola_eccentricity_is_sqrt3_l1673_167353


namespace expression_evaluation_l1673_167331

theorem expression_evaluation (x y z : ℝ) : (x + (y + z)) - ((-x + y) + z) = 2 * x := by
  sorry

end expression_evaluation_l1673_167331


namespace cube_sum_of_three_numbers_l1673_167336

theorem cube_sum_of_three_numbers (x y z : ℝ) 
  (sum_eq : x + y + z = 4)
  (sum_products_eq : x*y + x*z + y*z = 3)
  (product_eq : x*y*z = -10) :
  x^3 + y^3 + z^3 = 10 := by
sorry

end cube_sum_of_three_numbers_l1673_167336


namespace gold_coins_percentage_l1673_167382

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  beads : ℝ
  sculptures : ℝ
  coins : ℝ
  silverCoins : ℝ
  goldCoins : ℝ

/-- Theorem stating the percentage of gold coins in the urn -/
theorem gold_coins_percentage (u : UrnComposition) 
  (beads_percent : u.beads = 0.3)
  (sculptures_percent : u.sculptures = 0.1)
  (total_percent : u.beads + u.sculptures + u.coins = 1)
  (silver_coins_percent : u.silverCoins = 0.3 * u.coins)
  (coins_composition : u.silverCoins + u.goldCoins = u.coins) : 
  u.goldCoins = 0.42 := by
  sorry


end gold_coins_percentage_l1673_167382


namespace constant_radius_is_cylinder_l1673_167333

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) (c : ℝ) : Prop :=
  ∀ p : CylindricalPoint, p ∈ S ↔ p.r = c

/-- The set of points satisfying r = c -/
def ConstantRadiusSet (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = c}

/-- Theorem: The set of points satisfying r = c forms a cylinder -/
theorem constant_radius_is_cylinder (c : ℝ) :
    IsCylinder (ConstantRadiusSet c) c := by
  sorry


end constant_radius_is_cylinder_l1673_167333


namespace three_families_ten_lines_form_150_triangles_l1673_167339

/-- Represents a family of parallel lines -/
structure LineFamily :=
  (count : ℕ)

/-- Calculates the maximum number of triangles formed by three families of parallel lines -/
def max_triangles (f1 f2 f3 : LineFamily) : ℕ :=
  sorry

/-- Theorem stating that three families of 10 parallel lines form 150 triangles -/
theorem three_families_ten_lines_form_150_triangles :
  ∀ (f1 f2 f3 : LineFamily),
    f1.count = 10 → f2.count = 10 → f3.count = 10 →
    max_triangles f1 f2 f3 = 150 :=
by sorry

end three_families_ten_lines_form_150_triangles_l1673_167339


namespace largest_four_digit_congruent_to_17_mod_26_l1673_167358

theorem largest_four_digit_congruent_to_17_mod_26 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n ≡ 17 [ZMOD 26] → n ≤ 9972 :=
by sorry

end largest_four_digit_congruent_to_17_mod_26_l1673_167358


namespace cubic_roots_product_l1673_167369

theorem cubic_roots_product (α₁ α₂ α₃ : ℂ) : 
  (5 * α₁^3 - 6 * α₁^2 + 7 * α₁ + 8 = 0) ∧ 
  (5 * α₂^3 - 6 * α₂^2 + 7 * α₂ + 8 = 0) ∧ 
  (5 * α₃^3 - 6 * α₃^2 + 7 * α₃ + 8 = 0) →
  (α₁^2 + α₁*α₂ + α₂^2) * (α₂^2 + α₂*α₃ + α₃^2) * (α₁^2 + α₁*α₃ + α₃^2) = 764/625 := by
  sorry

end cubic_roots_product_l1673_167369


namespace simplify_fraction_l1673_167375

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by sorry

end simplify_fraction_l1673_167375


namespace quadratic_equation_solutions_linear_quadratic_equation_solutions_l1673_167314

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x => 2 * x^2 + 4 * x - 6
  (f 1 = 0 ∧ f (-3) = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = -3) :=
sorry

theorem linear_quadratic_equation_solutions :
  let g : ℝ → ℝ := λ x => 2 * (x - 3) - 3 * x * (x - 3)
  (g 3 = 0 ∧ g (2/3) = 0) ∧
  (∀ x : ℝ, g x = 0 → x = 3 ∨ x = 2/3) :=
sorry

end quadratic_equation_solutions_linear_quadratic_equation_solutions_l1673_167314


namespace tangent_point_coordinates_l1673_167342

theorem tangent_point_coordinates (f : ℝ → ℝ) (h : f = λ x ↦ Real.exp x) :
  ∃ (x y : ℝ), x = 1 ∧ y = Real.exp 1 ∧
  (∀ t : ℝ, f t = Real.exp t) ∧
  (∃ m : ℝ, ∀ t : ℝ, y - f x = m * (t - x) ∧ 0 = m * (-x)) :=
sorry

end tangent_point_coordinates_l1673_167342


namespace trees_per_square_meter_l1673_167309

/-- Given a forest and a square-shaped street, calculate the number of trees per square meter in the forest. -/
theorem trees_per_square_meter
  (street_side : ℝ)
  (forest_area_multiplier : ℝ)
  (total_trees : ℕ)
  (h1 : street_side = 100)
  (h2 : forest_area_multiplier = 3)
  (h3 : total_trees = 120000) :
  (total_trees : ℝ) / (forest_area_multiplier * street_side^2) = 4 := by
sorry

end trees_per_square_meter_l1673_167309


namespace two_xy_equals_seven_l1673_167389

theorem two_xy_equals_seven (x y : ℝ) 
  (h1 : (8 : ℝ)^x / (2 : ℝ)^(x+y) = 64)
  (h2 : (9 : ℝ)^(x+y) / (3 : ℝ)^(4*y) = 243) :
  2 * x * y = 7 := by
  sorry

end two_xy_equals_seven_l1673_167389


namespace slope_greater_than_one_line_passes_through_point_distance_not_sqrt_two_not_four_lines_l1673_167301

-- Define the line equations
def line1 (x y : ℝ) : Prop := 5 * x - 4 * y + 1 = 0
def line2 (m x y : ℝ) : Prop := (2 + m) * x + 4 * y - 2 + m = 0
def line3 (x y : ℝ) : Prop := x + y - 1 = 0
def line4 (x y : ℝ) : Prop := 2 * x + 2 * y + 1 = 0

-- Define points
def point_A : ℝ × ℝ := (-1, 2)
def point_B : ℝ × ℝ := (3, -1)

-- Statement 1
theorem slope_greater_than_one : 
  ∃ m : ℝ, (∀ x y : ℝ, line1 x y → y = m * x + (1/4)) ∧ m > 1 := by sorry

-- Statement 2
theorem line_passes_through_point :
  ∀ m : ℝ, line2 m (-1) 1 := by sorry

-- Statement 3
theorem distance_not_sqrt_two :
  ∃ d : ℝ, (d = (|1 + 2|) / Real.sqrt (2^2 + 2^2)) ∧ d ≠ Real.sqrt 2 := by sorry

-- Statement 4
theorem not_four_lines :
  ¬(∃ (lines : Finset (ℝ → ℝ → Prop)), lines.card = 4 ∧
    (∀ l ∈ lines, ∃ d1 d2 : ℝ, d1 = 1 ∧ d2 = 4 ∧
      (∀ x y : ℝ, l x y → 
        (Real.sqrt ((x - point_A.1)^2 + (y - point_A.2)^2) = d1 ∧
         Real.sqrt ((x - point_B.1)^2 + (y - point_B.2)^2) = d2)))) := by sorry

end slope_greater_than_one_line_passes_through_point_distance_not_sqrt_two_not_four_lines_l1673_167301


namespace polygon_angle_sum_l1673_167343

theorem polygon_angle_sum (n : ℕ) (A : ℝ) (h1 : n ≥ 3) (h2 : A > 0) :
  (n - 2) * 180 = A + 2460 →
  A = 60 := by
  sorry

end polygon_angle_sum_l1673_167343


namespace point_b_location_l1673_167367

/-- Represents a point on the number line -/
structure Point where
  value : ℝ

/-- The distance between two points on the number line -/
def distance (p q : Point) : ℝ := |p.value - q.value|

theorem point_b_location (a b : Point) :
  a.value = -2 ∧ distance a b = 3 → b.value = -5 ∨ b.value = 1 := by
  sorry

end point_b_location_l1673_167367


namespace soccer_team_selection_l1673_167392

/-- The number of ways to choose an ordered selection of 5 players from a team of 15 players -/
def choose_squad (team_size : Nat) : Nat :=
  team_size * (team_size - 1) * (team_size - 2) * (team_size - 3) * (team_size - 4)

/-- Theorem stating that choosing 5 players from a team of 15 results in 360,360 possibilities -/
theorem soccer_team_selection :
  choose_squad 15 = 360360 := by
  sorry

end soccer_team_selection_l1673_167392


namespace current_speed_l1673_167302

/-- Proves that given a woman swimming downstream 81 km in 9 hours and upstream 36 km in 9 hours, the speed of the current is 2.5 km/h. -/
theorem current_speed (v : ℝ) (c : ℝ) : 
  (v + c) * 9 = 81 → 
  (v - c) * 9 = 36 → 
  c = 2.5 := by
sorry

end current_speed_l1673_167302


namespace tangent_line_at_2_when_a_1_monotonic_intervals_f_geq_2ln_x_iff_l1673_167335

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + (a - 2) / x + 2 - 2 * a

-- State the theorems to be proved
theorem tangent_line_at_2_when_a_1 :
  ∀ x y : ℝ, f 1 2 = 3/2 ∧ (5 * x - 4 * y - 4 = 0 ↔ y - 3/2 = 5/4 * (x - 2)) :=
sorry

theorem monotonic_intervals (a : ℝ) (h : a > 0) :
  (0 < a ∧ a ≤ 2 → 
    StrictMono (f a) ∧ 
    StrictMonoOn (f a) (Set.Ioi 0)) ∧
  (a > 2 → 
    (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧
      StrictMonoOn (f a) (Set.Iic x₁) ∧
      StrictAntiOn (f a) (Set.Ioc x₁ 0) ∧
      StrictAntiOn (f a) (Set.Ioc 0 x₂) ∧
      StrictMonoOn (f a) (Set.Ioi x₂))) :=
sorry

theorem f_geq_2ln_x_iff (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≥ 2 * Real.log x) ↔ a ≥ 1 :=
sorry

end

end tangent_line_at_2_when_a_1_monotonic_intervals_f_geq_2ln_x_iff_l1673_167335


namespace lcm_inequality_l1673_167352

theorem lcm_inequality (m n : ℕ) (h1 : 0 < m) (h2 : m < n) :
  Nat.lcm m n + Nat.lcm (m + 1) (n + 1) ≥ 2 * m * Real.sqrt n :=
by sorry

end lcm_inequality_l1673_167352


namespace inverse_f_at_8_l1673_167377

def f (x : ℝ) : ℝ := 1 - 3*(x - 1) + 3*(x - 1)^2 - (x - 1)^3

theorem inverse_f_at_8 : f 0 = 8 := by
  sorry

end inverse_f_at_8_l1673_167377


namespace max_value_ahn_operation_l1673_167322

theorem max_value_ahn_operation :
  ∃ (max : ℕ), max = 600 ∧
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 →
  3 * (300 - n) ≤ max :=
by sorry

end max_value_ahn_operation_l1673_167322


namespace correct_calculation_l1673_167349

theorem correct_calculation (a b : ℝ) : 3 * a * b - 2 * a * b = a * b := by
  sorry

end correct_calculation_l1673_167349


namespace cube_volume_problem_l1673_167391

theorem cube_volume_problem (V₁ : ℝ) (A₂ : ℝ) : 
  V₁ = 8 → 
  A₂ = 3 * (6 * (V₁^(1/3))^2) → 
  (A₂ / 6)^(3/2) = 24 * Real.sqrt 3 := by
sorry

end cube_volume_problem_l1673_167391


namespace vertex_of_quadratic_l1673_167394

/-- The quadratic function -/
def f (x : ℝ) : ℝ := -x^2 + 6*x + 3

/-- The x-coordinate of the vertex -/
def h : ℝ := 3

/-- The y-coordinate of the vertex -/
def k : ℝ := 12

/-- Theorem: The vertex of the quadratic function f(x) = -x^2 + 6x + 3 is at (3, 12) -/
theorem vertex_of_quadratic :
  (∀ x : ℝ, f x = -(x - h)^2 + k) ∧ f h = k :=
sorry

end vertex_of_quadratic_l1673_167394


namespace two_new_players_joined_l1673_167347

/-- Given an initial group of players and some new players joining, 
    calculates the number of new players based on the total lives. -/
def new_players (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  (total_lives - initial_players * lives_per_player) / lives_per_player

/-- Proves that 2 new players joined the game given the initial conditions. -/
theorem two_new_players_joined :
  new_players 7 7 63 = 2 := by
  sorry

end two_new_players_joined_l1673_167347


namespace walters_money_percentage_l1673_167365

/-- The value of a penny in cents -/
def penny : ℕ := 1

/-- The value of a nickel in cents -/
def nickel : ℕ := 5

/-- The value of a dime in cents -/
def dime : ℕ := 10

/-- The value of a quarter in cents -/
def quarter : ℕ := 25

/-- The total number of cents in Walter's pocket -/
def walters_money : ℕ := penny + 2 * nickel + dime + 2 * quarter

/-- Theorem: Walter's money is 71% of a dollar -/
theorem walters_money_percentage :
  (walters_money : ℚ) / 100 = 71 / 100 := by sorry

end walters_money_percentage_l1673_167365


namespace min_value_when_a_neg_one_range_of_expression_when_a_neg_nine_l1673_167318

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + x * |x - 2*a|

-- Part 1: Minimum value when a = -1
theorem min_value_when_a_neg_one :
  ∃ (m : ℝ), m = -1/2 ∧ ∀ (x : ℝ), f (-1) x ≥ m :=
sorry

-- Part 2: Range of x₁/x₂ + x₁ when a = -9
theorem range_of_expression_when_a_neg_nine :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f (-9) x₁ = f (-9) x₂ →
  (x₁/x₂ + x₁ < -16 ∨ x₁/x₂ + x₁ ≥ -4) :=
sorry

end min_value_when_a_neg_one_range_of_expression_when_a_neg_nine_l1673_167318


namespace last_two_digits_problem_l1673_167378

theorem last_two_digits_problem (x y : ℕ+) (h1 : x ≠ y) (h2 : (x : ℚ)⁻¹ + (y : ℚ)⁻¹ = 2/13) :
  (x.val ^ y.val + y.val ^ x.val) % 100 = 74 := by
  sorry

end last_two_digits_problem_l1673_167378


namespace linear_coefficient_of_quadratic_l1673_167371

/-- 
Given a quadratic equation 5x^2 - 2x + 2 = 0, 
the coefficient of the linear term is -2 
-/
theorem linear_coefficient_of_quadratic (x : ℝ) : 
  (5 * x^2 - 2 * x + 2 = 0) → 
  (∃ a b c : ℝ, a * x^2 + b * x + c = 0 ∧ b = -2) :=
by sorry

end linear_coefficient_of_quadratic_l1673_167371


namespace N_is_composite_l1673_167363

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ Nat.Prime N := by
  sorry

end N_is_composite_l1673_167363


namespace expectation_problem_l1673_167380

/-- Given E(X) + E(2X + 1) = 8, prove that E(X) = 7/3 -/
theorem expectation_problem (X : ℝ → ℝ) (E : (ℝ → ℝ) → ℝ) 
  (h : E X + E (λ x => 2 * X x + 1) = 8) :
  E X = 7/3 := by
  sorry

end expectation_problem_l1673_167380


namespace polynomial_factorization_l1673_167385

theorem polynomial_factorization (x : ℝ) :
  x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 := by
  sorry

end polynomial_factorization_l1673_167385


namespace horizontal_axis_independent_l1673_167379

/-- Represents the different types of variables in a graph --/
inductive AxisVariable
  | Dependent
  | Constant
  | Independent
  | Function

/-- Represents a standard graph showing relationships between variables --/
structure StandardGraph where
  horizontalAxis : AxisVariable
  verticalAxis : AxisVariable

/-- Theorem stating that the horizontal axis in a standard graph usually represents the independent variable --/
theorem horizontal_axis_independent (g : StandardGraph) : g.horizontalAxis = AxisVariable.Independent := by
  sorry

end horizontal_axis_independent_l1673_167379
