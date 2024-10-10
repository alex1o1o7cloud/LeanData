import Mathlib

namespace circle_and_line_problem_l1639_163963

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 2
def l (x y : ℝ) : Prop := x + y - 2 = 0
def C₂ (x y : ℝ) : Prop := (x-2)^2 + (y-4)^2 = 20

-- Define the ray
def ray (x y : ℝ) : Prop := 2*x - y = 0 ∧ x ≥ 0

-- Theorem statement
theorem circle_and_line_problem :
  -- Given conditions
  (∀ x y, C₁ x y → l x y → (x = 1 ∧ y = 1)) →  -- l is tangent to C₁ at (1,1)
  (∃ a b, ray a b ∧ ∀ x y, C₂ x y → (x - a)^2 + (y - b)^2 = (x^2 + y^2)) →  -- Center of C₂ is on the ray and C₂ passes through origin
  (∃ x₁ y₁ x₂ y₂, C₂ x₁ y₁ ∧ C₂ x₂ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧ (x₁ - x₂)^2 + (y₁ - y₂)^2 = 48) →  -- Chord length is 4√3
  -- Conclusion
  (∀ x y, l x y ↔ x + y - 2 = 0) ∧
  (∀ x y, C₂ x y ↔ (x-2)^2 + (y-4)^2 = 20) :=
by sorry

end circle_and_line_problem_l1639_163963


namespace chris_dana_distance_difference_l1639_163923

/-- The difference in distance traveled between two bikers after a given time -/
def distance_difference (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed1 * time) - (speed2 * time)

/-- Theorem stating the difference in distance traveled between Chris and Dana -/
theorem chris_dana_distance_difference :
  distance_difference 17 12 6 = 30 := by
  sorry

end chris_dana_distance_difference_l1639_163923


namespace probability_of_two_as_median_l1639_163968

def S : Finset ℕ := {2, 0, 1, 5}

def is_median (a b c : ℕ) : Prop :=
  (a ≤ b ∧ b ≤ c) ∨ (c ≤ b ∧ b ≤ a)

def favorable_outcomes : Finset (ℕ × ℕ × ℕ) :=
  {(0, 2, 5), (1, 2, 5)}

def total_outcomes : Finset (ℕ × ℕ × ℕ) :=
  {(0, 1, 2), (0, 1, 5), (0, 2, 5), (1, 2, 5)}

theorem probability_of_two_as_median :
  (Finset.card favorable_outcomes : ℚ) / (Finset.card total_outcomes : ℚ) = 1 / 2 :=
sorry

end probability_of_two_as_median_l1639_163968


namespace unique_prime_six_digit_number_l1639_163930

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def six_digit_number (B : ℕ) : ℕ := 303700 + B

theorem unique_prime_six_digit_number :
  ∃! B : ℕ, B < 10 ∧ is_prime (six_digit_number B) ∧ six_digit_number B = 303703 :=
sorry

end unique_prime_six_digit_number_l1639_163930


namespace tangent_circle_height_difference_l1639_163907

/-- A circle tangent to the parabola y = x^2 at two points and lying inside the parabola --/
structure TangentCircle where
  /-- The x-coordinate of one tangent point (the other is at -a) --/
  a : ℝ
  /-- The y-coordinate of the circle's center --/
  b : ℝ
  /-- The radius of the circle --/
  r : ℝ
  /-- The circle lies inside the parabola --/
  inside : b > a^2
  /-- The circle is tangent to the parabola at (a, a^2) and (-a, a^2) --/
  tangent : (a^2 + (a^2 - b)^2 = r^2) ∧ (a^2 + (a^2 - b)^2 = r^2)

/-- The difference between the y-coordinate of the circle's center and the y-coordinate of either tangent point is 1/2 --/
theorem tangent_circle_height_difference (c : TangentCircle) : c.b - c.a^2 = 1/2 := by
  sorry

end tangent_circle_height_difference_l1639_163907


namespace hockey_puck_price_comparison_l1639_163981

theorem hockey_puck_price_comparison (P : ℝ) (h : P > 0) : P > 0.99 * P := by
  sorry

end hockey_puck_price_comparison_l1639_163981


namespace germination_expectation_l1639_163960

/-- The germination rate of seeds -/
def germination_rate : ℝ := 0.8

/-- The number of seeds sown -/
def seeds_sown : ℕ := 100

/-- The expected number of germinated seeds -/
def expected_germinated_seeds : ℝ := germination_rate * seeds_sown

theorem germination_expectation :
  expected_germinated_seeds = 80 := by sorry

end germination_expectation_l1639_163960


namespace quadratic_inequality_solution_l1639_163998

theorem quadratic_inequality_solution (x : ℝ) : 
  -3 * x^2 + 8 * x + 1 < 0 ↔ (4 - Real.sqrt 19) / 3 < x ∧ x < (4 + Real.sqrt 19) / 3 :=
by sorry

end quadratic_inequality_solution_l1639_163998


namespace sum_of_series_equals_two_l1639_163978

/-- The sum of the infinite series ∑(n=1 to ∞) (4n-2)/(3^n) is equal to 2. -/
theorem sum_of_series_equals_two :
  ∑' n : ℕ, (4 * n - 2 : ℝ) / (3 ^ n) = 2 := by sorry

end sum_of_series_equals_two_l1639_163978


namespace relationship_abc_l1639_163986

theorem relationship_abc : 
  let a := (1/2)^(2/3)
  let b := (1/3)^(1/3)
  let c := Real.log 3
  c > b ∧ b > a := by sorry

end relationship_abc_l1639_163986


namespace smallest_positive_multiple_of_45_l1639_163982

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 :=
by sorry

end smallest_positive_multiple_of_45_l1639_163982


namespace factorization_xy_squared_minus_x_l1639_163965

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end factorization_xy_squared_minus_x_l1639_163965


namespace geometric_series_common_ratio_l1639_163943

/-- The common ratio of the geometric series 7/8 - 35/72 + 175/432 - ... is -5/9 -/
theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -35/72
  let a₃ : ℚ := 175/432
  let r := a₂ / a₁
  r = -5/9 := by sorry

end geometric_series_common_ratio_l1639_163943


namespace percentage_problem_l1639_163945

theorem percentage_problem (n : ℝ) (p : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * n = 10 →
  (p / 100) * n = 120 →
  p = 40 := by
sorry

end percentage_problem_l1639_163945


namespace perfect_square_sum_in_pile_l1639_163961

theorem perfect_square_sum_in_pile (n : ℕ) (h : n ≥ 100) :
  ∀ (S₁ S₂ : Set ℕ), 
    (∀ k, n ≤ k ∧ k ≤ 2*n → k ∈ S₁ ∨ k ∈ S₂) →
    (S₁ ∩ S₂ = ∅) →
    (∃ (a b : ℕ), (a ∈ S₁ ∧ b ∈ S₁ ∧ a ≠ b ∧ ∃ (m : ℕ), a + b = m^2) ∨
                   (a ∈ S₂ ∧ b ∈ S₂ ∧ a ≠ b ∧ ∃ (m : ℕ), a + b = m^2)) :=
by sorry

end perfect_square_sum_in_pile_l1639_163961


namespace average_weight_problem_l1639_163984

theorem average_weight_problem (a b c : ℝ) :
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 46 →
  b = 37 →
  (a + b) / 2 = 40 := by
sorry

end average_weight_problem_l1639_163984


namespace sinusoidal_vertical_shift_l1639_163972

/-- Given a sinusoidal function y = a * sin(b * x + c) + d with positive constants a, b, c, and d,
    if the maximum value of y is 3 and the minimum value is -1, then d = 1. -/
theorem sinusoidal_vertical_shift 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (f : ℝ → ℝ) 
  (hf : ∀ x, f x = a * Real.sin (b * x + c) + d)
  (hmax : ∀ x, f x ≤ 3)
  (hmin : ∀ x, f x ≥ -1)
  (hex_max : ∃ x, f x = 3)
  (hex_min : ∃ x, f x = -1) :
  d = 1 := by
sorry

end sinusoidal_vertical_shift_l1639_163972


namespace binomial_coefficient_divisibility_l1639_163974

theorem binomial_coefficient_divisibility (m n : ℕ) (h1 : m > 0) (h2 : n > 1) :
  (∀ k : ℕ, 1 ≤ k ∧ k < m → n ∣ Nat.choose m k) →
  ∃ (p u : ℕ), Prime p ∧ u > 0 ∧ m = p^u ∧ n = p :=
by sorry

end binomial_coefficient_divisibility_l1639_163974


namespace john_taxes_l1639_163953

/-- Calculate the total tax given a progressive tax system and taxable income -/
def calculate_tax (taxable_income : ℕ) : ℕ :=
  let tax1 := min taxable_income 20000 * 10 / 100
  let tax2 := min (max (taxable_income - 20000) 0) 30000 * 15 / 100
  let tax3 := min (max (taxable_income - 50000) 0) 50000 * 20 / 100
  let tax4 := max (taxable_income - 100000) 0 * 25 / 100
  tax1 + tax2 + tax3 + tax4

/-- John's financial situation -/
theorem john_taxes :
  let main_job := 75000
  let freelance := 25000
  let rental := 15000
  let dividends := 10000
  let mortgage_deduction := 32000
  let retirement_deduction := 15000
  let charitable_deduction := 10000
  let education_credit := 3000
  let total_income := main_job + freelance + rental + dividends
  let total_deductions := mortgage_deduction + retirement_deduction + charitable_deduction + education_credit
  let taxable_income := total_income - total_deductions
  taxable_income = 65000 ∧ calculate_tax taxable_income = 9500 := by
  sorry


end john_taxes_l1639_163953


namespace area_right_triangle_45_deg_l1639_163957

theorem area_right_triangle_45_deg (a : ℝ) (h1 : a = 8) (h2 : a > 0) : 
  (1 / 2 : ℝ) * a * a = 32 := by
sorry

end area_right_triangle_45_deg_l1639_163957


namespace geometric_configurations_l1639_163994

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (passes_through : Plane → Line → Prop)
variable (skew : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- State the theorem
theorem geometric_configurations 
  (α β : Plane) (l m : Line) 
  (h1 : passes_through α l)
  (h2 : passes_through α m)
  (h3 : passes_through β l)
  (h4 : passes_through β m)
  (h5 : skew l m)
  (h6 : perpendicular l m) :
  (∃ (α' β' : Plane) (l' m' : Line), 
    passes_through α' l' ∧ 
    passes_through α' m' ∧ 
    passes_through β' l' ∧ 
    passes_through β' m' ∧ 
    skew l' m' ∧ 
    perpendicular l' m' ∧
    ((parallel α' β') ∨ 
     (perpendicular_planes α' β') ∨ 
     (parallel_line_plane l' β') ∨ 
     (perpendicular_line_plane m' α'))) :=
by sorry

end geometric_configurations_l1639_163994


namespace division_multiplication_example_l1639_163937

theorem division_multiplication_example : (180 / 6) * 3 = 90 := by
  sorry

end division_multiplication_example_l1639_163937


namespace sum_inequality_l1639_163964

theorem sum_inequality (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : (a : ℚ) / (b + c^2) = (a + c^2 : ℚ) / b) : 
  a + b + c ≤ 0 := by
sorry

end sum_inequality_l1639_163964


namespace red_button_probability_main_theorem_l1639_163950

/-- Represents a jar containing buttons of different colors -/
structure Jar :=
  (red : ℕ)
  (blue : ℕ)

/-- Calculates the total number of buttons in a jar -/
def Jar.total (j : Jar) : ℕ := j.red + j.blue

/-- Represents the state of the jars after Carla's action -/
structure JarState :=
  (jarA : Jar)
  (jarB : Jar)

/-- The probability of selecting a red button from a jar -/
def redProbability (j : Jar) : ℚ :=
  j.red / j.total

/-- The initial state of Jar A -/
def initialJarA : Jar := ⟨6, 10⟩

/-- Theorem stating the probability of selecting red buttons from both jars -/
theorem red_button_probability (state : JarState) : 
  redProbability state.jarA * redProbability state.jarB = 1/6 :=
sorry

/-- Main theorem combining all conditions and the result -/
theorem main_theorem (state : JarState) :
  initialJarA.total = 16 →
  state.jarA.total = (3/4 : ℚ) * initialJarA.total →
  state.jarB.total = initialJarA.total - state.jarA.total →
  state.jarB.red = state.jarB.blue →
  state.jarA.red + state.jarB.red = initialJarA.red →
  state.jarA.blue + state.jarB.blue = initialJarA.blue →
  redProbability state.jarA * redProbability state.jarB = 1/6 :=
sorry

end red_button_probability_main_theorem_l1639_163950


namespace decreasing_quadratic_function_l1639_163903

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem decreasing_quadratic_function (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → f a x > f a y) → a ≤ -3 := by
  sorry

end decreasing_quadratic_function_l1639_163903


namespace gcd_5280_12155_l1639_163959

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 55 := by
  sorry

end gcd_5280_12155_l1639_163959


namespace sum_of_angles_with_tangent_roots_l1639_163904

theorem sum_of_angles_with_tangent_roots (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  (∃ x y : Real, x^2 - 5*x + 6 = 0 ∧ y^2 - 5*y + 6 = 0 ∧ Real.tan α = x ∧ Real.tan β = y) →
  α + β = 3*π/4 := by
sorry

end sum_of_angles_with_tangent_roots_l1639_163904


namespace complement_of_union_equals_five_l1639_163977

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {3, 4}

-- Theorem statement
theorem complement_of_union_equals_five : 
  (U \ (M ∪ N)) = {5} := by sorry

end complement_of_union_equals_five_l1639_163977


namespace infinitely_many_composites_l1639_163928

def last_digit (n : ℕ) : ℕ := n % 10

def remove_last_digit (n : ℕ) : ℕ := n / 10

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

def sequence_property (p : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, last_digit (p (i + 1)) ≠ 9 ∧ remove_last_digit (p (i + 1)) = p i

theorem infinitely_many_composites (p : ℕ → ℕ) (h : sequence_property p) :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ is_composite (p n) :=
sorry

end infinitely_many_composites_l1639_163928


namespace sum_of_transformed_numbers_l1639_163912

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
  3 * (a - 5) + 3 * (b - 5) = 3 * S - 30 := by
  sorry

end sum_of_transformed_numbers_l1639_163912


namespace john_completion_time_l1639_163939

/-- The number of days it takes Jane to complete the task alone -/
def jane_days : ℝ := 12

/-- The total number of days it took to complete the task -/
def total_days : ℝ := 10.8

/-- The number of days Jane was indisposed before the work was completed -/
def jane_indisposed : ℝ := 6

/-- The number of days it takes John to complete the task alone -/
def john_days : ℝ := 18

theorem john_completion_time :
  (jane_indisposed / john_days) + 
  ((total_days - jane_indisposed) * (1 / john_days + 1 / jane_days)) = 1 :=
sorry

#check john_completion_time

end john_completion_time_l1639_163939


namespace snail_max_distance_l1639_163991

/-- Represents the movement of a snail over time -/
structure SnailMovement where
  /-- The total observation time in hours -/
  total_time : ℝ
  /-- The observation duration of each observer in hours -/
  observer_duration : ℝ
  /-- The distance traveled during each observation in meters -/
  distance_per_observation : ℝ
  /-- Ensures there is always at least one observer -/
  always_observed : Prop

/-- The maximum distance the snail can travel given the conditions -/
def max_distance (sm : SnailMovement) : ℝ :=
  18

/-- Theorem stating the maximum distance the snail can travel is 18 meters -/
theorem snail_max_distance (sm : SnailMovement) 
    (h1 : sm.total_time = 10)
    (h2 : sm.observer_duration = 1)
    (h3 : sm.distance_per_observation = 1)
    (h4 : sm.always_observed) : 
  max_distance sm = 18 := by
  sorry

end snail_max_distance_l1639_163991


namespace work_completion_time_l1639_163997

/-- Given that:
  - B can do a work in 24 days
  - A and B working together can finish the work in 8 days
  Prove that A can do the work alone in 12 days -/
theorem work_completion_time (work : ℝ) (a_rate b_rate combined_rate : ℝ) :
  work / b_rate = 24 →
  work / combined_rate = 8 →
  combined_rate = a_rate + b_rate →
  work / a_rate = 12 := by
  sorry

end work_completion_time_l1639_163997


namespace translated_line_y_axis_intersection_l1639_163940

/-- The intersection point of a line translated upward with the y-axis -/
theorem translated_line_y_axis_intersection
  (original_line : ℝ → ℝ)
  (h_original : ∀ x, original_line x = x - 3)
  (translation : ℝ)
  (h_translation : translation = 2)
  (translated_line : ℝ → ℝ)
  (h_translated : ∀ x, translated_line x = original_line x + translation)
  : translated_line 0 = -1 :=
by sorry

end translated_line_y_axis_intersection_l1639_163940


namespace remaining_packs_eq_26_l1639_163958

/-- The number of cookie packs Tory needs to sell -/
def total_goal : ℕ := 50

/-- The number of cookie packs Tory sold to his grandmother -/
def sold_to_grandmother : ℕ := 12

/-- The number of cookie packs Tory sold to his uncle -/
def sold_to_uncle : ℕ := 7

/-- The number of cookie packs Tory sold to a neighbor -/
def sold_to_neighbor : ℕ := 5

/-- The number of remaining cookie packs Tory needs to sell -/
def remaining_packs : ℕ := total_goal - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor)

theorem remaining_packs_eq_26 : remaining_packs = 26 := by
  sorry

end remaining_packs_eq_26_l1639_163958


namespace max_books_borrowed_l1639_163918

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (h1 : total_students = 20)
  (h2 : zero_books = 2)
  (h3 : one_book = 8)
  (h4 : two_books = 3)
  (h5 : (total_students - (zero_books + one_book + two_books)) * 3 ≤ 
        total_students * 2 - (one_book * 1 + two_books * 2)) :
  ∃ (max_books : ℕ), max_books = 8 ∧ 
    ∀ (student_books : ℕ), student_books ≤ max_books := by
  sorry

end max_books_borrowed_l1639_163918


namespace cos_pi_fourth_plus_alpha_l1639_163931

theorem cos_pi_fourth_plus_alpha (α : ℝ) (h : Real.sin (π/4 - α) = 1/3) : 
  Real.cos (π/4 + α) = 1/3 := by
  sorry

end cos_pi_fourth_plus_alpha_l1639_163931


namespace birds_beetles_per_day_l1639_163947

-- Define the constants
def birds_per_snake : ℕ := 3
def snakes_per_jaguar : ℕ := 5
def num_jaguars : ℕ := 6
def total_beetles : ℕ := 1080

-- Define the theorem
theorem birds_beetles_per_day :
  ∀ (beetles_per_bird : ℕ),
    beetles_per_bird * (birds_per_snake * snakes_per_jaguar * num_jaguars) = total_beetles →
    beetles_per_bird = 12 := by
  sorry

end birds_beetles_per_day_l1639_163947


namespace simplify_expression_evaluate_expression_l1639_163980

-- Part 1
theorem simplify_expression (x y : ℝ) : x - (2*x - y) + (3*x - 2*y) = 2*x - y := by
  sorry

-- Part 2
theorem evaluate_expression : 
  let x : ℚ := -2/3
  let y : ℚ := 3/2
  2*x*y + (-3*x^3 + 5*x*y + 2) - 3*(2*x*y - x^3 + 1) = -2 := by
  sorry

end simplify_expression_evaluate_expression_l1639_163980


namespace opposite_sides_line_condition_l1639_163909

theorem opposite_sides_line_condition (a : ℝ) : 
  (∃ (x1 y1 x2 y2 : ℝ), x1 = 1 ∧ y1 = 3 ∧ x2 = -1 ∧ y2 = -4 ∧ 
    ((a * x1 + 3 * y1 + 1) * (a * x2 + 3 * y2 + 1) < 0)) ↔ 
  (a < -11 ∨ a > -10) := by
sorry

end opposite_sides_line_condition_l1639_163909


namespace complex_number_problem_l1639_163932

theorem complex_number_problem (m : ℝ) (z z₁ : ℂ) :
  z₁ = m * (m - 1) + (m - 1) * Complex.I ∧
  z₁.re = 0 ∧
  z₁.im ≠ 0 ∧
  (3 + z₁) * z = 4 + 2 * Complex.I →
  m = 0 ∧ z = 1 + Complex.I := by
  sorry

end complex_number_problem_l1639_163932


namespace integer_x_is_seven_l1639_163946

theorem integer_x_is_seven (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 8)
  (h3 : -2 < x ∧ x < 9)
  (h4 : 0 < x ∧ x < 8)
  (h5 : x + 1 < 9) :
  x = 7 := by
  sorry

end integer_x_is_seven_l1639_163946


namespace rationality_of_expressions_l1639_163962

theorem rationality_of_expressions :
  (∃ (a b : ℤ), b ≠ 0 ∧ (1.728 : ℚ) = a / b) ∧
  (∃ (c d : ℤ), d ≠ 0 ∧ (0.0032 : ℚ) = c / d) ∧
  (∃ (e f : ℤ), f ≠ 0 ∧ (-8 : ℚ) = e / f) ∧
  (∃ (g h : ℤ), h ≠ 0 ∧ (0.25 : ℚ) = g / h) ∧
  ¬(∃ (i j : ℤ), j ≠ 0 ∧ Real.pi = (i : ℚ) / j) :=
by sorry

end rationality_of_expressions_l1639_163962


namespace f_lower_bound_l1639_163954

noncomputable section

def f (x t : ℝ) : ℝ := ((x + t) / (x - 1)) * Real.exp (x - 1)

theorem f_lower_bound (x t : ℝ) (hx : x > 1) (ht : t > -1) :
  f x t > Real.sqrt x * (1 + (1/2) * Real.log x) := by
  sorry

end f_lower_bound_l1639_163954


namespace set_difference_equiv_l1639_163971

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x) ∧ -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1}

theorem set_difference_equiv : A \ B = {x | x < 0} := by sorry

end set_difference_equiv_l1639_163971


namespace converse_parallel_supplementary_true_converse_vertical_angles_false_converse_squares_equal_false_converse_sum_squares_positive_false_only_parallel_supplementary_has_true_converse_l1639_163936

-- Define the concept of vertical angles
def vertical_angles (a b : Angle) : Prop := sorry

-- Define the concept of consecutive interior angles
def consecutive_interior_angles (a b : Angle) (l1 l2 : Line) : Prop := sorry

-- Define the concept of supplementary angles
def supplementary (a b : Angle) : Prop := sorry

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

theorem converse_parallel_supplementary_true :
  ∀ (l1 l2 : Line) (a b : Angle),
    parallel l1 l2 → consecutive_interior_angles a b l1 l2 → supplementary a b := by sorry

theorem converse_vertical_angles_false :
  ∃ (a b : Angle), a = b ∧ ¬(vertical_angles a b) := by sorry

theorem converse_squares_equal_false :
  ∃ (a b : ℝ), a^2 = b^2 ∧ a ≠ b := by sorry

theorem converse_sum_squares_positive_false :
  ∃ (a b : ℝ), a^2 + b^2 > 0 ∧ (a ≤ 0 ∨ b ≤ 0) := by sorry

theorem only_parallel_supplementary_has_true_converse :
  (∀ (l1 l2 : Line) (a b : Angle),
    parallel l1 l2 → consecutive_interior_angles a b l1 l2 → supplementary a b) ∧
  (∃ (a b : Angle), a = b ∧ ¬(vertical_angles a b)) ∧
  (∃ (a b : ℝ), a^2 = b^2 ∧ a ≠ b) ∧
  (∃ (a b : ℝ), a^2 + b^2 > 0 ∧ (a ≤ 0 ∨ b ≤ 0)) := by sorry

end converse_parallel_supplementary_true_converse_vertical_angles_false_converse_squares_equal_false_converse_sum_squares_positive_false_only_parallel_supplementary_has_true_converse_l1639_163936


namespace marias_salary_l1639_163915

theorem marias_salary (S : ℝ) : 
  (S * 0.2 + S * 0.05 + (S - S * 0.2 - S * 0.05) * 0.25 + 1125 = S) → S = 2000 := by
sorry

end marias_salary_l1639_163915


namespace triangle_third_side_length_l1639_163913

theorem triangle_third_side_length 
  (a b : ℝ) 
  (angle : ℝ) 
  (ha : a = 9) 
  (hb : b = 8) 
  (hangle : angle = 150 * π / 180) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*(Real.cos angle) ∧ 
            c = Real.sqrt (145 + 72 * Real.sqrt 3) :=
by sorry

end triangle_third_side_length_l1639_163913


namespace infinitely_many_pairs_smallest_pair_l1639_163921

/-- Predicate defining the conditions for x and y -/
def satisfies_conditions (x y : ℕ+) : Prop :=
  (x * (x + 1) ∣ y * (y + 1)) ∧
  ¬(x ∣ y) ∧
  ¬((x + 1) ∣ y) ∧
  ¬(x ∣ (y + 1)) ∧
  ¬((x + 1) ∣ (y + 1))

/-- There exist infinitely many pairs of positive integers satisfying the conditions -/
theorem infinitely_many_pairs :
  ∀ n : ℕ, ∃ x y : ℕ+, x > n ∧ y > n ∧ satisfies_conditions x y :=
sorry

/-- The smallest pair satisfying the conditions is (14, 20) -/
theorem smallest_pair :
  satisfies_conditions 14 20 ∧
  ∀ x y : ℕ+, satisfies_conditions x y → x ≥ 14 ∧ y ≥ 20 :=
sorry

end infinitely_many_pairs_smallest_pair_l1639_163921


namespace coordinate_uniqueness_l1639_163973

/-- A type representing a location description -/
inductive LocationDescription
| Coordinates (longitude : Real) (latitude : Real)
| CityLandmark (city : String) (landmark : String)
| Direction (angle : Real)
| VenueSeat (venue : String) (seat : String)

/-- Function to check if a location description uniquely determines a location -/
def uniquelyDeterminesLocation (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.Coordinates _ _ => True
  | _ => False

/-- Theorem stating that only coordinate-based descriptions uniquely determine locations -/
theorem coordinate_uniqueness 
  (descriptions : List LocationDescription) 
  (h_contains_coordinates : ∃ (long lat : Real), LocationDescription.Coordinates long lat ∈ descriptions) :
  ∃! (desc : LocationDescription), desc ∈ descriptions ∧ uniquelyDeterminesLocation desc :=
sorry

end coordinate_uniqueness_l1639_163973


namespace f_eq_g_g_is_right_shift_f_is_right_shift_of_x_squared_l1639_163975

/-- The original quadratic function -/
def f (x : ℝ) := x^2 - 2*x + 1

/-- The shifted quadratic function -/
def g (x : ℝ) := (x - 1)^2

/-- Theorem stating that f and g are equivalent -/
theorem f_eq_g : ∀ x, f x = g x := by sorry

/-- Theorem stating that g is a right shift of x^2 by 1 unit -/
theorem g_is_right_shift : ∀ x, g x = (x - 1)^2 := by sorry

/-- Main theorem: f is a right shift of x^2 by 1 unit -/
theorem f_is_right_shift_of_x_squared : 
  ∃ h : ℝ, h > 0 ∧ (∀ x, f x = (x - h)^2) := by sorry

end f_eq_g_g_is_right_shift_f_is_right_shift_of_x_squared_l1639_163975


namespace mary_baseball_cards_l1639_163990

theorem mary_baseball_cards :
  ∀ (initial_cards torn_cards fred_cards bought_cards : ℕ),
    initial_cards = 18 →
    torn_cards = 8 →
    fred_cards = 26 →
    bought_cards = 40 →
    initial_cards - torn_cards + fred_cards + bought_cards = 76 :=
by
  sorry

end mary_baseball_cards_l1639_163990


namespace expression_value_l1639_163905

theorem expression_value (a b c k : ℤ) 
  (ha : a = 30) (hb : b = 10) (hc : c = 7) (hk : k = 3) : 
  k * ((a - (b - c)) - ((a - b) - c)) = 42 := by
  sorry

end expression_value_l1639_163905


namespace some_ai_in_machines_l1639_163992

-- Define the sets
variable (Robot : Type) -- Set of all robots
variable (Machine : Type) -- Set of all machines
variable (AdvancedAI : Type) -- Set of all advanced AI systems

-- Define the relations
variable (has_ai : Robot → AdvancedAI → Prop) -- Relation: robot has advanced AI
variable (is_machine : Robot → Machine → Prop) -- Relation: robot is a machine

-- State the theorem
theorem some_ai_in_machines 
  (h1 : ∀ (r : Robot), ∃ (ai : AdvancedAI), has_ai r ai) -- All robots have advanced AI
  (h2 : ∃ (r : Robot) (m : Machine), is_machine r m) -- Some robots are machines
  : ∃ (ai : AdvancedAI) (m : Machine), 
    ∃ (r : Robot), has_ai r ai ∧ is_machine r m :=
by sorry

end some_ai_in_machines_l1639_163992


namespace point_A_in_fourth_quadrant_l1639_163995

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_A_in_fourth_quadrant :
  is_in_fourth_quadrant 2 (-3) := by
  sorry

end point_A_in_fourth_quadrant_l1639_163995


namespace arithmetic_mean_of_first_four_primes_reciprocals_l1639_163925

-- Define the first four prime numbers
def first_four_primes : List ℕ := [2, 3, 5, 7]

-- Define the function to calculate the arithmetic mean of reciprocals
def arithmetic_mean_of_reciprocals (numbers : List ℕ) : ℚ :=
  let reciprocals := numbers.map (fun n => (1 : ℚ) / n)
  reciprocals.sum / numbers.length

-- Theorem statement
theorem arithmetic_mean_of_first_four_primes_reciprocals :
  arithmetic_mean_of_reciprocals first_four_primes = 247 / 840 := by
  sorry

end arithmetic_mean_of_first_four_primes_reciprocals_l1639_163925


namespace triangle_properties_l1639_163908

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states properties of a specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * (Real.cos (t.B / 2))^2 + t.b * (Real.cos (t.A / 2))^2 = (3/2) * t.c)
  (h2 : t.a = 2 * t.b)
  (h3 : (1/2) * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 15) :
  (t.A > π / 2) ∧ (t.b = 4) := by
  sorry

end triangle_properties_l1639_163908


namespace gravelingCostIs3600_l1639_163983

/-- Represents the dimensions and cost parameters of a rectangular lawn with intersecting roads -/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  roadWidth : ℝ
  costPerSqm : ℝ

/-- Calculates the total cost of graveling two intersecting roads in a rectangular lawn -/
def totalGravelingCost (lawn : LawnWithRoads) : ℝ :=
  let lengthRoadArea := lawn.length * lawn.roadWidth
  let widthRoadArea := (lawn.width - lawn.roadWidth) * lawn.roadWidth
  let totalArea := lengthRoadArea + widthRoadArea
  totalArea * lawn.costPerSqm

/-- Theorem stating that the total cost of graveling for the given lawn is 3600 -/
theorem gravelingCostIs3600 (lawn : LawnWithRoads) 
  (h1 : lawn.length = 80)
  (h2 : lawn.width = 50)
  (h3 : lawn.roadWidth = 10)
  (h4 : lawn.costPerSqm = 3) :
  totalGravelingCost lawn = 3600 := by
  sorry

#eval totalGravelingCost { length := 80, width := 50, roadWidth := 10, costPerSqm := 3 }

end gravelingCostIs3600_l1639_163983


namespace min_tetrahedron_volume_l1639_163970

/-- Given a point P(1, 4, 5) in 3D Cartesian coordinate system O-xyz,
    and a plane passing through P intersecting positive axes at points A, B, and C,
    prove that the minimum volume V of tetrahedron O-ABC is 15. -/
theorem min_tetrahedron_volume (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_plane : 1 / a + 4 / b + 5 / c = 1) :
  (1 / 6 : ℝ) * a * b * c ≥ 15 := by
  sorry

end min_tetrahedron_volume_l1639_163970


namespace necklace_arrangement_count_l1639_163948

/-- The number of distinct circular arrangements of balls in a necklace -/
def necklace_arrangements (red : ℕ) (green : ℕ) (yellow : ℕ) : ℕ :=
  let total := red + green + yellow
  let linear_arrangements := Nat.choose (total - 1) red * Nat.choose (total - 1 - red) yellow
  (linear_arrangements - Nat.choose (total / 2) (red / 2)) / 2 + Nat.choose (total / 2) (red / 2)

/-- Theorem stating the number of distinct arrangements for the given problem -/
theorem necklace_arrangement_count :
  necklace_arrangements 6 1 8 = 1519 := by
  sorry

#eval necklace_arrangements 6 1 8

end necklace_arrangement_count_l1639_163948


namespace strawberries_left_l1639_163985

/-- Given an initial amount of strawberries and amounts eaten over two days, 
    calculate the remaining amount. -/
def remaining_strawberries (initial : ℝ) (eaten_day1 : ℝ) (eaten_day2 : ℝ) : ℝ :=
  initial - eaten_day1 - eaten_day2

/-- Theorem stating that given the specific amounts in the problem, 
    the remaining amount of strawberries is 0.5 kg. -/
theorem strawberries_left : 
  remaining_strawberries 1.6 0.8 0.3 = 0.5 := by
  sorry

end strawberries_left_l1639_163985


namespace units_digit_37_pow_37_l1639_163935

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of 37^37 is 7 -/
theorem units_digit_37_pow_37 : unitsDigit (37^37) = 7 := by
  sorry

end units_digit_37_pow_37_l1639_163935


namespace sum_of_coefficients_l1639_163951

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 := by
sorry

end sum_of_coefficients_l1639_163951


namespace circle_equation_proof_l1639_163906

-- Define a circle with center (1, 1) passing through (0, 0)
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

-- Theorem statement
theorem circle_equation_proof :
  ∀ x y : ℝ, circle_equation x y ↔ 
    ((x = 1 ∧ y = 1) ∨ (x = 0 ∧ y = 0) → 
      (x - 1)^2 + (y - 1)^2 = 2) :=
by
  sorry


end circle_equation_proof_l1639_163906


namespace earnings_ratio_l1639_163914

/-- Proves that given Mork's tax rate of 30%, Mindy's tax rate of 20%, and their combined tax rate of 22.5%, the ratio of Mindy's earnings to Mork's earnings is 3:1. -/
theorem earnings_ratio (mork_earnings mindy_earnings : ℝ) 
  (mork_tax_rate : ℝ) (mindy_tax_rate : ℝ) (combined_tax_rate : ℝ)
  (h1 : mork_tax_rate = 0.3)
  (h2 : mindy_tax_rate = 0.2)
  (h3 : combined_tax_rate = 0.225)
  (h4 : mork_earnings > 0)
  (h5 : mindy_earnings > 0)
  (h6 : mindy_tax_rate * mindy_earnings + mork_tax_rate * mork_earnings = 
        combined_tax_rate * (mindy_earnings + mork_earnings)) :
  mindy_earnings / mork_earnings = 3 := by
  sorry


end earnings_ratio_l1639_163914


namespace polynomial_division_quotient_l1639_163938

theorem polynomial_division_quotient : 
  let dividend := fun (z : ℝ) => 5*z^5 - 3*z^4 + 4*z^3 - 7*z^2 + 2*z - 1
  let divisor := fun (z : ℝ) => 3*z^2 + 4*z + 1
  let quotient := fun (z : ℝ) => (5/3)*z^3 - (29/9)*z^2 + (71/27)*z - 218/81
  ∀ z : ℝ, dividend z = (divisor z) * (quotient z) + (dividend z % divisor z) := by
  sorry

end polynomial_division_quotient_l1639_163938


namespace probability_of_white_ball_l1639_163910

/-- Represents the number of red balls in the bag -/
def red_balls : ℕ := 4

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 2

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := red_balls + white_balls

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 2

/-- The probability of drawing at least one white ball when randomly selecting 2 balls from a bag containing 4 red balls and 2 white balls -/
theorem probability_of_white_ball : 
  (1 : ℚ) - (Nat.choose red_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 3/5 := by
  sorry

end probability_of_white_ball_l1639_163910


namespace edwards_initial_money_l1639_163902

/-- Given that Edward spent $16 and has $2 left, his initial amount of money was $18. -/
theorem edwards_initial_money :
  ∀ (initial spent left : ℕ),
    spent = 16 →
    left = 2 →
    initial = spent + left →
    initial = 18 := by
  sorry

end edwards_initial_money_l1639_163902


namespace quadratic_solution_square_l1639_163917

theorem quadratic_solution_square (x : ℝ) :
  7 * x^2 + 6 = 5 * x + 11 →
  (8 * x - 5)^2 = (2865 - 120 * Real.sqrt 165) / 49 := by
  sorry

end quadratic_solution_square_l1639_163917


namespace opposite_of_negative_two_l1639_163949

theorem opposite_of_negative_two : (- (-2)) = 2 := by
  sorry

end opposite_of_negative_two_l1639_163949


namespace orange_removal_theorem_l1639_163969

/-- Represents the number of oranges Mary must put back to achieve the desired average price -/
def oranges_to_remove (apple_price orange_price : ℚ) (total_fruits : ℕ) (initial_avg_price desired_avg_price : ℚ) : ℚ :=
  (total_fruits * initial_avg_price - total_fruits * desired_avg_price) / (orange_price - desired_avg_price)

theorem orange_removal_theorem (apple_price orange_price : ℚ) (total_fruits : ℕ) (initial_avg_price desired_avg_price : ℚ) :
  apple_price = 40/100 ∧ 
  orange_price = 60/100 ∧ 
  total_fruits = 10 ∧ 
  initial_avg_price = 54/100 ∧ 
  desired_avg_price = 50/100 → 
  oranges_to_remove apple_price orange_price total_fruits initial_avg_price desired_avg_price = 4 := by
  sorry

#eval oranges_to_remove (40/100) (60/100) 10 (54/100) (50/100)

end orange_removal_theorem_l1639_163969


namespace sam_speed_l1639_163920

/-- Given the biking speeds of Eugene, Clara, and Sam, prove Sam's speed --/
theorem sam_speed (eugene_speed : ℚ) (clara_ratio : ℚ) (sam_ratio : ℚ) :
  eugene_speed = 5 →
  clara_ratio = 3 / 4 →
  sam_ratio = 4 / 3 →
  sam_ratio * (clara_ratio * eugene_speed) = 5 := by
  sorry

end sam_speed_l1639_163920


namespace sum_of_complex_unit_magnitude_l1639_163927

theorem sum_of_complex_unit_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1)
  (h4 : a^2 / (b*c) + b^2 / (a*c) + c^2 / (a*b) = 3)
  (h5 : a + b + c ≠ 0) :
  Complex.abs (a + b + c) = Real.sqrt 3 := by
sorry

end sum_of_complex_unit_magnitude_l1639_163927


namespace function_properties_l1639_163926

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_properties (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_shift : ∀ x, f (x + 1) = -f x)
  (h_incr : is_increasing_on f (-1) 0) :
  (∃ p, p > 0 ∧ ∀ x, f (x + p) = f x) ∧
  (∀ x, f (2 - x) = f x) ∧
  f 2 = f 0 := by
sorry

end function_properties_l1639_163926


namespace chocolate_gain_percent_l1639_163993

theorem chocolate_gain_percent :
  ∀ (C S : ℝ),
  C > 0 →
  S > 0 →
  24 * C = 16 * S →
  (S - C) / C * 100 = 50 :=
by
  sorry

end chocolate_gain_percent_l1639_163993


namespace continuity_at_zero_l1639_163976

noncomputable def f (x : ℝ) : ℝ := 
  (Real.rpow (1 + x) (1/3) - 1) / (Real.sqrt (4 + x) - 2)

theorem continuity_at_zero : 
  Filter.Tendsto f (nhds 0) (nhds (4/3)) := by sorry

end continuity_at_zero_l1639_163976


namespace log_equality_l1639_163929

theorem log_equality (a b : ℝ) (h1 : a = Real.log 484 / Real.log 4) (h2 : b = Real.log 22 / Real.log 2) : a = b := by
  sorry

end log_equality_l1639_163929


namespace f_max_value_l1639_163987

/-- The function f(x) = 8x - 3x^2 -/
def f (x : ℝ) : ℝ := 8 * x - 3 * x^2

/-- The maximum value of f(x) for any real x is 16/3 -/
theorem f_max_value : ∃ (M : ℝ), M = 16/3 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end f_max_value_l1639_163987


namespace triangle_rotation_theorem_l1639_163996

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle with vertices O, P, and Q -/
structure Triangle where
  O : Point
  P : Point
  Q : Point

/-- Calculates the angle between two vectors -/
def angle (v1 v2 : Point) : ℝ := sorry

/-- Rotates a point 90 degrees counterclockwise around the origin -/
def rotate90 (p : Point) : Point := 
  { x := -p.y, y := p.x }

theorem triangle_rotation_theorem (t : Triangle) : 
  t.O = ⟨0, 0⟩ → 
  t.Q = ⟨6, 0⟩ → 
  t.P.x > 0 → 
  t.P.y > 0 → 
  angle ⟨t.P.x - t.Q.x, t.P.y - t.Q.y⟩ ⟨t.O.x - t.Q.x, t.O.y - t.Q.y⟩ = π / 2 →
  angle ⟨t.P.x - t.O.x, t.P.y - t.O.y⟩ ⟨t.Q.x - t.O.x, t.Q.y - t.O.y⟩ = π / 4 →
  t.P = ⟨6, 6⟩ ∧ rotate90 t.P = ⟨-6, 6⟩ := by
sorry

end triangle_rotation_theorem_l1639_163996


namespace line_tangent_to_circle_l1639_163988

/-- The line y = 2 is tangent to the circle (x - 2)² + y² = 4 -/
theorem line_tangent_to_circle :
  ∃ (x y : ℝ), y = 2 ∧ (x - 2)^2 + y^2 = 4 ∧
  ∀ (x' y' : ℝ), y' = 2 → (x' - 2)^2 + y'^2 ≥ 4 :=
sorry

end line_tangent_to_circle_l1639_163988


namespace shepherd_puzzle_l1639_163989

/-- The number of sheep after passing through a gate, given the number before the gate -/
def sheep_after_gate (n : ℕ) : ℕ := n / 2 + 1

/-- The number of sheep after passing through n gates, given the initial number -/
def sheep_after_gates (initial : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initial
  | n + 1 => sheep_after_gate (sheep_after_gates initial n)

theorem shepherd_puzzle :
  ∃ initial : ℕ, sheep_after_gates initial 6 = 2 ∧ initial = 2 := by sorry

end shepherd_puzzle_l1639_163989


namespace find_q_l1639_163900

theorem find_q (p q : ℝ) (h1 : p > 1) (h2 : q > 1) (h3 : 1/p + 1/q = 3/2) (h4 : p*q = 9) : q = 6 := by
  sorry

end find_q_l1639_163900


namespace chess_tournament_games_l1639_163922

theorem chess_tournament_games (total_games : ℕ) (participants : ℕ) 
  (h1 : total_games = 120) (h2 : participants = 16) :
  (participants - 1 : ℕ) = 15 ∧ total_games = participants * (participants - 1) / 2 := by
  sorry

end chess_tournament_games_l1639_163922


namespace solution_to_linear_equation_l1639_163979

theorem solution_to_linear_equation :
  ∃ (x y : ℤ), x + 2 * y = 6 ∧ x = 2 ∧ y = 2 := by
  sorry

end solution_to_linear_equation_l1639_163979


namespace min_value_theorem_l1639_163934

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 5/b = 1) :
  a + 5*b ≥ 36 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 5/b₀ = 1 ∧ a₀ + 5*b₀ = 36 :=
sorry

end min_value_theorem_l1639_163934


namespace elastic_band_radius_increase_l1639_163916

theorem elastic_band_radius_increase (r₁ r₂ : ℝ) : 
  2 * π * r₁ = 40 →  -- Initial circumference
  2 * π * r₂ = 80 →  -- Final circumference
  r₂ - r₁ = 20 / π := by
  sorry

end elastic_band_radius_increase_l1639_163916


namespace rectangle_area_with_equal_perimeter_to_triangle_l1639_163941

/-- The area of a rectangle with equal perimeter to a specific triangle -/
theorem rectangle_area_with_equal_perimeter_to_triangle : 
  ∀ (rectangle_side1 rectangle_side2 : ℝ),
  rectangle_side1 = 12 →
  2 * (rectangle_side1 + rectangle_side2) = 10 + 12 + 15 →
  rectangle_side1 * rectangle_side2 = 78 :=
by sorry

end rectangle_area_with_equal_perimeter_to_triangle_l1639_163941


namespace joan_total_games_l1639_163952

/-- The total number of football games Joan attended over two years -/
def total_games (this_year last_year : ℕ) : ℕ :=
  this_year + last_year

/-- Theorem: Joan attended 9 football games in total over two years -/
theorem joan_total_games : total_games 4 5 = 9 := by
  sorry

end joan_total_games_l1639_163952


namespace cookies_for_lunch_is_five_l1639_163999

/-- Calculates the number of cookies needed to reach the target calorie count for lunch -/
def cookiesForLunch (totalCalories burgerCalories carrotCalories cookieCalories : ℕ) 
                    (numCarrots : ℕ) : ℕ :=
  let remainingCalories := totalCalories - burgerCalories - (carrotCalories * numCarrots)
  remainingCalories / cookieCalories

/-- Proves that the number of cookies each kid gets is 5 -/
theorem cookies_for_lunch_is_five :
  cookiesForLunch 750 400 20 50 5 = 5 := by
  sorry

#eval cookiesForLunch 750 400 20 50 5

end cookies_for_lunch_is_five_l1639_163999


namespace product_difference_equality_l1639_163911

theorem product_difference_equality : 2012.25 * 2013.75 - 2010.25 * 2015.75 = 7 := by
  sorry

end product_difference_equality_l1639_163911


namespace max_k_no_intersection_l1639_163933

noncomputable def f (x : ℝ) : ℝ := x - 1 + (Real.exp x)⁻¹

theorem max_k_no_intersection : 
  (∃ k : ℝ, ∀ x : ℝ, f x ≠ k * x - 1) ∧ 
  (∀ k : ℝ, k > 1 → ∃ x : ℝ, f x = k * x - 1) :=
sorry

end max_k_no_intersection_l1639_163933


namespace pattern_D_cannot_fold_into_cube_only_pattern_D_cannot_fold_into_cube_l1639_163955

-- Define a type for the patterns
inductive Pattern : Type
  | A : Pattern
  | B : Pattern
  | C : Pattern
  | D : Pattern

-- Define a predicate to check if a pattern can be folded into a cube
def can_fold_into_cube (p : Pattern) : Prop :=
  match p with
  | Pattern.A => true
  | Pattern.B => true
  | Pattern.C => true
  | Pattern.D => false

-- Theorem stating that Pattern D cannot be folded into a cube
theorem pattern_D_cannot_fold_into_cube :
  ¬(can_fold_into_cube Pattern.D) :=
by sorry

-- Theorem stating that Pattern D is the only pattern that cannot be folded into a cube
theorem only_pattern_D_cannot_fold_into_cube :
  ∀ (p : Pattern), ¬(can_fold_into_cube p) ↔ p = Pattern.D :=
by sorry

end pattern_D_cannot_fold_into_cube_only_pattern_D_cannot_fold_into_cube_l1639_163955


namespace power_product_equals_l1639_163919

theorem power_product_equals : 2^4 * 3^2 * 5^2 * 11 = 39600 := by sorry

end power_product_equals_l1639_163919


namespace rope_length_l1639_163944

/-- Calculates the length of a rope in centimeters given specific conditions -/
theorem rope_length : 
  let total_pieces : ℕ := 154
  let equal_pieces : ℕ := 150
  let equal_piece_length : ℕ := 75  -- in millimeters
  let remaining_piece_length : ℕ := 100  -- in millimeters
  let total_length : ℕ := equal_pieces * equal_piece_length + 
                          (total_pieces - equal_pieces) * remaining_piece_length
  total_length / 10 = 1165  -- length in centimeters
  := by sorry

end rope_length_l1639_163944


namespace puzzle_solution_l1639_163901

def concatenate (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

def special_sum (a b c : ℕ) : ℕ := 
  10000 * (a * b) + 100 * (a * c) + concatenate c b a

theorem puzzle_solution (h1 : special_sum 5 3 2 = 151022)
                        (h2 : special_sum 9 2 4 = 183652)
                        (h3 : special_sum 7 2 5 = 143547) :
  ∃ x, special_sum 7 2 x = 143547 ∧ x = 5 :=
sorry

end puzzle_solution_l1639_163901


namespace largest_common_term_under_300_l1639_163956

-- Define the first arithmetic progression
def seq1 (n : ℕ) : ℕ := 3 * n + 1

-- Define the second arithmetic progression
def seq2 (n : ℕ) : ℕ := 10 * n + 2

-- Define a function to check if a number is in both sequences
def isCommonTerm (x : ℕ) : Prop :=
  ∃ n m : ℕ, seq1 n = x ∧ seq2 m = x

-- Theorem statement
theorem largest_common_term_under_300 :
  (∀ x : ℕ, x < 300 → isCommonTerm x → x ≤ 290) ∧
  isCommonTerm 290 := by sorry

end largest_common_term_under_300_l1639_163956


namespace part1_part2_l1639_163942

/-- Defines the sequence a_n satisfying the given recurrence relation -/
def a : ℕ → ℤ
| 0 => 3  -- We define a₀ = 3 to match a₁ = 3 in the original problem
| n + 1 => 2 * a n + n^2 - 4*n + 1

/-- The arithmetic sequence b_n -/
def b (n : ℕ) : ℤ := -2*n + 3

theorem part1 : ∀ n : ℕ, a n = 2^n - n^2 + 2*n := by sorry

theorem part2 (h : ∀ n : ℕ, a n = (n + 1) * b (n + 1) - n * b n) : 
  a 0 = 1 ∧ ∀ n : ℕ, b n = -2*n + 3 := by sorry

end part1_part2_l1639_163942


namespace coach_spending_difference_l1639_163967

-- Define the purchases and discounts for each coach
def coach_A_basketballs : Nat := 10
def coach_A_basketball_price : ℝ := 29
def coach_A_soccer_balls : Nat := 5
def coach_A_soccer_ball_price : ℝ := 15
def coach_A_discount : ℝ := 0.05

def coach_B_baseballs : Nat := 14
def coach_B_baseball_price : ℝ := 2.50
def coach_B_baseball_bats : Nat := 1
def coach_B_baseball_bat_price : ℝ := 18
def coach_B_hockey_sticks : Nat := 4
def coach_B_hockey_stick_price : ℝ := 25
def coach_B_hockey_masks : Nat := 1
def coach_B_hockey_mask_price : ℝ := 72
def coach_B_discount : ℝ := 0.10

def coach_C_volleyball_nets : Nat := 8
def coach_C_volleyball_net_price : ℝ := 32
def coach_C_volleyballs : Nat := 12
def coach_C_volleyball_price : ℝ := 12
def coach_C_discount : ℝ := 0.07

-- Define the theorem
theorem coach_spending_difference :
  let coach_A_total := (1 - coach_A_discount) * (coach_A_basketballs * coach_A_basketball_price + coach_A_soccer_balls * coach_A_soccer_ball_price)
  let coach_B_total := (1 - coach_B_discount) * (coach_B_baseballs * coach_B_baseball_price + coach_B_baseball_bats * coach_B_baseball_bat_price + coach_B_hockey_sticks * coach_B_hockey_stick_price + coach_B_hockey_masks * coach_B_hockey_mask_price)
  let coach_C_total := (1 - coach_C_discount) * (coach_C_volleyball_nets * coach_C_volleyball_net_price + coach_C_volleyballs * coach_C_volleyball_price)
  coach_A_total - (coach_B_total + coach_C_total) = -227.75 := by
  sorry

end coach_spending_difference_l1639_163967


namespace sin_shift_l1639_163966

theorem sin_shift (x : ℝ) : 
  Real.sin (2 * x - π / 3) = Real.sin (2 * (x - π / 4) + π / 6) := by
  sorry

end sin_shift_l1639_163966


namespace sum_of_specific_S_values_l1639_163924

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

theorem sum_of_specific_S_values : S 17 + S 33 + S 50 = 1 := by
  sorry

end sum_of_specific_S_values_l1639_163924
