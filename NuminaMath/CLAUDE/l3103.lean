import Mathlib

namespace bottle_caps_added_l3103_310356

theorem bottle_caps_added (initial_caps : ℕ) (final_caps : ℕ) (added_caps : ℕ) : 
  initial_caps = 7 → final_caps = 14 → added_caps = final_caps - initial_caps → added_caps = 7 :=
by sorry

end bottle_caps_added_l3103_310356


namespace geometric_to_arithmetic_to_geometric_l3103_310397

/-- Represents a geometric progression with first term a and common ratio q -/
structure GeometricProgression (α : Type*) [Field α] where
  a : α
  q : α

/-- Checks if three numbers form an arithmetic progression -/
def is_arithmetic_progression {α : Type*} [Field α] (x y z : α) : Prop :=
  2 * y = x + z

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression {α : Type*} [Field α] (x y z : α) : Prop :=
  y * y = x * z

theorem geometric_to_arithmetic_to_geometric 
  {α : Type*} [Field α] (gp : GeometricProgression α) :
  is_arithmetic_progression gp.a (gp.a * gp.q + 2) (gp.a * gp.q^2) ∧
  is_geometric_progression gp.a (gp.a * gp.q + 2) (gp.a * gp.q^2 + 9) →
  (gp.a = 64 ∧ gp.q = 5/4) ∨ (gp.a = 64/25 ∧ gp.q = -5/4) :=
by sorry

end geometric_to_arithmetic_to_geometric_l3103_310397


namespace roots_product_l3103_310314

theorem roots_product (a b c d : ℝ) (h1 : 36 * a^3 - 66 * a^2 + 31 * a - 4 = 0)
  (h2 : 36 * b^3 - 66 * b^2 + 31 * b - 4 = 0)
  (h3 : 36 * c^3 - 66 * c^2 + 31 * c - 4 = 0)
  (h4 : b - a = c - b) -- arithmetic progression
  (h5 : a < b ∧ b < c) -- ordering of roots
  : a * c = 2/9 := by
  sorry

end roots_product_l3103_310314


namespace smallest_x_value_l3103_310395

theorem smallest_x_value (x : ℚ) : 
  (7 * (8 * x^2 + 8 * x + 11) = x * (8 * x - 45)) → x ≥ -7/3 :=
by sorry

end smallest_x_value_l3103_310395


namespace sum_of_coefficients_l3103_310366

def polynomial (x : ℝ) : ℝ :=
  -3 * (x^8 - 2*x^5 + 4*x^3 - 6) + 5 * (2*x^4 + 3*x^2 - x) - 2 * (3*x^6 - 7)

theorem sum_of_coefficients : 
  (polynomial 1) = 37 := by sorry

end sum_of_coefficients_l3103_310366


namespace complex_number_conditions_complex_number_on_line_l3103_310324

def z (a : ℝ) : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2)

theorem complex_number_conditions (a : ℝ) :
  (z a).re < 0 ∧ (z a).im > 0 ↔ -2 < a ∧ a < 1 :=
sorry

theorem complex_number_on_line (a : ℝ) :
  (z a).re = (z a).im ↔ a = 1 :=
sorry

end complex_number_conditions_complex_number_on_line_l3103_310324


namespace adjacent_removal_unequal_sums_l3103_310340

theorem adjacent_removal_unequal_sums (arrangement : List ℕ) : 
  arrangement.length = 2005 → 
  ∃ (i : Fin 2005), 
    ¬∃ (partition : List ℕ → List ℕ × List ℕ), 
      let remaining := arrangement.removeNth i.val ++ arrangement.removeNth ((i.val + 1) % 2005)
      (partition remaining).1.sum = (partition remaining).2.sum :=
by sorry

end adjacent_removal_unequal_sums_l3103_310340


namespace unique_exaggeration_combination_l3103_310344

/-- Represents the number of people who exaggerated the wolf's tail length --/
structure TailExaggeration where
  simple : Nat
  creative : Nat

/-- Calculates the final tail length given the number of simple and creative people --/
def finalTailLength (e : TailExaggeration) : Nat :=
  (2 ^ e.simple) * (3 ^ e.creative)

/-- Theorem stating that there is a unique combination of simple and creative people
    that results in a tail length of 864 meters --/
theorem unique_exaggeration_combination :
  ∃! e : TailExaggeration, finalTailLength e = 864 :=
sorry

end unique_exaggeration_combination_l3103_310344


namespace negation_of_universal_proposition_l3103_310370

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^3 + x ≥ 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^3 + x < 0) := by
  sorry

end negation_of_universal_proposition_l3103_310370


namespace binary_101110_to_octal_56_l3103_310371

def binary_to_octal (b : List Bool) : Nat :=
  let binary_to_decimal := b.foldl (λ acc x => 2 * acc + if x then 1 else 0) 0
  let decimal_to_octal := binary_to_decimal.digits 8
  decimal_to_octal.foldl (λ acc x => 10 * acc + x) 0

theorem binary_101110_to_octal_56 :
  binary_to_octal [true, false, true, true, true, false] = 56 := by
  sorry

end binary_101110_to_octal_56_l3103_310371


namespace expression_evaluation_l3103_310374

theorem expression_evaluation (a b : ℝ) (h1 : a = 6) (h2 : b = 2) :
  ((3 / (a + b))^2) * (a - b) = 9/16 := by
  sorry

end expression_evaluation_l3103_310374


namespace complex_number_operations_l3103_310398

theorem complex_number_operations (z₁ z₂ : ℂ) 
  (hz₁ : z₁ = 2 - 3*I) 
  (hz₂ : z₂ = (15 - 5*I) / (2 + I^2)) : 
  (z₁ - z₂ = -13 + 2*I) ∧ (z₁ * z₂ = 15 - 55*I) := by
  sorry

end complex_number_operations_l3103_310398


namespace problem_solution_l3103_310392

theorem problem_solution (x : ℝ) (h : x - Real.sqrt (x^2 - 4) + 1 / (x + Real.sqrt (x^2 - 4)) = 10) :
  x^2 - Real.sqrt (x^4 - 16) + 1 / (x^2 - Real.sqrt (x^4 - 16)) = 237/2 := by
  sorry

end problem_solution_l3103_310392


namespace product_of_geometric_terms_l3103_310385

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * q

/-- The main theorem -/
theorem product_of_geometric_terms (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n ≠ 0) →
  a 6 - a 7 ^ 2 + a 8 = 0 →
  geometric_sequence b →
  b 7 = a 7 →
  b 2 * b 8 * b 11 = 8 := by
  sorry

end product_of_geometric_terms_l3103_310385


namespace decagon_triangle_probability_l3103_310375

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of vertices required to form a triangle -/
def k : ℕ := 3

/-- The total number of possible triangles formed by choosing 3 vertices from a decagon -/
def total_triangles : ℕ := n.choose k

/-- The number of triangles with one side being a side of the decagon -/
def one_side_triangles : ℕ := n * (n - 4)

/-- The number of triangles with two sides being sides of the decagon -/
def two_side_triangles : ℕ := n

/-- The total number of favorable outcomes (triangles with at least one side being a side of the decagon) -/
def favorable_outcomes : ℕ := one_side_triangles + two_side_triangles

/-- The probability of a triangle having at least one side that is also a side of the decagon -/
def probability : ℚ := favorable_outcomes / total_triangles

theorem decagon_triangle_probability : probability = 7 / 12 := by
  sorry

end decagon_triangle_probability_l3103_310375


namespace quadratic_roots_bound_l3103_310315

theorem quadratic_roots_bound (a b c : ℝ) (x₁ x₂ : ℝ) (ha : a > 0) :
  let P : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (P x₁ = 0 ∧ P x₂ = 0) →
  (abs x₁ ≤ 1 ∧ abs x₂ ≤ 1) ↔ (a + b + c ≥ 0 ∧ a - b + c ≥ 0 ∧ a - c ≥ 0) := by
  sorry

end quadratic_roots_bound_l3103_310315


namespace coefficient_sum_equals_eight_l3103_310348

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x - 1) ^ 4

-- Define the coefficients a₀, a₁, a₂, a₃, a₄
variables (a₀ a₁ a₂ a₃ a₄ : ℝ)

-- State the theorem
theorem coefficient_sum_equals_eight :
  (∀ x, f x = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
  a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ = 8 := by
  sorry

end coefficient_sum_equals_eight_l3103_310348


namespace four_times_three_plus_two_l3103_310351

theorem four_times_three_plus_two : (4 * 3) + 2 = 14 := by
  sorry

end four_times_three_plus_two_l3103_310351


namespace courtyard_length_l3103_310380

theorem courtyard_length 
  (breadth : ℝ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (num_bricks : ℝ) 
  (h1 : breadth = 12)
  (h2 : brick_length = 0.15)
  (h3 : brick_width = 0.13)
  (h4 : num_bricks = 11076.923076923076) :
  (num_bricks * brick_length * brick_width) / breadth = 18 := by
  sorry

end courtyard_length_l3103_310380


namespace complex_number_location_l3103_310386

/-- Given a complex number z satisfying z * (-1 + 3*I) = 1 + 7*I,
    prove that z is located in the fourth quadrant of the complex plane. -/
theorem complex_number_location (z : ℂ) (h : z * (-1 + 3*I) = 1 + 7*I) :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 := by
  sorry

end complex_number_location_l3103_310386


namespace kishore_savings_percentage_l3103_310387

-- Define Mr. Kishore's expenses and savings
def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 2500
def savings : ℕ := 2000

-- Define total expenses
def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous

-- Define total salary
def total_salary : ℕ := total_expenses + savings

-- Theorem to prove
theorem kishore_savings_percentage :
  (savings : ℚ) / (total_salary : ℚ) * 100 = 10 := by
  sorry


end kishore_savings_percentage_l3103_310387


namespace percentage_soccer_players_is_12_5_l3103_310350

/-- The percentage of students who play sports that also play soccer -/
def percentage_soccer_players (total_students : ℕ) (sports_percentage : ℚ) (soccer_players : ℕ) : ℚ :=
  (soccer_players : ℚ) / (sports_percentage * total_students) * 100

/-- Theorem: The percentage of students who play sports that also play soccer is 12.5% -/
theorem percentage_soccer_players_is_12_5 :
  percentage_soccer_players 400 (52 / 100) 26 = 25 / 2 := by
  sorry

end percentage_soccer_players_is_12_5_l3103_310350


namespace tricycle_count_l3103_310328

theorem tricycle_count (total_children : ℕ) (total_wheels : ℕ) 
  (h1 : total_children = 12) 
  (h2 : total_wheels = 32) : ∃ (bicycles tricycles : ℕ), 
  bicycles + tricycles = total_children ∧ 
  2 * bicycles + 3 * tricycles = total_wheels ∧ 
  tricycles = 8 := by
sorry

end tricycle_count_l3103_310328


namespace extreme_values_imply_a_range_extreme_values_imply_a_in_range_l3103_310301

/-- A function f with two extreme values in R -/
structure TwoExtremeFunction (f : ℝ → ℝ) : Prop where
  has_two_extremes : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (∀ (x : ℝ), f x ≤ f x₁) ∧ 
    (∀ (x : ℝ), f x ≤ f x₂)

/-- The main theorem -/
theorem extreme_values_imply_a_range 
  (a : ℝ) 
  (ha : a ≠ 0) 
  (f : ℝ → ℝ)
  (hf : f = λ x => (1 + a * x^2) * Real.exp x)
  (h_two_extremes : TwoExtremeFunction f) :
  a < 0 ∨ a > 1 := by
  sorry

/-- The range of a as a set -/
def a_range : Set ℝ := {a | a < 0 ∨ a > 1}

/-- An equivalent formulation of the main theorem using sets -/
theorem extreme_values_imply_a_in_range 
  (a : ℝ) 
  (ha : a ≠ 0) 
  (f : ℝ → ℝ)
  (hf : f = λ x => (1 + a * x^2) * Real.exp x)
  (h_two_extremes : TwoExtremeFunction f) :
  a ∈ a_range := by
  sorry

end extreme_values_imply_a_range_extreme_values_imply_a_in_range_l3103_310301


namespace annual_concert_ticket_sales_l3103_310332

theorem annual_concert_ticket_sales 
  (total_tickets : ℕ) 
  (student_price non_student_price : ℚ) 
  (total_revenue : ℚ) 
  (h1 : total_tickets = 150)
  (h2 : student_price = 5)
  (h3 : non_student_price = 8)
  (h4 : total_revenue = 930) :
  ∃ (student_tickets : ℕ), 
    student_tickets = 90 ∧ 
    ∃ (non_student_tickets : ℕ), 
      student_tickets + non_student_tickets = total_tickets ∧
      student_price * student_tickets + non_student_price * non_student_tickets = total_revenue :=
by sorry

end annual_concert_ticket_sales_l3103_310332


namespace family_average_age_l3103_310379

theorem family_average_age (grandparents_avg : ℝ) (parents_avg : ℝ) (grandchildren_avg : ℝ)
  (h1 : grandparents_avg = 64)
  (h2 : parents_avg = 39)
  (h3 : grandchildren_avg = 6) :
  (2 * grandparents_avg + 2 * parents_avg + 3 * grandchildren_avg) / 7 = 32 := by
  sorry

end family_average_age_l3103_310379


namespace prime_sum_ways_8_l3103_310335

/-- A function that returns the number of unique ways to sum prime numbers to form a given natural number,
    where the prime numbers in the sum are in non-decreasing order. -/
def prime_sum_ways (n : ℕ) : ℕ := sorry

/-- A function that checks if a list of natural numbers is a valid prime sum for a given number,
    where the numbers in the list are prime and in non-decreasing order. -/
def is_valid_prime_sum (n : ℕ) (sum : List ℕ) : Prop := sorry

theorem prime_sum_ways_8 : prime_sum_ways 8 = 2 := by sorry

end prime_sum_ways_8_l3103_310335


namespace binomial_square_condition_l3103_310310

/-- If 9x^2 + 30x + a is the square of a binomial, then a = 25 -/
theorem binomial_square_condition (a : ℝ) : 
  (∃ b c : ℝ, ∀ x : ℝ, 9*x^2 + 30*x + a = (b*x + c)^2) → a = 25 := by
sorry

end binomial_square_condition_l3103_310310


namespace smallest_n_with_non_decimal_digit_in_g_l3103_310376

/-- Sum of digits in base-three representation -/
def f (n : ℕ) : ℕ := sorry

/-- Sum of digits in base-six representation of f(n) -/
def g (n : ℕ) : ℕ := sorry

/-- Check if a number in base-twelve contains a digit not in {0, 1, ..., 9} -/
def has_non_decimal_digit (n : ℕ) : Prop := sorry

/-- The main theorem -/
theorem smallest_n_with_non_decimal_digit_in_g : 
  (∃ n : ℕ, n > 0 ∧ has_non_decimal_digit (g n)) ∧ 
  (∀ m : ℕ, m > 0 ∧ m < 32 → ¬has_non_decimal_digit (g m)) ∧
  has_non_decimal_digit (g 32) := by sorry

end smallest_n_with_non_decimal_digit_in_g_l3103_310376


namespace perfect_square_condition_l3103_310312

theorem perfect_square_condition (k : ℝ) : 
  (∃ (p : ℝ → ℝ), ∀ x, x^2 + 6*x + k^2 = (p x)^2) → (k = 3 ∨ k = -3) :=
by sorry

end perfect_square_condition_l3103_310312


namespace propositions_P_and_Q_l3103_310342

theorem propositions_P_and_Q : 
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 → 1/a + 1/b > 3) ∧
  (∀ x : ℝ, x^2 - x + 1 ≥ 0) := by sorry

end propositions_P_and_Q_l3103_310342


namespace sqrt_225_equals_15_l3103_310360

theorem sqrt_225_equals_15 : Real.sqrt 225 = 15 := by
  sorry

end sqrt_225_equals_15_l3103_310360


namespace probability_four_twos_in_five_rolls_l3103_310384

theorem probability_four_twos_in_five_rolls (p : ℝ) :
  p = 1 / 8 →
  (5 : ℝ) * p^4 * (1 - p) = 35 / 32768 := by
  sorry

end probability_four_twos_in_five_rolls_l3103_310384


namespace cubic_roots_sum_of_cubes_l3103_310321

theorem cubic_roots_sum_of_cubes (p q : ℝ) (r s : ℂ) : 
  (r^3 - p*r^2 + q*r - p = 0) → 
  (s^3 - p*s^2 + q*s - p = 0) → 
  r^3 + s^3 = p^3 := by
sorry

end cubic_roots_sum_of_cubes_l3103_310321


namespace robot_fifth_minute_distance_l3103_310354

def robot_distance (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 2
  | k + 1 => 2 * robot_distance k

theorem robot_fifth_minute_distance :
  robot_distance 5 = 32 := by
  sorry

end robot_fifth_minute_distance_l3103_310354


namespace function_range_l3103_310349

theorem function_range (a : ℝ) (h_a : a ≠ 0) :
  (∀ x : ℝ, a * Real.sin x - 1/2 * Real.cos (2*x) + a - 3/a + 1/2 ≤ 0) →
  (0 < a ∧ a ≤ 1) :=
by sorry

end function_range_l3103_310349


namespace unique_divisible_digit_l3103_310345

def is_divisible (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def seven_digit_number (A : ℕ) : ℕ := 3538080 + A

theorem unique_divisible_digit :
  ∃! A : ℕ,
    A < 10 ∧
    is_divisible (seven_digit_number A) 2 ∧
    is_divisible (seven_digit_number A) 3 ∧
    is_divisible (seven_digit_number A) 4 ∧
    is_divisible (seven_digit_number A) 5 ∧
    is_divisible (seven_digit_number A) 6 ∧
    is_divisible (seven_digit_number A) 8 ∧
    is_divisible (seven_digit_number A) 9 ∧
    A = 0 :=
by sorry

end unique_divisible_digit_l3103_310345


namespace wire_pieces_lengths_l3103_310341

/-- Represents the lengths of four pieces of wire --/
structure WirePieces where
  piece1 : ℝ
  piece2 : ℝ
  piece3 : ℝ
  piece4 : ℝ

/-- The total length of the wire is 72 feet --/
def totalLength : ℝ := 72

/-- Theorem stating the correct lengths of the wire pieces --/
theorem wire_pieces_lengths : ∃ (w : WirePieces),
  w.piece1 = 14.75 ∧
  w.piece2 = 11.75 ∧
  w.piece3 = 21.5 ∧
  w.piece4 = 24 ∧
  w.piece1 = w.piece2 + 3 ∧
  w.piece3 = 2 * w.piece2 - 2 ∧
  w.piece4 = (w.piece1 + w.piece2 + w.piece3) / 2 ∧
  w.piece1 + w.piece2 + w.piece3 + w.piece4 = totalLength := by
  sorry

end wire_pieces_lengths_l3103_310341


namespace smallest_prime_factor_of_2985_l3103_310326

theorem smallest_prime_factor_of_2985 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2985 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2985 → p ≤ q :=
  sorry

end smallest_prime_factor_of_2985_l3103_310326


namespace apex_at_vertex_a_l3103_310390

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle with pillars -/
structure TriangleWithPillars where
  A : Point3D
  B : Point3D
  C : Point3D
  heightA : ℝ
  heightB : ℝ
  heightC : ℝ

/-- Check if three points form an equilateral triangle on the ground (z = 0) -/
def isEquilateral (t : TriangleWithPillars) : Prop :=
  let d1 := (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2
  let d2 := (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2
  let d3 := (t.C.x - t.A.x)^2 + (t.C.y - t.A.y)^2
  d1 = d2 ∧ d2 = d3 ∧ t.A.z = 0 ∧ t.B.z = 0 ∧ t.C.z = 0

/-- Find the point directly below the apex of the inclined plane -/
def apexProjection (t : TriangleWithPillars) : Point3D :=
  { x := t.A.x, y := t.A.y, z := 0 }

/-- Theorem: The apex projection is at vertex A for the given triangle -/
theorem apex_at_vertex_a (t : TriangleWithPillars) :
  isEquilateral t ∧ t.heightA = 10 ∧ t.heightB = 8 ∧ t.heightC = 6 →
  apexProjection t = t.A :=
by sorry

end apex_at_vertex_a_l3103_310390


namespace apartment_keys_theorem_l3103_310372

/-- The number of keys needed for apartment complexes -/
def keys_needed (num_complexes : ℕ) (apartments_per_complex : ℕ) (keys_per_apartment : ℕ) : ℕ :=
  num_complexes * apartments_per_complex * keys_per_apartment

/-- Theorem: Given two apartment complexes with 12 apartments each, 
    and requiring 3 keys per apartment, the total number of keys needed is 72 -/
theorem apartment_keys_theorem :
  keys_needed 2 12 3 = 72 := by
  sorry


end apartment_keys_theorem_l3103_310372


namespace inverse_proportion_l3103_310355

/-- Given that x is inversely proportional to y, if x = 4 when y = -2, 
    then x = 4/5 when y = -10 -/
theorem inverse_proportion (x y : ℝ) (h : ∃ k : ℝ, ∀ x y, x * y = k) 
  (h1 : 4 * (-2) = x * y) : 
  x * (-10) = 4/5 * (-10) := by sorry

end inverse_proportion_l3103_310355


namespace square_is_quadratic_and_power_l3103_310388

/-- A function f: ℝ → ℝ is a power function if there exists a real number a such that f(x) = x^a for all x in the domain of f. -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x, f x = x ^ a

/-- A function f: ℝ → ℝ is a quadratic function if there exist real numbers a, b, and c with a ≠ 0 such that f(x) = ax^2 + bx + c for all x in ℝ. -/
def IsQuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x^2 is both a quadratic function and a power function. -/
theorem square_is_quadratic_and_power :
  let f : ℝ → ℝ := fun x ↦ x^2
  IsQuadraticFunction f ∧ IsPowerFunction f := by
  sorry

end square_is_quadratic_and_power_l3103_310388


namespace second_number_is_204_l3103_310391

def number_list : List ℕ := [201, 204, 205, 206, 209, 209, 210, 212, 212]

theorem second_number_is_204 : number_list[1] = 204 := by
  sorry

end second_number_is_204_l3103_310391


namespace f_g_properties_l3103_310394

/-- The absolute value function -/
def f (m : ℝ) (x : ℝ) : ℝ := |x - m|

/-- The function g defined in terms of f -/
def g (m : ℝ) (x : ℝ) : ℝ := 2 * f m x - f m (x + m)

/-- The theorem stating the properties of f and g -/
theorem f_g_properties :
  ∃ (m : ℝ), m > 0 ∧
  (∀ x, g m x ≥ -1) ∧
  (∃ x, g m x = -1) ∧
  m = 1 ∧
  ∀ (a b : ℝ), |a| < m → |b| < m → a ≠ 0 → f m (a * b) > |a| * f m (b / a) :=
sorry

end f_g_properties_l3103_310394


namespace smallest_candy_count_l3103_310383

theorem smallest_candy_count : ∃ (n : ℕ), 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (n + 6) % 9 = 0 ∧ 
  (n - 9) % 6 = 0 ∧ 
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (m + 6) % 9 = 0 ∧ (m - 9) % 6 = 0) → False) ∧
  n = 111 := by
sorry

end smallest_candy_count_l3103_310383


namespace base_difference_in_right_trapezoid_l3103_310361

/-- A right trapezoid with specific properties -/
structure RightTrapezoid where
  /-- The measure of the largest angle in degrees -/
  largest_angle : ℝ
  /-- The length of the shorter leg -/
  shorter_leg : ℝ
  /-- The length of the longer base -/
  longer_base : ℝ
  /-- The length of the shorter base -/
  shorter_base : ℝ
  /-- Condition that the largest angle is 135° -/
  largest_angle_eq : largest_angle = 135
  /-- Condition that the shorter leg is 18 -/
  shorter_leg_eq : shorter_leg = 18
  /-- Condition that it's a right trapezoid (one angle is 90°) -/
  is_right : True

/-- Theorem stating the difference between bases in a right trapezoid with specific properties -/
theorem base_difference_in_right_trapezoid (t : RightTrapezoid) : 
  t.longer_base - t.shorter_base = 18 := by
  sorry

end base_difference_in_right_trapezoid_l3103_310361


namespace min_value_2a_plus_b_min_value_2a_plus_b_equality_l3103_310305

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 1) + 2 / (b - 2) = 1 / 2) : 
  2 * a + b ≥ 16 := by
  sorry

theorem min_value_2a_plus_b_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 1) + 2 / (b - 2) = 1 / 2) : 
  (2 * a + b = 16) ↔ (a = 3 ∧ b = 10) := by
  sorry

end min_value_2a_plus_b_min_value_2a_plus_b_equality_l3103_310305


namespace nine_points_chords_l3103_310308

/-- The number of different chords that can be drawn by connecting two points
    out of n points on the circumference of a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: There are 36 different chords that can be drawn by connecting two
    points out of nine points on the circumference of a circle -/
theorem nine_points_chords : num_chords 9 = 36 := by
  sorry

end nine_points_chords_l3103_310308


namespace lcm_problem_l3103_310309

theorem lcm_problem (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 28) :
  ∃ (a' c' : ℕ), a' ∣ a ∧ c' ∣ c ∧ Nat.lcm a' c' = 35 ∧ 
  ∀ (x y : ℕ), x ∣ a → y ∣ c → Nat.lcm x y ≥ 35 :=
sorry

end lcm_problem_l3103_310309


namespace line_slope_m_l3103_310333

theorem line_slope_m (m : ℝ) : 
  m > 0 → 
  ((m - 4) / (2 - m) = 2 * m) →
  m = (3 + Real.sqrt 41) / 4 :=
by
  sorry

end line_slope_m_l3103_310333


namespace equation_solution_l3103_310381

theorem equation_solution : 
  ∃ y : ℚ, 3 * (4 * y - 5) + 1 = -3 * (2 - 5 * y) ∧ y = -8/3 := by
  sorry

end equation_solution_l3103_310381


namespace rectangle_diagonal_l3103_310359

/-- Given a rectangle with perimeter 72 meters and length-to-width ratio of 3:2,
    prove that its diagonal length is √673.92 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 72) →
  (length / width = 3 / 2) →
  Real.sqrt (length^2 + width^2) = Real.sqrt 673.92 := by
  sorry

end rectangle_diagonal_l3103_310359


namespace math_reading_homework_difference_l3103_310322

theorem math_reading_homework_difference (reading_pages math_pages : ℕ) 
  (h1 : reading_pages = 12) 
  (h2 : math_pages = 23) : 
  math_pages - reading_pages = 11 := by
  sorry

end math_reading_homework_difference_l3103_310322


namespace min_value_constrained_min_value_achieved_l3103_310358

theorem min_value_constrained (x y : ℝ) (h : 2 * x + 8 * y = 3) :
  x^2 + 4 * y^2 - 2 * x ≥ -19/20 := by
  sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, 2 * x + 8 * y = 3 ∧ x^2 + 4 * y^2 - 2 * x < -19/20 + ε := by
  sorry

end min_value_constrained_min_value_achieved_l3103_310358


namespace set_operations_l3103_310338

def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

theorem set_operations :
  (Set.compl (A ∩ B) = {x | x < 3 ∨ x ≥ 6}) ∧
  ((Set.compl B ∪ A) = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9}) := by
  sorry

end set_operations_l3103_310338


namespace least_possible_difference_l3103_310318

theorem least_possible_difference (x y z : ℤ) : 
  x < y ∧ y < z ∧ 
  y - x > 5 ∧ 
  Even x ∧ Odd y ∧ Odd z →
  ∀ w, w = z - x → w ≥ 9 :=
by sorry

end least_possible_difference_l3103_310318


namespace loan_amount_correct_l3103_310334

/-- The amount of money (in Rs.) that A lent to B -/
def loan_amount : ℝ := 3500

/-- B's net interest rate per annum (as a decimal) -/
def net_interest_rate : ℝ := 0.01

/-- B's gain in 3 years (in Rs.) -/
def gain_in_three_years : ℝ := 105

/-- Proves that the loan amount is correct given the conditions -/
theorem loan_amount_correct : 
  loan_amount * net_interest_rate * 3 = gain_in_three_years :=
by sorry

end loan_amount_correct_l3103_310334


namespace income_expenditure_ratio_theorem_l3103_310393

/-- Represents the financial data of a person -/
structure FinancialData where
  income : ℕ
  savings : ℕ

/-- Calculates the expenditure given income and savings -/
def calculateExpenditure (data : FinancialData) : ℕ :=
  data.income - data.savings

/-- Calculates the ratio of income to expenditure -/
def incomeToExpenditureRatio (data : FinancialData) : Rat :=
  data.income / (calculateExpenditure data)

/-- Theorem stating that for a person with an income of 20000 and savings of 4000,
    the ratio of income to expenditure is 5/4 -/
theorem income_expenditure_ratio_theorem (data : FinancialData)
    (h1 : data.income = 20000)
    (h2 : data.savings = 4000) :
    incomeToExpenditureRatio data = 5 / 4 := by
  sorry


end income_expenditure_ratio_theorem_l3103_310393


namespace circle_symmetry_line_l3103_310368

/-- A circle in the Cartesian plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- A line in the Cartesian plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Symmetry of a circle with respect to a line -/
def isSymmetric (c : Circle) (l : Line) : Prop := sorry

/-- The main theorem -/
theorem circle_symmetry_line (a : ℝ) :
  let c : Circle := { equation := fun x y => x^2 + y^2 - 4*x - 8*y + 19 = 0 }
  let l : Line := { equation := fun x y => x + 2*y - a = 0 }
  isSymmetric c l → a = 10 := by sorry

end circle_symmetry_line_l3103_310368


namespace range_of_m_l3103_310331

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, Real.exp (|2*x + 1|) + m ≥ 0) ↔ m ≥ -1 := by sorry

end range_of_m_l3103_310331


namespace box_cubes_required_l3103_310327

theorem box_cubes_required (length width height cube_volume : ℕ) 
  (h1 : length = 12)
  (h2 : width = 16)
  (h3 : height = 6)
  (h4 : cube_volume = 3) : 
  (length * width * height) / cube_volume = 384 := by
  sorry

end box_cubes_required_l3103_310327


namespace super_soup_revenue_theorem_l3103_310313

def super_soup_revenue (
  initial_stores : ℕ)
  (initial_avg_revenue : ℝ)
  (new_stores_2019 : ℕ)
  (new_revenue_2019 : ℝ)
  (closed_stores_2019 : ℕ)
  (closed_revenue_2019 : ℝ)
  (closed_expense_2019 : ℝ)
  (new_stores_2020 : ℕ)
  (new_revenue_2020 : ℝ)
  (closed_stores_2020 : ℕ)
  (closed_revenue_2020 : ℝ)
  (closed_expense_2020 : ℝ)
  (avg_expense : ℝ) : ℝ :=
  let initial_revenue := initial_stores * initial_avg_revenue
  let revenue_2019 := initial_revenue + new_stores_2019 * new_revenue_2019 - closed_stores_2019 * closed_revenue_2019
  let net_revenue_2019 := revenue_2019 + closed_stores_2019 * (closed_revenue_2019 - closed_expense_2019)
  let stores_2019 := initial_stores + new_stores_2019 - closed_stores_2019
  let revenue_2020 := net_revenue_2019 + new_stores_2020 * new_revenue_2020 - closed_stores_2020 * closed_revenue_2020
  let net_revenue_2020 := revenue_2020 + closed_stores_2020 * (closed_expense_2020 - closed_revenue_2020)
  let final_stores := stores_2019 + new_stores_2020 - closed_stores_2020
  net_revenue_2020 - final_stores * avg_expense

theorem super_soup_revenue_theorem :
  super_soup_revenue 23 500000 5 450000 2 300000 350000 10 600000 6 350000 380000 400000 = 5130000 := by
  sorry

end super_soup_revenue_theorem_l3103_310313


namespace probability_four_blue_marbles_l3103_310364

def total_marbles : ℕ := 20
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 12
def num_trials : ℕ := 8
def num_blue_picked : ℕ := 4

theorem probability_four_blue_marbles :
  (Nat.choose num_trials num_blue_picked) *
  (blue_marbles / total_marbles : ℚ) ^ num_blue_picked *
  (red_marbles / total_marbles : ℚ) ^ (num_trials - num_blue_picked) =
  90720 / 390625 := by
  sorry

end probability_four_blue_marbles_l3103_310364


namespace wire_cut_problem_l3103_310378

theorem wire_cut_problem (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) :
  total_length = 90 →
  ratio = 2 / 7 →
  shorter_length = ratio * (total_length - shorter_length) →
  shorter_length = 20 := by
  sorry

end wire_cut_problem_l3103_310378


namespace geometric_sequence_and_parabola_vertex_l3103_310396

/-- Given that a, b, c, and d form a geometric sequence, and the vertex of the curve y = x^2 - 2x + 3 is (b, c), then ad = 2 -/
theorem geometric_sequence_and_parabola_vertex (a b c d : ℝ) : 
  (∃ (r : ℝ), r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric sequence condition
  (∀ x : ℝ, x^2 - 2*x + 3 ≥ c) →  -- vertex condition
  (b^2 - 2*b + 3 = c) →  -- vertex condition
  a * d = 2 := by
sorry

end geometric_sequence_and_parabola_vertex_l3103_310396


namespace sphere_radius_from_great_circle_area_l3103_310302

theorem sphere_radius_from_great_circle_area (A : ℝ) (R : ℝ) :
  A = 4 * Real.pi → A = Real.pi * R^2 → R = 2 := by
  sorry

end sphere_radius_from_great_circle_area_l3103_310302


namespace largest_whole_number_nine_times_less_than_150_l3103_310357

theorem largest_whole_number_nine_times_less_than_150 :
  ∃ (x : ℤ), x = 16 ∧ (∀ y : ℤ, 9 * y < 150 → y ≤ x) :=
by
  sorry

end largest_whole_number_nine_times_less_than_150_l3103_310357


namespace cone_base_diameter_l3103_310337

theorem cone_base_diameter (r : ℝ) (h1 : r > 0) : 
  (π * r^2 + π * r * (2 * r) = 3 * π) → 2 * r = 2 := by
  sorry

end cone_base_diameter_l3103_310337


namespace simultaneous_equations_solution_l3103_310330

theorem simultaneous_equations_solution :
  ∃! (x y : ℚ), 3 * x - 4 * y = 11 ∧ 9 * x + 6 * y = 33 :=
by
  sorry

end simultaneous_equations_solution_l3103_310330


namespace parabola_properties_l3103_310365

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the focus F
def focus : ℝ × ℝ := (4, 0)

-- Define a point M on the parabola
def point_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2

-- Define N as a point on the y-axis
def point_on_y_axis (N : ℝ × ℝ) : Prop :=
  N.1 = 0

-- Define M as the midpoint of FN
def is_midpoint (F M N : ℝ × ℝ) : Prop :=
  M.1 = (F.1 + N.1) / 2 ∧ M.2 = (F.2 + N.2) / 2

-- Main theorem
theorem parabola_properties (M N : ℝ × ℝ) 
  (h1 : point_on_parabola M)
  (h2 : point_on_y_axis N)
  (h3 : is_midpoint focus M N) :
  (∀ x y, y^2 = 16 * x → x = -4 → False) ∧  -- Directrix equation
  (Real.sqrt ((focus.1 - N.1)^2 + (focus.2 - N.2)^2) = 12) ∧  -- |FN| = 12
  (1/2 * focus.1 * N.2 = 16 * Real.sqrt 2) :=  -- Area of triangle ONF
by sorry

end parabola_properties_l3103_310365


namespace set_equality_proof_l3103_310347

theorem set_equality_proof (M N : Set ℕ) : M = {3, 2} → N = {2, 3} → M = N := by
  sorry

end set_equality_proof_l3103_310347


namespace minimum_distance_between_curves_l3103_310329

noncomputable def min_distance : ℝ := Real.sqrt 2 / 2 * (1 - Real.log 2)

theorem minimum_distance_between_curves :
  ∃ (a b : ℝ),
    (1/2 : ℝ) * Real.exp a = (1/2 : ℝ) * Real.exp a ∧
    b = b ∧
    ∀ (x y : ℝ),
      (1/2 : ℝ) * Real.exp x = (1/2 : ℝ) * Real.exp x →
      y = y →
      Real.sqrt ((x - y)^2 + ((1/2 : ℝ) * Real.exp x - y)^2) ≥ min_distance :=
by sorry

end minimum_distance_between_curves_l3103_310329


namespace rectangle_length_fraction_l3103_310339

theorem rectangle_length_fraction (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ) :
  square_area = 2025 →
  rectangle_area = 180 →
  rectangle_breadth = 10 →
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 :=
by sorry

end rectangle_length_fraction_l3103_310339


namespace grapes_needed_theorem_l3103_310319

/-- The amount of grapes needed in a year after a 20% increase in production -/
def grapes_needed_after_increase (initial_usage : ℝ) : ℝ :=
  2 * (initial_usage * 1.2)

/-- Theorem stating that given an initial grape usage of 90 kg per 6 months 
    and a 20% increase in production, the total amount of grapes needed in a year is 216 kg -/
theorem grapes_needed_theorem :
  grapes_needed_after_increase 90 = 216 := by
  sorry

#eval grapes_needed_after_increase 90

end grapes_needed_theorem_l3103_310319


namespace last_remaining_number_l3103_310362

/-- Represents the process of skipping and marking numbers -/
def josephus_process (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that for n = 50, the last remaining number is 49 -/
theorem last_remaining_number : josephus_process 50 = 49 := by
  sorry

end last_remaining_number_l3103_310362


namespace basketball_free_throws_l3103_310306

theorem basketball_free_throws 
  (two_point_shots : ℕ) 
  (three_point_shots : ℕ) 
  (free_throws : ℕ) : 
  (3 * three_point_shots = 2 * two_point_shots) → 
  (free_throws = two_point_shots + 1) → 
  (2 * two_point_shots + 3 * three_point_shots + free_throws = 84) → 
  free_throws = 16 := by
sorry

end basketball_free_throws_l3103_310306


namespace quadratic_equation_at_negative_two_l3103_310352

theorem quadratic_equation_at_negative_two :
  let x : ℤ := -2
  x^2 + 6*x - 10 = -18 := by sorry

end quadratic_equation_at_negative_two_l3103_310352


namespace ab_nonzero_sufficient_for_a_nonzero_l3103_310377

theorem ab_nonzero_sufficient_for_a_nonzero (a b : ℝ) : 
  (∀ a b, a * b ≠ 0 → a ≠ 0) ∧ 
  ¬(∀ a b, a ≠ 0 → a * b ≠ 0) := by
  sorry

end ab_nonzero_sufficient_for_a_nonzero_l3103_310377


namespace range_of_a_l3103_310316

-- Define the conditions
def condition_p (a : ℝ) : Prop := ∃ m : ℝ, m ∈ Set.Icc (-1) 1 ∧ a^2 - 5*a + 5 ≥ m + 2

def condition_q (a : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + a*x₁ + 2 = 0 ∧ x₂^2 + a*x₂ + 2 = 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (condition_p a ∨ condition_q a) ∧ ¬(condition_p a ∧ condition_q a) →
  a ≤ 1 ∨ (2 * Real.sqrt 2 ≤ a ∧ a < 4) :=
sorry

end range_of_a_l3103_310316


namespace document_download_income_increase_sales_target_increase_basketball_success_rate_l3103_310307

-- Define percentages as real numbers between 0 and 1
def Percentage := { x : ℝ // 0 ≤ x ∧ x ≤ 1 }

-- 1. Document download percentages
theorem document_download (a b : Percentage) :
  (a.val + b.val = 1) → ((a.val = 0.58 ∧ b.val = 0.42) ∨ (a.val = 0.42 ∧ b.val = 0.58)) :=
sorry

-- 2. Xiao Ming's income increase
theorem income_increase (last_year current_year : ℝ) (h : current_year = 1.24 * last_year) :
  current_year > last_year :=
sorry

-- 3. Shopping mall sales target
theorem sales_target_increase (august_target september_target : ℝ) 
  (h : september_target = 1.5 * august_target) :
  september_target > 0.5 * august_target :=
sorry

-- 4. Luo Luo's basketball shot success rate
theorem basketball_success_rate (attempts successes : ℕ) :
  attempts = 5 ∧ successes = 5 → (successes : ℝ) / attempts = 1 :=
sorry

end document_download_income_increase_sales_target_increase_basketball_success_rate_l3103_310307


namespace lisa_caffeine_over_goal_l3103_310311

/-- The amount of caffeine Lisa consumed over her goal -/
def caffeine_over_goal (caffeine_per_cup : ℕ) (daily_goal : ℕ) (cups_consumed : ℕ) : ℕ :=
  max ((caffeine_per_cup * cups_consumed) - daily_goal) 0

/-- Theorem stating that Lisa consumed 40 mg of caffeine over her goal -/
theorem lisa_caffeine_over_goal :
  caffeine_over_goal 80 200 3 = 40 := by
  sorry

end lisa_caffeine_over_goal_l3103_310311


namespace mark_remaining_hours_l3103_310353

def sick_days : ℕ := 10
def vacation_days : ℕ := 10
def hours_per_day : ℕ := 8
def used_fraction : ℚ := 1/2

theorem mark_remaining_hours : 
  (sick_days + vacation_days) * (1 - used_fraction) * hours_per_day = 80 := by
  sorry

end mark_remaining_hours_l3103_310353


namespace gaussian_function_properties_l3103_310346

-- Define the Gaussian function (floor function)
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Theorem statement
theorem gaussian_function_properties :
  -- 1. The range of floor is ℤ
  (∀ n : ℤ, ∃ x : ℝ, floor x = n) ∧
  -- 2. floor is not an odd function
  (∃ x : ℝ, floor (-x) ≠ -floor x) ∧
  -- 3. x - floor x is periodic with period 1
  (∀ x : ℝ, x - floor x = (x + 1) - floor (x + 1)) ∧
  -- 4. floor is not monotonically increasing on ℝ
  (∃ x y : ℝ, x < y ∧ floor x > floor y) :=
by sorry

end gaussian_function_properties_l3103_310346


namespace no_valid_assignment_l3103_310367

/-- Represents a mapping of characters to digits -/
def DigitAssignment := Char → Nat

/-- Checks if a DigitAssignment is valid for the given cryptarithmic problem -/
def is_valid_assignment (assignment : DigitAssignment) : Prop :=
  let s := assignment 'S'
  let t := assignment 'T'
  let i := assignment 'I'
  let k := assignment 'K'
  let m := assignment 'M'
  let a := assignment 'A'
  (s ≠ 0) ∧ 
  (m ≠ 0) ∧
  (s ≠ t) ∧ (s ≠ i) ∧ (s ≠ k) ∧ (s ≠ m) ∧ (s ≠ a) ∧
  (t ≠ i) ∧ (t ≠ k) ∧ (t ≠ m) ∧ (t ≠ a) ∧
  (i ≠ k) ∧ (i ≠ m) ∧ (i ≠ a) ∧
  (k ≠ m) ∧ (k ≠ a) ∧
  (m ≠ a) ∧
  (s < 10) ∧ (t < 10) ∧ (i < 10) ∧ (k < 10) ∧ (m < 10) ∧ (a < 10) ∧
  (10000 * s + 1000 * t + 100 * i + 10 * k + s +
   10000 * s + 1000 * t + 100 * i + 10 * k + s =
   100000 * m + 10000 * a + 1000 * s + 100 * t + 10 * i + k + s)

theorem no_valid_assignment : ¬∃ (assignment : DigitAssignment), is_valid_assignment assignment :=
sorry

end no_valid_assignment_l3103_310367


namespace fraction_equality_l3103_310389

theorem fraction_equality (a b : ℝ) (h : (a - b) / b = 3 / 5) : a / b = 8 / 5 := by
  sorry

end fraction_equality_l3103_310389


namespace count_divisible_by_3_5_7_60_l3103_310325

def count_divisible (n : ℕ) (d : ℕ) : ℕ := n / d

def count_divisible_by_3_5_7 (upper_bound : ℕ) : ℕ :=
  let div3 := count_divisible upper_bound 3
  let div5 := count_divisible upper_bound 5
  let div7 := count_divisible upper_bound 7
  let div3_5 := count_divisible upper_bound 15
  let div3_7 := count_divisible upper_bound 21
  let div5_7 := count_divisible upper_bound 35
  div3 + div5 + div7 - (div3_5 + div3_7 + div5_7)

theorem count_divisible_by_3_5_7_60 : count_divisible_by_3_5_7 60 = 33 := by
  sorry

end count_divisible_by_3_5_7_60_l3103_310325


namespace difference_percentages_l3103_310343

theorem difference_percentages : (800 * 75 / 100) - (1200 * 7 / 8) = 450 := by
  sorry

end difference_percentages_l3103_310343


namespace inscribed_circle_square_area_l3103_310399

/-- A circle inscribed in a square with sides parallel to the axes -/
structure InscribedCircle where
  /-- The equation of the circle: x^2 + y^2 + 2x - 8y = 0 -/
  eq : ∀ (x y : ℝ), x^2 + y^2 + 2*x - 8*y = 0

/-- The area of the square that inscribes the circle -/
def squareArea (c : InscribedCircle) : ℝ := 68

/-- Theorem: The area of the square that inscribes the circle is 68 square units -/
theorem inscribed_circle_square_area (c : InscribedCircle) : squareArea c = 68 := by
  sorry

end inscribed_circle_square_area_l3103_310399


namespace equation_solution_l3103_310317

theorem equation_solution (x : ℝ) :
  x ≠ -1 → x ≠ 1 → (x / (x + 1) = 2 / (x^2 - 1)) → x = 2 :=
by
  sorry

end equation_solution_l3103_310317


namespace units_digit_problem_l3103_310382

theorem units_digit_problem : ∃ n : ℕ, n % 10 = 7 ∧ n = (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1) + 2 := by
  sorry

end units_digit_problem_l3103_310382


namespace arithmetic_sequence_sum_modulo_l3103_310304

theorem arithmetic_sequence_sum_modulo (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 5 →
  aₙ = 145 →
  d = 5 →
  n = (aₙ - a₁) / d + 1 →
  (n * (a₁ + aₙ) / 2) % 12 = 3 := by
  sorry

end arithmetic_sequence_sum_modulo_l3103_310304


namespace jack_jogging_speed_l3103_310373

-- Define the given conditions
def melt_time : ℚ := 10 / 60  -- 10 minutes converted to hours
def num_blocks : ℕ := 16
def block_length : ℚ := 1 / 8  -- in miles

-- Define the total distance
def total_distance : ℚ := num_blocks * block_length

-- Define the required speed
def required_speed : ℚ := total_distance / melt_time

-- Theorem statement
theorem jack_jogging_speed :
  required_speed = 12 := by sorry

end jack_jogging_speed_l3103_310373


namespace high_school_nine_games_l3103_310300

/-- The number of teams in the league -/
def num_teams : ℕ := 9

/-- The number of games each team plays against non-league opponents -/
def non_league_games : ℕ := 6

/-- The total number of games played in a season -/
def total_games : ℕ := 126

/-- Theorem stating the total number of games in a season -/
theorem high_school_nine_games :
  (num_teams * (num_teams - 1)) + (num_teams * non_league_games) = total_games :=
sorry

end high_school_nine_games_l3103_310300


namespace line_m_equation_l3103_310336

-- Define the xy-plane
structure XYPlane where
  x : ℝ
  y : ℝ

-- Define a line in the xy-plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given lines and points
def line_l : Line := { a := 3, b := -1, c := 0 }
def point_P : XYPlane := { x := -3, y := 2 }
def point_P'' : XYPlane := { x := 2, y := -1 }

-- Define the reflection operation
def reflect (p : XYPlane) (l : Line) : XYPlane :=
  sorry

-- State the theorem
theorem line_m_equation :
  ∃ (line_m : Line),
    (line_m.a ≠ line_l.a ∨ line_m.b ≠ line_l.b) ∧
    (line_m.a * 0 + line_m.b * 0 + line_m.c = 0) ∧
    (∃ (point_P' : XYPlane),
      reflect point_P line_l = point_P' ∧
      reflect point_P' line_m = point_P'') ∧
    line_m.a = 1 ∧ line_m.b = 3 ∧ line_m.c = 0 :=
  sorry

end line_m_equation_l3103_310336


namespace regular_polygon_exterior_angle_l3103_310323

theorem regular_polygon_exterior_angle (n : ℕ) (exterior_angle : ℝ) :
  n ≥ 3 → exterior_angle = 36 → n = 10 := by sorry

end regular_polygon_exterior_angle_l3103_310323


namespace smiley_red_smile_l3103_310369

def tulip_smiley (red_smile : ℕ) : Prop :=
  let red_eyes : ℕ := 8 * 2
  let yellow_background : ℕ := 9 * red_smile
  red_eyes + red_smile + yellow_background = 196

theorem smiley_red_smile :
  ∃ (red_smile : ℕ), tulip_smiley red_smile ∧ red_smile = 18 :=
by sorry

end smiley_red_smile_l3103_310369


namespace sum_of_four_digit_odd_and_multiples_of_ten_l3103_310303

/-- The number of four-digit odd numbers -/
def A : ℕ := 4500

/-- The number of four-digit multiples of 10 -/
def B : ℕ := 900

/-- The sum of four-digit odd numbers and four-digit multiples of 10 is 5400 -/
theorem sum_of_four_digit_odd_and_multiples_of_ten : A + B = 5400 := by
  sorry

end sum_of_four_digit_odd_and_multiples_of_ten_l3103_310303


namespace matrix_multiplication_l3103_310320

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 6]

theorem matrix_multiplication :
  A * B = !![17, -3; 16, -24] := by sorry

end matrix_multiplication_l3103_310320


namespace trigonometric_identity_l3103_310363

theorem trigonometric_identity : 
  3.4173 * Real.sin (2 * Real.pi / 17) + Real.sin (4 * Real.pi / 17) - 
  Real.sin (6 * Real.pi / 17) - 0.5 * Real.sin (8 * Real.pi / 17) = 
  8 * Real.sin (2 * Real.pi / 17) ^ 3 * Real.cos (Real.pi / 17) ^ 2 := by
sorry

end trigonometric_identity_l3103_310363
