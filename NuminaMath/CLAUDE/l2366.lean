import Mathlib

namespace savings_comparison_l2366_236646

theorem savings_comparison (S : ℝ) (h1 : S > 0) : 
  let last_year_savings := 0.06 * S
  let this_year_salary := 1.1 * S
  let this_year_savings := 0.09 * this_year_salary
  (this_year_savings / last_year_savings) * 100 = 165 := by
sorry

end savings_comparison_l2366_236646


namespace triangle_side_range_l2366_236607

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_side_range (t : Triangle) 
  (h1 : t.a^2 + t.c^2 - t.b^2 = t.a * t.c)
  (h2 : t.b = Real.sqrt 3) :
  Real.sqrt 3 < 2 * t.a + t.c ∧ 2 * t.a + t.c ≤ 2 * Real.sqrt 7 := by
  sorry


end triangle_side_range_l2366_236607


namespace jennifers_spending_l2366_236669

theorem jennifers_spending (total : ℚ) (sandwich_frac : ℚ) (museum_frac : ℚ) (leftover : ℚ)
  (h1 : total = 150)
  (h2 : sandwich_frac = 1/5)
  (h3 : museum_frac = 1/6)
  (h4 : leftover = 20) :
  let spent_on_sandwich := total * sandwich_frac
  let spent_on_museum := total * museum_frac
  let total_spent := total - leftover
  let spent_on_book := total_spent - spent_on_sandwich - spent_on_museum
  spent_on_book / total = 1/2 := by
sorry

end jennifers_spending_l2366_236669


namespace expanded_identity_properties_l2366_236611

theorem expanded_identity_properties (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 1)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1) ∧
  (a₀ - a₁ + a₂ - a₃ + a₄ - a₅ = -243) ∧
  (a₀ + a₂ + a₄ = -121) := by
  sorry

end expanded_identity_properties_l2366_236611


namespace ball_max_height_l2366_236603

-- Define the height function
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 55

-- State the theorem
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 135 :=
sorry

end ball_max_height_l2366_236603


namespace quadratic_roots_sum_and_product_l2366_236634

theorem quadratic_roots_sum_and_product :
  let a : ℝ := 9
  let b : ℝ := -45
  let c : ℝ := 50
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  sum_of_roots = 5 ∧ product_of_roots = 50 / 9 := by
sorry

end quadratic_roots_sum_and_product_l2366_236634


namespace line_through_two_points_line_with_special_intercepts_l2366_236688

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def pointOnLine (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Part 1
theorem line_through_two_points :
  ∃ (l : Line), pointOnLine l ⟨4, 1⟩ ∧ pointOnLine l ⟨-1, 6⟩ →
  l.slope * 1 + l.intercept = 5 :=
sorry

-- Part 2
theorem line_with_special_intercepts :
  ∃ (l : Line), pointOnLine l ⟨4, 1⟩ ∧ 
  (l.intercept = 2 * (- l.intercept / l.slope)) →
  (l.slope = 1/4 ∧ l.intercept = 0) ∨ (l.slope = -2 ∧ l.intercept = 9) :=
sorry

end line_through_two_points_line_with_special_intercepts_l2366_236688


namespace A_gives_B_150m_start_l2366_236674

-- Define the speeds of runners A, B, and C
variable (Va Vb Vc : ℝ)

-- Define the conditions
def A_gives_C_300m_start : Prop := Va / Vc = 1000 / 700
def B_gives_C_176_47m_start : Prop := Vb / Vc = 1000 / 823.53

-- Define the theorem
theorem A_gives_B_150m_start 
  (h1 : A_gives_C_300m_start Va Vc) 
  (h2 : B_gives_C_176_47m_start Vb Vc) : 
  Va / Vb = 1000 / 850 := by sorry

end A_gives_B_150m_start_l2366_236674


namespace power_sum_equality_l2366_236620

theorem power_sum_equality : (-2)^23 + 2^(2^4 + 5^2 - 7^2) = -8388607.99609375 := by
  sorry

end power_sum_equality_l2366_236620


namespace multiple_of_a_l2366_236654

theorem multiple_of_a (a : ℤ) : 
  (∃ k : ℤ, 97 * a^2 + 84 * a - 55 = k * a) ↔ 
  (a = 1 ∨ a = 5 ∨ a = 11 ∨ a = 55 ∨ a = -1 ∨ a = -5 ∨ a = -11 ∨ a = -55) :=
by sorry

end multiple_of_a_l2366_236654


namespace linear_function_properties_l2366_236633

def f (x : ℝ) := -2 * x + 4

theorem linear_function_properties :
  (∀ x y, x < y → f x > f y) ∧
  (∀ x y, x < 0 ∧ y < 0 → ¬(f x < 0 ∧ f y < 0)) ∧
  (f 0 ≠ 0 ∨ 4 ≠ 0) ∧
  (∀ x, f x - 4 = -2 * x) :=
by sorry

end linear_function_properties_l2366_236633


namespace smallest_balanced_number_l2366_236653

/-- A function that returns true if a number is a three-digit number with distinct non-zero digits -/
def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  (n / 100 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10) ∧
  (n / 100 ≠ 0) ∧
  ((n / 10) % 10 ≠ 0) ∧
  (n % 10 ≠ 0)

/-- A function that calculates the sum of all two-digit numbers formed from the digits of a three-digit number -/
def sum_of_two_digit_numbers (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (10 * a + b) + (10 * b + a) + (10 * a + c) + (10 * c + a) + (10 * b + c) + (10 * c + b)

/-- The main theorem stating that 132 is the smallest balanced number -/
theorem smallest_balanced_number :
  is_valid_number 132 ∧
  132 = sum_of_two_digit_numbers 132 ∧
  ∀ n < 132, is_valid_number n → n ≠ sum_of_two_digit_numbers n :=
sorry

end smallest_balanced_number_l2366_236653


namespace smallest_x_multiple_of_53_l2366_236684

theorem smallest_x_multiple_of_53 : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(53 ∣ (3*y)^2 + 3*43*3*y + 43^2)) ∧
  (53 ∣ (3*x)^2 + 3*43*3*x + 43^2) ∧
  x = 21 := by
sorry

end smallest_x_multiple_of_53_l2366_236684


namespace triangle_perimeter_l2366_236665

/-- Given a triangle with inradius 5.0 cm and area 105 cm², its perimeter is 42 cm. -/
theorem triangle_perimeter (inradius : ℝ) (area : ℝ) (perimeter : ℝ) : 
  inradius = 5.0 → area = 105 → area = inradius * (perimeter / 2) → perimeter = 42 := by
  sorry

end triangle_perimeter_l2366_236665


namespace sandwich_cost_l2366_236630

/-- The cost of a sandwich given the total cost of sandwiches and sodas -/
theorem sandwich_cost (total_cost : ℚ) (num_sandwiches : ℕ) (num_sodas : ℕ) (soda_cost : ℚ) : 
  total_cost = 8.36 ∧ 
  num_sandwiches = 2 ∧ 
  num_sodas = 4 ∧ 
  soda_cost = 0.87 → 
  (total_cost - num_sodas * soda_cost) / num_sandwiches = 2.44 := by
sorry

end sandwich_cost_l2366_236630


namespace carol_trivia_score_l2366_236645

/-- Carol's trivia game score calculation -/
theorem carol_trivia_score (first_round : ℤ) (second_round : ℤ) (last_round : ℤ) (total : ℤ) : 
  second_round = 6 →
  last_round = -16 →
  total = 7 →
  first_round + second_round + last_round = total →
  first_round = 17 := by
sorry

end carol_trivia_score_l2366_236645


namespace inequality_proof_l2366_236685

theorem inequality_proof (x : ℝ) (h1 : (3/2 : ℝ) ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end inequality_proof_l2366_236685


namespace pitcher_problem_l2366_236671

theorem pitcher_problem (C : ℝ) (h : C > 0) :
  let juice_volume := C / 2
  let num_cups := 8
  let cup_volume := juice_volume / num_cups
  (cup_volume / C) * 100 = 6.25 := by sorry

end pitcher_problem_l2366_236671


namespace max_equidistant_circles_l2366_236647

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Four points in a 2D plane -/
def FourPoints := Fin 4 → Point

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- Predicate to check if four points lie on the same circle -/
def on_same_circle (points : FourPoints) : Prop := sorry

/-- Predicate to check if a circle is equidistant from all four points -/
def equidistant_circle (c : Circle) (points : FourPoints) : Prop := sorry

/-- The main theorem -/
theorem max_equidistant_circles (points : FourPoints) 
  (h1 : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k))
  (h2 : ¬on_same_circle points) :
  (∃ (circles : Finset Circle), 
    (∀ c ∈ circles, equidistant_circle c points) ∧ 
    circles.card = 7 ∧
    (∀ circles' : Finset Circle, 
      (∀ c ∈ circles', equidistant_circle c points) → 
      circles'.card ≤ 7)) := by sorry

end max_equidistant_circles_l2366_236647


namespace sin_2beta_value_l2366_236636

theorem sin_2beta_value (α β : ℝ) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : Real.cos (2 * α + β) - 2 * Real.cos (α + β) * Real.cos α = 3/5) :
  Real.sin (2 * β) = -24/25 := by
  sorry

end sin_2beta_value_l2366_236636


namespace calculation_proof_l2366_236642

theorem calculation_proof : 
  (Real.sqrt 18) / 3 + |Real.sqrt 2 - 2| + 2023^0 - (-1)^1 = 2 := by
  sorry

end calculation_proof_l2366_236642


namespace soda_box_cans_l2366_236693

/-- The number of people attending the reunion -/
def attendees : ℕ := 5 * 12

/-- The number of cans each person consumes -/
def cans_per_person : ℕ := 2

/-- The cost of each box of soda in dollars -/
def cost_per_box : ℕ := 2

/-- The number of family members -/
def family_members : ℕ := 6

/-- The amount each family member pays in dollars -/
def payment_per_member : ℕ := 4

/-- The number of cans in each box -/
def cans_per_box : ℕ := 10

theorem soda_box_cans : 
  cans_per_box = (attendees * cans_per_person) / 
    ((family_members * payment_per_member) / cost_per_box) :=
by sorry

end soda_box_cans_l2366_236693


namespace problem_solution_l2366_236614

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/6 = 3*y) : x = 108 := by
  sorry

end problem_solution_l2366_236614


namespace range_of_f_l2366_236627

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x - 4)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range (fun (x : ℝ) => f x) = {y : ℝ | y < -27 ∨ y > -27} :=
by sorry

end range_of_f_l2366_236627


namespace largest_n_value_l2366_236663

/-- Represents a number in a given base -/
structure BaseRepresentation (base : ℕ) where
  digits : List ℕ
  valid : ∀ d ∈ digits, d < base

/-- The value of n in base 10 given its representation in another base -/
def toBase10 (base : ℕ) (repr : BaseRepresentation base) : ℕ :=
  repr.digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

theorem largest_n_value (n : ℕ) 
  (base8_repr : BaseRepresentation 8)
  (base12_repr : BaseRepresentation 12)
  (h1 : toBase10 8 base8_repr = n)
  (h2 : toBase10 12 base12_repr = n)
  (h3 : base8_repr.digits.length = 3)
  (h4 : base12_repr.digits.length = 3)
  (h5 : base8_repr.digits.reverse = base12_repr.digits) :
  n ≤ 509 := by
  sorry

#check largest_n_value

end largest_n_value_l2366_236663


namespace route_b_saves_six_hours_l2366_236609

-- Define the time for each route (one way)
def route_a_time : ℕ := 5
def route_b_time : ℕ := 2

-- Define the function to calculate round trip time
def round_trip_time (one_way_time : ℕ) : ℕ := 2 * one_way_time

-- Define the function to calculate time saved
def time_saved (longer_route : ℕ) (shorter_route : ℕ) : ℕ :=
  round_trip_time longer_route - round_trip_time shorter_route

-- Theorem statement
theorem route_b_saves_six_hours :
  time_saved route_a_time route_b_time = 6 := by
  sorry

end route_b_saves_six_hours_l2366_236609


namespace trivia_team_score_l2366_236687

theorem trivia_team_score (total_members : ℕ) (absent_members : ℕ) (total_score : ℕ) :
  total_members = 5 →
  absent_members = 2 →
  total_score = 18 →
  ∃ (points_per_member : ℕ),
    points_per_member * (total_members - absent_members) = total_score ∧
    points_per_member = 6 := by
  sorry

end trivia_team_score_l2366_236687


namespace solve_exponential_equation_l2366_236624

theorem solve_exponential_equation (x : ℝ) : 
  (12 : ℝ)^x * 6^4 / 432 = 432 → x = 2 := by
  sorry

end solve_exponential_equation_l2366_236624


namespace a_positive_sufficient_not_necessary_for_abs_a_positive_l2366_236658

theorem a_positive_sufficient_not_necessary_for_abs_a_positive :
  (∃ a : ℝ, a > 0 → abs a > 0) ∧ 
  (∃ a : ℝ, abs a > 0 ∧ ¬(a > 0)) :=
by sorry

end a_positive_sufficient_not_necessary_for_abs_a_positive_l2366_236658


namespace min_horses_oxen_solution_l2366_236604

theorem min_horses_oxen_solution :
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧
    344 * x - 265 * y = 33 ∧
    ∀ (x' y' : ℕ), x' > 0 → y' > 0 → 344 * x' - 265 * y' = 33 → x' ≥ x ∧ y' ≥ y :=
by
  -- The proof would go here
  sorry

#check min_horses_oxen_solution

end min_horses_oxen_solution_l2366_236604


namespace arithmetic_sequence_sum_l2366_236696

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 2 + a 3 + a 4 + a 5 + a 6 = 90) →
  (a 1 + a 7 = 36) := by
sorry

end arithmetic_sequence_sum_l2366_236696


namespace h_sqrt_two_equals_zero_min_a_plus_b_h_not_arbitrary_quadratic_l2366_236656

-- Define the functions f, g, l, and h
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x
def g (b : ℝ) (x : ℝ) : ℝ := x + b
def l (x : ℝ) : ℝ := 2*x^2 + 3*x - 1

-- Define the property of h being generated by f and g
def is_generated_by_f_and_g (h : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ (m n : ℝ), ∀ x, h x = m * (f a x) + n * (g b x)

-- Define h as a quadratic function
def h (a b m n : ℝ) (x : ℝ) : ℝ := m * (f a x) + n * (g b x)

-- Theorem 1
theorem h_sqrt_two_equals_zero (a b m n : ℝ) :
  a = 1 → b = 2 → (∀ x, h a b m n x = h a b m n (-x)) → 
  h a b m n (Real.sqrt 2) = 0 := by sorry

-- Theorem 2
theorem min_a_plus_b (a b m n : ℝ) :
  b > 0 → is_generated_by_f_and_g (h a b m n) a b →
  (∃ m' n', ∀ x, h a b m n x = m' * g b x + n' * l x) →
  a + b ≥ 3/2 + Real.sqrt 2 := by sorry

-- Theorem 3
theorem h_not_arbitrary_quadratic (a b : ℝ) :
  ¬ ∀ (p q r : ℝ), ∃ (m n : ℝ), ∀ x, h a b m n x = p*x^2 + q*x + r := by sorry

end h_sqrt_two_equals_zero_min_a_plus_b_h_not_arbitrary_quadratic_l2366_236656


namespace triangle_perimeter_impossibility_l2366_236638

theorem triangle_perimeter_impossibility (a b x : ℝ) (h1 : a = 12) (h2 : b = 30) : 
  (a + b + x = 58 ∨ a + b + x = 85) → ¬(a + b > x ∧ a + x > b ∧ b + x > a) :=
sorry

end triangle_perimeter_impossibility_l2366_236638


namespace inequality_equivalence_l2366_236617

theorem inequality_equivalence (x : ℝ) : 
  |((x^2 + 2*x - 3) / 4)| ≤ 3 ↔ -5 ≤ x ∧ x ≤ 3 := by sorry

end inequality_equivalence_l2366_236617


namespace total_sundaes_l2366_236641

def num_flavors : ℕ := 8

def sundae_combinations (n : ℕ) : ℕ := Nat.choose num_flavors n

theorem total_sundaes : 
  sundae_combinations 1 + sundae_combinations 2 + sundae_combinations 3 = 92 := by
  sorry

end total_sundaes_l2366_236641


namespace complex_number_in_first_quadrant_l2366_236613

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 * I - 1) / I
  z.re > 0 ∧ z.im > 0 :=
by sorry

end complex_number_in_first_quadrant_l2366_236613


namespace next_draw_highest_probability_l2366_236650

/-- The probability of drawing a specific number in a lottery draw -/
def draw_probability : ℚ := 5 / 90

/-- The probability of not drawing a specific number in a lottery draw -/
def not_draw_probability : ℚ := 1 - draw_probability

/-- The probability of drawing a specific number in the n-th future draw -/
def future_draw_probability (n : ℕ) : ℚ :=
  (not_draw_probability ^ (n - 1)) * draw_probability

theorem next_draw_highest_probability :
  ∀ n : ℕ, n > 1 → draw_probability > future_draw_probability n :=
sorry

end next_draw_highest_probability_l2366_236650


namespace factorization_equality_l2366_236692

theorem factorization_equality (x y : ℝ) : 
  x^2 * (y^2 - 1) + 2 * x * (y^2 - 1) + (y^2 - 1) = (y + 1) * (y - 1) * (x + 1)^2 := by
  sorry

end factorization_equality_l2366_236692


namespace percentage_decrease_proof_l2366_236678

def original_price : ℝ := 250
def new_price : ℝ := 200

theorem percentage_decrease_proof :
  (original_price - new_price) / original_price * 100 = 20 := by
  sorry

end percentage_decrease_proof_l2366_236678


namespace defective_units_shipped_percentage_l2366_236657

/-- The percentage of units with Type A defects in the first stage -/
def type_a_defect_rate : ℝ := 0.07

/-- The percentage of units with Type B defects in the second stage -/
def type_b_defect_rate : ℝ := 0.08

/-- The percentage of Type A defects that are reworked and repaired -/
def type_a_repair_rate : ℝ := 0.4

/-- The percentage of Type B defects that are reworked and repaired -/
def type_b_repair_rate : ℝ := 0.3

/-- The percentage of remaining Type A defects that are shipped for sale -/
def type_a_ship_rate : ℝ := 0.03

/-- The percentage of remaining Type B defects that are shipped for sale -/
def type_b_ship_rate : ℝ := 0.06

/-- The theorem stating the percentage of units produced that are defective (Type A or B) and shipped for sale -/
theorem defective_units_shipped_percentage :
  (type_a_defect_rate * (1 - type_a_repair_rate) * type_a_ship_rate +
   type_b_defect_rate * (1 - type_b_repair_rate) * type_b_ship_rate) * 100 =
  0.462 := by sorry

end defective_units_shipped_percentage_l2366_236657


namespace modified_triangular_array_100th_row_sum_l2366_236619

/-- Sum of numbers in the nth row of the modified triangular array -/
def row_sum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * row_sum (n - 1) + 2

theorem modified_triangular_array_100th_row_sum :
  row_sum 100 = 2^100 - 2 :=
sorry

end modified_triangular_array_100th_row_sum_l2366_236619


namespace sum_after_2023_operations_l2366_236697

def starting_sequence : List Int := [7, 3, 5]

def operation (seq : List Int) : List Int :=
  seq ++ (List.zip seq (List.tail seq)).map (fun (a, b) => a - b)

def sum_after_n_operations (n : Nat) : Int :=
  n * 2 + (starting_sequence.sum)

theorem sum_after_2023_operations :
  sum_after_n_operations 2023 = 4061 := by sorry

end sum_after_2023_operations_l2366_236697


namespace multiplication_problem_l2366_236644

theorem multiplication_problem (A B C D : Nat) : 
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  A < 10 → B < 10 → C < 10 → D < 10 →
  C * 10 + D = 25 →
  (A * 100 + B * 10 + A) * (C * 10 + D) = C * 1000 + D * 100 + C * 10 + 0 →
  A + B = 2 := by
sorry

end multiplication_problem_l2366_236644


namespace purely_imaginary_complex_number_l2366_236608

theorem purely_imaginary_complex_number (a : ℝ) : 
  (((a^2 - 3*a + 2) : ℂ) + (a - 1)*I).re = 0 ∧ (((a^2 - 3*a + 2) : ℂ) + (a - 1)*I).im ≠ 0 → a = 2 := by
  sorry

end purely_imaginary_complex_number_l2366_236608


namespace min_cans_correct_l2366_236615

/-- The number of ounces in one can of soda -/
def ounces_per_can : ℕ := 12

/-- The number of ounces in a gallon -/
def ounces_per_gallon : ℕ := 128

/-- The minimum number of cans needed to provide at least a gallon of soda -/
def min_cans : ℕ := 11

/-- Theorem stating that min_cans is the minimum number of cans needed to provide at least a gallon of soda -/
theorem min_cans_correct : 
  (∀ n : ℕ, n * ounces_per_can ≥ ounces_per_gallon → n ≥ min_cans) ∧ 
  (min_cans * ounces_per_can ≥ ounces_per_gallon) :=
sorry

end min_cans_correct_l2366_236615


namespace strictly_decreasing_quadratic_function_l2366_236632

/-- A function f(x) = kx² - 4x - 8 is strictly decreasing on [4, 16] iff k ∈ (-∞, 1/8] -/
theorem strictly_decreasing_quadratic_function (k : ℝ) :
  (∀ x₁ x₂ : ℝ, 4 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 16 →
    k * x₂^2 - 4*x₂ - 8 < k * x₁^2 - 4*x₁ - 8) ↔
  k ≤ 1/8 :=
sorry

end strictly_decreasing_quadratic_function_l2366_236632


namespace half_radius_of_equal_area_circle_l2366_236612

/-- Given two circles with the same area, where one has a circumference of 12π,
    half of the radius of the other circle is 3. -/
theorem half_radius_of_equal_area_circle (x y : ℝ) :
  (π * x^2 = π * y^2) →  -- Circles x and y have the same area
  (2 * π * x = 12 * π) →  -- Circle x has a circumference of 12π
  y / 2 = 3 := by  -- Half of the radius of circle y is 3
  sorry

end half_radius_of_equal_area_circle_l2366_236612


namespace set_formation_criterion_l2366_236626

-- Define a type for objects
variable {α : Type}

-- Define a predicate for definiteness and distinctness
def is_definite_and_distinct (S : Set α) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x = y ∨ x ≠ y)

-- Define a predicate for forming a set
def can_form_set (S : Set α) : Prop :=
  is_definite_and_distinct S

-- Theorem statement
theorem set_formation_criterion (S : Set α) :
  can_form_set S ↔ is_definite_and_distinct S :=
by
  sorry


end set_formation_criterion_l2366_236626


namespace suhwan_milk_consumption_l2366_236635

/-- Amount of milk Suhwan drinks per time in liters -/
def milk_per_time : ℝ := 0.2

/-- Number of times Suhwan drinks milk per day -/
def times_per_day : ℕ := 3

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Suhwan's weekly milk consumption in liters -/
def weekly_milk_consumption : ℝ :=
  milk_per_time * (times_per_day : ℝ) * (days_in_week : ℝ)

theorem suhwan_milk_consumption :
  weekly_milk_consumption = 4.2 := by
  sorry

end suhwan_milk_consumption_l2366_236635


namespace prove_first_ingot_weight_l2366_236640

theorem prove_first_ingot_weight (weights : Fin 11 → ℕ) 
  (h_distinct : Function.Injective weights)
  (h_range : ∀ i, weights i ∈ Finset.range 12 \ {0}) : 
  ∃ (a b c d e f : Fin 11), 
    weights a + weights b + weights c + weights d ≤ 11 ∧
    weights a + weights e + weights f ≤ 11 ∧
    weights a = 1 :=
sorry

end prove_first_ingot_weight_l2366_236640


namespace quadratic_inequality_range_l2366_236672

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 6 * k * x + k + 8 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) := by
sorry

end quadratic_inequality_range_l2366_236672


namespace gcd_765432_654321_l2366_236668

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := by
  sorry

end gcd_765432_654321_l2366_236668


namespace hyperbola_equation_l2366_236664

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  h_equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
  h_asymptote : ∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt 3 * x
  h_focus_on_directrix : ∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ x = -6

/-- The theorem stating the specific equation of the hyperbola -/
theorem hyperbola_equation (h : Hyperbola) : 
  h.a^2 = 9 ∧ h.b^2 = 27 :=
sorry

end hyperbola_equation_l2366_236664


namespace cost_of_six_lollipops_l2366_236670

/-- The cost of 6 giant lollipops with discounts and promotions -/
theorem cost_of_six_lollipops (regular_price : ℝ) (discount_rate : ℝ) : 
  regular_price = 2.4 / 2 →
  discount_rate = 0.1 →
  6 * regular_price * (1 - discount_rate) = 6.48 := by
  sorry

end cost_of_six_lollipops_l2366_236670


namespace unique_fraction_exists_l2366_236659

def is_relatively_prime (x y : ℕ+) : Prop := Nat.gcd x.val y.val = 1

theorem unique_fraction_exists : ∃! (x y : ℕ+), 
  is_relatively_prime x y ∧ 
  (x.val + 1 : ℚ) / (y.val + 1) = 1.2 * (x.val : ℚ) / y.val := by
  sorry

end unique_fraction_exists_l2366_236659


namespace ruth_gave_two_sandwiches_to_brother_l2366_236691

/-- The number of sandwiches Ruth prepared -/
def total_sandwiches : ℕ := 10

/-- The number of sandwiches Ruth ate -/
def ruth_ate : ℕ := 1

/-- The number of sandwiches the first cousin ate -/
def first_cousin_ate : ℕ := 2

/-- The number of other cousins -/
def other_cousins : ℕ := 2

/-- The number of sandwiches each other cousin ate -/
def each_other_cousin_ate : ℕ := 1

/-- The number of sandwiches left -/
def sandwiches_left : ℕ := 3

/-- The number of sandwiches Ruth gave to her brother -/
def sandwiches_to_brother : ℕ := total_sandwiches - (ruth_ate + first_cousin_ate + other_cousins * each_other_cousin_ate + sandwiches_left)

theorem ruth_gave_two_sandwiches_to_brother : sandwiches_to_brother = 2 := by
  sorry

end ruth_gave_two_sandwiches_to_brother_l2366_236691


namespace sequence_formula_l2366_236695

/-- Given a sequence {a_n} where the sum of the first n terms S_n satisfies S_n = 2a_n + 1,
    prove that the general formula for a_n is -2^(n-1) -/
theorem sequence_formula (a : ℕ → ℤ) (S : ℕ → ℤ) 
    (h : ∀ n, S n = 2 * a n + 1) :
  ∀ n, a n = -2^(n-1) := by
sorry

end sequence_formula_l2366_236695


namespace olympic_audience_conversion_l2366_236698

def opening_ceremony_audience : ℕ := 316000000
def closing_ceremony_audience : ℕ := 236000000

def million_to_full_number (x : ℕ) : ℕ := x * 1000000
def million_to_billion (x : ℕ) : ℚ := x / 1000

/-- Rounds a rational number to one decimal place -/
def round_to_one_decimal (x : ℚ) : ℚ :=
  (x * 10).floor / 10

theorem olympic_audience_conversion :
  (million_to_full_number 316 = opening_ceremony_audience) ∧
  (round_to_one_decimal (million_to_billion closing_ceremony_audience) = 2.4) :=
sorry

end olympic_audience_conversion_l2366_236698


namespace log_ratio_squared_l2366_236661

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1)
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log x)
  (h2 : x * y = 27) :
  ((Real.log x - Real.log y) / Real.log 3) ^ 2 = 9 := by
sorry

end log_ratio_squared_l2366_236661


namespace work_completion_proof_l2366_236649

/-- The number of days it takes W women to complete the work -/
def women_days : ℕ := 8

/-- The number of days it takes W children to complete the work -/
def children_days : ℕ := 12

/-- The number of days it takes 6 women and 3 children to complete the work -/
def combined_days : ℕ := 10

/-- The number of women in the combined group -/
def combined_women : ℕ := 6

/-- The number of children in the combined group -/
def combined_children : ℕ := 3

/-- The initial number of women working on the task -/
def initial_women : ℕ := 10

theorem work_completion_proof :
  (combined_women : ℚ) / (women_days * initial_women) +
  (combined_children : ℚ) / (children_days * initial_women) =
  1 / combined_days :=
sorry

end work_completion_proof_l2366_236649


namespace expression_evaluation_l2366_236686

theorem expression_evaluation : 
  Real.sqrt 2 * (2 ^ (3/2)) + 15 / 5 * 3 - Real.sqrt 9 = 10 := by
  sorry

end expression_evaluation_l2366_236686


namespace book_sale_price_l2366_236690

def book_sale (total_books : ℕ) (unsold_books : ℕ) (total_amount : ℚ) : Prop :=
  let sold_books := total_books - unsold_books
  let price_per_book := total_amount / sold_books
  (2 : ℚ) / 3 * total_books = sold_books ∧
  unsold_books = 36 ∧
  total_amount = 252 ∧
  price_per_book = (7 : ℚ) / 2

theorem book_sale_price :
  ∃ (total_books : ℕ) (unsold_books : ℕ) (total_amount : ℚ),
    book_sale total_books unsold_books total_amount :=
by
  sorry

end book_sale_price_l2366_236690


namespace age_range_count_l2366_236682

/-- Calculates the number of integer ages within one standard deviation of the average age -/
def count_ages_within_std_dev (average_age : ℕ) (std_dev : ℕ) : ℕ :=
  (average_age + std_dev) - (average_age - std_dev) + 1

/-- Proves that given an average age of 31 and a standard deviation of 9, 
    the number of integer ages within one standard deviation of the average is 19 -/
theorem age_range_count : count_ages_within_std_dev 31 9 = 19 := by
  sorry

end age_range_count_l2366_236682


namespace solution_satisfies_system_l2366_236675

-- Define the system of equations
def equation1 (x y z : ℚ) : Prop :=
  6 / (3 * x + 4 * y) + 4 / (5 * x - 4 * z) = 7 / 12

def equation2 (x y z : ℚ) : Prop :=
  9 / (4 * y + 3 * z) - 4 / (3 * x + 4 * y) = 1 / 3

def equation3 (x y z : ℚ) : Prop :=
  2 / (5 * x - 4 * z) + 6 / (4 * y + 3 * z) = 1 / 2

-- Theorem statement
theorem solution_satisfies_system :
  equation1 4 3 2 ∧ equation2 4 3 2 ∧ equation3 4 3 2 := by
  sorry

end solution_satisfies_system_l2366_236675


namespace mark_current_age_l2366_236631

/-- Mark's current age -/
def mark_age : ℕ := 28

/-- Aaron's current age -/
def aaron_age : ℕ := 11

/-- Theorem stating that Mark's current age is 28, given the conditions about their ages -/
theorem mark_current_age :
  (mark_age - 3 = 3 * (aaron_age - 3) + 1) ∧
  (mark_age + 4 = 2 * (aaron_age + 4) + 2) →
  mark_age = 28 := by
  sorry


end mark_current_age_l2366_236631


namespace marcus_baseball_cards_l2366_236677

/-- The number of baseball cards Carter has -/
def carterCards : ℕ := 152

/-- The number of additional cards Marcus has compared to Carter -/
def marcusExtraCards : ℕ := 58

/-- The number of baseball cards Marcus has -/
def marcusCards : ℕ := carterCards + marcusExtraCards

theorem marcus_baseball_cards : marcusCards = 210 := by
  sorry

end marcus_baseball_cards_l2366_236677


namespace probability_exact_hits_l2366_236683

def probability_single_hit : ℝ := 0.7
def total_shots : ℕ := 5
def desired_hits : ℕ := 2

theorem probability_exact_hits :
  let p := probability_single_hit
  let n := total_shots
  let k := desired_hits
  let q := 1 - p
  (Nat.choose n k : ℝ) * p ^ k * q ^ (n - k) = 0.1323 := by sorry

end probability_exact_hits_l2366_236683


namespace distribute_six_among_four_l2366_236655

/-- The number of ways to distribute n indistinguishable objects among k distinct containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 84 ways to distribute 6 objects among 4 containers -/
theorem distribute_six_among_four : distribute 6 4 = 84 := by sorry

end distribute_six_among_four_l2366_236655


namespace not_all_tangents_equal_l2366_236662

/-- A convex quadrilateral where the tangent of one angle is m -/
structure ConvexQuadrilateral (m : ℝ) where
  angles : Fin 4 → ℝ
  sum_360 : angles 0 + angles 1 + angles 2 + angles 3 = 360
  all_positive : ∀ i, 0 < angles i
  all_less_180 : ∀ i, angles i < 180
  one_tangent_m : ∃ i, Real.tan (angles i) = m

/-- Theorem stating that it's impossible for all angles to have tangent m -/
theorem not_all_tangents_equal (m : ℝ) (q : ConvexQuadrilateral m) :
  ¬(∀ i, Real.tan (q.angles i) = m) :=
sorry

end not_all_tangents_equal_l2366_236662


namespace keychain_arrangement_theorem_l2366_236679

def number_of_arrangements (n : ℕ) : ℕ := n.factorial

def number_of_adjacent_arrangements (n : ℕ) : ℕ := 2 * (n - 1).factorial

theorem keychain_arrangement_theorem :
  let total_arrangements := number_of_arrangements 5
  let adjacent_arrangements := number_of_adjacent_arrangements 5
  total_arrangements - adjacent_arrangements = 72 := by
  sorry

end keychain_arrangement_theorem_l2366_236679


namespace interleave_sequences_count_l2366_236673

def interleave_sequences (n₁ n₂ n₃ : ℕ) : ℕ :=
  Nat.factorial (n₁ + n₂ + n₃) / (Nat.factorial n₁ * Nat.factorial n₂ * Nat.factorial n₃)

theorem interleave_sequences_count (n₁ n₂ n₃ : ℕ) :
  interleave_sequences n₁ n₂ n₃ = 
    Nat.choose (n₁ + n₂ + n₃) n₁ * Nat.choose (n₂ + n₃) n₂ :=
by sorry

end interleave_sequences_count_l2366_236673


namespace division_result_l2366_236652

theorem division_result : (-0.91) / (-0.13) = 7 := by sorry

end division_result_l2366_236652


namespace area_ratio_squares_l2366_236681

/-- Given squares A, B, and C with specified properties, prove the ratio of areas of A to C -/
theorem area_ratio_squares (sideA sideB sideC : ℝ) : 
  sideA * 4 = 16 →  -- Perimeter of A is 16
  sideB * 4 = 40 →  -- Perimeter of B is 40
  sideC = 1.5 * sideA →  -- Side of C is 1.5 times side of A
  (sideA ^ 2) / (sideC ^ 2) = 4 / 9 := by
  sorry

end area_ratio_squares_l2366_236681


namespace inequality_implication_l2366_236606

theorem inequality_implication (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x - Real.log y > y - Real.log x → x - y > 1 / x - 1 / y := by
  sorry

end inequality_implication_l2366_236606


namespace negation_of_proposition_l2366_236676

theorem negation_of_proposition (a : ℝ) :
  (¬ ∀ x : ℝ, x ≥ 0 → x^2 - a*x + 3 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 - a*x + 3 ≤ 0) :=
by sorry

end negation_of_proposition_l2366_236676


namespace triangle_altitude_and_area_l2366_236605

/-- Triangle with sides a, b, c and altitude h from the vertex opposite side b --/
structure Triangle (a b c : ℝ) where
  h : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_altitude_and_area 
  (t : Triangle 11 13 16) : t.h = 168 / 13 ∧ (1 / 2 : ℝ) * 13 * t.h = 84 := by
  sorry

end triangle_altitude_and_area_l2366_236605


namespace highest_probability_l2366_236689

-- Define the sample space
variable (Ω : Type*)

-- Define the events A, B, and C
variable (A B C : Set Ω)

-- Define a probability measure
variable (P : Set Ω → ℝ)

-- State the theorem
theorem highest_probability 
  (h_subset1 : C ⊆ B) 
  (h_subset2 : B ⊆ A) 
  (h_prob : ∀ X : Set Ω, 0 ≤ P X ∧ P X ≤ 1) 
  (h_monotone : ∀ X Y : Set Ω, X ⊆ Y → P X ≤ P Y) : 
  P A ≥ P B ∧ P A ≥ P C :=
by sorry

end highest_probability_l2366_236689


namespace egg_price_calculation_l2366_236610

theorem egg_price_calculation (num_eggs : ℕ) (num_chickens : ℕ) (price_per_chicken : ℚ) (total_spent : ℚ) :
  num_eggs = 20 →
  num_chickens = 6 →
  price_per_chicken = 8 →
  total_spent = 88 →
  (total_spent - (num_chickens * price_per_chicken)) / num_eggs = 2 :=
by sorry

end egg_price_calculation_l2366_236610


namespace binomial_equation_solution_l2366_236621

theorem binomial_equation_solution (x : ℕ) : 
  (Nat.choose 10 (2*x) - Nat.choose 10 (x+1) = 0) → (x = 1 ∨ x = 3) :=
by sorry

end binomial_equation_solution_l2366_236621


namespace blue_candy_probability_l2366_236628

def green_candies : ℕ := 5
def blue_candies : ℕ := 3
def red_candies : ℕ := 4

def total_candies : ℕ := green_candies + blue_candies + red_candies

theorem blue_candy_probability :
  (blue_candies : ℚ) / total_candies = 1 / 4 := by sorry

end blue_candy_probability_l2366_236628


namespace arithmetic_arrangement_proof_l2366_236618

theorem arithmetic_arrangement_proof :
  (1 / 8 * 1 / 9 * 1 / 28 = 1 / 2016) ∧
  ((1 / 8 - 1 / 9) * 1 / 28 = 1 / 2016) := by
  sorry

end arithmetic_arrangement_proof_l2366_236618


namespace max_good_word_length_l2366_236616

/-- An alphabet is a finite set of letters. -/
def Alphabet (n : ℕ) := Fin n

/-- A word is a finite sequence of letters where consecutive letters are different. -/
def Word (α : Type) := List α

/-- A good word is one where it's impossible to delete all but four letters to obtain aabb. -/
def isGoodWord {α : Type} (w : Word α) : Prop :=
  ∀ (a b : α), a ≠ b → ¬∃ (i j k l : ℕ), i < j ∧ j < k ∧ k < l ∧
    w.get? i = some a ∧ w.get? j = some a ∧ w.get? k = some b ∧ w.get? l = some b

/-- The maximum length of a good word in an alphabet with n > 1 letters is 2n + 1. -/
theorem max_good_word_length {n : ℕ} (h : n > 1) :
  ∃ (w : Word (Alphabet n)), isGoodWord w ∧ w.length = 2 * n + 1 ∧
  ∀ (w' : Word (Alphabet n)), isGoodWord w' → w'.length ≤ 2 * n + 1 :=
sorry

end max_good_word_length_l2366_236616


namespace append_digit_twice_divisible_by_three_l2366_236625

theorem append_digit_twice_divisible_by_three (N d : ℕ) 
  (hN : N % 3 ≠ 0) (hd : d % 3 ≠ 0) (hd_last : d < 10) :
  ∃ k, N * 100 + d * 10 + d = 3 * k :=
sorry

end append_digit_twice_divisible_by_three_l2366_236625


namespace ticket_sales_problem_l2366_236623

/-- Proves that the total number of tickets sold is 42 given the conditions of the ticket sales problem. -/
theorem ticket_sales_problem (adult_price child_price total_sales child_tickets : ℕ)
  (h1 : adult_price = 5)
  (h2 : child_price = 3)
  (h3 : total_sales = 178)
  (h4 : child_tickets = 16) :
  ∃ (adult_tickets : ℕ), adult_price * adult_tickets + child_price * child_tickets = total_sales ∧
                          adult_tickets + child_tickets = 42 :=
by
  sorry

end ticket_sales_problem_l2366_236623


namespace blue_tissue_length_l2366_236600

theorem blue_tissue_length (red blue : ℝ) : 
  red = blue + 12 →
  2 * red = 3 * blue →
  blue = 24 := by
sorry

end blue_tissue_length_l2366_236600


namespace triangle_area_l2366_236660

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  A = π / 6 →  -- 30°
  C = π / 4 →  -- 45°
  a = 2 →
  B + C + A = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 + 1 :=
by sorry

end triangle_area_l2366_236660


namespace original_bill_amount_l2366_236667

theorem original_bill_amount (new_bill : ℝ) (increase_percent : ℝ) 
  (h1 : new_bill = 78)
  (h2 : increase_percent = 30) : 
  ∃ (original_bill : ℝ), 
    original_bill * (1 + increase_percent / 100) = new_bill ∧ 
    original_bill = 60 := by
  sorry

end original_bill_amount_l2366_236667


namespace cube_root_and_seventh_root_sum_l2366_236602

theorem cube_root_and_seventh_root_sum (m n : ℤ) 
  (hm : m ^ 3 = 61629875)
  (hn : n ^ 7 = 170859375) :
  100 * m + n = 39515 := by
sorry

end cube_root_and_seventh_root_sum_l2366_236602


namespace farmers_harvest_l2366_236637

/-- Farmer's harvest problem -/
theorem farmers_harvest
  (total_potatoes : ℕ)
  (potatoes_per_bundle : ℕ)
  (potato_bundle_price : ℚ)
  (total_carrots : ℕ)
  (carrot_bundle_price : ℚ)
  (total_revenue : ℚ)
  (h1 : total_potatoes = 250)
  (h2 : potatoes_per_bundle = 25)
  (h3 : potato_bundle_price = 190/100)
  (h4 : total_carrots = 320)
  (h5 : carrot_bundle_price = 2)
  (h6 : total_revenue = 51)
  : (total_carrots / ((total_revenue - (total_potatoes / potatoes_per_bundle * potato_bundle_price)) / carrot_bundle_price) : ℚ) = 20 := by
  sorry

end farmers_harvest_l2366_236637


namespace sum_of_coefficients_l2366_236666

-- Define the polynomial g(x)
def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem sum_of_coefficients (a b c d : ℝ) : 
  g a b c d (1 + I) = 0 → g a b c d (3*I) = 0 → a + b + c + d = 9 := by
  sorry

end sum_of_coefficients_l2366_236666


namespace complex_sum_of_powers_l2366_236699

theorem complex_sum_of_powers (x y : ℂ) (hxy : x ≠ 0 ∧ y ≠ 0) (h : x^2 + x*y + y^2 = 0) :
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 := by
  sorry

end complex_sum_of_powers_l2366_236699


namespace fundraiser_problem_l2366_236639

/-- The fundraiser problem -/
theorem fundraiser_problem
  (total_promised : ℕ)
  (sally_owes : ℕ)
  (carl_owes : ℕ)
  (amy_owes : ℕ)
  (derek_owes : ℕ)
  (h1 : total_promised = 400)
  (h2 : sally_owes = 35)
  (h3 : carl_owes = 35)
  (h4 : amy_owes = 30)
  (h5 : derek_owes = amy_owes / 2)
  : total_promised - (sally_owes + carl_owes + amy_owes + derek_owes) = 285 := by
  sorry

#check fundraiser_problem

end fundraiser_problem_l2366_236639


namespace polynomial_sum_l2366_236680

theorem polynomial_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end polynomial_sum_l2366_236680


namespace image_of_two_three_l2366_236622

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * p.2, p.1 + p.2)

-- State the theorem
theorem image_of_two_three :
  f (2, 3) = (6, 5) := by
  sorry

end image_of_two_three_l2366_236622


namespace A_final_value_l2366_236601

def update_A (initial_A : Int) : Int :=
  -initial_A + 5

theorem A_final_value (initial_A : Int) (h : initial_A = 15) :
  update_A initial_A = -10 := by
  sorry

end A_final_value_l2366_236601


namespace max_gcd_consecutive_b_l2366_236648

def b (n : ℕ) : ℕ := n.factorial + 2 * n

theorem max_gcd_consecutive_b : 
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (b n) (b (n + 1)) ≤ k ∧ 
  ∃ (m : ℕ), Nat.gcd (b m) (b (m + 1)) = k :=
by sorry

end max_gcd_consecutive_b_l2366_236648


namespace geometric_sequence_minimum_sum_l2366_236694

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_minimum_sum 
  (a : ℕ → ℝ) 
  (h_geom : GeometricSequence a) 
  (h_positive : ∀ n, a n > 0) 
  (h_product : a 3 * a 5 = 12) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x * y = 12 → x + y ≥ 4 * Real.sqrt 3) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = 12 ∧ x + y = 4 * Real.sqrt 3) :=
sorry

end geometric_sequence_minimum_sum_l2366_236694


namespace union_of_A_and_B_l2366_236651

-- Define sets A and B
def A : Set ℝ := {x | 2 * x - 1 > 0}
def B : Set ℝ := {x | |x| < 1}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > -1} := by sorry

end union_of_A_and_B_l2366_236651


namespace complete_square_equation_l2366_236643

theorem complete_square_equation (x : ℝ) : 
  (x^2 - 8*x + 15 = 0) ↔ ((x - 4)^2 = 1) := by
  sorry

end complete_square_equation_l2366_236643


namespace triangle_side_length_l2366_236629

theorem triangle_side_length (a b c : ℝ) (B : ℝ) : 
  a = 8 → b = 7 → B = Real.pi / 3 → (c = 3 ∨ c = 5) := by
  sorry

end triangle_side_length_l2366_236629
