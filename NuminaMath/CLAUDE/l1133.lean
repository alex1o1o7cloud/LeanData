import Mathlib

namespace min_sum_of_reciprocal_sum_eq_one_l1133_113394

theorem min_sum_of_reciprocal_sum_eq_one (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 1/b = 1) : a + b ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 1/b₀ = 1 ∧ a₀ + b₀ = 4 := by
  sorry

end min_sum_of_reciprocal_sum_eq_one_l1133_113394


namespace color_copies_comparison_l1133_113334

/-- The cost per color copy at print shop X -/
def cost_x : ℚ := 120 / 100

/-- The cost per color copy at print shop Y -/
def cost_y : ℚ := 170 / 100

/-- The difference in total cost between print shops Y and X -/
def cost_difference : ℚ := 35

/-- The number of color copies being compared -/
def n : ℚ := 70

theorem color_copies_comparison :
  cost_y * n = cost_x * n + cost_difference :=
by sorry

end color_copies_comparison_l1133_113334


namespace incenter_coordinates_specific_triangle_l1133_113342

/-- Given a triangle PQR with side lengths p, q, r, this function returns the coordinates of the incenter I -/
def incenter_coordinates (p q r : ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating that for a triangle with side lengths 8, 10, and 6, the incenter coordinates are (1/3, 5/12, 1/4) -/
theorem incenter_coordinates_specific_triangle :
  let (x, y, z) := incenter_coordinates 8 10 6
  x = 1/3 ∧ y = 5/12 ∧ z = 1/4 ∧ x + y + z = 1 := by sorry

end incenter_coordinates_specific_triangle_l1133_113342


namespace min_value_F_range_of_m_l1133_113356

noncomputable section

def f (x : ℝ) := x * Real.exp x
def g (x : ℝ) := (1/2) * x^2 + x
def F (x : ℝ) := f x + g x

-- Part 1
theorem min_value_F :
  ∃ (x_min : ℝ), ∀ (x : ℝ), F x_min ≤ F x ∧ F x_min = -1 - 1/Real.exp 1 :=
sorry

-- Part 2
theorem range_of_m (m : ℝ) :
  (∀ (x₁ x₂ : ℝ), -1 ≤ x₂ ∧ x₂ < x₁ →
    m * (f x₁ - f x₂) > g x₁ - g x₂) ↔ m ≥ Real.exp 1 :=
sorry

end

end min_value_F_range_of_m_l1133_113356


namespace book_division_l1133_113347

theorem book_division (total_books : ℕ) (first_division : ℕ) (second_division : ℕ) (books_per_category : ℕ) 
  (h1 : total_books = 1200)
  (h2 : first_division = 3)
  (h3 : second_division = 4)
  (h4 : books_per_category = 15) :
  (total_books / first_division / second_division / books_per_category) * 
  first_division * second_division = 84 :=
by sorry

end book_division_l1133_113347


namespace dragon_lion_equivalence_l1133_113322

theorem dragon_lion_equivalence (P Q : Prop) : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) := by sorry

end dragon_lion_equivalence_l1133_113322


namespace impossible_all_white_l1133_113344

-- Define the grid as a function from coordinates to colors
def Grid := Fin 8 → Fin 8 → Bool

-- Define the initial grid configuration
def initial_grid : Grid :=
  fun i j => (i = 0 ∧ j = 0) ∨ (i = 0 ∧ j = 7) ∨ (i = 7 ∧ j = 0) ∨ (i = 7 ∧ j = 7)

-- Define a row flip operation
def flip_row (g : Grid) (row : Fin 8) : Grid :=
  fun i j => if i = row then !g i j else g i j

-- Define a column flip operation
def flip_column (g : Grid) (col : Fin 8) : Grid :=
  fun i j => if j = col then !g i j else g i j

-- Define a predicate for an all-white grid
def all_white (g : Grid) : Prop :=
  ∀ i j, g i j = false

-- Theorem: It's impossible to achieve an all-white configuration
theorem impossible_all_white :
  ¬ ∃ (flips : List (Sum (Fin 8) (Fin 8))),
    all_white (flips.foldl (fun g flip => 
      match flip with
      | Sum.inl row => flip_row g row
      | Sum.inr col => flip_column g col
    ) initial_grid) :=
  sorry


end impossible_all_white_l1133_113344


namespace audit_options_l1133_113314

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem audit_options (initial_OR initial_GTU first_week_OR first_week_GTU : ℕ) 
  (h1 : initial_OR = 13)
  (h2 : initial_GTU = 15)
  (h3 : first_week_OR = 2)
  (h4 : first_week_GTU = 3) :
  (choose (initial_OR - first_week_OR) first_week_OR) * 
  (choose (initial_GTU - first_week_GTU) first_week_GTU) = 12100 := by
  sorry

end audit_options_l1133_113314


namespace quadratic_root_zero_l1133_113391

theorem quadratic_root_zero (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + x + m^2 - 1 = 0) ∧
  ((m - 1) * 0^2 + 0 + m^2 - 1 = 0) →
  m = -1 := by
  sorry

end quadratic_root_zero_l1133_113391


namespace triangle_inequality_l1133_113343

/-- Given an acute triangle ABC with circumradius 1, 
    prove that the sum of the ratios of each side to (1 - sine of its opposite angle) 
    is greater than or equal to 18 + 12√3 -/
theorem triangle_inequality (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  A < π/2 ∧ B < π/2 ∧ C < π/2 ∧
  a = 2 * Real.sin A ∧
  b = 2 * Real.sin B ∧
  c = 2 * Real.sin C →
  (a / (1 - Real.sin A)) + (b / (1 - Real.sin B)) + (c / (1 - Real.sin C)) ≥ 18 + 12 * Real.sqrt 3 := by
  sorry

end triangle_inequality_l1133_113343


namespace optimal_purchase_max_profit_l1133_113395

/-- Represents the types of multimedia --/
inductive MultimediaType
| A
| B

/-- Represents the cost and price of each type of multimedia --/
def cost_price (t : MultimediaType) : ℝ × ℝ :=
  match t with
  | MultimediaType.A => (3, 3.3)
  | MultimediaType.B => (2.4, 2.8)

/-- The total number of sets to purchase --/
def total_sets : ℕ := 50

/-- The total cost in million yuan --/
def total_cost : ℝ := 132

/-- Theorem for part 1 of the problem --/
theorem optimal_purchase :
  ∃ (a b : ℕ),
    a + b = total_sets ∧
    a * (cost_price MultimediaType.A).1 + b * (cost_price MultimediaType.B).1 = total_cost ∧
    a = 20 ∧ b = 30 := by sorry

/-- Function to calculate profit --/
def profit (a : ℕ) : ℝ :=
  let b := total_sets - a
  a * ((cost_price MultimediaType.A).2 - (cost_price MultimediaType.A).1) +
  b * ((cost_price MultimediaType.B).2 - (cost_price MultimediaType.B).1)

/-- Theorem for part 2 of the problem --/
theorem max_profit :
  ∃ (a : ℕ),
    10 < a ∧ a < 20 ∧
    (∀ m, 10 < m → m < 20 → profit m ≤ profit a) ∧
    a = 11 ∧ profit a = 18.9 := by sorry

end optimal_purchase_max_profit_l1133_113395


namespace inequality_proof_l1133_113377

theorem inequality_proof : -2 < (-1)^3 ∧ (-1)^3 < (-0.6)^2 := by
  sorry

end inequality_proof_l1133_113377


namespace q_zero_at_two_two_l1133_113324

def q (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℝ) (x y : ℝ) : ℝ :=
  b₀ + b₁*x + b₂*y + b₃*x^2 + b₄*x*y + b₅*y^2 + b₆*x^3 + b₇*y^3 + b₈*x^4 + b₉*y^4

theorem q_zero_at_two_two 
  (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℝ) 
  (h₀ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 0 = 0)
  (h₁ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 0 = 0)
  (h₂ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (-1) 0 = 0)
  (h₃ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 1 = 0)
  (h₄ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 (-1) = 0)
  (h₅ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 1 1 = 0)
  (h₆ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ (-1) (-1) = 0)
  (h₇ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 2 0 = 0)
  (h₈ : q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 0 2 = 0) :
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ 2 2 = 0 := by
sorry

end q_zero_at_two_two_l1133_113324


namespace last_two_nonzero_digits_of_80_factorial_l1133_113366

theorem last_two_nonzero_digits_of_80_factorial (n : ℕ) : n = 80 →
  ∃ k : ℕ, n.factorial = 100 * k + 48 ∧ k % 10 ≠ 0 :=
sorry

end last_two_nonzero_digits_of_80_factorial_l1133_113366


namespace eighth_number_is_four_l1133_113354

/-- A sequence of 12 numbers satisfying the given conditions -/
def SpecialSequence : Type := 
  {s : Fin 12 → ℕ // s 0 = 5 ∧ s 11 = 10 ∧ ∀ i, i < 10 → s i + s (i + 1) + s (i + 2) = 19}

/-- The theorem stating that the 8th number (index 7) in the sequence is 4 -/
theorem eighth_number_is_four (s : SpecialSequence) : s.val 7 = 4 := by
  sorry

end eighth_number_is_four_l1133_113354


namespace gcd_sum_and_sum_of_squares_l1133_113325

theorem gcd_sum_and_sum_of_squares (a b : ℤ) : 
  Int.gcd a b = 1 → Int.gcd (a + b) (a^2 + b^2) = 1 ∨ Int.gcd (a + b) (a^2 + b^2) = 2 := by
  sorry

end gcd_sum_and_sum_of_squares_l1133_113325


namespace inequality_proof_l1133_113352

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a + b + c = 1) : 
  (2*a*b + b*c + c*a + c^2/2 ≤ 1/2) ∧ 
  ((a^2 + c^2)/b + (b^2 + a^2)/c + (c^2 + b^2)/a ≥ 2) := by
  sorry

end inequality_proof_l1133_113352


namespace no_prime_solution_l1133_113372

def base_p_to_decimal (digits : List Nat) (p : Nat) : Nat :=
  digits.foldr (fun d acc => d + p * acc) 0

theorem no_prime_solution :
  ¬∃ p : Nat, Nat.Prime p ∧
    (base_p_to_decimal [4, 1, 0, 1] p +
     base_p_to_decimal [2, 0, 5] p +
     base_p_to_decimal [7, 1, 2] p +
     base_p_to_decimal [1, 3, 2] p +
     base_p_to_decimal [2, 1] p =
     base_p_to_decimal [4, 5, 2] p +
     base_p_to_decimal [7, 4, 5] p +
     base_p_to_decimal [5, 7, 6] p) :=
by sorry

end no_prime_solution_l1133_113372


namespace cloth_selling_amount_l1133_113374

/-- Calculates the total selling amount for cloth given the quantity, cost price, and loss per metre. -/
def totalSellingAmount (quantity : ℕ) (costPrice : ℕ) (lossPerMetre : ℕ) : ℕ :=
  quantity * (costPrice - lossPerMetre)

/-- Proves that the total selling amount for 200 metres of cloth with a cost price of 66 and a loss of 6 per metre is 12000. -/
theorem cloth_selling_amount :
  totalSellingAmount 200 66 6 = 12000 := by
  sorry

end cloth_selling_amount_l1133_113374


namespace g_evaluation_and_derivative_l1133_113341

def g (x : ℝ) : ℝ := 3 * x^5 - 4 * x^4 + 2 * x^3 - 28 * x^2 + 15 * x - 90

theorem g_evaluation_and_derivative :
  g 6 = 17568 ∧ (deriv g) 6 = 15879 := by sorry

end g_evaluation_and_derivative_l1133_113341


namespace parabola_symmetric_point_l1133_113346

/-- Parabola type -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  h_positive : p > 0
  h_equation : ∀ x y, equation x y ↔ y^2 = 2*p*x
  h_focus : focus = (p/2, 0)

/-- Line type -/
structure Line where
  angle : ℝ
  point : ℝ × ℝ

/-- Symmetric points with respect to a line -/
def symmetric (P Q : ℝ × ℝ) (l : Line) : Prop :=
  sorry

theorem parabola_symmetric_point
  (C : Parabola)
  (l : Line)
  (h_angle : l.angle = π/6)
  (h_passes : l.point = C.focus)
  (P : ℝ × ℝ)
  (h_on_parabola : C.equation P.1 P.2)
  (h_symmetric : symmetric P (5, 0) l) :
  P.1 = 2 :=
sorry

end parabola_symmetric_point_l1133_113346


namespace sum_of_three_numbers_l1133_113375

theorem sum_of_three_numbers (a b c : ℝ) : 
  a^2 + b^2 + c^2 = 252 → 
  a*b + b*c + c*a = 116 → 
  a + b + c = 22 := by
sorry

end sum_of_three_numbers_l1133_113375


namespace sum_vector_magnitude_l1133_113359

/-- Given planar vectors a and b satisfying specific conditions, 
    prove that the magnitude of their sum is 5. -/
theorem sum_vector_magnitude (a b : ℝ × ℝ) : 
  (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 3) →
  (a = (1/2, Real.sqrt 3/2)) →
  (Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5) →
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5 := by
  sorry

end sum_vector_magnitude_l1133_113359


namespace opposite_reciprocal_abs_neg_seven_l1133_113327

theorem opposite_reciprocal_abs_neg_seven :
  -(1 / |(-7 : ℤ)|) = -((1 : ℚ) / 7) := by sorry

end opposite_reciprocal_abs_neg_seven_l1133_113327


namespace simplify_expression_l1133_113303

theorem simplify_expression (a b : ℝ) (h : a + b ≠ 1) :
  1 - (1 / (1 + (a + b) / (1 - a - b))) = a + b := by
  sorry

end simplify_expression_l1133_113303


namespace intersection_of_M_and_N_l1133_113312

def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

theorem intersection_of_M_and_N : M ∩ N = Set.Ioo 1 2 ∪ {2} := by sorry

end intersection_of_M_and_N_l1133_113312


namespace bicycle_trip_speed_l1133_113335

/-- Proves that given a 12-mile trip divided into three equal parts, each taking 15 minutes,
    with speeds of 16 mph and 12 mph for the first two parts respectively,
    the speed for the last part must be 16 mph. -/
theorem bicycle_trip_speed (total_distance : ℝ) (part_time : ℝ) (speed1 speed2 : ℝ) :
  total_distance = 12 →
  part_time = 0.25 →
  speed1 = 16 →
  speed2 = 12 →
  (speed1 * part_time + speed2 * part_time + 4) = total_distance →
  4 / part_time = 16 := by
  sorry


end bicycle_trip_speed_l1133_113335


namespace unique_solution_l1133_113390

-- Define the equation
def equation (x a : ℝ) : Prop :=
  3 * x^2 + 2 * a * x - a^2 = Real.log ((x - a) / (2 * x))

-- Define the domain conditions
def domain_conditions (x a : ℝ) : Prop :=
  x - a > 0 ∧ 2 * x > 0

-- Theorem statement
theorem unique_solution (a : ℝ) (h : a ≠ 0) :
  ∃! x : ℝ, equation x a ∧ domain_conditions x a :=
by
  -- The unique solution is x = -a
  use -a
  sorry -- Proof omitted

end unique_solution_l1133_113390


namespace juans_speed_l1133_113329

/-- Given a distance of 800 miles traveled in 80.0 hours, prove that the speed is 10 miles per hour -/
theorem juans_speed (distance : ℝ) (time : ℝ) (h1 : distance = 800) (h2 : time = 80) :
  distance / time = 10 := by
  sorry

end juans_speed_l1133_113329


namespace melanie_initial_plums_l1133_113360

/-- The number of plums Melanie initially picked -/
def initial_plums : ℕ := sorry

/-- The number of plums Melanie gave to Sam -/
def plums_given : ℕ := 3

/-- The number of plums Melanie has left -/
def plums_left : ℕ := 4

/-- Theorem: Melanie initially picked 7 plums -/
theorem melanie_initial_plums : initial_plums = 7 := by
  sorry

end melanie_initial_plums_l1133_113360


namespace cubic_equation_root_sum_l1133_113316

/-- Given a cubic equation with roots a, b, c and parameter k, prove that k = 5 -/
theorem cubic_equation_root_sum (k : ℝ) (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - (k+1)*x^2 + k*x + 12 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  (a - 2)^3 + (b - 2)^3 + (c - 2)^3 = -18 →
  k = 5 := by
sorry

end cubic_equation_root_sum_l1133_113316


namespace inverse_proportion_k_value_l1133_113369

/-- Given an inverse proportion function y = (k-1)x^(k^2-5) where k is a constant,
    if y decreases as x increases when x > 0, then k = 2 -/
theorem inverse_proportion_k_value (k : ℝ) : 
  (∀ x y : ℝ, x > 0 → y = (k - 1) * x^(k^2 - 5)) →  -- y is a function of x
  (∀ x1 x2 y1 y2 : ℝ, x1 > 0 → x2 > 0 → x1 < x2 → 
    y1 = (k - 1) * x1^(k^2 - 5) → y2 = (k - 1) * x2^(k^2 - 5) → y1 > y2) →  -- y decreases as x increases
  k = 2 := by
  sorry

end inverse_proportion_k_value_l1133_113369


namespace increase_by_percentage_seventy_five_increased_by_150_percent_l1133_113371

theorem increase_by_percentage (x : ℝ) (p : ℝ) :
  x + x * (p / 100) = x * (1 + p / 100) := by sorry

theorem seventy_five_increased_by_150_percent :
  75 + 75 * (150 / 100) = 187.5 := by sorry

end increase_by_percentage_seventy_five_increased_by_150_percent_l1133_113371


namespace second_team_made_131_pieces_l1133_113363

/-- The number of fish fillet pieces made by the second team -/
def second_team_pieces (total : ℕ) (first_team : ℕ) (third_team : ℕ) : ℕ :=
  total - (first_team + third_team)

/-- Theorem stating that the second team made 131 pieces of fish fillets -/
theorem second_team_made_131_pieces : 
  second_team_pieces 500 189 180 = 131 := by
  sorry

end second_team_made_131_pieces_l1133_113363


namespace quadratic_equation_solution_l1133_113315

theorem quadratic_equation_solution (x : ℝ) : 
  x^2 - 2 * Real.sqrt 3 * x + 1 = 0 → (x - 1/x = 2 * Real.sqrt 2 ∨ x - 1/x = -2 * Real.sqrt 2) :=
by sorry

end quadratic_equation_solution_l1133_113315


namespace marys_sheep_ratio_l1133_113388

theorem marys_sheep_ratio (initial : ℕ) (remaining : ℕ) : 
  initial = 400 → remaining = 150 → (initial - remaining * 2) / initial = 1 / 4 := by
  sorry

#check marys_sheep_ratio

end marys_sheep_ratio_l1133_113388


namespace triangle_side_inequality_l1133_113332

/-- For any triangle with side lengths a, b, and c, 
    the sum of squares of the sides is less than 
    twice the sum of the products of pairs of sides. -/
theorem triangle_side_inequality (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 + b^2 + c^2 < 2*(a*b + b*c + c*a) := by
  sorry

end triangle_side_inequality_l1133_113332


namespace circle_radius_problem_l1133_113310

theorem circle_radius_problem (circle_A circle_B : ℝ) : 
  circle_A = 4 * circle_B →  -- Radius of A is 4 times radius of B
  2 * circle_A = 80 →        -- Diameter of A is 80 cm
  circle_B = 10 := by        -- Radius of B is 10 cm
sorry

end circle_radius_problem_l1133_113310


namespace f_is_quadratic_l1133_113398

/-- Definition of a quadratic function -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function we want to prove is quadratic -/
def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x

/-- Theorem stating that f is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by sorry

end f_is_quadratic_l1133_113398


namespace derivative_of_x_plus_exp_l1133_113349

/-- The derivative of f(x) = x + e^x is f'(x) = 1 + e^x -/
theorem derivative_of_x_plus_exp (x : ℝ) :
  deriv (fun x => x + Real.exp x) x = 1 + Real.exp x := by
  sorry

end derivative_of_x_plus_exp_l1133_113349


namespace unique_zero_implies_a_range_l1133_113385

/-- The cubic function f(x) = ax^3 - 3x^2 + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

theorem unique_zero_implies_a_range 
  (a : ℝ) 
  (h_unique : ∃! x₀ : ℝ, f a x₀ = 0) 
  (h_neg : ∃ x₀ : ℝ, f a x₀ = 0 ∧ x₀ < 0) :
  a > 2 :=
sorry

end unique_zero_implies_a_range_l1133_113385


namespace min_value_of_f_l1133_113337

def f (x : ℝ) : ℝ := x^2 - 6*x + 9

theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ f 3 := by sorry

end min_value_of_f_l1133_113337


namespace alloy_price_calculation_l1133_113383

/-- Calculates the price of an alloy per kg given the prices of two metals and their mixing ratio -/
theorem alloy_price_calculation (price_a price_b : ℚ) (ratio : ℚ) :
  price_a = 68 →
  price_b = 96 →
  ratio = 3 →
  (ratio * price_a + price_b) / (ratio + 1) = 75 :=
by sorry

end alloy_price_calculation_l1133_113383


namespace function_domain_implies_k_range_l1133_113331

/-- Given a function f(x) = √(kx² + kx + 3) with domain ℝ, k must be in [0, 12] -/
theorem function_domain_implies_k_range (k : ℝ) : 
  (∀ x, ∃ y, y = Real.sqrt (k * x^2 + k * x + 3)) → 0 ≤ k ∧ k ≤ 12 := by
  sorry

end function_domain_implies_k_range_l1133_113331


namespace yellow_candy_percentage_l1133_113345

theorem yellow_candy_percentage :
  ∀ (r b y : ℝ),
  r + b + y = 1 →
  y = 1.14 * b →
  r = 0.86 * b →
  y = 0.38 :=
by
  sorry

end yellow_candy_percentage_l1133_113345


namespace factorization_count_mod_1000_l1133_113323

/-- A polynomial x^2 + ax + b can be factored into linear factors with integer coefficients -/
def HasIntegerFactors (a b : ℤ) : Prop :=
  ∃ c d : ℤ, a = c + d ∧ b = c * d

/-- The count of pairs (a,b) satisfying the conditions -/
def S : ℕ :=
  (Finset.range 100).sum (fun a => 
    (Finset.range (a + 1)).card)

/-- The main theorem -/
theorem factorization_count_mod_1000 : S % 1000 = 50 := by
  sorry

end factorization_count_mod_1000_l1133_113323


namespace locus_of_P_l1133_113399

def M : ℝ × ℝ := (0, 5)
def N : ℝ × ℝ := (0, -5)

def perimeter : ℝ := 36

def is_on_locus (P : ℝ × ℝ) : Prop :=
  P.1 ≠ 0 ∧ (P.1^2 / 144 + P.2^2 / 169 = 1)

theorem locus_of_P (P : ℝ × ℝ) : 
  (dist M P + dist N P + dist M N = perimeter) → is_on_locus P :=
by sorry


end locus_of_P_l1133_113399


namespace union_determines_k_l1133_113330

def A (k : ℕ) : Set ℕ := {1, 2, k}
def B : Set ℕ := {2, 5}

theorem union_determines_k (k : ℕ) : A k ∪ B = {1, 2, 3, 5} → k = 3 := by
  sorry

end union_determines_k_l1133_113330


namespace franks_books_l1133_113392

theorem franks_books (days_per_book : ℕ) (total_days : ℕ) (h1 : days_per_book = 12) (h2 : total_days = 492) :
  total_days / days_per_book = 41 := by
  sorry

end franks_books_l1133_113392


namespace existence_of_factors_l1133_113336

theorem existence_of_factors : ∃ (a b c d : ℕ), 
  (10 ≤ a ∧ a < 100) ∧ 
  (10 ≤ b ∧ b < 100) ∧ 
  (10 ≤ c ∧ c < 100) ∧ 
  (10 ≤ d ∧ d < 100) ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * a * b * c * d = 2016000 :=
by sorry

end existence_of_factors_l1133_113336


namespace inequality_solution_range_l1133_113393

theorem inequality_solution_range (k : ℝ) : 
  (∀ x : ℝ, (k - 1) * x^2 + (k - 1) * x + 2 > 0) ↔ k ∈ Set.Icc 1 9 := by
  sorry

end inequality_solution_range_l1133_113393


namespace yellow_ball_percentage_l1133_113353

/-- Given the number of yellow and brown balls, calculate the percentage of yellow balls -/
theorem yellow_ball_percentage (yellow_balls brown_balls : ℕ) : 
  yellow_balls = 27 → brown_balls = 33 → 
  (yellow_balls : ℚ) / (yellow_balls + brown_balls : ℚ) * 100 = 45 := by
  sorry

end yellow_ball_percentage_l1133_113353


namespace x_range_l1133_113307

theorem x_range (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -4) : x > 1 / 3 := by
  sorry

end x_range_l1133_113307


namespace seventh_term_is_five_l1133_113380

/-- A geometric sequence with given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)
  a_3_eq_1 : a 3 = 1
  a_11_eq_25 : a 11 = 25

/-- The 7th term of the geometric sequence is 5 -/
theorem seventh_term_is_five (seq : GeometricSequence) : seq.a 7 = 5 := by
  sorry

end seventh_term_is_five_l1133_113380


namespace lcm_18_30_l1133_113319

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l1133_113319


namespace roots_distance_bound_l1133_113302

theorem roots_distance_bound (v w : ℂ) : 
  v ≠ w → 
  (v^401 = 1) → 
  (w^401 = 1) → 
  Complex.abs (v + w) < Real.sqrt (3 + Real.sqrt 5) := by
sorry

end roots_distance_bound_l1133_113302


namespace farm_chickens_count_l1133_113362

/-- Proves that the total number of chickens on a farm is 69, given the number of ducks, geese, and their relationships to hens and roosters. -/
theorem farm_chickens_count (ducks geese : ℕ) 
  (h1 : ducks = 45)
  (h2 : geese = 28)
  (h3 : ∃ hens : ℕ, hens = ducks - 13)
  (h4 : ∃ roosters : ℕ, roosters = geese + 9) :
  ∃ total_chickens : ℕ, total_chickens = 69 ∧ 
    ∃ (hens roosters : ℕ), 
      hens = ducks - 13 ∧ 
      roosters = geese + 9 ∧ 
      total_chickens = hens + roosters := by
sorry

end farm_chickens_count_l1133_113362


namespace incorrect_statement_l1133_113318

-- Define the concept of planes
variable (α β : Set (ℝ × ℝ × ℝ))

-- Define perpendicularity between planes
def perpendicular (p q : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the concept of a line
def Line : Type := Set (ℝ × ℝ × ℝ)

-- Define perpendicularity between a line and a plane
def line_perp_plane (l : Line) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the intersection line of two planes
def intersection_line (p q : Set (ℝ × ℝ × ℝ)) : Line := sorry

-- Define a function to create a perpendicular line from a point to a line
def perp_line_to_line (point : ℝ × ℝ × ℝ) (l : Line) : Line := sorry

-- Theorem to be disproved
theorem incorrect_statement 
  (h1 : perpendicular α β)
  (point : ℝ × ℝ × ℝ)
  (h2 : point ∈ α) :
  line_perp_plane (perp_line_to_line point (intersection_line α β)) β := by
  sorry

end incorrect_statement_l1133_113318


namespace simplify_fourth_roots_l1133_113351

theorem simplify_fourth_roots : 64^(1/4) - 144^(1/4) = 2 * Real.sqrt 2 - 12 := by
  sorry

end simplify_fourth_roots_l1133_113351


namespace equilateral_triangle_area_perimeter_ratio_l1133_113326

theorem equilateral_triangle_area_perimeter_ratio :
  let s : ℝ := 10
  let altitude : ℝ := s * Real.sqrt 3 / 2
  let area : ℝ := s * altitude / 2
  let perimeter : ℝ := 3 * s
  area / perimeter = 5 * Real.sqrt 3 / 6 := by
sorry

end equilateral_triangle_area_perimeter_ratio_l1133_113326


namespace vote_count_proof_l1133_113381

theorem vote_count_proof (total votes_against votes_in_favor : ℕ) 
  (h1 : votes_in_favor = votes_against + 68)
  (h2 : votes_against = (40 : ℕ) * total / 100)
  (h3 : total = votes_in_favor + votes_against) :
  total = 340 :=
sorry

end vote_count_proof_l1133_113381


namespace rectangular_box_volume_l1133_113340

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 30)
  (area2 : w * h = 40)
  (area3 : l * h = 12) :
  l * w * h = 120 := by
sorry

end rectangular_box_volume_l1133_113340


namespace claire_photos_l1133_113306

theorem claire_photos (lisa robert claire : ℕ) 
  (h1 : lisa = robert)
  (h2 : lisa = 3 * claire)
  (h3 : robert = claire + 20) :
  claire = 10 := by
  sorry

end claire_photos_l1133_113306


namespace exists_eight_numbers_sum_divisible_l1133_113305

theorem exists_eight_numbers_sum_divisible : 
  ∃ (S : Finset ℕ), 
    S.card = 8 ∧ 
    (∀ n ∈ S, n ≤ 100) ∧
    (∀ n ∈ S, (S.sum id) % n = 0) :=
sorry

end exists_eight_numbers_sum_divisible_l1133_113305


namespace cubic_factorization_l1133_113396

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end cubic_factorization_l1133_113396


namespace unique_solution_two_power_minus_three_power_l1133_113361

theorem unique_solution_two_power_minus_three_power : 
  ∀ m n : ℕ+, 2^(m:ℕ) - 3^(n:ℕ) = 7 → m = 4 ∧ n = 2 :=
by sorry

end unique_solution_two_power_minus_three_power_l1133_113361


namespace initial_white_lights_equal_total_colored_lights_l1133_113339

/-- The number of white lights Malcolm had initially -/
def initialWhiteLights : ℕ := sorry

/-- The number of red lights Malcolm bought -/
def redLights : ℕ := 12

/-- The number of blue lights Malcolm bought -/
def blueLights : ℕ := 3 * redLights

/-- The number of green lights Malcolm bought -/
def greenLights : ℕ := 6

/-- The number of colored lights Malcolm still needs to buy -/
def remainingLights : ℕ := 5

/-- Theorem stating that the initial number of white lights equals the total number of colored lights -/
theorem initial_white_lights_equal_total_colored_lights :
  initialWhiteLights = redLights + blueLights + greenLights + remainingLights := by sorry

end initial_white_lights_equal_total_colored_lights_l1133_113339


namespace short_story_booklets_l1133_113301

theorem short_story_booklets (pages_per_booklet : ℕ) (total_pages : ℕ) (h1 : pages_per_booklet = 9) (h2 : total_pages = 441) :
  total_pages / pages_per_booklet = 49 := by
  sorry

end short_story_booklets_l1133_113301


namespace x_twelfth_power_l1133_113364

theorem x_twelfth_power (x : ℝ) (h : x + 1/x = 2 * Real.sqrt 2) : x^12 = 46656 := by
  sorry

end x_twelfth_power_l1133_113364


namespace edge_sum_greater_than_3d_l1133_113355

-- Define a convex polyhedron
structure ConvexPolyhedron where
  vertices : Set (Fin 3 → ℝ)
  edges : Set (Fin 2 → Fin 3 → ℝ)
  is_convex : Bool

-- Define the maximum distance between vertices
def max_distance (p : ConvexPolyhedron) : ℝ :=
  sorry

-- Define the sum of edge lengths
def sum_edge_lengths (p : ConvexPolyhedron) : ℝ :=
  sorry

-- The theorem to prove
theorem edge_sum_greater_than_3d (p : ConvexPolyhedron) :
  p.is_convex → sum_edge_lengths p > 3 * max_distance p :=
by sorry

end edge_sum_greater_than_3d_l1133_113355


namespace triangle_reciprocal_side_angle_bisector_equality_l1133_113378

/-- For any triangle, the sum of reciprocals of side lengths equals the sum of cosines of half angles divided by their respective angle bisector lengths. -/
theorem triangle_reciprocal_side_angle_bisector_equality
  (a b c : ℝ) (α β γ : ℝ) (f_α f_β f_γ : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_angle_sum : α + β + γ = π)
  (h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_f_α : f_α = (2 * b * c * Real.cos (α / 2)) / (b + c))
  (h_f_β : f_β = (2 * a * c * Real.cos (β / 2)) / (a + c))
  (h_f_γ : f_γ = (2 * a * b * Real.cos (γ / 2)) / (a + b)) :
  1 / a + 1 / b + 1 / c = Real.cos (α / 2) / f_α + Real.cos (β / 2) / f_β + Real.cos (γ / 2) / f_γ :=
by sorry

end triangle_reciprocal_side_angle_bisector_equality_l1133_113378


namespace geometric_sequence_a6_l1133_113321

/-- Given a geometric sequence {aₙ} where a₁ = 1 and a₄ = 8, prove that a₆ = 32 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 1 = 1 →                                  -- a₁ = 1
  a 4 = 8 →                                  -- a₄ = 8
  a 6 = 32 := by
sorry

end geometric_sequence_a6_l1133_113321


namespace smallest_k_sum_squares_multiple_180_sum_squares_360_multiple_180_smallest_k_is_360_l1133_113338

theorem smallest_k_sum_squares_multiple_180 :
  ∀ k : ℕ+, (k.val * (k.val + 1) * (2 * k.val + 1)) % 1080 = 0 → k.val ≥ 360 :=
by sorry

theorem sum_squares_360_multiple_180 :
  (360 * 361 * 721) % 1080 = 0 :=
by sorry

theorem smallest_k_is_360 :
  ∃! k : ℕ+, k.val = 360 ∧
    (∀ m : ℕ+, (m.val * (m.val + 1) * (2 * m.val + 1)) % 1080 = 0 → k ≤ m) ∧
    (k.val * (k.val + 1) * (2 * k.val + 1)) % 1080 = 0 :=
by sorry

end smallest_k_sum_squares_multiple_180_sum_squares_360_multiple_180_smallest_k_is_360_l1133_113338


namespace probability_x_gt_9y_l1133_113365

/-- The probability that a randomly chosen point (x,y) from a rectangle
    with vertices (0,0), (2017,0), (2017,2018), and (0,2018) satisfies x > 9y -/
theorem probability_x_gt_9y : Real := by
  -- Define the rectangle
  let rectangle_width : ℕ := 2017
  let rectangle_height : ℕ := 2018

  -- Define the condition x > 9y
  let condition (x y : Real) : Prop := x > 9 * y

  -- Define the probability
  let probability : Real := 2017 / 36324

  -- Proof goes here
  sorry

end probability_x_gt_9y_l1133_113365


namespace second_caterer_cheaper_at_34_l1133_113368

def first_caterer_cost (n : ℕ) : ℝ := 50 + 18 * n

def second_caterer_cost (n : ℕ) : ℝ :=
  if n ≥ 30 then 150 + 15 * n else 180 + 15 * n

theorem second_caterer_cheaper_at_34 :
  (∀ n : ℕ, n ≥ 34 → second_caterer_cost n < first_caterer_cost n) ∧
  (∀ n : ℕ, n < 34 → second_caterer_cost n ≥ first_caterer_cost n) :=
sorry

end second_caterer_cheaper_at_34_l1133_113368


namespace expression_evaluation_l1133_113379

theorem expression_evaluation : 
  (0.8 : ℝ)^3 - (0.5 : ℝ)^3 / (0.8 : ℝ)^2 + 0.40 + (0.5 : ℝ)^2 = 0.9666875 := by
  sorry

end expression_evaluation_l1133_113379


namespace angle_in_linear_pair_l1133_113382

/-- 
Given a line segment AB with three angles:
- ACD = 90°
- ECB = 52°
- DCE = x°
Prove that x = 38°
-/
theorem angle_in_linear_pair (x : ℝ) : 
  90 + x + 52 = 180 → x = 38 := by sorry

end angle_in_linear_pair_l1133_113382


namespace besfamilies_children_count_l1133_113304

/-- Represents the Besfamilies family structure and age calculations -/
structure Besfamilies where
  initialAge : ℕ  -- Family age when youngest child was born
  finalAge : ℕ    -- Family age after several years
  yearsPassed : ℕ -- Number of years passed

/-- Calculates the number of children in the Besfamilies -/
def numberOfChildren (family : Besfamilies) : ℕ :=
  ((family.finalAge - family.initialAge) / family.yearsPassed) - 2

/-- Theorem stating the number of children in the Besfamilies -/
theorem besfamilies_children_count 
  (family : Besfamilies) 
  (h1 : family.initialAge = 101)
  (h2 : family.finalAge = 150)
  (h3 : family.yearsPassed > 1)
  (h4 : (family.finalAge - family.initialAge) % family.yearsPassed = 0) :
  numberOfChildren family = 5 := by
  sorry

#eval numberOfChildren { initialAge := 101, finalAge := 150, yearsPassed := 7 }

end besfamilies_children_count_l1133_113304


namespace equation_solution_polynomial_expansion_l1133_113348

-- Part 1: Equation solution
theorem equation_solution :
  {x : ℝ | 9 * (x - 3)^2 - 121 = 0} = {20/3, -2/3} := by sorry

-- Part 2: Polynomial expansion
theorem polynomial_expansion (x y : ℝ) :
  (x - 2*y) * (x^2 + 2*x*y + 4*y^2) = x^3 - 8*y^3 := by sorry

end equation_solution_polynomial_expansion_l1133_113348


namespace complex_division_result_l1133_113367

theorem complex_division_result : ∃ (i : ℂ), i * i = -1 ∧ (4 * i) / (1 + i) = 2 + 2 * i :=
by sorry

end complex_division_result_l1133_113367


namespace expression_nonnegative_iff_l1133_113373

theorem expression_nonnegative_iff (x : ℝ) : 
  (3*x - 12*x^2 + 48*x^3) / (27 - x^3) ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 3 := by
  sorry

end expression_nonnegative_iff_l1133_113373


namespace hex_numeric_count_and_sum_l1133_113389

/-- Represents a hexadecimal digit --/
inductive HexDigit
| Numeric (n : Fin 10)
| Alpha (a : Fin 6)

/-- Converts a natural number to its hexadecimal representation --/
def toHex (n : ℕ) : List HexDigit := sorry

/-- Checks if a hexadecimal representation contains only numeric digits --/
def onlyNumeric (hex : List HexDigit) : Bool := sorry

/-- Counts the number of positive integers up to n whose hexadecimal 
    representation contains only numeric digits --/
def countNumericHex (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a natural number --/
def sumOfDigits (n : ℕ) : ℕ := sorry

theorem hex_numeric_count_and_sum : 
  countNumericHex 2000 = 1999 ∧ sumOfDigits 1999 = 28 := by sorry

end hex_numeric_count_and_sum_l1133_113389


namespace water_for_lemonade_l1133_113320

/-- Represents the ratio of water to lemon juice in the lemonade mixture -/
def water_to_juice_ratio : ℚ := 7 / 1

/-- Represents the number of quarts in a gallon -/
def quarts_per_gallon : ℚ := 4

/-- Calculates the amount of water needed to make one gallon of lemonade -/
def water_needed (ratio : ℚ) (quarts_in_gallon : ℚ) : ℚ :=
  (ratio * quarts_in_gallon) / (ratio + 1)

/-- Theorem stating that the amount of water needed for one gallon of lemonade is 7/2 quarts -/
theorem water_for_lemonade :
  water_needed water_to_juice_ratio quarts_per_gallon = 7 / 2 := by
  sorry

end water_for_lemonade_l1133_113320


namespace trigonometric_identity_l1133_113311

theorem trigonometric_identity (α : Real) : 
  0 < α ∧ α < π/2 →
  Real.sin (5*π/12 + 2*α) = -3/5 →
  Real.sin (π/12 + α) * Real.sin (5*π/12 - α) = Real.sqrt 2 / 20 := by
  sorry

end trigonometric_identity_l1133_113311


namespace mixture_ratio_l1133_113370

/-- Given two solutions A and B with different alcohol-water ratios, 
    prove that mixing them in a specific ratio results in a mixture with 60% alcohol. -/
theorem mixture_ratio (V_A V_B : ℝ) : 
  V_A > 0 → V_B > 0 →
  (21 / 25 * V_A + 2 / 5 * V_B) / (V_A + V_B) = 3 / 5 →
  V_A / V_B = 5 / 6 := by
sorry

/-- The ratio of Solution A to Solution B in the mixture -/
def solution_ratio : ℚ := 5 / 6

#check mixture_ratio
#check solution_ratio

end mixture_ratio_l1133_113370


namespace graph_not_in_second_quadrant_implies_a_nonnegative_l1133_113300

-- Define the function f(x) = x^3 - a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a

-- Define the condition that the graph does not pass through the second quadrant
def not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f a x ≤ 0

-- Theorem statement
theorem graph_not_in_second_quadrant_implies_a_nonnegative (a : ℝ) :
  not_in_second_quadrant a → a ≥ 0 := by
  sorry

end graph_not_in_second_quadrant_implies_a_nonnegative_l1133_113300


namespace max_erasers_purchase_l1133_113350

def pen_cost : ℕ := 3
def pencil_cost : ℕ := 4
def eraser_cost : ℕ := 8
def total_budget : ℕ := 60

def is_valid_purchase (pens pencils erasers : ℕ) : Prop :=
  pens ≥ 1 ∧ pencils ≥ 1 ∧ erasers ≥ 1 ∧
  pens * pen_cost + pencils * pencil_cost + erasers * eraser_cost = total_budget

theorem max_erasers_purchase :
  ∃ (pens pencils : ℕ), is_valid_purchase pens pencils 5 ∧
  ∀ (p n e : ℕ), is_valid_purchase p n e → e ≤ 5 :=
sorry

end max_erasers_purchase_l1133_113350


namespace perpendicular_vectors_x_value_l1133_113313

/-- Given two vectors a and b in ℝ³, where a = (2, -3, 1) and b = (4, -6, x),
    if a is perpendicular to b, then x = -26. -/
theorem perpendicular_vectors_x_value :
  let a : Fin 3 → ℝ := ![2, -3, 1]
  let b : Fin 3 → ℝ := ![4, -6, x]
  (∀ i : Fin 3, a i * b i = 0) → x = -26 := by
  sorry

end perpendicular_vectors_x_value_l1133_113313


namespace fifth_element_is_35_l1133_113386

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  totalElements : ℕ
  sampleSize : ℕ
  firstElement : ℕ

/-- Calculates the nth element in a systematic sample -/
def nthElement (s : SystematicSampling) (n : ℕ) : ℕ :=
  s.firstElement + (n - 1) * (s.totalElements / s.sampleSize)

theorem fifth_element_is_35 (s : SystematicSampling) 
  (h1 : s.totalElements = 160)
  (h2 : s.sampleSize = 20)
  (h3 : s.firstElement = 3) :
  nthElement s 5 = 35 := by
  sorry

end fifth_element_is_35_l1133_113386


namespace students_behind_in_line_l1133_113308

/-- Given a line of students waiting for a bus, this theorem proves
    the number of students behind a specific student. -/
theorem students_behind_in_line
  (total_students : ℕ)
  (students_in_front : ℕ)
  (h1 : total_students = 30)
  (h2 : students_in_front = 20) :
  total_students - (students_in_front + 1) = 9 := by
  sorry

end students_behind_in_line_l1133_113308


namespace sum_of_bases_is_sixteen_l1133_113333

/-- Represents a repeating decimal in a given base -/
structure RepeatingDecimal (base : ℕ) where
  integerPart : ℕ
  repeatingPart : ℕ

/-- Given two bases and representations of G₁ and G₂ in those bases, proves their sum is 16 -/
theorem sum_of_bases_is_sixteen
  (S₁ S₂ : ℕ)
  (G₁_in_S₁ : RepeatingDecimal S₁)
  (G₂_in_S₁ : RepeatingDecimal S₁)
  (G₁_in_S₂ : RepeatingDecimal S₂)
  (G₂_in_S₂ : RepeatingDecimal S₂)
  (h₁ : G₁_in_S₁ = ⟨0, 45⟩)
  (h₂ : G₂_in_S₁ = ⟨0, 54⟩)
  (h₃ : G₁_in_S₂ = ⟨0, 14⟩)
  (h₄ : G₂_in_S₂ = ⟨0, 41⟩)
  : S₁ + S₂ = 16 :=
sorry

end sum_of_bases_is_sixteen_l1133_113333


namespace sequence_is_geometric_progression_l1133_113376

theorem sequence_is_geometric_progression (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) 
  (h : ∀ n : ℕ+, S n = (1 : ℚ) / 3 * (a n - 1)) :
  a 1 = -(1 : ℚ) / 2 ∧ 
  a 2 = (1 : ℚ) / 4 ∧ 
  (∀ n : ℕ+, n > 1 → a n / a (n - 1) = -(1 : ℚ) / 2) := by
  sorry

end sequence_is_geometric_progression_l1133_113376


namespace sarah_apple_slices_l1133_113357

/-- Given a number of boxes of apples, apples per box, and slices per apple,
    calculate the total number of apple slices -/
def total_apple_slices (boxes : ℕ) (apples_per_box : ℕ) (slices_per_apple : ℕ) : ℕ :=
  boxes * apples_per_box * slices_per_apple

/-- Theorem: Sarah has 392 apple slices -/
theorem sarah_apple_slices :
  total_apple_slices 7 7 8 = 392 := by
  sorry

end sarah_apple_slices_l1133_113357


namespace line_perp_plane_implies_planes_perp_l1133_113358

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Defines that a line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines that a line is perpendicular to a plane -/
def line_perp_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines that two planes are perpendicular -/
def planes_perpendicular (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line contained in a plane is perpendicular to another plane,
    then the two planes are perpendicular -/
theorem line_perp_plane_implies_planes_perp
  (l : Line3D) (α β : Plane3D)
  (h1 : line_in_plane l α)
  (h2 : line_perp_plane l β) :
  planes_perpendicular α β :=
sorry

end line_perp_plane_implies_planes_perp_l1133_113358


namespace isosceles_triangle_areas_sum_l1133_113384

/-- Represents the areas of right isosceles triangles constructed on the sides of a right triangle -/
structure TriangleAreas where
  A : ℝ  -- Area of the isosceles triangle on side 5
  B : ℝ  -- Area of the isosceles triangle on side 12
  C : ℝ  -- Area of the isosceles triangle on side 13

/-- Theorem: For a right triangle with sides 5, 12, and 13, 
    if right isosceles triangles are constructed on each side, 
    then the sum of the areas of the triangles on the two shorter sides 
    equals the area of the triangle on the hypotenuse -/
theorem isosceles_triangle_areas_sum (areas : TriangleAreas) 
  (h1 : areas.A = (5 * 5) / 2)
  (h2 : areas.B = (12 * 12) / 2)
  (h3 : areas.C = (13 * 13) / 2) : 
  areas.A + areas.B = areas.C := by
  sorry

end isosceles_triangle_areas_sum_l1133_113384


namespace series_sum_convergence_l1133_113397

open Real
open BigOperators

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 2)) converges to 5/6 -/
theorem series_sum_convergence :
  ∑' n : ℕ, (3 * n - 2 : ℝ) / (n * (n + 1) * (n + 2)) = 5/6 := by sorry

end series_sum_convergence_l1133_113397


namespace equal_roots_quadratic_l1133_113317

theorem equal_roots_quadratic (q : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + q = 0 ∧ 
   ∀ y : ℝ, y^2 - 3*y + q = 0 → y = x) ↔ 
  q = 9/4 := by
sorry

end equal_roots_quadratic_l1133_113317


namespace smallest_x_equals_f_2001_l1133_113309

def f (x : ℝ) : ℝ := sorry

axiom f_triple (x : ℝ) (h : 0 < x) : f (3 * x) = 3 * f x

axiom f_definition (x : ℝ) (h : 1 ≤ x ∧ x ≤ 3) : f x = 1 - |x - 2|

theorem smallest_x_equals_f_2001 :
  ∃ (x : ℝ), x > 0 ∧ f x = f 2001 ∧ ∀ (y : ℝ), y > 0 ∧ f y = f 2001 → x ≤ y :=
by
  sorry

end smallest_x_equals_f_2001_l1133_113309


namespace disjunction_truth_l1133_113328

theorem disjunction_truth (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end disjunction_truth_l1133_113328


namespace fresh_grape_water_content_l1133_113387

/-- The percentage of water in raisins -/
def raisin_water_percentage : ℝ := 25

/-- The weight of fresh grapes used -/
def fresh_grape_weight : ℝ := 100

/-- The weight of raisins produced -/
def raisin_weight : ℝ := 20

/-- The percentage of water in fresh grapes -/
def fresh_grape_water_percentage : ℝ := 85

theorem fresh_grape_water_content :
  fresh_grape_water_percentage = 85 :=
sorry

end fresh_grape_water_content_l1133_113387
