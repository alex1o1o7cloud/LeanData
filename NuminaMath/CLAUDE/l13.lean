import Mathlib

namespace condition_necessary_not_sufficient_l13_1329

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

def condition (a : ℕ → ℝ) : Prop :=
  ∀ n, |a (n + 1)| > a n

theorem condition_necessary_not_sufficient (a : ℕ → ℝ) :
  (is_increasing a → condition a) ∧ ¬(condition a → is_increasing a) :=
sorry

end condition_necessary_not_sufficient_l13_1329


namespace opposite_of_negative_three_l13_1350

theorem opposite_of_negative_three : -(-(3 : ℤ)) = 3 := by sorry

end opposite_of_negative_three_l13_1350


namespace total_sharks_l13_1388

/-- The total number of sharks on three beaches given specific ratios -/
theorem total_sharks (newport : ℕ) (dana_point : ℕ) (huntington : ℕ) 
  (h1 : newport = 22)
  (h2 : dana_point = 4 * newport)
  (h3 : huntington = dana_point / 2) :
  newport + dana_point + huntington = 154 := by
  sorry

end total_sharks_l13_1388


namespace percentage_equation_solution_l13_1383

/-- The solution to the equation (47% of 1442 - x% of 1412) + 63 = 252 is approximately 34.63% -/
theorem percentage_equation_solution : 
  ∃ x : ℝ, abs (x - 34.63) < 0.01 ∧ 
  ((47 / 100) * 1442 - (x / 100) * 1412) + 63 = 252 := by
  sorry

end percentage_equation_solution_l13_1383


namespace symmetry_of_shifted_even_function_l13_1340

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define the concept of axis of symmetry
def AxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop := 
  ∀ x : ℝ, f (a + x) = f (a - x)

theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) (h : EvenFunction f) :
  AxisOfSymmetry (fun x ↦ f (x + 1)) (-1) := by
sorry

end symmetry_of_shifted_even_function_l13_1340


namespace interest_earned_proof_l13_1377

def initial_investment : ℝ := 1200
def annual_interest_rate : ℝ := 0.12
def compounding_periods : ℕ := 4

def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

theorem interest_earned_proof :
  let final_amount := compound_interest initial_investment annual_interest_rate compounding_periods
  let total_interest := final_amount - initial_investment
  ∃ ε > 0, |total_interest - 688.22| < ε :=
sorry

end interest_earned_proof_l13_1377


namespace min_value_expression_l13_1390

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 128) :
  x^2 + 8*x*y + 4*y^2 + 8*z^2 ≥ 384 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 128 ∧ x₀^2 + 8*x₀*y₀ + 4*y₀^2 + 8*z₀^2 = 384 := by
  sorry

end min_value_expression_l13_1390


namespace pyramid_sum_l13_1313

/-- Given a pyramid of numbers where each number is the sum of the two above it,
    prove that the top number is 381. -/
theorem pyramid_sum (y z : ℕ) (h1 : y + 600 = 1119) (h2 : z + 1119 = 2019) (h3 : 381 + y = z) :
  ∃ x : ℕ, x = 381 ∧ x + y = z :=
by sorry

end pyramid_sum_l13_1313


namespace quadratic_function_properties_l13_1337

def quadratic_function (a m b x : ℝ) := a * x * (x - m) + b

theorem quadratic_function_properties
  (a m b : ℝ)
  (h_a_nonzero : a ≠ 0)
  (h_y_1_at_0 : quadratic_function a m b 0 = 1)
  (h_y_1_at_2 : quadratic_function a m b 2 = 1)
  (h_y_gt_4_at_3 : quadratic_function a m b 3 > 4)
  (k : ℝ)
  (h_passes_1_k : quadratic_function a m b 1 = k)
  (h_k_over_a : 0 < k / a ∧ k / a < 1) :
  m = 2 ∧ b = 1 ∧ a > 1 ∧ 1/2 < a ∧ a < 1 := by
  sorry

end quadratic_function_properties_l13_1337


namespace square_expression_l13_1399

theorem square_expression (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  (1 / (1 / x - 1 / (x + 1)) - x = x^2) ∧
  (1 / (1 / (x - 1) - 1 / x) + x = x^2) := by
  sorry

end square_expression_l13_1399


namespace min_fraction_sum_l13_1361

/-- The set of digits to choose from -/
def digits : Finset ℕ := {0, 1, 5, 6, 7, 8, 9}

/-- The proposition that four natural numbers are distinct digits from our set -/
def are_distinct_digits (w x y z : ℕ) : Prop :=
  w ∈ digits ∧ x ∈ digits ∧ y ∈ digits ∧ z ∈ digits ∧
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z

/-- The theorem stating the minimum value of the sum -/
theorem min_fraction_sum :
  ∃ (w x y z : ℕ), are_distinct_digits w x y z ∧ x ≠ 0 ∧ z ≠ 0 ∧
  (∀ (w' x' y' z' : ℕ), are_distinct_digits w' x' y' z' ∧ x' ≠ 0 ∧ z' ≠ 0 →
    (w : ℚ) / x + (y : ℚ) / z ≤ (w' : ℚ) / x' + (y' : ℚ) / z') ∧
  (w : ℚ) / x + (y : ℚ) / z = 1 / 8 :=
sorry

end min_fraction_sum_l13_1361


namespace debate_committee_combinations_l13_1374

/-- The number of teams in the debate club -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members selected from the organizing team -/
def organizing_team_selection : ℕ := 4

/-- The number of members selected from each non-organizing team -/
def other_team_selection : ℕ := 3

/-- The total number of members in the debate organizing committee -/
def committee_size : ℕ := 16

/-- The number of possible debate organizing committees -/
def num_committees : ℕ := 3442073600

theorem debate_committee_combinations :
  (num_teams * Nat.choose team_size organizing_team_selection * 
   (Nat.choose team_size other_team_selection ^ (num_teams - 1))) = num_committees :=
sorry

end debate_committee_combinations_l13_1374


namespace hyperbola_equilateral_triangle_l13_1354

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define an equilateral triangle
def is_equilateral_triangle (P Q R : ℝ × ℝ) : Prop :=
  let d₁ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let d₂ := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  let d₃ := Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2)
  d₁ = d₂ ∧ d₂ = d₃

-- Define the branches of the hyperbola
def on_branch_1 (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ hyperbola x y
def on_branch_2 (x y : ℝ) : Prop := x < 0 ∧ y < 0 ∧ hyperbola x y

-- Theorem statement
theorem hyperbola_equilateral_triangle :
  ∀ (P Q R : ℝ × ℝ),
  hyperbola P.1 P.2 → hyperbola Q.1 Q.2 → hyperbola R.1 R.2 →
  is_equilateral_triangle P Q R →
  P = (-1, -1) →
  on_branch_2 P.1 P.2 →
  on_branch_1 Q.1 Q.2 →
  on_branch_1 R.1 R.2 →
  (¬(on_branch_1 P.1 P.2 ∧ on_branch_1 Q.1 Q.2 ∧ on_branch_1 R.1 R.2) ∧
   ¬(on_branch_2 P.1 P.2 ∧ on_branch_2 Q.1 Q.2 ∧ on_branch_2 R.1 R.2)) ∧
  ((Q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧ R = (2 + Real.sqrt 3, 2 - Real.sqrt 3)) ∨
   (Q = (2 + Real.sqrt 3, 2 - Real.sqrt 3) ∧ R = (2 - Real.sqrt 3, 2 + Real.sqrt 3))) :=
by sorry

end hyperbola_equilateral_triangle_l13_1354


namespace vectors_not_collinear_l13_1397

/-- Given two vectors a and b in ℝ³, we define c₁ and c₂ as linear combinations of a and b.
    This theorem states that c₁ and c₂ are not collinear. -/
theorem vectors_not_collinear :
  let a : Fin 3 → ℝ := ![1, 0, 1]
  let b : Fin 3 → ℝ := ![-2, 3, 5]
  let c₁ : Fin 3 → ℝ := a + 2 • b
  let c₂ : Fin 3 → ℝ := 3 • a - b
  ¬ ∃ (k : ℝ), c₁ = k • c₂ := by
  sorry

end vectors_not_collinear_l13_1397


namespace lower_interest_rate_l13_1312

/-- Calculates simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem lower_interest_rate 
  (principal : ℝ) 
  (high_rate low_rate : ℝ) 
  (time : ℝ) 
  (interest_difference : ℝ) :
  principal = 12000 →
  high_rate = 0.15 →
  time = 2 →
  interest_difference = 720 →
  simple_interest principal high_rate time - simple_interest principal low_rate time = interest_difference →
  low_rate = 0.12 := by
sorry

end lower_interest_rate_l13_1312


namespace courtyard_width_l13_1348

theorem courtyard_width (length : ℝ) (num_bricks : ℕ) (brick_length brick_width : ℝ) :
  length = 25 ∧ 
  num_bricks = 20000 ∧ 
  brick_length = 0.2 ∧ 
  brick_width = 0.1 →
  (num_bricks : ℝ) * brick_length * brick_width / length = 16 :=
by sorry

end courtyard_width_l13_1348


namespace sqrt_equation_solution_l13_1336

theorem sqrt_equation_solution (x y : ℝ) :
  Real.sqrt (x^2 + y^2 - 1) = 1 - x - y ↔ (x = 1 ∧ y ≤ 0) ∨ (y = 1 ∧ x ≤ 0) := by
  sorry

end sqrt_equation_solution_l13_1336


namespace recurrence_sequence_properties_l13_1365

/-- A sequence that satisfies the given recurrence relation -/
def RecurrenceSequence (x : ℕ → ℝ) (a : ℝ) : Prop :=
  ∀ n, x (n + 2) = 3 * x (n + 1) - 2 * x n + a

/-- An arithmetic progression -/
def ArithmeticProgression (x : ℕ → ℝ) (b c : ℝ) : Prop :=
  ∀ n, x n = b + (c - b) * (n - 1)

/-- A geometric progression -/
def GeometricProgression (x : ℕ → ℝ) (b q : ℝ) : Prop :=
  ∀ n, x n = b * q^(n - 1)

theorem recurrence_sequence_properties
  (x : ℕ → ℝ) (a b c : ℝ) (h : a < 0) :
  (RecurrenceSequence x a ∧ ArithmeticProgression x b c) →
    (a = c - b ∧ c < b) ∧
  (RecurrenceSequence x a ∧ GeometricProgression x b 2) →
    (a = 0 ∧ c = 2*b ∧ b > 0) :=
by sorry

end recurrence_sequence_properties_l13_1365


namespace even_increasing_negative_inequality_l13_1381

/-- A function that is even and increasing on (-∞, -1] -/
def EvenIncreasingNegative (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, x ≤ y → y ≤ -1 → f x ≤ f y)

/-- Theorem stating the inequality for functions that are even and increasing on (-∞, -1] -/
theorem even_increasing_negative_inequality (f : ℝ → ℝ) 
  (h : EvenIncreasingNegative f) : 
  f 2 < f (-3/2) ∧ f (-3/2) < f (-1) := by
  sorry

end even_increasing_negative_inequality_l13_1381


namespace tyler_aquariums_l13_1392

-- Define the given conditions
def animals_per_aquarium : ℕ := 64
def total_animals : ℕ := 512

-- State the theorem
theorem tyler_aquariums : 
  total_animals / animals_per_aquarium = 8 := by
  sorry

end tyler_aquariums_l13_1392


namespace quadratic_function_properties_l13_1364

-- Define the quadratic function
def f (b c : ℝ) (x : ℝ) : ℝ := 4 * x^2 + b * x + c

-- Theorem statement
theorem quadratic_function_properties :
  ∀ b c : ℝ,
  (f b c (-1) = -1 ∧ f b c 0 = 0) →
  (∃ x₁ x₂ : ℝ, f b c x₁ = 20 ∧ f b c x₂ = 20 ∧ f b c (x₁ + x₂) = 0) →
  (∀ x : ℝ, (x < -5/4 ∨ x > 0) → f b c x > 0) :=
by sorry

end quadratic_function_properties_l13_1364


namespace expression_evaluation_l13_1322

theorem expression_evaluation : -20 + 8 * (10 / 2) - 4 = 16 := by
  sorry

end expression_evaluation_l13_1322


namespace polynomial_value_at_negative_two_l13_1341

-- Define the polynomial
def P (a b x : ℝ) : ℝ := a * (x^3 - x^2 + 3*x) + b * (2*x^2 + x) + x^3 - 5

-- State the theorem
theorem polynomial_value_at_negative_two 
  (a b : ℝ) 
  (h : P a b 2 = -17) : 
  P a b (-2) = -1 := by
sorry

end polynomial_value_at_negative_two_l13_1341


namespace distance_origin_to_point_l13_1394

/-- The distance from the origin (0, 0) to the point (12, -5) in a rectangular coordinate system is 13 units. -/
theorem distance_origin_to_point :
  Real.sqrt (12^2 + (-5)^2) = 13 := by sorry

end distance_origin_to_point_l13_1394


namespace race_speed_ratio_l13_1333

/-- Given two runners a and b, where a's speed is some multiple of b's speed,
    and a gives b a 0.05 part of the race length as a head start to finish at the same time,
    prove that the ratio of a's speed to b's speed is 1/0.95 -/
theorem race_speed_ratio (v_a v_b : ℝ) (h1 : v_a > 0) (h2 : v_b > 0) 
    (h3 : ∃ k : ℝ, v_a = k * v_b) 
    (h4 : ∀ L : ℝ, L > 0 → L / v_a = (L - 0.05 * L) / v_b) : 
  v_a / v_b = 1 / 0.95 := by
sorry

end race_speed_ratio_l13_1333


namespace eighteen_letter_arrangements_l13_1335

theorem eighteen_letter_arrangements :
  let n : ℕ := 6
  let total_letters : ℕ := 3 * n
  let arrangement_count : ℕ := (Finset.range (n + 1)).sum (fun k => (Nat.choose n k)^3)
  ∀ (arrangements : Finset (Fin total_letters → Fin 3)),
    (∀ i : Fin total_letters, 
      (arrangements.card = arrangement_count) ∧
      (arrangements.card = (Finset.filter (fun arr => 
        (∀ j : Fin n, arr (j) ≠ 0) ∧
        (∀ j : Fin n, arr (j + n) ≠ 1) ∧
        (∀ j : Fin n, arr (j + 2*n) ≠ 2) ∧
        (arrangements.filter (fun arr => arr i = 0)).card = n ∧
        (arrangements.filter (fun arr => arr i = 1)).card = n ∧
        (arrangements.filter (fun arr => arr i = 2)).card = n
      ) arrangements).card)) := by
  sorry

#check eighteen_letter_arrangements

end eighteen_letter_arrangements_l13_1335


namespace prism_with_27_edges_has_11_faces_l13_1369

/-- A prism is a polyhedron with two congruent and parallel faces (called bases) 
    and whose other faces (called lateral faces) are parallelograms. -/
structure Prism where
  edges : ℕ
  lateral_faces : ℕ
  base_edges : ℕ

/-- The number of edges in a prism is equal to 3 times the number of lateral faces. -/
axiom prism_edge_count (p : Prism) : p.edges = 3 * p.lateral_faces

/-- The number of edges in each base of a prism is equal to the number of lateral faces. -/
axiom prism_base_edge_count (p : Prism) : p.base_edges = p.lateral_faces

/-- The total number of faces in a prism is equal to the number of lateral faces plus 2 (for the bases). -/
def total_faces (p : Prism) : ℕ := p.lateral_faces + 2

/-- Theorem: A prism with 27 edges has 11 faces. -/
theorem prism_with_27_edges_has_11_faces (p : Prism) (h : p.edges = 27) : total_faces p = 11 := by
  sorry


end prism_with_27_edges_has_11_faces_l13_1369


namespace symmetric_sequence_second_term_l13_1375

def is_symmetric (s : Fin 21 → ℕ) : Prop :=
  ∀ i : Fin 21, s i = s (20 - i)

def is_arithmetic_sequence (s : Fin 11 → ℕ) (a d : ℕ) : Prop :=
  ∀ i : Fin 11, s i = a + i * d

theorem symmetric_sequence_second_term 
  (c : Fin 21 → ℕ) 
  (h_sym : is_symmetric c) 
  (h_arith : is_arithmetic_sequence (fun i => c (i + 10)) 1 2) : 
  c 1 = 19 := by
  sorry

end symmetric_sequence_second_term_l13_1375


namespace inequality_relation_l13_1384

theorem inequality_relation (x y : ℝ) : 2*x - 5 < 2*y - 5 → x < y := by
  sorry

end inequality_relation_l13_1384


namespace total_turnips_l13_1327

theorem total_turnips (keith_turnips alyssa_turnips : ℕ) 
  (h1 : keith_turnips = 6) 
  (h2 : alyssa_turnips = 9) : 
  keith_turnips + alyssa_turnips = 15 := by
  sorry

end total_turnips_l13_1327


namespace max_product_for_maximized_fraction_l13_1367

def Digits := Fin 8

def validDigit (d : Digits) : ℕ := d.val + 2

theorem max_product_for_maximized_fraction :
  ∃ (A B C D : Digits),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (∀ (A' B' C' D' : Digits),
      A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ B' ≠ C' ∧ B' ≠ D' ∧ C' ≠ D' →
      (validDigit A' * validDigit B') / (validDigit C' * validDigit D' : ℚ) ≤
      (validDigit A * validDigit B) / (validDigit C * validDigit D : ℚ)) ∧
    validDigit A * validDigit B = 72 :=
by sorry

end max_product_for_maximized_fraction_l13_1367


namespace max_digits_product_5_4_l13_1323

theorem max_digits_product_5_4 : 
  ∀ (a b : ℕ), 
  10000 ≤ a ∧ a ≤ 99999 → 
  1000 ≤ b ∧ b ≤ 9999 → 
  a * b < 1000000000 := by
sorry

end max_digits_product_5_4_l13_1323


namespace sophie_chocolates_l13_1345

theorem sophie_chocolates :
  ∃ (x : ℕ), x ≥ 150 ∧ x % 15 = 7 ∧ ∀ (y : ℕ), y ≥ 150 ∧ y % 15 = 7 → y ≥ x :=
by sorry

end sophie_chocolates_l13_1345


namespace canoe_kayak_difference_l13_1386

/-- Represents the daily rental business for canoes and kayaks. -/
structure RentalBusiness where
  canoe_price : ℕ
  kayak_price : ℕ
  canoe_count : ℕ
  kayak_count : ℕ

/-- The conditions of the rental business problem. -/
def rental_problem : RentalBusiness where
  canoe_price := 9
  kayak_price := 12
  canoe_count := 24  -- We know this from the solution, but it's derived from the conditions
  kayak_count := 18  -- We know this from the solution, but it's derived from the conditions

/-- The theorem stating the difference between canoes and kayaks rented. -/
theorem canoe_kayak_difference (b : RentalBusiness) 
  (h1 : b.canoe_price = 9)
  (h2 : b.kayak_price = 12)
  (h3 : 4 * b.kayak_count = 3 * b.canoe_count)
  (h4 : b.canoe_price * b.canoe_count + b.kayak_price * b.kayak_count = 432) :
  b.canoe_count - b.kayak_count = 6 := by
  sorry

#eval rental_problem.canoe_count - rental_problem.kayak_count

end canoe_kayak_difference_l13_1386


namespace crow_percentage_among_non_pigeons_l13_1307

theorem crow_percentage_among_non_pigeons (total_birds : ℝ) (crow_percentage : ℝ) (pigeon_percentage : ℝ)
  (h1 : crow_percentage = 40)
  (h2 : pigeon_percentage = 20)
  (h3 : 0 < total_birds) :
  (crow_percentage / (100 - pigeon_percentage)) * 100 = 50 := by
sorry

end crow_percentage_among_non_pigeons_l13_1307


namespace ellipse_problem_l13_1368

/-- The ellipse problem -/
theorem ellipse_problem (a b : ℝ) (P : ℝ × ℝ) :
  a > 0 ∧ b > 0 ∧ a > b →  -- a and b are positive real numbers with a > b
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) →  -- P is on the ellipse
  ((P.1 + 1)^2 + P.2^2)^(1/2) - ((P.1 - 1)^2 + P.2^2)^(1/2) = a / 2 →  -- |PF₁| - |PF₂| = a/2
  (P.1 - 1) * (P.1 + 1) + P.2^2 = 0 →  -- PF₂ is perpendicular to F₁F₂
  (∃ (m : ℝ), (1 + m * P.2)^2 / 4 + P.2^2 / 3 = 1 ∧  -- equation of ellipse G
              (∃ (M N : ℝ × ℝ), M ≠ N ∧  -- M and N are distinct points
                (M.1^2 / 4 + M.2^2 / 3 = 1) ∧ (N.1^2 / 4 + N.2^2 / 3 = 1) ∧  -- M and N are on the ellipse
                (M.1 - 1 = m * M.2) ∧ (N.1 - 1 = m * N.2) ∧  -- M and N are on line l passing through F₂
                ((0 - M.2) * (N.1 - 1)) / ((0 - N.2) * (M.1 - 1)) = 2))  -- ratio of areas of triangles BF₂M and BF₂N is 2
  := by sorry


end ellipse_problem_l13_1368


namespace geometric_sequence_general_term_l13_1338

/-- Geometric sequence with given properties -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  (a 1 = 3/2) ∧
  (∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q) ∧
  (a 1 + a 2 + a 3 = 9/2)

/-- The general term of the geometric sequence -/
theorem geometric_sequence_general_term (a : ℕ → ℚ) (h : geometric_sequence a) :
  (∀ n : ℕ, a n = 3/2 * (-2)^(n - 1)) ∨ (∀ n : ℕ, a n = 3/2) := by
  sorry

end geometric_sequence_general_term_l13_1338


namespace equation_one_solution_l13_1380

theorem equation_one_solution :
  ∃ x₁ x₂ : ℝ, (3 * x₁^2 - 9 = 0) ∧ (3 * x₂^2 - 9 = 0) ∧ (x₁ = Real.sqrt 3) ∧ (x₂ = -Real.sqrt 3) :=
by sorry

end equation_one_solution_l13_1380


namespace smallest_sum_of_reciprocals_l13_1332

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 → (x + y : ℕ) ≤ (a + b : ℕ)) → 
  (x + y : ℕ) = 50 :=
by sorry

end smallest_sum_of_reciprocals_l13_1332


namespace ellipse_m_values_l13_1349

/-- Definition of the ellipse equation -/
def is_ellipse (x y m : ℝ) : Prop :=
  x^2 / (10 - m) + y^2 / (m - 2) = 1

/-- Definition of focal length -/
def focal_length (m : ℝ) : ℝ := 4

/-- Theorem stating the possible values of m -/
theorem ellipse_m_values :
  ∀ m : ℝ, (∃ x y : ℝ, is_ellipse x y m) → focal_length m = 4 → m = 4 ∨ m = 8 :=
by sorry

end ellipse_m_values_l13_1349


namespace stuffed_animals_problem_l13_1317

theorem stuffed_animals_problem (M : ℕ) : 
  (M + 2*M + (2*M + 5) = 175) → M = 34 := by
  sorry

end stuffed_animals_problem_l13_1317


namespace sphere_radii_ratio_l13_1314

/-- The ratio of radii of two spheres given their volumes -/
theorem sphere_radii_ratio (V_large V_small : ℝ) (h1 : V_large = 432 * Real.pi) 
  (h2 : V_small = 0.275 * V_large) : 
  (V_small / V_large)^(1/3 : ℝ) = 2/3 := by
  sorry

end sphere_radii_ratio_l13_1314


namespace solution_set_transformation_l13_1393

theorem solution_set_transformation (a b c : ℝ) :
  (∀ x, ax^2 + b*x + c < 0 ↔ x < -2 ∨ x > 1/3) →
  (∀ x, c*x^2 - b*x + a ≥ 0 ↔ x ≤ -3 ∨ x ≥ 1/2) :=
by sorry

end solution_set_transformation_l13_1393


namespace min_value_reciprocal_sum_min_value_reciprocal_sum_equality_l13_1360

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 4 / y ≥ 9 :=
sorry

theorem min_value_reciprocal_sum_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ x y, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1 / x + 4 / y = 9 :=
sorry

end min_value_reciprocal_sum_min_value_reciprocal_sum_equality_l13_1360


namespace other_diagonal_length_l13_1311

/-- Represents a rhombus with given properties -/
structure Rhombus where
  d1 : ℝ  -- Length of one diagonal
  d2 : ℝ  -- Length of the other diagonal
  area : ℝ -- Area of the rhombus

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.d1 * r.d2) / 2

/-- Given a rhombus with one diagonal of 16 cm and an area of 88 cm², 
    the length of the other diagonal is 11 cm -/
theorem other_diagonal_length (r : Rhombus) 
    (h1 : r.d2 = 16) 
    (h2 : r.area = 88) : 
    r.d1 = 11 := by
  sorry


end other_diagonal_length_l13_1311


namespace average_value_of_sequence_l13_1304

theorem average_value_of_sequence (z : ℝ) : 
  (0 + 3*z + 6*z + 12*z + 24*z) / 5 = 9*z := by sorry

end average_value_of_sequence_l13_1304


namespace inequality_solution_set_l13_1391

def solution_set : Set ℝ := {x | x < -1 ∨ (-1 < x ∧ x < 0) ∨ x > 2}

def inequality (x : ℝ) : Prop := x^2 * (x^2 + 2*x + 1) > 2*x * (x^2 + 2*x + 1)

theorem inequality_solution_set : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
sorry

end inequality_solution_set_l13_1391


namespace xy_equals_ten_l13_1331

theorem xy_equals_ten (x y : ℝ) (h : x * (x + y) = x^2 + 10) : x * y = 10 := by
  sorry

end xy_equals_ten_l13_1331


namespace staircase_ratio_proof_l13_1302

theorem staircase_ratio_proof (steps_first : ℕ) (step_height : ℚ) (total_height : ℚ) 
  (h1 : steps_first = 20)
  (h2 : step_height = 1/2)
  (h3 : total_height = 45) :
  ∃ (r : ℚ), 
    r * steps_first = (total_height / step_height - steps_first - (r * steps_first - 10)) ∧ 
    r = 2 := by
  sorry

end staircase_ratio_proof_l13_1302


namespace consecutive_sum_product_l13_1346

theorem consecutive_sum_product (n : ℕ) (h : n > 100) :
  ∃ (a b c : ℕ), (a > 1 ∧ b > 1 ∧ c > 1) ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
  ((n + (n + 1) + (n + 2) = a * b * c) ∨
   ((n + 1) + (n + 2) + (n + 3) = a * b * c)) := by
sorry


end consecutive_sum_product_l13_1346


namespace evaluate_expression_l13_1362

theorem evaluate_expression (x y : ℝ) (hx : x = -1) (hy : y = 2) : y^2 * (y - 2*x) = 16 := by
  sorry

end evaluate_expression_l13_1362


namespace average_monthly_sales_l13_1324

def monthly_sales : List ℝ := [80, 100, 75, 95, 110, 180, 90, 115, 130, 200, 160, 140]

theorem average_monthly_sales :
  (monthly_sales.sum / monthly_sales.length : ℝ) = 122.92 := by
  sorry

end average_monthly_sales_l13_1324


namespace tailor_trim_problem_l13_1301

/-- Given a square cloth with side length 18 feet, if 4 feet are trimmed from two opposite edges
    and x feet are trimmed from the other two edges, resulting in 120 square feet of remaining cloth,
    then x = 6. -/
theorem tailor_trim_problem (x : ℝ) : 
  (18 : ℝ) > 0 ∧ x > 0 ∧ (18 - 4 - 4 : ℝ) * (18 - x) = 120 → x = 6 := by
  sorry

end tailor_trim_problem_l13_1301


namespace five_letter_word_count_l13_1370

/-- The number of letters in the alphabet -/
def alphabet_size : Nat := 26

/-- The number of vowels -/
def vowel_count : Nat := 5

/-- The number of five-letter words that begin and end with the same letter, 
    with the second letter always being a vowel -/
def word_count : Nat := alphabet_size * vowel_count * alphabet_size * alphabet_size

theorem five_letter_word_count : word_count = 87700 := by
  sorry

end five_letter_word_count_l13_1370


namespace spinster_cat_problem_l13_1396

theorem spinster_cat_problem (S C : ℕ) : 
  S * 9 = C * 2 →  -- Ratio of spinsters to cats is 2:9
  C = S + 42 →     -- There are 42 more cats than spinsters
  S = 12           -- The number of spinsters is 12
:= by sorry

end spinster_cat_problem_l13_1396


namespace find_set_A_l13_1357

def U : Set ℕ := {1,2,3,4,5,6,7,8}

theorem find_set_A (A B : Set ℕ)
  (h1 : A ∩ (U \ B) = {1,8})
  (h2 : (U \ A) ∩ B = {2,6})
  (h3 : (U \ A) ∩ (U \ B) = {4,7}) :
  A = {1,3,5,8} := by
  sorry

end find_set_A_l13_1357


namespace bobs_roommates_l13_1321

theorem bobs_roommates (john_roommates : ℕ) (h1 : john_roommates = 25) :
  ∃ (bob_roommates : ℕ), john_roommates = 2 * bob_roommates + 5 → bob_roommates = 10 := by
  sorry

end bobs_roommates_l13_1321


namespace product_over_sum_equals_180_l13_1309

theorem product_over_sum_equals_180 : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7) / (1 + 2 + 3 + 4 + 5 + 6 + 7) = 180 := by
  sorry

end product_over_sum_equals_180_l13_1309


namespace craftsman_production_theorem_l13_1318

/-- The number of parts manufactured by a master craftsman during a shift -/
def parts_manufactured : ℕ → ℕ → ℕ → ℕ
  | initial_rate, rate_increase, additional_parts =>
    initial_rate + additional_parts

/-- The time needed to manufacture parts at a given rate -/
def time_needed : ℕ → ℕ → ℚ
  | parts, rate => (parts : ℚ) / (rate : ℚ)

theorem craftsman_production_theorem 
  (initial_rate : ℕ) 
  (rate_increase : ℕ) 
  (additional_parts : ℕ) :
  initial_rate = 35 →
  rate_increase = 15 →
  time_needed additional_parts initial_rate - 
    time_needed additional_parts (initial_rate + rate_increase) = (3 : ℚ) / 2 →
  parts_manufactured initial_rate rate_increase additional_parts = 210 :=
by sorry

end craftsman_production_theorem_l13_1318


namespace flower_bunch_count_l13_1343

theorem flower_bunch_count (total_flowers : ℕ) (flowers_per_bunch : ℕ) (bunches : ℕ) : 
  total_flowers = 12 * 6 →
  flowers_per_bunch = 9 →
  bunches = total_flowers / flowers_per_bunch →
  bunches = 8 := by
sorry

end flower_bunch_count_l13_1343


namespace college_entrance_exam_score_l13_1330

theorem college_entrance_exam_score (total_questions unanswered_questions answered_questions correct_answers incorrect_answers : ℕ)
  (raw_score : ℚ) :
  total_questions = 85 →
  unanswered_questions = 3 →
  answered_questions = 82 →
  answered_questions = correct_answers + incorrect_answers →
  raw_score = 67 →
  raw_score = correct_answers - 0.25 * incorrect_answers →
  correct_answers = 70 := by
sorry

end college_entrance_exam_score_l13_1330


namespace smallest_add_to_multiple_of_five_l13_1372

theorem smallest_add_to_multiple_of_five : ∃ (n : ℕ), n > 0 ∧ (729 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (729 + m) % 5 = 0 → n ≤ m :=
by sorry

end smallest_add_to_multiple_of_five_l13_1372


namespace complex_power_magnitude_l13_1371

theorem complex_power_magnitude : 
  Complex.abs ((2/5 : ℂ) + (7/5 : ℂ) * Complex.I) ^ 8 = 7890481/390625 := by
  sorry

end complex_power_magnitude_l13_1371


namespace addition_to_reach_81_l13_1379

theorem addition_to_reach_81 : 5 * 12 / (180 / 3) + 80 = 81 := by
  sorry

end addition_to_reach_81_l13_1379


namespace new_average_after_exclusion_l13_1310

/-- Theorem: New average after excluding students with low marks -/
theorem new_average_after_exclusion
  (total_students : ℕ)
  (original_average : ℚ)
  (excluded_students : ℕ)
  (excluded_average : ℚ)
  (h1 : total_students = 33)
  (h2 : original_average = 90)
  (h3 : excluded_students = 3)
  (h4 : excluded_average = 40) :
  let remaining_students := total_students - excluded_students
  let total_marks := total_students * original_average
  let excluded_marks := excluded_students * excluded_average
  let remaining_marks := total_marks - excluded_marks
  (remaining_marks / remaining_students : ℚ) = 95 := by
sorry

end new_average_after_exclusion_l13_1310


namespace hyperbola_line_intersection_l13_1315

theorem hyperbola_line_intersection (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  ∃ P Q : ℝ × ℝ,
    (P.1^2 / a - P.2^2 / b = 1) ∧
    (Q.1^2 / a - Q.2^2 / b = 1) ∧
    (P.1 + P.2 = 1) ∧
    (Q.1 + Q.2 = 1) ∧
    (P.1 * Q.1 + P.2 * Q.2 = 0) →
    1 / a - 1 / b = 2 := by
  sorry

end hyperbola_line_intersection_l13_1315


namespace ice_pop_probability_l13_1359

def total_ice_pops : ℕ := 17
def cherry_ice_pops : ℕ := 5
def children : ℕ := 5

theorem ice_pop_probability :
  1 - (Nat.factorial cherry_ice_pops : ℚ) / (Nat.factorial total_ice_pops / Nat.factorial (total_ice_pops - children)) = 1 - 1 / 4762 := by
  sorry

end ice_pop_probability_l13_1359


namespace clothing_company_wage_promise_l13_1305

/-- Represents the wage calculation and constraints for skilled workers in a clothing company. -/
theorem clothing_company_wage_promise (base_salary : ℝ) (wage_a : ℝ) (wage_b : ℝ) 
  (hours_per_day : ℝ) (days_per_month : ℝ) (time_a : ℝ) (time_b : ℝ) :
  base_salary = 800 →
  wage_a = 16 →
  wage_b = 12 →
  hours_per_day = 8 →
  days_per_month = 25 →
  time_a = 2 →
  time_b = 1 →
  ∀ a : ℝ, 
    a ≥ (hours_per_day * days_per_month - 2 * a) / 2 →
    a ≥ 0 →
    a ≤ hours_per_day * days_per_month / (2 * time_a) →
    base_salary + wage_a * a + wage_b * (hours_per_day * days_per_month / time_b - 2 * a / time_b) < 3000 :=
by sorry

end clothing_company_wage_promise_l13_1305


namespace quadratic_inequality_l13_1353

def f (x : ℝ) := x^2 - 2*x - 3

theorem quadratic_inequality (y₁ y₂ y₃ : ℝ) 
  (h₁ : f (-3) = y₁) 
  (h₂ : f (-2) = y₂) 
  (h₃ : f 2 = y₃) : 
  y₃ < y₂ ∧ y₂ < y₁ := by
  sorry

end quadratic_inequality_l13_1353


namespace parabola_above_line_l13_1342

/-- Given non-zero real numbers a, b, and c, if the parabola y = ax^2 + bx + c is positioned
    above the line y = cx, then the parabola y = cx^2 - bx + a is positioned above
    the line y = cx - b. -/
theorem parabola_above_line (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_above : ∀ x, a * x^2 + b * x + c > c * x) :
  ∀ x, c * x^2 - b * x + a > c * x - b :=
by sorry

end parabola_above_line_l13_1342


namespace expression_simplification_l13_1319

theorem expression_simplification (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) :
  (a - 3*a/(a+1)) / ((a^2 - 4*a + 4)/(a+1)) = a / (a-2) := by
  sorry

end expression_simplification_l13_1319


namespace lindas_hourly_rate_l13_1306

/-- Proves that Linda's hourly rate for babysitting is $10.00 -/
theorem lindas_hourly_rate (application_fee : ℝ) (num_colleges : ℕ) (hours_worked : ℝ) :
  application_fee = 25 →
  num_colleges = 6 →
  hours_worked = 15 →
  (application_fee * num_colleges) / hours_worked = 10 := by
  sorry

end lindas_hourly_rate_l13_1306


namespace average_roots_quadratic_l13_1334

theorem average_roots_quadratic (x₁ x₂ : ℝ) : 
  (3 * x₁^2 + 4 * x₁ - 5 = 0) → 
  (3 * x₂^2 + 4 * x₂ - 5 = 0) → 
  x₁ ≠ x₂ → 
  (x₁ + x₂) / 2 = -2/3 := by
sorry

end average_roots_quadratic_l13_1334


namespace tiffany_bags_total_l13_1344

theorem tiffany_bags_total (monday_bags : ℕ) (next_day_bags : ℕ) 
  (h1 : monday_bags = 4) 
  (h2 : next_day_bags = 8) : 
  monday_bags + next_day_bags = 12 := by
  sorry

end tiffany_bags_total_l13_1344


namespace minute_hand_angle_2h40m_l13_1395

/-- The angle turned by the minute hand when the hour hand moves for a given time -/
def minute_hand_angle (hours : ℝ) (minutes : ℝ) : ℝ :=
  -(hours * 360 + minutes * 6)

/-- Theorem: When the hour hand moves for 2 hours and 40 minutes, 
    the angle turned by the minute hand is -960° -/
theorem minute_hand_angle_2h40m :
  minute_hand_angle 2 40 = -960 := by sorry

end minute_hand_angle_2h40m_l13_1395


namespace teacherStudentArrangements_eq_144_l13_1398

/-- The number of ways to arrange 2 teachers and 4 students in a row
    with exactly 2 students between the teachers -/
def teacherStudentArrangements : ℕ :=
  3 * 2 * 24

/-- Proof that the number of arrangements is 144 -/
theorem teacherStudentArrangements_eq_144 :
  teacherStudentArrangements = 144 := by
  sorry

end teacherStudentArrangements_eq_144_l13_1398


namespace train_length_calculation_l13_1308

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : Real) (crossing_time : Real) (bridge_length : Real) :
  train_speed = 54 * (1000 / 3600) →
  crossing_time = 16.13204276991174 →
  bridge_length = 132 →
  train_speed * crossing_time - bridge_length = 109.9806415486761 := by
  sorry

end train_length_calculation_l13_1308


namespace cyclic_sum_inequality_l13_1316

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a + b)^2 + (b + c)^2 + (c + a)^2 = 2*(a + b + c) + 6*a*b*c) :
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ |2*(a + b + c) - 6*a*b*c| := by
  sorry

end cyclic_sum_inequality_l13_1316


namespace min_value_x_plus_3y_l13_1325

theorem min_value_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 3/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 3/b = 1 → x + 3*y ≤ a + 3*b ∧ ∃ c d : ℝ, c > 0 ∧ d > 0 ∧ 1/c + 3/d = 1 ∧ c + 3*d = 16 :=
sorry

end min_value_x_plus_3y_l13_1325


namespace max_value_fraction_sum_l13_1389

theorem max_value_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (2 * x + y)) + (y / (x + 2 * y)) ≤ 2 / 3 ∧
  ∃ x y, x > 0 ∧ y > 0 ∧ (x / (2 * x + y)) + (y / (x + 2 * y)) = 2 / 3 :=
by sorry

end max_value_fraction_sum_l13_1389


namespace stratified_sample_correct_l13_1300

/-- Represents the number of students in each category -/
structure StudentPopulation where
  total : ℕ
  junior : ℕ
  undergraduate : ℕ
  graduate : ℕ

/-- Represents the sample size and the number of students to be drawn from each category -/
structure SampleSize where
  total : ℕ
  junior : ℕ
  undergraduate : ℕ
  graduate : ℕ

/-- Calculates the correct sample size for stratified sampling -/
def calculateStratifiedSample (pop : StudentPopulation) (sampleTotal : ℕ) : SampleSize :=
  { total := sampleTotal,
    junior := (sampleTotal * pop.junior) / pop.total,
    undergraduate := (sampleTotal * pop.undergraduate) / pop.total,
    graduate := (sampleTotal * pop.graduate) / pop.total }

/-- Theorem: The calculated stratified sample is correct for the given population -/
theorem stratified_sample_correct (pop : StudentPopulation) (sample : SampleSize) :
  pop.total = 5400 ∧ 
  pop.junior = 1500 ∧ 
  pop.undergraduate = 3000 ∧ 
  pop.graduate = 900 ∧
  sample.total = 180 →
  calculateStratifiedSample pop sample.total = 
    { total := 180, junior := 50, undergraduate := 100, graduate := 30 } := by
  sorry


end stratified_sample_correct_l13_1300


namespace h_of_3_eq_3_l13_1352

/-- The function h(x) is defined implicitly by this equation -/
def h_equation (x : ℝ) (h : ℝ → ℝ) : Prop :=
  (x^(2^2007 - 1) - 1) * h x = (x + 1) * (x^2 + 1) * (x^4 + 1) * (x^(2^2006) + 1) - 1

/-- The theorem states that h(3) = 3 for the function h defined by h_equation -/
theorem h_of_3_eq_3 :
  ∃ h : ℝ → ℝ, h_equation 3 h ∧ h 3 = 3 := by sorry

end h_of_3_eq_3_l13_1352


namespace average_weight_of_class_class_average_weight_l13_1328

theorem average_weight_of_class (group1_count : ℕ) (group1_avg : ℚ) 
                                (group2_count : ℕ) (group2_avg : ℚ) : ℚ :=
  let total_count := group1_count + group2_count
  let total_weight := group1_count * group1_avg + group2_count * group2_avg
  total_weight / total_count

theorem class_average_weight :
  average_weight_of_class 24 (50.25 : ℚ) 8 (45.15 : ℚ) = 49 := by
  sorry

end average_weight_of_class_class_average_weight_l13_1328


namespace complex_vector_sum_l13_1363

theorem complex_vector_sum (z₁ z₂ z₃ : ℂ) (x y : ℝ) 
  (h₁ : z₁ = -1 + I)
  (h₂ : z₂ = 1 + I)
  (h₃ : z₃ = 1 + 4*I)
  (h₄ : z₃ = x • z₁ + y • z₂) :
  x + y = 4 := by sorry

end complex_vector_sum_l13_1363


namespace equations_not_equivalent_l13_1320

/-- The solution set of the equation 2√(x+5) = x+2 -/
def SolutionSet1 : Set ℝ :=
  {x : ℝ | 2 * Real.sqrt (x + 5) = x + 2}

/-- The solution set of the equation 4(x+5) = (x+2)² -/
def SolutionSet2 : Set ℝ :=
  {x : ℝ | 4 * (x + 5) = (x + 2)^2}

/-- Theorem stating that the equations are not equivalent -/
theorem equations_not_equivalent : SolutionSet1 ≠ SolutionSet2 := by
  sorry

#check equations_not_equivalent

end equations_not_equivalent_l13_1320


namespace smallest_multiple_square_l13_1347

theorem smallest_multiple_square (a : ℕ) : 
  (∃ k : ℕ, a = 6 * k) ∧ 
  (∃ m : ℕ, a = 15 * m) ∧ 
  (∃ n : ℕ, a = n * n) ∧ 
  (∀ b : ℕ, b > 0 ∧ 
    (∃ k : ℕ, b = 6 * k) ∧ 
    (∃ m : ℕ, b = 15 * m) ∧ 
    (∃ n : ℕ, b = n * n) → 
    a ≤ b) → 
  a = 900 := by
sorry

end smallest_multiple_square_l13_1347


namespace sqrt_product_equality_l13_1385

theorem sqrt_product_equality : Real.sqrt 6 * Real.sqrt 2 = 2 * Real.sqrt 3 := by
  sorry

end sqrt_product_equality_l13_1385


namespace extra_bananas_per_child_l13_1366

/-- Given the total number of children, number of absent children, and original banana allocation,
    calculate the number of extra bananas each present child received. -/
theorem extra_bananas_per_child 
  (total_children : ℕ) 
  (absent_children : ℕ) 
  (original_allocation : ℕ) 
  (h1 : total_children = 780)
  (h2 : absent_children = 390)
  (h3 : original_allocation = 2)
  (h4 : absent_children < total_children) :
  (total_children * original_allocation) / (total_children - absent_children) - original_allocation = 2 :=
by sorry

end extra_bananas_per_child_l13_1366


namespace village_birth_probability_l13_1351

/-- Represents the gender of a child -/
inductive Gender
| Boy
| Girl

/-- A village with a custom of having children until a boy is born -/
structure Village where
  /-- The probability of having a boy in a single birth -/
  prob_boy : ℝ
  /-- The probability of having a girl in a single birth -/
  prob_girl : ℝ
  /-- The proportion of boys to girls in the village after some time -/
  boy_girl_ratio : ℝ
  /-- The probabilities sum to 1 -/
  prob_sum_one : prob_boy + prob_girl = 1
  /-- The proportion of boys to girls is 1:1 -/
  equal_ratio : boy_girl_ratio = 1

/-- Theorem: In a village with the given custom, the probability of having a boy or a girl is 1/2 -/
theorem village_birth_probability (v : Village) : v.prob_boy = 1/2 ∧ v.prob_girl = 1/2 := by
  sorry


end village_birth_probability_l13_1351


namespace square_of_real_number_proposition_l13_1356

theorem square_of_real_number_proposition :
  ∃ (p q : Prop), (∀ x : ℝ, x^2 > 0 ∨ x^2 = 0) ↔ (p ∨ q) :=
by sorry

end square_of_real_number_proposition_l13_1356


namespace simplify_and_sum_exponents_l13_1358

theorem simplify_and_sum_exponents (a b d : ℝ) : 
  ∃ (k : ℝ), (54 * a^5 * b^9 * d^14)^(1/3) = 3 * a * b^3 * d^4 * k ∧ 1 + 3 + 4 = 8 := by
  sorry

end simplify_and_sum_exponents_l13_1358


namespace tom_has_nine_balloons_l13_1382

/-- The number of yellow balloons Sara has -/
def sara_balloons : ℕ := 8

/-- The total number of yellow balloons Tom and Sara have together -/
def total_balloons : ℕ := 17

/-- The number of yellow balloons Tom has -/
def tom_balloons : ℕ := total_balloons - sara_balloons

theorem tom_has_nine_balloons : tom_balloons = 9 := by
  sorry

end tom_has_nine_balloons_l13_1382


namespace jeff_scores_mean_l13_1378

def jeff_scores : List ℝ := [89, 92, 88, 95, 91]

theorem jeff_scores_mean : (jeff_scores.sum / jeff_scores.length) = 91 := by
  sorry

end jeff_scores_mean_l13_1378


namespace range_of_a_inequality_l13_1303

theorem range_of_a_inequality (a : ℝ) : 
  (∃ x : ℝ, |a| ≥ |x + 1| + |x - 2|) ↔ a ∈ Set.Iic 3 := by
  sorry

end range_of_a_inequality_l13_1303


namespace salt_proportion_is_one_twenty_first_l13_1376

/-- The proportion of salt in a saltwater solution -/
def salt_proportion (salt_mass : ℚ) (water_mass : ℚ) : ℚ :=
  salt_mass / (salt_mass + water_mass)

/-- Proof that the proportion of salt in the given saltwater solution is 1/21 -/
theorem salt_proportion_is_one_twenty_first :
  let salt_mass : ℚ := 50
  let water_mass : ℚ := 1000
  salt_proportion salt_mass water_mass = 1 / 21 := by
  sorry

end salt_proportion_is_one_twenty_first_l13_1376


namespace sum_of_quotient_dividend_divisor_l13_1326

theorem sum_of_quotient_dividend_divisor : 
  ∀ (N D : ℕ), 
  N = 50 → 
  D = 5 → 
  N + D + (N / D) = 65 :=
by
  sorry

end sum_of_quotient_dividend_divisor_l13_1326


namespace smallest_four_digit_number_l13_1373

def is_valid_equation (a b c : ℕ) : Prop :=
  a + b = c ∧ 
  a ≥ 1000 ∧ a < 10000 ∧
  b ≥ 10 ∧ b < 100 ∧
  c ≥ 1000 ∧ c < 10000

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  (∀ i j, i ≠ j → digits.nthLe i sorry ≠ digits.nthLe j sorry) ∧
  digits.length ≤ 10

theorem smallest_four_digit_number (a b c : ℕ) :
  is_valid_equation a b c →
  has_distinct_digits a →
  has_distinct_digits b →
  has_distinct_digits c →
  (∀ x, has_distinct_digits x → x ≥ 1000 → x < c → False) →
  c = 2034 :=
sorry

end smallest_four_digit_number_l13_1373


namespace f_derivative_at_zero_l13_1355

def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

theorem f_derivative_at_zero : 
  deriv f 0 = -120 := by sorry

end f_derivative_at_zero_l13_1355


namespace scientific_notation_570_million_l13_1339

theorem scientific_notation_570_million :
  (570000000 : ℝ) = 5.7 * (10 : ℝ)^8 := by sorry

end scientific_notation_570_million_l13_1339


namespace min_max_sum_l13_1387

theorem min_max_sum (x₁ x₂ x₃ x₄ x₅ : ℝ) (h_nonneg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧ x₅ ≥ 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 300) : 
  (max (x₁ + x₂) (max (x₂ + x₃) (max (x₃ + x₄) (x₄ + x₅)))) ≥ 100 ∧ 
  ∃ (y₁ y₂ y₃ y₄ y₅ : ℝ), y₁ ≥ 0 ∧ y₂ ≥ 0 ∧ y₃ ≥ 0 ∧ y₄ ≥ 0 ∧ y₅ ≥ 0 ∧ 
  y₁ + y₂ + y₃ + y₄ + y₅ = 300 ∧ 
  max (y₁ + y₂) (max (y₂ + y₃) (max (y₃ + y₄) (y₄ + y₅))) = 100 :=
by sorry

end min_max_sum_l13_1387
