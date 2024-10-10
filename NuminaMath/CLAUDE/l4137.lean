import Mathlib

namespace work_completion_time_l4137_413789

theorem work_completion_time (b_days : ℝ) (a_wage_ratio : ℝ) (a_days : ℝ) : 
  b_days = 15 →
  a_wage_ratio = 3/5 →
  a_wage_ratio = (1/a_days) / (1/a_days + 1/b_days) →
  a_days = 10 := by
sorry

end work_completion_time_l4137_413789


namespace min_red_vertices_l4137_413775

/-- Given a square partitioned into n^2 unit squares, each divided into two triangles,
    the minimum number of red vertices needed to ensure each triangle has a red vertex is ⌈n^2/2⌉ -/
theorem min_red_vertices (n : ℕ) (h : n > 0) :
  ∃ (red_vertices : Finset (ℕ × ℕ)),
    (∀ i j : ℕ, i < n → j < n →
      (∃ k l : ℕ, (k = i ∨ k = i + 1) ∧ (l = j ∨ l = j + 1) ∧ (k, l) ∈ red_vertices)) ∧
    red_vertices.card = ⌈(n^2 : ℝ) / 2⌉ ∧
    (∀ rv : Finset (ℕ × ℕ), 
      (∀ i j : ℕ, i < n → j < n →
        (∃ k l : ℕ, (k = i ∨ k = i + 1) ∧ (l = j ∨ l = j + 1) ∧ (k, l) ∈ rv)) →
      rv.card ≥ ⌈(n^2 : ℝ) / 2⌉) := by
  sorry


end min_red_vertices_l4137_413775


namespace five_fourths_of_sum_l4137_413708

theorem five_fourths_of_sum : ∀ (a b c d : ℚ),
  a = 6 / 3 ∧ b = 8 / 4 → (5 / 4) * (a + b) = 5 :=
by
  sorry

end five_fourths_of_sum_l4137_413708


namespace at_least_eight_empty_columns_at_least_eight_people_in_one_column_l4137_413710

/-- Represents the state of people on columns -/
structure ColumnState where
  num_people : Nat
  num_columns : Nat
  initial_column : Nat

/-- Proves that at least 8 columns are empty after any number of steps -/
theorem at_least_eight_empty_columns (state : ColumnState) 
  (h1 : state.num_people = 65)
  (h2 : state.num_columns = 17)
  (h3 : state.initial_column = 9) :
  ∀ (steps : Nat), ∃ (empty_columns : Nat), empty_columns ≥ 8 := by
  sorry

/-- Proves that there is always at least one column with at least 8 people -/
theorem at_least_eight_people_in_one_column (state : ColumnState) 
  (h1 : state.num_people = 65)
  (h2 : state.num_columns = 17)
  (h3 : state.initial_column = 9) :
  ∀ (steps : Nat), ∃ (column : Nat), ∃ (people_in_column : Nat), 
    people_in_column ≥ 8 ∧ column ≤ state.num_columns := by
  sorry

end at_least_eight_empty_columns_at_least_eight_people_in_one_column_l4137_413710


namespace sin_2alpha_in_terms_of_y_l4137_413782

theorem sin_2alpha_in_terms_of_y (α y : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : y > 0) 
  (h3 : Real.cos (α/2) = Real.sqrt ((y+1)/(2*y))) : 
  Real.sin (2*α) = (2 * Real.sqrt (y^2 - 1)) / y := by
sorry

end sin_2alpha_in_terms_of_y_l4137_413782


namespace eulers_formula_eulers_identity_complex_exp_sum_bound_l4137_413734

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := sorry

-- Define the imaginary unit
def i : ℂ := sorry

-- Define pi
noncomputable def π : ℝ := sorry

-- Theorem 1: Euler's formula
theorem eulers_formula (x : ℝ) : cexp (i * x) = Complex.cos x + i * Complex.sin x := by sorry

-- Theorem 2: Euler's identity
theorem eulers_identity : cexp (i * π) + 1 = 0 := by sorry

-- Theorem 3: Bound on sum of complex exponentials
theorem complex_exp_sum_bound (x : ℝ) : Complex.abs (cexp (i * x) + cexp (-i * x)) ≤ 2 := by sorry

end eulers_formula_eulers_identity_complex_exp_sum_bound_l4137_413734


namespace max_students_distribution_l4137_413702

def max_students (pens pencils : ℕ) : ℕ :=
  Nat.gcd pens pencils

theorem max_students_distribution (pens pencils : ℕ) :
  pens = 100 → pencils = 50 → max_students pens pencils = 50 := by
  sorry

end max_students_distribution_l4137_413702


namespace odd_sum_squared_plus_product_not_both_even_l4137_413716

theorem odd_sum_squared_plus_product_not_both_even (p q : ℤ) 
  (h : Odd (p^2 + q^2 + p*q)) : ¬(Even p ∧ Even q) := by
  sorry

end odd_sum_squared_plus_product_not_both_even_l4137_413716


namespace trapezoid_area_l4137_413737

/-- The area of a trapezoid with given vertices in a standard rectangular coordinate system -/
theorem trapezoid_area (E F G H : ℝ × ℝ) : 
  E = (2, -3) → 
  F = (2, 2) → 
  G = (7, 8) → 
  H = (7, 3) → 
  (1/2 : ℝ) * ((F.2 - E.2) + (G.2 - H.2)) * (G.1 - E.1) = 25 := by
  sorry

#check trapezoid_area

end trapezoid_area_l4137_413737


namespace rita_calculation_l4137_413784

theorem rita_calculation (a b c : ℝ) 
  (h1 : a - (2*b - 3*c) = 23) 
  (h2 : a - 2*b - 3*c = 5) : 
  a - 2*b = 14 := by
sorry

end rita_calculation_l4137_413784


namespace f_monotone_iff_a_range_f_lower_bound_l4137_413790

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - x^2 - a*x

theorem f_monotone_iff_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≤ 2 - 2 * Real.log 2 :=
sorry

theorem f_lower_bound (x : ℝ) (hx : x > 0) :
  f 1 x > 1 - (Real.log 2) / 2 - ((Real.log 2) / 2)^2 :=
sorry

end f_monotone_iff_a_range_f_lower_bound_l4137_413790


namespace dave_spent_102_l4137_413718

/-- The amount Dave spent on books -/
def dave_spent (animal_books outer_space_books train_books cost_per_book : ℕ) : ℕ :=
  (animal_books + outer_space_books + train_books) * cost_per_book

/-- Theorem stating that Dave spent $102 on books -/
theorem dave_spent_102 :
  dave_spent 8 6 3 6 = 102 := by
  sorry

end dave_spent_102_l4137_413718


namespace problem_solution_l4137_413756

theorem problem_solution (x : ℝ) : (0.65 * x = 0.20 * 422.50) → x = 130 := by
  sorry

end problem_solution_l4137_413756


namespace circle_diameter_from_area_ratio_l4137_413794

/-- Given two circles A and B where A is inside B, this theorem proves the diameter of A
    given the diameter of B and the ratio of areas. -/
theorem circle_diameter_from_area_ratio (dB : ℝ) (r : ℝ) :
  dB = 20 →  -- Diameter of circle B is 20 cm
  r = 1/7 →  -- Ratio of area of A to shaded area is 1:7
  ∃ dA : ℝ,  -- There exists a diameter for circle A
    (π * (dA/2)^2) / (π * (dB/2)^2 - π * (dA/2)^2) = r ∧  -- Area ratio condition
    abs (dA - 7.08) < 0.01  -- Diameter of A is approximately 7.08 cm
    := by sorry

end circle_diameter_from_area_ratio_l4137_413794


namespace inscribed_circle_diameter_l4137_413745

/-- Given a right triangle with sides 9, 12, and 15, the diameter of its inscribed circle is 6. -/
theorem inscribed_circle_diameter (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_sides : a = 9 ∧ b = 12 ∧ c = 15) : 
  2 * ((a + b - c) / 2) = 6 := by
  sorry


end inscribed_circle_diameter_l4137_413745


namespace plates_problem_l4137_413705

theorem plates_problem (x : ℚ) : 
  (1/3 * x - 2/3) - 1/2 * ((2/3 * x) - 4/3) = 9 → x = 29 := by
  sorry

end plates_problem_l4137_413705


namespace mistaken_division_l4137_413762

theorem mistaken_division (n : ℕ) : 
  (n / 9 = 8 ∧ n % 9 = 6) → n / 6 = 13 := by
sorry

end mistaken_division_l4137_413762


namespace quadratic_factor_evaluation_l4137_413713

theorem quadratic_factor_evaluation (b c : ℤ) : 
  let p : ℝ → ℝ := λ x => x^2 + b*x + c
  (∃ q : ℝ → ℝ, (∀ x, x^4 + 8*x^2 + 36 = p x * q x)) →
  (∃ r : ℝ → ℝ, (∀ x, 2*x^4 + 9*x^2 + 37*x + 18 = p x * r x)) →
  p (-1) = 12 := by
sorry

end quadratic_factor_evaluation_l4137_413713


namespace pie_eating_contest_l4137_413753

theorem pie_eating_contest (erik_pie frank_pie : Float) 
  (h1 : erik_pie = 0.6666666666666666)
  (h2 : frank_pie = 0.3333333333333333) :
  erik_pie - frank_pie = 0.3333333333333333 := by
  sorry

end pie_eating_contest_l4137_413753


namespace right_triangle_altitude_condition_l4137_413758

theorem right_triangle_altitude_condition 
  (a b m : ℝ) 
  (h_positive : a > 0 ∧ b > 0) 
  (h_right_triangle : a^2 + b^2 = (a + b)^2 / 4) 
  (h_altitude : m = (1/5) * Real.sqrt (9*b^2 - 16*a^2)) : 
  (m = (1/5) * Real.sqrt (9*b^2 - 16*a^2)) ↔ b = 2*a := by
sorry

end right_triangle_altitude_condition_l4137_413758


namespace number_plus_19_equals_47_l4137_413755

theorem number_plus_19_equals_47 (x : ℤ) : x + 19 = 47 → x = 28 := by
  sorry

end number_plus_19_equals_47_l4137_413755


namespace max_y_value_l4137_413761

theorem max_y_value (x y : ℤ) (h : 2*x*y + 8*x + 2*y = -14) : 
  ∃ (max_y : ℤ), (∃ (x' : ℤ), 2*x'*max_y + 8*x' + 2*max_y = -14) ∧ 
  (∀ (y' : ℤ), (∃ (x'' : ℤ), 2*x''*y' + 8*x'' + 2*y' = -14) → y' ≤ max_y) ∧
  max_y = 5 := by
sorry

end max_y_value_l4137_413761


namespace M_eq_real_l4137_413778

/-- The set M of complex numbers z where (z-1)^2 = |z-1|^2 -/
def M : Set ℂ := {z : ℂ | (z - 1)^2 = Complex.abs (z - 1)^2}

/-- Theorem stating that M is equal to the set of real numbers -/
theorem M_eq_real : M = {z : ℂ | z.im = 0} := by sorry

end M_eq_real_l4137_413778


namespace system_solution_l4137_413779

-- Define the system of equations
def equation1 (x y : ℝ) : Prop :=
  (x^4 / y^2)^(Real.log y) = (-x)^(Real.log (-x*y))

def equation2 (x y : ℝ) : Prop :=
  2*y^2 - x*y - x^2 - 4*x - 8*y = 0

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {(-4, 2), (-2, 2), ((Real.sqrt 17 - 9)/2, (Real.sqrt 17 - 1)/2)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solution_set :=
sorry

end system_solution_l4137_413779


namespace unique_number_division_multiplication_l4137_413732

theorem unique_number_division_multiplication : ∃! x : ℚ, (x / 6) * 12 = 12 := by
  sorry

end unique_number_division_multiplication_l4137_413732


namespace characterize_u_l4137_413795

/-- A function is strictly monotonic if it preserves the order relation -/
def StrictlyMonotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The main theorem statement -/
theorem characterize_u (u : ℝ → ℝ) :
  (∃ f : ℝ → ℝ, StrictlyMonotonic f ∧
    (∀ x y : ℝ, f (x + y) = f x * u y + f y)) →
  (∃ a : ℝ, ∀ x : ℝ, u x = Real.exp (a * x)) :=
by sorry

end characterize_u_l4137_413795


namespace investment_sum_l4137_413731

/-- Proves that if a sum P invested at 18% p.a. simple interest for two years yields Rs. 600 more interest than if invested at 12% p.a. simple interest for the same period, then P = 5000. -/
theorem investment_sum (P : ℝ) : 
  P * (18 / 100) * 2 - P * (12 / 100) * 2 = 600 → P = 5000 := by
  sorry

end investment_sum_l4137_413731


namespace x_squared_mod_26_l4137_413703

theorem x_squared_mod_26 (x : ℤ) (h1 : 6 * x ≡ 14 [ZMOD 26]) (h2 : 4 * x ≡ 20 [ZMOD 26]) :
  x^2 ≡ 12 [ZMOD 26] := by
  sorry

end x_squared_mod_26_l4137_413703


namespace sum_of_a_and_b_is_zero_l4137_413752

theorem sum_of_a_and_b_is_zero (a b : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : (a + i) / i = 1 + b * i) : a + b = 0 := by
  sorry

end sum_of_a_and_b_is_zero_l4137_413752


namespace average_problem_l4137_413765

theorem average_problem (x : ℝ) : (2 + 76 + x) / 3 = 5 → x = -63 := by
  sorry

end average_problem_l4137_413765


namespace percentage_increase_theorem_l4137_413787

theorem percentage_increase_theorem (initial_value : ℝ) 
  (first_increase_percent : ℝ) (second_increase_percent : ℝ) :
  let first_increase := initial_value * (1 + first_increase_percent / 100)
  let final_value := first_increase * (1 + second_increase_percent / 100)
  initial_value = 5000 ∧ first_increase_percent = 65 ∧ second_increase_percent = 45 →
  final_value = 11962.5 := by
sorry

end percentage_increase_theorem_l4137_413787


namespace counterfeit_coin_strategy_exists_l4137_413743

/-- Represents a weighing operation that can compare two groups of coins. -/
def Weighing := List Nat → List Nat → Ordering

/-- Represents a strategy for finding the counterfeit coin. -/
def Strategy := List Nat → List Weighing → Option Nat

/-- The number of coins. -/
def n : Nat := 81

/-- The maximum number of weighings allowed. -/
def max_weighings : Nat := 4

/-- Theorem stating that there exists a strategy to find the counterfeit coin. -/
theorem counterfeit_coin_strategy_exists :
  ∃ (s : Strategy),
    ∀ (counterfeit : Nat),
      counterfeit < n →
      ∃ (weighings : List Weighing),
        weighings.length ≤ max_weighings ∧
        s (List.range n) weighings = some counterfeit :=
by sorry

end counterfeit_coin_strategy_exists_l4137_413743


namespace largest_angle_in_specific_hexagon_l4137_413730

/-- Represents the ratio of angles in a hexagon -/
structure HexagonRatio :=
  (a b c d e f : ℕ)

/-- Calculates the largest angle in a hexagon given a ratio of angles -/
def largestAngleInHexagon (ratio : HexagonRatio) : ℚ :=
  let sum := ratio.a + ratio.b + ratio.c + ratio.d + ratio.e + ratio.f
  let angleUnit := 720 / sum
  angleUnit * (max ratio.a (max ratio.b (max ratio.c (max ratio.d (max ratio.e ratio.f)))))

theorem largest_angle_in_specific_hexagon :
  largestAngleInHexagon ⟨2, 3, 3, 4, 4, 5⟩ = 1200 / 7 := by
  sorry

#eval largestAngleInHexagon ⟨2, 3, 3, 4, 4, 5⟩

end largest_angle_in_specific_hexagon_l4137_413730


namespace initial_average_production_l4137_413727

theorem initial_average_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) 
  (h1 : n = 9)
  (h2 : today_production = 90)
  (h3 : new_average = 54) :
  ∃ initial_average : ℕ, 
    initial_average * n + today_production = new_average * (n + 1) ∧ 
    initial_average = 50 := by
  sorry

end initial_average_production_l4137_413727


namespace pencil_distribution_l4137_413742

theorem pencil_distribution (num_pens : ℕ) (num_pencils : ℕ) (num_students : ℕ) :
  num_pens = 781 →
  num_students = 71 →
  num_pens % num_students = 0 →
  num_pencils % num_students = 0 →
  ∃ k : ℕ, num_pencils = 71 * k :=
by
  sorry

end pencil_distribution_l4137_413742


namespace rounding_estimate_larger_l4137_413709

theorem rounding_estimate_larger (a b c d a' b' c' d' : ℕ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  a' ≥ a → b' ≤ b → c' ≤ c → d' ≤ d →
  (a' : ℚ) / b' - c' - d' > (a : ℚ) / b - c - d :=
by sorry

end rounding_estimate_larger_l4137_413709


namespace sixth_root_of_1061520150601_l4137_413785

theorem sixth_root_of_1061520150601 :
  let n : ℕ := 1061520150601
  ∃ (m : ℕ), m = 101 ∧ m^6 = n :=
by
  sorry

end sixth_root_of_1061520150601_l4137_413785


namespace smallest_prime_after_six_nonprimes_l4137_413796

/-- A function that returns true if a natural number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns true if all numbers in the range [a, b] are nonprime, false otherwise -/
def allNonPrime (a b : ℕ) : Prop := sorry

theorem smallest_prime_after_six_nonprimes : 
  ∃ (k : ℕ), 
    isPrime 97 ∧ 
    (∀ p < 97, isPrime p → ¬(allNonPrime (p + 1) (p + 6))) ∧
    allNonPrime 91 96 := by
  sorry

end smallest_prime_after_six_nonprimes_l4137_413796


namespace spoons_multiple_of_groups_l4137_413724

/-- Represents the number of commemorative plates Daniel has -/
def num_plates : ℕ := 44

/-- Represents the number of groups Daniel can form -/
def num_groups : ℕ := 11

/-- Represents the number of commemorative spoons Daniel has -/
def num_spoons : ℕ := sorry

/-- Theorem stating that the number of spoons is a multiple of the number of groups -/
theorem spoons_multiple_of_groups :
  ∃ k : ℕ, num_spoons = k * num_groups :=
sorry

end spoons_multiple_of_groups_l4137_413724


namespace floor_painting_theorem_l4137_413736

/-- The number of integer solutions to the floor painting problem -/
def floor_painting_solutions : Nat :=
  (Finset.filter
    (fun p : Nat × Nat =>
      let a := p.1
      let b := p.2
      b > a ∧ (a - 4) * (b - 4) = a * b / 2)
    (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- The floor painting problem has exactly 3 solutions -/
theorem floor_painting_theorem : floor_painting_solutions = 3 := by
  sorry

end floor_painting_theorem_l4137_413736


namespace candy_solution_l4137_413750

def candy_problem (f b j : ℕ) : Prop :=
  f = 12 ∧ b = f + 6 ∧ j = 10 * (f + b)

theorem candy_solution : 
  ∀ f b j : ℕ, candy_problem f b j → (40 * j^2) / 100 = 36000 := by
  sorry

end candy_solution_l4137_413750


namespace jogger_distance_ahead_l4137_413748

/-- Calculates the distance a jogger is ahead of a train given their speeds and the time it takes for the train to pass the jogger. -/
theorem jogger_distance_ahead (jogger_speed train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  passing_time = 38 →
  (train_speed - jogger_speed) * passing_time - train_length = 260 :=
by sorry

end jogger_distance_ahead_l4137_413748


namespace vectors_form_basis_l4137_413747

theorem vectors_form_basis (a b : ℝ × ℝ) : 
  a = (1, 2) → b = (3, 1) → LinearIndependent ℝ ![a, b] :=
by
  sorry

end vectors_form_basis_l4137_413747


namespace area_of_region_S_l4137_413783

/-- A rhombus with side length 3 and angle B = 150° --/
structure Rhombus :=
  (side_length : ℝ)
  (angle_B : ℝ)
  (h_side : side_length = 3)
  (h_angle : angle_B = 150)

/-- The region S inside the rhombus closer to vertex B than to A, C, or D --/
def region_S (r : Rhombus) : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² --/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the area of region S is approximately 1.1 --/
theorem area_of_region_S (r : Rhombus) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |area (region_S r) - 1.1| < ε :=
sorry

end area_of_region_S_l4137_413783


namespace inequality_of_distinct_reals_l4137_413729

theorem inequality_of_distinct_reals (a b c : ℝ) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  |a / (b - c)| + |b / (c - a)| + |c / (a - b)| ≥ 2 := by
  sorry

end inequality_of_distinct_reals_l4137_413729


namespace no_rational_solutions_for_positive_k_l4137_413723

theorem no_rational_solutions_for_positive_k : 
  ¬ ∃ (k : ℕ+) (x : ℚ), k * x^2 + 30 * x + k = 0 := by
  sorry

end no_rational_solutions_for_positive_k_l4137_413723


namespace find_a_l4137_413780

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := (a * x - 1) / (x + 1) > 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | x < -1 ∨ x > 1/2}

-- Theorem statement
theorem find_a : ∃ a : ℝ, ∀ x : ℝ, inequality a x ↔ x ∈ solution_set a :=
sorry

end find_a_l4137_413780


namespace smallest_n_with_properties_l4137_413726

theorem smallest_n_with_properties : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 2 * n = k^2) ∧ 
  (∃ (l : ℕ), 3 * n = l^3) ∧ 
  (∃ (m : ℕ), 5 * n = m^5) ∧ 
  (∀ (x : ℕ), x > 0 ∧ 
    (∃ (k : ℕ), 2 * x = k^2) ∧ 
    (∃ (l : ℕ), 3 * x = l^3) ∧ 
    (∃ (m : ℕ), 5 * x = m^5) → 
    x ≥ n) ∧ 
  n = 11250 :=
sorry

end smallest_n_with_properties_l4137_413726


namespace base_conversion_theorem_l4137_413791

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base k to base 10 -/
def baseKToBase10 (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The theorem stating that the base k in which 167₈ is written as 315ₖ is equal to 6 -/
theorem base_conversion_theorem : 
  ∃ k : ℕ, k > 1 ∧ base8ToBase10 167 = baseKToBase10 315 k ∧ k = 6 := by sorry

end base_conversion_theorem_l4137_413791


namespace vector_projection_l4137_413751

/-- Given two vectors a and b in ℝ², and a vector c such that a + c = 0,
    prove that the projection of c onto b is -√65/5 -/
theorem vector_projection (a b c : ℝ × ℝ) :
  a = (2, 3) →
  b = (-4, 7) →
  a + c = (0, 0) →
  (c.1 * b.1 + c.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -Real.sqrt 65 / 5 := by
  sorry

end vector_projection_l4137_413751


namespace range_of_a_for_inequality_l4137_413786

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |x^2 + a*x + 2| ≤ 4) ↔ a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end range_of_a_for_inequality_l4137_413786


namespace sam_supplies_cost_school_supplies_cost_proof_l4137_413712

/-- Represents the school supplies -/
structure Supplies :=
  (glue_sticks : ℕ)
  (pencils : ℕ)
  (erasers : ℕ)

/-- Calculates the cost of supplies given their quantities and prices -/
def calculate_cost (s : Supplies) (glue_price pencil_price eraser_price : ℚ) : ℚ :=
  s.glue_sticks * glue_price + s.pencils * pencil_price + s.erasers * eraser_price

theorem sam_supplies_cost (total : Supplies) (emily : Supplies) (sophie : Supplies)
    (glue_price pencil_price eraser_price : ℚ) : ℚ :=
  let sam : Supplies := {
    glue_sticks := total.glue_sticks - emily.glue_sticks - sophie.glue_sticks,
    pencils := total.pencils - emily.pencils - sophie.pencils,
    erasers := total.erasers - emily.erasers - sophie.erasers
  }
  calculate_cost sam glue_price pencil_price eraser_price

/-- The main theorem to prove -/
theorem school_supplies_cost_proof 
    (total : Supplies)
    (emily : Supplies)
    (sophie : Supplies)
    (glue_price pencil_price eraser_price : ℚ) :
    total.glue_sticks = 27 ∧ 
    total.pencils = 40 ∧ 
    total.erasers = 15 ∧
    glue_price = 1 ∧
    pencil_price = 1/2 ∧
    eraser_price = 3/4 ∧
    emily.glue_sticks = 9 ∧
    emily.pencils = 18 ∧
    emily.erasers = 5 ∧
    sophie.glue_sticks = 12 ∧
    sophie.pencils = 14 ∧
    sophie.erasers = 4 →
    sam_supplies_cost total emily sophie glue_price pencil_price eraser_price = 29/2 := by
  sorry

end sam_supplies_cost_school_supplies_cost_proof_l4137_413712


namespace cube_root_equation_solution_l4137_413777

theorem cube_root_equation_solution :
  let x : ℝ := 168 / 5
  (15 * x + (15 * x + 8) ^ (1/3)) ^ (1/3) = 8 := by sorry

end cube_root_equation_solution_l4137_413777


namespace water_pumped_30_minutes_l4137_413704

/-- Represents a water pumping system -/
structure WaterPump where
  gallons_per_hour : ℝ

/-- Calculates the amount of water pumped in a given time -/
def water_pumped (pump : WaterPump) (hours : ℝ) : ℝ :=
  pump.gallons_per_hour * hours

theorem water_pumped_30_minutes (pump : WaterPump) 
  (h : pump.gallons_per_hour = 500) : 
  water_pumped pump (30 / 60) = 250 := by
  sorry

#check water_pumped_30_minutes

end water_pumped_30_minutes_l4137_413704


namespace parallel_vectors_subtraction_l4137_413722

/-- Given two parallel vectors a and b, prove that 2a - b equals (4, -8) -/
theorem parallel_vectors_subtraction (m : ℝ) :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (m, 4)
  (a.1 * b.2 = a.2 * b.1) →  -- Condition for parallel vectors
  (2 * a.1 - b.1, 2 * a.2 - b.2) = (4, -8) := by
sorry

end parallel_vectors_subtraction_l4137_413722


namespace abc_sum_reciprocal_l4137_413717

theorem abc_sum_reciprocal (a b c : ℝ) (ha : a ≠ 1) (hb : b ≠ 1) (hc : c ≠ 1)
  (h1 : a * b * c = 1)
  (h2 : a^2 + b^2 + c^2 - (1/a^2 + 1/b^2 + 1/c^2) = 8*(a+b+c) - 8*(a*b+b*c+c*a)) :
  1/(a-1) + 1/(b-1) + 1/(c-1) = -3/2 := by
  sorry

end abc_sum_reciprocal_l4137_413717


namespace cereal_box_servings_l4137_413725

/-- Calculates the number of servings in a cereal box -/
def servings_in_box (total_cups : ℕ) (cups_per_serving : ℕ) : ℕ :=
  total_cups / cups_per_serving

/-- Theorem: The number of servings in a cereal box that holds 18 cups,
    with each serving being 2 cups, is 9. -/
theorem cereal_box_servings :
  servings_in_box 18 2 = 9 := by
  sorry

end cereal_box_servings_l4137_413725


namespace spending_ratio_l4137_413759

-- Define the amounts spent by each person
def akeno_spent : ℚ := 2985
def lev_spent : ℚ := 995  -- This is derived from the solution, but we'll use it as a given
def ambrocio_spent : ℚ := lev_spent - 177

-- State the theorem
theorem spending_ratio :
  -- Conditions
  (akeno_spent = lev_spent + ambrocio_spent + 1172) →
  (ambrocio_spent = lev_spent - 177) →
  -- Conclusion
  (lev_spent / akeno_spent = 1 / 3) := by
sorry


end spending_ratio_l4137_413759


namespace tan_alpha_plus_pi_third_and_fraction_l4137_413760

noncomputable def α : ℝ := Real.arctan 3 * 2

theorem tan_alpha_plus_pi_third_and_fraction (h : Real.tan (α/2) = 3) :
  Real.tan (α + π/3) = (48 - 25 * Real.sqrt 3) / 11 ∧
  (Real.sin α + 2 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = -5/17 := by
  sorry

end tan_alpha_plus_pi_third_and_fraction_l4137_413760


namespace perfect_square_trinomial_l4137_413721

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a b : ℝ, ∀ x, x^2 - (m-3)*x + 16 = (a*x + b)^2) ↔ (m = -5 ∨ m = 11) :=
sorry

end perfect_square_trinomial_l4137_413721


namespace angle_between_vectors_l4137_413781

theorem angle_between_vectors (a b : ℝ × ℝ) :
  a = (Real.sqrt 3, 1) →
  b = (Real.sqrt 3, -1) →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 3 :=
by sorry

end angle_between_vectors_l4137_413781


namespace geometric_sequence_first_term_l4137_413749

/-- A geometric sequence with third term 5 and sixth term 40 has first term 5/4 -/
theorem geometric_sequence_first_term (a : ℝ) (r : ℝ) : 
  a * r^2 = 5 → a * r^5 = 40 → a = 5/4 := by
sorry

end geometric_sequence_first_term_l4137_413749


namespace double_then_half_sixteen_l4137_413788

theorem double_then_half_sixteen : 
  let initial_number := 16
  let doubled := initial_number * 2
  let halved := doubled / 2
  halved = 2^4 := by sorry

end double_then_half_sixteen_l4137_413788


namespace power_product_equality_l4137_413772

theorem power_product_equality : (3^3 * 5^3) * (3^8 * 5^8) = 15^11 := by
  sorry

end power_product_equality_l4137_413772


namespace no_real_roots_l4137_413799

-- Define the sequence of polynomials P_n(x)
def P : ℕ → (ℝ → ℝ)
  | 0 => λ _ => 1
  | n + 1 => λ x => x^(17*(n+1)) - P n x

-- Theorem statement
theorem no_real_roots : ∀ n : ℕ, ∀ x : ℝ, P n x ≠ 0 := by
  sorry

end no_real_roots_l4137_413799


namespace hamburger_combinations_l4137_413793

/-- The number of different condiments available. -/
def num_condiments : ℕ := 10

/-- The number of patty options available. -/
def num_patty_options : ℕ := 4

/-- The total number of different hamburger combinations. -/
def total_combinations : ℕ := 2^num_condiments * num_patty_options

/-- Theorem stating that the total number of different hamburger combinations is 4096. -/
theorem hamburger_combinations : total_combinations = 4096 := by
  sorry

end hamburger_combinations_l4137_413793


namespace equation_three_solutions_l4137_413728

/-- The equation has exactly three solutions when a is 0, 5, or 9 -/
theorem equation_three_solutions (x : ℝ) (a : ℝ) : 
  (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, 
    (Real.sqrt (x - 1) * (|x^2 - 10*x + 16| - a)) / 
    (a*x^2 - 7*x^2 - 10*a*x + 70*x + 21*a - 147) = 0) ↔ 
  (a = 0 ∨ a = 5 ∨ a = 9) :=
sorry

end equation_three_solutions_l4137_413728


namespace optimal_meal_plan_l4137_413711

/-- Represents the nutritional content of a meal -/
structure Nutrition :=
  (carbs : ℕ)
  (protein : ℕ)
  (vitaminC : ℕ)

/-- Represents the meal plan -/
structure MealPlan :=
  (lunch : ℕ)
  (dinner : ℕ)

def lunch_nutrition : Nutrition := ⟨12, 6, 6⟩
def dinner_nutrition : Nutrition := ⟨8, 6, 10⟩

def lunch_cost : ℚ := 2.5
def dinner_cost : ℚ := 4

def minimum_nutrition : Nutrition := ⟨64, 42, 54⟩

def total_nutrition (plan : MealPlan) : Nutrition :=
  ⟨plan.lunch * lunch_nutrition.carbs + plan.dinner * dinner_nutrition.carbs,
   plan.lunch * lunch_nutrition.protein + plan.dinner * dinner_nutrition.protein,
   plan.lunch * lunch_nutrition.vitaminC + plan.dinner * dinner_nutrition.vitaminC⟩

def meets_requirements (plan : MealPlan) : Prop :=
  let total := total_nutrition plan
  total.carbs ≥ minimum_nutrition.carbs ∧
  total.protein ≥ minimum_nutrition.protein ∧
  total.vitaminC ≥ minimum_nutrition.vitaminC

def total_cost (plan : MealPlan) : ℚ :=
  plan.lunch * lunch_cost + plan.dinner * dinner_cost

theorem optimal_meal_plan :
  ∃ (plan : MealPlan),
    meets_requirements plan ∧
    (∀ (other : MealPlan), meets_requirements other → total_cost plan ≤ total_cost other) ∧
    plan.lunch = 4 ∧ plan.dinner = 3 :=
sorry

end optimal_meal_plan_l4137_413711


namespace real_roots_condition_l4137_413798

theorem real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - x + 3 = 0) ↔ (k ≤ 1/12 ∧ k ≠ 0) :=
sorry

end real_roots_condition_l4137_413798


namespace room_length_proof_l4137_413763

/-- The length of a rectangular room given its width, number of tiles, and tile size. -/
theorem room_length_proof (width : ℝ) (num_tiles : ℕ) (tile_size : ℝ) 
  (h1 : width = 12)
  (h2 : num_tiles = 6)
  (h3 : tile_size = 4)
  : width * (num_tiles * tile_size / width) = 2 := by
  sorry

end room_length_proof_l4137_413763


namespace johns_number_proof_l4137_413733

theorem johns_number_proof : 
  ∃! x : ℕ, 
    10 ≤ x ∧ x < 100 ∧ 
    (∃ a b : ℕ, 
      4 * x + 17 = 10 * a + b ∧
      10 * b + a ≥ 91 ∧ 
      10 * b + a ≤ 95) ∧
    x = 8 := by
  sorry

end johns_number_proof_l4137_413733


namespace perp_para_implies_perp_line_perp_para_planes_implies_perp_perp_two_planes_implies_para_l4137_413754

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Plane → Prop)
variable (perpLine : Line → Line → Prop)
variable (paraPlane : Plane → Plane → Prop)

-- Axioms for the relations
axiom perp_antisymm {l : Line} {p : Plane} : perp l p ↔ perp l p
axiom para_antisymm {l : Line} {p : Plane} : para l p ↔ para l p
axiom perpLine_antisymm {l1 l2 : Line} : perpLine l1 l2 ↔ perpLine l2 l1
axiom paraPlane_antisymm {p1 p2 : Plane} : paraPlane p1 p2 ↔ paraPlane p2 p1

-- Theorem 1
theorem perp_para_implies_perp_line {m n : Line} {α : Plane} 
  (h1 : perp m α) (h2 : para n α) : perpLine m n := by sorry

-- Theorem 2
theorem perp_para_planes_implies_perp {m : Line} {α β : Plane}
  (h1 : perp m α) (h2 : paraPlane α β) : perp m β := by sorry

-- Theorem 3
theorem perp_two_planes_implies_para {m : Line} {α β : Plane}
  (h1 : perp m α) (h2 : perp m β) : paraPlane α β := by sorry

end perp_para_implies_perp_line_perp_para_planes_implies_perp_perp_two_planes_implies_para_l4137_413754


namespace no_consecutive_prime_roots_l4137_413738

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if two numbers are consecutive primes -/
def areConsecutivePrimes (p q : ℕ) : Prop := sorry

/-- The theorem stating that there are no values of k satisfying the conditions -/
theorem no_consecutive_prime_roots :
  ¬ ∃ (k : ℤ) (p q : ℕ), 
    p < q ∧ 
    areConsecutivePrimes p q ∧ 
    p + q = 65 ∧ 
    p * q = k ∧
    ∀ (x : ℤ), x^2 - 65*x + k = 0 ↔ (x = p ∨ x = q) :=
sorry

end no_consecutive_prime_roots_l4137_413738


namespace angle_trigonometry_l4137_413770

open Real

-- Define the angle θ
variable (θ : ℝ)

-- Define the condition that the terminal side of θ lies on y = 2x (x ≥ 0)
def terminal_side_condition (θ : ℝ) : Prop :=
  ∃ (x : ℝ), x ≥ 0 ∧ tan θ = 2

-- Theorem statement
theorem angle_trigonometry (h : terminal_side_condition θ) :
  (tan θ = 2) ∧
  ((2 * cos θ + 3 * sin θ) / (cos θ - 3 * sin θ) + sin θ * cos θ = -6/5) := by
  sorry

end angle_trigonometry_l4137_413770


namespace tan_2alpha_value_l4137_413776

theorem tan_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : 3 * Real.cos (2 * α) - 4 * Real.sin α = 1) : 
  Real.tan (2 * α) = -4 * Real.sqrt 2 / 7 := by
  sorry

end tan_2alpha_value_l4137_413776


namespace circle_tangency_l4137_413792

/-- Two circles are internally tangent if the distance between their centers
    is equal to the difference of their radii. -/
def internally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r2 - r1)^2

theorem circle_tangency (n : ℝ) : 
  internally_tangent (0, 0) (3, 4) 1 (Real.sqrt (25 - n)) → n = -11 := by
  sorry

#check circle_tangency

end circle_tangency_l4137_413792


namespace soccer_team_wins_l4137_413739

theorem soccer_team_wins (total_games : ℕ) (win_percentage : ℚ) (games_won : ℕ) : 
  total_games = 158 →
  win_percentage = 40 / 100 →
  games_won = (total_games : ℚ) * win_percentage →
  games_won = 63 := by
sorry

end soccer_team_wins_l4137_413739


namespace parallel_vectors_x_value_l4137_413766

/-- Two vectors are parallel if their corresponding components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (2, 4)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → x = -1 := by
sorry

end parallel_vectors_x_value_l4137_413766


namespace multiply_divide_example_l4137_413715

theorem multiply_divide_example : (3.242 * 15) / 100 = 0.4863 := by
  sorry

end multiply_divide_example_l4137_413715


namespace cubic_root_coefficient_a_l4137_413720

theorem cubic_root_coefficient_a (a b : ℚ) : 
  ((-1 - 4 * Real.sqrt 2)^3 + a * (-1 - 4 * Real.sqrt 2)^2 + b * (-1 - 4 * Real.sqrt 2) + 31 = 0) →
  a = 1 := by
  sorry

end cubic_root_coefficient_a_l4137_413720


namespace dispersion_measures_l4137_413741

/-- A sample of data points -/
def Sample : Type := List ℝ

/-- Standard deviation of a sample -/
noncomputable def standardDeviation (s : Sample) : ℝ := sorry

/-- Range of a sample -/
noncomputable def range (s : Sample) : ℝ := sorry

/-- Median of a sample -/
noncomputable def median (s : Sample) : ℝ := sorry

/-- Mean of a sample -/
noncomputable def mean (s : Sample) : ℝ := sorry

/-- A measure of dispersion is a function that quantifies the spread of a sample -/
def isDispersionMeasure (f : Sample → ℝ) : Prop := sorry

theorem dispersion_measures (s : Sample) :
  isDispersionMeasure standardDeviation ∧
  isDispersionMeasure range ∧
  ¬isDispersionMeasure median ∧
  ¬isDispersionMeasure mean :=
sorry

end dispersion_measures_l4137_413741


namespace rhombus_longer_diagonal_l4137_413735

/-- A rhombus with side length 37 units and shorter diagonal 40 units has a longer diagonal of 62 units. -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ) : 
  side = 37 → shorter_diagonal = 40 → longer_diagonal = 62 → 
  side^2 = (shorter_diagonal / 2)^2 + (longer_diagonal / 2)^2 :=
by sorry

end rhombus_longer_diagonal_l4137_413735


namespace vertices_form_hyperbola_branch_l4137_413764

/-- Given a real number k and a constant c, the set of vertices (x_t, y_t) of the parabola
    y = t^2 x^2 + 2ktx + c for varying t forms one branch of a hyperbola. -/
theorem vertices_form_hyperbola_branch (k : ℝ) (c : ℝ) :
  ∃ (A B C D : ℝ), A ≠ 0 ∧
    (∀ x_t y_t : ℝ, x_t ≠ 0 →
      (∃ t : ℝ, y_t = t^2 * x_t^2 + 2*k*t*x_t + c ∧
                x_t = -k/t) →
      A * x_t * y_t + B * x_t + C * y_t + D = 0) :=
sorry

end vertices_form_hyperbola_branch_l4137_413764


namespace initial_sale_percentage_l4137_413769

theorem initial_sale_percentage (P : ℝ) (x : ℝ) (h : x ≥ 0 ∧ x ≤ 1) : 
  ((1 - x) * P * 0.9 = 0.45 * P) → x = 0.5 := by
  sorry

end initial_sale_percentage_l4137_413769


namespace arithmetic_sequence_solutions_l4137_413773

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  first_fourth_sum : a 1 + a 4 = 4
  second_third_product : a 2 * a 3 = 3
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_solutions (seq : ArithmeticSequence) :
  (seq.a 1 = -1 ∧ seq.d = 2 ∧ (∀ n, seq.a n = 2 * n - 3) ∧ (∀ n, S seq n = n^2 - 2*n)) ∨
  (seq.a 1 = 5 ∧ seq.d = -2 ∧ (∀ n, seq.a n = 7 - 2 * n) ∧ (∀ n, S seq n = 6*n - n^2)) :=
sorry

end arithmetic_sequence_solutions_l4137_413773


namespace smallest_angle_for_complete_circle_l4137_413707

theorem smallest_angle_for_complete_circle : 
  ∃ (t : ℝ), t > 0 ∧ 
  (∀ (θ : ℝ), 0 ≤ θ ∧ θ ≤ t → ∃ (r : ℝ), r = Real.sin θ) ∧
  (∀ (s : ℝ), s > 0 ∧ s < t → ¬(∀ (θ : ℝ), 0 ≤ θ ∧ θ ≤ s → ∃ (r : ℝ), r = Real.sin θ)) ∧
  t = π :=
sorry

end smallest_angle_for_complete_circle_l4137_413707


namespace self_inverse_sum_zero_l4137_413740

def is_self_inverse (a d : ℝ) : Prop :=
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![a, 5; -12, d]
  M * M = 1

theorem self_inverse_sum_zero (a d : ℝ) (h : is_self_inverse a d) : a + d = 0 := by
  sorry

end self_inverse_sum_zero_l4137_413740


namespace developed_countries_completed_transformation_l4137_413771

-- Define the different stages of population growth patterns
inductive PopulationGrowthStage
| Traditional
| Transitional
| Modern

-- Define the types of countries
inductive CountryType
| Developed
| Developing

-- Define the world population distribution
def worldPopulationDistribution : CountryType → Bool
| CountryType.Developing => true
| CountryType.Developed => false

-- Define the population growth stage for each country type
def populationGrowthStage : CountryType → PopulationGrowthStage
| CountryType.Developing => PopulationGrowthStage.Traditional
| CountryType.Developed => PopulationGrowthStage.Modern

-- Define the overall global population growth stage
def globalPopulationGrowthStage : PopulationGrowthStage :=
  PopulationGrowthStage.Transitional

-- Theorem statement
theorem developed_countries_completed_transformation :
  (∀ c : CountryType, worldPopulationDistribution c → populationGrowthStage c = PopulationGrowthStage.Traditional) →
  globalPopulationGrowthStage = PopulationGrowthStage.Transitional →
  populationGrowthStage CountryType.Developed = PopulationGrowthStage.Modern :=
by
  sorry

end developed_countries_completed_transformation_l4137_413771


namespace jeffreys_farm_chickens_total_chickens_is_76_l4137_413744

/-- Calculates the total number of chickens on Jeffrey's farm -/
theorem jeffreys_farm_chickens (num_hens : ℕ) (hen_rooster_ratio : ℕ) (chicks_per_hen : ℕ) : ℕ :=
  let num_roosters := num_hens / hen_rooster_ratio
  let num_chicks := num_hens * chicks_per_hen
  num_hens + num_roosters + num_chicks

/-- Proves that the total number of chickens on Jeffrey's farm is 76 -/
theorem total_chickens_is_76 :
  jeffreys_farm_chickens 12 3 5 = 76 := by
  sorry

end jeffreys_farm_chickens_total_chickens_is_76_l4137_413744


namespace crease_length_of_folded_equilateral_triangle_l4137_413701

-- Define an equilateral triangle
structure EquilateralTriangle :=
  (side_length : ℝ)

-- Define the folded triangle
structure FoldedTriangle extends EquilateralTriangle :=
  (crease_length : ℝ)

-- Theorem statement
theorem crease_length_of_folded_equilateral_triangle 
  (triangle : EquilateralTriangle) 
  (h : triangle.side_length = 6) : 
  ∃ (folded : FoldedTriangle), 
    folded.side_length = triangle.side_length ∧ 
    folded.crease_length = 3 * Real.sqrt 3 :=
sorry

end crease_length_of_folded_equilateral_triangle_l4137_413701


namespace paving_stones_required_l4137_413746

theorem paving_stones_required (courtyard_length courtyard_width stone_length stone_width : ℝ) 
  (h1 : courtyard_length = 75)
  (h2 : courtyard_width = 20 + 3/4)
  (h3 : stone_length = 3 + 1/4)
  (h4 : stone_width = 2 + 1/2) : 
  ⌈(courtyard_length * courtyard_width) / (stone_length * stone_width)⌉ = 192 := by
  sorry

end paving_stones_required_l4137_413746


namespace tank_water_supply_l4137_413700

theorem tank_water_supply (C V : ℝ) 
  (h1 : C = 75 * (V + 10))
  (h2 : C = 60 * (V + 20)) :
  C / V = 100 := by
  sorry

end tank_water_supply_l4137_413700


namespace water_distribution_solution_l4137_413714

/-- Represents the water distribution problem -/
structure WaterDistribution where
  totalWater : Nat
  fiveOunceGlasses : Nat
  eightOunceGlasses : Nat
  sevenOunceGlasses : Nat

/-- Calculates the maximum number of friends and remaining 4-ounce glasses -/
def distributeWater (w : WaterDistribution) : Nat × Nat :=
  let usedWater := w.fiveOunceGlasses * 5 + w.eightOunceGlasses * 8 + w.sevenOunceGlasses * 7
  let remainingWater := w.totalWater - usedWater
  let fourOunceGlasses := remainingWater / 4
  let totalGlasses := w.fiveOunceGlasses + w.eightOunceGlasses + w.sevenOunceGlasses + fourOunceGlasses
  (totalGlasses, fourOunceGlasses)

/-- Theorem stating the solution to the water distribution problem -/
theorem water_distribution_solution (w : WaterDistribution) 
  (h1 : w.totalWater = 122)
  (h2 : w.fiveOunceGlasses = 6)
  (h3 : w.eightOunceGlasses = 4)
  (h4 : w.sevenOunceGlasses = 3) :
  distributeWater w = (22, 9) := by
  sorry

end water_distribution_solution_l4137_413714


namespace bankers_gain_interest_rate_l4137_413767

/-- Given a banker's gain, present worth, and time period, 
    calculate the annual interest rate. -/
theorem bankers_gain_interest_rate 
  (bankers_gain : ℝ) 
  (present_worth : ℝ) 
  (time_period : ℕ) 
  (h1 : bankers_gain = 36) 
  (h2 : present_worth = 400) 
  (h3 : time_period = 3) :
  ∃ r : ℝ, bankers_gain = present_worth * (1 + r)^time_period - present_worth :=
sorry

end bankers_gain_interest_rate_l4137_413767


namespace millet_exceeds_half_on_fourth_day_l4137_413719

/-- Represents the fraction of millet seeds in the feeder on a given day -/
def milletFraction (day : ℕ) : ℚ :=
  match day with
  | 0 => 3/10
  | n + 1 => (1/2 * milletFraction n + 3/10)

/-- Theorem stating that on the 4th day, the fraction of millet seeds exceeds 1/2 for the first time -/
theorem millet_exceeds_half_on_fourth_day :
  (milletFraction 4 > 1/2) ∧
  (∀ d : ℕ, d < 4 → milletFraction d ≤ 1/2) :=
sorry

end millet_exceeds_half_on_fourth_day_l4137_413719


namespace find_a_l4137_413757

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The expansion of (x + 1/x)^n -/
def expansion (n : ℕ) (x : ℝ) : ℝ := sorry

/-- The coefficient of x^k in the expansion of (x + 1/x)^n -/
def coefficient (n k : ℕ) : ℝ := sorry

theorem find_a : ∃ (a : ℝ), 
  (coefficient 6 3 * a + coefficient 6 2) = 30 ∧ 
  ∀ (b : ℝ), (coefficient 6 3 * b + coefficient 6 2) = 30 → b = a :=
sorry

end find_a_l4137_413757


namespace quadratic_equation_roots_l4137_413706

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, 2 * x₁^2 - 7 * x₁ + k = 0 ∧ 2 * x₂^2 - 7 * x₂ + k = 0 ∧ x₁ = 2) →
  (∃ x₂ : ℝ, x₂ = 3/2 ∧ k = 6) :=
by sorry

end quadratic_equation_roots_l4137_413706


namespace cheaper_module_cost_l4137_413797

-- Define the total number of modules
def total_modules : ℕ := 22

-- Define the number of cheaper modules
def cheaper_modules : ℕ := 21

-- Define the cost of the expensive module
def expensive_module_cost : ℚ := 10

-- Define the total stock value
def total_stock_value : ℚ := 62.5

-- Theorem to prove
theorem cheaper_module_cost :
  ∃ (x : ℚ), x > 0 ∧ x < expensive_module_cost ∧
  x * cheaper_modules + expensive_module_cost = total_stock_value ∧
  x = 2.5 := by sorry

end cheaper_module_cost_l4137_413797


namespace compare_negative_roots_l4137_413768

theorem compare_negative_roots : -3 * Real.sqrt 3 > -2 * Real.sqrt 7 := by
  sorry

end compare_negative_roots_l4137_413768


namespace inscribed_circle_rectangle_area_l4137_413774

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (length width : ℝ),
    r = 6 →
    length = 3 * width →
    width = 2 * r →
    length * width = 432 :=
by
  sorry

end inscribed_circle_rectangle_area_l4137_413774
