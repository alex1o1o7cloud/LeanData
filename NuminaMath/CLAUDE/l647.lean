import Mathlib

namespace NUMINAMATH_CALUDE_wine_cost_proof_l647_64750

/-- The current cost of a bottle of wine -/
def current_cost : ℝ := sorry

/-- The future cost of a bottle of wine after the price increase -/
def future_cost : ℝ := 1.25 * current_cost

/-- The increase in cost for five bottles -/
def total_increase : ℝ := 25

/-- The number of bottles -/
def num_bottles : ℕ := 5

theorem wine_cost_proof :
  (future_cost - current_cost) * num_bottles = total_increase ∧ current_cost = 20 := by sorry

end NUMINAMATH_CALUDE_wine_cost_proof_l647_64750


namespace NUMINAMATH_CALUDE_least_common_addition_of_primes_l647_64781

theorem least_common_addition_of_primes (x y : ℕ) : 
  Nat.Prime x → Nat.Prime y → x < y → 4 * x + y = 87 → x + y = 81 := by
  sorry

end NUMINAMATH_CALUDE_least_common_addition_of_primes_l647_64781


namespace NUMINAMATH_CALUDE_parallel_cuts_three_pieces_intersecting_cuts_four_pieces_l647_64705

-- Define a square
def Square : Type := Unit

-- Define a straight cut from edge to edge
def StraightCut (s : Square) : Type := Unit

-- Define parallel cuts
def ParallelCuts (s : Square) (c1 c2 : StraightCut s) : Prop := sorry

-- Define intersecting cuts
def IntersectingCuts (s : Square) (c1 c2 : StraightCut s) : Prop := sorry

-- Define the number of pieces resulting from cuts
def NumberOfPieces (s : Square) (c1 c2 : StraightCut s) : ℕ := sorry

-- Theorem for parallel cuts
theorem parallel_cuts_three_pieces (s : Square) (c1 c2 : StraightCut s) 
  (h : ParallelCuts s c1 c2) : NumberOfPieces s c1 c2 = 3 := by sorry

-- Theorem for intersecting cuts
theorem intersecting_cuts_four_pieces (s : Square) (c1 c2 : StraightCut s) 
  (h : IntersectingCuts s c1 c2) : NumberOfPieces s c1 c2 = 4 := by sorry

end NUMINAMATH_CALUDE_parallel_cuts_three_pieces_intersecting_cuts_four_pieces_l647_64705


namespace NUMINAMATH_CALUDE_x_varies_linearly_with_z_l647_64742

/-- Given that x varies as the cube of y and y varies as the cube root of z,
    prove that x varies linearly with z. -/
theorem x_varies_linearly_with_z 
  (k : ℝ) (j : ℝ) (x y z : ℝ → ℝ) 
  (h1 : ∀ t, x t = k * (y t)^3) 
  (h2 : ∀ t, y t = j * (z t)^(1/3)) :
  ∃ m : ℝ, ∀ t, x t = m * z t :=
sorry

end NUMINAMATH_CALUDE_x_varies_linearly_with_z_l647_64742


namespace NUMINAMATH_CALUDE_min_people_to_ask_l647_64727

theorem min_people_to_ask (knights : ℕ) (civilians : ℕ) : 
  knights = 50 → civilians = 15 → 
  ∃ (n : ℕ), n > civilians ∧ n - civilians ≤ knights ∧ 
  ∀ (m : ℕ), m < n → (m - civilians ≤ knights → m ≤ civilians) :=
sorry

end NUMINAMATH_CALUDE_min_people_to_ask_l647_64727


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l647_64714

theorem trigonometric_equation_solution (z : ℝ) : 
  (Real.sin (3 * z) + Real.sin z ^ 3 = (3 * Real.sqrt 3 / 4) * Real.sin (2 * z)) ↔ 
  (∃ k : ℤ, z = k * Real.pi) ∨ 
  (∃ n : ℤ, z = Real.pi / 2 * (2 * n + 1)) ∨ 
  (∃ l : ℤ, z = Real.pi / 6 + 2 * Real.pi * l ∨ z = -Real.pi / 6 + 2 * Real.pi * l) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l647_64714


namespace NUMINAMATH_CALUDE_min_value_cube_square_sum_l647_64794

theorem min_value_cube_square_sum (x y z : ℝ) 
  (h_non_neg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_constraint : 5*x + 16*y + 33*z ≥ 136) : 
  x^3 + y^3 + z^3 + x^2 + y^2 + z^2 ≥ 50 := by
  sorry

end NUMINAMATH_CALUDE_min_value_cube_square_sum_l647_64794


namespace NUMINAMATH_CALUDE_remainder_theorem_l647_64787

theorem remainder_theorem (n : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l647_64787


namespace NUMINAMATH_CALUDE_chicken_surprise_weight_theorem_l647_64776

/-- The weight of one serving of Chicken Surprise -/
def chicken_surprise_serving_weight (total_servings : ℕ) (chicken_weight_pounds : ℚ) (stuffing_weight_ounces : ℕ) : ℚ :=
  (chicken_weight_pounds * 16 + stuffing_weight_ounces) / total_servings

/-- Theorem: Given 12 servings of Chicken Surprise, 4.5 pounds of chicken, and 24 ounces of stuffing, one serving of Chicken Surprise is 8 ounces. -/
theorem chicken_surprise_weight_theorem :
  chicken_surprise_serving_weight 12 (9/2) 24 = 8 := by
  sorry

end NUMINAMATH_CALUDE_chicken_surprise_weight_theorem_l647_64776


namespace NUMINAMATH_CALUDE_expression_value_l647_64739

theorem expression_value (a b c : ℤ) (h1 : a = 12) (h2 : b = 2) (h3 : c = 7) :
  (a - (b - c)) - ((a - b) - c) = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l647_64739


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l647_64780

theorem simplify_trig_expression :
  Real.sqrt (1 - 2 * Real.sin 4 * Real.cos 4) = Real.cos 4 - Real.sin 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l647_64780


namespace NUMINAMATH_CALUDE_furniture_shop_cost_price_l647_64767

/-- Given a markup percentage and a selling price, calculates the cost price -/
def calculate_cost_price (markup_percentage : ℚ) (selling_price : ℚ) : ℚ :=
  selling_price / (1 + markup_percentage)

/-- Proves that for a 25% markup and selling price of 4800, the cost price is 3840 -/
theorem furniture_shop_cost_price :
  calculate_cost_price (25 / 100) 4800 = 3840 := by
  sorry

end NUMINAMATH_CALUDE_furniture_shop_cost_price_l647_64767


namespace NUMINAMATH_CALUDE_complement_intersection_M_N_l647_64729

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 3, 5}
def N : Set ℕ := {1, 3, 4, 6}

theorem complement_intersection_M_N : (M ∩ N)ᶜ = {2, 4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_M_N_l647_64729


namespace NUMINAMATH_CALUDE_total_wheels_count_l647_64708

theorem total_wheels_count (num_bicycles num_tricycles : ℕ) 
  (wheels_per_bicycle wheels_per_tricycle : ℕ) :
  num_bicycles = 24 →
  num_tricycles = 14 →
  wheels_per_bicycle = 2 →
  wheels_per_tricycle = 3 →
  num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_count_l647_64708


namespace NUMINAMATH_CALUDE_fourth_friend_age_l647_64736

theorem fourth_friend_age (age1 age2 age3 avg : ℕ) (h1 : age1 = 7) (h2 : age2 = 9) (h3 : age3 = 12) 
  (h_avg : (age1 + age2 + age3 + age4) / 4 = avg) (h_avg_val : avg = 9) : age4 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_friend_age_l647_64736


namespace NUMINAMATH_CALUDE_minimum_employees_needed_l647_64779

theorem minimum_employees_needed 
  (water_pollution : ℕ) 
  (air_pollution : ℕ) 
  (both : ℕ) 
  (h1 : water_pollution = 85) 
  (h2 : air_pollution = 73) 
  (h3 : both = 27) 
  (h4 : both ≤ water_pollution ∧ both ≤ air_pollution) : 
  water_pollution + air_pollution - both = 131 :=
sorry

end NUMINAMATH_CALUDE_minimum_employees_needed_l647_64779


namespace NUMINAMATH_CALUDE_problem_1_l647_64760

theorem problem_1 (m n : ℕ) (h1 : 3^m = 8) (h2 : 3^n = 2) : 3^(2*m - 3*n + 1) = 24 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l647_64760


namespace NUMINAMATH_CALUDE_rolles_theorem_application_l647_64726

-- Define the function f(x) = x^2 + 2x + 7
def f (x : ℝ) : ℝ := x^2 + 2*x + 7

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 2*x + 2

-- Theorem statement
theorem rolles_theorem_application :
  ∃ c ∈ Set.Ioo (-6 : ℝ) 4, f' c = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_rolles_theorem_application_l647_64726


namespace NUMINAMATH_CALUDE_smallest_prime_digit_sum_28_l647_64744

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a natural number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem: 1999 is the smallest prime number whose digits sum to 28 -/
theorem smallest_prime_digit_sum_28 :
  (∀ p : ℕ, p < 1999 → (is_prime p ∧ digit_sum p = 28) → False) ∧
  is_prime 1999 ∧ digit_sum 1999 = 28 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_digit_sum_28_l647_64744


namespace NUMINAMATH_CALUDE_f_pi_third_eq_half_l647_64710

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.sin x
  else if 1 ≤ x ∧ x ≤ Real.sqrt 2 then Real.cos x
  else Real.tan x

-- Theorem statement
theorem f_pi_third_eq_half : f (Real.pi / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_pi_third_eq_half_l647_64710


namespace NUMINAMATH_CALUDE_unique_solution_l647_64761

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  x + y - 1 = 0 ∧ x - 2*y + 2 = 0

/-- The solution to the system of equations -/
def solution : ℝ × ℝ := (0, 1)

/-- Theorem stating that the solution is unique and satisfies the system -/
theorem unique_solution :
  system solution.1 solution.2 ∧
  ∀ x y : ℝ, system x y → (x, y) = solution := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l647_64761


namespace NUMINAMATH_CALUDE_binomial_30_3_minus_10_l647_64785

theorem binomial_30_3_minus_10 : Nat.choose 30 3 - 10 = 4050 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_minus_10_l647_64785


namespace NUMINAMATH_CALUDE_a_share_profit_l647_64704

/-- Calculates the share of profit for an investor in a partnership business -/
def calculate_share_profit (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let ratio_a := investment_a / total_investment
  ratio_a * total_profit

/-- Theorem: A's share in the profit is 3660 given the investments and total profit -/
theorem a_share_profit (investment_a investment_b investment_c total_profit : ℚ) 
  (h1 : investment_a = 6300)
  (h2 : investment_b = 4200)
  (h3 : investment_c = 10500)
  (h4 : total_profit = 12200) :
  calculate_share_profit investment_a investment_b investment_c total_profit = 3660 := by
  sorry

#eval calculate_share_profit 6300 4200 10500 12200

end NUMINAMATH_CALUDE_a_share_profit_l647_64704


namespace NUMINAMATH_CALUDE_habitable_earth_surface_habitable_earth_surface_fraction_l647_64723

theorem habitable_earth_surface : Real → Prop :=
  λ x =>
    let water_coverage : Real := 3/5
    let inhabitable_land_ratio : Real := 2/3
    let non_agricultural_ratio : Real := 1/2
    let land_area : Real := 1 - water_coverage
    let inhabitable_area : Real := land_area * inhabitable_land_ratio
    let habitable_area : Real := inhabitable_area * non_agricultural_ratio
    x = habitable_area

theorem habitable_earth_surface_fraction :
  habitable_earth_surface (2/15) := by
  sorry

end NUMINAMATH_CALUDE_habitable_earth_surface_habitable_earth_surface_fraction_l647_64723


namespace NUMINAMATH_CALUDE_line_circle_intersection_k_range_l647_64777

/-- Given a line y = kx + 3 intersecting a circle (x-4)^2 + (y-3)^2 = 4 at two points M and N,
    where |MN| ≥ 2√3, prove that -√15/15 ≤ k ≤ √15/15 -/
theorem line_circle_intersection_k_range (k : ℝ) :
  (∃ M N : ℝ × ℝ,
    (M.1 - 4)^2 + (M.2 - 3)^2 = 4 ∧
    (N.1 - 4)^2 + (N.2 - 3)^2 = 4 ∧
    M.2 = k * M.1 + 3 ∧
    N.2 = k * N.1 + 3 ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12) →
  -Real.sqrt 15 / 15 ≤ k ∧ k ≤ Real.sqrt 15 / 15 := by
  sorry


end NUMINAMATH_CALUDE_line_circle_intersection_k_range_l647_64777


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l647_64759

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

theorem collinear_points_k_value :
  ∀ k : ℚ,
  let p1 : Point := ⟨2, -1⟩
  let p2 : Point := ⟨10, k⟩
  let p3 : Point := ⟨23, 4⟩
  collinear p1 p2 p3 → k = 19 / 21 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l647_64759


namespace NUMINAMATH_CALUDE_g_derivative_at_one_l647_64718

/-- The sequence of functions gₖ(x) -/
noncomputable def g : ℕ → (ℝ → ℝ)
| 0 => λ x => x^2 / (2 - x)
| (k+1) => λ x => x * g k x / (2 - g k x)

/-- The statement to be proved -/
theorem g_derivative_at_one (k : ℕ) :
  HasDerivAt (g k) (2^(k+1) - 1) 1 :=
sorry

end NUMINAMATH_CALUDE_g_derivative_at_one_l647_64718


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l647_64746

/-- A quadratic function with specific properties -/
def QuadraticFunction (m n : ℝ) : ℝ → ℝ := fun x ↦ m * x^2 - 2 * m * x + n + 1

/-- The derived function f based on g -/
def DerivedFunction (g : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := fun x ↦ g x + (2 - a) * x

/-- The theorem statement -/
theorem quadratic_function_properties 
  (m n : ℝ) 
  (h_m : m > 0)
  (h_max : ∃ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, QuadraticFunction m n x ≥ QuadraticFunction m n y)
  (h_min : ∃ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, QuadraticFunction m n x ≤ QuadraticFunction m n y)
  (h_max_val : ∃ x ∈ Set.Icc 0 3, QuadraticFunction m n x = 4)
  (h_min_val : ∃ x ∈ Set.Icc 0 3, QuadraticFunction m n x = 0)
  : 
  (∀ x, QuadraticFunction m n x = x^2 - 2*x + 1) ∧
  (∃ a, (a = -5 ∨ a = 4) ∧ 
    (∃ x ∈ Set.Icc (-1) 2, ∀ y ∈ Set.Icc (-1) 2, 
      DerivedFunction (QuadraticFunction m n) a x ≤ DerivedFunction (QuadraticFunction m n) a y) ∧
    (∃ x ∈ Set.Icc (-1) 2, DerivedFunction (QuadraticFunction m n) a x = -3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l647_64746


namespace NUMINAMATH_CALUDE_frog_jump_probability_l647_64717

-- Define the grid size
def gridSize : ℕ := 6

-- Define the jump size
def jumpSize : ℕ := 2

-- Define a position on the grid
structure Position where
  x : ℕ
  y : ℕ

-- Define the starting position
def startPos : Position := ⟨2, 3⟩

-- Define a function to check if a position is on the vertical side
def isOnVerticalSide (p : Position) : Prop :=
  p.x = 0 ∨ p.x = gridSize

-- Define a function to check if a position is on any side
def isOnAnySide (p : Position) : Prop :=
  p.x = 0 ∨ p.x = gridSize ∨ p.y = 0 ∨ p.y = gridSize

-- Define the probability of ending on a vertical side
def probEndVertical (p : Position) : ℝ := sorry

-- State the theorem
theorem frog_jump_probability :
  probEndVertical startPos = 3/4 := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l647_64717


namespace NUMINAMATH_CALUDE_probability_of_specific_student_selection_l647_64755

/- Define the total number of students and the number to be selected -/
def total_students : ℕ := 303
def selected_students : ℕ := 50

/- Define the probability of a specific student being selected -/
def probability_of_selection : ℚ := selected_students / total_students

/- Theorem statement -/
theorem probability_of_specific_student_selection :
  probability_of_selection = 50 / 303 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_student_selection_l647_64755


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l647_64763

/-- Jessie's weight loss journey -/
theorem jessie_weight_loss (current_weight weight_lost : ℕ) 
  (h1 : current_weight = 66)
  (h2 : weight_lost = 126) : 
  current_weight + weight_lost = 192 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l647_64763


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l647_64719

theorem complex_equation_solutions :
  let f : ℂ → ℂ := λ z => (z^4 - 1) / (z^3 + z^2 - 2*z)
  ∃! (S : Finset ℂ), S.card = 3 ∧ ∀ z ∈ S, f z = 0 ∧ ∀ z ∉ S, f z ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l647_64719


namespace NUMINAMATH_CALUDE_sam_digits_of_pi_l647_64747

theorem sam_digits_of_pi (carlos mina sam : ℕ) : 
  sam = carlos + 6 →
  mina = 6 * carlos →
  mina = 24 →
  sam = 10 := by sorry

end NUMINAMATH_CALUDE_sam_digits_of_pi_l647_64747


namespace NUMINAMATH_CALUDE_sqrt_19_minus_1_between_3_and_4_l647_64721

theorem sqrt_19_minus_1_between_3_and_4 :
  let a := Real.sqrt 19 - 1
  3 < a ∧ a < 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_19_minus_1_between_3_and_4_l647_64721


namespace NUMINAMATH_CALUDE_hexagonal_prism_diagonals_l647_64734

/-- A regular hexagonal prism --/
structure RegularHexagonalPrism where
  /-- Number of sides in the base --/
  n : ℕ
  /-- Assertion that the base has 6 sides --/
  base_is_hexagon : n = 6

/-- The number of diagonals in a regular hexagonal prism --/
def num_diagonals (prism : RegularHexagonalPrism) : ℕ := prism.n * (prism.n - 3)

/-- Theorem: The number of diagonals in a regular hexagonal prism is 18 --/
theorem hexagonal_prism_diagonals (prism : RegularHexagonalPrism) : 
  num_diagonals prism = 18 := by
  sorry

#check hexagonal_prism_diagonals

end NUMINAMATH_CALUDE_hexagonal_prism_diagonals_l647_64734


namespace NUMINAMATH_CALUDE_unique_operation_assignment_l647_64766

-- Define the possible operations
inductive Operation
| Division
| Equal
| Multiplication
| Addition
| Subtraction

-- Define a function to apply an operation
def apply_operation (op : Operation) (x y : ℕ) : Prop :=
  match op with
  | Operation.Division => x / y = 2
  | Operation.Equal => x = y
  | Operation.Multiplication => x * y = 8
  | Operation.Addition => x + y = 5
  | Operation.Subtraction => x - y = 4

-- Define the theorem
theorem unique_operation_assignment :
  ∃! (A B C D E : Operation),
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
     C ≠ D ∧ C ≠ E ∧
     D ≠ E) ∧
    apply_operation A 4 2 ∧
    apply_operation B 2 2 ∧
    apply_operation B 8 (4 * 2) ∧
    apply_operation C 4 2 ∧
    apply_operation D 2 3 ∧
    apply_operation B 5 5 ∧
    apply_operation B 4 (5 - 1) ∧
    apply_operation E 5 1 :=
sorry

end NUMINAMATH_CALUDE_unique_operation_assignment_l647_64766


namespace NUMINAMATH_CALUDE_emily_beads_count_l647_64797

theorem emily_beads_count (beads_per_necklace : ℕ) (necklaces_made : ℕ) (total_beads : ℕ) : 
  beads_per_necklace = 8 → necklaces_made = 2 → total_beads = beads_per_necklace * necklaces_made → total_beads = 16 := by
  sorry

end NUMINAMATH_CALUDE_emily_beads_count_l647_64797


namespace NUMINAMATH_CALUDE_prime_divisibility_l647_64740

theorem prime_divisibility (p q : ℕ) (n : ℕ) 
  (h_p_prime : Prime p) 
  (h_q_prime : Prime q) 
  (h_distinct : p ≠ q) 
  (h_pq_div : (p * q) ∣ (n^(p*q) + 1)) 
  (h_p3q3_div : (p^3 * q^3) ∣ (n^(p*q) + 1)) :
  p^2 ∣ (n + 1) ∨ q^2 ∣ (n + 1) := by
sorry

end NUMINAMATH_CALUDE_prime_divisibility_l647_64740


namespace NUMINAMATH_CALUDE_mod_eight_congruence_l647_64748

theorem mod_eight_congruence (m : ℕ) : 
  12^7 % 8 = m → 0 ≤ m → m < 8 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_eight_congruence_l647_64748


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l647_64771

-- Define the normal distribution parameters
def mean : ℝ := 17.5
def std_dev : ℝ := 2.5

-- Define the value we want to prove
def value : ℝ := 12.5

-- Theorem statement
theorem two_std_dev_below_mean : 
  mean - 2 * std_dev = value := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l647_64771


namespace NUMINAMATH_CALUDE_office_network_connections_l647_64784

/-- Represents a network of switches with their connections -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ

/-- Calculates the total number of connections in the network -/
def total_connections (network : SwitchNetwork) : ℕ :=
  (network.num_switches * network.connections_per_switch) / 2

/-- Theorem stating that a network of 30 switches, each connected to 4 others, has 60 connections -/
theorem office_network_connections :
  let network : SwitchNetwork := { num_switches := 30, connections_per_switch := 4 }
  total_connections network = 60 := by
  sorry

end NUMINAMATH_CALUDE_office_network_connections_l647_64784


namespace NUMINAMATH_CALUDE_smallest_abb_value_l647_64707

theorem smallest_abb_value (A B : Nat) : 
  A ≠ B →
  1 ≤ A ∧ A ≤ 9 →
  1 ≤ B ∧ B ≤ 9 →
  10 * A + B = (100 * A + 11 * B) / 7 →
  ∀ (X Y : Nat), 
    X ≠ Y →
    1 ≤ X ∧ X ≤ 9 →
    1 ≤ Y ∧ Y ≤ 9 →
    10 * X + Y = (100 * X + 11 * Y) / 7 →
    100 * A + 11 * B ≤ 100 * X + 11 * Y →
  100 * A + 11 * B = 466 :=
sorry

end NUMINAMATH_CALUDE_smallest_abb_value_l647_64707


namespace NUMINAMATH_CALUDE_fraction_integer_iff_p_range_l647_64799

theorem fraction_integer_iff_p_range (p : ℕ+) :
  (∃ (k : ℕ+), (3 * p.val + 25 : ℤ) = k.val * (2 * p.val - 5)) ↔ 3 ≤ p.val ∧ p.val ≤ 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_p_range_l647_64799


namespace NUMINAMATH_CALUDE_square_of_difference_l647_64735

theorem square_of_difference (y : ℝ) (h : 4 * y^2 - 36 ≥ 0) :
  (10 - Real.sqrt (4 * y^2 - 36))^2 = 4 * y^2 + 64 - 20 * Real.sqrt (4 * y^2 - 36) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l647_64735


namespace NUMINAMATH_CALUDE_converse_correctness_l647_64703

-- Define the original proposition
def original_prop (a b : ℝ) : Prop := (a^2 + b^2 = 0) → (a = 0 ∧ b = 0)

-- Define the converse proposition
def converse_prop (a b : ℝ) : Prop := (a^2 + b^2 ≠ 0) → (a ≠ 0 ∨ b ≠ 0)

-- Theorem stating that the converse_prop is indeed the converse of original_prop
theorem converse_correctness : 
  ∀ (a b : ℝ), converse_prop a b ↔ (¬(a^2 + b^2 = 0) → ¬(a = 0 ∧ b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_converse_correctness_l647_64703


namespace NUMINAMATH_CALUDE_monkey_feeding_problem_l647_64732

theorem monkey_feeding_problem :
  ∀ (x : ℝ),
    (3/4 * x + 2 = 4/3 * (x - 2)) →
    (3/4 * x + x = 14) :=
by
  sorry

end NUMINAMATH_CALUDE_monkey_feeding_problem_l647_64732


namespace NUMINAMATH_CALUDE_quadratic_minimum_change_l647_64731

/-- Given a quadratic polynomial f(x) = ax^2 + bx + c, 
    if adding x^2 to f(x) increases its minimum value by 1
    and subtracting x^2 from f(x) decreases its minimum value by 3,
    then adding 2x^2 to f(x) will increase its minimum value by 3/2. -/
theorem quadratic_minimum_change 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h_quad : ∀ x, f x = a * x^2 + b * x + c)
  (h_pos : a > 0)
  (h_add : (- b^2 / (4 * (a + 1)) + c) - (- b^2 / (4 * a) + c) = 1)
  (h_sub : (- b^2 / (4 * a) + c) - (- b^2 / (4 * (a - 1)) + c) = 3) :
  (- b^2 / (4 * a) + c) - (- b^2 / (4 * (a + 2)) + c) = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_minimum_change_l647_64731


namespace NUMINAMATH_CALUDE_ben_game_probability_l647_64711

theorem ben_game_probability (p_lose p_tie : ℚ) 
  (h_lose : p_lose = 5 / 11)
  (h_tie : p_tie = 1 / 11)
  (h_total : p_lose + p_tie + (1 - p_lose - p_tie) = 1) :
  1 - p_lose - p_tie = 5 / 11 := by
sorry

end NUMINAMATH_CALUDE_ben_game_probability_l647_64711


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l647_64788

/-- A quadratic function satisfying specific conditions -/
def f (x : ℝ) : ℝ := 2 * (x - 2)^2 - 8

/-- The theorem stating the properties of the quadratic function f -/
theorem quadratic_function_properties :
  (∀ x, f (x + 2) = f (2 - x)) ∧
  (∀ x, f x ≥ -8) ∧
  (f 1 = -6) ∧
  (∀ x ∈ Set.Ioo (-1 : ℝ) 4, -8 ≤ f x ∧ f x < f (-1)) := by
  sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_quadratic_function_properties_l647_64788


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l647_64754

def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a 1 + (n - 1 : ℚ) * (a 2 - a 1)) / 2

theorem arithmetic_sequence_solution 
  (a : ℕ → ℚ) (d : ℚ) (n : ℕ) 
  (h1 : arithmetic_sequence a d)
  (h2 : a 1 + a 3 = 8)
  (h3 : a 4 ^ 2 = a 2 * a 9) :
  ((a 1 = 4 ∧ d = 0 ∧ sum_of_n_terms a n = 4 * n) ∨
   (a 1 = 20 / 9 ∧ d = 16 / 9 ∧ sum_of_n_terms a n = (8 * n^2 + 12 * n) / 9)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l647_64754


namespace NUMINAMATH_CALUDE_base9_to_base3_7254_l647_64720

/-- Converts a single digit from base 9 to its two-digit representation in base 3 -/
def base9_to_base3_digit (d : Nat) : Nat := sorry

/-- Converts a number from base 9 to base 3 -/
def base9_to_base3 (n : Nat) : Nat := sorry

theorem base9_to_base3_7254 :
  base9_to_base3 7254 = 210212113 := by sorry

end NUMINAMATH_CALUDE_base9_to_base3_7254_l647_64720


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l647_64783

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℝ) 
  (group1_size : Nat) 
  (avg_age_group1 : ℝ) 
  (group2_size : Nat) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : group1_size = 4)
  (h4 : avg_age_group1 = 14)
  (h5 : group2_size = 9)
  (h6 : avg_age_group2 = 16)
  : ℝ := by
  sorry

#check fifteenth_student_age

end NUMINAMATH_CALUDE_fifteenth_student_age_l647_64783


namespace NUMINAMATH_CALUDE_skipping_odometer_theorem_l647_64728

/-- Represents an odometer that skips the digit 6 -/
def SkippingOdometer : Type := ℕ

/-- Converts a regular odometer reading to a skipping odometer reading -/
def toSkippingReading (n : ℕ) : SkippingOdometer :=
  sorry

/-- Converts a skipping odometer reading back to the actual distance -/
def toActualDistance (s : SkippingOdometer) : ℕ :=
  sorry

theorem skipping_odometer_theorem :
  toActualDistance (toSkippingReading 1464) = 2005 :=
sorry

end NUMINAMATH_CALUDE_skipping_odometer_theorem_l647_64728


namespace NUMINAMATH_CALUDE_triangle_side_values_l647_64730

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_values :
  ∀ y : ℕ+,
    (triangle_exists 8 11 (y.val ^ 2 - 1) ↔ y.val = 3 ∨ y.val = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_values_l647_64730


namespace NUMINAMATH_CALUDE_unique_perpendicular_tangent_perpendicular_tangent_equation_slope_angle_range_l647_64768

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*x + a

-- Statement for the unique perpendicular tangent line
theorem unique_perpendicular_tangent :
  ∃! a : ℝ, ∃! x : ℝ, f' a x = -1 ∧ a = 3 := by sorry

-- Statement for the equation of the perpendicular tangent line
theorem perpendicular_tangent_equation (a : ℝ) (h : a = 3) :
  ∃ x y : ℝ, 3*x + 3*y - 8 = 0 ∧ y = f a x ∧ f' a x = -1 := by sorry

-- Statement for the range of the slope angle
theorem slope_angle_range (a : ℝ) (h : a = 3) :
  ∀ x : ℝ, -π/4 ≤ Real.arctan (f' a x) ∧ Real.arctan (f' a x) < π/2 := by sorry

end

end NUMINAMATH_CALUDE_unique_perpendicular_tangent_perpendicular_tangent_equation_slope_angle_range_l647_64768


namespace NUMINAMATH_CALUDE_convex_quad_probability_l647_64772

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The probability of forming a convex quadrilateral -/
def prob_convex_quad : ℚ := (n.choose k : ℚ) / (total_chords.choose k : ℚ)

/-- Theorem stating the probability of forming a convex quadrilateral -/
theorem convex_quad_probability : prob_convex_quad = 2 / 585 := by
  sorry

end NUMINAMATH_CALUDE_convex_quad_probability_l647_64772


namespace NUMINAMATH_CALUDE_souvenir_shop_properties_l647_64722

/-- Represents the cost and profit structure of souvenirs --/
structure SouvenirShop where
  costA : ℕ → ℕ  -- Cost function for type A
  costB : ℕ → ℕ  -- Cost function for type B
  profitA : ℕ    -- Profit per piece of type A
  profitB : ℕ    -- Profit per piece of type B

/-- Theorem stating the properties of the souvenir shop problem --/
theorem souvenir_shop_properties (shop : SouvenirShop) :
  (shop.costA 7 + shop.costB 4 = 760) ∧
  (shop.costA 5 + shop.costB 8 = 800) ∧
  (shop.profitA = 30) ∧
  (shop.profitB = 20) →
  (∃ (x y : ℕ), 
    (∀ n : ℕ, shop.costA n = n * x) ∧
    (∀ n : ℕ, shop.costB n = n * y) ∧
    x = 80 ∧ 
    y = 50) ∧
  (∃ (plans : List ℕ),
    plans.length = 7 ∧
    (∀ a ∈ plans, 
      80 * a + 50 * (100 - a) ≥ 7000 ∧
      80 * a + 50 * (100 - a) ≤ 7200)) ∧
  (∃ (maxA : ℕ) (maxB : ℕ),
    maxA + maxB = 100 ∧
    maxA = 73 ∧
    maxB = 27 ∧
    ∀ a b : ℕ, 
      a + b = 100 →
      shop.profitA * a + shop.profitB * b ≤ shop.profitA * maxA + shop.profitB * maxB) :=
by sorry


end NUMINAMATH_CALUDE_souvenir_shop_properties_l647_64722


namespace NUMINAMATH_CALUDE_two_true_statements_l647_64752

theorem two_true_statements : 
  let statements := [
    "Corresponding angles are equal",
    "Perpendicular segments are the shortest",
    "There is only one parallel line passing through a point outside a given line",
    "Rational numbers correspond one-to-one with points on the number line",
    "The integer part of √63 - 1 is 7"
  ]
  ∃ (true_statements : List String), 
    true_statements.length = 2 ∧ 
    true_statements ⊆ statements ∧
    (∀ s ∈ true_statements, s = "Perpendicular segments are the shortest" ∨ 
      s = "There is only one parallel line passing through a point outside a given line") ∧
    (∀ s ∈ statements, s ∉ true_statements → 
      s ≠ "Perpendicular segments are the shortest" ∧ 
      s ≠ "There is only one parallel line passing through a point outside a given line") :=
by sorry

end NUMINAMATH_CALUDE_two_true_statements_l647_64752


namespace NUMINAMATH_CALUDE_equation_equivalence_l647_64701

theorem equation_equivalence (x : ℝ) (Q : ℝ) (h : 5 * (3 * x + 7 * Real.pi) = Q) :
  10 * (6 * x + 14 * Real.pi) = 4 * Q := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l647_64701


namespace NUMINAMATH_CALUDE_same_gender_probability_same_school_probability_l647_64741

structure School where
  male_count : Nat
  female_count : Nat

def total_teachers (s : School) : Nat :=
  s.male_count + s.female_count

def school_A : School :=
  { male_count := 2, female_count := 1 }

def school_B : School :=
  { male_count := 1, female_count := 2 }

def total_schools : Nat := 2

def total_all_teachers : Nat :=
  total_teachers school_A + total_teachers school_B

theorem same_gender_probability :
  (school_A.male_count * school_B.male_count + school_A.female_count * school_B.female_count) /
  (total_teachers school_A * total_teachers school_B) = 4 / 9 := by
  sorry

theorem same_school_probability :
  (Nat.choose (total_teachers school_A) 2 + Nat.choose (total_teachers school_B) 2) /
  Nat.choose total_all_teachers 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_same_gender_probability_same_school_probability_l647_64741


namespace NUMINAMATH_CALUDE_sum_of_zeros_negative_l647_64715

/-- The function f(x) = ln(x) - x + m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - x + m

/-- The function g(x) = f(x+m) -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m (x + m)

/-- Theorem: Given f(x) = ln(x) - x + m, m > 1, g(x) = f(x+m), 
    and x₁, x₂ are zeros of g(x), then x₁ + x₂ < 0 -/
theorem sum_of_zeros_negative (m : ℝ) (x₁ x₂ : ℝ) 
  (hm : m > 1) 
  (hx₁ : g m x₁ = 0) 
  (hx₂ : g m x₂ = 0) : 
  x₁ + x₂ < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_zeros_negative_l647_64715


namespace NUMINAMATH_CALUDE_mary_jenny_red_marbles_equal_l647_64782

/-- Represents the number of marbles collected by each person -/
structure MarbleCollection where
  red : ℕ
  blue : ℕ

/-- Given information about marble collections -/
def problem_setup (mary anie jenny : MarbleCollection) : Prop :=
  mary.blue = anie.blue / 2 ∧
  anie.red = mary.red + 20 ∧
  anie.blue = 2 * jenny.blue ∧
  jenny.red = 30 ∧
  jenny.blue = 25

/-- Theorem stating that Mary and Jenny collected the same number of red marbles -/
theorem mary_jenny_red_marbles_equal 
  (mary anie jenny : MarbleCollection) 
  (h : problem_setup mary anie jenny) : 
  mary.red = jenny.red := by
  sorry

end NUMINAMATH_CALUDE_mary_jenny_red_marbles_equal_l647_64782


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l647_64765

def U : Set ℕ := {1, 3, 5, 7}
def M : Set ℕ := {1, 3}

theorem complement_of_M_in_U :
  (U \ M) = {5, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l647_64765


namespace NUMINAMATH_CALUDE_complete_square_expression_l647_64753

theorem complete_square_expression (y : ℝ) : 
  ∃ (k : ℤ) (b : ℝ), y^2 + 16*y + 60 = (y + b)^2 + k ∧ k = -4 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_expression_l647_64753


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l647_64743

-- Define sets M and N
def M (a : ℝ) : Set ℝ := {3, 2*a}
def N (a : ℝ) : Set ℝ := {a+1, 3}

-- State the theorem
theorem subset_implies_a_equals_one :
  ∀ a : ℝ, M a ⊆ N a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l647_64743


namespace NUMINAMATH_CALUDE_fiona_probability_l647_64791

/-- Represents a lily pad with its number and whether it contains a predator or food -/
structure LilyPad where
  number : Nat
  hasPredator : Bool
  hasFood : Bool

/-- Represents the possible moves Fiona can make -/
inductive Move
  | Forward
  | ForwardTwo
  | Backward

/-- Represents Fiona's current position and the probability of reaching that position -/
structure FionaState where
  position : Nat
  probability : Rat

/-- The probability of each move -/
def moveProbability : Rat := 1 / 3

/-- The total number of lily pads -/
def totalPads : Nat := 15

/-- Creates the initial state of the lily pads -/
def initLilyPads : List LilyPad := sorry

/-- Checks if a move is valid given Fiona's current position -/
def isValidMove (currentPos : Nat) (move : Move) : Bool := sorry

/-- Calculates Fiona's new position after a move -/
def newPosition (currentPos : Nat) (move : Move) : Nat := sorry

/-- Calculates the probability of Fiona reaching pad 13 without landing on pads 4 or 8 -/
def probReachPad13 (initialState : FionaState) (lilyPads : List LilyPad) : Rat := sorry

theorem fiona_probability :
  probReachPad13 ⟨0, 1⟩ initLilyPads = 16 / 177147 := by sorry

end NUMINAMATH_CALUDE_fiona_probability_l647_64791


namespace NUMINAMATH_CALUDE_point_coordinates_l647_64749

/-- Given a point P in the Cartesian coordinate system, prove its coordinates. -/
theorem point_coordinates :
  ∀ m : ℝ,
  let P : ℝ × ℝ := (-m - 1, 2 * m + 1)
  (P.1 < 0 ∧ P.2 > 0) →  -- P is in the second quadrant
  P.2 = 5 →              -- Distance from M to x-axis is 5
  P = (-3, 5) :=         -- Coordinates of P are (-3, 5)
by
  sorry


end NUMINAMATH_CALUDE_point_coordinates_l647_64749


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l647_64790

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 6 ∧ x ≠ -3 →
    (4 * x + 8) / (x^2 - 3*x - 18) = 32 / (9 * (x - 6)) + 4 / (9 * (x + 3)) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l647_64790


namespace NUMINAMATH_CALUDE_calculation_proof_l647_64792

theorem calculation_proof : ((4 + 6 + 5) * 2) / 4 - (3 * 2 / 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l647_64792


namespace NUMINAMATH_CALUDE_probability_walk_300_or_less_l647_64757

/-- Represents an airport with gates in a straight line. -/
structure Airport where
  num_gates : ℕ
  gate_distance : ℝ

/-- Calculates the number of gate pairs within a given distance. -/
def count_gate_pairs_within_distance (a : Airport) (max_distance : ℝ) : ℕ :=
  sorry

/-- Calculates the total number of possible gate pair assignments. -/
def total_gate_pairs (a : Airport) : ℕ :=
  sorry

/-- The main theorem stating the probability of walking 300 feet or less. -/
theorem probability_walk_300_or_less (a : Airport) :
  a.num_gates = 16 ∧ a.gate_distance = 75 →
  (count_gate_pairs_within_distance a 300 : ℚ) / (total_gate_pairs a : ℚ) = 9 / 20 :=
by sorry

end NUMINAMATH_CALUDE_probability_walk_300_or_less_l647_64757


namespace NUMINAMATH_CALUDE_lcm_of_20_28_45_l647_64713

theorem lcm_of_20_28_45 : Nat.lcm (Nat.lcm 20 28) 45 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_28_45_l647_64713


namespace NUMINAMATH_CALUDE_min_tries_for_blue_and_yellow_is_nine_l647_64798

/-- Represents the number of balls of each color in the box -/
structure BoxContents where
  purple : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the minimum number of tries to get one blue and one yellow ball -/
def minTriesForBlueAndYellow (box : BoxContents) : Nat :=
  box.purple + 2

/-- Theorem stating the minimum number of tries for the given box contents -/
theorem min_tries_for_blue_and_yellow_is_nine :
  let box : BoxContents := { purple := 7, blue := 5, yellow := 11 }
  minTriesForBlueAndYellow box = 9 := by
  sorry


end NUMINAMATH_CALUDE_min_tries_for_blue_and_yellow_is_nine_l647_64798


namespace NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l647_64725

/-- The cost ratio of a muffin to a banana given purchase information -/
theorem muffin_banana_cost_ratio :
  ∀ (m b : ℝ), 
    m > 0 →  -- m is positive (cost of muffin)
    b > 0 →  -- b is positive (cost of banana)
    5 * m + 2 * b > 0 →  -- Susie's purchase is positive
    3 * (5 * m + 2 * b) = 4 * m + 10 * b →  -- Jason's purchase is 3 times Susie's
    m / b = 4 / 11 :=
by
  sorry

end NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l647_64725


namespace NUMINAMATH_CALUDE_square_of_binomial_coefficient_l647_64706

theorem square_of_binomial_coefficient (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 - 18*x + a = (3*x + b)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_coefficient_l647_64706


namespace NUMINAMATH_CALUDE_living_room_area_l647_64738

/-- Given a rectangular carpet covering 60% of a room's floor area,
    if the carpet measures 4 feet by 9 feet,
    then the total floor area of the room is 60 square feet. -/
theorem living_room_area
  (carpet_length : ℝ)
  (carpet_width : ℝ)
  (carpet_coverage : ℝ)
  (h1 : carpet_length = 4)
  (h2 : carpet_width = 9)
  (h3 : carpet_coverage = 0.6)
  : carpet_length * carpet_width / carpet_coverage = 60 := by
  sorry

end NUMINAMATH_CALUDE_living_room_area_l647_64738


namespace NUMINAMATH_CALUDE_min_max_values_of_f_l647_64778

def f (x : ℝ) : ℝ := 4 * x^2 + 8 * x + 16

theorem min_max_values_of_f :
  let a := -3
  let b := 2
  ∃ (x_min x_max : ℝ), a ≤ x_min ∧ x_min ≤ b ∧ a ≤ x_max ∧ x_max ≤ b ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x_min ≤ f x) ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f x_max) ∧
    f x_min = 12 ∧ f x_max = 48 :=
by sorry

end NUMINAMATH_CALUDE_min_max_values_of_f_l647_64778


namespace NUMINAMATH_CALUDE_wanda_blocks_count_l647_64702

/-- The total number of blocks Wanda has after receiving more blocks from Theresa -/
def total_blocks (initial : ℕ) (additional : ℕ) : ℕ := initial + additional

/-- Theorem stating that given the initial and additional blocks, Wanda has 83 blocks in total -/
theorem wanda_blocks_count : total_blocks 4 79 = 83 := by
  sorry

end NUMINAMATH_CALUDE_wanda_blocks_count_l647_64702


namespace NUMINAMATH_CALUDE_alien_martian_limb_difference_l647_64793

/-- The number of arms an Alien has -/
def alien_arms : ℕ := 3

/-- The number of legs an Alien has -/
def alien_legs : ℕ := 8

/-- The number of arms a Martian has -/
def martian_arms : ℕ := 2 * alien_arms

/-- The number of legs a Martian has -/
def martian_legs : ℕ := alien_legs / 2

/-- The total number of limbs for one Alien -/
def alien_limbs : ℕ := alien_arms + alien_legs

/-- The total number of limbs for one Martian -/
def martian_limbs : ℕ := martian_arms + martian_legs

/-- The number of Aliens and Martians we're comparing -/
def number_of_creatures : ℕ := 5

theorem alien_martian_limb_difference :
  number_of_creatures * alien_limbs - number_of_creatures * martian_limbs = 5 := by
  sorry

end NUMINAMATH_CALUDE_alien_martian_limb_difference_l647_64793


namespace NUMINAMATH_CALUDE_cheese_block_volume_l647_64737

/-- Given a normal block of cheese with volume 3 cubic feet, 
    a large block with twice the width, twice the depth, and three times the length 
    of the normal block will have a volume of 36 cubic feet. -/
theorem cheese_block_volume : 
  ∀ (w d l : ℝ), 
    w * d * l = 3 → 
    (2 * w) * (2 * d) * (3 * l) = 36 := by
  sorry

end NUMINAMATH_CALUDE_cheese_block_volume_l647_64737


namespace NUMINAMATH_CALUDE_point_B_coordinates_l647_64775

-- Define the point A
def A : ℝ × ℝ := (-3, 2)

-- Define the length of AB
def AB_length : ℝ := 4

-- Define the possible coordinates of point B
def B1 : ℝ × ℝ := (-7, 2)
def B2 : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem point_B_coordinates :
  ∀ B : ℝ × ℝ,
  (B.2 = A.2) →                        -- AB is parallel to x-axis
  ((B.1 - A.1)^2 + (B.2 - A.2)^2 = AB_length^2) →  -- Length of AB is 4
  (B = B1 ∨ B = B2) :=
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_point_B_coordinates_l647_64775


namespace NUMINAMATH_CALUDE_intersection_M_N_l647_64745

def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 + 1}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 1}

theorem intersection_M_N : M ∩ N = {(0, 1), (1, 2)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l647_64745


namespace NUMINAMATH_CALUDE_min_fence_posts_for_field_l647_64770

/-- Calculates the number of fence posts needed for a rectangular field -/
def fence_posts (length width post_spacing_long post_spacing_short : ℕ) : ℕ :=
  let long_side_posts := length / post_spacing_long + 1
  let short_side_posts := width / post_spacing_short + 1
  long_side_posts + 2 * (short_side_posts - 1)

/-- Theorem stating the minimum number of fence posts required for the given field -/
theorem min_fence_posts_for_field : 
  fence_posts 150 50 15 10 = 21 :=
by sorry

end NUMINAMATH_CALUDE_min_fence_posts_for_field_l647_64770


namespace NUMINAMATH_CALUDE_M_remainder_1000_l647_64774

/-- The greatest integer multiple of 9 with no two digits being the same -/
def M : ℕ :=
  sorry

/-- M has no repeated digits -/
axiom M_distinct_digits : ∀ d₁ d₂, d₁ ≠ d₂ → (M / 10^d₁ % 10) ≠ (M / 10^d₂ % 10)

/-- M is divisible by 9 -/
axiom M_div_by_9 : M % 9 = 0

/-- M is the greatest such number -/
axiom M_greatest : ∀ n : ℕ, n % 9 = 0 → (∀ d₁ d₂, d₁ ≠ d₂ → (n / 10^d₁ % 10) ≠ (n / 10^d₂ % 10)) → n ≤ M

theorem M_remainder_1000 : M % 1000 = 810 := by
  sorry

end NUMINAMATH_CALUDE_M_remainder_1000_l647_64774


namespace NUMINAMATH_CALUDE_bracket_mult_example_bracket_mult_equation_roots_l647_64724

-- Define the operation for real numbers
def bracket_mult (a b c d : ℝ) : ℝ := a * c - b * d

-- Theorem 1
theorem bracket_mult_example : bracket_mult (-4) 3 2 (-6) = 10 := by sorry

-- Theorem 2
theorem bracket_mult_equation_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ bracket_mult x (2*x - 1) (m*x + 1) m = 0) ↔ 
  (m ≤ 1/4 ∧ m ≠ 0) := by sorry

end NUMINAMATH_CALUDE_bracket_mult_example_bracket_mult_equation_roots_l647_64724


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l647_64709

def M : Set ℕ := {1, 2, 4, 8, 16}
def N : Set ℕ := {2, 4, 6, 8}

theorem intersection_of_M_and_N : M ∩ N = {2, 4, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l647_64709


namespace NUMINAMATH_CALUDE_five_digit_number_divisible_by_37_and_173_l647_64751

theorem five_digit_number_divisible_by_37_and_173 (n : ℕ) : 
  (n ≥ 10000 ∧ n < 100000) →  -- five-digit number
  n % 37 = 0 →  -- divisible by 37
  n % 173 = 0 →  -- divisible by 173
  (n / 1000) % 10 = 3 →  -- thousands digit is 3
  (n / 100) % 10 = 2  -- hundreds digit is 2
  := by sorry

end NUMINAMATH_CALUDE_five_digit_number_divisible_by_37_and_173_l647_64751


namespace NUMINAMATH_CALUDE_walter_chores_l647_64700

theorem walter_chores (total_days : ℕ) (regular_pay : ℕ) (exceptional_pay : ℕ) (total_earnings : ℕ) :
  total_days = 15 ∧ regular_pay = 4 ∧ exceptional_pay = 6 ∧ total_earnings = 78 →
  ∃ (regular_days exceptional_days : ℕ),
    regular_days + exceptional_days = total_days ∧
    regular_days * regular_pay + exceptional_days * exceptional_pay = total_earnings ∧
    exceptional_days = 9 :=
by sorry

end NUMINAMATH_CALUDE_walter_chores_l647_64700


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l647_64712

theorem power_fraction_simplification :
  (3^2023 + 3^2021) / (3^2023 - 3^2021) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l647_64712


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l647_64795

theorem necessary_but_not_sufficient (a b c d : ℝ) :
  ((a > b ∧ c > d) → a + c > b + d) ∧
  (∃ a b c d : ℝ, a + c > b + d ∧ ¬(a > b ∧ c > d)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l647_64795


namespace NUMINAMATH_CALUDE_given_square_is_magic_l647_64764

/-- Represents a 3x3 magic square -/
def MagicSquare : Type := Fin 3 → Fin 3 → ℕ

/-- Converts a number from base 5 to base 10 -/
def toBase10 (n : ℕ) : ℕ := 
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 5 => 5
  | 6 => 6
  | 10 => 5
  | 20 => 10
  | 21 => 11
  | 22 => 12
  | 23 => 13
  | _ => n  -- Default case

/-- The given magic square -/
def givenSquare : MagicSquare :=
  fun i j => match i, j with
    | 0, 0 => 22
    | 0, 1 => 2
    | 0, 2 => 20
    | 1, 0 => 5
    | 1, 1 => 10
    | 1, 2 => 21
    | 2, 0 => 6
    | 2, 1 => 23
    | 2, 2 => 3

/-- Sum of a row in the magic square -/
def rowSum (s : MagicSquare) (i : Fin 3) : ℕ :=
  (toBase10 (s i 0)) + (toBase10 (s i 1)) + (toBase10 (s i 2))

/-- Sum of a column in the magic square -/
def colSum (s : MagicSquare) (j : Fin 3) : ℕ :=
  (toBase10 (s 0 j)) + (toBase10 (s 1 j)) + (toBase10 (s 2 j))

/-- Sum of the main diagonal of the magic square -/
def mainDiagSum (s : MagicSquare) : ℕ :=
  (toBase10 (s 0 0)) + (toBase10 (s 1 1)) + (toBase10 (s 2 2))

/-- Sum of the other diagonal of the magic square -/
def otherDiagSum (s : MagicSquare) : ℕ :=
  (toBase10 (s 0 2)) + (toBase10 (s 1 1)) + (toBase10 (s 2 0))

/-- Theorem: The given square is a magic square when interpreted in base 5 -/
theorem given_square_is_magic : 
  (∀ i : Fin 3, rowSum givenSquare i = 21) ∧ 
  (∀ j : Fin 3, colSum givenSquare j = 21) ∧ 
  mainDiagSum givenSquare = 21 ∧ 
  otherDiagSum givenSquare = 21 := by
  sorry

end NUMINAMATH_CALUDE_given_square_is_magic_l647_64764


namespace NUMINAMATH_CALUDE_g_inverse_sum_l647_64796

/-- The function g(x) defined piecewise -/
noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then c * x + d else 10 - 4 * x

/-- Theorem stating that c + d = 7.25 given the conditions -/
theorem g_inverse_sum (c d : ℝ) :
  (∀ x, g c d (g c d x) = x) →
  c + d = 7.25 := by
  sorry

end NUMINAMATH_CALUDE_g_inverse_sum_l647_64796


namespace NUMINAMATH_CALUDE_sector_properties_l647_64758

-- Define the sector
def Sector (R : ℝ) (α : ℝ) : Prop :=
  R > 0 ∧ α > 0 ∧ (1 / 2) * R^2 * α = 1 ∧ 2 * R + R * α = 4

-- Theorem statement
theorem sector_properties :
  ∃ (R α : ℝ), Sector R α ∧ α = 2 ∧ 2 * Real.sin 1 = 2 * R * Real.sin (α / 2) :=
sorry

end NUMINAMATH_CALUDE_sector_properties_l647_64758


namespace NUMINAMATH_CALUDE_kelvin_expected_score_l647_64716

/-- Represents the coin flipping game --/
structure CoinGame where
  /-- The number of coins Kelvin starts with --/
  initialCoins : ℕ
  /-- The probability of getting heads on a single coin flip --/
  headsProbability : ℝ

/-- Calculates the expected score for the game --/
noncomputable def expectedScore (game : CoinGame) : ℝ :=
  sorry

/-- Theorem stating the expected score for Kelvin's specific game --/
theorem kelvin_expected_score :
  let game : CoinGame := { initialCoins := 2, headsProbability := 1/2 }
  expectedScore game = 64/9 := by
  sorry

end NUMINAMATH_CALUDE_kelvin_expected_score_l647_64716


namespace NUMINAMATH_CALUDE_inverse_at_five_l647_64786

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 1

-- State that f has an inverse
def f_inv : ℝ → ℝ := sorry

-- Assume f_inv is the inverse of f
axiom f_inverse (x : ℝ) : f (f_inv x) = x
axiom inv_f (x : ℝ) : f_inv (f x) = x

-- Theorem to prove
theorem inverse_at_five : f_inv 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_at_five_l647_64786


namespace NUMINAMATH_CALUDE_mongolian_olympiad_inequality_l647_64773

theorem mongolian_olympiad_inequality 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a^4 + b^4 + c^4 + a^2/(b+c)^2 + b^2/(c+a)^2 + c^2/(a+b)^2 ≥ a*b + b*c + c*a :=
by sorry

end NUMINAMATH_CALUDE_mongolian_olympiad_inequality_l647_64773


namespace NUMINAMATH_CALUDE_polygon_quadrilateral_iff_exterior_eq_interior_l647_64756

/-- A polygon is a quadrilateral if and only if the sum of its exterior angles
    equals the sum of its interior angles. -/
theorem polygon_quadrilateral_iff_exterior_eq_interior :
  ∀ n : ℕ, n ≥ 3 →
  (n = 4 ↔ (n - 2) * 180 = 360) :=
by sorry

end NUMINAMATH_CALUDE_polygon_quadrilateral_iff_exterior_eq_interior_l647_64756


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l647_64733

theorem shaded_fraction_of_rectangle (length width : ℕ) (h1 : length = 15) (h2 : width = 24) :
  let total_area := length * width
  let third_area := total_area / 3
  let shaded_area := third_area / 3
  (shaded_area : ℚ) / total_area = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l647_64733


namespace NUMINAMATH_CALUDE_extreme_value_of_f_l647_64789

-- Define the function
def f (x : ℝ) : ℝ := (x^2 - 1)^3 + 1

-- State the theorem
theorem extreme_value_of_f :
  ∃ (e : ℝ), e = 0 ∧ ∀ (x : ℝ), f x ≥ e :=
sorry

end NUMINAMATH_CALUDE_extreme_value_of_f_l647_64789


namespace NUMINAMATH_CALUDE_tangent_segment_region_area_l647_64762

theorem tangent_segment_region_area (r : ℝ) (l : ℝ) (h1 : r = 3) (h2 : l = 6) : 
  let outer_radius := r * Real.sqrt 2
  let area := π * (outer_radius^2 - r^2)
  area = 9 * π := by sorry

end NUMINAMATH_CALUDE_tangent_segment_region_area_l647_64762


namespace NUMINAMATH_CALUDE_billy_weight_l647_64769

theorem billy_weight (carl_weight brad_weight billy_weight : ℕ) 
  (h1 : brad_weight = carl_weight + 5)
  (h2 : billy_weight = brad_weight + 9)
  (h3 : carl_weight = 145) :
  billy_weight = 159 := by
  sorry

end NUMINAMATH_CALUDE_billy_weight_l647_64769
