import Mathlib

namespace NUMINAMATH_CALUDE_no_solution_quadratic_l3652_365297

theorem no_solution_quadratic (p q r s : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + p*x + q ≠ 0)
  (h2 : ∀ x : ℝ, x^2 + r*x + s ≠ 0) :
  ∀ x : ℝ, 2017*x^2 + (1009*p + 1008*r)*x + 1009*q + 1008*s ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_solution_quadratic_l3652_365297


namespace NUMINAMATH_CALUDE_remainder_of_3_power_500_mod_17_l3652_365252

theorem remainder_of_3_power_500_mod_17 : 3^500 % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_power_500_mod_17_l3652_365252


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l3652_365244

theorem inequality_solution_implies_m_value : 
  ∀ m : ℝ, 
  (∀ x : ℝ, 0 < x ∧ x < 2 ↔ -1/2 * x^2 + 2*x > m*x) → 
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l3652_365244


namespace NUMINAMATH_CALUDE_cubic_function_monotonicity_l3652_365259

-- Define the function f(x) = ax^3
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3

-- State the theorem
theorem cubic_function_monotonicity (a : ℝ) (h : a ≠ 0) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (a > 0 → f a x₁ < f a x₂) ∧ (a < 0 → f a x₁ > f a x₂)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_monotonicity_l3652_365259


namespace NUMINAMATH_CALUDE_residue_mod_17_l3652_365209

theorem residue_mod_17 : (195 * 15 - 18 * 8 + 4) % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_17_l3652_365209


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3652_365294

theorem complex_equation_solution :
  let z : ℂ := ((1 + Complex.I)^2 + 3*(1 - Complex.I)) / (2 + Complex.I)
  ∀ a b : ℝ,
  z^2 + a*z + b = 1 + Complex.I →
  a = -3 ∧ b = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3652_365294


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l3652_365260

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (winner_percentage : ℚ)
  (loser_votes : ℕ)
  (h1 : total_votes = 7500)
  (h2 : winner_percentage = 55 / 100)
  (h3 : loser_votes = 2700) :
  (total_votes - (loser_votes / (1 - winner_percentage))) / total_votes = 1 / 5 :=
sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l3652_365260


namespace NUMINAMATH_CALUDE_ellipse_incircle_area_l3652_365239

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-4, 0)
def F2 : ℝ × ℝ := (4, 0)

-- Define collinearity
def collinear (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

-- Define the theorem
theorem ellipse_incircle_area (x1 y1 x2 y2 : ℝ) :
  is_on_ellipse x1 y1 →
  is_on_ellipse x2 y2 →
  collinear (x1, y1) (x2, y2) F1 →
  (area_incircle_ABF2 : ℝ) →
  area_incircle_ABF2 = 4 * Real.pi →
  |y1 - y2| = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ellipse_incircle_area_l3652_365239


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3652_365241

/-- The complex number (1-i)^2 / (1+i) lies in the third quadrant of the complex plane -/
theorem complex_number_in_third_quadrant :
  let z : ℂ := (1 - Complex.I)^2 / (1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3652_365241


namespace NUMINAMATH_CALUDE_arctan_sum_equation_l3652_365227

theorem arctan_sum_equation (y : ℝ) : 
  2 * Real.arctan (1/3) + 2 * Real.arctan (1/15) + Real.arctan (1/y) = π/2 → y = 261/242 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equation_l3652_365227


namespace NUMINAMATH_CALUDE_chess_team_selection_l3652_365266

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem chess_team_selection :
  let total_boys : ℕ := 8
  let total_girls : ℕ := 10
  let boys_to_select : ℕ := 5
  let girls_to_select : ℕ := 3
  (choose total_boys boys_to_select) * (choose total_girls girls_to_select) = 6720 := by
sorry

end NUMINAMATH_CALUDE_chess_team_selection_l3652_365266


namespace NUMINAMATH_CALUDE_t_shirts_per_package_l3652_365230

theorem t_shirts_per_package (total_t_shirts : ℕ) (num_packages : ℕ) 
  (h1 : total_t_shirts = 39)
  (h2 : num_packages = 3)
  (h3 : total_t_shirts % num_packages = 0) :
  total_t_shirts / num_packages = 13 := by
  sorry

end NUMINAMATH_CALUDE_t_shirts_per_package_l3652_365230


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3652_365221

theorem trigonometric_identity : 
  2 * (Real.sin (35 * π / 180) * Real.cos (25 * π / 180) + 
       Real.cos (35 * π / 180) * Real.cos (65 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3652_365221


namespace NUMINAMATH_CALUDE_percentage_relation_l3652_365207

theorem percentage_relation (x y z : ℝ) : 
  x = 1.2 * y ∧ y = 0.5 * z → x = 0.6 * z := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3652_365207


namespace NUMINAMATH_CALUDE_karlson_candy_theorem_l3652_365228

/-- The maximum number of candies Karlson can eat given n initial units -/
def max_candies (n : ℕ) : ℕ := Nat.choose n 2

/-- The theorem stating that for 31 initial units, the maximum number of candies is 465 -/
theorem karlson_candy_theorem :
  max_candies 31 = 465 := by
  sorry

end NUMINAMATH_CALUDE_karlson_candy_theorem_l3652_365228


namespace NUMINAMATH_CALUDE_third_car_manufacture_year_l3652_365231

def year_first_car : ℕ := 1970
def years_between_first_and_second : ℕ := 10
def years_between_second_and_third : ℕ := 20

theorem third_car_manufacture_year :
  year_first_car + years_between_first_and_second + years_between_second_and_third = 2000 := by
  sorry

end NUMINAMATH_CALUDE_third_car_manufacture_year_l3652_365231


namespace NUMINAMATH_CALUDE_tank_capacity_l3652_365295

theorem tank_capacity (initial_fraction : Rat) (final_fraction : Rat) (used_gallons : ℕ) 
  (h1 : initial_fraction = 3/4)
  (h2 : final_fraction = 1/3)
  (h3 : used_gallons = 18) :
  ∃ (capacity : ℕ), 
    capacity * initial_fraction - capacity * final_fraction = used_gallons ∧ 
    capacity = 43 := by
sorry


end NUMINAMATH_CALUDE_tank_capacity_l3652_365295


namespace NUMINAMATH_CALUDE_total_spending_equals_49_l3652_365211

/-- Represents the total amount spent by Paula and Olive at the kiddy gift shop -/
def total_spent (bracelet_price keychain_price coloring_book_price sticker_price toy_car_price : ℕ)
  (paula_bracelets paula_keychains paula_coloring_books paula_stickers : ℕ)
  (olive_coloring_books olive_bracelets olive_toy_cars olive_stickers : ℕ) : ℕ :=
  (bracelet_price * (paula_bracelets + olive_bracelets)) +
  (keychain_price * paula_keychains) +
  (coloring_book_price * (paula_coloring_books + olive_coloring_books)) +
  (sticker_price * (paula_stickers + olive_stickers)) +
  (toy_car_price * olive_toy_cars)

/-- Theorem stating that Paula and Olive's total spending equals $49 -/
theorem total_spending_equals_49 :
  total_spent 4 5 3 1 6 3 2 1 4 1 2 1 3 = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_spending_equals_49_l3652_365211


namespace NUMINAMATH_CALUDE_function_inequality_l3652_365214

/-- Given a differentiable function f: ℝ → ℝ such that f'(x) + f(x) < 0 for all x in ℝ,
    prove that f(m-m^2) / e^(m^2-m+1) > f(1) for all m in ℝ. -/
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h : ∀ x, deriv f x + f x < 0) :
    ∀ m, f (m - m^2) / Real.exp (m^2 - m + 1) > f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3652_365214


namespace NUMINAMATH_CALUDE_intersection_implies_p_value_l3652_365279

noncomputable section

-- Define the curves C₁ and C₂
def C₁ (p : ℝ) (t : ℝ) : ℝ × ℝ := (2 * p * t, 2 * p * Real.sqrt t)
def C₂ (θ : ℝ) : ℝ := 4 * Real.sin θ

-- Define the distance between two points
def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- State the theorem
theorem intersection_implies_p_value (p : ℝ) :
  p > 0 →
  ∃ (A B : ℝ × ℝ) (t₁ t₂ θ₁ θ₂ : ℝ),
    C₁ p t₁ = A ∧
    C₁ p t₂ = B ∧
    C₂ θ₁ = Real.sqrt (A.1^2 + A.2^2) ∧
    C₂ θ₂ = Real.sqrt (B.1^2 + B.2^2) ∧
    distance A B = 2 * Real.sqrt 3 →
    p = 3 * Real.sqrt 3 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_intersection_implies_p_value_l3652_365279


namespace NUMINAMATH_CALUDE_root_less_than_one_l3652_365248

theorem root_less_than_one (p q x₁ x₂ : ℝ) : 
  x₁^2 + p*x₁ - q = 0 →
  x₂^2 + p*x₂ - q = 0 →
  x₁ > 1 →
  p + q + 3 > 0 →
  x₂ < 1 :=
by sorry

end NUMINAMATH_CALUDE_root_less_than_one_l3652_365248


namespace NUMINAMATH_CALUDE_tournament_handshakes_count_l3652_365208

/-- Calculates the total number of handshakes in a basketball tournament -/
def tournament_handshakes (num_teams : Nat) (players_per_team : Nat) (num_referees : Nat) : Nat :=
  let total_players := num_teams * players_per_team
  let player_handshakes := (total_players * (total_players - players_per_team)) / 2
  let referee_handshakes := total_players * num_referees
  player_handshakes + referee_handshakes

/-- Theorem: In a tournament with 3 teams of 7 players each and 3 referees, 
    there are 210 handshakes in total -/
theorem tournament_handshakes_count :
  tournament_handshakes 3 7 3 = 210 := by
  sorry

end NUMINAMATH_CALUDE_tournament_handshakes_count_l3652_365208


namespace NUMINAMATH_CALUDE_perpendicular_line_coordinates_l3652_365238

/-- Given two points P and Q in a 2D plane, where Q has fixed coordinates
    and P's coordinates depend on a parameter 'a', prove that if the line PQ
    is perpendicular to the y-axis, then P has specific coordinates. -/
theorem perpendicular_line_coordinates 
  (Q : ℝ × ℝ) 
  (P : ℝ → ℝ × ℝ) 
  (h1 : Q = (2, -3))
  (h2 : ∀ a, P a = (2*a + 2, a - 5))
  (h3 : ∀ a, (P a).1 = Q.1) :
  ∃ a, P a = (6, -3) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_coordinates_l3652_365238


namespace NUMINAMATH_CALUDE_triangle_existence_l3652_365276

theorem triangle_existence (a b c A B C : ℝ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : A > 0 ∧ B > 0 ∧ C > 0)
  (h3 : a + b > c ∧ b + c > a ∧ c + a > b)
  (h4 : A + B > C ∧ B + C > A ∧ C + A > B) :
  ∃ (x y z : ℝ), 
    x = Real.sqrt (a^2 + A^2) ∧
    y = Real.sqrt (b^2 + B^2) ∧
    z = Real.sqrt (c^2 + C^2) ∧
    x + y > z ∧ y + z > x ∧ z + x > y :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_l3652_365276


namespace NUMINAMATH_CALUDE_max_a4_in_geometric_sequence_l3652_365258

/-- A positive geometric sequence -/
def IsPositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = r * a n ∧ a n > 0

/-- The theorem statement -/
theorem max_a4_in_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : IsPositiveGeometricSequence a)
  (h_sum : a 3 + a 5 = 4) :
  ∀ b : ℝ, a 4 ≤ b → b ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_a4_in_geometric_sequence_l3652_365258


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l3652_365269

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l3652_365269


namespace NUMINAMATH_CALUDE_coprime_product_and_sum_l3652_365224

theorem coprime_product_and_sum (a b : ℤ) (h : Nat.Coprime a.natAbs b.natAbs) :
  Nat.Coprime (a * b).natAbs (a + b).natAbs := by
  sorry

end NUMINAMATH_CALUDE_coprime_product_and_sum_l3652_365224


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_implies_a_value_l3652_365233

theorem infinitely_many_solutions_implies_a_value 
  (a b : ℝ) 
  (h : ∀ x : ℝ, 2*a*(x-1) = (5-a)*x + 3*b) :
  a = 5/3 := by
sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_implies_a_value_l3652_365233


namespace NUMINAMATH_CALUDE_total_defective_rate_proof_l3652_365271

/-- The fraction of products checked by worker y -/
def worker_y_fraction : ℝ := 0.1666666666666668

/-- The defective rate for products checked by worker x -/
def worker_x_defective_rate : ℝ := 0.005

/-- The defective rate for products checked by worker y -/
def worker_y_defective_rate : ℝ := 0.008

/-- The total defective rate for all products -/
def total_defective_rate : ℝ := 0.0055

theorem total_defective_rate_proof :
  (1 - worker_y_fraction) * worker_x_defective_rate +
  worker_y_fraction * worker_y_defective_rate = total_defective_rate := by
  sorry

end NUMINAMATH_CALUDE_total_defective_rate_proof_l3652_365271


namespace NUMINAMATH_CALUDE_number_division_and_addition_l3652_365216

theorem number_division_and_addition (x : ℝ) : x / 9 = 8 → x + 11 = 83 := by
  sorry

end NUMINAMATH_CALUDE_number_division_and_addition_l3652_365216


namespace NUMINAMATH_CALUDE_perfect_cubes_difference_l3652_365274

theorem perfect_cubes_difference (n : ℕ) : 
  (∃ x y : ℕ, (n + 195 = x^3) ∧ (n - 274 = y^3)) ↔ n = 2002 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cubes_difference_l3652_365274


namespace NUMINAMATH_CALUDE_minimum_newspapers_to_recover_cost_l3652_365263

/-- The cost of Mary's scooter in dollars -/
def scooter_cost : ℕ := 3000

/-- The amount Mary earns per newspaper delivered in dollars -/
def earning_per_newspaper : ℕ := 8

/-- The transportation cost per newspaper delivery in dollars -/
def transport_cost_per_newspaper : ℕ := 4

/-- The net earning per newspaper in dollars -/
def net_earning_per_newspaper : ℕ := earning_per_newspaper - transport_cost_per_newspaper

theorem minimum_newspapers_to_recover_cost :
  ∃ n : ℕ, n * net_earning_per_newspaper ≥ scooter_cost ∧
  ∀ m : ℕ, m * net_earning_per_newspaper ≥ scooter_cost → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_minimum_newspapers_to_recover_cost_l3652_365263


namespace NUMINAMATH_CALUDE_greatest_k_value_l3652_365281

theorem greatest_k_value (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + k*x₁ + 5 = 0 ∧ 
    x₂^2 + k*x₂ + 5 = 0 ∧ 
    |x₁ - x₂| = Real.sqrt 61) →
  k ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l3652_365281


namespace NUMINAMATH_CALUDE_correct_ranking_l3652_365236

/-- Represents a contestant's score -/
structure Score where
  value : ℝ
  positive : value > 0

/-- Represents the scores of the four contestants -/
structure ContestScores where
  ann : Score
  bill : Score
  carol : Score
  dick : Score
  sum_equality : bill.value + dick.value = ann.value + carol.value
  interchange_inequality : carol.value + bill.value > dick.value + ann.value
  carol_exceeds_sum : carol.value > ann.value + bill.value

/-- Represents the ranking of contestants -/
inductive Ranking
  | CDBA : Ranking  -- Carol, Dick, Bill, Ann
  | CDAB : Ranking  -- Carol, Dick, Ann, Bill
  | DCBA : Ranking  -- Dick, Carol, Bill, Ann
  | ACDB : Ranking  -- Ann, Carol, Dick, Bill
  | DCAB : Ranking  -- Dick, Carol, Ann, Bill

/-- The theorem stating that given the contest conditions, the correct ranking is CDBA -/
theorem correct_ranking (scores : ContestScores) : Ranking.CDBA = 
  (match scores with
  | ⟨ann, bill, carol, dick, _, _, _⟩ => 
      if carol.value > dick.value ∧ dick.value > bill.value ∧ bill.value > ann.value
      then Ranking.CDBA
      else Ranking.CDBA) := by
  sorry

end NUMINAMATH_CALUDE_correct_ranking_l3652_365236


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3652_365206

/-- Given a right triangle with one leg of 15 inches and the angle opposite to that leg being 30°,
    the length of the hypotenuse is 30 inches. -/
theorem right_triangle_hypotenuse (a b c : ℝ) (θ : ℝ) : 
  a = 15 →  -- One leg is 15 inches
  θ = 30 * π / 180 →  -- Angle opposite to that leg is 30° (converted to radians)
  θ = Real.arcsin (a / c) →  -- Sine of the angle is opposite over hypotenuse
  a ^ 2 + b ^ 2 = c ^ 2 →  -- Pythagorean theorem
  c = 30 :=  -- Hypotenuse is 30 inches
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3652_365206


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3652_365265

/-- Given a hyperbola with equation x^2/a^2 - y^2/4 = 1 and an asymptote y = (1/2)x,
    prove that the equation of the hyperbola is x^2/16 - y^2/4 = 1 -/
theorem hyperbola_equation (a : ℝ) (h : a ≠ 0) :
  (∀ x y : ℝ, x^2/a^2 - y^2/4 = 1 → ∃ t : ℝ, y = (1/2) * x * t) →
  (∀ x y : ℝ, x^2/16 - y^2/4 = 1 ↔ x^2/a^2 - y^2/4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3652_365265


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3652_365229

theorem contrapositive_equivalence (p q : Prop) :
  (p → ¬q) ↔ (q → ¬p) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3652_365229


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l3652_365268

theorem smallest_next_divisor_after_221 (n : ℕ) (h1 : 1000 ≤ n ∧ n ≤ 9999) 
  (h2 : Even n) (h3 : 221 ∣ n) : 
  ∃ (d : ℕ), d ∣ n ∧ d > 221 ∧ d ≥ 238 ∧ ∀ (x : ℕ), x ∣ n → x > 221 → x ≥ d :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l3652_365268


namespace NUMINAMATH_CALUDE_hat_number_sum_l3652_365232

/-- Represents a four-digit perfect square number with tens digit 0 and non-zero units digit -/
structure FourDigitPerfectSquare where
  value : Nat
  is_four_digit : value ≥ 1000 ∧ value < 10000
  is_perfect_square : ∃ n, value = n * n
  tens_digit_zero : (value / 10) % 10 = 0
  units_digit_nonzero : value % 10 ≠ 0

/-- The set of all valid FourDigitPerfectSquare numbers -/
def ValidNumbers : Finset FourDigitPerfectSquare := sorry

/-- Predicate to check if two FourDigitPerfectSquare numbers have the same units digit -/
def SameUnitsDigit (a b : FourDigitPerfectSquare) : Prop :=
  a.value % 10 = b.value % 10

/-- Predicate to check if a FourDigitPerfectSquare number has an even units digit -/
def EvenUnitsDigit (a : FourDigitPerfectSquare) : Prop :=
  a.value % 2 = 0

theorem hat_number_sum :
  ∃ (a b c : FourDigitPerfectSquare),
    a ∈ ValidNumbers ∧
    b ∈ ValidNumbers ∧
    c ∈ ValidNumbers ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    SameUnitsDigit b c ∧
    EvenUnitsDigit a ∧
    (∀ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z → SameUnitsDigit y z → x = a ∧ y = b ∧ z = c) ∧
    a.value + b.value + c.value = 14612 :=
  sorry

end NUMINAMATH_CALUDE_hat_number_sum_l3652_365232


namespace NUMINAMATH_CALUDE_unknown_number_proof_l3652_365247

theorem unknown_number_proof (y : ℝ) : (12^2 : ℝ) * y^3 / 432 = 72 → y = 6 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l3652_365247


namespace NUMINAMATH_CALUDE_grid_sum_example_unique_transformed_grid_sum_constant_grid_sum_difference_count_grids_with_sum_104_l3652_365298

/-- Definition of a 2x2 grid of positive digits -/
structure Grid :=
  (a b c d : ℕ)
  (ha : 0 < a ∧ a < 10)
  (hb : 0 < b ∧ b < 10)
  (hc : 0 < c ∧ c < 10)
  (hd : 0 < d ∧ d < 10)

/-- The grid sum operation -/
def gridSum (g : Grid) : ℕ := 10*g.a + g.b + 10*g.c + g.d + 10*g.a + g.c + 10*g.b + g.d

/-- Theorem for part (a) -/
theorem grid_sum_example : 
  ∃ g : Grid, g.a = 7 ∧ g.b = 3 ∧ g.c = 2 ∧ g.d = 7 ∧ gridSum g = 209 := sorry

/-- Theorem for part (b) -/
theorem unique_transformed_grid_sum :
  ∃! x y : ℕ, ∀ b c : ℕ, 0 < b ∧ b < 9 ∧ 0 < c ∧ c < 10 →
    ∃ g1 g2 : Grid,
      g1.a = 5 ∧ g1.b = b ∧ g1.c = c ∧ g1.d = 7 ∧
      g2.a = x ∧ g2.b = b+1 ∧ g2.c = c-3 ∧ g2.d = y ∧
      gridSum g1 = gridSum g2 := sorry

/-- Theorem for part (c) -/
theorem constant_grid_sum_difference :
  ∃ k : ℤ, ∀ g : Grid,
    gridSum g - gridSum ⟨g.a+1, g.b-2, g.c-1, g.d+1, sorry, sorry, sorry, sorry⟩ = k := sorry

/-- Theorem for part (d) -/
theorem count_grids_with_sum_104 :
  ∃! ls : List Grid, (∀ g ∈ ls, gridSum g = 104) ∧ ls.length = 5 := sorry

end NUMINAMATH_CALUDE_grid_sum_example_unique_transformed_grid_sum_constant_grid_sum_difference_count_grids_with_sum_104_l3652_365298


namespace NUMINAMATH_CALUDE_total_snacks_weight_l3652_365262

theorem total_snacks_weight (peanuts_weight raisins_weight : ℝ) 
  (h1 : peanuts_weight = 0.1)
  (h2 : raisins_weight = 0.4) :
  peanuts_weight + raisins_weight = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_total_snacks_weight_l3652_365262


namespace NUMINAMATH_CALUDE_range_of_a_l3652_365285

theorem range_of_a (p q : Prop) (h_p : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x)
  (h_q : ∃ x : ℝ, x^2 - 4*x + a ≤ 0) (h_pq : p ∧ q) :
  a ∈ Set.Icc (Real.exp 1) 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3652_365285


namespace NUMINAMATH_CALUDE_systematic_sampling_smallest_number_l3652_365237

theorem systematic_sampling_smallest_number 
  (total_classes : ℕ) 
  (selected_classes : ℕ) 
  (sum_of_selected : ℕ) 
  (h1 : total_classes = 24)
  (h2 : selected_classes = 4)
  (h3 : sum_of_selected = 48) :
  let interval := total_classes / selected_classes
  let smallest := (sum_of_selected - (selected_classes - 1) * selected_classes * interval / 2) / selected_classes
  smallest = 3 := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_smallest_number_l3652_365237


namespace NUMINAMATH_CALUDE_distance_negative_five_to_negative_fourteen_l3652_365272

/-- The distance between two points on a number line -/
def numberLineDistance (a b : ℝ) : ℝ := |a - b|

/-- Theorem: The distance between -5 and -14 on a number line is 9 -/
theorem distance_negative_five_to_negative_fourteen :
  numberLineDistance (-5) (-14) = 9 := by
  sorry

end NUMINAMATH_CALUDE_distance_negative_five_to_negative_fourteen_l3652_365272


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3652_365299

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the 5th through 7th terms of an arithmetic sequence. -/
def SumFifthToSeventh (a : ℕ → ℤ) : ℤ := a 5 + a 6 + a 7

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  ArithmeticSequence a →
  a 8 = 16 →
  a 9 = 22 →
  a 10 = 28 →
  SumFifthToSeventh a = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3652_365299


namespace NUMINAMATH_CALUDE_f_is_odd_l3652_365200

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom add_property : ∀ x y : ℝ, f (x + y) = f x + f y
axiom not_identically_zero : ∃ x : ℝ, f x ≠ 0

-- Define what it means for f to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem f_is_odd : is_odd f := by sorry

end NUMINAMATH_CALUDE_f_is_odd_l3652_365200


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3652_365255

theorem equilateral_triangle_perimeter (area : ℝ) (p : ℝ) : 
  area = 50 * Real.sqrt 12 → p = 60 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3652_365255


namespace NUMINAMATH_CALUDE_base8_175_equals_base10_125_l3652_365277

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (n : ℕ) : ℕ :=
  let d₂ := n / 100
  let d₁ := (n / 10) % 10
  let d₀ := n % 10
  d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

theorem base8_175_equals_base10_125 : base8ToBase10 175 = 125 := by
  sorry

end NUMINAMATH_CALUDE_base8_175_equals_base10_125_l3652_365277


namespace NUMINAMATH_CALUDE_sqrt_product_l3652_365296

theorem sqrt_product (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) : 
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_l3652_365296


namespace NUMINAMATH_CALUDE_correct_factorization_l3652_365212

theorem correct_factorization (a : ℝ) : -1 + 4 * a^2 = (2*a + 1) * (2*a - 1) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l3652_365212


namespace NUMINAMATH_CALUDE_y_value_l3652_365235

theorem y_value : ∃ y : ℝ, (3 * y) / 7 = 15 ∧ y = 35 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l3652_365235


namespace NUMINAMATH_CALUDE_two_intersection_points_l3652_365291

-- Define the three lines
def line1 (x y : ℝ) : Prop := 2*y - 3*x = 3
def line2 (x y : ℝ) : Prop := x + 3*y = 3
def line3 (x y : ℝ) : Prop := 5*x - 3*y = 6

-- Define a function to check if a point lies on at least two lines
def onAtLeastTwoLines (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line2 x y ∧ line3 x y)

-- Theorem statement
theorem two_intersection_points :
  ∃ (p1 p2 : ℝ × ℝ), 
    p1 ≠ p2 ∧ 
    onAtLeastTwoLines p1.1 p1.2 ∧ 
    onAtLeastTwoLines p2.1 p2.2 ∧
    ∀ (p : ℝ × ℝ), onAtLeastTwoLines p.1 p.2 → p = p1 ∨ p = p2 :=
sorry

end NUMINAMATH_CALUDE_two_intersection_points_l3652_365291


namespace NUMINAMATH_CALUDE_discount_calculation_l3652_365292

theorem discount_calculation (marked_price : ℝ) (discount_rate : ℝ) (num_articles : ℕ) 
  (h1 : marked_price = 15)
  (h2 : discount_rate = 0.4)
  (h3 : num_articles = 2) :
  marked_price * num_articles * (1 - discount_rate) = 18 :=
by sorry

end NUMINAMATH_CALUDE_discount_calculation_l3652_365292


namespace NUMINAMATH_CALUDE_percent_profit_calculation_l3652_365205

theorem percent_profit_calculation (C S : ℝ) (h : 60 * C = 50 * S) :
  (S - C) / C * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percent_profit_calculation_l3652_365205


namespace NUMINAMATH_CALUDE_bake_sale_group_composition_l3652_365278

theorem bake_sale_group_composition (p : ℕ) : 
  (p : ℚ) > 0 →
  (p / 2 : ℚ) / p = 1 / 2 →
  ((p / 2 - 3) : ℚ) / p = 2 / 5 →
  p / 2 = 15 := by
sorry

end NUMINAMATH_CALUDE_bake_sale_group_composition_l3652_365278


namespace NUMINAMATH_CALUDE_pq_relation_l3652_365261

theorem pq_relation (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 8) : 
  q = 4 + 2 * Real.sqrt 2 ∨ q = 4 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_pq_relation_l3652_365261


namespace NUMINAMATH_CALUDE_brothers_puzzle_l3652_365203

-- Define the possible identities
inductive Identity : Type
| Tweedledee : Identity
| Tweedledum : Identity

-- Define the days of the week
inductive DayOfWeek : Type
| Sunday : DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek

-- Define the brothers
structure Brother :=
(identity : Identity)

-- Define the scenario
structure Scenario :=
(brother1 : Brother)
(brother2 : Brother)
(day : DayOfWeek)

-- Define the statements of the brothers
def statement1 (s : Scenario) : Prop :=
  s.brother1.identity = Identity.Tweedledee → s.brother2.identity = Identity.Tweedledum

def statement2 (s : Scenario) : Prop :=
  s.brother2.identity = Identity.Tweedledum → s.brother1.identity = Identity.Tweedledee

-- Theorem: The scenario must be on Sunday and identities cannot be determined
theorem brothers_puzzle (s : Scenario) :
  (statement1 s ∧ statement2 s) →
  (s.day = DayOfWeek.Sunday ∧
   ¬(s.brother1.identity ≠ s.brother2.identity)) :=
by sorry

end NUMINAMATH_CALUDE_brothers_puzzle_l3652_365203


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l3652_365284

theorem expression_equals_negative_one :
  |(-Real.sqrt 2)| + (2016 + Real.pi)^(0 : ℝ) + (-1/2)⁻¹ - 2 * Real.sin (45 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l3652_365284


namespace NUMINAMATH_CALUDE_frank_candy_count_l3652_365242

/-- Given a number of bags and pieces per bag, calculates the total number of pieces -/
def totalPieces (n m : ℕ) : ℕ := n * m

/-- Theorem: For 2 bags with 21 pieces each, the total number of pieces is 42 -/
theorem frank_candy_count : totalPieces 2 21 = 42 := by
  sorry

end NUMINAMATH_CALUDE_frank_candy_count_l3652_365242


namespace NUMINAMATH_CALUDE_number_division_addition_l3652_365220

theorem number_division_addition : ∃ x : ℝ, 7500 + x / 50 = 7525 := by
  sorry

end NUMINAMATH_CALUDE_number_division_addition_l3652_365220


namespace NUMINAMATH_CALUDE_power_sum_div_diff_equals_17_15_l3652_365245

theorem power_sum_div_diff_equals_17_15 :
  (2^2020 + 2^2016) / (2^2020 - 2^2016) = 17/15 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_div_diff_equals_17_15_l3652_365245


namespace NUMINAMATH_CALUDE_kalebs_savings_l3652_365257

/-- The amount of money Kaleb needs to buy the toys -/
def total_cost (num_toys : ℕ) (price_per_toy : ℕ) : ℕ := num_toys * price_per_toy

/-- The amount of money Kaleb has saved initially -/
def initial_savings (total_cost additional_money : ℕ) : ℕ := total_cost - additional_money

/-- Theorem stating Kaleb's initial savings -/
theorem kalebs_savings (num_toys price_per_toy additional_money : ℕ) 
  (h1 : num_toys = 6)
  (h2 : price_per_toy = 6)
  (h3 : additional_money = 15) :
  initial_savings (total_cost num_toys price_per_toy) additional_money = 21 := by
  sorry

#check kalebs_savings

end NUMINAMATH_CALUDE_kalebs_savings_l3652_365257


namespace NUMINAMATH_CALUDE_combined_solution_x_percentage_l3652_365273

/-- Represents a solution composed of liquid X and water -/
structure Solution where
  total_mass : ℝ
  x_percentage : ℝ

/-- The initial solution Y1 -/
def Y1 : Solution :=
  { total_mass := 12
  , x_percentage := 0.3 }

/-- The mass of water that evaporates -/
def evaporated_water : ℝ := 3

/-- The solution Y2 after evaporation -/
def Y2 : Solution :=
  { total_mass := Y1.total_mass - evaporated_water
  , x_percentage := 0.4 }

/-- The mass of Y2 added to the remaining solution -/
def added_Y2_mass : ℝ := 4

/-- Calculates the mass of liquid X in a given solution -/
def liquid_x_mass (s : Solution) : ℝ :=
  s.total_mass * s.x_percentage

/-- Calculates the mass of water in a given solution -/
def water_mass (s : Solution) : ℝ :=
  s.total_mass * (1 - s.x_percentage)

/-- The combined solution after adding Y2 -/
def combined_solution : Solution :=
  { total_mass := Y2.total_mass + added_Y2_mass
  , x_percentage := 0 }  -- Placeholder value, to be proved

theorem combined_solution_x_percentage :
  combined_solution.x_percentage = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_combined_solution_x_percentage_l3652_365273


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3652_365256

/-- Given vectors a and b in ℝ², prove that they are perpendicular if and only if x = -3 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) : 
  a = (-1, 3) → b = (-3, x) → (a.1 * b.1 + a.2 * b.2 = 0 ↔ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3652_365256


namespace NUMINAMATH_CALUDE_equidistant_function_property_l3652_365286

/-- A function that scales a complex number by a complex factor -/
def g (c d : ℝ) (z : ℂ) : ℂ := (c + d * Complex.I) * z

/-- The theorem stating the properties and result of the problem -/
theorem equidistant_function_property (c d : ℝ) :
  (c > 0) →
  (d > 0) →
  (∀ z : ℂ, Complex.abs (g c d z - z) = Complex.abs (g c d z)) →
  Complex.abs (c + d * Complex.I) = 10 →
  d^2 = 99.75 := by sorry

end NUMINAMATH_CALUDE_equidistant_function_property_l3652_365286


namespace NUMINAMATH_CALUDE_area_of_five_presentable_set_l3652_365267

/-- A complex number is five-presentable if it can be represented as w - 1/w for some complex number w with |w| = 5 -/
def FivePresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 5 ∧ z = w - 1 / w

/-- The set of all five-presentable complex numbers -/
def S : Set ℂ :=
  {z : ℂ | FivePresentable z}

/-- The area of a set in the complex plane -/
noncomputable def area (A : Set ℂ) : ℝ := sorry

theorem area_of_five_presentable_set :
  area S = 624 * Real.pi / 25 := by sorry

end NUMINAMATH_CALUDE_area_of_five_presentable_set_l3652_365267


namespace NUMINAMATH_CALUDE_miles_guitars_l3652_365219

/-- Represents the number of musical instruments Miles owns. -/
structure MilesInstruments where
  guitars : ℕ
  trumpets : ℕ
  trombones : ℕ
  frenchHorns : ℕ

/-- The total number of Miles' fingers. -/
def numFingers : ℕ := 10

/-- The number of Miles' hands. -/
def numHands : ℕ := 2

/-- The number of Miles' heads. -/
def numHeads : ℕ := 1

/-- The total number of musical instruments Miles owns. -/
def totalInstruments : ℕ := 17

/-- Theorem stating the number of guitars Miles owns. -/
theorem miles_guitars :
  ∃ (m : MilesInstruments),
    m.trumpets = numFingers - 3
    ∧ m.trombones = numHeads + 2
    ∧ m.frenchHorns = m.guitars - 1
    ∧ m.guitars = numHands + 2
    ∧ m.trumpets + m.trombones + m.guitars + m.frenchHorns = totalInstruments
    ∧ m.guitars = 4 := by
  sorry

end NUMINAMATH_CALUDE_miles_guitars_l3652_365219


namespace NUMINAMATH_CALUDE_friendship_theorem_l3652_365249

/-- A graph representing friendships in a city --/
structure FriendshipGraph where
  vertices : Finset ℕ
  edges : Finset (Finset ℕ)
  edge_size : ∀ e ∈ edges, Finset.card e = 2
  vertex_bound : Finset.card vertices = 2000000

/-- Property that every subgraph of 2000 vertices contains a triangle --/
def has_triangle_in_subgraphs (G : FriendshipGraph) : Prop :=
  ∀ S : Finset ℕ, S ⊆ G.vertices → Finset.card S = 2000 →
    ∃ T : Finset ℕ, T ⊆ S ∧ Finset.card T = 3 ∧ T ∈ G.edges

/-- Theorem stating the existence of K₄ in the graph --/
theorem friendship_theorem (G : FriendshipGraph) 
  (h : has_triangle_in_subgraphs G) : 
  ∃ K : Finset ℕ, K ⊆ G.vertices ∧ Finset.card K = 4 ∧ 
    ∀ e : Finset ℕ, e ⊆ K → Finset.card e = 2 → e ∈ G.edges :=
sorry

end NUMINAMATH_CALUDE_friendship_theorem_l3652_365249


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3652_365280

theorem trigonometric_identity (x : ℝ) : 
  (4 * Real.sin x ^ 3 * Real.cos (3 * x) + 4 * Real.cos x ^ 3 * Real.sin (3 * x) = 3 * Real.sin (2 * x)) ↔ 
  (∃ n : ℤ, x = π / 6 * (2 * ↑n + 1)) ∨ (∃ k : ℤ, x = π * ↑k) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3652_365280


namespace NUMINAMATH_CALUDE_arcade_spending_fraction_l3652_365270

theorem arcade_spending_fraction (allowance : ℚ) (arcade_fraction : ℚ) : 
  allowance = 3/2 →
  (2/3 * (1 - arcade_fraction) * allowance = 2/5) →
  arcade_fraction = 3/5 := by
sorry

end NUMINAMATH_CALUDE_arcade_spending_fraction_l3652_365270


namespace NUMINAMATH_CALUDE_cosine_inequality_l3652_365210

theorem cosine_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : 0 < x^2 + y^2 ∧ x^2 + y^2 ≤ π) :
  1 + Real.cos (x * y) ≥ Real.cos x + Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_cosine_inequality_l3652_365210


namespace NUMINAMATH_CALUDE_second_project_grade_l3652_365293

/-- Represents the grading system for a computer programming course project. -/
structure ProjectGrade where
  /-- The proportion of influence from time spent on the project. -/
  timeProportion : ℝ
  /-- The proportion of influence from effort spent on the project. -/
  effortProportion : ℝ
  /-- Calculates the influence score based on time and effort. -/
  influenceScore : ℝ → ℝ → ℝ
  /-- The proportionality constant between influence score and grade. -/
  gradeProportionality : ℝ

/-- Theorem stating the grade for the second project given the conditions. -/
theorem second_project_grade (pg : ProjectGrade)
  (h1 : pg.timeProportion = 0.70)
  (h2 : pg.effortProportion = 0.30)
  (h3 : pg.influenceScore t e = pg.timeProportion * t + pg.effortProportion * e)
  (h4 : pg.gradeProportionality = 84 / (pg.influenceScore 5 70))
  : pg.gradeProportionality * (pg.influenceScore 6 80) = 96.49 := by
  sorry

#check second_project_grade

end NUMINAMATH_CALUDE_second_project_grade_l3652_365293


namespace NUMINAMATH_CALUDE_train_bridge_time_l3652_365246

/-- Given a train of length 18 meters that passes a pole in 9 seconds,
    prove that it takes 27 seconds to pass a bridge of length 36 meters. -/
theorem train_bridge_time (train_length : ℝ) (pole_pass_time : ℝ) (bridge_length : ℝ) :
  train_length = 18 →
  pole_pass_time = 9 →
  bridge_length = 36 →
  (train_length + bridge_length) / (train_length / pole_pass_time) = 27 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_time_l3652_365246


namespace NUMINAMATH_CALUDE_chip_notes_theorem_l3652_365223

/-- Represents the number of pages Chip takes for each class every day -/
def pages_per_class : ℕ :=
  let days_per_week : ℕ := 5
  let classes_per_day : ℕ := 5
  let weeks : ℕ := 6
  let packs_used : ℕ := 3
  let sheets_per_pack : ℕ := 100
  let total_sheets : ℕ := packs_used * sheets_per_pack
  let total_days : ℕ := weeks * days_per_week
  let total_classes : ℕ := total_days * classes_per_day
  total_sheets / total_classes

theorem chip_notes_theorem : pages_per_class = 2 := by
  sorry

end NUMINAMATH_CALUDE_chip_notes_theorem_l3652_365223


namespace NUMINAMATH_CALUDE_exists_cut_sequence_for_1003_l3652_365243

/-- Represents the number of pieces selected for cutting at each step -/
def CutSequence := List Nat

/-- Calculates the number of pieces after a sequence of cuts -/
def numPieces (cuts : CutSequence) : Nat :=
  3 * (cuts.sum + 1) + 1

/-- Theorem: It's possible to obtain 1003 pieces through the cutting process -/
theorem exists_cut_sequence_for_1003 : ∃ (cuts : CutSequence), numPieces cuts = 1003 := by
  sorry

end NUMINAMATH_CALUDE_exists_cut_sequence_for_1003_l3652_365243


namespace NUMINAMATH_CALUDE_quadrilateral_property_l3652_365222

theorem quadrilateral_property (α β γ δ : Real) 
  (convex : α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0)
  (sum_angles : α + β + γ + δ = 2 * π)
  (sum_cosines : Real.cos α + Real.cos β + Real.cos γ + Real.cos δ = 0) :
  (α + β = π ∨ γ + δ = π) ∨ (α + γ = β + δ) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_property_l3652_365222


namespace NUMINAMATH_CALUDE_percentage_calculation_l3652_365234

theorem percentage_calculation (A B : ℝ) (x : ℝ) 
  (h1 : A - B = 1670)
  (h2 : A = 2505)
  (h3 : (7.5 / 100) * A = (x / 100) * B) :
  x = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3652_365234


namespace NUMINAMATH_CALUDE_cloth_gain_proof_l3652_365289

/-- 
Given:
- A shop owner sells 40 meters of cloth
- The gain percentage is 33.33333333333333%

Prove that the gain is equivalent to the selling price of 10 meters of cloth
-/
theorem cloth_gain_proof (total_meters : ℝ) (gain_percentage : ℝ) 
  (h1 : total_meters = 40)
  (h2 : gain_percentage = 33.33333333333333) :
  (gain_percentage / 100 * total_meters) / (1 + gain_percentage / 100) = 10 := by
  sorry

end NUMINAMATH_CALUDE_cloth_gain_proof_l3652_365289


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3652_365201

theorem inequality_solution_set (a : ℝ) (h : a < 2) :
  {x : ℝ | a * x > 2 * x + a - 2} = {x : ℝ | x < 1} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3652_365201


namespace NUMINAMATH_CALUDE_scientific_notation_of_1300000_l3652_365290

/-- Express 1,300,000 in scientific notation -/
theorem scientific_notation_of_1300000 :
  (1300000 : ℝ) = 1.3 * (10 : ℝ) ^ 6 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1300000_l3652_365290


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l3652_365215

/-- Represents the number of handshakes in a gymnastics championship. -/
def total_handshakes : ℕ := 456

/-- Calculates the number of handshakes between gymnasts given the total number of gymnasts. -/
def gymnast_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents the total number of gymnasts. -/
def total_gymnasts : ℕ := 30

/-- Represents the number of handshakes involving coaches. -/
def coach_handshakes : ℕ := total_handshakes - gymnast_handshakes total_gymnasts

/-- Theorem stating the minimum number of handshakes involving at least one coach. -/
theorem min_coach_handshakes : ∃ (k₁ k₂ : ℕ), k₁ + k₂ = coach_handshakes ∧ min k₁ k₂ = 1 :=
sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l3652_365215


namespace NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l3652_365202

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

theorem cryptarithmetic_puzzle :
  ∀ (I X E L V : ℕ),
    (I < 10) →
    (X < 10) →
    (E < 10) →
    (L < 10) →
    (V < 10) →
    (is_odd X) →
    (I ≠ 7) →
    (X ≠ 7) →
    (E ≠ 7) →
    (L ≠ 7) →
    (V ≠ 7) →
    (I ≠ X) →
    (I ≠ E) →
    (I ≠ L) →
    (I ≠ V) →
    (X ≠ E) →
    (X ≠ L) →
    (X ≠ V) →
    (E ≠ L) →
    (E ≠ V) →
    (L ≠ V) →
    (700 + 10*I + X + 700 + 10*I + X = 1000*E + 100*L + 10*E + V) →
    (I = 2) :=
by sorry

end NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l3652_365202


namespace NUMINAMATH_CALUDE_rainfall_ratio_l3652_365250

theorem rainfall_ratio (total_rainfall : ℝ) (second_week_rainfall : ℝ) 
  (h1 : total_rainfall = 40)
  (h2 : second_week_rainfall = 24) :
  (second_week_rainfall) / (total_rainfall - second_week_rainfall) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_ratio_l3652_365250


namespace NUMINAMATH_CALUDE_max_value_of_function_l3652_365225

theorem max_value_of_function (a : ℕ+) :
  (∃ (y : ℕ+), ∀ (x : ℝ), x + Real.sqrt (13 - 2 * (a : ℝ) * x) ≤ (y : ℝ)) →
  (∃ (y_max : ℕ+), ∀ (x : ℝ), x + Real.sqrt (13 - 2 * (a : ℝ) * x) ≤ (y_max : ℝ) ∧ y_max = 7) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3652_365225


namespace NUMINAMATH_CALUDE_johns_playing_days_l3652_365217

def beats_per_minute : ℕ := 200
def hours_per_day : ℕ := 2
def total_beats : ℕ := 72000

def minutes_per_hour : ℕ := 60
def minutes_per_day : ℕ := hours_per_day * minutes_per_hour
def beats_per_day : ℕ := beats_per_minute * minutes_per_day

theorem johns_playing_days :
  total_beats / beats_per_day = 3 :=
by sorry

end NUMINAMATH_CALUDE_johns_playing_days_l3652_365217


namespace NUMINAMATH_CALUDE_q_zero_at_sqrt2_l3652_365283

-- Define the polynomial q
def q (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ : ℝ) (x y : ℝ) : ℝ :=
  b₀ + b₁*x + b₂*y + b₃*x^2 + b₄*x*y + b₅*y^2 + b₆*x^3 + b₈*x*y^2 + b₉*y^3

-- State the theorem
theorem q_zero_at_sqrt2 (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ : ℝ) :
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 0 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 1 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ (-1) 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 0 1 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 0 (-1) = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 1 1 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ (-2) 1 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 3 (-1) = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ (Real.sqrt 2) (Real.sqrt 2) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_q_zero_at_sqrt2_l3652_365283


namespace NUMINAMATH_CALUDE_isosceles_triangles_not_necessarily_congruent_l3652_365254

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  /-- Length of the equal sides -/
  side_length : ℝ
  /-- Length of the base -/
  base_length : ℝ
  /-- Radius of the inscribed circle -/
  incircle_radius : ℝ
  /-- side_length is positive -/
  side_length_pos : 0 < side_length
  /-- base_length is positive -/
  base_length_pos : 0 < base_length
  /-- incircle_radius is positive -/
  incircle_radius_pos : 0 < incircle_radius
  /-- The base cannot be longer than twice the side length -/
  base_bound : base_length ≤ 2 * side_length
  /-- Relation between side length, base length, and incircle radius -/
  geometry_constraint : incircle_radius = (base_length * Real.sqrt (side_length^2 - (base_length/2)^2)) / (2 * side_length + base_length)

/-- Two isosceles triangles with the same side length and incircle radius are not necessarily congruent -/
theorem isosceles_triangles_not_necessarily_congruent :
  ∃ (t1 t2 : IsoscelesTriangle), 
    t1.side_length = t2.side_length ∧ 
    t1.incircle_radius = t2.incircle_radius ∧ 
    t1.base_length ≠ t2.base_length :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangles_not_necessarily_congruent_l3652_365254


namespace NUMINAMATH_CALUDE_inequality_proof_l3652_365288

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_eq : a + b + c = a * b + b * c + c * a) : 
  3 + (((a^3 + 1) / 2)^(1/3) + ((b^3 + 1) / 2)^(1/3) + ((c^3 + 1) / 2)^(1/3)) ≤ 2 * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3652_365288


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3652_365264

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a.1 * k = b.1 ∧ a.2 * k = b.2

/-- Given parallel vectors (2,3) and (x,-6), x equals -4 -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (2, 3) (x, -6) → x = -4 :=
by
  sorry

#check parallel_vectors_x_value

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3652_365264


namespace NUMINAMATH_CALUDE_inequality_proof_l3652_365226

theorem inequality_proof (a b c : ℝ) 
  (h_nonneg_a : a ≥ 0) (h_nonneg_b : b ≥ 0) (h_nonneg_c : c ≥ 0)
  (h_ineq_a : a^2 ≤ b^2 + c^2)
  (h_ineq_b : b^2 ≤ c^2 + a^2)
  (h_ineq_c : c^2 ≤ a^2 + b^2) :
  (a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) ≥ 4 * (a^6 + b^6 + c^6) ∧
  ((a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) = 4 * (a^6 + b^6 + c^6) ↔ a = b ∧ b = c) :=
by sorry


end NUMINAMATH_CALUDE_inequality_proof_l3652_365226


namespace NUMINAMATH_CALUDE_chef_michel_pies_l3652_365213

/-- Represents the number of pies sold given the number of pieces and customers -/
def pies_sold (pieces : ℕ) (customers : ℕ) : ℕ :=
  (customers + pieces - 1) / pieces

/-- The total number of pies sold by Chef Michel -/
def total_pies : ℕ :=
  pies_sold 4 52 + pies_sold 8 76 + pies_sold 5 80 + pies_sold 10 130

/-- Theorem stating that Chef Michel sold 52 pies in total -/
theorem chef_michel_pies :
  total_pies = 52 := by
  sorry

#eval total_pies

end NUMINAMATH_CALUDE_chef_michel_pies_l3652_365213


namespace NUMINAMATH_CALUDE_olga_sons_daughters_l3652_365204

/-- Represents the family structure of Grandma Olga -/
structure OlgaFamily where
  daughters : Nat
  sons : Nat
  grandchildren : Nat
  daughters_sons : Nat
  sons_daughters : Nat

/-- The theorem stating the number of daughters each of Grandma Olga's sons has -/
theorem olga_sons_daughters (family : OlgaFamily) :
  family.daughters = 3 →
  family.sons = 3 →
  family.daughters_sons = 6 →
  family.grandchildren = 33 →
  family.sons_daughters = 5 := by
  sorry

end NUMINAMATH_CALUDE_olga_sons_daughters_l3652_365204


namespace NUMINAMATH_CALUDE_solution_set_f_plus_x_positive_range_of_a_for_full_solution_set_l3652_365275

def f (x : ℝ) := |x - 2| - |x + 1|

theorem solution_set_f_plus_x_positive :
  {x : ℝ | f x + x > 0} = Set.union (Set.union (Set.Ioo (-3) (-1)) (Set.Ico (-1) 1)) (Set.Ioi 3) :=
sorry

theorem range_of_a_for_full_solution_set :
  {a : ℝ | ∀ x, f x ≤ a^2 - 2*a} = Set.union (Set.Iic (-1)) (Set.Ici 3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_plus_x_positive_range_of_a_for_full_solution_set_l3652_365275


namespace NUMINAMATH_CALUDE_largest_sum_proof_l3652_365218

theorem largest_sum_proof : 
  let sums : List ℚ := [1/4 + 1/2, 1/4 + 1/9, 1/4 + 1/3, 1/4 + 1/10, 1/4 + 1/6]
  (∀ x ∈ sums, x ≤ 1/4 + 1/2) ∧ (1/4 + 1/2 = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_proof_l3652_365218


namespace NUMINAMATH_CALUDE_tamika_driving_time_l3652_365253

-- Define the variables
def tamika_speed : ℝ := 45
def logan_speed : ℝ := 55
def logan_time : ℝ := 5
def distance_difference : ℝ := 85

-- Theorem statement
theorem tamika_driving_time :
  ∃ (h : ℝ), h * tamika_speed = logan_speed * logan_time + distance_difference ∧ h = 8 := by
  sorry

end NUMINAMATH_CALUDE_tamika_driving_time_l3652_365253


namespace NUMINAMATH_CALUDE_f_min_at_one_plus_inv_sqrt_three_l3652_365240

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + 6 * x + 1

-- State the theorem
theorem f_min_at_one_plus_inv_sqrt_three :
  ∃ (x_min : ℝ), x_min = 1 + 1 / Real.sqrt 3 ∧
  ∀ (x : ℝ), f x ≥ f x_min :=
by sorry

end NUMINAMATH_CALUDE_f_min_at_one_plus_inv_sqrt_three_l3652_365240


namespace NUMINAMATH_CALUDE_z_equation_solution_l3652_365282

theorem z_equation_solution :
  let z : ℝ := Real.sqrt ((Real.sqrt 29) / 2 + 7 / 2)
  ∃! (d e f : ℕ+),
    z^100 = 2*z^98 + 14*z^96 + 11*z^94 - z^50 + (d : ℝ)*z^46 + (e : ℝ)*z^44 + (f : ℝ)*z^40 ∧
    d + e + f = 205 := by
  sorry

end NUMINAMATH_CALUDE_z_equation_solution_l3652_365282


namespace NUMINAMATH_CALUDE_trapezoid_diagonal_length_l3652_365251

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of base AD
  ad : ℝ
  -- Length of base BC
  bc : ℝ
  -- Length of diagonal AC
  ac : ℝ
  -- Circles on AB, BC, CD as diameters intersect at a single point
  circles_intersect : Prop

/-- The theorem stating that under given conditions, diagonal BD has length 24 -/
theorem trapezoid_diagonal_length (t : Trapezoid) 
  (h1 : t.ad = 16)
  (h2 : t.bc = 10)
  (h3 : t.ac = 10)
  (h4 : t.circles_intersect) :
  ∃ (bd : ℝ), bd = 24 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_diagonal_length_l3652_365251


namespace NUMINAMATH_CALUDE_probability_one_triple_one_pair_l3652_365287

def num_dice : ℕ := 5
def faces_per_die : ℕ := 6

def favorable_outcomes : ℕ := faces_per_die * (num_dice.choose 3) * (faces_per_die - 1) * 1

def total_outcomes : ℕ := faces_per_die ^ num_dice

theorem probability_one_triple_one_pair :
  (favorable_outcomes : ℚ) / total_outcomes = 25 / 648 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_triple_one_pair_l3652_365287
