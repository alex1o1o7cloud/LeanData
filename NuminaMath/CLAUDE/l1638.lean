import Mathlib

namespace nate_search_speed_l1638_163857

/-- The number of rows in Section G of the parking lot -/
def section_g_rows : ℕ := 15

/-- The number of cars per row in Section G -/
def section_g_cars_per_row : ℕ := 10

/-- The number of rows in Section H of the parking lot -/
def section_h_rows : ℕ := 20

/-- The number of cars per row in Section H -/
def section_h_cars_per_row : ℕ := 9

/-- The time Nate spent searching in minutes -/
def search_time : ℕ := 30

/-- The number of cars Nate can walk past per minute -/
def cars_per_minute : ℕ := 11

theorem nate_search_speed :
  (section_g_rows * section_g_cars_per_row + section_h_rows * section_h_cars_per_row) / search_time = cars_per_minute := by
  sorry

end nate_search_speed_l1638_163857


namespace sum_of_pairwise_ratios_geq_three_halves_l1638_163888

theorem sum_of_pairwise_ratios_geq_three_halves 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ 3/2 := by
  sorry

end sum_of_pairwise_ratios_geq_three_halves_l1638_163888


namespace circles_common_chord_common_chord_length_l1638_163808

/-- Circle C₁ with equation x² + y² - 2x + 10y - 24 = 0 -/
def C₁ (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 10*y - 24 = 0

/-- Circle C₂ with equation x² + y² + 2x + 2y - 8 = 0 -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y - 8 = 0

/-- The line on which the common chord of C₁ and C₂ lies -/
def common_chord_line (x y : ℝ) : Prop :=
  x - 6*y + 6 = 0

theorem circles_common_chord (x y : ℝ) :
  (C₁ x y ∧ C₂ x y) → common_chord_line x y :=
sorry

theorem common_chord_length : 
  ∃ (a b : ℝ), C₁ a b ∧ C₂ a b ∧ 
  ∃ (c d : ℝ), C₁ c d ∧ C₂ c d ∧ 
  ((a - c)^2 + (b - d)^2)^(1/2 : ℝ) = 2 * 13^(1/2 : ℝ) :=
sorry

end circles_common_chord_common_chord_length_l1638_163808


namespace min_tests_correct_l1638_163884

/-- Represents the result of a test between two balls -/
inductive TestResult
| Same
| Different

/-- Represents a ball -/
structure Ball :=
  (id : Nat)
  (metal : Bool)  -- True for copper, False for zinc

/-- Represents a test between two balls -/
structure Test :=
  (ball1 : Ball)
  (ball2 : Ball)
  (result : TestResult)

/-- The minimum number of tests required to determine the material of each ball -/
def min_tests (n : Nat) (copper_count : Nat) (zinc_count : Nat) : Nat :=
  n - 1

theorem min_tests_correct (n : Nat) (copper_count : Nat) (zinc_count : Nat) 
  (h1 : n = 99)
  (h2 : copper_count = 50)
  (h3 : zinc_count = 49)
  (h4 : copper_count + zinc_count = n) :
  min_tests n copper_count zinc_count = 98 := by
  sorry

#eval min_tests 99 50 49

end min_tests_correct_l1638_163884


namespace p_costs_more_after_10_years_l1638_163823

/-- Represents the yearly price increase in paise -/
structure PriceIncrease where
  p : ℚ  -- Price increase for commodity P
  q : ℚ  -- Price increase for commodity Q

/-- Represents the initial prices in rupees -/
structure InitialPrice where
  p : ℚ  -- Initial price for commodity P
  q : ℚ  -- Initial price for commodity Q

/-- Calculates the year when commodity P costs 40 paise more than commodity Q -/
def yearWhenPCostsMoreThanQ (increase : PriceIncrease) (initial : InitialPrice) : ℕ :=
  sorry

/-- The theorem stating that P costs 40 paise more than Q after 10 years -/
theorem p_costs_more_after_10_years 
  (increase : PriceIncrease) 
  (initial : InitialPrice) 
  (h1 : increase.p = 40/100) 
  (h2 : increase.q = 15/100) 
  (h3 : initial.p = 420/100) 
  (h4 : initial.q = 630/100) : 
  yearWhenPCostsMoreThanQ increase initial = 10 := by sorry

end p_costs_more_after_10_years_l1638_163823


namespace f_values_l1638_163818

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt ((1 - Real.sin x) / (1 + Real.sin x)) - Real.sqrt ((1 + Real.sin x) / (1 - Real.sin x))) *
  (Real.sqrt ((1 - Real.cos x) / (1 + Real.cos x)) - Real.sqrt ((1 + Real.cos x) / (1 - Real.cos x)))

theorem f_values (x : ℝ) 
  (h1 : x ∈ Set.Ioo 0 (2 * Real.pi))
  (h2 : x ≠ Real.pi / 2 ∧ x ≠ Real.pi ∧ x ≠ 3 * Real.pi / 2) :
  (0 < x ∧ x < Real.pi / 2 ∨ Real.pi < x ∧ x < 3 * Real.pi / 2) → f x = 4 ∧
  (Real.pi / 2 < x ∧ x < Real.pi ∨ 3 * Real.pi / 2 < x ∧ x < 2 * Real.pi) → f x = -4 := by
  sorry

#check f_values

end f_values_l1638_163818


namespace quadratic_equation_m_value_l1638_163839

/-- The equation is quadratic if and only if the exponent of x in the first term is 2 -/
def is_quadratic (m : ℝ) : Prop := m^2 - 2 = 2

/-- The coefficient of the highest degree term should not be zero -/
def coeff_nonzero (m : ℝ) : Prop := m ≠ 2

theorem quadratic_equation_m_value :
  ∀ m : ℝ, is_quadratic m ∧ coeff_nonzero m → m = -2 :=
by sorry

end quadratic_equation_m_value_l1638_163839


namespace complement_of_25_36_l1638_163846

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  let totalMinutes := 180 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- Theorem statement
theorem complement_of_25_36 :
  complement ⟨25, 36⟩ = ⟨154, 24⟩ := by
  sorry

end complement_of_25_36_l1638_163846


namespace folded_rectangle_perimeter_l1638_163844

/-- A rectangle ABCD with a fold from A to A' on CD creating a crease EF -/
structure FoldedRectangle where
  -- Length of AE
  ae : ℝ
  -- Length of EB
  eb : ℝ
  -- Length of CF
  cf : ℝ

/-- The perimeter of the folded rectangle -/
def perimeter (r : FoldedRectangle) : ℝ :=
  2 * (r.ae + r.eb + r.cf + (r.ae + r.eb - r.cf))

/-- Theorem stating that the perimeter of the specific folded rectangle is 82 -/
theorem folded_rectangle_perimeter :
  let r : FoldedRectangle := { ae := 3, eb := 15, cf := 8 }
  perimeter r = 82 := by sorry

end folded_rectangle_perimeter_l1638_163844


namespace polar_to_cartesian_l1638_163873

theorem polar_to_cartesian (M : ℝ × ℝ) :
  M.1 = 3 ∧ M.2 = π / 6 →
  (M.1 * Real.cos M.2 = 3 * Real.sqrt 3 / 2) ∧
  (M.1 * Real.sin M.2 = 3 / 2) := by
sorry

end polar_to_cartesian_l1638_163873


namespace floor_sqrt_50_l1638_163898

theorem floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
  sorry

end floor_sqrt_50_l1638_163898


namespace bella_roses_l1638_163842

/-- The number of roses in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of roses Bella received from her parents -/
def roses_from_parents : ℕ := 2

/-- The number of Bella's dancer friends -/
def number_of_friends : ℕ := 10

/-- The number of roses each friend gave to Bella -/
def roses_per_friend : ℕ := 2

/-- The total number of roses Bella received -/
def total_roses : ℕ := roses_from_parents * dozen + number_of_friends * roses_per_friend

theorem bella_roses : total_roses = 44 := by
  sorry

end bella_roses_l1638_163842


namespace sqrt_equation_solution_l1638_163833

theorem sqrt_equation_solution :
  ∃! x : ℝ, 4 * x - 3 ≥ 0 ∧ Real.sqrt (4 * x - 3) + 16 / Real.sqrt (4 * x - 3) = 8 :=
by
  -- The unique solution is x = 19/4
  use 19/4
  sorry

end sqrt_equation_solution_l1638_163833


namespace cos_three_pi_halves_l1638_163869

theorem cos_three_pi_halves : Real.cos (3 * π / 2) = 0 := by
  sorry

end cos_three_pi_halves_l1638_163869


namespace polynomial_division_l1638_163867

theorem polynomial_division (x : ℝ) (h : x ≠ 0) :
  (x^4 - 3*x^2) / x^2 = x^2 - 3 := by
  sorry

end polynomial_division_l1638_163867


namespace distance_between_points_l1638_163800

def point1 : ℝ × ℝ := (3, -5)
def point2 : ℝ × ℝ := (-4, 4)

theorem distance_between_points :
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 130 := by
  sorry

end distance_between_points_l1638_163800


namespace number_of_factors_of_power_l1638_163849

theorem number_of_factors_of_power (b n : ℕ+) (hb : b = 8) (hn : n = 15) :
  (Finset.range ((n * (Nat.factorization b).sum (fun _ e => e)) + 1)).card = 46 := by
  sorry

end number_of_factors_of_power_l1638_163849


namespace min_value_problem_l1638_163855

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 2) : 
  1/x + 1/(3*y) ≥ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    Real.log 2 * x₀ + Real.log 8 * y₀ = Real.log 2 ∧ 1/x₀ + 1/(3*y₀) = 4 :=
by sorry

end min_value_problem_l1638_163855


namespace product_cube_l1638_163859

theorem product_cube (a b c : ℕ+) (h : a * b * c = 180) : (a * b) ^ 3 = 216 := by
  sorry

end product_cube_l1638_163859


namespace smallest_three_digit_product_l1638_163892

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem smallest_three_digit_product :
  ∀ n x y : ℕ,
    n = x * y * (10 * x + y) →
    100 ≤ n →
    n < 1000 →
    is_prime x →
    is_prime y →
    is_prime (10 * x + y) →
    x < 10 →
    y < 10 →
    x % 2 = 0 →
    y % 2 = 1 →
    x ≠ y →
    x ≠ 10 * x + y →
    y ≠ 10 * x + y →
    n ≥ 138 :=
by sorry

end smallest_three_digit_product_l1638_163892


namespace min_value_of_t_l1638_163812

theorem min_value_of_t (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 1 → 1 / x + 1 / y ≥ 3 + 2 * Real.sqrt 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 2 * y = 1 ∧ 1 / x + 1 / y = 3 + 2 * Real.sqrt 2) :=
by sorry

end min_value_of_t_l1638_163812


namespace count_integers_correct_l1638_163854

/-- Count of three-digit positive integers starting with 2 and greater than 217 -/
def count_integers : ℕ := 82

/-- The smallest three-digit integer starting with 2 and greater than 217 -/
def min_integer : ℕ := 218

/-- The largest three-digit integer starting with 2 -/
def max_integer : ℕ := 299

/-- Theorem stating that the count of integers is correct -/
theorem count_integers_correct :
  count_integers = max_integer - min_integer + 1 :=
by sorry

end count_integers_correct_l1638_163854


namespace prob_green_is_0_15_l1638_163862

/-- The probability of selecting a green jelly bean from a jar -/
def prob_green (prob_red prob_orange prob_blue prob_yellow : ℝ) : ℝ :=
  1 - (prob_red + prob_orange + prob_blue + prob_yellow)

/-- Theorem: The probability of selecting a green jelly bean is 0.15 -/
theorem prob_green_is_0_15 :
  prob_green 0.15 0.35 0.2 0.15 = 0.15 := by
  sorry

end prob_green_is_0_15_l1638_163862


namespace tan_alpha_value_l1638_163821

theorem tan_alpha_value (α : Real) : 
  Real.tan (π / 4 + α) = 1 / 2 → Real.tan α = -1 / 3 := by
  sorry

end tan_alpha_value_l1638_163821


namespace percentage_of_girls_in_class_l1638_163889

theorem percentage_of_girls_in_class (girls boys : ℕ) (h1 : girls = 10) (h2 : boys = 15) :
  (girls : ℚ) / ((girls : ℚ) + (boys : ℚ)) * 100 = 40 := by
  sorry

end percentage_of_girls_in_class_l1638_163889


namespace rational_sum_and_power_integers_l1638_163886

theorem rational_sum_and_power_integers (n : ℕ) : 
  (Odd n) ↔ 
  (∃ (a b : ℚ), 
    0 < a ∧ 0 < b ∧ 
    ¬(∃ (i : ℤ), a = i) ∧ ¬(∃ (j : ℤ), b = j) ∧
    (∃ (k : ℤ), (a + b : ℚ) = k) ∧ 
    (∃ (l : ℤ), (a^n + b^n : ℚ) = l)) :=
sorry

end rational_sum_and_power_integers_l1638_163886


namespace reaction_masses_l1638_163843

-- Define the molar masses
def molar_mass_HCl : ℝ := 36.46
def molar_mass_AgNO3 : ℝ := 169.87
def molar_mass_AgCl : ℝ := 143.32

-- Define the number of moles of AgNO3
def moles_AgNO3 : ℝ := 3

-- Define the reaction stoichiometry
def stoichiometry : ℝ := 1

-- Theorem statement
theorem reaction_masses :
  let mass_HCl := moles_AgNO3 * molar_mass_HCl * stoichiometry
  let mass_AgNO3 := moles_AgNO3 * molar_mass_AgNO3
  let mass_AgCl := moles_AgNO3 * molar_mass_AgCl * stoichiometry
  (mass_HCl = 109.38) ∧ (mass_AgNO3 = 509.61) ∧ (mass_AgCl = 429.96) := by
  sorry

end reaction_masses_l1638_163843


namespace m_plus_n_equals_one_l1638_163831

theorem m_plus_n_equals_one (m n : ℤ) (h : |m - 2| + (n + 1)^2 = 0) : m + n = 1 := by
  sorry

end m_plus_n_equals_one_l1638_163831


namespace john_remaining_money_l1638_163871

/-- Converts a base 8 number to base 10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- The amount John has saved in base 8 --/
def john_savings : ℕ := 5555

/-- The cost of the round-trip airline ticket in base 10 --/
def ticket_cost : ℕ := 1200

/-- The amount John will have left after buying the ticket --/
def remaining_money : ℕ := base8_to_base10 john_savings - ticket_cost

theorem john_remaining_money :
  remaining_money = 1725 := by sorry

end john_remaining_money_l1638_163871


namespace ellipse_parabola_triangle_area_l1638_163885

/-- Definition of the ellipse C₁ -/
def ellipse (x y : ℝ) : Prop := x^2 / 12 + y^2 / 4 = 1

/-- Definition of the parabola C₂ -/
def parabola (x y : ℝ) : Prop := x^2 = 8 * y

/-- The focus F of the parabola, which is also the vertex of the ellipse -/
def F : ℝ × ℝ := (0, 2)

/-- Definition of a point being on the ellipse -/
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

/-- Definition of two vectors being orthogonal -/
def orthogonal (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

/-- Definition of a line being tangent to the parabola -/
def tangent_to_parabola (P Q : ℝ × ℝ) : Prop :=
  ∃ k m : ℝ, (∀ x y : ℝ, y = k * x + m → (x^2 = 8 * y ↔ x = P.1 ∧ y = P.2))

theorem ellipse_parabola_triangle_area :
  ∀ P Q : ℝ × ℝ,
  on_ellipse P → on_ellipse Q →
  orthogonal (P.1 - F.1, P.2 - F.2) (Q.1 - F.1, Q.2 - F.2) →
  tangent_to_parabola P Q →
  P ≠ F → Q ≠ F → P ≠ Q →
  abs ((P.1 - F.1) * (Q.2 - F.2) - (P.2 - F.2) * (Q.1 - F.1)) / 2 = 18 * Real.sqrt 3 / 5 :=
sorry

end ellipse_parabola_triangle_area_l1638_163885


namespace fencing_required_l1638_163827

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) (fencing : ℝ) : 
  area = 650 ∧ uncovered_side = 20 → fencing = 85 := by
  sorry

end fencing_required_l1638_163827


namespace largest_square_area_l1638_163881

theorem largest_square_area (X Y Z : ℝ) (h_right_angle : X^2 + Y^2 = Z^2)
  (h_equal_sides : X = Y) (h_sum_areas : X^2 + Y^2 + Z^2 + (2*Y)^2 = 650) :
  Z^2 = 650/3 := by
sorry

end largest_square_area_l1638_163881


namespace larger_number_proof_l1638_163887

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1375) (h2 : L = 6 * S + 15) : L = 1647 := by
  sorry

end larger_number_proof_l1638_163887


namespace perimeter_difference_l1638_163852

/-- Represents a rectangle with length and height -/
structure Rectangle where
  length : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.height)

/-- Theorem stating the difference in perimeters of two rectangles -/
theorem perimeter_difference (inner outer : Rectangle) 
  (h1 : outer.length = 7)
  (h2 : outer.height = 5) :
  perimeter outer - perimeter inner = 24 :=
by sorry

end perimeter_difference_l1638_163852


namespace number_of_divisors_of_fermat_like_expression_l1638_163879

theorem number_of_divisors_of_fermat_like_expression : 
  ∃ (S : Finset Nat), 
    (∀ n ∈ S, n > 1 ∧ ∀ a : ℤ, (n : ℤ) ∣ (a^25 - a)) ∧ 
    (∀ n : Nat, n > 1 → (∀ a : ℤ, (n : ℤ) ∣ (a^25 - a)) → n ∈ S) ∧
    Finset.card S = 31 :=
sorry

end number_of_divisors_of_fermat_like_expression_l1638_163879


namespace percentage_increase_l1638_163825

/-- Given two positive real numbers a and b with a ratio of 4:5, 
    and x and m derived from a and b respectively, 
    prove that the percentage increase from a to x is 25% --/
theorem percentage_increase (a b x m : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  a / b = 4 / 5 →
  ∃ p, x = a * (1 + p / 100) →
  m = b * 0.6 →
  m / x = 0.6 →
  p = 25 := by
sorry


end percentage_increase_l1638_163825


namespace expression_values_l1638_163803

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / abs a + b / abs b + c / abs c + d / abs d + (a * b * c * d) / abs (a * b * c * d)
  expr = 5 ∨ expr = 1 ∨ expr = -3 ∨ expr = -5 := by
  sorry

end expression_values_l1638_163803


namespace some_number_value_l1638_163850

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = n * 25 * 45 * 49) : n = 3 := by
  sorry

end some_number_value_l1638_163850


namespace range_of_m_l1638_163807

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2 - x - y = 0

-- Define the point A
def A (m : ℝ) : ℝ × ℝ := (m, m)

-- Define the condition that any line through A intersects C
def intersects_C (m : ℝ) : Prop :=
  ∀ (k b : ℝ), ∃ (x y : ℝ), C x y ∧ y = k * x + b ∧ m * k + b = m

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, intersects_C m ↔ 0 ≤ m ∧ m ≤ 1 := by sorry

end range_of_m_l1638_163807


namespace hyperbola_asymptote_slope_l1638_163814

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 81 = 1

-- Define the asymptote equation
def asymptote (m x y : ℝ) : Prop := y = m * x ∨ y = -m * x

-- Theorem statement
theorem hyperbola_asymptote_slope :
  ∃ m : ℝ, (∀ x y : ℝ, hyperbola x y → asymptote m x y) ∧ m = 3/4 := by
  sorry

end hyperbola_asymptote_slope_l1638_163814


namespace no_integer_solutions_l1638_163861

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ),
    x^6 + x^3 + x^3 * y + y = 147^157 ∧
    x^3 + x^3 * y + y^2 + y + z^9 = 157^1177 :=
by sorry

end no_integer_solutions_l1638_163861


namespace a_is_perfect_square_l1638_163875

/-- Sequence c_n satisfying the given recurrence relation -/
def c : ℕ → ℤ
  | 0 => 1
  | 1 => 0
  | 2 => 2005
  | (n + 3) => -3 * c (n + 1) - 4 * c n + 2008

/-- Definition of a_n based on c_n -/
def a (n : ℕ) : ℤ :=
  5 * (c (n + 2) - c n) * (502 - c (n - 1) - c (n - 2)) + 4^n * 2004 * 501

/-- Theorem stating that a_n is a perfect square for n > 2 -/
theorem a_is_perfect_square (n : ℕ) (h : n > 2) : ∃ (k : ℤ), a n = k^2 := by
  sorry


end a_is_perfect_square_l1638_163875


namespace square_sum_implies_product_l1638_163865

theorem square_sum_implies_product (x : ℝ) :
  Real.sqrt (10 + x) + Real.sqrt (15 - x) = 6 →
  (10 + x) * (15 - x) = 121 / 4 := by
  sorry

end square_sum_implies_product_l1638_163865


namespace barn_paint_area_l1638_163866

/-- Calculates the total area to be painted for a rectangular barn with given dimensions and windows. -/
def total_paint_area (width length height window_width window_height window_count : ℕ) : ℕ :=
  let wall_area_1 := 2 * (width * height)
  let wall_area_2 := 2 * (length * height - window_width * window_height * window_count)
  let ceiling_area := width * length
  2 * (wall_area_1 + wall_area_2) + ceiling_area

/-- The total area to be painted for the given barn is 780 sq yd. -/
theorem barn_paint_area :
  total_paint_area 12 15 6 2 3 2 = 780 :=
by sorry

end barn_paint_area_l1638_163866


namespace arithmetic_sequence_product_l1638_163816

-- Define the arithmetic sequence
def arithmetic_sequence (b : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, b (n + 1) = b n + d

-- Define the increasing property
def increasing_sequence (b : ℕ → ℤ) : Prop :=
  ∀ n m, n < m → b n < b m

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) :
  arithmetic_sequence b d →
  increasing_sequence b →
  b 3 * b 4 = 21 →
  b 2 * b 5 = -11 := by
  sorry

end arithmetic_sequence_product_l1638_163816


namespace ellipse_focal_distance_l1638_163819

/-- Given an ellipse with equation x²/16 + y²/9 = 1, 
    the length of the focal distance is 2√7 -/
theorem ellipse_focal_distance : 
  let ellipse := {(x, y) : ℝ × ℝ | x^2/16 + y^2/9 = 1}
  ∃ c : ℝ, c = 2 * Real.sqrt 7 ∧ 
    ∀ (x y : ℝ), (x, y) ∈ ellipse → 
      c = Real.sqrt ((x^2 + y^2) - 4 * Real.sqrt (x^2 * y^2)) :=
sorry

end ellipse_focal_distance_l1638_163819


namespace period_of_f_l1638_163809

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f x

theorem period_of_f (f : ℝ → ℝ) (h : has_property f) : is_periodic f 4 := by
  sorry

end period_of_f_l1638_163809


namespace no_one_common_tangent_l1638_163841

/-- Represents a circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if two circles have different radii --/
def hasDifferentRadii (c1 c2 : Circle) : Prop :=
  c1.radius ≠ c2.radius

/-- Counts the number of common tangents between two circles --/
def commonTangentsCount (c1 c2 : Circle) : ℕ := sorry

/-- Theorem stating that two circles with different radii cannot have exactly one common tangent --/
theorem no_one_common_tangent (c1 c2 : Circle) 
  (h : hasDifferentRadii c1 c2) : 
  commonTangentsCount c1 c2 ≠ 1 := by sorry

end no_one_common_tangent_l1638_163841


namespace closest_vertex_of_dilated_square_l1638_163824

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  center : Point
  area : ℝ
  verticalSide : Bool

/-- Dilates a point from the origin by a given factor -/
def dilatePoint (p : Point) (factor : ℝ) : Point :=
  { x := p.x * factor, y := p.y * factor }

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Finds the vertex of a dilated square closest to the origin -/
def closestVertexToDilatedSquare (s : Square) (dilationFactor : ℝ) : Point :=
  sorry

theorem closest_vertex_of_dilated_square :
  let originalSquare : Square := {
    center := { x := 5, y := -3 },
    area := 16,
    verticalSide := true
  }
  let dilationFactor : ℝ := 3
  let closestVertex := closestVertexToDilatedSquare originalSquare dilationFactor
  closestVertex.x = 9 ∧ closestVertex.y = -3 := by
  sorry

end closest_vertex_of_dilated_square_l1638_163824


namespace f_properties_l1638_163882

def f (x : ℝ) := x^2 + x - 2

theorem f_properties :
  (∀ y : ℝ, y ∈ Set.Icc (-1) 1 → ∃ x : ℝ, x ∈ Set.Ico (-1) 1 ∧ f x > f y) ∧
  (∃ x : ℝ, x ∈ Set.Ico (-1) 1 ∧ f x = -9/4 ∧ ∀ y : ℝ, y ∈ Set.Ico (-1) 1 → f y ≥ f x) :=
by sorry

end f_properties_l1638_163882


namespace new_student_weight_l1638_163853

theorem new_student_weight (n : ℕ) (w_avg : ℝ) (w_new_avg : ℝ) :
  n = 29 →
  w_avg = 28 →
  w_new_avg = 27.5 →
  (n : ℝ) * w_avg + (n + 1) * w_new_avg - n * w_avg = 13 :=
by sorry

end new_student_weight_l1638_163853


namespace total_balloons_l1638_163858

-- Define the number of red and green balloons
def red_balloons : ℕ := 8
def green_balloons : ℕ := 9

-- Theorem stating that the total number of balloons is 17
theorem total_balloons : red_balloons + green_balloons = 17 := by
  sorry

end total_balloons_l1638_163858


namespace fraction_of_fraction_fraction_of_three_fifths_is_two_fifteenths_l1638_163880

theorem fraction_of_fraction (a b c d : ℚ) (h : a / b = c / d) :
  (c / d) / (a / b) = d / a :=
by sorry

theorem fraction_of_three_fifths_is_two_fifteenths :
  (2 / 15) / (3 / 5) = 2 / 9 :=
by sorry

end fraction_of_fraction_fraction_of_three_fifths_is_two_fifteenths_l1638_163880


namespace new_dwelling_points_order_l1638_163826

open Real

-- Define the "new dwelling point" for each function
def α : ℝ := 1

-- β is implicitly defined by the equation ln(β+1) = 1/(β+1)
def β : ℝ := sorry

-- γ is implicitly defined by the equation cos γ = -sin γ, where γ ∈ (π/2, π)
noncomputable def γ : ℝ := sorry

axiom β_eq : log (β + 1) = 1 / (β + 1)
axiom γ_eq : cos γ = -sin γ
axiom γ_range : π / 2 < γ ∧ γ < π

-- Theorem statement
theorem new_dwelling_points_order : γ > α ∧ α > β := by sorry

end new_dwelling_points_order_l1638_163826


namespace isosceles_triangle_largest_angle_l1638_163828

theorem isosceles_triangle_largest_angle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α = β →            -- The triangle is isosceles (two angles are equal)
  α = 50 →           -- One of the equal angles is 50°
  max α (max β γ) = 80 := by
sorry

end isosceles_triangle_largest_angle_l1638_163828


namespace max_bracelet_earnings_l1638_163877

theorem max_bracelet_earnings :
  let total_bracelets : ℕ := 235
  let bracelets_per_bag : ℕ := 10
  let price_per_bag : ℕ := 3000
  let full_bags : ℕ := total_bracelets / bracelets_per_bag
  let max_earnings : ℕ := full_bags * price_per_bag
  max_earnings = 69000 := by
  sorry

end max_bracelet_earnings_l1638_163877


namespace no_97_points_l1638_163870

/-- Represents the score on a test with the given scoring system -/
structure TestScore where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ
  total : correct + unanswered + incorrect = 20

/-- Calculates the total points for a given TestScore -/
def calculatePoints (score : TestScore) : ℕ :=
  5 * score.correct + score.unanswered

/-- Theorem stating that 97 points is not possible on the test -/
theorem no_97_points : ¬ ∃ (score : TestScore), calculatePoints score = 97 := by
  sorry


end no_97_points_l1638_163870


namespace polynomial_value_at_2_l1638_163847

-- Define the polynomial coefficients
def a₃ : ℝ := 7
def a₂ : ℝ := 3
def a₁ : ℝ := -5
def a₀ : ℝ := 11

-- Define the point at which to evaluate the polynomial
def x : ℝ := 2

-- Define Horner's method for a cubic polynomial
def horner_cubic (a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((a₃ * x + a₂) * x + a₁) * x + a₀

-- Theorem statement
theorem polynomial_value_at_2 :
  horner_cubic a₃ a₂ a₁ a₀ x = 69 := by
  sorry

end polynomial_value_at_2_l1638_163847


namespace unique_integer_satisfying_conditions_l1638_163896

theorem unique_integer_satisfying_conditions : ∃! (n : ℤ), n + 15 > 16 ∧ -3*n > -9 :=
  sorry

end unique_integer_satisfying_conditions_l1638_163896


namespace candle_burning_theorem_l1638_163838

theorem candle_burning_theorem (n : ℕ) (h : n > 0) :
  (∃ k : ℕ, k > 0 ∧ n * k = n * (n + 1) / 2) → Odd n :=
by
  sorry

#check candle_burning_theorem

end candle_burning_theorem_l1638_163838


namespace quadratic_roots_relation_l1638_163845

theorem quadratic_roots_relation (a b p q : ℝ) (r₁ r₂ : ℂ) : 
  (r₁ + r₂ = -a ∧ r₁ * r₂ = b) →  -- roots of x² + ax + b = 0
  (r₁^2 + r₂^2 = -p ∧ r₁^2 * r₂^2 = q) →  -- r₁² and r₂² are roots of x² + px + q = 0
  p = -a^2 + 2*b :=
by sorry

end quadratic_roots_relation_l1638_163845


namespace bobby_candy_consumption_l1638_163810

/-- The number of candies Bobby eats per day from Monday through Friday -/
def daily_candies : ℕ := 2

/-- The number of packets Bobby buys -/
def num_packets : ℕ := 2

/-- The number of candies in each packet -/
def candies_per_packet : ℕ := 18

/-- The number of weeks it takes Bobby to finish the packets -/
def num_weeks : ℕ := 3

/-- The number of candies Bobby eats on weekend days -/
def weekend_candies : ℕ := 1

theorem bobby_candy_consumption :
  daily_candies * 5 * num_weeks + weekend_candies * 2 * num_weeks = num_packets * candies_per_packet :=
sorry

end bobby_candy_consumption_l1638_163810


namespace quadratic_root_and_coefficient_l1638_163802

def quadratic_polynomial (x : ℂ) : ℂ := 3 * x^2 - 24 * x + 60

theorem quadratic_root_and_coefficient : 
  (quadratic_polynomial (4 + 2*Complex.I) = 0) ∧ 
  (∃ (a b : ℝ), ∀ x, quadratic_polynomial x = 3 * x^2 + a * x + b) :=
by sorry

end quadratic_root_and_coefficient_l1638_163802


namespace saltwater_solution_bounds_l1638_163835

theorem saltwater_solution_bounds :
  let solution_A : ℝ := 5  -- Concentration of solution A (%)
  let solution_B : ℝ := 8  -- Concentration of solution B (%)
  let solution_C : ℝ := 9  -- Concentration of solution C (%)
  let weight_A : ℝ := 60   -- Weight of solution A (g)
  let weight_B : ℝ := 60   -- Weight of solution B (g)
  let weight_C : ℝ := 47   -- Weight of solution C (g)
  let target_concentration : ℝ := 7  -- Target concentration (%)
  let target_weight : ℝ := 100       -- Target weight (g)

  ∀ x y z : ℝ,
    x + y + z = target_weight →
    solution_A * x + solution_B * y + solution_C * z = target_concentration * target_weight →
    0 ≤ x ∧ x ≤ weight_A →
    0 ≤ y ∧ y ≤ weight_B →
    0 ≤ z ∧ z ≤ weight_C →
    (∃ x_max : ℝ, x ≤ x_max ∧ x_max = 49) ∧
    (∃ x_min : ℝ, x_min ≤ x ∧ x_min = 35) :=
by sorry

end saltwater_solution_bounds_l1638_163835


namespace max_xy_constraint_l1638_163864

theorem max_xy_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5 * x + 8 * y = 65) :
  x * y ≤ 25 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 5 * x₀ + 8 * y₀ = 65 ∧ x₀ * y₀ = 25 := by
  sorry

end max_xy_constraint_l1638_163864


namespace smallest_z_satisfying_conditions_l1638_163894

theorem smallest_z_satisfying_conditions : ∃ (z : ℕ), z = 10 ∧ 
  (∀ (x y : ℕ), x > 0 ∧ y > 0 →
    (27 ^ z) * (5 ^ x) > (3 ^ 24) * (2 ^ y) ∧
    x + y = z ∧
    x * y < z ^ 2) ∧
  (∀ (z' : ℕ), z' < z →
    ¬(∃ (x y : ℕ), x > 0 ∧ y > 0 ∧
      (27 ^ z') * (5 ^ x) > (3 ^ 24) * (2 ^ y) ∧
      x + y = z' ∧
      x * y < z' ^ 2)) :=
by sorry

end smallest_z_satisfying_conditions_l1638_163894


namespace fred_paper_count_l1638_163893

theorem fred_paper_count (initial_sheets received_sheets given_sheets : ℕ) :
  initial_sheets + received_sheets - given_sheets =
  initial_sheets + received_sheets - given_sheets :=
by sorry

end fred_paper_count_l1638_163893


namespace max_soda_bottles_problem_l1638_163829

/-- Represents the maximum number of soda bottles that can be consumed given a certain amount of money, cost per bottle, and exchange rate for empty bottles. -/
def max_soda_bottles (total_money : ℚ) (cost_per_bottle : ℚ) (exchange_rate : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given 30 yuan, a soda cost of 2.5 yuan per bottle, and the ability to exchange 3 empty bottles for 1 new bottle, the maximum number of soda bottles that can be consumed is 18. -/
theorem max_soda_bottles_problem :
  max_soda_bottles 30 2.5 3 = 18 :=
sorry

end max_soda_bottles_problem_l1638_163829


namespace decreasing_function_implies_a_geq_3_l1638_163813

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 6

theorem decreasing_function_implies_a_geq_3 :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 3 → f a x₁ > f a x₂) → a ≥ 3 := by
  sorry

end decreasing_function_implies_a_geq_3_l1638_163813


namespace circles_internally_tangent_l1638_163899

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 12 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 14*x - 2*y + 14 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (3, -2)
def center2 : ℝ × ℝ := (7, 1)
def radius1 : ℝ := 1
def radius2 : ℝ := 6

-- Theorem stating that the circles are internally tangent
theorem circles_internally_tangent :
  let d := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  d = radius2 - radius1 :=
sorry

end circles_internally_tangent_l1638_163899


namespace coin_problem_l1638_163820

theorem coin_problem (total_coins : ℕ) (total_value : ℕ) 
  (pennies nickels dimes quarters : ℕ) :
  total_coins = 11 →
  total_value = 165 →
  pennies ≥ 1 →
  nickels ≥ 1 →
  dimes ≥ 1 →
  quarters ≥ 1 →
  total_coins = pennies + nickels + dimes + quarters →
  total_value = pennies + 5 * nickels + 10 * dimes + 25 * quarters →
  quarters = 4 :=
by sorry

end coin_problem_l1638_163820


namespace savings_to_earnings_ratio_l1638_163876

/-- Proves that the ratio of monthly savings to monthly earnings is 1/2 -/
theorem savings_to_earnings_ratio
  (monthly_earnings : ℕ)
  (vehicle_cost : ℕ)
  (saving_period : ℕ)
  (h1 : monthly_earnings = 4000)
  (h2 : vehicle_cost = 16000)
  (h3 : saving_period = 8) :
  (vehicle_cost / saving_period) / monthly_earnings = 1 / 2 :=
by sorry

end savings_to_earnings_ratio_l1638_163876


namespace preimage_of_two_three_l1638_163836

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

-- Theorem statement
theorem preimage_of_two_three :
  ∃ (x y : ℝ), f (x, y) = (2, 3) ∧ x = 5/2 ∧ y = -1/2 := by
  sorry

end preimage_of_two_three_l1638_163836


namespace matching_probability_is_four_fifteenths_l1638_163890

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.green + jb.red + jb.blue + jb.yellow

/-- Alice's jelly bean distribution -/
def alice : JellyBeans := { green := 2, red := 2, blue := 1, yellow := 0 }

/-- Carl's jelly bean distribution -/
def carl : JellyBeans := { green := 3, red := 1, blue := 0, yellow := 2 }

/-- The probability of selecting matching colors -/
def matchingProbability (a c : JellyBeans) : ℚ :=
  (a.green * c.green + a.red * c.red) / (a.total * c.total)

theorem matching_probability_is_four_fifteenths :
  matchingProbability alice carl = 4 / 15 := by
  sorry

end matching_probability_is_four_fifteenths_l1638_163890


namespace solution_characterization_l1638_163891

def equation (x y z : ℝ) : Prop :=
  Real.sqrt (3^x * (5^y + 7^z)) + Real.sqrt (5^y * (7^z + 3^x)) + Real.sqrt (7^z * (3^x + 5^y)) = 
  Real.sqrt 2 * (3^x + 5^y + 7^z)

theorem solution_characterization (x y z : ℝ) :
  equation x y z → ∃ t : ℝ, x = t / Real.log 3 ∧ y = t / Real.log 5 ∧ z = t / Real.log 7 := by
  sorry

end solution_characterization_l1638_163891


namespace lcm_problem_l1638_163874

theorem lcm_problem (n : ℕ+) : ∃ n, 
  n > 0 ∧ 
  216 % n = 0 ∧ 
  n % 3 = 0 ∧ 
  n ≤ 9 ∧
  Nat.lcm (Nat.lcm (Nat.lcm 8 24) 36) n = 216 := by
sorry

end lcm_problem_l1638_163874


namespace middle_term_value_l1638_163840

/-- An arithmetic sequence with 9 terms -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

theorem middle_term_value
  (a : ℕ → ℚ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum_first_4 : (a 1) + (a 2) + (a 3) + (a 4) = 3)
  (h_sum_last_3 : (a 7) + (a 8) + (a 9) = 4) :
  a 5 = 19 / 148 := by
  sorry

end middle_term_value_l1638_163840


namespace set_operations_l1638_163872

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 1 ∨ x ≤ -3}
def B : Set ℝ := {x | -4 < x ∧ x < 0}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x | -4 < x ∧ x ≤ -3}) ∧
  (A ∪ B = {x | x < 0 ∨ x ≥ 1}) ∧
  (A ∪ (Set.univ \ B) = {x | x ≤ -3 ∨ x ≥ 0}) := by
  sorry

end set_operations_l1638_163872


namespace die_events_l1638_163801

-- Define the sample space and events
def Ω : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1}
def B : Set Nat := {2, 4, 6}
def C : Set Nat := {1, 2}
def D : Set Nat := {3, 4, 5, 6}
def E : Set Nat := {3, 6}

-- Theorem to prove the relationships and set operations
theorem die_events :
  (A ⊆ C) ∧
  (C ∪ D = Ω) ∧
  (E ⊆ D) ∧
  (Dᶜ = {1, 2}) ∧
  (Aᶜ ∩ C = {2}) ∧
  (Bᶜ ∪ C = {1, 2, 3}) ∧
  (Dᶜ ∪ Eᶜ = {1, 2, 4, 5}) :=
by sorry

end die_events_l1638_163801


namespace max_triangle_area_l1638_163897

-- Define the curve C
def C (x y : ℝ) : Prop :=
  Real.sqrt ((x + 1)^2 + y^2) + Real.sqrt ((x - 1)^2 + y^2) + Real.sqrt (x^2 + (y - 1)^2) = 2 * Real.sqrt 2

-- Define the area of triangle F₁PF₂
def triangle_area (x y : ℝ) : ℝ :=
  abs (y) -- The base of the triangle is 2, so the area is |y|

-- Theorem statement
theorem max_triangle_area :
  ∀ x y : ℝ, C x y → triangle_area x y ≤ 1 :=
sorry

end max_triangle_area_l1638_163897


namespace complex_number_problem_l1638_163837

theorem complex_number_problem (a : ℝ) (z : ℂ) : 
  z = (1 + a * Complex.I) / Complex.I → 
  z.re = 1 → 
  a = 1 ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_number_problem_l1638_163837


namespace multiplication_and_exponentiation_l1638_163832

theorem multiplication_and_exponentiation : 121 * (5^4) = 75625 := by
  sorry

end multiplication_and_exponentiation_l1638_163832


namespace entrance_exam_marks_l1638_163817

/-- Proves that the number of marks awarded for each correct answer is 3 -/
theorem entrance_exam_marks : 
  ∀ (total_questions correct_answers total_marks : ℕ) 
    (wrong_answer_penalty : ℤ),
  total_questions = 70 →
  correct_answers = 27 →
  total_marks = 38 →
  wrong_answer_penalty = -1 →
  ∃ (marks_per_correct_answer : ℕ),
    marks_per_correct_answer * correct_answers + 
    wrong_answer_penalty * (total_questions - correct_answers) = total_marks ∧
    marks_per_correct_answer = 3 :=
by sorry

end entrance_exam_marks_l1638_163817


namespace inequality_equivalence_l1638_163848

theorem inequality_equivalence (x : ℝ) : 
  -1 < (x^2 - 12*x + 35) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 12*x + 35) / (x^2 - 4*x + 8) < 1 ↔ 
  x > 27/8 := by
  sorry

end inequality_equivalence_l1638_163848


namespace am_gm_inequality_l1638_163806

theorem am_gm_inequality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a ≤ b) :
  (b - a)^3 / (8 * b) > (a + b) / 2 - Real.sqrt (a * b) :=
by sorry

end am_gm_inequality_l1638_163806


namespace v_2008_value_l1638_163804

-- Define the sequence v_n
def v : ℕ → ℕ 
| n => sorry  -- The exact definition would be complex to write out

-- Define the function g(n) for the last term in a group with n terms
def g (n : ℕ) : ℕ := 2 * n^2 - 3 * n + 2

-- Define the function for the total number of terms up to and including group n
def totalTerms (n : ℕ) : ℕ := n * (n + 1) / 2

-- The theorem to prove
theorem v_2008_value : v 2008 = 7618 := by sorry

end v_2008_value_l1638_163804


namespace existence_of_uncuttable_rectangle_l1638_163822

/-- A rectangle with natural number side lengths -/
structure Rectangle where
  length : ℕ+
  width : ℕ+

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- A predicate that checks if two numbers are almost equal -/
def almost_equal (a b : ℕ) : Prop := a = b ∨ a = b + 1 ∨ a = b - 1

/-- A predicate that checks if a rectangle can be cut out from another rectangle -/
def can_cut_out (small big : Rectangle) : Prop :=
  small.length ≤ big.length ∧ small.width ≤ big.width ∨
  small.length ≤ big.width ∧ small.width ≤ big.length

theorem existence_of_uncuttable_rectangle :
  ∃ (r : Rectangle), ¬∃ (s : Rectangle), 
    can_cut_out s r ∧ almost_equal (area s) ((area r) / 2) :=
sorry

end existence_of_uncuttable_rectangle_l1638_163822


namespace lcm_16_24_l1638_163811

theorem lcm_16_24 : Nat.lcm 16 24 = 48 := by sorry

end lcm_16_24_l1638_163811


namespace expected_zeroes_l1638_163863

/-- Represents the probability of getting heads on the unfair coin. -/
def A : ℚ := 1/5

/-- Represents the length of the generated string. -/
def B : ℕ := 4

/-- Represents the expected number of zeroes in the string. -/
def C : ℚ := B/2

/-- Proves that the expected number of zeroes in the string is half its length,
    and that for the given probabilities, the string length is 4. -/
theorem expected_zeroes :
  C = B/2 ∧ A = 3*B/((B+1)*(B+2)*2) ∧ B = (4 - A*B)/(4*A) := by
  sorry

#eval C  -- Should output 2

end expected_zeroes_l1638_163863


namespace perfect_correlation_l1638_163805

/-- A sample point in a 2D plane -/
structure SamplePoint where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The correlation coefficient -/
def correlationCoefficient (points : List SamplePoint) : ℝ :=
  sorry

/-- Theorem: If all sample points lie on a straight line with non-zero slope, 
    then the correlation coefficient R^2 is 1 -/
theorem perfect_correlation 
  (points : List SamplePoint) 
  (line : Line) 
  (h1 : line.slope ≠ 0) 
  (h2 : ∀ p ∈ points, p.y = line.slope * p.x + line.intercept) : 
  correlationCoefficient points = 1 :=
sorry

end perfect_correlation_l1638_163805


namespace dislike_sector_angle_l1638_163815

-- Define the ratios for the four categories
def ratio_extremely_like : ℕ := 6
def ratio_like : ℕ := 9
def ratio_somewhat_like : ℕ := 2
def ratio_dislike : ℕ := 1

-- Define the total ratio
def total_ratio : ℕ := ratio_extremely_like + ratio_like + ratio_somewhat_like + ratio_dislike

-- Define the central angle of the dislike sector
def central_angle_dislike : ℚ := (ratio_dislike : ℚ) / (total_ratio : ℚ) * 360

-- Theorem statement
theorem dislike_sector_angle :
  central_angle_dislike = 20 := by sorry

end dislike_sector_angle_l1638_163815


namespace special_linear_function_properties_l1638_163895

/-- A linear function y = mx + c, where m is the slope and c is the y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- The linear function y = (2a-4)x + (3-b) -/
def specialLinearFunction (a b : ℝ) : LinearFunction where
  slope := 2*a - 4
  intercept := 3 - b

theorem special_linear_function_properties (a b : ℝ) :
  let f := specialLinearFunction a b
  (∃ k : ℝ, ∀ x, f.slope * x = k * x) ↔ (a ≠ 2 ∧ b = 3) ∧
  (f.slope < 0 ∧ f.intercept ≤ 0) ↔ (a < 2 ∧ b ≥ 3) := by
  sorry

end special_linear_function_properties_l1638_163895


namespace exponent_division_l1638_163878

theorem exponent_division (a : ℝ) : a^6 / a^3 = a^3 := by
  sorry

end exponent_division_l1638_163878


namespace perfect_square_trinomial_l1638_163868

theorem perfect_square_trinomial (x y : ℝ) :
  x^2 - x*y + (1/4)*y^2 = (x - (1/2)*y)^2 := by sorry

end perfect_square_trinomial_l1638_163868


namespace series_sum_equals_two_l1638_163883

/-- Given a real number k > 1 such that the infinite sum of (7n-3)/k^n from n=1 to infinity equals 2,
    prove that k = 2 + (3√2)/2 -/
theorem series_sum_equals_two (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (7 * n - 3) / k^n = 2) : k = 2 + 3 * Real.sqrt 2 / 2 := by
  sorry

end series_sum_equals_two_l1638_163883


namespace common_roots_product_l1638_163860

-- Define the polynomials
def p (C : ℝ) (x : ℝ) : ℝ := x^3 + C*x^2 - 20
def q (D : ℝ) (x : ℝ) : ℝ := x^3 + D*x - 80

-- Define the theorem
theorem common_roots_product (C D : ℝ) :
  ∃ (r₁ r₂ : ℝ) (a b c : ℕ),
    (p C r₁ = 0 ∧ q D r₁ = 0) ∧
    (p C r₂ = 0 ∧ q D r₂ = 0) ∧
    r₁ ≠ r₂ ∧
    (r₁ * r₂ = a * (c ^ (1 / b : ℝ))) ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = 25 :=
  sorry

end common_roots_product_l1638_163860


namespace percentage_increase_l1638_163851

theorem percentage_increase (initial final : ℝ) (h : initial > 0) :
  let increase := (final - initial) / initial * 100
  initial = 200 ∧ final = 250 → increase = 25 := by
  sorry

end percentage_increase_l1638_163851


namespace mean_height_is_68_25_l1638_163856

def heights : List ℕ := [57, 59, 62, 64, 64, 65, 65, 68, 69, 70, 71, 73, 75, 75, 77, 78]

theorem mean_height_is_68_25 : 
  let total_height : ℕ := heights.sum
  let num_players : ℕ := heights.length
  (total_height : ℚ) / num_players = 68.25 := by
sorry

end mean_height_is_68_25_l1638_163856


namespace five_balls_four_boxes_l1638_163830

/-- The number of ways to place n distinguishable balls into k distinguishable boxes -/
def placeBalls (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 1024 ways to place 5 distinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : placeBalls 5 4 = 1024 := by
  sorry

end five_balls_four_boxes_l1638_163830


namespace repair_time_30_workers_l1638_163834

/-- Represents the time taken to complete a road repair job given the number of workers -/
def repair_time (num_workers : ℕ) : ℚ :=
  3 * 45 / num_workers

/-- Proves that 30 workers would take 4.5 days to complete the road repair -/
theorem repair_time_30_workers :
  repair_time 30 = 4.5 := by sorry

end repair_time_30_workers_l1638_163834
