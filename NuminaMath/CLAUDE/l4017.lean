import Mathlib

namespace NUMINAMATH_CALUDE_shirt_tie_combinations_l4017_401757

theorem shirt_tie_combinations (num_shirts num_ties : ℕ) : 
  num_shirts = 8 → num_ties = 7 → num_shirts * num_ties = 56 := by
  sorry

end NUMINAMATH_CALUDE_shirt_tie_combinations_l4017_401757


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l4017_401797

theorem cylinder_surface_area (r l : ℝ) : 
  r = 1 → l = 2*r → 2*π*r*(r + l) = 6*π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l4017_401797


namespace NUMINAMATH_CALUDE_complex_number_theorem_l4017_401774

def complex_number_problem (z₁ z₂ : ℂ) : Prop :=
  Complex.abs (z₁ * z₂) = 3 ∧ z₁ + z₂ = Complex.I * 2

theorem complex_number_theorem (z₁ z₂ : ℂ) 
  (h : complex_number_problem z₁ z₂) :
  (∀ w₁ w₂ : ℂ, complex_number_problem w₁ w₂ → Complex.abs w₁ ≤ 3) ∧
  (∀ w₁ w₂ : ℂ, complex_number_problem w₁ w₂ → Complex.abs w₁ ≥ 1) ∧
  (∃ w₁ w₂ : ℂ, complex_number_problem w₁ w₂ ∧ Complex.abs w₁ = 3) ∧
  (∃ w₁ w₂ : ℂ, complex_number_problem w₁ w₂ ∧ Complex.abs w₁ = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l4017_401774


namespace NUMINAMATH_CALUDE_soap_box_length_l4017_401779

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Proves that the length of each soap box is 7 inches -/
theorem soap_box_length 
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (h_carton_length : carton.length = 25)
  (h_carton_width : carton.width = 42)
  (h_carton_height : carton.height = 60)
  (h_soap_width : soap.width = 6)
  (h_soap_height : soap.height = 5)
  (h_max_boxes : ↑300 * boxVolume soap = boxVolume carton) :
  soap.length = 7 := by
  sorry

end NUMINAMATH_CALUDE_soap_box_length_l4017_401779


namespace NUMINAMATH_CALUDE_max_sum_of_three_primes_l4017_401776

theorem max_sum_of_three_primes (a b c : ℕ) : 
  Prime a → Prime b → Prime c →
  a < b → b < c → c < 100 →
  (b - a) * (c - b) * (c - a) = 240 →
  a + b + c ≤ 111 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_three_primes_l4017_401776


namespace NUMINAMATH_CALUDE_factorization_equality_l4017_401706

theorem factorization_equality (m : ℝ) : m^2 + 3*m = m*(m+3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l4017_401706


namespace NUMINAMATH_CALUDE_power_of_fraction_l4017_401724

theorem power_of_fraction (x y : ℝ) : 
  (-(1/3) * x^2 * y)^3 = -(x^6 * y^3) / 27 := by
  sorry

end NUMINAMATH_CALUDE_power_of_fraction_l4017_401724


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l4017_401745

theorem quadratic_roots_problem (m : ℝ) (x₁ x₂ : ℝ) :
  (x₁^2 + 2*(m+1)*x₁ + m^2 - 1 = 0) →
  (x₂^2 + 2*(m+1)*x₂ + m^2 - 1 = 0) →
  ((x₁ - x₂)^2 = 16 - x₁*x₂) →
  (m = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l4017_401745


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l4017_401727

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    (P = -8/15 ∧ Q = -7/6 ∧ R = 27/10) ∧
    ∀ (x : ℚ), x ≠ 1 → x ≠ 4 → x ≠ 6 →
      (x^2 - 9) / ((x - 1) * (x - 4) * (x - 6)) =
      P / (x - 1) + Q / (x - 4) + R / (x - 6) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l4017_401727


namespace NUMINAMATH_CALUDE_equation_solutions_l4017_401744

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  log (x^2 + 1) - 2 * log (x + 3) + log 2 = 0

-- State the theorem
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = -1 ∨ x = 7) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l4017_401744


namespace NUMINAMATH_CALUDE_no_integer_square_root_product_l4017_401729

theorem no_integer_square_root_product (n1 n2 : ℤ) : 
  (n1 : ℚ) / n2 = 3 / 4 →
  n1 + n2 = 21 →
  n2 > n1 →
  ¬ ∃ (n3 : ℤ), n1 * n2 = n3^2 := by
sorry

end NUMINAMATH_CALUDE_no_integer_square_root_product_l4017_401729


namespace NUMINAMATH_CALUDE_correct_swap_l4017_401788

def swap_values (m n : ℕ) : ℕ × ℕ := 
  let s := m
  let m' := n
  let n' := s
  (m', n')

theorem correct_swap : 
  ∀ (m n : ℕ), swap_values m n = (n, m) := by
  sorry

end NUMINAMATH_CALUDE_correct_swap_l4017_401788


namespace NUMINAMATH_CALUDE_f_continuous_at_x₀_delta_epsilon_relation_l4017_401716

def f (x : ℝ) : ℝ := 5 * x^2 + 1

def x₀ : ℝ := 7

theorem f_continuous_at_x₀ :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - f x₀| < ε :=
sorry

theorem delta_epsilon_relation :
  ∀ ε > 0, ∃ δ > 0, δ = ε / 70 ∧
    ∀ x, |x - x₀| < δ → |f x - f x₀| < ε :=
sorry

end NUMINAMATH_CALUDE_f_continuous_at_x₀_delta_epsilon_relation_l4017_401716


namespace NUMINAMATH_CALUDE_unique_pairs_satisfying_equation_l4017_401749

theorem unique_pairs_satisfying_equation :
  ∀ n m : ℕ, m^2 + 2 * 3^n = m * (2^(n + 1) - 1) ↔ (n = 3 ∧ m = 6) ∨ (n = 3 ∧ m = 9) :=
by sorry

end NUMINAMATH_CALUDE_unique_pairs_satisfying_equation_l4017_401749


namespace NUMINAMATH_CALUDE_square_minus_twice_plus_one_equals_three_l4017_401773

theorem square_minus_twice_plus_one_equals_three :
  let x : ℝ := Real.sqrt 3 + 1
  x^2 - 2*x + 1 = 3 := by sorry

end NUMINAMATH_CALUDE_square_minus_twice_plus_one_equals_three_l4017_401773


namespace NUMINAMATH_CALUDE_at_most_one_integer_root_l4017_401735

theorem at_most_one_integer_root (k : ℝ) :
  ∃! (n : ℤ), (n : ℝ)^3 - 24*(n : ℝ) + k = 0 ∨
  ∀ (m : ℤ), (m : ℝ)^3 - 24*(m : ℝ) + k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_at_most_one_integer_root_l4017_401735


namespace NUMINAMATH_CALUDE_age_ratio_l4017_401766

/-- Represents the ages of two people A and B -/
structure Ages where
  a : ℕ  -- Present age of A
  b : ℕ  -- Present age of B

/-- Conditions for the age problem -/
def AgeConditions (ages : Ages) : Prop :=
  (ages.a - 10 = (ages.b - 10) / 2) ∧ (ages.a + ages.b = 35)

/-- Theorem stating the ratio of present ages -/
theorem age_ratio (ages : Ages) (h : AgeConditions ages) : 
  (ages.a : ℚ) / ages.b = 3 / 4 := by
  sorry

#check age_ratio

end NUMINAMATH_CALUDE_age_ratio_l4017_401766


namespace NUMINAMATH_CALUDE_no_integer_root_seven_l4017_401769

theorem no_integer_root_seven
  (P : Int → Int)  -- P is a polynomial with integer coefficients
  (h_int_coeff : ∀ x, ∃ y, P x = y)  -- P has integer coefficients
  (a b c d : Int)  -- a, b, c, d are integers
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)  -- a, b, c, d are distinct
  (h_equal_four : P a = 4 ∧ P b = 4 ∧ P c = 4 ∧ P d = 4)  -- P(a) = P(b) = P(c) = P(d) = 4
  : ¬ ∃ e : Int, P e = 7 := by  -- There does not exist an integer e such that P(e) = 7
  sorry

end NUMINAMATH_CALUDE_no_integer_root_seven_l4017_401769


namespace NUMINAMATH_CALUDE_pizza_promotion_savings_l4017_401746

/-- Calculates the total savings from a pizza promotion -/
theorem pizza_promotion_savings 
  (regular_price : ℕ) 
  (promo_price : ℕ) 
  (num_pizzas : ℕ) 
  (h1 : regular_price = 18) 
  (h2 : promo_price = 5) 
  (h3 : num_pizzas = 3) : 
  (regular_price - promo_price) * num_pizzas = 39 := by
  sorry

#check pizza_promotion_savings

end NUMINAMATH_CALUDE_pizza_promotion_savings_l4017_401746


namespace NUMINAMATH_CALUDE_special_triples_characterization_l4017_401784

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℤ → ℤ := fun x ↦ a * x^2 + b * x + c

/-- The property that for any integer n, there exists an integer m such that f(m) = f(n)f(n+1) -/
def HasSpecialProperty (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, ∃ m : ℤ, f m = f n * f (n + 1)

/-- The set of all integer triples (a, b, c) satisfying the special property -/
def SpecialTriples : Set (ℤ × ℤ × ℤ) :=
  {abc | let (a, b, c) := abc
         a ≠ 0 ∧ HasSpecialProperty (QuadraticFunction a b c)}

/-- The characterization of the special triples -/
def CharacterizedTriples : Set (ℤ × ℤ × ℤ) :=
  {abc | let (a, b, c) := abc
         (a = 1) ∨
         (∃ k l : ℤ, k > 0 ∧ a = k^2 ∧ b = 2*k*l ∧ c = l^2 ∧
          (k ∣ (l^2 - l) ∨ k ∣ (l^2 + l)))}

theorem special_triples_characterization :
  SpecialTriples = CharacterizedTriples :=
sorry


end NUMINAMATH_CALUDE_special_triples_characterization_l4017_401784


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l4017_401750

theorem simplify_sqrt_sum : 
  Real.sqrt (8 + 4 * Real.sqrt 3) + Real.sqrt (8 - 4 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l4017_401750


namespace NUMINAMATH_CALUDE_moving_circle_theorem_l4017_401726

-- Define the circles F1 and F2
def F1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def F2 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the locus of the center of E
def E_locus (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the slope range
def slope_range (k : ℝ) : Prop := 
  (k ≥ -Real.sqrt 6 / 4 ∧ k < 0) ∨ (k > 0 ∧ k ≤ Real.sqrt 6 / 4)

-- State the theorem
theorem moving_circle_theorem 
  (E : ℝ → ℝ → Prop) -- The moving circle E
  (A B M H : ℝ × ℝ) -- Points A, B, M, H
  (l : ℝ → ℝ) -- Line l
  (h1 : ∀ x y, E x y → (∃ r > 0, ∀ u v, F1 u v → ((x - u)^2 + (y - v)^2 = (r + 1)^2))) -- E externally tangent to F1
  (h2 : ∀ x y, E x y → (∃ r > 0, ∀ u v, F2 u v → ((x - u)^2 + (y - v)^2 = (3 - r)^2))) -- E internally tangent to F2
  (h3 : A.1 > 0 ∧ A.2 = 0 ∧ E A.1 A.2) -- A on positive x-axis and on E
  (h4 : E B.1 B.2 ∧ B.2 ≠ 0) -- B on E and not on x-axis
  (h5 : ∀ x, l x = (B.2 / (B.1 - A.1)) * (x - A.1)) -- l passes through A and B
  (h6 : M.2 = l M.1 ∧ H.1 = 0) -- M on l, H on y-axis
  (h7 : (B.1 - 1) * (H.1 - 1) + B.2 * H.2 = 0) -- BF2 ⊥ HF2
  (h8 : (M.1 - A.1)^2 + (M.2 - A.2)^2 ≥ M.1^2 + M.2^2) -- ∠MOA ≥ ∠MAO
  : (∀ x y, E x y ↔ E_locus x y) ∧ 
    (∀ k, (∃ x, l x = k * (x - A.1)) → slope_range k) :=
sorry

end NUMINAMATH_CALUDE_moving_circle_theorem_l4017_401726


namespace NUMINAMATH_CALUDE_unique_s_value_l4017_401710

theorem unique_s_value : ∃! s : ℝ, ∀ x : ℝ, 
  (3 * x^2 - 4 * x + 8) * (5 * x^2 + s * x + 15) = 
  15 * x^4 - 29 * x^3 + 87 * x^2 - 60 * x + 120 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_s_value_l4017_401710


namespace NUMINAMATH_CALUDE_mikes_books_l4017_401748

/-- Calculates the final number of books Mike has after selling, receiving gifts, and buying books. -/
def final_book_count (initial : ℝ) (sold : ℝ) (gifts : ℝ) (bought : ℝ) : ℝ :=
  initial - sold + gifts + bought

/-- Theorem stating that Mike's final book count is 21.5 given the problem conditions. -/
theorem mikes_books :
  final_book_count 51.5 45.75 12.25 3.5 = 21.5 := by
  sorry

end NUMINAMATH_CALUDE_mikes_books_l4017_401748


namespace NUMINAMATH_CALUDE_flour_cost_for_cheapest_pie_l4017_401796

/-- The cost of flour for the cheapest pie -/
def flour_cost : ℝ := 2

/-- The cost of sugar for both pies -/
def sugar_cost : ℝ := 1

/-- The cost of eggs and butter for both pies -/
def eggs_butter_cost : ℝ := 1.5

/-- The weight of blueberries needed for the blueberry pie in pounds -/
def blueberry_weight : ℝ := 3

/-- The weight of a container of blueberries in ounces -/
def blueberry_container_weight : ℝ := 8

/-- The cost of a container of blueberries -/
def blueberry_container_cost : ℝ := 2.25

/-- The weight of cherries needed for the cherry pie in pounds -/
def cherry_weight : ℝ := 4

/-- The cost of a four-pound bag of cherries -/
def cherry_bag_cost : ℝ := 14

/-- The total price to make the cheapest pie -/
def cheapest_pie_cost : ℝ := 18

theorem flour_cost_for_cheapest_pie :
  flour_cost = cheapest_pie_cost - min
    (sugar_cost + eggs_butter_cost + (blueberry_weight * 16 / blueberry_container_weight) * blueberry_container_cost)
    (sugar_cost + eggs_butter_cost + cherry_bag_cost) :=
by sorry

end NUMINAMATH_CALUDE_flour_cost_for_cheapest_pie_l4017_401796


namespace NUMINAMATH_CALUDE_triangle_properties_l4017_401755

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  area : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.A = π / 6)
  (h2 : (1 + Real.sqrt 3) * Real.sin t.B = 2 * Real.sin t.C)
  (h3 : t.area = 2 + 2 * Real.sqrt 3) :
  (t.b = Real.sqrt 2 * t.a) ∧ (t.b = 4) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l4017_401755


namespace NUMINAMATH_CALUDE_purely_imaginary_fraction_l4017_401783

theorem purely_imaginary_fraction (a : ℝ) : 
  (Complex.I * ((a - Complex.I) / (1 + Complex.I))).re = ((a - Complex.I) / (1 + Complex.I)).re → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_fraction_l4017_401783


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_minimum_value_condition_function_inequality_condition_l4017_401785

open Real

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + 2 * a * x

theorem tangent_line_at_origin (h : ℝ → ℝ := fun x ↦ Real.exp x + 2 * x) :
  ∃ (m b : ℝ), m = 3 ∧ b = 1 ∧ ∀ x y, y = h x → m * x - y + b = 0 := by sorry

theorem minimum_value_condition (a : ℝ) :
  (∀ x ≥ 1, f a x ≥ 0) ∧ (∃ x ≥ 1, f a x = 0) → a = -Real.exp 1 / 2 := by sorry

theorem function_inequality_condition (a : ℝ) :
  (∀ x ≥ 0, f a x ≥ Real.exp (-x)) ↔ a ≥ -1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_minimum_value_condition_function_inequality_condition_l4017_401785


namespace NUMINAMATH_CALUDE_uma_income_is_20000_l4017_401738

-- Define the income ratio
def income_ratio : ℚ := 4 / 3

-- Define the expenditure ratio
def expenditure_ratio : ℚ := 3 / 2

-- Define the savings amount
def savings : ℕ := 5000

-- Define Uma's income as a function of x
def uma_income (x : ℚ) : ℚ := 4 * x

-- Define Bala's income as a function of x
def bala_income (x : ℚ) : ℚ := 3 * x

-- Define Uma's expenditure as a function of y
def uma_expenditure (y : ℚ) : ℚ := 3 * y

-- Define Bala's expenditure as a function of y
def bala_expenditure (y : ℚ) : ℚ := 2 * y

-- Theorem stating Uma's income is $20000
theorem uma_income_is_20000 :
  ∃ (x y : ℚ),
    uma_income x - uma_expenditure y = savings ∧
    bala_income x - bala_expenditure y = savings ∧
    uma_income x = 20000 :=
  sorry

end NUMINAMATH_CALUDE_uma_income_is_20000_l4017_401738


namespace NUMINAMATH_CALUDE_cistern_fill_time_l4017_401787

/-- The time it takes for pipe p to fill the cistern -/
def p_time : ℝ := 10

/-- The time both pipes are opened together -/
def both_open_time : ℝ := 2

/-- The additional time it takes to fill the cistern after pipe p is turned off -/
def additional_time : ℝ := 10

/-- The time it takes for pipe q to fill the cistern -/
def q_time : ℝ := 15

theorem cistern_fill_time : 
  (both_open_time * (1 / p_time + 1 / q_time)) + 
  (additional_time * (1 / q_time)) = 1 := by sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l4017_401787


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l4017_401705

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_planes
  (α β : Plane) (m : Line)
  (h1 : perpendicular m α)
  (h2 : parallel α β) :
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l4017_401705


namespace NUMINAMATH_CALUDE_number_divided_by_five_equals_number_plus_three_l4017_401792

theorem number_divided_by_five_equals_number_plus_three : 
  ∃ x : ℚ, x / 5 = x + 3 ∧ x = -15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_five_equals_number_plus_three_l4017_401792


namespace NUMINAMATH_CALUDE_minimum_total_balls_l4017_401752

/-- Given a set of balls with red, blue, and green colors, prove that there are at least 23 balls in total -/
theorem minimum_total_balls (red green blue : ℕ) : 
  green = 12 → red + green < 24 → red + green + blue ≥ 23 := by
  sorry

end NUMINAMATH_CALUDE_minimum_total_balls_l4017_401752


namespace NUMINAMATH_CALUDE_last_two_digits_a_2015_l4017_401715

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => if n % 2 = 0 then a n + 2 else 2 * a n

theorem last_two_digits_a_2015 : a 2015 % 100 = 72 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_a_2015_l4017_401715


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l4017_401722

theorem diophantine_equation_solution (x y z : ℤ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l4017_401722


namespace NUMINAMATH_CALUDE_total_players_on_ground_l4017_401732

/-- The number of cricket players -/
def cricket_players : ℕ := 16

/-- The number of hockey players -/
def hockey_players : ℕ := 12

/-- The number of football players -/
def football_players : ℕ := 18

/-- The number of softball players -/
def softball_players : ℕ := 13

/-- Theorem: The total number of players on the ground is 59 -/
theorem total_players_on_ground : 
  cricket_players + hockey_players + football_players + softball_players = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_players_on_ground_l4017_401732


namespace NUMINAMATH_CALUDE_sum_in_interval_l4017_401711

theorem sum_in_interval :
  let a := 2 + 3 / 9
  let b := 3 + 3 / 4
  let c := 5 + 3 / 25
  let sum := a + b + c
  8 < sum ∧ sum < 9 := by
sorry

end NUMINAMATH_CALUDE_sum_in_interval_l4017_401711


namespace NUMINAMATH_CALUDE_dihedral_angle_is_45_degrees_l4017_401723

-- Define the regular triangular prism
structure RegularTriangularPrism :=
  (A B C A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 3))

-- Define points D and E on the lateral edges
def D (prism : RegularTriangularPrism) : EuclideanSpace ℝ (Fin 3) := sorry
def E (prism : RegularTriangularPrism) : EuclideanSpace ℝ (Fin 3) := sorry

-- Define the conditions
def conditions (prism : RegularTriangularPrism) : Prop :=
  (dist (E prism) (prism.C) = dist prism.B prism.C) ∧
  (dist (E prism) (prism.C) = 2 * dist (D prism) prism.B)

-- Define the dihedral angle between ADE and ABC
def dihedralAngle (prism : RegularTriangularPrism) : ℝ := sorry

-- State the theorem
theorem dihedral_angle_is_45_degrees (prism : RegularTriangularPrism) :
  conditions prism → dihedralAngle prism = 45 * π / 180 := by sorry

end NUMINAMATH_CALUDE_dihedral_angle_is_45_degrees_l4017_401723


namespace NUMINAMATH_CALUDE_repeating_base_representation_l4017_401740

theorem repeating_base_representation (k : ℕ) : 
  k > 0 ∧ (12 : ℚ) / 65 = (3 * k + 1 : ℚ) / (k^2 - 1) → k = 17 :=
by sorry

end NUMINAMATH_CALUDE_repeating_base_representation_l4017_401740


namespace NUMINAMATH_CALUDE_impossible_segment_arrangement_l4017_401707

/-- A segment on the number line -/
structure Segment where
  start : ℕ
  length : ℕ
  h1 : start ≥ 1
  h2 : start + length ≤ 100

/-- The set of all possible segments -/
def AllSegments : Set Segment :=
  { s : Segment | s.start ≥ 1 ∧ s.start + s.length ≤ 100 ∧ s.length ∈ Finset.range 51 }

/-- The theorem stating the impossibility of the segment arrangement -/
theorem impossible_segment_arrangement :
  ¬ ∃ (segments : Finset Segment),
    segments.card = 50 ∧
    (∀ s ∈ segments, s ∈ AllSegments) ∧
    (∀ n ∈ Finset.range 51, ∃ s ∈ segments, s.length = n) :=
sorry

end NUMINAMATH_CALUDE_impossible_segment_arrangement_l4017_401707


namespace NUMINAMATH_CALUDE_paintable_area_four_bedrooms_l4017_401771

theorem paintable_area_four_bedrooms 
  (length : ℝ) (width : ℝ) (height : ℝ) (unpaintable_area : ℝ) (num_bedrooms : ℕ) :
  length = 15 →
  width = 11 →
  height = 9 →
  unpaintable_area = 80 →
  num_bedrooms = 4 →
  (2 * (length * height + width * height) - unpaintable_area) * num_bedrooms = 1552 := by
  sorry

end NUMINAMATH_CALUDE_paintable_area_four_bedrooms_l4017_401771


namespace NUMINAMATH_CALUDE_three_by_five_uncoverable_l4017_401760

/-- Represents a chessboard --/
structure Chessboard where
  rows : Nat
  cols : Nat

/-- Represents a domino --/
structure Domino where
  black : Unit
  white : Unit

/-- Defines a complete covering of a chessboard by dominoes --/
def CompleteCovering (board : Chessboard) (dominoes : List Domino) : Prop :=
  dominoes.length * 2 = board.rows * board.cols

/-- Theorem: A 3x5 chessboard cannot be completely covered by dominoes --/
theorem three_by_five_uncoverable :
  ¬ ∃ (dominoes : List Domino), CompleteCovering { rows := 3, cols := 5 } dominoes := by
  sorry

end NUMINAMATH_CALUDE_three_by_five_uncoverable_l4017_401760


namespace NUMINAMATH_CALUDE_subtraction_result_l4017_401790

theorem subtraction_result : 6102 - 2016 = 4086 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l4017_401790


namespace NUMINAMATH_CALUDE_z_squared_abs_l4017_401764

-- Define the complex number z
def z : ℂ := sorry

-- State the theorem
theorem z_squared_abs : z * (1 + Complex.I) = 1 + 3 * Complex.I → Complex.abs (z^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_z_squared_abs_l4017_401764


namespace NUMINAMATH_CALUDE_sin_two_phi_l4017_401725

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) :
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_phi_l4017_401725


namespace NUMINAMATH_CALUDE_solve_equation_l4017_401731

theorem solve_equation (y : ℚ) : (5 * y + 2) / (6 * y - 3) = 3 / 4 ↔ y = -17 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4017_401731


namespace NUMINAMATH_CALUDE_rohan_entertainment_spending_l4017_401799

/-- Represents Rohan's monthly finances --/
structure RohanFinances where
  salary : ℝ
  food_percent : ℝ
  rent_percent : ℝ
  conveyance_percent : ℝ
  savings : ℝ

/-- The conditions of Rohan's finances --/
def rohan_finances : RohanFinances :=
  { salary := 12500
  , food_percent := 40
  , rent_percent := 20
  , conveyance_percent := 10
  , savings := 2500 }

/-- Theorem stating that Rohan spends 10% on entertainment --/
theorem rohan_entertainment_spending (rf : RohanFinances := rohan_finances) :
  let total_percent := rf.food_percent + rf.rent_percent + rf.conveyance_percent + (rf.savings / rf.salary * 100)
  let entertainment_percent := 100 - total_percent
  entertainment_percent = 10 := by sorry

end NUMINAMATH_CALUDE_rohan_entertainment_spending_l4017_401799


namespace NUMINAMATH_CALUDE_A_value_l4017_401712

noncomputable def A (x : ℝ) : ℝ :=
  (Real.sqrt 3 * x^(3/2) - 5 * x^(1/3) + 5 * x^(4/3) - Real.sqrt (3*x)) /
  (Real.sqrt (3*x + 10 * Real.sqrt 3 * x^(5/6) + 25 * x^(2/3)) *
   Real.sqrt (1 - 2/x + 1/x^2))

theorem A_value (x : ℝ) (hx : x > 0) :
  (0 < x ∧ x < 1 → A x = -x) ∧
  (x > 1 → A x = x) := by
  sorry

end NUMINAMATH_CALUDE_A_value_l4017_401712


namespace NUMINAMATH_CALUDE_construct_square_l4017_401703

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Four points lying on the sides of a quadrilateral -/
structure SidePoints where
  K : Point  -- on side AB
  P : Point  -- on side BC
  R : Point  -- on side CD
  Q : Point  -- on side AD

/-- Predicate to check if a point lies on a line segment -/
def liesBetween (P Q R : Point) : Prop := sorry

/-- Predicate to check if two line segments are perpendicular -/
def perpendicular (P Q R S : Point) : Prop := sorry

/-- Predicate to check if two line segments have equal length -/
def equalLength (P Q R S : Point) : Prop := sorry

/-- Main theorem: Given four points on the sides of a quadrilateral, 
    if certain conditions are met, then the quadrilateral is a square -/
theorem construct_square (ABCD : Quadrilateral) (sides : SidePoints) : 
  liesBetween ABCD.A sides.K ABCD.B ∧
  liesBetween ABCD.B sides.P ABCD.C ∧
  liesBetween ABCD.C sides.R ABCD.D ∧
  liesBetween ABCD.D sides.Q ABCD.A ∧
  perpendicular ABCD.A ABCD.B ABCD.B ABCD.C ∧
  perpendicular ABCD.B ABCD.C ABCD.C ABCD.D ∧
  perpendicular ABCD.C ABCD.D ABCD.D ABCD.A ∧
  perpendicular ABCD.D ABCD.A ABCD.A ABCD.B ∧
  equalLength ABCD.A ABCD.B ABCD.B ABCD.C ∧
  equalLength ABCD.B ABCD.C ABCD.C ABCD.D ∧
  equalLength ABCD.C ABCD.D ABCD.D ABCD.A →
  -- Conclusion: ABCD is a square
  -- (We don't provide a formal definition of a square here, 
  -- as it would typically be defined elsewhere in a real geometry library)
  True := by
  sorry

end NUMINAMATH_CALUDE_construct_square_l4017_401703


namespace NUMINAMATH_CALUDE_age_sum_theorem_l4017_401789

theorem age_sum_theorem (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 72 → a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_theorem_l4017_401789


namespace NUMINAMATH_CALUDE_probability_of_more_than_five_draws_l4017_401782

def total_pennies : ℕ := 9
def shiny_pennies : ℕ := 4
def dull_pennies : ℕ := 5

def probability_more_than_five_draws : ℚ := 20 / 63

theorem probability_of_more_than_five_draws :
  let total_combinations := Nat.choose total_pennies shiny_pennies
  let favorable_combinations := Nat.choose 5 3 * Nat.choose 4 1
  (favorable_combinations : ℚ) / total_combinations = probability_more_than_five_draws :=
sorry

end NUMINAMATH_CALUDE_probability_of_more_than_five_draws_l4017_401782


namespace NUMINAMATH_CALUDE_sequence_lower_bound_l4017_401741

/-- Given a sequence of positive integers satisfying certain conditions, 
    the last element is greater than or equal to 2n² - 1 -/
theorem sequence_lower_bound (n : ℕ) (a : ℕ → ℕ) : n > 1 →
  (∀ i, 1 ≤ i → i < n → a i < a (i + 1)) →
  (∀ i, 1 ≤ i → i < n → ∃ k : ℕ, (a i ^ 2 + a (i + 1) ^ 2) / 2 = k ^ 2) →
  a n ≥ 2 * n ^ 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_lower_bound_l4017_401741


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l4017_401714

theorem binomial_expansion_coefficient (a : ℝ) : 
  (20 : ℝ) * a^3 = 160 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l4017_401714


namespace NUMINAMATH_CALUDE_min_area_of_two_squares_l4017_401781

/-- Given a wire of length 20 cm cut into two parts, with each part forming a square 
    where the part's length is the square's perimeter, the minimum combined area 
    of the two squares is 12.5 square centimeters. -/
theorem min_area_of_two_squares (x : ℝ) : 
  0 ≤ x → 
  x ≤ 20 → 
  (x^2 / 16 + (20 - x)^2 / 16) ≥ 12.5 := by
  sorry

end NUMINAMATH_CALUDE_min_area_of_two_squares_l4017_401781


namespace NUMINAMATH_CALUDE_octagon_square_ratio_l4017_401728

theorem octagon_square_ratio (s r : ℝ) (h : s > 0) (k : r > 0) :
  s^2 = 2 * r^2 * Real.sqrt 2 → r / s = 1 / Real.sqrt (2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_octagon_square_ratio_l4017_401728


namespace NUMINAMATH_CALUDE_special_function_at_two_l4017_401793

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 0 ∧ ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x + f y)

/-- Theorem stating that f(2) = 0 for any function satisfying the given conditions -/
theorem special_function_at_two (f : ℝ → ℝ) (h : special_function f) : f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_two_l4017_401793


namespace NUMINAMATH_CALUDE_right_triangle_area_l4017_401765

theorem right_triangle_area (h : ℝ) (angle : ℝ) :
  h = 10 * Real.sqrt 3 →
  angle = 30 * π / 180 →
  let s := h / 2
  let l := Real.sqrt 3 / 2 * h
  0.5 * s * l = 37.5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l4017_401765


namespace NUMINAMATH_CALUDE_pawns_left_l4017_401770

/-- The number of pawns each player starts with in a standard chess game -/
def standard_pawns : ℕ := 8

/-- The number of pawns Sophia has lost -/
def sophia_lost : ℕ := 5

/-- The number of pawns Chloe has lost -/
def chloe_lost : ℕ := 1

/-- Theorem: The total number of pawns left in the game is 10 -/
theorem pawns_left : 
  (standard_pawns - sophia_lost) + (standard_pawns - chloe_lost) = 10 := by
  sorry

end NUMINAMATH_CALUDE_pawns_left_l4017_401770


namespace NUMINAMATH_CALUDE_gcd_diff_is_square_l4017_401786

theorem gcd_diff_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, Nat.gcd x (Nat.gcd y z) * (y - x) = k^2 := by sorry

end NUMINAMATH_CALUDE_gcd_diff_is_square_l4017_401786


namespace NUMINAMATH_CALUDE_no_cubic_four_primes_pm3_l4017_401743

theorem no_cubic_four_primes_pm3 : 
  ¬∃ (f : ℤ → ℤ) (p q r s : ℕ), 
    (∀ x : ℤ, ∃ a b c d : ℤ, f x = a*x^3 + b*x^2 + c*x + d) ∧ 
    Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧ 
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
    ((f p = 3 ∨ f p = -3) ∧ 
     (f q = 3 ∨ f q = -3) ∧ 
     (f r = 3 ∨ f r = -3) ∧ 
     (f s = 3 ∨ f s = -3)) :=
by sorry

end NUMINAMATH_CALUDE_no_cubic_four_primes_pm3_l4017_401743


namespace NUMINAMATH_CALUDE_equation_solution_l4017_401720

theorem equation_solution : 
  ∀ x : ℝ, x^2 - 2*|x - 1| - 2 = 0 ↔ x = 2 ∨ x = -1 - Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4017_401720


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l4017_401777

-- Define the number of knights
def total_knights : ℕ := 30

-- Define the number of knights chosen
def chosen_knights : ℕ := 4

-- Function to calculate the probability
def probability_adjacent_knights : ℚ :=
  1 - (Nat.choose (total_knights - chosen_knights + 1) (chosen_knights - 1) : ℚ) / 
      (Nat.choose total_knights chosen_knights : ℚ)

-- Theorem statement
theorem adjacent_knights_probability :
  probability_adjacent_knights = 4961 / 5481 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l4017_401777


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_exists_l4017_401795

theorem rectangular_parallelepiped_exists : ∃ (a b c : ℕ+), 2 * (a * b + b * c + c * a) = 4 * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_exists_l4017_401795


namespace NUMINAMATH_CALUDE_jacket_price_theorem_l4017_401798

theorem jacket_price_theorem (SRP : ℝ) (marked_discount : ℝ) (additional_discount : ℝ) :
  SRP = 120 →
  marked_discount = 0.4 →
  additional_discount = 0.2 →
  let marked_price := SRP * (1 - marked_discount)
  let final_price := marked_price * (1 - additional_discount)
  (final_price / SRP) * 100 = 48 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_theorem_l4017_401798


namespace NUMINAMATH_CALUDE_subset_M_l4017_401708

def P : Set ℝ := {x | 0 ≤ x ∧ x < 1}
def Q : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}
def M : Set ℝ := P ∪ Q

theorem subset_M : {0, 2, 3} ⊆ M := by sorry

end NUMINAMATH_CALUDE_subset_M_l4017_401708


namespace NUMINAMATH_CALUDE_functional_equation_solution_l4017_401704

/-- The functional equation satisfied by f --/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) - f (x - y) = 2 * y * (3 * x^2 + y^2)

/-- The theorem statement --/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f →
  ∃ a : ℝ, ∀ x : ℝ, f x = x^3 + a :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l4017_401704


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l4017_401709

/-- The surface area of a cylinder with base radius 1 and volume 2π is 6π. -/
theorem cylinder_surface_area (r h : ℝ) : 
  r = 1 → π * r^2 * h = 2*π → 2*π*r*h + 2*π*r^2 = 6*π :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l4017_401709


namespace NUMINAMATH_CALUDE_younger_person_age_l4017_401791

/-- Proves that the younger person's age is 8 years, given the conditions of the problem. -/
theorem younger_person_age (y e : ℕ) : 
  e = y + 12 →  -- The elder person's age is 12 years more than the younger person's
  e - 5 = 5 * (y - 5) →  -- Five years ago, the elder was 5 times as old as the younger
  y = 8 :=  -- The younger person's present age is 8 years
by sorry

end NUMINAMATH_CALUDE_younger_person_age_l4017_401791


namespace NUMINAMATH_CALUDE_no_integer_points_between_A_and_B_l4017_401739

/-- A point with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- The line passing through points A(2,3) and B(50,305) -/
def line_AB (p : IntPoint) : Prop :=
  (p.y - 3) * (50 - 2) = (p.x - 2) * (305 - 3)

/-- A point is strictly between A and B -/
def between_A_and_B (p : IntPoint) : Prop :=
  2 < p.x ∧ p.x < 50

theorem no_integer_points_between_A_and_B :
  ¬ ∃ p : IntPoint, line_AB p ∧ between_A_and_B p :=
sorry

end NUMINAMATH_CALUDE_no_integer_points_between_A_and_B_l4017_401739


namespace NUMINAMATH_CALUDE_permutation_equation_solution_combination_equation_solution_l4017_401702

-- Define the factorial function
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the permutation function
def permutation (n k : ℕ) : ℕ := 
  if k ≤ n then factorial n / factorial (n - k) else 0

-- Define the combination function
def combination (n k : ℕ) : ℕ := 
  if k ≤ n then factorial n / (factorial k * factorial (n - k)) else 0

theorem permutation_equation_solution : 
  ∃! x : ℕ, permutation (2 * x) 4 = 60 * permutation x 3 ∧ x > 0 := by sorry

theorem combination_equation_solution : 
  ∃! n : ℕ, combination (n + 3) (n + 1) = 
    combination (n + 1) (n - 1) + combination (n + 1) n + combination n (n - 2) := by sorry

end NUMINAMATH_CALUDE_permutation_equation_solution_combination_equation_solution_l4017_401702


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_length_l4017_401736

/-- An isosceles right triangle with given properties -/
structure IsoscelesRightTriangle where
  -- The length of the equal sides
  leg : ℝ
  -- The area of the triangle
  area : ℝ
  -- Condition that the area is equal to half the square of the leg
  area_eq : area = (1/2) * leg^2

/-- The main theorem -/
theorem isosceles_right_triangle_hypotenuse_length 
  (t : IsoscelesRightTriangle) (h : t.area = 25) : 
  t.leg * Real.sqrt 2 = 10 := by
  sorry

#check isosceles_right_triangle_hypotenuse_length

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_length_l4017_401736


namespace NUMINAMATH_CALUDE_interest_rate_difference_l4017_401794

/-- The difference between two simple interest rates given specific conditions -/
theorem interest_rate_difference (principal : ℝ) (time : ℝ) (interest_diff : ℝ) :
  principal = 2600 →
  time = 3 →
  interest_diff = 78 →
  ∃ (rate1 rate2 : ℝ), rate2 - rate1 = 0.01 ∧
    principal * time * (rate2 - rate1) / 100 = interest_diff :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l4017_401794


namespace NUMINAMATH_CALUDE_problem_statement_l4017_401717

theorem problem_statement (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 - x^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4017_401717


namespace NUMINAMATH_CALUDE_trisha_remaining_money_l4017_401780

/-- Calculates the remaining money after shopping given the initial amount and expenses. -/
def remaining_money (initial : ℕ) (meat chicken veggies eggs dog_food : ℕ) : ℕ :=
  initial - (meat + chicken + veggies + eggs + dog_food)

/-- Proves that Trisha's remaining money after shopping is $35. -/
theorem trisha_remaining_money :
  remaining_money 167 17 22 43 5 45 = 35 := by
  sorry

end NUMINAMATH_CALUDE_trisha_remaining_money_l4017_401780


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l4017_401758

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l4017_401758


namespace NUMINAMATH_CALUDE_tan_30_degrees_l4017_401742

theorem tan_30_degrees : Real.tan (π / 6) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_degrees_l4017_401742


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l4017_401763

theorem angle_sum_in_circle (x : ℝ) : 6 * x + 3 * x + 4 * x + x + 2 * x = 360 → x = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_circle_l4017_401763


namespace NUMINAMATH_CALUDE_expression_simplification_l4017_401721

theorem expression_simplification (x : ℝ) 
  (h1 : x * (x^2 - 4) = 0) 
  (h2 : x ≠ 0) 
  (h3 : x ≠ 2) :
  (x - 3) / (3 * x^2 - 6 * x) / (x + 2 - 5 / (x - 2)) = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4017_401721


namespace NUMINAMATH_CALUDE_profit_maximum_l4017_401756

/-- Represents the daily sales profit function -/
def profit (x : ℕ) : ℝ := -10 * (x : ℝ)^2 + 90 * (x : ℝ) + 1900

/-- The maximum daily profit -/
def max_profit : ℝ := 2100

theorem profit_maximum :
  ∃ x : ℕ, profit x = max_profit ∧
  ∀ y : ℕ, profit y ≤ max_profit :=
sorry

end NUMINAMATH_CALUDE_profit_maximum_l4017_401756


namespace NUMINAMATH_CALUDE_bulb_arrangement_count_l4017_401700

/-- The number of ways to arrange blue and red bulbs -/
def arrange_blue_red : ℕ := Nat.choose 16 8

/-- The number of ways to place white bulbs between blue and red bulbs -/
def place_white : ℕ := Nat.choose 17 11

/-- The total number of blue bulbs -/
def blue_bulbs : ℕ := 8

/-- The total number of red bulbs -/
def red_bulbs : ℕ := 8

/-- The total number of white bulbs -/
def white_bulbs : ℕ := 11

/-- The theorem stating the number of ways to arrange the bulbs -/
theorem bulb_arrangement_count :
  arrange_blue_red * place_white = 159279120 :=
sorry

end NUMINAMATH_CALUDE_bulb_arrangement_count_l4017_401700


namespace NUMINAMATH_CALUDE_insufficient_evidence_l4017_401730

/-- Represents the data from a 2x2 contingency table --/
structure ContingencyTable :=
  (irregular_disease : Nat)
  (irregular_no_disease : Nat)
  (regular_disease : Nat)
  (regular_no_disease : Nat)

/-- Represents the result of a statistical test --/
inductive TestResult
  | Significant
  | NotSignificant

/-- Performs a statistical test on the contingency table data --/
def statisticalTest (data : ContingencyTable) : TestResult :=
  sorry

/-- Theorem stating that the given survey data does not provide sufficient evidence
    for a relationship between stomach diseases and living habits --/
theorem insufficient_evidence (survey_data : ContingencyTable) 
  (h1 : survey_data.irregular_disease = 5)
  (h2 : survey_data.irregular_no_disease = 15)
  (h3 : survey_data.regular_disease = 40)
  (h4 : survey_data.regular_no_disease = 10) :
  statisticalTest survey_data = TestResult.NotSignificant :=
sorry

end NUMINAMATH_CALUDE_insufficient_evidence_l4017_401730


namespace NUMINAMATH_CALUDE_function_characterization_l4017_401737

/-- Given a positive real number α, prove that any function f from positive integers to reals
    satisfying f(k + m) = f(k) + f(m) for any positive integers k and m where αm ≤ k < (α + 1)m,
    must be of the form f(n) = bn for some real number b and all positive integers n. -/
theorem function_characterization (α : ℝ) (hα : α > 0) :
  ∀ f : ℕ+ → ℝ,
  (∀ (k m : ℕ+), α * m.val ≤ k.val ∧ k.val < (α + 1) * m.val → f (k + m) = f k + f m) →
  ∃ b : ℝ, ∀ n : ℕ+, f n = b * n.val :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l4017_401737


namespace NUMINAMATH_CALUDE_fourteenth_root_of_unity_l4017_401753

theorem fourteenth_root_of_unity : 
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 13 ∧ 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * π * n / 14)) := by
  sorry

end NUMINAMATH_CALUDE_fourteenth_root_of_unity_l4017_401753


namespace NUMINAMATH_CALUDE_inequality_system_solution_l4017_401747

theorem inequality_system_solution : 
  {x : ℕ | 3 * (x - 1) < 5 * x + 1 ∧ (x - 1) / 2 ≥ 2 * x - 4} = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l4017_401747


namespace NUMINAMATH_CALUDE_largest_amount_l4017_401768

theorem largest_amount (milk : Rat) (cider : Rat) (orange_juice : Rat)
  (h_milk : milk = 3/8)
  (h_cider : cider = 7/10)
  (h_orange_juice : orange_juice = 11/15) :
  max milk (max cider orange_juice) = orange_juice :=
by sorry

end NUMINAMATH_CALUDE_largest_amount_l4017_401768


namespace NUMINAMATH_CALUDE_reinforcement_arrival_theorem_l4017_401733

/-- Represents the number of days after which the reinforcement arrived -/
def reinforcement_arrival_day : ℕ := 20

/-- The size of the initial garrison -/
def initial_garrison : ℕ := 2000

/-- The number of days the initial provisions would last -/
def initial_provision_days : ℕ := 40

/-- The size of the reinforcement -/
def reinforcement_size : ℕ := 2000

/-- The number of days the provisions last after reinforcement arrival -/
def remaining_days : ℕ := 10

theorem reinforcement_arrival_theorem :
  initial_garrison * initial_provision_days =
  initial_garrison * reinforcement_arrival_day +
  (initial_garrison + reinforcement_size) * remaining_days :=
by sorry

end NUMINAMATH_CALUDE_reinforcement_arrival_theorem_l4017_401733


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_set_l4017_401713

theorem quadratic_equation_solution_set :
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x + 2
  {x : ℝ | f x = 0} = {1, 2} := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_set_l4017_401713


namespace NUMINAMATH_CALUDE_no_real_roots_range_l4017_401762

theorem no_real_roots_range (p q : ℝ) : 
  (∀ x : ℝ, x^2 + 2*p*x - (q^2 - 2) ≠ 0) → p + q ∈ Set.Ioo (-2 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_range_l4017_401762


namespace NUMINAMATH_CALUDE_average_salary_is_8800_l4017_401772

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 14000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def num_people : ℕ := 5

theorem average_salary_is_8800 : 
  (total_salary : ℚ) / num_people = 8800 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_is_8800_l4017_401772


namespace NUMINAMATH_CALUDE_car_robot_ratio_l4017_401718

theorem car_robot_ratio : 
  ∀ (tom_michael_robots : ℕ) (bob_robots : ℕ),
    tom_michael_robots = 9 →
    bob_robots = 81 →
    (bob_robots : ℚ) / tom_michael_robots = 9 := by
  sorry

end NUMINAMATH_CALUDE_car_robot_ratio_l4017_401718


namespace NUMINAMATH_CALUDE_restaurant_cooks_count_l4017_401759

/-- Proves that the number of cooks is 9 given the initial and final ratios of cooks to waiters -/
theorem restaurant_cooks_count : ∀ (C W : ℕ),
  C / W = 3 / 11 →
  C / (W + 12) = 1 / 5 →
  C = 9 := by
sorry

end NUMINAMATH_CALUDE_restaurant_cooks_count_l4017_401759


namespace NUMINAMATH_CALUDE_subtraction_properties_l4017_401719

theorem subtraction_properties (a b : ℝ) : 
  ((a - b)^2 = (b - a)^2) ∧ 
  (|a - b| = |b - a|) ∧ 
  (a - b = -b + a) ∧
  ((a - b = b - a) ↔ (a = b)) :=
by sorry

end NUMINAMATH_CALUDE_subtraction_properties_l4017_401719


namespace NUMINAMATH_CALUDE_exponent_multiplication_l4017_401701

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l4017_401701


namespace NUMINAMATH_CALUDE_system_a_solutions_system_b_solutions_l4017_401775

-- Part (a)
theorem system_a_solutions (x y z : ℝ) : 
  (2 * x = (y + z)^2 ∧ 2 * y = (z + x)^2 ∧ 2 * z = (x + y)^2) → 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2)) := by
  sorry

-- Part (b)
theorem system_b_solutions (x y z : ℝ) :
  (x^2 - x*y - x*z + z^2 = 0 ∧ 
   x^2 - x*z - y*z + 3*y^2 = 2 ∧ 
   y^2 + x*y + y*z - z^2 = 2) → 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1 ∧ y = -1 ∧ z = -1)) := by
  sorry

end NUMINAMATH_CALUDE_system_a_solutions_system_b_solutions_l4017_401775


namespace NUMINAMATH_CALUDE_infinitely_many_decimals_between_3_3_and_3_6_l4017_401751

/-- The set of decimals between 3.3 and 3.6 is infinite -/
theorem infinitely_many_decimals_between_3_3_and_3_6 :
  (∀ n : ℕ, ∃ x : ℝ, 3.3 < x ∧ x < 3.6 ∧ ∃ k : ℕ, x = ↑k / 10^n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_decimals_between_3_3_and_3_6_l4017_401751


namespace NUMINAMATH_CALUDE_zigzag_outward_angle_regular_polygon_l4017_401761

/-- The number of degrees at each outward point of a zigzag extension of a regular polygon -/
def outward_angle (n : ℕ) : ℚ :=
  720 / n

theorem zigzag_outward_angle_regular_polygon (n : ℕ) (h : n > 4) :
  outward_angle n = 720 / n :=
by sorry

end NUMINAMATH_CALUDE_zigzag_outward_angle_regular_polygon_l4017_401761


namespace NUMINAMATH_CALUDE_inequality_multiplication_l4017_401778

theorem inequality_multiplication (a b c d : ℝ) :
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a > b ∧ c > d → a * c > b * d) ∧
  (a < 0 ∧ b < 0 ∧ c < 0 ∧ d < 0 ∧ a < b ∧ c < d → a * c > b * d) :=
sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l4017_401778


namespace NUMINAMATH_CALUDE_factorization_equality_l4017_401754

theorem factorization_equality (m n : ℝ) : m^2 * n - 9 * n = n * (m + 3) * (m - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l4017_401754


namespace NUMINAMATH_CALUDE_tickets_per_friend_is_four_l4017_401767

/-- The number of tickets each friend bought on the first day -/
def tickets_per_friend : ℕ := sorry

/-- The total number of tickets to be sold -/
def total_tickets : ℕ := 80

/-- The number of friends who bought tickets on the first day -/
def num_friends : ℕ := 5

/-- The number of tickets sold on the second day -/
def second_day_tickets : ℕ := 32

/-- The number of tickets that need to be sold on the third day -/
def third_day_tickets : ℕ := 28

/-- Theorem stating that the number of tickets each friend bought on the first day is 4 -/
theorem tickets_per_friend_is_four :
  tickets_per_friend = 4 ∧
  tickets_per_friend * num_friends + second_day_tickets + third_day_tickets = total_tickets :=
by sorry

end NUMINAMATH_CALUDE_tickets_per_friend_is_four_l4017_401767


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l4017_401734

def is_prime (n : ℕ) : Prop := sorry

def is_divisible_by (a b : ℕ) : Prop := sorry

theorem least_number_divisible_by_five_primes :
  ∃ (n : ℕ), n > 0 ∧
  (∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧ is_prime p₅ ∧
    is_divisible_by n p₁ ∧ is_divisible_by n p₂ ∧ is_divisible_by n p₃ ∧ 
    is_divisible_by n p₄ ∧ is_divisible_by n p₅) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ (q₁ q₂ q₃ q₄ q₅ : ℕ), 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧
      q₄ ≠ q₅ ∧
      is_prime q₁ ∧ is_prime q₂ ∧ is_prime q₃ ∧ is_prime q₄ ∧ is_prime q₅ ∧
      is_divisible_by m q₁ ∧ is_divisible_by m q₂ ∧ is_divisible_by m q₃ ∧ 
      is_divisible_by m q₄ ∧ is_divisible_by m q₅) → 
    m ≥ n) ∧
  n = 2310 :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l4017_401734
